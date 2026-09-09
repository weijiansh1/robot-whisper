"""Verify frozen choices, independent alarm counts, and raw online replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from recovery import AlertState, HERE, METHODS, RecoveryGuardMonitor, fit_profile, score_streams
from monitor import ALPHAS, ROOT
from run_analysis import archive, digest, write_json
from run_experiment import ALL_METHODS, BASE, KINDS, PRIMARY_ALPHA, PRIMARY_KIND, RUN_B, VALIDATION, split_roles


def quantile(values, alpha):
    ordered = sorted(float(x) for x in values)
    rank = int(np.ceil((len(ordered) + 1) * (1 - alpha)))
    return ordered[rank - 1] if rank <= len(ordered) else np.inf


def independent_first(score, valid, threshold):
    crossing = np.isfinite(score) & valid & (score > threshold)
    return np.asarray([np.flatnonzero(row)[0] if row.any() else -1 for row in crossing], np.int16)


def check_design(frame, cache, raw, output):
    checks = 0
    selection = pd.read_csv(output / "selection_grid.csv", dtype={"fold": str})
    for fold in [*map(str, range(5)), "deploy"]:
        profile = json.loads((output / ("profile_%s.json" % fold)).read_text())
        reference = np.asarray(profile["reference_rows"], int)
        calibration = np.asarray(profile["calibration_rows"], int)
        development = np.asarray(profile["development_rows"], int)
        assert not frame.iloc[np.r_[reference, calibration, development]].run_id.eq(RUN_B).any()
        ids = [set(zip(frame.iloc[ix].task, frame.iloc[ix].init_state_id)) for ix in (reference, calibration, development)]
        assert not any(ids[i] & ids[j] for i in range(3) for j in range(i))
        if fold != "deploy":
            for expected, observed in zip(split_roles(frame, int(fold)), (reference, calibration, development, profile["test_rows"])):
                np.testing.assert_array_equal(expected, observed)
        ref_cache = {key: value[reference] for key, value in cache.items()}
        rebuilt = fit_profile(ref_cache, raw[reference], np.arange(len(reference)))
        assert profile["heads"] == rebuilt["heads"] and profile["periodicity_scale"] == rebuilt["periodicity_scale"]
        design = archive(output / ("design_%s.npz" % fold))
        np.testing.assert_array_equal(design["calibration_rows"], calibration)
        np.testing.assert_array_equal(design["development_rows"], development)
        success = ~frame.iloc[calibration].failure.to_numpy(bool)
        group_frame = frame.iloc[calibration].loc[success]
        keys = (group_frame.task + "|" + group_frame.init_state_id.astype(str)).to_numpy()
        for mi, method in enumerate((*METHODS, "clock")):
            score = design["calibration_scores"][mi, success]
            peaks = np.where(np.isfinite(score), score, -np.inf).max(axis=1)
            grouped = pd.Series(peaks).groupby(keys).max().to_numpy()
            for ki, units in enumerate((peaks, grouped)):
                assert quantile(units, .1) == design["watch_thresholds"][ki, mi]
                for ai, alpha in enumerate(ALPHAS):
                    assert quantile(units, alpha) == design["thresholds"][ki, ai, mi]
                    checks += 1
        failure = frame.iloc[development].failure.to_numpy(bool)
        for ki, kind in enumerate(KINDS):
            for ai, alpha in enumerate(ALPHAS):
                candidates = []
                for mi, method in enumerate(METHODS):
                    values = design["development_first"][ki, ai, mi]
                    fired = values >= 0
                    tp, fp = int((fired & failure).sum()), int((fired & ~failure).sum())
                    utility = sum(np.exp(-max(int(q) - 7, 0) / 12.) for q in values[failure] if q >= 0) / failure.sum()
                    candidates.append((tp, fp, utility))
                    record = selection.loc[selection.fold.eq(fold) & selection.calibration.eq(kind)
                                           & selection.alpha.eq(alpha) & selection.method.eq(method)].iloc[0]
                    assert (record.tp, record.fp) == (tp, fp)
                    np.testing.assert_allclose(record.utility, utility, rtol=1e-12)
                eligible = [i for i, (tp, fp, _) in enumerate(candidates) if tp >= candidates[0][0] and fp <= candidates[0][1]]
                selected = max(eligible, key=lambda i: (candidates[i][2], candidates[i][0], -candidates[i][1], -i))
                assert selected == design["selected"][ki, ai]
        selected = design["selected"][PRIMARY_KIND, PRIMARY_ALPHA]
        assert profile["method"] == METHODS[selected]
        np.testing.assert_allclose(profile["threshold"], design["thresholds"][PRIMARY_KIND, PRIMARY_ALPHA, selected])
    return checks


def raw_replay(frame, cache, raw, prediction, output):
    samples = pd.read_csv(VALIDATION / "raw_replay_checks.csv").global_row.astype(int).tolist()
    samples.extend(pd.read_csv(output / "recovery_candidates.csv").global_row.astype(int).tolist())
    episodes = pd.read_csv(output / "state_episodes.csv")
    for _, part in episodes.loc[episodes.first_alarm >= 0].groupby(["suite", "failure"]):
        samples.append(int(part.global_row.iloc[0]))
    results = []
    for row_id in sorted(set(samples)):
        row = frame.iloc[row_id]
        fold = str(int(row.init_state_id) % 5) if row.run_id == RUN_B else "deploy"
        profile_path = output / ("profile_%s.json" % fold)
        profile = json.loads(profile_path.read_text())
        monitor = RecoveryGuardMonitor(profile_path, row.checkpoint)
        source = ROOT / "VLA_MUI_HUB" / row.source
        summaries = sorted(json.loads((source / "client/summaries.json").read_text()), key=lambda e: e["episode_index"])
        offset = 0
        for entry in summaries:
            if int(entry["episode_index"]) == int(row.episode):
                break
            offset += int(entry["inference_calls"])
        store = zarr.open_group(str(source / "server/routes.zarr"), mode="r")
        np.testing.assert_array_equal(np.asarray(store["episode_id"][offset:offset + row.length]), int(row.episode))
        inputs = np.asarray(store["hb_router_probs"][offset:offset + row.length])
        expected = score_streams({key: value[row_id:row_id + 1, :row.length] for key, value in cache.items()},
                                raw[row_id:row_id + 1, :row.length], profile)[METHODS.index(profile["method"]), 0]
        if row.run_id == RUN_B:
            position = int(np.flatnonzero(prediction["global_rows"] == row_id)[0])
            np.testing.assert_allclose(expected, prediction["scores"][-1, position, :row.length], equal_nan=True)
        observed = [monitor.update(query) for query in inputs]
        values = np.asarray([r["score"] for r in observed])
        np.testing.assert_allclose(values, expected, rtol=5e-4, atol=.002, equal_nan=True,
                                   err_msg="raw score mismatch row %d" % row_id)
        expected_first = independent_first(expected[None], np.ones((1, len(expected)), bool), profile["threshold"])[0]
        assert observed[-1]["first_alarm_query"] == expected_first, row_id
        state = AlertState(profile["threshold"], profile["watch_threshold"])
        cached_state = [state.update(score)["state"] for score in expected]
        assert [r["state"] for r in observed] == cached_state, row_id
        good = np.isfinite(values) & np.isfinite(expected)
        results.append(dict(global_row=row_id, source=row.source, episode=int(row.episode), queries=int(row.length),
                            method=profile["method"], first=int(expected_first),
                            max_score_error=float(np.max(np.abs(values[good] - expected[good]))) if good.any() else 0.))
    pd.DataFrame(results).to_csv(output / "raw_online_checks.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_recovery_20260908")
    args = parser.parse_args()
    output = args.output
    manifest = json.loads((output / "verification.json").read_text())
    for name, expected in manifest["artifact_hashes"].items():
        assert digest(output / name) == expected, name
    for name, expected in manifest["sources"].items():
        assert digest(HERE / name) == expected, name
    sealed = json.loads((output / "selection_frozen.json").read_text())
    for name, expected in sealed["design_files"].items():
        assert digest(output / name) == expected, name
    provenance = json.loads((output / "input_verification.json").read_text())
    for name, expected in provenance["inputs"].items():
        assert digest(ROOT / name) == expected, name
    frame = pd.read_csv(output / "index.csv")
    cache = archive(BASE / "round3_safe/v7/v7_inputs.npz")
    raw = archive(BASE / "round7_temporal_fusion/v8_inputs.npz")["raw"]
    design_checks = check_design(frame, cache, raw, output)
    prediction = archive(output / "predictions.npz")
    rows = prediction["global_rows"]
    np.testing.assert_array_equal(rows, np.flatnonzero(frame.run_id.eq(RUN_B)))
    np.testing.assert_array_equal(prediction["valid"], cache["valid"][rows])
    assert not np.isfinite(prediction["scores"][:, ~prediction["valid"]]).any()
    checks = 0
    for fold in range(5):
        selected_rows = prediction["fold_ids"] == fold
        assert selected_rows.sum() == 3200
        np.testing.assert_array_equal(frame.iloc[rows[selected_rows]].init_state_id.to_numpy() % 5, fold)
        for ki in range(2):
            for ai in range(len(ALPHAS)):
                for mi, method in enumerate(ALL_METHODS):
                    chosen = int(prediction["choices"][fold, ki, ai]) if method == "selected" else mi
                    if method == "clock":
                        score = np.broadcast_to(np.arange(52)[None], (selected_rows.sum(), 52))
                        threshold = prediction["thresholds"][fold, ki, ai, -1]
                    else:
                        score = prediction["scores"][chosen, selected_rows]
                        threshold = prediction["thresholds"][fold, ki, ai, chosen]
                    expected = independent_first(score, prediction["valid"][selected_rows], threshold)
                    np.testing.assert_array_equal(expected, prediction["first"][ki, ai, mi, selected_rows])
                    checks += len(expected)
    curves = pd.read_csv(output / "cumulative_curves.csv")
    for _, part in curves.groupby(["method", "calibration", "alpha", "scope"]):
        assert (np.diff(part.sort_values("query").fp) >= 0).all()
        assert part.failures.nunique() == part.successes.nunique() == 1
    states = pd.read_csv(output / "state_episodes.csv")
    assert len(states) == 16000 and not states.global_row.duplicated().any()
    np.testing.assert_array_equal(states.first_alarm, prediction["first"][PRIMARY_KIND, PRIMARY_ALPHA, len(METHODS)])
    raw_checks = raw_replay(frame, cache, raw, prediction, output)
    write_json(output / "independent_verification.json", dict(all_checks_passed=True,
                calibration_order_statistics=design_checks, independent_alarm_checks=checks,
                raw_online_episodes=len(raw_checks), raw_online_queries=sum(r["queries"] for r in raw_checks),
                raw_sources=len(set(r["source"] for r in raw_checks)),
                maximum_raw_score_error=max(r["max_score_error"] for r in raw_checks),
                sources={"verify_recovery.py": digest(Path(__file__))}))
    print("VERIFIED %d alarm decisions and %d online raw episodes" % (checks, len(raw_checks)), flush=True)


if __name__ == "__main__":
    main()
