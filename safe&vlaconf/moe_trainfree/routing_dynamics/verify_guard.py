"""Independent cache, rank, raw-routing, and retrospective metric verification."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import mannwhitneyu
import zarr

from encoder import FEATURE_NAMES
from guard import (ALPHAS, BRANCHES, KINDS, METHOD_BRANCHES, PeakBank, RoutingGuardMonitor,
                   dynamics_scores, first_trigger, v82_scores)
from run_guard_experiment import ACTION_CAPS, BASE, FEATURE_DIR, HERE, LEGACY_DIR, METHODS, ROOT, S05
from run_analysis import digest, write_json


def replay_cached_baseline(frame, valid, profiles, old_scores):
    manifest = json.loads((LEGACY_DIR / "input_verification.json").read_text())
    paths = (BASE / "round3_safe/v7/v7_inputs.npz", BASE / "round7_temporal_fusion/v8_inputs.npz")
    for path in paths:
        assert digest(path) == manifest["inputs"][str(path.relative_to(ROOT))]
    with np.load(paths[0], allow_pickle=False) as z:
        cache = {name: z[name] for name in ("mobility", "acceleration", "periodicity", "valid")}
    with np.load(paths[1], allow_pickle=False) as z:
        raw = z["raw"]
    np.testing.assert_array_equal(cache["valid"], valid)
    comparisons = 0
    for profile in profiles:
        cal, test = np.asarray(profile["calibration_rows"]), np.asarray(profile["test_rows"])
        rows = np.r_[cal, test]
        computed = v82_scores({k: value[rows] for k, value in cache.items()}, raw[rows], profile["v82_profile"])
        with np.load(LEGACY_DIR / f"fold_{profile['fold']}_calibration.npz", allow_pickle=False) as z:
            expected = z["scores"][list(z["methods"]).index("v82")]
        np.testing.assert_array_equal(computed[:len(cal)], expected)
        np.testing.assert_array_equal(computed[len(cal):], old_scores[test])
        comparisons += int(valid[rows].sum())
    return comparisons


def raw_rows(b, profiles, branches):
    selected, boundary_rows = set(), set()
    for _, part in b.groupby(["task", "failure"]):
        selected.add(int(part.global_row.iloc[0]))
    selected.update(b.loc[b.task.eq(S05) & b.episode.isin([77, 96, 115, 174, 190, 102, 286, 363, 366, 367]), "global_row"])
    for profile in profiles:
        rows = np.asarray(profile["test_rows"])
        for kind in KINDS:
            for name in BRANCHES:
                threshold, _ = PeakBank.from_list(profile["banks"][kind][name]).threshold(.01 / 3)
                values = branches[name][rows]
                delta = values - threshold
                for side in (delta <= 0, delta > 0):
                    distance = np.where(np.isfinite(delta) & side, np.abs(delta), np.inf)
                    local, _ = np.unravel_index(distance.argmin(), distance.shape)
                    if np.isfinite(distance.min()):
                        boundary_rows.add(int(rows[local]))
    selected.update(boundary_rows)
    return sorted(selected), sorted(boundary_rows)


def replay_raw(frame, b, profiles, branches, valid, first, output):
    rows, boundary_rows = raw_rows(b, profiles, branches)
    inverse = {int(r): i for i, r in enumerate(b.global_row)}
    groups, summaries = {}, {}
    maximum_error = dict.fromkeys(BRANCHES, 0.)
    records, query_checks, primary_differences, all_setting_differences = [], 0, 0, 0
    tail_rank_changes = 0
    for number, row_id in enumerate(rows):
        row = frame.iloc[row_id]
        fold = int(row.init_state_id % 5)
        profile = profiles[fold]
        if row.source not in groups:
            run = ROOT / "VLA_MUI_HUB" / row.source
            groups[row.source] = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
            entries = sorted(json.loads((run / "client/summaries.json").read_text()), key=lambda x: x["episode_index"])
            summaries[row.source] = np.r_[0, np.cumsum([entry["inference_calls"] for entry in entries])]
        start = int(summaries[row.source][row.episode])
        raw = groups[row.source]["hb_router_probs"][start:start + row.length]
        live = RoutingGuardMonitor(output / f"fold_{fold}_profile.json", row.checkpoint)
        results = [live.update(probability) for probability in raw]
        for name in BRANCHES:
            current = np.asarray([r["branch_scores"][name] for r in results])
            expected = branches[name][row_id, :row.length]
            np.testing.assert_array_equal(np.isfinite(current), np.isfinite(expected))
            finite = np.isfinite(current)
            if finite.any():
                maximum_error[name] = max(maximum_error[name], float(np.max(np.abs(current[finite] - expected[finite]))))
                np.testing.assert_allclose(current[finite], expected[finite], atol=2e-3, rtol=2e-3)
        online = {name: np.asarray([r["branch_scores"][name] for r in results]) for name in BRANCHES}
        ix = inverse[row_id]
        for ki, kind in enumerate(KINDS):
            bank = {name: PeakBank.from_list(profile["banks"][kind][name]) for name in BRANCHES}
            # Count ranks directly, independently of the searchsorted implementation.
            tails = {name: np.asarray([(1 + (bank[name].peaks >= x).sum()) / (len(bank[name].peaks) + 1)
                                      if np.isfinite(x) else np.nan for x in online[name]]) for name in BRANCHES}
            for name in BRANCHES:
                cached = bank[name].tail(branches[name][row_id, :row.length])
                tail_rank_changes += int((np.isfinite(cached) & (cached != tails[name])).sum())
            for mi, method in enumerate(METHODS):
                names = METHOD_BRANCHES[method]
                combined = np.fmin.reduce([tails[name] for name in names]) * len(names)
                combined = np.minimum(1., combined)
                for ai, alpha in enumerate(ALPHAS):
                    hits = np.flatnonzero(np.isfinite(combined) & (combined <= alpha))
                    actual = int(hits[0]) if len(hits) else -1
                    expected = int(first[ki, ai, mi, ix])
                    mismatch = actual != expected
                    all_setting_differences += mismatch
                    if alpha == .01 and method == "v82_integrated":
                        primary_differences += mismatch
                    records.append(dict(global_row=row_id, task=row.task, episode=row.episode, failure=row.failure,
                                        boundary_selected=row_id in boundary_rows, calibration=kind,
                                        alpha=alpha, method=method, cached_first=expected, raw_first=actual,
                                        exact_first_match=not mismatch))
        expected_first = int(first[KINDS.index("task_init"), ALPHAS.index(.01), METHODS.index("v82_integrated"), ix])
        primary_differences += live.alert.first_alarm_query != expected_first
        query_checks += row.length
        if (number + 1) % 20 == 0:
            print(f"RAW REPLAY {number+1}/{len(rows)} episodes, {query_checks} queries", flush=True)
    pd.DataFrame(records).to_csv(output / "raw_replay.csv", index=False)
    assert primary_differences == 0, "primary alarm changed under raw replay"
    return dict(raw_episodes=len(rows), raw_queries=query_checks, selected_rows=rows,
                boundary_selected_rows=boundary_rows, maximum_branch_error=maximum_error,
                branch_tail_rank_differences=tail_rank_changes, primary_first_alarm_differences=primary_differences,
                all_setting_first_alarm_differences=all_setting_differences)


def verify_metrics(b, output, tails, first, frozen):
    from run_guard_experiment import settings

    table = pd.read_csv(output / "alarm_metrics.csv")
    positions = b.suite.map(ACTION_CAPS).to_numpy()
    y = b.failure.to_numpy(bool)
    checks = 0
    for kind, alpha, method, alarms in settings(first, frozen):
        alarms = alarms.astype(np.int64)
        selected = table.loc[table.calibration.eq(kind) & table.method.eq(method)]
        selected = selected.loc[selected.alpha.isna() if np.isnan(alpha) else selected.alpha.eq(alpha)]
        for row in selected.itertuples():
            mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
            fired = alarms >= 0
            assert row.tp == int((mask & fired & y).sum()) and row.fp == int((mask & fired & ~y).sum())
            hit = mask & fired & y
            if hit.any():
                expected = np.median([100 * (10 * int(q)) / int(cap) for q, cap in zip(alarms[hit], positions[hit])])
                np.testing.assert_allclose(row.median_failure_cap_percent, expected, atol=1e-12)
            checks += 1
    episodes = pd.read_csv(output / "episode_alarms.csv")
    assert episodes.alarm_cap_percent.dropna().between(0, 100).all()
    query = pd.read_csv(output / "query_auroc.csv")
    independent_auc = 0
    for row in query.iloc[::13].itertuples():
        mi, ki = METHODS.index(row.method), KINDS.index(row.calibration)
        mask = b.task.eq(row.task).to_numpy() & np.isfinite(tails[ki, mi, :, row.query])
        labels, scores = y[mask], -np.log(tails[ki, mi, mask, row.query])
        expected = mannwhitneyu(scores[labels], scores[~labels]).statistic / (labels.sum() * (~labels).sum())
        np.testing.assert_allclose(row.auc, expected, atol=1e-12)
        independent_auc += 1
    image = np.asarray(Image.open(output / "alarm_curves.png").convert("RGB"))
    assert image.shape[0] > 700 and image.shape[1] > 1500 and (image.min(-1) < 245).mean() > .02
    return dict(metric_rows=checks, independent_mannwhitney_checks=independent_auc,
                plot_shape=image.shape, plot_nonwhite_share=float((image.min(-1) < 245).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "routing_guard_20260908")
    output = parser.parse_args().output.resolve()
    manifest = json.loads((output / "experiment_verification.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    contract = json.loads((output / "execution_contract.json").read_text())
    for name, expected in contract["sources"].items():
        assert digest(HERE / name) == expected, name
    frame = pd.read_csv(FEATURE_DIR / "index.csv")
    b = pd.read_csv(output / "test_index.csv")
    profiles = [json.loads((output / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    with np.load(FEATURE_DIR / "features.npz", allow_pickle=False) as z:
        valid = z["valid"]
        dynamics = dynamics_scores(z["features"], valid)
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        tails, first, frozen = z["tails"], z["first"], z["frozen_first"]
        old_scores = np.full(valid.shape, np.nan, np.float32)
        old_scores[z["global_rows"]] = z["raw_v82_scores"]
        # The reporting-only overflow correction and import fix must not change predictions.
        before = output.with_name("routing_guard_20260908_before_position_fix")
        if before.exists():
            with np.load(before / "predictions.npz", allow_pickle=False) as original:
                for name in z.files:
                    np.testing.assert_array_equal(z[name], original[name])
    cache_checks = replay_cached_baseline(frame, valid, profiles, old_scores)
    print(f"CACHE REPLAY {cache_checks} observed A-calibration/B scores match exactly", flush=True)
    raw = replay_raw(frame, b, profiles, dict(v82=old_scores, **dynamics), valid, first, output)
    metrics = verify_metrics(b, output, tails, first, frozen)
    write_json(output / "independent_verification.json", dict(
        all_checks_passed=True, cached_score_matches=cache_checks, **raw, **metrics,
        verifier_sha256=digest(__file__), package_import_supported=True,
        artifacts={name: digest(output / name) for name in ("raw_replay.csv", "predictions.npz", "alarm_metrics.csv")}))
    print(json.dumps(dict(cache_checks=cache_checks, **raw, **metrics), default=str, indent=2), flush=True)


if __name__ == "__main__":
    main()
