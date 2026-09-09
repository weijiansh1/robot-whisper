"""Audit disjoint calibration, raw replay, and new combination metrics."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import zarr

from combinations import COMPONENTS, METHODS, PRIMARY, CombinationMonitor, component_scores
from guard import ALPHAS, KINDS, PeakBank
from run_combinations import BASE, HERE, PREVIOUS, ROOT, S05, load_inputs
from run_analysis import digest, write_json
from monitor import union


def audit_calibration(frame, valid, components, profiles, output):
    checks = 0
    for profile in profiles:
        ref = np.asarray(profile["reference_success_rows"])
        cal = np.asarray(profile["calibration_rows"])
        test = np.asarray(profile["test_rows"])
        group_sets = [set(zip(frame.iloc[ix].task, frame.iloc[ix].init_state_id)) for ix in (ref, cal, test)]
        assert not any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i))
        assert not frame.iloc[ref].failure.any()
        for name in COMPONENTS:
            values = components[name][ref]
            expected = np.sort(np.where(np.isfinite(values), values, -np.inf).max(1))
            actual = PeakBank.from_list(profile["references"][name]).peaks
            np.testing.assert_array_equal(actual, expected)
            checks += 1
        path = output / f"fold_{profile['fold']}_calibration.npz"
        with np.load(path, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["rows"], cal)
            np.testing.assert_array_equal(z["valid"], valid[cal])
            success, scores = z["success"], z["scores"]
        np.testing.assert_array_equal(success, ~frame.iloc[cal].failure.to_numpy(bool))
        np.testing.assert_array_equal(cal[success], profile["calibration_success_rows"])
        for mi, name in enumerate(METHODS):
            current = scores[mi, success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            grouped = frame.iloc[cal].loc[success, ["task", "init_state_id"]].copy()
            grouped["peak"] = peaks
            units = dict(episode=peaks, task_init=grouped.groupby(["task", "init_state_id"]).peak.max().to_numpy())
            for kind, expected in units.items():
                np.testing.assert_array_equal(PeakBank.from_list(profile["banks"][kind][name]).peaks, np.sort(expected))
                checks += 1
    return checks


def select_raw(b, scores, first, frozen, profiles):
    selected, boundary = set(), set()
    for _, part in b.groupby(["task", "failure"]):
        selected.add(int(part.index[0]))
    selected.update(b.index[b.task.eq(S05) & b.episode.isin([77, 96, 102, 115, 174, 190, 286, 363, 366, 367])])
    primary = first[KINDS.index("task_init"), ALPHAS.index(.01), METHODS.index(PRIMARY)]
    selected.update(np.flatnonzero((primary >= 0) & (frozen < 0)))
    for profile in profiles:
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        for kind in KINDS:
            for mi, name in enumerate(METHODS):
                threshold, _ = PeakBank.from_list(profile["banks"][kind][name]).threshold(.01)
                delta = scores[mi, ix] - threshold
                for side in (delta <= 0, delta > 0):
                    distance = np.where(np.isfinite(delta) & side, np.abs(delta), np.inf)
                    if np.isfinite(distance).any():
                        local, _ = np.unravel_index(distance.argmin(), distance.shape)
                        boundary.add(int(ix[local]))
    selected.update(boundary)
    return sorted(selected), boundary


def raw_replay(b, components, scores, first, frozen, profiles, output):
    selected, boundary = select_raw(b, scores, first, frozen, profiles)
    groups, offsets, records = {}, {}, []
    errors = dict.fromkeys(METHODS, 0.)
    component_errors = dict.fromkeys(COMPONENTS, 0.)
    queries, mismatches, primary_mismatches, frozen_mismatches = 0, 0, 0, 0
    for number, ix in enumerate(selected):
        row = b.iloc[ix]
        profile = profiles[int(row.init_state_id % 5)]
        if row.source not in groups:
            run = ROOT / "VLA_MUI_HUB" / row.source
            groups[row.source] = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
            summaries = sorted(json.loads((run / "client/summaries.json").read_text()), key=lambda x: x["episode_index"])
            offsets[row.source] = np.r_[0, np.cumsum([x["inference_calls"] for x in summaries])]
        lo, hi = offsets[row.source][row.episode:row.episode + 2]
        assert hi - lo == row.length
        np.testing.assert_array_equal(groups[row.source]["episode_id"][lo:hi], row.episode)
        raw = groups[row.source]["hb_router_probs"][lo:hi]
        live = CombinationMonitor(output / f"fold_{profile['fold']}_profile.json", row.checkpoint)
        observed = [live.update(x) for x in raw]
        current = component_scores({name: np.asarray([r["base_scores"][name] for r in observed])
                                    for name in ("acceleration", "decoupling", "gap", "state_relative_low")})
        for name, values in current.items():
            expected = components[name][row.global_row, :row.length]
            np.testing.assert_array_equal(np.isfinite(values), np.isfinite(expected))
            np.testing.assert_allclose(values, expected, atol=2e-3, rtol=2e-3, equal_nan=True)
            finite = np.isfinite(values)
            if finite.any():
                component_errors[name] = max(component_errors[name], float(np.max(np.abs(values[finite] - expected[finite]))))
        for before, after in zip(observed, observed[1:]):
            if before["ever_alarm"]:
                assert after["ever_alarm"] and before["first_alarm_query"] == after["first_alarm_query"]
        frozen_mismatches += observed[-1]["frozen_first_query"] != frozen[ix]
        actual_scores = np.asarray([[r["scores"][name] for r in observed] for name in METHODS])
        for mi, name in enumerate(METHODS):
            expected = scores[mi, ix, :row.length]
            np.testing.assert_array_equal(np.isfinite(actual_scores[mi]), np.isfinite(expected))
            finite = np.isfinite(expected)
            if finite.any():
                errors[name] = max(errors[name], float(np.max(np.abs(actual_scores[mi, finite] - expected[finite]))))
        for ki, kind in enumerate(KINDS):
            for mi, name in enumerate(METHODS):
                bank = PeakBank.from_list(profile["banks"][kind][name])
                # Count ranks directly to independently check the binary decisions.
                tail = np.asarray([(1 + (bank.peaks >= x).sum()) / (len(bank.peaks) + 1)
                                   if np.isfinite(x) else np.nan for x in actual_scores[mi]])
                for ai, alpha in enumerate(ALPHAS):
                    hits = np.flatnonzero(np.isfinite(tail) & (tail <= alpha))
                    actual = int(hits[0]) if len(hits) else -1
                    expected = int(first[ki, ai, mi, ix])
                    mismatch = actual != expected
                    mismatches += mismatch
                    if kind == "task_init" and alpha == .01 and name == PRIMARY:
                        primary_mismatches += mismatch
                        first_both = min([x for x in (actual, int(frozen[ix])) if x >= 0], default=-1)
                        assert live.first_alarm == first_both
                    records.append(dict(global_row=row.global_row, task=row.task, episode=row.episode,
                                        failure=row.failure, boundary=ix in boundary, calibration=kind,
                                        alpha=alpha, method=name, cached_first=expected, raw_first=actual,
                                        exact=not mismatch))
        queries += int(row.length)
        if (number + 1) % 20 == 0:
            print(f"RAW COMBINATIONS {number+1}/{len(selected)} episodes, {queries} queries", flush=True)
    pd.DataFrame(records).to_csv(output / "raw_replay.csv", index=False)
    assert frozen_mismatches == 0
    return dict(raw_episodes=len(selected), raw_queries=queries, boundary_episodes=len(boundary),
                selected_global_rows=b.iloc[selected].global_row.tolist(), maximum_component_error=component_errors,
                maximum_combination_score_error=errors, first_alarm_checks=len(records),
                first_alarm_mismatches=mismatches, primary_first_alarm_mismatches=primary_mismatches,
                original_frozen_first_alarm_mismatches=int(frozen_mismatches))


def audit_metrics(b, saved, output):
    first, frozen, controls = saved["first"], saved["frozen_first"].astype(np.int64), saved["control_first"]
    y = b.failure.to_numpy(bool)

    def alarms(kind, alpha, method):
        if method == "v82_frozen":
            return frozen
        ki, ai = KINDS.index(kind), ALPHAS.index(alpha)
        branch = first[ki, ai, METHODS.index(method)] if method in METHODS else controls[ki, ai, list(saved["control_names"]).index(method)]
        branch = branch.astype(np.int64)
        return np.minimum(np.where(frozen < 0, 1_000_000, frozen), np.where(branch < 0, 1_000_000, branch))

    metrics = pd.read_csv(output / "alarm_metrics.csv")
    for row in metrics.itertuples():
        alarm = alarms(row.calibration, row.alpha, row.method)
        hit = (alarm >= 0) & (alarm < 1_000_000)
        mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
        assert row.tp == int((mask & y & hit).sum()) and row.fp == int((mask & ~y & hit).sum())
    paired = pd.read_csv(output / "paired_changes.csv")
    for row in paired.itertuples():
        new, old = alarms(row.calibration, row.alpha, row.method), alarms(row.calibration, row.alpha, row.baseline)
        a, o = (new >= 0) & (new < 1_000_000), (old >= 0) & (old < 1_000_000)
        mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
        assert row.gained_tp == int((mask & y & a & ~o).sum())
        assert row.lost_tp == int((mask & y & ~a & o).sum())
        assert row.added_fp == int((mask & ~y & a & ~o).sum())
        assert row.removed_fp == int((mask & ~y & ~a & o).sum())
    auc_checks = 0
    for row in pd.read_csv(output / "query_auroc.csv").iloc[::53].itertuples():
        ki, mi, q = KINDS.index(row.calibration), METHODS.index(row.method), row.query
        values = saved["scores"] if row.score_kind == "raw_score" else -np.log(saved["tails"][ki])
        mask = (b.task.eq(row.task).to_numpy() & np.isfinite(values[mi, :, q])
                & np.isfinite(values[METHODS.index("ac_recalibrated"), :, q])
                & np.isfinite(saved["old_ac_rank"][ki, :, q]) & np.isfinite(saved["raw_v82_scores"][:, q]))
        positive, negative = values[mi, mask & y, q], values[mi, mask & ~y, q]
        expected = mannwhitneyu(positive, negative).statistic / (len(positive) * len(negative))
        np.testing.assert_allclose(expected, row.auc, atol=1e-12)
        assert len(positive) == row.failures and len(negative) == row.successes
        auc_checks += 1
    return dict(metric_rows=len(metrics), paired_rows=len(paired), independent_auc_checks=auc_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "state_action_combinations_20260909")
    output = parser.parse_args().output.resolve()
    for name, expected in json.loads((output / "experiment_verification.json").read_text())["artifacts"].items():
        assert digest(output / name) == expected, name
    for name, expected in json.loads((output / "execution_contract.json").read_text())["sources"].items():
        assert digest(HERE / name) == expected, name
    frame, valid, components, _ = load_inputs()
    b = pd.read_csv(output / "test_index.csv")
    profiles = [json.loads((output / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    checks = audit_calibration(frame, valid, components, profiles, output)
    print(f"CALIBRATION: {checks} reference and joint peak banks independently rebuilt", flush=True)
    raw = raw_replay(b, components, saved["scores"], saved["first"], saved["frozen_first"], profiles, output)
    metrics = audit_metrics(b, saved, output)
    result = dict(all_checks_passed=raw["first_alarm_mismatches"] == 0, calibration_banks_checked=checks,
                  **raw, **metrics, verifier_sha256=digest(__file__), raw_replay_sha256=digest(output / "raw_replay.csv"))
    write_json(output / "independent_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_global_rows"}, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise SystemExit("raw/cached combination decisions differ; inspect raw_replay.csv")


if __name__ == "__main__":
    main()
