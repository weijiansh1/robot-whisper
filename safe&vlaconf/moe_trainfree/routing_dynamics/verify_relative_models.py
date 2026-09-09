"""Rebuild calibration banks and independently audit raw routing model decisions."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import zarr

from relative_models import COMPONENTS, RAW_NAMES, METHODS, PRIMARY, RelativeModelMonitor
from guard import ALPHAS, KINDS, PeakBank
from run_relative_models import BASE, HERE, inputs
from run_guard_experiment import ROOT, S05
from run_analysis import digest, write_json


def audit_calibration(frame, valid, components, profiles, output):
    checks = 0
    for profile in profiles:
        ref_all = np.asarray(profile["reference_rows"])
        ref = np.asarray(profile["reference_success_rows"])
        cal = np.asarray(profile["calibration_rows"])
        test = np.asarray(profile["test_rows"])
        groups = [set(zip(frame.iloc[ix].task, frame.iloc[ix].init_state_id)) for ix in (ref_all, cal, test)]
        assert not any(groups[i] & groups[j] for i in range(3) for j in range(i))
        np.testing.assert_array_equal(ref, ref_all[~frame.iloc[ref_all].failure.to_numpy(bool)])
        for name in COMPONENTS:
            current = components[name][ref]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            np.testing.assert_array_equal(PeakBank.from_list(profile["references"][name]).peaks, np.sort(peaks))
            checks += 1
        with np.load(output / f"fold_{profile['fold']}_calibration.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["rows"], cal)
            np.testing.assert_array_equal(z["valid"], valid[cal])
            np.testing.assert_array_equal(z["methods"], METHODS)
            success, scores = z["success"], z["scores"]
        np.testing.assert_array_equal(success, ~frame.iloc[cal].failure.to_numpy(bool))
        np.testing.assert_array_equal(cal[success], profile["calibration_success_rows"])
        for mi, name in enumerate(METHODS):
            current = scores[mi, success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            grouped = frame.iloc[cal].loc[success, ["task", "init_state_id"]].copy()
            grouped["peak"] = peaks
            for kind, units in dict(episode=peaks, task_init=grouped.groupby(["task", "init_state_id"]).peak.max().to_numpy()).items():
                np.testing.assert_array_equal(PeakBank.from_list(profile["banks"][kind][name]).peaks, np.sort(units))
                checks += 1
    return checks


def select_raw(b, saved, profiles):
    selected, boundary = set(), set()
    for _, part in b.groupby(["task", "failure"]):
        selected.add(int(part.index[0]))
    selected.update(b.index[b.task.eq(S05) & b.episode.isin([77, 96, 102, 115, 174, 190, 286, 363, 366, 367])])
    primary = saved["first"][KINDS.index("task_init"), ALPHAS.index(.01), METHODS.index(PRIMARY)]
    selected.update(np.flatnonzero((primary >= 0) & (saved["frozen_first"] < 0)))
    for profile in profiles:
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        for kind in KINDS:
            for mi, name in enumerate(METHODS):
                threshold, _ = PeakBank.from_list(profile["banks"][kind][name]).threshold(.01)
                delta = saved["scores"][mi, ix] - threshold
                for side in (delta <= 0, delta > 0):
                    distance = np.where(np.isfinite(delta) & side, np.abs(delta), np.inf)
                    if np.isfinite(distance).any():
                        local, _ = np.unravel_index(distance.argmin(), distance.shape)
                        boundary.add(int(ix[local]))
    selected.update(boundary)
    return sorted(selected), boundary


def raw_replay(b, components, saved, profiles, output):
    selected, boundary = select_raw(b, saved, profiles)
    groups, offsets, records = {}, {}, []
    errors, component_errors = dict.fromkeys(METHODS, 0.), dict.fromkeys(RAW_NAMES, 0.)
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
        live = RelativeModelMonitor(output / f"fold_{profile['fold']}_profile.json", row.checkpoint)
        observed = [live.update(x) for x in raw]
        for name in RAW_NAMES:
            actual = np.asarray([r["components"][name] for r in observed])
            expected = components[name][row.global_row, :row.length]
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
            np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3, equal_nan=True)
            finite = np.isfinite(actual)
            if finite.any():
                component_errors[name] = max(component_errors[name], float(np.max(np.abs(actual[finite] - expected[finite]))))
        for before, after in zip(observed, observed[1:]):
            if before["ever_alarm"]:
                assert after["ever_alarm"] and before["first_alarm_query"] == after["first_alarm_query"]
        frozen_mismatches += int(observed[-1]["frozen_first_query"] != saved["frozen_first"][ix])
        actual_scores = np.asarray([[r["scores"][name] for r in observed] for name in METHODS])
        for mi, name in enumerate(METHODS):
            expected = saved["scores"][mi, ix, :row.length]
            np.testing.assert_array_equal(np.isfinite(actual_scores[mi]), np.isfinite(expected))
            finite = np.isfinite(expected)
            if finite.any():
                errors[name] = max(errors[name], float(np.max(np.abs(actual_scores[mi, finite] - expected[finite]))))
        for ki, kind in enumerate(KINDS):
            for mi, name in enumerate(METHODS):
                bank = PeakBank.from_list(profile["banks"][kind][name])
                tail = np.asarray([(1 + (bank.peaks >= x).sum()) / (len(bank.peaks) + 1)
                                   if np.isfinite(x) else np.nan for x in actual_scores[mi]])
                for ai, alpha in enumerate(ALPHAS):
                    hits = np.flatnonzero(np.isfinite(tail) & (tail <= alpha))
                    actual = int(hits[0]) if len(hits) else -1
                    expected = int(saved["first"][ki, ai, mi, ix])
                    mismatch = actual != expected
                    mismatches += int(mismatch)
                    if kind == "task_init" and alpha == .01 and name == PRIMARY:
                        primary_mismatches += int(mismatch)
                        first_both = min([x for x in (actual, int(saved["frozen_first"][ix])) if x >= 0], default=-1)
                        assert live.first_alarm == first_both
                    records.append(dict(global_row=row.global_row, task=row.task, episode=row.episode,
                                        failure=row.failure, boundary=ix in boundary, calibration=kind,
                                        alpha=alpha, method=name, cached_first=expected, raw_first=actual, exact=not mismatch))
        queries += int(row.length)
        if (number + 1) % 20 == 0:
            print(f"RAW RELATIVE {number+1}/{len(selected)} episodes, {queries} queries", flush=True)
    pd.DataFrame(records).to_csv(output / "raw_replay.csv", index=False)
    assert frozen_mismatches == 0
    return dict(raw_episodes=len(selected), raw_queries=queries, boundary_episodes=len(boundary),
                selected_global_rows=b.iloc[selected].global_row.tolist(), maximum_component_error=component_errors,
                maximum_model_score_error=errors, first_alarm_checks=len(records), first_alarm_mismatches=mismatches,
                primary_first_alarm_mismatches=primary_mismatches, original_frozen_first_alarm_mismatches=frozen_mismatches)


def audit_metrics(b, saved, output):
    frozen, controls = saved["frozen_first"].astype(np.int64), saved["control_first"]
    y = b.failure.to_numpy(bool)

    def alarms(kind, alpha, method):
        if method == "v82_frozen":
            return frozen
        ki, ai = KINDS.index(kind), ALPHAS.index(alpha)
        branch = (saved["first"][ki, ai, METHODS.index(method)] if method in METHODS
                  else controls[ki, ai, list(saved["control_names"]).index(method)]).astype(np.int64)
        both = np.minimum(np.where(frozen < 0, 1_000_000, frozen), np.where(branch < 0, 1_000_000, branch))
        return np.where(both == 1_000_000, -1, both)

    preservation = 0
    for kind in KINDS:
        for alpha in ALPHAS:
            for method in METHODS:
                alarm = alarms(kind, alpha, method)
                old_hit = frozen >= 0
                assert ((alarm[old_hit] >= 0) & (alarm[old_hit] <= frozen[old_hit])).all()
                preservation += len(b)
    metrics = pd.read_csv(output / "alarm_metrics.csv")
    for row in metrics.itertuples():
        hit = alarms(row.calibration, row.alpha, row.method) >= 0
        mask = np.ones(len(b), bool) if row.scope == "all" else (b.task.eq(row.scope) | b.suite.eq(row.scope)).to_numpy()
        assert row.tp == int((mask & y & hit).sum()) and row.fp == int((mask & ~y & hit).sum())
    paired = pd.read_csv(output / "paired_changes.csv")
    for row in paired.itertuples():
        a = alarms(row.calibration, row.alpha, row.method) >= 0
        o = alarms(row.calibration, row.alpha, row.baseline) >= 0
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
                & np.isfinite(values[METHODS.index("ac_reference"), :, q])
                & np.isfinite(saved["old_ac_rank"][ki, :, q]) & np.isfinite(saved["raw_v82_scores"][:, q]))
        positive, negative = values[mi, mask & y, q], values[mi, mask & ~y, q]
        expected = mannwhitneyu(positive, negative).statistic / (len(positive) * len(negative))
        np.testing.assert_allclose(expected, row.auc, atol=1e-12)
        assert len(positive) == row.failures and len(negative) == row.successes
        auc_checks += 1
    return dict(metric_rows=len(metrics), paired_rows=len(paired), independent_auc_checks=auc_checks,
                original_alarm_preservation_checks=preservation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "relative_routing_models_20260909")
    output = parser.parse_args().output.resolve()
    for name, expected in json.loads((output / "experiment_verification.json").read_text())["artifacts"].items():
        assert digest(output / name) == expected, name
    for name, expected in json.loads((output / "execution_contract.json").read_text())["sources"].items():
        assert digest(HERE / name) == expected, name
    frame, valid, components, _ = inputs()
    b = pd.read_csv(output / "test_index.csv")
    profiles = [json.loads((output / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    np.testing.assert_array_equal(saved["global_rows"], b.global_row)
    checks = audit_calibration(frame, valid, components, profiles, output)
    print(f"CALIBRATION: {checks} reference and final peak banks independently rebuilt", flush=True)
    raw = raw_replay(b, components, saved, profiles, output)
    metrics = audit_metrics(b, saved, output)
    result = dict(all_checks_passed=raw["first_alarm_mismatches"] == 0, calibration_banks_checked=checks,
                  **raw, **metrics, verifier_sha256=digest(__file__), raw_replay_sha256=digest(output / "raw_replay.csv"))
    write_json(output / "independent_verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "selected_global_rows"}, indent=2), flush=True)
    if not result["all_checks_passed"]:
        raise SystemExit("raw/cached decisions differ; inspect raw_replay.csv")


if __name__ == "__main__":
    main()
