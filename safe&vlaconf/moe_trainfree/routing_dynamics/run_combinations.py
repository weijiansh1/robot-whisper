"""Fixed retrospective comparison of five new train-free ST/AC combinations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from combinations import (COMPONENTS, METHODS, PRIMARY, SCHEMA, combination_scores,
                          component_scores, reference_evidence)
from guard import ALPHAS, KINDS, PeakBank, dynamics_scores, first_trigger
from state_action import confirmed_relations
from run_guard_experiment import ACTION_CAPS, BASE, FEATURE_DIR, HERE, ROOT, RUN_B, S05, verified_inputs
from run_analysis import bootstrap_indices, digest, rate_interval, state_splits, write_json
from monitor import auc, union

PREVIOUS = BASE / "state_action_confirmation_20260909"
CONTROL_NAMES = ("old_ac", "old_hard_gate")


def load_inputs():
    frame, features, valid, hashes = verified_inputs()
    manifest = json.loads((PREVIOUS / "final_verification.json").read_text())
    for name in ("relations.npz", "predictions.npz", *(f"fold_{f}_profile.json" for f in range(5))):
        path = PREVIOUS / name
        assert digest(path) == manifest["artifacts"][name], name
        hashes[str(path)] = digest(path)
    with np.load(PREVIOUS / "relations.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["global_rows"], frame.global_row)
        np.testing.assert_array_equal(z["valid"], valid)
        relations = confirmed_relations(z["features"], valid)
    components = component_scores(dict(**dynamics_scores(features, valid), **relations))
    return frame, valid, components, hashes


def calibrate(frame, valid, components, output):
    profiles, threshold_rows = [], []
    for fold, ref, cal, test in state_splits(frame):
        base_path = PREVIOUS / f"fold_{fold}_profile.json"
        base = json.loads(base_path.read_text())
        for name, rows in (("reference_rows", ref), ("calibration_rows", cal), ("test_rows", test)):
            np.testing.assert_array_equal(base[name], rows)
        ref_success = ref[~frame.iloc[ref].failure.to_numpy(bool)]
        cal_success = ~frame.iloc[cal].failure.to_numpy(bool)
        reference_banks = {}
        for name, values in components.items():
            current = values[ref_success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            reference_banks[name] = PeakBank(peaks)
        evidence = reference_evidence({name: values[cal] for name, values in components.items()}, reference_banks)
        scores = combination_scores(evidence, valid[cal])
        group_index, groups = pd.factorize(pd.MultiIndex.from_frame(
            frame.iloc[cal].loc[cal_success, ["task", "init_state_id"]]), sort=True)
        banks = {kind: {} for kind in KINDS}
        for name, values in scores.items():
            current = values[cal_success]
            peaks = np.where(np.isfinite(current), current, -np.inf).max(1)
            grouped = np.full(len(groups), -np.inf)
            np.maximum.at(grouped, group_index, peaks)
            for kind, units in zip(KINDS, (peaks, grouped)):
                bank = PeakBank(units)
                banks[kind][name] = bank.to_list()
                for alpha in ALPHAS:
                    threshold, rank = bank.threshold(alpha)
                    threshold_rows.append(dict(fold=fold, calibration=kind, method=name, alpha=alpha,
                                               threshold=threshold, rank=rank, units=len(units),
                                               attainable=rank <= len(units), no_exposure_units=int(np.isneginf(units).sum())))
        profile = dict(schema=SCHEMA, fold=fold, methods=list(METHODS), primary=PRIMARY,
                       default_kind="task_init", default_alpha=.01,
                       base_profile=os.path.relpath(base_path, output), base_profile_sha256=digest(base_path),
                       references={name: bank.to_list() for name, bank in reference_banks.items()},
                       banks=banks, reference_rows=ref, reference_success_rows=ref_success,
                       calibration_rows=cal, calibration_success_rows=cal[cal_success], test_rows=test,
                       retrospective=True, no_classifier_training=True)
        write_json(output / f"fold_{fold}_profile.json", profile)
        np.savez_compressed(output / f"fold_{fold}_calibration.npz", rows=cal, success=cal_success,
                            valid=valid[cal], scores=np.stack([scores[name] for name in METHODS]), methods=np.asarray(METHODS))
        profiles.append(json.loads((output / f"fold_{fold}_profile.json").read_text()))
        print(f"CALIBRATED fold={fold}: reference successes={len(ref_success)}, calibration successes={cal_success.sum()}", flush=True)
    pd.DataFrame(threshold_rows).to_csv(output / "calibration_thresholds.csv", index=False)
    paths = [output / "execution_contract.json", output / "calibration_thresholds.csv", *output.glob("fold_*")]
    write_json(output / "calibration_seal.json", dict(phase="before_B_prediction", artifacts={p.name: digest(p) for p in paths}))
    return profiles


def predict(frame, valid, components, profiles, output):
    rows_b = np.flatnonzero(frame.run_id.eq(RUN_B))
    b = frame.iloc[rows_b].reset_index(drop=True)
    assert len(b) == 16000 and b.failure.sum() == 564
    inverse = np.full(len(frame), -1, int)
    inverse[rows_b] = np.arange(len(b))
    scores = np.full((len(METHODS), len(b), valid.shape[1]), np.nan)
    tails = np.full((len(KINDS), *scores.shape), np.nan)
    first = np.full((len(KINDS), len(ALPHAS), len(METHODS), len(b)), -2, np.int16)
    assigned = np.zeros(len(b), bool)
    for profile in profiles:
        rows = np.asarray(profile["test_rows"])
        ix = inverse[rows]
        assert not assigned[ix].any()
        assigned[ix] = True
        banks = {name: PeakBank.from_list(values) for name, values in profile["references"].items()}
        evidence = reference_evidence({name: values[rows] for name, values in components.items()}, banks)
        values = combination_scores(evidence, valid[rows])
        for mi, name in enumerate(METHODS):
            scores[mi, ix] = values[name]
            for ki, kind in enumerate(KINDS):
                tail = PeakBank.from_list(profile["banks"][kind][name]).tail(values[name])
                tails[ki, mi, ix] = tail
                for ai, alpha in enumerate(ALPHAS):
                    first[ki, ai, mi, ix] = first_trigger(tail, valid[rows], alpha)
    assert assigned.all() and (first >= -1).all()
    with np.load(PREVIOUS / "predictions.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["global_rows"], rows_b)
        np.testing.assert_array_equal(z["valid"], valid[rows_b])
        frozen, old_score = z["frozen_first"], z["raw_v82_scores"]
        indices = [list(z["methods"]).index(name) for name in ("ac_only", "confirmed_only")]
        controls = z["first"][:, :, indices]
        a, d = [list(z["branches"]).index(name) for name in ("acceleration", "decoupling")]
        old_ac_rank = -np.log(2 * np.fmin(z["tails"][:, a], z["tails"][:, d]))
    combined = first.copy()
    for ki in range(len(KINDS)):
        for ai in range(len(ALPHAS)):
            for mi in range(len(METHODS)):
                combined[ki, ai, mi] = union(frozen, first[ki, ai, mi])
                old_hit = frozen >= 0
                assert ((combined[ki, ai, mi, old_hit] >= 0) & (combined[ki, ai, mi, old_hit] <= frozen[old_hit])).all()
    np.savez_compressed(output / "predictions.npz", scores=scores, tails=tails, first=first,
                        combined_first=combined, frozen_first=frozen, control_first=controls,
                        control_names=np.asarray(CONTROL_NAMES), old_ac_rank=old_ac_rank, raw_v82_scores=old_score,
                        global_rows=rows_b, valid=valid[rows_b], methods=np.asarray(METHODS),
                        kinds=np.asarray(KINDS), alphas=np.asarray(ALPHAS))
    b.to_csv(output / "test_index.csv", index=False)
    print("PREDICTED: six fixed candidates on all B episodes; every original alarm preserved", flush=True)
    return b, scores, tails, first, combined, frozen, controls, old_ac_rank, old_score


def evaluate_alarms(b, first, combined, frozen, controls, output):
    y = b.failure.to_numpy(bool)
    caps = b.suite.map(ACTION_CAPS).to_numpy()
    scopes = [("all", np.arange(len(b)))]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("suite")]
    scopes += [(str(name), part.index.to_numpy()) for name, part in b.groupby("task")]
    metrics, paired, s05_rows, cases = [], [], [], []
    settings = [("original", np.nan, "v82_frozen", frozen)]
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            settings += [(kind, alpha, name, union(frozen, controls[ki, ai, ci])) for ci, name in enumerate(CONTROL_NAMES)]
            settings += [(kind, alpha, name, combined[ki, ai, mi]) for mi, name in enumerate(METHODS)]
    for kind, alpha, method, alarm in settings:
        alarm = alarm.astype(np.int64)
        for scope, ix in scopes:
            labels, hit = y[ix], alarm[ix] >= 0
            tp, fp = int((labels & hit).sum()), int((~labels & hit).sum())
            failures, successes = int(labels.sum()), int((~labels).sum())
            q = alarm[ix][hit & labels]
            row = dict(calibration=kind, alpha=alpha, method=method, scope=scope,
                       tp=tp, fp=fp, failures=failures, successes=successes,
                       recall=tp / failures if failures else np.nan, fpr=fp / successes if successes else np.nan,
                       median_failure_q=float(np.median(q)) if tp else np.nan,
                       median_failure_cap_percent=float(np.median(1000 * q / caps[ix][hit & labels])) if tp else np.nan)
            for fraction in (.5, .6):
                early = hit & (10 * alarm[ix] <= fraction * caps[ix])
                row[f"tp_by_cap_{int(100*fraction)}"] = int((early & labels).sum())
                row[f"fp_by_cap_{int(100*fraction)}"] = int((early & ~labels).sum())
            if scope == "all":
                row.update(rate_interval(b, alarm >= 0, y))
            metrics.append(row)
        if alpha == .01 or kind == "original":
            for row in b.loc[b.task.eq(S05)].itertuples():
                s05_rows.append(dict(calibration=kind, method=method, global_row=row.global_row,
                                     episode=row.episode, failure=row.failure, first_alarm=int(alarm[row.Index])))
        if method not in METHODS:
            continue
        ki, ai, mi = KINDS.index(kind), ALPHAS.index(alpha), METHODS.index(method)
        references = dict(v82_frozen=frozen, old_ac=union(frozen, controls[ki, ai, 0]),
                          ac_recalibrated=combined[ki, ai, METHODS.index("ac_recalibrated")])
        for baseline, old in references.items():
            if method == baseline:
                continue
            for scope, ix in scopes:
                a, o, labels = alarm[ix] >= 0, old[ix] >= 0, y[ix]
                shared = a & o & labels
                delta = alarm[ix][shared] - old[ix][shared]
                paired.append(dict(calibration=kind, alpha=alpha, method=method, baseline=baseline, scope=scope,
                                   gained_tp=int((a & ~o & labels).sum()), lost_tp=int((~a & o & labels).sum()),
                                   added_fp=int((a & ~o & ~labels).sum()), removed_fp=int((~a & o & ~labels).sum()),
                                   earlier=int((delta < 0).sum()), same=int((delta == 0).sum()), later=int((delta > 0).sum())))
        if alpha == .01:
            changed = (alarm != frozen) | ((alarm >= 0) & ~y)
            for i in np.flatnonzero(changed):
                row = b.iloc[i]
                cases.append(dict(calibration=kind, method=method, global_row=row.global_row, suite=row.suite,
                                  task=row.task, episode=row.episode, failure=row.failure,
                                  frozen_first=int(frozen[i]), added_first=int(first[ki, ai, mi, i]),
                                  combined_first=int(alarm[i]), added_new_alarm=bool(frozen[i] < 0)))
    for name, values in (("alarm_metrics.csv", metrics), ("paired_changes.csv", paired),
                         ("s05_alarms.csv", s05_rows), ("alarm_cases_alpha01.csv", cases)):
        pd.DataFrame(values).to_csv(output / name, index=False)
    return pd.DataFrame(metrics)


def evaluate_auroc(b, scores, tails, old_ac_rank, old_score, output):
    records = []
    for ki, kind in enumerate(KINDS):
        for score_kind, values in (("raw_score", scores), ("calibrated_rank", -np.log(tails[ki]))):
            reference = values[METHODS.index("ac_recalibrated")]
            for (suite, task), part in b.groupby(["suite", "task"]):
                ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
                for q in range(8, scores.shape[-1]):
                    for mi, method in enumerate(METHODS):
                        available = (np.isfinite(values[mi, ix, q]) & np.isfinite(reference[ix, q])
                                     & np.isfinite(old_ac_rank[ki, ix, q]) & np.isfinite(old_score[ix, q]))
                        labels = y[available]
                        if not labels.any() or labels.all():
                            continue
                        rows = ix[available]
                        actual = auc(labels, values[mi, rows, q])
                        control = auc(labels, reference[rows, q])
                        old = auc(labels, old_ac_rank[ki, rows, q])
                        records.append(dict(calibration=kind, score_kind=score_kind, method=method, suite=suite,
                                            task=task, query=q, auc=actual, control_auc=control, old_ac_auc=old,
                                            delta_control=actual - control, delta_old_ac=actual - old,
                                            v82_auc=auc(labels, old_score[rows, q]),
                                            failures=int(labels.sum()), successes=int((~labels).sum())))
    queries = pd.DataFrame(records)
    queries.to_csv(output / "query_auroc.csv", index=False)
    tasks, summaries = [], []
    fields = ("auc", "control_auc", "old_ac_auc", "delta_control", "delta_old_ac", "v82_auc")
    for window, part in (("q8_13", queries.loc[queries["query"].between(8, 13)]), ("all_comparable_queries", queries)):
        grouped = part.groupby(["calibration", "score_kind", "method", "suite", "task"], as_index=False).agg(
            **{field: (field, "mean") for field in fields}, queries=("query", "nunique"))
        grouped["window"] = window
        tasks.append(grouped)
        for (kind, score_kind, method), selected in grouped.groupby(["calibration", "score_kind", "method"]):
            for scope, rows in (("all", selected), ("excluding_S05", selected.loc[selected.task.ne(S05)])):
                draws = bootstrap_indices(rows.task, rows.suite)
                for field in fields:
                    values = rows[field].to_numpy()
                    lo, hi = np.quantile(values[draws].mean(1), [.025, .975])
                    summaries.append(dict(calibration=kind, score_kind=score_kind, method=method, window=window,
                                          scope=scope, metric=field, estimate=float(values.mean()),
                                          lo=float(lo), hi=float(hi), tasks=len(rows)))
    pd.concat(tasks, ignore_index=True).to_csv(output / "task_auroc.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "auroc_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "state_action_combinations_20260909")
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame, valid, components, hashes = load_inputs()
    source_names = ("combinations.py", "run_combinations.py", "COMBINATIONS_PROTOCOL_ZH.md", "state_action.py", "guard.py", "encoder.py")
    write_json(output / "execution_contract.json", dict(
        retrospective=True, no_classifier_training=True, B_parameter_selection=False,
        primary=dict(method=PRIMARY, calibration="task_init", alpha=.01), methods=METHODS,
        kinds=KINDS, alphas=ALPHAS, reference_and_calibration_disjoint=True,
        raw_routing_only=True, duration_baseline_computed=False, inherited_v82_query_slope=.0015,
        inputs=hashes, sources={name: digest(HERE / name) for name in source_names}))
    profiles = calibrate(frame, valid, components, output)
    b, scores, tails, first, combined, frozen, controls, old_ac_rank, old_score = predict(frame, valid, components, profiles, output)
    metrics = evaluate_alarms(b, first, combined, frozen, controls, output)
    evaluate_auroc(b, scores, tails, old_ac_rank, old_score, output)
    for name, expected in json.loads((output / "calibration_seal.json").read_text())["artifacts"].items():
        assert digest(output / name) == expected, name
    write_json(output / "experiment_verification.json", dict(
        all_checks_passed=True, B_episodes=len(b), B_failures=int(b.failure.sum()),
        every_B_episode_once=True, frozen_alarm_preservation_checks=2 * 4 * len(METHODS) * len(b),
        calibration_seal_unchanged=True, artifacts={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
    print(metrics.loc[metrics.scope.eq("all") & (metrics.alpha.eq(.01) | metrics.calibration.eq("original")),
                      ["calibration", "method", "tp", "fp", "recall", "fpr"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
