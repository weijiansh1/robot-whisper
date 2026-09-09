"""Post-result ablations and matched comparisons; no changes to fixed models."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from relative_models import COMPONENTS, METHODS, RAW_NAMES, reference_evidence
from encoder import FEATURE_NAMES
from guard import ALPHAS, KINDS, PeakBank, first_trigger
from model_tables import combined_alarm
from run_relative_models import BASE, PREVIOUS, inputs
from run_guard_experiment import ACTION_CAPS, FEATURE_DIR, S05
from run_analysis import bootstrap_indices, digest, write_json
from monitor import auc, union


def paired_comparisons(b, saved, out):
    records, intervals, cases = [], [], []
    scopes = [("all", b)] + [(str(name), part) for field in ("suite", "task") for name, part in b.groupby(field)]
    for kind in KINDS:
        for method in METHODS:
            first = combined_alarm(saved, kind, .01, method)
            for baseline in ("ac_reference", "old_ac", "previous_absolute_acc"):
                if method == baseline:
                    continue
                old = combined_alarm(saved, kind, .01, baseline)
                for scope, part in scopes:
                    ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
                    a, o = first[ix] >= 0, old[ix] >= 0
                    dt = first[ix][a & o & y] - old[ix][a & o & y]
                    records.append(dict(calibration=kind, method=method, baseline=baseline, scope=scope,
                        gained_tp=int((a & ~o & y).sum()), lost_tp=int((~a & o & y).sum()),
                        added_fp=int((a & ~o & ~y).sum()), removed_fp=int((~a & o & ~y).sum()),
                        earlier=int((dt < 0).sum()), later=int((dt > 0).sum())))
                counts, tasks, suites = [], [], []
                for task, part in b.groupby("task"):
                    ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
                    delta = (first[ix] >= 0).astype(int) - (old[ix] >= 0).astype(int)
                    counts.append([delta[y].sum(), y.sum(), delta[~y].sum(), (~y).sum()])
                    tasks.append(task)
                    suites.append(part.suite.iloc[0])
                values = np.asarray(counts)
                totals = values[bootstrap_indices(tasks, suites)].sum(1)
                for metric, num, den in (("recall", 0, 1), ("fpr", 2, 3)):
                    lo, hi = np.nanquantile(totals[:, num]/totals[:, den], [.025, .975])
                    intervals.append(dict(calibration=kind, method=method, baseline=baseline, metric=metric,
                        delta=values[:, num].sum()/values[:, den].sum(), lo=lo, hi=hi))
                if kind == "task_init":
                    for ix in np.flatnonzero((first >= 0) != (old >= 0)):
                        row = b.iloc[ix]
                        q = int(first[ix] if first[ix] >= 0 else old[ix])
                        cases.append(dict(method=method, baseline=baseline, global_row=row.global_row, task=row.task,
                            episode=row.episode, init_state_id=row.init_state_id, failure=row.failure,
                            old_first=int(old[ix]), new_first=int(first[ix]), configured_cap_percent=1000*q/ACTION_CAPS[row.suite]))
    pd.DataFrame(records).to_csv(out / "expanded_paired_changes.csv", index=False)
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    pd.DataFrame(cases).to_csv(out / "changed_cases.csv", index=False)


def evidence_and_decoupling_ablation(frame, b, valid, components, saved, profiles, out):
    evidence = {name: np.full_like(saved["scores"][0], np.nan) for name in RAW_NAMES}
    d_first = np.full((len(KINDS), len(ALPHAS), len(b)), -2, np.int16)
    banks_out, calibration = [], []
    for profile in profiles:
        refs = {name: PeakBank.from_list(bank) for name, bank in profile["references"].items()}
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        rows = b.iloc[ix].global_row.to_numpy()
        current = reference_evidence({name: value[rows] for name, value in components.items()}, refs)
        for name, value in current.items():
            evidence[name][ix] = value
        cal = np.asarray(profile["calibration_rows"])
        cal_e = reference_evidence({name: value[cal] for name, value in components.items()}, refs)
        success = ~frame.iloc[cal].failure.to_numpy(bool)

        def d_only(values):
            ready = np.isfinite(values["acceleration"]) | np.isfinite(values["decoupling"])
            return np.where(ready, np.where(np.isfinite(values["decoupling"]), values["decoupling"], 0.), np.nan)

        values = d_only(cal_e)[success]
        peaks = np.where(np.isfinite(values), values, -np.inf).max(1)
        grouped = frame.iloc[cal].loc[success, ["task", "init_state_id"]].copy()
        grouped["peak"] = peaks
        units = dict(episode=peaks, task_init=grouped.groupby(["task", "init_state_id"]).peak.max().to_numpy())
        banks = {kind: PeakBank(values) for kind, values in units.items()}
        banks_out.append(dict(fold=profile["fold"], banks={kind: bank.to_list() for kind, bank in banks.items()}))
        test_d = d_only(current)
        with np.load(out / f"fold_{profile['fold']}_calibration.npz", allow_pickle=False) as z:
            scores = z["scores"][:, success]
        old_peaks = np.where(np.isfinite(scores[0]), scores[0], -np.inf).max(1)
        for mi, method in enumerate(METHODS):
            peaks = np.where(np.isfinite(scores[mi]), scores[mi], -np.inf).max(1)
            for kind in KINDS:
                threshold, _ = PeakBank.from_list(profile["banks"][kind][method]).threshold(.01)
                gate = np.minimum(current["acceleration"], current["relative_curvature"])
                calibration.append(dict(fold=profile["fold"], calibration=kind, method=method,
                    threshold=threshold, control_threshold=PeakBank.from_list(profile["banks"][kind]["ac_reference"]).threshold(.01)[0],
                    d_only_threshold=banks[kind].threshold(.01)[0], success_peaks_reduced=int((peaks < old_peaks).sum()),
                    primary_acc_gate_queries_above_threshold=int((gate > threshold).sum()) if method == "relative_curvature" else None))
        for ki, kind in enumerate(KINDS):
            for ai, alpha in enumerate(ALPHAS):
                d_first[ki, ai, ix] = first_trigger(banks[kind].tail(test_d), valid[rows], alpha)
    np.savez_compressed(out / "reference_evidence.npz", **evidence, global_rows=b.global_row.to_numpy())
    np.savez_compressed(out / "decoupling_ablation.npz", first=d_first, kinds=np.asarray(KINDS), alphas=np.asarray(ALPHAS))
    write_json(out / "decoupling_ablation_profiles.json", dict(post_result_diagnostic=True, profiles=banks_out))
    pd.DataFrame(calibration).to_csv(out / "calibration_effect.csv", index=False)
    rows = []
    for ki, kind in enumerate(KINDS):
        for ai, alpha in enumerate(ALPHAS):
            first = union(saved["frozen_first"], d_first[ki, ai])
            y = b.failure.to_numpy(bool)
            for method in METHODS:
                original = saved["first"][ki, ai, METHODS.index(method)]
                rows.append(dict(calibration=kind, alpha=alpha, method=method,
                    d_only_tp=int(((first >= 0) & y).sum()), d_only_fp=int(((first >= 0) & ~y).sum()),
                    branch_first_differences=int((original != d_first[ki, ai]).sum()),
                    combined_first_differences=int((combined_alarm(saved, kind, alpha, method) != first).sum())))
    pd.DataFrame(rows).to_csv(out / "decoupling_ablation_comparison.csv", index=False)
    return evidence


def recovery_audit(b, evidence, saved, out):
    rows = []
    primary_i, recovery_i = [METHODS.index(name) for name in ("relative_curvature", "relative_recovery")]
    relief = evidence["routing_relief"]
    ready_relief = np.isfinite(relief) & (relief > 0)
    for ki, kind in enumerate(KINDS):
        first = saved["first"][ki, ALPHAS.index(.01), primary_i]
        q = np.arange(relief.shape[1])[None]
        for label in (False, True):
            selected = b.failure.eq(label).to_numpy()
            triggered = selected & (first >= 0)
            lowered = saved["scores"][recovery_i] < saved["scores"][primary_i]
            rows.append(dict(calibration=kind, failure=label, episodes=int(selected.sum()),
                positive_relief_episodes=int(ready_relief[selected].any(1).sum()),
                score_lowered_queries=int(lowered[selected].sum()), branch_alarm_episodes=int(triggered.sum()),
                relief_at_or_before_first=int((triggered & (ready_relief & (q <= first[:, None])).any(1)).sum()),
                relief_only_after_first=int((triggered & ready_relief.any(1) & ~(ready_relief & (q <= first[:, None])).any(1)).sum()),
                first_alarm_differences=int((saved["first"][ki, ALPHAS.index(.01), recovery_i, selected] != first[selected]).sum())))
    pd.DataFrame(rows).to_csv(out / "recovery_timing.csv", index=False)
    order_checks = 0
    for low, high in (("relative_curvature", "ac_reference"), ("relative_both", "relative_curvature"),
                      ("relative_recovery", "relative_curvature"), ("relative_curvature", "relative_soft"),
                      ("layer_consensus", "ac_reference")):
        a, c = saved["scores"][METHODS.index(low)], saved["scores"][METHODS.index(high)]
        valid = np.isfinite(a) & np.isfinite(c)
        assert (a[valid] <= c[valid] + 1e-12).all()
        order_checks += int(valid.sum())
    return order_checks


def relative_diagnostics(b, saved, out):
    with np.load(FEATURE_DIR / "features.npz", allow_pickle=False) as z:
        f = z["features"][b.global_row.to_numpy()].astype(float)
    a, p, m = [f[..., FEATURE_NAMES.index(name)] for name in ("acceleration_log_ratio", "path_log_ratio", "mobility_log_ratio")]
    values = dict(acceleration=a, path=p, signed_acc_minus_path=a-p, positive_acc_minus_path=np.maximum(a-p, 0.), signed_path_minus_mobility=p-m)
    records, correlations = [], []
    for (suite, task), part in b.groupby(["suite", "task"]):
        ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
        for q in range(8, f.shape[1]):
            observed = np.isfinite(a[ix, q]) & np.isfinite(saved["raw_v82_scores"][ix, q])
            labels, rows = y[observed], ix[observed]
            if not labels.any() or labels.all():
                continue
            for name, value in values.items():
                records.append(dict(task=task, suite=suite, query=q, feature=name, auc=auc(labels, value[rows, q])))
            for label in (False, True):
                select = rows[labels == label]
                av, pv = a[select, q], p[select, q]
                if len(select) >= 5 and np.unique(av).size > 1 and np.unique(pv).size > 1:
                    correlations.append(dict(task=task, suite=suite, query=q, failure=label, samples=len(select),
                                             acceleration_path_spearman=float(spearmanr(av, pv).statistic)))
    data = pd.DataFrame(records)
    data.to_csv(out / "relative_feature_query_auroc.csv", index=False)
    pd.DataFrame(correlations).to_csv(out / "within_query_correlations.csv", index=False)
    rows = []
    for window, selected in (("q8_13", data.loc[data["query"].between(8,13)]), ("all_comparable_queries", data)):
        tasks = selected.groupby(["feature", "suite", "task"], as_index=False).auc.mean()
        for name, part in tasks.groupby("feature"):
            draws = bootstrap_indices(part.task, part.suite)
            lo, hi = np.quantile(part.auc.to_numpy()[draws].mean(1), [.025,.975])
            rows.append(dict(feature=name, window=window, auc=part.auc.mean(), lo=lo, hi=hi, tasks=len(part)))
    pd.DataFrame(rows).to_csv(out / "relative_feature_auroc_summary.csv", index=False)


def compare_absolute_auc(out):
    current = pd.read_csv(out / "query_auroc.csv")
    previous = pd.read_csv(PREVIOUS / "query_auroc.csv")
    previous = previous.loc[previous.method.eq("absolute_acc"), ["calibration", "score_kind", "suite", "task", "query", "auc", "failures", "successes"]]
    merged = current.merge(previous, on=["calibration", "score_kind", "suite", "task", "query"], suffixes=("", "_absolute"), validate="many_to_one")
    assert len(merged) == len(current)
    np.testing.assert_array_equal(merged.failures, merged.failures_absolute)
    np.testing.assert_array_equal(merged.successes, merged.successes_absolute)
    merged["delta_absolute"] = merged.auc - merged.auc_absolute
    rows = []
    for window, selected in (("q8_13", merged.loc[merged["query"].between(8,13)]), ("all_comparable_queries", merged)):
        fields = ("auc", "auc_absolute", "delta_absolute")
        tasks = selected.groupby(["calibration", "score_kind", "method", "suite", "task"], as_index=False)[list(fields)].mean()
        for (kind, score_kind, method), part in tasks.groupby(["calibration", "score_kind", "method"]):
            draws = bootstrap_indices(part.task, part.suite)
            for field in fields:
                values = part[field].to_numpy()
                lo, hi = np.quantile(values[draws].mean(1), [.025,.975])
                rows.append(dict(calibration=kind, score_kind=score_kind, method=method, window=window,
                                 metric=field, estimate=values.mean(), lo=lo, hi=hi, tasks=len(part)))
    pd.DataFrame(rows).to_csv(out / "absolute_auroc_comparison.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "relative_routing_models_20260909")
    out = parser.parse_args().output.resolve()
    assert json.loads((out / "independent_verification.json").read_text())["all_checks_passed"]
    frame, valid, components, _ = inputs()
    b = pd.read_csv(out / "test_index.csv")
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    profiles = [json.loads((out / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    paired_comparisons(b, saved, out)
    evidence = evidence_and_decoupling_ablation(frame, b, valid, components, saved, profiles, out)
    order_checks = recovery_audit(b, evidence, saved, out)
    relative_diagnostics(b, saved, out)
    compare_absolute_auc(out)
    write_json(out / "mechanism_verification.json", dict(all_checks_passed=True, diagnostic_only=True,
        runtime_models_unchanged=True, pointwise_order_checks=order_checks, analyzer_sha256=digest(__file__)))
    print(f"MECHANISMS: {order_checks} ordering checks; D-only diagnostic and matched absolute comparison complete", flush=True)


if __name__ == "__main__":
    main()
