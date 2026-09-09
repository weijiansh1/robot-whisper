"""Post-diagnostic direction comparisons and active-branch attribution."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from relative_direction_models import COMPONENTS, METHODS, reference_evidence
from guard import ALPHAS, KINDS, PeakBank
from model_tables import combined_alarm
from run_relative_direction_models import BASE, PREVIOUS, inputs
from run_guard_experiment import ACTION_CAPS
from run_analysis import bootstrap_indices, digest, write_json


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
    parser.add_argument("--output", type=Path, default=BASE / "relative_direction_models_20260909")
    out = parser.parse_args().output.resolve()
    assert json.loads((out / "independent_verification.json").read_text())["all_checks_passed"]
    frame, valid, components, _ = inputs()
    b = pd.read_csv(out / "test_index.csv")
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    profiles = [json.loads((out / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    paired_comparisons(b, saved, out)
    compare_absolute_auc(out)
    branches, order_checks = [], 0
    for profile in profiles:
        refs = {name: PeakBank.from_list(bank) for name, bank in profile["references"].items()}
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        rows = b.iloc[ix].global_row.to_numpy()
        e = reference_evidence({name: value[rows] for name, value in components.items()}, refs)
        for kind in KINDS:
            ki = KINDS.index(kind)
            for method in METHODS[1:]:
                mi = METHODS.index(method)
                source = e["curvature_magnitude"] if method == "relative_magnitude" else e["curvature_low"]
                gate = np.sqrt(e["acceleration"]*source) if method == "relative_low_soft" else np.minimum(e["acceleration"], source)
                threshold = PeakBank.from_list(profile["banks"][kind][method]).threshold(.01)[0]
                first = saved["first"][ki, ALPHAS.index(.01), mi, ix]
                for label in (False, True):
                    selected = b.iloc[ix].failure.eq(label).to_numpy()
                    hit = np.flatnonzero(selected & (first >= 0))
                    supports = gate[hit, first[hit]] > threshold
                    branches.append(dict(fold=profile["fold"], calibration=kind, method=method, failure=label,
                        threshold=threshold, branch_alarm_episodes=len(hit), acceleration_gate_supports_first=int(supports.sum())))
    for low, high in (("relative_low","ac_reference"), ("relative_magnitude","ac_reference"), ("relative_low","relative_low_soft")):
        a, c = saved["scores"][METHODS.index(low)], saved["scores"][METHODS.index(high)]
        observed = np.isfinite(a) & np.isfinite(c)
        assert (a[observed] <= c[observed] + 1e-12).all()
        order_checks += int(observed.sum())
    pd.DataFrame(branches).to_csv(out / "branch_attribution.csv", index=False)
    write_json(out / "mechanism_verification.json", dict(all_checks_passed=True, diagnostic_only=True,
        runtime_models_unchanged=True, pointwise_order_checks=order_checks, analyzer_sha256=digest(__file__)))
    print(f"DIRECTION ANALYSIS: {order_checks} score-ordering checks passed", flush=True)


if __name__ == "__main__":
    main()
