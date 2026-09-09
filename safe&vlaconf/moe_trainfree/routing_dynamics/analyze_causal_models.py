"""Post-result mechanism checks; no model, threshold, or parameter selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_models import COMPONENTS, METHODS, PRIMARY, reference_evidence
from guard import ALPHAS, KINDS, PeakBank, first_trigger
from model_tables import combined_alarm
from run_causal_models import BASE, inputs
from run_guard_experiment import ACTION_CAPS, S05
from run_analysis import bootstrap_indices, digest, write_json
from monitor import union


def paired_intervals(b, saved, out):
    rows = []
    for kind in KINDS:
        for method in METHODS:
            new = combined_alarm(saved, kind, .01, method) >= 0
            for baseline in ("old_ac", "ac_reference", "v82_frozen"):
                if baseline == method:
                    continue
                old = combined_alarm(saved, kind, .01, baseline) >= 0
                for scope, selected in (("all", b), ("excluding_S05", b.loc[b.task.ne(S05)])):
                    counts, tasks, suites = [], [], []
                    for task, part in selected.groupby("task"):
                        ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
                        delta = new[ix].astype(int) - old[ix].astype(int)
                        counts.append([delta[y].sum(), y.sum(), delta[~y].sum(), (~y).sum()])
                        tasks.append(task)
                        suites.append(part.suite.iloc[0])
                    values = np.asarray(counts)
                    draws = bootstrap_indices(tasks, suites)
                    totals = values[draws].sum(1)
                    for metric, num, den in (("recall", 0, 1), ("fpr", 2, 3)):
                        ci = np.nanquantile(totals[:, num] / totals[:, den], [.025, .975])
                        rows.append(dict(calibration=kind, alpha=.01, method=method, baseline=baseline, scope=scope,
                                         metric=metric, delta=values[:, num].sum()/values[:, den].sum(), lo=ci[0], hi=ci[1],
                                         bootstrap="2000 task clusters stratified by suite"))
    pd.DataFrame(rows).to_csv(out / "paired_intervals.csv", index=False)


def evidence_and_calibration(b, valid, components, profiles, saved, out):
    evidence = {name: np.full_like(saved["scores"][0], np.nan) for name in COMPONENTS}
    calibration, fixed_threshold = [], []
    control_i = METHODS.index("ac_reference")
    for profile in profiles:
        ix = np.flatnonzero(b.init_state_id.to_numpy() % 5 == profile["fold"])
        global_rows = b.iloc[ix].global_row.to_numpy()
        refs = {name: PeakBank.from_list(bank) for name, bank in profile["references"].items()}
        current = reference_evidence({name: value[global_rows] for name, value in components.items()}, refs)
        for name, values in current.items():
            evidence[name][ix] = values
        with np.load(out / f"fold_{profile['fold']}_calibration.npz", allow_pickle=False) as z:
            cal_scores = z["scores"][:, z["success"]]
        cal_peaks = np.where(np.isfinite(cal_scores), cal_scores, -np.inf).max(-1)
        for kind in KINDS:
            reference_bank = PeakBank.from_list(profile["banks"][kind]["ac_reference"])
            reference_threshold, _ = reference_bank.threshold(.01)
            for name in ("absolute_acc", "absolute_both"):
                mi = METHODS.index(name)
                bank = PeakBank.from_list(profile["banks"][kind][name])
                threshold, _ = bank.threshold(.01)
                exposed = np.isfinite(cal_peaks[control_i])
                calibration.append(dict(fold=profile["fold"], calibration=kind, method=name,
                                        control_threshold=reference_threshold, threshold=threshold,
                                        success_episodes=len(exposed), exposed_success_episodes=int(exposed.sum()),
                                        success_peaks_reduced=int((cal_peaks[mi] < cal_peaks[control_i]).sum())))
                first = first_trigger(reference_bank.tail(saved["scores"][mi, ix]), valid[global_rows], .01)
                for i, q in zip(ix, first):
                    fixed_threshold.append(dict(calibration=kind, method=name, row=int(i), first=int(q)))
    pd.DataFrame(calibration).to_csv(out / "absolute_calibration_effect.csv", index=False)
    rows = []
    counterfactual = pd.DataFrame(fixed_threshold)
    for (kind, method), part in counterfactual.groupby(["calibration", "method"]):
        first = part.sort_values("row")["first"].to_numpy()
        combined = union(saved["frozen_first"], first)
        control = combined_alarm(saved, kind, .01, "ac_reference")
        y = b.failure.to_numpy(bool)
        assert not ((combined >= 0) & (control < 0)).any()
        rows.append(dict(calibration=kind, method=method, use_control_threshold=True,
                         tp=int(((combined >= 0) & y).sum()), fp=int(((combined >= 0) & ~y).sum()),
                         gained_tp=0, lost_tp=int(((control >= 0) & (combined < 0) & y).sum()),
                         removed_fp=int(((control >= 0) & (combined < 0) & ~y).sum())))
    pd.DataFrame(rows).to_csv(out / "fixed_threshold_diagnostic.csv", index=False)
    return evidence


def case_mechanisms(b, evidence, profiles, saved, out):
    kind, alpha = "task_init", .01
    ki = KINDS.index(kind)
    old = combined_alarm(saved, kind, alpha, "old_ac")
    reference = combined_alarm(saved, kind, alpha, "ac_reference")
    records = []
    for method in METHODS:
        first = combined_alarm(saved, kind, alpha, method)
        for ix in np.flatnonzero((first >= 0) != (old >= 0)):
            row = b.iloc[ix]
            profile = profiles[int(row.init_state_id % 5)]
            q = int(first[ix] if first[ix] >= 0 else old[ix])
            values = {name: evidence[name][ix, q] for name in COMPONENTS}
            abs_branch = min(values["acceleration"], values["absolute_acceleration"])
            d = values["decoupling"]
            path = "absolute_acceleration" if not np.isfinite(d) or abs_branch > d else ("decoupling" if d > abs_branch else "tie")
            records.append(dict(method=method, global_row=row.global_row, task=row.task, suite=row.suite,
                                episode=row.episode, failure=row.failure, change="added" if first[ix] >= 0 else "removed",
                                frozen_first=int(saved["frozen_first"][ix]), old_ac_first=int(old[ix]),
                                reference_first=int(reference[ix]), new_first=int(first[ix]), query=q,
                                configured_cap_percent=1000*q/ACTION_CAPS[row.suite],
                                absolute_model_winning_path=path,
                                model_score=saved["scores"][METHODS.index(method), ix, q],
                                reference_score=saved["scores"][METHODS.index("ac_reference"), ix, q],
                                model_tail=saved["tails"][ki, METHODS.index(method), ix, q],
                                reference_tail=saved["tails"][ki, METHODS.index("ac_reference"), ix, q],
                                threshold=PeakBank.from_list(profile["banks"][kind][method]).threshold(alpha)[0],
                                **{f"evidence_{name}": value for name, value in values.items()}))
    pd.DataFrame(records).to_csv(out / "changed_case_mechanisms.csv", index=False)
    comparisons = []
    for method, baseline in (("absolute_state_trap", PRIMARY), ("leaky_state_trap", "leaky_ac")):
        for kind in KINDS:
            a, o = combined_alarm(saved, kind, .01, method), combined_alarm(saved, kind, .01, baseline)
            for ix in np.flatnonzero((a >= 0) != (o >= 0)):
                row = b.iloc[ix]
                comparisons.append(dict(calibration=kind, method=method, baseline=baseline, global_row=row.global_row,
                                        task=row.task, episode=row.episode, failure=row.failure,
                                        old_first=int(o[ix]), new_first=int(a[ix])))
    pd.DataFrame(comparisons).to_csv(out / "state_increment_cases.csv", index=False)


def audit_shapes_and_ties(b, saved, out):
    s = {name: saved["scores"][mi] for mi, name in enumerate(METHODS)}
    order_checks = 0
    for low, high in (("absolute_acc", "ac_reference"), ("absolute_both", "absolute_acc"),
                      ("absolute_acc", "absolute_state_trap"), ("ac_reference", "joint_stop_or")):
        valid = np.isfinite(s[low]) & np.isfinite(s[high])
        assert (s[low][valid] <= s[high][valid]).all()
        order_checks += int(valid.sum())
    rows = []
    for method in ("leaky_ac", "leaky_state_trap"):
        for task, part in b.groupby("task"):
            ix, y = part.index.to_numpy(), part.failure.to_numpy(bool)
            for q in range(s[method].shape[1]):
                value = s[method][ix, q]
                good, bad = value[(~y) & np.isfinite(value)], value[y & np.isfinite(value)]
                if len(good) and len(bad):
                    rows.append(dict(method=method, task=task, query=q, successes=len(good), failures=len(bad),
                                     success_zero_fraction=float((good == 0).mean()), failure_zero_fraction=float((bad == 0).mean()),
                                     zero_cross_class_tie_fraction=float((good == 0).mean() * (bad == 0).mean())))
    pd.DataFrame(rows).to_csv(out / "leaky_zero_ties.csv", index=False)
    return order_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "causal_routing_models_20260909")
    out = parser.parse_args().output.resolve()
    assert json.loads((out / "independent_verification.json").read_text())["all_checks_passed"]
    frame, valid, components, _ = inputs()
    b = pd.read_csv(out / "test_index.csv")
    with np.load(out / "predictions.npz", allow_pickle=False) as z:
        saved = {name: z[name] for name in z.files}
    profiles = [json.loads((out / f"fold_{f}_profile.json").read_text()) for f in range(5)]
    paired_intervals(b, saved, out)
    evidence = evidence_and_calibration(b, valid, components, profiles, saved, out)
    case_mechanisms(b, evidence, profiles, saved, out)
    checks = audit_shapes_and_ties(b, saved, out)
    write_json(out / "mechanism_verification.json", dict(all_checks_passed=True, diagnostic_only=True,
               runtime_models_unchanged=True, pointwise_order_checks=checks, analyzer_sha256=digest(__file__)))
    print(f"MECHANISMS: {checks} pointwise ordering checks passed; no model changes", flush=True)


if __name__ == "__main__":
    main()
