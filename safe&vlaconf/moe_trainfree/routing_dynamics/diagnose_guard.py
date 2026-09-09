"""Post-hoc attribution and a ranking diagnostic that cannot change alarms."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from guard import ALPHAS, BRANCHES, KINDS, METHOD_BRANCHES, first_trigger
from run_guard_experiment import BASE, METHODS, auroc_tables
from run_analysis import digest, write_json


def main():
    output = BASE / "routing_guard_20260908"
    diagnostic = output / "unclipped_rank_diagnostic"
    diagnostic.mkdir(exist_ok=False)
    with np.load(output / "predictions.npz", allow_pickle=False) as z:
        original, branches, first = z["tails"], z["branch_tails"], z["first"]
        raw, valid = z["raw_v82_scores"], z["valid"]
    b = pd.read_csv(output / "test_index.csv")
    unadjusted = np.full_like(original, np.nan)
    attribution, clipping = [], []
    y = b.failure.to_numpy(bool)
    for ki, kind in enumerate(KINDS):
        for mi, method in enumerate(METHODS):
            names = METHOD_BRANCHES[method]
            current = np.stack([branches[ki, BRANCHES.index(name)] for name in names])
            unadjusted[ki, mi] = np.fmin.reduce(current)
            reconstructed = np.minimum(1., len(names) * unadjusted[ki, mi])
            np.testing.assert_array_equal(reconstructed, original[ki, mi])
            for ai, alpha in enumerate(ALPHAS):
                np.testing.assert_array_equal(first_trigger(unadjusted[ki, mi], valid, alpha / len(names)),
                                              first[ki, ai, mi])
            for name in names:
                tail = branches[ki, BRANCHES.index(name)]
                branch_first = first_trigger(tail, valid, .01 / len(names))
                integrated_first = first[ki, ALPHAS.index(.01), mi]
                fired = branch_first >= 0
                first_cause = fired & (branch_first == integrated_first)
                attribution.append(dict(calibration=kind, method=method, branch=name,
                                        branch_tp=int((fired & y).sum()), branch_fp=int((fired & ~y).sum()),
                                        first_cause_tp=int((first_cause & y).sum()),
                                        first_cause_fp=int((first_cause & ~y).sum())))
            for window, queries in (("q7_13", slice(7, 14)), ("all", slice(7, None))):
                finite = np.isfinite(original[ki, mi, :, queries])
                clipped = finite & (original[ki, mi, :, queries] == 1)
                clipping.append(dict(calibration=kind, method=method, window=window,
                                     observed_scores=int(finite.sum()), clipped_at_one=int(clipped.sum()),
                                     clipped_fraction=float(clipped.sum() / finite.sum())))
    pd.DataFrame(attribution).to_csv(output / "branch_attribution.csv", index=False)
    pd.DataFrame(clipping).to_csv(diagnostic / "clipping_counts.csv", index=False)
    auroc_tables(b, unadjusted, raw, diagnostic)
    summary = pd.read_csv(diagnostic / "auroc_summary.csv")
    selected = summary.loc[summary.scope.eq("all") & summary.metric.isin(["auc", "delta"])]
    print(selected[["calibration", "method", "window", "metric", "estimate", "lo", "hi"]].to_string(index=False))
    write_json(diagnostic / "verification.json", dict(
        posthoc_diagnostic=True, no_alarm_change=True, checks=2 * 4 * 3 * len(b),
        purpose="separate loss of rank resolution from the fixed alarm operating point",
        score="negative log of minimum branch tail; no clipping at combined tail=1",
        source_sha256=digest(__file__), prediction_sha256=digest(output / "predictions.npz"),
        artifacts={p.name: digest(p) for p in sorted(diagnostic.iterdir()) if p.is_file()}))


if __name__ == "__main__":
    main()
