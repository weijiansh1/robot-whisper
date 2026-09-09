"""Exploratory task/time/motion matched pairs using a fixed percentile caliper."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from monitor import HERE, METHODS
from run_analysis import RUN_B, archive, bootstrap_indices, digest, write_json


def main():
    output = HERE.parent / "results/v82_validation_20260908"
    frame = pd.read_csv(output / "index.csv")
    prediction = archive(output / "crossfit_predictions.npz")
    b_rows = prediction["global_rows"]
    inverse = np.full(len(frame), -1, int)
    inverse[b_rows] = np.arange(len(b_rows))
    motion = np.full((len(frame), 52), np.nan, np.float32)
    audits = json.loads((HERE.parent / "results/round3_safe/extraction_audit.json").read_text())
    for audit in audits:
        if not audit["source"].endswith(RUN_B):
            continue
        rows = frame.index[frame.source.eq(audit["source"])].to_numpy()
        with np.load(HERE.parent / "results/round3_safe/features" / audit["output"]) as z:
            motion[rows] = z["direct"][:, :, 5]
    pairs = []
    for task, part in frame.loc[frame.run_id.eq(RUN_B)].groupby("task", sort=True):
        for q in range(7, 14):
            active = part.loc[(part.length > q) & np.isfinite(motion[part.index, q])]
            ix = active.index.to_numpy()
            failure = active.failure.to_numpy(bool)
            if not failure.any() or failure.all():
                continue
            ranks = rankdata(motion[ix, q]) / len(ix)
            negative = np.flatnonzero(~failure)
            for case in np.flatnonzero(failure):
                distance = np.abs(ranks[negative] - ranks[case])
                match = negative[int(np.argmin(distance))]
                good = float(distance.min()) <= .02
                record = dict(task=task, suite=active.suite.iloc[0], query=q, case_row=int(ix[case]),
                              control_row=int(ix[match]) if good else -1,
                              percentile_gap=float(distance.min()),
                              motion_difference=float(motion[ix[case], q] - motion[ix[match], q]) if good else np.nan)
                for method in ("v7", "v82", "eef_motion_low"):
                    a, b = prediction["scores"][METHODS.index(method), inverse[ix[[case, match]]], q]
                    record[method + "_win"] = float(a > b) + .5 * float(a == b) if good and np.isfinite([a, b]).all() else np.nan
                pairs.append(record)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(output / "motion_matched_pairs.csv", index=False)
    task = pairs.groupby(["suite", "task", "query"], as_index=False)[[m + "_win" for m in ("v7", "v82", "eef_motion_low")]].mean()
    task = task.groupby(["suite", "task"], as_index=False)[[m + "_win" for m in ("v7", "v82", "eef_motion_low")]].mean()
    records = []
    for scope, current in [("all", task)] + list(task.groupby("suite", sort=True)):
        for method in ("v7", "v82", "eef_motion_low"):
            part = current.loc[current[method + "_win"].notna()]
            draws = bootstrap_indices(part.task.to_numpy(), part.suite.to_numpy())
            values = part[method + "_win"].to_numpy()
            lo, hi = np.quantile(values[draws].mean(axis=1), [.025, .975])
            records.append(dict(scope=scope, method=method, win_rate=float(values.mean()), lo=lo, hi=hi, tasks=len(part)))
    pd.DataFrame(records).to_csv(output / "motion_matched_summary.csv", index=False)
    selected = pairs.loc[pairs.control_row >= 0]
    write_json(output / "motion_match_verification.json", dict(protocol_sha256=digest(HERE / "MOTION_MATCH_ADDENDUM_ZH.md"),
                code_sha256=digest(__file__), exploratory_after_primary_results=True, percentile_caliper=.02,
                eligible_case_queries=len(pairs), matched_case_queries=len(selected),
                unique_case_episodes=int(selected.case_row.nunique()), unique_control_episodes=int(selected.control_row.nunique()),
                median_percentile_gap=float(selected.percentile_gap.median()),
                median_absolute_motion_difference=float(selected.motion_difference.abs().median())))
    print(pd.DataFrame(records).loc[lambda x: x.scope.eq("all")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
