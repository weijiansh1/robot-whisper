#!/usr/bin/env python3
"""Consolidated operating-point table: every detector and bundle this bundle
reports, scored the same way, with per-physical-mode recall attached.

All scores use the project's standard decomposition: TP / FP / precision /
recall / lift against the matched survival prior / early (low-prior) TP and FP.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"


def or_alarm(arrays):
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        a = np.asarray(a, int)
        fired = a >= 0
        out[fired] = np.minimum(out[fired], a[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


ENTRIES = {
    # published operating points, recomputed here
    "single|mobility|global": (["mobility|global"], "global", "published best single global"),
    "single|expert_load_effective_rank|global": (["expert_load_effective_rank|global"],
                                                 "global", "frozen frame-survey head"),
    "single|expert_load_effective_rank|per_task": (["expert_load_effective_rank|per_task"],
                                                   "per_task", "published best single per_task"),
    "single|v7_guard": (["v7_guard|task_agnostic"], "task_agnostic",
                        "published v7 guard (itself a 4-head bundle)"),
    # bundles selected on development in this study
    "bundle|global_prereg_16fp": (["expert_load_effective_rank|global", "flow_path|global"],
                                  "global", "pre-registered greedy, dev FP<=16"),
    "bundle|global_frontier_dev32": (["expert_load_effective_rank|global", "mobility|global",
                                      "flow_path|global", "conditional_energy|global"],
                                     "global", "greedy at dev FP budget 32 (frontier point)"),
    "bundle|global_frontier_dev52": (["expert_load_effective_rank|global", "mobility|global",
                                      "conditional_query_d1|global", "flow_path|global"],
                                     "global", "greedy at dev FP budget 52 (frontier point)"),
    "bundle|per_task_prereg_89fp": (["expert_load_effective_rank|per_task", "mobility|per_task",
                                     "action_consensus|per_task"], "per_task",
                                    "pre-registered greedy, dev FP<=88"),
    "bundle|v7_plus_global_prereg": (["v7_guard|task_agnostic", "flow_endpoint|global",
                                      "conditional_energy|global", "flow_path|global"],
                                     "task_agnostic", "pre-registered greedy, dev FP<=76"),
    # the interpretable two-head (POST-HOC)
    "twohead|per_task_plus_global": (["expert_load_effective_rank|per_task", "mobility|global"],
                                     "mixed", "POST-HOC: task-relative head OR absolute head"),
    "twohead|per_task_plus_global_cq": (["expert_load_effective_rank|per_task",
                                         "conditional_query_d1|global"], "mixed",
                                        "POST-HOC: the other predicted grasp head"),
    "twohead|v7_plus_global_grasp": (["v7_guard|task_agnostic", "mobility|global"],
                                     "task_agnostic", "POST-HOC: v7 guard OR the grasp head"),
}


def main() -> None:
    dev_alarms = C.load_npz(OUT / "first_alarms_development.npz"); dev_alarms.pop("schema")
    ext_alarms = C.load_npz(OUT / "first_alarms_external.npz"); ext_alarms.pop("schema")
    dev_frame = C.cohort_index("development_main"); ext_frame = C.cohort_index("external_8b")
    dev_risk = dev_frame.risk.to_numpy(bool); ext_risk = ext_frame.risk.to_numpy(bool)
    dev_suite = dev_frame.suite.to_numpy(str); ext_suite = ext_frame.suite.to_numpy(str)
    dev_priors = C.survival_prior(dev_suite, dev_frame.length.to_numpy(int), dev_risk)
    ext_priors = C.survival_prior(ext_suite, ext_frame.length.to_numpy(int), ext_risk)
    mode = ext_frame.primary_failure_reason.fillna("").to_numpy(str)
    mode_counts = pd.Series(mode[ext_risk]).value_counts()

    rows = []
    for label, (heads, family, note) in ENTRIES.items():
        d = or_alarm([dev_alarms[k] for k in heads])
        e = or_alarm([ext_alarms[k] for k in heads])
        ds = C.score_candidate(np.asarray(d, int), dev_risk,
                               C.prior_of(np.asarray(d, int), dev_suite, dev_priors))
        es = C.score_candidate(np.asarray(e, int), ext_risk,
                               C.prior_of(np.asarray(e, int), ext_suite, ext_priors))
        row = {"label": label, "family": family, "heads": " OR ".join(heads),
               "n_heads": len(heads), "note": note,
               "dev_tp": ds["tp"], "dev_fp": ds["fp"],
               "ext_tp": es["tp"], "ext_fp": es["fp"],
               "ext_precision": es["precision"], "ext_recall": es["risk_recall"],
               "ext_lift": es["lift"], "ext_mean_alarm_prior": es["mean_alarm_prior"],
               "ext_early_tp": es["low_prior_tp"], "ext_early_fp": es["low_prior_fp"],
               "ext_timely_fpr": es["timely_fpr"]}
        for m, n in mode_counts.items():
            take = (mode == m) & ext_risk
            row[f"recall::{C.MODE_SHORT.get(m, m)}(n={n})"] = float(
                ((e >= 0) & take).sum() / n)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "headline_operating_points.csv", index=False)
    cols = ["label", "n_heads", "dev_tp", "dev_fp", "ext_tp", "ext_fp", "ext_precision",
            "ext_recall", "ext_lift", "ext_early_tp", "ext_early_fp"]
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    recall_cols = [c for c in frame.columns if c.startswith("recall::")]
    print(frame[["label"] + recall_cols].to_string(index=False,
                                                   float_format=lambda x: f"{x:.3f}"))
    (OUT / "headline_operating_points.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2, default=float) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
