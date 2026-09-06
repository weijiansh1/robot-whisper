#!/usr/bin/env python3
"""The non-circular test of the interpretability claim.

The task-stratified selectivity analysis says only two `global` heads are
positively selective for `stable_grasp_not_observed`: `mobility|global` and
`conditional_query_d1|global`. That statement was derived from the simulator
labels, never from detector agreement.

Prediction, made from the labels alone: if a per_task head is grasp-blind, then
adding one of those two heads -- and only those -- should raise its
`stable_grasp_not_observed` recall sharply.

Test: pair the best per_task head with every one of the 12 `global` heads in
turn and read off what happens. The ranking criterion here is outcome-only
(development TP / FP); the physical labels only supplied the prediction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"
BASE = "expert_load_effective_rank|per_task"
PREDICTED = ("mobility|global", "conditional_query_d1|global")


def or_alarm(arrays):
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        a = np.asarray(a, int)
        fired = a >= 0
        out[fired] = np.minimum(out[fired], a[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


def main() -> None:
    dev_alarms = C.load_npz(OUT / "first_alarms_development.npz"); dev_alarms.pop("schema")
    ext_alarms = C.load_npz(OUT / "first_alarms_external.npz"); ext_alarms.pop("schema")
    dev_frame = C.cohort_index("development_main"); ext_frame = C.cohort_index("external_8b")
    dev_risk = dev_frame.risk.to_numpy(bool); ext_risk = ext_frame.risk.to_numpy(bool)
    dev_suite = dev_frame.suite.to_numpy(str); ext_suite = ext_frame.suite.to_numpy(str)
    dev_priors = C.survival_prior(dev_suite, dev_frame.length.to_numpy(int), dev_risk)
    ext_priors = C.survival_prior(ext_suite, ext_frame.length.to_numpy(int), ext_risk)
    mode = ext_frame.primary_failure_reason.fillna("").to_numpy(str)
    counts = pd.Series(mode[ext_risk]).value_counts()

    rows = []
    for partner in [None] + sorted(k for k in ext_alarms if k.endswith("|global")):
        heads = [BASE] if partner is None else [BASE, partner]
        d = or_alarm([dev_alarms[k] for k in heads])
        e = or_alarm([ext_alarms[k] for k in heads])
        ds = C.score_candidate(np.asarray(d, int), dev_risk,
                               C.prior_of(np.asarray(d, int), dev_suite, dev_priors))
        es = C.score_candidate(np.asarray(e, int), ext_risk,
                               C.prior_of(np.asarray(e, int), ext_suite, ext_priors))
        row = {"partner": partner or "(none)",
               "predicted_grasp_head": partner in PREDICTED,
               "dev_tp": ds["tp"], "dev_fp": ds["fp"],
               "ext_tp": es["tp"], "ext_fp": es["fp"],
               "ext_precision": es["precision"], "ext_recall": es["risk_recall"],
               "ext_lift": es["lift"],
               "ext_early_tp": es["low_prior_tp"], "ext_early_fp": es["low_prior_fp"]}
        for m, n in counts.items():
            take = (mode == m) & ext_risk
            row[f"recall::{C.MODE_SHORT.get(m, m)}"] = float(((e >= 0) & take).sum() / n)
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "partner_control.csv", index=False)
    body = frame[frame.partner != "(none)"]
    predicted = body[body.predicted_grasp_head]
    other = body[~body.predicted_grasp_head]
    summary = {
        "schema": "himoe.failure_modes_0906.partner_control.v1",
        "base_head": BASE,
        "base_no_grasp_recall": float(frame.loc[frame.partner == "(none)",
                                                "recall::no_grasp"].iloc[0]),
        "predicted_from_physical_labels": list(PREDICTED),
        "predicted_no_grasp_recall": predicted["recall::no_grasp"].tolist(),
        "other_no_grasp_recall_max": float(other["recall::no_grasp"].max()),
        "other_no_grasp_recall_median": float(other["recall::no_grasp"].median()),
        "prediction_holds": bool(predicted["recall::no_grasp"].min()
                                 > other["recall::no_grasp"].max()),
        "predicted_are_also_top_two_by_development_tp": bool(
            set(body.sort_values("dev_tp", ascending=False).head(2).partner) == set(PREDICTED)),
    }
    (OUT / "partner_control.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cols = ["partner", "predicted_grasp_head", "dev_tp", "dev_fp", "ext_tp", "ext_fp",
            "ext_precision", "recall::no_grasp", "recall::dropped",
            "ext_early_tp", "ext_early_fp"]
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
