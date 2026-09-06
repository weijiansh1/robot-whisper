#!/usr/bin/env python3
"""Threshold-free information content, plus uncertainty on the detector table.

Two additions to the alarm table:

1. Survival-conditioned AUC.  At every chunk q, among the episodes of one suite
   that are *still running* at q, how well does the quantity separate risk from
   non-risk?  This is exactly the information a detector could exploit above
   the survival prior, with no threshold and no calibration choice.  Strata are
   pooled with Mann-Whitney weights (n_pos * n_neg), which makes the pooled
   number a stratified concordance index.  0.5 means nothing above the prior.

2. Wilson 95% intervals on precision, propagated to lift by dividing by the
   mean matched prior (treated as fixed).  Lift intervals that contain 1 mean
   the detector is not distinguishable from "still running".
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import protocol as P


DEFAULT_OUTPUT = P.BUNDLE / "results/detectors"
MIN_STRATUM = 20  # minimum positives and negatives in a (suite, chunk) stratum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def wilson(successes: int, total: int, z: float = 1.959963985) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def auc(values: np.ndarray, positive: np.ndarray) -> float:
    ranks = stats.rankdata(values)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def survival_auc(dense: np.ndarray, valid: np.ndarray, suite: np.ndarray,
                 risk: np.ndarray, early_cut: dict[str, int]) -> tuple[float, float]:
    """Return (pooled AUC over all chunks, pooled AUC over early chunks only)."""
    total, weight = 0.0, 0.0
    early_total, early_weight = 0.0, 0.0
    for name in np.unique(suite):
        take = suite == name
        cut = early_cut.get(str(name))
        for query in range(valid.shape[1]):
            running = take & valid[:, query]
            if not running.any():
                continue
            positive = risk[running]
            n_pos = int(positive.sum())
            n_neg = int(len(positive) - n_pos)
            if n_pos < MIN_STRATUM or n_neg < MIN_STRATUM:
                continue
            column = dense[running, query]
            if not np.isfinite(column).all():
                continue
            value = auc(column, positive)
            weight_here = n_pos * n_neg
            total += value * weight_here
            weight += weight_here
            if cut is not None and query < cut:
                early_total += value * weight_here
                early_weight += weight_here
    return (total / weight if weight else float("nan"),
            early_total / early_weight if early_weight else float("nan"))


def main() -> None:
    args = parse_args()
    cohorts = {name: P.load_cohort(name) for name in ("development_main", "external_8b")}
    quantities = P.quantity_list(cohorts["external_8b"])

    context = {}
    for name, cohort in cohorts.items():
        circuit = cohort["circuit"]
        labels = pd.read_csv(P.LABEL_PATHS[name])
        risk = labels["original_failure"].to_numpy(bool)
        suite = P.suite_of(circuit)
        priors = P.survival_prior(suite, circuit["length"].astype(int), risk)
        context[name] = {
            "risk": risk, "suite": suite, "valid": circuit["valid"].astype(bool),
            "early_cut": {s: next((q for q, p in sorted(t.items()) if p >= P.LOW_PRIOR), None)
                          for s, t in priors.items()},
        }

    rows = []
    for family, quantity, index in quantities:
        for cohort_name, cohort in cohorts.items():
            dense = P.dense_values(cohort, family, index)
            ctx = context[cohort_name]
            for position, layer in enumerate(cohort["circuit"]["layer_names"].astype(str)):
                overall, early = survival_auc(
                    dense[:, :, position], ctx["valid"], ctx["suite"], ctx["risk"],
                    ctx["early_cut"],
                )
                rows.append({
                    "family": family, "quantity": quantity, "cohort": cohort_name,
                    "layer": layer, "survival_auc": overall,
                    "survival_auc_early": early,
                    "abs_auc_gap": abs(overall - 0.5),
                })
        print(f"  auc {family}/{quantity}", flush=True)
    auc_frame = pd.DataFrame(rows)
    auc_frame.to_csv(args.output / "survival_conditioned_auc.csv", index=False)

    detectors = pd.read_csv(args.output / "external_detectors.csv")
    lower, upper, lift_lo, lift_hi = [], [], [], []
    for _, record in detectors.iterrows():
        fired = int(record["tp"] + record["fp"])
        low, high = wilson(int(record["tp"]), fired)
        lower.append(low)
        upper.append(high)
        lift_lo.append(low / record["mean_alarm_prior"])
        lift_hi.append(high / record["mean_alarm_prior"])
    detectors["precision_lo95"] = lower
    detectors["precision_hi95"] = upper
    detectors["lift_lo95"] = lift_lo
    detectors["lift_hi95"] = lift_hi
    detectors["lift_significant"] = detectors["lift_lo95"] > 1.0
    detectors.to_csv(args.output / "external_detectors_with_ci.csv", index=False)

    best = (auc_frame[auc_frame["cohort"] == "external_8b"]
            .sort_values("abs_auc_gap", ascending=False))
    summary = {
        "schema": "himoe.circuit_analogy.summary.v1",
        "min_stratum_positives_and_negatives": MIN_STRATUM,
        "auc_top_overall": best.head(12)[
            ["family", "quantity", "layer", "survival_auc", "survival_auc_early"]
        ].to_dict("records"),
        "auc_top_circuit": best[best["family"] == "circuit"].head(10)[
            ["quantity", "layer", "survival_auc", "survival_auc_early"]
        ].to_dict("records"),
        "auc_top_published": best[best["family"] == "published"].head(10)[
            ["quantity", "layer", "survival_auc", "survival_auc_early"]
        ].to_dict("records"),
        "external_lift_significant_count": {
            mode: int(detectors[(detectors["mode"] == mode)]["lift_significant"].sum())
            for mode in ("global", "per_task")
        },
        "external_lift_significant_circuit_global": detectors[
            (detectors["mode"] == "global") & (detectors["family"] == "circuit")
            & detectors["lift_significant"]
        ][["quantity", "tp", "fp", "precision", "lift", "lift_lo95", "lift_hi95",
           "low_prior_tp"]].to_dict("records"),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
