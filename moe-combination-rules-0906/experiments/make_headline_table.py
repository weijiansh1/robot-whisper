#!/usr/bin/env python3
"""Compact head-to-head table behind every number quoted in REPORT_ZH.md.

Reads only artifacts that already exist; computes nothing new about which rule
to prefer. For each frozen rule it also reports where the plain-OR sensitivity
curve of the same pool sits at the same false alarm count, by linear
interpolation, so "above the OR curve" is a checkable statement rather than an
impression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combination_core as core


DEFAULT_OUTPUT = core.BUNDLE / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def interpolate(curve: pd.DataFrame, fp: float, column: str) -> float:
    ordered = curve.sort_values("fp")
    return float(np.interp(fp, ordered["fp"].to_numpy(), ordered[column].to_numpy()))


def main() -> None:
    args = parse_args()
    shortlist = pd.read_csv(args.output / "external_shortlist.csv")
    references = pd.read_csv(args.output / "reference_operating_points.csv")
    context = pd.read_csv(args.output / "union_coverage_context.csv")
    external_reference = references[references["cohort"] == "external_8b"]

    rows = []
    for _, row in shortlist.iterrows():
        curve = context[
            (context["cohort"] == "external_8b")
            & (context["mode"] == row["mode"])
            & (context["pool"] == row["pool"])
        ]
        rows.append(
            {
                "label": row["selected_by"],
                "kind": "frozen_rule",
                "mode": row["mode"],
                "structure": (
                    f"{row['family']}/{row['pool']} level={row['level']}"
                    + (
                        f"->{row['level_loose']}"
                        if row["family"] == "cascade"
                        else ""
                    )
                    + f" k={row['k']} W={'inf' if row['window'] < 0 else row['window']}"
                    f" f={row['frames']}"
                ),
                "dev_tp": row["dev_tp"],
                "dev_fp": row["dev_fp"],
                "tp": row["tp"],
                "fp": row["fp"],
                "precision": row["precision"],
                "risk_recall": row["risk_recall"],
                "timely_fpr": row["timely_fpr"],
                "early_tp": row["low_prior_tp"],
                "early_fp": row["low_prior_fp"],
                "early_precision": row["low_prior_precision"],
                "mean_alarm_prior": row["mean_alarm_prior"],
                "lift": row["lift"],
                "or_curve_tp_at_same_fp": interpolate(curve, row["fp"], "tp"),
                "or_curve_early_tp_at_same_early_fp": interpolate(
                    curve, row["low_prior_fp"], "low_prior_tp"
                ),
            }
        )
    for _, row in external_reference.iterrows():
        curve = context[
            (context["cohort"] == "external_8b")
            & (context["mode"] == row["mode"])
            & (context["pool"] == "all12")
        ]
        rows.append(
            {
                "label": f"{row['block']}:{row['detector']}",
                "kind": row["block"],
                "mode": row["mode"],
                "structure": row["detector"],
                "dev_tp": np.nan,
                "dev_fp": np.nan,
                "tp": row["tp"],
                "fp": row["fp"],
                "precision": row["precision"],
                "risk_recall": row["risk_recall"],
                "timely_fpr": row["timely_fpr"],
                "early_tp": row["low_prior_tp"],
                "early_fp": row["low_prior_fp"],
                "early_precision": row["low_prior_precision"],
                "mean_alarm_prior": row["mean_alarm_prior"],
                "lift": row["lift"],
                "or_curve_tp_at_same_fp": interpolate(curve, row["fp"], "tp"),
                "or_curve_early_tp_at_same_early_fp": interpolate(
                    curve, row["low_prior_fp"], "low_prior_tp"
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["tp_above_or_curve"] = table["tp"] - table["or_curve_tp_at_same_fp"]
    table["early_tp_above_or_curve"] = (
        table["early_tp"] - table["or_curve_early_tp_at_same_early_fp"]
    )
    table.to_csv(args.output / "headline_comparison.csv", index=False)

    pd.set_option("display.width", 250)
    for mode in core.MODES:
        block = table[table["mode"] == mode]
        keep = block[
            (block["kind"] == "frozen_rule")
            | (block["kind"] == "reference_single")
            | (
                (block["kind"] == "reference_pairwise_and")
                & (block["tp"] >= block["tp"].where(block["kind"] == "reference_pairwise_and").max() * 0.5)
            )
        ]
        print(f"\n=== {mode} head to head (external) ===")
        print(
            keep.sort_values("fp")[
                [
                    "label", "kind", "tp", "fp", "precision", "risk_recall",
                    "early_tp", "early_fp", "mean_alarm_prior", "lift",
                    "tp_above_or_curve", "early_tp_above_or_curve",
                ]
            ].to_string(index=False, float_format="%.3f")
        )

    print("\n=== external plain-OR sensitivity curves (all12) ===")
    print(
        context[(context["cohort"] == "external_8b") & (context["pool"] == "all12")][
            ["mode", "level", "tp", "fp", "risk_recall", "timely_fpr",
             "low_prior_tp", "low_prior_fp", "precision", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )

    summary = {
        "schema": "himoe.combination_rules.headline.v1",
        "external_risks": 564,
        "external_timely": 15036,
        "reference_best_global_single": "mobility|global|selected: 195 TP / 17 FP",
        "reference_best_global_and": "conditional_query_d1+mobility: 140 TP / 9 FP",
        "reference_best_per_task_single": "expert_load_effective_rank: 370 TP / 93 FP",
        "reference_best_per_task_and": "expert_load_effective_rank+mobility: 245 TP / 14 FP",
        "artifact": "headline_comparison.csv",
    }
    (args.output / "headline.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
