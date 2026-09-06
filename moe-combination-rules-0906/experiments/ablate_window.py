#!/usr/bin/env python3
"""Development-side ablation: does the bounded window buy earliness?

The motivating claim was that a plain AND alarms at the later of its two
component alarms, which pushes the decision into the high-prior regime where it
is worth little. A bounded confirmation window is supposed to fix that. This
compares matched pairs of rules that differ only in W, on development only, so
the claim is testable independently of the external replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import combination_core as core


DEFAULT_OUTPUT = core.BUNDLE / "results"
MATCH = ["family", "mode", "pool", "level", "level_loose", "k", "frames"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rules = pd.read_csv(args.output / "development_rules.csv")
    rules = rules[rules["k"] >= 2]

    latching = rules[rules["window"] < 0].set_index(MATCH)
    rows = []
    for window in (0, 1, 2, 4, 8):
        block = rules[rules["window"] == window].set_index(MATCH)
        joined = block.join(latching, how="inner", rsuffix="_latching")
        both_fire = joined[(joined["tp"] > 0) & (joined["tp_latching"] > 0)]
        rows.append(
            {
                "window": window,
                "matched_pairs": int(len(both_fire)),
                "median_mean_prior_windowed": float(both_fire["mean_alarm_prior"].median()),
                "median_mean_prior_latching": float(
                    both_fire["mean_alarm_prior_latching"].median()
                ),
                "median_prior_reduction": float(
                    (both_fire["mean_alarm_prior_latching"] - both_fire["mean_alarm_prior"]).median()
                ),
                "share_with_lower_prior": float(
                    (both_fire["mean_alarm_prior"] < both_fire["mean_alarm_prior_latching"]).mean()
                ),
                "median_tp_ratio": float(
                    (both_fire["tp"] / both_fire["tp_latching"]).median()
                ),
                "median_low_prior_tp_delta": float(
                    (both_fire["low_prior_tp"] - both_fire["low_prior_tp_latching"]).median()
                ),
                "median_fp_ratio": float(
                    (both_fire["fp"] / both_fire["fp_latching"].replace(0, np.nan)).median()
                ),
                "share_with_more_low_prior_tp": float(
                    (both_fire["low_prior_tp"] > both_fire["low_prior_tp_latching"]).mean()
                ),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "development_window_ablation.csv", index=False)

    # Pareto question: at a matched development false alarm budget, does any
    # windowed rule beat every latching rule?
    pareto_rows = []
    for mode in core.MODES:
        for cap in core.FPR_CAPS:
            block = rules[
                (rules["mode"] == mode)
                & (rules["timely_fpr"] <= cap)
                & (rules["low_prior_precision"] >= core.MIN_LOW_PRIOR_PRECISION)
            ]
            if block.empty:
                continue
            for name, subset in (
                ("latching_only", block[block["window"] < 0]),
                ("windowed_only", block[block["window"] >= 0]),
            ):
                if subset.empty:
                    continue
                pareto_rows.append(
                    {
                        "mode": mode,
                        "cap": cap,
                        "family": name,
                        "best_tp": int(subset["tp"].max()),
                        "best_low_prior_tp": int(subset["low_prior_tp"].max()),
                        "lowest_mean_alarm_prior": float(subset["mean_alarm_prior"].min()),
                        "rules": int(len(subset)),
                    }
                )
    pareto = pd.DataFrame(pareto_rows)
    pareto.to_csv(args.output / "development_window_pareto.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== development: windowed vs latching, matched on everything but W ===")
    print(table.to_string(index=False, float_format="%.4f"))
    print("\n=== development: best achievable under each cap, by window family ===")
    print(pareto.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
