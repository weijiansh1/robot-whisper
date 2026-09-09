#!/usr/bin/env python3
"""Three panels: where the information is, what it buys, and why late alarms are cheap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import common as C  # noqa: E402

SUITES = ("libero_spatial", "libero_goal", "libero_object", "libero_long")
COLOURS = {
    "libero_spatial": "#2E6F9E",
    "libero_goal": "#C4682B",
    "libero_object": "#3F8F5B",
    "libero_long": "#8A4B8A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = pd.read_csv(args.output / "information_profile_by_chunk.csv")
    frontier = pd.read_csv(args.output / "deadline_frontier.csv")
    survival = pd.read_csv(args.output / "survival_baseline.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ax = axes[0]
    for suite in SUITES:
        d = profile[(profile["cohort"] == "external_8b") & (profile["suite"] == suite)]
        d = d.sort_values("phase")
        ax.plot(d["phase"], d["max_abs_effect"], "-o", ms=3, color=COLOURS[suite],
                label=suite.replace("libero_", ""))
        ax.plot(d["phase"], d["control_max_abs_effect"], ":", lw=1, color=COLOURS[suite])
    ax.axvline(0.65, color="k", ls="--", lw=1)
    ax.set_xlabel("alarm phase  (q+1)/cap")
    ax.set_ylabel(r"max |AUC $-$ 0.5|, task-stratified")
    ax.set_title("a  information, external_8b\nsolid: routing   dotted: null control")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    for suite in SUITES:
        d = frontier[(frontier["cohort"] == "external_8b") & (frontier["suite"] == suite)]
        d = d.sort_values("phase")
        ax.plot(d["phase"], d["frontier_routing"], "-", color=COLOURS[suite],
                label=suite.replace("libero_", ""))
        ax.plot(d["phase"], d["frontier_null_control"], ":", lw=1, color=COLOURS[suite])
    ax.axvline(0.65, color="k", ls="--", lw=1)
    ax.axhline(0.80, color="crimson", ls="-.", lw=1)
    ax.set_xlabel("alarm deadline, phase")
    ax.set_ylabel("recall at timely FPR $\\leq$ 0.005")
    ax.set_title("b  best single-chunk rule up to the deadline\nred: the 80% target")
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    ax = axes[2]
    for suite in SUITES:
        d = survival[(survival["cohort"] == "external_8b") & (survival["suite"] == suite)]
        d = d.sort_values("phase")
        ax.semilogy(d["phase"], d["fpr_if_all_survivors_alarm"].clip(lower=1e-4),
                    color=COLOURS[suite], label=suite.replace("libero_", ""))
    ax.axhline(0.005, color="crimson", ls="-.", lw=1)
    ax.axvline(0.65, color="k", ls="--", lw=1)
    ax.set_xlabel("alarm phase")
    ax.set_ylabel("timely FPR of 'alarm on every survivor'")
    ax.set_title("c  the length rule's price\nred: the 0.005 budget")
    ax.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig(args.output / "early_window_summary.png", dpi=170)
    print("wrote", args.output / "early_window_summary.png")


if __name__ == "__main__":
    main()
