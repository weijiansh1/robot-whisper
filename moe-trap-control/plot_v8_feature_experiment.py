#!/usr/bin/env python3
"""Export task outcomes separately from feature-manipulation measurements."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(args):
    data = json.loads(args.analysis.read_text())
    arms = list(data["primary"])
    label = {"native": "Native", "balance": "Layer balance", "curvature": "Curvature", "mobility": "Mobility",
        "combined_half": "Combined 0.5", "combined": "Combined 1.0", "random_half": "Random 0.5", "random": "Random 1.0"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    x = np.arange(len(arms))
    rows = [data["primary"][arm] for arm in arms]
    axes[0, 0].barh(x - .18, [r["paired_wins"] for r in rows], .35, label="Wins vs paired native", color="#167d75")
    axes[0, 0].barh(x + .18, [-r["paired_losses"] for r in rows], .35, label="Losses vs paired native", color="#c64747")
    axes[0, 0].set_yticks(x, [label[arm] for arm in arms])
    axes[0, 0].invert_yaxis()
    axes[0, 0].axvline(0, c="#aaaaaa", lw=.8)
    axes[0, 0].set_xlim(-max(1, max(r["paired_losses"] for r in rows)) * 1.4,
                       max(1, max(r["paired_wins"] for r in rows)) * 1.4)
    for i, row in enumerate(rows):
        if row["paired_wins"] == row["paired_losses"] == 0:
            axes[0, 0].text(.06, i, "0 / 0", va="center", fontsize=9, color="#555555")
    axes[0, 0].set_title("Full-suffix task outcomes")
    axes[0, 0].set_xlabel("Paired suffixes (2 repeats per parent)")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    controlled = arms[1:]
    values = np.asarray([data["primary"][arm]["manipulation_branch_median"] for arm in controlled])
    colors = ["#347ea1"] * 3 + ["#167d75"] * 2 + ["#8e8e8e"] * 2
    for ax, column, title, baseline, ylabel in (
        (axes[0, 1], 1, "Three-step curvature during control", 1, "Ratio to same-observation native shadow"),
        (axes[1, 0], 2, "Cross-query mobility during control", 1, "Ratio to same-observation native shadow"),
        (axes[1, 1], 4, "Action change during control", 0, "RMS in 7-D policy action output")):
        ax.bar(np.arange(len(controlled)), values[:, column], color=colors)
        ax.axhline(baseline, c="#444444", ls="--", lw=1)
        ax.set_xticks(np.arange(len(controlled)), [label[arm] for arm in controlled], rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    figure.suptitle("V8 feature intervention: mechanisms and task outcomes\nLong on LIBERO-Pro / LIBERO-Plus, frozen alarms, batch 1", fontsize=14)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
