#!/usr/bin/env python3
"""Render the event-level summary for the causal online multi-head alarm."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/online_multihead_hub"
EXTERNAL = HERE / "results/online_multihead_hub_external"
OUTPUT = HERE / "figures/online_multihead_alarm"

COLORS = {
    "instability": "#d95f02",
    "lock_in": "#1b9e77",
    "flat_narrow_support": "#7570b3",
    "feedback_decoupling": "#e7298a",
    "dual_mean": "#66a61e",
    "multi_max": "#1f78b4",
    "instant_multi_max": "#a6761d",
    "clock": "#333333",
}

LABELS = {
    "instability": "Instability",
    "lock_in": "Lock-in",
    "flat_narrow_support": "Flat/narrow",
    "feedback_decoupling": "Feedback split",
    "dual_mean": "Dual mean",
    "multi_max": "Four-head max",
    "instant_multi_max": "Instant four-head",
    "clock": "Elapsed-time",
}


def main() -> None:
    outcome = pd.read_csv(RESULTS / "outcome_metrics.csv")
    subtype = pd.read_csv(RESULTS / "failure_subtype_alarm_rates.csv")
    onset = pd.read_csv(RESULTS / "onset_event_metrics.csv")
    external = pd.read_csv(EXTERNAL / "outcome_metrics.csv")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 8.7), constrained_layout=True)

    primary = outcome[np.isclose(outcome["quantile"], 0.95)]
    shown = [
        "instability",
        "lock_in",
        "dual_mean",
        "multi_max",
        "instant_multi_max",
        "clock",
    ]
    ax = axes[0, 0]
    for detector in shown:
        row = primary[primary["detector"] == detector].iloc[0]
        ax.scatter(
            100 * row.success_fpr,
            100 * row.failure_recall,
            s=58 if detector in {"multi_max", "clock"} else 42,
            color=COLORS[detector],
            zorder=3,
        )
        ax.annotate(
            LABELS[detector],
            (100 * row.success_fpr, 100 * row.failure_recall),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set(xlabel="Successful trajectories alarmed (%)", ylabel="Failed trajectories alarmed (%)")
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 44)
    ax.grid(alpha=0.22)
    ax.set_title("A  Main cohort at unlabeled q95")

    ax = axes[0, 1]
    for detector in ("multi_max", "dual_mean", "instant_multi_max", "clock"):
        rows = outcome[outcome["detector"] == detector].sort_values("success_fpr")
        ax.plot(
            100 * rows.success_fpr,
            100 * rows.failure_recall,
            marker="o",
            linewidth=1.8,
            color=COLORS[detector],
            label=LABELS[detector],
        )
        for row in rows.itertuples(index=False):
            ax.annotate(
                f"q{100 * row.quantile:g}",
                (100 * row.success_fpr, 100 * row.failure_recall),
                xytext=(3, -10),
                textcoords="offset points",
                fontsize=6.8,
                color=COLORS[detector],
            )
    ax.set(xlabel="Successful trajectories alarmed (%)", ylabel="Failed trajectories alarmed (%)")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("B  Frozen alarm-budget sweep")

    behaviors = [
        "gripper_cycling",
        "regrasp_or_drop",
        "stagnation",
        "subtask_undo",
        "other",
    ]
    heads = [
        "instability",
        "lock_in",
        "flat_narrow_support",
        "feedback_decoupling",
        "multi_max",
    ]
    matrix = np.full((len(behaviors), len(heads)), np.nan)
    for row_index, behavior in enumerate(behaviors):
        for column_index, detector in enumerate(heads):
            row = subtype[
                (subtype["primary_behavior"] == behavior)
                & (subtype["detector"] == detector)
            ]
            if len(row):
                matrix[row_index, column_index] = 100 * row.iloc[0].alarm_rate
    ax = axes[1, 0]
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=50, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(
                column,
                row,
                f"{value:.1f}" if np.isfinite(value) else "-",
                ha="center",
                va="center",
                color="white" if value >= 29 else "#202020",
                fontsize=8,
            )
    ax.set_xticks(range(len(heads)), ["Instability", "Lock-in", "Flat/narrow", "Feedback", "Four-head"], rotation=23, ha="right")
    ax.set_yticks(range(len(behaviors)), ["Gripper cycling", "Regrasp/drop", "Stagnation", "Subtask undo", "Other"])
    ax.set_title("C  Failure subtype alarm rate at q95 (%)")
    figure.colorbar(image, ax=ax, shrink=0.75, label="Alarm rate (%)")

    ax = axes[1, 1]
    onset_behaviors = ["all", "gripper_cycling", "regrasp_or_drop", "stagnation"]
    onset_labels = ["All onsets", "Gripper cycling", "Regrasp/drop", "Stagnation"]
    multi_onset = onset[onset["detector"] == "multi_max"].set_index("primary_behavior")
    metrics = [
        ("strict_precursor_m4_m1_rate", "Precursor [-4,-1]", "#d95f02"),
        ("timely_m2_p2_rate", "Timely [-2,+2]", "#1f78b4"),
        ("reaction_p0_p4_rate", "Reaction [0,+4]", "#66a61e"),
    ]
    x = np.arange(len(onset_behaviors))
    width = 0.23
    for offset, (column, label, color) in enumerate(metrics):
        values = np.asarray(
            [100 * multi_onset.loc[behavior, column] for behavior in onset_behaviors]
        )
        bars = ax.bar(x + (offset - 1) * width, values, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x, onset_labels, rotation=18, ha="right")
    ax.set_ylabel("Episodes with first alarm in window (%)")
    ax.set_ylim(0, 39)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.22)
    ax.set_title("D  Four-head timing around physical proxy onset")

    external_primary = external[
        (external["detector"] == "multi_max")
        & np.isclose(external["quantile"], 0.95)
    ].iloc[0]
    figure.suptitle(
        "Train-free causal MoE alarm: phenotype information, weak overall trigger\n"
        f"Main: 14,800 trajectories, 487 failures. External: 1,200 trajectories, "
        f"45 failures; four-head recall {100 * external_primary.failure_recall:.1f}%, "
        f"FPR {100 * external_primary.success_fpr:.1f}%.",
        fontsize=13,
        fontweight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight")
    print(OUTPUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
