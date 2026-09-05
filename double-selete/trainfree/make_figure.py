#!/usr/bin/env python3
"""Render the primary train-free two-head result summary."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def row(frame: pd.DataFrame, **query) -> pd.Series:
    take = np.ones(len(frame), bool)
    for key, value in query.items():
        take &= frame[key] == value
    return frame[take].iloc[0]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    auc = pd.read_csv(RESULTS / "snapshot_auc.csv")
    metrics = pd.read_csv(RESULTS / "selection_metrics.csv")
    onset = pd.read_csv(RESULTS / "onset_alignment.csv")

    colors = {"typed": "#177E89", "mean": "#D95F59", "mobility": "#555555"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)

    labels = ["Loop", "Static"]
    x = np.arange(2)
    typed = [
        row(auc, query=34, score="loop_soft", target="loop").group_macro_auc,
        row(auc, query=34, score="static_soft", target="static").group_macro_auc,
    ]
    mean = [
        row(auc, query=34, score="single_soft_mean", target=target).group_macro_auc
        for target in ("loop", "static")
    ]
    mobility = [
        row(auc, query=34, score="single_mobility", target=target).group_macro_auc
        for target in ("loop", "static")
    ]
    for shift, values, name, color in (
        (-0.24, typed, "Typed head", colors["typed"]),
        (0.0, mean, "One-head mean", colors["mean"]),
        (0.24, mobility, "Mobility", colors["mobility"]),
    ):
        axes[0, 0].bar(x + shift, values, 0.22, label=name, color=color)
    axes[0, 0].set_title("Subtype AUC at query 34")
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Group-macro AUC")
    axes[0, 0].legend(frameon=False, fontsize=8)

    typed_recall = [
        row(
            metrics,
            query=34,
            budget_per_head=8,
            selector="loop_soft_topk",
        ).loop_recall,
        row(
            metrics,
            query=34,
            budget_per_head=8,
            selector="static_soft_topk",
        ).static_recall,
    ]
    mean_recall = [
        row(
            metrics,
            query=34,
            budget_per_head=8,
            selector="single_soft_mean_topk",
        ).loop_recall,
        row(
            metrics,
            query=34,
            budget_per_head=8,
            selector="single_soft_mean_topk",
        ).static_recall,
    ]
    axes[0, 1].bar(
        x - 0.18, typed_recall, 0.36, color=colors["typed"], label="Typed head"
    )
    axes[0, 1].bar(
        x + 0.18, mean_recall, 0.36, color=colors["mean"], label="One-head mean"
    )
    axes[0, 1].set_title("Independent Top-8 recall")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0, 0.65)
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].legend(frameon=False, fontsize=8)

    onset_values = {
        "Typed head": [
            row(onset, event="loop", score="loop_soft").pair_weighted_auc,
            row(onset, event="static", score="static_soft").pair_weighted_auc,
        ],
        "One-head mean": [
            row(onset, event="loop", score="single_soft_mean").pair_weighted_auc,
            row(onset, event="static", score="single_soft_mean").pair_weighted_auc,
        ],
        "Mobility": [
            row(onset, event="loop", score="single_mobility").pair_weighted_auc,
            row(onset, event="static", score="single_mobility").pair_weighted_auc,
        ],
    }
    for shift, (name, values), color in zip(
        (-0.24, 0.0, 0.24), onset_values.items(), colors.values(), strict=True
    ):
        axes[1, 0].bar(x + shift, values, 0.22, label=name, color=color)
    axes[1, 0].set_title("Onset-aligned discrimination")
    axes[1, 0].set_xticks(x, ["Loop onset - 2", "Static onset"])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Pair-weighted AUC")

    scalar_names = ["Double max", "One-head mean", "Mobility"]
    scalar_scores = ["double_soft_max", "single_soft_mean", "single_mobility"]
    scalar_values = [
        row(auc, query=34, score=name, target="trap").group_macro_auc
        for name in scalar_scores
    ]
    axes[1, 1].bar(
        np.arange(3),
        scalar_values,
        color=[colors["typed"], colors["mean"], colors["mobility"]],
    )
    axes[1, 1].set_title("Collapsed any-Trap ranking")
    axes[1, 1].set_xticks(np.arange(3), scalar_names)
    axes[1, 1].set_ylim(0, 0.9)
    axes[1, 1].set_ylabel("Group-macro AUC")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Train-free routing heads: keep the two outputs separate", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"trainfree_double_selector.{suffix}", dpi=180)
    print(FIGURES / "trainfree_double_selector.png")


if __name__ == "__main__":
    main()
