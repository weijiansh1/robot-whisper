#!/usr/bin/env python3
"""Render the compact result figure from validated CSV artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ranking = pd.read_csv(RESULTS / "selection_metrics.csv")
    ranking = ranking[
        (ranking["query"] == 34)
        & (ranking.budget == 8)
        & (ranking.label_scope == "eventual")
    ].set_index("method")
    sequential = pd.read_csv(RESULTS / "sequential_metrics.csv").set_index("method")

    methods = [
        "single",
        "double_typed_quota",
        "double_typed_noisy_or",
        "double_typed_soft_noisy_or",
    ]
    labels = ["Single", "Typed quota", "Typed noisy-OR", "Soft typed noisy-OR"]
    colors = ["#30343b", "#cf5c36", "#2b7a78", "#4c6a92"]
    metrics = ["precision_trap", "recall_loop", "recall_static"]
    metric_labels = ["Trap precision", "Loop recall", "Static recall"]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(metrics))
    width = 0.19
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = ranking.loc[method, metrics].to_numpy(float) * 100
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=color)
    ax.axhline(25, color="#777777", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks(x, metric_labels)
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 70)
    ax.set_title("Fixed pool selection: q=34, K=8 of 32")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    ax = axes[1]
    seq_methods = [
        "single",
        "double_shared_max",
        "double_shared_noisy_or",
        "double_typed_max",
        "double_typed_noisy_or",
    ]
    seq_labels = [
        "Single",
        "Shared max",
        "Shared noisy-OR",
        "Typed max",
        "Typed noisy-OR",
    ]
    seq_colors = ["#30343b", "#7b6d8d", "#5f8f72", "#cf5c36", "#2b7a78"]
    for method, label, color in zip(seq_methods, seq_labels, seq_colors):
        row = sequential.loc[method]
        xval = float(row.false_alarm_rate) * 100
        yval = float(row.recall_trap) * 100
        ax.scatter(xval, yval, s=60, color=color, zorder=3)
        ax.annotate(label, (xval, yval), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax.axvline(8, color="#777777", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Observed no-Trap false alarm (%)")
    ax.set_ylabel("Eventual-Trap recall (%)")
    ax.set_xlim(5.5, 9.5)
    ax.set_ylim(35, 92)
    ax.set_title("Sequential scan: nested-LOIO calibration")
    ax.grid(color="#dddddd", linewidth=0.7)

    fig.suptitle("Type-aware double selector: coverage trade-off, no aggregate gain", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"double_selector_summary.{suffix}", dpi=180)
    plt.close(fig)

    independent = pd.read_csv(RESULTS / "independent_heads_metrics.csv")
    row = independent[
        (independent["query"] == 34)
        & (independent.per_head_budget == 8)
        & (independent.family == "typed")
    ].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)

    ax = axes[0]
    x = np.arange(2)
    single_values = np.array(
        [row.single_k_loop_recall, row.single_k_static_recall]
    ) * 100
    specialist_values = np.array(
        [row.loop_head_recall, row.static_head_recall]
    ) * 100
    ax.bar(x - 0.18, single_values, 0.36, label="Union-label single head", color="#30343b")
    ax.bar(x + 0.18, specialist_values, 0.36, label="Independent typed head", color="#2b7a78")
    ax.set_xticks(x, ["Loop recall", "Static recall"])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 75)
    ax.set_title("Each head receives K=8")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    metric_names = [
        "union_trap_precision",
        "union_trap_recall",
        "union_loop_recall",
        "union_static_recall",
        "union_macro_type_recall",
    ]
    matched_names = [
        "matched_single_trap_precision",
        "matched_single_trap_recall",
        "matched_single_loop_recall",
        "matched_single_static_recall",
        "matched_single_macro_type_recall",
    ]
    labels = ["Trap precision", "Trap recall", "Loop recall", "Static recall", "Macro recall"]
    x = np.arange(len(labels))
    ax.bar(
        x - 0.18,
        row[matched_names].to_numpy(float) * 100,
        0.36,
        label="Single, matched union size",
        color="#30343b",
    )
    ax.bar(
        x + 0.18,
        row[metric_names].to_numpy(float) * 100,
        0.36,
        label="Two-head union",
        color="#cf5c36",
    )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 80)
    ax.set_title(f"Union: {row.union_budget_mean:.1f} selected per pool")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Independent subtype heads: large loop gain, no aggregate efficiency gain", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"independent_heads_summary.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
