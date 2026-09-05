#!/usr/bin/env python3
"""Render the sealed MoE clustering and post-hoc phenotype audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results/unsupervised_cluster"
DEFAULT_OUTPUT = HERE / "figures/unsupervised_moe_peak_clusters.png"
COMPONENTS = (
    "gate_concentration_delta2_high",
    "gate_concentration_w4_low",
    "late_flow_volatility_w4_high",
    "route_acceleration_w4_high",
    "route_mobility_w4_low",
    "deep_soft_consensus_delta2_high",
    "deep_soft_mixture_delta2_high",
    "l15_soft_mixture_now_high",
    "l15_soft_consensus_now_high",
    "l15_soft_final_jump_high",
)
COMPONENT_LABELS = (
    "Gate change (high)",
    "Gate level (low)",
    "Flow volatility (high)",
    "Route acceleration (high)",
    "Route mobility (low)",
    "Deep consensus change (high)",
    "Deep mixture change (high)",
    "L15 mixture (high)",
    "L15 consensus (high)",
    "L15 final jump (high)",
)
CLUSTER_COLORS = {-1: "#B8B8B8", 0: "#D1493F", 1: "#2878B5"}
PHENOTYPE_COLORS = {
    "normal": "#696969",
    "loop_only": "#E68613",
    "static_only": "#2A9D6F",
    "both": "#775DA6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.07,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.results / "labeled_cluster_assignments.csv").sort_values(
        "episode_id"
    )
    summary = json.loads((args.results / "evaluation_summary.json").read_text())
    profiles = pd.read_csv(args.results / "cluster_peak_profiles.csv")
    composition = pd.read_csv(args.results / "cluster_composition.csv")
    bic = pd.read_csv(args.results / "gmm_bic.csv")
    with np.load(args.results / "unlabeled_embedding.npz", allow_pickle=False) as data:
        episode = np.asarray(data["episode"], dtype=np.int64)
        embedding = np.asarray(data["embedding"], dtype=np.float64)
        explained = np.asarray(data["explained_variance_ratio"], dtype=np.float64)
    if not np.array_equal(episode, frame["episode_id"].to_numpy(dtype=np.int64)):
        raise ValueError("embedding and post-hoc labels are not aligned")

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(16.5, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.08))
    ax_cluster = fig.add_subplot(grid[0, 0])
    ax_label = fig.add_subplot(grid[0, 1])
    ax_comp = fig.add_subplot(grid[0, 2])
    ax_c0 = fig.add_subplot(grid[1, 0])
    ax_c1 = fig.add_subplot(grid[1, 1])
    ax_bic = fig.add_subplot(grid[1, 2])

    cluster_names = {-1: "Noise", 0: "C0", 1: "C1"}
    for cluster in (-1, 0, 1):
        take = frame["hdbscan_cluster"].to_numpy() == cluster
        ax_cluster.scatter(
            embedding[take, 0],
            embedding[take, 1],
            s=15 if cluster < 0 else 27,
            alpha=0.33 if cluster < 0 else 0.82,
            color=CLUSTER_COLORS[cluster],
            edgecolors="none",
            label=f"{cluster_names[cluster]} (n={take.sum()})",
        )
    ax_cluster.set_title("Sealed HDBSCAN assignment")
    ax_cluster.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax_cluster.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax_cluster.legend(frameon=False, loc="best")
    ax_cluster.text(
        0.02,
        0.02,
        "Clustering used all 45 retained PCs",
        transform=ax_cluster.transAxes,
        color="#555555",
        fontsize=8,
    )
    panel_label(ax_cluster, "A")

    for phenotype in ("normal", "loop_only", "static_only", "both"):
        take = frame["phenotype"].to_numpy() == phenotype
        ax_label.scatter(
            embedding[take, 0],
            embedding[take, 1],
            s=20,
            alpha=0.65,
            color=PHENOTYPE_COLORS[phenotype],
            edgecolors="none",
            label=f"{phenotype} (n={take.sum()})",
        )
    ax_label.set_title("Labels opened after clustering was sealed")
    ax_label.set_xlabel("PC1")
    ax_label.set_ylabel("PC2")
    ax_label.legend(frameon=False, loc="best")
    panel_label(ax_label, "B")

    composition = composition.set_index("cluster").loc[[-1, 0, 1]]
    phenotype_order = ("normal", "loop_only", "static_only", "both")
    left = np.zeros(3, dtype=np.float64)
    for phenotype in phenotype_order:
        values = composition[f"{phenotype}_rate"].to_numpy() * 100.0
        bars = ax_comp.barh(
            ["Noise", "C0", "C1"],
            values,
            left=left,
            color=PHENOTYPE_COLORS[phenotype],
            label=phenotype,
            height=0.62,
        )
        for bar, value, start in zip(bars, values, left):
            if value >= 6.0:
                ax_comp.text(
                    start + value / 2.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if phenotype != "static_only" else "#102C23",
                )
        left += values
    ax_comp.set_xlim(0, 100)
    ax_comp.invert_yaxis()
    ax_comp.set_xlabel("Post-hoc composition (%)")
    ax_comp.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.9,
    )
    primary_test = next(
        row for row in summary["association_tests"] if row["target"] == "phenotype_4way"
    )
    ax_comp.set_title(
        "No loop/static/normal partition\n"
        f"4-way AMI={primary_test['adjusted_mutual_info']:.3f}, "
        f"conditional p={primary_test['conditional_permutation_p']:.3f}"
    )
    panel_label(ax_comp, "C")

    images = []
    for axis, cluster, title, label in (
        (ax_c0, 0, "C0: transient-instability tail", "D"),
        (ax_c1, 1, "C1: low-change / consensus tail", "E"),
    ):
        part = profiles[profiles["cluster"] == cluster]
        matrix = (
            part.pivot(index="component", columns="relative_query", values="mean_rank")
            .loc[list(COMPONENTS), list(range(-3, 4))]
            .to_numpy()
        )
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
        images.append(image)
        axis.axvline(3, color="black", linewidth=1.0)
        axis.set_xticks(range(7), range(-3, 4))
        axis.set_yticks(range(len(COMPONENT_LABELS)), COMPONENT_LABELS)
        axis.set_xlabel("Query relative to MoE-selected extreme")
        axis.set_title(title)
        panel_label(axis, label)
    colorbar = fig.colorbar(images[0], ax=[ax_c0, ax_c1], shrink=0.82, pad=0.02)
    colorbar.set_label("Within-pool/query mean percentile rank")

    delta_bic = bic["bic"].to_numpy() - bic["bic"].min()
    ax_bic.plot(
        bic["components"], delta_bic, marker="o", color="#465362", linewidth=1.8
    )
    ax_bic.scatter([1], [0], s=65, color="#2A9D6F", zorder=3, label="BIC optimum")
    ax_bic.set_xticks(bic["components"].astype(int))
    ax_bic.set_xlabel("Gaussian mixture components K")
    ax_bic.set_ylabel("BIC minus best BIC (lower is better)")
    ax_bic.set_title("Global density favors one component")
    ax_bic.legend(frameon=False)
    ax_bic.text(
        0.04,
        0.80,
        "HDBSCAN: 2 tail clusters\ncoverage: 14.8%\nOPTICS-Xi: 1 cluster",
        transform=ax_bic.transAxes,
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "pad": 5},
    )
    panel_label(ax_bic, "F")

    fig.suptitle(
        "Train-free MoE peak clustering: reproducible tail shapes, not Trap phenotypes",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
