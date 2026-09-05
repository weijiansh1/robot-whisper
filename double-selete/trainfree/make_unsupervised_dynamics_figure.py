#!/usr/bin/env python3
"""Render the MoE state-sequence grammar clustering audit."""

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
RESULTS = HERE / "results/unsupervised_dynamics"
OUTPUT = HERE / "figures/unsupervised_moe_dynamics.png"
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
    "Gate change high",
    "Gate level low",
    "Flow volatility high",
    "Route acceleration high",
    "Route mobility low",
    "Deep consensus change",
    "Deep mixture change",
    "L15 mixture high",
    "L15 consensus high",
    "L15 final jump high",
)
PHENOTYPES = ("normal", "loop_only", "static_only", "both")
PHENOTYPE_COLORS = {
    "normal": "#696969",
    "loop_only": "#E68613",
    "static_only": "#2A9D6F",
    "both": "#775DA6",
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.12,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )


def scatter_assignment(
    axis: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    colors = plt.get_cmap("tab10")
    for label in np.unique(labels):
        take = labels == label
        axis.scatter(
            embedding[take, 0],
            embedding[take, 1],
            s=19,
            alpha=0.72,
            edgecolors="none",
            color=colors(int(label) % 10),
            label=f"C{int(label)} (n={take.sum()})",
        )
    axis.set_title(title)
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend(frameon=False, ncol=2, fontsize=7.5, loc="best")


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.results / "labeled_trajectory_assignments.csv").sort_values(
        "episode_id"
    )
    state = pd.read_csv(args.results / "state_vocabulary.csv").sort_values("state")
    composition = pd.read_csv(args.results / "cluster_composition.csv")
    onset = pd.read_csv(args.results / "onset_state_profiles.csv")
    summary = json.loads((args.results / "evaluation_summary.json").read_text())
    with np.load(args.results / "unlabeled_dynamics.npz", allow_pickle=False) as data:
        early_embedding = np.asarray(data["early_embedding"], dtype=np.float64)
        full_embedding = np.asarray(data["full_embedding"], dtype=np.float64)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(16.8, 9.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.05))
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    ax_state, ax_early, ax_early_label, ax_full, ax_composition, ax_grammar = axes

    state_matrix = state[
        [f"center_mean__{component}" for component in COMPONENTS]
    ].to_numpy()
    image_state = ax_state.imshow(
        state_matrix, aspect="auto", cmap="RdBu_r", vmin=0.2, vmax=0.8
    )
    ax_state.set_xticks(range(len(COMPONENT_LABELS)), COMPONENT_LABELS, rotation=55, ha="right")
    ax_state.set_yticks(range(len(state)), [f"S{value}" for value in state["state"]])
    ax_state.set_title("BIC-selected routing vocabulary (K=3)")
    colorbar = fig.colorbar(image_state, ax=ax_state, shrink=0.76, pad=0.02)
    colorbar.set_label("Mean within-pool/query rank")
    panel_label(ax_state, "A")

    scatter_assignment(
        ax_early,
        early_embedding,
        frame["early_gmm_bic_cluster"].to_numpy(),
        "Early grammar: GMM-BIC K=5, silhouette=0.055",
    )
    panel_label(ax_early, "B")

    for phenotype in PHENOTYPES:
        take = frame["phenotype"].to_numpy() == phenotype
        ax_early_label.scatter(
            early_embedding[take, 0],
            early_embedding[take, 1],
            s=19,
            alpha=0.65,
            edgecolors="none",
            color=PHENOTYPE_COLORS[phenotype],
            label=f"{phenotype} (n={take.sum()})",
        )
    early_test = summary["key_label_tests"]["early_gmm_4way"]
    ax_early_label.set_title(
        f"Early labels overlap: AMI={early_test['adjusted_mutual_info']:.4f}, "
        f"p={early_test['conditional_permutation_p']:.3f}"
    )
    ax_early_label.set_xlabel("PC1")
    ax_early_label.set_ylabel("PC2")
    ax_early_label.legend(frameon=False, fontsize=7.5, loc="best")
    panel_label(ax_early_label, "C")

    scatter_assignment(
        ax_full,
        full_embedding,
        frame["full_gmm_bic_cluster"].to_numpy(),
        "Full observed grammar: GMM-BIC K=7, silhouette=0.021",
    )
    panel_label(ax_full, "D")

    full_composition = composition[
        composition["assignment"] == "full_gmm_bic_partition"
    ].sort_values("cluster")
    y = [f"C{cluster}" for cluster in full_composition["cluster"]]
    left = np.zeros(len(full_composition), dtype=np.float64)
    for phenotype in PHENOTYPES:
        values = full_composition[f"{phenotype}_rate"].to_numpy() * 100
        bars = ax_composition.barh(
            y,
            values,
            left=left,
            height=0.67,
            color=PHENOTYPE_COLORS[phenotype],
            label=phenotype,
        )
        for bar, start, value in zip(bars, left, values):
            if value >= 9:
                ax_composition.text(
                    start + value / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if phenotype != "static_only" else "#153C30",
                )
        left += values
    full_test = summary["key_label_tests"]["full_gmm_4way"]
    ax_composition.set_xlim(0, 100)
    ax_composition.invert_yaxis()
    ax_composition.set_xlabel("Post-hoc phenotype composition (%)")
    ax_composition.set_title(
        f"Retrospective structure: AMI={full_test['adjusted_mutual_info']:.3f}, p<5e-5"
    )
    ax_composition.legend(
        frameon=False,
        ncol=4,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        columnspacing=0.8,
    )
    panel_label(ax_composition, "E")

    difference = (
        onset.pivot(index="event", columns="state", values="delta_mean")
        .loc[["loop", "static"], [-1, 0, 1, 2]]
        .to_numpy()
    )
    limit = max(0.05, float(np.abs(difference).max()))
    image_grammar = ax_grammar.imshow(
        difference, aspect="auto", cmap="PuOr_r", vmin=-limit, vmax=limit
    )
    for row in range(difference.shape[0]):
        for column in range(difference.shape[1]):
            ax_grammar.text(
                column,
                row,
                f"{difference[row, column]:+.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
            )
    ax_grammar.set_xticks(range(4), ["Noise", "S0", "S1", "S2"])
    ax_grammar.set_yticks(range(2), ["loop: post - pre", "static: post - pre"])
    ax_grammar.set_title("Paired state-occupancy change around physical onset")
    grammar_colorbar = fig.colorbar(image_grammar, ax=ax_grammar, shrink=0.76, pad=0.02)
    grammar_colorbar.set_label("Mean feature difference")
    panel_label(ax_grammar, "F")

    fig.suptitle(
        "MoE routing dynamics grammar: no early separation, modest retrospective subtype structure",
        fontsize=14.5,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
