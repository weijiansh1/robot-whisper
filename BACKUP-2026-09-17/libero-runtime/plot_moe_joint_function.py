"""Plot measured functional quantities from the twelve actual replay probes."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/data/libero-runtime/samples/moe-joint-patterns-20260915/functional-probe")


def main():
    with (OUT / "functional-readouts.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["layers"] == "back" and r["tokens"] == "action" and r["flow"] == "late"]
    labels = [r["episode"] + " q" + r["query"] + (" *" if r["joint_active"] == "True" else "") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), layout="constrained", sharey=True)
    measures = [("routed_authority", "Routed / (routed + shared) norms (%)"),
                ("expert_cancellation", "Vector cancellation (%)"),
                ("largest_weight_and_contribution_differ", "Different top gate / contribution (%)")]
    for ax, (key, title) in zip(axes, measures):
        for i, row in enumerate(rows):
            color = "#32916c" if row["success"] == "True" else "#bf4c53"
            ax.scatter(100 * float(row[key]), i, color=color, s=48)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks(np.arange(len(rows)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[2].scatter([], [], c="#32916c", label="Successful source")
    axes[2].scatter([], [], c="#bf4c53", label="Failed source")
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.06), ncol=2, fontsize=8)
    fig.suptitle("Back layers, action tokens, last 3 flow iterations | * = active joint motif\n"
                 "12 selected snapshots from 6 episodes; norm share is not causal importance", fontsize=12)
    fig.savefig(OUT / "functional-probe.png", dpi=180, facecolor="white")
    fig.savefig(OUT / "functional-probe.pdf", facecolor="white")
    plt.close(fig)
    print("Wrote functional probe PNG and PDF")


if __name__ == "__main__":
    main()
