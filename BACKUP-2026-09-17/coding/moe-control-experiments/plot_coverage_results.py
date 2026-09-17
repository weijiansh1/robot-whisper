"""Plot frozen parent-balanced coverage results; no fitting or outcome selection."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    root = parser.parse_args().run
    result = json.loads((root / "analysis.json").read_text())
    rows = result["table"]
    names = [r["generator"] for r in rows]
    labels = ["IID\nnoise", "State gate\n0.10", "Action gate\n0.10", "Action gate\nmatched L2", "2 noise +\n2 state gate"]
    colors = ["#487da5", "#2e8b73", "#b35758", "#856b9b", "#8a8d42"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout="constrained")
    x = np.arange(len(rows))
    axes[0].bar(x, [r["mean_radius_rms"] for r in rows], color=colors, width=.58, alpha=.85)
    axes[0].set_ylabel("Mean normalized action RMS from default")
    axes[0].set_title("Discrete candidate diversity: 8 parents, 4 base tasks, 2 pools per parent", fontsize=12)
    axes[1].bar(x, [100 * r["coverage_gain"] for r in rows], color=colors, width=.58, alpha=.85)
    axes[1].set_ylabel("Reference-distance reduction (%)")
    axes[1].set_title("Coverage of independent IID-noise reference actions; not recovery success", fontsize=12)
    parents = sorted({r["parent"] for r in result["per_parent"]})
    for i, name in enumerate(names):
        for j, parent in enumerate(parents):
            row, = [r for r in result["per_parent"] if r["generator"] == name and r["parent"] == parent]
            offset = (j - (len(parents) - 1) / 2) * .045
            axes[0].scatter(i + offset, row["mean_radius_rms"], color="#232323", s=16, zorder=3)
            axes[1].scatter(i + offset, 100 * row["coverage_gain"], color="#232323", s=16, zorder=3)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=10)
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "coverage.png", dpi=160)
    fig.savefig(root / "coverage.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
