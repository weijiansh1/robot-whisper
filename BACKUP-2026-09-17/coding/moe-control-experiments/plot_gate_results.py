"""Plot precomputed gate-response metrics without fitting or selecting a model."""

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
    args = parser.parse_args()
    result = json.loads((args.run / "analysis.json").read_text())
    primary = [row for row in result["table"] if row["amplitude"] == .1]
    low = {row["scope"]: row for row in result["table"] if row["amplitude"] == .05}
    names = [row["scope"] for row in primary]
    x = np.arange(len(names))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    axes[0].bar(x - .19, [low[name]["mean_normalized_action_rms"] for name in names], .36,
                color="#397ead", label="Per-site bias RMS 0.05")
    axes[0].bar(x + .19, [row["mean_normalized_action_rms"] for row in primary], .36,
                color="#bc4b50", label="Per-site bias RMS 0.10")
    axes[0].set_ylabel("Normalized 6D action RMS change")
    axes[0].legend(frameon=False)
    axes[0].set_title("Gate-to-action response: 8 parent states, 4 base tasks; no environment actions", fontsize=13)
    colors = ["#288771" if "_state_" in name else "#a37535" for name in names]
    axes[1].bar(x, [row["mean_action_l2_per_bias_l2"] for row in primary], .65, color=colors)
    axes[1].set_ylabel("Normalized action L2 / total bias L2")
    axes[1].set_title("Energy-normalized response at per-site RMS 0.10", fontsize=12)
    for ax in axes:
        ax.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=9)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.run / "gate-response.png", dpi=160)
    fig.savefig(args.run / "gate-response.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
