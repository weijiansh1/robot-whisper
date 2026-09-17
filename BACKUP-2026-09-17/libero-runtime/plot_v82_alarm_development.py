"""Plot original, task-held-out, and fitted alarm development results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root, result_file="results.json", stem="alarm-development"):
    result = json.loads((root / result_file).read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.7), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.subplots_adjust(left=.065, right=.97, top=.76, bottom=.25, wspace=.27)
    fig.suptitle("v8.2 alarm development", x=.065, y=.95, ha="left", fontsize=18, fontweight="bold")
    fig.text(.065, .88, "60 existing episodes: 51 failures and 9 successes. Original model and feature calculations preserved.", fontsize=11)
    series = [("original", "Original v8.2", "#69737a", "-"),
              ("task_held_out", "Task-held-out selection", "#267365", "-"),
              ("all_cohort_fitted", "Development fit (in-sample)", "#b34748", "--")]
    windows = ["q15", "q18", "q26", "q30", "q36", "q40", "full_episode"]
    x = np.array([150, 180, 260, 300, 360, 400, 520])
    for key, label, color, style in series:
        y = [result[key][window]["all"]["tp"] for window in windows]
        axes[0].plot(x, y, color=color, linestyle=style, marker="o", markersize=4, linewidth=2, label=label)
    axes[0].axvline(360, color="#a6aaa8", linewidth=1, linestyle=":")
    axes[0].set_xticks(x, ["150", "180", "260", "300", "360", "400", "End"])
    axes[0].set_ylim(0, 55)
    axes[0].set_yticks([0, 10, 20, 30, 40, 51])
    axes[0].set_xlabel("Action deadline (primary comparison: 360)", labelpad=10)
    axes[0].set_ylabel("Failed episodes detected / 51")
    axes[0].set_title("Cumulative alarms", loc="left", fontsize=12, pad=14)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    for i, (key, label, color, _) in enumerate(series):
        metrics = result[key]["full_episode"]["all"]
        value = metrics["false_positive_rate"] * 100
        ci = metrics["fpr_wilson_95"]
        lower, upper = ci["lower"] * 100, ci["upper"] * 100
        axes[1].bar(i, value, width=.5, color=color, alpha=.8)
        axes[1].errorbar(i, value, yerr=[[max(0., value - lower)], [max(0., upper - value)]],
                         fmt="none", ecolor="#333c3a", capsize=5, linewidth=1.2)
        axes[1].text(i, upper + 2, f"{metrics['fp']} / 9", ha="center", fontsize=10)
    axes[1].set_xticks(range(3), ["Original\nv8.2", "Task-held-out\nselection", "Development\nfit"])
    axes[1].set_ylim(0, 55)
    axes[1].set_ylabel("False-alarm rate (%)")
    axes[1].set_title("Any false alarm before success", loc="left", fontsize=12, pad=14)
    axes[1].set_xlabel("Approximate 95% Wilson intervals", labelpad=10)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e8e6", linewidth=.7)
    a = result["original"]["q36"]["all"]["tp"]
    b = result["task_held_out"]["q36"]["all"]["tp"]
    fig.text(.065, .12, f"Primary task-held-out result: {a}/51 to {b}/51 failures detected by 360 actions.", fontsize=11)
    fig.text(.065, .067, "Both benchmarks and all initial states of each held-out task are excluded from that fold's parameter choice.", fontsize=10)
    fig.text(.065, .025, "Exploratory development on previously inspected data. Mid-episode histories have natural, unequal termination lengths.", fontsize=10, color="#525b58")
    for suffix in ("png", "pdf"):
        fig.savefig(root / (stem + "." + suffix), dpi=180, facecolor="white")
    plt.close(fig)
    print(root / (stem + ".png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--result-file", default="results.json")
    parser.add_argument("--stem", default="alarm-development")
    args = parser.parse_args()
    main(args.root, args.result_file, args.stem)
