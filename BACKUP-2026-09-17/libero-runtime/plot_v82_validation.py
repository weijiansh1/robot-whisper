"""Render the fixed-prefix validation artifacts without importing policy code."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(root):
    result = json.loads((root / "results.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.1))
    fig.subplots_adjust(left=.055, right=.985, top=.76, bottom=.24, wspace=.32)
    fig.suptitle("Frozen v8.2: equal-length prefix validation", x=.055, y=.96,
                 ha="left", fontsize=17, fontweight="bold")
    fig.text(.055, .895, "60 existing episodes: 51 failures, 9 successes. All episodes are active at every tested prefix.",
             ha="left", fontsize=11)
    colors = {"v7": "#5d8066", "v8": "#747b82", "v82": "#b34048"}
    qs, x = (10, 15, 18), np.arange(3)
    for i, version in enumerate(colors):
        detected = [result["metrics"]["q%d" % q][version]["all"]["tp"] for q in qs]
        bars = axes[0].bar(x + (i - 1) * .25, detected, width=.23,
                          color=colors[version], label={"v7": "v7", "v8": "Fixed v8", "v82": "v8.2"}[version])
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].set_xticks(x, [f"{q + 1} chunks\n{q * 10} actions" for q in qs])
    axes[0].set_ylim(0, 22)
    axes[0].set_yticks([0, 5, 10, 15, 20])
    axes[0].set_ylabel("Detected failures (out of 51)")
    axes[0].set_title("Same observation budget", loc="left", fontsize=12, pad=15)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", fontsize=9,
                   handlelength=1.1, columnspacing=.9)
    false_counts = [result["metrics"]["q%d" % q]["v82"]["all"]["fp"] for q in qs]
    fig.text(.055, .12, "v8.2 false alarms / 9: " + ", ".join(map(str, false_counts)), fontsize=9)
    primary = result["permutation_tests"]["q15"]["benchmark_task_exact"]
    with np.load(root / "permutation-distributions.npz") as distributions:
        for ax, statistic, title, xlabel in (
            (axes[1], "alarm_youden_j", "Primary: alarm association", "Youden J (recall - false-alarm rate)"),
            (axes[2], "conditional_continuous_auc", "Secondary: continuous risk", "Within-task AUC (16 failure/success pairs)"),
        ):
            test = primary[statistic]
            null = distributions["q15__benchmark_task_exact__" + statistic]
            values, counts = np.unique(np.round(null, 10), return_counts=True)
            width = float(np.min(np.diff(values))) * .78 if len(values) > 1 else .035
            ax.bar(values, counts / len(null), width=width, color="#d2d7d6", edgecolor="#64726f", linewidth=.7)
            ax.axvline(test["observed"], color=colors["v82"], linewidth=2, label="Observed")
            ax.set_title(title, loc="left", fontsize=12, pad=15)
            ax.set_xlabel(xlabel, labelpad=11, fontsize=9)
            ax.set_ylabel("Fraction of exact assignments")
            ax.text(.04, .96, f"Observed = {test['observed']:.3f}\nExact p = {test['p_one_sided']:.4f}",
                    va="top", transform=ax.transAxes, fontsize=10)
            ax.set_ylim(0, max(counts / len(null)) * 1.42)
            ax.legend(frameon=False, loc="upper right", fontsize=9)
            if statistic.endswith("auc"):
                ax.set_xlim(-.03, 1.03)
    fig.text(.055, .065, "Null: 6,561 outcome assignments within benchmark and task; complete routing prefixes remain intact.", fontsize=10)
    fig.text(.055, .025, "Exploratory analysis; secondary p values are unadjusted. No tuning, intervention, or independent holdout.", fontsize=10, color="#52585b")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e5e7e6", linewidth=.7)
    for extension in ("png", "pdf"):
        fig.savefig(root / ("validation." + extension), dpi=180, facecolor="white")
    plt.close(fig)
    print(root / "validation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    main(parser.parse_args().root)
