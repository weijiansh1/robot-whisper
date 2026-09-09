"""Standalone figures showing alarm distributions, errors, and event timing."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from fusion import HERE

COLORS = {"knn10": "#333333", "knn10_delta_q7": "#8b6597", "v7_guard": "#4788b7", "v8_guard": "#2c947b",
          "v82_guard": "#8a9e41", "knn12_v8": "#d28430", "knn_and_v7": "#876357",
          "knn_and_v8": "#c34f5e", "knn_or_v8": "#638445", "knn10_persist3": "#888888"}
LABELS = {"knn10": "Original kNN-10D", "knn10_delta_q7": "kNN change from q7", "v7_guard": "v7 (same split)",
          "v8_guard": "v8 (same split)", "v82_guard": "v8.2 (same split)", "knn12_v8": "kNN-12D + v8 features",
          "knn_and_v7": "kNN AND v7", "knn_and_v8": "kNN AND v8", "knn_or_v8": "kNN OR v8",
          "knn10_persist3": "kNN persistent-3"}
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")


def save(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def original_histograms(data, directory):
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, layout="constrained")
    for row, suite in enumerate(SUITES):
        for column, failure in enumerate((True, False)):
            group = data.loc[data.suite.eq(suite) & data.failure.eq(failure) & data.method.eq("knn10")]
            ax = axes[row, column]
            ax.bar(group["query"], group.new_alarms / group.episodes, color="#b95462" if failure else "#428c79", width=.85)
            total = int(group.episodes.iloc[0])
            fired = int(group.new_alarms.sum())
            ax.set_title(f"{suite.removeprefix('libero_').title()} | {'Failed' if failure else 'Successful'} episodes", fontsize=11, loc="left")
            ax.text(.98, .93, f"Alarmed: {fired}/{total}\nNo alarm: {total-fired}/{total}", ha="right", va="top", transform=ax.transAxes, fontsize=9)
            ax.axvline(7, color="#aaaaaa", linewidth=.8, linestyle=":")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
            ax.set_xlim(5, 52)
            ax.set_xticks([7, 10, 15, 20, 30, 40, 50])
            if column == 0:
                ax.set_ylabel("Fraction of all failures")
            else:
                ax.set_ylabel("Fraction of all successes")
    for ax in axes[-1]:
        ax.set_xlabel("First alarm query q (zero-based; completed task actions = 10q)")
    fig.suptitle("Original kNN: full first-alarm distributions\nUnseen tasks; group calibration 5%; pooled appearances across 12 folds", fontsize=13)
    save(fig, directory, "original_alarm_distribution")


def cumulative_curves(data, directory):
    chosen = ("knn10", "v7_guard", "v8_guard", "knn12_v8", "knn_and_v8", "knn_or_v8")
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, layout="constrained")
    handles = []
    for row, suite in enumerate(SUITES):
        for column, failure in enumerate((True, False)):
            ax = axes[row, column]
            for method in chosen:
                group = data.loc[data.suite.eq(suite) & data.failure.eq(failure) & data.method.eq(method)]
                line, = ax.step(group["query"], group.cumulative_fraction, where="post", color=COLORS[method],
                    linewidth=1.8 if method in ("knn10", "knn_and_v8") else 1.3,
                    linestyle="--" if method == "knn_or_v8" else "-", label=LABELS[method])
                if row == 0 and column == 0:
                    handles.append(line)
            ax.set_title(f"{suite.removeprefix('libero_').title()} | {'Cumulative recall' if failure else 'Cumulative FPR'}", fontsize=11, loc="left")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.set_xlim(5, 52)
            if failure:
                ax.set_ylim(0, 1.04)
            else:
                ax.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel("Query q (zero-based)")
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("When the detectors fire\nDenominators include every failed / successful episode, including those without alarms", fontsize=13)
    save(fig, directory, "cumulative_alarm_comparison")


def error_and_event(data, physical, summary, directory):
    methods = ["knn10", "knn10_delta_q7", "v7_guard", "v8_guard", "v82_guard", "knn12_v8", "knn_and_v7", "knn_and_v8", "knn_or_v8"]
    physical = physical.loc[physical.scope.eq("unseen")].set_index("method")
    summary = summary.loc[summary.scope.eq("unseen")].set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.7), gridspec_kw={"width_ratios": [1.35, 1, 1]}, layout="constrained")
    ypos = np.arange(len(methods))
    left = np.zeros(len(methods))
    for field, color, label in (("before", "#337f67", "Before event"), ("at_event", "#60a6b8", "Same query"),
                                ("after", "#d08c47", "After event"), ("missed", "#d6d6d6", "No alarm")):
        values = np.array([physical.loc[method, field] / physical.loc[method, "events"] for method in methods])
        axes[0].barh(ypos, values, left=left, height=.65, color=color, label=label)
        left += values
    axes[0].set_yticks(ypos, [LABELS[method] for method in methods], fontsize=9)
    axes[0].set_title("Timing relative to physical release", fontsize=11)
    axes[0].set_xlabel("Fraction of 175 event appearances")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2, fontsize=8, frameon=False)
    for ax, metric, title in ((axes[1], "recall_lead4", "Failure recall with >=4 queries left"), (axes[2], "fpr", "All successful-episode false alarms")):
        values = summary.loc[methods, metric].to_numpy()
        ax.barh(ypos, values, color=[COLORS[method] for method in methods], height=.65)
        ax.set_yticks(ypos, [""]*len(methods))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Mean across the same 12 folds")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(0, max(values)*1.2)
        for y, value in zip(ypos, values):
            ax.text(value + max(values)*.025, y, f"{value:.1%}", va="center", fontsize=8)
    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle("Early detection and its false-alarm cost\nUnseen tasks; fixed group calibration 5%; TP lead filters never remove FP", fontsize=13)
    save(fig, directory, "physical_timing_and_error_cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    args = parser.parse_args()
    output = args.input.resolve()
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "savefig.facecolor": "white"})
    data = pd.read_csv(output / "query_distribution.csv")
    data = data.loc[data.scope.eq("unseen")]
    original_histograms(data, directory)
    cumulative_curves(data, directory)
    error_and_event(data, pd.read_csv(output / "physical_summary.csv"), pd.read_csv(output / "fold_macro_summary.csv"), directory)
    print("DISTRIBUTION FIGURES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
