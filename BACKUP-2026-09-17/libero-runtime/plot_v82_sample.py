"""Plot frozen v8.2 episode outcomes and first-alarm positions from saved results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    report = json.loads((root / "summary.json").read_text())
    if not report["complete"]:
        raise ValueError("Plot the complete sample only")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    colors = dict(TP="#157f62", FN="#bd3b46", FP="#bb741d", TN="#65727b")
    labels = dict(TP="Failure: alarm", FN="Failure: no alarm",
                  FP="Success: alarm", TN="Success: no alarm")
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, layout="constrained")
    for ax, benchmark, title in zip(axes, ("plus", "pro"), ("LIBERO-Plus: camera viewpoints", "LIBERO-Pro: position swap")):
        episodes = sorted((row for row in report["episodes"] if row["benchmark"] == benchmark),
                          key=lambda row: (row["base_task_id"], row["init_state_id"]))
        for y, episode in enumerate(episodes):
            kind = episode["classification_v82"]
            first = episode["first_alarm_action_step_before"]["v82"]
            endpoint = episode["action_steps"]
            ax.plot([0, endpoint], [y, y], color="#e7e9e9", linewidth=1, zorder=0)
            if first is not None:
                ax.scatter(first, y, s=35, marker="o", color=colors[kind], zorder=3)
                ax.scatter(endpoint, y, s=12, marker="|", color="#9da5a8", zorder=2)
            else:
                ax.scatter(endpoint, y, s=35, marker="x", color=colors[kind], linewidth=1.5, zorder=3)
        ax.set_yticks(range(len(episodes)), ["T%02d / init %02d" % (row["base_task_id"], row["init_state_id"])
                                               for row in episodes], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(-10, 535)
        ax.set_xticks([0, 100, 200, 300, 400, 520])
        ax.set_xlabel("Executed actions before first alarm (x = no alarm by endpoint)")
        item = report["metrics"]["v82"][benchmark]
        successes = item["fp"]+item["tn"]
        false_alarms = ("%d false alarm%s / %d successes" % (
            item["fp"], "" if item["fp"] == 1 else "s", successes)
            if successes else "FPR N/A (no successes)")
        ax.set_title("%s\n%d detected, %d missed; %s" % (
            title, item["tp"], item["fn"], false_alarms), fontsize=11, pad=12)
        for boundary in np.arange(2.5, len(episodes)-1, 3):
            ax.axhline(boundary, color="#d0d5d6", linewidth=0.6)
    handles = [plt.Line2D([], [], linestyle="none", marker="o" if kind in ("TP", "FP") else "x",
                          color=colors[kind], label=labels[kind]) for kind in ("TP", "FN", "FP", "TN")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Frozen v8.2: 60 original episodes, 51 failures and 9 successes\n"
                 "Exact observation/noise replay; fixed thresholds; no control intervention", fontsize=13)
    fig.savefig(root / "alarm-timeline.png", dpi=180)
    fig.savefig(root / "alarm-timeline.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    for ax, key, title in zip(axes, ("recall", "false_positive_rate"),
                             ("Failure recall (51 failures)", "False-positive rate (9 successes)")):
        for i, (version, color) in enumerate(zip(("v7", "v8", "v82"), ("#8e9698", "#367a9c", "#157f62"))):
            value = report["metrics"][version]["all"][key]
            point = value["value"]
            bounds = value["wilson_95"]
            ax.bar(i, 100*point, color=color, width=0.6)
            ax.errorbar(i, 100*point, yerr=[[100*(point-bounds["lower"])],
                                          [100*(bounds["upper"]-point)]], color="#30393d", capsize=4)
            ax.text(i, 105, "%d/%d" % (value["numerator"], value["denominator"]), ha="center")
        ax.set_xticks([0, 1, 2], ["v7", "v8", "v8.2"])
        ax.set_ylim(0, 114)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color="#e7e9e9")
    fig.suptitle("Same routing histories and frozen thresholds\nApproximate 95% Wilson intervals; related tasks limit independence", fontsize=11)
    fig.savefig(root / "detection-rates.png", dpi=180)
    fig.savefig(root / "detection-rates.pdf")
    plt.close(fig)
    print(json.dumps({"plots": [str(root / "alarm-timeline.png"), str(root / "detection-rates.png")]}))


if __name__ == "__main__":
    main()
