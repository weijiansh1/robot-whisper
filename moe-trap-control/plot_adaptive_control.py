#!/usr/bin/env python3
"""Render audited control comparisons or measured four-GPU resource traces."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collection_storage import atomic_json, digest


def save(fig, output, sources, details):
    destinations = [output.with_suffix(suffix) for suffix in (".png", ".pdf", ".json")]
    if any(path.exists() for path in destinations):
        raise ValueError("Refusing to replace a figure")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for path in destinations[:2]:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    atomic_json(destinations[2], dict(source_sha256={str(p): digest(p) for p in sources},
        plotter_sha256=digest(__file__), artifacts={p.name: digest(p) for p in destinations[:2]}, **details))
    print(str(destinations[0]))


def resources(run, output):
    summary_path, samples_path = run / "summary.json", run / "gpu_samples.json"
    summary = json.loads(summary_path.read_text())
    samples = json.loads(samples_path.read_text())
    if summary["status"] != "completed" or summary["gpus"] != [0, 1, 2, 3]:
        raise ValueError("Incomplete run or GPU scope mismatch")
    x = (np.asarray([s["monotonic"] for s in samples]) - samples[0]["monotonic"]) / 60
    power = np.asarray([[g["power_w"] for g in s["gpus"]] for s in samples])
    util = np.asarray([[g["utilization_percent"] for g in s["gpus"]] for s in samples])
    active = np.asarray([sum(s["active_tasks_by_gpu"].values()) for s in samples])
    memory = np.asarray([s["host_memory_bytes"] for s in samples]) / 1024**3
    supplied = np.asarray([all(s["active_tasks_by_gpu"][str(g)] >= summary["replicas_per_gpu"]
                        for g in summary["gpus"]) for s in samples])
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    colors = ("#258476", "#b8574b", "#526dac", "#8d7a38")
    for i, color in enumerate(colors):
        axes[0].plot(x, power[:, i], color=color, linewidth=.7, alpha=.7, label="GPU %d" % i)
    axes[0].plot(x, power.mean(1), color="black", linewidth=1.3, label="Mean")
    axes[0].axhline(300, color="gray", linestyle="--", linewidth=.8)
    axes[0].set_ylabel("Power (W)")
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_title("Route-control collection: four GPUs, batch=1, independent event/repeat jobs")
    axes[1].plot(x, util.mean(1), color=colors[0], label="Mean GPU-Util")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("GPU-Util (%)")
    twin = axes[1].twinx()
    twin.plot(x, memory, color=colors[1], linewidth=.8)
    twin.set_ylabel("Host memory (GiB)", color=colors[1])
    axes[2].plot(x, active, color=colors[2], label="Active jobs")
    axes[2].axhline(4 * summary["replicas_per_gpu"], color="gray", linestyle="--", linewidth=.8, label="Available slots")
    axes[2].set_ylabel("Job slots")
    axes[2].set_xlabel("Collection time (min), excluding model loading")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, max(x[-1], .01))
    save(fig, output, [summary_path, samples_path], dict(
        sample_mean_power_w=float(power.mean()), peak_power_w=float(power.max()),
        sample_mean_gpu_utilization=float(util.mean()), full_slot_samples=int(supplied.sum()),
        full_slot_mean_power_w=float(power[supplied].mean()) if supplied.any() else None,
        full_slot_mean_gpu_utilization=float(util[supplied].mean()) if supplied.any() else None,
        host_peak_memory_gib=float(memory.max()), collection_seconds=summary["collection_elapsed_seconds"],
        query_per_second=summary["queries_per_second"], gpu_scope=[0, 1, 2, 3]))


def comparison(analysis, output):
    path = analysis / "summary.json"
    data = json.loads(path.read_text())
    rows = [r for r in data["metrics"] if r["cohort"] == "primary" and r["timing"] == "after_alarm" and r["method"] != "oracle_once"]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, .4 * len(rows) + 1.4)), sharey=True)
    colors = ["#697077" if r["method"] == "candidate0" else "#218677" if r["method"].startswith("short") else "#bd6853" for r in rows]
    rescue = [100 * (r["rescue_rate"] or 0) for r in rows]
    axes[0].barh(y, rescue, color=colors, height=.65)
    axes[0].set_yticks(y, [r["method"].replace("candidate0", "random baseline") for r in rows])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Rescue rate among native-failed alarm states (%)")
    axes[0].set_xlim(0, max(max(rescue) * 1.35, 1))
    for index, row in enumerate(rows):
        axes[0].text(rescue[index] + axes[0].get_xlim()[1] * .02, index,
                     "%d/%d" % (row["rescued_repeats"], row["failed_repeats"]), va="center", fontsize=8)
    axes[1].axvline(0, color="gray", linewidth=1)
    for index, row in enumerate(rows):
        value, interval = row["delta_vs_random_pp"], row["cluster_ci95_pp"]
        axes[1].plot(value, index, "o", color=colors[index], markersize=5)
        if interval is not None:
            axes[1].hlines(index, interval[0], interval[1], color=colors[index], linewidth=1.4)
    axes[1].set_xlabel("Success difference vs random (percentage points)")
    for axis in axes:
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle("%s: %d primary parents; after-alarm control only\nTask-cluster intervals are exploratory; repeats are not independent parents" %
                 (" + ".join(data["stages"]), data["primary_parents"]), fontsize=10)
    save(fig, output, [path], dict(scope="primary after-alarm policies; oracle and retrospective timing excluded"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", type=Path)
    group.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resources(args.run, args.output) if args.run else comparison(args.analysis, args.output)
