#!/usr/bin/env python3
"""Plot measured GPU load and task completion during a completed collection."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from collection_storage import atomic_json, digest


def run(args):
    summary_path, samples_path = args.run / "summary.json", args.run / "gpu_samples.json"
    summary = json.loads(summary_path.read_text())
    samples = json.loads(samples_path.read_text())
    if summary["status"] != "completed" or 6 in summary["gpus"]:
        raise ValueError("Only a completed collection excluding GPU 6 may be plotted")
    gpus = summary["gpus"]
    started = datetime.fromisoformat(summary["collection_started_utc"]).timestamp()
    offset = datetime.fromisoformat(samples[0]["time"]).timestamp() - started
    elapsed = (np.asarray([sample["monotonic"] for sample in samples]) - samples[0]["monotonic"] + offset) / 60
    power, utilization, active, completed, memory = [], [], [], [], []
    for sample in samples:
        entries = {row["gpu"]: row for row in sample["gpus"]}
        if set(entries) != set(gpus):
            raise ValueError("Telemetry GPU set differs from run")
        power.append([entries[gpu]["power_w"] for gpu in gpus])
        utilization.append([entries[gpu]["utilization_percent"] for gpu in gpus])
        active.append(sum(sample["active_tasks_by_gpu"].values()))
        completed.append(len(summary["tasks"]) - sample["queued_tasks"] - active[-1])
        memory.append(sample["host_memory_bytes"] / 1024**3)
    power, utilization = np.asarray(power), np.asarray(utilization)
    if not np.isfinite(power).all() or not np.all(np.diff(elapsed) >= 0):
        raise ValueError("Invalid telemetry values or timestamps")
    if not np.all(np.diff(completed) >= 0) or summary["completed_tasks"] != len(summary["tasks"]):
        raise ValueError("Incomplete task sequence")
    destinations = [args.output.with_suffix(suffix) for suffix in (".png", ".pdf", ".json")]
    if any(path.exists() for path in destinations):
        raise ValueError("Resource report already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for index, gpu in enumerate(gpus):
        axes[0].plot(elapsed, power[:, index], linewidth=.6, alpha=.65, label=f"GPU {gpu}")
    axes[0].plot(elapsed, power.mean(1), color="black", linewidth=1.5, label="Mean")
    axes[0].axhline(300, color="gray", linestyle="--", linewidth=.9, label="300 W")
    axes[0].set_ylabel("GPU power (W)")
    axes[0].legend(ncol=5, fontsize=8, loc="upper left")
    axes[0].set_ylim(0, max(350, power.max() + 15))
    workload = "reused parents with C0 and four branch arms" if summary.get("continuation_experiment") else "native mains with C0/C1/T1"
    axes[0].set_title(f"{summary['model'].title()} collection: {len(summary['tasks'])} {workload}")
    axes[1].plot(elapsed, utilization.mean(1), color="#13866a", linewidth=1, label="Mean GPU-Util")
    axes[1].set_ylabel("GPU-Util (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend(loc="lower left", fontsize=8)
    secondary = axes[1].twinx()
    secondary.plot(elapsed, memory, color="#bd583e", linewidth=1, label="Host memory")
    secondary.set_ylabel("Host memory (GiB)", color="#bd583e")
    secondary.set_ylim(0, 512)
    secondary.legend(loc="lower right", fontsize=8)
    final_minute = summary["collection_elapsed_seconds"] / 60
    axes[2].step(np.r_[elapsed, max(final_minute, elapsed[-1])],
                 np.r_[completed, summary["completed_tasks"]], where="post",
                 color="#2468a2", label="Complete main + required suffixes")
    axes[2].plot(elapsed, active, color="#a06c13", linewidth=1, label="Active task slots")
    axes[2].set_ylabel("Tasks")
    axes[2].set_xlabel("Collection time (min)")
    axes[2].set_ylim(0, len(summary["tasks"]) * 1.05)
    axes[2].legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xlim(0, max(final_minute, elapsed[-1]))
    fig.tight_layout()
    for path in destinations[:2]:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    atomic_json(destinations[2], dict(
        run=str(args.run.resolve()), source_sha256=digest(Path(__file__)),
        summary_sha256=digest(summary_path), samples_sha256=digest(samples_path),
        samples=len(samples), physical_gpus=gpus, excluded_gpu=6,
        sample_mean_power_w=float(power.mean()), peak_sample_power_w=float(power.max()),
        gpu_samples_at_least_300w_fraction=float((power >= 300).mean()),
        sample_mean_gpu_utilization_percent=float(utilization.mean()),
        figures={path.name: digest(path) for path in destinations[:2]},
        scope="Measured collection including environment work; GPU-Util is not FLOPs utilization"))
    print(json.dumps(dict(png=str(destinations[0]), pdf=str(destinations[1]),
                         mean_power_w=float(power.mean()), peak_power_w=float(power.max()))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
