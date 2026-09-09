"""Matched-population uncertainty and cumulative curves for recalibration."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from fusion import HERE
from core import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round7_temporal_fusion")
    parent = parser.parse_args().input.resolve()
    output = parent / "cross_task_calibration"
    data = pd.read_csv(output / "matched_episode_decisions.csv")
    data = data.loc[data.scope.eq("unseen")].copy()
    events = pd.read_csv(parent / "physical_timing.csv")[["global_row", "drop_goal_release"]].drop_duplicates()
    data = data.merge(events, on="global_row", how="left", validate="many_to_one")
    data["tp"] = data.failure & data.first_alarm_query.ge(0)
    data["fp"] = ~data.failure & data.first_alarm_query.ge(0)
    data["success"] = ~data.failure
    data["tp_by14"] = data.tp & data.first_alarm_query.le(14)
    data["event"] = data.drop_goal_release.notna()
    data["before_event"] = data.first_alarm_query.ge(0) & data.first_alarm_query.lt(data.drop_goal_release)
    fields = ["tp", "fp", "failure", "success", "tp_by14", "before_event", "event"]
    groups = data.groupby(["suite", "task", "method"])[fields].sum()
    tasks = groups.index.droplevel("method").unique()
    methods = sorted(data.method.unique())
    values = np.stack([groups.xs(method, level="method").loc[tasks].to_numpy(float) for method in methods])
    totals = values.sum(axis=1)
    pd.DataFrame(totals, index=pd.Index(methods, name="method"), columns=fields).to_csv(output / "pooled_matched_counts.csv")
    rng = np.random.default_rng(20260912)
    draws = np.concatenate([rng.choice(np.flatnonzero(tasks.get_level_values("suite") == suite),
        size=(2000, int((tasks.get_level_values("suite") == suite).sum())), replace=True)
        for suite in tasks.get_level_values("suite").unique()], axis=1)
    boot = values[:, draws].sum(axis=2)
    baseline = methods.index("knn10")
    records = []
    for metric, num, den in (("recall", 0, 2), ("fpr", 1, 3), ("recall_by14", 4, 2), ("before_event_fraction", 5, 6)):
        ratio = boot[..., num] / np.maximum(boot[..., den], 1)
        point = totals[:, num] / totals[:, den]
        for mi, method in enumerate(methods):
            lo, hi = np.quantile(ratio[mi]-ratio[baseline], [.025, .975])
            records.append(dict(method=method, baseline="knn10", metric=metric,
                difference=point[mi]-point[baseline], low=lo, high=hi, tasks=len(tasks),
                bootstrap_samples=2000, unit="task with all fold appearances, stratified by suite"))
    pd.DataFrame(records).to_csv(output / "paired_task_bootstrap.csv", index=False)
    styles = {
        "knn10": ("Original kNN-10D", "#333333", "-"),
        "knn12_v8": ("kNN-12D + v8 features", "#d28430", "-"),
        "knn12_cross_task": ("kNN-12D, cross-task calibration", "#438dbe", "-"),
        "v82_frozen": ("Frozen v8.2", "#328b72", "-"),
        "knn12_cross_task_or_v82_frozen": ("Cross-task kNN-12D OR frozen v8.2", "#b84e66", "-"),
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    handles = []
    for method, (label, color, linestyle) in styles.items():
        group = data.loc[data.method.eq(method)]
        for ax, failure in zip(axes, (True, False)):
            first = group.loc[group.failure.eq(failure), "first_alarm_query"].to_numpy()
            cdf = [np.mean((first >= 0) & (first <= q)) for q in range(52)]
            line, = ax.step(np.arange(52), cdf, where="post", label=label, color=color, linestyle=linestyle, linewidth=1.8)
            if failure:
                handles.append(line)
    for ax, title in zip(axes, ("Recall by query: all 491 failures", "FPR by query: all 13,509 successes")):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Query q (zero-based; completed task actions = 10q)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.set_xlim(5, 52)
        ax.set_ylim(bottom=0)
    fig.suptitle("Cross-task calibration and the existing v8.2\nSame 14,000 test appearances; historical global v8.2 profile; OR retains branch thresholds", fontsize=12)
    fig.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=8, frameon=False)
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"cross_task_cumulative.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_json(output / "evaluation_summary.json", dict(evaluator_sha256=digest(Path(__file__)),
        sealed_manifest_sha256=digest(output / "sealed_manifest.json"),
        population="14,000 matched unseen appearances, 491 failures; repeated across folds",
        artifacts={str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
                   if path.is_file() and path.suffix in (".csv", ".png", ".pdf")}))
    print("CROSS-TASK SUMMARY COMPLETE", flush=True)


if __name__ == "__main__":
    main()
