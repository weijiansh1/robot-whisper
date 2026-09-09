#!/usr/bin/env python3
"""Export audited outcomes, paired task-cluster intervals and inference costs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest
from continuation_experiment import ARMS, METHODS, REPLICATES


def estimates(tasks, method, arm):
    rates, random_rates, queries = [], [], []
    for task in tasks:
        event = next((e for e in task["events"] if method in e["methods"]), None)
        if event is None:
            rates.append(float(task["success"]))
            random_rates.append(float(task["success"]))
            queries.append(float(task["c0_queries"]))
            continue
        branches = [b for b in task["branches"] if b["event_id"] == event["event_id"]]
        chosen = [b for b in branches if b["arm"] == arm]
        baseline = [b for b in branches if b["arm"] == "random"]
        if len(chosen) != REPLICATES or len(baseline) != REPLICATES:
            raise ValueError("Incomplete audited branch pairing")
        rates.append(np.mean([b["success"] for b in chosen]))
        random_rates.append(np.mean([b["success"] for b in baseline]))
        queries.append(event["start_query"] + np.mean([b["queries"] for b in chosen]) + (3 if arm == "edge_noise" else 0))
    return np.asarray(rates), np.asarray(random_rates), np.asarray(queries)


def cluster_interval(values, groups):
    names = sorted(set(groups))
    if len(names) < 2 or np.all(values == values[0]):
        return None
    totals = np.asarray([values[np.asarray(groups) == group].sum() for group in names])
    counts = np.asarray([sum(value == group for value in groups) for group in names])
    draws = np.random.default_rng(20260908).integers(0, len(names), size=(10000, len(names)))
    bootstrap = totals[draws].sum(1) / counts[draws].sum(1)
    return np.quantile(bootstrap, [.025, .975]).tolist()


def run(audit_path, output):
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed":
        raise ValueError("Only a fully passed experiment may be analyzed")
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for metric in audit["metrics"]:
        row = dict(metric)
        tasks = [t for t in audit["tasks"] if t["analysis_role"] == "perturbation" and
                 (row["benchmark"] == "all" or t["benchmark"] == row["benchmark"])]
        if tasks:
            rates, baseline, queries = estimates(tasks, row["method"], row["arm"])
            np.testing.assert_allclose(rates.mean(), row["policy_success_rate"], rtol=0, atol=1e-15)
            np.testing.assert_allclose((rates - baseline).mean(), row["policy_delta_vs_random"], rtol=0, atol=1e-15)
            row["base_task_clusters"] = len({t["base_task"] for t in tasks})
            interval = cluster_interval(rates - baseline, [t["base_task"] for t in tasks])
            row["policy_delta_cluster_ci_low"], row["policy_delta_cluster_ci_high"] = interval or (None, None)
            row["mean_deployment_queries"] = float(queries.mean())
        rows.append(row)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    branch_rows = [{key: t[key] for key in ("main_id", "benchmark", "category", "analysis_role", "base_task", "success")} |
                   {"native_success": t["success"], **b} for t in audit["tasks"] for b in t["branches"]]
    with (output / "branches.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(branch_rows[0]))
        writer.writeheader()
        writer.writerows(branch_rows)
    pairs = [pair for task in audit["tasks"] for pair in task["pairs"]]
    result = dict(status="completed", audit=str(audit_path.resolve()), audit_sha256=digest(audit_path),
        analyzer_sha256=digest(Path(__file__)), metrics=rows,
        inference_cost="deployment costs include original prefix, branch suffix and three extra candidate forwards for edge_noise; actual shared collection calls reported separately",
        interval="95% percentile bootstrap over base tasks, 10000 draws, seed 20260908; exploratory, no multiple-comparison adjustment; unavailable with fewer than two clusters or no observed effect variation",
        unique_parent_coverage_added=0, reused_parents=audit["reused_parents"],
        primary_parents=audit["primary_parents"], event_states=audit["event_states"],
        paired_branches=audit["paired_branches"], actual_model_queries=audit["actual_model_queries"],
        short_chunk_limits=dict(Counter(pair["short_limit"] for pair in pairs)),
        edge_selected_candidates=dict(Counter(pair["selected_edge"] for pair in pairs)),
        edge_changed_actions=sum(pair["edge_action_rms"] > 0 for pair in pairs),
        swap_changed_actions=sum(pair["swap_action_rms"] > 0 for pair in pairs), paired_pools=len(pairs),
        runs=audit["runs"])
    plot(rows, output)
    result["artifacts"] = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
    atomic_json(output / "summary.json", result)
    print(json.dumps({key: result[key] for key in ("status", "reused_parents", "event_states", "paired_branches", "actual_model_queries", "short_chunk_limits", "edge_selected_candidates")}))


def plot(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = {(row["method"], row["arm"]): row for row in rows if row["benchmark"] == "all"}
    colors = dict(random="#5c6570", short_chunk="#c75452", edge_noise="#178c83", swap1="#486fa8")
    names = dict(random="Random noise", short_chunk="Short chunk", edge_noise="Peripheral noise", swap1="Expert swap")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(METHODS))
    for i, arm in enumerate(ARMS):
        values = [100 * (selected[method, arm]["rescue_rate"] or 0) for method in METHODS]
        axes[0].bar(x + (i - 1.5) * .18, values, width=.17, color=colors[arm], label=names[arm])
    axes[0].set_ylabel("Rescue rate on failed eligible states (%)")
    axes[0].set_ylim(0, max(5, axes[0].get_ylim()[1]))
    if not any(row["rescue_successes"] for row in selected.values()):
        axes[0].text(.5, .45, "No observed rescues", transform=axes[0].transAxes, ha="center", color="#555555")
    any_interval = False
    for i, arm in enumerate(ARMS[1:]):
        values = np.asarray([100 * selected[method, arm]["policy_delta_vs_random"] for method in METHODS])
        lower = np.asarray([100 * selected[method, arm]["policy_delta_cluster_ci_low"] if
                            selected[method, arm]["policy_delta_cluster_ci_low"] is not None else np.nan for method in METHODS])
        upper = np.asarray([100 * selected[method, arm]["policy_delta_cluster_ci_high"] if
                            selected[method, arm]["policy_delta_cluster_ci_high"] is not None else np.nan for method in METHODS])
        positions = x + (i - 1) * .18
        finite = np.isfinite(lower) & np.isfinite(upper)
        any_interval = any_interval or bool(finite.any())
        axes[1].vlines(positions[finite], lower[finite], upper[finite], color=colors[arm], linewidth=1.3)
        axes[1].scatter(positions, values, color=colors[arm], s=22, label=names[arm])
    axes[1].axhline(0, color="#555555", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Policy success vs random (percentage points)")
    labels = ["v7", "v8", "kNN Euclidean", "kNN Eucl. OR cos."]
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Four repeats averaged per state")
    axes[1].set_title("95% task-cluster intervals; exploratory" if any_interval else "Intervals unavailable: no observed effect variation")
    fig.suptitle("Long next-query interventions, primary perturbation parents", fontsize=12)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / ("interventions." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.audit, args.output)
