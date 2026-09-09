#!/usr/bin/env python3
"""Report task utility separately from v8 suppression and population convergence."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from analyze_long_continuation import cluster_interval
from analyze_v8_strength import resources
from collection_storage import atomic_json, digest
from v8_closed_loop import PROTOCOL, ARMS


def summarize(tasks):
    output = {}
    failures = sum(not t["native_success"] for t in tasks)
    lookup = {t["main_id"]: {b["arm"]: b for b in t["branches"]} for t in tasks}
    for arm in ARMS:
        values = [lookup[t["main_id"]][arm] for t in tasks]
        base = [lookup[t["main_id"]]["native"] for t in tasks]
        rescue = [t["main_id"] for t, b in zip(tasks, values) if not t["native_success"] and b["success"]]
        harm = [t["main_id"] for t, b in zip(tasks, values) if t["native_success"] and not b["success"]]
        comparison = "native" if arm == "native" else arm.split("_")[0]+"_random"
        controls = [lookup[t["main_id"]][comparison] for t in tasks]
        deltas = np.array([int(a["success"])-int(b["success"]) for a, b in zip(values, controls)])
        interval = cluster_interval(deltas, [t["base_task"] for t in tasks]) if tasks else None
        total_queries = sum(t["start_query"]+b["actual_model_queries"] for t, b in zip(tasks, values))
        baseline_queries = sum(t["start_query"]+b["actual_model_queries"] for t, b in zip(tasks, base))
        pools = [p for b in values for p in b["pool_diagnostics"]]
        output[arm] = dict(parents=len(tasks), failed_parents=failures, successful_parents=len(tasks)-failures,
            rescues=len(rescue), harms=len(harm), rescued_main_ids=rescue, harmed_main_ids=harm,
            successes=sum(b["success"] for b in values), net_rescues=len(rescue)-len(harm),
            comparison=comparison, paired_wins=int((deltas > 0).sum()), paired_losses=int((deltas < 0).sum()),
            conditional_delta_pp=float(deltas.mean()*100) if tasks else None,
            cluster_ci95_pp=None if interval is None else (100*np.asarray(interval)).tolist(),
            candidate_pools=len(pools), changed_chunks=sum(b["changed_chunks"] for b in values),
            selected_below_target=sum(b["selected_below_target"] for b in values),
            selected_below_alarm=sum(b["selected_below_alarm"] for b in values),
            pool_below_target=sum(p["below_target"] == 16 for p in pools),
            initial_population_low_fraction=float(np.mean([p["original_population_below_target"]/8 for p in pools])) if pools else None,
            second_population_low_fraction=float(np.mean([p["second_population_below_target"]/8 for p in pools])) if pools else None,
            mean_selected_minus_default=float(np.mean([p["selected"]-p["default"] for p in pools])) if pools else None,
            dominant_default=dict(Counter(p["dominant_default"] for p in pools)),
            dominant_selected=dict(Counter(p["dominant_selected"] for p in pools)),
            exits=dict(Counter(b["exit_reason"] for b in values if b["exit_reason"] is not None)),
            post_exit_recurrences=sum(b["post_exit_recurrence"] for b in values),
            full_policy_model_queries=total_queries, baseline_model_queries=baseline_queries,
            query_overhead_percent=100*(total_queries/baseline_queries-1) if baseline_queries else None,
            mean_full_action_steps=float(np.mean([b["final_action_steps"] for b in values])) if values else None)
    return output


def plot(tasks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(native="#697780", iid_random="#a86343", iid_v8="#007e91", guided_random="#86743d", guided_v8="#b74368")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for arm in ARMS:
        branches = [next(b for b in t["branches"] if b["arm"] == arm) for t in tasks]
        medians = []
        for index in range(12):
            values = [r["selected"] for b in branches for r in b["risk_trace"] if r["index"] == index and r["selected"] is not None]
            medians.append(float(np.median(values)) if values else np.nan)
        axes[0].plot(range(12), medians, label=arm, color=colors[arm], marker=".")
        if arm != "native":
            counts = []
            for index in range(12):
                pools = [p for b in branches for p in b["pool_diagnostics"] if p["index"] == index]
                counts.append(np.mean([p["below_target"]/16 for p in pools]) if pools else np.nan)
            axes[1].plot(range(12), counts, label=arm, color=colors[arm], marker=".")
    axes[0].axhline(-1, color="#222222", linestyle="--", linewidth=1, label="recovery target")
    axes[0].set(xlabel="Executed query after fork", ylabel="Median selected v8 margin score", title="Internal score; observed suffixes only")
    axes[1].set(xlabel="Executed query after fork", ylabel="Fraction of all candidates below target", ylim=(-.03, 1.03),
                title="Unfiltered populations; observed pools only")
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8)
    figure.savefig(output / "internal_scores.png", dpi=180)
    figure.savefig(output / "internal_scores.pdf")
    plt.close(figure)


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed" or audit["protocol"] != PROTOCOL:
        raise ValueError("Only complete audited runs may be analyzed")
    tasks = audit["tasks"]
    summary = summarize(tasks)
    qualifying = [arm for arm in ("iid_v8", "guided_v8") if summary[arm]["net_rescues"] > 0 and
                  summary[arm]["paired_wins"] > summary[arm]["paired_losses"]]
    result = dict(protocol=PROTOCOL, stage=audit["stage"], audit=str(args.audit.resolve()),
        audit_sha256=digest(args.audit), analyzer_sha256=digest(__file__),
        parents=len(tasks), summary=summary,
        by_benchmark={name: summarize([t for t in tasks if t["benchmark"] == name]) for name in ("pro", "plus")},
        continuation_gate=dict(passed=bool(qualifying), qualifying_arms=qualifying,
            rule="positive rescue-minus-harm and positive paired wins-minus-losses against the matched random chooser"),
        resources=resources(Path(audit["run"])),
        notes=["Conditional v8-alarmed recovery cohort deliberately stratified by original success; not overall deployment success.",
               "Native is a complete exact replay; extra candidate noises do not advance the native policy RNG.",
               "Four search arms have equal per-active-query forward budgets; exit/success can change total costs.",
               "Internal minimization, population threshold confirmation and task recovery are separate outcomes.",
               "Candidate generation does not run simulator lookahead or use outcomes/hidden states.",
               "Historical native states were known, current closed-loop outcomes are new; confirmation states reserved before outcomes.",
               "Candidate accounting uses audited shard counts and actual_model_queries, not inherited legacy candidate counters.",
               "Intervals are descriptive task-cluster bootstrap; no multiple-comparison adjustment; tiny feasibility cohort."])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "summary.json", result)
    rows = []
    for arm, metrics in summary.items():
        rows.append(dict(arm=arm, **{key: metrics[key] for key in ("parents", "failed_parents", "successful_parents",
            "rescues", "harms", "paired_wins", "paired_losses", "candidate_pools", "changed_chunks",
            "selected_below_target", "pool_below_target", "full_policy_model_queries", "query_overhead_percent")}))
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(tasks, args.output)
    print(json.dumps(dict(stage=audit["stage"], metrics=rows, continuation_gate=result["continuation_gate"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
