#!/usr/bin/env python3
"""Compare deployable control policies separately from retrospective diagnostics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from adaptive_control import ARMS, PROTOCOL, noise_for
from route_noise_selector import pairwise_rms, centrality, stable_argmin
from analyze_long_continuation import cluster_interval
from collection_storage import atomic_json, digest

STATIC = ("center_once", "edge_once", "knn_once", "guarded_knn_once")


def noise_center(main_id, replicate):
    noises = np.stack([noise_for(main_id, replicate, 0, i) for i in range(4)])
    return stable_argmin(centrality(pairwise_rms(noises)))


def policy_rows(tasks, timing, name):
    rows = []
    for task in tasks:
        events = [e for e in task["events"] if e["timing"] == timing]
        if not events:
            if timing != "after_alarm":
                continue
            rows.append(dict(main_id=task["main_id"], base_task=task["base_task"], native=task["success"],
                success=float(task["success"]), random=float(task["success"]), queries=task["parent_queries"],
                baseline_queries=task["parent_queries"], eligible=False, successes=0, harms=0, repeats=0))
            continue
        if len(events) != 1:
            raise ValueError("Duplicate recovery timing")
        event = events[0]
        branches = [b for b in task["branches"] if b["event_id"] == event["event_id"]]
        pools = {p["replicate"]: p for p in task["pools"] if p["event_id"] == event["event_id"]}
        base = {b["replicate"]: b for b in branches if b["arm"] == "candidate0"}
        success, query_cost, baseline = [], [], []
        for rep in sorted(base):
            if name in STATIC or name == "noise_center_once":
                if name == "noise_center_once":
                    candidate = noise_center(task["main_id"], rep)
                else:
                    selector = name.removesuffix("_once")
                    candidate = pools[rep]["winners"][selector]
                selected = next(b for b in branches if b["replicate"] == rep and b["arm"] == "candidate%d" % candidate)
                success.append(float(selected["success"]))
                query_cost.append(event["start_query"] + selected["queries"] + (0 if name == "noise_center_once" else 3))
            elif name == "oracle_once":
                success.append(float(any(c["success"] for c in pools[rep]["candidates"])))
                query_cost.append(float("nan"))
            else:
                selected = next(b for b in branches if b["replicate"] == rep and b["arm"] == name)
                success.append(float(selected["success"]))
                query_cost.append(event["start_query"] + selected["deployment_model_queries"])
            baseline.append(float(base[rep]["success"]))
        rows.append(dict(main_id=task["main_id"], base_task=task["base_task"], native=task["success"],
            success=float(np.mean(success)), random=float(np.mean(baseline)), queries=float(np.mean(query_cost)),
            baseline_queries=float(np.mean([event["start_query"] + b["queries"] for b in base.values()])),
            eligible=True, successes=int(sum(success)), harms=int(len(success) - sum(success)), repeats=len(success)))
    return rows


def metric(tasks, timing, name, cohort):
    rows = policy_rows(tasks, timing, name)
    if not rows:
        return None
    values = np.asarray([r["success"] for r in rows])
    baseline = np.asarray([r["random"] for r in rows])
    delta = values - baseline
    failed = [r for r in rows if r["eligible"] and not r["native"]]
    successful = [r for r in rows if r["eligible"] and r["native"]]
    rescue_den = sum(r["repeats"] for r in failed)
    harm_den = sum(r["repeats"] for r in successful)
    interval = cluster_interval(delta, [r["base_task"] for r in rows])
    costs = np.asarray([r["queries"] for r in rows])
    baseline_costs = np.asarray([r["baseline_queries"] for r in rows])
    return dict(cohort=cohort, timing=timing, method=name, parents=len(rows),
        task_clusters=len({r["base_task"] for r in rows}), eligible_parents=sum(r["eligible"] for r in rows),
        success_rate=float(values.mean()), native_success_rate=float(np.mean([r["native"] for r in rows])),
        delta_vs_native_pp=float((values - np.asarray([r["native"] for r in rows])).mean() * 100),
        random_success_rate=float(baseline.mean()), delta_vs_random_pp=float(delta.mean() * 100),
        cluster_ci95_pp=None if interval is None else (np.asarray(interval) * 100).tolist(),
        rescued_repeats=sum(r["successes"] for r in failed), failed_repeats=rescue_den,
        rescue_rate=sum(r["successes"] for r in failed) / rescue_den if rescue_den else None,
        harmed_repeats=sum(r["harms"] for r in successful), successful_repeats=harm_den,
        harm_rate=sum(r["harms"] for r in successful) / harm_den if harm_den else None,
        rescued_unique_parents=sum(r["successes"] > 0 for r in failed),
        harmed_unique_parents=sum(r["harms"] > 0 for r in successful),
        changed_vs_random_parents=int(np.count_nonzero(delta)),
        mean_model_queries=float(costs.mean()) if np.isfinite(costs).all() else None,
        query_overhead_percent=float((costs.mean() / baseline_costs.mean() - 1) * 100) if np.isfinite(costs).all() else None,
        deployable=timing == "after_alarm" and name != "oracle_once")


def rank_key(row):
    return (-row["delta_vs_random_pp"], row["harm_rate"] or 0,
            row["mean_model_queries"] if row["mean_model_queries"] is not None else float("inf"), row["method"])


def pool_diagnostics(tasks):
    rows = []
    for timing in ("after_alarm", "before_alarm_diagnostic"):
        for native in (False, True):
            pools = [p for t in tasks if t["success"] == native for p in t["pools"] if p["timing"] == timing]
            if not pools:
                continue
            mixed = [p for p in pools if 0 < sum(c["success"] for c in p["candidates"]) < 4]
            rows.append(dict(timing=timing, native_success=native, pools=len(pools),
                any_success=sum(any(c["success"] for c in p["candidates"]) for p in pools),
                all_success=sum(all(c["success"] for c in p["candidates"]) for p in pools), mixed_pools=len(mixed),
                selector_successes_on_mixed={name: sum(p["candidates"][p["winners"][name]]["success"] for p in mixed)
                    for name in ("center", "edge", "knn", "guarded_knn")},
                random_successes_on_mixed=sum(p["candidates"][0]["success"] for p in mixed)))
    return rows


def temporal_selection_agreement(tasks):
    pairs = []
    noise_agreement = {name: 0 for name in ("center", "edge", "knn", "guarded_knn")}
    pools = 0
    for task in tasks:
        for pool in task["pools"]:
            candidate = noise_center(task["main_id"], pool["replicate"])
            for name in noise_agreement:
                noise_agreement[name] += pool["winners"][name] == candidate
            pools += 1
        repeats = {p["replicate"] for p in task["pools"]}
        for rep in repeats:
            times = {p["timing"]: p for p in task["pools"] if p["replicate"] == rep}
            if "after_alarm" in times and "before_alarm_diagnostic" in times:
                pairs.append((times["after_alarm"]["winners"], times["before_alarm_diagnostic"]["winners"]))
    return dict(same_noise_cross_time_pairs=len(pairs),
        same_selected_candidate={name: sum(a[name] == b[name] for a, b in pairs) for name in noise_agreement},
        total_pools=pools, agreement_with_noise_center=noise_agreement,
        interpretation="Descriptive choice agreement; identical candidate noises across time, changed state and native route history.")


def run(args):
    audits = [json.loads(path.read_text()) for path in args.audits]
    if any(a["status"] != "passed" or a["protocol"] != PROTOCOL for a in audits):
        raise ValueError("Only independently audited recovery data may be analyzed")
    tasks = [task for audit in audits for task in audit["tasks"]]
    if len(tasks) != len({task["main_id"] for task in tasks}):
        raise ValueError("Duplicate parent IDs across audits")
    available = set.intersection(*[{b["arm"] for b in t["branches"]} for t in tasks if t["branches"]])
    names = [name for name in ARMS if name in available and name not in ("candidate1", "candidate2", "candidate3")]
    names += list(STATIC) + ["noise_center_once", "oracle_once"]
    primary = [t for t in tasks if t["analysis_role"] == "perturbation"]
    cohorts = dict(primary=primary, primary_pro=[t for t in primary if t["benchmark"] == "pro"],
                   primary_plus=[t for t in primary if t["benchmark"] == "plus"],
                   controls=[t for t in tasks if t["analysis_role"] != "perturbation"])
    metrics = []
    for cohort, part in cohorts.items():
        for timing in ("after_alarm", "before_alarm_diagnostic"):
            for name in names:
                row = metric(part, timing, name, cohort)
                if row is not None:
                    metrics.append(row)
    report = dict(protocol=PROTOCOL, audits={str(p.resolve()): digest(p) for p in args.audits},
        analyzer_sha256=digest(__file__), parents=len(tasks), primary_parents=len(primary),
        stages=[a["stage"] for a in audits], metrics=metrics, pool_diagnostics=pool_diagnostics(primary),
        temporal_selection_agreement=temporal_selection_agreement(primary),
        notes=["Repeated suffixes are averaged within parent; intervals cluster by base task.",
               "Before-alarm timing and oracle are retrospective diagnostics, not deployable improvements.",
               "Development parents were selected conditional on an eligible alarm; validation includes unalarmed parents.",
               "Native and restored branches have different future random streams; paired control effects use candidate0 with the same policy/environment streams.",
               "All configurations retain the same global frozen alarm parameters."])
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "summary.json", report)
    with (args.output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    for row in metrics:
        if row["cohort"] == "primary":
            print(json.dumps({k: row[k] for k in ("timing", "method", "parents", "rescued_repeats", "failed_repeats",
                "harmed_repeats", "successful_repeats", "delta_vs_random_pp", "mean_model_queries")}))
    if args.select:
        if "validation" in report["stages"]:
            raise ValueError("Validation outcomes must not select the tested control")
        rows = [row for row in metrics if row["cohort"] == "primary" and row["timing"] == "after_alarm"]
        short = sorted([r for r in rows if r["method"].startswith("short")], key=rank_key)[0]
        noise = sorted([r for r in rows if r["method"] in STATIC or r["method"] in
                        ("center5_5", "edge5_5", "knn5_5", "guarded_knn5_5")], key=rank_key)[0]
        selected = [short["method"], noise["method"]]
        validation_arms = ["candidate%d" % i for i in range(4)] + [name for name in selected if name in ARMS]
        selection = dict(protocol=PROTOCOL, development_audits=report["audits"],
            analyzer_sha256=digest(__file__), development_summary_sha256=digest(args.output / "summary.json"),
            rule="best short-chunk and best route-selector family by paired net success; ties by harm, then inference cost, then name",
            selected_policies=selected, validation_arms=validation_arms, chosen_metrics=[short, noise],
            positive_development_effect=any(r["delta_vs_random_pp"] > 0 for r in (short, noise)),
            positive_vs_native=any(r["delta_vs_native_pp"] > 0 for r in (short, noise)),
            random_baseline="candidate0", threshold_fitting=False, validation_outcomes_read=False)
        atomic_json(args.output / "selection.json", selection)
        print("SELECTION " + json.dumps(selection))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audits", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select", action="store_true")
    run(parser.parse_args())
