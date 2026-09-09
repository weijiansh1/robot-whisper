#!/usr/bin/env python3
"""Summarize paired amplitude/duration outcomes and measured gate/action changes."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest
from v8_strength_control import ARMS


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.all(values == values[0]):
        return None
    rng = np.random.default_rng(20260908)
    estimates = values[rng.integers(len(values), size=(20000, len(values)))].mean(axis=1)
    return (100 * np.quantile(estimates, [.025, .975])).tolist()


def success(job, arm):
    if arm == "historical_native":
        return int(job["reference_native_success"])
    return int(next(b["success"] for b in job["branches"] if b["arm"] == arm))


def compare(jobs, arm, reference):
    by_parent = defaultdict(list)
    wins, losses = [], []
    for job in jobs:
        delta = success(job, arm) - success(job, reference)
        by_parent[job["main_id"]].append(delta)
        if delta > 0:
            wins.append(job["main_id"])
        elif delta < 0:
            losses.append(job["main_id"])
    differences = [np.mean(v) for v in by_parent.values()]
    return dict(arm=arm, reference=reference, paired_suffixes=len(jobs), parents=len(by_parent),
        wins=len(wins), losses=len(losses), winning_parents=sorted(set(wins)), losing_parents=sorted(set(losses)),
        parent_mean_delta_pp=float(100 * np.mean(differences)) if differences else None,
        parent_bootstrap_95ci_pp=interval(differences))


def manipulation(branches, period):
    if not branches:
        return None
    values = np.asarray([b[period] for b in branches])
    return dict(branches=len(branches), branch_median=np.median(values, axis=0).tolist(),
        branch_q25=np.quantile(values, .25, axis=0).tolist(),
        branch_q75=np.quantile(values, .75, axis=0).tolist(),
        branch_mean=np.mean(values, axis=0).tolist(),
        target_direction_fractions=dict(frontback_ratio_increases=float(np.mean(values[:, 6] > 0)),
            curvature_decreases=float(np.mean(values[:, 7] < 1)),
            mobility_increases=float(np.mean(values[:, 8] > 1)),
            all_three=float(np.mean((values[:, 6] > 0) & (values[:, 7] < 1) & (values[:, 8] > 1)))),
        branches_with_gripper_sign_change=int(np.sum(values[:, 2] > 0)))


def group(jobs):
    parents = {j["main_id"]: j for j in jobs}
    failures = [j for j in jobs if not j["native_success"]]
    healthy = [j for j in jobs if j["native_success"]]
    arms = {}
    for arm, settings in ARMS.items():
        branches = [next(b for b in j["branches"] if b["arm"] == arm) for j in jobs]
        rescued = [j["main_id"] for j in failures if success(j, arm)]
        harmed = [j["main_id"] for j in healthy if not success(j, arm)]
        arms[arm] = dict(successes=sum(b["success"] for b in branches), suffixes=len(branches),
            original_failure_attempts=len(failures), rescues=len(rescued),
            original_success_attempts=len(healthy), harms=len(harmed),
            rescued_parents=sorted(set(rescued)), harmed_parents=sorted(set(harmed)),
            versus_native=compare(jobs, arm, "historical_native"),
            versus_original_strength=compare(jobs, arm, "combined1_5"),
            actual_model_queries=sum(b["actual_model_queries"] for b in branches),
            suffix_queries=sum(b["queries"] for b in branches),
            extra_shadow_queries=sum(b["controlled_queries"] for b in branches),
            controlled_query_counts=sorted(b["controlled_queries"] for b in branches),
            branches_reaching_planned_duration=sum(b["controlled_queries"] == settings["duration"] for b in branches),
            manipulation={period: manipulation(branches, period)
                          for period in ("first_query", "first5_mean", "control_mean")})
    pairs = {
        "amplitude": [compare(jobs, "combined%d_%d" % (m, d), "combined1_%d" % d)
                      for d in (5, 20) for m in (4, 16)],
        "duration": [compare(jobs, "combined%d_20" % m, "combined%d_5" % m) for m in (1, 4, 16)],
        "direction": [compare(jobs, "combined%d_20" % m, "random%d_20" % m) for m in (4, 16)]}
    return dict(parents=len(parents), jobs=len(jobs),
        original_failure_parents=sum(not j["native_success"] for j in parents.values()),
        original_success_parents=sum(j["native_success"] for j in parents.values()),
        historical_native_successes=sum(j["reference_native_success"] for j in jobs), arms=arms, comparisons=pairs)


def resources(run):
    summary = json.loads((run / "summary.json").read_text())
    samples = json.loads((run / "gpu_samples.json").read_text())
    times = np.asarray([s["monotonic"] for s in samples])
    weights = np.diff(times)
    output = {}
    for gpu in (0, 1, 2, 3):
        rows = [next(r for r in s["gpus"] if r["gpu"] == gpu) for s in samples]
        busy = np.asarray([s["active_tasks_by_gpu"][str(gpu)] == 8 for s in samples[:-1]])
        output[str(gpu)] = {}
        for key in ("utilization_percent", "power_w", "memory_mib"):
            values = np.asarray([r[key] for r in rows])
            output[str(gpu)][key] = dict(time_weighted_mean=float(np.average(values[:-1], weights=weights)),
                median=float(np.median(values)), p95=float(np.quantile(values, .95)), maximum=float(values.max()),
                eight_active_time_weighted_mean=(float(np.average(values[:-1][busy], weights=weights[busy]))
                    if np.any(busy) else None))
        output[str(gpu)]["eight_active_seconds"] = float(weights[busy].sum())
    return dict(collection_seconds=summary["collection_elapsed_seconds"],
        sampling_model_queries=summary["actual_model_queries"], queries_per_second=summary["queries_per_second"],
        storage_bytes=summary["storage_bytes"], telemetry_span_seconds=float(weights.sum()), gpus=output,
        temporary_models_stopped=summary["temporary_models_stopped"],
        environment_workers_stopped=summary["environment_workers_stopped"],
        live_replica_pids_after_cleanup=summary["live_replica_pids_after_cleanup"])


def run(args):
    audit = json.loads(args.audit.read_text())
    if audit["status"] != "passed":
        raise ValueError("A passed complete audit is required")
    jobs = audit["jobs"]
    parent_repeats = defaultdict(set)
    for job in jobs:
        if job["replicate"] in parent_repeats[job["main_id"]]:
            raise ValueError("Duplicate state/repeat")
        parent_repeats[job["main_id"]].add(job["replicate"])
    if any(repeats != {0, 1} for repeats in parent_repeats.values()):
        raise ValueError("Incomplete paired repeats")
    primary = [j for j in jobs if j["analysis_role"] == "perturbation"]
    result = dict(status="completed", audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
        analyzer_sha256=digest(__file__), stage="exploratory_amplitude_duration_test",
        primary_unit="original main_id; two repeats are paired attempts, not independent new states",
        scope="24 alarm-conditioned Long Pro/Plus states, stratified by benchmark/head/original outcome; no new main coverage",
        baseline="historical paired native suffix with identical noise/seeds; combined1_5 fully reproduced in this run",
        duration="at most 5/20 queries, bounded by success or original 520-step horizon; unchanged chunk10",
        feature_comparison="same-observation same-noise native shadow; raw features, not recalibrated alarm scores",
        feature_columns=audit["feature_columns"],
        feature_aggregation="first query or within-branch temporal mean, then descriptive cross-branch median; no independence assumed",
        direction_control="expert-permuted combined bias with identical per-position amplitude at same input; paths diverge later",
        multiple_comparisons="prespecified 8 arms; descriptive exploration, no validated winner selected",
        degenerate_bootstrap="constant paired parent differences yield null intervals; zero events do not prove equivalence or safety",
        all=group(jobs), primary=group(primary),
        by_benchmark={name: group([j for j in primary if j["benchmark"] == name]) for name in ("pro", "plus")},
        by_alarm_head={name: group([j for j in primary if name in j["active_heads"]])
                       for name in ("freeze", "turbulence", "balance", "curvature")},
        by_alarm_time={"q_le_20": group([j for j in primary if j["first_v8_alarm"] <= 20]),
                       "q_gt_20": group([j for j in primary if j["first_v8_alarm"] > 20])},
        new_main_coverage=0, branches=audit["branches"], resources=resources(Path(audit["run"])))
    if sum(r["actual_model_queries"] for r in result["all"]["arms"].values()) != audit["actual_model_queries"]:
        raise ValueError("Analysis query ledger differs")
    atomic_json(args.output, result)
    print(json.dumps(dict(parents=result["all"]["parents"], primary_parents=result["primary"]["parents"],
        arms={arm: {k: r[k] for k in ("rescues", "original_failure_attempts", "harms", "original_success_attempts")}
              for arm, r in result["primary"]["arms"].items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
