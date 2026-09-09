#!/usr/bin/env python3
"""Freeze pending mains from a declared manifest partition without using outcomes."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil

from collection_protocol import (
    HERE, MANIFEST_SHA256, PARAMETERS_SHA256, PROTOCOL, REPLICATES, load_plan, verify_frozen_alarm,
)
from collection_storage import atomic_json, digest

SUITES = dict(long="libero_10", goal="libero_goal", spatial="libero_spatial", object="libero_object")


def run(args):
    source = HERE / "design/collection_manifest.csv"
    index_path = HERE / "design/collection_status.json"
    if digest(source) != MANIFEST_SHA256 or args.output.exists():
        raise ValueError("Changed source manifest or existing frozen plan")
    verify_frozen_alarm()
    index = json.loads(index_path.read_text())
    if index["source_manifest_sha256"] != MANIFEST_SHA256:
        raise ValueError("Completion index belongs to a different manifest")
    if args.selection == "long_seed_extension" and (args.model != "long" or args.benchmark == "plus"):
        raise ValueError("Long seed extension is defined for Pro Long only")
    with source.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["suite"] == SUITES[args.model]]
    if args.selection == "screen":
        rows = [row for row in rows if row["screen"] == "1"]
        selection = "All remaining frozen screening main IDs in the declared suite/benchmark; fixed-hash dispatch; no outcome selection"
        scope = "Complete the declared existing screening cohort; no full benchmark score claim"
        stage = "P1_" + args.model + "_" + args.benchmark + "_remaining"
    else:
        rows = [row for row in rows if row["benchmark"] == "pro" and row["screen"] == "0"
                and row["policy_seed_index"] == "1" and int(row["init_index"]) < 5]
        selection = "All Pro Long variants at init 0..4 and original policy seed index 1; fixed-hash dispatch; no outcome selection"
        scope = "Additional Long Pro policy seeds from the frozen coverage manifest; separate from the P1 screen"
        stage = "E1_long_policy_seed1"
    if args.benchmark != "all":
        rows = [row for row in rows if row["benchmark"] == args.benchmark]
    completed = set(index["completed"])
    pending = [row for row in rows if row["main_id"] not in completed]
    if not pending:
        raise ValueError("No pending mains in the selected manifest partition")
    controls = set(json.loads((HERE / "benchmarks/READINESS.json").read_text())
                   ["pro_environment_generation"]["bddl_unchanged_from_base"])
    tasks = sorted(pending, key=lambda row: hashlib.sha256(
        ("p1-remaining-20260908:" + row["main_id"]).encode()).hexdigest())
    for row in tasks:
        row["analysis_role"] = "unchanged_scene_control" if row["variant_id"] in controls else "perturbation"
        row["status"] = "planned"
    baseline_path = args.budget_source or HERE / "design/experiment_long_batch1_analysis_20260908.json"
    baseline = json.loads(baseline_path.read_text())
    expected_bytes = baseline["mean_bytes_per_main"] * len(tasks)
    free_bytes = shutil.disk_usage(HERE).free
    if free_bytes < (args.disk_floor_gib + 8) * 1024**3:
        raise ValueError("Insufficient disk headroom for a large concurrent run")
    if expected_bytes * 1.15 > min(args.storage_quota_gib * 1024**3, free_bytes - args.disk_floor_gib * 1024**3):
        raise ValueError("Estimated batch exceeds the storage budget")
    horizons = {int(row["horizon_steps"]) for row in tasks}
    if len(horizons) != 1:
        raise ValueError("Mixed horizons within a suite")
    plan = dict(
        schema="moe_control.run_plan.v1", protocol=PROTOCOL, model=args.model,
        stage=stage, source_manifest_sha256=MANIFEST_SHA256,
        sample_partition="screen" if args.selection == "screen" else "coverage_extension",
        selection_name=args.selection, benchmark_filter=args.benchmark,
        frozen_parameters_sha256=PARAMETERS_SHA256, completion_index_sha256=digest(index_path),
        generator_sha256=digest(Path(__file__)), replicates=REPLICATES,
        main_intervention=False, hidden_capture=False, allowed_gpus=[0, 1, 2, 3, 4, 5, 7],
        render_gpus=[0, 3, 4, 5, 7], mps=True, replicas_per_gpu=8, workers_per_gpu=8,
        batch_size=1, storage_quota_gib=args.storage_quota_gib, disk_floor_gib=args.disk_floor_gib,
        selection=selection, inference_scope=scope,
        no_alarm_c0="One native replay from q5 or q0 for fidelity QC; not an intervention event",
        branch_arms={"C0": 1, "C1": 4, "T1": 4}, checkpoint_model=args.model,
        horizon_steps=horizons.pop(), online_trigger="v7_frozen",
        branch_trigger="All first v7 alarms, regardless of final main success; after main commit and exact C0",
        offline_alarm_comparison="Same frozen v7/v8/v8.2, Euclidean/cosine kNN and k-means profiles",
        eligible_cohort_mains=len(rows), previously_completed_cohort_mains=len(rows) - len(tasks),
        cohort_counts=dict(Counter(row["benchmark"] for row in tasks)),
        role_counts=dict(Counter(row["analysis_role"] for row in tasks)),
        category_counts=dict(Counter(row["benchmark"] + ":" + row["category"] for row in tasks)),
        base_task_count=len({row["base_task"] for row in tasks}),
        storage_estimate=dict(source=str(baseline_path), source_sha256=digest(baseline_path),
            expected_bytes=expected_bytes, with_15_percent_margin=1.15 * expected_bytes,
            disk_free_bytes=free_bytes, limitation="Observed Long batch mix; not a storage upper bound or a measured estimate for other suites"),
        tasks=tasks,
    )
    atomic_json(args.output, plan)
    load_plan(args.output, args.model, pending_only=True)
    print(json.dumps(dict(plan=str(args.output), sha256=digest(args.output), mains=len(tasks),
        counts=plan["cohort_counts"], roles=plan["role_counts"],
        expected_gib=expected_bytes / 1024**3, disk_free_gib=free_bytes / 1024**3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(SUITES), default="long")
    parser.add_argument("--selection", choices=("screen", "long_seed_extension"), default="screen")
    parser.add_argument("--benchmark", choices=("all", "pro", "plus"), default="all")
    parser.add_argument("--budget-source", type=Path)
    parser.add_argument("--storage-quota-gib", type=float, default=24)
    parser.add_argument("--disk-floor-gib", type=float, default=8)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
