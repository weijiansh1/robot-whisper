#!/usr/bin/env python3
"""Freeze outcome-independent development or disjoint validation parent sets."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil

from adaptive_control import (ARMS, CONTRACT, GPUS, RENDER_GPUS, PROTOCOL, REFERENCE, PARAMETERS,
                              events_for, branch_bound, load_plan)
from collection_protocol import HERE, PARAMETERS_SHA256, stable_id
from collection_storage import atomic_json, digest


def run(args):
    development = args.stage in ("development", "smoke")
    stem = "experiment_long_batch1" if development else "experiment_long_scale"
    audit_path = HERE / ("design/%s_audit_20260908.json" % stem)
    alarm_dir = HERE / ("design/%s_alarm_comparison_20260908" % stem)
    audit = json.loads(audit_path.read_text())
    with (alarm_dir / "first_alarms.csv").open(newline="") as stream:
        alarms = {row["main_id"]: row for row in csv.DictReader(stream)}
    groups = defaultdict(list)
    for parent in audit["tasks"]:
        if development and not 0 <= int(alarms[parent["main_id"]]["knn20"]) < parent["main_queries"] - 1:
            continue
        groups[parent["benchmark"], parent["category"]].append(parent)
    chosen = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda t: stable_id(PROTOCOL, "parent_selection", t["main_id"]))
        if len(group) < args.per_category:
            raise ValueError("Too few eligible parents in " + str(key))
        chosen.extend(group[:args.per_category])
    if args.stage == "smoke":
        chosen = sorted(chosen, key=lambda t: (t["benchmark"], t["category"]))
        chosen = [chosen[0], chosen[1], next(t for t in chosen if t["benchmark"] == "pro"),
                  next(t for t in chosen if t["benchmark"] == "pro" and t["category"] == "Task")]
    excluded = set()
    if args.exclude_plan:
        excluded = {t["main_id"] for t in json.loads(args.exclude_plan.read_text())["tasks"]}
        chosen = [t for t in chosen if t["main_id"] not in excluded]
    arms = list(ARMS) if args.arms is None else args.arms
    diagnostic = development
    tasks = []
    for parent in chosen:
        directory = Path(parent["directory"])
        original = json.loads((directory / "result.json").read_text())
        task = {key: parent[key] for key in ("main_id", "variant_id", "benchmark", "category", "analysis_role")}
        task.update(parent_directory=str(directory), parent_queries=parent["main_queries"],
            parent_commit_sha256=digest(directory / "main_complete.json"),
            parent_manifest_sha256=digest(directory / "main/manifest.json"),
            noise_seed=original["seed"], init_index=original["init_index"],
            first_alarm=int(alarms[parent["main_id"]]["knn20"]))
        task["events"] = events_for(task["main_id"], task["first_alarm"], task["parent_queries"], diagnostic)
        task["max_output_bytes"] = task["parent_queries"] * 190 * 1024 + (len(task["events"]) + 1) * 1024**2
        tasks.append(task)
    sources = [audit_path, alarm_dir / "first_alarms.csv", alarm_dir / "verification.json", alarm_dir / "contract.json",
               REFERENCE, PARAMETERS, HERE / "adaptive_control.py", HERE / "prepare_adaptive_control.py"]
    if args.exclude_plan:
        sources.append(args.exclude_plan.resolve())
    if args.selection:
        sources.append(args.selection.resolve())
    bound = sum(t["max_output_bytes"] + args.replicates * sum(branch_bound(e, arms) for e in t["events"]) for t in tasks)
    if shutil.disk_usage(HERE).free - bound < 8 * 1024**3:
        raise ValueError("Recovery plan exceeds conservative disk headroom: %.2f GiB" % (bound / 1024**3))
    plan = dict(protocol=PROTOCOL, model="long", stage=args.stage, arms=arms, arms_registry=ARMS,
        replicates=args.replicates, candidates=4, diagnostic=diagnostic, contract=CONTRACT,
        selection="fixed hash within benchmark/category; development conditions on an eligible alarm, never on outcome; validation includes unalarmed parents",
        selection_record=str(args.selection.resolve()) if args.selection else None,
        parent_audit=str(audit_path), alarm_table=str(alarm_dir / "first_alarms.csv"),
        alarm_verification=str(alarm_dir / "verification.json"), alarm_contract=str(alarm_dir / "contract.json"),
        source_sha256={str(p): digest(p) for p in sources}, frozen_parameters_sha256=PARAMETERS_SHA256,
        tasks=tasks, allowed_gpus=GPUS, render_gpus=RENDER_GPUS, replicas_per_gpu=args.replicas,
        workers_per_gpu=args.replicas, mps=True, parameter_fitting=False, repeated_parents_not_new_coverage=True,
        benchmark_counts=dict(Counter(t["benchmark"] for t in tasks)),
        event_states=sum(len(t["events"]) for t in tasks), maximum_output_bytes=bound,
        storage_quota_gib=14, disk_floor_gib=8)
    if args.output.exists():
        raise ValueError("Refusing to replace a frozen plan")
    atomic_json(args.output, plan)
    load_plan(args.output)
    print(json.dumps(dict(plan=str(args.output), sha256=digest(args.output), parents=len(tasks),
        events=plan["event_states"], branches=plan["event_states"] * args.replicates * len(arms),
        maximum_gib=bound / 1024**3)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "development", "validation"), required=True)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--replicas", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS))
    parser.add_argument("--exclude-plan", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
