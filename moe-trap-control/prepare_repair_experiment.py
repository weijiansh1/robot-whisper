#!/usr/bin/env python3
"""Freeze the parent set and fork events for the late-alarm repair experiment (no outcome beyond failed/success)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import shutil

from repair_control import (ARMS, BYTES_PER_QUERY, CONTRACT, CONTRACT_V2, CONTRACT_V3, CONTROLLER, EPISODE_ARMS, GPUS,
                            RENDER_GPUS, PROTOCOL, PROTOCOL_V2, PROTOCOL_V3, REFERENCE, PARAMETERS, REGULATOR,
                            REGULATOR_ARMS, SUPERVISOR, SUPERVISOR_ARMS, branch_bound, episode_bound, events_for,
                            load_plan)
from collection_protocol import HERE, PARAMETERS_SHA256, stable_id
from collection_storage import atomic_json, digest

AUDITS = ("experiment_long_scale", "experiment_long_batch1")
ALARM_DIR = HERE / "design/experiment_long_screen_summary_20260908"
PICK_PLACE_BASES = (
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
)
LATE_ELIGIBLE_ALARM = 44


def base_task(task_name):
    stem = task_name.split("_view_")[0]
    for base in PICK_PLACE_BASES:
        if stem == base or stem.startswith(base + "_"):
            return base
    return None


def supervisor_plan(args):
    """v2: reuse the audited v1 run's parents, fork snapshots and fork physics; only the arms change."""
    run_dir, audit_path = Path(args.replay_run), Path(args.replay_audit)
    v1 = json.loads((run_dir / "plan.json").read_text())
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "passed" or audit["plan_sha256"] != digest(run_dir / "plan.json") or v1["protocol"] != PROTOCOL:
        raise ValueError("Supervisor plan needs an audited v1 run")
    tasks = [dict(t) for t in v1["tasks"]]
    if args.stage in ("supervisor_smoke", "regulator_smoke"):
        wanted = {b: 0 for b in args.smoke_bases}
        chosen = []
        for task in tasks:
            if args.smoke_parents:
                if task["main_id"] in args.smoke_parents:
                    chosen.append(task)
            elif task["failed"] and task["base_task"] in wanted and wanted[task["base_task"]] < 1:
                wanted[task["base_task"]] += 1
                chosen.append(task)
        if not args.smoke_parents:
            chosen += [t for t in tasks if not t["failed"]][:1]
        tasks = chosen
    regulator = args.stage.startswith("regulator")
    arms = list(REGULATOR_ARMS) if regulator else list(SUPERVISOR_ARMS)
    episode_arms = list(EPISODE_ARMS) if regulator else []
    for task in tasks:
        task["max_output_bytes"] = 0
    bound = sum(branch_bound(arms, args.replicates) + (episode_bound(episode_arms, args.replicates) if episode_arms else 0)
                for _ in tasks)
    if shutil.disk_usage(HERE).free - bound < 8 * 1024**3:
        raise ValueError("Derived plan exceeds conservative disk headroom: %.2f GiB" % (bound / 1024**3))
    sources = [run_dir / "plan.json", audit_path, REFERENCE, PARAMETERS, HERE / "repair_control.py",
               HERE / "repair_controller.py", HERE / "collect_repair_control.py", HERE / "prepare_repair_experiment.py"]
    plan = dict(protocol=PROTOCOL_V3 if regulator else PROTOCOL_V2, model="long", stage=args.stage, arms=arms,
        arms_registry=REGULATOR_ARMS if regulator else SUPERVISOR_ARMS, episode_arms=episode_arms,
        episode_arms_registry=EPISODE_ARMS if regulator else {},
        controller=CONTROLLER, supervisor=None if regulator else SUPERVISOR, regulator=REGULATOR if regulator else None,
        replicates=args.replicates, contract=CONTRACT_V3 if regulator else CONTRACT_V2,
        timings=list(args.timings), replay_run=str(run_dir), replay_audit=str(audit_path),
        selection="every parent of the audited v1 run (%s); fork snapshots and physics reused; no outcome read" % v1["stage"],
        source_sha256={str(p): digest(p) for p in sources}, frozen_parameters_sha256=PARAMETERS_SHA256,
        tasks=tasks, allowed_gpus=list(GPUS), render_gpus=list(RENDER_GPUS), replicas_per_gpu=args.replicas,
        workers_per_gpu=args.replicas, mps=True, parameter_fitting=False, training=False,
        base_task_counts=dict(Counter(t["base_task"] for t in tasks)),
        failed_parents=sum(t["failed"] for t in tasks), success_parents=sum(not t["failed"] for t in tasks),
        planned_events=sum(len(t["events"]) for t in tasks), maximum_output_bytes=bound,
        storage_quota_gib=args.storage_quota_gib, disk_floor_gib=8)
    if args.output.exists():
        raise ValueError("Refusing to replace a frozen plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, plan)
    load_plan(args.output)
    print(json.dumps(dict(plan=str(args.output), sha256=digest(args.output), parents=len(tasks), arms=arms,
        failed=plan["failed_parents"], successes=plan["success_parents"], maximum_gib=bound / 1024**3)))


def run(args):
    if args.stage.startswith(("supervisor", "regulator")):
        return supervisor_plan(args)
    parents, sources = [], []
    for stem in AUDITS:
        path = HERE / ("design/%s_audit_20260908.json" % stem)
        audit = json.loads(path.read_text())
        if audit["status"] != "passed":
            raise ValueError("Unverified parent audit: " + stem)
        parents.extend(audit["tasks"])
        sources.append(path)
    with (ALARM_DIR / "first_alarms.csv").open(newline="") as stream:
        alarms = {row["main_id"]: row for row in csv.DictReader(stream)}
    sources += [ALARM_DIR / "first_alarms.csv", ALARM_DIR / "verification.json", REFERENCE, PARAMETERS,
                HERE / "repair_control.py", HERE / "repair_controller.py", HERE / "collect_repair_control.py",
                HERE / "prepare_repair_experiment.py"]
    failed_groups, success_groups = defaultdict(list), defaultdict(list)
    for parent in parents:
        base = base_task(parent["task_name"])
        alarm = alarms.get(parent["main_id"])
        if base is None or alarm is None or parent["suite"] != "libero_10":
            continue
        if alarm["analysis_role"] != "perturbation":
            continue
        knn = int(alarm["knn20"])
        if alarm["failure"] == "True":
            if 0 <= knn <= LATE_ELIGIBLE_ALARM and int(parent["main_queries"]) == 52:
                failed_groups[base].append(parent)
        elif knn >= 0:
            success_groups[base].append(parent)

    def ordered(group):
        return sorted(group, key=lambda t: stable_id(PROTOCOL, "parent_selection", t["main_id"]))

    chosen = []
    if args.stage == "smoke":
        bases = [b for b in PICK_PLACE_BASES if failed_groups.get(b)][:3]
        chosen = [(ordered(failed_groups[b])[0], True) for b in bases]
    else:
        for base in PICK_PLACE_BASES:
            chosen += [(t, True) for t in ordered(failed_groups.get(base, []))[:args.failed_per_task]]
            chosen += [(t, False) for t in ordered(success_groups.get(base, []))[:args.success_per_task]]
    if not chosen:
        raise ValueError("No eligible parents")
    tasks = []
    for parent, failed in chosen:
        directory = Path(parent["directory"])
        original = json.loads((directory / "result.json").read_text())
        if bool(original["success"]) == failed:
            raise ValueError("Outcome label mismatch for " + parent["main_id"])
        first = int(alarms[parent["main_id"]]["knn20"])
        task = {key: parent[key] for key in ("main_id", "variant_id", "benchmark", "category")}
        task.update(analysis_role=alarms[parent["main_id"]]["analysis_role"], base_task=base_task(parent["task_name"]),
            parent_directory=str(directory), parent_queries=int(parent["main_queries"]),
            parent_commit_sha256=digest(directory / "main_complete.json"),
            parent_manifest_sha256=digest(directory / "main/manifest.json"),
            noise_seed=original["seed"], init_index=original["init_index"], first_alarm=first, failed=failed,
            events=events_for(parent["main_id"], first, int(parent["main_queries"]), failed))
        task["max_output_bytes"] = task["parent_queries"] * BYTES_PER_QUERY + 4 * 1024**2
        tasks.append(task)
    branch_bytes = sum(branch_bound(list(ARMS), args.replicates) for _ in tasks)
    bound = sum(t["max_output_bytes"] for t in tasks) + branch_bytes
    if shutil.disk_usage(HERE).free - bound < 8 * 1024**3:
        raise ValueError("Repair plan exceeds conservative disk headroom: %.2f GiB" % (bound / 1024**3))
    plan = dict(protocol=PROTOCOL, model="long", stage=args.stage, arms=list(ARMS), arms_registry=ARMS,
        controller=CONTROLLER, replicates=args.replicates, contract=CONTRACT,
        selection="fixed hash within pick-place base task; failed parents need a kNN-20 alarm at q<=44 and a full 52-query run; "
                  "success parents need a kNN-20 false alarm; nothing else about outcomes or branches is read",
        source_sha256={str(p): digest(p) for p in sources}, frozen_parameters_sha256=PARAMETERS_SHA256,
        tasks=tasks, allowed_gpus=list(GPUS), render_gpus=list(RENDER_GPUS), replicas_per_gpu=args.replicas,
        workers_per_gpu=args.replicas, mps=True, parameter_fitting=False, training=False,
        base_task_counts=dict(Counter(t["base_task"] for t in tasks)),
        failed_parents=sum(t["failed"] for t in tasks), success_parents=sum(not t["failed"] for t in tasks),
        planned_events=sum(len(t["events"]) for t in tasks), maximum_output_bytes=bound,
        storage_quota_gib=args.storage_quota_gib, disk_floor_gib=8)
    if args.output.exists():
        raise ValueError("Refusing to replace a frozen plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, plan)
    load_plan(args.output)
    print(json.dumps(dict(plan=str(args.output), sha256=digest(args.output), parents=len(tasks),
        failed=plan["failed_parents"], successes=plan["success_parents"], planned_events=plan["planned_events"],
        maximum_gib=bound / 1024**3, eligible_failed={b: len(v) for b, v in failed_groups.items()},
        eligible_success={b: len(v) for b, v in success_groups.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "main", "supervisor_smoke", "supervisor", "regulator_smoke", "regulator"), required=True)
    parser.add_argument("--replay-run", default=str(HERE / "design/repair_main_20260908"))
    parser.add_argument("--replay-audit", default=str(HERE / "design/repair_main_audit_20260908.json"))
    parser.add_argument("--timings", nargs="+", default=["early", "mid", "late"])
    parser.add_argument("--smoke-parents", nargs="*", default=[])
    parser.add_argument("--smoke-bases", nargs="+", default=[
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"])
    parser.add_argument("--failed-per-task", type=int, default=6)
    parser.add_argument("--success-per-task", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--replicas", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--storage-quota-gib", type=float, default=9)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
