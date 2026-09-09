#!/usr/bin/env python3
"""Resolve v8.2 events in separate prior and prospectively frozen fresh cohorts."""

import argparse
import copy
import json
import math
from pathlib import Path

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from control_bank import PROTOCOL, SETTINGS, ARMS
from native_long_runtime import verify_source
from v82_closed_loop import V82TriggerMonitor


def prepare(args):
    verify_frozen_alarm()
    if args.output.exists():
        raise ValueError("Cannot overwrite a frozen intervention plan")
    old = json.loads((HERE / "design/native_long_modes_plan_20260909.json").read_text())
    fresh = json.loads((args.fresh_run / "plan.json").read_text())
    fresh_summary = json.loads((args.fresh_run / "summary.json").read_text())
    if fresh_summary["status"] != "completed" or fresh["bank_settings"] != SETTINGS or fresh["bank_arms"] != list(ARMS):
        raise ValueError("Fresh mains must complete with the prospectively frozen controllers")
    if len(fresh["cohort"]) != 50 or len(old["cohort"]) != 100:
        raise ValueError("Prespecified independent cohort sizes changed")
    for plan in (old, fresh):
        for path, expected in plan["source_sha256"].items():
            if digest(path) != expected:
                raise ValueError("Frozen source changed: "+path)
    cohort, checked_queries = [], 0
    for label, parents in (("development", old["cohort"]), ("fresh", fresh["cohort"])):
        for parent in parents:
            task = copy.deepcopy(parent)
            directory = Path(task["parent_directory"])
            for name, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
                if digest(directory / name) != task[key]:
                    raise ValueError("Main commitment changed")
            live = V82TriggerMonitor()
            for row in records(directory / "main"):
                live.update(row[PROBS_KEY], row["proprio"][:3])
                checked_queries += 1
            if any(live.first[key] != value for key, value in task["first_alarms"].items()):
                raise ValueError("Frozen first alarm changed")
            task.update(first_alarms=live.first, events=[], cohort_group=label)
            q0 = live.first["v82_frozen"]+1
            if q0 > 0 and q0 < task["parent_queries"]:
                task["events"] = [dict(event_id=stable_id(PROTOCOL, task["main_id"], q0),
                    start_query=q0, alarm_query=q0-1, methods=["v82_frozen"], deployable=True)]
                task["maximum_queries"] = task["parent_queries"]+len(ARMS)*(52-q0+8)
            cohort.append(task)
    if len({t["main_id"] for t in cohort}) != len(cohort):
        raise ValueError("Cohorts overlap")
    tasks = [task for task in cohort if task["events"]]
    runtime = list(dict.fromkeys(old["runtime_sources"]+fresh["runtime_sources"]+[
        "control_bank.py", "test_control_bank.py", "collect_control_bank.py", "run_control_bank.py",
        "prepare_control_bank.py", "audit_control_bank.py", "verify_control_bank_actions.py",
        "audit_native_long_modes_labels.py", "CONTROL_BANK_THEORY.zh.md"]))
    sources = dict(old["source_sha256"], **fresh["source_sha256"])
    sources.update({str(HERE / name): digest(HERE / name) for name in runtime})
    sources.update({str(path.resolve()): digest(path) for path in
        (args.fresh_run / "cohort_plan.json", args.fresh_run / "plan.json", args.fresh_run / "summary.json",
         HERE / "design/native_long_modes_audit_20260909.json")})
    maximum = sum(task["maximum_queries"] for task in tasks)
    bound = maximum*200*1024+len(tasks)*32*1024**2
    plan = dict(protocol=PROTOCOL, stage="native_control_bank_paired", benchmark="native_long", model="long",
        settings=SETTINGS, arms=list(ARMS), tasks=tasks, cohort=cohort, original_source=verify_source(),
        thresholds=old["thresholds"], margins=old["margins"], methods=["v82_frozen"],
        allowed_gpus=[0,1,2,3], render_gpus=[0,3], replicas_per_gpu=8, workers_per_gpu=8,
        batch_size=1, hidden_capture=False, threshold_fitting=False,
        fresh_run=str(args.fresh_run.resolve()), fresh_cohort_sha256=digest(args.fresh_run / "cohort_plan.json"),
        new_main_coverage=50, prior_main_coverage=100, cohorts=["development", "fresh"],
        first_alarms_queries_checked=checked_queries, selection="all usable first v8.2 alarms, no outcome filtering",
        timing="complete original main and exact full C0 before alarm q+1; original total 520 steps",
        runtime_sources=runtime, source_sha256=sources, maximum_model_queries=maximum,
        maximum_output_bytes=bound, storage_quota_gib=max(8, math.ceil(bound/1024**3)+1), disk_floor_gib=8)
    atomic_json(args.output, plan)
    groups = {label: dict(mains=sum(t["cohort_group"] == label for t in cohort),
        successes=sum(t["native_success"] for t in cohort if t["cohort_group"] == label),
        alarms=sum(t["cohort_group"] == label for t in tasks)) for label in plan["cohorts"]}
    print(json.dumps(dict(status="frozen", cohorts=groups, arms=len(ARMS), jobs=len(tasks)*(len(ARMS)+1),
        maximum_model_queries=maximum, maximum_output_gib=bound/1024**3, plan_sha256=digest(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
