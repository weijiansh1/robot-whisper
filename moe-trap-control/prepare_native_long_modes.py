#!/usr/bin/env python3
"""Freeze the complete prior native v8.2 event cohort for mode comparison."""

import argparse
import copy
import json
from pathlib import Path

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_storage import atomic_json, digest
from mode_control import ARMS, PROTOCOL, SETTINGS
from native_long_runtime import verify_source
from v8_closed_loop import limits
from v82_closed_loop import PROTOCOL as PARENT_PROTOCOL


def prepare(args):
    verify_frozen_alarm()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen plan")
    original = json.loads(args.parent_plan.read_text())
    if original["protocol"] != PARENT_PROTOCOL or len(original["cohort"]) != 100:
        raise ValueError("Audited native v8.2 cohort required")
    for path, expected in original["source_sha256"].items():
        if digest(path) != expected:
            raise ValueError("Prior frozen source changed: "+path)
    for task in original["cohort"]:
        for name, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(Path(task["parent_directory"]) / name) != task[key]:
                raise ValueError("Original main identity changed")
    cohort = copy.deepcopy(original["cohort"])
    for task in cohort:
        for event in task["events"]:
            event["event_id"] = stable_id(PROTOCOL, task["main_id"], event["start_query"])
        if task["events"]:
            q0 = task["events"][0]["start_query"]
            remaining = 52-q0
            task["maximum_queries"] = task["parent_queries"]+len(ARMS)*(remaining+5)+6*15*min(5, remaining)
    tasks = [task for task in cohort if task["events"]]
    if len(tasks) != 14 or sum(not task["native_success"] for task in tasks) != 12:
        raise ValueError("Complete original alarm cohort required without outcome filtering")
    runtime = list(dict.fromkeys(original["runtime_sources"]+[
        "mode_control.py", "test_mode_control.py", "collect_native_long_modes.py", "run_native_long_modes.py",
        "prepare_native_long_modes.py", "audit_native_long_modes.py", "verify_mode_dispatched_actions.py",
        "audit_long_continuation.py", "audit_native_long_v82.py"]))
    sources = dict(original["source_sha256"])
    sources.update({str(HERE / name): digest(HERE / name) for name in runtime})
    sources[str(args.parent_plan.resolve())] = digest(args.parent_plan)
    thresholds, margins = limits()
    maximum = sum(task["maximum_queries"] for task in tasks)
    plan = dict(protocol=PROTOCOL, stage="native_mode_paired_intervention", benchmark="native_long", model="long",
        settings=SETTINGS, arms=list(ARMS), tasks=tasks, cohort=cohort, original_source=verify_source(),
        thresholds=thresholds.tolist(), margins=margins.tolist(), methods=["v82_frozen"],
        allowed_gpus=[0, 1, 2, 3], render_gpus=[0, 3], batch_size=1, replicas_per_gpu=8, workers_per_gpu=8,
        hidden_capture=False, threshold_fitting=False, new_main_coverage=0, reused_main_coverage=100,
        selection="all 14 usable first v8.2 alarms from the original 100 mains, including two original successes",
        timing="complete original main, complete exact C0, first alarm q+1; original total 520 steps",
        evaluation="paired two-repeat intervention families conditional on original-noise pre-intervention mode",
        no_alarm_policy="retain the 86 untriggered original outcomes; report each repeat and their mean",
        noise_pairing="same parent/repeat/query/candidate noise across non-native families; environment tape by absolute step",
        source_run=original["source_run"], source_plan_sha256=digest(args.parent_plan),
        source_plan=str(args.parent_plan.resolve()), source_audit=original["source_audit"],
        source_audit_sha256=original["source_audit_sha256"],
        runtime_sources=runtime, source_sha256=sources, maximum_model_queries=maximum,
        maximum_output_bytes=maximum*200*1024+len(tasks)*16*1024**2, storage_quota_gib=12, disk_floor_gib=8)
    if plan["maximum_output_bytes"] > plan["storage_quota_gib"]*1024**3:
        raise ValueError("Conservative data bound exceeds quota")
    atomic_json(args.output, plan)
    print(json.dumps(dict(status="frozen", events=len(tasks), arms=len(ARMS), jobs=len(tasks)*(1+len(ARMS)),
        maximum_model_queries=maximum, maximum_output_gib=plan["maximum_output_bytes"]/1024**3,
        plan_sha256=digest(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
