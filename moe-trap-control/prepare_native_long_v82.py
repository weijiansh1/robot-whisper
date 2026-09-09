#!/usr/bin/env python3
"""Freeze every usable v8.2 event in the complete audited 100-main cohort."""

import argparse
import copy
import csv
import json
from pathlib import Path

from collection_protocol import HERE, stable_id, verify_frozen_alarm
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, records
from native_long_runtime import verify_source
from v8_closed_loop import limits
from v82_closed_loop import ARMS, PROTOCOL, SETTINGS, V82TriggerMonitor


def prepare(args):
    verify_frozen_alarm()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen experiment")
    original = json.loads((args.parent_run / "plan.json").read_text())
    audit_path = HERE / "design/native_long_v8_audit_20260909.json"
    audit = json.loads(audit_path.read_text())
    comparison = HERE / "design/native_long_v82_replay_20260909"
    detection = json.loads((comparison / "summary.json").read_text())
    if (audit["status"] != "passed" or detection["status"] != "passed" or
            digest(args.parent_run / "plan.json") != audit["plan_sha256"] or
            audit["plan_sha256"] != detection["plan_sha256"] or len(original["cohort"]) != 100):
        raise ValueError("Complete audited native cohort required")
    if digest(comparison / "first_alarms.csv") != detection["first_alarms_sha256"]:
        raise ValueError("Frozen v8.2 comparison changed")
    with (comparison / "first_alarms.csv").open(newline="") as stream:
        alarms = {r["main_id"]: r for r in csv.DictReader(stream)}
    cohort = copy.deepcopy(original["cohort"])
    checked = 0
    for task in cohort:
        live = V82TriggerMonitor()
        for row in records(Path(task["parent_directory"]) / "main"):
            live.update(row[PROBS_KEY], row["proprio"][:3])
            checked += 1
        if any(live.first[k] != v for k, v in task["first_alarms"].items()):
            raise ValueError("Original alarm history changed")
        if live.first["v82_frozen"] != int(alarms[task["main_id"]]["v82_frozen"]):
            raise ValueError("Online v8.2 differs from independently frozen offline replay")
        task["first_alarms"] = live.first
        q0 = live.first["v82_frozen"]+1
        task["events"] = []
        if q0 > 0 and q0 < task["parent_queries"]:
            task["events"] = [dict(event_id=stable_id(PROTOCOL, task["main_id"], q0),
                start_query=q0, alarm_query=q0-1, methods=["v82_frozen"], deployable=True)]
            task["maximum_queries"] = task["parent_queries"]+9*(52-q0)+8*15*min(12, 52-q0)
    tasks = [t for t in cohort if t["events"]]
    if len(tasks) != 14 or sum(not t["native_success"] for t in tasks) != 12:
        raise ValueError("Frozen event cohort differs")
    runtime = list(dict.fromkeys(original["runtime_sources"]+[
        "run_v8_closed_loop.py", "v82_closed_loop.py", "collect_native_long_v82.py",
        "prepare_native_long_v82.py", "run_native_long_v82.py", "test_v82_closed_loop.py"]))
    sources = {p: digest(p) for p in original["source_sha256"]}
    for p, expected in original["source_sha256"].items():
        if sources[p] != expected:
            raise ValueError("Prior frozen runtime changed: "+p)
    sources.update({str(HERE / name): digest(HERE / name) for name in runtime})
    sources.update({str(p): digest(p) for p in (audit_path, comparison / "summary.json", comparison / "first_alarms.csv")})
    thresholds, margins = limits()
    maximum = sum(t["maximum_queries"] for t in tasks)
    plan = dict(protocol=PROTOCOL, stage="native_v82_paired_intervention", benchmark="native_long", model="long",
        settings=SETTINGS, arms=list(ARMS), tasks=tasks, cohort=cohort, original_source=verify_source(),
        thresholds=thresholds.tolist(), margins=margins.tolist(), methods=["v82_frozen"],
        allowed_gpus=[0, 1, 2, 3], render_gpus=[0, 3], batch_size=1, replicas_per_gpu=8, workers_per_gpu=8,
        hidden_capture=False, threshold_fitting=False, new_main_coverage=0, reused_main_coverage=100,
        selection="all usable first v8.2 alarms in the entire original 100-main cohort, without outcome filtering",
        timing="complete original main, complete exact C0, first v8.2 alarm q+1; chunk10, original total520",
        evaluation="fixed-v8 selector isolates trigger change; v82 selector also shifts candidate and convergence thresholds",
        no_alarm_policy="retain each untriggered main outcome and original policy query cost",
        noise_pairing="original v8_closed_loop streams shared by all arms; one policy RNG draw per executed query",
        verification=dict(all_mains_online_v82_match_offline=True, main_queries_checked=checked),
        source_run=str(args.parent_run.resolve()), source_plan_sha256=digest(args.parent_run / "plan.json"),
        source_audit=str(audit_path), source_audit_sha256=digest(audit_path),
        runtime_sources=runtime, source_sha256=sources, maximum_model_queries=maximum,
        maximum_output_bytes=maximum*200*1024+len(tasks)*16*1024**2, storage_quota_gib=8, disk_floor_gib=8)
    atomic_json(args.output, plan)
    print(json.dumps(dict(status="frozen", mains=100, events=len(tasks), native_failures=12,
        native_successes=2, arms=len(ARMS), jobs=len(tasks)*(1+len(ARMS)), maximum_queries=maximum,
        maximum_output_gib=plan["maximum_output_bytes"]/1024**3, plan_sha256=digest(args.output))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args())
