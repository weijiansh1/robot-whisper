#!/usr/bin/env python3
"""Independent completeness and fidelity audit for one late-alarm repair run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from repair_control import ARMS, PROTOCOL, WINDOW_STEPS, seed_for


def audit_task(task_dir, plan_task, plan):
    findings, replay_dir = [], task_dir / "replay"
    replay = json.loads((replay_dir / "result.json").read_text())
    if replay["status"] != "completed" or replay["c0"]["status"] != "passed":
        findings.append("replay_not_passed")
    if replay["c0"]["compared_queries"] != plan_task["parent_queries"]:
        findings.append("c0_query_count")
    if replay["c0"]["online_knn_first"] != plan_task["first_alarm"]:
        findings.append("knn_trigger_mismatch")
    events = replay["events"]
    for event in events:
        location = replay_dir / "events" / event["event_id"]
        manifest = json.loads((location / "snapshot/manifest.json").read_text())
        if digest(location / "snapshot/manifest.json") != event["snapshot_manifest_sha256"]:
            findings.append("snapshot_hash:" + event["event_id"])
        for name, expected in manifest.items():
            if digest(location / "snapshot" / name) != expected:
                findings.append("snapshot_file_hash:" + name)
        if digest(location / "physics.json") != event["physics_sha256"]:
            findings.append("physics_hash:" + event["event_id"])
        saved = load_snapshot(location / "snapshot")
        if saved["query"] != event["start_query"] or saved["action_steps"] != event["action_steps_before"]:
            findings.append("snapshot_query:" + event["event_id"])
    branches_dir = task_dir / "branches"
    result = json.loads((branches_dir / "result.json").read_text()) if (branches_dir / "result.json").exists() else None
    if result is None or result["status"] != "completed":
        findings.append("branches_not_completed")
        return dict(main_id=plan_task["main_id"], findings=findings, events=len(events), branches=0)
    expected = {(e["event_id"], r, a) for e in events for r in range(plan["replicates"]) for a in plan["arms"]}
    seen, suffix_rows, repair_rows, hidden = set(), 0, 0, False
    for branch in result["branches"]:
        key = (branch["event_id"], branch["replicate"], branch["arm"])
        seen.add(key)
        directory = branches_dir / "branches" / branch["event_id"] / ("repeat%d" % branch["replicate"]) / branch["arm"]
        if branch["status"] != "completed" or json.loads((directory / "branch.json").read_text())["status"] != "completed":
            findings.append("branch_incomplete:%s/%s" % (branch["event_id"][:8], branch["arm"]))
        steps = 0
        if (directory / "repair/manifest.json").exists():
            for row in records(directory / "repair"):
                repair_rows += 1
                steps += int(row["executed_action_count"])
                if str(row["reason"].astype(str)) not in ("reached", "max_chunks", "open", "close", "", "regrasp"):
                    findings.append("repair_reason:%s" % row["reason"])
        if branch.get("repair_steps", 0) != steps:
            findings.append("repair_steps:%s/%s" % (branch["event_id"][:8], branch["arm"]))
        if (directory / "suffix/manifest.json").exists():
            for index, row in enumerate(records(directory / "suffix")):
                suffix_rows += 1
                steps += int(row["executed_action_count"])
                hidden |= any("hidden" in k.lower() for k in row)
                if int(row["policy_seed"]) != seed_for(plan_task["main_id"], branch["replicate"], index, "policy", 0):
                    findings.append("policy_seed:%s/%s" % (branch["event_id"][:8], branch["arm"]))
                if not np.isfinite(np.asarray(row[PROBS_KEY], np.float32)).all():
                    findings.append("nonfinite_probs")
        if steps != branch["action_steps"]:
            findings.append("action_steps:%s/%s" % (branch["event_id"][:8], branch["arm"]))
        if branch["action_steps"] > WINDOW_STEPS or (branch["success"] and branch["success_step"] is None):
            findings.append("window_or_success:%s/%s" % (branch["event_id"][:8], branch["arm"]))
        if branch["success_within_original"] and not (branch["success"] and branch["success_step"] <= branch["original_remaining_steps"]):
            findings.append("original_endpoint:%s/%s" % (branch["event_id"][:8], branch["arm"]))
    if seen != expected:
        findings.append("missing_branches:%d" % len(expected - seen))
    if hidden:
        findings.append("hidden_capture")
    return dict(main_id=plan_task["main_id"], findings=findings, events=len(events), branches=len(seen),
                suffix_rows=suffix_rows, repair_rows=repair_rows)


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    if plan["protocol"] != PROTOCOL or plan["arms_registry"] != ARMS:
        raise ValueError("Wrong protocol")
    tasks = [audit_task(args.run / "tasks" / t["main_id"], t, plan) for t in plan["tasks"]]
    report = dict(status="passed" if all(not t["findings"] for t in tasks) else "failed", run=str(args.run),
        plan_sha256=digest(args.run / "plan.json"), summary_status=summary["status"], gpus=summary["gpus"],
        parents=len(tasks), events=sum(t["events"] for t in tasks), branches=sum(t["branches"] for t in tasks),
        suffix_rows=sum(t.get("suffix_rows", 0) for t in tasks), repair_rows=sum(t.get("repair_rows", 0) for t in tasks),
        tasks=tasks)
    atomic_json(args.out, report)
    print(json.dumps({k: report[k] for k in ("status", "parents", "events", "branches", "suffix_rows", "repair_rows")}))
    for t in tasks:
        if t["findings"]:
            print(t["main_id"], t["findings"][:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
