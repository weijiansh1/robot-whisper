#!/usr/bin/env python3
"""Independent completeness and fidelity audit for one late-alarm repair run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from repair_control import (ARMS, EPISODE_ARMS, HORIZON_STEPS, PROTOCOL, PROTOCOL_V2, PROTOCOL_V3, REGULATOR_ARMS,
                            SUPERVISOR_ARMS, WINDOW_STEPS, seed_for)

REPAIR_REASONS = ("reached", "max_chunks", "open", "close", "", "regrasp", "release", "contact")


def twin_prefix_equal(directory, twin_dir, limit):
    """v2 branches must reproduce their v1 twin bitwise up to the first extra intervention."""
    mine = list(records(directory / "suffix"))
    theirs = list(records(twin_dir / "suffix"))
    n = len(mine) if limit is None else min(limit, len(mine))
    n = min(n, len(theirs))   # the twin may have ended earlier (success inside the window): compare the overlap
    for a, b in zip(mine[:n], theirs[:n]):
        if str(a["input_sha256"]) != str(b["input_sha256"]) or not np.array_equal(a["actions"], b["actions"]):
            return False
    return True


def audit_episodes(task_dir, plan_task, plan):
    """Full-episode jobs: vla replicate 0 reproduces the parent; regulated arms stay within the horizon."""
    findings, path = [], task_dir / "episodes/result.json"
    if not path.exists():
        return ["episodes_missing"], 0, 0
    result = json.loads(path.read_text())
    if result["status"] != "completed":
        findings.append("episodes_not_completed")
    expected = {(a, r) for a in plan["episode_arms"] for r in range(plan["replicates"])}
    seen, rows = set(), 0
    for episode in result.get("episodes", []):
        seen.add((episode["arm"], episode["replicate"]))
        directory = task_dir / "episodes" / "episodes" / episode["arm"] / ("repeat%d" % episode["replicate"])
        if episode["status"] != "completed" or json.loads((directory / "episode.json").read_text())["status"] != "completed":
            findings.append("episode_incomplete:%s/%d" % (episode["arm"], episode["replicate"]))
        if episode["arm"] == "vla" and episode["replicate"] == 0 and episode.get("fidelity") != "passed":
            findings.append("episode_fidelity:%s" % plan_task["main_id"][:8])
        steps = 0
        for index, row in enumerate(records(directory / "suffix")):
            rows += 1
            steps += int(row["executed_action_count"])
            if any("hidden" in k.lower() for k in row):
                findings.append("hidden_capture")
            if episode["replicate"] > 0 and int(row["policy_seed"]) != seed_for(plan_task["main_id"], episode["replicate"], index, "policy", 0):
                findings.append("episode_policy_seed:%s" % episode["arm"])
            if not episode["features"] and float(np.abs(np.asarray(row["regulator_log"])).sum()) != 0.0:
                findings.append("vla_episode_modified:%s" % episode["arm"])
        if steps != episode["action_steps"] or steps > HORIZON_STEPS or (episode["success"] and episode["success_step"] is None):
            findings.append("episode_steps:%s/%d" % (episode["arm"], episode["replicate"]))
    if seen != expected:
        findings.append("missing_episodes:%d" % len(expected - seen))
    return findings, len(seen), rows


def audit_task(task_dir, plan_task, plan):
    v2 = plan["protocol"] in (PROTOCOL_V2, PROTOCOL_V3)
    findings = []
    if v2:
        replay_dir = Path(plan["replay_run"]) / "tasks" / plan_task["main_id"] / "replay"
        replay = json.loads((replay_dir / "result.json").read_text())
        mine = json.loads((task_dir / "branches/result.json").read_text()) if (task_dir / "branches/result.json").exists() else None
        if mine is not None and mine["c0"].get("replay_result_sha256") != digest(replay_dir / "result.json"):
            findings.append("replay_source_hash")
    else:
        replay_dir = task_dir / "replay"
        replay = json.loads((replay_dir / "result.json").read_text())
        if replay["c0"]["compared_queries"] != plan_task["parent_queries"]:
            findings.append("c0_query_count")
        if replay["c0"]["online_knn_first"] != plan_task["first_alarm"]:
            findings.append("knn_trigger_mismatch")
    if replay["status"] != "completed" or replay["c0"]["status"] != "passed":
        findings.append("replay_not_passed")
    events = [e for e in replay["events"] if not plan.get("timings") or e["timing"] in plan["timings"]]
    if v2 and not plan["arms"]:                      # episodes-only plan: no fork branches to audit
        report = dict(main_id=plan_task["main_id"], findings=findings, events=len(events), branches=0, suffix_rows=0, repair_rows=0)
        episode_findings, episodes, rows = audit_episodes(task_dir, plan_task, plan)
        findings.extend(episode_findings)
        report.update(episodes=episodes, episode_rows=rows)
        return report
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
                if str(row["reason"].astype(str)) not in REPAIR_REASONS:
                    findings.append("repair_reason:%s" % row["reason"])
        if v2:
            twin = branch.get("twin")
            if plan.get("supervisor") and len(branch.get("interventions", [])) > plan["supervisor"]["max_interventions"]:
                findings.append("too_many_interventions:%s/%s" % (branch["event_id"][:8], branch["arm"]))
            if twin is not None:
                twin_dir = replay_dir.parent / "branches/branches" / branch["event_id"] / ("repeat%d" % branch["replicate"]) / twin
                if not twin_prefix_equal(directory, twin_dir, branch.get("first_extra_intervention_query")):
                    findings.append("twin_prefix:%s/%s" % (branch["event_id"][:8], branch["arm"]))
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
    report = dict(main_id=plan_task["main_id"], findings=findings, events=len(events), branches=len(seen),
                  suffix_rows=suffix_rows, repair_rows=repair_rows)
    if plan.get("episode_arms"):
        episode_findings, episodes, rows = audit_episodes(task_dir, plan_task, plan)
        findings.extend(episode_findings)
        report.update(episodes=episodes, episode_rows=rows)
    return report


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    if plan["protocol"] == PROTOCOL_V3:
        if plan["arms_registry"] != REGULATOR_ARMS or plan["episode_arms_registry"] != EPISODE_ARMS:
            raise ValueError("Wrong regulator registry")
    elif plan["protocol"] == PROTOCOL_V2:
        if plan["arms_registry"] != SUPERVISOR_ARMS:
            raise ValueError("Wrong supervisor registry")
    elif plan["protocol"] != PROTOCOL or plan["arms_registry"] != ARMS:
        raise ValueError("Wrong protocol")
    tasks = [audit_task(args.run / "tasks" / t["main_id"], t, plan) for t in plan["tasks"]]
    report = dict(status="passed" if all(not t["findings"] for t in tasks) else "failed", run=str(args.run),
        plan_sha256=digest(args.run / "plan.json"), summary_status=summary["status"], gpus=summary["gpus"],
        parents=len(tasks), events=sum(t["events"] for t in tasks), branches=sum(t["branches"] for t in tasks),
        suffix_rows=sum(t.get("suffix_rows", 0) for t in tasks), repair_rows=sum(t.get("repair_rows", 0) for t in tasks),
        episodes=sum(t.get("episodes", 0) for t in tasks), episode_rows=sum(t.get("episode_rows", 0) for t in tasks),
        tasks=tasks)
    atomic_json(args.out, report)
    print(json.dumps({k: report[k] for k in ("status", "parents", "events", "branches", "suffix_rows", "repair_rows", "episodes", "episode_rows")}))
    for t in tasks:
        if t["findings"]:
            print(t["main_id"], t["findings"][:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
