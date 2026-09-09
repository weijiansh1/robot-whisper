#!/usr/bin/env python3
"""Freeze and replay an internal selector on previously audited complete pools."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist

import adaptive_control
import analyze_adaptive_control
import analyze_long_continuation
import collection_protocol
import collection_routes
import collection_storage
import internal_action_selector
import intrinsic_guard_monitor
import route_noise_selector
import v8_feature_control
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import atomic_json, atomic_npz, digest, records
from internal_action_selector import InternalActionSelector, PROTOCOL, RULES, SIGNALS


def require(value, message):
    if not value:
        raise ValueError(message)


def sources():
    modules = (adaptive_control, analyze_adaptive_control, analyze_long_continuation,
               collection_protocol, collection_routes, collection_storage,
               internal_action_selector, intrinsic_guard_monitor, route_noise_selector,
               v8_feature_control)
    paths = [Path(module.__file__).resolve() for module in modules]
    paths += [Path(__file__).resolve(), Path(__file__).with_name("test_internal_action_selector.py")]
    return {str(path): digest(path) for path in paths}


def freeze(args):
    require(not args.plan.exists(), "A frozen selector plan must not be overwritten")
    audit = json.loads(args.audit.read_text())
    require(audit["status"] == "passed" and audit["protocol"] == adaptive_control.PROTOCOL,
            "Expected a passed complete-candidate audit")
    old_plan = Path(audit["run"]) / "plan.json"
    require(digest(old_plan) == audit["plan_sha256"], "Historical collection plan changed")
    selector = InternalActionSelector()
    plan = dict(protocol=PROTOCOL, created_utc=datetime.now(timezone.utc).isoformat(),
        rules=RULES, signals=list(SIGNALS), thresholds=selector.thresholds.tolist(),
        audit=str(args.audit.resolve()), audit_sha256=digest(args.audit),
        collection_plan=str(old_plan), collection_plan_sha256=digest(old_plan),
        parameters_sha256=digest(adaptive_control.PARAMETERS),
        reference_sha256=digest(adaptive_control.REFERENCE), source_sha256=sources(),
        cohort="all parents from the existing validation audit; primary role=perturbation",
        timing="first frozen Euclidean kNN alarm q+1 only; one whole native chunk",
        outcome_use="decisions saved before joining suffix labels; labels historically known, not blinded validation",
        comparisons=["candidate0", "internal_selector", "noise_center_once", "center_once",
                     "edge_once", "knn_once", "guarded_knn_once", "oracle_once"],
        hypotheses="descriptive test of frozen signal dominance; no tuning after this replay",
        new_gpu_forwards=0, hidden_capture=False,
        cost="counterfactual deployment: one forward by default, four if expanded; original horizon")
    atomic_json(args.plan, plan)
    print(json.dumps(dict(plan=str(args.plan), sha256=digest(args.plan), protocol=PROTOCOL)))


def check_contract(scores, actions, limits, decision, count):
    """Check inequalities directly, independently of choose()'s diagnostic arrays."""
    s, a = np.asarray(scores, float), np.asarray(actions, float)
    eps = 64*np.finfo(np.float32).eps*np.maximum(1., np.maximum(np.abs(s[0]), np.abs(limits)))
    excess = np.maximum(s-limits, 0.)
    active = np.isfinite(s[0]).all() and np.any(excess[0] > eps)
    require(decision["expanded"] == bool(active), "Expansion disagrees with baseline evidence")
    rejection = [[] for _ in s]
    if not active:
        require(decision["candidate"] == 0, "No evidence must retain default")
        return [["default_has_no_active_excess"] for _ in s]
    deltas = np.sqrt(((a[:, :count, :6]-a[0, :count, :6])**2).mean(axis=(1, 2)))
    radius = np.median(deltas[np.isfinite(a).all(axis=(1, 2))])
    resolution = 64*np.finfo(np.float32).eps*max(1., np.max(np.abs(a[0, :count, :6])))
    eligible = []
    for i in range(1, len(s)):
        if not np.isfinite(s[i]).all() or not np.isfinite(a[i]).all():
            rejection[i].append("nonfinite")
        if np.any(np.sign(a[i, :count, 6]) != np.sign(a[0, :count, 6])):
            rejection[i].append("gripper_change")
        if deltas[i] > radius+resolution:
            rejection[i].append("action_radius")
        if deltas[i] <= resolution:
            rejection[i].append("no_resolved_action_change")
        if np.any(excess[i] > excess[0]+eps):
            rejection[i].append("increased_excess")
        if not np.any(excess[i, excess[0] > eps] < excess[0, excess[0] > eps]-eps[excess[0] > eps]):
            rejection[i].append("no_active_improvement")
        if not rejection[i]:
            eligible.append(i)
    expected = min(eligible, key=lambda i: (np.count_nonzero(excess[i] > eps), deltas[i], i)) if eligible else 0
    require(decision["candidate"] == expected, "Decision violates dominance/guard/ranking contract")
    np.testing.assert_array_equal(decision["admissible"], [i in eligible for i in range(len(s))])
    return rejection


def decide(plan, output):
    collection = json.loads(Path(plan["collection_plan"]).read_text())
    run = Path(plan["collection_plan"]).parent
    decisions, evidence = [], []
    for index, task in enumerate(collection["tasks"]):
        events = [e for e in task["events"] if e["timing"] == "after_alarm"]
        require(len(events) <= 1, "Expected at most one causal fork per parent")
        if not events:
            continue
        event, parent = events[0], Path(task["parent_directory"])
        require(digest(parent / "main/manifest.json") == task["parent_manifest_sha256"], "Parent manifest changed")
        require(digest(parent / "main_complete.json") == task["parent_commit_sha256"], "Parent commit changed")
        selector = InternalActionSelector()
        np.testing.assert_array_equal(selector.thresholds, plan["thresholds"])
        q0 = event["start_query"]
        require(event["deployable"] and q0 == event["alarm_query"]+1, "Noncausal timing")
        source = None
        for row in records(parent / "main"):
            if int(row["query"]) == q0:
                source = row
                break
            selector.observe(row[PROBS_KEY])
        require(source is not None and selector.monitor.v7.query == q0-1, "Incomplete causal prefix")
        count = min(10, 520-int(source["action_steps_before"]))
        for rep in range(collection["replicates"]):
            root = run / "tasks" / task["main_id"] / "events" / event["event_id"] / ("repeat%d" % rep)
            directory = root / "initial_pool"
            old = json.loads((directory / "selection.json").read_text())
            require(digest(directory / "scores.npz") == old["scores_sha256"] and
                    digest(directory / "candidates/manifest.json") == old["candidate_manifest_sha256"], "Pool evidence changed")
            require(old["source_query"] == q0 and old["relative_query"] == 0 and old["replicate"] == rep,
                    "Incorrect candidate pool position")
            pool = list(records(directory / "candidates"))
            require(len(pool) == 4, "Incomplete pool")
            for candidate, row in enumerate(pool):
                require(int(row["query"]) == candidate and int(row["source_query"]) == q0, "Candidate ID mismatch")
                np.testing.assert_array_equal(row["input_sha256"], source["input_sha256"])
                np.testing.assert_array_equal(row["noise"], adaptive_control.noise_for(task["main_id"], rep, 0, candidate))
            actions = np.stack([r["actions"] for r in pool])
            probabilities = np.stack([r[PROBS_KEY] for r in pool])
            ids = [r["input_sha256"].item().decode() for r in pool]
            decision, scores = selector.propose(probabilities, actions, ids, True, count)
            with np.load(directory / "scores.npz", allow_pickle=False) as saved:
                np.testing.assert_allclose(scores[:, 5], saved["knn"], rtol=2e-6, atol=1e-7)
                distance = cdist(saved["vectors"].astype(float), selector.bank.astype(float), metric="cosine")
                expected_cosine = np.sort(distance, axis=1)[:, :20].mean(axis=1)
                np.testing.assert_allclose(scores[:, 6], expected_cosine, rtol=2e-6, atol=1e-7)
            rejection = check_contract(scores, actions, selector.thresholds, decision, count)
            decisions.append(dict(main_id=task["main_id"], benchmark=task["benchmark"],
                analysis_role=task["analysis_role"], event_id=event["event_id"], replicate=rep,
                start_query=q0, requested_action_count=count, input_sha256=ids[0], root=str(root),
                initial_selection_sha256=digest(directory / "selection.json"),
                candidate_manifest_sha256=old["candidate_manifest_sha256"],
                scores_sha256=old["scores_sha256"], decision=decision, rejection_reasons=rejection))
            evidence.append(dict(scores=scores, actions=actions))
        print(json.dumps(dict(phase="decisions", parent=index+1, total=len(collection["tasks"]), pools=len(decisions))), flush=True)
    atomic_npz(output / "evidence.npz", dict(scores=np.stack([e["scores"] for e in evidence]),
        actions=np.stack([e["actions"] for e in evidence]), thresholds=np.asarray(plan["thresholds"])))
    atomic_json(output / "decisions.json", dict(protocol=PROTOCOL, labels_used_for_selection=False,
        evidence_sha256=digest(output / "evidence.npz"), rows=decisions))
    return decisions


def verify_branch(root, summary, selection_sha256, pool, candidate):
    directory = root / "branches" / ("candidate%d" % candidate)
    branch = json.loads((directory / "branch.json").read_text())
    require(all(branch[key] == value for key, value in summary.items()), "Audited suffix label changed")
    require(branch["status"] == "completed" and branch["first_candidate"] == candidate and
            branch["initial_selection_sha256"] == selection_sha256 and
            branch["control"] == adaptive_control.ARMS["candidate%d" % candidate], "Branch identity/control changed")
    suffix = directory / "suffix"
    manifest = json.loads((suffix / "manifest.json").read_text())
    require(manifest["queries"] == branch["queries"], "Incomplete suffix manifest")
    for block in manifest["blocks"]:
        require(digest(suffix / block["path"]) == block["sha256"], "Suffix checksum mismatch")
    first = next(records(suffix))
    for key in (*ALL_FIELDS, "noise", "actions", "input_sha256"):
        np.testing.assert_array_equal(first[key], pool[candidate][key], err_msg="Dispatched first candidate: "+key)
    with np.load(suffix / manifest["blocks"][-1]["path"], allow_pickle=False) as last:
        end = int(last["action_steps_before"][-1])+int(last["executed_action_count"][-1])
        require(bool(last["success"][-1]) == branch["success"] and (branch["success"] or end == 520), "Incomplete suffix outcome")
    return dict(branch_sha256=digest(directory / "branch.json"), suffix_manifest_sha256=digest(suffix / "manifest.json"))


def outcomes(plan, decisions, output):
    audit = json.loads(Path(plan["audit"]).read_text())
    tasks = copy.deepcopy(audit["tasks"])
    by_id = {task["main_id"]: task for task in tasks}
    paired = []
    for index, row in enumerate(decisions):
        task, root = by_id[row["main_id"]], Path(row["root"])
        branches = {b["first_candidate"]: b for b in task["branches"]
            if b["event_id"] == row["event_id"] and b["replicate"] == row["replicate"] and b["arm"].startswith("candidate")}
        require(set(branches) == set(range(4)), "Missing complete fixed-candidate outcomes")
        pool = list(records(root / "initial_pool/candidates"))
        evidence = [verify_branch(root, branches[i], row["initial_selection_sha256"], pool, i) for i in range(4)]
        picked = row["decision"]["candidate"]
        selected = copy.deepcopy(branches[picked])
        selected.update(arm="internal_selector", deployment_model_queries=selected["queries"]+3*row["decision"]["expanded"])
        task["branches"].append(selected)
        old_pool = next(p for p in task["pools"] if p["event_id"] == row["event_id"] and p["replicate"] == row["replicate"])
        labels = [bool(branches[i]["success"]) for i in range(4)]
        require(labels == [bool(c["success"]) for c in old_pool["candidates"]], "Pool and suffix outcomes differ")
        paired.append(dict(main_id=row["main_id"], base_task=task["base_task"], benchmark=task["benchmark"],
            analysis_role=task["analysis_role"], event_id=row["event_id"], replicate=row["replicate"],
            native_success=task["success"], candidate=picked, success=labels[picked], baseline_success=labels[0],
            labels=labels, expanded=row["decision"]["expanded"], reason=row["decision"]["reason"],
            rejection_reasons=row["rejection_reasons"], branch_evidence=evidence))
        if (index+1) % 20 == 0:
            print(json.dumps(dict(phase="outcome_evidence", pools=index+1, total=len(decisions))), flush=True)
    primary = [t for t in tasks if t["analysis_role"] == "perturbation"]
    cohorts = dict(primary=primary, primary_pro=[t for t in primary if t["benchmark"] == "pro"],
        primary_plus=[t for t in primary if t["benchmark"] == "plus"], controls=[t for t in tasks if t not in primary])
    metrics = [analyze_adaptive_control.metric(part, "after_alarm", name, cohort)
        for cohort, part in cohorts.items() for name in plan["comparisons"] if part]
    diagnostics = {}
    for cohort, part in cohorts.items():
        ids = {t["main_id"] for t in part}
        rows = [r for r in paired if r["main_id"] in ids]
        failed = [r for r in rows if not r["native_success"]]
        diagnostics[cohort] = dict(parents=len(part), pools=len(rows),
            expanded=sum(r["expanded"] for r in rows), changed=sum(r["candidate"] != 0 for r in rows),
            abstained=sum(r["candidate"] == 0 for r in rows), reasons=dict(Counter(r["reason"] for r in rows)),
            wins_vs_candidate0=sum(r["success"] and not r["baseline_success"] for r in rows),
            losses_vs_candidate0=sum(r["baseline_success"] and not r["success"] for r in rows),
            failed_pools=len(failed), failed_pools_with_any_success=sum(any(r["labels"]) for r in failed),
            oracle_rescuable_unique_parents=len({r["main_id"] for r in failed if any(r["labels"])}),
            extra_selection_forwards=sum(3*r["expanded"] for r in rows),
            rejected_successful_alternatives=[dict(main_id=r["main_id"], replicate=r["replicate"],
                candidate=i, reasons=r["rejection_reasons"][i]) for r in failed for i in range(1, 4)
                if r["labels"][i] and r["candidate"] != i])
    return dict(metrics=metrics, diagnostics=diagnostics, pairs=paired)


def replay(args):
    started = time.monotonic()
    plan = json.loads(args.plan.read_text())
    require(plan["protocol"] == PROTOCOL and plan["rules"] == RULES, "Selector rules changed")
    require(plan["source_sha256"] == sources(), "Frozen source changed")
    for key in ("audit", "collection_plan"):
        require(digest(plan[key]) == plan[key+"_sha256"], "Frozen input changed: "+key)
    require(digest(adaptive_control.PARAMETERS) == plan["parameters_sha256"] and
            digest(adaptive_control.REFERENCE) == plan["reference_sha256"], "Frozen calibration changed")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "plan.json", plan)
    decisions = decide(plan, args.output)
    decisions_sha = digest(args.output / "decisions.json")
    report = outcomes(plan, decisions, args.output)
    require(digest(args.output / "decisions.json") == decisions_sha, "Decisions changed after label join")
    require(plan["source_sha256"] == sources() and digest(plan["audit"]) == plan["audit_sha256"], "Inputs changed during replay")
    report.update(status="passed", protocol=PROTOCOL, plan_sha256=digest(args.plan),
        historical_audit_sha256=plan["audit_sha256"], decisions_sha256=decisions_sha,
        evidence_sha256=digest(args.output / "evidence.npz"), elapsed_seconds=time.monotonic()-started,
        new_gpu_forwards=0, hidden_capture=False, independent_contract_checks=len(decisions),
        notes=["Retrospective replay of historically inspected labels; not an independent validation cohort.",
               "Only one selected chunk is evaluated; all later actions use the native policy and original horizon.",
               "Candidate0 uses a fresh branch noise stream, not the recorded original main action.",
               "No changes to triggers, thresholds, the reference bank, model weights or gate logits.",
               "Query costs are counterfactual deployment counts, not new GPU work in this CPU replay.",
               "Repeats averaged within parent; descriptive 95% task-cluster bootstrap, 10000 draws, seed 20260908.",
               "No-variation intervals are unavailable, not evidence of equivalence.",
               "Whole-chunk gripper sign and relative action radius are heuristic guards, not a physical safety proof."])
    atomic_json(args.output / "summary.json", report)
    print(json.dumps(dict(status=report["status"], diagnostics=report["diagnostics"]["primary"],
                          elapsed_seconds=report["elapsed_seconds"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("freeze")
    prepare.add_argument("--audit", type=Path, required=True)
    prepare.add_argument("--plan", type=Path, required=True)
    execute = commands.add_parser("replay")
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args) if args.command == "freeze" else replay(args)
