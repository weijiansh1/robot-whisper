#!/usr/bin/env python3
"""Audit exact parent replay, causal branch positions, candidates and interventions."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_collection_preflight import require, audit_as, GlobalIntrinsicProfile, IntrinsicGuardMonitor
from audit_collection_experiment import expected_swap
from collection_protocol import stable_id
from collection_routes import (ALL_FIELDS, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                               EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY, INTERVENTION_PROBS_KEY)
from collection_storage import atomic_json, digest, load_snapshot, records
from continuation_experiment import (ARMS, CANDIDATES, METHODS, PROTOCOL, REPLICATES, HERE,
                                     load_plan, noise_for, seed_for, short_count)


def route_check(row, swap_probability=None):
    require(not any("hidden" in field.lower() for field in row), "Hidden capture")
    for field in ALL_FIELDS[:5]:
        require(row[field].shape == (8, 10, 11, 32 if field == PROBS_KEY else 4), "HB shape")
        require(np.isfinite(row[field]).all(), "Non-finite HB")
    p, ids, weights = row[PROBS_KEY], row[NATIVE_IDS_KEY], row[NATIVE_WEIGHTS_KEY]
    effective = row[EFFECTIVE_IDS_KEY]
    require(np.max(np.abs(p.astype(np.float32).sum(-1) - 1)) < .001, "HB normalization")
    require(np.all((ids >= 0) & (ids < 32)) and np.all((effective >= 0) & (effective < 32)), "Expert range")
    require(np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0) and
            np.all(np.diff(np.sort(effective, axis=-1), axis=-1) > 0), "Duplicate expert")
    require(np.all(weights >= 0) and np.max(np.abs(weights.sum(-1) - 1)) < 1e-5, "Actual combine weights")
    np.testing.assert_array_equal(weights, row[EFFECTIVE_WEIGHTS_KEY])
    if swap_probability is None:
        np.testing.assert_array_equal(ids, effective)
    else:
        np.testing.assert_array_equal(swap_probability.astype(np.float16), p)
        np.testing.assert_array_equal(expected_swap(swap_probability, ids), effective)
        require(np.count_nonzero(ids != effective) == 400, "Exactly 400 swap sites required")
    require(np.isfinite(row["actions"]).all() and row["noise"].shape == (10, 24), "Action/noise contract")
    audit_as(row)


def independent_edge(rows):
    p = np.stack([r[PROBS_KEY][4:, :3, 1:] for r in rows]).astype(np.float64)
    p /= p.sum(-1, keepdims=True)
    distances = np.zeros((CANDIDATES, CANDIDATES))
    for i in range(CANDIDATES):
        for j in range(i + 1, CANDIDATES):
            distances[i, j] = distances[j, i] = np.sqrt(.5 * np.square(np.sqrt(p[i]) - np.sqrt(p[j])).sum(-1).mean())
    scores = distances.sum(1) / (CANDIDATES - 1)
    selected = int(np.flatnonzero(np.isclose(scores, scores.max(), atol=1e-12, rtol=0))[0])
    return selected, scores


def audit_task(directory, task, profile):
    directory, parent = Path(directory), Path(task["parent_directory"])
    result = json.loads((directory / "result.json").read_text())
    original = json.loads((parent / "result.json").read_text())
    require(result["status"] == "completed" and not result["invalid_pair"], "Incomplete continuation")
    require(result["protocol"] == PROTOCOL and result["main_id"] == task["main_id"], "Experiment identity")
    require(result["reused_main"] and result["queries"] == 0 and not result["main_intervention"] and
            not result["hidden_capture"], "Main reuse contract")
    require(result["gpu"] in (0, 1, 2, 3, 4, 5, 7) and result["render_gpu"] in (0, 3, 4, 5, 7), "GPU exclusion")
    require(result["model_metadata"]["bundle_physical_gpu"] == result["gpu"] and result["model"] == "long", "Model GPU")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Parent model identity")
    require(result["seed"] == original["seed"] and result["init_index"] == original["init_index"] and
            result["success"] == original["success"], "Parent outcome/init")
    require(result["first_alarms"] == task["first_alarms"] and result["terminal_alarms"] == task["terminal_alarms"], "Trigger identities")
    main = list(records(parent / "main"))
    c0 = list(records(directory / "c0"))
    require(len(main) == len(c0) == result["c0"]["compared_queries"] == task["parent_queries"], "Complete C0")
    require(result["c0"]["status"] == "passed" and result["c0"]["starts_after_main_complete"], "C0 status")
    require(result["c0"] == json.loads((directory / "c0/branch.json").read_text()), "C0 commit")
    for expected, actual in zip(main, c0):
        for field in expected:
            if field not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(actual[field], expected[field], err_msg="C0: " + field)
    require(len(result["events"]) == len(task["events"]), "Event count")
    require(len(result["branches"]) == len(task["events"]) * REPLICATES * len(ARMS), "Branch count")
    audited = dict(main_id=task["main_id"], benchmark=task["benchmark"], category=task["category"],
        analysis_role=task["analysis_role"], base_task=original["variant"]["base_task"],
        success=original["success"], first_alarms=task["first_alarms"], terminal_alarms=task["terminal_alarms"],
        c0_queries=len(c0), directory=str(directory), events=[], branches=[])
    pairs = []
    for planned, event in zip(task["events"], result["events"]):
        for key, expected in planned.items():
            require(event[key] == expected, "Event plan field: " + key)
        event_id, q0 = event["event_id"], event["start_query"]
        root = directory / "events" / event_id
        require(json.loads((root / "event.json").read_text()) == event, "Event commit")
        saved = load_snapshot(root / "snapshot")
        require(event["snapshot_manifest_sha256"] == digest(root / "snapshot/manifest.json"), "Snapshot hash")
        require(q0 == saved["query"] and saved["action_steps"] == int(main[q0]["action_steps_before"]), "Next query snapshot")
        require(q0 == event["alarm_query"] + 1 and not bool(main[q0 - 1]["success"]), "Causal intervention timing")
        np.testing.assert_array_equal(saved["physics"]["sim"], main[q0]["sim_before"])
        rng = np.random.default_rng()
        rng.bit_generator.state = saved["policy_rng"]
        np.testing.assert_array_equal(rng.standard_normal((10, 24)).astype(np.float32), main[q0]["noise"])
        event_branches = [b for b in result["branches"] if b["event_id"] == event_id]
        require({(b["arm"], b["replicate"]) for b in event_branches} ==
                {(arm, rep) for arm in ARMS for rep in range(REPLICATES)}, "Missing/duplicate event branches")
        prefix_monitor = IntrinsicGuardMonitor(profile)
        for row in main[:q0]:
            prefix_monitor.update(row[PROBS_KEY])
        for replicate in range(REPLICATES):
            pool_dir = root / "pools" / str(replicate)
            selection = json.loads((pool_dir / "selection.json").read_text())
            require(selection["candidate_manifest_sha256"] == digest(pool_dir / "candidates/manifest.json"), "Candidate commit")
            pool = list(records(pool_dir / "candidates"))
            require(len(pool) == CANDIDATES, "Candidate pool size")
            for index, row in enumerate(pool):
                route_check(row)
                require(int(row["query"]) == index and int(row["source_query"]) == q0, "Candidate position")
                np.testing.assert_array_equal(row["noise"], noise_for(task["main_id"], event_id, replicate, q0, index))
                require(int(row["policy_seed"]) == seed_for(task["main_id"], event_id, replicate, q0, "policy", index), "Candidate seed")
                np.testing.assert_array_equal(row["input_sha256"], main[q0]["input_sha256"])
            selected, scores = independent_edge(pool)
            require(selected == selection["selected"], "Independent peripheral selection")
            np.testing.assert_allclose(scores, selection["scores"], rtol=1e-10, atol=1e-12)
            first_rows = {}
            for arm in ARMS:
                path = root / "branches" / arm / str(replicate)
                branch = json.loads((path / "branch.json").read_text())
                require(branch in event_branches and branch["status"] == "completed", "Branch commit")
                require(branch["branch_id"] == stable_id(PROTOCOL, task["main_id"], event_id, arm, replicate), "Branch ID")
                require(branch["parent_commit_sha256"] == task["parent_commit_sha256"] and
                        branch["snapshot_manifest_sha256"] == event["snapshot_manifest_sha256"], "Branch parent hashes")
                require(branch["selection_sha256"] == digest(pool_dir / "selection.json"), "Selection checksum")
                require(branch["parent_prefix_queries"] == [0, q0] and branch["remaining_action_budget"] == 520 - saved["action_steps"], "Prefix/horizon")
                require(branch["starts_after_c0"] and branch["starts_after_main_complete"] and
                        (path / "branch.json").stat().st_mtime_ns >= (directory / "c0/branch.json").stat().st_mtime_ns >=
                        (parent / "main_complete.json").stat().st_mtime_ns, "Branch preceded C0 or parent completion")
                limit0 = short_count(task["main_id"], event_id, replicate, q0) if arm == "short_chunk" else 10
                require(branch["first_chunk_limit"] == limit0, "Random chunk limit")
                import copy
                monitor = copy.deepcopy(prefix_monitor)
                previous, steps, prior_success, count = saved["physics"]["sim"], saved["action_steps"], False, 0
                for row in records(path / "suffix"):
                    q = int(row["query"])
                    require(not prior_success and q == q0 + count and int(row["action_steps_before"]) == steps, "Branch continuity")
                    candidate = selected if count == 0 and arm == "edge_noise" else 0
                    np.testing.assert_array_equal(row["noise"], noise_for(task["main_id"], event_id, replicate, q, candidate))
                    require(int(row["candidate_id"]) == candidate and int(row["policy_seed"]) ==
                            seed_for(task["main_id"], event_id, replicate, q, "policy", candidate), "Branch candidate/seed")
                    require(int(row["environment_seed"]) == seed_for(task["main_id"], event_id, replicate, q, "environment"), "Environment seed")
                    swap = count == 0 and arm == "swap1"
                    require(bool(row["route_swap_applied"]) == swap and bool(row["intervention_applied"]) ==
                            (count == 0 and arm != "random"), "Intervention scope")
                    require(bool(row["cached_candidate"]) == (count == 0 and not swap), "Candidate query accounting")
                    evidence = None
                    if swap:
                        require(digest(path / "intervention.npz") == branch["intervention_evidence_sha256"], "Swap evidence hash")
                        with np.load(path / "intervention.npz", allow_pickle=False) as archive:
                            evidence = archive[INTERVENTION_PROBS_KEY]
                    route_check(row, evidence)
                    np.testing.assert_array_equal(row["sim_before"], previous)
                    require(np.isfinite(row["sim_after"]).all() and np.isfinite(row["proprio"]).all(), "Finite physical state")
                    require(hashlib.sha256(row["proprio"].tobytes()).hexdigest() == row["input_component_sha256"][2].decode(), "Actual proprio hash")
                    if count == 0:
                        first_rows[arm] = row
                        np.testing.assert_array_equal(row["input_sha256"], main[q0]["input_sha256"])
                        if not swap:
                            for field in (*ALL_FIELDS, "actions", "noise", "input_sha256"):
                                np.testing.assert_array_equal(row[field], pool[candidate][field], err_msg="Candidate reuse: " + field)
                    alarm = monitor.update(row[PROBS_KEY])
                    require(bool(row["alarm"]) == alarm["alarm"], "Branch diagnostic alarm")
                    np.testing.assert_array_equal(row["alarm_scores"], np.asarray([alarm[key] for key in
                        ("freeze_score", "acceleration_score", "periodicity_score")], np.float32))
                    requested = min(limit0 if count == 0 else 10, 520 - steps)
                    executed = int(row["executed_action_count"])
                    require(int(row["requested_action_count"]) == requested and 0 < executed <= requested, "Executed chunk limit")
                    require(executed == requested or bool(row["success"]), "Unexpected chunk truncation")
                    previous, prior_success = row["sim_after"], bool(row["success"])
                    count += 1
                    steps += executed
                require(count == branch["queries"] and steps - saved["action_steps"] == branch["action_steps"], "Branch totals")
                require(prior_success == branch["success"] and (prior_success or steps == 520), "Incomplete suffix")
                audited["branches"].append({key: branch[key] for key in
                    ("event_id", "methods", "arm", "replicate", "queries", "action_steps", "success", "first_chunk_limit", "first_candidate")})
            np.testing.assert_array_equal(first_rows["random"]["actions"], first_rows["short_chunk"]["actions"])
            np.testing.assert_array_equal(first_rows["random"][PROBS_KEY][:4, 0], first_rows["swap1"][PROBS_KEY][:4, 0])
            np.testing.assert_array_equal(first_rows["random"][PROBS_KEY][4, 0], first_rows["swap1"][PROBS_KEY][4, 0])
            pairs.append(dict(event_id=event_id, replicate=replicate, selected_edge=selected,
                short_limit=limit0 if arm == "short_chunk" else short_count(task["main_id"], event_id, replicate, q0),
                edge_action_rms=float(np.sqrt(np.mean(np.square(first_rows["edge_noise"]["actions"].astype(np.float64) - first_rows["random"]["actions"])))),
                swap_action_rms=float(np.sqrt(np.mean(np.square(first_rows["swap1"]["actions"].astype(np.float64) - first_rows["random"]["actions"]))))))
        audited["events"].append(event)
    branch_queries = sum(b["queries"] for b in audited["branches"])
    candidates = len(task["events"]) * REPLICATES * CANDIDATES
    reused = len(task["events"]) * REPLICATES * 3
    require(result["candidate_queries"] == candidates and result["reused_candidate_queries"] == reused, "Candidate totals")
    require(result["actual_model_queries"] == len(c0) + branch_queries + candidates - reused, "Actual inference accounting")
    audited.update(branch_queries=branch_queries, candidate_queries=candidates, actual_model_queries=result["actual_model_queries"], pairs=pairs)
    return audited


def summarize(tasks):
    result = []
    for benchmark in ("all", "pro", "plus"):
        cohort = [t for t in tasks if t["analysis_role"] == "perturbation" and (benchmark == "all" or t["benchmark"] == benchmark)]
        for method in METHODS:
            states = [(t, next((e for e in t["events"] if method in e["methods"]), None)) for t in cohort]
            alarmed = [t for t in cohort if t["first_alarms"][method] >= 0]
            failure = sum(not t["success"] for t in cohort)
            tp, fp = sum(not t["success"] for t in alarmed), sum(t["success"] for t in alarmed)
            active = [(t, e) for t, e in states if e is not None]
            for arm in ARMS:
                state_rates = [(t, np.mean([b["success"] for b in t["branches"] if b["event_id"] == e["event_id"] and b["arm"] == arm])) for t, e in active]
                failed = [float(rate) for t, rate in state_rates if not t["success"]]
                successful = [float(rate) for t, rate in state_rates if t["success"]]
                policy = [float(t["success"]) if e is None else float(np.mean([b["success"] for b in t["branches"]
                    if b["event_id"] == e["event_id"] and b["arm"] == arm])) for t, e in states]
                baseline = [float(t["success"]) if e is None else float(np.mean([b["success"] for b in t["branches"]
                    if b["event_id"] == e["event_id"] and b["arm"] == "random"])) for t, e in states]
                result.append(dict(benchmark=benchmark, method=method, arm=arm, parents=len(cohort),
                    native_successes=len(cohort) - failure, alarms=len(alarmed), true_positives=tp, false_positives=fp,
                    precision=tp / (tp + fp) if tp + fp else None, recall=tp / failure if failure else None,
                    false_positive_rate=fp / (len(cohort) - failure) if len(cohort) > failure else None,
                    eligible_states=len(active), failed_states=len(failed), successful_states=len(successful),
                    terminal_alarms=sum(method in t["terminal_alarms"] for t in cohort),
                    rescue_successes=int(round(sum(failed) * REPLICATES)), failed_parent_branches=len(failed) * REPLICATES,
                    harm_failures=int(round(sum(1 - s for s in successful) * REPLICATES)), successful_parent_branches=len(successful) * REPLICATES,
                    rescue_rate=float(np.mean(failed)) if failed else None,
                    harm_rate=float(np.mean([1 - s for s in successful])) if successful else None,
                    alarm_state_success_rate=float(np.mean([s for _, s in state_rates])) if state_rates else None,
                    policy_success_rate=float(np.mean(policy)) if policy else None,
                    policy_delta_vs_random=float(np.mean(np.asarray(policy) - baseline)) if policy else None))
    return result


def run(runs, output):
    profile = GlobalIntrinsicProfile.load(HERE.parent / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz")
    audited, failures, run_records, identities = [], [], [], set()
    for root in runs:
        summary = json.loads((root / "summary.json").read_text())
        plan = load_plan(root / "plan.json")
        require(summary["status"] == "completed" and summary["continuation_experiment"], "Run incomplete")
        require(summary["batch_size"] == 1 and 6 not in summary["gpus"], "GPU/batch constraint")
        require(summary["plan_sha256"] == digest(root / "plan.json"), "Run plan hash")
        require(summary["temporary_models_stopped"] and not summary["live_replica_pids_after_cleanup"], "Temporary models still active")
        for source, expected in summary["source_sha256"].items():
            require(digest(root / "sources" / Path(source).name) == expected, "Archived source hash")
        planned = {task["main_id"]: task for task in plan["tasks"]}
        require({t["main_id"] for t in summary["tasks"]} == set(planned), "Planned parents missing")
        for record in summary["tasks"]:
            main_id = record["main_id"]
            require(main_id not in identities, "Duplicate reused parent across stages")
            identities.add(main_id)
            try:
                require(record["status"] == "completed", "Task incomplete")
                audited.append(audit_task(record["output"], planned[main_id], profile))
            except Exception as error:
                failures.append(dict(main_id=main_id, directory=record["output"], error=repr(error)))
            print("AUDIT " + json.dumps(dict(parents=len(audited), failures=len(failures), total=len(identities))), flush=True)
        run_records.append(dict(run=str(root.resolve()), plan_sha256=digest(root / "plan.json"),
            summary_sha256=digest(root / "summary.json"), actual_model_queries=summary["actual_model_queries"],
            collection_elapsed_seconds=summary["collection_elapsed_seconds"], gpu_summary=summary["gpu_summary"],
            all_replicas_supplied_samples=summary["all_replicas_supplied_samples"],
            all_replicas_supplied_mean_utilization_percent=summary["all_replicas_supplied_mean_utilization_percent"]))
    result = dict(status="passed" if not failures else "failed", protocol=PROTOCOL,
        auditor_sha256=digest(Path(__file__)), reused_parents=len(audited), new_main_coverage=0,
        benchmarks=dict(Counter(t["benchmark"] for t in audited)),
        primary_parents=sum(t["analysis_role"] == "perturbation" for t in audited),
        event_states=sum(len(t["events"]) for t in audited),
        paired_branches=sum(len(t["branches"]) for t in audited), c0_queries=sum(t["c0_queries"] for t in audited),
        branch_queries=sum(t["branch_queries"] for t in audited), candidate_queries=sum(t["candidate_queries"] for t in audited),
        actual_model_queries=sum(t["actual_model_queries"] for t in audited),
        outcome_definition="four replicates averaged per parent/state; no best-of-four; native labels from committed mains",
        metrics=summarize(audited), runs=run_records, tasks=audited, failures=failures)
    atomic_json(output, result)
    print(json.dumps({k: result[k] for k in ("status", "reused_parents", "event_states", "paired_branches", "actual_model_queries", "failures")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Audit output already exists")
    raise SystemExit(run(args.runs, args.output))
