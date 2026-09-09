#!/usr/bin/env python3
"""Independently audit recovery timing, complete suffixes, route scores and costs."""

import argparse
from collections import Counter
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_control import (ARMS, CONTRACT, GPUS, RENDER_GPUS, PROTOCOL, RouteRisk,
                              chunk_limit, load_plan, noise_for, seed_for)
from audit_long_continuation import route_check, independent_edge
from collection_protocol import HERE, PARAMETERS_SHA256
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records


def require(value, message):
    if not value:
        raise ValueError(message)


def independent_risk(monitor, risk):
    from compare_frozen_alarm_methods import normalized_vectors
    if monitor.query < 7:
        return np.full(10, np.nan), float("nan")
    cache = dict(mobility=np.asarray(monitor._mobility_history, np.float32)[None],
        acceleration=np.asarray(monitor._acceleration_history, np.float32)[None],
        periodicity=np.asarray(monitor._periodicity_history, np.float32)[None],
        valid=np.ones((1, monitor.query + 1), bool))
    vector = normalized_vectors(cache, np.asarray([0]), risk.reference)[0, -1]
    distance = np.linalg.norm(risk.reference["success_dynamic"] - vector, axis=1)
    return vector, float(np.float32(np.sort(distance)[:20].mean()))


def check_result(path, task):
    result = json.loads(path.read_text())
    require(result["status"] == "completed" and not result["invalid_pair"], "Incomplete recovery job")
    require(result["protocol"] == PROTOCOL and result["contract"] == CONTRACT, "Recovery protocol")
    require(result["gpu"] in GPUS and result["render_gpu"] in RENDER_GPUS, "Unauthorized GPU")
    require(result["main_id"] == task["main_id"] and result["parent_commit_sha256"] == task["parent_commit_sha256"], "Parent identity")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["queries"] == 0 and result["reused_main"], "Native reuse")
    original = json.loads((Path(task["parent_directory"]) / "result.json").read_text())
    require(result["seed"] == original["seed"] and result["init_index"] == original["init_index"], "Parent seed/init")
    require(result["success"] == original["success"], "Native outcome identity")
    require(result["model_metadata"]["bundle_physical_gpu"] == result["gpu"], "Endpoint GPU")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout", "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Model identity: " + key)
    return result, original


def check_pool(directory, monitor, risk, main_id, replicate, index, q, expected_input):
    pool = list(records(directory / "candidates"))
    require(len(pool) == 4, "Incomplete candidate pool")
    selection = json.loads((directory / "selection.json").read_text())
    require(selection["candidate_manifest_sha256"] == digest(directory / "candidates/manifest.json") and
            selection["scores_sha256"] == digest(directory / "scores.npz"), "Candidate evidence hash")
    require(selection["relative_query"] == index and selection["source_query"] == q and selection["replicate"] == replicate,
            "Candidate pool position")
    vectors, knn = [], []
    for candidate, row in enumerate(pool):
        route_check(row)
        require(int(row["query"]) == candidate and int(row["source_query"]) == q and int(row["relative_query"]) == index, "Candidate query")
        np.testing.assert_array_equal(row["noise"], noise_for(main_id, replicate, index, candidate))
        require(int(row["policy_seed"]) == seed_for(main_id, replicate, index, "policy", candidate), "Candidate policy seed")
        np.testing.assert_array_equal(row["input_sha256"], expected_input)
        trial = copy.deepcopy(monitor)
        trial.update(row[PROBS_KEY])
        vector, score = independent_risk(trial, risk)
        vectors.append(vector)
        knn.append(score)
    _, centrality = independent_edge(pool)
    actions = np.stack([row["actions"] for row in pool])[:, :, :7].astype(np.float64)
    delta = np.asarray([np.sqrt(np.mean((a - actions[0]) ** 2)) for a in actions])
    eligible = np.asarray([delta[i] <= np.median(delta) and np.sign(actions[i, 0, 6]) == np.sign(actions[0, 0, 6]) for i in range(4)])
    expected = dict(centrality=centrality, vectors=np.asarray(vectors), knn=np.asarray(knn), action_rms=delta, eligible=eligible)
    with np.load(directory / "scores.npz", allow_pickle=False) as archive:
        require(set(archive.files) == set(expected), "Candidate score fields")
        for key, value in expected.items():
            np.testing.assert_allclose(archive[key], value, rtol=2e-6, atol=1e-9, err_msg=key)
    def minimum(values, ids):
        values = np.asarray(values)
        ids = np.asarray(ids)
        return int(ids[np.isclose(values, values.min(), rtol=0, atol=1e-12)].min()) if len(ids) else 0
    finite = np.flatnonzero(np.isfinite(knn))
    guarded = np.flatnonzero(np.isfinite(knn) & eligible)
    winners = dict(center=minimum(centrality, np.arange(4)), edge=minimum(-centrality, np.arange(4)),
                   knn=minimum(np.asarray(knn)[finite], finite), guarded_knn=minimum(np.asarray(knn)[guarded], guarded))
    require(selection["winners"] == winners and selection["input_sha256"] == expected_input.item().decode(), "Independent selector differs")
    return pool, selection


def audit_parent(payload):
    run, task, arms, replicates = payload
    run, parent = Path(run), Path(task["parent_directory"])
    replay_dir = run / "tasks" / task["main_id"] / "replay"
    replay, original = check_result(replay_dir / "result.json", task)
    risk = RouteRisk()
    main, c0 = list(records(parent / "main")), list(records(replay_dir / "c0"))
    require(len(main) == len(c0) == task["parent_queries"] == replay["c0"]["compared_queries"], "Complete C0")
    require(replay["c0"]["status"] == "passed" and replay["c0"] == json.loads((replay_dir / "c0/branch.json").read_text()), "C0 commit")
    monitor, first = risk.monitor(), -1
    for expected, actual in zip(main, c0):
        route_check(actual)
        for key in expected:
            if key not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(expected[key], actual[key], err_msg="C0: " + key)
        monitor.update(actual[PROBS_KEY])
        _, score = independent_risk(monitor, risk)
        np.testing.assert_allclose(actual["knn_score"], score, rtol=2e-6, atol=1e-7)
        if first < 0 and score > risk.threshold:
            first = int(actual["query"])
    require(first == task["first_alarm"] == replay["c0"]["online_knn_first"], "Frozen online kNN first trigger")
    require(len(replay["events"]) == len(task["events"]), "Event count")
    audited = dict(main_id=task["main_id"], benchmark=task["benchmark"], category=task["category"],
        analysis_role=task["analysis_role"], base_task=original["variant"]["base_task"], success=original["success"],
        first_alarm=first, parent_queries=len(main), c0_queries=len(c0), events=[], branches=[], pools=[],
        actual_model_queries=len(c0), candidate_queries=0, reused_candidate_queries=0, branch_queries=0)
    require(replay["actual_model_queries"] == len(c0), "C0 compute ledger")
    for planned, event in zip(task["events"], replay["events"]):
        for key, value in planned.items():
            require(event[key] == value, "Event position differs")
        q0, event_id = event["start_query"], event["event_id"]
        snapshot_dir = replay_dir / "events" / event_id / "snapshot"
        saved = load_snapshot(snapshot_dir)
        require(digest(snapshot_dir / "manifest.json") == event["snapshot_manifest_sha256"], "Snapshot hash")
        require(saved["query"] == q0 and saved["action_steps"] == int(main[q0]["action_steps_before"]), "Snapshot position")
        np.testing.assert_array_equal(saved["physics"]["sim"], main[q0]["sim_before"])
        require(event == json.loads((replay_dir / "events" / event_id / "event.json").read_text()), "Event commit")
        require((event["deployable"] and q0 == first + 1) or (not event["deployable"] and q0 == max(0, first - 5)), "Causal/diagnostic distinction")
        audited["events"].append(event)
        prefix = risk.monitor()
        for row in main[:q0]:
            prefix.update(row[PROBS_KEY])
        for replicate in range(replicates):
            root = run / "tasks" / task["main_id"] / "events" / event_id / ("repeat%d" % replicate)
            result, _ = check_result(root / "result.json", task)
            require(result["c0"]["replay_result_sha256"] == digest(replay_dir / "result.json") and
                    result["c0"]["replay_branch_sha256"] == digest(replay_dir / "c0/branch.json"), "C0 dependency")
            require((root / "initial_pool/selection.json").stat().st_mtime_ns >= (replay_dir / "c0/branch.json").stat().st_mtime_ns >=
                    (parent / "main_complete.json").stat().st_mtime_ns, "Branches ran before C0 or parent completion")
            initial_pool, initial_selection = check_pool(root / "initial_pool", prefix, risk, task["main_id"], replicate, 0, q0, main[q0]["input_sha256"])
            require(len(result["branches"]) == len(arms) and {b["arm"] for b in result["branches"]} == set(arms), "Missing/duplicated branches")
            pools, reused, queries, first_rows = 1, 0, 0, {}
            for arm in arms:
                directory = root / "branches" / arm
                branch = json.loads((directory / "branch.json").read_text())
                require(branch in result["branches"] and branch["status"] == "completed" and branch["control"] == ARMS[arm], "Branch commit/control")
                require(branch["event_id"] == event_id and branch["replicate"] == replicate and branch["start_query"] == q0, "Branch identity")
                require(branch["remaining_action_budget"] == 520 - saved["action_steps"] and branch["deployable"] == event["deployable"], "Branch horizon/timing")
                require(branch["snapshot_manifest_sha256"] == event["snapshot_manifest_sha256"] and
                        branch["parent_commit_sha256"] == task["parent_commit_sha256"] and
                        branch["initial_selection_sha256"] == digest(root / "initial_pool/selection.json"), "Branch evidence identity")
                monitor, previous = copy.deepcopy(prefix), saved["physics"]["sim"]
                steps, count, success, deployment = saved["action_steps"], 0, False, 0
                for row in records(directory / "suffix"):
                    q = int(row["query"])
                    require(not success and q == q0 + count and int(row["relative_query"]) == count, "Suffix query continuity")
                    require(int(row["action_steps_before"]) == steps, "Suffix action continuity")
                    control = ARMS[arm]
                    selecting = count < control["duration"] and control["selector"] != "fixed"
                    pool, selection = (initial_pool, initial_selection) if count == 0 else (None, None)
                    if selecting and count > 0:
                        pool, selection = check_pool(directory / "pools" / str(count), monitor, risk,
                            task["main_id"], replicate, count, q, row["input_sha256"])
                        pools += 1
                    candidate = selection["winners"][control["selector"]] if selecting else control["candidate"] if count == 0 else 0
                    require(int(row["candidate_id"]) == candidate and bool(row["selecting"]) == selecting, "Candidate/control execution")
                    require(bool(row["cached_candidate"]) == (pool is not None) and not row["route_swap_applied"], "Cache/route execution")
                    np.testing.assert_array_equal(row["noise"], noise_for(task["main_id"], replicate, count, candidate))
                    require(int(row["policy_seed"]) == seed_for(task["main_id"], replicate, count, "policy", candidate), "Suffix policy seed")
                    if pool is not None:
                        reused += 1
                        for key in (*ALL_FIELDS, "actions", "noise", "input_sha256"):
                            np.testing.assert_array_equal(row[key], pool[candidate][key], err_msg="Executed candidate: " + key)
                    route_check(row)
                    np.testing.assert_array_equal(row["sim_before"], previous)
                    require(np.isfinite(row["sim_after"]).all(), "Non-finite physics")
                    require(hashlib.sha256(row["proprio"].tobytes()).hexdigest() == row["input_component_sha256"][2].decode(), "Proprio hash")
                    alarm = monitor.update(row[PROBS_KEY])
                    vector, score = independent_risk(monitor, risk)
                    np.testing.assert_allclose(row["knn_vector"], vector, rtol=2e-6, atol=1e-9)
                    np.testing.assert_allclose(row["knn_score"], score, rtol=2e-6, atol=1e-7)
                    require(bool(row["alarm"]) == alarm["alarm"], "Diagnostic v7 alarm")
                    np.testing.assert_array_equal(row["alarm_scores"], np.asarray([alarm[k] for k in
                        ("freeze_score", "acceleration_score", "periodicity_score")], np.float32))
                    requested, executed = chunk_limit(arm, count, 520 - steps), int(row["executed_action_count"])
                    require(int(row["requested_action_count"]) == requested and 0 < executed <= requested, "Short-chunk execution")
                    require(executed == requested or bool(row["success"]), "Unexpected truncated chunk")
                    require(int(row["environment_first_seed"]) == seed_for(task["main_id"], replicate, steps, "environment") and
                            int(row["environment_last_seed"]) == seed_for(task["main_id"], replicate, steps + executed - 1, "environment"), "Environment step seeds")
                    if count == 0:
                        first_rows[arm] = row
                        np.testing.assert_array_equal(row["input_sha256"], main[q0]["input_sha256"])
                        require(branch["first_candidate"] == candidate, "Initial candidate ID")
                    deployment += 4 if selecting else 1
                    previous, success = row["sim_after"], bool(row["success"])
                    steps += executed
                    count += 1
                require((success or steps == 520) and success == branch["success"], "Incomplete suffix")
                require(count == branch["queries"] and steps - saved["action_steps"] == branch["action_steps"] and
                        deployment == branch["deployment_model_queries"], "Branch accounting")
                queries += count
                audited["branches"].append({key: branch[key] for key in ("arm", "replicate", "event_id", "timing", "deployable",
                    "start_query", "queries", "action_steps", "success", "deployment_model_queries", "first_candidate")})
            for arm in arms:
                if ARMS[arm]["selector"] == "fixed" and ARMS[arm]["candidate"] == 0:
                    np.testing.assert_array_equal(first_rows[arm]["actions"], first_rows["candidate0"]["actions"])
            actual = queries + 4 * pools - reused
            require(actual == result["actual_model_queries"] and 4 * pools == result["candidate_queries"] and
                    reused == result["reused_candidate_queries"], "Actual inference ledger")
            audited["actual_model_queries"] += actual
            audited["candidate_queries"] += 4 * pools
            audited["reused_candidate_queries"] += reused
            audited["branch_queries"] += queries
            audited["pools"].append(dict(event_id=event_id, replicate=replicate, timing=event["timing"],
                winners=initial_selection["winners"], candidates=[dict(candidate=i,
                    success=next(b["success"] for b in result["branches"] if b["arm"] == "candidate%d" % i),
                    queries=next(b["queries"] for b in result["branches"] if b["arm"] == "candidate%d" % i)) for i in range(4)]))
    return audited


def run(args):
    plan = load_plan(args.run / "plan.json")
    summary = json.loads((args.run / "summary.json").read_text())
    require(summary["status"] == "completed" and summary["gpus"] == list(GPUS), "Run incomplete or GPU scope")
    require(summary["plan_sha256"] == digest(args.run / "plan.json"), "Plan identity")
    for source, expected in summary["source_sha256"].items():
        require(digest(args.run / "sources" / Path(source).name) == expected, "Archived source changed")
        require(digest(source) == expected, "Runtime source changed before audit")
    payloads = [(str(args.run), task, plan["arms"], plan["replicates"]) for task in plan["tasks"]]
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in pool.map(audit_parent, payloads):
            tasks.append(task)
            print("AUDITED " + json.dumps(dict(parents=len(tasks), total=len(payloads), main_id=task["main_id"], branches=len(task["branches"]))), flush=True)
    actual = sum(t["actual_model_queries"] for t in tasks)
    require(actual == summary["actual_model_queries"], "Run inference ledger")
    report = dict(status="passed", protocol=PROTOCOL, stage=plan["stage"], run=str(args.run.resolve()),
        plan_sha256=digest(args.run / "plan.json"), auditor_sha256=digest(__file__),
        frozen_parameters_sha256=PARAMETERS_SHA256, parents=len(tasks),
        benchmark_counts=dict(Counter(t["benchmark"] for t in tasks)), events=sum(len(t["events"]) for t in tasks),
        branches=sum(len(t["branches"]) for t in tasks), actual_model_queries=actual, tasks=tasks,
        verification="all native queries, all pools and all suffix queries; independent existing frozen geometry formulas and sorted exact neighbors")
    atomic_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args())
