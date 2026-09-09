#!/usr/bin/env python3
"""Audit full C0, all candidate evidence, committed history and bounded recovery."""

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

import numpy as np

from audit_long_continuation import require, route_check
from collection_protocol import stable_id
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from fixed_recovery_control import TriggerMonitor
from v8_closed_loop import PROTOCOL, ARMS, SETTINGS


def independent_risk(scores, thresholds, margins):
    values = np.asarray(scores, float)
    if not np.isfinite(values).all():
        return float("inf")
    f, a, p, b, c = (values-thresholds)/margins
    return float(max(f, min(a, p), b, c))


def independent_scores(monitor, probabilities):
    trial = copy.deepcopy(monitor)
    status = trial.update(probabilities, np.zeros(3))
    raw = np.asarray(probabilities, np.float32)
    raw = np.maximum(raw, 0)
    raw /= raw.sum(-1, keepdims=True)
    root = np.sqrt(raw[:, :, 1:])
    speeds = np.sqrt(((root[:, 1:]-root[:, :-1])**2).sum(-1))/np.sqrt(2.)
    speeds = speeds.mean(-1).astype(np.float32)
    lengths = speeds.sum(-1)
    inversion = -np.log(max(lengths[:4].mean(), 1e-12)/max(lengths[4:].mean(), 1e-12))
    curvature = np.abs(speeds[4:, 8]-2*speeds[4:, 4]+speeds[4:, 0]).mean()
    independent_raw = np.array([inversion, curvature], np.float32)
    np.testing.assert_allclose(independent_raw, status["v8_raw"], rtol=2e-5, atol=1e-7)
    history = np.asarray(monitor.v8.raw+[independent_raw], np.float32)
    if trial.v8.v7.query >= 4:
        relative = np.log(np.maximum(history[:, 1], 1e-12)/max(history[1:5, 1].mean(dtype=np.float32), 1e-12))
        expected = [history[-6:, 0].mean(dtype=np.float64), relative[-6:].mean(dtype=np.float64)]
        np.testing.assert_allclose(status["v8_scores"], expected, rtol=2e-5, atol=2e-6)
    return np.r_[[status[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], status["v8_scores"]]


def check_identity(result, task, original):
    require(result["status"] == "completed" and result["protocol"] == PROTOCOL and not result["invalid_pair"], "Incomplete job")
    require(result["contract"] == SETTINGS and result["gpu"] in (0, 1, 2, 3) and result["render_gpu"] in (0, 3), "Settings/device changed")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["reused_main"], "Native/no-hidden contract")
    require(result["main_id"] == task["main_id"] and result["seed"] == original["seed"] and result["init_index"] == original["init_index"], "Parent identity")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout", "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Model identity: "+key)


def check_pool(path, arm, row, live, native_noise, task, index, thresholds, margins):
    selection = json.loads((path / "selection.json").read_text())
    require(digest(path / "selection.json") == row["selection_sha256"].item().decode(), "Selection ledger hash")
    require(digest(path / "candidates/manifest.json") == selection["manifest_sha256"] and
            digest(path / "optimizer.npz") == selection["optimizer_sha256"], "Candidate/optimizer commitment")
    pool = list(records(path / "candidates"))
    require(len(pool) == selection["pool_size"] == 16, "Full unfiltered population missing")
    require(selection["arm"] == arm and selection["relative_query"] == index and selection["source_query"] == int(row["query"]), "Pool position")
    generator = np.random.default_rng(int(stable_id(PROTOCOL, task["main_id"], index, "extra_noise")[:8], 16))
    z = generator.standard_normal((15, 10, 24)).astype(np.float32)
    first = np.concatenate([native_noise[None], z[:7]])
    scores, risks = [], []
    for i, candidate in enumerate(pool):
        require(int(candidate["query"]) == i and int(candidate["relative_query"]) == index and
                int(candidate["source_query"]) == int(row["query"]), "Candidate index")
        np.testing.assert_array_equal(candidate["input_sha256"], row["input_sha256"])
        route_check(candidate)
        component = independent_scores(live, candidate[PROBS_KEY])
        np.testing.assert_allclose(candidate["component_scores"], component, rtol=2e-5, atol=2e-6)
        value = independent_risk(candidate["component_scores"], thresholds, margins)
        np.testing.assert_allclose(candidate["risk"], value, rtol=0, atol=1e-12)
        scores.append(candidate["component_scores"])
        risks.append(value)
    with np.load(path / "optimizer.npz", allow_pickle=False) as optimizer:
        if arm.startswith("guided"):
            elite = sorted(range(8), key=lambda i: (risks[i], i))[:4]
            mean = .5*first[elite].astype(float).mean(0)
            std = np.clip(np.sqrt(.5+.5*first[elite].astype(float).var(0)), .5, 1.5)
            np.testing.assert_array_equal(optimizer["elite"], elite)
            np.testing.assert_allclose(optimizer["mean"], mean, rtol=0, atol=1e-12)
            np.testing.assert_allclose(optimizer["std"], std, rtol=0, atol=1e-12)
            second = (mean+std*z[7:].astype(float)).astype(np.float32)
        else:
            np.testing.assert_array_equal(optimizer["mean"], np.zeros((10, 24)))
            np.testing.assert_array_equal(optimizer["std"], np.ones((10, 24)))
            second = z[7:]
    for candidate, noise in zip(pool, np.concatenate([first, second])):
        np.testing.assert_array_equal(candidate["noise"], noise)
    if arm.endswith("random"):
        rng = np.random.default_rng(int(stable_id(PROTOCOL, task["main_id"], index, "selection")[:8], 16))
        picked = int(rng.integers(16))
    else:
        picked = min(range(16), key=lambda i: (risks[i], i))
    require(picked == selection["candidate"] == int(row["candidate_id"]), "Selected candidate differs")
    np.testing.assert_allclose(selection["scores"], risks, rtol=0, atol=1e-12)
    for key in (*ALL_FIELDS, "actions", "noise", "input_sha256"):
        np.testing.assert_array_equal(row[key], pool[picked][key], err_msg="Actual dispatch: "+key)
    require(selection["input_sha256"] == row["input_sha256"].item().decode(), "Input digest differs")
    np.testing.assert_allclose(row["default_risk"], risks[0], rtol=0, atol=1e-12)
    normalized = (np.asarray(scores)-thresholds)/margins
    mechanisms = np.column_stack([normalized[:, 0], np.minimum(normalized[:, 1], normalized[:, 2]),
                                  normalized[:, 3], normalized[:, 4]])
    names = ("freeze", "turbulence", "inversion", "curvature")
    return np.asarray(risks), dict(index=index, candidate=picked, default=risks[0], selected=risks[picked],
        minimum=min(risks), maximum=max(risks), below_target=sum(v <= -1 for v in risks),
        original_population_below_target=sum(v <= -1 for v in risks[:8]),
        second_population_below_target=sum(v <= -1 for v in risks[8:]),
        default_components=np.asarray(scores[0]).tolist(), selected_components=np.asarray(scores[picked]).tolist(),
        dominant_default=names[int(np.argmax(mechanisms[0]))], dominant_selected=names[int(np.argmax(mechanisms[picked]))],
        first_noise_sha256=digest(path / "candidates" / "block_00000.npz"))


def audit_parent(payload):
    run, task, thresholds, margins = payload
    run, parent = Path(run), Path(task["parent_directory"])
    thresholds, margins = np.asarray(thresholds), np.asarray(margins)
    original = json.loads((parent / "result.json").read_text())
    replay_path = run / "replays" / task["main_id"]
    replay = json.loads((replay_path / "result.json").read_text())
    check_identity(replay, task, original)
    require(replay["c0"]["status"] == "passed" and replay["online_triggers_exact"], "Exact C0 missing")
    for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
        require(digest(parent / filename) == task[key] == replay[key], "Native commitment changed")
    main, c0 = list(records(parent / "main")), list(records(replay_path / "c0"))
    require(len(main) == len(c0) == task["parent_queries"] == replay["c0"]["compared_queries"], "Complete native replay")
    require(replay["actual_model_queries"] == len(c0), "C0 inference ledger")
    live, prefix = TriggerMonitor(), None
    q0 = task["events"][0]["start_query"]
    for expected, actual in zip(main, c0):
        route_check(actual)
        for key in expected:
            if key not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(actual[key], expected[key], err_msg="Exact C0: "+key)
        live.update(actual[PROBS_KEY], actual["proprio"][:3])
        if int(actual["query"]) == q0-1:
            prefix = copy.deepcopy(live)
    require(live.first == task["first_alarms"] and prefix.first["v8_frozen"] == q0-1, "Frozen v8 timing changed")
    event = replay["events"][0]
    require(all(event[key] == value for key, value in task["events"][0].items()), "Fork identity")
    location = replay_path / "events" / event["event_id"] / "snapshot"
    require(digest(location / "manifest.json") == event["snapshot_manifest_sha256"], "Snapshot changed")
    saved = load_snapshot(location)
    require(saved["query"] == q0 and saved["action_steps"] == int(main[q0]["action_steps_before"]), "Fork time")
    require(digest(replay_path / "environment_rng/manifest.json") == replay["rng_tape_sha256"], "RNG tape hash")
    tape = list(records(replay_path / "environment_rng"))
    require(len(tape) == original["action_steps"], "Incomplete RNG tape")
    total, branches, first_pools = len(c0), [], {}
    for arm in ARMS:
        path = run / "events" / event["event_id"] / arm
        result = json.loads((path / "result.json").read_text())
        branch = json.loads((path / "branch.json").read_text())
        check_identity(result, task, original)
        require(result["events"] == [event] and result["branches"] == [branch] and branch["status"] == "completed", "Branch commit")
        require(result["c0"]["replay_result_sha256"] == digest(replay_path / "result.json") and
                result["c0"]["replay_branch_sha256"] == digest(replay_path / "c0/branch.json"), "C0 dependency changed")
        require(branch["starts_after_c0"] and branch["starts_after_main_complete"] and
                (path / "branch.json").stat().st_mtime_ns >= (replay_path / "c0/branch.json").stat().st_mtime_ns,
                "Premature branch")
        require(result["rng_tape_sha256"] == replay["rng_tape_sha256"], "Branch RNG tape differs")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(saved["policy_rng"])
        monitor = copy.deepcopy(prefix)
        before, steps, success = main[q0]["sim_before"], saved["action_steps"], False
        active, streak, recovery_queries, forwards, exit_reason, exit_query = arm != "native", 0, 0, 0, None, None
        pools, selected_risks, stall_flags, positions, risk_trace = [], [], [], [], []
        recurred = False
        suffix = list(records(path / "suffix"))
        for index, row in enumerate(suffix):
            require(not success and int(row["query"]) == q0+index and int(row["relative_query"]) == index, "Query continuity")
            require(int(row["action_steps_before"]) == steps and bool(row["recovery_active_before"]) == active, "Recovery timeline")
            np.testing.assert_array_equal(row["sim_before"], before)
            native_noise = rng.standard_normal((10, 24)).astype(np.float32)
            component = independent_scores(monitor, row[PROBS_KEY])
            np.testing.assert_allclose(row["component_scores"], component, rtol=2e-5, atol=2e-6)
            score = independent_risk(row["component_scores"], thresholds, margins)
            np.testing.assert_allclose(row["selected_risk"], score, rtol=0, atol=1e-12)
            if int(row["candidate_count"]) == 16:
                require(active, "Candidate pool generated after recovery exit")
                values, diagnostic = check_pool(path / "pools" / str(index), arm, row, monitor, native_noise, task, index, thresholds, margins)
                pools.append(diagnostic)
                if arm not in first_pools:
                    first_pools[arm] = list(records(path / "pools" / str(index) / "candidates"))
            else:
                require(int(row["candidate_count"]) == 1 and int(row["candidate_id"]) == 0, "Invalid non-search query")
                np.testing.assert_array_equal(row["noise"], native_noise)
                np.testing.assert_allclose(row["default_risk"], score, rtol=0, atol=1e-12)
                require(not active or not np.isfinite(score), "Ready recovery query skipped its pool")
                values = np.asarray([score])
            if active:
                selected_risks.append(score)
                all_low = len(values) == 16 and np.all(values <= -1-1e-6)
                streak = streak+1 if all_low else 0
                recovery_queries += 1
                if streak >= 3:
                    active, exit_reason, exit_query = False, "population_confirmed", q0+index
                elif recovery_queries >= 12:
                    active, exit_reason, exit_query = False, "recovery_budget_exhausted", q0+index
            elif arm != "native" and score > 0:
                recurred = True
            require(bool(row["recovery_active_after"]) == active and int(row["stable_streak"]) == streak, "Exit confirmation differs")
            route_check(row)
            status = monitor.update(row[PROBS_KEY], row["proprio"][:3])
            np.testing.assert_allclose(row["alarm_scores"], [status[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], rtol=2e-5, atol=2e-6)
            for key in ("knn_score", "cosine_score", "v8_scores"):
                np.testing.assert_allclose(row[key], status[key], rtol=2e-5, atol=2e-6)
            require(bool(row["alarm"]) == bool(status["alarm"]), "Latched alarm changed")
            positions.append(row["proprio"][:3])
            require(bool(row["post_recovery_stall_valid"]) == (len(positions) >= 6), "Stall window readiness")
            stall = bool(len(positions) >= 6 and max(np.linalg.norm(a-b) for a in positions[-6:] for b in positions[-6:]) <= .005)
            require(bool(row["post_recovery_stall"]) == stall, "Stall geometry differs")
            if len(positions) >= 6:
                stall_flags.append(stall)
            risk_trace.append(dict(index=index, query=q0+index, action_step=steps,
                selected=float(score) if np.isfinite(score) else None,
                default=float(row["default_risk"]) if np.isfinite(row["default_risk"]) else None,
                recovering=bool(row["recovery_active_before"])))
            count, success = int(row["executed_action_count"]), bool(row["success"])
            require(0 < count <= min(10, 520-steps) and (success or count == min(10, 520-steps)), "Chunk/horizon changed")
            if arm == "native":
                require(q0+index < len(main), "Native suffix extended")
                for key in main[q0+index]:
                    if key not in ("inference_seconds", "environment_seconds"):
                        np.testing.assert_array_equal(row[key], main[q0+index][key], err_msg="Native suffix: "+key)
            steps += count
            before = row["sim_after"]
            require(np.isfinite(before).all(), "Non-finite simulator state")
            forwards += int(row["candidate_count"])
        require(success or steps == 520, "Incomplete suffix")
        require(branch["queries"] == len(suffix) and branch["actual_model_queries"] == result["actual_model_queries"] == forwards, "Forward ledger")
        require(branch["success"] == success and branch["final_action_steps"] == steps and
                branch["action_steps"] == steps-saved["action_steps"], "Final outcome/budget")
        if arm == "native":
            require(result["all_native_suffixes_exact"] and success == original["success"] and len(suffix) == len(main)-q0, "Native exact outcome")
        else:
            if active:
                exit_reason = "task_success" if success else "environment_horizon"
            require(branch["exit_reason"] == exit_reason and branch["exit_query"] == exit_query, "Final recovery status")
        require(branch["post_exit_recurrence"] == recurred, "Post-exit recurrence differs")
        require(branch["candidate_pools"] == len(pools) and branch["changed_chunks"] == sum(p["candidate"] != 0 for p in pools), "Pool accounting")
        finite = [r for r in selected_risks if np.isfinite(r)]
        total += forwards
        branches.append(dict(arm=arm, success=success, queries=len(suffix), actual_model_queries=forwards,
            final_action_steps=steps, exit_reason=exit_reason, exit_query=exit_query,
            recovery_queries=recovery_queries, candidate_pools=len(pools), changed_chunks=branch["changed_chunks"],
            selected_below_target=sum(r <= -1 for r in selected_risks), selected_below_alarm=sum(r < 0 for r in selected_risks),
            mean_selected_risk=float(np.mean(finite)) if finite else None,
            population_below_target=sum(p["below_target"] == 16 for p in pools),
            post_exit_recurrence=branch["post_exit_recurrence"], any_observed_stall=any(stall_flags),
            stalled_windows=sum(stall_flags), observed_stall_windows=len(stall_flags),
            inference_seconds=branch["inference_seconds"], environment_seconds=branch["environment_seconds"],
            elapsed_seconds=branch["elapsed_seconds"], pool_diagnostics=pools, risk_trace=risk_trace,
            branch_sha256=digest(path / "branch.json"), suffix_manifest_sha256=digest(path / "suffix/manifest.json")))
    if first_pools:
        reference = next(iter(first_pools.values()))
        for pool in first_pools.values():
            for a, b in zip(reference[:8], pool[:8]):
                for key in (*ALL_FIELDS, "actions", "noise", "input_sha256"):
                    np.testing.assert_array_equal(a[key], b[key], err_msg="First population across arms: "+key)
        for family in ("iid", "guided"):
            if family+"_random" in first_pools and family+"_v8" in first_pools:
                for a, b in zip(first_pools[family+"_random"], first_pools[family+"_v8"]):
                    for key in (*ALL_FIELDS, "actions", "noise", "input_sha256"):
                        np.testing.assert_array_equal(a[key], b[key], err_msg="Matched generator first pool: "+key)
    return dict(main_id=task["main_id"], benchmark=task["benchmark"], category=task["category"],
        base_task=task["base_task"], native_success=original["success"], native_queries=len(main), start_query=q0,
        trigger_heads=dict(freeze=prefix.v8.v7.first_freeze_query == q0-1,
            turbulence=prefix.v8.v7.first_turbulence_query == q0-1,
            inversion=prefix.v8.first[0] == q0-1, curvature=prefix.v8.first[1] == q0-1),
        branches=branches, c0_queries=len(c0), actual_model_queries=total)


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    require(summary["status"] == "completed" and plan["protocol"] == PROTOCOL and plan["settings"] == SETTINGS, "Run incomplete")
    require(summary["plan_sha256"] == digest(args.run / "plan.json"), "Plan changed")
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected, "Frozen source changed: "+path)
    for name, expected in summary["sources"].items():
        require(digest(args.run / "sources" / name) == expected == digest(Path(__file__).parent / name), "Runtime archive differs")
    require(summary["temporary_models_stopped"] and summary["environment_workers_stopped"] and not summary["live_replica_pids_after_cleanup"], "Cleanup incomplete")
    require(all(row["actions_and_top4_exact"] and row["concurrent_full_hb_exact"] for row in summary["model_equivalence"]), "Model equivalence failed")
    require(all(job["status"] == "completed" and not Path("/proc/%d" % job["pid"]).exists() for job in summary["tasks"]), "Incomplete jobs")
    payload = [(str(args.run), task, plan["thresholds"], plan["margins"]) for task in plan["tasks"]]
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in pool.map(audit_parent, payload):
            tasks.append(task)
            print("AUDITED "+task["main_id"], flush=True)
    require(sum(t["actual_model_queries"] for t in tasks) == summary["actual_model_queries"], "Total inference ledger")
    result = dict(status="passed", protocol=PROTOCOL, stage=plan["stage"], run=str(args.run.resolve()),
        plan_sha256=digest(args.run / "plan.json"), auditor_sha256=digest(__file__),
        parents=len(tasks), branches=sum(len(t["branches"]) for t in tasks), tasks=tasks,
        actual_model_queries=summary["actual_model_queries"], c0_queries=sum(t["c0_queries"] for t in tasks),
        candidate_pools=sum(b["candidate_pools"] for t in tasks for b in t["branches"]),
        hidden_capture=False, gpu6_used=False, new_main_coverage=0,
        verification="all native queries and suffixes, all candidates and dispatches, noise search, same-input pairing, private history and full-population exit")
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "parents", "branches", "actual_model_queries", "candidate_pools")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
