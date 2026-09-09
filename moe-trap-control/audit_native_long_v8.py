#!/usr/bin/env python3
"""Audit original Long coverage, exact native replays, and every selector branch."""

import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_long_continuation import require, route_check
from audit_v8_closed_loop import audit_parent
from collection_protocol import stable_id
from collection_routes import PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from fixed_recovery_control import TriggerMonitor
from native_long_runtime import ROOT, COMMIT, verify_source
from v8_closed_loop import PROTOCOL, SETTINGS


def audit_main(payload):
    run, task = payload
    run, path = Path(run), Path(task["parent_directory"])
    original = json.loads((path / "result.json").read_text())
    commit = json.loads((path / "main_complete.json").read_text())
    replay_dir = run / "replays" / task["main_id"]
    replay = json.loads((replay_dir / "result.json").read_text())
    for result in (original, replay):
        require(result["status"] == "completed" and result["main_complete"] and not result["invalid_pair"], "Incomplete original main/C0")
        require(not result["main_intervention"] and not result["hidden_capture"] and not result["perturbation"], "Main collection contract")
        require(result["gpu"] in (0, 1, 2, 3) and result["render_gpu"] in (0, 3), "Wrong native device")
        require(result["original_libero_root"] == str(ROOT) and result["original_libero_commit"] == COMMIT, "Native source differs")
        require(result["variant"] == task["variant"] and result["main_id"] == task["main_id"] and
            result["seed"] == task["noise_seed"] and result["init_index"] == task["init_index"], "Native task identity")
    require(original["job_kind"] == "native_main" and not original["reused_main"], "Main is not a new rollout")
    require(replay["job_kind"] == "fixed_replay" and replay["reused_main"] and
        replay["c0"]["status"] == "passed" and replay["online_triggers_exact"], "Missing exact C0")
    require(digest(path / "main_complete.json") == task["parent_commit_sha256"] == replay["parent_commit_sha256"], "Parent commit")
    require(digest(path / "main/manifest.json") == task["parent_manifest_sha256"] == commit["manifest_sha256"], "Main manifest")
    require(digest(original["bddl_path"]) == task["variant"]["bddl_sha256"] and
        original["initial_state_sha256"] == task["initial_state_sha256"], "Native BDDL/state")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout", "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(replay["model_metadata"][key] == original["model_metadata"][key], "Replay model changed")
    main, c0 = list(records(path / "main")), list(records(replay_dir / "c0"))
    require(len(main) == len(c0) == original["queries"] == task["parent_queries"] == commit["queries"], "Incomplete native trace")
    require(original["actual_model_queries"] == replay["actual_model_queries"] == len(main), "Native forward ledger")
    saved = load_snapshot(path / "preflight_q000")
    require(saved["query"] == saved["action_steps"] == 0, "Initial snapshot time")
    generator = np.random.default_rng(task["noise_seed"])
    require(saved["policy_rng"] == generator.bit_generator.state, "Original policy seed")
    monitor, steps, successful, before = TriggerMonitor(), 0, False, saved["physics"]["sim"]
    for q, (row, actual) in enumerate(zip(main, c0)):
        require(not successful and int(row["query"]) == q and int(row["action_steps_before"]) == steps, "Main timeline")
        route_check(row)
        require(not any("hidden" in key.lower() for key in row), "Forbidden hidden state")
        for key in row:
            if key not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(actual[key], row[key], err_msg="Full native C0: "+key)
        np.testing.assert_array_equal(row["sim_before"], before)
        np.testing.assert_array_equal(row["noise"], generator.standard_normal((10, 24)).astype(np.float32))
        alarm = monitor.update(row[PROBS_KEY], row["proprio"][:3])
        require(bool(row["alarm"]) == alarm["alarm"], "Main causal alarm")
        count, successful = int(row["executed_action_count"]), bool(row["success"])
        require(1 <= count <= 10 and (count == 10 or successful), "Native chunk length")
        require(np.isfinite(row["actions"]).all() and np.isfinite(row["sim_after"]).all(), "Non-finite main")
        steps += count
        before = row["sim_after"]
    require(steps == original["action_steps"] == commit["action_steps"] and
        successful == original["success"] == task["native_success"] == commit["success"], "Main outcome")
    require(successful or steps == 520, "Incomplete main horizon")
    require(monitor.first == task["first_alarms"] == original["first_alarms"], "Main first alarms")
    first = monitor.first["v8_frozen"]
    effective = first >= 0 and first+1 < len(main)
    require(bool(task["events"]) == effective, "Alarm cohort was selected using something besides causal timing")
    if effective:
        alarm_snapshot = load_snapshot(path / "alarm_snapshot")
        require(alarm_snapshot["query"] == first+1, "Original alarm snapshot not saved")
        np.testing.assert_array_equal(alarm_snapshot["physics"]["sim"], main[first+1]["sim_before"])
    require(digest(replay_dir / "environment_rng/manifest.json") == replay["rng_tape_sha256"], "Environment tape manifest")
    tape = list(records(replay_dir / "environment_rng"))
    require(len(tape) == steps and all(int(r["query"]) == i for i, r in enumerate(tape)), "Incomplete RNG tape")
    return dict(main_id=task["main_id"], task_index=task["variant"]["registry_index"],
        base_task=task["base_task"], init_index=task["init_index"], native_success=successful,
        native_queries=len(main), action_steps=steps, first_v8_alarm=first, effective_alarm=effective,
        first_alarms=monitor.first, main_and_c0_queries=2*len(main),
        snapshot_verified=True, full_c0_exact=True, parent_manifest_sha256=task["parent_manifest_sha256"])


def run(args):
    plan = json.loads((args.run / "plan.json").read_text())
    predeclared = json.loads((args.run / "cohort_plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    require(summary["status"] == "completed" and plan["protocol"] == PROTOCOL and plan["settings"] == SETTINGS, "Run incomplete")
    require(plan["benchmark"] == "native_long" and plan["original_source"] == verify_source(), "Native source")
    require(summary["plan_sha256"] == digest(args.run / "plan.json") and
        summary["cohort_plan_sha256"] == plan["cohort_plan_sha256"] == digest(args.run / "cohort_plan.json"), "Frozen plan commitment")
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected, "Source changed: "+path)
    for name, expected in summary["sources"].items():
        require(digest(args.run / "sources" / name) == expected == digest(Path(__file__).parent / name), "Archived runtime differs")
    require(summary["temporary_models_stopped"] and summary["environment_workers_stopped"] and
        not summary["live_replica_pids_after_cleanup"], "Cleanup incomplete")
    require(all(j["status"] == "completed" and not Path("/proc/%d" % j["pid"]).exists() for j in summary["tasks"]), "Incomplete job")
    require(all(row["actions_and_top4_exact"] and row["concurrent_full_hb_exact"] for row in summary["model_equivalence"]), "Model equivalence")
    expected = {t["main_id"]: t for t in predeclared["tasks"]}
    require(set(expected) == {t["main_id"] for t in plan["cohort"]}, "Predeclared coverage incomplete")
    for task in plan["cohort"]:
        require(all(task[key] == value for key, value in expected[task["main_id"]].items()), "Predeclared task changed")
        seed = int(stable_id("native_long_v8_20260909", task["variant_id"], task["init_index"], "policy_seed")[:8], 16)
        require(seed == task["noise_seed"], "Outcome-dependent seed")
    require({t["main_id"] for t in plan["tasks"]} == {t["main_id"] for t in plan["cohort"] if t["events"]}, "Alarm cohort omission")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        mains = list(pool.map(audit_main, [(str(args.run), t) for t in plan["cohort"]]))
        branches = []
        for result in pool.map(audit_parent, [(str(args.run), t, plan["thresholds"], plan["margins"]) for t in plan["tasks"]]):
            branches.append(result)
            print("AUDITED_BRANCHES "+result["main_id"], flush=True)
    queries = sum(m["main_and_c0_queries"] for m in mains)+sum(t["actual_model_queries"]-t["c0_queries"] for t in branches)
    require(queries == summary["actual_model_queries"] == sum(j["actual_model_queries"] for j in summary["tasks"]), "Total forward ledger")
    result = dict(status="passed", protocol=PROTOCOL, stage=plan["stage"], run=str(args.run.resolve()),
        plan_sha256=digest(args.run / "plan.json"), cohort_plan_sha256=digest(args.run / "cohort_plan.json"),
        auditor_sha256=digest(__file__), parents=len(mains), new_main_coverage=len(mains),
        alarmed_parents=len(branches), branches=sum(len(t["branches"]) for t in branches),
        cohort=mains, tasks=branches, actual_model_queries=queries,
        c0_queries=sum(m["native_queries"] for m in mains),
        candidate_pools=sum(b["candidate_pools"] for t in branches for b in t["branches"]),
        hidden_capture=False, gpu6_used=False,
        verification="Every predeclared original task, complete main and exact C0; all candidate shards, actual dispatches, RNG and private signal history")
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "parents", "alarmed_parents", "branches", "actual_model_queries")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
