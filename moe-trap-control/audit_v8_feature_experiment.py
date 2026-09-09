#!/usr/bin/env python3
"""Audit complete paired branches, real dispatch, causal targets and query accounting."""

import argparse
import concurrent.futures
import copy
import json
from pathlib import Path

import numpy as np

from collection_storage import atomic_json, digest, records, load_snapshot
from collection_routes import (ALL_FIELDS, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                                EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY)
from v8_feature_control import (PROTOCOL, ARMS, V8Monitor, make_bias, noise_for, seed_for,
                                flow_features, normalize)
from audit_long_continuation import route_check, audit_as


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability_weights(p, ids, weights):
    selected = np.take_along_axis(p, ids.astype(np.int64), -1)
    require(np.all(selected.min(-1) >= np.partition(p, -4, axis=-1)[..., -4] - 1e-6), "Dispatched experts are not effective top4")
    np.testing.assert_allclose(weights, selected / selected.sum(-1, keepdims=True), atol=1e-5, rtol=0)


def audit_parent(payload):
    run, task, arms, replicates = payload
    parent = Path(task["parent_directory"])
    directory = Path(run) / "tasks" / task["main_id"]
    result = json.loads((directory / "result.json").read_text())
    original = json.loads((parent / "result.json").read_text())
    require(result["status"] == "completed" and not result["invalid_pair"], "Incomplete parent job")
    require(result["protocol"] == PROTOCOL and result["gpu"] in (0, 1, 2, 3) and result["render_gpu"] in (0, 3), "Protocol/device mismatch")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["reused_main"], "Native/no-hidden contract")
    require(result["main_id"] == task["main_id"] and result["seed"] == original["seed"] and result["init_index"] == original["init_index"], "Parent seed/identity")
    for file, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
        require(digest(parent / file) == task[key] == result[key], "Parent commitment changed")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Checkpoint/runtime identity")
    main, c0 = list(records(parent / "main")), list(records(directory / "c0"))
    require(len(main) == len(c0) == task["parent_queries"] == result["c0"]["compared_queries"], "Full C0 length")
    require(result["c0"]["status"] == "passed" and result["c0"]["final_success"] == original["success"], "C0 endpoint")
    monitor, q0 = V8Monitor(), task["events"][0]["start_query"]
    prefix = None
    for expected, actual in zip(main, c0):
        route_check(actual)
        for field in expected:
            if field not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(actual[field], expected[field], err_msg="C0 " + field)
        monitor.update(actual[PROBS_KEY])
        if int(actual["query"]) == q0 - 1:
            prefix = copy.deepcopy(monitor)
    require(monitor.first_alarm == task["first_v8_alarm"] == q0 - 1 == prefix.first_alarm, "Frozen first v8 timing")
    event = result["events"][0]
    for key, expected in task["events"][0].items():
        require(event[key] == expected, "Scheduled event mismatch")
    location = directory / "events" / event["event_id"] / "snapshot"
    require(event["snapshot_manifest_sha256"] == digest(location / "manifest.json"), "Snapshot commitment")
    saved = load_snapshot(location)
    require(saved["query"] == q0 and saved["action_steps"] == int(main[q0]["action_steps_before"]), "Snapshot position")
    require(len(result["branches"]) == len(arms) * replicates, "Missing paired arms")
    seen, total, branches = set(), len(c0), []
    log_bias_residual_max = 0.
    initial_shadow = {}
    for branch in result["branches"]:
        arm, rep = branch["arm"], branch["replicate"]
        require((rep, arm) not in seen and arm in arms and 0 <= rep < replicates, "Duplicated arm/replicate")
        seen.add((rep, arm))
        path = directory / "branches" / ("r%d_%s" % (rep, arm))
        require(branch == json.loads((path / "branch.json").read_text()) and branch["status"] == "completed", "Branch commit")
        suffix, control = list(records(path / "suffix")), list(records(path / "control"))
        require(len(suffix) == branch["queries"] and len(control) == min(ARMS[arm]["duration"], len(suffix)), "Incomplete suffix/control capture")
        require(branch["starts_after_c0"] and branch["starts_after_main_complete"], "Branch started before C0/main")
        effective_monitor = copy.deepcopy(prefix)
        previous = main[q0 - 1][PROBS_KEY]
        before = main[q0]["sim_before"]
        steps = saved["action_steps"]
        measures, released = [], []
        for index, row in enumerate(suffix):
            require(int(row["query"]) == q0 + index and int(row["relative_query"]) == index, "Suffix query index")
            require(int(row["action_steps_before"]) == steps, "Suffix action timeline")
            np.testing.assert_array_equal(row["sim_before"], before)
            count, success = int(row["executed_action_count"]), bool(row["success"])
            require(0 < count <= min(10, 520 - steps) and (success or count == min(10, 520 - steps)), "Invalid execution chunk")
            require(not success or index == len(suffix) - 1, "Continued after success")
            np.testing.assert_array_equal(row["noise"], noise_for(task["main_id"], rep, index))
            require(int(row["policy_seed"]) == seed_for(task["main_id"], rep, index, "policy"), "Policy stream")
            for key, offset in (("environment_first_seed", 0), ("environment_last_seed", count - 1)):
                require(int(row[key]) == seed_for(task["main_id"], rep, steps + offset, "environment"), "Environment stream")
            active = index < ARMS[arm]["duration"]
            require(bool(row["control_active"]) == active, "Incorrect control duration")
            if active:
                evidence = control[index]
                shadow = {key: evidence["shadow/" + key] for key in ALL_FIELDS}
                shadow.update(actions=evidence["shadow_actions"], noise=evidence["noise"])
                route_check(shadow)
                np.testing.assert_array_equal(evidence["noise"], row["noise"])
                np.testing.assert_array_equal(evidence["input_sha256"], row["input_sha256"])
                np.testing.assert_array_equal(evidence["previous_shadow"], previous)
                bias = make_bias(shadow[PROBS_KEY], previous, ARMS[arm]["operator"], ARMS[arm]["strength"],
                                 seed_for(task["main_id"], rep, index, "direction"))
                np.testing.assert_allclose(evidence["logit_bias"], bias, atol=2e-6, rtol=2e-5)
                native, effective = evidence["native_probs_fp32"], evidence["effective_probs_fp32"]
                require(native.shape == effective.shape == (8, 10, 11, 32), "Controlled probability shape")
                require(np.isfinite(native).all() and np.isfinite(effective).all(), "Non-finite control probabilities")
                np.testing.assert_allclose(native.sum(-1), 1., atol=1e-6)
                np.testing.assert_allclose(effective.sum(-1), 1., atol=1e-6)
                audit_as(row)
                np.testing.assert_array_equal(row[PROBS_KEY], native.astype(np.float16))
                # CUDA autocast rounds the logit addition; softmax(log(p)+bias) is
                # not an exact reconstruction without the original absolute logits.
                observed_bias = np.log(np.maximum(effective.astype(np.float64), 1e-30)) - np.log(np.maximum(native, 1e-30))
                residual = observed_bias - bias
                residual -= residual.mean(-1, keepdims=True)
                log_bias_residual_max = max(log_bias_residual_max, float(np.abs(residual).max()))
                probability_weights(native, row[NATIVE_IDS_KEY], row[NATIVE_WEIGHTS_KEY])
                probability_weights(effective, row[EFFECTIVE_IDS_KEY], row[EFFECTIVE_WEIGHTS_KEY])
                for field in (NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY):
                    require(np.all(np.diff(np.sort(row[field], axis=-1), axis=-1) > 0), "Duplicate expert in dispatch")
                np.testing.assert_array_equal(row[NATIVE_IDS_KEY][:4], row[EFFECTIVE_IDS_KEY][:4])
                np.testing.assert_array_equal(row[NATIVE_IDS_KEY][:, :, 0], row[EFFECTIVE_IDS_KEY][:, :, 0])
                np.testing.assert_array_equal(row[NATIVE_WEIGHTS_KEY][:, :, 0], row[EFFECTIVE_WEIGHTS_KEY][:, :, 0])
                action_rms = np.sqrt(np.square(row["actions"].astype(np.float64) - evidence["shadow_actions"]).mean())
                np.testing.assert_allclose(row["action_rms_vs_shadow"], action_rms, rtol=1e-6, atol=1e-8)
                changed = np.any(np.sort(row[EFFECTIVE_IDS_KEY][4:, :, 1:], -1) !=
                                 np.sort(shadow[EFFECTIVE_IDS_KEY][4:, :, 1:], -1), -1).mean()
                np.testing.assert_allclose(row["changed_top4_fraction"], changed, atol=1e-7, rtol=0)
                np.testing.assert_allclose(row["bias_rms"], np.sqrt(np.square(bias[4:, :, 1:]).mean()), atol=1e-7)
                raw_shadow, raw_effective = flow_features(shadow[PROBS_KEY])[0], flow_features(effective)[0]
                root_previous = np.sqrt(normalize(previous)[4:, -1, 1:])
                mobility_shadow = np.linalg.norm(np.sqrt(normalize(shadow[PROBS_KEY])[4:, -1, 1:]) - root_previous, axis=-1).mean() / np.sqrt(2.)
                mobility_effective = np.linalg.norm(np.sqrt(normalize(effective)[4:, -1, 1:]) - root_previous, axis=-1).mean() / np.sqrt(2.)
                measures.append([float(raw_shadow[0] - raw_effective[0]),
                    float(raw_effective[1] / max(raw_shadow[1], 1e-12)), float(mobility_effective / max(mobility_shadow, 1e-12)),
                    float(row["changed_top4_fraction"]), float(action_rms), float(row["bias_rms"])])
                previous = shadow[PROBS_KEY]
            else:
                route_check(row)
                effective, previous = row[PROBS_KEY], row[PROBS_KEY]
                shadow = row
            if index == 0:
                key = rep
                value = (row["input_sha256"], shadow[PROBS_KEY])
                if key in initial_shadow:
                    for a, b in zip(value, initial_shadow[key]):
                        np.testing.assert_array_equal(a, b)
                else:
                    initial_shadow[key] = value
            status = effective_monitor.update(effective)
            np.testing.assert_allclose(row["v8_effective_raw"], status["v8_raw"], rtol=2e-5, atol=1e-7)
            np.testing.assert_allclose(row["v8_effective_scores"], status["v8_scores"], rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(row["v8_effective_freeze"], status["freeze_score"], rtol=2e-5, atol=2e-6)
            if index >= 5:
                released.append([float(index), float(status["v8_raw"][0]), float(status["v8_raw"][1]), float(status["freeze_score"])])
            steps += count
            before = row["sim_after"]
        require(steps == 520 or bool(suffix[-1]["success"]), "Truncated suffix")
        require(branch["action_steps"] == steps - saved["action_steps"] and branch["success"] == bool(suffix[-1]["success"]), "Final outcome mismatch")
        require(branch["deployment_model_queries"] == len(suffix) + len(control), "Shadow compute omitted")
        total += len(suffix) + len(control)
        branches.append(dict(arm=arm, replicate=rep, success=branch["success"], queries=len(suffix),
            controlled_queries=len(control), deployment_model_queries=branch["deployment_model_queries"],
            manipulation_mean=np.mean(measures, axis=0).tolist() if measures else None,
            release_first=released[0] if released else None,
            release_after_seven=released[6] if len(released) > 6 else None))
    require(result["actual_model_queries"] == total, "Parent query ledger differs")
    return dict(main_id=task["main_id"], benchmark=task["benchmark"], category=task["category"],
        analysis_role=task["analysis_role"], base_task=original["variant"]["base_task"],
        native_success=original["success"], active_heads=task["active_heads"],
        first_v8_alarm=task["first_v8_alarm"], c0_queries=len(c0), actual_model_queries=total,
        max_centered_log_bias_residual=log_bias_residual_max, branches=branches)


def run(args):
    run_path = args.run.resolve()
    summary = json.loads((run_path / "summary.json").read_text())
    plan = json.loads((run_path / "plan.json").read_text())
    require(summary["status"] == "completed", "Collection has not completed")
    require(summary["plan_sha256"] == digest(run_path / "plan.json"), "Plan changed")
    for name, expected in summary["sources"].items():
        require(digest(run_path / "sources" / name) == expected == digest(Path(__file__).parent / name), "Runtime source changed: " + name)
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected, "Frozen source changed")
    require(summary["temporary_models_stopped"] and summary["environment_workers_stopped"] and not summary["live_replica_pids_after_cleanup"], "Incomplete process cleanup")
    require(all(not Path("/proc/%d" % task["pid"]).exists() for task in summary["tasks"]), "A feature environment worker is still live")
    for check in summary["feature_hook_preflight"]:
        require(check["zero_bias_exact"] and check["release_exact"], "Feature hook preflight failed")
    payloads = [(str(run_path), t, plan["arms"], plan["replicates"]) for t in plan["tasks"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        tasks = []
        for task in pool.map(audit_parent, payloads):
            tasks.append(task)
            print("AUDITED " + task["main_id"], flush=True)
    total = sum(t["actual_model_queries"] for t in tasks)
    require(total == summary["actual_model_queries"], "Run query ledger differs")
    result = dict(status="passed", run=str(run_path), auditor_sha256=digest(__file__), tasks=tasks,
        parents=len(tasks), branches=sum(len(t["branches"]) for t in tasks), actual_model_queries=total,
        c0_queries=sum(t["c0_queries"] for t in tasks), new_main_coverage=0,
        manipulation_columns=["frontback_log_ratio_increase", "curvature_ratio_to_shadow", "mobility_ratio_to_shadow", "changed_top4_fraction", "action_rms", "bias_rms"],
        release_columns=["relative_query", "frontback_inversion", "curvature_raw", "freeze_score"],
        numeric_scope="actual native/effective probabilities reproduce actual top4/weights; requested bias regenerated; absolute logits not stored, so BF16 addition is not independently reconstructed",
        max_centered_log_bias_residual=max(t["max_centered_log_bias_residual"] for t in tasks),
        frozen_alarm=True, hidden_capture=False, gpu6_used=False)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "parents", "branches", "actual_model_queries")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    run(parser.parse_args())
