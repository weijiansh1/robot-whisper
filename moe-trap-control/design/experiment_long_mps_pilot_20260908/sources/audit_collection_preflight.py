#!/usr/bin/env python3
"""Audit committed native collection shards, alarm snapshots, and C0 replay."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "himoe-route-capture"))
sys.path.insert(0, str(HERE.parent / "moe-v7-0905/method"))
from collection_routes import (FIELDS, PROBS_KEY, NATIVE_IDS_KEY, NATIVE_WEIGHTS_KEY,
                               EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY, AS_FIELDS)
from collection_storage import atomic_json, digest, load_snapshot, records
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_task(directory, profile, paired=False):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    require(result["status"] == "completed" and result["main_complete"], "Main incomplete")
    require(not result["hidden_capture"] and (paired or not result["intervention"]), "Unexpected collection mode")
    require(not result.get("main_intervention", False), "Main was intervened")
    # Early Goal records used the inference GPU for rendering and had no separate field.
    require("render_gpu" in result or "model" not in result, "New task missing renderer identity")
    require(result["gpu"] in (0, 1, 2, 3, 4, 5, 7) and result.get("render_gpu", result["gpu"]) in (0, 1, 2, 3, 4, 5, 7),
            "Excluded physical GPU")
    require(result["model_metadata"]["bundle_physical_gpu"] == result["gpu"], "Model GPU identity")
    require(result["alarm_profile_sha256"] == "a92c9d1103487ddf62e4b369c7f23b6637127730dffa8780f54423cf580e3054",
            "Task alarm profile changed")
    if "model" in result:
        require(result["variant"]["suite"] == dict(goal="libero_goal", spatial="libero_spatial",
            object="libero_object", long="libero_10")[result["model"]], "Task model/suite mismatch")
        require(result["model_metadata"]["bundle_model"] == result["model"], "Server model identity")
    commit = json.loads((directory / "main_complete.json").read_text())
    require(commit["manifest_sha256"] == digest(directory / "main/manifest.json"), "Main commit mismatch")
    monitor = IntrinsicGuardMonitor(profile)
    count, steps, first_alarm, prior_state = 0, 0, None, None
    inference_seconds, environment_seconds = [], []
    raw_bytes = 0
    for row in records(directory / "main"):
        q = int(row["query"])
        require(q == count and int(row["action_steps_before"]) == steps, "Discontinuous main")
        require(not any("hidden" in field.lower() for field in row), "Hidden column present")
        for field in FIELDS:
            value = row[field]
            require(value.shape == (8, 10, 11, 32 if field == PROBS_KEY else 4), "HB shape: " + field)
            require(np.isfinite(value).all(), "Non-finite HB: " + field)
        probability = row[PROBS_KEY].astype(np.float32)
        require(np.max(np.abs(probability.sum(-1) - 1)) < .001, "Probability normalization")
        ids, weights = row[NATIVE_IDS_KEY], row[NATIVE_WEIGHTS_KEY]
        require(np.all((ids >= 0) & (ids < 32)), "Expert ID range")
        require(np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0), "Duplicate selected expert")
        require(np.all(weights >= 0) and np.max(np.abs(weights.sum(-1) - 1)) < 1e-5, "Combine normalization")
        require(np.array_equal(ids, row[EFFECTIVE_IDS_KEY]) and
                np.array_equal(weights, row[EFFECTIVE_WEIGHTS_KEY]), "Native-only routes changed")
        if AS_FIELDS[0] in row:
            audit_as(row)
        elif paired:
            raise ValueError("AS routes missing from experiment")
        alarm = monitor.update(row[PROBS_KEY])
        require(bool(alarm["alarm"]) == bool(row["alarm"]), "Offline alarm mismatch")
        scores = np.asarray([alarm[key] for key in ("freeze_score", "acceleration_score", "periodicity_score")],
                            dtype=np.float32)
        np.testing.assert_array_equal(scores, row["alarm_scores"])
        if alarm["alarm"] and first_alarm is None:
            first_alarm = q
        if prior_state is not None:
            np.testing.assert_array_equal(row["sim_before"], prior_state)
        prior_state = row["sim_after"]
        executed = int(row["executed_action_count"])
        require(0 < executed <= 10, "Executed action count")
        steps += executed
        count += 1
        raw_bytes += sum(value.nbytes for value in row.values())
        inference_seconds.append(float(row["inference_seconds"]))
        environment_seconds.append(float(row["environment_seconds"]))
        if first_alarm == q:
            saved = load_snapshot(directory / "alarm_snapshot")
            require(saved["query"] == q and saved["action_steps"] == int(row["action_steps_before"]), "Alarm snapshot position")
            np.testing.assert_array_equal(saved["physics"]["sim"], row["sim_before"])
            rng = np.random.default_rng()
            rng.bit_generator.state = saved["policy_rng"]
            np.testing.assert_array_equal(rng.standard_normal((10, 24)).astype(np.float32), row["noise"])
    require(count == result["queries"] and steps == result["action_steps"], "Main count mismatch")
    require(first_alarm == result["first_alarm_query"], "First alarm mismatch")
    require(count == commit["queries"] and steps == commit["action_steps"], "Commit count mismatch")
    require(bool(row["success"]) == result["success"] == commit["success"], "Main outcome mismatch")
    require(result["success"] or steps == result["variant"]["horizon_steps"], "Truncated main")
    for path in directory.glob("preflight_q*"):
        load_snapshot(path)
    c0 = result["c0"]
    require(c0["status"] == "passed" and c0["starts_after_main_complete"], "C0 did not pass")
    require(c0["final_success"] == result["success"], "C0 outcome mismatch")
    require((directory / "c0/manifest.json").stat().st_mtime_ns >=
            (directory / "main_complete.json").stat().st_mtime_ns, "C0 committed before main")
    suffix = (row for row in records(directory / "main") if int(row["query"]) >= c0["start_query"])
    compared = 0
    for actual in records(directory / "c0"):
        expected = next(suffix)
        route_fields = FIELDS + AS_FIELDS if AS_FIELDS[0] in expected else FIELDS
        for field in (*route_fields, "query", "action_steps_before", "executed_action_count", "actions", "noise",
                      "input_sha256", "sim_before", "sim_after", "success", "alarm", "alarm_scores"):
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
        compared += 1
    require(next(suffix, None) is None and compared == c0["compared_queries"], "Incomplete C0 suffix")
    require(compared == count - c0["start_query"], "C0 query count mismatch")
    return dict(variant_id=result["variant"]["variant_id"], benchmark=result["variant"]["benchmark"],
        suite=result["variant"]["suite"], task_name=result["variant"]["task_name"],
        category=result["variant"]["category"], directory=str(directory.resolve()), gpu=result["gpu"],
        main_queries=count, c0_queries=compared, action_steps=steps, success=result["success"],
        first_alarm_query=first_alarm, main_inference_seconds=sum(inference_seconds),
        main_environment_seconds=sum(environment_seconds), main_raw_bytes=raw_bytes,
        main_compressed_bytes=result["main_shards"]["bytes"],
        main_mean_inference_seconds=float(np.mean(inference_seconds)),
        main_mean_environment_seconds=float(np.mean(environment_seconds)),
        total_storage_bytes=sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()))


def audit_as(row):
    for field in AS_FIELDS:
        require(row[field].shape == (4, 10, 11, 3 if field == AS_FIELDS[0] else 1), "AS shape")
        require(np.isfinite(row[field]).all(), "Non-finite AS")
    probs, ids, weights, effective, effective_weights = (row[field] for field in AS_FIELDS)
    require(np.all((ids >= 0) & (ids < 3)), "AS expert range")
    require(np.max(np.abs(probs.astype(np.float32).sum(-1) - 1)) < .01, "AS normalization")
    require(np.array_equal(ids, effective) and np.array_equal(weights, effective_weights), "AS route changed")
    selected = np.take_along_axis(probs, ids.astype(np.int64), axis=-1)
    require(np.array_equal(selected, weights) and np.array_equal(selected[..., 0], probs.max(-1)), "AS actual combine weight")


def run(args):
    profile_path = HERE.parent / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
    profile_sha = digest(profile_path)
    require(profile_sha == "a92c9d1103487ddf62e4b369c7f23b6637127730dffa8780f54423cf580e3054", "Alarm profile changed")
    frozen_sha = digest(HERE / "design/frozen_alarm_comparison_20260908/profiles/parameters.json")
    require(frozen_sha == "f4e65d359411309756b008423d14d8f3e5d6bf604ee71b63ae8a7e510c5f8216", "Frozen parameters changed")
    profile = GlobalIntrinsicProfile.load(profile_path)
    chosen, attempts = {}, []
    for root in args.runs:
        summary = json.loads((root / "summary.json").read_text())
        require(summary["frozen_alarm_sha256"] == frozen_sha, "Run alarm parameters mismatch")
        require(6 not in summary["gpus"] and summary["batch_size"] == 1, "GPU or batch mismatch")
        for task in summary["tasks"]:
            variant_id = task["variant"]["variant_id"]
            attempts.append(dict(variant_id=variant_id, status=task["status"], directory=task["output"]))
            if variant_id not in chosen or task["status"] == "completed":
                chosen[variant_id] = task
    audited, failures = [], []
    for variant_id, task in chosen.items():
        try:
            require(task["status"] == "completed", "No completed attempt")
            audited.append(audit_task(task["output"], profile))
        except Exception as error:
            failures.append(dict(variant_id=variant_id, error=repr(error)))
    main_queries = sum(task["main_queries"] for task in audited)
    total_bytes = sum(task["total_storage_bytes"] for task in audited)
    report = dict(status="passed" if not failures else "failed", unique_variants=len(chosen),
        audited_variants=len(audited), main_queries=main_queries, c0_queries=sum(task["c0_queries"] for task in audited),
        action_steps=sum(task["action_steps"] for task in audited), successes=sum(task["success"] for task in audited),
        alarmed_trajectories=sum(task["first_alarm_query"] is not None for task in audited),
        alarm_profile_sha256=profile_sha, frozen_parameters_sha256=frozen_sha,
        main_compressed_bytes=sum(task["main_compressed_bytes"] for task in audited),
        total_storage_bytes=total_bytes, mean_total_bytes_per_trajectory=total_bytes / max(len(audited), 1),
        disk_free_bytes=shutil.disk_usage(HERE).free, failures=failures, tasks=audited, attempts=attempts,
        checks=["chunk checksums and continuity", "HB shapes and finite probabilities",
                "native/effective actual IDs and weights", "no hidden columns", "frozen v7 offline reproduction",
                "pre-inference alarm snapshot and policy RNG", "complete main horizon or success",
                "post-main C0 full suffix exact input/action/route/physics/termination"])
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "unique_variants", "audited_variants", "main_queries",
        "c0_queries", "action_steps", "successes", "alarmed_trajectories", "total_storage_bytes", "failures")}))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
