#!/usr/bin/env python3
"""Independently audit branch identities, pairing, actual swaps, and complete suffixes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_collection_preflight import (HERE, GlobalIntrinsicProfile, IntrinsicGuardMonitor,
                                       audit_as, audit_task, require)
from collection_protocol import (PROTOCOL, REPLICATES, PARAMETERS_SHA256, branch_noise,
                                event_id, load_plan, stable_id, stream_seed, verify_frozen_alarm)
from collection_routes import (HB_LAYERS, SWAP_LAYERS, ALL_FIELDS, PROBS_KEY, NATIVE_IDS_KEY,
    NATIVE_WEIGHTS_KEY, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY, INTERVENTION_PROBS_KEY)
from collection_storage import atomic_json, digest, load_snapshot, records


def expected_swap(probs, native):
    expected = native.copy()
    for layer_index, layer in enumerate(HB_LAYERS):
        if layer not in SWAP_LAYERS:
            continue
        for denoise in range(10):
            for token in range(1, 11):
                p, ids = probs[layer_index, denoise, token], native[layer_index, denoise, token]
                removed = min(ids, key=lambda expert: (float(p[expert]), int(expert)))
                added = min((expert for expert in range(32) if expert not in ids),
                            key=lambda expert: (-float(p[expert]), expert))
                expected[layer_index, denoise, token, np.flatnonzero(ids == removed)[0]] = added
    return expected


def audit_paired_task(directory, profile):
    directory = Path(directory)
    audited = audit_task(directory, profile, paired=True)
    result = json.loads((directory / "result.json").read_text())
    require(result["intervention"] and not result["invalid_pair"], "Invalid paired run")
    main_id, q0 = result["main_id"], result["first_alarm_query"]
    c0 = result["c0"]
    require(c0 == json.loads((directory / "c0/branch.json").read_text()), "C0 metadata commit")
    require(c0["parent_main_id"] == main_id and c0["event_id"] == result["event_id"], "C0 parent identity")
    require(c0["parent_commit_sha256"] == digest(directory / "main_complete.json") and
            c0["snapshot_manifest_sha256"] == digest(directory / c0["snapshot"] / "manifest.json"), "C0 parent hashes")
    require(c0["parent_prefix_queries"] == [0, c0["start_query"]], "C0 prefix")
    if "branch_id" in c0:
        require(c0["branch_id"] == stable_id(PROTOCOL, main_id, result["event_id"] or c0["snapshot"], "C0", 0), "C0 branch ID")
    audited.update(main_id=main_id, branches=[], analysis_role="perturbation")
    if q0 is None:
        require(not result["branches"] and result["event_id"] is None, "Branches without an alarm")
        return audited
    event = event_id(main_id, q0)
    require(event == result["event_id"], "Event identity mismatch")
    target = directory / "alarm_snapshot"
    saved = load_snapshot(target)
    commit_sha, snapshot_sha = digest(directory / "main_complete.json"), digest(target / "manifest.json")
    main_alarm = next(row for row in records(directory / "main") if int(row["query"]) == q0)
    expected_arms = {(arm, replicate) for arm in ("C1", "T1") for replicate in range(REPLICATES)}
    require(len(result["branches"]) == 8 and {(b["arm"], b["replicate"]) for b in result["branches"]} == expected_arms,
            "Missing or duplicate branch")
    first_rows = {}
    for reported in result["branches"]:
        arm, replicate = reported["arm"], reported["replicate"]
        path = directory / "branches" / arm / str(replicate)
        branch = json.loads((path / "branch.json").read_text())
        require(branch == reported and branch["status"] == "completed", "Branch commit mismatch")
        require(branch["branch_id"] == stable_id(PROTOCOL, main_id, event, arm, replicate), "Branch ID")
        require(branch["parent_main_id"] == main_id and branch["event_id"] == event, "Branch parent")
        require(branch["parent_prefix_queries"] == [0, q0] and branch["start_query"] == q0, "Branch prefix")
        require(branch["parent_commit_sha256"] == commit_sha and branch["snapshot_manifest_sha256"] == snapshot_sha,
                "Branch parent hashes")
        require(branch["starts_after_main_complete"] and (path / "branch.json").stat().st_mtime_ns >=
                (directory / "main_complete.json").stat().st_mtime_ns, "Branch preceded main commit")
        monitor = IntrinsicGuardMonitor(profile)
        for prefix in records(directory / "main"):
            if int(prefix["query"]) >= q0:
                break
            monitor.update(prefix[PROBS_KEY])
        count, steps, previous, prior_success = 0, int(saved["action_steps"]), saved["physics"]["sim"], False
        for row in records(path / "suffix"):
            q = int(row["query"])
            require(not prior_success, "Branch continued after success")
            require(q == q0 + count and int(row["action_steps_before"]) == steps, "Discontinuous branch")
            require(not any("hidden" in key.lower() for key in row), "Hidden column")
            if "proprio" in row:
                require(np.isfinite(row["proprio"]).all() and hashlib.sha256(row["proprio"].tobytes()).hexdigest() ==
                        row["input_component_sha256"][2].decode(), "Branch proprio input hash")
            require(np.isfinite(row["actions"]).all() and np.isfinite(row["sim_after"]).all(), "Non-finite branch")
            np.testing.assert_array_equal(row["sim_before"], previous)
            np.testing.assert_array_equal(row["noise"], branch_noise(main_id, event, replicate, q))
            for name in ("policy", "environment"):
                require(int(row[name + "_seed"]) == stream_seed(main_id, event, replicate, q, name), "Paired seed")
            if count == 0:
                first_rows[arm, replicate] = row
                np.testing.assert_array_equal(row["input_sha256"], main_alarm["input_sha256"])
                require(branch["first_input_sha256"] == row["input_sha256"].item().decode(), "Branch first observation")
            for field in ALL_FIELDS[:5]:
                require(row[field].shape == (8, 10, 11, 32 if field == PROBS_KEY else 4), "Branch HB shape")
                require(np.isfinite(row[field]).all(), "Non-finite HB")
            native, effective = row[NATIVE_IDS_KEY], row[EFFECTIVE_IDS_KEY]
            require(np.all((native >= 0) & (native < 32)) and np.all((effective >= 0) & (effective < 32)), "Expert range")
            require(np.all(np.diff(np.sort(effective, axis=-1), axis=-1) > 0), "Duplicate effective expert")
            require(np.all(row[NATIVE_WEIGHTS_KEY] >= 0) and
                    np.max(np.abs(row[NATIVE_WEIGHTS_KEY].sum(-1) - 1)) < 1e-5, "HB combine normalization")
            require(np.max(np.abs(row[PROBS_KEY].astype(np.float32).sum(-1) - 1)) < .001, "HB probability normalization")
            np.testing.assert_array_equal(row[NATIVE_WEIGHTS_KEY], row[EFFECTIVE_WEIGHTS_KEY])
            applied = arm == "T1" and count == 0
            require(bool(row["intervention_applied"]) == applied, "Wrong intervention scope")
            if applied:
                evidence_path = path / "intervention.npz"
                require(digest(evidence_path) == branch["intervention_evidence_sha256"], "Intervention evidence checksum")
                with np.load(evidence_path, allow_pickle=False) as data:
                    probability = data[INTERVENTION_PROBS_KEY]
                np.testing.assert_array_equal(probability.astype(np.float16), row[PROBS_KEY])
                np.testing.assert_array_equal(expected_swap(probability, native), effective)
                require(int(np.count_nonzero(native != effective)) == 400, "Expected exactly 400 changed route slots")
            else:
                np.testing.assert_array_equal(native, effective)
            audit_as(row)
            alarm = monitor.update(row[PROBS_KEY])
            require(bool(alarm["alarm"]) == bool(row["alarm"]), "Branch alarm mismatch")
            np.testing.assert_array_equal(np.asarray([alarm[key] for key in
                ("freeze_score", "acceleration_score", "periodicity_score")], np.float32), row["alarm_scores"])
            executed = int(row["executed_action_count"])
            require(0 < executed <= min(10, result["variant"]["horizon_steps"] - steps), "Branch action budget")
            count += 1
            steps += executed
            previous, prior_success = row["sim_after"], bool(row["success"])
        require(count == branch["queries"] and steps - saved["action_steps"] == branch["action_steps"], "Branch counts")
        require(branch["remaining_action_budget"] == result["variant"]["horizon_steps"] - saved["action_steps"], "Extended branch horizon")
        require(prior_success == branch["success"] and (prior_success or steps == result["variant"]["horizon_steps"]), "Truncated branch")
        audited["branches"].append({key: branch[key] for key in ("arm", "replicate", "queries", "action_steps", "success")})
    for replicate in range(REPLICATES):
        c1, t1 = first_rows["C1", replicate], first_rows["T1", replicate]
        np.testing.assert_array_equal(c1["noise"], t1["noise"])
        np.testing.assert_array_equal(c1["input_sha256"], t1["input_sha256"])
        # Before the first changed gate, equal input/noise must reproduce equal routes.
        np.testing.assert_array_equal(c1[PROBS_KEY][:4, 0], t1[PROBS_KEY][:4, 0])
        np.testing.assert_array_equal(c1[PROBS_KEY][4, 0], t1[PROBS_KEY][4, 0])
    return audited


def run(args):
    active_alarm_sources = verify_frozen_alarm()
    summary = json.loads((args.run / "summary.json").read_text())
    require(summary["frozen_alarm_sha256"] == PARAMETERS_SHA256, "Frozen parameters")
    require(summary["batch_size"] == 1 and 6 not in summary["gpus"], "GPU/batch constraint")
    plan = load_plan(args.run / "plan.json", summary["model"]) if summary.get("formal_collection") else None
    if plan is not None:
        require(digest(args.run / "plan.json") == summary["plan_sha256"], "Plan checksum")
    for path, sha in summary["source_sha256"].items():
        require(digest(args.run / "sources" / Path(path).name) == sha, "Archived source checksum")
    profile_path = HERE.parent / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
    require(digest(profile_path) == "a92c9d1103487ddf62e4b369c7f23b6637127730dffa8780f54423cf580e3054", "Profile hash")
    profile = GlobalIntrinsicProfile.load(profile_path)
    audited, failures = [], []
    reference_tasks = None
    reference_checks = []
    if args.reference_native_run is not None:
        reference = json.loads((args.reference_native_run / "summary.json").read_text())
        require(not reference["intervention"], "Reference must be a native run")
        reference_tasks = {task["variant"]["variant_id"]: task for task in reference["tasks"]}
    controls = set(json.loads((HERE / "benchmarks/READINESS.json").read_text())["pro_environment_generation"]["bddl_unchanged_from_base"])
    for task in summary["tasks"]:
        try:
            require(task["status"] == "completed", "Task did not complete")
            result = audit_paired_task(task["output"], profile)
            if reference_tasks is not None:
                old_task = reference_tasks[result["variant_id"]]
                old_dir, new_dir = Path(old_task["output"]), Path(task["output"])
                old_result = json.loads((old_dir / "result.json").read_text())
                new_result = json.loads((new_dir / "result.json").read_text())
                require(old_result["seed"] == new_result["seed"] and old_result["init_index"] == new_result["init_index"], "Reference seed/init")
                checked = 0
                for stream in ("main", "c0"):
                    old_rows = iter(records(old_dir / stream))
                    for actual in records(new_dir / stream):
                        expected = next(old_rows)
                        for field in expected:
                            if field not in ("inference_seconds", "environment_seconds"):
                                np.testing.assert_array_equal(actual[field], expected[field], err_msg="native reference: " + field)
                        checked += 1
                    require(next(old_rows, None) is None, "Incomplete native reference comparison")
                reference_checks.append(dict(variant_id=result["variant_id"], exact_queries=checked))
            if plan is not None:
                source = next(row for row in plan["tasks"] if row["main_id"] == result["main_id"])
                actual = json.loads((Path(task["output"]) / "result.json").read_text())
                require("branch_id" in actual["c0"], "Missing formal C0 identity")
                require("proprio" in next(records(Path(task["output"]) / "main")), "Missing formal proprio")
                require(actual["seed"] == int(source["noise_seed"]) and actual["init_index"] == int(source["init_index"]), "Formal seed/init")
                require(actual["variant"]["variant_id"] == source["variant_id"], "Formal variant")
            if result["variant_id"] in controls:
                result["analysis_role"] = "unchanged_scene_control"
            audited.append(result)
        except Exception as error:
            failures.append(dict(main_id=task.get("main_id"), directory=task["output"], error=repr(error)))
    branches = [branch for task in audited for branch in task["branches"]]
    report = dict(status="passed" if not failures else "failed", run=str(args.run.resolve()),
        auditor_sha256=digest(Path(__file__)),
        frozen_active_alarm_sources=active_alarm_sources,
        formal_collection=plan is not None, audited_mains=len(audited), planned_mains=len(summary["tasks"]),
        benchmark_counts=dict(Counter(task["benchmark"] for task in audited)),
        main_queries=sum(task["main_queries"] for task in audited), c0_queries=sum(task["c0_queries"] for task in audited),
        paired_branch_queries=sum(branch["queries"] for branch in branches), paired_branches=len(branches),
        alarmed_mains=sum(task["first_alarm_query"] is not None for task in audited),
        successes=sum(task["success"] for task in audited),
        total_storage_bytes=sum(task["total_storage_bytes"] for task in audited),
        tasks=audited, failures=failures, native_reference_checks=reference_checks)
    atomic_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}))
    return 0 if not failures else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-native-run", type=Path)
    raise SystemExit(run(parser.parse_args()))
