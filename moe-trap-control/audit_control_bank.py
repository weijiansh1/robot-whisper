#!/usr/bin/env python3
"""Audit fresh mains, independent controller math, exact dispatch, and cohort outcomes."""

import argparse
from collections import Counter
import concurrent.futures
import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_long_continuation import require, route_check
from audit_native_long_modes_labels import components, expected_noise, signature, text, threshold
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from control_bank import PROTOCOL, SETTINGS, ARMS, FAMILIES, PHYSICAL, TRANSFORM
from v8_closed_loop import score_status
from v82_closed_loop import V82TriggerMonitor


def equal_records(left, right, label):
    require(len(left) == len(right), label+" length")
    for a, b in zip(left, right):
        for key in a:
            if key not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(a[key], b[key], err_msg=label+": "+key)


def identity(result, task, original):
    require(result["status"] == "completed" and not result["invalid_pair"], "Complete result")
    require(result["protocol"] == PROTOCOL and result["contract"] == SETTINGS, "Frozen runtime")
    require(result["gpu"] in (0,1,2,3) and result["render_gpu"] in (0,3), "GPU exclusion")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["reused_main"], "Collection contract")
    require(result["main_id"] == task["main_id"] and result["seed"] == original["seed"] and
        result["init_index"] == original["init_index"] and result["cohort_group"] == task["cohort_group"], "Parent identity")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Model identity")


def expected_operator(family, scores, tau, main_id, repeat):
    if family == "random_switch":
        key = json.dumps([PROTOCOL, main_id, repeat, "operator"], separators=(",", ":"))
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16], "little"))
        return ("resample", "withdraw4", "smooth2", "replan2", "side_plus", "side_minus")[int(rng.integers(6))]
    if family != "moe_switch":
        return family
    mask, _ = signature(scores, tau)
    if not np.isfinite(scores).all():
        return "resample"
    if np.array_equal(mask, [True,False,False,False,False]):
        return "withdraw4"
    if mask[1]:
        return "smooth2"
    if mask[2]:
        return "side_plus" if repeat == 0 else "side_minus"
    return "replan2" if mask[3:].any() else "resample"


def cap(vector, radius):
    value = np.asarray(vector, float)
    return value/max(1., float(np.linalg.norm(value))/radius)


def expected_waypoints(operator, initial, history, gripper):
    sign = 1. if gripper > 0 else -1.
    lift4, lift8 = initial+[0,0,.04], initial+[0,0,.08]
    if operator == "withdraw4":
        return np.stack((lift4, lift4+np.r_[cap(history[2,:2]-initial[:2], .04), 0.])), [8,8], [sign]*2
    if operator == "lift8":
        return lift8[None], [24], [sign]
    if operator == "retrace":
        return np.stack([initial+cap(p-initial, .08) for p in history]), [8]*3, [sign]*3
    if operator.startswith("side_"):
        dx, dy = initial[:2]-history[2,:2]
        direction = np.array([-dy, dx])
        direction = direction/np.linalg.norm(direction) if np.linalg.norm(direction) >= 1e-5 else np.array([1.,0.])
        middle = lift4+np.r_[direction*(.04 if operator == "side_plus" else -.04), 0.]
        return np.stack((lift4,middle,lift4)), [8]*3, [sign]*3
    if operator == "open_lift8":
        return np.stack((initial,lift8)), [8,16], [-1.]*2
    return np.stack((initial,initial)), [8,8], ([-sign,sign] if operator == "grip_cycle" else [sign]*2)


def audit_task(run, task, plan):
    main_id, parent = task["main_id"], Path(task["parent_directory"])
    original = json.loads((parent / "result.json").read_text())
    main = list(records(parent / "main"))
    replay_dir = run / "replays" / main_id
    replay = json.loads((replay_dir / "result.json").read_text())
    identity(replay, task, original)
    c0 = list(records(replay_dir / "c0"))
    equal_records(main, c0, "Full C0")
    require(len(c0) == task["parent_queries"] == replay["c0"]["compared_queries"], "C0 query count")
    require(replay["c0"]["status"] == "passed" and replay["online_triggers_exact"] and replay["v82_triggers_exact"], "C0 trigger verification")
    live = V82TriggerMonitor()
    for row in c0:
        route_check(row)
        live.update(row[PROBS_KEY], row["proprio"][:3])
    require(live.first == task["first_alarms"] == replay["replay_first_alarms"], "Frozen alarms")
    event = replay["events"][0]
    for key, value in task["events"][0].items():
        require(event[key] == value, "Event contract")
    q0, steps0 = event["start_query"], event["action_steps_before"]
    require(q0 == live.first["v82_frozen"]+1 < len(main), "Causal fork")
    location = replay_dir / "events" / event["event_id"] / "snapshot"
    saved = load_snapshot(location)
    require(digest(location / "manifest.json") == event["snapshot_manifest_sha256"], "Snapshot hash")
    np.testing.assert_array_equal(saved["physics"]["sim"], main[q0]["sim_before"])
    require(saved["action_steps"] == steps0 and saved["query"] == q0, "Snapshot timeline")
    tape = list(records(replay_dir / "environment_rng"))
    require(len(tape) == original["action_steps"] and digest(replay_dir / "environment_rng/manifest.json") == replay["rng_tape_sha256"], "RNG tape")
    prefix, previous, streak = V82TriggerMonitor(), np.full(5,np.nan), np.zeros(5,int)
    for q, row in enumerate(main[:q0]):
        previous = score_status(prefix.update(row[PROBS_KEY], row["proprio"][:3]))
        mask, _ = signature(previous, threshold(q, plan))
        streak = np.where(mask, streak+1, 0)
    histories = np.stack([main[q0-offset]["proprio"][:3] for offset in (1,2,3)])
    last_native = main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1]
    outcomes, traces, total_queries = [], {}, len(c0)
    for arm in ARMS:
        family, repeat = ("native",-1) if arm == "native" else (arm[:-3],int(arm[-1]))
        directory = run / "events" / event["event_id"] / arm
        result, branch = (json.loads((directory / name).read_text()) for name in ("result.json","branch.json"))
        identity(result, task, original)
        require(result["branches"] == [branch] and branch["status"] == "completed", "Branch commitment")
        require(branch["starts_after_c0"] and branch["starts_after_main_complete"] and
            result["c0"]["replay_result_sha256"] == digest(replay_dir / "result.json"), "C0 ordering/linkage")
        probe_rows = list(records(directory / "entry_probe"))
        require(len(probe_rows) == 1 and digest(directory / "entry_probe/manifest.json") == branch["entry_probe_manifest_sha256"], "Entry probe")
        probe = probe_rows[0]
        route_check(probe)
        for key in (*ALL_FIELDS,"actions","noise","proprio","input_sha256","input_component_sha256","sim_before"):
            np.testing.assert_array_equal(probe[key], main[q0][key], err_msg="Original pre-intervention probe: "+key)
        np.testing.assert_array_equal(probe["sim_before"], probe["sim_after"])
        require(int(probe["executed_action_count"]) == 0, "Probe did not execute")
        np.testing.assert_allclose(probe["component_scores"], components(prefix, probe[PROBS_KEY]), rtol=2e-5, atol=2e-6)
        _, mode = signature(probe["component_scores"], threshold(q0, plan))
        require(mode == branch["entry_mode"] == text(probe["mode"]), "Pre-treatment mode")
        operator = expected_operator(family, probe["component_scores"], threshold(q0,plan), main_id, repeat)
        require(operator == branch["operator"], "Independent supervisory choice")
        physical = list(records(directory / "physical")) if operator in PHYSICAL else []
        rows = list(records(directory / "suffix"))
        traces[arm] = (operator, physical, rows)
        state, steps, success = main[q0]["sim_before"].copy(), steps0, False
        error_decreases, prediction_errors = 0, []
        if physical:
            targets, durations, grips = expected_waypoints(operator, physical[0]["eef_before"], histories, last_native[6])
            np.testing.assert_allclose(branch["recovery_targets"], targets, rtol=0, atol=1e-12)
            require(branch["phase_durations"] == durations and branch["phase_grippers"] == grips, "Physical phases")
            np.testing.assert_array_equal(branch["history_eef"], histories)
            phases = np.repeat(np.arange(len(durations)), durations)
            scale = np.asarray(branch["controller_translation_scale"], float)
            for index, row in enumerate(physical):
                phase = int(phases[index])
                require(int(row["phase"]) == phase and int(row["action_step"]) == steps and not success, "Physical timeline")
                np.testing.assert_array_equal(row["sim_before"], state)
                np.testing.assert_allclose(row["target"], targets[phase], rtol=0, atol=1e-12)
                action = np.zeros(7)
                if operator not in ("hold16","grip_cycle"):
                    action[:3] = cap(targets[phase]-row["eef_before"], .01)/scale
                action[6] = grips[phase]
                np.testing.assert_allclose(row["action"], action, rtol=0, atol=1e-12)
                require(np.linalg.norm(row["action"][:3]*scale) <= .01000000001, "Cartesian command bound")
                expected_errors = [np.linalg.norm(row[key]-targets[phase]) for key in ("eef_before","eef_after")]
                np.testing.assert_allclose([row["target_error_before_m"],row["target_error_after_m"]], expected_errors, rtol=0,atol=1e-12)
                error = row["eef_after"]-row["eef_before"]-row["action"][:3]*scale
                np.testing.assert_array_equal(row["nominal_prediction_error_m"],error)
                prediction_errors.append(float(np.linalg.norm(error)))
                error_decreases += int(expected_errors[1] <= expected_errors[0]+1e-12)
                steps, state, success = steps+1, row["sim_after"], bool(row["success"])
            require(len(physical) == min(sum(durations),520-steps0) or success, "Complete physical phases")
        require(len(physical) == branch["physical_steps"], "Physical count")
        live, prev, counts = copy.deepcopy(prefix), previous.copy(), streak.copy()
        previous_action = (physical[-1]["action"] if physical else last_native)[:6].astype(float)
        changed_chunks, changed_commands = 0, 0
        for index, row in enumerate(rows):
            q, tau = q0+index, threshold(q0+index,plan)
            route_check(row)
            require(int(row["query"]) == q and int(row["action_steps_before"]) == steps and not success, "Policy timeline")
            np.testing.assert_array_equal(row["sim_before"],state)
            np.testing.assert_array_equal(row["filter_previous"],previous_action)
            if family != "native":
                np.testing.assert_array_equal(row["noise"],expected_noise(main_id,repeat,index))
            require(int(row["candidate_count"]) == 1 and int(row["candidate_id"]) == 0 and text(row["operator"]) == operator, "Single actual policy query")
            raw, count = row["generated_actions"], int(row["executed_action_count"])
            actual = raw.copy()
            chunk = 2 if operator in TRANSFORM and index < 8 else 10
            require(int(row["requested_chunk"]) == chunk and 0 < count <= min(chunk,520-steps), "Chunk/horizon bound")
            for offset in range(count):
                command = raw[offset].copy()
                if index < 8:
                    if operator in ("damp2","boost2"):
                        command[:6] = np.clip(command[:6]*(.5 if operator == "damp2" else 2.), -1,1)
                    elif operator == "smooth2":
                        command[:6] = np.clip(.25*command[:6]+.75*previous_action,-1,1)
                actual[offset] = command
                previous_action = command[:6].astype(float).copy()
            np.testing.assert_array_equal(row["actions"],actual,err_msg="Actual transformed commands")
            np.testing.assert_array_equal(row["filter_after"],previous_action)
            np.testing.assert_array_equal(row["actions"][:,6],raw[:,6])
            changed = np.any(raw[:count] != actual[:count],axis=1)
            require(int(row["transformed_commands"]) == int(changed.sum()), "Changed command ledger")
            changed_chunks += int(changed.any())
            changed_commands += int(changed.sum())
            scores = components(live,row[PROBS_KEY])
            np.testing.assert_allclose(row["component_scores"],scores,rtol=2e-5,atol=2e-6)
            status = live.update(row[PROBS_KEY],row["proprio"][:3])
            require(live.v8.v7.query == q and bool(row["v82_alarm"]) == status["v82_alarm"], "Executed history only")
            np.testing.assert_array_equal(row["selector_thresholds"],tau)
            np.testing.assert_array_equal(row["raw_score_delta"],row["component_scores"]-prev)
            mask, current_mode = signature(row["component_scores"],tau)
            counts = np.where(mask,counts+1,0)
            np.testing.assert_array_equal(row["active_streaks"],counts)
            require(text(row["mode"]) == current_mode, "Current signal mode")
            prev = row["component_scores"].copy()
            success, steps, state = bool(row["success"]),steps+count,row["sim_after"]
            require(count == min(chunk,520-(steps-count)) or success, "No unexplained truncation")
            if family == "native":
                for key in main[q]:
                    if key not in ("inference_seconds","environment_seconds"):
                        np.testing.assert_array_equal(row[key],main[q][key],err_msg="Native suffix: "+key)
        require(success or steps == 520,"Complete suffix")
        require(success == branch["success"] and steps == branch["final_action_steps"] and
            steps-steps0 == branch["action_steps"],"Outcome/step ledger")
        require(len(rows) == branch["queries"] and len(rows)+1 == result["actual_model_queries"] == branch["actual_model_queries"],"Model query ledger")
        require(changed_chunks == branch["changed_chunks"] and changed_commands == branch["changed_commands"],"Action intervention ledger")
        if family == "native":
            require(result["all_native_suffixes_exact"] and success == original["success"] and len(rows) == len(main)-q0,"Exact original suffix")
        outcomes.append(dict(main_id=main_id,cohort_group=task["cohort_group"],base_task=task["base_task"],
            arm=arm,family=family,repeat=repeat,operator=operator,entry_mode=mode,original_success=bool(original["success"]),
            success=success,rescued=not original["success"] and success,harmed=bool(original["success"]) and not success,
            queries=len(rows),actual_model_queries=len(rows)+1,physical_steps=len(physical),action_steps=steps-steps0,
            changed_chunks=changed_chunks,changed_commands=changed_commands,target_error_nonincreasing_steps=error_decreases,
            mean_nominal_prediction_error_m=float(np.mean(prediction_errors)) if prediction_errors else None,
            suffix_manifest_sha256=digest(directory / "suffix/manifest.json")))
        total_queries += len(rows)+1
    for repeat in (0,1):
        baseline = traces["resample_r"+str(repeat)][2]
        for operator in TRANSFORM:
            first = traces[operator+"_r"+str(repeat)][2][0]
            for key in (*ALL_FIELDS,"noise","input_sha256","sim_before","generated_actions"):
                np.testing.assert_array_equal(first[key],baseline[0][key],err_msg="Common initial raw proposal: "+key)
        for family in ("moe_switch","random_switch"):
            operator, physical, rows = traces[family+"_r"+str(repeat)]
            _, fixed_physical, fixed_rows = traces[operator+"_r"+str(repeat)]
            equal_records(physical,fixed_physical,"Supervisor physical dispatch")
            equal_records(rows,fixed_rows,"Supervisor complete policy dispatch")
    return dict(main_id=main_id,cohort_group=task["cohort_group"],c0_queries=len(c0),actual_model_queries=total_queries,branches=outcomes)


def audit_fresh_mains(plan):
    run, count = Path(plan["fresh_run"]), 0
    for task in (t for t in plan["cohort"] if t["cohort_group"] == "fresh"):
        main = list(records(Path(task["parent_directory"]) / "main"))
        c0 = list(records(run / "replays" / task["main_id"] / "c0"))
        equal_records(main,c0,"Independent fresh full C0")
        for row in main:
            route_check(row)
        count += len(main)
    return dict(mains=50,main_queries=count,c0_queries=count,all_exact=True)


def audit(args):
    plan, summary = (json.loads((args.run / name).read_text()) for name in ("plan.json","summary.json"))
    require(plan["protocol"] == PROTOCOL and plan["settings"] == SETTINGS and plan["arms"] == list(ARMS),"Plan identity")
    require(summary["status"] == "completed" and all(j["status"] == "completed" for j in summary["tasks"]),"Complete scheduled experiment")
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected,"Frozen source changed: "+path)
    fresh = audit_fresh_mains(plan)
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_task,args.run,task,plan) for task in plan["tasks"]]
        for future in futures:
            tasks.append(future.result())
            print(json.dumps(dict(audited=len(tasks),total=len(futures))),flush=True)
    rows = [row for task in tasks for row in task["branches"]]
    arms, modes = [], []
    for group in plan["cohorts"]:
        cohort = [t for t in plan["cohort"] if t["cohort_group"] == group]
        retained = sum(t["native_success"] for t in cohort if not t["events"])
        for arm in ARMS:
            selected = [r for r in rows if r["cohort_group"] == group and r["arm"] == arm]
            arms.append(dict(cohort_group=group,arm=arm,alarmed_parents=len(selected),rescued=sum(r["rescued"] for r in selected),
                harmed=sum(r["harmed"] for r in selected),full_cohort_success=retained+sum(r["success"] for r in selected),
                cohort_size=len(cohort),actual_branch_model_queries=sum(r["actual_model_queries"] for r in selected)))
        for family in FAMILIES:
            for mode in sorted({r["entry_mode"] for r in rows if r["cohort_group"] == group}):
                selected = [r for r in rows if r["cohort_group"] == group and r["family"] == family and r["entry_mode"] == mode]
                modes.append(dict(cohort_group=group,family=family,entry_mode=mode,parents=len({r["main_id"] for r in selected}),
                    repeated_branches=len(selected),rescued=sum(r["rescued"] for r in selected),harmed=sum(r["harmed"] for r in selected)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="passed",run=str(args.run.resolve()),protocol=PROTOCOL,plan_sha256=digest(args.run / "plan.json"),
        auditor_sha256=digest(__file__),parents=len(tasks),suffixes=len(rows),fresh_mains=fresh,
        actual_model_queries=sum(t["actual_model_queries"] for t in tasks),
        original_failed_parents_rescued_anywhere=len({r["main_id"] for r in rows if r["rescued"]}),
        arms=arms,mode_responses=modes,tasks=tasks,rows_sha256=digest(csv_path),
        caveat="Any-controller rescue union is descriptive, not the success rate of a deployable policy")
    require(result["actual_model_queries"] == summary["actual_model_queries"],"Total forward count")
    atomic_json(args.output,result)
    print(json.dumps({key:result[key] for key in ("status","parents","suffixes","actual_model_queries","original_failed_parents_rescued_anywhere")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    audit(parser.parse_args())
