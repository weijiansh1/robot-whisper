#!/usr/bin/env python3
"""Independently audit causal mode scores, paired pools, and executed suffixes."""

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
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import atomic_json, digest, load_snapshot, records
from fixed_recovery_control import MODULE, recovery_action, targets
from mode_control import ARMS, FAMILIES, PROTOCOL, SETTINGS
from v8_closed_loop import score_status
from v82_closed_loop import V82TriggerMonitor


def text(value):
    return value.item().decode()


def components(live, probability):
    trial = copy.deepcopy(live.v8)
    status = trial.update(probability)
    raw = np.maximum(np.asarray(probability, np.float32), 0)
    raw /= raw.sum(-1, keepdims=True)
    root = np.sqrt(raw[:, :, 1:])
    speeds = (np.sqrt(((root[:, 1:]-root[:, :-1])**2).sum(-1))/np.sqrt(2.)).mean(-1).astype(np.float32)
    lengths = speeds.sum(-1)
    inversion = -np.log(max(lengths[:4].mean(), 1e-12)/max(lengths[4:].mean(), 1e-12))
    curvature = np.abs(speeds[4:, 8]-2*speeds[4:, 4]+speeds[4:, 0]).mean()
    independent_raw = np.array([inversion, curvature], np.float32)
    np.testing.assert_allclose(independent_raw, status["v8_raw"], rtol=2e-5, atol=1e-7)
    history = np.asarray(live.v8.raw+[independent_raw], np.float32)
    if trial.v7.query >= 4:
        relative = np.log(np.maximum(history[:, 1], 1e-12)/max(history[1:5, 1].mean(dtype=np.float32), 1e-12))
        np.testing.assert_allclose(status["v8_scores"], [history[-6:, 0].mean(dtype=np.float64),
            relative[-6:].mean(dtype=np.float64)], rtol=2e-5, atol=2e-6)
    return score_status(status)


def threshold(q, plan):
    values = np.asarray(plan["thresholds"], float).copy()
    values[3:] -= .0015*q
    return values


def signature(values, tau):
    active = np.isfinite(values) & np.array([values[i] > tau[i] if i < 3 else values[i] >= tau[i] for i in range(5)])
    name = "+".join(k for k, v in zip("FAPIC", active) if v) or "none"
    return active, name if np.isfinite(values).all() else "unavailable"


def generator(main_id, repeat, index, candidate, kind):
    key = json.dumps([PROTOCOL, main_id, repeat, index, candidate, kind], separators=(",", ":"))
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16], "little"))


def expected_noise(main_id, repeat, index, candidate=0):
    return generator(main_id, repeat, index, candidate, "flow").standard_normal((10, 24)).astype(np.float32)


def costs(scores, tau, margins, active):
    if not np.isfinite(scores).all():
        return float("inf"), float("inf")
    r = (scores-tau)/margins
    return float(max(r[0], min(r[1], r[2]), r[3], r[4])), float(sum(max(r[i]+int(active[i]), 0.)**2 for i in range(5)))


def identity(result, task, original):
    require(result["status"] == "completed" and not result["invalid_pair"], "Incomplete result")
    require(result["protocol"] == PROTOCOL and result["contract"] == SETTINGS, "Frozen runtime identity")
    require(result["gpu"] in (0, 1, 2, 3) and result["render_gpu"] in (0, 3), "Device exclusion")
    require(not result["hidden_capture"] and not result["main_intervention"] and result["reused_main"], "Main/hook contract")
    require(result["main_id"] == task["main_id"] and result["seed"] == original["seed"] and
        result["init_index"] == original["init_index"], "Parent identity")
    for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
        require(result["model_metadata"][key] == original["model_metadata"][key], "Model identity: "+key)


def audit_task(run, task, plan):
    main_id, parent = task["main_id"], Path(task["parent_directory"])
    original = json.loads((parent / "result.json").read_text())
    main = list(records(parent / "main"))
    replay_dir = run / "replays" / main_id
    replay = json.loads((replay_dir / "result.json").read_text())
    identity(replay, task, original)
    c0 = list(records(replay_dir / "c0"))
    require(len(c0) == len(main) == replay["c0"]["compared_queries"], "Complete C0")
    require(replay["c0"]["status"] == "passed" and replay["online_triggers_exact"] and replay["v82_triggers_exact"], "C0/trigger status")
    live = V82TriggerMonitor()
    for expected, actual in zip(main, c0):
        for key in expected:
            if key not in ("inference_seconds", "environment_seconds"):
                np.testing.assert_array_equal(actual[key], expected[key], err_msg="C0: "+key)
        route_check(actual)
        live.update(actual[PROBS_KEY], actual["proprio"][:3])
    require(live.first == task["first_alarms"] == replay["replay_first_alarms"], "Frozen first alarms")
    event = replay["events"][0]
    for key, value in task["events"][0].items():
        require(event[key] == value, "Event plan changed")
    q0 = event["start_query"]
    require(q0 == live.first["v82_frozen"]+1 < len(main), "Causal next-query fork")
    saved = load_snapshot(replay_dir / "events" / event["event_id"] / "snapshot")
    require(event["snapshot_manifest_sha256"] == digest(replay_dir / "events" / event["event_id"] / "snapshot/manifest.json"), "Snapshot hash")
    np.testing.assert_array_equal(saved["physics"]["sim"], main[q0]["sim_before"])
    tape = list(records(replay_dir / "environment_rng"))
    require(len(tape) == original["action_steps"] and digest(replay_dir / "environment_rng/manifest.json") == replay["rng_tape_sha256"], "Environment RNG tape")
    prefix, previous, streaks = V82TriggerMonitor(), np.full(5, np.nan), np.zeros(5, int)
    for q, row in enumerate(main[:q0]):
        previous = score_status(prefix.update(row[PROBS_KEY], row["proprio"][:3]))
        active, alarm_mode = signature(previous, threshold(q, plan))
        streaks = np.where(active, streaks+1, 0)
    margins, first_pools, first_rows, entry_reference = np.asarray(plan["margins"]), {}, {}, None
    outcomes, audited_pools, audited_candidates, audited_queries = [], 0, 0, len(c0)
    for arm in ARMS:
        family, repeat = ("native", -1) if arm == "native" else (arm[:-3], int(arm[-1]))
        directory = run / "events" / event["event_id"] / arm
        result = json.loads((directory / "result.json").read_text())
        branch = json.loads((directory / "branch.json").read_text())
        identity(result, task, original)
        require(result["branches"] == [branch] and branch["status"] == "completed", "Branch commit")
        require(branch["starts_after_main_complete"] and branch["starts_after_c0"], "Replay ordering")
        require(result["c0"]["replay_result_sha256"] == digest(replay_dir / "result.json") and
            result["rng_tape_sha256"] == replay["rng_tape_sha256"], "C0 linkage")
        probes = list(records(directory / "entry_probe"))
        require(len(probes) == 1 and branch["entry_probe_manifest_sha256"] == digest(directory / "entry_probe/manifest.json"), "Entry probe")
        probe = probes[0]
        route_check(probe)
        for key in (*ALL_FIELDS, "actions", "noise", "proprio", "input_sha256", "input_component_sha256", "sim_before"):
            np.testing.assert_array_equal(probe[key], main[q0][key], err_msg="Entry probe: "+key)
            if entry_reference is not None:
                np.testing.assert_array_equal(probe[key], entry_reference[key])
        entry_reference = probe
        np.testing.assert_allclose(probe["component_scores"], components(prefix, probe[PROBS_KEY]), rtol=2e-5, atol=2e-6)
        _, entry_mode = signature(probe["component_scores"], threshold(q0, plan))
        require(text(probe["mode"]) == entry_mode == branch["entry_mode"], "Pre-treatment mode")
        np.testing.assert_array_equal(probe["sim_before"], probe["sim_after"])
        require(int(probe["executed_action_count"]) == 0, "Probe must not execute")
        live, prev, counts = copy.deepcopy(prefix), previous.copy(), streaks.copy()
        state, steps, success = main[q0]["sim_before"].copy(), event["action_steps_before"], False
        physical = list(records(directory / "physical")) if family in ("hold", "withdraw") else []
        target = targets(physical[0]["eef_before"], main[max(0, q0-3)]["proprio"][:3]) if physical else None
        for index, row in enumerate(physical):
            np.testing.assert_array_equal(row["sim_before"], state)
            np.testing.assert_allclose(row["target"], target[index//8], rtol=0, atol=1e-12)
            action = recovery_action(family, row["eef_before"], target[index//8], branch["last_native_gripper"], branch["controller_translation_scale"])
            np.testing.assert_array_equal(row["action"], action)
            require(int(row["action_step"]) == steps and not success, "Physical action order")
            steps, state, success = steps+1, row["sim_after"], bool(row["success"])
        require(len(physical) == branch["physical_steps"], "Physical action count")
        if physical:
            require(len(physical) == min(16, 520-event["action_steps_before"]) or success, "Truncated physical module")
        rows, model_queries, pool_count, changed = list(records(directory / "suffix")), 1, 0, 0
        first_rows[arm] = rows[0] if rows else None
        for index, row in enumerate(rows):
            q, tau = q0+index, threshold(q0+index, plan)
            route_check(row)
            require(int(row["query"]) == q and int(row["action_steps_before"]) == steps and not success, "Suffix order")
            np.testing.assert_array_equal(row["sim_before"], state)
            np.testing.assert_array_equal(row["selector_thresholds"], tau)
            if int(row["candidate_count"]) == 16:
                require(family in ("random", "scalar", "mode") and index < 5, "Candidate query budget")
                path = directory / "pools" / str(index)
                selection = json.loads((path / "selection.json").read_text())
                require(text(row["selection_sha256"]) == digest(path / "selection.json") and
                    selection["pool_manifest_sha256"] == digest(path / "candidates/manifest.json"), "Candidate ledger hash")
                pool = list(records(path / "candidates"))
                require(len(pool) == 16, "Unfiltered candidate population")
                active, default_mode = signature(pool[0]["component_scores"], tau)
                scalar, vector = [], []
                for candidate, item in enumerate(pool):
                    route_check(item)
                    np.testing.assert_array_equal(item["noise"], expected_noise(main_id, repeat, index, candidate))
                    for key in ("input_sha256", "input_component_sha256", "proprio", "sim_before"):
                        np.testing.assert_array_equal(item[key], row[key])
                    np.testing.assert_array_equal(item["sim_after"], item["sim_before"])
                    require(int(item["executed_action_count"]) == 0 and int(item["source_query"]) == q, "Counterfactual-only pool")
                    values = components(live, item[PROBS_KEY])
                    np.testing.assert_allclose(item["component_scores"], values, rtol=2e-5, atol=2e-6)
                    sc, vc = costs(item["component_scores"], tau, margins, active)
                    scalar.append(sc)
                    vector.append(vc)
                    np.testing.assert_allclose([item["scalar_cost"], item["vector_cost"]], [sc, vc], rtol=0, atol=1e-10)
                if family == "random":
                    picked = int(generator(main_id, repeat, index, 0, "choice").integers(16))
                elif family == "mode" and (not active.any() or not np.isfinite(pool[0]["component_scores"]).all()):
                    picked = 0
                else:
                    picked = min(range(16), key=lambda i: ((scalar if family == "scalar" else vector)[i], i))
                require(picked == int(row["candidate_id"]) == selection["candidate"], "Independent selection")
                for key in (*ALL_FIELDS, "actions", "noise", "input_sha256", "component_scores"):
                    np.testing.assert_array_equal(row[key], pool[picked][key], err_msg="Actual dispatch: "+key)
                np.testing.assert_array_equal(row["default_component_scores"], pool[0]["component_scores"])
                count = int(row["executed_action_count"])
                changed += int(not np.array_equal(row["actions"][:count], pool[0]["actions"][:count]))
                if index == 0:
                    first_pools[arm] = pool
                pool_count += 1
                audited_candidates += len(pool)
            else:
                require(int(row["candidate_count"]) == 1 and int(row["candidate_id"]) == 0, "Ordinary query accounting")
                require(family not in ("random", "scalar", "mode") or index >= 5, "Missing paired pool")
                if family != "native":
                    np.testing.assert_array_equal(row["noise"], expected_noise(main_id, repeat, index))
            expected = components(live, row[PROBS_KEY])
            np.testing.assert_allclose(row["component_scores"], expected, rtol=2e-5, atol=2e-6)
            status = live.update(row[PROBS_KEY], row["proprio"][:3])
            require(live.v8.v7.query == q and bool(row["v82_alarm"]) == status["v82_alarm"], "Executed history only")
            active, name = signature(row["component_scores"], tau)
            counts = np.where(active, counts+1, 0)
            np.testing.assert_array_equal(row["active_streaks"], counts)
            np.testing.assert_array_equal(row["active_components"], active)
            require(text(row["mode"]) == name, "Mode identity")
            np.testing.assert_array_equal(row["raw_score_delta"], row["component_scores"]-prev)
            np.testing.assert_array_equal(row["threshold_delta"], tau-threshold(q-1, plan))
            np.testing.assert_allclose(row["normalized_delta"], (row["component_scores"]-tau)/margins-
                (prev-threshold(q-1, plan))/margins, rtol=0, atol=1e-12)
            prev = row["component_scores"].copy()
            expected_chunk = 2 if family == "short" and index < 5 else 10
            count = int(row["executed_action_count"])
            require(int(row["requested_chunk"]) == expected_chunk and 0 < count <= min(expected_chunk, 520-steps), "Chunk/horizon budget")
            success, steps, state = bool(row["success"]), steps+count, row["sim_after"]
            require(count == min(expected_chunk, 520-(steps-count)) or success, "Premature chunk truncation")
            model_queries += int(row["candidate_count"])
            if family == "native":
                for key in main[q]:
                    if key not in ("inference_seconds", "environment_seconds"):
                        np.testing.assert_array_equal(row[key], main[q][key], err_msg="Native exact: "+key)
        require(success or steps == 520, "Incomplete suffix horizon")
        require(success == branch["success"] and steps == branch["final_action_steps"] and
            model_queries == result["actual_model_queries"] == branch["actual_model_queries"], "Outcome/cost ledger")
        require(len(rows) == branch["queries"] and pool_count == branch["candidate_pools"] and changed == branch["changed_chunks"], "Branch counters")
        if family == "native":
            require(result["all_native_suffixes_exact"] and success == original["success"] and len(rows) == len(main)-q0, "Native reproduction")
        outcomes.append(dict(main_id=main_id, base_task=task["base_task"], arm=arm, family=family, repeat=repeat,
            alarm_mode=alarm_mode, entry_mode=entry_mode, original_success=bool(original["success"]), success=success,
            rescued=not original["success"] and success, harmed=bool(original["success"]) and not success,
            queries=len(rows), actual_model_queries=model_queries, physical_steps=len(physical),
            action_steps=steps-event["action_steps_before"], candidate_pools=pool_count, changed_chunks=changed,
            final_mode=text(rows[-1]["mode"]) if rows else None,
            low_risk_queries=sum(float(row["selected_risk"]) < 0 for row in rows),
            below_margin_queries=sum(float(row["selected_risk"]) <= -1 for row in rows),
            suffix_manifest_sha256=digest(directory / "suffix/manifest.json")))
        audited_queries += model_queries
        audited_pools += pool_count
    for repeat in range(2):
        random_pool = first_pools["random_r"+str(repeat)]
        for family in ("scalar", "mode"):
            pool = first_pools[family+"_r"+str(repeat)]
            for lhs, rhs in zip(random_pool, pool):
                for key in (*ALL_FIELDS, "actions", "noise", "input_sha256", "component_scores"):
                    np.testing.assert_array_equal(lhs[key], rhs[key], err_msg="Common first candidate pool: "+key)
        resample = first_rows["resample_r"+str(repeat)]
        short = first_rows["short_r"+str(repeat)]
        for key in (*ALL_FIELDS, "actions", "noise", "input_sha256", "sim_before"):
            np.testing.assert_array_equal(resample[key], short[key], err_msg="Short/resample pairing: "+key)
            np.testing.assert_array_equal(resample[key], random_pool[0][key], err_msg="Resample/candidate-zero pairing: "+key)
    return dict(main_id=main_id, c0_queries=len(c0), actual_model_queries=audited_queries,
        candidate_pools=audited_pools, candidates=audited_candidates, branches=outcomes)


def audit(args):
    plan = json.loads((args.run / "plan.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    require(plan["protocol"] == PROTOCOL and plan["settings"] == SETTINGS and plan["arms"] == list(ARMS), "Plan identity")
    require(summary["status"] == "completed" and all(t["status"] == "completed" for t in summary["tasks"]), "All scheduled jobs complete")
    for path, expected in plan["source_sha256"].items():
        require(digest(path) == expected, "Frozen source changed: "+path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_task, args.run, task, plan) for task in plan["tasks"]]
        tasks = []
        for future in futures:
            task = future.result()
            tasks.append(task)
            print(json.dumps(dict(audited_parents=len(tasks), main_id=task["main_id"])), flush=True)
    rows = [row for task in tasks for row in task["branches"]]
    untriggered = [task for task in plan["cohort"] if not task["events"]]
    retained_success = sum(task["native_success"] for task in untriggered)
    arms = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        arms.append(dict(arm=arm, parents=len(selected), success=sum(r["success"] for r in selected),
            rescued=sum(r["rescued"] for r in selected), harmed=sum(r["harmed"] for r in selected),
            full_cohort_success=retained_success+sum(r["success"] for r in selected), full_cohort_size=len(plan["cohort"]),
            actual_branch_model_queries=sum(r["actual_model_queries"] for r in selected),
            changed_chunks=sum(r["changed_chunks"] for r in selected)))
    pairs = []
    lookup = {(r["main_id"], r["arm"]): r for r in rows}
    for family in FAMILIES:
        for control in ("resample", "random"):
            if family == control:
                continue
            for mode in ["all"]+sorted({r["entry_mode"] for r in rows}):
                selected = [r for r in rows if r["family"] == family and (mode == "all" or r["entry_mode"] == mode)]
                wins = losses = 0
                for row in selected:
                    baseline = lookup[row["main_id"], control+"_r"+str(row["repeat"])]
                    wins += int(row["success"] and not baseline["success"])
                    losses += int(baseline["success"] and not row["success"])
                pairs.append(dict(family=family, control=control, entry_mode=mode, pairs=len(selected),
                    parents=len({r["main_id"] for r in selected}), wins=wins, losses=losses))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status="passed", run=str(args.run.resolve()), protocol=PROTOCOL,
        plan_sha256=digest(args.run / "plan.json"), auditor_sha256=digest(__file__), parents=len(tasks),
        suffixes=len(rows), c0_queries=sum(t["c0_queries"] for t in tasks),
        actual_model_queries=sum(t["actual_model_queries"] for t in tasks),
        candidate_pools=sum(t["candidate_pools"] for t in tasks), candidates=sum(t["candidates"] for t in tasks),
        untriggered_parents=len(untriggered), retained_successes=retained_success,
        entry_mode_counts=dict(Counter(r["entry_mode"] for r in rows if r["arm"] == "native")),
        arms=arms, paired_comparisons=pairs, rows_sha256=digest(csv_path), tasks=tasks,
        interpretation="Repeated branches are clustered by original parent; exploratory mode groups are not validated physical failure classes",
        physical_verification="Separate verify_mode_dispatched_actions.py must re-execute commands in MuJoCo")
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "parents", "suffixes", "actual_model_queries", "arms")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    audit(parser.parse_args())
