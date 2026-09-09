#!/usr/bin/env python3
"""Run paired mode interventions after complete exact native Long replay."""

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from collect_native_long_v8 import NativeLongSession
from collect_native_long_v82 import NativeV82Session
from collect_preflight_worker import restore
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from fixed_recovery_control import MODULE, recovery_action, stalled, targets
from mode_control import (ARMS, PROTOCOL, SETTINGS, ModeTrace, active_mask, decode_arm,
    mode_name, noise_for, normalized, scalar_cost, select, thresholds_at)
from v8_closed_loop import SETTINGS as V8_SETTINGS, limits, score_status
from v82_closed_loop import V82TriggerMonitor


class NativeModesSession(NativeV82Session):
    def __init__(self, args, cache):
        if args.control["contract"] != SETTINGS or args.control["kind"] not in ("fixed_replay", "fixed_branches"):
            raise ValueError("Frozen mode experiment required")
        compatible = copy.copy(args)
        compatible.control = dict(args.control, contract=V8_SETTINGS)
        NativeLongSession.__init__(self, compatible, cache)
        self.args = args
        self.base_thresholds, self.margins = limits()
        self.report.update(protocol=PROTOCOL, contract=SETTINGS, trigger_method="v82_frozen")
        self.save()

    def entry_probe(self, obs, live, trace, main, q, steps):
        noise = copy.deepcopy(self.rng).standard_normal((10, 24)).astype(np.float32)
        request, response, inference = self.query(obs, noise)
        alarm = copy.deepcopy(live).update(response[PROBS_KEY], request["observation/state"][:3])
        state = np.asarray(self.env.get_sim_state(), np.float64).copy()
        row = self.record(request, response, q, steps, 0, state, state, False, alarm, inference, 0.)
        for key in (*ALL_FIELDS, "noise", "actions", "proprio", "input_sha256", "input_component_sha256", "sim_before"):
            np.testing.assert_array_equal(row[key], main[q][key], err_msg="Pre-intervention probe: "+key)
        tau = thresholds_at(q, self.base_thresholds)
        row.update(copy.deepcopy(trace).commit(score_status(alarm), tau, self.margins))
        writer = ShardWriter(self.args.output / "entry_probe")
        try:
            writer.append(row)
        finally:
            writer.close()
        return row

    def candidate_pool(self, obs, live, default, family, repeat, index, q, steps):
        items, alarms = [default], []
        for candidate in range(1, SETTINGS["candidates"]):
            items.append(self.query(obs, noise_for(self.args.main_id, repeat, index, candidate)))
        for item in items:
            alarms.append(copy.deepcopy(live.v8).update(item[1][PROBS_KEY]))
        scores = np.stack([score_status(alarm) for alarm in alarms])
        tau = thresholds_at(q, self.base_thresholds)
        picked, scalar, vector = select(family, scores, tau, self.margins, self.args.main_id, repeat, index)
        location = self.args.output / "pools" / str(index)
        state = np.asarray(self.env.get_sim_state(), np.float64).copy()
        writer = ShardWriter(location / "candidates")
        try:
            for candidate, (item, alarm) in enumerate(zip(items, alarms)):
                request, response, inference = item
                row = self.record(request, response, candidate, steps, 0, state, state, False, alarm, inference, 0.)
                row.update(source_query=np.int32(q), relative_query=np.int32(index),
                    component_scores=scores[candidate], selector_thresholds=tau.copy(),
                    normalized_scores=normalized(scores[candidate], tau, self.margins),
                    scalar_cost=np.float64(scalar[candidate]), vector_cost=np.float64(vector[candidate]),
                    default_active_components=active_mask(scores[0], tau),
                    mode=np.asarray(mode_name(scores[candidate], tau), dtype="S32"))
                writer.append(row)
        finally:
            writer.close()
        selection = dict(family=family, repeat=repeat, relative_query=index, source_query=q, candidate=picked,
            candidate_count=len(items), default_mode=mode_name(scores[0], tau),
            selected_mode=mode_name(scores[picked], tau), thresholds=tau.tolist(), margins=self.margins.tolist(),
            abstained=bool(family == "mode" and (not active_mask(scores[0], tau).any() or not np.isfinite(scores[0]).all())),
            pool_manifest_sha256=digest(location / "candidates/manifest.json"))
        atomic_json(location / "selection.json", selection)
        return items[picked], scores[0], picked, sum(item[2] for item in items), digest(location / "selection.json")

    def physical(self, obs, main, q0, steps, family, branch):
        controller = self.env.env.robots[0].controller
        scale = np.asarray(controller.output_max, float)[:3]
        for value, expected in ((controller.input_max, 1), (controller.input_min, -1)):
            np.testing.assert_allclose(value, expected, atol=0, rtol=0)
        np.testing.assert_allclose(controller.output_min, -np.asarray(controller.output_max), atol=0, rtol=0)
        if np.min(scale) < MODULE["max_translation_step_m"]:
            raise ValueError("Controller cannot represent recovery translation")
        target = targets(obs["robot0_eef_pos"], main[max(0, q0-MODULE["target_history_queries"])]["proprio"][:3])
        gripper = float(main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1, 6])
        branch.update(recovery_targets=target.tolist(), controller_translation_scale=scale.tolist(), last_native_gripper=gripper)
        writer, success = ShardWriter(self.args.output / "physical"), False
        try:
            for index in range(min(SETTINGS["physical_steps"], SETTINGS["horizon"]-steps)):
                phase = index//MODULE["phase_steps"]
                position = np.asarray(obs["robot0_eef_pos"], float).copy()
                action = recovery_action(family, position, target[phase], gripper, scale)
                before, tick = np.asarray(self.env.get_sim_state(), float).copy(), time.monotonic()
                obs, success = self.step(action, steps)
                duration = time.monotonic()-tick
                writer.append(dict(query=np.int32(index), action_step=np.int32(steps), phase=np.int16(phase),
                    action=action, target=target[phase].copy(), eef_before=position,
                    eef_after=np.asarray(obs["robot0_eef_pos"], float).copy(), sim_before=before,
                    sim_after=np.asarray(self.env.get_sim_state(), float).copy(), success=np.bool_(success),
                    environment_seconds=np.float64(duration)))
                steps += 1
                branch["physical_steps"] += 1
                branch["environment_seconds"] += duration
                if success:
                    break
        finally:
            branch["physical_shards"] = writer.close()
        return obs, steps, success

    def branches(self):
        arm = self.args.control["arm"]
        family, repeat = decode_arm(arm)
        replay_dir = Path(self.args.control["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if (replay["status"] != "completed" or replay["c0"]["status"] != "passed" or
                not replay["online_triggers_exact"] or not replay["v82_triggers_exact"]):
            raise ValueError("Complete exact C0 and causal v8.2 trigger required")
        event = replay["events"][0]
        if any(event[key] != value for key, value in self.args.control["event"].items()):
            raise ValueError("Scheduled fork differs")
        location = replay_dir / "events" / event["event_id"] / "snapshot"
        if (digest(location / "manifest.json") != event["snapshot_manifest_sha256"] or
                digest(replay_dir / "environment_rng/manifest.json") != replay["rng_tape_sha256"]):
            raise ValueError("Fork snapshot or RNG tape changed")
        self.tape = list(records(replay_dir / "environment_rng"))
        saved, main = load_snapshot(location), list(records(self.parent / "main"))
        q0, initial_steps = event["start_query"], event["action_steps_before"]
        if saved["query"] != q0 or saved["action_steps"] != initial_steps:
            raise ValueError("Wrong fork position")
        self.report.update(events=[event], rng_tape_sha256=replay["rng_tape_sha256"],
            c0=dict(status="passed", replay_directory=str(replay_dir),
                replay_result_sha256=digest(replay_dir / "result.json"),
                replay_branch_sha256=digest(replay_dir / "c0/branch.json")))
        live, trace = V82TriggerMonitor(), ModeTrace()
        for q, row in enumerate(main[:q0]):
            status = live.update(row[PROBS_KEY], row["proprio"][:3])
            trace.commit(score_status(status), thresholds_at(q, self.base_thresholds), self.margins)
        if live.first["v82_frozen"] != q0-1:
            raise ValueError("Noncausal intervention time")
        self.new_env()
        obs = restore(self.env, saved, self.rng)
        probe = self.entry_probe(obs, live, trace, main, q0, initial_steps)
        branch = dict(status="running", arm=arm, family=family, repeat=repeat, event_id=event["event_id"],
            start_query=q0, entry_mode=bytes(probe["mode"]).decode(), entry_probe_exact=True,
            entry_probe_manifest_sha256=digest(self.args.output / "entry_probe/manifest.json"),
            starts_after_main_complete=True, starts_after_c0=True, queries=0, actual_model_queries=1,
            entry_probe_queries=1, physical_steps=0, candidate_pools=0, changed_chunks=0,
            initial_action_steps=initial_steps, action_steps=0, success=False,
            inference_seconds=float(probe["inference_seconds"]), environment_seconds=0.)
        self.report["branches"] = [branch]
        writer = ShardWriter(self.args.output / "suffix")
        steps, success, positions = initial_steps, False, []
        started = time.monotonic()
        try:
            if family in ("hold", "withdraw"):
                obs, steps, success = self.physical(obs, main, q0, steps, family, branch)
            while not success and steps < SETTINGS["horizon"]:
                index, q = branch["queries"], q0+branch["queries"]
                noise = (self.rng.standard_normal((10, 24)).astype(np.float32) if family == "native"
                         else noise_for(self.args.main_id, repeat, index))
                item = self.query(obs, noise)
                default_actions = np.asarray(item[1]["actions"]).copy()
                default_scores = score_status(copy.deepcopy(live.v8).update(item[1][PROBS_KEY]))
                picked, forwards, inference, selection_hash = 0, 1, item[2], ""
                if family in ("random", "scalar", "mode") and index < SETTINGS["search_queries"]:
                    item, default_scores, picked, inference, selection_hash = self.candidate_pool(
                        obs, live, item, family, repeat, index, q, steps)
                    forwards = SETTINGS["candidates"]
                    branch["candidate_pools"] += 1
                request, response, _ = item
                alarm = live.update(response[PROBS_KEY], request["observation/state"][:3])
                if live.v8.v7.query != q:
                    raise ValueError("Unexecuted probes advanced history")
                tau = thresholds_at(q, self.base_thresholds)
                features = trace.commit(score_status(alarm), tau, self.margins)
                positions.append(np.asarray(request["observation/state"][:3]).copy())
                before = np.asarray(self.env.get_sim_state(), float).copy()
                chunk = SETTINGS["short_chunk"] if family == "short" and index < SETTINGS["short_queries"] else SETTINGS["chunk"]
                obs, count, success, duration = self.advance(obs, response["actions"], min(chunk, 520-steps), steps)
                after = np.asarray(self.env.get_sim_state(), float).copy()
                row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, duration)
                row.update(features)
                changed = not np.array_equal(response["actions"][:count], default_actions[:count])
                row.update(relative_query=np.int32(index), candidate_id=np.int16(picked), candidate_count=np.int16(forwards),
                    selection_sha256=np.asarray(selection_hash, dtype="S64"), default_component_scores=default_scores,
                    default_mode=np.asarray(mode_name(default_scores, tau), dtype="S32"),
                    selected_risk=np.float64(scalar_cost(features["normalized_scores"])),
                    default_risk=np.float64(scalar_cost(normalized(default_scores, tau, self.margins))),
                    action_changed_from_default=np.bool_(changed), requested_chunk=np.int16(chunk),
                    v82_alarm=np.bool_(alarm["v82_alarm"]), v82_first=np.int32(alarm["v82_first"]),
                    knn_score=np.float32(alarm["knn_score"]), cosine_score=np.float32(alarm["cosine_score"]),
                    v8_scores=alarm["v8_scores"], behavior_stall=np.bool_(alarm["behavior_stall"]),
                    post_intervention_stall_valid=np.bool_(len(positions) >= 6), post_intervention_stall=np.bool_(stalled(positions)))
                if family == "native":
                    if q >= len(main):
                        raise ValueError("Native exceeded original")
                    for key in main[q]:
                        if key not in ("inference_seconds", "environment_seconds"):
                            np.testing.assert_array_equal(row[key], main[q][key], err_msg="Native exact: "+key)
                writer.append(row)
                steps += count
                branch.update(queries=index+1, actual_model_queries=branch["actual_model_queries"]+forwards,
                    changed_chunks=branch["changed_chunks"]+int(changed),
                    inference_seconds=branch["inference_seconds"]+inference,
                    environment_seconds=branch["environment_seconds"]+duration)
                if branch["queries"] % 4 == 0:
                    self.save()
            if family == "native":
                if branch["queries"] != len(main)-q0 or success != self.original["success"]:
                    raise ValueError("Native suffix incomplete")
                self.report["all_native_suffixes_exact"] = True
            if branch["actual_model_queries"] != self.report["actual_model_queries"]:
                raise ValueError("Unaccounted model calls")
            branch.update(status="completed", success=success, action_steps=steps-initial_steps,
                final_action_steps=steps, elapsed_seconds=time.monotonic()-started)
        finally:
            branch["suffix_shards"] = writer.close()
        atomic_json(self.args.output / "branch.json", branch)
        self.close_env()
        self.save()


def run(args, cache):
    session, started = NativeModesSession(args, cache), time.monotonic()
    try:
        session.replay() if args.control["kind"] == "fixed_replay" else session.branches()
        session.report["status"] = "completed"
    except BaseException:
        session.report.update(status="failed", invalid_pair=True, error=traceback.format_exc())
        raise
    finally:
        session.close_env()
        session.report["elapsed_seconds"] = time.monotonic()-started
        session.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("native_long",), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--render-gpu", type=int, choices=(0, 3), required=True)
    parser.add_argument("--port", type=int, required=True)
    args, cache = parser.parse_args(), {}
    args.model, args.exact_noise_fastpath = "long", False
    try:
        for line in sys.stdin:
            message, task = json.loads(line), copy.copy(args)
            for key in ("variant", "seed", "main_id", "init_index", "control"):
                setattr(task, key, message[key])
            task.output = Path(message["output"])
            cache["jobs"] = cache.get("jobs", 0)+1
            result = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                result["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"] = 1
            print("COLLECTION_RESULT "+json.dumps(result), flush=True)
    finally:
        if cache.get("client"):
            cache["client"].close()


if __name__ == "__main__":
    main()
