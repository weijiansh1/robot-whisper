#!/usr/bin/env python3
"""Execute a frozen bank of causal recovery controllers after exact native replay."""

import copy
import json
from pathlib import Path
import time

import numpy as np

from collect_native_long_modes import NativeModesSession
from collect_native_long_v8 import NativeLongSession
from collect_preflight_worker import restore
from collection_routes import PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from control_bank import (PROTOCOL, SETTINGS, PHYSICAL, TRANSFORM, ActionFilter, choose_operator,
    decode_arm, noise_for, physical_action, thresholds_at, waypoints)
from mode_control import ModeTrace, scalar_cost
from v8_closed_loop import SETTINGS as V8_SETTINGS, limits, score_status
from v82_closed_loop import V82TriggerMonitor


class ControlBankSession(NativeModesSession):
    def __init__(self, args, cache):
        if args.control["contract"] != SETTINGS or args.control["kind"] not in ("fixed_replay", "fixed_branches"):
            raise ValueError("Frozen control-bank contract required")
        compatible = copy.copy(args)
        compatible.control = dict(args.control, contract=V8_SETTINGS)
        NativeLongSession.__init__(self, compatible, cache)
        self.args = args
        self.base_thresholds, self.margins = limits()
        self.report.update(protocol=PROTOCOL, contract=SETTINGS, trigger_method="v82_frozen",
            cohort_group=self.task["cohort_group"])
        self.save()

    def physical_recovery(self, obs, main, q0, steps, operator, branch):
        controller = self.env.env.robots[0].controller
        scale = np.asarray(controller.output_max, float)[:3]
        for value, expected in ((controller.input_max, 1), (controller.input_min, -1)):
            np.testing.assert_allclose(value, expected, atol=0, rtol=0)
        np.testing.assert_allclose(controller.output_min, -np.asarray(controller.output_max), atol=0, rtol=0)
        if np.min(scale) < .01:
            raise ValueError("Controller cannot represent bounded Cartesian commands")
        initial = np.asarray(obs["robot0_eef_pos"], float).copy()
        history = np.stack([main[max(0, q0-offset)]["proprio"][:3] for offset in (1, 2, 3)])
        last = main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1].copy()
        target, durations, grips = waypoints(operator, initial, history, float(last[6]))
        branch.update(initial_eef=initial.tolist(), history_eef=history.tolist(),
            recovery_targets=target.tolist(), phase_durations=durations, phase_grippers=grips,
            controller_translation_scale=scale.tolist(), last_native_gripper=float(last[6]))
        writer, success, index = ShardWriter(self.args.output / "physical"), False, 0
        try:
            for phase, duration in enumerate(durations):
                for _ in range(duration):
                    if success or steps >= 520:
                        break
                    position = np.asarray(obs["robot0_eef_pos"], float).copy()
                    action = physical_action(operator, position, target[phase], grips[phase], scale)
                    before, tick = np.asarray(self.env.get_sim_state(), float).copy(), time.monotonic()
                    obs, success = self.step(action, steps)
                    elapsed = time.monotonic()-tick
                    after_position = np.asarray(obs["robot0_eef_pos"], float).copy()
                    writer.append(dict(query=np.int32(index), action_step=np.int32(steps), phase=np.int16(phase),
                        action=action, target=target[phase].copy(), eef_before=position, eef_after=after_position,
                        sim_before=before, sim_after=np.asarray(self.env.get_sim_state(), float).copy(),
                        success=np.bool_(success), environment_seconds=np.float64(elapsed),
                        nominal_delta_m=action[:3]*scale,
                        nominal_prediction_error_m=after_position-position-action[:3]*scale,
                        target_error_before_m=np.float64(np.linalg.norm(position-target[phase])),
                        target_error_after_m=np.float64(np.linalg.norm(after_position-target[phase]))))
                    last, steps, index = action, steps+1, index+1
                    branch["environment_seconds"] += elapsed
        finally:
            branch["physical_shards"] = writer.close()
        branch["physical_steps"] = index
        return obs, steps, success, last

    def branches(self):
        arm = self.args.control["arm"]
        family, repeat = decode_arm(arm)
        replay_dir = Path(self.args.control["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if (replay["status"] != "completed" or replay["c0"]["status"] != "passed" or
                not replay["online_triggers_exact"] or not replay["v82_triggers_exact"]):
            raise ValueError("Complete exact C0 and causal v8.2 required")
        event = replay["events"][0]
        if any(event[key] != value for key, value in self.args.control["event"].items()):
            raise ValueError("Scheduled event differs")
        location = replay_dir / "events" / event["event_id"] / "snapshot"
        if (digest(location / "manifest.json") != event["snapshot_manifest_sha256"] or
                digest(replay_dir / "environment_rng/manifest.json") != replay["rng_tape_sha256"]):
            raise ValueError("Snapshot or RNG tape changed")
        self.tape = list(records(replay_dir / "environment_rng"))
        saved, main = load_snapshot(location), list(records(self.parent / "main"))
        q0, initial_steps = event["start_query"], event["action_steps_before"]
        if saved["query"] != q0 or saved["action_steps"] != initial_steps:
            raise ValueError("Wrong causal fork")
        self.report.update(events=[event], rng_tape_sha256=replay["rng_tape_sha256"],
            c0=dict(status="passed", replay_directory=str(replay_dir),
                replay_result_sha256=digest(replay_dir / "result.json"),
                replay_branch_sha256=digest(replay_dir / "c0/branch.json")))
        live, trace = V82TriggerMonitor(), ModeTrace()
        for q, row in enumerate(main[:q0]):
            status = live.update(row[PROBS_KEY], row["proprio"][:3])
            trace.commit(score_status(status), thresholds_at(q, self.base_thresholds), self.margins)
        if live.first["v82_frozen"] != q0-1:
            raise ValueError("Intervention precedes the frozen alarm")
        self.new_env()
        obs = restore(self.env, saved, self.rng)
        probe = self.entry_probe(obs, live, trace, main, q0, initial_steps)
        operator = choose_operator(family, probe["component_scores"], probe["selector_thresholds"], self.args.main_id, repeat)
        branch = dict(status="running", arm=arm, family=family, repeat=repeat, operator=operator,
            event_id=event["event_id"], start_query=q0, entry_mode=probe["mode"].item().decode(),
            entry_probe_exact=True, entry_probe_manifest_sha256=digest(self.args.output / "entry_probe/manifest.json"),
            starts_after_main_complete=True, starts_after_c0=True, queries=0, actual_model_queries=1,
            entry_probe_queries=1, physical_steps=0, changed_chunks=0, changed_commands=0,
            initial_action_steps=initial_steps, action_steps=0, success=False,
            inference_seconds=float(probe["inference_seconds"]), environment_seconds=0.)
        self.report["branches"] = [branch]
        writer = ShardWriter(self.args.output / "suffix")
        steps, success, started = initial_steps, False, time.monotonic()
        last = main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1].copy()
        try:
            if operator in PHYSICAL:
                obs, steps, success, last = self.physical_recovery(obs, main, q0, steps, operator, branch)
            action_filter = ActionFilter(operator, last)
            while not success and steps < 520:
                index, q = branch["queries"], q0+branch["queries"]
                noise = (self.rng.standard_normal((10, 24)).astype(np.float32) if family == "native"
                         else noise_for(self.args.main_id, repeat, index))
                request, response, inference = self.query(obs, noise)
                alarm = live.update(response[PROBS_KEY], request["observation/state"][:3])
                if live.v8.v7.query != q:
                    raise ValueError("Unexecuted probe advanced history")
                features = trace.commit(score_status(alarm), thresholds_at(q, self.base_thresholds), self.margins)
                before = np.asarray(self.env.get_sim_state(), float).copy()
                raw = np.asarray(response["actions"]).copy()
                actual, previous = raw.copy(), action_filter.previous.copy()
                chunk = action_filter.chunk_size(index)
                tick, count = time.monotonic(), 0
                for offset in range(min(chunk, 520-steps)):
                    actual[offset] = action_filter.command(raw[offset], index)
                    obs, success = self.step(actual[offset], steps+offset)
                    count += 1
                    if success:
                        break
                elapsed = time.monotonic()-tick
                after = np.asarray(self.env.get_sim_state(), float).copy()
                row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, elapsed)
                different = np.any(actual[:count] != raw[:count], axis=1)
                row.update(features)
                row.update(generated_actions=raw, actions=actual, filter_previous=previous,
                    filter_after=action_filter.previous.copy(), operator=np.asarray(operator, dtype="S24"),
                    relative_query=np.int32(index), candidate_count=np.int16(1), candidate_id=np.int16(0),
                    requested_chunk=np.int16(chunk), transformed_commands=np.int16(different.sum()),
                    transform_active=np.bool_(operator in TRANSFORM and index < 8),
                    selected_risk=np.float64(scalar_cost(features["normalized_scores"])),
                    v82_alarm=np.bool_(alarm["v82_alarm"]), v82_first=np.int32(alarm["v82_first"]),
                    knn_score=np.float32(alarm["knn_score"]), cosine_score=np.float32(alarm["cosine_score"]),
                    v8_scores=alarm["v8_scores"], behavior_stall=np.bool_(alarm["behavior_stall"]))
                if family == "native":
                    if q >= len(main):
                        raise ValueError("Native suffix exceeded original")
                    for key in main[q]:
                        if key not in ("inference_seconds", "environment_seconds"):
                            np.testing.assert_array_equal(row[key], main[q][key], err_msg="Native exact: "+key)
                writer.append(row)
                steps += count
                branch.update(queries=index+1, actual_model_queries=branch["actual_model_queries"]+1,
                    changed_chunks=branch["changed_chunks"]+int(different.any()),
                    changed_commands=branch["changed_commands"]+int(different.sum()),
                    inference_seconds=branch["inference_seconds"]+inference,
                    environment_seconds=branch["environment_seconds"]+elapsed)
                if branch["queries"] % 4 == 0:
                    self.save()
            if family == "native":
                if branch["queries"] != len(main)-q0 or success != self.original["success"]:
                    raise ValueError("Native suffix incomplete")
                self.report["all_native_suffixes_exact"] = True
            if branch["actual_model_queries"] != self.report["actual_model_queries"]:
                raise ValueError("Model queries unaccounted for")
            branch.update(status="completed", success=success, action_steps=steps-initial_steps,
                final_action_steps=steps, elapsed_seconds=time.monotonic()-started)
        finally:
            branch["suffix_shards"] = writer.close()
        atomic_json(self.args.output / "branch.json", branch)
        self.close_env()
        self.save()


if __name__ == "__main__":
    import collect_native_long_modes as worker
    worker.NativeModesSession = ControlBankSession
    worker.main()
