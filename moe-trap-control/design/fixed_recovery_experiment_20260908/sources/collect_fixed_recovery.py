#!/usr/bin/env python3
"""Replay native mains and compare a fixed physical recovery at causal event times."""

from __future__ import annotations

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

from collect_adaptive_control import RecoverySession
from collect_preflight_worker import restore
from collection_routes import PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from fixed_recovery_control import (PROTOCOL, ARMS, MODULE, TriggerMonitor, targets, recovery_action,
                                    rng_record, set_environment_rng, stalled)


class FixedSession(RecoverySession):
    def __init__(self, args, cache):
        super().__init__(args, cache)
        self.report.update(protocol=PROTOCOL, contract=args.control["contract"],
            first_alarms=self.task["first_alarms"], native_success=self.original["success"],
            rng_tape_sha256=None, all_native_suffixes_exact=False)
        self.tape_writer, self.tape = None, None

    def step(self, action, step):
        if self.tape_writer is not None:
            self.tape_writer.append(rng_record(step))
        if self.tape is not None:
            set_environment_rng(self.tape, self.args.main_id, step)
        obs, _, done, _ = self.env.step(np.asarray(action).tolist())
        success = bool(self.env.check_success())
        if done and not success and step + 1 < 520:
            raise ValueError("Premature physical suffix termination")
        if not np.isfinite(np.asarray(self.env.get_sim_state())).all():
            raise ValueError("Non-finite physics")
        return obs, success

    def advance(self, obs, actions, limit, steps, replicate=None):
        started, count, success = time.monotonic(), 0, False
        for action in actions[:limit]:
            obs, success = self.step(action, steps + count)
            count += 1
            if success:
                break
        return obs, count, success, time.monotonic() - started

    def replay(self):
        self.tape_writer = ShardWriter(self.args.output / "environment_rng", block_size=64)
        try:
            super().replay()
        finally:
            self.tape_writer.close()
            self.tape_writer = None
        self.report["rng_tape_sha256"] = digest(self.args.output / "environment_rng/manifest.json")
        tape = list(records(self.args.output / "environment_rng"))
        if len(tape) != self.original["action_steps"]:
            raise ValueError("Incomplete original environment RNG tape")
        monitor = TriggerMonitor()
        for row in records(self.args.output / "c0"):
            monitor.update(row[PROBS_KEY], row["proprio"][:3])
        for method, query in monitor.first.items():
            if query != self.task["first_alarms"][method]:
                raise ValueError("C0 causal trigger differs: " + method)
        self.report["online_triggers_exact"] = True

    def branches(self):
        replay_dir = Path(self.args.control["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if replay["status"] != "completed" or replay["c0"]["status"] != "passed" or not replay["online_triggers_exact"]:
            raise ValueError("Physical branches require complete exact C0")
        event = next(e for e in replay["events"] if e["event_id"] == self.args.control["event"]["event_id"])
        location = replay_dir / "events" / event["event_id"] / "snapshot"
        if digest(location / "manifest.json") != event["snapshot_manifest_sha256"]:
            raise ValueError("Physical snapshot commitment changed")
        if digest(replay_dir / "environment_rng/manifest.json") != replay["rng_tape_sha256"]:
            raise ValueError("Original RNG tape changed")
        self.tape = list(records(replay_dir / "environment_rng"))
        saved, main = load_snapshot(location), list(records(self.parent / "main"))
        q0, initial_steps = event["start_query"], event["action_steps_before"]
        if saved["query"] != q0 or saved["action_steps"] != initial_steps:
            raise ValueError("Physical event placement")
        self.report.update(events=[event], rng_tape_sha256=replay["rng_tape_sha256"],
            c0=dict(status="passed", replay_directory=str(replay_dir),
                replay_result_sha256=digest(replay_dir / "result.json"),
                replay_branch_sha256=digest(replay_dir / "c0/branch.json")))
        prefix = TriggerMonitor()
        for row in main[:q0]:
            prefix.update(row[PROBS_KEY], row["proprio"][:3])
        for arm in ARMS:
            directory = self.args.output / "branches" / arm
            self.new_env()
            obs = restore(self.env, saved, self.rng)
            controller = self.env.env.robots[0].controller
            scale = np.asarray(controller.output_max, dtype=np.float64)[:3]
            for value, expected in ((controller.input_max, 1), (controller.input_min, -1)):
                np.testing.assert_allclose(value, expected, atol=0, rtol=0)
            np.testing.assert_allclose(controller.output_min, -np.asarray(controller.output_max), atol=0, rtol=0)
            if np.min(scale) < MODULE["max_translation_step_m"]:
                raise ValueError("Controller action scaling cannot represent the recovery step")
            initial_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
            target = targets(initial_eef, main[max(0, q0-MODULE["target_history_queries"])]["proprio"][:3])
            gripper = float(main[q0-1]["actions"][int(main[q0-1]["executed_action_count"])-1, 6])
            branch = dict(status="running", arm=arm, start_query=q0, event_id=event["event_id"],
                starts_after_main_complete=True, starts_after_c0=True, queries=0, recovery_steps=0,
                action_steps=0, success=False, actual_model_queries=0, initial_action_steps=initial_steps,
                recovery_targets=target.tolist(), controller_translation_scale=scale.tolist(),
                initial_eef=initial_eef.tolist(), last_native_gripper=gripper,
                inference_seconds=0., environment_seconds=0.)
            self.report["branches"].append(branch)
            physical, writer = ShardWriter(directory / "physical", block_size=8), ShardWriter(directory / "suffix")
            monitor, positions = copy.deepcopy(prefix), []
            started = time.monotonic()
            steps, success = initial_steps, False
            try:
                if arm != "native":
                    for index in range(min(2*MODULE["phase_steps"], 520-steps)):
                        phase = index // MODULE["phase_steps"]
                        position = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                        action = recovery_action(arm, position, target[phase], gripper, scale)
                        before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        tick = time.monotonic()
                        obs, success = self.step(action, steps)
                        duration = time.monotonic()-tick
                        after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                        physical.append(dict(query=np.int32(index), action_step=np.int32(steps), phase=np.int8(phase),
                            action=action, target=target[phase], eef_before=position,
                            eef_after=np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy(),
                            sim_before=before, sim_after=after, success=np.bool_(success),
                            environment_seconds=np.float64(duration)))
                        steps += 1
                        branch["recovery_steps"] += 1
                        branch["environment_seconds"] += duration
                        if success:
                            break
                branch.update(after_recovery_eef=np.asarray(obs["robot0_eef_pos"], float).tolist(),
                              success_during_recovery=success)
                while not success and steps < 520:
                    index, q = branch["queries"], q0+branch["queries"]
                    request, response, inference = self.query(obs, self.rng.standard_normal((10,24)).astype(np.float32))
                    alarm = monitor.update(response[PROBS_KEY], request["observation/state"][:3])
                    positions.append(np.asarray(request["observation/state"][:3]).copy())
                    before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    obs, count, success, duration = self.advance(obs, response["actions"], min(10,520-steps), steps)
                    after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    row = self.record(request,response,q,steps,count,before,after,success,alarm,inference,duration)
                    row.update(relative_query=np.int32(index), knn_score=np.float32(alarm["knn_score"]),
                        cosine_score=np.float32(alarm["cosine_score"]), v8_scores=alarm["v8_scores"],
                        behavior_stall=np.bool_(alarm["behavior_stall"]),
                        post_recovery_stall_valid=np.bool_(len(positions)>=6),
                        post_recovery_stall=np.bool_(stalled(positions)))
                    if arm == "native":
                        if q >= len(main):
                            raise ValueError("Native suffix grew beyond original")
                        for key in main[q]:
                            if key not in ("inference_seconds","environment_seconds"):
                                np.testing.assert_array_equal(row[key],main[q][key],err_msg="Native RNG-tape equivalence "+key)
                    writer.append(row)
                    steps += count
                    branch.update(queries=index+1, actual_model_queries=index+1,
                        inference_seconds=branch["inference_seconds"]+inference,
                        environment_seconds=branch["environment_seconds"]+duration)
                    if branch["queries"] % 8 == 0:
                        self.save()
                if arm == "native":
                    if branch["queries"] != len(main)-q0 or success != self.original["success"]:
                        raise ValueError("Native complete suffix mismatch")
                    self.report["all_native_suffixes_exact"] = True
                branch.update(action_steps=steps-initial_steps, success=success, status="completed",
                    elapsed_seconds=time.monotonic()-started, final_action_steps=steps)
            finally:
                branch["physical_shards"],branch["suffix_shards"] = physical.close(),writer.close()
            atomic_json(directory / "branch.json", branch)
            self.close_env()
            self.save()


def run(args, cache):
    started = time.monotonic()
    session = FixedSession(args, cache)
    try:
        if args.control["kind"] == "fixed_replay":
            session.replay()
        elif args.control["kind"] == "fixed_branches":
            session.branches()
        else:
            raise ValueError("Unknown physical recovery job")
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
    parser.add_argument("--benchmark",choices=("pro","plus"),required=True)
    parser.add_argument("--model",choices=("long",),default="long")
    parser.add_argument("--gpu",type=int,choices=(0,1,2,3),required=True)
    parser.add_argument("--render-gpu",type=int,choices=(0,3),required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--exact-noise-fastpath",action="store_true")
    args,cache = parser.parse_args(),{}
    try:
        for line in sys.stdin:
            message=json.loads(line)
            task=copy.copy(args)
            task.variant,task.seed,task.output=message["variant"],message["seed"],Path(message["output"])
            task.main_id,task.init_index,task.control=message["main_id"],message["init_index"],message["control"]
            cache["jobs"]=cache.get("jobs",0)+1
            result=dict(variant=task.variant,pid=os.getpid(),worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task,cache)
                result["exit_code"]=0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"]=1
            print("COLLECTION_RESULT "+json.dumps(result),flush=True)
    finally:
        if cache.get("client"):
            cache["client"].close()


if __name__ == "__main__":
    main()
