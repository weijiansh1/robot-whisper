#!/usr/bin/env python3
"""Execute paired v8/v8.2 selectors after exact original-Long C0 replay."""

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
from collect_preflight_worker import restore
from collection_routes import PROBS_KEY
from collection_storage import ShardWriter, atomic_json, digest, load_snapshot, records
from fixed_recovery_control import stalled
from v8_closed_loop import SETTINGS as V8_SETTINGS, RecoveryState, limits, preview, risk, score_status
from v82_closed_loop import ARMS, PROTOCOL, SETTINGS, V82TriggerMonitor, base_arm, selector_version, thresholds_at


class NativeV82Session(NativeLongSession):
    def __init__(self, args, cache):
        if args.control["contract"] != SETTINGS or args.control["kind"] not in ("fixed_replay", "fixed_branches"):
            raise ValueError("Frozen v8.2 recovery contract required")
        compatible = copy.copy(args)
        compatible.control = dict(args.control, contract=V8_SETTINGS)
        super().__init__(compatible, cache)
        self.args = args
        self.base_thresholds, self.margins = limits()
        self.report.update(protocol=PROTOCOL, contract=SETTINGS, trigger_method="v82_frozen")
        self.save()

    def replay(self):
        super().replay()
        live = V82TriggerMonitor()
        for row in records(self.args.output / "c0"):
            live.update(row[PROBS_KEY], row["proprio"][:3])
        if live.first != self.task["first_alarms"]:
            raise ValueError("Online v8.2 does not match the frozen offline first alarm")
        if live.first["v82_frozen"]+1 != self.task["events"][0]["start_query"]:
            raise ValueError("v8.2 fork must follow its first alarm")
        self.report.update(v82_triggers_exact=True, replay_first_alarms=live.first)

    def candidates(self, directory, obs, live, native_noise, default, arm, index, q):
        np.testing.assert_array_equal(self.thresholds, thresholds_at(arm, q, self.base_thresholds))
        item, values, selection, inference = super().candidates(
            directory, obs, live, native_noise, default, base_arm(arm), index, q)
        selection.update(arm=arm, selector_version=selector_version(arm),
            thresholds=self.thresholds.tolist(), margins=self.margins.tolist())
        atomic_json(directory / "selection.json", selection)
        return item, values, selection, inference

    def branches(self):
        arm = self.args.control["arm"]
        if arm not in ARMS:
            raise ValueError("Unknown v8.2 arm")
        replay_dir = Path(self.args.control["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if (replay["status"] != "completed" or replay["c0"]["status"] != "passed" or
                not replay["online_triggers_exact"] or not replay["v82_triggers_exact"]):
            raise ValueError("Complete exact C0 and online v8.2 required")
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
            selector_version=selector_version(arm), c0=dict(status="passed", replay_directory=str(replay_dir),
                replay_result_sha256=digest(replay_dir / "result.json"),
                replay_branch_sha256=digest(replay_dir / "c0/branch.json")))
        live = V82TriggerMonitor()
        for row in main[:q0]:
            live.update(row[PROBS_KEY], row["proprio"][:3])
        if live.first["v82_frozen"] != q0-1:
            raise ValueError("Noncausal v8.2 intervention time")
        self.new_env()
        obs = restore(self.env, saved, self.rng)
        state, positions = RecoveryState(), []
        branch = dict(status="running", arm=arm, selector_version=selector_version(arm),
            trigger_method="v82_frozen", event_id=event["event_id"], start_query=q0,
            starts_after_main_complete=True, starts_after_c0=True, queries=0, actual_model_queries=0,
            action_steps=0, success=False, initial_action_steps=initial_steps, candidate_pools=0,
            changed_chunks=0, selected_below_target=0, selected_below_alarm=0, population_below_target=0,
            inference_seconds=0., environment_seconds=0., selector_seconds=0., exit_reason=None,
            exit_query=None, post_exit_recurrence=False)
        self.report["branches"] = [branch]
        writer = ShardWriter(self.args.output / "suffix")
        started, steps, success = time.monotonic(), initial_steps, False
        try:
            while not success and steps < 520:
                index, q = branch["queries"], q0+branch["queries"]
                self.thresholds = thresholds_at(arm, q, self.base_thresholds)
                native_noise = self.rng.standard_normal((10, 24)).astype(np.float32)
                item = self.query(obs, native_noise)
                default_scores = preview(live.v8, [item[1][PROBS_KEY]])[0]
                default_risk = float(risk(default_scores, self.thresholds, self.margins))
                recovering = arm != "native" and state.active
                picked, pool_scores, selection_hash, forwards = 0, np.asarray([default_risk]), "", 1
                inference = default_inference = item[2]
                tick = time.monotonic()
                if recovering and np.isfinite(default_risk):
                    path = self.args.output / "pools" / str(index)
                    item, pool_scores, selection, inference = self.candidates(path, obs, live, native_noise, item, arm, index, q)
                    picked, forwards = selection["candidate"], 16
                    selection_hash = digest(path / "selection.json")
                    branch["candidate_pools"] += 1
                selector_seconds = time.monotonic()-tick-(inference-default_inference if forwards == 16 else 0.)
                request, response, _ = item
                alarm = live.update(response[PROBS_KEY], request["observation/state"][:3])
                if live.v8.v7.query != q:
                    raise ValueError("Candidate evaluation advanced executed-query history")
                selected_risk = float(risk(score_status(alarm), self.thresholds, self.margins))
                positions.append(np.asarray(request["observation/state"][:3]).copy())
                before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                obs, count, success, duration = self.advance(obs, response["actions"], min(10, 520-steps), steps)
                after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                if recovering:
                    state.commit(pool_scores)
                    if not state.active:
                        branch.update(exit_reason=state.exit_reason, exit_query=q)
                elif arm != "native" and selected_risk > 0:
                    branch["post_exit_recurrence"] = True
                row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, duration)
                row.update(relative_query=np.int32(index), candidate_id=np.int16(picked), candidate_count=np.int16(forwards),
                    recovery_active_before=np.bool_(recovering), recovery_active_after=np.bool_(state.active if arm != "native" else False),
                    stable_streak=np.int16(state.streak if arm != "native" else 0),
                    selection_sha256=np.asarray(selection_hash, dtype="S64"), component_scores=score_status(alarm),
                    selected_risk=np.float64(selected_risk), default_risk=np.float64(default_risk),
                    selector_thresholds=self.thresholds.copy(), v82_alarm=np.bool_(alarm["v82_alarm"]),
                    v82_first=np.int32(alarm["v82_first"]), knn_score=np.float32(alarm["knn_score"]),
                    cosine_score=np.float32(alarm["cosine_score"]), v8_scores=alarm["v8_scores"],
                    behavior_stall=np.bool_(alarm["behavior_stall"]), post_recovery_stall_valid=np.bool_(len(positions) >= 6),
                    post_recovery_stall=np.bool_(stalled(positions)))
                if arm == "native":
                    if q >= len(main):
                        raise ValueError("Native suffix exceeded original")
                    for key in main[q]:
                        if key not in ("inference_seconds", "environment_seconds"):
                            np.testing.assert_array_equal(row[key], main[q][key], err_msg="Native exact: "+key)
                writer.append(row)
                steps += count
                branch.update(queries=index+1, actual_model_queries=branch["actual_model_queries"]+forwards,
                    changed_chunks=branch["changed_chunks"]+int(picked != 0),
                    selected_below_target=branch["selected_below_target"]+int(recovering and selected_risk <= -1),
                    selected_below_alarm=branch["selected_below_alarm"]+int(recovering and selected_risk < 0),
                    population_below_target=branch["population_below_target"]+int(recovering and len(pool_scores) == 16 and np.max(pool_scores) <= -1),
                    inference_seconds=branch["inference_seconds"]+inference, environment_seconds=branch["environment_seconds"]+duration,
                    selector_seconds=branch["selector_seconds"]+max(0., selector_seconds))
                if branch["queries"] % 4 == 0:
                    self.save()
            if arm == "native":
                if branch["queries"] != len(main)-q0 or success != self.original["success"]:
                    raise ValueError("Native suffix incomplete")
                self.report["all_native_suffixes_exact"] = True
            if arm != "native" and state.active:
                branch["exit_reason"] = "task_success" if success else "environment_horizon"
            branch.update(status="completed", success=success, action_steps=steps-initial_steps,
                final_action_steps=steps, elapsed_seconds=time.monotonic()-started)
        finally:
            branch["suffix_shards"] = writer.close()
        atomic_json(self.args.output / "branch.json", branch)
        self.close_env()
        self.save()


def run(args, cache):
    session = NativeV82Session(args, cache)
    started = time.monotonic()
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
