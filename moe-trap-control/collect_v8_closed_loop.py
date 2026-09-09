#!/usr/bin/env python3
"""Exact C0 and single-arm closed-loop v8 recovery in isolated benchmark workers."""

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

from collect_fixed_recovery import FixedSession
from collect_preflight_worker import input_digest, restore
from collection_routes import ALL_FIELDS, PROBS_KEY
from collection_storage import ShardWriter, atomic_json, atomic_npz, digest, load_snapshot, records
from fixed_recovery_control import TriggerMonitor, stalled
from v8_closed_loop import (PROTOCOL, ARMS, SETTINGS, RecoveryState, choose, extra_noises,
                            guided_population, limits, preview, risk, score_status)


class ClosedLoopSession(FixedSession):
    def __init__(self, args, cache):
        super().__init__(args, cache)
        if args.control["contract"] != SETTINGS:
            raise ValueError("Frozen selector settings changed")
        self.thresholds, self.margins = limits()
        self.report.update(protocol=PROTOCOL, contract=SETTINGS)

    def candidates(self, directory, obs, live, native_noise, default, arm, index, q):
        noise = extra_noises(self.args.main_id, index)
        first_noises = np.concatenate([native_noise[None], noise[:7]])
        pool = [default]+[self.query(obs, z) for z in first_noises[1:]]
        first_scores = preview(live.v8, [p[1][PROBS_KEY] for p in pool])
        first_risk = risk(first_scores, self.thresholds, self.margins)
        optimizer = dict(elite=np.full(4, -1, np.int32), mean=np.zeros((10, 24)), std=np.ones((10, 24)))
        second_noises = noise[7:]
        if arm.startswith("guided"):
            second_noises, optimizer = guided_population(first_noises, first_risk, second_noises)
        pool.extend(self.query(obs, z) for z in second_noises)
        scores = np.vstack([first_scores, preview(live.v8, [p[1][PROBS_KEY] for p in pool[8:]])])
        values = risk(scores, self.thresholds, self.margins)
        picked = choose(arm, values, self.args.main_id, index)
        writer = ShardWriter(directory / "candidates", block_size=8)
        try:
            expected_input = input_digest(default[0])
            for i, (request, response, inference) in enumerate(pool):
                if input_digest(request) != expected_input or not np.isfinite(response["actions"]).all():
                    raise ValueError("Candidate input/action mismatch")
                row = {key: np.asarray(response[key]) for key in ALL_FIELDS}
                row.update(query=np.int32(i), source_query=np.int32(q), relative_query=np.int32(index),
                    noise=request["flow/noise"], actions=response["actions"],
                    input_sha256=np.asarray(expected_input, dtype="S64"),
                    component_scores=scores[i], risk=np.float64(values[i]), inference_seconds=np.float64(inference))
                writer.append(row)
        finally:
            writer.close()
        atomic_npz(directory / "optimizer.npz", {key: np.asarray(value) for key, value in optimizer.items()})
        selection = dict(candidate=picked, arm=arm, source_query=q, relative_query=index,
            pool_size=len(pool), input_sha256=expected_input,
            manifest_sha256=digest(directory / "candidates/manifest.json"),
            optimizer_sha256=digest(directory / "optimizer.npz"),
            scores=values.tolist(), selected_risk=float(values[picked]), default_risk=float(values[0]))
        atomic_json(directory / "selection.json", selection)
        return pool[picked], values, selection, sum(item[2] for item in pool)

    def branches(self):
        arm = self.args.control["arm"]
        if arm not in ARMS:
            raise ValueError("Unknown selector arm")
        replay_dir = Path(self.args.control["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if replay["status"] != "completed" or replay["c0"]["status"] != "passed" or not replay["online_triggers_exact"]:
            raise ValueError("Recovery requires complete exact C0")
        event = next(e for e in replay["events"] if e["event_id"] == self.args.control["event"]["event_id"])
        location = replay_dir / "events" / event["event_id"] / "snapshot"
        if digest(location / "manifest.json") != event["snapshot_manifest_sha256"] or \
                digest(replay_dir / "environment_rng/manifest.json") != replay["rng_tape_sha256"]:
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
        live = TriggerMonitor()
        for row in main[:q0]:
            live.update(row[PROBS_KEY], row["proprio"][:3])
        self.new_env()
        obs = restore(self.env, saved, self.rng)
        state, positions = RecoveryState(), []
        branch = dict(status="running", arm=arm, event_id=event["event_id"], start_query=q0,
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
                row.update(relative_query=np.int32(index), candidate_id=np.int16(picked),
                    candidate_count=np.int16(forwards), recovery_active_before=np.bool_(recovering),
                    recovery_active_after=np.bool_(state.active if arm != "native" else False),
                    stable_streak=np.int16(state.streak if arm != "native" else 0),
                    selection_sha256=np.asarray(selection_hash, dtype="S64"),
                    component_scores=score_status(alarm), selected_risk=np.float64(selected_risk),
                    default_risk=np.float64(default_risk), knn_score=np.float32(alarm["knn_score"]),
                    cosine_score=np.float32(alarm["cosine_score"]), v8_scores=alarm["v8_scores"],
                    behavior_stall=np.bool_(alarm["behavior_stall"]),
                    post_recovery_stall_valid=np.bool_(len(positions) >= 6),
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
                    inference_seconds=branch["inference_seconds"]+inference,
                    environment_seconds=branch["environment_seconds"]+duration,
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
            self.close_env()
        if branch["actual_model_queries"] != self.report["actual_model_queries"]:
            raise ValueError("Candidate query ledger mismatch")
        atomic_json(self.args.output / "branch.json", branch)
        self.save()


def run(args, cache):
    started = time.monotonic()
    session = ClosedLoopSession(args, cache)
    try:
        if args.control["kind"] == "fixed_replay":
            session.replay()
        elif args.control["kind"] == "fixed_branches":
            session.branches()
        else:
            raise ValueError("Unknown closed-loop job")
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
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--model", choices=("long",), default="long")
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--render-gpu", type=int, choices=(0, 3), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    args, cache = parser.parse_args(), {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
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
