#!/usr/bin/env python3
"""Exact native replay followed by independently scheduled route-control branches."""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

import numpy as np

from adaptive_control import (ARMS, CONTRACT, GPUS, RENDER_GPUS, PROTOCOL, RouteRisk,
                              noise_for, seed_for, chunk_limit, candidate_scores, select_candidate)
from collect_preflight_worker import (component_digests, input_digest, restore, snapshot,
                                      valid_response, verify_egl_device)
from collection_routes import ALL_FIELDS, CAPTURE_KEY, PROBS_KEY
from collection_storage import ShardWriter, atomic_json, atomic_npz, digest, load_snapshot, records, save_snapshot


class RecoverySession:
    def __init__(self, args, cache):
        from benchmarks.run_benchmarks import ROOTS, variants, load_suite
        from libero.libero import get_libero_path
        import libero.libero
        from libero.libero.envs import OffScreenRenderEnv
        from himoe_libero_bridge.client import PolicyClient
        from himoe_libero_bridge.preprocess import build_policy_observation
        from himoe_libero_bridge.suites import get_suite

        self.args, self.cache, self.task = args, cache, args.control["parent"]
        self.parent = Path(self.task["parent_directory"])
        self.original = json.loads((self.parent / "result.json").read_text())
        original = self.original
        if (args.gpu not in GPUS or args.render_gpu not in RENDER_GPUS or
                os.environ.get("CUDA_VISIBLE_DEVICES") != "" or ROOTS[args.benchmark] not in Path(libero.libero.__file__).parents):
            raise ValueError("Recovery GPU/benchmark isolation")
        if original["status"] != "completed" or not original["main_complete"] or original["main_intervention"]:
            raise ValueError("Uncommitted or intervened parent")
        for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
            if digest(self.parent / filename) != self.task[key]:
                raise ValueError("Native parent changed")
        if args.exact_noise_fastpath and args.benchmark == "plus" and "noise_fastpath" not in cache:
            from collection_noise import install
            cache["noise_fastpath"] = install()
        if "variants" not in cache:
            cache["variants"] = {row["variant_id"]: row for row in variants(args.benchmark)}
            cache["risk"] = RouteRisk()
        self.row, self.risk = cache["variants"][args.variant], cache["risk"]
        if (args.model != "long" or self.row != original["variant"] or args.main_id != original["main_id"] or
                args.seed != original["seed"] or args.init_index != original["init_index"]):
            raise ValueError("Recovery parent identity")
        suite = load_suite(self.row["registry"], cache.setdefault("suites", {}))
        task = suite.get_task(self.row["registry_index"])
        self.initial = np.asarray(suite.get_task_init_states(self.row["registry_index"]))
        if (task.name != self.row["task_name"] or hashlib.sha256(np.ascontiguousarray(
                self.initial[args.init_index]).tobytes()).hexdigest() != original["initial_state_sha256"]):
            raise ValueError("Changed initial state")
        if digest(original["bddl_path"]) != original["bddl_sha256"]:
            raise ValueError("Changed BDDL")
        self.bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.env_class, self.build_observation = OffScreenRenderEnv, build_policy_observation
        self.prompt, self.horizon = original["prompt"], self.row["horizon_steps"]
        self.rng, self.env = np.random.default_rng(args.seed), None
        self.client = cache.get("client")
        if self.client is None:
            self.client = PolicyClient(port=args.port, connect_timeout=30, inference_timeout=180)
            cache["client"] = self.client
        metadata = self.client.metadata
        if metadata.get("bundle_physical_gpu") != args.gpu or metadata.get("checkpoint_sha256") != get_suite(args.model).weights_sha256:
            raise ValueError("Wrong recovery model endpoint")
        for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                    "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
            if metadata[key] != original["model_metadata"][key]:
                raise ValueError("Model identity differs: " + key)
        args.output.mkdir(parents=True, exist_ok=False)
        self.report = dict(status="running", protocol=PROTOCOL, contract=CONTRACT,
            main_id=args.main_id, job_kind=args.control["kind"], variant=self.row, parent_directory=str(self.parent),
            parent_commit_sha256=self.task["parent_commit_sha256"], parent_manifest_sha256=self.task["parent_manifest_sha256"],
            model=args.model, model_metadata=metadata, gpu=args.gpu, render_gpu=args.render_gpu,
            egl_device_uuid=verify_egl_device(args.render_gpu), seed=args.seed, init_index=args.init_index,
            hidden_capture=False, main_intervention=False, main_complete=True, reused_main=True,
            queries=0, action_steps=original["action_steps"], success=original["success"],
            first_alarm_query=original["first_alarm_query"], first_knn_alarm=self.task["first_alarm"],
            analysis_role=self.task["analysis_role"], c0=None, events=[], branches=[], invalid_pair=False,
            actual_model_queries=0, candidate_queries=0, reused_candidate_queries=0,
            environment_pid=os.getpid(), worker_job_index=cache["jobs"])
        self.save()

    def save(self):
        atomic_json(self.args.output / "result.json", self.report)

    def new_env(self):
        self.close_env()
        args = self.args
        lock_path = Path(tempfile.gettempdir()) / ("himoe-collection-egl-%d-%d.lock" % (os.getuid(), args.render_gpu))
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            random.seed(args.seed)
            np.random.seed(args.seed)
            self.env = self.env_class(bddl_file_name=str(self.bddl), camera_heights=224,
                camera_widths=224, render_gpu_device_id=args.render_gpu, horizon=self.horizon + 11)
            self.env.seed(args.seed)
            self.env.reset()
            self.env.set_init_state(self.initial[args.init_index])

    def close_env(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def query(self, obs, noise):
        request = self.build_observation(obs, self.prompt)
        request.update({"flow/noise": noise, "routing/capture": True, CAPTURE_KEY: True})
        tick = time.monotonic()
        response = self.client.infer(request)
        duration = time.monotonic() - tick
        self.report["actual_model_queries"] += 1
        valid_response(response)
        if response["flow/noise_sha256"] != hashlib.sha256(noise.tobytes()).hexdigest():
            raise ValueError("Recovery noise identity")
        return request, response, duration

    def advance(self, obs, actions, limit, steps, replicate=None):
        tick, count, success = time.monotonic(), 0, False
        for action in actions[:limit]:
            if replicate is not None:
                seed = seed_for(self.args.main_id, replicate, steps + count, "environment")
                random.seed(seed)
                np.random.seed(seed)
            obs, _, done, _ = self.env.step(action.tolist())
            count += 1
            success = bool(self.env.check_success())
            if success:
                break
            if done:
                raise ValueError("Premature recovery termination")
        return obs, count, success, time.monotonic() - tick

    @staticmethod
    def record(request, response, q, steps, count, before, after, success, alarm, inference, simulation):
        row = {field: np.asarray(response[field]) for field in ALL_FIELDS}
        row.update(query=np.int32(q), action_steps_before=np.int32(steps), executed_action_count=np.int16(count),
            actions=response["actions"], noise=request["flow/noise"], proprio=np.asarray(request["observation/state"]).copy(),
            input_sha256=np.asarray(input_digest(request), dtype="S64"), input_component_sha256=component_digests(request),
            sim_before=before, sim_after=after, success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
            alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], np.float32),
            inference_seconds=np.float64(inference), environment_seconds=np.float64(simulation))
        return row

    def replay(self):
        self.new_env()
        obs = restore(self.env, load_snapshot(self.parent / "preflight_q000"), self.rng)
        monitor, first = self.risk.monitor(), -1
        self.report["events"] = copy.deepcopy(self.task["events"])
        targets = {event["start_query"]: event for event in self.report["events"]}
        c0 = dict(status="running", start_query=0, compared_queries=0, starts_after_main_complete=True,
            parent_main_id=self.args.main_id, parent_commit_sha256=self.task["parent_commit_sha256"], final_success=None)
        self.report["c0"] = c0
        writer = ShardWriter(self.args.output / "c0")
        try:
            for expected in records(self.parent / "main"):
                q, steps = int(expected["query"]), int(expected["action_steps_before"])
                if q in targets:
                    location = self.args.output / "events" / targets[q]["event_id"] / "snapshot"
                    save_snapshot(location, snapshot(self.env, obs, self.rng, q, steps, self.prompt))
                    targets[q].update(snapshot_manifest_sha256=digest(location / "manifest.json"), action_steps_before=steps)
                request, response, inference = self.query(obs, self.rng.standard_normal((10, 24)).astype(np.float32))
                alarm = monitor.update(response[PROBS_KEY])
                _, score = self.risk.current(monitor)
                if first < 0 and score > self.risk.threshold:
                    first = q
                before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                obs, count, success, simulation = self.advance(obs, response["actions"], int(expected["executed_action_count"]), steps)
                after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                actual = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                for field in expected:
                    if field not in ("inference_seconds", "environment_seconds"):
                        np.testing.assert_array_equal(actual[field], expected[field], err_msg="C0 %d: %s" % (q, field))
                actual["knn_score"] = np.float32(score)
                writer.append(actual)
                c0.update(compared_queries=q + 1, final_success=success)
                if (q + 1) % 8 == 0:
                    self.save()
            c0["shards"] = writer.close()
        finally:
            writer.close()
        if (c0["compared_queries"] != self.original["queries"] or c0["final_success"] != self.original["success"] or
                first != self.task["first_alarm"]):
            raise ValueError("Incomplete C0 or online kNN trigger mismatch")
        c0.update(status="passed", online_knn_first=first, frozen_threshold=self.risk.threshold)
        atomic_json(self.args.output / "c0/branch.json", c0)
        for event in self.report["events"]:
            atomic_json(self.args.output / "events" / event["event_id"] / "event.json", event)
        self.close_env()

    def pool(self, directory, obs, monitor, replicate, index, q):
        pool = []
        writer = ShardWriter(directory / "candidates", block_size=4)
        try:
            for candidate in range(4):
                item = self.query(obs, noise_for(self.args.main_id, replicate, index, candidate))
                request, response, inference = item
                pool.append(item)
                row = {field: np.asarray(response[field]) for field in ALL_FIELDS}
                row.update(query=np.int32(candidate), source_query=np.int32(q), relative_query=np.int32(index),
                    noise=request["flow/noise"], actions=response["actions"],
                    input_sha256=np.asarray(input_digest(request), dtype="S64"),
                    policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy", candidate)),
                    inference_seconds=np.float64(inference))
                writer.append(row)
                self.report["candidate_queries"] += 1
            writer.close()
        finally:
            writer.close()
        scores = candidate_scores(np.stack([r[1][PROBS_KEY] for r in pool]),
            np.stack([r[1]["actions"] for r in pool]), monitor, self.risk)
        atomic_npz(directory / "scores.npz", scores)
        selection = dict(winners={name: select_candidate(name, scores) for name in ("center", "edge", "knn", "guarded_knn")},
            candidate_manifest_sha256=digest(directory / "candidates/manifest.json"),
            scores_sha256=digest(directory / "scores.npz"), relative_query=index, source_query=q,
            replicate=replicate, input_sha256=input_digest(pool[0][0]))
        atomic_json(directory / "selection.json", selection)
        return pool, selection

    def branches(self):
        spec = self.args.control
        replay_dir = Path(spec["replay_directory"])
        replay = json.loads((replay_dir / "result.json").read_text())
        if replay["status"] != "completed" or replay["c0"]["status"] != "passed" or replay["main_id"] != self.args.main_id:
            raise ValueError("Branches require a completed C0 for this parent")
        if replay["parent_commit_sha256"] != self.task["parent_commit_sha256"]:
            raise ValueError("Replay parent differs")
        event = next(row for row in replay["events"] if row["event_id"] == spec["event"]["event_id"])
        for key, value in spec["event"].items():
            if event[key] != value:
                raise ValueError("Scheduled branch event differs")
        self.report["events"] = [event]
        self.report["c0"] = dict(status="passed", compared_queries=0, replay_directory=str(replay_dir),
            replay_result_sha256=digest(replay_dir / "result.json"), replay_branch_sha256=digest(replay_dir / "c0/branch.json"))
        saved = load_snapshot(replay_dir / "events" / event["event_id"] / "snapshot")
        q0, replicate = event["start_query"], int(spec["replicate"])
        prefix_monitor = self.risk.monitor()
        for row in records(self.parent / "main"):
            if int(row["query"]) >= q0:
                break
            prefix_monitor.update(row[PROBS_KEY])
        self.new_env()
        obs = restore(self.env, saved, self.rng)
        initial_pool, initial_selection = self.pool(self.args.output / "initial_pool", obs, prefix_monitor, replicate, 0, q0)
        self.close_env()
        for arm in spec["arms"]:
            directory, control = self.args.output / "branches" / arm, ARMS[arm]
            self.new_env()
            obs = restore(self.env, saved, self.rng)
            monitor = copy.deepcopy(prefix_monitor)
            branch = dict(status="running", arm=arm, control=control, replicate=replicate, event_id=event["event_id"],
                timing=event["timing"], deployable=event["deployable"], start_query=q0,
                parent_main_id=self.args.main_id, parent_commit_sha256=self.task["parent_commit_sha256"],
                snapshot_manifest_sha256=event["snapshot_manifest_sha256"] ,
                starts_after_main_complete=True, starts_after_c0=True,
                remaining_action_budget=self.horizon - saved["action_steps"],
                queries=0, action_steps=0, success=False, deployment_model_queries=0,
                first_candidate=None, initial_selection_sha256=digest(self.args.output / "initial_pool/selection.json"))
            self.report["branches"].append(branch)
            writer = ShardWriter(directory / "suffix")
            try:
                while not branch["success"] and branch["action_steps"] < branch["remaining_action_budget"]:
                    index = branch["queries"]
                    q, steps = q0 + index, saved["action_steps"] + branch["action_steps"]
                    selecting = index < control["duration"] and control["selector"] != "fixed"
                    pool, selection = None, None
                    if index == 0:
                        pool, selection = initial_pool, initial_selection
                    elif selecting:
                        pool, selection = self.pool(directory / "pools" / str(index), obs, monitor, replicate, index, q)
                    candidate = (selection["winners"][control["selector"]] if selecting else
                                 control["candidate"] if index == 0 else 0)
                    cached = pool is not None
                    if cached:
                        request, response, _ = pool[candidate]
                        inference = 0.0
                        if input_digest(request) != input_digest(self.build_observation(obs, self.prompt)):
                            raise ValueError("Cached candidate observation mismatch")
                        self.report["reused_candidate_queries"] += 1
                    else:
                        request, response, inference = self.query(obs, noise_for(self.args.main_id, replicate, index, candidate))
                    branch["deployment_model_queries"] += 4 if selecting else 1
                    if index == 0:
                        branch["first_candidate"] = candidate
                    alarm = monitor.update(response[PROBS_KEY])
                    vector, score = self.risk.current(monitor)
                    before = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    limit = chunk_limit(arm, index, self.horizon - steps)
                    obs, count, success, simulation = self.advance(obs, response["actions"], limit, steps, replicate)
                    after = np.asarray(self.env.get_sim_state(), np.float64).copy()
                    if not np.isfinite(after).all():
                        raise ValueError("Non-finite recovery physics")
                    row = self.record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                    row.update(relative_query=np.int32(index), candidate_id=np.int16(candidate),
                        policy_seed=np.uint32(seed_for(self.args.main_id, replicate, index, "policy", candidate)),
                        environment_first_seed=np.uint32(seed_for(self.args.main_id, replicate, steps, "environment")),
                        environment_last_seed=np.uint32(seed_for(self.args.main_id, replicate, steps + count - 1, "environment")),
                        requested_action_count=np.int16(limit), selecting=np.bool_(selecting), cached_candidate=np.bool_(cached),
                        knn_score=np.float32(score), knn_vector=vector, route_swap_applied=np.bool_(False))
                    writer.append(row)
                    branch.update(queries=index + 1, action_steps=branch["action_steps"] + count, success=success)
                    if branch["queries"] % 8 == 0:
                        self.save()
                branch["shards"] = writer.close()
            finally:
                writer.close()
            branch["status"] = "completed"
            atomic_json(directory / "branch.json", branch)
            self.close_env()
            self.save()


def run(args, cache):
    started = time.monotonic()
    session = RecoverySession(args, cache)
    try:
        if args.control["kind"] == "replay":
            session.replay()
        elif args.control["kind"] == "branches":
            session.branches()
        else:
            raise ValueError("Unknown recovery job")
        session.report["status"] = "completed"
    except BaseException:
        session.report.update(status="failed", invalid_pair=True, error=traceback.format_exc())
        raise
    finally:
        session.close_env()
        session.report["elapsed_seconds"] = time.monotonic() - started
        session.save()


def worker_loop(args):
    cache = {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            task.variant, task.seed, task.output = message["variant"], message["seed"], Path(message["output"])
            task.main_id, task.init_index, task.control = message["main_id"], int(message["init_index"]), message["control"]
            cache["jobs"] = cache.get("jobs", 0) + 1
            result = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                result["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                result["exit_code"] = 1
            print("COLLECTION_RESULT " + json.dumps(result), flush=True)
    finally:
        if cache.get("client") is not None:
            cache["client"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--model", choices=("long",), required=True)
    parser.add_argument("--gpu", type=int, choices=GPUS, required=True)
    parser.add_argument("--render-gpu", type=int, choices=RENDER_GPUS, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-loop", action="store_true", required=True)
    parser.add_argument("--paired-branches", action="store_true", required=True)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    worker_loop(parser.parse_args())
