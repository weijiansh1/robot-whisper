#!/usr/bin/env python3
"""Restore audited Long parents, verify C0, then run next-query interventions."""

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

from collect_preflight_worker import (ALLOWED_GPUS, PROFILE, PROFILE_SHA256, component_digests,
    input_digest, restore, snapshot, valid_response, verify_egl_device,
    GlobalIntrinsicProfile, IntrinsicGuardMonitor)
from collection_routes import ALL_FIELDS, CAPTURE_KEY, PROBS_KEY, INTERVENTION_KEY, INTERVENTION_PROBS_KEY
from collection_storage import ShardWriter, atomic_json, atomic_npz, digest, load_snapshot, records, save_snapshot
from collection_protocol import stable_id
from continuation_experiment import (ARMS, CANDIDATES, INTERVENTIONS, PROTOCOL, REPLICATES,
                                     noise_for, seed_for, select_edge, short_count)


def run(args, cache):
    from benchmarks.run_benchmarks import ROOTS, variants, load_suite
    from libero.libero import get_libero_path
    import libero.libero
    from libero.libero.envs import OffScreenRenderEnv
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.preprocess import build_policy_observation
    from himoe_libero_bridge.suites import get_suite

    task = args.continuation
    parent = Path(task["parent_directory"])
    original = json.loads((parent / "result.json").read_text())
    if not original["main_complete"] or original["status"] != "completed" or original["main_intervention"]:
        raise ValueError("Native parent is not complete")
    for filename, key in (("main_complete.json", "parent_commit_sha256"), ("main/manifest.json", "parent_manifest_sha256")):
        if digest(parent / filename) != task[key]:
            raise ValueError("Native parent changed")
    if args.gpu not in ALLOWED_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only environment and allowed GPU required")
    if ROOTS[args.benchmark] not in Path(libero.libero.__file__).parents:
        raise ValueError("Wrong LIBERO extension")
    if args.exact_noise_fastpath and args.benchmark == "plus" and "noise_fastpath" not in cache:
        from collection_noise import install
        cache["noise_fastpath"] = install()
    if "variants" not in cache:
        cache["variants"] = {r["variant_id"]: r for r in variants(args.benchmark)}
        cache["profile"] = GlobalIntrinsicProfile.load(PROFILE)
    row, profile = cache["variants"][args.variant], cache["profile"]
    if (args.model != "long" or row != original["variant"] or args.main_id != original["main_id"] or
            args.seed != original["seed"] or args.init_index != original["init_index"] or
            digest(PROFILE) != PROFILE_SHA256):
        raise ValueError("Parent task, seed, model or alarm identity differs")
    suite = load_suite(row["registry"], cache.setdefault("suites", {}))
    benchmark_task = suite.get_task(row["registry_index"])
    initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
    if (benchmark_task.name != row["task_name"] or
            hashlib.sha256(np.ascontiguousarray(initial[args.init_index]).tobytes()).hexdigest() != original["initial_state_sha256"]):
        raise ValueError("Parent initial state changed")
    bddl = Path(original["bddl_path"])
    if digest(bddl) != original["bddl_sha256"]:
        raise ValueError("Parent BDDL changed")
    requested_bddl = Path(get_libero_path("bddl_files")) / benchmark_task.problem_folder / benchmark_task.bddl_file
    render_gpu = args.render_gpu
    egl_uuid = verify_egl_device(render_gpu)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", protocol=PROTOCOL, main_id=args.main_id, variant=row,
        parent_directory=str(parent), parent_commit_sha256=task["parent_commit_sha256"],
        parent_manifest_sha256=task["parent_manifest_sha256"], parent_queries=original["queries"],
        model=args.model, gpu=args.gpu, render_gpu=render_gpu, egl_device_uuid=egl_uuid,
        seed=args.seed, init_index=args.init_index, hidden_capture=False, main_intervention=False,
        main_complete=True, reused_main=True, queries=0, action_steps=original["action_steps"],
        success=original["success"], first_alarm_query=original["first_alarm_query"],
        first_alarms=task["first_alarms"], terminal_alarms=task["terminal_alarms"],
        events=copy.deepcopy(task["events"]), branches=[], c0=None, invalid_pair=False,
        candidate_queries=0, reused_candidate_queries=0, actual_model_queries=0,
        environment_pid=os.getpid(), worker_job_index=cache["jobs"],
        interventions=INTERVENTIONS, analysis_role=task["analysis_role"])
    started = time.monotonic()
    env, writer, client = None, None, None
    rng = np.random.default_rng(args.seed)
    prompt = original["prompt"]
    horizon = row["horizon_steps"]

    def new_env():
        lock_path = Path(tempfile.gettempdir()) / ("himoe-collection-egl-%d-%d.lock" % (os.getuid(), render_gpu))
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            random.seed(args.seed)
            np.random.seed(args.seed)
            instance = OffScreenRenderEnv(bddl_file_name=str(requested_bddl), camera_heights=224,
                camera_widths=224, render_gpu_device_id=render_gpu, horizon=horizon + 11)
            try:
                instance.seed(args.seed)
                instance.reset()
                instance.set_init_state(initial[args.init_index])
            except BaseException:
                instance.close()
                raise
        return instance

    def query(obs, noise=None, swap=False):
        request = build_policy_observation(obs, prompt)
        request["flow/noise"] = rng.standard_normal((10, 24)).astype(np.float32) if noise is None else noise
        request["routing/capture"] = True
        request[CAPTURE_KEY] = True
        if swap:
            request[INTERVENTION_KEY] = "swap1_near"
        tick = time.monotonic()
        response = client.infer(request)
        duration = time.monotonic() - tick
        report["actual_model_queries"] += 1
        valid_response(response, swap)
        if response["flow/noise_sha256"] != hashlib.sha256(request["flow/noise"].tobytes()).hexdigest():
            raise ValueError("Noise acknowledgement mismatch")
        return request, response, duration

    def advance(obs, actions, count):
        tick, executed, success = time.monotonic(), 0, False
        for action in actions[:count]:
            obs, _, done, _ = env.step(action.tolist())
            executed += 1
            success = bool(env.check_success())
            if success:
                break
            if done:
                raise ValueError("Premature simulator termination")
        return obs, executed, success, time.monotonic() - tick

    def make_record(request, response, q, steps, count, before, after, success, alarm, inference, simulation):
        record = {field: np.asarray(response[field]) for field in ALL_FIELDS}
        record.update(query=np.int32(q), action_steps_before=np.int32(steps), executed_action_count=np.int16(count),
            actions=response["actions"], noise=request["flow/noise"], proprio=np.asarray(request["observation/state"]).copy(),
            input_sha256=np.asarray(input_digest(request), dtype="S64"), input_component_sha256=component_digests(request),
            sim_before=before, sim_after=after, success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
            alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], np.float32),
            inference_seconds=np.float64(inference), environment_seconds=np.float64(simulation))
        return record

    try:
        client = cache.get("client")
        if client is None:
            client = PolicyClient(port=args.port, connect_timeout=30, inference_timeout=180)
            cache["client"] = client
        metadata = client.metadata
        if metadata.get("bundle_physical_gpu") != args.gpu or metadata.get("checkpoint_sha256") != get_suite(args.model).weights_sha256:
            raise ValueError("Wrong model service")
        for key in ("checkpoint_sha256", "normalization_stats_sha256", "libero_wrist_layout",
                    "himoe_upstream_commit", "himoe_working_tree_diff_sha256"):
            if metadata[key] != original["model_metadata"][key]:
                raise ValueError("Parent model identity differs: " + key)
        report["model_metadata"] = metadata
        atomic_json(args.output / "result.json", report)
        env = new_env()
        saved = load_snapshot(parent / "preflight_q000")
        obs = restore(env, saved, rng)
        monitor = IntrinsicGuardMonitor(profile)
        writer = ShardWriter(args.output / "c0")
        targets = {event["start_query"]: event for event in report["events"]}
        c0 = dict(status="running", start_query=0, compared_queries=0, starts_after_main_complete=True,
            parent_main_id=args.main_id, parent_commit_sha256=task["parent_commit_sha256"],
            source_snapshot_manifest_sha256=digest(parent / "preflight_q000/manifest.json"),
            sim_max_error=0.0, final_success=None)
        report["c0"] = c0
        for expected in records(parent / "main"):
            q, steps = int(expected["query"]), int(expected["action_steps_before"])
            if q in targets:
                event = targets[q]
                location = args.output / "events" / event["event_id"] / "snapshot"
                save_snapshot(location, snapshot(env, obs, rng, q, steps, prompt))
                event.update(snapshot_manifest_sha256=digest(location / "manifest.json"), action_steps_before=steps)
            request, response, inference = query(obs)
            alarm = monitor.update(response[PROBS_KEY])
            before = np.asarray(env.get_sim_state(), np.float64).copy()
            obs, count, success, simulation = advance(obs, response["actions"], int(expected["executed_action_count"]))
            after = np.asarray(env.get_sim_state(), np.float64).copy()
            actual = make_record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
            for field in expected:
                if field not in ("inference_seconds", "environment_seconds"):
                    np.testing.assert_array_equal(actual[field], expected[field], err_msg="C0 query %d: %s" % (q, field))
            writer.append(actual)
            c0.update(compared_queries=q + 1, final_success=success)
            if (q + 1) % 8 == 0:
                atomic_json(args.output / "result.json", report)
        c0["shards"] = writer.close()
        writer = None
        if c0["compared_queries"] != original["queries"] or c0["final_success"] != original["success"]:
            raise ValueError("Incomplete C0")
        c0["status"] = "passed"
        atomic_json(args.output / "c0/branch.json", c0)
        atomic_json(args.output / "result.json", report)
        env.close()
        env = None

        for event in report["events"]:
            event_dir = args.output / "events" / event["event_id"]
            saved = load_snapshot(event_dir / "snapshot")
            prefix_monitor = IntrinsicGuardMonitor(profile)
            for prefix in records(parent / "main"):
                if int(prefix["query"]) >= saved["query"]:
                    break
                prefix_monitor.update(prefix[PROBS_KEY])
            for replicate in range(REPLICATES):
                q0, event_key = event["start_query"], event["event_id"]
                env = new_env()
                obs = restore(env, saved, rng)
                pool = []
                pool_dir = event_dir / "pools" / str(replicate)
                writer = ShardWriter(pool_dir / "candidates", block_size=CANDIDATES)
                for candidate in range(CANDIDATES):
                    request, response, inference = query(obs, noise_for(args.main_id, event_key, replicate, q0, candidate))
                    pool.append((request, response, inference))
                    evidence = {field: np.asarray(response[field]) for field in ALL_FIELDS}
                    evidence.update(query=np.int32(candidate), source_query=np.int32(q0),
                        actions=response["actions"], noise=request["flow/noise"],
                        input_sha256=np.asarray(input_digest(request), dtype="S64"),
                        inference_seconds=np.float64(inference),
                        policy_seed=np.uint32(seed_for(args.main_id, event_key, replicate, q0, "policy", candidate)))
                    writer.append(evidence)
                    report["candidate_queries"] += 1
                writer.close()
                writer = None
                selected, scores = select_edge(np.stack([item[1][PROBS_KEY] for item in pool]))
                selection = dict(selected=selected, scores=scores.tolist(), random_candidate=0,
                    candidate_manifest_sha256=digest(pool_dir / "candidates/manifest.json"),
                    event_id=event_key, replicate=replicate, source_query=q0,
                    selector=INTERVENTIONS["edge_noise"], candidate_queries=CANDIDATES)
                atomic_json(pool_dir / "selection.json", selection)
                env.close()
                env = None
                for arm in ARMS:
                    directory = event_dir / "branches" / arm / str(replicate)
                    branch = dict(status="running", branch_id=stable_id(PROTOCOL, args.main_id, event_key, arm, replicate),
                        parent_main_id=args.main_id, event_id=event_key, arm=arm, replicate=replicate,
                        start_query=q0, parent_prefix_queries=[0, q0], alarm_query=q0 - 1, methods=event["methods"],
                        parent_commit_sha256=task["parent_commit_sha256"],
                        snapshot_manifest_sha256=event["snapshot_manifest_sha256"],
                        selection_sha256=digest(pool_dir / "selection.json"), starts_after_main_complete=True,
                        starts_after_c0=True, remaining_action_budget=horizon - saved["action_steps"],
                        first_chunk_limit=short_count(args.main_id, event_key, replicate, q0) if arm == "short_chunk" else 10,
                        first_candidate=selected if arm == "edge_noise" else 0,
                        queries=0, action_steps=0, success=False)
                    report["branches"].append(branch)
                    env = new_env()
                    obs = restore(env, saved, rng)
                    monitor = copy.deepcopy(prefix_monitor)
                    writer = ShardWriter(directory / "suffix")
                    while branch["action_steps"] < branch["remaining_action_budget"] and not branch["success"]:
                        index = branch["queries"]
                        q, steps = q0 + index, saved["action_steps"] + branch["action_steps"]
                        env_seed = seed_for(args.main_id, event_key, replicate, q, "environment")
                        random.seed(env_seed)
                        np.random.seed(env_seed)
                        swap = arm == "swap1" and index == 0
                        candidate = branch["first_candidate"] if index == 0 else 0
                        cached = index == 0 and not swap
                        if cached:
                            request, response, _ = pool[candidate]
                            inference = 0.0
                            if input_digest(build_policy_observation(obs, prompt)) != input_digest(request):
                                raise ValueError("Candidate observation changed on restore")
                            report["reused_candidate_queries"] += 1
                        else:
                            request, response, inference = query(obs, noise_for(args.main_id, event_key, replicate, q, candidate), swap)
                        if index == 0:
                            branch["first_input_sha256"] = input_digest(request)
                        if swap:
                            atomic_npz(directory / "intervention.npz", {INTERVENTION_PROBS_KEY: response[INTERVENTION_PROBS_KEY]})
                            branch["intervention_evidence_sha256"] = digest(directory / "intervention.npz")
                        alarm = monitor.update(response[PROBS_KEY])
                        before = np.asarray(env.get_sim_state(), np.float64).copy()
                        limit = min(branch["first_chunk_limit"] if index == 0 else 10, horizon - steps)
                        obs, count, success, simulation = advance(obs, response["actions"], limit)
                        after = np.asarray(env.get_sim_state(), np.float64).copy()
                        if not np.isfinite(after).all():
                            raise ValueError("Non-finite branch state")
                        record = make_record(request, response, q, steps, count, before, after, success, alarm, inference, simulation)
                        record.update(policy_seed=np.uint32(seed_for(args.main_id, event_key, replicate, q, "policy", candidate)),
                            environment_seed=np.uint32(env_seed), candidate_id=np.int16(candidate),
                            requested_action_count=np.int16(limit), intervention_applied=np.bool_(index == 0 and arm != "random"),
                            route_swap_applied=np.bool_(swap), cached_candidate=np.bool_(cached))
                        writer.append(record)
                        branch.update(queries=index + 1, action_steps=branch["action_steps"] + count, success=success)
                        if branch["queries"] % 8 == 0:
                            atomic_json(args.output / "result.json", report)
                    branch["shards"] = writer.close()
                    writer = None
                    branch["status"] = "completed"
                    atomic_json(directory / "branch.json", branch)
                    env.close()
                    env = None
                    atomic_json(args.output / "result.json", report)
            atomic_json(event_dir / "event.json", event)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["invalid_pair"] = True
        report["error"] = traceback.format_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if client is not None and report["status"] == "failed":
            client.close()
            cache.pop("client", None)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "result.json", report)
    return report


def worker_loop(args):
    cache = {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            task.variant, task.seed, task.output = message["variant"], message["seed"], Path(message["output"])
            task.main_id, task.init_index = message["main_id"], int(message["init_index"])
            task.continuation = message["continuation"]
            cache["jobs"] = cache.get("jobs", 0) + 1
            response = dict(variant=task.variant, pid=os.getpid(), worker_job_index=cache["jobs"])
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    run(task, cache)
                response["exit_code"] = 0
            except Exception:
                traceback.print_exc(file=sys.stderr)
                response["exit_code"] = 1
            print("COLLECTION_RESULT " + json.dumps(response), flush=True)
    finally:
        if cache.get("client") is not None:
            cache["client"].close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("pro", "plus"), required=True)
    parser.add_argument("--model", choices=("long",), required=True)
    parser.add_argument("--gpu", type=int, choices=ALLOWED_GPUS, required=True)
    parser.add_argument("--render-gpu", type=int, choices=ALLOWED_GPUS, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--worker-loop", action="store_true", required=True)
    parser.add_argument("--paired-branches", action="store_true", required=True)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    worker_loop(parser.parse_args())
