#!/usr/bin/env python3
"""Collect a complete native Pro/Plus rollout, then test restored C0 continuation."""

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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "himoe-route-capture"))
sys.path.insert(0, str(HERE.parent / "moe-v7-0905/method"))
from branch_snapshot import save_full_state, restore_full_state
from collection_routes import (CAPTURE_KEY, PROBS_KEY, FIELDS, AS_FIELDS, ALL_FIELDS,
                               INTERVENTION_KEY, INTERVENTION_PROBS_KEY)
from collection_storage import ShardWriter, atomic_json, atomic_npz, digest, load_snapshot, records, save_snapshot
from collection_protocol import (PROTOCOL, REPLICATES, branch_noise, event_id, stable_id, stream_seed)
from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor
from probe_collection_render import verify_egl_device

ALLOWED_GPUS = (0, 1, 2, 3, 4, 5, 7)
PROFILE = HERE.parent / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
PROFILE_SHA256 = "a92c9d1103487ddf62e4b369c7f23b6637127730dffa8780f54423cf580e3054"
OBSERVABLE_FIELDS = ("_time_since_last_sample", "_current_delay", "_current_observed_value", "_sampled")
MJ_EXTRA_FIELDS = ("ctrl", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat", "userdata")


def numeric_attributes(instance):
    return {key: copy.deepcopy(value) for key, value in vars(instance).items()
            if value is None or isinstance(value, (bool, int, float, np.number, np.ndarray))}


def input_digest(request):
    result = hashlib.sha256()
    for key in ("observation/image", "observation/wrist_image", "observation/state"):
        array = np.ascontiguousarray(request[key])
        result.update(key.encode())
        result.update(str(array.dtype).encode())
        result.update(str(array.shape).encode())
        result.update(array.tobytes())
    result.update(request["prompt"].encode())
    return result.hexdigest()


def component_digests(request):
    return np.asarray([hashlib.sha256(np.ascontiguousarray(request[key]).tobytes()).hexdigest()
        for key in ("observation/image", "observation/wrist_image", "observation/state")], dtype="S64")


def snapshot(env, obs, rng, query, steps, prompt):
    return dict(physics=save_full_state(env), observation=copy.deepcopy(obs),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(), policy_rng=copy.deepcopy(rng.bit_generator.state),
        observable_cache=copy.deepcopy(env.env._obs_cache),
        observables={name: {field: copy.deepcopy(getattr(item, field)) for field in OBSERVABLE_FIELDS}
                     for name, item in env.env._observables.items()},
        controller_numeric=[numeric_attributes(robot.controller) for robot in env.env.robots],
        gripper_current_action=[np.array(robot.gripper.current_action, copy=True) for robot in env.env.robots],
        mjdata_extra={key: np.array(getattr(env.env.sim.data, key), copy=True) for key in MJ_EXTRA_FIELDS},
        query=query, action_steps=steps, prompt=str(prompt))


def restore(env, saved, rng):
    restore_full_state(env, saved["physics"])
    if len(env.env.robots) != len(saved["controller_numeric"]):
        raise ValueError("Restored robot count differs")
    for index, robot in enumerate(env.env.robots):
        for key, value in saved["controller_numeric"][index].items():
            setattr(robot.controller, key, copy.deepcopy(value))
        robot.gripper.current_action = saved["gripper_current_action"][index].copy()
    for key, value in saved["mjdata_extra"].items():
        getattr(env.env.sim.data, key)[:] = value
    env.env._obs_cache = copy.deepcopy(saved["observable_cache"])
    if set(env.env._observables) != set(saved["observables"]):
        raise ValueError("Restored observable names differ")
    for name, state in saved["observables"].items():
        for key, value in state.items():
            setattr(env.env._observables[name], key, copy.deepcopy(value))
    np.random.set_state(saved["numpy_rng"])
    random.setstate(saved["python_rng"])
    rng.bit_generator.state = copy.deepcopy(saved["policy_rng"])
    return copy.deepcopy(saved["observation"])


def valid_response(response, intervention=False):
    if response.get("collection/routing_mode") != ("swap1_near" if intervention else "native_only"):
        raise ValueError("Unexpected routing mode")
    for field in ALL_FIELDS:
        expected = ((8, 10, 11, 32 if field == PROBS_KEY else 4) if field in FIELDS else
                    (4, 10, 11, 3 if field == AS_FIELDS[0] else 1))
        if np.asarray(response[field]).shape != expected or not np.isfinite(response[field]).all():
            raise ValueError("Invalid full HB response: " + field)
    if any("hidden" in key.lower() for key in response):
        raise ValueError("Hidden capture is forbidden")


def run(args, cache=None):
    from PIL import Image
    from benchmarks.run_benchmarks import ROOTS, variants, load_suite
    from libero.libero import get_libero_path
    import libero.libero
    from libero.libero.envs import OffScreenRenderEnv
    from himoe_libero_bridge.client import PolicyClient
    from himoe_libero_bridge.preprocess import build_policy_observation
    from himoe_libero_bridge.suites import get_suite

    persistent = cache is not None
    cache = {} if cache is None else cache
    noise_fastpath = dict(enabled=False)
    if args.exact_noise_fastpath and args.benchmark == "plus":
        from collection_noise import install
        if "noise_fastpath" not in cache:
            cache["noise_fastpath"] = install()
        noise_fastpath = cache["noise_fastpath"]
    if args.gpu not in ALLOWED_GPUS or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Environment worker must be CPU-only and exclude physical GPU 6")
    render_gpu = args.gpu if args.render_gpu is None else args.render_gpu
    egl_uuid = verify_egl_device(render_gpu)
    if ROOTS[args.benchmark] not in Path(libero.libero.__file__).parents:
        raise ValueError("Incorrect LIBERO extension package")
    if digest(PROFILE) != PROFILE_SHA256:
        raise ValueError("Frozen v7 profile identity changed")
    if "profile" not in cache:
        cache["profile"] = GlobalIntrinsicProfile.load(PROFILE)
        cache["variants"] = {row["variant_id"]: row for row in variants(args.benchmark)}
    profile = cache["profile"]
    row = cache["variants"][args.variant]
    if row["suite"] != get_suite(args.model).benchmark:
        raise ValueError("Variant suite differs from the selected model")
    suite = load_suite(row["registry"], cache.setdefault("suites", {}))
    task = suite.get_task(row["registry_index"])
    if task.name != row["task_name"]:
        raise ValueError("Registry identity changed")
    initial = np.asarray(suite.get_task_init_states(row["registry_index"]))
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    args.output.mkdir(parents=True, exist_ok=False)
    main_id = args.main_id or stable_id("pilot", row["variant_id"], args.seed, args.init_index)
    if not 0 <= args.init_index < len(initial):
        raise ValueError("Requested initial state is unavailable")
    report = dict(status="running", purpose="paired_intervention" if args.paired_branches else "full_rollout_and_c0_preflight",
        protocol=PROTOCOL, main_id=main_id, variant=row, gpu=args.gpu,
        model=args.model, model_port=args.port, seed=args.seed, init_index=args.init_index,
        hidden_capture=False, intervention=args.paired_branches, main_intervention=False,
        branches=[], event_id=None, invalid_pair=False,
        persistent_worker=persistent, environment_pid=os.getpid(), worker_job_index=cache.get("jobs", 0),
        render_gpu=render_gpu, egl_device_uuid=egl_uuid,
        noise_fastpath=noise_fastpath,
        alarm_profile_sha256=PROFILE_SHA256, formal_manifest_modified=False, action_steps=0, queries=0,
        first_alarm_query=None, success=False, c0=None, main_complete=False)
    if bddl.is_file():
        report["bddl_sha256"] = digest(bddl)
    actual_bddl = bddl
    if not actual_bddl.is_file() and args.benchmark == "plus" and "_view_" in bddl.name:
        actual_bddl = bddl.with_name(bddl.name.split("_view_")[0] + ".bddl")
    report["bddl_path"] = str(actual_bddl)
    report["bddl_sha256"] = digest(actual_bddl)
    report["initial_state_sha256"] = hashlib.sha256(np.ascontiguousarray(initial[args.init_index]).tobytes()).hexdigest()
    started = time.monotonic()
    env, client, writer = None, None, None
    main_snapshot = None
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    def new_env(settle=False):
        lock_path = Path(tempfile.gettempdir()) / ("himoe-collection-egl-%d-%d.lock" % (os.getuid(), render_gpu))
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            random.seed(args.seed)
            np.random.seed(args.seed)
            instance = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=224, camera_widths=224,
                                         render_gpu_device_id=render_gpu, horizon=row["horizon_steps"] + 11)
            try:
                instance.seed(args.seed)
                instance.reset()
                initial_obs = instance.set_init_state(initial[args.init_index])
                if settle:
                    for _ in range(10):
                        initial_obs, _, _, _ = instance.step([0.0] * 6 + [-1.0])
            except BaseException:
                instance.close()
                raise
        return instance, initial_obs

    def query(obs, prompt, noise=None, intervention=False):
        request = build_policy_observation(obs, prompt)
        request["flow/noise"] = rng.standard_normal((10, 24)).astype(np.float32) if noise is None else noise
        request["routing/capture"] = True
        request[CAPTURE_KEY] = True
        if intervention:
            request[INTERVENTION_KEY] = "swap1_near"
        tick = time.monotonic()
        response = client.infer(request)
        inference_seconds = time.monotonic() - tick
        valid_response(response, intervention)
        report["cuda_memory_mib"] = response.get("collection/cuda_memory_mib")
        if response["flow/noise_sha256"] != hashlib.sha256(request["flow/noise"].tobytes()).hexdigest():
            raise ValueError("Flow noise acknowledgement mismatch")
        return request, response, inference_seconds

    def advance(obs, actions, count):
        executed = 0
        success = False
        tick = time.monotonic()
        for action in actions[:count]:
            obs, _, done, _ = env.step(action.tolist())
            executed += 1
            success = bool(env.check_success())
            if success:
                break
            if done:
                raise RuntimeError("Simulator terminated before the allowed policy horizon")
        return obs, executed, success, time.monotonic() - tick

    try:
        client = cache.get("client")
        if client is None:
            client = PolicyClient(port=args.port, connect_timeout=30, inference_timeout=180)
            cache["client"] = client
        metadata = client.metadata
        if metadata.get("bundle_physical_gpu") != args.gpu or metadata.get("checkpoint_sha256") != get_suite(args.model).weights_sha256:
            raise ValueError("Model GPU or checkpoint identity mismatch")
        if not metadata.get("collection_full_hb_supported") or not metadata.get("collection_full_as_supported"):
            raise ValueError("Model server lacks complete HB/AS capture")
        if args.paired_branches and "swap1_near" not in metadata.get("collection_interventions", []):
            raise ValueError("Model server lacks the frozen intervention")
        report["model_metadata"] = metadata
        env, obs = new_env(settle=True)
        prompt = str(env.language_instruction)
        report["prompt"] = prompt
        first_request = build_policy_observation(obs, prompt)
        report["initial_camera_std"] = {}
        for field, name in (("observation/image", "agentview"), ("observation/wrist_image", "wrist")):
            report["initial_camera_std"][name] = float(np.asarray(first_request[field]).std())
            Image.fromarray(first_request[field]).save(args.output / (name + ".png"))
            if report["initial_camera_std"][name] < 1:
                raise ValueError("Blank %s camera on physical GPU %d" % (name, render_gpu))
        monitor = IntrinsicGuardMonitor(profile)
        writer = ShardWriter(args.output / "main")
        atomic_json(args.output / "result.json", report)
        while report["action_steps"] < row["horizon_steps"] and not report["success"]:
            q, steps = report["queries"], report["action_steps"]
            saved = snapshot(env, obs, rng, q, steps, prompt)
            request, response, inference_seconds = query(obs, prompt)
            alarm = monitor.update(response[PROBS_KEY])
            if q in (0, 5):
                target = args.output / ("preflight_q%03d" % q)
                save_snapshot(target, saved)
                main_snapshot = target
            if alarm["alarm"] and report["first_alarm_query"] is None:
                report["first_alarm_query"] = q
                report["alarm_mechanism"] = alarm["mechanism"]
                report["event_id"] = event_id(main_id, q)
                save_snapshot(args.output / "alarm_snapshot", saved)
            sim_before = saved["physics"]["sim"]
            obs, executed, success, env_seconds = advance(obs, response["actions"],
                min(10, row["horizon_steps"] - steps))
            sim_after = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            if not np.isfinite(sim_after).all():
                raise ValueError("Non-finite simulator state")
            record = {field: np.asarray(response[field]) for field in ALL_FIELDS}
            record.update(query=np.int32(q), action_steps_before=np.int32(steps),
                executed_action_count=np.int16(executed), actions=response["actions"], noise=request["flow/noise"],
                input_sha256=np.asarray(input_digest(request), dtype="S64"), sim_before=sim_before, sim_after=sim_after,
                input_component_sha256=component_digests(request),
                success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
                alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], dtype=np.float32),
                inference_seconds=np.float64(inference_seconds), environment_seconds=np.float64(env_seconds))
            writer.append(record)
            report.update(queries=q + 1, action_steps=steps + executed, success=success)
            if (q + 1) % 8 == 0:
                atomic_json(args.output / "result.json", report)
        report["main_shards"] = writer.close()
        writer = None
        report["main_complete"] = True
        report["main_elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "main_complete.json", dict(queries=report["queries"], action_steps=report["action_steps"],
            main_id=main_id, success=report["success"], manifest_sha256=digest(args.output / "main/manifest.json")))
        atomic_json(args.output / "result.json", report)
        Image.fromarray(build_policy_observation(obs, prompt)["observation/image"]).save(args.output / "agentview-final.png")
        env.close()
        env = None
        target = args.output / "alarm_snapshot" if report["first_alarm_query"] is not None else main_snapshot
        saved = load_snapshot(target)
        parent_commit_sha = digest(args.output / "main_complete.json")
        snapshot_sha = digest(target / "manifest.json")
        env, _ = new_env()
        obs = restore(env, saved, rng)
        monitor = IntrinsicGuardMonitor(profile)
        c0 = dict(status="running", start_query=int(saved["query"]), compared_queries=0,
                  snapshot=target.name, first_divergence=None, sim_max_error=0.0,
                  starts_after_main_complete=True, final_success=None, parent_main_id=main_id,
                  event_id=report["event_id"], arm="C0", replicate=0,
                  parent_prefix_queries=[0, int(saved["query"])], parent_commit_sha256=parent_commit_sha,
                  snapshot_manifest_sha256=snapshot_sha)
        writer = ShardWriter(args.output / "c0")
        for expected in records(args.output / "main"):
            q = int(expected["query"])
            if q < saved["query"]:
                monitor.update(expected[PROBS_KEY])
                continue
            request, response, inference_seconds = query(obs, prompt)
            alarm = monitor.update(response[PROBS_KEY])
            differences = []
            if input_digest(request) != expected["input_sha256"].item().decode():
                differences.append("observation")
                if "input_component_sha256" in expected:
                    differences.extend(name for name, actual, saved_hash in zip(("image", "wrist_image", "state"),
                        component_digests(request), expected["input_component_sha256"]) if actual != saved_hash)
            for field, actual in (("noise", request["flow/noise"]), ("actions", response["actions"]),
                                  *[(field, response[field]) for field in ALL_FIELDS]):
                if not np.array_equal(actual, expected[field]):
                    differences.append(field)
            if bool(alarm["alarm"]) != bool(expected["alarm"]):
                differences.append("alarm")
            sim_before = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            obs, executed, success, env_seconds = advance(obs, response["actions"], int(expected["executed_action_count"]))
            sim_after = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            error = float(np.abs(sim_after - expected["sim_after"]).max())
            c0["sim_max_error"] = max(c0["sim_max_error"], error)
            if not np.array_equal(sim_before, expected["sim_before"]) or not np.array_equal(sim_after, expected["sim_after"]):
                differences.append("sim_state")
            if executed != int(expected["executed_action_count"]) or success != bool(expected["success"]):
                differences.append("termination")
            record = {field: np.asarray(response[field]) for field in ALL_FIELDS}
            record.update(query=np.int32(q), action_steps_before=expected["action_steps_before"],
                executed_action_count=np.int16(executed), actions=response["actions"], noise=request["flow/noise"],
                input_sha256=np.asarray(input_digest(request), dtype="S64"), sim_before=sim_before, sim_after=sim_after,
                input_component_sha256=component_digests(request),
                success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
                alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], dtype=np.float32),
                inference_seconds=np.float64(inference_seconds), environment_seconds=np.float64(env_seconds))
            writer.append(record)
            c0["compared_queries"] += 1
            c0["final_success"] = success
            if differences:
                c0["first_divergence"] = dict(query=q, fields=differences, sim_max_error=error)
                break
        c0["shards"] = writer.close()
        writer = None
        c0["status"] = "passed" if c0["first_divergence"] is None else "fidelity_failed"
        report["c0"] = c0
        report["invalid_pair"] = c0["status"] != "passed"
        atomic_json(args.output / "c0/branch.json", c0)
        atomic_json(args.output / "result.json", report)
        env.close()
        env = None
        if args.paired_branches and report["first_alarm_query"] is not None and not report["invalid_pair"]:
            for replicate in range(REPLICATES):
                for arm in ("C1", "T1"):
                    branch_id = stable_id(PROTOCOL, main_id, report["event_id"], arm, replicate)
                    directory = args.output / "branches" / arm / str(replicate)
                    branch = dict(status="running", branch_id=branch_id, arm=arm, replicate=replicate,
                        parent_main_id=main_id, event_id=report["event_id"], start_query=int(saved["query"]),
                        parent_prefix_queries=[0, int(saved["query"])], parent_commit_sha256=parent_commit_sha,
                        snapshot_manifest_sha256=snapshot_sha, starts_after_main_complete=True,
                        remaining_action_budget=row["horizon_steps"] - int(saved["action_steps"]),
                        queries=0, action_steps=0, success=False)
                    report["branches"].append(branch)
                    atomic_json(directory / "branch.json", branch)
                    env, _ = new_env()
                    obs = restore(env, saved, rng)
                    monitor = IntrinsicGuardMonitor(profile)
                    for prefix in records(args.output / "main"):
                        if int(prefix["query"]) >= saved["query"]:
                            break
                        monitor.update(prefix[PROBS_KEY])
                    writer = ShardWriter(directory / "suffix")
                    while branch["action_steps"] < branch["remaining_action_budget"] and not branch["success"]:
                        q = int(saved["query"]) + branch["queries"]
                        steps = int(saved["action_steps"]) + branch["action_steps"]
                        apply_swap = arm == "T1" and branch["queries"] == 0
                        env_seed = stream_seed(main_id, report["event_id"], replicate, q, "environment")
                        # Equal streams are derived per query, independently of branch length and execution order.
                        random.seed(env_seed)
                        np.random.seed(env_seed)
                        request, response, inference_seconds = query(obs, prompt,
                            branch_noise(main_id, report["event_id"], replicate, q), apply_swap)
                        if branch["queries"] == 0:
                            branch["first_input_sha256"] = input_digest(request)
                        if apply_swap:
                            atomic_npz(directory / "intervention.npz", {INTERVENTION_PROBS_KEY: response[INTERVENTION_PROBS_KEY]})
                            branch["intervention_evidence_sha256"] = digest(directory / "intervention.npz")
                        alarm = monitor.update(response[PROBS_KEY])
                        sim_before = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                        obs, executed, success, env_seconds = advance(obs, response["actions"],
                            min(10, row["horizon_steps"] - steps))
                        sim_after = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
                        if not np.isfinite(sim_after).all():
                            raise ValueError("Non-finite branch simulator state")
                        record = {field: np.asarray(response[field]) for field in ALL_FIELDS}
                        record.update(query=np.int32(q), action_steps_before=np.int32(steps),
                            executed_action_count=np.int16(executed), actions=response["actions"], noise=request["flow/noise"],
                            input_sha256=np.asarray(input_digest(request), dtype="S64"),
                            input_component_sha256=component_digests(request), sim_before=sim_before, sim_after=sim_after,
                            success=np.bool_(success), alarm=np.bool_(alarm["alarm"]),
                            alarm_scores=np.asarray([alarm[k] for k in ("freeze_score", "acceleration_score", "periodicity_score")], dtype=np.float32),
                            policy_seed=np.uint32(stream_seed(main_id, report["event_id"], replicate, q, "policy")),
                            environment_seed=np.uint32(env_seed), intervention_applied=np.bool_(apply_swap),
                            inference_seconds=np.float64(inference_seconds), environment_seconds=np.float64(env_seconds))
                        writer.append(record)
                        branch.update(queries=branch["queries"] + 1, action_steps=branch["action_steps"] + executed, success=success)
                        if branch["queries"] % 8 == 0:
                            atomic_json(args.output / "result.json", report)
                    branch["shards"] = writer.close()
                    writer = None
                    branch["status"] = "completed"
                    atomic_json(directory / "branch.json", branch)
                    env.close()
                    env = None
                    atomic_json(args.output / "result.json", report)
        report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        if client is not None and (not persistent or report["status"] == "failed"):
            client.close()
            cache.pop("client", None)
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(args.output / "result.json", report)
    print(json.dumps({key: report[key] for key in ("status", "queries", "action_steps", "success", "first_alarm_query", "c0")}), flush=True)
    return report


def worker_loop(args):
    cache = {}
    try:
        for line in sys.stdin:
            message = json.loads(line)
            task = copy.copy(args)
            task.variant, task.seed, task.output = message["variant"], message["seed"], Path(message["output"])
            task.main_id, task.init_index = message.get("main_id"), int(message.get("init_index", 0))
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
    parser.add_argument("--model", choices=("goal", "spatial", "object", "long"), default="goal")
    parser.add_argument("--variant")
    parser.add_argument("--main-id")
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--paired-branches", action="store_true")
    parser.add_argument("--worker-loop", action="store_true")
    parser.add_argument("--gpu", type=int, choices=ALLOWED_GPUS, required=True)
    parser.add_argument("--render-gpu", type=int, choices=ALLOWED_GPUS)
    parser.add_argument("--exact-noise-fastpath", action="store_true")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.worker_loop and (args.variant is None or args.output is None):
        parser.error("--variant and --output are required for a single rollout")
    worker_loop(args) if args.worker_loop else run(args)
