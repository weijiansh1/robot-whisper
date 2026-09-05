#!/usr/bin/env python3
"""Exactly replay saved failure prefixes, then continue them online."""

from __future__ import print_function

import argparse
import csv
import hashlib
import json
import os
import pathlib
import tempfile
import time
import traceback

import numpy as np


WORKSPACE = pathlib.Path(__file__).resolve().parents[2]
BRIDGE_SRC = WORKSPACE / "himoe-libero-wrist-fix/src"
if str(BRIDGE_SRC) not in os.sys.path:
    os.sys.path.insert(0, str(BRIDGE_SRC))

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.stem, suffix=".npz"
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, str(path))
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def next_flow_noises(seed, source_queries, extra_queries):
    rng = np.random.default_rng(int(seed))
    for _ in range(int(source_queries)):
        rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    return [
        rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
        for _ in range(int(extra_queries))
    ]


def result_paths(output, case):
    root = output / "episodes" / case["cohort"] / case["server_suite"]
    stem = "%s_task%02d_ep%03d" % (
        case["case_id"],
        int(case["task_id"]),
        int(case["episode"]),
    )
    return root / (stem + ".json"), root / (stem + ".npz")


def is_complete(path, case):
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (
        value.get("status") == "complete"
        and value.get("case_id") == case["case_id"]
        and value.get("source_npz_sha256") == case["source_npz_sha256"]
    )


def max_abs(left, right):
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.float64)
                - np.asarray(right, dtype=np.float64)
            )
        )
    )


def run_case(case, client, args, server_identity):
    started = time.time()
    json_path, npz_path = result_paths(args.output, case)
    source_path = pathlib.Path(case["source_npz"])
    if sha256_file(source_path) != case["source_npz_sha256"]:
        raise RuntimeError("source NPZ hash changed")
    with np.load(str(source_path), allow_pickle=False) as archive:
        source_states = np.asarray(archive["state"], dtype=np.float32)
        source_actions = np.asarray(archive["actions"], dtype=np.float32)
        source_sim_states = np.asarray(archive["sim_state"], dtype=np.float32)

    source_queries = int(case["original_inference_calls"])
    if source_actions.shape != (source_queries, 10, 7):
        raise RuntimeError("source action shape changed after freezing")
    config = EpisodeConfig(
        task_suite=case["benchmark"],
        task_id=int(case["task_id"]),
        init_state_id=int(case["init_state_id"]),
        seed=int(case["environment_seed"]),
        host=args.host,
        port=args.port,
        libero_root=args.libero_root,
        output_root=str(args.output),
        settle_steps=10,
        max_steps=int(case["original_action_steps"]) + args.extra_queries * 10,
        replan_steps=10,
        render_size=224,
        fps=20,
        inference_timeout=args.inference_timeout,
    )
    environment, observation, task, prompt = _load_task(config)
    max_sim_error = 0.0
    max_policy_state_error = 0.0
    extension_noises = []
    extension_actions = []
    extension_sim_states = []
    extension_successes = []
    extension_policy_states = []
    first_success_query = -1
    first_success_action = -1
    try:
        if str(task.name) != case["task"].split("/", 1)[1]:
            raise RuntimeError("task name differs from frozen source")
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        for query, chunk in enumerate(source_actions):
            actual_sim = np.asarray(environment.get_sim_state(), dtype=np.float32)
            sim_error = max_abs(actual_sim, source_sim_states[query])
            max_sim_error = max(max_sim_error, sim_error)
            policy_observation = build_policy_observation(observation, prompt)
            state_error = max_abs(
                policy_observation["observation/state"], source_states[query]
            )
            max_policy_state_error = max(max_policy_state_error, state_error)
            if sim_error != 0.0 or state_error != 0.0:
                raise RuntimeError(
                    "source prefix mismatch at query %d: sim=%g state=%g"
                    % (query, sim_error, state_error)
                )
            for action in chunk:
                observation, _, _, _ = environment.step(action.tolist())
                if environment.check_success():
                    raise RuntimeError(
                        "source prefix became successful before its recorded horizon"
                    )

        if bool(environment.check_success()):
            raise RuntimeError("source prefix is successful at its recorded horizon")
        horizon_sim_state = np.asarray(environment.get_sim_state(), dtype=np.float64)
        noises = next_flow_noises(
            int(case["flow_noise_seed"]), source_queries, args.extra_queries
        )
        extra_action_ordinal = 0
        inference_ms = []
        for extra_query, noise in enumerate(noises, 1):
            policy_observation = build_policy_observation(observation, prompt)
            extension_policy_states.append(
                np.asarray(policy_observation["observation/state"], dtype=np.float32)
            )
            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = noise
            before = time.time()
            response = validate_action_response(client.infer(request))
            inference_ms.append((time.time() - before) * 1000.0)
            if response.get(FLOW_NOISE_SHA256_KEY) != sha256_array(noise):
                raise RuntimeError("server did not acknowledge continuation flow noise")
            actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
            extension_noises.append(noise)
            extension_actions.append(actions)
            for action in actions[:10]:
                observation, _, _, _ = environment.step(action.tolist())
                extra_action_ordinal += 1
                success = bool(environment.check_success())
                extension_successes.append(success)
                extension_sim_states.append(
                    np.asarray(environment.get_sim_state(), dtype=np.float32)
                )
                if success:
                    first_success_query = extra_query
                    first_success_action = extra_action_ordinal
                    break
            if first_success_query >= 0:
                break

        final_sim_state = np.asarray(environment.get_sim_state(), dtype=np.float64)
        result = {
            "schema": "himoe.timeout_extension.case.v1",
            "status": "complete",
            "case_id": case["case_id"],
            "cohort": case["cohort"],
            "task": case["task"],
            "server_suite": case["server_suite"],
            "task_id": int(case["task_id"]),
            "episode": int(case["episode"]),
            "init_state_id": int(case["init_state_id"]),
            "flow_noise_seed": int(case["flow_noise_seed"]),
            "source_npz": case["source_npz"],
            "source_npz_sha256": case["source_npz_sha256"],
            "original_action_steps": int(case["original_action_steps"]),
            "original_inference_calls": source_queries,
            "exact_source_prefix_replay": True,
            "max_source_sim_state_error": max_sim_error,
            "max_source_policy_state_error": max_policy_state_error,
            "requested_extra_queries": args.extra_queries,
            "extra_queries_executed": len(extension_actions),
            "extra_action_steps_executed": len(extension_successes),
            "success_after_extension": first_success_query >= 0,
            "first_success_extra_query": first_success_query,
            "first_success_extra_action": first_success_action,
            "success_within_one_extra_query": 0 < first_success_action <= 10,
            "inference_ms": inference_ms,
            "wall_s": round(time.time() - started, 3),
            "server_instance_id": server_identity.get("server_instance_id"),
            "checkpoint_sha256": server_identity.get("checkpoint_sha256"),
            "libero_wrist_layout": server_identity.get("libero_wrist_layout"),
            "arrays": str(npz_path),
        }
        atomic_npz(
            npz_path,
            schema=np.asarray("himoe.timeout_extension.arrays.v1"),
            horizon_sim_state=horizon_sim_state,
            extension_flow_noises=np.stack(extension_noises),
            extension_actions=np.stack(extension_actions),
            extension_policy_states=np.stack(extension_policy_states),
            extension_sim_states_after=np.stack(extension_sim_states),
            extension_successes=np.asarray(extension_successes, dtype=np.bool_),
            final_sim_state=final_sim_state,
        )
        result["arrays_sha256"] = sha256_file(npz_path)
        atomic_json(json_path, result)
        return result
    finally:
        environment.close()


def parse_cases(path, suite, shard_index, shard_count):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        row
        for row in rows
        if row["server_suite"] == suite
        and int(row["case_id"]) % shard_count == shard_index
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--suite", choices=("goal", "spatial", "object", "long"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--extra-queries", type=int, default=10)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument(
        "--max-cases",
        type=int,
        help="debug-only cap after suite/shard selection; omit for formal runs",
    )
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.extra_queries != 10:
        raise ValueError("the frozen protocol requires exactly ten extra queries")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    cases = parse_cases(
        args.manifest.resolve(), args.suite, args.shard_index, args.shard_count
    )
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("max-cases must be positive")
        cases = cases[: args.max_cases]
    complete = errors = skipped = 0
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        metadata = dict(client.metadata)
        benchmark = "libero_10" if args.suite == "long" else "libero_" + args.suite
        validate_policy_suite(metadata, benchmark)
        if metadata.get("libero_wrist_layout") != "paper-right":
            raise RuntimeError("server does not use the source paper-right layout")
        atomic_json(
            args.output
            / "logs"
            / ("server_%s_shard%d.json" % (args.suite, args.shard_index)),
            metadata,
        )
        for position, case in enumerate(cases, 1):
            json_path, _ = result_paths(args.output, case)
            if is_complete(json_path, case):
                skipped += 1
                continue
            try:
                result = run_case(case, client, args, metadata)
                complete += 1
                print(
                    "[%s shard %d/%d] %d/%d case=%s success=%s at q=%d action=%d wall=%.1fs"
                    % (
                        args.suite,
                        args.shard_index,
                        args.shard_count,
                        position,
                        len(cases),
                        case["case_id"],
                        result["success_after_extension"],
                        result["first_success_extra_query"],
                        result["first_success_extra_action"],
                        result["wall_s"],
                    ),
                    flush=True,
                )
            except BaseException as error:
                errors += 1
                atomic_json(
                    json_path,
                    {
                        "schema": "himoe.timeout_extension.case.v1",
                        "status": "error",
                        "case_id": case["case_id"],
                        "cohort": case["cohort"],
                        "task": case["task"],
                        "episode": int(case["episode"]),
                        "source_npz_sha256": case["source_npz_sha256"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(
                    "[%s shard %d/%d] ERROR case=%s: %s"
                    % (
                        args.suite,
                        args.shard_index,
                        args.shard_count,
                        case["case_id"],
                        error,
                    ),
                    flush=True,
                )
    print(
        json.dumps(
            {
                "suite": args.suite,
                "shard": args.shard_index,
                "planned": len(cases),
                "completed_now": complete,
                "skipped": skipped,
                "errors": errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
