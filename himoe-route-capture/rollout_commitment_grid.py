"""Paired LIBERO rollouts for testing early routing commitment.

Each row fixes the first control step's flow noise.  Since the initial
observation is also fixed, every episode in a row must have the same complete
control-step-0 routing trajectory.  Each column supplies a common, independent
noise stream for all later control steps.  The outcome matrix therefore asks a
direct question: does one early route keep the same outcome when only future
policy noise changes?

The script is resumable.  Completed grid cells are read from summaries.json,
and --max-new-episodes can be used for a small end-to-end validation before
continuing the same server capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _observation_sha256(observation: dict) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = observation[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, str):
            digest.update(value.encode("utf-8"))
        else:
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _blocked_grid_plan(n_first: int, n_future: int) -> list[tuple[int, int]]:
    """Visit 2x2 blocks first so a four-episode pilot checks both grid axes."""
    plan = []
    for row0 in range(0, n_first, 2):
        for col0 in range(0, n_future, 2):
            for row in range(row0, min(row0 + 2, n_first)):
                for col in range(col0, min(col0 + 2, n_future)):
                    plan.append((row, col))
    if len(plan) != n_first * n_future or len(set(plan)) != len(plan):
        raise AssertionError("grid planner did not produce every cell exactly once")
    return plan


def run_one(
    config: EpisodeConfig,
    client: PolicyClient,
    episode_index: int,
    first_noise_seed: int,
    future_noise_seed: int,
):
    """Run one cell and return its summary plus auditable per-control-step arrays."""
    environment, observation, task, prompt = _load_task(config)
    first_rng = np.random.default_rng(first_noise_seed)
    future_rng = np.random.default_rng(future_noise_seed)
    first_noise = first_rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)

    states = []
    action_chunks = []
    flow_noises = []
    observation_hash = None
    steps = 0
    success = False

    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        inference_index = 0
        while steps < config.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            if observation_hash is None:
                observation_hash = _observation_sha256(policy_observation)

            if inference_index == 0:
                flow_noise = first_noise
            else:
                flow_noise = future_rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)

            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = flow_noise
            # serve_with_recorder.py consumes this before validating the model input.
            request["episode_id"] = episode_index
            response = validate_action_response(client.infer(request))

            expected_sha = _array_sha256(flow_noise)
            if response.get(FLOW_NOISE_SHA256_KEY) != expected_sha:
                raise RuntimeError(
                    "server did not acknowledge the exact flow noise for inference %d"
                    % inference_index
                )

            actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
            flow_noises.append(np.array(flow_noise, copy=True))
            states.append(np.asarray(policy_observation["observation/state"], dtype=np.float32))
            action_chunks.append(actions[: config.replan_steps].copy())

            for action in actions[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _reward, _done, _info = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
            inference_index += 1
    finally:
        try:
            environment.close()
        except BaseException:
            pass

    summary = {
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "environment_seed": config.seed,
        "first_noise_seed": first_noise_seed,
        "future_noise_seed": future_noise_seed,
        # Compatibility with readers that expect the old single-seed field.
        "flow_noise_seed": first_noise_seed,
        "task_name": str(task.name),
        "prompt": prompt,
        "success": success,
        "action_steps": steps,
        "inference_calls": len(flow_noises),
        "initial_observation_sha256": observation_hash,
        "first_noise_sha256": _array_sha256(first_noise),
    }
    arrays = {
        "flow_noises": np.stack(flow_noises),
        "states": np.stack(states),
        "action_chunks": np.stack(action_chunks),
    }
    return summary, arrays


def _experiment_config(args) -> dict:
    return {
        "protocol": "early-route-commitment-grid-v1",
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "environment_seed": args.seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "n_first": args.n_first,
        "n_future": args.n_future,
        "first_seed_base": args.first_seed_base,
        "future_seed_base": args.future_seed_base,
        "label": args.label,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--benchmark", default="libero_goal")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-state-id", type=int, default=24)
    ap.add_argument("--n-first", type=int, default=16)
    ap.add_argument("--n-future", type=int, default=8)
    ap.add_argument("--first-seed-base", type=int, default=2000)
    ap.add_argument("--future-seed-base", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", default="commitment-grid-s24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    ap.add_argument(
        "--max-new-episodes",
        type=int,
        default=0,
        help="Stop after this many new cells; zero completes the grid.",
    )
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.n_first <= 0 or args.n_future <= 0:
        raise ValueError("n-first and n-future must be positive")
    if not 1 <= args.replan_steps <= FLOW_NOISE_SHAPE[0]:
        raise ValueError("replan-steps must be between 1 and %d" % FLOW_NOISE_SHAPE[0])

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summaries_path = out / "summaries.json"
    config_path = out / "experiment_config.json"
    expected_config = _experiment_config(args)

    if config_path.exists():
        actual_config = json.loads(config_path.read_text())
        if actual_config != expected_config:
            raise RuntimeError("existing experiment_config.json does not match this command")
    else:
        config_path.write_text(json.dumps(expected_config, indent=2, sort_keys=True))

    if summaries_path.exists():
        if not args.resume:
            raise RuntimeError("summaries.json exists; pass --resume to continue")
        summaries = json.loads(summaries_path.read_text())
    else:
        summaries = []

    cells = _blocked_grid_plan(args.n_first, args.n_future)
    plan = [
        {
            "grid_row": row,
            "grid_col": col,
            "first_noise_seed": args.first_seed_base + row,
            "future_noise_seed": args.future_seed_base + col,
        }
        for row, col in cells
    ]
    if len(summaries) > len(plan):
        raise RuntimeError("summaries.json has more episodes than the configured grid")
    for index, old in enumerate(summaries):
        expected = plan[index]
        for key, value in expected.items():
            if int(old[key]) != value:
                raise RuntimeError("resume plan mismatch at episode %d for %s" % (index, key))

    new_limit = args.max_new_episodes if args.max_new_episodes > 0 else len(plan)
    completed_at_start = len(summaries)

    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        metadata = client.metadata
        validate_policy_suite(metadata, args.benchmark)
        if metadata.get("route_recorder") != "himoe_router_recorder":
            raise RuntimeError("this experiment requires serve_with_recorder.py")

        metadata_path = out / "server_metadata.json"
        if metadata_path.exists():
            previous = json.loads(metadata_path.read_text())
            if previous.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
                raise RuntimeError("server checkpoint changed during resume")
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

        for episode_index in range(len(summaries), len(plan)):
            if episode_index - completed_at_start >= new_limit:
                break
            cell = plan[episode_index]
            episode_config = EpisodeConfig(
                task_suite=args.benchmark,
                task_id=args.task_id,
                init_state_id=args.init_state_id,
                seed=args.seed,
                host=args.host,
                port=args.port,
                libero_root=args.libero_root,
                output_root=str(out),
                settle_steps=args.settle_steps,
                max_steps=args.max_steps,
                replan_steps=args.replan_steps,
                inference_timeout=args.inference_timeout,
            )

            started = time.time()
            summary, arrays = run_one(
                episode_config,
                client,
                episode_index=episode_index,
                first_noise_seed=cell["first_noise_seed"],
                future_noise_seed=cell["future_noise_seed"],
            )
            summary.update(cell)
            summary["episode_index"] = episode_index
            summary["label"] = args.label
            summary["wall_s"] = round(time.time() - started, 2)

            np.savez_compressed(out / ("episode_%04d.npz" % episode_index), **arrays)
            summaries.append(summary)
            summaries_path.write_text(json.dumps(summaries, indent=2))

            print(
                "cell %3d/%3d row=%2d col=%2d first=%d future=%d "
                "success=%-5s actions=%3d controls=%2d %.1fs"
                % (
                    episode_index + 1,
                    len(plan),
                    cell["grid_row"],
                    cell["grid_col"],
                    cell["first_noise_seed"],
                    cell["future_noise_seed"],
                    summary["success"],
                    summary["action_steps"],
                    summary["inference_calls"],
                    summary["wall_s"],
                ),
                flush=True,
            )

    print("completed %d/%d grid cells" % (len(summaries), len(plan)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
