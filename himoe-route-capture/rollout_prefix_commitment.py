"""Paired rollouts for locating when an early trajectory becomes committed.

For a branch point k, every cell in a row replays the same k policy-noise
tensors from the same initial observation.  The complete state, action, and MoE
routing prefix is therefore fixed.  Independent future-noise streams begin at
control step k.  Repeating the grid at nested k values estimates a commitment
curve rather than testing only the first control step.
"""

from __future__ import annotations

import argparse
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
from rollout_commitment_grid import _array_sha256, _blocked_grid_plan, _observation_sha256


def _prefix_noises(seed: int, controls: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack(
        [rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32) for _ in range(controls)]
    )


def _prefix_grid_plan(
    branch_controls: list[int], n_prefix: int, n_future: int
) -> list[tuple[int, int, int]]:
    """Visit the same 2x2 row/column block at every branch point first."""
    branches = sorted(set(branch_controls))
    plan = []
    blocked = _blocked_grid_plan(n_prefix, n_future)
    for offset in range(0, len(blocked), 4):
        block = blocked[offset : offset + 4]
        for branch in branches:
            for row, col in block:
                plan.append((branch, row, col))
    expected = len(branches) * n_prefix * n_future
    if len(plan) != expected or len(set(plan)) != expected:
        raise AssertionError("prefix grid planner did not produce every cell exactly once")
    return plan


def run_one(
    config: EpisodeConfig,
    client: PolicyClient,
    episode_index: int,
    branch_controls: int,
    prefix_noise_seed: int,
    future_noise_seed: int,
):
    environment, observation, task, prompt = _load_task(config)
    prefix_noises = _prefix_noises(prefix_noise_seed, branch_controls)
    future_rng = np.random.default_rng(future_noise_seed)

    states = []
    action_chunks = []
    flow_noises = []
    observation_hash = None
    steps = 0
    success = False
    success_control_index = None

    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        inference_index = 0
        while steps < config.max_steps and not success:
            policy_observation = build_policy_observation(observation, prompt)
            if observation_hash is None:
                observation_hash = _observation_sha256(policy_observation)

            if inference_index < branch_controls:
                flow_noise = prefix_noises[inference_index]
            else:
                flow_noise = future_rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)

            request = dict(policy_observation)
            request[FLOW_NOISE_KEY] = flow_noise
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
                    success_control_index = inference_index
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
        "branch_controls": branch_controls,
        "prefix_noise_seed": prefix_noise_seed,
        "future_noise_seed": future_noise_seed,
        "flow_noise_seed": prefix_noise_seed,
        "task_name": str(task.name),
        "prompt": prompt,
        "success": success,
        "success_control_index": success_control_index,
        "success_before_branch": bool(
            success_control_index is not None and success_control_index < branch_controls
        ),
        "branch_reached": len(flow_noises) > branch_controls,
        "action_steps": steps,
        "inference_calls": len(flow_noises),
        "initial_observation_sha256": observation_hash,
        "prefix_noise_sha256": [_array_sha256(noise) for noise in prefix_noises],
    }
    arrays = {
        "flow_noises": np.stack(flow_noises),
        "states": np.stack(states),
        "action_chunks": np.stack(action_chunks),
    }
    return summary, arrays


def _experiment_config(args) -> dict:
    return {
        "protocol": "early-prefix-commitment-grid-v1",
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "environment_seed": args.seed,
        "settle_steps": args.settle_steps,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "branch_controls": sorted(set(args.branch_controls)),
        "n_prefix": args.n_prefix,
        "n_future": args.n_future,
        "prefix_seed_base": args.prefix_seed_base,
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
    ap.add_argument("--branch-controls", type=int, nargs="+", default=[4, 8, 10, 12, 14])
    ap.add_argument("--n-prefix", type=int, default=8)
    ap.add_argument("--n-future", type=int, default=8)
    ap.add_argument("--prefix-seed-base", type=int, default=2000)
    ap.add_argument("--future-seed-base", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--settle-steps", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--replan-steps", type=int, default=10)
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--label", default="prefix-commitment-s24")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inference-timeout", type=float, default=300.0)
    ap.add_argument("--max-new-episodes", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    branches = sorted(set(args.branch_controls))
    if not branches or branches[0] <= 0:
        raise ValueError("branch-controls must contain positive integers")
    if branches[-1] * args.replan_steps >= args.max_steps:
        raise ValueError("latest branch must occur before the rollout horizon")
    if args.n_prefix <= 0 or args.n_future <= 0:
        raise ValueError("n-prefix and n-future must be positive")
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

    cells = _prefix_grid_plan(branches, args.n_prefix, args.n_future)
    plan = [
        {
            "branch_controls": branch,
            "grid_row": row,
            "grid_col": col,
            "prefix_noise_seed": args.prefix_seed_base + row,
            "future_noise_seed": args.future_seed_base + col,
        }
        for branch, row, col in cells
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
                branch_controls=cell["branch_controls"],
                prefix_noise_seed=cell["prefix_noise_seed"],
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
                "cell %3d/%3d k=%2d row=%d col=%d prefix=%d future=%d "
                "success=%-5s actions=%3d controls=%2d %.1fs"
                % (
                    episode_index + 1,
                    len(plan),
                    cell["branch_controls"],
                    cell["grid_row"],
                    cell["grid_col"],
                    cell["prefix_noise_seed"],
                    cell["future_noise_seed"],
                    summary["success"],
                    summary["action_steps"],
                    summary["inference_calls"],
                    summary["wall_s"],
                ),
                flush=True,
            )

    print("completed %d/%d prefix-grid cells" % (len(summaries), len(plan)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
