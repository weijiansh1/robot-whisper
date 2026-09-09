"""Paired online LIBERO rollout for the route-only denoising stopper."""

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
from himoe_libero_bridge.protocol import ACTION_KEY, FLOW_NOISE_KEY, FLOW_NOISE_SHAPE, validate_action_response

from serve_adaptive_denoise import (
    ARM_KEY,
    CHANGES_KEY,
    ENABLED_KEY,
    EPISODE_KEY,
    FIRST_ELIGIBLE_KEY,
    SAMPLE_MS_KEY,
    STEPS_KEY,
    STOPPED_KEY,
)


def run_arm(config, client, arm: str, enabled: bool, flow_noise_seed: int, episode_id: int):
    environment, observation, task, prompt = _load_task(config)
    rng = np.random.default_rng(flow_noise_seed)
    steps = 0
    success = False
    queries = []
    chunks = []
    try:
        for _ in range(config.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())

        while steps < config.max_steps and not success:
            request = dict(build_policy_observation(observation, prompt))
            request[FLOW_NOISE_KEY] = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
            request[ENABLED_KEY] = bool(enabled)
            request[ARM_KEY] = arm
            request[EPISODE_KEY] = episode_id
            response = validate_action_response(client.infer(request))
            actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
            chunks.append(actions)
            queries.append(
                {
                    "arm": arm,
                    "flow_noise_seed": flow_noise_seed,
                    "query": len(queries),
                    "action_steps_before": steps,
                    "denoise_steps": int(response[STEPS_KEY]),
                    "stopped_early": bool(response[STOPPED_KEY]),
                    "first_eligible_step": int(response[FIRST_ELIGIBLE_KEY]),
                    "route_changes": np.asarray(response[CHANGES_KEY], dtype=np.float32).tolist(),
                    "sample_ms": float(response[SAMPLE_MS_KEY]),
                }
            )
            for action in actions[: config.replan_steps]:
                if steps >= config.max_steps:
                    break
                observation, _, _, _ = environment.step(action.tolist())
                steps += 1
                success = bool(environment.check_success())
                if success:
                    break
    finally:
        try:
            environment.close()
        except BaseException:
            pass

    denoise = np.asarray([row["denoise_steps"] for row in queries], dtype=np.int64)
    summary = {
        "arm": arm,
        "adaptive_enabled": enabled,
        "task_name": str(task.name),
        "prompt": prompt,
        "task_id": config.task_id,
        "init_state_id": config.init_state_id,
        "flow_noise_seed": flow_noise_seed,
        "success": success,
        "action_steps": steps,
        "policy_queries": len(queries),
        "mean_denoise_steps": float(denoise.mean()),
        "early_stop_queries": int(np.sum(denoise < 10)),
        "denoise_pass_saving_fraction": float(1.0 - denoise.sum() / (10.0 * len(denoise))),
        "mean_sample_ms": float(np.mean([row["sample_ms"] for row in queries])),
    }
    return summary, queries, np.stack(chunks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=24)
    parser.add_argument("--flow-noise-seed", type=int, default=2000)
    parser.add_argument(
        "--flow-noise-seeds",
        help="Comma-separated paired seeds; overrides --flow-noise-seed",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--arms",
        default="baseline,adaptive",
        help="Comma-separated subset/order of baseline,adaptive",
    )
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()

    arm_names = [value.strip() for value in args.arms.split(",") if value.strip()]
    if not arm_names or any(value not in ("baseline", "adaptive") for value in arm_names):
        raise ValueError("--arms must be a comma-separated subset of baseline,adaptive")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("--arms contains a duplicate")
    if args.flow_noise_seeds:
        flow_noise_seeds = [
            int(value.strip())
            for value in args.flow_noise_seeds.split(",")
            if value.strip()
        ]
    else:
        flow_noise_seeds = [args.flow_noise_seed]
    if not flow_noise_seeds or len(set(flow_noise_seeds)) != len(flow_noise_seeds):
        raise ValueError("flow-noise seeds must be a non-empty unique list")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = EpisodeConfig(
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

    results = []
    all_queries = []
    arrays = {}
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        metadata = client.metadata
        validate_policy_suite(metadata, args.benchmark)
        if not metadata.get("adaptive_denoise_supported"):
            raise RuntimeError("server does not advertise adaptive denoising")
        (out / "server_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        episode_id = 0
        for pair_index, flow_noise_seed in enumerate(flow_noise_seeds):
            # Alternate arm order to avoid assigning all temporal drift to one arm.
            pair_arms = arm_names if pair_index % 2 == 0 else list(reversed(arm_names))
            for arm in pair_arms:
                enabled = arm == "adaptive"
                started = time.time()
                summary, queries, chunks = run_arm(
                    config, client, arm, enabled, flow_noise_seed, episode_id
                )
                summary["pair_index"] = pair_index
                summary["wall_s"] = time.time() - started
                for query in queries:
                    query["pair_index"] = pair_index
                results.append(summary)
                all_queries.extend(queries)
                arrays["seed%d_%s_actions" % (flow_noise_seed, arm)] = chunks
                print(
                    "seed=%d %s: success=%s actions=%d queries=%d "
                    "mean_rounds=%.2f early=%d saving=%.1f%%"
                    % (
                        flow_noise_seed,
                        arm,
                        summary["success"],
                        summary["action_steps"],
                        summary["policy_queries"],
                        summary["mean_denoise_steps"],
                        summary["early_stop_queries"],
                        100.0 * summary["denoise_pass_saving_fraction"],
                    ),
                    flush=True,
                )
                (out / "results.json").write_text(json.dumps(results, indent=2))
                (out / "queries.json").write_text(json.dumps(all_queries, indent=2))
                np.savez_compressed(out / "action_chunks.npz", **arrays)
                episode_id += 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
