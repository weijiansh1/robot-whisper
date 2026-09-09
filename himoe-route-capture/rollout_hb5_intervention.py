"""Collect paired HB5/d0 baseline/drop/random arms at one fixed observation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from himoe_hb5_intervention import ARMS
from himoe_intervention_protocol import (
    ARM_KEY,
    CANDIDATE_KEY,
    DRAW_KEY,
    PAIR_KEY,
    QUERY_KEY,
)


def build_paired_requests(
    observation: dict[str, Any],
    flow_noise: np.ndarray,
    *,
    pair_id: int,
    draw_id: int,
) -> list[dict[str, Any]]:
    """Build a triad with byte-identical observation arrays and explicit x0."""
    noise = np.ascontiguousarray(flow_noise, dtype=np.float32)
    if noise.shape != (10, 24):
        raise ValueError("flow_noise must have shape (10, 24)")
    if pair_id < 0 or draw_id < 0:
        raise ValueError("pair_id and draw_id must be non-negative")
    requests = []
    for arm in ARMS:
        request = dict(observation)
        # A separate array prevents an accidental in-place mutation in one arm
        # from contaminating another while preserving exact bytes.
        request["flow/noise"] = noise.copy()
        request[PAIR_KEY] = int(pair_id)
        request[DRAW_KEY] = int(draw_id)
        request[ARM_KEY] = arm
        request[QUERY_KEY] = int(pair_id)
        request[CANDIDATE_KEY] = int(draw_id)
        requests.append(request)
    return requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--n-draws", type=int, default=1)
    parser.add_argument("--noise-seed", type=int, default=8100)
    parser.add_argument("--pair-id", type=int, default=0)
    parser.add_argument("--draw-base", type=int, default=0)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--inference-timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.n_draws <= 0:
        raise ValueError("--n-draws must be positive")

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
        FLOW_NOISE_SHA256_KEY,
        validate_action_response,
    )

    out = Path(args.out)
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
        max_steps=1,
        replan_steps=1,
        inference_timeout=args.inference_timeout,
    )
    environment, observation, _task, prompt = _load_task(config)
    started = time.time()
    try:
        for _ in range(args.settle_steps):
            observation, _, _, _ = environment.step(LIBERO_DUMMY_ACTION.tolist())
        fixed_observation = build_policy_observation(observation, prompt)
        rng = np.random.default_rng(args.noise_seed)
        actions = np.empty((args.n_draws, len(ARMS), 10, 7), dtype=np.float32)
        noise_digests: list[str] = []
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            metadata = client.metadata
            if metadata.get("hb5_intervention_recorder") != (
                "himoe_hb5_d0_intervention_v4"
            ):
                raise RuntimeError("server is not the paired HB5/d0 v4 server")
            if metadata.get("hb5_intervention_drop_policy") != "categorical_weighted":
                raise RuntimeError("the frozen smoke requires categorical_weighted")
            validate_policy_suite(metadata, args.benchmark)
            (out / "server_metadata.json").write_text(
                json.dumps(metadata, indent=2, default=str)
            )
            for draw_offset in range(args.n_draws):
                draw_id = args.draw_base + draw_offset
                noise = rng.standard_normal((10, 24)).astype(np.float32)
                expected_digest = hashlib.sha256(noise.tobytes()).hexdigest()
                noise_digests.append(expected_digest)
                requests = build_paired_requests(
                    fixed_observation,
                    noise,
                    pair_id=args.pair_id,
                    draw_id=draw_id,
                )
                for arm_index, (arm, request) in enumerate(zip(ARMS, requests)):
                    response = validate_action_response(client.infer(request))
                    if response.get(FLOW_NOISE_SHA256_KEY) != expected_digest:
                        raise RuntimeError(
                            "server did not acknowledge the exact explicit x0 for %s"
                            % arm
                        )
                    actions[draw_offset, arm_index] = response[ACTION_KEY]
                    print(
                        "pair=%d draw=%d arm=%s done" % (args.pair_id, draw_id, arm),
                        flush=True,
                    )
    finally:
        try:
            environment.close()
        except BaseException:
            pass

    np.savez_compressed(
        out / "paired_actions.npz",
        actions=actions,
        arm=np.asarray(ARMS),
        pair_id=np.full(args.n_draws, args.pair_id, dtype=np.int64),
        draw_id=np.arange(
            args.draw_base, args.draw_base + args.n_draws, dtype=np.int64
        ),
        flow_noise_sha256=np.asarray(noise_digests),
    )
    config_payload = {
        "protocol": "himoe-hb5-d0-intervention-v4",
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_id": args.init_state_id,
        "pair_id": args.pair_id,
        "draw_base": args.draw_base,
        "n_draws": args.n_draws,
        "arms": list(ARMS),
        "drop_policy": "categorical_weighted",
        "noise_seed": args.noise_seed,
        "explicit_x0_reused_across_arms": True,
        "elapsed_s": time.time() - started,
    }
    (out / "experiment_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True)
    )
    print(
        "captured %d paired triad(s) in %.1f s" % (args.n_draws, time.time() - started),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
