#!/usr/bin/env python3
"""Exercise the actual Top-K execution trace on an already-loaded server."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
for source in (
    ROOT / "analysis_moe_execution_signals",
    pathlib.Path("/home/jovyan/work/himoe-libero-wrist-fix/src"),
    pathlib.Path(
        "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA/packages/openpi-client/src"
    ),
):
    sys.path.insert(0, str(source))

from execution_features import (  # noqa: E402
    normalized_entropy,
    sparse_hellinger_distance,
    support_jaccard_distance,
)
from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    ROUTING_EXPERT_WEIGHTS_KEY,
    ROUTING_LAYER_INDICES_KEY,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    rng = np.random.default_rng(args.seed)
    noise = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    request = {
        IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        STATE_KEY: (0.1 * rng.standard_normal(STATE_DIM)).astype(np.float32),
        PROMPT_KEY: args.prompt,
        FLOW_NOISE_KEY: noise,
    }
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=10.0,
        inference_timeout=args.timeout,
    ) as client:
        metadata = dict(client.metadata)
        started = time.perf_counter()
        plain = client.infer(request)
        plain_seconds = time.perf_counter() - started
        started = time.perf_counter()
        captured = client.infer({**request, ROUTING_CAPTURE_KEY: True})
        capture_seconds = time.perf_counter() - started

    action_plain = np.asarray(plain[ACTION_KEY], np.float32)
    action_captured = np.asarray(captured[ACTION_KEY], np.float32)
    ids = np.asarray(captured[ROUTING_EXPERT_IDS_KEY], np.uint8)
    weights = np.asarray(captured[ROUTING_EXPERT_WEIGHTS_KEY], np.float32)
    layers = np.asarray(captured[ROUTING_LAYER_INDICES_KEY], np.int16)
    support_flow = support_jaccard_distance(ids[1:], ids[:-1])
    execution_flow = sparse_hellinger_distance(
        ids[1:], weights[1:], ids[:-1], weights[:-1]
    )
    expected_noise_digest = _digest(noise)
    bitwise = bool(np.array_equal(action_plain, action_captured))
    result = {
        "schema": "himoe.live_dispatch.audit.v1",
        "passed": bool(
            bitwise
            and captured.get(FLOW_NOISE_SHA256_KEY) == expected_noise_digest
            and ids.shape == (10, 8, 10, 4)
            and weights.shape == ids.shape
            and layers.shape == (8,)
        ),
        "endpoint": "ws://%s:%d" % (args.host, args.port),
        "bundle": {
            key: metadata.get(key)
            for key in (
                "bundle_schema",
                "bundle_model",
                "bundle_physical_gpu",
                "bundle_logical_gpu",
                "checkpoint_sha256",
            )
        },
        "seed": int(args.seed),
        "checks": {
            "action_bitwise_equal": bitwise,
            "max_abs_action_difference": float(
                np.max(np.abs(action_plain - action_captured))
            ),
            "flow_noise_ack": captured.get(FLOW_NOISE_SHA256_KEY)
            == expected_noise_digest,
            "route_shape": list(ids.shape),
            "layer_indices": layers.tolist(),
            "weight_sum_max_error": float(
                np.abs(weights.sum(axis=-1) - 1.0).max()
            ),
        },
        "timing_seconds": {
            "capture_off": plain_seconds,
            "capture_on": capture_seconds,
        },
        "execution_summary": {
            "adjacent_flow_support_jaccard_mean": float(support_flow.mean()),
            "late_flow_support_jaccard_mean": float(support_flow[6:].mean()),
            "adjacent_flow_sparse_hellinger_mean": float(execution_flow.mean()),
            "late_flow_sparse_hellinger_mean": float(execution_flow[6:].mean()),
            "final_flow_execution_entropy_mean": float(
                normalized_entropy(weights[-1]).mean()
            ),
            "unique_support_experts": int(np.unique(ids).size),
        },
        "action_sha256": {
            "capture_off": _digest(action_plain),
            "capture_on": _digest(action_captured),
        },
        "scope_note": (
            "The live protocol returns action-token Top-K IDs/weights only; "
            "state token, full probabilities, logits, and functional outputs are absent."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        expert_ids=ids,
        expert_weights=weights.astype(np.float16),
        layer_indices=layers,
        actions=action_captured,
    )
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8803)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--prompt", default="put the red mug on the plate")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
