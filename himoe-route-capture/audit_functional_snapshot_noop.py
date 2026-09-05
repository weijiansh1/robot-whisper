#!/usr/bin/env python3
"""GPU no-op audit for request-gated HB functional snapshots.

One policy is loaded once and evaluated with one fixed observation and flow
noise under three arms: no recorder, recorder attached but disabled, and one
enabled rich snapshot.  Action equality is the hard transparency gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIDGE_SRC = pathlib.Path("/home/jovyan/work/himoe-libero-wrist-fix/src")
UPSTREAM = pathlib.Path(
    "/home/jovyan/.cache/himoe-libero-bridge/upstream/HiMoE-VLA"
)
CHECKPOINT_ROOT = pathlib.Path(
    "/home/jovyan/.cache/himoe-libero-bridge/checkpoints"
)
for source in (ROOT / "online-servers", ROOT / "himoe-route-capture", BRIDGE_SRC):
    sys.path.insert(0, str(source))

from himoe_functional_recorder import (  # noqa: E402
    HBFunctionalSnapshotRecorder,
    save_functional_record,
)
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    IMAGE_SHAPE,
    PROMPT_KEY,
    STATE_DIM,
    STATE_KEY,
    WRIST_IMAGE_KEY,
    validate_action_response,
)
from himoe_libero_bridge.suites import get_suite  # noqa: E402
from himoe_libero_bridge.policies import HiMoEPolicy  # noqa: E402
from himoe_low_memory_load import low_memory_himoe_load  # noqa: E402


CHECKPOINTS = {
    "goal": "HiMoE-VLA-Libero-Goal",
    "spatial": "HiMoE-VLA-Libero-Spatial",
    "object": "HiMoE-VLA-Libero-Object",
    "long": "HiMoE-VLA-Libero-10",
}


def _sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _infer(policy: Any, request: dict[str, Any]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    response = validate_action_response(policy.infer(dict(request)))
    return np.asarray(response[ACTION_KEY], dtype=np.float32), time.perf_counter() - started


def _load_policy(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    checkpoint = args.checkpoint_root / CHECKPOINTS[args.suite]
    train_config = get_suite(args.suite).train_config
    with low_memory_himoe_load(
        checkpoint,
        args.upstream_root,
        train_config,
        target_device="cuda:0",
    ) as audit:
        policy = HiMoEPolicy(
            checkpoint_dir=str(checkpoint),
            suite=args.suite,
            upstream_root=str(args.upstream_root),
            require_cuda=True,
            libero_wrist_layout=args.libero_wrist_layout,
        )
    return policy, audit.as_metadata()


def run(
    args: argparse.Namespace,
    *,
    loaded_policy: Any | None = None,
    load_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("audit requires CUDA")
    if loaded_policy is None:
        if torch.cuda.device_count() != 1:
            raise RuntimeError("standalone audit requires one CUDA_VISIBLE_DEVICES entry")
        torch.cuda.set_device(0)
        policy, load_audit = _load_policy(args)
    else:
        policy = loaded_policy
        if load_audit is None:
            raise ValueError("load_audit is required with loaded_policy")
    rng = np.random.default_rng(args.seed)
    request = {
        IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        STATE_KEY: (0.1 * rng.standard_normal(STATE_DIM)).astype(np.float32),
        PROMPT_KEY: args.prompt,
        FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
    }

    action_a, seconds_a = _infer(policy, request)
    core = policy._policy.model
    recorder = HBFunctionalSnapshotRecorder(
        core,
        expected_denoise=10,
        sketch_dim=args.sketch_dim,
    ).attach()
    try:
        action_b, seconds_b = _infer(policy, request)
        recorder.begin(episode_id=0, control_step=0)
        action_c, seconds_c = _infer(policy, request)
        record = recorder.end()
    finally:
        recorder.close()

    save_functional_record(record, args.snapshot)
    expected = {
        "router_logits_centered": [1, 8, 10, 11, 32],
        "topk_idx": [1, 8, 10, 11, 4],
        "topk_exec_weight": [1, 8, 10, 11, 4],
        "expert_contrib_sketch": [1, 8, 10, 11, 4, args.sketch_dim],
    }
    shapes = {
        name: list(getattr(record, name).shape)
        for name in expected
    }
    finite = {
        name: bool(np.isfinite(value).all())
        for name, value in vars(record).items()
        if isinstance(value, np.ndarray)
    }
    actions_bitwise = bool(
        np.array_equal(action_a, action_b) and np.array_equal(action_a, action_c)
    )
    shapes_complete = all(shapes[name] == shape for name, shape in expected.items())
    weights_normalized = bool(
        np.allclose(record.topk_exec_weight.sum(axis=-1), 1.0, atol=2e-3)
    )
    bounded = bool(
        np.all((record.tail_mass >= 0.0) & (record.tail_mass <= 1.0))
        and np.all((record.routed_authority >= 0.0) & (record.routed_authority <= 1.0))
        and np.all((record.expert_cancellation >= 0.0) & (record.expert_cancellation <= 1.0))
        and np.all(
            (record.expert_disagreement_ratio >= 0.0)
            & (record.expert_disagreement_ratio <= 1.0)
        )
    )
    passed = bool(
        actions_bitwise
        and shapes_complete
        and weights_normalized
        and all(finite.values())
        and bounded
    )
    result = {
        "schema": "himoe.functional_snapshot.noop_audit.v1",
        "passed": passed,
        "physical_gpu": int(args.gpu),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "suite": args.suite,
        "checkpoint_sha256": policy.metadata.get("checkpoint_sha256"),
        "low_memory_load": load_audit,
        "model_device": str(next(policy._policy.model.parameters()).device),
        "seed": int(args.seed),
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_array_bytes": record.array_bytes,
        "snapshot_file_bytes": args.snapshot.stat().st_size,
        "timing_seconds": {
            "A_no_recorder": seconds_a,
            "B_attached_disabled": seconds_b,
            "C_enabled": seconds_c,
        },
        "action_sha256": {
            "A": _sha256(action_a),
            "B": _sha256(action_b),
            "C": _sha256(action_c),
        },
        "checks": {
            "actions_bitwise_equal": actions_bitwise,
            "max_abs_action_A_B": float(np.max(np.abs(action_a - action_b))),
            "max_abs_action_A_C": float(np.max(np.abs(action_a - action_c))),
            "expected_shapes": shapes_complete,
            "actual_shapes": shapes,
            "all_arrays_finite": bool(all(finite.values())),
            "topk_weights_normalized": weights_normalized,
            "bounded_ratios": bounded,
        },
        "ranges": {
            "top4_top5_logit_margin": [
                float(record.top4_top5_logit_margin.min()),
                float(record.top4_top5_logit_margin.max()),
            ],
            "tail_mass": [float(record.tail_mass.min()), float(record.tail_mass.max())],
            "routed_authority": [
                float(record.routed_authority.min()),
                float(record.routed_authority.max()),
            ],
            "expert_cancellation": [
                float(record.expert_cancellation.min()),
                float(record.expert_cancellation.max()),
            ],
            "expert_disagreement_ratio": [
                float(record.expert_disagreement_ratio.min()),
                float(record.expert_disagreement_ratio.max()),
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--suite", choices=tuple(CHECKPOINTS), default="long")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--sketch-dim", type=int, default=16)
    parser.add_argument("--prompt", default="put the red mug on the plate")
    parser.add_argument("--checkpoint-root", type=pathlib.Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--upstream-root", type=pathlib.Path, default=UPSTREAM)
    parser.add_argument("--libero-wrist-layout", default="checkpoint-right")
    parser.add_argument("--snapshot", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(args)
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    print("passed=%s; wrote %s" % (result["passed"], args.out))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
