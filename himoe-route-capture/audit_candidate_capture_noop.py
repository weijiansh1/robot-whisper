"""GPU A/B/C audit that candidate instrumentation is numerically inert.

One fixed observation and one fixed flow-noise tensor are evaluated in a single
loaded policy under three progressively instrumented arms:

* A: no capture hooks;
* B: normalized flow trajectory only;
* C: full router probabilities, router inputs, and flow trajectory.

The formal gate is bitwise equality of all three returned action chunks and of
the B/C flow trajectories.  This is a deployment-GPU audit; CPU results cannot
substitute for it because BF16 router ties are kernel dependent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, "/home/jovyan/work/himoe-libero-wrist-fix/src")

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


def _digest(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _infer(policy: Any, request: dict[str, Any]) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    response = validate_action_response(policy.infer(dict(request)))
    elapsed = time.perf_counter() - started
    return np.asarray(response[ACTION_KEY], dtype=np.float32), elapsed


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from himoe_libero_bridge.server import create_policy
    from himoe_router_recorder import HiMoERouteRecorder
    from serve_flow_trace import FlowTracer

    policy = create_policy(
        "himoe",
        args.checkpoint_dir,
        args.upstream_root,
        args.gpu,
        args.suite,
        args.libero_wrist_layout,
    )
    import torch

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    rng = np.random.default_rng(args.seed)
    request = {
        IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        WRIST_IMAGE_KEY: rng.integers(0, 256, IMAGE_SHAPE, dtype=np.uint8),
        STATE_KEY: (0.1 * rng.standard_normal(STATE_DIM)).astype(np.float32),
        PROMPT_KEY: args.prompt,
        FLOW_NOISE_KEY: rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32),
    }

    action_a, time_a = _infer(policy, request)

    core = policy._policy.model
    tracer = FlowTracer(core)
    tracer.begin()
    action_b, time_b = _infer(policy, request)
    trajectory_b, residual_b = tracer.finish()

    recorder = HiMoERouteRecorder(
        core,
        store_full_probs=True,
        store_hidden=True,
        verify_steps=1,
    ).attach()
    recorder.begin_control_step(episode_id=0, control_step=0)
    tracer.begin()
    action_c, time_c = _infer(policy, request)
    record = recorder.end_control_step()
    trajectory_c, residual_c = tracer.finish()

    actions_bitwise = bool(
        np.array_equal(action_a, action_b) and np.array_equal(action_a, action_c)
    )
    trajectories_bitwise = bool(np.array_equal(trajectory_b, trajectory_c))
    route_shape = list(np.asarray(record.hb_router_probs).shape)
    hidden_shape = list(np.asarray(record.hb_hidden).shape)
    expected_route = [1, 8, 10, 11, 32]
    route_complete = route_shape == expected_route and hidden_shape[:4] == expected_route[:4]
    result = {
        "schema": "himoe.candidate_capture.noop_audit.v1",
        "implementation_sha256": hashlib.sha256(
            pathlib.Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "passed": bool(actions_bitwise and trajectories_bitwise and route_complete),
        "deployment_gpu_required": True,
        "gpu": str(args.gpu),
        "gpu_runtime": {
            "visible_device": int(device),
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "torch_cuda_version": torch.version.cuda,
        },
        "checkpoint_sha256": policy.metadata.get("checkpoint_sha256"),
        "libero_wrist_layout": policy.metadata.get("libero_wrist_layout"),
        "seed": int(args.seed),
        "arms": {
            "A_capture_off": {"action_sha256": _digest(action_a), "seconds": time_a},
            "B_flow_only": {
                "action_sha256": _digest(action_b),
                "flow_sha256": _digest(trajectory_b),
                "euler_residual": float(residual_b),
                "seconds": time_b,
            },
            "C_full": {
                "action_sha256": _digest(action_c),
                "flow_sha256": _digest(trajectory_c),
                "euler_residual": float(residual_c),
                "route_shape": route_shape,
                "hidden_shape": hidden_shape,
                "seconds": time_c,
            },
        },
        "checks": {
            "actions_bitwise_equal": actions_bitwise,
            "flow_trajectories_bitwise_equal": trajectories_bitwise,
            "full_route_and_hidden_shapes": bool(route_complete),
            "max_abs_action_a_b": float(np.max(np.abs(action_a - action_b))),
            "max_abs_action_a_c": float(np.max(np.abs(action_a - action_c))),
            "max_abs_flow_b_c": float(np.max(np.abs(trajectory_b - trajectory_c))),
        },
    }
    recorder.close()
    tracer.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--suite", default="goal")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--libero-wrist-layout", default="released-left")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--prompt", default="open the middle drawer of the cabinet")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], indent=2, sort_keys=True))
    print("passed=%s; wrote %s" % (result["passed"], args.out))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
