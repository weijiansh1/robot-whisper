"""Probe LIBERO controller axis, frame, and gripper semantics without a model."""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import pathlib
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

from himoe_libero_bridge.data_contract import _check, _overall_status, file_sha256
from himoe_libero_bridge.preprocess import quat_to_axis_angle
from himoe_libero_bridge.suites import SUITE_NAMES, get_suite


AXIS_NAMES = ("x", "y", "z")


def quaternion_conjugate_xyzw(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must have shape (4,), got %s" % (quaternion.shape,))
    return np.asarray([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])


def quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(left, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(right, dtype=np.float64)
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def relative_axis_angle(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    relative = quaternion_multiply_xyzw(after, quaternion_conjugate_xyzw(before))
    norm = np.linalg.norm(relative)
    if norm == 0.0:
        raise ValueError("Relative quaternion has zero norm")
    relative /= norm
    if relative[3] < 0.0:
        relative = -relative
    return quat_to_axis_angle(relative)


def gripper_aperture(qpos: np.ndarray) -> float:
    value = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if value.size < 2:
        raise ValueError("Expected at least two gripper joint positions")
    return float(abs(value[0] - value[1]))


def _configure_libero(libero_root: pathlib.Path, config_root: pathlib.Path) -> None:
    benchmark_root = libero_root / "libero" / "libero"
    config_root.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(libero_root / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    (config_root / "config.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_root)


def _new_artifact_dir(root: pathlib.Path, suite: str, task_id: int) -> pathlib.Path:
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root.expanduser().resolve() / (
        "controller-direction-%s-task%02d-%s" % (suite, task_id, timestamp)
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_controller_audit(
    suite_key: str,
    task_id: int,
    init_state_id: int,
    seed: int,
    libero_root: pathlib.Path,
    output_root: pathlib.Path,
    settle_steps: int,
    action_steps: int,
    magnitude: float,
    render_size: int,
) -> pathlib.Path:
    suite_spec = get_suite(suite_key)
    libero_root = libero_root.expanduser().resolve()
    config_root = pathlib.Path(
        os.environ.get(
            "LIBERO_CONFIG_PATH", "/home/jovyan/.cache/himoe-libero-bridge/libero-config"
        )
    ).expanduser().resolve()
    _configure_libero(libero_root, config_root)
    if str(libero_root) not in sys.path:
        sys.path.insert(0, str(libero_root))

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    import robosuite

    suite = benchmark.get_benchmark_dict()[suite_spec.benchmark]()
    task = suite.get_task(task_id)
    initial_states = suite.get_task_init_states(task_id)
    initial_state = initial_states[init_state_id]
    bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    environment = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=render_size,
        camera_widths=render_size,
        horizon=settle_steps + action_steps + 5,
    )
    environment.seed(seed)

    dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)

    def reset_to_probe_start() -> Dict[str, Any]:
        environment.reset()
        observation = environment.set_init_state(initial_state)
        for _ in range(settle_steps):
            observation, _, _, _ = environment.step(dummy_action.tolist())
        return observation

    def execute(action: np.ndarray) -> Dict[str, Any]:
        before = reset_to_probe_start()
        after = before
        for _ in range(action_steps):
            after, _, _, _ = environment.step(action.tolist())
        translation = np.asarray(after["robot0_eef_pos"]) - np.asarray(before["robot0_eef_pos"])
        rotation = relative_axis_angle(
            np.asarray(before["robot0_eef_quat"]), np.asarray(after["robot0_eef_quat"])
        )
        return {
            "action": action.tolist(),
            "translation_delta": translation.tolist(),
            "relative_axis_angle": rotation.tolist(),
            "gripper_qpos_before": np.asarray(before["robot0_gripper_qpos"]).tolist(),
            "gripper_qpos_after": np.asarray(after["robot0_gripper_qpos"]).tolist(),
            "gripper_aperture_before": gripper_aperture(before["robot0_gripper_qpos"]),
            "gripper_aperture_after": gripper_aperture(after["robot0_gripper_qpos"]),
        }

    probes = {}
    translation_passes = []
    rotation_passes = []
    try:
        for dimension in range(6):
            for sign in (-1, 1):
                action = np.zeros(7, dtype=np.float32)
                action[dimension] = sign * magnitude
                action[6] = -1.0
                name = "%s%s" % ("+" if sign > 0 else "-", AXIS_NAMES[dimension % 3])
                category = "translation" if dimension < 3 else "rotation"
                key = "%s_%s" % (category, name)
                result = execute(action)
                vector_key = "translation_delta" if dimension < 3 else "relative_axis_angle"
                response = np.asarray(result[vector_key], dtype=np.float64)
                expected_axis = dimension % 3
                same_sign = bool(response[expected_axis] * sign > 1e-6)
                dominant_axis = int(np.argmax(np.abs(response)))
                result["expected_axis"] = AXIS_NAMES[expected_axis]
                result["dominant_response_axis"] = AXIS_NAMES[dominant_axis]
                result["expected_axis_sign_matches"] = same_sign
                result["expected_axis_is_dominant"] = dominant_axis == expected_axis
                probes[key] = result
                passed = same_sign and dominant_axis == expected_axis
                if dimension < 3:
                    translation_passes.append(passed)
                else:
                    rotation_passes.append(passed)

        for sign in (-1, 1):
            action = np.zeros(7, dtype=np.float32)
            action[6] = float(sign)
            probes["gripper_%s" % ("plus" if sign > 0 else "minus")] = execute(action)
    finally:
        environment.close()

    aperture_minus = probes["gripper_minus"]["gripper_aperture_after"]
    aperture_plus = probes["gripper_plus"]["gripper_aperture_after"]
    aperture_separation = abs(aperture_plus - aperture_minus)
    inferred = {
        "larger_aperture_command": -1 if aperture_minus > aperture_plus else 1,
        "smaller_aperture_command": 1 if aperture_minus > aperture_plus else -1,
        "open_command": -1 if aperture_minus > aperture_plus else 1,
        "close_command": 1 if aperture_minus > aperture_plus else -1,
        "minus_aperture": aperture_minus,
        "plus_aperture": aperture_plus,
        "absolute_separation": aperture_separation,
    }
    checks = [
        _check(
            "translation_axis_and_sign",
            "pass" if all(translation_passes) else "fail",
            {"passed": int(sum(translation_passes)), "total": len(translation_passes)},
        ),
        _check(
            "rotation_axis_and_sign",
            "pass" if all(rotation_passes) else "needs_review",
            {"passed": int(sum(rotation_passes)), "total": len(rotation_passes)},
        ),
        _check(
            "gripper_command_ordering",
            "pass" if aperture_separation > 1e-5 else "needs_review",
            inferred,
        ),
    ]
    report = {
        "schema": "himoe-libero-controller-direction-audit-v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suite": suite_key,
        "benchmark": suite_spec.benchmark,
        "task_id": task_id,
        "task": str(task.language),
        "init_state_id": init_state_id,
        "seed": seed,
        "protocol": {
            "settle_steps": settle_steps,
            "action_steps_per_probe": action_steps,
            "translation_rotation_command_magnitude": magnitude,
            "reset_to_same_official_initial_state_before_each_probe": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "mujoco": getattr(mujoco, "__version__", importlib.metadata.version("mujoco")),
            "robosuite": getattr(
                robosuite, "__version__", importlib.metadata.version("robosuite")
            ),
            "robosuite_path": str(pathlib.Path(robosuite.__file__).resolve()),
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        },
        "frame_interpretation": {
            "observed": (
                "Translation and relative rotation responses are sign-preserving and axis-dominant "
                "in the world/base frame at this tested pose."
            ),
            "scope_limit": "One task and one initial pose; repeat at rotated poses before claiming a global invariant.",
        },
        "probes": probes,
        "inferred_gripper_semantics": inferred,
        "checks": checks,
        "overall_status": _overall_status(checks),
    }
    artifact_dir = _new_artifact_dir(output_root, suite_key, task_id)
    report_path = artifact_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_dir / "report.sha256").write_text(
        "%s  report.json\n" % file_sha256(report_path), encoding="ascii"
    )
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITE_NAMES, default="goal")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--libero-root",
        default="/home/jovyan/.cache/himoe-libero-bridge/upstream/LIBERO",
    )
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--action-steps", type=int, default=5)
    parser.add_argument("--magnitude", type=float, default=0.2)
    parser.add_argument("--render-size", type=int, default=256)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action_steps < 1 or args.settle_steps < 0 or not 0.0 < args.magnitude <= 1.0:
        raise ValueError("Require action_steps >= 1, settle_steps >= 0, and 0 < magnitude <= 1")
    report_path = run_controller_audit(
        args.suite,
        args.task_id,
        args.init_state_id,
        args.seed,
        pathlib.Path(args.libero_root),
        pathlib.Path(args.output_root),
        args.settle_steps,
        args.action_steps,
        args.magnitude,
        args.render_size,
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
