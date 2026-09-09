#!/usr/bin/env python3
"""Restore frozen VLA_MUI_HUB states and render policy-ready RGB inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    FLOW_NOISE_SHAPE,
    IMAGE_KEY,
    PROMPT_KEY,
    STATE_KEY,
    WRIST_IMAGE_KEY,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "himoe.rich_event_inputs.v1"
POSITION_TOLERANCE_M = 0.002
ROTATION_TOLERANCE_RAD = 0.010
GRIPPER_TOLERANCE = 1e-4
SIM_STATE_TOLERANCE = 1e-9


def _sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _noise_at(seed: int, query: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = None
    for _ in range(query + 1):
        noise = rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    if noise is None:
        raise ValueError("query must be non-negative")
    return noise


def render(args: argparse.Namespace) -> dict:
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema") != "himoe.rich_event_capture_plan.v1":
        raise RuntimeError("unexpected capture-plan schema")
    rows = list(plan["rows"])
    if args.limit is not None:
        rows = rows[: args.limit]
    config = EpisodeConfig(
        libero_root=str(args.libero_root),
        output_root=str(args.out.parent),
        task_suite="libero_10",
        task_id=8,
        init_state_id=0,
        seed=7,
        settle_steps=10,
        max_steps=520,
        replan_steps=10,
        render_size=224,
    )
    environment, _observation, task, prompt = _load_task(config)
    if str(task.name) != "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove":
        raise RuntimeError("LIBERO task identity mismatch")
    if prompt != "put both moka pots on the stove":
        raise RuntimeError("LIBERO prompt identity mismatch")

    images = []
    wrist_images = []
    states = []
    noises = []
    expected_actions = []
    state_errors = []
    position_errors = []
    rotation_errors = []
    gripper_errors = []
    sim_errors = []
    row_metadata = []
    try:
        for expected_row_id, row in enumerate(rows):
            if int(row["row_id"]) != expected_row_id:
                raise RuntimeError("plan rows must form a contiguous prefix")
            episode_path = ROOT / row["episode_path"]
            query = int(row["query_index"])
            with np.load(episode_path, allow_pickle=False) as episode:
                sim_state = np.asarray(episode["sim_state"][query], np.float64)
                saved_state = np.asarray(episode["state"][query], np.float32)
                expected_action = np.asarray(episode["actions"][query], np.float32)
            observation = environment.regenerate_obs_from_state(sim_state)
            policy_input = build_policy_observation(observation, prompt)
            restored_state = np.asarray(policy_input[STATE_KEY], np.float32)
            restored_sim = np.asarray(environment.get_sim_state(), np.float64)
            state_error = float(np.max(np.abs(restored_state - saved_state)))
            position_error = float(np.max(np.abs(restored_state[:3] - saved_state[:3])))
            rotation_error = float(np.max(np.abs(restored_state[3:6] - saved_state[3:6])))
            gripper_error = float(np.max(np.abs(restored_state[6:] - saved_state[6:])))
            sim_error = float(np.max(np.abs(restored_sim - sim_state)))
            image = np.asarray(policy_input[IMAGE_KEY], np.uint8)
            wrist = np.asarray(policy_input[WRIST_IMAGE_KEY], np.uint8)
            if image.shape != (224, 224, 3) or wrist.shape != image.shape:
                raise RuntimeError("restored policy image shape mismatch")
            if np.ptp(image) == 0 or np.ptp(wrist) == 0:
                raise RuntimeError("restored policy image is blank")

            images.append(image)
            wrist_images.append(wrist)
            states.append(restored_state)
            noises.append(_noise_at(int(row["flow_noise_seed"]), query))
            expected_actions.append(expected_action)
            state_errors.append(state_error)
            position_errors.append(position_error)
            rotation_errors.append(rotation_error)
            gripper_errors.append(gripper_error)
            sim_errors.append(sim_error)
            row_metadata.append(
                {
                    **row,
                    "episode_npz_sha256": _sha256_file(episode_path),
                    "image_sha256": _sha256_bytes(image),
                    "wrist_image_sha256": _sha256_bytes(wrist),
                    "state_restore_max_abs_error": state_error,
                    "position_restore_max_abs_error_m": position_error,
                    "rotation_restore_max_abs_error_rad": rotation_error,
                    "gripper_restore_max_abs_error": gripper_error,
                    "sim_restore_max_abs_error": sim_error,
                }
            )
            if (expected_row_id + 1) % 16 == 0 or expected_row_id + 1 == len(rows):
                print("rendered %d/%d" % (expected_row_id + 1, len(rows)), flush=True)
    finally:
        environment.close()

    payload = {
        "image": np.stack(images).astype(np.uint8),
        "wrist_image": np.stack(wrist_images).astype(np.uint8),
        "state": np.stack(states).astype(np.float32),
        "flow_noise": np.stack(noises).astype(np.float32),
        "expected_actions": np.stack(expected_actions).astype(np.float32),
        "row_id": np.asarray([row["row_id"] for row in rows], np.int32),
        "pair_id": np.asarray([row["pair_id"] for row in rows], np.int16),
        "event": np.asarray([row["role"] == "event" for row in rows], bool),
        "relative_query": np.asarray([row["relative_query"] for row in rows], np.int8),
        "query_index": np.asarray([row["query_index"] for row in rows], np.int16),
        "global_episode": np.asarray([row["global_episode"] for row in rows], np.int16),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    passed = bool(
        max(position_errors) <= POSITION_TOLERANCE_M
        and max(rotation_errors) <= ROTATION_TOLERANCE_RAD
        and max(gripper_errors) <= GRIPPER_TOLERANCE
        and max(sim_errors) <= SIM_STATE_TOLERANCE
    )
    audit = {
        "schema": SCHEMA,
        "passed": passed,
        "capture_plan": str(args.plan.resolve()),
        "capture_plan_sha256": _sha256_file(args.plan),
        "source_plan_rows": int(plan["n_rows"]),
        "rendered_rows": len(rows),
        "complete_plan": len(rows) == int(plan["n_rows"]),
        "task_suite": "libero_10",
        "task_id": 8,
        "task_name": str(task.name),
        "prompt": prompt,
        "wrist_layout_for_model": "paper-right",
        "dataset": str(args.out.resolve()),
        "dataset_sha256": _sha256_file(args.out),
        "dataset_file_bytes": args.out.stat().st_size,
        "shapes": {name: list(value.shape) for name, value in payload.items()},
        "max_state_restore_abs_error": float(max(state_errors)),
        "max_position_restore_abs_error_m": float(max(position_errors)),
        "max_rotation_restore_abs_error_rad": float(max(rotation_errors)),
        "max_gripper_restore_abs_error": float(max(gripper_errors)),
        "max_sim_restore_abs_error": float(max(sim_errors)),
        "restore_tolerances": {
            "position_m": POSITION_TOLERANCE_M,
            "rotation_rad": ROTATION_TOLERANCE_RAD,
            "gripper": GRIPPER_TOLERANCE,
            "sim_state": SIM_STATE_TOLERANCE,
        },
        "unique_agent_images": len({row["image_sha256"] for row in row_metadata}),
        "unique_wrist_images": len(
            {row["wrist_image_sha256"] for row in row_metadata}
        ),
        "rows": row_metadata,
        "limitations": [
            "RGB frames are regenerated from saved float32 MuJoCo checkpoints, not stored originals.",
            "Exact action reproduction is checked later with the original flow noise and checkpoint.",
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "passed": passed,
                "max_state_restore_abs_error": audit["max_state_restore_abs_error"],
                "max_position_restore_abs_error_m": audit[
                    "max_position_restore_abs_error_m"
                ],
                "max_rotation_restore_abs_error_rad": audit[
                    "max_rotation_restore_abs_error_rad"
                ],
                "max_gripper_restore_abs_error": audit[
                    "max_gripper_restore_abs_error"
                ],
                "max_sim_restore_abs_error": audit["max_sim_restore_abs_error"],
                "unique_agent_images": audit["unique_agent_images"],
                "unique_wrist_images": audit["unique_wrist_images"],
                "dataset_file_bytes": audit["dataset_file_bytes"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = render(args)
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
