from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr

DEFAULT_HUB = Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
RUN_ID = "right-16x32"


@dataclass(frozen=True)
class RunSpec:
    key: str
    suite: str
    task: str
    path: Path
    summaries: tuple[dict[str, Any], ...]

    @property
    def output_stem(self) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "__", self.key)
        return value.strip("_.-")


def discover_runs(hub: Path = DEFAULT_HUB, run_id: str = RUN_ID) -> list[RunSpec]:
    root = hub / "cache" / "HiMoE-VLA"
    paths = sorted(root.glob(f"*/*/{run_id}"))
    runs: list[RunSpec] = []
    for path in paths:
        summary_path = path / "client" / "summaries.json"
        route_path = path / "server" / "routes.zarr"
        meta_path = path / "meta.json"
        if not (summary_path.is_file() and route_path.is_dir() and meta_path.is_file()):
            continue
        rows = sorted(
            json.loads(summary_path.read_text()), key=lambda row: int(row["episode_index"])
        )
        if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"non-contiguous episode indices in {summary_path}")
        suite = path.parents[1].name
        task = path.parent.name
        runs.append(
            RunSpec(
                key=f"{suite}/{task}",
                suite=suite,
                task=task,
                path=path,
                summaries=tuple(rows),
            )
        )
    if not runs:
        raise FileNotFoundError(f"no complete {run_id!r} runs under {root}")
    return runs


def validate_episode_axes(
    run: RunSpec, store: zarr.Group
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_id = np.asarray(store["episode_id"][:], dtype=np.int32)
    control_step = np.asarray(store["control_step"][:], dtype=np.int32)
    lengths = np.asarray([int(row["inference_calls"]) for row in run.summaries], dtype=np.int32)
    expected_episode = np.repeat(np.arange(len(lengths), dtype=np.int32), lengths)
    if not np.array_equal(episode_id, expected_episode):
        raise ValueError(f"{run.key}: route rows do not match client episode lengths")
    if not np.array_equal(control_step, np.arange(len(control_step), dtype=np.int32)):
        raise ValueError(f"{run.key}: control_step is not contiguous")
    episode_step = np.concatenate([np.arange(length, dtype=np.int16) for length in lengths])
    return episode_id, episode_step, control_step


def query_metadata(run: RunSpec, episode_id: np.ndarray) -> dict[str, np.ndarray]:
    success_by_episode = np.asarray([bool(row["success"]) for row in run.summaries])
    state_by_episode = np.asarray(
        [int(row["init_state_id"]) for row in run.summaries], dtype=np.int16
    )
    seed_by_episode = np.asarray(
        [int(row["flow_noise_seed"]) for row in run.summaries], dtype=np.int16
    )
    return {
        "success": success_by_episode[episode_id],
        "init_state_id": state_by_episode[episode_id],
        "flow_noise_seed": seed_by_episode[episode_id],
    }


def load_behavior_features(run: RunSpec) -> tuple[np.ndarray, tuple[str, ...]]:
    """Load small action/proprio controls aligned one-to-one with route queries."""
    parts: list[np.ndarray] = []
    names = (
        "action_arm_rms",
        "action_arm_token_std",
        "action_mean_delta",
        "proprio_delta",
        "gripper_flip_fraction",
    )
    for row in run.summaries:
        episode = int(row["episode_index"])
        path = run.path / "client" / f"episode_{episode:02d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            actions = np.asarray(payload["actions"], dtype=np.float32)
            state = np.asarray(payload["state"], dtype=np.float32)
        expected = int(row["inference_calls"])
        if actions.shape != (expected, 10, 7) or state.shape != (expected, 8):
            raise ValueError(f"unexpected action/state shape in {path}")

        arm = actions[..., :6]
        action_mean = arm.mean(axis=1)
        action_delta = np.zeros(expected, dtype=np.float32)
        proprio_delta = np.zeros(expected, dtype=np.float32)
        if expected > 1:
            action_delta[1:] = np.sqrt(np.mean(np.square(np.diff(action_mean, axis=0)), axis=1))
            proprio_delta[1:] = np.sqrt(np.mean(np.square(np.diff(state[:, :7], axis=0)), axis=1))
        gripper_sign = np.sign(actions[..., 6])
        gripper_flip = np.mean(gripper_sign[:, 1:] != gripper_sign[:, :-1], axis=1)
        parts.append(
            np.column_stack(
                [
                    np.sqrt(np.mean(np.square(arm), axis=(1, 2))),
                    np.mean(np.std(arm, axis=1), axis=1),
                    action_delta,
                    proprio_delta,
                    gripper_flip,
                ]
            ).astype(np.float32)
        )
    return np.concatenate(parts, axis=0), names
