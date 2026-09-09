#!/usr/bin/env python3
"""Analyze matched K=16 rolling-star LIBERO trajectories and HiMoE routes.

Outcome labels in this file are computed only from simulator, end-effector,
gripper, and action trajectories.  Routing is loaded afterwards and is never
used to define a failure type.  The primary routing comparisons use only q0 or
q0:q2, so a terminal timeout/stagnation tail cannot leak into the signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LAYER_NUMBERS = (2, 3, 4, 5, 12, 13, 14, 15)
LAYER_GROUPS = {
    "front_2_5": np.asarray((0, 1, 2, 3), dtype=np.int64),
    "back_12_15": np.asarray((4, 5, 6, 7), dtype=np.int64),
}
N_DENOISE = 10
N_EXPERTS = 32
PREFIXES = (1, 3)
SEED = 20260828
FROZEN_SIGNAL = {
    "freeze_utc": "2026-08-29T01:17:01Z",
    "prefix_queries": 1,
    "token_family": "action",
    "layer_group": "front_2_5",
    "denoise_step": 0,
    "expert": 0,
    "direction": "higher_probability_predicts_failure",
    "discovery_max_snapshot": 2,
    # These are the first snapshot boundaries not yet started when the signal
    # was frozen. In-flight/observed snapshots between discovery and validation
    # remain an explicitly excluded transition set.
    "validation_start_by_worker": {0: 4, 1: 4, 2: 4, 3: 5},
}
FROZEN_STATE_SIGNAL = {
    "freeze_utc": "2026-08-29T01:22:04Z",
    "prefix_queries": 1,
    "token_family": "state",
    "layer_group": "front_2_5",
    "denoise_step": 1,
    "metric": "top1_mass",
    "direction": "higher_value_predicts_failure",
    "discovery_max_snapshot": 2,
    "validation_start_by_worker": {0: 4, 1: 4, 2: 4, 3: 5},
}
FAILURE_LABELS = (
    "stagnation",
    "loop_or_cycling",
    "drop_or_regrasp",
    "subtask_undo",
    "goal_regression",
    "goal_contact_near_miss",
    "stove_not_on",
    "single_subtask_omission",
    "active_retry",
    "no_progress",
)
PRIMARY_PRIORITY = (
    "stove_not_on",
    "goal_contact_near_miss",
    "subtask_undo",
    "drop_or_regrasp",
    "loop_or_cycling",
    "single_subtask_omission",
    "stagnation",
    "goal_regression",
    "active_retry",
    "no_progress",
)

# Physical thresholds.  Dense controls are at the simulator action rate, so
# stagnation is defined over 20-step windows rather than single tiny steps.
GOAL_ENTER_M = 0.055
GOAL_EXIT_M = 0.100
LIFT_ENTER_M = 0.035
LIFT_EXIT_M = 0.012
DROP_M = 0.035
STATIC_WINDOW = 20
STATIC_EEF_PATH_M = 0.020
STATIC_OBJECT_PATH_M = 0.005
STATIC_GRIPPER_PATH_M = 0.001
LOOP_MIN_QUERY_LAG = 3
LOOP_EEF_RETURN_M = 0.045
LOOP_OBJECT_RETURN_M = 0.030
LOOP_GRIPPER_RETURN_M = 0.012
LOOP_EEF_PATH_M = 0.120
ACTIVE_EEF_PATH_M = 0.80
ACTIVE_MAX_GOAL_PROGRESS_PER_EEF = 0.08


@dataclass(frozen=True)
class Candidate:
    worker: int
    init_state: int
    snapshot: int
    candidate: int
    episode_id: int
    success: bool
    inference_calls: int
    snapshot_key: str
    npz_path: Path
    json_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def markdown_table(frame: pd.DataFrame, *, digits: int = 4) -> str:
    if frame.empty:
        return "(no rows)"
    columns = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid router probabilities")
    return result / mass


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    return -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=-1
    ) / np.log(probability.shape[-1])


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            0.0,
            0.5
            * np.sum(
                (np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0)))
                ** 2,
                axis=-1,
            ),
        )
    )


def longest_true_run(mask: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(mask, dtype=bool)
    best = (0, 0, 0)
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(values) and values[stop]:
            stop += 1
        if stop - start > best[0]:
            best = (stop - start, start, stop)
        start = stop
    return best


def hysteresis_events(
    values: np.ndarray, enter: float, exit: float, *, lower_is_inside: bool
) -> tuple[list[int], list[int]]:
    inside = False
    entries: list[int] = []
    exits: list[int] = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        enter_now = value <= enter if lower_is_inside else value >= enter
        exit_now = value >= exit if lower_is_inside else value <= exit
        if not inside and enter_now:
            inside = True
            entries.append(index)
        elif inside and exit_now:
            inside = False
            exits.append(index)
    return entries, exits


def discover_candidates(run_root: Path) -> tuple[list[Candidate], dict[str, Any]]:
    formal = run_root / "formal"
    candidates: list[Candidate] = []
    audit: dict[str, Any] = {
        "workers": {},
        "snapshots": [],
        "ignored_incomplete_snapshots": [],
    }
    for worker_dir in sorted(formal.glob("worker*")):
        config_path = worker_dir / "experiment_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        worker = int(config["worker_id"])
        init_state = int(config["init_state_id"])
        manifests = sorted(worker_dir.glob("snapshot_*/manifest.json"))
        audit["workers"][str(worker)] = {
            "init_state_id": init_state,
            "complete_snapshots": len(manifests),
        }
        for snapshot_dir in sorted(worker_dir.glob("snapshot_*")):
            if not (snapshot_dir / "manifest.json").is_file():
                audit["ignored_incomplete_snapshots"].append(str(snapshot_dir))
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                continue
            snapshot = int(manifest["snapshot_index"])
            snapshot_key = f"w{worker:02d}_s{snapshot:03d}"
            rows = manifest["candidate_summaries"]
            if len(rows) != 16 or {int(row["candidate"]) for row in rows} != set(range(16)):
                raise ValueError(f"{manifest_path}: committed snapshot is not exact K=16")
            order = [int(value) for value in manifest["candidate_execution_order"]]
            if len(order) != 16 or set(order) != set(range(16)):
                raise ValueError(f"{manifest_path}: execution order is not a K=16 permutation")
            snapshot_hash_paths = {
                "full_state_json_sha256": manifest_path.parent / "full_state.json",
                "full_state_npz_sha256": manifest_path.parent / "full_state.npz",
                "policy_input_sha256": manifest_path.parent / "policy_input.npz",
                "trunk_npz_sha256": manifest_path.parent / "trunk.npz",
            }
            for hash_key, artifact_path in snapshot_hash_paths.items():
                if not artifact_path.is_file() or sha256_file(artifact_path) != str(manifest[hash_key]):
                    raise ValueError(f"{manifest_path}: {hash_key} mismatch")
            successes = int(sum(bool(row["success"]) for row in rows))
            if successes != int(manifest["candidate_successes"]):
                raise ValueError(f"{manifest_path}: success count mismatch")
            audit["snapshots"].append(
                {
                    "snapshot_key": snapshot_key,
                    "candidate_successes": successes,
                    "candidate_failures": 16 - successes,
                    "candidate_inference_calls": int(manifest["candidate_inference_calls"]),
                    "wall_s": float(manifest["wall_s"]),
                    "trunk_action_steps_before": int(manifest["trunk_action_steps_before"]),
                    "trunk_action_steps_after": int(manifest["trunk_action_steps_after"]),
                    "trunk_success_after": bool(manifest["trunk_success_after"]),
                    "snapshot_artifact_hashes_verified": len(snapshot_hash_paths),
                }
            )
            for row in rows:
                candidate = int(row["candidate"])
                npz_path = manifest_path.parent / f"candidate_{candidate:02d}.npz"
                json_path = manifest_path.parent / f"candidate_{candidate:02d}.json"
                if not npz_path.is_file() or not json_path.is_file():
                    raise FileNotFoundError(f"missing candidate artifact under {manifest_path.parent}")
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                for key in (
                    "candidate",
                    "episode_id",
                    "success",
                    "inference_calls",
                    "branch_action_steps",
                ):
                    if metadata[key] != row[key]:
                        raise ValueError(f"{json_path}: differs from manifest field {key}")
                if sha256_file(npz_path) != str(metadata["npz_sha256"]):
                    raise ValueError(f"candidate checksum mismatch: {npz_path}")
                candidates.append(
                    Candidate(
                        worker=worker,
                        init_state=init_state,
                        snapshot=snapshot,
                        candidate=candidate,
                        episode_id=int(metadata["episode_id"]),
                        success=bool(metadata["success"]),
                        inference_calls=int(metadata["inference_calls"]),
                        snapshot_key=snapshot_key,
                        npz_path=npz_path,
                        json_path=json_path,
                    )
                )
    candidates.sort(key=lambda item: (item.worker, item.snapshot, item.candidate))
    if not candidates:
        raise RuntimeError("no committed K=16 snapshots found")
    return candidates, audit


def load_layout(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = sorted((run_root / "formal").glob("worker*/sim_layout.json"))
    if not paths:
        raise FileNotFoundError("sim_layout.json is missing")
    layouts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("worker simulator layouts differ")
    targets = [
        joint
        for joint in layouts[0]["joints"]
        if not bool(joint["is_robot"])
        and int(joint["state_hi"]) - int(joint["state_lo"]) == 7
    ]
    if not targets:
        raise RuntimeError("no free-joint object target found")
    return targets, layouts[0]


def load_trajectory(candidate: Candidate) -> dict[str, np.ndarray]:
    required = {
        "flow_noise",
        "policy_state",
        "sim_state",
        "action_chunks",
        "action_executed",
        "control_sim_state",
        "control_eef_position",
        "control_gripper_qpos",
        "control_action",
        "control_query_index",
    }
    with np.load(candidate.npz_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"{candidate.npz_path}: missing {sorted(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    query = len(arrays["flow_noise"])
    actions = len(arrays["control_action"])
    if query != candidate.inference_calls:
        raise ValueError(f"{candidate.npz_path}: inference call count mismatch")
    if len(arrays["control_sim_state"]) != actions + 1:
        raise ValueError(f"{candidate.npz_path}: dense state/action alignment failed")
    return arrays


def build_goal_references(
    candidates: list[Candidate], trajectories: dict[int, dict[str, np.ndarray]], targets: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    successes = [candidate for candidate in candidates if candidate.success]
    if not successes:
        raise RuntimeError("physical taxonomy needs at least one successful branch")
    # Both moka pots share the same LIBERO `flat_stove_1_cook_region` goal.
    # Pool terminal positions across object identities so a valid placement in
    # the other stove slot is not treated as a target-specific miss.
    pooled = []
    for target in targets:
        lo = int(target["state_lo"])
        pooled.append(np.stack(
            [trajectories[item.episode_id]["control_sim_state"][-1, lo : lo + 3] for item in successes]
        ).astype(np.float32))
    shared = np.concatenate(pooled, axis=0)
    return {str(target["joint"]): shared.copy() for target in targets}


def audit_branch_noises(
    candidates: list[Candidate], trajectories: dict[int, dict[str, np.ndarray]]
) -> dict[str, Any]:
    rows = []
    for snapshot_key in sorted({item.snapshot_key for item in candidates}):
        siblings = [item for item in candidates if item.snapshot_key == snapshot_key]
        if len(siblings) != 16:
            raise ValueError(f"{snapshot_key}: noise audit expected 16 siblings")
        q0_hashes = []
        stream_hashes = []
        for item in siblings:
            noise = np.ascontiguousarray(
                trajectories[item.episode_id]["flow_noise"], dtype=np.float32
            )
            q0_hashes.append(hashlib.sha256(noise[0].tobytes(order="C")).hexdigest())
            stream_hashes.append(hashlib.sha256(noise.tobytes(order="C")).hexdigest())
        if len(set(q0_hashes)) != 16 or len(set(stream_hashes)) != 16:
            raise ValueError(f"{snapshot_key}: branch flow-noise collision")
        rows.append(
            {
                "snapshot_key": snapshot_key,
                "q0_unique": len(set(q0_hashes)),
                "full_stream_unique": len(set(stream_hashes)),
            }
        )
    return {
        "snapshots_checked": len(rows),
        "all_q0_exactly_k16_unique": True,
        "all_full_streams_exactly_k16_unique": True,
        "per_snapshot": rows,
    }


def candidate_id_negative_control(
    candidates: list[Candidate], *, permutations: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    snapshots = sorted({item.snapshot_key for item in candidates})
    outcomes = np.empty((len(snapshots), 16), dtype=np.float32)
    for snapshot_axis, snapshot_key in enumerate(snapshots):
        siblings = [item for item in candidates if item.snapshot_key == snapshot_key]
        if len(siblings) != 16:
            raise ValueError(f"{snapshot_key}: candidate-ID control requires K=16")
        for item in siblings:
            outcomes[snapshot_axis, item.candidate] = float(not item.success)
    observed_rate = outcomes.mean(axis=0)
    overall = float(outcomes.mean())
    observed_max = float(np.max(np.abs(observed_rate - overall)))
    rng = np.random.default_rng(seed + 1709)
    null_max = np.empty(permutations, dtype=np.float32)
    for draw in range(permutations):
        shuffled = np.stack([rng.permutation(row) for row in outcomes])
        null_max[draw] = np.max(np.abs(shuffled.mean(axis=0) - overall))
    p_value = float(
        (1 + np.sum(null_max >= observed_max)) / (1 + permutations)
    )
    table = pd.DataFrame(
        {
            "candidate": np.arange(16, dtype=np.int64),
            "snapshots": len(snapshots),
            "failures": outcomes.sum(axis=0).astype(np.int64),
            "failure_rate": observed_rate,
            "deviation_from_overall": observed_rate - overall,
        }
    )
    return table, {
        "snapshots": len(snapshots),
        "overall_failure_rate": overall,
        "max_abs_candidate_rate_deviation": observed_max,
        "within_snapshot_permutation_p": p_value,
        "permutations": permutations,
    }


def audit_auxiliary_joints(
    candidates: list[Candidate],
    trajectories: dict[int, dict[str, np.ndarray]],
    layout: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for joint in layout["joints"]:
        lo, hi = int(joint["state_lo"]), int(joint["state_hi"])
        if bool(joint["is_robot"]) or hi - lo != 1:
            continue
        starts = []
        ends = []
        ranges = []
        snapshot_start: dict[str, list[float]] = defaultdict(list)
        for item in candidates:
            values = np.asarray(
                trajectories[item.episode_id]["control_sim_state"][:, lo],
                dtype=np.float64,
            )
            starts.append(float(values[0]))
            ends.append(float(values[-1]))
            ranges.append(float(values.max() - values.min()))
            snapshot_start[item.snapshot_key].append(float(values[0]))
        rows.append(
            {
                "joint": str(joint["joint"]),
                "global_start_min": min(starts),
                "global_start_max": max(starts),
                "global_end_min": min(ends),
                "global_end_max": max(ends),
                "max_within_branch_range": max(ranges),
                "branches_with_range_gt_1e_3": int(
                    np.sum(np.asarray(ranges) > 1e-3)
                ),
                "max_within_snapshot_start_span": max(
                    max(values) - min(values) for values in snapshot_start.values()
                ),
            }
        )
    return {"scalar_nonrobot_joints": rows}


def distance_to_references(position: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(position, dtype=np.float32)[:, None, :]
        - np.asarray(references, dtype=np.float32)[None, :, :],
        axis=-1,
    ).min(axis=1)


def static_window_metrics(
    eef: np.ndarray, objects: np.ndarray, gripper: np.ndarray
) -> dict[str, float]:
    steps = len(eef) - 1
    if steps < STATIC_WINDOW:
        return {
            "static_window_fraction": 0.0,
            "late_static_window_fraction": 0.0,
            "longest_static_window_steps": 0.0,
            "terminal_static_window_steps": 0.0,
        }
    eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    object_step = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    grip_step = np.abs(np.diff(gripper))
    kernel = np.ones(STATIC_WINDOW, dtype=np.float64)
    eef_path = np.convolve(eef_step, kernel, mode="valid")
    object_path = np.convolve(object_step, kernel, mode="valid")
    grip_path = np.convolve(grip_step, kernel, mode="valid")
    static = (
        (eef_path <= STATIC_EEF_PATH_M)
        & (object_path <= STATIC_OBJECT_PATH_M)
        & (grip_path <= STATIC_GRIPPER_PATH_M)
    )
    run = longest_true_run(static)
    terminal = 0
    for value in static[::-1]:
        if not value:
            break
        terminal += 1
    late_start = len(static) // 2
    return {
        "static_window_fraction": float(static.mean()),
        "late_static_window_fraction": float(static[late_start:].mean()),
        "longest_static_window_steps": float(run[0] + STATIC_WINDOW - 1 if run[0] else 0),
        "terminal_static_window_steps": float(terminal + STATIC_WINDOW - 1 if terminal else 0),
    }


def query_loop_metrics(
    eef: np.ndarray,
    objects: np.ndarray,
    gripper: np.ndarray,
    goal_distance: np.ndarray,
) -> dict[str, float]:
    n_query = len(eef)
    if n_query <= LOOP_MIN_QUERY_LAG:
        return {
            "loop_return_count": 0.0,
            "loop_return_fraction": 0.0,
            "loop_max_lag": 0.0,
            "loop_best_eef_path_m": 0.0,
        }
    eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    eef_cumulative = np.r_[0.0, np.cumsum(eef_step)]
    count = 0
    max_lag = 0
    best_path = 0.0
    returned_query: set[int] = set()
    for right in range(LOOP_MIN_QUERY_LAG, n_query):
        for left in range(0, right - LOOP_MIN_QUERY_LAG + 1):
            eef_return = float(np.linalg.norm(eef[right] - eef[left]))
            object_return = float(
                np.linalg.norm(objects[right] - objects[left], axis=1).max()
            )
            grip_return = float(abs(gripper[right] - gripper[left]))
            path = float(eef_cumulative[right] - eef_cumulative[left])
            progress = float(goal_distance[left] - goal_distance[right])
            if (
                eef_return <= LOOP_EEF_RETURN_M
                and object_return <= LOOP_OBJECT_RETURN_M
                and grip_return <= LOOP_GRIPPER_RETURN_M
                and path >= LOOP_EEF_PATH_M
                and progress <= 0.035
            ):
                count += 1
                returned_query.add(right)
                max_lag = max(max_lag, right - left)
                best_path = max(best_path, path)
    return {
        "loop_return_count": float(count),
        "loop_return_fraction": float(len(returned_query) / n_query),
        "loop_max_lag": float(max_lag),
        "loop_best_eef_path_m": best_path,
    }


def query_reversal_rate(eef: np.ndarray) -> float:
    delta = np.diff(np.asarray(eef, dtype=np.float64), axis=0)
    norm = np.linalg.norm(delta, axis=1)
    reversals = 0
    eligible = 0
    for index in range(1, len(delta)):
        if norm[index - 1] <= 0.015 or norm[index] <= 0.015:
            continue
        eligible += 1
        cosine = float(
            np.dot(delta[index - 1], delta[index]) / (norm[index - 1] * norm[index])
        )
        reversals += int(cosine < -0.25)
    return float(reversals / eligible) if eligible else 0.0


def physical_metrics(
    candidate: Candidate,
    arrays: dict[str, np.ndarray],
    targets: list[dict[str, Any]],
    references: dict[str, np.ndarray],
) -> dict[str, Any]:
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    eef = np.asarray(arrays["control_eef_position"], dtype=np.float32)
    gripper_pair = np.asarray(arrays["control_gripper_qpos"], dtype=np.float32)
    gripper = gripper_pair.mean(axis=1)
    actions = np.asarray(arrays["control_action"], dtype=np.float32)
    object_positions = []
    object_goal_distances = []
    row: dict[str, Any] = {
        "worker": candidate.worker,
        "init_state_id": candidate.init_state,
        "snapshot": candidate.snapshot,
        "snapshot_key": candidate.snapshot_key,
        "candidate": candidate.candidate,
        "episode_id": candidate.episode_id,
        "success": candidate.success,
        "failure": not candidate.success,
        "inference_calls": candidate.inference_calls,
        "action_steps": len(actions),
    }
    labels: set[str] = set()
    placed_end = 0
    ever_goal = 0
    unmoved_end = 0
    unplaced_unmoved_end = 0
    lift_count = 0
    drop_count = 0
    undo_count = 0
    regression_max = 0.0
    total_object_path = 0.0
    target_summaries: dict[str, Any] = {}
    for target in targets:
        name = str(target["joint"])
        lo = int(target["state_lo"])
        position = sim[:, lo : lo + 3]
        object_positions.append(position)
        distance = distance_to_references(position, references[name])
        object_goal_distances.append(distance)
        path = float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum())
        net = float(np.linalg.norm(position[-1] - position[0]))
        height = position[:, 2] - position[0, 2]
        lifts, lift_exits = hysteresis_events(
            height, LIFT_ENTER_M, LIFT_EXIT_M, lower_is_inside=False
        )
        goal_entries, goal_exits = hysteresis_events(
            distance, GOAL_ENTER_M, GOAL_EXIT_M, lower_is_inside=True
        )
        target_drops = 0
        for lift in lifts:
            future = height[lift:]
            if len(future) and float(height[lift] - future.min()) >= DROP_M:
                low = lift + int(np.argmin(future))
                if distance[low] > GOAL_EXIT_M:
                    target_drops += 1
        regression = float(np.max(distance[np.argmin(distance) :]) - np.min(distance))
        placed = bool(distance[-1] <= GOAL_ENTER_M)
        placed_end += int(placed)
        ever_goal += int(bool(goal_entries))
        unmoved_end += int(net <= 0.025)
        unplaced_unmoved_end += int((not placed) and net <= 0.025)
        lift_count += len(lifts)
        drop_count += target_drops
        undo_count += len(goal_exits)
        regression_max = max(regression_max, regression)
        total_object_path += path
        target_summaries[name] = {
            "path_m": path,
            "net_m": net,
            "max_lift_m": float(height.max()),
            "lift_count": len(lifts),
            "drop_count": target_drops,
            "goal_min_m": float(distance.min()),
            "goal_end_m": float(distance[-1]),
            "goal_entry_count": len(goal_entries),
            "goal_exit_count": len(goal_exits),
            "placed_end": placed,
            "goal_regression_m": regression,
        }
    objects = np.stack(object_positions, axis=1)
    goal_matrix = np.stack(object_goal_distances, axis=1)
    aggregate_goal = goal_matrix.mean(axis=1)
    query_starts = np.r_[0, np.flatnonzero(np.diff(arrays["control_query_index"]) != 0) + 1]
    query_starts = query_starts[: candidate.inference_calls]
    query_eef = eef[query_starts]
    query_objects = objects[query_starts]
    query_gripper = gripper[query_starts]
    query_goal = aggregate_goal[query_starts]
    row.update(static_window_metrics(eef, objects, gripper))
    row.update(
        query_loop_metrics(query_eef, query_objects, query_gripper, query_goal)
    )
    eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    eef_path = float(eef_step.sum())
    eef_net = float(np.linalg.norm(eef[-1] - eef[0]))
    progress = float(aggregate_goal[0] - aggregate_goal[-1])
    efficiency = max(progress, 0.0) / max(eef_path, 1e-8)
    query_actions = []
    for query in range(candidate.inference_calls):
        chosen = actions[arrays["control_query_index"] == query]
        if len(chosen):
            query_actions.append(float(chosen[:, 6].mean()))
    query_actions_array = np.asarray(query_actions)
    command_flips = int(
        np.sum(
            np.sign(query_actions_array[1:]) != np.sign(query_actions_array[:-1])
        )
    ) if len(query_actions_array) > 1 else 0
    row.update(
        {
            "target_count": len(targets),
            "placed_end_count": placed_end,
            "ever_goal_count": ever_goal,
            "unmoved_end_count": unmoved_end,
            "unplaced_unmoved_end_count": unplaced_unmoved_end,
            "lift_count": lift_count,
            "drop_count": drop_count,
            "subtask_undo_count": undo_count,
            "goal_regression_max_m": regression_max,
            "object_path_total_m": total_object_path,
            "goal_distance_start_m": float(aggregate_goal[0]),
            "goal_distance_end_m": float(aggregate_goal[-1]),
            "goal_progress_m": progress,
            "eef_path_m": eef_path,
            "eef_net_m": eef_net,
            "eef_path_efficiency": eef_net / max(eef_path, 1e-8),
            "goal_progress_per_eef_m": efficiency,
            "descriptor_active_low_efficiency": bool(
                eef_path >= ACTIVE_EEF_PATH_M
                and efficiency <= ACTIVE_MAX_GOAL_PROGRESS_PER_EEF
            ),
            "eef_query_reversal_rate": query_reversal_rate(query_eef),
            "gripper_query_flip_count": command_flips,
            "target_metrics_json": json.dumps(target_summaries, sort_keys=True),
        }
    )
    if candidate.success:
        primary = "success"
    else:
        if row["terminal_static_window_steps"] >= 80 or row["late_static_window_fraction"] >= 0.45:
            labels.add("stagnation")
        if row["loop_return_count"] >= 2 or row["loop_return_fraction"] >= 0.10:
            labels.add("loop_or_cycling")
        if drop_count >= 1 or lift_count >= len(targets) + 2:
            labels.add("drop_or_regrasp")
        if undo_count >= 1:
            labels.add("subtask_undo")
        if regression_max >= 0.060:
            labels.add("goal_regression")
        if placed_end == len(targets):
            labels.add("goal_contact_near_miss")
        if 0 < placed_end < len(targets) and unplaced_unmoved_end >= 1:
            labels.add("single_subtask_omission")
        if total_object_path <= 0.035 and eef_path <= 0.30:
            labels.add("no_progress")
        if bool(row["descriptor_active_low_efficiency"]) and not labels:
            labels.add("active_retry")
        primary = next((name for name in PRIMARY_PRIORITY if name in labels), "timeout_other")
    row["physical_labels"] = ";".join(sorted(labels)) if labels else primary
    row["primary_failure_type"] = primary
    for label in FAILURE_LABELS:
        row[f"label_{label}"] = bool(label in labels)
    return row


def merge_terminal_predicates(
    run_root: Path, physical: pd.DataFrame, candidates: list[Candidate]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = run_root / "formal" / "physical_audit" / "terminal_predicates.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"terminal predicate audit is required before analysis: {path}"
        )
    predicates = pd.read_csv(path)
    expected = {item.episode_id for item in candidates}
    actual = set(predicates["episode_id"].astype(int))
    if actual != expected or len(predicates) != len(candidates):
        raise ValueError(
            f"terminal predicate episodes differ: expected={len(expected)} actual={len(actual)}"
        )
    predicate_columns = [
        column
        for column in predicates
        if column.startswith("on__") or column.startswith("turnon__")
    ]
    on_columns = [column for column in predicate_columns if column.startswith("on__")]
    turnon_columns = [
        column for column in predicate_columns if column.startswith("turnon__")
    ]
    if len(on_columns) != 2 or len(turnon_columns) != 1:
        raise ValueError(f"unexpected terminal predicates: {predicate_columns}")
    for column in ["recorded_success", "replayed_success", *predicate_columns]:
        if predicates[column].dtype != bool:
            predicates[column] = predicates[column].map(
                {"True": True, "False": False, True: True, False: False}
            )
        if predicates[column].isna().any():
            raise ValueError(f"invalid boolean terminal predicate column: {column}")
    if not np.array_equal(
        predicates["recorded_success"].to_numpy(dtype=bool),
        predicates["replayed_success"].to_numpy(dtype=bool),
    ):
        raise ValueError("terminal predicate replay disagrees with recorded success")
    keep = ["episode_id", "recorded_success", "replayed_success", *predicate_columns]
    merged = physical.merge(
        predicates[keep], on="episode_id", how="left", validate="one_to_one"
    )
    if not np.array_equal(
        merged["success"].to_numpy(dtype=bool),
        merged["replayed_success"].to_numpy(dtype=bool),
    ):
        raise ValueError("physical outcome disagrees with terminal predicate replay")
    merged["terminal_exact_on_count"] = merged[on_columns].sum(axis=1).astype(int)
    merged["terminal_exact_turnon"] = merged[turnon_columns[0]].astype(bool)
    failure = merged["failure"].to_numpy(dtype=bool)
    merged["label_goal_contact_near_miss"] = (
        failure
        & (merged["placed_end_count"].to_numpy(dtype=int) == merged["target_count"].to_numpy(dtype=int))
        & (merged["terminal_exact_on_count"].to_numpy(dtype=int) < merged["target_count"].to_numpy(dtype=int))
    )
    merged["label_stove_not_on"] = failure & ~merged["terminal_exact_turnon"].to_numpy(dtype=bool)
    exact_omission = []
    for _, row in merged.iterrows():
        omission = False
        if bool(row["failure"]) and int(row["terminal_exact_on_count"]) == 1:
            missing = [
                column.split("__", 2)[1]
                for column in on_columns
                if not bool(row[column])
            ]
            target_metrics = json.loads(str(row["target_metrics_json"]))
            matching = [
                metrics
                for joint, metrics in target_metrics.items()
                if any(str(joint).startswith(object_name) for object_name in missing)
            ]
            omission = bool(
                len(missing) == 1
                and len(matching) == 1
                and float(matching[0]["net_m"]) <= 0.025
            )
        exact_omission.append(omission)
    merged["label_single_subtask_omission"] = exact_omission
    primary = []
    signatures = []
    for row in merged.itertuples(index=False):
        if bool(row.success):
            primary.append("success")
            signatures.append("success")
            continue
        active = [
            label for label in FAILURE_LABELS if bool(getattr(row, f"label_{label}"))
        ]
        chosen = next(
            (label for label in PRIMARY_PRIORITY if label in active), "timeout_other"
        )
        primary.append(chosen)
        signatures.append(";".join(sorted(active)) if active else chosen)
    merged["primary_failure_type"] = primary
    merged["physical_labels"] = signatures
    merged["label_non_stagnation_non_loop"] = (
        merged["failure"].to_numpy(dtype=bool)
        & ~merged["label_stagnation"].to_numpy(dtype=bool)
        & ~merged["label_loop_or_cycling"].to_numpy(dtype=bool)
    )
    return merged, {
        "path": str(path),
        "branches": len(predicates),
        "predicate_columns": predicate_columns,
        "recorded_replay_mismatches": 0,
        "pattern_counts": {
            "|".join(str(int(value)) for value in key): int(count)
            for key, count in predicates.groupby(predicate_columns).size().items()
        },
    }


def load_routes(
    run_root: Path, candidates: list[Candidate]
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    route_path = run_root / "formal" / "server" / "routes.zarr"
    store = zarr.open_group(str(route_path), mode="r")
    required = {"episode_id", "control_step", "hb_router_probs", "as_probs"}
    missing = required - set(store.array_keys())
    if missing:
        raise ValueError(f"route store missing {sorted(missing)}")
    episode = np.asarray(store["episode_id"][:], dtype=np.int64)
    control = np.asarray(store["control_step"][:], dtype=np.int64)
    if len(episode) != len(control) or not np.array_equal(control, np.arange(len(control))):
        raise ValueError("route store global control-step alignment failed")
    candidate_episode_ids = {item.episode_id for item in candidates}
    expected_trunk_ids = {
        (item.worker + 1) * 100_000_000 + 90_000_000 + item.snapshot
        for item in candidates
    }
    candidate_mask = np.isin(episode, list(candidate_episode_ids))
    trunk_mask = np.isin(episode, list(expected_trunk_ids))
    unknown_route_rows = int(np.sum(~candidate_mask & ~trunk_mask))
    trunk_route_rows = int(trunk_mask.sum())
    bad_trunk_counts = {
        str(episode_id): int(np.sum(episode == episode_id))
        for episode_id in expected_trunk_ids
        if int(np.sum(episode == episode_id)) != 1
    }
    capture_complete = (
        run_root / "formal" / "server" / "capture_summary.json"
    ).is_file()
    if capture_complete and (
        unknown_route_rows != 0
        or trunk_route_rows != len(expected_trunk_ids)
        or bad_trunk_counts
    ):
        raise ValueError(
            "closed route store does not equal committed candidate + trunk queries: "
            f"unknown={unknown_route_rows} trunk={trunk_route_rows}/"
            f"{len(expected_trunk_ids)} bad_counts={bad_trunk_counts}"
        )
    hb_by_episode: dict[int, np.ndarray] = {}
    as_by_episode: dict[int, np.ndarray] = {}
    early_rows = max(PREFIXES)
    early_indices_by_episode: dict[int, np.ndarray] = {}
    for candidate in candidates:
        indices = np.flatnonzero(episode == candidate.episode_id)
        if len(indices) != candidate.inference_calls:
            raise ValueError(
                f"episode {candidate.episode_id}: {len(indices)} routes != "
                f"{candidate.inference_calls} candidate calls"
            )
        early_indices_by_episode[candidate.episode_id] = indices[:early_rows]
    batch_size = 64
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        early_indices = np.concatenate(
            [early_indices_by_episode[item.episode_id] for item in batch]
        )
        hb_batch = normalize_probabilities(
            np.asarray(
                store["hb_router_probs"].oindex[early_indices], dtype=np.float32
            )
        )
        as_batch = normalize_probabilities(
            np.asarray(store["as_probs"].oindex[early_indices], dtype=np.float32)
        )
        if hb_batch.shape[1:] != (8, 10, 11, 32):
            raise ValueError(f"unexpected HB shape {hb_batch.shape}")
        for offset, candidate in enumerate(batch):
            lo = offset * early_rows
            hi = lo + early_rows
            hb_by_episode[candidate.episode_id] = hb_batch[lo:hi]
            as_by_episode[candidate.episode_id] = as_batch[lo:hi]
        print(
            f"loaded routes {min(start + len(batch), len(candidates))}/{len(candidates)}",
            flush=True,
        )
    audit = {
        "route_rows_total": len(episode),
        "candidate_route_rows": int(sum(item.inference_calls for item in candidates)),
        "candidate_route_rows_loaded_for_features": int(
            early_rows * len(candidates)
        ),
        "capture_complete": capture_complete,
        "expected_trunk_route_rows": len(expected_trunk_ids),
        "trunk_route_rows": trunk_route_rows,
        "unknown_or_active_route_rows": unknown_route_rows,
        "bad_trunk_counts": bad_trunk_counts,
        "route_store_arrays": {
            key: {"shape": list(store[key].shape), "dtype": str(store[key].dtype)}
            for key in sorted(store.array_keys())
        },
        "store_hidden": (run_root / "formal" / "server" / "hidden.zarr").exists(),
    }
    return hb_by_episode, as_by_episode, audit


def validate_capture_summary(
    capture_summary: dict[str, Any], route_audit: dict[str, Any]
) -> dict[str, Any]:
    """Cross-check the closed server summary against the durable route store."""
    if not bool(capture_summary.get("store_full_probs")):
        raise ValueError("closed capture did not store full router probabilities")
    if bool(capture_summary.get("store_hidden")) or route_audit["store_hidden"]:
        raise ValueError("hidden states unexpectedly present")
    if capture_summary.get("hook_verify_failures"):
        raise ValueError(
            f"route hook verification failures: {capture_summary['hook_verify_failures']}"
        )

    total_route_rows = int(route_audit["route_rows_total"])
    for key in ("control_steps", "as_collapsed"):
        if int(capture_summary.get(key, -1)) != total_route_rows:
            raise ValueError(
                f"capture summary {key}={capture_summary.get(key)} != "
                f"route rows {total_route_rows}"
            )
    for key in ("captured_rows", "session_queries", "session_captured"):
        if key in capture_summary and int(capture_summary[key]) != total_route_rows:
            raise ValueError(
                f"capture summary {key}={capture_summary[key]} != "
                f"route rows {total_route_rows}"
            )
    if "durable_through_row" in capture_summary and int(
        capture_summary["durable_through_row"]
    ) != total_route_rows - 1:
        raise ValueError("capture summary durable boundary is not the final row")

    # The recorder verifies each gate invocation, not each stored query. For an
    # unpinned run it checks the first four control steps, each with every gate at
    # every denoising step: 4 * 12 * 10 = 480 calls for this checkpoint.
    gate_count = len(capture_summary.get("gates", []))
    denoise_steps = int(capture_summary.get("n_denoise", 0))
    calls_per_control_step = gate_count * denoise_steps
    expected_verified_steps = (
        0
        if capture_summary.get("pin") is not None
        else min(4, total_route_rows)
    )
    expected_verified_calls = expected_verified_steps * calls_per_control_step
    verified_calls = int(capture_summary.get("hook_verified_calls", -1))
    if calls_per_control_step <= 0 or verified_calls != expected_verified_calls:
        raise ValueError(
            "capture summary hook verification count is inconsistent: "
            f"observed={verified_calls}, expected={expected_verified_calls} "
            f"({expected_verified_steps} control steps x {gate_count} gates x "
            f"{denoise_steps} denoise steps)"
        )
    return {
        "route_rows_matched": total_route_rows,
        "hook_verified_gate_calls": verified_calls,
        "hook_verified_control_steps": expected_verified_steps,
        "hook_calls_per_control_step": calls_per_control_step,
    }


def route_feature_banks(
    candidates: list[Candidate],
    trajectories: dict[int, dict[str, np.ndarray]],
    hb_routes: dict[int, np.ndarray],
    as_routes: dict[int, np.ndarray],
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame, dict[str, Any]]:
    banks: dict[int, dict[str, np.ndarray]] = {prefix: {} for prefix in PREFIXES}
    cell_rows: list[dict[str, Any]] = []
    state_span = {prefix: 0.0 for prefix in PREFIXES}
    as_span = {prefix: 0.0 for prefix in PREFIXES}
    for prefix in PREFIXES:
        summary_vectors = []
        action_full_vectors = []
        state_full_vectors = []
        as_full_vectors = []
        control_vectors = []
        noise_vectors = []
        action_vectors = []
        state_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        as_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
        candidate_cell_values: dict[
            int, dict[tuple[str, str, int, str], float]
        ] = {}
        candidate_cell_raw: dict[
            int, dict[tuple[str, str, int], np.ndarray]
        ] = {}
        for candidate in candidates:
            hb = hb_routes[candidate.episode_id][:prefix]
            action = hb[:, :, :, 1:, :]
            state = hb[:, :, :, 0, :]
            state_vectors[candidate.snapshot_key].append(state.reshape(-1))
            as_vectors[candidate.snapshot_key].append(
                as_routes[candidate.episode_id][:prefix].reshape(-1)
            )
            entropy = normalized_entropy(action)
            top1 = action.max(axis=-1)
            token_center = action.mean(axis=3, keepdims=True)
            dispersion = hellinger(action, token_center)
            state_entropy = normalized_entropy(state)
            state_top1 = state.max(axis=-1)
            summary = []
            cell_value: dict[tuple[str, str, int, str], float] = {}
            cell_raw: dict[tuple[str, str, int], np.ndarray] = {}
            for group_name, layer_axes in LAYER_GROUPS.items():
                for denoise in range(N_DENOISE):
                    selection = np.ix_(
                        np.arange(len(action)), layer_axes, [denoise],
                        np.arange(action.shape[3]), np.arange(action.shape[4])
                    )
                    cell = action[selection].reshape(
                        len(action), len(layer_axes), action.shape[3], action.shape[4]
                    )
                    raw_vector = cell.mean(axis=0).reshape(-1)
                    cell_raw[("action", group_name, denoise)] = raw_vector
                    values = {
                        "entropy": float(entropy[:, layer_axes, denoise].mean()),
                        "top1_mass": float(top1[:, layer_axes, denoise].mean()),
                        "token_dispersion": float(
                            dispersion[:, layer_axes, denoise].mean()
                        ),
                    }
                    for metric, value in values.items():
                        cell_value[("action", group_name, denoise, metric)] = value
                        summary.append(value)
                    state_cell = state[:, layer_axes, denoise, :]
                    cell_raw[("state", group_name, denoise)] = (
                        state_cell.mean(axis=0).reshape(-1)
                    )
                    cell_value[("state", group_name, denoise, "entropy")] = float(
                        state_entropy[:, layer_axes, denoise].mean()
                    )
                    cell_value[("state", group_name, denoise, "top1_mass")] = float(
                        state_top1[:, layer_axes, denoise].mean()
                    )
            candidate_cell_values[candidate.episode_id] = cell_value
            candidate_cell_raw[candidate.episode_id] = cell_raw
            # Keep query order for the full banks.  Prefix-3 models can then
            # distinguish an initial route from the response after two chunks.
            action_full_vectors.append(action.mean(axis=3).reshape(-1))
            state_full_vectors.append(state.reshape(-1))
            as_full_vectors.append(
                as_routes[candidate.episode_id][:prefix].reshape(-1)
            )
            summary_vectors.append(np.asarray(summary, dtype=np.float32))
            arrays = trajectories[candidate.episode_id]
            noise = arrays["flow_noise"][:prefix]
            chunks = arrays["action_chunks"][:prefix]
            noise_vectors.append(noise.reshape(-1).astype(np.float32))
            action_vectors.append(chunks.reshape(-1).astype(np.float32))
            control_vectors.append(
                np.asarray(
                    [
                        float(noise.mean()),
                        float(noise.std()),
                        float(np.linalg.norm(noise)),
                        float(np.linalg.norm(chunks[..., :3], axis=-1).sum()),
                        float(np.linalg.norm(chunks[..., 3:6], axis=-1).sum()),
                        float(chunks[..., 6].mean()),
                        float(chunks[..., 6].std()),
                        float(np.linalg.norm(chunks[:, -1, :3] - chunks[:, 0, :3], axis=-1).sum()),
                    ],
                    dtype=np.float32,
                )
            )
        for snapshot_key, vectors in state_vectors.items():
            values = np.stack(vectors)
            state_span[prefix] = max(state_span[prefix], float(np.ptp(values, axis=0).max()))
        for snapshot_key, vectors in as_vectors.items():
            values = np.stack(vectors)
            as_span[prefix] = max(
                as_span[prefix], float(np.ptp(values, axis=0).max())
            )
        # Consensus distance is intentionally unsupervised and computed only
        # among the 16 matched siblings at each snapshot.
        for snapshot_key in sorted({item.snapshot_key for item in candidates}):
            siblings = [item for item in candidates if item.snapshot_key == snapshot_key]
            for token_family in ("action", "state"):
                for group_name in LAYER_GROUPS:
                    for denoise in range(N_DENOISE):
                        matrix = np.stack(
                            [
                                candidate_cell_raw[item.episode_id][
                                    (token_family, group_name, denoise)
                                ]
                                for item in siblings
                            ]
                        )
                        total = matrix.sum(axis=0)
                        # These vectors concatenate distributions. Use the other
                        # K-1 siblings as the center, then average Hellinger over
                        # layer/token distributions.
                        for item, vector in zip(siblings, matrix):
                            center = (total - vector) / max(1, len(siblings) - 1)
                            distance = float(
                                hellinger(
                                    vector.reshape(-1, N_EXPERTS),
                                    center.reshape(-1, N_EXPERTS),
                                ).mean()
                            )
                            candidate_cell_values[item.episode_id][
                                (
                                    token_family,
                                    group_name,
                                    denoise,
                                    "consensus_distance",
                                )
                            ] = distance
        for candidate in candidates:
            for (token_family, group_name, denoise, metric), value in sorted(
                candidate_cell_values[candidate.episode_id].items()
            ):
                cell_rows.append(
                    {
                        "prefix_queries": prefix,
                        "snapshot_key": candidate.snapshot_key,
                        "episode_id": candidate.episode_id,
                        "success": candidate.success,
                        "failure": not candidate.success,
                        "token_family": token_family,
                        "layer_group": group_name,
                        "denoise_step": denoise,
                        "metric": metric,
                        "value": value,
                    }
                )
        banks[prefix] = {
            "moe_action_summary": np.stack(summary_vectors),
            "moe_action_full": np.stack(action_full_vectors),
            "moe_state_full": np.stack(state_full_vectors),
            "moe_as_full": np.stack(as_full_vectors),
            "noise_raw": np.stack(noise_vectors),
            "action_raw": np.stack(action_vectors),
            "controls_summary": np.stack(control_vectors),
        }
    audit = {
        "q0_state_within_snapshot_max_abs_probability_span": state_span[1],
        "q0_q2_state_within_snapshot_max_abs_probability_span": state_span[3],
        "q0_as_within_snapshot_max_abs_probability_span": as_span[1],
        "q0_q2_as_within_snapshot_max_abs_probability_span": as_span[3],
        "early_prefixes": list(PREFIXES),
        "tail_features_used": False,
        "denoise_order": "d0 noisiest (flow time 1.0), d9 final recorded forward (0.1)",
        "feature_shapes": {
            str(prefix): {
                key: list(value.shape) for key, value in banks[prefix].items()
            }
            for prefix in PREFIXES
        },
    }
    return banks, pd.DataFrame(cell_rows), audit


def expert_route_screen(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    hb_routes: dict[int, np.ndarray],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Screen every expert probability with labels permuted within snapshots."""
    failures = physical.set_index("episode_id").loc[
        [item.episode_id for item in candidates], "failure"
    ].to_numpy(dtype=bool)
    groups = np.asarray([item.snapshot_key for item in candidates])
    mixed_groups = [
        group
        for group in np.unique(groups)
        if len(np.unique(failures[groups == group])) == 2
    ]
    if not mixed_groups:
        return pd.DataFrame(), {"mixed_snapshots": 0}
    snapshot_worker = {
        item.snapshot_key: item.worker for item in candidates
    }
    mixed_workers = np.asarray(
        [snapshot_worker[group] for group in mixed_groups], dtype=np.int64
    )
    rows = []
    rng = np.random.default_rng(seed + 313)
    for prefix in PREFIXES:
        feature_rows = []
        feature_keys: list[tuple[str, str, int, int]] = []
        for candidate_axis, candidate in enumerate(candidates):
            hb = hb_routes[candidate.episode_id][:prefix]
            action = hb[:, :, :, 1:, :]
            state = hb[:, :, :, 0, :]
            parts = []
            candidate_keys = []
            for token_family in ("action", "state"):
                token = action if token_family == "action" else state
                for group_name, layer_axes in LAYER_GROUPS.items():
                    if token_family == "action":
                        cell = token[:, layer_axes].mean(axis=(0, 1, 3))
                    else:
                        cell = token[:, layer_axes].mean(axis=(0, 1))
                    if cell.shape != (N_DENOISE, N_EXPERTS):
                        raise ValueError(f"unexpected expert cell shape {cell.shape}")
                    parts.append(cell.reshape(-1))
                    candidate_keys.extend(
                        (token_family, group_name, denoise, expert)
                        for denoise in range(N_DENOISE)
                        for expert in range(N_EXPERTS)
                    )
            if candidate_axis == 0:
                feature_keys = candidate_keys
            elif candidate_keys != feature_keys:
                raise RuntimeError("expert feature order changed across candidates")
            feature_rows.append(np.concatenate(parts).astype(np.float32))
        matrix = np.stack(feature_rows)
        group_effects = []
        for group in mixed_groups:
            index = np.flatnonzero(groups == group)
            group_effects.append(
                matrix[index[failures[index]]].mean(axis=0)
                - matrix[index[~failures[index]]].mean(axis=0)
            )
        group_effect = np.stack(group_effects)
        observed = group_effect.mean(axis=0)

        permutation_weights = np.zeros(
            (permutations, len(candidates)), dtype=np.float32
        )
        for group in mixed_groups:
            index = np.flatnonzero(groups == group)
            labels = failures[index]
            n_failure = int(labels.sum())
            n_success = int((~labels).sum())
            for draw in range(permutations):
                shuffled = rng.permutation(labels)
                permutation_weights[draw, index] = np.where(
                    shuffled, 1.0 / n_failure, -1.0 / n_success
                ) / len(mixed_groups)
        null_effect = permutation_weights @ matrix
        max_null = np.max(np.abs(null_effect), axis=1)

        bootstrap_weights = np.zeros(
            (bootstraps, len(mixed_groups)), dtype=np.float32
        )
        for draw in range(bootstraps):
            sampled = rng.integers(
                0, len(mixed_groups), size=len(mixed_groups)
            )
            bootstrap_weights[draw] = np.bincount(
                sampled, minlength=len(mixed_groups)
            ) / len(mixed_groups)
        bootstrap_effect = bootstrap_weights @ group_effect
        ci_low, ci_high = np.quantile(
            bootstrap_effect, [0.025, 0.975], axis=0
        )
        unique_workers = np.unique(mixed_workers)
        worker_bootstrap_weights = np.zeros(
            (bootstraps, len(mixed_groups)), dtype=np.float32
        )
        for draw in range(bootstraps):
            sampled_workers = rng.choice(
                unique_workers, size=len(unique_workers), replace=True
            )
            worker_counts = Counter(sampled_workers.tolist())
            weights = np.asarray(
                [worker_counts[int(worker)] for worker in mixed_workers],
                dtype=np.float32,
            )
            worker_bootstrap_weights[draw] = weights / weights.sum()
        worker_bootstrap_effect = worker_bootstrap_weights @ group_effect
        worker_ci_low, worker_ci_high = np.quantile(
            worker_bootstrap_effect, [0.025, 0.975], axis=0
        )
        p_raw = (
            1 + np.sum(np.abs(null_effect) >= np.abs(observed), axis=0)
        ) / (1 + permutations)
        p_max = (
            1 + np.sum(max_null[:, None] >= np.abs(observed)[None, :], axis=0)
        ) / (1 + permutations)
        for feature_axis, key in enumerate(feature_keys):
            rows.append(
                {
                    "prefix_queries": prefix,
                    "token_family": key[0],
                    "layer_group": key[1],
                    "denoise_step": key[2],
                    "expert": key[3],
                    "failure_minus_success_probability": float(observed[feature_axis]),
                    "ci95_low": float(ci_low[feature_axis]),
                    "ci95_high": float(ci_high[feature_axis]),
                    "ci95_worker_low": float(worker_ci_low[feature_axis]),
                    "ci95_worker_high": float(worker_ci_high[feature_axis]),
                    "snapshot_effect_min": float(group_effect[:, feature_axis].min()),
                    "snapshot_effect_max": float(group_effect[:, feature_axis].max()),
                    "snapshot_effect_sd": float(
                        group_effect[:, feature_axis].std(ddof=1)
                        if len(group_effect) > 1
                        else 0.0
                    ),
                    "snapshots_effect_positive": int(
                        np.sum(group_effect[:, feature_axis] > 0.0)
                    ),
                    "snapshots_effect_negative": int(
                        np.sum(group_effect[:, feature_axis] < 0.0)
                    ),
                    "permutation_p_raw": float(p_raw[feature_axis]),
                    "permutation_p_maxT": float(p_max[feature_axis]),
                    "n_mixed_snapshots": len(mixed_groups),
                }
            )
    output = pd.DataFrame(rows).sort_values(
        ["prefix_queries", "permutation_p_maxT", "permutation_p_raw"],
        kind="stable",
    ).reset_index(drop=True)
    probability_conservation = output.groupby(
        ["prefix_queries", "token_family", "layer_group", "denoise_step"]
    )["failure_minus_success_probability"].sum()
    return output, {
        "mixed_snapshots": len(mixed_groups),
        "mixed_workers": len(np.unique(mixed_workers)),
        "cells_per_prefix": len(output) // len(PREFIXES),
        "maxT_family": "all token_family x layer_group x denoise_step x expert cells within each prefix",
        "max_abs_sum_of_32_expert_effects": float(
            np.abs(probability_conservation).max()
        ),
        "tail_features_used": False,
    }


def frozen_data_split(candidate: Candidate) -> str:
    validation_start = FROZEN_SIGNAL["validation_start_by_worker"]
    if candidate.snapshot <= int(FROZEN_SIGNAL["discovery_max_snapshot"]):
        return "discovery"
    if candidate.snapshot >= int(validation_start[candidate.worker]):
        return "validation"
    return "transition_excluded"


def summarize_frozen_feature(
    frame: pd.DataFrame,
    *,
    value_column: str,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    """Directed within-snapshot tests for one already frozen scalar feature."""
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    for split in ("discovery", "transition_excluded", "validation"):
        selected = frame[frame["split"] == split]
        groups = list(selected["snapshot_key"].drop_duplicates())
        group_data = []
        for group in groups:
            group_rows = selected[selected["snapshot_key"] == group]
            group_data.append(
                (
                    group_rows[value_column].to_numpy(dtype=np.float64),
                    group_rows["failure"].to_numpy(dtype=bool),
                )
            )
        mixed_data = [data for data in group_data if len(np.unique(data[1])) == 2]
        mean_effects = np.asarray(
            [
                values[failures].mean() - values[~failures].mean()
                for values, failures in mixed_data
            ],
            dtype=np.float64,
        )
        quartile_effects = []
        for values, failures in mixed_data:
            order = np.argsort(values, kind="stable")
            take = max(1, len(order) // 4)
            quartile_effects.append(
                float(failures[order[-take:]].mean() - failures[order[:take]].mean())
            )
        quartile_effects_array = np.asarray(quartile_effects, dtype=np.float64)
        observed_mean = (
            float(mean_effects.mean()) if len(mean_effects) else float("nan")
        )
        observed_quartile = (
            float(quartile_effects_array.mean())
            if len(quartile_effects_array)
            else float("nan")
        )
        null_mean = np.full(permutations, np.nan, dtype=np.float64)
        null_quartile = np.full(permutations, np.nan, dtype=np.float64)
        if mixed_data:
            for draw in range(permutations):
                draw_mean = []
                draw_quartile = []
                for values, failures in mixed_data:
                    shuffled = rng.permutation(failures)
                    draw_mean.append(
                        float(values[shuffled].mean() - values[~shuffled].mean())
                    )
                    order = np.argsort(values, kind="stable")
                    take = max(1, len(order) // 4)
                    draw_quartile.append(
                        float(
                            shuffled[order[-take:]].mean()
                            - shuffled[order[:take]].mean()
                        )
                    )
                null_mean[draw] = float(np.mean(draw_mean))
                null_quartile[draw] = float(np.mean(draw_quartile))

        def bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
            if not len(values):
                return float("nan"), float("nan")
            draws = np.empty(bootstraps, dtype=np.float64)
            for draw in range(bootstraps):
                draws[draw] = float(
                    rng.choice(values, size=len(values), replace=True).mean()
                )
            low, high = np.quantile(draws, [0.025, 0.975])
            return float(low), float(high)

        mean_ci = bootstrap_interval(mean_effects)
        quartile_ci = bootstrap_interval(quartile_effects_array)
        if mixed_data:
            mean_p = float(
                (1 + np.sum(null_mean >= observed_mean)) / (1 + permutations)
            )
            quartile_p = float(
                (1 + np.sum(null_quartile >= observed_quartile))
                / (1 + permutations)
            )
        else:
            mean_p = float("nan")
            quartile_p = float("nan")

        snapshot_gains = []
        selected_successes = []
        for values, failures in group_data:
            chosen_axis = int(np.argmin(values))
            selected_successes.append(float(not failures[chosen_axis]))
            snapshot_gains.append(
                float((not failures[chosen_axis]) - (~failures).mean())
            )
        gain_values = np.asarray(snapshot_gains, dtype=np.float64)
        observed_gain = (
            float(gain_values.mean()) if len(gain_values) else float("nan")
        )
        null_gain = np.full(permutations, np.nan, dtype=np.float64)
        if group_data:
            for draw in range(permutations):
                gains = []
                for values, failures in group_data:
                    outcomes = rng.permutation((~failures).astype(np.float64))
                    chosen_axis = int(np.argmin(values))
                    gains.append(float(outcomes[chosen_axis] - outcomes.mean()))
                null_gain[draw] = float(np.mean(gains))
            gain_p = float(
                (1 + np.sum(null_gain >= observed_gain)) / (1 + permutations)
            )
        else:
            gain_p = float("nan")
        gain_ci = bootstrap_interval(gain_values)
        top_rate = float("nan")
        bottom_rate = float("nan")
        if mixed_data:
            top_outcomes = []
            bottom_outcomes = []
            for values, failures in mixed_data:
                order = np.argsort(values, kind="stable")
                take = max(1, len(order) // 4)
                top_outcomes.extend(failures[order[-take:]].tolist())
                bottom_outcomes.extend(failures[order[:take]].tolist())
            top_rate = float(np.mean(top_outcomes))
            bottom_rate = float(np.mean(bottom_outcomes))
        summary_rows.append(
            {
                "split": split,
                "branches": len(selected),
                "snapshots": len(groups),
                "mixed_snapshots": len(mixed_data),
                "successes": int((~selected["failure"]).sum()),
                "failures": int(selected["failure"].sum()),
                "failure_minus_success_value": observed_mean,
                "mean_effect_ci95_low": mean_ci[0],
                "mean_effect_ci95_high": mean_ci[1],
                "mean_effect_permutation_p_one_sided": mean_p,
                "top_quartile_failure_rate": top_rate,
                "bottom_quartile_failure_rate": bottom_rate,
                "quartile_failure_rate_difference": observed_quartile,
                "quartile_effect_ci95_low": quartile_ci[0],
                "quartile_effect_ci95_high": quartile_ci[1],
                "quartile_effect_permutation_p_one_sided": quartile_p,
                "lowest_value_selected_success_rate": float(
                    np.mean(selected_successes)
                )
                if selected_successes
                else float("nan"),
                "random_success_rate": float((~selected["failure"]).mean())
                if len(selected)
                else float("nan"),
                "selection_gain": observed_gain,
                "selection_gain_ci95_low": gain_ci[0],
                "selection_gain_ci95_high": gain_ci[1],
                "selection_gain_permutation_p_one_sided": gain_p,
            }
        )
    return pd.DataFrame(summary_rows)


def frozen_signal_validation(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    hb_routes: dict[int, np.ndarray],
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate one pilot-frozen q0 expert coordinate on later snapshots only."""
    labels = physical.set_index("episode_id")["failure"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        hb = hb_routes[candidate.episode_id]
        probability = float(
            hb[
                : int(FROZEN_SIGNAL["prefix_queries"]),
                LAYER_GROUPS[str(FROZEN_SIGNAL["layer_group"])],
                int(FROZEN_SIGNAL["denoise_step"]),
                1:,
                int(FROZEN_SIGNAL["expert"]),
            ].mean()
        )
        rows.append(
            {
                "split": frozen_data_split(candidate),
                "worker": candidate.worker,
                "snapshot": candidate.snapshot,
                "snapshot_key": candidate.snapshot_key,
                "candidate": candidate.candidate,
                "episode_id": candidate.episode_id,
                "failure": bool(labels.loc[candidate.episode_id]),
                "frozen_expert_probability": probability,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["split", "worker", "snapshot", "candidate"], kind="stable"
    ).reset_index(drop=True)
    summary = summarize_frozen_feature(
        frame,
        value_column="frozen_expert_probability",
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed + 1701,
    )
    audit = {
        "feature": FROZEN_SIGNAL,
        "discovery_definition": "snapshot_index <= 2 for every worker",
        "transition_definition": (
            "snapshots already in flight or outcome-visible when the coordinate was frozen"
        ),
        "validation_definition": "first not-yet-started boundary per worker and later",
        "validation_is_single_coordinate": True,
        "validation_direction_was_frozen": True,
        "tail_features_used": False,
    }
    return frame, summary, audit


def frozen_state_signal_validation(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    cell_frame: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Validate the second pilot-frozen q0 state-route statistic."""
    selected = cell_frame[
        (cell_frame["prefix_queries"] == FROZEN_STATE_SIGNAL["prefix_queries"])
        & (cell_frame["token_family"] == FROZEN_STATE_SIGNAL["token_family"])
        & (cell_frame["layer_group"] == FROZEN_STATE_SIGNAL["layer_group"])
        & (cell_frame["denoise_step"] == FROZEN_STATE_SIGNAL["denoise_step"])
        & (cell_frame["metric"] == FROZEN_STATE_SIGNAL["metric"])
    ][["episode_id", "value"]]
    if len(selected) != len(candidates) or selected["episode_id"].nunique() != len(
        candidates
    ):
        raise ValueError("frozen state signal does not have exactly one row per candidate")
    values = selected.set_index("episode_id")["value"]
    labels = physical.set_index("episode_id")["failure"]
    frame = pd.DataFrame(
        [
            {
                "split": frozen_data_split(candidate),
                "worker": candidate.worker,
                "snapshot": candidate.snapshot,
                "snapshot_key": candidate.snapshot_key,
                "candidate": candidate.candidate,
                "episode_id": candidate.episode_id,
                "failure": bool(labels.loc[candidate.episode_id]),
                "frozen_state_top1_mass": float(values.loc[candidate.episode_id]),
            }
            for candidate in candidates
        ]
    ).sort_values(["split", "worker", "snapshot", "candidate"], kind="stable")
    frame = frame.reset_index(drop=True)
    summary = summarize_frozen_feature(
        frame,
        value_column="frozen_state_top1_mass",
        permutations=permutations,
        bootstraps=bootstraps,
        seed=seed + 2204,
    )
    audit = {
        "feature": FROZEN_STATE_SIGNAL,
        "discovery_definition": "snapshot_index <= 2 for every worker",
        "transition_definition": (
            "snapshots already in flight or outcome-visible when the coordinate was frozen"
        ),
        "validation_definition": "first not-yet-started boundary per worker and later",
        "validation_direction_was_frozen": True,
        "tail_features_used": False,
    }
    return frame, summary, audit


def quartile_effect(
    values: np.ndarray, failures: np.ndarray, groups: np.ndarray
) -> tuple[float, float, float, int, int]:
    top_y: list[float] = []
    bottom_y: list[float] = []
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        if len(index) < 4 or len(np.unique(failures[index])) < 2:
            continue
        order = index[np.argsort(values[index], kind="stable")]
        take = max(1, len(index) // 4)
        bottom_y.extend(failures[order[:take]].tolist())
        top_y.extend(failures[order[-take:]].tolist())
    if not top_y or not bottom_y:
        return float("nan"), float("nan"), float("nan"), 0, 0
    top_rate = float(np.mean(top_y))
    bottom_rate = float(np.mean(bottom_y))
    return top_rate - bottom_rate, top_rate, bottom_rate, len(top_y), len(bottom_y)


def screen_route_cells(
    cell_frame: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = []
    for prefix in PREFIXES:
        selected = cell_frame[cell_frame["prefix_queries"] == prefix]
        key_columns = ["token_family", "layer_group", "denoise_step", "metric"]
        feature_keys = list(selected[key_columns].drop_duplicates().itertuples(index=False, name=None))
        candidate = selected[["snapshot_key", "episode_id", "failure"]].drop_duplicates()
        candidate = candidate.sort_values(["snapshot_key", "episode_id"]).reset_index(drop=True)
        failures = candidate["failure"].to_numpy(dtype=np.float64)
        groups = candidate["snapshot_key"].to_numpy()
        matrices = []
        for feature_key in feature_keys:
            rows = selected[
                (selected["layer_group"] == feature_key[1])
                & (selected["denoise_step"] == feature_key[2])
                & (selected["metric"] == feature_key[3])
                & (selected["token_family"] == feature_key[0])
            ][["snapshot_key", "episode_id", "value"]]
            merged = candidate.merge(rows, on=["snapshot_key", "episode_id"], validate="one_to_one")
            values = merged["value"].to_numpy(dtype=np.float64)
            matrices.append(values)
        matrix = np.stack(matrices, axis=1)
        n_feature = len(feature_keys)
        n_row = len(candidate)
        top_mask = np.zeros((n_feature, n_row), dtype=np.float64)
        bottom_mask = np.zeros((n_feature, n_row), dtype=np.float64)
        valid_groups = []
        for group in np.unique(groups):
            index = np.flatnonzero(groups == group)
            if len(index) < 4 or len(np.unique(failures[index])) < 2:
                continue
            valid_groups.append(group)
            take = max(1, len(index) // 4)
            order = np.argsort(matrix[index], axis=0, kind="stable").T
            feature_axis = np.arange(n_feature)[:, None]
            bottom_mask[feature_axis, index[order[:, :take]]] = 1.0
            top_mask[feature_axis, index[order[:, -take:]]] = 1.0
        top_n = top_mask.sum(axis=1)
        bottom_n = bottom_mask.sum(axis=1)
        if np.any(top_n == 0) or np.any(bottom_n == 0):
            raise RuntimeError(f"prefix {prefix}: no mixed snapshot supports quartile screen")
        top_rate_observed = (top_mask @ failures) / top_n
        bottom_rate_observed = (bottom_mask @ failures) / bottom_n
        observed_difference = top_rate_observed - bottom_rate_observed

        permutation_effect = np.empty(
            (permutations, n_feature), dtype=np.float32
        )
        for draw in range(permutations):
            shuffled = failures.copy()
            for group in np.unique(groups):
                index = np.flatnonzero(groups == group)
                shuffled[index] = rng.permutation(shuffled[index])
            permutation_effect[draw] = (
                (top_mask @ shuffled) / top_n
                - (bottom_mask @ shuffled) / bottom_n
            )
        max_null = np.nanmax(np.abs(permutation_effect), axis=1)

        group_top_sum = np.zeros((n_feature, len(valid_groups)), dtype=np.float64)
        group_bottom_sum = np.zeros_like(group_top_sum)
        group_top_n = np.zeros_like(group_top_sum)
        group_bottom_n = np.zeros_like(group_top_sum)
        for group_axis, group in enumerate(valid_groups):
            index = np.flatnonzero(groups == group)
            group_top_sum[:, group_axis] = top_mask[:, index] @ failures[index]
            group_bottom_sum[:, group_axis] = bottom_mask[:, index] @ failures[index]
            group_top_n[:, group_axis] = top_mask[:, index].sum(axis=1)
            group_bottom_n[:, group_axis] = bottom_mask[:, index].sum(axis=1)
        bootstrap_effect = np.empty(
            (bootstraps, n_feature), dtype=np.float32
        )
        for draw in range(bootstraps):
            sampled = rng.integers(0, len(valid_groups), size=len(valid_groups))
            bootstrap_effect[draw] = (
                group_top_sum[:, sampled].sum(axis=1)
                / group_top_n[:, sampled].sum(axis=1)
                - group_bottom_sum[:, sampled].sum(axis=1)
                / group_bottom_n[:, sampled].sum(axis=1)
            )
        for feature_index, feature_key in enumerate(feature_keys):
            difference = float(observed_difference[feature_index])
            top_rate = float(top_rate_observed[feature_index])
            bottom_rate = float(bottom_rate_observed[feature_index])
            valid = permutation_effect[:, feature_index]
            p_raw = float(
                (1 + np.sum(np.abs(valid) >= abs(difference))) / (1 + np.sum(np.isfinite(valid)))
            )
            p_max = float((1 + np.sum(max_null >= abs(difference))) / (1 + len(max_null)))
            ci = np.nanquantile(bootstrap_effect[:, feature_index], [0.025, 0.975])
            results.append(
                {
                    "prefix_queries": prefix,
                    "token_family": feature_key[0],
                    "layer_group": feature_key[1],
                    "denoise_step": feature_key[2],
                    "metric": feature_key[3],
                    "top_quartile_failure_rate": top_rate,
                    "bottom_quartile_failure_rate": bottom_rate,
                    "failure_rate_difference": difference,
                    "ci95_low": float(ci[0]),
                    "ci95_high": float(ci[1]),
                    "permutation_p_raw": p_raw,
                    "permutation_p_maxT": p_max,
                    "n_top": int(top_n[feature_index]),
                    "n_bottom": int(bottom_n[feature_index]),
                }
            )
    output = pd.DataFrame(results)
    return output.sort_values(
        ["prefix_queries", "permutation_p_maxT", "permutation_p_raw"], kind="stable"
    ).reset_index(drop=True)


def group_cv_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    use_pca: bool,
    seed: int,
) -> np.ndarray:
    unique_groups = np.unique(groups)
    splits = min(5, len(unique_groups))
    if splits < 2 or len(np.unique(labels)) < 2:
        return np.full(len(labels), np.nan)
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=splits)
    for train, test in splitter.split(matrix, labels, groups):
        if len(np.unique(labels[train])) < 2:
            predictions[test] = float(labels[train].mean())
            continue
        steps: list[Any] = [StandardScaler()]
        if use_pca:
            components = min(12, matrix.shape[1], max(2, len(train) // 5))
            steps.append(PCA(n_components=components, whiten=False, random_state=seed))
        steps.append(
            LogisticRegression(
                C=0.1,
                max_iter=3000,
                class_weight=None,
                random_state=seed,
            )
        )
        model = make_pipeline(*steps)
        model.fit(matrix[train], labels[train])
        predictions[test] = model.predict_proba(matrix[test])[:, 1]
    return predictions


def center_within_groups(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Remove the common snapshot/phase component using no outcome labels."""
    centered = np.asarray(matrix, dtype=np.float32).copy()
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        centered[index] -= centered[index].mean(axis=0, keepdims=True)
    return centered


def grouped_brier_delta_interval(
    labels: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    groups: np.ndarray,
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Reference error minus model error, resampling whole CV units."""
    per_row = (labels - reference) ** 2 - (labels - prediction) ** 2
    return grouped_mean_interval(
        per_row, groups, bootstraps=bootstraps, rng=rng
    )


def grouped_mean_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Mean and whole-group bootstrap interval using vectorized group sums."""
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    group_sum = np.bincount(inverse, weights=values, minlength=len(unique))
    group_count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sampled = rng.integers(
        0, len(unique), size=(bootstraps, len(unique)), endpoint=False
    )
    draws = group_sum[sampled].sum(axis=1) / group_count[sampled].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def model_comparisons(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    banks: dict[int, dict[str, np.ndarray]],
    *,
    bootstraps: int,
    seed: int,
    cv_scheme: str,
    include_models: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = physical.set_index("episode_id").loc[
        [item.episode_id for item in candidates], "failure"
    ].to_numpy(dtype=np.int64)
    snapshot_groups = np.asarray([item.snapshot_key for item in candidates])
    if cv_scheme == "grouped_5fold_snapshot":
        cv_groups = snapshot_groups
    elif cv_scheme == "leave_one_worker_out":
        cv_groups = np.asarray([item.worker for item in candidates])
    else:
        raise ValueError(f"unknown CV scheme: {cv_scheme}")
    cv_folds = min(5, len(np.unique(cv_groups)))
    rows = []
    prediction_rows = []
    all_predictions: dict[tuple[int, str], np.ndarray] = {}
    for prefix in PREFIXES:
        bank = banks[prefix]
        action_moe = np.c_[
            bank["moe_action_summary"], bank["moe_action_full"]
        ]
        action_state_moe = np.c_[action_moe, bank["moe_state_full"]]
        absolute_matrices = {
            "snapshot_rate_only": np.ones((len(candidates), 1), dtype=np.float32),
            "noise": bank["noise_raw"],
            "action": bank["action_raw"],
            "noise+action": np.c_[bank["noise_raw"], bank["action_raw"]],
            "moe_action": action_moe,
            "moe_state": bank["moe_state_full"],
            "moe_action+state": action_state_moe,
            "noise+action+moe_action": np.c_[
                bank["noise_raw"], bank["action_raw"], action_moe
            ],
            "noise+action+moe_action+state": np.c_[
                bank["noise_raw"],
                bank["action_raw"],
                action_state_moe,
            ],
        }
        matrices = dict(absolute_matrices)
        for name, matrix in absolute_matrices.items():
            if name == "snapshot_rate_only":
                continue
            matrices[f"{name}_within_snapshot"] = center_within_groups(
                matrix, snapshot_groups
            )
        if include_models is not None:
            matrices = {
                name: matrix
                for name, matrix in matrices.items()
                if name in include_models
            }
            missing_models = include_models - set(matrices)
            if missing_models:
                raise ValueError(f"unknown requested models: {sorted(missing_models)}")
        predictions_by_model: dict[str, np.ndarray] = {}
        for model_axis, (name, matrix) in enumerate(matrices.items()):
            if name == "snapshot_rate_only":
                prediction = np.full(len(labels), np.nan, dtype=np.float64)
                splitter = GroupKFold(n_splits=cv_folds)
                for train, test in splitter.split(matrix, labels, cv_groups):
                    prediction[test] = labels[train].mean()
            else:
                prediction = group_cv_predictions(
                    matrix,
                    labels,
                    cv_groups,
                    use_pca=matrix.shape[1] > 64,
                    seed=seed,
                )
            predictions_by_model[name] = prediction
            all_predictions[(prefix, name)] = prediction
            valid = np.isfinite(prediction)
            brier = float(brier_score_loss(labels[valid], prediction[valid]))
            clipped = np.clip(prediction[valid], 1e-6, 1 - 1e-6)
            loss = float(log_loss(labels[valid], clipped, labels=[0, 1]))
            selection_success = []
            random_success = []
            selected_candidates = []
            selection_units = []
            for group in np.unique(snapshot_groups):
                index = np.flatnonzero(snapshot_groups == group)
                if len(np.unique(labels[index])) < 2:
                    continue
                random_rate = float(np.mean(1 - labels[index]))
                random_success.append(random_rate)
                selection_units.append(
                    int(candidates[index[0]].worker)
                    if cv_scheme == "leave_one_worker_out"
                    else str(group)
                )
                if name == "snapshot_rate_only":
                    # A constant score cannot choose among matched siblings.
                    selection_success.append(random_rate)
                else:
                    chosen = index[int(np.argmin(prediction[index]))]
                    selection_success.append(1 - labels[chosen])
                    selected_candidates.append(int(candidates[chosen].candidate))
            if selection_success:
                selection_delta = np.asarray(selection_success) - np.asarray(random_success)
                selection_rng = np.random.default_rng(
                    seed + 10000 * prefix + 101 * model_axis
                )
                selection_units_array = np.asarray(selection_units)
                selection_interval = grouped_mean_interval(
                    selection_delta,
                    selection_units_array,
                    bootstraps=bootstraps,
                    rng=selection_rng,
                )
                selection_ci = selection_interval[1:]
            else:
                selection_ci = (np.nan, np.nan)
            rows.append(
                {
                    "prefix_queries": prefix,
                    "cv_scheme": cv_scheme,
                    "cv_folds": cv_folds,
                    "model": name,
                    "feature_scope": (
                        "within_snapshot_centered"
                        if name.endswith("_within_snapshot")
                        else "absolute"
                    ),
                    "n_candidates": int(valid.sum()),
                    "n_mixed_snapshots_for_selection": len(selection_success),
                    "brier": brier,
                    "log_loss": loss,
                    "selected_success_rate": float(np.mean(selection_success)) if selection_success else np.nan,
                    "random_success_rate": float(np.mean(random_success)) if random_success else np.nan,
                    "selection_gain": float(np.mean(selection_success) - np.mean(random_success)) if selection_success else np.nan,
                    "selection_gain_ci95_low": float(selection_ci[0]),
                    "selection_gain_ci95_high": float(selection_ci[1]),
                }
            )
            for candidate, label, value in zip(candidates, labels, prediction):
                prediction_rows.append(
                    {
                        "prefix_queries": prefix,
                        "cv_scheme": cv_scheme,
                        "cv_folds": cv_folds,
                        "model": name,
                        "snapshot_key": candidate.snapshot_key,
                        "episode_id": candidate.episode_id,
                        "candidate": candidate.candidate,
                        "failure": bool(label),
                        "predicted_failure": value,
                    }
                )
    comparison = pd.DataFrame(rows)
    rng = np.random.default_rng(seed + 73)
    for prefix in PREFIXES:
        baseline = float(
            comparison[
                (comparison["prefix_queries"] == prefix)
                & (comparison["model"] == "snapshot_rate_only")
            ]["brier"].iloc[0]
        )
        index = comparison["prefix_queries"] == prefix
        comparison.loc[index, "brier_improvement_vs_rate"] = (
            baseline - comparison.loc[index, "brier"]
        )
        comparison.loc[index, "brier_relative_improvement_vs_rate"] = (
            (baseline - comparison.loc[index, "brier"]) / baseline
        )
        for row_index in comparison.index[index]:
            model_name = str(comparison.loc[row_index, "model"])
            prediction = all_predictions[(prefix, model_name)]
            noise_action_reference = (
                "noise+action_within_snapshot"
                if model_name.endswith("_within_snapshot")
                else "noise+action"
            )
            comparison.loc[
                row_index, "noise_action_reference"
            ] = noise_action_reference
            for reference_name, suffix in (
                ("snapshot_rate_only", "rate"),
                (noise_action_reference, "noise_action"),
            ):
                reference = all_predictions[(prefix, reference_name)]
                delta, low, high = grouped_brier_delta_interval(
                    labels,
                    prediction,
                    reference,
                    cv_groups,
                    bootstraps=bootstraps,
                    rng=rng,
                )
                comparison.loc[row_index, f"brier_gain_vs_{suffix}"] = delta
                comparison.loc[row_index, f"brier_gain_vs_{suffix}_ci95_low"] = low
                comparison.loc[row_index, f"brier_gain_vs_{suffix}_ci95_high"] = high
    return comparison, pd.DataFrame(prediction_rows)


def snapshot_difficulty_models(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    trajectories: dict[int, dict[str, np.ndarray]],
    banks: dict[int, dict[str, np.ndarray]],
    targets: list[dict[str, Any]],
    *,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Predict a snapshot's K=16 failure rate from its common q0 phase signal."""
    groups = np.asarray([item.snapshot_key for item in candidates])
    snapshot_keys = np.unique(groups)
    if len(snapshot_keys) < 3:
        return pd.DataFrame(), pd.DataFrame(), {"snapshots": len(snapshot_keys)}
    outcome = physical.set_index("episode_id")["failure"].to_dict()
    target = np.asarray(
        [
            np.mean(
                [outcome[item.episode_id] for item in candidates if item.snapshot_key == key]
            )
            for key in snapshot_keys
        ],
        dtype=np.float64,
    )
    bank = banks[1]
    sim_rows = []
    policy_rows = []
    geometry_rows = []
    output_rows = []
    as_rows = []
    action_moe_rows = []
    state_moe_rows = []
    max_sim_span = 0.0
    max_policy_span = 0.0
    for key in snapshot_keys:
        index = np.flatnonzero(groups == key)
        sim = np.stack(
            [trajectories[candidates[i].episode_id]["control_sim_state"][0] for i in index]
        )
        policy = np.stack(
            [trajectories[candidates[i].episode_id]["policy_state"][0] for i in index]
        )
        max_sim_span = max(max_sim_span, float(np.ptp(sim, axis=0).max()))
        max_policy_span = max(max_policy_span, float(np.ptp(policy, axis=0).max()))
        sim_rows.append(sim.mean(axis=0))
        policy_rows.append(policy.mean(axis=0))
        initial_sim = sim.mean(axis=0)
        initial_policy = policy.mean(axis=0)
        object_positions = np.stack(
            [
                initial_sim[
                    int(target_item["state_lo"]) : int(target_item["state_lo"]) + 3
                ]
                for target_item in targets
            ]
        )
        if len(object_positions) != 2:
            raise ValueError("initial geometry currently expects exactly two objects")
        geometry_rows.append(
            np.r_[
                object_positions.reshape(-1),
                np.linalg.norm(object_positions - initial_policy[:3], axis=1),
                np.linalg.norm(object_positions[0] - object_positions[1]),
                initial_policy[6:8].mean(),
            ]
        )
        output_rows.append(bank["action_raw"][index].mean(axis=0))
        as_rows.append(bank["moe_as_full"][index].mean(axis=0))
        action_moe_rows.append(
            np.c_[
                bank["moe_action_summary"][index],
                bank["moe_action_full"][index],
            ].mean(axis=0)
        )
        state_moe_rows.append(bank["moe_state_full"][index].mean(axis=0))
    sim = np.stack(sim_rows).astype(np.float32)
    policy = np.stack(policy_rows).astype(np.float32)
    geometry = np.stack(geometry_rows).astype(np.float32)
    output = np.stack(output_rows).astype(np.float32)
    as_moe = np.stack(as_rows).astype(np.float32)
    action_moe = np.stack(action_moe_rows).astype(np.float32)
    state_moe = np.stack(state_moe_rows).astype(np.float32)
    snapshot_workers = np.asarray(
        [
            next(item.worker for item in candidates if item.snapshot_key == key)
            for key in snapshot_keys
        ],
        dtype=np.int64,
    )
    matrices = {
        "mean_rate": np.ones((len(snapshot_keys), 1), dtype=np.float32),
        "initial_sim_state": sim,
        "initial_policy_state": policy,
        "initial_geometry": geometry,
        "mean_output_action": output,
        "as_moe": as_moe,
        "mean_moe_action": action_moe,
        "mean_moe_state": state_moe,
        "mean_moe_action+state": np.c_[action_moe, state_moe],
        "geometry+mean_moe_action+state": np.c_[
            geometry, action_moe, state_moe
        ],
        "policy_state+mean_moe_action+state": np.c_[
            policy, action_moe, state_moe
        ],
        "sim_state+mean_moe_action+state": np.c_[sim, action_moe, state_moe],
    }
    cv_schemes = {
        "leave_one_snapshot_out": np.arange(len(snapshot_keys), dtype=np.int64),
        "leave_one_worker_out": snapshot_workers,
    }
    rng = np.random.default_rng(seed + 811)
    rows = []
    prediction_rows = []
    for cv_scheme, cv_groups in cv_schemes.items():
        predictions: dict[str, np.ndarray] = {}
        for name, matrix in matrices.items():
            values = np.full(len(snapshot_keys), np.nan, dtype=np.float64)
            for heldout in np.unique(cv_groups):
                test = np.flatnonzero(cv_groups == heldout)
                train = np.flatnonzero(cv_groups != heldout)
                if name == "mean_rate":
                    values[test] = float(target[train].mean())
                    continue
                model = make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=10.0, solver="lsqr", tol=1e-4),
                )
                model.fit(matrix[train], target[train])
                values[test] = model.predict(matrix[test])
            predictions[name] = np.clip(values, 0.0, 1.0)
        reference_errors = {
            name: (target - predictions[name]) ** 2
            for name in (
                "mean_rate",
                "initial_policy_state",
                "initial_geometry",
                "initial_sim_state",
            )
        }
        unique_units = np.unique(cv_groups)
        for name, prediction in predictions.items():
            error = (target - prediction) ** 2
            row = {
                "cv_scheme": cv_scheme,
                "model": name,
                "snapshots": len(snapshot_keys),
                "heldout_units": len(unique_units),
                "ridge_alpha": 10.0 if name != "mean_rate" else np.nan,
                "failure_rate_mse": float(error.mean()),
                "failure_rate_mae": float(np.abs(target - prediction).mean()),
            }
            for reference_name, reference_error in reference_errors.items():
                per_snapshot_gain = reference_error - error
                gain_interval = grouped_mean_interval(
                    per_snapshot_gain,
                    cv_groups,
                    bootstraps=bootstraps,
                    rng=rng,
                )
                suffix = reference_name.removeprefix("initial_")
                row[f"mse_gain_vs_{suffix}"] = gain_interval[0]
                row[f"mse_gain_vs_{suffix}_ci95_low"] = gain_interval[1]
                row[f"mse_gain_vs_{suffix}_ci95_high"] = gain_interval[2]
            rows.append(row)
            for key, worker, observed, predicted in zip(
                snapshot_keys, snapshot_workers, target, prediction
            ):
                prediction_rows.append(
                    {
                        "cv_scheme": cv_scheme,
                        "snapshot_key": key,
                        "worker": int(worker),
                        "model": name,
                        "observed_failure_rate": observed,
                        "predicted_failure_rate": predicted,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["cv_scheme", "failure_rate_mse"])
        .reset_index(drop=True),
        pd.DataFrame(prediction_rows),
        {
            "snapshots": len(snapshot_keys),
            "k_per_snapshot": 16,
            "initial_sim_state_max_within_snapshot_span": max_sim_span,
            "initial_policy_state_max_within_snapshot_span": max_policy_span,
            "cv": list(cv_schemes),
            "ridge_alpha_fixed": 10.0,
            "ridge_solver": "lsqr",
            "initial_geometry_features": (
                "two object xyz positions; EEF-to-object distances; "
                "object separation; mean gripper position"
            ),
        },
    )


def subtype_model_comparisons(
    candidates: list[Candidate],
    physical: pd.DataFrame,
    banks: dict[int, dict[str, np.ndarray]],
    *,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    """Separate one physical failure mechanism from the other failures."""
    ordered = physical.set_index("episode_id").loc[
        [item.episode_id for item in candidates]
    ].reset_index()
    failure_index = np.flatnonzero(ordered["failure"].to_numpy(dtype=bool))
    groups_all = np.asarray([item.snapshot_key for item in candidates])
    workers_all = np.asarray([item.worker for item in candidates], dtype=np.int64)
    rows = []
    rng = np.random.default_rng(seed + 1901)
    label_columns = [column for column in physical if column.startswith("label_")]
    for label_column in label_columns:
        target_all = ordered[label_column].to_numpy(dtype=np.int64)
        target = target_all[failure_index]
        if int(target.sum()) < 6 or int((1 - target).sum()) < 6:
            continue
        snapshot_groups = groups_all[failure_index]
        worker_groups = workers_all[failure_index]
        if len(np.unique(snapshot_groups)) < 3:
            continue
        for prefix in PREFIXES:
            bank = banks[prefix]
            action_moe_all = np.c_[
                bank["moe_action_summary"], bank["moe_action_full"]
            ]
            action_state_moe_all = np.c_[
                action_moe_all, bank["moe_state_full"]
            ]
            absolute_all = {
                "noise+action": np.c_[bank["noise_raw"], bank["action_raw"]],
                "moe_action": action_moe_all,
                "moe_state": bank["moe_state_full"],
                "moe_action+state": action_state_moe_all,
                "noise+action+moe_action": np.c_[
                    bank["noise_raw"], bank["action_raw"], action_moe_all
                ],
                "noise+action+moe_action+state": np.c_[
                    bank["noise_raw"], bank["action_raw"], action_state_moe_all
                ],
            }
            matrices = {
                "failure_rate_only": np.ones(
                    (len(failure_index), 1), dtype=np.float32
                ),
                **{
                    name: matrix[failure_index]
                    for name, matrix in absolute_all.items()
                },
                **{
                    f"{name}_within_snapshot": center_within_groups(
                        matrix, groups_all
                    )[failure_index]
                    for name, matrix in absolute_all.items()
                },
            }
            for cv_scheme, groups in (
                ("grouped_5fold_snapshot", snapshot_groups),
                ("leave_one_worker_out", worker_groups),
            ):
                unique_groups = np.unique(groups)
                if len(unique_groups) < 2:
                    continue
                predictions: dict[str, np.ndarray] = {}
                for name, matrix in matrices.items():
                    if name == "failure_rate_only":
                        prediction = np.full(len(target), np.nan, dtype=np.float64)
                        splitter = GroupKFold(
                            n_splits=min(5, len(unique_groups))
                        )
                        for train, test in splitter.split(matrix, target, groups):
                            prediction[test] = target[train].mean()
                    else:
                        prediction = group_cv_predictions(
                            matrix,
                            target,
                            groups,
                            use_pca=matrix.shape[1] > 64,
                            seed=seed,
                        )
                    if np.any(~np.isfinite(prediction)):
                        raise ValueError(
                            f"non-finite subtype prediction for {label_column}, "
                            f"{prefix}, {cv_scheme}, {name}"
                        )
                    predictions[name] = prediction
                baseline_prediction = predictions["failure_rate_only"]
                baseline_brier = float(
                    brier_score_loss(target, baseline_prediction)
                )
                for name, prediction in predictions.items():
                    brier = float(brier_score_loss(target, prediction))
                    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
                    noise_action_name = (
                        "noise+action_within_snapshot"
                        if name.endswith("_within_snapshot")
                        else "noise+action"
                    )
                    noise_prediction = predictions[noise_action_name]
                    gain_rate = grouped_brier_delta_interval(
                        target,
                        prediction,
                        baseline_prediction,
                        groups,
                        bootstraps=bootstraps,
                        rng=rng,
                    )
                    gain_noise = grouped_brier_delta_interval(
                        target,
                        prediction,
                        noise_prediction,
                        groups,
                        bootstraps=bootstraps,
                        rng=rng,
                    )
                    rows.append(
                        {
                            "failure_type": label_column.removeprefix("label_"),
                            "prefix_queries": prefix,
                            "cv_scheme": cv_scheme,
                            "cv_folds": min(5, len(unique_groups)),
                            "model": name,
                            "feature_scope": (
                                "within_snapshot_centered"
                                if name.endswith("_within_snapshot")
                                else "absolute"
                            ),
                            "n_failure_branches": len(target),
                            "n_type_positive": int(target.sum()),
                            "heldout_units": len(unique_groups),
                            "brier": brier,
                            "log_loss": float(
                                log_loss(target, clipped, labels=[0, 1])
                            ),
                            "noise_action_reference": noise_action_name,
                            "brier_improvement_vs_rate": gain_rate[0],
                            "brier_improvement_vs_rate_ci95_low": gain_rate[1],
                            "brier_improvement_vs_rate_ci95_high": gain_rate[2],
                            "brier_relative_improvement_vs_rate": (
                                gain_rate[0] / baseline_brier
                            ),
                            "brier_improvement_vs_noise_action": gain_noise[0],
                            "brier_improvement_vs_noise_action_ci95_low": (
                                gain_noise[1]
                            ),
                            "brier_improvement_vs_noise_action_ci95_high": (
                                gain_noise[2]
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def subtype_route_screens(
    cell_frame: pd.DataFrame,
    physical: pd.DataFrame,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    labels = physical.set_index("episode_id")
    failure_ids = labels.index[labels["failure"]]
    outputs = []
    for offset, label_column in enumerate(
        column for column in physical if column.startswith("label_")
    ):
        positive = int(labels.loc[failure_ids, label_column].sum())
        negative = int(len(failure_ids) - positive)
        if positive < 6 or negative < 6:
            continue
        selected = cell_frame[cell_frame["episode_id"].isin(failure_ids)].copy()
        mapping = labels[label_column].to_dict()
        selected["failure"] = selected["episode_id"].map(mapping).astype(bool)
        selected["success"] = ~selected["failure"]
        branch_labels = selected[
            ["snapshot_key", "episode_id", "failure"]
        ].drop_duplicates()
        mixed_groups = branch_labels.groupby("snapshot_key")["failure"].nunique()
        if not bool((mixed_groups >= 2).any()):
            continue
        result = screen_route_cells(
            selected,
            permutations=permutations,
            bootstraps=bootstraps,
            seed=seed + 1009 * (offset + 1),
        )
        result.insert(0, "failure_type", label_column.removeprefix("label_"))
        outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def snapshot_summary(physical: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for snapshot_key, group in physical.groupby("snapshot_key", sort=True):
        counts = Counter(group.loc[group["failure"], "primary_failure_type"])
        rows.append(
            {
                "snapshot_key": snapshot_key,
                "worker": int(group["worker"].iloc[0]),
                "init_state_id": int(group["init_state_id"].iloc[0]),
                "snapshot": int(group["snapshot"].iloc[0]),
                "successes": int(group["success"].sum()),
                "failures": int(group["failure"].sum()),
                "mixed": bool(group["success"].any() and group["failure"].any()),
                "failure_types": json.dumps(counts, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def failure_mode_coverage(physical: pd.DataFrame) -> dict[str, Any]:
    failures = physical[physical["failure"]].copy()
    classic = (
        failures["label_stagnation"].to_numpy(dtype=bool)
        | failures["label_loop_or_cycling"].to_numpy(dtype=bool)
    )
    uncovered = failures.loc[~classic]
    return {
        "failures": len(failures),
        "stagnation": int(failures["label_stagnation"].sum()),
        "loop_or_cycling": int(failures["label_loop_or_cycling"].sum()),
        "stagnation_or_loop": int(classic.sum()),
        "stagnation_or_loop_fraction": float(classic.mean()) if len(classic) else None,
        "not_stagnation_or_loop": int((~classic).sum()),
        "uncovered_primary_types": {
            str(name): int(count)
            for name, count in uncovered["primary_failure_type"]
            .value_counts()
            .items()
        },
        "uncovered_episode_ids": uncovered["episode_id"].astype(int).tolist(),
    }


def plot_outputs(
    physical: pd.DataFrame,
    snapshots: pd.DataFrame,
    screens: pd.DataFrame,
    models: pd.DataFrame,
    out_dir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    counts = physical.loc[physical["failure"], "primary_failure_type"].value_counts()
    fig, axis = plt.subplots(figsize=(8, 4.5))
    counts.sort_values().plot.barh(ax=axis, color="#3f6f8f")
    axis.set_xlabel("failed branches")
    axis.set_ylabel("")
    axis.set_title("Physical failure taxonomy (MoE-blind labels)")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_taxonomy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#36688d", "#b04f4f", "#4f7f52", "#8a6d3b"]
    for color, (worker, group) in zip(
        colors, snapshots.groupby("worker", sort=True)
    ):
        ordered = group.sort_values("snapshot")
        axis.plot(
            ordered["snapshot"],
            ordered["successes"] / 16.0,
            marker="o",
            linewidth=1.8,
            label=f"worker {int(worker)} / init {int(ordered['init_state_id'].iloc[0])}",
            color=color,
        )
    axis.set_xlabel("rolling trunk query")
    axis.set_ylabel("success fraction among K=16")
    axis.set_ylim(-0.04, 1.04)
    axis.set_xticks(sorted(snapshots["snapshot"].unique()))
    axis.set_title("Matched-branch success rate along each rolling trunk")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "rolling_snapshot_success_rates.png", dpi=180)
    plt.close(fig)

    consensus = screens[
        (screens["metric"] == "consensus_distance")
        & (screens["prefix_queries"] == 1)
        & (screens["token_family"] == "action")
    ]
    if len(consensus):
        matrix = consensus.pivot(
            index="layer_group", columns="denoise_step", values="failure_rate_difference"
        ).reindex(list(LAYER_GROUPS))
        fig, axis = plt.subplots(figsize=(9, 2.8))
        image = axis.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set_xticks(range(N_DENOISE), range(N_DENOISE))
        axis.set_xlabel("denoise step (0=noisiest, 9=latest)")
        axis.set_title("q0 route-consensus: top minus bottom quartile failure rate")
        fig.colorbar(image, ax=axis, label="failure-rate difference")
        fig.tight_layout()
        fig.savefig(out_dir / "q0_consensus_layer_denoise.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 8.0))
    plotted_models = models[
        models["cv_scheme"] == "grouped_5fold_snapshot"
    ]
    table = plotted_models.pivot(
        index="model", columns="prefix_queries", values="brier"
    )
    table.plot.bar(ax=axis, color=["#607d8b", "#b05a47"])
    axis.set_ylabel("held-out Brier error (lower is better)")
    axis.set_xlabel("")
    axis.set_title("Early failure prediction without tail features")
    axis.legend(title="prefix queries")
    fig.tight_layout()
    fig.savefig(out_dir / "early_model_brier.png", dpi=180)
    plt.close(fig)


def build_report(
    physical: pd.DataFrame,
    snapshots: pd.DataFrame,
    screens: pd.DataFrame,
    expert_effects: pd.DataFrame,
    models: pd.DataFrame,
    difficulty_models: pd.DataFrame,
    subtype_models: pd.DataFrame,
    subtype_screens: pd.DataFrame,
    frozen_signal_summary: pd.DataFrame,
    frozen_signal_audit: dict[str, Any],
    route_audit: dict[str, Any],
    feature_audit: dict[str, Any],
    coverage: dict[str, Any],
    candidate_id_audit: dict[str, Any],
    out_dir: Path,
) -> None:
    failures = physical[physical["failure"]]
    counts = failures["primary_failure_type"].value_counts()
    lines = [
        "# Rolling-star K=16 experiment",
        "",
        "## Dataset",
        "",
        f"- {len(physical)} terminal branches from {len(snapshots)} committed snapshots; "
        f"{int(physical['success'].sum())} success and {int(physical['failure'].sum())} failure.",
        f"- {int(snapshots['mixed'].sum())} snapshots contain matched success/failure siblings.",
        f"- {route_audit['candidate_route_rows']} candidate query rows with full HB probabilities; hidden state stored: {route_audit['store_hidden']}.",
        "- Every candidate in a snapshot starts from the same exact simulator/controller state and policy input; only its recorded flow-noise stream changes.",
        f"- Candidate-ID negative control: max failure-rate deviation {candidate_id_audit['max_abs_candidate_rate_deviation']:.3f}, "
        f"within-snapshot permutation p={candidate_id_audit['within_snapshot_permutation_p']:.4g}.",
        "",
        "## Physical failure labels",
        "",
        "Labels use only dense simulator, EEF, gripper, action, and success trajectories. MoE routes are not read until after labels are frozen.",
        "",
    ]
    for name, count in counts.items():
        lines.append(f"- `{name}`: {int(count)}")
    lines.extend(
        [
            "",
            f"Stagnation or loop/cycling covers {coverage['stagnation_or_loop']}/{coverage['failures']} failures; "
            f"{coverage['not_stagnation_or_loop']} failures require other physical mechanisms: "
            f"{json.dumps(coverage['uncovered_primary_types'], sort_keys=True)}.",
            "",
            "## Early MoE signal",
            "",
            "No AUC is reported. q0 and q0-q2 analyses exclude the rollout tail and therefore cannot exploit timeout/remaining-time sentinels.",
            "",
            f"The q0 state-token route maximum probability span within matched siblings is {feature_audit['q0_state_within_snapshot_max_abs_probability_span']:.3g}.",
            f"The q0 AS-MoE within-snapshot span is {feature_audit['q0_as_within_snapshot_max_abs_probability_span']:.3g}; a zero span confirms that AS is a constant negative control on this task.",
            "",
            "Pilot-frozen two-signal family (the transition set is excluded from independent validation):",
            "",
            markdown_table(frozen_signal_summary),
            "",
            "Both coordinates and positive failure directions were frozen before their validation boundaries; "
            "only `validation` rows are prospective tests, and `primary_p_bonferroni` corrects the two-signal family.",
            "",
            "Cross-validated models (snapshot-grouped 5-fold and core leave-one-worker-out checks):",
            "",
            markdown_table(models),
            "",
            "Snapshot difficulty (one row per rolling state; snapshot and worker holdouts):",
            "",
            (
                markdown_table(difficulty_models)
                if len(difficulty_models)
                else "Insufficient snapshots for difficulty modeling."
            ),
            "",
            "Strongest layer/denoise quartile contrasts (maxT corrects the full screen):",
            "",
            markdown_table(screens.head(12)),
            "",
            "Strongest individual-expert probability contrasts (maxT corrects all 1,280 cells per prefix):",
            "",
            (
                markdown_table(expert_effects.head(12))
                if len(expert_effects)
                else "Insufficient mixed snapshots for expert screening."
            ),
            "",
            "Failure-subtype models, evaluated only among failed branches:",
            "",
            (
                subtype_models.sort_values(
                    ["cv_scheme", "failure_type", "prefix_queries", "brier"],
                    kind="stable",
                )
                .groupby(
                    ["cv_scheme", "failure_type", "prefix_queries"],
                    sort=False,
                )
                .head(3)
                .pipe(markdown_table)
                if len(subtype_models)
                else "Insufficient support for subtype cross-validation."
            ),
            "",
            "Strongest subtype-specific route contrasts:",
            "",
            (
                subtype_screens.sort_values(
                    ["permutation_p_maxT", "permutation_p_raw"], kind="stable"
                ).head(12).pipe(markdown_table)
                if len(subtype_screens)
                else "Insufficient within-snapshot subtype variation."
            ),
            "",
            "## Interpretation guardrails",
            "",
            "- A route contrast is evidence that routing accompanies an early risky sample, not proof that a specific expert causes failure.",
            "- Absolute models can encode initial-state difficulty. Models ending in `_within_snapshot` subtract the unlabeled K=16 sibling mean first, removing the common phase component.",
            "- Noise and first action chunks are explicit controls; each MoE model must beat the `noise+action` control with the same absolute/within-snapshot scope before claiming incremental MoE information.",
            "- Snapshot-grouped cross-validation, within-snapshot permutations, and snapshot bootstrap prevent treating correlated queries or siblings as independent.",
            "- Small numbers of mixed snapshots make effect intervals more important than a selected best cell.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_self_test() -> None:
    mask = np.asarray([False, True, True, False, True])
    assert longest_true_run(mask) == (2, 1, 3)
    entries, exits = hysteresis_events(
        np.asarray([0.2, 0.04, 0.03, 0.12]), 0.055, 0.1, lower_is_inside=True
    )
    assert entries == [1] and exits == [3]
    p = np.asarray([[0.25, 0.75]], dtype=np.float32)
    assert np.allclose(normalize_probabilities(p), p)
    assert np.allclose(hellinger(p, p), 0.0)
    values = np.arange(16, dtype=np.float64)
    failures = np.asarray([0] * 8 + [1] * 8, dtype=np.float64)
    groups = np.asarray(["a"] * 16)
    effect, top, bottom, n_top, n_bottom = quartile_effect(values, failures, groups)
    assert effect == 1.0 and top == 1.0 and bottom == 0.0
    assert n_top == n_bottom == 4
    stationary = static_window_metrics(
        np.zeros((121, 3), dtype=np.float32),
        np.zeros((121, 2, 3), dtype=np.float32),
        np.zeros(121, dtype=np.float32),
    )
    assert stationary["late_static_window_fraction"] == 1.0
    centered = center_within_groups(
        np.asarray([[1.0, 3.0], [3.0, 5.0], [10.0, 20.0], [14.0, 24.0]]),
        np.asarray(["a", "a", "b", "b"]),
    )
    assert np.allclose(centered[:2].mean(axis=0), 0.0)
    assert np.allclose(centered[2:].mean(axis=0), 0.0)
    grouped_values = np.asarray([1.0, 3.0, 5.0, 7.0, 9.0])
    grouped_ids = np.asarray([0, 0, 1, 2, 2])
    bootstrap_seed = 41
    bootstrap_count = 50
    sampled = np.random.default_rng(bootstrap_seed).integers(
        0, 3, size=(bootstrap_count, 3)
    )
    explicit_draws = np.asarray(
        [
            np.concatenate(
                [grouped_values[grouped_ids == group] for group in draw]
            ).mean()
            for draw in sampled
        ]
    )
    expected_interval = np.quantile(explicit_draws, [0.025, 0.975])
    actual_interval = grouped_mean_interval(
        grouped_values,
        grouped_ids,
        bootstraps=bootstrap_count,
        rng=np.random.default_rng(bootstrap_seed),
    )
    assert np.allclose(actual_interval[0], grouped_values.mean())
    assert np.allclose(actual_interval[1:], expected_interval)
    assert stationary["terminal_static_window_steps"] == 120.0
    loop = query_loop_metrics(
        np.asarray(
            [[0.0, 0.0, 0.0], [0.06, 0.0, 0.0], [0.12, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        np.zeros((4, 2, 3), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        np.ones(4, dtype=np.float32),
    )
    assert loop["loop_return_count"] == 1.0
    assert loop["loop_max_lag"] == 3.0
    capture_audit = validate_capture_summary(
        {
            "control_steps": 10,
            "as_collapsed": 10,
            "store_full_probs": True,
            "store_hidden": False,
            "hook_verified_calls": 480,
            "hook_verify_failures": [],
            "gates": [{}] * 12,
            "n_denoise": 10,
            "pin": None,
        },
        {"route_rows_total": 10, "store_hidden": False},
    )
    assert capture_audit["hook_verified_control_steps"] == 4
    assert capture_audit["hook_calls_per_control_step"] == 120
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    if args.run_root is None:
        raise SystemExit("--run-root is required unless --self-test is used")
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or (run_root / "analysis")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, collection_audit = discover_candidates(run_root)
    candidate_id_table, candidate_id_audit = candidate_id_negative_control(
        candidates, permutations=args.permutations, seed=args.seed
    )
    candidate_id_table.to_csv(
        out_dir / "candidate_id_negative_control.csv", index=False
    )
    targets, layout = load_layout(run_root)
    trajectories = {
        candidate.episode_id: load_trajectory(candidate) for candidate in candidates
    }
    noise_audit = audit_branch_noises(candidates, trajectories)
    auxiliary_audit = audit_auxiliary_joints(candidates, trajectories, layout)
    references = build_goal_references(candidates, trajectories, targets)
    physical = pd.DataFrame(
        [
            physical_metrics(candidate, trajectories[candidate.episode_id], targets, references)
            for candidate in candidates
        ]
    )
    physical, terminal_predicate_audit = merge_terminal_predicates(
        run_root, physical, candidates
    )
    coverage = failure_mode_coverage(physical)
    snapshots = snapshot_summary(physical)
    physical.to_csv(out_dir / "candidate_physical_labels.csv", index=False)
    physical.to_json(
        out_dir / "candidate_physical_labels.jsonl",
        orient="records",
        lines=True,
    )
    snapshots.to_csv(out_dir / "snapshot_outcomes.csv", index=False)
    write_json(out_dir / "failure_mode_coverage.json", coverage)
    write_json(
        out_dir / "physical_label_definitions.json",
        {
            "thresholds": {
                "goal_enter_m": GOAL_ENTER_M,
                "goal_exit_m": GOAL_EXIT_M,
                "lift_enter_m": LIFT_ENTER_M,
                "lift_exit_m": LIFT_EXIT_M,
                "drop_m": DROP_M,
                "static_window_actions": STATIC_WINDOW,
                "static_eef_path_m": STATIC_EEF_PATH_M,
                "static_object_path_m": STATIC_OBJECT_PATH_M,
                "static_gripper_path_m": STATIC_GRIPPER_PATH_M,
                "loop_min_query_lag": LOOP_MIN_QUERY_LAG,
                "loop_eef_return_m": LOOP_EEF_RETURN_M,
                "loop_object_return_m": LOOP_OBJECT_RETURN_M,
                "loop_gripper_return_m": LOOP_GRIPPER_RETURN_M,
                "loop_eef_path_m": LOOP_EEF_PATH_M,
                "active_eef_path_m": ACTIVE_EEF_PATH_M,
                "active_max_goal_progress_per_eef": (
                    ACTIVE_MAX_GOAL_PROGRESS_PER_EEF
                ),
            },
            "target_joints": [str(target["joint"]) for target in targets],
            "success_terminal_references": references,
            "label_source": "physical trajectories only; no MoE routing",
            "derived_subtype_targets": {
                "non_stagnation_non_loop": (
                    "failure and neither physical stagnation nor loop/cycling"
                )
            },
        },
    )
    hb_routes, as_routes, route_audit = load_routes(run_root, candidates)
    capture_summary_path = run_root / "formal" / "server" / "capture_summary.json"
    capture_summary = (
        json.loads(capture_summary_path.read_text(encoding="utf-8"))
        if capture_summary_path.is_file()
        else None
    )
    capture_summary_audit = (
        validate_capture_summary(capture_summary, route_audit)
        if capture_summary is not None
        else None
    )
    banks, cell_frame, feature_audit = route_feature_banks(
        candidates, trajectories, hb_routes, as_routes
    )
    cell_frame.to_csv(out_dir / "early_route_cell_features.csv", index=False)
    screens = screen_route_cells(
        cell_frame,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    screens.to_csv(out_dir / "early_route_quartile_effects.csv", index=False)
    expert_effects, expert_audit = expert_route_screen(
        candidates,
        physical,
        hb_routes,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    expert_effects.to_csv(out_dir / "early_expert_probability_effects.csv", index=False)
    frozen_candidates, frozen_expert_summary, frozen_expert_audit = (
        frozen_signal_validation(
            candidates,
            physical,
            hb_routes,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            seed=args.seed,
        )
    )
    frozen_state_candidates, frozen_state_summary, frozen_state_audit = (
        frozen_state_signal_validation(
            candidates,
            physical,
            cell_frame,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            seed=args.seed,
        )
    )
    frozen_expert_summary.insert(0, "signal", "q0_action_front_d0_expert0")
    frozen_expert_summary.insert(
        2, "primary_statistic", "mean_failure_minus_success"
    )
    frozen_expert_summary.insert(
        3,
        "primary_p_one_sided",
        frozen_expert_summary["mean_effect_permutation_p_one_sided"],
    )
    frozen_state_summary.insert(0, "signal", "q0_state_front_d1_top1_mass")
    frozen_state_summary.insert(
        2, "primary_statistic", "top_minus_bottom_quartile_failure_rate"
    )
    frozen_state_summary.insert(
        3,
        "primary_p_one_sided",
        frozen_state_summary["quartile_effect_permutation_p_one_sided"],
    )
    frozen_summary = pd.concat(
        [frozen_expert_summary, frozen_state_summary], ignore_index=True
    )
    frozen_summary.insert(4, "validation_family_size", 2)
    frozen_summary.insert(
        5,
        "primary_p_bonferroni",
        np.minimum(1.0, 2.0 * frozen_summary["primary_p_one_sided"]),
    )
    frozen_audit = {
        "family_size": 2,
        "correction": "Bonferroni across the two frozen directional primary tests",
        "expert0": frozen_expert_audit,
        "state_top1_mass": frozen_state_audit,
        "no_more_signals_will_be_added": True,
    }
    frozen_candidates.to_csv(
        out_dir / "frozen_expert0_candidate_values.csv", index=False
    )
    frozen_state_candidates.to_csv(
        out_dir / "frozen_state_top1_candidate_values.csv", index=False
    )
    frozen_summary.to_csv(out_dir / "frozen_signal_validation_family.csv", index=False)
    snapshot_models, snapshot_predictions = model_comparisons(
        candidates,
        physical,
        banks,
        bootstraps=args.bootstraps,
        seed=args.seed,
        cv_scheme="grouped_5fold_snapshot",
    )
    worker_models_to_run = {
        "snapshot_rate_only",
        "noise+action",
        "moe_action",
        "moe_state",
        "noise+action+moe_action+state",
        "noise+action_within_snapshot",
        "moe_action_within_snapshot",
        "moe_state_within_snapshot",
        "noise+action+moe_action+state_within_snapshot",
    }
    worker_models, worker_predictions = model_comparisons(
        candidates,
        physical,
        banks,
        bootstraps=args.bootstraps,
        seed=args.seed,
        cv_scheme="leave_one_worker_out",
        include_models=worker_models_to_run,
    )
    models = pd.concat([snapshot_models, worker_models], ignore_index=True)
    predictions = pd.concat(
        [snapshot_predictions, worker_predictions], ignore_index=True
    )
    models.to_csv(out_dir / "early_prediction_models.csv", index=False)
    predictions.to_csv(out_dir / "early_prediction_oof.csv", index=False)
    difficulty_models, difficulty_predictions, difficulty_audit = (
        snapshot_difficulty_models(
            candidates,
            physical,
            trajectories,
            banks,
            targets,
            bootstraps=args.bootstraps,
            seed=args.seed,
        )
    )
    difficulty_models.to_csv(out_dir / "snapshot_difficulty_models.csv", index=False)
    difficulty_predictions.to_csv(
        out_dir / "snapshot_difficulty_oof.csv", index=False
    )
    subtype_models = subtype_model_comparisons(
        candidates,
        physical,
        banks,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    subtype_models.to_csv(out_dir / "early_failure_subtype_models.csv", index=False)
    subtype_screens = subtype_route_screens(
        cell_frame,
        physical,
        permutations=args.permutations,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    subtype_screens.to_csv(
        out_dir / "early_failure_subtype_route_effects.csv", index=False
    )
    code_dir = Path(__file__).resolve().parent
    provenance_files = {
        "analysis": code_dir / "analyze_rolling_star_experiment.py",
        "collector": code_dir / "rolling_star_collect.py",
        "terminal_predicate_auditor": code_dir
        / "audit_rolling_star_terminal_predicates.py",
        "protocol": code_dir / "ROLLING_STAR_EXPERIMENT_PROTOCOL.md",
    }
    provenance = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in provenance_files.items()
    }
    write_json(
        out_dir / "audit.json",
        {
            "collection": collection_audit,
            "layout": layout,
            "routes": route_audit,
            "features": feature_audit,
            "expert_screen": expert_audit,
            "frozen_signal_validation_family": frozen_audit,
            "branch_noise": noise_audit,
            "candidate_id_negative_control": candidate_id_audit,
            "auxiliary_joints": auxiliary_audit,
            "terminal_predicates": terminal_predicate_audit,
            "snapshot_difficulty": difficulty_audit,
            "statistics": {
                "permutations": args.permutations,
                "bootstraps": args.bootstraps,
                "seed": args.seed,
                "auc_reported": False,
                "candidate_prediction_cv": {
                    "grouped_5fold_snapshot": (
                        "GroupKFold(5), with every K=16 snapshot wholly in one fold"
                    ),
                    "leave_one_worker_out": "GroupKFold(4), one worker per fold",
                },
                "snapshot_difficulty_cv": {
                    "leave_one_snapshot_out": "22 folds, one snapshot per fold",
                    "leave_one_worker_out": "4 folds, one worker per fold",
                },
            },
            "provenance": provenance,
            "server_capture_summary": capture_summary,
            "server_capture_validation": capture_summary_audit,
            "candidate_npz_checksums_verified": len(candidates),
        },
    )
    plot_outputs(physical, snapshots, screens, models, out_dir)
    build_report(
        physical,
        snapshots,
        screens,
        expert_effects,
        models,
        difficulty_models,
        subtype_models,
        subtype_screens,
        frozen_summary,
        frozen_audit,
        route_audit,
        feature_audit,
        coverage,
        candidate_id_audit,
        out_dir,
    )
    print(
        f"analyzed {len(candidates)} branches, {len(snapshots)} snapshots, "
        f"{int(snapshots['mixed'].sum())} mixed",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
