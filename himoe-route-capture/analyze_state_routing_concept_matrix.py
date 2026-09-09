#!/usr/bin/env python3
"""Cross-fitted concept probes for state-token soft MoE routing.

The unit held out during fitting is a complete task/init pool: all 32 flow
seeds and every query from that initial state move together.  Route-only probes
ask whether a concept is linearly readable from token-0 soft router
probabilities.  Joint-minus-geometry asks the stricter question of whether that
readout adds to current raw kinematics.  No hard expert IDs enter any feature.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import roc_auc_score


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/state-routing-concept-matrix"
SEED = 20260830
BOOTSTRAPS = 1000
RIDGE_ALPHA = 100.0
HIDDEN_PROJECTION_DIM = 128
N_LAYERS = 8
N_EXPERTS = 32

MIDDLE_DRAWER_TASK = "libero_goal/open_the_middle_drawer_of_the_cabinet"
TOP_DRAWER_TASK = "libero_goal/open_the_top_drawer_and_put_the_bowl_inside"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RAMEKIN_TASK = (
    "libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
)
STOVE_TASK = (
    "libero_spatial/"
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
)

TASK_TARGET_JOINTS = {
    MIDDLE_DRAWER_TASK: (),
    TOP_DRAWER_TASK: ("akita_black_bowl_1_joint0",),
    LONG_TASK: ("moka_pot_1_joint0", "moka_pot_2_joint0"),
    RAMEKIN_TASK: ("akita_black_bowl_1_joint0",),
    STOVE_TASK: ("akita_black_bowl_1_joint0",),
}
TASK_DRAWER_JOINT = {
    MIDDLE_DRAWER_TASK: "wooden_cabinet_1_middle_level",
    TOP_DRAWER_TASK: "wooden_cabinet_1_top_level",
}

BINARY_CONCEPTS = {
    "gripper_closed_proxy",
    "close_command",
    "hold_proxy",
    "near_target_close_command_proxy",
    "target_lifted_1cm_proxy",
}

CONCEPT_META: dict[str, tuple[str, str, str]] = {
    "gripper_aperture_m": ("continuous", "gripper", "m"),
    "gripper_command_mean": ("continuous", "action_command", "native"),
    "eef_x_m": ("continuous", "eef_position", "m"),
    "eef_y_m": ("continuous", "eef_position", "m"),
    "eef_z_m": ("continuous", "eef_position", "m"),
    "eef_axis_angle_x": ("continuous", "eef_orientation", "rad"),
    "eef_axis_angle_y": ("continuous", "eef_orientation", "rad"),
    "eef_axis_angle_z": ("continuous", "eef_orientation", "rad"),
    "eef_axis_angle_norm_rad": ("continuous", "eef_orientation", "rad"),
    "target_x_m": ("continuous", "target_position", "m"),
    "target_y_m": ("continuous", "target_position", "m"),
    "target_z_m": ("continuous", "target_position", "m"),
    "target_quat_w": ("continuous", "target_orientation", "unitless"),
    "target_quat_x": ("continuous", "target_orientation", "unitless"),
    "target_quat_y": ("continuous", "target_orientation", "unitless"),
    "target_quat_z": ("continuous", "target_orientation", "unitless"),
    "target_tilt_rad": ("continuous", "target_orientation", "rad"),
    "target_minus_eef_x_m": ("continuous", "relative_geometry", "m"),
    "target_minus_eef_y_m": ("continuous", "relative_geometry", "m"),
    "target_minus_eef_z_m": ("continuous", "relative_geometry", "m"),
    "eef_target_distance_m": ("continuous", "relative_geometry", "m"),
    "target_goal_distance_m": ("continuous", "goal_progress", "m"),
    "drawer_position_native": ("continuous", "drawer", "native"),
    "drawer_goal_distance_native": ("continuous", "drawer", "native"),
    "task_progress": ("continuous", "goal_progress", "normalized"),
    "task_progress_speed": ("continuous", "motion", "per_query"),
    "eef_speed_m_per_query": ("continuous", "motion", "m_per_query"),
    "target_speed_m_per_query": ("continuous", "motion", "m_per_query"),
    "phase_budget": ("continuous", "phase", "fraction"),
    "phase_episode_posthoc": ("continuous", "phase", "fraction"),
    "gripper_closed_proxy": ("binary", "gripper", "bool"),
    "close_command": ("binary", "action_command", "bool"),
    "hold_proxy": ("binary", "hold", "bool"),
    "near_target_close_command_proxy": ("binary", "hold", "bool"),
    "target_lifted_1cm_proxy": ("binary", "hold", "bool"),
}
CONCEPTS = tuple(CONCEPT_META)


@dataclass
class LoadedData:
    query: pd.DataFrame
    geometry: np.ndarray
    geometry_names: list[str]
    route: np.ndarray
    hidden: np.ndarray | None
    audit: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    parser.add_argument("--hidden-projection-dim", type=int, default=HIDDEN_PROJECTION_DIM)
    parser.add_argument("--skip-hidden", action="store_true")
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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = sorted(
        path.parents[1]
        for path in cache_root.glob("libero_*/*/right-16x32/client/summaries.json")
        if (path.parents[1] / "server/routes.zarr").exists()
    )
    if len(runs) != 5:
        raise RuntimeError(f"expected five complete runs, found {len(runs)}")
    return runs


def task_key(run: pathlib.Path, cache_root: pathlib.Path) -> str:
    return str(run.relative_to(cache_root).parent)


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid router probabilities")
    return result / mass


def step_vectors(values: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        result[1:] = np.diff(values, axis=0)
    return result


def step_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(step_vectors(values), axis=1)


def canonical_quaternion_wxyz(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64).copy()
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion /= np.maximum(norm, 1e-12)
    quaternion[quaternion[:, 0] < 0.0] *= -1.0
    return quaternion.astype(np.float32)


def quaternion_tilt_wxyz(values: np.ndarray) -> np.ndarray:
    quaternion = canonical_quaternion_wxyz(values)
    x = quaternion[:, 1]
    y = quaternion[:, 2]
    vertical_cosine = 1.0 - 2.0 * (x * x + y * y)
    return np.arccos(np.clip(vertical_cosine, -1.0, 1.0)).astype(np.float32)


def joint_slice(layout: dict[str, Any], name: str) -> slice:
    matches = [row for row in layout["joints"] if str(row["joint"]) == name]
    if len(matches) != 1:
        raise ValueError(f"joint {name!r} missing or duplicated")
    row = matches[0]
    return slice(int(row["state_lo"]), int(row["state_hi"]))


def load_client_tapes(
    run: pathlib.Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, np.ndarray]]]:
    client = run / "client"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 512 or [int(row["episode_index"]) for row in summaries] != list(range(512)):
        raise ValueError(f"unexpected episode grid in {run}")
    layout = json.loads((client / "sim_layout.json").read_text())
    tapes = []
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(client / f"episode_{episode:02d}.npz", allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
        expected = int(row["inference_calls"])
        if state.shape != (expected, 8) or actions.shape != (expected, 10, 7) or len(sim) != expected:
            raise ValueError(f"client/query alignment drift in {run} episode {episode}")
        tapes.append({"state": state, "actions": actions, "sim": sim})
    return summaries, layout, tapes


def success_references(
    task: str,
    summaries: list[dict[str, Any]],
    layout: dict[str, Any],
    tapes: list[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], float | None]:
    success = [index for index, row in enumerate(summaries) if bool(row["success"])]
    if not success:
        raise ValueError(f"task has no successful endpoint reference: {task}")
    target_goals = {}
    for name in TASK_TARGET_JOINTS[task]:
        bounds = joint_slice(layout, name)
        if bounds.stop - bounds.start != 7:
            raise ValueError(f"target {name} is not a free joint")
        target_goals[name] = np.mean(
            [tapes[index]["sim"][-1, bounds.start : bounds.start + 3] for index in success],
            axis=0,
        )
    drawer_goal = None
    if task in TASK_DRAWER_JOINT:
        bounds = joint_slice(layout, TASK_DRAWER_JOINT[task])
        drawer_goal = float(
            np.mean([tapes[index]["sim"][-1, bounds.start] for index in success])
        )
    return target_goals, drawer_goal


def episode_concepts(
    task: str,
    row: dict[str, Any],
    layout: dict[str, Any],
    tape: dict[str, np.ndarray],
    target_goals: dict[str, np.ndarray],
    drawer_goal: float | None,
    budget_queries: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    state = tape["state"]
    actions = tape["actions"]
    sim = tape["sim"]
    n = len(state)
    eef = state[:, :3]
    eef_axis = state[:, 3:6]
    aperture = np.abs(state[:, 6] - state[:, 7])
    gap = np.mean(np.abs(state[:, 6:8]), axis=1)
    command = actions[:, :, 6].mean(axis=1)
    close_command = command > 0.5
    eef_delta = step_vectors(eef)
    eef_speed = np.linalg.norm(eef_delta, axis=1)

    target_names = TASK_TARGET_JOINTS[task]
    target_available = bool(target_names)
    progress_components: list[np.ndarray] = []
    hold_components: list[np.ndarray] = []
    near_close_components: list[np.ndarray] = []
    lifted_components: list[np.ndarray] = []
    if target_available:
        positions = []
        quaternions = []
        goal_distances = []
        target_speeds = []
        eef_distances = []
        for name in target_names:
            bounds = joint_slice(layout, name)
            position = sim[:, bounds.start : bounds.start + 3]
            quaternion = canonical_quaternion_wxyz(
                sim[:, bounds.start + 3 : bounds.start + 7]
            )
            goal_distance = np.linalg.norm(position - target_goals[name], axis=1)
            initial_goal_distance = max(float(goal_distance[0]), 0.05)
            progress_components.append(1.0 - goal_distance / initial_goal_distance)
            target_delta_one = step_vectors(position)
            target_speed_one = np.linalg.norm(target_delta_one, axis=1)
            distance = np.linalg.norm(position - eef, axis=1)
            residual = np.linalg.norm(target_delta_one - eef_delta, axis=1)
            displacement = np.linalg.norm(position - position[:1], axis=1)
            hold_threshold = 0.115 if task == LONG_TASK else 0.085
            near_threshold = 0.13 if task == LONG_TASK else 0.10
            hold_components.append(
                (distance < hold_threshold)
                & (gap < 0.025)
                & (target_speed_one > 0.005)
                & (residual < 0.04)
                & (displacement > 0.012)
            )
            near_close_components.append((distance < near_threshold) & close_command)
            lifted_components.append(position[:, 2] - position[0, 2] > 0.01)
            positions.append(position)
            quaternions.append(quaternion)
            goal_distances.append(goal_distance)
            target_speeds.append(target_speed_one)
            eef_distances.append(distance)
        position_stack = np.stack(positions, axis=1)
        quaternion_stack = np.stack(quaternions, axis=1)
        distance_stack = np.stack(eef_distances, axis=1)
        selected_index = np.argmin(distance_stack, axis=1)
        query_index = np.arange(n)
        selected_position = position_stack[query_index, selected_index]
        selected_quaternion = quaternion_stack[query_index, selected_index]
        selected_goal_distance = np.stack(goal_distances, axis=1)[
            query_index, selected_index
        ]
        selected_target_speed = np.stack(target_speeds, axis=1)[
            query_index, selected_index
        ]
        target_delta = step_vectors(selected_position)
        relative = selected_position - eef
        target_distance = np.linalg.norm(relative, axis=1)
        target_tilt = quaternion_tilt_wxyz(selected_quaternion)
        hold_proxy = np.any(np.stack(hold_components, axis=1), axis=1)
        near_close_proxy = np.any(np.stack(near_close_components, axis=1), axis=1)
        lifted_proxy = np.any(np.stack(lifted_components, axis=1), axis=1)
    else:
        selected_position = np.full((n, 3), np.nan, dtype=np.float32)
        selected_quaternion = np.full((n, 4), np.nan, dtype=np.float32)
        selected_goal_distance = np.full(n, np.nan, dtype=np.float32)
        selected_target_speed = np.full(n, np.nan, dtype=np.float32)
        target_delta = np.zeros((n, 3), dtype=np.float32)
        relative = np.full((n, 3), np.nan, dtype=np.float32)
        target_distance = np.full(n, np.nan, dtype=np.float32)
        target_tilt = np.full(n, np.nan, dtype=np.float32)
        hold_proxy = np.full(n, np.nan, dtype=np.float32)
        near_close_proxy = np.full(n, np.nan, dtype=np.float32)
        lifted_proxy = np.full(n, np.nan, dtype=np.float32)

    drawer_available = task in TASK_DRAWER_JOINT
    if drawer_available:
        bounds = joint_slice(layout, TASK_DRAWER_JOINT[task])
        drawer = sim[:, bounds.start]
        if drawer_goal is None:
            raise ValueError("drawer goal reference missing")
        drawer_goal_distance = np.abs(drawer - drawer_goal)
        denominator = drawer_goal - float(drawer[0])
        if abs(denominator) < 1e-4:
            denominator = math.copysign(1e-4, denominator if denominator else 1.0)
        progress_components.append((drawer - drawer[0]) / denominator)
    else:
        drawer = np.full(n, np.nan, dtype=np.float32)
        drawer_goal_distance = np.full(n, np.nan, dtype=np.float32)

    task_progress = np.mean(np.stack(progress_components, axis=1), axis=1)
    progress_speed = np.zeros(n, dtype=np.float32)
    progress_speed[1:] = np.diff(task_progress)
    phase_budget = np.arange(n, dtype=np.float32) / max(budget_queries - 1, 1)
    phase_episode = np.arange(n, dtype=np.float32) / max(n - 1, 1)

    concept = {
        "gripper_aperture_m": aperture,
        "gripper_command_mean": command,
        "eef_x_m": eef[:, 0],
        "eef_y_m": eef[:, 1],
        "eef_z_m": eef[:, 2],
        "eef_axis_angle_x": eef_axis[:, 0],
        "eef_axis_angle_y": eef_axis[:, 1],
        "eef_axis_angle_z": eef_axis[:, 2],
        "eef_axis_angle_norm_rad": np.linalg.norm(eef_axis, axis=1),
        "target_x_m": selected_position[:, 0],
        "target_y_m": selected_position[:, 1],
        "target_z_m": selected_position[:, 2],
        "target_quat_w": selected_quaternion[:, 0],
        "target_quat_x": selected_quaternion[:, 1],
        "target_quat_y": selected_quaternion[:, 2],
        "target_quat_z": selected_quaternion[:, 3],
        "target_tilt_rad": target_tilt,
        "target_minus_eef_x_m": relative[:, 0],
        "target_minus_eef_y_m": relative[:, 1],
        "target_minus_eef_z_m": relative[:, 2],
        "eef_target_distance_m": target_distance,
        "target_goal_distance_m": selected_goal_distance,
        "drawer_position_native": drawer,
        "drawer_goal_distance_native": drawer_goal_distance,
        "task_progress": task_progress,
        "task_progress_speed": progress_speed,
        "eef_speed_m_per_query": eef_speed,
        "target_speed_m_per_query": selected_target_speed,
        "phase_budget": phase_budget,
        "phase_episode_posthoc": phase_episode,
        "gripper_closed_proxy": aperture < 0.025,
        "close_command": close_command,
        "hold_proxy": hold_proxy,
        "near_target_close_command_proxy": near_close_proxy,
        "target_lifted_1cm_proxy": lifted_proxy,
    }
    metadata = {
        "task": np.repeat(task, n),
        "episode": np.repeat(int(row["episode_index"]), n),
        "query": np.arange(n),
        "init_state_id": np.repeat(int(row["init_state_id"]), n),
        "flow_noise_seed": np.repeat(int(row["flow_noise_seed"]), n),
        "success": np.repeat(bool(row["success"]), n),
        "episode_length": np.repeat(n, n),
    }
    frame = pd.DataFrame({**metadata, **concept})

    geometry_names = [
        "eef_x",
        "eef_y",
        "eef_z",
        "eef_axis_x",
        "eef_axis_y",
        "eef_axis_z",
        "finger_qpos_1",
        "finger_qpos_2",
        "target_x",
        "target_y",
        "target_z",
        "target_quat_w",
        "target_quat_x",
        "target_quat_y",
        "target_quat_z",
        "drawer_qpos",
        "eef_delta_x",
        "eef_delta_y",
        "eef_delta_z",
        "target_delta_x",
        "target_delta_y",
        "target_delta_z",
        "target_available",
        "drawer_available",
        "phase_budget",
        "phase_episode_posthoc",
    ]
    geometry = np.column_stack(
        [
            state,
            np.nan_to_num(selected_position),
            np.nan_to_num(selected_quaternion),
            np.nan_to_num(drawer),
            eef_delta,
            target_delta,
            np.full(n, float(target_available)),
            np.full(n, float(drawer_available)),
            phase_budget,
            phase_episode,
        ]
    ).astype(np.float32)
    if geometry.shape[1] != len(geometry_names):
        raise RuntimeError("geometry feature/name width mismatch")
    return frame, geometry, geometry_names


def load_state_soft_route(
    run: pathlib.Path, summaries: list[dict[str, Any]], block: int = 512
) -> tuple[np.ndarray, dict[str, Any]]:
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries])
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    expected_episode = np.repeat(episodes, lengths)
    if not np.array_equal(np.asarray(store["episode_id"][:]), expected_episode):
        raise ValueError(f"route episode alignment failed: {run}")
    total = int(lengths.sum())
    result = np.empty((total, N_LAYERS * N_EXPERTS), dtype=np.float32)
    probability_mass_min = float("inf")
    probability_mass_max = float("-inf")
    for start in range(0, total, block):
        stop = min(start + block, total)
        raw = np.asarray(
            store["hb_router_probs"][start:stop, :, 0, 0, :], dtype=np.float32
        )
        mass = raw.sum(axis=-1)
        probability_mass_min = min(probability_mass_min, float(mass.min()))
        probability_mass_max = max(probability_mass_max, float(mass.max()))
        result[start:stop] = normalize_probabilities(raw).reshape(stop - start, -1)
    sample_index = np.unique(np.linspace(0, total - 1, min(total, 32), dtype=int))
    full_denoise = np.asarray(
        store["hb_router_probs"].oindex[sample_index, :, :, 0, :], dtype=np.float32
    )
    denoise_span = float(
        np.max(np.abs(full_denoise - full_denoise[:, :, :1, :]))
    )
    return result, {
        "queries": total,
        "state_token_denoise_max_probability_span_sample": denoise_span,
        "probability_mass_min_before_normalization": probability_mass_min,
        "probability_mass_max_before_normalization": probability_mass_max,
        "hard_expert_ids_read": False,
    }


def load_hidden_projection(
    run: pathlib.Path,
    summaries: list[dict[str, Any]],
    projection: np.ndarray,
    block: int = 64,
) -> tuple[np.ndarray, dict[str, Any]]:
    hidden = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    if not np.array_equal(hidden["episode_id"][:], route["episode_id"][:]) or not np.array_equal(
        hidden["control_step"][:], route["control_step"][:]
    ):
        raise ValueError(f"hidden/route key alignment failed: {run}")
    total = sum(int(row["inference_calls"]) for row in summaries)
    result = np.empty((total, projection.shape[1]), dtype=np.float32)
    for start in range(0, total, block):
        stop = min(start + block, total)
        # Final HB layer, denoise 0, state token 0. The projection touches all
        # 1024 coordinates but only the compact result is retained.
        values = np.asarray(
            hidden["hb_hidden"][start:stop, -1, 0, 0, :], dtype=np.float32
        )
        result[start:stop] = values @ projection
        if stop % 2048 == 0 or stop == total:
            print(f"  hidden projection {stop}/{total}", flush=True)
    sample = np.asarray(hidden["hb_hidden"][:8, -1, :, 0, :], dtype=np.float32)
    denoise_span = float(np.max(np.abs(sample - sample[:, :1])))
    return result, {
        "queries": total,
        "source": "hb_hidden final HB layer, state token, denoise 0",
        "source_width": int(projection.shape[0]),
        "projection_width": int(projection.shape[1]),
        "projection": "fixed Gaussian random projection, label-free",
        "state_token_hidden_denoise_max_span_first_chunk": denoise_span,
        "alignment": "episode_id and control_step exactly equal routes.zarr",
    }


def load_data(
    cache_root: pathlib.Path,
    include_hidden: bool,
    hidden_projection_dim: int,
    seed: int,
) -> LoadedData:
    rng = np.random.default_rng(seed + 11)
    projection = rng.normal(
        0.0, 1.0 / math.sqrt(hidden_projection_dim), size=(1024, hidden_projection_dim)
    ).astype(np.float32)
    frames = []
    geometry_parts = []
    route_parts = []
    hidden_parts = []
    run_audit = {}
    geometry_names: list[str] | None = None
    offset = 0
    for run in discover_runs(cache_root):
        task = task_key(run, cache_root)
        if task not in TASK_TARGET_JOINTS:
            raise ValueError(f"task mapping missing: {task}")
        print(f"loading {task}", flush=True)
        summaries, layout, tapes = load_client_tapes(run)
        target_goals, drawer_goal = success_references(
            task, summaries, layout, tapes
        )
        meta = json.loads((run / "meta.json").read_text())
        budget_queries = int(meta["max_steps"]) // 10
        task_frames = []
        task_geometry = []
        for row, tape in zip(summaries, tapes):
            frame, geometry, names = episode_concepts(
                task,
                row,
                layout,
                tape,
                target_goals,
                drawer_goal,
                budget_queries,
            )
            task_frames.append(frame)
            task_geometry.append(geometry)
            if geometry_names is None:
                geometry_names = names
            elif geometry_names != names:
                raise ValueError("geometry contract changed across tasks")
        task_frame = pd.concat(task_frames, ignore_index=True)
        task_frame["global_query_row"] = np.arange(offset, offset + len(task_frame))
        offset += len(task_frame)
        route, route_audit = load_state_soft_route(run, summaries)
        if len(route) != len(task_frame):
            raise ValueError("route/client query count mismatch")
        if include_hidden:
            hidden, hidden_audit = load_hidden_projection(
                run, summaries, projection
            )
            hidden_parts.append(hidden)
        else:
            hidden_audit = {"available": (run / "server/hidden.zarr").exists(), "loaded": False}
        frames.append(task_frame)
        geometry_parts.append(np.concatenate(task_geometry))
        route_parts.append(route)
        run_audit[task] = {
            "episodes": len(summaries),
            "queries": len(task_frame),
            "initial_states": int(task_frame["init_state_id"].nunique()),
            "flow_noise_seeds": int(task_frame["flow_noise_seed"].nunique()),
            "successes": int(sum(bool(row["success"]) for row in summaries)),
            "target_joints": list(TASK_TARGET_JOINTS[task]),
            "drawer_joint": TASK_DRAWER_JOINT.get(task),
            "route": route_audit,
            "hidden": hidden_audit,
        }
    query = pd.concat(frames, ignore_index=True)
    if not np.array_equal(query["global_query_row"], np.arange(len(query))):
        raise RuntimeError("global query row drifted")
    return LoadedData(
        query=query,
        geometry=np.concatenate(geometry_parts),
        geometry_names=geometry_names or [],
        route=np.concatenate(route_parts),
        hidden=np.concatenate(hidden_parts) if include_hidden else None,
        audit={
            "tasks": run_audit,
            "queries": int(len(query)),
            "episodes": int(query[["task", "episode"]].drop_duplicates().shape[0]),
            "task_init_clusters": int(
                query[["task", "init_state_id"]].drop_duplicates().shape[0]
            ),
            "goal_reference": "all successful terminal states within task (transductive descriptive reference)",
            "target_selection": "nearest task target to EEF at each query; long task may switch target",
        },
    )


def ridge_fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train = np.asarray(train_x, dtype=np.float64)
    x_test = np.asarray(test_x, dtype=np.float64)
    y_train = np.asarray(train_y, dtype=np.float64)
    finite = np.any(np.isfinite(x_train), axis=0)
    x_train = x_train[:, finite]
    x_test = x_test[:, finite]
    median = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isfinite(x_train), x_train, median)
    x_test = np.where(np.isfinite(x_test), x_test, median)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    varying = scale > 1e-10
    x_train = (x_train[:, varying] - mean[varying]) / scale[varying]
    x_test = (x_test[:, varying] - mean[varying]) / scale[varying]
    y_mean = y_train.mean(axis=0)
    centered_y = y_train - y_mean
    if not x_train.shape[1]:
        return np.broadcast_to(y_mean, (len(x_test), len(y_mean))).copy()
    gram = x_train.T @ x_train
    gram.flat[:: len(gram) + 1] += alpha
    coefficient = np.linalg.solve(gram, x_train.T @ centered_y)
    return x_test @ coefficient + y_mean


def cross_fitted_predictions(
    data: LoadedData, alpha: float
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    query = data.query
    target = query[list(CONCEPTS)].to_numpy(dtype=np.float64)
    predictions = {
        "geometry": np.full_like(target, np.nan, dtype=np.float32),
        "state_soft_routing": np.full_like(target, np.nan, dtype=np.float32),
        "geometry_plus_state_soft": np.full_like(target, np.nan, dtype=np.float32),
    }
    if data.hidden is not None:
        predictions["hidden_reference"] = np.full_like(target, np.nan, dtype=np.float32)
    null = np.full_like(target, np.nan, dtype=np.float32)
    fold_rows = []
    for task in query["task"].drop_duplicates():
        task_mask = query["task"].to_numpy() == task
        task_rows = np.flatnonzero(task_mask)
        task_target = target[task_rows]
        finite_all = np.all(np.isfinite(task_target), axis=0)
        spread = np.zeros(task_target.shape[1], dtype=np.float64)
        spread[finite_all] = np.std(task_target[:, finite_all], axis=0)
        available = np.flatnonzero(finite_all & (spread > 1e-8))
        if not len(available):
            continue
        task_init = query.iloc[task_rows]["init_state_id"].to_numpy()
        for fold_id, held_init in enumerate(np.unique(task_init)):
            test_local = task_init == held_init
            train_local = ~test_local
            train_rows = task_rows[train_local]
            test_rows = task_rows[test_local]
            train_y = target[train_rows][:, available]
            null[test_rows[:, None], available] = train_y.mean(axis=0)
            blocks = {
                "geometry": data.geometry,
                "state_soft_routing": data.route,
                "geometry_plus_state_soft": np.column_stack(
                    [data.geometry, data.route]
                ),
            }
            if data.hidden is not None:
                blocks["hidden_reference"] = data.hidden
            for block_name, features in blocks.items():
                value = ridge_fit_predict(
                    features[train_rows], train_y, features[test_rows], alpha
                )
                predictions[block_name][np.ix_(test_rows, available)] = value.astype(
                    np.float32
                )
            fold_rows.append(
                {
                    "task": task,
                    "fold": fold_id,
                    "held_init_state_id": int(held_init),
                    "train_queries": int(len(train_rows)),
                    "test_queries": int(len(test_rows)),
                    "concepts": int(len(available)),
                }
            )
        print(f"cross-fitted {task}: {len(available)} concepts", flush=True)
    return predictions, null, {"folds": fold_rows, "ridge_alpha": alpha}


def continuous_skill(y: np.ndarray, prediction: np.ndarray, null: np.ndarray) -> float:
    denominator = float(np.sum(np.square(y - null)))
    if denominator <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum(np.square(y - prediction)) / denominator)


def cluster_metric(
    kind: str, y: np.ndarray, prediction: np.ndarray, null: np.ndarray
) -> float:
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, prediction))
    return continuous_skill(y, prediction, null)


def hierarchical_cluster_bootstrap(
    cluster_rows: pd.DataFrame,
    blocks: list[str],
    draws: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    tasks = cluster_rows["task"].unique()
    result = {block: np.empty(draws, dtype=np.float64) for block in blocks}
    for draw in range(draws):
        sampled_tasks = rng.choice(tasks, size=len(tasks), replace=True)
        per_block = {block: [] for block in blocks}
        for task in sampled_tasks:
            source = cluster_rows[cluster_rows["task"] == task]
            selected = rng.integers(0, len(source), size=len(source))
            for block in blocks:
                values = source.iloc[selected][block].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if len(values):
                    per_block[block].append(float(values.mean()))
        for block in blocks:
            result[block][draw] = (
                float(np.mean(per_block[block])) if per_block[block] else np.nan
            )
    return result


def evaluate_concepts(
    data: LoadedData,
    predictions: dict[str, np.ndarray],
    null: np.ndarray,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query = data.query
    target = query[list(CONCEPTS)].to_numpy(dtype=np.float64)
    blocks = list(predictions)
    cluster_rows = []
    for concept_index, concept in enumerate(CONCEPTS):
        kind = CONCEPT_META[concept][0]
        available = np.isfinite(target[:, concept_index]) & np.isfinite(
            null[:, concept_index]
        )
        for (task, init_state), indices in query[available].groupby(
            ["task", "init_state_id"]
        ).groups.items():
            rows = np.asarray(list(indices), dtype=np.int64)
            y = target[rows, concept_index]
            item: dict[str, Any] = {
                "concept": concept,
                "kind": kind,
                "task": task,
                "init_state_id": int(init_state),
                "queries": int(len(rows)),
                "target_mean": float(y.mean()),
                "target_sd": float(y.std()),
            }
            for block in blocks:
                item[block] = cluster_metric(
                    kind,
                    y,
                    predictions[block][rows, concept_index],
                    null[rows, concept_index],
                )
            cluster_rows.append(item)
    cluster_frame = pd.DataFrame(cluster_rows)
    task_rows = []
    matrix_rows = []
    for concept_index, concept in enumerate(CONCEPTS):
        subset = cluster_frame[cluster_frame["concept"] == concept]
        if subset.empty:
            continue
        kind, family, unit = CONCEPT_META[concept]
        for task, group in subset.groupby("task"):
            task_item: dict[str, Any] = {
                "concept": concept,
                "kind": kind,
                "family": family,
                "unit": unit,
                "task": task,
                "valid_init_clusters": int(group["init_state_id"].nunique()),
                "queries": int(group["queries"].sum()),
            }
            for block in blocks:
                task_item[block] = float(group[block].mean())
            task_rows.append(task_item)
        point = {
            block: float(subset.groupby("task")[block].mean().mean())
            for block in blocks
        }
        boot = hierarchical_cluster_bootstrap(
            subset,
            blocks,
            bootstrap,
            seed + concept_index * 97,
        )
        route_boot = boot["state_soft_routing"]
        delta_boot = (
            boot["geometry_plus_state_soft"] - boot["geometry"]
        )
        chance = 0.5 if kind == "binary" else 0.0
        route_ci = np.nanquantile(route_boot, [0.025, 0.975])
        delta_ci = np.nanquantile(delta_boot, [0.025, 0.975])
        row: dict[str, Any] = {
            "concept": concept,
            "kind": kind,
            "family": family,
            "unit": unit,
            "tasks": int(subset["task"].nunique()),
            "valid_task_init_clusters": int(len(subset)),
            "queries": int(subset["queries"].sum()),
            "chance": chance,
            **point,
            "route_ci95_low": float(route_ci[0]),
            "route_ci95_high": float(route_ci[1]),
            "routing_minus_chance": float(point["state_soft_routing"] - chance),
            "joint_minus_geometry": float(
                point["geometry_plus_state_soft"] - point["geometry"]
            ),
            "joint_minus_geometry_ci95_low": float(delta_ci[0]),
            "joint_minus_geometry_ci95_high": float(delta_ci[1]),
            "mde80_joint_minus_geometry": float(
                2.80 * np.nanstd(delta_boot, ddof=1)
            ),
            "readable_from_state_soft_routing": bool(route_ci[0] > chance),
            "increment_beyond_geometry": bool(delta_ci[0] > 0.0),
        }
        matrix_rows.append(row)
    return (
        pd.DataFrame(matrix_rows),
        pd.DataFrame(task_rows),
        cluster_frame,
    )


def make_matrix_plot(matrix: pd.DataFrame, out: pathlib.Path) -> None:
    ordered = matrix.sort_values(["family", "concept"]).reset_index(drop=True)
    display = np.column_stack(
        [
            ordered["geometry"] - ordered["chance"],
            ordered["state_soft_routing"] - ordered["chance"],
            ordered["geometry_plus_state_soft"] - ordered["chance"],
            ordered["joint_minus_geometry"],
        ]
    )
    bound = max(0.1, float(np.nanquantile(np.abs(display), 0.95)))
    fig, ax = plt.subplots(
        figsize=(8.5, max(8.0, 0.29 * len(ordered))), constrained_layout=True
    )
    image = ax.imshow(display, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    ax.set_xticks(range(4), ["geometry-chance", "route-chance", "joint-chance", "joint-geometry"])
    ax.set_yticks(range(len(ordered)), ordered["concept"], fontsize=7)
    ax.set_title("Cross-fitted state-token concept readout")
    for row in range(len(ordered)):
        for column in range(4):
            ax.text(column, row, f"{display[row, column]:+.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, label="metric points above chance / incremental")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def render_report(matrix: pd.DataFrame, data: LoadedData, protocol: dict[str, Any]) -> str:
    readable = matrix[matrix["readable_from_state_soft_routing"]]
    incremental = matrix[matrix["increment_beyond_geometry"]]
    ordered = matrix.sort_values("routing_minus_chance", ascending=False)
    rows = []
    for _, row in ordered.iterrows():
        hidden = (
            f"{row['hidden_reference']:.3f}"
            if "hidden_reference" in row and np.isfinite(row["hidden_reference"])
            else "NA"
        )
        rows.append(
            f"| {row['concept']} | {row['kind']} | {int(row['tasks'])} | "
            f"{row['geometry']:.3f} | {row['state_soft_routing']:.3f} | "
            f"{row['geometry_plus_state_soft']:.3f} | "
            f"{row['joint_minus_geometry']:+.3f} "
            f"[{row['joint_minus_geometry_ci95_low']:+.3f}, {row['joint_minus_geometry_ci95_high']:+.3f}] | "
            f"{row['mde80_joint_minus_geometry']:.3f} | {hidden} |"
        )
    route_contract = [
        value["route"]["state_token_denoise_max_probability_span_sample"]
        for value in data.audit["tasks"].values()
    ]
    hidden_text = (
        "五个 hidden store 的 episode/control key 均与 route store 精确一致；报告加入了最终 HB 层 state-token 1024 维 hidden 的固定 128 维无标签随机投影参考。它比 routing 更丰富，但随机投影有损，因此不是严格的 full-hidden 上界。"
        if data.hidden is not None
        else "hidden store 存在且键可核对，但本次以 `--skip-hidden` 运行；主矩阵没有 hidden 上限列。"
    )
    return f"""# State-token 概念可读出矩阵

## 直接结论

这里分开回答两个问题：

1. `state_soft_routing` 超过 chance：概念可以从 state-token 的 soft gate 分布线性读出；
2. `geometry+state - geometry` 的 cluster-bootstrap CI 高于 0：routing 在当前 raw kinematics 之外仍有增量。

共分析 {data.audit['queries']:,} 个 query、{data.audit['episodes']:,} 条 rollout、{data.audit['task_init_clusters']} 个 task×init cluster。{len(readable)}/{len(matrix)} 个概念满足第一个探索性门槛，{len(incremental)}/{len(matrix)} 个满足第二个更严格门槛。两者不能互换：routing 可读出一个概念，常常只说明 gate 是物理/视觉状态的低维影子。

## 验证协议

- 每个 task 分别拟合；一折完整留出一个 init，其 32 个 flow seed 和全部 query 不进入训练。
- 指标先在 held-out task×init 内计算，再 task-macro；CI 分层重采样 task 和 init cluster。
- 连续概念指标是相对该折训练均值的 OOF skill，0 为均值基线，1 为完美；二元概念是 within-init AUC，chance=0.5。
- `geometry` 是 raw proprio、所选目标 free-joint pose、相关 drawer qpos、一步差分和 phase；它不含 RGB。
- `state_soft_routing` 只有 token 0 的 `8×32` soft probabilities。没有读取 hard expert IDs；抽查 denoise 轴最大概率差为 {max(route_contract):.8f}。
- `MDE80` 是固定 OOF 预测下，用 cluster-bootstrap delta 标准差乘 2.80 的正态近似；它是 metric-point 检出尺度，不是事前功效保证。

## 主矩阵

| concept | kind | tasks | geometry | state soft | joint | joint−geometry [95% CI] | MDE80 | hidden ref |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Geometry 定义

- EEF `state[0:3]` 是位置，`state[3:6]` 是由 bridge 明确定义的 axis-angle，`state[6:8]` 是两指 qpos；aperture 是两指位置差的绝对值。
- transport task 的 target 是当前离 EEF 最近的任务目标；long task 因而可能在两个 moka pot 之间切换。target free-joint quaternion 按 MuJoCo `wxyz` 解释并规范到 `w≥0`，tilt 是局部 z 轴相对世界 z 的夹角。
- target goal 使用同 task 全部成功 rollout 的 terminal position 均值；这是描述性的 transductive reference，不应用于声称 unseen-init 在线泛化。
- task progress 是各 transport target 的归一化 goal-distance progress，与相关 drawer opening progress 的均值。hold/grasp 均为闭爪、邻近、物体和 EEF 共动的运动学 proxy，不是接触或真抓取标签。
- `phase_budget` 使用任务已知 query budget；`phase_episode_posthoc` 使用最终 rollout 长度，只作后验对照。

## Hidden 参考

{hidden_text}

## 解释边界

- state token 结构上能看 image/language prefix 与 proprio，本分析无法把视觉来源和 proprio 来源拆开。
- geometry baseline 是线性模型；routing 对非线性距离/阈值概念的增量可能只是另一种非线性编码，不等于含有物理状态之外的信息。
- query 高度时序相关；task×init cluster CI 比逐 query CI 更保守，但只有 5 个 task，仍是探索性矩阵。
- soft routing 的观察性读出不证明某个 expert 实现该概念，也不证明 routing 对行为有因果作用。

## 产物

- `concept_matrix.csv`: task-macro 主矩阵、增量 CI 与 MDE。
- `concept_task_metrics.csv`: 每 task 的 held-init 指标。
- `concept_cluster_metrics.csv`: 每个 task×init 的 OOF 指标。
- `query_concepts.csv.gz`: query 元数据与物理概念，不含大路由张量。
- `compact_features.npz`: float16 state-soft/geometry 与可选 hidden 投影。
- `concept_matrix.png`: chance-centered 主矩阵。
- `summary.json`: 数据、路由契约、cross-fitting 协议和计数。
"""


def run_self_test() -> None:
    quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0]])
    tilt = quaternion_tilt_wxyz(quaternion)
    np.testing.assert_allclose(tilt, [0.0, math.pi / 2.0], atol=1e-6)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 6))
    y = np.column_stack([2.0 * x[:, 0] - x[:, 1], x[:, 2] > 0.0])
    prediction = ridge_fit_predict(x[:150], y[:150], x[150:], 1.0)
    null = np.broadcast_to(y[:150].mean(axis=0), prediction.shape)
    assert continuous_skill(y[150:, 0], prediction[:, 0], null[:, 0]) > 0.95
    assert roc_auc_score(y[150:, 1], prediction[:, 1]) > 0.95
    probabilities = normalize_probabilities(np.asarray([[0.2, 0.3, 0.5]]))
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstrap < 100 or args.ridge_alpha <= 0.0:
        raise ValueError("bootstrap must be >=100 and ridge alpha must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load_data(
        args.cache_root,
        include_hidden=not args.skip_hidden,
        hidden_projection_dim=args.hidden_projection_dim,
        seed=args.seed,
    )
    predictions, null, crossfit = cross_fitted_predictions(data, args.ridge_alpha)
    matrix, task_metrics, cluster_metrics = evaluate_concepts(
        data, predictions, null, args.bootstrap, args.seed
    )
    data.query.to_csv(
        args.out_dir / "query_concepts.csv.gz", index=False, compression="gzip"
    )
    arrays: dict[str, np.ndarray] = {
        "geometry": data.geometry.astype(np.float16),
        "state_soft_routing": data.route.astype(np.float16),
    }
    if data.hidden is not None:
        arrays["hidden_projection"] = data.hidden.astype(np.float16)
    np.savez_compressed(args.out_dir / "compact_features.npz", **arrays)
    np.savez_compressed(
        args.out_dir / "oof_predictions.npz",
        null=null.astype(np.float16),
        **{name: value.astype(np.float16) for name, value in predictions.items()},
    )
    matrix.to_csv(args.out_dir / "concept_matrix.csv", index=False)
    task_metrics.to_csv(args.out_dir / "concept_task_metrics.csv", index=False)
    cluster_metrics.to_csv(args.out_dir / "concept_cluster_metrics.csv", index=False)
    make_matrix_plot(matrix, args.out_dir / "concept_matrix.png")
    summary = {
        "data": data.audit,
        "protocol": {
            "cross_fitting": "within each task, leave one complete init pool out",
            "cluster": "task x init_state_id",
            "continuous_metric": "OOF skill relative to training-fold mean",
            "binary_metric": "within-held-init ROC AUC",
            "bootstrap_draws": args.bootstrap,
            "ridge_alpha": args.ridge_alpha,
            "route_features": "state token soft probabilities [8 layers x 32 experts], denoise 0",
            "hard_expert_ids_used": False,
            "hidden_loaded": data.hidden is not None,
            "hidden_projection_dim": args.hidden_projection_dim if data.hidden is not None else None,
        },
        "crossfit": crossfit,
        "concepts_reported": int(len(matrix)),
        "readable_from_state_soft_routing": int(
            matrix["readable_from_state_soft_routing"].sum()
        ),
        "increment_beyond_geometry": int(matrix["increment_beyond_geometry"].sum()),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(
        render_report(matrix, data, summary["protocol"])
    )
    print(
        matrix[
            [
                "concept",
                "geometry",
                "state_soft_routing",
                "geometry_plus_state_soft",
                "joint_minus_geometry",
                "joint_minus_geometry_ci95_low",
                "joint_minus_geometry_ci95_high",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
