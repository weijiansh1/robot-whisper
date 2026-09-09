#!/usr/bin/env python3
"""Build an interpretable physical/control behavior taxonomy for failed rollouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from analyze_replanning_reset_trap import GOAL_M, LIFT_M, identify_targets


HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/failure-behavior-taxonomy"
FAILURE_MODE_CSV = HERE / "analysis/replanning-reset-trap/episode_metrics.csv"
EVENT_AUDIT_CSV = HERE / "analysis/failure-event-audit/episode_events.csv"
METHOD_REVIEW_REPORT = HERE / "analysis/failure-type-method-review/report.md"

N_CURVE = 20
STATIC_EEF_M = 0.005
STATIC_OBJECT_M = 0.003
STATIC_GRIPPER_M = 0.001
ACTIVE_EEF_M = 0.005
REVERSAL_COSINE = -0.25
APPROACH_M = 0.10
APPROACH_EXIT_M = 0.15
UNDO_EXIT_M = 0.075
CELL_SHRINKAGE = 8.0

LABELS = [
    "stagnation",
    "active_retry",
    "eef_oscillation",
    "gripper_cycling",
    "goal_regression",
    "goal_approach_leave",
    "subtask_undo",
    "regrasp_or_drop",
]
PRIMARY_PRIORITY = [
    "subtask_undo",
    "goal_approach_leave",
    "regrasp_or_drop",
    "gripper_cycling",
    "eef_oscillation",
    "active_retry",
    "stagnation",
    "goal_regression",
]

FEATURE_COLUMNS = [
    "eef_mean_step_m",
    "eef_late_mean_step_m",
    "eef_midlate_mean_step_m",
    "eef_path_efficiency",
    "eef_late_path_efficiency",
    "eef_reversal_rate",
    "eef_late_reversal_rate",
    "longest_static_full_fraction",
    "longest_static_pre90_fraction",
    "terminal_static_fraction",
    "static_onset_phase",
    "static_end_phase",
    "target_mean_step_m",
    "gripper_flip_count",
    "gripper_flip_rate",
    "gripper_late_flip_rate",
    "gripper_first_flip_phase",
    "late_goal_progress_efficiency",
    "goal_regression_m",
    "goal_regression_onset_phase",
    "goal_approach_exit_count",
    "goal_approach_exit_per_target",
    "goal_approach_first_exit_phase",
    "subtask_undo_count",
    "subtask_undo_fraction",
    "subtask_undo_first_phase",
    "lift_onset_count",
    "lift_onsets_per_target",
    "offgoal_lift_loss_count",
    "offgoal_lift_losses_per_target",
    "targets_lifted",
    "targets_reached",
    "targets_placed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def longest_true_run(mask: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(mask, dtype=bool)
    best_length = best_start = best_end = 0
    start = 0
    while start < len(values):
        if not values[start]:
            start += 1
            continue
        end = start + 1
        while end < len(values) and values[end]:
            end += 1
        if end - start > best_length:
            best_length, best_start, best_end = end - start, start, end
        start = end
    return best_length, best_start, best_end


def nonzero_sign_flip_indices(values: np.ndarray) -> np.ndarray:
    signs = np.sign(np.asarray(values, dtype=np.float64))
    nonzero = np.flatnonzero(signs != 0)
    if len(nonzero) < 2:
        return np.empty(0, dtype=np.int64)
    return nonzero[1:][signs[nonzero[1:]] != signs[nonzero[:-1]]]


def reversal_indices(delta: np.ndarray, threshold: float = ACTIVE_EEF_M) -> np.ndarray:
    vector = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(vector, axis=1)
    output = []
    for index in range(1, len(vector)):
        if norm[index - 1] <= threshold or norm[index] <= threshold:
            continue
        cosine = float(
            np.dot(vector[index - 1], vector[index]) / (norm[index - 1] * norm[index])
        )
        if cosine < REVERSAL_COSINE:
            output.append(index)
    return np.asarray(output, dtype=np.int64)


def hysteresis_exits(
    distance: np.ndarray, enter_radius: float, exit_radius: float
) -> np.ndarray:
    inside = False
    exits = []
    for index, value in enumerate(np.asarray(distance, dtype=np.float64)):
        if not inside and value <= enter_radius:
            inside = True
        elif inside and value >= exit_radius:
            exits.append(index)
            inside = False
    return np.asarray(exits, dtype=np.int64)


def resample_curve(values: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("curve must be a non-empty vector")
    if len(array) == 1:
        return np.repeat(array, size)
    return np.interp(np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, len(array)), array)


def event_curve(indices: np.ndarray, length: int, size: int = N_CURVE) -> np.ndarray:
    output = np.zeros(size, dtype=np.float64)
    if length <= 1:
        return output
    for index in np.asarray(indices, dtype=np.int64):
        phase = float(index / (length - 1))
        output[min(int(phase * size), size - 1)] += 1.0
    return output


def episode_path(client: Path, episode: int) -> Path:
    return client / f"episode_{episode:02d}.npz"


def discover_runs(cache_root: Path) -> list[Path]:
    runs = []
    for summary in sorted(cache_root.glob("libero_*/*/right-16x32/client/summaries.json")):
        run = summary.parents[1]
        if (run / "client/sim_layout.json").exists():
            runs.append(run)
    if len(runs) != 5:
        raise RuntimeError(f"expected five complete runs, found {len(runs)}")
    return runs


def extract_episode_features(
    state_full: np.ndarray,
    actions_full: np.ndarray,
    sim_full: np.ndarray,
    targets: list[dict[str, Any]],
    *,
    stop_fraction: float,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    total = len(state_full)
    keep = max(3, int(math.floor((total - 1) * stop_fraction)) + 1)
    state = np.asarray(state_full[:keep], dtype=np.float64)
    actions = np.asarray(actions_full[:keep], dtype=np.float64)
    sim = np.asarray(sim_full[:keep], dtype=np.float64)
    transitions = keep - 1
    phase = (np.arange(transitions, dtype=np.float64) + 0.5) / (total - 1)

    eef = state[:, :3]
    eef_delta = np.diff(eef, axis=0)
    eef_step = np.linalg.norm(eef_delta, axis=1)
    late = phase >= 0.5
    pre90 = phase <= 0.9 + 1e-12
    midlate = late & pre90
    terminal = phase >= max(0.0, stop_fraction - 0.1)

    gripper_aperture = state[:, 6:8].mean(axis=1)
    gripper_step = np.abs(np.diff(gripper_aperture))
    gripper_command = actions[:, :, 6].mean(axis=1)
    flip_index = nonzero_sign_flip_indices(gripper_command)
    reversals = reversal_indices(eef_delta)

    target_positions = []
    goals = []
    for target in targets:
        target_positions.append(sim[:, int(target["lo"]) : int(target["hi"])])
        goals.append(np.asarray(target["goal"], dtype=np.float64))
    n_targets = len(target_positions)
    if n_targets:
        position = np.stack(target_positions, axis=1)
        goal = np.stack(goals, axis=0)
        target_step_each = np.linalg.norm(np.diff(position, axis=0), axis=2)
        target_step = target_step_each.max(axis=1)
        distance = np.linalg.norm(position - goal[None, :, :], axis=2)
        aggregate_distance = distance.mean(axis=1)
    else:
        position = np.empty((keep, 0, 3), dtype=np.float64)
        target_step_each = np.empty((transitions, 0), dtype=np.float64)
        target_step = np.zeros(transitions, dtype=np.float64)
        distance = np.empty((keep, 0), dtype=np.float64)
        aggregate_distance = np.full(keep, np.nan, dtype=np.float64)

    static = (
        (eef_step <= STATIC_EEF_M)
        & (target_step <= STATIC_OBJECT_M)
        & (gripper_step <= STATIC_GRIPPER_M)
    )
    full_run = longest_true_run(static)
    pre90_static = static[pre90]
    pre90_run = longest_true_run(pre90_static)
    pre90_denominator = max(int(pre90.sum()), 1)
    static_onset = (
        float(phase[pre90_run[1]]) if pre90_run[0] and pre90_run[1] < len(phase) else -1.0
    )
    static_end = (
        float(phase[pre90_run[2] - 1]) if pre90_run[0] and pre90_run[2] else -1.0
    )

    eef_path = float(eef_step.sum())
    eef_net = float(np.linalg.norm(eef[-1] - eef[0]))
    late_path = float(eef_step[late].sum()) if np.any(late) else 0.0
    late_indices = np.flatnonzero(late)
    if len(late_indices):
        late_start = int(late_indices[0])
        eef_late_net = float(np.linalg.norm(eef[-1] - eef[late_start]))
    else:
        eef_late_net = 0.0

    if n_targets:
        half_state = min(int(round(0.5 * (total - 1))), keep - 1)
        goal_progress_late = float(aggregate_distance[half_state] - aggregate_distance[-1])
        late_goal_efficiency = goal_progress_late / max(late_path, 0.01)
        target_regret = distance[-1] - distance.min(axis=0)
        regression_target = int(np.argmax(target_regret))
        goal_regression = float(max(0.0, target_regret[regression_target]))
        minimum_index = int(np.argmin(distance[:, regression_target]))
        regression_onset = float(minimum_index / max(total - 1, 1))
    else:
        late_goal_efficiency = np.nan
        goal_regression = np.nan
        regression_onset = -1.0

    approach_exits = []
    undo_targets = []
    lift_onsets = []
    lift_losses = []
    lifted_count = reached_count = placed_count = 0
    for target_index in range(n_targets):
        target_distance = distance[:, target_index]
        exits = hysteresis_exits(target_distance, APPROACH_M, APPROACH_EXIT_M)
        approach_exits.extend(exits.tolist())
        at_goal = np.flatnonzero(target_distance <= GOAL_M)
        undo = np.empty(0, dtype=np.int64)
        if len(at_goal):
            after = np.flatnonzero(
                (np.arange(keep) > at_goal[0]) & (target_distance >= UNDO_EXIT_M)
            )
            undo = after[:1]
        if len(undo):
            undo_targets.append(int(undo[0]))

        lifted = position[:, target_index, 2] - position[0, target_index, 2] > LIFT_M
        onset = np.flatnonzero(lifted & ~np.r_[False, lifted[:-1]])
        loss = np.flatnonzero(~lifted & np.r_[False, lifted[:-1]])
        offgoal_loss = loss[target_distance[loss] > GOAL_M] if len(loss) else loss
        lift_onsets.extend(onset.tolist())
        lift_losses.extend(offgoal_loss.tolist())
        lifted_count += int(np.any(lifted))
        reached_count += int(np.min(target_distance) <= GOAL_M)
        placed_count += int(target_distance[-1] <= GOAL_M)

    approach_exits_array = np.asarray(sorted(approach_exits), dtype=np.int64)
    undo_array = np.asarray(sorted(undo_targets), dtype=np.int64)
    lift_onsets_array = np.asarray(sorted(lift_onsets), dtype=np.int64)
    lift_losses_array = np.asarray(sorted(lift_losses), dtype=np.int64)

    active_pairs = max(int(np.sum(eef_step > ACTIVE_EEF_M)) - 1, 1)
    late_active_pairs = max(int(np.sum((eef_step > ACTIVE_EEF_M) & late)) - 1, 1)
    late_reversals = reversals[phase[np.minimum(reversals, transitions - 1)] >= 0.5]
    late_flips = flip_index[flip_index / max(total - 1, 1) >= 0.5]

    features = {
        "eef_mean_step_m": float(eef_step.mean()),
        "eef_late_mean_step_m": float(eef_step[late].mean()) if np.any(late) else 0.0,
        "eef_midlate_mean_step_m": (
            float(eef_step[midlate].mean()) if np.any(midlate) else 0.0
        ),
        "eef_path_efficiency": eef_net / max(eef_path, 1e-8),
        "eef_late_path_efficiency": eef_late_net / max(late_path, 1e-8),
        "eef_reversal_rate": float(len(reversals) / active_pairs),
        "eef_late_reversal_rate": float(len(late_reversals) / late_active_pairs),
        "longest_static_full_fraction": float(full_run[0] / transitions),
        "longest_static_pre90_fraction": float(pre90_run[0] / pre90_denominator),
        "terminal_static_fraction": float(static[terminal].mean()) if np.any(terminal) else 0.0,
        "static_onset_phase": static_onset,
        "static_end_phase": static_end,
        "target_mean_step_m": float(target_step.mean()),
        "gripper_flip_count": float(len(flip_index)),
        "gripper_flip_rate": float(len(flip_index) / max(keep - 1, 1)),
        "gripper_late_flip_rate": float(len(late_flips) / max(int(late.sum()), 1)),
        "gripper_first_flip_phase": (
            float(flip_index[0] / max(total - 1, 1)) if len(flip_index) else -1.0
        ),
        "late_goal_progress_efficiency": late_goal_efficiency,
        "goal_regression_m": goal_regression,
        "goal_regression_onset_phase": regression_onset,
        "goal_approach_exit_count": float(len(approach_exits_array)),
        "goal_approach_exit_per_target": float(len(approach_exits_array) / max(n_targets, 1)),
        "goal_approach_first_exit_phase": (
            float(approach_exits_array[0] / max(total - 1, 1))
            if len(approach_exits_array)
            else -1.0
        ),
        "subtask_undo_count": float(len(undo_array)),
        "subtask_undo_fraction": float(len(undo_array) / max(n_targets, 1)),
        "subtask_undo_first_phase": (
            float(undo_array[0] / max(total - 1, 1)) if len(undo_array) else -1.0
        ),
        "lift_onset_count": float(len(lift_onsets_array)),
        "lift_onsets_per_target": float(len(lift_onsets_array) / max(n_targets, 1)),
        "offgoal_lift_loss_count": float(len(lift_losses_array)),
        "offgoal_lift_losses_per_target": float(len(lift_losses_array) / max(n_targets, 1)),
        "targets_lifted": float(lifted_count),
        "targets_reached": float(reached_count),
        "targets_placed": float(placed_count),
        "n_targets": float(n_targets),
    }

    speed_shape = resample_curve(eef_step, N_CURVE)
    speed_shape /= max(float(np.median(speed_shape)), 1e-8)
    static_shape = resample_curve(static.astype(np.float64), N_CURVE)
    if n_targets:
        goal_curve = resample_curve(aggregate_distance, N_CURVE + 1)
        goal_curve /= max(float(goal_curve[0]), GOAL_M)
    else:
        goal_curve = np.full(N_CURVE + 1, np.nan, dtype=np.float64)
    curves = {
        "eef_speed_m": resample_curve(eef_step, N_CURVE).astype(np.float32),
        "eef_speed_shape": speed_shape.astype(np.float32),
        "static_probability": static_shape.astype(np.float32),
        "gripper_flip_events": event_curve(flip_index, total).astype(np.float32),
        "goal_distance_normalized": goal_curve.astype(np.float32),
    }
    return features, curves


def load_feature_cohort(
    cache_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], list[dict[str, Any]]]:
    full_rows = []
    truncate_rows = []
    curve_store: dict[str, list[np.ndarray]] = defaultdict(list)
    target_audit = []
    for run in discover_runs(cache_root):
        task = str(run.relative_to(cache_root).parent)
        client = run / "client"
        summaries = sorted(
            json.loads((client / "summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        layout = json.loads((client / "sim_layout.json").read_text())
        tapes = []
        sims = []
        for episode in range(len(summaries)):
            with np.load(episode_path(client, episode), allow_pickle=False) as data:
                state = np.asarray(data["state"], dtype=np.float32)
                actions = np.asarray(data["actions"], dtype=np.float32)
                sim = np.asarray(data["sim_state"], dtype=np.float32)
            tapes.append((state, actions, sim))
            sims.append(sim)
        targets = identify_targets(summaries, layout, sims)
        target_audit.append(
            {
                "task": task,
                "targets": [str(target["name"]) for target in targets],
                "goal_positions": [np.asarray(target["goal"]).tolist() for target in targets],
            }
        )
        for episode, (summary, tape) in enumerate(zip(summaries, tapes)):
            state, actions, sim = tape
            metadata = {
                "task": task,
                "episode": episode,
                "init_state_id": int(summary["init_state_id"]),
                "flow_noise_seed": int(summary["flow_noise_seed"]),
                "episode_length": int(summary["inference_calls"]),
                "failure": not bool(summary["success"]),
            }
            if len(state) != metadata["episode_length"]:
                raise ValueError(f"episode length mismatch: {task}/{episode}")
            full_feature, curves = extract_episode_features(
                state, actions, sim, targets, stop_fraction=1.0
            )
            truncate_feature, _ = extract_episode_features(
                state, actions, sim, targets, stop_fraction=0.9
            )
            full_rows.append({**metadata, **full_feature})
            truncate_rows.append({**metadata, **truncate_feature})
            for name, values in curves.items():
                curve_store[name].append(values)
    full = pd.DataFrame(full_rows)
    truncate = pd.DataFrame(truncate_rows)
    keys = ["task", "episode", "init_state_id", "flow_noise_seed", "failure"]
    if not full[keys].equals(truncate[keys]):
        raise ValueError("full/truncate feature row mismatch")
    curves = {name: np.stack(values) for name, values in curve_store.items()}
    return full, truncate, curves, target_audit


def attach_failure_modes(frame: pd.DataFrame) -> pd.DataFrame:
    modes = pd.read_csv(FAILURE_MODE_CSV)[["task", "episode", "failure_mode"]]
    if modes.duplicated(["task", "episode"]).any():
        raise ValueError("duplicate failure-mode episode keys")
    merged = frame.merge(modes, on=["task", "episode"], how="left", validate="one_to_one")
    if merged["failure_mode"].isna().any():
        raise ValueError("missing heuristic failure-mode labels")
    return merged


def stratified_success_sample(
    frame: pd.DataFrame, rng: np.random.Generator
) -> np.ndarray:
    selected = []
    success = frame[~frame["failure"]]
    for _, subset in success.groupby(["task", "init_state_id"], sort=False):
        index = subset.index.to_numpy()
        selected.extend(rng.choice(index, size=len(index), replace=True).tolist())
    return np.asarray(selected, dtype=np.int64)


def calibrated_threshold(
    frame: pd.DataFrame,
    metric: str,
    quantile: float,
    success_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if success_index is None:
        success_index = frame.index[~frame["failure"]].to_numpy()
    sample = frame.loc[success_index, ["task", "init_state_id", metric]].copy()
    sample = sample[np.isfinite(sample[metric])]
    task_quantile = sample.groupby("task")[metric].quantile(quantile).to_dict()
    cell_quantile = sample.groupby(["task", "init_state_id"])[metric].quantile(quantile).to_dict()
    cell_count = sample.groupby(["task", "init_state_id"])[metric].size().to_dict()
    threshold = np.full(len(frame), np.nan, dtype=np.float64)
    count = np.zeros(len(frame), dtype=np.int32)
    weight = np.zeros(len(frame), dtype=np.float64)
    for index, row in enumerate(frame[["task", "init_state_id"]].itertuples(index=False)):
        task_value = task_quantile.get(row.task, np.nan)
        key = (row.task, row.init_state_id)
        n = int(cell_count.get(key, 0))
        count[index] = n
        if n and np.isfinite(task_value):
            w = n / (n + CELL_SHRINKAGE)
            threshold[index] = w * cell_quantile[key] + (1.0 - w) * task_value
            weight[index] = w
        else:
            threshold[index] = task_value
    return threshold, count, weight


def classify_behaviors(
    frame: pd.DataFrame,
    *,
    tail_quantile: float = 0.95,
    success_index: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_high = tail_quantile
    q_repeat = max(0.80, tail_quantile - 0.05)
    q_low = 1.0 - tail_quantile
    cache: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def threshold(metric: str, quantile: float) -> np.ndarray:
        key = (metric, quantile)
        if key not in cache:
            cache[key] = calibrated_threshold(frame, metric, quantile, success_index)
        return cache[key][0]

    static_threshold = np.maximum(
        threshold("longest_static_pre90_fraction", q_high), 0.15
    )
    reversal_threshold = np.maximum(threshold("eef_reversal_rate", q_high), 0.08)
    late_reversal_threshold = np.maximum(
        threshold("eef_late_reversal_rate", q_repeat), 0.06
    )
    activity_threshold = threshold("eef_late_mean_step_m", 0.25)
    efficiency_threshold = threshold("eef_late_path_efficiency", q_low)
    progress_threshold = threshold("late_goal_progress_efficiency", q_low)
    gripper_threshold = np.maximum(threshold("gripper_flip_rate", q_high), 0.06)
    late_gripper_threshold = np.maximum(
        threshold("gripper_late_flip_rate", q_repeat), 0.05
    )
    regression_threshold = np.maximum(threshold("goal_regression_m", q_high), 0.05)
    approach_threshold = np.maximum(
        threshold("goal_approach_exit_per_target", q_high), 0.25
    )
    lift_threshold = np.maximum(threshold("lift_onsets_per_target", q_high), 1.25)
    loss_threshold = np.maximum(
        threshold("offgoal_lift_losses_per_target", q_high), 0.25
    )

    labels = pd.DataFrame(index=frame.index)
    labels["stagnation"] = (
        frame["longest_static_pre90_fraction"].to_numpy() > static_threshold
    )
    active = frame["eef_late_mean_step_m"].to_numpy() >= activity_threshold
    labels["eef_oscillation"] = active & (
        frame["eef_reversal_rate"].to_numpy() > reversal_threshold
    )
    labels["gripper_cycling"] = (
        frame["gripper_flip_count"].to_numpy() >= 3
    ) & (frame["gripper_flip_rate"].to_numpy() > gripper_threshold)
    labels["goal_regression"] = (
        frame["goal_regression_m"].to_numpy() > regression_threshold
    )
    labels["goal_approach_leave"] = (
        frame["goal_approach_exit_count"].to_numpy() >= 1
    ) & (frame["goal_approach_exit_per_target"].to_numpy() > approach_threshold)
    labels["subtask_undo"] = frame["subtask_undo_count"].to_numpy() >= 1
    labels["regrasp_or_drop"] = (
        (
            (frame["lift_onset_count"].to_numpy() >= frame["n_targets"].to_numpy() + 1)
            & (frame["lift_onsets_per_target"].to_numpy() > lift_threshold)
        )
        | (
            (frame["offgoal_lift_loss_count"].to_numpy() >= 1)
            & (frame["offgoal_lift_losses_per_target"].to_numpy() > loss_threshold)
        )
    )
    repeated_control = (
        labels["eef_oscillation"].to_numpy()
        | labels["gripper_cycling"].to_numpy()
        | labels["goal_approach_leave"].to_numpy()
        | labels["regrasp_or_drop"].to_numpy()
        | (frame["eef_late_reversal_rate"].to_numpy() > late_reversal_threshold)
        | (frame["gripper_late_flip_rate"].to_numpy() > late_gripper_threshold)
    )
    poor_progress = (
        (frame["late_goal_progress_efficiency"].to_numpy() < progress_threshold)
        | labels["goal_regression"].to_numpy()
        | (frame["eef_late_path_efficiency"].to_numpy() < efficiency_threshold)
    )
    labels["active_retry"] = (
        active & poor_progress & repeated_control & ~labels["stagnation"].to_numpy()
    )
    labels = labels[LABELS].astype(bool)

    threshold_frame = pd.DataFrame(
        {
            "threshold_static_pre90": static_threshold,
            "threshold_eef_reversal": reversal_threshold,
            "threshold_late_activity_min": activity_threshold,
            "threshold_late_path_efficiency_low": efficiency_threshold,
            "threshold_late_goal_progress_low": progress_threshold,
            "threshold_gripper_flip_rate": gripper_threshold,
            "threshold_goal_regression_m": regression_threshold,
            "threshold_goal_approach_exits": approach_threshold,
            "threshold_lift_onsets_per_target": lift_threshold,
            "threshold_lift_losses_per_target": loss_threshold,
        },
        index=frame.index,
    )
    _, baseline_count, baseline_weight = calibrated_threshold(
        frame, "eef_late_mean_step_m", 0.25, success_index
    )
    threshold_frame["success_cell_n"] = baseline_count
    threshold_frame["success_cell_weight"] = baseline_weight
    return labels, threshold_frame


def attach_labels(
    frame: pd.DataFrame, labels: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    output = pd.concat([frame, thresholds], axis=1)
    for label in LABELS:
        output[f"label_{label}"] = labels[label].to_numpy()
    output["primary_behavior"] = primary_from_labels(labels)
    output["behavior_signature"] = [
        ";".join(label for label in LABELS if bool(labels.at[index, label])) or "other"
        for index in labels.index
    ]
    output["behavior_label_count"] = labels.sum(axis=1).to_numpy()
    return output


def primary_from_labels(labels: pd.DataFrame) -> np.ndarray:
    primary = np.full(len(labels), "other", dtype=object)
    for label in reversed(PRIMARY_PRIORITY):
        primary[labels[label].to_numpy(dtype=bool)] = label
    return primary


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.sum(left | right)
    return float(np.sum(left & right) / union) if union else 1.0


def label_summary(frame: pd.DataFrame) -> dict[str, Any]:
    failure = frame["failure"].to_numpy()
    result = {}
    for label in LABELS:
        values = frame[f"label_{label}"].to_numpy(dtype=bool)
        task_counts = (
            frame.loc[failure & values].groupby("task").size().sort_index().to_dict()
        )
        result[label] = {
            "failure_n": int(np.sum(failure & values)),
            "failure_rate": float(np.mean(values[failure])),
            "success_n": int(np.sum(~failure & values)),
            "success_rate": float(np.mean(values[~failure])),
            "failure_task_counts": task_counts,
        }
    return result


def overlap_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure = frame[frame["failure"]]
    counts = pd.DataFrame(0, index=LABELS, columns=LABELS, dtype=np.int64)
    overlap = pd.DataFrame(0.0, index=LABELS, columns=LABELS, dtype=np.float64)
    for left in LABELS:
        a = failure[f"label_{left}"].to_numpy(dtype=bool)
        for right in LABELS:
            b = failure[f"label_{right}"].to_numpy(dtype=bool)
            counts.loc[left, right] = int(np.sum(a & b))
            union = int(np.sum(a | b))
            overlap.loc[left, right] = float(np.sum(a & b) / union) if union else np.nan
    return counts, overlap


def bootstrap_stability(
    frame: pd.DataFrame,
    default_labels: pd.DataFrame,
    repeats: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    failure = frame["failure"].to_numpy(dtype=bool)
    failure_frame = frame.loc[failure, ["task", "episode", "init_state_id"]].reset_index(
        drop=True
    )
    if repeats <= 0:
        for label in LABELS:
            failure_frame[f"probability_{label}"] = np.nan
        return failure_frame, {"repeats": 0, "labels": {}}

    rng = np.random.default_rng(seed)
    assignment_count = np.zeros((int(failure.sum()), len(LABELS)), dtype=np.int32)
    jaccard_values = {label: [] for label in LABELS}
    positive_counts = {label: [] for label in LABELS}
    primary_agreement = []
    primary_ari = []
    default_matrix = default_labels.loc[failure, LABELS].to_numpy(dtype=bool)
    default_primary = primary_from_labels(default_labels.loc[failure, LABELS])
    for _ in range(repeats):
        success_sample = stratified_success_sample(frame, rng)
        labels, _ = classify_behaviors(frame, success_index=success_sample)
        matrix = labels.loc[failure, LABELS].to_numpy(dtype=bool)
        assignment_count += matrix
        for column, label in enumerate(LABELS):
            jaccard_values[label].append(
                jaccard(default_matrix[:, column], matrix[:, column])
            )
            positive_counts[label].append(int(matrix[:, column].sum()))
        bootstrap_primary = primary_from_labels(labels.loc[failure, LABELS])
        primary_agreement.append(float(np.mean(default_primary == bootstrap_primary)))
        primary_ari.append(float(adjusted_rand_score(default_primary, bootstrap_primary)))

    probabilities = assignment_count / repeats
    for column, label in enumerate(LABELS):
        failure_frame[f"probability_{label}"] = probabilities[:, column]

    label_stability = {}
    for column, label in enumerate(LABELS):
        default_positive = default_matrix[:, column]
        probability = probabilities[:, column]
        values = np.asarray(jaccard_values[label], dtype=np.float64)
        label_stability[label] = {
            "default_failure_n": int(default_positive.sum()),
            "jaccard_median": float(np.median(values)),
            "jaccard_p10": float(np.quantile(values, 0.10)),
            "jaccard_p90": float(np.quantile(values, 0.90)),
            "stable_episode_fraction": float(
                np.mean((probability <= 0.10) | (probability >= 0.90))
            ),
            "default_positive_probability_median": (
                float(np.median(probability[default_positive]))
                if np.any(default_positive)
                else None
            ),
            "bootstrap_positive_n_median": float(
                np.median(positive_counts[label])
            ),
        }
    stability = {
        "repeats": repeats,
        "seed": seed,
        "sampling": "success rollouts resampled with replacement within task x init",
        "primary_agreement_median": float(np.median(primary_agreement)),
        "primary_agreement_p10": float(np.quantile(primary_agreement, 0.10)),
        "primary_ari_median": float(np.median(primary_ari)),
        "primary_ari_p10": float(np.quantile(primary_ari, 0.10)),
        "labels": label_stability,
    }
    return failure_frame, stability


def stagnation_audit(
    frame: pd.DataFrame, truncate_labels: pd.DataFrame
) -> dict[str, Any]:
    failure = frame["failure"].to_numpy(dtype=bool)
    default = frame["label_stagnation"].to_numpy(dtype=bool)
    terminal = frame["terminal_static_fraction"].to_numpy() > 0.5
    full_threshold = np.maximum(
        calibrated_threshold(frame, "longest_static_full_fraction", 0.95)[0], 0.15
    )
    naive_full = frame["longest_static_full_fraction"].to_numpy() > full_threshold
    onset = frame.loc[failure & default, "static_onset_phase"].to_numpy()
    truncated = truncate_labels["stagnation"].to_numpy(dtype=bool)
    return {
        "definition_excludes_final_fraction": 0.10,
        "failure_stagnation_n": int(np.sum(failure & default)),
        "failure_naive_full_trajectory_n": int(np.sum(failure & naive_full)),
        "failure_naive_full_only_n": int(np.sum(failure & naive_full & ~default)),
        "failure_terminal_static_majority_n": int(np.sum(failure & terminal)),
        "terminal_static_but_not_stagnation_n": int(np.sum(failure & terminal & ~default)),
        "stagnation_without_terminal_static_majority_n": int(
            np.sum(failure & default & ~terminal)
        ),
        "stagnation_onset_phase_median": float(np.median(onset)),
        "stagnation_onset_phase_q25": float(np.quantile(onset, 0.25)),
        "stagnation_onset_phase_q75": float(np.quantile(onset, 0.75)),
        "stagnation_onset_before_75pct_n": int(np.sum(onset <= 0.75)),
        "truncate90_jaccard": jaccard(default[failure], truncated[failure]),
    }


def residual_audit(
    frame: pd.DataFrame, truncate_labels: pd.DataFrame
) -> dict[str, Any]:
    failure = frame["failure"].to_numpy(dtype=bool)
    residual = failure & frame["behavior_label_count"].eq(0).to_numpy()
    subset = frame.loc[residual]
    midlate_threshold = calibrated_threshold(
        frame, "eef_midlate_mean_step_m", 0.25
    )[0]
    low_midlate = frame["eef_midlate_mean_step_m"].to_numpy() < midlate_threshold
    low_path = (
        frame["eef_late_path_efficiency"].to_numpy()
        < frame["threshold_late_path_efficiency_low"].to_numpy()
    )
    low_goal = (
        frame["late_goal_progress_efficiency"].to_numpy()
        < frame["threshold_late_goal_progress_low"].to_numpy()
    )
    normalized_flip = (
        frame["gripper_flip_rate"].to_numpy()
        > frame["threshold_gripper_flip_rate"].to_numpy()
    )
    truncate_unlabeled = ~truncate_labels[LABELS].any(axis=1).to_numpy(dtype=bool)
    success_midlate = frame.loc[~failure, "eef_midlate_mean_step_m"].to_numpy()
    stagnation_midlate = frame.loc[
        failure & frame["label_stagnation"].to_numpy(dtype=bool),
        "eef_midlate_mean_step_m",
    ].to_numpy()
    activity_ratio = (
        frame["eef_midlate_mean_step_m"].to_numpy()
        / np.maximum(midlate_threshold, 1e-8)
    )
    return {
        "n": int(residual.sum()),
        "failure_fraction": float(residual.sum() / failure.sum()),
        "task_counts": subset["task"].value_counts().sort_index().to_dict(),
        "failure_mode_counts": subset["failure_mode"].value_counts().sort_index().to_dict(),
        "midlate_speed_below_success_q25_n": int(np.sum(residual & low_midlate)),
        "late_path_efficiency_below_success_q05_n": int(np.sum(residual & low_path)),
        "late_goal_progress_below_success_q05_n": int(np.sum(residual & low_goal)),
        "either_progress_metric_below_success_q05_n": int(
            np.sum(residual & (low_path | low_goal))
        ),
        "at_least_three_gripper_flips_n": int(
            np.sum(residual & (frame["gripper_flip_count"].to_numpy() >= 3))
        ),
        "normalized_gripper_flip_rate_above_success_q95_n": int(
            np.sum(residual & normalized_flip)
        ),
        "any_pre90_static_transition_n": int(
            np.sum(
                residual
                & (frame["longest_static_pre90_fraction"].to_numpy() > 0)
            )
        ),
        "terminal_static_majority_n": int(
            np.sum(residual & (frame["terminal_static_fraction"].to_numpy() > 0.5))
        ),
        "still_unlabeled_after_truncate90_n": int(
            np.sum(residual & truncate_unlabeled)
        ),
        "median_midlate_speed_m": float(subset["eef_midlate_mean_step_m"].median()),
        "success_median_midlate_speed_m": float(np.median(success_midlate)),
        "stagnation_median_midlate_speed_m": float(np.median(stagnation_midlate)),
        "residual_median_activity_ratio_to_success_q25": float(
            np.median(activity_ratio[residual])
        ),
        "success_median_activity_ratio_to_success_q25": float(
            np.median(activity_ratio[~failure])
        ),
        "stagnation_median_activity_ratio_to_success_q25": float(
            np.median(
                activity_ratio[
                    failure & frame["label_stagnation"].to_numpy(dtype=bool)
                ]
            )
        ),
    }


def event_stasis_crosscheck(frame: pd.DataFrame) -> dict[str, Any]:
    events = pd.read_csv(EVENT_AUDIT_CSV)[
        ["task", "episode", "late_motion_class", "primary_physical_pattern"]
    ]
    failure = frame[frame["failure"]].merge(
        events, on=["task", "episode"], how="left", validate="one_to_one"
    )
    if failure["late_motion_class"].isna().any():
        raise ValueError("missing event-audit rows for failure stasis cross-check")
    taxonomy = failure["label_stagnation"].to_numpy(dtype=bool)
    event_stasis = failure["late_motion_class"].eq("stasis").to_numpy()
    strict_long = failure["primary_physical_pattern"].eq(
        "stalled_at_unfinished_pot1"
    ).to_numpy()
    return {
        "source": str(EVENT_AUDIT_CSV.relative_to(HERE)),
        "method_review": str(METHOD_REVIEW_REPORT.relative_to(HERE)),
        "taxonomy_stagnation_n": int(taxonomy.sum()),
        "event_terminal_window_stasis_n": int(event_stasis.sum()),
        "intersection_n": int(np.sum(taxonomy & event_stasis)),
        "taxonomy_stagnation_later_nonstasis_n": int(
            np.sum(taxonomy & ~event_stasis)
        ),
        "strict_long_unfinished_pot1_stasis_n": int(strict_long.sum()),
        "strict_long_also_taxonomy_stagnation_n": int(
            np.sum(strict_long & taxonomy)
        ),
        "interpretation": (
            "taxonomy detects any long joint-static run before 90%; event stasis "
            "tests the final quarter, so taxonomy positives need not remain terminally static"
        ),
    }


def failure_mode_tables(frame: pd.DataFrame) -> dict[str, Any]:
    failure = frame[frame["failure"]]
    primary = pd.crosstab(failure["primary_behavior"], failure["failure_mode"])
    multilabel = {}
    for label in LABELS:
        subset = failure[failure[f"label_{label}"]]
        multilabel[label] = subset["failure_mode"].value_counts().sort_index().to_dict()
    return {
        "primary": primary.to_dict(orient="index"),
        "multilabel": multilabel,
    }


def quick_sensitivity(
    full: pd.DataFrame,
    truncate: pd.DataFrame,
    default_labels: pd.DataFrame,
) -> dict[str, Any]:
    failure = full["failure"].to_numpy()
    output: dict[str, Any] = {"quantile": {}, "truncate90": {}}
    for quantile in (0.90, 0.975):
        alternative, _ = classify_behaviors(full, tail_quantile=quantile)
        output["quantile"][str(quantile)] = {
            label: jaccard(
                default_labels.loc[failure, label].to_numpy(),
                alternative.loc[failure, label].to_numpy(),
            )
            for label in LABELS
        }
    truncate_labels, _ = classify_behaviors(truncate)
    output["truncate90"] = {
        label: jaccard(
            default_labels.loc[failure, label].to_numpy(),
            truncate_labels.loc[failure, label].to_numpy(),
        )
        for label in LABELS
    }
    output["truncate90"]["primary_ari"] = float(
        adjusted_rand_score(
            primary_from_labels(default_labels.loc[failure, LABELS]),
            primary_from_labels(truncate_labels.loc[failure, LABELS]),
        )
    )
    return output


def make_overview_plot(
    frame: pd.DataFrame, sensitivity: dict[str, Any], output: Path
) -> None:
    failure = frame["failure"].to_numpy()
    rates_failure = [frame.loc[failure, f"label_{label}"].mean() for label in LABELS]
    rates_success = [frame.loc[~failure, f"label_{label}"].mean() for label in LABELS]
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.float64)
    for left, label_left in enumerate(LABELS):
        a = frame.loc[failure, f"label_{label_left}"].to_numpy(dtype=bool)
        for right, label_right in enumerate(LABELS):
            b = frame.loc[failure, f"label_{label_right}"].to_numpy(dtype=bool)
            union = int(np.sum(a | b))
            matrix[left, right] = float(np.sum(a & b) / union) if union else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    x = np.arange(len(LABELS))
    axes[0].bar(x - 0.18, rates_success, 0.36, color="#6B7280", label="success")
    axes[0].bar(x + 0.18, rates_failure, 0.36, color="#C43C39", label="failure")
    axes[0].set_xticks(x, [label.replace("_", "\n") for label in LABELS], rotation=25)
    axes[0].set_ylabel("episode fraction")
    axes[0].set_title("Behavior prevalence")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#F3F4F6")
    image = axes[1].imshow(matrix, vmin=0.0, vmax=1.0, cmap=colormap)
    axes[1].set_xticks(x, [label.replace("_", "\n") for label in LABELS], rotation=45)
    axes[1].set_yticks(x, [label.replace("_", " ") for label in LABELS])
    axes[1].set_title("Failure-label Jaccard overlap")
    fig.colorbar(image, ax=axes[1], fraction=0.046)

    trunc = [sensitivity["truncate90"][label] for label in LABELS]
    q90 = [sensitivity["quantile"]["0.9"][label] for label in LABELS]
    q975 = [sensitivity["quantile"]["0.975"][label] for label in LABELS]
    axes[2].plot(x, trunc, marker="o", label="truncate90")
    axes[2].plot(x, q90, marker="s", label="success q90")
    axes[2].plot(x, q975, marker="^", label="success q97.5")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_xticks(x, [label.replace("_", "\n") for label in LABELS], rotation=25)
    axes[2].set_ylabel("Jaccard with default")
    axes[2].set_title("Sensitivity")
    axes[2].legend(frameon=False)
    axes[2].grid(alpha=0.2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_profile_plot(
    frame: pd.DataFrame, curves: dict[str, np.ndarray], output: Path
) -> None:
    failure = frame["failure"].to_numpy(dtype=bool)
    groups = {
        "success": ~failure,
        "stagnation": failure & frame["label_stagnation"].to_numpy(dtype=bool),
        "lift/drop proxy": failure
        & frame["label_regrasp_or_drop"].to_numpy(dtype=bool),
        "unlabeled residual": failure
        & frame["behavior_label_count"].eq(0).to_numpy(),
    }
    colors = {
        "success": "#4C78A8",
        "stagnation": "#C43C39",
        "lift/drop proxy": "#E28E2C",
        "unlabeled residual": "#4E8B57",
    }
    speed_scale = np.maximum(
        frame["threshold_midlate_activity_q25"].to_numpy(dtype=np.float64), 1e-6
    )
    panels = [
        (
            curves["eef_speed_m"] / speed_scale[:, None],
            np.linspace(0.0, 1.0, N_CURVE),
            "EEF speed / sibling-success q25",
            "median ratio",
            False,
        ),
        (
            curves["static_probability"],
            np.linspace(0.0, 1.0, N_CURVE),
            "Static-state profile",
            "median indicator",
            False,
        ),
        (
            curves["goal_distance_normalized"],
            np.linspace(0.0, 1.0, N_CURVE + 1),
            "Goal distance / initial distance",
            "median ratio",
            False,
        ),
        (
            np.cumsum(curves["gripper_flip_events"], axis=1),
            np.linspace(0.0, 1.0, N_CURVE),
            "Cumulative gripper sign changes",
            "mean events",
            True,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for axis, (values, phase, title, ylabel, use_mean) in zip(axes.flat, panels):
        for name, mask in groups.items():
            subset = np.asarray(values[mask], dtype=np.float64)
            if not len(subset) or np.all(~np.isfinite(subset)):
                continue
            if use_mean:
                center = np.nanmean(subset, axis=0)
                axis.plot(phase, center, color=colors[name], linewidth=2.0, label=name)
            else:
                center = np.nanmedian(subset, axis=0)
                lower = np.nanquantile(subset, 0.25, axis=0)
                upper = np.nanquantile(subset, 0.75, axis=0)
                axis.plot(phase, center, color=colors[name], linewidth=2.0, label=name)
                axis.fill_between(phase, lower, upper, color=colors[name], alpha=0.12)
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel("normalized episode phase")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.2)
    axes[0, 0].axhline(1.0, color="#111827", linewidth=1.0, linestyle="--", alpha=0.5)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_report(summary: dict[str, Any]) -> str:
    labels = summary["labels"]
    sensitivity = summary["sensitivity"]
    stability = summary["bootstrap_stability"]
    residual = summary["residual_audit"]
    stagnation = summary["stagnation_audit"]
    event_crosscheck = summary["event_stasis_crosscheck"]
    primary_counts = summary["primary_counts"]
    lines = [
        "# 失败 rollout 的物理/控制行为谱",
        "",
        "## 直接结论",
        "",
        "这里不把 307 条失败强行切成互斥簇，而是给每条 rollout 标注可重叠的物理/控制行为。所有连续阈值都只由同 task、同初态的成功 siblings 校准；初态成功样本不足时向 task 成功阈值收缩。episode length、task id 和终点坐标都不作为分类特征。",
        "",
        f"最强的结构是停滞：{labels['stagnation']['failure_n']}/307 条失败满足，而且 primary label 中有 {primary_counts.get('stagnation', 0)} 条。repeated-lift/drop proxy 覆盖 {labels['regrasp_or_drop']['failure_n']} 条；它比旧 metadata 中保守的 transport-loss 更宽，只能解释为运动学代理。",
        "",
        f"另外有 {residual['n']} 条（{residual['failure_fraction']:.1%}）没有满足任何离散行为规则。它们并非像成功轨迹一样正常：其中 {residual['midlate_speed_below_success_q25_n']} 条在 50%--90% 相位的 EEF 速度低于 sibling-success q25，{residual['either_progress_metric_below_success_q05_n']} 条至少一个进展效率低于 success q05。更准确的描述是“分散式减速和低进展”，而不是新的一种异常重试簇。",
        "",
        "`active_retry` 和 `eef_oscillation` 在失败中都为 0。这是需要保留的负结果：此前路由残余组相对停滞核心更活跃，不等于它们相对成功轨迹出现了异常重试。",
        "",
        "停滞规则只看 0%--90% 相位中的连续低运动段，最后 10% 完全不参与定义；下面另有专门审计。",
        "",
        "## 标签数量",
        "",
        "| 行为 | failure n/rate | success n/rate | truncate90 Jaccard | bootstrap Jaccard median [p10,p90] |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        row = labels[label]
        boot = stability["labels"].get(label, {})
        if row["failure_n"]:
            bootstrap_text = (
                f"{boot['jaccard_median']:.3f} "
                f"[{boot['jaccard_p10']:.3f},{boot['jaccard_p90']:.3f}]"
            )
        else:
            bootstrap_text = "vacuous (0 positive)"
        lines.append(
            f"| {label} | {row['failure_n']} / {row['failure_rate']:.3f} | "
            f"{row['success_n']} / {row['success_rate']:.3f} | "
            f"{sensitivity['truncate90'][label]:.3f} | "
            f"{bootstrap_text} |"
        )
    lines.extend(["", "## Primary behavior（仅为摘要，不替代多标签）", ""])
    lines.extend(["| primary | n | fraction |", "|---|---:|---:|"])
    for name, count in sorted(primary_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {name} | {count} | {count / 307:.3f} |")

    lines.extend(["", "## Multi-label overlap", ""])
    overlap_counts = summary["overlap_counts"]
    overlap_jaccard = summary["overlap_jaccard"]
    pairs = []
    for left_index, left in enumerate(LABELS):
        for right in LABELS[left_index + 1 :]:
            count = int(overlap_counts[left][right])
            if count:
                pairs.append((count, left, right, float(overlap_jaccard[left][right])))
    lines.extend(["| label A | label B | intersection n | Jaccard |", "|---|---|---:|---:|"])
    for count, left, right, value in sorted(pairs, reverse=True):
        lines.append(f"| {left} | {right} | {count} | {value:.3f} |")
    if not pairs:
        lines.append("| none | none | 0 | 0.000 |")

    lines.extend(
        [
            "",
            "## 未标注 residual 的特点",
            "",
            f"| 检查项 | n / {residual['n']} |",
            "|---|---:|",
            f"| 50%--90% EEF speed < sibling-success q25 | {residual['midlate_speed_below_success_q25_n']} |",
            f"| late path efficiency < success q05 | {residual['late_path_efficiency_below_success_q05_n']} |",
            f"| late goal progress < success q05 | {residual['late_goal_progress_below_success_q05_n']} |",
            f"| 至少 3 次 gripper 符号切换 | {residual['at_least_three_gripper_flips_n']} |",
            f"| 归一化 flip rate > success q95 | {residual['normalized_gripper_flip_rate_above_success_q95_n']} |",
            f"| 有任意 pre-90% 静止 transition，但没有长停滞段 | {residual['any_pre90_static_transition_n']} |",
            f"| 末端 10% 多数静止 | {residual['terminal_static_majority_n']} |",
            f"| 去掉末尾 10% 后仍未标注 | {residual['still_unlabeled_after_truncate90_n']} |",
            "",
            f"按各自 sibling-success q25 归一化后，50%--90% EEF 活动比的中位数依次是：成功 {residual['success_median_activity_ratio_to_success_q25']:.2f}、未标注 residual {residual['residual_median_activity_ratio_to_success_q25']:.2f}、停滞 {residual['stagnation_median_activity_ratio_to_success_q25']:.2f}。所以 residual 位于连续的 activity-to-stasis 轴中间，并没有支持把它们再硬切成几种稳定循环。",
            "",
            f"而且 residual 有 {residual['task_counts'].get('libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove', 0)}/{residual['n']} 条来自双 moka-pot 任务，旧结局标签中 {residual['failure_mode_counts'].get('partial', 0)}/{residual['n']} 条是 `partial`。其余还横跨 never_grasped、misplaced、lifted_not_placed 和 reached_then_lost。这个任务集中性和结局异质性进一步反对把 residual 命名成一个普适机制。",
            "",
            "夹爪结果尤其说明长度归一化为什么重要：许多 residual 有多次符号切换，但按相位/transition 归一化后没有一条超过成功 q95，不能称为异常 cycling。",
            "",
            "## 停滞不是末端低位移的自动标签",
            "",
            f"- 默认 pre-90% 规则标出 {stagnation['failure_stagnation_n']} 条；直接看整条轨迹的 naive 规则标出 {stagnation['failure_naive_full_trajectory_n']} 条，其中只有 {stagnation['failure_naive_full_only_n']} 条是整条规则独有。",
            f"- 有 {stagnation['terminal_static_but_not_stagnation_n']} 条失败虽然末端 10% 多数静止，却没有被标成 stagnation；反过来有 {stagnation['stagnation_without_terminal_static_majority_n']} 条 stagnation 在末端并非多数静止。",
            f"- 停滞最长段开始相位中位数为 {stagnation['stagnation_onset_phase_median']:.3f}（IQR {stagnation['stagnation_onset_phase_q25']:.3f}--{stagnation['stagnation_onset_phase_q75']:.3f}），{stagnation['stagnation_onset_before_75pct_n']} 条在 75% 相位前已经开始。",
            f"- 砍掉最后 10% 后 failure assignment Jaccard={stagnation['truncate90_jaccard']:.3f}。",
            f"- 独立末 25% event audit 的 stasis 是 {event_crosscheck['event_terminal_window_stasis_n']} 条，与 taxonomy 166 的交集为 {event_crosscheck['intersection_n']}；taxonomy 的 166 中有 {event_crosscheck['taxonomy_stagnation_later_nonstasis_n']} 条后来恢复过运动。因此 166 的准确含义是“前 90% 曾出现长连续全系统低变化段”，不是“166 条都静止到 timeout”。",
            f"- 双 moka-pot 中严格的 `stalled_at_unfinished_pot1` 有 {event_crosscheck['strict_long_unfinished_pot1_stasis_n']} 条，{event_crosscheck['strict_long_also_taxonomy_stagnation_n']}/{event_crosscheck['strict_long_unfinished_pot1_stasis_n']} 同时属于 taxonomy stagnation；独立方法复核还确认这 109 条与 MoE stagnation core 一致。来源见 `{event_crosscheck['method_review']}`。",
            "",
            "## 与旧 heuristic failure_mode 的交叉表",
            "",
        ]
    )
    mode_table = summary["failure_mode_cross_tables"]["primary"]
    modes = sorted({mode for row in mode_table.values() for mode in row})
    lines.append("| primary | " + " | ".join(modes) + " | total |")
    lines.append("|---|" + "---:|" * (len(modes) + 1))
    for primary, row in sorted(mode_table.items()):
        values = [int(row.get(mode, 0)) for mode in modes]
        lines.append(
            f"| {primary} | " + " | ".join(map(str, values)) + f" | {sum(values)} |"
        )
    lines.extend(
        [
            "",
            "这里的 `failure_mode` 只用于外部一致性检查，不参与行为规则；因此交叉表不是标签定义的同义反复。",
        ]
    )
    lines.extend(
        [
            "",
            "## 可解释规则",
            "",
            f"- `stagnation`: 前 90% 相位内，EEF 每 query 位移不超过 {STATIC_EEF_M * 1000:.0f} mm、任务物体不超过 {STATIC_OBJECT_M * 1000:.0f} mm、夹爪孔径变化不超过 {STATIC_GRIPPER_M * 1000:.0f} mm 的最长连续段，超过成功 q95（且至少占 15%）。",
            "- `active_retry`: 后半段仍达到成功 q25 的运动速度，但目标进展效率处于成功 q05 以下，并同时出现 EEF 反向、夹爪反转、目标接近后离开或重新抓取信号。",
            f"- `eef_oscillation`: 有效 EEF 位移之间夹角余弦小于 {REVERSAL_COSINE:.2f} 的反向率超过成功 q95。",
            "- `gripper_cycling`: chunk 平均夹爪命令的符号反转率超过成功 q95，且至少出现 3 次反转。",
            f"- `goal_regression`: 相对历史最佳目标距离回退至少 {0.05 * 100:.0f} cm，并超过成功 q95。",
            f"- `goal_approach_leave`: 进入目标 {APPROACH_M * 100:.0f} cm 后又退出到 {APPROACH_EXIT_M * 100:.0f} cm 外，事件率超过成功 q95。",
            f"- `subtask_undo`: 任务物体曾到达 {GOAL_M * 100:.0f} cm 目标区，之后又离开到 {UNDO_EXIT_M * 100:.1f} cm 外。",
            "- `regrasp_or_drop`: 单个目标重复 lift，或 lifted 后在目标区外掉落，并超过成功 q95。",
            "",
            f"Bootstrap 共 {stability['repeats']} 次，每次在 task x init 内对成功 siblings 有放回重采样；primary agreement 中位数为 {stability['primary_agreement_median']:.3f}，p10={stability['primary_agreement_p10']:.3f}。q90/q97.5 和 truncate90 的完整敏感性结果在 `summary.json`。",
            "",
            "## 限制",
            "",
            "- 这些是 query 边界上的运动学代理，不是人工视频语义标注。",
            "- 多标签重叠是有意保留的；例如 active retry 可以同时伴随夹爪 cycling。",
            "- success-normalized 阈值减少 task/init 混杂，但不能证明任何内部路由信号导致了这些行为。",
            "",
            "## 产物",
            "",
            "- `episode_features_labels.csv`: 逐 episode 特征、阈值、多标签和 primary label。",
            "- `behavior_curves.npz`: 20 相位压缩曲线；不含原始 sim state 或路由矩阵。",
            "- `bootstrap_assignment_stability.csv`: 每条失败在成功基线重采样下的标签概率。",
            "- `thresholds_by_task_init.csv`: success-normalized 阈值审计表。",
            "- `multilabel_overlap_counts.csv` / `multilabel_overlap_jaccard.csv`: 失败标签重叠。",
            "- `summary.json`: 标签、failure_mode 交叉表和敏感性结果。",
            "- `behavior_overview.png`: prevalence、标签重叠和敏感性；`normalized_behavior_profiles.png`: 相对相位行为曲线。",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    assert longest_true_run(np.asarray([0, 1, 1, 0, 1], bool)) == (2, 1, 3)
    assert nonzero_sign_flip_indices(np.asarray([-1, -1, 1, 1, -1])).tolist() == [2, 4]
    assert hysteresis_exits(np.asarray([0.2, 0.08, 0.12, 0.16]), 0.1, 0.15).tolist() == [3]
    delta = np.asarray([[0.01, 0, 0], [-0.01, 0, 0], [0.01, 0, 0]])
    assert reversal_indices(delta).tolist() == [1, 2]
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    full, truncate, curves, target_audit = load_feature_cohort(args.cache_root)
    full = attach_failure_modes(full)
    truncate = attach_failure_modes(truncate)
    labels, thresholds = classify_behaviors(full)
    thresholds["threshold_midlate_activity_q25"] = calibrated_threshold(
        full, "eef_midlate_mean_step_m", 0.25
    )[0]
    output = attach_labels(full, labels, thresholds)
    truncate_labels, _ = classify_behaviors(truncate)
    sensitivity = quick_sensitivity(full, truncate, labels)
    bootstrap_frame, bootstrap_summary = bootstrap_stability(
        full, labels, args.bootstrap, args.seed
    )
    overlap_counts, overlap_jaccard = overlap_tables(output)
    failure_output = output[output["failure"]]
    summary = {
        "schema": "himoe.failure_behavior_taxonomy.v2",
        "scope": {
            "episodes": len(output),
            "failures": int(output["failure"].sum()),
            "successes": int((~output["failure"]).sum()),
            "relative_time_only": True,
            "episode_length_used_for_rules": False,
        },
        "definitions": {
            "static_thresholds_m": {
                "eef": STATIC_EEF_M,
                "object": STATIC_OBJECT_M,
                "gripper_aperture": STATIC_GRIPPER_M,
            },
            "goal_radii_m": {
                "goal": GOAL_M,
                "undo_exit": UNDO_EXIT_M,
                "approach": APPROACH_M,
                "approach_exit": APPROACH_EXIT_M,
            },
            "success_baseline": "task x init empirical quantile shrunk toward task quantile",
            "cell_shrinkage": CELL_SHRINKAGE,
        },
        "target_audit": target_audit,
        "labels": label_summary(output),
        "primary_counts": failure_output["primary_behavior"]
        .value_counts()
        .to_dict(),
        "overlap_counts": overlap_counts.to_dict(orient="index"),
        "overlap_jaccard": overlap_jaccard.to_dict(orient="index"),
        "failure_mode_cross_tables": failure_mode_tables(output),
        "sensitivity": sensitivity,
        "bootstrap_stability": bootstrap_summary,
        "stagnation_audit": stagnation_audit(output, truncate_labels),
        "event_stasis_crosscheck": event_stasis_crosscheck(output),
        "residual_audit": residual_audit(output, truncate_labels),
    }

    assert len(output) == 2560
    assert int(output["failure"].sum()) == 307
    assert output.loc[output["failure"], "failure_mode"].ne("success").all()
    assert output.loc[~output["failure"], "failure_mode"].eq("success").all()
    assert np.isfinite(output[FEATURE_COLUMNS[:16]].to_numpy()).all()
    assert output["success_cell_n"].ge(0).all()

    output.to_csv(args.out_dir / "episode_features_labels.csv", index=False)
    bootstrap_frame.to_csv(
        args.out_dir / "bootstrap_assignment_stability.csv", index=False
    )
    threshold_columns = [column for column in output if column.startswith("threshold_")]
    (
        output[
            ["task", "init_state_id", "success_cell_n", "success_cell_weight"]
            + threshold_columns
        ]
        .drop_duplicates(["task", "init_state_id"])
        .sort_values(["task", "init_state_id"])
        .to_csv(args.out_dir / "thresholds_by_task_init.csv", index=False)
    )
    overlap_counts.to_csv(args.out_dir / "multilabel_overlap_counts.csv")
    overlap_jaccard.to_csv(args.out_dir / "multilabel_overlap_jaccard.csv")
    np.savez_compressed(args.out_dir / "behavior_curves.npz", **curves)
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "report.md").write_text(
        render_report(plain(summary)), encoding="utf-8"
    )
    make_overview_plot(output, sensitivity, args.out_dir / "behavior_overview.png")
    make_profile_plot(
        output, curves, args.out_dir / "normalized_behavior_profiles.png"
    )
    print("analysis complete")


if __name__ == "__main__":
    main()
