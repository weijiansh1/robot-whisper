#!/usr/bin/env python3
"""Build a simulator-kinematic event ledger for all failed rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_post_error_recovery import (
    LONG_GRASP_NEAR_M,
    LONG_HOLD_DISTANCE_M,
    MAIN_RULE,
    OTHER_GRASP_NEAR_M,
    OTHER_HOLD_DISTANCE_M,
    detect_target_events,
    evidence_tape,
    identify_targets,
)


HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/failure-event-audit"
FAILURE_MODE_CSV = HERE / "analysis/replanning-reset-trap/episode_metrics.csv"
GOAL_DISTANCE_M = 0.05
LIFT_DISTANCE_M = 0.01
DISTRACTOR_MIN_DISPLACEMENT_M = 0.03
ACTIVE_STEP_M = 0.01
ACTIVE_FRACTION = 0.50
STASIS_STEP_M = 0.005
STASIS_FRACTION = 0.25
SEED = 20260828

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
FAILURE_TASKS = (TOP_DRAWER_TASK, LONG_TASK, RAMEKIN_TASK, STOVE_TASK)
DRAWER_JOINT = "wooden_cabinet_1_top_level"
GOAL_RECEPTACLE_JOINT = {
    RAMEKIN_TASK: "plate_1_joint0",
    STOVE_TASK: "plate_1_joint0",
}


@dataclass(frozen=True)
class EpisodeTape:
    episode: int
    init_state_id: int
    flow_noise_seed: int
    success: bool
    state: np.ndarray
    actions: np.ndarray
    sim: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_bouts(mask: np.ndarray) -> tuple[int, list[int]]:
    values = np.asarray(mask, dtype=bool)
    if not len(values):
        return 0, []
    onset = np.flatnonzero(values & ~np.r_[False, values[:-1]])
    return int(len(onset)), list(map(int, onset))


def continued_hold_bouts(
    position: np.ndarray,
    state: np.ndarray,
    hold_distance_m: float,
) -> list[tuple[int, int]]:
    eef = state[:, :3]
    gap = np.mean(np.abs(state[:, 6:8]), axis=1)
    evidence, distance, _step, _residual = evidence_tape(
        position, eef, gap, hold_distance_m, MAIN_RULE
    )
    result: list[tuple[int, int]] = []
    active = False
    start = -1
    for query in range(len(position)):
        if not active:
            if evidence[query]:
                active, start = True, query
            continue
        if (
            distance[query] < hold_distance_m
            and gap[query] < MAIN_RULE.continue_gap_max_m
        ):
            continue
        result.append((start, query - 1))
        active = False
        if evidence[query]:
            active, start = True, query
    if active:
        result.append((start, len(position) - 1))
    return result


def load_task(
    cache_root: Path, task: str
) -> tuple[list[dict[str, Any]], dict[str, Any], list[EpisodeTape]]:
    client = cache_root / task / "right-16x32/client"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if [int(row["episode_index"]) for row in summaries] != list(range(len(summaries))):
        raise ValueError(f"non-contiguous episodes for {task}")
    layout = json.loads((client / "sim_layout.json").read_text())
    tapes: list[EpisodeTape] = []
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(client / f"episode_{episode:02d}.npz", allow_pickle=False) as data:
            state = np.asarray(data["state"], dtype=np.float64)
            actions = np.asarray(data["actions"], dtype=np.float64)
            sim = np.asarray(data["sim_state"], dtype=np.float64)
        expected = int(row["inference_calls"])
        if state.shape != (expected, 8) or actions.shape != (expected, 10, 7):
            raise ValueError(f"state/action shape drift in {task} episode {episode}")
        if len(sim) != expected:
            raise ValueError(f"sim length drift in {task} episode {episode}")
        tapes.append(
            EpisodeTape(
                episode=episode,
                init_state_id=int(row["init_state_id"]),
                flow_noise_seed=int(row["flow_noise_seed"]),
                success=bool(row["success"]),
                state=state,
                actions=actions,
                sim=sim,
            )
        )
    return summaries, layout, tapes


def terminal_goal_distances(
    tapes: list[EpisodeTape], lo: int, hi: int
) -> np.ndarray:
    return np.stack(
        [tape.sim[-1, lo:hi] for tape in tapes if tape.success], axis=0
    )


def distance_to_success_endpoints(
    position: np.ndarray, endpoints: np.ndarray
) -> np.ndarray:
    return np.linalg.norm(position[:, None, :] - endpoints[None, :, :], axis=2).min(
        axis=1
    )


def movement_class(state: np.ndarray) -> dict[str, Any]:
    steps = np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1)
    start = int(np.floor(0.75 * len(steps)))
    late = steps[start:]
    mean = float(late.mean())
    fraction = float(np.mean(late > ACTIVE_STEP_M))
    if mean >= ACTIVE_STEP_M and fraction >= ACTIVE_FRACTION:
        label = "active"
    elif mean < STASIS_STEP_M and fraction <= STASIS_FRACTION:
        label = "stasis"
    else:
        label = "intermittent"
    return {
        "late_motion_class": label,
        "late_eef_step_mean_m": mean,
        "late_eef_moving_fraction": fraction,
        "late_eef_path_m": float(late.sum()),
        "total_eef_path_m": float(steps.sum()),
        "late_window_first_transition": start,
    }


def target_analysis(
    tape: EpisodeTape,
    targets: list[Any],
    endpoints: dict[str, np.ndarray],
    near_distance_m: float,
    hold_distance_m: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    close_command = tape.actions[:, :, 6].mean(axis=1) > 0.5
    target_rows: list[dict[str, Any]] = []
    progress_curves: list[np.ndarray] = []
    for target in targets:
        position = tape.sim[:, target.lo : target.hi]
        distance = distance_to_success_endpoints(position, endpoints[target.name])
        initial = max(float(distance[0]), GOAL_DISTANCE_M)
        progress = 1.0 - distance / initial
        progress_curves.append(progress)
        eef_distance = np.linalg.norm(position - tape.state[:, :3], axis=1)
        near_close = (eef_distance < near_distance_m) & close_command
        near_bouts, near_onsets = count_bouts(near_close)
        hold_bouts = continued_hold_bouts(position, tape.state, hold_distance_m)
        drops, _evidence, _distance = detect_target_events(
            position, tape.state, tape.actions, hold_distance_m
        )
        reached = distance <= GOAL_DISTANCE_M
        target_rows.append(
            {
                "name": target.name,
                "initial_goal_distance_m": float(distance[0]),
                "minimum_goal_distance_m": float(distance.min()),
                "final_goal_distance_m": float(distance[-1]),
                "max_progress": float(progress.max()),
                "final_progress": float(progress[-1]),
                "max_progress_query": int(np.argmax(progress)),
                "goal_ever": bool(reached.any()),
                "goal_final": bool(reached[-1]),
                "goal_first_query": (
                    int(np.flatnonzero(reached)[0]) if reached.any() else None
                ),
                "goal_loss": bool(reached.any() and not reached[-1]),
                "lift_m": float(np.max(position[:, 2] - position[0, 2])),
                "near_close_bouts": near_bouts,
                "near_close_onsets": near_onsets,
                "hold_bouts": len(hold_bouts),
                "hold_intervals": hold_bouts,
                "drop_proxy_events": len(drops),
                "eef_min_distance_m": float(eef_distance.min()),
                "eef_terminal_distance_m": float(eef_distance[-1]),
            }
        )
    return target_rows, progress_curves


def drawer_reference(
    layout: dict[str, Any], tapes: list[EpisodeTape]
) -> dict[str, Any] | None:
    if not any(joint["joint"] == DRAWER_JOINT for joint in layout["joints"]):
        return None
    joint = next(joint for joint in layout["joints"] if joint["joint"] == DRAWER_JOINT)
    lo = int(joint["state_lo"])
    successful_delta = np.asarray(
        [tape.sim[-1, lo] - tape.sim[0, lo] for tape in tapes if tape.success]
    )
    direction = float(np.sign(np.median(successful_delta)))
    signed = direction * successful_delta
    return {
        "lo": lo,
        "direction": direction,
        "reference": float(np.median(signed)),
        "goal_threshold": float(np.quantile(signed, 0.01)),
    }


def drawer_analysis(
    tape: EpisodeTape, reference: dict[str, Any] | None
) -> tuple[dict[str, Any], np.ndarray | None]:
    if reference is None:
        return {}, None
    lo = int(reference["lo"])
    signed = reference["direction"] * (tape.sim[:, lo] - tape.sim[0, lo])
    progress = signed / max(float(reference["reference"]), 1e-8)
    reached = signed >= float(reference["goal_threshold"])
    return (
        {
            "drawer_open_max_m": float(signed.max()),
            "drawer_open_final_m": float(signed[-1]),
            "drawer_goal_threshold_m": float(reference["goal_threshold"]),
            "drawer_goal_ever": bool(reached.any()),
            "drawer_goal_final": bool(reached[-1]),
            "drawer_goal_loss": bool(reached.any() and not reached[-1]),
            "drawer_goal_first_query": (
                int(np.flatnonzero(reached)[0]) if reached.any() else None
            ),
        },
        progress,
    )


def distractor_thresholds(
    layout: dict[str, Any], tapes: list[EpisodeTape], target_names: set[str]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    distractors = [
        joint
        for joint in layout["joints"]
        if not bool(joint["is_robot"])
        and int(joint["state_hi"]) - int(joint["state_lo"]) == 7
        and str(joint["joint"]) not in target_names
    ]
    thresholds: dict[str, float] = {}
    for joint in distractors:
        lo = int(joint["state_lo"])
        movement = []
        for tape in tapes:
            if not tape.success:
                continue
            position = tape.sim[:, lo : lo + 3]
            movement.append(
                float(np.linalg.norm(position - position[:1], axis=1).max())
            )
        thresholds[str(joint["joint"])] = max(
            DISTRACTOR_MIN_DISPLACEMENT_M, float(np.quantile(movement, 0.99))
        )
    return distractors, thresholds


def distractor_analysis(
    tape: EpisodeTape,
    targets: list[Any],
    distractors: list[dict[str, Any]],
    thresholds: dict[str, float],
    near_distance_m: float,
    goal_receptacle: str | None,
) -> dict[str, Any]:
    target_distance = np.min(
        [
            np.linalg.norm(
                tape.sim[:, target.lo : target.hi] - tape.state[:, :3], axis=1
            )
            for target in targets
        ],
        axis=0,
    )
    close = tape.actions[:, :, 6].mean(axis=1) > 0.5
    close_onsets = np.flatnonzero(close & ~np.r_[False, close[:-1]])
    wrong_object_events = []
    goal_side_events = []
    movement_outliers = []
    for joint in distractors:
        name = str(joint["joint"])
        lo = int(joint["state_lo"])
        position = tape.sim[:, lo : lo + 3]
        displacement = float(np.linalg.norm(position - position[:1], axis=1).max())
        if displacement > thresholds[name]:
            movement_outliers.append(
                {
                    "joint": name,
                    "max_displacement_m": displacement,
                    "success_q99_threshold_m": thresholds[name],
                }
            )
        distance = np.linalg.norm(position - tape.state[:, :3], axis=1)
        for query in close_onsets:
            if distance[query] >= near_distance_m or target_distance[query] < near_distance_m:
                continue
            event = {
                "query": int(query),
                "joint": name,
                "eef_joint_distance_m": float(distance[query]),
                "eef_target_distance_m": float(target_distance[query]),
            }
            if name == goal_receptacle:
                goal_side_events.append(event)
            else:
                wrong_object_events.append(event)
    return {
        "distractor_displacement_outlier": bool(movement_outliers),
        "distractor_displacement_details": movement_outliers,
        "wrong_object_close_proxy": bool(wrong_object_events),
        "wrong_object_close_details": wrong_object_events,
        "goal_side_without_target_close_proxy": bool(goal_side_events),
        "goal_side_close_details": goal_side_events,
    }


def long_controller_analysis(
    tape: EpisodeTape,
    target_rows: list[dict[str, Any]],
    targets: list[Any],
) -> dict[str, Any]:
    if len(targets) != 2:
        raise ValueError("long task target count drifted")
    by_name = {row["name"]: row for row in target_rows}
    positions = {
        target.name: tape.sim[:, target.lo : target.hi] for target in targets
    }
    pot1_name = "moka_pot_1_joint0"
    pot2_name = "moka_pot_2_joint0"
    pot1 = positions[pot1_name]
    pot2 = positions[pot2_name]
    row1 = by_name[pot1_name]
    row2 = by_name[pot2_name]
    start = row2["goal_first_query"]
    d1 = np.linalg.norm(tape.state[:, :3] - pot1, axis=1)
    d2 = np.linalg.norm(tape.state[:, :3] - pot2, axis=1)
    approach = None
    return_query = None
    if start is not None:
        hits = np.flatnonzero(
            (np.arange(len(d1)) >= int(start)) & (d1 < LONG_GRASP_NEAR_M)
        )
        if len(hits):
            approach = int(hits[0])
            transitions = np.flatnonzero(
                (np.arange(len(d1)) > approach)
                & (d2 < d1)
                & ~np.r_[False, (d2 < d1)[:-1]]
            )
            if len(transitions):
                return_query = int(transitions[0])
    regression = bool(
        row2["goal_final"]
        and not row1["goal_final"]
        and approach is not None
        and d2[-1] < d1[-1]
    )
    return {
        "controller_subgoal_regression_proxy": regression,
        "controller_pot1_approach_query": approach,
        "controller_return_to_pot2_query": return_query,
        "eef_terminal_basin": "pot1" if d1[-1] < d2[-1] else "pot2",
        "eef_terminal_pot1_distance_m": float(d1[-1]),
        "eef_terminal_pot2_distance_m": float(d2[-1]),
        "eef_post_pot2_pot1_min_distance_m": (
            float(d1[int(start) :].min()) if start is not None else np.nan
        ),
    }


def serialize_details(value: Any) -> str:
    return json.dumps(plain(value), separators=(",", ":"), sort_keys=True)


def process_task(
    cache_root: Path, task: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _summaries, layout, tapes = load_task(cache_root, task)
    targets = identify_targets(task, layout)
    endpoints = {
        target.name: terminal_goal_distances(tapes, target.lo, target.hi)
        for target in targets
    }
    drawer = drawer_reference(layout, tapes) if task == TOP_DRAWER_TASK else None
    target_names = {target.name for target in targets}
    distractors, distractor_limit = distractor_thresholds(layout, tapes, target_names)
    near = LONG_GRASP_NEAR_M if task == LONG_TASK else OTHER_GRASP_NEAR_M
    hold = LONG_HOLD_DISTANCE_M if task == LONG_TASK else OTHER_HOLD_DISTANCE_M
    goal_receptacle = GOAL_RECEPTACLE_JOINT.get(task)
    rows: list[dict[str, Any]] = []
    progress_curves: dict[int, np.ndarray] = {}
    for tape in tapes:
        target_rows, component_progress = target_analysis(
            tape, targets, endpoints, near, hold
        )
        drawer_row, drawer_progress = drawer_analysis(tape, drawer)
        if drawer_progress is not None:
            component_progress.append(drawer_progress)
        progress = np.mean(np.stack(component_progress), axis=0)
        progress_curves[tape.episode] = progress
        target_final = [bool(row["goal_final"]) for row in target_rows]
        target_ever = [bool(row["goal_ever"]) for row in target_rows]
        subgoal_final = sum(target_final) + int(drawer_row.get("drawer_goal_final", False))
        subgoal_ever = sum(target_ever) + int(drawer_row.get("drawer_goal_ever", False))
        subgoal_total = len(targets) + int(drawer is not None)
        incomplete_retry_bouts = [
            int(row["near_close_bouts"])
            for row in target_rows
            if not bool(row["goal_final"])
        ]
        row: dict[str, Any] = {
            "task": task,
            "episode": tape.episode,
            "init_state_id": tape.init_state_id,
            "flow_noise_seed": tape.flow_noise_seed,
            "episode_length": len(tape.state),
            "success": tape.success,
            **movement_class(tape.state),
            "task_progress_max": float(progress.max()),
            "task_progress_final": float(progress[-1]),
            "task_progress_regression": float(progress.max() - progress[-1]),
            "task_progress_max_query": int(np.argmax(progress)),
            "subgoals_total": subgoal_total,
            "subgoals_ever": subgoal_ever,
            "subgoals_final": subgoal_final,
            "partial_subgoal_final": bool(0 < subgoal_final < subgoal_total),
            "object_goal_loss": any(bool(item["goal_loss"]) for item in target_rows),
            "object_goal_loss_count": sum(bool(item["goal_loss"]) for item in target_rows),
            "target_lifted_count": sum(
                float(item["lift_m"]) > LIFT_DISTANCE_M for item in target_rows
            ),
            "target_hold_evidence_count": sum(
                int(item["hold_bouts"]) > 0 for item in target_rows
            ),
            "target_hold_bout_total": sum(int(item["hold_bouts"]) for item in target_rows),
            "repeated_target_hold_proxy": any(
                int(item["hold_bouts"]) >= 2 and not bool(item["goal_final"])
                for item in target_rows
            ),
            "target_near_close_bout_total": sum(
                int(item["near_close_bouts"]) for item in target_rows
            ),
            "repeated_incomplete_target_near_close_proxy": bool(
                incomplete_retry_bouts and max(incomplete_retry_bouts) >= 2
            ),
            "drop_proxy": any(int(item["drop_proxy_events"]) > 0 for item in target_rows),
            "drop_proxy_count": sum(int(item["drop_proxy_events"]) for item in target_rows),
            "target_details": serialize_details(target_rows),
            **drawer_row,
            **distractor_analysis(
                tape,
                targets,
                distractors,
                distractor_limit,
                near,
                goal_receptacle,
            ),
        }
        for column in (
            "distractor_displacement_details",
            "wrong_object_close_details",
            "goal_side_close_details",
        ):
            row[column] = serialize_details(row[column])
        if task == LONG_TASK:
            row.update(long_controller_analysis(tape, target_rows, targets))
        else:
            row.update(
                {
                    "controller_subgoal_regression_proxy": False,
                    "controller_pot1_approach_query": np.nan,
                    "controller_return_to_pot2_query": np.nan,
                    "eef_terminal_basin": "not_applicable",
                    "eef_terminal_pot1_distance_m": np.nan,
                    "eef_terminal_pot2_distance_m": np.nan,
                    "eef_post_pot2_pot1_min_distance_m": np.nan,
                }
            )
        row["active_retry_proxy"] = bool(
            row["late_motion_class"] == "active"
            and row["repeated_incomplete_target_near_close_proxy"]
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    success_regression = frame.loc[frame["success"], "task_progress_regression"]
    regression_threshold = max(0.10, float(np.quantile(success_regression, 0.99)))
    frame["progress_regression_outlier"] = (
        frame["task_progress_regression"] > regression_threshold
    )
    frame["progress_regression_threshold"] = regression_threshold
    metadata = {
        "task": task,
        "targets": [target.name for target in targets],
        "successes": int(frame["success"].sum()),
        "failures": int((~frame["success"]).sum()),
        "progress_regression_threshold": regression_threshold,
        "distractor_thresholds": distractor_limit,
        "drawer_reference": drawer,
    }
    return frame, metadata


def primary_pattern(row: pd.Series) -> str:
    task = str(row["task"])
    if task == TOP_DRAWER_TASK:
        if bool(row["drop_proxy"]):
            return "bowl_transport_loss_proxy"
        if bool(row.get("drawer_goal_loss", False)):
            return "drawer_subgoal_regression"
        if bool(row["active_retry_proxy"]):
            return "active_bowl_retry_proxy"
        return (
            "bowl_placement_stall"
            if row["late_motion_class"] == "stasis"
            else "ongoing_bowl_placement_failure"
        )
    if task == LONG_TASK:
        if bool(row["object_goal_loss"]):
            return "placed_object_goal_loss"
        if bool(row["controller_subgoal_regression_proxy"]):
            return "return_to_completed_pot2_proxy"
        if bool(row["partial_subgoal_final"]):
            if row["eef_terminal_basin"] == "pot1":
                return (
                    "stalled_at_unfinished_pot1"
                    if row["late_motion_class"] == "stasis"
                    else "active_at_unfinished_pot1"
                )
            return "partial_completion_other_terminal"
        return "other_long_incomplete"
    if task == RAMEKIN_TASK:
        if bool(row["distractor_displacement_outlier"]):
            return "ramekin_displacement_interference_proxy"
        if bool(row["object_goal_loss"]):
            return "bowl_goal_loss"
        if bool(row["drop_proxy"]):
            return "bowl_transport_loss_proxy"
        return "transported_bowl_not_placed"
    if task == STOVE_TASK:
        if not bool(row["target_hold_evidence_count"]):
            if bool(row["goal_side_without_target_close_proxy"]):
                return "empty_goal_side_close_proxy"
            return "bowl_never_transported"
        if bool(row["drop_proxy"]):
            return "bowl_transport_loss_proxy"
        if bool(row["distractor_displacement_outlier"]):
            return "plate_displacement_interference_proxy"
        return "transported_bowl_not_placed"
    raise ValueError(task)


def task_event_counts(failure: pd.DataFrame) -> list[dict[str, Any]]:
    flags = [
        "partial_subgoal_final",
        "object_goal_loss",
        "drawer_goal_loss",
        "progress_regression_outlier",
        "drop_proxy",
        "repeated_incomplete_target_near_close_proxy",
        "repeated_target_hold_proxy",
        "active_retry_proxy",
        "controller_subgoal_regression_proxy",
        "distractor_displacement_outlier",
        "wrong_object_close_proxy",
        "goal_side_without_target_close_proxy",
    ]
    result = []
    for task, group in failure.groupby("task", sort=True):
        row: dict[str, Any] = {"task": task, "failures": int(len(group))}
        for flag in flags:
            row[flag] = int(group.get(flag, pd.Series(False, index=group.index)).sum())
        for label in ("active", "intermittent", "stasis"):
            row[f"motion_{label}"] = int(group["late_motion_class"].eq(label).sum())
        result.append(row)
    return result


def representative_rows(failure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, pattern), group in failure.groupby(
        ["task", "primary_physical_pattern"], sort=True
    ):
        center = np.asarray(
            [
                group["task_progress_regression"].median(),
                group["late_eef_step_mean_m"].median(),
            ]
        )
        scale = np.asarray(
            [
                max(float(group["task_progress_regression"].std()), 1e-6),
                max(float(group["late_eef_step_mean_m"].std()), 1e-6),
            ]
        )
        values = group[
            ["task_progress_regression", "late_eef_step_mean_m"]
        ].to_numpy()
        distance = np.linalg.norm((values - center) / scale, axis=1)
        chosen = group.iloc[int(np.argmin(distance))]
        rows.append(
            {
                "task": task,
                "primary_physical_pattern": pattern,
                "group_n": int(len(group)),
                "episode": int(chosen["episode"]),
                "init_state_id": int(chosen["init_state_id"]),
                "flow_noise_seed": int(chosen["flow_noise_seed"]),
                "late_motion_class": chosen["late_motion_class"],
                "task_progress_final": float(chosen["task_progress_final"]),
                "task_progress_regression": float(chosen["task_progress_regression"]),
                "selection": "closest to group median in progress-regression and late-motion plane",
            }
        )
    return pd.DataFrame(rows)


def render_report(summary: dict[str, Any]) -> str:
    events = summary["task_event_counts"]
    patterns = summary["primary_pattern_counts"]
    event_rows = []
    for row in events:
        event_rows.append(
            "| `%s` | %d | %d/%d/%d | %d | %d | %d | %d | %d | %d |"
            % (
                row["task"].split("/", 1)[-1],
                row["failures"],
                row["motion_active"],
                row["motion_intermittent"],
                row["motion_stasis"],
                row["partial_subgoal_final"],
                row["object_goal_loss"] + row["drawer_goal_loss"],
                row["drop_proxy"],
                row["active_retry_proxy"],
                row["controller_subgoal_regression_proxy"],
                row["distractor_displacement_outlier"],
            )
        )
    pattern_rows = [
        f"| `{row['task'].split('/', 1)[-1]}` | `{row['primary_physical_pattern']}` | {row['n']} |"
        for row in patterns
    ]
    total = summary["totals"]
    return "\n".join(
        [
            "# 全部失败 rollout 的物理事件审计",
            "",
            "## 直接结论",
            "",
            f"本审计覆盖全部 {total['failures']} 条失败。按最后 25% query 的 EEF 运动，{total['motion_active']} 条仍持续运动，{total['motion_intermittent']} 条间歇运动，{total['motion_stasis']} 条进入物理静滞。因此，失败并不等同于停滞。",
            "",
            f"缓存能确认的是物体/EEF/抽屉的运动学状态：{total['partial_subgoal_final']} 条终点仍保留部分子目标，{total['subgoal_loss']} 条曾达到目标代理后又失去，{total['distractor_displacement_outlier']} 条出现超出成功 99% 基线的非目标物位移。另有 {total['drop_proxy']} 条满足保守运输丢失代理，{total['active_retry_proxy']} 条满足“仍活跃且对未完成目标重复靠近闭爪”的 retry 代理。",
            "",
            "不能确认的是接触、碰撞、真实抓取力和视觉误认。下文的 `drop/retry/interference/wrong-object` 都明确保留 `proxy`，不能改写成真值事件。",
            "",
            "## 事件账本",
            "",
            "| task | failures | active/intermittent/stasis | partial final | subgoal loss | drop proxy | active retry | controller regression | distractor displacement |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            *event_rows,
            "",
            "这些列是多标签，可以重叠，不能横向相加。",
            "",
            "## 物理模式摘要",
            "",
            "`primary_physical_pattern` 只是为了浏览账本而设置的任务内优先级摘要；正式结论应使用上面的多标签事件列。",
            "",
            "| task | primary pattern | n |",
            "|---|---|---:|",
            *pattern_rows,
            "",
            "## 四个失败任务在做什么",
            "",
            summary["task_narratives"][TOP_DRAWER_TASK],
            "",
            summary["task_narratives"][LONG_TASK],
            "",
            summary["task_narratives"][RAMEKIN_TASK],
            "",
            summary["task_narratives"][STOVE_TASK],
            "",
            "## 判据与证据等级",
            "",
            "- 可确认运动学：目标是否进入任一成功终点 5 cm 邻域、是否在终点保留、抽屉开度、EEF 路径、非目标物位移。",
            "- 强代理：闭合夹爪附近的物体与 EEF 同动、相对运动残差受限；随后下降分离记为 drop proxy。",
            "- 弱代理：重复靠近并闭爪记为 retry proxy；靠近 goal receptacle 但目标仍远时闭爪记为 empty-goal-side proxy。",
            "- 无法判断：真实 contact/collision、夹爪受力、遮挡、视觉误识别和动作 chunk 内事件时刻。缓存没有 RGB、视频、contact 或 force。",
            "",
            "持续运动定义为最后 25% query transition 中，EEF 平均位移至少 1 cm/query，且至少一半 transition 超过 1 cm。静滞定义为平均不足 5 mm/query 且超过 1 cm 的比例不高于 25%；其余为 intermittent。",
            "",
            "## 产物",
            "",
            "- `episode_events.csv`: 307 条失败的完整多轴事件账本。",
            "- `representative_episodes.csv`: 每个任务内摘要模式的中心代表 episode。",
            "- `summary.json`: 阈值、任务计数和证据限制。",
        ]
    ) + "\n"


def build_summary(
    all_frame: pd.DataFrame, task_metadata: list[dict[str, Any]]
) -> dict[str, Any]:
    failure = all_frame[~all_frame["success"]].copy()
    failure["primary_physical_pattern"] = failure.apply(primary_pattern, axis=1)
    counts = task_event_counts(failure)
    pattern_counts = (
        failure.groupby(["task", "primary_physical_pattern"], sort=True)
        .size()
        .rename("n")
        .reset_index()
        .to_dict("records")
    )
    count_by_task = {row["task"]: row for row in counts}
    top = count_by_task[TOP_DRAWER_TASK]
    long = count_by_task[LONG_TASK]
    ramekin = count_by_task[RAMEKIN_TASK]
    stove = count_by_task[STOVE_TASK]
    narratives = {
        TOP_DRAWER_TASK: (
            f"**Top drawer + bowl（{top['failures']}）**：所有失败都曾打开抽屉并抬起碗。"
            f"其中 {top['drawer_goal_loss']} 条终点失去 drawer-open 子目标，"
            f"{top['drop_proxy']} 条有保守运输丢失代理；终端仍 active 的有 {top['motion_active']} 条。"
        ),
        LONG_TASK: (
            f"**Two moka pots（{long['failures']}）**：主体是完成 pot 2 后未完成 pot 1。"
            f"{long['partial_subgoal_final']} 条终点保留部分完成，"
            f"{long['controller_subgoal_regression_proxy']} 条在接近 pot 1 后回到 pot 2 侧，"
            f"{long['object_goal_loss']} 条把物体带到 goal 邻域后又失去；"
            f"只有 {long['active_retry_proxy']} 条满足 active retry 代理。"
        ),
        RAMEKIN_TASK: (
            f"**Bowl on ramekin（{ramekin['failures']}）**：目标碗均出现运输证据，"
            f"但 {ramekin['distractor_displacement_outlier']} 条同时让非目标 ramekin 产生远超成功基线的位移。"
            "这确认了非目标物被明显扰动，但没有 contact 字段，原因只能称 interference proxy。"
        ),
        STOVE_TASK: (
            f"**Bowl on stove（{stove['failures']}）**："
            f"{stove['failures'] - int(failure[failure['task'].eq(STOVE_TASK)]['target_hold_evidence_count'].gt(0).sum())} 条没有保守运输证据；"
            f"{stove['goal_side_without_target_close_proxy']} 条在目标碗仍远时于 plate 一侧闭爪，"
            f"{stove['drop_proxy']} 条有运输丢失代理。"
        ),
    }
    totals = {
        "failures": int(len(failure)),
        "motion_active": int(failure["late_motion_class"].eq("active").sum()),
        "motion_intermittent": int(
            failure["late_motion_class"].eq("intermittent").sum()
        ),
        "motion_stasis": int(failure["late_motion_class"].eq("stasis").sum()),
        "partial_subgoal_final": int(failure["partial_subgoal_final"].sum()),
        "subgoal_loss": int(
            (failure["object_goal_loss"] | failure["drawer_goal_loss"].fillna(False)).sum()
        ),
        "drop_proxy": int(failure["drop_proxy"].sum()),
        "active_retry_proxy": int(failure["active_retry_proxy"].sum()),
        "distractor_displacement_outlier": int(
            failure["distractor_displacement_outlier"].sum()
        ),
    }
    return {
        "schema": "failure-event-audit/1",
        "run_class": "exploratory simulator-kinematic audit",
        "totals": totals,
        "task_event_counts": counts,
        "primary_pattern_counts": pattern_counts,
        "task_metadata": task_metadata,
        "task_narratives": narratives,
        "definitions": {
            "object_goal": "XYZ within 0.05 m of any successful terminal XYZ for that target",
            "task_progress": "mean of target-specific 1-distance/initial-distance, plus normalized drawer opening for top-drawer task",
            "progress_regression_outlier": "max-minus-final progress above max(0.10, task success q99)",
            "late_active": "last-quarter EEF mean >=0.01 m/query and moving fraction >=0.50",
            "late_stasis": "last-quarter EEF mean <0.005 m/query and moving fraction <=0.25",
            "drop_proxy": "existing conservative closed-gripper co-motion then lower-and-separated rule",
            "retry_proxy": "at least two target-near close-command bouts for an unfinished target",
            "distractor_outlier": "max XYZ displacement above max(0.03 m, task/joint success q99)",
        },
        "evidence_limits": [
            "No RGB frames or video are stored in the client NPZ files.",
            "No contact, force, or dense within-chunk simulator state is stored.",
            "Goal neighborhoods are successful-terminal proxies, not reconstructed environment predicates.",
            "Near-close, drop, interference, and wrong-side labels are proxies rather than semantic ground truth.",
        ],
    }


def self_test() -> None:
    count, onset = count_bouts(np.asarray([False, True, True, False, True]))
    assert count == 2 and onset == [1, 4]
    state = np.zeros((8, 8), dtype=np.float64)
    state[:, 0] = np.arange(8) * 0.02
    assert movement_class(state)["late_motion_class"] == "active"
    state[:, 0] = 0.0
    assert movement_class(state)["late_motion_class"] == "stasis"
    distance = distance_to_success_endpoints(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.asarray([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    np.testing.assert_allclose(distance, [0.2, 0.8])
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    frames = []
    metadata = []
    for task in FAILURE_TASKS:
        frame, task_metadata = process_task(args.cache_root, task)
        frames.append(frame)
        metadata.append(task_metadata)
    all_frame = pd.concat(frames, ignore_index=True)
    summary = build_summary(all_frame, metadata)
    failure = all_frame[~all_frame["success"]].copy()
    failure["primary_physical_pattern"] = failure.apply(primary_pattern, axis=1)
    failure_mode = pd.read_csv(FAILURE_MODE_CSV)[
        ["task", "episode", "failure_mode"]
    ]
    failure = failure.merge(
        failure_mode, on=["task", "episode"], how="left", validate="one_to_one"
    )
    if len(failure) != 307 or failure["failure_mode"].isna().any():
        raise ValueError("formal 307-failure cohort drifted")
    representatives = representative_rows(failure)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    failure.to_csv(args.out_dir / "episode_events.csv", index=False)
    representatives.to_csv(args.out_dir / "representative_episodes.csv", index=False)
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    outputs = [
        "episode_events.csv",
        "representative_episodes.csv",
        "summary.json",
        "report.md",
    ]
    completion = {
        "schema": "failure-event-audit-completion/1",
        "status": "complete",
        "script_sha256": sha256_file(Path(__file__)),
        "output_sha256": {
            name: sha256_file(args.out_dir / name) for name in outputs
        },
    }
    (args.out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {args.out_dir} ({len(failure)} failures)")


if __name__ == "__main__":
    main()
