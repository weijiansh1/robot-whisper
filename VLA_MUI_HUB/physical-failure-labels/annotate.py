#!/usr/bin/env python3
"""Attach auditable physical failure-mode labels to LIBERO rollout episodes.

The source rollout directories are read-only inputs.  Successful outcomes retain
the online LIBERO ``check_success`` result recorded during collection.  Failed
episodes are inspected by restoring every recorded MuJoCo control state and
evaluating the original BDDL predicates, contacts, grasps, object motion, and
articulated joints in the matching simulator.

The primary label is an observed or inferred failure mode, not a claim about the
policy's internal causal mechanism.  The output keeps the raw physical evidence
and confidence next to every inferred label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA = "himoe.libero.physical_failure_labels.v1"
MOVE_THRESHOLD_M = 0.02
LIFT_THRESHOLD_M = 0.025
DROP_THRESHOLD_M = 0.04
REACH_THRESHOLD_M = 0.12
JOINT_MOVE_THRESHOLD = 0.01
DISTRACTOR_MOVE_THRESHOLD_M = 0.05
REPLAY_STATE_TOLERANCE = 5e-3
DEFAULT_REPLAN_STEPS = 10
DEFAULT_SETTLE_STEPS = 10

SUITE_TO_BENCHMARK = {
    "libero_goal": "libero_goal",
    "libero_object": "libero_object",
    "libero_spatial": "libero_spatial",
    "libero_long": "libero_10",
}

REASON_PRIORITY = {
    "goal_predicate_regressed": 0,
    "object_released_or_dropped_before_goal": 1,
    "object_released_outside_goal": 2,
    "timeout_while_holding_target": 3,
    "mechanism_goal_regressed": 4,
    "mechanism_threshold_not_reached": 5,
    "mechanism_not_actuated": 6,
    "stable_grasp_not_observed": 7,
    "object_moved_but_goal_unmet": 8,
    "approached_target_without_observed_contact": 9,
    "no_meaningful_target_progress": 10,
    "goal_predicate_unmet": 11,
}


@dataclass
class RunInput:
    data_root: str
    run_dir: Path
    client_dir: Path
    relative_run: str
    suite_dir: str
    benchmark: str
    task_id: int
    task_name: str
    seed: int
    summaries: List[Dict[str, Any]]
    meta: Dict[str, Any]
    summaries_sha256: str


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=here.parent)
    parser.add_argument(
        "--libero-root",
        type=Path,
        required=True,
        help="Root of the pinned upstream LIBERO checkout",
    )
    parser.add_argument("--out", type=Path, default=here / "results")
    parser.add_argument(
        "--data-root",
        action="append",
        dest="data_roots",
        help="Hub child to scan; repeatable (default: cache and cache_new)",
    )
    parser.add_argument(
        "--task-contains",
        help="Only process task names containing this substring (smoke/debug)",
    )
    parser.add_argument(
        "--limit-failures",
        type=int,
        default=0,
        help="Cap failed episodes per task group; 0 means all",
    )
    parser.add_argument(
        "--dense-audit-per-task",
        type=int,
        default=1,
        help="Fully replay this many failed episodes per task; 0 disables",
    )
    parser.add_argument(
        "--strict-dense-audit",
        action="store_true",
        help="Exit nonzero when an action-only dense replay diverges",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress after this many physically inspected failures",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def json_text(value: Any, *, pretty: bool = False) -> str:
    return json.dumps(
        value,
        indent=2 if pretty else None,
        sort_keys=pretty,
        ensure_ascii=False,
        allow_nan=False,
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write_text(path, "".join(json_text(row) + "\n" for row in rows))


def round_float(value: float, digits: int = 8) -> float:
    return round(float(value), digits)


def first_index(values: Sequence[bool], expected: bool = True) -> Optional[int]:
    for index, value in enumerate(values):
        if bool(value) is expected:
            return index
    return None


def last_index(values: Sequence[bool], expected: bool = True) -> Optional[int]:
    for index in range(len(values) - 1, -1, -1):
        if bool(values[index]) is expected:
            return index
    return None


def first_false_after_true(values: Sequence[bool]) -> Optional[int]:
    seen_true = False
    for index, value in enumerate(values):
        if value:
            seen_true = True
        elif seen_true:
            return index
    return None


def predicate_id(index: int, goal: Sequence[str]) -> str:
    arguments = ",".join(str(value).lower() for value in goal[1:])
    return "%02d:%s(%s)" % (index, str(goal[0]).lower(), arguments)


def episode_filename(index: int) -> str:
    return "episode_%02d.npz" % index


def discover_runs(
    hub: Path, data_roots: Sequence[str], task_contains: Optional[str]
) -> Tuple[List[RunInput], List[Dict[str, str]]]:
    runs: List[RunInput] = []
    skipped: List[Dict[str, str]] = []
    for data_root in data_roots:
        root = hub / data_root / "HiMoE-VLA"
        if not root.is_dir():
            skipped.append({"path": str(root), "reason": "data_root_missing"})
            continue
        pattern = "libero_*/*/*/client/summaries.json"
        for summaries_path in sorted(root.glob(pattern)):
            client_dir = summaries_path.parent
            run_dir = client_dir.parent
            task_name = run_dir.parent.name
            suite_dir = run_dir.parent.parent.name
            if suite_dir not in SUITE_TO_BENCHMARK:
                skipped.append({"path": str(run_dir), "reason": "unsupported_suite"})
                continue
            if task_contains and task_contains not in task_name:
                continue
            meta_path = run_dir / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            if meta.get("status") not in (None, "complete"):
                skipped.append({"path": str(run_dir), "reason": "run_not_complete"})
                continue
            sampling = meta.get("sampling")
            if isinstance(sampling, Mapping) and sampling.get("complete") is False:
                skipped.append({"path": str(run_dir), "reason": "sampling_not_complete"})
                continue
            loaded = json.loads(summaries_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                loaded = loaded.get("episodes", loaded.get("summaries", []))
            if not isinstance(loaded, list) or not loaded:
                skipped.append({"path": str(run_dir), "reason": "empty_summaries"})
                continue
            task_ids = {int(row["task_id"]) for row in loaded}
            task_names = {str(row["task_name"]) for row in loaded}
            seeds = {int(row["seed"]) for row in loaded}
            if len(task_ids) != 1 or len(task_names) != 1 or len(seeds) != 1:
                raise RuntimeError("Run mixes task identity or seeds: %s" % run_dir)
            actual_task_name = next(iter(task_names))
            if actual_task_name != task_name:
                raise RuntimeError(
                    "Task directory and summary disagree: %s != %s"
                    % (task_name, actual_task_name)
                )
            indices = [int(row["episode_index"]) for row in loaded]
            if len(indices) != len(set(indices)):
                raise RuntimeError("Duplicate episode_index in %s" % summaries_path)
            missing = [
                index
                for index in indices
                if not (client_dir / episode_filename(index)).is_file()
            ]
            if missing:
                raise RuntimeError(
                    "%s is missing %d episode files (first=%d)"
                    % (run_dir, len(missing), missing[0])
                )
            runs.append(
                RunInput(
                    data_root=data_root,
                    run_dir=run_dir,
                    client_dir=client_dir,
                    relative_run=str(run_dir.relative_to(hub)),
                    suite_dir=suite_dir,
                    benchmark=SUITE_TO_BENCHMARK[suite_dir],
                    task_id=next(iter(task_ids)),
                    task_name=actual_task_name,
                    seed=next(iter(seeds)),
                    summaries=[dict(row) for row in loaded],
                    meta=dict(meta),
                    summaries_sha256=sha256_file(summaries_path),
                )
            )
    return runs, skipped


def task_group_key(run: RunInput) -> Tuple[str, int, str, int]:
    return run.benchmark, run.task_id, run.task_name, run.seed


def qpos_joint_names(core: Any) -> List[str]:
    names = []
    for name in core.sim.model.joint_names:
        text = str(name)
        if text.startswith(("robot0_", "gripper0_")):
            continue
        address = core.sim.model.get_joint_qpos_addr(name)
        if not isinstance(address, tuple):
            names.append(text)
    return sorted(names)


def mechanism_joint_names(core: Any, subject: str) -> List[str]:
    state = core.object_states_dict.get(subject)
    if state is None:
        return []
    if getattr(state, "object_state_type", None) == "site":
        model = core.object_sites_dict.get(subject)
    else:
        model = core.get_object(subject)
    return sorted(str(name) for name in getattr(model, "joints", []))


class PhysicsTask:
    def __init__(
        self,
        benchmark_name: str,
        task_id: int,
        expected_task_name: str,
        seed: int,
        horizon: int,
    ) -> None:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.env_wrapper import ControlEnv

        suite = benchmark.get_benchmark_dict()[benchmark_name]()
        task = suite.get_task(task_id)
        if str(task.name) != expected_task_name:
            raise RuntimeError(
                "LIBERO task mismatch: expected %s, loaded %s"
                % (expected_task_name, task.name)
            )
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        self.environment = ControlEnv(
            bddl_file_name=str(bddl_file),
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            horizon=horizon + DEFAULT_SETTLE_STEPS + 1,
        )
        self.environment.seed(seed)
        self.environment.reset()
        self.core = self.environment.env
        self.suite = suite
        self.initial_states = suite.get_task_init_states(task_id)
        self.task = task
        self.goals = [list(state) for state in self.core.parsed_problem["goal_state"]]
        self.goal_ids = [predicate_id(index, goal) for index, goal in enumerate(self.goals)]
        self.robot = self.core.robots[0]
        self.mobile_subjects = sorted(
            {
                str(goal[1])
                for goal in self.goals
                if len(goal) == 3 and str(goal[1]) in self.core.objects_dict
            }
        )
        self.movable_names = sorted(
            name
            for name in self.core.objects_dict
            if name in self.core.obj_body_id
        )
        self.articulation_joints = qpos_joint_names(self.core)
        self.goal_mechanism_joints = {
            self.goal_ids[index]: mechanism_joint_names(self.core, str(goal[1]))
            for index, goal in enumerate(self.goals)
            if len(goal) == 2
        }

    def close(self) -> None:
        self.environment.close()


def summarize_predicate(
    goal_id: str, expression: Sequence[str], values: Sequence[bool]
) -> Dict[str, Any]:
    first_true = first_index(values)
    regression = first_false_after_true(values)
    return {
        "id": goal_id,
        "expression": [str(value).lower() for value in expression],
        "first_checkpoint": bool(values[0]),
        "last_checkpoint": bool(values[-1]),
        "ever_satisfied": first_true is not None,
        "first_satisfied_snapshot": first_true,
        "last_satisfied_snapshot": last_index(values),
        "regressed_after_satisfaction": regression is not None,
        "first_regression_snapshot": regression,
    }


def summarize_mobile_object(
    positions: Sequence[np.ndarray],
    distances: Sequence[float],
    contacts: Sequence[bool],
    grasps: Sequence[bool],
) -> Dict[str, Any]:
    xyz = np.asarray(positions, dtype=np.float64)
    displacement = np.linalg.norm(xyz - xyz[0], axis=1)
    rise = xyz[:, 2] - xyz[0, 2]
    first_grasp = first_index(grasps)
    release = first_false_after_true(grasps)
    first_motion_candidates = np.flatnonzero(displacement >= MOVE_THRESHOLD_M)
    peak_index = int(np.argmax(xyz[:, 2]))
    return {
        "initial_xyz_m": [round_float(value) for value in xyz[0]],
        "last_checkpoint_xyz_m": [round_float(value) for value in xyz[-1]],
        "max_displacement_m": round_float(displacement.max()),
        "last_displacement_m": round_float(displacement[-1]),
        "max_vertical_rise_m": round_float(rise.max()),
        "min_vertical_change_m": round_float(rise.min()),
        "peak_height_snapshot": peak_index,
        "fall_from_peak_to_last_m": round_float(xyz[peak_index, 2] - xyz[-1, 2]),
        "min_gripper_distance_m": round_float(min(distances)),
        "last_gripper_distance_m": round_float(distances[-1]),
        "ever_gripper_contact": any(contacts),
        "first_contact_snapshot": first_index(contacts),
        "last_contact_snapshot": last_index(contacts),
        "contact_at_last_checkpoint": bool(contacts[-1]),
        "ever_stable_grasp": first_grasp is not None,
        "first_stable_grasp_snapshot": first_grasp,
        "last_stable_grasp_snapshot": last_index(grasps),
        "stable_grasp_at_last_checkpoint": bool(grasps[-1]),
        "released_after_observed_grasp": release is not None,
        "first_release_snapshot": release,
        "first_motion_snapshot": (
            int(first_motion_candidates[0]) if len(first_motion_candidates) else None
        ),
    }


def classify_goal_failure(
    predicate: Mapping[str, Any],
    object_evidence: Optional[Mapping[str, Any]],
    mechanism_excursion: float,
) -> Dict[str, Any]:
    expression = predicate["expression"]
    if predicate["last_checkpoint"]:
        return {
            "reason": "goal_satisfied_at_last_checkpoint",
            "confidence": "high",
            "basis": ["original_bddl_predicate_true"],
        }
    if predicate["regressed_after_satisfaction"]:
        reason = "mechanism_goal_regressed" if len(expression) == 2 else "goal_predicate_regressed"
        return {
            "reason": reason,
            "confidence": "high",
            "basis": ["original_bddl_predicate_true_then_false"],
        }
    if len(expression) == 2:
        if mechanism_excursion < JOINT_MOVE_THRESHOLD:
            return {
                "reason": "mechanism_not_actuated",
                "confidence": "medium",
                "basis": ["unary_goal_never_true", "associated_joint_motion_below_threshold"],
            }
        return {
            "reason": "mechanism_threshold_not_reached",
            "confidence": "medium",
            "basis": ["unary_goal_never_true", "associated_joint_moved"],
        }
    if object_evidence is None:
        return {
            "reason": "goal_predicate_unmet",
            "confidence": "low",
            "basis": ["original_bddl_predicate_false"],
        }

    moved = float(object_evidence["max_displacement_m"]) >= MOVE_THRESHOLD_M
    lifted = float(object_evidence["max_vertical_rise_m"]) >= LIFT_THRESHOLD_M
    if object_evidence["ever_stable_grasp"]:
        if (
            object_evidence["released_after_observed_grasp"]
            and float(object_evidence["fall_from_peak_to_last_m"]) >= DROP_THRESHOLD_M
        ):
            return {
                "reason": "object_released_or_dropped_before_goal",
                "confidence": "medium",
                "basis": ["observed_stable_grasp", "later_release", "height_loss"],
            }
        if object_evidence["released_after_observed_grasp"]:
            return {
                "reason": "object_released_outside_goal",
                "confidence": "medium",
                "basis": ["observed_stable_grasp", "later_release", "goal_never_true"],
            }
        if object_evidence["stable_grasp_at_last_checkpoint"]:
            return {
                "reason": "timeout_while_holding_target",
                "confidence": "medium",
                "basis": ["stable_grasp_at_last_checkpoint", "goal_never_true"],
            }
        return {
            "reason": "object_moved_but_goal_unmet",
            "confidence": "medium",
            "basis": ["observed_stable_grasp", "goal_never_true"],
        }
    if (
        object_evidence["ever_gripper_contact"]
        and not object_evidence["contact_at_last_checkpoint"]
        and lifted
        and float(object_evidence["fall_from_peak_to_last_m"]) >= DROP_THRESHOLD_M
    ):
        return {
            "reason": "object_released_or_dropped_before_goal",
            "confidence": "medium",
            "basis": [
                "gripper_contact_observed",
                "target_lift_observed",
                "later_contact_loss",
                "height_loss",
            ],
        }
    if moved or lifted:
        return {
            "reason": "object_moved_but_goal_unmet",
            "confidence": "medium",
            "basis": ["target_motion_observed", "goal_never_true"],
        }
    if object_evidence["ever_gripper_contact"]:
        return {
            "reason": "stable_grasp_not_observed",
            "confidence": "medium",
            "basis": ["gripper_contact_observed", "no_two_finger_grasp_at_checkpoints"],
        }
    if float(object_evidence["min_gripper_distance_m"]) <= REACH_THRESHOLD_M:
        return {
            "reason": "approached_target_without_observed_contact",
            "confidence": "medium",
            "basis": ["end_effector_near_target", "no_contact_at_checkpoints"],
        }
    return {
        "reason": "no_meaningful_target_progress",
        "confidence": "medium",
        "basis": ["target_not_reached", "target_motion_below_threshold"],
    }


def choose_primary(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidates:
        return {
            "reason": "goal_predicate_unmet",
            "confidence": "low",
            "basis": ["recorded_failure_without_unsatisfied_checkpoint_goal"],
        }
    return min(candidates, key=lambda row: REASON_PRIORITY.get(str(row["reason"]), 999))


def inspect_failure(task: PhysicsTask, run: RunInput, row: Mapping[str, Any]) -> Dict[str, Any]:
    episode_index = int(row["episode_index"])
    episode_path = run.client_dir / episode_filename(episode_index)
    with np.load(episode_path, allow_pickle=False) as archive:
        if "sim_state" not in archive or "actions" not in archive:
            raise RuntimeError("Episode lacks sim_state/actions: %s" % episode_path)
        sim_states = np.asarray(archive["sim_state"], dtype=np.float64)
        action_chunks = np.asarray(archive["actions"], dtype=np.float32)
    if sim_states.ndim != 2 or len(sim_states) != int(row["inference_calls"]):
        raise RuntimeError("sim_state shape disagrees with summary: %s" % episode_path)
    expected_state_dim = 1 + int(task.core.sim.model.nq) + int(task.core.sim.model.nv)
    if sim_states.shape[1] != expected_state_dim:
        raise RuntimeError(
            "sim_state width mismatch for %s: %d != %d"
            % (episode_path, sim_states.shape[1], expected_state_dim)
        )

    goal_values: List[List[bool]] = [[] for _ in task.goals]
    subject_data = {
        name: {"positions": [], "distances": [], "contacts": [], "grasps": []}
        for name in task.mobile_subjects
    }
    all_object_positions: Dict[str, List[np.ndarray]] = {
        name: [] for name in task.movable_names
    }
    joint_values: Dict[str, List[float]] = {
        name: [] for name in task.articulation_joints
    }
    destination_distances: Dict[str, List[float]] = {
        goal_id: []
        for goal_id, goal in zip(task.goal_ids, task.goals)
        if len(goal) == 3
    }
    checkpoint_success_mismatches = 0
    predicate_api_mismatches = 0

    for sim_state in sim_states:
        task.core.sim.set_state_from_flattened(sim_state)
        task.core.sim.forward()
        values = [bool(task.core._eval_predicate(goal)) for goal in task.goals]
        online_api_value = bool(task.environment.check_success())
        if online_api_value != all(values):
            predicate_api_mismatches += 1
        if online_api_value:
            checkpoint_success_mismatches += 1
        for index, value in enumerate(values):
            goal_values[index].append(value)

        for name, raw in subject_data.items():
            state = task.core.object_states_dict[name]
            model = task.core.get_object(name)
            raw["positions"].append(
                np.asarray(state.get_geom_state()["pos"], dtype=np.float64).copy()
            )
            raw["distances"].append(
                float(
                    task.core._gripper_to_target(
                        task.robot.gripper, model, return_distance=True
                    )
                )
            )
            raw["contacts"].append(bool(task.core.check_contact(task.robot.gripper, model)))
            raw["grasps"].append(bool(task.core._check_grasp(task.robot.gripper, model)))

        for name in task.movable_names:
            position = task.core.object_states_dict[name].get_geom_state()["pos"]
            all_object_positions[name].append(np.asarray(position, dtype=np.float64).copy())
        for name in task.articulation_joints:
            value = task.core.sim.data.get_joint_qpos(name)
            joint_values[name].append(float(np.asarray(value).reshape(-1)[0]))

        for goal_id, goal in zip(task.goal_ids, task.goals):
            if len(goal) != 3:
                continue
            source_state = task.core.object_states_dict.get(str(goal[1]))
            destination_state = task.core.object_states_dict.get(str(goal[2]))
            if source_state is None or destination_state is None:
                destination_distances[goal_id].append(float("inf"))
                continue
            source = np.asarray(source_state.get_geom_state()["pos"], dtype=np.float64)
            destination = np.asarray(
                destination_state.get_geom_state()["pos"], dtype=np.float64
            )
            destination_distances[goal_id].append(float(np.linalg.norm(source - destination)))

    if checkpoint_success_mismatches:
        raise RuntimeError(
            "%s has %d restored checkpoints that satisfy the full goal"
            % (episode_path, checkpoint_success_mismatches)
        )
    if predicate_api_mismatches:
        raise RuntimeError("Predicate API mismatch while restoring %s" % episode_path)

    predicates = [
        summarize_predicate(goal_id, goal, values)
        for goal_id, goal, values in zip(task.goal_ids, task.goals, goal_values)
    ]
    objects = {
        name: summarize_mobile_object(**raw)
        for name, raw in subject_data.items()
    }
    for predicate in predicates:
        distances = destination_distances.get(str(predicate["id"]), [])
        finite = [value for value in distances if np.isfinite(value)]
        if finite:
            predicate["subject_destination_distance_m"] = {
                "initial": round_float(finite[0]),
                "minimum": round_float(min(finite)),
                "last_checkpoint": round_float(finite[-1]),
            }

    articulation = []
    joint_excursion = {}
    for name, values in joint_values.items():
        array = np.asarray(values, dtype=np.float64)
        excursion = float(array.max() - array.min())
        joint_excursion[name] = excursion
        if excursion > 1e-6:
            articulation.append(
                {
                    "joint": name,
                    "initial": round_float(array[0]),
                    "last_checkpoint": round_float(array[-1]),
                    "range": round_float(excursion),
                    "final_delta": round_float(array[-1] - array[0]),
                }
            )
    articulation.sort(key=lambda item: (-abs(float(item["range"])), item["joint"]))

    goal_arguments = {str(value) for goal in task.goals for value in goal[1:]}
    distractors = []
    for name, positions in all_object_positions.items():
        if name in goal_arguments:
            continue
        xyz = np.asarray(positions, dtype=np.float64)
        movement = float(np.linalg.norm(xyz - xyz[0], axis=1).max())
        if movement > 1e-6:
            distractors.append({"object": name, "max_displacement_m": round_float(movement)})
    distractors.sort(key=lambda item: (-float(item["max_displacement_m"]), item["object"]))

    candidates = []
    goal_labels = []
    for goal_id, goal, predicate in zip(task.goal_ids, task.goals, predicates):
        subject = str(goal[1])
        mechanism_range = max(
            (joint_excursion.get(name, 0.0) for name in task.goal_mechanism_joints.get(goal_id, [])),
            default=0.0,
        )
        label = classify_goal_failure(predicate, objects.get(subject), mechanism_range)
        labeled = {"goal_id": goal_id, **label}
        goal_labels.append(labeled)
        if not predicate["last_checkpoint"]:
            candidates.append(labeled)

    primary = dict(choose_primary(candidates))
    contributing = sorted(
        {str(label["reason"]) for label in candidates if label["reason"] != primary["reason"]}
    )
    max_subject_motion = max(
        (float(item["max_displacement_m"]) for item in objects.values()), default=0.0
    )
    max_distractor_motion = max(
        (float(item["max_displacement_m"]) for item in distractors), default=0.0
    )
    if (
        max_distractor_motion >= DISTRACTOR_MOVE_THRESHOLD_M
        and max_distractor_motion > max_subject_motion + MOVE_THRESHOLD_M
    ):
        contributing.append("non_goal_object_moved_more_than_goal_subject")
    contributing = sorted(set(contributing))

    true_at_last = sum(bool(predicate["last_checkpoint"]) for predicate in predicates)
    unobserved_tail = int(row["action_steps"]) - DEFAULT_REPLAN_STEPS * (len(sim_states) - 1)
    if unobserved_tail < 0 or unobserved_tail > DEFAULT_REPLAN_STEPS:
        raise RuntimeError("Unexpected unobserved action tail for %s: %d" % (episode_path, unobserved_tail))

    return {
        "schema": SCHEMA,
        "source": {
            "run": run.relative_run,
            "summaries_sha256": run.summaries_sha256,
            "episode_file": str(episode_path.relative_to(run.run_dir.parents[3])),
            "episode_npz_sha256": sha256_file(episode_path),
        },
        "data_root": run.data_root,
        "suite": run.suite_dir,
        "benchmark": run.benchmark,
        "task_id": int(row["task_id"]),
        "task_name": str(row["task_name"]),
        "prompt": str(row["prompt"]),
        "run_id": run.run_dir.name,
        "episode_index": episode_index,
        "init_state_id": int(row["init_state_id"]),
        "flow_noise_seed": int(row["flow_noise_seed"]),
        "recorded_success": False,
        "action_steps": int(row["action_steps"]),
        "inference_calls": int(row["inference_calls"]),
        "primary_failure_reason": str(primary["reason"]),
        "failure_reason_confidence": str(primary["confidence"]),
        "failure_reason_basis": list(primary["basis"]),
        "reason_scope": "observed_failure_mode_not_proven_policy_cause",
        "contributing_failure_modes": contributing,
        "completion_state_at_last_checkpoint": (
            "partial" if 0 < true_at_last < len(predicates) else "none"
        ),
        "goal_failure_labels": goal_labels,
        "goal_predicates": predicates,
        "goal_subject_physics": objects,
        "articulated_joint_motion": articulation,
        "non_goal_object_motion": distractors,
        "physics_validation": {
            "status": "passed",
            "mode": "recorded_control_state_restore",
            "simulator": "pinned_LIBERO_robosuite_MuJoCo",
            "snapshots_restored": int(len(sim_states)),
            "snapshot_stride_actions": DEFAULT_REPLAN_STEPS,
            "saved_state_dtype": "float32",
            "full_goal_true_at_any_restored_checkpoint": False,
            "predicate_api_mismatches": 0,
            "unobserved_terminal_tail_actions": unobserved_tail,
            "terminal_outcome_source": "online_check_success_recorded_during_rollout",
        },
        "thresholds": {
            "motion_m": MOVE_THRESHOLD_M,
            "lift_m": LIFT_THRESHOLD_M,
            "drop_m": DROP_THRESHOLD_M,
            "reach_m": REACH_THRESHOLD_M,
            "joint_motion": JOINT_MOVE_THRESHOLD,
        },
    }


def success_record(run: RunInput, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "source": {
            "run": run.relative_run,
            "summaries_sha256": run.summaries_sha256,
            "episode_file": str(
                (run.client_dir / episode_filename(int(row["episode_index"]))).relative_to(
                    run.run_dir.parents[3]
                )
            ),
        },
        "data_root": run.data_root,
        "suite": run.suite_dir,
        "benchmark": run.benchmark,
        "task_id": int(row["task_id"]),
        "task_name": str(row["task_name"]),
        "prompt": str(row["prompt"]),
        "run_id": run.run_dir.name,
        "episode_index": int(row["episode_index"]),
        "init_state_id": int(row["init_state_id"]),
        "flow_noise_seed": int(row["flow_noise_seed"]),
        "recorded_success": True,
        "action_steps": int(row["action_steps"]),
        "inference_calls": int(row["inference_calls"]),
        "primary_failure_reason": None,
        "failure_reason_confidence": None,
        "physics_validation": {
            "status": "passed_online",
            "mode": "online_LIBERO_check_success",
            "terminal_outcome_source": "online_check_success_recorded_during_rollout",
        },
    }


def sim_state_component_errors(
    current: np.ndarray, saved: np.ndarray, nq: int, nv: int
) -> Dict[str, float]:
    current = np.asarray(current, dtype=np.float64)
    saved = np.asarray(saved, dtype=np.float64)
    if current.shape != saved.shape:
        raise RuntimeError(
            "Replay state shape mismatch: %s != %s" % (current.shape, saved.shape)
        )
    required = 1 + nq + nv
    if current.ndim != 1 or len(current) < required:
        raise RuntimeError(
            "Replay state length %d is smaller than time + nq + nv (%d)"
            % (len(current), required)
        )
    absolute = np.abs(current - saved)
    auxiliary = absolute[required:]
    return {
        "state": float(absolute.max()),
        "time": float(absolute[0]),
        "qpos": float(absolute[1 : 1 + nq].max()),
        "qvel": float(absolute[1 + nq : required].max()),
        "auxiliary": float(auxiliary.max()) if len(auxiliary) else 0.0,
    }


def policy_state_from_observation(observation: Mapping[str, Any]) -> np.ndarray:
    from himoe_libero_bridge.preprocess import quat_to_axis_angle

    return np.concatenate(
        (
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        )
    )


def dense_replay_audit(task: PhysicsTask, run: RunInput, row: Mapping[str, Any]) -> Dict[str, Any]:
    episode_index = int(row["episode_index"])
    episode_path = run.client_dir / episode_filename(episode_index)
    with np.load(episode_path, allow_pickle=False) as archive:
        actions = np.asarray(archive["actions"], dtype=np.float32)
        saved_states = np.asarray(archive["sim_state"], dtype=np.float64)
        saved_policy_states = np.asarray(archive["state"], dtype=np.float64)
    if len(saved_policy_states) != len(actions) or len(saved_states) != len(actions):
        raise RuntimeError(
            "Dense replay arrays do not align for episode %d" % episode_index
        )

    init_state_id = int(row["init_state_id"])
    if not 0 <= init_state_id < len(task.initial_states):
        raise RuntimeError("Invalid init_state_id for dense replay: %d" % init_state_id)
    task.environment.seed(int(row["seed"]))
    task.environment.reset()
    observation = task.environment.set_init_state(task.initial_states[init_state_id])
    from himoe_libero_bridge.libero_runtime import LIBERO_DUMMY_ACTION

    for _ in range(DEFAULT_SETTLE_STEPS):
        observation, _, _, _ = task.environment.step(LIBERO_DUMMY_ACTION.tolist())

    nq = int(task.core.sim.model.nq)
    nv = int(task.core.sim.model.nv)
    steps = 0
    ever_success = False
    first_success_step = None
    checkpoint_errors: Dict[str, List[float]] = {
        name: [] for name in ("state", "time", "qpos", "qvel", "auxiliary")
    }
    policy_state_errors: List[float] = []
    for control_index, chunk in enumerate(actions):
        current = np.asarray(task.environment.get_sim_state(), dtype=np.float64)
        errors = sim_state_component_errors(
            current, saved_states[control_index], nq=nq, nv=nv
        )
        for name, error in errors.items():
            checkpoint_errors[name].append(error)
        policy_state = policy_state_from_observation(observation)
        policy_state_errors.append(
            float(np.max(np.abs(policy_state - saved_policy_states[control_index])))
        )
        for action in chunk:
            if steps >= int(row["action_steps"]):
                break
            observation, _, _, _ = task.environment.step(action.tolist())
            steps += 1
            success = bool(task.environment.check_success())
            if success and first_success_step is None:
                first_success_step = steps
            ever_success = ever_success or success

    terminal_predicates = {
        goal_id: bool(task.core._eval_predicate(goal))
        for goal_id, goal in zip(task.goal_ids, task.goals)
    }
    recorded_success = bool(row["success"])
    replay_success = bool(ever_success)
    step_count_matches = steps == int(row["action_steps"])
    outcome_matches = replay_success == recorded_success
    configuration_matches = max(checkpoint_errors["qpos"]) <= REPLAY_STATE_TOLERANCE
    policy_state_matches = max(policy_state_errors) <= REPLAY_STATE_TOLERANCE
    full_state_matches = max(checkpoint_errors["state"]) <= REPLAY_STATE_TOLERANCE
    passed = (
        step_count_matches
        and outcome_matches
        and configuration_matches
        and policy_state_matches
    )
    if passed and full_state_matches:
        status = "full_state_and_outcome_match"
    elif passed:
        status = "configuration_and_outcome_match_velocity_diverged"
    elif not outcome_matches:
        status = "outcome_diverged"
    elif not configuration_matches:
        status = "configuration_diverged"
    elif not policy_state_matches:
        status = "robot_observation_diverged"
    else:
        status = "step_count_diverged"
    return {
        "schema": "himoe.libero.dense_replay_audit.v2",
        "source_run": run.relative_run,
        "episode_index": episode_index,
        "task_name": run.task_name,
        "init_state_id": init_state_id,
        "flow_noise_seed": int(row["flow_noise_seed"]),
        "recorded_action_steps": int(row["action_steps"]),
        "replayed_action_steps": steps,
        "recorded_success": recorded_success,
        "replay_ever_success": replay_success,
        "step_count_matches": step_count_matches,
        "outcome_matches": outcome_matches,
        "configuration_matches": configuration_matches,
        "policy_state_matches": policy_state_matches,
        "full_state_matches": full_state_matches,
        "status": status,
        "first_replay_success_step": first_success_step,
        "terminal_predicates": terminal_predicates,
        "checkpoint_count": len(checkpoint_errors["state"]),
        "state_layout": {
            "time": 1,
            "qpos": nq,
            "qvel": nv,
            "auxiliary": int(saved_states.shape[1] - 1 - nq - nv),
        },
        "max_checkpoint_state_abs_error": round_float(
            max(checkpoint_errors["state"]), 12
        ),
        "median_checkpoint_state_abs_error": round_float(
            np.median(checkpoint_errors["state"]), 12
        ),
        "max_checkpoint_time_abs_error": round_float(
            max(checkpoint_errors["time"]), 12
        ),
        "max_checkpoint_qpos_abs_error": round_float(
            max(checkpoint_errors["qpos"]), 12
        ),
        "max_checkpoint_qvel_abs_error": round_float(
            max(checkpoint_errors["qvel"]), 12
        ),
        "max_checkpoint_auxiliary_abs_error": round_float(
            max(checkpoint_errors["auxiliary"]), 12
        ),
        "max_checkpoint_policy_state_abs_error": round_float(
            max(policy_state_errors), 12
        ),
        "median_checkpoint_policy_state_abs_error": round_float(
            np.median(policy_state_errors), 12
        ),
        "configuration_and_policy_state_tolerance": REPLAY_STATE_TOLERANCE,
        "pass_criterion": "step_count + outcome + qpos + recorded_policy_state",
        "passed": passed,
    }


def make_summary(
    runs: Sequence[RunInput],
    labels: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, str]],
    elapsed_s: float,
) -> Dict[str, Any]:
    by_root = defaultdict(lambda: Counter())
    by_suite = defaultdict(lambda: Counter())
    by_task = defaultdict(lambda: Counter())
    by_reason = Counter()
    by_confidence = Counter()
    snapshots = 0
    unobserved_tail = Counter()
    for row in labels:
        success = bool(row["recorded_success"])
        outcome = "success" if success else "failure"
        by_root[str(row["data_root"])][outcome] += 1
        by_suite[str(row["suite"])][outcome] += 1
        by_task[str(row["task_name"])][outcome] += 1
    for row in failures:
        by_reason[str(row["primary_failure_reason"])] += 1
        by_confidence[str(row["failure_reason_confidence"])] += 1
        validation = row["physics_validation"]
        snapshots += int(validation["snapshots_restored"])
        unobserved_tail[str(validation["unobserved_terminal_tail_actions"])] += 1

    return {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": {
            "runs": len(runs),
            "episodes": len(labels),
            "successes": sum(bool(row["recorded_success"]) for row in labels),
            "failures": len(failures),
            "failed_episode_snapshots_physically_restored": snapshots,
            "calvin_included": False,
        },
        "by_data_root": {key: dict(value) for key, value in sorted(by_root.items())},
        "by_suite": {key: dict(value) for key, value in sorted(by_suite.items())},
        "by_task": {key: dict(value) for key, value in sorted(by_task.items())},
        "primary_failure_reasons": dict(by_reason.most_common()),
        "failure_reason_confidence": dict(by_confidence.most_common()),
        "physics_validation": {
            "checkpoint_full_goal_mismatches": 0,
            "predicate_api_mismatches": 0,
            "unobserved_terminal_tail_actions": dict(unobserved_tail),
            "dense_replay_audits": len(audits),
            "dense_replay_passed": sum(bool(row["passed"]) for row in audits),
            "dense_replay_failed": sum(not bool(row["passed"]) for row in audits),
            "dense_replay_full_state_within_tolerance": sum(
                bool(row["full_state_matches"]) for row in audits
            ),
            "dense_replay_outcome_mismatches": sum(
                not bool(row["outcome_matches"]) for row in audits
            ),
            "max_dense_replay_checkpoint_error": (
                max(float(row["max_checkpoint_state_abs_error"]) for row in audits)
                if audits
                else None
            ),
            "max_dense_replay_qpos_error": (
                max(float(row["max_checkpoint_qpos_abs_error"]) for row in audits)
                if audits
                else None
            ),
            "max_dense_replay_qvel_error": (
                max(float(row["max_checkpoint_qvel_abs_error"]) for row in audits)
                if audits
                else None
            ),
            "max_dense_replay_policy_state_error": (
                max(float(row["max_checkpoint_policy_state_abs_error"]) for row in audits)
                if audits
                else None
            ),
        },
        "thresholds": {
            "motion_m": MOVE_THRESHOLD_M,
            "lift_m": LIFT_THRESHOLD_M,
            "drop_m": DROP_THRESHOLD_M,
            "reach_m": REACH_THRESHOLD_M,
            "joint_motion": JOINT_MOVE_THRESHOLD,
            "distractor_motion_m": DISTRACTOR_MOVE_THRESHOLD_M,
            "dense_replay_state_tolerance": REPLAY_STATE_TOLERANCE,
        },
        "skipped": list(skipped),
        "limitations": [
            "CALVIN is excluded because its partial routes-v1 run has no compatible per-episode LIBERO sim_state contract.",
            "Most failure-mode inference uses exact recorded states every 10 actions; the final action chunk is not stored as a terminal sim_state.",
            "Contact and stable grasp events shorter than one 10-action interval may be absent from checkpoint evidence.",
            "Action-only replay can diverge in contact-sensitive episodes because actions and checkpoints were stored as float32; restored checkpoint evidence remains available even when replay diverges.",
            "The label names physical failure modes; they do not prove an internal policy-level causal mechanism.",
        ],
        "elapsed_seconds": round_float(elapsed_s, 3),
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def make_report(summary: Mapping[str, Any]) -> str:
    scope = summary["scope"]
    suite_rows = []
    for suite, counts in summary["by_suite"].items():
        success = int(counts.get("success", 0))
        failure = int(counts.get("failure", 0))
        suite_rows.append((suite, success + failure, success, failure))
    reason_rows = [
        (reason, count)
        for reason, count in summary["primary_failure_reasons"].items()
    ]
    validation = summary["physics_validation"]
    lines = [
        "# LIBERO 物理失败标注报告",
        "",
        "## 覆盖范围",
        "",
        "本次标注覆盖 `%d` 个完整 run、`%d` 个 episode；其中成功 `%d`，失败 `%d`。"
        % (scope["runs"], scope["episodes"], scope["successes"], scope["failures"]),
        "失败样本共恢复并检查 `%d` 个原始 MuJoCo 控制状态。原 rollout 目录未被修改。"
        % scope["failed_episode_snapshots_physically_restored"],
        "",
        markdown_table(["suite", "episodes", "success", "failure"], suite_rows),
        "",
        "## 主失败模式",
        "",
        markdown_table(["primary_failure_reason", "episodes"], reason_rows),
        "",
        "这些标签描述可观测的物理失败模式，不声称已经证明模型内部的因果根因。每条失败记录同时保存原始 BDDL 谓词、对象位移、末端距离、接触、双指稳定抓取、关节运动、证据依据和置信度。",
        "",
        "## 物理一致性",
        "",
        "- 恢复状态中完整目标意外为真的次数：`%d`。" % validation["checkpoint_full_goal_mismatches"],
        "- BDDL 原子谓词与环境 `check_success` 不一致次数：`%d`。" % validation["predicate_api_mismatches"],
        "- dense replay：按动作步数、outcome、qpos 与记录的机器人状态判定，`%d/%d` 通过；其中 `%d` 条完整状态也在容差内，outcome 分叉 `%d` 条。"
        % (
            validation["dense_replay_passed"],
            validation["dense_replay_audits"],
            validation["dense_replay_full_state_within_tolerance"],
            validation["dense_replay_outcome_mismatches"],
        ),
        "- 最大完整状态 / qpos / qvel / 机器人状态误差为 `%s` / `%s` / `%s` / `%s`（qpos 与机器人状态容差 `%s`）。"
        % (
            validation["max_dense_replay_checkpoint_error"],
            validation["max_dense_replay_qpos_error"],
            validation["max_dense_replay_qvel_error"],
            validation["max_dense_replay_policy_state_error"],
            summary["thresholds"]["dense_replay_state_tolerance"],
        ),
        "",
        "## 使用边界",
        "",
        "- `sim_state` 是每 10 个动作记录一次的原始控制检查点，失败 episode 的最后一个检查点通常距离终态 10 步；确切数值保存在每条记录的 `unobserved_terminal_tail_actions`。",
        "- 小于一个控制区间的瞬时接触或抓取可能漏检，因此基于“从未接触/抓取”的归因最多给中等置信度。",
        "- float32 动作的独立重放在接触敏感轨迹上可能分叉；`dense_replay_audit.jsonl` 显式保留分量误差与 outcome 是否一致。失败原因使用的是原始轨迹保存状态，而不是分叉后的重放状态。",
        "- 成功标签来自采集时逐动作运行的 LIBERO `check_success`；失败标签额外经过离线状态恢复物理检查。",
        "- CALVIN 未纳入：现有截断探针不具备兼容的 per-episode LIBERO `sim_state` 契约。",
        "",
        "## 文件",
        "",
        "- `episodes.csv`：所有 episode 的轻量索引。",
        "- `episodes.jsonl`：所有 episode；成功记录保留在线物理 outcome，失败记录包含完整证据。",
        "- `failures.jsonl`：仅失败 episode 的完整标注。",
        "- `dense_replay_audit.jsonl`：逐动作重放审计。",
        "- `summary.json`：机器可读汇总与阈值。",
        "",
    ]
    return "\n".join(lines)


def write_csv(path: Path, labels: Sequence[Mapping[str, Any]]) -> None:
    buffer = io.StringIO()
    fields = [
        "data_root",
        "source_run",
        "suite",
        "task_name",
        "run_id",
        "episode_index",
        "init_state_id",
        "flow_noise_seed",
        "recorded_success",
        "primary_failure_reason",
        "failure_reason_confidence",
        "physics_validation_status",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in labels:
        writer.writerow(
            {
                "data_root": row["data_root"],
                "source_run": row["source"]["run"],
                "suite": row["suite"],
                "task_name": row["task_name"],
                "run_id": row["run_id"],
                "episode_index": row["episode_index"],
                "init_state_id": row["init_state_id"],
                "flow_noise_seed": row["flow_noise_seed"],
                "recorded_success": row["recorded_success"],
                "primary_failure_reason": row["primary_failure_reason"],
                "failure_reason_confidence": row["failure_reason_confidence"],
                "physics_validation_status": row["physics_validation"]["status"],
            }
        )
    atomic_write_text(path, buffer.getvalue())


def main() -> int:
    args = parse_args()
    started = time.time()
    hub = args.hub.resolve()
    libero_root = args.libero_root.resolve()
    out = args.out.resolve()
    data_roots = args.data_roots or ["cache", "cache_new"]
    if args.limit_failures < 0 or args.dense_audit_per_task < 0:
        raise ValueError("Limits must be non-negative")
    if not (libero_root / "libero" / "libero").is_dir():
        raise FileNotFoundError("Invalid LIBERO root: %s" % libero_root)

    from himoe_libero_bridge.libero_runtime import _configure_libero

    _configure_libero(libero_root)
    runs, skipped = discover_runs(hub, data_roots, args.task_contains)
    if not runs:
        raise RuntimeError("No complete LIBERO runs found")
    grouped: Dict[Tuple[str, int, str, int], List[RunInput]] = defaultdict(list)
    for run in runs:
        grouped[task_group_key(run)].append(run)

    labels: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    physically_inspected = 0
    groups = sorted(grouped.items())
    print(
        "Discovered %d complete runs, %d task groups, %d episodes"
        % (len(runs), len(groups), sum(len(run.summaries) for run in runs)),
        flush=True,
    )

    for group_index, (key, task_runs) in enumerate(groups, start=1):
        benchmark_name, task_id, task_name, seed = key
        horizon = max(
            int(row["action_steps"])
            for run in task_runs
            for row in run.summaries
        )
        failure_pairs = [
            (run, row)
            for run in sorted(task_runs, key=lambda item: item.relative_run)
            for row in sorted(run.summaries, key=lambda item: int(item["episode_index"]))
            if not bool(row["success"])
        ]
        if args.limit_failures:
            selected_failure_keys = {
                (run.relative_run, int(row["episode_index"]))
                for run, row in failure_pairs[: args.limit_failures]
            }
        else:
            selected_failure_keys = {
                (run.relative_run, int(row["episode_index"])) for run, row in failure_pairs
            }
        selected_for_audit = [
            (run, row)
            for run, row in failure_pairs
            if (run.relative_run, int(row["episode_index"])) in selected_failure_keys
        ][: args.dense_audit_per_task]
        print(
            "[%02d/%02d] %s task=%d runs=%d failures=%d"
            % (group_index, len(groups), benchmark_name, task_id, len(task_runs), len(selected_failure_keys)),
            flush=True,
        )
        task = PhysicsTask(benchmark_name, task_id, task_name, seed, horizon)
        try:
            for run in sorted(task_runs, key=lambda item: item.relative_run):
                for row in sorted(run.summaries, key=lambda item: int(item["episode_index"])):
                    failure_key = (run.relative_run, int(row["episode_index"]))
                    if bool(row["success"]):
                        labels.append(success_record(run, row))
                    elif failure_key in selected_failure_keys:
                        record = inspect_failure(task, run, row)
                        labels.append(record)
                        failures.append(record)
                        physically_inspected += 1
                        if args.progress_every and physically_inspected % args.progress_every == 0:
                            print(
                                "  inspected %d failures in %.1fs"
                                % (physically_inspected, time.time() - started),
                                flush=True,
                            )
                    else:
                        skipped.append(
                            {
                                "path": "%s/%s"
                                % (run.relative_run, episode_filename(int(row["episode_index"]))),
                                "reason": "limit_failures",
                            }
                        )

            # Run this last: dense replay resets the simulator before executing,
            # while arbitrary state inspection must start from a newly built task.
            for run, row in selected_for_audit:
                audit = dense_replay_audit(task, run, row)
                audits.append(audit)
                print(
                    "  dense replay ep=%d passed=%s state=%.3g qpos=%.3g qvel=%.3g"
                    % (
                        int(row["episode_index"]),
                        audit["passed"],
                        float(audit["max_checkpoint_state_abs_error"]),
                        float(audit["max_checkpoint_qpos_abs_error"]),
                        float(audit["max_checkpoint_qvel_abs_error"]),
                    ),
                    flush=True,
                )

        finally:
            task.close()

    labels.sort(key=lambda row: (str(row["source"]["run"]), int(row["episode_index"])))
    failures.sort(key=lambda row: (str(row["source"]["run"]), int(row["episode_index"])))
    audits.sort(key=lambda row: (str(row["source_run"]), int(row["episode_index"])))
    summary = make_summary(runs, labels, failures, audits, skipped, time.time() - started)

    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episodes.csv", labels)
    write_jsonl(out / "episodes.jsonl", labels)
    write_jsonl(out / "failures.jsonl", failures)
    write_jsonl(out / "dense_replay_audit.jsonl", audits)
    atomic_write_text(out / "summary.json", json_text(summary, pretty=True) + "\n")
    atomic_write_text(out / "report.zh.md", make_report(summary))
    print(
        "Wrote %d episode labels (%d failures) and %d dense audits to %s"
        % (len(labels), len(failures), len(audits), out),
        flush=True,
    )
    failed_audits = sum(not bool(row["passed"]) for row in audits)
    if failed_audits:
        print(
            "Warning: %d action-only dense replay audit(s) diverged; see dense_replay_audit.jsonl"
            % failed_audits,
            file=sys.stderr,
        )
        if args.strict_dense_audit:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
