from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("annotate.py")
sys.path.insert(0, str(MODULE_PATH.parents[2] / "himoe-libero-wrist-fix" / "src"))
SPEC = importlib.util.spec_from_file_location("physical_failure_annotate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def predicate(*, last=False, ever=False, regressed=False, unary=False):
    return {
        "expression": ["open", "drawer"] if unary else ["on", "object", "goal"],
        "last_checkpoint": last,
        "ever_satisfied": ever,
        "regressed_after_satisfaction": regressed,
    }


def object_evidence(**overrides):
    result = {
        "max_displacement_m": 0.0,
        "max_vertical_rise_m": 0.0,
        "fall_from_peak_to_last_m": 0.0,
        "ever_stable_grasp": False,
        "released_after_observed_grasp": False,
        "stable_grasp_at_last_checkpoint": False,
        "ever_gripper_contact": False,
        "contact_at_last_checkpoint": False,
        "min_gripper_distance_m": 1.0,
    }
    result.update(overrides)
    return result


def test_observed_predicate_regression_has_priority():
    label = MODULE.classify_goal_failure(
        predicate(ever=True, regressed=True), object_evidence(), 0.0
    )
    assert label["reason"] == "goal_predicate_regressed"
    assert label["confidence"] == "high"


def test_unmoved_unary_goal_is_mechanism_not_actuated():
    label = MODULE.classify_goal_failure(predicate(unary=True), None, 0.001)
    assert label["reason"] == "mechanism_not_actuated"


def test_grasp_release_and_height_loss_is_drop_mode():
    label = MODULE.classify_goal_failure(
        predicate(),
        object_evidence(
            max_displacement_m=0.2,
            max_vertical_rise_m=0.1,
            fall_from_peak_to_last_m=0.08,
            ever_stable_grasp=True,
            released_after_observed_grasp=True,
        ),
        0.0,
    )
    assert label["reason"] == "object_released_or_dropped_before_goal"


def test_contact_without_grasp_is_distinct_from_no_progress():
    label = MODULE.classify_goal_failure(
        predicate(), object_evidence(ever_gripper_contact=True), 0.0
    )
    assert label["reason"] == "stable_grasp_not_observed"


def test_sparse_contact_lift_and_fall_still_identifies_drop_mode():
    label = MODULE.classify_goal_failure(
        predicate(),
        object_evidence(
            max_displacement_m=0.15,
            max_vertical_rise_m=0.04,
            fall_from_peak_to_last_m=0.07,
            ever_gripper_contact=True,
            contact_at_last_checkpoint=False,
        ),
        0.0,
    )
    assert label["reason"] == "object_released_or_dropped_before_goal"


def test_satisfied_goal_is_not_a_failure_candidate():
    label = MODULE.classify_goal_failure(predicate(last=True, ever=True), None, 0.0)
    assert label["reason"] == "goal_satisfied_at_last_checkpoint"


def test_release_after_true_finds_first_transition():
    assert MODULE.first_false_after_true([False, True, True, False, True]) == 3
    assert MODULE.first_false_after_true([False, False]) is None


def test_sim_state_errors_are_split_by_mujoco_component():
    saved = np.zeros(8)
    current = np.array([0.1, 1.0, 2.0, 0.3, 0.4, 4.0, 5.0, 6.0])
    errors = MODULE.sim_state_component_errors(current, saved, nq=2, nv=2)
    assert errors == {
        "state": 6.0,
        "time": 0.1,
        "qpos": 2.0,
        "qvel": 0.4,
        "auxiliary": 6.0,
    }


def test_policy_state_matches_capture_layout():
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }
    state = MODULE.policy_state_from_observation(observation)
    np.testing.assert_allclose(
        state, np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04, -0.04])
    )
