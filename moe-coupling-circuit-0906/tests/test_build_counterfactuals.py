"""Unit tests for the parts of the counterfactual builder that need no simulator."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from build_coupling_counterfactuals import (  # noqa: E402
    APPROACH_NEAR_M,
    CLOSE_APERTURE,
    EEF_DEPARTURE_M,
    LIFTED_M,
    VISIBILITY_PIXEL_DELTA,
    VISIBILITY_PIXEL_FRACTION,
    anchors,
    bearing,
    edited_state,
    flow_noise_at,
    qvel_slices,
    visibility,
)


REAL_LAYOUT = {
    "nq": 48,
    "nv": 43,
    "state_dim": 92,
    "joints": [
        {"joint": "robot0_joint%d" % i, "state_lo": i, "state_hi": i + 1, "is_robot": True}
        for i in range(1, 8)
    ]
    + [
        {"joint": "gripper0_finger_joint1", "state_lo": 8, "state_hi": 9, "is_robot": True},
        {"joint": "gripper0_finger_joint2", "state_lo": 9, "state_hi": 10, "is_robot": True},
        {"joint": "akita_black_bowl_1_joint0", "state_lo": 10, "state_hi": 17, "is_robot": False},
        {"joint": "akita_black_bowl_2_joint0", "state_lo": 17, "state_hi": 24, "is_robot": False},
        {"joint": "cookies_1_joint0", "state_lo": 24, "state_hi": 31, "is_robot": False},
        {"joint": "glazed_rim_porcelain_ramekin_1_joint0", "state_lo": 31, "state_hi": 38, "is_robot": False},
        {"joint": "plate_1_joint0", "state_lo": 38, "state_hi": 45, "is_robot": False},
        {"joint": "wooden_cabinet_1_top_level", "state_lo": 45, "state_hi": 46, "is_robot": False},
        {"joint": "wooden_cabinet_1_middle_level", "state_lo": 46, "state_hi": 47, "is_robot": False},
        {"joint": "wooden_cabinet_1_bottom_level", "state_lo": 47, "state_hi": 48, "is_robot": False},
        {"joint": "flat_stove_1_button", "state_lo": 48, "state_hi": 49, "is_robot": False},
    ],
}


def test_qvel_walk_matches_a_real_layout():
    slices = qvel_slices(REAL_LAYOUT)
    # The qvel block starts right after [time] + qpos.
    assert slices["robot0_joint1"] == (49, 50)
    # Nine single-dof robot joints come first, so the first free joint's six
    # velocity dims start at 49 + 9.
    assert slices["akita_black_bowl_1_joint0"] == (58, 64)
    assert slices["akita_black_bowl_2_joint0"] == (64, 70)
    # The walk must land exactly on state_dim.
    assert max(hi for _, hi in slices.values()) == REAL_LAYOUT["state_dim"]


def test_qvel_walk_rejects_a_layout_that_does_not_close():
    broken = json.loads(json.dumps(REAL_LAYOUT))
    broken["nv"] = 44
    with pytest.raises(RuntimeError, match="qvel walk produced"):
        qvel_slices(broken)


def test_qvel_walk_rejects_an_unknown_joint_width():
    broken = json.loads(json.dumps(REAL_LAYOUT))
    broken["joints"][-1]["state_hi"] = 50
    with pytest.raises(RuntimeError, match="unsupported qpos width"):
        qvel_slices(broken)


def _trace(n=12, close_at=5, lift_at=7, near=True, depart_at=7):
    state = np.zeros((n, 8), np.float32)
    sim = np.zeros((n, 20), np.float64)
    state[:, 6] = 0.04  # each finger, so the aperture sums to 0.08 -> open
    state[close_at:, 6] = 0.01  # aperture 0.02 -> closed
    state[:, 7] = state[:, 6]
    lo = 10
    sim[:, lo] = 0.30
    sim[:, lo + 2] = 0.90
    sim[lift_at:, lo + 2] = 0.90 + LIFTED_M + 0.01
    state[:, 0] = 0.30 - (0.05 if near else 0.40)
    state[:, 2] = 0.90
    # The arm carries the object away only from ``depart_at`` on.
    state[depart_at:, 2] = 0.90 + EEF_DEPARTURE_M + 0.01
    return state, sim, lo


def test_anchors_finds_the_approach_and_the_lift():
    state, sim, lo = _trace()
    found = anchors(state, sim, lo)
    assert found["ok"]
    assert found["closure"] == 5
    # q_pre is the last open query still near the target.
    assert found["q_pre"] == 4
    assert found["q_post"] == 7
    assert np.abs(state[found["q_pre"], 6:8]).sum() >= CLOSE_APERTURE
    assert np.abs(state[found["q_post"], 6:8]).sum() < CLOSE_APERTURE


def test_anchors_rejects_a_trace_that_never_closes():
    state, sim, lo = _trace()
    state[:, 6:8] = 0.04
    assert anchors(state, sim, lo) == {"ok": False, "reason": "no aperture closure"}


def test_anchors_rejects_a_trace_whose_target_never_lifts():
    state, sim, lo = _trace()
    sim[:, lo + 2] = 0.90
    assert anchors(state, sim, lo)["reason"] == "target never lifted and carried away"


def test_anchors_requires_the_arm_to_have_departed_not_just_lifted():
    """The gate the first smoke run was missing.

    A 2 cm lift leaves the closed gripper 2 cm above the table, so writing the
    object back to its resting pose puts it inside the fingers. Every `post`
    anchor was rejected for touching the robot. q_post now also requires the
    end effector to have travelled EEF_DEPARTURE_M from the closure pose.
    """
    lifted_but_close = _trace(lift_at=7, depart_at=11)
    found = anchors(*lifted_but_close)
    assert found["ok"]
    assert found["q_post"] == 11, "must wait for departure, not stop at the lift"

    never_departs = _trace(lift_at=7, depart_at=99)
    assert anchors(*never_departs)["reason"] == "target never lifted and carried away"


def test_anchors_rejects_a_closure_far_from_the_target():
    state, sim, lo = _trace(near=False)
    assert anchors(state, sim, lo)["reason"] == "closure is not near the target"


def test_approach_query_must_be_near():
    state, sim, lo = _trace()
    # Push the arm just outside the approach radius at every open query.
    state[:, 0] = 0.30 - (APPROACH_NEAR_M + 0.01)
    assert anchors(state, sim, lo)["reason"] == "no open-gripper query near the target"


def test_bearing_is_a_horizontal_unit_vector():
    direction = bearing(np.array([0.0, 0.0, 0.9]), np.array([0.3, 0.4, 1.0]))
    assert direction is not None
    assert np.allclose(direction, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_bearing_is_undefined_directly_above_the_target():
    assert bearing(np.array([0.0, 0.0, 0.9]), np.array([0.001, 0.0, 0.8])) is None


def test_edited_state_translates_and_zeroes_velocity_only():
    base = np.arange(92, dtype=np.float64)
    out = edited_state(base, 10, (58, 64), translation=np.array([0.1, -0.2, 0.0]))
    assert np.allclose(out[10:13], base[10:13] + [0.1, -0.2, 0.0])
    assert np.allclose(out[13:17], base[13:17])  # quaternion untouched
    assert np.allclose(out[58:64], 0.0)
    untouched = np.ones(92, bool)
    untouched[10:13] = False
    untouched[58:64] = False
    assert np.array_equal(out[untouched], base[untouched])


def test_edited_state_writes_a_whole_pose():
    base = np.arange(92, dtype=np.float64)
    pose = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    out = edited_state(base, 10, (58, 64), pose=pose)
    assert np.allclose(out[10:17], pose)
    assert np.allclose(out[58:64], 0.0)


def test_flow_noise_is_the_query_th_draw_not_the_first():
    first = flow_noise_at(1000, 0)
    third = flow_noise_at(1000, 2)
    assert first.shape == (10, 24)
    assert not np.allclose(first, third)
    # Deterministic given the seed and query.
    assert np.array_equal(third, flow_noise_at(1000, 2))


def _frames(fraction, delta):
    reference = {
        "image": np.zeros((224, 224, 3), np.uint8),
        "wrist_image": np.zeros((224, 224, 3), np.uint8),
    }
    candidate = {key: value.copy() for key, value in reference.items()}
    count = int(round(fraction * 224 * 224))
    flat = candidate["image"].reshape(-1, 3)
    flat[:count, 0] = delta
    return reference, candidate


def test_visibility_passes_when_enough_pixels_move():
    reference, candidate = _frames(VISIBILITY_PIXEL_FRACTION * 2, VISIBILITY_PIXEL_DELTA)
    seen = visibility(reference, candidate)
    assert seen["passed"]
    assert seen["image_changed_fraction"] > VISIBILITY_PIXEL_FRACTION


def test_visibility_fails_on_a_change_too_small_to_see():
    reference, candidate = _frames(0.5, VISIBILITY_PIXEL_DELTA - 1)
    assert not visibility(reference, candidate)["passed"]


def test_visibility_fails_on_a_change_too_sparse_to_see():
    reference, candidate = _frames(VISIBILITY_PIXEL_FRACTION / 4, 255)
    assert not visibility(reference, candidate)["passed"]


def test_visibility_is_asymmetric_in_neither_direction():
    reference, candidate = _frames(VISIBILITY_PIXEL_FRACTION * 2, 60)
    assert visibility(reference, candidate)["passed"]
    assert visibility(candidate, reference)["passed"]
