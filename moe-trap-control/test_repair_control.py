"""CPU tests for the late-alarm repair experiment: controller math, guards, targets, protocol."""

import numpy as np
import pytest

from repair_controller import (retract_chunk, RepairPhase, physical_class, aperture,
                               eef_position, ProportionalRetract)


def test_retract_chunk_moves_toward_target_and_opens_gripper():
    eef = np.array([0.0, 0.0, 1.0])
    target = np.array([0.10, 0.0, 1.10])
    chunk = retract_chunk(eef, target, gain=0.8, unit_metres=0.05)
    assert chunk.shape == (10, 7)
    assert np.all(chunk[:, 6] == -1.0)
    assert chunk[0, 0] > 0 and chunk[0, 2] > 0
    assert np.all(np.abs(chunk[:, :3]) <= 1.0)
    assert np.all(chunk[:, 3:6] == 0.0)


def test_retract_chunk_is_zero_when_at_target():
    eef = np.array([0.3, -0.1, 0.9])
    chunk = retract_chunk(eef, eef.copy(), gain=0.8, unit_metres=0.05)
    assert np.all(chunk[:, :6] == 0.0)


def test_phase_guard_completes_when_close_for_five_steps():
    phase = RepairPhase(target=np.array([0.0, 0.0, 1.0]), tolerance=0.02, settle_steps=5, max_chunks=6)
    phase.begin_chunk()
    for _ in range(4):
        assert not phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.reason == "reached"


def test_phase_guard_times_out_after_max_chunks():
    phase = RepairPhase(target=np.array([1.0, 0.0, 1.0]), tolerance=0.02, settle_steps=5, max_chunks=2)
    for _ in range(2):
        phase.begin_chunk()
        for _ in range(10):
            phase.observe(np.array([0.0, 0.0, 1.0]))
    assert phase.exhausted and phase.reason == "max_chunks"


def test_physical_class_untouched_displaced_dropped():
    initial = np.array([0.0, 0.0, 0.90])
    eef = np.array([0.0, 0.0, 1.0])
    assert physical_class(np.array([0.005, 0.0, 0.90]), initial, eef, table_z=0.85, tilt_deg=0.0, grasped=False) == "untouched"
    assert physical_class(np.array([0.05, 0.0, 0.90]), initial, eef, table_z=0.85, tilt_deg=5.0, grasped=False) == "displaced_reachable"
    assert physical_class(np.array([0.05, 0.0, 0.70]), initial, eef, table_z=0.85, tilt_deg=0.0, grasped=False) == "dropped_or_tipped"
    assert physical_class(np.array([0.05, 0.0, 0.90]), initial, eef, table_z=0.85, tilt_deg=60.0, grasped=False) == "dropped_or_tipped"
    assert physical_class(np.array([0.0, 0.0, 1.0]), initial, eef, table_z=0.85, tilt_deg=0.0, grasped=True) == "in_hand"


def test_aperture_and_eef_from_observation():
    obs = {"robot0_gripper_qpos": np.array([0.03, -0.03]), "robot0_eef_pos": np.array([1.0, 2.0, 3.0])}
    assert aperture(obs) == pytest.approx(0.06)
    assert np.allclose(eef_position(obs), [1.0, 2.0, 3.0])


def test_proportional_retract_plans_chunks_until_guard():
    controller = ProportionalRetract(target=np.array([0.0, 0.0, 1.10]), gain=0.8, unit_metres=0.05,
                                     tolerance=0.02, settle_steps=5, max_chunks=6)
    eef = np.array([0.0, 0.0, 1.00])
    chunk = controller.next_chunk(eef)
    assert chunk.shape == (10, 7) and chunk[0, 2] > 0
    assert controller.phase.chunks == 1


class _FakeSimData:
    body_xpos = np.array([[0.0, 0.0, 0.9], [0.3, 0.0, 0.9]])
    body_xquat = np.array([[1.0, 0, 0, 0], [1.0, 0, 0, 0]])


class _FakeSim:
    data = _FakeSimData()


class _FakeInner:
    parsed_problem = {"goal_state": [["On", "pot_1", "stove_1"], ["On", "pot_2", "stove_1"]],
                      "objects": {"moka_pot": ["pot_1", "pot_2"]}, "fixtures": {"stove": ["stove_1"]}}
    obj_body_id = {"pot_1": 0, "pot_2": 1}
    sim = _FakeSim()

    def _eval_predicate(self, state):
        return state[1] == "pot_2"


class _FakeEnv:
    env = _FakeInner()


def test_resolve_target_returns_first_unsatisfied_movable_object():
    from repair_controller import resolve_target
    target = resolve_target(_FakeEnv())
    assert target["name"] == "pot_1"
    assert np.allclose(target["position"], [0.0, 0.0, 0.9])
    assert target["unsatisfied"] == [["On", "pot_1", "stove_1"]]
    assert target["articulated_pending"] is False


def test_tilt_degrees_identity_and_flat():
    from repair_controller import tilt_degrees
    assert tilt_degrees([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert tilt_degrees([0.0, 1.0, 0.0, 0.0]) == pytest.approx(180.0)


def test_repair_events_three_fork_times():
    from repair_control import events_for, LATE_QUERY, early_event
    events = events_for("abc", knn_first=25, length=52, failed=True)
    assert [e["timing"] for e in events] == ["mid", "late"]
    assert events[0]["start_query"] == 26 and events[1]["start_query"] == LATE_QUERY == 44
    assert events_for("abc", knn_first=25, length=52, failed=False) == events[:1]
    assert events_for("abc", knn_first=-1, length=52, failed=True) == [dict(events[1], alarm_query=-1)]
    assert early_event("abc", 30)["timing"] == "early" and early_event("abc", 30)["start_query"] == 30
    assert len({e["event_id"] for e in events + [early_event("abc", 30)]}) == 3


def test_arm_registry_orders_by_strength():
    from repair_control import ARMS, seed_for, noise_for
    assert list(ARMS) == ["new_noise", "open_only", "retract_history", "retract_above_target",
                          "retract_above_target_noisy", "scripted_regrasp"]
    assert ARMS["retract_above_target_noisy"]["xy_noise_m"] == 0.02
    assert seed_for("abc", 0, 0, "policy") != seed_for("abc", 1, 0, "policy")
    assert noise_for("abc", 0, 3).shape == (10, 24)
    with pytest.raises(ValueError):
        seed_for("abc", 0, 0, "other")


def test_repair_plan_covers_all_arms():
    from collect_repair_control import repair_plan
    physics = dict(history_pose=[0.1, 0.2, 1.0], target_position=[0.3, 0.0, 0.9])
    eef = np.array([0.0, 0.0, 1.0])
    assert repair_plan("new_noise", physics, eef, "abc", 0)[0] == "none"
    kind, ctrl, target = repair_plan("open_only", physics, eef, "abc", 0)
    assert kind == "open" and ctrl is None and np.allclose(target, eef)
    kind, ctrl, target = repair_plan("retract_history", physics, eef, "abc", 0)
    assert kind == "retract" and np.allclose(target, [0.1, 0.2, 1.0])
    kind, ctrl, target = repair_plan("retract_history", dict(physics, history_pose=None), eef, "abc", 0)
    assert np.allclose(target, [0.0, 0.0, 1.10])
    kind, ctrl, target = repair_plan("retract_above_target", physics, eef, "abc", 0)
    assert kind == "retract" and np.allclose(target, [0.3, 0.0, 1.0])
    kind, _, noisy = repair_plan("retract_above_target_noisy", physics, eef, "abc", 0)
    assert kind == "retract" and abs(noisy[2] - 1.0) < 1e-9 and np.linalg.norm(noisy[:2] - [0.3, 0.0]) < 0.1
    _, _, noisy_again = repair_plan("retract_above_target_noisy", physics, eef, "abc", 0)
    assert np.allclose(noisy, noisy_again)
    kind, ctrl, target = repair_plan("scripted_regrasp", physics, eef, "abc", 1)
    assert kind == "regrasp" and ctrl.phase.max_chunks == 6
    assert repair_plan("retract_above_target", dict(target_position=None), eef, "abc", 0)[0] == "unavailable"


def test_late_event_dropped_when_mid_alarm_lands_on_it():
    from repair_control import events_for, LATE_QUERY
    events = events_for("abc", knn_first=LATE_QUERY - 1, length=52, failed=True)
    assert [e["timing"] for e in events] == ["mid"] and events[0]["start_query"] == LATE_QUERY
