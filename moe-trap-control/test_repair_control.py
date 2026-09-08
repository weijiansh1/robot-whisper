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


# ---------------------------------------------------------------------------- supervisor v2
def test_phantom_guard_fires_on_departure_and_on_hold():
    from repair_controller import PhantomGuard
    guard = PhantomGuard(departure_m=0.05, hold_steps=40, moved_m=0.01)
    target = np.array([0.0, 0.0, 1.0])
    assert guard.observe([0.0, 0.0, 1.1], 0.08, target, 0, False) is None          # open: nothing
    assert guard.observe([0.0, 0.0, 1.02], 0.01, target, 1, False) is None         # closure recorded
    assert guard.observe([0.0, 0.0, 1.04], 0.01, target, 2, False) is None         # departed 2 cm only
    assert guard.observe([0.0, 0.0, 1.08], 0.01, target, 3, False) == "phantom_departed"
    guard.reset()
    assert guard.observe([0.0, 0.0, 1.02], 0.01, target, 10, False) is None
    assert guard.observe([0.0, 0.0, 1.02], 0.01, target, 49, False) is None
    assert guard.observe([0.0, 0.0, 1.02], 0.01, target, 50, False) == "phantom_held"
    guard.reset()                                                                   # in hand: never fires
    assert guard.observe([0.0, 0.0, 1.02], 0.03, target, 60, True) is None
    assert guard.observe([0.0, 0.0, 1.12], 0.03, target + [0, 0, 0.1], 70, True) is None
    guard.reset()                                                                   # pushed: never fires
    assert guard.observe([0.0, 0.0, 1.02], 0.03, target, 80, False) is None
    assert guard.observe([0.1, 0.0, 1.02], 0.03, target + [0.1, 0, 0], 81, False) is None


def test_hand_state_requires_tracking_motion():
    from repair_controller import HandState
    hand = HandState(rest_z=1.0, lifted_m=0.02)
    target = np.array([0.0, 0.0, 1.05])
    for i in range(12):                                                             # eef and object rise together
        assert hand.update([0.0, 0.0, 1.10 + 0.005 * i], 0.03, target + [0, 0, 0.005 * i]) is True
    for i in range(12):                                                             # eef moves away, object stays
        held = hand.update([0.02 * i, 0.0, 1.16], 0.03, target + [0, 0, 0.055])
    assert held is False
    assert hand.update([0.0, 0.0, 1.16], 0.08, target + [0, 0, 0.055]) is False   # open gripper: never in hand
    lonely = HandState(rest_z=1.0, lifted_m=0.02)
    assert lonely.update([0.3, 0.0, 1.2], 0.01, np.array([0.0, 0.0, 1.04])) is False   # lifted object far from the eef


def test_carry_guard_waits_for_patience_and_resets_on_progress():
    from repair_controller import CarryGuard
    guard = CarryGuard(patience_steps=120)
    assert guard.observe(2, 0, False) is None
    assert guard.observe(2, 10, True) is None
    assert guard.observe(2, 129, True) is None
    assert guard.observe(2, 130, True) == "carry_stalled"
    guard.reset()
    assert guard.observe(2, 200, True) is None
    assert guard.observe(1, 250, True) is None                                     # progress resets the clock
    assert guard.observe(1, 369, True) is None
    assert guard.observe(1, 370, True) == "carry_stalled"
    assert guard.observe(1, 371, False) is None                                    # put down: disarmed


def test_contact_descent_stops_when_the_object_stops_dropping():
    from repair_controller import ContactDescent
    ctrl = ContactDescent(np.array([0.0, 0.0, 0.5]), gain=0.4, unit_metres=0.05, settle_steps=5, max_chunks=4)
    chunk = ctrl.next_chunk(np.array([0.0, 0.0, 1.0]))
    assert chunk[0, 2] < 0 and chunk[0, 6] == 1.0
    z = 1.0
    for step in range(8):                                                          # falling
        z -= 0.01
        assert ctrl.phase.observe([0.0, 0.0, z + 0.05], [0.0, 0.0, z]) is False
    for step in range(5):                                                          # resting on the surface
        assert ctrl.phase.observe([0.0, 0.0, z + 0.05], [0.0, 0.0, z]) is (step == 4)
    assert ctrl.phase.reason == "contact"


def test_place_waypoints_keep_the_object_offset_and_land_inside_the_box():
    from repair_controller import place_waypoints
    region = dict(centre=np.array([0.3, -0.2, 1.0]), half=np.array([0.1, 0.1, 0.05]), predicate="In")
    eef, target = np.array([0.0, 0.0, 1.10]), np.array([0.0, 0.0, 1.05])
    w = place_waypoints(eef, target, region, half_height=0.04, height_m=0.15, clearance_m=0.02, rise_m=0.08)
    np.testing.assert_allclose(w["offset"], [0, 0, 0.05])
    assert w["floor"] == 0.95
    assert w["rise"][2] == 1.05 + 0.15 + 0.05 and w["rise"][0] == 0.0
    np.testing.assert_allclose(w["transfer"][:2], region["centre"][:2])
    object_centre = w["lower"] - w["offset"]
    assert 0.95 < object_centre[2] < 1.05 and object_centre[2] == 0.95 + 0.04 + 0.02
    assert w["retreat"][2] == w["lower"][2] + 0.08
    on = place_waypoints(eef, target, dict(region, predicate="On"), 0.04, 0.15, 0.02, 0.08)
    assert on["floor"] == 1.0 and (on["lower"] - on["offset"])[2] == 1.06


def test_supervisor_jobs_reuse_the_v1_replay_and_skip_c0():
    from repair_control import scheduled_jobs, SUPERVISOR
    plan = dict(replay_run="/run/v1", arms=["supervisor_full"], replicates=2, timings=["mid", "late"], supervisor=SUPERVISOR,
                tasks=[dict(main_id="m1", variant_id="v1", noise_seed=3, init_index=0)])
    jobs = scheduled_jobs(plan, {"v1": dict(benchmark="pro")}, "/out")
    assert [j["job_id"] for j in jobs] == ["m1/branches"] and jobs[0]["depends_on"] is None
    assert jobs[0]["sampling"]["replay_directory"] == "/run/v1/tasks/m1/replay"
    assert jobs[0]["sampling"]["timings"] == ["mid", "late"] and jobs[0]["sampling"]["supervisor"] == SUPERVISOR


def test_v1_jobs_still_replay_first():
    from repair_control import scheduled_jobs
    plan = dict(arms=["new_noise"], replicates=1, tasks=[dict(main_id="m1", variant_id="v1", noise_seed=3, init_index=0,
                                                              max_output_bytes=10)])
    jobs = scheduled_jobs(plan, {"v1": dict(benchmark="pro")}, "/out")
    assert [j["job_id"] for j in jobs] == ["m1/replay", "m1/branches"] and jobs[1]["depends_on"] == "m1/replay"


def test_place_point_moves_to_the_empty_half():
    from repair_controller import place_point
    region = dict(centre=np.array([0.0, 0.3, 0.5]), half=np.array([0.04, 0.08, 0.06]), predicate="In")
    point, occupied = place_point(region, [])
    np.testing.assert_allclose(point, region["centre"]) and occupied == 0
    point, occupied = place_point(region, [np.array([0.0, 0.34, 0.48]), np.array([0.5, 0.5, 0.5])])
    assert occupied == 1 and point[1] == 0.3 - 0.04 and point[0] == 0.0
    plate = dict(centre=np.array([0.0, -0.3, 0.44]), half=np.zeros(3), predicate="On")
    point, occupied = place_point(plate, [np.array([0.0, -0.3, 0.46])])
    np.testing.assert_allclose(point, plate["centre"])
