import numpy as np
import pandas as pd

from monitor import (auc, clean_first, conformal_threshold, cumulative_counts, first_alarm,
                     first_and, persistent, prefix_peak, v8_streams)
from physical_controls import events_for_episode
from run_analysis import state_splits


def test_cumulative_false_alarm_keeps_ended_success():
    first = np.asarray([3, -1, 7, 10])
    failure = np.asarray([False, False, True, False])
    length = np.asarray([5, 6, 12, 10])
    counts = cumulative_counts(first, failure, length, 8)
    assert counts == dict(tp=1, fp=1, failures=1, successes=3, active_failure=1, active_success=1)
    assert cumulative_counts(first, failure, length, 20)["fp"] == 1


def test_padding_cannot_create_an_alarm():
    values = np.asarray([[0., 0., 0., 100., 100.]])
    valid = np.asarray([[True, True, True, False, False]])
    assert first_alarm(values, valid, 1.).tolist() == [-1]
    assert clean_first(np.asarray([3, 2]), np.asarray([3, 3])).tolist() == [-1, 2]


def test_historical_strict_and_inclusive_thresholds():
    values = np.asarray([[0., 0., 1.]])
    valid = np.ones_like(values, bool)
    assert first_alarm(values, valid, 0.).tolist() == [2]
    assert first_alarm(values, valid, 0., inclusive=True).tolist() == [0]


def test_new_head_needs_two_eligible_queries():
    values = np.ones((1, 10))
    values[:, :6] = np.nan
    held = persistent(values, 2)
    assert first_alarm(held, np.ones_like(values, bool), .5).tolist() == [7]
    values[0, 7] = 0
    assert first_alarm(persistent(values, 2), np.ones_like(values, bool), .5).tolist() == [9]


def test_turbulence_combines_latched_crossings():
    acceleration = np.asarray([[0., 2., 0., 0., 0.]])
    periodicity = np.asarray([[0., 0., 0., 2., 0.]])
    valid = np.ones_like(acceleration, bool)
    first = first_and(first_alarm(acceleration, valid, 1.), first_alarm(periodicity, valid, 1.))
    assert first.tolist() == [3]
    guard = np.minimum(prefix_peak(acceleration), prefix_peak(periodicity))
    assert first_alarm(guard, valid, 1.).tolist() == [3]


def test_v8_prefix_causality_and_padding_invariance():
    rng = np.random.default_rng(3)
    raw = rng.uniform(.01, 1., (2, 20, 2)).astype(np.float32)
    valid = np.arange(20)[None] < np.asarray([12, 17])[:, None]
    full = v8_streams(raw, valid)
    corrupted = raw.copy()
    corrupted[~valid] = 1e8
    np.testing.assert_allclose(full, v8_streams(corrupted, valid), equal_nan=True)
    assert not np.isfinite(full[~valid]).any()
    for stop in (7, 9, 12, 17):
        prefix = v8_streams(raw[:, :stop], valid[:, :stop])
        np.testing.assert_allclose(full[:, :stop], prefix, equal_nan=True)


def test_curvature_self_baseline_removes_multiplicative_scale():
    raw = np.random.default_rng(4).uniform(.01, 1., (1, 15, 2)).astype(np.float32)
    valid = np.ones(raw.shape[:2], bool)
    scaled = raw.copy()
    scaled[:, :, 1] *= 10
    np.testing.assert_allclose(v8_streams(raw, valid), v8_streams(scaled, valid), atol=3e-7, equal_nan=True)


def test_small_calibration_set_cannot_support_tiny_budget():
    threshold, rank = conformal_threshold(np.arange(19), .01)
    assert np.isinf(threshold) and rank == 20
    threshold, rank = conformal_threshold(np.arange(99), .05)
    assert threshold == 94 and rank == 95


def test_calibration_ties_are_conservative():
    peaks = np.r_[np.zeros(90), np.ones(10)]
    threshold, _ = conformal_threshold(peaks, .05)
    assert threshold == 1 and (peaks > threshold).sum() == 0


def test_auc_ties_and_no_comparison():
    y = np.asarray([False, False, True, True])
    assert auc(y, [1., 1., 1., 1.]) == .5
    assert auc(y, [0., 0., 1., 1.]) == 1.
    assert auc(y, [1., 1., 0., 0.]) == 0.
    assert np.isnan(auc([False, False], [0., 1.]))


def test_initial_state_crossfit_owns_every_b_episode_once():
    frame = pd.DataFrame([dict(run_id=run, task=task, init_state_id=init)
                          for run in ("right-50x8-20260903", "right-50x8b-20260903")
                          for task in ("a", "b") for init in range(50)])
    seen = []
    for _, reference, calibration, test in state_splits(frame):
        assert (len(reference), len(calibration), len(test)) == (60, 20, 20)
        assert not set(frame.iloc[reference].init_state_id) & set(frame.iloc[calibration].init_state_id)
        assert not set(frame.iloc[test].init_state_id) & set(frame.iloc[reference].init_state_id)
        seen.extend(test)
    assert sorted(seen) == list(range(100, 200))


def test_release_definition_does_not_use_eventual_outcome():
    goals = np.asarray([[False], [False], [False], [False], [True]])
    grasps = np.asarray([[False], [True], [False], [False], [True]])
    xyz = np.zeros((5, 1, 3), np.float32)
    xyz[:, 0, 2] = [0., .2, .19, .13, .2]
    positive = events_for_episode({"failure": True}, goals, grasps, xyz, ["obj"], [["in", "obj", "dest"]])
    negative = events_for_episode({"failure": False}, goals, grasps, xyz, ["obj"], [["in", "obj", "dest"]])
    assert [x["query"] for x in positive] == [2, 3]
    assert [x["kind"] for x in positive] == ["release_outside_goal", "release_height_loss"]
    assert positive[0]["regrasp_q"] == 4 and positive[0]["goal_reached_q"] == 4
    assert [{k: v for k, v in x.items() if k != "failure"} for x in positive] == [
        {k: v for k, v in x.items() if k != "failure"} for x in negative]


def test_release_in_goal_is_separate_from_outside_goal():
    events = events_for_episode({}, np.asarray([[False], [True]]), np.asarray([[True], [False]]),
                                np.zeros((2, 1, 3)), ["obj"], [["in", "obj", "dest"]])
    assert len(events) == 1 and events[0]["kind"] == "release_in_goal"
