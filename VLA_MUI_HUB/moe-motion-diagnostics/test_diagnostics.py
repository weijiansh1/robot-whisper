import numpy as np
import pandas as pd
import pytest

from diagnostics import (command_features, confirmed, event_window_start,
                         first_hit, motion_features, route_features, valid_alarms)
from analyze import join_history, matched_auc, matched_correlation, method_metrics


def state_from_xyz(xyz):
    state = np.zeros((len(xyz), 8))
    state[:, :3] = xyz
    return state


def actions(n):
    return np.zeros((n, 10, 7))


def test_monotone_motion_is_not_periodic_or_backtracking():
    xyz = np.arange(30)[:, None] * np.array([[0.01, 0.003, 0]])
    out = motion_features(state_from_xyz(xyz), actions(30))
    assert not out["periodic_return"].any()
    assert not out["backtracking"].any()
    np.testing.assert_allclose(out["pose_efficiency"][6:], 1)
    np.testing.assert_allclose(out["pose_lag4"][12:] / out["pose_lag1"][12:], 4)


@pytest.mark.parametrize("period", [2, 3, 4, 5])
def test_true_cycles_require_sufficient_history_and_return(period):
    angle = 2 * np.pi * np.arange(35) / period
    xyz = 0.02 * np.stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)], axis=-1)
    out = motion_features(state_from_xyz(xyz), actions(35))
    assert first_hit(out["periodic_return"]) == 13
    assert np.max(out["pose_return_ratio"][12:]) < 1e-10


def test_tiny_jitter_does_not_pass_movement_floor():
    q = np.arange(30)
    xyz = np.stack([1e-7 * (-1.0) ** q, q * 1e-9, np.zeros(30)], axis=-1)
    out = motion_features(state_from_xyz(xyz), actions(30))
    assert not out["periodic_return"].any()
    assert not out["backtracking"].any()


def test_zero_initial_movement_abstains_and_later_freezing_is_separate():
    out = motion_features(np.zeros((30, 8)), actions(30))
    assert not out["still"].any()
    assert np.isnan(out["pose_relative_speed"]).all()
    xyz = np.minimum(np.arange(30), 5)[:, None] * np.array([[0.01, 0, 0]])
    out = motion_features(state_from_xyz(xyz), actions(30))
    assert out["still"][-1]
    assert not out["periodic_return"].any()


def test_predicted_command_jitter_has_frequency_and_amplitude_gates():
    u = actions(4)
    u[:, :, 0] = 0.2 * (-1.0) ** np.arange(10)
    f = command_features(u)
    assert first_hit(f["command_jitter"]) == 1
    assert np.all(f["command_high_fraction"] > 0.95)
    assert not command_features(u * 1e-5)["command_jitter"].any()
    u[:, :, 0] = np.arange(10) * 0.1
    assert not command_features(u)["command_jitter"].any()


def test_route_cycle_is_layer_specific_and_frozen_routes_are_not_cycles():
    p = np.full((30, 8, 10, 32), 1 / 32)
    cycle = 0.01 * np.cos(2 * np.pi * np.arange(30) / 4)
    p[:, :4, :, 0] += cycle[:, None, None]
    p[:, :4, :, 1] -= cycle[:, None, None]
    out = route_features(p)
    assert first_hit(out["route_periodic_front"]) == 13
    assert not out["route_periodic_back"].any()


def test_all_descriptors_are_prefix_invariant():
    rng = np.random.default_rng(42)
    xyz = np.cumsum(rng.normal(0, 0.01, (30, 3)), axis=0)
    state = state_from_xyz(xyz)
    commands = rng.normal(0, 0.1, (30, 10, 7))
    p = rng.dirichlet(np.ones(32), size=(30, 8, 10))
    full = motion_features(state, commands) | route_features(p)
    for n in (1, 4, 5, 7, 13, 18, 24):
        prefix = motion_features(state[:n], commands[:n]) | route_features(p[:n])
        for key, value in prefix.items():
            np.testing.assert_allclose(value, full[key][:n], equal_nan=True, err_msg=key)


def test_confirmation_is_not_backdated_and_restarts_after_gap():
    mask = np.array([False, True, True, False, True, True, True])
    np.testing.assert_array_equal(confirmed(mask, 2), [False, False, True, False, False, True, True])
    assert first_hit(confirmed(mask, 2)) == 2


def test_padding_and_length_boundary_cannot_be_alarms():
    alarms, invalid = valid_alarms(np.array([-1, 3, 4, 9]), np.array([4, 4, 4, 4]))
    np.testing.assert_array_equal(alarms, [-1, 3, -1, -1])
    np.testing.assert_array_equal(invalid, [False, False, True, True])


def test_alarm_metrics_keep_late_success_false_alarms():
    frame = pd.DataFrame({"task": ["a", "a", "a"], "failure": [True, False, True],
                          "length": [12, 5, 12]})
    table = method_metrics(frame, {"test": np.array([2, 4, -1])})
    row = table[table.group == "all"].iloc[0]
    assert row.tp_lead4 == 1
    assert row.failures == 2
    assert row.fp_any == 1
    assert row.historical_fp_lead4 == 0


def test_event_window_uses_oldest_required_observation():
    assert event_window_start("periodic_return", 14) == 1
    assert event_window_start("still", 14) == 7
    assert event_window_start("command_jitter", 14) == 13
    assert event_window_start("still", -1) == -1


def test_matched_auc_does_not_make_cross_task_pairs():
    frame = pd.DataFrame({"task": ["a", "a", "b", "b"], "init_state_id": [0] * 4,
                          "failure": [True, True, False, False]})
    result = matched_auc(frame, np.array([1, 1, 0, 0]))
    assert result["pairs"] == 0
    assert np.isnan(result["auc"])
    frame["failure"] = [True, False, True, False]
    result = matched_auc(frame, np.array([0, 0, 100, 100]))
    assert result["pairs"] == 2
    assert result["auc"] == 0.5


def test_history_join_is_by_identity_not_file_order():
    frame = pd.DataFrame({"row": [0, 1], "source_run": ["a", "a"], "episode": [0, 1],
                          "init_state_id": [3, 4], "flow_noise_seed": [100, 101],
                          "length": [10, 11], "success": [True, False]})
    history = pd.DataFrame({"source_run": ["a", "a"], "episode": [1, 0], "init_state": [4, 3],
                            "seed": [101, 100], "length": [11, 10], "success": [False, True],
                            "freeze_back_half_k4": [9, -1], "primary_failure_reason": ["test", ""]})
    out = join_history(frame, history)
    assert out.freeze_back_half_k4.tolist() == [-1, 9]
    history.loc[0, "seed"] = 999
    with pytest.raises(ValueError, match="history identity mismatch"):
        join_history(frame, history)


def test_motion_route_correlation_controls_task_and_init_offsets():
    frame = pd.DataFrame({"task": ["a"] * 3 + ["b"] * 3, "init_state_id": [0] * 6})
    result = matched_correlation(frame, np.array([1, 2, 3, 101, 102, 103]),
                                  np.array([3, 2, 1, 203, 202, 201]))
    assert result["rank_correlation"] == pytest.approx(-1)
    assert result["groups"] == 2
    assert result["contributing_episodes"] == 6
