"""Checks for time leakage, reference labels, grouping, and conformal tails."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


def load_local(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


core = load_local("safe_protocol_core_test", "core.py")
old_core = sys.modules.get("core")
sys.modules["core"] = core
try:
    runner = load_local("safe_protocol_runner_test", "run.py")
    evaluator = load_local("safe_protocol_evaluator_test", "evaluate.py")
finally:
    if old_core is None:
        del sys.modules["core"]
    else:
        sys.modules["core"] = old_core


def test_route_and_behavior_features_use_only_prefix():
    rng = np.random.default_rng(7)
    routes = rng.random((13, 8, 11, 32))
    state, actions = rng.normal(size=(13, 8)), rng.normal(size=(13, 10, 7))
    whole = core.route_features(routes)
    b, d = core.behavior_features(state, actions, np.ones(7))
    for stop in range(1, 14):
        prefix = core.route_features(routes[:stop])
        for name in whole:
            np.testing.assert_allclose(whole[name][:stop], prefix[name], equal_nan=True)
        bp, dp = core.behavior_features(state[:stop], actions[:stop], np.ones(7))
        np.testing.assert_allclose(b[:stop], bp, equal_nan=True)
        np.testing.assert_allclose(d[:stop], dp, equal_nan=True)
    assert np.isnan(whole["history"][:5, 32:]).all()


def test_label_reference_changes_direction_without_fitting_parameters():
    success = np.asarray([[0.0], [0.1], [-0.1], [0.2], [-0.2]])
    failure = success + 10
    target = np.asarray([[0.0], [10.0]])
    original = core.ReferenceDistance(success, failure).score(target)
    flipped = core.ReferenceDistance(failure, success).score(target)
    assert original[0, 0] < 0 < original[1, 0]
    np.testing.assert_allclose(original[:, 0], -flipped[:, 0])


def test_conformal_uses_finite_sample_rank_strict_ties_and_infinity():
    values = np.arange(19, dtype=float)
    threshold, rank = core.conformal_threshold(values, 0.05)
    assert rank == 19 and threshold == 18
    threshold, rank = core.conformal_threshold(values, 0.01)
    assert rank == 20 and np.isposinf(threshold)
    np.testing.assert_array_equal(core.first_alarm(np.asarray([[1., 2., 2.], [1., 2., 3.]]), 2.), [-1, 2])
    assert np.isneginf(core.trajectory_peak(np.full((1, 3), np.nan))[0])


def test_task_and_initial_state_splits_are_disjoint():
    frame = pd.DataFrame([dict(task=f"suite/task{t}", run_id=run, init_state_id=s,
                               noise_seed=1000 + j + offset)
                          for offset, run in zip((0, 8), core.RUNS)
                          for t in range(10) for s in range(50) for j in range(8)])
    reference, cal, test, seen, unseen = runner.make_split(frame, core.SEEDS[0], 0)
    assert (len(reference), len(cal), len(test)) == (1680, 560, 1760)
    assert len(seen) == 7 and len(unseen) == 3
    assert not set(frame.iloc[reference].task) & set(unseen)


def test_cumulative_abstains_before_first_valid_query():
    values = np.asarray([[np.nan, np.nan, -2., 4., np.nan]])
    np.testing.assert_allclose(core.aggregate(values, "cumsum"), [[np.nan, np.nan, -2., 2., np.nan]], equal_nan=True)
    with pytest.raises(ValueError):
        core.aggregate(values, "unknown")


def test_future_test_labels_cannot_change_calibration():
    scores = np.arange(30 * 52, dtype=float).reshape(1, 30, 52)
    frame = pd.DataFrame({"task": ["task"] * 30, "init_state_id": np.arange(30)})
    labels = np.zeros(30, int)
    ref, cal, test = np.arange(10), np.arange(10, 20), np.arange(20, 30)
    left = runner.calibrate(scores, labels, frame, ref, cal, test)
    labels[test] = 1
    right = runner.calibrate(scores, labels, frame, ref, cal, test)
    for a, b in zip(left[:5], right[:5]):
        np.testing.assert_array_equal(a, b)


def test_end_of_success_alarm_counts_and_misses_have_unit_time():
    frame = pd.DataFrame({"failure": [False, True, True], "length": [5, 10, 10]})
    metrics = evaluator.alarm_metrics(frame, np.asarray([4, -1, 1]))
    assert metrics["fp"] == 1 and metrics["fpr"] == 1
    assert metrics["recall"] == 0.5 and metrics["t_det"] == 0.6


def test_task_macro_rejects_pure_between_task_length_signal():
    frame = pd.DataFrame({"task": ["a"] * 4 + ["b"] * 4,
                          "init_state_id": [0] * 8,
                          "failure": [False, False, False, True, False, True, True, True]})
    score = np.asarray([5.] * 4 + [20.] * 4)
    result = evaluator.ranking(frame, score)
    assert result["auc"] > 0.5
    assert result["task_macro_auc"] == 0.5 and result["within_init_macro_auc"] == 0.5
