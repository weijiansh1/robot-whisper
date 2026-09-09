import copy
import json

import numpy as np
import pandas as pd
import pytest

from probability.branches import branch_candidates, summarize_branches
from probability.data import RUN_A, RUN_B, json_records, split_for, validate_record
from probability.evaluation import alarm_statistics, metrics, wilson
from probability.features import FEATURE_NAMES, PrefixFeatures, detector, history_features
from probability.model import SuccessMonitor, fit_readout, predict_readout


def raw_history(length=16):
    rng = np.random.default_rng(73)
    p = rng.uniform(0.5, 1.5, (length, 8, 10, 32))
    return (p / p.sum(axis=-1, keepdims=True)).astype(np.float16)


def test_features_are_invariant_to_unseen_suffix():
    raw = raw_history()
    features = detector.features_from_roots(detector.root_action_routes(raw)).astype(np.float32)
    whole, _ = history_features(features, 220)
    for q in range(len(raw)):
        prefix, _ = history_features(features[:q+1], 220)
        np.testing.assert_array_equal(prefix[-1], whole[q])
    changed = features.copy()
    changed[4:] *= 100
    altered, _ = history_features(changed, 220)
    np.testing.assert_array_equal(altered[:4], whole[:4])


def test_online_features_equal_batch_and_budget_is_not_endpoint():
    raw = raw_history(9)
    f = detector.features_from_roots(detector.root_action_routes(raw)).astype(np.float32)
    batch, _ = history_features(f, 520)
    stream = PrefixFeatures(520)
    online = np.concatenate([stream.update(p)[0] for p in raw])
    np.testing.assert_array_equal(online, batch)
    assert batch[-1, 0] == 440
    assert batch.shape[1] == len(FEATURE_NAMES)


def test_alarm_status_is_distinct_from_latched_history():
    f = np.ones((16, 8, 3), dtype=np.float32)
    f[0, :, 0] = np.nan
    f[:4, :, 1] = np.nan
    f[5:10, :, 0] = 0.1
    _, info = history_features(f, 220)
    assert info["first_alarm"] == 7
    assert not info["alarm_now"][-1]
    assert info["ever_alarm"][-1]


def test_initial_state_groups_are_separate_across_both_batches():
    a = {i: split_for("some_task", i, RUN_A) for i in range(50)}
    b = {i: split_for("some_task", i, RUN_B) for i in range(50)}
    assert list(a.values()).count("train") == 30
    assert list(a.values()).count("calibration") == 10
    assert {i for i in a if a[i] == "test_unseen_init"} == {i for i in b if b[i] == "test_unseen_init"}
    assert all(b[i] == "test_new_noise_seen_init" for i in a if a[i] != "test_unseen_init")


def test_timeout_is_negative_but_interruption_is_not():
    validate_record(dict(success=False, action_steps=220, inference_calls=22), 220, 10)
    validate_record(dict(success=True, action_steps=23, inference_calls=3), 220, 10)
    with pytest.raises(ValueError, match="censored"):
        validate_record(dict(success=False, action_steps=30, inference_calls=3), 220, 10)
    with pytest.raises(ValueError, match="incomplete"):
        validate_record(dict(success=True, action_steps=23, inference_calls=3, intervened=True), 220, 10)


def test_terminal_boundaries_and_invalid_budget():
    assert SuccessMonitor.terminal_probability(success=True, remaining_action_steps=0) == 1
    assert SuccessMonitor.terminal_probability(remaining_action_steps=0) == 0
    assert SuccessMonitor.terminal_probability(irreversible_failure=True, remaining_action_steps=100) == 0
    with pytest.raises(ValueError):
        history_features(np.ones((23, 8, 3)), 220)


def test_probability_metrics_have_expected_values_and_single_class_support():
    result = metrics(np.array([0, 1]), np.array([0.25, 0.75]))
    assert result["brier"] == pytest.approx(0.0625)
    assert result["log_loss"] == pytest.approx(-np.log(0.75))
    assert result["auroc"] == 1
    assert metrics(np.ones(2), np.full(2, 0.9))["auroc"] is None
    assert wilson(0, 64)[1] > 0
    assert wilson(64, 64)[0] < 1


def test_zero_alarm_successes_do_not_report_a_degenerate_cluster_interval():
    episodes = pd.DataFrame([dict(split="test_unseen_init", suite="suite", task="task", cluster=str(i),
                                  first_alarm_q=7, success=False) for i in range(8)])
    row = alarm_statistics(episodes).iloc[0]
    assert row.cluster_interval_status == "undefined_or_degenerate"
    assert row.cluster_low is None
    assert row.wilson_high > 0


def test_single_class_metrics_serialize_as_null_in_strict_json():
    frame = pd.DataFrame([dict(auc=0.5, value=1), dict(auc=np.nan, value=2)])
    records = json_records(frame)
    assert records[1]["auc"] is None
    json.dumps(records, allow_nan=False)


def branch_records():
    return pd.DataFrame([dict(checkpoint_id="checkpoint", branch_id=i, future_seed=100+i,
                             snapshot_sha256="a"*64, policy_sha256="b"*64, current_chunk_sha256="c"*64,
                             restore_audit_sha256="d"*64, remaining_action_steps=100,
                             success=False, complete=True, policy_unchanged=True,
                             current_chunk_preserved=True, full_restore_verified=True,
                             independent_future_rng=True) for i in range(8)])


def test_branch_soft_labels_include_uncertainty_at_zero_successes():
    result = summarize_branches(branch_records()).iloc[0]
    assert result.success_probability == 0
    assert result.wilson_high > 0
    assert result.soft_label_trials == 8


@pytest.mark.parametrize("column,value", [("current_chunk_preserved", False), ("complete", False),
                                         ("full_restore_verified", False), ("snapshot_sha256", "other")])
def test_branches_reject_misalignment_and_incomplete_evidence(column, value):
    records = branch_records()
    records.loc[0, column] = value
    with pytest.raises(ValueError):
        summarize_branches(records)


def test_branches_cannot_duplicate_future_draws():
    records = branch_records()
    records.loc[1, "future_seed"] = records.loc[0, "future_seed"]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_branches(records)


def test_checkpoint_selection_does_not_use_terminal_outcome():
    records = pd.DataFrame([dict(episode_row=i, source_run="run", episode=i, task="task", suite="suite",
                                 run_id=RUN_A, init_state=i, split="test_unseen_init", max_steps=220,
                                 replan_steps=10, first_alarm_q=7 if i < 3 else -1, length=12,
                                 success=i % 2 == 0) for i in range(8)])
    before = branch_candidates(records)
    records["success"] = ~records.success
    pd.testing.assert_frame_equal(before, branch_candidates(records))
    assert "success" not in before


def test_model_and_calibrator_use_explicit_disjoint_inputs():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(500, len(FEATURE_NAMES)))
    y = (x[:, 3] + rng.normal(size=500) > 0).astype(int)
    policy = dict(checkpoint_sha256="b"*64, max_steps=220, n_action_steps=10)
    model = fit_readout(x[:300], y[:300], x[300:400], y[300:400], policy)
    probabilities = predict_readout(model, x[400:])
    assert all(np.isfinite(v).all() and (v > 0).all() and (v < 1).all() for v in probabilities.values())
    assert model["estimators"]["moe"]["estimator"].early_stopping is False
    with pytest.raises(ValueError, match="different policy"):
        SuccessMonitor(model, "bad_checkpoint")
    changed = copy.deepcopy(model)
    changed["feature_names"] = []
    with pytest.raises(ValueError, match="version"):
        SuccessMonitor(changed, "b"*64)
