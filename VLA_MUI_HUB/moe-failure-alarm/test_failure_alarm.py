from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("moe_failure_alarm", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_mask_churn_is_set_jaccard() -> None:
    base = MODULE.top4_mask(np.asarray([[0, 1, 2, 3]], np.uint8))
    reordered = MODULE.top4_mask(np.asarray([[3, 2, 1, 0]], np.uint8))
    half = MODULE.top4_mask(np.asarray([[0, 1, 4, 5]], np.uint8))
    disjoint = MODULE.top4_mask(np.asarray([[4, 5, 6, 7]], np.uint8))
    assert np.allclose(MODULE.mask_churn(base, reordered), 0.0)
    assert np.allclose(MODULE.mask_churn(base, half), 1.0 - 2.0 / 6.0)
    assert np.allclose(MODULE.mask_churn(base, disjoint), 1.0)


def test_top4_mask_rejects_duplicate_experts() -> None:
    with np.testing.assert_raises(ValueError):
        MODULE.top4_mask(np.asarray([[0, 0, 1, 2]], np.uint8))


def test_rolling_features_are_causal() -> None:
    features = np.arange(18, dtype=np.float32).reshape(6, 3)
    features[0] = np.nan
    starts = np.asarray([0], np.int64)
    lengths = np.asarray([6], np.int32)
    before = MODULE.rolling_features(features, starts, lengths, 3)
    changed = features.copy()
    changed[5] = 1_000
    after = MODULE.rolling_features(changed, starts, lengths, 3)
    assert np.allclose(before[3:5], after[3:5])
    assert not np.allclose(before[5], after[5])
    assert np.isnan(before[:3]).all()


def test_lower_tail_empirical_risk_uses_midranks() -> None:
    reference = np.asarray([[0.0], [1.0], [1.0], [3.0]], np.float32)
    values = np.asarray([[0.0], [1.0], [2.0], [4.0]], np.float32)
    risk = MODULE.empirical_lower_tail_risk(values, reference)[:, 0]
    assert np.allclose(risk, [0.875, 0.5, 0.25, 0.0])


def test_sustained_score_requires_consecutive_values() -> None:
    score = np.asarray([np.nan, 0.9, 0.2, 0.8, 0.85], np.float32)
    sustained, maximum = MODULE.sustained_scores(
        score, np.asarray([0]), np.asarray([5]), persistence=2
    )
    assert np.isnan(sustained[:2]).all()
    assert np.allclose(sustained[2:], [0.2, 0.2, 0.8])
    assert np.allclose(maximum, [0.8])
    assert MODULE.first_alarm(sustained, 0.75) == 4


def test_conservative_threshold_respects_calibration_budget() -> None:
    values = np.linspace(0.0, 1.0, 101)
    threshold, achieved = MODULE.conservative_threshold(values, 0.05)
    assert achieved <= 0.05
    assert np.mean(values >= threshold) == achieved


def test_auc_handles_ties() -> None:
    assert MODULE.auc_higher_is_failure(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([False, True, False, True]),
    ) == 0.5
    assert MODULE.auc_higher_is_failure(
        np.asarray([0.0, 0.1, 0.9, 1.0]),
        np.asarray([False, False, True, True]),
    ) == 1.0
