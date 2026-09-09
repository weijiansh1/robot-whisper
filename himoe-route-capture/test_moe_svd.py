from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from analyze_moe_svd import (
    correlations,
    fit_svd,
    normalize_probabilities,
    sequence_arrow,
    two_way_residual,
)


def test_probability_normalization_is_exact_and_reports_raw_bounds():
    raw = np.zeros((2, 32), dtype=np.float32)
    raw[0, :2] = [1.0, 3.0]
    raw[1, :2] = [2.0, 3.0]
    normalized, bounds = normalize_probabilities(raw)
    np.testing.assert_allclose(normalized.sum(axis=-1), 1.0, atol=1e-7)
    assert bounds == (4.0, 5.0)


def test_svd_reports_low_rank_variance_against_full_frobenius_energy():
    rng = np.random.default_rng(3)
    left = rng.normal(size=(40, 2))
    right = rng.normal(size=(2, 12))
    matrix = left @ right
    fit = fit_svd(matrix, components=6, seed=4)
    assert float(fit["explained"][:2].sum()) > 1.0 - 1e-6
    assert float(fit["explained"][2:].sum()) < 1e-10


def test_two_way_residual_removes_both_balanced_main_effects():
    first = np.repeat(np.arange(3), 4)
    second = np.tile(np.arange(4), 3)
    values = 10.0 * first + 3.0 * second + first * second
    residual = two_way_residual(values, first, second)
    for labels in (first, second):
        means = [residual[labels == value].mean() for value in np.unique(labels)]
        np.testing.assert_allclose(means, 0.0, atol=1e-12)


def test_sequence_arrow_uses_order_without_outcome_labels():
    episodes = np.repeat(np.arange(4), 6)
    progress = np.tile(np.arange(6), 4)
    scores = np.stack(
        [progress, -2.0 * progress, np.sin(progress)], axis=1
    ).astype(np.float64)
    scores += np.repeat(np.arange(4), 6)[:, None] * np.asarray([3.0, 1.0, -2.0])
    arrow = sequence_arrow(scores, episodes)
    temporal = scores @ arrow
    assert correlations(temporal, progress)[0] > 0.98
