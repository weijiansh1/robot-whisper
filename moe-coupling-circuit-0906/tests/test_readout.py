"""Unit tests for the behavioural readouts and the clustered bootstrap."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from readout import (  # noqa: E402
    chunk_distance,
    cluster_bootstrap,
    cumulative_translation,
    direction_cosine,
    gripper_mean,
    tracking_cosine,
    translation_norm,
)


def chunk(dx=0.0, dy=0.0, dz=0.0, gripper=0.0, steps=10):
    out = np.zeros((steps, 7), np.float32)
    out[:, 0] = dx / steps
    out[:, 1] = dy / steps
    out[:, 2] = dz / steps
    out[:, 6] = gripper
    return out


def test_cumulative_translation_sums_the_chunk():
    assert np.allclose(cumulative_translation(chunk(dx=0.3, dy=-0.1, dz=0.2)), [0.3, -0.1, 0.2])


def test_cumulative_translation_rejects_a_wrong_shape():
    with pytest.raises(ValueError, match="steps, 7"):
        cumulative_translation(np.zeros((10, 6), np.float32))


def test_direction_cosine_is_one_when_aimed_at_the_target():
    value = direction_cosine(chunk(dx=0.1), eef_xyz=[0, 0, 0.9], target_xyz=[0.5, 0, 0.9])
    assert np.isclose(value, 1.0)


def test_direction_cosine_is_minus_one_when_aimed_away():
    value = direction_cosine(chunk(dx=-0.1), eef_xyz=[0, 0, 0.9], target_xyz=[0.5, 0, 0.9])
    assert np.isclose(value, -1.0)


def test_direction_cosine_ignores_the_vertical_component():
    """A pure descent, which every condition shares, must not create a bearing."""
    straight_down = direction_cosine(
        chunk(dx=0.1, dz=-5.0), eef_xyz=[0, 0, 0.9], target_xyz=[0.5, 0, 0.2]
    )
    no_descent = direction_cosine(
        chunk(dx=0.1), eef_xyz=[0, 0, 0.9], target_xyz=[0.5, 0, 0.9]
    )
    assert np.isclose(straight_down, no_descent)


def test_direction_cosine_is_nan_rather_than_zero_when_undefined():
    assert np.isnan(direction_cosine(chunk(), eef_xyz=[0, 0, 0.9], target_xyz=[0.5, 0, 0.9]))
    assert np.isnan(direction_cosine(chunk(dx=0.1), eef_xyz=[0, 0, 0.9], target_xyz=[0, 0, 0.5]))


def test_tracking_cosine_follows_the_shift():
    held = chunk(dx=0.20)
    edited = chunk(dx=0.20, dy=0.05)
    assert np.isclose(tracking_cosine(edited, held, [0.0, 1.0, 0.0]), 1.0)
    assert np.isclose(tracking_cosine(edited, held, [0.0, -1.0, 0.0]), -1.0)


def test_tracking_cosine_is_zero_for_a_change_orthogonal_to_the_shift():
    held = chunk(dx=0.20)
    edited = chunk(dx=0.25)
    assert np.isclose(tracking_cosine(edited, held, [0.0, 1.0, 0.0]), 0.0, atol=1e-12)


def test_tracking_cosine_is_nan_when_the_command_did_not_change():
    held = chunk(dx=0.20)
    assert np.isnan(tracking_cosine(held, held, [0.0, 1.0, 0.0]))


def test_gripper_and_magnitude_readouts():
    assert np.isclose(gripper_mean(chunk(gripper=-0.8)), -0.8)
    assert np.isclose(translation_norm(chunk(dx=0.3, dy=0.4)), 0.5)


def test_magnitude_and_direction_are_independent_readouts():
    """Bigger and better-aimed must be separable; that is why both are reported."""
    aimed_small = chunk(dx=0.05)
    aimed_large = chunk(dx=0.50)
    wrong_large = chunk(dx=-0.50)
    eef, target = [0, 0, 0.9], [0.5, 0, 0.9]
    assert np.isclose(direction_cosine(aimed_small, eef, target), direction_cosine(aimed_large, eef, target))
    assert translation_norm(aimed_large) > translation_norm(aimed_small)
    assert np.isclose(translation_norm(wrong_large), translation_norm(aimed_large))
    assert direction_cosine(wrong_large, eef, target) < 0


def test_chunk_distance_is_symmetric_and_zero_on_identity():
    left, right = chunk(dx=0.1), chunk(dx=0.2)
    assert chunk_distance(left, left) == 0.0
    assert np.isclose(chunk_distance(left, right), chunk_distance(right, left))


def test_bootstrap_interval_brackets_a_clear_mean():
    values = np.full(40, 0.5)
    result = cluster_bootstrap(values, np.arange(40), resamples=500)
    assert result["n"] == 40 and result["n_clusters"] == 40
    assert np.isclose(result["point"], 0.5)
    assert result["low"] <= 0.5 <= result["high"]


def test_bootstrap_resamples_clusters_not_rows():
    """Twenty identical rows in one episode must not look like twenty episodes."""
    values = np.concatenate([np.full(20, 1.0), np.full(20, -1.0)])
    rows = cluster_bootstrap(values, np.arange(40), resamples=2000)
    clustered = cluster_bootstrap(values, np.repeat([0, 1], 20), resamples=2000)
    assert np.isclose(rows["point"], clustered["point"])
    width_rows = rows["high"] - rows["low"]
    width_clustered = clustered["high"] - clustered["low"]
    assert width_clustered > width_rows


def test_bootstrap_reports_dropped_non_finite_values():
    values = [1.0, 2.0, float("nan"), 3.0]
    result = cluster_bootstrap(values, [0, 1, 2, 3], resamples=200)
    assert result["dropped_non_finite"] == 1
    assert result["n"] == 3
    assert np.isclose(result["point"], 2.0)


def test_bootstrap_on_an_all_nan_column_returns_nan_not_an_exception():
    result = cluster_bootstrap([float("nan")] * 4, [0, 1, 2, 3], resamples=100)
    assert result["n"] == 0 and result["dropped_non_finite"] == 4
    assert np.isnan(result["point"])


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    values = np.linspace(-1, 1, 30)
    clusters = np.repeat(np.arange(10), 3)
    first = cluster_bootstrap(values, clusters, resamples=300, seed=7)
    second = cluster_bootstrap(values, clusters, resamples=300, seed=7)
    assert first == second


def test_bootstrap_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        cluster_bootstrap([1.0, 2.0], [0], resamples=10)
