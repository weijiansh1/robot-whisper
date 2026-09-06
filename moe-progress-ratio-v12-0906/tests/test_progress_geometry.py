import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402


def _random_walk_routes(n_query: int, seed: int) -> np.ndarray:
    """[Q, 8, 10, 32] 的一条平滑路由轨迹。"""
    rng = np.random.default_rng(seed)
    base = rng.random((8, 10, 32)).astype(np.float32)
    routes = np.empty((n_query, 8, 10, 32), dtype=np.float32)
    current = base
    for q in range(n_query):
        current = np.abs(current + 0.05 * rng.standard_normal(current.shape).astype(np.float32))
        routes[q] = current / current.sum(axis=-1, keepdims=True)
    return routes


def test_lag_distances_lag1_equals_adjacent_hellinger():
    routes = _random_walk_routes(12, seed=3)
    lags = pr.lag_distances(routes, (1, 4))
    expected = pr.route_distance(routes[1:], routes[:-1])
    np.testing.assert_allclose(lags[1:, 0], expected, rtol=1e-6)
    assert np.isnan(lags[0, 0]).all()
    assert np.isnan(lags[:4, 1]).all()


def test_displacement_never_exceeds_path_length():
    """三角不等式：D <= L。这条一旦破了，R 就不在 [0, 1] 里。"""
    routes = _random_walk_routes(30, seed=4)
    lags = pr.lag_distances(routes, pr.LAGS)
    adjacent = lags[None, :, 0, :]
    for window in pr.W_GRID:
        length = pr.path_length(adjacent, window)
        displacement = lags[None, :, pr.LAGS.index(window), :]
        both = np.isfinite(length) & np.isfinite(displacement)
        assert both.any()
        assert np.all(displacement[both] <= length[both] + 1e-5)


def test_path_length_validity_starts_at_window():
    routes = _random_walk_routes(20, seed=5)
    adjacent = pr.lag_distances(routes, (1,))[None, :, 0, :]
    length = pr.path_length(adjacent, 4)
    assert np.isnan(length[0, :4]).all()
    assert np.isfinite(length[0, 4:]).all()


def test_progress_ratio_within_unit_interval():
    routes = _random_walk_routes(30, seed=6)
    lags = pr.lag_distances(routes, pr.LAGS)
    adjacent = lags[None, :, 0, :]
    length = pr.path_length(adjacent, 4)
    displacement = lags[None, :, pr.LAGS.index(4), :]
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-4)
    finite = np.isfinite(ratio)
    assert finite.any()
    assert np.all(ratio[finite] >= 0.0)
    assert np.all(ratio[finite] <= 1.0)


def test_frozen_prefix_gives_zero_ratio_not_nan():
    """L -> 0 时 D <= L 也 -> 0；这是 0/0，正确的极限值是 0（毫无净运动）。"""
    displacement = np.array([[[1e-9]]], dtype=np.float32)
    length = np.array([[[2e-9]]], dtype=np.float32)
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-6)
    assert ratio[0, 0, 0] == 0.0


def test_straight_line_gives_ratio_one():
    """单调朝一个方向走满窗口时 D == L，R == 1。"""
    displacement = np.array([[[0.4]]], dtype=np.float32)
    length = np.array([[[0.4]]], dtype=np.float32)
    ratio = pr.progress_ratio(displacement, length, eps_length=1e-6)
    assert ratio[0, 0, 0] == pytest.approx(1.0)


def test_group_ratio_takes_median_within_group():
    ratio = np.full((1, 1, 8), np.nan, dtype=np.float32)
    ratio[0, 0, :4] = [0.1, 0.2, 0.3, 0.4]
    ratio[0, 0, 4:] = [0.6, 0.7, 0.8, 0.9]
    assert pr.group_ratio(ratio, "front")[0, 0] == pytest.approx(0.25)
    assert pr.group_ratio(ratio, "back")[0, 0] == pytest.approx(0.75)
    assert pr.group_ratio(ratio, "all")[0, 0] == pytest.approx(0.5)


def test_group_ratio_rejects_unknown_group():
    with pytest.raises(ValueError, match="unknown layer group"):
        pr.group_ratio(np.zeros((1, 1, 8), dtype=np.float32), "middle")
