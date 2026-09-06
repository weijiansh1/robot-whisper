import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(BUNDLE / "experiments"))

import build_progress_cache as bpc  # noqa: E402
import progress_ratio as pr  # noqa: E402


def test_episode_lag_distances_matches_reference_mobility():
    """lag 1 必须复现 v4 缓存里 mobility 的定义。"""
    rng = np.random.default_rng(21)
    probs = rng.random((9, 8, 10, 11, 32)).astype(np.float32)
    routes = pr.action_route(probs)
    computed = bpc.episode_lag_distances(probs, pr.LAGS)
    expected = pr.route_distance(routes[1:], routes[:-1])
    np.testing.assert_allclose(computed[1:, 0], expected, rtol=0, atol=2e-5)


def test_episode_lag_distances_pads_short_episodes():
    rng = np.random.default_rng(22)
    probs = rng.random((3, 8, 10, 11, 32)).astype(np.float32)
    computed = bpc.episode_lag_distances(probs, pr.LAGS)
    assert computed.shape == (3, len(pr.LAGS), 8)
    assert np.isnan(computed[:, pr.LAGS.index(4)]).all()


def test_anchor_check_rejects_mismatched_mobility():
    lag_distance = np.zeros((2, 5, len(pr.LAGS), 8), dtype=np.float32)
    mobility = np.ones((2, 5, 8), dtype=np.float32)
    valid = np.ones((2, 5), dtype=bool)
    with pytest.raises(ValueError, match="lag-1 distances disagree"):
        bpc.assert_mobility_anchor(lag_distance, mobility, valid)


def test_anchor_check_accepts_matching_mobility():
    lag_distance = np.zeros((2, 5, len(pr.LAGS), 8), dtype=np.float32)
    mobility = np.zeros((2, 5, 8), dtype=np.float32)
    valid = np.ones((2, 5), dtype=bool)
    valid[:, 0] = False
    bpc.assert_mobility_anchor(lag_distance, mobility, valid)
