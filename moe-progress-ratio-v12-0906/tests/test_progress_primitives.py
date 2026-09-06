import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(WORKSPACE / "moe-v7-0905/method"))

import progress_ratio as pr  # noqa: E402


def test_hellinger_matches_v7_bitwise():
    """本 bundle 自带的 hellinger 必须与 v7 的实现数值一致。"""
    import intrinsic_guard_monitor as v7

    rng = np.random.default_rng(20260906)
    left = rng.random((7, 32), dtype=np.float32)
    right = rng.random((7, 32), dtype=np.float32)
    np.testing.assert_allclose(
        pr.hellinger(left, right), v7.hellinger(left, right), rtol=0, atol=0
    )


def test_hellinger_is_a_metric_on_simple_cases():
    a = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    assert pr.hellinger(a, a) == pytest.approx(0.0, abs=1e-6)
    assert pr.hellinger(a, b) == pytest.approx(1.0, abs=1e-6)


def test_action_route_selects_final_flow_and_action_tokens():
    rng = np.random.default_rng(1)
    probs = rng.random((8, 10, 11, 32)).astype(np.float32)
    routes = pr.action_route(probs)
    assert routes.shape == (8, 10, 32)
    np.testing.assert_allclose(routes.sum(axis=-1), 1.0, rtol=1e-6)
    # token 0 是 state，必须被排除
    expected = probs[:, 9, 1:11, :]
    expected = expected / expected.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(routes, expected, rtol=1e-5)


def test_action_route_rejects_wrong_shape():
    with pytest.raises(ValueError, match="router probability"):
        pr.action_route(np.zeros((8, 10, 11, 16), dtype=np.float32))


def test_route_distance_reduces_tokens():
    rng = np.random.default_rng(2)
    left = rng.random((8, 10, 32)).astype(np.float32)
    right = rng.random((8, 10, 32)).astype(np.float32)
    assert pr.route_distance(left, right).shape == (8,)
