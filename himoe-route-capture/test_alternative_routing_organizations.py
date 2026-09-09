from __future__ import annotations

import math

import numpy as np

from analyze_alternative_routing_organizations import (
    _signature_one,
    _topology_one,
    geometry_by_layer,
    json_safe,
    lag_spectrum_features,
    pair_layout,
    peer_rank_features,
    rank_within_groups,
)


def test_pair_layout_is_upper_triangle_in_canonical_order() -> None:
    left, right, lag = pair_layout(4)
    assert list(zip(left, right)) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    np.testing.assert_array_equal(lag, right - left)


def test_peer_rank_quotients_group_shared_route_path() -> None:
    rng = np.random.default_rng(1)
    geometry = rng.normal(size=(64, 10, 320))
    tasks = np.repeat(np.asarray(["a", "b"]), 32)
    initial = np.zeros(64, dtype=np.int32)
    reference, _audit = peer_rank_features(geometry, tasks, initial)
    shifted = geometry.copy()
    shifted[:32] += rng.normal(size=(1, 10, 320))
    shifted[32:] += rng.normal(size=(1, 10, 320))
    candidate, _audit = peer_rank_features(shifted, tasks, initial)
    np.testing.assert_allclose(candidate, reference, atol=1e-6)


def test_rank_within_groups_handles_ties() -> None:
    values = np.asarray([[1.0], [1.0], [3.0], [4.0]])
    ranked = rank_within_groups(values, np.zeros(4, dtype=np.int32))
    np.testing.assert_allclose(ranked[:, 0], [0.25, 0.25, 0.625, 0.875])


def test_lag_spectrum_quotients_per_episode_channel_scale() -> None:
    rng = np.random.default_rng(2)
    pairs = rng.uniform(0.02, 0.8, size=(7, 45, 24))
    reference, _audit = lag_spectrum_features(pairs.reshape(7, -1))
    scale = rng.uniform(0.2, 4.0, size=(7, 1, 24))
    candidate, _audit = lag_spectrum_features((pairs * scale).reshape(7, -1))
    np.testing.assert_allclose(candidate, reference, rtol=2e-6, atol=2e-6)


def test_signature_is_translation_and_positive_scale_invariant() -> None:
    rng = np.random.default_rng(3)
    path = rng.normal(size=(10, 8)).cumsum(axis=0)
    reference = _signature_one(path)
    candidate = _signature_one(7.5 * path + rng.normal(size=(1, 8)))
    np.testing.assert_allclose(candidate, reference, rtol=1e-10, atol=1e-10)


def test_topology_is_positive_scale_invariant() -> None:
    points = np.arange(10, dtype=np.float64)[:, None]
    distance = np.abs(points - points.T)
    reference = _topology_one(distance)
    candidate = _topology_one(2.75 * distance)
    np.testing.assert_allclose(candidate, reference, rtol=1e-7, atol=1e-7)


def test_geometry_layer_parser_preserves_all_channels() -> None:
    values = np.arange(2 * 10 * 320, dtype=np.float64).reshape(2, 10, 320)
    layered = geometry_by_layer(values)
    assert layered.shape == (2, 10, 8, 40)
    reconstructed = []
    reconstructed.extend(layered[..., offset : offset + 3].reshape(2, 10, -1) for offset in range(0, 24, 3))
    reconstructed.extend(layered[..., offset : offset + 4].reshape(2, 10, -1) for offset in range(24, 40, 4))
    np.testing.assert_array_equal(np.concatenate(reconstructed, axis=-1), values)


def test_json_safe_replaces_nonfinite_scalars() -> None:
    result = json_safe({"a": [float("nan"), np.float64(math.inf)], "b": np.int64(4)})
    assert result == {"a": [None, None], "b": 4}
