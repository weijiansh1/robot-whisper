"""Causality and missing-data checks for the new flow and fusion logic."""

import numpy as np

from fusion import METHODS, flow_speed, raw_v8_features, v8_heads, combine_streams


def test_v8_padding_stays_missing():
    speed = np.full((2, 20, 8, 9), np.nan, np.float32)
    speed[0, :9] = .1
    speed[1, :15] = .2
    raw = raw_v8_features(speed)
    heads = v8_heads(raw)
    assert not np.isfinite(raw[0, 9:]).any()
    assert not np.isfinite(raw[1, 15:]).any()
    assert not np.isfinite(heads[0, 9:]).any()
    assert not np.isfinite(heads[1, 15:]).any()
    assert not np.isfinite(heads[:, :6]).any()


def test_v8_prefix_and_suffix_invariance():
    rng = np.random.default_rng(17)
    speed = rng.uniform(.01, .2, (3, 25, 8, 9)).astype(np.float32)
    raw = raw_v8_features(speed)
    full = v8_heads(raw)
    for stop in range(7, 26):
        np.testing.assert_array_equal(v8_heads(raw[:, :stop]), full[:, :stop])
    changed = raw.copy()
    changed[:, 15:] *= 100
    np.testing.assert_array_equal(v8_heads(changed)[:, :15], full[:, :15])


def test_flow_speed_uses_action_tokens_and_flow_axis():
    rng = np.random.default_rng(19)
    raw = rng.uniform(.01, 1, (2, 8, 10, 11, 32)).astype(np.float32)
    normalized = raw / raw.sum(-1, keepdims=True)
    expected = np.empty((2, 8, 9), np.float32)
    for query in range(2):
        for layer in range(8):
            for step in range(9):
                delta = np.sqrt(normalized[query, layer, step+1, 1:]) - np.sqrt(normalized[query, layer, step, 1:])
                expected[query, layer, step] = (np.linalg.norm(delta, axis=-1) / np.sqrt(2)).mean()
    actual = flow_speed(raw)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    raw[..., 0, :] *= 100
    np.testing.assert_array_equal(flow_speed(raw), actual)


def test_consensus_requires_current_agreement_and_delta_is_causal():
    k = np.full((1, 14), np.nan, np.float32)
    k[0, 7:] = [3, 0, 0, 4, 4, 4, 0]
    g = np.zeros_like(k)
    g[0, 8] = 3
    g[0, 11] = 3
    base = dict(knn10=k, knn12_v8=k, v7_guard=g, v8_guard=g, v82_guard=g)
    profile = dict(fusion_center=np.zeros(3), fusion_scale=np.ones(3))
    scores = combine_streams(base, profile)
    consensus = scores[METHODS.index('knn_and_v8'), 0]
    assert np.flatnonzero(consensus > 2).tolist() == [11]
    delta = scores[METHODS.index('knn10_delta_q7'), 0]
    assert delta[7] == 0 and delta[10] == 1
    persistent = scores[METHODS.index('knn10_persist3'), 0]
    assert np.flatnonzero(persistent > 2).tolist() == [12]
