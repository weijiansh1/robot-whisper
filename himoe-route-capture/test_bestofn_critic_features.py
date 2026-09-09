import numpy as np

from bestofn_critic_features import (
    as_route_features,
    hidden_candidate_features,
    hidden_state_features,
    route_candidate_features,
    route_state_features,
)


def _routes(seed=7):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(3, 2, 6, 5, 4)).astype(np.float32)
    logits -= logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=-1, keepdims=True)


def test_route_features_are_candidate_and_snapshot_aligned():
    routes = _routes()
    candidate = route_candidate_features(routes)
    state = route_state_features(routes)
    assert candidate.shape[0] == 3
    assert state.ndim == 1
    assert candidate.shape[1] > state.shape[0]
    assert np.all(np.isfinite(candidate))
    assert np.all(np.isfinite(state))


def test_flow_order_and_flow_subset_change_dynamic_features():
    routes = _routes()
    original = route_candidate_features(routes)
    reversed_flow = route_candidate_features(routes, list(reversed(range(6))))
    final_only = route_candidate_features(routes, [5])
    assert original.shape == reversed_flow.shape == final_only.shape
    assert not np.allclose(original, reversed_flow)
    assert not np.allclose(original, final_only)


def test_hidden_aggregates_retain_candidate_layer_and_width_axes():
    rng = np.random.default_rng(11)
    hidden = rng.normal(size=(3, 2, 6, 5, 9)).astype(np.float32)
    assert hidden_candidate_features(hidden).shape == (3, 6, 2, 9)
    assert hidden_state_features(hidden).shape == (6, 2, 9)


def test_as_negative_control_is_finite():
    probs = np.asarray(
        [
            [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]],
            [[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]],
        ],
        dtype=np.float32,
    )
    features = as_route_features(probs)
    assert features.shape == (2, 10)
    assert np.all(np.isfinite(features))
