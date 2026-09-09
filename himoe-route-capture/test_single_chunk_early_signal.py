import numpy as np

from analyze_single_chunk_early_signal import (
    HB_LAYERS,
    N_DENOISE,
    N_EXPERTS,
    N_TOKENS,
    _sequence_summary,
    conditional_pair_stats,
    hidden_features,
    router_features,
)


def test_sequence_summary_preserves_flow_direction():
    values = np.tile(np.arange(N_DENOISE, dtype=np.float32), (2, len(HB_LAYERS), 1))
    features = _sequence_summary(values).reshape(2, len(HB_LAYERS), 3)

    assert np.all(features[..., 2] > 0)


def test_router_and_hidden_features_are_finite_and_sample_local():
    rng = np.random.default_rng(4)
    raw = rng.gamma(
        1.0,
        size=(2, len(HB_LAYERS), N_DENOISE, N_TOKENS, N_EXPERTS),
    ).astype(np.float32)
    probabilities = raw / raw.sum(axis=-1, keepdims=True)
    ids = np.argsort(probabilities, axis=-1)[..., -4:][..., ::-1]
    hidden = rng.normal(
        size=(2, len(HB_LAYERS), N_DENOISE, N_TOKENS, 16)
    ).astype(np.float32)

    route_result = router_features(probabilities, ids)
    hidden_result = hidden_features(hidden)
    probabilities[1] = np.roll(probabilities[1], 1, axis=-1)
    ids[1] = np.argsort(probabilities[1], axis=-1)[..., -4:][..., ::-1]
    hidden[1] *= 3.0

    changed_route = router_features(probabilities, ids)
    changed_hidden = hidden_features(hidden)

    assert route_result.shape == (2, 320)
    assert hidden_result.shape == (2, 296)
    assert np.all(np.isfinite(route_result))
    assert np.all(np.isfinite(hidden_result))
    assert np.array_equal(route_result[0], changed_route[0])
    assert np.array_equal(hidden_result[0], changed_hidden[0])


def test_conditional_auc_counts_only_within_group_pairs():
    labels = np.array([0, 1, 0, 1, 0, 0], dtype=bool)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 10.0, 11.0])
    groups = np.array([0, 0, 1, 1, 2, 2])

    unique, wins, pairs = conditional_pair_stats(labels, scores, groups)

    assert np.array_equal(unique, [0, 1, 2])
    assert np.array_equal(pairs, [1, 1, 0])
    assert np.array_equal(wins, [1.0, 0.0, 0.0])
