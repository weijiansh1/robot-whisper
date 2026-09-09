from __future__ import annotations

import numpy as np

from audit_expert_activation_stop import (
    activation_views,
    first_stop,
    token_aggregation_proxies,
)


def test_first_stop_requires_consecutive_causal_rounds() -> None:
    condition = np.asarray(
        [
            [False, True, True, False, False, False, False, False, False, False],
            [False, True, False, True, True, False, False, False, False, False],
            [False, False, False, False, False, False, False, False, True, True],
        ]
    )
    np.testing.assert_array_equal(
        first_stop(condition, consecutive=2, min_round=3), [3, 5, 10]
    )


def test_full_round_is_fallback_not_a_claimed_saving() -> None:
    condition = np.zeros((2, 10), dtype=bool)
    condition[0, -1] = True
    np.testing.assert_array_equal(first_stop(condition, min_round=3), [10, 10])


def test_activation_views_distinguish_aggregate_and_layer_normalization() -> None:
    values = np.ones((1, 10, 2), dtype=np.float64)
    values[:, 0] = [1.0, 9.0]
    values[:, 1:] = [0.5, 9.0]
    views = activation_views(values, np.asarray([2, 5]))
    assert np.isclose(views["aggregate_then_norm"][0, 1], 0.95)
    assert np.isclose(views["layernorm_mean"][0, 1], 0.75)
    assert np.isclose(views["layernorm_max"][0, 1], 1.0)


def test_query_normalized_views_start_at_one() -> None:
    rng = np.random.default_rng(7)
    values = np.exp(rng.normal(size=(4, 10, 3)))
    views = activation_views(values, np.asarray([2, 5, 12]))
    for name, value in views.items():
        if name != "absolute_layer_mean":
            np.testing.assert_allclose(value[:, 0], 1.0)


def test_token_rms_recovery_and_maximum_bound() -> None:
    tokens = np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 20.0]])
    mean = tokens.mean(axis=1)
    sample_std = tokens.std(axis=1, ddof=1)
    proxies = token_aggregation_proxies(mean, sample_std, token_count=10)
    np.testing.assert_allclose(
        proxies["token_rms_recovered"], np.sqrt(np.mean(np.square(tokens), axis=1))
    )
    assert np.all(proxies["max_upper_bound_mean_plus_3std"] >= tokens.max(axis=1))
