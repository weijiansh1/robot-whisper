import numpy as np

import analyze_raw_expert_amplitude as raw


def test_pool_correlation_is_pool_local_and_supports_ranks():
    x = np.asarray([[1.0, 2.0, 3.0], [30.0, 20.0, 10.0]])
    y = np.asarray([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]])
    np.testing.assert_allclose(raw._pool_correlation(x, y), [1.0, -1.0])
    np.testing.assert_allclose(raw._pool_correlation(x, y, rank=True), [1.0, -1.0])


def test_pair_feature_block_keeps_token_differences_and_levels():
    value = np.asarray([[1.0, 5.0], [3.0, 1.0], [9.0, 9.0]])
    result = raw._pair_feature_block(value, np.asarray([0]), np.asarray([1]))
    np.testing.assert_allclose(result, [[2.0, 4.0, 2.0, 3.0]])


def test_seed_pair_masks_are_endpoint_disjoint():
    pair_i, pair_j = np.triu_indices(32, 1)
    train, test = raw._seed_pair_masks(32, pair_i, pair_j, fold=2)
    held_out = set(range(16, 24))
    assert train.sum() == 276
    assert test.sum() == 28
    assert all(i not in held_out and j not in held_out for i, j in zip(pair_i[train], pair_j[train]))
    assert all(i in held_out and j in held_out for i, j in zip(pair_i[test], pair_j[test]))


def test_hierarchical_bootstrap_constant_has_degenerate_interval():
    rng = np.random.default_rng(1)
    interval = raw._hierarchical_bootstrap_ci(np.ones((5, 16)), 100, rng)
    np.testing.assert_allclose(interval, [1.0, 1.0])


def test_nanmean_columns_preserves_empty_columns_as_nan():
    value = np.asarray([[1.0, np.nan], [3.0, np.nan]])
    result = raw._nanmean_columns(value)
    np.testing.assert_allclose(result[0], 2.0)
    assert np.isnan(result[1])
