import numpy as np

from analyze_moe_value_signal import (
    EXPERT_SIZE,
    Dataset,
    build_ridge_operator,
    make_index_grid,
    metric_features,
    normalize_router_probabilities,
    pool_standardize,
)


def test_router_probability_normalization_is_per_vector():
    values = np.arange(1, 1 + 2 * 3 * 2 * 4 * 32, dtype=np.float64).reshape(
        2, 3, 2, 4, 32
    )
    normalized, bounds = normalize_router_probabilities(values)
    assert np.allclose(normalized.sum(axis=-1), 1.0)
    assert bounds == (float(values.sum(axis=-1).min()), float(values.sum(axis=-1).max()))


def test_metric_features_preserve_sample_layer_metric_order():
    metrics = np.arange(2 * 10 * 3 * 6, dtype=np.float64).reshape(2, 10, 3, 6)
    names = (
        "unused0",
        "unused1",
        *EXPERT_SIZE,
    )
    data = Dataset(
        scenes=np.array([0, 1]),
        seeds=np.array([10, 11]),
        seed_folds=np.array([0, 0]),
        seed_positions=np.array([0, 0]),
        success=np.array([0, 1]),
        noise=np.empty((2, 10, 24)),
        metrics=metrics,
        metric_names=names,
        layer_numbers=np.array([2, 5, 12]),
        routes=np.empty((2, 1, 1, 1, 32)),
        index_grid=np.empty((0, 4, 0), dtype=np.int64),
        route_sum_before_normalization=(1.0, 1.0),
    )
    actual = metric_features(data, 3, EXPERT_SIZE)
    expected = metrics[:, 3, :, 2:6].reshape(2, -1)
    assert np.array_equal(actual, expected)


def test_pool_standardization_is_independent_per_scene_and_fold():
    scenes = np.repeat(np.arange(3), 8)
    folds = np.tile(np.repeat(np.arange(2), 4), 3)
    values = np.column_stack([np.arange(24), np.arange(24) ** 2]).astype(np.float64)
    standardized = pool_standardize(values, scenes, folds)
    for scene in np.unique(scenes):
        for fold in np.unique(folds):
            index = (scenes == scene) & (folds == fold)
            assert np.allclose(standardized[index].mean(axis=0), 0.0, atol=1e-12)
            assert np.allclose(standardized[index].std(axis=0), 1.0, atol=1e-12)


def test_ridge_operator_excludes_held_scene_and_seed_fold():
    rng = np.random.default_rng(7)
    scenes = np.repeat(np.arange(4), 4 * 3)
    folds = np.tile(np.repeat(np.arange(4), 3), 4)
    features = rng.normal(size=(len(scenes), 5))
    operator = build_ridge_operator([(features, None)], scenes, folds, alpha=10.0)
    for row in range(len(scenes)):
        forbidden = (scenes == scenes[row]) | (folds == folds[row])
        assert np.array_equal(operator[row, forbidden], np.zeros(forbidden.sum()))


def test_index_grid_covers_each_candidate_once():
    scenes = np.repeat(np.arange(2), 4 * 3)
    folds = np.tile(np.repeat(np.arange(4), 3), 2)
    positions = np.tile(np.arange(3), 2 * 4)
    grid = make_index_grid(scenes, folds, positions)
    assert grid.shape == (2, 4, 3)
    assert np.array_equal(np.sort(grid.ravel()), np.arange(len(scenes)))
