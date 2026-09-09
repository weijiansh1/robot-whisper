import numpy as np

from analyze_intraquery_markov_transfer import (
    CANDIDATES,
    SEED_FOLDS,
    candidate_grid,
    coherent_seed_permutation,
)


def test_candidate_grid_makes_eight_disjoint_k8_pools():
    seeds = np.arange(1000, 1064, dtype=np.int64)

    grid, folds = candidate_grid(seeds)

    assert grid.shape == (SEED_FOLDS, CANDIDATES)
    assert np.array_equal(np.sort(grid.ravel()), np.arange(64))
    assert np.array_equal(seeds[grid[3]], np.arange(1003, 1064, 8))
    assert np.array_equal(np.bincount(folds), np.full(SEED_FOLDS, CANDIDATES))


def test_coherent_permutation_detects_known_top2_selector():
    labels = np.zeros((1, SEED_FOLDS, CANDIDATES), dtype=bool)
    labels[:, :, :2] = True
    weights = np.zeros_like(labels, dtype=np.float64)
    weights[:, :, :2] = 1.0

    result = coherent_seed_permutation(
        labels,
        weights,
        draws=4999,
        rng=np.random.default_rng(7),
    )

    assert result["observed"] == 0.75
    assert result["one_sided_p"] < 0.06
