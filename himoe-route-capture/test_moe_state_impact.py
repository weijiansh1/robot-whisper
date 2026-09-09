import numpy as np

from analyze_moe_state_impact import (
    LAYERS,
    N_DENOISE,
    N_TOKENS,
    WIDTH,
    conditional_scene_stats,
    contribution_commitment,
    project_token_directions,
    seed_folds,
)


def test_projection_preserves_sample_flow_shape_and_is_finite():
    rng = np.random.default_rng(2)
    values = rng.normal(
        size=(2, N_DENOISE, len(LAYERS), N_TOKENS, WIDTH)
    ).astype(np.float16)

    projected = project_token_directions(values, seed=4)

    assert projected.shape[:2] == (2, N_DENOISE)
    assert np.all(np.isfinite(projected))


def test_commitment_is_one_at_final_flow_position():
    rng = np.random.default_rng(3)
    values = rng.normal(
        size=(3, N_DENOISE, len(LAYERS), N_TOKENS, 16)
    ).astype(np.float32)

    commitment = contribution_commitment(values)

    assert np.allclose(commitment[:, -1], 1.0, atol=1e-6)


def test_seed_folds_hold_out_whole_seed_columns():
    seeds = np.tile(np.arange(1000, 1032), 3)
    folds = seed_folds(seeds)

    assert np.array_equal(folds[:32], np.repeat(np.arange(4), 8))
    for seed in np.unique(seeds):
        assert len(np.unique(folds[seeds == seed])) == 1


def test_conditional_stats_do_not_compare_across_strata():
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 100.0, -100.0])
    scenes = np.array([0, 0, 0, 0, 1, 1])
    strata = np.array([0, 0, 1, 1, 2, 3])

    wins, pairs = conditional_scene_stats(labels, scores, scenes, strata)

    assert np.array_equal(pairs, [2, 0])
    assert np.array_equal(wins, [1.0, 0.0])
