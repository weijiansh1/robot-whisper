import numpy as np
from scipy import sparse

from analyze_moe_rollout_trend import (
    N_DENOISE,
    N_GROUPS,
    WIDTH,
    conditional_scene_stats,
    _row_grid,
    project_group_vectors,
    seed_folds,
    vector_dynamics,
)


def test_projection_preserves_axes_and_normalizes():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, N_DENOISE, N_GROUPS, WIDTH)).astype(np.float32)
    bucket = np.arange(WIDTH) % 32
    projection = sparse.csr_matrix(
        (np.ones(WIDTH), (np.arange(WIDTH), bucket)), shape=(WIDTH, 32)
    )
    projected = project_group_vectors(values, projection)
    assert projected.shape == (3, N_DENOISE, N_GROUPS, 32)
    assert np.allclose(np.linalg.norm(projected.astype(np.float32), axis=-1), 1, atol=2e-3)


def test_vector_dynamics_has_final_commitment_one():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(4, N_DENOISE, N_GROUPS, 17)).astype(np.float32)
    commitment, adjacent = vector_dynamics(values)
    assert commitment.shape == (4, N_DENOISE, N_GROUPS)
    assert adjacent.shape == (4, N_DENOISE - 1, N_GROUPS)
    assert np.allclose(commitment[:, -1], 1.0, atol=1e-3)


def test_seed_folds_hold_out_complete_seed_columns():
    seeds = np.tile(np.arange(32), 16)
    folds = seed_folds(seeds)
    assert np.array_equal(folds[:32], np.repeat(np.arange(4), 8))
    for seed in np.unique(seeds):
        assert len(np.unique(folds[seeds == seed])) == 1


def test_conditional_auc_never_compares_strata():
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    scores = np.array([0.1, 0.9, 0.8, 0.2, 100.0, -100.0])
    scenes = np.array([0, 0, 0, 0, 1, 1])
    strata = np.array([0, 0, 1, 1, 2, 3])
    wins, pairs = conditional_scene_stats(labels, scores, scenes, strata)
    assert np.array_equal(pairs, [2, 0])
    assert np.array_equal(wins, [1.0, 0.0])


def test_row_grid_maps_multiple_queries_to_each_episode():
    summaries = [
        {
            "episode_index": episode,
            "init_state_id": episode,
            "flow_noise_seed": 10 + episode,
            "success": bool(episode),
            "inference_calls": count,
        }
        for episode, count in enumerate((4, 5))
    ]
    episode_axis = np.repeat(np.arange(2), (4, 5))
    episodes, _, _, _, grid = _row_grid(summaries, 3, episode_axis)
    assert np.array_equal(episode_axis[grid], episodes[:, None] + np.zeros((2, 3), int))
    assert np.array_equal(grid, [[0, 1, 2], [4, 5, 6]])
