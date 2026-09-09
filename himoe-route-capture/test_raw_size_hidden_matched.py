import numpy as np

import analyze_raw_size_hidden_matched as audit


def test_pair_rms_uses_every_action_token_and_hidden_coordinate():
    value = np.zeros((1, 3, 2, 2), dtype=np.float64)
    value[0, 1, 0, 0] = 2.0
    value[0, 2] = 2.0
    pair_i, pair_j = audit._pair_indices(3)
    result = audit._pair_rms_pools(value, pair_i, pair_j)
    np.testing.assert_allclose(result[0], [1.0, 2.0, np.sqrt(3.0)])


def test_hidden_matching_is_fixed_before_action_tail_selection():
    hidden = np.asarray([[[9.0, 1.0, 2.0, 3.0, 4.0, 8.0]]])
    action = np.asarray([[[7.0, 0.1, 0.9, 0.2, 0.8, 6.0]]])
    nearest = audit._hidden_nearest_pairs(hidden, nearest=4)
    stable, divergent = audit._action_tail_pairs(action, nearest, tail=1)
    np.testing.assert_array_equal(nearest, [[[1, 2, 3, 4]]])
    np.testing.assert_array_equal(stable, [[[1]]])
    np.testing.assert_array_equal(divergent, [[[2]]])


def test_four_pair_scores_separate_level_from_absolute_difference():
    raw = np.asarray([[[2.0, 6.0, 10.0]]])
    mass = np.asarray([[[1.0, 4.0, 9.0]]])
    pair_i, pair_j = audit._pair_indices(3)
    result = audit._pair_score_tensor(raw, mass, pair_i, pair_j)
    np.testing.assert_allclose(result[0, 0, 0], [4.0, 6.0, 8.0])
    np.testing.assert_allclose(result[1, 0, 0], [2.5, 5.0, 6.5])
    np.testing.assert_allclose(result[2, 0, 0], [4.0, 8.0, 4.0])
    np.testing.assert_allclose(result[3, 0, 0], [3.0, 8.0, 5.0])


def test_selected_pair_auc_gives_half_credit_for_ties():
    scores = np.asarray([[[[0.0, 1.0, 2.0, 1.0]]]])
    stable = np.asarray([[[0, 1]]])
    divergent = np.asarray([[[2, 3]]])
    result = audit._selected_pair_auc(scores, stable, divergent)
    np.testing.assert_allclose(result, [[[0.875]]])


def test_action_mean_subtraction_cannot_change_pair_distance():
    actions = np.asarray(
        [
            [
                [[1.0, 4.0], [2.0, 8.0]],
                [[3.0, 2.0], [5.0, 7.0]],
                [[-1.0, 6.0], [4.0, 9.0]],
            ]
        ]
    )
    pair_i, pair_j = audit._pair_indices(3)
    original = audit._pair_rms_pools(actions, pair_i, pair_j)
    shifted = audit._pair_rms_pools(actions - np.asarray([20.0, -30.0]), pair_i, pair_j)
    np.testing.assert_allclose(original, shifted)


def test_common_seed_permutation_is_reused_and_max_t_is_reproducible():
    pair_i, pair_j = audit._pair_indices(4)
    raw = np.asarray([[[0.0, 1.0, 3.0, 7.0]]])
    mass = np.asarray([[[0.0, 2.0, 4.0, 8.0]]])
    scores = np.tile(audit._pair_score_tensor(raw, mass, pair_i, pair_j), (1, 2, 2, 1))
    candidate_action = np.asarray([0.0, 1.0, 4.0, 10.0])
    upper = np.abs(candidate_action[pair_i] - candidate_action[pair_j])
    upper = np.broadcast_to(upper, (2, 2, len(pair_i)))
    matrix = audit._upper_to_symmetric_matrix(upper, pair_i, pair_j, 4)
    nearest = np.broadcast_to(np.asarray([0, 1, 2, 3]), (2, 2, 4))

    first, first_arrays = audit._common_seed_max_t(
        scores, matrix, nearest, pair_i, pair_j, 1, 29, 17
    )
    second, second_arrays = audit._common_seed_max_t(
        scores, matrix, nearest, pair_i, pair_j, 1, 29, 17
    )
    np.testing.assert_array_equal(
        first_arrays["seed_permutations"], second_arrays["seed_permutations"]
    )
    np.testing.assert_allclose(
        first_arrays["null_macro_auc"], second_arrays["null_macro_auc"]
    )

    order = first_arrays["seed_permutations"][0]
    permuted_action = matrix[..., order[pair_i], order[pair_j]]
    stable, divergent = audit._action_tail_pairs(permuted_action, nearest, 1)
    expected = audit._selected_pair_auc(scores, stable, divergent).mean(axis=(1, 2))
    np.testing.assert_allclose(first_arrays["null_macro_auc"][0], expected)
    for row in first["metrics"].values():
        assert (
            row["two_sided_maxT_p_over_four_scores"]
            >= row["two_sided_common_seed_permutation_p"]
        )


def test_archive_loader_recomputes_both_raw_size_definitions(tmp_path):
    grid = (5, 16, 32)
    slot = np.broadcast_to(np.asarray([1.0, 2.0, 3.0, 4.0]), (*grid, 1, 10, 4)).copy()
    weight = np.full(slot.shape, 0.25)
    raw_mean = np.full((*grid, 1), 2.5)
    mass = np.full((*grid, 1), 2.5)
    path = tmp_path / "archive.npz"
    np.savez_compressed(
        path,
        task_names=np.asarray([f"task-{index}" for index in range(5)]),
        scene_ids=np.broadcast_to(np.arange(16), (5, 16)),
        seed_ids=np.broadcast_to(np.arange(32), (5, 32)),
        selected_expert_weight=weight,
        selected_expert_raw_rms=slot,
        raw_mean=raw_mean,
        weighted_expert_mass=mass,
        final_actions_standardized=np.zeros((*grid, 10, 7)),
    )
    loaded = audit._load_archive(path, denoise=0)
    assert loaded["audit"]["raw_mean_recompute_max_abs_error"] == 0.0
    assert loaded["audit"]["expert_mass_recompute_max_abs_error"] == 0.0
    assert loaded["audit"]["selected_weight_sum_max_abs_error"] == 0.0
