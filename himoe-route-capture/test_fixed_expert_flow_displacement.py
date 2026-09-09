import numpy as np

import analyze_fixed_expert_flow_displacement as audit


def _moving_slot_fixture():
    ids = np.empty((1, 1, 4, 10, 1, 2), dtype=np.int64)
    raw = np.full(ids.shape, 99.0, dtype=np.float64)
    weight = np.full(ids.shape, 0.5, dtype=np.float64)
    candidate_ids = np.asarray([[7, 0], [1, 7], [7, 2], [3, 7]])
    candidate_raw = np.asarray([[1.0, 99.0], [99.0, 2.0], [3.0, 99.0], [99.0, 4.0]])
    for denoise in range(10):
        ids[0, 0, :, denoise, 0] = candidate_ids
        raw[0, 0, :, denoise, 0] = candidate_raw
    return ids, raw, weight


def test_action_mean_correction_recovers_model_normalized_coordinates():
    raw_action = np.asarray([[[2.0, -3.0]]])
    mean = np.asarray([1.0, -1.0])
    std = np.asarray([2.0, 4.0])
    archived = raw_action / std
    corrected = audit._correct_actions_over_std(archived, mean, std)
    np.testing.assert_allclose(corrected, (raw_action - mean) / std)


def test_state_loo_seed_template_excludes_the_current_state():
    target = np.asarray([[[1.0, 10.0], [3.0, 30.0], [8.0, 80.0]]])
    result = audit._state_loo_seed_template(target)
    expected = np.asarray([[[5.5, 55.0], [4.5, 45.0], [2.0, 20.0]]])
    np.testing.assert_allclose(result, expected)


def test_discovery_template_never_reads_validation_states():
    target = np.asarray(
        [[[1.0, 10.0], [3.0, 30.0], [100.0, 1000.0], [200.0, 2000.0]]]
    )
    template, residual = audit._discovery_seed_template_residual(
        target,
        discovery_states=np.asarray([0, 1]),
        validation_states=np.asarray([2, 3]),
    )
    np.testing.assert_allclose(template, [[2.0, 20.0]])
    np.testing.assert_allclose(residual[0, 2], [98.0, 980.0])
    np.testing.assert_allclose(residual[0, 3], [198.0, 1980.0])


def test_descriptive_pool_spearman_is_pool_then_task_equal():
    target = np.asarray(
        [
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        ]
    )
    predictor = np.asarray(
        [
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        ]
    )
    result = audit._descriptive_pool_spearman(predictor, target)
    np.testing.assert_allclose(result["equal_pool_per_task_spearman"], [0.0, 1.0])
    assert result["equal_task_macro_spearman"] == 0.5


def test_fixed_expert_stratum_follows_identity_when_selected_slot_moves():
    ids, raw, weight = _moving_slot_fixture()
    coefficients, valid, names, coverage = audit._fixed_expert_pair_coefficients(
        ids, raw, weight, min_selected=4
    )
    axis = names.index("raw.d0")
    # Expert 7 has raw norms 1,2,3,4.  For every i<j, sign(raw_i-raw_j)=-1.
    np.testing.assert_allclose(coefficients[axis, 0, 0], -np.ones(6) / 6.0)
    assert valid[axis, 0, 0]
    assert coverage["raw.d0"]["median_strata_per_valid_state"] == [1.0]

    gate_axis = names.index("gate.d0")
    np.testing.assert_allclose(coefficients[gate_axis, 0, 0], 0.0)


def test_unselected_candidate_is_not_filled_with_a_zero_norm():
    ids, raw, weight = _moving_slot_fixture()
    # Remove expert 7 from candidate 3.  Only three candidates now contain it,
    # so the fixed-ID stratum must fail min_selected=4 instead of adding a zero.
    ids[0, 0, 3, :, 0] = np.asarray([3, 4])
    coefficients, valid, names, _ = audit._fixed_expert_pair_coefficients(
        ids, raw, weight, min_selected=4
    )
    axis = names.index("raw.d0")
    assert not valid[axis, 0, 0]
    np.testing.assert_allclose(coefficients[axis, 0, 0], 0.0)


def test_frozen_split_permutation_is_reproducible():
    # Four features, two tasks, two states, and K=4 (six candidate pairs).
    pair_i, pair_j, _ = audit._pair_index(4)
    coefficients = np.zeros((4, 2, 2, 6), dtype=np.float64)
    coefficients[:, :, :, :] = 1.0 / 6.0
    # The positive-orientation rebound feature should use the opposite ordering.
    coefficients[3] *= -1.0
    valid = np.ones((4, 2, 2), dtype=bool)
    target = np.broadcast_to(np.arange(4.0), (2, 2, 4)).copy()
    first = audit._frozen_split_test(
        coefficients,
        valid,
        target,
        pair_i,
        pair_j,
        np.asarray([0, 1]),
        permutations=99,
        seed=17,
    )
    second = audit._frozen_split_test(
        coefficients,
        valid,
        target,
        pair_i,
        pair_j,
        np.asarray([0, 1]),
        permutations=99,
        seed=17,
    )
    assert first == second
    assert all(
        row["all_five_tasks_expected_direction"] for row in first["features"].values()
    )
    assert first["features"]["raw.d6"][
        "expected_direction_pair_ranking_advantage_percentage_points"
    ] == 50.0
