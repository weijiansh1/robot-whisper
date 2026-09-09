import numpy as np

import analyze_unweighted_expert_norm as raw


def test_slot_summary_separates_unweighted_mean_from_gate_weighted_mass():
    slot = np.asarray([[2.0, 4.0, 6.0, 8.0]])
    weight_a = np.asarray([[0.25, 0.25, 0.25, 0.25]])
    weight_b = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    mean_a, mass_a, spread_a = raw._slot_summaries(slot, weight_a)
    mean_b, mass_b, spread_b = raw._slot_summaries(slot, weight_b)
    np.testing.assert_allclose(mean_a, mean_b)
    np.testing.assert_allclose(spread_a, spread_b)
    np.testing.assert_allclose(mean_a, [5.0])
    np.testing.assert_allclose(mass_a, [5.0])
    np.testing.assert_allclose(mass_b, [2.0])


def test_linear_slope_keeps_raw_units_per_round():
    value = np.asarray([[1.0, 3.0, 5.0], [5.0, 4.0, 3.0]])
    np.testing.assert_allclose(raw._linear_slope(value), [2.0, -1.0])


def test_pool_auc_handles_ties_and_single_class_pool():
    scores = np.asarray([[[[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0]]]])
    labels = np.asarray([[[0, 0, 1, 1], [1, 1, 1, 1]]])
    pool, task, macro = raw._pool_auc_from_scores(scores, labels)
    np.testing.assert_allclose(pool[0, 0, 0], 1.0)
    assert np.isnan(pool[0, 0, 1])
    np.testing.assert_allclose(task, [[1.0]])
    np.testing.assert_allclose(macro, [1.0])


def test_pool_spearman_is_computed_within_each_pool():
    scores = np.asarray([[[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]]])
    target = np.asarray([[[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]])
    pool, task, macro = raw._pool_spearman(scores, target)
    np.testing.assert_allclose(pool, [[[1.0, -1.0]]])
    np.testing.assert_allclose(task, [[0.0]])
    np.testing.assert_allclose(macro, [0.0])


def test_action_eccentricity_uses_other_candidates_only():
    actions = np.asarray([[[[[0.0]], [[2.0]], [[4.0]]]]])
    result = raw._action_eccentricity(actions)
    np.testing.assert_allclose(result, [[[3.0, 2.0, 3.0]]])
