import numpy as np

import analyze_expert_activation_proxy as proxy


def test_fast_auc_orders_positive_scores_above_negative_scores():
    assert proxy._fast_auc(np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])) == 1.0
    assert proxy._fast_auc(np.asarray([2.0, 3.0]), np.asarray([0.0, 1.0])) == 0.0


def test_fast_auc_gives_half_credit_to_ties():
    assert proxy._fast_auc(np.asarray([1.0]), np.asarray([1.0])) == 0.5


def test_pair_rms_uses_all_non_candidate_axes():
    value = np.asarray([[[0.0, 0.0]], [[3.0, 4.0]]])
    result = proxy._pair_rms(value, np.asarray([0]), np.asarray([1]))
    np.testing.assert_allclose(result, [np.sqrt(12.5)])


def test_row_standardize_is_scene_local():
    value = np.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    result = proxy._row_standardize(value)
    np.testing.assert_allclose(result.mean(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=1), 1.0, atol=1e-12)
