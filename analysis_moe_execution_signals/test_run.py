from types import SimpleNamespace

import numpy as np

from run import query_execution_features, stage_rank_residual_multi, trailing_mean


def test_trailing_mean_preserves_feature_axis_and_requires_complete_window():
    values = np.arange(2 * 5 * 3, dtype=float).reshape(2, 5, 3)
    values[1, 1] = np.nan
    got = trailing_mean(values, 3)
    assert got.shape == values.shape
    assert np.isnan(got[:, :2]).all()
    assert np.isfinite(got[0, 2:]).all()
    assert np.isnan(got[1, 2:4]).all()
    np.testing.assert_allclose(got[0, 2], values[0, :3].mean(axis=0))


def test_query_features_use_state_early_layers_and_action_late_layers():
    ids = np.zeros((2, 8, 11, 4), dtype=np.uint8)
    ids[:] = np.array([0, 1, 2, 3], dtype=np.uint8)
    # On query 1, replace one expert in all late-layer action supports only.
    ids[1, 4:, 1:, 3] = 4
    weights = np.ones(ids.shape, dtype=np.float16)
    arrays = {"final_ids": ids, "final_weights": weights}
    corpus = SimpleNamespace(
        rowidx=np.array([[0, 1]], dtype=np.int32),
        valid=np.array([[True, True]]),
    )

    scalar, token = query_execution_features(arrays, corpus)

    expected = 1.0 - 3.0 / 5.0
    assert np.isnan(scalar["query_support_churn"][0, 0])
    np.testing.assert_allclose(scalar["query_support_churn"][0, 1], expected)
    np.testing.assert_allclose(scalar["early_state_query_churn"][0, 1], 0.0)
    np.testing.assert_allclose(scalar["state_action_churn_gap"][0, 1], expected)
    np.testing.assert_allclose(token["query_support_churn"][0, 1], expected)


def test_multi_residual_removes_monotone_rank_predictor_without_labels():
    groups = np.repeat(["a", "b"], 6)
    predictor = np.tile(np.arange(6, dtype=float)[:, None], (2, 5)).reshape(12, 5)
    score = 4.0 * predictor + 7.0
    residual = stage_rank_residual_multi(score, (predictor,), groups)
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)
