import numpy as np

from execution_features import (
    actual_support_boundary,
    combine_weights,
    extract_block_features,
    linear_position_slope,
    sparse_hellinger_distance,
    support_jaccard_distance,
)


def test_combine_weights_reconstructs_runtime_normalization():
    raw = np.array([[0.10, 0.20, 0.05, 0.15]], dtype=np.float32)
    got = combine_weights(raw)
    np.testing.assert_allclose(got.sum(axis=-1), 1.0)
    np.testing.assert_allclose(got, [[0.2, 0.4, 0.1, 0.3]])


def test_support_distance_is_order_invariant_and_has_exact_jaccard_value():
    a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.uint8)
    b = np.array([[4, 3, 2, 1], [3, 4, 5, 6]], dtype=np.uint8)
    got = support_jaccard_distance(a, b)
    np.testing.assert_allclose(got, [0.0, 1.0 - 2.0 / 6.0])


def test_sparse_hellinger_matches_dense_reference():
    ids_a = np.array([[0, 2]], dtype=np.uint8)
    ids_b = np.array([[1, 2]], dtype=np.uint8)
    wa = np.array([[1.0, 3.0]], dtype=np.float32)
    wb = np.array([[2.0, 2.0]], dtype=np.float32)
    dense_a = np.array([[0.25, 0.0, 0.75]])
    dense_b = np.array([[0.0, 0.5, 0.5]])
    expected = np.sqrt(1.0 - np.sqrt(dense_a * dense_b).sum(axis=-1))
    np.testing.assert_allclose(
        sparse_hellinger_distance(ids_a, wa, ids_b, wb), expected
    )


def test_boundary_uses_runtime_support_during_probability_tie():
    p = np.array([[0.30, 0.20, 0.20, 0.20, 0.10]], dtype=np.float32)
    ids = np.array([[0, 1, 2, 4]], dtype=np.uint8)
    raw = np.take_along_axis(p, ids, axis=-1)
    margin, fourth, fifth, p5 = actual_support_boundary(p, ids, raw)
    np.testing.assert_allclose(margin, [-0.10])
    np.testing.assert_array_equal(fourth, [4])
    np.testing.assert_array_equal(fifth, [3])
    np.testing.assert_allclose(p5, [0.20])


def test_position_slope_recovers_linear_increment():
    values = np.stack((np.arange(10), 3.0 + 2.5 * np.arange(10)))
    np.testing.assert_allclose(linear_position_slope(values), [1.0, 2.5])


def test_block_extraction_shapes_and_constant_route_zero_churn():
    rng = np.random.default_rng(7)
    p = rng.uniform(size=(2, 8, 10, 11, 32)).astype(np.float32)
    p /= p.sum(axis=-1, keepdims=True)
    ids = np.argpartition(p, -4, axis=-1)[..., -4:].astype(np.uint8)
    raw = np.take_along_axis(p, ids, axis=-1)
    ids[:, :, 1:] = ids[:, :, :1]
    raw[:, :, 1:] = raw[:, :, :1]
    p[:, :, 1:] = p[:, :, :1]

    got = extract_block_features(p, ids, raw)

    assert got.final_ids.shape == (2, 8, 11, 4)
    assert got.final_weights.shape == (2, 8, 11, 4)
    assert got.token_maps["top45_margin"].shape == (2, 10)
    assert got.layer_maps["late_flow_support_churn"].shape == (2, 8)
    np.testing.assert_allclose(got.scalars["late_flow_support_churn"], 0.0)
    np.testing.assert_allclose(got.scalars["late_flow_exec_churn"], 0.0, atol=1e-4)
    assert got.audit["combine_sum_max_error"] < 1e-6
