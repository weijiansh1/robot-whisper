from __future__ import annotations

import json

import numpy as np

import analyze_internal_mlp_activation as probe


def test_internal_statistics_preserve_raw_l2_l1_and_fixed_rms_identity() -> None:
    gate = np.asarray([[-5.0, -1.0, 4.0, 6.0]])
    value = np.asarray([[0.0, 1e-4, -3.0, 4.0]])
    result = probe.internal_statistics(gate, value)
    assert result.shape == (1, 9)
    l2 = np.sqrt(9.0 + 16.0 + 1e-8)
    np.testing.assert_allclose(result[0, 0], l2)
    np.testing.assert_allclose(result[0, 1], 1e-4 + 3.0 + 4.0)
    np.testing.assert_allclose(result[0, 2], (1e-4 - 3.0 + 4.0) / 4)
    np.testing.assert_allclose(result[0, 3:5], [0.5, 4.0])
    np.testing.assert_allclose(result[0, 5:], [l2 / 2.0, 0.5, 0.25, 0.5])


def test_scalar_aggregation_is_slot_permutation_invariant() -> None:
    stats = np.arange(2 * 3 * 6, dtype=np.float64).reshape(1, 2, 3, 6)
    weights = np.asarray([[[0.2, 0.3, 0.5], [0.1, 0.7, 0.2]]])
    original = probe.aggregate_selected_scalars(stats, weights)
    order = np.asarray([2, 0, 1])
    permuted = probe.aggregate_selected_scalars(
        stats[:, :, order], weights[:, :, order]
    )
    np.testing.assert_allclose(permuted, original)
    assert original.shape == (1, 2, 12)


def test_scalar_aggregation_renormalizes_only_router_mass_not_activation() -> None:
    stats = np.zeros((1, 1, 2, 6), dtype=np.float64)
    stats[0, 0, :, 0] = [2.0, 10.0]
    weights = np.asarray([[[0.25, 0.75]]])
    result = probe.aggregate_selected_scalars(stats, weights)
    np.testing.assert_allclose(result[0, 0, 0], 8.0)
    np.testing.assert_allclose(result[0, 0, 6], np.sqrt(12.0))
    scaled_weight = probe.aggregate_selected_scalars(stats, weights * 0.994)
    np.testing.assert_allclose(scaled_weight, result)


def test_internal_and_postdown_controls_have_exactly_matched_width() -> None:
    rng = np.random.default_rng(3)
    internal_slot = rng.uniform(size=(4, 10, 4, 9))
    postdown = rng.normal(size=(4, 10, 4, 1024))
    internal = probe.sorted_raw_l2_feature(internal_slot)
    control = probe.sorted_raw_l2_feature(probe.vector_statistics(postdown))
    assert internal.shape == control.shape == (4, 10, 4)
    assert internal.reshape(4, -1).shape[1] == probe.FEATURE_WIDTH == 40


def test_primary_sorted_l2_does_not_use_router_weights() -> None:
    slot = np.zeros((1, 1, 4, 9), dtype=np.float64)
    slot[0, 0, :, 0] = [8.0, 2.0, 11.0, 5.0]
    original = probe.sorted_raw_l2_feature(slot)
    permuted = probe.sorted_raw_l2_feature(slot[:, :, [2, 0, 3, 1]])
    np.testing.assert_array_equal(original, [[[2.0, 5.0, 8.0, 11.0]]])
    np.testing.assert_array_equal(permuted, original)


def test_internal_feature_contract_excludes_cross_expert_vector_geometry() -> None:
    assert probe.INTERNAL_SLOT_FEATURES == (
        "m_raw_l2",
        "m_raw_l1",
        "m_raw_signed_mean",
        "m_positive_fraction",
        "m_raw_linf",
    )
    assert not any(
        forbidden in name
        for name in probe.INTERNAL_SLOT_FEATURES
        for forbidden in ("angle", "cosine", "cancellation")
    )


def test_discover_checkpoints_uses_captured_hash_and_size(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    weight = checkpoint_dir / "pytorch_model.pth"
    weight.write_bytes(b"weights")
    run = tmp_path / "client" / "task" / "state-01"
    run.mkdir(parents=True)
    (run / "server_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": "declared-hash",
                "checkpoint": str(checkpoint_dir),
                "checkpoint_bytes": len(b"weights"),
            }
        )
    )
    result = probe.discover_checkpoints(tmp_path)
    assert result == {"declared-hash": weight.resolve()}


def test_primary_target_is_x10_minus_x1_not_path_energy() -> None:
    x = np.zeros((1, 11, 10, 7), dtype=np.float64)
    x[:, 0] = 1000.0
    for step in range(2, 11):
        x[:, step] = x[:, step - 1] + 3.0
    correction = probe.runtime.remaining_correction_target(x)
    assert correction.shape == (1, 70)
    np.testing.assert_allclose(correction, 27.0)


def test_exact_additive_kernel_has_no_projection_and_equal_family_weight() -> None:
    families = {
        "wide": np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        "narrow": np.asarray([[0.0], [1.0], [2.0]]),
    }
    rows = np.arange(3)
    combined = probe.exact_additive_kernel(families, rows, rows, rows)
    wide = probe._one_standardized_linear_kernel(families["wide"], rows, rows, rows)
    narrow = probe._one_standardized_linear_kernel(families["narrow"], rows, rows, rows)
    np.testing.assert_allclose(combined, (wide + narrow) / 2.0)


def test_added_feature_is_ninth_equal_family_not_equal_to_whole_b1() -> None:
    b1 = np.full((2, 2), 3.0)
    feature = np.full((2, 2), 12.0)
    expected = (8.0 * b1 + feature) / 9.0
    np.testing.assert_allclose(probe.add_one_equal_family_kernel(b1, feature), expected)


def test_held_target_cannot_change_its_exact_kernel_prediction() -> None:
    rng = np.random.default_rng(9)
    states = np.repeat(np.arange(3), 3)
    seeds = np.tile(np.arange(3), 3)
    families = {"family": rng.normal(size=(9, 5))}
    additions = {"b1": None, "b1_plus_raw": rng.normal(size=(9, 2))}
    target = rng.normal(size=(9, 3))
    original, _, fold_id = probe.cross_validated_exact_arms(
        families, additions, target, states, seeds, 3, 17
    )
    changed = target.copy()
    changed[0] += 1e6
    repeated, _, repeated_fold = probe.cross_validated_exact_arms(
        families, additions, changed, states, seeds, 3, 17
    )
    for arm in additions:
        np.testing.assert_allclose(repeated[arm][0], original[arm][0])
    np.testing.assert_array_equal(repeated_fold, fold_id)
