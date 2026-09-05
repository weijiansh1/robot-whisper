import numpy as np
import torch

from moe_grammar.features import (
    build_query_descriptors,
    deterministic_flow_permutations,
    extract_multitrack_features,
    step_feature_names,
)


def test_uniform_routes_have_expected_geometry() -> None:
    probability = torch.full((2, 8, 10, 11, 32), 1.0 / 32.0)
    expert_ids = torch.arange(4).view(1, 1, 1, 1, 4).expand(2, 8, 10, 11, 4)
    features = extract_multitrack_features(probability, expert_ids).numpy()
    names = list(step_feature_names())
    assert features.shape == (2, 10, 81)
    np.testing.assert_allclose(features[..., names.index("entropy|layer_0")], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        features[..., names.index("soft_token_consensus|layer_0")], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        features[..., names.index("top4_token_consensus|layer_0")], 1.0, atol=1e-6
    )
    np.testing.assert_allclose(features[..., names.index("flow_velocity|layer_0")], 0.0)


def test_descriptor_retains_levels_and_derivatives() -> None:
    values = np.zeros((3, 10, 81), dtype=np.float32)
    values[:, :, 0] = np.arange(10)
    descriptor = build_query_descriptors(values)
    assert descriptor.shape == (3, 2187)
    level_size = 10 * 81
    delta_size = 9 * 81
    np.testing.assert_allclose(descriptor[:, level_size : level_size + delta_size : 81], 1.0)
    np.testing.assert_allclose(descriptor[:, level_size + delta_size :: 81], 0.0)


def test_flow_permutations_are_stable_across_batches() -> None:
    whole = deterministic_flow_permutations(12, seed=91)
    split = np.concatenate(
        [
            deterministic_flow_permutations(5, seed=91, row_offset=0),
            deterministic_flow_permutations(7, seed=91, row_offset=5),
        ]
    )
    np.testing.assert_array_equal(whole, split)
    assert all(not np.array_equal(row, np.arange(10)) for row in whole)
