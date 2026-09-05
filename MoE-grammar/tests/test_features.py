import numpy as np
import torch

from moe_grammar.features import (
    build_clean_query_descriptors,
    build_query_descriptors,
    clean_descriptor_feature_names,
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


def test_clean_descriptor_does_not_redifferentiate_native_dynamics() -> None:
    names = step_feature_names()
    values = np.zeros((2, 10, len(names)), dtype=np.float32)
    velocity = names.index("flow_velocity|layer_0")
    acceleration = names.index("flow_acceleration|layer_0")
    values[:, :, velocity] = np.arange(10)
    values[:, :, acceleration] = np.square(np.arange(10))
    descriptor_names = clean_descriptor_feature_names(names)
    descriptor = build_clean_query_descriptors(values, names)
    assert descriptor.shape[1] == len(descriptor_names)
    assert not any(
        "flow_slope|flow_velocity" in name or "flow_slope|flow_acceleration" in name
        for name in descriptor_names
    )


def test_features_are_invariant_to_independent_expert_permutations_by_layer() -> None:
    generator = torch.Generator().manual_seed(17)
    logits = torch.randn((2, 8, 10, 11, 32), generator=generator)
    probability = torch.softmax(logits, dim=-1)
    expert_ids = torch.topk(probability, k=4, dim=-1).indices
    permutations = torch.stack([torch.randperm(32, generator=generator) for _ in range(8)])
    inverse = torch.argsort(permutations, dim=1)
    permuted_probability = torch.empty_like(probability)
    permuted_ids = torch.empty_like(expert_ids)
    for layer in range(8):
        permuted_probability[:, layer] = probability[:, layer, ..., permutations[layer]]
        permuted_ids[:, layer] = inverse[layer][expert_ids[:, layer]]
    original = extract_multitrack_features(probability, expert_ids)
    permuted = extract_multitrack_features(permuted_probability, permuted_ids)
    torch.testing.assert_close(original, permuted, rtol=1e-5, atol=1e-6)


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
