from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graph"))

from structure import (  # noqa: E402
    QUERY_FEATURE_NAMES,
    WITHIN_FEATURE_NAMES,
    compute_query_structure,
    compute_within_structure,
    encode_graph_views,
)


def uniform(rows: int) -> torch.Tensor:
    return torch.full((rows, 8, 10, 11, 32), 1.0 / 32.0)


def test_schema_and_static_graph_identities():
    result = compute_within_structure(uniform(3))
    assert result["features"].shape == (3, len(WITHIN_FEATURE_NAMES))
    names = list(WITHIN_FEATURE_NAMES)
    flow = [index for index, name in enumerate(names) if name.startswith("flow|")]
    np.testing.assert_allclose(result["features"][:, flow], 0.0, atol=1e-7)
    rank = names.index("terminal|all|effective_rank_fraction")
    affinity = names.index("terminal|all|token_affinity")
    np.testing.assert_allclose(result["features"][:, rank], 1.0 / 11.0, atol=1e-6)
    np.testing.assert_allclose(result["features"][:, affinity], 1.0, atol=1e-6)


def test_gram_and_spectrum_ignore_expert_relabeling():
    generator = torch.Generator().manual_seed(7)
    probability = torch.rand((2, 8, 10, 11, 32), generator=generator)
    probability /= probability.sum(dim=-1, keepdim=True)
    permutation = torch.randperm(32, generator=generator)
    original, _ = encode_graph_views(probability)
    relabeled, _ = encode_graph_views(probability[..., permutation])
    torch.testing.assert_close(original["token_gram"], relabeled["token_gram"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(original["spectrum"], relabeled["spectrum"], atol=2e-5, rtol=2e-5)
    assert not torch.allclose(original["edge_action"], relabeled["edge_action"])
    assert not torch.allclose(original["expert_load"], relabeled["expert_load"])


def test_query_features_are_prefix_causal():
    within = compute_within_structure(uniform(7))
    episode = np.zeros(7, dtype=np.int32)
    step = np.arange(7, dtype=np.int32)
    first = compute_query_structure(within["final_views"], episode, step)
    altered = {name: value.copy() for name, value in within["final_views"].items()}
    for value in altered.values():
        value[-1] += 0.25
    second = compute_query_structure(altered, episode, step)
    np.testing.assert_allclose(first[:-1], second[:-1], equal_nan=True)
    assert not np.allclose(first[-1], second[-1], equal_nan=True)
    assert first.shape == (7, len(QUERY_FEATURE_NAMES))


def test_feature_inventory_is_small_and_unique():
    names = WITHIN_FEATURE_NAMES + QUERY_FEATURE_NAMES
    assert len(names) == 218
    assert len(names) == len(set(names))
