from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "transfer"))

from field import (  # noqa: E402
    FLOW_METRICS,
    QUERY_FEATURE_NAMES,
    RELATION_VIEWS,
    WITHIN_FEATURE_NAMES,
    compute_query_transfer,
    compute_within_transfer,
    conditional_action_graph,
    flow_transfer_features,
)


def test_identical_front_back_has_zero_transfer_gap() -> None:
    generator = torch.Generator().manual_seed(7)
    front = torch.softmax(torch.randn((3, 4, 10, 11, 32), generator=generator), dim=-1)
    probability = torch.cat((front, front.clone()), dim=1)
    result = compute_within_transfer(probability)
    features = result["features"]
    names = {name: index for index, name in enumerate(WITHIN_FEATURE_NAMES)}
    for view in RELATION_VIEWS:
        assert np.allclose(features[:, names[f"flow|{view}|gap_initial"]], 0.0, atol=1e-6)
        assert np.allclose(features[:, names[f"flow|{view}|gap_terminal"]], 0.0, atol=1e-6)


def test_positive_handoff_lag_is_detected() -> None:
    front = torch.zeros((2, 10, 3), dtype=torch.float32)
    front[:, 2:, 0] = torch.arange(8, dtype=torch.float32)
    back = torch.zeros_like(front)
    back[:, 4:, 0] = torch.arange(6, dtype=torch.float32)
    feature = flow_transfer_features(front, back)
    index = {name: position for position, name in enumerate(FLOW_METRICS)}
    assert torch.all(feature[:, index["handoff_delay"]] > 0.0)


def test_state_conditioned_action_graph_is_positive_semidefinite() -> None:
    generator = torch.Generator().manual_seed(17)
    vectors = torch.rand((5, 11, 32), generator=generator)
    vectors = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    gram = vectors @ vectors.transpose(-1, -2)
    conditional, partial = conditional_action_graph(gram[:, 1:, 1:], gram[:, 0, 1:])
    eigenvalue = torch.linalg.eigvalsh(conditional)
    assert torch.all(eigenvalue > -1e-5)
    assert torch.all(partial <= 1.0)
    assert torch.all(partial >= -1.0)


def test_query_features_are_prefix_causal() -> None:
    generator = torch.Generator().manual_seed(11)
    probability = torch.softmax(torch.randn((7, 8, 10, 11, 32), generator=generator), dim=-1)
    computed = compute_within_transfer(probability)
    episode = np.zeros(7, dtype=np.int32)
    step = np.arange(7, dtype=np.int32)
    first = compute_query_transfer(
        computed["terminal_pairs"],
        computed["terminal_channels"],
        computed["features"],
        episode,
        step,
    )

    probability[-1] = torch.softmax(torch.randn((8, 10, 11, 32), generator=generator), dim=-1)
    changed = compute_within_transfer(probability)
    second = compute_query_transfer(
        changed["terminal_pairs"],
        changed["terminal_channels"],
        changed["features"],
        episode,
        step,
    )
    assert first.shape[1] == len(QUERY_FEATURE_NAMES)
    assert np.allclose(first[:-1], second[:-1], equal_nan=True)
