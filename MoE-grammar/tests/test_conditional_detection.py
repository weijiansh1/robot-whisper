import numpy as np
import torch

from moe_grammar.conditional_detection import (
    PhaseConditionalCDF,
    causal_aggregates,
    multiscale_scan,
    page_cusum,
)
from moe_grammar.train_prefix_gru import PrefixGRU


def test_phase_conditional_cdf_removes_position_shift() -> None:
    task = np.zeros(8, dtype=np.int16)
    position = np.tile(np.arange(2), 4)
    values = {"score": np.asarray([0.0, 10.0, 1.0, 11.0, 2.0, 12.0, 3.0, 13.0])}
    model = PhaseConditionalCDF(min_support=4).fit(
        values, task, position, np.arange(8)
    )

    result = model.transform("score", np.asarray([2.0, 12.0]), 0, np.asarray([0, 1]))

    np.testing.assert_allclose(result[0], result[1])


def test_sequential_statistics_are_causal() -> None:
    prefix = np.asarray([0.4, 0.6, 0.9])
    extended = np.r_[prefix, 0.999, 0.001]
    prefix_scores = causal_aggregates(prefix)
    extended_scores = causal_aggregates(extended)

    for name in prefix_scores:
        np.testing.assert_allclose(prefix_scores[name], extended_scores[name][: len(prefix)])


def test_page_and_scan_react_to_persistent_positive_shift() -> None:
    values = np.asarray([-0.5, 0.0, 1.5, 1.5, 1.5])
    page = page_cusum(values, 0.5)
    scan = multiscale_scan(values)

    assert page[-1] > page[1]
    assert scan[-1] > scan[1]


def test_prefix_gru_cannot_read_future_queries() -> None:
    torch.manual_seed(3)
    model = PrefixGRU(input_dim=5, task_count=2, hidden_dim=8).eval()
    prefix = torch.randn(2, 6, 5)
    changed_future = prefix.clone()
    changed_future[:, 3:] += 100.0
    task = torch.tensor([0, 1])

    with torch.no_grad():
        original = model(prefix, task)
        changed = model(changed_future, task)

    torch.testing.assert_close(original[:, :3], changed[:, :3])
