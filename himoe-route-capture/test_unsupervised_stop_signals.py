from __future__ import annotations

import dataclasses

import numpy as np

from analyze_denoise_stop_sweep import TOTAL_ROUNDS, TraceData
from analyze_unsupervised_stop_signals import (
    build_prefix_signals,
    controller_endpoint_error,
)


def _trace(rows: int = 3) -> TraceData:
    rng = np.random.default_rng(14)
    x = rng.normal(size=(rows, TOTAL_ROUNDS + 1, 10, 7))
    probability = rng.uniform(size=(rows, 8, TOTAL_ROUNDS, 10, 32))
    probability /= probability.sum(axis=-1, keepdims=True)
    ids = np.argsort(probability, axis=-1)[..., -4:][..., ::-1].astype(np.int16)
    selected = np.take_along_axis(probability, ids, axis=-1)
    return TraceData(
        name="synthetic",
        run=None,  # type: ignore[arg-type]
        x=x,
        probability=probability,
        state_probability=probability[:, :, :, :1],
        expert_ids=ids,
        state_expert_ids=ids[:, :, :, :1],
        selected_probability=selected,
        query_id=np.arange(rows),
        candidate_id=np.zeros(rows, dtype=np.int64),
        rollout_id=np.arange(rows),
        action_std=np.asarray([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0]),
    )


def test_controller_error_ignores_gripper_magnitude_but_flags_sign() -> None:
    data = _trace(rows=2)
    x = data.x.copy()
    x[:, TOTAL_ROUNDS] = 0.0
    x[:, TOTAL_ROUNDS, :, 6] = 0.25
    data = dataclasses.replace(data, x=x)
    prediction = np.zeros((2, 8, 10, 7), dtype=np.float64)
    prediction[0, ..., 6] = 100.0
    prediction[1, ..., 6] = 0.0

    score, continuous, flip = controller_endpoint_error(data, prediction, 0.05)

    np.testing.assert_allclose(continuous, 0.0)
    np.testing.assert_allclose(score[0], 0.0)
    assert not np.any(flip[0])
    assert np.all(flip[1])
    assert np.all(score[1] > 0.05)


def test_controller_error_uses_physical_scale_on_continuous_dims() -> None:
    data = _trace(rows=1)
    x = data.x.copy()
    x[:, TOTAL_ROUNDS] = 0.0
    x[:, TOTAL_ROUNDS, :, 6] = 1.0
    data = dataclasses.replace(data, x=x)
    prediction = np.zeros((1, 8, 10, 7), dtype=np.float64)
    prediction[..., 0] = 0.1
    prediction[..., 6] = 2.0

    score, continuous, flip = controller_endpoint_error(data, prediction, 1.0)

    expected = 0.2 / np.sqrt(6.0)
    np.testing.assert_allclose(continuous, expected)
    np.testing.assert_allclose(score, expected)
    assert not np.any(flip)


def test_all_stop_signals_are_prefix_only() -> None:
    data = _trace()
    changed = data.x.copy()
    changed[:, TOTAL_ROUNDS] += 1000.0
    altered = dataclasses.replace(data, x=changed)

    original_signals = build_prefix_signals(data)
    altered_signals = build_prefix_signals(altered)

    assert [(item.family, item.name) for item in original_signals] == [
        (item.family, item.name) for item in altered_signals
    ]
    for original, modified in zip(original_signals, altered_signals, strict=True):
        np.testing.assert_allclose(original.values, modified.values)
