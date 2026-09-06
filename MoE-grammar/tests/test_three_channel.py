import numpy as np
import pytest
import torch

from moe_grammar.run_three_channel_audit import event_bootstrap_interval
from moe_grammar.three_channel import (
    ContinuousVARGrammar,
    causal_recurrence_features,
    combine_overregularity,
)


def test_var_score_is_causal() -> None:
    rng = np.random.default_rng(8)
    episodes = 50
    length = 10
    starts = np.arange(episodes) * length
    lengths = np.full(episodes, length)
    values = rng.normal(size=(episodes * length, 3)).astype(np.float32)
    for start in starts:
        for position in range(1, length):
            values[start + position] += 0.8 * values[start + position - 1]
    model = ContinuousVARGrammar(samples_per_episode=5).fit(
        values, starts, lengths, np.arange(episodes)
    )
    sequence = values[:length].copy()
    changed = sequence.copy()
    changed[6:] += 100.0
    original = model.score(sequence, np.asarray([0]), np.asarray([length]))
    modified = model.score(changed, np.asarray([0]), np.asarray([length]))
    np.testing.assert_allclose(original[:6], modified[:6])


def test_recurrence_distinguishes_abab_from_progression() -> None:
    abab = np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0]], dtype=np.float32)
    progress = np.arange(5, dtype=np.float32)[:, None]
    _, abab_periodic, _, _ = causal_recurrence_features(abab, [0], [5])
    _, progress_periodic, _, _ = causal_recurrence_features(progress, [0], [5])
    assert abab_periodic[-1] == 0.0
    assert progress_periodic[-1] > 0.0


def test_recurrence_masks_undefined_early_positions() -> None:
    values = np.arange(6, dtype=np.float32)[:, None]
    speed, periodic, _, valid = causal_recurrence_features(values, [0], [6], max_lag=4)
    # Lag-1 speed is undefined only at position 0; lag-2..4 recurrence at 0 and 1.
    assert not valid[0, 0] and valid[1, 0]
    assert not valid[0, 1] and not valid[1, 1] and valid[2, 1]
    # Undefined entries must not carry a large sentinel that distorts the CDF.
    assert np.isnan(speed[0]) and np.isnan(periodic[0]) and np.isnan(periodic[1])
    assert np.isfinite(speed[valid[:, 0]]).all()
    assert np.isfinite(periodic[valid[:, 1]]).all()


def test_event_bootstrap_widens_with_fewer_events() -> None:
    rng = np.random.default_rng(3)
    many = rng.uniform(size=400).tolist()
    few = many[:20]
    wide = event_bootstrap_interval(few, draws=2000)
    narrow = event_bootstrap_interval(many, draws=2000)
    assert wide is not None and narrow is not None
    assert wide[1] - wide[0] > narrow[1] - narrow[0]
    # A degenerate single event carries no interval rather than a fake one.
    assert event_bootstrap_interval([0.7]) is None
    constant = event_bootstrap_interval([0.5] * 50, draws=500)
    np.testing.assert_allclose(constant, [0.5, 0.5])


def test_overregularity_needs_two_high_components() -> None:
    output = combine_overregularity(
        np.asarray([1.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.5]),
    )
    np.testing.assert_allclose(output, [0.5, 1.0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_var_matches_cpu_ridge() -> None:
    rng = np.random.default_rng(19)
    episodes = 30
    length = 8
    starts = np.arange(episodes) * length
    lengths = np.full(episodes, length)
    values = rng.normal(size=(episodes * length, 4)).astype(np.float32)
    indexes = np.arange(episodes)
    cpu = ContinuousVARGrammar(samples_per_episode=5).fit(
        values, starts, lengths, indexes
    )
    cuda = ContinuousVARGrammar(samples_per_episode=5).fit(
        values, starts, lengths, indexes, device="cuda:0"
    )
    np.testing.assert_allclose(cuda.coef_, cpu.coef_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cuda.intercept_, cpu.intercept_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        cuda.score(values, starts, lengths, device="cuda:0"),
        cpu.score(values, starts, lengths),
        rtol=1e-4,
        atol=1e-5,
    )
