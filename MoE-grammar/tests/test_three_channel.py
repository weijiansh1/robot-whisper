import numpy as np
import pytest
import torch

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
    _, abab_periodic, _ = causal_recurrence_features(abab, [0], [5])
    _, progress_periodic, _ = causal_recurrence_features(progress, [0], [5])
    assert abab_periodic[-1] == 0.0
    assert progress_periodic[-1] > 0.0


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
