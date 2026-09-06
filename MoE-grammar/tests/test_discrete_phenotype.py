import numpy as np

from moe_grammar.discrete_phenotype import (
    FactorialCategoricalHMM,
    FactorialContextGrammar,
    FactorialDurationModel,
    TernaryChordTokenizer,
    categorical_recurrence,
    chord_codes,
    phenotype_track_scores,
)
from moe_grammar.open_world import PHENOTYPE_NAMES


def _column(name: str) -> int:
    return PHENOTYPE_NAMES.index(name)


def test_track_signs_match_interpretable_symbols() -> None:
    values = np.zeros((2, len(PHENOTYPE_NAMES)), dtype=np.float32)
    values[1, _column("entropy_terminal_late_layers")] = 4.0
    values[1, _column("soft_consensus_terminal_late_layers")] = 2.0
    values[1, _column("top4_consensus_terminal_late_layers")] = 2.0
    values[1, _column("effective_rank_mean")] = -3.0
    values[1, _column("effective_rank_terminal_late_layers")] = -3.0
    values[1, _column("velocity_mean")] = 6.0
    values[1, _column("entropy_flow_slope_late_layers")] = -4.0
    tracks = phenotype_track_scores(values)
    assert tracks[1, 0] < tracks[0, 0]  # Fl
    assert tracks[1, 1] > tracks[0, 1]  # Sy
    assert tracks[1, 2] < tracks[0, 2]  # LR
    assert tracks[1, 3] > tracks[0, 3]  # Sw
    assert tracks[1, 4] > tracks[0, 4]  # Fs


def test_stateless_tokenizer_does_not_inject_sequence_memory() -> None:
    tracks = np.tile(np.linspace(-2.0, 2.0, 20)[:, None], (1, 5))
    starts = np.array([0, 10])
    lengths = np.array([10, 10])
    tokenizer = TernaryChordTokenizer().fit(
        tracks, starts, lengths, np.array([0, 1])
    )
    original = tokenizer.transform(tracks)
    permutation = np.array([4, 1, 8, 0, 9, 2, 7, 3, 6, 5, 14, 11, 18, 10, 19, 12, 17, 13, 16, 15])
    shuffled = tokenizer.transform(tracks[permutation])
    np.testing.assert_array_equal(shuffled, original[permutation])


def test_full_prefix_hmm_is_causal() -> None:
    starts = np.array([0, 6])
    lengths = np.array([6, 6])
    progress = np.tile(np.linspace(0.0, 1.0, 6), 2)
    chords = np.ones((12, 5), dtype=np.int8)
    chords[3:6, 0] = 2
    chords[9:12, 0] = 2
    model = FactorialCategoricalHMM(phase_states=3).fit(
        chords, progress, starts, lengths, np.array([0, 1])
    )
    baseline = model.score(chords, starts, lengths)[0]
    changed = chords.copy()
    changed[4:] = 0
    altered = model.score(changed, starts, lengths)[0]
    np.testing.assert_allclose(baseline[:4], altered[:4], rtol=0, atol=0)
    last_only = model.score_last_observation(chords, starts, lengths)[0]
    # At q0 and q1, the complete prefix contains at most the same one observation.
    np.testing.assert_allclose(baseline[[0, 1, 6, 7]], last_only[[0, 1, 6, 7]])


def test_ordered_context_distinguishes_histories_that_bag_merges() -> None:
    # AB->C is common; BA->D is common. Their two-chord bags are identical.
    a = np.array([0, 1, 1, 1, 1], dtype=np.int8)
    b = np.array([2, 1, 1, 1, 1], dtype=np.int8)
    c = np.array([1, 2, 1, 1, 1], dtype=np.int8)
    d = np.array([1, 0, 1, 1, 1], dtype=np.int8)
    sequences = []
    for _ in range(20):
        sequences.extend([a, b, c, b, a, d])
    chords = np.asarray(sequences)
    starts = np.arange(0, len(chords), 3)
    lengths = np.full(len(starts), 3)
    episodes = np.arange(len(starts))
    ordered = FactorialContextGrammar(max_order=2, min_support=1.0, ordered=True).fit(
        chords, starts, lengths, episodes
    )
    bag = FactorialContextGrammar(max_order=2, min_support=1.0, ordered=False).fit(
        chords, starts, lengths, episodes
    )
    code_a, code_b = map(int, chord_codes(np.stack([a, b])))
    ordered_ab, depth = ordered.distribution([code_a, code_b])
    bag_ab, _ = bag.distribution([code_a, code_b])
    ordered_ba, _ = ordered.distribution([code_b, code_a])
    bag_ba, _ = bag.distribution([code_b, code_a])
    order_one, order_one_depth = ordered.distribution(
        [code_a, code_b], max_order=1
    )
    assert depth == 2
    assert order_one_depth == 1
    assert ordered_ab[1, 2] > ordered_ba[1, 2]
    assert not np.allclose(order_one, ordered_ab)
    np.testing.assert_allclose(bag_ab, bag_ba)


def test_duration_and_period_two_are_explicit() -> None:
    healthy = np.ones((12, 5), dtype=np.int8)
    healthy[1::2, 0] = 2
    starts = np.array([0, 6])
    lengths = np.array([6, 6])
    model = FactorialDurationModel(samples_per_episode=6).fit(
        healthy, starts, lengths, np.array([0, 1])
    )
    long_dwell = np.ones((6, 5), dtype=np.int8)
    dwell_score, _ = model.score(long_dwell, np.array([0]), np.array([6]))
    assert dwell_score[-1] > dwell_score[0]

    alternating = np.ones((6, 5), dtype=np.int8)
    alternating[1::2, 0] = 2
    freeze, periodic, lag = categorical_recurrence(
        alternating, np.array([0]), np.array([6])
    )
    assert periodic[-1] == 1.0
    assert lag[-1] == 2
    assert freeze[-1] < periodic[-1]
