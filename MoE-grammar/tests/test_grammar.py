import numpy as np

from moe_grammar.grammar import (
    CountGrammar,
    DurationGrammar,
    DurationModel,
    LagRecurrenceModel,
    PositionContextGrammar,
    PositionGrammar,
)
from moe_grammar.run_dual_axis_audit import axis_groups, crossed_cluster_interval
from moe_grammar.run_task_holdout_audit import task_bootstrap_interval
from moe_grammar.statistics import auc_pairwise, cusum, empirical_percentile, stratified_pair_auc


def test_ordered_grammar_prefers_training_pattern() -> None:
    training = [np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16) for _ in range(30)]
    grammar = CountGrammar(
        vocabulary_size=4, max_order=3, min_support=5, alpha=0.5, kl_delta=0.001
    ).fit(training)
    ordered = grammar.hard_nll(np.asarray([0, 1, 2, 0, 1, 2]))[0].mean()
    shuffled = grammar.hard_nll(np.asarray([2, 1, 0, 2, 1, 0]))[0].mean()
    assert ordered < shuffled


def test_bag_context_is_invariant_to_context_order() -> None:
    sequences = [np.asarray([0, 1, 2, 3], dtype=np.int16)] * 20
    grammar = CountGrammar(vocabulary_size=5, max_order=3, context_mode="bag", min_support=1).fit(
        sequences
    )
    left = grammar.distribution([0, 1, 2])[0]
    right = grammar.distribution([2, 0, 1])[0]
    np.testing.assert_allclose(left, right)


def test_duration_model_changes_repeat_probability() -> None:
    sequences = [np.asarray([0, 0, 0, 1, 1], dtype=np.int16) for _ in range(20)]
    grammar = CountGrammar(vocabulary_size=3, max_order=1).fit(sequences)
    duration = DurationModel(n_words=2, min_support=5).fit(sequences)
    model = DurationGrammar(grammar, duration)
    priors, _ = model.priors(np.asarray([0, 0, 0, 1]))
    assert priors[1, 0] > priors[3, 0]


def test_position_grammar_uses_clock_not_history() -> None:
    sequences = [np.asarray([0, 1, 2], dtype=np.int16) for _ in range(20)]
    grammar = PositionGrammar(vocabulary_size=4, min_support=5).fit(sequences)
    ordered = grammar.hard_nll(np.asarray([0, 1, 2]))[0].mean()
    reversed_order = grammar.hard_nll(np.asarray([2, 1, 0]))[0].mean()
    assert ordered < reversed_order


def test_position_context_adds_history_beyond_clock() -> None:
    left = np.asarray([0, 1, 2, 3], dtype=np.int16)
    right = np.asarray([4, 5, 6, 7], dtype=np.int16)
    training = [left, right] * 40
    position = PositionGrammar(vocabulary_size=9, min_support=5).fit(training)
    nested = PositionContextGrammar(
        vocabulary_size=9,
        max_order=1,
        min_support=5,
    ).fit(training)
    position_nll = np.mean([position.hard_nll(sequence)[0].mean() for sequence in (left, right)])
    nested_nll = np.mean([nested.hard_nll(sequence)[0].mean() for sequence in (left, right)])
    assert nested_nll < position_nll
    unseen, depth, support = nested.distribution(2, [8])
    np.testing.assert_allclose(unseen, nested._position_distribution(2))
    assert depth == 0 and support == 0


def test_position_bag_context_ignores_local_order() -> None:
    sequences = [np.asarray([0, 1, 2, 3], dtype=np.int16)] * 20
    grammar = PositionContextGrammar(
        vocabulary_size=5,
        max_order=2,
        min_support=5,
        context_mode="bag",
    ).fit(sequences)
    left = grammar.distribution(3, [0, 1, 2])[0]
    right = grammar.distribution(3, [0, 2, 1])[0]
    np.testing.assert_allclose(left, right)


def test_order_residual_separates_lexical_from_context_surprise() -> None:
    training = [np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16) for _ in range(30)]
    grammar = CountGrammar(vocabulary_size=4, max_order=2, min_support=5).fit(training)
    # Emissions that identify each word exactly, so only the priors differ.
    def emission(sequence: np.ndarray) -> np.ndarray:
        values = np.full((len(sequence), 3), -50.0)
        values[np.arange(len(sequence)), sequence] = 0.0
        return values

    ordered = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16)
    scrambled = np.asarray([0, 2, 1, 0, 2, 1], dtype=np.int16)
    context_a, lexical_a, residual_a, _ = grammar.continuous_nll_components(
        ordered, emission(ordered)
    )
    context_b, lexical_b, residual_b, _ = grammar.continuous_nll_components(
        scrambled, emission(scrambled)
    )
    np.testing.assert_allclose(residual_a, context_a - lexical_a, atol=1e-9)
    # Both sequences use each word equally often, so lexical cost matches and only the
    # order residual can distinguish them.
    np.testing.assert_allclose(sorted(lexical_a), sorted(lexical_b), atol=1e-9)
    assert residual_b.sum() > residual_a.sum()


def test_beam_history_matches_hard_history_when_words_are_unambiguous() -> None:
    training = [np.asarray([0, 1, 2, 3], dtype=np.int16) for _ in range(40)]
    grammar = PositionContextGrammar(vocabulary_size=5, max_order=2, min_support=5).fit(training)
    sequence = np.asarray([0, 1, 2, 3], dtype=np.int16)
    emission = np.full((4, 4), -60.0)
    emission[np.arange(4), sequence] = 0.0
    hard, _ = grammar.continuous_nll(sequence, emission)
    beam = grammar.beam_continuous_nll(emission)
    np.testing.assert_allclose(beam, hard, atol=1e-6)


def test_beam_history_recovers_when_argmax_picks_the_wrong_branch() -> None:
    # Two healthy continuations that diverge after the first word.
    training = [np.asarray([0, 1, 2, 3], dtype=np.int16) for _ in range(30)]
    training += [np.asarray([0, 2, 4, 5], dtype=np.int16) for _ in range(30)]
    grammar = PositionContextGrammar(vocabulary_size=7, max_order=2, min_support=5).fit(training)
    emission = np.full((4, 6), -60.0)
    emission[0, 0] = 0.0
    emission[1, 1] = emission[1, 2] = 0.0  # genuinely ambiguous query
    emission[2, 4] = 0.0  # the continuation only the word-2 branch predicts
    emission[3, 5] = 0.0
    # argmax breaks the tie toward word 1, committing the rest of the episode to the
    # branch the observations then contradict.
    hard, _ = grammar.continuous_nll(np.asarray([0, 1, 4, 5], dtype=np.int16), emission)
    beam = grammar.beam_continuous_nll(emission)
    assert beam[2] < hard[2]
    assert beam.sum() < hard.sum()


def test_unigram_baseline_ignores_clock_and_history() -> None:
    training = [np.asarray([0, 1, 2], dtype=np.int16) for _ in range(40)]
    grammar = PositionGrammar(vocabulary_size=4, min_support=5).fit(training)
    emission = np.full((3, 3), -50.0)
    emission[np.arange(3), [0, 1, 2]] = 0.0
    unigram = grammar.unigram_continuous_nll(emission)
    # Each word appears equally often in training, so the lexical cost is flat even
    # though the clock-conditioned model would strongly prefer a specific order.
    np.testing.assert_allclose(unigram, unigram[0], atol=1e-9)
    clock, _ = grammar.continuous_nll(np.asarray([0, 1, 2]), emission)
    assert clock.mean() < unigram.mean()


def test_end_hitting_table_matches_sampling_on_first_order_grammar() -> None:
    training = [np.asarray([0, 1, 2], dtype=np.int16) for _ in range(60)]
    grammar = CountGrammar(vocabulary_size=4, max_order=1, min_support=5).fit(training)
    table = grammar.end_hitting_table(horizon=1)
    assert table.shape == (3,) and np.all((table >= 0.0) & (table <= 1.0))
    # Word 2 always ends the sentence; word 0 never does.
    assert table[2] > 0.9 and table[0] < 0.1
    sampled = grammar.end_hitting_probability([2], horizon=1, draws=400, seed=2)
    assert abs(sampled - table[2]) < 0.1
    longer = grammar.end_hitting_table(horizon=3)
    assert np.all(longer >= table - 1e-9)


def test_end_hitting_probability_grows_with_horizon() -> None:
    training = [np.asarray([0, 1, 2], dtype=np.int16) for _ in range(40)]
    grammar = CountGrammar(vocabulary_size=4, max_order=2, min_support=5).fit(training)
    near = grammar.end_hitting_probability([0, 1, 2], horizon=1, draws=200, seed=1)
    far = grammar.end_hitting_probability([0], horizon=1, draws=200, seed=1)
    assert 0.0 <= far <= near <= 1.0
    assert near > 0.8 and far < 0.2


def test_lag_recurrence_flags_periodic_patterns() -> None:
    training = [np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int16) for _ in range(30)]
    model = LagRecurrenceModel(n_words=4, max_lag=3).fit(training)
    # Exact-run duration sees six length-one runs here and cannot react at all.
    alternating = model.surprisal(np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int16))
    healthy = model.surprisal(np.asarray([0, 1, 2, 3, 0, 1], dtype=np.int16))
    assert alternating[2:].sum() > healthy[2:].sum()


def test_calibration_and_auc_helpers() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    percentile = empirical_percentile(reference, np.asarray([0.0, 2.0, 4.0]))
    assert np.all(np.diff(percentile) > 0)
    np.testing.assert_allclose(cusum(np.asarray([0.9, 0.9]), 0.8), [0.1, 0.2])
    labels = np.asarray([0, 1, 0, 1], dtype=bool)
    scores = np.asarray([0.0, 1.0, 0.2, 1.2])
    assert auc_pairwise(labels, scores) == 1.0
    value, _, pairs = stratified_pair_auc(labels, scores, np.asarray([0, 0, 1, 1]))
    assert value == 1.0 and pairs == 2


def test_dual_axis_groups_and_interval() -> None:
    groups = axis_groups(np.arange(8), folds=4, seed=3)
    assert sorted(np.concatenate(groups).tolist()) == list(range(8))
    assert all(len(group) == 2 for group in groups)
    values = np.full(16, -0.25)
    first = np.repeat(np.arange(4), 4)
    second = np.tile(np.arange(4), 4)
    interval = crossed_cluster_interval(values, first, second, draws=100, seed=7)
    np.testing.assert_allclose(interval, (-0.25, -0.25))


def test_task_bootstrap_interval() -> None:
    interval = task_bootstrap_interval(np.full(5, -0.1), draws=100, seed=7)
    np.testing.assert_allclose(interval, (-0.1, -0.1))
