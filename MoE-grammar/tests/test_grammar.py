import numpy as np

from moe_grammar.grammar import (
    CountGrammar,
    DurationGrammar,
    DurationModel,
    PositionContextGrammar,
    PositionGrammar,
)
from moe_grammar.run_dual_axis_audit import axis_groups, crossed_cluster_interval
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
