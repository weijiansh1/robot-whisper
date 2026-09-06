import numpy as np

from moe_grammar.semantics import (
    OUTCOMES,
    InterventionTable,
    SemanticSuffixTrie,
    label_outcomes,
)


def test_longer_context_wins_when_it_has_support() -> None:
    trie = SemanticSuffixTrie(max_order=2, min_support=10, alpha=1.0)
    # After word 7 the sentence returns; after 9 it persists. The root mixes both.
    trie.fit([(np.asarray([1, 7]), "return") for _ in range(40)])
    trie.fit([(np.asarray([1, 9]), "persist") for _ in range(40)])
    after_seven, order_seven, support = trie.posterior(np.asarray([1, 7]))
    assert order_seven >= 1 and support >= 40
    assert after_seven[OUTCOMES.index("return")] > 0.9
    after_nine, _, _ = trie.posterior(np.asarray([1, 9]))
    assert after_nine[OUTCOMES.index("persist")] > 0.9


def test_unsupported_context_backs_off_to_the_root() -> None:
    trie = SemanticSuffixTrie(max_order=3, min_support=50, alpha=1.0)
    trie.fit([(np.asarray([2, 3]), "return") for _ in range(60)])
    # A context never seen at high order must fall back rather than invent a semantics.
    posterior, order, support = trie.posterior(np.asarray([41, 42]))
    assert order == 0 and support == 60
    np.testing.assert_allclose(posterior, trie.posterior(np.asarray([]))[0])


def test_probabilities_are_normalized_and_prior_dominates_empty_counts() -> None:
    trie = SemanticSuffixTrie(max_order=1, min_support=1, alpha=1.0)
    posterior, order, support = trie.posterior(np.asarray([5]))
    assert order == 0 and support == 0
    np.testing.assert_allclose(posterior.sum(), 1.0)
    np.testing.assert_allclose(posterior, np.full(len(OUTCOMES), 1.0 / len(OUTCOMES)))


def test_label_outcomes_prefers_episode_end_over_return() -> None:
    deviation = np.asarray([0.9, 0.9, 0.2, 0.9], dtype=np.float32)
    labels = dict(label_outcomes(deviation, 0, 4, horizon=1, success=True))
    # Position 0 deviates and drops below threshold two steps later, which is outside a
    # horizon of one, so it persists; position 3 ends the episode.
    assert labels[0] == "persist"
    assert labels[1] == "return"
    assert labels[3] == "end_success"
    assert 2 not in labels  # not deviating, so it carries no return question


def test_intervention_table_blocks_by_trunk() -> None:
    table = InterventionTable()
    for trunk in ("a", "b", "c", "d"):
        for _ in range(4):
            table.add(trunk, "control", False)
    for trunk, wins in (("a", 3), ("b", 0), ("c", 1), ("d", 2)):
        for index in range(4):
            table.add(trunk, "triggered", index < wins)
    assert table.arms == ("control", "triggered")
    assert table.posterior_mean("control") < table.posterior_mean("triggered")
    interval = table.trunk_blocked_interval("control")
    assert interval is not None and interval[1] < 0.3
    observed, (low, high) = table.contrast_interval("triggered", "control")
    assert observed > 0.3 and low < observed < high
    # A single trunk cannot support a blocked interval.
    single = InterventionTable()
    single.add("only", "triggered", True)
    assert single.trunk_blocked_interval("triggered") is None
