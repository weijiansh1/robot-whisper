import numpy as np

from analyze_markov_routing import (
    fit_markov_models,
    mean_run_length,
    seed_disjoint_folds,
    sequence_log_probabilities,
    stationary_distribution,
    stratified_auc,
    stratified_auc_details,
    transition_phases,
)


def test_seed_folds_hold_out_each_seed_exactly_once():
    seeds = np.tile(np.arange(8), 3)
    masks = seed_disjoint_folds(seeds, folds=4, random_seed=7)
    assert np.array_equal(np.sum(masks, axis=0), np.ones(len(seeds)))
    for seed in np.unique(seeds):
        memberships = [np.unique(mask[seeds == seed]).tolist() for mask in masks]
        assert sum(items == [True] for items in memberships) == 1


def test_transition_counts_do_not_cross_episode_boundaries():
    sequences = np.array([[0, 1, 1], [1, 0, 0]])
    models = fit_markov_models(sequences, clusters=2)
    assert np.array_equal(models.transition_counts, [[1, 1], [1, 1]])


def test_first_order_model_beats_marginal_on_deterministic_chain():
    sequences = np.tile([0, 1, 0, 1, 0, 1], (40, 1))
    models = fit_markov_models(sequences[:30], clusters=2, alpha=1.0)
    logp = sequence_log_probabilities(models, sequences[30:])
    assert -logp["first"].mean() < 0.1
    assert -logp["zero"].mean() > 0.9


def test_second_order_model_detects_history_missing_from_first_order():
    base = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    sequences = np.stack([np.roll(base, shift) for shift in range(3) for _ in range(30)])
    models = fit_markov_models(sequences[:60], clusters=2, alpha=1.0)
    logp = sequence_log_probabilities(models, sequences[60:])
    assert -logp["second"].mean() < -logp["first"].mean() - 0.3


def test_stationary_distribution_and_run_length():
    matrix = np.array([[0.9, 0.1], [0.2, 0.8]])
    stationary = stationary_distribution(matrix)
    assert np.allclose(stationary, [2.0 / 3.0, 1.0 / 3.0], atol=1e-10)
    assert mean_run_length(np.array([[0, 0, 1, 1], [1, 0, 0, 0]])) == 2.0


def test_transition_phases_cover_three_ordered_segments():
    assert np.array_equal(transition_phases(7), [0, 0, 1, 1, 2, 2])


def test_stratified_auc_ignores_between_group_score_offsets():
    labels = np.array([False, True, False, True])
    groups = np.array([0, 0, 1, 1])
    scores = np.array([100.0, 101.0, -100.0, -99.0])
    assert stratified_auc(scores, labels, groups) == 1.0
    assert stratified_auc_details(scores, labels, groups) == {
        "auc": 1.0,
        "pair_count": 2,
        "valid_groups": 2,
    }
