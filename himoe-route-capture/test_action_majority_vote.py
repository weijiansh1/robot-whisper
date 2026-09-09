"""Unit tests for the action-space majority-voting selectors."""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from analyze_action_majority_vote import (
    ENUMERATION_LIMIT,
    centroid_choice,
    exact_expected_maximum,
    medoid_choice,
    normalized_actions,
    subset_index,
)
from analyze_action_vote_cross_task import task_macro_bootstrap


def test_exact_expected_maximum_matches_brute_force_enumeration():
    rng = np.random.default_rng(7)
    values = rng.normal(size=9)
    for budget in range(1, 10):
        brute = np.mean([values[list(subset)].max() for subset in combinations(range(9), budget)])
        assert exact_expected_maximum(values, budget) == pytest.approx(brute)


def test_exact_expected_maximum_handles_ties():
    values = np.array([1.0, 1.0, 1.0, 2.0])
    # Only subsets containing the single 2.0 attain it: C(3,1)/C(4,2) = 3/6.
    assert exact_expected_maximum(values, 2) == pytest.approx(0.5 * 1.0 + 0.5 * 2.0)


def test_medoid_choice_picks_the_geometric_centre():
    # Candidate 1 sits between 0 and 2, so it is the medoid of the full triple.
    distance = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    chosen = medoid_choice(distance, np.array([[0, 1, 2]]))
    assert chosen.tolist() == [1]


def test_medoid_choice_breaks_ties_on_the_lowest_candidate_id():
    # Every pair is a tie, so the lower id must always win.
    distance = 1.0 - np.eye(4)
    subsets = np.asarray(list(combinations(range(4), 2)), dtype=np.int64)
    chosen = medoid_choice(distance, subsets)
    assert chosen.tolist() == [pair[0] for pair in combinations(range(4), 2)]


def test_centroid_choice_selects_the_nearest_neighbour_of_the_subset_mean():
    normalized = np.zeros((3, 2, 7), dtype=np.float64)
    normalized[0, :, 0] = -1.0
    normalized[1, :, 0] = 0.25
    normalized[2, :, 0] = 1.0
    chosen = centroid_choice(normalized, np.array([[0, 1, 2]]))
    assert chosen.tolist() == [1]


def test_centroid_choice_agrees_with_a_direct_computation():
    rng = np.random.default_rng(11)
    normalized = rng.normal(size=(6, 4, 7))
    subsets = np.asarray(list(combinations(range(6), 3)), dtype=np.int64)
    chosen = centroid_choice(normalized, subsets)
    for row, subset in zip(chosen, subsets):
        values = normalized[subset]
        centroid = values.mean(axis=0)
        distance = np.linalg.norm(values - centroid, axis=-1).mean(axis=-1)
        assert row == subset[int(np.argmin(distance))]


def test_subset_index_enumerates_exhaustively_below_the_limit():
    subsets, exhaustive = subset_index(8, 3, draws=100, seed=1)
    assert exhaustive
    assert len(subsets) == comb(8, 3)
    assert {tuple(row) for row in subsets} == set(combinations(range(8), 3))


def test_subset_index_falls_back_to_sorted_monte_carlo_subsets():
    subsets, exhaustive = subset_index(32, 8, draws=64, seed=3)
    assert not exhaustive
    assert subsets.shape == (64, 8)
    assert comb(32, 8) > ENUMERATION_LIMIT
    for row in subsets:
        assert len(set(row.tolist())) == 8
        assert list(row) == sorted(row)


def test_subset_index_is_deterministic_for_a_fixed_seed():
    left, _ = subset_index(32, 8, draws=32, seed=5)
    right, _ = subset_index(32, 8, draws=32, seed=5)
    np.testing.assert_array_equal(left, right)


def test_subset_index_rejects_budgets_outside_the_pool():
    with pytest.raises(ValueError):
        subset_index(8, 9, draws=10, seed=1)
    with pytest.raises(ValueError):
        subset_index(8, 1, draws=10, seed=1)


def test_normalized_actions_applies_the_gripper_weight_to_the_last_dimension():
    actions = np.ones((2, 3, 7), dtype=np.float64)
    std = np.full(7, 2.0)
    result = normalized_actions(actions, std, 0.25)
    assert result[..., :6] == pytest.approx(0.5)
    assert result[..., 6] == pytest.approx(0.125)


@pytest.mark.parametrize(
    "std, weight",
    [(np.zeros(7), 0.25), (np.ones(6), 0.25), (np.ones(7), -1.0), (np.ones(7), np.nan)],
)
def test_normalized_actions_rejects_invalid_scaling(std, weight):
    with pytest.raises(ValueError):
        normalized_actions(np.ones((2, 3, 7)), std, weight)


def test_normalized_actions_rejects_wrong_chunk_shape():
    with pytest.raises(ValueError):
        normalized_actions(np.ones((2, 3, 6)), np.ones(7), 0.25)


def test_task_macro_bootstrap_collapses_on_constant_effects():
    effects = [np.full(4, 0.02), np.full(4, -0.01)]
    result = task_macro_bootstrap(effects, draws=200, seed=1)
    assert result["mean"] == pytest.approx(0.005)
    assert result["lower"] == pytest.approx(0.005)
    assert result["upper"] == pytest.approx(0.005)


def test_task_macro_bootstrap_weights_tasks_equally_not_states():
    # The large task must not dominate: the macro mean is the mean of task means.
    effects = [np.full(100, 1.0), np.full(2, 0.0)]
    result = task_macro_bootstrap(effects, draws=200, seed=2)
    assert result["mean"] == pytest.approx(0.5)
