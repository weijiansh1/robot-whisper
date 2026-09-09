import math

import numpy as np

from analyze_route_prune_vote import (
    action_distance,
    central_subset,
    exact_random_subset_metrics,
    flow_compute_fraction,
    medoid_index,
    normalized_route_distributions,
    prefix_route_distances,
    random_retention_probability,
)


def test_action_distance_is_checkpoint_normalized_rms():
    actions = np.zeros((3, 2, 2), dtype=np.float64)
    actions[1, :, 0] = 2.0
    actions[2, :, 1] = 4.0
    distance = action_distance(actions, np.array([2.0, 4.0]))

    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.isclose(distance[0, 1], math.sqrt(0.5))
    assert np.isclose(distance[0, 2], math.sqrt(0.5))


def test_route_prefix_tv_preserves_flow_alignment_and_normalizes_weights():
    # [candidate=2, flow=2, layer=1, action=1, top-k=1]
    ids = np.array([[[[[0]]], [[[0]]]], [[[[1]]], [[[0]]]]], dtype=np.int16)
    weights = np.array([[[[[2.0]]], [[[3.0]]]], [[[[7.0]]], [[[5.0]]]]])
    dense = normalized_route_distributions(ids, weights, expert_count=2)
    prefix = prefix_route_distances(ids, weights, expert_count=2)

    assert np.allclose(dense.sum(axis=-1), 1.0)
    assert prefix.shape == (2, 2, 2)
    assert np.isclose(prefix[0, 0, 1], 1.0)
    assert np.isclose(prefix[1, 0, 1], 0.5)


def test_medoid_ties_use_lowest_global_candidate_id():
    distance = np.ones((4, 4)) - np.eye(4)
    assert medoid_index(distance, [3, 1]) == 1
    assert np.array_equal(central_subset(distance, 2), np.array([0, 1]))


def test_central_subset_uses_same_tolerance_for_near_ties():
    epsilon = 5e-13
    distance = np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0 - epsilon], [1.0, 1.0 - epsilon, 0.0]]
    )
    assert np.array_equal(central_subset(distance, 1), np.array([0]))


def test_exact_random_subset_retention_matches_combinatorics():
    positions = np.arange(8, dtype=np.float64)[:, None]
    distance = pair_distance = np.abs(positions - positions.T)
    full_winner = medoid_index(distance)
    top2 = np.argsort(distance.sum(axis=1), kind="stable")[:2]
    scale = np.median(pair_distance[np.triu_indices(8, 1)])
    metrics = exact_random_subset_metrics(distance, 4, full_winner, top2, scale)

    assert np.isclose(metrics["full_winner_retained"], 0.5)
    assert np.isclose(
        metrics["any_full_top2_retained"],
        random_retention_probability(8, 4, 2),
    )
    assert np.isclose(metrics["any_full_top2_retained"], 11.0 / 14.0)


def test_idealized_flow_compute_fraction():
    assert np.isclose(flow_compute_fraction(8, 6, 1, 10), 0.775)
    assert np.isclose(flow_compute_fraction(8, 4, 1, 10), 0.55)
    assert np.isclose(flow_compute_fraction(8, 2, 1, 10), 0.325)
    assert np.isclose(flow_compute_fraction(8, 4, 0, 10), 0.5)
