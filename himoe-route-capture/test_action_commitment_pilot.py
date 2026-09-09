import numpy as np

from analyze_action_commitment_pilot import (
    central_subset,
    exact_random_subset_metrics,
    idealized_saving,
    medoid_index,
    pairwise_rms,
    route_prefix_distances,
    stabilized_stage,
)


def test_pairwise_rms_and_medoid_are_stable():
    values = np.array([[0.0], [1.0], [3.0]])
    distance = pairwise_rms(values)
    assert np.allclose(distance, np.abs(values - values.T))
    assert medoid_index(distance) == 1
    assert np.array_equal(central_subset(distance, 2), np.array([1, 0]))


def test_exact_random_subset_metrics_enumerates_every_subset():
    values = np.arange(4, dtype=np.float64)[:, None]
    metrics = exact_random_subset_metrics(pairwise_rms(values), 2)
    assert np.isclose(metrics["full_medoid_retained"], 0.5)
    assert 0.0 <= metrics["selected_equals_full_medoid"] <= 1.0
    assert metrics["normalized_global_medoid_regret"] >= 0.0


def test_route_prefix_distance_accumulates_observed_rounds():
    probability = np.zeros((2, 1, 2, 1, 32), dtype=np.float64)
    probability[0, 0, :, 0, 0] = 1.0
    probability[1, 0, 0, 0, 1] = 1.0
    probability[1, 0, 1, 0, 0] = 1.0
    distance = route_prefix_distances(probability)
    assert distance.shape == (2, 2, 2)
    assert distance[0, 0, 1] > 0.0
    assert np.isclose(distance[1, 0, 1], distance[0, 0, 1] / 2.0)


def test_stabilized_stage_requires_no_later_flip():
    assert stabilized_stage(np.array([False, True, False, True, True])) == 3


def test_idealized_saving_for_k16_to_k8():
    assert np.isclose(idealized_saving(16, 8, 1, 10), 0.45)
    assert np.isclose(idealized_saving(16, 8, 3, 10), 0.35)
    assert np.isclose(idealized_saving(16, 8, 10, 10), 0.0)
