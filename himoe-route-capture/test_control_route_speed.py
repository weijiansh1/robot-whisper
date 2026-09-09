import numpy as np

from analyze_control_route_speed import (
    N_DENOISE,
    N_EXPERTS,
    N_LAYERS,
    normalize_probabilities,
    route_distance_matrix,
    summarize_episode,
    transition_phase,
)


def _constant_route(expert: int) -> np.ndarray:
    route = np.zeros((N_LAYERS, N_DENOISE, N_EXPERTS), dtype=np.float64)
    route[..., expert] = 1.0
    return route


def test_normalization_is_per_router_site():
    values = np.arange(1, 1 + 2 * N_EXPERTS, dtype=np.float64).reshape(
        2, N_EXPERTS
    )
    normalized, bounds = normalize_probabilities(values)
    assert np.allclose(normalized.sum(axis=-1), 1.0)
    assert bounds == (float(values[0].sum()), float(values[1].sum()))


def test_product_hellinger_has_expected_endpoints():
    routes = np.stack([_constant_route(0), _constant_route(0), _constant_route(1)])
    distance = route_distance_matrix(routes)
    assert np.allclose(np.diag(distance), 0.0)
    assert distance[0, 1] == 0.0
    assert np.isclose(distance[0, 2], 1.0)
    assert np.isclose(distance[1, 2], 1.0)


def test_episode_summary_uses_all_control_queries_in_order():
    routes = np.stack(
        [
            _constant_route(0),
            _constant_route(0),
            _constant_route(1),
            _constant_route(1),
        ]
    )
    result = summarize_episode(routes, "goal", "toy", 0)
    assert np.allclose(result.adjacent, [0.0, 1.0, 0.0])
    assert np.isclose(result.lag_mean[0], 1.0 / 3.0)
    assert np.isclose(result.lag_mean[1], 1.0)
    assert result.controls == 4


def test_transition_phase_spans_three_ordered_parts():
    assert np.array_equal(transition_phase(7), [0, 0, 1, 1, 2, 2])
