import numpy as np
import pytest

from route_noise_selector import (
    N_EXPERTS,
    RidgeRouteHead,
    RouteNoiseController,
    centrality,
    early_route_descriptors,
    hellinger_embedding,
    late_route_centrality_target,
    normalize_router_probabilities,
    pairwise_hellinger,
    pool_standardize,
    route_centrality_scores,
    stable_argmin,
)


def _one_hot(expert: int) -> np.ndarray:
    result = np.zeros(N_EXPERTS, dtype=np.float64)
    result[expert] = 1.0
    return result


def test_normalization_is_per_site_and_exact():
    raw = np.arange(1, 1 + 2 * 3 * N_EXPERTS, dtype=np.float64).reshape(
        2, 3, N_EXPERTS
    )
    normalized = normalize_router_probabilities(raw)
    assert np.allclose(normalized.sum(axis=-1), 1.0, atol=2e-12, rtol=0.0)
    assert np.allclose(normalized[0, 0] / normalized[0, 0, 0], np.arange(1, 33))


def test_normalization_rejects_invalid_mass_and_negative_values():
    with pytest.raises(ValueError, match="positive probability mass"):
        normalize_router_probabilities(np.zeros((2, N_EXPERTS)))
    bad = np.ones((2, N_EXPERTS))
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative"):
        normalize_router_probabilities(bad)


def test_hellinger_distance_has_known_endpoints():
    routes = np.stack([_one_hot(0), _one_hot(1)])[:, None, :]
    assert hellinger_embedding(routes).shape == (2, N_EXPERTS)
    distance = pairwise_hellinger(routes)
    assert np.isclose(distance[0, 1], 1.0)
    assert np.allclose(np.diag(distance), 0.0)


def test_route_selection_is_candidate_permutation_equivariant():
    probabilities = np.full((4, 2, 3, 2, N_EXPERTS), 1e-6)
    for candidate, expert in enumerate((0, 0, 1, 5)):
        probabilities[candidate, ..., expert] = 1.0
    ids = np.array([40, 10, 30, 20])
    scores = route_centrality_scores(probabilities)
    selected = stable_argmin(scores, ids)
    permutation = np.array([2, 0, 3, 1])
    permuted = stable_argmin(scores[permutation], ids[permutation])
    assert selected == permuted == 10


def test_centrality_validates_distance_geometry():
    with pytest.raises(ValueError, match="symmetric"):
        centrality(np.array([[0.0, 1.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        centrality(np.ones((2, 2)))


def test_descriptors_keep_layers_separate_and_target_uses_only_future():
    routes = np.full((3, 2, 5, 2, N_EXPERTS), 1e-5)
    routes[..., 0] = 1.0
    routes[1, 0, :3, :, 1] = 2.0
    routes[2, 1, 3:, :, 2] = 3.0
    descriptor = early_route_descriptors(routes, prefix_steps=3)
    # global centrality plus four descriptors for each of two layers
    assert descriptor.shape == (3, 1 + 4 * 2)
    target = late_route_centrality_target(routes, prefix_steps=3)
    routes_changed_early = routes.copy()
    routes_changed_early[:, :, :3] = np.roll(routes_changed_early[:, :, :3], 4, -1)
    assert np.array_equal(
        target, late_route_centrality_target(routes_changed_early, prefix_steps=3)
    )


def test_pool_standardization_and_ridge_head_fit_toy_relation():
    raw = np.arange(24, dtype=np.float64).reshape(8, 3)
    features = pool_standardize(raw)
    assert np.allclose(features.mean(axis=0), 0.0)
    assert np.allclose(features.std(axis=0), 1.0)
    x = np.column_stack([np.linspace(-1.0, 1.0, 20), np.ones(20)])
    target = 2.0 * x[:, 0] + 0.5
    head = RidgeRouteHead.fit(x, target, alpha=1e-8)
    assert np.allclose(head.predict(x), target, atol=1e-7)


def test_controller_uses_only_configured_early_prefix_and_loads_artifact(tmp_path):
    rng = np.random.default_rng(9)
    routes = rng.random((8, 2, 6, 3, N_EXPERTS))
    feature_width = 1 + 4 * 2
    path = tmp_path / "head.npz"
    np.savez_compressed(
        path,
        future_route_head__coefficients=np.linspace(-0.2, 0.2, feature_width + 1),
        future_route_head__feature_mean=np.zeros(feature_width),
        future_route_head__feature_scale=np.ones(feature_width),
        future_route_head__alpha=np.asarray(1.0),
    )
    controller = RouteNoiseController.from_npz(
        path, method="future_route_head", prefix_steps=3
    )
    score = controller.score(routes)
    changed_future = routes.copy()
    changed_future[:, :, 3:] = rng.random(changed_future[:, :, 3:].shape)
    assert np.allclose(score, controller.score(changed_future))

    permutation = np.array([5, 1, 7, 0, 3, 6, 2, 4])
    ids = np.arange(100, 108)
    selected = controller.select(routes, candidate_ids=ids)
    assert controller.select(
        routes[permutation], candidate_ids=ids[permutation]
    ) == selected
