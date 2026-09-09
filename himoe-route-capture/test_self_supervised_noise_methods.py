from dataclasses import replace

import numpy as np

from analyze_self_supervised_noise_methods import (
    N_EXPERTS,
    RidgeModel,
    Pool,
    action_target,
    future_route_change_target,
    normalized_ranks,
    prefix_diagnostics,
    score_outer_fold,
)


def _route_sequence(experts):
    routes = np.zeros((1, 2, len(experts), 3, N_EXPERTS), dtype=np.float64)
    for step, expert in enumerate(experts):
        routes[:, :, step, :, expert] = 1.0
    return routes


def test_normalized_ranks_are_permutation_equivariant():
    values = np.asarray([3.0, 1.0, 2.0, 2.0])
    permutation = np.asarray([2, 0, 3, 1])
    expected = normalized_ranks(values)
    assert np.allclose(normalized_ranks(values[permutation]), expected[permutation])


def test_action_target_is_translation_and_scale_invariant_for_centrality():
    actions = np.zeros((3, 10, 7), dtype=np.float64)
    actions[1] = 1.0
    actions[2] = 4.0
    residual, central = action_target(actions, np.ones(7))
    _, transformed = action_target(5.0 * actions + 11.0, np.ones(7))
    assert residual.shape == (3, 70)
    assert np.allclose(central, transformed)
    assert central[1] < central[0] < central[2]


def test_future_change_uses_only_unfinished_prefix_suffix():
    steady = _route_sequence([0, 0, 0, 0, 0])
    moving = _route_sequence([0, 0, 0, 1, 2])
    routes = np.concatenate([steady, moving], axis=0)
    target = future_route_change_target(routes, prefix_steps=3)
    assert target[0] < target[1]
    changed_early = routes.copy()
    changed_early[:, :, :2] = np.roll(changed_early[:, :, :2], 5, axis=-1)
    assert np.allclose(target, future_route_change_target(changed_early, 3))


def test_prefix_diagnostics_detect_contraction_and_keep_layers_separate():
    routes = np.full((3, 2, 3, 2, N_EXPERTS), 1e-6)
    routes[..., 0] = 1.0
    routes[1, :, 1, :, 1] = 0.5
    routes[1, :, 2, :, 1] = 0.1
    routes[2, 0, :, :, 2] = 0.4
    contraction, agreement = prefix_diagnostics(routes, 3)
    assert contraction.shape == (3,)
    assert agreement.shape == (3,)
    assert contraction[1] < 0.0


def test_multioutput_ridge_fits_toy_mapping():
    x = np.column_stack([np.linspace(-1.0, 1.0, 30), np.ones(30)])
    y = np.column_stack([2.0 * x[:, 0] + 0.5, -x[:, 0] + 3.0])
    model = RidgeModel.fit(x, y, alpha=1e-8)
    assert np.allclose(model.predict(x), y, atol=1e-7)


def _toy_pool(task, state, fold, seed):
    rng = np.random.default_rng(seed)
    candidates = 8
    return Pool(
        task=task,
        suite="toy",
        state=state,
        fold=fold,
        seeds=np.arange(fold, 32, 4),
        route_features=rng.normal(size=(candidates, 5)),
        noise_features=rng.normal(size=(candidates, 6)),
        action_residual=rng.normal(size=(candidates, 7)),
        action_centrality=normalized_ranks(rng.normal(size=candidates)),
        late_route_centrality=normalized_ranks(rng.normal(size=candidates)),
        future_route_change=normalized_ranks(rng.normal(size=candidates)),
        early_route_score=rng.random(candidates),
        initial_noise_score=rng.random(candidates),
        prefix_contraction=rng.normal(size=candidates),
        cross_layer_agreement=rng.random(candidates),
        success=rng.random(candidates) > 0.5,
    )


def test_scoring_is_outcome_blind():
    train = [
        _toy_pool("task%d" % task, state, 0, 100 * task + state)
        for task in range(4)
        for state in range(2)
    ]
    test = [_toy_pool("held", state, 1, 900 + state) for state in range(2)]
    original, _ = score_outer_fold(train, test)
    flipped_train = [replace(pool, success=~pool.success) for pool in train]
    flipped_test = [replace(pool, success=~pool.success) for pool in test]
    flipped, _ = score_outer_fold(flipped_train, flipped_test)
    for method in original:
        for left, right in zip(original[method], flipped[method]):
            assert np.allclose(left, right)
