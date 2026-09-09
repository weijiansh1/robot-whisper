from __future__ import annotations

import numpy as np

from analyze_moe_opportunity import (
    hierarchical_bootstrap_ci,
    mean_upper,
    pool_features,
    task_held_out_ridge,
    task_macro_spearman,
)


def test_pool_features_keep_state_token_candidate_invariance() -> None:
    rng = np.random.default_rng(4)
    routes = rng.uniform(0.01, 1.0, size=(32, 8, 10, 11, 32))
    routes[:, :, :, 0] = routes[0, :, :, 0]
    action, state, diagnostics = pool_features(routes)
    assert action.shape == (5,)
    assert state.shape == (24,)
    assert diagnostics["state_token_candidate_max_hellinger"] < 1e-7
    assert diagnostics["action_route_dispersion_full"] > 0


def test_task_macro_spearman_ignores_constant_task() -> None:
    score = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=float)
    target = np.asarray([0, 1, 2, 3, 1, 1, 1, 1], dtype=float)
    tasks = np.asarray(["a"] * 4 + ["b"] * 4)
    macro, per_task = task_macro_spearman(score, target, tasks)
    assert macro == 1.0
    assert per_task == {"a": 1.0}


def test_task_held_out_ridge_and_bootstrap_are_finite() -> None:
    features = np.asarray([[0], [1], [2], [3], [4], [5]], dtype=float)
    target = np.asarray([0, 1, 2, 3, 4, 5], dtype=float)
    tasks = np.asarray(["a", "a", "b", "b", "c", "c"])
    result = task_held_out_ridge(features, target, tasks, alpha=1.0)
    assert np.isfinite(result["r2"])
    assert np.isfinite(result["spearman"])
    ci = hierarchical_bootstrap_ci(
        features[:, 0], target, tasks, draws=20, rng=np.random.default_rng(9)
    )
    assert len(ci) == 2
    assert np.all(np.isfinite(ci))


def test_mean_upper() -> None:
    distance = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert mean_upper(distance) == 2.0
