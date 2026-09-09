from dataclasses import replace

import numpy as np

from analyze_oracle_gate import (
    constant_score_control,
    gate_pass,
    is_ceiling_task,
    validate_pools,
)
from analyze_self_supervised_noise_methods import Pool


def make_pool(task: str, state: int, fold: int, success=None) -> Pool:
    candidates = 8
    if success is None:
        success = np.arange(candidates) % 2 == 0
    zeros = np.zeros(candidates)
    return Pool(
        task=task,
        suite="toy",
        state=state,
        fold=fold,
        seeds=np.arange(fold, 32, 4),
        route_features=np.zeros((candidates, 2)),
        noise_features=np.zeros((candidates, 2)),
        action_residual=np.zeros((candidates, 2)),
        action_centrality=zeros,
        late_route_centrality=zeros,
        future_route_change=zeros,
        early_route_score=zeros,
        initial_noise_score=zeros,
        prefix_contraction=zeros,
        cross_layer_agreement=zeros,
        success=np.asarray(success, dtype=bool),
    )


def make_task(task: str) -> list[Pool]:
    return [make_pool(task, state, fold) for state in range(16) for fold in range(4)]


def test_gate_is_strictly_greater_than_threshold():
    assert not gate_pass(0.03, 0.03)
    assert gate_pass(0.0300001, 0.03)


def test_ceiling_rule_uses_candidate_outcomes():
    pools = make_task("ceiling")
    assert not is_ceiling_task(pools)
    all_success = [replace(pool, success=np.ones(8, dtype=bool)) for pool in pools]
    assert is_ceiling_task(all_success)


def test_constant_negative_control_is_exact():
    result = constant_score_control(make_task("task"))
    assert result["candidate_score_max_range"] == 0.0
    assert result["mean_pool_auc"] == 0.5
    assert result["evaluable_auc_pools"] == 64


def test_seed_validation_rejects_overlap_and_accepts_grid():
    tasks = {"a": make_task("a"), "b": make_task("b")}
    result = validate_pools(tasks)
    assert result["unique_seed_ids"] == 32
    broken = make_task("broken")
    broken[1] = replace(broken[1], seeds=broken[0].seeds.copy())
    try:
        validate_pools({"broken": broken})
    except ValueError as error:
        assert "fold changed" in str(error)
    else:
        raise AssertionError("overlapping or inconsistent folds must fail")
