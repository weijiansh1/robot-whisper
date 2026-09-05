from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluate_risk_recovery_judge import crossfit_scores, joint_policy_metrics
from risk_recovery_monitor import (
    COMPONENT_NAMES,
    RecoveryProfile,
    allocation_state,
    quantile_higher,
    terminal_components,
)


def test_terminal_components_ignore_current_and_future_actions() -> None:
    state = np.zeros((7, 8), dtype=np.float32)
    state[:, 0] = np.arange(7, dtype=np.float32) * 0.01
    actions = np.zeros((7, 10, 7), dtype=np.float32)
    actions[:, :, 0] = 0.02
    route = {
        "route_mobility": 0.03,
        "front_feedback_split": -0.1,
        "lag_recurrence": 0.8,
    }

    expected = terminal_components(state, actions, route, query=4)
    modified_state = state.copy()
    modified_actions = actions.copy()
    modified_state[5:] = 1000.0
    modified_actions[4:] = 1000.0
    observed = terminal_components(modified_state, modified_actions, route, query=4)

    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_allclose(
        expected,
        np.asarray((0.01, 0.01, 0.5, 0.03, -0.1, -0.8)),
        rtol=1e-5,
        atol=1e-6,
    )


def test_recovery_profile_is_an_unweighted_empirical_rank_mean() -> None:
    base = np.linspace(0.0, 1.0, 40, dtype=np.float32)
    reference = np.repeat(base[:, None], len(COMPONENT_NAMES), axis=1)
    profile = RecoveryProfile(reference)
    values = np.repeat(
        np.asarray((0.25, 0.75), dtype=np.float32)[:, None],
        len(COMPONENT_NAMES),
        axis=1,
    )

    scores = profile.scores(values)

    assert scores[0] < scores[1]
    assert profile.score(values[1]) == pytest.approx(scores[1])
    assert scores[0] == pytest.approx(0.25, abs=0.03)
    assert scores[1] == pytest.approx(0.75, abs=0.03)


def test_quantiles_and_allocation_states_are_ordered() -> None:
    values = np.arange(40, dtype=np.float32)
    q75 = quantile_higher(values, 0.75)
    q90 = quantile_higher(values, 0.90)

    assert allocation_state(q90, q75, q90) == "strong_extend"
    assert allocation_state(q75, q75, q90) == "extend_watch"
    assert allocation_state(q75 - 1.0, q75, q90) == "intervene_first"


def test_crossfit_scores_exclude_the_held_out_initial_state() -> None:
    count = 80
    components = np.repeat(
        np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None],
        len(COMPONENT_NAMES),
        axis=1,
    )
    cache = {
        "task_names": np.asarray(["libero_spatial/example"]),
        "task_index": np.zeros(count, dtype=np.int16),
        "init_state_id": np.repeat(np.arange(2), count // 2).astype(np.int16),
        "length": np.full(count, 10, dtype=np.int16),
    }

    scores = crossfit_scores(cache, components)

    assert np.isfinite(scores).all()
    assert scores[0] < scores[-1]


def test_safety_policy_protects_late_rows_selected_by_either_head() -> None:
    table = pd.DataFrame(
        {
            "cohort": ["test"] * 4,
            "original_failure": [True] * 4,
            "late_success_plus10_queries": [True, True, False, False],
            "failure": [False, False, True, True],
            "allocation": [
                "extend_watch",
                "intervene_first",
                "extend_watch",
                "intervene_first",
            ],
            "hard_alarm": [False, True, False, False],
            "late_success_first_extra_query": [2, 3, np.nan, np.nan],
        }
    )

    metrics = joint_policy_metrics(table)
    safety = metrics[metrics["policy"] == "safety_hard_or_active_q75"].iloc[0]

    assert safety["late_recovered"] == 2
    assert safety["late_recall"] == 1.0
    assert safety["persistent_intervened"] == 1
