from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import analyze_fusion as fusion  # noqa: E402
import analyze_graph_stage2 as graph  # noqa: E402
import analyze_stage2 as stage2  # noqa: E402


def test_scalar_empirical_midrank_handles_ties_and_nan() -> None:
    reference = np.asarray([0.0, 1.0, 1.0, 2.0, np.nan])
    assert stage2.empirical_rank(reference, 1.0) == 0.5
    assert np.isnan(stage2.empirical_rank(reference, np.nan))


def test_vector_empirical_ranks_match_columnwise_definition() -> None:
    reference = np.asarray([[0.0, 3.0], [1.0, 2.0], [1.0, np.nan], [2.0, 1.0]])
    current = np.asarray([[1.0, 2.0], [2.0, 0.0]])
    actual = graph.empirical_ranks(reference, current)
    expected = np.asarray([[0.5, 0.5], [0.875, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected)


def test_response_cycle_uses_only_supplied_prefix() -> None:
    output: dict[str, float] = {}
    stage2.add_response_cycle_features(
        output, np.asarray([0.1, 0.8, 0.9, 0.4, 0.2])
    )
    assert output["mobility_high_fraction"] == 0.4
    assert output["mobility_burst_count_norm"] == 0.2
    assert output["mobility_center_cross_rate"] == 0.5
    assert output["mobility_recent_highness"] == 0.6


def test_transition_scores_are_terminal_minus_alarm() -> None:
    keys = {
        "cohort": ["external_8b"],
        "row": [7],
        "task": ["suite/task"],
        "suite": ["suite"],
        "episode": [3],
        "outcome": ["late"],
    }
    alarm = pd.DataFrame(
        {
            **keys,
            "conditional_front_d1": [0.2],
            "conditional_gap_d1": [0.3],
            "graph_homeostasis": [0.4],
            "route_activity_w4": [0.1],
            "front_minus_back_w4": [0.5],
        }
    )
    terminal = pd.DataFrame(
        {
            **keys,
            "conditional_front_d1": [0.8],
            "conditional_gap_d1": [0.7],
            "graph_homeostasis": [0.6],
            "route_activity_w4": [0.5],
            "front_minus_back_w4": [0.1],
        }
    )
    output = fusion.transition_scores(alarm, terminal).iloc[0]
    assert np.isclose(output["graph_conditional_release"], 0.6)
    assert np.isclose(output["route_activity_release"], 0.4)
    assert np.isclose(output["layer_back_handoff"], 0.4)
    assert np.isclose(output["restart_graph_route_layer_equal"], 1.4 / 3.0)
