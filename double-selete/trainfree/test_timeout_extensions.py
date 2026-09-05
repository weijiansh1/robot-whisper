from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import continue_timeout_failures as continuation  # noqa: E402
import prepare_timeout_extensions as preparation  # noqa: E402
import summarize_timeout_extensions as summarization  # noqa: E402


def test_next_flow_noise_matches_sequential_generator() -> None:
    seed = 1007
    source_queries = 30
    expected_rng = np.random.default_rng(seed)
    expected = [
        expected_rng.standard_normal(continuation.FLOW_NOISE_SHAPE).astype(np.float32)
        for _ in range(source_queries + 3)
    ][source_queries:]
    actual = continuation.next_flow_noises(seed, source_queries, 3)
    assert len(actual) == len(expected)
    for observed, reference in zip(actual, expected):
        np.testing.assert_array_equal(observed, reference)


def test_suite_horizons_are_whole_action_chunks() -> None:
    assert preparation.SUITE_MAX_ACTION_STEPS == {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_long": 520,
    }
    assert all(value % 10 == 0 for value in preparation.SUITE_MAX_ACTION_STEPS.values())


def test_result_paths_are_unique_across_cohorts(tmp_path: Path) -> None:
    base = {
        "case_id": "0001",
        "server_suite": "goal",
        "task_id": "3",
        "episode": "15",
    }
    main_json, main_npz = continuation.result_paths(
        tmp_path, {**base, "cohort": "development_main"}
    )
    external_json, external_npz = continuation.result_paths(
        tmp_path, {**base, "cohort": "external_8b"}
    )
    assert main_json != external_json
    assert main_npz != external_npz
    assert main_json.suffix == ".json"
    assert main_npz.suffix == ".npz"


def test_summaries_report_one_and_ten_query_windows(tmp_path: Path) -> None:
    rows = [
        {
            "cohort": "development_main",
            "server_suite": "spatial",
            "task": "libero_spatial/example",
            "success_after_extension": True,
            "success_within_one_extra_query": False,
            "first_success_extra_query": 3,
        },
        {
            "cohort": "external_8b",
            "server_suite": "spatial",
            "task": "libero_spatial/example",
            "success_after_extension": True,
            "success_within_one_extra_query": True,
            "first_success_extra_query": 1,
        },
    ]

    cohorts, curve, suites, tasks = summarization.summaries(tmp_path, rows)

    all_spatial = next(
        row
        for row in suites
        if row["cohort"] == "all" and row["suite"] == "spatial"
    )
    assert all_spatial["late_successes_plus10_queries"] == 2
    assert all_spatial["late_successes_plus1_query"] == 1
    assert curve[0]["maximum_extra_low_level_actions"] == 10
    assert cohorts[0]["remaining_failures"] == 0
    assert tasks[0]["late_successes_plus1_query"] == 0
