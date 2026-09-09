from __future__ import annotations

from pathlib import Path

import numpy as np

from analysis_moe_execution_signals.analyze_horizon_recovery import (
    exact_joint_sign_flip_max_t,
    loop_onset_index,
    random_subset_targeting,
)
from analysis_moe_execution_signals.collect_horizon_recovery import (
    ALARM_THRESHOLD,
    _load_inputs,
    future_noise,
    horizon_at,
    split_pairs,
)


def test_short_burst_schedule_returns_to_released_horizon() -> None:
    assert [horizon_at("h2_burst10", offset) for offset in (0, 2, 4, 6, 8, 10, 20)] == [
        2,
        2,
        2,
        2,
        2,
        10,
        10,
    ]
    assert horizon_at("h10", 0) == horizon_at("h10", 190) == 10


def test_future_noise_is_common_by_pair_and_physical_offset() -> None:
    first, words = future_noise(7, 10)
    repeated, repeated_words = future_noise(7, 10)
    another, _ = future_noise(7, 20)
    np.testing.assert_array_equal(first, repeated)
    assert words == repeated_words == [20260905, 7, 10]
    assert not np.array_equal(first, another)


def test_frozen_pair_shards_cover_every_pair_once() -> None:
    pairs = [pair for pair in range(24) if pair != 4]
    shards = split_pairs(pairs, 3)
    assert [len(shard) for shard in shards] == [8, 8, 7]
    assert sorted(sum(shards, [])) == pairs


def test_frozen_alarm_counts_and_restore_eligibility() -> None:
    rows, _audit, features, _noise = _load_inputs(
        Path("analysis_moe_execution_signals/rich_event_plan.json"),
        Path("analysis_moe_execution_signals/rich_event_functional_32d/records.jsonl"),
        Path("analysis_moe_execution_signals/rich_event_row_features.npz"),
        Path("analysis_moe_execution_signals/rich_event_inputs.npz"),
        Path("analysis_moe_execution_signals/rich_event_inputs_audit.json"),
    )
    assert len(rows) == 46
    assert 4 not in {int(row["pair_id"]) for row in rows}
    event = [row for row in rows if row["role"] == "event"]
    control = [row for row in rows if row["role"] == "control"]
    assert sum(features[int(row["row_id"])] > ALARM_THRESHOLD for row in event) == 19
    assert sum(features[int(row["row_id"])] > ALARM_THRESHOLD for row in control) == 3


def test_exact_sign_flip_handles_ties_and_detects_consistent_effect() -> None:
    values = np.column_stack((np.ones(5), np.asarray([1, 1, 1, 0, 0], np.float64)))
    result = exact_joint_sign_flip_max_t(values, chunk_size=7)
    assert result["permutations"] == 32
    assert result["raw_p"][0] == 2 / 32
    assert 0.0 <= result["max_t_p"][0] <= 1.0
    zeros = exact_joint_sign_flip_max_t(np.zeros((4, 2)), chunk_size=3)
    assert zeros["raw_p"] == [1.0, 1.0]


def test_random_subset_control_uses_same_treatment_count() -> None:
    benefits = np.asarray([1, 0, -1, 1], np.float64)
    alarm = np.asarray([True, False, False, True])
    result = random_subset_targeting(benefits, alarm)
    assert result["selected"] == 2
    assert result["subsets"] == 6
    assert result["alarm_gated_success_gain"] == 0.5


def test_loop_rule_runs_on_a_fixed_grid() -> None:
    eef = np.asarray([[0, 0, 0], [0.06, 0, 0], [0.12, 0, 0], [0.0, 0, 0]], np.float64)
    objects = np.zeros((4, 2, 3), np.float64)
    gripper = np.zeros(4, np.float64)
    references = np.ones((1, 3), np.float64)
    assert loop_onset_index(eef, objects, gripper, references) == (3, 0)
