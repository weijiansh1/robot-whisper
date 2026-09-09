from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

from intrinsic_guard_monitor import (  # noqa: E402
    GlobalIntrinsicProfile, IntrinsicGuardMonitor, intrinsic_score_arrays, persistent_score,
)
from temporal_budget_guard import (  # noqa: E402
    SCHEMA, SCHEDULES, BoundarySchedule, TemporalGuardMonitor, TemporalProfile, TimeOnlyProfile,
    adjusted_score_arrays, calibrate_temporal_reference, calibrate_time_only, time_priorities,
)
from unlabeled_budget_calibration import alarms_from_scores, calibrate_reference  # noqa: E402
from test_unlabeled_budget import reference_features  # noqa: E402


@pytest.mark.parametrize("budget", [0.03, 0.1, 0.3])
def test_each_schedule_calibrates_the_actual_union_under_the_same_budget(budget: float) -> None:
    mobility, acceleration, periodicity, valid = reference_features()
    profiles, audit = calibrate_temporal_reference(mobility, acceleration, periodicity, valid, alarm_budget=budget)
    assert tuple(profiles) == SCHEDULES
    static = calibrate_reference(mobility, acceleration, periodicity, valid, alarm_budget=budget)
    assert profiles["constant"].base == static.profile
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, static.profile.periodicity_scale)
    for name, profile in profiles.items():
        adjusted = adjusted_score_arrays(scores, profile.schedule)
        first = alarms_from_scores(adjusted, valid, profile.base)
        variant = audit["variants"][name]
        assert (first["guard"] >= 0).sum() == variant["reference_replay_alarms"]
        assert variant["reference_replay_alarms"] <= variant["reference_alarm_capacity"]
        if name == "constant":
            for head, values in alarms_from_scores(scores, valid, static.profile).items():
                np.testing.assert_array_equal(values, first[head])


def test_padding_and_reference_order_do_not_change_calibration() -> None:
    features = reference_features()
    profiles, audit = calibrate_temporal_reference(*features)
    padded = [np.pad(values, ((0, 0), (0, 7), (0, 0)) if values.ndim == 3 else ((0, 0), (0, 7)),
                     constant_values=False if values.dtype == bool else np.inf) for values in features]
    assert calibrate_temporal_reference(*padded) == (profiles, audit)
    order = np.random.default_rng(84).permutation(len(features[0]))
    assert calibrate_temporal_reference(*(values[order] for values in features)) == (profiles, audit)


def test_shared_schedule_scales_come_only_from_reference() -> None:
    features = reference_features()
    profiles, _ = calibrate_temporal_reference(*features)
    for name, profile in profiles.items():
        schedule = profile.schedule
        assert schedule.tau_queries == 15
        assert replace(schedule, name="constant") == profiles["constant"].schedule
        offsets = schedule.offsets(100)
        for head, values in offsets.items():
            assert np.isfinite(values).all()
            if name == "constant":
                assert np.all(values == 0)
            else:
                assert np.all(np.diff(values) < 0) if name == "early_strict" else np.all(np.diff(values) > 0)
                assert abs(values[0]) == np.float32(getattr(schedule, f"{head}_amplitude"))
                np.testing.assert_array_equal(values[:12], schedule.offsets(12)[head])


def test_persistence_uses_historical_boundaries_not_current_relaxation() -> None:
    score = np.full((1, 24), 0.5, dtype=np.float32)
    streams = {"freeze": np.full_like(score, -10), "acceleration": score,
               "periodicity": np.full_like(score, 10),
               "acceleration_persistent": persistent_score(score, 8),
               "periodicity_persistent": persistent_score(np.full_like(score, 10), 4)}
    schedule = BoundarySchedule("early_strict", 10, 1, 1, 1)
    adjusted = adjusted_score_arrays(streams, schedule)
    naive = streams["acceleration_persistent"] - schedule.offsets(24)["acceleration"]
    assert naive[0, 11] > 0
    assert adjusted["acceleration_persistent"][0, 11] < 0
    profile = GlobalIntrinsicProfile(0, 0, 0, 0.01)
    first = alarms_from_scores(adjusted, np.ones(score.shape, bool), profile)
    assert first["acceleration"][0] == 18
    assert first["guard"][0] == 18
    assert np.all(adjusted["acceleration"][0, 11:19] > 0)
    assert adjusted["acceleration"][0, 10] == 0


@pytest.mark.parametrize("name", SCHEDULES)
def test_raw_streaming_matches_batch_causally_without_backdating(name: str) -> None:
    rng = np.random.default_rng(591)
    raw = rng.uniform(0.01, 1, (27, 8, 10, 11, 32)).astype(np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)
    raw[14:] = raw[13]
    profile = TemporalProfile(GlobalIntrinsicProfile(0.5, 0.1, 0.2, 0.01),
                              BoundarySchedule(name, 12, 0.2, 0.1, 0.4))
    monitor = TemporalGuardMonitor(profile)
    records = [monitor.update(query) for query in raw]
    mobility = np.stack([row["layer_mobility"] for row in records])[None]
    acceleration = np.array([row["route_acceleration"] for row in records], dtype=np.float32)[None]
    periodicity = np.array([row["lag_periodicity"] for row in records], dtype=np.float32)[None]
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, profile.base.periodicity_scale)
    adjusted = adjusted_score_arrays(scores, profile.schedule)
    fields = {"freeze_score": "freeze", "acceleration_score": "acceleration_persistent",
              "periodicity_score": "periodicity_persistent"}
    for key, value in fields.items():
        np.testing.assert_array_equal([row[key] for row in records], adjusted[value][0])
    expected = alarms_from_scores(adjusted, np.ones((1, len(raw)), bool), profile.base)
    for head in ("freeze", "acceleration", "periodicity", "turbulence"):
        assert getattr(monitor, f"first_{head}_query") == expected[head][0]
    first = expected["guard"][0]
    assert first >= 14
    for query, row in enumerate(records):
        assert row["alarm"] == (query >= first)
        assert row["first_alarm_query"] == (first if query >= first else -1)
    changed = raw.copy()
    changed[13:] = np.roll(changed[13:], 3, axis=-1)
    other_monitor = TemporalGuardMonitor(profile)
    other = [other_monitor.update(query) for query in changed]
    for key in (*fields, "alarm", "first_alarm_query"):
        np.testing.assert_array_equal([row[key] for row in records[:13]], [row[key] for row in other[:13]])
    if name == "constant":
        original_monitor = IntrinsicGuardMonitor(profile.base)
        original = [original_monitor.update(query) for query in raw]
        for key in (*fields, "alarm", "first_alarm_query"):
            np.testing.assert_array_equal([row[key] for row in records], [row[key] for row in original])


def test_profiles_roundtrip_and_reject_task_metadata_or_wrong_schema(tmp_path: Path) -> None:
    profiles, _ = calibrate_temporal_reference(*reference_features())
    for name, profile in profiles.items():
        path = tmp_path / f"{name}.npz"
        profile.save(path)
        assert TemporalProfile.load(path) == profile
        with pytest.raises(ValueError, match="schema"):
            GlobalIntrinsicProfile.load(path)
        with np.load(path, allow_pickle=False) as archive:
            fields = dict(archive)
        np.savez(path, **fields, task_index=np.asarray(0))
        with pytest.raises(ValueError, match="metadata"):
            TemporalProfile.load(path)
        fields["schema"] = np.asarray("wrong")
        np.savez(path, **fields)
        with pytest.raises(ValueError, match="schema"):
            TemporalProfile.load(path)
        fields["schema"] = np.asarray(SCHEMA)
        fields["tau_queries"] = np.array([15.0])
        np.savez(path, **fields)
        with pytest.raises(ValueError, match="scalar"):
            TemporalProfile.load(path)


@pytest.mark.parametrize("changes", [{"name": "unknown"}, {"tau_queries": 0},
                                      {"freeze_amplitude": -1}, {"periodicity_amplitude": np.nan},
                                      {"acceleration_amplitude": np.inf}])
def test_invalid_boundary_parameters_are_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        BoundarySchedule(**{"name": "constant", "tau_queries": 12, "freeze_amplitude": 1,
                            "acceleration_amplitude": 1, "periodicity_amplitude": 1, **changes})


def test_calibration_api_has_no_labels_identities_or_tunable_schedules() -> None:
    assert set(inspect.signature(calibrate_temporal_reference).parameters) == {
        "layer_mobility", "route_acceleration", "lag_periodicity", "valid", "alarm_budget"}
    with pytest.raises(TypeError):
        calibrate_temporal_reference(*reference_features(), failure=np.zeros(96))


@pytest.mark.parametrize("name", SCHEDULES)
def test_batch_decisions_at_every_prefix_match_the_full_path(name: str) -> None:
    mobility, acceleration, periodicity, valid = reference_features()
    profiles, _ = calibrate_temporal_reference(mobility, acceleration, periodicity, valid, alarm_budget=0.1)
    profile = profiles[name]
    full = adjusted_score_arrays(intrinsic_score_arrays(mobility, acceleration, periodicity,
                                                        profile.base.periodicity_scale), profile.schedule)
    expected = alarms_from_scores(full, valid, profile.base)
    for length in range(1, valid.shape[1] + 1):
        prefix = intrinsic_score_arrays(mobility[:, :length], acceleration[:, :length], periodicity[:, :length],
                                        profile.base.periodicity_scale)
        actual = alarms_from_scores(adjusted_score_arrays(prefix, profile.schedule), valid[:, :length], profile.base)
        for head in expected:
            np.testing.assert_array_equal(actual[head], np.where(expected[head] < length, expected[head], -1))


@pytest.mark.parametrize("budget", [0.03, 0.1, 0.3])
def test_time_only_matches_budget_even_with_tied_lengths(budget: float) -> None:
    valid = np.ones((100, 30), dtype=bool)
    priorities = time_priorities(len(valid), 89)
    profile, audit = calibrate_time_only(valid, priorities, alarm_budget=budget)
    assert profile.boundary_query == 29
    assert audit["reference_replay_alarms"] == audit["reference_alarm_capacity"]
    first = profile.first_alarms(valid, priorities)
    assert np.all(first[first >= 0] == 29)
    np.testing.assert_array_equal(priorities, time_priorities(len(valid), 89))
    extended = np.ones((100, 40), dtype=bool)
    later = profile.first_alarms(extended, priorities)
    assert np.all(later[first < 0] == 30)
    np.testing.assert_array_equal(later[first >= 0], first[first >= 0])
    assert np.all(profile.first_alarms(valid[:, :25], priorities) == -1)


def test_time_only_varied_lengths_and_padding() -> None:
    lengths = np.random.default_rng(58).integers(5, 40, 100)
    valid = np.arange(40)[None] < lengths[:, None]
    priorities = time_priorities(100, 481)
    profile, audit = calibrate_time_only(valid, priorities)
    assert audit["reference_replay_alarms"] == 3
    padded = np.pad(valid, ((0, 0), (0, 9)), constant_values=False)
    assert calibrate_time_only(padded, priorities) == (profile, audit)
    order = np.random.default_rng(4).permutation(100)
    assert calibrate_time_only(valid[order], priorities[order]) == (profile, audit)
    repeated = np.full(100, 0.5)
    tied, tied_audit = calibrate_time_only(valid, repeated)
    assert tied_audit["reference_replay_alarms"] <= 3
    assert tied_audit["unique_priorities"] is False


def test_time_only_validates_inputs() -> None:
    valid = np.ones((100, 20), dtype=bool)
    profile = TimeOnlyProfile(10, 0.5)
    with pytest.raises(ValueError, match="priorities"):
        profile.first_alarms(valid, np.full(100, np.nan))
    with pytest.raises(ValueError, match="priorities"):
        profile.first_alarms(valid, np.ones(100))
    with pytest.raises(ValueError, match="prefix"):
        profile.first_alarms(valid.astype(int), np.full(100, 0.5))
    valid[0, 2] = False
    with pytest.raises(ValueError, match="prefix"):
        calibrate_time_only(valid, np.full(100, 0.5))
    with pytest.raises(ValueError, match="boundary"):
        TimeOnlyProfile(1.5, 0.5)
