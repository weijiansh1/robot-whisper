from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

from intrinsic_guard_monitor import (  # noqa: E402
    SCHEMA as PROFILE_SCHEMA, GlobalIntrinsicProfile, IntrinsicGuardMonitor,
    intrinsic_score_arrays,
)
from unlabeled_budget_calibration import (  # noqa: E402
    alarm_capacity, alarms_from_scores, calibrate_peaks, calibrate_reference,
)


def reference_features(episodes: int = 96, queries: int = 30) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(761)
    mobility = rng.uniform(0.05, 0.2, (episodes, queries, 8)).astype(np.float32)
    mobility[:, 0] = np.nan
    mobility[:12, 15:] *= 0.02
    acceleration = rng.uniform(0.02, 0.04, (episodes, queries)).astype(np.float32)
    acceleration[12:24, 12:] *= 5
    periodicity = rng.normal(0, 0.005, (episodes, queries)).astype(np.float32)
    periodicity[:, :2] = np.nan
    periodicity[12:24, 12:] -= 0.1
    valid = np.ones((episodes, queries), dtype=bool)
    return mobility, acceleration, periodicity, valid


def peak_fixture(episodes: int = 80) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(527)
    freeze, acceleration, periodicity = rng.uniform(0.5, 3, (3, episodes)).astype(np.float32)
    ap = acceleration - rng.uniform(0, 1, episodes).astype(np.float32)
    rp = periodicity - rng.uniform(0, 1, episodes).astype(np.float32)
    return freeze, acceleration, periodicity, ap, rp


def brute_force(peaks: tuple[np.ndarray, ...], budget: float) -> tuple[GlobalIntrinsicProfile, int]:
    freeze, acceleration, periodicity, ap, rp = peaks
    n = len(freeze)
    capacity = alarm_capacity(n, budget)
    ordered = [np.sort(v[np.isfinite(v)]) for v in (freeze, acceleration, periodicity)]
    best = None
    for allowance in range(capacity + 1):
        ft = ordered[0][max(len(ordered[0]) - allowance - 1, 0)]
        for j in range(n + 1):
            at = ordered[1][(j * (len(ordered[1]) - 1) + n - 1) // n]
            rt = ordered[2][(j * (len(ordered[2]) - 1) + n - 1) // n]
            turbulent = (ap > at) & (rp > rt)
            if int(turbulent.sum()) <= allowance:
                break
        alarms = (freeze > ft) | turbulent
        if int(alarms.sum()) <= capacity:
            best = GlobalIntrinsicProfile(float(ft), float(at), float(rt), 0.25), allowance
    assert best is not None
    return best


@pytest.mark.parametrize("budget", [0.001, 0.03, 0.1, 0.25, 0.9])
def test_rank_search_matches_brute_force_and_respects_budget(budget: float) -> None:
    peaks = peak_fixture()
    result = calibrate_peaks(*peaks, 0.25, alarm_budget=budget)
    expected, allowance = brute_force(peaks, budget)
    assert result.profile == expected
    assert result.audit["branch_allowance"] == allowance
    assert result.audit["union_reference_alarms"] <= alarm_capacity(len(peaks[0]), budget)
    f, _, _, ap, rp = peaks
    p = result.profile
    actual = (f > p.freeze_threshold) | ((ap > p.acceleration_threshold) & (rp > p.periodicity_threshold))
    assert actual.sum() == result.audit["union_reference_alarms"]


def test_budget_decimal_boundaries_do_not_lose_an_episode() -> None:
    assert alarm_capacity(100, 0.29) == 29
    assert alarm_capacity(16_000, 0.03) == 480


def test_equal_peaks_do_not_break_ties_to_exhaust_budget() -> None:
    peaks = tuple(np.ones(80, dtype=np.float32) for _ in range(5))
    result = calibrate_peaks(*peaks, 0.25, alarm_budget=0.1)
    assert result.audit["union_reference_alarms"] == 0
    assert result.audit["unused_reference_capacity"] == 8


def test_nan_peaks_never_count_as_exceedances() -> None:
    peaks = tuple(v.copy() for v in peak_fixture())
    for array in peaks:
        array[:10] = np.nan
    result = calibrate_peaks(*peaks, 0.25, alarm_budget=0.1)
    expected, _ = brute_force(peaks, 0.1)
    assert result.profile == expected


def test_reference_order_does_not_change_profile_or_audit() -> None:
    peaks = peak_fixture()
    permutation = np.random.default_rng(3).permutation(len(peaks[0]))
    original = calibrate_peaks(*peaks, 0.25)
    reordered = calibrate_peaks(*(v[permutation] for v in peaks), 0.25)
    assert original == reordered


def test_larger_budget_only_relaxes_thresholds() -> None:
    profiles = [calibrate_peaks(*peak_fixture(), 0.25, alarm_budget=b).profile
                for b in (0.01, 0.03, 0.05, 0.1, 0.2)]
    for name in ("freeze_threshold", "acceleration_threshold", "periodicity_threshold"):
        assert np.all(np.diff([getattr(p, name) for p in profiles]) <= 0)


@pytest.mark.parametrize("budget", [0, 1, -0.1, np.inf, np.nan])
def test_invalid_budgets_are_rejected(budget: float) -> None:
    with pytest.raises(ValueError, match="budget"):
        calibrate_peaks(*peak_fixture(), 0.25, alarm_budget=budget)


def test_invalid_peak_shapes_counts_and_persistence_are_rejected() -> None:
    with pytest.raises(ValueError, match="reference"):
        calibrate_peaks(*(v[:10] for v in peak_fixture()), 0.25)
    peaks = peak_fixture()
    peaks[3][0] = peaks[1][0] + 1
    with pytest.raises(ValueError, match="persistent"):
        calibrate_peaks(*peaks, 0.25)
    peaks = peak_fixture()
    peaks[0][0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        calibrate_peaks(*peaks, 0.25)
    with pytest.raises(ValueError, match="aligned"):
        calibrate_peaks(peaks[0][None], *peaks[1:], 0.25)


def test_feature_calibration_replays_and_roundtrips_existing_profile(tmp_path: Path) -> None:
    mobility, acceleration, periodicity, valid = reference_features()
    result = calibrate_reference(mobility, acceleration, periodicity, valid)
    path = tmp_path / "profile.npz"
    np.savez(path, schema=np.asarray(PROFILE_SCHEMA), **result.profile.__dict__)
    loaded = GlobalIntrinsicProfile.load(path)
    assert loaded == result.profile
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, loaded.periodicity_scale)
    first = alarms_from_scores(scores, valid, loaded)
    assert (first["guard"] >= 0).sum() <= result.audit["reference_alarm_capacity"]
    assert result.audit["reference_replay_verified"] is True
    assert (first["guard"] >= 0).sum() == result.audit["reference_replay_alarms"]


def test_invalid_padding_cannot_affect_calibration() -> None:
    features = reference_features()
    original = calibrate_reference(*features)
    padded = [np.pad(v, ((0, 0), (0, 7), (0, 0)) if v.ndim == 3 else ((0, 0), (0, 7)),
                     constant_values=False if v.dtype == bool else np.inf) for v in features]
    repeated = calibrate_reference(*padded)
    assert original == repeated


def test_nonfinite_real_features_and_nonprefix_validity_are_rejected() -> None:
    features = reference_features()
    features[-1][0, 4] = False
    with pytest.raises(ValueError, match="contiguous"):
        calibrate_reference(*features)
    features[-1][0, 4] = True
    features[0][0, 4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calibrate_reference(*features)


def test_calibrator_accepts_neither_labels_nor_task_identifiers() -> None:
    parameters = inspect.signature(calibrate_reference).parameters
    assert set(parameters) == {"layer_mobility", "route_acceleration", "lag_periodicity", "valid", "alarm_budget"}
    with pytest.raises(TypeError):
        calibrate_reference(*reference_features(), original_failure=np.zeros(96))


def test_calibrated_profile_changes_only_decisions_not_scores_or_causality() -> None:
    profile = calibrate_reference(*reference_features()).profile
    another = GlobalIntrinsicProfile(profile.freeze_threshold + 1, profile.acceleration_threshold + 1,
                                      profile.periodicity_threshold + 1, profile.periodicity_scale)
    rng = np.random.default_rng(370)
    raw = rng.uniform(0.01, 1, (24, 8, 10, 11, 32)).astype(np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)
    monitors = [IntrinsicGuardMonitor(p) for p in (profile, another)]
    records = [[monitor.update(query) for query in raw] for monitor in monitors]
    for name in ("freeze_score", "acceleration_score", "periodicity_score"):
        np.testing.assert_array_equal([r[name] for r in records[0]], [r[name] for r in records[1]])
    mobility = np.stack([r["layer_mobility"] for r in records[0]])[None]
    acceleration = np.array([r["route_acceleration"] for r in records[0]])[None]
    periodicity = np.array([r["lag_periodicity"] for r in records[0]])[None]
    scores = intrinsic_score_arrays(mobility, acceleration, periodicity, profile.periodicity_scale)
    first = alarms_from_scores(scores, np.ones((1, len(raw)), bool), profile)
    assert records[0][-1]["first_alarm_query"] == first["guard"][0]
    changed = raw.copy()
    changed[13:] = np.roll(changed[13:], shift=4, axis=-1)
    monitor = IntrinsicGuardMonitor(profile)
    counterfactual = [monitor.update(query) for query in changed]
    for name in ("freeze_score", "acceleration_score", "periodicity_score", "alarm", "first_alarm_query"):
        np.testing.assert_array_equal([r[name] for r in records[0][:13]], [r[name] for r in counterfactual[:13]])
