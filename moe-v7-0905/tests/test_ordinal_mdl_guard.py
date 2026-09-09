from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

from ordinal_mdl_guard import (  # noqa: E402
    KTCode,
    OrdinalFeatureMonitor,
    OrdinalMDLGuard,
    _decode_prefix,
    score_feature_arrays,
)


def synthetic_features(queries: int = 60) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(409)
    mobility = rng.uniform(0.2, 0.4, (2, queries, 8))
    acceleration = rng.uniform(0.1, 0.2, (2, queries))
    periodicity = rng.uniform(-0.01, 0.01, (2, queries))
    mobility[0, 22:] *= 0.01
    acceleration[1, 22:] += 1.0
    periodicity[1, 22:] -= 1.0
    mobility[:, 0] = np.nan
    periodicity[:, :2] = np.nan
    return mobility, acceleration, periodicity, np.ones((2, queries), dtype=bool)


def test_kt_probability_is_normalized_and_matches_sequential_prediction() -> None:
    code = KTCode(5)
    mass = 0.0
    for sequence in itertools.product(range(3), repeat=5):
        counts = np.zeros(3, dtype=np.int32)
        probability = 1.0
        for symbol in sequence:
            probability *= (counts[symbol] + 0.5) / (counts.sum() + 1.5)
            counts[symbol] += 1
        coded = 2.0 ** -float(code.bits(counts))
        assert coded == pytest.approx(probability, rel=1e-12)
        mass += coded
    assert mass == pytest.approx(1.0, abs=1e-12)
    assert float(code.bits(np.zeros(3, dtype=np.int32))) == 0.0


@pytest.mark.parametrize("counts", [np.array([-1, 0, 1]), np.array([6, 0, 0]),
                                    np.array([1.0, 0.0, 0.0]), np.array([1, 2])])
def test_kt_rejects_invalid_counts(counts: np.ndarray) -> None:
    with pytest.raises(ValueError):
        KTCode(5).bits(counts)


def test_known_split_pays_location_and_model_cost() -> None:
    symbols = np.ones((1, 40, 3), dtype=np.int32)
    symbols[:, :20, 0] = 0
    symbols[:, 20:, 0] = 2
    cumulative = (symbols[..., None] == np.arange(3)).cumsum(axis=1)
    code = KTCode(40)
    gain, split = _decode_prefix(cumulative, code)
    assert split[0, 0] == 20
    expected = float(code.bits(np.array([20, 0, 20]))) - (
        float(code.bits(np.array([20, 0, 0]))) + float(code.bits(np.array([0, 0, 20])))
        + math.log2(20 * 21) + 1
    )
    assert gain[0, 0] == pytest.approx(expected)
    assert gain[0, 0] > 0
    assert gain[0, 1] == -np.inf


def test_constant_features_and_consistent_drift_do_not_imply_a_failure() -> None:
    mobility = np.ones((2, 80, 8))
    mobility[1] = np.linspace(1.0, 0.1, 80)[:, None]
    acceleration = np.ones((2, 80))
    periodicity = np.zeros((2, 80))
    result = score_feature_arrays(mobility, acceleration, periodicity, np.ones((2, 80), bool))
    np.testing.assert_array_equal(result["first_alarm_query"], [-1, -1])


def test_synthetic_freeze_and_turbulence_are_detected_without_backdating() -> None:
    features = synthetic_features()
    result = score_feature_arrays(*features)
    assert (result["first_alarm_query"] >= 22).all()
    assert (result["first_keypoint_query"] >= 0).all()
    assert (result["first_keypoint_query"] <= result["first_alarm_query"]).all()
    assert result["first_freeze_query"][0] >= 22
    assert result["first_turbulence_query"][1] >= 22


def test_stream_and_batch_agree_at_every_prefix() -> None:
    mobility, acceleration, periodicity, valid = synthetic_features()
    result = score_feature_arrays(mobility, acceleration, periodicity, valid)
    for row in range(len(mobility)):
        monitor = OrdinalFeatureMonitor()
        records = [monitor.update(mobility[row, q], acceleration[row, q], periodicity[row, q])
                   for q in range(mobility.shape[1])]
        for field in ("gain_bits", "freeze_gain_bits", "turbulence_gain_bits",
                      "candidate_query", "selected_model"):
            np.testing.assert_allclose([record[field] for record in records], result[field][row], atol=1e-12)
        assert records[-1]["first_alarm_query"] == result["first_alarm_query"][row]
        assert records[-1]["first_keypoint_query"] == result["first_keypoint_query"][row]


def test_future_mutation_truncation_and_padding_preserve_decisions() -> None:
    features = synthetic_features()
    full = score_feature_arrays(*features)
    cut = 29
    prefix = score_feature_arrays(*(array[:, :cut] for array in features))
    changed = tuple(array.copy() for array in features)
    for array in changed[:3]:
        array[:, cut:] = 1e6
    counterfactual = score_feature_arrays(*changed)
    changed[-1][:, cut:] = False
    for array in changed[:3]:
        array[:, cut:] = np.nan
    padded = score_feature_arrays(*changed)
    for name in ("gain_bits", "selected_model", "candidate_query"):
        np.testing.assert_array_equal(full[name][:, :cut], prefix[name])
        np.testing.assert_array_equal(full[name][:, :cut], counterfactual[name][:, :cut])
        np.testing.assert_array_equal(full[name][:, :cut], padded[name][:, :cut])
    np.testing.assert_array_equal(prefix["first_alarm_query"], padded["first_alarm_query"])
    assert (padded["selected_model"][:, cut:] == 0).all()
    assert (padded["candidate_query"][:, cut:] == -1).all()


def test_positive_affine_feature_scales_need_no_recalibration() -> None:
    mobility, acceleration, periodicity, valid = synthetic_features()
    original = score_feature_arrays(mobility, acceleration, periodicity, valid)
    transformed = score_feature_arrays(mobility * 17 + 2, acceleration * 3 + 5,
                                       periodicity * 13 - 7, valid)
    for name in original:
        np.testing.assert_array_equal(original[name], transformed[name])


@pytest.mark.parametrize("queries", [0, 1, 2, 3, 4])
def test_short_sequences_are_safe(queries: int) -> None:
    result = score_feature_arrays(np.ones((2, queries, 8)), np.ones((2, queries)),
                                  np.zeros((2, queries)), np.ones((2, queries), bool))
    np.testing.assert_array_equal(result["first_alarm_query"], [-1, -1])


def test_invalid_feature_inputs_are_rejected() -> None:
    features = synthetic_features()
    features[-1][0, 4] = False
    with pytest.raises(ValueError, match="contiguous"):
        score_feature_arrays(*features)
    features[-1][0, 4] = True
    features[0][0, 4, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_feature_arrays(*features)


def test_raw_routes_equal_cached_feature_replay_and_do_not_mutate_input() -> None:
    rng = np.random.default_rng(603)
    raw = rng.uniform(0.01, 1.0, (32, 8, 10, 11, 32)).astype(np.float32)
    raw /= raw.sum(axis=-1, keepdims=True)
    original = raw.copy()
    monitor = OrdinalMDLGuard()
    records = [monitor.update(query) for query in raw]
    mobility = np.stack([record["layer_mobility"] for record in records])[None]
    acceleration = np.asarray([record["route_acceleration"] for record in records])[None]
    periodicity = np.asarray([record["lag_periodicity"] for record in records])[None]
    replay = score_feature_arrays(mobility, acceleration, periodicity, np.ones((1, len(raw)), bool))
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal([r["selected_model"] for r in records], replay["selected_model"][0])
    np.testing.assert_allclose([r["gain_bits"] for r in records], replay["gain_bits"][0], atol=1e-12)
    changed = raw.copy()
    changed[17:] = np.roll(changed[17:], 3, axis=-1)
    altered = OrdinalMDLGuard()
    altered_records = [altered.update(query) for query in changed]
    np.testing.assert_array_equal([r["gain_bits"] for r in records[:17]],
                                  [r["gain_bits"] for r in altered_records[:17]])


@pytest.mark.parametrize("bad", ["shape", "negative", "nan", "empty"])
def test_invalid_raw_input_does_not_advance_monitor(bad: str) -> None:
    raw = np.full((8, 10, 11, 32), 1 / 32, dtype=np.float32)
    monitor = OrdinalMDLGuard()
    invalid = raw.copy()
    if bad == "shape":
        invalid = invalid[0]
    elif bad == "negative":
        invalid.flat[0] = -1
    elif bad == "nan":
        invalid.flat[0] = np.nan
    else:
        invalid[0, 0, 0] = 0
    with pytest.raises(ValueError):
        monitor.update(invalid)
    assert monitor.update(raw)["query"] == 0
