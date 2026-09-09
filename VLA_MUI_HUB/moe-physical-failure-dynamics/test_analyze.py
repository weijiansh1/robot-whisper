from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("analyze.py")
SPEC = importlib.util.spec_from_file_location("moe_physical_failure_dynamics", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hellinger_endpoints() -> None:
    same = np.asarray([[0.5, 0.5]], dtype=np.float32)
    assert np.allclose(MODULE.hellinger(same, same), 0.0)
    left = np.asarray([[1.0, 0.0]], dtype=np.float32)
    right = np.asarray([[0.0, 1.0]], dtype=np.float32)
    assert np.allclose(MODULE.hellinger(left, right), 1.0)


def test_top4_churn_uses_set_jaccard() -> None:
    base = np.asarray([[0, 1, 2, 3]], dtype=np.uint8)
    reordered = np.asarray([[3, 2, 1, 0]], dtype=np.uint8)
    half = np.asarray([[0, 1, 4, 5]], dtype=np.uint8)
    disjoint = np.asarray([[4, 5, 6, 7]], dtype=np.uint8)
    assert np.allclose(MODULE.top4_churn(base, reordered), 0.0)
    assert np.allclose(MODULE.top4_churn(base, half), 1.0 - 2.0 / 6.0)
    assert np.allclose(MODULE.top4_churn(base, disjoint), 1.0)


def test_vectorized_resampling() -> None:
    values = np.asarray([[0.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    result = MODULE.resample_at(
        values,
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.25, 1.0]),
    )
    assert np.allclose(result, [[0.0, 2.0], [0.5, 2.5], [2.0, 4.0]])


def test_matched_comparison_midrank_and_unmatched() -> None:
    specs = [
        MODULE.EpisodeSpec(0, "r", 0, 0, 2, 1, 1, "s", "t", "run", "sha", True, "success", "success", "success", "observational"),
        MODULE.EpisodeSpec(1, "r", 1, 2, 2, 1, 2, "s", "t", "run", "sha", True, "success", "success", "success", "observational"),
        MODULE.EpisodeSpec(2, "r", 2, 4, 2, 1, 3, "s", "t", "run", "sha", False, "reason", "none", "medium", "observational"),
        MODULE.EpisodeSpec(3, "r", 3, 6, 2, 2, 4, "s", "t", "run", "sha", False, "reason", "none", "medium", "observational"),
    ]
    values = np.asarray([[1.0], [3.0], [3.0], [2.0]], dtype=np.float32)
    refs = MODULE.reference_map(specs)
    percentile, mean, counts = MODULE.matched_comparison(values, specs[2:], refs)
    assert np.allclose(percentile[0], [0.75])
    assert np.allclose(mean[0], [2.0])
    assert counts.tolist() == [2, 0]
    assert np.isnan(percentile[1]).all()


def test_matched_comparison_treats_float_noise_as_tie() -> None:
    specs = [
        MODULE.EpisodeSpec(0, "r", 0, 0, 2, 1, 1, "s", "t", "run", "sha", True, "success", "success", "success", "observational"),
        MODULE.EpisodeSpec(1, "r", 1, 2, 2, 1, 2, "s", "t", "run", "sha", False, "reason", "none", "medium", "observational"),
    ]
    values = np.asarray([[0.4], [0.4 + 5e-8]], dtype=np.float32)
    percentile, _, _ = MODULE.matched_comparison(
        values, specs[1:], MODULE.reference_map(specs)
    )
    assert np.allclose(percentile, 0.5)


def test_episode_feature_shapes_and_load_normalization() -> None:
    rng = np.random.default_rng(11)
    raw = rng.uniform(size=(5, 8, 10, 11, 32)).astype(np.float32)
    raw[:, :, :, 0] = raw[:, :, :1, 0]
    probability = MODULE.normalize_probability(raw)
    ids = np.argpartition(probability, -4, axis=-1)[..., -4:].astype(np.uint8)
    as_probability = rng.uniform(size=(5, 4, 3)).astype(np.float32)
    as_ids = as_probability.argmax(axis=-1).astype(np.uint8)
    features, audit = MODULE.extract_episode_features(raw, ids, as_probability, as_ids)
    assert features["trajectory"].shape == (11, 2, 10)
    assert features["within"].shape == (3, 2, 10, 11, 3)
    assert features["cross"].shape == (10, 11, 3)
    assert features["boundary"].shape == (10, 2)
    assert features["as_phase"].shape == (11, 4)
    assert features["expert_load"].shape == (3, 2, 11, 32)
    assert np.allclose(features["expert_load"].sum(axis=-1), 1.0)
    assert audit["state_denoise_span"] == 0.0


def test_pin_regime_is_not_primary_observational_data() -> None:
    assert MODULE.run_regime("right-50x8-20260903") == "observational"
    assert MODULE.run_regime("pin-base") == "duplicate_pin_base"
    assert MODULE.run_regime("pin-on") == "front_layer_intervention"
    assert MODULE.run_regime("pin-off") == "front_layer_intervention"
