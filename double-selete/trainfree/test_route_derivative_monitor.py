from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

import online_multihead_alarm as v1
from online_route_derivative_alarm import raw_derivatives, score_against_reference
from route_derivative_monitor import (
    DETECTORS,
    DerivativeTaskProfile,
    RouteDerivativeMonitor,
)


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results/online_route_derivative"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_query_derivative_normalization() -> None:
    heads = {
        name: np.asarray([[0.2, 0.4, 0.7]], dtype=np.float32)
        for name in v1.HEADS
    }
    _, velocity, acceleration = raw_derivatives(heads)
    np.testing.assert_allclose(velocity[0, 1:], [0.2, 0.3], atol=2e-8)
    np.testing.assert_allclose(acceleration[0, 2], 0.05, atol=2e-8)


def test_live_monitor_uses_only_current_routing_and_respects_warmup() -> None:
    parameters = tuple(inspect.signature(RouteDerivativeMonitor.update).parameters)
    assert parameters == ("self", "router_prob", "expert_ids")

    rng = np.random.default_rng(91)
    reference_n, query_n = 40, 12
    profile = DerivativeTaskProfile(
        task="synthetic/task",
        route_reference=rng.normal(
            size=(reference_n, query_n, len(v1.FEATURES))
        ).astype(np.float32),
        valid=np.ones((reference_n, query_n), dtype=bool),
        velocity_reference=rng.uniform(
            size=(reference_n, query_n)
        ).astype(np.float32),
        acceleration_reference=rng.uniform(
            size=(reference_n, query_n)
        ).astype(np.float32),
        detector_names=DETECTORS,
        quantiles=np.asarray([0.90, 0.95, 0.975]),
        thresholds=np.full((len(DETECTORS), 3), 2.0, dtype=np.float32),
    )
    monitor = RouteDerivativeMonitor(profile)
    outputs = []
    for _ in range(8):
        router = rng.uniform(size=(8, 10, 11, 32)).astype(np.float32)
        router /= router.sum(axis=-1, keepdims=True)
        experts = np.argsort(router, axis=-1)[..., -4:].astype(np.uint8)
        outputs.append(monitor.update(router, experts))
    for output in outputs[:7]:
        assert output["status"] == "normal"
        assert not output["alarm"]
        assert all(np.isnan(value) for value in output["scores"].values())
    assert all(np.isfinite(value) for value in outputs[7]["scores"].values())


def test_heldout_prefix_is_invariant_to_heldout_future() -> None:
    rng = np.random.default_rng(92)
    features = rng.normal(size=(400, 14, len(v1.FEATURES))).astype(np.float32)
    valid = np.ones((400, 14), dtype=bool)
    reference = np.ones(400, dtype=bool)
    reference[:8] = False
    baseline, *_ = score_against_reference(features, valid, reference)
    changed = features.copy()
    changed[:8, 10:] += 1000.0
    replay, *_ = score_against_reference(changed, valid, reference)
    for detector in DETECTORS:
        np.testing.assert_allclose(
            baseline[detector][:8, :10],
            replay[detector][:8, :10],
            equal_nan=True,
        )


def test_seal_and_persistent_baseline_are_unchanged() -> None:
    manifest = json.loads((RESULTS / "sealed_manifest.json").read_text())
    assert manifest["episodes"] == 14_800
    assert manifest["tasks"] == 37
    assert manifest["labels_used"] == []
    assert manifest["query_causal"] is True
    assert manifest["runtime_inputs"] == [
        "current_router_prob",
        "current_expert_ids",
    ]
    artifacts = manifest["artifacts"]
    assert digest(RESULTS / "sealed_online_scores.npz") == artifacts[
        "sealed_scores_sha256"
    ]
    assert digest(RESULTS / "deployment_profiles.npz") == artifacts[
        "deployment_profiles_sha256"
    ]

    with np.load(RESULTS / "sealed_online_scores.npz", allow_pickle=False) as new:
        with np.load(
            HERE / "results/online_single_rollout/sealed_online_scores.npz",
            allow_pickle=False,
        ) as old:
            new_position = new["detector_names"].astype(str).tolist().index(
                "persistent_level"
            )
            old_position = old["detector_names"].astype(str).tolist().index(
                "persistent_route"
            )
            np.testing.assert_array_equal(
                new["scores"][:, :, new_position],
                old["scores"][:, :, old_position],
            )
            np.testing.assert_array_equal(
                new["thresholds"][:, new_position],
                old["thresholds"][:, old_position],
            )
            np.testing.assert_array_equal(
                new["alarms"][:, :, new_position],
                old["alarms"][:, :, old_position],
            )
