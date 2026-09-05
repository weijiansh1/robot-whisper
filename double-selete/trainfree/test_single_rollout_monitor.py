from __future__ import annotations

import numpy as np

import online_closed_loop_alarm_v2 as v2
import online_multihead_alarm as v1
from online_single_rollout_alarm import state_machine_scores
from single_rollout_monitor import (
    DETECTORS,
    SingleRolloutMonitor,
    StreamingStateMachine,
    TaskProfile,
)


def test_streaming_state_machine_matches_vectorized_replay() -> None:
    rng = np.random.default_rng(3)
    n_episode, n_query = 4, 16
    route = {
        name: rng.uniform(size=(n_episode, n_query)).astype(np.float32)
        for name in v1.HEADS
    }
    physical = {
        name: rng.uniform(size=(n_episode, n_query)).astype(np.float32)
        for name in v2.PHYSICAL_HEADS
    }
    expected, _ = state_machine_scores(route, physical)
    for episode in range(n_episode):
        monitor = StreamingStateMachine()
        observed = {name: [] for name in DETECTORS}
        for query in range(n_query):
            scores, _ = monitor.update(
                {name: float(value[episode, query]) for name, value in route.items()},
                {
                    name: float(value[episode, query])
                    for name, value in physical.items()
                },
            )
            for name in DETECTORS:
                observed[name].append(scores[name])
        for name in DETECTORS:
            np.testing.assert_allclose(
                observed[name],
                expected[name][episode],
                rtol=1e-6,
                atol=1e-7,
                equal_nan=True,
            )


def test_monitor_has_no_public_alarm_during_warmup() -> None:
    rng = np.random.default_rng(5)
    reference_n, query_n = 40, 12
    profile = TaskProfile(
        task="synthetic/task",
        route_reference=rng.uniform(
            size=(reference_n, query_n, len(v1.FEATURES))
        ).astype(np.float32),
        physical_reference=rng.uniform(
            size=(reference_n, query_n, len(v2.PHYSICAL_FEATURES))
        ).astype(np.float32),
        valid=np.ones((reference_n, query_n), dtype=bool),
        detector_names=DETECTORS,
        quantiles=np.asarray([0.90, 0.95, 0.975]),
        thresholds=np.full((len(DETECTORS), 3), 2.0, dtype=np.float32),
    )
    monitor = SingleRolloutMonitor(
        profile, 0.95, alarm_detector="persistent_route"
    )
    for query in range(7):
        router = rng.uniform(size=(8, 10, 11, 32)).astype(np.float32)
        router /= router.sum(axis=-1, keepdims=True)
        experts = np.argsort(router, axis=-1)[..., -4:].astype(np.uint8)
        state = rng.normal(size=8).astype(np.float32)
        actions = rng.normal(size=(10, 7)).astype(np.float32)
        output = monitor.update(router, experts, state, actions)
        assert output["query"] == query
        assert output["alarm_detector"] == "persistent_route"
        assert output["status"] == "normal"
        assert not output["alarm"]
        assert all(np.isnan(value) for value in output["scores"].values())


def test_physical_prefix_replay_is_causal() -> None:
    from single_rollout_monitor import PhysicalPrefixExtractor

    rng = np.random.default_rng(13)
    state = rng.normal(size=(10, 8)).astype(np.float32)
    actions = rng.normal(size=(10, 10, 7)).astype(np.float32)
    expected = v2.extract_physical_features(state, actions)
    extractor = PhysicalPrefixExtractor()
    observed = np.stack(
        [extractor.update(state[q], actions[q]) for q in range(len(state))]
    )
    np.testing.assert_allclose(observed, expected, equal_nan=True)
