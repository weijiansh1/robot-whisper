from __future__ import annotations

import numpy as np

import online_closed_loop_alarm_v2 as alarm
import online_multihead_alarm as v1


def test_empirical_midrank_handles_ties() -> None:
    reference = np.asarray([0.0] * 20 + [1.0] * 20)
    values = np.asarray([-1.0, 0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(
        alarm.empirical_midrank(reference, values),
        [0.0, 0.25, 0.5, 0.75, 1.0],
    )


def test_physical_features_are_prefix_causal() -> None:
    rng = np.random.default_rng(7)
    state = rng.normal(size=(12, 8)).astype(np.float32)
    actions = rng.normal(size=(12, 10, 7)).astype(np.float32)
    full = alarm.extract_physical_features(state, actions)
    for length in range(alarm.WARMUP_QUERY + 1, len(state) + 1):
        prefix = alarm.extract_physical_features(state[:length], actions[:length])
        np.testing.assert_allclose(prefix[-1], full[length - 1])


def test_score_fold_excludes_censored_references_and_test_length() -> None:
    rng = np.random.default_rng(11)
    n_episode = 400
    n_query = 10
    init_state = np.repeat(np.arange(50), 8)
    valid = np.ones((n_episode, n_query), dtype=bool)
    route = rng.normal(size=(n_episode, n_query, len(v1.FEATURES))).astype(np.float32)
    physical = rng.normal(
        size=(n_episode, n_query, len(alarm.PHYSICAL_FEATURES))
    ).astype(np.float32)
    lengths = np.full(n_episode, 9, dtype=np.int16)
    lengths[::7] = n_query

    result = alarm.score_fold(
        route, physical, valid, init_state, lengths, held_out=0, query_limit=n_query
    )
    scores, _, _, _, train, completed = result
    assert train.sum() == 392
    assert np.array_equal(completed, train & (lengths < n_query))
    assert not completed[lengths == n_query].any()

    changed_test_lengths = lengths.copy()
    changed_test_lengths[init_state == 0] = n_query
    changed = alarm.score_fold(
        route,
        physical,
        valid,
        init_state,
        changed_test_lengths,
        held_out=0,
        query_limit=n_query,
    )[0]
    test = init_state == 0
    for detector in alarm.DETECTORS:
        np.testing.assert_allclose(
            scores[detector][test], changed[detector][test], equal_nan=True
        )
