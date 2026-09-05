import numpy as np

from moe_grammar.open_world import (
    HealthyPrefixGrammar,
    HealthyReturnTable,
    build_dynamic_features,
    causal_dwell,
    open_set_scores,
)


def test_dynamic_features_cannot_read_future() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(8, 3)).astype(np.float32)
    phase = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    changed = values.copy()
    changed[5:] += 100.0
    original = build_dynamic_features(values, phase, np.asarray([0]), np.asarray([8]))
    modified = build_dynamic_features(changed, phase, np.asarray([0]), np.asarray([8]))
    np.testing.assert_allclose(original[:5], modified[:5])


def test_dwell_resets_after_return_to_health() -> None:
    percentile = np.asarray([0.5, 0.95, 0.96, 0.7, 0.99], dtype=np.float32)
    dwell = causal_dwell(percentile, np.asarray([0]), np.asarray([5]))
    np.testing.assert_array_equal(dwell, [0, 1, 2, 0, 1])


def test_healthy_return_table_marks_sustained_deviation_as_more_persistent() -> None:
    healthy = np.tile(np.asarray([0.95, 0.4, 0.3, 0.2, 0.1]), 80)
    starts = np.arange(0, len(healthy), 5)
    lengths = np.full(len(starts), 5)
    phase_velocity = np.full(len(healthy), 0.05)
    model = HealthyReturnTable(horizon=3, min_support=5).fit(
        healthy,
        phase_velocity,
        starts,
        lengths,
        np.arange(len(starts)),
    )
    test = np.asarray([0.95, 0.96, 0.97, 0.98, 0.99])
    probability, dwell = model.predict(test, np.zeros(5), np.asarray([0]), np.asarray([5]))
    scores = open_set_scores(test, probability, dwell, np.asarray([0]), np.asarray([5]))
    assert scores["dwell_persistent"][2] > scores["dwell_persistent"][0]
    assert np.isfinite(scores["learned_persistent"]).all()


def test_prefix_grammar_is_causal_and_uses_remote_history() -> None:
    rng = np.random.default_rng(19)
    episode_count = 80
    length = 8
    starts = np.arange(episode_count) * length
    lengths = np.full(episode_count, length)
    progress = np.tile(np.linspace(0.0, 1.0, length), episode_count)
    values = np.column_stack(
        [
            4.0 * progress + rng.normal(scale=0.15, size=len(progress)),
            np.sin(progress * np.pi) + rng.normal(scale=0.08, size=len(progress)),
        ]
    ).astype(np.float32)
    model = HealthyPrefixGrammar(phase_states=4, min_support=20).fit(
        values,
        progress,
        starts,
        lengths,
        np.arange(episode_count),
    )

    sequence = values[:length].copy()
    original, original_phase, _ = model.score(sequence, np.asarray([0]), np.asarray([length]))
    changed_future = sequence.copy()
    changed_future[5:] += 50.0
    modified, modified_phase, _ = model.score(changed_future, np.asarray([0]), np.asarray([length]))
    np.testing.assert_allclose(original[:5], modified[:5])
    np.testing.assert_allclose(original_phase[:5], modified_phase[:5])

    changed_history = sequence.copy()
    changed_history[0] += 20.0
    history_score, _, _ = model.score(changed_history, np.asarray([0]), np.asarray([length]))
    clock_score, _, _ = model.score(
        changed_history,
        np.asarray([0]),
        np.asarray([length]),
        update_with_observations=False,
    )
    assert not np.isclose(history_score[3], clock_score[3])

    online_score = []
    online_phase = []
    belief = None
    for value in sequence:
        surprise, belief, filtered_phase, _ = model.step(value, belief)
        online_score.append(surprise)
        online_phase.append(filtered_phase)
    np.testing.assert_allclose(online_score, original, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(online_phase, original_phase, rtol=1e-5, atol=1e-5)


def test_prefix_grammar_scores_candidates_without_cross_candidate_updates() -> None:
    rng = np.random.default_rng(29)
    episode_count = 60
    length = 6
    starts = np.arange(episode_count) * length
    lengths = np.full(episode_count, length)
    progress = np.tile(np.linspace(0.0, 1.0, length), episode_count)
    values = np.column_stack(
        [
            2.0 * progress + rng.normal(scale=0.1, size=len(progress)),
            np.cos(progress * np.pi) + rng.normal(scale=0.1, size=len(progress)),
        ]
    ).astype(np.float32)
    model = HealthyPrefixGrammar(phase_states=3, min_support=20).fit(
        values,
        progress,
        starts,
        lengths,
        np.arange(episode_count),
    )

    belief = None
    for value in values[:3]:
        _, belief, _, _ = model.step(value, belief)
    candidates = np.stack([values[3], values[3] + 0.5, values[3] - 0.5])
    surprise, posterior, phase, entropy = model.score_candidates(candidates, belief)

    expected = [model.step(candidate, belief) for candidate in candidates]
    np.testing.assert_allclose(surprise, [item[0] for item in expected], rtol=1e-6)
    np.testing.assert_allclose(posterior, [item[1] for item in expected], rtol=1e-6)
    np.testing.assert_allclose(phase, [item[2] for item in expected], rtol=1e-6)
    np.testing.assert_allclose(entropy, [item[3] for item in expected], rtol=1e-6)
