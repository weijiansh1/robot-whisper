import numpy as np
import pytest

from monitor import HistoryOnlyMonitor, PRIMARY, Rule, distance, features_from_roots, first_alarm, fixed_rules, root_action_routes, score_stream


def random_roots(seed=1, queries=18):
    rng = np.random.default_rng(seed)
    return root_action_routes(rng.dirichlet(np.ones(32), size=(queries, 8, 10)))


def test_exact_identity_and_small_changes():
    roots = random_roots()
    np.testing.assert_array_equal(distance(roots, roots), 0)
    changed = roots.copy()
    changed[..., 0] += 1e-7
    assert (distance(roots, changed) > 0).all()


@pytest.mark.parametrize("rule", fixed_rules(), ids=lambda r: r.name)
def test_prefix_causality_and_future_mutation(rule):
    features = features_from_roots(random_roots())[None]
    full = score_stream(features, rule)
    for end in range(1, features.shape[1]+1):
        prefix = score_stream(features[:, :end], rule)
        np.testing.assert_allclose(prefix, full[:, :end], equal_nan=True, rtol=0, atol=0)
        changed = features.copy()
        changed[:, end:] = 123456
        np.testing.assert_allclose(score_stream(changed, rule)[:, :end], full[:, :end], equal_nan=True, rtol=0, atol=0)


def test_batches_do_not_mix_episodes():
    a = features_from_roots(random_roots())[None]
    b = features_from_roots(random_roots(seed=9))[None]
    np.testing.assert_allclose(score_stream(a, PRIMARY), score_stream(np.concatenate([a,b]), PRIMARY)[:1], equal_nan=True)


def test_streaming_matches_batch_and_reset():
    roots = random_roots()
    for rule in fixed_rules():
        monitor = HistoryOnlyMonitor(rule)
        stream = [monitor.update(np.square(r))["score"] for r in roots]
        expected = score_stream(features_from_roots(roots)[None], rule)[0]
        np.testing.assert_allclose(stream, expected, equal_nan=True, atol=1e-12)
        assert HistoryOnlyMonitor(rule).first_alarm_query == -1


def test_constant_routes_abstain():
    roots = np.repeat(random_roots(queries=1), 20, axis=0)
    features = features_from_roots(roots)[None]
    for rule in fixed_rules():
        assert first_alarm(score_stream(features, rule), rule.threshold)[0] == -1


def test_freeze_detected_only_after_confirmation():
    features = np.ones((1, 20, 8, 3))
    features[:, 0, :, 0] = np.nan
    features[:, 5:, :, 0] = 0.1
    first = first_alarm(score_stream(features, PRIMARY), PRIMARY.threshold)
    assert first[0] == 7
    assert first_alarm(score_stream(features[:, :7], PRIMARY), PRIMARY.threshold)[0] == -1


def test_loop_detected_but_straight_progress_not():
    roots = random_roots(queries=2)
    loop = np.stack([roots[q % 2] for q in range(18)])
    rule = Rule("loop", "recurrence", threshold=0.25, width=1)
    assert first_alarm(score_stream(features_from_roots(loop)[None], rule), rule.threshold)[0] == 5
    # Positive orthant of a unit circle: nearly straight at these small angles.
    angles = np.arange(18) * 0.005 + 0.1
    straight = np.zeros((18,8,10,32))
    straight[...,0] = np.cos(angles)[:,None,None]
    straight[...,1] = np.sin(angles)[:,None,None]
    assert first_alarm(score_stream(features_from_roots(straight)[None], rule), rule.threshold)[0] == -1


def test_invalid_input_is_rejected():
    monitor = HistoryOnlyMonitor()
    with pytest.raises(ValueError):
        monitor.update(np.zeros((8,10,11,32)))
    with pytest.raises(ValueError):
        monitor.update(np.ones((2,8,10,11,32)))


def test_equality_at_threshold_counts_as_alarm():
    assert first_alarm(np.array([[np.nan, 0.25, 0.26]]), 0.25)[0] == 1
