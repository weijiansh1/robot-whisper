import json

import numpy as np
import pandas as pd
import pytest

from recovery import (AlertState, METHODS, RecoveryGuardMonitor, SCHEMA, fit_profile,
                      raw_features, recent_peak, score_streams)
from monitor import build_budget_streams, first_alarm, prefix_peak


def sample():
    rng = np.random.default_rng(82)
    n, q = 24, 22
    valid = np.arange(q)[None] < rng.integers(16, q + 1, n)[:, None]
    mobility = rng.uniform(.002, .05, (n, q, 8)).astype(np.float32)
    mobility[:, 0] = np.nan
    acceleration = rng.uniform(.01, .07, (n, q)).astype(np.float32)
    periodicity = rng.normal(0, .01, (n, q)).astype(np.float32)
    periodicity[:, :2] = np.nan
    cache = dict(mobility=mobility, acceleration=acceleration, periodicity=periodicity, valid=valid)
    for key, values in cache.items():
        if key != "valid":
            values[~valid] = np.nan
    raw = rng.uniform(.01, .5, (n, q, 2)).astype(np.float32)
    raw[~valid] = np.nan
    return cache, raw


def test_recent_evidence_expires_before_later_disjoint_hit():
    left = np.zeros((1, 10))
    right = left.copy()
    left[0, 1] = right[0, 7] = 10
    assert np.minimum(prefix_peak(left), prefix_peak(right))[0, 7] == 10
    assert np.minimum(recent_peak(left), recent_peak(right))[0, 7] == 0


def test_recent_window_preserves_overlapping_hits_and_can_clear():
    left = np.zeros((1, 10))
    right = left.copy()
    left[0, 2] = right[0, 4] = 10
    joint = np.minimum(recent_peak(left), recent_peak(right))
    assert joint[0, 4] == joint[0, 5] == 10
    assert joint[0, 6] == 0


def test_baseline_matches_existing_budget_guard():
    cache, raw = sample()
    ref = np.arange(12)
    profile = fit_profile(cache, raw, ref)
    actual = score_streams(cache, raw, profile)[0]
    legacy, _ = build_budget_streams(cache, raw, np.zeros(cache['valid'].shape), ref)
    from monitor import METHODS as old_methods
    np.testing.assert_allclose(actual, legacy[old_methods.index('v82')], equal_nan=True)


def test_prefix_causality_and_padding_invariance():
    cache, raw = sample()
    profile = fit_profile(cache, raw, np.arange(12))
    full = score_streams(cache, raw, profile)
    for stop in (7, 10, 15, 19):
        prefix = score_streams({k: v[:, :stop] for k, v in cache.items()}, raw[:, :stop], profile)
        np.testing.assert_allclose(prefix, full[:, :, :stop], equal_nan=True)
    corrupted = {k: v.copy() for k, v in cache.items()}
    for key in ('mobility', 'acceleration', 'periodicity'):
        corrupted[key][~cache['valid']] = 1e9
    extra = raw.copy()
    extra[~cache['valid']] = -1e9
    np.testing.assert_allclose(score_streams(corrupted, extra, profile), full, equal_nan=True)
    assert fit_profile(corrupted, extra, np.arange(12)) == profile


def test_absolute_freeze_gate_cannot_increase_score_at_fixed_threshold():
    cache, raw = sample()
    scores = score_streams(cache, raw, fit_profile(cache, raw, np.arange(12)))
    for old, gated in (('v82_reference', 'absolute_gate'), ('recent4', 'recovery')):
        a, b = scores[METHODS.index(old)], scores[METHODS.index(gated)]
        good = np.isfinite(a) & np.isfinite(b)
        assert (b[good] <= a[good]).all()


def test_signal_can_clear_without_erasing_alarm_history():
    state = AlertState(3, 1)
    rows = [state.update(s) for s in [0, 2, 4, 2, .5, .4, 0, 4]]
    assert [r['state'] for r in rows] == ['NORMAL', 'WATCH', 'ALARM', 'ALARM', 'ALARM', 'NORMAL', 'NORMAL', 'ALARM']
    assert rows[5]['signal_cleared'] and rows[5]['ever_alarm']
    assert rows[-1]['first_alarm_query'] == 2
    assert first_alarm(np.asarray([[0, 2, 4, 2, .5, .4, 0, 4]]), np.ones((1, 8), bool), 3)[0] == 2


def test_missing_scores_do_not_count_as_clearance():
    state = AlertState(3, 1)
    rows = [state.update(s) for s in [4, 0, np.nan, 0, 0]]
    assert rows[2]['state'] == rows[3]['state'] == 'ALARM'
    assert rows[4]['signal_cleared']


def test_strict_threshold_and_unsupported_budget():
    state = AlertState(3, 1)
    assert not state.update(3)['ever_alarm']
    assert state.update(3.1)['ever_alarm']
    assert not AlertState(np.inf, np.inf).update(1e20)['ever_alarm']
    with pytest.raises(ValueError):
        AlertState(1, 2)


def test_online_raw_probabilities_match_batch_and_reset(tmp_path):
    cache, raw = sample()
    profile = fit_profile(cache, raw, np.arange(12))
    profile.update(checkpoints=['test'], method='recovery', threshold=1., watch_threshold=.5)
    path = tmp_path / 'profile.json'
    path.write_text(json.dumps(profile))
    rng = np.random.default_rng(12)
    queries = rng.dirichlet(np.ones(32), size=(18, 8, 10, 11)).astype(np.float32)
    m, a, p, extras = [], [], [], []
    previous, history = None, []
    for query in queries:
        mobility, acceleration, period, extra, final = raw_features(query, previous, history)
        m.append(mobility)
        a.append(acceleration)
        p.append(period)
        extras.append(extra)
        history.append(final[4:].reshape(40, 32))
        previous = final
    batch = score_streams(dict(mobility=np.asarray(m)[None], acceleration=np.asarray(a)[None],
                               periodicity=np.asarray(p)[None], valid=np.ones((1, len(queries)), bool)),
                          np.asarray(extras)[None], profile)[METHODS.index('recovery'), 0]
    monitor = RecoveryGuardMonitor(path, 'test')
    actual = [monitor.update(query)['score'] for query in queries]
    np.testing.assert_allclose(actual, batch, equal_nan=True)
    monitor.reset()
    assert monitor.update(queries[0])['query'] == 0
    with pytest.raises(ValueError):
        RecoveryGuardMonitor(path, 'other')
    with pytest.raises(ValueError):
        monitor.update(np.zeros((8, 10, 11, 32)))


def test_selection_and_splits_use_only_designated_development_rows():
    from run_experiment import choose, split_roles
    frame = pd.DataFrame([dict(run_id=run, task=task, init_state_id=init)
                          for run in ('right-50x8-20260903', 'right-50x8b-20260903')
                          for task in ('a', 'b') for init in range(50)])
    seen = []
    for fold in range(5):
        roles = split_roles(frame, fold)
        assert tuple(map(len, roles)) == (40, 20, 20, 20)
        identities = [set(zip(frame.iloc[r].task, frame.iloc[r].init_state_id)) for r in roles]
        assert not any(identities[i] & identities[j] for i in range(4) for j in range(i))
        seen.extend(roles[-1])
    assert sorted(seen) == list(range(100, 200))
    first = np.full((len(METHODS), 3), -1)
    first[0] = [10, 10, -1]
    first[1] = [9, 9, 9]
    first[2] = [9, 10, -1]
    selected, _ = choose(first, np.asarray([True, True, False]))
    assert selected == 2
