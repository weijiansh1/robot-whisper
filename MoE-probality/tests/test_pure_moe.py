import inspect

import numpy as np
import pytest

from probability.features import FEATURE_NAMES, detector, history_features
from probability.pure_moe import (LOCAL_INDICES, LOCAL_NAMES, MoEWindowMonitor,
                                  features_from_window, select_local_features)


def routes(length):
    rng = np.random.default_rng(941)
    p = rng.uniform(0.5, 1.5, (length, 8, 10, 32))
    return (p / p.sum(axis=-1, keepdims=True)).astype(np.float16)


def test_no_budget_task_or_clock_in_local_feature_contract():
    assert len(LOCAL_NAMES) == 18
    assert all(not any(word in name for word in ("budget", "remaining", "elapsed", "baseline", "alarm", "age"))
               for name in LOCAL_NAMES)
    x = np.random.default_rng(1).normal(size=(10, len(FEATURE_NAMES)))
    before = select_local_features(x)
    excluded = sorted(set(range(x.shape[1])) - set(LOCAL_INDICES))
    x[:, excluded] = 99999
    np.testing.assert_array_equal(before, select_local_features(x))
    assert list(inspect.signature(MoEWindowMonitor).parameters) == ["bundle"]
    assert list(inspect.signature(MoEWindowMonitor.update).parameters) == ["self", "hb_router_probs"]


def test_fixed_window_matches_audited_cache_at_every_supported_prefix():
    raw = routes(19)
    roots = detector.root_action_routes(raw)
    primitives = detector.features_from_roots(roots).astype(np.float32)
    cached, _ = history_features(primitives, 520)
    for q in range(7, len(roots)):
        expected = select_local_features(cached[q:q+1])
        np.testing.assert_array_equal(features_from_window(roots[q-7:q+1]), expected)
    with pytest.raises(ValueError, match="full eight-query window"):
        select_local_features(cached[:7])


def test_identical_recent_routes_ignore_earlier_history_and_budget():
    roots = detector.root_action_routes(routes(20))
    changed = roots.copy()
    changed[:12] = changed[:12, :, :, ::-1]
    for data, cap in ((roots, 220), (changed, 520)):
        cached, _ = history_features(detector.features_from_roots(data).astype(np.float32), cap)
        np.testing.assert_array_equal(select_local_features(cached[-1:]), features_from_window(roots[-8:]))
