import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_monitor as pm  # noqa: E402
import progress_ratio as pr  # noqa: E402


def _profile(window=4, confirmations=2, threshold=0.5, group="back"):
    return pm.GlobalProgressProfile(
        window=window,
        confirmations=confirmations,
        ratio_threshold=threshold,
        eps_length=1e-5,
        layer_group=group,
    )


def _episode(n_query, seed, step=0.05):
    rng = np.random.default_rng(seed)
    current = rng.random((8, 10, 11, 32)).astype(np.float32)
    out = np.empty((n_query, 8, 10, 11, 32), dtype=np.float32)
    for q in range(n_query):
        current = np.abs(current + step * rng.standard_normal(current.shape).astype(np.float32))
        out[q] = current / current.sum(axis=-1, keepdims=True)
    return out


def test_monitor_emits_no_ratio_before_the_window_fills():
    monitor = pm.ProgressGuardMonitor(_profile(window=4))
    episode = _episode(6, seed=31)
    rows = [monitor.update(query) for query in episode]
    assert all(not np.isfinite(row["ratio"]) for row in rows[:4])
    assert np.isfinite(rows[4]["ratio"])


def test_monitor_alarm_latches_and_records_first_query():
    # R <= 1 恒成立，阈值 1.0 使随机游走必然触发
    monitor = pm.ProgressGuardMonitor(_profile(threshold=1.0, confirmations=1))
    episode = _episode(8, seed=32)
    rows = [monitor.update(query) for query in episode]
    assert rows[-1]["alarm"] is True
    first = rows[-1]["first_alarm_query"]
    assert first == next(i for i, row in enumerate(rows) if row["alarm"])
    assert all(row["alarm"] for row in rows[first:])


def test_monitor_never_alarms_above_threshold():
    # R >= 0 恒成立，阈值 0.0 使 R < 0 永不成立
    monitor = pm.ProgressGuardMonitor(_profile(threshold=0.0))
    episode = _episode(10, seed=33)
    rows = [monitor.update(query) for query in episode]
    assert not any(row["alarm"] for row in rows)
    assert rows[-1]["first_alarm_query"] == -1


def test_rewriting_the_future_does_not_change_an_existing_prefix():
    """严格因果：q 的分数只由 P(0..q) 决定。"""
    profile = _profile()
    episode = _episode(12, seed=34)
    baseline_monitor = pm.ProgressGuardMonitor(profile)
    baseline = [baseline_monitor.update(q) for q in episode][:7]

    mutated = episode.copy()
    rng = np.random.default_rng(99)
    tail = np.abs(rng.random(mutated[7:].shape).astype(np.float32))
    mutated[7:] = tail / tail.sum(axis=-1, keepdims=True)
    replayed_monitor = pm.ProgressGuardMonitor(profile)
    replayed = [replayed_monitor.update(q) for q in mutated][:7]

    # 该断言必须非空洞：窗口填满后前缀里要有真实取值可比，否则本测试形同虚设
    assert sum(np.isfinite(row["ratio"]) for row in baseline) >= 3

    for left, right in zip(baseline, replayed):
        assert left["alarm"] == right["alarm"]
        np.testing.assert_allclose(left["ratio"], right["ratio"], equal_nan=True)


def test_batch_windows_are_trailing_only():
    """批量路径的因果性：截断未来不得改变已算出的前缀。

    在线 monitor 每次只收到一个 query，结构上看不见未来，所以真正的前瞻风险在
    批量路径的窗口实现上。path_length 与 persistent_low 必须是纯拖尾窗口。
    """
    episode = _episode(20, seed=36)
    lags = pr.lag_distances(pr.action_route(episode), pr.LAGS)
    adjacent = lags[None, :, 0, :]

    full_length = pr.path_length(adjacent, 4)
    truncated_length = pr.path_length(adjacent[:, :12], 4)
    np.testing.assert_allclose(
        full_length[:, :12], truncated_length, equal_nan=True
    )
    assert np.isfinite(truncated_length[0, 4:]).all()

    scores = pr.group_ratio(
        pr.progress_ratio(lags[None, :, pr.LAGS.index(4), :], full_length, 1e-5), "back"
    )
    full_persistent = pr.persistent_low(scores, 3)
    truncated_persistent = pr.persistent_low(scores[:, :12], 3)
    np.testing.assert_allclose(
        full_persistent[:, :12], truncated_persistent, equal_nan=True
    )
    assert np.isfinite(truncated_persistent[0, 6:]).any()


def test_online_matches_batch_computation():
    """在线 monitor 与批量路径必须给出同一条 R 序列。"""
    profile = _profile(window=4, group="back")
    episode = _episode(15, seed=35)
    monitor = pm.ProgressGuardMonitor(profile)
    online = np.array([monitor.update(q)["ratio"] for q in episode])
    # 重新用批量路径算一遍
    lags = pr.lag_distances(pr.action_route(episode), pr.LAGS)
    adjacent = lags[None, :, 0, :]
    length = pr.path_length(adjacent, profile.window)
    displacement = lags[None, :, pr.LAGS.index(profile.window), :]
    ratio = pr.progress_ratio(displacement, length, profile.eps_length)
    batch = pr.group_ratio(ratio, profile.layer_group)[0]
    np.testing.assert_allclose(online, batch, rtol=1e-5, equal_nan=True)


def test_profile_rejects_task_metadata(tmp_path):
    path = tmp_path / "bad_profile.npz"
    np.savez(
        path,
        schema=pr.SCHEMA,
        window=4,
        confirmations=2,
        ratio_threshold=0.5,
        eps_length=1e-5,
        layer_group="back",
        task_names=np.array(["a"]),
    )
    with pytest.raises(ValueError, match="must not contain task metadata"):
        pm.GlobalProgressProfile.load(path)
