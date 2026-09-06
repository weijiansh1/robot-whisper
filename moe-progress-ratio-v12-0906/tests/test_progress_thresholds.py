import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402


def test_row_min_ignores_nan_and_empty_rows():
    values = np.array([[0.5, np.nan, 0.2], [np.nan, np.nan, np.nan]], dtype=np.float32)
    result = pr.row_min(values)
    assert result[0] == pytest.approx(0.2)
    assert np.isnan(result[1])


def test_quantile_lower_controls_per_trajectory_crossing_rate():
    """r* 应使约 alpha 比例的参考轨迹曾经跌破。"""
    minima = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    threshold = pr.quantile_lower(minima, 0.05)
    assert (minima < threshold).mean() == pytest.approx(0.05, abs=0.02)


def test_quantile_lower_is_the_dual_of_v7_quantile_higher():
    import sys as _sys

    _sys.path.insert(0, str(BUNDLE.parent / "moe-v7-0905/method"))
    import intrinsic_guard_monitor as v7

    rng = np.random.default_rng(11)
    values = rng.standard_normal(500).astype(np.float32)
    # v7.quantile_higher(w, p) 的 p 是"低于阈值的比例"，不是"尾部概率"；
    # 因此取负后要在 p 上做 1-q 才能对齐 quantile_lower(v, q) 的下尾比例 q。
    # 恒等式 ceil((n-1)-x) = (n-1)-floor(x) 保证以下等式对任意 q、n 精确成立
    # （非近似）：quantile_lower(v, q) == -quantile_higher(-v, 1 - q)。
    assert pr.quantile_lower(values, 0.1) == pytest.approx(
        -v7.quantile_higher(-values, 1 - 0.1), abs=1e-6
    )


def test_quantile_lower_requires_enough_references():
    with pytest.raises(ValueError, match="at least 32"):
        pr.quantile_lower(np.zeros(8, dtype=np.float32), 0.05)


def test_persistent_low_is_a_rolling_max():
    """跌破持续 K 次 <=> 窗口内最大值也跌破。"""
    values = np.array([[0.9, 0.1, 0.1, 0.1, 0.8]], dtype=np.float32)
    result = pr.persistent_low(values, 3)
    assert np.isnan(result[0, :2]).all()
    assert result[0, 2] == pytest.approx(0.9)
    assert result[0, 3] == pytest.approx(0.1)
    assert result[0, 4] == pytest.approx(0.8)


def test_persistent_low_with_one_confirmation_is_identity():
    values = np.array([[0.4, 0.6]], dtype=np.float32)
    np.testing.assert_allclose(pr.persistent_low(values, 1), values)


def test_first_below_returns_first_valid_crossing():
    values = np.array([[0.9, 0.2, 0.1]], dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    assert pr.first_below(values, 0.5, valid)[0] == 1
    valid[0, 1] = False
    assert pr.first_below(values, 0.5, valid)[0] == 2


def test_first_below_returns_minus_one_when_never_crossing():
    values = np.array([[0.9, 0.8]], dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    assert pr.first_below(values, 0.5, valid)[0] == -1
