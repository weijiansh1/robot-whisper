"""预注册决策函数的单测：先钉死规则，再让扫描调用它们。

修订一的判据是合取式 ``L >= ell* AND R < r*``，连续确认 K 次。`ell*` 固定为
pooled reference `L` 的中位数，不进入网格，因此这里也不存在"扫 ell*"的测试。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(BUNDLE / "experiments"))

import select_loop_operating_point as loop  # noqa: E402


def _candidates(**overrides):
    base = pd.DataFrame(
        {
            "window": [4, 6, 8, 4],
            "layer_group": ["back", "back", "all", "front"],
            "confirmations": [2, 2, 3, 2],
            "quantile": [0.05, 0.05, 0.05, 0.05],
            "ratio_threshold": [0.3, 0.3, 0.3, 0.3],
            "ell_star": [0.5, 0.5, 0.5, 0.5],
            "timely_fpr": [0.004, 0.004, 0.010, 0.004],
            "low_prior_loop_tp": [25, 25, 90, 12],
            "low_prior_loop_precision": [0.6, 0.7, 0.9, 0.9],
        }
    )
    return base.assign(**overrides)


# --------------------------------------------------------------------------- #
# A1：选择规则
# --------------------------------------------------------------------------- #


def test_choose_maximises_low_prior_loop_tp_under_constraints():
    chosen = loop.choose(_candidates())
    # 第 3 行低先验 loop TP 最高但 FPR 超标；第 4 行 TP 不足 20
    # 第 1、2 行并列 TP=25，取 precision 更高的第 2 行
    assert chosen["window"] == 6
    assert chosen["low_prior_loop_precision"] == pytest.approx(0.7)


def test_choose_breaks_precision_ties_with_smaller_window():
    candidates = _candidates(low_prior_loop_precision=[0.7, 0.7, 0.9, 0.9])
    assert loop.choose(candidates)["window"] == 4


def test_choose_rejects_candidates_over_the_fpr_cap():
    assert loop.choose(_candidates(timely_fpr=[0.02, 0.02, 0.02, 0.02])) is None


def test_choose_rejects_candidates_under_the_low_prior_loop_floor():
    # 19 < 20：差一个也不放宽
    assert loop.choose(_candidates(low_prior_loop_tp=[19, 19, 19, 19])) is None
    assert loop.choose(_candidates(low_prior_loop_tp=[20, 1, 1, 1]))["window"] == 4


def test_choose_returns_none_when_no_candidate_is_feasible():
    assert loop.choose(_candidates(low_prior_loop_tp=[1, 2, 3, 4])) is None


def test_choose_uses_the_pre_registered_constants():
    assert loop.FPR_CAP == 0.005
    assert loop.MIN_LOW_PRIOR_LOOP_TP == 20


# --------------------------------------------------------------------------- #
# A2：loop 特异性
# --------------------------------------------------------------------------- #


def test_a2_gate_passes_only_when_loop_dominates_both():
    assert loop.a2_gate(5.6, 1.0, 0.3) is True


def test_a2_gate_fails_when_static_matches_loop():
    assert loop.a2_gate(5.6, 5.6, 0.3) is False
    assert loop.a2_gate(5.6, 9.0, 0.3) is False


def test_a2_gate_fails_when_phantom_matches_loop():
    assert loop.a2_gate(5.6, 1.0, 5.6) is False
    assert loop.a2_gate(5.6, 1.0, 9.0) is False


def test_a2_gate_fails_on_a_missing_lift():
    assert loop.a2_gate(float("nan"), 1.0, 0.3) is False
    assert loop.a2_gate(5.6, float("nan"), 0.3) is False


# --------------------------------------------------------------------------- #
# 合取判据的两条纯函数：滚动最小值与 ell*
# --------------------------------------------------------------------------- #


def test_persistent_high_is_a_rolling_minimum():
    values = np.asarray([[3.0, 1.0, 2.0, 4.0, 5.0]], dtype=np.float32)
    got = loop.persistent_high(values, 2)
    assert np.isnan(got[0, 0])
    np.testing.assert_allclose(got[0, 1:], [1.0, 1.0, 2.0, 4.0])


def test_persistent_high_with_one_confirmation_is_the_identity():
    values = np.asarray([[3.0, 1.0, np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(loop.persistent_high(values, 1), values)


def test_persistent_high_propagates_nan_over_the_window():
    values = np.asarray([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32)
    got = loop.persistent_high(values, 2)
    assert np.isnan(got[0, :3]).all()
    assert got[0, 3] == pytest.approx(2.0)


def test_persistent_high_is_the_dual_of_persistent_low():
    """``>= ell*`` 全窗成立 <=> 滚动最小值 >= ell*，与 persistent_low 对偶。"""
    import progress_ratio as pr

    rng = np.random.default_rng(12)
    values = rng.random((16, 20)).astype(np.float32)
    high = loop.persistent_high(values, 3)
    low = pr.persistent_low(-values, 3)
    finite = np.isfinite(high)
    np.testing.assert_allclose(high[finite], -low[finite], atol=1e-6)


def test_persistent_high_rejects_bad_arguments():
    with pytest.raises(ValueError):
        loop.persistent_high(np.zeros((2, 3, 4), dtype=np.float32), 2)
    with pytest.raises(ValueError):
        loop.persistent_high(np.zeros((2, 3), dtype=np.float32), 0)


def test_length_threshold_is_the_median_over_valid_finite_entries():
    values = np.asarray([[1.0, 2.0, 3.0, 100.0], [4.0, 5.0, np.nan, 100.0]], dtype=np.float32)
    valid = np.asarray([[True, True, True, False], [True, True, True, False]])
    assert loop.length_threshold(values, valid) == pytest.approx(3.0)


def test_length_threshold_rejects_an_empty_reference():
    values = np.full((2, 3), np.nan, dtype=np.float32)
    valid = np.ones((2, 3), dtype=bool)
    with pytest.raises(ValueError):
        loop.length_threshold(values, valid)


# --------------------------------------------------------------------------- #
# 合取报警本身
# --------------------------------------------------------------------------- #


def _alarm(ratio, length, valid=None, ratio_threshold=0.5, floor=1.0):
    ratio = np.asarray(ratio, dtype=np.float32)
    length = np.asarray(length, dtype=np.float32)
    if valid is None:
        valid = np.ones(ratio.shape, dtype=bool)
    return loop.first_conjunctive_alarm(ratio, length, ratio_threshold, floor, valid)


def test_first_conjunctive_alarm_needs_both_conditions():
    # 行 0：低 R 且高 L -> 报警；行 1：低 R 但低 L（freeze 格）-> 不报警
    first = _alarm([[0.9, 0.1], [0.9, 0.1]], [[2.0, 2.0], [2.0, 0.2]])
    np.testing.assert_array_equal(first, [1, -1])


def test_first_conjunctive_alarm_ignores_high_ratio_with_high_mobility():
    first = _alarm([[0.9, 0.9]], [[2.0, 2.0]])
    np.testing.assert_array_equal(first, [-1])


def test_first_conjunctive_alarm_takes_the_earliest_eligible_chunk():
    first = _alarm([[0.1, 0.1, 0.1]], [[0.2, 2.0, 2.0]])
    np.testing.assert_array_equal(first, [1])


def test_first_conjunctive_alarm_respects_the_validity_mask():
    valid = np.asarray([[True, False, True]])
    first = _alarm([[0.1, 0.1, 0.1]], [[2.0, 2.0, 0.2]], valid=valid)
    np.testing.assert_array_equal(first, [0])
    first = _alarm([[0.9, 0.1, 0.1]], [[2.0, 2.0, 0.2]], valid=valid)
    np.testing.assert_array_equal(first, [-1])


def test_first_conjunctive_alarm_uses_the_persistent_arrays_for_k_confirmations():
    """K 次确认由传入的滚动统计量承载，报警函数本身不再看历史。"""
    import progress_ratio as pr

    ratio = np.asarray([[0.9, 0.1, 0.1, 0.1]], dtype=np.float32)
    length = np.asarray([[2.0, 2.0, 2.0, 2.0]], dtype=np.float32)
    valid = np.ones(ratio.shape, dtype=bool)
    first = loop.first_conjunctive_alarm(
        pr.persistent_low(ratio, 2), loop.persistent_high(length, 2), 0.5, 1.0, valid
    )
    # q1 只有一次低 R，q2 才满足连续两次
    np.testing.assert_array_equal(first, [2])


def test_first_conjunctive_alarm_is_never_earlier_than_the_ratio_only_rule():
    """合取只会推迟或取消报警，不会提前。"""
    import progress_ratio as pr

    rng = np.random.default_rng(7)
    ratio = rng.random((64, 30)).astype(np.float32)
    length = rng.random((64, 30)).astype(np.float32)
    valid = np.ones(ratio.shape, dtype=bool)
    ratio_only = pr.first_below(ratio, 0.4, valid)
    both = loop.first_conjunctive_alarm(ratio, length, 0.4, 0.5, valid)
    fired = both >= 0
    assert (ratio_only[fired] >= 0).all()
    assert (both[fired] >= ratio_only[fired]).all()


# --------------------------------------------------------------------------- #
# 提升倍数
# --------------------------------------------------------------------------- #


def test_lift_is_observed_over_expected():
    assert loop.lift(10.0, 2.0) == pytest.approx(5.0)


def test_lift_is_nan_without_an_expectation():
    assert np.isnan(loop.lift(10.0, 0.0))
