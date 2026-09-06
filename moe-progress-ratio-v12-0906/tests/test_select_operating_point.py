import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(BUNDLE / "experiments"))

import select_operating_point_v12 as sel  # noqa: E402


def _candidates(**overrides):
    base = pd.DataFrame(
        {
            "window": [4, 6, 8, 4],
            "layer_group": ["back", "back", "all", "front"],
            "confirmations": [2, 2, 3, 2],
            "quantile": [0.05, 0.05, 0.05, 0.05],
            "ratio_threshold": [0.3, 0.3, 0.3, 0.3],
            "timely_fpr": [0.004, 0.004, 0.010, 0.004],
            "low_prior_tp": [25, 25, 90, 12],
            "low_prior_precision": [0.6, 0.7, 0.9, 0.9],
        }
    )
    return base.assign(**overrides)


def test_choose_maximises_low_prior_tp_under_constraints():
    chosen = sel.choose(_candidates())
    # 第 3 行低先验 TP 最高但 FPR 超标；第 4 行 TP 不足 20
    # 第 1、2 行并列 TP=25，取 precision 更高的第 2 行
    assert chosen["window"] == 6
    assert chosen["low_prior_precision"] == pytest.approx(0.7)


def test_choose_breaks_precision_ties_with_smaller_window():
    candidates = _candidates(low_prior_precision=[0.7, 0.7, 0.9, 0.9])
    assert sel.choose(candidates)["window"] == 4


def test_choose_returns_none_when_no_candidate_is_feasible():
    assert sel.choose(_candidates(low_prior_tp=[1, 2, 3, 4])) is None


def test_choose_rejects_candidates_over_the_fpr_cap():
    assert sel.choose(_candidates(timely_fpr=[0.02, 0.02, 0.02, 0.02])) is None


def test_k2_gate_requires_half_the_per_task_yield():
    assert sel.k2_gate(global_tp=50, per_task_tp=90) is True
    assert sel.k2_gate(global_tp=40, per_task_tp=90) is False
    assert sel.k2_gate(global_tp=3, per_task_tp=83) is False  # flow_settling 的对照点


def test_calibrate_threshold_ignores_outcomes():
    """阈值只能是无标签 order statistic。"""
    rng = np.random.default_rng(41)
    scores = rng.random((200, 30)).astype(np.float32)
    valid = np.ones(scores.shape, dtype=bool)
    threshold = sel.calibrate_threshold(scores, valid, confirmations=2, quantile=0.05)
    assert 0.0 <= threshold <= 1.0
    # 打乱"标签"不改变阈值：函数签名里根本没有 outcome
    assert threshold == sel.calibrate_threshold(scores, valid, confirmations=2, quantile=0.05)
