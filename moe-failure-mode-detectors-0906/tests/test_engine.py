#!/usr/bin/env python3
"""Tests for the arm engine.

The engine stores per-task alarm *counts*, never alarm vectors, so the one
thing that must be proved is that those counts are the same numbers a direct
replay of `capfree_protocol.score` would produce.  Everything else here guards
a documented failure mode: strict `>` dropping tie groups, a constant channel
scoring anything other than 0.5, a statistic peeking at the future, and the
cap leaking into a detector.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

import arms as A  # noqa: E402
import common as EW  # noqa: E402
import modes_common as M  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    return M.load("development_main")


# --------------------------------------------------------------------------- #
# causality: no statistic may see a chunk it has not reached


@pytest.mark.parametrize("stat_name", [
    "threshold", "change_point", "self_baseline",
    "persist_fraction", "persist_run", "persist_cycles"])
def test_statistics_are_causal(stat_name):
    rng = np.random.default_rng(0)
    n, T = 40, 20
    u = rng.standard_normal((n, T))
    valid = np.arange(T)[None, :] < rng.integers(8, T + 1, n)[:, None]
    fn = getattr(M, f"stat_{stat_name}")
    args = (u, valid) if stat_name in ("threshold", "change_point", "self_baseline") \
        else (u, valid, 0.5)
    full = fn(*args)
    cut = 12
    u2 = u.copy()
    u2[:, cut:] = rng.standard_normal((n, T - cut)) * 50 + 100
    args2 = (u2, valid) if len(args) == 2 else (u2, valid, 0.5)
    partial = fn(*args2)
    a, b = full[:, :cut], partial[:, :cut]
    assert np.array_equal(np.isneginf(a), np.isneginf(b))
    fin = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[fin], b[fin])


def test_change_point_fires_on_a_step_and_not_on_a_flat_line():
    T = 30
    flat = np.zeros((1, T))
    step = np.concatenate([np.zeros(10), np.ones(T - 10) * 5.0])[None, :]
    valid = np.ones((1, T), bool)
    s_flat = M.stat_change_point(flat, valid)[0, -1]
    s_step = M.stat_change_point(step, valid)[0, -1]
    assert s_flat == pytest.approx(0.0, abs=1e-9)
    assert s_step > 10.0


def test_persistence_separates_one_spike_from_a_sustained_state():
    T = 30
    spike = np.zeros((1, T))
    spike[0, 15] = 10.0
    sustained = np.concatenate([np.zeros(4), np.ones(T - 4) * 1.0])[None, :]
    valid = np.ones((1, T), bool)
    gate = 0.5
    assert M.stat_threshold(spike, valid).max() > M.stat_threshold(sustained, valid).max()
    assert (M.stat_persist_fraction(sustained, valid, gate)[0, -1]
            > M.stat_persist_fraction(spike, valid, gate)[0, -1])
    assert (M.stat_persist_run(sustained, valid, gate)[0, -1]
            > M.stat_persist_run(spike, valid, gate)[0, -1])


def test_persist_cycles_counts_repetition_not_level():
    T = 24
    valid = np.ones((1, T), bool)
    one_long = np.concatenate([np.zeros(4), np.ones(20)])[None, :]
    many_short = np.tile([1.0, 0.0], 12)[None, :]
    gate = 0.5
    assert M.stat_persist_run(one_long, valid, gate)[0, -1] > \
        M.stat_persist_run(many_short, valid, gate)[0, -1]
    assert M.stat_persist_cycles(many_short, valid, gate)[0, -1] > \
        M.stat_persist_cycles(one_long, valid, gate)[0, -1]


# --------------------------------------------------------------------------- #
# ties


def test_ties_use_ge_not_gt():
    stat = np.array([[0.0, 1.0, 1.0, 1.0]])
    rm = A._episode_running_max(stat)
    assert M.first_alarm(rm, 1.0)[0] == 1
    strict = rm > 1.0
    assert not strict.any()


# --------------------------------------------------------------------------- #
# the count table must equal a direct replay


def test_count_table_matches_direct_scoring(frame):
    rng = np.random.default_rng(20260906)
    names, code = A.task_codes(frame)
    group = A.episode_groups(frame)
    for quantity, layer in [("flow_path", "L12"), ("hb_entropy_action", "L15"),
                            ("mobility_d0", "L15"), ("set_dwell", "L5")]:
        x = M.channel(frame, quantity, layer)
        calib = A.calibrate(x, frame)
        for sign in (+1, -1):
            table = A.channel_table(x, frame, calib, sign, code, len(names),
                                    group, None)
            stat_name = rng.choice(sorted(table))
            blob = table[stat_name]
            u, ok = A.standardised(x, frame["valid"], calib["ref"], sign)
            stat = A.statistics(u, ok, calib["gates"][sign])[stat_name]
            rm = A._episode_running_max(stat)
            j = int(rng.integers(len(blob["thresholds"])))
            first = M.first_alarm(rm, float(blob["thresholds"][j]))
            direct = M.score_modes(first, frame)
            counts = blob["counts"][j]
            for li, lead in enumerate(M.LEADS):
                assert counts[:, A.GROUP_NONRISK, li].sum() == direct[f"fp_lead{lead}"]
                assert counts[:, A.GROUP_DROP, li].sum() == direct[f"tp_drop_lead{lead}"]
                assert counts[:, A.GROUP_GRASP, li].sum() == direct[f"tp_grasp_lead{lead}"]
                assert counts[:, :, li].sum() - counts[:, A.GROUP_NONRISK, li].sum() \
                    == direct[f"tp_lead{lead}"]


def test_group_partition_is_exact(frame):
    group = A.episode_groups(frame)
    assert (group == A.GROUP_NONRISK).sum() == int((~frame["risk"]).sum())
    assert (group == A.GROUP_DROP).sum() == \
        int((frame["risk"] & (frame["mode"] == M.MODE_DROP)).sum())
    assert (group == A.GROUP_GRASP).sum() == \
        int((frame["risk"] & (frame["mode"] == M.MODE_GRASP)).sum())
    assert (group != A.GROUP_NONRISK).sum() == int(frame["risk"].sum())


# --------------------------------------------------------------------------- #
# the estimator


def test_constant_channel_scores_exactly_half(frame):
    take = (frame["suite"] == "libero_goal") & (frame["length"] > 6)
    col = np.zeros((int(take.sum()), 1))
    cell = EW.stratified_cell(col, frame["risk"][take], frame["task"][take])
    assert float(cell["auc"][0]) == 0.5
    assert float(cell["se"][0]) == 0.0


# --------------------------------------------------------------------------- #
# no cap anywhere in a detector


def test_no_detector_input_uses_the_cap(frame):
    """The block a detector reads must be independent of the suite's cap.

    Proved by construction: the only inputs are the channel value and the chunk
    index, and the calibration is a per-chunk population statistic.  The test
    checks the negative case - a cap-dependent quantity is present in the block
    but is excluded from the channel pool.
    """
    pool = {q for q, _ in M.channel_names(frame, include_controls=True)}
    assert "leak_full_length" not in pool
    assert "expert_load_effective_rank" not in pool
    assert "expert_load_effective_rank" not in frame["quantities"]
    assert "token_entropy_d0" in pool and "token_differentiation_d0" in pool


def test_modes_only_on_failures(frame):
    assert (frame["mode"][~frame["risk"]] == "unlabelled").all()
    assert set(np.unique(frame["mode"][frame["risk"]])) != {"unlabelled"}
