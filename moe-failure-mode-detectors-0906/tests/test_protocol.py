#!/usr/bin/env python3
"""Protocol invariants: the things that would silently invalidate every number.

These are the claims the report leans on that are cheap to check directly:
the physical-mode join, the published anchors, the cap-free baseline, the
negative control, the pool definitions, and the arithmetic of the count table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

import arms as A  # noqa: E402
import capfree_protocol as CF  # noqa: E402
import modes_common as M  # noqa: E402
from select_and_score import (baseline_frame, baseline_at, totals,  # noqa: E402
                              union, Q0_PROTOCOL)

MODE_COUNTS = {
    "development_main": {"drop": 218, "grasp": 116, "risk": 487, "n": 14800},
    "external_8b": {"drop": 241, "grasp": 143, "risk": 564, "n": 15600},
}


@pytest.fixture(scope="module")
def frames():
    return {c: M.load(c) for c in MODE_COUNTS}


# --------------------------------------------------------------------------- #
# labels


def test_mode_join_counts(frames):
    for cohort, want in MODE_COUNTS.items():
        f = frames[cohort]
        assert len(f["risk"]) == want["n"]
        assert int(f["risk"].sum()) == want["risk"]
        assert int((f["risk"] & (f["mode"] == M.MODE_DROP)).sum()) == want["drop"]
        assert int((f["risk"] & (f["mode"] == M.MODE_GRASP)).sum()) == want["grasp"]


def test_no_success_carries_a_mode(frames):
    for f in frames.values():
        assert (f["mode"][~f["risk"]] == "unlabelled").all()


def test_both_modes_are_medium_confidence():
    labels = pd.read_csv(M.LABEL_CSV)
    for mode in M.MODES:
        conf = labels.loc[labels.primary_failure_reason == mode,
                          "failure_reason_confidence"]
        assert set(conf.unique()) == {"medium"}, mode


def test_grasp_mode_is_concentrated_in_one_task(frames):
    """The confound the whole leave-one-task-out analysis exists for."""
    for cohort, f in frames.items():
        take = f["risk"] & (f["mode"] == M.MODE_GRASP)
        top = pd.Series(f["task"][take]).value_counts()
        assert "KITCHEN_SCENE8" in top.index[0]
        assert top.iloc[0] / take.sum() > 0.6


# --------------------------------------------------------------------------- #
# anchors and baselines


def test_v7_guard_capfree_anchor(frames):
    want = {"development_main": (303, 38, 382, 67),
            "external_8b": (347, 57, 439, 80)}
    for cohort, (tp4, fp4, tp, fp) in want.items():
        f = frames[cohort]
        det = CF.load_detectors(cohort, len(f["risk"]))
        got = M.score_modes(det[next(k for k in det if "v7_guard|global" in k)], f)
        assert (got["tp_lead4"], got["fp_lead4"], got["tp"], got["fp"]) == \
            (tp4, fp4, tp, fp)


def test_fixed_chunk_baseline_matches_the_published_point(frames):
    """external, q0 = 37: 274/564 true positives with 60 false alarms, all long."""
    f = frames["external_8b"]
    b = baseline_frame(f, Q0_PROTOCOL).set_index("q0")
    assert int(b.loc[37, "tp_all_lead4"]) == 274
    assert int(b.loc[37, "fp_lead4"]) == 60
    first = np.where(f["length"] > 37, 37, -1)
    timely = (first >= 0) & ((f["length"] - first) >= 4) & f["risk"]
    assert set(np.unique(f["suite"][timely])) == {"libero_long"}


def test_baseline_becomes_the_label_at_q0_equal_cap_minus_lead(frames):
    """Why the sweep stops at 39: at q0 = cap - 4 the rule *is* the risk label."""
    f = frames["external_8b"]
    long = f["suite"] == "libero_long"
    cap = int(f["length"][long].max())
    q0 = cap - M.HEADLINE_LEAD
    timely = long & (f["length"] > q0) & ((f["length"] - q0) >= M.HEADLINE_LEAD)
    assert np.array_equal(timely[long], f["risk"][long])
    assert q0 not in Q0_PROTOCOL


def test_length_is_a_negative_control_not_a_baseline(frames):
    """Risk is defined as not finishing before the cap, so length recalls 100%."""
    for f in frames.values():
        cap = pd.Series(f["length"]).groupby(f["suite"]).transform("max").to_numpy()
        alarm = np.where(f["length"] >= cap, 0, -1)
        got = M.score_modes(alarm, f)
        assert got["tp"] == int(f["risk"].sum())
        assert got["recall"] == 1.0


# --------------------------------------------------------------------------- #
# pools and controls


def test_headline_pool_excludes_the_mostly_constant_channel(frames):
    from select_and_score import load_table
    tab = load_table(M.CACHE / "development_main_ops.npz")
    pools = A.pool_masks(tab["ops"])
    q = tab["ops"]["quantity"].to_numpy()
    assert not (q[pools["headline"]] == "set_dwell").any()
    assert (q[pools["flagged"]] == "set_dwell").any()
    assert not pools["headline"][tab["ops"]["is_control"].to_numpy()].any()
    assert set(np.unique(q[pools["null_epconst"]])) == {
        "ctrl_episode_const_rand", "ctrl_episode_const_rand2", "ctrl_flow_noise_seed"}


def test_elapsed_counter_persistence_is_the_fixed_chunk_baseline(frames):
    """A run-length statistic on a content-free channel reproduces the baseline."""
    from select_and_score import load_table
    tab = load_table(M.CACHE / "external_8b_ops_ratematched.npz")
    ops = tab["ops"]
    rows = ((ops["quantity"] == "ctrl_const_elapsed")
            & (ops["stat"] == "persist_run|g900")).to_numpy()
    tot = totals(tab["counts"], np.ones(len(tab["tasks"]), bool), M.HEADLINE_LEAD)
    f = frames["external_8b"]
    base = baseline_frame(f, Q0_PROTOCOL)
    best = tot["tp_all"][rows].max()
    j = int(np.flatnonzero(rows)[int(np.argmax(tot["tp_all"][rows]))])
    assert best >= baseline_at(base, tot["fp"][j], "tp_all")


def test_rate_matched_null_does_not_beat_the_baseline(frames):
    """The published claim, restated as the quantity it can only mean."""
    rng = np.random.default_rng(M.SEED)
    for cohort, f in frames.items():
        base = CF.fixed_chunk_baseline(f["risk"], f["length"])
        at = CF.baseline_frontier(base, M.HEADLINE_LEAD)
        x = M.channel(f, "ctrl_white_noise", "L2")
        ref = M.chunk_reference(x, f["valid"])
        u, ok = A.standardised(x, f["valid"], ref, +1)
        rm = A._episode_running_max(M.stat_threshold(u, ok))
        excess = []
        for k in (10, 50, 200, 800):
            th = np.sort(rm[:, -1][np.isfinite(rm[:, -1])])[::-1][k - 1]
            first = M.first_alarm(rm, float(th))
            nul = M.score_modes(CF.rate_matched_null(first, rng, f["length"]), f)
            excess.append(nul["tp_lead4"] - at(nul["fp_lead4"]))
        assert max(excess) <= 2, (cohort, excess)


# --------------------------------------------------------------------------- #
# machinery


def test_totals_matmul_matches_the_naive_sum():
    from select_and_score import load_table
    tab = load_table(M.CACHE / "development_main_ops.npz")
    rng = np.random.default_rng(3)
    mask = rng.random(len(tab["tasks"])) < 0.6
    got = totals(tab["counts"], mask, 4)
    li = {b: i for i, b in enumerate(M.LEADS)}[4]
    naive = tab["counts"][..., li][:, mask, :].sum(axis=1)
    assert np.array_equal(got["fp"], naive[:, A.GROUP_NONRISK])
    assert np.array_equal(got["tp_drop"], naive[:, A.GROUP_DROP])
    assert np.array_equal(got["tp_all"], naive[:, 1:].sum(axis=1))


def test_union_takes_the_earliest_alarm():
    a = np.array([-1, 5, 3, -1])
    b = np.array([2, -1, 7, -1])
    assert np.array_equal(union(a, b), np.array([2, 5, 3, -1]))


def test_frozen_thresholds_are_identical_between_cohorts():
    from select_and_score import load_table
    dev = load_table(M.CACHE / "development_main_ops.npz")
    ext = load_table(M.CACHE / "external_8b_ops_frozenvalue.npz")
    assert np.allclose(dev["ops"]["threshold"], ext["ops"]["threshold"])
    for col in ("quantity", "layer", "sign", "stat", "grid_k"):
        assert (dev["ops"][col].to_numpy() == ext["ops"][col].to_numpy()).all()


def test_rate_matched_table_shares_the_index_but_not_the_values():
    from select_and_score import load_table
    dev = load_table(M.CACHE / "development_main_ops.npz")
    rat = load_table(M.CACHE / "external_8b_ops_ratematched.npz")
    for col in ("quantity", "layer", "sign", "stat", "grid_k"):
        assert (dev["ops"][col].to_numpy() == rat["ops"][col].to_numpy()).all()
    assert not np.allclose(dev["ops"]["threshold"], rat["ops"]["threshold"])
