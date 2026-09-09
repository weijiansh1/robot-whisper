"""Protocol tests.  Run with `python -m pytest tests/ -q` from the bundle root.

These check the things that, if broken, would make every number in the bundle
wrong in a way that is invisible in the output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

from common import (  # noqa: E402
    ALL_CELLS,
    BASE_CELLS,
    COHORTS,
    HEADLINE_LEAD,
    V7_ANCHOR,
    V8_ANCHOR,
    chunk_rank,
    confirmed_first,
    load_all,
    score,
    score_all_leads,
    trailing_mean,
    union,
)
from heads import (  # noqa: E402
    first_true_from,
    fire_from,
    held_runs,
    threshold_of,
    window_of,
)


@pytest.fixture(scope="module")
def data():
    return load_all()


# --------------------------------------------------------------------------
# anchors
# --------------------------------------------------------------------------
def test_v7_and_v8_anchors_reproduce(data):
    """load_all() already asserts these; this pins the totals the brief states."""
    tot = {"v7": [0, 0], "v8": [0, 0], "risk": 0}
    for cohort, d in data.items():
        for arm in ("v7", "v8"):
            s = score(d[arm], d["risk"], d["length"], HEADLINE_LEAD)
            assert (s["tp"], s["fp"]) == (V7_ANCHOR if arm == "v7"
                                          else V8_ANCHOR)[cohort]
            tot[arm][0] += s["tp"]
            tot[arm][1] += s["fp"]
        tot["risk"] += int(d["risk"].sum())
    assert tot["risk"] == 1358
    assert tuple(tot["v7"]) == (870, 111)
    assert tuple(tot["v8"]) == (932, 126)


# --------------------------------------------------------------------------
# the seam is measured, and measured causally
# --------------------------------------------------------------------------
def test_chunk_zero_is_nan_everywhere(data):
    """The seam at chunk q needs query q, so it is indexed at the *arrival*
    chunk and chunk 0 can never be defined.  A finite value at chunk 0 would
    mean the detector had one chunk of lookahead."""
    for cohort, d in data.items():
        for cell in ALL_CELLS:
            assert not np.isfinite(d[cell][:, 0]).any(), (cohort, cell)


def test_series_are_defined_exactly_on_chunks_1_to_length(data):
    for cohort, d in data.items():
        n_finite = np.isfinite(d["front_state"]).sum(axis=1)
        assert np.array_equal(n_finite, np.maximum(d["length"] - 1, 0)), cohort


def test_within_episode_information_is_nonconstant(data):
    """Anything bit-constant inside episodes carries no within-episode signal
    and must be excluded.  Nothing here is."""
    for cohort, d in data.items():
        for cell in ALL_CELLS:
            v = np.where(np.isfinite(d[cell]), d[cell], np.nan)
            usable = np.isfinite(d[cell]).sum(axis=1) >= 2
            with np.errstate(invalid="ignore"):
                lo, hi = np.nanmin(v, axis=1), np.nanmax(v, axis=1)
            varying = (hi > lo)[usable]
            # the state chord is genuinely constant (the state token does not
            # move along the denoising axis) - that is the finding, not a bug
            if cell in ("chord_front_state", "chord_back_state"):
                continue
            assert varying.mean() > 0.99, (cohort, cell, varying.mean())


def test_state_chord_is_essentially_zero(data):
    """The core structural finding: the state token's route does not move
    across the ten denoising steps, so the state 'boundary' cell is not a
    boundary measurement at all."""
    for cohort, d in data.items():
        for cell in ("front_state", "back_state"):
            chord = float(np.nanmean(d[f"chord_{cell}"]))
            seam = float(np.nanmean(d[cell]))
            assert chord / seam < 0.01, (cohort, cell, chord, seam)


def test_state_seam_equals_state_within_chunk(data):
    """...and therefore the two are the same series."""
    for cohort, d in data.items():
        for cell in ("front_state", "back_state"):
            a, b = d[cell], d[f"wc_{cell}"]
            m = np.isfinite(a)
            assert float(np.corrcoef(a[m], b[m])[0, 1]) > 0.99999, (cohort, cell)
    legacy = data["legacy_main16x32"]
    for cell in ("front_state", "back_state"):
        m = np.isfinite(legacy[cell])
        assert np.array_equal(legacy[cell][m], legacy[f"wc_{cell}"][m]), cell


def test_action_seam_is_not_the_same_as_within_chunk(data):
    """The action half *is* a different quantity - just a highly redundant one."""
    for cohort, d in data.items():
        for cell in ("front_action", "back_action"):
            a, b = d[cell], d[f"wc_{cell}"]
            m = np.isfinite(a)
            r = float(np.corrcoef(a[m], b[m])[0, 1])
            assert 0.7 < r < 0.95, (cohort, cell, r)


# --------------------------------------------------------------------------
# thresholding
# --------------------------------------------------------------------------
def test_threshold_is_an_attained_order_statistic_and_ties_fire(data):
    d = data["development_main"]
    for cell in BASE_CELLS:
        for direction in ("high", "low"):
            sm = trailing_mean(d[cell], 3)
            thr = threshold_of(sm, 0.99, direction)
            pool = sm[np.isfinite(sm)]
            assert np.isclose(pool, thr).any(), (cell, direction)
            at = sm == thr
            hit = (sm >= thr) if direction == "high" else (sm <= thr)
            assert bool(hit[at].all()), (cell, direction)


def test_strict_inequality_would_drop_the_tie_group(data):
    """The reason the protocol insists on `>=`: on this data the tie group at
    the threshold is not empty."""
    d = data["development_main"]
    sm = trailing_mean(d["front_state"], 3)
    thr = threshold_of(sm, 0.99, "high")
    assert int((sm == thr).sum()) >= 1


# --------------------------------------------------------------------------
# the fast kernel must equal the reference implementation
# --------------------------------------------------------------------------
def test_fast_kernel_matches_v8_confirmed_first(data):
    """`held_runs` + `first_true_from` + `fire_from` must reproduce v8's own
    `confirmed_first` exactly whenever there is no chunk gate."""
    d = data["development_main"]
    n_chunk = d["front_state"].shape[1]
    for cell in BASE_CELLS:
        for width in (1, 3, 6):
            sm = trailing_mean(d[cell], width)
            for direction in ("high", "low"):
                thr = threshold_of(sm, 0.99, direction)
                for earliest in (2, 4, 6):
                    run = held_runs(sm, thr, direction, earliest)
                    for confirm in (1, 2, 3):
                        mine = fire_from(first_true_from(run >= confirm),
                                         earliest, None, n_chunk)
                        ref = confirmed_first(sm, thr, direction, confirm,
                                              earliest)
                        assert np.array_equal(mine, ref), (cell, width,
                                                           direction, confirm,
                                                           earliest)


def test_gate_restricts_firing_to_the_window(data):
    d = data["development_main"]
    n_chunk = d["front_state"].shape[1]
    sm = trailing_mean(d["back_action"], 3)
    thr = threshold_of(sm, 0.99, "high")
    start, hi = window_of(2, (8, 24))
    ft = first_true_from(held_runs(sm, thr, "high", start) >= 2)
    first = fire_from(ft, start, hi, n_chunk)
    fired = first >= 0
    assert (first[fired] >= 8).all()
    assert (first[fired] < 24).all()


# --------------------------------------------------------------------------
# the metric itself
# --------------------------------------------------------------------------
def test_score_all_leads_matches_score(data):
    d = data["external_8b"]
    got = score_all_leads(d["v8"], d["risk"], d["length"])
    for lead in (0, 2, 4, 8, 12):
        ref = score(d["v8"], d["risk"], d["length"], lead)
        assert (got[f"tp{lead}"], got[f"fp{lead}"]) == (ref["tp"], ref["fp"])


def test_union_never_delays_an_alarm(data):
    d = data["external_8b"]
    head = np.where(np.arange(len(d["risk"])) % 3 == 0, 7, -1)
    u = union(d["v8"], head)
    fired = (d["v8"] >= 0) | (head >= 0)
    assert np.array_equal(u >= 0, fired)
    both = (d["v8"] >= 0) & (head >= 0)
    assert (u[both] == np.minimum(d["v8"], head)[both]).all()


def test_lead_filter_can_hide_false_alarms(data):
    """Why selection happens at lead >= 0.  Every risk episode is at least 22
    chunks long while the median success is 13, so a late-firing head has its
    false alarms deleted by the lead filter rather than by being right."""
    for cohort, d in data.items():
        assert d["length"][d["risk"]].min() >= 22, cohort
        assert np.median(d["length"][~d["risk"]]) <= 15, cohort
    d = data["development_main"]
    n_chunk = d["front_state"].shape[1]
    sm = trailing_mean(chunk_rank(d["back_action"], d["back_action"]), 6)
    thr = threshold_of(sm, 0.98, "high")
    start, hi = window_of(2, (8, 24))
    first = fire_from(first_true_from(held_runs(sm, thr, "high", start) >= 2),
                      start, hi, n_chunk)
    s0 = score(first, d["risk"], d["length"], 0)
    s4 = score(first, d["risk"], d["length"], 4)
    assert s0["fp"] > 50 and s4["fp"] == 0


# --------------------------------------------------------------------------
# the rank transform must not leak the test cohort
# --------------------------------------------------------------------------
def test_chunk_rank_against_reference_is_frozen(data):
    """Ranking external against development must not depend on external."""
    dev, ext = data["development_main"], data["external_8b"]
    full = chunk_rank(ext["front_action"], dev["front_action"])
    half = chunk_rank(ext["front_action"][:5000], dev["front_action"])
    m = np.isfinite(full[:5000])
    assert np.allclose(full[:5000][m], half[m])


def test_chunk_rank_is_monotone_within_a_chunk(data):
    d = data["development_main"]
    r = chunk_rank(d["back_action"], d["back_action"])
    for q in (5, 12, 20):
        v, rr = d["back_action"][:, q], r[:, q]
        m = np.isfinite(v)
        order = np.argsort(v[m], kind="mergesort")
        assert np.all(np.diff(rr[m][order]) >= -1e-12), q
