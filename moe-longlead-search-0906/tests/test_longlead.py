"""Protocol tests for the long-lead search bundle.

These check the rules that are non-negotiable, not the conclusions:
  * the published v8.3 anchors reproduce exactly;
  * every comparison is `>=` / `<=` and the tie group is kept;
  * every threshold is an order statistic of an UNLABELED development pool;
  * the frozen arms are cap-free -- they never see risk, length, suite or task;
  * adding an arm to the union is monotone in TP at every lead;
  * the newly extracted quantities are row-aligned with the published caches
    and reproduce from the raw zarr.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

import bank  # noqa: E402
import common as C  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return C.load_all()


# ------------------------------------------------------------------ anchors
def test_v83_anchor_per_cohort(data):
    for cohort, (tp, n, fp) in C.ANCHOR_LEAD4.items():
        d = data[cohort]
        s = C.score(d["v83"], d["risk"], d["length"], 4)
        assert (s["tp"], int(d["risk"].sum()), s["fp"]) == (tp, n, fp), cohort


def test_v83_anchor_profile(data):
    prof = C.profile({c: d["v83"] for c, d in data.items()}, data)
    for lead, expected in C.ANCHOR_PROFILE.items():
        assert tuple(prof[lead]) == expected, (lead, prof[lead], expected)


def test_corpus_shape(data):
    assert sum(len(d["risk"]) for d in data.values()) == 32960
    assert sum(int(d["risk"].sum()) for d in data.values()) == 1358
    assert sum(int((~d["risk"]).sum()) for d in data.values()) == 31602


def test_v83_rebuilds_from_flow_speed(data):
    rebuilt = C.rebuild_v83(data)
    for cohort, d in data.items():
        assert np.array_equal(rebuilt[cohort], d["v83"]), cohort


def test_risk_runs_to_the_cap(data):
    """The reachability argument depends on this and nothing else.

    Every risk sits exactly at its suite cap, so `lead >= L` needs an alarm at
    chunk <= cap - L.  The converse is almost exact: only 6 of 31,602
    successes reach the cap (goal 1, long 1, object 1, spatial 3), so
    `length == cap` predicts risk with precision 1358/1364.  That near-identity
    is why no length- or horizon-stratified null is usable here.
    """
    at_cap_safe = 0
    for suite, cap in C.CAP.items():
        for d in data.values():
            m = (d["suite"] == suite) & d["risk"]
            if m.any():
                assert (d["length"][m] == cap).all(), suite
            s = (d["suite"] == suite) & ~d["risk"]
            at_cap_safe += int((d["length"][s] >= cap).sum())
    assert at_cap_safe == 6, at_cap_safe


# ------------------------------------------------------------ comparison
def test_ge_not_gt_keeps_the_tie_group():
    """A strict `>` silently drops everything sitting exactly on the
    threshold.  Construct a series where the whole positive class ties."""
    s = np.array([[0.0, 1.0, 1.0, 1.0]] * 6 + [[0.0, 0.0, 0.0, 0.0]] * 6)
    ge = C.confirmed_first(s, 1.0, "high", 2, 0)
    hit = (s > 1.0) & np.isfinite(s)
    held = np.zeros_like(hit)
    run = np.zeros(hit.shape[0], dtype=int)
    for q in range(hit.shape[1]):
        run = np.where(hit[:, q], run + 1, 0)
        held[:, q] = run >= 2
    gt = np.where(held.any(axis=1), held.argmax(axis=1), -1)
    assert (ge >= 0).sum() == 6
    assert (gt >= 0).sum() == 0


# ------------------------------------------------------- order statistics
def test_thresholds_are_order_statistics(data):
    """Every frozen threshold must be a value that actually occurs in the
    unlabeled development pool; `method='lower'` guarantees it."""
    from step4_ceiling_raw import build_bank
    from step9_freeze_and_seal import ARM_A, ARM_B, fit_on_development

    series = build_bank("development_main", data["development_main"])
    fitted = fit_on_development(series)
    pool = series[ARM_A["series"]]
    pool = pool[np.isfinite(pool)]
    assert np.isclose(pool, fitted["A_thr"]).any()
    pool = series[ARM_B["series"]]
    pool = pool[np.isfinite(pool)]
    for thr in fitted["B_bins"].values():
        assert np.isclose(pool, thr).any()


def test_arms_never_see_labels(data):
    """Permuting the outcome must leave the frozen arms bit-identical."""
    from step4_ceiling_raw import build_bank
    from step9_freeze_and_seal import apply_arms, fit_on_development

    dev = data["development_main"]
    series = build_bank("development_main", dev)
    fitted = fit_on_development(series)
    a0, b0 = apply_arms(series, fitted)
    saved = dev["risk"].copy()
    try:
        rng = np.random.default_rng(0)
        dev["risk"] = rng.permutation(saved)
        series2 = build_bank("development_main", dev)
        a1, b1 = apply_arms(series2, fit_on_development(series2))
        assert np.array_equal(a0, a1)
        assert np.array_equal(b0, b1)
    finally:
        dev["risk"] = saved


def test_union_is_monotone(data):
    """Adding a head can only move an alarm earlier, so TP cannot fall."""
    from step4_ceiling_raw import build_bank
    from step9_freeze_and_seal import apply_arms, fit_on_development

    series = {c: build_bank(c, d) for c, d in data.items()}
    fitted = fit_on_development(series["development_main"])
    base, new = {}, {}
    for c, d in data.items():
        a, b = apply_arms(series[c], fitted)
        base[c] = d["v83"]
        new[c] = C.union(d["v83"], a, b)
        assert (new[c] <= np.where(base[c] < 0, 1 << 20, base[c])).all()
    pb, pn = C.profile(base, data), C.profile(new, data)
    for lead in C.LEADS:
        assert pn[lead][0] >= pb[lead][0], lead
        assert pn[lead][1] >= pb[lead][1], lead


# --------------------------------------------------------- new quantities
@pytest.mark.parametrize("cohort", C.COHORTS)
def test_raw_quantities_aligned(cohort, data):
    raw = np.load(C.RESULTS / f"{cohort}_rawq.npz")
    mob = C.mobility(cohort)
    d = data[cohort]
    assert raw["state_mob"].shape == mob[:, :, :, 9].shape
    # identical support: both are adjacent-chunk quantities on the same rows
    assert np.array_equal(np.isfinite(raw["state_mob"]),
                          np.isfinite(mob[:, :, :, 9]))
    alive = np.isfinite(raw["tok_disp"]).any(axis=2)
    last = alive.shape[1] - 1 - np.argmax(alive[:, ::-1], axis=1)
    assert np.array_equal(last, d["length"] - 1)


def test_raw_quantities_reproduce_from_zarr():
    """Recompute one task's episodes straight from routes.zarr."""
    import zarr

    import extract_raw_quantities as E

    task = "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
    path = (C.ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_object" / task
            / "right-50x8-20260903/server/routes.zarr")
    g = zarr.open_group(str(path), mode="r")
    ep = np.asarray(g["episode_id"][:], dtype=int)
    lo, hi = 0, int(np.searchsorted(ep, ep[0], side="right"))
    p9 = np.asarray(g["hb_router_probs"][lo:hi, :, E.STEP])
    p0 = np.asarray(g["hb_router_probs"][lo:hi, :, 0])
    se = np.asarray(g["hb_selected_prob"][lo:hi, :, E.STEP])
    ii = np.asarray(g["hb_expert_ids"][lo:hi, :, E.STEP])
    f = E.episode_features(p9, se, ii, p0)

    dev = C.load_cohort("development_main")
    rows = np.flatnonzero((dev["suite"] == "libero_object")
                          & (dev["task"] == task))
    stored = np.load(C.RESULTS / "development_main_rawq.npz")
    row = rows[int(ep[0])]
    for name in ("tok_disp", "state_mob", "set_inflow", "set_jacc_adj"):
        got = stored[name][row, :hi - lo]
        want = f[name]
        m = np.isfinite(got) & np.isfinite(want)
        assert m.any(), name
        assert np.allclose(got[m], want[m], atol=2e-6), name


def test_probability_inputs_are_normalised():
    """Hellinger and the set statistics assume proper distributions."""
    import zarr
    path = (C.ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_object"
            / "pick_up_the_bbq_sauce_and_place_it_in_the_basket"
            / "right-50x8-20260903/server/routes.zarr")
    g = zarr.open_group(str(path), mode="r")
    p = np.asarray(g["hb_router_probs"][:64]).astype(np.float64)
    assert np.allclose(p.sum(-1), 1.0, atol=2e-3)


def test_mask_after_end_blocks_post_hoc_alarms(data):
    """Padded chunks must be NaN, or a head can 'fire' after the episode."""
    d = data["development_main"]
    s = {"x": np.ones((len(d["risk"]), 52))}
    masked = bank.mask_after_end(s, d["length"])["x"]
    for i in (0, 17, 100, 5000):
        n = int(d["length"][i])
        assert np.isfinite(masked[i, :n]).all()
        assert not np.isfinite(masked[i, n:]).any()
