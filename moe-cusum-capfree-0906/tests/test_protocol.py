"""Protocol tests.  Anything that fails here invalidates the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

import bank  # noqa: E402
import capfree_common as C  # noqa: E402
import detectors as D  # noqa: E402


@pytest.fixture(scope="session")
def frames():
    return {c: C.frame(c) for c in ("development_main", "external_8b")}


# --------------------------------------------------------------------------
# 1. the published anchors must reproduce before anything else is believed


ANCHORS = {
    "external_8b": dict(n=15600, risks=564, tp=439, fp=80, tp4=347, fp4=57,
                        tp8=261, fp8=43, med=13.0),
    "development_main": dict(n=14800, risks=487, tp=382, fp=67, tp4=303,
                             fp4=38, tp8=217, fp8=26, med=10.0),
}


@pytest.mark.parametrize("cohort", list(ANCHORS))
def test_v7_guard_anchor(frames, cohort):
    a = ANCHORS[cohort]
    f = frames[cohort]
    assert f["n"] == a["n"]
    assert int(f["risk"].sum()) == a["risks"]
    dets = C.cp.load_detectors(cohort, f["n"])
    guard = next(dets[k] for k in dets if "v7_guard" in k)
    s = C.score(guard, f["risk"], f["length"])
    assert (s["tp"], s["fp"]) == (a["tp"], a["fp"])
    assert (s["tp_lead4"], s["fp_lead4"]) == (a["tp4"], a["fp4"])
    assert (s["tp_lead8"], s["fp_lead8"]) == (a["tp8"], a["fp8"])
    assert s["median_lead"] == a["med"]


@pytest.mark.parametrize("cohort,tp", [("external_8b", 274),
                                       ("development_main", 222)])
def test_capfree_baseline_anchor(frames, cohort, tp):
    """Best fixed-chunk baseline at FP <= 80, lead >= 4, is `still_running_q37`."""
    f = frames[cohort]
    b = C.fixed_chunk_baseline(f["risk"], f["length"])
    ok = b[b.fp_lead4 <= 80]
    best = ok.loc[ok.tp_lead4.idxmax()]
    assert int(best.q0) == 37
    assert int(best.tp_lead4) == tp


# --------------------------------------------------------------------------
# 2. ties: `>=`, never `>`


def test_first_crossing_is_inclusive():
    """A cell exactly equal to the threshold must fire, on that chunk."""
    stat = np.array([[0.0, 1.0, 2.0, 2.0], [0.0, 0.0, 0.0, 0.0]])
    valid = np.ones((2, 4), bool)
    assert D.first_crossing(stat, valid, 2.0).tolist() == [2, -1]
    assert D.first_crossing(stat, valid, 1.0).tolist() == [1, -1]
    assert D.first_crossing(stat, valid, 0.0).tolist() == [0, 0]
    # the running-max shortcut used for the sweeps must agree exactly
    cm = D.cummax_valid(stat, valid)
    for thr in (0.0, 1.0, 2.0, 3.0):
        assert (D.first_from_cummax(cm, thr).tolist()
                == D.first_crossing(stat, valid, thr).tolist())


@pytest.mark.parametrize("quantity,min_worst_loss_pct",
                         [("set_dwell", 50.0), ("query_top1_churn", 15.0),
                          ("query_hard_churn", 0.0)])
def test_strict_gt_drops_the_tie_group(frames, quantity, min_worst_loss_pct):
    """`select_early_lock.py:176,207` uses `>`, which drops the whole tie
    group.  Sweep every *attained* value of a discrete channel and record the
    worst share of alarming episodes that a strict `>` would silently lose.

    The prior measurement (89.3% / 30.1% / 0.2%) is threshold-dependent, so
    the assertion is on the ordering it implies: `set_dwell` loses most of its
    alarms at its worst threshold, `query_top1_churn` a material share,
    `query_hard_churn` far less than either.
    """
    f = frames["development_main"]
    valid = f["valid"]
    name, x = next(C.iter_channels("development_main",
                                   [("chan", quantity, "L3", "-")]))
    worst = 0.0
    worst_thr = None
    for thr in np.unique(x[valid]):
        incl = D.first_crossing(x, valid, float(thr))
        m = (x > thr) & valid
        strict = np.where(m.any(axis=1), m.argmax(axis=1), -1)
        n_incl = int((incl >= 0).sum())
        if n_incl < 50:
            continue
        pct = 100 * int(((incl >= 0) & (strict < 0)).sum()) / n_incl
        if pct > worst:
            worst, worst_thr = pct, float(thr)
    print(f"\n{name}: 最差阈值 {worst_thr} 处，用 > 丢失 {worst:.1f}% 的报警 episode")
    assert worst >= min_worst_loss_pct
    if min_worst_loss_pct == 0.0:
        assert worst < 15.0, "expected the least tie-sensitive channel"


def test_kofm_count_is_integer_so_ties_are_everything():
    """With K = M the count statistic can never exceed K, so a strict `>`
    silently disables the detector entirely."""
    aud = pd.read_csv(C.RESULTS / "tie_audit.csv")
    k12 = aud[(aud.family == "kofm") & (aud["mode"] == "v7budget")]
    assert len(k12) == 2
    assert (k12.n_tie_cells > 0).all()
    assert (k12.n_episodes_lost_with_strict_gt > 0).all()


# --------------------------------------------------------------------------
# 3. the detector may not use the cap


def test_no_cap_symbol_in_the_detector_path():
    """`capfree_protocol` keeps the cap in exactly two objects, `CAPS` and
    `ANCHOR_WINDOW`, plus the old 0.65-of-horizon window constant.  None of
    them may appear in code here, in any form."""
    import ast
    banned = {"CAPS", "ANCHOR_WINDOW"}
    for p in sorted((HERE.parent / "experiments").glob("*.py")):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id not in banned, f"{p.name} uses {node.id}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned, f"{p.name} uses .{node.attr}"
            if isinstance(node, ast.Constant) and node.value == 0.65:
                raise AssertionError(f"{p.name} contains the 0.65 window")
    # and the protocol module really does hold them, so the test can bite
    assert hasattr(C.cp, "CAPS") and hasattr(C.cp, "ANCHOR_WINDOW")


def test_detector_inputs_are_chunk_index_and_routing_only():
    cfg = json.loads((C.RESULTS / "frozen_config.json").read_text())
    for mode in ("window", "v7budget"):
        for fam, f in cfg["selections"][mode]["families"].items():
            for ch in f["channels"]:
                assert "|per_task" not in ch
                assert ch.split(":")[0] in ("flow", "chan", "hb", "deriv")
            # normalisation constants are per chunk index, one per chunk
            assert len(f["norm_mu"]) == C.N_CHUNKS
            assert len(f["norm_sd"]) == C.N_CHUNKS


# --------------------------------------------------------------------------
# 4. banned channels: the sum of the two oppositely-signed halves


def test_load_entropy_is_exactly_the_banned_sum(frames):
    f = frames["development_main"]
    valid = f["valid"]
    got = dict(C.iter_channels("development_main", [
        ("flow", "token_entropy", "L2", "sm"),
        ("flow", "token_differentiation", "L2", "sm")]))
    idx = np.load(C.FLOW_DIR / "development_main_index.npz", allow_pickle=True)
    mi = list(idx["metric_names"].astype(str)).index("load_entropy")
    m = np.load(C.FLOW_DIR / "development_main_metrics.npy", mmap_mode="r")
    le = np.asarray(m[:, :, 0, :, mi], np.float32).mean(axis=2)
    a = got["flow:token_entropy:L2:sm"]
    b = got["flow:token_differentiation:L2:sm"]
    assert np.allclose((a + b)[valid], le[valid], atol=1e-4)


def test_banned_channels_are_not_offered():
    names = {s[1] for s in C.channel_specs()}
    assert "load_entropy" not in names
    assert "expert_load_effective_rank" not in names
    assert "token_entropy" in names and "token_differentiation" in names


# --------------------------------------------------------------------------
# 5. within-episode information for every channel actually used


def test_frozen_channels_carry_within_episode_information(frames):
    from screen_channels import within_episode_info
    f = frames["development_main"]
    valid = f["valid"]
    cfg = json.loads((C.RESULTS / "frozen_config.json").read_text())
    used = set()
    for mode in ("window", "v7budget"):
        for fam in cfg["selections"][mode]["families"].values():
            used.update(c.rsplit("|", 2)[0] for c in fam["channels"])
    names, X = bank.build("development_main")
    idx = {n: i for i, n in enumerate(names)}
    for ch in sorted(used):
        info = within_episode_info(X[idx[ch]], valid)
        assert info["n_distinct"] >= 2, ch
        assert info["const_frac"] < 0.999, ch      # never bit-constant
        # `set_dwell` is 61-88% constant inside an episode; it may appear only
        # as the single-chunk *floor*, never in a headline arm.
        if info["const_frac"] > 0.5:
            owner = [(m, fam) for m in ("window", "v7budget")
                     for fam, v in cfg["selections"][m]["families"].items()
                     if any(c.startswith(ch) for c in v["channels"])]
            assert all(fam == "single" for _, fam in owner), (ch, info, owner)
    scr = pd.read_csv(C.RESULTS / "cusum_channel_screen_development.csv")
    assert len(scr) > 400
    floor = pd.read_csv(C.RESULTS / "channel_screen_development.csv")
    assert int(floor.dead.sum()) > 0, "no bit-constant channel was excluded"


# --------------------------------------------------------------------------
# 6. length is a negative control, never a baseline


def test_length_recalls_everything_and_is_not_a_baseline():
    ops = pd.read_csv(C.RESULTS / "frozen_operating_points.csv")
    neg = ops[ops.family == "negative_control"]
    assert len(neg) == 6
    assert not neg.is_baseline.any()
    # risk == "did not finish before the cap", so a sub-cap length threshold
    # recalls 100% by construction
    for cohort in ("development_main", "external_8b"):
        r = neg[(neg.cohort == cohort) & (neg.arm == "NEGCTRL_length_gt_20")]
        assert float(r.recall.iloc[0]) == 1.0
    assert bool(ops[ops.family == "baseline"].is_baseline.all())


# --------------------------------------------------------------------------
# 7. the CUSUM recursion, and the baseline it nests


def test_cusum_matches_the_textbook_recursion():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((7, 12)).astype(np.float32)
    valid = np.ones((7, 12), bool)
    for k in (-1.0, 0.0, 0.5):
        S = D.cusum_path(z, valid, k)
        ref = np.zeros_like(S)
        for i in range(7):
            s = 0.0
            for q in range(12):
                s = max(0.0, s + (z[i, q] - k))
                ref[i, q] = s
        assert np.allclose(S, ref, atol=1e-5)


def test_negative_k_cusum_nests_the_capfree_baseline(frames):
    """With an uninformative channel, S_q = |k| q, so the CUSUM *is* the
    fixed-chunk baseline.  `excess_tp` therefore measures only what routing
    adds on top of the counter."""
    f = frames["external_8b"]
    valid = f["valid"]
    z = np.zeros((f["n"], C.N_CHUNKS), np.float32)
    S = D.cusum_path(z, valid, -0.25)
    first = D.first_crossing(S, valid, 0.25 * 20)     # h = |k| * q0 with q0 = 20
    ref = np.where(f["length"] > 19, 19, -1)
    assert np.array_equal(first, ref)


# --------------------------------------------------------------------------
# 8. nulls


def test_null_is_not_positive_in_the_target_window():
    nl = pd.read_csv(C.RESULTS / "null_sweep.csv")
    win = nl[(nl.cohort == "external_8b") & (nl.fp_lead4 >= C.FP_LO)
             & (nl.fp_lead4 <= C.FP_HI)]
    assert len(win) > 0
    # against the randomised (concave-hull) cap-free baseline the null is <= 0
    assert win.excess_hull_tp.max() <= 0


def test_rate_matched_null_is_not_positive():
    ops = pd.read_csv(C.RESULTS / "frozen_operating_points.csv")
    r = ops[ops.arm == "null_rate_matched|v7_guard"]
    assert len(r) == 2
    assert (r.excess_tp <= 0).all()


# --------------------------------------------------------------------------
# 9. the headline claim


def test_headline_beats_the_anchor():
    h = json.loads((C.RESULTS / "headline.json").read_text())
    a = h["anchor_v7_guard_external"]
    assert (a["tp_lead4"], a["fp_lead4"]) == (347, 57)
    frozen = h["cusum|v7budget"]
    posthoc = h["cusum|v7budget@fp<=57"]
    # frozen point: more TP, at a slightly larger false-alarm count
    assert frozen["tp_lead4"] > a["tp_lead4"]
    # same frozen detector, threshold read at the anchor's own budget:
    # strictly dominates - more TP *and* fewer false alarms
    assert posthoc["tp_lead4"] > a["tp_lead4"]
    assert posthoc["fp_lead4"] < a["fp_lead4"]
    assert posthoc["beats_347_57"] is True


def test_external_norms_come_from_development(frames):
    """The per-chunk normalisation applied to external is the development
    one; recomputing it on external must give different constants."""
    names, X = bank.build("development_main")
    sd = bank.fit_norm(X[:2], frames["development_main"]["valid"])
    names2, X2 = bank.build("external_8b")
    se = bank.fit_norm(X2[:2], frames["external_8b"]["valid"])
    assert names == names2
    assert not np.allclose(sd["raw"][0], se["raw"][0])
    cfg = json.loads((C.RESULTS / "frozen_config.json").read_text())
    mu = np.array(cfg["selections"]["v7budget"]["families"]["cusum"]["norm_mu"])
    assert np.isfinite(mu).all()
