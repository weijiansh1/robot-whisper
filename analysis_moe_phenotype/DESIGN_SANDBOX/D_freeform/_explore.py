"""Exploration only (not part of the frozen deliverable).

Checks, on the two SCENE8 calibration units:
  1. per-episode q reconstruction from control_step
  2. clean-success reference availability vs absolute q  (=> operating range q_hi)
  3. per-group clean-success counts (=> group-centering feasibility)
  4. direction sanity of the E2/E3 template signs on the event-aligned windows
Data read-only; writes nothing.
"""
import os, sys, csv, json
import numpy as np

ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = {"main": "main16x32", "grid": "grid50x8"}

SIGNALS = ["late_flow_volatility", "route_acceleration", "gate_entropy", "top12_margin",
           "token_consensus", "token_dispersion", "layer_disagreement", "flow_com", "mob1_w8"]
SHORT = {"late_flow_volatility": "V", "route_acceleration": "A", "gate_entropy": "S_ent",
         "top12_margin": "S_margin", "token_consensus": "C", "token_dispersion": "D_tok",
         "layer_disagreement": "D_layer", "flow_com": "flow_com", "mob1_w8": "mob1_w8",
         "min_mobk": "min_mobk"}


def load(unit):
    corp = UNITS[unit]
    d = np.load(f"{ROOT}/features/{corp}/{TASK}/rows.npz", allow_pickle=True)
    ep = d["episode_id"].astype(np.int64)
    cs = d["control_step"].astype(np.int64)
    # per-episode q = control_step - first control_step of that episode
    order = np.lexsort((cs, ep))
    assert np.all(order == np.arange(len(ep))), "rows not sorted by (episode, control_step)"
    first = {}
    q = np.empty_like(cs)
    for i in range(len(ep)):
        e = ep[i]
        if e not in first:
            first[e] = cs[i]
        q[i] = cs[i] - first[e]
    X = {}
    for s in SIGNALS:
        X[s] = d[s].astype(np.float64)
    mk = d["mob_k"].astype(np.float64)          # (N, 8) k = 1..8
    X["min_mobk"] = np.nanmin(mk[:, 1:], axis=1)  # min over k>=2
    ev = {}
    with open(f"{ROOT}/events/{corp}/{TASK}/events.csv") as f:
        for r in csv.DictReader(f):
            ev[int(r["episode_id"])] = dict(scene=int(r["scene"]), repeat=int(r["repeat"]),
                                            success=int(r["success"]), n_queries=int(r["n_queries"]),
                                            loop=int(r["loop_onset_q"]), static=int(r["static_onset_q"]),
                                            grade=r["proxy_grade"])
    return dict(ep=ep, q=q, X=X, ev=ev, scene=d["scene"].astype(np.int64))


for unit in ["main", "grid"]:
    D = load(unit)
    ep, q, ev = D["ep"], D["q"], D["ev"]
    print("=" * 78)
    print(unit, "rows", len(ep), "episodes", len(np.unique(ep)))
    # q contiguity
    for e in np.unique(ep)[:5]:
        qq = q[ep == e]
        assert np.all(qq == np.arange(len(qq)))
    print("  q reconstruction OK; max q", q.max())
    # nan audit
    for s in ["min_mobk", "mob1_w8"]:
        v = D["X"][s]
        nn = np.isnan(v)
        print(f"  {s}: nan {nn.sum()}/{len(v)}; first non-nan q of an episode:",
              q[(~nn)][:1], " min q with value:", q[~nn].min())
    # clean success per group
    eps = np.array(sorted(ev))
    scn = np.array([ev[e]["scene"] for e in eps])
    suc = np.array([ev[e]["success"] for e in eps])
    lo = np.array([ev[e]["loop"] for e in eps])
    st = np.array([ev[e]["static"] for e in eps])
    nq = np.array([ev[e]["n_queries"] for e in eps])
    clean = (suc == 1) & (lo < 0) & (st < 0)
    cnt = np.array([clean[scn == g].sum() for g in np.unique(scn)])
    print("  groups", len(np.unique(scn)), " clean-success/group: min/med/max",
          cnt.min(), int(np.median(cnt)), cnt.max(), " #grp<3:", (cnt < 3).sum())
    # pooled reference availability (leave-one-group-out worst case)
    print("  pooled clean-success ROWS at q (+-2 window), LOGO worst case:")
    for qq in [30, 34, 36, 37, 38, 39, 40, 41, 42, 44]:
        tot = 0
        worst = 10**9
        for g in np.unique(scn):
            m = clean & (scn != g)
            n = int(np.sum([(max(0, min(nq[i], 100)) > x) for i in np.where(m)[0] for x in range(max(0, qq - 2), qq + 3)]))
            worst = min(worst, n)
        n_all = int(np.sum([[nq[i] > x for x in range(max(0, qq - 2), qq + 3)] for i in np.where(clean)[0]]))
        print(f"    q={qq:3d}  all-group rows={n_all:5d}   LOGO-worst rows={worst:5d}")
    # event counts
    print("  fail loop", ((lo >= 0) & (suc == 0)).sum(), " fail static", ((st >= 0) & (suc == 0)).sum(),
          " fail both", ((lo >= 0) & (st >= 0) & (suc == 0)).sum(),
          " succ loop", ((lo >= 0) & (suc == 1)).sum(), " succ static", ((st >= 0) & (suc == 1)).sum(),
          " clean fail", ((lo < 0) & (st < 0) & (suc == 0)).sum())
