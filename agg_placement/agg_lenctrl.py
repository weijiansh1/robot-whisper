"""Scope-note control: within-group episode length T separates success/failure at
det AUC 0.989 (A) / 0.999 (B) -- the length confound flagged in section 3.3 of the
2026-08-29 organisation sweep.  It affects the baseline exactly as much as every
candidate, so the *relative* question (does an aggregation beat the baseline) is
unaffected; but it caps what the absolute AUCs mean.

Here: residualise the whole grid on (baseline, T) jointly, within group, and
re-price with the same 200-draw permutation null and the same joint criterion.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import agg_core as A

TS = A.TS
NP = 200
BIDX = A.CFG_IDX[A.BASELINE]


def lengths(tag, eids):
    if tag == "A":
        lab = pd.read_csv(A.A_LAB)
        m = dict(zip(lab.episode_id.astype(int), lab.inference_calls))
    else:
        s = json.load(open(f"{A.B_DIR}/client/summaries.json"))
        m = {i: x["inference_calls"] for i, x in enumerate(s)}
    return np.array([m[int(e)] for e in eids], float)


def build(tag, mode):
    """mode: 'T' -> residualise on length only; 'BT' -> on baseline and length."""
    z = np.load(f"/tmp/agg_cube_{tag}.npz")
    cube, y, g, e = z["cube"], z["y"], z["g"], z["eids"]
    T = lengths(tag, e)
    P = {}
    for j, t in enumerate(TS):
        X = cube[:, :, j].astype(np.float64)
        groups, npairs = [], 0
        for gg in np.unique(g):
            m = g == gg
            ns, nf = int(y[m].sum()), int((1 - y[m]).sum())
            if ns == 0 or nf == 0:
                continue
            R = rankdata(X[m], axis=0)
            Rc = R - R.mean(0, keepdims=True)
            preds = [rankdata(T[m])]
            if mode == "BT":
                preds.insert(0, R[:, BIDX])
            Z = np.stack([p - p.mean() for p in preds], 1)
            beta = np.linalg.lstsq(Z, Rc, rcond=None)[0]
            Rr = rankdata(Rc - Z @ beta, axis=0).astype(np.float32)
            groups.append(dict(gid=int(gg), Rr=Rr, y=y[m], ns=ns, nf=nf))
            npairs += ns * nf
        P[t] = dict(groups=groups, npairs=npairs)
    return P


def pooled(pt, yl=None):
    num = np.zeros(pt["groups"][0]["Rr"].shape[1])
    for k, G in enumerate(pt["groups"]):
        yy = G["y"] if yl is None else yl[k]
        num += (yy == 0).astype(np.float32) @ G["Rr"] - G["nf"] * (G["nf"] + 1) / 2.0
    return num / pt["npairs"]


def perm(P, seed):
    rng = np.random.default_rng(seed)
    pm = {}
    for G in P[TS[0]]["groups"]:
        pm[G["gid"]] = (np.stack([rng.permutation(G["y"]) for _ in range(NP)]) == 0
                        ).astype(np.float32)
    d = {}
    for t in TS:
        pt = P[t]
        num = np.zeros((NP, pt["groups"][0]["Rr"].shape[1]), np.float32)
        for G in pt["groups"]:
            num += pm[G["gid"]] @ G["Rr"] - G["nf"] * (G["nf"] + 1) / 2.0
        a = num / pt["npairs"]
        d[t] = np.maximum(a, 1 - a)
    return np.stack([d[t] for t in TS], -1)


def main():
    names = [A.cfg_name(c) for c in A.CFGS]
    for mode, lab in [("T", "residualised on episode length T only"),
                      ("BT", "residualised on BOTH the baseline and T")]:
        print("=" * 96)
        print(f"GRID {lab}")
        obs, drw = {}, {}
        for tag in ("A", "B"):
            P = build(tag, mode)
            a = np.stack([pooled(P[t]) for t in TS], -1)
            obs[tag] = a
            drw[tag] = perm(P, 4242 + (tag == "B"))
            d = np.maximum(a, 1 - a)
            fam = drw[tag].reshape(NP, -1).max(1)
            p = (1 + (fam >= d.max()).sum()) / (1 + NP)
            k = np.unravel_index(d.argmax(), d.shape)
            print(f"  corpus {tag}: max det {d.max():.4f} at {names[k[0]]} t={TS[k[1]]}; "
                  f"null p95 {np.percentile(fam,95):.4f}; family p = {p:.4f}")
            print(f"            baseline cell -> {d[BIDX,2]:.4f} at t=30")
        same = np.sign(obs["A"] - .5) == np.sign(obs["B"] - .5)
        mn = np.where(same, np.minimum(np.maximum(obs["A"], 1 - obs["A"]),
                                       np.maximum(obs["B"], 1 - obs["B"])), 0)
        fam = np.minimum(drw["A"], drw["B"]).reshape(NP, -1).max(1)
        thr = np.percentile(fam, 95)
        p = (1 + (fam >= mn.max()).sum()) / (1 + NP)
        print(f"  JOINT: max min(det_A,det_B) = {mn.max():.4f}; threshold {thr:.4f}; "
              f"joint family p = {p:.4f}; cells clearing = {(mn>=thr).sum()}")
        print(f"  sign agreement = {same.mean()*100:.1f}%")
        for kk in np.argsort(mn.ravel())[::-1][:6]:
            i, j = np.unravel_index(kk, mn.shape)
            print(f"    {names[i]:50s} t={TS[j]} A={np.maximum(obs['A'],1-obs['A'])[i,j]:.3f} "
                  f"B={np.maximum(obs['B'],1-obs['B'])[i,j]:.3f}")


if __name__ == "__main__":
    main()
