"""Evaluate the aggregation-method grid.

Protocol (identical to the 2026-08-29 organisation sweep so numbers are comparable):
  * target = BDDL success; risk set at query t = branches with T > t (all alive for t<=30)
  * AUC computed WITHIN group (worker / init_state_id), pooled by pair count
  * raw directional auc = P(feature_failure > feature_success); detection = max(auc, 1-auc)
  * residual grid: each cell's per-branch score is rank-residualised on the BASELINE
    cell's per-branch score, within group, before the AUC.  The baseline therefore
    residualises to exactly 0.5 and is NOT in the residual family.
  * 200-draw label permutation null, permuted at branch level WITHIN group once per
    draw and reused across all t; family statistic = max detection AUC over the grid.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import agg_core as A

TS = A.TS
NPERM = 200
CSV = "/tmp/moe_agg_placement.csv"
COLS = ["corpus", "placement", "layer_op", "denoise_op", "token_op",
        "layer_scope", "denoise_scope", "token_scope", "t",
        "auc", "auc_resid", "n_pos", "n_neg"]
BIDX = A.CFG_IDX[A.BASELINE]


def prep(cube, y, g):
    out = {}
    for j, t in enumerate(TS):
        X = cube[:, :, j].astype(np.float64)
        groups, npairs, npos, nneg = [], 0, 0, 0
        for gg in np.unique(g):
            m = g == gg
            ns, nf = int(y[m].sum()), int((1 - y[m]).sum())
            if ns == 0 or nf == 0:
                continue
            R = rankdata(X[m], axis=0)
            rb = R[:, BIDX]
            rbc = rb - rb.mean()
            Rc = R - R.mean(0, keepdims=True)
            den = float(rbc @ rbc)
            beta = (rbc @ Rc) / den if den > 0 else np.zeros(R.shape[1])
            Rr = rankdata(Rc - np.outer(rbc, beta), axis=0)
            # within-group rank correlation with the baseline (for the redundancy table)
            sd, sdb, n = Rc.std(0), rbc.std(), len(rb)
            cov = (rbc @ Rc) / n
            rho = np.where((sd > 0) & (sdb > 0), cov / np.maximum(sd * sdb, 1e-12), 0.0)
            groups.append(dict(gid=int(gg), R=R.astype(np.float32),
                               Rr=Rr.astype(np.float32), y=y[m], ns=ns, nf=nf,
                               rho=rho, w=ns * nf, sd=sd))
            npairs += ns * nf
            npos += ns
            nneg += nf
        out[t] = dict(groups=groups, npairs=npairs, npos=npos, nneg=nneg)
    return out


def pooled_auc(pt, key, yl=None):
    num = np.zeros(pt["groups"][0][key].shape[1])
    for k, G in enumerate(pt["groups"]):
        yy = G["y"] if yl is None else yl[k]
        fail = (yy == 0).astype(np.float32)
        num += fail @ G[key] - G["nf"] * (G["nf"] + 1) / 2.0
    return num / pt["npairs"]


def perm(P, key, seed, nperm=NPERM):
    """Returns (family_max_null (nperm,), pointwise exceedance counts {t: (NCFG,)})."""
    rng = np.random.default_rng(seed)
    perms = {}
    for G in P[TS[0]]["groups"]:
        Y = np.stack([rng.permutation(G["y"]) for _ in range(nperm)])
        perms[G["gid"]] = (Y == 0).astype(np.float32)
    best = np.zeros(nperm)
    draws = {}
    for t in TS:
        pt = P[t]
        num = np.zeros((nperm, pt["groups"][0][key].shape[1]), np.float32)
        for G in pt["groups"]:
            num += perms[G["gid"]] @ G[key] - G["nf"] * (G["nf"] + 1) / 2.0
        a = num / pt["npairs"]
        d = np.maximum(a, 1 - a)
        draws[t] = d
        best = np.maximum(best, d.max(1))
    return best, draws


def run(tag, seed=20260829):
    z = np.load(f"/tmp/agg_cube_{tag}.npz")
    cube, y, g = z["cube"], z["y"], z["g"]
    P = prep(cube, y, g)
    sizes = [(G["gid"], G["ns"], G["nf"]) for G in P[TS[0]]["groups"]]
    grid = {}
    for t in TS:
        grid[(t, "raw")] = pooled_auc(P[t], "R")
        grid[(t, "res")] = pooled_auc(P[t], "Rr")

    rows = []
    for t in TS:
        gr, ge = grid[(t, "raw")], grid[(t, "res")]
        for i, c in enumerate(A.CFGS):
            (ls, lp, lo), (ds, dp, do), (ts_, tp, to) = c
            rows.append((tag, A.placement_str(c), f"{lp}:{lo}", f"{dp}:{do}", f"{tp}:{to}",
                         ls, ds, ts_, t, round(float(gr[i]), 6), round(float(ge[i]), 6),
                         P[t]["npos"], P[t]["nneg"]))
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
    print(f"[{tag}] appended {len(df)} rows -> {CSV} (groups {sizes})", flush=True)

    nulls, pw, drawsave = {}, {}, {}
    for i, (nm, key) in enumerate([("raw", "R"), ("res", "Rr")]):
        b, d = perm(P, key, seed + i)
        nulls[nm] = b
        drawsave[nm] = np.stack([d[t] for t in TS], -1).astype(np.float32)  # (nperm,NCFG,nt)
        for t in TS:
            obs = np.maximum(grid[(t, nm)], 1 - grid[(t, nm)])
            pw[f"{nm}_{t}"] = (1 + (d[t] >= obs[None, :]).sum(0)) / (1 + NPERM)
        print(f"[{tag}] {nm} family null: mean {b.mean():.4f} p95 {np.percentile(b,95):.4f} "
              f"max {b.max():.4f}", flush=True)

    # redundancy with the baseline (pair-count-weighted within-group rank correlation)
    rho = {}
    for t in TS:
        w = np.array([G["w"] for G in P[t]["groups"]], float)
        M = np.stack([G["rho"] for G in P[t]["groups"]])
        rho[t] = (w[:, None] * M).sum(0) / w.sum()

    np.savez(f"/tmp/agg_stats_{tag}.npz",
             **{f"auc_{k[1]}_{k[0]}": v for k, v in grid.items()},
             **{f"pw_{k}": v for k, v in pw.items()},
             **{f"rho_{t}": rho[t] for t in TS},
             null_raw=nulls["raw"], null_res=nulls["res"],
             npos=np.array([P[t]["npos"] for t in TS]),
             nneg=np.array([P[t]["nneg"] for t in TS]))
    np.savez(f"/tmp/agg_draws_{tag}.npz", **drawsave)
    print(f"[{tag}] wrote /tmp/agg_stats_{tag}.npz + /tmp/agg_draws_{tag}.npz", flush=True)


if __name__ == "__main__":
    import sys
    tags = sys.argv[1:] or ["A", "B"]
    if os.path.exists(CSV) and tags == ["A", "B"]:
        os.remove(CSV)
    for tag in tags:
        run(tag)
