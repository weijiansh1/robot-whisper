import numpy as np, pandas as pd, json, zarr
from scipy.stats import rankdata
import token_axis_eval as E

ROOT = "/home/jovyan/work/himoe-vla"
PATHS = {"A": E.__dict__.get("A_ZARR") or f"{ROOT}/himoe-route-capture/runs/rolling-star-a100-long-t08-k16-20260828/formal/server/routes.zarr",
         "B": f"{E.B_DIR}/server/routes.zarr"}

print("### denoise-invariance of each token (mean Hellinger between denoise iter 0 and 9, layers 12-15)")
for tag, p in PATHS.items():
    z = zarr.open(store=p, mode="r")
    blk = z["hb_router_probs"][:1500, 4:8, :, :, :].astype(np.float32)
    blk = np.clip(blk, 1e-12, None); blk /= blk.sum(-1, keepdims=True)
    S = np.sqrt(blk)
    h09 = np.sqrt(np.clip(1 - np.einsum("nlke,nlke->nlk", S[:, :, 0], S[:, :, 9]), 0, None)).mean((0, 1))
    hadj = np.sqrt(np.clip(1 - np.einsum("nldke,nldke->nldk", S[:, :, 1:], S[:, :, :-1]), 0, None)).mean((0, 1, 2))
    print(f"  {tag}: H(d0,d9) per token = " + " ".join(f"{v:.4f}" for v in h09))
    print(f"     mean H(d,d-1)         = " + " ".join(f"{v:.4f}" for v in hadj))
    del blk, S

print("\n### corr(score, episode length T) within group, t=30 (degenerate groups skipped)")
FE2 = ["tok00|hell", "act_all_1_10|hell", "act_all_1_10|ent",
       "disp11_hell|ent", "state_vs_actmean_hell|ent"]
for tag in ("A", "B"):
    eids, X, allf, alive = E.build_matrix(tag, 9, 30)
    ymap, gmap = E.meta(tag); idx = np.where(alive)[0]
    ids = np.array([int(e) for e in eids])[idx]
    Xa = X[idx][:, [allf.index(f) for f in FE2]]
    y = np.array([ymap[i] for i in ids]); g = np.array([gmap[i] for i in ids])
    e2, l, off, F, fn = E.load(tag, 9); lm = dict(zip([int(x) for x in e2], l))
    T = np.array([lm[i] for i in ids], float)
    for j, f in enumerate(FE2):
        num = den = 0.0
        for gg in np.unique(g):
            m = g == gg
            if np.std(T[m]) == 0 or np.std(Xa[m, j]) == 0: continue
            num += np.corrcoef(rankdata(Xa[m, j]), rankdata(T[m]))[0, 1] * m.sum(); den += m.sum()
        print(f"  {tag} {f:28s} rho(score,T)={num/den:+.3f}")

print("\n### exploratory (NOT in the tested family): within-group rank-average of state + action route change")
for tag in ("A", "B"):
    ymap, gmap = E.meta(tag)
    for t in (25, 30, 35):
        eids, X, allf, alive = E.build_matrix(tag, 9, t)
        idx = np.where(alive)[0]
        ids = np.array([int(e) for e in eids])[idx]
        y = np.array([ymap[i] for i in ids]); g = np.array([gmap[i] for i in ids])
        A_ = X[idx][:, allf.index("tok00|hell")]; B_ = X[idx][:, allf.index("act_all_1_10|hell")]
        comb = np.zeros(len(ids))
        for gg in np.unique(g):
            m = g == gg
            comb[m] = rankdata(A_[m]) + rankdata(B_[m])
        print(f"  {tag} t={t}: state-alone={1-E.group_auc(A_,y,g)[0]:.4f} "
              f"action-alone={1-E.group_auc(B_,y,g)[0]:.4f} "
              f"rank-avg={1-E.group_auc(comb,y,g)[0]:.4f}")
