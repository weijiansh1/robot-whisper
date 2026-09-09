"""(5-optional) Does MEMORY help? first-order vs second-order vs semi-Markov (dwell-augmented).

Same cross-fitting, same evaluation, same shared permutation sequence.
order1 : state = routing cell
order2 : state = (previous routing cell, current routing cell), observed pairs only
semi   : state = (routing cell, dwell bucket in {1, 2-3, 4-6, 7+})
bag    : memoryless control -- the per-state empirical failure rate is already order-free,
         so `emp_pool_w8` under order1 IS the bag model.
"""
import time
import numpy as np
import pandas as pd

import mdp_lib as L
from mdp_02_detect import FoldModel, W

VARS = ["absorb", "absorb_w8", "emp_pool", "emp_pool_w8"]
NDRAW = 200
ORDERS = {"order1": L.KS, "order2": (4, 8, 16), "semi": L.KS}


def augment(S, valid, mode):
    B, T = S.shape
    if mode == "order1":
        return S.copy(), int(S.max()) + 1
    if mode == "order2":
        K = int(S.max()) + 1
        A = np.full((B, T), -1, np.int64)
        for b in range(B):
            n = int(valid[b].sum())
            s = S[b, :n].astype(np.int64)
            p = np.concatenate([[s[0]], s[:-1]])
            A[b, :n] = p * K + s
        obs = np.unique(A[valid])
        remap = np.full(obs.max() + 1, -1, np.int64)
        remap[obs] = np.arange(len(obs))
        A2 = np.full((B, T), -1, np.int64)
        A2[valid] = remap[A[valid]]
        return A2.astype(np.int32), len(obs)
    if mode == "semi":
        K = int(S.max()) + 1
        A = np.full((B, T), -1, np.int64)
        for b in range(B):
            n = int(valid[b].sum())
            s = S[b, :n]
            d, prev = 0, -99
            for i in range(n):
                d = d + 1 if s[i] == prev else 1
                bk = 0 if d == 1 else (1 if d <= 3 else (2 if d <= 6 else 3))
                A[b, i] = int(s[i]) * 4 + bk
                prev = s[i]
        obs = np.unique(A[valid])
        remap = np.full(obs.max() + 1, -1, np.int64)
        remap[obs] = np.arange(len(obs))
        A2 = np.full((B, T), -1, np.int64)
        A2[valid] = remap[A[valid]]
        return A2.astype(np.int32), len(obs)
    raise ValueError(mode)


def main():
    rows = []
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        y = meta["success"].values.astype(bool)
        g = meta["group"].values
        T = meta["T"].values
        B = len(meta)
        fd = L.folds(meta)
        d = L.agglib.divergence_series(P, valid, "hell|prev")
        base = {t: L.agglib.aggregate(d, t, "mean", W) for t in L.TS}
        rng = np.random.default_rng(4242)
        Y = np.empty((NDRAW + 1, B), bool)
        Y[0] = y
        for i in range(1, NDRAW + 1):
            yp = y.copy()
            for gg in np.unique(g):
                idx = np.where(g == gg)[0]
                yp[idx] = y[idx][rng.permutation(len(idx))]
            Y[i] = yp

        F = L.features(P, valid, "cell1280")
        base_states = {}
        for K in L.KS:
            base_states[K] = [L.fit_states(F, valid, np.where(fd != f)[0], K)[0]
                              for f in range(L.NFOLD)]
        del F
        for mode, kk in ORDERS.items():
            for K in kk:
                t0 = time.time()
                models, ns = [], []
                for f in range(L.NFOLD):
                    A, n = augment(base_states[K][f], valid, mode)
                    ns.append(n)
                    models.append(FoldModel(A.astype(np.int16) if n < 32000 else A,
                                            valid, np.where(fd != f)[0], np.where(fd == f)[0],
                                            n, L.TS))
                obs, null = {}, {}
                for i in range(NDRAW + 1):
                    yy = Y[i]
                    for m in models:
                        m.fit(~yy)
                    sc = {t: {v: np.full(B, np.nan) for v in VARS} for t in L.TS}
                    for m in models:
                        for t in L.TS:
                            te, s = m.scores(t)
                            for v in VARS:
                                sc[t][v][te] = s[v]
                    for t in L.TS:
                        alive = T > t
                        for v in VARS:
                            s = sc[t][v]
                            _, dd, npr = L.auc_pair(s[alive], yy[alive], g[alive])
                            r = L.agglib.rank_residualise(s, base[t], g)
                            ar, dr, _ = L.auc_pair(r[alive], yy[alive], g[alive])
                            if i == 0:
                                obs[(t, v)] = (dd, dr, ar, npr)
                            else:
                                null.setdefault((t, v), []).append((dd, dr))
                for (t, v), o in obs.items():
                    n = np.array(null[(t, v)])
                    rows.append(dict(corpus=tag, mode=mode, K=K, nstates=int(np.mean(ns)), t=t,
                                     variant=v, det_auc=o[0], res_det_auc=o[1],
                                     res_auc_succ=o[2], npairs=o[3],
                                     base_det_auc=L.auc_pair(base[t][T > t], y[T > t],
                                                             g[T > t])[1],
                                     p_raw=(1 + (n[:, 0] >= o[0]).sum()) / (NDRAW + 1),
                                     p_res=(1 + (n[:, 1] >= o[1]).sum()) / (NDRAW + 1)))
                print(f"{tag} {mode} K={K} nstates={int(np.mean(ns))} {time.time()-t0:.1f}s",
                      flush=True)
                pd.DataFrame(rows).to_csv(f"{L.OUT}/order_raw.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{L.OUT}/order_raw.csv", index=False)
    pd.DataFrame(rows).to_csv("/tmp/mdp_order_raw.csv", index=False)


if __name__ == "__main__":
    main()
