"""Family-wise null + the decisive control: a PURE CLOCK state space.

Differences from mdp_02:
  * ONE permutation sequence per corpus, shared by every (feat, K, t, variant) cell, so
    the maximum over the grid is a legitimate family-wise statistic.
  * adds feat = "clock": the state is a K-quantile bin of the query index t and contains
    ZERO routing information. Everything downstream (transition matrix, absorption,
    empirical per-state rate, cross-fitting) is identical. If the routing MDP does not
    beat the clock MDP, the Markov framing is reading the clock.
  * adds feat = "shuf": routing states, but the within-branch chunk ORDER is permuted
    before the chain is built and before scoring. Keeps each branch's bag of states,
    destroys the dynamics. Isolates what the Markov/order structure contributes.
  * stores the whole null grid so the cross-corpus sign-agreement null can be built.
"""
import time
import numpy as np
import pandas as pd

import mdp_lib as L
from mdp_02_detect import FoldModel, VARIANTS, assemble, W

FEATS = ("cell1280", "mean32", "clock", "shuf")
NDRAW = 200


def clock_states(valid, K):
    """State = K-quantile bin of the absolute query index t. Zero routing information."""
    B, Tmax = valid.shape
    tt = np.tile(np.arange(Tmax), (B, 1))
    edges = np.quantile(tt[valid].astype(float), np.linspace(0, 1, K + 1)[1:-1])
    S = np.full((B, Tmax), -1, np.int16)
    S[valid] = np.searchsorted(edges, tt[valid].astype(float)).astype(np.int16)
    return S


def shuffle_within_branch(S, valid, seed=11):
    rng = np.random.default_rng(seed)
    S2 = S.copy()
    for b in range(S.shape[0]):
        n = int(valid[b].sum())
        S2[b, :n] = S[b, :n][rng.permutation(n)]
    return S2


def main():
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        y = meta["success"].values.astype(bool)
        g = meta["group"].values
        T = meta["T"].values
        B = len(meta)
        fd = L.folds(meta)
        d = L.agglib.divergence_series(P, valid, "hell|prev")
        base = {t: L.agglib.aggregate(d, t, "mean", W) for t in L.TS}

        # one shared permutation sequence; draw 0 = the observed labels
        rng = np.random.default_rng(4242)
        Y = np.empty((NDRAW + 1, B), bool)
        Y[0] = y
        for i in range(1, NDRAW + 1):
            yp = y.copy()
            for gg in np.unique(g):
                idx = np.where(g == gg)[0]
                yp[idx] = y[idx][rng.permutation(len(idx))]
            Y[i] = yp

        Fcell = L.features(P, valid, "cell1280")
        cells, raw, res, res_signed = [], [], [], []
        for feat in FEATS:
            F = Fcell if feat in ("cell1280", "shuf") else (
                L.features(P, valid, feat) if feat == "mean32" else None)
            for K in L.KS:
                t0 = time.time()
                models = []
                for f in range(L.NFOLD):
                    tr = np.where(fd != f)[0]
                    te = np.where(fd == f)[0]
                    if feat == "clock":
                        S = clock_states(valid, K)
                    elif feat == "shuf":
                        S0, _, _ = L.fit_states(F, valid, tr, K)
                        S = shuffle_within_branch(S0, valid)
                    else:
                        S, _, _ = L.fit_states(F, valid, tr, K)
                    models.append(FoldModel(S, valid, tr, te, K, L.TS))
                cellidx = {}
                for t in L.TS:
                    for v in VARIANTS:
                        cellidx[(t, v)] = len(cellidx)      # LOCAL column index
                        cells.append((feat, K, t, v))
                rr = np.full((NDRAW + 1, len(cellidx)), np.nan)
                ss = np.full((NDRAW + 1, len(cellidx)), np.nan)
                sg = np.full((NDRAW + 1, len(cellidx)), np.nan)
                for i in range(NDRAW + 1):
                    yy = Y[i]
                    sc = assemble(models, ~yy, B, L.TS)
                    for t in L.TS:
                        alive = T > t
                        b = base[t]
                        for v in VARIANTS:
                            s = sc[t][v]
                            _, dd, _ = L.auc_pair(s[alive], yy[alive], g[alive])
                            r = L.agglib.rank_residualise(s, b, g)
                            ar, dr, _ = L.auc_pair(r[alive], yy[alive], g[alive])
                            j = cellidx[(t, v)]
                            rr[i, j], ss[i, j], sg[i, j] = dd, dr, ar
                raw.append(rr); res.append(ss); res_signed.append(sg)
                print(f"{tag} {feat} K={K} {time.time()-t0:.1f}s obs_raw_max="
                      f"{np.nanmax(rr[0]):.3f} obs_res_max={np.nanmax(ss[0]):.3f}", flush=True)
            if feat == "mean32":
                del F
        RAW = np.concatenate(raw, 1)
        RES = np.concatenate(res, 1)
        SGN = np.concatenate(res_signed, 1)
        cd = pd.DataFrame(cells, columns=["feat", "K", "t", "variant"])
        np.savez_compressed(f"{L.OUT}/null_{tag}.npz", raw=RAW, res=RES, sgn=SGN,
                            cells=cd.to_numpy().astype(str))
        cd.to_csv(f"{L.OUT}/null_cells_{tag}.csv", index=False)
        print(tag, "saved", RAW.shape, flush=True)


if __name__ == "__main__":
    main()
