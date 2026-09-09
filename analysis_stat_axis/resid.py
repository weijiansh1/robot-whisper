"""Observed + permutation-null grids for the RESIDUALISED family.

Every statistic is rank-residualised, within group, on the pre-existing baseline
(Hellinger to previous chunk, W=8, MAIN slice = deep layers 12-15 / action tokens 1-10 /
denoise 9) before the within-group AUC is taken. The single main-slice baseline is used
for BOTH slices, because that is the pre-existing baseline being defended.
"""
import os, sys, time, pickle
sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_stat_axis")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from stats_lib import (consecutive_series, instantaneous_bank, historical_bank,
                       reference_bank, within_group_auc, rank_residualise)
from run_grid import load, TS

OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
NDRAW = 200
SLICES = ["main", "front"]
BASE_STAT, BASE_SLICE = "H:hel_prev_mean_W8", "main"
G = {}


def init(tag):
    if G.get("tag") == tag:
        return
    G.clear(); G["tag"] = tag
    for sl in SLICES:
        P, valid, meta, nl, nt = load(tag, sl)
        d_hel = consecutive_series(P, valid, "hellinger")
        d_js = consecutive_series(P, valid, "js")
        d_l1 = consecutive_series(P, valid, "l1")
        lf, B = {}, P.shape[0]
        for t in TS:
            alive = valid[:, t]
            bank = {}
            for k, v in instantaneous_bank(P[alive, t], nt, nl).items():
                a = np.full(B, np.nan, np.float32); a[alive] = v; bank[k] = a
            bank.update(historical_bank(P, valid, d_hel, d_js, d_l1, t))
            lf[t] = bank
        G[sl] = dict(P=P, valid=valid, d_hel=d_hel, lf=lf)
    G["meta"] = meta
    G["base"] = {t: G[BASE_SLICE]["lf"][t][BASE_STAT] for t in TS}
    G["keys"] = pickle.load(open(f"{OUT}/keys_{tag}.pkl", "rb"))


def grid(y):
    g = G["meta"]["group"].values
    res = {}
    for sl in SLICES:
        S = G[sl]
        for t in TS:
            if not S["valid"][:, t].any():
                continue
            bank = dict(S["lf"][t])
            bank.update(reference_bank(S["P"], S["valid"], S["d_hel"], t, g,
                                       y.astype(int), seed=abs(hash((sl, t))) % 10**6))
            b = G["base"][t]
            for k, v in bank.items():
                r = rank_residualise(np.asarray(v, np.float64), np.asarray(b, np.float64), g)
                a, n = within_group_auc(r, y, g)
                res[(sl, k, t)] = a if (n and np.isfinite(a)) else np.nan
    return np.array([res.get(k, np.nan) for k in G["keys"]], np.float64)


def _work(args):
    tag, seed = args
    init(tag)
    g = G["meta"]["group"].values
    y0 = G["meta"]["success"].values.astype(bool)
    if seed < 0:                                    # seed -1 = observed labels
        return grid(y0)
    rng = np.random.default_rng(10_000 + seed)
    y = y0.copy()
    for gg in np.unique(g):
        m = np.where(g == gg)[0]
        y[m] = y0[m][rng.permutation(len(m))]
    return grid(y)


if __name__ == "__main__":
    for tag in ["A", "B"]:
        obs_f, null_f = f"/tmp/resid_obs_{tag}.npy", f"/tmp/perm_null_resid_{tag}.npy"
        if not os.path.exists(obs_f):
            init(tag)
            np.save(obs_f, _work((tag, -1)))
            print(tag, "observed residual grid saved", flush=True)
        if os.path.exists(null_f) and np.load(null_f).shape[0] == NDRAW:
            print(tag, "resid null cached"); continue
        t0, acc = time.time(), []
        with ProcessPoolExecutor(max_workers=32) as ex:
            for i, r in enumerate(ex.map(_work, [(tag, s) for s in range(NDRAW)], chunksize=2)):
                acc.append(r)
                if (i + 1) % 50 == 0:
                    np.save(null_f + ".part.npy", np.array(acc))
                    print(f"  {tag} {i+1}/{NDRAW} {time.time()-t0:.0f}s", flush=True)
        np.save(null_f, np.array(acc))
        if os.path.exists(null_f + ".part.npy"):
            os.remove(null_f + ".part.npy")
        print(tag, "resid null done", f"{time.time()-t0:.0f}s", flush=True)
