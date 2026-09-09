"""Within-group label-permutation null for the whole statistic family, both corpora, both slices.

One branch-level permutation per draw, reused across ALL t and BOTH slices, so the
nesting of the risk sets (T > t) is preserved exactly.
Returns, per draw, the FULL permuted AUC grid -> family maxima computed in the parent.
"""
import os, sys, time, pickle
sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_stat_axis")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from stats_lib import (consecutive_series, instantaneous_bank, historical_bank,
                       reference_bank, within_group_auc)
from run_grid import load, TS

OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
NDRAW = 200
SLICES = ["main", "front"]
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
        lf = {}
        B = P.shape[0]
        for t in TS:
            alive = valid[:, t]
            bank = {}
            for k, v in instantaneous_bank(P[alive, t], nt, nl).items():
                a = np.full(B, np.nan, np.float32); a[alive] = v; bank[k] = a
            bank.update(historical_bank(P, valid, d_hel, d_js, d_l1, t))
            lf[t] = bank
        G[sl] = dict(P=P, valid=valid, d_hel=d_hel, lf=lf)
    G["meta"] = meta   # identical branch ordering for both slices
    G["keys"] = pickle.load(open(f"{OUT}/keys_{tag}.pkl", "rb"))


def draw_auc(y):
    """Return permuted-AUC vector aligned to G['keys'] = list of (slice, stat, t)."""
    meta = G["meta"]; g = meta["group"].values
    res = {}
    for sl in SLICES:
        S = G[sl]
        for t in TS:
            if not S["valid"][:, t].any():
                continue
            bank = dict(S["lf"][t])
            bank.update(reference_bank(S["P"], S["valid"], S["d_hel"], t, g,
                                       y.astype(int), seed=abs(hash((sl, t))) % 10**6))
            for k, v in bank.items():
                a, n = within_group_auc(v, y, g)
                res[(sl, k, t)] = a if (n and np.isfinite(a)) else np.nan
    return np.array([res.get(k, np.nan) for k in G["keys"]], np.float64)


def _work(args):
    tag, seed = args
    init(tag)
    meta = G["meta"]; g = meta["group"].values
    y0 = meta["success"].values.astype(bool)
    rng = np.random.default_rng(10_000 + seed)
    y = y0.copy()
    for gg in np.unique(g):                       # permute WITHIN group, branch level
        m = np.where(g == gg)[0]
        y[m] = y0[m][rng.permutation(len(m))]
    return draw_auc(y)


if __name__ == "__main__":
    obs = pd.read_csv(f"{OUT}/grid_raw.csv")
    for tag in ["A", "B"]:
        o = obs[obs.corpus == tag]
        keys = [(r.slice, r.stat, int(r.t)) for r in o.itertuples()]
        pickle.dump(keys, open(f"{OUT}/keys_{tag}.pkl", "wb"))
        npy = f"/tmp/perm_null_stat_{tag}.npy"
        if os.path.exists(npy) and np.load(npy).shape == (NDRAW, len(keys)):
            print(tag, "null cached, skip"); continue
        t0 = time.time()
        acc = []
        with ProcessPoolExecutor(max_workers=32) as ex:
            for i, r in enumerate(ex.map(_work, [(tag, s) for s in range(NDRAW)], chunksize=2)):
                acc.append(r)
                if (i + 1) % 50 == 0:
                    np.save(npy + ".part.npy", np.array(acc))
                    print(f"  {tag} {i+1}/{NDRAW}  {time.time()-t0:.0f}s", flush=True)
        arr = np.array(acc)
        np.save(npy, arr)
        if os.path.exists(npy + ".part.npy"):
            os.remove(npy + ".part.npy")
        print(tag, "null done", arr.shape, f"{time.time()-t0:.0f}s", flush=True)
