"""Within-group label-permutation null for the family-wise max |AUC-0.5|.

Family = every statistic x every t on the MAIN slice. Labels are permuted within
group; the reference (R:) statistics are FULLY RECOMPUTED under the permuted labels,
so this doubles as the leakage check on the cross-fitted success-reference family.
Resumable: per-draw results cached to /tmp/moe_perm_<tag>.pkl.
"""
import sys, os, time, pickle
sys.path.insert(0, "/home/jovyan/work/himoe-vla/analysis_stat_axis")
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from stats_lib import (consecutive_series, instantaneous_bank, historical_bank,
                       reference_bank, within_group_auc)
from run_grid import load, TS

OUT = "/home/jovyan/work/himoe-vla/analysis_stat_axis"
NDRAW = 200
G = {}


def init(tag, slice_name="main"):
    P, valid, meta, nl, nt = load(tag, slice_name)
    d0 = consecutive_series(P, valid, "hellinger")
    dser = (d0, consecutive_series(P, valid, "js"), consecutive_series(P, valid, "l1"))
    lf = {}
    for t in TS:
        if not valid[:, t].any():
            continue
        alive = valid[:, t]; B = P.shape[0]; bank = {}
        for k, v in instantaneous_bank(P[alive, t], nt, nl).items():
            a = np.full(B, np.nan, np.float32); a[alive] = v; bank[k] = a
        bank.update(historical_bank(P, valid, *dser, t))
        lf[t] = bank
    G.update(dict(P=P, valid=valid, meta=meta, dser=dser, lf=lf))


def one_draw(seed):
    P, valid, meta, lf = G["P"], G["valid"], G["meta"], G["lf"]
    g = meta["group"].values
    y0 = meta["success"].values.astype(bool)
    rng = np.random.default_rng(seed)
    y = y0.copy()
    for gg in np.unique(g):                      # permute WITHIN group only
        m = np.where(g == gg)[0]
        y[m] = y0[m][rng.permutation(len(m))]
    best = dict(ALL=0.0, INST=0.0, HIST=0.0, REF=0.0, REF_UNSUP=0.0)
    for t in TS:
        if t not in lf:
            continue
        ref = reference_bank(P, valid, G["dser"][0], t, g, y.astype(int), seed=int(seed) + 991)
        for src, bank in (("lf", lf[t]), ("ref", ref)):
            for k, v in bank.items():
                a, n = within_group_auc(v, y, g)
                if not n or not np.isfinite(a):
                    continue
                dv = abs(a - 0.5)
                best["ALL"] = max(best["ALL"], dv)
                if k.startswith("I:"):
                    best["INST"] = max(best["INST"], dv)
                elif k.startswith("H:"):
                    best["HIST"] = max(best["HIST"], dv)
                else:
                    best["REF"] = max(best["REF"], dv)
                    if "cohort" in k:
                        best["REF_UNSUP"] = max(best["REF_UNSUP"], dv)
    return [best[k] for k in ["ALL", "INST", "HIST", "REF", "REF_UNSUP"]]


def _w(a):
    tag, seed = a
    if "P" not in G:
        init(tag)
    return one_draw(seed)


FAMS = ["ALL", "INST", "HIST", "REF", "REF_UNSUP"]
SEL = {"ALL": lambda s: True, "INST": lambda s: s.startswith("I:"),
       "HIST": lambda s: s.startswith("H:"), "REF": lambda s: s.startswith("R:"),
       "REF_UNSUP": lambda s: "cohort" in s}

if __name__ == "__main__":
    obs = pd.read_csv(f"{OUT}/grid_raw.csv")
    rows = []
    for tag in ["A", "B"]:
        pkl = f"/tmp/moe_perm_{tag}.pkl"
        if os.path.exists(pkl):
            d = pickle.load(open(pkl, "rb"))
            print(f"{tag}: reusing cached null {d.shape}", flush=True)
        else:
            t0 = time.time()
            with ProcessPoolExecutor(max_workers=50) as ex:
                d = np.array(list(ex.map(_w, [(tag, s) for s in range(NDRAW)], chunksize=1)))
            pickle.dump(d, open(pkl, "wb"))
            print(f"{tag}: perm {time.time()-t0:.0f}s -> {pkl}", flush=True)
        o = obs[(obs.corpus == tag) & (obs.slice == "main")].copy()
        o["dev"] = (o.auc_success - 0.5).abs()
        for j, fn in enumerate(FAMS):
            f = o[o.stat.map(SEL[fn])]
            if not len(f):
                continue
            arg = f.loc[f.dev.idxmax()]
            null = d[:, j]
            rows.append(dict(corpus=tag, family=fn, n_cells=len(f), best_stat=arg.stat,
                             best_t=int(arg.t), auc_success=round(float(arg.auc_success), 4),
                             auc_failure=round(1 - float(arg.auc_success), 4),
                             obs_dev=round(float(arg.dev), 4),
                             null_med=round(float(np.median(null)), 4),
                             null_p95=round(float(np.quantile(null, .95)), 4),
                             null_max=round(float(null.max()), 4),
                             fwer_p=(1 + int((null >= arg.dev).sum())) / (NDRAW + 1), ndraw=NDRAW))
        pd.DataFrame(rows).to_csv(f"{OUT}/perm_results.csv", index=False)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    print(res.to_string(index=False))
