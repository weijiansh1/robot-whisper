"""Post-hoc: increments with grouped bootstrap CIs, tier-1-blind stratified
routing AUC, and the persisted long-format result table."""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

OUT = "/home/jovyan/work/himoe-vla/analysis_events"


def pooled(per):
    w = sum(p[2] for p in per)
    return sum(p[1] * p[2] for p in per) / w if w else np.nan


def boot(perA, perB=None, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ga = {p[0]: p for p in perA}
    keys = sorted(ga)
    gb = None
    if perB is not None:
        gb = {p[0]: p for p in perB}
        keys = [k for k in keys if k in gb]
    if len(keys) < 2:
        return np.nan, np.nan
    out = np.empty(n)
    for i in range(n):
        ks = rng.choice(keys, size=len(keys), replace=True)
        wa = sum(ga[k][2] for k in ks)
        va = sum(ga[k][1] * ga[k][2] for k in ks) / wa
        if gb is None:
            out[i] = va
        else:
            wb = sum(gb[k][2] for k in ks)
            out[i] = va - sum(gb[k][1] * gb[k][2] for k in ks) / wb
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


PAIRS = [("phase", "routing"), ("tier1", "routing"),
         ("tier1", "tier1+routing"), ("tier1+routing_perm", "tier1+routing"),
         ("tier2", "tier2+routing"), ("phase", "tier1")]


def main(tag):
    PER = pickle.load(open(f"{OUT}/pergroup_{tag}.pkl", "rb"))
    SCO = pickle.load(open(f"{OUT}/oof_{tag}.pkl", "rb"))
    auc = pd.read_csv(f"{OUT}/auc_{tag}.csv")
    cache = np.load(f"{OUT}/cache_{tag}.npz")
    meta = pd.read_parquet(f"{OUT}/meta_{tag}.parquet")
    groups = meta.worker.values
    evs = sorted({k[0] for k in PER})

    rows = []
    for ev in evs:
        base_ev, hor = ev.split("_m")[0], int(ev.split("_m")[1])
        sel = auc[(auc.event == base_ev) & (auc.horizon == hor)]
        npos = int(sel.n_pos.iloc[0])
        for b, c in PAIRS:
            if (ev, b) not in PER or (ev, c) not in PER:
                continue
            d = pooled(PER[(ev, c)]) - pooled(PER[(ev, b)])
            lo, hi = boot(PER[(ev, c)], PER[(ev, b)])
            rows.append(dict(corpus=tag, event=ev, base=b, combined=c, delta=d,
                             lo=lo, hi=hi, n_pos=npos,
                             sig=bool(lo > 0 or hi < 0) if np.isfinite(lo) else None))
    inc = pd.DataFrame(rows)
    inc.to_csv(f"{OUT}/increments_{tag}.csv", index=False)

    # --------- blindness stratification: routing AUC where tier1 is uninformative
    strat = []
    for ev in evs:
        if (ev, "tier1") not in SCO or (ev, "routing") not in SCO:
            continue
        y = cache["y_" + ev].astype(int)
        mask = cache["m_" + ev]
        s1 = SCO[(ev, "tier1")]
        sr = SCO[(ev, "routing")]
        ok = mask & ~np.isnan(s1) & ~np.isnan(sr)
        base = y[ok].mean()
        # Conditioning on a model's own out-of-fold score forces that model
        # toward chance inside the stratum, so the mirror-image selection
        # (condition on routing, read tier1) is reported as the symmetry check.
        for w, sel_on, name in [(0.25, "t1", "sel_on_tier1_x0.75-1.33"),
                                (0.5, "t1", "sel_on_tier1_x0.5-2.0"),
                                (0.25, "rt", "sel_on_routing_x0.75-1.33"),
                                (0.5, "rt", "sel_on_routing_x0.5-2.0")]:
            ss = s1 if sel_on == "t1" else sr
            sel = ok & (ss > base * (1 - w)) & (ss < base / max(1e-6, (1 - w)))
            yy, gg = y[sel], groups[sel]
            if yy.sum() < 30 or (1 - yy).sum() < 30:
                continue
            def pa(sc):
                tot = acc = 0.0
                for g in np.unique(gg):
                    i = gg == g
                    if yy[i].sum() == 0 or (1 - yy[i]).sum() == 0:
                        continue
                    wgt = yy[i].sum() * (1 - yy[i]).sum()
                    acc += roc_auc_score(yy[i], sc[sel][i]) * wgt
                    tot += wgt
                return acc / tot if tot else np.nan
            strat.append(dict(corpus=tag, event=ev, stratum=name, n=int(sel.sum()),
                              n_pos=int(yy.sum()), base_rate=float(base),
                              auc_tier1=pa(s1), auc_routing=pa(sr)))
    pd.DataFrame(strat).to_csv(f"{OUT}/stratified_{tag}.csv", index=False)
    print(inc.to_string(index=False))
    print()
    print(pd.DataFrame(strat).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
