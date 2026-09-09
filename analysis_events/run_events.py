"""Grouped-CV evaluation of MoE routing vs matched physical controls for
discrete physical events.  Every feature block gets the same reduction
(standardise -> PCA<=NC) and the same classifier."""
import os
import sys
import time
import json

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = "/home/jovyan/work/himoe-vla/analysis_events"
NC = 128
MIN_POS = 30
HGB = dict(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
           min_samples_leaf=30, l2_regularization=1.0,
           early_stopping=False, random_state=0)

FEATSETS = {
    "phase":            ("PH",),
    "tier1":            ("T1",),
    "tier2":            ("T2",),
    "routing":          ("R",),
    "routing_shufchunk": ("Rs",),
    "tier1+routing":    ("T1", "R"),
    "tier2+routing":    ("T2", "R"),
    "tier1+routing_perm": ("T1", "Rp"),
    "tier2+routing_perm": ("T2", "Rp"),
}


def pooled_auc(y, s, g):
    """within-group AUC pooled by pair count; returns (auc, per-group list)."""
    tot_w, acc, per = 0.0, 0.0, []
    for gg in np.unique(g):
        i = g == gg
        yy, ss = y[i], s[i]
        npos, nneg = int(yy.sum()), int((~yy.astype(bool)).sum())
        if npos == 0 or nneg == 0:
            continue
        a = roc_auc_score(yy, ss)
        w = npos * nneg
        acc += a * w
        tot_w += w
        per.append((gg, a, w, npos, nneg))
    return (acc / tot_w if tot_w else np.nan), per


def boot_ci(perA, perB=None, n=2000, seed=0):
    """grouped bootstrap over CV groups of pooled AUC (or its difference)."""
    rng = np.random.default_rng(seed)
    ga = {p[0]: p for p in perA}
    keys = sorted(ga)
    if perB is not None:
        gb = {p[0]: p for p in perB}
        keys = [k for k in keys if k in gb]
    if len(keys) < 2:
        return (np.nan, np.nan)
    out = []
    for _ in range(n):
        ks = rng.choice(keys, size=len(keys), replace=True)
        wa = sum(ga[k][2] for k in ks)
        va = sum(ga[k][1] * ga[k][2] for k in ks) / wa
        if perB is None:
            out.append(va)
        else:
            wb = sum(gb[k][2] for k in ks)
            vb = sum(gb[k][1] * gb[k][2] for k in ks) / wb
            out.append(va - vb)
    return tuple(np.percentile(out, [2.5, 97.5]))


def reduce_blocks(cache, meta, groups, folds, rng_seed=0):
    """Per fold, fit scaler+PCA on the training rows of each raw block."""
    raw = {"PH": cache["PH"], "T1": cache["T1"], "T2": cache["T2"],
           "R": cache["R"].astype(np.float32)}
    Z = {}
    for fi, (tr, te) in enumerate(folds):
        for b, X in raw.items():
            sc = StandardScaler().fit(X[tr])
            Xs = sc.transform(X)
            nc = min(NC, X.shape[1], len(tr) - 1)
            p = PCA(n_components=nc, svd_solver="randomized", random_state=0).fit(Xs[tr])
            Z[(b, fi)] = p.transform(Xs).astype(np.float32)
        # derived routing variants (row permutations commute with the linear map)
        R = Z[("R", fi)]
        rng = np.random.default_rng(1000 + fi)
        idx = np.arange(len(R))
        for ep in np.unique(meta.episode.values):
            i = np.where(meta.episode.values == ep)[0]
            idx[i] = rng.permutation(i)
        Z[("Rs", fi)] = R[idx]
        idx2 = np.arange(len(R))
        for gg in np.unique(groups):
            i = np.where(groups == gg)[0]
            idx2[i] = rng.permutation(i)
        Z[("Rp", fi)] = R[idx2]
    del raw
    return Z


def one_cell(Z, folds, groups, y, mask, fs, blocks):
    scores = np.full(len(y), np.nan)
    for fi, (tr, te) in enumerate(folds):
        trm = tr[mask[tr]]
        tem = te[mask[te]]
        if len(trm) == 0 or len(tem) == 0:
            continue
        ytr = y[trm]
        if ytr.sum() < 2 or (1 - ytr).sum() < 2:
            continue
        Xtr = np.hstack([Z[(b, fi)][trm] for b in blocks])
        Xte = np.hstack([Z[(b, fi)][tem] for b in blocks])
        clf = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
        scores[tem] = clf.predict_proba(Xte)[:, 1]
    ok = mask & ~np.isnan(scores)
    a, per = pooled_auc(y[ok].astype(int), scores[ok], groups[ok])
    return fs, a, per, int(y[mask].sum()), int((1 - y[mask]).sum())


def run(tag, events, njobs=14):
    cache = np.load(f"{OUT}/cache_{tag}.npz")
    meta = pd.read_parquet(f"{OUT}/meta_{tag}.parquet")
    groups = meta.worker.values
    ug = np.unique(groups)
    folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in ug]
    t0 = time.time()
    Z = reduce_blocks(cache, meta, groups, folds)
    print(f"[{tag}] PCA done in {time.time()-t0:.0f}s, {len(ug)} folds", flush=True)

    jobs, keys = [], []
    for ev in events:
        y = cache["y_" + ev].astype(np.float32)
        mask = cache["m_" + ev]
        if y[mask].sum() < MIN_POS:
            continue
        for fs, blocks in FEATSETS.items():
            keys.append((ev, fs))
            jobs.append(delayed(one_cell)(Z, folds, groups, y, mask, fs, blocks))
    print(f"[{tag}] {len(jobs)} cells", flush=True)
    res = Parallel(n_jobs=njobs, prefer="threads", verbose=5)(jobs)

    store = {}
    rows = []
    for (ev, fs), (fs2, a, per, npos, nneg) in zip(keys, res):
        store[(ev, fs)] = per
        rows.append(dict(corpus=tag, event=ev.split("_m")[0], horizon=int(ev.split("_m")[1]),
                         feature_block=fs, metric="grouped_auc", value=a,
                         n_pos=npos, n_neg=nneg))
    df = pd.DataFrame(rows)
    np.save(f"{OUT}/pergroup_{tag}.npy",
            np.array([(k[0], k[1], json.dumps([[str(p[0])] + [float(x) for x in p[1:]] for p in v]))
                      for k, v in store.items()], dtype=object), allow_pickle=True)
    df.to_csv(f"{OUT}/auc_{tag}.csv", index=False)

    # increments with grouped bootstrap CIs
    inc = []
    for ev in df.event.unique():
        for h in sorted(df[df.event == ev].horizon.unique()):
            key = f"{ev}_m{h}"
            get = lambda f: store.get((key, f))
            for base, comb in [("tier1", "tier1+routing"), ("tier2", "tier2+routing"),
                               ("tier1+routing_perm", "tier1+routing"),
                               ("tier2+routing_perm", "tier2+routing"),
                               ("phase", "routing")]:
                pa, pb = get(comb), get(base)
                if pa is None or pb is None:
                    continue
                da = pooled_auc_from(pa) - pooled_auc_from(pb)
                lo, hi = boot_ci(pa, pb)
                inc.append(dict(corpus=tag, event=ev, horizon=h, base=base, combined=comb,
                                delta=da, lo=lo, hi=hi))
    pd.DataFrame(inc).to_csv(f"{OUT}/increments_{tag}.csv", index=False)
    print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)
    return df


def pooled_auc_from(per):
    w = sum(p[2] for p in per)
    return sum(p[1] * p[2] for p in per) / w if w else np.nan


EVENTS = [f"{b}_m{m}" for b in ["disp1", "disp2", "tilt1", "tilt2", "grip",
                                "drop1", "drop2", "dropfail1", "dropfail2",
                                "jext", "jlim", "twist", "leavegoal1", "leavegoal2"]
          for m in (1, 2, 4)]

if __name__ == "__main__":
    tag = sys.argv[1]
    nj = int(sys.argv[2]) if len(sys.argv) > 2 else 14
    run(tag, EVENTS, nj)
