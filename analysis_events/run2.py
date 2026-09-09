"""v2: priority-ordered grouped-CV evaluation with incremental persistence.

Feature blocks all get the same treatment (standardise -> PCA<=NC on the
training fold only) and the same classifier.  Tier-1 (deployable proprio +
own action chunk + relative geometry to the goal) is the meaningful control;
tier-2 (privileged 47-dim sim_state, all object poses) is a footnote.
"""
import os
import sys
import time
import pickle

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

OUT = "/home/jovyan/work/himoe-vla/analysis_events"
# routing needs many more PCs than the controls to reach the same explained
# variance (48 PCs: routing 61%, tier1 99.3%, tier2 96.2%), so routing is given
# the larger budget.  The capacity-matched null is tier1+routing_perm.
NCB = {"PH": 5, "T1": 64, "T2": 64, "R": 96}
ROWCAP = 8000
MIN_POS = 30
HGB = dict(max_iter=100, learning_rate=0.12, max_leaf_nodes=8,
           min_samples_leaf=30, l2_regularization=1.0,
           early_stopping=False, random_state=0)

BASE_FS = ["phase", "tier1", "routing", "routing_shufchunk",
           "tier1+routing", "tier1+routing_perm", "tier2"]
HEAD_FS = BASE_FS + ["tier2+routing"]

FSDEF = {"phase": ("PH",), "tier1": ("T1",), "tier2": ("T2",), "routing": ("R",),
         "routing_shufchunk": ("Rs",), "tier1+routing": ("T1", "R"),
         "tier1+routing_perm": ("T1", "Rp"), "tier2+routing": ("T2", "R")}

# priority order: proprio-blind family first, then core physical events
PRIORITY = ["whichpot_m1", "whichpot_m2", "whichpot_m4",
            "dispnc1_m4", "dispnc2_m4", "dispnc1_m2",
            "disp1_m1", "disp1_m2", "disp1_m4",
            "disp2_m1", "disp2_m2", "disp2_m4",
            "tilt1_m1", "tilt1_m2", "tilt1_m4",
            "tilt2_m1", "tilt2_m2", "tilt2_m4",
            "grip_m1", "grip_m2", "grip_m4",
            "drop1_m1", "drop1_m2", "drop1_m4",
            "drop2_m1", "drop2_m2", "drop2_m4",
            "dropfail1_m1", "dropfail1_m2", "dropfail1_m4",
            "dropfail2_m1", "dropfail2_m2", "dropfail2_m4",
            "jext_m1", "jext_m2", "jext_m4",
            "jlim_m1", "jlim_m2", "jlim_m4",
            "twist_m1", "twist_m2", "twist_m4",
            "goalreg1_m4", "goalreg2_m4"]
HEADLINE = {"whichpot_m2", "dispnc1_m4", "disp1_m2", "tilt1_m4", "drop2_m2"}
if os.environ.get("EVENTS"):
    PRIORITY = os.environ["EVENTS"].split(",")
ROWCAP = int(os.environ.get("ROWCAP", ROWCAP))


def pooled_auc(y, s, g):
    tot, acc, per = 0.0, 0.0, []
    for gg in np.unique(g):
        i = g == gg
        yy, ss = y[i], s[i]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos == 0 or nneg == 0:
            continue
        a = roc_auc_score(yy, ss)
        w = npos * nneg
        acc += a * w
        tot += w
        per.append((int(gg), float(a), int(w), npos, nneg))
    return (acc / tot if tot else np.nan), per


def reduce_blocks(cache, meta, groups, folds):
    raw = {"PH": cache["PH"], "T1": cache["T1"], "T2": cache["T2"],
           "R": cache["R"].astype(np.float32)}
    Z = {}
    for fi, (tr, te) in enumerate(folds):
        for b, X in raw.items():
            sc = StandardScaler().fit(X[tr])
            Xs = sc.transform(X).astype(np.float32)
            nc = min(NCB[b], X.shape[1], len(tr) - 1)
            p = PCA(n_components=nc, svd_solver="randomized", random_state=0).fit(Xs[tr])
            Z[(b, fi)] = p.transform(Xs).astype(np.float32)
            del Xs
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


def one_fit(Z, folds, y, mask, fs, blocks, fi):
    tr, te = folds[fi]
    trm = tr[mask[tr]]
    tem = te[mask[te]]
    if len(trm) == 0 or len(tem) == 0:
        return fs, tem, None
    ytr = y[trm]
    if ytr.sum() < 2 or (1 - ytr).sum() < 2:
        return fs, tem, None
    if len(trm) > ROWCAP:                       # keep every positive, subsample negatives
        rs = np.random.default_rng(7 + fi)
        pos = np.where(ytr > 0)[0]
        neg = np.where(ytr == 0)[0]
        keep = np.sort(np.r_[pos, rs.choice(neg, size=max(1, ROWCAP - len(pos)),
                                            replace=False)]) if len(pos) < ROWCAP else \
            np.sort(rs.choice(len(ytr), size=ROWCAP, replace=False))
        trm, ytr = trm[keep], ytr[keep]
    Xtr = np.hstack([Z[(b, fi)][trm] for b in blocks])
    Xte = np.hstack([Z[(b, fi)][tem] for b in blocks])
    clf = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
    return fs, tem, clf.predict_proba(Xte)[:, 1].astype(np.float32)


def run(tag, njobs=26):
    otag = os.environ.get("OUTTAG", tag)
    cache = np.load(f"{OUT}/cache_{tag}.npz")
    meta = pd.read_parquet(f"{OUT}/meta_{tag}.parquet")
    groups = meta.worker.values
    ug = np.unique(groups)
    folds = [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in ug]
    t0 = time.time()
    Z = reduce_blocks(cache, meta, groups, folds)
    print(f"[{tag}] PCA {time.time()-t0:.0f}s  folds={len(ug)}", flush=True)

    csv = f"{OUT}/auc_{otag}.csv"
    pg = f"{OUT}/pergroup_{otag}.pkl"
    sc = f"{OUT}/oof_{otag}.pkl"
    PER, SCO, ROWS = {}, {}, []
    if os.path.exists(csv):
        os.remove(csv)
    for ev in PRIORITY:
        if "y_" + ev not in cache:
            continue
        y = cache["y_" + ev].astype(np.float32)
        mask = cache["m_" + ev]
        npos = int(y[mask].sum())
        nneg = int((1 - y[mask]).sum())
        if npos < MIN_POS or nneg < MIN_POS:
            print(f"[{tag}] SKIP {ev}: n_pos={npos} n_neg={nneg}", flush=True)
            continue
        fss = HEAD_FS if ev in HEADLINE else BASE_FS
        res = Parallel(n_jobs=njobs, prefer="threads")(
            delayed(one_fit)(Z, folds, y, mask, f, FSDEF[f], fi)
            for f in fss for fi in range(len(folds)))
        SC = {f: np.full(len(y), np.nan, np.float32) for f in fss}
        for f, tem, s in res:
            if s is not None:
                SC[f][tem] = s
        for f in fss:
            scores = SC[f]
            ok = mask & ~np.isnan(scores)
            a, per = pooled_auc(y[ok].astype(int), scores[ok], groups[ok])
            PER[(ev, f)] = per
            SCO[(ev, f)] = scores
            ROWS.append(dict(corpus=otag, event=ev.split("_m")[0],
                             horizon=int(ev.split("_m")[1]), feature_block=f,
                             metric="grouped_auc", value=a, n_pos=npos, n_neg=nneg))
        pd.DataFrame(ROWS).to_csv(csv, index=False)
        pickle.dump(PER, open(pg, "wb"))
        pickle.dump(SCO, open(sc, "wb"))
        d = {r["feature_block"]: r["value"] for r in ROWS if
             r["event"] + "_m" + str(r["horizon"]) == ev}
        print(f"[{tag}] {ev:16s} pos={npos:5d} " +
              " ".join(f"{k}={v:.3f}" for k, v in d.items()) +
              f"  [{time.time()-t0:.0f}s]", flush=True)
    print(f"[{tag}] finished {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 26)
