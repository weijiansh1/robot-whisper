"""Grouped leave-one-group-out evaluation of routing vs. two control tiers.

Every feature block gets the identical treatment:
    center/standardise -> PCA to K whitened components -> RidgeCV (same alpha grid).
Route is pre-reduced (PCA 256 on the 2816-dim prob tensor) so the [p_k, p_k - p_{k-1}]
matrix stays in memory; that can only lose routing information, never add any.

Metrics are computed WITHIN each held-out group and pooled with equal weight,
because the four rolling-star workers have success rates 0%/6%/26%/91%.
"""
import gc
import json
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

CACHE = "/tmp/moe_future_cache"
ROOT = "/home/jovyan/work/himoe-vla"
K = int(os.environ.get("PCA_K", 128))
PRE_K = 256
ALPHAS = np.logspace(-3, 5, 17)
HORIZONS = (1, 2, 4, 8)
TARGETS = ("eef_disp", "eef_path", "obj_disp", "dgoal", "undo", "grip_flips")
BINARY = {"undo"}
BLOCKS = ("clock", "clock_causal", "route", "t1", "t2", "t1+route", "t2+route")


# --------------------------------------------------------------- projections

def fold_projections(d, tr, te, seed=0):
    """Fit every block's reducer on train rows only; return {block: (Xtr, Xte)}."""
    out = {}
    P, branch, kidx = d["P"], d["branch"], d["kidx"]
    prev = np.arange(len(P)) - 1
    prev[kidx == 0] = np.flatnonzero(kidx == 0)          # k=0 -> itself, so dz = 0

    mu = P[tr].mean(0, keepdims=True)
    pre = PCA(n_components=PRE_K, svd_solver="randomized", random_state=seed)
    pre.fit(P[tr] - mu)
    z = pre.transform(P - mu).astype(np.float32)
    evr = float(pre.explained_variance_ratio_.sum())
    route_raw = np.concatenate([z, z - z[prev]], axis=1)
    del z
    gc.collect()

    for name, X, standardise in (
        ("route", route_raw, True),
        ("t1", d["T1"], True),
        ("t2", d["T2"], True),
        ("clock", d["CLOCK"], True),
        ("clock_causal", d["CLOCKC"], True),
    ):
        X = np.asarray(X, np.float32)
        m = X[tr].mean(0, keepdims=True)
        s = X[tr].std(0, keepdims=True) if standardise else np.ones((1, X.shape[1]), np.float32)
        s[s < 1e-8] = 1.0
        Xs = (X - m) / s
        k = min(K, Xs.shape[1], len(tr) - 1)
        if k < Xs.shape[1]:
            p = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=seed)
            p.fit(Xs[tr])
            out[name] = (p.transform(Xs[tr]).astype(np.float32),
                         p.transform(Xs[te]).astype(np.float32))
        else:                                            # already below budget
            out[name] = (Xs[tr].copy(), Xs[te].copy())
        del Xs
        gc.collect()
    del route_raw
    gc.collect()
    out["t1+route"] = (np.hstack([out["t1"][0], out["route"][0]]),
                       np.hstack([out["t1"][1], out["route"][1]]))
    out["t2+route"] = (np.hstack([out["t2"][0], out["route"][0]]),
                       np.hstack([out["t2"][1], out["route"][1]]))
    return out, evr


# -------------------------------------------------------------------- scoring

def within_r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float("nan") if ss <= 0 else 1.0 - ((y - p) ** 2).sum() / ss


def within_auc(y, p):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def pooled(oof, metric):
    """oof: list of (group, branch_ids, y, pred).  Equal-weight mean over groups."""
    vals = [metric(y, p) for _, _, y, p in oof]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def boot_ci(oof_a, oof_b, metric, n=1000, seed=7):
    """Cluster (branch) bootstrap inside each held-out group; paired if oof_b given."""
    rng = np.random.default_rng(seed)
    idx_by_group = []
    for g, br, y, p in oof_a:
        ub, inv = np.unique(br, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(len(ub))]
        idx_by_group.append(members)
    da, db = [], []
    for _ in range(n):
        va, vb = [], []
        for gi, members in enumerate(idx_by_group):
            pick = rng.integers(0, len(members), len(members))
            sel = np.concatenate([members[j] for j in pick])
            _, _, y, p = oof_a[gi]
            r = metric(y[sel], p[sel])
            if np.isfinite(r):
                va.append(r)
            if oof_b is not None:
                _, _, y2, p2 = oof_b[gi]
                r2 = metric(y2[sel], p2[sel])
                if np.isfinite(r) and np.isfinite(r2):
                    vb.append(r2 - r)
        if va:
            da.append(np.mean(va))
        if vb:
            db.append(np.mean(vb))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if a else (np.nan, np.nan)
    return q(da), q(db)


# ----------------------------------------------------------------------- run

def run(corpus, shuffle=False, tag="main", tbins=False):
    d = dict(np.load(os.path.join(CACHE, f"corpus{corpus}.npz")))
    group, branch, kidx = d["group"], d["branch"], d["kidx"]
    groups = np.unique(group)
    perm = np.arange(len(branch))
    if shuffle:
        rng = np.random.default_rng(11)
        for b in np.unique(branch):
            ix = np.flatnonzero(branch == b)
            perm[ix] = rng.permutation(ix)

    proj_path = os.path.join(CACHE, f"proj{corpus}_K{K}.npz")
    if os.path.exists(proj_path):
        store = dict(np.load(proj_path, allow_pickle=True))
        evr = float(store["evr"])
    else:
        store, evr = {}, None
        for g in groups:
            te = np.flatnonzero(group == g)
            tr = np.flatnonzero(group != g)
            pr, e = fold_projections(d, tr, te)
            evr = e
            for bname, (a, b) in pr.items():
                if bname in ("t1+route", "t2+route"):
                    continue
                store[f"{g}|{bname}|tr"] = a
                store[f"{g}|{bname}|te"] = b
            store[f"{g}|tr_idx"] = tr
            store[f"{g}|te_idx"] = te
            print(f"  [{corpus}] fold g={g} projections done, route pre-PCA EVR={e:.4f}", flush=True)
            gc.collect()
        store["evr"] = np.array(evr)
        np.savez(proj_path, **store)
    print(f"[{corpus}] route pre-PCA EVR (256 comps of 2816) = {evr:.4f}", flush=True)
    d.pop("P", None); d.pop("T1", None); d.pop("T2", None); gc.collect()

    def blockmat(g, name, side):
        if "+" in name:
            a, b = name.split("+")
            return np.hstack([store[f"{g}|{a}|{side}"], store[f"{g}|{b}|{side}"]])
        return store[f"{g}|{name}|{side}"]

    rows = []
    oof_store = {}
    for target in TARGETS:
        Y = d[f"Y_{target}"]
        metric = within_auc if target in BINARY else within_r2
        mname = "auc" if target in BINARY else "r2"
        for mi, m in enumerate(HORIZONS):
            y_all = Y[:, mi]
            for bname in BLOCKS:
                oof = []
                for g in groups:
                    tr = store[f"{g}|tr_idx"]
                    te = store[f"{g}|te_idx"]
                    Xtr = blockmat(g, bname, "tr")
                    Xte = blockmat(g, bname, "te")
                    if shuffle:                       # features move, targets stay
                        ptr = np.searchsorted(tr, perm[tr])
                        pte = np.searchsorted(te, perm[te])
                        Xtr, Xte = Xtr[ptr], Xte[pte]
                    ytr, yte = y_all[tr], y_all[te]
                    vtr, vte = np.isfinite(ytr), np.isfinite(yte)
                    if vtr.sum() < 50 or vte.sum() < 20:
                        continue
                    if target in BINARY and len(np.unique(ytr[vtr])) < 2:
                        continue
                    mdl = RidgeCV(alphas=ALPHAS)
                    mdl.fit(Xtr[vtr], ytr[vtr])
                    pred = mdl.predict(Xte[vte])
                    oof.append((g, branch[te][vte], yte[vte], pred))
                if not oof:
                    continue
                val = pooled(oof, metric)
                n = int(sum(len(o[2]) for o in oof))
                rows.append((corpus, target, m, bname, "all", mname, val, n))
                oof_store[(target, m, bname)] = oof
                # within-branch (branch-mean removed) R2 for continuous targets
                if target not in BINARY:
                    wb = []
                    for g, br, y, p in oof:
                        ub, inv = np.unique(br, return_inverse=True)
                        ym = np.bincount(inv, y) / np.bincount(inv)
                        pm = np.bincount(inv, p) / np.bincount(inv)
                        wb.append(within_r2(y - ym[inv], p - pm[inv]))
                    wb = [v for v in wb if np.isfinite(v)]
                    if wb:
                        rows.append((corpus, target, m, bname, "all", "r2_within_branch",
                                     float(np.mean(wb)), n))
                if tbins:
                    for lo, hi, lbl in ((0, 12, "q0-11"), (12, 24, "q12-23"), (24, 10 ** 6, "q24+")):
                        sub = []
                        for g, br, y, p in oof:
                            kk = kidx[store[f"{g}|te_idx"]][np.isfinite(y_all[store[f"{g}|te_idx"]])]
                            sel = (kk >= lo) & (kk < hi)
                            if sel.sum() > 30:
                                sub.append((g, br[sel], y[sel], p[sel]))
                        if sub:
                            rows.append((corpus, target, m, bname, lbl, mname,
                                         pooled(sub, metric),
                                         int(sum(len(o[2]) for o in sub))))
            print(f"  [{corpus}/{tag}] {target} m={m} done", flush=True)

    # ------------------------------------------------ increments with CIs
    inc_rows = []
    for target in TARGETS:
        metric = within_auc if target in BINARY else within_r2
        for m in HORIZONS:
            for base, comb in (("t1", "t1+route"), ("t2", "t2+route")):
                a = oof_store.get((target, m, base))
                b = oof_store.get((target, m, comb))
                if a is None or b is None:
                    continue
                (lo_a, hi_a), (lo_d, hi_d) = boot_ci(a, b, metric)
                delta = pooled(b, metric) - pooled(a, metric)
                n = int(sum(len(o[2]) for o in a))
                inc_rows.append((corpus, target, m, f"inc_route_over_{base}", "all",
                                 "delta", delta, n))
                inc_rows.append((corpus, target, m, f"inc_route_over_{base}", "all",
                                 "delta_lo95", lo_d, n))
                inc_rows.append((corpus, target, m, f"inc_route_over_{base}", "all",
                                 "delta_hi95", hi_d, n))
                inc_rows.append((corpus, target, m, base, "all", "lo95", lo_a, n))
                inc_rows.append((corpus, target, m, base, "all", "hi95", hi_a, n))
    rows += inc_rows

    if shuffle:
        rows = [(c, t, m, f"SHUF_{b}", tb, mm, v, n) for c, t, m, b, tb, mm, v, n in rows]
    return rows


def append(rows):
    import csv
    for path in ("/tmp/moe_future.csv", os.path.join(ROOT, "analysis_future/moe_future.csv")):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


if __name__ == "__main__":
    corpus = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "main"
    rows = run(corpus, shuffle=(mode == "shuffle"), tag=mode, tbins=(mode == "main"))
    append(rows)
    print(f"appended {len(rows)} rows for corpus {corpus} / {mode}")
