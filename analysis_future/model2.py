"""Grouped leave-one-group-out evaluation of routing vs. control tiers (fast path).

Identical model class and capacity for every feature block:
    centre/standardise -> PCA to K whitened components -> ridge with exact-LOO alpha.
The ridge is solved from one economy SVD per (fold, block, validity mask) and reused
across every target, which is what makes the block x target x horizon grid affordable
on 16 folds.

Metrics are computed WITHIN each held-out group and pooled with equal weight, because
the four rolling-star workers have success rates 0% / 6% / 26% / 91%.

Tier 1 (proprio + own action chunk + eef/landmark geometry) is the live comparison.
Tier 2 (privileged full 47-dim sim_state) is computed for the record only.
"""
import csv
import gc
import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

CACHE = "/tmp/moe_future_cache"
ROOT = "/home/jovyan/work/himoe-vla"
K = int(os.environ.get("PCA_K", 128))
PRE_K = 256
ALPHAS = np.logspace(-4, 6, 21)
HORIZONS = (1, 2, 4, 8)

VISION = ("obj_disp_p1", "obj_disp_p2", "which_moves", "which_moves_late",
          "obj_nc", "obj_ncfar", "obj_free_disp", "obj_gt_arm", "undo_nc")
TARGETS = ("eef_disp", "eef_path", "obj_disp", "dgoal", "undo", "grip_flips") + VISION
BINARY = {"undo", "undo_nc", "obj_nc", "obj_ncfar", "obj_gt_arm",
          "which_moves", "which_moves_late"}
# targets that share a validity mask -> share one SVD
MASK_GROUPS = {
    "full": ("eef_disp", "eef_path", "obj_disp", "dgoal", "undo", "grip_flips",
             "obj_disp_p1", "obj_disp_p2", "obj_nc", "obj_ncfar",
             "obj_free_disp", "obj_gt_arm", "undo_nc"),
    "which_moves": ("which_moves",),
    "which_moves_late": ("which_moves_late",),
}
MASK_REF = {"full": "eef_disp", "which_moves": "which_moves",
            "which_moves_late": "which_moves_late"}
BASE_BLOCKS = ("clock", "clock_causal", "clock_rich", "route", "t1", "t2")
BLOCKS = BASE_BLOCKS + ("t1+route", "t2+route", "t1+clock_rich")


# ------------------------------------------------------------ ridge with LOO

class LooRidge:
    """One economy SVD; exact leave-one-out alpha selection; reused across targets."""

    def __init__(self, Xtr):
        # Gram + eigh is the same decomposition as the economy SVD but ~35x faster here.
        self.mu = Xtr.mean(0, keepdims=True)
        Xc = np.asarray(Xtr - self.mu, dtype=np.float64)
        w, V = np.linalg.eigh(Xc.T @ Xc)
        keep = w > max(w.max(), 0.0) * 1e-10
        self.s2, self.Vt = w[keep], V[:, keep].T
        self.s = np.sqrt(self.s2)
        self.U = (Xc @ V[:, keep]) / self.s
        # all alphas at once: two GEMMs instead of 2 * len(ALPHAS) mat-vecs
        self.D = self.s2 / (self.s2[None, :] + ALPHAS[:, None])          # (A, r)
        self.Hc = np.maximum(1.0 - (self.U ** 2) @ self.D.T, 1e-6)       # (n, A)

    def fit_predict(self, y, Xte):
        ym = y.mean()
        yc = np.asarray(y, np.float64) - ym
        Uty = self.U.T @ yc
        R = (yc[:, None] - self.U @ (self.D * Uty[None, :]).T) / self.Hc
        ia = int(np.argmin((R ** 2).mean(axis=0)))
        coef = self.Vt.T @ ((self.s / (self.s2 + ALPHAS[ia])) * Uty)
        return (Xte - self.mu) @ coef + ym


# --------------------------------------------------------------- projections

def clock_rich(CL):
    """A deliberately strong clock sentinel: smooth basis in t, T, t/T."""
    t, T, r, rem, gt = (CL[:, i] for i in range(5))
    cols = [t, T, r, rem, gt, t ** 2, r ** 2, r ** 3, np.log1p(t), np.log1p(rem), t * T, r * T]
    for c, w in ((r, 0.10), (t / 55.0, 0.10), (rem / 55.0, 0.10)):
        for mu in np.linspace(0, 1, 12):
            cols.append(np.exp(-0.5 * ((c - mu) / w) ** 2))
    return np.stack(cols, axis=1).astype(np.float32)


def fold_projections(d, tr, te, seed=0):
    out = {}
    P, kidx = d["P"], d["kidx"]
    prev = np.arange(len(P)) - 1
    prev[kidx == 0] = np.flatnonzero(kidx == 0)

    mu = P[tr].mean(0, keepdims=True)
    pre = PCA(n_components=PRE_K, svd_solver="randomized", random_state=seed)
    pre.fit(P[tr] - mu)
    z = pre.transform(P - mu).astype(np.float32)
    evr = float(pre.explained_variance_ratio_.sum())
    route_raw = np.concatenate([z, z - z[prev]], axis=1)
    del z
    gc.collect()

    for name, X in (("route", route_raw), ("t1", d["T1"]), ("t2", d["T2"]),
                    ("clock", d["CLOCK"]), ("clock_causal", d["CLOCKC"]),
                    ("clock_rich", clock_rich(d["CLOCK"]))):
        X = np.asarray(X, np.float32)
        m = X[tr].mean(0, keepdims=True)
        s = X[tr].std(0, keepdims=True)
        s[s < 1e-8] = 1.0
        Xs = (X - m) / s
        k = min(K, Xs.shape[1], len(tr) - 1)
        if k < Xs.shape[1]:
            p = PCA(n_components=k, whiten=True, svd_solver="randomized", random_state=seed)
            p.fit(Xs[tr])
            out[name] = (p.transform(Xs[tr]).astype(np.float32),
                         p.transform(Xs[te]).astype(np.float32))
        else:
            out[name] = (Xs[tr].copy(), Xs[te].copy())
        del Xs
        gc.collect()
    return out, evr


# -------------------------------------------------------------------- scoring

def within_r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return float("nan") if ss <= 0 else 1.0 - ((y - p) ** 2).sum() / ss


def within_auc(y, p):
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))


def pooled(oof, metric):
    v = [metric(y, p) for _, _, y, p in oof]
    v = [x for x in v if np.isfinite(x)]
    return float(np.mean(v)) if v else float("nan")


def _prep(br, y, p, is_auc):
    """Per-branch sufficient statistics so a bootstrap replicate is O(#branches)."""
    ub, inv = np.unique(br, return_inverse=True)
    if is_auc:
        o = np.argsort(p, kind="mergesort")
        return ("auc", y[o].astype(np.float64), inv[o], len(ub))
    return ("r2", np.bincount(inv).astype(np.float64), np.bincount(inv, y),
            np.bincount(inv, y * y), np.bincount(inv, (y - p) ** 2), len(ub))


def _eval(c, st):
    if st[0] == "auc":
        _, ys, bs, _ = st
        w = c[bs]
        wp = w * ys
        wn = w - wp
        Wp, Wn = wp.sum(), wn.sum()
        if Wp <= 0 or Wn <= 0:
            return np.nan
        return float((wp * (np.cumsum(wn) - wn)).sum() / (Wp * Wn))
    _, nb, S1, S2, E, _ = st
    n = c @ nb
    s1 = c @ S1
    sst = c @ S2 - s1 * s1 / n
    return np.nan if sst <= 0 else 1.0 - (c @ E) / sst


def boot(oof_a, oof_b, metric, n=int(os.environ.get("BOOT_N", 1000)), seed=7):
    """Branch-cluster bootstrap inside each held-out group; paired with oof_b.

    Identical statistic to resampling rows directly (verified exactly), but the
    metric is rebuilt from per-branch sufficient statistics.
    """
    is_auc = metric is within_auc
    rng = np.random.default_rng(seed)
    sa = [_prep(br, y, p, is_auc) for _, br, y, p in oof_a]
    sb = [_prep(oof_b[i][1], oof_b[i][2], oof_b[i][3], is_auc)
          for i in range(len(oof_a))] if oof_b is not None else None
    nbs = [s[-1] for s in sa]
    da, db = [], []
    for _ in range(n):
        va, vb = [], []
        for gi, nb in enumerate(nbs):
            c = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(np.float64)
            r = _eval(c, sa[gi])
            if np.isfinite(r):
                va.append(r)
            if sb is not None:
                r2 = _eval(c, sb[gi])
                if np.isfinite(r) and np.isfinite(r2):
                    vb.append(r2 - r)
        if va:
            da.append(np.mean(va))
        if vb:
            db.append(np.mean(vb))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
        if len(a) else (np.nan, np.nan)
    return q(da), q(db)


# ------------------------------------------------------------------------ run

def get_store(corpus, d):
    path = os.path.join(CACHE, f"proj2{corpus}_K{K}.npz")
    if os.path.exists(path):
        s = dict(np.load(path))
        return s, float(s["evr"])
    store, evr = {}, None
    for g in np.unique(d["group"]):
        te = np.flatnonzero(d["group"] == g)
        tr = np.flatnonzero(d["group"] != g)
        pr, evr = fold_projections(d, tr, te)
        for b, (a, c) in pr.items():
            store[f"{g}|{b}|tr"] = a
            store[f"{g}|{b}|te"] = c
        store[f"{g}|tr_idx"] = tr
        store[f"{g}|te_idx"] = te
        print(f"  [{corpus}] fold g={g} EVR={evr:.4f}", flush=True)
        gc.collect()
    store["evr"] = np.array(evr)
    np.savez(path, **store)
    return store, evr


def run(corpus, shuffle=False, tbins=False):
    d = dict(np.load(os.path.join(CACHE, f"corpus{corpus}.npz")))
    group, branch, kidx = d["group"], d["branch"], d["kidx"]
    groups = np.unique(group)
    store, evr = get_store(corpus, d)
    for k in ("P", "T1", "T2"):
        d.pop(k, None)
    gc.collect()
    print(f"[{corpus}] route pre-PCA EVR (256 of 2816) = {evr:.4f}", flush=True)

    perm = np.arange(len(branch))
    if shuffle:
        rng = np.random.default_rng(11)
        for b in np.unique(branch):
            ix = np.flatnonzero(branch == b)
            perm[ix] = rng.permutation(ix)

    def mat(g, name, side):
        if "+" in name:
            a, b = name.split("+")
            return np.hstack([store[f"{g}|{a}|{side}"], store[f"{g}|{b}|{side}"]])
        return store[f"{g}|{name}|{side}"]

    rows, oof_store, dump = [], {}, {}
    for mi, m in enumerate(HORIZONS):
        for mg, tnames in MASK_GROUPS.items():
            ref = d[f"Y_{MASK_REF[mg]}"][:, mi]
            for bname in BLOCKS:
                acc = {t: [] for t in tnames}
                for g in groups:
                    tr, te = store[f"{g}|tr_idx"], store[f"{g}|te_idx"]
                    Xtr, Xte = mat(g, bname, "tr"), mat(g, bname, "te")
                    if shuffle:
                        Xtr = Xtr[np.searchsorted(tr, perm[tr])]
                        Xte = Xte[np.searchsorted(te, perm[te])]
                    vtr, vte = np.isfinite(ref[tr]), np.isfinite(ref[te])
                    if vtr.sum() < 50 or vte.sum() < 20:
                        continue
                    R = LooRidge(Xtr[vtr])
                    Xv = Xte[vte]
                    for t in tnames:
                        y = d[f"Y_{t}"][:, mi]
                        ytr, yte = y[tr][vtr], y[te][vte]
                        if not np.isfinite(ytr).all() or not np.isfinite(yte).all():
                            continue
                        if t in BINARY and len(np.unique(ytr)) < 2:
                            continue
                        acc[t].append((g, branch[te][vte], yte, R.fit_predict(ytr, Xv)))
                    del R, Xtr, Xte, Xv
                    gc.collect()
                for t in tnames:
                    oof = acc[t]
                    if not oof:
                        continue
                    metric = within_auc if t in BINARY else within_r2
                    mn = "auc" if t in BINARY else "r2"
                    n = int(sum(len(o[2]) for o in oof))
                    rows.append((corpus, t, m, bname, "all", mn, pooled(oof, metric), n))
                    oof_store[(t, m, bname)] = oof
                    key = f"{t}|{m}"
                    if f"{key}|__y" not in dump:
                        dump[f"{key}|__y"] = np.concatenate([o[2] for o in oof])
                        dump[f"{key}|__g"] = np.concatenate(
                            [np.full(len(o[2]), o[0]) for o in oof])
                        dump[f"{key}|__b"] = np.concatenate([o[1] for o in oof])
                    dump[f"{key}|{bname}"] = np.concatenate(
                        [o[3] for o in oof]).astype(np.float32)
                    if t not in BINARY:
                        wb = []
                        for g, br, y, p in oof:
                            ub, inv = np.unique(br, return_inverse=True)
                            c = np.bincount(inv)
                            wb.append(within_r2(y - (np.bincount(inv, y) / c)[inv],
                                                p - (np.bincount(inv, p) / c)[inv]))
                        wb = [x for x in wb if np.isfinite(x)]
                        if wb:
                            rows.append((corpus, t, m, bname, "all", "r2_within_branch",
                                         float(np.mean(wb)), n))
                    if tbins:
                        for lo, hi, lbl in ((0, 12, "q0-11"), (12, 24, "q12-23"),
                                            (24, 10 ** 6, "q24+")):
                            sub = []
                            for g, br, y, p in oof:
                                te = store[f"{g}|te_idx"]
                                kk = kidx[te][np.isfinite(ref[te])]
                                sel = (kk >= lo) & (kk < hi)
                                if sel.sum() > 30:
                                    sub.append((g, br[sel], y[sel], p[sel]))
                            if sub:
                                rows.append((corpus, t, m, bname, lbl, mn,
                                             pooled(sub, metric),
                                             int(sum(len(o[2]) for o in sub))))
                print(f"  [{corpus}] m={m} {mg} {bname} done", flush=True)

    for t in TARGETS:
        metric = within_auc if t in BINARY else within_r2
        for m in HORIZONS:
            for base in ("t1", "t2"):
                a = oof_store.get((t, m, base))
                b = oof_store.get((t, m, f"{base}+route"))
                if a is None or b is None:
                    continue
                (lo_a, hi_a), (lo_d, hi_d) = boot(a, b, metric)
                n = int(sum(len(o[2]) for o in a))
                for nm, v in (("delta", pooled(b, metric) - pooled(a, metric)),
                              ("delta_lo95", lo_d), ("delta_hi95", hi_d)):
                    rows.append((corpus, t, m, f"inc_route_over_{base}", "all", nm, v, n))
                rows.append((corpus, t, m, base, "all", "lo95", lo_a, n))
                rows.append((corpus, t, m, base, "all", "hi95", hi_a, n))
            a = oof_store.get((t, m, "t1"))
            c = oof_store.get((t, m, "t1+clock_rich"))
            if a is not None and c is not None:
                _, (lo_c, hi_c) = boot(a, c, metric)
                for nm, v in (("delta", pooled(c, metric) - pooled(a, metric)),
                              ("delta_lo95", lo_c), ("delta_hi95", hi_c)):
                    rows.append((corpus, t, m, "inc_clock_over_t1", "all", nm, v, n))
            r = oof_store.get((t, m, "route"))
            if r is not None:
                (lo_r, hi_r), _ = boot(r, None, metric)
                nr = int(sum(len(o[2]) for o in r))
                rows.append((corpus, t, m, "route", "all", "lo95", lo_r, nr))
                rows.append((corpus, t, m, "route", "all", "hi95", hi_r, nr))
        print(f"  [{corpus}] bootstrap {t} done", flush=True)

    tagn = "shuffle" if shuffle else "main"
    np.savez(os.path.join(CACHE, f"oof_{corpus}_{tagn}.npz"), **dump)
    np.savez(os.path.join(ROOT, "analysis_future/cache",
                          f"oof_{corpus}_{tagn}.npz"), **dump)
    if shuffle:
        rows = [(c, t, m, f"SHUF_{b}", tb, mm, v, n) for c, t, m, b, tb, mm, v, n in rows]
    return rows


def append(rows):
    for path in ("/tmp/moe_future.csv", os.path.join(ROOT, "analysis_future/moe_future.csv")):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)


if __name__ == "__main__":
    corpus = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "main"
    rows = run(corpus, shuffle=(mode == "shuffle"), tbins=(mode == "main"))
    append(rows)
    print(f"appended {len(rows)} rows :: {corpus} / {mode}")
