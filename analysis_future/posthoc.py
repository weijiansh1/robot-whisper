"""Post-hoc metrics from the stored OOF prediction tables.

Within-group R2 is the metric the protocol asks for, but on the object targets the
control's OOF R2 is negative in some held-out groups, which makes R2 ratios and their
cluster-bootstrap CIs numerically unstable.  Spearman rho within group is bounded and
is reported alongside, together with per-group sign consistency, which is what this
project's own guidance requires (never pool across initial states).
"""
import os
import sys
import numpy as np

CACHE = "/tmp/moe_future_cache"
ROOT = "/home/jovyan/work/himoe-vla"
BINARY = {"undo", "undo_nc", "obj_nc", "obj_ncfar", "obj_gt_arm",
          "which_moves", "which_moves_late"}
BLOCKS = ("clock_causal", "clock", "clock_rich", "route", "t1", "t2",
          "t1+route", "t2+route", "t1+clock_rich")


from scipy.stats import rankdata as rank        # C-speed average-tie ranking


def spearman(y, p):
    if len(y) < 5:
        return np.nan
    a, b = rank(y), rank(p)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return np.nan if den <= 0 else float((a * b).sum() / den)


def auc(y, p):
    o = np.argsort(p, kind="mergesort")
    ys = y[o].astype(np.float64)
    Wp, Wn = ys.sum(), len(ys) - ys.sum()
    if Wp <= 0 or Wn <= 0:
        return np.nan
    wn = 1.0 - ys
    return float((ys * (np.cumsum(wn) - wn)).sum() / (Wp * Wn))


def r2(y, p):
    ss = ((y - y.mean()) ** 2).sum()
    return np.nan if ss <= 0 else 1.0 - ((y - p) ** 2).sum() / ss


def per_group(y, p, g, fn):
    out = {}
    for gg in np.unique(g):
        s = g == gg
        out[int(gg)] = fn(y[s], p[s])
    return out


def pooled(y, p, g, fn):
    v = [x for x in per_group(y, p, g, fn).values() if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def boot_delta(y, pa, pb, g, br, fn, n=500, seed=3):
    """Paired branch-cluster bootstrap of pooled(fn(b)) - pooled(fn(a))."""
    rng = np.random.default_rng(seed)
    idx = []
    for gg in np.unique(g):
        s = np.flatnonzero(g == gg)
        ub, inv = np.unique(br[s], return_inverse=True)
        idx.append([s[inv == i] for i in range(len(ub))])
    da, dd = [], []
    for _ in range(n):
        va, vd = [], []
        for mem in idx:
            sel = np.concatenate([mem[j] for j in rng.integers(0, len(mem), len(mem))])
            fa, fb = fn(y[sel], pa[sel]), fn(y[sel], pb[sel])
            if np.isfinite(fa):
                va.append(fa)
            if np.isfinite(fa) and np.isfinite(fb):
                vd.append(fb - fa)
        if va:
            da.append(np.mean(va))
        if vd:
            dd.append(np.mean(vd))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) \
        if len(a) else (np.nan, np.nan)
    return q(da), q(dd)


def main(corpus, mode="main"):
    z = np.load(os.path.join(CACHE, f"oof_{corpus}_{mode}.npz"))
    keys = sorted({k.rsplit("|", 1)[0] for k in z.files if k.endswith("|__y")})
    keys = [k.rsplit("|", 1)[0] for k in z.files if k.endswith("|__y")]
    rows = []
    for key in sorted(set(keys), key=lambda s: (s.split("|")[0], int(s.split("|")[1]))):
        t, m = key.split("|")[0], int(key.split("|")[1])
        y = z[f"{key}|__y"]
        g = z[f"{key}|__g"]
        br = z[f"{key}|__b"]
        fn = auc if t in BINARY else spearman
        name = "auc" if t in BINARY else "spearman"
        got = {}
        for b in BLOCKS:
            k = f"{key}|{b}"
            if k not in z.files:
                continue
            p = z[k]
            got[b] = p
            rows.append((corpus, t, m, b, "all", name, pooled(y, p, g, fn), len(y)))
            if t not in BINARY:
                rows.append((corpus, t, m, b, "all", "r2_pg",
                             pooled(y, p, g, r2), len(y)))
        for base in ("t1", "t2"):
            comb = f"{base}+route"
            if base not in got or comb not in got:
                continue
            pg_a = per_group(y, got[base], g, fn)
            pg_b = per_group(y, got[comb], g, fn)
            dsign = [pg_b[k] - pg_a[k] for k in pg_a
                     if np.isfinite(pg_a[k]) and np.isfinite(pg_b[k])]
            (lo_a, hi_a), (lo_d, hi_d) = boot_delta(y, got[base], got[comb], g, br, fn)
            d = pooled(y, got[comb], g, fn) - pooled(y, got[base], g, fn)
            blk = f"inc_route_over_{base}_{name}"
            for nm, v in (("delta", d), ("delta_lo95", lo_d), ("delta_hi95", hi_d),
                          ("n_groups_positive", float(sum(x > 0 for x in dsign))),
                          ("n_groups", float(len(dsign)))):
                rows.append((corpus, t, m, blk, "all", nm, v, len(y)))
    return rows


if __name__ == "__main__":
    import csv
    corpus, mode = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "main")
    rows = main(corpus, mode)
    if mode == "shuffle":
        rows = [(c, t, m, f"SHUF_{b}", tb, mm, v, n) for c, t, m, b, tb, mm, v, n in rows]
    for path in ("/tmp/moe_future.csv", os.path.join(ROOT, "analysis_future/moe_future.csv")):
        with open(path, "a", newline="") as f:
            csv.writer(f).writerows(rows)
    print(f"appended {len(rows)} post-hoc rows :: {corpus}/{mode}")
