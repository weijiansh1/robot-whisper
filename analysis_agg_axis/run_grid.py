"""Stage 2: the divergence x temporal-aggregation grid, raw and residualised, + 200-draw null.

Persists incrementally:
  /tmp/moe_agg_divtime.csv     corpus,divergence,temporal,window,t,auc,auc_resid,n_pos,n_neg
  /tmp/moe_agg_grid_{tag}.pkl  score matrices + ranks (for the permutation stage)
  /tmp/moe_agg_null_{tag}_{raw,resid}.npy   (n_perm, ncells, nt) detection-AUC null cube
"""
import os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

CSV = "/tmp/moe_agg_divtime.csv"
TS = (20, 25, 30)
BASE = ("hell|prev", "mean", 8)
N_PERM = 200


def load(tag):
    with open(f"/tmp/moe_agg_series_{tag}.pkl", "rb") as fh:
        return pickle.load(fh)


def thresholds(D):
    """Label-free global quantiles per divergence, over d[b, 1:31] (region all branches share)."""
    out = {}
    for k, d in D.items():
        v = d[:, 1:31]
        v = v[np.isfinite(v)]
        out[k] = tuple(float(np.quantile(v, q)) for q in (0.10, 0.25, 0.75, 0.90))
    return out


def score_matrix(o, thr, t):
    """(ncells, B) score matrix at query index t, plus the cell key list."""
    D, T = o["D"], o["T"]
    alive = T > t
    assert alive.all(), "risk set is not everybody; t too large"
    keys, rows = [], []
    for dv in lib.DIVERGENCES:
        d = D[dv]
        for (tm, W) in lib.cells():
            keys.append((dv, tm, W))
            rows.append(lib.aggregate(d, t, tm, W, thr=thr[dv]))
    S = np.asarray(rows, np.float64)
    assert np.isfinite(S).all(), "non-finite scores"
    return S, keys


def group_rank_pack(groups, y):
    """Contributing branches (groups with both classes) + per-group slices."""
    keep = np.zeros(len(y), bool)
    slices = []
    off = 0
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if y[m].sum() == 0 or (~y[m]).sum() == 0:
            continue
        keep[m] = True
        slices.append((off, off + len(m), int(y[m].sum()), int((~y[m]).sum())))
        off += len(m)
    idx = np.concatenate([np.where(groups == g)[0] for g in np.unique(groups)
                          if keep[np.where(groups == g)[0]].all()])
    C = sum(np1 * (np1 + 1) / 2.0 for _, _, np1, _ in slices)
    Pairs = sum(np1 * nn for _, _, np1, nn in slices)
    return idx, slices, C, Pairs


def ranks_of(S, idx, slices):
    """Within-group average ranks of each row of S, restricted to contributing branches."""
    R = np.empty((S.shape[0], len(idx)))
    Sk = S[:, idx]
    for a, b, _, _ in slices:
        blk = Sk[:, a:b]
        for i in range(blk.shape[0]):
            R[i, a:b] = lib._avg_rank(blk[i])
    return R


def main():
    if os.path.exists(CSV):
        os.remove(CSV)
    store = {}
    for tag in ("A", "B"):
        t0 = time.time()
        o = load(tag)
        thr = thresholds(o["D"])
        groups, y = o["groups"], o["y"]
        idx, slices, Cconst, Pairs = group_rank_pack(groups, y)
        yk = y[idx]
        n_pos = int(yk.sum()); n_neg = int((~yk).sum())
        per_t = {}
        for t in TS:
            S, keys = score_matrix(o, thr, t)
            bi = keys.index(BASE)
            base = S[bi]
            # residualise every cell on the baseline, within group (label-free)
            Sres = np.empty_like(S)
            for i in range(S.shape[0]):
                Sres[i] = lib.rank_residualise(S[i], base, groups)
            R = ranks_of(S, idx, slices)
            Rr = ranks_of(Sres, idx, slices)
            auc = (R @ yk.astype(float) - Cconst) / Pairs
            aucr = (Rr @ yk.astype(float) - Cconst) / Pairs
            per_t[t] = dict(R=R, Rr=Rr, auc=auc, auc_resid=aucr)
            rows = pd.DataFrame({
                "corpus": tag,
                "divergence": [k[0] for k in keys],
                "temporal": [k[1] for k in keys],
                "window": [k[2] for k in keys],
                "t": t,
                "auc": np.round(auc, 6),
                "auc_resid": np.round(aucr, 6),
                "n_pos": n_pos, "n_neg": n_neg})
            rows.to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
            print(f"[{tag}] t={t} cells={len(keys)} baseline raw={auc[bi]:.4f} "
                  f"resid={aucr[bi]:.4f} -> {CSV}", flush=True)
        store[tag] = dict(keys=keys, per_t=per_t, yk=yk, idx=idx, slices=slices,
                          Cconst=Cconst, Pairs=Pairs, n_pos=n_pos, n_neg=n_neg,
                          groups=groups, y=y)
        with open(f"/tmp/moe_agg_grid_{tag}.pkl", "wb") as fh:
            pickle.dump({k: v for k, v in store[tag].items()}, fh, 4)
        print(f"[{tag}] grid done {time.time()-t0:.1f}s  n_pos={n_pos} n_neg={n_neg} "
              f"pairs={Pairs}", flush=True)

    # ---------------- permutation null: branch-level, within group, reused across t
    for tag in ("A", "B"):
        st = store[tag]
        rng = np.random.default_rng(20260829)
        Y = np.empty((len(st["yk"]), N_PERM))
        for p in range(N_PERM):
            v = st["yk"].copy()
            for a, b, _, _ in st["slices"]:
                v[a:b] = rng.permutation(v[a:b])
            Y[:, p] = v.astype(float)
        for kind, key in (("raw", "R"), ("resid", "Rr")):
            null = np.empty((N_PERM, st["per_t"][TS[0]][key].shape[0], len(TS)), np.float32)
            for j, t in enumerate(TS):
                a = (st["per_t"][t][key] @ Y - st["Cconst"]) / st["Pairs"]   # (ncells, nperm)
                null[:, :, j] = a.T                                          # DIRECTIONAL
            np.save(f"/tmp/moe_agg_null_{tag}_{kind}.npy", null)
            print(f"[{tag}] null {kind} {null.shape} (directional) -> "
                  f"/tmp/moe_agg_null_{tag}_{kind}.npy", flush=True)


if __name__ == "__main__":
    main()
