"""Re-run the WHOLE grid residualised on the richer baselines B2 / B3, then redo the
family-max test and the joint cross-corpus clearing test. This closes out the only
statistic that survived residualisation on the single stated baseline."""
import os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
import run_grid as RG

TS = (20, 25, 30)
N_PERM = 200
SETS = {
    "B1_stated": [("hell|prev", "mean", 8)],
    "B2_all_window_means": [("hell|prev", "mean", W) for W in (3, 5, 8, 12, 20)],
    "B3_means+moments": ([("hell|prev", "mean", W) for W in (3, 5, 8, 12, 20)] +
                         [("hell|prev", "std", 8), ("hell|prev", "max", 8),
                          ("hell|prev", "min", 8), ("hell|prev", "median", 8),
                          ("hell|prev", "ewmall0.15", 0)]),
}


def det(a):
    return np.maximum(a, 1 - a)


def multi_resid_all(S, Bmat, groups):
    """S (ncells, B) -> residual of rank(S) on ranks of Bmat columns, within group."""
    out = np.full(S.shape, np.nan)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        if len(m) < Bmat.shape[1] + 3:
            continue
        X = np.column_stack([lib._avg_rank(Bmat[m, k]) for k in range(Bmat.shape[1])])
        X = X - X.mean(0)
        Q, _ = np.linalg.qr(X)
        Y = np.empty((S.shape[0], len(m)))
        for i in range(S.shape[0]):
            Y[i] = lib._avg_rank(S[i, m])
        Y -= Y.mean(1, keepdims=True)
        r = Y - (Y @ Q) @ Q.T
        # cells lying in the span of the baseline set leave only float noise; ranking that
        # noise would resurrect the baseline, so snap it to zero.
        r[np.abs(r) < 1e-8 * len(m)] = 0.0
        out[:, m] = r
    return out


def main():
    res = {}
    for tag in ("A", "B"):
        t0 = time.time()
        o = RG.load(tag)
        thr = RG.thresholds(o["D"])
        groups, y = o["groups"], o["y"]
        idx, slices, Cc, Pairs = RG.group_rank_pack(groups, y)
        yk = y[idx]
        rng = np.random.default_rng(20260829)
        Y = np.empty((len(yk), N_PERM))
        for p in range(N_PERM):
            v = yk.copy()
            for a, b, _, _ in slices:
                v[a:b] = rng.permutation(v[a:b])
            Y[:, p] = v.astype(float)
        for setname, bset in SETS.items():
            obs = np.empty((0, 0)); nulls = None
            for j, t in enumerate(TS):
                S, keys = RG.score_matrix(o, thr, t)
                Bm = np.column_stack([lib.aggregate(o["D"][b[0]], t, b[1], b[2], thr=thr[b[0]])
                                      for b in bset])
                Sr = multi_resid_all(S, Bm, groups)
                R = RG.ranks_of(Sr, idx, slices)
                a = (R @ yk.astype(float) - Cc) / Pairs
                if nulls is None:
                    obs = np.empty((len(keys), len(TS)))
                    nulls = np.empty((N_PERM, len(keys), len(TS)), np.float32)
                obs[:, j] = a
                nulls[:, :, j] = ((R @ Y - Cc) / Pairs).T
            res[(tag, setname)] = (obs, nulls, keys)
            np.save(f"/tmp/moe_agg_b3_obs_{tag}_{setname}.npy", obs)
            np.save(f"/tmp/moe_agg_b3_null_{tag}_{setname}.npy", nulls)
            print(f"[{tag}] {setname} done {time.time()-t0:.1f}s  max det = "
                  f"{det(obs).max():.4f}", flush=True)

    keys = res[("A", "B1_stated")][2]
    cellnames = np.array([f"{a}|{b}|W{c}" for a, b, c in keys])
    lines = ["\n## 20. The whole grid re-residualised on the richer baselines\n",
             "Sections 5/9 residualised on the single stated baseline. If the surviving cells are "
             "really only reading the *window width* of the same shared factor, they must collapse "
             "once every window mean is in the baseline. Full 1794-cell grid, same 200-draw "
             "within-group permutation null.\n",
             "### 20a. Family-max test\n"]
    rows = []
    for tag in ("A", "B"):
        for sn in SETS:
            obs, nulls, _ = res[(tag, sn)]
            o = det(obs).max()
            nm = det(nulls).max(axis=(1, 2))
            p = (1 + (nm >= o).sum()) / (1 + N_PERM)
            ij = np.unravel_index(det(obs).argmax(), obs.shape)
            lines.append(f"* corpus **{tag}**, baseline set **{sn}**: argmax "
                         f"`{cellnames[ij[0]]}` at t={TS[ij[1]]}, observed max det-AUC "
                         f"**{o:.4f}**, null p95 {np.percentile(nm,95):.4f}, null max "
                         f"{nm.max():.4f} -> **family-wise p = {p:.4f}**")
            rows.append(dict(corpus=tag, baseline_set=sn, test="family_max",
                             argmax=cellnames[ij[0]], t=TS[ij[1]], obs=o, p=p))
    lines.append("\n### 20b. Joint cross-corpus clearing count (the test that survived in "
                 "section 9)\n")
    for sn in SETS:
        oA, nA, _ = res[("A", sn)]
        oB, nB, _ = res[("B", sn)]
        thA = np.percentile(det(nA), 95, axis=0)
        thB = np.percentile(det(nB), 95, axis=0)
        cl = (det(oA) > thA) & (det(oB) > thB) & (np.sign(oA - .5) == np.sign(oB - .5))
        ct = int(cl.sum())
        nc = ((det(nA) > thA[None]) & (det(nB) > thB[None]) &
              (np.sign(nA - .5) == np.sign(nB - .5))).sum(axis=(1, 2))
        p = (1 + (nc >= ct).sum()) / (1 + N_PERM)
        top = ""
        if ct:
            ii, jj = np.where(cl)
            mn = np.minimum(det(oA)[ii, jj], det(oB)[ii, jj])
            k = np.argsort(-mn)[:6]
            top = "  best: " + "; ".join(
                f"{cellnames[ii[q]]}@t{TS[jj[q]]} A={det(oA)[ii[q],jj[q]]:.3f} "
                f"B={det(oB)[ii[q],jj[q]]:.3f}" for q in k)
        lines.append(f"* baseline set **{sn}**: observed joint count **{ct}** / 5382; null mean "
                     f"{nc.mean():.1f}, p95 {np.percentile(nc,95):.0f}, max of 200 {nc.max()} "
                     f"-> **p = {p:.4f}**\n{top}")
        rows.append(dict(corpus="both", baseline_set=sn, test="joint_count", argmax="", t=-1,
                         obs=ct, p=p))
    pd.DataFrame(rows).to_csv("/tmp/moe_agg_b3_tests.csv", index=False)
    with open("/tmp/moe_agg_divtime.md", "a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("appended section 20")


if __name__ == "__main__":
    main()
