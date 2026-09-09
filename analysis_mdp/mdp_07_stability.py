"""Seed stability of the single best cell found anywhere in the search.

Cell = semi-Markov (routing state x dwell bucket), K=64, score = cross-fitted empirical
per-state failure rate averaged over the 8-chunk window, evaluated at t=30. It is the only
cell in ~570 that beat the scalar baseline's raw det AUC in BOTH corpora. If it is a real
effect it must survive re-drawing the folds and re-fitting k-means; if it is selection
noise on a 352/512-branch corpus it will not.

order1 K=64 (no dwell augmentation) is carried alongside as the contrast.
"""
import numpy as np
import pandas as pd

import mdp_lib as L
from mdp_02_detect import FoldModel, W
from mdp_05_order import augment

SEEDS = (20260829, 1, 7, 12345, 999)
NDRAW = 200


def main(report_only=False):
    if report_only:
        df = pd.read_csv(f"{L.OUT}/stability_raw.csv.gz")
        return report(df)
    rows = []
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        y = meta["success"].values.astype(bool)
        g = meta["group"].values
        T = meta["T"].values
        B = len(meta)
        d = L.agglib.divergence_series(P, valid, "hell|prev")
        base = {t: L.agglib.aggregate(d, t, "mean", W) for t in L.TS}
        F = L.features(P, valid, "cell1280")
        for seed in SEEDS:
            fd = L.folds(meta, seed=seed)
            S = [L.fit_states(F, valid, np.where(fd != f)[0], 64, seed=seed)[0]
                 for f in range(L.NFOLD)]
            for mode in ("order1", "semi"):
                models = []
                for f in range(L.NFOLD):
                    A, n = augment(S[f], valid, mode)
                    models.append(FoldModel(A.astype(np.int16), valid, np.where(fd != f)[0],
                                            np.where(fd == f)[0], n, L.TS))
                rng = np.random.default_rng(555)
                for i in range(NDRAW + 1):
                    if i == 0:
                        yy = y
                    else:
                        yy = y.copy()
                        for gg in np.unique(g):
                            idx = np.where(g == gg)[0]
                            yy[idx] = y[idx][rng.permutation(len(idx))]
                    for m in models:
                        m.fit(~yy)
                    for t in L.TS:
                        sc = np.full(B, np.nan)
                        for m in models:
                            te, s = m.scores(t)
                            sc[te] = s["emp_pool_w8"]
                        alive = T > t
                        _, dd, _ = L.auc_pair(sc[alive], yy[alive], g[alive])
                        r = L.agglib.rank_residualise(sc, base[t], g)
                        ar, dr, _ = L.auc_pair(r[alive], yy[alive], g[alive])
                        rows.append(dict(corpus=tag, seed=seed, mode=mode, t=t, draw=i,
                                         det_auc=dd, res_det_auc=dr, res_auc_succ=ar,
                                         base=L.auc_pair(base[t][alive], y[alive], g[alive])[1]))
                print(tag, seed, mode, "done", flush=True)
        del F
    df = pd.DataFrame(rows)
    df.to_csv(f"{L.OUT}/stability_raw.csv.gz", index=False)
    report(df)


def report(df):
    obs = df[df.draw == 0]
    L.note("\n## 5. Seed stability of the ONE cell that beat the scalar baseline in both corpora\n")
    L.note("semi-Markov (state x dwell bucket), K=64, emp_pool_w8. Five independent redraws of "
           "the folds and of the k-means initialisation.\n")
    L.note("| corpus | t | baseline | mode | det AUC per seed | mean | min | "
           "seeds beating baseline | per-seed perm p_res |")
    L.note("|---|---|---|---|---|---|---|---|---|")
    for tag in ("A", "B"):
        for t in L.TS:
            for mode in ("order1", "semi"):
                x = obs[(obs.corpus == tag) & (obs.t == t) & (obs["mode"] == mode)]
                nl = df[(df.corpus == tag) & (df.t == t) & (df["mode"] == mode) & (df.draw > 0)]
                ps = []
                for s in SEEDS:
                    o = x[x.seed == s].res_det_auc.iloc[0]
                    n = nl[nl.seed == s].res_det_auc.values
                    ps.append((1 + (n >= o).sum()) / (len(n) + 1))
                b = x.base.iloc[0]
                v = x.set_index("seed").loc[list(SEEDS), "det_auc"].values
                L.note(f"| {tag} | {t} | {b:.3f} | {mode} | " +
                       ", ".join(f"{q:.3f}" for q in v) +
                       f" | {v.mean():.3f} | {v.min():.3f} | {(v>b).sum()}/5 | " +
                       ", ".join(f"{q:.3f}" for q in ps) + " |")
                L.emit(dict(corpus=tag, block="stability", t=t, mode=mode, base=b,
                            mean_det_auc=v.mean(), min_det_auc=v.min(), max_det_auc=v.max(),
                            n_beat_base=int((v > b).sum()), p_res_median=float(np.median(ps))))
    L.flush()


if __name__ == "__main__":
    import sys
    main(report_only="--report-only" in sys.argv)
