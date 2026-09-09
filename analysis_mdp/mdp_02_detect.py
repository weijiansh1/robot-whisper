"""(4)(5) Cross-fitted absorption-probability detection vs the scalar baseline.

Every model quantity that touches the label (absorption probs, empirical per-state
failure rates) is fitted on 4/5 of the branches and applied to the held-out 1/5, so a
branch never contributes to the model that scores it. The state space itself
(PCA + k-means) is also fitted on training folds only.

Permutation null: within-group branch-level label permutation, WITH REFIT of every
label-dependent model quantity (the state assignment is label-free and is held fixed).
"""
import argparse
import time
import numpy as np
import pandas as pd

import mdp_lib as L

FEATS = ("cell1280", "mean32")
W = 8
DWELL_BUCKETS = np.array([0, 1, 3, 6, 10**9])   # dwell 1 / 2-3 / 4-6 / 7+


class FoldModel:
    """Everything label-free, precomputed once; label-dependent parts refit per draw."""

    def __init__(self, S, valid, train_b, test_b, K, ts):
        self.S, self.valid, self.K = S, valid, K
        self.train_b, self.test_b, self.ts = train_b, test_b, ts
        B = S.shape[0]
        self.C, self.term = L.chain_counts(S, valid, train_b, K)
        exit_ct = np.bincount(self.term[self.term >= 0], minlength=K).astype(float)
        tot = np.maximum(self.C.sum(1) + exit_ct, 1e-9)
        self.Pm = self.C / tot[:, None]
        self.logP = np.log(np.maximum(self.Pm, 1e-6))
        # chunk-occupancy matrix for the pooled empirical rate
        self.M = np.zeros((K, len(train_b)))
        for j, b in enumerate(train_b):
            s = S[b][valid[b]]
            if len(s):
                np.add.at(self.M[:, j], s, 1.0)
        self.Mn = self.M.sum(1)
        # time-local index sets
        self.alive_tr = {t: np.asarray(train_b)[valid[np.asarray(train_b), t]] for t in ts}
        self.st_tr = {t: S[self.alive_tr[t], t] for t in ts}
        # dwell length at every chunk (for the semi-Markov variant + dwell feature)
        self.dw = np.zeros(S.shape, np.int32)
        for b in range(B):
            s = S[b]
            n = int(valid[b].sum())
            d = 0
            prev = -99
            for i in range(n):
                d = d + 1 if s[i] == prev else 1
                self.dw[b, i] = d
                prev = s[i]

    # -------------------------------------------------- label-dependent refit
    def fit(self, fail):
        K = self.K
        self.h, _, _, self.esteps = L.absorption(self.C, self.term, self.train_b, fail, K)
        f = fail[np.asarray(self.train_b)].astype(float)
        g = f.mean()
        self.rate_pool = (self.M @ f + 5.0 * g) / (self.Mn + 5.0)
        self.rate_t = {}
        for t in self.ts:
            al = self.alive_tr[t]
            s = self.st_tr[t]
            ff = fail[al].astype(float)
            n = np.bincount(s, minlength=K).astype(float)
            num = np.bincount(s, weights=ff, minlength=K)
            gg = ff.mean() if len(ff) else g
            self.rate_t[t] = (num + 5.0 * gg) / (n + 5.0)

    # -------------------------------------------------- scores on the held-out branches
    def scores(self, t):
        te = self.test_b[self.valid[self.test_b, t]]
        if len(te) == 0:
            return te, {}
        lo = max(0, t - W + 1)
        win = self.S[np.ix_(te, np.arange(lo, t + 1))]        # (n, W)
        st = self.S[te, t]
        u0 = max(1, lo)                                       # transitions u-1 -> u
        frm = self.S[np.ix_(te, np.arange(u0 - 1, t))]
        to = self.S[np.ix_(te, np.arange(u0, t + 1))]
        out = {}
        out["absorb"] = self.h[st]
        out["absorb_w8"] = self.h[win].mean(1)
        out["absorb_drift"] = self.h[st] - self.h[win[:, 0]]
        out["absorb_max_w8"] = self.h[win].max(1)
        med = np.median(self.h)
        out["frac_hirisk_w8"] = (self.h[win] > med).mean(1)
        out["emp_pool"] = self.rate_pool[st]
        out["emp_pool_w8"] = self.rate_pool[win].mean(1)
        out["emp_time"] = self.rate_t[t][st]
        out["esteps"] = self.esteps[st]
        out["esteps_w8"] = self.esteps[win].mean(1)
        out["dwell"] = self.dw[te, t].astype(float)
        out["stayp_w8"] = np.diag(self.Pm)[win].mean(1)
        out["nll_w8"] = -self.logP[frm, to].mean(1)
        return te, out


VARIANTS = ["absorb", "absorb_w8", "absorb_drift", "absorb_max_w8", "frac_hirisk_w8",
            "emp_pool", "emp_pool_w8", "emp_time", "esteps", "esteps_w8",
            "dwell", "stayp_w8", "nll_w8"]


def assemble(models, fail, B, ts):
    for m in models:
        m.fit(fail)
    out = {t: {v: np.full(B, np.nan) for v in VARIANTS} for t in ts}
    for m in models:
        for t in ts:
            te, sc = m.scores(t)
            for v, arr in sc.items():
                out[t][v][te] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=0)
    ap.add_argument("--feats", default=",".join(FEATS))
    args = ap.parse_args()
    feats = args.feats.split(",")

    rows = []
    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        y = meta["success"].values.astype(bool)
        fail = ~y
        g = meta["group"].values
        T = meta["T"].values
        B = len(meta)
        fd = L.folds(meta)
        # baseline (label-free, no cross-fitting needed)
        d = L.agglib.divergence_series(P, valid, "hell|prev")
        base = {t: L.agglib.aggregate(d, t, "mean", W) for t in L.TS}

        for feat in feats:
            F = L.features(P, valid, feat)
            for K in L.KS:
                t0 = time.time()
                models = []
                for f in range(L.NFOLD):
                    tr = np.where(fd != f)[0]
                    te = np.where(fd == f)[0]
                    S, _, _ = L.fit_states(F, valid, tr, K)
                    models.append(FoldModel(S, valid, tr, te, K, L.TS))
                sc = assemble(models, fail, B, L.TS)

                # ---- observed
                obs = {}
                for t in L.TS:
                    alive = T > t
                    b = base[t]
                    ab, db, _ = L.auc_pair(b[alive], y[alive], g[alive])
                    for v in VARIANTS:
                        s = sc[t][v]
                        a, dd, n = L.auc_pair(s[alive], y[alive], g[alive])
                        r = L.agglib.rank_residualise(s, b, g)
                        ar, dr, _ = L.auc_pair(r[alive], y[alive], g[alive])
                        obs[(t, v)] = (dd, dr)
                        rows.append(dict(corpus=tag, block="detect", feat=feat, K=K, t=t,
                                         variant=v, auc_succ=a, det_auc=dd,
                                         res_auc_succ=ar, res_det_auc=dr,
                                         base_det_auc=db, npairs=n,
                                         n_scored=int(np.isfinite(s[alive]).sum())))

                # ---- permutation null (full refit of the label-dependent model)
                if args.perm:
                    rng = np.random.default_rng(7 + K)
                    cnt = {k: [0, 0] for k in obs}
                    for it in range(args.perm):
                        yp = y.copy()
                        for gg in np.unique(g):
                            idx = np.where(g == gg)[0]
                            yp[idx] = y[idx][rng.permutation(len(idx))]
                        scp = assemble(models, ~yp, B, L.TS)
                        for t in L.TS:
                            alive = T > t
                            b = base[t]
                            for v in VARIANTS:
                                s = scp[t][v]
                                _, dd, _ = L.auc_pair(s[alive], yp[alive], g[alive])
                                r = L.agglib.rank_residualise(s, b, g)
                                _, dr, _ = L.auc_pair(r[alive], yp[alive], g[alive])
                                o = obs[(t, v)]
                                cnt[(t, v)][0] += (dd >= o[0])
                                cnt[(t, v)][1] += (dr >= o[1])
                    for (t, v), c in cnt.items():
                        rows.append(dict(corpus=tag, block="perm", feat=feat, K=K, t=t, variant=v,
                                         p_raw=(c[0] + 1) / (args.perm + 1),
                                         p_res=(c[1] + 1) / (args.perm + 1), ndraw=args.perm))
                print(f"{tag} {feat} K={K} done {time.time()-t0:.1f}s", flush=True)
                pd.DataFrame(rows).to_csv(f"{L.OUT}/detect_raw.csv", index=False)
                pd.DataFrame(rows).to_csv("/tmp/mdp_detect_raw.csv", index=False)
            del F
    df = pd.DataFrame(rows)
    df.to_csv(f"{L.OUT}/detect_raw.csv", index=False)
    df.to_csv("/tmp/mdp_detect_raw.csv", index=False)
    print("wrote", len(df), "rows")


if __name__ == "__main__":
    main()
