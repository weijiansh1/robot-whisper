"""Nuisance check, part 2: reference ceilings and the AT-FIXED-t version.

NMI is uninterpretable without a scale. Two references:
  * clock ceiling  : NMI(t-quantile-bins(K), t)  -- what a PURE K-state clock scores
  * length ceiling : NMI(T-quantile-bins(K), T)  -- what a PURE K-state length code scores
And the operative question for the detection protocol, which conditions on t:
  at a fixed query index t, does the state at t carry T / success?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

import mdp_lib as L

FEATS = ("cell1280", "mean32")


def main():
    L.note("\n### 1b. Reference ceilings and the at-fixed-t view\n")
    L.note("`clock_ceiling` = NMI(K-quantile-bins of t, t): the score a state that is nothing "
           "but a K-level clock would get. `frac_of_clock` = NMI(state,t)/clock_ceiling.\n")
    L.note("| corpus | feat | K | NMI(s,t) | clock_ceil | frac_of_clock | NMI(s,T) | len_ceil | "
           "frac_of_len | t=30: NMI(s_t,T) | t=30: NMI(s_t,succ) |")
    L.note("|---|---|---|---|---|---|---|---|---|---|---|")

    for tag in ("A", "B"):
        P, valid, meta = L.load(tag)
        T = meta["T"].values
        y = meta["success"].values.astype(bool)
        B, Tmax = valid.shape
        tt = np.tile(np.arange(Tmax), (B, 1))
        TT = np.tile(T[:, None], (1, Tmax))
        v = valid
        tv, Tv = tt[v], TT[v]
        for feat in FEATS:
            F = L.features(P, valid, feat)
            for K in L.KS:
                S, _, _ = L.fit_states(F, valid, np.arange(B), K)
                s = S[v]
                # ceilings: the best K-state pure clock / pure length code
                cb = pd.qcut(tv.astype(float) + np.random.default_rng(0).normal(0, 1e-6, tv.size),
                             K, labels=False, duplicates="drop")
                lb = pd.qcut(Tv.astype(float) + np.random.default_rng(1).normal(0, 1e-6, Tv.size),
                             K, labels=False, duplicates="drop")
                ceil_t, ceil_T = nmi(cb, tv), nmi(lb, Tv)
                n_st, n_sT = nmi(s, tv), nmi(s, Tv)
                row = dict(corpus=tag, block="nmi_ref", feat=feat, K=K, nmi_t=n_st,
                           clock_ceil=ceil_t, frac_of_clock=n_st / ceil_t,
                           nmi_T=n_sT, len_ceil=ceil_T, frac_of_len=n_sT / ceil_T)
                for t in L.TS:
                    al = valid[:, t]
                    row[f"nmi_sT_t{t}"] = nmi(S[al, t], T[al])
                    row[f"nmi_sy_t{t}"] = nmi(S[al, t], y[al].astype(int))
                L.emit(row)
                L.note(f"| {tag} | {feat} | {K} | {n_st:.3f} | {ceil_t:.3f} | "
                       f"{n_st/ceil_t:.3f} | {n_sT:.3f} | {ceil_T:.3f} | {n_sT/ceil_T:.3f} | "
                       f"{row['nmi_sT_t30']:.3f} | {row['nmi_sy_t30']:.3f} |")
            del F
        L.flush()
    L.note("")


if __name__ == "__main__":
    main()
