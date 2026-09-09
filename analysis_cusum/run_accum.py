"""Deliverable (6): is the t=25 information accumulable or concentrated?

Alarm mechanics are stripped out.  At every index t we compare the within-group
paired AUC of

  fixed_W8   the 8-step window mean at t (the current rule's statistic)
  cusum      the causal CUSUM statistic at t
  sprt       the causal cumulative LLR at t
  cummean    the plain causal mean of the standardised score over 1..t

If accumulation works, `sprt`/`cummean` at t is above `fixed_W8` at t, because
they see the whole prefix.  If the information is instead concentrated in a few
branches, the extra prefix buys nothing and the curves coincide.
"""
import sys

import numpy as np
import pandas as pd

import driver
import lib
import seq

CFG = dict(W=1, tstd=True, whiten="ar1", dens="tvd", rule="cusum")


def main():
    rows = []
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        M8 = seq.smooth(S, 8)
        llr, use, aux = seq.build_llr(S, meta, CFG)
        Lc = seq.accumulate(llr, use, "cusum")
        Ls = seq.accumulate(llr, use, "sprt")
        # plain causal mean of the standardised score (no density model at all)
        Z = np.full_like(S, np.nan)
        grp, su = meta["group"].values, meta["success"].values.astype(bool)
        for g in np.unique(grp):
            tr, te = grp != g, grp == g
            for t in range(1, seq.TMAX):
                v = S[tr, t]
                v = v[np.isfinite(v)]
                if len(v) >= 10:
                    Z[te, t] = (S[te, t] - v.mean()) / max(v.std(), 1e-9)
        Cm = np.full_like(S, np.nan)
        for b in range(len(meta)):
            idx = np.where(np.isfinite(Z[b, :seq.TMAX]))[0]
            if len(idx):
                Cm[b, idx] = np.cumsum(Z[b, idx]) / np.arange(1, len(idx) + 1)
        y = ~su
        for t in range(8, seq.TMAX):
            for name, V, sgn in (("fixed_W8", M8, -1.0), ("cusum", Lc, 1.0),
                                 ("sprt", Ls, 1.0), ("cummean", Cm, -1.0)):
                a, n = lib.within_group_auc(sgn * V[:, t], y, grp)
                rows.append(dict(section="accum", corpus=tag, stat=name, t=t,
                                 auc=a, n_pair=n))
        # per-branch evidence trajectory summary: how concentrated is the
        # total evidence?
        tot = np.nansum(np.where(np.isfinite(llr[:, :seq.TMAX]), llr[:, :seq.TMAX], 0), 1)
        for pop, sel in ([("all_fail", y), ("success", su)] +
                         [(c, cls[c].values) for c in cls.columns if cls[c].values.sum()]):
            v = tot[sel]
            if len(v) == 0:
                continue
            sv = np.sort(v)[::-1]
            rows.append(dict(section="accum_concentration", corpus=tag, stat=pop,
                             t=-1, auc=np.nan, n_pair=len(v),
                             mean_total_llr=float(v.mean()),
                             med_total_llr=float(np.median(v)),
                             frac_positive=float((v > 0).mean()),
                             p90_share=float(sv[:max(1, len(v) // 10)].sum()
                                             / max(v.sum(), 1e-9))))
        np.save(f"llr_{tag}.npy", llr)
        np.save(f"Lcusum_{tag}.npy", Lc)
        np.save(f"Lsprt_{tag}.npy", Ls)
    df = pd.DataFrame(rows)
    df.to_csv("accum.csv", index=False)
    lib.append_rows(rows)
    piv = df[df.section == "accum"].pivot_table(index=["corpus", "t"], columns="stat",
                                                values="auc")
    print(piv.round(3).to_string())
    print(df[df.section == "accum_concentration"].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
