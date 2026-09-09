"""Shuffle + clock sentinels re-run on the EXACT configurations reported, and
the branch-level breadth of the late-window separation."""
import json
import sys

import numpy as np
import pandas as pd

import driver
import evalm
import lib
import seq

TARGET = {"A": 0.137, "B": 0.051}
ALPHAS = np.round(np.arange(0.005, 0.35, 0.005), 4)


def op(L, meta, target, on, cls):
    best, ba, bal = None, 1e9, None
    su = meta["success"].values.astype(bool)
    for a in ALPHAS:
        al, _ = seq.logo_alarms(L, meta, float(a))
        fa = (al[su] >= 0).mean()
        if abs(fa - target) < ba:
            best, ba, bal = al, abs(fa - target), float(a)
    return evalm.metrics(best, meta, onset=on), bal


def main():
    xs = json.load(open("xsel.json"))
    rows = []
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        su = meta["success"].values.astype(bool)
        grp = meta["group"].values
        xc, xi = xs[tag], xs["B" if tag == "A" else "A"]
        cfgs = {
            "seq_cusum": dict(W=1, tstd=True, whiten="ar1", dens="tvd", rule="cusum",
                              tstart=1),
            "seq_sprt": dict(W=1, tstd=True, whiten="ar1", dens="tvd", rule="sprt",
                             tstart=1),
            "seq_xcorpus": dict(W=int(xc["W"]), tstd=True, whiten=xc["whiten"],
                                dens=xc["dens"], rule=xc["rule"], tstart=int(xc["tstart"])),
            "seq_bestinsample": dict(W=int(xi["W"]), tstd=True, whiten=xi["whiten"],
                                     dens=xi["dens"], rule=xi["rule"],
                                     tstart=int(xi["tstart"])),
        }
        for variant in ("real", "shuffle11", "shuffle12", "shuffle13", "clock"):
            if variant == "real":
                S = driver.score_real(H, meta)
            elif variant == "clock":
                S = driver.score_clock(meta, H.shape[1])
            else:
                S = driver.score_shuffle(H, meta, int(variant[-2:]))
            for name, cfg in cfgs.items():
                L = driver.statistic(S, meta, cfg)[0]
                auc, npair = driver.peak_auc(L, meta)
                m, bal = op(L, meta, TARGET[tag], on, cls)
                rows.append(dict(section="sentinel2", corpus=tag, detector=name,
                                 variant=variant, peak_auc=auc, alpha_nominal=bal, **m))
            Lf = seq.fixed_statistic(S, W=8, K=3)
            auc, npair = driver.peak_auc(Lf, meta)
            m, bal = op(Lf, meta, TARGET[tag], on, cls)
            rows.append(dict(section="sentinel2", corpus=tag, detector="fixed_W8K3",
                             variant=variant, peak_auc=auc, alpha_nominal=bal, **m))
            print(tag, variant, "done", flush=True)

        # ---- branch-level breadth of the late-window separation
        S = driver.score_real(H, meta)
        for W, t in ((8, 25), (8, 30), (1, 25), (1, 30)):
            MW = seq.smooth(S, W)
            frac = []
            for g in np.unique(grp):
                m = (grp == g) & np.isfinite(MW[:, t])
                med = np.median(MW[m & su, t])
                frac.append(((MW[m & ~su, t] < med).sum(), (m & ~su).sum()))
            k = sum(a for a, _ in frac)
            n = sum(b for _, b in frac)
            a, _ = lib.within_group_auc(-MW[:, t], ~su, grp)
            rows.append(dict(section="breadth", corpus=tag, detector=f"W{W}", variant=f"t{t}",
                             peak_auc=a, frac_fail_below_succ_median=k / n, n_pop=n))
    df = pd.DataFrame(rows)
    df.to_csv("sentinel2.csv", index=False)
    lib.append_rows(rows)
    s = df[df.section == "sentinel2"]
    print(s.pivot_table(index=["corpus", "detector"], columns="variant",
                        values="peak_auc").round(3).to_string())
    print(s.pivot_table(index=["corpus", "detector"], columns="variant",
                        values="det").round(3).to_string())
    print(df[df.section == "breadth"].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
