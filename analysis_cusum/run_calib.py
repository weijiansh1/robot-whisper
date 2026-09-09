"""Deliverable (5): was the serial-correlation correction needed, and how much
did it change the ACHIEVED false-alarm rate?

Three bounds are compared at the same nominal alpha:
  wald      h = log((1-beta)/alpha), the i.i.d. Wald/SPRT bound, applied as
            theory prescribes with no calibration at all
  emp       empirical leave-one-group-out calibration, NO whitening
  emp_ar1   empirical leave-one-group-out calibration, AR(1) innovations
Plus the lag-1 autocorrelation of the standardised score and of the AR(1)
innovations, measured on the success class.
"""
import sys

import numpy as np
import pandas as pd

import driver
import evalm
import lib
import seq

NOM = [0.02, 0.05, 0.10, 0.137, 0.20]
BETA = 0.5    # power assumed by the Wald bound; log((1-beta)/alpha)


def acf1(V, meta, sel):
    n = d = 0.0
    for b in np.where(sel)[0]:
        v = V[b, :seq.TMAX]
        v = v[np.isfinite(v)]
        if len(v) > 3:
            n += (v[:-1] * v[1:]).sum()
            d += (v[:-1] ** 2).sum()
    return float(n / max(d, 1e-9))


def main():
    rows = []
    lines = ["", "## 5. Serial correlation and bound calibration", "",
             "| corpus | whiten | acf1(succ) | nominal alpha | achieved FA (Wald bound) "
             "| achieved FA (empirical LOGO) |", "|---|---|---|---|---|---|"]
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        Tb = meta["T"].values
        su = meta["success"].values.astype(bool)
        for wh in ("none", "ar1"):
            cfg = dict(W=1, tstd=True, whiten=wh, dens="tvd", rule="sprt", tstart=1)
            llr, use, aux = seq.build_llr(S, meta, cfg)
            L = seq.accumulate(llr, use, "sprt")
            # the series the LLR is computed on, for the acf report
            a1 = acf1(np.where(use, llr, np.nan), meta, su)
            for al in NOM:
                hw = np.log((1 - BETA) / al)
                aw = seq.alarm_times(L, Tb, hw)
                fa_w = float((aw[su] >= 0).mean())
                det_w = float((aw[~su] >= 0).mean())
                ae, bounds = seq.logo_alarms(L, meta, al)
                fa_e = float((ae[su] >= 0).mean())
                det_e = float((ae[~su] >= 0).mean())
                rows.append(dict(section="calibration", corpus=tag, whiten=wh,
                                 dens="tvd", rule="sprt", W=1, acf1_llr_succ=a1,
                                 alpha_nominal=al, wald_bound=hw,
                                 fa_wald=fa_w, det_wald=det_w,
                                 fa_emp=fa_e, det_emp=det_e,
                                 emp_bound_med=float(np.median(list(bounds.values())))))
                lines.append(f"| {tag} | {wh} | {a1:.3f} | {al:.3f} | {fa_w:.3f} | {fa_e:.3f} |")
                print(lines[-1], flush=True)
    # raw-score autocorrelation for context
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        su = meta["success"].values.astype(bool)
        Z = np.full_like(S, np.nan)
        for t in range(1, seq.TMAX):
            m = np.isfinite(S[:, t])
            Z[m, t] = (S[m, t] - S[m, t].mean()) / (S[m, t].std() + 1e-12)
        cfg = dict(W=1, tstd=True, whiten="ar1", dens="tvd", rule="sprt", tstart=1)
        _, _, aux = seq.build_llr(S, meta, cfg)
        rows.append(dict(section="calibration_acf", corpus=tag,
                         acf1_z_succ=acf1(Z, meta, su),
                         acf1_z_all=acf1(Z, meta, np.ones(len(meta), bool)),
                         ar1_phi=aux["phi"]))
        lines.append(f"corpus {tag}: acf1(z|succ) = {acf1(Z, meta, su):.3f}, "
                     f"fitted AR(1) phi = {aux['phi']:.3f}")
        print(lines[-1], flush=True)
    lib.append_rows(rows)
    lib.append_md("\n".join(lines))
    pd.DataFrame(rows).to_csv("calibration.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
