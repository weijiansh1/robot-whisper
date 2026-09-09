"""Final operating points: class breakouts, alarm-phase and alarm-index
distributions, delay to the frozen physical loop onset, ARL0/ARL1, and the
accumulable-vs-concentrated evidence analysis."""
import ast
import sys

import numpy as np
import pandas as pd

import driver
import evalm
import lib
import seq

TARGET = {"A": 0.137, "B": 0.051}
ALPHAS = np.round(np.arange(0.005, 0.35, 0.005), 4)


def best_alpha(L, meta, target):
    best, ba, bal = None, 1e9, None
    for a in ALPHAS:
        al, bd = seq.logo_alarms(L, meta, float(a))
        fa = (al[meta["success"].values.astype(bool)] >= 0).mean()
        if abs(fa - target) < ba:
            best, ba, bal = al, abs(fa - target), float(a)
    return best, bal, ba


def main():
    import json; xs = json.load(open("xsel.json"))
    lines = []
    rows = []
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        Tb = meta["T"].values
        su = meta["success"].values.astype(bool)
        dets = {}
        for name, cfg in [
            ("seq_cusum", dict(W=1, tstd=True, whiten="ar1", dens="tvd",
                               rule="cusum", tstart=1)),
            ("seq_sprt", dict(W=1, tstd=True, whiten="ar1", dens="tvd",
                              rule="sprt", tstart=1)),
        ]:
            L, llr, use, aux = driver.statistic(S, meta, cfg)
            dets[name] = L
            if name == "seq_cusum":
                np.save(f"llr_final_{tag}.npy", llr)
        # xs[tag] is the config that scored best on the OTHER corpus (honest
        # transfer); xs[other] is the one that scored best on THIS corpus and is
        # kept only as an explicitly selection-optimistic ceiling.
        for nm, key in (("seq_xcorpus", tag), ("seq_bestinsample", "B" if tag == "A" else "A")):
            xc = xs[key]
            cfgx = dict(W=int(xc["W"]), tstd=True, whiten=xc["whiten"], dens=xc["dens"],
                        rule=xc["rule"], tstart=int(xc["tstart"]))
            dets[nm] = driver.statistic(S, meta, cfgx)[0]
        dets["fixed_W8K3"] = seq.fixed_statistic(S, W=8, K=3)

        for name, L in dets.items():
            alarm, al, err = best_alpha(L, meta, TARGET[tag])
            np.save(f"alarm_{tag}_{name}.npy", alarm)
            for pop, sel in ([("all_fail", ~su)] +
                             [(c, cls[c].values) for c in cls.columns
                              if cls[c].values.sum()]):
                m = evalm.metrics(alarm, meta, onset=on, sel=sel)
                rows.append(dict(section="final", corpus=tag, detector=name,
                                 pop=pop, alpha_nominal=al, **m))
        # published rule, literal global threshold, no calibration
        alarm = seq.fixed_alarms(S, Tb, 0.0348, K=3, W=8)
        np.save(f"alarm_{tag}_fixed_literal.npy", alarm)
        for pop, sel in ([("all_fail", ~su)] +
                         [(c, cls[c].values) for c in cls.columns if cls[c].values.sum()]):
            m = evalm.metrics(alarm, meta, onset=on, sel=sel)
            rows.append(dict(section="final", corpus=tag, detector="fixed_literal_0.0348",
                             pop=pop, alpha_nominal=np.nan, **m))

        # ---- per-step information profile: where does the evidence live?
        Z = np.full_like(S, np.nan)
        grp = meta["group"].values
        for t in range(1, seq.TMAX):
            mm = np.isfinite(S[:, t])
            Z[mm, t] = S[mm, t]
        for t in range(8, seq.TMAX):
            a1, n1 = lib.within_group_auc(-S[:, t], ~su, grp)          # single step
            recs = dict(section="perstep", corpus=tag, t=t, auc_step=a1, n_pair=n1)
            for W in (2, 4, 8, 12):
                MW = seq.smooth(S, W)
                a2, _ = lib.within_group_auc(-MW[:, t], ~su, grp)
                recs[f"auc_W{W}"] = a2
            # anchored-at-1 causal mean == what a plain accumulator sees
            Cm = np.nanmean(S[:, 1:t + 1], axis=1)
            a3, _ = lib.within_group_auc(-Cm, ~su, grp)
            recs["auc_prefix_mean"] = a3
            rows.append(recs)
    df = pd.DataFrame(rows)
    df.to_csv("final.csv", index=False)
    lib.append_rows(rows)
    print(df[df.section == "final"][
        ["corpus", "detector", "pop", "n_pop", "fa", "det", "phase_p50", "t_p50",
         "arl0", "arl1", "n_delay", "frac_early", "delay_p50"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
