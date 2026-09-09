"""Deliverable (2)-(5): matched-false-alarm comparison, sequential vs fixed."""
import itertools
import sys

import numpy as np
import pandas as pd

import driver
import evalm
import lib
import seq

TARGET = {"A": 0.137, "B": 0.051}
ALPHAS = np.round(np.concatenate([np.arange(0.01, 0.31, 0.01), [0.137, 0.051]]), 4)


def sweep(tag, L, meta, cls, on, base):
    rows = []
    for a in ALPHAS:
        alarm, bounds = seq.logo_alarms(L, meta, float(a))
        m = evalm.metrics(alarm, meta, onset=on)
        r = dict(base, alpha_nominal=float(a), pop="all_fail",
                 bound_med=float(np.median([v for v in bounds.values()])), **m)
        rows.append(r)
        for c in cls.columns:
            sel = cls[c].values
            if sel.sum() == 0:
                continue
            mc = evalm.metrics(alarm, meta, onset=on, sel=sel)
            rows.append(dict(base, alpha_nominal=float(a), pop=c,
                             bound_med=r["bound_med"], **mc))
    return rows


def main():
    grid = list(itertools.product([1, 2, 4, 8], ["cusum", "sprt"],
                                  ["gauss", "kde"], ["none", "ar1"]))
    allrows = []
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        np.save(f"Sreal_{tag}.npy", S)
        for W, rule, dens, wh in grid:
            cfg = dict(W=W, tstd=True, whiten=wh, dens=dens, rule=rule)
            L, llr, use, aux = driver.statistic(S, meta, cfg)
            auc, npair = driver.peak_auc(L, meta)
            base = dict(section="seq_sweep", corpus=tag, method="sequential",
                        W=W, rule=rule, dens=dens, whiten=wh, tstd=True,
                        phi=aux.get("phi"), peak_auc=auc)
            rows = sweep(tag, L, meta, cls, on, base)
            allrows += rows
            lib.append_rows(rows)
            f = [r for r in rows if r["pop"] == "all_fail"
                 and abs(r["fa"] - TARGET[tag]) == min(
                     abs(x["fa"] - TARGET[tag]) for x in rows if x["pop"] == "all_fail")][0]
            print(f"{tag} W={W} {rule:5s} {dens:5s} {wh:4s} auc={auc:.3f} "
                  f"@FA={f['fa']:.3f} det={f['det']:.3f} phase={f['phase_p50']:.3f} "
                  f"t={f['t_p50']:.0f}", flush=True)
        # ---- fixed-threshold rule under the identical LOGO protocol
        for W, K in [(8, 3), (8, 1), (8, 2), (8, 4), (4, 3), (12, 3), (1, 3)]:
            Lf = seq.fixed_statistic(S, W=W, K=K)
            auc, npair = driver.peak_auc(Lf, meta)
            base = dict(section="seq_sweep", corpus=tag, method="fixed",
                        W=W, rule=f"K{K}", dens="-", whiten="-", tstd=False,
                        phi=None, peak_auc=auc)
            rows = sweep(tag, Lf, meta, cls, on, base)
            allrows += rows
            lib.append_rows(rows)
            f = [r for r in rows if r["pop"] == "all_fail"
                 and abs(r["fa"] - TARGET[tag]) == min(
                     abs(x["fa"] - TARGET[tag]) for x in rows if x["pop"] == "all_fail")][0]
            print(f"{tag} FIXED W={W} K={K} auc={auc:.3f} @FA={f['fa']:.3f} "
                  f"det={f['det']:.3f} phase={f['phase_p50']:.3f} t={f['t_p50']:.0f}",
                  flush=True)
    pd.DataFrame(allrows).to_csv("sweep_full.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
