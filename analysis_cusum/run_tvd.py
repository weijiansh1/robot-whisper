"""Second sweep: properly specified (time-varying) class densities, and the
accumulation start index t_start -- how much of the prefix is worth using."""
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


def main():
    grid = list(itertools.product([1, 8], ["cusum", "sprt"], ["tvd", "gauss"],
                                  ["ar1", "none"], [1, 8, 16, 20]))
    allrows = []
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        S = driver.score_real(H, meta)
        for W, rule, dens, wh, ts in grid:
            cfg = dict(W=W, tstd=True, whiten=wh, dens=dens, rule=rule, tstart=ts)
            L, llr, use, aux = driver.statistic(S, meta, cfg)
            auc, npair = driver.peak_auc(L, meta)
            base = dict(section="tvd_sweep", corpus=tag, method="sequential",
                        W=W, rule=rule, dens=dens, whiten=wh, tstart=ts,
                        tstd=True, phi=aux.get("phi"), peak_auc=auc)
            rows = []
            for a in ALPHAS:
                alarm, bounds = seq.logo_alarms(L, meta, float(a))
                m = evalm.metrics(alarm, meta, onset=on)
                rows.append(dict(base, alpha_nominal=float(a), pop="all_fail",
                                 bound_med=float(np.median(list(bounds.values()))), **m))
                for c in cls.columns:
                    if cls[c].values.sum() == 0:
                        continue
                    mc = evalm.metrics(alarm, meta, onset=on, sel=cls[c].values)
                    rows.append(dict(base, alpha_nominal=float(a), pop=c,
                                     bound_med=rows[-1]["bound_med"], **mc))
            lib.append_rows(rows)
            allrows += rows
            af = [r for r in rows if r["pop"] == "all_fail"]
            f = min(af, key=lambda r: abs(r["fa"] - TARGET[tag]))
            print(f"{tag} W={W} {rule:5s} {dens:5s} {wh:4s} ts={ts:2d} auc={auc:.3f} "
                  f"@FA={f['fa']:.3f} det={f['det']:.3f} phase={f['phase_p50']:.3f} "
                  f"t={f['t_p50']:.0f}", flush=True)
    pd.DataFrame(allrows).to_csv("sweep_tvd.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
