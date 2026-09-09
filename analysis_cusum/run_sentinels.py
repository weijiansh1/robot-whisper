"""Deliverable (1): the two mandatory sentinels, run before anything else."""
import sys

import numpy as np

import driver
import evalm
import lib
import seq

ALPHA = {"A": 0.137, "B": 0.051}
CFGS = {
    "TS": dict(W=1, tstd=True, whiten="ar1", dens="gauss", rule="cusum"),
    "TH": dict(W=1, tstd=False, whiten="ar1", dens="gauss", rule="cusum"),
    "TS_sprt": dict(W=1, tstd=True, whiten="ar1", dens="gauss", rule="sprt"),
    "TH_sprt": dict(W=1, tstd=False, whiten="ar1", dens="gauss", rule="sprt"),
}


def main():
    rows = []
    lines = ["", "## 1. Sentinels", "",
             "| corpus | score | pipeline | rule | peak AUC | FA | det | med phase | med t |",
             "|---|---|---|---|---|---|---|---|---|"]
    for tag in "AB":
        meta, H, cls, on = driver.load(tag)
        alpha = ALPHA[tag]
        Tm = H.shape[1]
        variants = {"real": driver.score_real(H, meta)}
        for s in (11, 12, 13):
            variants[f"shuffle{s}"] = driver.score_shuffle(H, meta, s)
        variants["clock"] = driver.score_clock(meta, Tm)
        for sname, S in variants.items():
            for cname, cfg in CFGS.items():
                L, llr, use, aux = driver.statistic(S, meta, cfg)
                auc, npair = driver.peak_auc(L, meta)
                alarm, bounds = seq.logo_alarms(L, meta, alpha)
                m = evalm.metrics(alarm, meta, onset=on)
                r = dict(section="sentinel", corpus=tag, score=sname, pipeline=cname,
                         rule=cfg["rule"], W=cfg["W"], tstd=cfg["tstd"],
                         whiten=cfg["whiten"], dens=cfg["dens"], phi=aux.get("phi"),
                         alpha_nominal=alpha, peak_auc=auc, n_pair=npair, **m)
                rows.append(r)
                lines.append(f"| {tag} | {sname} | {cname} | {cfg['rule']} | {auc:.3f} | "
                             f"{m['fa']:.3f} | {m['det']:.3f} | {m['phase_p50']:.3f} | "
                             f"{m['t_p50']:.1f} |")
                print(lines[-1], flush=True)
        # strongest pure-clock detector: statistic == t
        Lc = driver.clock_statistic(meta, Tm)
        auc, npair = driver.peak_auc(Lc, meta)
        alarm, bounds = seq.logo_alarms(Lc, meta, alpha)
        m = evalm.metrics(alarm, meta, onset=on)
        rows.append(dict(section="sentinel", corpus=tag, score="clock_oracle",
                         pipeline="direct", rule="threshold_t", W=0, tstd=None,
                         whiten=None, dens=None, phi=None, alpha_nominal=alpha,
                         peak_auc=auc, n_pair=npair, **m))
        lines.append(f"| {tag} | clock_oracle | direct | t>=h | {auc:.3f} | {m['fa']:.3f} | "
                     f"{m['det']:.3f} | {m['phase_p50']:.3f} | {m['t_p50']:.1f} |")
        print(lines[-1], flush=True)
    lib.append_rows(rows)
    lib.append_md("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
