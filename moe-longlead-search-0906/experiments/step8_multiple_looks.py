"""Step 8: the mechanism, and the last compliant sweep.  Development only.

Step 7 produced a surprise: a product grid of opening-regime quantiles
recovers the suite with purity 0.947 (49 cells) to 0.995 (2,002 cells), so the
cap-free constraint is *not* what is blocking the per-suite thresholds -- the
episode's first five chunks already say which suite it is in.  Yet stratifying
by that grid still buys only +3 to +7 TP at lead >= 16.

The remaining explanation is arithmetic, and it is about how many times the
detector looks.  For lead >= L the alarm must land in the window
[earliest, cap - L], and a causal detector tests every chunk in that window.
With per-chunk false-alarm rate a and window width W the episode-level rate is
1 - (1-a)^W, so the affordable per-chunk a is roughly the corpus budget divided
by W.  v8.3 spends 134 false alarms on 31,602 successes = 0.42%; over a
ten-chunk window that is a per-chunk a of about 4e-4, the q99.96 tail.  Every
within-suite fixed-chunk number in the literature on this problem -- including
the 0.85-0.89 AUCs -- is measured at a of about 4e-2, a hundred times looser.

Part 1 measures that compounding directly.
Part 2 is the last compliant sweep: opening-regime x chunk-band thresholds
across the full quantile range, which is the combination step 7 left out.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import bank
import common as C
from step4_ceiling_raw import build_bank
from step5_rules import lead_profile, rule_excursion, rule_hard, task_concentration
from step7_opening_regime import CHUNK_BANDS, FINGERPRINT, fingerprint_bins

SERIES = ("set_inflow_all", "set_outflow_all", "set_outflow_back",
          "set_jacc_adj_all", "state_mob_all", "rank_footrule_open_all")
QUANTILES = (0.90, 0.95, 0.96, 0.98, 0.99, 0.995)
RULES = (("hard", {"confirm": 1}), ("hard", {"confirm": 2}),
         ("hard", {"confirm": 3}), ("excursion", {"m": 2, "k": 3}),
         ("excursion", {"m": 3, "k": 5}), ("excursion", {"m": 4, "k": 6}))
EARLIEST = 4
FP_BUDGET = 20
SEED = 20260906


def strat_threshold(s, code, quant, direction, min_n=200):
    level = quant if direction == "high" else 1 - quant
    pool = s[np.isfinite(s)]
    thr = np.full(s.shape, float(np.quantile(pool, level, method="lower")))
    for lo, hi in CHUNK_BANDS:
        hi = min(hi, s.shape[1])
        if lo >= hi:
            continue
        band = s[:, lo:hi]
        for c in np.unique(code):
            m = code == c
            p = band[m]
            p = p[np.isfinite(p)]
            if p.size >= min_n:
                thr[np.ix_(np.flatnonzero(m), np.arange(lo, hi))] = float(
                    np.quantile(p, level, method="lower"))
    return thr


def fire(s, thr, direction, rule, params, earliest=EARLIEST):
    shifted = s - thr
    if rule == "hard":
        return rule_hard(shifted, 0.0, direction, params["confirm"], earliest)
    return rule_excursion(shifted, 0.0, direction, params["m"], params["k"],
                          earliest)


def main() -> None:
    dev = C.load_cohort("development_main")
    series = build_bank("development_main", dev)
    base = dev["v83"]
    bp = lead_profile(base, dev)

    # ---- Part 1: how the false-alarm rate compounds over the window -----
    print("=== how many looks does a long lead force? ===")
    print("  for lead >= L the alarm must land at chunk <= cap - L, and a"
          " causal detector tests every chunk from `earliest` on\n")
    print("%-16s %4s %10s %10s %10s"
          % ("suite", "cap", "L>=4 W", "L>=12 W", "L>=16 W"))
    for suite in C.SUITES:
        cap = C.CAP[suite]
        print("%-16s %4d %10d %10d %10d"
              % (suite, cap, max(cap - 4 - EARLIEST + 1, 0),
                 max(cap - 12 - EARLIEST + 1, 0),
                 max(cap - 16 - EARLIEST + 1, 0)))

    print("\n  measured compounding: one fixed per-chunk quantile threshold,"
          " applied over a widening window")
    print("%-10s %12s %12s %12s"
          % ("per-chunk", "W=1 (q10)", "W=5 (q6..10)", "W=11 (q4..14)"))
    s = series["set_inflow_all"]
    safe = ~dev["risk"]
    for alpha in (0.04, 0.01, 0.004, 0.001):
        thr = float(np.quantile(s[np.isfinite(s)], alpha, method="lower"))
        hit = (s <= thr) & np.isfinite(s)
        cells = []
        for lo, hi in ((10, 11), (6, 11), (4, 15)):
            fired = hit[:, lo:hi].any(axis=1)
            cells.append("%d/%d" % (int((fired & safe).sum()),
                                    int(safe.sum())))
        print("%-10.3f %12s %12s %12s" % (alpha, *cells))
    print("  (v8.3 spends 82 false alarms at lead>=0 on 13,299 development"
          " successes = 0.62%)")

    # ---- Part 2: the last compliant sweep -------------------------------
    code = fingerprint_bins(series, FINGERPRINT[:3], 4, dev["length"])
    print("\n=== compliant sweep: opening grid (%d cells, suite purity 0.947)"
          " x chunk band x quantile x rule ===" % len(np.unique(code)))
    rows = []
    for name, direction, quant, (rule, params) in itertools.product(
            SERIES, ("high", "low"), QUANTILES, RULES):
        s = series[name]
        for basis in ("global", "strat"):
            if basis == "global":
                level = quant if direction == "high" else 1 - quant
                thr = np.full(s.shape, float(np.quantile(
                    s[np.isfinite(s)], level, method="lower")))
            else:
                thr = strat_threshold(s, code, quant, direction)
            first = fire(s, thr, direction, rule, params)
            u = C.union(base, first)
            prof = lead_profile(u, dev)
            conc, gain, top = task_concentration(base, u, dev, 12)
            rows.append({"series": name, "direction": direction,
                         "quantile": quant, "rule": rule,
                         "params": json.dumps(params), "basis": basis,
                         "d_fp0": prof[0]["fp"] - bp[0]["fp"],
                         "d_tp4": prof[4]["tp"] - bp[4]["tp"],
                         "d_fp4": prof[4]["fp"] - bp[4]["fp"],
                         "d_tp12": prof[12]["tp"] - bp[12]["tp"],
                         "d_tp16": prof[16]["tp"] - bp[16]["tp"],
                         "conc12": conc, "gain12": gain, "top_task": top})
    grid = pd.DataFrame(rows)
    grid.to_csv(C.RESULTS / "strat_sweep_development.csv", index=False)
    print("configurations: %d" % len(grid))

    ok = grid[(grid.d_fp0 <= FP_BUDGET) & (grid.conc12 <= 0.5)]
    print("inside the +%d FP@L0 budget with no single task carrying >50%%"
          " of the L12 gain: %d" % (FP_BUDGET, len(ok)))

    for metric in ("d_tp16", "d_tp12", "d_tp4"):
        print("\n--- best 6 by %s (development, inside budget) ---" % metric)
        top = ok.sort_values([metric, "d_tp4", "d_fp0"],
                             ascending=[False, False, True]).head(6)
        for _, r in top.iterrows():
            print("  %-22s %-4s q%.3f %-9s %-14s %-6s  dFP0 %+3d  dTP4 %+3d"
                  "  dTP12 %+3d  dTP16 %+3d  conc %.2f"
                  % (r.series, r.direction, r["quantile"], r.rule,
                     r.params[:14], r.basis, r.d_fp0, r.d_tp4, r.d_tp12,
                     r.d_tp16, r.conc12))

    print("\n--- widened budget: the frontier of dTP16 against dFP0 ---")
    for budget in (10, 20, 40, 80, 160):
        sub = grid[(grid.d_fp0 <= budget) & (grid.conc12 <= 0.5)]
        if sub.empty:
            continue
        r = sub.sort_values(["d_tp16", "d_tp12"], ascending=False).iloc[0]
        print("  dFP0 <= %3d :  best dTP16 %+3d  (dTP12 %+3d, dTP4 %+3d)"
              "  %-22s %-6s q%.3f %s"
              % (budget, r.d_tp16, r.d_tp12, r.d_tp4, r.series, r.basis,
                 r["quantile"], r.rule))

    (C.RESULTS / "multiple_looks.json").write_text(json.dumps({
        "n_configs": len(grid), "n_inside_budget": len(ok),
        "opening_cells": int(len(np.unique(code))),
        "best_tp16": ok.sort_values("d_tp16", ascending=False)
        .head(3).to_dict("records"),
        "best_tp12": ok.sort_values("d_tp12", ascending=False)
        .head(3).to_dict("records"),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
