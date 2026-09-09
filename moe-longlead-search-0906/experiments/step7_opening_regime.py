"""Step 7: can the OPENING REGIME stand in for the suite?  Development only.

Step 6 shows the oracle -- handed the suite, the chunk and a labelled
threshold -- reaches +59 TP at lead >= 16 while every compliant head reaches
+1 to +4.  Almost all of the oracle's advantage is that its threshold is
*suite-conditional*: the same routing value means different things in a
22-chunk suite and a 52-chunk one.

The protocol allows exactly one conditioner that could stand in for that: the
episode's own opening regime, read from routing at chunks 1..5, before any
decision is made.  If the opening fingerprint identifies which suite an episode
belongs to, a threshold stratified by opening-quantile bins is a *compliant*
route to the oracle's per-suite thresholds -- no labels, no classifier, just a
product grid of order statistics.

Three questions:
  1. how much of the suite does the opening fingerprint actually carry?
  2. does a 2-D (opening-quantile x chunk-band) threshold grid, which is pure
     arithmetic on unlabeled order statistics, recover any of the gap?
  3. does the answer change if the fingerprint is made as informative as the
     protocol permits (several series, 16 or 64 bins)?
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bank
import common as C
from step4_ceiling_raw import build_bank
from step5_rules import (lead_profile, rule_excursion, rule_hard,
                         task_concentration)

FINGERPRINT = ("set_inflow_all", "state_mob_all", "mob_back_s9",
               "flow_path_back", "tok_disp_fbr")
CHUNK_BANDS = ((0, 10), (10, 18), (18, 52))
CANDIDATES = (
    ("set_inflow_all", "low", 0.99, "excursion", {"m": 3, "k": 5}),
    ("set_outflow_all", "low", 0.99, "excursion", {"m": 3, "k": 5}),
    ("set_outflow_back", "low", 0.995, "hard", {"confirm": 2}),
)
EARLIEST = 4


def fingerprint_bins(series, names, n_per_axis, length):
    """Product grid of opening quantiles -- order statistics only.

    Each axis is the episode's own mean of one series over chunks 1..5, cut at
    development quantiles.  No clustering, no fitted weights, no labels.
    """
    axes = []
    for n in names:
        op = bank.open_baseline(series[n], 1, 6)[:, 0]
        good = op[np.isfinite(op)]
        if good.size < 500:
            continue
        edges = np.quantile(good, np.linspace(0, 1, n_per_axis + 1)[1:-1])
        b = np.digitize(np.nan_to_num(op, nan=np.inf), edges)
        b[~np.isfinite(op)] = n_per_axis
        axes.append(b)
    code = np.zeros(len(length), dtype=np.int64)
    for b in axes:
        code = code * (n_per_axis + 1) + b
    return code


def purity(code, suite):
    """Fraction of episodes whose suite is the plurality suite of its bin."""
    ok = 0
    for c in np.unique(code):
        m = code == c
        _, counts = np.unique(suite[m], return_counts=True)
        ok += counts.max()
    return ok / len(code)


def grid_threshold(s, code, band_of, quant, direction, min_n=200):
    """Threshold per (opening bin, chunk band); fall back up the hierarchy."""
    level = quant if direction == "high" else 1 - quant
    pool = s[np.isfinite(s)]
    gthr = float(np.quantile(pool, level, method="lower"))
    thr = np.full(s.shape, gthr)
    for band_id, (lo, hi) in enumerate(CHUNK_BANDS):
        cols = slice(lo, min(hi, s.shape[1]))
        band = s[:, cols]
        bpool = band[np.isfinite(band)]
        bthr = (float(np.quantile(bpool, level, method="lower"))
                if bpool.size >= min_n else gthr)
        thr[:, cols] = bthr
        for c in np.unique(code):
            m = code == c
            p = band[m]
            p = p[np.isfinite(p)]
            if p.size >= min_n:
                thr[np.ix_(np.flatnonzero(m), np.arange(*cols.indices(
                    s.shape[1])))] = float(np.quantile(p, level,
                                                       method="lower"))
    return thr


def main() -> None:
    dev = C.load_cohort("development_main")
    series = build_bank("development_main", dev)
    base = dev["v83"]
    bp = lead_profile(base, dev)

    # ---- 1. how much suite information is in the opening? ---------------
    print("=== how much of the suite does the opening regime carry? ===")
    print("  (development, %d episodes, 4 suites; chance purity = largest"
          " suite share)" % len(dev["risk"]))
    _, counts = np.unique(dev["suite"], return_counts=True)
    chance = counts.max() / len(dev["suite"])
    print("  chance                                     purity %.3f" % chance)
    codes = {}
    for n_axis in (2, 4, 8):
        for k in (1, 3, 5):
            code = fingerprint_bins(series, FINGERPRINT[:k], n_axis,
                                    dev["length"])
            codes[(n_axis, k)] = code
            print("  %d quantile bins x %d series = %5d cells   purity %.3f"
                  % (n_axis, k, len(np.unique(code)), purity(code,
                                                             dev["suite"])))
    # the reference: how well does the *cap* separate?  (not available online)
    print("  suite identity itself                      purity 1.000"
          "  (forbidden)")

    # ---- 2. does the 2-D grid recover any of the oracle gap? ------------
    print("\n=== 2-D (opening bin x chunk band) threshold grid ===")
    print("%-20s %-24s %6s %6s %6s %6s %6s"
          % ("series", "threshold basis", "dFP0", "dTP4", "dTP12", "dTP16",
             "conc"))
    rows = []
    for name, direction, quant, rule, params in CANDIDATES:
        s = series[name]
        variants = {"global": np.full(s.shape, float(np.quantile(
            s[np.isfinite(s)], quant if direction == "high" else 1 - quant,
            method="lower")))}
        for (n_axis, k), code in codes.items():
            if k not in (1, 3):
                continue
            variants[f"open{n_axis}^{k} x band"] = grid_threshold(
                s, code, None, quant, direction)
        # chunk band alone, no opening conditioning
        variants["band only"] = grid_threshold(
            s, np.zeros(len(dev["risk"]), int), None, quant, direction)
        for label, thr in variants.items():
            shifted = s - thr
            first = (rule_hard(shifted, 0.0, direction, params.get("confirm", 2),
                               EARLIEST) if rule == "hard"
                     else rule_excursion(shifted, 0.0, direction, params["m"],
                                         params["k"], EARLIEST))
            u = C.union(base, first)
            prof = lead_profile(u, dev)
            conc, _, _ = task_concentration(base, u, dev, 12)
            rows.append({"series": name, "variant": label,
                         "d_fp0": prof[0]["fp"] - bp[0]["fp"],
                         "d_tp4": prof[4]["tp"] - bp[4]["tp"],
                         "d_tp12": prof[12]["tp"] - bp[12]["tp"],
                         "d_tp16": prof[16]["tp"] - bp[16]["tp"],
                         "conc12": conc})
            print("%-20s %-24s %+6d %+6d %+6d %+6d %6.2f"
                  % (name[:20], label, rows[-1]["d_fp0"], rows[-1]["d_tp4"],
                     rows[-1]["d_tp12"], rows[-1]["d_tp16"], conc))
        print()
    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS / "opening_regime_development.csv", index=False)

    (C.RESULTS / "opening_regime.json").write_text(json.dumps({
        "chance_purity": float(chance),
        "purity": {f"{a}bins_x_{k}series": float(purity(c, dev["suite"]))
                   for (a, k), c in codes.items()},
        "n_cells": {f"{a}bins_x_{k}series": int(len(np.unique(c)))
                    for (a, k), c in codes.items()},
        "grid": table.to_dict("records"),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
