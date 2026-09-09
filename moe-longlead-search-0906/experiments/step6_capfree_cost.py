"""Step 6: is the plateau the protocol, or is it the signal?  Development only.

Step 5 reproduces the plateau exactly: the best cap-free head buys +52 TP at
lead >= 4 for +18 false alarms, and +13 / +1 at lead >= 12 / 16.  Two
explanations are possible and they lead to opposite next moves:

  (A) the signal is there but the CAP-FREE PROTOCOL destroys it.  A single
      global threshold must serve four suites whose caps are 30 / 52 / 28 / 22
      and whose series distributions differ; the ceiling scan measured its
      numbers *within* a suite at a *fixed* chunk, which is a luxury the
      detector does not have.
  (B) the signal is not there early, and the within-suite fixed-chunk numbers
      are a base-rate illusion.

These are separated by building the same head three ways and comparing:

  global   one threshold from the unlabeled development pool  (COMPLIANT)
  strat    one threshold per bin of the episode's own OPENING REGIME, binned
           by development quantiles                            (COMPLIANT)
  suite    one threshold per suite                             (VIOLATES the
           protocol -- reported only as an upper bound on what (A) could buy)

and against the absolute ceiling: an ORACLE that is handed the suite, the
chunk and the labelled threshold, and fires at the single best (suite, chunk,
series) cell inside each suite's lead>=16 deadline at a fixed 20-false-alarm
budget.  If even the oracle cannot move lead >= 16, the answer is (B) and no
amount of protocol relaxation will help.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bank
import common as C
from step3_ceiling import DEADLINE
from step4_ceiling_raw import build_bank
from step5_rules import (lead_profile, opening_bin, rule_excursion, rule_hard,
                         task_concentration)

CANDIDATES = (
    ("set_inflow_all", "low", 0.99, "excursion", {"m": 3, "k": 5}),
    ("set_outflow_all", "low", 0.99, "excursion", {"m": 3, "k": 5}),
    ("set_outflow_back", "low", 0.995, "hard", {"confirm": 2}),
    ("set_jacc_adj_all", "low", 0.99, "hard", {"confirm": 2}),
    ("state_mob_all", "high", 0.975, "excursion", {"m": 3, "k": 5}),
)
BINS = (4, 8, 16)
EARLIEST = 4


def apply_rule(s, thr, direction, rule, params, earliest=EARLIEST):
    if rule == "hard":
        return rule_hard(s, thr, direction, params["confirm"], earliest)
    return rule_excursion(s, thr, direction, params["m"], params["k"],
                          earliest)


def thresholded(s, groups, quant, direction, min_n=200):
    """Per-group order statistic, falling back to the global one."""
    level = quant if direction == "high" else 1 - quant
    pool = s[np.isfinite(s)]
    gthr = float(np.quantile(pool, level, method="lower"))
    out = np.full(s.shape[0], gthr)
    for g in np.unique(groups):
        m = groups == g
        p = s[m]
        p = p[np.isfinite(p)]
        if p.size >= min_n:
            out[m] = float(np.quantile(p, level, method="lower"))
    return out


def first_with_row_threshold(s, thr_row, direction, rule, params):
    """Row-wise thresholds: run the rule on the centred series against 0."""
    shifted = s - thr_row[:, None]
    return apply_rule(shifted, 0.0, direction, rule, params)


def main() -> None:
    dev = C.load_cohort("development_main")
    series = build_bank("development_main", dev)
    base = dev["v83"]
    bp = lead_profile(base, dev)
    print("v8.3 development: " + "  ".join(
        "L%d %d/%d" % (l, bp[l]["tp"], bp[l]["fp"]) for l in C.LEADS))

    rows = []
    for name, direction, quant, rule, params in CANDIDATES:
        s = series[name]
        variants = {}
        variants["global"] = np.full(s.shape[0], float(np.quantile(
            s[np.isfinite(s)],
            quant if direction == "high" else 1 - quant, method="lower")))
        variants["suite(VIOLATION)"] = thresholded(s, dev["suite"], quant,
                                                   direction)
        variants["task(VIOLATION)"] = thresholded(s, dev["task"], quant,
                                                  direction, min_n=100)
        for nb in BINS:
            b, _ = opening_bin(s, nb)
            variants[f"strat{nb}"] = thresholded(s, b, quant, direction)
        for label, thr_row in variants.items():
            first = first_with_row_threshold(s, thr_row, direction, rule,
                                             params)
            u = C.union(base, first)
            prof = lead_profile(u, dev)
            conc, gain, top = task_concentration(base, u, dev, 12)
            rows.append({"series": name, "rule": rule, "variant": label,
                         "d_fp0": prof[0]["fp"] - bp[0]["fp"],
                         "d_tp0": prof[0]["tp"] - bp[0]["tp"],
                         "d_tp4": prof[4]["tp"] - bp[4]["tp"],
                         "d_fp4": prof[4]["fp"] - bp[4]["fp"],
                         "d_tp12": prof[12]["tp"] - bp[12]["tp"],
                         "d_tp16": prof[16]["tp"] - bp[16]["tp"],
                         "conc12": conc, "top_task": top})
    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS / "capfree_cost_development.csv", index=False)

    print("\n=== does knowing the suite / task help?  (development) ===")
    print("%-20s %-18s %6s %6s %6s %6s"
          % ("series", "threshold basis", "dFP0", "dTP4", "dTP12", "dTP16"))
    for name, _, _, _, _ in CANDIDATES:
        for _, r in table[table.series == name].iterrows():
            print("%-20s %-18s %+6d %+6d %+6d %+6d"
                  % (r.series[:20], r.variant, r.d_fp0, r.d_tp4, r.d_tp12,
                     r.d_tp16))
        print()

    # ---- the absolute ceiling -------------------------------------------
    # Hand the oracle everything the protocol forbids: the suite, the chunk,
    # the direction and a labelled threshold at exactly 20 false alarms inside
    # that suite.  Fire on every survivor above it.  This bounds explanation
    # (A) from above.
    scan = pd.read_csv(C.RESULTS / "ceiling_scan_raw_development.csv")
    scan = scan[~scan.degenerate & scan.deadline]
    print("=== ORACLE ceiling (violates every protocol rule; upper bound) ===")
    oracle_first = np.full(len(dev["risk"]), -1)
    picks = []
    for suite in C.SUITES:
        sub = scan[(scan.suite == suite) & (scan.chunk <= DEADLINE[suite])]
        if sub.empty:
            continue
        r = sub.sort_values("tp@fp20", ascending=False).iloc[0]
        s = series[r.series]
        m_suite = dev["suite"] == suite
        alive = m_suite & (dev["length"] > r.chunk)
        v = s[:, int(r.chunk)]
        vv = v[alive]
        risk = dev["risk"][alive]
        x = vv if r.direction == "high" else -vv
        neg = np.sort(x[np.isfinite(x) & ~risk])[::-1]
        thr = neg[min(20, len(neg) - 1)]
        sel = np.zeros(len(dev["risk"]), bool)
        xx = v if r.direction == "high" else -v
        sel[alive] = np.isfinite(xx[alive]) & (xx[alive] >= thr)
        oracle_first[sel & (oracle_first < 0)] = int(r.chunk)
        picks.append({"suite": suite, "chunk": int(r.chunk),
                      "series": r.series, "auc": float(r.auc),
                      "tp_at_fp20": int(r["tp@fp20"])})
        print("  %-16s q%-3d %-26s AUC %.3f  fires on %d episodes"
              % (suite, r.chunk, r.series, r.auc, int(sel.sum())))
    u = C.union(base, oracle_first)
    prof = lead_profile(u, dev)
    print("  ORACLE union with v8.3:  dFP0 %+d  dTP4 %+d  dTP12 %+d  dTP16 %+d"
          % (prof[0]["fp"] - bp[0]["fp"], prof[4]["tp"] - bp[4]["tp"],
             prof[12]["tp"] - bp[12]["tp"], prof[16]["tp"] - bp[16]["tp"]))

    # per suite, so a single suite cannot hide the rest
    print("\n  per suite at lead>=16 (development):")
    for suite in C.SUITES:
        m = dev["suite"] == suite
        n = int((m & dev["risk"]).sum())
        if not n:
            continue
        a = ((base >= 0) & ((dev["length"] - base) >= 16) & m & dev["risk"]).sum()
        b = ((u >= 0) & ((dev["length"] - u) >= 16) & m & dev["risk"]).sum()
        afp = ((base >= 0) & ((dev["length"] - base) >= 16) & m & ~dev["risk"]).sum()
        bfp = ((u >= 0) & ((dev["length"] - u) >= 16) & m & ~dev["risk"]).sum()
        print("    %-16s v8.3 %3d/%-4d %3dFP -> oracle %3d/%-4d %3dFP"
              % (suite, a, n, afp, b, n, bfp))

    (C.RESULTS / "capfree_cost.json").write_text(json.dumps({
        "v83_development": {str(l): [bp[l]["tp"], bp[l]["fp"]]
                            for l in C.LEADS},
        "oracle_picks": picks,
        "oracle_delta": {"d_fp0": int(prof[0]["fp"] - bp[0]["fp"]),
                         "d_tp4": int(prof[4]["tp"] - bp[4]["tp"]),
                         "d_tp12": int(prof[12]["tp"] - bp[12]["tp"]),
                         "d_tp16": int(prof[16]["tp"] - bp[16]["tp"])},
        "table": table.to_dict("records"),
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
