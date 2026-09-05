#!/usr/bin/env python
"""E1-B analysis: does the routing flow contract, and does it contract less
before a loop?

Reads e1b_inject_probe records.json.  Statistical unit is the TRUNK (directions
and epsilons within a snapshot are conditional replicates), per the program's
statistics rules; loop-vs-control differences use the hierarchical bootstrap.

Guards enforced before any number is reported:
  * every record must show zero pre-injection leakage (D_pre_max == 0);
  * a cell is only compared when both arms have >= 2 trunks.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np

import control_metrics as cm


def load(path):
    recs = json.loads(pathlib.Path(path).read_text())
    bad = [r for r in recs if r.get("D_pre_max", 0) > 0]
    if bad:
        raise SystemExit(f"{len(bad)} record(s) leak before the injection step; "
                         "the mechanism is not sound -- refusing to analyse")
    return recs


def cell_table(recs, metric):
    """(arm, q_offset, f_star, eps) -> {trunk_uid: [values]}"""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in recs:
        if metric not in r or not np.isfinite(r[metric]):
            continue
        key = (r["arm"], r["q_offset"], r["f_star"], r["eps"])
        out[key][r["trunk_uid"]].append(float(r[metric]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", default="lambda,D_peak,D_terminal,decay_ratio")
    args = ap.parse_args()

    recs = load(args.records)
    for r in recs:                      # derived: how much of the kick survives
        if r.get("D_peak", 0) > 0:
            r["decay_ratio"] = r["D_terminal"] / r["D_peak"]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = args.metrics.split(",")
    summary = {"n_records": len(recs), "pre_injection_leakage": 0, "cells": []}

    print(f"records {len(recs)}  trunks {len({r['trunk_uid'] for r in recs})}  "
          f"arms {sorted({r['arm'] for r in recs})}")
    for metric in metrics:
        table = cell_table(recs, metric)
        print(f"\n=== {metric} (median over directions, then trunk bootstrap) ===")
        print("%-6s %5s %5s  %-22s %-22s %s"
              % ("f*", "eps", "q", "loop [95% CI]", "control [95% CI]",
                 "loop-control [95% CI]"))
        keys = sorted({(k[2], k[3], k[1]) for k in table})
        for fstar, eps, q in keys:
            arms = {}
            for arm in ("loop", "control"):
                d = table.get((arm, q, fstar, eps))
                if d and len(d) >= 2:
                    arms[arm] = cm.trunk_bootstrap(d, np.median, 4000, seed=1)
                elif d:
                    v = float(np.median([np.median(x) for x in d.values()]))
                    arms[arm] = {"estimate": v, "ci_lo": float("nan"),
                                 "ci_hi": float("nan"), "n_trunks": len(d)}
            if "loop" not in arms:
                continue
            row = {"metric": metric, "f_star": fstar, "eps": eps, "q_offset": q,
                   "loop": arms.get("loop"), "control": arms.get("control")}
            diff = ""
            if "control" in arms:
                lt = table[("loop", q, fstar, eps)]
                ct = table[("control", q, fstar, eps)]
                if len(lt) >= 2 and len(ct) >= 2:
                    rng = np.random.default_rng(2)
                    lv = np.array([np.median(v) for v in lt.values()])
                    cv = np.array([np.median(v) for v in ct.values()])
                    b = np.array([np.median(rng.choice(lv, len(lv)))
                                  - np.median(rng.choice(cv, len(cv)))
                                  for _ in range(4000)])
                    d0 = float(np.median(lv) - np.median(cv))
                    lo, hi = np.percentile(b, [2.5, 97.5])
                    row["diff"] = {"estimate": d0, "ci_lo": float(lo),
                                   "ci_hi": float(hi),
                                   "n_loop": len(lv), "n_control": len(cv)}
                    star = "*" if (lo > 0 or hi < 0) else " "
                    diff = f"{d0:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
                else:
                    row["diff"] = {"status": "insufficient_trunks"}
                    diff = "insufficient trunks"
            fmt = lambda a: (f"{a['estimate']:+.4f} [{a['ci_lo']:+.4f},"
                             f"{a['ci_hi']:+.4f}] n={a['n_trunks']}"
                             if a and np.isfinite(a["ci_lo"])
                             else (f"{a['estimate']:+.4f} n={a['n_trunks']}"
                                   if a else "-"))
            print("%-6d %5.2f %+5d  %-22s %-22s %s"
                  % (fstar, eps, q, fmt(arms.get("loop")),
                     fmt(arms.get("control")), diff))
            summary["cells"].append(row)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\nwrote {out/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
