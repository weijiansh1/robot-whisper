#!/usr/bin/env python3
"""Regenerate every CSV in ../tables/ from the cached distance series.

    python3 scripts/make_tables.py

Reads only ../data/*.pkl — no zarr access, runs in about three minutes.
See common.py for the protocol; see ../README.md for what the numbers mean.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats as sps

from common import STOP_CLEAN, classes, detect, load, summarise

OUT = Path(__file__).resolve().parent.parent / "tables"
OUT.mkdir(exist_ok=True)

DIST = ["hellinger", "wjaccard", "tv", "js", "cos", "l2", "top4_jac", "top4_mass"]
CONTINUOUS = DIST[:6]


def write(name, rows, header):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}  ({len(rows)} rows)")


def main() -> None:
    D = load("distB")
    H, y, grp, T = D["H"], D["y"], D["g"], D["T"]
    kd = classes(y)
    n = len(y)
    ser = lambda ch, dn: [H[ch][dn][i] for i in range(n)]

    print("tables/")

    # 1. the headline: every channel x distance in the uncontaminated window
    rows = []
    for ch in ("st", "ac"):
        for dn in DIST:
            for W in (2, 4, 6, 8):
                for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                    d = detect(ser(ch, dn), y, grp, W, q)
                    s = summarise(d, y, kd)
                    rows.append([ch, dn, W, q, round(s["fa"], 4),
                                 round(s["recall"], 4), s["n_detected"],
                                 round(s["stagnation"], 3), round(s["other"], 3),
                                 round(s["loop"], 3)])
    write("clean_window_grid.csv", rows,
          ["channel", "distance", "W", "q", "fa_observed", "recall",
           "n_detected_of_216", "recall_stagnation", "recall_other", "recall_loop"])

    # 2. the methodological centrepiece: detection tracks opportunity, not signal
    rows = []
    for ch, dn, W, q in (("ac", "top4_mass", 8, 0.20), ("ac", "top4_mass", 4, 0.30),
                         ("st", "l2", 6, 0.25), ("ac", "l2", 6, 0.25),
                         ("st", "hellinger", 6, 0.25)):
        for stop in (35, 40, 52):
            d = detect(ser(ch, dn), y, grp, W, q, stop=stop)
            s = summarise(d, y, kd)
            opp_f = float(np.median([max(0, min(T[i], stop) - W)
                                     for i in range(n) if y[i] == 1]))
            opp_s = float(np.median([max(0, min(T[i], stop) - W)
                                     for i in range(n) if y[i] == 0]))
            rows.append([ch, dn, W, q, stop, round(s["fa"], 4), round(s["recall"], 4),
                         round(s["loop"], 3), opp_f, opp_s,
                         round(opp_f / opp_s, 3)])
    write("window_inflation.csv", rows,
          ["channel", "distance", "W", "q", "stop", "fa_observed", "recall",
           "recall_loop", "chances_failure_median", "chances_success_median",
           "chance_ratio"])

    # 3. does the score respond to the phenomenon at all?
    rows = []
    for ch in ("st", "ac"):
        for dn in DIST:
            allv = np.concatenate([H[ch][dn][i][:STOP_CLEAN] for i in range(n)])
            v30 = np.array([H[ch][dn][i][:30][-6:].mean() for i in range(n)])
            ratios = np.array([H[ch][dn][i][24:30].mean() / H[ch][dn][i][:10].mean()
                               for i in range(n) if len(H[ch][dn][i]) >= 30
                               and H[ch][dn][i][:10].mean() > 0])
            m = y[:len(ratios)] == 1
            rows.append([ch, dn, round(float(np.median(allv)), 5),
                         round(float(np.median(v30[y == 0])), 5),
                         round(float(np.median(v30[y == 1])), 5),
                         round(float(np.median(v30[y == 1]) / np.median(v30[y == 0])), 4),
                         round(float(np.median(ratios[m])), 4),
                         round(float(np.median(ratios[~m])), 4)])
    write("distance_response.csv", rows,
          ["channel", "distance", "median_value", "median_t30_success",
           "median_t30_failure", "failure_over_success",
           "self_collapse_failure", "self_collapse_success"])

    # 4. distances measure the same thing (except the two top4 ones)
    V = {dn: np.concatenate([H["st"][dn][i][:40] for i in range(n)]) for dn in DIST}
    rows = [[a, b, round(float(sps.spearmanr(V[a], V[b]).statistic), 4)]
            for i, a in enumerate(DIST) for b in DIST[i + 1:]]
    write("distance_correlations.csv", rows, ["distance_a", "distance_b", "spearman"])

    # 5. one chunk carries almost all of it, and only from step 29
    Dt = load("tokB")
    rows = []
    for ch, dn in (("ac", "wjaccard"), ("st", "hellinger")):
        for t in range(24, 38):
            v = np.array([H[ch][dn][i][t - 1] if len(H[ch][dn][i]) >= t else np.nan
                          for i in range(n)])
            acc = []
            for q in (0.05, 0.10, 0.15, 0.20, 0.30):
                d = np.zeros(n, bool)
                for g in set(grp):
                    o = np.setdiff1d(np.arange(n), np.where(grp == g)[0])
                    ref = v[o][(y[o] == 0) & np.isfinite(v[o])]
                    idx = np.where(grp == g)[0]
                    d[idx] = np.isfinite(v[idx]) & (v[idx] < np.quantile(ref, q))
                acc.append((d == (y == 1)).mean())
            rows.append([ch, dn, t, round(float(max(acc)), 4)])
    write("single_chunk_onset.csv", rows,
          ["channel", "distance", "step", "best_accuracy_512"])

    # 6. ten action tokens are one measurement, not ten
    X = Dt["hel"]
    rows = []
    for k in range(11):
        best = None
        for W in (2, 4, 6, 8):
            for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                d = detect([x[:, k] for x in X], y, grp, W, q)
                s = summarise(d, y, kd)
                if s["fa"] <= 0.08 and (best is None or s["recall"] > best[0]):
                    best = (s["recall"], s, W, q)
        rows.append([f"token{k}" + (" (st_deep)" if k == 0 else ""), best[2], best[3],
                     round(best[1]["fa"], 4), round(best[0], 4),
                     round(best[1]["stagnation"], 3), round(best[1]["other"], 3),
                     round(best[1]["loop"], 3)])
    ops = {"mean_1_10 (ac_deep)": lambda x: x[:, 1:].mean(1),
           "mean_all_11": lambda x: x.mean(1),
           "near_1_3": lambda x: x[:, 1:4].mean(1),
           "far_8_10": lambda x: x[:, 8:11].mean(1),
           "min_1_10": lambda x: x[:, 1:].min(1),
           "max_1_10": lambda x: x[:, 1:].max(1),
           "std_1_10": lambda x: x[:, 1:].std(1),
           "range_1_10": lambda x: x[:, 1:].max(1) - x[:, 1:].min(1),
           "far_over_near": lambda x: x[:, 8:11].mean(1) / np.maximum(x[:, 1:4].mean(1), 1e-9),
           "far_minus_near": lambda x: x[:, 8:11].mean(1) - x[:, 1:4].mean(1),
           "token_slope": lambda x: np.polyfit(np.arange(10), x[:, 1:].T, 1)[0]}
    for nm, f in ops.items():
        S = [f(x) for x in X]
        best = None
        for W in (2, 4, 6, 8):
            for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
                d = detect(S, y, grp, W, q)
                s = summarise(d, y, kd)
                if s["fa"] <= 0.08 and (best is None or s["recall"] > best[0]):
                    best = (s["recall"], s, W, q)
        rows.append([nm, best[2], best[3], round(best[1]["fa"], 4), round(best[0], 4),
                     round(best[1]["stagnation"], 3), round(best[1]["other"], 3),
                     round(best[1]["loop"], 3)])
    write("token_axis.csv", rows,
          ["organization", "W", "q", "fa_observed", "recall",
           "recall_stagnation", "recall_other", "recall_loop"])

    V = [np.concatenate([x[:STOP_CLEAN, k] for x in X]) for k in range(11)]
    rows = [[a, b, round(float(sps.spearmanr(V[a], V[b]).statistic), 4)]
            for a in range(11) for b in range(a + 1, 11)]
    write("token_correlations.csv", rows, ["token_a", "token_b", "spearman"])

    print("\ndone.  See ../FINDINGS.md for what to quote and ../CORRECTIONS.md "
          "for what was retracted.")


if __name__ == "__main__":
    main()
