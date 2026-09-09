"""Step 3: the ceiling scan.  DEVELOPMENT ONLY, labels used only to bound.

Two things are measured, and the second is the one that matters.

1. `auc`  -- within-task, survivor-only Mann-Whitney AUC at a fixed chunk.
   Necessary but, as the whole history of this problem shows, far from
   sufficient.

2. `tp_at_fp` -- the *operational* conversion: among survivors at chunk q in
   one suite, take the threshold that admits exactly B successes and count the
   risks it admits.  This is what a hard threshold actually buys, and it is
   what an AUC does not tell you.

Both are reported per suite at the chunks that matter for a long lead
(goal q14, object q12, spatial q6, long q36).  For calibration the script also
prints the Gaussian-equal-variance prediction: what TP@FP a score with the
observed AUC *would* deliver if the two classes were normal.  Where the
observed conversion is far below the Gaussian prediction, the signal lives in
the bulk and not in the tail, and no threshold can extract it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import bank
import common as C

CHUNKS = (6, 8, 10, 12, 14, 16, 18, 20)
DEADLINE = {"libero_goal": 14, "libero_long": 36,
            "libero_object": 12, "libero_spatial": 6}
BUDGETS = (10, 20, 40)


def task_auc(values, risk, task):
    """Within-task Mann-Whitney AUC pooled with n+ * n- weights."""
    num = den = 0.0
    for t in np.unique(task):
        m = (task == t) & np.isfinite(values)
        y, x = risk[m], values[m]
        npos, nneg = int(y.sum()), int((~y).sum())
        if npos == 0 or nneg == 0:
            continue
        r = stats.rankdata(x)
        auc = (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
        num += auc * npos * nneg
        den += npos * nneg
    return num / den if den else np.nan


def tp_at_fp(values, risk, budget, direction):
    """Threshold admitting exactly `budget` successes -> how many risks?

    Uses `>=`, so the whole tie group at the threshold is admitted; the
    reported FP is the realised count, which can exceed the budget on a tie.
    """
    m = np.isfinite(values)
    x = values[m] if direction == "high" else -values[m]
    y = risk[m]
    neg = np.sort(x[~y])[::-1]
    if len(neg) <= budget:
        return int(y.sum()), int((~y).sum())
    thr = neg[budget]           # the (budget+1)-th highest success
    sel = x >= thr
    return int((sel & y).sum()), int((sel & ~y).sum())


def gaussian_tp(auc, n_pos, n_neg, budget):
    """TP a Gaussian equal-variance score with this AUC would deliver."""
    if not np.isfinite(auc) or auc <= 0.5:
        return 0.0
    d = np.sqrt(2) * stats.norm.ppf(auc)
    z = stats.norm.isf(budget / max(n_neg, 1))
    return float(n_pos * stats.norm.sf(z - d))


def main() -> None:
    dev = C.load_cohort("development_main")
    mob = C.mobility("development_main")
    series = bank.from_cached(mob, dev["speed"])
    series |= bank.with_open_ratio(series)
    series = bank.mask_after_end(series, dev["length"])
    print("bank: %d series" % len(series))

    rows = []
    for suite in C.SUITES:
        m_suite = dev["suite"] == suite
        if not m_suite.any():
            continue
        for q in CHUNKS:
            if q > C.CAP[suite]:
                continue
            alive = m_suite & (dev["length"] > q)
            if alive.sum() < 50:
                continue
            risk = dev["risk"][alive]
            task = dev["task"][alive]
            if risk.sum() == 0 or (~risk).sum() == 0:
                continue
            for name, s in series.items():
                v = s[alive, q]
                if np.isfinite(v).sum() < 50:
                    continue
                a = task_auc(v, risk, task)
                direction = "high" if (np.isfinite(a) and a >= 0.5) else "low"
                a_dir = a if direction == "high" else 1 - a
                row = {"suite": suite, "chunk": q, "series": name,
                       "auc": a_dir, "direction": direction,
                       "n_risk": int(risk.sum()), "n_safe": int((~risk).sum()),
                       "deadline": q <= DEADLINE[suite]}
                for b in BUDGETS:
                    tp, fp = tp_at_fp(v, risk, b, direction)
                    row[f"tp@fp{b}"] = tp
                    row[f"fp@fp{b}"] = fp
                    row[f"gauss_tp@fp{b}"] = gaussian_tp(
                        a_dir, int(risk.sum()), int((~risk).sum()), b)
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(C.RESULTS / "ceiling_scan_development.csv", index=False)
    print("cells: %d" % len(table))

    print("\n=== best conversion at each suite's lead>=16 deadline"
          " (development, oracle threshold) ===")
    for suite in C.SUITES:
        q = DEADLINE[suite]
        sub = table[(table.suite == suite) & (table.chunk == q)]
        if sub.empty:
            print("  %-16s chunk q%d not in the scan grid" % (suite, q))
            continue
        best = sub.sort_values("tp@fp20", ascending=False).head(4)
        print("  %-16s q%-3d  (%d risk / %d safe alive)"
              % (suite, q, best.iloc[0].n_risk, best.iloc[0].n_safe))
        for _, r in best.iterrows():
            print("      %-24s AUC %.3f | oracle %3d TP @ %2d FP"
                  "  (Gaussian would give %.0f)"
                  % (r.series, r.auc, r["tp@fp20"], r["fp@fp20"],
                     r["gauss_tp@fp20"]))

    print("\n=== highest AUC anywhere in the early window, and what it buys ===")
    early = table[table.deadline]
    for _, r in early.sort_values("auc", ascending=False).head(12).iterrows():
        print("  %-16s q%-3d %-24s AUC %.3f -> %3d TP @ %2d FP  (Gauss %.0f)"
              % (r.suite, r.chunk, r.series, r.auc, r["tp@fp20"],
                 r["fp@fp20"], r["gauss_tp@fp20"]))

    print("\n=== does AUC convert?  observed vs Gaussian, in the early window ===")
    e = early[early.auc >= 0.65].copy()
    e["ratio"] = e["tp@fp20"] / np.maximum(e["gauss_tp@fp20"], 1e-9)
    print("  cells with AUC>=0.65 in the early window: %d" % len(e))
    if len(e):
        print("  observed TP@FP20 / Gaussian prediction:"
              " median %.2f, p10 %.2f, p90 %.2f"
              % (e.ratio.median(), e.ratio.quantile(.1), e.ratio.quantile(.9)))
        for suite in C.SUITES:
            s = e[e.suite == suite]
            if len(s):
                print("    %-16s n=%3d  median ratio %.2f  best observed"
                      " TP@FP20 = %d (AUC %.3f)"
                      % (suite, len(s), s.ratio.median(), s["tp@fp20"].max(),
                         s.loc[s["tp@fp20"].idxmax(), "auc"]))

    (C.RESULTS / "ceiling.json").write_text(json.dumps({
        "n_cells": len(table), "n_series": len(series),
        "deadline": DEADLINE,
        "best_at_deadline": {
            s: (table[(table.suite == s) & (table.chunk == DEADLINE[s])]
                .sort_values("tp@fp20", ascending=False)
                .head(1)[["series", "auc", "tp@fp20", "fp@fp20",
                          "gauss_tp@fp20", "n_risk", "n_safe"]]
                .to_dict("records"))
            for s in C.SUITES},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
