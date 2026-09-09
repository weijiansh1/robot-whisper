"""v8.1: a threshold that relaxes monotonically with chunk index.

v8 uses one constant threshold per head, an order statistic of the pooled
unlabeled development scores.  That is time-invariant, but the decision problem
is not: the survival prior rises from roughly 0.05 to 0.44 across the window,
so the same evidence is worth more late than early.  A threshold that relaxes
linearly with the chunk index encodes that.

Three adaptive alternatives were tested first and all failed, for a reason
worth recording: **a threshold estimated from the episode's own recent history
adapts to the anomaly itself.**

  * running median + MAD of the episode's own past: 935/1358 at 963 FP (7.6x)
  * expanding self-baseline instead of fixed q1..q4: 920/1358 at 117 FP
  * per-chunk z-normalisation against the reference: 933/1358 at 596 FP (4.7x)

v7's fixed q1..q4 baseline works because it is a *reference regime* - "compared
with how this episode opened" - not a *current* one.  The slope here is a
deterministic function of the chunk index only, so it cannot be dragged along
by the anomaly.

Selection: slope chosen on `development_main` alone, by maximising development
TP subject to a declared development false-alarm budget.  `external_8b` and
`legacy_main16x32` are scored once per budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE.parent / "results"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-v7-0905" / "method"))

import evaluate_full_corpus as V  # noqa: E402

COHORTS = ("development_main", "external_8b", "legacy_main16x32")
SLOPES = np.round(np.arange(-0.0100, 0.0011, 0.0005), 5)
# Declared before looking: development false-alarm budgets as multiples of
# v8's own development count (44).
BUDGETS = (44, 55, 66, 88)
LEADS = (0, 4, 8, 12)


def alarms_with_slope(series, threshold, direction, slope, confirm=V.CONFIRM,
                      earliest=V.WIDTH):
    """Threshold moves linearly with the chunk index; `slope < 0` relaxes it.

    The sign flip for `low` heads keeps "negative slope = more permissive"
    true for both directions.
    """
    q = np.arange(series.shape[1])[None, :]
    moving = threshold + slope * q * (1 if direction == "high" else -1)
    hit = (series >= moving) if direction == "high" else (series <= moving)
    hit &= np.isfinite(series)
    hit[:, :earliest] = False
    held = np.zeros_like(hit)
    run = np.zeros(hit.shape[0], dtype=int)
    for step in range(hit.shape[1]):
        run = np.where(hit[:, step], run + 1, 0)
        held[:, step] = run >= confirm
    return np.where(held.any(axis=1), held.argmax(axis=1), -1)


def main() -> None:
    data = {n: V.load_cohort(n) for n in COHORTS}
    heads = {n: V.heads_from_flow_speed(d["speed"]) for n, d in data.items()}

    thresholds = {}
    for name, series in heads["development_main"].items():
        pool = series[np.isfinite(series)]
        level = V.QUANTILE if V.DIRECTION[name] == "high" else 1 - V.QUANTILE
        thresholds[name] = float(np.quantile(pool, level, method="lower"))

    rows = []
    for slope in SLOPES:
        record = {"slope": float(slope)}
        for cohort, d in data.items():
            firsts = [alarms_with_slope(heads[cohort][h], thresholds[h],
                                        V.DIRECTION[h], slope) for h in heads[cohort]]
            union = V.union(d["v7"], *firsts)
            for lead in LEADS:
                s = V.score(union, d["risk"], d["length"], lead)
                record[f"{cohort}_tp{lead}"] = s["tp"]
                record[f"{cohort}_fp{lead}"] = s["fp"]
            record[f"{cohort}_lead"] = V.score(union, d["risk"], d["length"])["median_lead"]
        rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "slope_frontier.csv", index=False)

    print("v8 (slope = 0): 全量 932/1358, 126 FP")
    print("\n===== development 前沿（选择只看这一列）=====")
    print("%8s | %-16s %-16s" % ("slope", "lead>=0", "lead>=4"))
    for _, r in table.iterrows():
        print("%8.4f | %4d/487 %4d FP  %4d/487 %4d FP"
              % (r.slope, r.development_main_tp0, r.development_main_fp0,
                 r.development_main_tp4, r.development_main_fp4))

    print("\n===== 按预先声明的 development FP 预算选 slope，"
          "external + legacy 各评一次 =====")
    print("%6s %8s | %-18s | %-18s | %-18s | %s"
          % ("dev预算", "slope", "development", "external", "legacy", "全量"))
    chosen = []
    for budget in BUDGETS:
        ok = table[table.development_main_fp4 <= budget]
        if not len(ok):
            print("%6d  无可行 slope" % budget)
            continue
        r = ok.loc[ok.development_main_tp4.idxmax()]
        tot_tp = int(sum(r[f"{c}_tp4"] for c in COHORTS))
        tot_fp = int(sum(r[f"{c}_fp4"] for c in COHORTS))
        print("%6d %8.4f | %3d/487 %3d FP | %3d/564 %3d FP | %3d/307 %3d FP | "
              "%4d/1358 %4d FP  精度 %.3f"
              % (budget, r.slope, r.development_main_tp4, r.development_main_fp4,
                 r.external_8b_tp4, r.external_8b_fp4,
                 r.legacy_main16x32_tp4, r.legacy_main16x32_fp4,
                 tot_tp, tot_fp, tot_tp / (tot_tp + tot_fp)))
        chosen.append({"budget": budget, "slope": float(r.slope),
                       "total_tp": tot_tp, "total_fp": tot_fp,
                       "external": [int(r.external_8b_tp4), int(r.external_8b_fp4)],
                       "legacy": [int(r.legacy_main16x32_tp4),
                                  int(r.legacy_main16x32_fp4)]})

    print("\n===== 选定 slope 后的提前量代价（全量合计）=====")
    for entry in chosen:
        r = table[table.slope == entry["slope"]].iloc[0]
        cells = []
        for lead in LEADS:
            tp = int(sum(r[f"{c}_tp{lead}"] for c in COHORTS))
            fp = int(sum(r[f"{c}_fp{lead}"] for c in COHORTS))
            cells.append("%4d/%-4d" % (tp, fp))
        print("  slope %+.4f  %s" % (entry["slope"],
              "  ".join(f"lead>={b}: {c}" for b, c in zip(LEADS, cells))))

    (OUT / "slope_selection.json").write_text(json.dumps(
        {"thresholds": thresholds, "budgets": list(BUDGETS),
         "chosen": chosen}, indent=2))


if __name__ == "__main__":
    main()
