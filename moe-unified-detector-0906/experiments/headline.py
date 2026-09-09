#!/usr/bin/env python3
"""The reported headline numbers, derived from the frozen operating points.

Three questions are answered here and nowhere else:

  1. what does the detector's recall/FPR curve look like on each cohort;
  2. where, if anywhere, does it beat the information-free survival baseline
     ("the episode is still running at chunk q"), which recalls 100% of the
     risks by construction because risk *is* "did not finish before the cap";
  3. what is the highest in-window recall that is still an improvement on that
     baseline, and what is its task-matched lift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import protocol as P

COHORTS = ["development_main", "external_8b", "legacy_main16x32", "corpus"]


def envelope(sub: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Monotone (fpr -> best recall) envelope of a development-selected front."""
    x, y = f"{cohort}.iw_fpr", f"{cohort}.iw_recall"
    front = P.pareto_front(
        sub.dropna(subset=[x, y]),
        "development_main.iw_fpr", "development_main.iw_recall",
    ).sort_values(x)
    front = front.reset_index(drop=True)
    front[y] = np.maximum.accumulate(front[y].to_numpy())
    return front


def recall_at(front: pd.DataFrame, cohort: str, fpr: float) -> float:
    x, y = f"{cohort}.iw_fpr", f"{cohort}.iw_recall"
    ok = front[front[x] <= fpr + 1e-12]
    return float(ok[y].max()) if len(ok) else 0.0


def main() -> None:
    table = pd.read_csv(P.RESULTS / "operating_points_pooled.csv.gz")
    surv = table[table["detector"] == "survival_prior"]
    rows = []
    grid = np.unique(np.concatenate([
        np.geomspace(1e-4, 0.5, 60), [0.002, 0.005, 0.01, 0.02, 0.05, 0.0678, 0.1, 0.2]
    ]))
    for name, sub in table.groupby("detector"):
        if name in ("survival_prior", "length_leak_NOT_a_baseline"):
            continue
        for cohort in COHORTS:
            if f"{cohort}.iw_fpr" not in sub.columns:
                continue
            if sub[f"{cohort}.iw_fpr"].isna().all():
                continue
            front = envelope(sub, cohort)
            s = surv.dropna(subset=[f"{cohort}.iw_fpr"]).sort_values(f"{cohort}.iw_fpr")
            for fpr in grid:
                det_r = recall_at(front, cohort, fpr)
                ok = s[s[f"{cohort}.iw_fpr"] <= fpr + 1e-12]
                base_r = float(ok[f"{cohort}.iw_recall"].max()) if len(ok) else 0.0
                rows.append({
                    "detector": name, "cohort": cohort, "fpr": float(fpr),
                    "iw_recall": det_r, "survival_iw_recall": base_r,
                    "excess": det_r - base_r,
                })
    curve = pd.DataFrame(rows)
    curve.to_csv(P.RESULTS / "recall_vs_fpr_curve.csv", index=False)

    summary: dict = {"survival_baseline": {}, "detectors": {}}
    for cohort in COHORTS:
        s = surv.dropna(subset=[f"{cohort}.iw_fpr"])
        full = s[s[f"{cohort}.iw_recall"] >= 0.999]
        summary["survival_baseline"][cohort] = {
            "recall_1.0_at_fpr": float(full[f"{cohort}.iw_fpr"].min()) if len(full) else None,
            "recall_at_fpr_0.02": float(
                s[s[f"{cohort}.iw_fpr"] <= 0.02][f"{cohort}.iw_recall"].max()
            ),
            "note": "not a detector; task-matched lift is 1.0 by construction",
        }
    for name, sub in curve.groupby("detector"):
        entry = {}
        for cohort in COHORTS:
            c = sub[sub["cohort"] == cohort]
            if c.empty:
                continue
            beat = c[c["excess"] > 0]
            entry[cohort] = {
                "recall_at_fpr": {
                    f"{f:.4g}": float(
                        c.loc[(c["fpr"] - f).abs().idxmin(), "iw_recall"]
                    )
                    for f in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
                },
                "max_recall_while_beating_survival": float(beat["iw_recall"].max())
                if len(beat) else 0.0,
                "fpr_where_survival_overtakes": float(
                    c.loc[c["excess"] < 0, "fpr"].min()
                ) if (c["excess"] < 0).any() else None,
                "reaches_0.80": bool((c["iw_recall"] >= 0.80).any()),
                "fpr_at_0.80": float(c.loc[c["iw_recall"] >= 0.80, "fpr"].min())
                if (c["iw_recall"] >= 0.80).any() else None,
            }
        summary["detectors"][name] = entry
    P.write_json(P.RESULTS / "headline_summary.json", summary)

    # per-suite breakdown at three named operating points of the headline arms
    budget = pd.read_csv(P.RESULTS / "budget_table.csv")
    keep = budget[(budget["scope"] == "any_rule")]
    cols = ["detector", "budget", "rule", "k", "drift"]
    for cohort in P.COHORTS:
        for s in P.SUITES:
            for key in ("iw_recall", "iw_tp", "iw_fp"):
                col = f"{cohort}.{key}_{s}"
                if col in table.columns:
                    cols.append(col)
    have = [c for c in cols if c in keep.columns]
    sub = keep[have]
    sub.to_csv(P.RESULTS / "per_suite_breakdown.csv", index=False)
    print("wrote recall_vs_fpr_curve.csv, headline_summary.json, per_suite_breakdown.csv")


if __name__ == "__main__":
    main()
