"""Step 4: the ceiling scan again, now including the seventeen new quantities.

Same estimator as step 3 (within-task survivor-only AUC, plus the operational
oracle conversion TP@FP), applied to a bank of 77 series on DEVELOPMENT ONLY.

The null is a task-stratified permutation of the outcome, run at the full
sweep width (128 series) and reduced to a floor *per (suite, chunk) cell*, so
it is matched both to the number of comparisons and to the cell's own n.  A
circular shift is provided for the detector sweep, where the head reads the
series over time; it is the wrong null for a fixed-chunk AUC, which reads one
number per episode -- and for a quantity that barely varies within an episode
the shift is near the identity, which is why it returns a spurious 0.36 floor.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bank
import common as C
from step3_ceiling import DEADLINE, gaussian_tp, task_auc, tp_at_fp

CHUNKS = (6, 8, 10, 12, 14, 16, 18)
SEED = 20260906
N_NULL = 3


def build_bank(cohort: str, d: dict) -> dict[str, np.ndarray]:
    raw = np.load(C.RESULTS / f"{cohort}_rawq.npz")
    series = bank.from_cached(C.mobility(cohort), d["speed"])
    series |= bank.from_raw(raw)
    series |= bank.with_open_ratio(series)
    return bank.mask_after_end(series, d["length"])


def circular_shift(series: np.ndarray, length: np.ndarray, rng) -> np.ndarray:
    """Preserve each episode's length, marginal and autocorrelation; destroy
    only its alignment with the chunk index.  Used for the *detector* null,
    where the head reads the series over time."""
    out = np.full_like(series, np.nan)
    off = rng.integers(1, np.maximum(length, 2))
    for i in range(series.shape[0]):
        n = int(length[i])
        if n < 2:
            continue
        out[i, :n] = np.roll(series[i, :n], int(off[i]))
    return out


def permuted_risk(d: dict, rng) -> np.ndarray:
    """Null for the *fixed-chunk* scan: permute the outcome within task.

    A circular shift is the wrong null here twice over.  A fixed-chunk AUC
    reads one number per episode, so there is no temporal structure for a
    shift to destroy; and for a quantity that is nearly constant within an
    episode the shift is close to the identity, which is why it returned a
    0.36 'floor'.  Permuting the label within task destroys exactly the one
    association being tested and nothing else.  Length cannot be conditioned
    on: risk is *defined* as running to the cap, so a length-stratified
    permutation would be the identity map.
    """
    out = d["risk"].copy()
    for t in np.unique(d["task"]):
        m = np.flatnonzero(d["task"] == t)
        out[m] = out[rng.permutation(m)]
    return out


def scan(series: dict, d: dict, chunks=CHUNKS, risk_override=None,
         max_tie_factor: float = 2.0) -> pd.DataFrame:
    rows = []
    risk_all = d["risk"] if risk_override is None else risk_override
    for suite in C.SUITES:
        m_suite = d["suite"] == suite
        if not m_suite.any():
            continue
        for q in chunks:
            if q > C.CAP[suite]:
                continue
            alive = m_suite & (d["length"] > q)
            risk, task = risk_all[alive], d["task"][alive]
            if risk.sum() < 5 or (~risk).sum() < 50:
                continue
            for name, s in series.items():
                v = s[alive, q]
                if np.isfinite(v).sum() < 50:
                    continue
                a = task_auc(v, risk, task)
                if not np.isfinite(a):
                    continue
                direction = "high" if a >= 0.5 else "low"
                a_dir = a if direction == "high" else 1 - a
                row = {"suite": suite, "chunk": q, "series": name,
                       "auc": a_dir, "direction": direction,
                       "n_risk": int(risk.sum()), "n_safe": int((~risk).sum()),
                       "deadline": q <= DEADLINE[suite]}
                degenerate = False
                for b in (10, 20, 40):
                    tp, fp = tp_at_fp(v, risk, b, direction)
                    # a tie group far larger than the budget means the series
                    # is effectively constant here; `>=` then admits everything
                    # and the "oracle" is meaningless, not good.
                    if fp > max_tie_factor * b:
                        tp = fp = 0
                        degenerate = True
                    row[f"tp@fp{b}"] = tp
                    row[f"fp@fp{b}"] = fp
                row["degenerate"] = degenerate
                row["gauss_tp@fp20"] = gaussian_tp(a_dir, int(risk.sum()),
                                                   int((~risk).sum()), 20)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    dev = C.load_cohort("development_main")
    series = build_bank("development_main", dev)
    print("bank: %d series, %d chunks, %d suites"
          % (len(series), len(CHUNKS), len(C.SUITES)))

    table = scan(series, dev)
    table.to_csv(C.RESULTS / "ceiling_scan_raw_development.csv", index=False)
    print("cells: %d" % len(table))

    # ---- null, matched to the sweep AND to the cell size ----------------
    # A single pooled floor is wrong here: a (suite, chunk) cell with 5 risks
    # reaches recall 0.8 at FP20 by chance while one with 102 risks cannot, so
    # a pooled max is set by the smallest cell and would veto everything.  The
    # floor is therefore computed *per cell*, as the max over the same 128
    # series and N_NULL permutations -- identical sweep width, identical n.
    rng = np.random.default_rng(SEED)
    null_rows = []
    for k in range(N_NULL):
        nt = scan(series, dev, risk_override=permuted_risk(dev, rng))
        nt["absauc"] = (nt.auc - 0.5).abs()
        null_rows.append(nt)
    nullcat = pd.concat(null_rows)
    floor = (nullcat.groupby(["suite", "chunk"])
             .agg(floor_auc=("absauc", "max"),
                  floor_tp=("tp@fp20", "max")).reset_index())
    floor.to_csv(C.RESULTS / "null_floor_by_cell.csv", index=False)
    table = table.merge(floor, on=["suite", "chunk"], how="left")
    table["auc_margin"] = (table.auc - 0.5).abs() - table.floor_auc
    table["tp_margin"] = table["tp@fp20"] - table.floor_tp
    table.to_csv(C.RESULTS / "ceiling_scan_raw_development.csv", index=False)
    print("null: %d draws x %d series, floor computed per (suite, chunk) cell"
          % (N_NULL, len(series)))
    for _, r in floor.iterrows():
        print("   %-16s q%-3d  floor |AUC-.5| %.3f   floor TP@FP20 %3d"
              % (r.suite, r.chunk, r.floor_auc, r.floor_tp))
    null_floor = float(floor.floor_auc.max())

    print("\n=== NEW quantities only, in the early window, ranked by what they"
          " actually buy ===")
    new = table[table.deadline & ~table.degenerate & table.series.str.startswith(
        tuple(n for n in bank.RAW_NAMES))]
    for _, r in new.sort_values("tp@fp20", ascending=False).head(15).iterrows():
        print("  %-16s q%-3d %-28s AUC %.3f (margin %+.3f) -> %3d/%-4d TP @"
              " %2d FP (null floor %d, margin %+d)"
              % (r.suite, r.chunk, r.series, r.auc, r.auc_margin,
                 r["tp@fp20"], r.n_risk, r["fp@fp20"], r.floor_tp,
                 r.tp_margin))

    print("\n=== everything, at each suite's lead>=16 deadline ===")
    for suite in C.SUITES:
        q = DEADLINE[suite]
        sub = table[(table.suite == suite) & (table.chunk == q)
                    & ~table.degenerate]
        if sub.empty:
            print("  %-16s q%d outside the grid" % (suite, q))
            continue
        best = sub.sort_values("tp@fp20", ascending=False).head(5)
        print("  %-16s q%-3d  %d risk / %d safe alive"
              % (suite, q, best.iloc[0].n_risk, best.iloc[0].n_safe))
        for _, r in best.iterrows():
            print("      %-28s AUC %.3f (margin %+.3f) | %3d TP @ %2d FP"
                  " | null floor %d | Gauss %.0f"
                  % (r.series, r.auc, r.auc_margin, r["tp@fp20"],
                     r["fp@fp20"], r.floor_tp, r["gauss_tp@fp20"]))

    print("\n=== highest AUC anywhere in the early window ===")
    for _, r in table[table.deadline & ~table.degenerate].sort_values(
            "auc", ascending=False).head(12).iterrows():
        print("  %-16s q%-3d %-28s AUC %.3f -> %3d TP @ %2d FP"
              % (r.suite, r.chunk, r.series, r.auc, r["tp@fp20"], r["fp@fp20"]))

    (C.RESULTS / "ceiling_raw.json").write_text(json.dumps({
        "n_series": len(series), "n_cells": len(table),
        "null_floor_auc_max": null_floor,
        "null_floor_by_cell": floor.to_dict("records"),
        "null_kind": "task-stratified label permutation among survivors",
        "best_at_deadline": {
            s: table[(table.suite == s) & (table.chunk == DEADLINE[s])
                     & ~table.degenerate]
            .sort_values("tp@fp20", ascending=False).head(3)
            .to_dict("records") for s in C.SUITES},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
