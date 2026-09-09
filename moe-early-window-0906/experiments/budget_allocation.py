#!/usr/bin/env python3
"""The ceiling under the operating point the deployed guard actually uses.

`reachability.py` held every suite to a timely FPR of 0.005 *within that suite*.
The deployed v7 guard does not: its per-suite timely FPR is 0.000 on
libero_spatial and 0.0233 on libero_long, i.e. it spends nearly the whole cohort
budget on the one suite where alarms pay.  Holding each suite to 0.005
separately is therefore a stricter constraint than the thing being compared
against, and would understate the ceiling.

This script re-derives the ceiling under the cohort-level constraint

    total false alarms <= 0.005 * (non-risk episodes in the cohort)

allocated across suites optimally.  For each suite the (fp, tp) Pareto frontier
over every in-window rule is computed, then a small exact DP over the four
frontiers maximises total true positives at the shared budget.  Because both the
per-suite rule and the allocation are chosen with the outcomes in hand, this is
an upper bound and is labelled as one; the same DP is run on the null-control
channels so the part of it that is pure selection is visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from reachability import FPR_BUDGET

NULL_CONTROLS = (
    "ctrl_const_elapsed",
    "ctrl_episode_const_rand",
    "ctrl_episode_const_rand2",
    "ctrl_flow_noise_seed",
    "ctrl_white_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def pareto(table: pd.DataFrame, budget: int) -> np.ndarray:
    """best tp achievable with exactly <= f false alarms, for f in 0..budget."""
    best = np.zeros(budget + 1, dtype=int)
    if len(table):
        grouped = table.groupby("fp")["tp"].max()
        for fp, tp in grouped.items():
            if fp <= budget:
                best[int(fp)] = max(best[int(fp)], int(tp))
    return np.maximum.accumulate(best)


def allocate(frontiers: dict[str, np.ndarray], budget: int) -> tuple[int, dict[str, int]]:
    suites = list(frontiers)
    dp = np.zeros(budget + 1, dtype=int)
    choice: list[np.ndarray] = []
    for suite in suites:
        f = frontiers[suite]
        new = np.zeros(budget + 1, dtype=int)
        pick = np.zeros(budget + 1, dtype=int)
        for total in range(budget + 1):
            spend = np.arange(total + 1)
            values = dp[total - spend] + f[spend]
            k = int(np.argmax(values))
            new[total] = values[k]
            pick[total] = k
        dp = new
        choice.append(pick)
    alloc: dict[str, int] = {}
    remaining = budget
    for suite, pick in zip(reversed(suites), reversed(choice), strict=True):
        spend = int(pick[remaining])
        alloc[suite] = spend
        remaining -= spend
    return int(dp[budget]), alloc


def main() -> None:
    args = parse_args()
    single = pd.read_csv(args.output / "reachability_single_chunk.csv.gz")
    mv = pd.read_csv(args.output / "upper_bound_multivariate.csv")

    single["in_window_65"] = single.apply(
        lambda r: r["chunk"] <= C.WINDOW_65[r["suite"]], axis=1
    )
    single["arm"] = np.where(
        single["quantity"] == "leak_full_length", "length_leak",
        np.where(single["quantity"].isin(NULL_CONTROLS), "null_control", "routing_single"),
    )
    mv["arm"] = np.where(mv["arm"] == "stacked_mv", "routing_stacked_mv", "null_stacked_mv")
    mv["arm"] = mv["arm"] + ":" + mv["variant"]
    mv["in_window_65"] = mv.apply(lambda r: r["chunk"] <= C.WINDOW_65[r["suite"]], axis=1)

    rows: list[dict] = []
    detail: list[dict] = []
    for cohort in ("development_main", "external_8b"):
        n_safe = int(
            single[(single["cohort"] == cohort)]
            .groupby("suite")["n_safe"].first().sum()
        )
        budget = int(np.floor(FPR_BUDGET * n_safe))
        n_risk_total = int(
            single[(single["cohort"] == cohort)]
            .groupby("suite")["n_risk"].first().sum()
        )
        arms = {
            arm: single[
                (single["cohort"] == cohort) & single["in_window_65"] & (single["arm"] == arm)
            ]
            for arm in ("routing_single", "null_control", "length_leak")
        }
        for arm in mv["arm"].unique():
            sub = mv[(mv["cohort"] == cohort) & mv["in_window_65"] & (mv["arm"] == arm)]
            if len(sub):
                arms[arm] = sub
        # the strongest routing arm: best of single-channel and multivariate per suite
        combo = pd.concat(
            [arms["routing_single"]]
            + [arms[a] for a in arms if a.startswith("routing_stacked_mv")],
            ignore_index=True,
        )
        arms["routing_best_of_all"] = combo

        for arm, table in arms.items():
            frontiers = {
                suite: pareto(table[table["suite"] == suite], budget)
                for suite in sorted(C.CAPS)
            }
            total_tp, alloc = allocate(frontiers, budget)
            rows.append(
                {
                    "cohort": cohort, "arm": arm, "scope": "in_window_phase<=0.65",
                    "fp_budget": budget, "cohort_timely_fpr": budget / n_safe,
                    "tp": total_tp, "risks": n_risk_total,
                    "recall": total_tp / max(n_risk_total, 1),
                    **{f"fp_{s}": alloc[s] for s in sorted(C.CAPS)},
                    **{
                        f"recall_{s}": frontiers[s][alloc[s]]
                        / max(int(table[table["suite"] == s]["n_risk"].iloc[0]) if len(
                            table[table["suite"] == s]) else 1, 1)
                        for s in sorted(C.CAPS)
                    },
                }
            )
            for suite in sorted(C.CAPS):
                sub = table[(table["suite"] == suite) & (table["fp"] <= alloc[suite])]
                if not len(sub):
                    continue
                best = sub.sort_values("tp", ascending=False).iloc[0]
                detail.append(
                    {
                        "cohort": cohort, "arm": arm, "suite": suite,
                        "fp_allocated": alloc[suite], "chunk": int(best["chunk"]),
                        "phase": float(best["phase"]), "alpha": float(best["alpha"]),
                        "tp": int(best["tp"]), "fp": int(best["fp"]),
                        "n_risk": int(best["n_risk"]),
                        "recall": float(best["tp"]) / max(int(best["n_risk"]), 1),
                        "head": (
                            f"{best['quantity']}|{best['layer']}|{best['direction']}"
                            if "quantity" in best.index and isinstance(best.get("quantity"), str)
                            else arm
                        ),
                    }
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "cohort_budget_ceiling.csv", index=False)
    pd.DataFrame(detail).to_csv(args.output / "cohort_budget_detail.csv", index=False)
    cols = ["cohort", "arm", "fp_budget", "tp", "risks", "recall"] + [
        f"recall_{s}" for s in sorted(C.CAPS)
    ] + [f"fp_{s}" for s in sorted(C.CAPS)]
    print(table[cols].to_string(index=False))


if __name__ == "__main__":
    main()
