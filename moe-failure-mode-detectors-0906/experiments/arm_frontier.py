#!/usr/bin/env python3
"""Arm ceilings at matched false alarms - the fair form of the mode x arm test.

A single development-selected operating point cannot be compared across arms on
external, because the two cohorts differ in composition (development has 2 800
libero_long episodes, external 4 000), so the same alarm rate produces a
different number of *timely* false alarms.  Holding the false alarm count fixed
on the cohort being scored is the only way to ask whether one arm carries more
information about a mode than another.

So this computes, per (cohort, pool, arm, mode), the upper envelope

    ceiling(f) = max over operating points of that arm of
                 TP(mode) at lead >= 4, subject to FP at lead >= 4 <= f

This is an *upper bound*, not a deployment number: the operating point is
chosen using the outcomes of the cohort being scored.  It is only interpretable
next to the same ceiling computed on the null pool (white noise, searched with
the same statistics and the same number of operating points) and next to the
cap-free fixed-chunk baseline.  All three are reported together.

Uncertainty is a task-clustered bootstrap in which the maximum is *recomputed
inside every draw*, over a candidate set of the top operating points, so the
interval carries the selection as well as the sampling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import arms as A
import modes_common as M
from select_and_score import (BUDGETS, HEADLINE_BUDGET, load_table, totals,
                              baseline_frame, baseline_at, Q0_PROTOCOL)

FP_GRID = (0, 10, 20, 38, 57, 80, 120, 200, 400)
HEADLINE_FP = 57          # the v7_guard external anchor at lead >= 4
BOOT_DRAWS = 2000
N_CANDIDATE = 300


def ceiling(tot: dict, rows: np.ndarray, key: str, fp_budget: int) -> tuple[int, int]:
    ok = rows & (tot["fp"] <= fp_budget)
    if not ok.any():
        return 0, -1
    tp = np.where(ok, tot[key], -1)
    j = int(np.argmax(tp))
    return int(tp[j]), j


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=M.CACHE)
    p.add_argument("--out", type=Path, default=M.RESULTS)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(M.SEED + 7)

    frames = {c: M.load(c) for c in ("development_main", "external_8b")}
    tables = {"development_main": load_table(args.cache / "development_main_ops.npz"),
              "external_8b": load_table(args.cache / "external_8b_ops_ratematched.npz")}
    ops = tables["development_main"]["ops"]
    arm_of = ops["arm"].to_numpy()
    pools = A.pool_masks(ops)

    n_mode = {c: {"drop": int((f["risk"] & (f["mode"] == M.MODE_DROP)).sum()),
                  "grasp": int((f["risk"] & (f["mode"] == M.MODE_GRASP)).sum()),
                  "all": int(f["risk"].sum())} for c, f in frames.items()}
    base = {c: baseline_frame(f, Q0_PROTOCOL) for c, f in frames.items()}

    rows = []
    for cohort, table in tables.items():
        mask = np.ones(len(table["tasks"]), bool)
        tot = totals(table["counts"], mask, M.HEADLINE_LEAD)
        for pool, pmask in pools.items():
            for arm in list(A.ARMS) + ["any"]:
                sel = pmask if arm == "any" else (pmask & (arm_of == arm))
                for mode, key in (("drop", "tp_drop"), ("grasp", "tp_grasp"),
                                  ("all", "tp_all")):
                    for f in FP_GRID:
                        tp, j = ceiling(tot, sel, key, f)
                        r = ops.iloc[j] if j >= 0 else None
                        rows.append({
                            "cohort": cohort, "pool": pool, "arm": arm,
                            "mode": mode, "fp_budget": f, "tp": tp,
                            "n_mode": n_mode[cohort][mode],
                            "fp": int(tot["fp"][j]) if j >= 0 else 0,
                            "baseline_tp": baseline_at(
                                base[cohort], f, f"tp_{mode}" if mode != "all"
                                else "tp_all"),
                            "channel": f"{r['quantity']}|{r['layer']}" if r is not None else "",
                            "sign": int(r["sign"]) if r is not None else 0,
                            "stat": r["stat"] if r is not None else "",
                            "n_ops_searched": int(sel.sum()),
                        })
    frontier = pd.DataFrame(rows)
    frontier["excess_over_baseline"] = frontier.tp - frontier.baseline_tp

    # The fixed-chunk baseline is not the only floor.  Under a cap-free
    # protocol any statistic that accumulates over chunks can rediscover the
    # horizon rule from a channel with no within-episode information at all -
    # a chunk counter, or a number drawn once per episode.  The floor is
    # therefore the *maximum* of the baseline and every null pool's ceiling at
    # the same false-alarm budget, and that is what excess is measured against.
    null_pools = ("null_noise", "null_elapsed", "null_epconst")
    nul = (frontier[frontier.pool.isin(null_pools)]
           .groupby(["cohort", "mode", "fp_budget"]).tp.max()
           .rename("null_ceiling").reset_index())
    frontier = frontier.merge(nul, on=["cohort", "mode", "fp_budget"], how="left")
    frontier["floor"] = frontier[["baseline_tp", "null_ceiling"]].max(axis=1)
    frontier["excess_over_floor"] = frontier.tp - frontier.floor
    M.write_csv(args.out / "arm_frontier.csv", frontier)

    # ---------------- bootstrap of the ceilings and their contrasts -------- #
    cohort = "external_8b"
    table = tables[cohort]
    n_task = len(table["tasks"])
    tot = totals(table["counts"], np.ones(n_task, bool), M.HEADLINE_LEAD)
    li = {b: i for i, b in enumerate(M.LEADS)}[M.HEADLINE_LEAD]
    draw = rng.integers(0, n_task, size=(BOOT_DRAWS, n_task))

    boot_ceiling: dict[str, np.ndarray] = {}
    used: dict[str, list[int]] = {}
    for pool in ("headline", "null_noise", "null_elapsed"):
        for arm in list(A.ARMS) + ["any"]:
            sel = pools[pool] if arm == "any" else (pools[pool] & (arm_of == arm))
            for mode, key in (("drop", "tp_drop"), ("grasp", "tp_grasp")):
                ok = sel & (tot["fp"] <= HEADLINE_FP)
                if not ok.any():
                    continue
                idx = np.flatnonzero(ok)
                cand = idx[np.argsort(tot[key][idx])[::-1][:N_CANDIDATE]]
                used[f"{pool}|{arm}|{mode}"] = cand.tolist()
                sub = table["counts"][cand][:, :, :, li]          # (c, task, group)
                gi = A.GROUP_DROP if mode == "drop" else A.GROUP_GRASP
                tp_d = sub[:, :, gi][:, draw].sum(axis=2)          # (c, draws)
                fp_d = sub[:, :, A.GROUP_NONRISK][:, draw].sum(axis=2)
                scale = HEADLINE_FP * n_task / n_task              # same budget
                allowed = fp_d <= scale
                masked = np.where(allowed, tp_d, -1)
                boot_ceiling[f"{pool}|{arm}|{mode}"] = masked.max(axis=0)

    summary = {}
    for k, v in boot_ceiling.items():
        pool, arm, mode = k.split("|")
        point = int(frontier[(frontier.cohort == cohort) & (frontier.pool == pool)
                             & (frontier.arm == arm) & (frontier["mode"] == mode)
                             & (frontier.fp_budget == HEADLINE_FP)].tp.iloc[0])
        summary[k] = {"point": point, "n_mode": n_mode[cohort][mode],
                      "boot_ci_lo": float(np.quantile(v, 0.025)),
                      "boot_ci_hi": float(np.quantile(v, 0.975)),
                      "n_candidates": len(used[k])}

    contrasts = {}
    for arm in ("change_point", "persistence"):
        try:
            d_drop = (boot_ceiling[f"headline|{arm}|drop"]
                      - boot_ceiling["headline|threshold|drop"])
            d_grasp = (boot_ceiling[f"headline|{arm}|grasp"]
                       - boot_ceiling["headline|threshold|grasp"])
        except KeyError:
            continue
        did = d_drop - d_grasp
        contrasts[arm] = {
            "gain_on_drop": {
                "point": summary[f"headline|{arm}|drop"]["point"]
                         - summary["headline|threshold|drop"]["point"],
                "ci_lo": float(np.quantile(d_drop, 0.025)),
                "ci_hi": float(np.quantile(d_drop, 0.975))},
            "gain_on_grasp": {
                "point": summary[f"headline|{arm}|grasp"]["point"]
                         - summary["headline|threshold|grasp"]["point"],
                "ci_lo": float(np.quantile(d_grasp, 0.025)),
                "ci_hi": float(np.quantile(d_grasp, 0.975))},
            "difference_in_differences": {
                "point": (summary[f"headline|{arm}|drop"]["point"]
                          - summary["headline|threshold|drop"]["point"])
                         - (summary[f"headline|{arm}|grasp"]["point"]
                            - summary["headline|threshold|grasp"]["point"]),
                "ci_lo": float(np.quantile(did, 0.025)),
                "ci_hi": float(np.quantile(did, 0.975)),
                "p_two_sided_sign": float(2 * min((did <= 0).mean(),
                                                  (did >= 0).mean()))},
        }
    M.write_json(args.out / "arm_frontier_interaction.json", {
        "fp_budget_lead4": HEADLINE_FP, "cohort": cohort,
        "bootstrap_draws": BOOT_DRAWS, "n_candidates_per_cell": N_CANDIDATE,
        "note": ("ceilings choose the operating point with the scored cohort's "
                 "own outcomes; they are upper bounds, not deployment numbers"),
        "cells": summary, "contrasts": contrasts})

    print("== arm ceiling at FP <= %d, external_8b, lead >= 4 ==" % HEADLINE_FP)
    show = frontier[(frontier.cohort == "external_8b")
                    & (frontier.fp_budget == HEADLINE_FP)
                    & (frontier["mode"] != "all")]
    print(show[["pool", "arm", "mode", "tp", "n_mode", "fp", "baseline_tp",
                "null_ceiling", "floor", "excess_over_floor", "channel",
                "stat"]].to_string(index=False))
    print("\n== interaction on the ceilings ==")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
