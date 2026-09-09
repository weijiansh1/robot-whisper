#!/usr/bin/env python3
"""Collect every headline number into one file, and build the report tables.

Reads only what the other scripts wrote.  Adds nothing new except:
  * an external false-alarm-matched comparison of the single detector, the
    mode-targeted pair and the undifferentiated pair;
  * a less extreme reference for the within-task white-noise sweep (the 99th
    percentile of |AUC - 0.5| as well as its maximum);
  * the per-mode decomposition of the published `v7_guard` anchor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import modes_common as M

FP_MATCH_TOL = 12   # external false alarms, for the matched comparison


def matched_union_table(union: pd.DataFrame, target_fp: int) -> pd.DataFrame:
    """Closest operating point to `target_fp` external false alarms per family."""
    ext = union[(union.cohort == "external_8b") & (union.budget_dev_fp_lead4 > 0)]
    rows = []
    for (arm, det), grp in ext.groupby(["arm", "detector"]):
        g = grp.assign(gap=(grp.fp_lead4 - target_fp).abs()).nsmallest(1, "gap")
        r = g.iloc[0]
        rows.append({"arm": arm, "detector": det,
                     "budget_dev_fp_lead4": int(r.budget_dev_fp_lead4),
                     "fp_lead4": int(r.fp_lead4), "tp_lead4": int(r.tp_lead4),
                     "n_risk": int(r.n_risk),
                     "tp_drop_lead4": int(r.tp_drop_lead4), "n_drop": int(r.n_drop),
                     "tp_grasp_lead4": int(r.tp_grasp_lead4), "n_grasp": int(r.n_grasp),
                     "median_lead": float(r.median_lead_at_lead4),
                     "fp_gap": int(r.gap), "config": r.config})
    return pd.DataFrame(rows).sort_values(["arm", "detector"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=M.RESULTS)
    args = p.parse_args()
    R = args.out

    sizes = json.loads((R / "mode_sizes.json").read_text())
    frontier = pd.read_csv(R / "arm_frontier.csv")
    picks = pd.read_csv(R / "selected_operating_points.csv")
    loto = pd.read_csv(R / "loto_summary.csv")
    union = pd.read_csv(R / "union_vs_single.csv")
    inter_sel = json.loads((R / "interaction.json").read_text())
    inter_fro = json.loads((R / "arm_frontier_interaction.json").read_text())
    ceiling = pd.read_csv(R / "within_task_null_ceiling.csv")
    by_chunk = pd.read_csv(R / "within_task_by_chunk.csv")
    repl = pd.read_csv(R / "within_task_replicated.csv.gz")
    nul = pd.read_csv(R / "within_task_auc_null.csv.gz")

    # ---- external FP-matched comparison at the v7_guard operating point ---- #
    guard = union[(union.cohort == "external_8b") & (union.detector == "v7_guard")].iloc[0]
    target_fp = int(guard.fp_lead4)
    matched = matched_union_table(union, target_fp)
    M.write_csv(R / "union_matched_fp.csv", matched)

    # ---- less extreme null reference for the AUC sweep -------------------- #
    if "headline_cell" not in nul.columns:
        nul["headline_cell"] = (nul.n_tasks >= 2) & (nul.pairs >= 2000)
    q99 = (nul[nul.headline_cell].groupby(["cohort", "target"]).abs_effect
           .quantile(0.99).rename("null_p99").reset_index())
    ceiling = ceiling.merge(q99, on=["cohort", "target"], how="left")
    M.write_csv(R / "within_task_null_ceiling.csv", ceiling)

    # ---- assemble ---------------------------------------------------------- #
    def cell(pool, arm, mode, fp=57, cohort="external_8b"):
        r = frontier[(frontier.cohort == cohort) & (frontier.pool == pool)
                     & (frontier.arm == arm) & (frontier["mode"] == mode)
                     & (frontier.fp_budget == fp)]
        return r.iloc[0].to_dict() if len(r) else {}

    key = {
        "cohort_sizes": sizes,
        "anchors_reproduced": json.loads((R / "harness_checks.json").read_text()),
        "v7_guard_external": {
            "tp_lead4": int(guard.tp_lead4), "n_risk": int(guard.n_risk),
            "fp_lead4": int(guard.fp_lead4),
            "tp_drop_lead4": int(guard.tp_drop_lead4), "n_drop": int(guard.n_drop),
            "tp_grasp_lead4": int(guard.tp_grasp_lead4), "n_grasp": int(guard.n_grasp),
            "tp_other_lead4": int(guard.tp_other_lead4),
            "median_lead_at_lead4": float(guard.median_lead_at_lead4),
        },
        "mode_x_arm_ceiling_external_fp57": {
            f"{mode}|{arm}": {
                "tp": int(cell("headline", arm, mode)["tp"]),
                "n_mode": int(cell("headline", arm, mode)["n_mode"]),
                "fp": int(cell("headline", arm, mode)["fp"]),
                "floor": int(cell("headline", arm, mode)["floor"]),
                "excess_over_floor": int(cell("headline", arm, mode)["excess_over_floor"]),
                "channel": cell("headline", arm, mode)["channel"],
                "stat": cell("headline", arm, mode)["stat"]}
            for mode in ("drop", "grasp")
            for arm in ("threshold", "change_point", "persistence")},
        "interaction_on_ceilings": inter_fro["contrasts"],
        "interaction_on_dev_selected": inter_sel["contrasts"],
        "floor_external_fp57": {
            "fixed_chunk_baseline_drop": int(cell("headline", "threshold", "drop")["baseline_tp"]),
            "fixed_chunk_baseline_grasp": int(cell("headline", "threshold", "grasp")["baseline_tp"]),
            "null_pool_ceiling_drop": int(cell("headline", "threshold", "drop")["null_ceiling"]),
            "null_pool_ceiling_grasp": int(cell("headline", "threshold", "grasp")["null_ceiling"]),
            "elapsed_counter_persistence": {
                "tp_drop": int(cell("null_elapsed", "persistence", "drop")["tp"]),
                "tp_grasp": int(cell("null_elapsed", "persistence", "grasp")["tp"]),
                "tp_all": int(cell("null_elapsed", "persistence", "all")["tp"]),
                "fp": int(cell("null_elapsed", "persistence", "all")["fp"])},
            "white_noise_persistence": {
                "tp_drop": int(cell("null_noise", "persistence", "drop")["tp"]),
                "tp_grasp": int(cell("null_noise", "persistence", "grasp")["tp"])},
            "episode_constant_change_point": {
                "tp_drop": int(cell("null_epconst", "change_point", "drop")["tp"]),
                "tp_grasp": int(cell("null_epconst", "change_point", "grasp")["tp"]),
                "tp_all": int(cell("null_epconst", "change_point", "all")["tp"]),
                "fp": int(cell("null_epconst", "change_point", "all")["fp"])},
        },
        "within_task_null_ceiling": ceiling.to_dict("records"),
        "within_task_replication": {
            f"{t}|headline={h}": {"replicated": int(g.replicated.sum()),
                                  "cells": int(len(g))}
            for (t, h), g in repl.groupby(["target", "headline_cell"])},
    }

    # LOTO at the headline budget
    lo = loto[(loto.budget_dev_fp_lead4 == 38)]
    key["loto_external_budget38"] = {
        f"{r.pool}|{r.arm}|{r.selected_for}": {
            "tp_drop": int(r.tp_drop), "n_drop": int(r.n_drop),
            "tp_grasp": int(r.tp_grasp), "n_grasp": int(r.n_grasp),
            "tp_all": int(r.tp_all), "n_risk": int(r.n_risk), "fp": int(r.fp),
            "n_folds": int(r.n_folds)}
        for r in lo.itertuples()}

    # pooled dev-selected picks at the headline budget
    hp = picks[(picks.budget_dev_fp_lead4 == 38) & (picks.pool == "headline")
               & picks.arm.isin(("threshold", "change_point", "persistence"))]
    key["pooled_dev_selected_budget38"] = {
        f"{r.arm}|{r.selected_for}": {
            "channel": r.channel, "sign": int(r.sign), "stat": r.stat,
            "dev_fp": int(r.dev_fp_lead4), "dev_tp_all": int(r.dev_tp_all_lead4),
            "dev_tp_drop": int(r.dev_tp_drop_lead4),
            "dev_tp_grasp": int(r.dev_tp_grasp_lead4),
            "ext_fp": int(r.extR_fp_lead4), "ext_tp_all": int(r.extR_tp_all_lead4),
            "ext_tp_drop": int(r.extR_tp_drop_lead4),
            "ext_tp_grasp": int(r.extR_tp_grasp_lead4),
            "ext_median_alarm_chunk": float(r.extR_median_alarm_chunk)}
        for r in hp.itertuples()}

    key["union_matched_external_fp"] = {
        "target_fp": target_fp,
        "rows": matched[matched.arm.isin(("threshold", "change_point",
                                          "persistence", "any", "published"))]
        .to_dict("records")}

    # detector-free temporal shape: max |AUC-0.5| by chunk, headline cells
    shape = by_chunk[(by_chunk.cohort == "external_8b")
                     & by_chunk.max_abs_effect_headline.notna()]
    key["temporal_shape_external"] = {
        f"{r.suite}|{r.target}|q{int(r.chunk)}": round(float(r.max_abs_effect_headline), 3)
        for r in shape.itertuples()
        if r.suite == "libero_long" and int(r.chunk) in (4, 6, 8, 12, 16, 20, 25, 30)}

    M.write_json(R / "key_numbers.json", key)
    print(json.dumps(key, indent=2, default=str)[:6000])


if __name__ == "__main__":
    main()
