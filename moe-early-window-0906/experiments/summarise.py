#!/usr/bin/env python3
"""Assemble the per-suite verdict and the reachability answer from the result files.

Nothing is recomputed here except aggregation.  A cell is admissible only if at
least MIN_TASKS tasks contribute and at least MIN_PAIRS comparable pairs exist;
the same rule gated the permutation null, so headline effects and the critical
value they are compared against come from the same family.

The three readings a suite verdict is built from:

  effect            max |AUC - 0.5| over admissible in-window cells, per cohort
  fwer_p95          95th percentile of the same maximum under within-(task,
                    alive-at-q) label permutation, i.e. what the sweep produces
                    when there is nothing there
  replicated        the same (quantity, layer, chunk, direction) clearing the
                    critical value in both development and external

Cross-cohort replication is the stronger test: a cell can miss a family-wise
threshold on one cohort and still be real if it reproduces independently.  Both
are reported and neither is used alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

MIN_TASKS = 4
MIN_PAIRS = 5000
CONTROL_NULL = (
    "ctrl_const_elapsed",
    "ctrl_episode_const_rand",
    "ctrl_episode_const_rand2",
    "ctrl_flow_noise_seed",
    "ctrl_white_noise",
)
# quoted in the brief as the current frozen operating point, all three cohorts
V7_IN_WINDOW = {
    "libero_long": (712, 0.596),
    "libero_spatial": (315, 0.000),
    "libero_goal": (250, 0.156),
    "libero_object": (81, 0.111),
}
V7_TOTAL = (1358, 0.348)
TARGET_RECALL = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    imap = pd.read_csv(args.output / "information_map.csv.gz")
    null = json.loads((args.output / "permutation_null.json").read_text())
    ceiling = pd.read_csv(args.output / "cohort_budget_ceiling.csv")
    frontier = pd.read_csv(args.output / "deadline_frontier_summary.csv")
    survival = pd.read_csv(args.output / "survival_baseline.csv")

    fwer = {
        (e["cohort"], e["suite"]): e["null_max_effect_p95"] for e in null["per_suite"]
    }
    imap["abs_effect"] = imap["effect"].abs()
    imap["is_control"] = imap["quantity"].isin(C.CONTROL_QUANTITIES)
    admissible = imap[(imap["n_tasks"] >= MIN_TASKS) & (imap["pairs"] >= MIN_PAIRS)]
    real = admissible[
        (admissible["representation"] == "raw") & (~admissible["is_control"])
    ]
    ctrl = admissible[
        (admissible["representation"] == "raw")
        & (admissible["quantity"].isin(CONTROL_NULL))
    ]

    # ---- per (cohort, suite, chunk) information profile
    profile = (
        real.groupby(["cohort", "suite", "chunk", "phase", "in_window_65"])
        .agg(
            max_abs_effect=("abs_effect", "max"),
            n_cells=("abs_effect", "size"),
            median_n_tasks=("n_tasks", "median"),
            n_alive=("n_alive", "first"),
            n_alive_risk=("n_alive_risk", "first"),
            survival_prior=("prior", "first"),
        )
        .reset_index()
    )
    ctrl_profile = (
        ctrl.groupby(["cohort", "suite", "chunk"])
        .agg(control_max_abs_effect=("abs_effect", "max"))
        .reset_index()
    )
    profile = profile.merge(ctrl_profile, on=["cohort", "suite", "chunk"], how="left")
    profile["fwer_p95"] = [
        fwer.get((c, s), np.nan) for c, s in zip(profile["cohort"], profile["suite"])
    ]
    profile.to_csv(args.output / "information_profile_by_chunk.csv", index=False)

    # ---- best replicated in-window cell per suite
    inwin = real[real["in_window_65"]]
    dev = inwin[inwin["cohort"] == "development_main"]
    ext = inwin[inwin["cohort"] == "external_8b"]
    joined = dev.merge(ext, on=["suite", "chunk", "quantity", "layer"], suffixes=("_d", "_e"))
    joined["same_sign"] = np.sign(joined["effect_d"]) == np.sign(joined["effect_e"])
    joined["min_abs"] = np.where(
        joined["same_sign"], np.minimum(joined["abs_effect_d"], joined["abs_effect_e"]), 0.0
    )
    joined.sort_values("min_abs", ascending=False).head(400).to_csv(
        args.output / "replicated_in_window_cells.csv", index=False
    )

    verdict: list[dict] = []
    for suite in sorted(C.CAPS):
        sub = joined[joined["suite"] == suite]
        best = sub.sort_values("min_abs", ascending=False).iloc[0]
        thr_d = fwer.get(("development_main", suite), np.nan)
        thr_e = fwer.get(("external_8b", suite), np.nan)
        n_repl = int(
            (
                (sub["abs_effect_d"] > thr_d)
                & (sub["abs_effect_e"] > thr_e)
                & sub["same_sign"]
            ).sum()
        )
        # earliest chunk whose best replicated effect clears both thresholds
        clears = sub[
            (sub["abs_effect_d"] > thr_d) & (sub["abs_effect_e"] > thr_e) & sub["same_sign"]
        ]
        earliest = int(clears["chunk"].min()) if len(clears) else -1
        row = {
            "suite": suite,
            "cap": C.CAPS[suite],
            "window_last_chunk": C.WINDOW_65[suite],
            "window_chunks_from_4": C.WINDOW_65[suite] - C.SWEEP_START + 1,
            "window_chunks_from_6": C.WINDOW_65[suite] - 6 + 1,
            "fwer_p95_dev": thr_d,
            "fwer_p95_ext": thr_e,
            "best_cell": f"{best['quantity']}|{best['layer']}|q{int(best['chunk'])}",
            "best_phase": float(best["phase_d"]),
            "best_auc_dev": float(best["auc_d"]),
            "best_ci_dev": f"[{best['ci_lo_d']:.3f},{best['ci_hi_d']:.3f}]",
            "best_auc_ext": float(best["auc_e"]),
            "best_ci_ext": f"[{best['ci_lo_e']:.3f},{best['ci_hi_e']:.3f}]",
            "best_auc_equal_dev": float(best["auc_equal_d"]),
            "best_auc_equal_ext": float(best["auc_equal_e"]),
            "n_tasks_dev": int(best["n_tasks_d"]),
            "n_tasks_ext": int(best["n_tasks_e"]),
            "pairs_dev": float(best["pairs_d"]),
            "pairs_ext": float(best["pairs_e"]),
            "replicated_cells_over_fwer": n_repl,
            "admissible_cells": int(len(sub)),
            "earliest_replicated_chunk": earliest,
            "earliest_replicated_phase": (earliest + 1) / C.CAPS[suite] if earliest >= 0 else np.nan,
        }
        for cohort, tag in (("development_main", "dev"), ("external_8b", "ext")):
            best_arm = ceiling[
                (ceiling["cohort"] == cohort) & (ceiling["arm"] == "routing_best_of_all")
            ]
            null_arm = ceiling[
                (ceiling["cohort"] == cohort) & (ceiling["arm"] == "null_control")
            ]
            if len(best_arm):
                row[f"ceiling_recall_{tag}"] = float(best_arm[f"recall_{suite}"].iloc[0])
            if len(null_arm):
                row[f"null_recall_{tag}"] = float(null_arm[f"recall_{suite}"].iloc[0])
        row["v7_in_window_recall"] = V7_IN_WINDOW[suite][1]
        row["v7_risks"] = V7_IN_WINDOW[suite][0]
        # how far into the window the survival prior alone can carry an alarm
        surv = survival[
            (survival["cohort"] == "external_8b") & (survival["suite"] == suite)
        ]
        free = surv[surv["fpr_if_all_survivors_alarm"] <= 0.005]
        row["length_only_first_affordable_chunk"] = (
            int(free["chunk"].min()) if len(free) else -1
        )
        row["length_only_first_affordable_phase"] = (
            float(free["phase"].min()) if len(free) else np.nan
        )
        verdict.append(row)

    table = pd.DataFrame(verdict)
    table.to_csv(args.output / "verdict_by_suite.csv", index=False)

    key = {
        "estimator": {
            "definition": (
                "within-task Mann-Whitney AUC for eventual risk among episodes still "
                "running at chunk q, pooled over tasks with weights n+ * n-"
            ),
            "phase_definition": "(q + 1) / episode_length; risk episodes have length == cap",
            "window_last_chunk": C.WINDOW_65,
            "admissibility": {"min_tasks": MIN_TASKS, "min_pairs": MIN_PAIRS},
            "delong_se_over_permutation_sd_median": 0.970,
            "const_elapsed_control_auc": 0.5,
            "length_leak_auc_median": 1.0,
        },
        "family_wise_null_p95_abs_effect": {f"{k[0]}|{k[1]}": v for k, v in fwer.items()},
        "in_window_ceiling_cohort_budget": (
            ceiling[ceiling["scope"] == "in_window_phase<=0.65"][
                ["cohort", "arm", "fp_budget", "tp", "risks", "recall"]
            ].to_dict("records")
        ),
        "deadline_frontier": frontier.to_dict("records"),
        "target": {
            "recall": TARGET_RECALL,
            "phase": 0.65,
            "current_v7_total": {"risks": V7_TOTAL[0], "recall": V7_TOTAL[1]},
        },
    }
    C.write_json(args.output / "key_numbers.json", key)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
