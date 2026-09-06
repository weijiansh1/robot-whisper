#!/usr/bin/env python3
"""Assemble the summary tables the report quotes, and the cross-checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_methods import collect  # noqa: E402
from ledger_core import Cohort  # noqa: E402
from reconcile_prior_replay import cross_check_modes  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
pd.set_option("display.width", 300)

HEADLINE = [
    "baseline:length_only_task_devcal@fpr0.005",
    "v7_guard",
    "v4_dual_regime",
    "combo:P-C",
    "combo:G-A",
    "mobility|global",
    "expert_load_effective_rank|per_task",
    "twotier:WATCH_a3",
    "twotier:ACT_b",
    "build:OR_crossframe_global",
    "prior:HB_MoE_top4churn@qfpr1pct",
]


def main() -> None:
    cohort = Cohort("external_8b")
    methods, _, _, _ = collect(cohort)
    detail = pd.read_csv(RESULTS / "method_detail_external.csv")
    dev_detail = pd.read_csv(RESULTS / "method_detail_development.csv")

    # ---- 1. mode cross-check against the prior replay bundle ---------------
    modes = cross_check_modes(cohort, detail)
    modes.to_csv(RESULTS / "mode_cross_check.csv", index=False)
    print("=== per-mode recall, ours vs the prior replay bundle "
          "(different denominators, never pooled)")
    print(modes.to_string(index=False))

    # ---- 2. dev / external side by side ------------------------------------
    shared = sorted(set(detail["method"]) & set(dev_detail["method"]))
    left = detail.set_index("method").loc[shared]
    right = dev_detail.set_index("method").loc[shared]
    transfer = pd.DataFrame(
        {
            "method": shared,
            "dev_tp": right["tp"].to_numpy(),
            "dev_fp": right["fp"].to_numpy(),
            "dev_precision": right["precision"].to_numpy(),
            "dev_timely_fpr": right["timely_fpr"].to_numpy(),
            "ext_tp": left["tp"].to_numpy(),
            "ext_fp": left["fp"].to_numpy(),
            "ext_precision": left["precision"].to_numpy(),
            "ext_timely_fpr": left["timely_fpr"].to_numpy(),
        }
    )
    transfer["fp_inflation"] = transfer["ext_timely_fpr"] / transfer[
        "dev_timely_fpr"
    ].replace(0, np.nan)
    transfer["recall_shift"] = (
        transfer["ext_tp"] / 564 - transfer["dev_tp"] / 487
    )
    transfer = transfer.sort_values("fp_inflation", ascending=False)
    transfer.to_csv(RESULTS / "development_to_external_transfer.csv", index=False)
    print("\n=== worst threshold transfer (development -> external)")
    print(transfer.head(12).to_string(index=False))
    print("\n=== best threshold transfer")
    print(transfer.tail(8).to_string(index=False))

    # ---- 3. headline table --------------------------------------------------
    cols = [
        "method", "threshold_mode", "alarms", "tp", "fp", "precision", "recall",
        "timely_fpr", "mean_matched_prior", "lift", "alarm_chunk_median",
        "remaining_budget_median", "lead_median", "early_tp", "early_fp",
        "budget_won", "frac_of_ceiling", "avg_budget_per_tp", "gain_per_loss",
        "worst_mode_recall", "worst_mode_name", "missed_risks",
    ]
    head = detail[detail["method"].isin(HEADLINE)][cols]
    head = head.set_index("method").loc[[m for m in HEADLINE if m in set(head["method"])] if False else HEADLINE].reset_index()
    head.to_csv(RESULTS / "headline_table.csv", index=False)
    print("\n=== headline table")
    print(head.to_string(index=False))

    # ---- 4. suite breakdown of the headline methods ------------------------
    by_suite = pd.read_csv(RESULTS / "method_by_suite_external.csv")
    block = by_suite[by_suite["method"].isin(HEADLINE)]
    print("\n=== headline methods by suite")
    print(
        block.pivot(index="method", columns="suite", values="recall")
        .round(3)
        .to_string()
    )
    print("\n(false alarms by suite)")
    print(block.pivot(index="method", columns="suite", values="fp").to_string())

    # ---- 5. budget view ----------------------------------------------------
    print("\n=== TP/FP retained under a minimum remaining-budget requirement")
    bcols = ["method"] + [f"budget{b}_{k}" for b in (0, 2, 4, 8, 12) for k in ("tp", "fp")]
    print(detail[detail["method"].isin(HEADLINE)][bcols].to_string(index=False))

    # ---- 6. failure-mode split, headline methods ---------------------------
    mode_cols = ["method"] + [
        f"mode_{m}_{k}"
        for m in ("dropped", "no_grasp", "moved_unmet", "regressed",
                  "released_outside", "no_contact", "timeout_holding")
        for k in ("n", "hit")
    ]
    print("\n=== failure-mode hits (n is over all 564 risks)")
    print(detail[detail["method"].isin(HEADLINE)][mode_cols].to_string(index=False))

    # ---- 7. operator view, pooled, for the headline methods ---------------
    chunks = pd.read_csv(RESULTS / "chunk_view_external.csv")
    pooled = chunks[(chunks["suite"] == "ALL") & (chunks["method"].isin(HEADLINE))]
    peak = (
        pooled.sort_values("alarms_at_chunk", ascending=False)
        .groupby("method")
        .head(3)
        .sort_values(["method", "chunk"])
    )
    print("\n=== the three busiest chunks for each headline method (pooled)")
    print(
        peak[
            ["method", "chunk", "episodes_still_running", "alarms_at_chunk",
             "alarms_at_chunk_risk", "alarms_at_chunk_timely"]
        ].to_string(index=False)
    )

    # ---- 8. how much of each method's TP set is unique? --------------------
    lookup = {m.name: m.first for m in methods}
    rows = []
    for name in HEADLINE:
        if name not in lookup:
            continue
        mine = (lookup[name] >= 0) & cohort.risk
        others = np.zeros(cohort.n, bool)
        for other, arr in lookup.items():
            if other == name or other.startswith(("baseline:", "prior:length_only")):
                continue
            others |= (arr >= 0) & cohort.risk
        rows.append(
            {
                "method": name,
                "tp": int(mine.sum()),
                "tp_not_found_by_any_other_routing_method": int((mine & ~others).sum()),
            }
        )
    uniq = pd.DataFrame(rows)
    uniq.to_csv(RESULTS / "unique_true_positives.csv", index=False)
    print("\n=== true positives no other routing method in the ledger found")
    print(uniq.to_string(index=False))

    summary = {
        "external_methods": len(methods),
        "risks_missed_by_all_routing_detectors": int(
            (
                cohort.risk
                & ~np.stack(
                    [
                        lookup[m.name] >= 0
                        for m in methods
                        if m.family != "length_only_baseline"
                        and not m.name.startswith("prior:length_only")
                    ]
                ).any(axis=0)
            ).sum()
        ),
        "episodes_reaching_cap": int((cohort.length >= cohort.horizon).sum()),
        "of_which_risks": int(cohort.risk.sum()),
        "length_rule_precision_by_construction": float(
            cohort.risk.sum() / (cohort.length >= cohort.horizon).sum()
        ),
    }
    (RESULTS / "summary_numbers.json").write_text(json.dumps(summary, indent=1))
    print("\n", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
