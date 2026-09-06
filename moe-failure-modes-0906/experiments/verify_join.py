#!/usr/bin/env python3
"""Verify the physical-label join independently, for both cohorts.

The controller stated the join is verified and complete. This re-derives it from
the raw files and writes an audit, so nothing downstream rests on a claim we did
not check ourselves.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"


def audit(cohort: str) -> tuple[pd.DataFrame, dict]:
    frame = C.cohort_index(cohort)
    risk = frame["risk"].to_numpy(bool)
    matched = frame["physical_matched"].to_numpy(bool)
    has_reason = frame["primary_failure_reason"].notna().to_numpy()

    # Cross-check the two independent key columns that were NOT used in the join.
    key_ok = matched & frame["phys_init_state_id"].notna().to_numpy()
    init_agree = int(
        (frame.loc[key_ok, "init_state_id"].to_numpy(int)
         == frame.loc[key_ok, "phys_init_state_id"].to_numpy(int)).sum()
    )
    seed_agree = int(
        (frame.loc[key_ok, "flow_noise_seed"].to_numpy(int)
         == frame.loc[key_ok, "phys_flow_noise_seed"].to_numpy(int)).sum()
    )

    # recorded_success from the simulator vs. the routing-side outcome label.
    rec = frame["recorded_success"]
    rec_bool = rec.map({True: True, False: False, "True": True, "False": False})
    agree = int(((~risk) == rec_bool.fillna(False).to_numpy(bool))[matched].sum())

    report = {
        "cohort": cohort,
        "run_id": C.RUN_IDS[cohort],
        "rows": int(len(frame)),
        "physical_matched": int(matched.sum()),
        "risks": int(risk.sum()),
        "risks_with_reason": int((risk & has_reason).sum()),
        "risks_missing_reason": int((risk & ~has_reason).sum()),
        "nonrisk_with_reason": int((~risk & has_reason).sum()),
        "init_state_id_agreement": init_agree,
        "flow_noise_seed_agreement": seed_agree,
        "key_checked": int(key_ok.sum()),
        "recorded_success_vs_outcome_agreement": agree,
        "physics_validation_status": frame.loc[
            risk & has_reason, "physics_validation_status"
        ].value_counts().to_dict(),
        "confidence_over_risks": frame.loc[risk & has_reason, "failure_reason_confidence"]
        .value_counts().to_dict(),
        "confidence_over_persistent": frame.loc[
            risk & has_reason & ~frame["late_success_plus10_queries"].astype(bool),
            "failure_reason_confidence",
        ].value_counts().to_dict(),
        "mode_counts_all_risks": frame.loc[risk & has_reason, "primary_failure_reason"]
        .value_counts().to_dict(),
        "mode_counts_persistent": frame.loc[
            risk & has_reason & ~frame["late_success_plus10_queries"].astype(bool),
            "primary_failure_reason",
        ].value_counts().to_dict(),
        "late_successes_among_risks": int(
            (risk & frame["late_success_plus10_queries"].astype(bool)).sum()
        ),
        "risk_length_equals_cap": int(
            (frame.loc[risk, "length"].to_numpy(int)
             == frame.loc[risk, "suite"].map(C.HORIZON_CAP).to_numpy(int)).sum()
        ),
        "mode_counts_by_suite": (
            frame.loc[risk & has_reason]
            .groupby(["suite", "primary_failure_reason"]).size().unstack(fill_value=0)
            .to_dict(orient="index")
        ),
    }
    return frame, report


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    for cohort in ("development_main", "external_8b"):
        frame, report = audit(cohort)
        reports[cohort] = report
        keep = [
            "row", "task_key", "suite", "task_name", "episode", "init_state_id",
            "flow_noise_seed", "length", "original_failure", "failure",
            "late_success_plus10_queries", "risk", "recorded_success",
            "primary_failure_reason", "failure_reason_confidence",
            "physics_validation_status", "physical_matched",
        ]
        frame[keep].to_csv(OUT / f"joined_{cohort}.csv", index=False)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))

    reports["inputs_sha256"] = {
        str(p.relative_to(C.PROJECT)): C.sha256(p)
        for p in (C.PHYSICAL, C.LABEL_PATHS["development_main"],
                  C.LABEL_PATHS["external_8b"], C.EXTERNAL_ALARMS, C.V7_ALARMS)
    }
    (OUT / "join_audit.json").write_text(
        json.dumps(reports, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
