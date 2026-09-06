#!/usr/bin/env python3
"""Machine-readable manifest and a consolidated headline table for the bundle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
RESULTS = BUNDLE / "results"

STAGES = [
    {
        "script": "experiments/extract_arc_geometry.py",
        "reads": ["VLA_MUI_HUB/cache_new/HiMoE-VLA/<task>/<run_id>/server/routes.zarr::hb_router_probs",
                  "moe-v4-0904/results/layerwise_mobility/*.npz"],
        "writes": ["results/arc/{cohort}_arc.npy", "results/arc/{cohort}_arc_index.npz",
                   "results/arc/extraction_summary.json"],
        "outcomes_read": False,
    },
    {
        "script": "experiments/analyze_formation.py",
        "reads": ["results/arc/*", "moe-flow-semantics-0906/results/step_profiles/*",
                  "double-selete/trainfree/.../development_main_clean_labels.csv"],
        "writes": ["results/formation/step_profile.csv", "results/formation/arc_emergence.csv",
                   "results/formation/development_survival_auc.csv",
                   "results/formation/formation_summary.json"],
        "outcomes_read": "development only",
    },
    {
        "script": "experiments/run_detectors.py",
        "reads": ["results/arc/*", "moe-flow-semantics-0906/results/step_profiles/*", "both label CSVs"],
        "writes": ["results/detectors/development_grid.csv",
                   "results/detectors/development_selection.csv",
                   "results/detectors/external_detectors.csv",
                   "results/detectors/false_alarm_dependence.csv",
                   "results/detectors/external_first_alarms.npz",
                   "results/detectors/summary.json"],
        "outcomes_read": "development for selection, external once for evaluation",
    },
    {
        "script": "experiments/verify_controller_probes.py",
        "reads": ["routes.zarr (8 external tasks x 400 rows)", "results/formation/step_profile.csv"],
        "writes": ["results/controller_probes/*"],
        "outcomes_read": False,
    },
    {
        "script": "experiments/analyze_increment.py",
        "reads": ["results/arc/*", "step_profiles", "both label CSVs",
                  "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"],
        "writes": ["results/increment/*"],
        "outcomes_read": "post-hoc; external already opened",
    },
    {
        "script": "experiments/analyze_coupling_causal.py",
        "reads": ["step_profiles", "both label CSVs", "results/detectors/external_first_alarms.npz"],
        "writes": ["results/coupling/*"],
        "outcomes_read": "post-hoc; external already opened",
    },
    {
        "script": "tests/test_anchors_and_identities.py",
        "reads": ["results/detectors/*", "results/arc/*", "routes.zarr sample"],
        "writes": [],
        "outcomes_read": False,
    },
]

HEADLINE = [
    ("mobility_s9", "anchor / published baseline"),
    ("load_entropy_s9", "anchor / published baseline"),
    ("td_slope", "A primary: formation slope of token differentiation"),
    ("td_late_early", "A primary"),
    ("centred_slope", "A primary"),
    ("pc1corr_slope", "A primary: formation slope of arc ordering"),
    ("pc1share_slope", "A primary"),
    ("stretch_slope", "A primary"),
    ("pc1_index_corr_s9", "A secondary: endpoint arc ordering"),
    ("arc_stretch_s9", "A secondary"),
    ("fiedler_index_rho_s9", "A secondary"),
    ("along_state_energy_s9", "A secondary: the component the Schur complement discards"),
    ("along_state_fraction_s9", "A secondary"),
    ("pc1_state_cos_s9", "A secondary"),
    ("saa_slope", "A negative control"),
    ("state_mobility_s9", "B primary: the state channel"),
    ("state_mobility_mean", "B"),
    ("state_minus_action_mobility", "B"),
    ("state_over_action_logratio", "B"),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    external = pd.read_csv(RESULTS / "detectors/external_detectors.csv")
    dependence = pd.read_csv(RESULTS / "detectors/false_alarm_dependence.csv")
    coupling = pd.read_csv(RESULTS / "coupling/external_detectors.csv")
    coupling_dep = pd.read_csv(RESULTS / "coupling/false_alarm_dependence.csv")

    rows = []
    for quantity, role in HEADLINE:
        for mode in ("per_task", "global"):
            block = external[(external["quantity"] == quantity) & (external["mode"] == mode)]
            if block.empty or not bool(block.iloc[0].get("feasible", False)):
                continue
            row = block.iloc[0]
            dep = dependence[
                (dependence["head"] == quantity) & (dependence["mode"] == mode)
            ]
            rows.append(
                {
                    "quantity": quantity, "role": role, "mode": mode,
                    "representation": row["representation"], "direction": row["direction"],
                    "quantile": row["quantile"], "tp": int(row["tp"]), "fp": int(row["fp"]),
                    "precision": row["precision"], "risk_recall": row["risk_recall"],
                    "early_tp": int(row["low_prior_tp"]), "early_fp": int(row["low_prior_fp"]),
                    "mean_alarm_prior": row["mean_alarm_prior"], "lift": row["lift"],
                    "lift_ci_low": row["lift_ci_low"], "lift_ci_high": row["lift_ci_high"],
                    "median_alarm_chunk": row["median_alarm_chunk"],
                    "false_alarm_dependence_vs_mobility": float(dep["false_alarm_dependence"].iloc[0])
                    if len(dep) else float("nan"),
                    "tp_missed_by_mobility": int(dep["tp_missed_by_mobility"].iloc[0]) if len(dep) else -1,
                }
            )
    for _, row in coupling.iterrows():
        if not bool(row.get("feasible", False)):
            continue
        dep = coupling_dep[
            (coupling_dep["quantity"] == row["quantity"]) & (coupling_dep["mode"] == row["mode"])
        ]
        rows.append(
            {
                "quantity": row["quantity"], "role": "post-hoc: causal state/action coupling",
                "mode": row["mode"], "representation": row["representation"],
                "direction": row["direction"], "quantile": row["quantile"],
                "tp": int(row["tp"]), "fp": int(row["fp"]), "precision": row["precision"],
                "risk_recall": row["risk_recall"], "early_tp": int(row["low_prior_tp"]),
                "early_fp": int(row["low_prior_fp"]), "mean_alarm_prior": row["mean_alarm_prior"],
                "lift": row["lift"], "lift_ci_low": row["lift_ci_low"],
                "lift_ci_high": row["lift_ci_high"],
                "median_alarm_chunk": row["median_alarm_chunk"],
                "false_alarm_dependence_vs_mobility": float(dep["false_alarm_dependence"].iloc[0])
                if len(dep) else float("nan"),
                "tp_missed_by_mobility": int(dep["tp_missed_by_mobility"].iloc[0]) if len(dep) else -1,
            }
        )
    headline = pd.DataFrame(rows)
    headline.to_csv(RESULTS / "headline_table.csv", index=False)

    files = {}
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.suffix in {".csv", ".json", ".npz", ".md"}:
            files[str(path.relative_to(BUNDLE))] = {
                "bytes": path.stat().st_size, "sha256_16": digest(path)
            }

    manifest = {
        "schema": "himoe.state_channel.manifest.v1",
        "bundle": "moe-state-channel-0906",
        "question": "the state token as an observation channel, and the within-chunk "
                    "formation of the action-token arc",
        "prereg": "results/PREREG.md",
        "protocol": "imported verbatim from moe-flow-semantics-0906/experiments/"
                    "{protocol,detect_step_alarm}.py; width 4, confirmations 4",
        "anchors_reproduced": json.loads(
            (RESULTS / "detectors/summary.json").read_text()
        )["anchors"],
        "stages": STAGES,
        "cohorts": {
            "development_main": {"episodes": 14800, "risks": 487},
            "development_extra": {"episodes": 1200},
            "external_8b": {"episodes": 15600, "risks": 564},
        },
        "raw_footprint": "hb_router_probs read once in full for all three cohorts "
                         "(501,977 query rows), plus 8 external tasks x 400 rows re-read "
                         "for the controller-probe check.  CPU only, no GPU.",
        "rules": {
            "moe_only": "every score reads hb_router_probs alone",
            "causal": "a score at query q reads queries 0..q and all ten denoising steps of q",
            "train_free": "no fitted weights; selection from a predeclared grid on development",
            "task_identity_inside_any_score": False,
            "physical_failure_labels_used_only_for_analysis": True,
        },
        "files": files,
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    pd.set_option("display.width", 260)
    print(headline.to_string(index=False, float_format="%.3f"))
    print(f"\n{len(files)} result files recorded in results/manifest.json")


if __name__ == "__main__":
    main()
