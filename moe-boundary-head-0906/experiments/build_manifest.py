"""Assemble results/manifest.json: what was produced, by what, and the verdict."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from common import OUT

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent

ORDER = [
    ("extract_boundary.py", "Raw zarr -> the twelve cross-query series "
                            "(boundary / within-chunk / within-query chord) "
                            "for all three cohorts, float64.",
     ["{cohort}_boundary.npz", "extract_boundary.log"]),
    ("reproduce_exploratory.py", "Reproduces the exploratory magnitudes and the "
                                 "external AUC table from the recomputed data, "
                                 "then extends it to development and legacy.",
     ["auc_table.csv", "reproduce_exploratory.json"]),
    ("decompose_axis.py", "Is the seam a new axis?  Decomposes it into the v7 "
                          "freeze axis and the v8 flow axis and scores the "
                          "residual.",
     ["triangle_magnitudes.csv", "seam_vs_step9.csv", "seam_decomposition.csv",
      "seam_residual_auc.csv", "decompose_axis.json"]),
    ("pairwise_scan.py", "All 28 pairwise rank differences x 3 chunks, on all "
                         "three cohorts, with task-clustered CIs on the gain "
                         "and three surrogates.",
     ["pairwise_development.csv", "pairwise_all_cohorts.csv",
      "pairwise_nulls.csv", "pairwise_scan.json", "pairwise.log"]),
    ("sweep_head.py", "The development-only design sweep, plus the v8-dial "
                      "control.  Selection at lead >= 0.",
     ["sweep_development.csv", "sweep_summary.json", "v8_dial_development.csv",
      "dial_comparison_development.csv", "family_comparison_development.csv",
      "sweep.log"]),
    ("split_half.py", "Task split-half: how much of the sweep winner is search "
                      "noise.  Freezes the spec.",
     ["split_half.csv", "frozen_spec.json", "full_development_configs.csv",
      "split_half.log"]),
    ("freeze_and_score.py", "The frozen spec applied once to external_8b and "
                            "legacy_main16x32.  Per suite, per task, LOTO, "
                            "nulls, length negative control.",
     ["frozen_scores.csv", "frozen_summary.json", "boundary_head_alarms.npz",
      "per_suite.csv", "per_task.csv", "leave_one_task_out.csv", "nulls.csv",
      "negative_control_length.csv", "freeze.log"]),
    ("null_equivalence.py", "How many false alarms a surrogate head needs to "
                            "buy the same true positives.",
     ["null_equivalence.csv", "null_equivalence.json", "null_equivalence.log"]),
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main() -> None:
    spec = json.loads((OUT / "frozen_spec.json").read_text())
    frozen = json.loads((OUT / "frozen_summary.json").read_text())
    scan = json.loads((OUT / "pairwise_scan.json").read_text())
    decomp = json.loads((OUT / "decompose_axis.json").read_text())
    sweep = json.loads((OUT / "sweep_summary.json").read_text())
    nulleq = json.loads((OUT / "null_equivalence.json").read_text())
    task = pd.read_csv(OUT / "per_task.csv")

    stages = []
    for script, why, outputs in ORDER:
        files = []
        for pattern in outputs:
            names = ([pattern.format(cohort=c) for c in
                      ("development_main", "external_8b", "legacy_main16x32")]
                     if "{cohort}" in pattern else [pattern])
            for name in names:
                path = OUT / name
                if path.exists():
                    files.append({"file": name, "bytes": path.stat().st_size,
                                  "sha256_16": digest(path)})
        stages.append({"script": f"experiments/{script}", "purpose": why,
                       "outputs": files})

    top = task.nlargest(1, "d_tp").iloc[0]
    manifest = {
        "bundle": "moe-boundary-head-0906",
        "question": "Does the chunk-boundary routing reshuffle earn a head in "
                    "the v8 intrinsic guard?",
        "verdict": "No, not as a general head.",
        "verdict_detail": {
            "headline": "v8 932/1358 at 126 FP -> v8+head 987/1358 at 139 FP, "
                        "net +55 TP / +13 FP = 4.23 TP per FP at lead >= 4, "
                        "which clears the ~2:1 bar on its face",
            "why_it_fails_anyway": [
                "87.3%% of the net (+48 of +55 TP) comes from one task, "
                "%s, which holds 505 of the 1,358 failures and which v8 "
                "already recalls at 0.80.  Removing it leaves +7 TP / +2 FP "
                "over the remaining 853 failures." % top.task,
                "The two suites v8 is worst on gain exactly nothing: object "
                "stays 41/81 and spatial stays 118/315, +0 TP / +0 FP each.",
                "Tasks with v8 recall <= 0.4 (10 tasks, 253 failures) gain "
                "+4 TP; tasks with v8 recall >= 0.8 (18 tasks, 819 failures) "
                "gain +50 TP.  It helps what v8 already handles.",
                "Development split-half: the same selection rule yields a "
                "pooled 3.16 TP/FP on the half it was chosen on and 0.06 TP/FP "
                "on the held-out half, with 15 different winners in 24 "
                "selections.  The development frontier is search noise.",
            ],
            "structural_finding": (
                "The state half of the boundary 2x2 is not a boundary "
                "measurement.  The state token's route is constant across the "
                "ten denoising steps (mean within-query chord 0.00005 against a "
                "seam of 0.182), so the state seam and adjacent-query state "
                "mobility are the same series: Pearson 1.000000 on "
                "development and external, bit-identical on legacy.  The "
                "action seam is genuinely distinct but is explained by the v7 "
                "freeze axis plus the v8 flow axis at R^2 0.80 (front) and "
                "0.93 (back); its headline q9 AUC of 0.660 falls to 0.513 in "
                "the residual."),
        },
        "frozen_spec": spec,
        "headline": frozen["headline"],
        "per_suite": frozen["per_suite"],
        "weak_tasks": frozen["weak_tasks"],
        "strong_tasks": frozen["strong_tasks"],
        "leave_one_task_out_min_d_tp": frozen["loto_min_d_tp"],
        "top_task_share_of_gain": frozen["top_task_share"],
        "pairwise_scan": {
            "argmax_agrees_across_cohorts": scan["argmax_agrees"],
            "argmax_per_cohort": scan["argmax_per_cohort"],
            "n_significant_gains": scan["n_sig"],
            "n_cells": scan["n_cells"],
            "mean_gain": scan["mean_gain"],
            "nulls": scan["nulls"],
        },
        "decomposition": {
            "state_chord_over_seam": {
                c: decomp["state_moves_on_denoising_axis"][c]
                for c in decomp["state_moves_on_denoising_axis"]},
            "seam_vs_step9_pearson": decomp["seam_vs_step9_pearson"],
            "n_residual_significant": decomp["n_residual_significant"],
            "n_residual_cells": decomp["n_residual_cells"],
        },
        "selection_intensity": {
            "n_configurations_swept": sweep["n_configs"],
            "n_configurations_selectable": spec["n_configs_enumerated"],
            "split_half": spec["split_half"],
        },
        "null_equivalence": nulleq["surrogates"],
        "protocol": {
            "cap_free": True,
            "selection_lead": 0,
            "reported_leads": [0, 2, 4, 8, 12],
            "headline_lead": 4,
            "thresholds": "order statistics of the unlabeled development "
                          "scores, method='lower', applied with >= / <=",
            "rank_reference": "development_main, frozen; no test cohort enters "
                              "any transform",
            "length_is_baseline": False,
            "development_iterations": 3,
            "saw_external_before_design_was_fixed": True,
        },
        "stages": stages,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                   default=float))
    n = sum(len(s["outputs"]) for s in stages)
    print("写出 manifest.json：%d 个阶段，%d 个产物" % (len(stages), n))


if __name__ == "__main__":
    main()
