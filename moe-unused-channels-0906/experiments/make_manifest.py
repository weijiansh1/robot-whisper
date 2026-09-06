#!/usr/bin/env python3
"""Write results/manifest.json: what was read, what was written, and the checksums."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import protocol as P


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    inputs = [
        "VLA_MUI_HUB/cache_new/HiMoE-VLA/<suite>/<task>/right-50x8-20260903/server/routes.zarr",
        "VLA_MUI_HUB/cache_new/HiMoE-VLA/<suite>/<task>/right-50x8b-20260903/server/routes.zarr",
        "moe-v4-0904/results/layerwise_mobility/{main_reference,extra_reference,external_8b}.npz",
        "moe-hb-front-back-0905/results/layer_graphs/{development_main,development_extra,external_8b}.npz",
        "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz",
        "moe-hb-front-back-0905/results/frame_survey/false_alarm_dependence_global.csv",
        "moe-flow-semantics-0906/results/step_profiles/{cohort}_{metrics,mobility}.npy",
        "double-selete/trainfree/results/timeout_extension_plus10/{development_main,external_8b}_clean_labels.csv",
    ]
    reused = {
        "moe-v4-0904/experiments/evaluate_layerwise_alarm_development.py": [
            "trailing_mean", "persistent_score", "row_max", "quantile_higher",
            "first_query", "crossfit_thresholds", "representations", "QUANTILES",
        ],
        "moe-hb-front-back-0905/experiments/select_early_lock.py": [
            "survival_prior", "prior_of", "score_candidate", "suite_of",
            "MAX_TIMELY_FPR", "LOW_PRIOR",
        ],
        "moe-hb-front-back-0905/experiments/compare_layers_early.py": [
            "WIDTH", "CONFIRMATIONS", "MIN_LOW_PRIOR_PRECISION",
        ],
        "moe-audit-0906/experiments/recheck_stratification.py": [
            "weighted-pooled Mann-Whitney stratified AUC, MIN_STRATUM=30, CHUNKS",
        ],
        "VLA_MUI_HUB/moe-physical-failure-dynamics/analyze.py": [
            "top4_churn (Jaccard distance on top-4 sets)",
        ],
        "VLA_MUI_HUB/moe-failure-alarm/analyze.py": [
            "top4_mask / mask_churn (bitmask form of the same statistic)",
        ],
        "moe-flow-semantics-0906/experiments/extract_flow_steps.py": [
            "raw-Zarr extraction pattern, batch lead-row seam handling",
        ],
    }
    order = [
        ("experiments/extract_discrete_channels.py", "62 s, 24 CPU workers, no GPU"),
        ("experiments/evaluate_discrete_churn.py", "8 min 39 s, single CPU process"),
        ("experiments/analyse_as_router.py", "38 s"),
        ("experiments/task_matched_lift.py", "43 s"),
        ("experiments/make_manifest.py", "seconds"),
    ]

    results = BUNDLE / "results"
    files = {}
    for path in sorted(results.rglob("*")):
        if path.is_dir() or path.suffix == ".npy" or path.name == ".gitignore":
            continue
        files[str(path.relative_to(BUNDLE))] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
    for path in sorted((BUNDLE / "experiments").glob("*.py")):
        files[str(path.relative_to(BUNDLE))] = {
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }

    manifest = {
        "schema": "himoe.unused_channels.manifest.v1",
        "bundle": "moe-unused-channels-0906",
        "written": "2026-09-06",
        "question": (
            "the four logged routing channels never opened by this project: "
            "hb_expert_ids, hb_selected_prob, hb_entropy, as_probs/as_expert_ids"
        ),
        "prereg": "results/PREREG.md, written before any external outcome was read here",
        "anchors_asserted_every_run": [
            "mobility|L12|low|q0.975|global = 195 TP / 17 FP, lift 1.7641",
            "mobility|L2|low|q0.700|per_task = 272 TP / 57 FP",
            "expert_load_effective_rank|L3|low|q0.85|per_task = 370 TP / 93 FP, lift 1.5479",
        ],
        "coverage": {
            "query_rows_read": 501977,
            "router_cells_verified": 441739760,
            "subsampling": "none",
            "cohorts": {
                "development_main": "14800 episodes / 487 risks / 221781 valid chunks",
                "development_extra": "1200 episodes / reference only / 31941 valid chunks",
                "external_8b": "15600 episodes / 564 risks / 248255 valid chunks",
            },
        },
        "inputs_read": inputs,
        "code_reused_not_reimplemented": reused,
        "run_order": [{"script": s, "cost": c} for s, c in order],
        "outputs_not_checksummed": [
            "results/channels/*_quantities.npy (402 MB of memory-mapped per-chunk "
            "quantities; regenerate with extract_discrete_channels.py)"
        ],
        "negative_controls": {
            "null_iid": "i.i.d. U(0,1) per (episode, chunk, layer), seeded 20260906",
            "null_episode_constant": "one i.i.d. draw per episode held over chunks",
            "as_entropy": (
                "not designed as a control but functions as the strongest one: "
                "bit-for-bit constant inside every episode, range exactly 0.0"
            ),
        },
        "files": files,
    }
    P.write_json(results / "manifest.json", manifest)
    print(json.dumps({"files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
