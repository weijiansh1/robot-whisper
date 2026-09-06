#!/usr/bin/env python3
"""Write results/manifest.json: every artifact, its hash, and how to rebuild it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
RESULTS = BUNDLE / "results"

PIPELINE = [
    {"step": 1, "script": "experiments/extract_circuit_features.py",
     "command": "python extract_circuit_features.py --workers 12",
     "reads": ["VLA_MUI_HUB/cache_new/HiMoE-VLA/<suite>/<task>/<run_id>/server/routes.zarr:hb_router_probs",
               "moe-hb-front-back-0905/results/layer_graphs/*.npz (episode index only)"],
     "writes": ["results/circuit_features/*.npz", "results/circuit_features/extract_summary.json"],
     "labels_used": False},
    {"step": 2, "script": "experiments/diagnose_structure.py",
     "command": "python diagnose_structure.py",
     "writes": ["results/diagnostics/*"], "labels_used": False},
    {"step": 3, "script": "experiments/select_on_development.py",
     "command": "python select_on_development.py --workers 16",
     "writes": ["results/detectors/development_candidates.csv",
                "results/detectors/development_selection.json"],
     "labels_used": True, "cohort": "development_main only"},
    {"step": 4, "script": "experiments/evaluate_on_external.py",
     "command": "python evaluate_on_external.py",
     "writes": ["results/detectors/external_detectors.csv",
                "results/detectors/external_evaluation.json",
                "results/detectors/external_first_alarms.npz"],
     "labels_used": True, "cohort": "external_8b, single pass, no selection"},
    {"step": 5, "script": "experiments/summarise_results.py",
     "command": "python summarise_results.py",
     "writes": ["results/detectors/survival_conditioned_auc.csv",
                "results/detectors/external_detectors_with_ci.csv",
                "results/detectors/summary.json"]},
    {"step": 6, "script": "experiments/bootstrap_increment.py",
     "command": "python bootstrap_increment.py",
     "writes": ["results/detectors/bootstrap_increment.json"],
     "post_hoc": True},
]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    artifacts = []
    for path in sorted(RESULTS.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({
                "path": str(path.relative_to(BUNDLE)),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
    scripts = [{
        "path": str(path.relative_to(BUNDLE)),
        "sha256": digest(path),
    } for path in sorted((BUNDLE / "experiments").glob("*.py"))]

    manifest = {
        "schema": "himoe.circuit_analogy.manifest.v1",
        "bundle": "moe-circuit-analogy-0906",
        "question": "do classical electrical-network quantities on the MoE routing graph "
                    "carry mechanistically meaningful, predictive information",
        "graph_construction": {
            "object": "Schur complement of the sqrt-probability Gram over the 11 tokens "
                      "with respect to the state token, at the final flow step",
            "conductance": "w_ij = max(C_ij, 0) for i != j",
            "ground": "state token, leak g_i = 1 - C_ii = G_{0i}^2",
            "negative_entries": "kept as the positive part; discarded |mass| fraction is "
                                "4.7e-7 of total off-diagonal |mass| on both cohorts",
            "sign_of_partial_form": "partial_ij = C_ij / sqrt(C_ii C_jj) with C_ii > 0, so "
                                    "sign(partial) == sign(conditional); the same audit covers both",
        },
        "protocol_reuse": {
            "trailing_mean/persistent_score/row_max/quantile_higher/first_query/"
            "crossfit_thresholds/representations/QUANTILES":
                "moe-v4-0904/experiments/evaluate_layerwise_alarm_development.py",
            "survival_prior/prior_of/score_candidate/LOW_PRIOR/MAX_TIMELY_FPR":
                "moe-hb-front-back-0905/experiments/select_early_lock.py",
            "WIDTH/CONFIRMATIONS/MIN_LOW_PRIOR_PRECISION":
                "moe-hb-front-back-0905/experiments/compare_layers_early.py",
            "reproduction_check": "published baselines reproduce exactly: mobility|L12|low|0.975|"
                                  "global -> 195 TP / 17 FP, lift 1.7641; "
                                  "expert_load_effective_rank|L3|low|0.85|per_task -> "
                                  "370 TP / 93 FP, lift 1.5479",
        },
        "sampling": {
            "feature_extraction": "full cohorts, no subsampling",
            "detector_evaluation": "full cohorts, no subsampling",
            "rank_correlation_diagnostics": "200000 of 221781 development queries, seed 20260906",
            "bootstrap_increment": "500 cluster replicates over episodes, seed 20260906",
        },
        "pipeline": PIPELINE,
        "scripts": scripts,
        "artifacts": artifacts,
    }
    (RESULTS / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"{len(artifacts)} artifacts, {len(scripts)} scripts")


if __name__ == "__main__":
    main()
