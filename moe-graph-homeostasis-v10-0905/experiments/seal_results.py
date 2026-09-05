#!/usr/bin/env python3
"""Validate the completed experiment and seal its compact result bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


BUNDLE = Path(__file__).resolve().parent.parent
RESULTS = BUNDLE / "results"
V9_SUMMARY = BUNDLE.parent / "moe-dynamic-assurance-v9-0905/results/final_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contraction(summary: dict[str, Any], dataset: str, group: str, view: str) -> dict[str, Any]:
    row = next(
        row for row in summary["contraction_summary"]
        if row["dataset"] == dataset and row["layer_group"] == group and row["view"] == view
    )
    return {
        key: row[key]
        for key in (
            "median_ratio_d9_d0",
            "bootstrap_95_low",
            "bootstrap_95_high",
            "fraction_contracting",
            "median_d0",
            "median_d9",
        )
    }


def composite(
    phase: dict[str, Any], section: str, corpus: str, score: str, lead: int = -2
) -> dict[str, Any]:
    row = next(
        row for row in phase[section]
        if row["corpus"] == corpus and row["score"] == score and row["lead"] == lead
    )
    return {key: row[key] for key in ("auc_event_high", "events", "pairs")}


def phase_coordinate(
    phase: dict[str, Any], corpus: str, event: str, coordinate: str
) -> dict[str, Any]:
    row = next(
        row for row in phase["phase_summary"]
        if row["corpus"] == corpus and row["event"] == event and row["coordinate"] == coordinate
    )
    return {
        key: row[key]
        for key in (
            "events",
            "event_mean",
            "control_mean",
            "paired_difference",
            "bootstrap_95_low",
            "bootstrap_95_high",
        )
    }


def main() -> None:
    build = read_json(RESULTS / "structural_profiles/build_summary.json")
    fork = read_json(RESULTS / "fork_contraction/summary.json")
    phase = read_json(RESULTS / "phase_portrait/summary.json")
    v9 = read_json(V9_SUMMARY)
    if (build["tasks"], build["episodes"], build["rows"]) != (45, 18560, 305030):
        raise ValueError("profile inventory changed")
    if build["total_dynamic_axes"] != 218 or build["outcomes_loaded"]:
        raise ValueError("profile semantics changed")
    if fork["datasets"]["fork_pilot_n32"]["snapshots"] != 20:
        raise ValueError("pilot inventory changed")
    if fork["datasets"]["rolling_star_k16"]["snapshots"] != 22:
        raise ValueError("rolling inventory changed")

    outcome_success = [
        row for row in fork["outcome_summary"]
        if row["dataset"] == "rolling_star_k16" and row["outcome"] == "success"
    ]
    summary = {
        "schema": "himoe.graph_homeostasis.final.v1",
        "date": "2026-09-05",
        "training": False,
        "profile": {
            "tasks": build["tasks"],
            "episodes": build["episodes"],
            "query_rows": build["rows"],
            "raw_router_shape": build["raw_router_shape"],
            "graph_views": build["graph_views"],
            "within_axes": build["within_feature_count"],
            "query_axes": build["query_feature_count"],
            "total_axes": build["total_dynamic_axes"],
            "profile_bytes": build["output_bytes"],
            "cuda_visible_devices": build["cuda_visible_devices"],
            "workers": build["workers"],
            "outcomes_loaded": build["outcomes_loaded"],
            "fitted_weights": build["fitted_weights"],
            "task_conditioned_features": build["task_conditioned_features"],
            "max_float16_quantization_error": max(
                row["max_float16_quantization_error"] for row in build["records"]
            ),
        },
        "same_snapshot_contraction": {
            "semantics": fork["interpretation"],
            "not_measured": fork["not_measured"],
            "pilot_front_action": contraction(fork, "fork_pilot_n32", "front", "edge_action"),
            "rolling_front_action": contraction(fork, "rolling_star_k16", "front", "edge_action"),
            "pilot_back_token_relation": contraction(fork, "fork_pilot_n32", "back", "token_gram"),
            "rolling_back_token_relation": contraction(fork, "rolling_star_k16", "back", "token_gram"),
            "replicated_contractions": fork["replicated_contractions"],
        },
        "outcome_boundary": {
            "rolling_mixed_success_snapshots": fork["datasets"]["rolling_star_k16"][
                "mixed_success_snapshots"
            ],
            "success_axes": [
                {
                    key: row[key]
                    for key in (
                        "axis",
                        "mean_within_snapshot_auc",
                        "auc_bootstrap_95_low",
                        "auc_bootstrap_95_high",
                        "auc_wilcoxon_p_vs_half",
                    )
                }
                for row in outcome_success
            ],
            "conclusion": "q0 graph stability is not a calibrated or significant success predictor",
        },
        "q_minus_2": {
            "prior_fixed": {
                "loop": {
                    corpus: composite(
                        phase, "fixed_composites", corpus, "loop_flow_instability"
                    )
                    for corpus in ("main16x32", "grid50x8")
                },
                "static": {
                    corpus: composite(
                        phase, "fixed_composites", corpus, "static_rigid_consensus"
                    )
                    for corpus in ("main16x32", "grid50x8")
                },
            },
            "main_rank_locked": {
                "semantics": phase["locked_discovery_method"],
                "loop": {
                    corpus: composite(
                        phase,
                        "locked_discovery_composites",
                        corpus,
                        "main_ranked_loop_top4",
                    )
                    for corpus in ("main16x32", "grid50x8")
                },
                "static": {
                    corpus: composite(
                        phase,
                        "locked_discovery_composites",
                        corpus,
                        "main_ranked_static_top4",
                    )
                    for corpus in ("main16x32", "grid50x8")
                },
            },
            "phase_coordinates": {
                event: {
                    corpus: {
                        coordinate: phase_coordinate(phase, corpus, event, coordinate)
                        for coordinate in ("settling", "responsiveness", "coherence")
                    }
                    for corpus in ("main16x32", "grid50x8")
                }
                for event in ("loop", "static")
            },
            "best_replicated_axes": {
                event: phase["post_hoc_axis_audit"][event][0]
                for event in ("loop", "static")
            },
        },
        "comparison_to_v9": {
            "profile_axis_reduction": v9["profile"]["total_dynamic_axes"] / build["total_dynamic_axes"],
            "profile_storage_reduction": v9["profile"]["profile_bytes"] / build["output_bytes"],
            "loop_grid": {
                "v8_best_route_acceleration": v9["comparison_to_v8"]["loop_external_q_minus_2"][
                    "v8_best_route_acceleration"
                ],
                "v9_full_dynamic_composite": v9["comparison_to_v8"]["loop_external_q_minus_2"][
                    "v9_full_dynamic_composite"
                ],
                "v10_main_ranked_graph_top4": composite(
                    phase,
                    "locked_discovery_composites",
                    "grid50x8",
                    "main_ranked_loop_top4",
                )["auc_event_high"],
            },
            "static_grid": {
                "v9_best_lag4": v9["comparison_to_v8"]["static_external_q_minus_2"][
                    "v9_state_lag4_axis"
                ],
                "v10_main_ranked_graph_top4": composite(
                    phase,
                    "locked_discovery_composites",
                    "grid50x8",
                    "main_ranked_static_top4",
                )["auc_event_high"],
            },
            "conclusion": (
                "graph modeling improves compactness and mechanism resolution, not loop discrimination; "
                "the static result recovers the same cross-query lock-in mechanism"
            ),
        },
        "semantic_boundary": {
            "routing_evidence": "all v10 graph metrics and AUCs",
            "structural_stability": "same-snapshot independent-noise ensemble contraction",
            "outcome_assurance": "not produced; requires repeated physical fork outcome calibration",
            "static_timing": (
                "q-2 is confirmation of an accumulating stationary run because onset uses consecutive windows"
            ),
            "confirmation": "grid50x8 is direction-locked but previously analyzed, not a pristine holdout",
        },
    }
    final_path = RESULTS / "final_summary.json"
    final_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    excluded = {RESULTS / "sealed_manifest.json"}
    artifacts = []
    for path in sorted(BUNDLE.rglob("*")):
        if not path.is_file() or path in excluded:
            continue
        if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(BUNDLE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "himoe.graph_homeostasis.sealed_manifest.v1",
        "files": len(artifacts),
        "bytes": sum(row["bytes"] for row in artifacts),
        "artifacts": artifacts,
    }
    (RESULTS / "sealed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "files": manifest["files"],
                "bytes": manifest["bytes"],
                "profile": summary["profile"],
                "comparison_to_v9": summary["comparison_to_v9"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
