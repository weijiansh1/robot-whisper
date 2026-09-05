#!/usr/bin/env python3
"""Build a compact nine-route summary and SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def select(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    matched = [row for row in rows if all(row.get(key) == value for key, value in criteria.items())]
    if len(matched) != 1:
        raise ValueError(f"selection {criteria} returned {len(matched)} rows")
    return matched[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    profile = load("results/profiles/build_summary.json")
    routes14 = load("results/routes_1_4/summary.json")
    full = load("results/full_moe_axes/summary.json")
    routes56 = load("results/routes_5_6/summary.json")
    route7 = load("results/route_7/summary.json")
    route8 = load("results/route_8/summary.json")
    route9 = load("results/route_9/summary.json")

    def onset(corpus: str, event: str, score: str) -> dict[str, Any]:
        row = select(routes14["rows"], corpus=corpus, event=event, score=score, lead=-2)
        return {
            "detection_auc": row["auc_detection"],
            "event_high_auc": row["auc_event_high"],
            "events": row["events"],
            "pairs": row["pairs"],
        }

    def hsmm_onset(corpus: str, event: str) -> float:
        return select(route7["matched_onset"], corpus=corpus, event=event, lead=-2)[
            "auc_detection"
        ]

    def hsmm_brier(event: str) -> dict[str, float]:
        row = select(route7["aggregate_brier"], corpus="grid50x8", event=event, horizon=2)
        return {
            "model": row["model_brier"],
            "within_task_prevalence_baseline": row[
                "within_task_prevalence_baseline_brier"
            ],
        }

    cross_calibration = {
        row["outcome"]: {
            "model_brier": row["model"]["candidate_brier"],
            "constant_brier": row["constant_train_rate"]["candidate_brier"],
            "model_log_loss": row["model"]["candidate_log_loss"],
            "constant_log_loss": row["constant_train_rate"]["candidate_log_loss"],
        }
        for row in route8["calibration_models"]
        if row["evaluation"] == "main16x32_to_grid50x8"
    }
    ltt = {row["event"]: row for row in route9["learn_then_test_style"]}
    summary = {
        "schema": "himoe.assurance.final_summary.v1",
        "date": "2026-09-05",
        "semantic_boundary": {
            "routing_evidence": "not an outcome probability",
            "mode_belief": "fitted state posterior, not a physical committor",
            "outcome_assurance": "only rows backed by repeated physical rollout/fork counts",
        },
        "inventory": {
            "tasks": profile["tasks"],
            "episodes": profile["episodes"],
            "query_rows": profile["rows"],
            "aggregate_axes": full["aggregate_axes"],
            "layer_graph_axes": full["layer_resolved_axes"],
            "flow_axes": full["flow_resolved_axes"],
            "recurrence_axes": full["recurrence_axes"],
            "total_axes": full["axes"],
            "q0_committor_cells": route8["q0_cells"],
        },
        "gpu_profile_build": {
            "cuda_visible_devices": profile["cuda_visible_devices"],
            "workers": profile["workers"],
            "training": False,
        },
        "routes": {
            "1_gate_commitment": {
                "fit": "none",
                "loop_q_minus_2": onset("grid50x8", "loop", "r1_commitment"),
                "static_q_minus_2": onset("grid50x8", "static", "r1_commitment"),
                "verdict": "interpretation axis only; not confidence",
            },
            "2_flow_coherence": {
                "fit": "none",
                "loop_route_acceleration_q_minus_2": onset(
                    "grid50x8", "loop", "r2_route_acceleration"
                ),
                "verdict": "retain as the strongest replicated loop evidence",
            },
            "3_token_expert_graph": {
                "fit": "none",
                "static_authority_q_minus_2": onset(
                    "grid50x8", "static", "r3_static_authority_loss"
                ),
                "static_lockin_q_minus_2": onset(
                    "grid50x8", "static", "r3_static_lockin"
                ),
                "verdict": "retain route lock-in; do not merge loop and static",
            },
            "4_healthy_manifold": {
                "fit": "healthy-reference Ledoit-Wolf; healthy selection uses labels",
                "loop_q_minus_2": onset("grid50x8", "loop", "r4_healthy_energy"),
                "static_q_minus_2": onset("grid50x8", "static", "r4_healthy_energy"),
                "verdict": "reject as primary route",
            },
            "5_perturbation_robustness": {
                "fit": "none",
                "local_sensitivity_to_progress": routes56["route5"][
                    "local_sensitivity_to_progress"
                ],
                "snapshot_dispersion_to_outcome_spread": routes56["route5"][
                    "snapshot_route_dispersion_to_outcome_spread"
                ],
                "verdict": "snapshot-level clue; no individual assurance",
            },
            "6_expert_disagreement": {
                "fit": "none; offline reconstruction",
                "routed_full_failure_controls": [
                    row
                    for row in routes56["route6"]["closed_loop_failure_controls"]
                    if row["family"] == "routed_full"
                ],
                "verdict": "negative outcome control; not confidence",
            },
            "7_hsmm": {
                "fit": "supervised statistical fit on main16x32",
                "external_q_minus_2_auc": {
                    "loop": hsmm_onset("grid50x8", "loop"),
                    "static": hsmm_onset("grid50x8", "static"),
                },
                "external_horizon_2_brier": {
                    "loop": hsmm_brier("loop"),
                    "static": hsmm_brier("static"),
                },
                "verdict": "reject for outcome forecasting",
            },
            "8_snapshot_fork_committor": {
                "fit": "Dirichlet posterior is count-based; optional mapping is fitted",
                "q0_cells": route8["q0_cells"],
                "cross_corpus_mapping": cross_calibration,
                "recovery_collection": route8["recovery_collection"],
                "verdict": "retain observed posterior tensor; reject learned MoE mapping",
            },
            "9_risk_control": {
                "fit": "calibration only",
                "unseen_task_ltt": {
                    event: {
                        "channel": row["selected"]["channel"],
                        "threshold": row["selected"]["threshold"],
                        "test_episode_fpr": row["test"]["episode_fpr"],
                        "test_q_minus_2_recall": row["test"]["recall_by_onset_minus_2"],
                        "test_scene_cluster_fpr": row["test"]["scene_cluster_fpr"],
                    }
                    for event, row in ltt.items()
                },
                "verdict": "static audit rule is useful; loop sensitivity is too low",
            },
        },
        "recommended_intrinsic_mechanisms": {
            "loop": "high denoising route acceleration / transition energy",
            "static": "low inter-query route distance plus late-layer token consensus",
            "generic_success_probability": "unsupported without new same-snapshot forks",
        },
        "probability_artifact": "results/route_8/assurance_tensor.csv",
        "report": "REPORT_ZH.md",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    final_path = RESULTS / "final_summary.json"
    final_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    candidates = [ROOT / "README.md", ROOT / "REPORT_ZH.md", ROOT / "requirements.txt"]
    for pattern in ("assurance/*.py", "experiments/*.py", "tests/*.py", "results/**/*"):
        candidates.extend(ROOT.glob(pattern))
    artifacts = []
    for path in sorted(set(candidates)):
        if not path.is_file() or path.name == "sealed_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "himoe.assurance.sealed_manifest.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "external_inputs": [
            "../analysis_moe_phenotype/features",
            "../analysis_moe_phenotype/events",
            "../analysis_committor/out/c1_cells.csv",
            "../himoe-route-capture/runs/fork-pilot-n32-client",
            "../himoe-route-capture/analysis/expert-activation-hidden-matched",
            "../trap-recovery-depth-20260904",
        ],
        "external_inputs_copied": False,
        "verification_command": "pytest -q",
    }
    manifest_path = RESULTS / "sealed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "final_summary": str(final_path),
                "manifest": str(manifest_path),
                "artifacts": len(artifacts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
