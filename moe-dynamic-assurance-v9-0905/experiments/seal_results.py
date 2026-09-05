#!/usr/bin/env python3
"""Create and verify the final dynamic-MoE summary and artifact manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


BUNDLE = Path(__file__).resolve().parent.parent
RESULTS = BUNDLE / "results"
V8 = BUNDLE.parent / "moe-assurance-v8-0905"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composite(rows: list[dict[str, str]], corpus: str, event: str, score: str, lead: int) -> dict:
    row = next(
        row
        for row in rows
        if row["corpus"] == corpus
        and row["event"] == event
        and row["score"] == score
        and int(row["lead"]) == lead
    )
    return {
        "auc_event_high": float(row["auc_event_high"]),
        "events": int(row["events"]),
        "pairs": int(row["pairs"]),
    }


def axis(rows: list[dict[str, str]], corpus: str, event: str, name: str, lead: int = -2) -> dict:
    row = next(
        row
        for row in rows
        if row["corpus"] == corpus
        and row["event"] == event
        and row["axis"] == name
        and int(row["lead"]) == lead
    )
    event_high = float(row["auc_event_high"])
    return {
        "auc_event_high": event_high,
        "auc_detection": max(event_high, 1.0 - event_high),
        "events": int(row["events"]),
        "pairs": int(row["pairs"]),
    }


def risk(summary: dict, event: str, channel: str) -> dict:
    row = next(
        row
        for row in summary["unseen_task_conformal"]
        if row["event"] == event and row["channel"] == channel
    )
    return {
        "channel": channel,
        "threshold": row["threshold"],
        "calibration_support": row["support"],
        "test": row["test"],
    }


def build_summary() -> dict:
    build = read_json(RESULTS / "dynamic_profiles/build_summary.json")
    evaluation = read_json(RESULTS / "dynamic_evaluation/summary.json")
    risk_summary = read_json(RESULTS / "dynamic_risk/summary.json")
    with (RESULTS / "dynamic_evaluation/composite_matched_auc.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        composite_rows = list(csv.DictReader(handle))
    with (RESULTS / "dynamic_evaluation/all_axis_matched_auc.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        axis_rows = list(csv.DictReader(handle))

    v8 = read_json(V8 / "results/final_summary.json")
    v8_axes = read_json(V8 / "results/full_moe_axes/summary.json")
    v8_loop_composite = next(row for row in v8_axes["composites"] if row["event"] == "loop")
    v8_static_composite = next(row for row in v8_axes["composites"] if row["event"] == "static")
    v8_risk = {
        row["event"]: row for row in read_json(V8 / "results/route_9/summary.json")["learn_then_test_style"]
    }

    loop_axis = "graph_level|L15|f1|top1_mass"
    static_axis = "query_aggregate|late|state|recurrence_lag4"
    output = {
        "schema": "himoe.dynamic_assurance.final_summary.v1",
        "date": "2026-09-05",
        "training": False,
        "profile": {
            "source": build["source"],
            "raw_router_shape": build["raw_router_shape"],
            "tasks": build["tasks"],
            "episodes": build["episodes"],
            "query_rows": build["rows"],
            "within_query_axes": build["within_feature_count"],
            "across_query_axes": build["query_feature_count"],
            "total_dynamic_axes": build["total_dynamic_axes"],
            "profile_bytes": build["output_bytes"],
            "max_float16_quantization_error": max(
                record["max_float16_quantization_error"] for record in build["records"]
            ),
            "cuda_visible_devices": build["cuda_visible_devices"],
            "workers": build["workers"],
            "outcomes_loaded_during_profile_build": build["outcomes_loaded"],
            "task_conditioned_features": build["task_conditioned_features"],
        },
        "evaluation_protocol": {
            "discovery": "main16x32 q-2 determines directions and selected axes",
            "confirmation": evaluation["external_confirmation"],
            "matched_control": evaluation["matched_control"],
            "coverage_policy": evaluation["coverage_policy"],
            "fitted_weights": evaluation["fitted_weights"],
            "classifier_trained": evaluation["classifier_trained"],
        },
        "loop": {
            "predeclared_q_minus_2": {
                corpus: composite(composite_rows, corpus, "loop", "predeclared_loop", -2)
                for corpus in ("main16x32", "grid50x8")
            },
            "discovery_full_q_minus_2": {
                corpus: composite(composite_rows, corpus, "loop", "discovery_full_loop", -2)
                for corpus in ("main16x32", "grid50x8")
            },
            "discovery_history_q_minus_2": {
                corpus: composite(composite_rows, corpus, "loop", "discovery_history_loop", -2)
                for corpus in ("main16x32", "grid50x8")
            },
            "cross_corpus_stable_axis": {
                "axis": loop_axis,
                "direction": "high",
                "main16x32": axis(axis_rows, "main16x32", "loop", loop_axis),
                "grid50x8": axis(axis_rows, "grid50x8", "loop", loop_axis),
            },
            "interpretation": (
                "loop combines elevated within-flow route motion/curvature with stronger early-flow "
                "late-layer commitment; individual layer, flow-step, and expert identities are selection-unstable"
            ),
            "verdict": "retain as graded routing evidence; reject as a low-FPR hard alarm",
        },
        "static": {
            "predeclared_q_minus_2": {
                corpus: composite(composite_rows, corpus, "static", "predeclared_static", -2)
                for corpus in ("main16x32", "grid50x8")
            },
            "discovery_diverse_q_minus_2": {
                corpus: composite(composite_rows, corpus, "static", "discovery_full_static", -2)
                for corpus in ("main16x32", "grid50x8")
            },
            "strongest_replicated_axis": {
                "axis": static_axis,
                "direction": "low",
                "main16x32": axis(axis_rows, "main16x32", "static", static_axis),
                "grid50x8": axis(axis_rows, "grid50x8", "static", static_axis),
            },
            "interpretation": (
                "the state-token route collapses onto nearly the same late-layer expert distribution "
                "across queries while route acceleration and jerk also shrink"
            ),
            "timing_caution": evaluation["static_timing_caution"],
            "verdict": "retain as a static-confirmation guard, not a future-success probability",
        },
        "risk_control": {
            "unseen_task_predeclared_loop": risk(
                risk_summary, "loop", "predeclared_loop"
            ),
            "unseen_task_predeclared_static": risk(
                risk_summary, "static", "predeclared_static"
            ),
            "semantics": risk_summary["semantics"],
            "guarantee_boundary": risk_summary["guarantee_boundary"],
        },
        "comparison_to_v8": {
            "candidate_axes": {"v8": v8["inventory"]["total_axes"], "v9": build["total_dynamic_axes"]},
            "loop_external_q_minus_2": {
                "v8_best_route_acceleration": v8["routes"]["2_flow_coherence"][
                    "loop_route_acceleration_q_minus_2"
                ]["event_high_auc"],
                "v8_discovery_composite": v8_loop_composite["external_evaluation"]["grid50x8"]["-2"][
                    "auc_event_high"
                ],
                "v9_full_dynamic_composite": composite(
                    composite_rows, "grid50x8", "loop", "discovery_full_loop", -2
                )["auc_event_high"],
                "v9_history_composite_reduced_coverage": composite(
                    composite_rows, "grid50x8", "loop", "discovery_history_loop", -2
                ),
            },
            "static_external_q_minus_2": {
                "v8_lag1_lockin": v8["routes"]["3_token_expert_graph"]["static_lockin_q_minus_2"][
                    "event_high_auc"
                ],
                "v8_discovery_composite": v8_static_composite["external_evaluation"]["grid50x8"]["-2"][
                    "auc_event_high"
                ],
                "v9_state_lag4_axis": axis(axis_rows, "grid50x8", "static", static_axis)[
                    "auc_detection"
                ],
                "v9_diverse_composite": composite(
                    composite_rows, "grid50x8", "static", "discovery_full_static", -2
                )["auc_event_high"],
            },
            "unseen_task_risk": {
                "v8_loop_ltt": v8_risk["loop"]["test"],
                "v8_static_ltt": v8_risk["static"]["test"],
            },
            "conclusion": (
                "the full dynamics improve resolution and expose stable mechanisms, but do not justify "
                "a claim that a denser score universally outperforms v8"
            ),
        },
        "runtime": {
            "feature_api": "dynamic.features.compute_dynamic_history",
            "score_api": "dynamic.scoring.DynamicScoreReference",
            "reference_artifact": "results/dynamic_evaluation/score_reference.npz",
        },
        "semantic_boundary": {
            "routing_evidence": "the v9 axes and scores in this bundle",
            "risk_rule": "episode-level calibrated thresholds; not individual outcome probability",
            "outcome_assurance": (
                "not produced in v9; only repeated physical snapshot-fork counts may populate it, "
                "as in v8 results/route_8/assurance_tensor.csv"
            ),
        },
    }
    (RESULTS / "final_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def seal() -> dict:
    excluded = {"results/sealed_manifest.json"}
    files = []
    for path in sorted(BUNDLE.rglob("*")):
        relative = path.relative_to(BUNDLE).as_posix()
        if (
            not path.is_file()
            or relative in excluded
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
        ):
            continue
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest = {
        "schema": "himoe.dynamic_assurance.sealed_manifest.v1",
        "files": files,
        "artifact_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
    }
    (RESULTS / "sealed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify() -> dict:
    manifest = read_json(RESULTS / "sealed_manifest.json")
    failures = []
    for record in manifest["files"]:
        path = BUNDLE / record["path"]
        if not path.is_file():
            failures.append({"path": record["path"], "reason": "missing"})
        elif path.stat().st_size != record["bytes"]:
            failures.append({"path": record["path"], "reason": "size"})
        elif sha256(path) != record["sha256"]:
            failures.append({"path": record["path"], "reason": "sha256"})
    if failures:
        raise RuntimeError(json.dumps(failures, indent=2))
    return {
        "verified": True,
        "artifact_count": manifest["artifact_count"],
        "total_bytes": manifest["total_bytes"],
    }


def main() -> None:
    args = parse_args()
    if args.verify:
        print(json.dumps(verify(), indent=2))
        return
    summary = build_summary()
    manifest = seal()
    print(
        json.dumps(
            {
                "schema": summary["schema"],
                "dynamic_axes": summary["profile"]["total_dynamic_axes"],
                "artifact_count": manifest["artifact_count"],
                "sealed_bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
