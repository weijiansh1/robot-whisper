#!/usr/bin/env python3
"""Retrospectively replay the frozen v2 dynamics alarm on saved routes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from moe_dynamics_online_selector import (
    DynamicsCalibration,
    MoeDynamicsOnlineAlarm,
    SELECTOR_VERSION,
)
from moe_only_online_selector import HealthySequenceBank


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz"
)
DEFAULT_CALIBRATION = (
    PACKAGE_ROOT / "results/moe_dynamics_online_alarm/calibration_v2/calibration.json"
)
DEFAULT_AUDIT = (
    PACKAGE_ROOT / "results/moe_only_online_alarm/failure_type_audit/episode_audit.csv"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_dynamics_online_alarm/offline_replay"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(hits: int, episodes: int) -> dict[str, Any]:
    return {
        "hits": hits,
        "episodes": episodes,
        "rate": None if not episodes else hits / episodes,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in ("target_missed_grasp_proxy", "other_failure", "success"):
        selected = [row for row in rows if row["failure_family"] == family]
        output[family] = ratio(
            sum(bool(row["formal_alarm"]) for row in selected), len(selected)
        )
    failures = [row for row in rows if row["failure_family"] != "success"]
    output["all_failures"] = ratio(
        sum(bool(row["formal_alarm"]) for row in failures), len(failures)
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    reference = args.healthy_reference.resolve()
    bank = HealthySequenceBank.load(reference)
    calibration = DynamicsCalibration.load(
        args.calibration.resolve(), reference, bank
    )
    with args.audit.open(newline="", encoding="utf-8") as stream:
        audit_rows = list(csv.DictReader(stream))

    query_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for audit in audit_rows:
        trajectory = PACKAGE_ROOT / audit["trajectory"]
        with np.load(trajectory, allow_pickle=False) as archive:
            routes = np.asarray(archive["hb_router_probs"], dtype=np.float32)
        selector = MoeDynamicsOnlineAlarm(bank, calibration)
        raw_queries: list[int] = []
        alarm_queries: list[int] = []
        scores: list[float] = []
        for route in routes:
            decision = selector.update(route)
            if decision is None:
                continue
            item = decision.to_dict()
            item["matched_reference_queries"] = json.dumps(
                item["matched_reference_queries"], separators=(",", ":")
            )
            query_rows.append(
                {
                    "run_role": audit["run_role"],
                    "episode_index": int(audit["episode_index"]),
                    "init_state_id": int(audit["init_state_id"]),
                    "failure_family": audit["failure_family"],
                    **item,
                }
            )
            scores.append(decision.normalized_acceleration_excess)
            if decision.raw_reject:
                raw_queries.append(decision.query)
            if decision.alarm:
                alarm_queries.append(decision.query)
        missed_queries = [int(value) for value in json.loads(audit["missed_grasp_queries"])]
        signed_deltas = [
            alarm - missed
            for alarm in alarm_queries
            for missed in missed_queries
        ]
        nearest_delta = (
            None
            if not signed_deltas
            else min(signed_deltas, key=lambda value: (abs(value), value))
        )
        episode_rows.append(
            {
                "run_role": audit["run_role"],
                "evaluation_role": (
                    "calibration_source_contaminated"
                    if audit["run_role"] == "preliminary_gpu4"
                    else "retrospective_feature_development"
                    if audit["run_role"] == "gpu5_extended"
                    else "legacy_heldout_for_v1_not_v2"
                ),
                "episode_index": int(audit["episode_index"]),
                "init_state_id": int(audit["init_state_id"]),
                "success": audit["success"] == "True",
                "failure_family": audit["failure_family"],
                "queries": len(routes),
                "score_max": max(scores),
                "raw_reject_queries": json.dumps(raw_queries, separators=(",", ":")),
                "alarm_queries": json.dumps(alarm_queries, separators=(",", ":")),
                "formal_alarm": bool(alarm_queries),
                "missed_grasp_queries": audit["missed_grasp_queries"],
                "nearest_alarm_minus_missed_query": nearest_delta,
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "query_decisions.csv", query_rows)
    write_csv(output / "episode_decisions.csv", episode_rows)
    by_run = {
        role: aggregate([row for row in episode_rows if row["run_role"] == role])
        for role in ("preliminary_gpu4", "heldout_gpu4", "gpu5_extended")
    }
    test_like = [
        row for row in episode_rows if row["run_role"] != "preliminary_gpu4"
    ]
    target_timing = [
        row["nearest_alarm_minus_missed_query"]
        for row in test_like
        if row["failure_family"] == "target_missed_grasp_proxy"
    ]
    summary = {
        "schema": "himoe.moe_dynamics_selector_replay.v1",
        "selector_version": SELECTOR_VERSION,
        "training": False,
        "failure_labels_used_to_set_threshold": False,
        "physical_inputs_to_selector": False,
        "feature_design_informed_by_gpu5_posthoc_analysis": True,
        "calibration": str(args.calibration.resolve()),
        "normalized_excess_threshold": calibration.threshold,
        "persistence": calibration.persistence,
        "healthy_calibration_loo_formal_alarms": 0,
        "by_run": by_run,
        "gpu4_heldout_plus_gpu5_development": aggregate(test_like),
        "target_nearest_alarm_minus_missed_query": target_timing,
        "prospective_validation_complete": False,
        "interpretation": [
            "The v2 threshold uses healthy trajectories only.",
            "The feature was designed after inspecting GPU5 failures, so this replay is not prospective validation.",
            "A new frozen-seed online batch is required before claiming improved detection.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

