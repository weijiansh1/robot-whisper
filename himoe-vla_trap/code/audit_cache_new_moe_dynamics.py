#!/usr/bin/env python3
"""Post-hoc missed-grasp audit for the cache_new Task-8 offline replay.

The alarm predictions must already exist.  This script never recomputes or
changes them; it uses query-sampled robot/simulator state only to assign
evaluation labels and measure alarm timing after the MoE-only pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import fisher_exact
import zarr

from audit_moe_only_online_failure_types import (
    missed_grasp_events,
    target_joints,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
RUN_ROOT = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache_new/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-50x8-20260903"
)
OLD_B_ROOT = (
    WORKSPACE_ROOT
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long/"
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
OUTPUT = (
    PACKAGE_ROOT
    / "results/moe_dynamics_online_alarm/cache_new_task8_replay"
)
PROSPECTIVE_ROOT = (
    PACKAGE_ROOT
    / "results/moe_dynamics_online_alarm/gpu5_prospective_seed20260908"
)
PROSPECTIVE_AUDIT = (
    PACKAGE_ROOT
    / "results/moe_dynamics_online_alarm/prospective_audit/episode_audit.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratio(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def family_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in ("target_missed_grasp_proxy", "other_failure", "success"):
        selected = [row for row in rows if row["failure_family"] == family]
        output[family] = {
            "episodes": len(selected),
            "any_raw_reject": ratio(
                sum(bool(row["any_raw_reject"]) for row in selected), len(selected)
            ),
            "formal_alarm_anywhere": ratio(
                sum(bool(row["formal_alarm"]) for row in selected), len(selected)
            ),
        }
        if family == "target_missed_grasp_proxy":
            output[family].update(
                {
                    "raw_reject_within_5q_before_or_at_miss": ratio(
                        sum(
                            bool(row["raw_within_5q_before_or_at_miss"])
                            for row in selected
                        ),
                        len(selected),
                    ),
                    "formal_alarm_within_5q_before_or_at_miss": ratio(
                        sum(
                            bool(row["alarm_within_5q_before_or_at_miss"])
                            for row in selected
                        ),
                        len(selected),
                    ),
                    "formal_alarm_within_5q_after_miss": ratio(
                        sum(
                            bool(row["alarm_within_5q_after_miss"])
                            for row in selected
                        ),
                        len(selected),
                    ),
                }
            )
    return output


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    positive = np.asarray(pos, dtype=np.float64)[:, None]
    negative = np.asarray(neg, dtype=np.float64)[None, :]
    return float(((positive > negative) + 0.5 * (positive == negative)).mean())


def score_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        family: [row for row in rows if row["failure_family"] == family]
        for family in ("target_missed_grasp_proxy", "other_failure", "success")
    }
    target = groups["target_missed_grasp_proxy"]
    other = groups["other_failure"]
    success = groups["success"]
    failure = target + other

    def scores(selected: list[dict[str, Any]]) -> list[float]:
        return [float(row["score_max"]) for row in selected]

    def alarm_table(
        positive: list[dict[str, Any]], negative: list[dict[str, Any]]
    ) -> list[list[int]]:
        return [
            [
                sum(bool(row["formal_alarm"]) for row in positive),
                sum(not bool(row["formal_alarm"]) for row in positive),
            ],
            [
                sum(bool(row["formal_alarm"]) for row in negative),
                sum(not bool(row["formal_alarm"]) for row in negative),
            ],
        ]

    comparisons = {
        "all_failure_vs_success": (failure, success),
        "target_vs_success": (target, success),
        "other_failure_vs_success": (other, success),
        "target_vs_other_failure": (target, other),
    }
    output: dict[str, Any] = {}
    for name, (positive, negative) in comparisons.items():
        table = alarm_table(positive, negative)
        exact = fisher_exact(table)
        output[name] = {
            "score_max_auc": auc(scores(positive), scores(negative)),
            "formal_alarm_odds_ratio": float(exact.statistic),
            "formal_alarm_fisher_two_sided_p": float(exact.pvalue),
            "formal_alarm_table": table,
        }
    return output


def validate_query_sampled_proxy() -> dict[str, Any]:
    dense_rows = {
        int(row["episode_index"]): row
        for row in csv.DictReader(
            PROSPECTIVE_AUDIT.open(newline="", encoding="utf-8")
        )
    }
    first_episode = next(iter(sorted(PROSPECTIVE_ROOT.glob("episode_*"))))
    targets = target_joints(
        json.loads((first_episode / "sim_layout.json").read_text(encoding="utf-8"))
    )
    counts = {"strict_true_positive": 0, "strict_false_negative": 0,
              "strict_false_positive": 0, "strict_true_negative": 0}
    added_episode_ids: list[int] = []
    additions_are_dense_borderline = True
    success_additions = 0
    for episode_dir in sorted(PROSPECTIVE_ROOT.glob("episode_*")):
        episode_id = int(episode_dir.name.split("_", 2)[1])
        with np.load(
            episode_dir / "trajectory_and_routes.npz", allow_pickle=False
        ) as archive:
            state = np.asarray(archive["policy_state"], dtype=np.float32)
            sim_state = np.asarray(archive["query_sim_state"], dtype=np.float32)
        events = missed_grasp_events(
            {
                "control_sim_state": sim_state,
                "control_eef_position": state[:, :3],
                "control_gripper_qpos": state[:, 6:8],
                "control_query_index": np.arange(len(state), dtype=np.int32),
            },
            targets,
        )
        query_proxy = any(
            event["kinematic_missed_grasp_then_departure"] for event in events
        )
        dense = dense_rows[episode_id]
        dense_strict = bool(json.loads(dense["missed_grasp_queries"]))
        key = (
            "strict_true_positive"
            if dense_strict and query_proxy
            else "strict_false_negative"
            if dense_strict
            else "strict_false_positive"
            if query_proxy
            else "strict_true_negative"
        )
        counts[key] += 1
        if query_proxy and not dense_strict:
            added_episode_ids.append(episode_id)
            additions_are_dense_borderline &= bool(
                json.loads(dense["borderline_15mm_queries"])
            )
            success_additions += int(dense["success"] == "True")
    return {
        **counts,
        "additional_query_proxy_episode_ids": added_episode_ids,
        "all_additions_are_dense_15mm_borderline": additions_are_dense_borderline,
        "successful_episode_additions": success_additions,
    }


def nominal_overlap_audit() -> dict[str, Any]:
    old_summaries = json.loads(
        (OLD_B_ROOT / "client/summaries.json").read_text(encoding="utf-8")
    )
    new_summaries = json.loads(
        (RUN_ROOT / "client/summaries.json").read_text(encoding="utf-8")
    )
    old_by_condition = {
        (int(row["init_state_id"]), int(row["flow_noise_seed"])): row
        for row in old_summaries
    }
    new_by_condition = {
        (int(row["init_state_id"]), int(row["flow_noise_seed"])): row
        for row in new_summaries
    }
    conditions = sorted(old_by_condition.keys() & new_by_condition.keys())
    old_store = zarr.open_group(
        str(OLD_B_ROOT / "server/routes.zarr"), mode="r"
    )
    new_store = zarr.open_group(str(RUN_ROOT / "server/routes.zarr"), mode="r")
    old_ids = np.asarray(old_store["episode_id"][:], dtype=np.int64)
    new_ids = np.asarray(new_store["episode_id"][:], dtype=np.int64)

    def route(store: zarr.Group, ids: np.ndarray, episode_id: int) -> np.ndarray:
        rows = np.flatnonzero(ids == episode_id)
        if not len(rows) or not np.all(np.diff(rows) == 1):
            raise RuntimeError("overlap audit expects contiguous episode rows")
        return np.asarray(
            store["hb_router_probs"][int(rows[0]) : int(rows[-1]) + 1]
        )

    exact = 0
    first_query_exact = 0
    same_length = 0
    same_outcome = 0
    for condition in conditions:
        old = old_by_condition[condition]
        new = new_by_condition[condition]
        old_route = route(old_store, old_ids, int(old["episode_index"]))
        new_route = route(new_store, new_ids, int(new["episode_index"]))
        same_shape = old_route.shape == new_route.shape
        same_length += int(same_shape)
        first_query_exact += int(np.array_equal(old_route[0], new_route[0]))
        exact += int(same_shape and np.array_equal(old_route, new_route))
        same_outcome += int(bool(old["success"]) == bool(new["success"]))
    return {
        "nominally_overlapping_init_noise_conditions": len(conditions),
        "conditions_not_present_in_old_B": len(new_summaries) - len(conditions),
        "whole_route_exact_duplicates": exact,
        "first_query_exact_duplicates": first_query_exact,
        "same_length": same_length,
        "same_outcome": same_outcome,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.output.resolve()
    prediction_path = output / "episode_predictions_moe_only.csv"
    if not prediction_path.exists():
        raise RuntimeError(f"MoE-only predictions do not exist: {prediction_path}")

    predictions = {
        int(row["episode_id"]): row
        for row in csv.DictReader(prediction_path.open(newline="", encoding="utf-8"))
    }
    summaries = json.loads(
        (run / "client/summaries.json").read_text(encoding="utf-8")
    )
    if len(predictions) != len(summaries):
        raise RuntimeError("prediction and trajectory counts differ")
    layout = json.loads(
        (run / "client/sim_layout.json").read_text(encoding="utf-8")
    )
    targets = target_joints(layout)
    old_conditions = {
        (int(row["init_state_id"]), int(row["flow_noise_seed"]))
        for row in json.loads(
            (OLD_B_ROOT / "client/summaries.json").read_text(encoding="utf-8")
        )
    }

    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for summary in summaries:
        episode_id = int(summary["episode_index"])
        prediction = predictions[episode_id]
        with np.load(
            run / "client" / f"episode_{episode_id:02d}.npz",
            allow_pickle=False,
        ) as archive:
            state = np.asarray(archive["state"], dtype=np.float32)
            sim_state = np.asarray(archive["sim_state"], dtype=np.float32)
        if len(state) != int(summary["inference_calls"]):
            raise RuntimeError(f"query count mismatch for episode {episode_id}")

        # These arrays are one sample per replan query, not per control action.
        proxy_arrays = {
            "control_sim_state": sim_state,
            "control_eef_position": state[:, :3],
            "control_gripper_qpos": state[:, 6:8],
            "control_query_index": np.arange(len(state), dtype=np.int32),
        }
        events = missed_grasp_events(proxy_arrays, targets)
        for event in events:
            event_rows.append(
                {
                    "episode_id": episode_id,
                    "init_state_id": int(summary["init_state_id"]),
                    "success": bool(summary["success"]),
                    **event,
                }
            )
        missed_queries = [
            int(event["query"])
            for event in events
            if event["kinematic_missed_grasp_then_departure"]
        ]
        borderline_queries = [
            int(event["query"])
            for event in events
            if event["missed_grasp_sensitivity_only_15mm"]
        ]
        success = bool(summary["success"])
        family = (
            "success"
            if success
            else "target_missed_grasp_proxy"
            if missed_queries
            else "other_failure"
        )
        raw_queries = [int(value) for value in json.loads(prediction["raw_reject_queries"])]
        alarm_queries = [int(value) for value in json.loads(prediction["alarm_queries"])]
        alarm_deltas = [
            alarm - missed
            for alarm in alarm_queries
            for missed in missed_queries
        ]
        raw_deltas = [
            raw - missed for raw in raw_queries for missed in missed_queries
        ]
        condition = (
            int(summary["init_state_id"]),
            int(summary["flow_noise_seed"]),
        )
        rows.append(
            {
                "episode_id": episode_id,
                "init_state_id": condition[0],
                "flow_noise_seed": condition[1],
                "condition_present_in_old_B": condition in old_conditions,
                "success": success,
                "failure_family": family,
                "queries": int(prediction["queries"]),
                "score_max": float(prediction["score_max"]),
                "raw_reject_queries": json.dumps(raw_queries, separators=(",", ":")),
                "formal_alarm_queries": json.dumps(
                    alarm_queries, separators=(",", ":")
                ),
                "any_raw_reject": bool(raw_queries),
                "formal_alarm": bool(alarm_queries),
                "missed_grasp_queries": json.dumps(
                    missed_queries, separators=(",", ":")
                ),
                "borderline_15mm_queries": json.dumps(
                    borderline_queries, separators=(",", ":")
                ),
                "nearest_alarm_minus_missed_query": (
                    None
                    if not alarm_deltas
                    else min(alarm_deltas, key=lambda value: (abs(value), value))
                ),
                "raw_within_5q_before_or_at_miss": any(
                    -5 <= delta <= 0 for delta in raw_deltas
                ),
                "alarm_within_5q_before_or_at_miss": any(
                    -5 <= delta <= 0 for delta in alarm_deltas
                ),
                "alarm_within_5q_after_miss": any(
                    0 < delta <= 5 for delta in alarm_deltas
                ),
            }
        )

    write_csv(output / "posthoc_missed_grasp_audit.csv", rows)
    with (output / "grasp_events_query_sampled.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in event_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    unseen = [row for row in rows if not row["condition_present_in_old_B"]]
    overlap = [row for row in rows if row["condition_present_in_old_B"]]
    success_proxy_events = sum(
        bool(json.loads(row["missed_grasp_queries"]))
        for row in rows
        if row["success"]
    )
    summary = {
        "schema": "himoe.cache_new_moe_dynamics_posthoc_audit.v1",
        "training": False,
        "prediction_input": "HB MoE router probabilities only",
        "physics_role": "posthoc_failure_typing_and_timing_only",
        "predictions_materialized_before_physics_loaded": True,
        "prediction_artifact": str(prediction_path),
        "prediction_artifact_sha256": sha256_file(prediction_path),
        "route_run": str(run),
        "route_run_status": json.loads(
            (run / "meta.json").read_text(encoding="utf-8")
        )["status"],
        "episodes": len(rows),
        "query_sampling": "one physical sample per replan query (10 actions)",
        "missed_grasp_proxy": {
            "source": "query-sampled eef, gripper qpos, and object qpos",
            "contact_or_force_used": False,
            "successful_episodes_with_proxy": success_proxy_events,
            "limitation": (
                "Within-chunk closure events can be missed; this is a conservative "
                "kinematic proxy, not contact ground truth."
            ),
        },
        "query_proxy_validation_on_dense_prospective_24": (
            validate_query_sampled_proxy()
        ),
        "old_B_overlap_audit": nominal_overlap_audit(),
        "all_400": family_summary(rows),
        "condition_not_in_old_B_272": family_summary(unseen),
        "condition_also_in_old_B_128": family_summary(overlap),
        "continuous_score_and_fixed_alarm_analysis": {
            "all_400": score_analysis(rows),
            "condition_not_in_old_B_272": score_analysis(unseen),
        },
        "interpretation": [
            "The frozen persistence-2 alarm is specific but has low recall on this corpus.",
            "Single-query raw rejects occur in most trajectories and are not a usable alarm alone.",
            "Cache_new was collected before this audit, so this is offline validation, not a prospective deployment trial.",
        ],
    }
    (output / "posthoc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
