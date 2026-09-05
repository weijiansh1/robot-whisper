#!/usr/bin/env python3
"""Diagnose whether the frozen MoE-only alarm fails at features or persistence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = PACKAGE_ROOT / "results/moe_only_online_alarm/failure_type_audit"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_only_online_alarm/method_diagnosis"
GPU5_ROLE = "gpu5_extended"
WINDOWS = {
    "pre5": np.arange(-5, 0, dtype=np.int64),
    "at": np.asarray([0], dtype=np.int64),
    "post5": np.arange(1, 6, dtype=np.int64),
    "around5": np.arange(-5, 6, dtype=np.int64),
}
FEATURES = (
    "frozen_conjunction_margin",
    "layer5_gap_log_ratio",
    "back_jump_log_ratio",
    "nominal_action_log_margin",
    "phase_match_distance",
    "front_late_flow_wj",
    "back_late_flow_wj",
    "front_route_acceleration",
    "back_route_acceleration",
    "front_chunk_jump_h",
    "back_chunk_jump_h",
    "layer5_state_action_gap",
)


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def normalize(probability: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(probability, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    coefficient = (np.sqrt(normalize(left)) * np.sqrt(normalize(right))).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def weighted_jaccard_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize(left)
    right = normalize(right)
    return 1.0 - np.minimum(left, right).sum(axis=-1) / np.maximum(
        np.maximum(left, right).sum(axis=-1), 1e-12
    )


def route_features(route: np.ndarray) -> dict[str, np.ndarray]:
    route = normalize(route)
    action = route[:, :, :, 1:, :]
    flow_distance = weighted_jaccard_distance(action[:, :, 1:], action[:, :, :-1])
    acceleration = (
        np.sqrt(action[:, :, 2:])
        - 2.0 * np.sqrt(action[:, :, 1:-1])
        + np.sqrt(action[:, :, :-2])
    )
    acceleration = np.linalg.norm(acceleration, axis=-1) / np.sqrt(2.0)
    chunk_jump = np.full((len(route), 8), np.nan, dtype=np.float64)
    chunk_jump[1:] = hellinger(action[1:], action[:-1]).mean(axis=(2, 3))
    front_chunk_jump = np.full(len(route), np.nan, dtype=np.float64)
    back_chunk_jump = np.full(len(route), np.nan, dtype=np.float64)
    front_chunk_jump[1:] = chunk_jump[1:, :4].mean(axis=1)
    back_chunk_jump[1:] = chunk_jump[1:, 4:].mean(axis=1)
    state = route[:, 3, -1, 0]
    action_mean = normalize(route[:, 3, -1, 1:].mean(axis=1))
    return {
        "front_late_flow_wj": flow_distance[:, :4, 6:].mean(axis=(1, 2, 3)),
        "back_late_flow_wj": flow_distance[:, 4:, 6:].mean(axis=(1, 2, 3)),
        "front_route_acceleration": acceleration[:, :4].mean(axis=(1, 2, 3)),
        "back_route_acceleration": acceleration[:, 4:].mean(axis=(1, 2, 3)),
        "front_chunk_jump_h": front_chunk_jump,
        "back_chunk_jump_h": back_chunk_jump,
        "layer5_state_action_gap": hellinger(state, action_mean),
    }


def selector_features(archive: Any) -> dict[str, np.ndarray]:
    names = [str(value) for value in archive["selector_metric_names"]]
    matrix = np.asarray(archive["selector_metrics"], dtype=np.float64)
    by_name = {name: matrix[:, index] for index, name in enumerate(names)}
    gap = np.log(np.maximum(by_name["layer5_gap_ratio"], 1e-12))
    jump = np.log(np.maximum(by_name["back_chunk_jump_ratio"], 1e-12))
    nominal = -np.log(np.maximum(by_name["front_action_distance_ratio"], 1e-12))
    return {
        "frozen_conjunction_margin": np.minimum(np.minimum(gap, jump), nominal),
        "layer5_gap_log_ratio": gap,
        "back_jump_log_ratio": jump,
        "nominal_action_log_margin": nominal,
        "phase_match_distance": by_name["mean_local_match_distance"],
    }


def alarm_rule(raw: np.ndarray, rule: str) -> bool:
    raw = np.asarray(raw, dtype=np.int64)
    if rule == "any_raw":
        return bool(raw.any())
    if rule == "consecutive2":
        return bool(len(raw) >= 2 and np.any(raw[:-1] + raw[1:] >= 2))
    needed, width = {
        "two_in5": (2, 5),
        "two_in10": (2, 10),
        "three_in10": (3, 10),
    }[rule]
    return bool(
        any(raw[max(0, right - width + 1) : right + 1].sum() >= needed for right in range(len(raw)))
    )


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparison = positive[:, None] - negative[None, :]
    return float((np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)) / comparison.size)


def event_score(values: np.ndarray, anchors: list[int], relative: np.ndarray) -> float:
    indices = sorted(
        {
            anchor + int(offset)
            for anchor in anchors
            for offset in relative
            if 0 <= anchor + int(offset) < len(values)
        }
    )
    selected = values[indices]
    selected = selected[np.isfinite(selected)]
    return float(selected.max()) if len(selected) else float("nan")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    audit_root = args.audit.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(audit_root / "episode_audit.csv")
    events = [
        json.loads(line)
        for line in (audit_root / "grasp_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    event_by_episode: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        key = (str(event["run_role"]), int(event["episode_index"]))
        event_by_episode.setdefault(key, []).append(event)

    episode_records: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    rules = ("any_raw", "consecutive2", "two_in5", "two_in10", "three_in10")
    for row in rows:
        path = PACKAGE_ROOT / row["trajectory"]
        with np.load(path, allow_pickle=False) as archive:
            raw = np.asarray(archive["selector_raw_reject"], dtype=np.bool_)
            features = selector_features(archive)
            features.update(route_features(np.asarray(archive["hb_router_probs"])))
        key = (row["run_role"], int(row["episode_index"]))
        episode_events = event_by_episode.get(key, [])
        family = row["failure_family"]
        if family == "target_missed_grasp_proxy":
            anchors = sorted(
                {
                    int(event["query"])
                    for event in episode_events
                    if bool(event["kinematic_missed_grasp_then_departure"])
                }
            )
        elif family == "success":
            anchors = sorted(
                {
                    int(event["query"])
                    for event in episode_events
                    if float(event["target_max_displacement_after_m"]) >= 0.05
                }
            )
        else:
            anchors = []
        record: dict[str, Any] = {
            "run_role": row["run_role"],
            "episode_index": int(row["episode_index"]),
            "init_state_id": int(row["init_state_id"]),
            "failure_family": family,
            "anchors": json.dumps(anchors, separators=(",", ":")),
            "raw_count": int(raw.sum()),
        }
        for rule in rules:
            record[rule] = alarm_rule(raw, rule)
        episode_records.append(record)
        if not anchors or family not in ("target_missed_grasp_proxy", "success"):
            continue
        for window_name, relative in WINDOWS.items():
            for feature in FEATURES:
                score_rows.append(
                    {
                        "run_role": row["run_role"],
                        "episode_index": int(row["episode_index"]),
                        "init_state_id": int(row["init_state_id"]),
                        "failure_family": family,
                        "window": window_name,
                        "feature": feature,
                        "score": event_score(features[feature], anchors, relative),
                    }
                )

    with (output / "episode_rule_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_records[0]))
        writer.writeheader()
        writer.writerows(episode_records)
    with (output / "event_window_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    scopes = {
        "gpu5_extended": lambda value: value == GPU5_ROLE,
        "all_online_runs": lambda _value: True,
    }
    rule_summary: dict[str, Any] = {}
    for scope, include in scopes.items():
        selected = [record for record in episode_records if include(record["run_role"])]
        rule_summary[scope] = {}
        for rule in rules:
            rule_summary[scope][rule] = {}
            for family in ("target_missed_grasp_proxy", "other_failure", "success"):
                cohort = [record for record in selected if record["failure_family"] == family]
                hits = sum(bool(record[rule]) for record in cohort)
                rule_summary[scope][rule][family] = {
                    "hits": hits,
                    "episodes": len(cohort),
                    "rate": None if not cohort else hits / len(cohort),
                }

    auc_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for scope, include in scopes.items():
        selected = [row for row in score_rows if include(row["run_role"])]
        episode_keys = sorted(
            {
                (row["run_role"], row["episode_index"], row["failure_family"])
                for row in selected
            }
        )
        tests = [(window, feature) for window in WINDOWS for feature in FEATURES]
        matrix = np.full((len(episode_keys), len(tests)), np.nan, dtype=np.float64)
        lookup = {
            (row["run_role"], row["episode_index"], row["failure_family"], row["window"], row["feature"]): float(row["score"])
            for row in selected
        }
        labels = np.asarray(
            [key[2] == "target_missed_grasp_proxy" for key in episode_keys], dtype=bool
        )
        for episode_axis, key in enumerate(episode_keys):
            for test_axis, (window, feature) in enumerate(tests):
                matrix[episode_axis, test_axis] = lookup.get((*key, window, feature), np.nan)
        valid_tests = np.all(np.isfinite(matrix), axis=0)
        tests = [test for test, valid in zip(tests, valid_tests) if valid]
        matrix = matrix[:, valid_tests]
        observed = np.asarray(
            [auc(matrix[labels, index], matrix[~labels, index]) for index in range(matrix.shape[1])]
        )
        null = np.empty((args.permutations, matrix.shape[1]), dtype=np.float32)
        for draw in range(args.permutations):
            permuted = rng.permutation(labels)
            null[draw] = [
                auc(matrix[permuted, index], matrix[~permuted, index])
                for index in range(matrix.shape[1])
            ]
        null_max = np.max(null - 0.5, axis=1)
        for index, ((window, feature), observed_auc) in enumerate(zip(tests, observed)):
            auc_rows.append(
                {
                    "scope": scope,
                    "window": window,
                    "feature": feature,
                    "target_episodes": int(labels.sum()),
                    "success_episodes": int((~labels).sum()),
                    "auc_target_higher": float(observed_auc),
                    "p_one_sided": float(
                        (1 + np.sum(null[:, index] >= observed_auc)) / (1 + args.permutations)
                    ),
                    "p_maxT": float(
                        (1 + np.sum(null_max >= observed_auc - 0.5)) / (1 + args.permutations)
                    ),
                }
            )
    with (output / "event_auc.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(auc_rows[0]))
        writer.writeheader()
        writer.writerows(auc_rows)

    gpu5_auc = [row for row in auc_rows if row["scope"] == "gpu5_extended"]
    frozen_pre = next(
        row
        for row in gpu5_auc
        if row["window"] == "pre5" and row["feature"] == "frozen_conjunction_margin"
    )
    best_pre = max(
        (row for row in gpu5_auc if row["window"] == "pre5"),
        key=lambda row: row["auc_target_higher"],
    )
    best_at = max(
        (row for row in gpu5_auc if row["window"] == "at"),
        key=lambda row: row["auc_target_higher"],
    )
    best_post = max(
        (row for row in gpu5_auc if row["window"] == "post5"),
        key=lambda row: row["auc_target_higher"],
    )
    summary = {
        "schema": "himoe.moe_only_alarm_method_diagnosis.v1",
        "training": False,
        "online_rule_changed_during_gpu5_collection": False,
        "physical_inputs_to_online_selector": False,
        "physical_labels_role": "posthoc_evaluation_and_event_alignment_only",
        "permutations": args.permutations,
        "seed": args.seed,
        "rule_diagnostics": rule_summary,
        "gpu5_event_aligned": {
            "frozen_conjunction_pre5": frozen_pre,
            "best_pre5_feature_descriptive_not_selected_online": best_pre,
            "best_at_event_feature_descriptive_not_selected_online": best_at,
            "best_post5_feature_descriptive_not_selected_online": best_post,
            "tests_maxT_corrected": len(gpu5_auc),
        },
        "diagnostic_conclusion": {
            "persistence_is_not_the_only_failure_mode": True,
            "frozen_rule_has_validated_pre_event_separation": False,
            "moe_telemetry_has_post_event_signal": bool(best_post["p_maxT"] < 0.05),
            "replacement_detector_validated_online": False,
        },
        "interpretation_guardrails": [
            "The GPU5 batch is held out for the frozen online rule and H20 healthy reference.",
            "Alternative persistence windows and best-feature results are posthoc diagnostics, not held-out selectors.",
            "At/post-event separation cannot be reported as early warning.",
            "Failure of this feature/rule combination does not prove that all MoE telemetry is uninformative.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
