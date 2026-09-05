#!/usr/bin/env python3
"""Build the v2 MoE dynamics threshold from healthy trajectories only."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from moe_dynamics_online_selector import (
    CALIBRATION_SCHEMA,
    SELECTOR_VERSION,
    normalized_acceleration_excess,
    sha256_file,
)
from moe_only_online_selector import HealthySequenceBank, OnlineMonotoneMatcher


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/h20_healthy_route_sequences_seed20260905.npz"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_dynamics_online_alarm/calibration_v2"


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sequence_scores(
    sequence: np.ndarray, bank: HealthySequenceBank, candidate: int
) -> tuple[list[dict[str, Any]], list[float]]:
    matcher = OnlineMonotoneMatcher(bank, max_advance=2)
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for query in range(1, len(sequence)):
        positions, match_distance = matcher.update(sequence[query], sequence[query - 1])
        references = np.stack(
            [route[int(position)] for route, position in zip(bank.routes, positions)]
        )
        front, back, contrast, healthy_max, score = normalized_acceleration_excess(
            sequence[query], references
        )
        scores.append(score)
        rows.append(
            {
                "candidate": candidate,
                "query": query,
                "matched_reference_queries": json.dumps(
                    positions.tolist(), separators=(",", ":")
                ),
                "mean_local_match_distance": match_distance,
                "front_route_acceleration": front,
                "back_route_acceleration": back,
                "back_front_acceleration_ratio": contrast,
                "matched_healthy_ratio_max": healthy_max,
                "normalized_acceleration_excess": score,
            }
        )
    return rows, scores


def persistence_support(scores: list[float], persistence: int) -> float:
    if len(scores) < persistence:
        raise ValueError("healthy sequence is shorter than persistence")
    return max(
        min(scores[start : start + persistence])
        for start in range(len(scores) - persistence + 1)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--healthy-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--persistence", type=int, default=2)
    args = parser.parse_args()
    if args.persistence != 2:
        raise ValueError("v2 freezes persistence=2")
    reference = args.healthy_reference.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = HealthySequenceBank.load(reference)

    query_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for index, (sequence, candidate) in enumerate(zip(bank.routes, bank.identities)):
        loo = HealthySequenceBank(
            [value for other, value in enumerate(bank.routes) if other != index],
            [value for other, value in enumerate(bank.identities) if other != index],
        )
        rows, scores = sequence_scores(sequence, loo, candidate)
        query_rows.extend(rows)
        episode_rows.append(
            {
                "candidate": candidate,
                "queries": len(sequence),
                "query_score_max": max(scores),
                "persistence2_support_max": persistence_support(
                    scores, args.persistence
                ),
            }
        )
    threshold = max(row["persistence2_support_max"] for row in episode_rows)
    for row in query_rows:
        row["raw_at_frozen_threshold"] = bool(
            row["normalized_acceleration_excess"] > threshold
        )

    with (output / "healthy_loo_query_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(query_rows[0]))
        writer.writeheader()
        writer.writerows(query_rows)
    with (output / "healthy_loo_episode_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)

    calibration = {
        "schema": CALIBRATION_SCHEMA,
        "selector_version": SELECTOR_VERSION,
        "training": False,
        "gradient_optimization": False,
        "feature_design_informed_by_posthoc_failure_analysis": True,
        "failure_labels_used_to_set_threshold": False,
        "physical_alignment_used_to_set_threshold": False,
        "healthy_reference": str(reference),
        "healthy_reference_sha256": sha256_file(reference),
        "healthy_reference_identities": list(bank.identities),
        "healthy_reference_trajectories": len(bank.routes),
        "feature": "back_front_route_acceleration / phase_matched_healthy_max",
        "route_acceleration": "mean Hellinger-chord second difference over full flow and action tokens",
        "matcher": "closed-begin online monotone DTW on front action routing",
        "max_reference_advance": 2,
        "persistence": args.persistence,
        "threshold_source": "maximum leave-one-out healthy episode persistence support",
        "threshold_comparison": "strict_greater_than",
        "normalized_excess_threshold": threshold,
        "healthy_loo_episode_scores": episode_rows,
        "healthy_loo_formal_alarms_at_threshold": 0,
        "deployment_role": "frozen_candidate_requires_new_seed_validation",
    }
    temporary = output / "calibration.tmp.json"
    temporary.write_text(
        json.dumps(plain(calibration), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "calibration.json")
    print(json.dumps(plain(calibration), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

