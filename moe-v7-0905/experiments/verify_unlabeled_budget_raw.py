#!/usr/bin/env python3
"""Replay alarming and non-alarming raw MoE episodes without reading outcomes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "method"))

from evaluate_unlabeled_budget import DEFAULT_OUTPUT, PRIMARY, verify_seal, write_json  # noqa: E402
from evaluate_intrinsic_guard_v7 import EXTERNAL_FEATURE, EXTERNAL_LAYER, feature, load_npz, sha256  # noqa: E402
from verify_raw_causal_gpu import (  # noqa: E402
    RUN_ID, load_raw_episode, replay_monitor, verify_prefix_invariance,
)


def sample_rows(first: np.ndarray, per_state: int) -> np.ndarray:
    if per_state < 1:
        raise ValueError("samples per alarm state must be positive")
    selected = []
    for state in (False, True):
        rows = np.flatnonzero((np.asarray(first) >= 0) == state)
        if len(rows):
            selected.extend(rows[np.linspace(0, len(rows) - 1, min(per_state, len(rows)), dtype=int)])
    return np.unique(selected).astype(int)


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    if any(not np.array_equal(test(left), test(right)) for test in (np.isnan, np.isposinf, np.isneginf)):
        return float("inf")
    finite = np.isfinite(left)
    return float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0


def verify(output: Path, per_state: int) -> dict:
    verify_seal(output)
    layer, route = load_npz(EXTERNAL_LAYER), load_npz(EXTERNAL_FEATURE)
    if str(layer["run_id"]) != RUN_ID:
        raise ValueError("raw replay helper is for a different source run")
    scores = load_npz(output / "external_scores.npz")
    sealed = load_npz(output / "sealed_first_alarms.npz")
    first = sealed[f"external_{PRIMARY}"]
    indices = sample_rows(first, per_state)
    acceleration, periodicity = feature(route, "route_acceleration"), feature(route, "lag_periodicity")
    profile_path = output / "global_profile.npz"
    records = []
    for index in indices:
        task = str(layer["task_names"][int(layer["task_index"][index])])
        episode, length = int(layer["episode"][index]), int(layer["length"][index])
        raw, digest = load_raw_episode(task, episode, length)
        stream, actual_first = replay_monitor(raw, profile_path)
        expected_features = {"mobility": layer["mobility"][index, :length],
                             "acceleration": acceleration[index, :length],
                             "periodicity": periodicity[index, :length]}
        feature_errors = {name: max_error(stream[name], expected) for name, expected in expected_features.items()}
        score_fields = {"freeze_score": "freeze", "acceleration_score": "acceleration_persistent",
                        "periodicity_score": "periodicity_persistent"}
        score_errors = {name: max_error(stream[name], scores[field][index, :length])
                        for name, field in score_fields.items()}
        expected_first = {head: int(sealed[f"external_auto_{head}"][index]) for head in actual_first}
        exact_first = actual_first == expected_first
        prefix_equal = verify_prefix_invariance(raw, profile_path)
        passed = exact_first and prefix_equal and max(feature_errors.values()) <= 2e-5 and max(score_errors.values()) <= 3e-5
        records.append({
            "cache_row": int(index), "task": task, "episode": episode, "queries": length,
            "sampled_by_alarm_not_outcome": bool(first[index] >= 0), "raw_sha256": digest,
            "feature_max_abs_error": feature_errors, "score_max_abs_error": score_errors,
            "expected_first_queries": expected_first, "actual_first_queries": actual_first,
            "exact_first_alarms": exact_first, "prefix_invariant": prefix_equal, "passed": passed,
        })
        print(f"raw v7 replay {len(records)}/{len(indices)}: row={index}, passed={passed}", flush=True)
    report = {
        "schema": "himoe.unlabeled_budget.raw_replay.v1", "outcomes_loaded": False,
        "sampling": "uniformly spaced rows within sealed alarm/no-alarm groups, without labels",
        "episodes": len(records), "queries": sum(r["queries"] for r in records),
        "alarming_episodes": int((first[indices] >= 0).sum()),
        "all_passed": bool(records) and all(r["passed"] for r in records),
        "verifier_sha256": sha256(Path(__file__)),
        "seal_sha256": sha256(output / "sealed_manifest.json"), "records": records,
    }
    write_json(output / "raw_replay_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-state", type=int, default=8)
    args = parser.parse_args()
    if args.samples_per_state < 1:
        parser.error("samples per state must be positive")
    if not verify(args.output, args.samples_per_state)["all_passed"]:
        raise RuntimeError("raw replay failed; inspect the verification artifact")


if __name__ == "__main__":
    main()
