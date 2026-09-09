#!/usr/bin/env python3
"""Check raw causal runtime against the sealed temporal score streams."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "method"))

from evaluate_intrinsic_guard_v7 import EXTERNAL_FEATURE, EXTERNAL_LAYER, feature, load_npz, sha256  # noqa: E402
from evaluate_temporal_budget import DEFAULT_OUTPUT, verify_seal, write_json  # noqa: E402
from temporal_budget_guard import SCHEDULES, TemporalGuardMonitor, TemporalProfile  # noqa: E402
from verify_raw_causal_gpu import RUN_ID, load_raw_episode  # noqa: E402
from verify_unlabeled_budget_raw import max_error, sample_rows  # noqa: E402


SCORE_FIELDS = {"freeze_score": "freeze", "acceleration_score": "acceleration_persistent",
                "periodicity_score": "periodicity_persistent"}


def sample_temporal_rows(first: dict[str, np.ndarray], per_state: int) -> np.ndarray:
    selected = [sample_rows(first[name], per_state) for name in SCHEDULES]
    for name in SCHEDULES[1:]:
        changed = np.flatnonzero(first[name] != first["constant"])
        if len(changed):
            selected.append(changed[np.linspace(0, len(changed) - 1, min(per_state, len(changed)), dtype=int)])
    return np.unique(np.concatenate(selected)).astype(int)


def replay(raw: np.ndarray, profile: TemporalProfile) -> tuple[list[dict], dict[str, int]]:
    monitor = TemporalGuardMonitor(profile)
    rows = [monitor.update(query) for query in raw]
    first = {head: int(getattr(monitor, f"first_{head}_query"))
             for head in ("freeze", "acceleration", "periodicity", "turbulence")}
    first["guard"] = int(monitor.first_alarm_query)
    return rows, first


def prefix_invariant(raw: np.ndarray, profile: TemporalProfile, rows: list[dict]) -> bool:
    cut = min(max(12, len(raw) // 2), len(raw))
    changed = raw.copy()
    changed[cut:] = np.roll(changed[cut:], 5, axis=-1)
    other, _ = replay(changed, profile)
    return all(max_error(np.asarray([row[key] for row in rows[:cut]]),
                         np.asarray([row[key] for row in other[:cut]])) == 0
               for key in (*SCORE_FIELDS, "query", "first_alarm_query")) and all(
                   all(left[key] == right[key] for key in ("alarm", "mechanism", "freeze_alarm",
                                                          "acceleration_alarm", "periodicity_alarm", "turbulence_alarm"))
                   for left, right in zip(rows[:cut], other[:cut]))


def verify(output: Path, per_state: int) -> dict:
    verify_seal(output)
    layer, route = load_npz(EXTERNAL_LAYER), load_npz(EXTERNAL_FEATURE)
    if str(layer["run_id"]) != RUN_ID:
        raise ValueError("raw data belongs to a different source run")
    scores = load_npz(output / "external_scores.npz")
    sealed = load_npz(output / "sealed_first_alarms.npz")
    profiles = {name: TemporalProfile.load(output / f"{name}_profile.npz") for name in SCHEDULES}
    first = {name: sealed[f"external_{name}"] for name in SCHEDULES}
    indices = sample_temporal_rows(first, per_state)
    acceleration, periodicity = feature(route, "route_acceleration"), feature(route, "lag_periodicity")
    records = []
    for index in indices:
        task = str(layer["task_names"][int(layer["task_index"][index])])
        episode, length = int(layer["episode"][index]), int(layer["length"][index])
        raw, digest = load_raw_episode(task, episode, length)
        variants = {}
        for name, profile in profiles.items():
            rows, actual_first = replay(raw, profile)
            expected = {"layer_mobility": layer["mobility"][index, :length],
                        "route_acceleration": acceleration[index, :length],
                        "lag_periodicity": periodicity[index, :length]}
            feature_errors = {key: max_error(np.asarray([row[key] for row in rows]), value)
                              for key, value in expected.items()}
            score_errors = {key: max_error(np.asarray([row[key] for row in rows]), scores[f"{name}_{field}"][index, :length])
                            for key, field in SCORE_FIELDS.items()}
            expected_first = {head: int(sealed[f"external_{name}_{head}"][index]) for head in actual_first}
            exact = actual_first == expected_first
            causal = prefix_invariant(raw, profile, rows)
            variants[name] = {"feature_max_abs_error": feature_errors, "score_max_abs_error": score_errors,
                              "actual_first_queries": actual_first, "expected_first_queries": expected_first,
                              "exact_first_alarms": exact, "prefix_invariant": causal,
                              "passed": exact and causal and max(feature_errors.values()) <= 2e-5
                              and max(score_errors.values()) <= 3e-5}
        passed = all(value["passed"] for value in variants.values())
        records.append({"cache_row": int(index), "task": task, "episode": episode, "queries": length,
                        "raw_sha256": digest, "variants": variants, "passed": passed})
        print(f"raw temporal replay {len(records)}/{len(indices)}: row={index}, passed={passed}", flush=True)
    report = {"schema": "himoe.temporal_budget_guard.raw_replay.v1", "outcomes_loaded": False,
              "sampling": "alarm/no-alarm and schedule disagreement, evenly spaced, without outcomes",
              "episodes": len(records), "queries": sum(row["queries"] for row in records),
              "episode_variant_replays": len(records) * len(profiles),
              "alarming_episodes_by_schedule": {name: int((values[indices] >= 0).sum()) for name, values in first.items()},
              "all_passed": bool(records) and all(row["passed"] for row in records),
              "verifier_sha256": sha256(Path(__file__)), "seal_sha256": sha256(output / "sealed_manifest.json"),
              "records": records}
    write_json(output / "raw_replay_verification.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-state", type=int, default=4)
    args = parser.parse_args()
    if args.samples_per_state < 1:
        parser.error("samples per state must be positive")
    if not verify(args.output, args.samples_per_state)["all_passed"]:
        raise RuntimeError("raw temporal replay failed")


if __name__ == "__main__":
    main()
