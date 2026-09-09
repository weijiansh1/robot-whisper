#!/usr/bin/env python3
"""Seal fixed temporal schedules and a clock-only diagnostic before outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_FEATURE, EXTERNAL_LAYER, EXTRA_FEATURE, EXTRA_LAYER, LABEL_ROOT,
    MAIN_FEATURE, MAIN_LAYER, aligned_labels, assert_aligned, combine_reference,
    feature, grouped_rows, load_npz, metric_row, sha256,
)
from evaluate_unlabeled_budget import (  # noqa: E402
    COHORTS, DEFAULT_OUTPUT as STATIC_OUTPUT, verify_seal as verify_static_seal, write_json,
)
from intrinsic_guard_monitor import GlobalIntrinsicProfile, intrinsic_score_arrays  # noqa: E402
from temporal_budget_guard import (  # noqa: E402
    EXTERNAL_TIME_SEED, REFERENCE_TIME_SEED, SCHEMA, SCHEDULES, TemporalProfile,
    adjusted_score_arrays, calibrate_temporal_reference, calibrate_time_only, time_priorities,
)
from unlabeled_budget_calibration import DEFAULT_ALARM_BUDGET, alarms_from_scores  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/temporal_budget_guard"
PROTOCOL = BUNDLE / "method/TEMPORAL_BUDGET_PROTOCOL.md"
DETECTORS = ("published_v7", *SCHEDULES, "time_only")
SCORE_FIELDS = ("freeze", "acceleration", "periodicity", "acceleration_persistent", "periodicity_persistent")


def verify_seal(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected temporal seal schema")
    for path, digest in manifest["artifacts"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def seal(output: Path, alarm_budget: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    previous = verify_static_seal(STATIC_OUTPUT)
    sources = [PROTOCOL, Path(__file__), BUNDLE / "method/temporal_budget_guard.py",
               BUNDLE / "method/unlabeled_budget_calibration.py", BUNDLE / "method/intrinsic_guard_monitor.py",
               HERE / "evaluate_intrinsic_guard_v7.py", HERE / "evaluate_unlabeled_budget.py",
               HERE / "verify_temporal_budget_raw.py", HERE / "verify_unlabeled_budget_raw.py",
               HERE / "verify_raw_causal_gpu.py", STATIC_OUTPUT / "sealed_manifest.json",
               STATIC_OUTPUT / "global_profile.npz", STATIC_OUTPUT / "sealed_first_alarms.npz",
               MAIN_LAYER, EXTRA_LAYER, EXTERNAL_LAYER, MAIN_FEATURE, EXTRA_FEATURE, EXTERNAL_FEATURE,
               BUNDLE / "tests/test_temporal_budget.py", BUNDLE / "tests/test_temporal_budget_evaluation.py"]
    artifacts = {str(path.resolve()): sha256(path) for path in sources}
    for path in (MAIN_LAYER, EXTRA_LAYER, EXTERNAL_LAYER, MAIN_FEATURE, EXTRA_FEATURE, EXTERNAL_FEATURE):
        if artifacts[str(path.resolve())] != previous["artifacts"][str(path.resolve())]:
            raise ValueError(f"static/temporal feature source mismatch: {path}")

    main_layer, main_route = load_npz(MAIN_LAYER), load_npz(MAIN_FEATURE)
    extra_layer, extra_route = load_npz(EXTRA_LAYER), load_npz(EXTRA_FEATURE)
    assert_aligned(main_layer, main_route, "main")
    assert_aligned(extra_layer, extra_route, "extra")
    reference = combine_reference(main_layer, extra_layer, main_route, extra_route)
    valid = np.concatenate((main_layer["valid"], extra_layer["valid"]), axis=0)
    profiles, audit = calibrate_temporal_reference(*reference, valid, alarm_budget=alarm_budget)
    priorities = time_priorities(len(valid), REFERENCE_TIME_SEED)
    time_profile, time_audit = calibrate_time_only(valid, priorities, alarm_budget=alarm_budget)
    audit["time_only"] = time_audit
    static = GlobalIntrinsicProfile.load(STATIC_OUTPUT / "global_profile.npz")
    same_static_budget = alarm_budget == previous["alarm_budget"]
    if same_static_budget and profiles["constant"].base != static:
        raise RuntimeError("constant schedule differs from the saved static automatic profile")
    old_first = load_npz(STATIC_OUTPUT / "sealed_first_alarms.npz")
    ref_scores = intrinsic_score_arrays(*reference, profiles["constant"].base.periodicity_scale)
    sealed = {"reference_time_priorities": priorities,
              "reference_time_only": time_profile.first_alarms(valid, priorities)}
    for name, profile in profiles.items():
        path = output / f"{name}_profile.npz"
        profile.save(path)
        if TemporalProfile.load(path) != profile:
            raise RuntimeError(f"temporal profile roundtrip changed values: {name}")
        artifacts[str(path.resolve())] = sha256(path)
        first = alarms_from_scores(adjusted_score_arrays(ref_scores, profile.schedule), valid, profile.base)
        for head, values in first.items():
            sealed[f"reference_{name}_{head}"] = values
            if same_static_budget and name == "constant" and not np.array_equal(values, old_first[f"reference_auto_{head}"]):
                raise RuntimeError(f"constant reference first-alarm mismatch: {head}")
        count = int((first["guard"] >= 0).sum())
        if count != audit["variants"][name]["reference_replay_alarms"]:
            raise RuntimeError(f"serialized temporal reference replay mismatch: {name}")
        print(f"calibrated {name}: {count}/{len(valid)} reference alarms; no outcomes loaded", flush=True)
    time_path = output / "time_only_profile.json"
    write_json(time_path, {"schema": SCHEMA, **asdict(time_profile),
                           "reference_seed": REFERENCE_TIME_SEED, "external_seed": EXTERNAL_TIME_SEED})
    artifacts[str(time_path.resolve())] = sha256(time_path)
    print(f"calibrated time_only: {time_audit['reference_replay_alarms']}/{len(valid)} reference alarms", flush=True)

    for prefix, cohort, layer_path, route_path, _ in COHORTS:
        layer, route = (main_layer, main_route) if prefix == "main" else (load_npz(layer_path), load_npz(route_path))
        assert_aligned(layer, route, cohort)
        scores = intrinsic_score_arrays(layer["mobility"], feature(route, "route_acceleration"),
                                        feature(route, "lag_periodicity"), profiles["constant"].base.periodicity_scale)
        cohort_valid = layer["valid"]
        metadata = {key: layer[key] for key in
                    ("task_names", "task_index", "episode", "init_state_id", "length", "valid", "run_id")}
        streams = {}
        for name, profile in profiles.items():
            adjusted = adjusted_score_arrays(scores, profile.schedule)
            first = alarms_from_scores(adjusted, cohort_valid, profile.base)
            for head, values in first.items():
                sealed[f"{prefix}_{name}_{head}"] = values
                if same_static_budget and name == "constant" and not np.array_equal(values, old_first[f"{prefix}_auto_{head}"]):
                    raise RuntimeError(f"saved constant first-alarm mismatch: {prefix}/{head}")
            sealed[f"{prefix}_{name}"] = first["guard"]
            streams.update({f"{name}_{key}": np.where(cohort_valid, adjusted[key], np.nan) for key in SCORE_FIELDS})
        current_priorities = priorities[:len(cohort_valid)] if prefix == "main" else time_priorities(len(cohort_valid), EXTERNAL_TIME_SEED)
        sealed[f"{prefix}_time_priorities"] = current_priorities
        sealed[f"{prefix}_time_only"] = time_profile.first_alarms(cohort_valid, current_priorities)
        sealed[f"{prefix}_published_v7"] = old_first[f"{prefix}_published_v7"]
        score_path = output / f"{prefix}_scores.npz"
        np.savez_compressed(score_path, schema=np.asarray(SCHEMA), **metadata, **streams)
        artifacts[str(score_path.resolve())] = sha256(score_path)
        print(f"sealed {cohort}: all {len(DETECTORS)} fixed detectors, {len(cohort_valid)} episodes", flush=True)

    audit_path = output / "calibration_audit.json"
    write_json(audit_path, audit)
    alarm_path = output / "sealed_first_alarms.npz"
    np.savez_compressed(alarm_path, schema=np.asarray(SCHEMA), **sealed)
    for path in (audit_path, alarm_path):
        artifacts[str(path.resolve())] = sha256(path)
    manifest = {
        "schema": SCHEMA, "sealed_at_utc": datetime.now(UTC).isoformat(), "alarm_budget": alarm_budget,
        "detectors": list(DETECTORS), "calibration": audit,
        "calibration_uses_outcome_labels": False, "calibration_uses_external_features": False,
        "calibration_uses_published_thresholds": False, "new_outcome_grid_search": False,
        "related_outcomes_already_seen_by_author": True, "external_pristine_holdout": False,
        "all_outcomes_loaded_after_seal": True, "temporal_runtime_moe_and_elapsed_queries_only": True,
        "time_only_is_randomized_diagnostic_not_moe_detector": True,
        "current_episode_final_length_used_for_boundary": False,
        "persistence_uses_each_observations_own_boundary": True, "alarms_backdated": False,
        "runtime_head_features_and_latches_unchanged": True, "no_outcome_based_schedule_selection": True,
        "constant_matches_previous_static_profile_and_alarms": same_static_budget,
        "budget_semantics": "in-reference total episode-alarm ceiling, not success FPR",
        "out_of_sample_alarm_budget_guarantee": False,
        "profiles": {name: asdict(profile) for name, profile in profiles.items()},
        "time_only_profile": asdict(time_profile), "artifacts": artifacts,
    }
    write_json(output / "sealed_manifest.json", manifest)
    verify_seal(output)
    return manifest


def evaluate(output: Path, bootstrap: int) -> dict[str, Any]:
    manifest = verify_seal(output)
    sealed = load_npz(output / "sealed_first_alarms.npz")
    rng = np.random.default_rng(20260906)
    rows, suites, tasks, frames, paired = [], [], [], [], []
    label_hashes = {}
    for prefix, cohort, _, _, label_name in COHORTS:
        cache = load_npz(output / f"{prefix}_scores.npz")
        label_path = LABEL_ROOT / label_name
        labels = aligned_labels(cache, label_path, cohort)
        label_hashes[str(label_path)] = sha256(label_path)
        detectors = {name: sealed[f"{prefix}_{name}"] for name in manifest["detectors"]}
        frame = labels.copy()
        risk = labels["original_failure"].to_numpy(bool)
        baseline = detectors["constant"]
        for name, first in detectors.items():
            row = metric_row(cohort, name, first, labels, bootstrap, rng)
            alarm = first >= 0
            half = alarm & ((first + 1) * 2 <= labels["length"].to_numpy(int))
            row.update(total_alarm_rate=float(alarm.mean()), before_half_tp=int((risk & half).sum()),
                       before_half_recall=float((risk & half).sum() / max(risk.sum(), 1)))
            rows.append(row)
            frame[f"first_{name}_query"] = first
            common = risk & alarm & (baseline >= 0)
            paired.append({"cohort": cohort, "detector": name,
                           "gained_tp": int((risk & alarm & (baseline < 0)).sum()),
                           "lost_tp": int((risk & ~alarm & (baseline >= 0)).sum()),
                           "gained_fp": int((~risk & alarm & (baseline < 0)).sum()),
                           "removed_fp": int((~risk & ~alarm & (baseline >= 0)).sum()),
                           "common_detected_failures": int(common.sum()),
                           "earlier_common_tp": int((common & (first < baseline)).sum()),
                           "later_common_tp": int((common & (first > baseline)).sum()),
                           "unchanged_common_tp": int((common & (first == baseline)).sum())})
        suites.extend(grouped_rows(cohort, detectors, labels, "suite", bootstrap, rng))
        tasks.extend(grouped_rows(cohort, detectors, labels, "task", bootstrap, rng))
        frames.append(frame)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "outcome_metrics.csv", index=False)
    pd.DataFrame(suites).to_csv(output / "outcome_metrics_by_suite.csv", index=False)
    pd.DataFrame(tasks).to_csv(output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / "episode_alarms.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_changes_vs_constant.csv", index=False)
    summary = {"schema": SCHEMA, "alarm_budget": manifest["alarm_budget"],
               "all_fixed_variant_metrics": rows, "paired_changes_vs_constant": paired,
               "external_pristine_holdout": False, "outcome_based_winner_selected": False,
               "timing_target": "original endpoint, not physical failure onset",
               "label_files": label_hashes, "seal_sha256": sha256(output / "sealed_manifest.json"),
               "artifacts": {path.name: sha256(path) for path in
                             (output / "outcome_metrics.csv", output / "outcome_metrics_by_suite.csv",
                              output / "outcome_metrics_by_task.csv", output / "episode_alarms.csv",
                              output / "paired_changes_vs_constant.csv")}}
    write_json(output / "evaluation_summary.json", summary)
    print(metrics[["cohort", "detector", "tp", "fp", "precision", "risk_recall", "timely_fpr",
                   "total_alarm_rate", "early4_risk_recall", "early8_risk_recall"]].to_string(index=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("seal", "evaluate", "all"), default="all")
    parser.add_argument("--alarm-budget", type=float, default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    if args.stage in ("seal", "all"):
        seal(args.output, DEFAULT_ALARM_BUDGET if args.alarm_budget is None else args.alarm_budget)
    elif args.alarm_budget is not None and args.alarm_budget != verify_seal(args.output)["alarm_budget"]:
        parser.error("evaluation cannot change the sealed budget")
    if args.stage in ("evaluate", "all"):
        evaluate(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
