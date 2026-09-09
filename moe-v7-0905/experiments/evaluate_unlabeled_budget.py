#!/usr/bin/env python3
"""Seal label-free v7 calibration and fixed ablations before evaluating outcomes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
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
    feature, grouped_rows, load_npz, metric_row, plain, sha256,
)
from intrinsic_guard_monitor import (  # noqa: E402
    SCHEMA as PROFILE_SCHEMA, GlobalIntrinsicProfile, intrinsic_score_arrays,
)
from unlabeled_budget_calibration import (  # noqa: E402
    DEFAULT_ALARM_BUDGET, SCHEMA, alarms_from_scores, calibrate_reference,
)


DEFAULT_OUTPUT = BUNDLE / "results/unlabeled_budget_guard"
PROTOCOL = BUNDLE / "method/UNLABELED_BUDGET_CALIBRATION_PROTOCOL.md"
METHOD = BUNDLE / "method/unlabeled_budget_calibration.py"
BASELINE = BUNDLE / "results/intrinsic_guard_v7"
PRIMARY = "unlabeled_budget_guard"
COHORTS = (
    ("main", "development_main", MAIN_LAYER, MAIN_FEATURE, "development_main_clean_labels.csv"),
    ("external", "external_8b", EXTERNAL_LAYER, EXTERNAL_FEATURE, "external_8b_clean_labels.csv"),
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def verify_seal(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected calibration seal schema")
    for path, digest in manifest["artifacts"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def seal(output: Path, alarm_budget: float) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    inputs = {"main_layer_cache_sha256": MAIN_LAYER, "extra_layer_cache_sha256": EXTRA_LAYER,
              "external_layer_cache_sha256": EXTERNAL_LAYER, "main_route_cache_sha256": MAIN_FEATURE,
              "extra_route_cache_sha256": EXTRA_FEATURE, "external_route_cache_sha256": EXTERNAL_FEATURE}
    sources = [PROTOCOL, METHOD, Path(__file__), BUNDLE / "method/intrinsic_guard_monitor.py",
               HERE / "evaluate_intrinsic_guard_v7.py", HERE / "verify_unlabeled_budget_raw.py",
               HERE / "verify_raw_causal_gpu.py", BASELINE / "global_profile.npz",
               BASELINE / "sealed_first_alarms.npz", BASELINE / "sealed_manifest.json",
               *inputs.values()]
    artifacts = {str(path.resolve()): sha256(path) for path in sources}

    main_layer, main_route = load_npz(MAIN_LAYER), load_npz(MAIN_FEATURE)
    extra_layer, extra_route = load_npz(EXTRA_LAYER), load_npz(EXTRA_FEATURE)
    assert_aligned(main_layer, main_route, "main")
    assert_aligned(extra_layer, extra_route, "extra")
    reference = combine_reference(main_layer, extra_layer, main_route, extra_route)
    reference_valid = np.concatenate((main_layer["valid"], extra_layer["valid"]), axis=0)
    result = calibrate_reference(*reference, reference_valid, alarm_budget=alarm_budget)
    profile_path = output / "global_profile.npz"
    np.savez_compressed(profile_path, schema=np.asarray(PROFILE_SCHEMA),
                        calibration_schema=np.asarray(SCHEMA),
                        alarm_budget=np.asarray(alarm_budget),
                        reference_episodes=np.asarray(result.audit["reference_episodes"]),
                        **{key: np.asarray(value, dtype=np.float32) for key, value in asdict(result.profile).items()})
    automatic = GlobalIntrinsicProfile.load(profile_path)
    if automatic != result.profile:
        raise RuntimeError("profile serialization changed a calibrated constant")
    reference_scores = intrinsic_score_arrays(*reference, automatic.periodicity_scale)
    reference_first = alarms_from_scores(reference_scores, reference_valid, automatic)
    replay_count = int((reference_first["guard"] >= 0).sum())
    if replay_count != result.audit["union_reference_alarms"]:
        raise RuntimeError("serialized profile disagrees with reference calibration")
    audit = {**result.audit, "profile": asdict(automatic),
             "profile_roundtrip_reference_alarms": replay_count,
             "reference_scale_quantile": 0.75}
    write_json(output / "calibration_audit.json", audit)
    print(f"calibrated without outcomes: {replay_count}/{len(reference_valid)} reference alarms; "
          f"budget={alarm_budget:g}, branch allowance={audit['branch_allowance']}", flush=True)

    # Published constants enter only these diagnostic ablations, never calibration.
    old_manifest = json.loads((BASELINE / "sealed_manifest.json").read_text())
    for field, path in inputs.items():
        if artifacts[str(path.resolve())] != old_manifest["artifacts"][field]:
            raise ValueError(f"v7 source mismatch: {path}")
    for name, field in (("global_profile.npz", "global_profile_sha256"),
                        ("sealed_first_alarms.npz", "sealed_first_alarms_sha256")):
        if artifacts[str((BASELINE / name).resolve())] != old_manifest["artifacts"][field]:
            raise ValueError(f"v7 baseline artifact changed: {name}")
    published = GlobalIntrinsicProfile.load(BASELINE / "global_profile.npz")
    if automatic.periodicity_scale != published.periodicity_scale:
        raise ValueError("shared score scale differs from published v7; this is not a threshold-only comparison")
    profiles = {
        "published_v7": published,
        "freeze_only_auto": replace(published, freeze_threshold=automatic.freeze_threshold),
        "turbulence_only_auto": replace(published, acceleration_threshold=automatic.acceleration_threshold,
                                        periodicity_threshold=automatic.periodicity_threshold),
        PRIMARY: automatic,
    }
    published_alarms = load_npz(BASELINE / "sealed_first_alarms.npz")
    sealed_alarms = {f"reference_auto_{head}": first for head, first in reference_first.items()}
    for prefix, cohort, layer_path, route_path, _ in COHORTS:
        layer, route = (main_layer, main_route) if prefix == "main" else (load_npz(layer_path), load_npz(route_path))
        assert_aligned(layer, route, cohort)
        scores = intrinsic_score_arrays(layer["mobility"], feature(route, "route_acceleration"),
                                        feature(route, "lag_periodicity"), automatic.periodicity_scale)
        valid = layer["valid"]
        for name, profile in profiles.items():
            first = alarms_from_scores(scores, valid, profile)
            sealed_alarms[f"{prefix}_{name}"] = first["guard"]
            if name == PRIMARY:
                sealed_alarms.update({f"{prefix}_auto_{head}": value for head, value in first.items()})
            if name == "published_v7":
                for head in first:
                    if not np.array_equal(first[head], published_alarms[f"{prefix}_{head}"]):
                        raise RuntimeError(f"published v7 replay mismatch: {prefix}/{head}")
        path = output / f"{prefix}_scores.npz"
        metadata = {field: layer[field] for field in
                    ("task_names", "task_index", "episode", "init_state_id", "length", "valid", "run_id")}
        np.savez_compressed(path, schema=np.asarray(SCHEMA), **metadata,
                            **{name: np.where(valid, score, np.nan) for name, score in scores.items()})
        artifacts[str(path.resolve())] = sha256(path)
        print(f"sealed all fixed variants for {cohort}: {len(valid)} episodes; no outcomes opened", flush=True)
    alarm_path = output / "sealed_first_alarms.npz"
    np.savez_compressed(alarm_path, schema=np.asarray(SCHEMA), **sealed_alarms)
    for path in (alarm_path, profile_path, output / "calibration_audit.json"):
        artifacts[str(path.resolve())] = sha256(path)
    manifest = {
        "schema": SCHEMA, "sealed_at_utc": datetime.now(UTC).isoformat(),
        "primary_detector": PRIMARY, "alarm_budget": alarm_budget,
        "budget_semantics": "in-reference total episode-alarm ceiling, not success FPR",
        "reference_episodes": len(reference_valid), "calibration": audit,
        "calibration_uses_outcome_labels": False, "calibration_uses_published_thresholds": False,
        "calibration_uses_external_features": False, "new_outcome_grid_search": False,
        "all_outcomes_loaded_after_seal": True, "runtime_monitor_unchanged": True,
        "runtime_moe_only": True, "runtime_query_causal": True, "offline_failure_model_fit": False,
        "inherited_method_design_used_outcome_feedback": True,
        "related_outcomes_already_seen_by_author": True, "external_pristine_holdout": False,
        "partial_ablations_retain_published_label_selected_thresholds": True,
        "primary_preselected_independent_of_metrics": True,
        "profiles": {name: asdict(profile) for name, profile in profiles.items()}, "artifacts": artifacts,
    }
    write_json(output / "sealed_manifest.json", manifest)
    verify_seal(output)
    return manifest


def evaluate(output: Path, bootstrap: int) -> dict[str, Any]:
    manifest = verify_seal(output)
    sealed = load_npz(output / "sealed_first_alarms.npz")
    rng = np.random.default_rng(20260906)
    rows, suites, tasks, frames = [], [], [], []
    label_hashes = {}
    for prefix, cohort, _, _, label_name in COHORTS:
        cache = load_npz(output / f"{prefix}_scores.npz")
        label_path = LABEL_ROOT / label_name
        labels = aligned_labels(cache, label_path, cohort)
        label_hashes[str(label_path)] = sha256(label_path)
        detectors = {name: sealed[f"{prefix}_{name}"] for name in manifest["profiles"]}
        frame = labels.copy()
        for name, first in detectors.items():
            row = metric_row(cohort, name, first, labels, bootstrap, rng)
            alarm = first >= 0
            risk = labels["original_failure"].to_numpy(bool)
            half = alarm & ((first + 1) * 2 <= labels["length"].to_numpy(int))
            row.update(total_alarm_rate=float(alarm.mean()),
                       before_half_tp=int((risk & half).sum()),
                       before_half_recall=float((risk & half).sum() / max(risk.sum(), 1)))
            rows.append(row)
            frame[f"first_{name}_query"] = first
        suites.extend(grouped_rows(cohort, detectors, labels, "suite", bootstrap, rng))
        tasks.extend(grouped_rows(cohort, detectors, labels, "task", bootstrap, rng))
        frames.append(frame)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "outcome_metrics.csv", index=False)
    pd.DataFrame(suites).to_csv(output / "outcome_metrics_by_suite.csv", index=False)
    pd.DataFrame(tasks).to_csv(output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / "episode_alarms.csv", index=False)
    summary = {
        "schema": SCHEMA, "primary_detector": PRIMARY, "alarm_budget": manifest["alarm_budget"],
        "calibration": manifest["calibration"],
        "primary_metrics": metrics.loc[metrics.detector == PRIMARY].to_dict(orient="records"),
        "all_fixed_variant_metrics": rows,
        "external_pristine_holdout": False, "label_files": label_hashes,
        "seal_sha256": sha256(output / "sealed_manifest.json"),
        "timing_target": "original-horizon endpoint, not physical failure onset",
        "artifacts": {path.name: sha256(path) for path in
                      (output / "outcome_metrics.csv", output / "outcome_metrics_by_suite.csv",
                       output / "outcome_metrics_by_task.csv", output / "episode_alarms.csv")},
    }
    write_json(output / "evaluation_summary.json", summary)
    print(metrics[["cohort", "detector", "tp", "fp", "total_alarm_rate", "precision", "risk_recall",
                   "timely_fpr", "early4_risk_recall"]].to_string(index=False), flush=True)
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
    elif args.alarm_budget is not None:
        manifest = verify_seal(args.output)
        if args.alarm_budget != manifest["alarm_budget"]:
            parser.error("evaluation cannot change a sealed budget; calibrate into a new output directory")
    if args.stage in ("evaluate", "all"):
        evaluate(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
