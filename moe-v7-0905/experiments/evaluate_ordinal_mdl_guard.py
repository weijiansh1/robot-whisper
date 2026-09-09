#!/usr/bin/env python3
"""Seal one calibration-free rule before outcome access, then compare with v7."""

from __future__ import annotations

import argparse
import json
import sys
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
    EXTERNAL_FEATURE, EXTERNAL_LAYER, LABEL_ROOT, MAIN_FEATURE, MAIN_LAYER,
    aligned_labels, assert_aligned, feature, load_npz, metric_row, plain, sha256,
)
from ordinal_mdl_guard import SCHEMA, score_feature_arrays  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/ordinal_mdl_guard"
PROTOCOL = BUNDLE / "method/ORDINAL_MDL_GUARD_PROTOCOL.md"
METHOD = BUNDLE / "method/ordinal_mdl_guard.py"
BASELINE = BUNDLE / "results/intrinsic_guard_v7"
COHORTS = (
    ("main", "development_main", MAIN_LAYER, MAIN_FEATURE, "development_main_clean_labels.csv"),
    ("external", "external_8b", EXTERNAL_LAYER, EXTERNAL_FEATURE, "external_8b_clean_labels.csv"),
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def seal(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    sources = [PROTOCOL, METHOD, Path(__file__), BUNDLE / "method/intrinsic_guard_monitor.py",
               HERE / "evaluate_intrinsic_guard_v7.py"]
    artifacts: dict[str, str] = {str(path.resolve()): sha256(path) for path in sources}
    episode_counts = {}
    for prefix, cohort, layer_path, route_path, _ in COHORTS:
        layer, route = load_npz(layer_path), load_npz(route_path)
        assert_aligned(layer, route, cohort)
        scores = score_feature_arrays(layer["mobility"], feature(route, "route_acceleration"),
                                      feature(route, "lag_periodicity"), layer["valid"])
        destination = output / f"{prefix}_scores.npz"
        np.savez_compressed(destination, schema=np.asarray(SCHEMA), **scores,
                            **{name: layer[name] for name in
                               ("task_names", "task_index", "episode", "init_state_id", "length", "valid")})
        artifacts[str(destination.resolve())] = sha256(destination)
        artifacts[str(layer_path.resolve())] = sha256(layer_path)
        artifacts[str(route_path.resolve())] = sha256(route_path)
        episode_counts[cohort] = len(layer["mobility"])
        print(f"sealed {cohort}: {episode_counts[cohort]} trajectories; no outcomes opened", flush=True)
    manifest = {
        "schema": SCHEMA,
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "candidate_count": 1,
        "new_outcome_grid_search": False,
        "outcome_files_read_before_seal": False,
        "reference_corpus_used_for_new_method": False,
        "threshold_calibration": False,
        "offline_model_fit": False,
        "online_adaptive_symbol_counts": True,
        "fixed_code_convention_and_implicit_boundary": True,
        "failure_probability_or_fpr_guarantee": False,
        "runtime_moe_only": True,
        "external_pristine_holdout": False,
        "prior_related_results_already_known": True,
        "episode_counts": episode_counts,
        "artifacts": artifacts,
    }
    write_json(output / "sealed_manifest.json", manifest)
    return manifest


def verify_seal(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("unexpected seal schema")
    for path, digest in manifest["artifacts"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"sealed artifact changed: {path}")
    return manifest


def evaluate(output: Path, bootstrap: int) -> None:
    manifest = verify_seal(output)
    old_manifest = json.loads((BASELINE / "sealed_manifest.json").read_text())
    baseline_path = BASELINE / "sealed_first_alarms.npz"
    if sha256(baseline_path) != old_manifest["artifacts"]["sealed_first_alarms_sha256"]:
        raise ValueError("v7 sealed alarms have changed")
    baseline = load_npz(baseline_path)
    rng = np.random.default_rng(20260906)
    rows, suite_rows, task_rows, frames = [], [], [], []
    for prefix, cohort, layer_path, _, label_file in COHORTS:
        if sha256(layer_path) != old_manifest["artifacts"][f"{prefix}_layer_cache_sha256"]:
            raise ValueError(f"v7 alignment source differs: {cohort}")
        cache = load_npz(output / f"{prefix}_scores.npz")
        labels = aligned_labels(cache, LABEL_ROOT / label_file, cohort)
        detectors = {"ordinal_mdl_guard": cache["first_alarm_query"],
                     "intrinsic_guard_v7": baseline[f"{prefix}_guard"]}
        frame = labels.copy()
        frame["mdl_estimated_keypoint_at_alarm"] = cache["first_keypoint_query"]
        for name, first in detectors.items():
            if first.shape != (len(labels),):
                raise ValueError("baseline alarm count does not align")
            frame[f"first_{name}_query"] = first
            row = metric_row(cohort, name, first, labels, bootstrap, rng)
            failed = labels["original_failure"].to_numpy(bool)
            half = (first >= 0) & ((first + 1) * 2 <= labels["length"].to_numpy(int))
            row["before_half_tp"] = int((failed & half).sum())
            row["before_half_recall"] = float((failed & half).sum() / max(failed.sum(), 1))
            rows.append(row)
            for column, destination in (("suite", suite_rows), ("task", task_rows)):
                for group in sorted(labels[column].unique()):
                    take = labels[column].to_numpy() == group
                    destination.append(metric_row(cohort, name, first[take], labels.loc[take],
                                                   bootstrap, rng, group=str(group), intervals=False))
        frames.append(frame)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "outcome_metrics.csv", index=False)
    pd.DataFrame(suite_rows).to_csv(output / "outcome_metrics_by_suite.csv", index=False)
    pd.DataFrame(task_rows).to_csv(output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(output / "episode_alarms.csv", index=False)
    summary = {
        "schema": SCHEMA,
        "seal_sha256": sha256(output / "sealed_manifest.json"),
        "primary_metrics": rows,
        "method_candidate_count": manifest["candidate_count"],
        "outcomes_used_only_after_seal": True,
        "external_pristine_holdout": False,
        "timing_target": "original episode endpoint, not physical failure onset",
        "no_accuracy_improvement_claim_without_comparison": True,
        "artifacts": {path.name: sha256(path) for path in
                      (output / "outcome_metrics.csv", output / "outcome_metrics_by_suite.csv",
                       output / "outcome_metrics_by_task.csv", output / "episode_alarms.csv")},
    }
    write_json(output / "evaluation_summary.json", summary)
    print(metrics[["cohort", "detector", "tp", "fp", "precision", "risk_recall",
                   "timely_fpr", "early4_risk_recall", "before_half_recall"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("seal", "evaluate", "all"), default="all")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("bootstrap must be positive")
    if args.stage in ("seal", "all"):
        seal(args.output)
    if args.stage in ("evaluate", "all"):
        evaluate(args.output, args.bootstrap)


if __name__ == "__main__":
    main()
