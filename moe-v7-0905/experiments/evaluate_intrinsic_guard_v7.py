#!/usr/bin/env python3
"""Calibrate, seal, and evaluate the task-agnostic intrinsic routing guard."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
METHOD = BUNDLE / "method"
sys.path.insert(0, str(METHOD))

from intrinsic_guard_monitor import (  # noqa: E402
    ACCELERATION_BASELINE_COUNT,
    ACCELERATION_BASELINE_START,
    ACCELERATION_CONFIRMATIONS,
    ACCELERATION_WIDTH,
    FREEZE_BASELINE_COUNT,
    FREEZE_BASELINE_START,
    FREEZE_CONFIRMATIONS,
    FREEZE_WIDTH,
    PERIODICITY_BASELINE_COUNT,
    PERIODICITY_BASELINE_START,
    PERIODICITY_CONFIRMATIONS,
    PERIODICITY_WIDTH,
    SCHEMA,
    first_and,
    first_from_score,
    first_or,
    intrinsic_score_arrays,
    quantile_higher,
    row_max,
)


LAYER_ROOT = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
FEATURE_ROOT = WORKSPACE / "double-selete/trainfree/results"
LABEL_ROOT = FEATURE_ROOT / "timeout_extension_plus10"
DEFAULT_OUTPUT = BUNDLE / "results/intrinsic_guard_v7"
PROTOCOL = METHOD / "ONLINE_INTRINSIC_GUARD_V7_PROTOCOL.md"

MAIN_LAYER = LAYER_ROOT / "main_reference.npz"
EXTRA_LAYER = LAYER_ROOT / "extra_reference.npz"
EXTERNAL_LAYER = LAYER_ROOT / "external_8b.npz"
MAIN_FEATURE = FEATURE_ROOT / "online_multihead_hub/unlabeled_query_features.npz"
EXTRA_FEATURE = (
    FEATURE_ROOT / "online_multihead_hub_external/unlabeled_query_features.npz"
)
EXTERNAL_FEATURE = (
    FEATURE_ROOT / "online_precision_cascade_external/unlabeled_query_features.npz"
)

FREEZE_QUANTILE = 0.975
ACCELERATION_QUANTILE = 0.70
PERIODICITY_QUANTILE = 0.65
PERIODICITY_SCALE_QUANTILE = 0.75
EARLY_LEADS = (2, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seal-only", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def feature(cache: dict[str, np.ndarray], name: str) -> np.ndarray:
    names = cache["feature_names"].astype(str).tolist()
    if name not in names:
        raise KeyError(f"missing route feature: {name}")
    return np.asarray(cache["features"][:, :, names.index(name)], dtype=np.float32)


def assert_aligned(
    layer: dict[str, np.ndarray], route: dict[str, np.ndarray], name: str
) -> None:
    for field in ("task_names", "task_index", "episode", "init_state_id", "length", "valid"):
        if not np.array_equal(layer[field], route[field]):
            raise ValueError(f"{name} layer/route cache mismatch in {field}")


def combine_reference(
    main_layer: dict[str, np.ndarray],
    extra_layer: dict[str, np.ndarray],
    main_feature: dict[str, np.ndarray],
    extra_feature: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mobility = np.concatenate((main_layer["mobility"], extra_layer["mobility"]), axis=0)
    acceleration = np.concatenate(
        (feature(main_feature, "route_acceleration"), feature(extra_feature, "route_acceleration")),
        axis=0,
    )
    periodicity = np.concatenate(
        (feature(main_feature, "lag_periodicity"), feature(extra_feature, "lag_periodicity")),
        axis=0,
    )
    if len(mobility) != 16_000:
        raise ValueError(f"expected 16,000 pooled reference trajectories, got {len(mobility)}")
    return mobility, acceleration, periodicity


def calibrate_and_score(
    reference: tuple[np.ndarray, np.ndarray, np.ndarray],
    main_layer: dict[str, np.ndarray],
    main_feature: dict[str, np.ndarray],
    external_layer: dict[str, np.ndarray],
    external_feature: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    ref_mobility, ref_acceleration, ref_periodicity = reference
    finite_periodicity = np.abs(ref_periodicity[np.isfinite(ref_periodicity)])
    periodicity_scale = float(
        np.quantile(finite_periodicity, PERIODICITY_SCALE_QUANTILE, method="linear")
    )
    reference_scores = intrinsic_score_arrays(
        ref_mobility, ref_acceleration, ref_periodicity, periodicity_scale
    )
    thresholds = {
        "freeze": quantile_higher(row_max(reference_scores["freeze"]), FREEZE_QUANTILE),
        "acceleration": quantile_higher(
            row_max(reference_scores["acceleration"]), ACCELERATION_QUANTILE
        ),
        "periodicity": quantile_higher(
            row_max(reference_scores["periodicity"]), PERIODICITY_QUANTILE
        ),
        "periodicity_scale": periodicity_scale,
    }

    cohort_scores: dict[str, dict[str, np.ndarray]] = {}
    alarms: dict[str, np.ndarray] = {}
    for cohort, layer, route in (
        ("main", main_layer, main_feature),
        ("external", external_layer, external_feature),
    ):
        scores = intrinsic_score_arrays(
            layer["mobility"],
            feature(route, "route_acceleration"),
            feature(route, "lag_periodicity"),
            periodicity_scale,
        )
        valid = layer["valid"].astype(bool)
        first_freeze = first_from_score(scores["freeze"], thresholds["freeze"], valid)
        first_acceleration = first_from_score(
            scores["acceleration_persistent"], thresholds["acceleration"], valid
        )
        first_periodicity = first_from_score(
            scores["periodicity_persistent"], thresholds["periodicity"], valid
        )
        first_turbulence = first_and(first_acceleration, first_periodicity)
        first_guard = first_or(first_freeze, first_turbulence)
        cohort_scores[cohort] = scores
        alarms[f"{cohort}_freeze"] = first_freeze
        alarms[f"{cohort}_acceleration"] = first_acceleration
        alarms[f"{cohort}_periodicity"] = first_periodicity
        alarms[f"{cohort}_turbulence"] = first_turbulence
        alarms[f"{cohort}_guard"] = first_guard
    return thresholds, cohort_scores, alarms


def write_seal(
    output: Path,
    thresholds: dict[str, float],
    alarms: dict[str, np.ndarray],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    profile_path = output / "global_profile.npz"
    np.savez_compressed(
        profile_path,
        schema=np.asarray(SCHEMA),
        freeze_threshold=np.asarray(thresholds["freeze"], dtype=np.float32),
        acceleration_threshold=np.asarray(thresholds["acceleration"], dtype=np.float32),
        periodicity_threshold=np.asarray(thresholds["periodicity"], dtype=np.float32),
        periodicity_scale=np.asarray(thresholds["periodicity_scale"], dtype=np.float32),
        freeze_quantile=np.asarray(FREEZE_QUANTILE),
        acceleration_quantile=np.asarray(ACCELERATION_QUANTILE),
        periodicity_quantile=np.asarray(PERIODICITY_QUANTILE),
        freeze_baseline_start=np.asarray(FREEZE_BASELINE_START),
        freeze_baseline_count=np.asarray(FREEZE_BASELINE_COUNT),
        freeze_width=np.asarray(FREEZE_WIDTH),
        freeze_confirmations=np.asarray(FREEZE_CONFIRMATIONS),
        acceleration_baseline_start=np.asarray(ACCELERATION_BASELINE_START),
        acceleration_baseline_count=np.asarray(ACCELERATION_BASELINE_COUNT),
        acceleration_width=np.asarray(ACCELERATION_WIDTH),
        acceleration_confirmations=np.asarray(ACCELERATION_CONFIRMATIONS),
        periodicity_baseline_start=np.asarray(PERIODICITY_BASELINE_START),
        periodicity_baseline_count=np.asarray(PERIODICITY_BASELINE_COUNT),
        periodicity_width=np.asarray(PERIODICITY_WIDTH),
        periodicity_confirmations=np.asarray(PERIODICITY_CONFIRMATIONS),
    )
    threshold_rows = [
        {
            "head": "freeze",
            "quantile": FREEZE_QUANTILE,
            "threshold": thresholds["freeze"],
            "reference_episodes": 16_000,
            "task_conditioned": False,
            "outcomes_used": False,
        },
        {
            "head": "acceleration",
            "quantile": ACCELERATION_QUANTILE,
            "threshold": thresholds["acceleration"],
            "reference_episodes": 16_000,
            "task_conditioned": False,
            "outcomes_used": False,
        },
        {
            "head": "periodicity",
            "quantile": PERIODICITY_QUANTILE,
            "threshold": thresholds["periodicity"],
            "reference_episodes": 16_000,
            "task_conditioned": False,
            "outcomes_used": False,
        },
    ]
    pd.DataFrame(threshold_rows).to_csv(output / "unlabeled_global_thresholds.csv", index=False)

    alarm_path = output / "sealed_first_alarms.npz"
    np.savez_compressed(
        alarm_path,
        schema=np.asarray("himoe.intrinsic_guard_v7.alarms.v1"),
        **alarms,
    )
    selection_path = output / "development_threshold_selection.json"
    manifest = {
        "schema": "himoe.intrinsic_guard_v7.seal.v1",
        "sealed_at_utc": datetime.now(UTC).isoformat(),
        "runtime_inputs": ["current_hb_router_probs"],
        "runtime_task_identity": False,
        "task_or_suite_parameters": False,
        "task_prototypes": False,
        "learned_weights": False,
        "gradient_updates": False,
        "query_causal": True,
        "future_queries_used": False,
        "raw_outcome_files_loaded_before_seal": False,
        "labels_used_for_threshold_calibration": [],
        "reference_policy": "one pooled 16,000-trajectory 7B routing corpus; all outcomes retained",
        "reference_episodes": 16_000,
        "reference_tasks": 40,
        "external_episodes": 15_600,
        "external_tasks": 39,
        "rule_selection_used_development_outcomes": True,
        "external_outcomes_used_for_rule_selection": False,
        "external_cohort_pristine_holdout": False,
        "thresholds": thresholds,
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "evaluator_sha256": sha256(Path(__file__)),
            "main_layer_cache_sha256": sha256(MAIN_LAYER),
            "extra_layer_cache_sha256": sha256(EXTRA_LAYER),
            "external_layer_cache_sha256": sha256(EXTERNAL_LAYER),
            "main_route_cache_sha256": sha256(MAIN_FEATURE),
            "extra_route_cache_sha256": sha256(EXTRA_FEATURE),
            "external_route_cache_sha256": sha256(EXTERNAL_FEATURE),
            "global_profile_sha256": sha256(profile_path),
            "sealed_first_alarms_sha256": sha256(alarm_path),
            "global_thresholds_sha256": sha256(output / "unlabeled_global_thresholds.csv"),
            **(
                {"development_selection_sha256": sha256(selection_path)}
                if selection_path.exists()
                else {}
            ),
        },
    }
    (output / "sealed_manifest.json").write_text(
        json.dumps(plain(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def aligned_labels(
    cache: dict[str, np.ndarray], path: Path, cohort: str
) -> pd.DataFrame:
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(path)[
        ["task", "episode", "original_failure", "failure", "late_success_plus10_queries"]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError(f"{cohort} labels do not align")
    merged["cohort"] = cohort
    merged["suite"] = merged["task"].str.split("/", n=1).str[0]
    return merged.reset_index(drop=True)


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def clustered_interval(
    tasks: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    names = np.unique(tasks)
    task_num = np.asarray([numerator[tasks == name].sum() for name in names])
    task_den = np.asarray([denominator[tasks == name].sum() for name in names])
    samples = rng.integers(0, len(names), size=(draws, len(names)))
    sampled_num = task_num[samples].sum(axis=1)
    sampled_den = task_den[samples].sum(axis=1)
    values = sampled_num / np.maximum(sampled_den, 1)
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def metric_row(
    cohort: str,
    detector: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
    group: str = "all",
    intervals: bool = True,
) -> dict[str, Any]:
    first = np.asarray(first, dtype=np.int16)
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(bool)
    timely = ~risk
    late = labels["late_success_plus10_queries"].to_numpy(bool)
    persistent = labels["failure"].to_numpy(bool)
    lead = labels["length"].to_numpy(int) - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    row: dict[str, Any] = {
        "cohort": cohort,
        "group": group,
        "detector": detector,
        "episodes": len(labels),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "late_n": int(late.sum()),
        "persistent_n": int(persistent.sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "late_recall": ratio((alarm & late).sum(), late.sum()),
        "persistent_recall": ratio((alarm & persistent).sum(), persistent.sum()),
        "detected_risk_lead_median": float(np.median(lead[alarm & risk])) if tp else float("nan"),
    }
    for early in EARLY_LEADS:
        row[f"early{early}_risk_recall"] = ratio(
            (alarm & risk & (lead >= early)).sum(), risk.sum()
        )
    if intervals:
        tasks = labels["task"].to_numpy(str)
        for name, numerator, denominator in (
            ("risk_recall", alarm & risk, risk),
            ("timely_fpr", alarm & timely, timely),
            ("precision", alarm & risk, alarm),
        ):
            low, high = clustered_interval(tasks, numerator, denominator, draws, rng)
            row[f"{name}_ci_low"] = low
            row[f"{name}_ci_high"] = high
    return row


def grouped_rows(
    cohort: str,
    detectors: dict[str, np.ndarray],
    labels: pd.DataFrame,
    column: str,
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = labels[column].to_numpy(str)
    for group in np.unique(groups):
        take = groups == group
        block = labels.loc[take].reset_index(drop=True)
        for detector, first in detectors.items():
            rows.append(
                metric_row(
                    cohort,
                    detector,
                    first[take],
                    block,
                    draws,
                    rng,
                    str(group),
                    intervals=False,
                )
            )
    return rows


def comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    unknown = pd.read_csv(
        WORKSPACE / "moe-v4-0904/results/unknown_task/cold_start_v1/outcome_metrics.csv"
    )
    v6 = pd.read_csv(WORKSPACE / "moe-v6-0905/results/spatial_guard_v6/outcome_metrics.csv")
    for cohort in ("development_main", "external_8b"):
        row = unknown[(unknown["cohort"] == cohort) & (unknown["detector"] == "dual")].iloc[0]
        records.append(
            {
                "cohort": cohort,
                "detector": "unknown_task_dual_with_task_prototypes",
                "task_agnostic": False,
                "tp": int(row.tp),
                "fp": int(row.fp),
                "risk_recall": float(row.recall),
                "precision": float(row.precision),
                "timely_fpr": float(row.nonrisk_fpr),
            }
        )
        for source_name, detector in (
            ("v4_task_profile", "v4_dual_regime_or"),
            ("v6_suite_task_profile", "spatial_guard_v6"),
        ):
            row = v6[(v6["cohort"] == cohort) & (v6["detector"] == detector)].iloc[0]
            records.append(
                {
                    "cohort": cohort,
                    "detector": source_name,
                    "task_agnostic": False,
                    "tp": int(row.tp),
                    "fp": int(row.fp),
                    "risk_recall": float(row.risk_recall),
                    "precision": float(row.precision),
                    "timely_fpr": float(row.timely_fpr),
                }
            )
        row = metrics[
            (metrics["cohort"] == cohort) & (metrics["detector"] == "intrinsic_guard_v7")
        ].iloc[0]
        records.append(
            {
                "cohort": cohort,
                "detector": "intrinsic_guard_v7",
                "task_agnostic": True,
                "tp": int(row.tp),
                "fp": int(row.fp),
                "risk_recall": float(row.risk_recall),
                "precision": float(row.precision),
                "timely_fpr": float(row.timely_fpr),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    main_layer = load_npz(MAIN_LAYER)
    extra_layer = load_npz(EXTRA_LAYER)
    external_layer = load_npz(EXTERNAL_LAYER)
    main_feature = load_npz(MAIN_FEATURE)
    extra_feature = load_npz(EXTRA_FEATURE)
    external_feature = load_npz(EXTERNAL_FEATURE)
    assert_aligned(main_layer, main_feature, "main")
    assert_aligned(extra_layer, extra_feature, "extra")
    assert_aligned(external_layer, external_feature, "external")

    reference = combine_reference(main_layer, extra_layer, main_feature, extra_feature)
    thresholds, _, alarms = calibrate_and_score(
        reference, main_layer, main_feature, external_layer, external_feature
    )
    seal = write_seal(args.output, thresholds, alarms)
    print(
        f"sealed one global profile and {len(alarms['external_guard']):,} external alarms",
        flush=True,
    )
    if args.seal_only:
        return

    # Outcome files are intentionally opened only after the profile and alarms are sealed.
    main_labels = aligned_labels(
        main_layer, LABEL_ROOT / "development_main_clean_labels.csv", "development_main"
    )
    external_labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    for prefix, cohort, labels in (
        ("main", "development_main", main_labels),
        ("external", "external_8b", external_labels),
    ):
        detectors = {
            "relative_freeze": alarms[f"{prefix}_freeze"],
            "flow_acceleration": alarms[f"{prefix}_acceleration"],
            "recurrence_loss": alarms[f"{prefix}_periodicity"],
            "confirmed_turbulence": alarms[f"{prefix}_turbulence"],
            "intrinsic_guard_v7": alarms[f"{prefix}_guard"],
        }
        for detector, first in detectors.items():
            rows.append(metric_row(cohort, detector, first, labels, args.bootstrap, rng))
        suite_rows.extend(grouped_rows(cohort, detectors, labels, "suite", args.bootstrap, rng))
        task_rows.extend(grouped_rows(cohort, detectors, labels, "task", args.bootstrap, rng))
        frame = labels.copy()
        for detector, first in detectors.items():
            frame[f"first_{detector}_query"] = first
        episode_frames.append(frame)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "outcome_metrics.csv", index=False)
    pd.DataFrame(suite_rows).to_csv(args.output / "outcome_metrics_by_suite.csv", index=False)
    pd.DataFrame(task_rows).to_csv(args.output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(episode_frames, ignore_index=True).to_csv(
        args.output / "episode_alarms.csv", index=False
    )
    comparison = comparison_table(metrics)
    comparison.to_csv(args.output / "comparison.csv", index=False)

    primary = metrics[metrics["detector"] == "intrinsic_guard_v7"]
    summary = {
        "schema": "himoe.intrinsic_guard_v7.evaluation.v1",
        "train_free": True,
        "task_agnostic_runtime": True,
        "single_global_profile": True,
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_outcomes_loaded_after_seal": True,
        "external_cohort_pristine_holdout": False,
        "risk_target": "late_or_persistent_original_horizon_failure",
        "primary_metrics": primary.to_dict(orient="records"),
        "all_metrics": metrics.to_dict(orient="records"),
        "thresholds": thresholds,
        "seal_artifacts": seal["artifacts"],
        "evaluation_artifacts": {
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
            "comparison_sha256": sha256(args.output / "comparison.csv"),
            "episode_alarms_sha256": sha256(args.output / "episode_alarms.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        metrics[
            [
                "cohort",
                "detector",
                "tp",
                "fp",
                "risk_recall",
                "timely_fpr",
                "precision",
                "early4_risk_recall",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
