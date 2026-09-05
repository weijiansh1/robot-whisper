#!/usr/bin/env python3
"""Evaluate the frozen train-free MoE lock-in/instability alarm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_layerwise_alarm_development as dev
import evaluate_layerwise_persistence_v3 as v3


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUNDLE = HERE.parent
CANONICAL = WORKSPACE / "double-selete/trainfree"
PROTOCOL = BUNDLE / "method/ONLINE_DUAL_REGIME_V4_PROTOCOL.md"
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = CANONICAL / "results/timeout_extension_plus10"
DEFAULT_LEGACY = CANONICAL / "results/online_moe_only_cascade_v2/first_alarms.npz"
DEFAULT_OUTPUT = BUNDLE / "results/cache_new_v4"

LOCK_QUANTILE = 0.75
LOCK_WIDTH = 4
LOCK_CONFIRMATIONS = 4
INSTABILITY_LAYER = "L5"
INSTABILITY_QUANTILE = 0.80
INSTABILITY_WIDTH = 4
INSTABILITY_CONFIRMATIONS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--legacy", type=Path, default=DEFAULT_LEGACY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
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


def head_scores(
    cache: dict[str, np.ndarray], head: str
) -> tuple[np.ndarray, np.ndarray]:
    mobility = np.asarray(cache["mobility"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    if head == "lock":
        values = np.median(np.where(valid[:, :, None], mobility, 0.0), axis=2)
        width = LOCK_WIDTH
        confirmations = LOCK_CONFIRMATIONS
        direction = -1.0
    elif head == "instability":
        layer_names = cache["layer_names"].astype(str).tolist()
        position = layer_names.index(INSTABILITY_LAYER)
        values = np.where(valid, mobility[:, :, position], 0.0)
        width = INSTABILITY_WIDTH
        confirmations = INSTABILITY_CONFIRMATIONS
        direction = 1.0
    else:
        raise KeyError(head)
    values = values.astype(np.float32, copy=False)
    values[~valid] = np.nan
    instant = direction * dev.trailing_mean(values, width)
    persistent = dev.persistent_score(instant, confirmations)
    instant[~valid] = np.nan
    persistent[~valid] = np.nan
    return instant, persistent


def first_from_threshold(
    persistent: np.ndarray, valid: np.ndarray, threshold: np.ndarray | float
) -> np.ndarray:
    threshold = np.asarray(threshold)
    if threshold.ndim == 1:
        threshold = threshold[:, None]
    trigger = np.isfinite(persistent) & (persistent > threshold) & valid
    return dev.first_query(trigger)


def main_head(
    cache: dict[str, np.ndarray], head: str, quantile: float
) -> tuple[np.ndarray, np.ndarray]:
    instant, persistent = head_scores(cache, head)
    all_thresholds = dev.crossfit_thresholds(
        dev.row_max(instant),
        cache["task_index"].astype(int),
        cache["init_state_id"].astype(int),
    )
    threshold = all_thresholds[:, dev.QUANTILES.index(quantile)]
    first = first_from_threshold(persistent, cache["valid"].astype(bool), threshold)
    return first, threshold


def external_head(
    external: dict[str, np.ndarray],
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
    head: str,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    _, persistent = head_scores(external, head)
    valid = external["valid"].astype(bool)
    task_index = external["task_index"].astype(int)
    task_names = external["task_names"].astype(str)
    first = np.full(len(persistent), -1, np.int16)
    thresholds = np.full(len(persistent), np.nan, np.float32)
    rows: list[dict[str, Any]] = []
    score_cache: dict[int, np.ndarray] = {}

    for task_position, task in enumerate(task_names):
        test = np.flatnonzero(task_index == task_position)
        reference, take = v3.reference_for_task(task, main_reference, extra_reference)
        key = id(reference)
        if key not in score_cache:
            score_cache[key] = head_scores(reference, head)[0]
        calibration_peak = dev.row_max(score_cache[key][take])
        threshold = dev.quantile_higher(calibration_peak, quantile)
        if not np.isfinite(threshold):
            raise ValueError(f"non-finite {head} threshold for {task}")
        thresholds[test] = threshold
        first[test] = first_from_threshold(persistent[test], valid[test], threshold)
        rows.append(
            {
                "task": task,
                "head": head,
                "quantile": quantile,
                "threshold": threshold,
                "reference_episodes": len(take),
                "outcomes_used": False,
            }
        )
    if not np.isfinite(thresholds).all():
        raise ValueError(f"incomplete external thresholds for {head}")
    return first, thresholds, rows


def first_or(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    stack = np.stack([left, right])
    available = stack >= 0
    output = np.where(available, stack, np.iinfo(np.int16).max).min(axis=0)
    output[~available.any(axis=0)] = -1
    return output.astype(np.int16)


def main() -> None:
    args = parse_args()
    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")

    main_lock, main_lock_threshold = main_head(main_cache, "lock", LOCK_QUANTILE)
    main_instability, main_instability_threshold = main_head(
        main_cache, "instability", INSTABILITY_QUANTILE
    )
    external_lock, external_lock_threshold, lock_rows = external_head(
        external_cache,
        main_cache,
        extra_cache,
        "lock",
        LOCK_QUANTILE,
    )
    (
        external_instability,
        external_instability_threshold,
        instability_rows,
    ) = external_head(
        external_cache,
        main_cache,
        extra_cache,
        "instability",
        INSTABILITY_QUANTILE,
    )
    main_dual = first_or(main_lock, main_instability)
    external_dual = first_or(external_lock, external_instability)

    # Persist all label-free alarm arrays before joining outcome tables.
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(lock_rows + instability_rows).to_csv(
        args.output / "external_outcome_blind_thresholds.csv", index=False
    )
    np.savez_compressed(
        args.output / "deployment_profiles.npz",
        schema=np.asarray("himoe.dual_regime_v4.profile.v1"),
        task_names=external_cache["task_names"],
        lock_thresholds=np.asarray(
            [row["threshold"] for row in lock_rows], dtype=np.float32
        ),
        instability_thresholds=np.asarray(
            [row["threshold"] for row in instability_rows], dtype=np.float32
        ),
        lock_quantile=np.asarray(LOCK_QUANTILE),
        instability_quantile=np.asarray(INSTABILITY_QUANTILE),
    )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.dual_regime_v4.first.v1"),
        lock_quantile=np.asarray(LOCK_QUANTILE),
        lock_width=np.asarray(LOCK_WIDTH),
        lock_confirmations=np.asarray(LOCK_CONFIRMATIONS),
        instability_layer=np.asarray(INSTABILITY_LAYER),
        instability_quantile=np.asarray(INSTABILITY_QUANTILE),
        instability_width=np.asarray(INSTABILITY_WIDTH),
        instability_confirmations=np.asarray(INSTABILITY_CONFIRMATIONS),
        main_lock=main_lock,
        main_instability=main_instability,
        main_dual=main_dual,
        main_lock_threshold=main_lock_threshold,
        main_instability_threshold=main_instability_threshold,
        external_lock=external_lock,
        external_instability=external_instability,
        external_dual=external_dual,
        external_lock_threshold=external_lock_threshold,
        external_instability_threshold=external_instability_threshold,
    )

    main_labels = v3.aligned_labels(
        main_cache,
        args.label_root / "development_main_clean_labels.csv",
        "development_main",
    )
    external_labels = v3.aligned_labels(
        external_cache,
        args.label_root / "external_8b_clean_labels.csv",
        "external_8b",
    )
    legacy = dev.load_npz(args.legacy)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []

    for cohort, labels, legacy_key, lock, instability, dual in (
        (
            "development_main",
            main_labels,
            "main_mobility_warning_k2",
            main_lock,
            main_instability,
            main_dual,
        ),
        (
            "external_8b",
            external_labels,
            "external_mobility_warning_k2",
            external_lock,
            external_instability,
            external_dual,
        ),
    ):
        detectors = {
            "legacy_back_mean_q95_k2": legacy[legacy_key].astype(np.int16),
            "lock_layer_median_q75_k4": lock,
            "instability_L5_q80_k8": instability,
            "dual_regime_or": dual,
        }
        for detector, first in detectors.items():
            rows.append(
                v3.metric_row(cohort, detector, first, labels, args.bootstrap, rng)
            )
        task_rows.extend(
            v3.grouped_rows(cohort, detectors, labels, "task", args.bootstrap, rng)
        )
        episode = labels.copy()
        for detector, first in detectors.items():
            episode[f"first_{detector}_query"] = first
        episode_frames.append(episode)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "outcome_metrics.csv", index=False)
    pd.DataFrame(task_rows).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )
    pd.concat(episode_frames, ignore_index=True).to_csv(
        args.output / "episode_alarms.csv", index=False
    )

    summary = {
        "schema": "himoe.dual_regime_v4.evaluation.v1",
        "status": (
            "post-hoc generalization replay; the external cohort was previously "
            "inspected for v3, but the v4 instability head was selected on "
            "development outcomes and frozen before its external alarm replay"
        ),
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_rule_frozen_before_candidate_alarm_generation": True,
        "external_cohort_pristine_holdout": False,
        "risk_target": "late_or_persistent_original_horizon_failure",
        "timely_success_is_negative": True,
        "metrics": metrics.to_dict(orient="records"),
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "evaluator_sha256": sha256(Path(__file__)),
            "sealed_first_alarms_sha256": sha256(
                args.output / "sealed_first_alarms.npz"
            ),
            "deployment_profiles_sha256": sha256(
                args.output / "deployment_profiles.npz"
            ),
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
                "late_recall",
                "persistent_recall",
                "early4_risk_recall",
                "detected_risk_lead_median",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
