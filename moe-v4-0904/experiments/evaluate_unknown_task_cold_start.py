#!/usr/bin/env python3
"""Evaluate a task-ID-free, early-MoE-calibrated dual-regime alarm."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_dual_regime_v4 as v4
import evaluate_layerwise_alarm_development as dev
import evaluate_layerwise_persistence_v3 as v3


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
DEFAULT_LAYER_ROOT = BUNDLE / "results/layerwise_mobility"
DEFAULT_REFERENCE_FINGERPRINT = (
    BUNDLE / "results/unknown_task/early_route_fingerprints.npz"
)
DEFAULT_EXTERNAL_FINGERPRINT = (
    BUNDLE / "results/unknown_task/early_route_fingerprints_external.npz"
)
DEFAULT_LABEL_ROOT = (
    WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
)
DEFAULT_OUTPUT = BUNDLE / "results/unknown_task/cold_start_v1"

PREFIX = 4
LOCK_REFERENCE_QUANTILE = 0.75
LOCK_RATIO_QUANTILE = 0.10
INSTABILITY_REFERENCE_QUANTILE = 0.80
INSTABILITY_NEIGHBORS = 12
INSTABILITY_NEIGHBOR_QUANTILE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument(
        "--reference-fingerprint", type=Path, default=DEFAULT_REFERENCE_FINGERPRINT
    )
    parser.add_argument(
        "--external-fingerprint", type=Path, default=DEFAULT_EXTERNAL_FINGERPRINT
    )
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def early_lock_mobility(cache: dict[str, np.ndarray]) -> np.ndarray:
    mobility = np.asarray(cache["mobility"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    output = np.median(np.where(valid[:, :, None], mobility, 0.0), axis=2)
    output[~valid] = np.nan
    return output


def reference_rows(cache: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    task_index = cache["task_index"].astype(int)
    lock_early = early_lock_mobility(cache)
    lock_instant, _ = v4.head_scores(cache, "lock")
    instability_instant, _ = v4.head_scores(cache, "instability")
    lock_peak = dev.row_max(lock_instant)
    instability_peak = dev.row_max(instability_instant)
    rows: list[dict[str, Any]] = []

    for position, task in enumerate(cache["task_names"].astype(str)):
        take = task_index == position
        lock_cutoff = -dev.quantile_higher(
            lock_peak[take], LOCK_REFERENCE_QUANTILE
        )
        instability_cutoff = dev.quantile_higher(
            instability_peak[take], INSTABILITY_REFERENCE_QUANTILE
        )
        episode_scale = np.nanmean(lock_early[take, :PREFIX], axis=1)
        task_scale = float(np.nanmean(episode_scale))
        rows.append(
            {
                "task": task,
                "reference_episodes": int(take.sum()),
                "lock_cutoff": lock_cutoff,
                "early_lock_scale": task_scale,
                "lock_ratio": lock_cutoff / task_scale,
                "instability_cutoff": instability_cutoff,
                "outcomes_used": False,
            }
        )
    return rows


def load_fingerprint(path: Path, cohort: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            key.removeprefix(f"{cohort}_"): np.asarray(archive[key])
            for key in archive.files
            if key.startswith(f"{cohort}_")
        }


def fingerprint_embedding(action_queries: np.ndarray) -> np.ndarray:
    probability = np.asarray(action_queries, dtype=np.float32).mean(axis=1)
    return np.sqrt(np.maximum(probability, 0.0)).reshape(len(probability), -1)


def build_reference(
    main_cache: dict[str, np.ndarray],
    extra_cache: dict[str, np.ndarray],
    main_fingerprint: dict[str, np.ndarray],
    extra_fingerprint: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = reference_rows(main_cache) + reference_rows(extra_cache)
    profiles = pd.DataFrame(rows)
    embedding = np.concatenate(
        [
            fingerprint_embedding(main_fingerprint["final_action_queries"]),
            fingerprint_embedding(extra_fingerprint["final_action_queries"]),
        ]
    )
    task_index = np.concatenate(
        [
            main_fingerprint["task_index"].astype(int),
            extra_fingerprint["task_index"].astype(int)
            + len(main_fingerprint["task_names"]),
        ]
    )
    centroids = np.vstack(
        [embedding[task_index == position].mean(axis=0) for position in profiles.index]
    ).astype(np.float32)
    return profiles, centroids


def cold_start_thresholds(
    cache: dict[str, np.ndarray],
    action_queries: np.ndarray,
    profiles: pd.DataFrame,
    centroids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    embedding = fingerprint_embedding(action_queries)
    distance = (
        np.square(embedding).sum(axis=1)[:, None]
        + np.square(centroids).sum(axis=1)[None, :]
        - 2.0 * embedding @ centroids.T
    )
    target_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    profile_names = profiles["task"].to_numpy(dtype=str)

    # In each audit fold, remove the target task's entire historical profile.
    # A genuinely new deployment task is absent from the library by construction.
    for position, task in enumerate(target_names):
        same = np.flatnonzero(profile_names == task)
        if len(same):
            distance[task_index == position, same] = np.inf
    nearest = np.argsort(distance, axis=1)[:, :INSTABILITY_NEIGHBORS]

    early_scale = np.nanmean(early_lock_mobility(cache)[:, :PREFIX], axis=1)
    lock_cutoff = np.full(len(task_index), np.nan, dtype=np.float32)
    lock_ratios = profiles["lock_ratio"].to_numpy(dtype=float)
    for position, task in enumerate(target_names):
        library = lock_ratios[profile_names != task]
        ratio = np.quantile(library, LOCK_RATIO_QUANTILE, method="higher")
        lock_cutoff[task_index == position] = ratio * early_scale[
            task_index == position
        ]

    instability_values = profiles["instability_cutoff"].to_numpy(dtype=float)
    instability_cutoff = np.quantile(
        instability_values[nearest],
        INSTABILITY_NEIGHBOR_QUANTILE,
        axis=1,
        method="higher",
    ).astype(np.float32)
    nearest_name = profile_names[nearest[:, 0]]
    nearest_distance = np.take_along_axis(distance, nearest[:, :1], axis=1)[:, 0]
    return lock_cutoff, instability_cutoff, nearest_name, nearest_distance


def alarms(
    cache: dict[str, np.ndarray],
    lock_cutoff: np.ndarray,
    instability_cutoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = cache["valid"].astype(bool)
    _, lock_persistent = v4.head_scores(cache, "lock")
    _, instability_persistent = v4.head_scores(cache, "instability")
    lock = v4.first_from_threshold(lock_persistent, valid, -lock_cutoff)
    instability = v4.first_from_threshold(
        instability_persistent, valid, instability_cutoff
    )
    return lock, instability, v4.first_or(lock, instability)


def metric(first: np.ndarray, labels: pd.DataFrame) -> dict[str, Any]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    tp = int((alarm & risk).sum())
    fp = int((alarm & ~risk).sum())
    return {
        "trajectories": len(labels),
        "risk_n": int(risk.sum()),
        "nonrisk_n": int((~risk).sum()),
        "alarms": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "recall": tp / risk.sum() if risk.any() else float("nan"),
        "precision": tp / max(tp + fp, 1),
        "nonrisk_fpr": fp / (~risk).sum(),
    }


def main() -> None:
    args = parse_args()
    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")
    main_fingerprint = load_fingerprint(args.reference_fingerprint, "main")
    extra_fingerprint = load_fingerprint(args.reference_fingerprint, "extra")
    external_fingerprint = load_fingerprint(args.external_fingerprint, "external")
    profiles, centroids = build_reference(
        main_cache, extra_cache, main_fingerprint, extra_fingerprint
    )

    args.output.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output / "outcome_blind_reference_profiles.csv", index=False)
    np.savez_compressed(
        args.output / "cold_start_reference.npz",
        schema=np.asarray("himoe.unknown_task_reference.v1"),
        prefix=np.asarray(PREFIX, dtype=np.int16),
        task_names=profiles["task"].to_numpy(dtype=str),
        action_centroids=centroids,
        lock_ratios=profiles["lock_ratio"].to_numpy(dtype=np.float32),
        instability_cutoffs=profiles["instability_cutoff"].to_numpy(
            dtype=np.float32
        ),
        lock_ratio_quantile=np.asarray(LOCK_RATIO_QUANTILE),
        instability_neighbors=np.asarray(INSTABILITY_NEIGHBORS, dtype=np.int16),
        instability_neighbor_quantile=np.asarray(INSTABILITY_NEIGHBOR_QUANTILE),
    )

    sealed: dict[str, np.ndarray] = {
        "schema": np.asarray("himoe.unknown_task_alarm.v1")
    }
    cohort_values: dict[str, dict[str, np.ndarray]] = {}
    for cohort, cache, fingerprint in (
        ("development_main", main_cache, main_fingerprint),
        ("external_8b", external_cache, external_fingerprint),
    ):
        lock_cutoff, instability_cutoff, nearest_name, nearest_distance = (
            cold_start_thresholds(
                cache,
                fingerprint["final_action_queries"],
                profiles,
                centroids,
            )
        )
        lock, instability, dual = alarms(cache, lock_cutoff, instability_cutoff)
        cohort_values[cohort] = {
            "lock": lock,
            "instability": instability,
            "dual": dual,
            "lock_cutoff": lock_cutoff,
            "instability_cutoff": instability_cutoff,
            "nearest_name": nearest_name,
            "nearest_distance": nearest_distance,
        }
        for key, value in cohort_values[cohort].items():
            sealed[f"{cohort}_{key}"] = value

    # Alarm generation is sealed before outcome labels are loaded.
    np.savez_compressed(args.output / "sealed_first_alarms.npz", **sealed)

    profile_cutoff = profiles.set_index("task")["instability_cutoff"]
    main_tasks = main_cache["task_names"].astype(str)[
        main_cache["task_index"].astype(int)
    ]
    diagnostic_rows: list[dict[str, Any]] = []
    for task in np.unique(main_tasks):
        take = main_tasks == task
        predicted = float(
            np.median(cohort_values["development_main"]["instability_cutoff"][take])
        )
        actual = float(profile_cutoff.loc[task])
        nearest = cohort_values["development_main"]["nearest_name"][take]
        suite = task.split("/", 1)[0]
        diagnostic_rows.append(
            {
                "task": task,
                "predicted_instability_cutoff": predicted,
                "held_out_task_cutoff": actual,
                "absolute_relative_error": abs(predicted - actual) / actual,
                "nearest_same_suite_fraction": float(
                    np.mean([value.split("/", 1)[0] == suite for value in nearest])
                ),
            }
        )
    diagnostic_frame = pd.DataFrame(diagnostic_rows)
    diagnostic_frame.to_csv(
        args.output / "routing_retrieval_diagnostics.csv", index=False
    )
    retrieval_diagnostics = {
        "task_cutoff_pearson": float(
            diagnostic_frame[
                ["predicted_instability_cutoff", "held_out_task_cutoff"]
            ].corr().iloc[0, 1]
        ),
        "task_cutoff_median_absolute_relative_error": float(
            diagnostic_frame["absolute_relative_error"].median()
        ),
        "episode_nearest_same_suite_fraction": float(
            diagnostic_frame["nearest_same_suite_fraction"].mean()
        ),
    }

    metrics: list[dict[str, Any]] = []
    task_metrics: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []
    for cohort, cache, labels_path in (
        (
            "development_main",
            main_cache,
            args.label_root / "development_main_clean_labels.csv",
        ),
        (
            "external_8b",
            external_cache,
            args.label_root / "external_8b_clean_labels.csv",
        ),
    ):
        labels = v3.aligned_labels(cache, labels_path, cohort)
        values = cohort_values[cohort]
        for detector in ("lock", "instability", "dual"):
            row = metric(values[detector], labels)
            row.update({"cohort": cohort, "detector": detector})
            metrics.append(row)
        tasks = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
        frame = labels.copy()
        frame["cohort"] = cohort
        frame["first_lock_query"] = values["lock"]
        frame["first_instability_query"] = values["instability"]
        frame["first_dual_query"] = values["dual"]
        frame["lock_cutoff"] = values["lock_cutoff"]
        frame["instability_cutoff"] = values["instability_cutoff"]
        frame["nearest_reference_task"] = values["nearest_name"]
        frame["nearest_reference_distance"] = values["nearest_distance"]
        episode_frames.append(frame)
        for task in np.unique(tasks):
            take = tasks == task
            row = metric(values["dual"][take], labels.loc[take].reset_index(drop=True))
            row.update({"cohort": cohort, "task": task})
            task_metrics.append(row)

    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output / "outcome_metrics.csv", index=False)
    pd.DataFrame(task_metrics).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )
    pd.concat(episode_frames, ignore_index=True).to_csv(
        args.output / "episode_alarms.csv", index=False
    )
    summary = {
        "schema": "himoe.unknown_task_evaluation.v1",
        "method": {
            "runtime_task_id_used": False,
            "runtime_inputs": ["hb_router_probs"],
            "prefix_queries": PREFIX,
            "development_protocol": "leave the target task profile out entirely",
            "lock_ratio_quantile": LOCK_RATIO_QUANTILE,
            "instability_neighbors": INSTABILITY_NEIGHBORS,
            "instability_neighbor_quantile": INSTABILITY_NEIGHBOR_QUANTILE,
        },
        "routing_retrieval_diagnostics": retrieval_diagnostics,
        "metrics": metric_frame.to_dict(orient="records"),
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2) + "\n", encoding="utf-8"
    )
    print(metric_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
