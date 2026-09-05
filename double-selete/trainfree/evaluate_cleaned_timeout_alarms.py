#!/usr/bin/env python3
"""Recompute online-alarm endpoint metrics after timeout-success cleaning."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_online_closed_loop_v2 as common
import online_multihead_alarm as v1


HERE = Path(__file__).resolve().parent
DEFAULT_EXTENSION = HERE / "results/timeout_extension_plus10"
DEFAULT_SHARED = HERE / "results/online_precision_cascade_shared"
DEFAULT_EXTERNAL = HERE / "results/online_precision_cascade_external"
DEFAULT_MAIN_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def metric_row(
    cohort: str,
    detector: str,
    quantile: float,
    alarm: np.ndarray,
    original_failure: np.ndarray,
    clean_failure: np.ndarray,
    late_success: np.ndarray,
) -> dict[str, Any]:
    success = ~clean_failure
    tp = int((alarm & clean_failure).sum())
    fp = int((alarm & success).sum())
    original_tp = int((alarm & original_failure).sum())
    original_fp = int((alarm & ~original_failure).sum())
    return {
        "cohort": cohort,
        "detector": detector,
        "quantile": quantile,
        "episodes": len(alarm),
        "original_failures": int(original_failure.sum()),
        "late_successes_removed": int(late_success.sum()),
        "remaining_failures": int(clean_failure.sum()),
        "alarm_n": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & clean_failure).sum()),
        "tn": int((~alarm & success).sum()),
        "failure_recall": ratio(tp, int(clean_failure.sum())),
        "success_fpr": ratio(fp, int(success.sum())),
        "precision": ratio(tp, tp + fp),
        "alarms_on_late_successes": int((alarm & late_success).sum()),
        "original_tp": original_tp,
        "original_fp": original_fp,
        "original_recall": ratio(original_tp, int(original_failure.sum())),
        "original_precision": ratio(original_tp, original_tp + original_fp),
    }


def main_alignment(
    extension: Path, cache: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = common.build_index(cache)[["sealed_row", "task", "episode"]]
    labels = pd.read_csv(extension / "development_main_clean_labels.csv")[
        ["task", "episode", "original_failure", "failure", "late_success_plus10_queries"]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("sealed_row")
    if merged["failure"].isna().any():
        raise RuntimeError("main cleaned labels do not align with the route cache")
    return (
        merged["original_failure"].to_numpy(dtype=bool),
        merged["failure"].to_numpy(dtype=bool),
        merged["late_success_plus10_queries"].to_numpy(dtype=bool),
    )


def external_alignment(
    extension: Path, sealed: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_names = sealed["task_names"].astype(str)
    index = pd.DataFrame(
        {
            "sealed_row": np.arange(len(sealed["episode"])),
            "task": task_names[sealed["task_index"].astype(int)],
            "episode": sealed["episode"].astype(int),
        }
    )
    labels = pd.read_csv(extension / "external_8b_clean_labels.csv")[
        ["task", "episode", "original_failure", "failure", "late_success_plus10_queries"]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("sealed_row")
    if merged["failure"].isna().any():
        raise RuntimeError("external cleaned labels do not align with the seal")
    return (
        merged["original_failure"].to_numpy(dtype=bool),
        merged["failure"].to_numpy(dtype=bool),
        merged["late_success_plus10_queries"].to_numpy(dtype=bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--main-cache", type=Path, default=DEFAULT_MAIN_CACHE)
    args = parser.parse_args()

    with np.load(args.shared / "first_alarms.npz", allow_pickle=False) as archive:
        first = {name: np.asarray(archive[name]) for name in archive.files}
    levels = first["levels"].astype(str).tolist()
    quantiles = first["quantiles"].astype(float)
    main_cache = v1.load_feature_cache(args.main_cache)
    with np.load(args.external / "sealed_online_scores.npz", allow_pickle=False) as archive:
        sealed = {name: np.asarray(archive[name]) for name in archive.files}

    rows: list[dict[str, Any]] = []
    cohorts = (
        (
            "development_main",
            first["main_first"],
            main_alignment(args.extension, main_cache),
        ),
        (
            "external_8b",
            first["external_first"],
            external_alignment(args.extension, sealed),
        ),
    )
    requested = (
        ("mobility_w4_k2", 0.95),
        ("mobility_w4_k4", 0.95),
        ("mobility_w4_k4", 0.975),
    )
    for cohort, alarms, labels in cohorts:
        original_failure, clean_failure, late_success = labels
        for level, quantile in requested:
            level_position = levels.index(level)
            quantile_position = int(
                np.flatnonzero(np.isclose(quantiles, quantile))[0]
            )
            alarm = alarms[:, level_position, quantile_position] >= 0
            rows.append(
                metric_row(
                    cohort,
                    level + "_shared_threshold",
                    quantile,
                    alarm,
                    original_failure,
                    clean_failure,
                    late_success,
                )
            )

    original_failure, clean_failure, late_success = external_alignment(
        args.extension, sealed
    )
    detectors = sealed["detector_names"].astype(str).tolist()
    source_quantiles = sealed["quantiles"].astype(float)
    detector_position = detectors.index("mobility_w4_k4")
    quantile_position = int(np.flatnonzero(np.isclose(source_quantiles, 0.99))[0])
    alarm = sealed["alarms"][:, :, detector_position, quantile_position].any(axis=1)
    rows.append(
        metric_row(
            "external_8b",
            "mobility_w4_k4_independently_sealed",
            0.99,
            alarm,
            original_failure,
            clean_failure,
            late_success,
        )
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.extension / "cleaned_alarm_metrics.csv", index=False)
    print(
        frame[
            [
                "cohort",
                "detector",
                "quantile",
                "late_successes_removed",
                "tp",
                "fp",
                "failure_recall",
                "success_fpr",
                "precision",
                "alarms_on_late_successes",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
