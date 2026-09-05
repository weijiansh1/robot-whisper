#!/usr/bin/env python3
"""Freeze and evaluate the causal MoE-only layerwise persistence alarm."""

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


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "ONLINE_LAYERWISE_PERSISTENCE_V3_PROTOCOL.md"
DEFAULT_LAYER_ROOT = HERE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = HERE / "results/timeout_extension_plus10"
DEFAULT_LEGACY = HERE / "results/online_moe_only_cascade_v2/first_alarms.npz"
DEFAULT_OUTPUT = HERE / "results/online_layerwise_persistence_v3"

OPERATING_QUANTILES = (0.70, 0.75)
PRIMARY_QUANTILE = 0.75
WIDTH = 4
CONFIRMATIONS = 4
EARLY_LEADS = (2, 4, 8, 12)


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


def route_scores(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    mobility = np.asarray(cache["mobility"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    combined = np.median(np.where(valid[:, :, None], mobility, 0.0), axis=2)
    combined[~valid] = np.nan
    smoothed = dev.trailing_mean(combined, WIDTH)
    instant = -smoothed
    persistent = dev.persistent_score(instant, CONFIRMATIONS)
    instant[~valid] = np.nan
    persistent[~valid] = np.nan
    return instant, persistent


def main_alarms(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    instant, persistent = route_scores(cache)
    task_index = cache["task_index"].astype(int)
    init_state = cache["init_state_id"].astype(int)
    original_quantiles = dev.QUANTILES
    try:
        threshold_all = dev.crossfit_thresholds(
            dev.row_max(instant), task_index, init_state
        )
    finally:
        if dev.QUANTILES != original_quantiles:
            raise AssertionError("development quantiles changed during replay")
    positions = [dev.QUANTILES.index(value) for value in OPERATING_QUANTILES]
    thresholds = threshold_all[:, positions]
    first = np.full((len(instant), len(OPERATING_QUANTILES)), -1, np.int16)
    valid = cache["valid"].astype(bool)
    for position in range(len(OPERATING_QUANTILES)):
        trigger = (
            np.isfinite(persistent)
            & (persistent > thresholds[:, position, None])
            & valid
        )
        first[:, position] = dev.first_query(trigger)
    return first, thresholds


def reference_for_task(
    task: str,
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    for cache in (main_reference, extra_reference):
        names = cache["task_names"].astype(str).tolist()
        if task in names:
            position = names.index(task)
            take = np.flatnonzero(cache["task_index"].astype(int) == position)
            if len(take) != 400:
                raise ValueError(f"reference task has {len(take)} episodes: {task}")
            return cache, take
    raise KeyError(f"no historical reference routes for {task}")


def external_alarms(
    external: dict[str, np.ndarray],
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    _, persistent = route_scores(external)
    valid = external["valid"].astype(bool)
    task_index = external["task_index"].astype(int)
    task_names = external["task_names"].astype(str)
    first = np.full((len(persistent), len(OPERATING_QUANTILES)), -1, np.int16)
    thresholds = np.full(
        (len(persistent), len(OPERATING_QUANTILES)), np.nan, np.float32
    )
    threshold_rows: list[dict[str, Any]] = []

    score_cache: dict[int, np.ndarray] = {}
    for task_position, task in enumerate(task_names):
        test = np.flatnonzero(task_index == task_position)
        reference, take = reference_for_task(task, main_reference, extra_reference)
        cache_key = id(reference)
        if cache_key not in score_cache:
            score_cache[cache_key] = route_scores(reference)[0]
        calibration_peak = dev.row_max(score_cache[cache_key][take])
        for quantile_position, quantile in enumerate(OPERATING_QUANTILES):
            threshold = dev.quantile_higher(calibration_peak, quantile)
            if not np.isfinite(threshold):
                raise ValueError(f"non-finite external threshold for {task}")
            thresholds[test, quantile_position] = threshold
            trigger = (
                np.isfinite(persistent[test])
                & (persistent[test] > threshold)
                & valid[test]
            )
            first[test, quantile_position] = dev.first_query(trigger)
            threshold_rows.append(
                {
                    "task": task,
                    "quantile": quantile,
                    "threshold": threshold,
                    "reference_episodes": len(take),
                    "outcomes_used": False,
                }
            )
    if not np.isfinite(thresholds).all():
        raise ValueError("external threshold assignment is incomplete")
    return first, thresholds, pd.DataFrame(threshold_rows)


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
        [
            "task",
            "episode",
            "original_failure",
            "failure",
            "late_success_plus10_queries",
        ]
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
) -> dict[str, Any]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    timely = ~risk
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    persistent = labels["failure"].to_numpy(dtype=bool)
    lead = labels["length"].to_numpy(dtype=int) - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    tasks = labels["task"].to_numpy(dtype=str)
    row: dict[str, Any] = {
        "cohort": cohort,
        "group": group,
        "detector": detector,
        "episodes": len(labels),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "late_n": int(late.sum()),
        "persistent_n": int(persistent.sum()),
        "tp": tp,
        "fp": fp,
        "fn": int((~alarm & risk).sum()),
        "tn": int((~alarm & timely).sum()),
        "risk_recall": ratio(tp, risk.sum()),
        "timely_fpr": ratio(fp, timely.sum()),
        "precision": ratio(tp, tp + fp),
        "late_recall": ratio((alarm & late).sum(), late.sum()),
        "persistent_recall": ratio((alarm & persistent).sum(), persistent.sum()),
        "detected_risk_lead_median": (
            float(np.median(lead[alarm & risk])) if tp else float("nan")
        ),
    }
    for name, numerator, denominator in (
        ("risk_recall", alarm & risk, risk),
        ("timely_fpr", alarm & timely, timely),
        ("precision", alarm & risk, alarm),
    ):
        low, high = clustered_interval(tasks, numerator, denominator, draws, rng)
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    for early in EARLY_LEADS:
        row[f"early{early}_risk_recall"] = ratio(
            (alarm & risk & (lead >= early)).sum(), risk.sum()
        )
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
    values = labels[column].to_numpy(dtype=str)
    for group in np.unique(values):
        take = values == group
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
                )
            )
    return rows


def main() -> None:
    args = parse_args()
    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")

    # Alarm generation is completed and persisted before outcomes are loaded.
    main_first, main_thresholds = main_alarms(main_cache)
    external_first, external_thresholds, threshold_table = external_alarms(
        external_cache, main_cache, extra_cache
    )
    args.output.mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(
        args.output / "external_outcome_blind_thresholds.csv", index=False
    )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.layerwise_persistence_v3.first.v1"),
        quantiles=np.asarray(OPERATING_QUANTILES),
        primary_quantile=np.asarray(PRIMARY_QUANTILE),
        width=np.asarray(WIDTH),
        confirmations=np.asarray(CONFIRMATIONS),
        main_first=main_first,
        main_thresholds=main_thresholds,
        external_first=external_first,
        external_thresholds=external_thresholds,
    )

    main_labels = aligned_labels(
        main_cache,
        args.label_root / "development_main_clean_labels.csv",
        "development_main",
    )
    external_labels = aligned_labels(
        external_cache,
        args.label_root / "external_8b_clean_labels.csv",
        "external_8b",
    )
    legacy = dev.load_npz(args.legacy)
    cohorts = (
        ("development_main", main_first, main_labels, "main_mobility_warning_k2"),
        (
            "external_8b",
            external_first,
            external_labels,
            "external_mobility_warning_k2",
        ),
    )
    rng = np.random.default_rng(args.seed)
    outcome_rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    episode_frames: list[pd.DataFrame] = []

    for cohort, first, labels, legacy_key in cohorts:
        detectors = {"legacy_back_mean_q95_k2": legacy[legacy_key].astype(np.int16)}
        for position, quantile in enumerate(OPERATING_QUANTILES):
            detectors[f"layer_median_q{int(quantile * 100)}_k4"] = first[:, position]
        for detector, values in detectors.items():
            outcome_rows.append(
                metric_row(cohort, detector, values, labels, args.bootstrap, rng)
            )
        suite_rows.extend(
            grouped_rows(cohort, detectors, labels, "suite", args.bootstrap, rng)
        )
        task_rows.extend(
            grouped_rows(cohort, detectors, labels, "task", args.bootstrap, rng)
        )
        episode = labels.copy()
        for detector, values in detectors.items():
            episode[f"first_{detector}_query"] = values
        episode_frames.append(episode)

    outcome = pd.DataFrame(outcome_rows)
    outcome.to_csv(args.output / "outcome_metrics.csv", index=False)
    pd.DataFrame(suite_rows).to_csv(
        args.output / "outcome_metrics_by_suite.csv", index=False
    )
    pd.DataFrame(task_rows).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )
    pd.concat(episode_frames, ignore_index=True).to_csv(
        args.output / "episode_alarms.csv", index=False
    )

    primary = outcome[outcome["detector"] == "layer_median_q75_k4"]
    summary = {
        "schema": "himoe.layerwise_persistence_v3.evaluation.v1",
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_operating_point_frozen_before_outcome_merge": True,
        "risk_target": "late_or_persistent_original_horizon_failure",
        "timely_success_is_negative": True,
        "primary_quantile": PRIMARY_QUANTILE,
        "width": WIDTH,
        "confirmations": CONFIRMATIONS,
        "primary_metrics": primary.to_dict(orient="records"),
        "all_metrics": outcome.to_dict(orient="records"),
        "artifacts": {
            "protocol_sha256": sha256(PROTOCOL),
            "evaluator_sha256": sha256(Path(__file__)),
            "sealed_first_alarms_sha256": sha256(
                args.output / "sealed_first_alarms.npz"
            ),
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        outcome[
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
