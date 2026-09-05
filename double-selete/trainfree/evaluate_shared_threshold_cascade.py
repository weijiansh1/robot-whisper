#!/usr/bin/env python3
"""Reproduce the post-hoc shared-threshold cascade on both route cohorts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_online_closed_loop_v2 as common
import evaluate_online_precision_cascade as external_evaluator
import online_closed_loop_alarm_v2 as v2
import online_multihead_alarm as v1
from precision_cascade_monitor import mobility_detector_scores


HERE = Path(__file__).resolve().parent
DEFAULT_MAIN_CACHE = HERE / "results/online_multihead_hub/unlabeled_query_features.npz"
DEFAULT_MAIN_LABELS = HERE / "results/hub_binary_audit/episode_physical_labels.csv"
DEFAULT_MAIN_ONSETS = HERE / "results/online_multihead_hub/physical_proxy_onsets.csv"
DEFAULT_EXTERNAL = HERE / "results/online_precision_cascade_external"
DEFAULT_OUTPUT = HERE / "results/online_precision_cascade_shared"
LEVELS = ("mobility_w4_k1", "mobility_w4_k2", "mobility_w4_k4")
QUANTILES = (0.95, 0.975, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-cache", type=Path, default=DEFAULT_MAIN_CACHE)
    parser.add_argument("--main-labels", type=Path, default=DEFAULT_MAIN_LABELS)
    parser.add_argument("--main-onsets", type=Path, default=DEFAULT_MAIN_ONSETS)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260904)
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


def first_query(trigger: np.ndarray) -> np.ndarray:
    trigger = np.asarray(trigger, dtype=bool)
    fired = trigger.any(axis=1)
    first = np.argmax(trigger, axis=1).astype(np.int16)
    first[~fired] = -1
    return first


def main_labels(cache: dict[str, np.ndarray], path: Path) -> pd.DataFrame:
    index = common.build_index(cache)
    labels = pd.read_csv(path)[
        ["task", "episode", "init_state_id", "flow_noise_seed", "episode_length", "failure"]
    ]
    merged = index.merge(
        labels,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_label"),
    ).sort_values("sealed_row")
    if merged["failure"].isna().any():
        raise ValueError("main labels do not align")
    return merged.reset_index(drop=True)


def main_first_alarms(cache: dict[str, np.ndarray]) -> np.ndarray:
    features = np.asarray(cache["features"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    task_index = np.asarray(cache["task_index"], dtype=int)
    init_state = np.asarray(cache["init_state_id"], dtype=int)
    feature_index = {name: index for index, name in enumerate(v1.FEATURES)}
    output = np.full(
        (len(features), len(LEVELS), len(QUANTILES)), -1, dtype=np.int16
    )
    for task_position in range(len(cache["task_names"])):
        take = np.flatnonzero(task_index == task_position)
        task_valid = valid[take]
        scores = mobility_detector_scores(
            features[take, :, feature_index["route_mobility"]], task_valid
        )
        task_init = init_state[take]
        for held_out in np.unique(task_init):
            reference = task_init != held_out
            test = ~reference
            global_test = take[test]
            calibration_peak = v2.nan_row_max(
                scores["mobility_w4_k1"][reference]
            )
            for quantile_position, quantile in enumerate(QUANTILES):
                threshold = v2.quantile_higher(calibration_peak, quantile)
                for level_position, level in enumerate(LEVELS):
                    trigger = (
                        np.isfinite(scores[level][test])
                        & (scores[level][test] > threshold)
                        & task_valid[test]
                    )
                    output[global_test, level_position, quantile_position] = (
                        first_query(trigger)
                    )
    return output


def external_first_alarms(sealed: dict[str, np.ndarray]) -> np.ndarray:
    detector_names = sealed["detector_names"].astype(str).tolist()
    source_quantiles = sealed["quantiles"].astype(float)
    output = np.full(
        (len(sealed["episode"]), len(LEVELS), len(QUANTILES)), -1, dtype=np.int16
    )
    threshold_source = detector_names.index("mobility_w4_k1")
    for quantile_position, quantile in enumerate(QUANTILES):
        source_q = int(np.argmin(np.abs(source_quantiles - quantile)))
        threshold = sealed["thresholds"][:, threshold_source, source_q, None]
        for level_position, level in enumerate(LEVELS):
            detector = detector_names.index(level)
            trigger = (
                np.isfinite(sealed["scores"][:, :, detector])
                & (sealed["scores"][:, :, detector] > threshold)
                & sealed["valid"]
            )
            output[:, level_position, quantile_position] = first_query(trigger)
    return output


def metric_tables(
    cohort: str,
    first: np.ndarray,
    labels: pd.DataFrame,
    lengths: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    failure = labels["failure"].to_numpy(dtype=bool)
    success = ~failure
    tasks = labels["task"].to_numpy(dtype=str)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    for level_position, level in enumerate(LEVELS):
        for quantile_position, quantile in enumerate(QUANTILES):
            alarm_query = first[:, level_position, quantile_position]
            alarm = alarm_query >= 0
            lead = lengths - 1 - alarm_query
            tp = int((alarm & failure).sum())
            fp = int((alarm & success).sum())
            row: dict[str, Any] = {
                "cohort": cohort,
                "threshold_source": "mobility_w4_k1",
                "level": level,
                "quantile": quantile,
                "failure_n": int(failure.sum()),
                "success_n": int(success.sum()),
                "tp": tp,
                "fp": fp,
                "fn": int((~alarm & failure).sum()),
                "tn": int((~alarm & success).sum()),
                "failure_recall": common.ratio(tp, failure.sum()),
                "success_fpr": common.ratio(fp, success.sum()),
                "precision": common.ratio(tp, tp + fp),
            }
            for metric, numerator, denominator in (
                ("failure_recall", alarm & failure, failure),
                ("success_fpr", alarm & success, success),
                ("precision", alarm & failure, alarm),
            ):
                low, high = common.clustered_interval(
                    tasks, numerator, denominator, draws, rng
                )
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            for early in common.EARLY_LEADS:
                value = alarm & failure & (lead >= early)
                row[f"early{early}_recall"] = common.ratio(
                    value.sum(), failure.sum()
                )
            detected_lead = lead[alarm & failure]
            row["detected_failure_lead_median"] = (
                float(np.median(detected_lead)) if len(detected_lead) else np.nan
            )
            rows.append(row)

            for task in np.unique(tasks):
                member = tasks == task
                fail_task = failure & member
                success_task = success & member
                alarm_task = alarm & member
                task_rows.append(
                    {
                        "cohort": cohort,
                        "level": level,
                        "quantile": quantile,
                        "task": task,
                        "failure_n": int(fail_task.sum()),
                        "success_n": int(success_task.sum()),
                        "tp": int((alarm & fail_task).sum()),
                        "fp": int((alarm & success_task).sum()),
                        "failure_recall": common.ratio(
                            (alarm & fail_task).sum(), fail_task.sum()
                        ),
                        "success_fpr": common.ratio(
                            (alarm & success_task).sum(), success_task.sum()
                        ),
                        "precision": common.ratio(
                            (alarm & fail_task).sum(), alarm_task.sum()
                        ),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(task_rows)


def onset_table(
    cache: dict[str, np.ndarray], first: np.ndarray, onset_path: Path
) -> pd.DataFrame:
    index = common.build_index(cache)[["sealed_row", "task", "episode"]]
    onsets = pd.read_csv(onset_path).drop(columns="sealed_row").merge(
        index, on=["task", "episode"], how="left", validate="one_to_one"
    )
    usable = onsets[onsets["onset_query"] >= 0]
    rows: list[dict[str, Any]] = []
    for level_position, level in enumerate(LEVELS):
        for quantile_position, quantile in enumerate(QUANTILES):
            alarm_query = first[
                usable["sealed_row"].to_numpy(dtype=int),
                level_position,
                quantile_position,
            ]
            onset = usable["onset_query"].to_numpy(dtype=int)
            alarm = alarm_query >= 0
            relative = alarm_query - onset
            rows.append(
                {
                    "level": level,
                    "quantile": quantile,
                    "onset_n": len(usable),
                    "alarm_any_rate": float(alarm.mean()),
                    "before_onset_rate": float(np.mean(alarm & (relative <= -1))),
                    "strict_precursor_m4_m1_rate": float(
                        np.mean(alarm & (relative >= -4) & (relative <= -1))
                    ),
                    "timely_m2_p2_rate": float(
                        np.mean(alarm & (relative >= -2) & (relative <= 2))
                    ),
                    "reaction_p0_p4_rate": float(
                        np.mean(alarm & (relative >= 0) & (relative <= 4))
                    ),
                    "first_alarm_relative_median": (
                        float(np.median(relative[alarm])) if alarm.any() else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    main_cache = v1.load_feature_cache(args.main_cache)
    main_outcome = main_labels(main_cache, args.main_labels)
    main_first = main_first_alarms(main_cache)

    external_evaluator.verify_seal(args.external)
    external_sealed = external_evaluator.load_sealed(args.external)
    external_outcome, _ = external_evaluator.reveal_labels(
        external_evaluator.DEFAULT_CACHE_ROOT, external_sealed
    )
    external_first = external_first_alarms(external_sealed)

    main_metrics, main_tasks = metric_tables(
        "development_main",
        main_first,
        main_outcome,
        main_cache["length"].astype(int),
        args.bootstrap,
        args.seed,
    )
    external_metrics, external_tasks = metric_tables(
        "external_8b_posthoc",
        external_first,
        external_outcome,
        external_sealed["length"].astype(int),
        args.bootstrap,
        args.seed + 1,
    )
    onset = onset_table(main_cache, main_first, args.main_onsets)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat((main_metrics, external_metrics), ignore_index=True).to_csv(
        args.output / "outcome_metrics.csv", index=False
    )
    pd.concat((main_tasks, external_tasks), ignore_index=True).to_csv(
        args.output / "outcome_metrics_by_task.csv", index=False
    )
    onset.to_csv(args.output / "development_onset_metrics.csv", index=False)
    np.savez_compressed(
        args.output / "first_alarms.npz",
        schema=np.asarray("himoe.shared_threshold_cascade.posthoc.v1"),
        levels=np.asarray(LEVELS),
        quantiles=np.asarray(QUANTILES, dtype=np.float32),
        main_first=main_first,
        external_first=external_first,
    )
    metrics = pd.concat((main_metrics, external_metrics), ignore_index=True)
    summary = {
        "schema": "himoe.shared_threshold_cascade.evaluation.v1",
        "status": "post-hoc; external outcomes were inspected before this rule was frozen",
        "runtime_outcome_free": True,
        "threshold_calibration_outcome_free": True,
        "method_selection_outcome_free": False,
        "recommended_warning": metrics[
            (metrics["cohort"] == "external_8b_posthoc")
            & (metrics["level"] == "mobility_w4_k2")
            & np.isclose(metrics["quantile"], 0.95)
        ].iloc[0].to_dict(),
        "recommended_hard_alarm": metrics[
            (metrics["cohort"] == "external_8b_posthoc")
            & (metrics["level"] == "mobility_w4_k4")
            & np.isclose(metrics["quantile"], 0.95)
        ].iloc[0].to_dict(),
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    show = metrics[
        (metrics["quantile"].isin([0.95, 0.975]))
        & (metrics["level"].isin(["mobility_w4_k2", "mobility_w4_k4"]))
    ]
    print(
        show[
            [
                "cohort",
                "level",
                "quantile",
                "tp",
                "fp",
                "failure_recall",
                "success_fpr",
                "precision",
                "early8_recall",
                "detected_failure_lead_median",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
