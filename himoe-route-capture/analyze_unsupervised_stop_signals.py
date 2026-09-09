#!/usr/bin/env python3
"""Evaluate label-free observables for stopping one flow-matching query early.

The observables use only the prefix available after a completed denoising round.
The full round-10 action is used offline for three sharply separated purposes:

1. an oracle gate asks whether an observable ranks remaining endpoint error;
2. Goal thresholds are calibrated with leave-one-rollout-out self-supervision; and
3. the selected Goal rule is frozen before the Long holdout is scored.

No reward or task-success labels are read.  The controller-aware endpoint score
uses physical RMS on the six continuous dimensions and only the sign of the
gripper dimension.  In particular, holdout endpoint error is never used to select
a signal, estimator, direction, minimum stage, or threshold.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata, spearmanr

from analyze_denoise_stop_sweep import (
    EPS,
    LIVE_DIMS,
    STAGES,
    TOTAL_ROUNDS,
    Signal,
    TraceData,
    _candidate_thresholds,
    _rms,
    _selected_error,
    _stops_for_threshold,
    calibrate_threshold,
    endpoint_predictions,
    load_trace,
)


ORACLE_STAGES = np.arange(4, 9, dtype=np.int64)
MIN_STAGES = (6, 8)
ESTIMATORS = ("constant_velocity", "constant_acceleration")
TRAIN_RISK = 0.0
ORACLE_RHO_GATE = 0.10
MATCHED_SAVINGS = (0.10, 0.20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/flow-lead-cpu-t0s24")
    parser.add_argument(
        "--holdout-run",
        default="runs/flow-stop-sweep-long-t08s0-k1-seeds6100-6103",
    )
    parser.add_argument("--out-dir", default="analysis/unsupervised-stop-signals-v2")
    parser.add_argument(
        "--physical-reference-rms",
        type=float,
        default=None,
        help=(
            "optional 6D physical RMS reference band; default is the Goal fixed "
            "stage-8 constant-acceleration P95 (an equivalence reference, not a "
            "claimed control-safety threshold)"
        ),
    )
    return parser.parse_args()


def _physical(value: np.ndarray, data: TraceData) -> np.ndarray:
    return value * data.action_std.reshape((1,) * (value.ndim - 1) + (LIVE_DIMS,))


def _relative(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.maximum(np.abs(denominator), 1e-8)


def _candidate_spread(value: np.ndarray, query_id: np.ndarray) -> np.ndarray:
    """Repeat each query cloud's RMS spread on every candidate row."""

    result = np.zeros(len(value), dtype=np.float64)
    for query in np.unique(query_id):
        rows = query_id == query
        if rows.sum() <= 1:
            continue
        centered = value[rows] - value[rows].mean(axis=0, keepdims=True)
        result[rows] = float(np.sqrt(np.mean(np.square(centered))))
    return result


def _endpoint_at_stage(
    predictions: dict[str, np.ndarray], estimator: str, stage: int
) -> np.ndarray:
    return predictions[estimator][:, stage - int(STAGES[0])]


def controller_endpoint_error(
    data: TraceData,
    prediction: np.ndarray,
    physical_reference_rms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return controller-aware score, 6D physical RMS, and gripper sign flips.

    PandaGripper consumes only ``sign(action)``.  A flip is encoded above the
    continuous reference band so existing scalar threshold machinery can enforce
    it without treating the magnitude of the seventh dimension as meaningful.
    """

    final = data.x[:, None, TOTAL_ROUNDS]
    continuous = (prediction[..., :6] - final[..., :6]) * data.action_std[
        None, None, None, :6
    ]
    continuous_rms = _rms(continuous, (2, 3))
    gripper_flip = np.any(
        np.sign(prediction[..., 6]) != np.sign(final[..., 6]), axis=2
    )
    flip_score = physical_reference_rms + np.maximum(
        continuous_rms, physical_reference_rms
    )
    score = np.where(gripper_flip, flip_score, continuous_rms)
    return score, continuous_rms, gripper_flip


def build_prefix_signals(data: TraceData) -> list[Signal]:
    """Build prefix-only instability scores; smaller consistently means safer."""

    x = data.x
    physical_x = _physical(x, data)
    predictions = endpoint_predictions(data)
    physical_predictions = {
        name: _physical(value, data) for name, value in predictions.items()
    }
    sequences: dict[tuple[str, str], list[np.ndarray]] = {}

    def add(family: str, name: str, value: np.ndarray) -> None:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (data.rows,):
            raise AssertionError(f"{name} has shape {array.shape}")
        sequences.setdefault((family, name), []).append(array)

    layer_hellinger_history: list[np.ndarray] = []
    base_for_normalization = {
        "velocity_change_relative",
        "physical_acceleration_rms",
        "ca_temporal_endpoint_rms",
        "endpoint_ensemble_rms",
        "route_layer_hellinger_mean",
        "route_layer_hellinger_max",
    }

    for stage_index, stage_value in enumerate(STAGES):
        stage = int(stage_value)
        velocity = physical_x[:, stage] - physical_x[:, stage - 1]
        previous_velocity = physical_x[:, stage - 1] - physical_x[:, stage - 2]
        acceleration = velocity - previous_velocity
        speed = _rms(velocity, (1, 2))
        previous_speed = _rms(previous_velocity, (1, 2))
        acceleration_rms = _rms(acceleration, (1, 2))
        velocity_flat = velocity.reshape(data.rows, -1)
        previous_flat = previous_velocity.reshape(data.rows, -1)
        velocity_change_relative = np.linalg.norm(
            velocity_flat - previous_flat, axis=1
        ) / np.maximum(
            np.linalg.norm(velocity_flat, axis=1)
            + np.linalg.norm(previous_flat, axis=1),
            EPS,
        )
        turning = 1.0 - np.sum(velocity_flat * previous_flat, axis=1) / np.maximum(
            np.linalg.norm(velocity_flat, axis=1)
            * np.linalg.norm(previous_flat, axis=1),
            EPS,
        )
        add("flow_consistency", "velocity_change_relative", velocity_change_relative)
        add(
            "flow_consistency",
            "velocity_speed_log_ratio_abs",
            np.abs(np.log(np.maximum(speed, EPS) / np.maximum(previous_speed, EPS))),
        )
        add("flow_consistency", "physical_velocity_turning", turning)
        add("physical_scale", "physical_update_rms", speed)
        add("physical_scale", "physical_acceleration_rms", acceleration_rms)
        add(
            "physical_scale",
            "physical_translation_acceleration_rms",
            _rms(acceleration[..., :3], (1, 2)),
        )
        add(
            "physical_scale",
            "physical_rotation_acceleration_rms",
            _rms(acceleration[..., 3:6], (1, 2)),
        )
        add(
            "physical_scale",
            "physical_gripper_acceleration_rms",
            _rms(acceleration[..., 6], 1),
        )
        add(
            "flow_consistency",
            "temporal_velocity_dispersion",
            _rms(velocity - velocity.mean(axis=1, keepdims=True), (1, 2))
            / np.maximum(speed, EPS),
        )
        add(
            "flow_consistency",
            "acceleration_max_over_rms",
            np.abs(acceleration).max(axis=(1, 2)) / np.maximum(acceleration_rms, EPS),
        )

        if stage >= 3:
            older_velocity = physical_x[:, stage - 2] - physical_x[:, stage - 3]
            jerk = acceleration - (previous_velocity - older_velocity)
            three_speeds = np.stack(
                [speed, previous_speed, _rms(older_velocity, (1, 2))], axis=1
            )
            add("flow_consistency", "physical_jerk_rms", _rms(jerk, (1, 2)))
            add(
                "flow_consistency",
                "three_velocity_speed_cv",
                three_speeds.std(axis=1)
                / np.maximum(three_speeds.mean(axis=1), EPS),
            )
            add(
                "flow_consistency",
                "jerk_over_acceleration",
                _rms(jerk, (1, 2)) / np.maximum(acceleration_rms, EPS),
            )
        else:
            for name in (
                "physical_jerk_rms",
                "three_velocity_speed_cv",
                "jerk_over_acceleration",
            ):
                add("flow_consistency", name, np.full(data.rows, np.inf))

        endpoint_names = (
            "constant_velocity",
            "mean_last2_velocity",
            "constant_acceleration",
            "linear_fit_3",
        )
        endpoint_stack = np.stack(
            [physical_predictions[name][:, stage_index] for name in endpoint_names],
            axis=1,
        )
        endpoint_center = endpoint_stack.mean(axis=1, keepdims=True)
        endpoint_ensemble = _rms(endpoint_stack - endpoint_center, (1, 2, 3))
        add("endpoint_consistency", "endpoint_ensemble_rms", endpoint_ensemble)
        add(
            "endpoint_consistency",
            "cv_vs_mean2_endpoint_rms",
            _rms(endpoint_stack[:, 0] - endpoint_stack[:, 1], (1, 2)),
        )
        add(
            "endpoint_consistency",
            "cv_vs_ca_endpoint_rms",
            _rms(endpoint_stack[:, 0] - endpoint_stack[:, 2], (1, 2)),
        )
        for label, dimensions in (
            ("translation", slice(0, 3)),
            ("rotation", slice(3, 6)),
            ("gripper", slice(6, 7)),
        ):
            add(
                "physical_scale",
                f"{label}_endpoint_ensemble_rms",
                _rms(
                    endpoint_stack[..., dimensions]
                    - endpoint_center[..., dimensions],
                    (1, 2, 3),
                ),
            )
        endpoint_sign = np.signbit(endpoint_stack[..., 6])
        add(
            "physical_scale",
            "gripper_endpoint_sign_disagreement",
            np.mean(np.any(endpoint_sign != endpoint_sign[:, :1], axis=1), axis=1),
        )

        for estimator, short in (
            ("constant_velocity", "cv"),
            ("constant_acceleration", "ca"),
        ):
            current_endpoint = physical_predictions[estimator][:, stage_index]
            if stage_index == 0:
                if estimator == "constant_velocity":
                    previous = physical_x[:, 1] + (TOTAL_ROUNDS - 1) * (
                        physical_x[:, 1] - physical_x[:, 0]
                    )
                    difference = current_endpoint - previous
                else:
                    difference = np.full_like(current_endpoint, np.inf)
            else:
                difference = current_endpoint - physical_predictions[estimator][
                    :, stage_index - 1
                ]
            add(
                "endpoint_consistency",
                f"{short}_temporal_endpoint_rms",
                _rms(difference, (1, 2)),
            )
            add(
                "physical_scale",
                f"{short}_temporal_translation_rms",
                _rms(difference[..., :3], (1, 2)),
            )
            add(
                "physical_scale",
                f"{short}_temporal_rotation_rms",
                _rms(difference[..., 3:6], (1, 2)),
            )
            add(
                "physical_scale",
                f"{short}_temporal_gripper_rms",
                _rms(difference[..., 6], 1),
            )

        current_index = stage - 1
        previous_index = current_index - 1
        current_probability = data.probability[:, :, current_index]
        previous_probability = data.probability[:, :, previous_index]
        hellinger = np.sqrt(
            0.5
            * np.sum(
                np.square(
                    np.sqrt(current_probability) - np.sqrt(previous_probability)
                ),
                axis=-1,
            )
        )
        layer_hellinger = hellinger.mean(axis=2)
        layer_hellinger_history.append(layer_hellinger)
        layer_mean = layer_hellinger.mean(axis=1)
        layer_std = layer_hellinger.std(axis=1)
        add("layer_route", "route_layer_hellinger_mean", layer_mean)
        add("layer_route", "route_layer_hellinger_max", layer_hellinger.max(axis=1))
        add(
            "layer_route",
            "route_layer_hellinger_cv",
            layer_std / np.maximum(layer_mean, EPS),
        )
        add(
            "layer_route",
            "route_layer_hellinger_range",
            np.ptp(layer_hellinger, axis=1),
        )
        current_ids = data.expert_ids[:, :, current_index]
        active_counts = np.empty((data.rows, 8), dtype=np.float64)
        for row in range(data.rows):
            for layer in range(8):
                active_counts[row, layer] = len(np.unique(current_ids[row, layer]))
        add("layer_route", "active_count_layer_std", active_counts.std(axis=1))
        add("layer_route", "active_count_layer_range", np.ptp(active_counts, axis=1))
        selected = data.selected_probability[:, :, current_index]
        selected_l2 = _rms(selected, (2, 3))
        add(
            "layer_route",
            "router_weight_amplitude_layer_cv",
            selected_l2.std(axis=1) / np.maximum(selected_l2.mean(axis=1), EPS),
        )
        previous_selected = data.selected_probability[:, :, previous_index]
        add(
            "layer_route",
            "router_weight_amplitude_change",
            _rms(selected - previous_selected, (1, 2, 3)),
        )

        add(
            "candidate_consistency",
            "candidate_cv_endpoint_spread",
            _candidate_spread(endpoint_stack[:, 0], data.query_id),
        )
        add(
            "candidate_consistency",
            "candidate_ca_endpoint_spread",
            _candidate_spread(endpoint_stack[:, 2], data.query_id),
        )
        add(
            "candidate_consistency",
            "candidate_route_change_spread",
            _candidate_spread(layer_hellinger, data.query_id),
        )

    # Query-internal ratios remove task-dependent absolute scales.  Their early
    # reference uses only stages 2 and 3 of the same inference.
    for (family, name), sequence in list(sequences.items()):
        if name not in base_for_normalization:
            continue
        array = np.stack(sequence, axis=1)
        early = np.nanmedian(np.where(np.isfinite(array[:, :2]), array[:, :2], np.nan), axis=1)
        ratio = array / np.maximum(early[:, None], 1e-8)
        sequences[(family, name + "__relative_to_early")] = [
            ratio[:, index] for index in range(ratio.shape[1])
        ]

    # A route trajectory can look stable in aggregate while individual layers
    # oscillate.  This score measures deviation from each layer's own early scale.
    route_history = np.stack(layer_hellinger_history, axis=1)
    early_layer = np.median(route_history[:, :2], axis=1)
    normalized_layer = route_history / np.maximum(early_layer[:, None], 1e-8)
    sequences[("layer_route", "route_layer_self_normalized_max")] = [
        normalized_layer[:, index].max(axis=1)
        for index in range(normalized_layer.shape[1])
    ]
    sequences[("layer_route", "route_layer_self_normalized_mean")] = [
        normalized_layer[:, index].mean(axis=1)
        for index in range(normalized_layer.shape[1])
    ]

    result = []
    for (family, name), sequence in sequences.items():
        values = np.stack(sequence, axis=1)
        if values.shape != (data.rows, len(STAGES)):
            raise AssertionError(f"{name} has shape {values.shape}")
        result.append(Signal(name=name, family=family, values=values))
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 8 or np.ptp(left[finite]) <= 0 or np.ptp(right[finite]) <= 0:
        return float("nan")
    return float(spearmanr(left[finite], right[finite]).statistic)


def _auc(score: np.ndarray, positive: np.ndarray) -> float:
    finite = np.isfinite(score)
    score = score[finite]
    positive = positive[finite]
    count_positive = int(positive.sum())
    count_negative = len(positive) - count_positive
    if not count_positive or not count_negative:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float(
        (ranks[positive].sum() - count_positive * (count_positive + 1) / 2)
        / (count_positive * count_negative)
    )


def oracle_gate_rows(
    data: TraceData,
    signals: list[Signal],
    errors: dict[str, np.ndarray],
    physical_reference_rms: float,
    split: str,
) -> list[dict[str, Any]]:
    rows = []
    for signal in signals:
        for estimator in ESTIMATORS:
            correlations = []
            aucs = []
            stage_metrics: dict[str, Any] = {}
            for stage in ORACLE_STAGES:
                index = int(stage - STAGES[0])
                rho = _spearman(signal.values[:, index], errors[estimator][:, index])
                auc = _auc(
                    signal.values[:, index],
                    errors[estimator][:, index] > physical_reference_rms,
                )
                correlations.append(rho)
                aucs.append(auc)
                stage_metrics[f"rho_s{stage}"] = rho
                stage_metrics[f"auc_s{stage}"] = auc
            finite_rho = np.asarray(correlations)[np.isfinite(correlations)]
            finite_auc = np.asarray(aucs)[np.isfinite(aucs)]
            mean_rho = float(np.mean(finite_rho)) if len(finite_rho) else float("nan")
            sign_consistency = (
                float(np.mean(finite_rho > 0.0)) if len(finite_rho) else 0.0
            )
            rows.append(
                {
                    "split": split,
                    "family": signal.family,
                    "signal": signal.name,
                    "endpoint_estimator": estimator,
                    "mean_spearman": mean_rho,
                    "positive_stage_fraction": sign_consistency,
                    "mean_reference_violation_auc": (
                        float(np.mean(finite_auc)) if len(finite_auc) else float("nan")
                    ),
                    "passes_gate": bool(
                        split == "development"
                        and mean_rho >= ORACLE_RHO_GATE
                        and sign_consistency >= 0.6
                    ),
                    **stage_metrics,
                }
            )
    return rows


def _threshold_for_target_saving(
    values: np.ndarray, min_stage: int, target_saving: float
) -> float:
    thresholds = _candidate_thresholds(values, "le")
    best: tuple[tuple[float, float], float] | None = None
    for threshold in thresholds:
        stops = _stops_for_threshold(values, float(threshold), "le", min_stage)
        saving = float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS))
        # Prefer a slight compute undershoot on an exact-distance tie.
        objective = (abs(saving - target_saving), max(0.0, saving - target_saving))
        if best is None or objective < best[0]:
            best = (objective, float(threshold))
    if best is None:
        raise AssertionError("at least the never-stop threshold must exist")
    return best[1]


def _aggregate(
    data: TraceData,
    signal: Signal,
    estimator: str,
    min_stage: int,
    protocol: str,
    stops: np.ndarray,
    selected_error: np.ndarray,
    thresholds: Iterable[float],
    physical_reference_rms: float,
) -> dict[str, Any]:
    outside_reference = selected_error > physical_reference_rms
    return {
        "protocol": protocol,
        "family": signal.family,
        "signal": signal.name,
        "endpoint_estimator": estimator,
        "direction": "le",
        "min_stage": min_stage,
        "saving": float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)),
        "outside_reference_rate": float(np.mean(outside_reference)),
        "outside_reference_count": int(outside_reference.sum()),
        "outside_reference_query_count": int(
            len(np.unique(data.query_id[outside_reference]))
        ),
        "outside_reference_rollout_count": int(
            len(np.unique(data.rollout_id[outside_reference]))
        ),
        "mean_controller_score": float(selected_error.mean()),
        "p95_controller_score": float(np.percentile(selected_error, 95)),
        "mean_completed_rounds": float(stops.mean()),
        "stop_histogram": {
            str(stage): int(np.sum(stops == stage))
            for stage in range(int(STAGES[0]), TOTAL_ROUNDS + 1)
        },
        "thresholds": [float(value) for value in thresholds],
    }


def loeo_rows(
    data: TraceData,
    signals: list[Signal],
    errors: dict[str, np.ndarray],
    gate: set[tuple[str, str]],
    physical_reference_rms: float,
) -> list[dict[str, Any]]:
    rollouts = np.unique(data.rollout_id)
    results = []
    for signal_index, signal in enumerate(signals):
        for estimator in ESTIMATORS:
            if (signal.name, estimator) not in gate:
                continue
            for min_stage in MIN_STAGES:
                protocols = ["safety_calibrated"] + [
                    f"compute_matched_{round(100 * target):02d}"
                    for target in MATCHED_SAVINGS
                ]
                all_stops = {
                    protocol: np.full(data.rows, TOTAL_ROUNDS, dtype=np.int64)
                    for protocol in protocols
                }
                thresholds = {protocol: [] for protocol in protocols}
                for held_out in rollouts:
                    train = data.rollout_id != held_out
                    test = ~train
                    threshold, _ = calibrate_threshold(
                        signal.values[train],
                        errors[estimator][train],
                        physical_reference_rms,
                        TRAIN_RISK,
                        "le",
                        min_stage,
                    )
                    thresholds["safety_calibrated"].append(threshold)
                    all_stops["safety_calibrated"][test] = _stops_for_threshold(
                        signal.values[test], threshold, "le", min_stage
                    )
                    for target in MATCHED_SAVINGS:
                        protocol = f"compute_matched_{round(100 * target):02d}"
                        threshold = _threshold_for_target_saving(
                            signal.values[train], min_stage, target
                        )
                        thresholds[protocol].append(threshold)
                        all_stops[protocol][test] = _stops_for_threshold(
                            signal.values[test], threshold, "le", min_stage
                        )
                for protocol in protocols:
                    selected = _selected_error(errors[estimator], all_stops[protocol])
                    results.append(
                        _aggregate(
                            data,
                            signal,
                            estimator,
                            min_stage,
                            protocol,
                            all_stops[protocol],
                            selected,
                            thresholds[protocol],
                            physical_reference_rms,
                        )
                    )
        if signal_index % 10 == 0:
            print(f"LOEO signal {signal_index + 1}/{len(signals)}: {signal.name}", flush=True)
    return results


def _rank_loeo(rows: list[dict[str, Any]], protocol: str) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["protocol"] == protocol]
    return sorted(
        selected,
        key=lambda row: (
            row["outside_reference_rate"] > 0.01,
            -row["saving"],
            row["outside_reference_rate"],
            row["mean_controller_score"],
            row["signal"],
        ),
    )


def _selected_prediction(prediction: np.ndarray, stops: np.ndarray) -> np.ndarray:
    result = np.empty((len(stops), 10, LIVE_DIMS), dtype=np.float64)
    final_rows = stops == TOTAL_ROUNDS
    result[final_rows] = np.nan
    early_rows = np.flatnonzero(~final_rows)
    if len(early_rows):
        indices = stops[early_rows] - int(STAGES[0])
        result[early_rows] = prediction[early_rows, indices]
    return result


def apply_frozen(
    train: TraceData,
    test: TraceData,
    train_signals: list[Signal],
    test_signals: list[Signal],
    train_errors: dict[str, np.ndarray],
    test_errors: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    selected_rows: list[dict[str, Any]],
    physical_reference_rms: float,
) -> list[dict[str, Any]]:
    train_by_key = {(signal.family, signal.name): signal for signal in train_signals}
    test_by_key = {(signal.family, signal.name): signal for signal in test_signals}
    results = []
    for selected in selected_rows:
        key = (selected["family"], selected["signal"])
        train_signal = train_by_key[key]
        test_signal = test_by_key[key]
        estimator = selected["endpoint_estimator"]
        min_stage = int(selected["min_stage"])
        threshold, train_stats = calibrate_threshold(
            train_signal.values,
            train_errors[estimator],
            physical_reference_rms,
            TRAIN_RISK,
            "le",
            min_stage,
        )
        stops = _stops_for_threshold(test_signal.values, threshold, "le", min_stage)
        selected_error = _selected_error(test_errors[estimator], stops)
        outside_reference = selected_error > physical_reference_rms
        prediction = _selected_prediction(test_predictions[estimator], stops)
        early = stops < TOTAL_ROUNDS
        physical_rms = 0.0
        physical_max_p95 = 0.0
        gripper_flip = 0.0
        if np.any(early):
            difference = (
                prediction[early] - test.x[early, TOTAL_ROUNDS]
            ) * test.action_std[None, None]
            physical_rms = float(_rms(difference[..., :6], (1, 2)).mean())
            physical_max_p95 = float(
                np.percentile(np.abs(difference[..., :6]).max(axis=(1, 2)), 95)
            )
            gripper_flip = float(
                np.mean(
                    np.any(
                        np.sign(prediction[early, :, 6])
                        != np.sign(test.x[early, TOTAL_ROUNDS, :, 6]),
                        axis=1,
                    )
                )
            )
        results.append(
            {
                "selection_scope": selected.get("selection_scope", "development"),
                "family": selected["family"],
                "signal": selected["signal"],
                "endpoint_estimator": estimator,
                "min_stage": min_stage,
                "threshold": float(threshold),
                "development_loeo_saving": float(selected["saving"]),
                "development_loeo_outside_reference_rate": float(
                    selected["outside_reference_rate"]
                ),
                "train_full_saving": float(train_stats["saving"]),
                "holdout_saving": float(
                    np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)
                ),
                "holdout_outside_reference_rate": float(
                    np.mean(outside_reference)
                ),
                "holdout_outside_reference_count": int(outside_reference.sum()),
                "holdout_mean_controller_score": float(selected_error.mean()),
                "holdout_p95_controller_score": float(
                    np.percentile(selected_error, 95)
                ),
                "holdout_early_fraction": float(np.mean(early)),
                "holdout_physical_rms_early": physical_rms,
                "holdout_physical_max_abs_p95_early": physical_max_p95,
                "holdout_gripper_flip_rate_early": gripper_flip,
                "stop_histogram": {
                    str(stage): int(np.sum(stops == stage))
                    for stage in range(int(STAGES[0]), TOTAL_ROUNDS + 1)
                },
            }
        )
    return results


def fixed_baselines(
    data: TraceData,
    predictions: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    physical_reference_rms: float,
) -> list[dict[str, Any]]:
    rows = []
    for estimator, stage in (("constant_velocity", 9), ("constant_acceleration", 8)):
        stops = np.full(data.rows, stage, dtype=np.int64)
        selected = _selected_error(errors[estimator], stops)
        outside_reference = selected > physical_reference_rms
        difference = (
            predictions[estimator][:, stage - int(STAGES[0])]
            - data.x[:, TOTAL_ROUNDS]
        ) * data.action_std[None, None]
        rows.append(
            {
                "estimator": estimator,
                "stage": stage,
                "saving": (TOTAL_ROUNDS - stage) / TOTAL_ROUNDS,
                "outside_reference_rate": float(np.mean(outside_reference)),
                "outside_reference_count": int(outside_reference.sum()),
                "mean_controller_score": float(selected.mean()),
                "physical_6d_rms": float(_rms(difference[..., :6], (1, 2)).mean()),
                "physical_max_abs_p95": float(
                    np.percentile(np.abs(difference[..., :6]).max(axis=(1, 2)), 95)
                ),
                "translation_rms_p95": float(
                    np.percentile(_rms(difference[..., :3], (1, 2)), 95)
                ),
                "rotation_rms_p95": float(
                    np.percentile(_rms(difference[..., 3:6], (1, 2)), 95)
                ),
                "gripper_sign_flip_rate": float(
                    np.mean(
                        np.any(
                            np.sign(
                                predictions[estimator][
                                    :, stage - int(STAGES[0]), :, 6
                                ]
                            )
                            != np.sign(data.x[:, TOTAL_ROUNDS, :, 6]),
                            axis=1,
                        )
                    )
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = [
        key
        for key in rows[0]
        if all(not isinstance(row.get(key), (dict, list)) for row in rows)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(out: Path, summary: dict[str, Any]) -> None:
    development_gate = summary["oracle_gate"]["development"]
    holdout_gate = {
        (row["family"], row["signal"], row["endpoint_estimator"]): row
        for row in summary["oracle_gate"]["holdout_diagnostic"]
    }
    passing = [row for row in development_gate if row["passes_gate"]]
    top_gate = sorted(passing, key=lambda row: -row["mean_spearman"])[:12]
    loeo_top = _rank_loeo(summary["loeo"], "safety_calibrated")[:12]
    matched20 = sorted(
        [row for row in summary["loeo"] if row["protocol"] == "compute_matched_20"],
        key=lambda row: (
            row["outside_reference_rate"],
            row["mean_controller_score"],
        ),
    )[:8]

    lines = [
        "# 无监督逐轮早停信号补充验证",
        "",
        "## 实验口径",
        "",
        f"- Goal：`{summary['development']['rows']}` 条轨迹、`{summary['development']['queries']}` 个 query、`{summary['development']['rollouts']}` 条 rollout；Long：`{summary['holdout']['rows']}` 条轨迹、`{summary['holdout']['rollouts']}` 条 rollout。",
        "- 信号只读取当前 query 已完成轮次的 flow trajectory 和路由，不读取 reward、success 或未来轮次。每个 query 独立重置。",
        "- 第 10 轮只在离线阶段充当自监督 endpoint：先作 oracle 可分性门，再在 Goal 的训练 rollout 上调阈值。Long 的信号、误差和结果均未参与选择或调阈值。",
        f"- controller-aware endpoint：前 6 维先乘 action std 回到物理动作尺度再算 RMS；第 7 维夹爪只比较符号。参考带 `{summary['definition']['physical_reference_rms']:.6f}` 是 Goal 固定 stage-8 常加速度的 P95，只表示‘不劣于这个开发集基准’，不是权威控制安全阈值。",
        "- 所有新信号预定义为‘数值越小越稳定’，不扫描反方向；阈值按完整 rollout 留一。",
        "",
        "## Oracle gate：信号里到底有没有剩余误差信息",
        "",
        f"共构造 `{summary['counts']['signals']}` 个前缀信号；`{len(passing)}` 个 signal×endpoint 组合达到预设门槛：stage 4–8 平均 Spearman `>= {ORACLE_RHO_GATE:.2f}` 且至少 3/5 个 stage 同方向。这里是信息存在性检查，不是可部署成绩。",
        "",
        "| 家族 | 信号 | 输出器 | Goal 平均 rho | Long 同方向 rho | Goal 超带 AUC |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in top_gate:
        hold = holdout_gate[(row["family"], row["signal"], row["endpoint_estimator"])]
        lines.append(
            f"| {row['family']} | `{row['signal']}` | {row['endpoint_estimator']} | {row['mean_spearman']:.3f} | {hold['mean_spearman']:.3f} | {row['mean_reference_violation_auc']:.3f} |"
        )

    lines += [
        "",
        "## Goal rollout 留一",
        "",
        "阈值只看其余 3 条 rollout 的 x10 endpoint 误差，并要求训练折零超带；下表测试整条未见 rollout。由于同时探索多个信号，这一列只用于选出一个冻结候选，不能当确认性结果。",
        "",
        "| 家族 | 信号 | 输出器 | 最早轮 | 节省 | 超参考带 | score P95 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in loeo_top:
        lines.append(
            f"| {row['family']} | `{row['signal']}` | {row['endpoint_estimator']} | {row['min_stage']} | {_pct(row['saving'])} | {_pct(row['outside_reference_rate'])} ({row['outside_reference_count']}) | {row['p95_controller_score']:.5f} |"
        )

    lines += [
        "",
        "### 匹配 20% 计算量的诊断",
        "",
        "这一阈值只用训练信号分布对齐节省率，不看训练或测试误差；用于判断动态规则是否真的优于固定第 8 轮。",
        "",
        "| 信号 | 输出器 | 实际节省 | 超参考带 |",
        "|---|---|---:|---:|",
    ]
    for row in matched20:
        lines.append(
            f"| `{row['signal']}` | {row['endpoint_estimator']} | {_pct(row['saving'])} | {_pct(row['outside_reference_rate'])} ({row['outside_reference_count']}) |"
        )

    lines += [
        "",
        "## Goal 选择后冻结到 Long",
        "",
        "| 选择范围 | 家族/信号 | 输出器 | Goal LOEO | Long 节省 | Long 超参考带 | 6D 物理 RMS（提前样本） | 6D max P95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["frozen_holdout"]:
        lines.append(
            f"| {row['selection_scope']} | {row['family']} / `{row['signal']}` | {row['endpoint_estimator']} | {_pct(row['development_loeo_saving'])}, {_pct(row['development_loeo_outside_reference_rate'])} 超带 | {_pct(row['holdout_saving'])} | {_pct(row['holdout_outside_reference_rate'])} ({row['holdout_outside_reference_count']}) | {row['holdout_physical_rms_early']:.5f} | {row['holdout_physical_max_abs_p95_early']:.5f} |"
        )
    for row in summary["holdout"]["fixed_baselines"]:
        lines.append(
            f"| 固定对照 | stage {row['stage']} | {row['estimator']} | - | {_pct(row['saving'])} | {_pct(row['outside_reference_rate'])} ({row['outside_reference_count']}) | {row['physical_6d_rms']:.5f} | {row['physical_max_abs_p95']:.5f} |"
        )

    lines += [
        "",
        "### Controller-aware 固定轮数拆分",
        "",
        "| 数据 | 固定输出 | 6D RMS mean | translation RMS P95 | rotation RMS P95 | gripper sign flip | 超参考带 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for split, baseline_rows in (
        ("Goal", summary["development"]["fixed_baselines"]),
        ("Long", summary["holdout"]["fixed_baselines"]),
    ):
        for row in baseline_rows:
            lines.append(
                f"| {split} | stage {row['stage']} + {row['estimator']} | {row['physical_6d_rms']:.5f} | {row['translation_rms_p95']:.5f} | {row['rotation_rms_p95']:.5f} | {_pct(row['gripper_sign_flip_rate'])} | {_pct(row['outside_reference_rate'])} ({row['outside_reference_count']}) |"
            )

    best_frozen = summary["frozen_holdout"][0]
    goal_stage8 = next(
        row
        for row in summary["development"]["fixed_baselines"]
        if row["stage"] == 8
    )
    best_matched20 = matched20[0]
    lines += [
        "",
        "## 裁决",
        "",
        f"- Goal 选出的全局候选是 `{best_frozen['signal']}` + `{best_frozen['endpoint_estimator']}`；冻结到 Long 后节省 `{_pct(best_frozen['holdout_saving'])}`，超参考带 `{_pct(best_frozen['holdout_outside_reference_rate'])}`。这只是 endpoint 等价性筛选，不等价于在线任务成功率。",
        "- 它在 Long 实际只是在 stage 9 停 111/181、其余跑满，节省 6.1%；固定 stage 9 对全部 181 条都停，节省 10.0%，同样 0 条超带。因此动态规则被固定 clock 严格支配，不进入在线 A/B。",
        f"- 匹配约 20% 节省时，新信号最多只少 `{goal_stage8['outside_reference_count'] - best_matched20['outside_reference_count']}` 个 Goal 超带样本（{best_matched20['outside_reference_count']} 对固定 stage 8 的 {goal_stage8['outside_reference_count']}），同时节省略低（{_pct(best_matched20['saving'])} 对 {_pct(goal_stage8['saving'])}）；这不是有说服力的动态选择增益。",
        "- 原 7D 相对 L2 的 Long stage-8 `28.2%` 不能继续当主结论：controller-aware 指标下同一输出只有 7/181（3.9%）超过 Goal 经验参考带，且夹爪符号翻转为 0。",
        "- 固定对照同时报告 translation、rotation 与 gripper sign。由于仓库里没有可作为权威依据的平移/旋转安全阈值，本报告不把经验参考带包装成控制安全保证。",
        "- endpoint 外推的一致性、flow 离散曲率、translation/rotation/gripper 物理分解、层间路由一致性都已纳入；候选云信号只在 Goal 的 K=16 上诊断，Long 是 K=1，因此不把它伪装成可迁移方法。",
        "- `router_weight_amplitude_*` 只是 top-k 路由权重幅度，不是 `w_e E_e(h)` 的真实专家输出幅度；现有 trace 没保存 expert output，不能用本报告回答真实激活贡献是否能早停。",
        "- Long oracle rho 仅用于事后解释迁移失败或成功，绝未参与候选、方向、轮数和阈值选择。",
        "",
        "## 复现",
        "",
        "```bash",
        "cd /home/jovyan/work/himoe-vla/himoe-route-capture",
        "python analyze_unsupervised_stop_signals.py \\",
        "  --run runs/flow-lead-cpu-t0s24 \\",
        "  --holdout-run runs/flow-stop-sweep-long-t08s0-k1-seeds6100-6103 \\",
        "  --out-dir analysis/unsupervised-stop-signals-v2",
        "```",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    development = load_trace(Path(args.run), name="Goal development")
    holdout = load_trace(Path(args.holdout_run), name="Long holdout")
    development_predictions = endpoint_predictions(development)
    holdout_predictions = endpoint_predictions(holdout)
    stage8_index = int(8 - STAGES[0])
    stage8_difference = (
        development_predictions["constant_acceleration"][:, stage8_index, :, :6]
        - development.x[:, TOTAL_ROUNDS, :, :6]
    ) * development.action_std[None, None, :6]
    empirical_reference = float(
        np.percentile(_rms(stage8_difference, (1, 2)), 95)
    )
    physical_reference_rms = (
        empirical_reference
        if args.physical_reference_rms is None
        else float(args.physical_reference_rms)
    )
    if physical_reference_rms <= 0.0:
        raise ValueError("--physical-reference-rms must be positive")
    development_errors = {
        estimator: controller_endpoint_error(
            development,
            development_predictions[estimator],
            physical_reference_rms,
        )[0]
        for estimator in ESTIMATORS
    }
    holdout_errors = {
        estimator: controller_endpoint_error(
            holdout,
            holdout_predictions[estimator],
            physical_reference_rms,
        )[0]
        for estimator in ESTIMATORS
    }
    development_signals = build_prefix_signals(development)
    holdout_signals = build_prefix_signals(holdout)
    if [(item.family, item.name) for item in development_signals] != [
        (item.family, item.name) for item in holdout_signals
    ]:
        raise AssertionError("development and holdout signal schemas differ")
    print(
        f"loaded Goal={development.rows}, Long={holdout.rows}; "
        f"built {len(development_signals)} signals",
        flush=True,
    )

    development_gate = oracle_gate_rows(
        development,
        development_signals,
        development_errors,
        physical_reference_rms,
        "development",
    )
    holdout_gate = oracle_gate_rows(
        holdout,
        holdout_signals,
        holdout_errors,
        physical_reference_rms,
        "holdout_diagnostic",
    )
    gate = {
        (row["signal"], row["endpoint_estimator"])
        for row in development_gate
        if row["passes_gate"]
    }
    print(f"oracle gate retained {len(gate)} signal-estimator pairs", flush=True)
    loeo = loeo_rows(
        development,
        development_signals,
        development_errors,
        gate,
        physical_reference_rms,
    )
    ranked = _rank_loeo(loeo, "safety_calibrated")
    deployable = [
        row for row in ranked if row["family"] != "candidate_consistency"
    ]
    if not deployable:
        raise RuntimeError("oracle gate left no deployable candidate")

    selected = [{**deployable[0], "selection_scope": "global Goal best"}]
    for family in sorted({row["family"] for row in deployable}):
        family_best = next(row for row in deployable if row["family"] == family)
        if family_best["signal"] != selected[0]["signal"]:
            selected.append(
                {**family_best, "selection_scope": f"Goal best in {family}"}
            )
    frozen = apply_frozen(
        development,
        holdout,
        development_signals,
        holdout_signals,
        development_errors,
        holdout_errors,
        holdout_predictions,
        selected,
        physical_reference_rms,
    )
    fixed = fixed_baselines(
        holdout,
        holdout_predictions,
        holdout_errors,
        physical_reference_rms,
    )
    development_fixed = fixed_baselines(
        development,
        development_predictions,
        development_errors,
        physical_reference_rms,
    )

    summary = {
        "schema": "himoe-unsupervised-stop-signals-v2",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "definition": {
            "stages": STAGES.tolist(),
            "oracle_stages": ORACLE_STAGES.tolist(),
            "endpoint_estimators": list(ESTIMATORS),
            "endpoint_error": (
                "physical RMS on continuous dims 0:6; any gripper sign flip is "
                "encoded outside the reference band"
            ),
            "physical_reference_rms": physical_reference_rms,
            "physical_reference_definition": (
                "Goal fixed stage-8 constant-acceleration 6D physical RMS P95; "
                "empirical equivalence band, not authoritative safety threshold"
            ),
            "oracle_rho_gate": ORACLE_RHO_GATE,
            "oracle_positive_stage_fraction_gate": 0.6,
            "direction": "all scores are predefined instability; stop when <= threshold",
            "cv": "leave one complete rollout out",
            "holdout_policy": "all choices and thresholds frozen on Goal before scoring Long",
        },
        "development": {
            "run": str(Path(args.run).resolve()),
            "rows": development.rows,
            "queries": int(len(np.unique(development.query_id))),
            "rollouts": int(len(np.unique(development.rollout_id))),
            "fixed_baselines": development_fixed,
        },
        "holdout": {
            "run": str(Path(args.holdout_run).resolve()),
            "rows": holdout.rows,
            "queries": int(len(np.unique(holdout.query_id))),
            "rollouts": int(len(np.unique(holdout.rollout_id))),
            "fixed_baselines": fixed,
        },
        "counts": {
            "signals": len(development_signals),
            "oracle_signal_estimator_pairs": len(development_gate),
            "oracle_pass_pairs": len(gate),
            "loeo_rows": len(loeo),
        },
        "oracle_gate": {
            "development": development_gate,
            "holdout_diagnostic": holdout_gate,
        },
        "loeo": loeo,
        "frozen_holdout": frozen,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    write_csv(out / "oracle_gate.csv", development_gate + holdout_gate)
    write_csv(out / "loeo.csv", loeo)
    write_csv(out / "frozen_holdout.csv", frozen)
    write_report(out, summary)
    print(f"wrote {out / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
