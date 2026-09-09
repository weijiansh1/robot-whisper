#!/usr/bin/env python3
"""Audit unsupervised per-query stopping rules from real MoE activations.

This script deliberately evaluates only trigger behavior.  The feature archive
contains ten rounds of expert activations but no per-round action trajectory, so
it cannot establish endpoint error or closed-loop safety.  It writes the stop
chosen for every query so a later action-trace collection can evaluate frozen
rules without another threshold search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOTAL_ROUNDS = 10
ACTION_TOKENS = 10
EPS = 1e-12
BASE_SIGNALS = {
    "expert_mass": "expert_mass_rms_mean",
    "routed_rms": "routed_rms_mean",
    "shared_rms": "shared_rms_mean",
    "routed_over_shared": "routed_over_shared_mean",
    "cancellation": "cancellation_mean",
}


@dataclass(frozen=True)
class AuditData:
    metrics: np.ndarray
    metric_names: tuple[str, ...]
    layers: np.ndarray
    scenes: np.ndarray
    noise_seeds: np.ndarray
    episode_ids: np.ndarray
    token_mean_vector_rms: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_source = here / "analysis/expert-activation-future/long-t08"
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(default_source / "features.npz"))
    parser.add_argument("--source-summary", default=str(default_source / "summary.json"))
    parser.add_argument(
        "--out-dir",
        default=str(here / "analysis/expert-activation-stop-audit/long-t08"),
    )
    return parser.parse_args()


def _source_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = json.loads(summary_path.read_text())
    run = Path(summary["run"])
    rows = json.loads((run / "client" / "summaries.json").read_text())
    return sorted(rows, key=lambda row: int(row["episode_index"]))


def load_data(features_path: Path, summary_path: Path) -> AuditData:
    with np.load(features_path, allow_pickle=False) as archive:
        metrics = np.asarray(archive["metrics"], dtype=np.float64)
        names = tuple(str(value) for value in archive["metric_names"].tolist())
        layers = np.asarray(archive["layer_numbers"], dtype=np.int64)
        routed_vectors = np.asarray(archive["routed_vectors"], dtype=np.float32)
        shared_vectors = np.asarray(archive["shared_vectors"], dtype=np.float32)

    if metrics.ndim != 4 or metrics.shape[1] != TOTAL_ROUNDS:
        raise ValueError("expected metrics [query, 10, layer, metric], got %s" % (metrics.shape,))
    if metrics.shape[2] != len(layers) or metrics.shape[3] != len(names):
        raise ValueError("metric axes do not match layer_numbers/metric_names")
    expected_vectors = metrics.shape[:3] + (1024,)
    if routed_vectors.shape != expected_vectors or shared_vectors.shape != expected_vectors:
        raise ValueError("mean-vector features do not align with scalar metrics")
    if not np.all(np.isfinite(metrics)):
        raise ValueError("metrics contain non-finite values")
    missing = sorted(set(BASE_SIGNALS.values()) - set(names))
    if missing:
        raise ValueError("missing activation metrics: %s" % missing)

    rows = _source_rows(summary_path)
    if len(rows) != len(metrics):
        raise ValueError("source rows and feature rows differ")
    scenes = np.asarray([int(row["init_state_id"]) for row in rows], dtype=np.int64)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in rows], dtype=np.int64)
    episodes = np.asarray([int(row["episode_index"]) for row in rows], dtype=np.int64)
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError("episode ids are not unique")
    unique_scenes = np.unique(scenes)
    unique_seeds = np.unique(seeds)
    if len(metrics) != len(unique_scenes) * len(unique_seeds):
        raise ValueError("source is not a complete scene x noise grid")
    expected_seeds = set(int(value) for value in unique_seeds)
    for scene in unique_scenes:
        if set(int(value) for value in seeds[scenes == scene]) != expected_seeds:
            raise ValueError("scene %d does not contain every noise seed" % int(scene))

    for metric_name in BASE_SIGNALS.values():
        values = metrics[..., names.index(metric_name)]
        if np.any(values <= 0.0):
            raise ValueError("%s must be positive for ratio features" % metric_name)
    token_mean_vector_rms = {
        "routed": np.sqrt(np.mean(np.square(routed_vectors), axis=-1)),
        "shared": np.sqrt(np.mean(np.square(shared_vectors), axis=-1)),
        "routed_plus_shared": np.sqrt(
            np.mean(np.square(routed_vectors + shared_vectors), axis=-1)
        ),
    }
    return AuditData(
        metrics,
        names,
        layers,
        scenes,
        seeds,
        episodes,
        token_mean_vector_rms,
    )


def first_stop(
    condition: np.ndarray,
    consecutive: int = 1,
    min_round: int = 3,
) -> np.ndarray:
    """Return first completed round satisfying a causal boolean condition.

    Round 10 is both the full-compute fallback and therefore never counted as a
    saving.  A consecutive window may include rounds before ``min_round`` but
    can only finish at or after it.
    """

    condition = np.asarray(condition, dtype=bool)
    if condition.ndim != 2 or condition.shape[1] != TOTAL_ROUNDS:
        raise ValueError("condition must have shape [query, 10]")
    if not 1 <= consecutive <= TOTAL_ROUNDS:
        raise ValueError("invalid consecutive count")
    if not 1 <= min_round <= TOTAL_ROUNDS:
        raise ValueError("invalid minimum round")
    stops = np.full(len(condition), TOTAL_ROUNDS, dtype=np.int64)
    start = max(min_round - 1, consecutive - 1)
    for round_index in range(start, TOTAL_ROUNDS - 1):
        eligible = np.all(
            condition[:, round_index - consecutive + 1 : round_index + 1], axis=1
        )
        stops[(stops == TOTAL_ROUNDS) & eligible] = round_index + 1
    return stops


def activation_views(values: np.ndarray, layers: np.ndarray) -> dict[str, np.ndarray]:
    """Build causal query-normalized layer aggregations [query, round]."""

    if values.ndim != 3 or values.shape[1] != TOTAL_ROUNDS:
        raise ValueError("activation values must be [query, 10, layer]")
    if values.shape[2] != len(layers) or np.any(values <= 0.0):
        raise ValueError("invalid activation values")
    per_layer = values / np.maximum(values[:, :1], EPS)
    aggregate = values.mean(axis=2)
    output = {
        "absolute_layer_mean": aggregate,
        "aggregate_then_norm": aggregate / np.maximum(aggregate[:, :1], EPS),
        "layernorm_mean": per_layer.mean(axis=2),
        "layernorm_median": np.median(per_layer, axis=2),
        "layernorm_max": per_layer.max(axis=2),
        "layernorm_min": per_layer.min(axis=2),
    }
    for layer_axis, layer in enumerate(layers):
        output["layer_%d_norm" % int(layer)] = per_layer[:, :, layer_axis]
    return output


def token_aggregation_proxies(
    mean: np.ndarray,
    sample_std: np.ndarray,
    token_count: int = ACTION_TOKENS,
) -> dict[str, np.ndarray]:
    """Recover token RMS and distribution-free tail proxies from mean/std."""

    mean = np.asarray(mean, dtype=np.float64)
    sample_std = np.asarray(sample_std, dtype=np.float64)
    if mean.shape != sample_std.shape or token_count < 2:
        raise ValueError("mean/std shapes or token count are invalid")
    if np.any(sample_std < 0.0) or not np.all(np.isfinite(mean + sample_std)):
        raise ValueError("mean/std must be finite and std must be nonnegative")
    return {
        "token_mean": mean,
        "token_rms_recovered": np.sqrt(
            np.square(mean)
            + ((token_count - 1) / token_count) * np.square(sample_std)
        ),
        "mean_plus_std_proxy": mean + sample_std,
        "max_upper_bound_mean_plus_3std": mean
        + math.sqrt(token_count - 1) * sample_std,
    }


def _group_savings(stops: np.ndarray, groups: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            100.0 * np.mean(TOTAL_ROUNDS - stops[groups == group]) / TOTAL_ROUNDS
            for group in np.unique(groups)
        ],
        dtype=np.float64,
    )


def rule_stats(
    rule_id: str,
    family: str,
    signal: str,
    aggregation: str,
    stops: np.ndarray,
    data: AuditData,
    **parameters: Any,
) -> dict[str, Any]:
    stops = np.asarray(stops, dtype=np.int64)
    if stops.shape != (len(data.metrics),) or np.any((stops < 1) | (stops > TOTAL_ROUNDS)):
        raise ValueError("invalid stop vector")
    values, counts = np.unique(stops, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    scene_savings = _group_savings(stops, data.scenes)
    seed_savings = _group_savings(stops, data.noise_seeds)
    result: dict[str, Any] = {
        "rule_id": rule_id,
        "family": family,
        "signal": signal,
        "aggregation": aggregation,
        "save_pct": float(100.0 * np.mean(TOTAL_ROUNDS - stops) / TOTAL_ROUNDS),
        "coverage_pct": float(100.0 * np.mean(stops < TOTAL_ROUNDS)),
        "mean_stop": float(stops.mean()),
        "median_stop": float(np.median(stops)),
        "q25_stop": float(np.percentile(stops, 25)),
        "q75_stop": float(np.percentile(stops, 75)),
        "modal_stop": int(values[np.argmax(counts)]),
        "modal_fraction": float(counts.max() / counts.sum()),
        "stop_entropy_bits": entropy,
        "scene_save_sd_pp": float(scene_savings.std()),
        "scene_save_min_pct": float(scene_savings.min()),
        "scene_save_max_pct": float(scene_savings.max()),
        "seed_save_sd_pp": float(seed_savings.std()),
        "seed_save_min_pct": float(seed_savings.min()),
        "seed_save_max_pct": float(seed_savings.max()),
        "stop_histogram": json.dumps(
            {str(int(value)): int(count) for value, count in zip(values, counts)},
            sort_keys=True,
        ),
    }
    result.update(parameters)
    return result


def _level_sweep(data: AuditData, views: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized_thresholds = np.round(np.arange(0.50, 2.0001, 0.025), 6)
    for signal, signal_views in views.items():
        for aggregation, values in signal_views.items():
            if aggregation == "absolute_layer_mean":
                thresholds = np.unique(
                    np.quantile(values[:, 2:9], np.linspace(0.05, 0.95, 19))
                )
                calibration = "pooled_q05_to_q95"
            else:
                thresholds = normalized_thresholds
                calibration = "fixed"
            for direction in ("le", "ge"):
                for consecutive in (1, 2, 3):
                    for threshold in thresholds:
                        condition = values <= threshold if direction == "le" else values >= threshold
                        stops = first_stop(condition, consecutive=consecutive, min_round=3)
                        rule_id = "%s__%s__%s_%.6g__k%d" % (
                            signal,
                            aggregation,
                            direction,
                            threshold,
                            consecutive,
                        )
                        rows.append(
                            rule_stats(
                                rule_id,
                                "level",
                                signal,
                                aggregation,
                                stops,
                                data,
                                direction=direction,
                                threshold=float(threshold),
                                threshold_2="",
                                consecutive=consecutive,
                                calibration=calibration,
                            )
                        )
    return rows


def _shape_sweep(data: AuditData, views: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal in ("expert_mass", "routed_rms", "shared_rms", "cancellation"):
        values = views[signal]["aggregate_then_norm"]
        relative_delta = np.full_like(values, np.inf)
        relative_delta[:, 1:] = (values[:, 1:] - values[:, :-1]) / np.maximum(
            np.abs(values[:, :-1]), EPS
        )
        curvature = np.full_like(values, -np.inf)
        curvature[:, 2:] = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
        for level in (0.80, 0.85, 0.875, 0.90, 0.925, 0.95, 1.00):
            for threshold in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
                condition = (np.abs(relative_delta) <= threshold) & (values <= level)
                for consecutive in (1, 2):
                    stops = first_stop(condition, consecutive=consecutive, min_round=3)
                    rule_id = "%s__flat_%.4g__level_%.4g__k%d" % (
                        signal,
                        threshold,
                        level,
                        consecutive,
                    )
                    rows.append(
                        rule_stats(
                            rule_id,
                            "flat_slope",
                            signal,
                            "aggregate_then_norm",
                            stops,
                            data,
                            direction="abs_relative_delta_le",
                            threshold=threshold,
                            threshold_2=level,
                            consecutive=consecutive,
                            calibration="fixed",
                        )
                    )
            for threshold in (-0.01, -0.005, 0.0, 0.005, 0.01, 0.02):
                condition = (relative_delta >= threshold) & (values <= level)
                stops = first_stop(condition, consecutive=1, min_round=3)
                rule_id = "%s__turn_%.4g__level_%.4g" % (signal, threshold, level)
                rows.append(
                    rule_stats(
                        rule_id,
                        "turn_or_rebound",
                        signal,
                        "aggregate_then_norm",
                        stops,
                        data,
                        direction="relative_delta_ge",
                        threshold=threshold,
                        threshold_2=level,
                        consecutive=1,
                        calibration="fixed",
                    )
                )
            for threshold in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03):
                condition = (curvature >= threshold) & (values <= level)
                stops = first_stop(condition, consecutive=1, min_round=3)
                rule_id = "%s__curvature_%.4g__level_%.4g" % (signal, threshold, level)
                rows.append(
                    rule_stats(
                        rule_id,
                        "positive_curvature",
                        signal,
                        "aggregate_then_norm",
                        stops,
                        data,
                        direction="curvature_ge",
                        threshold=threshold,
                        threshold_2=level,
                        consecutive=1,
                        calibration="fixed",
                    )
                )
    return rows


def _consensus_sweep(data: AuditData, raw: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal in ("expert_mass", "routed_rms", "shared_rms"):
        per_layer = raw[signal] / np.maximum(raw[signal][:, :1], EPS)
        for threshold in np.round(np.arange(0.70, 1.0001, 0.025), 6):
            count = np.sum(per_layer <= threshold, axis=2)
            for required in range(1, per_layer.shape[2] + 1):
                for consecutive in (1, 2, 3):
                    stops = first_stop(
                        count >= required, consecutive=consecutive, min_round=3
                    )
                    rule_id = "%s__layers_%dof%d_le_%.4g__k%d" % (
                        signal,
                        required,
                        per_layer.shape[2],
                        threshold,
                        consecutive,
                    )
                    rows.append(
                        rule_stats(
                            rule_id,
                            "layer_consensus",
                            signal,
                            "per_layer_norm",
                            stops,
                            data,
                            direction="layer_count_le",
                            threshold=float(threshold),
                            threshold_2=required,
                            consecutive=consecutive,
                            calibration="fixed",
                        )
                    )
    routed = raw["routed_rms"] / np.maximum(raw["routed_rms"][:, :1], EPS)
    mass = raw["expert_mass"] / np.maximum(raw["expert_mass"][:, :1], EPS)
    routed_mean = routed.mean(axis=2)
    mass_mean = mass.mean(axis=2)
    for routed_threshold in (0.80, 0.825, 0.85, 0.875, 0.90, 0.925):
        for mass_threshold in (0.80, 0.825, 0.85, 0.875, 0.90, 0.925):
            condition = (routed_mean <= routed_threshold) & (mass_mean <= mass_threshold)
            for consecutive in (1, 2):
                stops = first_stop(condition, consecutive=consecutive, min_round=3)
                rule_id = "routed_mass__le_%.3f_%.3f__k%d" % (
                    routed_threshold,
                    mass_threshold,
                    consecutive,
                )
                rows.append(
                    rule_stats(
                        rule_id,
                        "cross_signal_consensus",
                        "routed_rms+expert_mass",
                        "layernorm_mean",
                        stops,
                        data,
                        direction="both_le",
                        threshold=routed_threshold,
                        threshold_2=mass_threshold,
                        consecutive=consecutive,
                        calibration="fixed",
                    )
                )
    return rows


def _clock_diagnostics(
    raw: dict[str, np.ndarray], views: dict[str, dict[str, np.ndarray]], layers: np.ndarray
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    clock: dict[str, Any] = {}
    trajectories: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    for signal, signal_views in views.items():
        values = signal_views["aggregate_then_norm"]
        template = np.median(values, axis=0)
        correlations = []
        for row in values:
            correlations.append(float(np.corrcoef(row, template)[0, 1]))
        total_ss = float(np.sum(np.square(values - values.mean())))
        step_ss = float(len(values) * np.sum(np.square(values.mean(axis=0) - values.mean())))
        minima = np.argmin(values, axis=1) + 1
        minimum_values, minimum_counts = np.unique(minima, return_counts=True)
        clock[signal] = {
            "step_eta_squared": step_ss / total_ss if total_ss > 0 else 1.0,
            "template_correlation_median": float(np.median(correlations)),
            "template_correlation_q25": float(np.percentile(correlations, 25)),
            "template_correlation_q75": float(np.percentile(correlations, 75)),
            "minimum_round_histogram": {
                str(int(value)): int(count)
                for value, count in zip(minimum_values, minimum_counts)
            },
        }
        for round_index in range(TOTAL_ROUNDS):
            trajectories.append(
                {
                    "signal": signal,
                    "aggregation": "aggregate_then_norm",
                    "round": round_index + 1,
                    "median": float(np.median(values[:, round_index])),
                    "q25": float(np.percentile(values[:, round_index], 25)),
                    "q75": float(np.percentile(values[:, round_index], 75)),
                }
            )

        per_layer = raw[signal] / np.maximum(raw[signal][:, :1], EPS)
        for layer_axis, layer in enumerate(layers):
            layer_values = per_layer[:, :, layer_axis]
            minimum_round = np.argmin(layer_values, axis=1) + 1
            mode_values, mode_counts = np.unique(minimum_round, return_counts=True)
            stops = first_stop(layer_values <= 0.85, consecutive=2, min_round=3)
            layer_rows.append(
                {
                    "signal": signal,
                    "layer": int(layer),
                    "round1_absolute_median": float(
                        np.median(raw[signal][:, 0, layer_axis])
                    ),
                    "minimum_ratio_median": float(np.median(np.min(layer_values, axis=1))),
                    "round10_ratio_median": float(np.median(layer_values[:, -1])),
                    "minimum_round_mode": int(mode_values[np.argmax(mode_counts)]),
                    "minimum_round_mode_fraction": float(mode_counts.max() / len(minimum_round)),
                    "level_085_k2_save_pct": float(
                        100.0 * np.mean(TOTAL_ROUNDS - stops) / TOTAL_ROUNDS
                    ),
                    "level_085_k2_coverage_pct": float(100.0 * np.mean(stops < TOTAL_ROUNDS)),
                }
            )
    return clock, trajectories, layer_rows


def _token_aggregation_diagnostics(
    data: AuditData,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    """Compare stored token means with dispersion-sensitive recoverable proxies.

    PyTorch's stored token standard deviation uses the unbiased denominator.
    With ten tokens, sqrt(mean(x^2)) is therefore recoverable as
    sqrt(mean(x)^2 + 9/10 * std(x)^2).  The exact P90 and maximum are not.
    ``mean + 3 std`` is a distribution-free upper bound on the maximum for ten
    values, but is generally loose and is not labeled as an observed maximum.
    """

    diagnostics: list[dict[str, Any]] = []
    sweep_views: dict[str, dict[str, np.ndarray]] = {}
    representative_rows: list[dict[str, Any]] = []
    stems = {
        "expert_mass": "expert_mass_rms",
        "routed_rms": "routed_rms",
        "shared_rms": "shared_rms",
    }
    for signal, stem in stems.items():
        mean = data.metrics[..., data.metric_names.index(stem + "_mean")]
        std = data.metrics[..., data.metric_names.index(stem + "_token_std")]
        variants = token_aggregation_proxies(mean, std)
        for variant, values in variants.items():
            normalized = activation_views(values, data.layers)["aggregate_then_norm"]
            stops = first_stop(normalized <= 0.85, consecutive=2, min_round=3)
            diagnostics.append(
                {
                    "signal": signal,
                    "token_aggregation": variant,
                    "round1_absolute_median": float(np.median(values[:, 0])),
                    "round1_token_cv_median": float(
                        np.median(std[:, 0] / np.maximum(mean[:, 0], EPS))
                    ),
                    "normalized_trajectory_median": json.dumps(
                        [float(value) for value in np.median(normalized, axis=0)]
                    ),
                    "level_085_k2_save_pct": float(
                        100.0 * np.mean(TOTAL_ROUNDS - stops) / TOTAL_ROUNDS
                    ),
                    "level_085_k2_coverage_pct": float(100.0 * np.mean(stops < TOTAL_ROUNDS)),
                    "level_085_k2_median_stop": float(np.median(stops)),
                }
            )
            representative_rows.append(
                rule_stats(
                    "%s__%s__level_085_k2" % (signal, variant),
                    "token_aggregation_level",
                    signal,
                    variant,
                    stops,
                    data,
                    direction="le",
                    threshold=0.85,
                    threshold_2="",
                    consecutive=2,
                    calibration="fixed",
                )
            )
            if variant != "token_mean":
                sweep_views["%s__%s" % (signal, variant)] = activation_views(
                    values, data.layers
                )
    return diagnostics, sweep_views, representative_rows


def _vector_mean_diagnostics(data: AuditData) -> list[dict[str, Any]]:
    """Diagnose vectors that were already averaged across action tokens."""

    rows = []
    for signal, values in data.token_mean_vector_rms.items():
        normalized = activation_views(values, data.layers)["aggregate_then_norm"]
        stops = first_stop(normalized <= 0.85, consecutive=2, min_round=3)
        rows.append(
            {
                "signal": signal,
                "definition": "RMS(mean_over_action_tokens(vector))",
                "normalized_trajectory_median": json.dumps(
                    [float(value) for value in np.median(normalized, axis=0)]
                ),
                "level_085_k2_save_pct": float(
                    100.0 * np.mean(TOTAL_ROUNDS - stops) / TOTAL_ROUNDS
                ),
                "level_085_k2_coverage_pct": float(100.0 * np.mean(stops < TOTAL_ROUNDS)),
                "level_085_k2_median_stop": float(np.median(stops)),
            }
        )
    return rows


def _representative_rules(
    data: AuditData,
    raw: dict[str, np.ndarray],
    views: dict[str, dict[str, np.ndarray]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    stops_by_name: dict[str, np.ndarray] = {}
    routed = views["routed_rms"]["aggregate_then_norm"]
    routed_layer = raw["routed_rms"] / np.maximum(raw["routed_rms"][:, :1], EPS)
    mass = views["expert_mass"]["aggregate_then_norm"]
    relative_delta = np.full_like(routed, np.inf)
    relative_delta[:, 1:] = (routed[:, 1:] - routed[:, :-1]) / np.maximum(
        np.abs(routed[:, :-1]), EPS
    )
    curvature = np.full_like(routed, -np.inf)
    curvature[:, 2:] = routed[:, 2:] - 2.0 * routed[:, 1:-1] + routed[:, :-2]

    conditions = {
        "routed_level_085_k2": (routed <= 0.85, 2, "level"),
        "routed_layer_median_085_k2": (
            views["routed_rms"]["layernorm_median"] <= 0.85,
            2,
            "level",
        ),
        "routed_all4_090_k2": (
            np.all(routed_layer <= 0.90, axis=2),
            2,
            "layer_consensus",
        ),
        "routed_flat_0015_level_0925": (
            (np.abs(relative_delta) <= 0.015) & (routed <= 0.925),
            1,
            "flat_slope",
        ),
        "routed_curvature_001_level_085": (
            (curvature >= 0.01) & (routed <= 0.85),
            1,
            "positive_curvature",
        ),
        "routed_rebound_level_090": (
            (relative_delta >= 0.0) & (routed <= 0.90),
            1,
            "turn_or_rebound",
        ),
    }
    mass_delta = np.full_like(mass, np.inf)
    mass_delta[:, 1:] = (mass[:, 1:] - mass[:, :-1]) / np.maximum(
        np.abs(mass[:, :-1]), EPS
    )
    conditions["mass_rebound_level_090"] = (
        (mass_delta >= 0.0) & (mass <= 0.90),
        1,
        "turn_or_rebound",
    )

    rows = []
    for name, (condition, consecutive, family) in conditions.items():
        stops = first_stop(condition, consecutive=consecutive, min_round=3)
        stops_by_name[name] = stops
        rows.append(
            rule_stats(
                name,
                family,
                "expert_mass" if name.startswith("mass") else "routed_rms",
                "explicit_representative",
                stops,
                data,
                direction="see_rule_id",
                threshold="",
                threshold_2="",
                consecutive=consecutive,
                calibration="predefined_diagnostic",
            )
        )
    return rows, stops_by_name


def _pairwise_stop_agreement(stops: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    names = list(stops)
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            difference = np.abs(stops[left] - stops[right])
            correlation = float(np.corrcoef(stops[left], stops[right])[0, 1])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "exact_agreement": float(np.mean(difference == 0)),
                    "within_one_round": float(np.mean(difference <= 1)),
                    "mean_absolute_round_difference": float(difference.mean()),
                    "stop_correlation": correlation,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_hist(value: str) -> str:
    return ", ".join("%s:%s" % item for item in json.loads(value).items())


def render_report(summary: dict[str, Any]) -> str:
    clock = summary["clock_diagnostics"]
    representatives = summary["representative_rules"]
    old = next(row for row in representatives if row["rule_id"] == "routed_level_085_k2")
    layer_rows = [
        row for row in summary["layer_diagnostics"] if row["signal"] == "routed_rms"
    ]
    token_rows = [
        row
        for row in summary["token_aggregation_diagnostics"]
        if row["signal"] == "routed_rms"
    ]
    lines = [
        "# 真实专家激活幅度的无监督早停审计",
        "",
        "## 边界先说清楚",
        "",
        "本数据包含每个 query 的 10 轮真实 MoE 专家输出，但不包含对应的逐轮动作/flow 状态。"
        "因此这里能回答的是规则何时触发、能省多少轮、是否跨场景稳定；不能回答提前输出动作是否安全，"
        "也不能把节省率写成任务成功率。所有规则都只使用当前轮和历史轮，未偷看后续激活。",
        "",
        "数据为 LIBERO-Long t08 的首个 policy query：%d 个 query，%d 个初始场景 x %d 个共享噪声 seed；"
        "每个 query 10 轮，层为 %s。" % (
            summary["dataset"]["queries"],
            summary["dataset"]["scenes"],
            summary["dataset"]["noise_seeds"],
            ", ".join(str(value) for value in summary["dataset"]["layers"]),
        ),
        "",
        "## 核心结果",
        "",
        "`routed_rms` 是 top-4 专家输出先按路由权重合并、再计算 RMS，包含专家间抵消，"
        "是这里最接近 routed 分支真实写入幅度的量。按每个 query 的第 1 轮归一化后，"
        "总体中位轨迹为：`%s`。它在第 8 轮附近触底后回升，而不是趋近于 0。" % " ".join(
            "%.3f" % value for value in summary["trajectories"]["routed_rms"]
        ),
        "",
        "更重要的是，这个轨迹非常像去噪轮次时钟：轮次解释 `routed_rms` 总变异的 %.1f%%，"
        "单个 query 与总体模板的相关中位数为 %.3f。shared 幅度更极端，轮次解释 %.1f%%，"
        "模板相关中位数为 %.3f。变化稳定不等于动作已经收敛。" % (
            100.0 * clock["routed_rms"]["step_eta_squared"],
            clock["routed_rms"]["template_correlation_median"],
            100.0 * clock["shared_rms"]["step_eta_squared"],
            clock["shared_rms"]["template_correlation_median"],
        ),
        "",
        "| 信号 | 轮次解释率 | query-模板相关中位数 | 幅度最小轮分布 |",
        "|---|---:|---:|---|",
    ]
    for signal in BASE_SIGNALS:
        row = clock[signal]
        histogram = ", ".join(
            "%s:%s" % item for item in row["minimum_round_histogram"].items()
        )
        lines.append(
            "| %s | %.1f%% | %.3f | %s |"
            % (
                signal,
                100.0 * row["step_eta_squared"],
                row["template_correlation_median"],
                histogram,
            )
        )

    lines.extend(
        [
            "",
            "## 路由幅度规则",
            "",
            "复算了此前的规则：聚合 4 层 routed RMS 后除以本 query 第 1 轮，连续两轮低于 0.85。"
            "结果是中位第 %.0f 轮停止、平均节省 %.2f%%，%0.2f%% 的 query 会在第 10 轮前触发；"
            "停止分布为 `%s`。这只复现了触发/节省行为。" % (
                old["median_stop"],
                old["save_pct"],
                old["coverage_pct"],
                _fmt_hist(old["stop_histogram"]),
            ),
            "",
            "| 代表规则 | 平均节省 | 触发覆盖 | 中位停止轮 | 场景节省范围 | 停止分布 |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in representatives:
        lines.append(
            "| %s | %.2f%% | %.2f%% | %.0f | %.2f-%.2f%% | %s |"
            % (
                row["rule_id"],
                row["save_pct"],
                row["coverage_pct"],
                row["median_stop"],
                row["scene_save_min_pct"],
                row["scene_save_max_pct"],
                _fmt_hist(row["stop_histogram"]),
            )
        )
    lines.extend(
        [
            "",
            "同样约 20%% 的平均节省可以由不同形状规则得到，但逐 query 决策并不相同。"
            "例如 level 与 flat 代表规则的精确停止轮一致率只有 %.1f%%，停止轮相关为 %.3f。"
            "没有动作安全标签时，不能从这些规则中挑一个叫作最优。" % (
                100.0 * summary["key_agreement"]["exact_agreement"],
                summary["key_agreement"]["stop_correlation"],
            ),
            "",
            "## Action-token 聚合限制",
            "",
            "当前归档只保存了 10 个 action token 的均值和标准差，没有逐 token 数值。"
            "因此真实 max/P90 无法还原。下表中的 token-RMS 可由均值和方差精确恢复；`mean+std` 只是尾部代理；"
            "`mean+3std` 是 10 个 token 最大值的保守上界，不是观测到的最大值。",
            "",
            "| routed token 聚合 | 第1轮绝对量 | token CV | 0.85x2 节省 | 覆盖 | 中位停止轮 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in token_rows:
        lines.append(
            "| %s | %.4f | %.3f | %.2f%% | %.2f%% | %.0f |"
            % (
                row["token_aggregation"],
                row["round1_absolute_median"],
                row["round1_token_cv_median"],
                row["level_085_k2_save_pct"],
                row["level_085_k2_coverage_pct"],
                row["level_085_k2_median_stop"],
            )
        )
    vector_total = next(
        row
        for row in summary["vector_mean_diagnostics"]
        if row["signal"] == "routed_plus_shared"
    )
    lines.extend(
        [
            "",
            "routed 的四种可恢复聚合给出的理论节省约 20.6-23.5%%，说明加入 token 离散度后"
            "触发率没有发生数量级变化；但这不能排除少数 token 尖峰，因为 max/P90 本身仍缺失。"
            "另外，归档里的向量已经先跨 token 平均；用它构造 `routed+shared` 后，0.85x2 会节省 %.2f%%、"
            "中位第 %.0f 轮停，几乎完全跟随 shared 分支，不能作为 total 收敛证据。" % (
                vector_total["level_085_k2_save_pct"],
                vector_total["level_085_k2_median_stop"],
            ),
            "",
            "这里四个量必须分开解释：`expert_mass` 是各专家幅度加权和、忽略专家抵消；"
            "`routed_rms` 是同一 token 内专家向量加权合并后的净 RMS；`shared_rms` 是 shared MLP 幅度；"
            "真实逐 token 的 `routed+shared` 总幅度未保存，现有 token-mean 向量会同时隐藏 token 尖峰和 token 间抵消。",
            "",
            "## 逐层异质性",
            "",
            "相同的 `低于首轮 0.85、连续两轮` 放到单层，节省率从约 2% 到 36%；"
            "所以直接平均 4 层不是无争议的尺度选择。",
            "",
            "| 层 | 第1轮绝对幅度中位数 | 最小/首轮中位数 | 最小轮众数 | 0.85x2 节省 | 覆盖 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in layer_rows:
        lines.append(
            "| %d | %.4f | %.3f | %d | %.2f%% | %.2f%% |"
            % (
                row["layer"],
                row["round1_absolute_median"],
                row["minimum_ratio_median"],
                row["minimum_round_mode"],
                row["level_085_k2_save_pct"],
                row["level_085_k2_coverage_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## 裁决与下一步",
            "",
            "1. 激活幅度确实能稳定地产生早停触发，0.85x2 规则可复现约 20.8% 的理论计算节省。",
            "2. 目前最强的证据反而说明它主要编码去噪阶段：轨迹高度模板化，shared 分支也有更强的同类形状。",
            "3. query 内归一化解决了绝对层尺度，但没有解决层间语义不同，也没有建立“幅度小 = 剩余动作误差小”。",
            "4. 下一次只需给相同 query 额外保存 10 轮动作/flow 状态；直接读取 `representative_stops.csv`，"
            "冻结这些规则后比较各自的第 10 轮动作误差。优先检验 level、all4-veto、curvature 和 rebound 四个家族。",
            "",
            "完整扫描在 `all_rules.csv`，包含绝对值、query 内归一化、逐层阈值、连续阈值、斜率、"
            "曲率、回升和跨信号共识；它是探索清单，不是经过安全性选择的模型榜单。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    features_path = Path(args.features).resolve()
    summary_path = Path(args.source_summary).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(features_path, summary_path)
    raw = {
        signal: data.metrics[..., data.metric_names.index(metric_name)]
        for signal, metric_name in BASE_SIGNALS.items()
    }
    views = {
        signal: activation_views(values, data.layers) for signal, values in raw.items()
    }

    token_rows, token_sweep_views, token_representatives = (
        _token_aggregation_diagnostics(data)
    )
    vector_rows = _vector_mean_diagnostics(data)
    all_rules = _level_sweep(data, views)
    all_rules.extend(_level_sweep(data, token_sweep_views))
    all_rules.extend(_shape_sweep(data, views))
    all_rules.extend(_consensus_sweep(data, raw))
    all_rules.extend(token_representatives)
    representatives, representative_stops = _representative_rules(data, raw, views)
    agreements = _pairwise_stop_agreement(representative_stops)
    clock, trajectory_rows, layer_rows = _clock_diagnostics(raw, views, data.layers)

    trajectory_lookup = {
        signal: [
            row["median"]
            for row in trajectory_rows
            if row["signal"] == signal
            and row["aggregation"] == "aggregate_then_norm"
        ]
        for signal in BASE_SIGNALS
    }
    key_agreement = next(
        row
        for row in agreements
        if {row["left"], row["right"]}
        == {"routed_level_085_k2", "routed_flat_0015_level_0925"}
    )
    output_summary = {
        "scope": "trigger_and_compute_savings_only_no_action_safety",
        "dataset": {
            "features": str(features_path),
            "source_summary": str(summary_path),
            "queries": len(data.metrics),
            "rounds": TOTAL_ROUNDS,
            "layers": data.layers.tolist(),
            "scenes": int(len(np.unique(data.scenes))),
            "noise_seeds": int(len(np.unique(data.noise_seeds))),
        },
        "rule_count": len(all_rules),
        "clock_diagnostics": clock,
        "trajectories": trajectory_lookup,
        "representative_rules": representatives,
        "key_agreement": key_agreement,
        "layer_diagnostics": layer_rows,
        "token_aggregation_diagnostics": token_rows,
        "vector_mean_diagnostics": vector_rows,
    }

    _write_csv(out_dir / "all_rules.csv", all_rules)
    _write_csv(out_dir / "trajectory_summary.csv", trajectory_rows)
    _write_csv(out_dir / "layer_diagnostics.csv", layer_rows)
    _write_csv(out_dir / "token_aggregation_diagnostics.csv", token_rows)
    _write_csv(out_dir / "vector_mean_diagnostics.csv", vector_rows)
    _write_csv(out_dir / "representative_agreement.csv", agreements)
    stop_rows = []
    for row_index in range(len(data.metrics)):
        row: dict[str, Any] = {
            "episode_id": int(data.episode_ids[row_index]),
            "scene": int(data.scenes[row_index]),
            "noise_seed": int(data.noise_seeds[row_index]),
        }
        row.update(
            {
                name: int(stops[row_index])
                for name, stops in representative_stops.items()
            }
        )
        stop_rows.append(row)
    _write_csv(out_dir / "representative_stops.csv", stop_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(output_summary, ensure_ascii=True, indent=2) + "\n"
    )
    (out_dir / "report.md").write_text(render_report(output_summary))
    print("wrote %d rules to %s" % (len(all_rules), out_dir))


if __name__ == "__main__":
    main()
