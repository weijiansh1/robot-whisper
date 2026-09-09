#!/usr/bin/env python3
"""Pilot the chain: MoE route -> action commitment -> candidate pruning.

This deliberately separates three claims that are easy to conflate:

1. a route prefix predicts how much denoising remains;
2. a route prefix can preserve the final full-pool action consensus after pruning;
3. the preserved action consensus improves closed-loop task success.

The available captures can test (1), and can shadow-test (2).  They cannot test
(3), because candidates that were retrospectively pruned were never executed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from analyze_weighted_noise_evolution import load_capture


HERE = Path(__file__).resolve().parent
DEFAULT_FLOW_RUN = HERE / "runs/flow-lead-cpu-t0s24"
DEFAULT_SHADOW = HERE / "analysis/route-prune-vote/goal-v4.json"
DEFAULT_OUT = HERE / "analysis/action-commitment-pilot"
SCHEMA = "himoe-action-commitment-pilot-v1"
LIVE_DIMS = 7
SURVIVOR_COUNT = 8
LOW_REGRET = 0.02
RIDGE_ALPHA = 1.0


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    """Pairwise RMS distance for an array whose first axis is candidate."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite and have at least two candidates")
    flat = array.reshape(len(array), -1)
    return np.sqrt(np.mean(np.square(flat[:, None] - flat[None, :]), axis=-1))


def medoid_index(distance: np.ndarray, candidates: Iterable[int] | None = None) -> int:
    """Distance medoid with a stable lowest-candidate-ID tie break."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be square")
    indices = (
        np.arange(len(matrix), dtype=np.int64)
        if candidates is None
        else np.unique(np.fromiter(candidates, dtype=np.int64))
    )
    if not len(indices) or np.any(indices < 0) or np.any(indices >= len(matrix)):
        raise ValueError("candidate subset is empty or out of range")
    score = matrix[np.ix_(indices, indices)].sum(axis=1)
    optimum = np.min(score)
    return int(np.min(indices[np.isclose(score, optimum, atol=1e-12, rtol=0.0)]))


def central_subset(distance: np.ndarray, count: int) -> np.ndarray:
    """Keep the candidates with lowest all-candidate distance centrality."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance must be square")
    if not 1 <= count <= len(matrix):
        raise ValueError("invalid survivor count")
    return np.lexsort((np.arange(len(matrix)), matrix.sum(axis=1)))[:count]


def regret_vector(final_distance: np.ndarray) -> np.ndarray:
    """Global-medoid regret for selecting each candidate as the vote result."""

    matrix = np.asarray(final_distance, dtype=np.float64)
    upper = matrix[np.triu_indices(len(matrix), 1)]
    scale = float(np.median(upper))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("final action pool has no positive distance scale")
    centrality = matrix.mean(axis=1)
    optimum = centrality[medoid_index(matrix)]
    return np.maximum((centrality - optimum) / scale, 0.0)


def subset_metrics(final_distance: np.ndarray, survivors: Iterable[int]) -> dict[str, float]:
    """Quality of the final action-medoid vote after retaining ``survivors``."""

    keep = np.unique(np.fromiter(survivors, dtype=np.int64))
    full = medoid_index(final_distance)
    selected = medoid_index(final_distance, keep)
    regret = float(regret_vector(final_distance)[selected])
    return {
        "full_medoid_retained": float(full in keep),
        "selected_equals_full_medoid": float(selected == full),
        "normalized_global_medoid_regret": regret,
        "regret_at_most_0.02": float(regret <= LOW_REGRET),
        "regret_at_most_0.05": float(regret <= 0.05),
    }


def exact_random_subset_metrics(
    final_distance: np.ndarray, survivor_count: int
) -> dict[str, float]:
    """Exact expectation over every uniform survivor subset."""

    candidate_count = len(final_distance)
    subsets = np.asarray(
        list(itertools.combinations(range(candidate_count), survivor_count)),
        dtype=np.int64,
    )
    within = final_distance[subsets[:, :, None], subsets[:, None, :]].sum(axis=2)
    local = np.argmin(within, axis=1)
    selected = subsets[np.arange(len(subsets)), local]
    full = medoid_index(final_distance)
    regrets = regret_vector(final_distance)[selected]
    return {
        "full_medoid_retained": float(np.mean(np.any(subsets == full, axis=1))),
        "selected_equals_full_medoid": float(np.mean(selected == full)),
        "normalized_global_medoid_regret": float(np.mean(regrets)),
        "regret_at_most_0.02": float(np.mean(regrets <= LOW_REGRET)),
        "regret_at_most_0.05": float(np.mean(regrets <= 0.05)),
    }


def route_prefix_distances(probability: np.ndarray) -> np.ndarray:
    """Cumulative Hellinger geometry for [K, layer, flow, token, expert]."""

    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 5 or values.shape[-1] != 32:
        raise ValueError("router probability must have shape [K,L,T,A,32]")
    values = np.maximum(values, 0.0)
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("router probability contains a zero-mass site")
    values /= mass
    per_step = []
    for flow in range(values.shape[2]):
        embedding = np.sqrt(values[:, :, flow]).reshape(len(values), -1)
        per_step.append(pairwise_rms(embedding))
    per_step = np.stack(per_step)
    divisor = np.arange(1, len(per_step) + 1, dtype=np.float64)[:, None, None]
    return np.cumsum(per_step, axis=0) / divisor


def stabilized_stage(good: np.ndarray) -> int:
    """First stage after which a Boolean condition remains true."""

    values = np.asarray(good, dtype=bool)
    if values.ndim != 1 or not len(values) or not values[-1]:
        raise ValueError("condition must be one-dimensional and true at the final stage")
    for stage in range(len(values)):
        if np.all(values[stage:]):
            return stage
    raise AssertionError("the final true value guarantees a stabilization stage")


def idealized_saving(
    candidate_count: int, survivor_count: int, completed_rounds: int, total_rounds: int
) -> float:
    """Ideal candidate-round saving after pruning at a completed flow round."""

    if not 0 <= completed_rounds <= total_rounds:
        raise ValueError("completed rounds are out of range")
    if not 1 <= survivor_count <= candidate_count:
        raise ValueError("survivor count is out of range")
    used = candidate_count * completed_rounds
    used += survivor_count * (total_rounds - completed_rounds)
    return float(1.0 - used / (candidate_count * total_rounds))


def _mean_group_summary(values: list[float], groups: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    group = np.asarray(groups, dtype=np.int64)
    unique = np.unique(group)
    means = np.asarray([array[group == item].mean() for item in unique])
    return {
        "query_mean": float(array.mean()),
        "rollout_macro_mean": float(means.mean()),
        "rollout_mean_range": [float(means.min()), float(means.max())],
        "per_rollout_mean": {
            str(int(item)): float(value) for item, value in zip(unique, means)
        },
    }


def _metric_summary(rows: list[dict[str, float]], groups: list[int]) -> dict[str, Any]:
    return {
        key: _mean_group_summary([row[key] for row in rows], groups)
        for key in rows[0]
    }


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else 0.0


def _route_features(current: np.ndarray) -> np.ndarray:
    """Pool-level route observables available at one completed flow round."""

    probability = np.asarray(current, dtype=np.float64)
    probability = np.maximum(probability, 0.0)
    probability /= probability.sum(axis=-1, keepdims=True)
    ordered = np.sort(probability, axis=-1)[..., ::-1]
    entropy = -(
        probability * np.log(np.maximum(probability, 1e-20))
    ).sum(axis=-1) / math.log(32.0)
    embedding = np.sqrt(probability).reshape(len(probability), -1)
    cloud = float(np.sqrt(np.mean(np.square(embedding - embedding.mean(axis=0)))))

    top1 = np.argmax(probability, axis=-1).reshape(len(probability), -1)
    agreement = []
    for site in range(top1.shape[1]):
        counts = np.bincount(top1[:, site], minlength=32)
        agreement.append(float(np.sum(counts * (counts - 1)) / (len(top1) * (len(top1) - 1))))
    return np.asarray(
        [
            entropy.mean(),
            entropy.std(),
            ordered[..., 0].mean(),
            (ordered[..., 0] - ordered[..., 1]).mean(),
            ordered[..., :4].sum(axis=-1).mean(),
            cloud,
            np.mean(agreement),
            entropy[:, :4].mean(),
            entropy[:, 4:].mean(),
            ordered[:, :4, ..., 0].mean(),
            ordered[:, 4:, ..., 0].mean(),
        ],
        dtype=np.float64,
    )


def _latent_features(trajectory: np.ndarray, completed_rounds: int) -> np.ndarray:
    current = np.asarray(trajectory[:, completed_rounds], dtype=np.float64)
    previous = np.asarray(trajectory[:, completed_rounds - 1], dtype=np.float64)
    flat = current.reshape(len(current), -1)
    previous_flat = previous.reshape(len(previous), -1)
    dispersion = float(np.sqrt(np.mean(np.square(flat - flat.mean(axis=0)))))
    previous_dispersion = float(
        np.sqrt(np.mean(np.square(previous_flat - previous_flat.mean(axis=0))))
    )
    return np.asarray(
        [
            dispersion,
            np.sqrt(np.mean(np.square(flat))),
            np.sqrt(np.mean(np.square(current - previous))),
            dispersion - previous_dispersion,
        ],
        dtype=np.float64,
    )


def _oof_ridge(
    features: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> np.ndarray:
    prediction = np.empty_like(target, dtype=np.float64)
    for held in np.unique(groups):
        train = groups != held
        test = ~train
        scaler = StandardScaler().fit(features[train])
        model = Ridge(alpha=alpha).fit(scaler.transform(features[train]), target[train])
        prediction[test] = model.predict(scaler.transform(features[test]))
    return prediction


def _prediction_metrics(
    target: np.ndarray, prediction: np.ndarray, completed_rounds: np.ndarray
) -> dict[str, float]:
    total = float(np.sum(np.square(target - target.mean())))
    residual = float(np.sum(np.square(target - prediction)))
    centered_target = target.copy()
    centered_prediction = prediction.copy()
    for stage in np.unique(completed_rounds):
        selected = completed_rounds == stage
        centered_target[selected] -= centered_target[selected].mean()
        centered_prediction[selected] -= centered_prediction[selected].mean()
    centered_total = float(np.sum(np.square(centered_target)))
    centered_residual = float(
        np.sum(np.square(centered_target - centered_prediction))
    )
    return {
        "oof_r2": float(1.0 - residual / max(total, 1e-20)),
        "oof_spearman": _rank_correlation(target, prediction),
        "within_round_oof_r2": float(
            1.0 - centered_residual / max(centered_total, 1e-20)
        ),
        "within_round_oof_spearman": _rank_correlation(
            centered_target, centered_prediction
        ),
    }


def load_routes(run: Path, expected_query: np.ndarray) -> np.ndarray:
    group = zarr.open_group(str(run / "routes.zarr"), mode="r")
    route_query = np.asarray(group["episode_id"][:], dtype=np.int64)
    if not np.array_equal(route_query, expected_query):
        raise ValueError("flow trajectories and router rows are not aligned")
    probability = np.asarray(
        group["hb_router_probs"][:, :, :, 1:11, :], dtype=np.float64
    )
    probability = np.maximum(probability, 0.0)
    probability /= probability.sum(axis=-1, keepdims=True)
    return probability


def analyze_flow_commitment(run: Path, alpha: float) -> dict[str, Any]:
    data = load_capture(run)
    trajectory = np.asarray(data["trajectory"][:, :, :, :LIVE_DIMS], dtype=np.float64)
    query_id = np.asarray(data["query_id"], dtype=np.int64)
    candidate_id = np.asarray(data["candidate_id"], dtype=np.int64)
    probability = load_routes(run, query_id)
    records = json.loads((run / "query_records.json").read_text())
    rollout_by_query = {
        int(row["query_id"]): int(row["flow_noise_seed"]) for row in records
    }

    query_count = len(np.unique(query_id))
    flow_rounds = trajectory.shape[1] - 1
    stage_values: dict[int, dict[str, list[float]]] = {
        stage: {
            "geometry_spearman": [],
            "provisional_medoid_exact": [],
            "provisional_medoid_regret": [],
            "provisional_medoid_low_regret": [],
        }
        for stage in range(flow_rounds + 1)
    }
    route_rows: dict[int, list[dict[str, float]]] = {
        stage: [] for stage in range(1, flow_rounds + 1)
    }
    latent_rows: dict[int, list[dict[str, float]]] = {
        stage: [] for stage in range(1, flow_rounds + 1)
    }
    noise_rows: list[dict[str, float]] = []
    random_rows: list[dict[str, float]] = []
    rollout_groups: list[int] = []
    stabilization = {
        "exact_medoid": [],
        "regret_at_most_0.02": [],
        "geometry_spearman_at_least_0.90": [],
    }
    regression_route = []
    regression_latent = []
    regression_round = []
    regression_target = []
    regression_group = []

    for query in np.unique(query_id):
        indices = np.flatnonzero(query_id == query)
        indices = indices[np.argsort(candidate_id[indices])]
        if len(indices) != 16 or not np.array_equal(candidate_id[indices], np.arange(16)):
            raise ValueError("query %d is not an ordered K16 pool" % query)
        values = trajectory[indices]
        routes = probability[indices]
        group = rollout_by_query[int(query)]
        rollout_groups.append(group)
        final_distance = pairwise_rms(values[:, -1])
        final_medoid = medoid_index(final_distance)
        final_upper = np.triu_indices(len(final_distance), 1)
        final_pairs = final_distance[final_upper]
        regrets = regret_vector(final_distance)

        exact_curve = []
        low_regret_curve = []
        rho_curve = []
        for stage in range(flow_rounds + 1):
            provisional_distance = pairwise_rms(values[:, stage])
            provisional_medoid = medoid_index(provisional_distance)
            rho = (
                1.0
                if stage == flow_rounds
                else _rank_correlation(
                    provisional_distance[final_upper], final_pairs
                )
            )
            regret = float(regrets[provisional_medoid])
            exact = provisional_medoid == final_medoid
            low = regret <= LOW_REGRET
            stage_values[stage]["geometry_spearman"].append(rho)
            stage_values[stage]["provisional_medoid_exact"].append(float(exact))
            stage_values[stage]["provisional_medoid_regret"].append(regret)
            stage_values[stage]["provisional_medoid_low_regret"].append(float(low))
            exact_curve.append(exact)
            low_regret_curve.append(low)
            rho_curve.append(rho >= 0.90)
        stabilization["exact_medoid"].append(stabilized_stage(np.asarray(exact_curve)))
        stabilization["regret_at_most_0.02"].append(
            stabilized_stage(np.asarray(low_regret_curve))
        )
        stabilization["geometry_spearman_at_least_0.90"].append(
            stabilized_stage(np.asarray(rho_curve))
        )

        route_prefix = route_prefix_distances(routes)
        noise_distance = pairwise_rms(values[:, 0])
        noise_keep = central_subset(noise_distance, SURVIVOR_COUNT)
        noise_rows.append(subset_metrics(final_distance, noise_keep))
        random_rows.append(exact_random_subset_metrics(final_distance, SURVIVOR_COUNT))

        for completed in range(1, flow_rounds + 1):
            route_keep = central_subset(route_prefix[completed - 1], SURVIVOR_COUNT)
            latent_distance = pairwise_rms(values[:, completed])
            latent_keep = central_subset(latent_distance, SURVIVOR_COUNT)
            route_rows[completed].append(subset_metrics(final_distance, route_keep))
            latent_rows[completed].append(subset_metrics(final_distance, latent_keep))

            if completed < flow_rounds:
                regression_route.append(_route_features(routes[:, :, completed - 1]))
                regression_latent.append(_latent_features(values, completed))
                regression_round.append(completed)
                regression_target.append(
                    float(np.sqrt(np.mean(np.square(values[:, completed] - values[:, -1]))))
                )
                regression_group.append(group)

    groups = np.asarray(regression_group, dtype=np.int64)
    rounds = np.asarray(regression_round, dtype=np.int64)
    target = np.asarray(regression_target, dtype=np.float64)
    route_feature = np.asarray(regression_route, dtype=np.float64)
    latent_feature = np.asarray(regression_latent, dtype=np.float64)
    time_feature = np.eye(flow_rounds - 1, dtype=np.float64)[rounds - 1]
    feature_sets = {
        "round_only": time_feature,
        "round_plus_route": np.column_stack([time_feature, route_feature]),
        "round_plus_current_latent": np.column_stack([time_feature, latent_feature]),
        "round_plus_latent_plus_route": np.column_stack(
            [time_feature, latent_feature, route_feature]
        ),
    }
    prediction = {}
    for name, features in feature_sets.items():
        estimate = _oof_ridge(features, target, groups, alpha)
        prediction[name] = _prediction_metrics(target, estimate, rounds)

    stage_curve = []
    stage_groups = [rollout_by_query[int(query)] for query in np.unique(query_id)]
    for stage in range(flow_rounds + 1):
        row: dict[str, Any] = {"completed_rounds": stage}
        for key, values in stage_values[stage].items():
            row[key] = _mean_group_summary(values, stage_groups)
        stage_curve.append(row)

    pruning_curve = []
    for completed in range(1, flow_rounds + 1):
        pruning_curve.append(
            {
                "completed_rounds": completed,
                "idealized_saving_fraction": idealized_saving(
                    16, SURVIVOR_COUNT, completed, flow_rounds
                ),
                "route_prefix": _metric_summary(route_rows[completed], stage_groups),
                "current_action_latent": _metric_summary(
                    latent_rows[completed], stage_groups
                ),
            }
        )

    stabilization_out = {}
    for name, values in stabilization.items():
        array = np.asarray(values, dtype=np.int64)
        stabilization_out[name] = {
            "median_completed_round": float(np.median(array)),
            "q25_q75": [float(value) for value in np.percentile(array, [25, 75])],
            "count_by_round": {
                str(stage): int(np.sum(array == stage))
                for stage in range(flow_rounds + 1)
            },
        }

    return {
        "data": {
            "run": str(run.resolve()),
            "query_count": query_count,
            "rollout_count": len(set(rollout_groups)),
            "candidate_count": 16,
            "flow_rounds": flow_rounds,
            "action_shape": [10, LIVE_DIMS],
            "normalization_max_error": data["normalization_max_error"],
        },
        "definitions": {
            "provisional_geometry": "pairwise RMS geometry of normalized live7 x_m versus x_10",
            "route_prefix": "cumulative full-softmax Hellinger geometry over HB action tokens",
            "commitment_threshold": "normalized global-medoid regret <= 0.02",
            "stabilization": "first stage after which the condition stays true through x_10",
            "prediction_target": "pool RMS remaining correction ||x_m-x_10||",
            "prediction_cv": "leave one complete rollout out; fixed ridge alpha",
        },
        "stage_curve": stage_curve,
        "stabilization": stabilization_out,
        "pruning_baselines": {
            "initial_noise": _metric_summary(noise_rows, stage_groups),
            "exact_uniform_random_k8": _metric_summary(random_rows, stage_groups),
        },
        "pruning_curve": pruning_curve,
        "remaining_correction_prediction": {
            "sample_count": len(target),
            "rollout_groups": len(np.unique(groups)),
            "ridge_alpha": alpha,
            "models": prediction,
            "route_increment_over_round_r2": float(
                prediction["round_plus_route"]["oof_r2"]
                - prediction["round_only"]["oof_r2"]
            ),
            "route_increment_over_latent_r2": float(
                prediction["round_plus_latent_plus_route"]["oof_r2"]
                - prediction["round_plus_current_latent"]["oof_r2"]
            ),
        },
    }


def _compact_metric(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_macro_mean": metric["task_macro_mean"],
        "task_cluster_bootstrap_ci_95": metric["task_cluster_bootstrap_ci_95"],
    }


def load_multitask_shadow(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("schema") != "himoe-route-prune-vote-shadow-v1":
        raise ValueError("unexpected route-prune shadow schema")
    row = next(
        item
        for item in report["route_prefix_pruning"]
        if item["prefix_steps"] == 1 and item["survivor_count"] == 6
    )
    keys = (
        "full_winner_retained",
        "selected_equals_full_winner",
        "normalized_global_medoid_regret",
        "regret_at_most_0.02",
        "regret_at_most_0.05",
    )
    return {
        "source": str(path.resolve()),
        "data": report["data"],
        "condition": {
            "completed_rounds": 1,
            "candidate_count": 8,
            "survivor_count": 6,
            "idealized_saving_fraction": row["idealized_flow_saving_fraction"],
        },
        "route": {key: _compact_metric(row["metrics"][key]) for key in keys},
        "route_minus_exact_random": {
            key: _compact_metric(row["paired_difference_minus_exact_random"][key])
            for key in keys
        },
        "route_minus_initial_noise": {
            key: _compact_metric(row["paired_difference_minus_initial_noise"][key])
            for key in keys
        },
        "grid_status": report["statistics"]["grid_status"],
    }


def _mean(summary: dict[str, Any]) -> float:
    return float(summary["rollout_macro_mean"])


def make_figure(flow: dict[str, Any], out: Path) -> None:
    stages = np.asarray([row["completed_rounds"] for row in flow["stage_curve"]])
    rho = np.asarray(
        [_mean(row["geometry_spearman"]) for row in flow["stage_curve"]]
    )
    exact = np.asarray(
        [_mean(row["provisional_medoid_exact"]) for row in flow["stage_curve"]]
    )
    completed = np.asarray(
        [row["completed_rounds"] for row in flow["pruning_curve"]]
    )

    def curve(method: str, metric: str) -> np.ndarray:
        return np.asarray(
            [_mean(row[method][metric]) for row in flow["pruning_curve"]]
        )

    route_regret = curve("route_prefix", "normalized_global_medoid_regret")
    latent_regret = curve("current_action_latent", "normalized_global_medoid_regret")
    route_safe = curve("route_prefix", "regret_at_most_0.02")
    latent_safe = curve("current_action_latent", "regret_at_most_0.02")
    noise_regret = _mean(
        flow["pruning_baselines"]["initial_noise"][
            "normalized_global_medoid_regret"
        ]
    )
    random_regret = _mean(
        flow["pruning_baselines"]["exact_uniform_random_k8"][
            "normalized_global_medoid_regret"
        ]
    )
    saving = np.asarray(
        [row["idealized_saving_fraction"] for row in flow["pruning_curve"]]
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    axes[0].plot(stages, rho, marker="o", color="#2367a8", label="pair geometry rho")
    axes[0].plot(stages, exact, marker="s", color="#bd3d31", label="exact medoid")
    axes[0].axvline(10, color="#444444", linestyle=":", linewidth=1)
    axes[0].set(xlabel="completed flow rounds", ylabel="fraction / correlation", ylim=(-0.03, 1.04))
    axes[0].set_title("True action commitment")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(completed, route_regret, marker="o", color="#bd3d31", label="route prefix")
    axes[1].plot(completed, latent_regret, marker="s", color="#237a57", label="current latent")
    axes[1].axhline(noise_regret, color="#2367a8", linestyle="--", label="initial noise")
    axes[1].axhline(random_regret, color="#555555", linestyle=":", label="exact random")
    axes[1].set(xlabel="completed flow rounds", ylabel="mean normalized regret")
    axes[1].set_title("K16 to K8 shadow pruning")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(100 * saving, 100 * route_safe, marker="o", color="#bd3d31", label="route prefix")
    axes[2].plot(100 * saving, 100 * latent_safe, marker="s", color="#237a57", label="current latent")
    axes[2].axhline(95, color="#444444", linestyle=":", linewidth=1)
    axes[2].set(xlabel="ideal candidate-round saving (%)", ylabel="regret <= 0.02 pools (%)", ylim=(0, 103))
    axes[2].set_title("Quality versus ideal saving")
    axes[2].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("HiMoE-VLA action commitment pilot", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _pct(value: float) -> str:
    return "%.1f%%" % (100.0 * value)


def write_report(summary: dict[str, Any], path: Path) -> None:
    flow = summary["flow_commitment"]
    shadow = summary["multitask_pruning_shadow"]
    selected_stages = {0, 1, 3, 5, 7, 9, 10}
    stage_rows = []
    for row in flow["stage_curve"]:
        stage = row["completed_rounds"]
        if stage not in selected_stages:
            continue
        stage_rows.append(
            "| %d | %.3f | %s | %.4f |"
            % (
                stage,
                _mean(row["geometry_spearman"]),
                _pct(_mean(row["provisional_medoid_exact"])),
                _mean(row["provisional_medoid_regret"]),
            )
        )

    prediction = flow["remaining_correction_prediction"]
    prediction_rows = []
    labels = {
        "round_only": "flow round only",
        "round_plus_route": "flow round + MoE route",
        "round_plus_current_latent": "flow round + current latent",
        "round_plus_latent_plus_route": "flow round + latent + route",
    }
    for key, label in labels.items():
        metric = prediction["models"][key]
        prediction_rows.append(
            "| %s | %.3f | %.3f | %.3f |"
            % (
                label,
                metric["oof_r2"],
                metric["oof_spearman"],
                metric["within_round_oof_r2"],
            )
        )

    route = shadow["route"]
    vs_random = shadow["route_minus_exact_random"]
    vs_noise = shadow["route_minus_initial_noise"]
    report = f"""# Action commitment and MoE pruning pilot

## Question

Can the HB-MoE route estimate when an action computation is stable enough to reduce the candidate budget? This report does **not** treat action stability as task success.

## Result

The direction is partially supported as an **action-compute proxy**, but MoE-specific pruning is not yet established.

- Route features improve leave-one-rollout-out prediction of remaining denoising correction over the flow-round clock: R2 `{prediction['models']['round_only']['oof_r2']:.3f}` to `{prediction['models']['round_plus_route']['oof_r2']:.3f}`.
- The already-available current action latent is much stronger: R2 `{prediction['models']['round_plus_current_latent']['oof_r2']:.3f}`. Adding route changes it by `{prediction['route_increment_over_latent_r2']:+.4f}`.
- Exact action-medoid identity is not committed early: median stabilization round is `{flow['stabilization']['exact_medoid']['median_completed_round']:.0f}` of 10. Pairwise action geometry also reaches the predeclared rho>=0.90 condition only at the final round for all 44 queries.
- A separate 10-task/701-pool shadow does show that conservative route pruning can preserve final action consensus better than random, but initial-noise centrality remains better than route.

## True latent pilot

One Goal task, one initial scene, 4 correlated rollouts, 44 query states, K16 candidates. `x0...x10` are the true model-normalized action latents. Distances use the live `10x7` dimensions.

| completed rounds | provisional/final geometry rho | exact final medoid | provisional medoid regret |
|---:|---:|---:|---:|
{chr(10).join(stage_rows)}

`rho` compares all 120 candidate-pair distances to the final action geometry. Regret is the selected candidate's excess full-pool centrality, divided by the pool median pair distance. Identity is strict; low regret can still hold when several central candidates are near-tied.

## Commitment sensor

Target: pool RMS remaining correction `||x_m-x10||`. Models use a fixed ridge alpha `{prediction['ridge_alpha']}` and leave one complete rollout out.

| observable | OOF R2 | OOF Spearman | within-round OOF R2 |
|---|---:|---:|---:|
{chr(10).join(prediction_rows)}

This is the useful positive result: routing contains a readable "how unfinished is this computation?" signal beyond merely knowing the denoise round. It is not an independent information advantage because the sampler's current latent and last update are already available and predict the same target substantially better.

## Multi-task pruning shadow

The separate RAD shadow contains `{shadow['data']['pool_count']}` K8 pools across `{shadow['data']['task_count']}` LIBERO-Goal tasks. After one of ten flow rounds, route centrality keeps K6, for an ideal candidate-round saving of `{_pct(shadow['condition']['idealized_saving_fraction'])}`.

| metric | route K6 | route - exact random | route - initial noise |
|---|---:|---:|---:|
| exact K8 medoid reproduced | {_pct(route['selected_equals_full_winner']['task_macro_mean'])} | {100*vs_random['selected_equals_full_winner']['task_macro_mean']:+.1f} pp | {100*vs_noise['selected_equals_full_winner']['task_macro_mean']:+.1f} pp |
| normalized medoid regret | {route['normalized_global_medoid_regret']['task_macro_mean']:.4f} | {vs_random['normalized_global_medoid_regret']['task_macro_mean']:+.4f} | {vs_noise['normalized_global_medoid_regret']['task_macro_mean']:+.4f} |
| regret <= 0.02 | {_pct(route['regret_at_most_0.02']['task_macro_mean'])} | {100*vs_random['regret_at_most_0.02']['task_macro_mean']:+.1f} pp | {100*vs_noise['regret_at_most_0.02']['task_macro_mean']:+.1f} pp |
| regret <= 0.05 | {_pct(route['regret_at_most_0.05']['task_macro_mean'])} | {100*vs_random['regret_at_most_0.05']['task_macro_mean']:+.1f} pp | {100*vs_noise['regret_at_most_0.05']['task_macro_mean']:+.1f} pp |

For normalized regret, lower is better. Route minus random is `{vs_random['normalized_global_medoid_regret']['task_macro_mean']:+.4f}` with task-cluster 95% CI `{vs_random['normalized_global_medoid_regret']['task_cluster_bootstrap_ci_95']}`. Route minus noise is `{vs_noise['normalized_global_medoid_regret']['task_macro_mean']:+.4f}` with CI `{vs_noise['normalized_global_medoid_regret']['task_cluster_bootstrap_ci_95']}`, so the cheap noise baseline is significantly better on this development set.

## Decision

1. **Supported:** MoE routing carries a measurable action-computation progress signal, and conservative route-central pruning preserves final action consensus better than uniform random pruning.
2. **Not supported:** routing adds value beyond the current action latent or the known initial noise; exact action commitment before the final denoise round; any claim about task success, real GPU latency, or rescue decisions.
3. **Next confirmation:** recapture K32 with `x0...x10`, freeze one point (`m=3`, K32 to K8), and test `current latent + route` against `current latent` on fully held-out tasks and fresh, non-reused seeds. Only if route has positive incremental value should an online compaction/latency experiment be run.

![commitment curves](commitment_curve.png)

## Scope

- The K16 mechanism pilot has only four correlated rollouts from one task and one scene; its cross-validation is diagnostic, not confirmatory.
- The 701-pool pruning experiment is retrospective: all candidates completed all ten flow rounds.
- The multi-task grid was exploratory and was previously inspected; its confidence intervals are descriptive rather than a fresh preregistered test.
- Final episode success is not a valid label for whether the first action chunk was correct, and no alternative pruned candidate was executed here.
"""
    path.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-run", type=Path, default=DEFAULT_FLOW_RUN)
    parser.add_argument("--shadow-summary", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ridge-alpha", type=float, default=RIDGE_ALPHA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.ridge_alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    flow = analyze_flow_commitment(args.flow_run, args.ridge_alpha)
    shadow = load_multitask_shadow(args.shadow_summary)
    summary = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "question": "Can MoE routing estimate action commitment well enough to reduce candidate compute?",
        "flow_commitment": flow,
        "multitask_pruning_shadow": shadow,
        "decision": {
            "route_is_action_progress_proxy": True,
            "route_pruning_beats_uniform_random": True,
            "route_has_increment_over_current_latent": False,
            "route_has_increment_over_initial_noise": False,
            "early_exact_action_commitment_detected": False,
            "closed_loop_success_or_latency_validated": False,
        },
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    make_figure(flow, args.out_dir / "commitment_curve.png")
    write_report(summary, args.out_dir / "REPORT.md")
    print("wrote %s" % summary_path)
    print("wrote %s" % (args.out_dir / "REPORT.md"))
    print("wrote %s" % (args.out_dir / "commitment_curve.png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
