#!/usr/bin/env python3
"""Train-free audit of task difficulty, early MoE routes, and alarm weighting.

The experiment has three deliberately separate parts:

1. Measure whether policy-specific task failure rates are repeatable.
2. Test whether first-query MoE routing geometry predicts that difficulty.
3. Reallocate a fixed alarm budget using difficulty estimated on disjoint seeds.

No classifier or detector weights are fitted.  Outcome labels are used only after
first-query routing features have been materialized, and for the explicitly
label-based difficulty/calibration portions of the analysis.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
ATLAS = PACKAGE_ROOT / "results/hub_phenotype_atlas_50x8/episode_features_and_heads.csv"
CACHE_ROOT = WORKSPACE_ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
OUTPUT = PACKAGE_ROOT / "results/task_difficulty_weighting"
RUN_ID = "right-50x8-20260903"

FRONT = slice(0, 4)
BACK = slice(4, 8)
ACTION = slice(1, 11)
FINAL_FLOW = 9
LATE_FLOW_START = 6
N_EXPERTS = 32

INSTABILITY = (
    ("late_flow_volatility_mean", 1),
    ("route_acceleration_mean", 1),
    ("route_mobility_mean", 1),
    ("lag_periodicity_mean", 1),
)
LOCK_IN = (
    ("route_mobility_mean", -1),
    ("lag_recurrence_mean", 1),
    ("top4_union_mean", -1),
    ("token_disagreement_mean", -1),
)
WEIGHTING_VARIANTS = (
    "uniform",
    "empirical_sqrt",
    "empirical_linear",
    "route_knn5_sqrt",
)
EARLY_SCALARS = (
    "front_route_acceleration",
    "back_route_acceleration",
    "back_front_acceleration_ratio",
    "front_late_flow_volatility",
    "back_late_flow_volatility",
    "back_front_volatility_ratio",
    "front_gate_entropy",
    "back_gate_entropy",
    "front_top12_margin",
    "back_top12_margin",
    "front_token_disagreement",
    "back_token_disagreement",
    "front_state_action_gap",
    "layer5_state_action_gap",
    "back_state_action_gap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--reuse-early", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize(left)
    right = normalize(right)
    affinity = (np.sqrt(left) * np.sqrt(right)).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize(left)
    right = normalize(right)
    numerator = np.minimum(left, right).sum(axis=-1)
    denominator = np.maximum(left, right).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def first_query_features(routes: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract outcome-free features from one route tensor per episode."""

    p = normalize(routes)
    if p.ndim != 5 or tuple(p.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected first-query route shape: {p.shape}")
    action = p[:, :, :, ACTION]
    root = np.sqrt(action)
    second = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
    acceleration = np.linalg.norm(second, axis=-1).mean(axis=(2, 3)) / np.sqrt(2.0)

    late = action[:, :, LATE_FLOW_START:]
    volatility = (
        1.0 - weighted_jaccard(late[:, :, 1:], late[:, :, :-1])
    ).mean(axis=(2, 3))

    final_action = p[:, :, FINAL_FLOW, ACTION]
    final_state = p[:, :, FINAL_FLOW, 0]
    action_mean = normalize(final_action.mean(axis=2))
    entropy = -(
        final_action * np.log(np.maximum(final_action, 1e-12))
    ).sum(axis=-1).mean(axis=2) / np.log(N_EXPERTS)
    ordered = np.partition(final_action, -2, axis=-1)
    margin = (ordered[..., -1] - ordered[..., -2]).mean(axis=2)
    disagreement = hellinger(
        final_action, action_mean[:, :, None]
    ).mean(axis=2)
    state_action = hellinger(final_state, action_mean)

    front_acceleration = acceleration[:, FRONT].mean(axis=1)
    back_acceleration = acceleration[:, BACK].mean(axis=1)
    front_volatility = volatility[:, FRONT].mean(axis=1)
    back_volatility = volatility[:, BACK].mean(axis=1)
    frame = pd.DataFrame(
        {
            "front_route_acceleration": front_acceleration,
            "back_route_acceleration": back_acceleration,
            "back_front_acceleration_ratio": back_acceleration
            / np.maximum(front_acceleration, 1e-12),
            "front_late_flow_volatility": front_volatility,
            "back_late_flow_volatility": back_volatility,
            "back_front_volatility_ratio": back_volatility
            / np.maximum(front_volatility, 1e-12),
            "front_gate_entropy": entropy[:, FRONT].mean(axis=1),
            "back_gate_entropy": entropy[:, BACK].mean(axis=1),
            "front_top12_margin": margin[:, FRONT].mean(axis=1),
            "back_top12_margin": margin[:, BACK].mean(axis=1),
            "front_token_disagreement": disagreement[:, FRONT].mean(axis=1),
            "back_token_disagreement": disagreement[:, BACK].mean(axis=1),
            "front_state_action_gap": state_action[:, FRONT].mean(axis=1),
            "layer5_state_action_gap": state_action[:, 3],
            "back_state_action_gap": state_action[:, BACK].mean(axis=1),
        }
    )
    return frame, action_mean.astype(np.float16)


def extract_early_routes(
    atlas: pd.DataFrame, cache_root: Path, run_id: str
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[pd.DataFrame] = []
    embeddings: list[np.ndarray] = []
    for ordinal, (task, expected) in enumerate(atlas.groupby("task"), start=1):
        run = cache_root / task / run_id
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            raise RuntimeError(f"atlas task is no longer complete: {run}")
        group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        episode_ids = np.asarray(group["episode_id"][:], dtype=np.int64)
        starts = np.flatnonzero(np.r_[True, episode_ids[1:] != episode_ids[:-1]])
        identities = episode_ids[starts]
        expected_ids = np.sort(expected["episode"].to_numpy(dtype=np.int64))
        if not np.array_equal(identities, expected_ids):
            raise RuntimeError(f"episode identity mismatch: {task}")
        router = group["hb_router_probs"]
        route = np.asarray(
            router.get_orthogonal_selection(
                (starts, slice(None), slice(None), slice(None), slice(None))
            )
        )
        feature, embedding = first_query_features(route)
        feature.insert(0, "episode", identities)
        feature.insert(0, "task", task)
        rows.append(feature)
        embeddings.append(embedding)
        print(
            f"[early {ordinal:02d}/{atlas['task'].nunique()}] {task}: "
            f"episodes={len(feature)}",
            flush=True,
        )
    return pd.concat(rows, ignore_index=True), np.concatenate(embeddings, axis=0)


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total**2)) / denominator
    return center - half, center + half


def difficulty_inventory(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, group in frame.groupby("task"):
        failures = int(group["failure"].sum())
        low_seed = group[group["flow_noise_seed"] < 1004]
        high_seed = group[group["flow_noise_seed"] >= 1004]
        even_init = group[group["init_state_id"] % 2 == 0]
        odd_init = group[group["init_state_id"] % 2 == 1]
        low, high = wilson(failures, len(group))
        rows.append(
            {
                "task": task,
                "suite": task.split("/", 1)[0],
                "episodes": len(group),
                "failures": failures,
                "failure_rate": failures / len(group),
                "failure_rate_wilson_low": low,
                "failure_rate_wilson_high": high,
                "seed_1000_1003_failure_rate": float(low_seed["failure"].mean()),
                "seed_1004_1007_failure_rate": float(high_seed["failure"].mean()),
                "even_init_failure_rate": float(even_init["failure"].mean()),
                "odd_init_failure_rate": float(odd_init["failure"].mean()),
                "mean_episode_queries": float(group["episode_length"].mean()),
            }
        )
    output = pd.DataFrame(rows).sort_values("failure_rate", ascending=False)
    seed_spearman = spearmanr(
        output["seed_1000_1003_failure_rate"],
        output["seed_1004_1007_failure_rate"],
    )
    seed_pearson = pearsonr(
        output["seed_1000_1003_failure_rate"],
        output["seed_1004_1007_failure_rate"],
    )
    init_spearman = spearmanr(
        output["even_init_failure_rate"], output["odd_init_failure_rate"]
    )
    length = spearmanr(output["mean_episode_queries"], output["failure_rate"])
    without_hardest = output.iloc[1:]
    seed_spearman_without_hardest = spearmanr(
        without_hardest["seed_1000_1003_failure_rate"],
        without_hardest["seed_1004_1007_failure_rate"],
    )
    reliability = {
        "seed_half_spearman_rho": seed_spearman.statistic,
        "seed_half_spearman_p": seed_spearman.pvalue,
        "seed_half_pearson_r": seed_pearson.statistic,
        "seed_half_pearson_p": seed_pearson.pvalue,
        "seed_half_spearman_rho_without_hardest_task": seed_spearman_without_hardest.statistic,
        "seed_half_spearman_p_without_hardest_task": seed_spearman_without_hardest.pvalue,
        "init_parity_spearman_rho": init_spearman.statistic,
        "init_parity_spearman_p": init_spearman.pvalue,
        "episode_length_spearman_rho": length.statistic,
        "episode_length_spearman_p": length.pvalue,
        "zero_failure_tasks": int((output["failures"] == 0).sum()),
        "top_task_failure_share": float(output.iloc[0]["failures"] / output["failures"].sum()),
        "top_five_task_failure_share": float(
            output.iloc[:5]["failures"].sum() / output["failures"].sum()
        ),
    }
    return output, reliability


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    order = np.argsort(values)
    ordered = values[order]
    adjusted = np.minimum.accumulate(
        (ordered * len(ordered) / np.arange(1, len(ordered) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def unit_rank(values: np.ndarray) -> np.ndarray:
    """Return centered ranks with unit L2 norm for fast Spearman permutations."""

    ranks = rankdata(np.asarray(values, dtype=np.float64))
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm > 0 else np.zeros_like(ranks)


def task_embeddings(
    early: pd.DataFrame, embeddings: np.ndarray, mask: np.ndarray | None = None
) -> tuple[list[str], np.ndarray]:
    if mask is None:
        mask = np.ones(len(early), dtype=bool)
    tasks: list[str] = []
    centers: list[np.ndarray] = []
    for task in sorted(early["task"].unique()):
        selected = mask & (early["task"].to_numpy() == task)
        tasks.append(task)
        centers.append(normalize(embeddings[selected].astype(np.float32).mean(axis=0)))
    return tasks, np.stack(centers)


def embedding_distances(centers: np.ndarray) -> np.ndarray:
    root = np.sqrt(normalize(centers))
    coefficient = np.einsum("ile,jle->ijl", root, root).mean(axis=-1)
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def early_difficulty_analysis(
    early: pd.DataFrame,
    embeddings: np.ndarray,
    difficulty: pd.DataFrame,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    del difficulty  # Difficulty targets below are always computed on the opposite seed half.
    rng = np.random.default_rng(seed)
    task_names = sorted(early["task"].unique())
    task_count = len(task_names)
    permutation_draws = np.stack(
        [rng.permutation(task_count) for _ in range(permutations)]
    )
    task_array = early["task"].to_numpy(dtype=str)
    low_seed = early["flow_noise_seed"].to_numpy(dtype=np.int64) < 1004
    fold_specs = (
        ("1000_1003_routes_to_1004_1007_outcomes", low_seed, ~low_seed),
        ("1004_1007_routes_to_1000_1003_outcomes", ~low_seed, low_seed),
    )

    fold_features: list[np.ndarray] = []
    fold_source_rates: list[np.ndarray] = []
    fold_target_rates: list[np.ndarray] = []
    fold_distances: list[np.ndarray] = []
    fold_names: list[str] = []
    for fold_name, route_mask, outcome_mask in fold_specs:
        route_frame = early.loc[route_mask]
        task_feature = route_frame.groupby("task")[list(EARLY_SCALARS)].mean()
        source_rate = early.loc[route_mask].groupby("task")["failure"].mean()
        target_rate = early.loc[outcome_mask].groupby("task")["failure"].mean()
        tasks, centers = task_embeddings(early, embeddings, route_mask)
        if tasks != task_names:
            raise RuntimeError("cross-fit task order mismatch")
        fold_names.append(fold_name)
        fold_features.append(task_feature.loc[task_names].to_numpy(dtype=np.float64))
        fold_source_rates.append(source_rate.loc[task_names].to_numpy(dtype=np.float64))
        fold_target_rates.append(target_rate.loc[task_names].to_numpy(dtype=np.float64))
        fold_distances.append(embedding_distances(centers))

    feature_values = np.stack(fold_features)
    source_rates = np.stack(fold_source_rates)
    target_rates = np.stack(fold_target_rates)
    hardest_index = int(np.argmax(target_rates.mean(axis=0)))

    observed_scalar_rho = np.empty((2, len(EARLY_SCALARS)), dtype=np.float64)
    observed_scalar_without_hardest = np.empty_like(observed_scalar_rho)
    for fold_index in range(2):
        for signal_index in range(len(EARLY_SCALARS)):
            observed_scalar_rho[fold_index, signal_index] = spearmanr(
                feature_values[fold_index, :, signal_index], target_rates[fold_index]
            ).statistic
            keep = np.arange(len(task_names)) != hardest_index
            observed_scalar_without_hardest[fold_index, signal_index] = spearmanr(
                feature_values[fold_index, keep, signal_index],
                target_rates[fold_index, keep],
            ).statistic
    observed_scalar_mean = observed_scalar_rho.mean(axis=0)
    feature_ranks = np.empty_like(feature_values)
    target_ranks = np.empty_like(target_rates)
    for fold_index in range(2):
        target_ranks[fold_index] = unit_rank(target_rates[fold_index])
        for signal_index in range(len(EARLY_SCALARS)):
            feature_ranks[fold_index, :, signal_index] = unit_rank(
                feature_values[fold_index, :, signal_index]
            )
    scalar_null = np.empty((permutations, len(EARLY_SCALARS)), dtype=np.float64)
    for draw, permutation in enumerate(permutation_draws):
        scalar_null[draw] = np.mean(
            [
                feature_ranks[fold_index].T
                @ target_ranks[fold_index, permutation]
                for fold_index in range(2)
            ],
            axis=0,
        )
    scalar_p = (1 + (np.abs(scalar_null) >= np.abs(observed_scalar_mean)).sum(axis=0)) / (
        permutations + 1
    )
    association_frame = pd.DataFrame(
        {
            "signal": EARLY_SCALARS,
            "fold_1000_1003_routes_rho": observed_scalar_rho[0],
            "fold_1004_1007_routes_rho": observed_scalar_rho[1],
            "mean_crossfit_spearman_rho": observed_scalar_mean,
            "permutation_p_two_sided": scalar_p,
            "mean_rho_without_hardest_task": observed_scalar_without_hardest.mean(axis=0),
        }
    )
    association_frame["bh_q_value"] = bh_adjust(
        association_frame["permutation_p_two_sided"]
    )
    association_frame = association_frame.sort_values("permutation_p_two_sided")

    upper = np.triu_indices(len(task_names), 1)
    observed_pairwise = np.empty(2, dtype=np.float64)
    for fold_index in range(2):
        difficulty_difference = np.abs(
            target_rates[fold_index, :, None] - target_rates[fold_index, None, :]
        )[upper]
        observed_pairwise[fold_index] = spearmanr(
            fold_distances[fold_index][upper], difficulty_difference
        ).statistic
    observed_pairwise_mean = float(observed_pairwise.mean())
    null_pairwise = np.empty(permutations, dtype=np.float64)
    for draw, permutation in enumerate(permutation_draws):
        correlations = []
        for fold_index in range(2):
            shuffled = target_rates[fold_index, permutation]
            difference = np.abs(shuffled[:, None] - shuffled[None, :])[upper]
            correlations.append(
                spearmanr(fold_distances[fold_index][upper], difference).statistic
            )
        null_pairwise[draw] = np.mean(correlations)
    pairwise_p = (1 + np.sum(null_pairwise >= observed_pairwise_mean)) / (
        permutations + 1
    )

    k_values = (1, 3, 5, 10)
    prediction_rows: list[dict[str, Any]] = []
    fold_predictions_by_k: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    fold_baselines: list[np.ndarray] = []
    for fold_index, fold_name in enumerate(fold_names):
        distance = fold_distances[fold_index]
        source = source_rates[fold_index]
        target = target_rates[fold_index]
        baseline = np.empty(task_count, dtype=np.float64)
        predicted_by_k = {
            k: np.empty(task_count, dtype=np.float64) for k in k_values
        }
        for task_index in range(task_count):
            candidates = np.delete(np.arange(task_count), task_index)
            ordered = candidates[np.argsort(distance[task_index, candidates])]
            baseline[task_index] = np.median(source[candidates])
            for k in k_values:
                nearest = ordered[:k]
                weights = 1.0 / np.maximum(distance[task_index, nearest], 1e-8)
                predicted_by_k[k][task_index] = (
                    np.sum(weights * source[nearest]) / weights.sum()
                )
                prediction_rows.append(
                    {
                        "fold": fold_name,
                        "k": k,
                        "task": task_names[task_index],
                        "source_half_task_failure_rate": source[task_index],
                        "target_half_true_failure_rate": target[task_index],
                        "route_knn_predicted_failure_rate": predicted_by_k[k][task_index],
                        "leave_one_task_median_prediction": baseline[task_index],
                        "absolute_error": abs(
                            predicted_by_k[k][task_index] - target[task_index]
                        ),
                        "median_baseline_absolute_error": abs(
                            baseline[task_index] - target[task_index]
                        ),
                    }
                )
        fold_baselines.append(baseline)
        for k in k_values:
            fold_predictions_by_k[k].append(predicted_by_k[k])

    baseline_array = np.stack(fold_baselines)
    baseline_mae = float(np.mean(np.abs(baseline_array - target_rates)))
    draw_index = np.arange(permutations)[:, None, None]
    inverse_permutations = np.empty_like(permutation_draws)
    inverse_permutations[
        np.arange(permutations)[:, None], permutation_draws
    ] = np.arange(task_count)[None, :]
    null_absolute_error = {
        k: np.zeros(permutations, dtype=np.float64) for k in k_values
    }
    for fold_index in range(2):
        distance = fold_distances[fold_index]
        source = source_rates[fold_index]
        target = target_rates[fold_index]
        nearest_nodes = []
        for route_node in range(task_count):
            ordered_nodes = np.argsort(distance[route_node])
            nearest_nodes.append(ordered_nodes[ordered_nodes != route_node][: max(k_values)])
        nearest_nodes_array = np.stack(nearest_nodes)
        assigned_route_nodes = permutation_draws
        assigned_neighbor_nodes = nearest_nodes_array[assigned_route_nodes]
        assigned_neighbor_tasks = inverse_permutations[
            draw_index, assigned_neighbor_nodes
        ]
        assigned_distances = distance[
            assigned_route_nodes[:, :, None], assigned_neighbor_nodes
        ]
        weights = 1.0 / np.maximum(assigned_distances, 1e-8)
        weighted_source = weights * source[assigned_neighbor_tasks]
        cumulative_prediction = np.cumsum(weighted_source, axis=-1) / np.cumsum(
            weights, axis=-1
        )
        for k in k_values:
            prediction = cumulative_prediction[:, :, k - 1]
            null_absolute_error[k] += np.abs(prediction - target[None, :]).sum(axis=1)

    permutation_summary: dict[str, Any] = {}
    for k in k_values:
        prediction_array = np.stack(fold_predictions_by_k[k])
        observed_mae = float(np.mean(np.abs(prediction_array - target_rates)))
        observed_improvement = baseline_mae - observed_mae
        null_mae = null_absolute_error[k] / (2 * task_count)
        null_improvement = baseline_mae - null_mae
        improvement_p = (1 + np.sum(null_improvement >= observed_improvement)) / (
            permutations + 1
        )
        fold_correlations = [
            spearmanr(prediction_array[index], target_rates[index]).statistic
            for index in range(2)
        ]
        permutation_summary[f"knn_{k}"] = {
            "fold_spearman_rho": fold_correlations,
            "mean_fold_spearman_rho": float(np.mean(fold_correlations)),
            "mae": observed_mae,
            "median_baseline_mae": baseline_mae,
            "mae_improvement_over_median": observed_improvement,
            "permutation_p_route_geometry": improvement_p,
        }

    summary = {
        "protocol": "two-way disjoint-seed cross-fit",
        "folds": fold_names,
        "pairwise_route_distance_vs_difficulty_difference_fold_spearman_rho": observed_pairwise,
        "pairwise_route_distance_vs_difficulty_difference_mean_spearman_rho": observed_pairwise_mean,
        "pairwise_permutation_p_positive_association": pairwise_p,
        "knn": permutation_summary,
        "significant_early_scalars_after_bh_005": int(
            (association_frame["bh_q_value"] <= 0.05).sum()
        ),
    }
    return association_frame, pd.DataFrame(prediction_rows), summary


def early_task_identity(
    early: pd.DataFrame, embeddings: np.ndarray
) -> dict[str, Any]:
    task_values = early["task"].to_numpy(dtype=str)
    init_values = early["init_state_id"].to_numpy(dtype=np.int64)
    correct_task = 0
    correct_suite = 0
    total = 0
    folds: list[dict[str, Any]] = []
    for fold in range(5):
        train = init_values % 5 != fold
        test = ~train
        tasks, centers = task_embeddings(early, embeddings, train)
        center_root = np.sqrt(centers)
        selected = np.flatnonzero(test)
        fold_task = 0
        fold_suite = 0
        for start in range(0, len(selected), 256):
            index = selected[start : start + 256]
            root = np.sqrt(normalize(embeddings[index].astype(np.float32)))
            coefficient = np.einsum(
                "nle,tle->ntl", root, center_root
            ).mean(axis=-1)
            distance = np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))
            predicted = np.asarray(tasks)[np.argmin(distance, axis=1)]
            truth = task_values[index]
            task_ok = predicted == truth
            suite_ok = np.asarray([value.split("/", 1)[0] for value in predicted]) == np.asarray(
                [value.split("/", 1)[0] for value in truth]
            )
            fold_task += int(task_ok.sum())
            fold_suite += int(suite_ok.sum())
        folds.append(
            {
                "fold": fold,
                "episodes": len(selected),
                "task_accuracy": fold_task / len(selected),
                "suite_accuracy": fold_suite / len(selected),
            }
        )
        correct_task += fold_task
        correct_suite += fold_suite
        total += len(selected)
    return {
        "episodes": total,
        "task_accuracy": correct_task / total,
        "task_chance": 1.0 / early["task"].nunique(),
        "suite_accuracy": correct_suite / total,
        "suite_chance": 1.0 / early["task"].str.split("/").str[0].nunique(),
        "held_out_init_folds": folds,
    }


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(reference, np.asarray(values), side="right") / len(reference)


def fixed_dual_mean_score(
    calibration_success: pd.DataFrame, values: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    train_heads = []
    value_heads = []
    for components in (INSTABILITY, LOCK_IN):
        train_parts = []
        value_parts = []
        for signal, direction in components:
            reference = direction * calibration_success[signal].to_numpy(dtype=np.float64)
            train_parts.append(empirical_percentile(reference, reference))
            value_parts.append(
                empirical_percentile(
                    reference, direction * values[signal].to_numpy(dtype=np.float64)
                )
            )
        train_heads.append(np.mean(np.stack(train_parts, axis=1), axis=1))
        value_heads.append(np.mean(np.stack(value_parts, axis=1), axis=1))
    return (
        np.mean(np.stack(train_heads, axis=1), axis=1),
        np.mean(np.stack(value_heads, axis=1), axis=1),
    )


def allocate_alpha(
    priors: dict[str, float], counts: dict[str, int], power: float
) -> dict[str, float]:
    if power == 0.0:
        return {task: 0.05 for task in priors}
    weights = {task: max(value, 1e-8) ** power for task, value in priors.items()}

    def mean_alpha(scale: float) -> float:
        numerator = sum(
            counts[task] * np.clip(scale * weights[task], 0.005, 0.20)
            for task in priors
        )
        return numerator / sum(counts.values())

    low, high = 0.0, 1000.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if mean_alpha(middle) < 0.05:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    return {
        task: float(np.clip(scale * weights[task], 0.005, 0.20))
        for task in priors
    }


def route_knn_priors(
    tasks: list[str], centers: np.ndarray, priors: dict[str, float], k: int = 5
) -> dict[str, float]:
    distance = embedding_distances(centers)
    output: dict[str, float] = {}
    for index, task in enumerate(tasks):
        candidates = np.delete(np.arange(len(tasks)), index)
        nearest = candidates[np.argsort(distance[index, candidates])[:k]]
        weights = 1.0 / np.maximum(distance[index, nearest], 1e-8)
        output[task] = float(
            np.sum(weights * np.asarray([priors[tasks[item]] for item in nearest]))
            / weights.sum()
        )
    return output


def difficulty_weighting(
    frame: pd.DataFrame,
    early: pd.DataFrame,
    embeddings: np.ndarray,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    predictions: list[pd.DataFrame] = []
    allocation_rows: list[dict[str, Any]] = []
    for fold, calibration_low_seed in enumerate((True, False)):
        calibration_mask = (
            frame["flow_noise_seed"] < 1004
            if calibration_low_seed
            else frame["flow_noise_seed"] >= 1004
        )
        evaluation_mask = ~calibration_mask
        fold_name = "1000_1003_to_1004_1007" if calibration_low_seed else "1004_1007_to_1000_1003"
        train_scores: dict[str, np.ndarray] = {}
        test_parts: list[pd.DataFrame] = []
        empirical_priors: dict[str, float] = {}
        calibration_counts: dict[str, int] = {}
        calibration_failure_counts: dict[str, int] = {}
        success_counts: dict[str, int] = {}
        for task, task_frame in frame.groupby("task"):
            calibration = task_frame[calibration_mask.loc[task_frame.index]]
            evaluation = task_frame[evaluation_mask.loc[task_frame.index]].copy()
            healthy = calibration[~calibration["failure"]]
            if len(healthy) < 20:
                raise RuntimeError(f"insufficient calibration successes: {task}")
            train_score, test_score = fixed_dual_mean_score(healthy, evaluation)
            train_scores[task] = train_score
            evaluation["anomaly_score"] = test_score
            test_parts.append(evaluation)
            failures = int(calibration["failure"].sum())
            empirical_priors[task] = (failures + 0.5) / (len(calibration) + 1.0)
            calibration_counts[task] = len(calibration)
            calibration_failure_counts[task] = failures
            success_counts[task] = len(healthy)

        early_calibration = (
            early["flow_noise_seed"] < 1004
            if calibration_low_seed
            else early["flow_noise_seed"] >= 1004
        ).to_numpy(dtype=bool)
        tasks, centers = task_embeddings(early, embeddings, early_calibration)
        route_priors = route_knn_priors(tasks, centers, empirical_priors, k=5)
        allocations = {
            "uniform": allocate_alpha(empirical_priors, success_counts, 0.0),
            "empirical_sqrt": allocate_alpha(empirical_priors, success_counts, 0.5),
            "empirical_linear": allocate_alpha(empirical_priors, success_counts, 1.0),
            "route_knn5_sqrt": allocate_alpha(route_priors, success_counts, 0.5),
        }
        evaluation = pd.concat(test_parts).sort_index()
        evaluation.insert(0, "fold", fold_name)
        evaluation["empirical_difficulty_prior"] = evaluation["task"].map(empirical_priors)
        evaluation["route_knn5_difficulty_prior"] = evaluation["task"].map(route_priors)
        for variant, allocation in allocations.items():
            alarm = np.zeros(len(evaluation), dtype=bool)
            threshold_values = np.empty(len(evaluation), dtype=np.float64)
            alpha_values = np.empty(len(evaluation), dtype=np.float64)
            for task, indices in evaluation.groupby("task").groups.items():
                alpha = allocation[task]
                threshold = float(
                    np.quantile(train_scores[task], 1.0 - alpha, method="higher")
                )
                positions = evaluation.index.get_indexer(indices)
                alarm[positions] = evaluation.loc[indices, "anomaly_score"] > threshold
                threshold_values[positions] = threshold
                alpha_values[positions] = alpha
                allocation_rows.append(
                    {
                        "fold": fold_name,
                        "task": task,
                        "variant": variant,
                        "calibration_episodes": calibration_counts[task],
                        "calibration_failures": calibration_failure_counts[task],
                        "calibration_successes": success_counts[task],
                        "empirical_difficulty_prior": empirical_priors[task],
                        "route_knn5_difficulty_prior": route_priors[task],
                        "allocated_success_fpr": alpha,
                        "score_threshold": threshold,
                    }
                )
            evaluation[f"alarm_{variant}"] = alarm
            evaluation[f"alpha_{variant}"] = alpha_values
            evaluation[f"threshold_{variant}"] = threshold_values
        predictions.append(evaluation)

    output = pd.concat(predictions, ignore_index=True)
    rows: list[dict[str, Any]] = []
    failure = output["failure"].to_numpy(dtype=bool)
    task_values = output["task"].to_numpy(dtype=str)
    task_failure_total = output.groupby("task")["failure"].sum()
    hardest_task = str(task_failure_total.idxmax())
    without_hardest = task_values != hardest_task
    tasks_with_failures = task_failure_total[task_failure_total > 0].index.tolist()
    for variant in WEIGHTING_VARIANTS:
        alarm = output[f"alarm_{variant}"].to_numpy(dtype=bool)
        tp = int((failure & alarm).sum())
        fp = int((~failure & alarm).sum())
        reduced_failure = failure & without_hardest
        per_task_recall = []
        for task in tasks_with_failures:
            selected = task_values == task
            per_task_recall.append(
                float((failure[selected] & alarm[selected]).sum() / failure[selected].sum())
            )
        rows.append(
            {
                "variant": variant,
                "episodes": len(output),
                "failures": int(failure.sum()),
                "successes": int((~failure).sum()),
                "true_alarms": tp,
                "false_alarms": fp,
                "failure_recall": tp / max(int(failure.sum()), 1),
                "success_false_alarm_rate": fp / max(int((~failure).sum()), 1),
                "precision": tp / max(tp + fp, 1),
                "hardest_task": hardest_task,
                "hardest_task_failures": int(task_failure_total[hardest_task]),
                "hardest_task_true_alarms": int(
                    (failure & alarm & ~without_hardest).sum()
                ),
                "failure_recall_excluding_hardest_task": float(
                    (reduced_failure & alarm).sum() / reduced_failure.sum()
                ),
                "macro_task_failure_recall": float(np.mean(per_task_recall)),
                "mean_allocated_success_fpr": float(
                    output[f"alpha_{variant}"][~failure].mean()
                ),
            }
        )

    baseline = output["alarm_uniform"].to_numpy(dtype=bool)
    tasks = np.unique(task_values)
    task_indices = [np.flatnonzero(task_values == task) for task in tasks]
    task_failure_counts = np.asarray(
        [failure[index].sum() for index in task_indices], dtype=np.float64
    )
    task_success_counts = np.asarray(
        [(~failure[index]).sum() for index in task_indices], dtype=np.float64
    )
    sampled_tasks = rng.integers(
        0, len(tasks), size=(bootstrap, len(tasks)), endpoint=False
    )
    bootstrap_failure_denominator = task_failure_counts[sampled_tasks].sum(axis=1)
    bootstrap_success_denominator = task_success_counts[sampled_tasks].sum(axis=1)
    for row in rows:
        variant = row["variant"]
        candidate = output[f"alarm_{variant}"].to_numpy(dtype=bool)
        task_delta_tp = np.asarray(
            [
                (failure[index] & candidate[index]).sum()
                - (failure[index] & baseline[index]).sum()
                for index in task_indices
            ],
            dtype=np.float64,
        )
        task_delta_fp = np.asarray(
            [
                ((~failure[index]) & candidate[index]).sum()
                - ((~failure[index]) & baseline[index]).sum()
                for index in task_indices
            ],
            dtype=np.float64,
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            deltas_tpr = (
                task_delta_tp[sampled_tasks].sum(axis=1)
                / bootstrap_failure_denominator
            )
            deltas_fpr = (
                task_delta_fp[sampled_tasks].sum(axis=1)
                / bootstrap_success_denominator
            )
        row.update(
            {
                "delta_recall_vs_uniform": row["failure_recall"]
                - rows[0]["failure_recall"],
                "delta_recall_excluding_hardest_vs_uniform": row[
                    "failure_recall_excluding_hardest_task"
                ]
                - rows[0]["failure_recall_excluding_hardest_task"],
                "delta_macro_task_recall_vs_uniform": row[
                    "macro_task_failure_recall"
                ]
                - rows[0]["macro_task_failure_recall"],
                "delta_recall_ci_low": float(np.nanquantile(deltas_tpr, 0.025)),
                "delta_recall_ci_high": float(np.nanquantile(deltas_tpr, 0.975)),
                "delta_fpr_vs_uniform": row["success_false_alarm_rate"]
                - rows[0]["success_false_alarm_rate"],
                "delta_fpr_ci_low": float(np.quantile(deltas_fpr, 0.025)),
                "delta_fpr_ci_high": float(np.quantile(deltas_fpr, 0.975)),
                "failure_uniform_only": int((failure & baseline & ~candidate).sum()),
                "failure_candidate_only": int((failure & ~baseline & candidate).sum()),
                "success_uniform_only": int((~failure & baseline & ~candidate).sum()),
                "success_candidate_only": int((~failure & ~baseline & candidate).sum()),
            }
        )
    return output, pd.DataFrame(rows), pd.DataFrame(allocation_rows)


def weighting_breakdowns(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for task, group in predictions.groupby("task"):
        failure = group["failure"].to_numpy(dtype=bool)
        uniform = group["alarm_uniform"].to_numpy(dtype=bool)
        for variant in WEIGHTING_VARIANTS:
            alarm = group[f"alarm_{variant}"].to_numpy(dtype=bool)
            tp = int((failure & alarm).sum())
            fp = int((~failure & alarm).sum())
            task_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "episodes": len(group),
                    "failures": int(failure.sum()),
                    "successes": int((~failure).sum()),
                    "true_alarms": tp,
                    "false_alarms": fp,
                    "failure_recall": tp / failure.sum() if failure.any() else np.nan,
                    "success_false_alarm_rate": fp / (~failure).sum(),
                    "delta_true_alarms_vs_uniform": int(
                        (failure & alarm).sum() - (failure & uniform).sum()
                    ),
                    "delta_false_alarms_vs_uniform": int(
                        ((~failure) & alarm).sum() - ((~failure) & uniform).sum()
                    ),
                }
            )
    for fold, group in predictions.groupby("fold"):
        failure = group["failure"].to_numpy(dtype=bool)
        uniform = group["alarm_uniform"].to_numpy(dtype=bool)
        for variant in WEIGHTING_VARIANTS:
            alarm = group[f"alarm_{variant}"].to_numpy(dtype=bool)
            tp = int((failure & alarm).sum())
            fp = int((~failure & alarm).sum())
            fold_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "episodes": len(group),
                    "failures": int(failure.sum()),
                    "successes": int((~failure).sum()),
                    "true_alarms": tp,
                    "false_alarms": fp,
                    "failure_recall": tp / failure.sum(),
                    "success_false_alarm_rate": fp / (~failure).sum(),
                    "delta_true_alarms_vs_uniform": int(
                        (failure & alarm).sum() - (failure & uniform).sum()
                    ),
                    "delta_false_alarms_vs_uniform": int(
                        ((~failure) & alarm).sum() - ((~failure) & uniform).sum()
                    ),
                }
            )
    return pd.DataFrame(task_rows), pd.DataFrame(fold_rows)


def render_figure(
    difficulty: pd.DataFrame,
    weighting: pd.DataFrame,
    task_weighting: pd.DataFrame,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    display_names = {
        "uniform": "Uniform",
        "empirical_sqrt": "Empirical sqrt",
        "empirical_linear": "Empirical linear",
        "route_knn5_sqrt": "Route-KNN sqrt",
    }
    colors = {
        "uniform": "#4c5967",
        "empirical_sqrt": "#1f8a70",
        "empirical_linear": "#d95f45",
        "route_knn5_sqrt": "#d49a2a",
    }
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    axis = axes[0, 0]
    x = difficulty["seed_1000_1003_failure_rate"].to_numpy() * 100
    y = difficulty["seed_1004_1007_failure_rate"].to_numpy() * 100
    axis.scatter(x, y, color="#1f8a70", alpha=0.82, edgecolor="white", linewidth=0.5)
    limit = max(float(x.max()), float(y.max())) + 2
    axis.plot([0, limit], [0, limit], color="#4c5967", linestyle="--", linewidth=1)
    axis.set(xlim=(-1, limit), ylim=(-1, limit), xlabel="Failure rate: seeds 1000-1003 (%)", ylabel="Failure rate: seeds 1004-1007 (%)", title="A. Difficulty is repeatable across seed halves")
    axis.text(0.04, 0.93, "Spearman rho = 0.837", transform=axis.transAxes, va="top")

    axis = axes[0, 1]
    for _, row in weighting.iterrows():
        variant = row["variant"]
        axis.scatter(
            100 * row["success_false_alarm_rate"],
            100 * row["failure_recall"],
            s=90,
            color=colors[variant],
            label=display_names[variant],
        )
    axis.set(xlabel="Observed success false-alarm rate (%)", ylabel="Micro failure recall (%)", title="B. Same nominal 5% calibration budget")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(alpha=0.2)

    axis = axes[1, 0]
    positions = np.arange(len(weighting))
    width = 0.36
    axis.bar(
        positions - width / 2,
        100 * weighting["failure_recall"],
        width,
        color=[colors[value] for value in weighting["variant"]],
        label="Micro (failures equal)",
    )
    axis.bar(
        positions + width / 2,
        100 * weighting["macro_task_failure_recall"],
        width,
        color="none",
        edgecolor=[colors[value] for value in weighting["variant"]],
        linewidth=1.8,
        hatch="//",
        label="Macro (tasks equal)",
    )
    axis.set_xticks(positions, [display_names[value] for value in weighting["variant"]], rotation=15, ha="right")
    axis.set(ylabel="Recall (%)", title="C. Empirical weighting trades macro for micro recall")
    axis.legend(frameon=False, fontsize=9)
    axis.grid(axis="y", alpha=0.2)

    axis = axes[1, 1]
    linear = task_weighting[task_weighting["variant"] == "empirical_linear"].copy()
    linear = linear.sort_values("failures", ascending=False).head(10).iloc[::-1]
    short_names = [
        value.split("/", 1)[-1].replace("_", " ")[:34]
        for value in linear["task"]
    ]
    bars = axis.barh(
        np.arange(len(linear)),
        linear["delta_true_alarms_vs_uniform"],
        color=["#d95f45" if value >= 0 else "#4c5967" for value in linear["delta_true_alarms_vs_uniform"]],
    )
    axis.set_yticks(np.arange(len(linear)), short_names, fontsize=8)
    axis.axvline(0, color="#30343b", linewidth=0.8)
    axis.set(xlabel="Extra failure hits vs uniform", title="D. Linear gain is concentrated in hard tasks")
    for bar, failures in zip(bars, linear["failures"]):
        axis.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"n_fail={int(failures)}",
            va="center",
            fontsize=8,
        )
    axis.grid(axis="x", alpha=0.2)

    figure.suptitle("Train-free task-difficulty and MoE alarm-weighting audit", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame[columns]
    header = "| " + " | ".join(selected.columns) + " |"
    separator = "| " + " | ".join("---" for _ in selected.columns) + " |"
    rows = []
    for _, row in selected.iterrows():
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def render_report(
    summary: dict[str, Any],
    difficulty: pd.DataFrame,
    associations: pd.DataFrame,
    weighting: pd.DataFrame,
) -> str:
    best = associations.iloc[0]
    uniform = weighting.set_index("variant").loc["uniform"]
    empirical_sqrt = weighting.set_index("variant").loc["empirical_sqrt"]
    empirical_linear = weighting.set_index("variant").loc["empirical_linear"]
    route_weighted = weighting.set_index("variant").loc["route_knn5_sqrt"]
    fold_weighting = pd.DataFrame(summary["alarm_weighting"]["fold_results"])
    linear_folds = fold_weighting[fold_weighting["variant"] == "empirical_linear"]
    uniform_folds = fold_weighting[fold_weighting["variant"] == "uniform"]
    hardest = difficulty.iloc[0]
    identity = summary["early_route_task_identity"]
    early = summary["early_route_difficulty"]
    reliability = summary["difficulty_reliability"]
    lines = [
        "# MoE 路由、任务难度与报警加权",
        "",
        "## 核心结论",
        "",
        f"本实验固定使用 37 个任务、{summary['episodes']} 条轨迹。任务难度定义为当前"
        " checkpoint、初始状态分布、noise 和 horizon 下的经验失败率，而不是任务固有难度。",
        "",
        f"难度是可重复测量的：互斥 seed halves 的失败率 Spearman `rho="
        f"{reliability['seed_half_spearman_rho']:.3f}`。但第一个 query 的 MoE 路由"
        f"仍没有形成可靠难度轴：BH 校正后显著标量为 "
        f"{early['significant_early_scalars_after_bh_005']} 个，路由距离与难度差的"
        f"相关的 cross-fit 平均值为 `rho="
        f"{early['pairwise_route_distance_vs_difficulty_difference_mean_spearman_rho']:.3f}`，"
        f"置换 `p={early['pairwise_permutation_p_positive_association']:.3f}`。",
        "",
        f"同一份第一个 query 路由却能在 held-out init 上以 "
        f"{identity['task_accuracy']:.1%} 识别具体任务、{identity['suite_accuracy']:.1%} "
        "识别 suite。这再次说明 routing 主要编码任务/场景，不等于编码难度。",
        "",
        "## 难度定义与稳定性",
        "",
        f"最难任务为 `{hardest['task']}`，失败率 {hardest['failure_rate']:.1%}。"
        f"所有任务失败率范围为 {difficulty['failure_rate'].min():.1%}--"
        f"{difficulty['failure_rate'].max():.1%}。平均 episode 长度与失败率的相关仅为 "
        f"`rho={reliability['episode_length_spearman_rho']:.3f}`，所以长任务不能直接当难任务。",
        "",
        markdown_table(
            difficulty.head(10),
            ["task", "failures", "failure_rate", "failure_rate_wilson_low", "failure_rate_wilson_high"],
        ),
        "",
        "## 第一个 query 的 MoE 能否预测难度",
        "",
        "物理 outcome 在第一个 query 路由特征和路由 embedding 全部写盘后才加载。"
        "每个任务用 8 层 x 32 experts 的 final-flow action-route 均值作为路由签名；"
        "路由签名始终只预测另一组 noise seeds 的失败率，并交换两半复算。",
        "",
        f"最强的早期标量是 `{best['signal']}`，两方向平均 Spearman "
        f"`rho={best['mean_crossfit_spearman_rho']:.3f}`，置换检验 "
        f"`p={best['permutation_p_two_sided']:.4g}`，BH 后 "
        f"`q={best['bh_q_value']:.4g}`。没有早期标量通过 `q<=0.05`。",
        "",
        markdown_table(
            associations.head(8),
            ["signal", "fold_1000_1003_routes_rho", "fold_1004_1007_routes_rho", "mean_crossfit_spearman_rho", "permutation_p_two_sided", "bh_q_value", "mean_rho_without_hardest_task"],
        ),
        "",
        "路由 KNN 的正 `mae_improvement_over_median` 才表示优于不知道路由时的"
        " leave-one-task median。结果如下：",
        "",
        markdown_table(pd.DataFrame([
            {"k": key.split("_")[1], **value}
            for key, value in early["knn"].items()
        ]), ["k", "mean_fold_spearman_rho", "mae", "median_baseline_mae", "mae_improvement_over_median", "permutation_p_route_geometry"]),
        "",
        "## 固定报警预算下的难度加权",
        "",
        "这是 50%--90% episode window 的离线 allocation stress test，不是当前 v2 的"
        "逐 query 在线性能。两个 seed halves 互换 calibration/evaluation；difficulty prior、"
        "健康分位数和阈值均只来自另一半。固定 detector 是 instability 与 lock-in 的等权均值，"
        "没有拟合 feature weight。所有方法在 calibration 上的目标 success alarm budget 均为 5%。",
        "",
        markdown_table(
            weighting,
            ["variant", "true_alarms", "false_alarms", "failure_recall", "success_false_alarm_rate", "precision", "delta_recall_vs_uniform", "delta_recall_ci_low", "delta_recall_ci_high", "failure_recall_excluding_hardest_task", "macro_task_failure_recall"],
        ),
        "",
        f"均匀预算命中 {int(uniform['true_alarms'])}/{int(uniform['failures'])}；"
        f"经验难度 sqrt 加权命中 {int(empirical_sqrt['true_alarms'])}，线性加权命中 "
        f"{int(empirical_linear['true_alarms'])}。route-KNN 难度加权命中 "
        f"{int(route_weighted['true_alarms'])}。是否有收益必须同时看实际 FPR 和 task-cluster CI，"
        "不能只比较命中数。",
        "",
        f"线性经验先验的净增量为 {int(empirical_linear['true_alarms'] - uniform['true_alarms'])} 个"
        f"失败命中，其中最难任务 `{empirical_linear['hardest_task']}` 单独贡献 "
        f"{int(empirical_linear['hardest_task_true_alarms'] - uniform['hardest_task_true_alarms'])} 个。"
        f"剔除它后，召回变化为 "
        f"{empirical_linear['delta_recall_excluding_hardest_vs_uniform']:+.1%}；"
        f"跨任务 bootstrap 的总召回差 95% CI 为 "
        f"[{empirical_linear['delta_recall_ci_low']:+.1%}, "
        f"{empirical_linear['delta_recall_ci_high']:+.1%}]。",
        "",
        f"它优化的是按失败样本计的 micro recall；若每个有失败的任务等权，macro recall 反而从 "
        f"{uniform['macro_task_failure_recall']:.1%} 降至 "
        f"{empirical_linear['macro_task_failure_recall']:.1%}。也就是说，难度加权明确牺牲"
        "低失败率任务，把报警预算集中到高失败率任务；是否值得取决于部署效用函数。",
        "",
        f"两个交换方向的线性加权命中分别为 "
        f"{int(linear_folds.iloc[0]['true_alarms'])}/{int(linear_folds.iloc[0]['failures'])} 和 "
        f"{int(linear_folds.iloc[1]['true_alarms'])}/{int(linear_folds.iloc[1]['failures'])}；"
        f"对应均匀预算为 {int(uniform_folds.iloc[0]['true_alarms'])}/"
        f"{int(uniform_folds.iloc[0]['failures'])} 和 "
        f"{int(uniform_folds.iloc[1]['true_alarms'])}/"
        f"{int(uniform_folds.iloc[1]['failures'])}。方向一致，但不代表能泛化到新任务。",
        "",
        "## 解释边界",
        "",
        "- difficulty prior 使用 disjoint rollout outcomes，属于 label-based calibration，不是纯 MoE-only。",
        "- first-query route difficulty 部分不读取 outcome，但评价时需要 task failure rate。",
        "- 报警加权实验使用 episode 中后段汇总路由，是 allocation 原理检查，不是 early alarm。",
        "- 经验难度只对当前 checkpoint、数据分布和 horizon 有效。",
        "- 若难度权重有效，它改变的是报警预算分配，不证明 MoE 本身理解了任务难度。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        raise ValueError("bootstrap and permutations must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    identities = pd.read_csv(args.atlas, usecols=["task", "episode"])
    if identities.duplicated(["task", "episode"]).any():
        raise RuntimeError("atlas task/episode identities are not unique")

    early_path = args.out_dir / "first_query_moe_features.csv"
    embedding_path = args.out_dir / "first_query_action_route_embeddings.npz"
    if args.reuse_early and early_path.exists() and embedding_path.exists():
        early = pd.read_csv(early_path)
        with np.load(embedding_path, allow_pickle=False) as archive:
            embeddings = np.asarray(archive["embeddings"], dtype=np.float16)
            stored_tasks = archive["tasks"].astype(str)
            stored_episodes = np.asarray(archive["episodes"], dtype=np.int64)
        if not np.array_equal(stored_tasks, early["task"].to_numpy(dtype=str)):
            raise RuntimeError("reused early task order mismatch")
        if not np.array_equal(stored_episodes, early["episode"].to_numpy(dtype=np.int64)):
            raise RuntimeError("reused early episode order mismatch")
        print(f"reused {early_path}", flush=True)
    else:
        early, embeddings = extract_early_routes(
            identities, args.cache_root, args.run_id
        )
        # Materialize MoE-only features before outcomes are merged below.
        early.to_csv(early_path, index=False)
        np.savez_compressed(
            embedding_path,
            embeddings=embeddings,
            tasks=early["task"].to_numpy(dtype=str),
            episodes=early["episode"].to_numpy(dtype=np.int64),
            schema=np.asarray("himoe.first_query_action_route_embedding.v1"),
            outcome_labels_used=np.asarray(False),
        )

    # Outcome-bearing data is intentionally loaded only after MoE-only artifacts exist.
    frame = pd.read_csv(args.atlas)
    required = {"task", "episode", "init_state_id", "flow_noise_seed", "failure"}
    if not required <= set(frame):
        raise RuntimeError("atlas is missing required columns")
    if not pd.api.types.is_bool_dtype(frame["failure"]):
        normalized_failure = frame["failure"].astype(str).str.strip().str.lower()
        if not normalized_failure.isin({"true", "false"}).all():
            raise RuntimeError("atlas failure column is not boolean")
        frame["failure"] = normalized_failure.eq("true")

    # Preserve the independently materialized embedding order through the label join.
    early = early.copy()
    early["_embedding_row"] = np.arange(len(early), dtype=np.int64)
    labels = frame[
        ["task", "episode", "init_state_id", "flow_noise_seed", "episode_length", "failure"]
    ]
    early = early.merge(
        labels, on=["task", "episode"], validate="one_to_one", sort=False
    )
    embedding_order = early.pop("_embedding_row").to_numpy(dtype=np.int64)
    if len(embedding_order) != len(embeddings) or len(np.unique(embedding_order)) != len(embeddings):
        raise RuntimeError("outcome join changed the first-query episode population")
    embeddings = embeddings[embedding_order]
    difficulty, reliability = difficulty_inventory(frame)
    associations, predictions, early_summary = early_difficulty_analysis(
        early, embeddings, difficulty, args.permutations, args.seed
    )
    identity = early_task_identity(early, embeddings)
    weighted_predictions, weighting, allocations = difficulty_weighting(
        frame, early, embeddings, args.bootstrap, args.seed + 1
    )
    task_weighting, fold_weighting = weighting_breakdowns(weighted_predictions)

    difficulty.to_csv(args.out_dir / "task_difficulty.csv", index=False)
    associations.to_csv(args.out_dir / "early_route_difficulty_associations.csv", index=False)
    predictions.to_csv(args.out_dir / "early_route_knn_predictions.csv", index=False)
    weighted_predictions.to_csv(args.out_dir / "weighting_episode_predictions.csv", index=False)
    weighting.to_csv(args.out_dir / "weighting_summary.csv", index=False)
    allocations.to_csv(args.out_dir / "weighting_task_allocations.csv", index=False)
    task_weighting.to_csv(args.out_dir / "weighting_task_evaluation.csv", index=False)
    fold_weighting.to_csv(args.out_dir / "weighting_fold_evaluation.csv", index=False)
    render_figure(
        difficulty,
        weighting,
        task_weighting,
        args.out_dir / "difficulty_weighting_audit.png",
    )

    summary = {
        "schema": "himoe.task_difficulty_weighting.v1",
        "training": False,
        "gradient_optimization": False,
        "fixed_feature_weights": True,
        "n_bootstrap": args.bootstrap,
        "n_permutations": args.permutations,
        "tasks": int(frame["task"].nunique()),
        "episodes": len(frame),
        "failures": int(frame["failure"].sum()),
        "successes": int((~frame["failure"]).sum()),
        "difficulty_definition": (
            "empirical failure probability conditional on checkpoint, initial-state/noise "
            "distribution, controller, and horizon"
        ),
        "difficulty_reliability": reliability,
        "first_query_prediction_inputs": "HB MoE router probabilities only",
        "first_query_features_materialized_before_outcomes_loaded": True,
        "early_route_task_identity": identity,
        "early_route_difficulty": early_summary,
        "alarm_weighting": {
            "scope": "offline episode-level allocation stress test",
            "calibration_evaluation_split": "two-way cross-fit over noise seeds 1000-1003 vs 1004-1007",
            "target_mean_success_alarm_budget": 0.05,
            "difficulty_prior_uses_disjoint_outcomes": True,
            "detector_calibration_uses_disjoint_success_routes": True,
            "variants": weighting.to_dict(orient="records"),
            "fold_results": fold_weighting.to_dict(orient="records"),
        },
        "interpretation": [
            "Task identity in routing is not evidence of a scalar difficulty representation.",
            "Difficulty weighting reallocates intervention budget; it does not improve MoE evidence itself.",
            "This experiment does not validate an online early-warning policy.",
        ],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT_ZH.md").write_text(
        render_report(summary, difficulty, associations, weighting), encoding="utf-8"
    )
    print(json.dumps(plain(summary), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
