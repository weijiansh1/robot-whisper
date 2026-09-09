#!/usr/bin/env python3
"""Compare label-free noise selectors under task- and seed-disjoint evaluation.

Heads are trained only from initial noise, routing, and the model's own final
action output.  Success is isolated in the evaluation stage.  Each test pool's
complete task and its eight noise seeds are absent from head fitting.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr
from scipy.stats import rankdata

from route_noise_selector import (
    N_EXPERTS,
    centrality,
    early_route_descriptors,
    late_route_centrality_target,
    noise_descriptors,
    normalize_router_probabilities,
    pairwise_hellinger,
    pairwise_rms,
    pool_standardize,
    route_centrality_scores,
)


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis/self-supervised-noise-methods"
FLOW_NOISE_SHAPE = (10, 24)
PREFIX_STEPS = 3
SEED_FOLDS = 4
ALPHAS = (0.1, 1.0, 10.0, 100.0)

BASELINES = (
    "early_route_central",
    "initial_noise_central",
    "old_future_route_head",
)
IMPROVED = (
    "action_central_head",
    "residual_action_central_head",
    "residual_future_change_head",
    "residual_action_vector_head",
    "residual_route_central",
    "prefix_contraction",
    "cross_layer_agreement",
    "selfsup_ensemble",
)
ORACLES = (
    "final_action_central_oracle",
    "future_route_change_oracle",
    "late_route_central_oracle",
)
METHODS = BASELINES + IMPROVED + ORACLES
FWER_METHODS = BASELINES + IMPROVED

TARGET_BY_METHOD = {
    "old_future_route_head": "late_route_centrality",
    "action_central_head": "action_centrality",
    "residual_action_central_head": "action_centrality",
    "residual_future_change_head": "future_route_change",
    "residual_action_vector_head": "action_centrality",
    "selfsup_ensemble": "action_centrality",
}


@dataclass(frozen=True)
class Pool:
    task: str
    suite: str
    state: int
    fold: int
    seeds: np.ndarray
    route_features: np.ndarray
    noise_features: np.ndarray
    action_residual: np.ndarray
    action_centrality: np.ndarray
    late_route_centrality: np.ndarray
    future_route_change: np.ndarray
    early_route_score: np.ndarray
    initial_noise_score: np.ndarray
    prefix_contraction: np.ndarray
    cross_layer_agreement: np.ndarray
    success: np.ndarray

    def target(self, name: str) -> np.ndarray:
        if name == "action_centrality":
            return self.action_centrality
        if name == "late_route_centrality":
            return self.late_route_centrality
        if name == "future_route_change":
            return self.future_route_change
        if name == "action_residual":
            return self.action_residual
        raise KeyError(name)


@dataclass(frozen=True)
class RidgeModel:
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, features: np.ndarray, target: np.ndarray, alpha: float) -> "RidgeModel":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.shape[0] != len(x) or len(x) < 2:
            raise ValueError("ridge features and target are not aligned")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or alpha <= 0:
            raise ValueError("ridge inputs must be finite and alpha positive")
        mean = x.mean(axis=0)
        scale = np.maximum(x.std(axis=0), 1e-12)
        standardized = (x - mean) / scale
        design = np.column_stack([np.ones(len(x)), standardized])
        penalty = alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ y,
        )
        return cls(coefficients, mean, scale, float(alpha))

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("ridge prediction features have the wrong shape")
        standardized = (values - self.feature_mean) / self.feature_scale
        return np.column_stack([np.ones(len(values)), standardized]) @ self.coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--render-existing",
        action="store_true",
        help="repair aggregates and render an existing summary without recomputing",
    )
    return parser.parse_args()


def discover_runs(hub_root: Path) -> list[Path]:
    cache = hub_root / "cache/HiMoE-VLA"
    runs = sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/server/routes.zarr")
        if (path.parent.parent / "client/summaries.json").exists()
    )
    if len(runs) != 5:
        raise RuntimeError("expected five complete right-16x32 runs, found %d" % len(runs))
    return runs


def first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    if set(map(int, expected)) != set(lookup):
        raise RuntimeError("route episode ids do not match summaries")
    return np.asarray([lookup[int(episode)] for episode in expected], dtype=np.int64)


def reconstruct_initial_noise(seeds: Iterable[int]) -> np.ndarray:
    return np.stack(
        [
            np.random.default_rng(int(seed))
            .standard_normal(FLOW_NOISE_SHAPE)
            .astype(np.float32)
            for seed in seeds
        ]
    )


def normalized_ranks(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise ValueError("rank target must be a finite vector")
    return (rankdata(scores, method="average") - 1.0) / max(len(scores) - 1, 1)


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = normalized_ranks(left)
    b = normalized_ranks(right)
    a -= a.mean()
    b -= b.mean()
    scale = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / scale) if scale > 1e-20 else float("nan")


def action_target(actions: np.ndarray, action_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.asarray(actions, dtype=np.float64) / action_std[None, None, :]
    flat = normalized.reshape(len(normalized), -1)
    centered = flat - flat.mean(axis=0, keepdims=True)
    scale = math.sqrt(float(np.mean(np.square(centered))))
    residual = centered / max(scale, 1e-12)
    score = centrality(pairwise_rms(normalized))
    return residual, normalized_ranks(score)


def future_route_change_target(routes: np.ndarray, prefix_steps: int) -> np.ndarray:
    values = normalize_router_probabilities(routes)
    root = np.sqrt(values)
    difference = root[:, :, prefix_steps:] - root[:, :, prefix_steps - 1 : -1]
    site_h2 = 0.5 * np.square(difference).sum(axis=-1)
    remaining_motion = np.sqrt(site_h2.mean(axis=(1, 2, 3)))
    endpoint_h2 = 0.5 * np.square(
        root[:, :, -1] - root[:, :, prefix_steps - 1]
    ).sum(axis=-1)
    endpoint = np.sqrt(endpoint_h2.mean(axis=(1, 2)))
    return normalized_ranks(0.5 * remaining_motion + 0.5 * endpoint)


def prefix_diagnostics(routes: np.ndarray, prefix_steps: int) -> tuple[np.ndarray, np.ndarray]:
    values = normalize_router_probabilities(routes[:, :, :prefix_steps])
    root = np.sqrt(values)
    site_h2 = 0.5 * np.square(root[:, :, 1:] - root[:, :, :-1]).sum(axis=-1)
    movement = np.sqrt(site_h2.mean(axis=(1, 3)))
    contraction = np.log((movement[:, 1] + 1e-8) / (movement[:, 0] + 1e-8))

    per_layer_rank = np.empty((len(values), values.shape[1]), dtype=np.float64)
    for layer in range(values.shape[1]):
        score = centrality(pairwise_hellinger(values[:, layer]))
        per_layer_rank[:, layer] = normalized_ranks(score)
    agreement = per_layer_rank.std(axis=1)
    return contraction, agreement


def load_task_pools(run: Path, cache_root: Path) -> tuple[list[Pool], dict[str, Any]]:
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 16 * 32:
        raise RuntimeError("%s is not a complete 16x32 run" % run)
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    states = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    success = np.asarray([bool(row["success"]) for row in summaries])

    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:], dtype=np.int64)
    rows = first_rows(route_episode, episodes)
    raw = np.asarray(
        route_group["hb_router_probs"].oindex[rows, :, :, :, :],
        dtype=np.float64,
    )
    if raw.shape != (512, 8, 10, 11, N_EXPERTS):
        raise RuntimeError("unexpected route shape %s" % (raw.shape,))
    raw_error = float(np.max(np.abs(raw.sum(axis=-1) - 1.0)))
    routes = normalize_router_probabilities(raw)[:, :, :, 1:, :]
    normalized_error = float(np.max(np.abs(routes.sum(axis=-1) - 1.0)))

    metadata = json.loads((run / "client/server_metadata.json").read_text())
    stats = json.loads(Path(metadata["normalization_stats_path"]).read_text())
    action_std = np.asarray(stats["actions"]["std"], dtype=np.float64)
    if action_std.shape != (7,) or np.any(action_std <= 0):
        raise RuntimeError("invalid action normalization stats")
    actions = []
    for episode in episodes:
        with np.load(run / "client" / ("episode_%02d.npz" % int(episode))) as record:
            actions.append(np.asarray(record["actions"][0], dtype=np.float64))
    actions = np.stack(actions)
    if actions.shape != (512, 10, 7):
        raise RuntimeError("unexpected first action shape %s" % (actions.shape,))

    noise = reconstruct_initial_noise(seeds)
    relative = run.relative_to(cache_root)
    task = str(relative.parent)
    suite = relative.parts[0]
    pools = []
    for state in np.unique(states):
        state_rows = np.flatnonzero(states == state)
        state_rows = state_rows[np.argsort(seeds[state_rows])]
        if len(state_rows) != 32:
            raise RuntimeError("state does not have 32 candidates")
        for fold in range(SEED_FOLDS):
            selected = state_rows[np.arange(32)[fold::SEED_FOLDS]]
            candidate_routes = routes[selected]
            route_feature = pool_standardize(
                early_route_descriptors(candidate_routes, PREFIX_STEPS)
            )
            noise_feature = pool_standardize(noise_descriptors(noise[selected]))
            action_residual, action_centrality = action_target(
                actions[selected], action_std
            )
            contraction, agreement = prefix_diagnostics(
                candidate_routes, PREFIX_STEPS
            )
            pools.append(
                Pool(
                    task=task,
                    suite=suite,
                    state=int(state),
                    fold=fold,
                    seeds=seeds[selected].copy(),
                    route_features=route_feature,
                    noise_features=noise_feature,
                    action_residual=action_residual,
                    action_centrality=action_centrality,
                    late_route_centrality=late_route_centrality_target(
                        candidate_routes, PREFIX_STEPS
                    ),
                    future_route_change=future_route_change_target(
                        candidate_routes, PREFIX_STEPS
                    ),
                    early_route_score=route_centrality_scores(
                        candidate_routes[:, :, :PREFIX_STEPS]
                    ),
                    initial_noise_score=centrality(
                        pairwise_rms(noise[selected, :, :7])
                    ),
                    prefix_contraction=contraction,
                    cross_layer_agreement=agreement,
                    success=success[selected].copy(),
                )
            )
    return pools, {
        "task": task,
        "success_rate": float(success.mean()),
        "raw_probability_mass_max_error": raw_error,
        "normalized_probability_mass_max_error": normalized_error,
    }


def stack_field(pools: list[Pool], field: str) -> np.ndarray:
    if field == "route":
        return np.concatenate([pool.route_features for pool in pools])
    if field == "noise":
        return np.concatenate([pool.noise_features for pool in pools])
    if field == "route_noise":
        return np.concatenate(
            [
                np.concatenate([pool.route_features, pool.noise_features], axis=1)
                for pool in pools
            ]
        )
    return np.concatenate([pool.target(field) for pool in pools])


def tune_residual_alpha(pools: list[Pool]) -> tuple[float, dict[str, float]]:
    tasks = sorted({pool.task for pool in pools})
    losses = {alpha: 0.0 for alpha in ALPHAS}
    counts = {alpha: 0 for alpha in ALPHAS}
    for validation_task in tasks:
        train = [pool for pool in pools if pool.task != validation_task]
        validation = [pool for pool in pools if pool.task == validation_task]
        train_noise = stack_field(train, "noise")
        train_route = stack_field(train, "route")
        valid_noise = stack_field(validation, "noise")
        valid_route = stack_field(validation, "route")
        for alpha in ALPHAS:
            prediction = RidgeModel.fit(train_noise, train_route, alpha).predict(
                valid_noise
            )
            losses[alpha] += float(np.square(prediction - valid_route).sum())
            counts[alpha] += int(np.prod(valid_route.shape))
    mse = {str(alpha): losses[alpha] / counts[alpha] for alpha in ALPHAS}
    selected = min(ALPHAS, key=lambda value: (mse[str(value)], value))
    return float(selected), mse


def residual_features(
    train: list[Pool],
    test: list[Pool],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, RidgeModel]:
    model = RidgeModel.fit(
        stack_field(train, "noise"),
        stack_field(train, "route"),
        alpha,
    )
    train_route = stack_field(train, "route")
    test_route = stack_field(test, "route")
    return (
        train_route - model.predict(stack_field(train, "noise")),
        test_route - model.predict(stack_field(test, "noise")),
        model,
    )


def tune_head_alpha(
    pools: list[Pool],
    target: str,
    feature_kind: str,
    residual_alpha: float,
) -> tuple[float, dict[str, float]]:
    tasks = sorted({pool.task for pool in pools})
    losses = {alpha: 0.0 for alpha in ALPHAS}
    counts = {alpha: 0 for alpha in ALPHAS}
    for validation_task in tasks:
        train = [pool for pool in pools if pool.task != validation_task]
        validation = [pool for pool in pools if pool.task == validation_task]
        if feature_kind == "residual":
            train_x, valid_x, _ = residual_features(
                train, validation, residual_alpha
            )
        else:
            train_x = stack_field(train, feature_kind)
            valid_x = stack_field(validation, feature_kind)
        train_y = stack_field(train, target)
        valid_y = stack_field(validation, target)
        for alpha in ALPHAS:
            prediction = RidgeModel.fit(train_x, train_y, alpha).predict(valid_x)
            losses[alpha] += float(np.square(prediction - valid_y).sum())
            counts[alpha] += int(np.prod(valid_y.shape))
    mse = {str(alpha): losses[alpha] / counts[alpha] for alpha in ALPHAS}
    selected = min(ALPHAS, key=lambda value: (mse[str(value)], value))
    return float(selected), mse


def split_rows(values: np.ndarray, pools: list[Pool]) -> list[np.ndarray]:
    output = []
    cursor = 0
    for pool in pools:
        stop = cursor + len(pool.seeds)
        output.append(values[cursor:stop])
        cursor = stop
    if cursor != len(values):
        raise RuntimeError("row split did not consume all values")
    return output


def score_outer_fold(
    train: list[Pool], test: list[Pool]
) -> tuple[dict[str, list[np.ndarray]], dict[str, Any]]:
    residual_alpha, residual_mse = tune_residual_alpha(train)
    train_residual, test_residual, _ = residual_features(
        train, test, residual_alpha
    )
    train_features = {
        "route": stack_field(train, "route"),
        "route_noise": stack_field(train, "route_noise"),
        "residual": train_residual,
    }
    test_features = {
        "route": stack_field(test, "route"),
        "route_noise": stack_field(test, "route_noise"),
        "residual": test_residual,
    }
    specifications = {
        "old_future_route_head": ("late_route_centrality", "route_noise"),
        "action_central_head": ("action_centrality", "route"),
        "residual_action_central_head": ("action_centrality", "residual"),
        "residual_future_change_head": ("future_route_change", "residual"),
        "residual_action_vector_head": ("action_residual", "residual"),
    }
    row_scores: dict[str, np.ndarray] = {}
    head_diagnostics = {}
    for method, (target, feature_kind) in specifications.items():
        alpha, mse = tune_head_alpha(
            train, target, feature_kind, residual_alpha
        )
        model = RidgeModel.fit(
            train_features[feature_kind], stack_field(train, target), alpha
        )
        prediction = model.predict(test_features[feature_kind])
        if method == "residual_action_vector_head":
            scores = []
            for values in split_rows(prediction, test):
                scores.append(centrality(pairwise_rms(values)))
            row_scores[method] = np.concatenate(scores)
        else:
            row_scores[method] = np.asarray(prediction, dtype=np.float64)
        head_diagnostics[method] = {
            "alpha": alpha,
            "source_task_mse": mse,
        }

    residual_central = []
    for values in split_rows(test_residual, test):
        residual_central.append(centrality(pairwise_rms(values)))
    row_scores["residual_route_central"] = np.concatenate(residual_central)

    per_pool: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    learned_split = {
        method: split_rows(values, test) for method, values in row_scores.items()
    }
    for index, pool in enumerate(test):
        per_pool["early_route_central"].append(pool.early_route_score)
        per_pool["initial_noise_central"].append(pool.initial_noise_score)
        per_pool["prefix_contraction"].append(pool.prefix_contraction)
        per_pool["cross_layer_agreement"].append(pool.cross_layer_agreement)
        per_pool["final_action_central_oracle"].append(pool.action_centrality)
        per_pool["future_route_change_oracle"].append(pool.future_route_change)
        per_pool["late_route_central_oracle"].append(pool.late_route_centrality)
        for method in learned_split:
            per_pool[method].append(learned_split[method][index])
        ensemble_parts = [
            learned_split["residual_action_central_head"][index],
            learned_split["residual_future_change_head"][index],
            learned_split["residual_action_vector_head"][index],
            learned_split["residual_route_central"][index],
            pool.cross_layer_agreement,
        ]
        per_pool["selfsup_ensemble"].append(
            np.mean([normalized_ranks(values) for values in ensemble_parts], axis=0)
        )
    return per_pool, {
        "residualizer_alpha": residual_alpha,
        "residualizer_source_task_mse": residual_mse,
        "heads": head_diagnostics,
    }


def pool_auc(success: np.ndarray, score: np.ndarray) -> float:
    positive = -np.asarray(score)[np.asarray(success, dtype=bool)]
    negative = -np.asarray(score)[~np.asarray(success, dtype=bool)]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def select_index(scores: np.ndarray, seeds: np.ndarray) -> int:
    values = np.asarray(scores, dtype=np.float64)
    minimum = values.min()
    tied = np.flatnonzero(np.isclose(values, minimum, atol=1e-12, rtol=0.0))
    return int(tied[np.argmin(seeds[tied])])


def evaluate_task(
    task: str, all_pools: list[Pool]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    test_pools = sorted(
        [pool for pool in all_pools if pool.task == task],
        key=lambda pool: (pool.fold, pool.state),
    )
    if len(test_pools) != 64:
        raise RuntimeError("held task must have 64 K8 pools")
    scores: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    ordered_pools: list[Pool] = []
    diagnostics = []
    for fold in range(SEED_FOLDS):
        fold_test = [pool for pool in test_pools if pool.fold == fold]
        train = [
            pool for pool in all_pools if pool.task != task and pool.fold != fold
        ]
        test_seeds = {int(seed) for pool in fold_test for seed in pool.seeds}
        train_seeds = {int(seed) for pool in train for seed in pool.seeds}
        if test_seeds & train_seeds:
            raise RuntimeError("test seed leakage")
        fold_scores, diagnostic = score_outer_fold(train, fold_test)
        diagnostics.append({"held_fold": fold, **diagnostic})
        ordered_pools.extend(fold_test)
        for method in METHODS:
            scores[method].extend(fold_scores[method])

    result = {}
    selected_positions: dict[str, np.ndarray] = {}
    for method in METHODS:
        selected_index = np.asarray(
            [
                select_index(score, pool.seeds)
                for score, pool in zip(scores[method], ordered_pools)
            ],
            dtype=np.int64,
        )
        selected_positions[method] = selected_index
        selected = np.asarray(
            [
                pool.success[index]
                for pool, index in zip(ordered_pools, selected_index)
            ],
            dtype=np.float64,
        )
        baseline = np.asarray(
            [pool.success.mean() for pool in ordered_pools], dtype=np.float64
        )
        auc = [
            pool_auc(pool.success, score)
            for pool, score in zip(ordered_pools, scores[method])
        ]
        pseudo_rho = []
        target = TARGET_BY_METHOD.get(method)
        if target is not None:
            for pool, score in zip(ordered_pools, scores[method]):
                rho = rank_correlation(score, pool.target(target))
                if np.isfinite(rho):
                    pseudo_rho.append(rho)
        selected_seeds = np.asarray(
            [pool.seeds[index] for pool, index in zip(ordered_pools, selected_index)]
        )
        counts = Counter(map(int, selected_seeds))
        per_fold_unique = []
        for fold in range(SEED_FOLDS):
            indices = [i for i, pool in enumerate(ordered_pools) if pool.fold == fold]
            per_fold_unique.append(len(set(map(int, selected_seeds[indices]))))
        available_auc = [value for value in auc if np.isfinite(value)]
        result[method] = {
            "selected_success": float(selected.mean()),
            "random_expected_success": float(baseline.mean()),
            "delta_vs_random": float((selected - baseline).mean()),
            "mean_pool_auc": (
                float(np.mean(available_auc)) if available_auc else None
            ),
            "evaluable_auc_pools": len(available_auc),
            "pseudo_target_spearman": (
                float(np.mean(pseudo_rho)) if pseudo_rho else None
            ),
            "unique_selected_seeds": len(counts),
            "max_seed_share": max(counts.values()) / len(selected_seeds),
            "mean_unique_seeds_per_fold_across_16_states": float(
                np.mean(per_fold_unique)
            ),
        }
    arrays = {
        "baseline": np.asarray([pool.success.mean() for pool in ordered_pools]),
        "success": np.stack([pool.success for pool in ordered_pools]),
        "states": np.asarray([pool.state for pool in ordered_pools]),
        "folds": np.asarray([pool.fold for pool in ordered_pools]),
        **selected_positions,
    }
    return result, {"outer_folds": diagnostics}, arrays


def bootstrap_macro(
    task_arrays: dict[str, dict[str, np.ndarray]],
    method: str,
    draws: int,
    rng: np.random.Generator,
) -> list[float]:
    tasks = sorted(task_arrays)
    distribution = np.empty(draws)
    for draw in range(draws):
        sampled_tasks = rng.integers(0, len(tasks), len(tasks))
        task_delta = []
        for task_axis in sampled_tasks:
            arrays = task_arrays[tasks[task_axis]]
            unique_states = np.unique(arrays["states"])
            sampled_states = rng.choice(unique_states, len(unique_states), replace=True)
            state_delta = []
            for state in sampled_states:
                rows = np.flatnonzero(arrays["states"] == state)
                picked = arrays["success"][rows, arrays[method][rows]]
                state_delta.append(float((picked - arrays["baseline"][rows]).mean()))
            task_delta.append(float(np.mean(state_delta)))
        distribution[draw] = np.mean(task_delta)
    return [float(value) for value in np.percentile(distribution, [2.5, 97.5])]


def permutation_tests(
    task_arrays: dict[str, dict[str, np.ndarray]],
    observed: dict[str, float],
    draws: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    tasks = sorted(task_arrays)
    null = {method: np.empty(draws) for method in FWER_METHODS}
    for draw in range(draws):
        by_method = {method: [] for method in FWER_METHODS}
        for task in tasks:
            arrays = task_arrays[task]
            labels = arrays["success"].copy()
            for fold in range(SEED_FOLDS):
                rows = np.flatnonzero(arrays["folds"] == fold)
                labels[rows] = labels[rows][:, rng.permutation(labels.shape[1])]
            for method in FWER_METHODS:
                picked = labels[np.arange(len(labels)), arrays[method]]
                by_method[method].append(float((picked - arrays["baseline"]).mean()))
        for method in FWER_METHODS:
            null[method][draw] = np.mean(by_method[method])
    maximum = np.column_stack([null[method] for method in FWER_METHODS]).max(axis=1)
    return {
        method: {
            "permutation_p": float(
                (1 + np.count_nonzero(null[method] >= observed[method] - 1e-15))
                / (draws + 1)
            ),
            "permutation_p_fwer": float(
                (1 + np.count_nonzero(maximum >= observed[method] - 1e-15))
                / (draws + 1)
            ),
        }
        for method in FWER_METHODS
    }


def aggregate(
    task_results: dict[str, dict[str, Any]],
    task_arrays: dict[str, dict[str, np.ndarray]],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    macro = {}
    for method in METHODS:
        values = [task_results[task][method] for task in sorted(task_results)]
        macro[method] = {
            "selected_success": float(np.mean([row["selected_success"] for row in values])),
            "random_expected_success": float(
                np.mean([row["random_expected_success"] for row in values])
            ),
            "delta_vs_random": float(np.mean([row["delta_vs_random"] for row in values])),
            "delta_bootstrap_ci95": bootstrap_macro(
                task_arrays, method, bootstrap, rng
            ),
            "mean_pool_auc": float(
                np.mean(
                    [
                        row["mean_pool_auc"]
                        for row in values
                        if row["mean_pool_auc"] is not None
                    ]
                )
            ),
            "pseudo_target_spearman": (
                float(
                    np.mean(
                        [
                            row["pseudo_target_spearman"]
                            for row in values
                            if row["pseudo_target_spearman"] is not None
                        ]
                    )
                )
                if any(row["pseudo_target_spearman"] is not None for row in values)
                else None
            ),
            "mean_unique_seeds_per_fold": float(
                np.mean(
                    [
                        row["mean_unique_seeds_per_fold_across_16_states"]
                        for row in values
                    ]
                )
            ),
            "task_deltas": {
                task: task_results[task][method]["delta_vs_random"]
                for task in sorted(task_results)
            },
        }
    observed = {method: macro[method]["delta_vs_random"] for method in FWER_METHODS}
    tests = permutation_tests(task_arrays, observed, permutations, rng)
    for method, values in tests.items():
        macro[method].update(values)
    return macro


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def fmt(value: float, signed: bool = False) -> str:
    return ("%+.3f" if signed else "%.3f") % value


def render_report(summary: dict[str, Any]) -> str:
    result = summary["macro"]
    labels = {
        "early_route_central": "旧：早期路由中心",
        "initial_noise_central": "初始噪声中心",
        "old_future_route_head": "旧：预测未来路由中心",
        "action_central_head": "预测最终动作中心",
        "residual_action_central_head": "去噪声模板 + 动作中心",
        "residual_future_change_head": "去噪声模板 + 剩余路由变化",
        "residual_action_vector_head": "去噪声模板 + 预测动作向量",
        "residual_route_central": "去噪声模板后的路由中心",
        "prefix_contraction": "前三路由的收缩率",
        "cross_layer_agreement": "跨层一致性",
        "selfsup_ensemble": "自监督集成",
        "final_action_central_oracle": "完整最终动作中心 oracle",
        "future_route_change_oracle": "完整剩余路由变化 oracle",
        "late_route_central_oracle": "完整未来路由中心 oracle",
    }
    lines = [
        "# 自监督噪声选择方法改进",
        "",
        "## 实验思路",
        "",
        "- 仍从同一观测的 K8 初始噪声中选 1 个。head 训练只使用初始噪声、前三次完整 32 路 MoE 路由、后续路由和模型自己生成的最终动作；不使用 reward 或 success。",
        "- 同时测试旧未来路由中心目标、最终动作共识、剩余路由变化、预测动作向量、跨层一致性和前三步收缩；改进版先回归掉仅由初始噪声解释的路由模板。",
        "- 每次预测同时留出完整任务和测试 K8 的全部 noise seed。5 个任务全部轮流作为未见任务；成功只在选择完成后评价，多方法结论使用统一置乱和 FWER 校正。",
        "",
        "## 结果",
        "",
        "| 方法 | 选中成功率 | 随机期望 | 差值 | 95% CI | K8 内 AUC | 伪目标 rho | 每 fold 平均选中 seed 数 | FWER p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = result[method]
        low, high = row["delta_bootstrap_ci95"]
        rho = row["pseudo_target_spearman"]
        p = row.get("permutation_p_fwer")
        lines.append(
            "| %s | %s | %s | %s | [%s, %s] | %s | %s | %s | %s |"
            % (
                labels[method],
                fmt(row["selected_success"]),
                fmt(row["random_expected_success"]),
                fmt(row["delta_vs_random"], signed=True),
                fmt(low, signed=True),
                fmt(high, signed=True),
                fmt(row["mean_pool_auc"]),
                "-" if rho is None else fmt(rho),
                fmt(row["mean_unique_seeds_per_fold"]),
                "-" if p is None else fmt(p),
            )
        )
    old = result["old_future_route_head"]
    best_improved_name = max(
        IMPROVED, key=lambda method: result[method]["delta_vs_random"]
    )
    best = result[best_improved_name]
    action_oracle = result["final_action_central_oracle"]
    task_names = sorted(summary["tasks"])
    short_task_names = [
        "goal/middle",
        "goal/top+bowl",
        "long/moka",
        "spatial/ramekin",
        "spatial/stove",
    ]
    lines.extend(
        [
            "",
            "旧 head 的跨任务平均差值为 `%s`，复现了此前约 -1.1 个百分点的失败。它对自己的未来路由伪目标仍有较高相关 `%s`，再次说明问题主要是目标不等于价值，而不是回归器没学会。"
            % (
                fmt(old["delta_vs_random"], signed=True),
                fmt(old["pseudo_target_spearman"]),
            ),
            "",
            "本轮表现最好的改进方法是 `%s`，相对随机 `%s`，95%% CI `%s` 到 `%s`，FWER `p=%s`。"
            % (
                labels[best_improved_name],
                fmt(best["delta_vs_random"], signed=True),
                fmt(best["delta_bootstrap_ci95"][0], signed=True),
                fmt(best["delta_bootstrap_ci95"][1], signed=True),
                fmt(best["permutation_p_fwer"]),
            ),
            "",
            "完整最终动作中心 oracle 相对随机为 `%s`。如果该 oracle 本身不提升，预测动作中心再准确也不能成为可靠的成功率目标；这项诊断用于区分‘head 不准’和‘伪目标错误’。"
            % fmt(action_oracle["delta_vs_random"], signed=True),
            "",
            "| 未见任务 | 旧 future-route head | 新自监督集成 |",
            "|---|---:|---:|",
        ]
    )
    for label, task in zip(short_task_names, task_names):
        lines.append(
            "| %s | %s | %s |"
            % (
                label,
                fmt(
                    summary["tasks"][task]["old_future_route_head"][
                        "delta_vs_random"
                    ],
                    signed=True,
                ),
                fmt(
                    summary["tasks"][task]["selfsup_ensemble"][
                        "delta_vs_random"
                    ],
                    signed=True,
                ),
            )
        )
    lines.extend(
        [
            "",
            "其中 `goal/middle` 的 512 条全部成功，所有方法在该任务上的差值必然为 0，不能提供区分信息。其余四个任务才是有效比较。",
            "",
            "这些仍是 episode-start shadow 关联：一个 seed 还决定后续 replan 的噪声流，不是在线逐 query 干预的因果结果。只有跨任务、校正后仍为正的方法才值得进入闭环 A/B。",
            "",
        ]
    )
    return "\n".join(lines)


def repair_loaded_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Repair display-only aggregates from task rows in an existing result."""

    for method in METHODS:
        available = [
            task[method]["mean_pool_auc"]
            for task in summary["tasks"].values()
            if task[method]["mean_pool_auc"] is not None
        ]
        summary["macro"][method]["mean_pool_auc"] = (
            float(np.mean(available)) if available else None
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.render_existing:
        summary_path = args.out_dir / "summary.json"
        summary = repair_loaded_summary(json.loads(summary_path.read_text()))
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
        (args.out_dir / "report.md").write_text(render_report(summary))
        print("rendered", args.out_dir / "report.md", flush=True)
        return
    if args.bootstrap < 100 or args.permutations < 100:
        raise ValueError("bootstrap and permutations must be at least 100")
    cache_root = args.hub_root.resolve() / "cache/HiMoE-VLA"
    all_pools = []
    integrity = []
    for run in discover_runs(args.hub_root.resolve()):
        print("loading", run.relative_to(cache_root), flush=True)
        pools, diagnostic = load_task_pools(run, cache_root)
        all_pools.extend(pools)
        integrity.append(diagnostic)
    tasks = sorted({pool.task for pool in all_pools})
    task_results = {}
    task_diagnostics = {}
    task_arrays = {}
    for task in tasks:
        print("evaluating held-out task", task, flush=True)
        result, diagnostic, arrays = evaluate_task(task, all_pools)
        task_results[task] = result
        task_diagnostics[task] = diagnostic
        task_arrays[task] = arrays
    macro = aggregate(
        task_results,
        task_arrays,
        args.bootstrap,
        args.permutations,
        args.seed,
    )
    summary = finite_json(
        {
            "experiment": "self_supervised_noise_method_search",
            "protocol": {
                "tasks": len(tasks),
                "states_per_task": 16,
                "candidate_pool": "four disjoint K8 folds of a common K32 grid",
                "early_prefix": PREFIX_STEPS,
                "outcome_blinding": "heads and scores never consume success; success is isolated in evaluation",
                "cross_fit": "complete held task and held K8 seed fold excluded from training",
                "method_selection": "all predefined methods reported; no success-based winner tuning",
                "bootstrap_draws": args.bootstrap,
                "permutation_draws": args.permutations,
                "fwer_methods": list(FWER_METHODS),
                "seed": args.seed,
            },
            "integrity": integrity,
            "macro": macro,
            "tasks": task_results,
            "training_diagnostics": task_diagnostics,
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    print("wrote", args.out_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
