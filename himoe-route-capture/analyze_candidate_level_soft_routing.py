#!/usr/bin/env python3
"""Candidate-level query-0 soft-routing experiment on the 16x32 grids."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/candidate-level-soft-routing"
N_CANDIDATES, N_LAYERS, N_DENOISE, N_TOKENS, N_EXPERTS = 32, 8, 10, 11, 32
ACTION_SCALE = np.tile(np.asarray([0.1] * 6 + [1.0]), 10)
HASH_DIM, PCA_COMPONENTS = 512, 16
RIDGE_ALPHA, LOGISTIC_C = 10.0, 0.1
BOOTSTRAPS, SEED = 3000, 20260829

MIDDLE = "libero_goal/open_the_middle_drawer_of_the_cabinet"
TOP = "libero_goal/open_the_top_drawer_and_put_the_bowl_inside"
LONG = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RAMEKIN = "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"
STOVE = "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"
FIRST_TARGET = {
    MIDDLE: ("drawer", "wooden_cabinet_1_middle_level"),
    TOP: ("drawer", "wooden_cabinet_1_top_level"),
    LONG: ("object", "moka_pot_2_joint0"),
    RAMEKIN: ("object", "akita_black_bowl_1_joint0"),
    STOVE: ("object", "akita_black_bowl_1_joint0"),
}
ACTION_MODELS = {
    "seed_sentinel": ("seed",),
    "state_route_control": ("state_route",),
    "action_soft_route": ("action_route",),
    "seed_plus_route": ("seed", "action_route"),
    "hidden_upper": ("hidden",),
}
PHYSICAL_MODELS = {
    **ACTION_MODELS,
    "action": ("action",),
    "action_plus_route": ("action", "action_route"),
}


@dataclass
class TaskData:
    task: str
    meta: pd.DataFrame
    features: dict[str, np.ndarray]
    action_target: np.ndarray
    eef_target: np.ndarray
    approach_target: np.ndarray
    audit: dict[str, Any]
    geometry: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--pca-components", type=int, default=PCA_COMPONENTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def discover_runs(cache_root: Path) -> list[Path]:
    runs = sorted(
        path.parents[1]
        for path in cache_root.glob("libero_*/*/right-16x32/client/summaries.json")
        if (path.parents[1] / "server/routes.zarr").exists()
        and (path.parents[1] / "server/hidden.zarr").exists()
    )
    tasks = {str(run.relative_to(cache_root).parent) for run in runs}
    if tasks != set(FIRST_TARGET):
        raise RuntimeError(f"expected five K32 tasks, found {sorted(tasks)}")
    return runs


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = output.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(output)):
        raise ValueError("invalid soft router probability")
    return output / mass


def first_rows(episode_ids: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    missing = sorted(set(map(int, episodes)) - set(lookup))
    if missing:
        raise RuntimeError(f"server rows missing episodes {missing[:8]}")
    return np.asarray([lookup[int(episode)] for episode in episodes], dtype=np.int64)


def sequence_statistics(values: np.ndarray) -> np.ndarray:
    """Keep layer/token/expert identity via denoise mean, final, and slope."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 5 or array.shape[1:3] != (N_LAYERS, N_DENOISE):
        raise ValueError(f"unexpected route shape {array.shape}")
    time = np.linspace(-1.0, 1.0, N_DENOISE, dtype=np.float32)
    slope = np.einsum("nldte,d->nlte", array, time, optimize=True)
    slope /= float(time @ time)
    parts = (array.mean(axis=2), array[:, :, -1], slope)
    return np.concatenate([part.reshape(len(array), -1) for part in parts], axis=1)


def hash_project(values: np.ndarray, output_dim: int, seed: int) -> np.ndarray:
    """Fixed signed hashing; unlike PCA, this map is not fitted to data."""
    array = np.asarray(values, dtype=np.float32)
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, output_dim, size=array.shape[1])
    sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=array.shape[1])
    projection = sparse.csr_matrix(
        (sign, (np.arange(array.shape[1]), bucket)),
        shape=(array.shape[1], output_dim),
    )
    return np.asarray(array @ projection, dtype=np.float32)


def center_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    for group in np.unique(groups):
        selected = groups == group
        output[selected] -= output[selected].mean(axis=0, keepdims=True)
    return output.astype(np.float32)


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    square = np.sum(array * array, axis=1)
    distance2 = square[:, None] + square[None, :] - 2.0 * (array @ array.T)
    return np.sqrt(np.maximum(distance2 / max(array.shape[1], 1), 0.0))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if np.ptp(left) <= 1e-15 or np.ptp(right) <= 1e-15:
        return float("nan")
    value = spearmanr(left, right).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _joint(layout: dict[str, Any], name: str) -> dict[str, Any]:
    return next(row for row in layout["joints"] if row["joint"] == name)


def _within_sd(values: np.ndarray, groups: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean([array[groups == group].std() for group in np.unique(groups)]))


def light_hidden_features(hidden_group: Any, rows: np.ndarray) -> np.ndarray:
    """Lossy matched summary of HB router-input hidden states."""
    result = []
    time = np.linspace(-1.0, 1.0, N_DENOISE, dtype=np.float32)
    for lo in range(0, len(rows), 16):
        hidden = np.asarray(
            hidden_group["hb_hidden"].oindex[rows[lo : lo + 16], :, :, :, :],
            dtype=np.float32,
        )
        state = hidden[..., 0, :]
        action = hidden[..., 1:, :]
        action_mean = action.mean(axis=3)
        sequences = (
            np.sqrt(np.mean(state * state, axis=-1)),
            np.sqrt(np.mean(action * action, axis=(-1, -2))),
            np.sqrt(np.mean((state - action_mean) ** 2, axis=-1)),
        )
        features = []
        for sequence in sequences:
            slope = np.sum(sequence * time, axis=-1) / float(time @ time)
            features.extend(
                (sequence.mean(axis=-1), sequence.std(axis=-1), slope, sequence[..., -1])
            )
        result.append(np.column_stack(features).astype(np.float32))
    return np.concatenate(result)


def load_task(run: Path, cache_root: Path, seed: int) -> TaskData:
    task = str(run.relative_to(cache_root).parent)
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 512:
        raise RuntimeError(f"{task}: expected 512 episodes")
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    groups = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    success = np.asarray([bool(row["success"]) for row in summaries])
    counts = np.asarray([int(row["inference_calls"]) for row in summaries])
    if counts.min() < 2:
        raise RuntimeError(f"{task}: x1 is unavailable")
    canonical_seeds = None
    for group in np.unique(groups):
        current = tuple(sorted(map(int, seeds[groups == group])))
        if len(current) != N_CANDIDATES or len(set(current)) != N_CANDIDATES:
            raise RuntimeError(f"{task}/{group}: incomplete K32 pool")
        canonical_seeds = current if canonical_seeds is None else canonical_seeds
        if current != canonical_seeds:
            raise RuntimeError(f"{task}: seed grid drift")

    layout = json.loads((run / "client/sim_layout.json").read_text())
    target_kind, target_name = FIRST_TARGET[task]
    target = _joint(layout, target_name)
    target_lo, target_hi = int(target["state_lo"]), int(target["state_hi"])
    state0, state1, sim0, sim1, final, actions = [], [], [], [], [], []
    for episode in episodes:
        path = run / "client" / f"episode_{int(episode):02d}.npz"
        with np.load(path, allow_pickle=False) as archive:
            state = np.asarray(archive["state"][:2], dtype=np.float32)
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
            action = np.asarray(archive["actions"][0], dtype=np.float32)
        state0.append(state[0])
        state1.append(state[1])
        sim0.append(sim[0])
        sim1.append(sim[1])
        final.append(sim[-1])
        actions.append(action.reshape(-1) / ACTION_SCALE)
    state0, state1 = np.stack(state0), np.stack(state1)
    sim0, sim1, final = np.stack(sim0), np.stack(sim1), np.stack(final)
    actions = np.stack(actions).astype(np.float32)

    x0_state_max = x0_sim_max = 0.0
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        x0_state_max = max(
            x0_state_max,
            float(np.max(np.abs(state0[selected] - state0[selected[:1]]))),
        )
        x0_sim_max = max(
            x0_sim_max,
            float(np.max(np.abs(sim0[selected] - sim0[selected[:1]]))),
        )
    if x0_state_max != 0.0 or x0_sim_max != 0.0:
        raise RuntimeError(f"{task}: x0 differs within init")
    sim_dt = sim1[:, 0] - sim0[:, 0]
    if not np.allclose(sim_dt, 0.5, atol=1e-6, rtol=0.0):
        raise RuntimeError(f"{task}: x0-x1 simulator time is not 0.5 s")

    eef_delta = state1[:, :3] - state0[:, :3]
    eef_step = np.linalg.norm(eef_delta, axis=1)
    approach = np.full(len(episodes), np.nan, dtype=np.float32)
    lift = np.full(len(episodes), np.nan, dtype=np.float32)
    progress = np.zeros(len(episodes), dtype=np.float32)
    if target_kind == "object":
        if target_hi - target_lo != 7:
            raise ValueError(f"{task}: first target is not a free joint")
        object0 = sim0[:, target_lo : target_lo + 3]
        object1 = sim1[:, target_lo : target_lo + 3]
        approach = (
            np.linalg.norm(state0[:, :3] - object0, axis=1)
            - np.linalg.norm(state1[:, :3] - object1, axis=1)
        ).astype(np.float32)
        lift = (object1[:, 2] - object0[:, 2]).astype(np.float32)
        goal = final[success, target_lo : target_lo + 3].mean(axis=0)
        progress = (
            np.linalg.norm(object0 - goal, axis=1)
            - np.linalg.norm(object1 - goal, axis=1)
        ).astype(np.float32)
    else:
        final_delta = final[success, target_lo] - sim0[success, target_lo]
        direction = float(np.sign(np.median(final_delta)))
        progress = (direction * (sim1[:, target_lo] - sim0[:, target_lo])).astype(
            np.float32
        )

    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden_group = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:])
    hidden_episode = np.asarray(hidden_group["episode_id"][:])
    route_rows = first_rows(route_episode, episodes)
    hidden_rows = first_rows(hidden_episode, episodes)
    expected = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    if not np.array_equal(route_rows, expected) or not np.array_equal(
        hidden_rows, expected
    ):
        raise RuntimeError(f"{task}: server/client offsets disagree")
    if not np.array_equal(route_episode, hidden_episode):
        raise RuntimeError(f"{task}: route/hidden stores disagree")
    probability = normalize_probabilities(
        np.asarray(
            route_group["hb_router_probs"].oindex[route_rows, :, :, :, :],
            dtype=np.float32,
        )
    )
    action_probability = probability[..., 1:, :]
    state_probability = probability[..., :1, :]
    action_route = hash_project(
        sequence_statistics(action_probability), HASH_DIM, seed + 1
    )
    state_route = hash_project(
        sequence_statistics(state_probability), HASH_DIM, seed + 2
    )
    hidden = light_hidden_features(hidden_group, hidden_rows)

    geometry = []
    state_route_max = 0.0
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        upper = np.triu_indices(N_CANDIDATES, 1)
        state_root = np.sqrt(state_probability[selected])
        state_route_max = max(
            state_route_max,
            float(
                np.sqrt(
                    0.5 * np.sum((state_root - state_root[:1]) ** 2, axis=-1)
                ).max()
            ),
        )
        action_distance = pairwise_rms(actions[selected])[upper]
        route_distance = pairwise_rms(np.sqrt(action_probability[selected]))[upper]
        hidden_distance = pairwise_rms(hidden[selected])[upper]
        geometry.append(
            {
                "task": task,
                "init_state_id": int(group),
                "candidate_pairs": len(action_distance),
                "action_route_action_distance_rho": safe_spearman(
                    route_distance, action_distance
                ),
                "hidden_action_distance_rho": safe_spearman(
                    hidden_distance, action_distance
                ),
            }
        )

    seed_order = {value: index for index, value in enumerate(canonical_seeds or ())}
    seed_feature = np.zeros((len(seeds), N_CANDIDATES), dtype=np.float32)
    seed_feature[
        np.arange(len(seeds)), [seed_order[int(value)] for value in seeds]
    ] = 1.0
    features = {
        "seed": center_by_group(seed_feature, groups),
        "state_route": center_by_group(state_route, groups),
        "action_route": center_by_group(action_route, groups),
        "action": center_by_group(actions, groups),
        "hidden": center_by_group(hidden, groups),
    }
    mixed = sum(
        success[groups == group].any() and not success[groups == group].all()
        for group in np.unique(groups)
    )
    meta = pd.DataFrame(
        {
            "task": task,
            "episode": episodes,
            "init_state_id": groups,
            "flow_noise_seed": seeds,
            "success": success,
            "first_target_kind": target_kind,
            "first_target": target_name,
            "eef_delta_x_m": eef_delta[:, 0],
            "eef_delta_y_m": eef_delta[:, 1],
            "eef_delta_z_m": eef_delta[:, 2],
            "eef_step_m": eef_step,
            "eef_target_approach_m": approach,
            "first_target_lift_m": lift,
            "task_progress_change": progress,
        }
    )
    audit = {
        "task": task,
        "episodes": len(episodes),
        "initial_states": int(len(np.unique(groups))),
        "seeds_per_state": N_CANDIDATES,
        "successes": int(success.sum()),
        "mixed_outcome_initial_states": int(mixed),
        "minimum_queries": int(counts.min()),
        "maximum_query0_state_difference": x0_state_max,
        "maximum_query0_sim_difference": x0_sim_max,
        "sim_x0_x1_seconds": float(np.median(sim_dt)),
        "state_route_candidate_max_hellinger": state_route_max,
        "action_within_init_coordinate_sd": float(
            np.mean(
                [
                    actions[groups == group].std(axis=0).mean()
                    for group in np.unique(groups)
                ]
            )
        ),
        "eef_step_within_init_sd_m": _within_sd(eef_step, groups),
        "approach_within_init_sd_m": (
            _within_sd(approach, groups) if target_kind == "object" else None
        ),
        "lift_within_init_sd_m": (
            _within_sd(lift, groups) if target_kind == "object" else None
        ),
        "task_progress_within_init_sd": _within_sd(progress, groups),
    }
    approach_target = np.full((len(episodes), 1), np.nan, dtype=np.float32)
    if target_kind == "object":
        approach_target = center_by_group(approach[:, None], groups)
    return TaskData(
        task=task,
        meta=meta,
        features=features,
        action_target=center_by_group(actions, groups),
        eef_target=center_by_group(eef_delta, groups),
        approach_target=approach_target,
        audit=audit,
        geometry=geometry,
    )


def grouped_splits(groups: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("at least two groups are required")
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    held_out = np.array_split(shuffled, min(folds, len(shuffled)))
    splits = []
    for fold_groups in held_out:
        test = np.isin(groups, fold_groups)
        splits.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return splits


def fold_reduce(
    train: np.ndarray,
    test: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    varying = train.std(axis=0) > 1e-10
    if not np.any(varying):
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[:, varying])
    test_scaled = scaler.transform(test[:, varying])
    count = min(components, train_scaled.shape[0] - 1, train_scaled.shape[1])
    if count < 1:
        return np.zeros((len(train), 1)), np.zeros((len(test), 1))
    if train_scaled.shape[1] > count:
        reducer = PCA(n_components=count, svd_solver="randomized", random_state=seed)
        train_scaled = reducer.fit_transform(train_scaled)
        test_scaled = reducer.transform(test_scaled)
    return train_scaled, test_scaled


def concatenate_features(features: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.concatenate([features[name] for name in names], axis=1)


def fold_reduce_blocks(
    features: dict[str, np.ndarray],
    blocks: tuple[str, ...],
    train: np.ndarray,
    test: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit each block separately so a larger model retains the base subspace."""
    train_parts, test_parts = [], []
    for block in blocks:
        block_seed = stable_seed(f"fold-block:{block}", seed)
        block_components = (
            features[block].shape[1] if block == "action" else components
        )
        train_x, test_x = fold_reduce(
            features[block][train],
            features[block][test],
            block_components,
            block_seed,
        )
        train_parts.append(train_x)
        test_parts.append(test_x)
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def crossfit_regression(
    features: dict[str, np.ndarray],
    target: np.ndarray,
    groups: np.ndarray,
    models: dict[str, tuple[str, ...]],
    components: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    predictions = {
        name: np.full_like(np.asarray(target, dtype=np.float64), np.nan)
        for name in models
    }
    fold_id = np.full(len(target), -1, dtype=np.int16)
    for fold, (train, test) in enumerate(grouped_splits(groups, 8, seed)):
        fold_id[test] = fold
        for name, blocks in models.items():
            train_x, test_x = fold_reduce_blocks(
                features,
                blocks,
                train,
                test,
                components,
                seed + fold * 101,
            )
            estimator = Ridge(alpha=RIDGE_ALPHA)
            estimator.fit(train_x, target[train])
            estimate = estimator.predict(test_x)
            if predictions[name].ndim == 2 and estimate.ndim == 1:
                estimate = estimate[:, None]
            predictions[name][test] = estimate
    if np.any(fold_id < 0) or any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("incomplete regression cross-fit")
    return predictions, fold_id


def crossfit_success(
    features: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    models: dict[str, tuple[str, ...]],
    components: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    predictions = {
        name: np.full(len(labels), np.nan, dtype=np.float64) for name in models
    }
    fold_id = np.full(len(labels), -1, dtype=np.int16)
    for fold, (train, test) in enumerate(grouped_splits(groups, 8, seed)):
        fold_id[test] = fold
        for name, blocks in models.items():
            train_x, test_x = fold_reduce_blocks(
                features,
                blocks,
                train,
                test,
                components,
                seed + fold * 101,
            )
            estimator = LogisticRegression(
                C=LOGISTIC_C,
                max_iter=2000,
                solver="lbfgs",
            )
            estimator.fit(train_x, labels[train])
            predictions[name][test] = estimator.predict_proba(test_x)[:, 1]
    if np.any(fold_id < 0) or any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("incomplete classification cross-fit")
    return predictions, fold_id


def cluster_statistics(
    meta: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    records = []
    for (task, group), indices in meta.groupby(["task", "init_state_id"], sort=True).indices.items():
        index = np.asarray(indices, dtype=np.int64)
        truth, estimate = target[index], prediction[index]
        record: dict[str, Any] = {
            "task": task,
            "init_state_id": int(group),
            "candidates": len(index),
        }
        if metric == "r2":
            record["numerator"] = float(np.sum((truth - estimate) ** 2))
            record["denominator"] = float(np.sum(truth**2))
        elif metric == "pairwise":
            upper = np.triu_indices(len(index), 1)
            record["value"] = safe_spearman(
                pairwise_rms(truth)[upper], pairwise_rms(estimate)[upper]
            )
        elif metric == "rank":
            record["value"] = safe_spearman(truth.reshape(-1), estimate.reshape(-1))
        elif metric == "auc":
            record["value"] = float(roc_auc_score(truth, estimate))
        else:
            raise ValueError(f"unknown metric {metric}")
        records.append(record)
    return pd.DataFrame.from_records(records)


def aggregate_clusters(table: pd.DataFrame, metric: str) -> float:
    task_values = []
    for _, rows in table.groupby("task", sort=True):
        if metric == "r2":
            denominator = rows["denominator"].sum()
            task_values.append(1.0 - rows["numerator"].sum() / denominator)
        else:
            task_values.append(float(rows["value"].mean()))
    return float(np.mean(task_values))


def bootstrap_clusters(
    table: pd.DataFrame,
    metric: str,
    bootstraps: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    point = aggregate_clusters(table, metric)
    rng = np.random.default_rng(seed)
    by_task = [rows.reset_index(drop=True) for _, rows in table.groupby("task", sort=True)]
    draws = np.empty(bootstraps, dtype=np.float64)
    for bootstrap in range(bootstraps):
        task_values = []
        for rows in by_task:
            sampled = rows.iloc[rng.integers(0, len(rows), size=len(rows))]
            if metric == "r2":
                task_values.append(
                    1.0 - sampled["numerator"].sum() / sampled["denominator"].sum()
                )
            else:
                task_values.append(float(sampled["value"].mean()))
        draws[bootstrap] = np.mean(task_values)
    return point, draws


def metric_result(
    endpoint: str,
    model: str,
    metric: str,
    meta: pd.DataFrame,
    target: np.ndarray,
    prediction: np.ndarray,
    bootstraps: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    table = cluster_statistics(meta, target, prediction, metric)
    point, draws = bootstrap_clusters(table, metric, bootstraps, seed)
    row = {
        "endpoint": endpoint,
        "model": model,
        "metric": metric,
        "estimate": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "tasks": int(table["task"].nunique()),
        "initial_states": int(len(table)),
        "candidates": int(table["candidates"].sum()),
        "uncertainty_scope": "fixed_oof_table_cluster_resampling",
    }
    return row, draws


def stable_seed(label: str, base: int) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return base + int.from_bytes(digest[:4], "little") % 1_000_000


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def paired_increment(
    endpoint: str,
    metric: str,
    larger: str,
    smaller: str,
    metrics: dict[tuple[str, str, str], tuple[dict[str, Any], np.ndarray]],
) -> dict[str, Any]:
    large_row, large_draws = metrics[(endpoint, larger, metric)]
    small_row, small_draws = metrics[(endpoint, smaller, metric)]
    difference = large_draws - small_draws
    return {
        "endpoint": endpoint,
        "metric": metric,
        "larger_model": larger,
        "smaller_model": smaller,
        "estimate": float(large_row["estimate"] - small_row["estimate"]),
        "ci95_low": float(np.quantile(difference, 0.025)),
        "ci95_high": float(np.quantile(difference, 0.975)),
        "bootstrap_sd": float(difference.std(ddof=1)),
        "mde_80pct_two_sided_5pct": float(2.801585 * difference.std(ddof=1)),
        "tasks": large_row["tasks"],
        "initial_states": large_row["initial_states"],
        "candidates": large_row["candidates"],
        "uncertainty_scope": "fixed_oof_table_cluster_resampling",
    }


def direct_geometry_summary(
    geometry: pd.DataFrame,
    column: str,
    bootstraps: int,
    seed: int,
) -> dict[str, Any]:
    table = geometry[
        ["task", "init_state_id", "candidate_pairs", column]
    ].rename(columns={"candidate_pairs": "candidates", column: "value"})
    point, draws = bootstrap_clusters(table, "pairwise", bootstraps, seed)
    return {
        "signal": column,
        "mean_within_init_spearman": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "bootstrap_sd": float(draws.std(ddof=1)),
        "tasks": int(table["task"].nunique()),
        "initial_states": int(len(table)),
        "candidate_pairs": int(table["candidates"].sum()),
        "uncertainty_scope": "direct_cluster_resampling",
    }


def short_task(task: str) -> str:
    return {
        MIDDLE: "middle drawer",
        TOP: "top drawer + bowl",
        LONG: "two moka pots",
        RAMEKIN: "bowl on ramekin",
        STOVE: "bowl on stove",
    }[task]


def markdown_table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    display = frame if columns is None else frame[columns]
    display = display.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4g}"
        )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    rows = [
        [
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        ]
        for row in display.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    audits: pd.DataFrame,
    metrics: pd.DataFrame,
    increments: pd.DataFrame,
    geometry_summary: pd.DataFrame,
    success_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    audit_display = audits.copy()
    audit_display["task"] = audit_display["task"].map(short_task)
    audit_columns = [
        "task",
        "successes",
        "mixed_outcome_initial_states",
        "action_within_init_coordinate_sd",
        "eef_step_within_init_sd_m",
        "approach_within_init_sd_m",
        "lift_within_init_sd_m",
        "task_progress_within_init_sd",
    ]
    selected_metrics = metrics[
        metrics["model"].isin(
            [
                "seed_sentinel",
                "state_route_control",
                "action_soft_route",
                "action",
                "action_plus_route",
                "hidden_upper",
            ]
        )
    ]
    final_groups = (
        int(success_meta[["task", "init_state_id"]].drop_duplicates().shape[0])
        if len(success_meta)
        else 0
    )
    final_rate = float(success_meta["success"].mean()) if len(success_meta) else float("nan")
    route_action = geometry_summary.loc[
        geometry_summary["signal"] == "action_route_action_distance_rho"
    ].iloc[0]
    def estimate(endpoint: str, model: str, metric: str) -> float:
        row = metrics[
            (metrics["endpoint"] == endpoint)
            & (metrics["model"] == model)
            & (metrics["metric"] == metric)
        ]
        return float(row.iloc[0]["estimate"])

    def contrast(
        endpoint: str, metric: str, larger: str, smaller: str
    ) -> pd.Series:
        row = increments[
            (increments["endpoint"] == endpoint)
            & (increments["metric"] == metric)
            & (increments["larger_model"] == larger)
            & (increments["smaller_model"] == smaller)
        ]
        return row.iloc[0]

    route_action_r2 = estimate("action_chunk", "action_soft_route", "r2")
    action_eef_r2 = estimate("eef_delta_0p5s", "action", "r2")
    route_eef_r2 = estimate("eef_delta_0p5s", "action_soft_route", "r2")
    joint_eef_r2 = estimate("eef_delta_0p5s", "action_plus_route", "r2")
    route_success_auc = estimate(
        "final_success_mixed_pools", "action_soft_route", "auc"
    )
    hidden_success_auc = estimate(
        "final_success_mixed_pools", "hidden_upper", "auc"
    )
    eef_route_increment = contrast(
        "eef_delta_0p5s", "r2", "action_plus_route", "action"
    )
    success_route_increment = contrast(
        "final_success_mixed_pools",
        "conditional_auc",
        "action_plus_route",
        "action",
    )
    lines = [
        "# Query-0 candidate-level soft-routing audit",
        "",
        "## Result",
        "",
        "This is a candidate-level closing experiment, not a rollout clustering result. "
        "Every comparison is among the 32 flow-noise candidates of one task and one "
        "initial state at query 0. Complete initial-state pools are held out from fitting. "
        "Because features are centered using the observed K=32 pool, the estimand is "
        "transductive ranking within a complete candidate pool, not inductive prediction "
        "for one candidate in isolation.",
        "",
        f"Directly, action-token soft-routing distance and normalized action-chunk "
        f"distance have mean within-pool Spearman {route_action['mean_within_init_spearman']:.3f} "
        f"(cluster-bootstrap 95% CI {route_action['ci95_low']:.3f} to "
        f"{route_action['ci95_high']:.3f}). This is association, not proof that routing "
        "causes the action difference: routing and action can both be downstream of the "
        "flow seed and action hidden state.",
        "",
        f"The cross-fitted route decoder explains {route_action_r2:.3f} of centered "
        f"action-chunk variance. For x0-to-x1 EEF response, action explains "
        f"{action_eef_r2:.3f}, route alone {route_eef_r2:.3f}, and the blockwise "
        f"joint action-plus-route model {joint_eef_r2:.3f}. Thus routing clearly tracks "
        "which action candidate was sampled. The action block retains all 70 standardized "
        "chunk coordinates; route receives a separate 16-PC training-fold reduction. "
        "The joint model therefore retains exactly the full action representation used "
        "by action-only and adds the route subspace. Its response increment is "
        f"{eef_route_increment['estimate']:+.6f} (95% CI "
        f"{eef_route_increment['ci95_low']:+.6f} to "
        f"{eef_route_increment['ci95_high']:+.6f}), so routing adds no useful prediction "
        "of this observed response once the complete action chunk is known.",
        "",
        f"For final success, route-only conditional AUC is {route_success_auc:.3f} and "
        f"the hidden comparator is {hidden_success_auc:.3f}. The learned score direction "
        "is not a usable selector on held-out initial states; the route AUC below 0.5 is "
        "reverse transfer, not proof that the representation contains no separable outcome "
        "information. This analysis is retrospective and conditional because mixed pools "
        "are selected using final outcomes. It therefore closes only this prespecified "
        "decoder as a quality selector. Action-plus-route versus full action changes AUC "
        f"by {success_route_increment['estimate']:+.3f} (95% CI "
        f"{success_route_increment['ci95_low']:+.3f} to "
        f"{success_route_increment['ci95_high']:+.3f}; approximate MDE80 "
        f"{success_route_increment['mde_80pct_two_sided_5pct']:.3f}). Broader success "
        "informativeness remains unresolved, and no additional endpoint search is triggered.",
        "",
        "The first-chunk EEF endpoint is retained as a short-horizon physical-response "
        "endpoint. It is not labelled task quality. EEF-to-object approach is additionally "
        "available only for the three object-first tasks. Object lift, drawer displacement, "
        "and task-progress changes are reported below as identifiability audits because "
        "they are nearly or exactly degenerate over the first 0.5 seconds.",
        "",
        "## Endpoint identifiability",
        "",
        markdown_table(audit_display, audit_columns),
        "",
        "The query-0 robot state and full simulator state are bit-identical within every "
        "task-by-init pool (maximum difference 0), and x1 is the next policy boundary "
        "0.5 seconds after x0. RGB frames were not archived in these NPZ files, so there "
        "is no retrospective pixel hash. Equality is established for cached simulator and "
        "proprio state, not independently for pixels.",
        "",
        "## Cross-fitted estimates",
        "",
        markdown_table(
            selected_metrics,
            [
                "endpoint",
                "metric",
                "model",
                "estimate",
                "ci95_low",
                "ci95_high",
                "tasks",
                "initial_states",
            ],
        ),
        "",
        "All learned feature scaling and PCA are fitted inside the training fold. The action "
        "block retains all 70 standardized coordinates; every other block has at most "
        f"{args.pca_components} components. Eight grouped folds are used when "
        "possible. The action target and physical targets are centered within init, so "
        "R2 is relative to predicting the held-out pool mean. Pairwise/rank metrics do "
        "not depend on target centering. Feature centering uses the complete observed "
        "K=32 pool and therefore defines the transductive scope stated above.",
        "",
        "## Model contrasts and resolution",
        "",
        markdown_table(
            increments,
            [
                "endpoint",
                "metric",
                "larger_model",
                "smaller_model",
                "estimate",
                "ci95_low",
                "ci95_high",
                "mde_80pct_two_sided_5pct",
            ],
        ),
        "",
        "The reported MDE is 2.801585 times the paired cluster-bootstrap standard "
        "deviation. Both intervals and MDE resample the frozen OOF prediction table; "
        "they do not refit folds and therefore omit learning-procedure uncertainty from "
        "overlapping CV training sets. They describe conditional resolution of this OOF "
        "table, not prospective study power and not an observed effect.",
        "",
        "## Final success, secondary endpoint",
        "",
        f"Final success retrospectively uses only the {final_groups} initial states containing both "
        f"success and failure ({len(success_meta)} candidates; success rate "
        f"{final_rate:.3f}). It is deliberately secondary: labels occur far after query 0, "
        "fixed all-success or all-failure pools contain no within-init candidate signal, "
        "and eligibility cannot be known at query 0. These AUCs are not a deployable "
        "prospective screen.",
        "",
        "## Controls and interpretation",
        "",
        "- seed_sentinel tests whether a flow-noise seed identity generalizes across held-out initial states.",
        "- state_route_control is the state-token route negative control. It is candidate-invariant at query 0.",
        "- action is the direct sampled action chunk and is the positive control for x0-to-x1 response.",
        "- hidden_upper is retained as an artifact key, but the feature is a lossy RMS/moment hidden summary, not a true hidden-state upper bound.",
        "- action_plus_route retains the same full 70-coordinate action block as action-only and adds a separately fitted 16-PC route block. Its held-init score difference is an additive generalization test, not a causal effect.",
        "",
        "No hard expert IDs are used. Bootstrap resampling is by initial state within "
        "each fixed task and all reported predictions are frozen out-of-fold predictions.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def self_test(seed: int) -> None:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(8), N_CANDIDATES)
    latent = rng.normal(size=(len(groups), 4))
    group_offset = rng.normal(size=(8, 4))
    target = latent + group_offset[groups]
    target = center_by_group(target, groups)
    route = latent @ rng.normal(size=(4, 40)) + rng.normal(
        scale=0.05, size=(len(groups), 40)
    )
    feature = {
        "route": center_by_group(route, groups),
        "zero": np.zeros((len(groups), 5), dtype=np.float32),
    }
    split_train, split_test = grouped_splits(groups, 8, seed)[0]
    base_train, base_test = fold_reduce_blocks(
        feature, ("route",), split_train, split_test, 8, seed
    )
    joint_train, joint_test = fold_reduce_blocks(
        feature, ("route", "zero"), split_train, split_test, 8, seed
    )
    if not np.array_equal(base_train, joint_train[:, : base_train.shape[1]]):
        raise AssertionError("joint model did not retain the training base subspace")
    if not np.array_equal(base_test, joint_test[:, : base_test.shape[1]]):
        raise AssertionError("joint model did not retain the test base subspace")
    predictions, folds = crossfit_regression(
        feature,
        target,
        groups,
        {"route": ("route",), "zero": ("zero",)},
        8,
        seed,
    )
    if np.unique(folds).size != 8:
        raise AssertionError("group folds are incomplete")
    for fold in np.unique(folds):
        held = np.unique(groups[folds == fold])
        if np.any(np.isin(groups[folds != fold], held)):
            raise AssertionError("group leakage")
    route_error = np.sum((target - predictions["route"]) ** 2)
    zero_error = np.sum((target - predictions["zero"]) ** 2)
    if not route_error < 0.5 * zero_error:
        raise AssertionError("synthetic route decoder did not recover signal")
    centered = center_by_group(rng.normal(size=(len(groups), 3)), groups)
    if np.max(np.abs([centered[groups == group].mean() for group in np.unique(groups)])) > 1e-6:
        raise AssertionError("group centering failed")
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.seed)
        return
    if args.bootstrap < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "completion.json").unlink(missing_ok=True)
    task_data = []
    for index, run in enumerate(discover_runs(args.cache_root)):
        print(f"loading {run.relative_to(args.cache_root).parent}", flush=True)
        task_data.append(load_task(run, args.cache_root, args.seed + index * 1000))

    all_meta = []
    action_targets, eef_targets = [], []
    action_predictions = {name: [] for name in ACTION_MODELS}
    eef_predictions = {name: [] for name in PHYSICAL_MODELS}
    approach_meta_parts, approach_targets = [], []
    approach_predictions = {name: [] for name in PHYSICAL_MODELS}
    success_meta_parts, success_targets = [], []
    success_predictions = {name: [] for name in PHYSICAL_MODELS}
    audits, geometry = [], []
    row_offset = 0

    for task_index, data in enumerate(task_data):
        meta = data.meta.copy()
        meta["candidate_row"] = np.arange(len(meta)) + row_offset
        groups = meta["init_state_id"].to_numpy()
        action_oof, action_fold = crossfit_regression(
            data.features,
            data.action_target,
            groups,
            ACTION_MODELS,
            args.pca_components,
            args.seed + task_index * 100 + 1,
        )
        eef_oof, eef_fold = crossfit_regression(
            data.features,
            data.eef_target,
            groups,
            PHYSICAL_MODELS,
            args.pca_components,
            args.seed + task_index * 100 + 2,
        )
        meta["action_fold"] = action_fold
        meta["physical_fold"] = eef_fold
        meta["mixed_outcome_pool"] = False
        meta["success_fold"] = -1
        all_meta.append(meta)
        action_targets.append(data.action_target)
        eef_targets.append(data.eef_target)
        for name in ACTION_MODELS:
            action_predictions[name].append(action_oof[name])
        for name in PHYSICAL_MODELS:
            eef_predictions[name].append(eef_oof[name])

        if np.all(np.isfinite(data.approach_target)):
            approach_oof, _ = crossfit_regression(
                data.features,
                data.approach_target,
                groups,
                PHYSICAL_MODELS,
                args.pca_components,
                args.seed + task_index * 100 + 3,
            )
            approach_meta_parts.append(meta.copy())
            approach_targets.append(data.approach_target)
            for name in PHYSICAL_MODELS:
                approach_predictions[name].append(approach_oof[name])

        mixed_groups = []
        labels = meta["success"].to_numpy(dtype=np.int8)
        for group in np.unique(groups):
            current = labels[groups == group]
            if current.any() and not current.all():
                mixed_groups.append(group)
        selected = np.isin(groups, mixed_groups)
        if np.any(selected):
            selected_features = {
                name: values[selected] for name, values in data.features.items()
            }
            success_oof, success_fold = crossfit_success(
                selected_features,
                labels[selected],
                groups[selected],
                PHYSICAL_MODELS,
                args.pca_components,
                args.seed + task_index * 100 + 4,
            )
            meta.loc[selected, "mixed_outcome_pool"] = True
            meta.loc[selected, "success_fold"] = success_fold
            success_meta = meta.loc[selected].copy().reset_index(drop=True)
            success_meta_parts.append(success_meta)
            success_targets.append(labels[selected])
            for name in PHYSICAL_MODELS:
                success_predictions[name].append(success_oof[name])

        audits.append(data.audit)
        geometry.extend(data.geometry)
        row_offset += len(meta)

    meta = pd.concat(all_meta, ignore_index=True)
    action_meta = meta.reset_index(drop=True)
    eef_meta = meta.reset_index(drop=True)
    action_target = np.concatenate(action_targets)
    eef_target = np.concatenate(eef_targets)
    action_prediction = {
        name: np.concatenate(parts) for name, parts in action_predictions.items()
    }
    eef_prediction = {
        name: np.concatenate(parts) for name, parts in eef_predictions.items()
    }
    approach_meta = pd.concat(approach_meta_parts, ignore_index=True)
    approach_target = np.concatenate(approach_targets)
    approach_prediction = {
        name: np.concatenate(parts) for name, parts in approach_predictions.items()
    }
    success_meta = pd.concat(success_meta_parts, ignore_index=True)
    success_target = np.concatenate(success_targets)
    success_prediction = {
        name: np.concatenate(parts) for name, parts in success_predictions.items()
    }

    metric_store: dict[
        tuple[str, str, str], tuple[dict[str, Any], np.ndarray]
    ] = {}
    specifications = [
        ("action_chunk", "r2", action_meta, action_target, action_prediction),
        (
            "action_chunk",
            "pairwise_spearman",
            action_meta,
            action_target,
            action_prediction,
        ),
        ("eef_delta_0p5s", "r2", eef_meta, eef_target, eef_prediction),
        (
            "eef_target_approach_0p5s",
            "r2",
            approach_meta,
            approach_target,
            approach_prediction,
        ),
        (
            "eef_target_approach_0p5s",
            "within_init_rank",
            approach_meta,
            approach_target,
            approach_prediction,
        ),
        (
            "final_success_mixed_pools",
            "conditional_auc",
            success_meta,
            success_target,
            success_prediction,
        ),
    ]
    kind_map = {
        "r2": "r2",
        "pairwise_spearman": "pairwise",
        "within_init_rank": "rank",
        "conditional_auc": "auc",
    }
    for endpoint, metric, endpoint_meta, target, predictions in specifications:
        bootstrap_seed = stable_seed(f"{endpoint}:{metric}", args.seed)
        for model, prediction in predictions.items():
            metric_store[(endpoint, model, metric)] = metric_result(
                endpoint,
                model,
                kind_map[metric],
                endpoint_meta,
                target,
                prediction,
                args.bootstrap,
                bootstrap_seed,
            )
    metric_rows = [value[0] for value in metric_store.values()]
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["endpoint", "metric", "model"]
    )

    comparisons = [
        ("action_chunk", "r2", "action_soft_route", "seed_sentinel"),
        ("action_chunk", "r2", "action_soft_route", "state_route_control"),
        ("action_chunk", "pairwise_spearman", "action_soft_route", "seed_sentinel"),
        ("eef_delta_0p5s", "r2", "action", "state_route_control"),
        ("eef_delta_0p5s", "r2", "action_soft_route", "state_route_control"),
        ("eef_delta_0p5s", "r2", "action_plus_route", "action"),
        (
            "eef_target_approach_0p5s",
            "r2",
            "action_plus_route",
            "action",
        ),
        (
            "eef_target_approach_0p5s",
            "within_init_rank",
            "action_plus_route",
            "action",
        ),
        (
            "final_success_mixed_pools",
            "conditional_auc",
            "action_soft_route",
            "state_route_control",
        ),
        (
            "final_success_mixed_pools",
            "conditional_auc",
            "action_plus_route",
            "action",
        ),
    ]
    increments = pd.DataFrame(
        [
            paired_increment(endpoint, metric, larger, smaller, metric_store)
            for endpoint, metric, larger, smaller in comparisons
        ]
    )

    audits_frame = pd.DataFrame(audits)
    geometry_frame = pd.DataFrame(geometry)
    geometry_summaries = pd.DataFrame(
        [
            direct_geometry_summary(
                geometry_frame,
                column,
                args.bootstrap,
                stable_seed(f"geometry:{column}", args.seed),
            )
            for column in [
                "action_route_action_distance_rho",
                "hidden_action_distance_rho",
            ]
        ]
    )

    meta.to_csv(args.out_dir / "episode_query0.csv", index=False)
    audits_frame.to_json(
        args.out_dir / "alignment_audit.json", orient="records", indent=2
    )
    geometry_frame.to_csv(args.out_dir / "pairwise_geometry.csv", index=False)
    geometry_summaries.to_csv(args.out_dir / "pairwise_geometry_summary.csv", index=False)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    increments.to_csv(args.out_dir / "increment_tests.csv", index=False)
    prediction_archive: dict[str, np.ndarray] = {
        "action_target": action_target,
        "eef_target": eef_target,
        "approach_candidate_row": approach_meta["candidate_row"].to_numpy(),
        "approach_target": approach_target,
        "success_candidate_row": success_meta["candidate_row"].to_numpy(),
        "success_target": success_target,
    }
    for name, values in action_prediction.items():
        prediction_archive[f"action__{name}"] = values
    for name, values in eef_prediction.items():
        prediction_archive[f"eef__{name}"] = values
    for name, values in approach_prediction.items():
        prediction_archive[f"approach__{name}"] = values
    for name, values in success_prediction.items():
        prediction_archive[f"success__{name}"] = values
    np.savez_compressed(args.out_dir / "oof_predictions.npz", **prediction_archive)

    summary = {
        "design": {
            "tasks": len(task_data),
            "initial_states": int(
                meta[["task", "init_state_id"]].drop_duplicates().shape[0]
            ),
            "candidates": int(len(meta)),
            "candidates_per_initial_state": N_CANDIDATES,
            "query": 0,
            "crossfit": "8-fold complete-init holdout",
            "maximum_fold_fitted_components_non_action_block": args.pca_components,
            "action_coordinates_retained": 70,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_unit": "init_state nested within fixed task",
            "estimand": "transductive ranking within an observed K=32 candidate pool",
            "uncertainty_scope": (
                "cluster resampling of frozen OOF predictions; excludes model-refit "
                "and overlapping-CV training uncertainty"
            ),
            "hard_expert_ids_used": False,
        },
        "endpoint_scope": {
            "eef_delta": "0.5 s physical response, not task quality",
            "eef_target_approach": "three object-first tasks only",
            "final_success": "mixed-outcome init pools only; secondary endpoint",
            "final_success_selection": (
                "retrospective outcome-defined cohort; not prospectively deployable"
            ),
            "action_plus_route": (
                "blockwise additive comparator; same full 70-coordinate action block plus "
                "a separate 16-PC route subspace"
            ),
            "hidden_upper": (
                "artifact key for a lossy RMS/moment summary, not a true upper bound"
            ),
        },
        "metrics": metric_rows,
        "increments": increments.to_dict(orient="records"),
        "direct_geometry": geometry_summaries.to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    write_report(
        args.out_dir,
        audits_frame,
        metrics,
        increments,
        geometry_summaries,
        success_meta,
        args,
    )
    artifact_hashes = {}
    for path in sorted(args.out_dir.iterdir()):
        if path.name == "completion.json" or not path.is_file():
            continue
        artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    completion = {
        "status": "complete",
        "script": Path(__file__).name,
        "artifacts_sha256": artifact_hashes,
    }
    (args.out_dir / "completion.json").write_text(json.dumps(completion, indent=2))
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
