#!/usr/bin/env python3
"""Exploratory Early-d0 action-head and K8 coverage probe.

The source corpus repeats the same 32 flow-noise seeds in every init-state
pool.  This script therefore reports two evaluations:

1. Task-local leave-one-state-out (LOSO), which tests a practical fixed-seed
   setup but can learn a seed-specific action template.
2. A 4x8 seed-disjoint sensitivity.  For every held state and held eight-seed
   block, the head is trained on the other 15 states and other 24 seeds.  The
   target center is recomputed from those 24 allowed seeds, so held-seed action
   targets never enter training, even through a centering mean.

The feature tensors are HB-MLP/gate inputs after attention, not expert or
post-HB outputs.
This is a retrospective shadow analysis; it does not run a pruned policy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis" / "early-action-head"
ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)
K_CANDIDATES = 32
K_KEEP = 8
MONTE_CARLO_DRAWS = 10_000
MONTE_CARLO_SEED = 20260821
EARLY_POSITIONS = slice(0, 4)  # HB layer IDs 2, 3, 4, 5.
ACTION_TOKENS = slice(1, 11)
FLOW_NOISE_SHAPE = (10, 24)


@dataclass
class TaskData:
    task: str
    suite: str
    states: np.ndarray
    seeds: np.ndarray
    success: np.ndarray
    hidden_mean: np.ndarray
    router_mean: np.ndarray
    router_layers: np.ndarray
    flow_noise: np.ndarray
    actions: np.ndarray


def finite(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"non-finite result: {result}")
    return result


def first_server_rows(group: Any, episode_indices: np.ndarray) -> np.ndarray:
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    starts = np.r_[0, 1 + np.flatnonzero(episode_id[1:] != episode_id[:-1])]
    start_ids = episode_id[starts]
    if len(starts) != len(episode_indices) or len(np.unique(start_ids)) != len(starts):
        raise RuntimeError("server episodes are missing, duplicated, or non-contiguous")
    by_id = {int(episode_id[row]): int(row) for row in starts}
    if set(by_id) != set(map(int, episode_indices)):
        raise RuntimeError("server episode segmentation does not match client summaries")
    return np.asarray([by_id[int(episode)] for episode in episode_indices])


def discover_runs(hub: Path) -> list[Path]:
    runs = sorted(
        path.parent.parent
        for path in (hub / "cache" / "HiMoE-VLA").glob(
            "**/right-16x32/server/hidden.zarr"
        )
        if (path.parent.parent / "server" / "routes.zarr").exists()
        and (path.parent.parent / "client" / "summaries.json").exists()
    )
    if len(runs) != 5:
        raise RuntimeError(f"expected five complete right-16x32 runs, found {len(runs)}")
    return runs


def action_std_paths(repo: Path) -> dict[str, Path]:
    root = (
        repo
        / "himoe-vla-cache"
        / "himoe-libero-bridge"
        / "cache"
        / "checkpoints"
    )
    return {
        "libero_goal": root
        / "HiMoE-VLA-Libero-Goal/libero_goal_no_noops/meta/stats.json",
        "libero_spatial": root
        / "HiMoE-VLA-Libero-Spatial/libero_spatial_no_noops/meta/stats.json",
        # The selected libero_long task is served by the Libero-10 checkpoint.
        "libero_long": root
        / "HiMoE-VLA-Libero-10/libero_10_no_noops/meta/stats.json",
    }


def load_action_stds(repo: Path) -> dict[str, np.ndarray]:
    result = {}
    for suite, path in action_std_paths(repo).items():
        stats = json.loads(path.read_text())
        value = np.asarray(stats["actions"]["std"], dtype=np.float64)
        if value.shape != (7,) or np.any(value <= 0) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid action std in {path}")
        result[suite] = value
    return result


def load_task(
    run: Path,
    cache_root: Path,
    expected_action_std: np.ndarray,
) -> TaskData:
    summaries = sorted(
        json.loads((run / "client" / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 16 * K_CANDIDATES:
        raise RuntimeError(f"unexpected episode count in {run}")
    episode_indices = np.asarray(
        [int(row["episode_index"]) for row in summaries], dtype=np.int64
    )
    if not np.array_equal(episode_indices, np.arange(512, dtype=np.int64)):
        raise RuntimeError(f"episode_index is not the unique range 0..511 in {run}")
    hidden = zarr.open_group(str(run / "server" / "hidden.zarr"), mode="r")
    routes = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
    hidden_episode_axis = np.asarray(hidden["episode_id"][:], dtype=np.int64)
    route_episode_axis = np.asarray(routes["episode_id"][:], dtype=np.int64)
    hidden_control_axis = np.asarray(hidden["control_step"][:], dtype=np.int64)
    route_control_axis = np.asarray(routes["control_step"][:], dtype=np.int64)
    if not np.array_equal(hidden_episode_axis, route_episode_axis) or not np.array_equal(
        hidden_control_axis, route_control_axis
    ):
        raise RuntimeError(f"hidden/routes full identity axes differ in {run}")
    if not np.array_equal(
        hidden_control_axis, np.arange(len(hidden_control_axis), dtype=np.int64)
    ):
        raise RuntimeError(f"control_step is not the global contiguous row index in {run}")
    rows = first_server_rows(hidden, episode_indices)
    selected_hidden_episode = np.asarray(
        hidden["episode_id"].oindex[rows], dtype=np.int64
    )
    selected_route_episode = np.asarray(
        routes["episode_id"].oindex[rows], dtype=np.int64
    )
    if not np.array_equal(selected_hidden_episode, episode_indices) or not np.array_equal(
        selected_route_episode, episode_indices
    ):
        raise RuntimeError(f"hidden/routes row mismatch in {run}")
    selected_hidden_control = np.asarray(
        hidden["control_step"].oindex[rows], dtype=np.int64
    )
    selected_route_control = np.asarray(
        routes["control_step"].oindex[rows], dtype=np.int64
    )
    if not np.array_equal(selected_route_control, selected_hidden_control) or not np.array_equal(
        selected_hidden_control, rows
    ):
        raise RuntimeError(f"hidden/routes first-control alignment failed in {run}")
    metadata = json.loads((run / "client" / "server_metadata.json").read_text())
    declared_action_std = np.asarray(
        metadata.get("normalization_action_std"), dtype=np.float64
    )
    if declared_action_std.shape != (7,) or not np.allclose(
        declared_action_std,
        expected_action_std,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            f"checkpoint stats disagree with server metadata for {run}: "
            f"{declared_action_std} vs {expected_action_std}"
        )
    captured_layers = tuple(map(int, metadata.get("routing_hb_layer_indices", ())))
    if captured_layers[:4] != (2, 3, 4, 5):
        raise RuntimeError(f"first four HB slots are not layers 2-5 in {run}")

    hidden_tensor = np.asarray(
        hidden["hb_hidden"].oindex[
            rows, EARLY_POSITIONS, 0, ACTION_TOKENS, :
        ],
        dtype=np.float32,
    )
    router_tensor = np.asarray(
        routes["hb_router_probs"].oindex[
            rows, EARLY_POSITIONS, 0, ACTION_TOKENS, :
        ],
        dtype=np.float32,
    )
    if hidden_tensor.shape != (512, 4, 10, 1024):
        raise RuntimeError(f"unexpected hidden slice in {run}: {hidden_tensor.shape}")
    if router_tensor.shape != (512, 4, 10, 32):
        raise RuntimeError(f"unexpected router slice in {run}: {router_tensor.shape}")
    if not np.all(np.isfinite(hidden_tensor)) or not np.all(np.isfinite(router_tensor)):
        raise RuntimeError(f"non-finite hidden/router value in {run}")
    # The stored full softmax probabilities are float16.  Remove any impossible
    # negative round-off and restore unit mass at every routing site before
    # pooling; otherwise fp16 mass error becomes part of the feature.
    router_tensor = np.maximum(router_tensor, 0.0)
    router_mass = router_tensor.sum(axis=-1, keepdims=True, dtype=np.float32)
    if np.any(router_mass <= 0.0):
        raise RuntimeError(f"zero router probability mass in {run}")
    router_tensor /= router_mass
    if not np.allclose(router_tensor.sum(axis=-1), 1.0, atol=2e-6, rtol=0.0):
        raise RuntimeError(f"router probability normalization failed in {run}")

    actions = []
    for episode in episode_indices:
        path = run / "client" / f"episode_{int(episode):02d}.npz"
        with np.load(path) as payload:
            action = np.asarray(payload["actions"][0], dtype=np.float32)
        if action.shape != (10, 7) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"invalid actions[0] in {path}")
        actions.append(action)

    relative = run.relative_to(cache_root)
    suite = relative.parts[0]
    return TaskData(
        task=str(relative.parent),
        suite=suite,
        states=np.asarray([int(row["init_state_id"]) for row in summaries]),
        seeds=np.asarray([int(row["flow_noise_seed"]) for row in summaries]),
        success=np.asarray([bool(row["success"]) for row in summaries]),
        # Candidate-level mean features requested by the exploratory protocol.
        hidden_mean=hidden_tensor.mean(axis=(1, 2), dtype=np.float32),
        router_mean=router_tensor.mean(axis=(1, 2), dtype=np.float32),
        # Preserve HB2-5 identity while averaging the ten action tokens.
        router_layers=router_tensor.mean(axis=2, dtype=np.float32).reshape(512, 128),
        flow_noise=np.stack(
            [
                np.random.default_rng(int(row["flow_noise_seed"]))
                .standard_normal(FLOW_NOISE_SHAPE)
                .astype(np.float32)
                for row in summaries
            ]
        ),
        actions=np.stack(actions),
    )


def pool_array(task: TaskData, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.unique(task.states)
    values = np.asarray(getattr(task, field))
    pools = []
    seed_rows = []
    for state in states:
        indices = np.flatnonzero(task.states == state)
        indices = indices[np.argsort(task.seeds[indices])]
        if len(indices) != K_CANDIDATES:
            raise RuntimeError(f"{task.task} state {state} is not K32")
        pools.append(values[indices])
        seed_rows.append(task.seeds[indices])
    seed_rows = np.stack(seed_rows)
    if not np.all(seed_rows == seed_rows[:1]):
        raise RuntimeError(f"seed order differs across pools in {task.task}")
    return np.stack(pools), states, seed_rows[0]


def center_scale_features(values: np.ndarray) -> np.ndarray:
    """Candidate-center and scalar-RMS-normalize each pool using features only."""
    array = np.asarray(values, dtype=np.float64)
    array = array - array.mean(axis=1, keepdims=True)
    scale = np.sqrt(np.mean(np.square(array), axis=(1, 2), keepdims=True))
    return array / np.maximum(scale, 1e-12)


def center_targets(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array - array.mean(axis=1, keepdims=True)


def block_krr_cv_residuals(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    projected_targets: np.ndarray,
    valid: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Exact held-block residuals from one full-kernel eigendecomposition."""
    inverse = 1.0 / (eigenvalues + float(alpha))
    dual = eigenvectors @ (projected_targets * inverse[:, None])
    valid_vectors = eigenvectors[valid]
    inverse_block = (valid_vectors * inverse[None, :]) @ valid_vectors.T
    return np.linalg.solve(inverse_block, dual[valid])


def tune_and_fit_head(
    feature_groups: np.ndarray,
    target_groups: np.ndarray,
    test_features: np.ndarray,
    inner_folds: int,
) -> tuple[np.ndarray, float]:
    groups, candidates, width = feature_groups.shape
    if groups < inner_folds:
        raise ValueError("fewer training groups than inner folds")
    x = feature_groups.reshape(groups * candidates, width)
    y = target_groups.reshape(groups * candidates, -1)
    kernel = (x @ x.T) / width
    eigenvalues, eigenvectors = eigh(
        (kernel + kernel.T) * 0.5,
        check_finite=False,
        driver="evd",
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ y
    group_id = np.repeat(np.arange(groups), candidates)
    losses = np.zeros(len(ALPHAS), dtype=np.float64)
    folds = [
        np.flatnonzero(np.isin(group_id, np.arange(groups)[fold::inner_folds]))
        for fold in range(inner_folds)
    ]
    for index, alpha in enumerate(ALPHAS):
        for valid in folds:
            residual = block_krr_cv_residuals(
                eigenvalues, eigenvectors, projected, valid, alpha
            )
            losses[index] += np.sum(np.square(residual))
    selected = float(ALPHAS[int(np.argmin(losses))])
    inverse = 1.0 / (eigenvalues + selected)
    dual = eigenvectors @ (projected * inverse[:, None])
    test_kernel = (np.asarray(test_features, dtype=np.float64) @ x.T) / width
    prediction = test_kernel @ dual
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("kernel ridge produced non-finite predictions")
    return prediction, selected


def primary_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    inner_folds: int,
) -> tuple[np.ndarray, list[float]]:
    groups = len(features)
    predictions = np.zeros_like(targets, dtype=np.float64)
    selected_alphas = []
    for outer in range(groups):
        keep = np.arange(groups) != outer
        predictions[outer], alpha = tune_and_fit_head(
            features[keep], targets[keep], features[outer], inner_folds
        )
        selected_alphas.append(alpha)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("primary head produced non-finite predictions")
    return predictions, selected_alphas


def seed_disjoint_predictions(
    features: np.ndarray,
    raw_targets: np.ndarray,
    seed_folds: list[np.ndarray],
    inner_folds: int,
) -> tuple[np.ndarray, list[float]]:
    """Cross-fit every (state, seed block) with both identities absent from train."""
    groups, candidates, _ = features.shape
    if candidates != K_CANDIDATES:
        raise ValueError("seed-disjoint protocol requires K32")
    predictions = np.zeros_like(raw_targets, dtype=np.float64)
    selected_alphas = []
    all_seeds = np.arange(candidates)
    for outer in range(groups):
        train_groups = np.arange(groups) != outer
        for held in seed_folds:
            allowed = np.setdiff1d(all_seeds, held, assume_unique=True)
            # X centering is feature-only and transductive over all K32.  The
            # target center below is recomputed from only the 24 allowed seeds.
            train_x = features[train_groups][:, allowed]
            train_y = center_targets(raw_targets[train_groups][:, allowed])
            predictions[outer, held], alpha = tune_and_fit_head(
                train_x, train_y, features[outer, held], inner_folds
            )
            selected_alphas.append(alpha)
    # The four heads have centered targets but independent finite-sample offsets.
    # This common center is only a numerical convention.  Evaluation never uses
    # cross-block predicted distances; it uses within-held8 geometry only.
    predictions = center_targets(predictions)
    if not np.all(np.isfinite(predictions)):
        raise RuntimeError("seed-disjoint head produced non-finite predictions")
    return predictions, selected_alphas


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    squared_norm = np.einsum("nd,nd->n", flat, flat, optimize=True)
    distance2 = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (flat @ flat.T)
    ) / flat.shape[1]
    return np.sqrt(np.maximum(distance2, 0.0))


def distance_correlation(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    upper = np.triu_indices(len(score), 1)
    return {
        "spearman": finite(spearmanr(score[upper], target[upper]).statistic),
        "pearson": finite(pearsonr(score[upper], target[upper]).statistic),
    }


def pam(distance: np.ndarray, keep: int) -> np.ndarray:
    """Deterministic PAM BUILD+SWAP, resolving numerical ties by candidate ID."""
    matrix = np.asarray(distance, dtype=np.float64)
    medoids: list[int] = []
    current = np.full(len(matrix), np.inf)
    for _ in range(keep):
        options = [
            (finite(np.minimum(current, matrix[:, candidate]).sum()), candidate)
            for candidate in range(len(matrix))
            if candidate not in medoids
        ]
        _, selected = min(options)
        medoids.append(selected)
        current = np.minimum(current, matrix[:, selected])

    while True:
        base = finite(np.min(matrix[:, medoids], axis=1).sum())
        best_cost = base
        best_swap: tuple[int, int] | None = None
        for old in medoids:
            for candidate in range(len(matrix)):
                if candidate in medoids:
                    continue
                proposed = medoids.copy()
                proposed[proposed.index(old)] = candidate
                cost = finite(np.min(matrix[:, proposed], axis=1).sum())
                tie_is_better = (
                    best_swap is not None
                    and abs(cost - best_cost) <= 1e-12
                    and (old, candidate) < best_swap
                )
                if cost < best_cost - 1e-12 or tie_is_better:
                    best_cost = cost
                    best_swap = (old, candidate)
        if best_swap is None:
            break
        medoids[medoids.index(best_swap[0])] = best_swap[1]
    result = np.asarray(sorted(medoids), dtype=np.int64)
    if result.shape != (keep,) or len(np.unique(result)) != keep:
        raise RuntimeError("PAM did not return the requested number of unique medoids")
    return result


def medoid_index(distance: np.ndarray, candidates: np.ndarray | None = None) -> int:
    matrix = np.asarray(distance, dtype=np.float64)
    indices = (
        np.arange(len(matrix), dtype=np.int64)
        if candidates is None
        else np.sort(np.unique(np.asarray(candidates, dtype=np.int64)))
    )
    if not len(indices):
        raise ValueError("medoid candidate set is empty")
    scores = matrix[np.ix_(indices, indices)].sum(axis=1)
    return int(indices[int(np.argmin(scores))])


def vote_metrics(action_distance: np.ndarray, selected: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(action_distance, dtype=np.float64)
    selected = np.sort(np.unique(np.asarray(selected, dtype=np.int64)))
    full = medoid_index(matrix)
    voted = medoid_index(matrix, selected)
    global_centrality = matrix.mean(axis=1)
    scale = finite(np.median(matrix[np.triu_indices(len(matrix), 1)]))
    if scale <= 0.0:
        raise ValueError("action pool has no positive median pair distance")
    regret = max(finite((global_centrality[voted] - global_centrality[full]) / scale), 0.0)
    return {
        "normalized_global_medoid_regret": regret,
        "exact_global_medoid_reproduction": float(voted == full),
        "voted_candidate": voted,
        "global_medoid_candidate": full,
    }


def common_random_subsets(seed_folds: list[np.ndarray]) -> dict[str, np.ndarray]:
    uniform_rng = np.random.default_rng(MONTE_CARLO_SEED)
    stratified_rng = np.random.default_rng(MONTE_CARLO_SEED + 1)
    uniform = np.stack(
        [
            np.sort(uniform_rng.choice(K_CANDIDATES, K_KEEP, replace=False))
            for _ in range(MONTE_CARLO_DRAWS)
        ]
    )
    keep_per_fold = K_KEEP // len(seed_folds)
    stratified = np.stack(
        [
            np.sort(
                np.concatenate(
                    [
                        stratified_rng.choice(fold, keep_per_fold, replace=False)
                        for fold in seed_folds
                    ]
                )
            )
            for _ in range(MONTE_CARLO_DRAWS)
        ]
    )
    return {"uniform": uniform, "stratified": stratified}


def monte_carlo_vote_metrics(
    action_distance: np.ndarray,
    subsets: np.ndarray,
) -> dict[str, float]:
    matrix = np.asarray(action_distance, dtype=np.float64)
    draws = np.asarray(subsets, dtype=np.int64)
    local = matrix[draws[:, :, None], draws[:, None, :]]
    local_winner = np.argmin(local.sum(axis=2), axis=1)
    voted = draws[np.arange(len(draws)), local_winner]
    full = medoid_index(matrix)
    global_centrality = matrix.mean(axis=1)
    scale = finite(np.median(matrix[np.triu_indices(len(matrix), 1)]))
    regret = np.maximum((global_centrality[voted] - global_centrality[full]) / scale, 0.0)
    return {
        "normalized_global_medoid_regret": finite(regret.mean()),
        "exact_global_medoid_reproduction": finite(np.mean(voted == full)),
    }


def exact_random_coverage(distance: np.ndarray, keep: int) -> float:
    """Exact expected mean nearest-action distance for a uniform subset."""
    matrix = np.asarray(distance, dtype=np.float64)
    candidates = len(matrix)
    denominator = math.comb(candidates, keep)
    expected = 0.0
    for row in matrix:
        ordered = np.sort(row)
        expected += sum(
            ordered[rank]
            * math.comb(candidates - rank - 1, keep - 1)
            / denominator
            for rank in range(candidates - keep + 1)
        )
    return finite(expected / candidates)


def exact_stratified_random_coverage(
    distance: np.ndarray,
    folds: list[np.ndarray],
    keep_per_fold: int,
) -> float:
    """Exact coverage expectation when each fixed fold contributes equal K.

    For each target candidate, the survival probability of its nearest selected
    distance factorizes across folds.  Integrating that discrete survival curve
    avoids enumerating C(8,2)^4 subsets.
    """
    matrix = np.asarray(distance, dtype=np.float64)
    expected = 0.0
    for row in matrix:
        levels = np.unique(row)
        value = finite(levels[0])
        for lower, upper in zip(levels[:-1], levels[1:]):
            survival = 1.0
            for fold in folds:
                excluded = int(np.count_nonzero(row[fold] <= lower))
                remaining = len(fold) - excluded
                survival *= (
                    math.comb(remaining, keep_per_fold)
                    / math.comb(len(fold), keep_per_fold)
                    if remaining >= keep_per_fold
                    else 0.0
                )
            value += finite((upper - lower) * survival)
        expected += value
    return finite(expected / len(matrix))


def random_success_hit(success_count: int, keep: int = K_KEEP) -> float:
    failures = K_CANDIDATES - success_count
    miss = (
        math.comb(failures, keep) / math.comb(K_CANDIDATES, keep)
        if keep <= failures
        else 0.0
    )
    return finite(1.0 - miss)


def stratified_random_success_hit(
    success: np.ndarray,
    folds: list[np.ndarray],
    keep_per_fold: int,
) -> float:
    miss = 1.0
    for fold in folds:
        failures = int((~np.asarray(success, dtype=bool)[fold]).sum())
        miss *= (
            math.comb(failures, keep_per_fold)
            / math.comb(len(fold), keep_per_fold)
            if failures >= keep_per_fold
            else 0.0
        )
    return finite(1.0 - miss)


def evaluate_method(
    score_distance: np.ndarray,
    action_distance: np.ndarray,
    success: np.ndarray,
) -> dict[str, Any]:
    selected = pam(score_distance, K_KEEP)
    success_count = int(success.sum())
    return {
        **distance_correlation(score_distance, action_distance),
        "coverage": finite(np.min(action_distance[:, selected], axis=1).mean()),
        "vote": vote_metrics(action_distance, selected),
        "success_hit": float(np.any(success[selected])),
        "success_recall": (
            finite(success[selected].sum() / success_count)
            if success_count
            else None
        ),
        "selected": selected.tolist(),
    }


def evaluate_seed_disjoint_method(
    score_distance: np.ndarray,
    action_distance: np.ndarray,
    success: np.ndarray,
    folds: list[np.ndarray],
) -> dict[str, Any]:
    keep_per_fold = K_KEEP // len(folds)
    selected_parts = []
    block_correlations = []
    for fold in folds:
        local = pam(score_distance[np.ix_(fold, fold)], keep_per_fold)
        selected_parts.append(fold[local])
        block_correlations.append(
            distance_correlation(
                score_distance[np.ix_(fold, fold)],
                action_distance[np.ix_(fold, fold)],
            )["spearman"]
        )
    selected = np.sort(np.concatenate(selected_parts))
    if selected.shape != (K_KEEP,) or len(np.unique(selected)) != K_KEEP:
        raise RuntimeError("stratified PAM did not produce unique K8")
    success_count = int(success.sum())
    return {
        "within_held8_spearman": block_correlations,
        "coverage": finite(np.min(action_distance[:, selected], axis=1).mean()),
        "vote": vote_metrics(action_distance, selected),
        "success_hit": float(np.any(success[selected])),
        "success_recall": (
            finite(success[selected].sum() / success_count)
            if success_count
            else None
        ),
        "selected": selected.tolist(),
        "selection": "PAM K2 independently within each held8 fold; union K8",
    }


def evaluate_task(
    task: TaskData,
    action_std: np.ndarray,
    inner_folds: int,
    seed_fold_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hidden_raw, states, seeds = pool_array(task, "hidden_mean")
    router_raw, _, _ = pool_array(task, "router_mean")
    router_layers_raw, _, _ = pool_array(task, "router_layers")
    noise24_raw, _, _ = pool_array(task, "flow_noise")
    action_raw, _, _ = pool_array(task, "actions")
    success, _, _ = pool_array(task, "success")
    actions = (action_raw / action_std[None, None, None, :]).reshape(16, 32, 70)
    action_targets = center_targets(actions)
    hidden = center_scale_features(hidden_raw)
    router = center_scale_features(router_raw)
    router_layers = center_scale_features(router_layers_raw)
    noise24 = center_scale_features(noise24_raw.reshape(16, 32, -1))
    noise7 = center_scale_features(noise24_raw[:, :, :, :7].reshape(16, 32, -1))

    noise_head, noise_alphas = primary_predictions(noise24, action_targets, inner_folds)
    hidden_head, hidden_alphas = primary_predictions(
        hidden, action_targets, inner_folds
    )
    router_head, router_alphas = primary_predictions(
        router, action_targets, inner_folds
    )
    router_layers_head, router_layers_alphas = primary_predictions(
        router_layers, action_targets, inner_folds
    )
    seed_template = np.zeros_like(action_targets)
    for outer in range(len(states)):
        seed_template[outer] = action_targets[np.arange(len(states)) != outer].mean(
            axis=0
        )

    seed_folds = [
        np.arange(K_CANDIDATES, dtype=np.int64)[fold::seed_fold_count]
        for fold in range(seed_fold_count)
    ]
    if any(len(fold) != K_CANDIDATES // seed_fold_count for fold in seed_folds):
        raise ValueError("seed folds must divide K32 evenly")
    random_subsets = common_random_subsets(seed_folds)
    hidden_disjoint, hidden_disjoint_alphas = seed_disjoint_predictions(
        hidden, actions, seed_folds, inner_folds
    )
    router_disjoint, router_disjoint_alphas = seed_disjoint_predictions(
        router, actions, seed_folds, inner_folds
    )
    router_layers_disjoint, router_layers_disjoint_alphas = seed_disjoint_predictions(
        router_layers, actions, seed_folds, inner_folds
    )
    noise_disjoint, noise_disjoint_alphas = seed_disjoint_predictions(
        noise24, actions, seed_folds, inner_folds
    )

    rows = []
    for pool, state in enumerate(states):
        action_distance = pairwise_rms(actions[pool])
        random_coverage = exact_random_coverage(action_distance, K_KEEP)
        success_count = int(success[pool].sum())
        method_distances = {
            "raw_noise24": pairwise_rms(noise24[pool]),
            "raw_noise7": pairwise_rms(noise7[pool]),
            "raw_hidden": pairwise_rms(hidden[pool]),
            "raw_router": pairwise_rms(router[pool]),
            "raw_router_layers": pairwise_rms(router_layers[pool]),
            "seed_template": pairwise_rms(seed_template[pool]),
            "head_noise24": pairwise_rms(noise_head[pool]),
            "head_hidden": pairwise_rms(hidden_head[pool]),
            "head_router": pairwise_rms(router_head[pool]),
            "head_router_layers": pairwise_rms(router_layers_head[pool]),
            "head_noise24_seed_disjoint": pairwise_rms(noise_disjoint[pool]),
            "head_hidden_seed_disjoint": pairwise_rms(hidden_disjoint[pool]),
            "head_router_seed_disjoint": pairwise_rms(router_disjoint[pool]),
            "head_router_layers_seed_disjoint": pairwise_rms(
                router_layers_disjoint[pool]
            ),
            "oracle_action": action_distance,
        }
        seed_disjoint_names = {
            "head_noise24_seed_disjoint",
            "head_hidden_seed_disjoint",
            "head_router_seed_disjoint",
            "head_router_layers_seed_disjoint",
        }
        methods = {
            name: evaluate_method(distance, action_distance, success[pool])
            for name, distance in method_distances.items()
            if name not in seed_disjoint_names
        }
        methods.update(
            {
                name: evaluate_seed_disjoint_method(
                    method_distances[name],
                    action_distance,
                    success[pool],
                    seed_folds,
                )
                for name in seed_disjoint_names
            }
        )
        stratified_coverage = exact_stratified_random_coverage(
            action_distance, seed_folds, K_KEEP // len(seed_folds)
        )
        rows.append(
            {
                "task": task.task,
                "suite": task.suite,
                "state": int(state),
                "success_count": success_count,
                "random": {
                    "coverage": random_coverage,
                    "vote": monte_carlo_vote_metrics(
                        action_distance, random_subsets["uniform"]
                    ),
                    "success_hit": random_success_hit(success_count),
                    "success_recall": 0.25 if success_count else None,
                },
                "stratified_random": {
                    "coverage": stratified_coverage,
                    "vote": monte_carlo_vote_metrics(
                        action_distance, random_subsets["stratified"]
                    ),
                    "success_hit": stratified_random_success_hit(
                        success[pool], seed_folds, K_KEEP // len(seed_folds)
                    ),
                    "success_recall": 0.25 if success_count else None,
                    "selection": "exact expectation for two uniform candidates from each fixed held8 fold",
                },
                "methods": methods,
            }
        )

    diagnostics = {
        "task": task.task,
        "suite": task.suite,
        "states": states.tolist(),
        "seeds": seeds.tolist(),
        "seed_folds": [[int(seeds[index]) for index in fold] for fold in seed_folds],
        "alpha_counts": {
            "head_noise24": dict(sorted(Counter(map(str, noise_alphas)).items())),
            "head_hidden": dict(sorted(Counter(map(str, hidden_alphas)).items())),
            "head_router": dict(sorted(Counter(map(str, router_alphas)).items())),
            "head_router_layers": dict(
                sorted(Counter(map(str, router_layers_alphas)).items())
            ),
            "head_noise24_seed_disjoint": dict(
                sorted(Counter(map(str, noise_disjoint_alphas)).items())
            ),
            "head_hidden_seed_disjoint": dict(
                sorted(Counter(map(str, hidden_disjoint_alphas)).items())
            ),
            "head_router_seed_disjoint": dict(
                sorted(Counter(map(str, router_disjoint_alphas)).items())
            ),
            "head_router_layers_seed_disjoint": dict(
                sorted(Counter(map(str, router_layers_disjoint_alphas)).items())
            ),
        },
    }
    return rows, diagnostics


METHODS = (
    "raw_noise24",
    "raw_noise7",
    "raw_hidden",
    "raw_router",
    "raw_router_layers",
    "seed_template",
    "head_noise24",
    "head_hidden",
    "head_router",
    "head_router_layers",
    "head_noise24_seed_disjoint",
    "head_hidden_seed_disjoint",
    "head_router_seed_disjoint",
    "head_router_layers_seed_disjoint",
    "oracle_action",
)
SEED_DISJOINT_METHODS = (
    "head_noise24_seed_disjoint",
    "head_hidden_seed_disjoint",
    "head_router_seed_disjoint",
    "head_router_layers_seed_disjoint",
)
PRIMARY_METHODS = tuple(name for name in METHODS if name not in SEED_DISJOINT_METHODS)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = {}
    for name in METHODS:
        values = [row["methods"][name] for row in rows]
        recalls = [value["success_recall"] for value in values]
        recalls = [value for value in recalls if value is not None]
        baseline = "stratified_random" if name in SEED_DISJOINT_METHODS else "random"
        payload = {
            "coverage_mean": finite(np.mean([v["coverage"] for v in values])),
            "coverage_over_matching_random_mean": finite(
                np.mean(
                    [
                        value["coverage"] / row[baseline]["coverage"]
                        for value, row in zip(values, rows)
                    ]
                )
            ),
            "random_baseline": baseline,
            "success_hit_mean": finite(np.mean([v["success_hit"] for v in values])),
            "success_recall_mean": finite(np.mean(recalls)),
            "vote_normalized_global_medoid_regret_mean": finite(
                np.mean(
                    [v["vote"]["normalized_global_medoid_regret"] for v in values]
                )
            ),
            "vote_exact_global_medoid_reproduction_mean": finite(
                np.mean(
                    [
                        v["vote"]["exact_global_medoid_reproduction"]
                        for v in values
                    ]
                )
            ),
        }
        if "spearman" in values[0]:
            payload.update(
                {
                    "spearman_median": finite(
                        np.median([v["spearman"] for v in values])
                    ),
                    "spearman_mean": finite(
                        np.mean([v["spearman"] for v in values])
                    ),
                    "pearson_median": finite(
                        np.median([v["pearson"] for v in values])
                    ),
                }
            )
        else:
            block_values = [
                value
                for row_value in values
                for value in row_value["within_held8_spearman"]
            ]
            payload["within_held8_spearman_median"] = finite(
                np.median(block_values)
            )
            payload["within_held8_spearman_mean"] = finite(np.mean(block_values))
        methods[name] = payload

    random_recalls = [row["random"]["success_recall"] for row in rows]
    random_recalls = [value for value in random_recalls if value is not None]
    methods["exact_random"] = {
        "coverage_mean": finite(np.mean([r["random"]["coverage"] for r in rows])),
        "coverage_over_matching_random_mean": 1.0,
        "random_baseline": "random",
        "success_hit_mean": finite(
            np.mean([r["random"]["success_hit"] for r in rows])
        ),
        "success_recall_mean": finite(np.mean(random_recalls)),
        "vote_normalized_global_medoid_regret_mean": finite(
            np.mean(
                [
                    r["random"]["vote"]["normalized_global_medoid_regret"]
                    for r in rows
                ]
            )
        ),
        "vote_exact_global_medoid_reproduction_mean": finite(
            np.mean(
                [
                    r["random"]["vote"]["exact_global_medoid_reproduction"]
                    for r in rows
                ]
            )
        ),
    }
    methods["exact_stratified_random"] = {
        "coverage_mean": finite(
            np.mean([r["stratified_random"]["coverage"] for r in rows])
        ),
        "coverage_over_matching_random_mean": 1.0,
        "random_baseline": "stratified_random",
        "success_hit_mean": finite(
            np.mean([r["stratified_random"]["success_hit"] for r in rows])
        ),
        "success_recall_mean": finite(np.mean(random_recalls)),
        "vote_normalized_global_medoid_regret_mean": finite(
            np.mean(
                [
                    r["stratified_random"]["vote"][
                        "normalized_global_medoid_regret"
                    ]
                    for r in rows
                ]
            )
        ),
        "vote_exact_global_medoid_reproduction_mean": finite(
            np.mean(
                [
                    r["stratified_random"]["vote"][
                        "exact_global_medoid_reproduction"
                    ]
                    for r in rows
                ]
            )
        ),
    }
    return {"pool_count": len(rows), "methods": methods}


def report_markdown(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    all_methods = summary["all80"]["methods"]
    balanced = summary["balanced16"]["methods"]
    lines = [
        "# Early d0 action-head probe",
        "",
        "## Result",
        "",
        (
            "The task-local hidden head reconstructs candidate action geometry, "
            "but the repeated-seed template explains nearly all of that result. "
            "The seed-disjoint sensitivity is the relevant check for a reusable "
            "MoE-derived signal."
        ),
        "",
        "## All 80 state pools",
        "",
        "| method | distance Spearman (median) | K8 coverage / exact random |",
        "|---|---:|---:|",
    ]
    labels = {
        "raw_noise24": "raw initial flow noise (10x24)",
        "raw_noise7": "raw initial flow noise (live 10x7)",
        "raw_hidden": "raw Early hidden mean",
        "raw_router": "raw Early router (layers pooled, 32)",
        "raw_router_layers": "raw Early router (layers retained, 4x32)",
        "seed_template": "seed-only action template",
        "head_noise24": "initial-noise linear head (state LOSO)",
        "head_hidden": "hidden linear head (state LOSO)",
        "head_router": "router-32 linear head (state LOSO)",
        "head_router_layers": "router-4x32 linear head (state LOSO)",
        "head_noise24_seed_disjoint": "noise head (state + seed disjoint)",
        "head_hidden_seed_disjoint": "hidden head (state + seed disjoint)",
        "head_router_seed_disjoint": "router-32 head (state + seed disjoint)",
        "head_router_layers_seed_disjoint": "router-4x32 head (state + seed disjoint)",
        "oracle_action": "true-action PAM oracle",
    }
    for method in PRIMARY_METHODS:
        value = all_methods[method]
        lines.append(
            f"| {labels[method]} | {value['spearman_median']:.3f} | "
            f"{value['coverage_over_matching_random_mean']:.3f} |"
        )
    lines.extend(
        [
            "| exact uniform random K8 | - | 1.000 |",
            "",
            "## State + seed-disjoint sensitivity",
            "",
            "The four heads have different target centers, so cross-block "
            "distances are not calibrated. Correlation is therefore computed "
            "only inside each held-eight block. Selection uses PAM K2 inside "
            "each block and unions the four pairs into K8.",
            "",
            "| method | within-held8 Spearman (median) | stratified K8 coverage / exact stratified random |",
            "|---|---:|---:|",
        ]
    )
    for method in SEED_DISJOINT_METHODS:
        value = all_methods[method]
        lines.append(
            f"| {labels[method]} | {value['within_held8_spearman_median']:.3f} | "
            f"{value['coverage_over_matching_random_mean']:.3f} |"
        )
    lines.extend(
        [
            "| exact random: two from each held8 | - | 1.000 |",
            "",
            "## K8 action voting",
            "",
            "After pruning, the final vote is the ordinary unweighted medoid "
            "of the eight true completed action chunks. Regret is measured by "
            "its K32-global centrality gap divided by the pool median pair distance.",
            "",
            "| method | normalized global-medoid regret | exact K32 medoid reproduction |",
            "|---|---:|---:|",
        ]
    )
    for method in (*METHODS, "exact_random", "exact_stratified_random"):
        value = all_methods[method]
        label = labels.get(
            method,
            (
                "MC random: two from each held8"
                if method == "exact_stratified_random"
                else "MC uniform random K8"
            ),
        )
        lines.append(
            f"| {label} | "
            f"{value['vote_normalized_global_medoid_regret_mean']:.3f} | "
            f"{value['vote_exact_global_medoid_reproduction_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Random voting baselines use {MONTE_CARLO_DRAWS:,} common subsets "
            f"per pool (seeds {MONTE_CARLO_SEED} and {MONTE_CARLO_SEED + 1}); "
            "coverage and descriptive success retention remain exact, not Monte Carlo.",
            "",
            "## Balanced-min4 16 pools: descriptive success retention",
            "",
            "| method | any success retained | successful-candidate recall |",
            "|---|---:|---:|",
        ]
    )
    for method in (*METHODS, "exact_random", "exact_stratified_random"):
        value = balanced[method]
        label = labels.get(
            method,
            (
                "exact random: two from each held8"
                if method == "exact_stratified_random"
                else "exact uniform random K8"
            ),
        )
        lines.append(
            f"| {label} | {value['success_hit_mean']:.3f} | "
            f"{value['success_recall_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Here `balanced-min4` means 4-28 successes among K32, not a 16/16 "
            "split. Success is the eventual episode outcome and this saturated, "
            "descriptive retention check does not label the first chunk as correct. "
            "Success is never used to train a head or choose a medoid. The "
            "true-action oracle is an action-coverage oracle, not a success oracle.",
            "",
            "## Protocol",
            "",
            "- Target: the complete client `actions[0]` 10x7 chunk, divided by "
            "the official checkpoint action std for each of Goal, Spatial, or Libero-10.",
            "- Features: first server row, d0, ten action tokens, mean over HB "
            "layers 2-5 and tokens. Hidden is 1024-D. Router is tested both "
            "after pooling layers to 32-D and with layer identity retained as 4x32.",
            "- Noise baseline: the exact first request noise is reconstructed as "
            "NumPy PCG64 `standard_normal((10,24))`, cast to float32. Both all "
            "24 model dimensions and the seven live action dimensions are shown.",
            "- Semantics: captured hidden values are HB-MLP/gate inputs after "
            "layer attention, not expert outputs or post-HB representations.",
            "- Main CV: a complete K32 init-state pool is held out within each "
            "task/checkpoint. Ridge alpha is chosen by inner state-group CV.",
            "- Seed sensitivity: seeds are split by sorted position modulo four. "
            "Each held eight-seed block and held state are absent from training. "
            "Target centering uses only the 24 allowed seeds; feature centering "
            "uses all K32 and is explicitly transductive and seed-label-disjoint, "
            "not strictly feature-disjoint.",
            "- Coverage: mean true-action distance to the nearest selected K8 "
            "medoid. Random coverage is its exact combinatorial expectation.",
            "- Voting: choose the ordinary unweighted medoid among the selected "
            "eight true action chunks. No cluster weights are used. Random vote "
            "endpoints use fixed common-subset Monte Carlo; coverage stays exact.",
            "- Scope: exploratory shadow analysis only; no candidates were "
            "actually stopped early and no closed-loop latency or success was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    hub = args.hub_root.resolve()
    repo = HERE.parent
    cache_root = hub / "cache" / "HiMoE-VLA"
    action_stds = load_action_stds(repo)
    rows: list[dict[str, Any]] = []
    diagnostics = []
    canonical_seeds: tuple[int, ...] | None = None
    for index, run in enumerate(discover_runs(hub), start=1):
        print(f"[{index}/5] loading {run.relative_to(cache_root)}", flush=True)
        suite = run.relative_to(cache_root).parts[0]
        task = load_task(run, cache_root, action_stds[suite])
        _, _, seeds = pool_array(task, "actions")
        seed_tuple = tuple(map(int, seeds))
        if canonical_seeds is None:
            canonical_seeds = seed_tuple
        elif canonical_seeds != seed_tuple:
            raise RuntimeError("the five tasks do not share one canonical seed set")
        task_rows, task_diagnostics = evaluate_task(
            task,
            action_stds[task.suite],
            args.inner_folds,
            args.seed_folds,
        )
        rows.extend(task_rows)
        diagnostics.append(task_diagnostics)
        print(f"[{index}/5] evaluated {task.task}", flush=True)

    subsets = {
        "all80": rows,
        "mixed40": [r for r in rows if 0 < r["success_count"] < 32],
        "balanced16": [r for r in rows if 4 <= r["success_count"] <= 28],
        "non_long64": [r for r in rows if r["suite"] != "libero_long"],
    }
    expected = {"all80": 80, "mixed40": 40, "balanced16": 16, "non_long64": 64}
    actual = {name: len(value) for name, value in subsets.items()}
    if actual != expected:
        raise RuntimeError(f"unexpected subset counts: {actual}")
    return {
        "schema": "himoe-early-action-head-exploratory-v1",
        "protocol": {
            "source": "VLA_MUI_HUB right-16x32; first policy inference per episode",
            "tasks": 5,
            "states_per_task": 16,
            "candidates_per_state": K_CANDIDATES,
            "common_flow_noise_seeds": list(canonical_seeds or ()),
            "target": "client actions[0], complete 10x7 chunk, official checkpoint-std normalized",
            "feature_site": "HB-MLP/gate inputs after layer attention at layers 2-5, d0, action tokens 1-10",
            "feature_pooling": {
                "hidden": "mean over HB2-5 and ten action tokens",
                "router_probs_pooled32": "mean over HB2-5 and ten action tokens",
                "router_probs_layer_resolved128": "mean over ten action tokens only; concatenate HB2-5",
            },
            "feature_dimensions": {
                "initial_flow_noise": 240,
                "initial_flow_noise_live7": 70,
                "hidden": 1024,
                "router_probs": 32,
                "router_probs_layer_resolved": 128,
            },
            "flow_noise_reconstruction": "np.random.default_rng(flow_noise_seed).standard_normal((10,24)).astype(float32); identity was enforced by the capture server",
            "router_probability_preprocessing": "clamp decoded fp16 full probabilities to nonnegative and L1-normalize every layer/denoise/token site before pooling",
            "feature_centering": "candidate-center and scalar-RMS scale separately within each K32 pool; feature-only transductive operation",
            "main_cv": "within-task leave-one-init-state-pool-out; inner state-group CV selects ridge alpha",
            "seed_disjoint_cv": "sorted seeds modulo 4; each held state x held 8-seed block predicted from other 15 states x other 24 seeds; target means use allowed 24 only; feature centering remains transductive over K32",
            "seed_disjoint_evaluation": "pair distances only within held8; PAM K2 independently per held8 then union K8; exact random baseline independently draws two per fixed fold",
            "ridge_alphas": list(ALPHAS),
            "inner_folds": args.inner_folds,
            "pam_keep": K_KEEP,
            "action_vote": "ordinary unweighted true-action medoid within selected K8; no cluster weights",
            "vote_regret": "(K32 global mean distance of selected-K8 vote - K32 global-medoid mean distance) / median K32 pair distance",
            "vote_random_monte_carlo": {
                "draws": MONTE_CARLO_DRAWS,
                "uniform_seed": MONTE_CARLO_SEED,
                "stratified_seed": MONTE_CARLO_SEED + 1,
                "common_subsets_across_pools": True,
            },
            "random_coverage": "exact combinatorial expectation over all uniform K8 subsets",
            "stratified_random_coverage": "exact survival-function expectation for two uniform candidates from each fixed held8 fold",
            "subset_definitions": {
                "mixed40": "0 < eventual episode successes < 32 within a state pool",
                "balanced16": "4 <= eventual episode successes <= 28; balanced-min4, not a 16/16 split",
            },
        },
        "summary": {name: aggregate(value) for name, value in subsets.items()},
        "task_diagnostics": diagnostics,
        "pools": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed-folds", type=int, default=4)
    args = parser.parse_args()
    if args.seed_folds != 4:
        raise ValueError("the audited sensitivity protocol is fixed at four seed folds")

    # These kernels are only 240-480 rows; large BLAS thread teams cost much
    # more in launch/synchronization than they save in arithmetic.
    with threadpool_limits(limits=1):
        payload = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "REPORT.md"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    report_path.write_text(report_markdown(payload))
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
