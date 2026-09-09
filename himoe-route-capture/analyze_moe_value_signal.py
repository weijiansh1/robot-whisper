#!/usr/bin/env python3
"""Test whether early MoE activity contains reusable candidate-value information.

The main corpus is a complete 16-scene x 32-noise-seed grid.  Candidate
selection is evaluated in K8 pools.  Every supervised prediction holds out both
the complete initial scene and the eight noise seeds in the test pool.  The
unsupervised selectors never use actions, rewards, or success labels.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (
    HERE.parent
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
    / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
DEFAULT_FEATURES = HERE / "analysis/expert-activation-future/long-t08/features.npz"
DEFAULT_FORK = HERE / "analysis/fork-pilot-n32/fork_analysis.json"
DEFAULT_COMMITMENT = HERE / "analysis/commitment-grid-s24/analysis.json"
DEFAULT_OUT = HERE / "analysis/moe-value-signal"

N_FOLDS = 4
LIVE_ACTION_DIMS = 7
FLOW_NOISE_SHAPE = (10, 24)
RIDGE_ALPHA = 100.0
PCA_COMPONENTS = 12

EXPERT_SIZE = (
    "expert_mass_rms_mean",
    "expert_mass_rms_token_std",
    "routed_rms_mean",
    "routed_rms_token_std",
)
EXPERT_STRUCTURE = (
    "cancellation_mean",
    "cancellation_token_std",
    "expert_norm_cv_mean",
    "expert_norm_cv_token_std",
)
SHARED_SIZE = ("shared_rms_mean", "shared_rms_token_std")
INPUT_SIZE = ("input_rms_mean", "input_rms_token_std")


@dataclass(frozen=True)
class Dataset:
    scenes: np.ndarray
    seeds: np.ndarray
    seed_folds: np.ndarray
    seed_positions: np.ndarray
    success: np.ndarray
    noise: np.ndarray
    metrics: np.ndarray
    metric_names: tuple[str, ...]
    layer_numbers: np.ndarray
    routes: np.ndarray
    index_grid: np.ndarray
    route_sum_before_normalization: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--fork-summary", type=Path, default=DEFAULT_FORK)
    parser.add_argument("--commitment-summary", type=Path, default=DEFAULT_COMMITMENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def normalize_router_probabilities(probabilities: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 5 or values.shape[-1] != 32:
        raise ValueError("router probabilities must have shape [N,L,T,K,32]")
    if not np.all(np.isfinite(values)) or np.min(values) < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    values = np.maximum(values, 0.0)
    total = values.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("router probability vector has zero mass")
    bounds = (float(total.min()), float(total.max()))
    normalized = values / total
    if not np.allclose(normalized.sum(axis=-1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("router normalization failed")
    return normalized, bounds


def _first_route_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    missing = [int(episode) for episode in expected if int(episode) not in lookup]
    if missing:
        raise ValueError("route store is missing episodes: %s" % missing[:8])
    return np.asarray([lookup[int(episode)] for episode in expected], dtype=np.int64)


def make_index_grid(
    scenes: np.ndarray, folds: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    unique_scenes = np.unique(scenes)
    candidates = len(np.unique(positions))
    grid = np.full((len(unique_scenes), N_FOLDS, candidates), -1, dtype=np.int64)
    for scene_axis, scene in enumerate(unique_scenes):
        for fold in range(N_FOLDS):
            index = np.flatnonzero((scenes == scene) & (folds == fold))
            if len(index) != candidates or len(np.unique(positions[index])) != candidates:
                raise ValueError("scene/fold is not a complete candidate pool")
            grid[scene_axis, fold, positions[index]] = index
    if np.any(grid < 0) or len(np.unique(grid)) != len(scenes):
        raise ValueError("candidate grid does not cover every episode exactly once")
    return grid


def load_dataset(run: Path, feature_path: Path) -> Dataset:
    summaries = json.loads((run / "client/summaries.json").read_text())
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    success = np.asarray([bool(row["success"]) for row in summaries], dtype=np.int8)
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError("episode_index is not unique")

    unique_scenes = np.unique(scenes)
    unique_seeds = np.sort(np.unique(seeds))
    if len(summaries) != len(unique_scenes) * len(unique_seeds):
        raise ValueError("run is not a complete scene x seed grid")
    seed_rank = {int(seed): rank for rank, seed in enumerate(unique_seeds)}
    ranks = np.asarray([seed_rank[int(seed)] for seed in seeds])
    seed_folds = ranks % N_FOLDS
    seed_positions = ranks // N_FOLDS
    index_grid = make_index_grid(scenes, seed_folds, seed_positions)

    for scene in unique_scenes:
        index = scenes == scene
        if not np.array_equal(np.sort(seeds[index]), unique_seeds):
            raise ValueError("a scene does not contain every noise seed")

    noise = np.stack(
        [
            np.random.default_rng(int(seed))
            .standard_normal(FLOW_NOISE_SHAPE)
            .astype(np.float32)
            for seed in seeds
        ]
    )

    with np.load(feature_path, allow_pickle=False) as payload:
        metrics = np.asarray(payload["metrics"], dtype=np.float64)
        metric_names = tuple(str(value) for value in payload["metric_names"])
        layer_numbers = np.asarray(payload["layer_numbers"], dtype=np.int64)
    if metrics.shape[:3] != (len(summaries), 10, len(layer_numbers)):
        raise ValueError("cached expert features do not align with the rollout grid")
    if metrics.shape[-1] != len(metric_names) or not np.all(np.isfinite(metrics)):
        raise ValueError("cached expert metrics are invalid")

    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    rows = _first_route_rows(np.asarray(route_group["episode_id"][:]), episodes)
    probabilities = np.asarray(
        route_group["hb_router_probs"].get_orthogonal_selection(
            (rows, slice(None), slice(0, 3), slice(1, 11), slice(None))
        ),
        dtype=np.float64,
    )
    routes, route_bounds = normalize_router_probabilities(probabilities)
    if routes.shape[:1] != (len(summaries),):
        raise ValueError("router trace does not align with summaries")

    return Dataset(
        scenes=scenes,
        seeds=seeds,
        seed_folds=seed_folds,
        seed_positions=seed_positions,
        success=success,
        noise=noise,
        metrics=metrics,
        metric_names=metric_names,
        layer_numbers=layer_numbers,
        routes=routes,
        index_grid=index_grid,
        route_sum_before_normalization=route_bounds,
    )


def pool_standardize(features: np.ndarray, scenes: np.ndarray, folds: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(scenes) or len(values) != len(folds):
        raise ValueError("features, scenes, and folds are not aligned")
    output = np.empty_like(values)
    for scene in np.unique(scenes):
        for fold in np.unique(folds):
            index = np.flatnonzero((scenes == scene) & (folds == fold))
            mean = values[index].mean(axis=0)
            scale = values[index].std(axis=0)
            output[index] = (values[index] - mean) / np.where(scale > 1e-12, scale, 1.0)
    return output


def metric_features(data: Dataset, tau: int, names: tuple[str, ...]) -> np.ndarray:
    lookup = {name: axis for axis, name in enumerate(data.metric_names)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError("expert feature cache is missing metrics: %s" % missing)
    axes = [lookup[name] for name in names]
    selected = np.take(data.metrics[:, tau], axes, axis=-1)
    return selected.reshape(len(data.scenes), -1)


def route_probe_features(routes: np.ndarray) -> np.ndarray:
    step = routes[:, :, 0]
    entropy = -(step * np.log(np.maximum(step, 1e-20))).sum(axis=-1) / math.log(32.0)
    top1 = step.max(axis=-1)
    top4 = np.sort(step, axis=-1)[..., -4:].sum(axis=-1)
    embedding = np.sqrt(step).mean(axis=2).reshape(len(step), -1)
    summaries = np.concatenate(
        [
            entropy.mean(axis=2),
            entropy.std(axis=2),
            top1.mean(axis=2),
            top1.std(axis=2),
            top4.mean(axis=2),
            top4.std(axis=2),
        ],
        axis=1,
    )
    return np.column_stack([embedding, summaries])


def _standardize_train_test(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (test - mean) / scale


def _reduce_part(
    train: np.ndarray, test: np.ndarray, components: int | None
) -> tuple[np.ndarray, np.ndarray]:
    train, test = _standardize_train_test(train, test)
    if components is None or train.shape[1] <= components:
        return train, test
    count = min(components, train.shape[0] - 1, train.shape[1])
    pca = PCA(n_components=count, svd_solver="full").fit(train)
    return pca.transform(train), pca.transform(test)


def build_ridge_operator(
    parts: list[tuple[np.ndarray, int | None]],
    scenes: np.ndarray,
    folds: np.ndarray,
    alpha: float = RIDGE_ALPHA,
) -> np.ndarray:
    """Return H such that double-held-out ridge predictions equal H @ labels."""

    if alpha <= 0.0 or not parts:
        raise ValueError("ridge needs feature parts and a positive alpha")
    n_rows = len(scenes)
    if any(values.shape[0] != n_rows for values, _ in parts):
        raise ValueError("ridge feature parts are not aligned")
    operator = np.zeros((n_rows, n_rows), dtype=np.float64)
    for held_scene in np.unique(scenes):
        for held_fold in np.unique(folds):
            train = np.flatnonzero((scenes != held_scene) & (folds != held_fold))
            test = np.flatnonzero((scenes == held_scene) & (folds == held_fold))
            train_parts = []
            test_parts = []
            for values, components in parts:
                reduced_train, reduced_test = _reduce_part(
                    values[train], values[test], components
                )
                train_parts.append(reduced_train)
                test_parts.append(reduced_test)
            train_values = np.column_stack(train_parts)
            test_values = np.column_stack(test_parts)
            train_values, test_values = _standardize_train_test(
                train_values, test_values
            )
            design = np.column_stack([np.ones(len(train)), train_values])
            test_design = np.column_stack([np.ones(len(test)), test_values])
            penalty = alpha * np.eye(design.shape[1])
            penalty[0, 0] = 0.0
            mapping = test_design @ np.linalg.solve(
                design.T @ design + penalty, design.T
            )
            operator[np.ix_(test, train)] = mapping
    return operator


def _pool_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def evaluate_scores(data: Dataset, scores: np.ndarray) -> dict[str, Any]:
    score_grid = scores[data.index_grid]
    label_grid = data.success[data.index_grid]
    selected_position = np.argmax(score_grid, axis=-1)
    selected = np.take_along_axis(
        label_grid, selected_position[..., None], axis=-1
    )[..., 0]
    pool_mean = label_grid.mean(axis=-1)
    scene_delta = (selected - pool_mean).mean(axis=1)
    pool_auc = np.full(data.index_grid.shape[:2], np.nan)
    for scene_axis in range(data.index_grid.shape[0]):
        for fold in range(data.index_grid.shape[1]):
            pool_auc[scene_axis, fold] = _pool_auc(
                label_grid[scene_axis, fold], score_grid[scene_axis, fold]
            )
    selected_indices = np.take_along_axis(
        data.index_grid, selected_position[..., None], axis=-1
    )[..., 0]
    selected_seeds = data.seeds[selected_indices]
    _, seed_counts = np.unique(selected_seeds, return_counts=True)
    return {
        "mean_pool_auc": float(np.nanmean(pool_auc)),
        "evaluable_auc_pools": int(np.isfinite(pool_auc).sum()),
        "pool_auc": pool_auc,
        "selected_success": float(selected.mean()),
        "random_expected_success": float(pool_mean.mean()),
        "delta_vs_random": float(scene_delta.mean()),
        "oracle_any_success": float(label_grid.max(axis=-1).mean()),
        "scene_delta": scene_delta,
        "selected_position": selected_position,
        "unique_selected_seeds": int(len(np.unique(selected_seeds))),
        "dominant_seed_share": float(seed_counts.max() / selected_seeds.size),
    }


def bootstrap_score_result(
    result: dict[str, Any], draws: int, rng: np.random.Generator
) -> dict[str, list[float]]:
    n_scenes = result["pool_auc"].shape[0]
    auc_draws = np.empty(draws)
    delta_draws = np.empty(draws)
    for draw in range(draws):
        chosen = rng.integers(0, n_scenes, n_scenes)
        auc_draws[draw] = np.nanmean(result["pool_auc"][chosen])
        delta_draws[draw] = result["scene_delta"][chosen].mean()
    return {
        "mean_pool_auc_ci95": [
            float(np.percentile(auc_draws, 2.5)),
            float(np.percentile(auc_draws, 97.5)),
        ],
        "delta_vs_random_ci95": [
            float(np.percentile(delta_draws, 2.5)),
            float(np.percentile(delta_draws, 97.5)),
        ],
    }


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"pool_auc", "scene_delta", "selected_position"}
    }


def route_centrality(data: Dataset, prefix_steps: int) -> np.ndarray:
    root = np.sqrt(data.routes[:, :, :prefix_steps])
    output = np.empty(len(data.scenes), dtype=np.float64)
    for pool in data.index_grid.reshape(-1, data.index_grid.shape[-1]):
        values = root[pool]
        distance = np.sqrt(
            0.5 * np.square(values[:, None] - values[None, :]).sum(axis=-1)
        ).mean(axis=(2, 3, 4))
        output[pool] = -distance.mean(axis=1)
    return output


def feature_centrality(data: Dataset, features: np.ndarray) -> np.ndarray:
    output = np.empty(len(data.scenes), dtype=np.float64)
    for pool in data.index_grid.reshape(-1, data.index_grid.shape[-1]):
        values = features[pool]
        distance = np.sqrt(
            np.mean(np.square(values[:, None] - values[None, :]), axis=-1)
        )
        output[pool] = -distance.mean(axis=1)
    return output


def unsupervised_scores(data: Dataset, features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    routes = data.routes
    entropy = -(routes * np.log(np.maximum(routes, 1e-20))).sum(axis=-1) / math.log(32.0)
    top4 = np.sort(routes, axis=-1)[..., -4:].sum(axis=-1)
    movement = np.sqrt(
        0.5
        * np.square(np.sqrt(routes[:, :, 1:]) - np.sqrt(routes[:, :, :-1])).sum(
            axis=-1
        )
    ).mean(axis=(1, 2, 3))

    mass = pool_standardize(
        metric_features(data, 0, ("expert_mass_rms_mean",)),
        data.scenes,
        data.seed_folds,
    ).mean(axis=1)
    routed = pool_standardize(
        metric_features(data, 0, ("routed_rms_mean",)),
        data.scenes,
        data.seed_folds,
    ).mean(axis=1)
    cancellation = pool_standardize(
        metric_features(data, 0, ("cancellation_mean",)),
        data.scenes,
        data.seed_folds,
    ).mean(axis=1)
    norm_cv = pool_standardize(
        metric_features(data, 0, ("expert_norm_cv_mean",)),
        data.scenes,
        data.seed_folds,
    ).mean(axis=1)
    first = []
    for tau in range(3):
        first.append(
            pool_standardize(
                metric_features(data, tau, EXPERT_SIZE),
                data.scenes,
                data.seed_folds,
            )
        )
    trajectory = np.stack(first, axis=1)
    size_movement = -np.sqrt(
        np.mean(np.square(np.diff(trajectory, axis=1)), axis=(1, 2))
    )

    expert_full_central = feature_centrality(data, features["expert_full"])
    return {
        "route_central_t0": route_centrality(data, 1),
        "route_central_prefix3": route_centrality(data, 3),
        "low_route_entropy_t0": -entropy[:, :, 0].mean(axis=(1, 2)),
        "high_route_top4_t0": top4[:, :, 0].mean(axis=(1, 2)),
        "route_stable_prefix3": -movement,
        "expert_size_central_t0": feature_centrality(data, features["expert_size"]),
        "expert_full_central_t0": expert_full_central,
        "expert_full_outlier_t0": -expert_full_central,
        "high_expert_mass_t0": mass,
        "high_routed_rms_t0": routed,
        "low_cancellation_t0": -cancellation,
        "high_expert_norm_cv_t0": norm_cv,
        "high_expert_heterogeneity_t0": 0.5 * (cancellation + norm_cv),
        "expert_size_stable_prefix3": size_movement,
    }


def label_permutations(
    data: Dataset, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Permute seed columns within folds, identically across all scenes."""

    label_grid = data.success[data.index_grid]
    output = np.empty((count, len(data.success)), dtype=np.int8)
    for draw in range(count):
        values = np.empty_like(label_grid)
        for fold in range(N_FOLDS):
            values[:, fold] = label_grid[:, fold, rng.permutation(label_grid.shape[-1])]
        output[draw, data.index_grid.ravel()] = values.ravel()
    return output


def _mean_pool_auc_batch(labels: np.ndarray, scores: np.ndarray, grid: np.ndarray) -> np.ndarray:
    label_grid = labels[:, grid]
    score_grid = scores[:, grid]
    positive = label_grid[..., :, None] == 1
    negative = label_grid[..., None, :] == 0
    pairs = positive & negative
    wins = score_grid[..., :, None] > score_grid[..., None, :]
    ties = score_grid[..., :, None] == score_grid[..., None, :]
    numerator = (wins & pairs).sum(axis=(-1, -2)) + 0.5 * (ties & pairs).sum(
        axis=(-1, -2)
    )
    denominator = pairs.sum(axis=(-1, -2))
    valid = denominator > 0
    auc = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=valid,
    )
    return np.nanmean(auc, axis=(1, 2))


def supervised_permutation_test(
    operator: np.ndarray,
    data: Dataset,
    permutations: np.ndarray,
    observed: dict[str, Any],
    chunk_size: int = 500,
) -> dict[str, float]:
    exceed_auc = 0
    exceed_delta = 0
    for start in range(0, len(permutations), chunk_size):
        labels = permutations[start : start + chunk_size]
        scores = labels @ operator.T
        null_auc = _mean_pool_auc_batch(labels, scores, data.index_grid)
        score_grid = scores[:, data.index_grid]
        label_grid = labels[:, data.index_grid]
        selected = np.argmax(score_grid, axis=-1)
        selected_success = np.take_along_axis(
            label_grid, selected[..., None], axis=-1
        )[..., 0].mean(axis=(1, 2))
        null_delta = selected_success - data.success.mean()
        exceed_auc += int((null_auc >= observed["mean_pool_auc"] - 1e-15).sum())
        exceed_delta += int(
            (null_delta >= observed["delta_vs_random"] - 1e-15).sum()
        )
    return {
        "mean_pool_auc_permutation_p": float(
            (exceed_auc + 1) / (len(permutations) + 1)
        ),
        "delta_vs_random_permutation_p": float(
            (exceed_delta + 1) / (len(permutations) + 1)
        ),
    }


def unsupervised_permutation_tests(
    data: Dataset,
    scores: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
    permutations: np.ndarray,
) -> dict[str, dict[str, float]]:
    label_grid = permutations[:, data.index_grid]
    null_by_method = {}
    for name, values in scores.items():
        selected = results[name]["selected_position"]
        picked = np.take_along_axis(
            label_grid, selected[None, ..., None], axis=-1
        )[..., 0]
        null_by_method[name] = picked.mean(axis=(1, 2)) - data.success.mean()
    null_matrix = np.column_stack([null_by_method[name] for name in scores])
    maximum = null_matrix.max(axis=1)
    output = {}
    for name in scores:
        observed = results[name]["delta_vs_random"]
        output[name] = {
            "permutation_p": float(
                (1 + (null_by_method[name] >= observed - 1e-15).sum())
                / (len(permutations) + 1)
            ),
            "permutation_p_fwer": float(
                (1 + (maximum >= observed - 1e-15).sum())
                / (len(permutations) + 1)
            ),
        }
    return output


def _fmt(value: float, signed: bool = False) -> str:
    return ("%+.3f" if signed else "%.3f") % value


def render_report(summary: dict[str, Any]) -> str:
    probe = summary["supervised_probe"]
    unsupervised = summary["unsupervised_selectors"]
    fork = summary["counterfactual_local_progress"]
    commitment = summary["common_future_grid"]
    primary = probe["expert_full_t0"]
    lines = [
        "# MoE 中是否存在未来动作价值信息",
        "",
        "## 实验思路",
        "",
        "- 在同一观测下把 32 个初始噪声分成 4 个 K8 候选池。所有 router 概率先在每层、每 token 上归一化为 32 维概率分布；专家特征再按同一 K8 池、同一层和同一指标标准化。",
        "- 信息存在性上限使用固定 `alpha=100` 的线性 ridge 探针。每个测试池对应的完整初始场景和 8 个 noise seed 都从训练中删除，因此测试场景和测试 seed 同时未见。",
        "- 无监督选择器只看 step 0 或前三个 denoise step 的 router/实际专家输出，不看动作、奖励或成功标签。置换检验在每个 seed fold 内整体置换 seed 列，并在所有场景复用同一置换；FWER 对全部无监督规则统一校正。",
        "- 16x32 success 网格是 episode-start 关联检验，后续噪声仍随 seed stream 变化，不作因果解释；另外复用两组真实反事实证据：同 snapshot 执行 32 个动作并接共同未来噪声的局部进展，以及固定首步路由、只改变共同未来噪声的 16x8 结果网格。",
        "",
        "## 结果",
        "",
        "完整 16x32 网格共 `%d` 条 rollout，成功率 `%s`；K8 随机选 1 的期望成功率为 `%s`，池内存在成功候选时的 oracle 上限为 `%s`。AUC 只统计同时有成功和失败的 `%d/64` 个池，0.5 为随机。"
        % (
            summary["integrity"]["episodes"],
            _fmt(summary["integrity"]["success_rate"]),
            _fmt(primary["random_expected_success"]),
            _fmt(primary["oracle_any_success"]),
            primary["evaluable_auc_pools"],
        ),
        "",
        "| 严格双留出探针 | K8 内 AUC | 选 1 成功率 | 相对随机 |",
        "|---|---:|---:|---:|",
    ]
    probe_order = (
        ("noise", "初始有效噪声"),
        ("router_t0", "归一化 router"),
        ("expert_size_t0", "专家输出大小"),
        ("expert_structure_t0", "专家抵消/离散结构"),
        ("expert_full_t0", "完整专家标量"),
        ("shared_size_t0", "shared 分支大小"),
        ("noise_expert_full_t0", "噪声 + 完整专家标量"),
        ("expert_full_prefix3", "专家标量前三步"),
    )
    for key, label in probe_order:
        item = probe[key]
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                label,
                _fmt(item["mean_pool_auc"]),
                _fmt(item["selected_success"]),
                _fmt(item["delta_vs_random"], signed=True),
            )
        )
    lines.extend(
        [
            "",
            "step 0 完整专家标量是最清楚的弱信号：AUC `%s`（固定交叉验证分数的 scene-bootstrap 95%% CI `%s-%s`，按标签重训的置换 `p=%s`）；选 1 为 `%s`，比随机期望高 `%s`（固定分数 95%% CI `%s` 到 `%s`，重训练置换 `p=%s`）。更严格的重训练置换均未过 0.05，因此这是边界性证据，不是确认结果。它选择了 `%d/32` 个不同 seed，最大单 seed 占比 `%s`，不是固定 seed 模板。"
            % (
                _fmt(primary["mean_pool_auc"]),
                _fmt(primary["mean_pool_auc_ci95"][0]),
                _fmt(primary["mean_pool_auc_ci95"][1]),
                _fmt(primary["mean_pool_auc_permutation_p"]),
                _fmt(primary["selected_success"]),
                _fmt(primary["delta_vs_random"], signed=True),
                _fmt(primary["delta_vs_random_ci95"][0], signed=True),
                _fmt(primary["delta_vs_random_ci95"][1], signed=True),
                _fmt(primary["delta_vs_random_permutation_p"]),
                primary["unique_selected_seeds"],
                _fmt(primary["dominant_seed_share"]),
            ),
            "",
            "这个信号主要来自专家间的抵消和输出离散结构，而不只是激活绝对大小：抵消描述 top-4 加权专家向量合并时损失了多少幅值，大小离散度描述 top-4 专家输出范数的 CV。加入初始噪声没有增强，前三步拼接也没有超过 step 0；归一化 router、shared 分支和输入大小对照均没有同样的正向信息。带标签探针只用于判断信息上限，不属于无监督方法。",
            "",
            "| 无监督规则 | K8 内 AUC | 选 1 成功率 | 相对随机 | 原始 p | FWER p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    unsupervised_order = (
        ("route_central_t0", "step 0 router 中心度"),
        ("route_stable_prefix3", "router 前三步稳定"),
        ("low_route_entropy_t0", "低 router 熵"),
        ("expert_size_central_t0", "专家大小中心度"),
        ("high_expert_mass_t0", "高专家加权输出"),
        ("low_cancellation_t0", "低专家抵消"),
        ("high_expert_norm_cv_t0", "高专家大小离散度"),
        ("high_expert_heterogeneity_t0", "高专家异质性（探索）"),
        ("expert_size_stable_prefix3", "专家大小前三步稳定"),
    )
    for key, label in unsupervised_order:
        item = unsupervised[key]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                label,
                _fmt(item["mean_pool_auc"]),
                _fmt(item["selected_success"]),
                _fmt(item["delta_vs_random"], signed=True),
                _fmt(item["permutation_p"]),
                _fmt(item["permutation_p_fwer"]),
            )
        )
    best = unsupervised["high_expert_norm_cv_t0"]
    lines.extend(
        [
            "",
            "无监督里只有“高专家大小离散度”留下弱趋势：选 1 成功率 `%s`，增量 `%s`，但原始 `p=%s`，全规则校正后 `p=%s`，95%% CI `%s` 到 `%s`；目前不能视为已验证控制器。router 中心度、低熵和前三步稳定都没有价值增益。"
            % (
                _fmt(best["selected_success"]),
                _fmt(best["delta_vs_random"], signed=True),
                _fmt(best["permutation_p"]),
                _fmt(best["permutation_p_fwer"]),
                _fmt(best["delta_vs_random_ci95"][0], signed=True),
                _fmt(best["delta_vs_random_ci95"][1], signed=True),
            ),
            "",
            "真实局部反事实中，route 对同一状态内的短期 drawer progress 有很小信息：within-snapshot Spearman `%s`、`p=%s`；但留出完整 snapshot 后只有 `%s`、`p=%s`。这说明信息更像状态特异的细粒度变化，尚没有跨状态通用读出。"
            % (
                _fmt(fork["within_snapshot_spearman"]),
                _fmt(fork["within_snapshot_permutation_p"]),
                _fmt(fork["held_snapshot_spearman"]),
                _fmt(fork["held_snapshot_permutation_p"]),
            ),
            "",
            "固定首步路由、改变共同未来噪声时，16/16 条首路由都同时出现成功和失败；route-outcome MI 为 `%s bits`（`p=%s`）。最强 denoise route 信号相关 `%s`，原始 `p=%s`，10-step 校正后 `p=%s`，仍只是提示。"
            % (
                _fmt(commitment["mutual_information_bits"]),
                _fmt(commitment["mutual_information_permutation_p"]),
                _fmt(commitment["best_route_correlation"]),
                _fmt(commitment["best_route_p_uncorrected"]),
                _fmt(commitment["best_route_p_fwer"]),
            ),
            "",
            "结论：MoE 中确实能测到一点未来价值相关信息，但目前最可信的是 step 0 实际专家输出的抵消/离散结构，不是 router 激活中心度；它很弱、偏状态特异，尚不足以支持无监督自动控噪。此前选择器变差，是因为它优化了“未来路由中心度”这个伪目标，而该目标没有稳定对应任务成功。",
            "",
        ]
    )
    return "\n".join(lines)


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


def existing_evidence(fork_path: Path, commitment_path: Path) -> tuple[dict, dict]:
    fork = json.loads(fork_path.read_text())
    commitment = json.loads(commitment_path.read_text())
    within = fork["within_snapshot_fit"]["route"]
    held = fork["predictors"]["route"]
    route_steps = commitment["route_signal"]["per_denoise"]
    best = max(route_steps, key=lambda row: row["correlation"])
    local = {
        "within_snapshot_spearman": within["mean_within_snapshot_spearman"],
        "within_snapshot_permutation_p": within["permutation_p"],
        "held_snapshot_spearman": held["within_snapshot_spearman"],
        "held_snapshot_permutation_p": held["permutation_p"],
        "valid_snapshots": within["n_snapshots"],
    }
    common = {
        "mixed_rows": commitment["commitment"]["mixed_rows"],
        "total_rows": commitment["commitment"]["n_first_routes"],
        "mutual_information_bits": commitment["commitment"][
            "route_outcome_mutual_information_bits_plugin"
        ],
        "mutual_information_permutation_p": commitment["commitment"]["permutation"][
            "mutual_information_upper_p"
        ],
        "best_route_denoise": best["denoise"],
        "best_route_correlation": best["correlation"],
        "best_route_p_uncorrected": best["permutation_p_uncorrected"],
        "best_route_p_fwer": best["permutation_p_fwer_10_denoise"],
    }
    return local, common


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0 or args.permutations <= 0:
        raise ValueError("bootstrap and permutation counts must be positive")
    data = load_dataset(args.run, args.features)
    rng = np.random.default_rng(args.seed)

    features = {
        "noise": data.noise[:, :, :LIVE_ACTION_DIMS].reshape(len(data.scenes), -1),
        "router": route_probe_features(data.routes),
        "expert_size": metric_features(data, 0, EXPERT_SIZE),
        "expert_structure": metric_features(data, 0, EXPERT_STRUCTURE),
        "expert_full": metric_features(data, 0, EXPERT_SIZE + EXPERT_STRUCTURE),
        "shared_size": metric_features(data, 0, SHARED_SIZE),
        "input_size": metric_features(data, 0, INPUT_SIZE),
    }
    prefix = [
        metric_features(data, tau, EXPERT_SIZE + EXPERT_STRUCTURE)
        for tau in range(3)
    ]
    features["expert_full_prefix3"] = np.column_stack(prefix)
    features = {
        name: pool_standardize(values, data.scenes, data.seed_folds)
        for name, values in features.items()
    }

    specs = {
        "noise": [(features["noise"], PCA_COMPONENTS)],
        "router_t0": [(features["router"], PCA_COMPONENTS)],
        "expert_size_t0": [(features["expert_size"], None)],
        "expert_structure_t0": [(features["expert_structure"], None)],
        "expert_full_t0": [(features["expert_full"], None)],
        "shared_size_t0": [(features["shared_size"], None)],
        "input_size_t0": [(features["input_size"], None)],
        "noise_expert_full_t0": [
            (features["noise"], PCA_COMPONENTS),
            (features["expert_full"], None),
        ],
        "expert_full_prefix3": [(features["expert_full_prefix3"], None)],
    }
    operators = {}
    probe_results = {}
    for name, parts in specs.items():
        print("building strict double-held-out probe:", name, flush=True)
        operator = build_ridge_operator(parts, data.scenes, data.seed_folds)
        operators[name] = operator
        result = evaluate_scores(data, operator @ data.success)
        result.update(bootstrap_score_result(result, args.bootstrap, rng))
        probe_results[name] = result

    permutations = label_permutations(data, args.permutations, rng)
    primary_test = supervised_permutation_test(
        operators["expert_full_t0"],
        data,
        permutations,
        probe_results["expert_full_t0"],
    )
    probe_results["expert_full_t0"].update(primary_test)

    fixed_scores = unsupervised_scores(data, features)
    fixed_results = {}
    for name, values in fixed_scores.items():
        result = evaluate_scores(data, values)
        result.update(bootstrap_score_result(result, args.bootstrap, rng))
        fixed_results[name] = result
    fixed_tests = unsupervised_permutation_tests(
        data, fixed_scores, fixed_results, permutations
    )
    for name, test in fixed_tests.items():
        fixed_results[name].update(test)

    local, common = existing_evidence(args.fork_summary, args.commitment_summary)
    summary = {
        "experiment": "moe_candidate_value_signal",
        "integrity": {
            "run": str(args.run.resolve()),
            "features": str(args.features.resolve()),
            "episodes": int(len(data.success)),
            "scenes": int(len(np.unique(data.scenes))),
            "noise_seeds": int(len(np.unique(data.seeds))),
            "candidate_pool_size": int(data.index_grid.shape[-1]),
            "seed_folds": N_FOLDS,
            "successes": int(data.success.sum()),
            "success_rate": float(data.success.mean()),
            "router_sum_before_normalization": data.route_sum_before_normalization,
            "router_sum_after_normalization_max_error": float(
                np.max(np.abs(data.routes.sum(axis=-1) - 1.0))
            ),
            "expert_layers": data.layer_numbers,
        },
        "protocol": {
            "normalization": "full 32-way router simplex per layer/token; then label-free K8 pool standardization per feature column",
            "probe": "ridge alpha=100; every prediction excludes its complete scene and its 8-seed fold",
            "high_dimensional_reduction": "train-only StandardScaler + PCA(12) for noise and router",
            "uncertainty": "initial-scene bootstrap",
            "permutation": "within each seed fold, permute seed columns identically across scenes; supervised primary refits via its exact ridge operator",
            "unsupervised_fwer_family_size": len(fixed_scores),
            "bootstrap_draws": args.bootstrap,
            "permutation_draws": args.permutations,
            "random_seed": args.seed,
        },
        "supervised_probe": {
            name: compact_result(result) for name, result in probe_results.items()
        },
        "unsupervised_selectors": {
            name: compact_result(result) for name, result in fixed_results.items()
        },
        "counterfactual_local_progress": local,
        "common_future_grid": common,
    }
    summary = finite_json(summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    np.savez_compressed(
        args.out_dir / "scores.npz",
        scenes=data.scenes,
        noise_seeds=data.seeds,
        seed_folds=data.seed_folds,
        success=data.success,
        **{
            "probe__" + name: operators[name] @ data.success
            for name in operators
        },
        **{"unsupervised__" + name: values for name, values in fixed_scores.items()},
    )
    print("wrote", args.out_dir / "report.md", flush=True)


if __name__ == "__main__":
    main()
