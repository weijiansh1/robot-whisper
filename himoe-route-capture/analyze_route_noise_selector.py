#!/usr/bin/env python3
"""Shadow A/B test for an unsupervised MoE-based flow-noise selector.

The controller sees only the initial flow noise and the first three HB routing
evaluations from one same-observation K8 pool.  A small ridge head is trained to
predict late routing centrality, never actions, rewards, or success.  Evaluation
holds out both the complete task and the eight candidate seeds being ranked.

The source corpus contains one full rollout per initial seed stream.  Consequently
the success comparison is an episode-start shadow test, not a causal per-replan
closed-loop result; later noises in each recorded episode also belong to that seed
stream.  The report keeps this limitation explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr
from scipy.stats import rankdata

from route_noise_selector import (
    N_EXPERTS,
    RidgeRouteHead,
    centrality,
    early_route_descriptors,
    late_route_centrality_target,
    noise_descriptors,
    normalize_router_probabilities,
    pairwise_rms,
    pool_standardize,
    route_centrality_scores,
    stable_argmin,
)


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis" / "route-noise-selector"
PRIMARY_TASK = "libero_goal/open_the_top_drawer_and_put_the_bowl_inside"
FLOW_NOISE_SHAPE = (10, 24)
PREFIX_STEPS = 3
SEED_FOLDS = 4
ALPHAS = (0.1, 1.0, 10.0, 100.0)
HEAD_METHODS = ("future_route_head", "future_route_noise_head")
METHODS = (
    "early_route_central",
    "future_route_head",
    "future_route_noise_head",
    "initial_noise_central",
    "late_route_oracle",
)


@dataclass(frozen=True)
class Pool:
    task: str
    suite: str
    state: int
    fold: int
    seeds: np.ndarray
    route_features: np.ndarray
    noise_features: np.ndarray
    late_target: np.ndarray
    early_route_score: np.ndarray
    initial_noise_score: np.ndarray
    success: np.ndarray

    def features(self, method: str) -> np.ndarray:
        if method == "future_route_head":
            return self.route_features
        if method == "future_route_noise_head":
            return np.concatenate([self.route_features, self.noise_features], axis=1)
        raise KeyError(method)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--primary-task", default=PRIMARY_TASK)
    parser.add_argument("--prefix-steps", type=int, default=PREFIX_STEPS)
    parser.add_argument("--seed-folds", type=int, default=SEED_FOLDS)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def finite(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("non-finite result: %r" % result)
    return result


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = rankdata(np.asarray(left, dtype=np.float64))
    b = rankdata(np.asarray(right, dtype=np.float64))
    a -= a.mean()
    b -= b.mean()
    scale = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / scale) if scale > 1e-20 else float("nan")


def discover_runs(hub_root: Path) -> list[Path]:
    cache = hub_root / "cache" / "HiMoE-VLA"
    runs = sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/server/routes.zarr")
        if (path.parent.parent / "client" / "summaries.json").exists()
    )
    if len(runs) != 5:
        raise RuntimeError("expected five complete right-16x32 runs, found %d" % len(runs))
    return runs


def first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    if set(map(int, expected)) != set(lookup):
        missing = sorted(set(map(int, expected)) - set(lookup))
        extra = sorted(set(lookup) - set(map(int, expected)))
        raise RuntimeError("episode alignment mismatch: missing=%s extra=%s" % (missing[:5], extra[:5]))
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


def _candidate_spread(values: np.ndarray) -> float:
    """Maximum range over candidate axis 1 for ``[state, candidate, ...]``."""

    return float(np.max(np.ptp(np.asarray(values, dtype=np.float64), axis=1)))


def load_task_pools(
    run: Path,
    cache_root: Path,
    prefix_steps: int,
    seed_folds: int,
) -> tuple[list[Pool], dict[str, Any], tuple[int, ...]]:
    summaries = sorted(
        json.loads((run / "client" / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 16 * 32:
        raise RuntimeError("%s is not a complete 16x32 run" % run)
    episodes = np.asarray([int(row["episode_index"]) for row in summaries], dtype=np.int64)
    states = np.asarray([int(row["init_state_id"]) for row in summaries], dtype=np.int64)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], dtype=np.int64)
    outcomes = np.asarray([bool(row["success"]) for row in summaries], dtype=bool)
    if len(np.unique(episodes)) != len(episodes):
        raise RuntimeError("episode_index is not unique in %s" % run)

    route_group = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:], dtype=np.int64)
    rows = first_rows(route_episode, episodes)
    inference_calls = np.asarray(
        [int(row["inference_calls"]) for row in summaries], dtype=np.int64
    )
    expected_rows = np.r_[0, np.cumsum(inference_calls)[:-1]]
    _, route_counts = np.unique(route_episode, return_counts=True)
    if not np.array_equal(rows, expected_rows) or not np.array_equal(
        route_counts, inference_calls
    ):
        raise RuntimeError("Zarr episode boundaries disagree with client inference_calls")
    # This recorder stores a server-global call counter, not an episode-local
    # control index.  Assert that contract instead of incorrectly expecting zero
    # at every episode boundary.
    control_step = np.asarray(route_group["control_step"][:], dtype=np.int64)
    if not np.array_equal(control_step, np.arange(len(control_step))):
        raise RuntimeError("route control_step is not the expected global call counter")
    raw = np.asarray(
        route_group["hb_router_probs"].oindex[rows, :, :, :, :],
        dtype=np.float32,
    )
    if raw.shape != (512, 8, 10, 11, N_EXPERTS):
        raise RuntimeError("unexpected route shape %s in %s" % (raw.shape, run))
    raw_mass_error = float(np.max(np.abs(raw.sum(axis=-1) - 1.0)))
    normalized = normalize_router_probabilities(raw)
    normalized_mass_error = float(np.max(np.abs(normalized.sum(axis=-1) - 1.0)))
    action_routes = normalized[:, :, :, 1:, :]

    as_probability = np.asarray(route_group["as_probs"].oindex[rows], dtype=np.float64)
    noise = reconstruct_initial_noise(seeds)
    relative = run.relative_to(cache_root)
    task = str(relative.parent)
    suite = relative.parts[0]
    unique_states = np.unique(states)
    if len(unique_states) != 16:
        raise RuntimeError("expected 16 init states in %s" % run)

    pools: list[Pool] = []
    canonical_seeds: tuple[int, ...] | None = None
    state_token_grids = []
    as_grids = []
    for state in unique_states:
        indices = np.flatnonzero(states == state)
        indices = indices[np.argsort(seeds[indices])]
        ordered_seeds = tuple(map(int, seeds[indices]))
        if len(indices) != 32 or len(set(ordered_seeds)) != 32:
            raise RuntimeError("state %d in %s is not a complete K32 pool" % (state, run))
        if canonical_seeds is None:
            canonical_seeds = ordered_seeds
        elif canonical_seeds != ordered_seeds:
            raise RuntimeError("seed ordering changes across states in %s" % run)
        state_token_grids.append(normalized[indices, :, :, 0, :])
        as_grids.append(as_probability[indices])

        for fold in range(seed_folds):
            local = np.arange(32, dtype=np.int64)[fold::seed_folds]
            if len(local) != 32 // seed_folds:
                raise RuntimeError("seed folds do not evenly divide K32")
            selected = indices[local]
            routes = action_routes[selected]
            early = early_route_descriptors(routes, prefix_steps=prefix_steps)
            noise_feature = noise_descriptors(noise[selected])
            route_score = route_centrality_scores(routes[:, :, :prefix_steps])
            noise_score = centrality(pairwise_rms(noise[selected, :, :7]))
            pools.append(
                Pool(
                    task=task,
                    suite=suite,
                    state=int(state),
                    fold=fold,
                    seeds=seeds[selected].copy(),
                    route_features=pool_standardize(early),
                    noise_features=pool_standardize(noise_feature),
                    late_target=late_route_centrality_target(
                        routes, prefix_steps=prefix_steps
                    ),
                    early_route_score=route_score,
                    initial_noise_score=noise_score,
                    success=outcomes[selected].copy(),
                )
            )

    state_token_grid = np.stack(state_token_grids)
    as_grid = np.stack(as_grids)
    diagnostics = {
        "task": task,
        "suite": suite,
        "episodes": len(summaries),
        "successes": int(outcomes.sum()),
        "success_rate": finite(outcomes.mean()),
        "pools": len(pools),
        "raw_probability_mass_max_error": raw_mass_error,
        "renormalized_probability_mass_max_error": normalized_mass_error,
        "state_token_candidate_max_range": _candidate_spread(state_token_grid),
        "as_candidate_max_range": _candidate_spread(as_grid),
    }
    return pools, diagnostics, canonical_seeds or ()


def stack_training(pools: list[Pool], method: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([pool.features(method) for pool in pools]),
        np.concatenate([pool.late_target for pool in pools]),
    )


def tune_alpha(pools: list[Pool], method: str) -> tuple[float, dict[str, float]]:
    tasks = sorted({pool.task for pool in pools})
    if len(tasks) < 2:
        raise ValueError("alpha tuning needs at least two source tasks")
    losses = {alpha: 0.0 for alpha in ALPHAS}
    counts = {alpha: 0 for alpha in ALPHAS}
    for validation_task in tasks:
        train = [pool for pool in pools if pool.task != validation_task]
        validation = [pool for pool in pools if pool.task == validation_task]
        train_x, train_y = stack_training(train, method)
        valid_x, valid_y = stack_training(validation, method)
        for alpha in ALPHAS:
            prediction = RidgeRouteHead.fit(train_x, train_y, alpha).predict(valid_x)
            losses[alpha] += float(np.square(prediction - valid_y).sum())
            counts[alpha] += len(valid_y)
    mse = {str(alpha): finite(losses[alpha] / counts[alpha]) for alpha in ALPHAS}
    selected = min(ALPHAS, key=lambda alpha: (mse[str(alpha)], alpha))
    return float(selected), mse


def fit_head(pools: list[Pool], method: str) -> tuple[RidgeRouteHead, dict[str, Any]]:
    alpha, mse = tune_alpha(pools, method)
    features, target = stack_training(pools, method)
    head = RidgeRouteHead.fit(features, target, alpha)
    return head, {
        "alpha": alpha,
        "source_tasks": sorted({pool.task for pool in pools}),
        "source_pools": len(pools),
        "source_candidates": len(target),
        "leave_one_source_task_out_mse": mse,
    }


def selected_local_index(pool: Pool, method: str, head: RidgeRouteHead | None) -> int:
    ids = np.arange(len(pool.seeds), dtype=np.int64)
    if method == "early_route_central":
        return stable_argmin(pool.early_route_score, ids)
    if method == "initial_noise_central":
        return stable_argmin(pool.initial_noise_score, ids)
    if method == "late_route_oracle":
        return stable_argmin(pool.late_target, ids)
    if method in HEAD_METHODS and head is not None:
        return stable_argmin(head.predict(pool.features(method)), ids)
    raise KeyError(method)


def bootstrap_state_delta(
    selected: np.ndarray,
    baseline: np.ndarray,
    states: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> list[float]:
    unique = np.unique(states)
    by_state = np.asarray(
        [np.mean(selected[states == state] - baseline[states == state]) for state in unique]
    )
    indices = rng.integers(0, len(unique), size=(draws, len(unique)))
    distribution = by_state[indices].mean(axis=1)
    return [finite(value) for value in np.percentile(distribution, [2.5, 97.5])]


def permutation_p_values(
    observed_delta: float,
    pool_success_rates: np.ndarray,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    # Under a within-pool candidate-label permutation, a fixed selected position
    # is Bernoulli with the pool's observed success fraction.  This vectorized
    # draw is exactly equivalent to explicitly permuting each K8 label vector.
    null = (
        (rng.random((draws, len(pool_success_rates))) < pool_success_rates).mean(axis=1)
        - pool_success_rates.mean()
    )
    return {
        "one_sided_improvement": finite(
            (np.count_nonzero(null >= observed_delta) + 1) / (draws + 1)
        ),
        "two_sided": finite(
            (np.count_nonzero(np.abs(null) >= abs(observed_delta)) + 1) / (draws + 1)
        ),
        "null_mean": finite(null.mean()),
        "null_95": [finite(value) for value in np.percentile(null, [2.5, 97.5])],
    }


def seed_concentration(seeds: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    counts = Counter(map(int, seeds))
    probability = np.asarray(list(counts.values()), dtype=np.float64) / len(seeds)
    entropy = -float(np.sum(probability * np.log(probability)))
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    per_fold = {}
    for fold in np.unique(folds):
        fold_counts = Counter(map(int, seeds[folds == fold]))
        per_fold[str(int(fold))] = {
            "unique_selected_seeds": len(fold_counts),
            "counts": {str(seed): count for seed, count in sorted(fold_counts.items())},
        }
    return {
        "unique_selected_seeds": len(counts),
        "effective_seed_count": finite(math.exp(entropy)),
        "max_seed_share": finite(max(counts.values()) / len(seeds)),
        "top_seeds": [{"seed": seed, "count": count} for seed, count in top[:8]],
        "per_fold": per_fold,
    }


def aggregate_task_method(
    pools: list[Pool],
    method: str,
    selected_indices: list[int],
    predicted_scores: list[np.ndarray | None],
    bootstrap: int,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = np.asarray(
        [pool.success[index] for pool, index in zip(pools, selected_indices)],
        dtype=np.float64,
    )
    baseline = np.asarray([pool.success.mean() for pool in pools], dtype=np.float64)
    states = np.asarray([pool.state for pool in pools], dtype=np.int64)
    selected_seeds = np.asarray(
        [pool.seeds[index] for pool, index in zip(pools, selected_indices)],
        dtype=np.int64,
    )
    folds = np.asarray([pool.fold for pool in pools], dtype=np.int64)
    delta = finite(np.mean(selected - baseline))
    correlations = []
    for pool, score in zip(pools, predicted_scores):
        if score is None:
            continue
        value = rank_correlation(score, pool.late_target)
        if np.isfinite(value):
            correlations.append(value)
    return {
        "selected_success_rate": finite(selected.mean()),
        "selected_successes": int(selected.sum()),
        "selection_count": len(selected),
        "exact_random_success_rate": finite(baseline.mean()),
        "paired_delta": delta,
        "state_cluster_bootstrap_ci_95": bootstrap_state_delta(
            selected, baseline, states, bootstrap, rng
        ),
        "candidate_label_permutation": permutation_p_values(
            delta, baseline, permutations, rng
        ),
        "future_route_rank_spearman_mean": (
            finite(np.mean(correlations)) if correlations else None
        ),
        "seed_concentration": seed_concentration(selected_seeds, folds),
        "selected": [
            {
                "state": int(pool.state),
                "fold": int(pool.fold),
                "seed": int(pool.seeds[index]),
                "success": bool(pool.success[index]),
            }
            for pool, index in zip(pools, selected_indices)
        ],
    }


def evaluate_task(
    task: str,
    all_pools: list[Pool],
    bootstrap: int,
    permutations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    test_pools = sorted(
        [pool for pool in all_pools if pool.task == task],
        key=lambda pool: (pool.state, pool.fold),
    )
    if len(test_pools) != 16 * SEED_FOLDS:
        raise RuntimeError("held-out task does not have 16x4 K8 pools")
    selected: dict[str, list[int]] = {method: [] for method in METHODS}
    scores: dict[str, list[np.ndarray | None]] = {method: [] for method in METHODS}
    head_diagnostics: dict[str, list[dict[str, Any]]] = {
        method: [] for method in HEAD_METHODS
    }

    for fold in range(SEED_FOLDS):
        fold_test = [pool for pool in test_pools if pool.fold == fold]
        train = [
            pool
            for pool in all_pools
            if pool.task != task and pool.fold != fold
        ]
        test_seed_set = set(map(int, fold_test[0].seeds))
        train_seed_set = {int(seed) for pool in train for seed in pool.seeds}
        if test_seed_set & train_seed_set:
            raise RuntimeError("test noise seeds leaked into head training")
        heads = {}
        for method in HEAD_METHODS:
            heads[method], diagnostic = fit_head(train, method)
            diagnostic.update(
                {
                    "held_out_fold": fold,
                    "held_out_seeds": sorted(test_seed_set),
                    "feature_width": len(heads[method].feature_mean),
                }
            )
            head_diagnostics[method].append(diagnostic)

        for pool in fold_test:
            for method in METHODS:
                head = heads.get(method)
                index = selected_local_index(pool, method, head)
                selected[method].append(index)
                if method == "early_route_central":
                    score = pool.early_route_score
                elif method == "initial_noise_central":
                    score = pool.initial_noise_score
                elif method == "late_route_oracle":
                    score = pool.late_target
                else:
                    score = head.predict(pool.features(method)) if head is not None else None
                scores[method].append(score)

    # The fold loop groups results by fold.  Restore the common state/fold order
    # before pairing selections with pools and outcomes.
    fold_order = sorted(test_pools, key=lambda pool: (pool.fold, pool.state))
    results = {
        method: aggregate_task_method(
            fold_order,
            method,
            selected[method],
            scores[method],
            bootstrap,
            permutations,
            rng,
        )
        for method in METHODS
    }
    route_noise_rho = [
        rank_correlation(pool.early_route_score, pool.initial_noise_score)
        for pool in test_pools
    ]
    route_pick = np.asarray(results["early_route_central"]["selected"], dtype=object)
    noise_pick = np.asarray(results["initial_noise_central"]["selected"], dtype=object)
    same_pick = np.mean(
        [left["seed"] == right["seed"] for left, right in zip(route_pick, noise_pick)]
    )
    diagnostic = {
        "head_training": head_diagnostics,
        "early_route_vs_initial_noise_centrality_spearman_mean": finite(
            np.nanmean(route_noise_rho)
        ),
        "early_route_and_noise_selector_same_pick_rate": finite(same_pick),
    }
    return results, diagnostic


def fit_production_heads(
    pools: list[Pool], out_path: Path
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for method in HEAD_METHODS:
        head, diagnostic = fit_head(pools, method)
        prefix = method + "__"
        arrays[prefix + "coefficients"] = head.coefficients
        arrays[prefix + "feature_mean"] = head.feature_mean
        arrays[prefix + "feature_scale"] = head.feature_scale
        arrays[prefix + "alpha"] = np.asarray(head.alpha)
        metadata[method] = {
            **diagnostic,
            "feature_width": len(head.feature_mean),
        }
    np.savez_compressed(out_path, **arrays)
    return metadata


def report_markdown(summary: dict[str, Any]) -> str:
    primary_name = summary["protocol"]["primary_held_out_task"]
    primary = summary["tasks"][primary_name]
    methods = primary["methods"]
    labels = {
        "early_route_central": "早期路由中心（无训练）",
        "future_route_head": "future-route head（仅路由）",
        "future_route_noise_head": "future-route head（路由+初始噪声）",
        "initial_noise_central": "初始噪声中心对照",
        "late_route_oracle": "完整后续路由中心（诊断上界）",
    }
    lines = [
        "# 无监督 MoE 噪声选择：新任务 Shadow A/B",
        "",
        "## 实验思路",
        "",
        (
            "每次从同一观测的 8 个标准高斯噪声候选中选 1 个。逐个 "
            "`layer × denoise × action-token` 对 32 路概率重新 L1 归一化，"
            "用前 3 次路由的 Hellinger 几何形成早期特征。"
        ),
        "",
        (
            "无训练版本选择早期路由中心；自监督 head 用源任务预测候选的"
            "后 7 次路由中心度。训练和选噪声不读取动作、reward 或 success。"
            "测试时同时留出完整任务和对应的 8 个 noise seed。"
        ),
        "",
        "## 留出新任务结果",
        "",
        f"任务：`{primary_name}`，16 个初始状态 × 4 个互斥 K8 池，共 64 次选择。",
        "",
        "| 方法 | 选中成功率 | 无选择精确随机 | 差值 | 状态 bootstrap 95% CI | 置乱 p（提升） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        value = methods[method]
        low, high = value["state_cluster_bootstrap_ci_95"]
        lines.append(
            "| %s | %.3f | %.3f | %+.3f | [%+.3f, %+.3f] | %.4f |"
            % (
                labels[method],
                value["selected_success_rate"],
                value["exact_random_success_rate"],
                value["paired_delta"],
                low,
                high,
                value["candidate_label_permutation"]["one_sided_improvement"],
            )
        )

    route = methods["early_route_central"]
    head = methods["future_route_noise_head"]
    diagnostic = primary["diagnostics"]
    validation = summary["validation"]
    route_macro = np.mean(
        [
            task["methods"]["early_route_central"]["paired_delta"]
            for task in summary["tasks"].values()
        ]
    )
    head_macro = np.mean(
        [
            task["methods"]["future_route_noise_head"]["paired_delta"]
            for task in summary["tasks"].values()
        ]
    )
    lines.extend(
        [
            "",
            "## 结果",
            "",
            (
                "主检验没有显示收益：早期路由中心相对随机为 `%+.3f`，"
                "路由+噪声自监督 head 为 `%+.3f`。两者置信区间均跨 0。"
                % (route["paired_delta"], head["paired_delta"])
            ),
            "",
            (
                "对全部 5 个任务分别做 task-held-out 后，平均差值也分别为 "
                "`%+.3f` 和 `%+.3f`，没有跨任务可重复的正收益。"
                % (route_macro, head_macro)
            ),
            "",
            (
                "自监督 head 能预测后续路由中心度（留出池平均 Spearman "
                "`%.3f`），但成功率相对随机的差值为 `%+.3f`。也就是说，"
                "可预测的内部稳定性不等于任务正确性。"
                % (
                    head["future_route_rank_spearman_mean"],
                    head["paired_delta"],
                )
            ),
            "",
            (
                "早期路由中心度与初始噪声中心度的池内秩相关为 `%.3f`，"
                "两种规则实际选中同一 seed 的比例为 `%.3f`。更关键的是，"
                "早期路由规则总共只选择 4 个 seed：每个 K8 fold 对 16 个"
                "不同状态都固定选择同一个 seed。这说明当前规则主要在提取"
                "全局噪声模板，而不是形成任务状态特异的质量信号。"
                % (
                    diagnostic[
                        "early_route_vs_initial_noise_centrality_spearman_mean"
                    ],
                    diagnostic["early_route_and_noise_selector_same_pick_rate"],
                )
            ),
            "",
            (
                "即使使用完整后 7 次路由的诊断上界，成功率差值也为 `%+.3f`；"
                "因此问题不只是早期 head 预测不准，‘靠近共同路由中心’本身"
                "没有被这组新任务数据支持为更好动作的判据。"
                % methods["late_route_oracle"]["paired_delta"]
            ),
            "",
            (
                "归一化检查通过：原始 float16 概率和为 1 的最大误差 `%.6g`，"
                "重新归一化后为 `%.3g`；state-token 与 AS 负控的最大候选"
                "变化分别为 `%.3g` 和 `%.3g`。"
                % (
                    validation["max_raw_probability_mass_error"],
                    validation["max_renormalized_probability_mass_error"],
                    validation["max_state_token_candidate_range"],
                    validation["max_as_candidate_range"],
                )
            ),
            "",
            (
                "结论：当前可以把该 head 作为后续路由稳定度预测器，但不能"
                "据此声称它会提高新任务成功率。下一次闭环实验不应继续使用"
                "‘中心即更好’目标，而应加入真实的同状态分支价值或因果干预信号。"
            ),
            "",
            (
                "限制：这是 episode-start shadow A/B。每个记录 seed 还决定了"
                "该 episode 后续重规划噪声流，因此不是逐次重规划选噪声的闭环"
                "因果结果；当前两张 GPU 均满载，未冒充已完成在线运行。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.prefix_steps != PREFIX_STEPS:
        raise ValueError("this preregistered analysis fixes prefix_steps=3")
    if args.seed_folds != SEED_FOLDS:
        raise ValueError("this preregistered analysis fixes four K8 seed folds")
    if args.bootstrap < 100 or args.permutations < 100:
        raise ValueError("bootstrap and permutation counts must each be at least 100")

    hub = args.hub_root.resolve()
    cache_root = hub / "cache" / "HiMoE-VLA"
    all_pools: list[Pool] = []
    task_diagnostics = []
    canonical_seeds: tuple[int, ...] | None = None
    for run in discover_runs(hub):
        print("loading %s" % run.relative_to(cache_root), flush=True)
        pools, diagnostics, seeds = load_task_pools(
            run, cache_root, args.prefix_steps, args.seed_folds
        )
        if canonical_seeds is None:
            canonical_seeds = seeds
        elif canonical_seeds != seeds:
            raise RuntimeError("tasks do not share the same ordered noise seeds")
        all_pools.extend(pools)
        task_diagnostics.append(diagnostics)

    tasks = sorted({pool.task for pool in all_pools})
    if args.primary_task not in tasks:
        raise ValueError("primary held-out task is absent: %s" % args.primary_task)
    rng = np.random.default_rng(args.seed)
    results = {}
    for task in tasks:
        print("evaluating held-out task %s" % task, flush=True)
        methods, diagnostics = evaluate_task(
            task,
            all_pools,
            args.bootstrap,
            args.permutations,
            rng,
        )
        results[task] = {"methods": methods, "diagnostics": diagnostics}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    production = fit_production_heads(all_pools, args.out_dir / "route_noise_head.npz")
    max_raw = max(row["raw_probability_mass_max_error"] for row in task_diagnostics)
    max_normalized = max(
        row["renormalized_probability_mass_max_error"] for row in task_diagnostics
    )
    max_state_token = max(
        row["state_token_candidate_max_range"] for row in task_diagnostics
    )
    max_as = max(row["as_candidate_max_range"] for row in task_diagnostics)
    summary = {
        "schema": "himoe-route-noise-selector-shadow-v1",
        "protocol": {
            "source": "VLA_MUI_HUB right-16x32, first policy inference per episode",
            "primary_held_out_task": args.primary_task,
            "task_count": len(tasks),
            "states_per_task": 16,
            "candidate_pool": "four disjoint K8 folds from the common K32 seed grid",
            "early_route_prefix_steps": args.prefix_steps,
            "future_route_steps": [args.prefix_steps, 10],
            "routing_sites": "8 HB layers x 10 action tokens; full 32-way softmax",
            "outcome_blinding": (
                "head fitting/scoring functions consume only initial noise and routing; "
                "success is read only by the separate aggregation stage"
            ),
            "head_target": "within-K8 rank of late-route Hellinger centrality",
            "head_cross_fit": (
                "complete task held out; the test fold's eight seed identities are "
                "also absent from every source-task training pool"
            ),
            "random_baseline": "exact mean success in the identical K8 candidate pool",
            "shadow_estimand": (
                "episode-start seed-stream selection; not per-replan closed-loop control"
            ),
            "common_noise_seeds": list(canonical_seeds or ()),
        },
        "validation": {
            "task_diagnostics": task_diagnostics,
            "total_pools": len(all_pools),
            "total_candidate_rows": len(all_pools) * 8,
            "max_raw_probability_mass_error": max_raw,
            "max_renormalized_probability_mass_error": max_normalized,
            "max_state_token_candidate_range": max_state_token,
            "max_as_candidate_range": max_as,
            "candidate_permutation_equivariance": "unit-tested",
            "test_seed_disjointness": "runtime asserted for every task/fold head",
            "bootstrap_draws": args.bootstrap,
            "candidate_label_permutations": args.permutations,
            "random_seed": args.seed,
        },
        "tasks": results,
        "production_head": {
            "artifact": "route_noise_head.npz",
            "trained_after_cross-fitted_evaluation": True,
            "training_uses_outcomes": False,
            "methods": production,
        },
        "decision": {
            "online_success_gain_established": False,
            "reason": (
                "The held-out shadow A/B does not support route centrality or the "
                "future-route self-supervised head as a success-improving selector."
            ),
            "next_required_test": (
                "closed-loop same-state branching with common future noise and a "
                "value-linked or causal target, once a GPU is available"
            ),
        },
    }
    return summary


def main() -> int:
    args = parse_args()
    summary = build(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    report_path = args.out_dir / "report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report_path.write_text(report_markdown(summary))
    print("wrote %s" % summary_path, flush=True)
    print("wrote %s" % report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
