#!/usr/bin/env python3
"""Measure whether early HB routing entropy predicts rollout failure.

Entropy is recomputed in float32 from the saved 32-way soft router
probabilities. The main metric compares failures and successes only within the
same initial state. Coarse early summaries and a layer/denoise localization
scan receive separate maxT permutation corrections.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.stats import rankdata

import analyze_early_structure as core


HERE = Path(__file__).resolve().parent
HORIZONS = (0, 7, 12, 20, 27, 34)
EARLY_HORIZONS = (0, 7, 12)
LOCALIZATION_HORIZONS = (7, 12)
RECENT_WINDOW = 7
N_EXPERTS = 32
BOOTSTRAPS = 2_000

TOKEN_GROUPS = {
    "state": slice(0, 1),
    "action_all": slice(1, 11),
    "action_1_3": slice(1, 4),
    "action_4_7": slice(4, 8),
    "action_8_10": slice(8, 11),
    "all_tokens": slice(0, 11),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=core.RUN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--cache",
        type=Path,
        default=HERE / "results" / "route_entropy_values.npz",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--permutations", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def load_entropy(
    run: Path,
    episodes: list[core.Episode],
    cache: Path,
    rebuild: bool,
) -> np.ndarray:
    expected_episode = np.asarray([episode.index for episode in episodes], dtype=np.int32)
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=False) as payload:
            if not np.array_equal(payload["episode"], expected_episode):
                raise ValueError("entropy cache episode order does not match")
            if int(payload["max_control_step"]) != max(HORIZONS):
                raise ValueError("entropy cache horizon does not match")
            return np.asarray(payload["entropy"], dtype=np.float32)

    store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    episode_axis = np.asarray(store["episode_id"][:], dtype=np.int32)
    router = store["hb_router_probs"]
    output = np.empty(
        (len(episodes), max(HORIZONS) + 1, 8, 10, 11), dtype=np.float32
    )
    for position, episode in enumerate(episodes, start=1):
        stop = episode.offset + max(HORIZONS) + 1
        if not np.all(episode_axis[episode.offset:stop] == episode.index):
            raise ValueError(f"episode boundary mismatch for {episode.index}")
        probability = np.asarray(
            router[episode.offset:stop], dtype=np.float32
        )
        probability = np.maximum(probability, 0.0)
        probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-12)
        output[episode.index] = -np.sum(
            probability * np.log(np.maximum(probability, 1e-12)), axis=-1
        ) / math.log(N_EXPERTS)
        if position % 32 == 0:
            print(f"entropy {position:03d}/{len(episodes)}", flush=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        entropy=output,
        episode=expected_episode,
        max_control_step=np.asarray(max(HORIZONS), dtype=np.int16),
    )
    return output


def time_slope(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=np.float32)
    coordinate = np.linspace(-1.0, 1.0, values.shape[1], dtype=np.float32)
    return (values * coordinate[None]).sum(axis=1) / np.square(coordinate).sum()


def coarse_features(entropy: np.ndarray) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for horizon in HORIZONS:
        for group, token_slice in TOKEN_GROUPS.items():
            series = entropy[:, : horizon + 1, :, :, token_slice].mean(
                axis=(2, 3, 4)
            )
            features[f"t{horizon}/current/{group}"] = series[:, -1]
            if horizon == 0:
                continue
            recent = series[:, max(0, horizon - RECENT_WINDOW + 1) :]
            features[f"t{horizon}/prefix_mean/{group}"] = series.mean(axis=1)
            features[f"t{horizon}/recent_mean/{group}"] = recent.mean(axis=1)
            features[f"t{horizon}/recent_slope/{group}"] = time_slope(recent)
            features[f"t{horizon}/recent_std/{group}"] = recent.std(axis=1)
    return features


def localization_features(entropy: np.ndarray) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {}
    for horizon in LOCALIZATION_HORIZONS:
        action = entropy[:, horizon, :, :, 1:11]
        for layer in range(action.shape[1]):
            features[f"t{horizon}/current/action/L{layer}"] = action[:, layer].mean(
                axis=(1, 2)
            )
        for denoise in range(action.shape[2]):
            features[f"t{horizon}/current/action/d{denoise}"] = action[
                :, :, denoise
            ].mean(axis=(1, 2))
        for token in range(action.shape[3]):
            features[f"t{horizon}/current/action/T{token + 1}"] = action[
                :, :, :, token
            ].mean(axis=(1, 2))
        for layer in range(action.shape[1]):
            for denoise in range(action.shape[2]):
                features[
                    f"t{horizon}/current/action/L{layer}_d{denoise}"
                ] = action[:, layer, denoise].mean(axis=1)
    return features


def feature_matrix(
    features: dict[str, np.ndarray]
) -> tuple[list[str], np.ndarray]:
    keys = list(features)
    matrix = np.column_stack([features[key] for key in keys]).astype(np.float64)
    return keys, matrix


def auc_from_categories(
    scores: np.ndarray,
    category: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    numerator = np.zeros(scores.shape[1], dtype=np.float64)
    denominator = 0
    state_aucs = []
    state_pairs = []
    for state_value in np.unique(state):
        index = np.flatnonzero((state == state_value) & (category >= 0))
        labels = category[index]
        positive = int((labels == 1).sum())
        negative = int((labels == 0).sum())
        if positive == 0 or negative == 0:
            continue
        ranks = rankdata(scores[index], axis=0, method="average")
        rank_sum = (labels == 1).astype(np.float64) @ ranks
        pairs = positive * negative
        auc = (rank_sum - positive * (positive + 1) / 2.0) / pairs
        numerator += auc * pairs
        denominator += pairs
        state_aucs.append(auc)
        state_pairs.append(pairs)
    if denominator == 0:
        raise ValueError("no within-state positive/negative pairs")
    state_matrix = np.stack(state_aucs)
    return (
        numerator / denominator,
        state_matrix.mean(axis=0),
        state_matrix,
        denominator,
    )


def pair_weighted_delta(
    scores: np.ndarray,
    category: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    numerator = np.zeros(scores.shape[1], dtype=np.float64)
    denominator = 0
    for state_value in np.unique(state):
        positive = np.flatnonzero((state == state_value) & (category == 1))
        negative = np.flatnonzero((state == state_value) & (category == 0))
        if not len(positive) or not len(negative):
            continue
        pairs = len(positive) * len(negative)
        numerator += (scores[positive].mean(axis=0) - scores[negative].mean(axis=0)) * pairs
        denominator += pairs
    return numerator / denominator


def common_seed_permutations(
    category: np.ndarray,
    state: np.ndarray,
    noise_seed: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    states = np.unique(state)
    seeds = np.unique(noise_seed)
    if np.any(category < 0):
        raise ValueError("common seed-column permutations require a complete target")
    index = np.empty((len(states), len(seeds)), dtype=np.int32)
    for state_index, state_value in enumerate(states):
        for seed_index, seed_value in enumerate(seeds):
            match = np.flatnonzero(
                (state == state_value) & (noise_seed == seed_value)
            )
            if len(match) != 1:
                raise ValueError("expected a complete state x seed grid")
            index[state_index, seed_index] = match[0]
    orders = np.argsort(rng.random((count, len(seeds))), axis=1)
    output = np.empty((count, len(category)), dtype=np.int8)
    for row in index:
        output[:, row] = category[row][orders]
    return output


def within_state_permutations(
    category: np.ndarray,
    state: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.tile(category, (count, 1))
    for state_value in np.unique(state):
        index = np.flatnonzero((state == state_value) & (category >= 0))
        orders = np.argsort(rng.random((count, len(index))), axis=1)
        output[:, index] = category[index][orders]
    return output


def state_auc_table(
    scores: np.ndarray,
    category: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.unique(state)
    output = np.full((len(states), scores.shape[1]), 0.5, dtype=np.float64)
    weights = np.zeros(len(states), dtype=np.float64)
    for state_index, state_value in enumerate(states):
        index = np.flatnonzero((state == state_value) & (category >= 0))
        labels = category[index]
        positive = int((labels == 1).sum())
        negative = int((labels == 0).sum())
        if positive == 0 or negative == 0:
            continue
        ranks = rankdata(scores[index], axis=0, method="average")
        rank_sum = (labels == 1).astype(np.float64) @ ranks
        weights[state_index] = positive * negative
        output[state_index] = (
            rank_sum - positive * (positive + 1) / 2.0
        ) / weights[state_index]
    return output, weights


def permutation_state_auc(
    scores: np.ndarray,
    categories: np.ndarray,
    observed_category: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.unique(state)
    output = np.full(
        (len(categories), len(states), scores.shape[1]),
        0.5,
        dtype=np.float64,
    )
    weights = np.zeros(len(states), dtype=np.float64)
    for state_index, state_value in enumerate(states):
        index = np.flatnonzero(
            (state == state_value) & (observed_category >= 0)
        )
        observed = observed_category[index]
        positive = int((observed == 1).sum())
        negative = int((observed == 0).sum())
        if positive == 0 or negative == 0:
            continue
        ranks = rankdata(scores[index], axis=0, method="average")
        labels = categories[:, index] == 1
        permuted_positive = labels.sum(axis=1)
        permuted_negative = len(index) - permuted_positive
        permuted_pairs = permuted_positive * permuted_negative
        rank_sum = labels.astype(np.float64) @ ranks
        output[:, state_index] = (
            rank_sum
            - (permuted_positive * (permuted_positive + 1) / 2.0)[:, None]
        ) / permuted_pairs[:, None]
        weights[state_index] = positive * negative
    return output, weights


def crossfit_state_holdout(
    feature_keys: list[str],
    observed_state_auc: np.ndarray,
    null_state_auc: np.ndarray,
    state_weights: np.ndarray,
    state_values: np.ndarray,
    fold_mode: str,
) -> dict[str, Any]:
    if fold_mode == "four-fold":
        folds = np.array_split(np.arange(len(state_weights)), 4)
    elif fold_mode == "leave-one-state-out":
        folds = [np.asarray([index]) for index in range(len(state_weights))]
    else:
        raise ValueError(f"unknown state fold mode: {fold_mode}")
    observed_numerator = 0.0
    null_numerator = np.zeros(len(null_state_auc), dtype=np.float64)
    denominator = 0.0
    selections = []
    for held in folds:
        train = np.setdiff1d(np.arange(len(state_weights)), held)
        train_weight = state_weights[train].sum()
        test_weight = state_weights[held].sum()
        if test_weight == 0:
            continue
        if train_weight == 0:
            raise ValueError("state-holdout training fold lacks within-state pairs")

        train_auc = (
            observed_state_auc[train] * state_weights[train, None]
        ).sum(axis=0) / train_weight
        selected = int(np.argmax(np.abs(train_auc - 0.5)))
        direction = 1 if train_auc[selected] >= 0.5 else -1
        test_auc = float(
            (
                observed_state_auc[held, selected]
                * state_weights[held]
            ).sum()
            / test_weight
        )
        oriented_test = test_auc if direction > 0 else 1.0 - test_auc
        observed_numerator += oriented_test * test_weight
        denominator += test_weight
        selections.append(
            {
                "held_states": state_values[held].tolist(),
                "feature": feature_keys[selected],
                "training_auc": float(train_auc[selected]),
                "test_oriented_auc": oriented_test,
            }
        )

        null_train_auc = (
            null_state_auc[:, train]
            * state_weights[None, train, None]
        ).sum(axis=1) / train_weight
        null_selected = np.argmax(np.abs(null_train_auc - 0.5), axis=1)
        null_direction = np.where(
            null_train_auc[np.arange(len(null_train_auc)), null_selected] >= 0.5,
            1.0,
            -1.0,
        )
        null_test_auc = (
            null_state_auc[:, held]
            * state_weights[None, held, None]
        ).sum(axis=1) / test_weight
        selected_test_auc = null_test_auc[
            np.arange(len(null_test_auc)), null_selected
        ]
        oriented_null = np.where(
            null_direction > 0,
            selected_test_auc,
            1.0 - selected_test_auc,
        )
        null_numerator += oriented_null * test_weight

    observed = observed_numerator / denominator
    null = null_numerator / denominator
    return {
        "auc": float(observed),
        "one_sided_reselection_p": float(
            (1 + np.sum(null >= observed)) / (len(null) + 1)
        ),
        "null_quantiles": {
            str(q): float(np.quantile(null, q)) for q in (0.5, 0.9, 0.95, 0.99)
        },
        "fold_selections": selections,
        "scope": f"{fold_mode} whole-initial-state holdout; feature and direction reselected in each training fold",
    }


def hierarchical_bootstrap(
    scores: np.ndarray,
    category: np.ndarray,
    state: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    informative = []
    for state_value in np.unique(state):
        positive = np.flatnonzero((state == state_value) & (category == 1))
        negative = np.flatnonzero((state == state_value) & (category == 0))
        if len(positive) and len(negative):
            informative.append((positive, negative))
    output = np.empty((BOOTSTRAPS, scores.shape[1]), dtype=np.float64)
    for bootstrap in range(BOOTSTRAPS):
        numerator = np.zeros(scores.shape[1], dtype=np.float64)
        denominator = 0
        selected_states = rng.integers(0, len(informative), size=len(informative))
        for selected in selected_states:
            positive, negative = informative[selected]
            pos = scores[rng.choice(positive, size=len(positive), replace=True)]
            neg = scores[rng.choice(negative, size=len(negative), replace=True)]
            comparison = pos[:, None, :] - neg[None, :, :]
            pairs = len(positive) * len(negative)
            numerator += (
                (comparison > 0).sum(axis=(0, 1))
                + 0.5 * (comparison == 0).sum(axis=(0, 1))
            )
            denominator += pairs
        output[bootstrap] = numerator / denominator
    return np.quantile(output, (0.025, 0.975), axis=0).T


def evaluate_target(
    name: str,
    category: np.ndarray,
    permutation_mode: str,
    keys: list[str],
    scores: np.ndarray,
    coarse_indexes: np.ndarray,
    localization_indexes: np.ndarray,
    state: np.ndarray,
    noise_seed: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    auc, macro_auc, state_auc, pair_count = auc_from_categories(
        scores, category, state
    )
    delta = pair_weighted_delta(scores, category, state)
    rng = np.random.default_rng(seed)
    if permutation_mode == "common-seed-column":
        shuffled = common_seed_permutations(
            category, state, noise_seed, permutations, rng
        )
    else:
        shuffled = within_state_permutations(category, state, permutations, rng)
    tested = np.r_[coarse_indexes, localization_indexes]
    tested_scores = scores[:, tested]
    observed_state, state_weights = state_auc_table(
        tested_scores, category, state
    )
    null_state, null_state_weights = permutation_state_auc(
        tested_scores, shuffled, category, state
    )
    if not np.array_equal(state_weights, null_state_weights):
        raise ValueError("observed and permutation state weights differ")
    null = (
        null_state * state_weights[None, :, None]
    ).sum(axis=1) / state_weights.sum()
    observed_deviation = np.abs(auc[tested] - 0.5)
    null_deviation = np.abs(null - 0.5)
    raw_p = (1 + (null_deviation >= observed_deviation[None]).sum(axis=0)) / (
        permutations + 1
    )
    coarse_count = len(coarse_indexes)
    family_max = {
        "coarse_early": null_deviation[:, :coarse_count].max(axis=1),
        "localization": null_deviation[:, coarse_count:].max(axis=1),
        "all_early": null_deviation.max(axis=1),
    }

    rows: dict[str, Any] = {}
    test_position = {int(index): position for position, index in enumerate(tested)}
    coarse_set = set(coarse_indexes.tolist())
    for index, key in enumerate(keys):
        item = {
            "auc_failure_higher": float(auc[index]),
            "discrimination": float(max(auc[index], 1.0 - auc[index])),
            "macro_state_auc": float(macro_auc[index]),
            "pair_weighted_entropy_delta": float(delta[index]),
            "states_higher": int((state_auc[:, index] > 0.5).sum()),
            "states_lower": int((state_auc[:, index] < 0.5).sum()),
            "informative_states": int(state_auc.shape[0]),
        }
        if index in test_position:
            position = test_position[index]
            family = "coarse_early" if index in coarse_set else "localization"
            item.update(
                {
                    "raw_two_sided_p": float(raw_p[position]),
                    "family_maxT_p": float(
                        (
                            1
                            + (
                                family_max[family]
                                >= observed_deviation[position]
                            ).sum()
                        )
                        / (permutations + 1)
                    ),
                    "all_early_maxT_p": float(
                        (
                            1
                            + (
                                family_max["all_early"]
                                >= observed_deviation[position]
                            ).sum()
                        )
                        / (permutations + 1)
                    ),
                }
            )
        rows[key] = item

    best_coarse = max(
        coarse_indexes, key=lambda index: abs(float(auc[index]) - 0.5)
    )
    best_local = max(
        localization_indexes, key=lambda index: abs(float(auc[index]) - 0.5)
    )
    tested_keys = [keys[index] for index in tested]
    state_holdout = {}
    for fold_mode in ("four-fold", "leave-one-state-out"):
        state_holdout[fold_mode] = {
            "coarse_early": crossfit_state_holdout(
                tested_keys[:coarse_count],
                observed_state[:, :coarse_count],
                null_state[:, :, :coarse_count],
                state_weights,
                np.unique(state),
                fold_mode,
            ),
            "localization": crossfit_state_holdout(
                tested_keys[coarse_count:],
                observed_state[:, coarse_count:],
                null_state[:, :, coarse_count:],
                state_weights,
                np.unique(state),
                fold_mode,
            ),
        }
    return {
        "name": name,
        "permutation_mode": permutation_mode,
        "permutations": permutations,
        "pair_count": pair_count,
        "rows": rows,
        "best_coarse_early": {
            "feature": keys[best_coarse], **rows[keys[best_coarse]]
        },
        "best_localization": {
            "feature": keys[best_local], **rows[keys[best_local]]
        },
        "state_holdout_crossfit": state_holdout,
        "null_max_abs_auc_minus_half_quantiles": {
            family: {
                str(q): float(np.quantile(values, q))
                for q in (0.5, 0.9, 0.95, 0.99)
            }
            for family, values in family_max.items()
        },
    }


def report_keys() -> list[str]:
    keys = []
    for horizon in EARLY_HORIZONS:
        for group in ("state", "action_all", "all_tokens"):
            keys.append(f"t{horizon}/current/{group}")
        if horizon:
            keys.append(f"t{horizon}/recent_mean/action_all")
            keys.append(f"t{horizon}/recent_slope/action_all")
    for horizon in (20, 27, 34):
        for group in ("state", "action_all", "all_tokens"):
            keys.append(f"t{horizon}/current/{group}")
    return keys


def render_plot(
    output: Path,
    results: dict[str, Any],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    any_rows = results["targets"]["any_timeout_failure"]["rows"]
    colors = {"state": "#777777", "action_all": "#2878b5", "all_tokens": "#b45f06"}
    for group, color in colors.items():
        values = [any_rows[f"t{h}/current/{group}"]["auc_failure_higher"] for h in HORIZONS]
        axes[0].plot(HORIZONS, values, marker="o", color=color, label=group)
    axes[0].axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    axes[0].axvspan(0, 12, color="#dddddd", alpha=0.25)
    axes[0].set(
        title="Current routing entropy",
        xlabel="control step",
        ylabel="within-state AUC (higher = failure)",
        ylim=(0.2, 0.82),
    )
    axes[0].legend(fontsize=8)

    groups = ("state", "action_1_3", "action_4_7", "action_8_10", "action_all")
    x = np.arange(len(groups))
    width = 0.25
    for offset, target, color in (
        (-width, "any_timeout_failure", "#4c78a8"),
        (0, "stasis_vs_rest", "#59a14f"),
        (width, "clean_stasis_failure", "#f58518"),
    ):
        rows = results["targets"][target]["rows"]
        values = [rows[f"t12/current/{group}"]["auc_failure_higher"] for group in groups]
        axes[1].bar(x + offset, values, width, label=target, color=color)
    axes[1].axhline(0.5, color="#222222", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, ("state", "a1-3", "a4-7", "a8-10", "all action"))
    axes[1].set(title="q12 token groups", ylabel="within-state AUC", ylim=(0.35, 0.68))
    axes[1].legend(fontsize=7)

    rows = results["targets"]["any_timeout_failure"]["rows"]
    values = [rows[f"t12/current/{group}"]["pair_weighted_entropy_delta"] * 1e4 for group in groups]
    axes[2].bar(x, values, color=["#777777", "#6a9fb5", "#6a9fb5", "#6a9fb5", "#2878b5"])
    axes[2].axhline(0, color="#222222", linewidth=1)
    axes[2].set_xticks(x, ("state", "a1-3", "a4-7", "a8-10", "all action"))
    axes[2].set(title="q12 entropy difference", ylabel="failure - success (x 1e-4)")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(output: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 前期路由熵信号",
        "",
        "## 设计",
        "",
        "- 数据：scene8，16 个初始状态 × 32 个 flow-noise seed，共 512 条 rollout。",
        "- 熵：从完整 32-way soft router probabilities 以 float32 重算 Shannon entropy，并除以 log(32)；1 表示均匀路由。",
        "- 主指标：同一初始状态内的 pair-weighted AUC。AUC > 0.5 表示失败分支熵更高，AUC < 0.5 表示失败分支熵更低。",
        "- 标签 A：任意 timeout failure（216）vs success（296）；使用共同 seed-column 置换，保留跨初态共享 seed 结构。",
        "- 标签 B：行为定义的停滞失败（197）vs 其余全部（315）；使用共同 seed-column 置换。",
        "- 标签 C：停滞失败（197）vs success（296），排除 19 条其他失败；使用初态内置换。",
        f"- 搜索校正：{summary['permutations']} 次置换；粗粒度前期统计与 q7/q12 层/去噪定位扫描分别做 maxT。",
        "",
        "## 早期结果",
        "",
        "| target | feature | AUC [hierarchical 95% CI] | discrimination | entropy delta | raw p | coarse maxT p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    selected = (
        "t0/current/action_all",
        "t7/current/action_all",
        "t7/recent_mean/action_all",
        "t12/current/action_all",
        "t12/recent_mean/action_all",
        "t12/recent_slope/action_all",
        "t12/current/state",
        "t12/current/all_tokens",
    )
    for target in (
        "any_timeout_failure",
        "stasis_vs_rest",
        "clean_stasis_failure",
    ):
        rows = summary["targets"][target]["rows"]
        for key in selected:
            row = rows[key]
            ci = row["hierarchical_bootstrap_95_ci"]
            lines.append(
                f"| {target} | {key} | {row['auc_failure_higher']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] "
                f"| {row['discrimination']:.3f} "
                f"| {row['pair_weighted_entropy_delta']:+.2e} | {row.get('raw_two_sided_p', float('nan')):.4f} "
                f"| {row.get('family_maxT_p', float('nan')):.4f} |"
            )

    lines.extend(
        [
            "",
            "## q12 token 位置",
            "",
            "| target | token group | AUC | entropy delta | states higher / informative |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for target in (
        "any_timeout_failure",
        "stasis_vs_rest",
        "clean_stasis_failure",
    ):
        rows = summary["targets"][target]["rows"]
        for group in ("state", "action_1_3", "action_4_7", "action_8_10", "action_all"):
            row = rows[f"t12/current/{group}"]
            lines.append(
                f"| {target} | {group} | {row['auc_failure_higher']:.3f} "
                f"| {row['pair_weighted_entropy_delta']:+.2e} "
                f"| {row['states_higher']}/{row['informative_states']} |"
            )

    lines.extend(["", "## 搜索审计", ""])
    for target in (
        "any_timeout_failure",
        "stasis_vs_rest",
        "clean_stasis_failure",
    ):
        result = summary["targets"][target]
        coarse = result["best_coarse_early"]
        local = result["best_localization"]
        lines.append(
            f"- `{target}` 粗粒度最强：`{coarse['feature']}`，AUC={coarse['auc_failure_higher']:.3f}，"
            f"coarse maxT p={coarse['family_maxT_p']:.4f}；定位扫描最强：`{local['feature']}`，"
            f"AUC={local['auc_failure_higher']:.3f}，localization maxT p={local['family_maxT_p']:.4f}。"
        )

    lines.extend(["", "## 整初态留出", ""])
    for target in (
        "any_timeout_failure",
        "stasis_vs_rest",
        "clean_stasis_failure",
    ):
        crossfits = summary["targets"][target]["state_holdout_crossfit"]
        for mode, description in (
            ("four-fold", "4-fold"),
            ("leave-one-state-out", "LOSO"),
        ):
            coarse = crossfits[mode]["coarse_early"]
            local = crossfits[mode]["localization"]
            lines.append(
                f"- `{target}` {description}：重选粗粒度特征后 AUC={coarse['auc']:.3f}，"
                f"reselection p={coarse['one_sided_reselection_p']:.4f}；重选层/去噪位置后 AUC={local['auc']:.3f}，"
                f"reselection p={local['one_sided_reselection_p']:.4f}。"
            )

    lines.extend(
        [
            "",
            "## 后期对照",
            "",
            "| q | state-token current | action current | all-token current |",
            "|---:|---:|---:|---:|",
        ]
    )
    rows = summary["targets"]["any_timeout_failure"]["rows"]
    for horizon in (20, 27, 34):
        values = [
            rows[f"t{horizon}/current/{group}"]["auc_failure_higher"]
            for group in ("state", "action_all", "all_tokens")
        ]
        lines.append(
            f"| {horizon} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- q0/q7 没有稳定的全局路由熵信号。",
            "- q12 action entropy 是弱、局部且初态异质的相关信号；它不等同于已经识别到停滞。",
            "- 整初态留出后，任意超时和 clean 停滞目标都回到机会线；当前没有可迁移的单独熵预警量。",
            "- `stasis_vs_rest` 的 q12 定位在 LOSO 下仅为边缘趋势，且 4-fold 结果不一致，必须在新任务/新初态上固定位置后再验证。",
            "- normalized entropy 整体接近 1，组间差值很小，因此它更适合作为组合特征，而不是单独告警量。",
            "- q20 以后仅作为阶段读数；多数停滞失败的行为 onset 中位为 t19，不能称为前兆。",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    episodes = core.load_episodes(args.run)
    entropy = load_entropy(args.run, episodes, args.cache, args.rebuild_cache)
    sim = core.load_sim(args.run, episodes)
    stasis, included, onset, target_metadata = core.build_targets(episodes, sim)
    failure = np.asarray([episode.failure for episode in episodes], dtype=np.int8)
    state = np.asarray([episode.state for episode in episodes], dtype=np.int16)
    noise_seed = np.asarray([episode.noise_seed for episode in episodes], dtype=np.int16)

    coarse = coarse_features(entropy)
    localization = localization_features(entropy)
    combined = {**coarse, **localization}
    keys, scores = feature_matrix(combined)
    coarse_keys = [
        key
        for key in coarse
        if int(key.split("/")[0][1:]) in EARLY_HORIZONS
    ]
    coarse_indexes = np.asarray([keys.index(key) for key in coarse_keys], dtype=np.int32)
    localization_indexes = np.asarray(
        [keys.index(key) for key in localization], dtype=np.int32
    )

    clean_category = np.full(len(episodes), -1, dtype=np.int8)
    clean_category[included] = stasis[included]
    targets = {
        "any_timeout_failure": evaluate_target(
            "any_timeout_failure",
            failure,
            "common-seed-column",
            keys,
            scores,
            coarse_indexes,
            localization_indexes,
            state,
            noise_seed,
            args.permutations,
            args.seed + 1,
        ),
        "stasis_vs_rest": evaluate_target(
            "stasis_vs_rest",
            stasis,
            "common-seed-column",
            keys,
            scores,
            coarse_indexes,
            localization_indexes,
            state,
            noise_seed,
            args.permutations,
            args.seed + 2,
        ),
        "clean_stasis_failure": evaluate_target(
            "clean_stasis_failure",
            clean_category,
            "within-state",
            keys,
            scores,
            coarse_indexes,
            localization_indexes,
            state,
            noise_seed,
            args.permutations,
            args.seed + 3,
        ),
    }

    selected = report_keys()
    selected_indexes = np.asarray([keys.index(key) for key in selected], dtype=np.int32)
    for target_name, category in (
        ("any_timeout_failure", failure),
        ("stasis_vs_rest", stasis),
        ("clean_stasis_failure", clean_category),
    ):
        ci = hierarchical_bootstrap(
            scores[:, selected_indexes],
            category,
            state,
            np.random.default_rng(args.seed + 17 + len(target_name)),
        )
        for key, interval in zip(selected, ci):
            targets[target_name]["rows"][key]["hierarchical_bootstrap_95_ci"] = [
                float(interval[0]),
                float(interval[1]),
            ]

    summary = {
        "task": core.TASK,
        "entropy": {
            "definition": "Shannon entropy over 32 soft experts divided by log(32)",
            "source": "float32 recomputation from hb_router_probs",
            "overall_mean": float(entropy.mean()),
            "overall_min": float(entropy.min()),
            "overall_max": float(entropy.max()),
        },
        "horizons": list(HORIZONS),
        "recent_window": RECENT_WINDOW,
        "permutations": args.permutations,
        "coarse_early_tests": len(coarse_indexes),
        "localization_tests": len(localization_indexes),
        "target_metadata": target_metadata,
        "onset_median": float(np.median(onset[onset >= 0])),
        "targets": targets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "route_entropy_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_report(args.output_dir / "route_entropy_report.md", summary)
    render_plot(args.output_dir / "route_entropy.png", summary)
    np.savez_compressed(
        args.output_dir / "route_entropy_scores.npz",
        feature=np.asarray(keys),
        score=scores.astype(np.float32),
        failure=failure,
        stasis=stasis,
        included=included,
        state=state,
        noise_seed=noise_seed,
    )
    print(f"wrote {args.output_dir / 'route_entropy_report.md'}", flush=True)


if __name__ == "__main__":
    main()
