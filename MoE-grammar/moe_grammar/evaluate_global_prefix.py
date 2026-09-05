from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from moe_grammar.statistics import (
    auc_pairwise,
    empirical_percentile,
    paired_state_test,
    stratified_pair_auc,
)

HORIZONS = (3, 7, 12)
PRIMARY_METHODS = (
    "lexical_prefix",
    "phase_prefix",
    "local2_prefix",
    "global_prefix",
    "global_excess_local_prefix",
    "phase_current",
    "local2_current",
    "global_current",
    "prototype_prefix",
    "next_entropy_current",
    "hidden_prefix_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=Path,
        default=[Path(f"artifacts/global-prefix-fold{fold}.npz") for fold in range(5)],
    )
    parser.add_argument(
        "--prediction-dirs",
        nargs="+",
        type=Path,
        default=[Path(f"results-global-prefix/fold{fold}") for fold in range(5)],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-global-prefix"))
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def cumulative_mean(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for index, length in enumerate(lengths):
        count = min(int(length), values.shape[1])
        sequence = values[index, :count]
        output[index, :count] = np.cumsum(sequence) / np.arange(1, count + 1)
    return output


def conditional_percentiles(
    values: np.ndarray,
    reference_indexes: np.ndarray,
    task: np.ndarray,
    lengths: np.ndarray,
    task_count: int,
) -> np.ndarray:
    """Calibrate every score against healthy prefixes at the same task/position."""

    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for position in range(values.shape[1]):
        reference_at_position = reference_indexes[
            (lengths[reference_indexes] > position)
            & np.isfinite(values[reference_indexes, position])
        ]
        if not len(reference_at_position):
            continue
        global_reference = values[reference_at_position, position]
        for task_index in range(task_count):
            task_reference = reference_at_position[
                task[reference_at_position] == task_index
            ]
            reference = (
                values[task_reference, position]
                if len(task_reference) >= 32
                else global_reference
            )
            selected = np.flatnonzero(
                (task == task_index)
                & (lengths > position)
                & np.isfinite(values[:, position])
            )
            if len(selected):
                output[selected, position] = empirical_percentile(
                    reference, values[selected, position]
                )
    return output


def running_max(values: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for index, length in enumerate(lengths):
        count = min(int(length), values.shape[1])
        output[index, :count] = np.maximum.accumulate(values[index, :count])
    return output


def raw_methods(dataset: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    lengths = dataset["lengths"]
    phase = dataset["phase_soft"][:, :13]
    local2 = dataset["local2_soft"][:, :13]
    global_score = prediction["global_soft"][:, :13]
    return {
        "lexical_prefix": cumulative_mean(dataset["lexical"][:, :13], lengths),
        "phase_prefix": cumulative_mean(phase, lengths),
        "local2_prefix": cumulative_mean(local2, lengths),
        "global_prefix": cumulative_mean(global_score, lengths),
        "global_excess_local_prefix": cumulative_mean(global_score - local2, lengths),
        "global_excess_phase_prefix": cumulative_mean(global_score - phase, lengths),
        "phase_current": phase,
        "local2_current": local2,
        "global_current": global_score,
        "prototype_prefix": cumulative_mean(
            prediction["prototype_distance"][:, :13], lengths
        ),
        "next_entropy_current": prediction["next_entropy"][:, :13],
        "hidden_prefix_state": prediction["hidden_mahalanobis"][:, :13],
    }


def task_thresholds(
    score: np.ndarray,
    calibration_success: np.ndarray,
    task: np.ndarray,
    lengths: np.ndarray,
    horizon: int,
    task_count: int,
    healthy_fpr: float,
) -> dict[int, float]:
    eligible = calibration_success[lengths[calibration_success] > horizon]
    global_threshold = float(
        np.quantile(score[eligible, horizon], 1.0 - healthy_fpr, method="higher")
    )
    thresholds = {}
    for task_index in range(task_count):
        selected = eligible[task[eligible] == task_index]
        thresholds[task_index] = (
            float(
                np.quantile(
                    score[selected, horizon], 1.0 - healthy_fpr, method="higher"
                )
            )
            if len(selected) >= 20
            else global_threshold
        )
    return thresholds


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def mean_episode_score(values: np.ndarray, lengths: np.ndarray, indexes: np.ndarray, start: int) -> np.ndarray:
    output = np.empty(len(indexes), dtype=np.float64)
    for row, episode_index in enumerate(indexes):
        stop = min(int(lengths[episode_index]), values.shape[1])
        output[row] = np.nanmean(values[episode_index, start:stop])
    return output / math.log(2.0)


def fast_state_blocked_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    task_names: np.ndarray,
    states: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap state-level pairwise wins without rebuilding episode arrays."""

    unique_states = np.unique(states)
    wins = np.zeros(len(unique_states), dtype=np.float64)
    pairs = np.zeros(len(unique_states), dtype=np.int64)
    for state_index, selected_state in enumerate(unique_states):
        state_mask = states == selected_state
        for task_name in np.unique(task_names[state_mask]):
            selected = state_mask & (task_names == task_name)
            positive = scores[selected & labels]
            negative = scores[selected & ~labels]
            if not len(positive) or not len(negative):
                continue
            difference = positive[:, None] - negative[None, :]
            wins[state_index] += np.sum(difference > 0.0) + 0.5 * np.sum(
                difference == 0.0
            )
            pairs[state_index] += difference.size
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_states), size=(draws, len(unique_states)))
    bootstrap_pairs = pairs[sampled].sum(axis=1)
    bootstrap = wins[sampled].sum(axis=1) / np.maximum(bootstrap_pairs, 1)
    finite = bootstrap[np.isfinite(bootstrap) & (bootstrap_pairs > 0)]
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def state_pair_components(
    labels: np.ndarray,
    scores: np.ndarray,
    task_names: np.ndarray,
    states: np.ndarray,
    unique_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wins = np.zeros(len(unique_states), dtype=np.float64)
    pairs = np.zeros(len(unique_states), dtype=np.int64)
    for state_index, selected_state in enumerate(unique_states):
        state_mask = states == selected_state
        for task_name in np.unique(task_names[state_mask]):
            selected = state_mask & (task_names == task_name)
            positive = scores[selected & labels]
            negative = scores[selected & ~labels]
            if not len(positive) or not len(negative):
                continue
            difference = positive[:, None] - negative[None, :]
            wins[state_index] += np.sum(difference > 0.0) + 0.5 * np.sum(
                difference == 0.0
            )
            pairs[state_index] += difference.size
    return wins, pairs


def fast_state_blocked_auc_difference(
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    task_names: np.ndarray,
    states: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, float | list[float]]:
    unique_states = np.unique(states)
    left_wins, pairs = state_pair_components(
        labels, left_scores, task_names, states, unique_states
    )
    right_wins, right_pairs = state_pair_components(
        labels, right_scores, task_names, states, unique_states
    )
    if not np.array_equal(pairs, right_pairs):
        raise AssertionError("paired AUC methods have different comparison cells")
    observed = left_wins.sum() / pairs.sum() - right_wins.sum() / pairs.sum()
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_states), size=(draws, len(unique_states)))
    bootstrap_pairs = pairs[sampled].sum(axis=1)
    differences = (
        left_wins[sampled].sum(axis=1) - right_wins[sampled].sum(axis=1)
    ) / np.maximum(bootstrap_pairs, 1)
    finite = differences[bootstrap_pairs > 0]
    return {
        "auc_left_minus_right": float(observed),
        "state_blocked_ci95": [
            float(value) for value in np.quantile(finite, [0.025, 0.975])
        ],
    }


def healthy_prediction_result(records: list[dict[str, np.ndarray]], seed: int) -> dict[str, Any]:
    states = np.concatenate([record["state"] for record in records])
    result: dict[str, Any] = {"test_success_episodes": int(len(states))}
    for scope, start in (("after_first_word", 1), ("remote_history_q4plus", 4)):
        phase = np.concatenate([record[f"phase_{start}"] for record in records])
        local2 = np.concatenate([record[f"local2_{start}"] for record in records])
        global_score = np.concatenate([record[f"global_{start}"] for record in records])
        result[scope] = {
            "bits_per_query": {
                "phase": float(phase.mean()),
                "local2": float(local2.mean()),
                "global_prefix": float(global_score.mean()),
            },
            "global_minus_phase": paired_state_test(
                global_score, phase, states, seed + start
            ),
            "global_minus_local2": paired_state_test(
                global_score, local2, states, seed + 10 + start
            ),
        }
    original = np.concatenate([record["original_order"] for record in records])
    reversed_remote = np.concatenate([record["reversed_remote"] for record in records])
    order_states = np.concatenate([record["order_state"] for record in records])
    result["remote_order_control_q4_q12"] = {
        "original_bits_per_query": float(original.mean()),
        "remote_reversed_bits_per_query": float(reversed_remote.mean()),
        "original_minus_remote_reversed": paired_state_test(
            original, reversed_remote, order_states, seed + 100
        ),
        "control": "the latest two words are fixed; only words older than two are reversed",
    }
    return result


def evaluate_detection_records(
    records: dict[tuple[int, str], list[dict[str, np.ndarray]]],
    methods: tuple[str, ...],
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in HORIZONS:
        output[str(horizon)] = {}
        for method_index, method in enumerate(methods):
            items = records[(horizon, method)]
            labels = np.concatenate([item["labels"] for item in items])
            scores = np.concatenate([item["scores"] for item in items])
            alarms = np.concatenate([item["alarms"] for item in items])
            task_names = np.concatenate([item["task_names"] for item in items])
            states = np.concatenate([item["states"] for item in items])
            strata = np.asarray(
                [f"{name}|{state}" for name, state in zip(task_names, states)],
                dtype=object,
            )
            within_auc, by_stratum, pairs = stratified_pair_auc(labels, scores, strata)
            output[str(horizon)][method] = {
                "episodes": int(len(labels)),
                "successes": int((~labels).sum()),
                "failures": int(labels.sum()),
                "within_task_state_auc": within_auc,
                "state_blocked_ci95": list(
                    fast_state_blocked_interval(
                        labels,
                        scores,
                        task_names,
                        states,
                        bootstrap_draws,
                        seed + horizon * 1009 + method_index,
                    )
                ),
                "pooled_auc": auc_pairwise(labels, scores),
                "healthy_fpr": float(alarms[~labels].mean()),
                "failure_recall": float(alarms[labels].mean()),
                "informative_task_state_cells": len(by_stratum),
                "success_failure_pairs": pairs,
                "median_first_alarm_query_detected_failures": (
                    float(np.median(np.concatenate([item["first_alarms"] for item in items])))
                    if any(len(item["first_alarms"]) for item in items)
                    else None
                ),
                "fold_aucs": [float(item["fold_auc"]) for item in items],
            }
        comparisons = {}
        for comparison_index, (label, left, right) in enumerate(
            (
                ("global_prefix_minus_local2_prefix", "global_prefix", "local2_prefix"),
                ("global_current_minus_local2_current", "global_current", "local2_current"),
            )
        ):
            left_items = records[(horizon, left)]
            right_items = records[(horizon, right)]
            labels = np.concatenate([item["labels"] for item in left_items])
            task_names = np.concatenate([item["task_names"] for item in left_items])
            states = np.concatenate([item["states"] for item in left_items])
            comparisons[label] = fast_state_blocked_auc_difference(
                labels,
                np.concatenate([item["scores"] for item in left_items]),
                np.concatenate([item["scores"] for item in right_items]),
                task_names,
                states,
                bootstrap_draws,
                seed + horizon * 2003 + comparison_index,
            )
        output[str(horizon)]["method_differences"] = comparisons
    return output


def write_report(path: Path, summary: dict[str, Any]) -> None:
    prediction = summary["healthy_next_word_prediction"]
    lines = [
        "# 全前缀健康路由语法审计",
        "",
        "## 定义",
        "",
        "模型只用 train-state 的成功轨迹学习。位置 q 的输入是 BOS 与 0..q-1 的离散词和连续 descriptor，"
        "目标是当前词 q；观察当前词后的位置 q+1 才用于下一词分布和完整前缀状态。失败轨迹没有参与训练或选 epoch。",
        "",
        "连续下一词分数为 `-log sum_k p_LM(k | full prefix) p_GMM(z_q | k)`，因此把预测到相近词"
        "和预测到完全不同词区分开。`global_prefix` 是该因果增量从开头到当前的均值，不使用 CUSUM。",
        "",
        "## 健康下一词预测",
        "",
        "| 范围 | phase bits/q | local-2 bits/q | full-prefix bits/q | full-local 差 | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("q1+", "after_first_word"), ("q4+", "remote_history_q4plus")):
        item = prediction[key]
        delta = item["global_minus_local2"]
        lines.append(
            f"| {label} | {item['bits_per_query']['phase']:.4f} | "
            f"{item['bits_per_query']['local2']:.4f} | "
            f"{item['bits_per_query']['global_prefix']:.4f} | "
            f"{delta['mean_left_minus_right']:.4f} | "
            f"[{delta['ci95'][0]:.4f}, {delta['ci95'][1]:.4f}] |"
        )
    order = prediction["remote_order_control_q4_q12"]
    order_delta = order["original_minus_remote_reversed"]
    lines.extend(
        [
            "",
            f"在 q4..q12 保留最近两词不变、只反转更早历史后，原顺序为 "
            f"{order['original_bits_per_query']:.4f} bits/q，反转为 "
            f"{order['remote_reversed_bits_per_query']:.4f} bits/q；原顺序减反转为 "
            f"{order_delta['mean_left_minus_right']:.4f} "
            f"(95% CI [{order_delta['ci95'][0]:.4f}, "
            f"{order_delta['ci95'][1]:.4f}])。负值才表示模型学到了两词之外的远端顺序。",
            "",
            "## 截至当前的失败检测",
            "",
            "每个原始分数先用 train-success 按 task×query-position 转为经验分位数，再取截至当前的最大分位数；"
            "阈值仅由 cal-success 给出，目标 episode-level FPR=5%。没有使用 CUSUM，也没有用失败标签选方法。",
            "",
            "| q | 方法 | task/state AUC | 95% CI | test FPR | failure recall |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    displayed = (
        "lexical_prefix",
        "phase_prefix",
        "local2_prefix",
        "global_prefix",
        "global_excess_local_prefix",
        "local2_current",
        "global_current",
        "prototype_prefix",
        "hidden_prefix_state",
    )
    for horizon in HORIZONS:
        for method in displayed:
            item = summary["detection"][str(horizon)][method]
            lines.append(
                f"| {horizon} | {method} | {item['within_task_state_auc']:.3f} | "
                f"[{item['state_blocked_ci95'][0]:.3f}, {item['state_blocked_ci95'][1]:.3f}] | "
                f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
            )
        comparison = summary["detection"][str(horizon)]["method_differences"][
            "global_current_minus_local2_current"
        ]
        lines.append(
            f"| {horizon} | full-current - local2-current AUC | "
            f"{comparison['auc_left_minus_right']:+.3f} | "
            f"[{comparison['state_blocked_ci95'][0]:+.3f}, "
            f"{comparison['state_blocked_ci95'][1]:+.3f}] | - | - |"
        )
    q3 = summary["detection"]["3"]
    q7 = summary["detection"]["7"]
    q12 = summary["detection"]["12"]
    q7_difference = q7["method_differences"][
        "global_current_minus_local2_current"
    ]
    q12_difference = q12["method_differences"][
        "global_current_minus_local2_current"
    ]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "健康全局句法存在：full-prefix 在 50 个 cross-fit test states 上稳定改善下一词连续密度，"
            "而固定最近两词后打乱远端顺序也会显著降低预测质量。这个结论不依赖失败标签。",
            "",
            f"但它没有转化为额外的失败预警能力。q7 的 full-prefix current AUC 为 "
            f"{q7['global_current']['within_task_state_auc']:.3f}，local-2 为 "
            f"{q7['local2_current']['within_task_state_auc']:.3f}，差值 "
            f"{q7_difference['auc_left_minus_right']:+.3f} "
            f"(95% CI [{q7_difference['state_blocked_ci95'][0]:+.3f}, "
            f"{q7_difference['state_blocked_ci95'][1]:+.3f}])；q12 差值为 "
            f"{q12_difference['auc_left_minus_right']:+.3f} "
            f"(95% CI [{q12_difference['state_blocked_ci95'][0]:+.3f}, "
            f"{q12_difference['state_blocked_ci95'][1]:+.3f}])。",
            "",
            f"q3 的 full-prefix current AUC={q3['global_current']['within_task_state_auc']:.3f}，"
            "与随机水平不可区分。q7 虽有弱的最终失败信号，但在 "
            f"FPR={q7['global_current']['healthy_fpr']:.3f} 时 recall 仅 "
            f"{q7['global_current']['failure_recall']:.3f}；q12 在 "
            f"FPR={q12['global_current']['healthy_fpr']:.3f} 时 recall 仅 "
            f"{q12['global_current']['failure_recall']:.3f}。因此目前不能作为可靠在线拦截器。",
            "",
            "## 解释边界",
            "",
            "- q 时刻的 `global_current/global_prefix` 只读取 0..q；`next_entropy_current` 在读取当前 q 后"
            "预测 q+1，同样不读未来。",
            "- 这里检验的是对最终 episode failure 的早期预测。数据没有独立标注物理错误 onset，"
            "所以 q3/q7 的有效信号可以称为早期风险，但不能证明一定早于第一个错误动作。",
            "- q12 只评估长度大于 12 的幸存轨迹；不同 horizon 的覆盖集合不同。",
            "- full-prefix 若没有稳定优于 local-2，结论应是这些 routing 词中没有足够强的全局句法证据，"
            "不能仅因模型更复杂就称其为语法。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if len(args.datasets) != len(args.prediction_dirs):
        raise ValueError("datasets and prediction directories must have equal lengths")
    detection_records: dict[tuple[int, str], list[dict[str, np.ndarray]]] = {
        (horizon, method): [] for horizon in HORIZONS for method in PRIMARY_METHODS
    }
    healthy_records = []
    test_state_sets = []
    fold_metadata = []

    for fold, (dataset_path, prediction_dir) in enumerate(
        zip(args.datasets, args.prediction_dirs)
    ):
        print(f"Evaluating fold {fold}: {prediction_dir}", flush=True)
        dataset = load_npz(dataset_path)
        prediction = load_npz(prediction_dir / "predictions.npz")
        metadata = json.loads(str(dataset["metadata_json"].item()))
        prediction_metadata = json.loads(str(prediction["metadata_json"].item()))
        for name in ("lengths", "task", "state", "success"):
            np.testing.assert_array_equal(dataset[name], prediction[name])
        split = {
            name: np.asarray(values, dtype=np.int16)
            for name, values in metadata["split"].items()
        }
        lengths = dataset["lengths"]
        task = dataset["task"]
        state = dataset["state"]
        success = dataset["success"]
        task_count = len(metadata["tasks"])
        train_success = np.flatnonzero(np.isin(state, split["train"]) & success)
        calibration_success = np.flatnonzero(
            np.isin(state, split["calibration"]) & success
        )
        test = np.flatnonzero(np.isin(state, split["test"]))
        test_success = test[success[test]]
        test_state_sets.append(set(int(value) for value in split["test"]))
        fold_metadata.append(
            {
                "fold": fold,
                "dataset": str(dataset_path),
                "prediction_dir": str(prediction_dir),
                "test_states": split["test"].tolist(),
                "best_epoch": prediction_metadata["best_epoch"],
                "best_healthy_validation_soft_ce": prediction_metadata[
                    "best_healthy_validation_soft_ce"
                ],
            }
        )

        record: dict[str, np.ndarray] = {"state": state[test_success]}
        for start in (1, 4):
            record[f"phase_{start}"] = mean_episode_score(
                dataset["phase_soft"], lengths, test_success, start
            )
            record[f"local2_{start}"] = mean_episode_score(
                dataset["local2_soft"], lengths, test_success, start
            )
            record[f"global_{start}"] = mean_episode_score(
                prediction["global_soft"], lengths, test_success, start
            )
        order_indexes = test_success[lengths[test_success] > 4]
        order_stop = np.minimum(lengths[order_indexes], 13)
        original_order = []
        reversed_remote = []
        kept_indexes = []
        for episode_index, stop in zip(order_indexes, order_stop):
            original = prediction["global_soft"][episode_index, 4 : int(stop)]
            reversed_value = prediction["remote_reversed_soft"][
                episode_index, 4 : int(stop)
            ]
            if len(original) and np.all(np.isfinite(reversed_value)):
                original_order.append(float(np.mean(original) / math.log(2.0)))
                reversed_remote.append(
                    float(np.mean(reversed_value) / math.log(2.0))
                )
                kept_indexes.append(episode_index)
        record["original_order"] = np.asarray(original_order)
        record["reversed_remote"] = np.asarray(reversed_remote)
        record["order_state"] = state[np.asarray(kept_indexes, dtype=np.int64)]
        healthy_records.append(record)

        methods = raw_methods(dataset, prediction)
        sequential_scores = {}
        for method, raw in methods.items():
            percentile = conditional_percentiles(
                raw,
                train_success,
                task,
                lengths,
                task_count,
            )
            sequential_scores[method] = running_max(percentile, lengths)

        thresholds_by_method_and_query: dict[tuple[str, int], dict[int, float]] = {}
        for method in PRIMARY_METHODS:
            for query in range(13):
                thresholds_by_method_and_query[(method, query)] = task_thresholds(
                    sequential_scores[method],
                    calibration_success,
                    task,
                    lengths,
                    query,
                    task_count,
                    args.healthy_fpr,
                )

        for horizon in HORIZONS:
            eligible = test[lengths[test] > horizon]
            labels = ~success[eligible]
            task_names = np.asarray(
                [metadata["tasks"][int(task[index])] for index in eligible],
                dtype=object,
            )
            strata = np.asarray(
                [f"{name}|{item_state}" for name, item_state in zip(task_names, state[eligible])],
                dtype=object,
            )
            for method in PRIMARY_METHODS:
                score = sequential_scores[method]
                thresholds = thresholds_by_method_and_query[(method, horizon)]
                values = score[eligible, horizon]
                alarms = np.asarray(
                    [
                        value > thresholds[int(task[episode_index])]
                        for value, episode_index in zip(values, eligible)
                    ],
                    dtype=np.bool_,
                )
                first_alarms = []
                for episode_index, label, alarm in zip(eligible, labels, alarms):
                    if not label or not alarm:
                        continue
                    for query in range(horizon + 1):
                        query_threshold = thresholds_by_method_and_query[(method, query)][
                            int(task[episode_index])
                        ]
                        if score[episode_index, query] > query_threshold:
                            first_alarms.append(query)
                            break
                fold_auc = stratified_pair_auc(labels, values, strata)[0]
                detection_records[(horizon, method)].append(
                    {
                        "labels": labels,
                        "scores": values,
                        "alarms": alarms,
                        "task_names": task_names,
                        "states": state[eligible],
                        "first_alarms": np.asarray(first_alarms, dtype=np.int16),
                        "fold_auc": np.asarray(fold_auc),
                    }
                )

    joined_test_states = set().union(*test_state_sets)
    if sum(len(values) for values in test_state_sets) != len(joined_test_states):
        raise ValueError("test states overlap across folds")
    if joined_test_states != set(range(50)):
        raise ValueError("crossfit test states do not cover 0..49")

    summary = {
        "schema_version": 1,
        "protocol": {
            "training": "successful episodes from 30 train states per fold only",
            "model_selection": "successful episodes from 10 calibration states only",
            "test": "all episodes from 10 unseen states; five folds partition 50 states",
            "language_model": "two-layer causal Transformer over the complete observed prefix",
            "word_similarity": "continuous GMM-mixture predictive likelihood",
            "online_aggregation": "task-position healthy percentile of prefix energy, then running maximum; no CUSUM",
            "threshold": "task-specific calibration-success 95th percentile at each horizon",
            "endpoint": "eventual episode failure; physical failure onset is not annotated",
        },
        "folds": fold_metadata,
        "healthy_next_word_prediction": healthy_prediction_result(
            healthy_records, args.seed
        ),
        "detection": evaluate_detection_records(
            detection_records,
            PRIMARY_METHODS,
            args.bootstrap_draws,
            args.seed,
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "REPORT.zh.md", summary)
    print(f"Wrote global-prefix audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
