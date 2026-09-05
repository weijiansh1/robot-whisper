from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from moe_grammar.corpus import DEFAULT_STASIS_LABELS, Corpus, Episode, corpus_summary, load_corpus
from moe_grammar.features import build_clean_query_descriptors
from moe_grammar.run_experiments import (
    ExperimentConfig,
    episode_rows,
    fit_grammars,
    json_ready,
)
from moe_grammar.statistics import paired_state_test
from moe_grammar.tokenizer import GMMTokenizer, Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crossed init-state and flow-noise-seed holdout audit."
    )
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-dual-axis"))
    parser.add_argument("--stasis-labels", type=Path, default=DEFAULT_STASIS_LABELS)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--pca-dim", type=int, default=ExperimentConfig.pca_dim)
    parser.add_argument("--word-count", type=int, default=32)
    parser.add_argument("--gmm-max-iter", type=int, default=ExperimentConfig.gmm_max_iter)
    parser.add_argument("--gmm-n-init", type=int, default=ExperimentConfig.gmm_n_init)
    parser.add_argument("--bootstrap-draws", type=int, default=ExperimentConfig.bootstrap_draws)
    return parser.parse_args()


def axis_groups(values: np.ndarray, folds: int, seed: int) -> list[np.ndarray]:
    unique = np.unique(np.asarray(values, dtype=np.int64))
    if folds < 2 or folds > len(unique):
        raise ValueError("invalid number of axis folds")
    shuffled = unique.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    return [np.sort(group) for group in np.array_split(shuffled, folds)]


def crossed_cell_episodes(
    corpus: Corpus,
    test_states: np.ndarray,
    test_seeds: np.ndarray,
) -> tuple[list[Episode], list[Episode]]:
    state_set = set(int(value) for value in test_states)
    seed_set = set(int(value) for value in test_seeds)
    train = [
        episode
        for episode in corpus.episodes
        if episode.success
        and episode.init_state_id not in state_set
        and episode.flow_noise_seed not in seed_set
    ]
    test = [
        episode
        for episode in corpus.episodes
        if episode.success
        and episode.init_state_id in state_set
        and episode.flow_noise_seed in seed_set
    ]
    if {episode.init_state_id for episode in train} & state_set:
        raise AssertionError("state leakage in crossed cell")
    if {episode.flow_noise_seed for episode in train} & seed_set:
        raise AssertionError("noise-seed leakage in crossed cell")
    return train, test


def score_episode(
    episode: Episode,
    words: np.ndarray,
    component_log_likelihood: np.ndarray,
    grammars: Any,
    n_words: int,
    cell: str,
) -> dict[str, Any]:
    selected = slice(episode.start, episode.stop)
    sequence = words[selected]
    emission = component_log_likelihood[selected]
    record: dict[str, Any] = {
        "episode": episode.index,
        "cell": cell,
        "n_words": n_words,
        "task": episode.task,
        "state": episode.init_state_id,
        "noise_seed": episode.flow_noise_seed,
        "length": episode.length,
    }
    for name, model in grammars.models.items():
        hard, _ = model.hard_nll(sequence)
        continuous, _ = model.continuous_nll(sequence, emission)
        record[f"hard_bits_{name}"] = float(hard.mean() / math.log(2.0))
        record[f"continuous_nll_{name}"] = float(continuous.mean())
    for name, model in grammars.task_models[episode.task_index].items():
        hard, _ = model.hard_nll(sequence)
        continuous, _ = model.continuous_nll(sequence, emission)
        record[f"hard_bits_{name}"] = float(hard.mean() / math.log(2.0))
        record[f"continuous_nll_{name}"] = float(continuous.mean())
    return record


def crossed_cluster_interval(
    values: np.ndarray,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    first_unique, first_index = np.unique(first_axis, return_inverse=True)
    second_unique, second_index = np.unique(second_axis, return_inverse=True)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        first_weight = np.bincount(
            rng.integers(len(first_unique), size=len(first_unique)),
            minlength=len(first_unique),
        )
        second_weight = np.bincount(
            rng.integers(len(second_unique), size=len(second_unique)),
            minlength=len(second_unique),
        )
        weight = first_weight[first_index] * second_weight[second_index]
        estimates[draw] = np.average(values, weights=weight) if weight.sum() else np.nan
    finite = estimates[np.isfinite(estimates)]
    return tuple(float(value) for value in np.quantile(finite, [0.025, 0.975]))


def aggregate(records: list[dict[str, Any]], config: ExperimentConfig) -> dict[str, Any]:
    model_names = (
        "unigram",
        "position",
        "position_context1",
        "position_bag_context",
        "position_context",
        "bigram",
        "markov4",
        "bag6",
        "pst6",
        "pst6_duration",
        "task_unigram",
        "task_position",
        "task_position_context1",
        "task_position_bag_context",
        "task_position_context",
        "task_pst6",
    )
    comparisons = {
        "position_context_vs_position": (
            "hard_bits_position_context",
            "hard_bits_position",
        ),
        "position_context1_vs_position": (
            "hard_bits_position_context1",
            "hard_bits_position",
        ),
        "position_context_vs_position_context1": (
            "hard_bits_position_context",
            "hard_bits_position_context1",
        ),
        "position_context_vs_position_bag_context": (
            "hard_bits_position_context",
            "hard_bits_position_bag_context",
        ),
        "task_position_context_vs_task_position": (
            "hard_bits_task_position_context",
            "hard_bits_task_position",
        ),
        "task_position_context1_vs_task_position": (
            "hard_bits_task_position_context1",
            "hard_bits_task_position",
        ),
        "task_position_context_vs_task_position_context1": (
            "hard_bits_task_position_context",
            "hard_bits_task_position_context1",
        ),
        "task_position_context_vs_task_position_bag_context": (
            "hard_bits_task_position_context",
            "hard_bits_task_position_bag_context",
        ),
        "pst_vs_position": ("hard_bits_pst6", "hard_bits_position"),
        "pst_vs_bag": ("hard_bits_pst6", "hard_bits_bag6"),
        "pst_vs_markov4": ("hard_bits_pst6", "hard_bits_markov4"),
        "task_pst_vs_task_position": (
            "hard_bits_task_pst6",
            "hard_bits_task_position",
        ),
    }
    state = np.asarray([record["state"] for record in records], dtype=np.int64)
    noise_seed = np.asarray([record["noise_seed"] for record in records], dtype=np.int64)
    cells = np.asarray([record["cell"] for record in records], dtype=object)
    output: dict[str, Any] = {
        "episodes": len(records),
        "mean_hard_bits_per_token": {
            name: float(np.mean([record[f"hard_bits_{name}"] for record in records]))
            for name in model_names
        },
        "mean_continuous_nll": {
            name: float(np.mean([record[f"continuous_nll_{name}"] for record in records]))
            for name in model_names
        },
        "comparisons": {},
    }
    for index, (name, (left_name, right_name)) in enumerate(comparisons.items()):
        left = np.asarray([record[left_name] for record in records])
        right = np.asarray([record[right_name] for record in records])
        difference = left - right
        cell_effects = [float(np.mean(difference[cells == cell])) for cell in np.unique(cells)]
        output["comparisons"][name] = {
            "mean_left_minus_right": float(np.mean(difference)),
            "crossed_state_seed_bootstrap_ci95": list(
                crossed_cluster_interval(
                    difference,
                    state,
                    noise_seed,
                    config.bootstrap_draws,
                    config.seed + index * 1009,
                )
            ),
            "state_blocked_test": paired_state_test(
                left, right, state, seed=config.seed + index * 2
            ),
            "seed_blocked_test": paired_state_test(
                left, right, noise_seed, seed=config.seed + index * 2 + 1
            ),
            "crossed_cells_negative": int(np.sum(np.asarray(cell_effects) < 0.0)),
            "crossed_cells": len(cell_effects),
            "cell_effect_range": [float(np.min(cell_effects)), float(np.max(cell_effects))],
        }
    return output


def write_report(summary: dict[str, Any], output: Path) -> None:
    result = summary["grammar_existence"]
    primary = result["comparisons"]["task_position_context_vs_task_position"]
    history1 = result["comparisons"]["task_position_context1_vs_task_position"]
    history2 = result["comparisons"]["task_position_context_vs_task_position_context1"]
    ordered_bag = result["comparisons"]["task_position_context_vs_task_position_bag_context"]
    interval = primary["crossed_state_seed_bootstrap_ci95"]
    if interval[1] < 0.0:
        judgment = "支持。历史在 task 与绝对 query 阶段之外仍有稳定预测增益。"
    elif primary["mean_left_minus_right"] < 0.0:
        judgment = "方向一致但证据不足。双轴区间仍跨过零。"
    else:
        judgment = "不支持。加入历史未改善 task+position 基线。"
    lines = [
        "# State + noise-seed 双轴留出复验",
        "",
        "## 结论",
        "",
        f"- **主检验：{judgment}**",
        f"- task-position+history 相对 task-position 为 "
        f"`{primary['mean_left_minus_right']:+.4f}` bits/token，crossed bootstrap 95% CI "
        f"`[{interval[0]:+.4f}, {interval[1]:+.4f}]`。",
        f"- 16 个 crossed cells 中 `{primary['crossed_cells_negative']}/16` 个方向为改善；"
        f"cell effect 范围为 `[{primary['cell_effect_range'][0]:+.4f}, "
        f"{primary['cell_effect_range'][1]:+.4f}]`。",
        f"- 分解后：一阶历史 `{history1['mean_left_minus_right']:+.4f}` bits/token，"
        f"第二历史词增量 `{history2['mean_left_minus_right']:+.4f}`；ordered-2 相对 bag-2 为 "
        f"`{ordered_bag['mean_left_minus_right']:+.4f}`，crossed 95% CI "
        f"`[{ordered_bag['crossed_state_seed_bootstrap_ci95'][0]:+.4f}, "
        f"{ordered_bag['crossed_state_seed_bootstrap_ci95'][1]:+.4f}]`，"
        f"`{ordered_bag['crossed_cells_negative']}/16` cells 同方向。",
        "",
        "这是对共享 noise-seed ID 解释的泄漏审计，不是新数据集复现。词表 K=32、PCA=24 和语法"
        "超参数在审计前由主实验固定，因此本轮不重新调参。",
        "",
        "## 双轴协议",
        "",
        f"将 16 个 states 分成 `{summary['config']['folds']}` 组、32 个 noise seeds 分成 "
        f"`{summary['config']['folds']}` 组，取全部 16 个 state-group × seed-group 测试 cell。",
        "对每个 cell，训练只使用成功 episode，并排除所有共享测试 state **或**测试 seed 的样本。"
        "每条成功 episode 恰好作为测试样本一次。",
        "",
        f"测试覆盖 `{result['episodes']}` 条成功 episode；总语料为 "
        f"`{summary['corpus']['episodes']}` episodes / `{summary['corpus']['queries']}` queries。",
        "",
        "## 健康序列预测",
        "",
        "| 模型 | bits/token |",
        "|---|---:|",
    ]
    for name, value in result["mean_hard_bits_per_token"].items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.extend(
        [
            "",
            "## 配对效应",
            "",
            "负数表示左侧模型 NLL 更低。区间按 state 与 seed 两个聚类轴独立重采样。",
            "",
            "| 对比 | 差值 | crossed 95% CI | state p< | seed p< | 改善 cells |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in result["comparisons"].items():
        ci = item["crossed_state_seed_bootstrap_ci95"]
        lines.append(
            f"| {name} | {item['mean_left_minus_right']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{item['state_blocked_test']['one_sided_p_left_less']:.4g} | "
            f"{item['seed_blocked_test']['one_sided_p_left_less']:.4g} | "
            f"{item['crossed_cells_negative']}/{item['crossed_cells']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 双轴 bootstrap 用于稳健性区间；state-only 与 seed-only sign test 仅作敏感性检查。",
            "- task-conditioned 模型使用任务标签，是解释性 oracle control，不是严格 MoE-only 部署模型。",
            "- 复验仍来自同一批任务与 rollouts，不能替代新任务、新 checkpoint 或干预实验。",
            "- 本报告只复验健康序列预测，不利用失败标签，也不选择控制策略。",
            "",
            "完整 split、cell 和机器可读统计见 `summary.json`。",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    started = time.time()
    config = ExperimentConfig(
        seed=args.seed,
        folds=args.folds,
        pca_dim=args.pca_dim,
        word_counts=(args.word_count,),
        gmm_max_iter=args.gmm_max_iter,
        gmm_n_init=args.gmm_n_init,
        bootstrap_draws=args.bootstrap_draws,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading corpus and building ordered descriptors...", flush=True)
    corpus = load_corpus(args.features_dir, args.stasis_labels)
    descriptor = build_clean_query_descriptors(corpus.features, corpus.feature_names)
    states = np.asarray([episode.init_state_id for episode in corpus.episodes])
    seeds = np.asarray([episode.flow_noise_seed for episode in corpus.episodes])
    state_groups = axis_groups(states, config.folds, config.seed)
    seed_groups = axis_groups(seeds, config.folds, config.seed + 79_019)

    records: list[dict[str, Any]] = []
    cell_summaries: list[dict[str, Any]] = []
    for state_group_index, test_states in enumerate(state_groups):
        for seed_group_index, test_seeds in enumerate(seed_groups):
            cell = f"state_{state_group_index}|seed_{seed_group_index}"
            train_episodes, test_episodes = crossed_cell_episodes(corpus, test_states, test_seeds)
            print(
                f"Cell {len(cell_summaries) + 1}/16 {cell}: "
                f"train={len(train_episodes)}, test={len(test_episodes)}",
                flush=True,
            )
            train_rows = episode_rows(train_episodes)
            preprocessor = Preprocessor(
                config.pca_dim,
                config.seed + state_group_index * config.folds + seed_group_index,
            ).fit(descriptor[train_rows])
            projected = preprocessor.transform(descriptor)
            tokenizer = GMMTokenizer(
                n_words=args.word_count,
                seed=config.seed + state_group_index * 1009 + seed_group_index * 101,
                max_iter=config.gmm_max_iter,
                n_init=config.gmm_n_init,
            ).fit(projected[train_rows])
            words, component, _ = tokenizer.transform(projected)
            grammars = fit_grammars(
                corpus,
                train_episodes,
                words,
                args.word_count,
                config,
                state_group_index * config.folds + seed_group_index,
            )
            records.extend(
                score_episode(
                    episode,
                    words,
                    component,
                    grammars,
                    args.word_count,
                    cell,
                )
                for episode in test_episodes
            )
            cell_summaries.append(
                {
                    "cell": cell,
                    "test_states": test_states.tolist(),
                    "test_noise_seeds": test_seeds.tolist(),
                    "train_success_episodes": len(train_episodes),
                    "test_success_episodes": len(test_episodes),
                    "pca_explained_variance": float(
                        preprocessor.pca.explained_variance_ratio_.sum()
                    ),
                    "tokenizer": tokenizer.diagnostics(projected[train_rows]),
                }
            )

    expected = {episode.index for episode in corpus.episodes if episode.success}
    observed = [record["episode"] for record in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise AssertionError("crossed audit did not score every successful episode exactly once")

    summary = json_ready(
        {
            "schema_version": 1,
            "experiment": "crossed-state-noise-holdout-v1",
            "config": asdict(config),
            "protocol": {
                "state_groups": [group.tolist() for group in state_groups],
                "noise_seed_groups": [group.tolist() for group in seed_groups],
                "cells": cell_summaries,
                "train_rule": "success AND state not in test states AND seed not in test seeds",
                "test_rule": "success AND state in test states AND seed in test seeds",
            },
            "corpus": corpus_summary(corpus),
            "grammar_existence": aggregate(records, config),
            "runtime_seconds": time.time() - started,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(summary, args.output_dir / "REPORT.zh.md")
    print(
        f"Finished in {time.time() - started:.1f}s; report: {args.output_dir / 'REPORT.zh.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
