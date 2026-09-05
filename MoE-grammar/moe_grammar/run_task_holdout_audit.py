from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from moe_grammar.corpus import DEFAULT_STASIS_LABELS, Episode, corpus_summary, load_corpus
from moe_grammar.features import build_clean_query_descriptors
from moe_grammar.grammar import (
    CountGrammar,
    DurationGrammar,
    DurationModel,
    PositionContextGrammar,
    PositionGrammar,
)
from moe_grammar.run_experiments import (
    ExperimentConfig,
    episode_rows,
    episode_sequences,
    json_ready,
)
from moe_grammar.tokenizer import GMMTokenizer, Preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-task-out routing grammar audit.")
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-task-holdout"))
    parser.add_argument("--stasis-labels", type=Path, default=DEFAULT_STASIS_LABELS)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--pca-dim", type=int, default=ExperimentConfig.pca_dim)
    parser.add_argument("--word-count", type=int, default=32)
    parser.add_argument("--gmm-max-iter", type=int, default=ExperimentConfig.gmm_max_iter)
    parser.add_argument("--gmm-n-init", type=int, default=ExperimentConfig.gmm_n_init)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def fit_models(
    sequences: list[np.ndarray], n_words: int, config: ExperimentConfig
) -> dict[str, Any]:
    vocabulary = n_words + 1
    models: dict[str, Any] = {
        "unigram": CountGrammar(vocabulary, 0, alpha=config.alpha).fit(sequences),
        "position": PositionGrammar(
            vocabulary,
            alpha=config.alpha,
            min_support=config.min_context_support,
        ).fit(sequences),
        "position_context1": PositionContextGrammar(
            vocabulary,
            max_order=1,
            alpha=config.alpha,
            min_support=config.min_context_support,
            kl_delta=config.kl_delta,
        ).fit(sequences),
        "position_bag_context": PositionContextGrammar(
            vocabulary,
            max_order=config.position_context_order,
            alpha=config.alpha,
            min_support=config.min_context_support,
            kl_delta=config.kl_delta,
            context_mode="bag",
        ).fit(sequences),
        "position_context": PositionContextGrammar(
            vocabulary,
            max_order=config.position_context_order,
            alpha=config.alpha,
            min_support=config.min_context_support,
            kl_delta=config.kl_delta,
        ).fit(sequences),
        "bigram": CountGrammar(vocabulary, 1, alpha=config.alpha).fit(sequences),
        "markov4": CountGrammar(
            vocabulary,
            4,
            alpha=config.alpha,
            min_support=config.min_context_support,
        ).fit(sequences),
        "bag6": CountGrammar(
            vocabulary,
            config.context_order,
            alpha=config.alpha,
            min_support=config.min_context_support,
            context_mode="bag",
        ).fit(sequences),
        "pst6": CountGrammar(
            vocabulary,
            config.context_order,
            alpha=config.alpha,
            min_support=config.min_context_support,
            kl_delta=config.kl_delta,
        ).fit(sequences),
    }
    duration = DurationModel(
        n_words,
        alpha=config.alpha,
        min_support=config.duration_min_support,
    ).fit(sequences)
    models["pst6_duration"] = DurationGrammar(models["pst6"], duration)
    return models


def score_episode(
    episode: Episode,
    words: np.ndarray,
    component_log_likelihood: np.ndarray,
    models: dict[str, Any],
) -> dict[str, Any]:
    selected = slice(episode.start, episode.stop)
    sequence = words[selected]
    emission = component_log_likelihood[selected]
    record: dict[str, Any] = {
        "episode": episode.index,
        "task": episode.task,
        "task_index": episode.task_index,
        "state": episode.init_state_id,
        "noise_seed": episode.flow_noise_seed,
    }
    for name, model in models.items():
        hard, _ = model.hard_nll(sequence)
        continuous, _ = model.continuous_nll(sequence, emission)
        record[f"hard_bits_{name}"] = float(hard.mean() / math.log(2.0))
        record[f"continuous_nll_{name}"] = float(continuous.mean())
    return record


def task_bootstrap_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        rng.choice(values, size=(draws, len(values)), replace=True),
        axis=1,
    )
    return tuple(float(value) for value in np.quantile(bootstrap, [0.025, 0.975]))


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
    )
    comparisons = {
        "position_context1_vs_position": (
            "hard_bits_position_context1",
            "hard_bits_position",
        ),
        "position_context_vs_position": (
            "hard_bits_position_context",
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
        "pst_vs_bag": ("hard_bits_pst6", "hard_bits_bag6"),
        "pst_vs_markov4": ("hard_bits_pst6", "hard_bits_markov4"),
    }
    tasks = np.asarray([record["task"] for record in records], dtype=object)
    unique_tasks = np.unique(tasks)
    output: dict[str, Any] = {
        "episodes": len(records),
        "tasks": len(unique_tasks),
        "mean_hard_bits_per_token": {},
        "comparisons": {},
    }
    for name in model_names:
        values = np.asarray([record[f"hard_bits_{name}"] for record in records])
        task_means = np.asarray([np.mean(values[tasks == task]) for task in unique_tasks])
        output["mean_hard_bits_per_token"][name] = {
            "episode_weighted": float(np.mean(values)),
            "task_macro": float(np.mean(task_means)),
        }
    for index, (name, (left_name, right_name)) in enumerate(comparisons.items()):
        difference = np.asarray(
            [record[left_name] - record[right_name] for record in records],
            dtype=np.float64,
        )
        task_effects = {
            str(task): float(np.mean(difference[tasks == task])) for task in unique_tasks
        }
        task_values = np.asarray(list(task_effects.values()))
        output["comparisons"][name] = {
            "task_macro_left_minus_right": float(np.mean(task_values)),
            "episode_weighted_left_minus_right": float(np.mean(difference)),
            "task_bootstrap_ci95": list(
                task_bootstrap_interval(
                    task_values,
                    config.bootstrap_draws,
                    config.seed + index * 1009,
                )
            ),
            "tasks_negative": int(np.sum(task_values < 0.0)),
            "tasks_total": len(task_values),
            "per_held_out_task": task_effects,
        }
    return output


def write_report(summary: dict[str, Any], output: Path) -> None:
    result = summary["grammar_existence"]
    primary = result["comparisons"]["position_context_vs_position"]
    order = result["comparisons"]["position_context_vs_position_bag_context"]
    primary_ci = primary["task_bootstrap_ci95"]
    order_ci = order["task_bootstrap_ci95"]
    primary_supported = primary["task_macro_left_minus_right"] < 0.0 and primary_ci[1] < 0.0
    order_supported = order["task_macro_left_minus_right"] < 0.0 and order_ci[1] < 0.0
    if primary_supported:
        primary_label = "支持零样本跨任务历史迁移"
    elif primary["task_macro_left_minus_right"] < 0.0:
        primary_label = "方向一致但零样本跨任务证据不足"
    else:
        primary_label = "不支持零样本跨任务历史迁移"
    if order_supported:
        order_label = "顺序增量支持"
    elif order["task_macro_left_minus_right"] < 0.0:
        order_label = "顺序增量方向一致但证据不足"
    else:
        order_label = "顺序增量不支持"
    lines = [
        "# Leave-one-task-out 语法迁移审计",
        "",
        "## 结论",
        "",
        f"- **主检验：{primary_label}。** 在完全未见任务上，position+ordered-history 相对 "
        f"position 的 task-macro 差为 "
        f"`{primary['task_macro_left_minus_right']:+.4f}` bits/token，task-bootstrap 95% CI "
        f"`[{primary_ci[0]:+.4f}, {primary_ci[1]:+.4f}]`，"
        f"`{primary['tasks_negative']}/{primary['tasks_total']}` 个任务同方向。",
        f"- **{order_label}。** ordered-2 相对 bag-2 为 "
        f"`{order['task_macro_left_minus_right']:+.4f}` bits/token，"
        f"95% CI `[{order_ci[0]:+.4f}, {order_ci[1]:+.4f}]`，"
        f"`{order['tasks_negative']}/{order['tasks_total']}` 个任务同方向。",
        "- task+position 条件下的历史增益只在任务内成立；本审计不支持将它升级为零样本 "
        "task-invariant grammar。任务数只有 5，结论仍需新 benchmark family 验证。",
        "",
        "## 协议",
        "",
        "每折完整留出一个任务。RobustScaler、PCA、K=32 GMM tokenizer 和全部 grammar counts "
        "只使用其余四个任务的成功 episode；测试只使用被留出任务的成功 episode。失败标签未参与。",
        f"五折合计恰好覆盖 `{result['episodes']}` 条成功测试 episode。",
        "",
        "## 健康序列预测",
        "",
        "| 模型 | task-macro bits/token | episode-weighted |",
        "|---|---:|---:|",
    ]
    model_order = (
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
    )
    for name in model_order:
        values = result["mean_hard_bits_per_token"][name]
        lines.append(f"| {name} | {values['task_macro']:.4f} | {values['episode_weighted']:.4f} |")
    lines.extend(
        [
            "",
            "## 配对效应",
            "",
            "负数表示左侧模型 NLL 更低。区间以 5 个 held-out tasks 为重采样单位。",
            "",
            "| 对比 | task-macro 差值 | 95% CI | 同方向任务 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, item in result["comparisons"].items():
        ci = item["task_bootstrap_ci95"]
        lines.append(
            f"| {name} | {item['task_macro_left_minus_right']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{item['tasks_negative']}/{item['tasks_total']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "- 5 个任务不足以精确估计新任务分布，bootstrap 区间只表达当前任务集合的不确定性。",
            "- 同一 checkpoint、benchmark family 与数据生成流程仍可能产生共享结构。",
            "- 本实验验证预测迁移，不验证异常检测、控制收益或因果机制。",
            "- 结果应在新任务 family 或新 checkpoint 上确认后再使用 task-invariant 表述。",
            "",
            "完整逐任务效应和 split 诊断见 `summary.json`。",
        ]
    )
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    started = time.time()
    config = ExperimentConfig(
        seed=args.seed,
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
    records: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []

    for held_out_task in range(len(corpus.tasks)):
        train = [
            episode
            for episode in corpus.episodes
            if episode.success and episode.task_index != held_out_task
        ]
        test = [
            episode
            for episode in corpus.episodes
            if episode.success and episode.task_index == held_out_task
        ]
        print(
            f"Task {held_out_task + 1}/{len(corpus.tasks)}: {corpus.tasks[held_out_task]} "
            f"train={len(train)}, test={len(test)}",
            flush=True,
        )
        train_rows = episode_rows(train)
        preprocessor = Preprocessor(config.pca_dim, config.seed + held_out_task).fit(
            descriptor[train_rows]
        )
        projected = preprocessor.transform(descriptor)
        tokenizer = GMMTokenizer(
            args.word_count,
            seed=config.seed + held_out_task * 1009,
            max_iter=config.gmm_max_iter,
            n_init=config.gmm_n_init,
        ).fit(projected[train_rows])
        words, component, _ = tokenizer.transform(projected)
        models = fit_models(episode_sequences(train, words), args.word_count, config)
        records.extend(score_episode(episode, words, component, models) for episode in test)
        folds.append(
            {
                "held_out_task_index": held_out_task,
                "held_out_task": corpus.tasks[held_out_task],
                "train_success_episodes": len(train),
                "test_success_episodes": len(test),
                "pca_explained_variance": float(preprocessor.pca.explained_variance_ratio_.sum()),
                "tokenizer": tokenizer.diagnostics(projected[train_rows]),
            }
        )

    expected = {episode.index for episode in corpus.episodes if episode.success}
    observed = [record["episode"] for record in records]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise AssertionError("task holdout did not score every successful episode exactly once")
    summary = json_ready(
        {
            "schema_version": 1,
            "experiment": "leave-one-task-out-grammar-v1",
            "config": asdict(config),
            "protocol": {"folds": folds},
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
