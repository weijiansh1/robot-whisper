from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from moe_grammar.conditional_detection import (
    PhaseConditionalCDF,
    causal_aggregates,
)
from moe_grammar.corpus import Corpus, Episode, corpus_summary, load_corpus, make_state_folds
from moe_grammar.features import build_clean_query_descriptors
from moe_grammar.grammar import (
    CountGrammar,
    LagRecurrenceModel,
    PositionContextGrammar,
    PositionGrammar,
)
from moe_grammar.statistics import (
    auc_pairwise,
    cusum,
    empirical_percentile,
    paired_state_test,
    state_blocked_interval,
    stratified_pair_auc,
)
from moe_grammar.tokenizer import GMMTokenizer, Preprocessor


HORIZONS = (3, 7, 12)
RAW_METRICS = (
    "lexical",
    "phase",
    "bag",
    "history1",
    "history",
    "residual",
    # Audit channels the pooled context NLL could not express on its own.
    "order_residual",
    "unknown",
    "recurrence",
    "end_hazard",
)

# How many queries ahead a healthy sentence is allowed to still be running.
END_HORIZON = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features-full40"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-full40"))
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=24)
    parser.add_argument("--n-words", type=int, default=64)
    parser.add_argument("--rows-per-task", type=int, default=2000)
    parser.add_argument("--gmm-max-iter", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    return parser.parse_args()


def select_episodes(
    corpus: Corpus, states: np.ndarray, success: bool | None
) -> list[Episode]:
    selected_states = set(int(value) for value in states)
    return [
        episode
        for episode in corpus.episodes
        if episode.init_state_id in selected_states
        and (success is None or episode.success is success)
    ]


def episode_rows(episodes: list[Episode]) -> np.ndarray:
    return np.concatenate(
        [np.arange(episode.start, episode.stop, dtype=np.int64) for episode in episodes]
    )


def balanced_query_sample(
    episodes: list[Episode], rows_per_task: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    by_task: dict[int, list[Episode]] = {}
    for episode in episodes:
        by_task.setdefault(episode.task_index, []).append(episode)
    parts = []
    for task_index in sorted(by_task):
        rows = episode_rows(by_task[task_index])
        if len(rows) > rows_per_task:
            rows = np.sort(rng.choice(rows, size=rows_per_task, replace=False))
        parts.append(rows)
    return np.concatenate(parts)


def project_in_batches(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    preprocessor: Preprocessor,
    batch_size: int = 8192,
) -> np.ndarray:
    output = np.empty((len(features), preprocessor.pca.n_components_), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        descriptors = build_clean_query_descriptors(features[start:stop], feature_names)
        output[start:stop] = preprocessor.transform(descriptors)
        if stop % (batch_size * 10) == 0 or stop == len(features):
            print(f"  projected {stop:,}/{len(features):,} queries", flush=True)
    return output


def tokenize_in_batches(
    projected: np.ndarray, tokenizer: GMMTokenizer, batch_size: int = 8192
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    words = np.empty(len(projected), dtype=np.int16)
    emissions = np.empty((len(projected), tokenizer.n_words), dtype=np.float32)
    lexical = np.empty(len(projected), dtype=np.float32)
    for start in range(0, len(projected), batch_size):
        stop = min(start + batch_size, len(projected))
        batch_words, batch_emissions, batch_lexical = tokenizer.transform(
            projected[start:stop]
        )
        words[start:stop] = batch_words
        emissions[start:stop] = batch_emissions.astype(np.float32)
        lexical[start:stop] = batch_lexical.astype(np.float32)
    return words, emissions, lexical


def tokenizer_diagnostics(
    tokenizer: GMMTokenizer,
    words: np.ndarray,
    lexical: np.ndarray,
    rows: np.ndarray,
) -> dict[str, Any]:
    selected_words = words[rows]
    counts = np.bincount(selected_words, minlength=tokenizer.n_words)
    probability = counts / counts.sum()
    return {
        "converged": bool(tokenizer.model.converged_),
        "iterations": int(tokenizer.model.n_iter_),
        "lower_bound": float(tokenizer.model.lower_bound_),
        "train_mean_nll": float(-lexical[rows].mean()),
        "cluster_counts": counts.tolist(),
        "minimum_cluster_count": int(counts.min()),
        "effective_words": float(
            np.exp(-np.sum(probability * np.log(np.maximum(probability, 1e-12))))
        ),
    }


def episode_sequences(episodes: list[Episode], words: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(words[episode.start : episode.stop], dtype=np.int16) for episode in episodes]


def fit_task_grammars(
    corpus: Corpus,
    episodes: list[Episode],
    words: np.ndarray,
    n_words: int,
    min_support: int = 20,
) -> dict[int, dict[str, Any]]:
    vocabulary = n_words + 1
    output: dict[int, dict[str, Any]] = {}
    for task_index in range(len(corpus.tasks)):
        task_episodes = [episode for episode in episodes if episode.task_index == task_index]
        sequences = episode_sequences(task_episodes, words)
        if not sequences:
            raise ValueError(f"no healthy training episodes for task {corpus.tasks[task_index]}")
        output[task_index] = {
            "phase": PositionGrammar(
                vocabulary, alpha=0.5, min_support=min_support
            ).fit(sequences),
            "history1": PositionContextGrammar(
                vocabulary,
                max_order=1,
                alpha=0.5,
                min_support=min_support,
                kl_delta=0.01,
            ).fit(sequences),
            "bag": PositionContextGrammar(
                vocabulary,
                max_order=2,
                alpha=0.5,
                min_support=min_support,
                kl_delta=0.01,
                context_mode="bag",
            ).fit(sequences),
            "history": PositionContextGrammar(
                vocabulary,
                max_order=2,
                alpha=0.5,
                min_support=min_support,
                kl_delta=0.01,
            ).fit(sequences),
            "recurrence": LagRecurrenceModel(n_words=n_words, max_lag=4).fit(sequences),
            # A clock-free first-order model purely for the END reachability table.
            "end_table": CountGrammar(vocabulary, max_order=1, alpha=0.5)
            .fit(sequences)
            .end_hitting_table(END_HORIZON),
        }
    return output


def score_raw_metrics(
    corpus: Corpus,
    words: np.ndarray,
    emissions: np.ndarray,
    lexical_log_likelihood: np.ndarray,
    models: dict[int, dict[str, Any]],
    tokenizer: GMMTokenizer,
) -> dict[str, np.ndarray]:
    raw = {
        name: np.empty(len(corpus.features), dtype=np.float32) for name in RAW_METRICS
    }
    raw["lexical"][:] = -np.asarray(lexical_log_likelihood, dtype=np.float32)
    raw["unknown"][:] = tokenizer.unknown_posterior(lexical_log_likelihood).astype(np.float32)
    for index, episode in enumerate(corpus.episodes):
        selected = slice(episode.start, episode.stop)
        sequence = words[selected]
        emission = emissions[selected]
        task_models = models[episode.task_index]
        for name in ("phase", "bag", "history1", "history"):
            values, _ = task_models[name].continuous_nll(sequence, emission)
            raw[name][selected] = values.astype(np.float32)
        raw["residual"][selected] = raw["history"][selected] - raw["phase"][selected]
        unigram = task_models["phase"].unigram_continuous_nll(emission)
        raw["order_residual"][selected] = (
            raw["history"][selected] - unigram.astype(np.float32)
        )
        raw["recurrence"][selected] = (
            task_models["recurrence"].surprisal(sequence).astype(np.float32)
        )
        raw["end_hazard"][selected] = (
            1.0 - task_models["end_table"][np.asarray(sequence, dtype=np.int64)]
        ).astype(np.float32)
        if (index + 1) % 4000 == 0:
            print(f"  scored {index + 1:,}/{len(corpus.episodes):,} episodes", flush=True)
    return raw


def healthy_grammar_evaluation(
    episodes: list[Episode], raw: dict[str, np.ndarray], seed: int
) -> dict[str, Any]:
    values = {
        name: np.asarray(
            [raw[name][episode.start : episode.stop].mean() / math.log(2.0) for episode in episodes]
        )
        for name in ("phase", "history1", "bag", "history")
    }
    states = np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16)
    return {
        "episodes": len(episodes),
        "bits_per_query": {name: float(score.mean()) for name, score in values.items()},
        "history_minus_phase": paired_state_test(
            values["history"], values["phase"], states, seed
        ),
        "history1_minus_phase": paired_state_test(
            values["history1"], values["phase"], states, seed + 1
        ),
        "history_minus_history1": paired_state_test(
            values["history"], values["history1"], states, seed + 2
        ),
        "ordered_minus_bag": paired_state_test(
            values["history"], values["bag"], states, seed + 3
        ),
    }


def score_selected_healthy(
    episodes: list[Episode],
    words: np.ndarray,
    emissions: np.ndarray,
    models: dict[int, dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    values = {name: [] for name in ("phase", "history1", "bag", "history")}
    for episode in episodes:
        selected = slice(episode.start, episode.stop)
        for name in values:
            nll, _ = models[episode.task_index][name].continuous_nll(
                words[selected], emissions[selected]
            )
            values[name].append(float(nll.mean() / math.log(2.0)))
    arrays = {name: np.asarray(items) for name, items in values.items()}
    states = np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16)
    return {
        "bits_per_query": {name: float(item.mean()) for name, item in arrays.items()},
        "history_minus_phase": paired_state_test(
            arrays["history"], arrays["phase"], states, seed
        ),
        "ordered_minus_bag": paired_state_test(
            arrays["history"], arrays["bag"], states, seed + 1
        ),
    }


def detector_names() -> list[str]:
    names = ["old_pooled_cusum"]
    probe = causal_aggregates(np.asarray([0.5, 0.6]))
    for metric in RAW_METRICS:
        names.extend(f"{metric}.{aggregate}" for aggregate in probe)
    return names


def score_detectors(
    episode: Episode,
    raw: dict[str, np.ndarray],
    calibrator: PhaseConditionalCDF,
    pooled_history_reference: np.ndarray,
    names: list[str],
) -> np.ndarray:
    selected = slice(episode.start, episode.stop)
    positions = np.arange(episode.length, dtype=np.int16)
    columns: dict[str, np.ndarray] = {}
    pooled = empirical_percentile(pooled_history_reference, raw["history"][selected])
    columns["old_pooled_cusum"] = np.maximum.accumulate(cusum(pooled, 0.8))
    for metric in RAW_METRICS:
        percentile = calibrator.transform(
            metric, raw[metric][selected], episode.task_index, positions
        )
        for aggregate, values in causal_aggregates(percentile).items():
            columns[f"{metric}.{aggregate}"] = values
    return np.column_stack([columns[name] for name in names]).astype(np.float32)


def candidate_auc(
    episodes: list[Episode],
    scores: dict[int, np.ndarray],
    horizon: int,
    candidate_index: int,
) -> float:
    selected = [episode for episode in episodes if episode.length > horizon]
    labels = np.asarray([not episode.success for episode in selected], dtype=np.bool_)
    values = np.asarray([scores[episode.index][horizon, candidate_index] for episode in selected])
    strata = np.asarray(
        [f"{episode.task}|{episode.init_state_id}" for episode in selected], dtype=object
    )
    return stratified_pair_auc(labels, values, strata)[0]


def select_detector(
    episodes: list[Episode], scores: dict[int, np.ndarray], names: list[str]
) -> tuple[str, list[dict[str, Any]]]:
    ranking = []
    for index, name in enumerate(names):
        aucs = {str(horizon): candidate_auc(episodes, scores, horizon, index) for horizon in (7, 12)}
        finite = [value for value in aucs.values() if np.isfinite(value)]
        ranking.append(
            {
                "name": name,
                "calibration_auc": aucs,
                "mean_q7_q12_auc": float(np.mean(finite)) if finite else float("nan"),
            }
        )
    ranking.sort(key=lambda item: item["mean_q7_q12_auc"], reverse=True)
    return str(ranking[0]["name"]), ranking


def task_thresholds(
    episodes: list[Episode],
    scores: dict[int, np.ndarray],
    candidate_index: int,
    horizon: int,
    task_count: int,
    healthy_fpr: float,
) -> dict[int, float]:
    eligible = [
        episode for episode in episodes if episode.success and episode.length > horizon
    ]
    all_values = np.asarray(
        [scores[episode.index][horizon, candidate_index] for episode in eligible]
    )
    global_threshold = float(
        np.quantile(all_values, 1.0 - healthy_fpr, method="higher")
    )
    output = {}
    for task_index in range(task_count):
        values = np.asarray(
            [
                scores[episode.index][horizon, candidate_index]
                for episode in eligible
                if episode.task_index == task_index
            ]
        )
        output[task_index] = (
            float(np.quantile(values, 1.0 - healthy_fpr, method="higher"))
            if len(values) >= 20
            else global_threshold
        )
    return output


def evaluate_detector(
    corpus: Corpus,
    calibration_episodes: list[Episode],
    test_episodes: list[Episode],
    scores: dict[int, np.ndarray],
    candidate_index: int,
    horizon: int,
    healthy_fpr: float,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    thresholds = task_thresholds(
        calibration_episodes,
        scores,
        candidate_index,
        horizon,
        len(corpus.tasks),
        healthy_fpr,
    )
    episodes = [episode for episode in test_episodes if episode.length > horizon]
    labels = np.asarray([not episode.success for episode in episodes], dtype=np.bool_)
    values = np.asarray(
        [scores[episode.index][horizon, candidate_index] for episode in episodes]
    )
    alarms = np.asarray(
        [value > thresholds[episode.task_index] for value, episode in zip(values, episodes)]
    )
    task = np.asarray([episode.task for episode in episodes], dtype=object)
    state = np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16)
    strata = np.asarray(
        [f"{episode.task}|{episode.init_state_id}" for episode in episodes], dtype=object
    )
    within_auc, by_stratum, pairs = stratified_pair_auc(labels, values, strata)
    detected_failures = [
        (episode, thresholds[episode.task_index])
        for episode, alarm in zip(episodes, alarms)
        if not episode.success and alarm
    ]
    first_alarms = []
    for episode, threshold in detected_failures:
        crossing = np.flatnonzero(scores[episode.index][: horizon + 1, candidate_index] > threshold)
        if len(crossing):
            first_alarms.append(int(crossing[0]))
    by_suite = {}
    for suite in sorted({episode.task.split("/", 1)[0] for episode in episodes}):
        selected = np.asarray(
            [episode.task.startswith(f"{suite}/") for episode in episodes], dtype=np.bool_
        )
        suite_labels = labels[selected]
        suite_alarms = alarms[selected]
        by_suite[suite] = {
            "episodes": int(selected.sum()),
            "failures": int(suite_labels.sum()),
            "healthy_fpr": float(suite_alarms[~suite_labels].mean()),
            "failure_recall": (
                float(suite_alarms[suite_labels].mean()) if suite_labels.any() else None
            ),
            "pooled_auc": auc_pairwise(suite_labels, values[selected]),
        }
    return {
        "episodes": len(episodes),
        "successes": int((~labels).sum()),
        "failures": int(labels.sum()),
        "healthy_fpr": float(alarms[~labels].mean()),
        "failure_recall": float(alarms[labels].mean()),
        "pooled_auc": auc_pairwise(labels, values),
        "within_task_state_auc": within_auc,
        "state_blocked_ci95": list(
            state_blocked_interval(
                labels, values, task, state, bootstrap_draws, seed
            )
        ),
        "informative_task_state_cells": len(by_stratum),
        "success_failure_pairs": pairs,
        "median_first_alarm_query_detected_failures": (
            float(np.median(first_alarms)) if first_alarms else None
        ),
        "by_suite": by_suite,
        "thresholds_by_task": {
            corpus.tasks[task_index]: value for task_index, value in thresholds.items()
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    grammar = summary["healthy_grammar"]
    lines = [
        "# Full-40 健康语法与在线检测审计",
        "",
        "## 数据与隔离",
        "",
        f"使用 40 个 LIBERO 任务，共 {summary['corpus']['episodes']:,} 条轨迹、"
        f"{summary['corpus']['queries']:,} 个 query；成功 {summary['corpus']['successes']:,}，"
        f"失败 {summary['corpus']['failures']:,}。按 init-state 分成 30/10/10 个 train/cal/test state，"
        "同一 state 的 16 个 noise seed 不跨集合。",
        "",
        "Tokenizer、PCA 和语法只读取 train-success；失败轨迹不参与健康语法学习。"
        "检测阈值只读取 cal-success。带 `selected` 的检测器使用 cal-failure 选择方法，单独标注为监督调参。",
        "",
        "## 健康语法",
        "",
        "| 模型 | held-out success continuous mixture NLL / ln2 |",
        "|---|---:|",
    ]
    for name in ("phase", "history1", "bag", "history"):
        lines.append(f"| {name} | {grammar['bits_per_query'][name]:.4f} |")
    history = grammar["history_minus_phase"]
    ordered = grammar["ordered_minus_bag"]
    lines.extend(
        [
            "",
            f"二阶有序历史相对 task+position 的差为 {history['mean_left_minus_right']:.4f} bits/query "
            f"(state-blocked 95% CI {history['ci95']})；负值表示历史改善预测。",
            f"有序二阶相对同样上下文的 bag 差为 {ordered['mean_left_minus_right']:.4f} bits/query "
            f"(95% CI {ordered['ci95']})。",
            "",
            "## q7/q12 在线前缀检测",
            "",
            "所有统计量在 q 时只使用 0..q 的 routing。FPR/recall 使用按 task、按 horizon 从 cal-success "
            "得到的 5% episode-level 阈值。这里的阳性是最终失败，不等同于有物理 onset 标注的错误起点。",
            "",
        ]
    )
    for horizon in (7, 12):
        lines.extend(
            [
                f"### q{horizon}",
                "",
                "| 方法 | within task/state AUC | 95% CI | healthy FPR | failure recall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, result in summary["detection"][str(horizon)].items():
            lines.append(
                f"| {name} | {result['within_task_state_auc']:.3f} | "
                f"[{result['state_blocked_ci95'][0]:.3f}, {result['state_blocked_ci95'][1]:.3f}] | "
                f"{result['healthy_fpr']:.3f} | {result['failure_recall']:.3f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "- `old_pooled_cusum` 复现旧做法：跨 task/position 的 pooled percentile，再对 uniform score 使用 kappa=0.8。",
            "- 新 Page-CUSUM 先按 task+query-position 做经验 CDF，再映射到近似标准正态；kappa=0.5 "
            "因此有明确的零均值健康基线。GLR 同时扫描 1/2/4/8-query 窗口。",
            "- `calibration_selected` 是在 calibration failures 上从候选中选择，属于有监督上界；"
            "它不能被称为只由健康数据学习的异常检测器。",
            "- 这些数据只有最终 success 标签。它们能检验 q7/q12 是否预测最终失败，不能证明报警早于真实物理错误 onset。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if not 0 <= args.fold < 5:
        raise ValueError("--fold must be in [0, 4]")
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading full-40 corpus...", flush=True)
    corpus = load_corpus(
        args.features_dir,
        None,
        allow_task_segments=True,
        require_flow_shuffled=False,
        require_stasis_labels=False,
        feature_dtype=np.float16,
    )
    split = make_state_folds(
        np.asarray([episode.init_state_id for episode in corpus.episodes]), 5, args.seed
    )[args.fold]
    train_success = select_episodes(corpus, split["train"], True)
    calibration_all = select_episodes(corpus, split["calibration"], None)
    test_all = select_episodes(corpus, split["test"], None)
    test_success = [episode for episode in test_all if episode.success]
    print(
        f"Split: train-success={len(train_success):,}, cal={len(calibration_all):,}, "
        f"test={len(test_all):,}",
        flush=True,
    )

    tokenizer_rows = balanced_query_sample(train_success, args.rows_per_task, args.seed)
    print(f"Fitting train-only PCA on {len(tokenizer_rows):,} balanced queries...", flush=True)
    tokenizer_features = corpus.features[tokenizer_rows]
    tokenizer_descriptors = build_clean_query_descriptors(
        tokenizer_features, corpus.feature_names
    )
    del tokenizer_features
    preprocessor = Preprocessor(args.pca_dim, args.seed).fit(tokenizer_descriptors)
    del tokenizer_descriptors
    print("Projecting all queries in bounded-memory batches...", flush=True)
    projected = project_in_batches(corpus.features, corpus.feature_names, preprocessor)
    print(
        f"Fitting K={args.n_words} diagonal GMM on {len(tokenizer_rows):,} queries...",
        flush=True,
    )
    tokenizer = GMMTokenizer(
        args.n_words,
        seed=args.seed,
        max_iter=args.gmm_max_iter,
        n_init=1,
    ).fit(projected[tokenizer_rows])
    words, emissions, lexical = tokenize_in_batches(projected, tokenizer)
    print("Fitting task+position healthy grammars on all train-success episodes...", flush=True)
    task_models = fit_task_grammars(corpus, train_success, words, args.n_words)
    print("Scoring query-level healthy likelihoods...", flush=True)
    raw = score_raw_metrics(corpus, words, emissions, lexical, task_models, tokenizer)

    grammar_result = healthy_grammar_evaluation(test_success, raw, args.seed)
    state_order = np.asarray(split["train"], dtype=np.int16).copy()
    np.random.default_rng(args.seed + 17).shuffle(state_order)
    scaling = {}
    for state_count in (5, 10, 20, 30):
        subset = [
            episode
            for episode in train_success
            if episode.init_state_id in set(state_order[:state_count].tolist())
        ]
        print(
            f"Grammar learning curve: {state_count} states / {len(subset):,} successes...",
            flush=True,
        )
        models = (
            task_models
            if state_count == 30
            else fit_task_grammars(corpus, subset, words, args.n_words)
        )
        result = score_selected_healthy(
            test_success, words, emissions, models, args.seed + state_count
        )
        result["training_success_episodes"] = len(subset)
        scaling[str(state_count)] = result

    train_rows = episode_rows(train_success)
    print("Fitting task+position conditional CDFs from train-success...", flush=True)
    calibrator = PhaseConditionalCDF(min_support=32).fit(
        raw,
        corpus.query_task_index,
        corpus.query_episode_step,
        train_rows,
    )
    pooled_history = np.sort(raw["history"][train_rows].astype(np.float64))
    names = detector_names()
    detection_scores: dict[int, np.ndarray] = {}
    score_episodes = calibration_all + test_all
    for index, episode in enumerate(score_episodes):
        detection_scores[episode.index] = score_detectors(
            episode, raw, calibrator, pooled_history, names
        )
        if (index + 1) % 2000 == 0:
            print(f"  detector scores {index + 1:,}/{len(score_episodes):,}", flush=True)

    selected_name, ranking = select_detector(calibration_all, detection_scores, names)
    print(f"Calibration-selected detector: {selected_name}", flush=True)
    reported = [
        "old_pooled_cusum",
        "phase.page0.5",
        "history.point",
        "history.window4",
        "history.page0.5",
        "history.glr",
        "residual.page0.5",
    ]
    if selected_name not in reported:
        reported.append(selected_name)
    detection: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        detection[str(horizon)] = {}
        for method_index, name in enumerate(reported):
            label = "calibration_selected" if name == selected_name else name
            detection[str(horizon)][label] = evaluate_detector(
                corpus,
                calibration_all,
                test_all,
                detection_scores,
                names.index(name),
                horizon,
                args.healthy_fpr,
                args.bootstrap_draws,
                args.seed + horizon * 1009 + method_index,
            )

    summary = {
        "schema_version": 1,
        "corpus": corpus_summary(corpus),
        "split": {name: values.tolist() for name, values in split.items()},
        "protocol": {
            "tokenizer": f"balanced cap {args.rows_per_task}/task from train-success only",
            "pca_dim": args.pca_dim,
            "n_words": args.n_words,
            "grammar": "task + absolute query position + ordered history up to 2",
            "cdf": "train-success empirical CDF conditional on task and query position",
            "threshold": f"task-specific calibration-success {1.0 - args.healthy_fpr:.0%} quantile",
            "endpoint": "eventual episode failure; no physical failure-onset labels",
        },
        "counts": {
            "train_success_episodes": len(train_success),
            "calibration_episodes": len(calibration_all),
            "calibration_failures": sum(not episode.success for episode in calibration_all),
            "test_episodes": len(test_all),
            "test_failures": sum(not episode.success for episode in test_all),
            "tokenizer_queries": len(tokenizer_rows),
        },
        "tokenizer_diagnostics": tokenizer_diagnostics(
            tokenizer, words, lexical, tokenizer_rows
        ),
        "pca_retained_variance": float(preprocessor.pca.explained_variance_ratio_.sum()),
        "healthy_grammar": grammar_result,
        "grammar_learning_curve_fixed_tokenizer": scaling,
        "detector_selection": {
            "uses_calibration_failure_labels": True,
            "selected": selected_name,
            "ranking": ranking,
        },
        "detection": detection,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "REPORT.zh.md", summary)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "tokenizer": tokenizer,
            "task_models": task_models,
            "calibrator": calibrator,
            "pooled_history_reference": pooled_history,
            "tasks": corpus.tasks,
            "detector_names": names,
            "selected_detector": selected_name,
            "split": split,
        },
        args.output_dir / "full40_model.joblib",
        compress=3,
    )
    print(
        f"Wrote {args.output_dir} in {(time.time() - started) / 60.0:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    main()
