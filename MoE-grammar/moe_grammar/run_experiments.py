from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np

from moe_grammar.corpus import (
    DEFAULT_STASIS_LABELS,
    SCENE8_TASK,
    Corpus,
    Episode,
    corpus_summary,
    load_corpus,
    make_state_folds,
)
from moe_grammar.features import build_query_descriptors
from moe_grammar.grammar import (
    CountGrammar,
    DurationGrammar,
    DurationModel,
    PositionContextGrammar,
    PositionGrammar,
    shuffled_sequence,
)
from moe_grammar.statistics import (
    auc_pairwise,
    cusum,
    empirical_percentile,
    paired_state_test,
    safe_spearman,
    state_blocked_auc_difference,
    state_blocked_interval,
    stratified_pair_auc,
)
from moe_grammar.tokenizer import GMMTokenizer, Preprocessor


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260905
    folds: int = 4
    pca_dim: int = 24
    word_counts: tuple[int, ...] = (16, 32)
    context_order: int = 6
    position_context_order: int = 2
    min_context_support: int = 20
    duration_min_support: int = 10
    alpha: float = 0.5
    kl_delta: float = 0.01
    gmm_max_iter: int = 150
    gmm_n_init: int = 2
    query_shuffle_repeats: int = 4
    cusum_kappa: float = 0.8
    healthy_episode_fpr: float = 0.05
    bootstrap_draws: int = 1000
    horizons: tuple[int, ...] = (7, 12, 20, 27, 34)


@dataclass
class GrammarSet:
    models: dict[str, Any]
    task_models: dict[int, dict[str, Any]]
    shuffled_train_pst: CountGrammar
    duration_model: DurationModel


@dataclass
class TokenizedFold:
    n_words: int
    tokenizer: GMMTokenizer
    words: np.ndarray
    component_log_likelihood: np.ndarray
    lexical_log_likelihood: np.ndarray
    flow_words: np.ndarray
    flow_component_log_likelihood: np.ndarray
    flow_lexical_log_likelihood: np.ndarray
    grammars: GrammarSet
    calibration_nll: float
    diagnostics: dict[str, Any]


RAW_DETECTION_FIELDS = (
    "single",
    "bag",
    "ordered",
    "duration",
    "behavior",
    "phase",
    "phase_history",
    "phase_residual",
    "task_phase",
    "task_phase_history",
    "task_phase_residual",
)

DETECTION_METHODS = (
    "single",
    "bag",
    "ordered",
    "ordered_duration",
    "behavior",
    "phase",
    "phase_history",
    "phase_residual",
    "task_phase",
    "task_phase_history",
    "task_phase_residual",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--stasis-labels", type=Path, default=DEFAULT_STASIS_LABELS)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--folds", type=int, default=ExperimentConfig.folds)
    parser.add_argument("--pca-dim", type=int, default=ExperimentConfig.pca_dim)
    parser.add_argument("--word-counts", default="16,32")
    parser.add_argument("--context-order", type=int, default=ExperimentConfig.context_order)
    parser.add_argument(
        "--position-context-order", type=int, default=ExperimentConfig.position_context_order
    )
    parser.add_argument(
        "--min-context-support", type=int, default=ExperimentConfig.min_context_support
    )
    parser.add_argument("--gmm-max-iter", type=int, default=ExperimentConfig.gmm_max_iter)
    parser.add_argument("--gmm-n-init", type=int, default=ExperimentConfig.gmm_n_init)
    parser.add_argument(
        "--query-shuffle-repeats", type=int, default=ExperimentConfig.query_shuffle_repeats
    )
    parser.add_argument("--bootstrap-draws", type=int, default=ExperimentConfig.bootstrap_draws)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    word_counts = tuple(sorted({int(value) for value in args.word_counts.split(",")}))
    if not word_counts or word_counts[0] < 2:
        raise ValueError("--word-counts must contain vocabulary sizes >= 2")
    return ExperimentConfig(
        seed=args.seed,
        folds=args.folds,
        pca_dim=args.pca_dim,
        word_counts=word_counts,
        context_order=args.context_order,
        position_context_order=args.position_context_order,
        min_context_support=args.min_context_support,
        gmm_max_iter=args.gmm_max_iter,
        gmm_n_init=args.gmm_n_init,
        query_shuffle_repeats=args.query_shuffle_repeats,
        bootstrap_draws=args.bootstrap_draws,
    )


def selected_episodes(
    corpus: Corpus, states: np.ndarray, success: bool | None = None
) -> list[Episode]:
    state_set = set(int(value) for value in states)
    return [
        episode
        for episode in corpus.episodes
        if episode.init_state_id in state_set and (success is None or episode.success is success)
    ]


def episode_sequences(episodes: list[Episode], words: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(words[episode.start : episode.stop], dtype=np.int16) for episode in episodes]


def episode_rows(episodes: list[Episode]) -> np.ndarray:
    return np.concatenate(
        [np.arange(episode.start, episode.stop, dtype=np.int64) for episode in episodes]
    )


def fit_grammars(
    corpus: Corpus,
    train_episodes: list[Episode],
    words: np.ndarray,
    n_words: int,
    config: ExperimentConfig,
    fold: int,
) -> GrammarSet:
    vocabulary = n_words + 1
    sequences = episode_sequences(train_episodes, words)

    models: dict[str, Any] = {
        "unigram": CountGrammar(vocabulary, 0, alpha=config.alpha).fit(sequences),
        "position": PositionGrammar(
            vocabulary, alpha=config.alpha, min_support=config.min_context_support
        ).fit(sequences),
        "position_context": PositionContextGrammar(
            vocabulary,
            max_order=config.position_context_order,
            alpha=config.alpha,
            min_support=config.min_context_support,
            kl_delta=config.kl_delta,
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
        "bigram": CountGrammar(vocabulary, 1, alpha=config.alpha).fit(sequences),
        "markov4": CountGrammar(
            vocabulary, 4, alpha=config.alpha, min_support=config.min_context_support
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
        n_words=n_words,
        alpha=config.alpha,
        min_support=config.duration_min_support,
    ).fit(sequences)
    models["pst6_duration"] = DurationGrammar(models["pst6"], duration)

    task_models: dict[int, dict[str, Any]] = {}
    for task_index in range(len(corpus.tasks)):
        task_episodes = [episode for episode in train_episodes if episode.task_index == task_index]
        task_sequences = episode_sequences(task_episodes, words)
        task_models[task_index] = {
            "task_unigram": CountGrammar(vocabulary, 0, alpha=config.alpha).fit(task_sequences),
            "task_position": PositionGrammar(
                vocabulary,
                alpha=config.alpha,
                min_support=config.min_context_support,
            ).fit(task_sequences),
            "task_position_context": PositionContextGrammar(
                vocabulary,
                max_order=config.position_context_order,
                alpha=config.alpha,
                min_support=config.min_context_support,
                kl_delta=config.kl_delta,
            ).fit(task_sequences),
            "task_position_context1": PositionContextGrammar(
                vocabulary,
                max_order=1,
                alpha=config.alpha,
                min_support=config.min_context_support,
                kl_delta=config.kl_delta,
            ).fit(task_sequences),
            "task_position_bag_context": PositionContextGrammar(
                vocabulary,
                max_order=config.position_context_order,
                alpha=config.alpha,
                min_support=config.min_context_support,
                kl_delta=config.kl_delta,
                context_mode="bag",
            ).fit(task_sequences),
            "task_pst6": CountGrammar(
                vocabulary,
                config.context_order,
                alpha=config.alpha,
                min_support=config.min_context_support,
                kl_delta=config.kl_delta,
            ).fit(task_sequences),
        }

    shuffled = [
        shuffled_sequence(sequence, config.seed + fold * 1_000_003 + index)
        for index, sequence in enumerate(sequences)
    ]
    shuffled_train_pst = CountGrammar(
        vocabulary,
        config.context_order,
        alpha=config.alpha,
        min_support=config.min_context_support,
        kl_delta=config.kl_delta,
    ).fit(shuffled)
    return GrammarSet(models, task_models, shuffled_train_pst, duration)


def mean_continuous_nll(
    episodes: list[Episode], model: Any, words: np.ndarray, emissions: np.ndarray
) -> float:
    values = []
    for episode in episodes:
        selected = slice(episode.start, episode.stop)
        nll, _ = model.continuous_nll(words[selected], emissions[selected])
        values.append(nll)
    return float(np.concatenate(values).mean())


def fit_tokenizer_fold(
    corpus: Corpus,
    projected: np.ndarray,
    flow_projected: np.ndarray,
    train_episodes: list[Episode],
    calibration_episodes: list[Episode],
    n_words: int,
    config: ExperimentConfig,
    fold: int,
) -> TokenizedFold:
    train_rows = episode_rows(train_episodes)
    tokenizer = GMMTokenizer(
        n_words=n_words,
        seed=config.seed + fold * 101 + n_words,
        max_iter=config.gmm_max_iter,
        n_init=config.gmm_n_init,
    ).fit(projected[train_rows])
    words, component, lexical = tokenizer.transform(projected)
    flow_words, flow_component, flow_lexical = tokenizer.transform(flow_projected)
    grammars = fit_grammars(corpus, train_episodes, words, n_words, config, fold)
    calibration_nll = mean_continuous_nll(
        calibration_episodes, grammars.models["pst6"], words, component
    )
    diagnostics = tokenizer.diagnostics(projected[train_rows])
    diagnostics["calibration_pst_continuous_nll"] = calibration_nll
    return TokenizedFold(
        n_words=n_words,
        tokenizer=tokenizer,
        words=words,
        component_log_likelihood=component,
        lexical_log_likelihood=lexical,
        flow_words=flow_words,
        flow_component_log_likelihood=flow_component,
        flow_lexical_log_likelihood=flow_lexical,
        grammars=grammars,
        calibration_nll=calibration_nll,
        diagnostics=diagnostics,
    )


def score_existence_episode(
    episode: Episode,
    tokenized: TokenizedFold,
    config: ExperimentConfig,
    fold: int,
) -> dict[str, Any]:
    selected = slice(episode.start, episode.stop)
    sequence = tokenized.words[selected]
    emission = tokenized.component_log_likelihood[selected]
    record: dict[str, Any] = {
        "episode": episode.index,
        "fold": fold,
        "n_words": tokenized.n_words,
        "task": episode.task,
        "state": episode.init_state_id,
        "length": episode.length,
    }
    for name, model in tokenized.grammars.models.items():
        hard, depth = model.hard_nll(sequence)
        continuous, _ = model.continuous_nll(sequence, emission)
        record[f"hard_bits_{name}"] = float(hard.mean() / math.log(2.0))
        record[f"continuous_nll_{name}"] = float(continuous.mean())
        if name == "pst6":
            record["pst_mean_context_depth"] = float(depth.mean())

    task_models = tokenized.grammars.task_models[episode.task_index]
    for name, model in task_models.items():
        hard, _ = model.hard_nll(sequence)
        continuous, _ = model.continuous_nll(sequence, emission)
        record[f"hard_bits_{name}"] = float(hard.mean() / math.log(2.0))
        record[f"continuous_nll_{name}"] = float(continuous.mean())

    shuffled_hard = []
    shuffled_continuous = []
    shuffled_task_hard = []
    for repeat in range(config.query_shuffle_repeats):
        rng = np.random.default_rng(config.seed + fold * 10_000_019 + episode.index * 101 + repeat)
        permutation = rng.permutation(episode.length)
        shuffled_sequence_values = sequence[permutation]
        shuffled_emission = emission[permutation]
        hard, _ = tokenized.grammars.models["pst6"].hard_nll(shuffled_sequence_values)
        continuous, _ = tokenized.grammars.models["pst6"].continuous_nll(
            shuffled_sequence_values, shuffled_emission
        )
        task_hard, _ = task_models["task_pst6"].hard_nll(shuffled_sequence_values)
        shuffled_hard.append(hard.mean() / math.log(2.0))
        shuffled_continuous.append(continuous.mean())
        shuffled_task_hard.append(task_hard.mean() / math.log(2.0))
    record["hard_bits_pst6_query_shuffled"] = float(np.mean(shuffled_hard))
    record["continuous_nll_pst6_query_shuffled"] = float(np.mean(shuffled_continuous))
    record["hard_bits_task_pst6_query_shuffled"] = float(np.mean(shuffled_task_hard))

    train_shuffle_hard, _ = tokenized.grammars.shuffled_train_pst.hard_nll(sequence)
    record["hard_bits_pst6_train_shuffled"] = float(train_shuffle_hard.mean() / math.log(2.0))

    flow_sequence = tokenized.flow_words[selected]
    flow_emission = tokenized.flow_component_log_likelihood[selected]
    flow_hard, _ = tokenized.grammars.models["pst6"].hard_nll(flow_sequence)
    flow_continuous, _ = tokenized.grammars.models["pst6"].continuous_nll(
        flow_sequence, flow_emission
    )
    record["hard_bits_pst6_flow_shuffled"] = float(flow_hard.mean() / math.log(2.0))
    record["continuous_nll_pst6_flow_shuffled"] = float(flow_continuous.mean())
    record["lexical_nll_ordered"] = float(-tokenized.lexical_log_likelihood[selected].mean())
    record["lexical_nll_flow_shuffled"] = float(
        -tokenized.flow_lexical_log_likelihood[selected].mean()
    )
    record["flow_word_change_fraction"] = float(np.mean(sequence != flow_sequence))
    return record


def fit_behavior_reference(
    corpus: Corpus, train_episodes: list[Episode]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = list(corpus.behavior_feature_names)
    columns = [
        names.index("action_arm_token_std"),
        names.index("action_mean_delta"),
        names.index("proprio_delta"),
    ]
    rows = episode_rows(train_episodes)
    train = corpus.behavior_features[rows][:, columns]
    median = np.median(train, axis=0)
    scale = np.quantile(train, 0.75, axis=0) - np.quantile(train, 0.25, axis=0)
    scale = np.maximum(scale, 1e-6)
    stall = -np.mean((corpus.behavior_features[:, columns] - median) / scale, axis=1)
    return stall, median, scale


def raw_episode_metrics(
    episode: Episode,
    tokenized: TokenizedFold,
    behavior_stall: np.ndarray,
) -> dict[str, np.ndarray]:
    selected = slice(episode.start, episode.stop)
    sequence = tokenized.words[selected]
    emission = tokenized.component_log_likelihood[selected]
    bag, _ = tokenized.grammars.models["bag6"].continuous_nll(sequence, emission)
    ordered, _ = tokenized.grammars.models["pst6"].continuous_nll(sequence, emission)
    phase, _ = tokenized.grammars.models["position"].continuous_nll(sequence, emission)
    phase_history, _ = tokenized.grammars.models["position_context"].continuous_nll(
        sequence, emission
    )
    task_models = tokenized.grammars.task_models[episode.task_index]
    task_phase, _ = task_models["task_position"].continuous_nll(sequence, emission)
    task_phase_history, _ = task_models["task_position_context"].continuous_nll(sequence, emission)
    duration = tokenized.grammars.models["pst6_duration"].duration_nll(sequence)
    return {
        "single": -tokenized.lexical_log_likelihood[selected],
        "bag": bag,
        "ordered": ordered,
        "duration": duration,
        "behavior": behavior_stall[selected],
        "phase": phase,
        "phase_history": phase_history,
        "phase_residual": phase_history - phase,
        "task_phase": task_phase,
        "task_phase_history": task_phase_history,
        "task_phase_residual": task_phase_history - task_phase,
    }


def calibrate_detection(
    calibration_episodes: list[Episode],
    tokenized: TokenizedFold,
    behavior_stall: np.ndarray,
    config: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    raw_by_episode = [
        raw_episode_metrics(episode, tokenized, behavior_stall) for episode in calibration_episodes
    ]
    references = {
        name: np.concatenate([values[name] for values in raw_by_episode])
        for name in RAW_DETECTION_FIELDS
    }
    maxima: dict[str, list[float]] = {name: [] for name in DETECTION_METHODS}
    for raw in raw_by_episode:
        point = {
            name: empirical_percentile(references[name], raw[name]) for name in RAW_DETECTION_FIELDS
        }
        point["ordered_duration"] = np.maximum(point["ordered"], point["duration"])
        for name in maxima:
            maxima[name].append(float(cusum(point[name], config.cusum_kappa).max()))
    quantile = 1.0 - config.healthy_episode_fpr
    thresholds = {
        name: float(np.quantile(values, quantile, method="higher"))
        for name, values in maxima.items()
    }
    return references, thresholds


def score_detection_episode(
    episode: Episode,
    tokenized: TokenizedFold,
    behavior_stall: np.ndarray,
    references: dict[str, np.ndarray],
    thresholds: dict[str, float],
    config: ExperimentConfig,
    fold: int,
) -> dict[str, Any]:
    raw = raw_episode_metrics(episode, tokenized, behavior_stall)
    point = {
        name: empirical_percentile(references[name], raw[name]) for name in RAW_DETECTION_FIELDS
    }
    point["ordered_duration"] = np.maximum(point["ordered"], point["duration"])
    return {
        "episode": episode.index,
        "fold": fold,
        "selected_words": tokenized.n_words,
        "point": {name: point[name] for name in DETECTION_METHODS},
        "cusum": {name: cusum(point[name], config.cusum_kappa) for name in DETECTION_METHODS},
        "threshold": {name: thresholds[name] for name in DETECTION_METHODS},
    }


def aggregate_existence(records: list[dict[str, Any]], config: ExperimentConfig) -> dict[str, Any]:
    states = np.asarray([record["state"] for record in records], dtype=np.int16)

    def mean(field: str) -> float:
        return float(np.mean([record[field] for record in records]))

    hard_fields = (
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
    output: dict[str, Any] = {
        "episodes": len(records),
        "mean_hard_bits_per_token": {name: mean(f"hard_bits_{name}") for name in hard_fields},
        "mean_continuous_nll": {name: mean(f"continuous_nll_{name}") for name in hard_fields},
        "mean_pst_context_depth": mean("pst_mean_context_depth"),
        "flow_word_change_fraction": mean("flow_word_change_fraction"),
    }

    comparisons = {
        "pst_vs_unigram": ("hard_bits_pst6", "hard_bits_unigram"),
        "pst_vs_position": ("hard_bits_pst6", "hard_bits_position"),
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
        "pst_vs_bigram": ("hard_bits_pst6", "hard_bits_bigram"),
        "pst_vs_markov4": ("hard_bits_pst6", "hard_bits_markov4"),
        "pst_vs_bag": ("hard_bits_pst6", "hard_bits_bag6"),
        "duration_vs_pst": ("hard_bits_pst6_duration", "hard_bits_pst6"),
        "task_pst_vs_task_unigram": ("hard_bits_task_pst6", "hard_bits_task_unigram"),
        "task_pst_vs_task_position": (
            "hard_bits_task_pst6",
            "hard_bits_task_position",
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
        "ordered_vs_query_shuffle": (
            "hard_bits_pst6",
            "hard_bits_pst6_query_shuffled",
        ),
        "task_ordered_vs_query_shuffle": (
            "hard_bits_task_pst6",
            "hard_bits_task_pst6_query_shuffled",
        ),
        "ordered_vs_train_shuffle": (
            "hard_bits_pst6",
            "hard_bits_pst6_train_shuffled",
        ),
        "ordered_vs_flow_shuffle": (
            "hard_bits_pst6",
            "hard_bits_pst6_flow_shuffled",
        ),
        "lexical_vs_flow_shuffle": (
            "lexical_nll_ordered",
            "lexical_nll_flow_shuffled",
        ),
    }
    output["paired_state_tests"] = {
        name: paired_state_test(
            np.asarray([record[left] for record in records]),
            np.asarray([record[right] for record in records]),
            states,
            seed=config.seed + index,
        )
        for index, (name, (left, right)) in enumerate(comparisons.items())
    }
    output["controls"] = {
        "pst_query_shuffled_hard_bits": mean("hard_bits_pst6_query_shuffled"),
        "pst_train_shuffled_hard_bits": mean("hard_bits_pst6_train_shuffled"),
        "pst_flow_shuffled_hard_bits": mean("hard_bits_pst6_flow_shuffled"),
        "ordered_lexical_nll": mean("lexical_nll_ordered"),
        "flow_shuffled_lexical_nll": mean("lexical_nll_flow_shuffled"),
    }
    return output


def evaluate_failure_horizons(
    corpus: Corpus,
    detection: dict[int, dict[str, Any]],
    config: ExperimentConfig,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for horizon in config.horizons:
        episodes = [episode for episode in corpus.episodes if episode.length > horizon]
        labels = np.asarray([not episode.success for episode in episodes], dtype=np.bool_)
        task = np.asarray([episode.task for episode in episodes], dtype=object)
        state = np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16)
        strata = np.asarray(
            [f"{episode.task}|state_{episode.init_state_id}" for episode in episodes],
            dtype=object,
        )
        item: dict[str, Any] = {
            "episodes": len(episodes),
            "successes": int(np.sum(~labels)),
            "failures": int(np.sum(labels)),
            "methods": {},
        }
        score_by_method: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(DETECTION_METHODS):
            scores = np.asarray(
                [detection[episode.index]["cusum"][method][horizon] for episode in episodes]
            )
            score_by_method[method] = scores
            stratified, by_stratum, pairs = stratified_pair_auc(labels, scores, strata)
            interval = state_blocked_interval(
                labels,
                scores,
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 31 + method_index,
            )
            item["methods"][method] = {
                "within_task_state_auc": stratified,
                "state_blocked_ci95": list(interval),
                "success_failure_pairs": pairs,
                "pooled_auc": auc_pairwise(labels, scores),
                "informative_strata": len(by_stratum),
            }
        item["auc_improvements"] = {
            "ordered_minus_single": state_blocked_auc_difference(
                labels,
                score_by_method["ordered"],
                score_by_method["single"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 1,
            ),
            "ordered_minus_bag": state_blocked_auc_difference(
                labels,
                score_by_method["ordered"],
                score_by_method["bag"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 2,
            ),
            "duration_minus_ordered": state_blocked_auc_difference(
                labels,
                score_by_method["ordered_duration"],
                score_by_method["ordered"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 3,
            ),
            "phase_history_minus_phase": state_blocked_auc_difference(
                labels,
                score_by_method["phase_history"],
                score_by_method["phase"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 4,
            ),
            "task_phase_history_minus_task_phase": state_blocked_auc_difference(
                labels,
                score_by_method["task_phase_history"],
                score_by_method["task_phase"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 5,
            ),
            "task_phase_residual_minus_task_phase": state_blocked_auc_difference(
                labels,
                score_by_method["task_phase_residual"],
                score_by_method["task_phase"],
                task,
                state,
                config.bootstrap_draws,
                config.seed + horizon * 101 + 6,
            ),
        }
        output[str(horizon)] = item
    return output


def evaluate_false_alarms(corpus: Corpus, detection: dict[int, dict[str, Any]]) -> dict[str, Any]:
    success = [episode for episode in corpus.episodes if episode.success]
    output: dict[str, Any] = {"healthy_test_episodes": len(success), "methods": {}}
    for method in DETECTION_METHODS:
        alarms = []
        for episode in success:
            record = detection[episode.index]
            alarms.append(bool(np.any(record["cusum"][method] > record["threshold"][method])))
        output["methods"][method] = {
            "episodes_with_any_alarm": int(np.sum(alarms)),
            "episode_false_alarm_rate": float(np.mean(alarms)),
        }
    return output


def evaluate_stasis(corpus: Corpus, detection: dict[int, dict[str, Any]]) -> dict[str, Any]:
    stasis = [episode for episode in corpus.episodes if episode.stasis_onset >= 0]
    healthy = [
        episode for episode in corpus.episodes if episode.task == SCENE8_TASK and episode.success
    ]
    output: dict[str, Any] = {
        "definition": "scene8 physical no-further-progress onset from held-out successful terminal poses",
        "stasis_failures": len(stasis),
        "scene8_successes": len(healthy),
        "methods": {},
    }
    for method in DETECTION_METHODS:
        alarms: list[int | None] = []
        leads: list[int] = []
        by_onset = 0
        by_onset_plus_3 = 0
        for episode in stasis:
            record = detection[episode.index]
            candidates = np.flatnonzero(record["cusum"][method] > record["threshold"][method])
            alarm = int(candidates[0]) if len(candidates) else None
            alarms.append(alarm)
            if alarm is not None:
                leads.append(episode.stasis_onset - alarm)
                by_onset += int(alarm <= episode.stasis_onset)
                by_onset_plus_3 += int(alarm <= episode.stasis_onset + 3)
        healthy_alarm = []
        for episode in healthy:
            record = detection[episode.index]
            healthy_alarm.append(np.any(record["cusum"][method] > record["threshold"][method]))
        output["methods"][method] = {
            "detected_any": int(sum(alarm is not None for alarm in alarms)),
            "recall_any": float(np.mean([alarm is not None for alarm in alarms])),
            "recall_by_physical_onset": by_onset / len(stasis),
            "recall_by_onset_plus_3": by_onset_plus_3 / len(stasis),
            "lead_queries_median_detected": float(np.median(leads)) if leads else None,
            "lead_queries_q25_q75": (
                [float(value) for value in np.quantile(leads, [0.25, 0.75])] if leads else None
            ),
            "scene8_success_false_alarm_rate": float(np.mean(healthy_alarm)),
        }
    return output


def phenotype_analysis(corpus: Corpus, config: ExperimentConfig) -> dict[str, Any]:
    names = list(corpus.feature_names)
    metric_columns = {
        metric: [index for index, name in enumerate(names) if name.startswith(f"{metric}|")]
        for metric in (
            "entropy",
            "top4_token_consensus",
            "effective_rank",
            "flow_velocity",
            "flow_top4_switch",
        )
    }
    entropy = corpus.features[:, :, metric_columns["entropy"]].mean(axis=2)
    flow_flatten = entropy[:, 7:10].mean(axis=1) - entropy[:, :3].mean(axis=1)
    sync_l15_f9 = corpus.features[:, 9, names.index("top4_token_consensus|layer_7")]
    lowrank_l15_f9 = -corpus.features[:, 9, names.index("effective_rank|layer_7")]
    late_velocity = corpus.features[:, 9, names.index("flow_velocity|layer_7")]

    query_flatten = np.zeros(len(corpus.features), dtype=np.float32)
    for episode in corpus.episodes:
        values = entropy[episode.start : episode.stop].mean(axis=1)
        for step in range(episode.length):
            start = max(0, step - 6)
            window = values[start : step + 1]
            if len(window) >= 2:
                coordinate = np.linspace(-1.0, 1.0, len(window))
                query_flatten[episode.start + step] = float(
                    np.sum((window - window.mean()) * coordinate) / np.sum(coordinate**2)
                )

    metrics = {
        "within_query_flow_flattening": flow_flatten,
        "cross_query_entropy_slope_w7": query_flatten,
        "terminal_l15_f9_top4_sync": sync_l15_f9,
        "terminal_l15_f9_lowrank": lowrank_l15_f9,
        "terminal_l15_f9_velocity": late_velocity,
    }
    output: dict[str, Any] = {"scene8_fixed_horizon_stasis_vs_success": {}}
    for horizon in (12, 20, 34):
        episodes = [
            episode
            for episode in corpus.episodes
            if episode.task == SCENE8_TASK
            and episode.length > horizon
            and (episode.success or episode.stasis_onset >= 0)
        ]
        labels = np.asarray([not episode.success for episode in episodes], dtype=np.bool_)
        strata = np.asarray([episode.init_state_id for episode in episodes], dtype=np.int16)
        task = np.asarray([episode.task for episode in episodes], dtype=object)
        horizon_result: dict[str, Any] = {}
        for name, values in metrics.items():
            score = np.asarray([values[episode.start + horizon] for episode in episodes])
            auc, by_state, pairs = stratified_pair_auc(labels, score, strata)
            horizon_result[name] = {
                "within_state_auc": auc,
                "state_blocked_ci95": list(
                    state_blocked_interval(
                        labels,
                        score,
                        task,
                        strata,
                        config.bootstrap_draws,
                        config.seed + horizon * 313 + len(horizon_result),
                    )
                ),
                "pairs": pairs,
                "informative_states": len(by_state),
            }
        output["scene8_fixed_horizon_stasis_vs_success"][str(horizon)] = horizon_result

    scene8_rows = np.concatenate(
        [
            np.arange(episode.start, episode.stop)
            for episode in corpus.episodes
            if episode.task == SCENE8_TASK
        ]
    )
    output["axis_relation"] = {
        "spearman_all_scene8_queries_cross_query_flattening_vs_terminal_sync": safe_spearman(
            query_flatten[scene8_rows], sync_l15_f9[scene8_rows]
        ),
        "interpretation": "Correlation is descriptive only; the two metrics live on different axes.",
    }
    horizon_34_rows = np.asarray(
        [
            episode.start + 34
            for episode in corpus.episodes
            if episode.task == SCENE8_TASK and episode.length > 34
        ]
    )
    output["axis_relation"]["spearman_at_q34_cross_query_flattening_vs_terminal_sync"] = (
        safe_spearman(query_flatten[horizon_34_rows], sync_l15_f9[horizon_34_rows])
    )
    return output


def word_prototypes(
    corpus: Corpus,
    train_episodes: list[Episode],
    words: np.ndarray,
    n_words: int,
) -> list[dict[str, Any]]:
    rows = episode_rows(train_episodes)
    features = corpus.features[rows]
    assigned = words[rows]
    names = list(corpus.feature_names)

    def metric(values: np.ndarray, name: str) -> np.ndarray:
        columns = [index for index, item in enumerate(names) if item.startswith(f"{name}|")]
        return values[:, :, columns].mean(axis=2)

    summaries = {
        "entropy_delta": metric(features, "entropy")[:, 7:].mean(axis=1)
        - metric(features, "entropy")[:, :3].mean(axis=1),
        "sync_delta": metric(features, "top4_token_consensus")[:, 7:].mean(axis=1)
        - metric(features, "top4_token_consensus")[:, :3].mean(axis=1),
        "late_effective_rank": metric(features, "effective_rank")[:, 7:].mean(axis=1),
        "mean_switch": metric(features, "flow_top4_switch")[:, 1:].mean(axis=1),
        "late_layer_disagreement": features[:, 7:, names.index("layer_disagreement")].mean(axis=1),
        "terminal_l15_sync": features[:, 9, names.index("top4_token_consensus|layer_7")],
        "terminal_l15_effective_rank": features[:, 9, names.index("effective_rank|layer_7")],
    }
    center = {name: float(np.mean(values)) for name, values in summaries.items()}
    scale = {name: float(np.std(values) + 1e-12) for name, values in summaries.items()}
    output = []
    for word in range(n_words):
        selected = assigned == word
        item: dict[str, Any] = {"word": word, "support": int(np.sum(selected))}
        for name, values in summaries.items():
            item[name] = float(np.mean(values[selected]))
            item[f"z_{name}"] = (item[name] - center[name]) / scale[name]
        tags = []
        if item["z_entropy_delta"] > 0.5:
            tags.append("RelativeFlattening")
        elif item["z_entropy_delta"] < -0.5:
            tags.append("Sharpening")
        if item["z_sync_delta"] > 0.5:
            tags.append("Synchronizing")
        if item["z_mean_switch"] > 0.75:
            tags.append("Switching")
        if item["z_late_effective_rank"] < -0.5:
            tags.append("LowRank")
        if item["z_late_layer_disagreement"] < -0.5:
            tags.append("LayerLock")
        item["heuristic_tags"] = tags or ["Mixed"]
        output.append(item)
    return output


def behavior_association(corpus: Corpus, detection: dict[int, dict[str, Any]]) -> dict[str, float]:
    ordered = []
    behavior = []
    for episode in corpus.episodes:
        record = detection[episode.index]
        ordered.append(record["point"]["ordered_duration"])
        behavior.append(record["point"]["behavior"])
    return {
        "spearman_point_anomaly_vs_behavior_stall": safe_spearman(
            np.concatenate(ordered), np.concatenate(behavior)
        )
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_overview(summary: dict[str, Any], output: Path) -> None:
    existence = summary["grammar_existence_selected"]
    preferred = (
        "unigram",
        "position",
        "position_context",
        "bigram",
        "markov4",
        "bag6",
        "pst6",
        "pst6_duration",
        "task_unigram",
        "task_position",
        "task_position_context",
        "task_pst6",
    )
    methods = [name for name in preferred if name in existence["mean_hard_bits_per_token"]]
    values = [existence["mean_hard_bits_per_token"][name] for name in methods]
    horizon = summary["failure_detection_fixed_horizon"]
    detection_methods = ("single", "bag", "ordered", "ordered_duration", "behavior")

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].bar(np.arange(len(methods)), values, color="#3f6f8f")
    axes[0].set_xticks(np.arange(len(methods)), methods, rotation=35, ha="right")
    axes[0].set_ylabel("Held-out hard NLL (bits/token)")
    axes[0].set_title("Healthy next-word prediction")

    colors = ("#707070", "#ca6f1e", "#2874a6", "#148f77", "#922b21")
    for method, color in zip(detection_methods, colors):
        x = sorted(int(value) for value in horizon)
        y = [horizon[str(value)]["methods"][method]["within_task_state_auc"] for value in x]
        axes[1].plot(x, y, marker="o", label=method, color=color)
    axes[1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("Query horizon (0-based)")
    axes[1].set_ylabel("Within task/state failure AUC")
    axes[1].set_ylim(0.35, 1.0)
    axes[1].set_title("Outcome discrimination (secondary)")
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def conclusion_label(test: dict[str, Any]) -> str:
    effect = test["mean_left_minus_right"]
    p_value = test["one_sided_p_left_less"]
    if effect < 0.0 and p_value < 0.05:
        return "支持"
    if effect < 0.0:
        return "方向一致但证据不足"
    return "不支持"


def write_report(summary: dict[str, Any], path: Path) -> None:
    existence = summary["grammar_existence_selected"]
    tests = existence["paired_state_tests"]
    fixed = summary["failure_detection_fixed_horizon"]
    false_alarm = summary["healthy_false_alarm"]
    stasis_global = summary["scene8_stasis_detection_global_calibration"]
    stasis = summary["scene8_stasis_detection_task_calibration"]
    phenotype = summary["phenotype"]
    conditional_test = tests["task_position_context_vs_task_position"]
    conditional_ci = conditional_test["ci95"]
    conditional_supported = (
        conditional_test["mean_left_minus_right"] < 0.0
        and conditional_test["one_sided_p_left_less"] < 0.05
    )
    if conditional_supported:
        grammar_judgment = (
            "word history 在已知 task 与绝对 query 位置后仍有 held-out 增量，"
            "支持任务内的条件序列结构；由于该模型使用 task 标签，这不等于 task-invariant grammar。"
        )
    else:
        grammar_judgment = (
            "仍未证明 word history 在已知 task 与绝对 query 位置后有 held-out 增量，"
            "强的 compositional grammar 表述应继续收缩。"
        )

    lines = [
        "# HiMoE-VLA 多轨道路由语法实验报告",
        "",
        "## 结论",
        "",
        f"- **跨 query 有序性：{conclusion_label(tests['ordered_vs_query_shuffle'])}。** "
        f"真实顺序相对 query-shuffle 的 NLL 差为 "
        f"`{tests['ordered_vs_query_shuffle']['mean_left_minus_right']:.4f}` bits/token，"
        f"state-blocked 单侧 `p={tests['ordered_vs_query_shuffle']['one_sided_p_left_less']:.4g}`。",
        f"- **顺序相对词袋：{conclusion_label(tests['pst_vs_bag'])}。** "
        f"PST 相对 bag-context 的 NLL 差为 "
        f"`{tests['pst_vs_bag']['mean_left_minus_right']:.4f}` bits/token，"
        f"`p={tests['pst_vs_bag']['one_sided_p_left_less']:.4g}`。",
        f"- **相对绝对 query 时钟：{conclusion_label(tests['pst_vs_position'])}。** "
        f"PST 相对 position-only baseline 的 NLL 差为 "
        f"`{tests['pst_vs_position']['mean_left_minus_right']:.4f}` bits/token，"
        f"`p={tests['pst_vs_position']['one_sided_p_left_less']:.4g}`。",
        f"- **任务与阶段之后的历史增益：{conclusion_label(conditional_test)}。** "
        f"task-position+history 相对 task-position 的 NLL 差为 "
        f"`{conditional_test['mean_left_minus_right']:.4f}` bits/token，"
        f"95% CI `[{conditional_ci[0]:.4f}, {conditional_ci[1]:.4f}]`，"
        f"单侧 `p={conditional_test['one_sided_p_left_less']:.4g}`。",
        "- **阶段条件的早期失败监控：不支持。** q7/q12 的 task-phase+history 相对 "
        f"task-phase AUC 增量分别为 "
        f"`{fixed['7']['auc_improvements']['task_phase_history_minus_task_phase']['auc_left_minus_right']:+.4f}`/"
        f"`{fixed['12']['auc_improvements']['task_phase_history_minus_task_phase']['auc_left_minus_right']:+.4f}`；"
        "健康预测增益没有转化成早期失败区分。",
        f"- **query 内 flow 顺序：{conclusion_label(tests['lexical_vs_flow_shuffle'])}。** "
        f"真实 flow 相对重算后的 flow-shuffle lexical NLL 差为 "
        f"`{tests['lexical_vs_flow_shuffle']['mean_left_minus_right']:.4f}` nats/query，"
        f"`p={tests['lexical_vs_flow_shuffle']['one_sided_p_left_less']:.4g}`；"
        f"平均 word 改变率 `{existence['flow_word_change_fraction']:.1%}`。",
        f"- **duration 增益：{conclusion_label(tests['duration_vs_pst'])}。** "
        f"PST+duration 相对 PST 的差为 "
        f"`{tests['duration_vs_pst']['mean_left_minus_right']:.4f}` bits/token。",
        "",
        f"固定四阶 Markov 相对 PST 的 NLL 差为 "
        f"`{-tests['pst_vs_markov4']['mean_left_minus_right']:.4f}` bits/token；"
        "因此有序结构是否存在与 PST 是否是最佳模型是两个问题。",
        "",
        "因此最终判断是：支持多尺度 routing temporal structure。"
        + grammar_judgment
        + " flow-shuffle 只检验构词层。",
        "",
        "## 数据与防泄漏",
        "",
        f"共 `{summary['corpus']['episodes']}` episodes、`{summary['corpus']['queries']}` queries，"
        f"其中成功 `{summary['corpus']['successes']}`、失败 `{summary['corpus']['failures']}`。",
        f"按全局 init-state 做 `{summary['config']['folds']}` 折交叉拟合；每折 train、calibration、test "
        "state 互斥。同一 task/init-state 下 32 条 noise siblings 不会跨 split。",
        "Tokenizer、PCA、GMM、grammar 只看训练 state 的成功 episode；词表大小只按 calibration "
        "成功轨迹选择；所有报告数来自未见 test state。",
        "",
        "## 健康序列预测",
        "",
        "| 模型 | bits/token |",
        "|---|---:|",
    ]
    model_order = (
        "unigram",
        "position",
        "position_context",
        "bigram",
        "markov4",
        "bag6",
        "pst6",
        "pst6_duration",
        "task_unigram",
        "task_position",
        "task_position_context",
        "task_pst6",
    )
    for name in model_order:
        value = existence["mean_hard_bits_per_token"][name]
        lines.append(f"| {name} | {value:.4f} |")
    lines.extend(
        [
            "",
            "task-conditioned 项使用任务标签，只是检查任务异质性的 oracle control，不属于严格 MoE-only 在线模型。",
            "`position_context` 是严格嵌套对照：先给定绝对位置，再只用同一位置内的最近历史更新；"
            "`task_position_context` 进一步给定任务标签。后者是本轮判断 history 是否超出任务阶段时钟的主检验。",
            "",
            "## 固定前缀失败区分",
            "",
            "这是次级 outcome 评价，不等同于 loop/static 因果识别。AUC 在 task × init-state 内配对后汇总。",
            "",
            "| horizon | N (S/F) | single | bag | ordered | +duration | behavior |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(fixed, key=int):
        item = fixed[horizon]
        values = [
            item["methods"][name]["within_task_state_auc"]
            for name in ("single", "bag", "ordered", "ordered_duration", "behavior")
        ]
        lines.append(
            f"| {horizon} | {item['episodes']} ({item['successes']}/{item['failures']}) | "
            + " | ".join(f"{value:.3f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "有序模型相对基线的 AUC 增量（state-blocked 95% CI）：",
            "",
            "| horizon | ordered-single | ordered-bag | duration-ordered |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(fixed, key=int):
        item = fixed[horizon]
        improvement = item["auc_improvements"]
        cells = []
        for name in (
            "ordered_minus_single",
            "ordered_minus_bag",
            "duration_minus_ordered",
        ):
            value = improvement[name]
            cells.append(
                f"{value['auc_left_minus_right']:+.3f} "
                f"[{value['state_blocked_ci95'][0]:+.3f}, {value['state_blocked_ci95'][1]:+.3f}]"
            )
        lines.append(f"| {horizon} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "### 阶段条件失败审计",
            "",
            "下表仍是 within task × state AUC。`residual` 为 history NLL 减 position NLL，"
            "直接读取相对阶段基线的额外异常；task 项是使用任务标签的 oracle control。",
            "",
            "| horizon | phase | phase+history | residual | task+phase | task+phase+history | task residual |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(fixed, key=int):
        item = fixed[horizon]
        values = [
            item["methods"][name]["within_task_state_auc"]
            for name in (
                "phase",
                "phase_history",
                "phase_residual",
                "task_phase",
                "task_phase_history",
                "task_phase_residual",
            )
        ]
        lines.append(f"| {horizon} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    lines.extend(
        [
            "",
            "history 模型相对 phase baseline 的 AUC 增量（state-blocked 95% CI）：",
            "",
            "| horizon | phase+history - phase | task phase+history - task phase | task residual - task phase |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon in sorted(fixed, key=int):
        improvement = fixed[horizon]["auc_improvements"]
        cells = []
        for name in (
            "phase_history_minus_phase",
            "task_phase_history_minus_task_phase",
            "task_phase_residual_minus_task_phase",
        ):
            value = improvement[name]
            cells.append(
                f"{value['auc_left_minus_right']:+.3f} "
                f"[{value['state_blocked_ci95'][0]:+.3f}, {value['state_blocked_ci95'][1]:+.3f}]"
            )
        lines.append(f"| {horizon} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Episode 误报与 scene8 停滞",
            "",
            f"阈值目标为 calibration 健康 episode 级 `{summary['config']['healthy_episode_fpr']:.1%}` FPR。",
            "scene8 事件比较使用 scene8 成功 calibration 重新标定 CDF/阈值；这是带任务标签的公平 "
            "operating-point audit，不属于严格全局 MoE-only 部署值。",
            "",
            "| 方法 | 全局 test 健康 FPR | scene8 健康 FPR | onset 前召回 | onset+3 召回 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in ("single", "bag", "ordered", "ordered_duration", "behavior"):
        lines.append(
            f"| {method} | {false_alarm['methods'][method]['episode_false_alarm_rate']:.3f} | "
            f"{stasis['methods'][method]['scene8_success_false_alarm_rate']:.3f} | "
            f"{stasis['methods'][method]['recall_by_physical_onset']:.3f} | "
            f"{stasis['methods'][method]['recall_by_onset_plus_3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "阶段条件方法的 scene8 task-calibrated operating point：",
            "",
            "| 方法 | scene8 健康 FPR | onset 前召回 | onset+3 召回 |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in (
        "phase",
        "phase_history",
        "phase_residual",
        "task_phase",
        "task_phase_history",
        "task_phase_residual",
    ):
        lines.append(
            f"| {method} | {stasis['methods'][method]['scene8_success_false_alarm_rate']:.3f} | "
            f"{stasis['methods'][method]['recall_by_physical_onset']:.3f} | "
            f"{stasis['methods'][method]['recall_by_onset_plus_3']:.3f} |"
        )
    lines.extend(
        [
            "",
            "全局 calibration 下的 scene8 onset 前召回分别为："
            + ", ".join(
                f"{method}={stasis_global['methods'][method]['recall_by_physical_onset']:.3f}"
                for method in ("single", "bag", "ordered", "ordered_duration", "behavior")
            )
            + "。",
            "",
            "## 多尺度 Fl 与 Sy",
            "",
            "预先分开计算：`Fl_query` 为最近 7 个 query 的平均 entropy slope；"
            "`Fl_flow` 为同一 query 内 late-minus-early flow entropy；"
            "`Sy` 为 L15/f9 的 token Top-4 support consensus。下表为 scene8 停滞失败相对成功的 within-state AUC。",
            "",
            "| q | Fl_query | Fl_flow | Sy_L15/f9 | LowRank_L15/f9 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    phenotype_by_horizon = phenotype["scene8_fixed_horizon_stasis_vs_success"]
    for horizon in ("12", "20", "34"):
        item = phenotype_by_horizon[horizon]
        lines.append(
            f"| {horizon} | {item['cross_query_entropy_slope_w7']['within_state_auc']:.3f} | "
            f"{item['within_query_flow_flattening']['within_state_auc']:.3f} | "
            f"{item['terminal_l15_f9_top4_sync']['within_state_auc']:.3f} | "
            f"{item['terminal_l15_f9_lowrank']['within_state_auc']:.3f} |"
        )
    axis = phenotype["axis_relation"]
    lines.extend(
        [
            "",
            f"两轴在全部 scene8 queries 上的 Spearman `rho="
            f"{axis['spearman_all_scene8_queries_cross_query_flattening_vs_terminal_sync']:.3f}`，"
            f"在 q34 为 `rho={axis['spearman_at_q34_cross_query_flattening_vs_terminal_sync']:.3f}`。"
            "这支持将二者作为并行轨道读取，而不是硬编码成先后字母；但 q34 是晚期关联。",
            "",
            "## 解释边界",
            "",
            "- scene8 onset 来自仿真中两只 moka pot 的物理进展定义，不使用 MoE；19 条含糊失败被排除。",
            "- `Fl` 是跨 query 或 flow 的 entropy 趋势，`Sy` 是 L15/f9 的 Top-4 支持同步；报告分别计算，"
            "不会把它们硬编码成互斥的 `Fl Fl Sy Sy`。",
            "- 较晚 horizon 会有 survivor/episode-length 选择，尤其 t27/t34 只能解释为晚期读数。",
            "- 主实验 split 阻断了 task/init-state root siblings；独立的 state+seed 双轴留出结果由"
            " `run_dual_axis_audit` 生成，不能用主实验的全样本覆盖数字替代。",
            "- scaler/PCA/GMM 按健康 query 拟合，较长的 scene8 成功轨迹会贡献更多 tokenizer 权重；"
            "held-out NLL 汇总则以 episode 为单位。",
            "- behavior baseline 和相关性用于检查信号是否只是动作/物理停滞的读出；本实验不能给出路由因果结论。",
            "- 当前数据没有可靠的 loop/static/mixed 分类和延长 timeout rollout，因而不评价 `<END>` "
            "延长策略、候选选择或训练时正则化。",
            "",
            "机器可读的完整效应、置信区间、fold 选择和 phenotype 结果见 `summary.json`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(exist_ok=True)

    print("Loading extracted corpus...", flush=True)
    corpus = load_corpus(args.features_dir, args.stasis_labels)
    print(
        f"Corpus: {len(corpus.episodes):,} episodes, {len(corpus.features):,} queries, "
        f"{len(corpus.tasks)} tasks",
        flush=True,
    )
    print("Building 2,187-D ordered descriptors...", flush=True)
    descriptor = build_query_descriptors(corpus.features)
    flow_descriptor = build_query_descriptors(corpus.flow_shuffled_features)
    folds = make_state_folds(
        np.asarray([episode.init_state_id for episode in corpus.episodes]),
        config.folds,
        config.seed,
    )

    existence_records: list[dict[str, Any]] = []
    selected_existence_records: list[dict[str, Any]] = []
    detection: dict[int, dict[str, Any]] = {}
    scene_detection: dict[int, dict[str, Any]] = {}
    fold_summaries: list[dict[str, Any]] = []
    prototypes: list[dict[str, Any]] | None = None

    for fold_index, split in enumerate(folds):
        print(
            f"Fold {fold_index + 1}/{len(folds)}: train={split['train'].tolist()}, "
            f"cal={split['calibration'].tolist()}, test={split['test'].tolist()}",
            flush=True,
        )
        train_episodes = selected_episodes(corpus, split["train"], success=True)
        calibration_episodes = selected_episodes(corpus, split["calibration"], success=True)
        test_success = selected_episodes(corpus, split["test"], success=True)
        test_all = selected_episodes(corpus, split["test"], success=None)
        train_rows = episode_rows(train_episodes)

        preprocessor = Preprocessor(config.pca_dim, config.seed + fold_index).fit(
            descriptor[train_rows]
        )
        projected = preprocessor.transform(descriptor)
        flow_projected = preprocessor.transform(flow_descriptor)
        print(
            f"  PCA retained variance={preprocessor.pca.explained_variance_ratio_.sum():.3f}",
            flush=True,
        )

        candidates: dict[int, TokenizedFold] = {}
        fold_candidate_summary: dict[str, Any] = {}
        for n_words in config.word_counts:
            print(f"  fitting GMM/PST K={n_words}", flush=True)
            tokenized = fit_tokenizer_fold(
                corpus,
                projected,
                flow_projected,
                train_episodes,
                calibration_episodes,
                n_words,
                config,
                fold_index,
            )
            candidates[n_words] = tokenized
            fold_candidate_summary[str(n_words)] = tokenized.diagnostics
            for episode in test_success:
                existence_records.append(
                    score_existence_episode(episode, tokenized, config, fold_index)
                )

        selected_words = min(candidates.items(), key=lambda item: item[1].calibration_nll)[0]
        selected = candidates[selected_words]
        print(
            f"  selected K={selected_words} by healthy calibration NLL "
            f"({selected.calibration_nll:.4f})",
            flush=True,
        )
        selected_existence_records.extend(
            [
                record
                for record in existence_records
                if record["fold"] == fold_index and record["n_words"] == selected_words
            ]
        )

        behavior_stall, behavior_median, behavior_scale = fit_behavior_reference(
            corpus, train_episodes
        )
        references, thresholds = calibrate_detection(
            calibration_episodes, selected, behavior_stall, config
        )
        for episode in test_all:
            detection[episode.index] = score_detection_episode(
                episode,
                selected,
                behavior_stall,
                references,
                thresholds,
                config,
                fold_index,
            )
        scene_train = [episode for episode in train_episodes if episode.task == SCENE8_TASK]
        scene_calibration = [
            episode for episode in calibration_episodes if episode.task == SCENE8_TASK
        ]
        scene_test = [episode for episode in test_all if episode.task == SCENE8_TASK]
        scene_behavior_stall, _, _ = fit_behavior_reference(corpus, scene_train)
        scene_references, scene_thresholds = calibrate_detection(
            scene_calibration, selected, scene_behavior_stall, config
        )
        for episode in scene_test:
            scene_detection[episode.index] = score_detection_episode(
                episode,
                selected,
                scene_behavior_stall,
                scene_references,
                scene_thresholds,
                config,
                fold_index,
            )
        if prototypes is None:
            prototypes = word_prototypes(corpus, train_episodes, selected.words, selected.n_words)

        model_payload = {
            "schema_version": 2,
            "fold": fold_index,
            "split": split,
            "config": asdict(config),
            "preprocessor": preprocessor,
            "tokenizer": selected.tokenizer,
            "grammars": selected.grammars.models,
            "task_grammars": selected.grammars.task_models,
            "duration_model": selected.grammars.duration_model,
            "shuffled_train_pst": selected.grammars.shuffled_train_pst,
            "behavior_median": behavior_median,
            "behavior_scale": behavior_scale,
            "thresholds": thresholds,
            "calibration_references": references,
            "scene8_thresholds": scene_thresholds,
            "scene8_calibration_references": scene_references,
            "feature_names": corpus.feature_names,
        }
        joblib.dump(model_payload, model_dir / f"fold_{fold_index}.joblib", compress=3)
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_states": split["train"].tolist(),
                "calibration_states": split["calibration"].tolist(),
                "test_states": split["test"].tolist(),
                "train_success_episodes": len(train_episodes),
                "calibration_success_episodes": len(calibration_episodes),
                "test_episodes": len(test_all),
                "pca_explained_variance": float(preprocessor.pca.explained_variance_ratio_.sum()),
                "candidates": fold_candidate_summary,
                "selected_words": selected_words,
                "detection_thresholds": thresholds,
                "scene8_detection_thresholds": scene_thresholds,
            }
        )
        del projected, flow_projected, candidates

    if set(detection) != {episode.index for episode in corpus.episodes}:
        raise AssertionError("cross-fitting did not score every episode exactly once")
    expected_scene = {episode.index for episode in corpus.episodes if episode.task == SCENE8_TASK}
    if set(scene_detection) != expected_scene:
        raise AssertionError("scene8 task-calibrated scoring did not cover every episode")

    by_vocabulary = {
        str(n_words): aggregate_existence(
            [record for record in existence_records if record["n_words"] == n_words], config
        )
        for n_words in config.word_counts
    }
    summary = {
        "schema_version": 2,
        "experiment": "multitrack-routing-grammar-v2-phase-conditioned",
        "config": asdict(config),
        "corpus": corpus_summary(corpus),
        "folds": fold_summaries,
        "grammar_existence_by_vocabulary": by_vocabulary,
        "grammar_existence_selected": aggregate_existence(selected_existence_records, config),
        "failure_detection_fixed_horizon": evaluate_failure_horizons(corpus, detection, config),
        "healthy_false_alarm": evaluate_false_alarms(corpus, detection),
        "scene8_stasis_detection_global_calibration": evaluate_stasis(corpus, detection),
        "scene8_stasis_detection_task_calibration": evaluate_stasis(corpus, scene_detection),
        "phenotype": phenotype_analysis(corpus, config),
        "behavior_association": behavior_association(corpus, detection),
        "word_prototypes_fold0": prototypes,
        "runtime_seconds": time.time() - started,
        "guardrails": [
            "All tokenizer and grammar fits use successful train-state episodes only.",
            "Vocabulary size is selected on successful calibration-state episodes only.",
            "All reported predictions are cross-fitted on unseen init states.",
            "Noise-seed IDs recur across independent init states; this is not a dual-axis holdout.",
            "Tokenizer fitting is query-weighted, so longer healthy episodes contribute more rows.",
            "Outcome discrimination is associative, not evidence that MoE causes failure.",
            "t27/t34 are late readouts subject to survivor selection.",
        ],
    }
    summary = json_ready(summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "word_prototypes.json").write_text(
        json.dumps(summary["word_prototypes_fold0"], indent=2, sort_keys=True) + "\n"
    )
    write_report(summary, args.output_dir / "REPORT.zh.md")
    write_overview(summary, args.output_dir / "overview.png")
    print(
        f"Finished in {time.time() - started:.1f}s; report: {args.output_dir / 'REPORT.zh.md'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
