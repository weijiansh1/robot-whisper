from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import logsumexp

ContextMode = Literal["ordered", "bag"]


def shuffled_sequence(sequence: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.int16).copy()
    if len(values) > 1:
        np.random.default_rng(seed).shuffle(values)
    return values


@dataclass
class CountGrammar:
    vocabulary_size: int
    max_order: int
    alpha: float = 0.5
    min_support: int = 1
    kl_delta: float = 0.0
    context_mode: ContextMode = "ordered"

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary must include words and END")
        if self.max_order < 0 or self.alpha <= 0.0 or self.min_support < 1:
            raise ValueError("invalid grammar hyperparameters")
        self.end_token = self.vocabulary_size - 1
        self.bos_token = self.vocabulary_size
        self.counts: list[dict[tuple[int, ...], np.ndarray]] = [
            {} for _ in range(self.max_order + 1)
        ]

    def _context(self, history: list[int] | np.ndarray, order: int) -> tuple[int, ...]:
        if order == 0:
            return ()
        context = tuple(int(value) for value in history[-order:])
        if self.context_mode == "bag":
            return tuple(sorted(context))
        return context

    def fit(self, sequences: list[np.ndarray]) -> "CountGrammar":
        mutable: list[defaultdict[tuple[int, ...], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.vocabulary_size, dtype=np.int64))
            for _ in range(self.max_order + 1)
        ]
        for sequence in sequences:
            history = [self.bos_token]
            targets = [int(value) for value in sequence] + [self.end_token]
            for target in targets:
                if target < 0 or target >= self.vocabulary_size:
                    raise ValueError(f"target token {target} outside vocabulary")
                for order in range(min(self.max_order, len(history)) + 1):
                    mutable[order][self._context(history, order)][target] += 1
                history.append(target)
        self.counts = [dict(items) for items in mutable]
        if () not in self.counts[0]:
            raise ValueError("cannot fit grammar on no sequences")
        return self

    def _smoothed(self, counts: np.ndarray) -> np.ndarray:
        return (counts + self.alpha) / (counts.sum() + self.alpha * self.vocabulary_size)

    def distribution(self, history: list[int] | np.ndarray) -> tuple[np.ndarray, int, int]:
        maximum = min(self.max_order, len(history))
        root = self._smoothed(self.counts[0][()])
        chosen = root
        chosen_order = 0
        chosen_support = int(self.counts[0][()].sum())
        for order in range(1, maximum + 1):
            key = self._context(history, order)
            counts = self.counts[order].get(key)
            if counts is None or counts.sum() < self.min_support:
                continue
            candidate = self._smoothed(counts)
            if self.kl_delta > 0.0:
                divergence = float(np.sum(candidate * (np.log(candidate) - np.log(chosen))))
                if divergence < self.kl_delta:
                    continue
            chosen = candidate
            chosen_order = order
            chosen_support = int(counts.sum())
        return chosen, chosen_order, chosen_support

    def priors(
        self, sequence: np.ndarray, include_end: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        history = [self.bos_token]
        targets = [int(value) for value in sequence]
        if include_end:
            targets.append(self.end_token)
        output = np.empty((len(targets), self.vocabulary_size), dtype=np.float64)
        depths = np.empty(len(targets), dtype=np.int8)
        for index, target in enumerate(targets):
            output[index], depths[index], _ = self.distribution(history)
            history.append(target)
        return output, depths

    def hard_nll(
        self, sequence: np.ndarray, include_end: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(sequence, dtype=np.int64)
        if include_end:
            targets = np.r_[targets, self.end_token]
        priors, depths = self.priors(sequence, include_end=include_end)
        probability = priors[np.arange(len(targets)), targets]
        return -np.log(np.maximum(probability, 1e-300)), depths

    def continuous_nll(
        self, sequence: np.ndarray, component_log_likelihood: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(sequence) != len(component_log_likelihood):
            raise ValueError("sequence and emission lengths differ")
        if component_log_likelihood.shape[1] != self.vocabulary_size - 1:
            raise ValueError("emission vocabulary does not match grammar")
        priors, depths = self.priors(sequence, include_end=False)
        word_prior = priors[:, :-1]
        word_prior /= np.maximum(word_prior.sum(axis=1, keepdims=True), 1e-300)
        log_probability = np.log(np.maximum(word_prior, 1e-300))
        return -logsumexp(log_probability + component_log_likelihood, axis=1), depths


@dataclass
class PositionGrammar:
    """A clock-only baseline that predicts words from absolute query position."""

    vocabulary_size: int
    alpha: float = 0.5
    min_support: int = 20

    def fit(self, sequences: list[np.ndarray]) -> "PositionGrammar":
        self.end_token = self.vocabulary_size - 1
        self.root_counts = np.zeros(self.vocabulary_size, dtype=np.int64)
        maximum = max((len(sequence) for sequence in sequences), default=0)
        self.position_counts = np.zeros((maximum + 1, self.vocabulary_size), dtype=np.int64)
        for sequence in sequences:
            targets = np.r_[np.asarray(sequence, dtype=np.int64), self.end_token]
            for position, target in enumerate(targets):
                self.root_counts[target] += 1
                self.position_counts[position, target] += 1
        if self.root_counts.sum() == 0:
            raise ValueError("cannot fit position grammar on no sequences")
        return self

    def _smoothed(self, counts: np.ndarray) -> np.ndarray:
        return (counts + self.alpha) / (counts.sum() + self.alpha * self.vocabulary_size)

    def priors(
        self, sequence: np.ndarray, include_end: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        count = len(sequence) + int(include_end)
        output = np.empty((count, self.vocabulary_size), dtype=np.float64)
        depth = np.zeros(count, dtype=np.int8)
        root = self._smoothed(self.root_counts)
        for position in range(count):
            if (
                position < len(self.position_counts)
                and self.position_counts[position].sum() >= self.min_support
            ):
                output[position] = self._smoothed(self.position_counts[position])
                depth[position] = 1
            else:
                output[position] = root
        return output, depth

    def hard_nll(
        self, sequence: np.ndarray, include_end: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(sequence, dtype=np.int64)
        if include_end:
            targets = np.r_[targets, self.end_token]
        priors, depths = self.priors(sequence, include_end)
        return -np.log(np.maximum(priors[np.arange(len(targets)), targets], 1e-300)), depths

    def continuous_nll(
        self, sequence: np.ndarray, component_log_likelihood: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(sequence) != len(component_log_likelihood):
            raise ValueError("sequence and emission lengths differ")
        priors, depths = self.priors(sequence, include_end=False)
        word_prior = priors[:, :-1]
        word_prior /= np.maximum(word_prior.sum(axis=1, keepdims=True), 1e-300)
        value = -logsumexp(
            np.log(np.maximum(word_prior, 1e-300)) + component_log_likelihood,
            axis=1,
        )
        return value, depths


@dataclass
class PositionContextGrammar:
    """Predict words from absolute position, then update with local word history.

    This is a nested control for testing whether history helps after the episode
    clock is already known. Context counts are shrunk toward the position-only
    distribution, so unsupported cells reduce exactly to the clock baseline.
    """

    vocabulary_size: int
    max_order: int = 2
    alpha: float = 0.5
    min_support: int = 20
    kl_delta: float = 0.0
    context_mode: ContextMode = "ordered"

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ValueError("vocabulary must include words and END")
        if self.max_order < 1 or self.alpha <= 0.0 or self.min_support < 1:
            raise ValueError("invalid position-context hyperparameters")
        if self.context_mode not in ("ordered", "bag"):
            raise ValueError("invalid context mode")
        self.end_token = self.vocabulary_size - 1
        self.bos_token = self.vocabulary_size

    def _context(self, history: list[int] | np.ndarray, order: int) -> tuple[int, ...]:
        context = tuple(int(value) for value in history[-order:])
        if self.context_mode == "bag":
            return tuple(sorted(context))
        return context

    def fit(self, sequences: list[np.ndarray]) -> "PositionContextGrammar":
        self.position_model = PositionGrammar(
            self.vocabulary_size,
            alpha=self.alpha,
            min_support=self.min_support,
        ).fit(sequences)
        mutable: list[defaultdict[tuple[int, tuple[int, ...]], np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.vocabulary_size, dtype=np.int64))
            for _ in range(self.max_order + 1)
        ]
        for sequence in sequences:
            history = [self.bos_token]
            targets = [int(value) for value in sequence] + [self.end_token]
            for position, target in enumerate(targets):
                if target < 0 or target >= self.vocabulary_size:
                    raise ValueError(f"target token {target} outside vocabulary")
                for order in range(1, min(self.max_order, len(history)) + 1):
                    context = self._context(history, order)
                    mutable[order][(position, context)][target] += 1
                history.append(target)
        self.counts = [dict(items) for items in mutable]
        return self

    def _position_distribution(self, position: int) -> np.ndarray:
        if (
            position < len(self.position_model.position_counts)
            and self.position_model.position_counts[position].sum() >= self.min_support
        ):
            counts = self.position_model.position_counts[position]
        else:
            counts = self.position_model.root_counts
        return self.position_model._smoothed(counts)

    def distribution(
        self, position: int, history: list[int] | np.ndarray
    ) -> tuple[np.ndarray, int, int]:
        baseline = self._position_distribution(position)
        chosen = baseline
        chosen_order = 0
        chosen_support = 0
        prior_strength = self.alpha * self.vocabulary_size
        maximum = min(self.max_order, len(history))
        for order in range(1, maximum + 1):
            context = self._context(history, order)
            counts = self.counts[order].get((position, context))
            if counts is None or counts.sum() < self.min_support:
                continue
            support = float(counts.sum())
            candidate = (counts + prior_strength * baseline) / (support + prior_strength)
            if self.kl_delta > 0.0:
                divergence = float(np.sum(candidate * (np.log(candidate) - np.log(baseline))))
                if divergence < self.kl_delta:
                    continue
            chosen = candidate
            chosen_order = order
            chosen_support = int(support)
        return chosen, chosen_order, chosen_support

    def priors(
        self, sequence: np.ndarray, include_end: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        history = [self.bos_token]
        targets = [int(value) for value in sequence]
        if include_end:
            targets.append(self.end_token)
        output = np.empty((len(targets), self.vocabulary_size), dtype=np.float64)
        depths = np.empty(len(targets), dtype=np.int8)
        for position, target in enumerate(targets):
            output[position], depths[position], _ = self.distribution(position, history)
            history.append(target)
        return output, depths

    def hard_nll(
        self, sequence: np.ndarray, include_end: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(sequence, dtype=np.int64)
        if include_end:
            targets = np.r_[targets, self.end_token]
        priors, depths = self.priors(sequence, include_end=include_end)
        probability = priors[np.arange(len(targets)), targets]
        return -np.log(np.maximum(probability, 1e-300)), depths

    def continuous_nll(
        self, sequence: np.ndarray, component_log_likelihood: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(sequence) != len(component_log_likelihood):
            raise ValueError("sequence and emission lengths differ")
        if component_log_likelihood.shape[1] != self.vocabulary_size - 1:
            raise ValueError("emission vocabulary does not match grammar")
        priors, depths = self.priors(sequence, include_end=False)
        word_prior = priors[:, :-1]
        word_prior /= np.maximum(word_prior.sum(axis=1, keepdims=True), 1e-300)
        value = -logsumexp(
            np.log(np.maximum(word_prior, 1e-300)) + component_log_likelihood,
            axis=1,
        )
        return value, depths


@dataclass
class DurationModel:
    n_words: int
    alpha: float = 0.5
    min_support: int = 10

    def fit(self, sequences: list[np.ndarray]) -> "DurationModel":
        runs: list[list[int]] = [[] for _ in range(self.n_words)]
        for sequence in sequences:
            if len(sequence) == 0:
                continue
            current = int(sequence[0])
            length = 1
            for value in sequence[1:]:
                word = int(value)
                if word == current:
                    length += 1
                else:
                    runs[current].append(length)
                    current = word
                    length = 1
            runs[current].append(length)
        self.run_lengths = [np.asarray(items, dtype=np.int16) for items in runs]
        return self

    def repeat_probability(self, word: int, run_length: int) -> float | None:
        runs = self.run_lengths[word]
        eligible = int(np.sum(runs >= run_length))
        if eligible < self.min_support:
            return None
        repeated = int(np.sum(runs > run_length))
        return float((repeated + self.alpha) / (eligible + 2.0 * self.alpha))

    def survival_nll(self, word: int, run_length: int) -> float:
        runs = self.run_lengths[word]
        if len(runs) == 0:
            return 0.0
        survival = (np.sum(runs >= run_length) + self.alpha) / (len(runs) + self.alpha)
        return float(-np.log(max(survival, 1e-300)))

    def adjust(self, base: np.ndarray, word: int, run_length: int) -> np.ndarray:
        repeat = self.repeat_probability(word, run_length)
        if repeat is None:
            return base
        output = np.asarray(base, dtype=np.float64).copy()
        other_mass = output.sum() - output[word]
        if other_mass <= 1e-300:
            return base
        output *= (1.0 - repeat) / other_mass
        output[word] = repeat
        output /= output.sum()
        return output


@dataclass
class DurationGrammar:
    grammar: CountGrammar
    duration: DurationModel

    def priors(
        self, sequence: np.ndarray, include_end: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        history = [self.grammar.bos_token]
        targets = [int(value) for value in sequence]
        if include_end:
            targets.append(self.grammar.end_token)
        output = np.empty((len(targets), self.grammar.vocabulary_size), dtype=np.float64)
        depths = np.empty(len(targets), dtype=np.int8)
        current_word = -1
        run_length = 0
        for index, target in enumerate(targets):
            prior, depth, _ = self.grammar.distribution(history)
            if current_word >= 0:
                prior = self.duration.adjust(prior, current_word, run_length)
            output[index] = prior
            depths[index] = depth
            if target == current_word:
                run_length += 1
            else:
                current_word = target
                run_length = 1
            history.append(target)
        return output, depths

    def hard_nll(
        self, sequence: np.ndarray, include_end: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(sequence, dtype=np.int64)
        if include_end:
            targets = np.r_[targets, self.grammar.end_token]
        priors, depths = self.priors(sequence, include_end=include_end)
        return -np.log(np.maximum(priors[np.arange(len(targets)), targets], 1e-300)), depths

    def continuous_nll(
        self, sequence: np.ndarray, component_log_likelihood: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        priors, depths = self.priors(sequence, include_end=False)
        word_prior = priors[:, :-1]
        word_prior /= np.maximum(word_prior.sum(axis=1, keepdims=True), 1e-300)
        value = -logsumexp(
            np.log(np.maximum(word_prior, 1e-300)) + component_log_likelihood, axis=1
        )
        return value, depths

    def duration_nll(self, sequence: np.ndarray) -> np.ndarray:
        output = np.zeros(len(sequence), dtype=np.float64)
        current = -1
        run_length = 0
        for index, raw_word in enumerate(sequence):
            word = int(raw_word)
            if word == current:
                run_length += 1
            else:
                current = word
                run_length = 1
            output[index] = self.duration.survival_nll(word, run_length)
        return output
