"""Small age-expanded hidden semi-Markov filter for MoE mode belief."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HSMMModel:
    states: tuple[str, ...]
    means: np.ndarray
    variances: np.ndarray
    initial: np.ndarray
    survival: np.ndarray
    exit_probability: np.ndarray
    max_age: int

    def validate(self) -> None:
        count = len(self.states)
        if self.means.shape != self.variances.shape or self.means.shape[0] != count:
            raise ValueError("emission shape mismatch")
        if self.initial.shape != (count,):
            raise ValueError("initial shape mismatch")
        if self.survival.shape != (count, self.max_age):
            raise ValueError("survival shape mismatch")
        if self.exit_probability.shape != (count, count):
            raise ValueError("exit probability shape mismatch")
        if not np.allclose(self.initial.sum(), 1.0):
            raise ValueError("initial distribution does not sum to one")
        if not np.allclose(self.exit_probability.sum(axis=1), 1.0):
            raise ValueError("exit distributions do not sum to one")

    def transition(self, belief: np.ndarray) -> np.ndarray:
        """Advance an age-expanded belief by one query without an observation."""

        output = np.zeros_like(belief)
        staying = belief * self.survival
        output[:, 1:] += staying[:, :-1]
        output[:, -1] += staying[:, -1]
        exiting = (belief * (1.0 - self.survival)).sum(axis=1)
        output[:, 0] += exiting @ self.exit_probability
        return output

    def emission_log_likelihood(self, observation: np.ndarray) -> np.ndarray:
        delta = observation[None, :] - self.means
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variances) + delta**2 / self.variances,
            axis=1,
        )

    def filter_history(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return state posteriors and age beliefs for every causal prefix."""

        self.validate()
        observations = np.asarray(observations, dtype=np.float64)
        belief = np.zeros((len(self.states), self.max_age), dtype=np.float64)
        belief[:, 0] = self.initial
        posterior = np.empty((len(observations), len(self.states)), dtype=np.float64)
        history = np.empty(
            (len(observations), len(self.states), self.max_age), dtype=np.float64
        )
        for query, observation in enumerate(observations):
            if query:
                belief = self.transition(belief)
            log_likelihood = self.emission_log_likelihood(observation)
            log_likelihood -= np.max(log_likelihood)
            belief *= np.exp(log_likelihood)[:, None]
            normalizer = belief.sum()
            if not np.isfinite(normalizer) or normalizer <= 0:
                raise FloatingPointError("HSMM posterior collapsed")
            belief /= normalizer
            posterior[query] = belief.sum(axis=1)
            history[query] = belief
        return posterior, history

    def filter(self, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return state posterior and the full final age belief for a causal prefix."""

        posterior, history = self.filter_history(observations)
        return posterior, history[-1]

    def forecast(self, belief: np.ndarray, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
        output = {}
        current = np.asarray(belief, dtype=np.float64).copy()
        for step in range(1, max(horizons) + 1):
            current = self.transition(current)
            if step in horizons:
                output[step] = current.sum(axis=1)
        return output


def fit_model(
    observations: np.ndarray,
    labels: np.ndarray,
    sequences: list[np.ndarray],
    states: tuple[str, ...],
    max_age: int = 12,
) -> HSMMModel:
    observations = np.asarray(observations, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    count = len(states)
    global_variance = observations.var(axis=0) + 0.05
    means = np.empty((count, observations.shape[1]), dtype=np.float64)
    variances = np.empty_like(means)
    for state in range(count):
        selected = observations[labels == state]
        if len(selected) < 4:
            raise ValueError(f"state {states[state]} has only {len(selected)} observations")
        means[state] = selected.mean(axis=0)
        variances[state] = 0.8 * selected.var(axis=0) + 0.2 * global_variance
    variances = np.maximum(variances, 0.05)

    initial_count = np.full(count, 0.5, dtype=np.float64)
    boundary = np.full((count, count), 0.0, dtype=np.float64)
    lengths: list[list[int]] = [[] for _ in range(count)]
    allowed = {
        "H": ("C", "PL", "PS"),
        "C": ("H", "PL", "PS"),
        "PL": ("L", "H"),
        "L": ("H",),
        "PS": ("S", "H"),
        "S": ("H",),
    }
    state_index = {name: index for index, name in enumerate(states)}
    for sequence in sequences:
        if not len(sequence):
            continue
        initial_count[int(sequence[0])] += 1
        starts = np.r_[0, np.flatnonzero(sequence[1:] != sequence[:-1]) + 1]
        stops = np.r_[starts[1:], len(sequence)]
        for segment, (start, stop) in enumerate(zip(starts, stops)):
            state = int(sequence[start])
            lengths[state].append(int(stop - start))
            if segment + 1 < len(starts):
                following = int(sequence[starts[segment + 1]])
                boundary[state, following] += 1

    exit_probability = np.zeros((count, count), dtype=np.float64)
    for name, destinations in allowed.items():
        source = state_index[name]
        for destination in destinations:
            exit_probability[source, state_index[destination]] = 0.5
        exit_probability[source] += boundary[source]
        exit_probability[source, source] = 0.0
        exit_probability[source] /= exit_probability[source].sum()

    survival = np.empty((count, max_age), dtype=np.float64)
    for state, state_lengths in enumerate(lengths):
        sample = np.asarray(state_lengths, dtype=np.int64)
        for age in range(max_age):
            at_risk = np.count_nonzero(sample > age)
            survives = np.count_nonzero(sample > age + 1)
            survival[state, age] = (survives + 0.5) / (at_risk + 1.0)
        survival[state] = np.clip(survival[state], 0.01, 0.99)
    for absorbing in ("L", "S"):
        survival[state_index[absorbing]] = 0.995

    model = HSMMModel(
        states=states,
        means=means,
        variances=variances,
        initial=initial_count / initial_count.sum(),
        survival=survival,
        exit_probability=exit_probability,
        max_age=max_age,
    )
    model.validate()
    return model
