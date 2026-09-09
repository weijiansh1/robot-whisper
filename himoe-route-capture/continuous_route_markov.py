"""Continuous-state Markov models for intra-query MoE routing.

The state at denoise round ``tau`` is the concatenated square-root router
probability tensor over every captured HB layer, action token and expert.  This
is a Hellinger embedding of the complete routing state.  A class-conditional,
coordinate-factorized linear-Gaussian AR(1) model estimates

    p(X[tau + 1] | X[tau], outcome)

without quantization or random codebooks.  Each coordinate has its own slope
and intercept.  Residual variances are shared between outcome classes, making
the likelihood ratio depend on conditional means rather than class-specific
variance collapse.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from analyze_intraquery_markov import (
    HB_LAYERS,
    N_ACTION_TOKENS,
    N_DENOISE,
    N_EXPERTS,
    normalize_router_probabilities,
)


METHODS = (
    "occupancy_periodic",
    "markov_periodic",
    "occupancy_position",
    "markov_position",
)


@dataclass(frozen=True)
class ContinuousRouteMarkovModel:
    occupancy_periodic_mean: np.ndarray  # [feature]
    occupancy_periodic_variance: np.ndarray  # [feature], shared across classes
    occupancy_position_mean: np.ndarray  # [denoise, feature]
    occupancy_position_variance: np.ndarray  # [denoise, feature], shared
    transition_periodic_slope: np.ndarray  # [feature]
    transition_periodic_intercept: np.ndarray  # [feature]
    transition_periodic_variance: np.ndarray  # [feature], shared
    transition_position_slope: np.ndarray  # [denoise - 1, feature]
    transition_position_intercept: np.ndarray  # [denoise - 1, feature]
    transition_position_variance: np.ndarray  # [denoise - 1, feature], shared


def continuous_route_states(routes: np.ndarray) -> np.ndarray:
    """Return Hellinger states shaped ``[episode, denoise, feature]``."""

    probabilities = normalize_router_probabilities(routes)
    expected = (
        len(probabilities),
        len(HB_LAYERS),
        N_DENOISE,
        N_ACTION_TOKENS,
        N_EXPERTS,
    )
    if probabilities.shape != expected:
        raise ValueError("unexpected route tensor %s" % (probabilities.shape,))
    return (
        np.sqrt(probabilities)
        .transpose(0, 2, 1, 3, 4)
        .reshape(len(probabilities), N_DENOISE, -1)
    )


def _class_means(values: np.ndarray, labels: np.ndarray) -> dict[bool, np.ndarray]:
    means = {}
    for label in (False, True):
        selected = values[labels == label]
        if len(selected) < 2:
            raise ValueError("each outcome class needs at least two sequences")
        means[label] = selected.mean(axis=0)
    return means


def _shared_variance(
    values: np.ndarray,
    labels: np.ndarray,
    means: dict[bool, np.ndarray],
    prior_strength: float,
) -> np.ndarray:
    if prior_strength <= 0.0:
        raise ValueError("variance prior strength must be positive")
    residual = np.empty_like(values, dtype=np.float64)
    for label in (False, True):
        selected = labels == label
        residual[selected] = values[selected] - means[label]
    residual_sum = np.square(residual).sum(axis=0)
    target = np.var(values, axis=0)
    variance = (residual_sum + prior_strength * target) / (len(values) + prior_strength)
    floor = np.maximum(target * 1e-4, 1e-8)
    return np.maximum(variance, floor)


def _fit_class_regressions(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    ridge_strength: float,
) -> tuple[dict[bool, np.ndarray], dict[bool, np.ndarray]]:
    if ridge_strength <= 0.0:
        raise ValueError("ridge strength must be positive")
    pooled_left_variance = np.var(left, axis=0)
    variance_floor = np.maximum(pooled_left_variance * 1e-6, 1e-10)
    slopes: dict[bool, np.ndarray] = {}
    intercepts: dict[bool, np.ndarray] = {}
    for label in (False, True):
        selected = labels == label
        x = left[selected]
        y = right[selected]
        if len(x) < 2:
            raise ValueError("each outcome class needs at least two transitions")
        x_mean = x.mean(axis=0)
        y_mean = y.mean(axis=0)
        x_centered = x - x_mean
        y_centered = y - y_mean
        covariance = np.sum(x_centered * y_centered, axis=0)
        energy = np.sum(np.square(x_centered), axis=0)
        prior_scale = np.maximum(pooled_left_variance, variance_floor)
        slope = covariance / (energy + ridge_strength * prior_scale)
        slopes[label] = slope
        intercepts[label] = y_mean - slope * x_mean
    return slopes, intercepts


def _shared_transition_variance(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    slopes: dict[bool, np.ndarray],
    intercepts: dict[bool, np.ndarray],
    prior_strength: float,
) -> np.ndarray:
    residual = np.empty_like(right, dtype=np.float64)
    for label in (False, True):
        selected = labels == label
        prediction = slopes[label] * left[selected] + intercepts[label]
        residual[selected] = right[selected] - prediction
    residual_sum = np.square(residual).sum(axis=0)
    target = np.var(right, axis=0)
    variance = (residual_sum + prior_strength * target) / (len(right) + prior_strength)
    floor = np.maximum(target * 1e-4, 1e-8)
    return np.maximum(variance, floor)


def fit_continuous_route_models(
    states: np.ndarray,
    success: np.ndarray,
    ridge_strength: float = 20.0,
    variance_prior_strength: float = 20.0,
) -> dict[bool, ContinuousRouteMarkovModel]:
    """Fit success/failure continuous Markov models with shared variances."""

    values = np.asarray(states, dtype=np.float64)
    labels = np.asarray(success, dtype=bool)
    if values.ndim != 3 or values.shape[1] != N_DENOISE:
        raise ValueError("expected continuous states [episode,10,feature]")
    if len(values) != len(labels) or len(np.unique(labels)) != 2:
        raise ValueError("states need aligned binary outcome labels")
    if not np.all(np.isfinite(values)):
        raise ValueError("continuous route states must be finite")

    features = values.shape[-1]
    periodic_values = values.reshape(-1, features)
    periodic_labels = np.repeat(labels, N_DENOISE)
    periodic_means = _class_means(periodic_values, periodic_labels)
    periodic_variance = _shared_variance(
        periodic_values,
        periodic_labels,
        periodic_means,
        variance_prior_strength,
    )

    position_means = {
        label: np.stack(
            [values[labels == label, tau].mean(axis=0) for tau in range(N_DENOISE)]
        )
        for label in (False, True)
    }
    position_variance = np.stack(
        [
            _shared_variance(
                values[:, tau],
                labels,
                {label: position_means[label][tau] for label in (False, True)},
                variance_prior_strength,
            )
            for tau in range(N_DENOISE)
        ]
    )

    periodic_left = values[:, :-1].reshape(-1, features)
    periodic_right = values[:, 1:].reshape(-1, features)
    transition_labels = np.repeat(labels, N_DENOISE - 1)
    periodic_slopes, periodic_intercepts = _fit_class_regressions(
        periodic_left,
        periodic_right,
        transition_labels,
        ridge_strength,
    )
    periodic_transition_variance = _shared_transition_variance(
        periodic_left,
        periodic_right,
        transition_labels,
        periodic_slopes,
        periodic_intercepts,
        variance_prior_strength,
    )

    position_slopes = {False: [], True: []}
    position_intercepts = {False: [], True: []}
    position_transition_variance = []
    for tau in range(N_DENOISE - 1):
        slopes, intercepts = _fit_class_regressions(
            values[:, tau], values[:, tau + 1], labels, ridge_strength
        )
        for label in (False, True):
            position_slopes[label].append(slopes[label])
            position_intercepts[label].append(intercepts[label])
        position_transition_variance.append(
            _shared_transition_variance(
                values[:, tau],
                values[:, tau + 1],
                labels,
                slopes,
                intercepts,
                variance_prior_strength,
            )
        )
    stacked_slopes = {
        label: np.stack(position_slopes[label]) for label in (False, True)
    }
    stacked_intercepts = {
        label: np.stack(position_intercepts[label]) for label in (False, True)
    }
    stacked_transition_variance = np.stack(position_transition_variance)

    return {
        label: ContinuousRouteMarkovModel(
            occupancy_periodic_mean=periodic_means[label],
            occupancy_periodic_variance=periodic_variance,
            occupancy_position_mean=position_means[label],
            occupancy_position_variance=position_variance,
            transition_periodic_slope=periodic_slopes[label],
            transition_periodic_intercept=periodic_intercepts[label],
            transition_periodic_variance=periodic_transition_variance,
            transition_position_slope=stacked_slopes[label],
            transition_position_intercept=stacked_intercepts[label],
            transition_position_variance=stacked_transition_variance,
        )
        for label in (False, True)
    }


def _diagonal_gaussian_log_probability(
    values: np.ndarray, mean: np.ndarray, variance: np.ndarray
) -> np.ndarray:
    return -0.5 * np.sum(
        math.log(2.0 * math.pi)
        + np.log(variance)
        + np.square(values - mean) / variance,
        axis=-1,
    )


def continuous_sequence_log_likelihoods(
    model: ContinuousRouteMarkovModel, states: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != N_DENOISE:
        raise ValueError("expected continuous states [episode,10,feature]")
    if values.shape[-1] != model.occupancy_periodic_mean.shape[-1]:
        raise ValueError("continuous state width differs from the model")

    periodic_occupancy = _diagonal_gaussian_log_probability(
        values,
        model.occupancy_periodic_mean[None, None, :],
        model.occupancy_periodic_variance[None, None, :],
    )
    position_occupancy = _diagonal_gaussian_log_probability(
        values,
        model.occupancy_position_mean[None, :, :],
        model.occupancy_position_variance[None, :, :],
    )

    periodic_markov = np.empty((len(values), N_DENOISE), dtype=np.float64)
    position_markov = np.empty_like(periodic_markov)
    periodic_markov[:, 0] = periodic_occupancy[:, 0]
    position_markov[:, 0] = position_occupancy[:, 0]
    periodic_prediction = (
        model.transition_periodic_slope[None, None, :] * values[:, :-1]
        + model.transition_periodic_intercept[None, None, :]
    )
    periodic_markov[:, 1:] = _diagonal_gaussian_log_probability(
        values[:, 1:],
        periodic_prediction,
        model.transition_periodic_variance[None, None, :],
    )
    position_prediction = (
        model.transition_position_slope[None, :, :] * values[:, :-1]
        + model.transition_position_intercept[None, :, :]
    )
    position_markov[:, 1:] = _diagonal_gaussian_log_probability(
        values[:, 1:],
        position_prediction,
        model.transition_position_variance[None, :, :],
    )
    return {
        "occupancy_periodic": np.cumsum(periodic_occupancy, axis=1),
        "markov_periodic": np.cumsum(periodic_markov, axis=1),
        "occupancy_position": np.cumsum(position_occupancy, axis=1),
        "markov_position": np.cumsum(position_markov, axis=1),
    }
