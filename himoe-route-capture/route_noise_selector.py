"""Label-free noise selection from an early HiMoE routing prefix.

The selector operates on complete router softmax probabilities.  Every aligned
``[layer, denoise, action-token]`` site is independently L1-normalized, embedded
with the Hellinger map, and only then aggregated.  Expert IDs are therefore never
compared across layers.

This module intentionally contains no task outcome or action API.  It can rank a
same-observation candidate pool from initial noise and routing alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


N_EXPERTS = 32
LIVE_ACTION_DIMS = 7


def normalize_router_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return finite, non-negative, exactly L1-normalized router probabilities."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != N_EXPERTS:
        raise ValueError("router probabilities must end in 32 experts")
    if not np.all(np.isfinite(values)):
        raise ValueError("router probabilities must be finite")
    if np.any(values < -1e-7):
        raise ValueError("router probabilities contain a negative value")
    values = np.maximum(values, 0.0)
    mass = values.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every routing site must have positive probability mass")
    normalized = values / mass
    if not np.allclose(normalized.sum(axis=-1), 1.0, atol=2e-12, rtol=0.0):
        raise RuntimeError("router normalization failed")
    return normalized


def hellinger_embedding(probabilities: np.ndarray) -> np.ndarray:
    """Embed ``[candidate, ..., expert]`` so Euclidean distance is site-mean Hellinger."""

    normalized = normalize_router_probabilities(probabilities)
    if normalized.ndim < 3:
        raise ValueError("a candidate axis and at least one routing-site axis are required")
    site_count = int(np.prod(normalized.shape[1:-1]))
    return np.sqrt(normalized).reshape(len(normalized), -1) / np.sqrt(2.0 * site_count)


def pairwise_hellinger(probabilities: np.ndarray) -> np.ndarray:
    """Pairwise root-mean-square Hellinger distance over aligned routing sites."""

    embedded = hellinger_embedding(probabilities)
    squared_norm = np.einsum("nd,nd->n", embedded, embedded, optimize=True)
    distance2 = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (embedded @ embedded.T)
    result = np.sqrt(np.maximum(distance2, 0.0))
    np.fill_diagonal(result, 0.0)
    return result


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    """Pairwise RMS Euclidean distance for ``[candidate, ...]`` arrays."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 2 or len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a finite [candidate, ...] array")
    flat = array.reshape(len(array), -1)
    squared_norm = np.einsum("nd,nd->n", flat, flat, optimize=True)
    distance2 = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * (flat @ flat.T)
    ) / flat.shape[1]
    result = np.sqrt(np.maximum(distance2, 0.0))
    np.fill_diagonal(result, 0.0)
    return result


def centrality(distance: np.ndarray) -> np.ndarray:
    """Mean distance to the other candidates; low values are more central."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError("distance must be a square matrix with at least two rows")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < -1e-12):
        raise ValueError("distance must be finite and non-negative")
    if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=0.0):
        raise ValueError("distance must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("distance diagonal must be zero")
    return matrix.sum(axis=1) / (len(matrix) - 1)


def stable_argmin(scores: np.ndarray, candidate_ids: Iterable[int] | None = None) -> int:
    """Return the candidate ID with minimum score, resolving ties by candidate ID."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a non-empty finite vector")
    ids = (
        np.arange(len(values), dtype=np.int64)
        if candidate_ids is None
        else np.asarray(tuple(candidate_ids), dtype=np.int64)
    )
    if ids.shape != values.shape or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique and aligned with scores")
    optimum = values.min()
    tied = ids[np.isclose(values, optimum, atol=1e-12, rtol=0.0)]
    return int(tied.min())


def route_centrality_scores(route_prefix: np.ndarray) -> np.ndarray:
    """Score a same-observation candidate pool by early-route centrality."""

    return centrality(pairwise_hellinger(route_prefix))


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = normalize_router_probabilities(probabilities)
    return -np.sum(values * np.log(np.maximum(values, 1e-300)), axis=-1) / np.log(
        N_EXPERTS
    )


def _margin(probabilities: np.ndarray) -> np.ndarray:
    values = normalize_router_probabilities(probabilities)
    top2 = np.partition(values, -2, axis=-1)[..., -2:]
    return top2.max(axis=-1) - top2.min(axis=-1)


def early_route_descriptors(routes: np.ndarray, prefix_steps: int = 3) -> np.ndarray:
    """Build layer-local descriptors from ``[candidate, layer, flow, token, expert]``.

    The returned columns are, per layer: candidate-cloud centrality, consecutive
    flow-step motion, normalized entropy, and top-1/top-2 margin.  Layer outputs
    are concatenated as scalar descriptors; expert coordinates from distinct
    layers are never identified with one another.
    """

    values = normalize_router_probabilities(routes)
    if values.ndim != 5:
        raise ValueError("routes must have shape [candidate, layer, flow, token, expert]")
    candidates, layers, flow_steps, tokens, _ = values.shape
    if candidates < 2 or tokens < 1 or not 1 <= prefix_steps <= flow_steps:
        raise ValueError("invalid candidate/token count or prefix_steps")
    prefix = values[:, :, :prefix_steps]

    layer_centrality = np.empty((candidates, layers), dtype=np.float64)
    for layer in range(layers):
        layer_centrality[:, layer] = centrality(
            pairwise_hellinger(prefix[:, layer])
        )

    if prefix_steps == 1:
        motion = np.zeros((candidates, layers), dtype=np.float64)
    else:
        left = prefix[:, :, :-1]
        right = prefix[:, :, 1:]
        site_h2 = 0.5 * np.square(np.sqrt(left) - np.sqrt(right)).sum(axis=-1)
        motion = np.sqrt(site_h2.mean(axis=(2, 3)))

    entropy = _entropy(prefix).mean(axis=(2, 3))
    margin = _margin(prefix).mean(axis=(2, 3))
    global_centrality = centrality(pairwise_hellinger(prefix))[:, None]
    return np.concatenate(
        [global_centrality, layer_centrality, motion, entropy, margin], axis=1
    )


def late_route_centrality_target(routes: np.ndarray, prefix_steps: int = 3) -> np.ndarray:
    """Self-supervised target: late-route cloud centrality, converted to pool ranks."""

    values = normalize_router_probabilities(routes)
    if values.ndim != 5 or not 1 <= prefix_steps < values.shape[2]:
        raise ValueError("routes or prefix_steps do not leave a non-empty future")
    score = centrality(pairwise_hellinger(values[:, :, prefix_steps:]))
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(len(score), dtype=np.float64)
    return ranks / max(len(score) - 1, 1)


def pool_standardize(features: np.ndarray) -> np.ndarray:
    """Standardize candidate features inside one same-observation pool."""

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("features must be finite [candidate, feature]")
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, keepdims=True)
    return centered / np.maximum(scale, 1e-12)


def noise_descriptors(noise: np.ndarray) -> np.ndarray:
    """Deployment-available initial-noise features, emphasizing seven live dimensions."""

    values = np.asarray(noise, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < LIVE_ACTION_DIMS:
        raise ValueError("noise must have shape [candidate, action-token, >=7]")
    if not np.all(np.isfinite(values)):
        raise ValueError("noise must be finite")
    live = values[..., :LIVE_ACTION_DIMS]
    live_flat = live.reshape(len(live), -1)
    live_token_rms = np.sqrt(np.mean(np.square(live), axis=-1))
    full_token_rms = np.sqrt(np.mean(np.square(values), axis=-1))
    return np.concatenate([live_flat, live_token_rms, full_token_rms], axis=1)


@dataclass(frozen=True)
class RidgeRouteHead:
    """Small ridge head trained only against a future-routing pseudo-target."""

    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    alpha: float

    @classmethod
    def fit(
        cls, features: np.ndarray, target: np.ndarray, alpha: float
    ) -> "RidgeRouteHead":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(target, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
            raise ValueError("features and target are not aligned")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or alpha <= 0.0:
            raise ValueError("training values must be finite and alpha must be positive")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.maximum(scale, 1e-12)
        standardized = (x - mean) / scale
        design = np.column_stack([np.ones(len(x)), standardized])
        penalty = float(alpha) * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ y
        )
        return cls(coefficients, mean, scale, float(alpha))

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean):
            raise ValueError("prediction features have the wrong shape")
        if not np.all(np.isfinite(values)):
            raise ValueError("prediction features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        design = np.column_stack([np.ones(len(values)), standardized])
        return design @ self.coefficients


@dataclass(frozen=True)
class RouteNoiseController:
    """Loadable candidate-ranking controller for an early routing prefix."""

    head: RidgeRouteHead
    include_noise: bool
    prefix_steps: int = 3

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        method: str = "future_route_noise_head",
        prefix_steps: int = 3,
    ) -> "RouteNoiseController":
        if method not in {"future_route_head", "future_route_noise_head"}:
            raise ValueError("unknown route-noise head method")
        prefix = method + "__"
        with np.load(path, allow_pickle=False) as payload:
            required = {
                prefix + "coefficients",
                prefix + "feature_mean",
                prefix + "feature_scale",
                prefix + "alpha",
            }
            missing = required - set(payload.files)
            if missing:
                raise ValueError("head artifact is missing %s" % sorted(missing))
            head = RidgeRouteHead(
                coefficients=np.asarray(payload[prefix + "coefficients"], dtype=np.float64),
                feature_mean=np.asarray(payload[prefix + "feature_mean"], dtype=np.float64),
                feature_scale=np.asarray(payload[prefix + "feature_scale"], dtype=np.float64),
                alpha=float(payload[prefix + "alpha"]),
            )
        if head.coefficients.shape != (len(head.feature_mean) + 1,):
            raise ValueError("head artifact has inconsistent coefficient dimensions")
        if head.feature_scale.shape != head.feature_mean.shape or np.any(
            head.feature_scale <= 0.0
        ):
            raise ValueError("head artifact has invalid feature scaling")
        return cls(
            head=head,
            include_noise=method == "future_route_noise_head",
            prefix_steps=prefix_steps,
        )

    def features(self, routes: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        route = pool_standardize(
            early_route_descriptors(routes, prefix_steps=self.prefix_steps)
        )
        if not self.include_noise:
            return route
        if noise is None:
            raise ValueError("the route+noise head requires initial noise")
        if len(noise) != len(routes):
            raise ValueError("noise and routes have different candidate counts")
        return np.concatenate(
            [route, pool_standardize(noise_descriptors(noise))], axis=1
        )

    def score(self, routes: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
        """Predict late-route centrality; lower scores are preferred."""

        return self.head.predict(self.features(routes, noise))

    def select(
        self,
        routes: np.ndarray,
        noise: np.ndarray | None = None,
        candidate_ids: Iterable[int] | None = None,
    ) -> int:
        """Return the selected candidate ID with deterministic tie handling."""

        return stable_argmin(self.score(routes, noise), candidate_ids)
