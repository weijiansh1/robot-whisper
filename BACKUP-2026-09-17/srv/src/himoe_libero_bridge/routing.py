"""Routing-agreement metrics and candidate selectors."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from himoe_libero_bridge.protocol import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    HB_MOE_EXPERTS,
    ROUTING_TRACE_SHAPE,
)

ROUTING_METRIC_SCHEMA = "himoe-libero-routing-metrics-v2"
PRIMARY_ROUTING_METRIC = (
    "all_flow_aligned_layer_action_normalized_expert_distribution_mean_total_variation_medoid"
)
POOLED_WEIGHTED_JACCARD_ABLATION_METRIC = (
    "all_flow_pooled_layer_expert_weighted_jaccard_medoid_ablation"
)
ROUTING_ABLATION_METRICS = (POOLED_WEIGHTED_JACCARD_ABLATION_METRIC,)
PRIMARY_ACTION_METRIC = "checkpoint_std_normalized_rms_medoid"
SELECTOR_TIE_BREAK = "lowest_candidate_index_within_1e-12"
SELECTOR_TIE_ATOL = 1e-12
SELECTOR_TIE_RTOL = 0.0


def routing_metric_contract(
    posthoc_selection_allowed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return the frozen routing/action metric identity embedded in artifacts."""

    contract = {
        "name": ROUTING_METRIC_SCHEMA,
        "primary_routing_metric": PRIMARY_ROUTING_METRIC,
        "primary_action_metric": PRIMARY_ACTION_METRIC,
        "routing_ablation_metrics": list(ROUTING_ABLATION_METRICS),
        "routing_weight_normalization": "l1_per_flow_layer_action_site",
        "primary_routing_axes_preserved": ["flow", "layer", "action"],
        "selector_tie_break": SELECTOR_TIE_BREAK,
    }
    if posthoc_selection_allowed is not None:
        contract["posthoc_selection_allowed"] = bool(posthoc_selection_allowed)
    return contract


def sparse_routes_to_dense(
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    expert_count: int = HB_MOE_EXPERTS,
) -> np.ndarray:
    ids = np.asarray(expert_ids)
    weights = np.asarray(expert_weights, dtype=np.float64)
    if ids.shape != weights.shape or ids.ndim < 2:
        raise ValueError("expert_ids and expert_weights must have the same rank-2+ shape")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("expert_ids must be integers")
    if np.any(ids < 0) or np.any(ids >= expert_count):
        raise ValueError("expert_ids contains an out-of-range expert")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("expert_ids must be unique within every top-k routing site")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("expert_weights must be finite and non-negative")
    dense = np.zeros(ids.shape[:-1] + (expert_count,), dtype=np.float64)
    np.put_along_axis(dense, ids.astype(np.int64, copy=False), weights, axis=-1)
    return dense


def route_features(
    expert_ids: np.ndarray, expert_weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(expert_ids)
    weights = np.asarray(expert_weights)
    expected_tail = ROUTING_TRACE_SHAPE
    if ids.ndim != 5 or ids.shape[1:] != expected_tail:
        raise ValueError(
            "routing pool shape must be [candidate, %s], got %s"
            % (expected_tail, ids.shape)
        )
    dense = normalized_route_distributions(ids, weights)
    aligned = dense.reshape(dense.shape[0], -1)
    # This pooled occupancy representation is retained only for the WJ ablation.
    pooled = dense.mean(axis=(1, 3)).reshape(dense.shape[0], -1)
    return pooled, aligned


def normalized_route_distributions(
    expert_ids: np.ndarray, expert_weights: np.ndarray
) -> np.ndarray:
    """Reconstruct normalized expert probabilities at every aligned route site."""

    dense = sparse_routes_to_dense(expert_ids, expert_weights)
    mass = dense.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("every routing site must have positive expert-weight mass")
    return dense / mass


def aligned_total_variation_matrix(
    expert_ids: np.ndarray, expert_weights: np.ndarray
) -> np.ndarray:
    """Mean TV distance preserving the `[flow, layer, action]` alignment."""

    ids = np.asarray(expert_ids)
    weights = np.asarray(expert_weights)
    if (
        ids.ndim != 5
        or ids.shape[0] < 1
        or ids.shape[1:] != ROUTING_TRACE_SHAPE
        or weights.shape != ids.shape
    ):
        raise ValueError(
            "routing pool shape must be [candidate, %s], got %s"
            % (ROUTING_TRACE_SHAPE, ids.shape)
        )
    distributions = normalized_route_distributions(ids, weights)
    site_tv = 0.5 * np.abs(
        distributions[:, None, ...] - distributions[None, :, ...]
    ).sum(axis=-1)
    return np.clip(site_tv.mean(axis=(2, 3, 4)), 0.0, 1.0)


def weighted_jaccard_matrix(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("features must have shape [candidate, feature]")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("features must be finite and non-negative")
    numerator = np.minimum(values[:, None, :], values[None, :, :]).sum(axis=-1)
    denominator = np.maximum(values[:, None, :], values[None, :, :]).sum(axis=-1)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0.0,
    )
    return np.clip(similarity, 0.0, 1.0)


def _validated_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (ACTION_CHUNK_STEPS, ACTION_DIM):
        raise ValueError(
            "actions must have shape [candidate, %d, %d], got %s"
            % (ACTION_CHUNK_STEPS, ACTION_DIM, values.shape)
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must be finite")
    return values


def _action_rms_distance(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    delta = (values[:, None, :, :] - values[None, :, :, :]) / scale
    return np.sqrt(np.mean(np.square(delta), axis=(2, 3)))


def checkpoint_normalized_action_distance_matrix(
    actions: np.ndarray, action_std: np.ndarray
) -> np.ndarray:
    """Pairwise RMS in the checkpoint's normalized seven-dimensional action space."""

    values = _validated_actions(actions)
    scale = np.asarray(action_std, dtype=np.float64)
    if scale.shape != (ACTION_DIM,):
        raise ValueError("action_std must have shape (%d,), got %s" % (ACTION_DIM, scale.shape))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("action_std must be finite and strictly positive")
    return _action_rms_distance(values, scale[None, None, :])


def pool_std_action_distance_matrix(actions: np.ndarray) -> np.ndarray:
    """Exploratory ablation using candidate-pool scale; not the primary metric."""

    values = _validated_actions(actions)
    scale = np.std(values.reshape(-1, ACTION_DIM), axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return _action_rms_distance(values, scale[None, None, :])


def raw_action_distance_matrix(actions: np.ndarray) -> np.ndarray:
    """Exploratory unscaled RMS ablation in environment action units."""

    values = _validated_actions(actions)
    return _action_rms_distance(values, np.ones((1, 1, ACTION_DIM), dtype=np.float64))


def standardized_action_distance_matrix(actions: np.ndarray) -> np.ndarray:
    """Backward-compatible name for the pool-std ablation."""

    return pool_std_action_distance_matrix(actions)


def _score_index(scores: np.ndarray, maximize: bool) -> int:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError("selector scores must be a finite non-empty vector")
    optimum = np.max(values) if maximize else np.min(values)
    ties = np.flatnonzero(
        np.isclose(values, optimum, atol=SELECTOR_TIE_ATOL, rtol=SELECTOR_TIE_RTOL)
    )
    return int(ties[0])


def route_medoid(similarity: np.ndarray) -> Tuple[int, np.ndarray]:
    matrix = _square_matrix(similarity, "similarity")
    if matrix.shape[0] == 1:
        return 0, np.ones((1,), dtype=np.float64)
    scores = (matrix.sum(axis=1) - np.diag(matrix)) / (matrix.shape[0] - 1)
    return _score_index(scores, maximize=True), scores


def rad_knn(similarity: np.ndarray, neighbors: int) -> Tuple[int, np.ndarray]:
    matrix = _square_matrix(similarity, "similarity")
    candidate_count = matrix.shape[0]
    if not 1 <= neighbors < candidate_count:
        raise ValueError("neighbors must be in [1, candidate_count - 1]")
    scores = np.empty((candidate_count,), dtype=np.float64)
    for index in range(candidate_count):
        peers = np.delete(matrix[index], index)
        scores[index] = np.sort(peers)[-neighbors:].mean()

    return _score_index(scores, maximize=True), scores


def action_medoid(distance: np.ndarray) -> Tuple[int, np.ndarray]:
    matrix = _square_matrix(distance, "distance")
    scores = matrix.sum(axis=1)
    return _score_index(scores, maximize=False), scores


def analyze_candidate_pool(
    actions: np.ndarray,
    expert_ids: np.ndarray,
    expert_weights: np.ndarray,
    action_std: Optional[np.ndarray] = None,
    rad_neighbors: int = 2,
) -> Dict[str, Any]:
    actions = np.asarray(actions, dtype=np.float32)
    expert_ids = np.asarray(expert_ids)
    expert_weights = np.asarray(expert_weights, dtype=np.float32)
    if actions.shape[0] < 2:
        raise ValueError("candidate analysis requires at least two candidates")
    if expert_ids.shape[0] != actions.shape[0] or expert_weights.shape != expert_ids.shape:
        raise ValueError("actions and routing traces must have the same candidate count")
    if action_std is None:
        raise ValueError(
            "action_std is required for the primary checkpoint-normalized action metric"
        )

    pooled_features, aligned_features = route_features(expert_ids, expert_weights)
    aligned_total_variation = aligned_total_variation_matrix(expert_ids, expert_weights)
    pooled_similarity = weighted_jaccard_matrix(pooled_features)
    aligned_similarity = weighted_jaccard_matrix(aligned_features)
    action_distance = checkpoint_normalized_action_distance_matrix(actions, action_std)
    pool_std_action_distance = pool_std_action_distance_matrix(actions)
    raw_action_rms = raw_action_distance_matrix(actions)
    raw_action_delta = actions[:, None, :, :] - actions[None, :, :, :]
    raw_action_max_abs = np.max(np.abs(raw_action_delta), axis=(2, 3))

    route_medoid_index, route_medoid_scores = action_medoid(aligned_total_variation)
    pooled_route_medoid_index, pooled_route_medoid_scores = route_medoid(
        pooled_similarity
    )
    rad_index, rad_scores = rad_knn(pooled_similarity, rad_neighbors)
    aligned_route_medoid_index, aligned_route_medoid_scores = route_medoid(
        aligned_similarity
    )
    aligned_rad_index, aligned_rad_scores = rad_knn(aligned_similarity, rad_neighbors)
    action_medoid_index, action_medoid_scores = action_medoid(action_distance)
    pool_std_action_medoid_index, pool_std_action_medoid_scores = action_medoid(
        pool_std_action_distance
    )
    raw_action_medoid_index, raw_action_medoid_scores = action_medoid(raw_action_rms)
    route_pairs = _upper_triangle(1.0 - pooled_similarity)
    aligned_route_pairs = _upper_triangle(1.0 - aligned_similarity)
    action_pairs = _upper_triangle(action_distance)

    return {
        "metric_schema": routing_metric_contract(),
        "candidate_count": int(actions.shape[0]),
        "rad_neighbors": int(rad_neighbors),
        "selectors": {
            "first": 0,
            "action_medoid": action_medoid_index,
            "route_medoid": route_medoid_index,
            "pooled_wj_rad_knn_ablation": rad_index,
            "pooled_route_medoid_ablation": pooled_route_medoid_index,
            "pool_std_action_medoid_ablation": pool_std_action_medoid_index,
            "raw_action_medoid_ablation": raw_action_medoid_index,
            "aligned_wj_route_medoid_diagnostic": aligned_route_medoid_index,
            "aligned_wj_rad_knn_diagnostic": aligned_rad_index,
        },
        "selector_scores": {
            "action_medoid_checkpoint_normalized_rms_sum": action_medoid_scores.tolist(),
            "route_medoid_aligned_mean_total_variation_sum": (
                route_medoid_scores.tolist()
            ),
            "pooled_wj_rad_knn_ablation_density": rad_scores.tolist(),
            "pooled_route_medoid_ablation_mean_similarity": (
                pooled_route_medoid_scores.tolist()
            ),
            "pool_std_action_medoid_ablation_distance_sum": (
                pool_std_action_medoid_scores.tolist()
            ),
            "raw_action_medoid_ablation_distance_sum": raw_action_medoid_scores.tolist(),
            "aligned_wj_route_medoid_diagnostic_mean_similarity": (
                aligned_route_medoid_scores.tolist()
            ),
            "aligned_wj_rad_knn_diagnostic_density": aligned_rad_scores.tolist(),
        },
        "aligned_mean_total_variation": aligned_total_variation.tolist(),
        "pooled_weighted_jaccard_ablation": pooled_similarity.tolist(),
        "aligned_weighted_jaccard_diagnostic": aligned_similarity.tolist(),
        "checkpoint_normalized_action_rms": action_distance.tolist(),
        "pool_std_action_rms_ablation": pool_std_action_distance.tolist(),
        "raw_action_rms_ablation": raw_action_rms.tolist(),
        "routing": {
            "primary_feature_space": (
                "flow_layer_action_normalized_expert_distribution"
            ),
            "pooled_ablation_feature_space": "layer_expert_mean_over_flow_action",
            "unique_expert_id_traces": int(
                np.unique(expert_ids.reshape(expert_ids.shape[0], -1), axis=0).shape[0]
            ),
            "unique_weighted_traces": int(
                np.unique(
                    np.concatenate(
                        [
                            expert_ids.reshape(expert_ids.shape[0], -1).astype(np.float64),
                            np.round(expert_weights.reshape(expert_weights.shape[0], -1), 6),
                        ],
                        axis=1,
                    ),
                    axis=0,
                ).shape[0]
            ),
            "aligned_mean_tv_off_diagonal": _stats(
                _upper_triangle(aligned_total_variation)
            ),
            "pooled_wj_ablation_off_diagonal": _stats(
                _upper_triangle(pooled_similarity)
            ),
            "aligned_wj_diagnostic_off_diagonal": _stats(
                _upper_triangle(aligned_similarity)
            ),
            "candidate_dependent": bool(
                np.unique(expert_ids.reshape(expert_ids.shape[0], -1), axis=0).shape[0] > 1
                or not np.allclose(expert_weights, expert_weights[0:1], atol=1e-7, rtol=1e-7)
            ),
        },
        "actions": {
            "checkpoint_action_std": np.asarray(action_std, dtype=np.float64).tolist(),
            "unique_chunks": int(
                np.unique(actions.reshape(actions.shape[0], -1), axis=0).shape[0]
            ),
            "pairwise_checkpoint_normalized_rms": _stats(action_pairs),
            "pairwise_pool_std_rms_ablation": _stats(
                _upper_triangle(pool_std_action_distance)
            ),
            "pairwise_raw_rms": _stats(_upper_triangle(raw_action_rms)),
            "pairwise_raw_max_abs": _stats(_upper_triangle(raw_action_max_abs)),
            "per_dimension_std": np.std(actions, axis=(0, 1)).tolist(),
            "candidate_dependent": bool(
                np.unique(actions.reshape(actions.shape[0], -1), axis=0).shape[0] > 1
            ),
        },
        "route_action_association": {
            "aligned_mean_tv_pearson": _correlation(
                _upper_triangle(aligned_total_variation), action_pairs
            ),
            "aligned_mean_tv_spearman": _correlation(
                _rankdata(_upper_triangle(aligned_total_variation)),
                _rankdata(action_pairs),
            ),
            "pooled_wj_ablation_distance_pearson": _correlation(
                route_pairs, action_pairs
            ),
            "pooled_wj_ablation_distance_spearman": _correlation(
                _rankdata(route_pairs), _rankdata(action_pairs)
            ),
            "aligned_wj_diagnostic_distance_pearson": _correlation(
                aligned_route_pairs, action_pairs
            ),
        },
    }


def _square_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError("%s must be a non-empty square matrix" % name)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("%s must be finite" % name)
    return matrix


def _upper_triangle(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix)[np.triu_indices(np.asarray(matrix).shape[0], k=1)]


def _stats(values: np.ndarray) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(array.shape, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
