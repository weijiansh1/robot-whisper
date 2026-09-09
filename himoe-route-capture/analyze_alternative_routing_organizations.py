#!/usr/bin/env python3
"""Explore quotient and graph organizations of normalized MoE route paths.

The earlier analysis clustered a vectorized relative-phase trajectory.  This
script instead removes different nuisance coordinates before clustering:

* ``peer_rank`` keeps only deviation ranks against the other 31 rollouts with
  the same task and initial state;
* ``lag_spectrum`` forgets absolute phase and records the normalized routing
  variogram as a function of phase lag;
* ``path_signature`` keeps arc-length-normalized displacement, variation, and
  signed area of a task/initial-state residual route path;
* ``route_topology`` summarizes persistent homology and recurrence graphs of
  the within-episode routing distance matrix;
* ``layer_wave`` records where a residual routing change propagates through
  the eight MoE layers.

All representations and clustering decisions are label blind.  Outcomes are
opened only by the post-hoc audit inherited from the formal feature cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from collections import Counter
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ripser import ripser
from scipy.fft import dct
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from analyze_all_outcome_routing_clusters import (
    CACHE_ROOT,
    OUT_DIR as SOURCE_OUT_DIR,
    _load_feature_cache,
)
from analyze_failure_routing_clusters import association_summary


HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE / "analysis/alternative-routing-organizations"
FEATURE_CACHE = SOURCE_OUT_DIR / "feature_cache"
REPRESENTATIONS = (
    "peer_rank",
    "lag_spectrum",
    "path_signature",
    "route_topology",
    "layer_wave",
)
N_PHASE = 10
N_LAYER = 8
MIN_CLUSTER_SIZE = 32
MIN_SAMPLES = 16
LOUVAIN_NEIGHBORS = 25
REPRESENTATION_CACHE_SCHEMA = "himoe.alternative_route_features.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--feature-cache", type=pathlib.Path, default=FEATURE_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def robust_standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1, values.shape[-1])
    center = np.median(flat, axis=0)
    q25, q75 = np.quantile(flat, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = flat.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    keep = scale > 1e-9
    if not np.any(keep):
        raise ValueError("all representation channels are constant")
    standardized = (values[..., keep] - center[keep]) / scale[keep]
    return np.clip(standardized, -20.0, 20.0), {
        "input_dimensions": int(values.shape[-1]),
        "active_dimensions": int(keep.sum()),
        "scaling": "median_iqr_with_std_fallback",
    }


def group_codes(tasks: np.ndarray, initial_states: np.ndarray) -> np.ndarray:
    keys = np.asarray(
        [f"{task}::{int(initial)}" for task, initial in zip(tasks, initial_states)]
    )
    _unique, inverse = np.unique(keys, return_inverse=True)
    return inverse.astype(np.int32)


def sibling_centered(
    geometry: np.ndarray, tasks: np.ndarray, initial_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the label-blind median path for each task x initial-state cell."""
    values = np.asarray(geometry, dtype=np.float64)
    groups = group_codes(tasks, initial_states)
    residual = np.empty_like(values)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        if len(index) != 32:
            raise ValueError(f"expected 32 sibling rollouts, got {len(index)}")
        residual[index] = values[index] - np.median(values[index], axis=0)
    return residual, groups


def sibling_residuals(
    geometry: np.ndarray, tasks: np.ndarray, initial_states: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    residual, groups = sibling_centered(geometry, tasks, initial_states)
    standardized, _audit = robust_standardize(residual)
    return standardized, groups


def rank_within_groups(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranked = np.empty_like(values)
    for group in np.unique(groups):
        index = np.flatnonzero(groups == group)
        order = np.argsort(values[index], axis=0, kind="mergesort")
        local = np.empty_like(order, dtype=np.float64)
        for column in range(values.shape[1]):
            sorted_values = values[index, column][order[:, column]]
            first = 0
            while first < len(index):
                stop = first + 1
                while stop < len(index) and sorted_values[stop] == sorted_values[first]:
                    stop += 1
                local[order[first:stop, column], column] = 0.5 * (first + stop - 1)
                first = stop
        ranked[index] = (local + 0.5) / len(index)
    return ranked


def peer_rank_features(
    geometry: np.ndarray, tasks: np.ndarray, initial_states: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    residual, groups = sibling_residuals(geometry, tasks, initial_states)
    point = np.sqrt(np.mean(np.square(residual), axis=-1))
    velocity = np.diff(residual, axis=1)
    speed = np.sqrt(np.mean(np.square(velocity), axis=-1))
    acceleration = np.diff(velocity, axis=1)
    bend = np.sqrt(np.mean(np.square(acceleration), axis=-1))
    curves = [
        rank_within_groups(point, groups),
        rank_within_groups(speed, groups),
        rank_within_groups(bend, groups),
    ]
    # DCT separates persistent elevation from timing without expanding the view.
    transformed = [dct(curve, type=2, norm="ortho", axis=1) for curve in curves]
    matrix = np.concatenate(curves + transformed, axis=1).astype(np.float32)
    return matrix, {
        "dimensions": int(matrix.shape[1]),
        "quotient": "task_x_initial_state_sibling_median_and_within_cell_rank",
        "curves": ["route_deviation", "velocity_deviation", "bend_deviation"],
        "absolute_phase_retained": True,
        "outcome_used": False,
    }


def pair_layout(phases: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left, right = [], []
    for first in range(phases - 1):
        for second in range(first + 1, phases):
            left.append(first)
            right.append(second)
    left_array = np.asarray(left, dtype=np.int32)
    right_array = np.asarray(right, dtype=np.int32)
    return left_array, right_array, right_array - left_array


def recurrence_pairs(values: np.ndarray, phases: int = N_PHASE) -> np.ndarray:
    pairs = phases * (phases - 1) // 2
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % pairs:
        raise ValueError(f"recurrence matrix is incompatible with {phases} phases")
    return values.reshape(len(values), pairs, values.shape[1] // pairs)


def lag_spectrum_features(
    recurrence: np.ndarray, phases: int = N_PHASE
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs = recurrence_pairs(recurrence, phases)
    _left, _right, lag = pair_layout(phases)
    means, spreads = [], []
    for value in range(1, phases):
        selected = pairs[:, lag == value]
        means.append(selected.mean(axis=1))
        spreads.append(selected.std(axis=1))
    mean_curve = np.stack(means, axis=1)
    spread_curve = np.stack(spreads, axis=1)
    scale = np.maximum(mean_curve.mean(axis=1, keepdims=True), 1e-8)
    mean_curve /= scale
    spread_curve /= scale
    # The lag DCT is a stationary analogue of the phase-indexed trajectory.
    matrix = np.concatenate(
        [
            mean_curve.reshape(len(pairs), -1),
            spread_curve.reshape(len(pairs), -1),
            dct(mean_curve, type=2, norm="ortho", axis=1).reshape(len(pairs), -1),
        ],
        axis=1,
    ).astype(np.float32)
    return matrix, {
        "dimensions": int(matrix.shape[1]),
        "quotient": "per_channel_mean_distance",
        "absolute_phase_retained": False,
        "organization": "translation_invariant_route_variogram",
        "outcome_used": False,
    }


def _signature_one(path: np.ndarray) -> np.ndarray:
    increments = np.diff(path, axis=0)
    lengths = np.linalg.norm(increments, axis=1)
    arc = float(lengths.sum())
    if arc < 1e-10:
        return np.zeros(
            2 * path.shape[1]
            + path.shape[1] * (path.shape[1] - 1) // 2
            + (len(path) - 2)
            + (len(path) - 1)
            + 4,
            dtype=np.float64,
        )
    delta = increments / arc
    centered = (path[:-1] - path[0]) / arc
    first = delta.sum(axis=0)
    variation = np.abs(delta).sum(axis=0)
    area = []
    for left, right in combinations(range(path.shape[1]), 2):
        area.append(
            0.5
            * np.sum(
                centered[:, left] * delta[:, right]
                - centered[:, right] * delta[:, left]
            )
        )
    norms = np.linalg.norm(delta, axis=1)
    turn = []
    for index in range(len(delta) - 1):
        denominator = norms[index] * norms[index + 1]
        cosine = (
            float(delta[index] @ delta[index + 1]) / denominator
            if denominator > 1e-12
            else 1.0
        )
        turn.append(np.clip(cosine, -1.0, 1.0))
    chord = float(np.linalg.norm(first))
    radius = float(np.max(np.linalg.norm((path - path[0]) / arc, axis=1)))
    reversal = float(np.mean(np.asarray(turn) < 0.0)) if turn else 0.0
    concentration = float(np.sum(np.square(lengths / arc)))
    return np.concatenate(
        [
            first,
            variation,
            np.asarray(area),
            np.sort(np.asarray(turn)),
            np.sort(lengths / arc),
            np.asarray([chord, radius, reversal, concentration]),
        ]
    )


def path_signature_features(
    geometry: np.ndarray,
    tasks: np.ndarray,
    initial_states: np.ndarray,
    seed: int,
    latent_dimensions: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual, _groups = sibling_residuals(geometry, tasks, initial_states)
    flat = residual.reshape(-1, residual.shape[-1])
    components = min(latent_dimensions, flat.shape[1], len(flat) - 1)
    model = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    path = model.fit_transform(flat).reshape(len(residual), residual.shape[1], components)
    matrix = np.stack([_signature_one(value) for value in path]).astype(np.float32)
    return matrix, {
        "dimensions": int(matrix.shape[1]),
        "latent_dimensions": components,
        "latent_variance_explained": float(model.explained_variance_ratio_.sum()),
        "quotient": "task_x_initial_state_median_then_total_arc_length",
        "absolute_phase_retained": False,
        "organization": "truncated_path_signature_with_signed_levy_area",
        "outcome_used": False,
    }


def _distance_matrix(pair_values: np.ndarray, phases: int) -> np.ndarray:
    left, right, _lag = pair_layout(phases)
    matrix = np.zeros((phases, phases), dtype=np.float64)
    matrix[left, right] = pair_values
    matrix[right, left] = pair_values
    return matrix


def _topology_one(distance: np.ndarray) -> np.ndarray:
    phases = len(distance)
    off_diagonal = distance[np.triu_indices(phases, 1)]
    scale = float(np.median(off_diagonal))
    if scale < 1e-10:
        return np.zeros(9 + 15 + 14 + 9 + 6, dtype=np.float64)
    normalized = distance / scale
    diagram = ripser(normalized, distance_matrix=True, maxdim=1)["dgms"]
    h0_diagram = np.asarray(diagram[0])
    h1_diagram = np.asarray(diagram[1])
    h0 = np.sort(h0_diagram[np.isfinite(h0_diagram[:, 1]), 1])
    h0 = np.pad(h0[-9:], (max(0, 9 - len(h0)), 0))[-9:]
    h1 = h1_diagram
    if len(h1):
        finite = h1[np.isfinite(h1[:, 1])]
        lifetime = finite[:, 1] - finite[:, 0]
        order = np.argsort(lifetime)[::-1][:5]
        selected = finite[order]
        triples = np.column_stack(
            [selected[:, 0], selected[:, 1], selected[:, 1] - selected[:, 0]]
        ).reshape(-1)
    else:
        triples = np.empty(0, dtype=np.float64)
    triples = np.pad(triples, (0, max(0, 15 - len(triples))))[:15]
    thresholds = np.asarray([0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.50])
    betti = []
    for threshold in thresholds:
        components = 1 + int(np.sum(h0 > threshold))
        loops = int(
            np.sum(
                (h1_diagram[:, 0] <= threshold)
                & (h1_diagram[:, 1] > threshold)
            )
        ) if len(h1_diagram) else 0
        betti.extend([components / phases, loops / phases])

    adjacent = float(np.median(np.diag(normalized, 1)))
    graph_features = []
    for multiplier in (0.75, 1.00, 1.25):
        threshold = multiplier * max(adjacent, 1e-8)
        graph = nx.Graph()
        graph.add_nodes_from(range(phases))
        for left in range(phases - 1):
            graph.add_edge(left, left + 1)
        nonlocal_edges = []
        for left in range(phases - 2):
            for right in range(left + 2, phases):
                if normalized[left, right] <= threshold:
                    graph.add_edge(left, right)
                    nonlocal_edges.append((left, right))
        density = len(nonlocal_edges) / ((phases - 1) * (phases - 2) / 2)
        terminal = sum(right == phases - 1 for _left, right in nonlocal_edges) / max(
            phases - 2, 1
        )
        cycle_rank = graph.number_of_edges() - phases + nx.number_connected_components(graph)
        graph_features.extend([density, terminal, cycle_rank / phases])
    kernel = np.exp(-0.5 * np.square(normalized))
    degree = kernel.sum(axis=1)
    laplacian = np.eye(phases) - kernel / np.sqrt(np.outer(degree, degree))
    eigenvalues = np.linalg.eigvalsh(laplacian)
    spectral = eigenvalues[1:7]
    return np.concatenate([h0, triples, np.asarray(betti), graph_features, spectral])


def route_topology_features(
    recurrence: np.ndarray, phases: int = N_PHASE
) -> tuple[np.ndarray, dict[str, Any]]:
    pairs = recurrence_pairs(recurrence, phases)
    if pairs.shape[-1] != 24:
        raise ValueError("formal recurrence view must contain 24 channels per pair")
    soft = pairs[:, :, :8].mean(axis=-1)
    hard = pairs[:, :, 16:].mean(axis=-1)
    rows = []
    for soft_row, hard_row in zip(soft, hard):
        rows.append(
            np.concatenate(
                [
                    _topology_one(_distance_matrix(soft_row, phases)),
                    _topology_one(_distance_matrix(hard_row, phases)),
                ]
            )
        )
    matrix = np.stack(rows).astype(np.float32)
    return matrix, {
        "dimensions": int(matrix.shape[1]),
        "quotient": "median_pair_distance_separately_for_soft_and_hard_routes",
        "absolute_phase_retained": False,
        "organization": "persistent_homology_plus_multiscale_recurrence_graph",
        "outcome_used": False,
    }


def geometry_by_layer(geometry: np.ndarray) -> np.ndarray:
    values = np.asarray(geometry)
    if values.shape[-1] != 320:
        raise ValueError("formal query geometry must contain 320 features")
    blocks = []
    offset = 0
    for _ in range(8):
        blocks.append(values[..., offset : offset + 24].reshape(*values.shape[:-1], 8, 3))
        offset += 24
    for _ in range(4):
        blocks.append(values[..., offset : offset + 32].reshape(*values.shape[:-1], 8, 4))
        offset += 32
    if offset != values.shape[-1]:
        raise AssertionError("geometry parser did not consume all channels")
    return np.concatenate(blocks, axis=-1)


def layer_wave_features(
    geometry: np.ndarray, tasks: np.ndarray, initial_states: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    residual, _groups = sibling_centered(geometry, tasks, initial_states)
    layered = geometry_by_layer(residual)
    layered, _scaling = robust_standardize(layered)
    change = np.diff(layered, axis=1)
    speed = np.sqrt(np.mean(np.square(change), axis=-1))
    distribution = speed / np.maximum(speed.sum(axis=-1, keepdims=True), 1e-10)
    layer_axis = np.linspace(0.0, 1.0, N_LAYER)
    center = np.sum(distribution * layer_axis, axis=-1)
    spread = np.sqrt(
        np.sum(distribution * np.square(layer_axis[None, None, :] - center[..., None]), axis=-1)
    )
    entropy = -np.sum(distribution * np.log(np.maximum(distribution, 1e-12)), axis=-1)
    entropy /= math.log(N_LAYER)
    synchrony = np.empty(speed.shape[:2], dtype=np.float64)
    adjacent_sync = np.empty_like(synchrony)
    for episode in range(len(change)):
        for phase in range(change.shape[1]):
            vectors = change[episode, phase]
            norm = np.linalg.norm(vectors, axis=1)
            unit = vectors / np.maximum(norm[:, None], 1e-10)
            cosine = unit @ unit.T
            synchrony[episode, phase] = cosine[np.triu_indices(N_LAYER, 1)].mean()
            adjacent_sync[episode, phase] = np.diag(cosine, 1).mean()
    matrix = np.concatenate(
        [
            distribution.reshape(len(distribution), -1),
            center,
            spread,
            entropy,
            synchrony,
            adjacent_sync,
            dct(distribution, type=2, norm="ortho", axis=1).reshape(len(distribution), -1),
        ],
        axis=1,
    ).astype(np.float32)
    return matrix, {
        "dimensions": int(matrix.shape[1]),
        "quotient": "task_x_initial_state_median_and_per_transition_total_speed",
        "absolute_phase_retained": True,
        "organization": "phase_by_layer_route_change_wave",
        "outcome_used": False,
    }


def build_representations(
    geometry_flat: np.ndarray,
    recurrence: np.ndarray,
    tasks: np.ndarray,
    initial_states: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    geometry = np.asarray(geometry_flat).reshape(len(geometry_flat), N_PHASE, -1)
    builders = {
        "peer_rank": lambda: peer_rank_features(geometry, tasks, initial_states),
        "lag_spectrum": lambda: lag_spectrum_features(recurrence),
        "path_signature": lambda: path_signature_features(
            geometry, tasks, initial_states, seed
        ),
        "route_topology": lambda: route_topology_features(recurrence),
        "layer_wave": lambda: layer_wave_features(geometry, tasks, initial_states),
    }
    matrices, audit = {}, {}
    for name, builder in builders.items():
        matrix, details = builder()
        if len(matrix) != len(geometry) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"invalid {name} representation")
        matrices[name] = matrix
        audit[name] = details
    return matrices, audit


def prepare_embedding(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    standardized, scaling = robust_standardize(matrix)
    maximum = min(24, len(standardized) - 1, standardized.shape[1])
    model = PCA(n_components=maximum, svd_solver="randomized", random_state=seed)
    transformed = model.fit_transform(standardized)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    retained = min(maximum, max(2, int(np.searchsorted(cumulative, 0.90) + 1)))
    return transformed[:, :retained], {
        **scaling,
        "pca_components": retained,
        "variance_explained": float(cumulative[retained - 1]),
    }


def canonicalize(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    present = [int(value) for value in np.unique(labels) if value >= 0]
    ordered = sorted(present, key=lambda value: (float(score[labels == value].mean()), value))
    mapping = {value: index for index, value in enumerate(ordered)}
    return np.asarray([mapping.get(int(value), -1) for value in labels], dtype=np.int32)


def hdbscan_labels(embedding: np.ndarray, minimum: int, samples: int) -> tuple[np.ndarray, Any]:
    model = HDBSCAN(
        min_cluster_size=minimum,
        min_samples=samples,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        n_jobs=-1,
        copy=True,
    ).fit(embedding)
    return np.asarray(model.labels_, dtype=np.int32), model


def louvain_labels(
    embedding: np.ndarray, neighbors: int, seed: int, resolution: float = 1.0
) -> tuple[np.ndarray, float]:
    count = len(embedding)
    model = NearestNeighbors(n_neighbors=min(neighbors + 1, count)).fit(embedding)
    distance, index = model.kneighbors(embedding)
    positive = distance[:, 1:][distance[:, 1:] > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    graph = nx.Graph()
    graph.add_nodes_from(range(count))
    for source in range(count):
        for target, value in zip(index[source, 1:], distance[source, 1:]):
            weight = math.exp(-0.5 * (float(value) / scale) ** 2)
            if graph.has_edge(source, int(target)):
                graph[source][int(target)]["weight"] = max(
                    graph[source][int(target)]["weight"], weight
                )
            else:
                graph.add_edge(source, int(target), weight=weight)
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed
    )
    labels = np.empty(count, dtype=np.int32)
    for label, members in enumerate(communities):
        labels[np.fromiter(members, dtype=np.int32)] = label
    modularity = float(nx.community.modularity(graph, communities, weight="weight"))
    return labels, modularity


def label_agreement(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    both = (left >= 0) & (right >= 0)
    return {
        "ari_all_including_noise": float(adjusted_rand_score(left, right)),
        "ari_jointly_clustered": float(adjusted_rand_score(left[both], right[both]))
        if both.sum() >= 2
        else float("nan"),
        "jointly_clustered_fraction": float(both.mean()),
        "noise_agreement": float(np.mean((left < 0) == (right < 0))),
    }


def cluster_profile(
    labels: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
    lengths: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for label in np.unique(labels):
        index = labels == label
        counts = Counter(tasks[index].tolist())
        rows.append(
            {
                "cluster": "noise" if label < 0 else f"C{int(label)}",
                "n": int(index.sum()),
                "successes": int(np.sum(~failures[index])),
                "failures": int(np.sum(failures[index])),
                "failure_rate": float(failures[index].mean()),
                "task_count": len(counts),
                "dominant_task_fraction": float(max(counts.values()) / index.sum()),
                "tasks": dict(counts),
                "length_min": int(lengths[index].min()),
                "length_median": float(np.median(lengths[index])),
                "length_max": int(lengths[index].max()),
            }
        )
    return rows


def evaluate_labels(
    labels: np.ndarray,
    embedding: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
    initial_states: np.ndarray,
    lengths: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    clustered = labels >= 0
    unique = np.unique(labels[clustered])
    silhouette = float("nan")
    if len(unique) >= 2 and clustered.sum() > len(unique):
        rng = np.random.default_rng(seed)
        sample = np.flatnonzero(clustered)
        if len(sample) > 1200:
            sample = np.sort(rng.choice(sample, 1200, replace=False))
        silhouette = float(silhouette_score(embedding[sample], labels[sample]))
    strata = np.asarray(
        [f"{task}::{int(initial)}" for task, initial in zip(tasks, initial_states)]
    )
    # The shared contingency helper assumes nonnegative categorical labels.
    # Shift HDBSCAN's -1 noise label only for post-hoc association statistics.
    _categories, audit_labels = np.unique(labels, return_inverse=True)
    outcome = association_summary(
        failures.astype(str),
        audit_labels,
        strata,
        permutations,
        seed,
        "within_task_initial_state",
    )
    return {
        "clusters": int(len(unique)),
        "noise_fraction": float(np.mean(~clustered)),
        "noise_failure_rate": float(failures[~clustered].mean()) if np.any(~clustered) else None,
        "clustered_failure_rate": float(failures[clustered].mean()) if np.any(clustered) else None,
        "silhouette_clustered": silhouette,
        "outcome": outcome,
        "task_nmi": float(normalized_mutual_info_score(tasks, audit_labels)),
        "length_nmi": float(
            normalized_mutual_info_score(lengths.astype(str), audit_labels)
        ),
        "profiles": cluster_profile(labels, failures, tasks, lengths),
    }


def fit_view(
    matrix: np.ndarray,
    failures: np.ndarray,
    tasks: np.ndarray,
    initial_states: np.ndarray,
    lengths: np.ndarray,
    score: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, seed)
    hdb_raw, hdb_model = hdbscan_labels(embedding, MIN_CLUSTER_SIZE, MIN_SAMPLES)
    hdb = canonicalize(hdb_raw, score)
    louvain_raw, modularity = louvain_labels(embedding, LOUVAIN_NEIGHBORS, seed)
    louvain = canonicalize(louvain_raw, score)
    methods = {
        "hdbscan": evaluate_labels(
            hdb,
            embedding,
            failures,
            tasks,
            initial_states,
            lengths,
            permutations,
            seed + 10,
        ),
        "louvain": evaluate_labels(
            louvain,
            embedding,
            failures,
            tasks,
            initial_states,
            lengths,
            permutations,
            seed + 20,
        ),
    }
    persistence = getattr(hdb_model, "cluster_persistence_", None)
    methods["hdbscan"]["cluster_persistence"] = (
        [float(value) for value in persistence] if persistence is not None else None
    )
    methods["hdbscan"]["mean_membership_probability"] = float(
        np.mean(hdb_model.probabilities_[hdb_raw >= 0])
    ) if np.any(hdb_raw >= 0) else None
    methods["louvain"]["modularity"] = modularity

    hdb_sweeps = []
    for minimum, samples in ((16, 8), (32, 8), (32, 16), (64, 16), (64, 32)):
        candidate, _model = hdbscan_labels(embedding, minimum, samples)
        candidate = canonicalize(candidate, score)
        hdb_sweeps.append(
            {
                "min_cluster_size": minimum,
                "min_samples": samples,
                "clusters": int(len(np.unique(candidate[candidate >= 0]))),
                "noise_fraction": float(np.mean(candidate < 0)),
                "agreement_to_primary": label_agreement(hdb, candidate),
            }
        )
    louvain_sweeps = []
    for neighbors in (15, 25, 40):
        for resolution in (0.8, 1.0, 1.2):
            candidate, candidate_modularity = louvain_labels(
                embedding, neighbors, seed + neighbors, resolution
            )
            candidate = canonicalize(candidate, score)
            louvain_sweeps.append(
                {
                    "neighbors": neighbors,
                    "resolution": resolution,
                    "clusters": int(len(np.unique(candidate))),
                    "modularity": candidate_modularity,
                    "ari_to_primary": float(adjusted_rand_score(louvain, candidate)),
                }
            )
    return {
        "preprocessing": preprocessing,
        "methods": methods,
        "hdbscan_sensitivity": hdb_sweeps,
        "louvain_sensitivity": louvain_sweeps,
    }, {"hdbscan": hdb, "louvain": louvain}, embedding


def write_assignments(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = [
        "task",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "episode_length",
        "failure",
    ] + [f"{view}_{method}" for view in REPRESENTATIONS for method in ("hdbscan", "louvain")]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(arrays["meta_failure"])):
            row = {
                "task": str(arrays["meta_task"][index]),
                "episode": int(arrays["meta_episode"][index]),
                "init_state_id": int(arrays["meta_init_state_id"][index]),
                "flow_noise_seed": int(arrays["meta_flow_noise_seed"][index]),
                "episode_length": int(arrays["meta_episode_length"][index]),
                "failure": bool(arrays["meta_failure"][index]),
            }
            for view in REPRESENTATIONS:
                for method in ("hdbscan", "louvain"):
                    value = int(labels[view][method][index])
                    row[f"{view}_{method}"] = "noise" if value < 0 else f"C{value}"
            writer.writerow(row)


def make_plot(
    path: pathlib.Path,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, dict[str, np.ndarray]],
    failures: np.ndarray,
) -> None:
    figure, axes = plt.subplots(len(REPRESENTATIONS), 3, figsize=(15, 4 * len(REPRESENTATIONS)))
    for row, view in enumerate(REPRESENTATIONS):
        embedding = embeddings[view]
        if embedding.shape[1] < 2:
            xy = np.column_stack([embedding[:, 0], np.zeros(len(embedding))])
        else:
            xy = embedding[:, :2]
        axes[row, 0].scatter(xy[:, 0], xy[:, 1], c=failures, s=5, alpha=0.55, cmap="coolwarm")
        axes[row, 0].set_title(f"{view}: outcome (post-hoc)")
        for column, method in enumerate(("hdbscan", "louvain"), start=1):
            value = labels[view][method]
            axes[row, column].scatter(xy[:, 0], xy[:, 1], c=value, s=5, alpha=0.55, cmap="tab20")
            axes[row, column].set_title(f"{view}: {method}")
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Alternative MoE Routing Organizations",
        "",
        "This is an exploratory, label-blind comparison. Outcome labels were used only after every representation and clustering fit.",
        "",
        "## Primary Results",
        "",
        "| representation | method | clusters | noise | silhouette | outcome NMI excess | p | task NMI | length NMI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in REPRESENTATIONS:
        for method in ("hdbscan", "louvain"):
            result = summary["views"][view]["fit"]["methods"][method]
            lines.append(
                "| %s | %s | %d | %.3f | %.3f | %.3f | %.4f | %.3f | %.3f |"
                % (
                    view,
                    method,
                    result["clusters"],
                    result["noise_fraction"],
                    result["silhouette_clustered"],
                    result["outcome"]["nmi_excess_over_null"],
                    result["outcome"]["permutation_p_one_sided"],
                    result["task_nmi"],
                    result["length_nmi"],
                )
            )
    lines.extend(["", "## Block Composition", ""])
    for view in REPRESENTATIONS:
        for method in ("hdbscan", "louvain"):
            lines.extend(
                [
                    f"### {view} / {method}",
                    "",
                    "| block | n | success | failure | failure rate | tasks | dominant task | length median |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            profiles = summary["views"][view]["fit"]["methods"][method]["profiles"]
            for row in profiles:
                lines.append(
                    "| %s | %d | %d | %d | %.3f | %d | %.3f | %.1f |"
                    % (
                        row["cluster"],
                        row["n"],
                        row["successes"],
                        row["failures"],
                        row["failure_rate"],
                        row["task_count"],
                        row["dominant_task_fraction"],
                        row["length_median"],
                    )
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation Limits",
            "",
            "- These representations deliberately quotient different information and are not interchangeable measurements.",
            "- HDBSCAN noise is an explicit outlier category, not a discovered failure class.",
            "- Relative phase and arc-length normalization remove numeric duration but cannot make a timeout endpoint semantically equal to successful completion.",
            "- Outcome association remains post-hoc and exploratory; the five-task corpus cannot establish unseen-task generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def load_or_build_representations(
    output: pathlib.Path,
    feature_cache: pathlib.Path,
    arrays: dict[str, np.ndarray],
    tasks: np.ndarray,
    initial_states: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    cache_path = output / "representation_features.npz"
    audit_path = output / "representation_features.json"
    source_hash = sha256_file(feature_cache / "manifest.json")
    if cache_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if (
            audit.get("schema") == REPRESENTATION_CACHE_SCHEMA
            and audit.get("source_manifest_sha256") == source_hash
            and int(audit.get("episodes", -1)) == len(tasks)
        ):
            with np.load(cache_path, allow_pickle=False) as archive:
                matrices = {name: np.asarray(archive[name]) for name in REPRESENTATIONS}
            if all(len(value) == len(tasks) for value in matrices.values()):
                return matrices, audit["representations"]
    matrices, representation_audit = build_representations(
        np.asarray(arrays["feature_primary_geometry"]),
        np.asarray(arrays["feature_primary_recurrence"]),
        tasks,
        initial_states,
        seed,
    )
    np.savez_compressed(cache_path, **matrices)
    audit_path.write_text(
        json.dumps(
            {
                "schema": REPRESENTATION_CACHE_SCHEMA,
                "source_manifest_sha256": source_hash,
                "episodes": len(tasks),
                "representations": representation_audit,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return matrices, representation_audit


def self_test() -> None:
    rng = np.random.default_rng(3)
    tasks = np.repeat(np.asarray(["a", "b"]), 64)
    initial = np.tile(np.repeat(np.arange(2), 32), 2)
    geometry = rng.normal(size=(128, 10, 320)).astype(np.float32)
    residual, groups = sibling_residuals(geometry, tasks, initial)
    assert residual.shape == geometry.shape and len(np.unique(groups)) == 4
    peer, _ = peer_rank_features(geometry, tasks, initial)
    assert peer.shape == (128, 54) and np.all(np.isfinite(peer))
    pairs = rng.uniform(0.01, 0.3, size=(128, 45, 24)).astype(np.float32)
    recurrence = pairs.reshape(128, -1)
    lag, _ = lag_spectrum_features(recurrence)
    topology, _ = route_topology_features(recurrence[:4])
    signature, _ = path_signature_features(geometry, tasks, initial, 4)
    wave, _ = layer_wave_features(geometry, tasks, initial)
    assert lag.shape == (128, 648)
    assert topology.shape[0] == 4
    assert signature.shape == (128, 65)
    assert wave.shape[0] == 128
    for matrix in (lag, topology, signature, wave):
        assert np.all(np.isfinite(matrix))
    embedding, _ = prepare_embedding(peer, 2)
    hdb, _ = hdbscan_labels(embedding, 16, 8)
    graph, _ = louvain_labels(embedding, 10, 2)
    assert len(hdb) == len(graph) == 128
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, audit = _load_feature_cache(args.feature_cache, args.cache_root)
    tasks = np.asarray(arrays["meta_task"])
    initial_states = np.asarray(arrays["meta_init_state_id"])
    failures = np.asarray(arrays["meta_failure"], dtype=bool)
    lengths = np.asarray(arrays["meta_episode_length"], dtype=np.int32)
    score = np.asarray(arrays["diagnostic_mean_soft_speed"], dtype=np.float64)
    matrices, representation_audit = load_or_build_representations(
        args.out_dir,
        args.feature_cache,
        arrays,
        tasks,
        initial_states,
        args.seed,
    )
    summary: dict[str, Any] = {
        "schema": "himoe.alternative_routing_organizations.v1",
        "scope": {
            "episodes": int(len(failures)),
            "successes": int(np.sum(~failures)),
            "failures": int(np.sum(failures)),
            "relative_phase": [0.5, 1.0],
            "absolute_control_step_used": False,
            "outcome_used_for_representation_or_fit": False,
        },
        "data_audit": audit,
        "clustering": {
            "hdbscan": {"min_cluster_size": MIN_CLUSTER_SIZE, "min_samples": MIN_SAMPLES},
            "louvain": {"neighbors": LOUVAIN_NEIGHBORS, "resolution": 1.0},
            "permutations": args.permutations,
            "outcome_permutation_strata": "task_x_initial_state",
        },
        "views": {},
    }
    all_labels: dict[str, dict[str, np.ndarray]] = {}
    embeddings: dict[str, np.ndarray] = {}
    for index, view in enumerate(REPRESENTATIONS):
        print(f"fitting {view}", flush=True)
        fit, view_labels, embedding = fit_view(
            matrices[view],
            failures,
            tasks,
            initial_states,
            lengths,
            score,
            args.permutations,
            args.seed + 1000 * index,
        )
        summary["views"][view] = {
            "representation": representation_audit[view],
            "fit": fit,
        }
        all_labels[view] = view_labels
        embeddings[view] = embedding

    write_assignments(args.out_dir / "assignments.csv", arrays, all_labels)
    np.savez_compressed(
        args.out_dir / "embeddings_and_labels.npz",
        **{
            **{f"embedding_{name}": value for name, value in embeddings.items()},
            **{
                f"labels_{view}_{method}": values
                for view, methods in all_labels.items()
                for method, values in methods.items()
            },
        },
    )
    make_plot(args.out_dir / "embeddings.png", embeddings, all_labels, failures)
    (args.out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
