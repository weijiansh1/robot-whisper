#!/usr/bin/env python3
"""Analyze normalized MoE routing as a symbolic state-transition grammar."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_all_outcome_routing_clusters import (
    EXPECTED_EPISODES,
    EXPECTED_FAILURES,
    EXPECTED_SUCCESSES,
    cluster_joint_view,
)
from analyze_failure_routing_clusters import (
    association_summary,
    canonicalize,
    prepare_embedding,
    ward_labels,
)


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE / "analysis/all-outcome-routing-clusters"
FEATURE_CACHE = SOURCE_DIR / "feature_cache"
OUT_DIR = HERE / "analysis/routing-state-grammar"

PRIMARY_STATE_COUNT = 8
STATE_COUNT_SENSITIVITY = (6, 10)
MIN_SUCCESS_STATE_CELL = 20
OUTCOME_EXCESS_NMI_MIN = 0.05
TASK_SHADOW_NMI = 0.50
TASK_SHADOW_FRACTION = 0.90
STATE_SEED_REFITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=pathlib.Path, default=FEATURE_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--state-count", type=int, default=PRIMARY_STATE_COUNT)
    parser.add_argument("--subsamples", type=int, default=50)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_feature_cache(cache_dir: pathlib.Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    if manifest.get("schema") != "himoe.all_outcome_routing_features.v1":
        raise ValueError("unexpected all-outcome feature-cache schema")
    arrays = {
        name: np.load(cache_dir / item["file"], mmap_mode="r", allow_pickle=False)
        for name, item in manifest["arrays"].items()
    }
    expected = manifest["audit"]
    if (
        int(expected["episodes"]) != EXPECTED_EPISODES
        or int(expected["successes"]) != EXPECTED_SUCCESSES
        or int(expected["failures"]) != EXPECTED_FAILURES
    ):
        raise ValueError("formal cohort drifted")
    return arrays, manifest


def phase_geometry(values: np.ndarray, anchors: int) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2 or matrix.shape[1] % anchors:
        raise ValueError(f"cannot reshape {matrix.shape} into {anchors} phase anchors")
    result = matrix.reshape(len(matrix), anchors, matrix.shape[1] // anchors)
    if np.any(~np.isfinite(result)):
        raise ValueError("non-finite phase geometry")
    return result


def robust_state_embedding(
    sequences: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    flat = np.asarray(sequences, dtype=np.float64).reshape(-1, sequences.shape[-1])
    keep = flat.std(axis=0) > 1e-9
    if not np.any(keep):
        raise ValueError("all route-state descriptors are constant")
    active = flat[:, keep]
    center = np.median(active, axis=0)
    q25, q75 = np.quantile(active, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = active.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = np.clip((active - center) / scale, -20.0, 20.0)
    maximum = min(30, len(standardized) - 1, standardized.shape[1])
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed)
    full = pca.fit_transform(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    retained = min(maximum, max(2, int(np.searchsorted(cumulative, 0.90) + 1)))
    return full[:, :retained], {
        "query_states": int(len(flat)),
        "raw_dimensions": int(flat.shape[1]),
        "active_dimensions": int(keep.sum()),
        "pca_components": int(retained),
        "variance_explained": float(cumulative[retained - 1]),
        "scaling": "median_iqr",
    }


def canonical_state_labels(labels: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(
        range(len(centers)),
        key=lambda value: tuple(float(x) for x in centers[value, : min(3, centers.shape[1])]),
    )
    mapping = np.empty(len(order), dtype=np.int32)
    mapping[np.asarray(order)] = np.arange(len(order), dtype=np.int32)
    return mapping[labels], centers[np.asarray(order)]


def discover_route_states(
    sequences: np.ndarray, state_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    embedding, preprocessing = robust_state_embedding(sequences, seed)
    model = KMeans(
        n_clusters=state_count,
        n_init=20,
        max_iter=400,
        random_state=seed,
        algorithm="lloyd",
    )
    raw = model.fit_predict(embedding)
    labels, centers = canonical_state_labels(raw, model.cluster_centers_)
    sequence_labels = labels.reshape(sequences.shape[:2])
    counts = np.bincount(labels, minlength=state_count)
    preprocessing.update(
        {
            "state_count": int(state_count),
            "state_counts": counts.tolist(),
            "inertia_per_query": float(model.inertia_ / len(labels)),
        }
    )
    return sequence_labels, embedding, {"preprocessing": preprocessing, "centers": centers}


def state_seed_stability(
    embedding: np.ndarray, reference: np.ndarray, state_count: int, seed: int
) -> dict[str, Any]:
    scores = []
    for draw in range(STATE_SEED_REFITS):
        candidate = KMeans(
            n_clusters=state_count,
            n_init=5,
            max_iter=400,
            random_state=seed + 100 + draw,
            algorithm="lloyd",
        ).fit_predict(embedding)
        scores.append(float(adjusted_rand_score(reference.reshape(-1), candidate)))
    return {
        "refits": STATE_SEED_REFITS,
        "ari_values": scores,
        "ari_median": float(np.median(scores)),
        "ari_min": float(np.min(scores)),
    }


def _entropy(values: np.ndarray, maximum: float) -> float:
    positive = values[values > 0]
    if len(positive) == 0 or maximum <= 1.0:
        return 0.0
    return float(-np.sum(positive * np.log2(positive)) / math.log2(maximum))


def grammar_features(
    sequences: np.ndarray, state_count: int
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], dict[str, slice]]:
    labels = np.asarray(sequences, dtype=np.int32)
    if labels.ndim != 2 or labels.shape[1] < 3:
        raise ValueError("state grammar needs [episode, phase>=3]")
    if labels.min() < 0 or labels.max() >= state_count:
        raise ValueError("state label outside vocabulary")
    episodes, anchors = labels.shape
    occupancy = np.zeros((episodes, state_count), dtype=np.float64)
    transition = np.zeros((episodes, state_count * state_count), dtype=np.float64)
    phase_occupancy = np.zeros((episodes, 3 * state_count), dtype=np.float64)
    endpoints = np.zeros((episodes, 2 * state_count), dtype=np.float64)
    returns = np.zeros((episodes, state_count * state_count), dtype=np.float64)
    motifs = np.zeros((episodes, 5), dtype=np.float64)
    scalars = np.zeros((episodes, 6), dtype=np.float64)

    zones = np.array_split(np.arange(anchors), 3)
    for episode, sequence in enumerate(labels):
        occupancy[episode] = np.bincount(sequence, minlength=state_count) / anchors
        code = sequence[:-1] * state_count + sequence[1:]
        transition[episode] = np.bincount(
            code, minlength=state_count * state_count
        ) / (anchors - 1)
        for zone_index, zone in enumerate(zones):
            start = zone_index * state_count
            phase_occupancy[episode, start : start + state_count] = (
                np.bincount(sequence[zone], minlength=state_count) / len(zone)
            )
        endpoints[episode, sequence[0]] = 1.0
        endpoints[episode, state_count + sequence[-1]] = 1.0

        first, middle, third = sequence[:-2], sequence[1:-1], sequence[2:]
        aaa = (first == middle) & (middle == third)
        aab = (first == middle) & (middle != third)
        abb = (first != middle) & (middle == third)
        aba = (first == third) & (first != middle)
        abc = (first != middle) & (middle != third) & (first != third)
        motifs[episode] = np.asarray(
            [aaa.sum(), aab.sum(), abb.sum(), aba.sum(), abc.sum()], dtype=np.float64
        ) / (anchors - 2)
        if np.any(aba):
            return_code = first[aba] * state_count + middle[aba]
            returns[episode] = np.bincount(
                return_code, minlength=state_count * state_count
            ) / (anchors - 2)

        runs = [1]
        for position in range(1, anchors):
            if sequence[position] == sequence[position - 1]:
                runs[-1] += 1
            else:
                runs.append(1)
        revisits = 0
        for position in range(2, anchors):
            if sequence[position] != sequence[position - 1] and sequence[position] in sequence[: position - 1]:
                revisits += 1
        scalars[episode] = (
            _entropy(occupancy[episode], state_count),
            _entropy(transition[episode], state_count * state_count),
            np.mean(sequence[1:] != sequence[:-1]),
            len(np.unique(sequence)) / anchors,
            revisits / (anchors - 2),
            max(runs) / anchors,
        )

    blocks = [
        ("occupancy", occupancy),
        ("transition", transition),
        ("phase_occupancy", phase_occupancy),
        ("endpoints", endpoints),
        ("return_motif", returns),
        ("motif_class", motifs),
        ("scalar", scalars),
    ]
    names = []
    slices = {}
    cursor = 0
    for block, values in blocks:
        slices[block] = slice(cursor, cursor + values.shape[1])
        cursor += values.shape[1]
        names.extend(f"{block}_{index}" for index in range(values.shape[1]))
    matrix = np.concatenate([values for _block, values in blocks], axis=1)
    diagnostics = {
        "state_entropy": scalars[:, 0],
        "transition_entropy": scalars[:, 1],
        "switch_rate": scalars[:, 2],
        "unique_state_fraction": np.asarray(
            [len(np.unique(row)) / anchors for row in labels], dtype=np.float64
        ),
        "nonlocal_revisit_rate": scalars[:, 4],
        "longest_run_fraction": scalars[:, 5],
        "aba_return_rate": motifs[:, 3],
        "abc_progression_rate": motifs[:, 4],
    }
    if np.any(~np.isfinite(matrix)):
        raise ValueError("non-finite grammar feature")
    return matrix.astype(np.float32), names, diagnostics, slices


def metadata_from_cache(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        name.removeprefix("meta_"): np.asarray(values)
        for name, values in arrays.items()
        if name.startswith("meta_")
    }
    if len(result["failure"]) != EXPECTED_EPISODES:
        raise ValueError("metadata cohort mismatch")
    return result


def task_initial_strata(metadata: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            f"{task}::{initial}"
            for task, initial in zip(metadata["task"], metadata["init_state_id"])
        ]
    )


def bh_adjust(entries: list[dict[str, Any]]) -> None:
    p = np.asarray([entry["permutation_p_one_sided"] for entry in entries])
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(adjusted)
    q[order] = np.minimum(adjusted, 1.0)
    for entry, value in zip(entries, q):
        entry["fdr_bh_q"] = float(value)


def posthoc_associations(
    metadata: dict[str, np.ndarray], labels: np.ndarray, permutations: int, seed: int
) -> dict[str, Any]:
    tasks = metadata["task"]
    strata = task_initial_strata(metadata)
    outcome = np.where(metadata["failure"], "failure", "success")
    specifications = {
        "outcome_within_task_initial_state": (
            outcome,
            strata,
            "within_task_initial_state",
        ),
        "task": (tasks, tasks, "global"),
        "checkpoint": (metadata["checkpoint_sha256"], tasks, "global"),
        "episode_length": (metadata["episode_length"].astype(str), tasks, "global"),
        "episode_length_within_task": (
            metadata["episode_length"].astype(str),
            tasks,
            "within_task",
        ),
    }
    result = {
        name: association_summary(values, labels, group, permutations, seed + index, mode)
        for index, (name, (values, group, mode)) in enumerate(specifications.items())
    }
    bh_adjust(list(result.values()))
    return result


def per_task_outcome(
    metadata: dict[str, np.ndarray], labels: np.ndarray, permutations: int, seed: int
) -> dict[str, Any]:
    result = {}
    estimated = []
    for offset, task in enumerate(np.unique(metadata["task"])):
        index = metadata["task"] == task
        outcome = np.where(metadata["failure"][index], "failure", "success")
        if len(np.unique(outcome)) < 2:
            result[str(task)] = {"status": "no_outcome_variation", "episodes": int(index.sum())}
            continue
        local_labels = np.unique(labels[index], return_inverse=True)[1].astype(np.int32)
        association = association_summary(
            outcome,
            local_labels,
            metadata["init_state_id"][index].astype(str),
            permutations,
            seed + offset,
            "within_initial_state",
        )
        result[str(task)] = {
            "status": "estimated",
            "episodes": int(index.sum()),
            "failures": int(metadata["failure"][index].sum()),
            **association,
        }
        estimated.append(result[str(task)])
    bh_adjust(estimated)
    return result


def cluster_profiles(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    profiles = []
    for cluster in np.unique(labels):
        index = labels == cluster
        task_outcome = Counter(
            f"{task}::{outcome}"
            for task, outcome in zip(
                metadata["task"][index],
                np.where(metadata["failure"][index], "failure", "success"),
            )
        )
        task_counts = Counter(metadata["task"][index])
        lengths = metadata["episode_length"][index]
        failures = int(metadata["failure"][index].sum())
        profiles.append(
            {
                "cluster": int(cluster),
                "episodes": int(index.sum()),
                "successes": int(index.sum() - failures),
                "failures": failures,
                "failure_rate": float(failures / index.sum()),
                "dominant_task_fraction": float(max(task_counts.values()) / index.sum()),
                "task_counts": dict(sorted(task_counts.items())),
                "task_outcome_counts": dict(sorted(task_outcome.items())),
                "length_min": int(lengths.min()),
                "length_median": float(np.median(lengths)),
                "length_max": int(lengths.max()),
                "diagnostic_means": {
                    name: float(values[index].mean()) for name, values in diagnostics.items()
                },
            }
        )
    return profiles


def grammar_scalar_tests(
    diagnostics: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    names = list(diagnostics)
    matrix = np.column_stack([diagnostics[name] for name in names]).astype(np.float64)
    failure = metadata["failure"].astype(bool)
    strata = task_initial_strata(metadata)
    residual = matrix.copy()
    groups = []
    for stratum in np.unique(strata):
        index = np.flatnonzero(strata == stratum)
        residual[index] -= residual[index].mean(axis=0)
        groups.append(index)
    scale = residual.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    observed = (residual[failure].mean(axis=0) - residual[~failure].mean(axis=0)) / scale
    rng = np.random.default_rng(seed)
    hits = np.zeros(len(names), dtype=np.int64)
    for _draw in range(permutations):
        shuffled = failure.copy()
        for index in groups:
            shuffled[index] = shuffled[index][rng.permutation(len(index))]
        statistic = (
            residual[shuffled].mean(axis=0) - residual[~shuffled].mean(axis=0)
        ) / scale
        hits += np.abs(statistic) >= np.abs(observed) - 1e-15
    result = {}
    for index, name in enumerate(names):
        result[name] = {
            "success_mean": float(matrix[~failure, index].mean()),
            "failure_mean": float(matrix[failure, index].mean()),
            "task_initial_residual_effect_sd": float(observed[index]),
            "permutation": "within_task_initial_state",
            "permutation_p_one_sided": float((hits[index] + 1) / (permutations + 1)),
        }
    bh_adjust(list(result.values()))
    return result


def _binary_cv_metrics(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    tasks: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    del seed
    splitter = GroupKFold(n_splits=8)
    fold_auc = []
    per_task: dict[str, list[float]] = {str(task): [] for task in np.unique(tasks)}
    for train, test in splitter.split(matrix, target, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000),
        )
        model.fit(matrix[train], target[train])
        prediction = model.predict_proba(matrix[test])[:, 1]
        fold_auc.append(float(roc_auc_score(target[test], prediction)))
        for task in np.unique(tasks[test]):
            local = tasks[test] == task
            if len(np.unique(target[test][local])) == 2:
                per_task[str(task)].append(
                    float(roc_auc_score(target[test][local], prediction[local]))
                )
    per_task_mean = {
        task: float(np.mean(values)) for task, values in per_task.items() if values
    }
    return {
        "fold_mean_roc_auc": float(np.mean(fold_auc)),
        "fold_sd_roc_auc": float(np.std(fold_auc, ddof=1)),
        "per_task_fold_mean_roc_auc": per_task_mean,
        "mixed_task_macro_roc_auc": float(np.mean(list(per_task_mean.values()))),
    }


def fixed_representation_probes(
    grammar: np.ndarray,
    slices: dict[str, slice],
    metadata: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    tasks, task_id = np.unique(metadata["task"], return_inverse=True)
    task_onehot = np.eye(len(tasks), dtype=np.float64)[task_id]
    length = metadata["episode_length"].astype(np.float64)[:, None]
    occupancy = grammar[:, slices["occupancy"]]
    groups = metadata["init_state_id"].astype(str)
    failure = metadata["failure"].astype(np.int32)
    designs = {
        "task_only": task_onehot,
        "occupancy_only": occupancy,
        "grammar_only": grammar,
        "task_plus_occupancy": np.column_stack([task_onehot, occupancy]),
        "task_plus_grammar": np.column_stack([task_onehot, grammar]),
        "task_plus_length": np.column_stack([task_onehot, length]),
        "task_length_occupancy": np.column_stack([task_onehot, length, occupancy]),
        "task_length_grammar": np.column_stack([task_onehot, length, grammar]),
    }
    outcome = {}
    for offset, (name, matrix) in enumerate(designs.items()):
        outcome[name] = _binary_cv_metrics(
            matrix, failure, groups, metadata["task"], seed + offset
        )

    task_groups = metadata["init_state_id"].astype(str)
    task_splitter = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 20)
    task_probe = {}
    for name, matrix in (("occupancy_only", occupancy), ("grammar_only", grammar)):
        prediction = np.empty(len(task_id), dtype=np.int32)
        for train, test in task_splitter.split(matrix, task_id, task_groups):
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
            model.fit(matrix[train], task_id[train])
            prediction[test] = model.predict(matrix[test])
        task_probe[name] = {"balanced_accuracy": float(balanced_accuracy_score(task_id, prediction))}

    length_probe = {}
    length_splitter = GroupKFold(n_splits=8)
    for name, matrix in (
        ("task_only", task_onehot),
        ("occupancy_only", occupancy),
        ("grammar_only", grammar),
        ("task_plus_grammar", np.column_stack([task_onehot, grammar])),
    ):
        prediction = np.empty(len(length), dtype=np.float64)
        for train, test in length_splitter.split(matrix, length[:, 0], task_groups):
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(matrix[train], length[train, 0])
            prediction[test] = model.predict(matrix[test])
        length_probe[name] = {
            "r2": float(r2_score(length[:, 0], prediction)),
            "mae_queries": float(mean_absolute_error(length[:, 0], prediction)),
        }
    return {
        "protocol": "fixed label-blind full-cohort vocabulary; 8-fold initial-state-group CV; AUC aggregated within fold",
        "outcome": outcome,
        "task": task_probe,
        "episode_length": length_probe,
    }


def fixed_episode_partition(
    grammar: np.ndarray, clusters: int, score: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    embedding, preprocessing = prepare_embedding(grammar, "grammar", seed)
    labels = canonicalize(ward_labels(embedding, clusters), score)
    sizes = np.bincount(labels, minlength=clusters)
    return labels, {"sizes": sizes.tolist(), "preprocessing": preprocessing}


def sensitivity_analysis(
    arrays: dict[str, np.ndarray],
    primary_labels: np.ndarray,
    episode_clusters: int,
    seed: int,
) -> list[dict[str, Any]]:
    specifications = [
        ("state_k6", "feature_primary_geometry", 10, 6),
        ("state_k10", "feature_primary_geometry", 10, 10),
        ("phase_grid5", "feature_grid5_geometry", 5, PRIMARY_STATE_COUNT),
        ("truncate90", "feature_truncate90_geometry", 10, PRIMARY_STATE_COUNT),
    ]
    result = []
    for offset, (name, key, anchors, state_count) in enumerate(specifications):
        geometry = phase_geometry(arrays[key], anchors)
        states, _state_embedding, state_info = discover_route_states(
            geometry, state_count, seed + 1000 + offset
        )
        grammar, _names, diagnostics, _slices = grammar_features(states, state_count)
        labels, cluster_info = fixed_episode_partition(
            grammar, episode_clusters, diagnostics["switch_rate"], seed + 2000 + offset
        )
        result.append(
            {
                "configuration": name,
                "state_count": state_count,
                "phase_anchors": anchors,
                "episode_clusters": episode_clusters,
                "ari_to_primary": float(adjusted_rand_score(primary_labels, labels)),
                "state_preprocessing": state_info["preprocessing"],
                **cluster_info,
                "minimum_cluster_size": max(16, int(math.ceil(0.02 * len(labels)))),
                "minimum_size_pass": bool(
                    min(cluster_info["sizes"])
                    >= max(16, int(math.ceil(0.02 * len(labels))))
                ),
            }
        )
    return result


def plot_summary(
    path: pathlib.Path,
    state_sequences: np.ndarray,
    state_count: int,
    grammar_embedding: np.ndarray,
    cluster_labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    scalar_tests: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    phase = np.zeros((state_sequences.shape[1], state_count))
    for anchor in range(state_sequences.shape[1]):
        phase[anchor] = np.bincount(
            state_sequences[:, anchor], minlength=state_count
        ) / len(state_sequences)
    image = axes[0, 0].imshow(phase.T, aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Route-state occupancy by relative phase")
    axes[0, 0].set_xlabel("phase anchor")
    axes[0, 0].set_ylabel("route state")
    figure.colorbar(image, ax=axes[0, 0], fraction=0.046)

    failure = metadata["failure"]
    for outcome, mask, color in (("success", ~failure, "#2b8cbe"), ("failure", failure, "#d7301f")):
        axes[0, 1].scatter(
            grammar_embedding[mask, 0],
            grammar_embedding[mask, 1],
            s=8,
            alpha=0.35,
            color=color,
            label=outcome,
        )
    axes[0, 1].set_title("Grammar embedding")
    axes[0, 1].set_xlabel("PC1")
    axes[0, 1].set_ylabel("PC2")
    axes[0, 1].legend()

    cluster_failure = [
        metadata["failure"][cluster_labels == cluster].mean()
        for cluster in np.unique(cluster_labels)
    ]
    axes[1, 0].bar(np.arange(len(cluster_failure)), cluster_failure, color="#756bb1")
    axes[1, 0].axhline(metadata["failure"].mean(), color="black", linestyle=":")
    axes[1, 0].set_title("Failure rate by grammar cluster")
    axes[1, 0].set_xlabel("grammar cluster")
    axes[1, 0].set_ylabel("failure rate")

    names = list(scalar_tests)
    effects = [scalar_tests[name]["task_initial_residual_effect_sd"] for name in names]
    axes[1, 1].barh(np.arange(len(names)), effects, color="#31a354")
    axes[1, 1].axvline(0.0, color="black", linewidth=1)
    axes[1, 1].set_yticks(np.arange(len(names)), labels=names)
    axes[1, 1].set_title("Failure-minus-success grammar effects")
    axes[1, 1].set_xlabel("task/init residual SD")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def render_report(summary: dict[str, Any]) -> str:
    clustering = summary["episode_clustering"]
    trial = next(
        row for row in clustering["trials"] if row["clusters"] == clustering["reported_k"]
    )
    association = summary["posthoc_association"]
    lines = [
        "# Symbolic MoE routing-state grammar",
        "",
        "The ten normalized phase anchors are quantized into shared, label-blind route",
        "states. Episode vectors contain normalized state occupancies, directed Markov",
        "transitions, coarse phase occupancy, and return motifs; raw phase descriptors",
        "are not flattened into the episode representation.",
        "",
        "## State vocabulary",
        "",
        f"- Primary states: {summary['state_vocabulary']['state_count']}.",
        f"- State counts: {'/'.join(map(str, summary['state_vocabulary']['state_counts']))}.",
        f"- Seed-refit ARI median/min: {summary['state_vocabulary']['seed_stability']['ari_median']:.3f}/{summary['state_vocabulary']['seed_stability']['ari_min']:.3f}.",
        f"- Grammar dimensions: {summary['grammar']['dimensions']}.",
        "",
        "## Episode grammar partition",
        "",
        f"Status: `{clustering['status']}`; reported K={clustering['reported_k']}; sizes={'/'.join(map(str, clustering['reported_sizes']))}.",
        f"Reported-K subsample ARI median/P10={trial['ari_median']:.3f}/{trial['ari_p10']:.3f}, PAC={trial['consensus_pac_01_09']:.3f}.",
        "",
        "| association | NMI | null | excess | p | BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in association.items():
        lines.append(
            f"| {name} | {row['nmi']:.3f} | {row['null_nmi_mean']:.3f} | "
            f"{row['nmi_excess_over_null']:.3f} | {row['permutation_p_one_sided']:.4f} | {row['fdr_bh_q']:.4f} |"
        )
    lines.extend(
        [
            "",
            "### Outcome association within each task",
            "",
            "| task | episodes | failures | NMI excess | p | BH q |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, row in summary["per_task_outcome_association"].items():
        if row["status"] != "estimated":
            lines.append(
                f"| {task} | {row['episodes']} | 0 | no outcome variation | - | - |"
            )
            continue
        lines.append(
            f"| {task} | {row['episodes']} | {row['failures']} | "
            f"{row['nmi_excess_over_null']:.3f} | "
            f"{row['permutation_p_one_sided']:.4f} | {row['fdr_bh_q']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Interpretation flags: `{'+'.join(summary['interpretation_flags']) or 'none'}`.",
            "",
            "## Cluster composition",
            "",
            "| cluster | n | success | failure | failure rate | dominant task | length min/med/max |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["profiles"]:
        lines.append(
            f"| C{row['cluster']} | {row['episodes']} | {row['successes']} | {row['failures']} | "
            f"{row['failure_rate']:.3f} | {row['dominant_task_fraction']:.3f} | "
            f"{row['length_min']}/{row['length_median']:.1f}/{row['length_max']} |"
        )
    lines.extend(
        [
            "",
            "## Grammar scalar outcome tests",
            "",
            "| diagnostic | success | failure | residual effect SD | BH q |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["grammar_scalar_tests"].items():
        lines.append(
            f"| {name} | {row['success_mean']:.3f} | {row['failure_mean']:.3f} | "
            f"{row['task_initial_residual_effect_sd']:.3f} | {row['fdr_bh_q']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fixed-representation probes",
            "",
            "| outcome model | fold-mean AUC | mixed-task macro AUC |",
            "|---|---:|---:|",
        ]
    )
    for name, row in summary["probes"]["outcome"].items():
        lines.append(f"| {name} | {row['fold_mean_roc_auc']:.3f} | {row['mixed_task_macro_roc_auc']:.3f} |")
    lines.extend(
        [
            "",
            "Task balanced accuracy: occupancy-only %.3f; full grammar %.3f."
            % (
                summary["probes"]["task"]["occupancy_only"]["balanced_accuracy"],
                summary["probes"]["task"]["grammar_only"]["balanced_accuracy"],
            ),
            "",
            "| length model | R2 | MAE queries |",
            "|---|---:|---:|",
        ]
    )
    for name, row in summary["probes"]["episode_length"].items():
        lines.append(f"| {name} | {row['r2']:.3f} | {row['mae_queries']:.3f} |")
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| configuration | states | anchors | fixed episode K | ARI | sizes | min-size |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in summary["sensitivity"]:
        lines.append(
            f"| {row['configuration']} | {row['state_count']} | {row['phase_anchors']} | "
            f"{row['episode_clusters']} | {row['ari_to_primary']:.3f} | {'/'.join(map(str, row['sizes']))} | "
            f"{'yes' if row['minimum_size_pass'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The vocabulary is expert-permutation-invariant but can still encode task, checkpoint, and visual-state structure.",
            "- Relative phase and normalized transition counts remove explicit trajectory length, not completion-versus-timeout semantics.",
            "- Failure horizon has almost no overlap with successful lengths, so task/length-adjusted probes diagnose rather than identify an outcome-specific mechanism.",
            "- Episode K is searched only through six; selection at K=6 is an upper-bound result and may hide finer grammar states.",
            "- The cross-validation probes use a vocabulary fitted label-blind on the full cohort and are descriptive, not held-out vocabulary generalization.",
        ]
    )
    return "\n".join(lines) + "\n"


def self_test() -> None:
    sequences = np.asarray(
        [
            [0, 0, 1, 0, 2, 2],
            [1, 1, 1, 2, 2, 0],
        ],
        dtype=np.int32,
    )
    matrix, names, diagnostics, slices = grammar_features(sequences, 3)
    assert matrix.shape == (2, len(names))
    np.testing.assert_allclose(matrix[:, slices["occupancy"]].sum(axis=1), 1.0)
    np.testing.assert_allclose(matrix[:, slices["transition"]].sum(axis=1), 1.0)
    np.testing.assert_allclose(matrix[:, slices["motif_class"]].sum(axis=1), 1.0)
    assert diagnostics["aba_return_rate"][0] > 0.0
    assert diagnostics["aba_return_rate"][1] == 0.0
    assert np.all(np.isfinite(matrix))
    assert phase_geometry(np.zeros((4, 30)), 5).shape == (4, 5, 6)
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    arrays, manifest = load_feature_cache(args.feature_cache)
    metadata = metadata_from_cache(arrays)
    geometry = phase_geometry(arrays["feature_primary_geometry"], 10)
    state_sequences, state_embedding, state_info = discover_route_states(
        geometry, args.state_count, args.seed
    )
    grammar, feature_names, diagnostics, slices = grammar_features(
        state_sequences, args.state_count
    )
    cluster_result, labels, grammar_embedding = cluster_joint_view(
        grammar,
        "grammar",
        diagnostics["switch_rate"],
        args.subsamples,
        args.seed + 10,
    )
    cluster_result["reported_sizes"] = np.bincount(
        labels, minlength=cluster_result["reported_k"]
    ).tolist()
    associations = posthoc_associations(
        metadata, labels, args.permutations, args.seed + 20
    )
    profiles = cluster_profiles(labels, metadata, diagnostics)
    task_shadowed = bool(
        associations["task"]["nmi"] >= TASK_SHADOW_NMI
        or max(row["dominant_task_fraction"] for row in profiles) >= TASK_SHADOW_FRACTION
    )
    outcome = associations["outcome_within_task_initial_state"]
    outcome_structured = bool(
        outcome["nmi_excess_over_null"] >= OUTCOME_EXCESS_NMI_MIN
        and outcome["fdr_bh_q"] < 0.05
    )
    flags = []
    if cluster_result["status"] != "stable_partition":
        flags.append("exploratory_only")
    if outcome_structured:
        flags.append("outcome_structured")
    if task_shadowed:
        flags.append("task_shadowed")
    cluster_result["posthoc_association"] = associations
    cluster_result["profiles"] = profiles

    seed_stability = state_seed_stability(
        state_embedding, state_sequences, args.state_count, args.seed
    )
    scalar_tests = grammar_scalar_tests(
        diagnostics, metadata, args.permutations, args.seed + 30
    )
    per_task = per_task_outcome(metadata, labels, args.permutations, args.seed + 40)
    probes = fixed_representation_probes(grammar, slices, metadata, args.seed + 50)
    sensitivity = sensitivity_analysis(
        arrays, labels, cluster_result["reported_k"], args.seed + 60
    )

    summary = {
        "provenance": {
            "run_class": "formal",
            "source_feature_cache": str(args.feature_cache.resolve()),
            "source_schema": manifest["schema"],
            "episodes": EXPECTED_EPISODES,
            "successes": EXPECTED_SUCCESSES,
            "failures": EXPECTED_FAILURES,
            "seed": args.seed,
        },
        "state_vocabulary": {
            **state_info["preprocessing"],
            "seed_stability": seed_stability,
        },
        "grammar": {
            "dimensions": int(grammar.shape[1]),
            "feature_blocks": {
                name: [value.start, value.stop] for name, value in slices.items()
            },
            "feature_names": feature_names,
            "absolute_episode_length_in_features": False,
        },
        "episode_clustering": cluster_result,
        "posthoc_association": associations,
        "per_task_outcome_association": per_task,
        "profiles": profiles,
        "grammar_scalar_tests": scalar_tests,
        "probes": probes,
        "sensitivity": sensitivity,
        "outcome_structured": outcome_structured,
        "task_shadowed": task_shadowed,
        "interpretation_flags": flags,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    with (args.out_dir / "assignments.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "task",
                "episode",
                "init_state_id",
                "flow_noise_seed",
                "episode_length",
                "outcome",
                "grammar_cluster",
                "state_sequence",
                *diagnostics.keys(),
            ]
        )
        for index in range(EXPECTED_EPISODES):
            writer.writerow(
                [
                    metadata["task"][index],
                    int(metadata["episode"][index]),
                    int(metadata["init_state_id"][index]),
                    int(metadata["flow_noise_seed"][index]),
                    int(metadata["episode_length"][index]),
                    "failure" if metadata["failure"][index] else "success",
                    int(labels[index]),
                    "-".join(map(str, state_sequences[index].tolist())),
                    *(float(values[index]) for values in diagnostics.values()),
                ]
            )
    np.savez_compressed(
        args.out_dir / "arrays.npz",
        state_sequence=state_sequences,
        grammar_features=grammar,
        grammar_embedding=grammar_embedding,
        grammar_cluster=labels,
        failure=metadata["failure"],
        episode_length=metadata["episode_length"],
        state_centers=state_info["centers"],
    )
    plot_summary(
        args.out_dir / "overview.png",
        state_sequences,
        args.state_count,
        grammar_embedding,
        labels,
        metadata,
        scalar_tests,
    )
    (args.out_dir / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "run_class": "formal",
                "episodes": EXPECTED_EPISODES,
                "state_count": args.state_count,
                "grammar_dimensions": int(grammar.shape[1]),
                "reported_episode_k": cluster_result["reported_k"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
