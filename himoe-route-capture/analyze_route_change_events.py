#!/usr/bin/env python3
"""Organize rollout routing by change events rather than flattened phase traces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
from collections import Counter
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    normalized_mutual_info_score,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler


HERE = pathlib.Path(__file__).resolve().parent
CACHE_DIR = HERE / "analysis/all-outcome-routing-clusters/feature_cache"
PRIOR_OUTPUT = HERE / "analysis/all-outcome-routing-clusters/embeddings_and_labels.npz"
OUT_DIR = HERE / "analysis/route-change-events"

N_EPISODES = 2560
N_FAILURES = 307
N_PHASE = 10
N_LAYERS = 8
PAIR_WIDTH = 24
CANDIDATE_K = tuple(range(2, 9))
MIN_CLUSTER_FRACTION = 0.02
PEAK_PROMINENCE_REL = 0.20
RETURN_ADVANTAGE_REL = 0.05
LOW_SPEED_REL = 0.65
CHANGE_CONTRAST_REL = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=pathlib.Path, default=CACHE_DIR)
    parser.add_argument("--prior-output", type=pathlib.Path, default=PRIOR_OUTPUT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--subsamples", type=int, default=50)
    parser.add_argument("--association-permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest_arrays(cache_dir: pathlib.Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = (
        "feature_primary_recurrence",
        "meta_task",
        "meta_checkpoint_sha256",
        "meta_episode",
        "meta_init_state_id",
        "meta_flow_noise_seed",
        "meta_episode_length",
        "meta_failure",
    )
    arrays: dict[str, np.ndarray] = {}
    for name in required:
        record = manifest["arrays"][name]
        path = cache_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"cache hash mismatch for {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != record["shape"] or str(values.dtype) != record["dtype"]:
            raise ValueError(f"cache schema mismatch for {name}")
        arrays[name] = values
    if arrays["feature_primary_recurrence"].shape != (N_EPISODES, 1080):
        raise ValueError("unexpected recurrence feature matrix")
    failure = np.asarray(arrays["meta_failure"], dtype=bool)
    if int(failure.sum()) != N_FAILURES:
        raise ValueError("formal outcome cohort drifted")
    return arrays, manifest


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(abs(denominator), 1e-8))


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std() < 1e-10 or right.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def longest_true_run(values: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def event_time_summary(mask: np.ndarray, strength: np.ndarray, phase: np.ndarray) -> tuple[float, ...]:
    index = np.flatnonzero(mask)
    if not len(index):
        return 0.0, 0.0, -1.0, -1.0, -1.0, 0.0
    selected = np.maximum(strength[index], 0.0)
    weighted = float(np.average(phase[index], weights=selected)) if selected.sum() else float(
        phase[index].mean()
    )
    return (
        float(len(index) / len(mask)),
        float(selected.mean()),
        float(phase[index[0]]),
        float(phase[index[-1]]),
        weighted,
        float(np.mean(phase[index] >= 0.5)),
    )


def pairs_and_lookup() -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    pairs = np.asarray(
        [(left, right) for left in range(N_PHASE - 1) for right in range(left + 1, N_PHASE)],
        dtype=np.int32,
    )
    return pairs, {tuple(map(int, pair)): index for index, pair in enumerate(pairs)}


def extract_event_features(
    recurrence: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, list[str], dict[str, np.ndarray]]:
    values = np.asarray(recurrence, dtype=np.float32).reshape(-1, 45, PAIR_WIDTH)
    soft_layer = values[:, :, :N_LAYERS]
    soft_std_layer = values[:, :, N_LAYERS : 2 * N_LAYERS]
    hard_layer = values[:, :, 2 * N_LAYERS :]
    pairs, lookup = pairs_and_lookup()
    adjacent_index = np.asarray([lookup[(index, index + 1)] for index in range(N_PHASE - 1)])
    transition_phase = np.linspace(0.0, 1.0, N_PHASE - 1)
    core_rows: list[list[float]] = []
    scale_rows: list[list[float]] = []
    core_names: list[str] | None = None
    scale_names: list[str] | None = None
    event_tapes = {
        "soft_speed": [],
        "hard_speed": [],
        "peak_event": [],
        "return_event": [],
        "joint_return_event": [],
        "low_speed_event": [],
        "change_contrast": [],
    }

    lag = pairs[:, 1] - pairs[:, 0]
    early_pairs = (pairs[:, 0] < 5) & (pairs[:, 1] < 5)
    late_pairs = (pairs[:, 0] >= 5) & (pairs[:, 1] >= 5)
    x_lag = np.arange(1, N_PHASE, dtype=np.float64)
    centered_lag = x_lag - x_lag.mean()
    lag_denominator = float(np.sum(centered_lag**2))

    for episode in range(len(values)):
        soft_pair_layer = soft_layer[episode]
        hard_pair_layer = hard_layer[episode]
        soft_pair = soft_pair_layer.mean(axis=1)
        hard_pair = hard_pair_layer.mean(axis=1)
        speed_layer = soft_pair_layer[adjacent_index]
        hard_speed_layer = hard_pair_layer[adjacent_index]
        denoise_std = soft_std_layer[episode, adjacent_index].mean(axis=1)
        speed = speed_layer.mean(axis=1)
        hard_speed = hard_speed_layer.mean(axis=1)
        soft_scale = max(float(np.median(speed)), 1e-8)
        hard_scale = max(float(np.median(hard_speed)), 1e-8)

        early_speed = float(speed[:4].mean())
        late_speed = float(speed[-4:].mean())
        early_hard = float(hard_speed[:4].mean())
        late_hard = float(hard_speed[-4:].mean())

        prominence = speed[1:-1] - 0.5 * (speed[:-2] + speed[2:])
        prominence_norm = prominence / soft_scale
        peak_mask_inner = (prominence_norm > PEAK_PROMINENCE_REL) & (
            speed[1:-1] > np.median(speed)
        )
        peak_mask = np.zeros(len(speed), dtype=bool)
        peak_strength = np.zeros(len(speed), dtype=np.float32)
        peak_mask[1:-1] = peak_mask_inner
        peak_strength[1:-1] = prominence_norm
        peak_summary = event_time_summary(peak_mask, peak_strength, transition_phase)

        layer_peak = np.argmax(speed_layer, axis=0)
        global_peak = int(np.argmax(speed))
        layer_curves = speed_layer.T
        correlations = [
            safe_corr(layer_curves[left], layer_curves[right])
            for left in range(N_LAYERS - 1)
            for right in range(left + 1, N_LAYERS)
        ]

        splits = np.arange(2, N_PHASE - 1, dtype=np.int32)
        soft_contrast_layer = []
        hard_contrast_layer = []
        for split in splits:
            within_left = (pairs[:, 0] < split) & (pairs[:, 1] < split)
            within_right = (pairs[:, 0] >= split) & (pairs[:, 1] >= split)
            cross = (pairs[:, 0] < split) & (pairs[:, 1] >= split)
            soft_contrast_layer.append(
                soft_pair_layer[cross].mean(axis=0)
                - 0.5
                * (
                    soft_pair_layer[within_left].mean(axis=0)
                    + soft_pair_layer[within_right].mean(axis=0)
                )
            )
            hard_contrast_layer.append(
                hard_pair_layer[cross].mean(axis=0)
                - 0.5
                * (
                    hard_pair_layer[within_left].mean(axis=0)
                    + hard_pair_layer[within_right].mean(axis=0)
                )
            )
        soft_contrast_layer = np.asarray(soft_contrast_layer) / soft_scale
        hard_contrast_layer = np.asarray(hard_contrast_layer) / hard_scale
        soft_contrast = soft_contrast_layer.mean(axis=1)
        hard_contrast = hard_contrast_layer.mean(axis=1)
        best_split_position = int(np.argmax(soft_contrast))
        best_split = int(splits[best_split_position])
        layer_split_position = np.argmax(soft_contrast_layer, axis=0)
        hard_best_split = int(splits[int(np.argmax(hard_contrast))])

        return_advantage = []
        hard_return_advantage = []
        return_lag = []
        for current in range(2, N_PHASE):
            previous = soft_pair[lookup[(current - 1, current)]]
            older = np.asarray([soft_pair[lookup[(left, current)]] for left in range(current - 1)])
            nearest_position = int(np.argmin(older))
            return_advantage.append((previous - older[nearest_position]) / soft_scale)
            hard_previous = hard_pair[lookup[(current - 1, current)]]
            hard_older = np.asarray(
                [hard_pair[lookup[(left, current)]] for left in range(current - 1)]
            )
            hard_return_advantage.append((hard_previous - hard_older.min()) / hard_scale)
            return_lag.append(current - nearest_position)
        return_advantage = np.asarray(return_advantage, dtype=np.float32)
        hard_return_advantage = np.asarray(hard_return_advantage, dtype=np.float32)
        return_lag = np.asarray(return_lag, dtype=np.float32)
        return_phase = np.linspace(2 / 9, 1.0, len(return_advantage))
        return_event = return_advantage > RETURN_ADVANTAGE_REL
        joint_return_event = return_event & (hard_return_advantage > 0.0)
        return_summary = event_time_summary(return_event, return_advantage, return_phase)

        low_speed = speed < LOW_SPEED_REL * soft_scale
        layer_baseline = np.maximum(np.median(speed_layer, axis=0), 1e-8)
        layer_low = speed_layer < LOW_SPEED_REL * layer_baseline[None, :]

        soft_lag_curve = np.asarray(
            [soft_pair[lag == value].mean() for value in range(1, N_PHASE)]
        )
        hard_lag_curve = np.asarray(
            [hard_pair[lag == value].mean() for value in range(1, N_PHASE)]
        )
        soft_slope = float(
            np.sum((soft_lag_curve - soft_lag_curve.mean()) * centered_lag)
            / lag_denominator
        )
        hard_slope = float(
            np.sum((hard_lag_curve - hard_lag_curve.mean()) * centered_lag)
            / lag_denominator
        )

        core = {
            "soft_speed_cv": safe_ratio(float(speed.std()), float(speed.mean())),
            "soft_speed_max_ratio": safe_ratio(float(speed.max()), soft_scale),
            "soft_speed_min_ratio": safe_ratio(float(speed.min()), soft_scale),
            "soft_late_early_ratio": safe_ratio(late_speed, early_speed),
            "soft_terminal_speed_ratio": safe_ratio(float(speed[-1]), soft_scale),
            "hard_speed_cv": safe_ratio(float(hard_speed.std()), float(hard_speed.mean())),
            "hard_speed_max_ratio": safe_ratio(float(hard_speed.max()), hard_scale),
            "hard_late_early_ratio": safe_ratio(late_hard, early_hard),
            "denoise_std_over_speed": safe_ratio(float(denoise_std.mean()), soft_scale),
            "denoise_std_cv": safe_ratio(float(denoise_std.std()), float(denoise_std.mean())),
            "peak_count_fraction": peak_summary[0],
            "peak_max_prominence": float(max(0.0, prominence_norm.max())),
            "peak_mean_prominence": peak_summary[1],
            "peak_first_phase": peak_summary[2],
            "peak_last_phase": peak_summary[3],
            "peak_strength_phase": peak_summary[4],
            "peak_late_fraction": peak_summary[5],
            "global_peak_phase": float(transition_phase[global_peak]),
            "layer_peak_exact_fraction": float(np.mean(layer_peak == global_peak)),
            "layer_peak_within1_fraction": float(np.mean(np.abs(layer_peak - global_peak) <= 1)),
            "layer_peak_phase_sd": float(np.std(layer_peak / (len(speed) - 1))),
            "layer_speed_curve_correlation": float(np.mean(correlations)),
            "soft_hard_speed_correlation": safe_corr(speed, hard_speed),
            "soft_hard_peak_phase_gap": float(
                abs(global_peak - int(np.argmax(hard_speed))) / (len(speed) - 1)
            ),
            "change_max_contrast": float(soft_contrast.max()),
            "change_second_contrast": float(np.partition(soft_contrast, -2)[-2]),
            "change_count_fraction": float(np.mean(soft_contrast > CHANGE_CONTRAST_REL)),
            "change_phase": float(best_split / (N_PHASE - 1)),
            "change_layer_within1_fraction": float(
                np.mean(np.abs(layer_split_position - best_split_position) <= 1)
            ),
            "change_layer_phase_sd": float(np.std(splits[layer_split_position] / 9.0)),
            "hard_change_max_contrast": float(hard_contrast.max()),
            "soft_hard_change_phase_gap": float(abs(best_split - hard_best_split) / 9.0),
            "return_count_fraction": return_summary[0],
            "joint_return_count_fraction": float(np.mean(joint_return_event)),
            "return_max_advantage": float(max(0.0, return_advantage.max())),
            "return_mean_positive_advantage": return_summary[1],
            "return_first_phase": return_summary[2],
            "return_last_phase": return_summary[3],
            "return_strength_phase": return_summary[4],
            "return_late_fraction": return_summary[5],
            "return_mean_lag_fraction": float(
                np.mean(return_lag[return_event]) / 9.0 if np.any(return_event) else 0.0
            ),
            "terminal_return_advantage": float(return_advantage[-1]),
            "terminal_return_indicator": float(return_event[-1]),
            "return_sign_agreement": float(
                np.mean((return_advantage > 0.0) == (hard_return_advantage > 0.0))
            ),
            "low_speed_fraction": float(np.mean(low_speed)),
            "longest_stasis_run_fraction": float(longest_true_run(low_speed) / len(low_speed)),
            "late_low_speed_fraction": float(np.mean(low_speed[-3:])),
            "terminal_low_speed_indicator": float(low_speed[-1]),
            "late_internal_early_ratio": safe_ratio(
                float(soft_pair[late_pairs].mean()), float(soft_pair[early_pairs].mean())
            ),
            "late_stasis_indicator": float(late_speed < 0.75 * early_speed),
            "layer_low_max_fraction": float(layer_low.mean(axis=1).max()),
            "layer_low_terminal_fraction": float(layer_low[-1].mean()),
            "soft_far_adjacent_ratio": safe_ratio(
                float(soft_pair[lag >= 5].mean()), float(soft_pair[lag == 1].mean())
            ),
            "soft_lag_slope_normalized": safe_ratio(soft_slope, soft_scale),
            "soft_terminal_drift_normalized": safe_ratio(
                float(soft_pair[lookup[(0, 9)]]), soft_scale
            ),
            "hard_far_adjacent_ratio": safe_ratio(
                float(hard_pair[lag >= 5].mean()), float(hard_pair[lag == 1].mean())
            ),
            "hard_lag_slope_normalized": safe_ratio(hard_slope, hard_scale),
        }
        scale = {
            "soft_speed_mean_abs": float(speed.mean()),
            "soft_speed_max_abs": float(speed.max()),
            "hard_speed_mean_abs": float(hard_speed.mean()),
            "hard_speed_max_abs": float(hard_speed.max()),
            "soft_pair_mean_abs": float(soft_pair.mean()),
            "soft_denoise_std_mean_abs": float(denoise_std.mean()),
        }
        if core_names is None:
            core_names = list(core)
            scale_names = list(scale)
        elif list(core) != core_names or list(scale) != scale_names:
            raise RuntimeError("event feature order drifted")
        core_rows.append(list(core.values()))
        scale_rows.append(list(scale.values()))
        event_tapes["soft_speed"].append(speed)
        event_tapes["hard_speed"].append(hard_speed)
        event_tapes["peak_event"].append(peak_mask)
        event_tapes["return_event"].append(return_event)
        event_tapes["joint_return_event"].append(joint_return_event)
        event_tapes["low_speed_event"].append(low_speed)
        event_tapes["change_contrast"].append(soft_contrast)

    core_array = np.asarray(core_rows, dtype=np.float32)
    scale_array = np.asarray(scale_rows, dtype=np.float32)
    if not np.all(np.isfinite(core_array)) or not np.all(np.isfinite(scale_array)):
        raise ValueError("non-finite event features")
    return (
        core_array,
        list(core_names or []),
        scale_array,
        list(scale_names or []),
        {name: np.asarray(tape) for name, tape in event_tapes.items()},
    )


def prepare_embedding(matrix: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=np.float64)
    standard_deviation = values.std(axis=0)
    keep = standard_deviation > 1e-9
    values = values[:, keep]
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = np.clip((values - center) / scale, -20.0, 20.0)
    maximum = min(30, len(standardized) - 1, standardized.shape[1])
    model = PCA(n_components=maximum, svd_solver="randomized", random_state=seed)
    transformed = model.fit_transform(standardized)
    cumulative = np.cumsum(model.explained_variance_ratio_)
    retained = min(maximum, max(2, int(np.searchsorted(cumulative, 0.90) + 1)))
    return transformed[:, :retained], {
        "raw_dimensions": int(matrix.shape[1]),
        "active_dimensions": int(keep.sum()),
        "pca_components": int(retained),
        "variance_explained": float(cumulative[retained - 1]),
        "scaling": "median_iqr",
    }


def ward_labels(embedding: np.ndarray, clusters: int) -> np.ndarray:
    tree = linkage(embedding, method="ward", optimal_ordering=False)
    labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
    if len(np.unique(labels)) != clusters:
        raise ValueError("Ward returned fewer clusters than requested")
    return labels


def matched_cluster_jaccard(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.unique(reference)
    right = np.unique(candidate)
    scores = np.zeros((len(left), len(right)), dtype=np.float64)
    for row, first in enumerate(left):
        first_set = reference == first
        for column, second in enumerate(right):
            second_set = candidate == second
            scores[row, column] = np.sum(first_set & second_set) / np.sum(first_set | second_set)
    row, column = linear_sum_assignment(-scores)
    return float(scores[row, column].mean())


def canonicalize(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = sorted(
        np.unique(labels), key=lambda value: (float(score[labels == value].mean()), int(value))
    )
    mapping = {int(value): index for index, value in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int32)


def cluster_events(
    matrix: np.ndarray, score: np.ndarray, subsamples: int, seed: int
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, seed)
    count = len(embedding)
    minimum = max(16, int(math.ceil(MIN_CLUSTER_FRACTION * count)))
    full_labels = {clusters: ward_labels(embedding, clusters) for clusters in CANDIDATE_K}
    rng = np.random.default_rng(seed + 1000)
    probe = np.sort(rng.choice(count, min(256, count), replace=False))
    silhouette_index = np.sort(rng.choice(count, min(1000, count), replace=False))
    trials: dict[int, dict[str, Any]] = {}
    trackers = {}
    for clusters, labels in full_labels.items():
        sizes = np.bincount(labels, minlength=clusters)
        trials[clusters] = {
            "clusters": clusters,
            "full_silhouette_sampled": float(
                silhouette_score(embedding[silhouette_index], labels[silhouette_index])
            ),
            "sizes": sizes.tolist(),
            "minimum_size_pass": bool(sizes.min() >= minimum),
        }
        trackers[clusters] = {
            "ari": [],
            "jaccard": [],
            "silhouette": [],
            "pair_count": np.zeros((len(probe), len(probe)), dtype=np.uint16),
            "pair_same": np.zeros((len(probe), len(probe)), dtype=np.uint16),
        }

    sample_size = int(math.ceil(0.80 * count))
    for draw in range(subsamples):
        index = np.sort(rng.choice(count, sample_size, replace=False))
        subset, _unused = prepare_embedding(matrix[index], seed + 100000 + draw)
        tree = linkage(subset, method="ward", optimal_ordering=False)
        local_silhouette = np.sort(
            rng.choice(len(subset), min(512, len(subset)), replace=False)
        )
        probe_mask = np.isin(probe, index)
        probe_global = probe[probe_mask]
        probe_position = np.flatnonzero(probe_mask)
        subset_probe = np.searchsorted(index, probe_global)
        pair_index = np.ix_(probe_position, probe_position)
        for clusters in CANDIDATE_K:
            labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
            tracker = trackers[clusters]
            tracker["ari"].append(adjusted_rand_score(full_labels[clusters][index], labels))
            tracker["jaccard"].append(
                matched_cluster_jaccard(full_labels[clusters][index], labels)
            )
            tracker["silhouette"].append(
                float(silhouette_score(subset[local_silhouette], labels[local_silhouette]))
            )
            selected = labels[subset_probe]
            tracker["pair_count"][pair_index] += 1
            tracker["pair_same"][pair_index] += selected[:, None] == selected[None, :]

    for clusters in CANDIDATE_K:
        tracker = trackers[clusters]
        ari = np.asarray(tracker["ari"])
        jaccard = np.asarray(tracker["jaccard"])
        silhouettes = np.asarray(tracker["silhouette"])
        valid = (tracker["pair_count"] > 0) & ~np.eye(len(probe), dtype=bool)
        consensus = tracker["pair_same"][valid] / tracker["pair_count"][valid]
        trials[clusters].update(
            {
                "subsamples_completed": int(len(ari)),
                "ari_median": float(np.median(ari)),
                "ari_p10": float(np.quantile(ari, 0.10)),
                "matched_jaccard_median": float(np.median(jaccard)),
                "subsample_silhouette_mean": float(silhouettes.mean()),
                "subsample_silhouette_sd": float(silhouettes.std(ddof=1)),
                "consensus_pac_01_09": float(
                    np.mean((consensus > 0.1) & (consensus < 0.9))
                ),
            }
        )
        trials[clusters]["stable"] = bool(
            trials[clusters]["minimum_size_pass"]
            and trials[clusters]["ari_median"] >= 0.75
            and trials[clusters]["ari_p10"] >= 0.50
            and trials[clusters]["consensus_pac_01_09"] <= 0.20
        )

    stable = [trials[clusters] for clusters in CANDIDATE_K if trials[clusters]["stable"]]
    if stable:
        selected = max(stable, key=lambda row: row["subsample_silhouette_mean"])
        selected_k = int(selected["clusters"])
        status = "stable_partition"
    else:
        selected_k = None
        status = "no_stable_partition"
    valid = [row for row in trials.values() if row["minimum_size_pass"]]
    exploratory = int(
        max(valid if valid else list(trials.values()), key=lambda row: row["full_silhouette_sampled"])[
            "clusters"
        ]
    )
    reported_k = selected_k if selected_k is not None else exploratory
    labels = canonicalize(full_labels[reported_k], score)
    return (
        {
            "status": status,
            "selected_k": selected_k,
            "exploratory_best_k": exploratory,
            "reported_k": reported_k,
            "reported_sizes": np.bincount(labels, minlength=reported_k).tolist(),
            "minimum_cluster_size": minimum,
            "preprocessing": preprocessing,
            "trials": [trials[clusters] for clusters in CANDIDATE_K],
        },
        labels,
        embedding,
    )


def contingency(categories: np.ndarray, labels: np.ndarray) -> np.ndarray:
    values = np.unique(categories)
    table = np.zeros((len(values), int(labels.max()) + 1), dtype=np.int64)
    for row, value in enumerate(values):
        for cluster in range(table.shape[1]):
            table[row, cluster] = int(np.sum((categories == value) & (labels == cluster)))
    return table


def cramers_v(categories: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import chi2_contingency

    table = contingency(categories, labels)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(math.sqrt((chi2 / table.sum()) / min(table.shape[0] - 1, table.shape[1] - 1)))


def association(
    categories: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray | None,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    categories = np.asarray(categories).astype(str)
    observed = float(normalized_mutual_info_score(categories, labels))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for draw in range(permutations):
        shuffled = categories.copy()
        if strata is None:
            shuffled = shuffled[rng.permutation(len(shuffled))]
        else:
            for value in np.unique(strata):
                index = np.flatnonzero(strata == value)
                shuffled[index] = shuffled[index][rng.permutation(len(index))]
        null[draw] = normalized_mutual_info_score(shuffled, labels)
    return {
        "nmi": observed,
        "null_nmi_mean": float(null.mean()),
        "nmi_excess_over_null": float(observed - null.mean()),
        "cramers_v": cramers_v(categories, labels),
        "permutation_p_one_sided": float((1 + np.sum(null >= observed)) / (permutations + 1)),
    }


def cluster_profiles(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    core: np.ndarray,
    core_names: list[str],
) -> list[dict[str, Any]]:
    selected_names = (
        "peak_count_fraction",
        "change_max_contrast",
        "joint_return_count_fraction",
        "return_max_advantage",
        "longest_stasis_run_fraction",
        "late_stasis_indicator",
        "layer_peak_within1_fraction",
        "layer_speed_curve_correlation",
    )
    columns = {name: core[:, core_names.index(name)] for name in selected_names}
    profiles = []
    for cluster in np.unique(labels):
        index = labels == cluster
        failures = int(metadata["failure"][index].sum())
        profiles.append(
            {
                "cluster": int(cluster),
                "episodes": int(index.sum()),
                "successes": int(index.sum() - failures),
                "failures": failures,
                "failure_rate": float(failures / index.sum()),
                "task_counts": dict(sorted(Counter(metadata["task"][index]).items())),
                "length_counts": dict(
                    sorted(Counter(map(int, metadata["episode_length"][index])).items())
                ),
                "event_means": {
                    name: float(values[index].mean()) for name, values in columns.items()
                },
            }
        )
    return profiles


def double_holdout(metadata: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    task_state = np.asarray(
        [
            f"{task}::{state}"
            for task, state in zip(metadata["task"], metadata["init_state_id"])
        ]
    )
    state_keys = sorted(np.unique(task_state))
    state_rank = {key: index for index, key in enumerate(state_keys)}
    seeds = sorted(map(int, np.unique(metadata["flow_noise_seed"])))
    seed_rank = {seed: index for index, seed in enumerate(seeds)}
    state_fold = np.asarray([state_rank[value] % 4 for value in task_state])
    seed_fold = np.asarray([seed_rank[int(value)] % 4 for value in metadata["flow_noise_seed"]])
    folds = []
    tested = np.zeros(len(task_state), dtype=np.int32)
    for held_state in range(4):
        for held_seed in range(4):
            test = (state_fold == held_state) & (seed_fold == held_seed)
            train = (state_fold != held_state) & (seed_fold != held_seed)
            if len(np.unique(metadata["failure"][train])) != 2 or len(
                np.unique(metadata["failure"][test])
            ) != 2:
                raise ValueError("non-evaluable double-holdout fold")
            tested += test
            folds.append((np.flatnonzero(train), np.flatnonzero(test)))
    if not np.all(tested == 1):
        raise ValueError("double holdout does not test each episode once")
    return folds


def new_encoder() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)


def outcome_models(
    metadata: dict[str, np.ndarray],
    event_features: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    task = metadata["task"].astype(str)
    task_length = np.asarray(
        [f"{name}::T{int(length)}" for name, length in zip(task, metadata["episode_length"])]
    )
    categories = {
        "task": task[:, None],
        "task_length": np.column_stack((task, task_length)),
    }
    specifications = {
        "task": ("task", False),
        "task_length": ("task_length", False),
        "task_events": ("task", True),
        "task_length_events": ("task_length", True),
    }
    outcome = metadata["failure"].astype(np.int32)
    prediction = {name: np.full(len(outcome), np.nan) for name in specifications}
    for train, test in folds:
        for name, (category_name, use_events) in specifications.items():
            encoder = new_encoder()
            train_parts = [encoder.fit_transform(categories[category_name][train])]
            test_parts = [encoder.transform(categories[category_name][test])]
            if use_events:
                scaler = StandardScaler()
                train_parts.append(scaler.fit_transform(event_features[train]))
                test_parts.append(scaler.transform(event_features[test]))
            model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
            model.fit(np.column_stack(train_parts), outcome[train])
            prediction[name][test] = model.predict_proba(np.column_stack(test_parts))[:, 1]
    results = {}
    for name, values in prediction.items():
        if np.any(~np.isfinite(values)):
            raise ValueError(f"missing OOF predictions for {name}")
        results[name] = {
            "roc_auc": float(roc_auc_score(outcome, values)),
            "average_precision": float(average_precision_score(outcome, values)),
            "log_loss": float(log_loss(outcome, values, labels=[0, 1])),
            "brier": float(brier_score_loss(outcome, values)),
        }
    return results, prediction


def bootstrap_increment(
    outcome: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    keys = np.unique(groups)
    indices = {key: np.flatnonzero(groups == key) for key in keys}
    auc = []
    ap = []
    for _draw in range(draws):
        sampled = rng.choice(keys, len(keys), replace=True)
        index = np.concatenate([indices[key] for key in sampled])
        if len(np.unique(outcome[index])) < 2:
            continue
        auc.append(
            roc_auc_score(outcome[index], augmented[index])
            - roc_auc_score(outcome[index], baseline[index])
        )
        ap.append(
            average_precision_score(outcome[index], augmented[index])
            - average_precision_score(outcome[index], baseline[index])
        )
    return {
        "draws_completed": len(auc),
        "roc_auc_increment": float(roc_auc_score(outcome, augmented) - roc_auc_score(outcome, baseline)),
        "roc_auc_cluster_bootstrap_95ci": [float(value) for value in np.quantile(auc, [0.025, 0.975])],
        "average_precision_increment": float(
            average_precision_score(outcome, augmented)
            - average_precision_score(outcome, baseline)
        ),
        "average_precision_cluster_bootstrap_95ci": [
            float(value) for value in np.quantile(ap, [0.025, 0.975])
        ],
    }


def confound_prediction(
    metadata: dict[str, np.ndarray],
    event_features: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    task = metadata["task"].astype(str)
    length = metadata["episode_length"].astype(np.float64)
    task_prediction = np.empty(len(task), dtype=task.dtype)
    length_prediction = np.full(len(task), np.nan)
    for train, test in folds:
        scaler = StandardScaler()
        train_x = scaler.fit_transform(event_features[train])
        test_x = scaler.transform(event_features[test])
        classifier = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs")
        classifier.fit(train_x, task[train])
        task_prediction[test] = classifier.predict(test_x)
        regressor = Ridge(alpha=10.0)
        regressor.fit(train_x, length[train])
        length_prediction[test] = regressor.predict(test_x)
    return {
        "task_accuracy": float(np.mean(task_prediction == task)),
        "task_balanced_accuracy": float(balanced_accuracy_score(task, task_prediction)),
        "task_majority_accuracy": float(max(Counter(task).values()) / len(task)),
        "episode_length_r2": float(r2_score(length, length_prediction)),
        "episode_length_mae_queries": float(np.mean(np.abs(length - length_prediction))),
    }


def write_assignments(
    path: pathlib.Path,
    metadata: dict[str, np.ndarray],
    labels: np.ndarray,
    core: np.ndarray,
    core_names: list[str],
) -> None:
    selected = (
        "peak_count_fraction",
        "change_max_contrast",
        "joint_return_count_fraction",
        "return_max_advantage",
        "longest_stasis_run_fraction",
        "late_stasis_indicator",
        "layer_peak_within1_fraction",
    )
    rows = []
    for index in range(len(labels)):
        row = {
            "task": str(metadata["task"][index]),
            "episode": int(metadata["episode"][index]),
            "init_state_id": int(metadata["init_state_id"][index]),
            "flow_noise_seed": int(metadata["flow_noise_seed"][index]),
            "episode_length": int(metadata["episode_length"][index]),
            "failure": bool(metadata["failure"][index]),
            "event_cluster": int(labels[index]),
        }
        for name in selected:
            row[name] = float(core[index, core_names.index(name)])
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    cluster = summary["clustering"]
    lines = [
        "# Routing change-event organization",
        "",
        "Each episode is represented by event counts, strengths, timings, and",
        "cross-layer agreement for speed peaks, distance-matrix change points,",
        "nonlocal returns, and low-speed runs. Raw phase traces, task, length, and",
        "outcome are not clustering features.",
        "",
        "## Stability",
        "",
        f"Status: `{cluster['status']}`; reported K={cluster['reported_k']}; sizes={cluster['reported_sizes']}.",
        "",
        "| K | min size | silhouette | ARI med/p10 | PAC | stable |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in cluster["trials"]:
        lines.append(
            "| %d | %d | %s | %s / %s | %s | %s |"
            % (
                row["clusters"],
                min(row["sizes"]),
                fmt(row["subsample_silhouette_mean"]),
                fmt(row["ari_median"]),
                fmt(row["ari_p10"]),
                fmt(row["consensus_pac_01_09"]),
                str(row["stable"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Cluster profiles",
            "",
            "| C | n | failure rate | peak count | change | joint return | return strength | stasis run | late stasis | layer sync | tasks |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["profiles"]:
        event = row["event_means"]
        lines.append(
            "| %d | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["cluster"],
                row["episodes"],
                fmt(row["failure_rate"]),
                fmt(event["peak_count_fraction"]),
                fmt(event["change_max_contrast"]),
                fmt(event["joint_return_count_fraction"]),
                fmt(event["return_max_advantage"]),
                fmt(event["longest_stasis_run_fraction"]),
                fmt(event["late_stasis_indicator"]),
                fmt(event["layer_peak_within1_fraction"]),
                "; ".join(f"{name.split('/', 1)[1]}={count}" for name, count in row["task_counts"].items()),
            )
        )
    association_rows = summary["posthoc_association"]
    lines.extend(
        [
            "",
            "## Confounding",
            "",
            "| variable | NMI | null NMI | excess | Cramer's V | p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in association_rows.items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                name,
                fmt(row["nmi"]),
                fmt(row["null_nmi_mean"]),
                fmt(row["nmi_excess_over_null"]),
                fmt(row["cramers_v"]),
                fmt(row["permutation_p_one_sided"]),
            )
        )
    lines.extend(
        [
            "",
            "Event features predict task with accuracy %s (majority %s) and episode length with R2 %s / MAE %s queries."
            % (
                fmt(summary["confound_prediction"]["task_accuracy"]),
                fmt(summary["confound_prediction"]["task_majority_accuracy"]),
                fmt(summary["confound_prediction"]["episode_length_r2"]),
                fmt(summary["confound_prediction"]["episode_length_mae_queries"]),
            ),
            "",
            "ARI against the phase-pair recurrence K=6 partition: %s."
            % fmt(summary["comparison_to_phase_recurrence"]["ari"]),
            "",
            "## Outcome increment",
            "",
            "Predictions use 16 task-initial-state x flow-seed double-holdout folds.",
            "",
            "| model | ROC AUC | average precision | log loss | Brier |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["outcome_models"].items():
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (name, fmt(row["roc_auc"]), fmt(row["average_precision"]), fmt(row["log_loss"]), fmt(row["brier"]))
        )
    for name, row in summary["outcome_increments"].items():
        lines.extend(
            [
                "",
                "- `%s`: AUC increment %s (cluster-bootstrap 95%% CI %s to %s); AP increment %s (CI %s to %s)."
                % (
                    name,
                    fmt(row["roc_auc_increment"]),
                    fmt(row["roc_auc_cluster_bootstrap_95ci"][0]),
                    fmt(row["roc_auc_cluster_bootstrap_95ci"][1]),
                    fmt(row["average_precision_increment"]),
                    fmt(row["average_precision_cluster_bootstrap_95ci"][0]),
                    fmt(row["average_precision_cluster_bootstrap_95ci"][1]),
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Events are derived from ten normalized phase anchors, not dense physical contact labels.",
            "- A return is a nonlocal route-distance event; low-speed plateaus can satisfy it without a true control loop.",
            "- Successful phase 1.0 is completion, while failed phase 1.0 is timeout. This is descriptive, not causal forecasting.",
            "- Task and length are excluded from clustering but can remain recoverable from route dynamics.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    rng = np.random.default_rng(4)
    values = rng.uniform(0.001, 0.5, size=(6, 1080)).astype(np.float32)
    core, names, scale, scale_names, tapes = extract_event_features(values)
    assert core.shape[0] == 6 and core.shape[1] == len(names)
    assert scale.shape == (6, len(scale_names))
    assert tapes["soft_speed"].shape == (6, 9)
    assert np.all(np.isfinite(core))
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.subsamples < 10 or args.association_permutations < 100 or args.bootstrap_draws < 100:
        raise ValueError("too few resampling draws")
    arrays, manifest = load_manifest_arrays(args.cache_dir)
    metadata = {
        name.removeprefix("meta_"): np.asarray(values)
        for name, values in arrays.items()
        if name.startswith("meta_")
    }
    core, core_names, scale, scale_names, event_tapes = extract_event_features(
        arrays["feature_primary_recurrence"]
    )
    scale_augmented = np.column_stack((core, scale))
    cluster, labels, embedding = cluster_events(
        core,
        scale[:, scale_names.index("soft_speed_mean_abs")],
        args.subsamples,
        args.seed,
    )

    task_initial = np.asarray(
        [
            f"{task}::{state}"
            for task, state in zip(metadata["task"], metadata["init_state_id"])
        ]
    )
    outcome = np.where(metadata["failure"], "failure", "success")
    associations = {
        "task": association(
            metadata["task"], labels, None, args.association_permutations, args.seed + 1
        ),
        "checkpoint": association(
            metadata["checkpoint_sha256"],
            labels,
            None,
            args.association_permutations,
            args.seed + 2,
        ),
        "episode_length": association(
            metadata["episode_length"].astype(str),
            labels,
            None,
            args.association_permutations,
            args.seed + 3,
        ),
        "outcome_within_task_initial_state": association(
            outcome,
            labels,
            task_initial,
            args.association_permutations,
            args.seed + 4,
        ),
    }

    with np.load(args.prior_output, allow_pickle=False) as prior:
        prior_labels = np.asarray(prior["label_recurrence"], dtype=np.int32)
    if len(prior_labels) != len(labels):
        raise ValueError("prior partition cohort mismatch")

    folds = double_holdout(metadata)
    models, predictions = outcome_models(metadata, core, folds)
    bootstrap_group = task_initial
    increments = {
        "events_over_task": bootstrap_increment(
            metadata["failure"],
            predictions["task"],
            predictions["task_events"],
            bootstrap_group,
            args.bootstrap_draws,
            args.seed + 500,
        ),
        "events_over_task_length": bootstrap_increment(
            metadata["failure"],
            predictions["task_length"],
            predictions["task_length_events"],
            bootstrap_group,
            args.bootstrap_draws,
            args.seed + 600,
        ),
    }
    profiles = cluster_profiles(labels, metadata, core, core_names)
    confounds = confound_prediction(metadata, core, folds)

    summary = {
        "schema": "himoe.route_change_events.v1",
        "source_manifest": str((args.cache_dir / "manifest.json").resolve()),
        "source_manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
        "cohort": {
            "episodes": len(core),
            "successes": int((~metadata["failure"]).sum()),
            "failures": int(metadata["failure"].sum()),
            "statistical_unit": "episode",
        },
        "event_definition": {
            "phase_anchors": N_PHASE,
            "peak_prominence_relative_threshold": PEAK_PROMINENCE_REL,
            "return_advantage_relative_threshold": RETURN_ADVANTAGE_REL,
            "low_speed_relative_threshold": LOW_SPEED_REL,
            "change_contrast_relative_threshold": CHANGE_CONTRAST_REL,
            "core_feature_names": core_names,
            "absolute_scale_sensitivity_names": scale_names,
            "raw_phase_trace_in_core": False,
            "task_length_outcome_in_core": False,
        },
        "clustering": cluster,
        "profiles": profiles,
        "posthoc_association": associations,
        "comparison_to_phase_recurrence": {
            "ari": float(adjusted_rand_score(prior_labels, labels)),
            "prior_k": int(len(np.unique(prior_labels))),
            "event_k": int(len(np.unique(labels))),
        },
        "outcome_protocol": {
            "folds": len(folds),
            "split": "4x4 task-initial-state and flow-seed double holdout",
            "episode_tested_once": True,
            "model": "fixed L2 logistic regression",
        },
        "outcome_models": models,
        "outcome_increments": increments,
        "confound_prediction": confounds,
        "absolute_scale_sensitivity": {
            "dimensions": int(scale_augmented.shape[1]),
            "note": "saved for downstream sensitivity; primary clustering uses normalized event core only",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_assignments(args.out_dir / "assignments.csv", metadata, labels, core, core_names)
    np.savez_compressed(
        args.out_dir / "event_features.npz",
        core=core,
        core_names=np.asarray(core_names),
        absolute_scale=scale,
        absolute_scale_names=np.asarray(scale_names),
        embedding=embedding.astype(np.float32),
        labels=labels.astype(np.int16),
        **{name: values for name, values in event_tapes.items()},
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (args.out_dir / "report.md").write_text(render_report(summary))
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
