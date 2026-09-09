#!/usr/bin/env python3
"""Cluster the normalized middle/late routing trajectories of all failures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import BoundaryNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from analyze_single_chunk_early_signal import router_features


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/failure-routing-clusters"
FAILURE_MODE_CSV = HERE / "analysis/replanning-reset-trap/episode_metrics.csv"

N_LAYERS = 8
N_DENOISE = 10
N_TOKENS = 11
N_EXPERTS = 32
TOP_K = 4
PRIMARY_START = 0.5
PRIMARY_END = 1.0
PRIMARY_BINS = 10
CANDIDATE_K = tuple(range(2, 7))
MINIMUM_CLUSTER_SIZE = 16
PRIMARY_VIEWS = ("geometry", "change", "recurrence")
ALL_VIEWS = PRIMARY_VIEWS + ("expert_occupancy",)
EXPECTED_FORMAL_RUNS = {
    "libero_goal/open_the_middle_drawer_of_the_cabinet": (0, 0),
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside": (42, 1260),
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove": (216, 11232),
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate": (
        12,
        264,
    ),
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate": (
        37,
        814,
    ),
}
EXPECTED_EPISODES_PER_RUN = 512
TASK_SHADOW_NMI = 0.50
TASK_SHADOW_CLUSTER_FRACTION = 0.90
CROSS_VIEW_ARI_MIN = 0.50
PHASE_SENSITIVITY_ARI_MIN = 0.50


@dataclass(frozen=True)
class PhaseConfig:
    name: str
    start: float
    end: float
    bins: int


PRIMARY_PHASE = PhaseConfig("primary", PRIMARY_START, PRIMARY_END, PRIMARY_BINS)
SENSITIVITY_PHASES = (
    PhaseConfig("grid8", 0.5, 1.0, 8),
    PhaseConfig("grid12", 0.5, 1.0, 12),
    PhaseConfig("late60", 0.6, 1.0, 10),
    PhaseConfig("truncate90", 0.5, 0.9, 10),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--subsamples", type=int, default=200)
    parser.add_argument("--association-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def discover_runs(cache_root: pathlib.Path) -> list[pathlib.Path]:
    runs = []
    for path in sorted(cache_root.glob("libero_*/*/right-16x32/client/summaries.json")):
        run = path.parents[1]
        if (run / "server/routes.zarr").exists():
            runs.append(run)
    if not runs:
        raise RuntimeError("no complete right-16x32 runs found")
    return runs


def task_key(run: pathlib.Path, cache_root: pathlib.Path) -> str:
    return str(run.relative_to(cache_root).parent)


def normalize_probabilities(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid router probabilities")
    low, high = float(mass.min()), float(mass.max())
    result /= mass
    return result, low, high


def phase_grid(config: PhaseConfig) -> np.ndarray:
    if not 0.0 <= config.start < config.end <= 1.0 or config.bins < 2:
        raise ValueError(f"invalid phase configuration: {config}")
    return np.linspace(config.start, config.end, config.bins, dtype=np.float64)


def phase_interpolate(values: np.ndarray, config: PhaseConfig) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim < 2 or len(values) < 2:
        raise ValueError("phase interpolation needs [query,...] with at least two queries")
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = phase_grid(config)
    upper = np.searchsorted(source, target, side="left")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    width = source[upper] - source[lower]
    alpha = ((target - source[lower]) / width).astype(np.float32)
    shape = (len(target),) + (1,) * (values.ndim - 1)
    return (
        values[lower] * (1.0 - alpha.reshape(shape))
        + values[upper] * alpha.reshape(shape)
    )


def selected_expert_occupancy(expert_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(expert_ids)
    expected = (len(ids), N_LAYERS, N_DENOISE, N_TOKENS, TOP_K)
    if ids.shape != expected:
        raise ValueError(f"unexpected expert-id shape {ids.shape}; expected {expected}")
    action = ids[:, :, :, 1:, :]
    flattened = action.transpose(0, 1, 2, 3, 4).reshape(
        len(ids) * N_LAYERS, N_DENOISE * (N_TOKENS - 1) * TOP_K
    )
    occupancy = np.zeros((len(flattened), N_EXPERTS), dtype=np.float32)
    rows = np.repeat(np.arange(len(flattened)), flattened.shape[1])
    np.add.at(occupancy, (rows, flattened.reshape(-1)), 1.0)
    occupancy /= float(flattened.shape[1])
    return occupancy.reshape(len(ids), N_LAYERS, N_EXPERTS)


def hellinger_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5
        * np.sum(
            np.square(np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))),
            axis=-1,
        )
    )


def query_tapes(
    probabilities: np.ndarray, expert_ids: np.ndarray
) -> dict[str, np.ndarray]:
    probabilities, _low, _high = normalize_probabilities(probabilities)
    geometry = router_features(probabilities.copy(), expert_ids)
    action = probabilities[:, :, :, 1:, :].mean(axis=3)
    action, _low, _high = normalize_probabilities(action)
    hard = selected_expert_occupancy(expert_ids)
    hard, _low, _high = normalize_probabilities(hard)
    return {"geometry": geometry, "action": action, "hard": hard}


def recurrence_features(action: np.ndarray, hard: np.ndarray) -> np.ndarray:
    root_action = np.sqrt(np.maximum(action, 0.0))
    root_hard = np.sqrt(np.maximum(hard, 0.0))
    rows = []
    for left in range(len(action) - 1):
        for right in range(left + 1, len(action)):
            soft_site = np.sqrt(
                0.5 * np.sum(np.square(root_action[left] - root_action[right]), axis=-1)
            )
            hard_layer = np.sqrt(
                0.5 * np.sum(np.square(root_hard[left] - root_hard[right]), axis=-1)
            )
            rows.append(
                np.concatenate(
                    [soft_site.mean(axis=1), soft_site.std(axis=1), hard_layer]
                )
            )
    return np.concatenate(rows).astype(np.float32, copy=False)


def build_representations(
    tapes: dict[str, np.ndarray], config: PhaseConfig
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    geometry = phase_interpolate(tapes["geometry"], config).astype(np.float32)
    action = phase_interpolate(tapes["action"], config).astype(np.float32)
    hard = phase_interpolate(tapes["hard"], config).astype(np.float32)
    action, _low, _high = normalize_probabilities(action)
    hard, _low, _high = normalize_probabilities(hard)

    spacing = (config.end - config.start) / (config.bins - 1)
    features = {
        "geometry": geometry.reshape(-1),
        "change": (np.diff(geometry, axis=0) / spacing).reshape(-1),
        "recurrence": recurrence_features(action, hard),
        "expert_occupancy": np.sqrt(hard).reshape(-1),
    }

    soft_adjacent = hellinger_distance(action[1:], action[:-1]).mean(axis=(1, 2))
    hard_adjacent = hellinger_distance(hard[1:], hard[:-1]).mean(axis=1)
    phase_distance = np.empty((config.bins, config.bins), dtype=np.float32)
    for left in range(config.bins):
        phase_distance[left] = hellinger_distance(
            action[left][None, ...], action
        ).mean(axis=(1, 2))
    backward = []
    for current in range(2, config.bins):
        backward.append(
            float(np.min(phase_distance[current, : current - 1]) < phase_distance[current, current - 1])
        )
    entropy = -np.sum(action * np.log(np.maximum(action, 1e-12)), axis=-1)
    diagnostics = {
        "soft_speed_curve": soft_adjacent,
        "hard_speed_curve": hard_adjacent,
        "mean_soft_speed": float(soft_adjacent.mean()),
        "late_soft_speed": float(soft_adjacent[-max(1, len(soft_adjacent) // 3) :].mean()),
        "mean_hard_speed": float(hard_adjacent.mean()),
        "middle_to_terminal_drift": float(phase_distance[0, -1]),
        "terminal_nonlocal_return_distance": float(phase_distance[-1, :-2].min()),
        "backward_return_fraction": float(np.mean(backward)) if backward else 0.0,
        "mean_action_route_entropy": float(entropy.mean()),
    }
    return features, diagnostics


def load_failure_modes(path: pathlib.Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        raise FileNotFoundError(f"failure-mode proxy table not found: {path}")
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [
        row
        for row in rows
        if row.get("failure") == "True"
        and row.get("search") == "lag2_8"
        and row.get("window") == "prefix"
    ]
    result = {
        (row["task"], int(row["episode"])): row["failure_mode"] for row in selected
    }
    if len(result) != len(selected):
        raise ValueError("duplicate failure-mode proxy keys")
    return result


def load_all_failures(
    cache_root: pathlib.Path,
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, dict[str, np.ndarray]],
    dict[str, Any],
]:
    metadata: list[dict[str, Any]] = []
    features: dict[str, list[np.ndarray]] = {name: [] for name in ALL_VIEWS}
    sensitivity: dict[str, dict[str, list[np.ndarray]]] = {
        view: {config.name: [] for config in SENSITIVITY_PHASES}
        for view in PRIMARY_VIEWS
    }
    diagnostics: dict[str, list[Any]] = {
        "soft_speed_curve": [],
        "hard_speed_curve": [],
        "mean_soft_speed": [],
        "late_soft_speed": [],
        "mean_hard_speed": [],
        "middle_to_terminal_drift": [],
        "terminal_nonlocal_return_distance": [],
        "backward_return_fraction": [],
        "mean_action_route_entropy": [],
    }
    probability_sum_min = float("inf")
    probability_sum_max = float("-inf")
    checkpoints = {}
    coverage = []

    runs = discover_runs(cache_root)
    discovered = {task_key(run, cache_root) for run in runs}
    if discovered != set(EXPECTED_FORMAL_RUNS):
        raise ValueError(
            "formal run set drifted: missing=%s extra=%s"
            % (
                sorted(set(EXPECTED_FORMAL_RUNS) - discovered),
                sorted(discovered - set(EXPECTED_FORMAL_RUNS)),
            )
        )

    for run in runs:
        key = task_key(run, cache_root)
        run_metadata = json.loads((run / "meta.json").read_text())
        sampling = run_metadata.get("sampling", {})
        if (
            run_metadata.get("status") != "complete"
            or not sampling.get("complete", False)
            or int(sampling.get("actual_episodes", -1)) != EXPECTED_EPISODES_PER_RUN
            or int(sampling.get("designed_episodes", -1)) != EXPECTED_EPISODES_PER_RUN
        ):
            raise ValueError(f"incomplete or unexpected formal run metadata for {key}")
        summaries = sorted(
            json.loads((run / "client/summaries.json").read_text()),
            key=lambda row: int(row["episode_index"]),
        )
        episode_ids = np.asarray(
            [int(row["episode_index"]) for row in summaries], dtype=np.int64
        )
        if len(summaries) != EXPECTED_EPISODES_PER_RUN or not np.array_equal(
            episode_ids, np.arange(EXPECTED_EPISODES_PER_RUN)
        ):
            raise ValueError(f"non-contiguous formal episode IDs for {key}")
        lengths = np.asarray([int(row["inference_calls"]) for row in summaries], np.int64)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        expected_episode = np.repeat(episode_ids, lengths)
        if not np.array_equal(np.asarray(route["episode_id"][:]), expected_episode):
            raise ValueError(f"episode alignment failed for {key}")
        if not np.array_equal(
            np.asarray(route["control_step"][:]), np.arange(int(lengths.sum()))
        ):
            raise ValueError(f"control-step alignment failed for {key}")
        server_metadata = json.loads((run / "client/server_metadata.json").read_text())
        checkpoints[key] = server_metadata.get("checkpoint_sha256")
        failures = [row for row in summaries if not bool(row["success"])]
        expected_failures, expected_queries = EXPECTED_FORMAL_RUNS[key]
        actual_failure_queries = sum(int(row["inference_calls"]) for row in failures)
        if len(failures) != expected_failures or actual_failure_queries != expected_queries:
            raise ValueError(
                f"formal failure cohort drifted for {key}: "
                f"{len(failures)} failures/{actual_failure_queries} queries"
            )
        coverage.append(
            {
                "task": key,
                "episodes": len(summaries),
                "failures": len(failures),
                "failure_lengths": sorted(
                    {int(row["inference_calls"]) for row in failures}
                ),
                "checkpoint_sha256": checkpoints[key],
            }
        )
        for row in failures:
            episode = int(row["episode_index"])
            start = int(offsets[episode])
            stop = start + int(lengths[episode])
            raw = np.asarray(
                route["hb_router_probs"][start:stop, :, :, :, :], np.float32
            )
            ids = np.asarray(
                route["hb_expert_ids"][start:stop, :, :, :, :], np.uint8
            )
            normalized, low, high = normalize_probabilities(raw)
            probability_sum_min = min(probability_sum_min, low)
            probability_sum_max = max(probability_sum_max, high)
            tapes = query_tapes(normalized, ids)
            primary, episode_diagnostics = build_representations(tapes, PRIMARY_PHASE)
            for name in ALL_VIEWS:
                features[name].append(primary[name])
            for name, value in episode_diagnostics.items():
                diagnostics[name].append(value)
            for config in SENSITIVITY_PHASES:
                variants, _unused = build_representations(tapes, config)
                for view in PRIMARY_VIEWS:
                    sensitivity[view][config.name].append(variants[view])
            metadata.append(
                {
                    "task": key,
                    "checkpoint_sha256": checkpoints[key],
                    "episode": episode,
                    "init_state_id": int(row["init_state_id"]),
                    "flow_noise_seed": int(row["flow_noise_seed"]),
                    "episode_length": int(row["inference_calls"]),
                }
            )
        print(f"loaded {key}: {len(failures)} failures", flush=True)

    feature_arrays = {name: np.stack(values) for name, values in features.items()}
    sensitivity_arrays = {
        view: {name: np.stack(values) for name, values in configurations.items()}
        for view, configurations in sensitivity.items()
    }
    diagnostic_arrays = {
        name: np.stack(values) if isinstance(values[0], np.ndarray) else np.asarray(values)
        for name, values in diagnostics.items()
    }
    audit = {
        "coverage": coverage,
        "checkpoints": checkpoints,
        "distinct_checkpoint_sha256": len(set(checkpoints.values())),
        "probability_sum_min": probability_sum_min,
        "probability_sum_max": probability_sum_max,
        "failures": len(metadata),
        "failure_queries": int(sum(row["episode_length"] for row in metadata)),
        "feature_shapes": {name: list(value.shape) for name, value in feature_arrays.items()},
    }
    return metadata, feature_arrays, sensitivity_arrays, {
        "audit": audit,
        "diagnostics": diagnostic_arrays,
    }


def attach_failure_modes(metadata: list[dict[str, Any]], path: pathlib.Path) -> int:
    modes = load_failure_modes(path)
    cohort_keys = {(row["task"], row["episode"]) for row in metadata}
    if set(modes) != cohort_keys:
        raise ValueError(
            "failure-mode proxy cohort mismatch: missing=%d extra=%d"
            % (len(cohort_keys - set(modes)), len(set(modes) - cohort_keys))
        )
    for row in metadata:
        row["failure_mode_proxy"] = modes[(row["task"], row["episode"])]
    return len(modes)


def prepare_embedding(
    matrix: np.ndarray, view: str, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=np.float64)
    standard_deviation = values.std(axis=0)
    keep = standard_deviation > 1e-9
    if not np.any(keep):
        raise ValueError(f"all features are constant in {view}")
    values = values[:, keep]
    if view == "expert_occupancy":
        center = values.mean(axis=0)
        scale = np.ones(values.shape[1])
    else:
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
        "pca_components": retained,
        "variance_explained": float(cumulative[retained - 1]),
        "scaling": "center_only" if view == "expert_occupancy" else "median_iqr",
    }


def ward_labels(embedding: np.ndarray, clusters: int) -> np.ndarray:
    if not 2 <= clusters < len(embedding):
        raise ValueError("invalid cluster count")
    tree = linkage(embedding, method="ward", optimal_ordering=False)
    labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
    if len(np.unique(labels)) != clusters:
        raise ValueError("Ward linkage returned fewer clusters than requested")
    return labels


def matched_cluster_jaccard(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = np.unique(reference)
    right = np.unique(candidate)
    scores = np.zeros((len(left), len(right)), dtype=np.float64)
    for i, a in enumerate(left):
        set_a = reference == a
        for j, b in enumerate(right):
            set_b = candidate == b
            scores[i, j] = np.sum(set_a & set_b) / np.sum(set_a | set_b)
    row, column = linear_sum_assignment(-scores)
    return float(scores[row, column].mean())


def canonicalize(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    ordered = sorted(
        np.unique(labels),
        key=lambda label: (float(np.mean(score[labels == label])), int(label)),
    )
    mapping = {int(label): index for index, label in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def cluster_view(
    matrix: np.ndarray,
    view: str,
    score: np.ndarray,
    subsamples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, view, seed)
    count = len(embedding)
    full_labels = {clusters: ward_labels(embedding, clusters) for clusters in CANDIDATE_K}
    trials: dict[int, dict[str, Any]] = {}
    trackers = {}
    for clusters, labels in full_labels.items():
        sizes = np.bincount(labels, minlength=clusters)
        trials[clusters] = {
            "clusters": clusters,
            "silhouette": float(silhouette_score(embedding, labels)),
            "sizes": sizes.tolist(),
            "minimum_size_pass": bool(sizes.min() >= MINIMUM_CLUSTER_SIZE),
        }
        trackers[clusters] = {
            "ari": [],
            "jaccard": [],
            "silhouette": [],
            "pair_count": np.zeros((count, count), dtype=np.uint32),
            "pair_same": np.zeros((count, count), dtype=np.uint32),
        }

    rng = np.random.default_rng(seed + 1000)
    sample_size = int(math.ceil(0.80 * count))
    for _draw in range(subsamples):
        index = np.sort(rng.choice(count, sample_size, replace=False))
        subset, _unused = prepare_embedding(
            matrix[index], view, seed + 100000 + _draw
        )
        tree = linkage(subset, method="ward", optimal_ordering=False)
        pair_index = np.ix_(index, index)
        for clusters in CANDIDATE_K:
            labels = fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
            if len(np.unique(labels)) != clusters:
                continue
            tracker = trackers[clusters]
            reference = full_labels[clusters][index]
            tracker["ari"].append(adjusted_rand_score(reference, labels))
            tracker["jaccard"].append(matched_cluster_jaccard(reference, labels))
            tracker["silhouette"].append(silhouette_score(subset, labels))
            tracker["pair_count"][pair_index] += 1
            tracker["pair_same"][pair_index] += (labels[:, None] == labels[None, :])

    for clusters in CANDIDATE_K:
        tracker = trackers[clusters]
        ari = np.asarray(tracker["ari"], dtype=np.float64)
        jaccard = np.asarray(tracker["jaccard"], dtype=np.float64)
        silhouettes = np.asarray(tracker["silhouette"], dtype=np.float64)
        off_diagonal = ~np.eye(count, dtype=bool)
        valid = (tracker["pair_count"] > 0) & off_diagonal
        consensus = tracker["pair_same"][valid] / tracker["pair_count"][valid]
        trial = trials[clusters]
        trial.update(
            {
                "subsamples_completed": int(len(ari)),
                "ari_median": float(np.median(ari)),
                "ari_p10": float(np.quantile(ari, 0.10)),
                "matched_jaccard_median": float(np.median(jaccard)),
                "matched_jaccard_p10": float(np.quantile(jaccard, 0.10)),
                "subsample_silhouette_mean": float(silhouettes.mean()),
                "subsample_silhouette_se": float(
                    silhouettes.std(ddof=1) / math.sqrt(len(silhouettes))
                ),
                "consensus_pac_01_09": float(
                    np.mean((consensus > 0.1) & (consensus < 0.9))
                ),
                "consensus_dispersion": float(
                    4.0 * np.mean(consensus * (1.0 - consensus))
                ),
            }
        )
        trial["stable"] = bool(
            trial["minimum_size_pass"]
            and trial["ari_median"] >= 0.75
            and trial["ari_p10"] >= 0.50
            and trial["consensus_pac_01_09"] <= 0.20
        )

    stable = [trials[k] for k in CANDIDATE_K if trials[k]["stable"]]
    if stable:
        best = max(stable, key=lambda row: row["subsample_silhouette_mean"])
        selected_k = int(best["clusters"])
        status = "stable_partition"
    else:
        selected_k = None
        status = "no_stable_partition"
    size_valid = [
        trials[k] for k in CANDIDATE_K if trials[k]["minimum_size_pass"]
    ]
    exploratory = max(
        size_valid if size_valid else list(trials.values()),
        key=lambda row: row["silhouette"],
    )["clusters"]
    reported_k = selected_k if selected_k is not None else exploratory
    labels = canonicalize(full_labels[reported_k], score)
    return (
        {
            "view": view,
            "status": status,
            "selected_k": selected_k,
            "exploratory_best_k": exploratory,
            "reported_k": reported_k,
            "minimum_cluster_size": MINIMUM_CLUSTER_SIZE,
            "preprocessing": preprocessing,
            "trials": [trials[k] for k in CANDIDATE_K],
        },
        labels,
        embedding,
    )


def contingency(categories: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = sorted({str(value) for value in categories})
    table = np.zeros((len(names), int(labels.max()) + 1), dtype=np.int64)
    for row, name in enumerate(names):
        for cluster in range(table.shape[1]):
            table[row, cluster] = int(np.sum((categories == name) & (labels == cluster)))
    return table, names


def cramers_v(categories: np.ndarray, labels: np.ndarray) -> float:
    table, _names = contingency(categories, labels)
    if min(table.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(math.sqrt((chi2 / table.sum()) / min(table.shape[0] - 1, table.shape[1] - 1)))


def purity(categories: np.ndarray, labels: np.ndarray) -> float:
    return float(
        sum(
            max(Counter(categories[labels == cluster]).values())
            for cluster in np.unique(labels)
        )
        / len(labels)
    )


def association_summary(
    categories: np.ndarray,
    labels: np.ndarray,
    strata: np.ndarray,
    permutations: int,
    seed: int,
    permutation: str,
) -> dict[str, Any]:
    encoded = np.asarray([str(value) for value in categories])
    observed = normalized_mutual_info_score(encoded, labels)
    rng = np.random.default_rng(seed)
    hits = 0
    null_values = np.empty(permutations, dtype=np.float64)
    for _draw in range(permutations):
        shuffled = encoded.copy()
        if permutation != "global":
            for stratum in np.unique(strata):
                index = np.flatnonzero(strata == stratum)
                shuffled[index] = shuffled[index][rng.permutation(len(index))]
        else:
            shuffled = shuffled[rng.permutation(len(shuffled))]
        null = normalized_mutual_info_score(shuffled, labels)
        null_values[_draw] = null
        hits += null >= observed - 1e-15
    return {
        "nmi": float(observed),
        "null_nmi_mean": float(null_values.mean()),
        "nmi_excess_over_null": float(observed - null_values.mean()),
        "ami": float(adjusted_mutual_info_score(encoded, labels)),
        "cramers_v": cramers_v(encoded, labels),
        "purity": purity(encoded, labels),
        "permutation": permutation,
        "permutation_p_one_sided": float((hits + 1) / (permutations + 1)),
    }


def posthoc_associations(
    metadata: list[dict[str, Any]],
    labels: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    tasks = np.asarray([row["task"] for row in metadata])
    task_initial_state = np.asarray(
        [f"{row['task']}::{row['init_state_id']}" for row in metadata]
    )
    values = {
        "task": np.asarray([row["task"] for row in metadata]),
        "checkpoint": np.asarray([row["checkpoint_sha256"] for row in metadata]),
        "episode_length": np.asarray(
            [str(row["episode_length"]) for row in metadata]
        ),
        "initial_state_within_task": task_initial_state,
        "flow_seed_within_task_initial_state": np.asarray(
            [f"{row['task']}::{row['flow_noise_seed']}" for row in metadata]
        ),
        "failure_mode_proxy": np.asarray(
            [row["failure_mode_proxy"] for row in metadata]
        ),
    }
    permutation = {
        "task": (tasks, "global"),
        "checkpoint": (tasks, "global"),
        "episode_length": (tasks, "global"),
        "initial_state_within_task": (tasks, "within_task"),
        "flow_seed_within_task_initial_state": (
            task_initial_state,
            "within_task_initial_state",
        ),
        "failure_mode_proxy": (
            task_initial_state,
            "within_task_initial_state",
        ),
    }
    return {
        name: association_summary(
            categories,
            labels,
            permutation[name][0],
            permutations,
            seed + index,
            permutation[name][1],
        )
        for index, (name, categories) in enumerate(values.items())
    }


def add_fdr_correction(results: dict[str, dict[str, Any]]) -> None:
    entries = [
        association
        for view in ALL_VIEWS
        for association in results[view]["posthoc_association"].values()
    ]
    p_values = np.asarray(
        [entry["permutation_p_one_sided"] for entry in entries], dtype=np.float64
    )
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1.0)
    for entry, value in zip(entries, q_values):
        entry["fdr_bh_q"] = float(value)


def annotate_interpretation(
    results: dict[str, dict[str, Any]],
    cross_view: dict[str, dict[str, float]],
    phase_sensitivity: list[dict[str, Any]],
) -> None:
    for view in ALL_VIEWS:
        result = results[view]
        flags = []
        if result["status"] != "stable_partition":
            flags.append("exploratory_only")
        task_nmi = result["posthoc_association"]["task"]["nmi"]
        dominant = max(row["dominant_task_fraction"] for row in result["profiles"])
        if task_nmi >= TASK_SHADOW_NMI or dominant >= TASK_SHADOW_CLUSTER_FRACTION:
            flags.append("task_shadowed")
        if view == "expert_occupancy":
            flags.append("secondary_expert_id_view")
        else:
            stable_peers = [
                other
                for other in PRIMARY_VIEWS
                if other != view and results[other]["status"] == "stable_partition"
            ]
            agreements = [
                cross_view[view][other] for other in stable_peers
            ]
            if not agreements:
                flags.append("no_stable_cross_view_replication")
            elif all(value < CROSS_VIEW_ARI_MIN for value in agreements):
                flags.append("representation_specific")
            phase_checks = [
                row
                for row in phase_sensitivity
                if row["view"] == view
            ]
            if any(
                row["ari_to_primary"] < PHASE_SENSITIVITY_ARI_MIN
                or not row["minimum_size_pass"]
                for row in phase_checks
            ):
                flags.append("phase_sensitive")
        result["interpretation_flags"] = flags
        result["interpretation_status"] = (
            "shared_taxonomy_candidate" if not flags else "+".join(flags)
        )


def cluster_profiles(
    labels: np.ndarray,
    metadata: list[dict[str, Any]],
    diagnostics: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    tasks = np.asarray([row["task"] for row in metadata])
    modes = np.asarray([row["failure_mode_proxy"] for row in metadata])
    checkpoints = np.asarray([row["checkpoint_sha256"] for row in metadata])
    lengths = np.asarray([row["episode_length"] for row in metadata])
    profiles = []
    scalar_names = [
        name for name, value in diagnostics.items() if np.asarray(value).ndim == 1
    ]
    for cluster in np.unique(labels):
        index = labels == cluster
        task_counts = Counter(tasks[index])
        mode_counts = Counter(modes[index])
        checkpoint_counts = Counter(checkpoints[index])
        length_counts = Counter(lengths[index])
        profiles.append(
            {
                "cluster": int(cluster),
                "episodes": int(index.sum()),
                "task_counts": dict(sorted(task_counts.items())),
                "failure_mode_proxy_counts": dict(sorted(mode_counts.items())),
                "checkpoint_counts": dict(sorted(checkpoint_counts.items())),
                "episode_length_counts": {
                    str(key): value for key, value in sorted(length_counts.items())
                },
                "dominant_task_fraction": float(max(task_counts.values()) / index.sum()),
                "task_initial_state_groups": len(
                    {
                        (metadata[i]["task"], metadata[i]["init_state_id"])
                        for i in np.flatnonzero(index)
                    }
                ),
                "flow_noise_seed_groups": len(
                    {
                        (metadata[i]["task"], metadata[i]["flow_noise_seed"])
                        for i in np.flatnonzero(index)
                    }
                ),
                "diagnostic_means": {
                    name: float(np.mean(diagnostics[name][index]))
                    for name in scalar_names
                },
            }
        )
    return profiles


def sensitivity_partition(
    matrix: np.ndarray,
    base_labels: np.ndarray,
    clusters: int,
    view: str,
    name: str,
    seed: int,
) -> dict[str, Any]:
    embedding, preprocessing = prepare_embedding(matrix, view, seed)
    labels = ward_labels(embedding, clusters)
    sizes = np.bincount(labels, minlength=clusters)
    best_k = None
    best_silhouette = float("-inf")
    for candidate in CANDIDATE_K:
        candidate_labels = ward_labels(embedding, candidate)
        candidate_sizes = np.bincount(candidate_labels, minlength=candidate)
        if candidate_sizes.min() < MINIMUM_CLUSTER_SIZE:
            continue
        value = float(silhouette_score(embedding, candidate_labels))
        if value > best_silhouette:
            best_k, best_silhouette = candidate, value
    return {
        "view": view,
        "configuration": name,
        "fixed_k": clusters,
        "ari_to_primary": float(adjusted_rand_score(base_labels, labels)),
        "fixed_k_silhouette": float(silhouette_score(embedding, labels)),
        "fixed_k_sizes": sizes.tolist(),
        "minimum_size_pass": bool(sizes.min() >= MINIMUM_CLUSTER_SIZE),
        "size_valid_best_k": best_k,
        "size_valid_best_silhouette": best_silhouette if best_k is not None else None,
        "preprocessing": preprocessing,
    }


def write_assignments(
    path: pathlib.Path,
    metadata: list[dict[str, Any]],
    diagnostics: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
) -> None:
    scalar_names = [
        name for name, value in diagnostics.items() if np.asarray(value).ndim == 1
    ]
    rows = []
    for index, item in enumerate(metadata):
        row = dict(item)
        for name in scalar_names:
            row[name] = float(diagnostics[name][index])
        for view in ALL_VIEWS:
            row[f"{view}_cluster"] = int(labels[view][index])
            row[f"{view}_partition_status"] = results[view]["status"]
            row[f"{view}_interpretation_status"] = results[view][
                "interpretation_status"
            ]
        rows.append(row)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_selection(path: pathlib.Path, results: dict[str, dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, view in zip(axes.ravel(), ALL_VIEWS):
        trials = results[view]["trials"]
        k = [row["clusters"] for row in trials]
        axis.plot(
            k,
            [row["subsample_silhouette_mean"] for row in trials],
            "o-",
            color="C0",
            label="mean subsample silhouette",
        )
        axis.plot(
            k, [row["ari_median"] for row in trials], "s-", color="C1", label="median ARI"
        )
        axis.plot(k, [row["ari_p10"] for row in trials], "^-", color="C2", label="ARI p10")
        axis.plot(
            k,
            [row["consensus_pac_01_09"] for row in trials],
            "x-",
            color="C3",
            label="PAC",
        )
        axis.axhline(0.75, color="C1", linestyle=":", linewidth=1)
        axis.axhline(0.50, color="C2", linestyle=":", linewidth=1)
        axis.axhline(0.20, color="C3", linestyle=":", linewidth=1)
        axis.set_title(view.replace("_", " "))
        axis.set_xlabel("K")
        axis.set_xticks(k)
        axis.set_ylim(-0.05, 1.05)
    axes[0, 0].legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def cluster_color_scale(clusters: int) -> tuple[Any, BoundaryNorm]:
    cmap = plt.get_cmap("tab10", clusters)
    boundaries = np.arange(-0.5, clusters + 0.5, 1.0)
    return cmap, BoundaryNorm(boundaries, cmap.N)


def plot_embeddings(
    path: pathlib.Path,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    metadata: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> None:
    tasks = np.asarray([row["task"] for row in metadata])
    task_names = sorted(set(tasks))
    markers = ["o", "s", "^", "D"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for axis, view in zip(axes.ravel(), ALL_VIEWS):
        embedding = embeddings[view]
        cmap, norm = cluster_color_scale(results[view]["reported_k"])
        for task, marker in zip(task_names, markers):
            index = tasks == task
            axis.scatter(
                embedding[index, 0],
                embedding[index, 1],
                c=labels[view][index],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=24,
                alpha=0.8,
                edgecolors="none",
                label=task.split("/", 1)[1],
            )
        qualifier = "stable" if results[view]["selected_k"] is not None else "exploratory"
        axis.set_title(
            f"{view.replace('_', ' ')} ({qualifier} K={results[view]['reported_k']})"
        )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    axes[0, 0].legend(fontsize=6, loc="best")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_speed_profiles(
    path: pathlib.Path,
    labels: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    results: dict[str, dict[str, Any]],
) -> None:
    phase = np.linspace(PRIMARY_START, PRIMARY_END, PRIMARY_BINS)
    midpoint = 0.5 * (phase[1:] + phase[:-1])
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for axis, view in zip(axes, PRIMARY_VIEWS):
        for cluster in np.unique(labels[view]):
            index = labels[view] == cluster
            curve = diagnostics["soft_speed_curve"][index].mean(axis=0)
            axis.plot(midpoint, curve, marker="o", label=f"C{cluster} n={index.sum()}")
        qualifier = "stable" if results[view]["selected_k"] is not None else "exploratory"
        axis.set_title(f"{view} ({qualifier} K={results[view]['reported_k']})")
        axis.set_xlabel("normalized rollout phase")
        axis.set_ylabel("soft route Hellinger speed")
        axis.legend(fontsize=7)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Normalized middle/late routing clusters of all failed rollouts",
        "",
        "Run class: `%s`." % summary["provenance"]["run_class"],
        "",
        "The statistical unit is one failed rollout. Each rollout is independently",
        "mapped from relative phase 0.5 to 1.0 onto ten fixed anchors; absolute query",
        "index and episode length are absent from every clustering representation.",
        "",
        "## Coverage",
        "",
        "| task | failures | failure lengths | checkpoint |",
        "|---|---:|---|---|",
    ]
    for row in summary["data_audit"]["coverage"]:
        lines.append(
            "| %s | %d | %s | `%s` |"
            % (
                row["task"],
                row["failures"],
                ",".join(map(str, row["failure_lengths"])) or "none",
                (row["checkpoint_sha256"] or "unknown")[:12],
            )
        )
    lines.extend(
        [
            "",
            "Total: %d failed rollouts and %d recorded queries. There are %d distinct checkpoints."
            % (
                summary["data_audit"]["failures"],
                summary["data_audit"]["failure_queries"],
                summary["data_audit"]["distinct_checkpoint_sha256"],
            ),
            "",
            "## How many clusters?",
            "",
            "`selected K` is reported only when all frozen stability gates pass. When it",
            "does not, `exploratory K` is the best size-valid silhouette partition.",
            "",
            "| view | role | cluster status | interpretation | selected K | exploratory K | mean subsample silhouette | median ARI | ARI p10 | PAC | sizes | task NMI | mode NMI excess |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for view in ALL_VIEWS:
        result = summary["views"][view]
        trial = next(
            row for row in result["trials"] if row["clusters"] == result["reported_k"]
        )
        role = "primary invariant" if view in PRIMARY_VIEWS else "secondary ID-based"
        lines.append(
            "| %s | %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s |"
            % (
                view,
                role,
                result["status"],
                result["interpretation_status"],
                result["selected_k"] if result["selected_k"] is not None else "NA",
                result["exploratory_best_k"],
                fmt(trial["subsample_silhouette_mean"]),
                fmt(trial["ari_median"]),
                fmt(trial["ari_p10"]),
                fmt(trial["consensus_pac_01_09"]),
                "/".join(map(str, trial["sizes"])),
                fmt(result["posthoc_association"]["task"]["nmi"]),
                fmt(
                    result["posthoc_association"]["failure_mode_proxy"][
                        "nmi_excess_over_null"
                    ]
                ),
            )
        )

    lines.extend(["", "## Cross-view agreement", ""])
    lines.append("ARI compares each view's reported partition; low values mean there is no single taxonomy.")
    lines.extend(["", "| view | " + " | ".join(ALL_VIEWS) + " |", "|---|" + "---:|" * len(ALL_VIEWS)])
    for left in ALL_VIEWS:
        values = [fmt(summary["cross_view_ari"][left][right]) for right in ALL_VIEWS]
        lines.append("| %s | %s |" % (left, " | ".join(values)))

    lines.extend(
        [
            "",
            "## Post-cluster associations",
            "",
            "These variables were revealed only after all route-only labels were frozen.",
            "The failure-mode p-value uses task x initial-state permutations.",
            "",
            "| view | factor | NMI | null NMI | excess | permutation p | BH q | permutation |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for view in ALL_VIEWS:
        for factor, association in summary["views"][view][
            "posthoc_association"
        ].items():
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    view,
                    factor,
                    fmt(association["nmi"]),
                    fmt(association["null_nmi_mean"]),
                    fmt(association["nmi_excess_over_null"]),
                    fmt(association["permutation_p_one_sided"]),
                    fmt(association["fdr_bh_q"]),
                    association["permutation"],
                )
            )

    lines.extend(["", "## Cluster composition", ""])
    for view in ALL_VIEWS:
        result = summary["views"][view]
        lines.extend(
            [
                "### %s (%s, reported K=%d)"
                % (view, result["interpretation_status"], result["reported_k"]),
                "",
                "| cluster | n | dominant task fraction | task x init groups | task counts | length counts | checkpoints | failure-mode proxy counts | mean speed | late speed | drift | backward return |",
                "|---:|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in result["profiles"]:
            diagnostic = row["diagnostic_means"]
            lines.append(
                "| C%d | %d | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    row["cluster"],
                    row["episodes"],
                    fmt(row["dominant_task_fraction"]),
                    row["task_initial_state_groups"],
                    "; ".join(f"{k.split('/', 1)[1]}={v}" for k, v in row["task_counts"].items()),
                    "; ".join(f"{k}={v}" for k, v in row["episode_length_counts"].items()),
                    "; ".join(f"{k[:8]}={v}" for k, v in row["checkpoint_counts"].items()),
                    "; ".join(f"{k}={v}" for k, v in row["failure_mode_proxy_counts"].items()),
                    fmt(diagnostic["mean_soft_speed"]),
                    fmt(diagnostic["late_soft_speed"]),
                    fmt(diagnostic["middle_to_terminal_drift"]),
                    fmt(diagnostic["backward_return_fraction"]),
                )
            )

    lines.extend(
        [
            "",
            "## Relative-phase sensitivity",
            "",
            "Each primary view is re-fit at its reported K after changing only the",
            "relative phase window/grid.",
            "",
            "| view | configuration | fixed K | ARI to primary | silhouette | sizes | min-size pass | own best size-valid K |",
            "|---|---|---:|---:|---:|---|---|---:|",
        ]
    )
    for row in summary["phase_sensitivity"]:
        lines.append(
            "| %s | %s | %d | %s | %s | %s | %s | %s |"
            % (
                row["view"],
                row["configuration"],
                row["fixed_k"],
                fmt(row["ari_to_primary"]),
                fmt(row["fixed_k_silhouette"]),
                "/".join(map(str, row["fixed_k_sizes"])),
                "yes" if row["minimum_size_pass"] else "no",
                row["size_valid_best_k"] if row["size_valid_best_k"] is not None else "NA",
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Phase normalization removes absolute chunk position from the representation. Lengths 22/30/52 remain perfectly nested within task here, so horizon and task effects are not separately identifiable.",
            "- Geometry/change/recurrence are expert-permutation invariant. Expert occupancy is a checkpoint-shadow diagnostic only.",
            "- Failure-mode labels are external kinematic proxies loaded after clustering, not ground truth and not cluster inputs.",
            "- Stability is conditional on ordinary episode subsampling; it does not establish generalization to held-out tasks, initial-state groups, or seeds.",
            "- Rows marked exploratory_only failed the frozen stability gate; their labels and profiles are descriptive rather than a supported taxonomy.",
            "- A stable routing regime is an association, not evidence that routing caused the failure.",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    values = np.arange(6, dtype=np.float32)[:, None]
    interpolated = phase_interpolate(values, PhaseConfig("test", 0.0, 1.0, 3))
    assert np.allclose(interpolated[:, 0], [0.0, 2.5, 5.0])
    ids = np.zeros((2, N_LAYERS, N_DENOISE, N_TOKENS, TOP_K), dtype=np.uint8)
    ids[..., 0] = 0
    ids[..., 1] = 1
    ids[..., 2] = 2
    ids[..., 3] = 3
    occupancy = selected_expert_occupancy(ids)
    assert occupancy.shape == (2, N_LAYERS, N_EXPERTS)
    assert np.allclose(occupancy[..., :4], 0.25)
    assert np.allclose(occupancy.sum(axis=-1), 1.0)

    rng = np.random.default_rng(1)
    action = rng.random((4, N_LAYERS, N_DENOISE, N_EXPERTS), dtype=np.float32)
    action /= action.sum(axis=-1, keepdims=True)
    hard = rng.random((4, N_LAYERS, N_EXPERTS), dtype=np.float32)
    hard /= hard.sum(axis=-1, keepdims=True)
    permutation = rng.permutation(N_EXPERTS)
    original = recurrence_features(action, hard)
    permuted = recurrence_features(action[..., permutation], hard[..., permutation])
    assert np.allclose(original, permuted, atol=1e-6)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.subsamples < 10 or args.association_permutations < 100:
        raise ValueError("too few stability/permutation draws")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    method_source = OUT_DIR / "METHOD.md"
    method_target = args.out_dir / "METHOD.md"
    if method_source.resolve() != method_target.resolve():
        shutil.copyfile(method_source, method_target)
    formal_run = bool(
        args.subsamples == 200
        and args.association_permutations == 5000
        and args.cache_root.resolve() == CACHE_ROOT.resolve()
    )

    metadata, features, sensitivity_features, bundle = load_all_failures(args.cache_root)
    diagnostics = bundle["diagnostics"]
    audit = bundle["audit"]
    if len(metadata) != 307:
        raise ValueError(f"expected 307 failures in the formal Hub runs, found {len(metadata)}")

    results: dict[str, dict[str, Any]] = {}
    labels: dict[str, np.ndarray] = {}
    embeddings: dict[str, np.ndarray] = {}
    for view_index, view in enumerate(ALL_VIEWS):
        print(f"clustering {view}", flush=True)
        result, view_labels, embedding = cluster_view(
            features[view],
            view,
            diagnostics["mean_soft_speed"],
            args.subsamples,
            args.seed + 10000 * view_index,
        )
        results[view] = result
        labels[view] = view_labels
        embeddings[view] = embedding

    cross_view = {
        left: {
            right: float(adjusted_rand_score(labels[left], labels[right]))
            for right in ALL_VIEWS
        }
        for left in ALL_VIEWS
    }
    phase_sensitivity = []
    for view_index, view in enumerate(PRIMARY_VIEWS):
        for config_index, config in enumerate(SENSITIVITY_PHASES):
            phase_sensitivity.append(
                sensitivity_partition(
                    sensitivity_features[view][config.name],
                    labels[view],
                    results[view]["reported_k"],
                    view,
                    config.name,
                    args.seed + 40000 + 1000 * view_index + config_index,
                )
            )

    audit["failure_mode_proxy_matches"] = attach_failure_modes(
        metadata, FAILURE_MODE_CSV
    )
    for view_index, view in enumerate(ALL_VIEWS):
        results[view]["posthoc_association"] = posthoc_associations(
            metadata,
            labels[view],
            args.association_permutations,
            args.seed + 50000 + 1000 * view_index,
        )
        results[view]["profiles"] = cluster_profiles(
            labels[view], metadata, diagnostics
        )
    add_fdr_correction(results)
    annotate_interpretation(results, cross_view, phase_sensitivity)

    summary = {
        "schema": "himoe.failure_routing_clusters.v2",
        "method": str(method_target),
        "provenance": {
            "run_class": "formal" if formal_run else "nonformal",
            "cache_root": str(args.cache_root.resolve()),
            "failure_mode_source": str(FAILURE_MODE_CSV.resolve()),
            "seed": args.seed,
            "subsamples": args.subsamples,
            "association_permutations": args.association_permutations,
            "analysis_script_sha256": sha256_file(pathlib.Path(__file__)),
            "method_sha256": sha256_file(method_target),
            "failure_mode_source_sha256": sha256_file(FAILURE_MODE_CSV),
        },
        "scope": {
            "outcome": "all terminal failures in complete right-16x32 LIBERO runs",
            "episode_is_statistical_unit": True,
            "primary_phase": {
                "coordinate": "query_index/(episode_queries-1)",
                "start": PRIMARY_START,
                "end": PRIMARY_END,
                "anchors": PRIMARY_BINS,
            },
            "absolute_chunk_used_as_feature": False,
            "task_used_as_feature": False,
        },
        "stability_protocol": {
            "candidate_k": list(CANDIDATE_K),
            "minimum_cluster_size": MINIMUM_CLUSTER_SIZE,
            "subsamples": args.subsamples,
            "subsample_fraction": 0.80,
            "preprocessing_refit_within_each_subsample": True,
            "resampling_unit": "episode",
            "selection_rule": "highest_mean_subsample_silhouette_among_stable_k",
            "stable_requirements": {
                "ari_median_min": 0.75,
                "ari_p10_min": 0.50,
                "pac_max": 0.20,
            },
            "association_permutations": args.association_permutations,
        },
        "interpretation_gates": {
            "task_nmi_shadow_min": TASK_SHADOW_NMI,
            "single_cluster_task_fraction_shadow_min": TASK_SHADOW_CLUSTER_FRACTION,
            "cross_primary_view_ari_min": CROSS_VIEW_ARI_MIN,
            "phase_sensitivity_ari_min": PHASE_SENSITIVITY_ARI_MIN,
        },
        "data_audit": audit,
        "views": results,
        "cross_view_ari": cross_view,
        "phase_sensitivity": phase_sensitivity,
    }
    write_assignments(
        args.out_dir / "assignments.csv", metadata, diagnostics, labels, results
    )
    np.savez_compressed(
        args.out_dir / "embeddings_and_labels.npz",
        **{f"embedding_{name}": value.astype(np.float32) for name, value in embeddings.items()},
        **{f"label_{name}": value.astype(np.int16) for name, value in labels.items()},
        soft_speed_curve=diagnostics["soft_speed_curve"].astype(np.float32),
        hard_speed_curve=diagnostics["hard_speed_curve"].astype(np.float32),
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    (args.out_dir / "report.md").write_text(render_report(summary))
    plot_selection(args.out_dir / "model_selection.png", results)
    plot_embeddings(
        args.out_dir / "embeddings.png", embeddings, labels, metadata, results
    )
    plot_speed_profiles(
        args.out_dir / "cluster_speed_profiles.png", labels, diagnostics, results
    )
    completion = {
        "schema": "himoe.failure_routing_clusters.completion.v1",
        "run_class": "formal" if formal_run else "nonformal",
        "summary_sha256": sha256_file(args.out_dir / "summary.json"),
        "report_sha256": sha256_file(args.out_dir / "report.md"),
        "expected_outputs": [
            "METHOD.md",
            "assignments.csv",
            "embeddings_and_labels.npz",
            "summary.json",
            "report.md",
            "model_selection.png",
            "embeddings.png",
            "cluster_speed_profiles.png",
        ],
    }
    (args.out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2, allow_nan=False)
    )
    print(f"wrote {args.out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
