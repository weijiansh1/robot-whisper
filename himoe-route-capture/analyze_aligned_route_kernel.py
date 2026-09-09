#!/usr/bin/env python3
"""Cluster aligned route-signature trajectories through landmark distances."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)
from threadpoolctl import threadpool_limits

from analyze_failure_routing_clusters import (
    association_summary,
    canonicalize,
    matched_cluster_jaccard,
    prepare_embedding,
    sha256_file,
)


HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE / "analysis/all-outcome-routing-clusters"
OUT_DIR = HERE / "analysis/aligned-route-kernel"
METHOD_SOURCE = OUT_DIR / "METHOD.md"

SEED = 20260829
ANCHORS = 10
SIGNATURE_DIMENSIONS = 320
SIGNATURE_COMPONENTS = 32
LANDMARKS = 256
MAX_SHIFT = 2
CANDIDATE_K = tuple(range(2, 9))
SUBSAMPLES = 50
PERMUTATIONS = 5000
PROBE_EPISODES = 512
SILHOUETTE_SAMPLE = 1000
SUBSAMPLE_SILHOUETTE_SAMPLE = 512
VARIANTS = ("raw", "task_residual", "task_init_residual")
PRIMARY_VARIANT = "task_residual"
OUTCOME_EXCESS_MIN = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=pathlib.Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--subsamples", type=int, default=SUBSAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def minimum_cluster_size(count: int) -> int:
    return max(16, int(math.ceil(0.02 * count)))


def _load_source(
    source_dir: pathlib.Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    completion = json.loads((source_dir / "completion.json").read_text())
    if completion.get("run_class") != "formal":
        raise ValueError("aligned-kernel source must be a formal all-outcome run")
    expected_hash = completion["output_sha256"].get("feature_cache_manifest.json")
    manifest_path = source_dir / "feature_cache_manifest.json"
    if expected_hash != sha256_file(manifest_path):
        raise ValueError("source feature manifest is not completion-verified")
    manifest = json.loads(manifest_path.read_text())
    needed = (
        "feature_primary_geometry",
        "diagnostic_mean_soft_speed",
        "meta_task",
        "meta_checkpoint_sha256",
        "meta_episode",
        "meta_init_state_id",
        "meta_flow_noise_seed",
        "meta_episode_length",
        "meta_failure",
    )
    arrays: dict[str, np.ndarray] = {}
    cache_dir = source_dir / "feature_cache"
    for name in needed:
        item = manifest["arrays"][name]
        path = cache_dir / item["file"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"source feature hash mismatch: {name}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(values.shape) != item["shape"] or str(values.dtype) != item["dtype"]:
            raise ValueError(f"source feature metadata mismatch: {name}")
        arrays[name] = values
    return arrays, manifest, completion


def _metadata(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("meta_"): np.asarray(values)
        for name, values in arrays.items()
        if name.startswith("meta_")
    }


def task_initial_state(metadata: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            f"{task}::{initial}"
            for task, initial in zip(metadata["task"], metadata["init_state_id"])
        ]
    )


def build_anchor_signatures(
    geometry: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    count = len(geometry)
    expected = (count, ANCHORS * SIGNATURE_DIMENSIONS)
    if geometry.shape != expected:
        raise ValueError(f"unexpected geometry shape {geometry.shape}; expected {expected}")
    values = np.asarray(geometry, dtype=np.float64).reshape(-1, SIGNATURE_DIMENSIONS)
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
    components = min(SIGNATURE_COMPONENTS, standardized.shape[1])
    model = PCA(
        n_components=components, svd_solver="randomized", random_state=seed
    )
    transformed = model.fit_transform(standardized)
    sequences = transformed.reshape(count, ANCHORS, components).astype(np.float32)
    return sequences, {
        "raw_anchor_dimensions": int(geometry.shape[1] // ANCHORS),
        "active_anchor_dimensions": int(keep.sum()),
        "signature_components": components,
        "signature_variance_explained": float(model.explained_variance_ratio_.sum()),
        "scaling": "median_iqr",
    }


def residualize_sequences(
    sequences: np.ndarray, metadata: dict[str, np.ndarray], variant: str
) -> np.ndarray:
    result = np.asarray(sequences, dtype=np.float32).copy()
    if variant == "raw":
        return result
    if variant == "task_residual":
        groups = metadata["task"]
    elif variant == "task_init_residual":
        groups = task_initial_state(metadata)
    else:
        raise ValueError(f"unknown trajectory variant: {variant}")
    for group in np.unique(groups):
        index = groups == group
        result[index] -= np.median(result[index], axis=0)
    return result


def normalize_trajectory_shape(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float32)
    values = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(values.reshape(len(values), -1), axis=1)
    if np.any(norm <= 1e-12):
        raise ValueError("constant trajectory after shape centering")
    return values / norm[:, None, None]


def aligned_landmark_distances(
    sequences: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    flattened = sequences.reshape(len(sequences), -1)
    _centers, landmark_index = kmeans_plusplus(
        flattened, n_clusters=min(LANDMARKS, len(sequences)), random_state=seed
    )
    landmarks = sequences[landmark_index]
    similarities = np.full(
        (len(sequences), len(landmarks)), -np.inf, dtype=np.float32
    )
    length = sequences.shape[1]
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        if shift >= 0:
            left = sequences[:, shift:]
            right = landmarks[:, : length - shift]
        else:
            left = sequences[:, : length + shift]
            right = landmarks[:, -shift:]
        left = left.reshape(len(left), -1)
        right = right.reshape(len(right), -1)
        left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        similarities = np.maximum(similarities, left @ right.T)
    similarities = np.clip(similarities, -1.0, 1.0)
    distances = (1.0 - similarities).astype(np.float32)
    return distances, landmark_index.astype(np.int32), {
        "landmarks": int(len(landmark_index)),
        "maximum_integer_shift": MAX_SHIFT,
        "distance_min": float(distances.min()),
        "distance_median": float(np.median(distances)),
        "distance_max": float(distances.max()),
    }


def _sampled_silhouette(
    embedding: np.ndarray, labels: np.ndarray, index: np.ndarray
) -> float:
    sample_labels = labels[index]
    if len(np.unique(sample_labels)) == len(np.unique(labels)):
        return float(silhouette_score(embedding[index], sample_labels))
    return float(silhouette_score(embedding, labels))


def cluster_landmark_view(
    matrix: np.ndarray,
    score: np.ndarray,
    subsamples: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    embedding, preprocessing = prepare_embedding(matrix, "geometry", seed)
    count = len(embedding)
    minimum = minimum_cluster_size(count)
    tree = linkage(embedding, method="ward", optimal_ordering=False)
    full_labels = {
        clusters: fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1
        for clusters in CANDIDATE_K
    }
    rng = np.random.default_rng(seed + 1000)
    probe = np.sort(rng.choice(count, min(PROBE_EPISODES, count), replace=False))
    full_sample = np.sort(
        rng.choice(count, min(SILHOUETTE_SAMPLE, count), replace=False)
    )
    trials: dict[int, dict[str, Any]] = {}
    trackers: dict[int, dict[str, Any]] = {}
    for clusters, labels in full_labels.items():
        if len(np.unique(labels)) != clusters:
            raise RuntimeError(f"Ward returned fewer than {clusters} clusters")
        sizes = np.bincount(labels, minlength=clusters)
        trials[clusters] = {
            "clusters": clusters,
            "full_silhouette_sampled": _sampled_silhouette(
                embedding, labels, full_sample
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
        subset, _unused = prepare_embedding(matrix[index], "geometry", seed + 100000 + draw)
        subset_tree = linkage(subset, method="ward", optimal_ordering=False)
        silhouette_index = np.sort(
            rng.choice(
                len(subset),
                min(SUBSAMPLE_SILHOUETTE_SAMPLE, len(subset)),
                replace=False,
            )
        )
        probe_mask = np.isin(probe, index)
        probe_position = np.flatnonzero(probe_mask)
        subset_probe_position = np.searchsorted(index, probe[probe_mask])
        pair_index = np.ix_(probe_position, probe_position)
        for clusters in CANDIDATE_K:
            labels = (
                fcluster(subset_tree, clusters, criterion="maxclust").astype(np.int32)
                - 1
            )
            if len(np.unique(labels)) != clusters:
                raise RuntimeError(f"incomplete Ward cut for K={clusters}")
            tracker = trackers[clusters]
            reference = full_labels[clusters][index]
            tracker["ari"].append(adjusted_rand_score(reference, labels))
            tracker["jaccard"].append(matched_cluster_jaccard(reference, labels))
            tracker["silhouette"].append(
                _sampled_silhouette(subset, labels, silhouette_index)
            )
            probe_labels = labels[subset_probe_position]
            tracker["pair_count"][pair_index] += 1
            tracker["pair_same"][pair_index] += (
                probe_labels[:, None] == probe_labels[None, :]
            )

    for clusters in CANDIDATE_K:
        tracker = trackers[clusters]
        ari = np.asarray(tracker["ari"], dtype=np.float64)
        jaccard = np.asarray(tracker["jaccard"], dtype=np.float64)
        silhouettes = np.asarray(tracker["silhouette"], dtype=np.float64)
        valid = (tracker["pair_count"] > 0) & ~np.eye(len(probe), dtype=bool)
        consensus = tracker["pair_same"][valid] / tracker["pair_count"][valid]
        trial = trials[clusters]
        trial.update(
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
        trial["stable"] = bool(
            trial["minimum_size_pass"]
            and trial["ari_median"] >= 0.75
            and trial["ari_p10"] >= 0.50
            and trial["consensus_pac_01_09"] <= 0.20
        )

    stable = [trials[k] for k in CANDIDATE_K if trials[k]["stable"]]
    if stable:
        selected = max(stable, key=lambda row: row["subsample_silhouette_mean"])
        selected_k: int | None = int(selected["clusters"])
        status = "stable_partition"
    else:
        selected_k = None
        status = "no_stable_partition"
    size_valid = [trials[k] for k in CANDIDATE_K if trials[k]["minimum_size_pass"]]
    exploratory = max(
        size_valid if size_valid else list(trials.values()),
        key=lambda row: row["full_silhouette_sampled"],
    )["clusters"]
    reported_k = selected_k if selected_k is not None else int(exploratory)
    labels = canonicalize(full_labels[reported_k], score)
    return (
        {
            "status": status,
            "selected_k": selected_k,
            "exploratory_best_k": int(exploratory),
            "reported_k": reported_k,
            "reported_sizes": np.bincount(labels, minlength=reported_k).tolist(),
            "minimum_cluster_size": minimum,
            "preprocessing": preprocessing,
            "trials": [trials[k] for k in CANDIDATE_K],
        },
        labels,
        embedding,
    )


def association_bundle(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    task = metadata["task"]
    task_init = task_initial_state(metadata)
    outcome = np.where(metadata["failure"], "failure", "success")
    inputs = {
        "outcome_within_task_initial_state": (
            outcome,
            task_init,
            "within_task_initial_state",
        ),
        "task": (task, task, "global"),
        "episode_length": (metadata["episode_length"].astype(str), task, "global"),
        "initial_state_within_task": (task_init, task, "within_task"),
    }
    return {
        name: association_summary(
            categories,
            labels,
            strata,
            permutations,
            seed + offset,
            permutation,
        )
        for offset, (name, (categories, strata, permutation)) in enumerate(
            inputs.items()
        )
    }


def per_task_associations(
    labels: np.ndarray,
    metadata: dict[str, np.ndarray],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, task in enumerate(np.unique(metadata["task"])):
        index = metadata["task"] == task
        outcome = np.where(metadata["failure"][index], "failure", "success")
        if len(np.unique(outcome)) < 2:
            result[str(task)] = {
                "status": "no_outcome_variation",
                "episodes": int(index.sum()),
            }
            continue
        _values, local_labels = np.unique(labels[index], return_inverse=True)
        result[str(task)] = {
            "status": "estimated",
            "episodes": int(index.sum()),
            "failures": int(metadata["failure"][index].sum()),
            **association_summary(
                outcome,
                local_labels,
                metadata["init_state_id"][index].astype(str),
                permutations,
                seed + offset,
                "within_initial_state",
            ),
        }
    return result


def add_bh(entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
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


def cluster_profiles(
    labels: np.ndarray, metadata: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    profiles = []
    tasks = metadata["task"]
    failure = metadata["failure"]
    lengths = metadata["episode_length"]
    mixed_tasks = {
        str(task)
        for task in np.unique(tasks)
        if 0 < int(failure[tasks == task].sum()) < int((tasks == task).sum())
    }
    baselines = {
        task: float(failure[tasks == task].mean()) for task in mixed_tasks
    }
    for cluster in np.unique(labels):
        index = labels == cluster
        task_rows = []
        for task in np.unique(tasks):
            cell = index & (tasks == task)
            if not cell.any():
                continue
            failures = int(failure[cell].sum())
            row = {
                "task": str(task),
                "episodes": int(cell.sum()),
                "failures": failures,
                "failure_rate": float(failures / cell.sum()),
            }
            if str(task) in baselines:
                row["task_baseline_failure_rate"] = baselines[str(task)]
                row["failure_rate_delta"] = row["failure_rate"] - baselines[str(task)]
            task_rows.append(row)
        eligible = [
            row
            for row in task_rows
            if row["episodes"] >= 10 and "failure_rate_delta" in row
        ]
        failures = int(failure[index].sum())
        profiles.append(
            {
                "cluster": int(cluster),
                "episodes": int(index.sum()),
                "successes": int(index.sum() - failures),
                "failures": failures,
                "failure_rate": float(failures / index.sum()),
                "task_counts": dict(sorted(Counter(tasks[index]).items())),
                "dominant_task_fraction": float(
                    max(Counter(tasks[index]).values()) / index.sum()
                ),
                "length_median": float(np.median(lengths[index])),
                "task_cells": task_rows,
                "eligible_cross_task_cells": len(eligible),
                "all_eligible_deltas_positive": bool(
                    len(eligible) >= 3
                    and all(row["failure_rate_delta"] > 0.0 for row in eligible)
                ),
                "all_eligible_deltas_negative": bool(
                    len(eligible) >= 3
                    and all(row["failure_rate_delta"] < 0.0 for row in eligible)
                ),
            }
        )
    return profiles


def annotate_cross_task_blocks(result: dict[str, Any]) -> None:
    outcome = result["associations"]["outcome_within_task_initial_state"]
    gate = bool(
        result["status"] == "stable_partition"
        and outcome["nmi_excess_over_null"] >= OUTCOME_EXCESS_MIN
        and outcome["fdr_bh_q"] < 0.05
    )
    failure_blocks = []
    success_blocks = []
    for profile in result["profiles"]:
        profile["cross_task_failure_block"] = bool(
            gate and profile["all_eligible_deltas_positive"]
        )
        profile["cross_task_success_block"] = bool(
            gate and profile["all_eligible_deltas_negative"]
        )
        if profile["cross_task_failure_block"]:
            failure_blocks.append(profile["cluster"])
        if profile["cross_task_success_block"]:
            success_blocks.append(profile["cluster"])
    result["outcome_structure_gate_pass"] = gate
    result["cross_task_failure_blocks"] = failure_blocks
    result["cross_task_success_blocks"] = success_blocks


def write_assignments(
    path: pathlib.Path,
    metadata: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> None:
    fields = [
        "task",
        "checkpoint_sha256",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "episode_length",
        "outcome",
        *[f"{variant}_cluster" for variant in VARIANTS],
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(len(metadata["episode"])):
            writer.writerow(
                {
                    "task": str(metadata["task"][index]),
                    "checkpoint_sha256": str(metadata["checkpoint_sha256"][index]),
                    "episode": int(metadata["episode"][index]),
                    "init_state_id": int(metadata["init_state_id"][index]),
                    "flow_noise_seed": int(metadata["flow_noise_seed"][index]),
                    "episode_length": int(metadata["episode_length"][index]),
                    "outcome": "failure" if metadata["failure"][index] else "success",
                    **{
                        f"{variant}_cluster": int(labels[variant][index])
                        for variant in VARIANTS
                    },
                }
            )


def _short_task(task: str) -> str:
    return task.split("/", 1)[-1]


def fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def render_report(summary: dict[str, Any]) -> str:
    primary = summary["variants"][PRIMARY_VARIANT]
    failure_blocks = primary["cross_task_failure_blocks"]
    success_blocks = primary["cross_task_success_blocks"]
    if failure_blocks or success_blocks:
        answer = (
            "The frozen exploratory criterion found cross-task outcome blocks: "
            f"failure={failure_blocks}, success={success_blocks}."
        )
    else:
        answer = (
            "No block in the primary task-residual partition passed the frozen "
            "cross-task success/failure criterion."
        )
    lines = [
        "# Aligned route-signature landmark experiment",
        "",
        f"Run class: `{summary['provenance']['run_class']}`.",
        "",
        answer,
        "",
        "This is an exploratory posthoc analysis. Task residualization is explicit;",
        "clusters represent deviations from each task's median route shape.",
        "",
        "## Model selection and association",
        "",
        "| variant | role | status | K | sizes | silhouette | ARI med/p10 | PAC | task NMI | outcome excess | BH q | cross-task failure/success blocks |",
        "|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        result = summary["variants"][variant]
        trial = next(
            row for row in result["trials"] if row["clusters"] == result["reported_k"]
        )
        outcome = result["associations"]["outcome_within_task_initial_state"]
        lines.append(
            "| %s | %s | %s | %d | %s | %s | %s/%s | %s | %s | %s | %s | %s/%s |"
            % (
                variant,
                "primary" if variant == PRIMARY_VARIANT else "sensitivity",
                result["status"],
                result["reported_k"],
                "/".join(map(str, result["reported_sizes"])),
                fmt(trial["subsample_silhouette_mean"]),
                fmt(trial["ari_median"]),
                fmt(trial["ari_p10"]),
                fmt(trial["consensus_pac_01_09"]),
                fmt(result["associations"]["task"]["nmi"]),
                fmt(outcome["nmi_excess_over_null"]),
                fmt(outcome["fdr_bh_q"], 4),
                ",".join(map(str, result["cross_task_failure_blocks"])) or "none",
                ",".join(map(str, result["cross_task_success_blocks"])) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Primary block composition",
            "",
            "| block | n | success/failure | failure rate | dominant task | dominant fraction | length median | eligible tasks | direction |",
            "|---:|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in primary["profiles"]:
        dominant = max(row["task_counts"], key=row["task_counts"].get)
        direction = (
            "failure"
            if row["cross_task_failure_block"]
            else "success"
            if row["cross_task_success_block"]
            else "none"
        )
        lines.append(
            "| C%d | %d | %d/%d | %s | %s | %s | %s | %d | %s |"
            % (
                row["cluster"],
                row["episodes"],
                row["successes"],
                row["failures"],
                fmt(row["failure_rate"]),
                _short_task(dominant),
                fmt(row["dominant_task_fraction"]),
                fmt(row["length_median"], 1),
                row["eligible_cross_task_cells"],
                direction,
            )
        )
    lines.extend(
        [
            "",
            "### Per-task failure-rate deltas",
            "",
            "| block | task | n | failure rate | task baseline | delta |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in primary["profiles"]:
        for cell in row["task_cells"]:
            if "failure_rate_delta" not in cell:
                continue
            lines.append(
                "| C%d | %s | %d | %s | %s | %s |"
                % (
                    row["cluster"],
                    _short_task(cell["task"]),
                    cell["episodes"],
                    fmt(cell["failure_rate"]),
                    fmt(cell["task_baseline_failure_rate"]),
                    fmt(cell["failure_rate_delta"]),
                )
            )
    lines.extend(
        [
            "",
            "## Per-task outcome association",
            "",
            "| variant | task | failures | excess NMI | p | BH q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        for task, row in summary["variants"][variant]["per_task"].items():
            if row["status"] != "estimated":
                continue
            lines.append(
                "| %s | %s | %d | %s | %s | %s |"
                % (
                    variant,
                    _short_task(task),
                    row["failures"],
                    fmt(row["nmi_excess_over_null"]),
                    fmt(row["permutation_p_one_sided"], 4),
                    fmt(row["fdr_bh_q"], 4),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The primary representation uses task labels for median residualization; it tests shared relative deviations, not task-independent absolute router states.",
            "- The route-only landmark basis is fixed before subsampling, so stability is conditional on that basis.",
            "- Shift alignment can remove real timing differences, while ten-anchor interpolation can smooth short trajectories.",
            "- Relative completion and timeout semantics, episode length, and checkpoint remain potential outcome proxies.",
            "- This exploratory analysis does not establish causal mechanisms or unseen-task generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_model_selection(path: pathlib.Path, results: dict[str, dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for axis, variant in zip(axes, VARIANTS):
        rows = results[variant]["trials"]
        k = [row["clusters"] for row in rows]
        axis.plot(k, [row["subsample_silhouette_mean"] for row in rows], "o-", label="silhouette")
        axis.plot(k, [row["ari_median"] for row in rows], "s-", label="ARI median")
        axis.plot(k, [row["ari_p10"] for row in rows], "^-", label="ARI p10")
        axis.plot(k, [row["consensus_pac_01_09"] for row in rows], "x-", label="PAC")
        axis.set_title(variant.replace("_", " "))
        axis.set_xlabel("K")
        axis.set_xticks(k)
        axis.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=7)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_embeddings(
    path: pathlib.Path,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    failure: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, variant in zip(axes, VARIANTS):
        embedding = embeddings[variant]
        axis.scatter(
            embedding[~failure, 0],
            embedding[~failure, 1],
            c=labels[variant][~failure],
            cmap="tab10",
            marker="o",
            s=8,
            alpha=0.4,
            edgecolors="none",
        )
        axis.scatter(
            embedding[failure, 0],
            embedding[failure, 1],
            c=labels[variant][failure],
            cmap="tab10",
            marker="x",
            s=22,
            linewidths=0.8,
        )
        axis.set_title(variant.replace("_", " "))
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_primary_deltas(path: pathlib.Path, profiles: list[dict[str, Any]]) -> None:
    tasks = sorted(
        {
            cell["task"]
            for row in profiles
            for cell in row["task_cells"]
            if "failure_rate_delta" in cell
        }
    )
    matrix = np.full((len(profiles), len(tasks)), np.nan)
    for row in profiles:
        for cell in row["task_cells"]:
            if "failure_rate_delta" in cell:
                matrix[row["cluster"], tasks.index(cell["task"])] = cell[
                    "failure_rate_delta"
                ]
    limit = max(0.05, float(np.nanmax(np.abs(matrix))))
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(range(len(profiles)), [f"C{row['cluster']}" for row in profiles])
    axis.set_xticks(range(len(tasks)), [_short_task(task) for task in tasks], rotation=25, ha="right")
    axis.set_title("Task-residual block failure-rate delta from task baseline")
    figure.colorbar(image, ax=axis, label="failure-rate delta")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def self_test() -> None:
    rng = np.random.default_rng(1)
    sequences = rng.normal(size=(20, ANCHORS, 4)).astype(np.float32)
    normalized = normalize_trajectory_shape(sequences)
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(np.linalg.norm(normalized.reshape(20, -1), axis=1), 1.0)
    distances, landmarks, _audit = aligned_landmark_distances(normalized, 2)
    assert distances.shape == (20, 20)
    assert np.all(distances >= 0.0) and np.all(distances <= 2.0)
    assert np.allclose(distances[landmarks, np.arange(len(landmarks))], 0.0, atol=1e-5)
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.subsamples < 10 or args.permutations < 100:
        raise ValueError("too few subsamples or permutations")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "completion.json").unlink(missing_ok=True)
    method_target = args.out_dir / "METHOD.md"
    if METHOD_SOURCE.resolve() != method_target.resolve():
        shutil.copyfile(METHOD_SOURCE, method_target)

    arrays, source_manifest, source_completion = _load_source(args.source_dir)
    metadata = _metadata(arrays)
    score = np.asarray(arrays["diagnostic_mean_soft_speed"])
    formal = bool(
        args.source_dir.resolve() == SOURCE_DIR.resolve()
        and args.subsamples == SUBSAMPLES
        and args.permutations == PERMUTATIONS
        and args.seed == SEED
    )

    distances: dict[str, np.ndarray] = {}
    landmarks: dict[str, np.ndarray] = {}
    results: dict[str, dict[str, Any]] = {}
    labels: dict[str, np.ndarray] = {}
    embeddings: dict[str, np.ndarray] = {}
    with threadpool_limits(limits=4):
        sequences, signature_audit = build_anchor_signatures(
            arrays["feature_primary_geometry"], args.seed
        )
        for offset, variant in enumerate(VARIANTS):
            print(f"building aligned landmark distances: {variant}", flush=True)
            variant_sequences = normalize_trajectory_shape(
                residualize_sequences(sequences, metadata, variant)
            )
            matrix, landmark_index, distance_audit = aligned_landmark_distances(
                variant_sequences, args.seed + 1000 * offset
            )
            distances[variant] = matrix
            landmarks[variant] = landmark_index
            print(f"clustering aligned landmark distances: {variant}", flush=True)
            result, variant_labels, embedding = cluster_landmark_view(
                matrix,
                score,
                args.subsamples,
                args.seed + 10000 * offset,
            )
            result["distance_audit"] = distance_audit
            results[variant] = result
            labels[variant] = variant_labels
            embeddings[variant] = embedding

    print("trajectory labels frozen; revealing outcome metadata", flush=True)
    for offset, variant in enumerate(VARIANTS):
        result = results[variant]
        result["associations"] = association_bundle(
            labels[variant], metadata, args.permutations, args.seed + 50000 + 1000 * offset
        )
        result["per_task"] = per_task_associations(
            labels[variant], metadata, args.permutations, args.seed + 60000 + 1000 * offset
        )
        result["profiles"] = cluster_profiles(labels[variant], metadata)

    add_bh(
        [
            association
            for variant in VARIANTS
            for association in results[variant]["associations"].values()
        ]
    )
    add_bh(
        [
            association
            for variant in VARIANTS
            for association in results[variant]["per_task"].values()
            if association["status"] == "estimated"
        ]
    )
    for variant in VARIANTS:
        annotate_cross_task_blocks(results[variant])

    cross_variant_ari = {
        left: {
            right: float(adjusted_rand_score(labels[left], labels[right]))
            for right in VARIANTS
        }
        for left in VARIANTS
    }
    summary = {
        "schema": "himoe.aligned_route_kernel.v1",
        "method": str(method_target.resolve()),
        "provenance": {
            "run_class": "formal_exploratory" if formal else "nonformal",
            "source_dir": str(args.source_dir.resolve()),
            "source_completion_sha256": sha256_file(args.source_dir / "completion.json"),
            "source_feature_manifest_sha256": sha256_file(
                args.source_dir / "feature_cache_manifest.json"
            ),
            "analysis_script_sha256": sha256_file(pathlib.Path(__file__)),
            "method_sha256": sha256_file(method_target),
            "seed": args.seed,
            "subsamples": args.subsamples,
            "permutations": args.permutations,
            "numeric_thread_limit": 4,
        },
        "source_audit": source_manifest["audit"],
        "source_completion_schema": source_completion["schema"],
        "representation": {
            **signature_audit,
            "anchors": ANCHORS,
            "phase_window": [0.5, 1.0],
            "landmarks": LANDMARKS,
            "integer_shift_range": [-MAX_SHIFT, MAX_SHIFT],
            "primary_variant": PRIMARY_VARIANT,
            "outcome_used_before_label_freeze": False,
            "task_used_by_primary_residualization": True,
        },
        "selection": {
            "candidate_k": list(CANDIDATE_K),
            "minimum_cluster_size": minimum_cluster_size(len(metadata["failure"])),
            "subsamples": args.subsamples,
            "subsample_fraction": 0.80,
            "landmark_basis_refit_in_subsample": False,
            "distance_scaling_pca_ward_refit_in_subsample": True,
        },
        "cross_task_block_criterion": {
            "partition_stable": True,
            "outcome_excess_nmi_min": OUTCOME_EXCESS_MIN,
            "outcome_fdr_bh_q_max": 0.05,
            "minimum_tasks": 3,
            "minimum_cluster_task_cell_episodes": 10,
            "all_eligible_task_deltas_same_direction": True,
        },
        "variants": results,
        "cross_variant_ari": cross_variant_ari,
    }

    write_assignments(args.out_dir / "assignments.csv", metadata, labels)
    np.savez_compressed(
        args.out_dir / "embeddings_labels_landmarks.npz",
        **{
            f"embedding_{variant}": embeddings[variant].astype(np.float32)
            for variant in VARIANTS
        },
        **{
            f"label_{variant}": labels[variant].astype(np.int16)
            for variant in VARIANTS
        },
        **{
            f"landmark_index_{variant}": landmarks[variant].astype(np.int32)
            for variant in VARIANTS
        },
        **{
            f"landmark_distance_{variant}": distances[variant].astype(np.float32)
            for variant in VARIANTS
        },
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )
    (args.out_dir / "report.md").write_text(render_report(summary))
    plot_model_selection(args.out_dir / "model_selection.png", results)
    plot_embeddings(
        args.out_dir / "embeddings.png", embeddings, labels, metadata["failure"]
    )
    plot_primary_deltas(
        args.out_dir / "primary_cross_task_deltas.png",
        results[PRIMARY_VARIANT]["profiles"],
    )
    outputs = [
        "METHOD.md",
        "assignments.csv",
        "embeddings_labels_landmarks.npz",
        "summary.json",
        "report.md",
        "model_selection.png",
        "embeddings.png",
        "primary_cross_task_deltas.png",
    ]
    completion = {
        "schema": "himoe.aligned_route_kernel.completion.v1",
        "run_class": "formal_exploratory" if formal else "nonformal",
        "expected_outputs": outputs,
        "output_sha256": {
            name: sha256_file(args.out_dir / name) for name in outputs
        },
    }
    temporary = args.out_dir / "completion.json.tmp"
    temporary.write_text(json.dumps(completion, indent=2, allow_nan=False))
    temporary.replace(args.out_dir / "completion.json")
    print(f"wrote {args.out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
