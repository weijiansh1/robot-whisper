#!/usr/bin/env python3
"""Faithful re-implementation of the aligned-route-kernel downstream pipeline.

Purpose: run IDENTICAL downstream machinery on non-routing sentinel features,
permuted routing features and Gaussian dummies, so every baseline is scored
with the same evaluation convention as the published "consensus failure core".

Stages copied verbatim from analyze_aligned_route_kernel.py
(build_anchor_signatures -> residualize -> normalize_trajectory_shape ->
aligned_landmark_distances -> prepare_embedding -> ward_labels -> canonicalize).
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

BASE = pathlib.Path("/home/jovyan/work/himoe-vla/himoe-route-capture/analysis")
SRC = BASE / "all-outcome-routing-clusters"
OUT = BASE / "AUDIT-clustering-leakage-20260828"

SEED = 20260829
ANCHORS = 10
SIGNATURE_COMPONENTS = 32
LANDMARKS = 256
MAX_SHIFT = 2
CANDIDATE_K = tuple(range(2, 9))


# ---------------------------------------------------------------- source data
def load_cache() -> dict[str, np.ndarray]:
    manifest = json.loads((SRC / "feature_cache_manifest.json").read_text())
    out = {}
    for name, item in manifest["arrays"].items():
        out[name] = np.load(SRC / "feature_cache" / item["file"], allow_pickle=False)
    return out


# ------------------------------------------------- stages (verbatim semantics)
def build_anchor_signatures(
    geometry: np.ndarray, seed: int = SEED, components: int = SIGNATURE_COMPONENTS
) -> np.ndarray:
    """geometry (n, ANCHORS*D) -> (n, ANCHORS, comps).  Copy of aligned kernel."""
    count = len(geometry)
    per_anchor = geometry.shape[1] // ANCHORS
    values = np.asarray(geometry, dtype=np.float64).reshape(-1, per_anchor)
    keep = values.std(axis=0) > 1e-9
    values = values[:, keep]
    center = np.median(values, axis=0)
    q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = values.std(axis=0)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = np.clip((values - center) / scale, -20.0, 20.0)
    comps = min(components, standardized.shape[1])
    model = PCA(n_components=comps, svd_solver="randomized", random_state=seed)
    transformed = model.fit_transform(standardized)
    return transformed.reshape(count, ANCHORS, comps).astype(np.float32)


def normalize_trajectory_shape(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float32)
    values = values - values.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(values.reshape(len(values), -1), axis=1)
    if np.any(norm <= 1e-12):
        norm = np.maximum(norm, 1e-12)
    return values / norm[:, None, None]


def aligned_landmark_distances(sequences: np.ndarray, seed: int = SEED) -> np.ndarray:
    flattened = sequences.reshape(len(sequences), -1)
    _centers, landmark_index = kmeans_plusplus(
        flattened, n_clusters=min(LANDMARKS, len(sequences)), random_state=seed
    )
    landmarks = sequences[landmark_index]
    similarities = np.full((len(sequences), len(landmarks)), -np.inf, dtype=np.float32)
    length = sequences.shape[1]
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        if shift >= 0:
            left, right = sequences[:, shift:], landmarks[:, : length - shift]
        else:
            left, right = sequences[:, : length + shift], landmarks[:, -shift:]
        left = left.reshape(len(left), -1)
        right = right.reshape(len(right), -1)
        left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
        right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
        similarities = np.maximum(similarities, left @ right.T)
    similarities = np.clip(similarities, -1.0, 1.0)
    return (1.0 - similarities).astype(np.float32)


def prepare_embedding(matrix: np.ndarray, seed: int = SEED) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    keep = values.std(axis=0) > 1e-9
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
    return transformed[:, :retained]


def ward_labels(embedding: np.ndarray, clusters: int) -> np.ndarray:
    tree = linkage(embedding, method="ward", optimal_ordering=False)
    return fcluster(tree, clusters, criterion="maxclust").astype(np.int32) - 1


def canonicalize(labels: np.ndarray, score: np.ndarray) -> np.ndarray:
    ordered = sorted(
        np.unique(labels),
        key=lambda label: (float(np.mean(score[labels == label])), int(label)),
    )
    mapping = {int(label): index for index, label in enumerate(ordered)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=np.int32)


def run_pipeline(
    anchor_features: np.ndarray, score: np.ndarray, seed: int = SEED
) -> dict[int, np.ndarray]:
    """anchor_features (n, ANCHORS, D) -> {k: canonicalized labels}."""
    shaped = normalize_trajectory_shape(anchor_features)
    distances = aligned_landmark_distances(shaped, seed)
    embedding = prepare_embedding(distances, seed)
    return {k: canonicalize(ward_labels(embedding, k), score) for k in CANDIDATE_K}


# ------------------------------------------------------------------ evaluation
def block_table(labels: np.ndarray, failure: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    n_fail = int(failure.sum())
    for lab in np.unique(labels):
        m = labels == lab
        n = int(m.sum())
        f = int((m & failure).sum())
        prec = f / n
        rec = f / n_fail
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
        rows.append(
            dict(block=int(lab), n=n, failure=f, success=n - f,
                 precision=round(prec, 4), failure_recall=round(rec, 4),
                 f1=round(f1, 4))
        )
    return rows


def summarize(
    labels: np.ndarray, failure: np.ndarray, task: np.ndarray,
    length: np.ndarray, min_n: int = 100,
) -> dict[str, Any]:
    rows = block_table(labels, failure)
    best_f1 = max(rows, key=lambda r: r["f1"])
    eligible = [r for r in rows if r["n"] >= min_n]
    best_prec = max(eligible or rows, key=lambda r: (r["precision"], r["n"]))
    return dict(
        k=int(len(rows)),
        blocks=rows,
        best_f1_block=best_f1,
        best_precision_block_minN=best_prec,
        ami_outcome=round(float(adjusted_mutual_info_score(failure.astype(int), labels)), 4),
        ami_task=round(float(adjusted_mutual_info_score(task, labels)), 4),
        ami_length=round(float(adjusted_mutual_info_score(length, labels)), 4),
    )


__all__ = [
    "load_cache", "build_anchor_signatures", "normalize_trajectory_shape",
    "aligned_landmark_distances", "prepare_embedding", "ward_labels",
    "canonicalize", "run_pipeline", "block_table", "summarize",
    "adjusted_rand_score", "silhouette_score",
    "BASE", "SRC", "OUT", "SEED", "ANCHORS", "CANDIDATE_K",
]
