#!/usr/bin/env python3
"""Euclidean geometry of the ten action tokens in HB routing space.

The conditional kernel cached by ``extract_conditional_kernels.py`` is PSD, so
it is the Gram matrix of ten points ``x_1 .. x_10`` in some Euclidean space:

    K_ij = <x_i, x_j>,      d^2(i, j) = K_ii + K_jj - 2 K_ij

Everything here is a function of that point cloud.  Two views are used
throughout and must not be confused:

    K   the kernel about the origin.  Its spectrum mixes where the cloud sits
        with how it is shaped.  ``conditional_effective_rank`` in the
        moe-hb-front-back-0905 bundle is the entropy rank of this spectrum.
    B   = J K J with J = I - 11^T/10, the doubly centred kernel.  This is the
        Gram matrix of the cloud about its own centroid, so its spectrum is the
        *shape* spectrum: rank <= 9, invariant to translating the cloud.

The shape distance between two clouds is the full Procrustes disparity

    1 - (sum_i sigma_i(X^T Y))^2 / (tr B_X tr B_Y)

with sigma the singular values of the cross-product of the centred
configurations.  It is invariant to translation, rotation, reflection and
uniform scale, so it isolates deformation from size.  It equals sin^2 of the
Procrustes angle and lies in [0, 1].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
CACHE_ROOT = BUNDLE / "cache"
KERNEL_ROOT = CACHE_ROOT / "kernels"
FEATURE_ROOT = CACHE_ROOT / "features"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
FRONT = slice(0, 4)
BACK = slice(4, 8)
UPPER = np.triu_indices(10, 0)
OFFDIAG = np.triu_indices(10, 1)
TOKEN_LAG = np.abs(OFFDIAG[0] - OFFDIAG[1]).astype(np.float64)
EPSILON = 1e-12

FEATURE_NAMES = (
    "kernel_trace",
    "size",
    "centroid_norm2",
    "shape_pr",
    "shape_erank",
    "lam1_share",
    "lam2_share",
    "lam12_share",
    "kernel_erank",
    "procrustes_layer",
    "bandedness",
    "d_cv",
)


def load_index(cohort: str) -> dict[str, np.ndarray]:
    with np.load(KERNEL_ROOT / f"{cohort}_index.npz", allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_packed(cohort: str, mmap: bool = True) -> np.ndarray:
    return np.load(
        KERNEL_ROOT / f"{cohort}_conditional55.npy",
        mmap_mode="r" if mmap else None,
    )


def symmetrise(upper: np.ndarray) -> np.ndarray:
    """[..., 55] -> [..., 10, 10]"""
    shape = upper.shape[:-1] + (10, 10)
    out = np.empty(shape, dtype=np.float32)
    out[..., UPPER[0], UPPER[1]] = upper
    out[..., UPPER[1], UPPER[0]] = upper
    return out


def double_centre(kernel: np.ndarray) -> np.ndarray:
    row = kernel.mean(axis=-1, keepdims=True)
    col = kernel.mean(axis=-2, keepdims=True)
    total = kernel.mean(axis=(-2, -1))[..., None, None]
    return kernel - row - col + total


def distances(kernel: np.ndarray) -> np.ndarray:
    diagonal = np.diagonal(kernel, axis1=-2, axis2=-1)
    squared = diagonal[..., :, None] + diagonal[..., None, :] - 2.0 * kernel
    return np.sqrt(np.clip(squared, 0.0, None))


def configuration(centred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centred Gram -> (eigenvalues descending, coordinates [..., 10, 10])."""
    values, vectors = np.linalg.eigh(centred)
    values = np.clip(values[..., ::-1], 0.0, None)
    vectors = vectors[..., ::-1]
    return values, vectors * np.sqrt(values)[..., None, :]


def entropy_rank(spectrum: np.ndarray) -> np.ndarray:
    mass = np.clip(spectrum, 0.0, None)
    probability = mass / np.maximum(mass.sum(axis=-1, keepdims=True), EPSILON)
    entropy = -(probability * np.log(np.maximum(probability, 1e-30))).sum(axis=-1)
    return np.exp(entropy)


def participation_ratio(spectrum: np.ndarray) -> np.ndarray:
    mass = np.clip(spectrum, 0.0, None)
    total = mass.sum(axis=-1)
    return total**2 / np.maximum((mass**2).sum(axis=-1), EPSILON)


def procrustes_disparity(
    coordinates: np.ndarray, trace: np.ndarray, template: np.ndarray
) -> np.ndarray:
    """Full Procrustes disparity of each configuration against one template.

    ``template`` is a centred 10 x p configuration; ``coordinates`` is
    [..., 10, p] centred; ``trace`` is tr(B) of each configuration.
    """
    template_trace = float((template**2).sum())
    cross = np.einsum("...ij,ik->...jk", coordinates, template, optimize=True)
    nuclear = np.linalg.svd(cross, compute_uv=False).sum(axis=-1)
    denominator = np.maximum(trace * template_trace, EPSILON)
    return np.clip(1.0 - nuclear**2 / denominator, 0.0, 1.0)


def pairwise_procrustes(templates: np.ndarray) -> np.ndarray:
    """[m, 10, 10] centred Grams -> [m, m] full Procrustes disparity."""
    values, coordinates = configuration(templates)
    trace = values.sum(axis=-1)
    count = len(templates)
    out = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        out[i] = procrustes_disparity(coordinates, trace, coordinates[i])
    return 0.5 * (out + out.T)


def correlate_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis, broadcasting ``right``."""
    a = left - left.mean(axis=-1, keepdims=True)
    b = right - right.mean(axis=-1, keepdims=True)
    numerator = (a * b).sum(axis=-1)
    denominator = np.sqrt((a**2).sum(axis=-1) * (b**2).sum(axis=-1))
    return numerator / np.maximum(denominator, EPSILON)


def block_features(upper: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """[n, 8, 55] kernels + [8, 10, p] layer templates -> [n, 8, len(FEATURE_NAMES)]."""
    kernel = symmetrise(np.asarray(upper, dtype=np.float32))
    kernel_spectrum = np.clip(np.linalg.eigvalsh(kernel), 0.0, None)
    centred = double_centre(kernel)
    values, coordinates = configuration(centred)
    trace = values.sum(axis=-1)
    total = np.maximum(trace, EPSILON)

    procrustes = np.empty(kernel.shape[:2], dtype=np.float64)
    for layer in range(kernel.shape[1]):
        procrustes[:, layer] = procrustes_disparity(
            coordinates[:, layer], trace[:, layer], templates[layer]
        )

    edge = distances(kernel)[..., OFFDIAG[0], OFFDIAG[1]]
    lag = np.broadcast_to(TOKEN_LAG.astype(np.float32), edge.shape)
    out = np.stack(
        (
            np.trace(kernel, axis1=-2, axis2=-1),
            np.sqrt(trace),
            kernel.mean(axis=(-2, -1)),
            participation_ratio(values),
            entropy_rank(values),
            values[..., 0] / total,
            values[..., 1] / total,
            (values[..., 0] + values[..., 1]) / total,
            entropy_rank(kernel_spectrum) / 10.0,
            procrustes,
            correlate_rows(edge, lag),
            edge.std(axis=-1) / np.maximum(edge.mean(axis=-1), EPSILON),
        ),
        axis=-1,
    )
    return out.astype(np.float32, copy=False)


def features_for_cohort(
    cohort: str, templates: np.ndarray, chunk: int = 40000
) -> np.ndarray:
    packed = load_packed(cohort)
    out = np.empty((len(packed), 8, len(FEATURE_NAMES)), dtype=np.float32)
    for start in range(0, len(packed), chunk):
        stop = min(start + chunk, len(packed))
        out[start:stop] = block_features(np.asarray(packed[start:stop]), templates)
    return out


def expand(packed: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[n_valid, ...] -> [n_episode, max_query, ...] with NaN padding."""
    out = np.full(valid.shape + packed.shape[1:], np.nan, dtype=np.float32)
    out[valid] = packed
    return out


def mean_kernel(cohort: str, chunk: int = 40000) -> np.ndarray:
    """Pooled mean conditional kernel per layer, [8, 10, 10]."""
    packed = load_packed(cohort)
    total = np.zeros((8, 55), dtype=np.float64)
    for start in range(0, len(packed), chunk):
        stop = min(start + chunk, len(packed))
        total += np.asarray(packed[start:stop], dtype=np.float64).sum(axis=0)
    return symmetrise((total / len(packed)).astype(np.float32))


def group_mean_kernel(
    cohort: str, group: np.ndarray, n_group: int, chunk: int = 40000
) -> np.ndarray:
    """Mean conditional kernel per group of queries, [n_group, 8, 10, 10]."""
    packed = load_packed(cohort)
    total = np.zeros((n_group, 8, 55), dtype=np.float64)
    count = np.zeros(n_group, dtype=np.int64)
    for start in range(0, len(packed), chunk):
        stop = min(start + chunk, len(packed))
        block = np.asarray(packed[start:stop], dtype=np.float64)
        keys = group[start:stop]
        np.add.at(total, keys, block)
        np.add.at(count, keys, 1)
    if (count == 0).any():
        raise ValueError("empty group in group_mean_kernel")
    return symmetrise((total / count[:, None, None]).astype(np.float32))


def layer_templates(mean_kernels: np.ndarray) -> np.ndarray:
    """[8, 10, 10] mean kernels -> [8, 10, 10] centred template configurations."""
    _, coordinates = configuration(double_centre(mean_kernels))
    return coordinates
