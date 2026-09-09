"""Behavioural readouts on an action chunk, and the clustered bootstrap.

Kept apart from the policy runner so the arithmetic can be tested on a laptop
with no checkpoint, no GPU and no simulator.

A LIBERO action row is ``[dx, dy, dz, drx, dry, drz, gripper]``.  The chunk is
ten of those.  ``PROTOCOL.md`` fixes which readout is primary at which anchor;
nothing here chooses.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def cumulative_translation(action: np.ndarray) -> np.ndarray:
    """Where the chunk commands the end effector to end up, relative to now."""
    chunk = np.asarray(action, np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError("expected a (steps, 7) action chunk, got %s" % (chunk.shape,))
    return chunk[:, :3].sum(axis=0)


def direction_cosine(
    action: np.ndarray, eef_xyz: np.ndarray, target_xyz: np.ndarray
) -> float:
    """B_dir: how well the commanded motion points at the target, in the table plane.

    Horizontal because the displacement under test is horizontal.  Including the
    vertical component would let the descent, which is large and common to every
    condition, dominate a cosine that is supposed to measure bearing.

    Returns NaN when either vector is degenerate: a chunk that commands no
    horizontal motion has no direction, and pretending it has one would put a
    fabricated 0 into the mean.
    """
    commanded = cumulative_translation(action)[:2]
    toward = np.asarray(target_xyz, np.float64)[:2] - np.asarray(eef_xyz, np.float64)[:2]
    denominator = np.linalg.norm(commanded) * np.linalg.norm(toward)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(commanded, toward) / denominator)


def tracking_cosine(
    action_edited: np.ndarray, action_held: np.ndarray, shift_direction: np.ndarray
) -> float:
    """B_track: does the *change* in commanded motion follow the object's shift?

    Its null is exactly zero, and unlike ``direction_cosine`` it does not inherit
    however the approach happened to be aimed already.
    """
    delta = (cumulative_translation(action_edited) - cumulative_translation(action_held))[:2]
    direction = np.asarray(shift_direction, np.float64)[:2]
    denominator = np.linalg.norm(delta) * np.linalg.norm(direction)
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(delta, direction) / denominator)


def gripper_mean(action: np.ndarray) -> float:
    """B_grip: the mean gripper command over the chunk."""
    chunk = np.asarray(action, np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError("expected a (steps, 7) action chunk, got %s" % (chunk.shape,))
    return float(chunk[:, 6].mean())


def translation_norm(action: np.ndarray) -> float:
    """B_mag: how far the chunk commands the end effector to travel."""
    return float(np.linalg.norm(cumulative_translation(action)))


def chunk_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Diagnostic only.  Reported, never used as an effect size."""
    return float(np.linalg.norm(np.asarray(left, np.float64) - np.asarray(right, np.float64)))


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[int],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    resamples: int = 10000,
    seed: int = 20260906,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Percentile interval, resampling whole clusters.

    Episodes contribute two anchors and several conditions, so rows within an
    episode are not independent.  Resampling episodes rather than rows is what
    keeps the interval honest.  NaN values are dropped before the statistic, and
    the number dropped is reported rather than hidden.
    """
    values = np.asarray(values, np.float64)
    clusters = np.asarray(clusters)
    if values.shape != clusters.shape:
        raise ValueError("values and clusters must align")
    finite = np.isfinite(values)
    dropped = int((~finite).sum())
    values, clusters = values[finite], clusters[finite]
    if not len(values):
        return {"n": 0, "n_clusters": 0, "dropped_non_finite": dropped,
                "point": float("nan"), "low": float("nan"), "high": float("nan")}

    unique = np.unique(clusters)
    by_cluster = [values[clusters == key] for key in unique]
    generator = np.random.default_rng(seed)
    draws = np.empty(resamples, np.float64)
    for index in range(resamples):
        picked = generator.integers(0, len(by_cluster), len(by_cluster))
        draws[index] = statistic(np.concatenate([by_cluster[i] for i in picked]))
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n": int(len(values)),
        "n_clusters": int(len(unique)),
        "dropped_non_finite": dropped,
        "point": float(statistic(values)),
        "low": float(low),
        "high": float(high),
    }
