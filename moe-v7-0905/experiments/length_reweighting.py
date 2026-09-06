#!/usr/bin/env python3
"""Length-stratified reweighting of the calibration corpus.

The variant study showed v7's pooled trajectory-peak cut cannot be replaced:
it simultaneously supplies the per-trajectory multiple-testing correction, an
absolute scale reference, and a comparison base that is not conditioned on
survival. Every corpus-free reshaping breaks one of the three.

This module leaves the cut's *shape* alone and changes only the corpus's
effective composition. Reference trajectories are reweighted so their rollout
length distribution matches the deployment distribution's, then the same peak
quantile is taken. Nothing about the online monitor changes: it still sees one
scalar per stream and still reads only hb_router_probs.

The target length distribution is a deployment-time quantity, estimated from
unlabeled rollouts on the new distribution. It is not a runtime input and it
carries no outcome. It is, however, information about the held-out cohort, so
any evaluation using it makes a weaker claim than plain LOSO and must say so.
"""

from __future__ import annotations

import numpy as np


MIN_EFFECTIVE_SAMPLE = 32
DEFAULT_WEIGHT_CLIP = 20.0


def length_weights(
    reference_length: np.ndarray,
    target_length: np.ndarray,
    clip: float = DEFAULT_WEIGHT_CLIP,
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-reference-trajectory weights matching the target length histogram.

    Weights are the density ratio target/reference over integer rollout
    lengths, clipped so a length that is rare in the reference cannot dominate.
    Target lengths with no reference support get no weight at all; the mass
    lost that way is reported rather than redistributed.
    """
    reference_length = np.asarray(reference_length, dtype=int)
    target_length = np.asarray(target_length, dtype=int)
    size = int(max(reference_length.max(), target_length.max())) + 1
    reference_pmf = np.bincount(reference_length, minlength=size) / len(
        reference_length
    )
    target_pmf = np.bincount(target_length, minlength=size) / len(target_length)

    ratio = np.zeros(size, dtype=np.float64)
    supported = reference_pmf > 0.0
    ratio[supported] = target_pmf[supported] / reference_pmf[supported]
    clipped_mass = float(target_pmf[supported][ratio[supported] > clip].sum())
    ratio = np.clip(ratio, 0.0, clip)

    weights = ratio[reference_length]
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("no reference trajectory has a supported length")
    weights = weights / total

    diagnostics = {
        "uncovered_target_mass": float(target_pmf[~supported].sum()),
        "clipped_target_mass": clipped_mass,
        "effective_sample_size": float(1.0 / np.square(weights).sum()),
        "max_weight_ratio": float(ratio.max()),
    }
    return weights, diagnostics


def weighted_quantile_higher(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    """Weighted analogue of intrinsic_guard_monitor.quantile_higher.

    Uses the `higher` convention: the smallest observed value whose cumulative
    weight reaches the requested quantile. Non-finite values are dropped and
    their weight is removed from the normalisation, matching how the unweighted
    path drops trajectories too short to form the statistic.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape:
        raise ValueError("values and weights do not align")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    keep = np.isfinite(values) & (weights > 0.0)
    values, weights = values[keep], weights[keep]
    if len(values) == 0:
        raise ValueError("no finite weighted reference values")
    effective = float(np.square(weights.sum()) / np.square(weights).sum())
    if effective < MIN_EFFECTIVE_SAMPLE:
        raise ValueError(
            f"effective sample size {effective:.1f} is below {MIN_EFFECTIVE_SAMPLE}"
        )
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    index = int(np.searchsorted(cumulative, quantile, side="left"))
    return float(values[min(index, len(values) - 1)])
