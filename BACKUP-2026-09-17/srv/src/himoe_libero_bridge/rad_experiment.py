"""Frozen pure-logic contracts for paired embodied routing experiments."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from fractions import Fraction
from statistics import NormalDist
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from himoe_libero_bridge.protocol import ACTION_CHUNK_STEPS, ACTION_DIM, FLOW_NOISE_SHAPE
from himoe_libero_bridge.routing import (
    PRIMARY_ACTION_METRIC,
    PRIMARY_ROUTING_METRIC,
    aligned_total_variation_matrix,
    checkpoint_normalized_action_distance_matrix,
    action_medoid,
)

RAD_EXPERIMENT_SCHEMA = "himoe-libero-rad-paired-v2"
RAD_SUMMARY_SCHEMA = "himoe-libero-rad-paired-summary-v2"

SELECTOR_K1 = "k1"
SELECTOR_RANDOM = "random"
SELECTOR_ACTION_MEDOID = "action_medoid"
SELECTOR_ROUTING_MEDOID = "routing_medoid"
PAIRED_SELECTORS = (
    SELECTOR_K1,
    SELECTOR_RANDOM,
    SELECTOR_ACTION_MEDOID,
    SELECTOR_ROUTING_MEDOID,
)

PRIMARY_PAIRED_COMPARISON = "routing_medoid_vs_random"
SECONDARY_PAIRED_COMPARISONS = (
    "action_medoid_vs_random",
    "routing_medoid_vs_action_medoid",
)
DEPLOYMENT_PAIRED_COMPARISONS = (
    "random_vs_k1",
    "action_medoid_vs_k1",
    "routing_medoid_vs_k1",
)

_NOISE_STREAM_TAG = 0x4E4F4953
_RANDOM_SELECTOR_STREAM_TAG = 0x52414E44
_BOOTSTRAP_STREAM_TAG = 0x424F4F54


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("%s must be a non-negative integer" % name)
    result = int(value)
    if result < 0:
        raise ValueError("%s must be a non-negative integer" % name)
    return result


def generate_noise_schedule(
    replan_count: int, candidate_count: int, noise_seed: int
) -> np.ndarray:
    """Generate `[replan, candidate, 10, 24]` with K-invariant candidate zero.

    Each candidate owns a namespaced RNG stream. Consequently, candidate zero is
    bitwise identical when the same experiment is generated with K=1 or K=8.
    """

    replans = _nonnegative_int(replan_count, "replan_count")
    candidates = _nonnegative_int(candidate_count, "candidate_count")
    seed = _nonnegative_int(noise_seed, "noise_seed")
    if replans < 1 or candidates < 1:
        raise ValueError("replan_count and candidate_count must be at least one")

    schedule = np.empty(
        (replans, candidates) + FLOW_NOISE_SHAPE,
        dtype=np.float32,
    )
    for candidate_index in range(candidates):
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [seed, _NOISE_STREAM_TAG, candidate_index]
            )
        )
        schedule[:, candidate_index] = rng.standard_normal(
            (replans,) + FLOW_NOISE_SHAPE
        ).astype(np.float32)
    return np.ascontiguousarray(schedule)


def validate_noise_schedule(schedule: np.ndarray) -> np.ndarray:
    values = np.asarray(schedule)
    expected_tail = FLOW_NOISE_SHAPE
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError(
            "noise schedule must have shape [replan, candidate, %s], got %s"
            % (expected_tail, values.shape)
        )
    if values.shape[2:] != expected_tail:
        raise ValueError(
            "noise schedule must have shape [replan, candidate, %s], got %s"
            % (expected_tail, values.shape)
        )
    if values.dtype != np.float32:
        raise ValueError("noise schedule dtype must be float32, got %s" % values.dtype)
    if not np.all(np.isfinite(values)):
        raise ValueError("noise schedule contains NaN or infinity")
    return np.ascontiguousarray(values)


def k1_noise_schedule(schedule: np.ndarray) -> np.ndarray:
    """Return the candidate-zero schedule used by the true one-request control."""

    values = validate_noise_schedule(schedule)
    return np.ascontiguousarray(values[:, 0])


def generate_random_selector_schedule(
    replan_count: int, candidate_count: int, random_selector_seed: int
) -> np.ndarray:
    """Generate uniform selector indices from a stream independent of flow noise."""

    replans = _nonnegative_int(replan_count, "replan_count")
    candidates = _nonnegative_int(candidate_count, "candidate_count")
    seed = _nonnegative_int(random_selector_seed, "random_selector_seed")
    if replans < 1 or candidates < 1:
        raise ValueError("replan_count and candidate_count must be at least one")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, _RANDOM_SELECTOR_STREAM_TAG])
    )
    return np.ascontiguousarray(
        rng.integers(0, candidates, size=replans, dtype=np.int32)
    )


@dataclasses.dataclass(frozen=True)
class CandidateSelection:
    selector: str
    index: int
    metric: str
    scores: Optional[np.ndarray] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "selector": self.selector,
            "index": self.index,
            "metric": self.metric,
            "scores": None if self.scores is None else self.scores.tolist(),
        }


def _candidate_actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < 1
        or values.shape[1:] != (ACTION_CHUNK_STEPS, ACTION_DIM)
    ):
        raise ValueError(
            "actions must have shape [candidate, %d, %d], got %s"
            % (ACTION_CHUNK_STEPS, ACTION_DIM, values.shape)
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("actions must be finite")
    return np.ascontiguousarray(values)


def select_candidate(
    selector: str,
    actions: np.ndarray,
    *,
    action_std: Optional[np.ndarray] = None,
    expert_ids: Optional[np.ndarray] = None,
    expert_weights: Optional[np.ndarray] = None,
    random_index: Optional[int] = None,
) -> CandidateSelection:
    """Apply one frozen selector to a candidate pool."""

    action_pool = _candidate_actions(actions)
    candidate_count = action_pool.shape[0]
    if selector == SELECTOR_K1:
        return CandidateSelection(selector, 0, "candidate_zero_control")

    if selector == SELECTOR_RANDOM:
        if random_index is None:
            raise ValueError("random selector requires a precomputed random_index")
        index = _nonnegative_int(random_index, "random_index")
        if index >= candidate_count:
            raise ValueError("random_index is outside the candidate pool")
        return CandidateSelection(selector, index, "uniform_precomputed_index")

    if selector == SELECTOR_ACTION_MEDOID:
        if action_std is None:
            raise ValueError("action medoid requires checkpoint action_std")
        distance = checkpoint_normalized_action_distance_matrix(
            action_pool, action_std
        )
        index, scores = action_medoid(distance)
        return CandidateSelection(selector, index, PRIMARY_ACTION_METRIC, scores)

    if selector == SELECTOR_ROUTING_MEDOID:
        if expert_ids is None or expert_weights is None:
            raise ValueError("routing medoid requires expert_ids and expert_weights")
        ids = np.asarray(expert_ids)
        weights = np.asarray(expert_weights)
        if ids.shape[0] != candidate_count or weights.shape != ids.shape:
            raise ValueError("routing traces must match the action candidate count")
        distance = aligned_total_variation_matrix(ids, weights)
        index, scores = action_medoid(distance)
        return CandidateSelection(selector, index, PRIMARY_ROUTING_METRIC, scores)

    raise ValueError("unknown selector %r; expected one of %s" % (selector, PAIRED_SELECTORS))


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> Dict[str, Any]:
    successes_value = _nonnegative_int(successes, "successes")
    total_value = _nonnegative_int(total, "total")
    if total_value < 1 or successes_value > total_value:
        raise ValueError("successes/total must satisfy 0 <= successes <= total and total >= 1")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    rate = successes_value / total_value
    denominator = 1.0 + z * z / total_value
    center = (rate + z * z / (2.0 * total_value)) / denominator
    radius = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total_value
            + z * z / (4.0 * total_value * total_value)
        )
        / denominator
    )
    lower = 0.0 if successes_value == 0 else max(0.0, center - radius)
    upper = 1.0 if successes_value == total_value else min(1.0, center + radius)
    return {
        "successes": successes_value,
        "total": total_value,
        "rate": rate,
        "confidence": float(confidence),
        "lower": lower,
        "upper": upper,
    }


def exact_mcnemar_two_sided(
    baseline_only_success: int, comparator_only_success: int
) -> float:
    baseline_only = _nonnegative_int(
        baseline_only_success, "baseline_only_success"
    )
    comparator_only = _nonnegative_int(
        comparator_only_success, "comparator_only_success"
    )
    discordant = baseline_only + comparator_only
    if discordant == 0:
        return 1.0
    tail_limit = min(baseline_only, comparator_only)
    tail_count = sum(math.comb(discordant, value) for value in range(tail_limit + 1))
    probability = 2.0 * float(Fraction(tail_count, 1 << discordant))
    return min(1.0, probability)


def _comparison_stream_seed(seed: int, comparison: str) -> np.random.SeedSequence:
    comparison_tag = int.from_bytes(
        hashlib.sha256(comparison.encode("ascii")).digest()[:4], "little"
    )
    return np.random.SeedSequence([seed, _BOOTSTRAP_STREAM_TAG, comparison_tag])


def _task_cluster_bootstrap_interval(
    differences: np.ndarray,
    task_ids: Sequence[str],
    *,
    comparison: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence: float,
) -> Tuple[float, float]:
    rng = np.random.default_rng(
        _comparison_stream_seed(bootstrap_seed, comparison)
    )
    ordered_tasks = tuple(dict.fromkeys(task_ids))
    cluster_means = np.asarray(
        [
            differences[np.asarray(task_ids) == task_id].mean()
            for task_id in ordered_tasks
        ],
        dtype=np.float64,
    )
    cluster_count = len(ordered_tasks)
    estimates = np.empty((bootstrap_samples,), dtype=np.float64)
    batch_size = 1024
    for start in range(0, bootstrap_samples, batch_size):
        stop = min(start + batch_size, bootstrap_samples)
        indices = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count)
        )
        estimates[start:stop] = cluster_means[indices].mean(axis=1)
    alpha = 1.0 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def _paired_comparison(
    canonical: Mapping[str, np.ndarray],
    task_ids: Sequence[str],
    *,
    name: str,
    role: str,
    reference: str,
    comparator: str,
    bootstrap_seed: int,
    bootstrap_samples: int,
    confidence: float,
) -> Dict[str, Any]:
    reference_values = canonical[reference]
    comparator_values = canonical[comparator]
    reference_only = int(np.sum(reference_values & ~comparator_values))
    comparator_only = int(np.sum(~reference_values & comparator_values))
    differences = (
        comparator_values.astype(np.float64) - reference_values.astype(np.float64)
    )
    ordered_tasks = tuple(dict.fromkeys(task_ids))
    task_differences = np.asarray(
        [
            differences[np.asarray(task_ids) == task_id].mean()
            for task_id in ordered_tasks
        ],
        dtype=np.float64,
    )
    lower, upper = _task_cluster_bootstrap_interval(
        differences,
        task_ids,
        comparison=name,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
        confidence=confidence,
    )
    return {
        "name": name,
        "role": role,
        "reference": reference,
        "comparator": comparator,
        "paired_success_delta": float(np.mean(differences)),
        "task_macro_paired_success_delta": float(np.mean(task_differences)),
        "discordant": {
            "reference_success_comparator_failure": reference_only,
            "reference_failure_comparator_success": comparator_only,
            "total": reference_only + comparator_only,
        },
        "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
            reference_only, comparator_only
        ),
        "exact_mcnemar_interpretation": "case_level_sensitivity_analysis",
        "task_cluster_bootstrap": {
            "method": "task_cluster_percentile",
            "resampling_unit": "task_id",
            "estimand": "equal_task_weighted_mean_of_within_task_paired_differences",
            "cluster_count": len(ordered_tasks),
            "single_task_diagnostic_only": len(ordered_tasks) == 1,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence": confidence,
            "lower": lower,
            "upper": upper,
        },
    }


def summarize_paired_success(
    outcomes: Mapping[str, Sequence[bool]],
    *,
    task_ids: Sequence[Any],
    case_ids: Optional[Sequence[str]] = None,
    confidence: float = 0.95,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 10_000,
) -> Dict[str, Any]:
    """Summarize aligned success outcomes for the four frozen experiment arms."""

    if set(outcomes) != set(PAIRED_SELECTORS):
        raise ValueError("outcomes must contain exactly %s" % (PAIRED_SELECTORS,))
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    seed = _nonnegative_int(bootstrap_seed, "bootstrap_seed")
    samples = _nonnegative_int(bootstrap_samples, "bootstrap_samples")
    if samples < 1:
        raise ValueError("bootstrap_samples must be at least one")

    canonical = {}  # type: Dict[str, np.ndarray]
    case_count = None  # type: Optional[int]
    for selector in PAIRED_SELECTORS:
        values = np.asarray(outcomes[selector])
        if values.ndim != 1 or values.dtype != np.bool_:
            raise ValueError("%s outcomes must be a one-dimensional boolean vector" % selector)
        if case_count is None:
            case_count = int(values.size)
        elif values.size != case_count:
            raise ValueError("all selector outcome vectors must have the same length")
        canonical[selector] = np.ascontiguousarray(values)
    assert case_count is not None
    if case_count < 1:
        raise ValueError("paired outcomes must contain at least one case")

    canonical_task_ids = [str(value) for value in task_ids]
    if len(canonical_task_ids) != case_count:
        raise ValueError("task_ids length must match paired outcomes")
    if any(not value for value in canonical_task_ids):
        raise ValueError("task_ids must not contain empty identifiers")

    if case_ids is None:
        canonical_case_ids = [str(index) for index in range(case_count)]
    else:
        canonical_case_ids = [str(value) for value in case_ids]
        if len(canonical_case_ids) != case_count:
            raise ValueError("case_ids length must match paired outcomes")
        if len(set(canonical_case_ids)) != case_count:
            raise ValueError("case_ids must be unique")

    selector_summaries = {}
    for selector in PAIRED_SELECTORS:
        selector_summaries[selector] = wilson_interval(
            int(np.sum(canonical[selector])), case_count, confidence
        )

    comparison_specs = {
        PRIMARY_PAIRED_COMPARISON: (
            "primary",
            SELECTOR_RANDOM,
            SELECTOR_ROUTING_MEDOID,
        ),
        SECONDARY_PAIRED_COMPARISONS[0]: (
            "secondary",
            SELECTOR_RANDOM,
            SELECTOR_ACTION_MEDOID,
        ),
        SECONDARY_PAIRED_COMPARISONS[1]: (
            "secondary",
            SELECTOR_ACTION_MEDOID,
            SELECTOR_ROUTING_MEDOID,
        ),
        DEPLOYMENT_PAIRED_COMPARISONS[0]: (
            "deployment",
            SELECTOR_K1,
            SELECTOR_RANDOM,
        ),
        DEPLOYMENT_PAIRED_COMPARISONS[1]: (
            "deployment",
            SELECTOR_K1,
            SELECTOR_ACTION_MEDOID,
        ),
        DEPLOYMENT_PAIRED_COMPARISONS[2]: (
            "deployment",
            SELECTOR_K1,
            SELECTOR_ROUTING_MEDOID,
        ),
    }
    comparisons = {
        name: _paired_comparison(
            canonical,
            canonical_task_ids,
            name=name,
            role=role,
            reference=reference,
            comparator=comparator,
            bootstrap_seed=seed,
            bootstrap_samples=samples,
            confidence=float(confidence),
        )
        for name, (role, reference, comparator) in comparison_specs.items()
    }

    ordered_tasks = tuple(dict.fromkeys(canonical_task_ids))
    cluster_sizes = {
        task_id: canonical_task_ids.count(task_id) for task_id in ordered_tasks
    }
    return {
        "schema": RAD_SUMMARY_SCHEMA,
        "experiment_schema": RAD_EXPERIMENT_SCHEMA,
        "case_count": case_count,
        "case_ids": canonical_case_ids,
        "task_ids": canonical_task_ids,
        "task_clusters": {
            "cluster_count": len(ordered_tasks),
            "cluster_sizes": cluster_sizes,
            "primary_effect_estimand": (
                "equal_task_weighted_mean_of_within_task_paired_differences"
            ),
            "case_weighted_delta_also_reported": True,
            "single_task_diagnostic_only": len(ordered_tasks) == 1,
        },
        "comparison_protocol": {
            "primary": PRIMARY_PAIRED_COMPARISON,
            "secondary": list(SECONDARY_PAIRED_COMPARISONS),
            "deployment": list(DEPLOYMENT_PAIRED_COMPARISONS),
            "secondary_and_deployment_p_values": "descriptive_unadjusted",
        },
        "selectors": selector_summaries,
        "primary_comparison": comparisons[PRIMARY_PAIRED_COMPARISON],
        "secondary_comparisons": {
            name: comparisons[name] for name in SECONDARY_PAIRED_COMPARISONS
        },
        "deployment_comparisons": {
            name: comparisons[name] for name in DEPLOYMENT_PAIRED_COMPARISONS
        },
    }
