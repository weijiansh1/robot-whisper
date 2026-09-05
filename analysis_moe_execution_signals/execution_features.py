"""Numerical primitives for actual Top-K MoE execution features.

The route store contains both the full fp16 router distribution and the IDs
that were selected before the cast to fp16.  Support-sensitive quantities must
therefore use the stored IDs rather than recomputing Top-K from the stored
probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class BlockFeatures:
    scalars: dict[str, np.ndarray]
    token_maps: dict[str, np.ndarray]
    layer_maps: dict[str, np.ndarray]
    final_ids: np.ndarray
    final_weights: np.ndarray
    audit: dict[str, float | int]


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), EPS)


def combine_weights(selected_probability: np.ndarray) -> np.ndarray:
    """Recover the weights used by HBMoE after Top-K renormalization."""
    return normalize_probabilities(selected_probability)


def support_jaccard_distance(ids_a: np.ndarray, ids_b: np.ndarray) -> np.ndarray:
    """Jaccard distance between unordered, unique Top-K support sets."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    if ids_a.shape != ids_b.shape:
        raise ValueError("support arrays must have identical shapes")
    if ids_a.ndim < 1:
        raise ValueError("support arrays need a Top-K axis")
    matches = ids_a[..., :, None] == ids_b[..., None, :]
    intersection = matches.any(axis=-1).sum(axis=-1, dtype=np.int16)
    union = 2 * ids_a.shape[-1] - intersection
    return 1.0 - intersection / np.maximum(union, 1)


def sparse_hellinger_distance(
    ids_a: np.ndarray,
    weights_a: np.ndarray,
    ids_b: np.ndarray,
    weights_b: np.ndarray,
) -> np.ndarray:
    """Hellinger distance between two sparse Top-K execution mixtures."""
    ids_a = np.asarray(ids_a)
    ids_b = np.asarray(ids_b)
    weights_a = combine_weights(weights_a)
    weights_b = combine_weights(weights_b)
    if not (ids_a.shape == ids_b.shape == weights_a.shape == weights_b.shape):
        raise ValueError("IDs and weights must have identical shapes")
    matches = ids_a[..., :, None] == ids_b[..., None, :]
    products = np.sqrt(
        np.maximum(weights_a[..., :, None], 0.0)
        * np.maximum(weights_b[..., None, :], 0.0)
    )
    coefficient = (products * matches).sum(axis=(-2, -1))
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def actual_support_boundary(
    probabilities: np.ndarray,
    selected_ids: np.ndarray,
    selected_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return actual fourth/fifth boundary values and their expert IDs.

    The fourth expert is the minimum-probability member of the support that the
    runtime actually selected.  The fifth expert is the maximum-probability
    member outside that support.  This remains well-defined when fp16 ties make
    a recomputed ``topk`` disagree with the runtime decision.
    """
    probabilities = np.asarray(probabilities, np.float32)
    selected_ids = np.asarray(selected_ids)
    selected_probability = np.asarray(selected_probability, np.float32)
    if probabilities.shape[:-1] != selected_ids.shape[:-1]:
        raise ValueError("probability and support prefixes differ")
    if selected_ids.shape != selected_probability.shape:
        raise ValueError("selected IDs and probabilities differ")

    support_mask = np.zeros(probabilities.shape, dtype=bool)
    np.put_along_axis(support_mask, selected_ids.astype(np.intp), True, axis=-1)
    fifth_probability = np.where(support_mask, -np.inf, probabilities).max(axis=-1)
    fifth_id = np.where(support_mask, -np.inf, probabilities).argmax(axis=-1)
    fourth_slot = selected_probability.argmin(axis=-1)
    fourth_probability = np.take_along_axis(
        selected_probability, fourth_slot[..., None], axis=-1
    )[..., 0]
    fourth_id = np.take_along_axis(
        selected_ids, fourth_slot[..., None], axis=-1
    )[..., 0]
    return fourth_probability - fifth_probability, fourth_id, fifth_id, fifth_probability


def normalized_entropy(weights: np.ndarray) -> np.ndarray:
    weights = combine_weights(weights)
    entropy = -(weights * np.log(np.maximum(weights, EPS))).sum(axis=-1)
    return entropy / np.log(weights.shape[-1])


def linear_position_slope(values: np.ndarray) -> np.ndarray:
    """Least-squares slope along the final axis with fixed centered positions."""
    values = np.asarray(values, np.float32)
    if values.shape[-1] < 2:
        raise ValueError("at least two positions are required")
    x = np.arange(values.shape[-1], dtype=np.float32)
    x -= x.mean()
    return np.sum(values * x, axis=-1) / np.sum(x * x)


def extract_block_features(
    probabilities: np.ndarray,
    selected_ids: np.ndarray,
    selected_probability: np.ndarray,
    *,
    final_denoise: int = 9,
    late_flow_start: int = 6,
    action_start: int = 1,
    late_layer_start: int = 4,
) -> BlockFeatures:
    """Reduce a contiguous route-store block to execution-aware descriptors."""
    p = np.asarray(probabilities, np.float32)
    ids = np.asarray(selected_ids)
    raw = np.asarray(selected_probability, np.float32)
    if p.ndim != 5 or ids.ndim != 5 or raw.shape != ids.shape:
        raise ValueError("expected p=[B,L,D,S,E] and support=[B,L,D,S,K]")
    if p.shape[:-1] != ids.shape[:-1]:
        raise ValueError("probability and support axes differ")
    if not (0 <= final_denoise < p.shape[2]):
        raise ValueError("final denoise index is out of range")
    if not (0 <= late_flow_start < final_denoise):
        raise ValueError("late flow start must precede final denoise")
    if not (0 <= action_start < p.shape[3]):
        raise ValueError("action start is out of range")
    if not (0 <= late_layer_start < p.shape[1]):
        raise ValueError("late layer start is out of range")

    weights = combine_weights(raw)
    margin, fourth_id, fifth_id, fifth_probability = actual_support_boundary(
        p, ids, raw
    )
    selected_mass = raw.sum(axis=-1)
    tail_mass = np.clip(1.0 - selected_mass, 0.0, 1.0)
    exec_entropy = normalized_entropy(raw)

    current_ids = ids[:, :, late_flow_start + 1 :, action_start:, :]
    previous_ids = ids[:, :, late_flow_start:-1, action_start:, :]
    current_weights = weights[:, :, late_flow_start + 1 :, action_start:, :]
    previous_weights = weights[:, :, late_flow_start:-1, action_start:, :]
    support_flow = support_jaccard_distance(current_ids, previous_ids)
    exec_flow = sparse_hellinger_distance(
        current_ids, current_weights, previous_ids, previous_weights
    )

    final = (slice(None), slice(None), final_denoise, slice(action_start, None))
    late = (slice(None), slice(late_layer_start, None))
    token_support = support_flow.mean(axis=(1, 2))
    token_exec = exec_flow.mean(axis=(1, 2))
    token_margin = margin[final].mean(axis=1)
    token_tail = tail_mass[final].mean(axis=1)
    token_entropy = exec_entropy[final].mean(axis=1)
    layer_support = support_flow.mean(axis=(2, 3))
    layer_exec = exec_flow.mean(axis=(2, 3))

    late_final_margin = margin[final][:, late_layer_start:]
    late_final_fifth = fifth_probability[final][:, late_layer_start:]
    late_final_tail = tail_mass[final][:, late_layer_start:]
    late_final_entropy = exec_entropy[final][:, late_layer_start:]
    late_support = support_flow[late]
    late_exec = exec_flow[late]

    scalars = {
        "top45_margin": late_final_margin.mean(axis=(1, 2)),
        "top45_log_margin": np.log(
            np.maximum(late_final_margin + late_final_fifth, EPS)
        ).mean(axis=(1, 2)) - np.log(np.maximum(late_final_fifth, EPS)).mean(axis=(1, 2)),
        "tail_mass": late_final_tail.mean(axis=(1, 2)),
        "exec_entropy": late_final_entropy.mean(axis=(1, 2)),
        "late_flow_support_churn": late_support.mean(axis=(1, 2, 3)),
        "late_flow_exec_churn": late_exec.mean(axis=(1, 2, 3)),
        "all_layer_support_churn": support_flow.mean(axis=(1, 2, 3)),
        "all_layer_exec_churn": exec_flow.mean(axis=(1, 2, 3)),
        "token_support_slope": linear_position_slope(token_support),
        "token_exec_slope": linear_position_slope(token_exec),
        "token_margin_slope": linear_position_slope(token_margin),
        "layer_support_slope": linear_position_slope(layer_support),
    }
    token_maps = {
        "top45_margin": token_margin,
        "tail_mass": token_tail,
        "exec_entropy": token_entropy,
        "late_flow_support_churn": token_support,
        "late_flow_exec_churn": token_exec,
    }
    layer_maps = {
        "late_flow_support_churn": layer_support,
        "late_flow_exec_churn": layer_exec,
    }
    final_ids = ids[:, :, final_denoise].astype(np.uint8, copy=False)
    final_weights = weights[:, :, final_denoise].astype(np.float16)
    boundary_pairs = np.stack((fourth_id, fifth_id), axis=-1)
    # Match the established quantization audit's deterministic expert-index
    # tie break.  ``argpartition`` would manufacture disagreement at p4=p5.
    recomputed_ids = np.argsort(-p, axis=-1, kind="stable")[..., : ids.shape[-1]]
    recomputed_disagrees = support_jaccard_distance(ids, recomputed_ids) > 0
    audit = {
        "rows": int(p.shape[0]),
        "boundary_negative": int((margin < 0).sum()),
        "boundary_zero": int((margin == 0).sum()),
        "boundary_count": int(margin.size),
        "fp16_topk_set_disagreement": int(recomputed_disagrees.sum()),
        "boundary_pair_unique": int(
            np.unique(boundary_pairs.reshape(-1, 2), axis=0).shape[0]
        ),
        "selected_mass_min": float(selected_mass.min()),
        "selected_mass_max": float(selected_mass.max()),
        "combine_sum_max_error": float(np.abs(weights.sum(axis=-1) - 1.0).max()),
    }
    return BlockFeatures(
        scalars=scalars,
        token_maps=token_maps,
        layer_maps=layer_maps,
        final_ids=final_ids,
        final_weights=final_weights,
        audit=audit,
    )
