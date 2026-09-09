"""Outcome-blind structured features for Best-of-N chunk critics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _probabilities(value: np.ndarray) -> np.ndarray:
    probs = np.asarray(value, dtype=np.float32)
    if probs.ndim != 5 or probs.shape[-1] < 2:
        raise ValueError("router probabilities must have shape [K,L,D,U,E]")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("router probabilities must be finite and non-negative")
    total = probs.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("router probabilities have a zero-mass row")
    return probs / total


def _js(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    midpoint = 0.5 * (left + right)
    epsilon = np.finfo(np.float32).tiny
    return 0.5 * (
        np.sum(left * (np.log(left + epsilon) - np.log(midpoint + epsilon)), axis=-1)
        + np.sum(right * (np.log(right + epsilon) - np.log(midpoint + epsilon)), axis=-1)
    )


def _flow_slices(steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps <= 0:
        raise ValueError("at least one flow step is required")
    chunks = np.array_split(np.arange(steps, dtype=np.int64), 3)
    return tuple(chunk if len(chunk) else np.asarray([steps - 1]) for chunk in chunks)  # type: ignore[return-value]


def _summarize(probs: np.ndarray) -> np.ndarray:
    """Summarize [K,L,D,U,E] while retaining layer and expert identity."""

    p = _probabilities(probs)
    k, layers, steps, _tokens, _experts = p.shape
    early, middle, late = _flow_slices(steps)
    probability_blocks = (
        p.mean(axis=(2, 3)),
        p.std(axis=(2, 3)),
        p[:, :, early].mean(axis=(2, 3)),
        p[:, :, middle].mean(axis=(2, 3)),
        p[:, :, late].mean(axis=(2, 3)),
        p[:, :, -1].mean(axis=2),
    )
    epsilon = np.finfo(np.float32).tiny
    entropy = -np.sum(p * np.log(p + epsilon), axis=-1)
    ordered = np.sort(p, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    scalars = [
        entropy.mean(axis=(2, 3)),
        entropy.std(axis=(2, 3)),
        entropy[:, :, -1].mean(axis=2),
        margin.mean(axis=(2, 3)),
        margin.std(axis=(2, 3)),
        margin[:, :, -1].mean(axis=2),
    ]
    if steps > 1:
        adjacent_js = _js(p[:, :, :-1], p[:, :, 1:])
        adjacent_l1 = np.abs(p[:, :, :-1] - p[:, :, 1:]).sum(axis=-1)
        top1 = np.argmax(p, axis=-1)
        churn = top1[:, :, :-1] != top1[:, :, 1:]
        scalars.extend(
            [
                adjacent_js.mean(axis=(2, 3)),
                adjacent_js.std(axis=(2, 3)),
                adjacent_l1.mean(axis=(2, 3)),
                churn.mean(axis=(2, 3), dtype=np.float32),
            ]
        )
    else:
        scalars.extend([np.zeros((k, layers), dtype=np.float32)] * 4)
    token_disagreement = p.std(axis=3).mean(axis=(2, 3))
    scalars.append(token_disagreement)
    return np.concatenate(
        [*(block.reshape(k, -1) for block in probability_blocks), *scalars], axis=1
    ).astype(np.float32)


def route_candidate_features(
    probabilities: np.ndarray, flow_indices: Sequence[int] | None = None
) -> np.ndarray:
    """Candidate-specific action-token routing features."""

    p = _probabilities(probabilities)
    if flow_indices is not None:
        indices = np.asarray(flow_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= p.shape[2]):
            raise ValueError("flow_indices are outside the routing trajectory")
        p = p[:, :, indices]
    action = p[:, :, :, 1:, :]
    state = p[:, :, :, :1, :]
    features = _summarize(action)
    state_action_js = _js(state, action).mean(axis=(2, 3))
    return np.concatenate([features, state_action_js], axis=1).astype(np.float32)


def route_state_features(
    probabilities: np.ndarray, flow_indices: Sequence[int] | None = None
) -> np.ndarray:
    """Candidate-mean state-token routing, returned as one snapshot row."""

    p = _probabilities(probabilities)
    if flow_indices is not None:
        indices = np.asarray(flow_indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= p.shape[2]):
            raise ValueError("flow_indices are outside the routing trajectory")
        p = p[:, :, indices]
    state = p[:, :, :, :1, :].mean(axis=0, keepdims=True)
    return _summarize(state)[0]


def hidden_candidate_features(hidden: np.ndarray) -> np.ndarray:
    """Return [K,6,L,H] action-token router-input aggregates."""

    value = np.asarray(hidden, dtype=np.float32)
    if value.ndim != 5 or value.shape[3] < 2:
        raise ValueError("HB hidden state must have shape [K,L,D,U,H] with action tokens")
    action = value[:, :, :, 1:, :]
    early, middle, late = _flow_slices(action.shape[2])
    result = np.stack(
        [
            action.mean(axis=(2, 3)),
            action.std(axis=(2, 3)),
            action[:, :, early].mean(axis=(2, 3)),
            action[:, :, middle].mean(axis=(2, 3)),
            action[:, :, late].mean(axis=(2, 3)),
            action[:, :, -1].mean(axis=2),
        ],
        axis=1,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("HB hidden aggregates contain NaN or infinity")
    return result.astype(np.float32)


def hidden_state_features(hidden: np.ndarray) -> np.ndarray:
    """Return candidate-mean [6,L,H] state-token router-input aggregates."""

    value = np.asarray(hidden, dtype=np.float32)
    if value.ndim != 5 or value.shape[3] < 1:
        raise ValueError("HB hidden state must have shape [K,L,D,U,H]")
    state = value[:, :, :, 0, :].mean(axis=0, keepdims=True)
    early, middle, late = _flow_slices(state.shape[2])
    result = np.stack(
        [
            state.mean(axis=2),
            state.std(axis=2),
            state[:, :, early].mean(axis=2),
            state[:, :, middle].mean(axis=2),
            state[:, :, late].mean(axis=2),
            state[:, :, -1],
        ],
        axis=1,
    )[0]
    if not np.all(np.isfinite(result)):
        raise ValueError("HB hidden state aggregates contain NaN or infinity")
    return result.astype(np.float32)


def as_route_features(probabilities: np.ndarray) -> np.ndarray:
    """AS-MoE negative-control features from [K,L,E] probabilities."""

    p = np.asarray(probabilities, dtype=np.float32)
    if p.ndim != 3 or p.shape[-1] < 2 or np.any(p < 0) or not np.all(np.isfinite(p)):
        raise ValueError("AS probabilities must have shape [K,L,E]")
    p = p / p.sum(axis=-1, keepdims=True)
    entropy = -np.sum(p * np.log(p + np.finfo(np.float32).tiny), axis=-1)
    ordered = np.sort(p, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    return np.concatenate([p.reshape(len(p), -1), entropy, margin], axis=1)
