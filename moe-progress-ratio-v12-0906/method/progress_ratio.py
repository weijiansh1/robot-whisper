#!/usr/bin/env python3
"""Train-free, task-agnostic progress ratio over HiMoE routing trajectories.

R(q, W) = d(z_q, z_{q-W}) / sum_{k=1..W} d(z_{q-k+1}, z_{q-k})

The numerator is net displacement and the denominator is path length, both in
Hellinger units, so the ratio is dimensionless.  Task scale cancels by
construction rather than by an episode baseline.
"""

from __future__ import annotations

import numpy as np


SCHEMA = "himoe.progress_ratio_v12.profile.v1"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
LAYER_GROUPS = {"front": slice(0, 4), "back": slice(4, 8), "all": slice(0, 8)}
FINAL_FLOW = 9
ACTION = slice(1, 11)
N_EXPERTS = 32
N_ACTION_TOKENS = 10
EPSILON = 1e-6

W_GRID = (2, 3, 4, 6, 8, 10, 12)
LAGS = (1,) + W_GRID


def normalize_probability(values: np.ndarray) -> np.ndarray:
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def action_route(hb_router_probs: np.ndarray) -> np.ndarray:
    """[..., 8, 10, 11, 32] -> [..., 8, 10, 32] final-flow action routes."""
    probability = np.asarray(hb_router_probs, dtype=np.float32)
    expected = (len(LAYER_NAMES), 10, 11, N_EXPERTS)
    if probability.shape[-4:] != expected:
        raise ValueError(
            f"router probability must end with shape {expected}, got {probability.shape}"
        )
    return normalize_probability(probability[..., FINAL_FLOW, ACTION, :])


def route_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Mean-over-action-token Hellinger.  [..., 8, 10, 32] x same -> [..., 8].

    A positive-coefficient mean of metrics is a metric, so the triangle
    inequality survives the token reduction.  That is what guarantees D <= L.
    """
    return hellinger(left, right).mean(axis=-1, dtype=np.float32)


def lag_distances(routes: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """[Q, 8, 10, 32] -> [Q, len(lags), 8]; 元素 [q, i, l] = d_l(q, q - lags[i]).

    q < lag 处为 NaN。
    """
    routes = np.asarray(routes, dtype=np.float32)
    if routes.ndim != 4 or routes.shape[1:] != (
        len(LAYER_NAMES),
        N_ACTION_TOKENS,
        N_EXPERTS,
    ):
        raise ValueError(f"routes must be [query, 8, 10, 32], got {routes.shape}")
    n_query = len(routes)
    output = np.full((n_query, len(lags), len(LAYER_NAMES)), np.nan, dtype=np.float32)
    for index, lag in enumerate(lags):
        if lag < 1:
            raise ValueError("lags must be positive")
        if n_query > lag:
            output[lag:, index] = route_distance(routes[lag:], routes[:-lag])
    return output


def path_length(adjacent: np.ndarray, window: int) -> np.ndarray:
    """[E, Q, 8] 相邻距离 -> [E, Q, 8] 窗口路径长。q < window 处为 NaN。"""
    values = np.asarray(adjacent, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"adjacent distances must be [episode, query, layer], got {values.shape}")
    if window < 1:
        raise ValueError("window must be positive")
    output = np.full_like(values, np.nan)
    for query in range(window, values.shape[1]):
        block = values[:, query - window + 1 : query + 1]
        good = np.isfinite(block).all(axis=1)
        output[:, query] = np.where(good, block.sum(axis=1, dtype=np.float32), np.nan)
    return output


def progress_ratio(
    displacement: np.ndarray, length: np.ndarray, eps_length: float
) -> np.ndarray:
    """R = D / L，其中 L < eps_length 时取极限值 0（没有任何净运动）。"""
    numerator = np.asarray(displacement, dtype=np.float32)
    denominator = np.asarray(length, dtype=np.float32)
    if numerator.shape != denominator.shape:
        raise ValueError("displacement and path length do not align")
    if not np.isfinite(eps_length) or eps_length <= 0.0:
        raise ValueError("eps_length must be finite and positive")
    output = np.full_like(numerator, np.nan)
    finite = np.isfinite(numerator) & np.isfinite(denominator)
    frozen = finite & (denominator < eps_length)
    moving = finite & ~frozen
    output[frozen] = 0.0
    output[moving] = np.clip(numerator[moving] / denominator[moving], 0.0, 1.0)
    return output


def group_ratio(ratio: np.ndarray, group: str) -> np.ndarray:
    """[E, Q, 8] -> [E, Q]，组内取中位数。"""
    if group not in LAYER_GROUPS:
        raise ValueError(f"unknown layer group: {group}")
    values = np.asarray(ratio, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != len(LAYER_NAMES):
        raise ValueError(f"ratio must be [episode, query, 8], got {values.shape}")
    subset = values[:, :, LAYER_GROUPS[group]]
    output = np.full(subset.shape[:2], np.nan, dtype=np.float32)
    good = np.isfinite(subset).all(axis=2)
    output[good] = np.median(subset[good], axis=1).astype(np.float32)
    return output


def row_min(values: np.ndarray) -> np.ndarray:
    """每条轨迹的全程最小值；全 NaN 行返回 NaN。"""
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmin(values[good], axis=1)
    return output


def quantile_lower(values: np.ndarray, quantile: float) -> float:
    """逐轨迹最小值的下尾分位数，v7 quantile_higher 的对偶。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = np.sort(finite[np.isfinite(finite)])
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if len(finite) < 32:
        raise ValueError(f"at least 32 finite references are required, got {len(finite)}")
    position = int(np.floor(quantile * (len(finite) - 1)))
    return float(finite[position])


def persistent_low(values: np.ndarray, confirmations: int) -> np.ndarray:
    """滚动最大值。低于阈值即代表窗口内每一个 query 都低于阈值。"""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"persistent_low expects [episode, query], got {values.shape}")
    if confirmations < 1:
        raise ValueError("confirmations must be positive")
    if confirmations == 1:
        return values.copy()
    output = np.full_like(values, np.nan)
    for query in range(confirmations - 1, values.shape[1]):
        window = values[:, query - confirmations + 1 : query + 1]
        good = np.isfinite(window).all(axis=1)
        output[good, query] = window[good].max(axis=1)
    return output


def first_below(
    values: np.ndarray, threshold: float, valid: np.ndarray
) -> np.ndarray:
    """首次跌破阈值的 query 下标；从未跌破返回 -1。"""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape:
        raise ValueError("score and validity masks do not align")
    trigger = np.isfinite(values) & (values < threshold) & valid
    any_trigger = trigger.any(axis=1)
    first = np.full(len(values), -1, dtype=np.int16)
    first[any_trigger] = trigger[any_trigger].argmax(axis=1).astype(np.int16)
    return first
