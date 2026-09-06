#!/usr/bin/env python3
"""Extract the three v7 guard inputs from raw HiMoE router tensors.

The three inputs are ``layer_mobility``, ``route_acceleration`` and
``lag_periodicity``.  Every formula here is a batched restatement of
``IntrinsicGuardMonitor._query_features`` in
``moe-v7-0905/method/intrinsic_guard_monitor.py``; the helper functions
(``normalize_probability``, ``hellinger``, ``weighted_jaccard``) and the
structural constants (``BACK``, ``FINAL_FLOW``, ``ACTION``, ``LAGS``,
``N_EXPERTS``) are imported from that frozen module rather than re-declared,
so no feature definition is rewritten in this bundle.

Nothing in this file reads an outcome label.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
V7 = WORKSPACE / "moe-v7-0905"
sys.path.insert(0, str(V7 / "method"))

from intrinsic_guard_monitor import (  # noqa: E402
    ACTION,
    BACK,
    FINAL_FLOW,
    LAGS,
    LAYER_NAMES,
    N_EXPERTS,
    hellinger,
    normalize_probability,
    weighted_jaccard,
)


MAX_QUERY = 52
ROUTER_KEY = "hb_router_probs"
EXPECTED_QUERY_SHAPE = (len(LAYER_NAMES), 10, 11, N_EXPERTS)


@dataclass(frozen=True)
class CohortSpec:
    """Where a cohort's raw routes live and which rows it contains."""

    name: str
    cache_root: Path
    run_id: str
    layer_cache: Path


COHORTS = {
    "external_8b": CohortSpec(
        name="external_8b",
        cache_root=WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
        run_id="right-50x8b-20260903",
        layer_cache=WORKSPACE / "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
    ),
    "development_main": CohortSpec(
        name="development_main",
        cache_root=WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA",
        run_id="right-50x8-20260903",
        layer_cache=WORKSPACE / "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    ),
    "legacy_16x32": CohortSpec(
        name="legacy_16x32",
        cache_root=WORKSPACE / "VLA_MUI_HUB/cache/HiMoE-VLA",
        run_id="right-16x32",
        layer_cache=WORKSPACE / "moe-v4-0904/results/cache16x32_v4/layerwise_mobility.npz",
    ),
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def episode_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mobility [Q, 8], acceleration [Q], periodicity [Q]) for one episode.

    ``raw`` is the contiguous ``hb_router_probs`` block of a single episode,
    shape [Q, 8, 10, 11, 32].  The output is bit-for-bit what a query-by-query
    replay of ``IntrinsicGuardMonitor`` would have accumulated, except for the
    float32 summation order of the acceleration mean (checked downstream).
    """
    raw = np.asarray(raw)
    if raw.ndim != 5 or raw.shape[1:] != EXPECTED_QUERY_SHAPE:
        raise ValueError(f"router block must be [Q, 8, 10, 11, 32], got {raw.shape}")
    queries = len(raw)
    probability = normalize_probability(raw)
    final_action = probability[:, :, FINAL_FLOW, ACTION, :]

    mobility = np.full((queries, len(LAYER_NAMES)), np.nan, dtype=np.float32)
    if queries > 1:
        mobility[1:] = hellinger(final_action[1:], final_action[:-1]).mean(
            axis=2, dtype=np.float32
        )

    action_flow = probability[:, BACK, :, ACTION, :]
    root = np.sqrt(action_flow)
    second = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
    # The monitor accumulates the mean in float32, divides by a float64
    # ``np.sqrt(2.0)`` and only lands in float32 when the history is stacked.
    acceleration = (
        np.linalg.norm(second, axis=-1).mean(axis=(1, 2, 3), dtype=np.float32)
        / np.sqrt(2.0)
    ).astype(np.float32)

    action_route = final_action[:, BACK].reshape(queries, 40, N_EXPERTS)
    similarity = np.full((queries, len(LAGS)), np.nan, dtype=np.float32)
    for position, lag in enumerate(LAGS):
        if queries > lag:
            similarity[lag:, position] = weighted_jaccard(
                action_route[lag:], action_route[:-lag]
            ).mean(axis=1)
    good = np.isfinite(similarity[:, 0]) & np.isfinite(similarity[:, 1:]).any(axis=1)
    periodicity = np.full(queries, np.nan, dtype=np.float32)
    periodicity[good] = (
        np.nanmax(similarity[good, 1:], axis=1) - similarity[good, 0]
    ).astype(np.float32)
    return mobility, acceleration, periodicity


def task_block(
    spec: CohortSpec, task: str, episodes: np.ndarray, lengths: np.ndarray, block: int = 192
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one task's episodes; returns padded [n, 52, 8], [n, 52], [n, 52]."""
    route_path = spec.cache_root / task / spec.run_id / "server/routes.zarr"
    group = zarr.open_group(str(route_path), mode="r")
    source = group[ROUTER_KEY]
    if tuple(source.shape[1:]) != EXPECTED_QUERY_SHAPE:
        raise ValueError(f"unexpected router shape for {task}: {source.shape}")
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)

    count = len(episodes)
    mobility = np.full((count, MAX_QUERY, len(LAYER_NAMES)), np.nan, dtype=np.float32)
    acceleration = np.full((count, MAX_QUERY), np.nan, dtype=np.float32)
    periodicity = np.full((count, MAX_QUERY), np.nan, dtype=np.float32)

    for position in range(0, count, block):
        stop = min(position + block, count)
        wanted = episodes[position:stop]
        spans = []
        for local, episode in enumerate(wanted):
            index = np.flatnonzero(episode_id == int(episode))
            expected = int(lengths[position + local])
            if len(index) != expected or (len(index) > 1 and not np.all(np.diff(index) == 1)):
                raise ValueError(f"raw/route mismatch for {task} episode {episode}")
            spans.append((position + local, int(index[0]), int(index[-1]) + 1))
        lo = min(span[1] for span in spans)
        hi = max(span[2] for span in spans)
        chunk = np.asarray(source[lo:hi], dtype=np.float16)
        for row, start, end in spans:
            local_mobility, local_acceleration, local_periodicity = episode_features(
                chunk[start - lo : end - lo]
            )
            length = end - start
            mobility[row, :length] = local_mobility
            acceleration[row, :length] = local_acceleration
            periodicity[row, :length] = local_periodicity
    return mobility, acceleration, periodicity


def extract_cohort_task(payload: tuple[str, str, np.ndarray, np.ndarray]) -> tuple[str, tuple]:
    cohort, task, episodes, lengths = payload
    return task, task_block(COHORTS[cohort], task, episodes, lengths)
