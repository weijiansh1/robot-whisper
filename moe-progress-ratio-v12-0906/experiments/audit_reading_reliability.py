#!/usr/bin/env python3
"""Reading-reliability audit for HiMoE routing detectors.

Many detectors in this workspace are built on the same raw object: the HB
router probability simplex.  Gate sharpness, expert-usage spread, adjacent-query
mobility, hard top-k identity churn, weighted-Jaccard recurrence, routing-graph
metrics and grammar tokens all read ``hb_router_probs``.  If those readings are
numerically the same phenomenon, "four detectors agree" is one measurement
repeated four times, not four pieces of evidence.

Three quantities are routinely conflated and are measured separately here:

``H_gate``   sharpness of a single routing distribution at one (query, layer,
             token), Shannon entropy normalised by ``log 32``;
``H_usage``  entropy of the time-averaged route over a trailing window ``W``,
             i.e. how many distinct experts are used over a stretch of time;
``D_time``   Hellinger distance between adjacent-query routes.

They need not co-vary.  A perfectly uniform route at every step has maximal
``H_gate`` and zero ``D_time``, yet its hard top-4 identity can still churn on
numerical ties.  A route cycling ``A -> B -> C -> A`` with one expert per step
has minimal ``H_gate``, high ``H_usage``, high ``D_time`` and is perfectly
predictable.  Both constructions are exercised in
``tests/test_reading_reliability.py``.

The audit also bounds how much of the hard-identity signal survives float16
storage.  ``hb_router_probs`` is stored as float16.  The gap between the 4th and
5th largest route mass is compared against the local float16 ulp, the top-4 set
is re-derived under one-ulp rounding noise, and the recomputed top-4 set is
compared against ``hb_expert_ids`` -- the set the model itself selected in its
own (higher) precision before storage.

Nothing here trains or fits.  Every number is a descriptive statistic over a
seeded sample of the development corpus.  CPU only.

Outputs (under ``results/reliability``):
  quantities.csv    per (episode, layer) summary statistics
  correlations.csv  pooled and within-cell Spearman matrices
  summary.json      headline numbers, sampling record and provenance
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
V4 = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_CACHE = V4 / "main_reference.npz"
DEFAULT_OUTPUT = BUNDLE / "results/reliability"

SCHEMA = "himoe.reading_reliability.v1"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
N_EXPERTS = 32
N_FLOW = 10
N_TOKEN = 11
FINAL_FLOW = 9
STATE_TOKEN = 0
ACTION = slice(1, 11)
N_ACTION_TOKENS = 10
TOP_K = 4
USAGE_WINDOW = 4
ULP_DRAWS = 20
RAW_SHAPE = (len(LAYER_NAMES), N_FLOW, N_TOKEN, N_EXPERTS)

QUANTITIES = ("H_gate", "H_usage", "D_time", "hard_churn")
EXTRA_QUANTITY = "tie_gap_ulp"
ALL_QUANTITIES = QUANTITIES + (EXTRA_QUANTITY,)

# A (task, chunk) cell needs this many triples and this many alive episodes
# before its Spearman matrix is allowed into the aggregate.
MIN_CELL_TRIPLES = 300
MIN_CELL_EPISODES = 5
MIN_CELL_TRIPLES_LAYER = 100

ULP_BANDS = (1.0, 10.0, 100.0)


# --------------------------------------------------------------------------
# pure helpers -- no I/O, unit tested in tests/test_reading_reliability.py
# --------------------------------------------------------------------------


def normalise(values: np.ndarray) -> np.ndarray:
    """Clip to non-negative and renormalise the last axis to sum 1."""
    output = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return output / np.maximum(output.sum(axis=-1, keepdims=True), 1e-12)


def gate_entropy(probability: np.ndarray) -> np.ndarray:
    """Shannon entropy of the last axis, normalised by ``log(n_expert)``.

    Uniform -> 1.0, one-hot -> 0.0, independent of how many experts there are.
    """
    values = normalise(probability)
    n_expert = values.shape[-1]
    if n_expert < 2:
        raise ValueError("entropy normalisation needs at least two categories")
    safe = np.clip(values, 1e-12, None)
    entropy = -(values * np.log(safe)).sum(axis=-1)
    return (entropy / np.log(n_expert)).astype(np.float32)


def trailing_mean(routes: np.ndarray, window: int) -> np.ndarray:
    """``[Q, ..., E]`` -> ``[Q, ..., E]`` mean over the trailing ``window``.

    Positions ``q < window - 1`` are NaN: the window is not yet full and a
    partial window would silently mix two different time scales.
    """
    values = np.asarray(routes, dtype=np.float32)
    if window < 1:
        raise ValueError("window must be positive")
    if values.ndim < 2:
        raise ValueError("routes must be [query, ..., expert]")
    n_query = values.shape[0]
    output = np.full(values.shape, np.nan, dtype=np.float32)
    for query in range(window - 1, n_query):
        output[query] = values[query - window + 1 : query + 1].mean(axis=0)
    return output


def usage_entropy(routes: np.ndarray, window: int) -> np.ndarray:
    """Entropy of the time-averaged route over a trailing window, / ``log E``.

    This is *not* the average of the per-step entropies.  A one-expert-per-step
    cycle has zero gate entropy at every step yet a high usage entropy.
    """
    averaged = trailing_mean(routes, window)
    finite = np.isfinite(averaged).all(axis=-1)
    output = np.full(averaged.shape[:-1], np.nan, dtype=np.float32)
    if finite.any():
        output[finite] = gate_entropy(averaged[finite])
    return output


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hellinger distance over the last axis.  0 for equal, 1 for disjoint."""
    a = normalise(left)
    b = normalise(right)
    affinity = np.sqrt(a * b).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0)).astype(np.float32)


def adjacent_hellinger(routes: np.ndarray) -> np.ndarray:
    """``[Q, ..., E]`` -> ``[Q, ...]`` with ``D_time(0)`` undefined (NaN)."""
    values = np.asarray(routes, dtype=np.float32)
    output = np.full(values.shape[:-1], np.nan, dtype=np.float32)
    if values.shape[0] > 1:
        output[1:] = hellinger(values[1:], values[:-1])
    return output


def top_k_mask(probability: np.ndarray, k: int) -> np.ndarray:
    """Boolean ``[..., E]`` mask of the ``k`` largest entries.

    Ties are broken towards the lower expert index by a stable descending sort,
    which is deterministic.  Determinism is the point: any churn reported below
    is churn in the *values*, never churn in the tie-break.
    """
    values = np.asarray(probability, dtype=np.float32)
    n_expert = values.shape[-1]
    if not 1 <= k <= n_expert:
        raise ValueError(f"k must lie in [1, {n_expert}], got {k}")
    order = np.argsort(-values, axis=-1, kind="stable")[..., :k]
    mask = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


def churn_between(mask_a: np.ndarray, mask_b: np.ndarray, k: int) -> np.ndarray:
    """``1 - |A n B| / k`` for two top-k boolean masks."""
    overlap = np.logical_and(mask_a, mask_b).sum(axis=-1)
    return (1.0 - overlap / float(k)).astype(np.float32)


def hard_churn(routes: np.ndarray, k: int) -> np.ndarray:
    """``[Q, ..., E]`` -> ``[Q, ...]``; ``1 - |top_k(q) n top_k(q-1)| / k``."""
    values = np.asarray(routes, dtype=np.float32)
    output = np.full(values.shape[:-1], np.nan, dtype=np.float32)
    if values.shape[0] > 1:
        mask = top_k_mask(values, k)
        output[1:] = churn_between(mask[1:], mask[:-1], k)
    return output


def tie_gap(probability: np.ndarray, k: int) -> np.ndarray:
    """``p_(k) - p_(k+1)`` on the descending-sorted last axis."""
    values = np.asarray(probability, dtype=np.float32)
    n_expert = values.shape[-1]
    if not 1 <= k < n_expert:
        raise ValueError(f"k must lie in [1, {n_expert - 1}], got {k}")
    ordered = np.sort(values, axis=-1)[..., ::-1]
    return (ordered[..., k - 1] - ordered[..., k]).astype(np.float32)


def float16_ulp(values: np.ndarray) -> np.ndarray:
    """Local float16 spacing at ``values``, returned as float32.

    ``hb_router_probs`` is stored as float16, so the true pre-storage value of
    any entry is only known to within half of this.
    """
    return np.spacing(np.asarray(values, dtype=np.float16)).astype(np.float32)


def ulp_ratio(gap: np.ndarray, ulp: np.ndarray) -> np.ndarray:
    """``gap / ulp``.  A ratio below 1 means the gap is below storage noise."""
    gap = np.asarray(gap, dtype=np.float32)
    ulp = np.asarray(ulp, dtype=np.float32)
    if gap.shape != ulp.shape:
        raise ValueError("gap and ulp arrays do not align")
    if np.any(ulp <= 0.0):
        raise ValueError("float16 ulp must be strictly positive")
    return (gap / ulp).astype(np.float32)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Ranks with ties resolved to the mean rank of the tied block."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    count = flat.size
    if count == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(flat, kind="stable")
    ordered = flat[order]
    change = np.empty(count, dtype=bool)
    change[0] = True
    change[1:] = ordered[1:] != ordered[:-1]
    group = np.cumsum(change) - 1
    sizes = np.bincount(group)
    totals = np.bincount(group, weights=np.arange(1, count + 1, dtype=np.float64))
    ranked = (totals / sizes)[group]
    output = np.empty(count, dtype=np.float64)
    output[order] = ranked
    return output


def spearman_matrix(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Spearman rho matrix over the named columns, tie-corrected.

    ``hard_churn`` only takes the five values ``{0, .25, .5, .75, 1}``, so its
    rank vector is heavily tied and the tie correction matters.  Attenuation
    from those ties is real and is reported rather than patched.
    """
    names = list(columns)
    stacked = np.stack([np.asarray(columns[name], dtype=np.float64) for name in names])
    finite = np.isfinite(stacked).all(axis=0)
    kept = stacked[:, finite]
    size = len(names)
    output = np.full((size, size), np.nan, dtype=np.float64)
    if kept.shape[1] < 3:
        return output
    ranks = np.stack([average_rank(row) for row in kept])
    centred = ranks - ranks.mean(axis=1, keepdims=True)
    scale = np.sqrt((centred**2).sum(axis=1))
    good = scale > 0.0
    covariance = centred @ centred.T
    denominator = np.outer(scale, scale)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = covariance / denominator
    rho[~good, :] = np.nan
    rho[:, ~good] = np.nan
    np.fill_diagonal(rho, np.where(good, 1.0, np.nan))
    output[:] = np.clip(rho, -1.0, 1.0)
    return output


def fisher_mean(values: np.ndarray) -> float:
    """Unweighted mean of ``atanh(rho)`` mapped back through ``tanh``.

    Unweighted on purpose: the triples inside one (task, chunk) cell share
    layers and tokens, so an ``n - 3`` weight would claim precision the sample
    does not have.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    clipped = np.clip(finite, -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(clipped).mean()))


def conditional_table(
    driver: np.ndarray, response: np.ndarray, n_bins: int
) -> list[dict[str, float]]:
    """Mean ``response`` inside quantile bins of ``driver``.

    Spearman on a five-valued variable is attenuated; a conditional mean is not,
    so this is the honest cross-check on whether two readings move together.
    """
    x = np.asarray(driver, dtype=np.float64)
    y = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        return []
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return [{"bin": 0, "lo": float(x.min()), "hi": float(x.max()),
                 "n": int(x.size), "mean_response": float(y.mean())}]
    index = np.clip(np.digitize(x, edges[1:-1], right=False), 0, len(edges) - 2)
    rows: list[dict[str, float]] = []
    for position in range(len(edges) - 1):
        selected = index == position
        if not selected.any():
            continue
        rows.append(
            {
                "bin": int(position),
                "lo": float(edges[position]),
                "hi": float(edges[position + 1]),
                "n": int(selected.sum()),
                "mean_response": float(y[selected].mean()),
            }
        )
    return rows


def quantile_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Compact distribution record; empty input yields NaN rather than raising."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        keys = ("n", "mean", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "min", "max")
        record = {f"{prefix}_{key}": float("nan") for key in keys}
        record[f"{prefix}_n"] = 0
        return record
    levels = (1, 5, 25, 50, 75, 95, 99)
    percentiles = np.percentile(finite, levels)
    record = {f"{prefix}_n": int(finite.size), f"{prefix}_mean": float(finite.mean())}
    for level, value in zip(levels, percentiles):
        record[f"{prefix}_p{level:02d}"] = float(value)
    record[f"{prefix}_min"] = float(finite.min())
    record[f"{prefix}_max"] = float(finite.max())
    return record


# --------------------------------------------------------------------------
# per-episode measurement
# --------------------------------------------------------------------------


def episode_measurements(
    raw_block: np.ndarray,
    stored_ids: np.ndarray,
    rng: np.random.Generator,
    window: int = USAGE_WINDOW,
    k: int = TOP_K,
    draws: int = ULP_DRAWS,
) -> dict[str, np.ndarray]:
    """All five quantities for one episode's ``[Q, 8, 10, 11, 32]`` block.

    ``raw_block`` arrives as float16 straight off disk and is upcast once here.
    The upcast recovers no precision; it only stops the reductions from
    accumulating in float16.
    """
    raw = np.asarray(raw_block)
    if raw.shape[1:] != RAW_SHAPE:
        raise ValueError(f"expected [Q, 8, 10, 11, 32] block, got {raw.shape}")
    upcast = raw.astype(np.float32)

    # denoising axis, before collapsing to the final flow
    state = upcast[:, :, :, STATE_TOKEN, :]
    state_dev = np.abs(state - state[:, :, 0:1, :]).max(axis=(2, 3))
    action_all = upcast[:, :, :, ACTION, :]
    action_dev = np.abs(action_all - action_all[:, :, 0:1, :, :]).max(axis=(2, 3, 4))

    routes = normalise(upcast[:, :, FINAL_FLOW, ACTION, :])  # [Q, 8, 10, 32]
    n_query = routes.shape[0]

    h_gate = gate_entropy(routes)
    h_usage = usage_entropy(routes, window)
    d_time = adjacent_hellinger(routes)
    churn = hard_churn(routes, k)
    gap = tie_gap(routes, k)
    ordered = np.sort(routes, axis=-1)[..., ::-1]
    ulp = float16_ulp(ordered[..., k - 1])
    ratio = ulp_ratio(gap, np.maximum(ulp, np.finfo(np.float16).smallest_subnormal))
    max_prob = routes.max(axis=-1)

    baseline_mask = top_k_mask(routes, k)

    # recomputed top-k vs the set the model itself stored
    ids = np.asarray(stored_ids[:, :, FINAL_FLOW, ACTION, :], dtype=np.intp)
    stored_mask = np.zeros(routes.shape, dtype=bool)
    np.put_along_axis(stored_mask, ids, True, axis=-1)
    if not np.all(stored_mask.sum(axis=-1) == k):
        raise ValueError("hb_expert_ids does not hold k distinct experts")
    stored_match = (stored_mask == baseline_mask).all(axis=-1)
    stored_overlap = np.logical_and(stored_mask, baseline_mask).sum(axis=-1)

    # One-ulp rounding noise on the upcast probabilities.  Uniform on
    # [-ulp/2, +ulp/2] is exactly the float16 rounding error, so a draw is a
    # plausible pre-storage value rather than an arbitrary jitter.
    #
    # Two things are measured from the same draws.  ``churn_perturbed`` keeps
    # the real time step and asks whether the churn statistic survives storage
    # noise.  ``null_churn`` compares two independent draws of the *same*
    # query, so it is the churn that storage noise alone manufactures when
    # nothing moved -- the floor any hard-identity detector reads for free.
    element_ulp = float16_ulp(routes)
    set_changed = np.zeros(routes.shape[:-1], dtype=np.float64)
    churn_perturbed = np.zeros((n_query,) + routes.shape[1:-1], dtype=np.float64)
    churn_perturbed[0] = np.nan
    null_churn = np.zeros(routes.shape[:-1], dtype=np.float64)
    previous: np.ndarray | None = None
    for _ in range(draws):
        noise = (rng.random(routes.shape, dtype=np.float32) - np.float32(0.5)) * element_ulp
        mask = top_k_mask(routes + noise, k)
        set_changed += ~(mask == baseline_mask).all(axis=-1)
        if n_query > 1:
            churn_perturbed[1:] += churn_between(mask[1:], mask[:-1], k)
        if previous is not None:
            null_churn += churn_between(mask, previous, k)
        previous = mask
    set_changed /= float(draws)
    churn_perturbed /= float(draws)
    null_churn /= float(max(draws - 1, 1))

    return {
        "H_gate": h_gate,
        "H_usage": h_usage,
        "D_time": d_time,
        "hard_churn": churn,
        "tie_gap": gap,
        "tie_gap_ulp": ratio,
        "max_prob": max_prob,
        "ulp_set_changed": set_changed.astype(np.float32),
        "churn_perturbed": churn_perturbed.astype(np.float32),
        "null_churn": null_churn.astype(np.float32),
        "stored_match": stored_match,
        "stored_overlap": stored_overlap.astype(np.float32),
        "state_dev": state_dev,
        "action_dev": action_dev,
    }


# --------------------------------------------------------------------------
# sampling and driving
# --------------------------------------------------------------------------


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sample_rows(
    task_index: np.ndarray, n_tasks: int, per_task: int, seed: int
) -> np.ndarray:
    """Seeded, uniform-without-replacement draw of ``per_task`` rows per task.

    Uniform rather than length-stratified: a length-stratified draw would make
    late chunks look better populated than the corpus actually is, and chunk
    population is exactly what the within-cell analysis has to respect.
    """
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for task in range(n_tasks):
        rows = np.flatnonzero(task_index == task)
        if len(rows) == 0:
            raise ValueError(f"no cohort rows for task position {task}")
        take = min(per_task, len(rows))
        chosen.append(rng.choice(rows, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def scan_stored_dtypes(route_root: Path, tasks: np.ndarray, run_id: str) -> dict[str, Any]:
    """Storage precision of every zarr array, and the float16-grid test.

    A float32 array whose values all sit on the float16 grid would be an upcast
    performed after quantisation -- it would look like extra precision in every
    downstream dtype check while carrying none.
    """
    per_array: dict[str, dict[str, Any]] = {}
    for task in tasks:
        path = route_root / str(task) / run_id / "server/routes.zarr"
        group = zarr.open_group(str(path), mode="r")
        for name in sorted(group.array_keys()):
            array = group[name]
            record = per_array.setdefault(
                name, {"dtypes": set(), "shape_tail": tuple(array.shape[1:]), "tasks": 0}
            )
            record["dtypes"].add(str(array.dtype))
            record["tasks"] += 1
            if array.dtype.kind == "f" and array.dtype.itemsize > 2:
                head = np.asarray(array[: min(16, array.shape[0])])
                finite = head[np.isfinite(head)]
                on_grid = (
                    float(np.mean(finite == finite.astype(np.float16).astype(head.dtype)))
                    if finite.size
                    else float("nan")
                )
                record["frac_on_float16_grid"] = on_grid
    return {
        name: {
            "dtypes": sorted(record["dtypes"]),
            "shape_tail": list(record["shape_tail"]),
            "tasks_checked": record["tasks"],
            "frac_on_float16_grid": record.get("frac_on_float16_grid"),
        }
        for name, record in sorted(per_array.items())
    }


def scan_derived_caches(paths: list[Path]) -> list[dict[str, Any]]:
    """Same float16-grid test for the npz caches this bundle depends on."""
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                if array.dtype.kind != "f":
                    continue
                finite = array[np.isfinite(array)]
                on_grid = (
                    float(np.mean(finite == finite.astype(np.float16).astype(array.dtype)))
                    if finite.size
                    else float("nan")
                )
                records.append(
                    {
                        "path": str(path),
                        "array": name,
                        "dtype": str(array.dtype),
                        "n_finite": int(finite.size),
                        "frac_on_float16_grid": on_grid,
                    }
                )
    return records


def collect(
    cache: dict[str, np.ndarray],
    rows: np.ndarray,
    route_root: Path,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """Stream the sampled episodes and reduce each one immediately."""
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    run_id = str(cache["run_id"])
    rng = np.random.default_rng(seed + 1)

    summaries: list[dict[str, Any]] = []
    pool: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "H_gate",
            "H_usage",
            "D_time",
            "hard_churn",
            "tie_gap_ulp",
            "churn_perturbed",
            "null_churn",
            "task",
            "chunk",
            "layer",
        )
    }
    tie_pool: dict[str, list[np.ndarray]] = {"tie_gap": [], "tie_gap_ulp": [], "layer": [],
                                             "H_gate": [], "max_prob": []}
    denoise_pool: dict[str, list[np.ndarray]] = {"state": [], "action": [], "layer": []}
    stored_pool: list[np.ndarray] = []
    ulp_pool: dict[str, list[np.ndarray]] = {"changed": [], "layer": []}

    total_rows = 0
    for task_position in np.unique(task_index[rows]):
        task = task_names[task_position]
        path = route_root / task / run_id / "server/routes.zarr"
        group = zarr.open_group(str(path), mode="r")
        source = group["hb_router_probs"]
        source_ids = group["hb_expert_ids"]
        if tuple(source.shape[1:]) != RAW_SHAPE:
            raise ValueError(f"unexpected HB shape at {path}: {source.shape}")
        episode_id = np.asarray(group["episode_id"][:], dtype=int)
        wanted = rows[task_index[rows] == task_position]
        for row in wanted:
            episode = int(episodes[row])
            positions = np.flatnonzero(episode_id == episode)
            if len(positions) == 0:
                raise ValueError(f"episode {episode} absent from {path}")
            if len(positions) > 1 and not np.all(np.diff(positions) == 1):
                raise ValueError(f"non-contiguous episode {episode} in {path}")
            if len(positions) != int(lengths[row]):
                raise ValueError(
                    f"length mismatch for {task} episode {episode}: "
                    f"{len(positions)} != {int(lengths[row])}"
                )
            start, stop = int(positions[0]), int(positions[-1]) + 1
            block = np.asarray(source[start:stop])
            ids = np.asarray(source_ids[start:stop])
            total_rows += block.shape[0]
            measured = episode_measurements(block, ids, rng)
            n_query = block.shape[0]

            scored = np.zeros(n_query, dtype=bool)
            scored[max(1, USAGE_WINDOW - 1) :] = True
            layer_grid = np.broadcast_to(
                np.arange(len(LAYER_NAMES))[None, :, None],
                (n_query, len(LAYER_NAMES), N_ACTION_TOKENS),
            )
            chunk_grid = np.broadcast_to(
                np.arange(n_query)[:, None, None],
                (n_query, len(LAYER_NAMES), N_ACTION_TOKENS),
            )
            if scored.any():
                for key in ("H_gate", "H_usage", "D_time", "hard_churn",
                            "tie_gap_ulp", "churn_perturbed", "null_churn"):
                    pool[key].append(measured[key][scored].ravel())
                pool["task"].append(
                    np.full(int(scored.sum()) * len(LAYER_NAMES) * N_ACTION_TOKENS,
                            task_position, dtype=np.int16)
                )
                pool["chunk"].append(chunk_grid[scored].ravel().astype(np.int16))
                pool["layer"].append(layer_grid[scored].ravel().astype(np.int8))

            tie_pool["tie_gap"].append(measured["tie_gap"].ravel())
            tie_pool["tie_gap_ulp"].append(measured["tie_gap_ulp"].ravel())
            tie_pool["H_gate"].append(measured["H_gate"].ravel())
            tie_pool["max_prob"].append(measured["max_prob"].ravel())
            tie_pool["layer"].append(layer_grid.ravel().astype(np.int8))
            ulp_pool["changed"].append(measured["ulp_set_changed"].ravel())
            ulp_pool["layer"].append(layer_grid.ravel().astype(np.int8))
            stored_pool.append(measured["stored_match"].ravel())
            denoise_pool["state"].append(measured["state_dev"].ravel())
            denoise_pool["action"].append(measured["action_dev"].ravel())
            denoise_pool["layer"].append(
                np.broadcast_to(np.arange(len(LAYER_NAMES), dtype=np.int8),
                                (n_query, len(LAYER_NAMES))).ravel()
            )

            for layer_position, layer in enumerate(LAYER_NAMES):
                record: dict[str, Any] = {
                    "task": task,
                    "task_index": int(task_position),
                    "episode": episode,
                    "run_id": run_id,
                    "length": n_query,
                    "layer": layer,
                    "n_triple_total": n_query * N_ACTION_TOKENS,
                    "n_triple_scored": int(scored.sum()) * N_ACTION_TOKENS,
                }
                for key in ("H_gate", "H_usage", "D_time", "hard_churn",
                            "tie_gap", "tie_gap_ulp", "max_prob"):
                    values = measured[key][:, layer_position, :]
                    finite = values[np.isfinite(values)]
                    record[f"{key}_mean"] = float(finite.mean()) if finite.size else np.nan
                    record[f"{key}_p05"] = float(np.percentile(finite, 5)) if finite.size else np.nan
                    record[f"{key}_median"] = float(np.median(finite)) if finite.size else np.nan
                    record[f"{key}_p95"] = float(np.percentile(finite, 95)) if finite.size else np.nan
                ratios = measured["tie_gap_ulp"][:, layer_position, :].ravel()
                for band in ULP_BANDS:
                    record[f"frac_gap_lt_{int(band)}ulp"] = float((ratios < band).mean())
                record["ulp_set_changed_mean"] = float(
                    measured["ulp_set_changed"][:, layer_position, :].mean()
                )
                record["null_churn_mean"] = float(
                    measured["null_churn"][:, layer_position, :].mean()
                )
                perturbed = measured["churn_perturbed"][:, layer_position, :]
                baseline = measured["hard_churn"][:, layer_position, :]
                record["churn_perturbed_mean"] = (
                    float(np.nanmean(perturbed)) if np.isfinite(perturbed).any() else np.nan
                )
                record["churn_baseline_mean"] = (
                    float(np.nanmean(baseline)) if np.isfinite(baseline).any() else np.nan
                )
                record["stored_top4_match_frac"] = float(
                    measured["stored_match"][:, layer_position, :].mean()
                )
                record["stored_top4_overlap_mean"] = float(
                    measured["stored_overlap"][:, layer_position, :].mean()
                )
                state_dev = measured["state_dev"][:, layer_position]
                action_dev = measured["action_dev"][:, layer_position]
                record["denoise_state_max"] = float(state_dev.max())
                record["denoise_state_median"] = float(np.median(state_dev))
                record["denoise_state_frac_zero"] = float((state_dev == 0.0).mean())
                record["denoise_action_max"] = float(action_dev.max())
                record["denoise_action_median"] = float(np.median(action_dev))
                record["denoise_action_frac_zero"] = float((action_dev == 0.0).mean())
                summaries.append(record)
        print(
            f"[{task_position + 1}/{len(task_names)}] {task} "
            f"episodes={len(wanted)} rows={total_rows}",
            flush=True,
        )

    pooled = {key: np.concatenate(value) for key, value in pool.items() if value}
    pooled_tie = {key: np.concatenate(value) for key, value in tie_pool.items()}
    pooled_denoise = {key: np.concatenate(value) for key, value in denoise_pool.items()}
    pooled_ulp = {key: np.concatenate(value) for key, value in ulp_pool.items()}
    stored = np.concatenate(stored_pool)
    extras = {
        "tie": pooled_tie,
        "denoise": pooled_denoise,
        "ulp": pooled_ulp,
        "stored_match": stored,
        "total_rows": total_rows,
    }
    return pd.DataFrame(summaries), pooled, extras


def within_cell_matrices(
    pooled: dict[str, np.ndarray], by_layer: bool
) -> tuple[np.ndarray, np.ndarray, int, list[int]]:
    """Spearman per (task, chunk[, layer]) cell -> (median, fisher, n_cells, sizes)."""
    size = len(ALL_QUANTITIES)
    task = pooled["task"].astype(np.int64)
    chunk = pooled["chunk"].astype(np.int64)
    if by_layer:
        layer = pooled["layer"].astype(np.int64)
        key = (task * 4096 + chunk) * 16 + layer
        minimum = MIN_CELL_TRIPLES_LAYER
    else:
        key = task * 4096 + chunk
        minimum = MIN_CELL_TRIPLES
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    boundaries = np.flatnonzero(np.diff(sorted_key)) + 1
    segments = np.split(order, boundaries)
    collected: list[np.ndarray] = []
    sizes: list[int] = []
    for segment in segments:
        if len(segment) < minimum:
            continue
        if not by_layer:
            n_episode = len(segment) // (len(LAYER_NAMES) * N_ACTION_TOKENS)
            if n_episode < MIN_CELL_EPISODES:
                continue
        columns = {name: pooled[name][segment] for name in ALL_QUANTITIES}
        matrix = spearman_matrix(columns)
        if np.isnan(matrix).all():
            continue
        collected.append(matrix)
        sizes.append(len(segment))
    if not collected:
        empty = np.full((size, size), np.nan)
        return empty, empty, 0, []
    stack = np.stack(collected)
    median = np.nanmedian(stack, axis=0)
    fisher = np.empty((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            fisher[i, j] = fisher_mean(stack[:, i, j])
    return median, fisher, len(collected), sizes


def matrix_rows(matrix: np.ndarray, scope: str, n_obs: int, n_cells: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(ALL_QUANTITIES):
        for j, right in enumerate(ALL_QUANTITIES):
            rows.append(
                {
                    "scope": scope,
                    "var_a": left,
                    "var_b": right,
                    "rho": float(matrix[i, j]),
                    "n_obs": n_obs,
                    "n_cells": n_cells,
                }
            )
    return rows


def render_matrix(matrix: np.ndarray, title: str, names: tuple[str, ...] = ALL_QUANTITIES) -> str:
    width = max(len(name) for name in names) + 2
    header = " " * width + "".join(f"{name:>13}" for name in names)
    lines = [title, header]
    for i, name in enumerate(names):
        cells = "".join(
            "          nan" if not np.isfinite(matrix[i, j]) else f"{matrix[i, j]:>13.3f}"
            for j in range(len(names))
        )
        lines.append(f"{name:<{width}}{cells}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--draws", type=int, default=ULP_DRAWS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    cache = load_cache(args.cache)
    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    run_id = str(cache["run_id"])
    if not np.array_equal(cache["layer_names"].astype(str), np.asarray(LAYER_NAMES)):
        raise ValueError("layer names disagree with the audit module")

    rows = sample_rows(task_index, len(task_names), args.per_task, args.seed)
    print(
        f"sampling {len(rows)} episodes of {len(task_index)} "
        f"({100.0 * len(rows) / len(task_index):.2f}% of the cohort) "
        f"over {len(np.unique(task_index[rows]))}/{len(task_names)} tasks",
        flush=True,
    )

    frame, pooled, extras = collect(cache, rows, args.route_root, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "quantities.csv", index=False)

    # ---- correlation structure -------------------------------------------
    pooled_columns = {name: pooled[name] for name in ALL_QUANTITIES}
    pooled_matrix = spearman_matrix(pooled_columns)
    n_pooled = int(
        np.isfinite(np.stack([pooled[name] for name in ALL_QUANTITIES])).all(axis=0).sum()
    )
    correlation_rows = matrix_rows(pooled_matrix, "pooled", n_pooled, 0)

    per_layer: dict[str, np.ndarray] = {}
    for position, layer in enumerate(LAYER_NAMES):
        selected = pooled["layer"] == position
        columns = {name: pooled[name][selected] for name in ALL_QUANTITIES}
        matrix = spearman_matrix(columns)
        per_layer[layer] = matrix
        n_layer = int(np.isfinite(np.stack(list(columns.values()))).all(axis=0).sum())
        correlation_rows.extend(matrix_rows(matrix, f"pooled_layer_{layer}", n_layer, 0))

    cell_median, cell_fisher, n_cells, cell_sizes = within_cell_matrices(pooled, by_layer=False)
    correlation_rows.extend(
        matrix_rows(cell_median, "within_task_chunk_median", int(sum(cell_sizes)), n_cells)
    )
    correlation_rows.extend(
        matrix_rows(cell_fisher, "within_task_chunk_fisher", int(sum(cell_sizes)), n_cells)
    )
    layer_median, layer_fisher, n_layer_cells, layer_sizes = within_cell_matrices(
        pooled, by_layer=True
    )
    correlation_rows.extend(
        matrix_rows(
            layer_median, "within_task_chunk_layer_median", int(sum(layer_sizes)), n_layer_cells
        )
    )
    correlation_rows.extend(
        matrix_rows(
            layer_fisher, "within_task_chunk_layer_fisher", int(sum(layer_sizes)), n_layer_cells
        )
    )
    pd.DataFrame(correlation_rows).to_csv(args.output / "correlations.csv", index=False)

    # ---- tie-gap table ---------------------------------------------------
    tie = extras["tie"]
    ulp = extras["ulp"]
    tie_rows: list[dict[str, Any]] = []
    for position, layer in list(enumerate(LAYER_NAMES)) + [(-1, "ALL")]:
        selected = slice(None) if position < 0 else (tie["layer"] == position)
        ratios = tie["tie_gap_ulp"][selected]
        gaps = tie["tie_gap"][selected]
        changed = ulp["changed"][slice(None) if position < 0 else (ulp["layer"] == position)]
        record = {
            "layer": layer,
            "n_triple": int(ratios.size),
            "gap_median": float(np.median(gaps)),
            "gap_ulp_median": float(np.median(ratios)),
            "frac_gap_exactly_zero": float((gaps == 0.0).mean()),
        }
        for band in ULP_BANDS:
            record[f"frac_lt_{int(band)}ulp"] = float((ratios < band).mean())
        record["ulp_perturb_top4_change_frac"] = float(changed.mean())
        record["H_gate_median"] = float(np.median(tie["H_gate"][selected]))
        record["max_prob_median"] = float(np.median(tie["max_prob"][selected]))
        tie_rows.append(record)
    tie_frame = pd.DataFrame(tie_rows)

    # ---- denoising axis --------------------------------------------------
    denoise = extras["denoise"]

    def denoise_record(values: np.ndarray) -> dict[str, Any]:
        return {
            **quantile_summary(values, "dev"),
            "frac_exactly_zero": float((values == 0.0).mean()),
            "frac_below_1e_4": float((values < 1e-4).mean()),
        }

    denoise_summary = {
        "state": denoise_record(denoise["state"]),
        "action": denoise_record(denoise["action"]),
        "state_by_layer": {
            layer: denoise_record(denoise["state"][denoise["layer"] == position])
            for position, layer in enumerate(LAYER_NAMES)
        },
        "action_by_layer": {
            layer: denoise_record(denoise["action"][denoise["layer"] == position])
            for position, layer in enumerate(LAYER_NAMES)
        },
    }

    # ---- conditional cross-checks ---------------------------------------
    conditional = {
        "hard_churn_by_D_time_decile": conditional_table(
            pooled["D_time"], pooled["hard_churn"], 10
        ),
        "hard_churn_by_H_gate_decile": conditional_table(
            pooled["H_gate"], pooled["hard_churn"], 10
        ),
        "hard_churn_by_tie_gap_ulp_decile": conditional_table(
            pooled["tie_gap_ulp"], pooled["hard_churn"], 10
        ),
        "D_time_by_H_gate_decile": conditional_table(pooled["H_gate"], pooled["D_time"], 10),
    }

    stored_match = extras["stored_match"]
    elapsed = time.perf_counter() - started

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(elapsed, 2),
        "settings": {
            "cache": str(args.cache),
            "route_root": str(args.route_root),
            "run_id": run_id,
            "per_task": args.per_task,
            "seed": args.seed,
            "usage_window": USAGE_WINDOW,
            "top_k": TOP_K,
            "final_flow": FINAL_FLOW,
            "ulp_draws": args.draws,
            "min_cell_triples": MIN_CELL_TRIPLES,
            "min_cell_episodes": MIN_CELL_EPISODES,
        },
        "sampling": {
            "episodes_sampled": int(len(rows)),
            "episodes_in_cohort": int(len(task_index)),
            "fraction_of_cohort": float(len(rows) / len(task_index)),
            "tasks_sampled": int(len(np.unique(task_index[rows]))),
            "tasks_in_cohort": int(len(task_names)),
            "episodes_per_task": int(args.per_task),
            "rows_read": int(extras["total_rows"]),
            "triples_total": int(tie["tie_gap"].size),
            "triples_scored": int(n_pooled),
            "episode_length_min": int(frame["length"].min()),
            "episode_length_median": float(frame["length"].median()),
            "episode_length_max": int(frame["length"].max()),
            "task_names": sorted(task_names[np.unique(task_index[rows])].tolist()),
        },
        "distributions": {
            **{name: quantile_summary(pooled[name], "v") for name in ALL_QUANTITIES},
            "tie_gap": quantile_summary(tie["tie_gap"], "v"),
            "tie_gap_ulp_all_triples": quantile_summary(tie["tie_gap_ulp"], "v"),
            "max_prob": quantile_summary(tie["max_prob"], "v"),
            "H_gate_all_triples": quantile_summary(tie["H_gate"], "v"),
        },
        "correlations": {
            "variables": list(ALL_QUANTITIES),
            "pooled": pooled_matrix.tolist(),
            "pooled_n": n_pooled,
            "within_task_chunk_median": cell_median.tolist(),
            "within_task_chunk_fisher": cell_fisher.tolist(),
            "within_task_chunk_cells": n_cells,
            "within_task_chunk_layer_median": layer_median.tolist(),
            "within_task_chunk_layer_fisher": layer_fisher.tolist(),
            "within_task_chunk_layer_cells": n_layer_cells,
            "per_layer_pooled": {layer: matrix.tolist() for layer, matrix in per_layer.items()},
        },
        "tie_gap_table": tie_frame.to_dict(orient="records"),
        "quantisation": {
            "ulp_perturb_top4_change_frac": float(ulp["changed"].mean()),
            "ulp_perturb_top4_change_frac_by_layer": {
                layer: float(ulp["changed"][ulp["layer"] == position].mean())
                for position, layer in enumerate(LAYER_NAMES)
            },
            "churn_baseline_mean": float(np.nanmean(pooled["hard_churn"])),
            "churn_perturbed_mean": float(np.nanmean(pooled["churn_perturbed"])),
            "churn_perturbed_over_baseline": float(
                np.nanmean(pooled["churn_perturbed"]) / np.nanmean(pooled["hard_churn"])
            ),
            "null_churn_mean": float(np.nanmean(pooled["null_churn"])),
            "null_churn_over_baseline": float(
                np.nanmean(pooled["null_churn"]) / np.nanmean(pooled["hard_churn"])
            ),
            "null_churn_mean_by_layer": {
                layer: float(np.nanmean(pooled["null_churn"][pooled["layer"] == position]))
                for position, layer in enumerate(LAYER_NAMES)
            },
            "stored_top4_set_match_frac": float(stored_match.mean()),
            "stored_top4_set_disagree_frac": float(1.0 - stored_match.mean()),
        },
        "storage_precision": {
            "zarr_arrays": scan_stored_dtypes(args.route_root, task_names, run_id),
            "derived_caches": scan_derived_caches(
                [
                    args.cache,
                    BUNDLE / "results/progress_cache/development_main.npz",
                ]
            ),
        },
        "denoising_axis": denoise_summary,
        "conditional": conditional,
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    # ---- readable report -------------------------------------------------
    print()
    print(render_matrix(pooled_matrix, f"Spearman, pooled over {n_pooled} scored triples"))
    print()
    print(
        render_matrix(
            cell_median,
            f"Spearman, median over {n_cells} (task, chunk) cells "
            f"[min {MIN_CELL_TRIPLES} triples, {MIN_CELL_EPISODES} episodes]",
        )
    )
    print()
    print(render_matrix(cell_fisher, f"Spearman, Fisher-z mean over {n_cells} (task, chunk) cells"))
    print()
    print("tie gap p_(4) - p_(5) on final-flow action routes, vs local float16 ulp")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(tie_frame.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print()
    print()
    print(
        f"one-ulp perturbation, {args.draws} draws: top-4 set changes for "
        f"{100.0 * ulp['changed'].mean():.1f}% of triples; "
        f"mean hard_churn {summary['quantisation']['churn_baseline_mean']:.3f} -> "
        f"{summary['quantisation']['churn_perturbed_mean']:.3f}"
    )
    print(
        "storage-noise null (two independent one-ulp draws of the same query, "
        f"no real movement): mean churn {summary['quantisation']['null_churn_mean']:.3f}, "
        f"{100.0 * summary['quantisation']['null_churn_over_baseline']:.1f}% of the "
        "observed adjacent-query churn"
    )
    print(
        "recomputed top-4 from the stored float16 route reproduces the stored "
        f"hb_expert_ids set on {100.0 * stored_match.mean():.2f}% of triples "
        "(stable lowest-index tie-break)"
    )
    print()
    print("denoising axis max|P(flow_m) - P(flow_0)| per (query, layer)")
    denoise_frame = pd.DataFrame(
        [
            {
                "layer": layer,
                "state_frac_zero": denoise_summary["state_by_layer"][layer]["frac_exactly_zero"],
                "state_p50": denoise_summary["state_by_layer"][layer]["dev_p50"],
                "state_p95": denoise_summary["state_by_layer"][layer]["dev_p95"],
                "state_max": denoise_summary["state_by_layer"][layer]["dev_max"],
                "action_frac_zero": denoise_summary["action_by_layer"][layer]["frac_exactly_zero"],
                "action_p50": denoise_summary["action_by_layer"][layer]["dev_p50"],
                "action_p95": denoise_summary["action_by_layer"][layer]["dev_p95"],
                "action_max": denoise_summary["action_by_layer"][layer]["dev_max"],
            }
            for layer in LAYER_NAMES
        ]
    )
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(denoise_frame.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(
        "pooled: state exact-zero "
        f"{100.0 * denoise_summary['state']['frac_exactly_zero']:.1f}% "
        f"median {denoise_summary['state']['dev_p50']:.3e} "
        f"max {denoise_summary['state']['dev_max']:.3e}; action exact-zero "
        f"{100.0 * denoise_summary['action']['frac_exactly_zero']:.1f}% "
        f"median {denoise_summary['action']['dev_p50']:.3e} "
        f"max {denoise_summary['action']['dev_max']:.3e}"
    )
    print()
    print(f"wrote {args.output}/quantities.csv correlations.csv summary.json "
          f"in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
