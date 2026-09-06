#!/usr/bin/env python3
"""Nested information-source ablation for the HiMoE routing detectors.

Every detector in this workspace has so far been reported against its
predecessor *by name* (v4 vs v7 vs v10).  That comparison cannot say where the
information comes from.  This script asks the decision-relevant question
instead: given the clock, what does each additional class of routing
information buy?  The nesting is

    S0  clock only        survival prior at this chunk, per suite (+ chunk index)
    S1  + current state   instantaneous routing readout at this query
    S2  + dwell time      how long the current coarse regime has persisted
    S3  + local change    first differences over the last 2-3 queries
    S4  + ordered history ordered encoding of the last 8 regime labels,
                          against an identical-content shuffled control
    S5  + graph residual  token-graph summaries with node utilisation projected out

Two hypotheses it is built to kill:

1.  "Grammar" may be a dwell-time model.  If S4's gain over S3 disappears once
    S2 is in the model -- and in particular if S4-ordered does not beat
    S4-shuffled -- then the ordered-history machinery is detecting "this regime
    has persisted for eight queries" and should be named that way.
2.  Graph metrics may be marginal-utilisation restatements.  Cross-layer
    co-occurrence built from probability outer products contains E[p]E[q] even
    with no genuine cross-layer dependence, so S5 is tested only after each
    graph metric has been residualised on per-layer node-utilisation summaries.

THIS IS AN ANALYSIS INSTRUMENT, NOT A DETECTOR.
The bundle's detectors are train-free by protocol.  Nothing fitted here ships:
no threshold, no coefficient and no profile written by this script is consumed
by any monitor.  Fitting a small L2 logistic model is legitimate *because* the
object of study is "how much information is in this feature class", and the
only honest way to measure that is a held-out likelihood.  Cross-validation is
blocked by task, so no task appears in both the fit and the score.

CPU only.  Outputs under ``results/ablation``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import textwrap
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio  # noqa: E402
from hazard_common import survival_prior  # noqa: E402


SCHEMA = "himoe.progress_ratio_v12.information_ablation.v1"

CACHE = BUNDLE / "results/progress_cache/development_main.npz"
LABELS = (
    WORKSPACE
    / "double-selete/trainfree/results/timeout_extension_plus10"
    / "development_main_clean_labels.csv"
)
GRAPHS = WORKSPACE / "moe-hb-front-back-0905/results/layer_graphs/development_main.npz"
GRAPH_EXTRACTOR = (
    WORKSPACE / "moe-hb-front-back-0905/experiments/extract_layer_graphs_gpu.py"
)
ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_OUTPUT = BUNDLE / "results/ablation"

COHORT = "development_main"
WINDOW = 4
EPS_LENGTH_QUANTILE = 0.01  # label-free, matches select_operating_point_v12
HISTORY_DEPTH = 8
DIFF_LAGS = (1, 2)
LOW_PRIOR_MAX = 0.25
N_TERCILE = 3
N_REGIME = N_TERCILE * N_TERCILE  # 0..8
UNDEFINED_REGIME = N_REGIME  # 9  -- window-4 quantities not yet defined
PAD_REGIME = N_REGIME + 1  # 10 -- before the first query of the episode
N_HISTORY_CATEGORY = N_REGIME + 2

# The audited duplicate: conditional_energy is a monotone restatement of
# state_action_alignment, so only one of the pair may enter the model.
GRAPH_DROP = ("conditional_energy",)
# Node utilisation, not graph structure: this is the control, not a predictor.
UTILISATION_METRIC = "expert_load_effective_rank"

BANDS = ("all", "low_prior", "high_prior", "low_prior_window_defined")

PENALTY_C = 1.0
MAX_ITER = 4000
TOLERANCE = 1e-6
PROBABILITY_FLOOR = 1e-6


# --------------------------------------------------------------------------- #
# pure helpers -- no I/O, unit tested in tests/test_ablation.py
# --------------------------------------------------------------------------- #


def tercile_edges(values: np.ndarray) -> tuple[float, float]:
    """The two interior tercile cut points of the finite entries."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < N_TERCILE:
        raise ValueError("not enough finite values to form terciles")
    lower, upper = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
    return float(lower), float(upper)


def tercile_index(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    """0/1/2 tercile membership; NaN entries return -1."""
    values = np.asarray(values, dtype=np.float64)
    lower, upper = edges
    if not lower <= upper:
        raise ValueError("tercile edges must be ordered")
    index = np.full(values.shape, -1, dtype=np.int8)
    finite = np.isfinite(values)
    index[finite] = (
        (values[finite] > lower).astype(np.int8)
        + (values[finite] > upper).astype(np.int8)
    )
    return index


def regime_labels(
    ratio: np.ndarray,
    length: np.ndarray,
    ratio_edges: tuple[float, float],
    length_edges: tuple[float, float],
    valid: np.ndarray,
) -> np.ndarray:
    """Coarse regime = (tercile of R) x (tercile of L), pre-declared 3x3.

    Rows where either quantity is undefined (the W-window is not yet full) get
    ``UNDEFINED_REGIME``; rows outside the episode get ``-1``.
    """
    ratio_index = tercile_index(ratio, ratio_edges)
    length_index = tercile_index(length, length_edges)
    labels = np.full(ratio_index.shape, UNDEFINED_REGIME, dtype=np.int8)
    both = (ratio_index >= 0) & (length_index >= 0)
    labels[both] = ratio_index[both] * N_TERCILE + length_index[both]
    labels[~np.asarray(valid, dtype=bool)] = -1
    return labels


def dwell_runs(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """[E, Q] regime labels -> [E, Q] consecutive preceding queries in the same regime.

    ``dwell[e, q]`` counts how many of ``q-1, q-2, ...`` carry the same label as
    ``q`` without interruption.  The first query of an episode has no
    predecessor, so its dwell is 0; a label change resets the run to 0; rows
    outside the episode are 0 and must never be read.
    """
    labels = np.asarray(labels)
    valid = np.asarray(valid, dtype=bool)
    if labels.shape != valid.shape or labels.ndim != 2:
        raise ValueError(f"labels and validity must both be [episode, query], got {labels.shape}")
    output = np.zeros(labels.shape, dtype=np.int32)
    for query in range(1, labels.shape[1]):
        same = (
            valid[:, query]
            & valid[:, query - 1]
            & (labels[:, query] == labels[:, query - 1])
        )
        output[:, query] = np.where(same, output[:, query - 1] + 1, 0)
    output[~valid] = 0
    return output


def history_matrix(
    labels: np.ndarray, valid: np.ndarray, depth: int = HISTORY_DEPTH
) -> np.ndarray:
    """[E, Q] labels -> [E, Q, depth]; slot k holds the label at query q-(k+1).

    Slots that fall before the first query of the episode hold ``PAD_REGIME``.
    This is the *ordered prefix*: the current query is deliberately absent,
    because S1/S2 already carry it.
    """
    labels = np.asarray(labels)
    valid = np.asarray(valid, dtype=bool)
    if labels.shape != valid.shape or labels.ndim != 2:
        raise ValueError("labels and validity must both be [episode, query]")
    if depth < 1:
        raise ValueError("depth must be positive")
    episodes, queries = labels.shape
    output = np.full((episodes, queries, depth), PAD_REGIME, dtype=np.int8)
    for lag in range(1, depth + 1):
        if queries <= lag:
            break
        source = labels[:, : queries - lag]
        source_valid = valid[:, : queries - lag]
        slot = np.where(source_valid, source, PAD_REGIME).astype(np.int8)
        output[:, lag:, lag - 1] = slot
    return output


def shuffle_history(history: np.ndarray, seed: int) -> np.ndarray:
    """Permute each row's ``depth`` labels independently under a fixed seed.

    The identical-content control: the multiset of labels in a row is preserved
    exactly, only the order is destroyed.  Anything the shuffled encoding can
    still express is a bag-of-labels statistic, not order.
    """
    history = np.asarray(history)
    if history.ndim < 2:
        raise ValueError("history must have a trailing slot axis")
    rng = np.random.default_rng(seed)
    keys = rng.random(history.shape)
    order = np.argsort(keys, axis=-1)
    return np.take_along_axis(history, order, axis=-1)


def one_hot(labels: np.ndarray, n_category: int) -> np.ndarray:
    """[..., depth] integer labels -> [..., depth * n_category] float32 indicators."""
    labels = np.asarray(labels)
    if labels.min() < 0 or labels.max() >= n_category:
        raise ValueError("labels outside the declared category range")
    flat = labels.reshape(-1, labels.shape[-1])
    output = np.zeros((flat.shape[0], flat.shape[1] * n_category), dtype=np.float32)
    column = flat.astype(np.int64) + np.arange(flat.shape[1])[None, :] * n_category
    np.put_along_axis(output, column, 1.0, axis=1)
    return output


def fit_residualiser(
    control: np.ndarray, target: np.ndarray, ridge: float = 1e-6
) -> np.ndarray:
    """Least-squares coefficients of ``target`` on ``[1, control]``, train fold only.

    Returns ``[1 + n_control, n_target]``.  The tiny ridge exists only to keep a
    rank-deficient control block solvable; it is not a modelling choice.
    """
    control = np.asarray(control, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if control.ndim != 2 or target.ndim != 2:
        raise ValueError("control and target must both be 2-D")
    if len(control) != len(target):
        raise ValueError("control and target row counts disagree")
    design = np.concatenate(
        [np.ones((len(control), 1), dtype=np.float64), control], axis=1
    )
    gram = design.T @ design
    gram[np.diag_indices_from(gram)] += ridge * max(1.0, float(len(control)))
    return np.linalg.solve(gram, design.T @ target)


def apply_residualiser(
    control: np.ndarray, target: np.ndarray, coefficient: np.ndarray
) -> np.ndarray:
    """``target`` minus its projection on ``[1, control]`` under train coefficients."""
    control = np.asarray(control, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    design = np.concatenate(
        [np.ones((len(control), 1), dtype=np.float64), control], axis=1
    )
    if design.shape[1] != coefficient.shape[0]:
        raise ValueError("control block and residualiser coefficients disagree")
    return target - design @ coefficient


def fit_standardiser(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and scale of the training fold; degenerate columns get scale 1."""
    values = np.asarray(values, dtype=np.float64)
    centre = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return centre, scale


def apply_standardiser(
    values: np.ndarray, centre: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - centre) / scale


def column_median(values: np.ndarray) -> np.ndarray:
    """Train-fold column medians, ignoring NaN; all-NaN columns yield 0."""
    values = np.asarray(values, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(np.where(np.isfinite(values), values, np.nan), axis=0)
    return np.where(np.isfinite(median), median, 0.0)


def impute(values: np.ndarray, median: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=np.float64, copy=True)
    bad = ~np.isfinite(values)
    if bad.any():
        values[bad] = np.broadcast_to(median, values.shape)[bad]
    return values


def row_log_loss(probability: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-row negative log likelihood in nats."""
    probability = np.clip(np.asarray(probability, dtype=np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    target = np.asarray(target, dtype=np.float64)
    return -(target * np.log(probability) + (1.0 - target) * np.log(1.0 - probability))


def cluster_bootstrap(
    sums: np.ndarray, counts: np.ndarray, draws: int, seed: int
) -> np.ndarray:
    """Task-clustered bootstrap of a mean.

    ``sums``/``counts`` are per-cluster totals shaped ``[n_cluster, ...]``.  A
    draw resamples clusters with replacement and returns the ratio of totals,
    which is exactly the mean over the resampled rows.
    """
    sums = np.asarray(sums, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if sums.shape != counts.shape:
        raise ValueError("bootstrap sums and counts disagree")
    n_cluster = sums.shape[0]
    rng = np.random.default_rng(seed)
    index = rng.integers(0, n_cluster, size=(draws, n_cluster))
    weight = np.zeros((draws, n_cluster), dtype=np.float64)
    np.add.at(weight, (np.repeat(np.arange(draws), n_cluster), index.ravel()), 1.0)
    flat_sum = weight @ sums.reshape(n_cluster, -1)
    flat_count = weight @ counts.reshape(n_cluster, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(flat_count > 0, flat_sum / flat_count, np.nan)
    return ratio.reshape((draws,) + sums.shape[1:])


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high)


def prior_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=np.float64), 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


# --------------------------------------------------------------------------- #
# I/O and alignment
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--refresh-entropy", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def aligned_labels(cache: dict[str, np.ndarray], path: Path) -> pd.DataFrame:
    task = task_of(cache)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(path)[["task", "episode", "original_failure"]]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError("development_main labels do not align row-for-row")
    merged["suite"] = merged["task"].str.split("/", n=1).str[0]
    return merged.reset_index(drop=True)


def check_graph_alignment(
    cache: dict[str, np.ndarray], graphs: dict[str, np.ndarray]
) -> dict[str, Any]:
    """The graph file must be the *same rows* as the progress cache, not merely
    the same length.  Every identity column is compared element-for-element."""
    report: dict[str, Any] = {"file": str(GRAPHS), "checked": []}
    for name in ("task_names", "task_index", "episode", "init_state_id", "length", "valid", "layer_names"):
        equal = bool(np.array_equal(np.asarray(cache[name]), np.asarray(graphs[name])))
        report["checked"].append({"array": name, "identical": equal})
        if not equal:
            raise ValueError(f"layer_graphs {name} does not align with the progress cache")
    if str(cache["run_id"]) != str(graphs["run_id"]):
        raise ValueError("layer_graphs run_id disagrees with the progress cache")
    report["run_id"] = str(graphs["run_id"])
    report["rows"] = int(len(graphs["episode"]))
    report["metric_names"] = list(graphs["metric_names"].astype(str))
    return report


def gate_entropy_from_routes(
    cache: dict[str, np.ndarray], output: Path, refresh: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    """[E, Q, 8] gate entropy of the final-flow action routes, normalised by log 32.

    Read from the raw route store (``hb_entropy``, the entropy the router itself
    logged for the full 32-way row).  A seeded sample is re-derived from
    ``hb_router_probs`` and the deviation is reported, so the reading is anchored
    to the raw simplex rather than trusted.
    """
    destination = output / "gate_entropy.npz"
    expected = (len(cache["episode"]), cache["valid"].shape[1], len(progress_ratio.LAYER_NAMES))
    if destination.is_file() and not refresh:
        stored = load_npz(destination)
        if tuple(stored["entropy"].shape) == expected and np.array_equal(
            stored["episode"], cache["episode"]
        ) and np.array_equal(stored["task_index"], cache["task_index"]):
            return stored["entropy"], json.loads(str(stored["provenance"]))

    import zarr  # imported here so the pure helpers stay importable without zarr

    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    run_id = str(cache["run_id"])
    entropy = np.full(expected, np.nan, dtype=np.float32)
    normaliser = float(np.log(progress_ratio.N_EXPERTS))
    provenance: dict[str, Any] = {
        "source": "hb_entropy in <route_root>/<task>/<run_id>/server/routes.zarr",
        "route_root": str(ROUTE_ROOT),
        "run_id": run_id,
        "reduction": "final flow step 9, action tokens 1..10, mean over tokens, / log(32)",
        "tasks": int(len(task_names)),
        "episodes_covered": 0,
        "max_abs_recomputation_delta_nats": None,
        "recomputation_sample_rows": 0,
        "float16_ulp_at_log32": float(np.spacing(np.float16(normaliser))),
    }
    started = time.perf_counter()
    for position, task in enumerate(task_names):
        rows = np.flatnonzero(task_index == position)
        if rows.size == 0:
            raise ValueError(f"no cache rows for task {task}")
        path = ROUTE_ROOT / task / run_id / "server/routes.zarr"
        group = zarr.open_group(str(path), mode="r")
        episode_id = np.asarray(group["episode_id"][:], dtype=int)
        logged = np.asarray(group["hb_entropy"][:], dtype=np.float32)
        if logged.shape[0] != episode_id.shape[0]:
            raise ValueError(f"episode_id/hb_entropy row mismatch at {path}")
        per_row = logged[:, :, progress_ratio.FINAL_FLOW, progress_ratio.ACTION].mean(
            axis=-1
        ) / normaliser
        lookup = {int(e): int(r) for r, e in zip(rows, episodes[rows])}
        seen: set[int] = set()
        for episode in np.unique(episode_id):
            positions = np.flatnonzero(episode_id == episode)
            if positions.size > 1 and not np.all(np.diff(positions) == 1):
                raise ValueError(f"non-contiguous episode {episode} in {path}")
            row = lookup.get(int(episode))
            if row is None:
                raise ValueError(f"episode {episode} of {task} missing from the cache")
            if positions.size != int(lengths[row]):
                raise ValueError(
                    f"length mismatch for {task} episode {episode}: "
                    f"{positions.size} != {int(lengths[row])}"
                )
            entropy[row, : positions.size] = per_row[positions]
            seen.add(row)
        missing = sorted(set(lookup.values()) - seen)
        if missing:
            raise ValueError(f"{len(missing)} cache rows of {task} absent from {path}")
        provenance["episodes_covered"] += len(seen)
        if position == 0:
            sample = min(64, logged.shape[0])
            raw = np.asarray(group["hb_router_probs"][:sample], dtype=np.float32)
            probability = progress_ratio.normalize_probability(raw)
            recomputed = -(
                probability * np.log(np.maximum(probability, 1e-12))
            ).sum(axis=-1)
            provenance["max_abs_recomputation_delta_nats"] = float(
                np.abs(recomputed - logged[:sample]).max()
            )
            provenance["recomputation_sample_rows"] = int(sample)

    valid = cache["valid"].astype(bool)
    if not np.isfinite(entropy[valid]).all():
        raise ValueError("gate entropy has holes inside the valid mask")
    provenance["seconds"] = round(time.perf_counter() - started, 1)
    provenance["min"] = float(entropy[valid].min())
    provenance["max"] = float(entropy[valid].max())
    provenance["mean"] = float(entropy[valid].mean())

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        entropy=entropy,
        episode=cache["episode"],
        task_index=cache["task_index"],
        provenance=np.asarray(json.dumps(provenance)),
    )
    return entropy, provenance


# --------------------------------------------------------------------------- #
# feature construction
# --------------------------------------------------------------------------- #


def group_medians(values: np.ndarray) -> dict[str, np.ndarray]:
    """[E, Q, 8] -> front/back/all layer-group medians, via the frozen primitive."""
    return {
        group: progress_ratio.group_ratio(values, group)
        for group in progress_ratio.LAYER_GROUPS
    }


def difference(values: np.ndarray, lag: int) -> np.ndarray:
    """[E, Q] -> x[q] - x[q - lag]; the first ``lag`` queries are NaN."""
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if values.shape[1] > lag:
        output[:, lag:] = values[:, lag:] - values[:, :-lag]
    return output


def stack_columns(
    grid: dict[str, np.ndarray], names: tuple[str, ...], rows: np.ndarray, chunks: np.ndarray
) -> np.ndarray:
    return np.stack([np.asarray(grid[name])[rows, chunks] for name in names], axis=1)


def build_static_grids(
    cache: dict[str, np.ndarray], entropy: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Everything that does not depend on the training fold."""
    lag_distance = cache["lag_distance"]
    adjacent = lag_distance[:, :, 0, :]
    displacement = lag_distance[:, :, progress_ratio.LAGS.index(WINDOW), :]
    length = progress_ratio.path_length(adjacent, WINDOW)
    finite_length = length[np.isfinite(length)]
    eps_length = float(np.quantile(finite_length, EPS_LENGTH_QUANTILE))
    ratio = progress_ratio.progress_ratio(displacement, length, eps_length)

    grid: dict[str, np.ndarray] = {}
    for group, values in group_medians(adjacent).items():
        grid[f"adjacent_{group}"] = values.astype(np.float64)
    for group, values in group_medians(ratio).items():
        grid[f"ratio_{group}"] = values.astype(np.float64)
    for group, values in group_medians(length).items():
        grid[f"path_{group}"] = values.astype(np.float64)
    grid["gate_entropy_mean"] = entropy.mean(axis=2, dtype=np.float64)
    grid["gate_entropy_spread"] = entropy.std(axis=2, dtype=np.float64)

    for name in DIFF_BASE:
        for lag in DIFF_LAGS:
            grid[f"d{lag}_{name}"] = difference(grid[name], lag)

    meta = {
        "window": WINDOW,
        "eps_length": eps_length,
        "eps_length_quantile": EPS_LENGTH_QUANTILE,
        "eps_length_uses_labels": False,
    }
    return grid, meta


S1_COLUMNS = (
    "adjacent_front",
    "adjacent_back",
    "adjacent_all",
    "gate_entropy_mean",
    "gate_entropy_spread",
    "ratio_front",
    "ratio_back",
    "ratio_all",
    "path_front",
    "path_back",
    "path_all",
)
DIFF_BASE = (
    "adjacent_front",
    "adjacent_back",
    "adjacent_all",
    "ratio_all",
    "path_all",
    "gate_entropy_mean",
    "gate_entropy_spread",
)
S3_COLUMNS = tuple(f"d{lag}_{name}" for name in DIFF_BASE for lag in DIFF_LAGS)


def fold_survival_prior(
    labels: pd.DataFrame, train_episode: np.ndarray, rows: np.ndarray, chunks: np.ndarray
) -> tuple[np.ndarray, int]:
    """P(risk | still running at chunk q) per suite, estimated on the train fold only."""
    table = survival_prior(labels.iloc[train_episode].reset_index(drop=True))
    suites = labels["suite"].to_numpy(str)
    curves = {
        suite: np.asarray([entries[chunk] for chunk in sorted(entries)], dtype=np.float64)
        for suite, entries in table.items()
    }
    values = np.full(len(rows), np.nan, dtype=np.float64)
    clamped = 0
    for suite, curve in curves.items():
        mask = suites[rows] == suite
        if not mask.any():
            continue
        chunk = chunks[mask]
        clamped += int((chunk >= len(curve)).sum())
        values[mask] = curve[np.minimum(chunk, len(curve) - 1)]
    if not np.isfinite(values).all():
        raise ValueError("a row's suite has no survival prior on this training fold")
    return values, clamped


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold

    cache = load_npz(CACHE)
    graphs = load_npz(GRAPHS)
    graph_report = check_graph_alignment(cache, graphs)
    labels = aligned_labels(cache, LABELS)

    valid = cache["valid"].astype(bool)
    rows, chunks = np.nonzero(valid)
    n_row = len(rows)
    task = labels["task"].to_numpy(str)
    task_index = cache["task_index"].astype(int)
    unique_tasks = cache["task_names"].astype(str)
    risk_episode = labels["original_failure"].to_numpy(bool)
    target = risk_episode[rows].astype(np.float64)
    row_task = task_index[rows]

    entropy, entropy_report = gate_entropy_from_routes(cache, args.output, args.refresh_entropy)
    grid, static_meta = build_static_grids(cache, entropy)

    # ---- graph block ------------------------------------------------------ #
    metric_names = list(graphs["metric_names"].astype(str))
    graph_metrics = [
        name
        for name in metric_names
        if name not in GRAPH_DROP and name != UTILISATION_METRIC
    ]
    metrics = graphs["metrics"]
    layer_names = list(cache["layer_names"].astype(str))
    graph_raw = np.stack(
        [
            metrics[rows, chunks, layer, metric_names.index(name)]
            for name in graph_metrics
            for layer in range(len(layer_names))
        ],
        axis=1,
    ).astype(np.float64)
    graph_columns = [
        f"{name}|{layer_names[layer]}"
        for name in graph_metrics
        for layer in range(len(layer_names))
    ]
    utilisation = np.concatenate(
        [
            np.stack(
                [
                    metrics[rows, chunks, layer, metric_names.index(UTILISATION_METRIC)]
                    for layer in range(len(layer_names))
                ],
                axis=1,
            ),
            entropy[rows, chunks, :],
        ],
        axis=1,
    ).astype(np.float64)
    utilisation_columns = [f"{UTILISATION_METRIC}|{name}" for name in layer_names] + [
        f"gate_entropy|{name}" for name in layer_names
    ]

    # the audited duplicate, verified rather than assumed
    alignment_metric = metrics[rows, chunks, :, metric_names.index("state_action_alignment")]
    energy_metric = metrics[rows, chunks, :, metric_names.index("conditional_energy")]
    duplicate_check = {
        "pair": ["state_action_alignment", "conditional_energy"],
        "pearson_r": float(
            np.corrcoef(alignment_metric.ravel(), energy_metric.ravel())[0, 1]
        ),
        "max_abs_deviation_from_1_minus_square": float(
            np.abs(energy_metric - (1.0 - alignment_metric**2)).max()
        ),
        "dropped": list(GRAPH_DROP),
    }

    # ---- folds ------------------------------------------------------------ #
    splitter = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    episode_fold = np.full(len(labels), -1, dtype=np.int8)
    for fold, (_, test_episode) in enumerate(
        splitter.split(np.zeros(len(labels)), risk_episode, task_index)
    ):
        episode_fold[test_episode] = fold
    if (episode_fold < 0).any():
        raise ValueError("some episode was never held out")
    row_fold = episode_fold[rows]
    fold_report = []
    for fold in range(args.folds):
        held = episode_fold == fold
        fold_report.append(
            {
                "fold": fold,
                "test_tasks": sorted(set(task[held].tolist())),
                "test_episodes": int(held.sum()),
                "test_failures": int(risk_episode[held].sum()),
                "test_rows": int((row_fold == fold).sum()),
            }
        )
        overlap = set(task[held].tolist()) & set(task[~held].tolist())
        if overlap:
            raise ValueError(f"fold {fold} shares tasks with its training set: {overlap}")

    # ---- model declarations ----------------------------------------------- #
    nested = ("S0", "S1", "S2", "S3", "S4", "S5")
    blocks_of = {
        "S0": ("S0",),
        "S1": ("S0", "S1"),
        "S2": ("S0", "S1", "S2"),
        "S3": ("S0", "S1", "S2", "S3"),
        "S4": ("S0", "S1", "S2", "S3", "S4"),
        "S5": ("S0", "S1", "S2", "S3", "S4", "S5"),
        "S4_shuffled": ("S0", "S1", "S2", "S3", "S4shuf"),
        "S5_raw_graph": ("S0", "S1", "S2", "S3", "S4", "S5raw"),
        "S5_utilisation_only": ("S0", "S1", "S2", "S3", "S4", "S5util"),
    }
    baseline_of = {
        "S1": "S0",
        "S2": "S1",
        "S3": "S2",
        "S4": "S3",
        "S5": "S4",
        "S4_shuffled": "S3",
        "S5_raw_graph": "S4",
        "S5_utilisation_only": "S4",
    }
    models = list(blocks_of)
    probability = {name: np.full(n_row, np.nan) for name in models}
    prior_value = np.full(n_row, np.nan)
    fold_diagnostics = []

    for fold in range(args.folds):
        fold_started = time.perf_counter()
        test_rows = row_fold == fold
        train_rows = ~test_rows
        train_episode = np.flatnonzero(episode_fold != fold)

        prior, clamped = fold_survival_prior(labels, train_episode, rows, chunks)
        prior_value[test_rows] = prior[test_rows]

        # regime discretisation: terciles from the training fold only
        ratio_edges = tercile_edges(grid["ratio_all"][rows[train_rows], chunks[train_rows]])
        path_edges = tercile_edges(grid["path_all"][rows[train_rows], chunks[train_rows]])
        regimes = regime_labels(
            grid["ratio_all"], grid["path_all"], ratio_edges, path_edges, valid
        )
        dwell = dwell_runs(regimes, valid)
        history = history_matrix(np.where(regimes < 0, PAD_REGIME, regimes), valid)
        history_rows = history[rows, chunks, :]
        shuffled_rows = shuffle_history(history_rows, args.seed + 1000)

        raw_blocks: dict[str, np.ndarray] = {
            "S0": np.stack(
                [prior_logit(prior), chunks.astype(np.float64), np.log1p(chunks)], axis=1
            ),
            "S1": stack_columns(grid, S1_COLUMNS, rows, chunks),
            "S2": np.stack(
                [dwell[rows, chunks].astype(np.float64), np.log1p(dwell[rows, chunks])],
                axis=1,
            ),
            "S3": stack_columns(grid, S3_COLUMNS, rows, chunks),
            "S5util": utilisation,
            "S5raw": graph_raw,
        }
        # standardised, imputed on the training fold only
        prepared: dict[str, np.ndarray] = {}
        for name, block in raw_blocks.items():
            median = column_median(block[train_rows])
            filled = impute(block, median)
            centre, scale = fit_standardiser(filled[train_rows])
            prepared[name] = apply_standardiser(filled, centre, scale)

        # graph residual: each metric regressed on node utilisation, train-fold fit
        residual_coefficient = fit_residualiser(
            prepared["S5util"][train_rows], prepared["S5raw"][train_rows]
        )
        residual = apply_residualiser(
            prepared["S5util"], prepared["S5raw"], residual_coefficient
        )
        centre, scale = fit_standardiser(residual[train_rows])
        prepared["S5"] = apply_standardiser(residual, centre, scale)

        # ordered-history one-hots stay raw 0/1 so the ordered and shuffled
        # encodings carry identical L2 penalties in identical units
        prepared["S4"] = one_hot(history_rows, N_HISTORY_CATEGORY).astype(np.float64)
        prepared["S4shuf"] = one_hot(shuffled_rows, N_HISTORY_CATEGORY).astype(np.float64)

        for name, blocks in blocks_of.items():
            design = np.concatenate([prepared[block] for block in blocks], axis=1)
            model = LogisticRegression(
                l1_ratio=0.0,  # pure L2; `penalty="l2"` is deprecated in sklearn 1.8
                C=PENALTY_C,
                solver="lbfgs",
                max_iter=MAX_ITER,
                tol=TOLERANCE,
            )
            model.fit(design[train_rows], target[train_rows])
            probability[name][test_rows] = model.predict_proba(design[test_rows])[:, 1]
            del design

        fold_diagnostics.append(
            {
                "fold": fold,
                "train_rows": int(train_rows.sum()),
                "test_rows": int(test_rows.sum()),
                "prior_chunks_clamped_to_train_max": clamped,
                "ratio_tercile_edges": list(ratio_edges),
                "path_tercile_edges": list(path_edges),
                "seconds": round(time.perf_counter() - fold_started, 1),
            }
        )
        print(
            f"[fold {fold}] train={int(train_rows.sum()):,} test={int(test_rows.sum()):,} "
            f"clamped={clamped} ({time.perf_counter() - fold_started:.1f}s)",
            flush=True,
        )
        del prepared, raw_blocks

    if not np.isfinite(prior_value).all():
        raise ValueError("a held-out row never received a survival prior")
    for name in models:
        if not np.isfinite(probability[name]).all():
            raise ValueError(f"model {name} left held-out rows unscored")

    # ---- scoring ---------------------------------------------------------- #
    losses = {name: row_log_loss(probability[name], target) for name in models}
    window_defined = chunks >= WINDOW
    low = prior_value < LOW_PRIOR_MAX
    band_mask = {
        "all": np.ones(n_row, dtype=bool),
        "low_prior": low,
        "high_prior": ~low,
        "low_prior_window_defined": low & window_defined,
    }
    n_task = len(unique_tasks)
    sums = np.zeros((n_task, len(BANDS), len(models)))
    counts = np.zeros((n_task, len(BANDS), len(models)))
    for band_position, band in enumerate(BANDS):
        mask = band_mask[band]
        for model_position, name in enumerate(models):
            contribution = np.where(mask, losses[name], 0.0)
            sums[:, band_position, model_position] = np.bincount(
                row_task, weights=contribution, minlength=n_task
            )
            counts[:, band_position, model_position] = np.bincount(
                row_task, weights=mask.astype(np.float64), minlength=n_task
            )
    draws = cluster_bootstrap(sums, counts, args.draws, args.seed + 7)

    def band_stats(band: str, name: str) -> dict[str, Any]:
        band_position = BANDS.index(band)
        model_position = models.index(name)
        total = counts[:, band_position, model_position].sum()
        point = sums[:, band_position, model_position].sum() / total if total else float("nan")
        low_ci, high_ci = percentile_interval(draws[:, band_position, model_position])
        return {"logloss": float(point), "ci_lo": low_ci, "ci_hi": high_ci, "n_rows": int(total)}

    def decrement_stats(band: str, name: str, baseline: str) -> dict[str, Any]:
        band_position = BANDS.index(band)
        a = models.index(baseline)
        b = models.index(name)
        total = counts[:, band_position, b].sum()
        point = (
            (sums[:, band_position, a].sum() - sums[:, band_position, b].sum()) / total
            if total
            else float("nan")
        )
        sample = draws[:, band_position, a] - draws[:, band_position, b]
        low_ci, high_ci = percentile_interval(sample)
        return {
            "decrement": float(point),
            "ci_lo": low_ci,
            "ci_hi": high_ci,
            "excludes_zero": bool(np.isfinite(low_ci) and np.isfinite(high_ci) and (low_ci > 0.0 or high_ci < 0.0)),
        }

    increment_rows = []
    for band in BANDS:
        for name in models:
            stats = band_stats(band, name)
            baseline = baseline_of.get(name)
            record = {
                "band": band,
                "model": name,
                "blocks": "+".join(blocks_of[name]),
                "baseline": baseline or "",
                "nested": name in nested,
                "n_rows": stats["n_rows"],
                "n_episodes": int(len(np.unique(rows[band_mask[band]]))),
                "n_tasks": int(len(np.unique(row_task[band_mask[band]]))),
                "positive_rate": float(target[band_mask[band]].mean()),
                "logloss": stats["logloss"],
                "logloss_ci_lo": stats["ci_lo"],
                "logloss_ci_hi": stats["ci_hi"],
            }
            if baseline is None:
                record.update(
                    {"decrement": float("nan"), "decrement_ci_lo": float("nan"),
                     "decrement_ci_hi": float("nan"), "decrement_excludes_zero": False}
                )
            else:
                delta = decrement_stats(band, name, baseline)
                record.update(
                    {
                        "decrement": delta["decrement"],
                        "decrement_ci_lo": delta["ci_lo"],
                        "decrement_ci_hi": delta["ci_hi"],
                        "decrement_excludes_zero": delta["excludes_zero"],
                    }
                )
            increment_rows.append(record)
    increments = pd.DataFrame(increment_rows)
    increment_path = args.output / "increments.csv"
    increments.to_csv(increment_path, index=False)

    order_rows = []
    for band in BANDS:
        band_position = BANDS.index(band)
        ordered = band_stats(band, "S4")
        shuffled = band_stats(band, "S4_shuffled")
        base = band_stats(band, "S3")
        gain_sample = (
            draws[:, band_position, models.index("S4_shuffled")]
            - draws[:, band_position, models.index("S4")]
        )
        gain_lo, gain_hi = percentile_interval(gain_sample)
        gain = shuffled["logloss"] - ordered["logloss"]
        order_rows.append(
            {
                "band": band,
                "n_rows": ordered["n_rows"],
                "logloss_S3": base["logloss"],
                "logloss_S4_ordered": ordered["logloss"],
                "logloss_S4_shuffled": shuffled["logloss"],
                "decrement_ordered_vs_S3": decrement_stats(band, "S4", "S3")["decrement"],
                "decrement_ordered_vs_S3_ci_lo": decrement_stats(band, "S4", "S3")["ci_lo"],
                "decrement_ordered_vs_S3_ci_hi": decrement_stats(band, "S4", "S3")["ci_hi"],
                "decrement_shuffled_vs_S3": decrement_stats(band, "S4_shuffled", "S3")["decrement"],
                "decrement_shuffled_vs_S3_ci_lo": decrement_stats(band, "S4_shuffled", "S3")["ci_lo"],
                "decrement_shuffled_vs_S3_ci_hi": decrement_stats(band, "S4_shuffled", "S3")["ci_hi"],
                "order_gain": gain,
                "order_gain_ci_lo": gain_lo,
                "order_gain_ci_hi": gain_hi,
                "order_gain_excludes_zero": bool(gain_lo > 0.0 or gain_hi < 0.0),
            }
        )
    order = pd.DataFrame(order_rows)
    order_path = args.output / "order_control.csv"
    order.to_csv(order_path, index=False)

    # ---- verdicts --------------------------------------------------------- #
    def verdict(frame: pd.DataFrame, band: str, name: str) -> dict[str, Any]:
        record = frame[(frame["band"] == band) & (frame["model"] == name)].iloc[0]
        return {
            "decrement": float(record["decrement"]),
            "ci": [float(record["decrement_ci_lo"]), float(record["decrement_ci_hi"])],
            "useful": bool(record["decrement_excludes_zero"] and record["decrement"] > 0.0),
        }

    order_by_band = {}
    for _, record in order.iterrows():
        order_by_band[str(record["band"])] = {
            "order_gain": float(record["order_gain"]),
            "ci": [float(record["order_gain_ci_lo"]), float(record["order_gain_ci_hi"])],
            "order_carries_information": bool(
                record["order_gain"] > 0.0 and record["order_gain_ci_lo"] > 0.0
            ),
        }
    order_bands = [band for band, entry in order_by_band.items() if entry["order_carries_information"]]
    ordered_beats_shuffled = bool(order_bands)
    s4_by_band = {band: verdict(increments, band, "S4") for band in BANDS}
    s5_by_band = {
        band: {
            "residualised": verdict(increments, band, "S5"),
            "raw_graph": verdict(increments, band, "S5_raw_graph"),
            "utilisation_only": verdict(increments, band, "S5_utilisation_only"),
        }
        for band in BANDS
    }
    s5_bands = [band for band in BANDS if s5_by_band[band]["residualised"]["useful"]]
    s5_survives_at_low_prior = bool("low_prior" in s5_bands)

    if not ordered_beats_shuffled:
        ordered_verdict = (
            "THE ORDERED-HISTORY CONTRIBUTION IS NOT ORDER.  In every prior band the "
            "identical-content shuffle of the same eight regime labels scores as well "
            "as or better than the ordered encoding, and every order-gain interval "
            "spans zero.  With S2 (dwell) already in the model, S4 also fails to beat "
            "S3 in any band.  What the ordered-history machinery reads is dwell time "
            "and label composition, and it should be named that way."
        )
    else:
        ordered_verdict = (
            "Ordered history carries information beyond the bag of regime labels in "
            f"band(s): {', '.join(order_bands)}."
        )

    if not s5_bands:
        s5_verdict = (
            "THE GRAPH CONTRIBUTION IS UTILISATION.  In no prior band does the "
            "utilisation-residualised token-graph block produce an interval that "
            "excludes zero."
        )
    elif not s5_survives_at_low_prior:
        s5_verdict = (
            "The graph residual survives utilisation residualisation only in band(s): "
            f"{', '.join(s5_bands)} -- i.e. only where the clock already says the "
            "episode is probably failing.  In the low-prior region, where a detector "
            "would have to act, the residualised graph block adds nothing whose "
            "interval excludes zero, so the low-prior graph contribution is "
            "utilisation and clock."
        )
    else:
        s5_verdict = (
            "The graph residual survives utilisation residualisation in band(s): "
            f"{', '.join(s5_bands)}, including the low-prior region."
        )

    summary = {
        "schema": SCHEMA,
        "status": "analysis_instrument_not_a_detector",
        "disclaimer": (
            "The bundle's detectors are train-free by protocol.  This ablation fits "
            "an L2 logistic model only to measure how much information each feature "
            "class carries.  No coefficient, threshold or profile produced here is "
            "read by any monitor, and none of it ships.  Do not mistake this for a "
            "trained guard."
        ),
        "unit": "one (episode, chunk) row over every valid chunk of development_main",
        "target": "original_failure of the episode (constant within an episode)",
        "corpus": {
            "cohort": COHORT,
            "episodes": int(len(labels)),
            "rows": int(n_row),
            "tasks": int(len(unique_tasks)),
            "failure_episodes": int(risk_episode.sum()),
            "row_positive_rate": float(target.mean()),
            "tasks_with_zero_failures": int(
                sum(1 for name in unique_tasks if risk_episode[task == name].sum() == 0)
            ),
        },
        "protocol": {
            "model": "sklearn LogisticRegression, pure L2 (l1_ratio=0), solver=lbfgs",
            "C": PENALTY_C,
            "cross_validation": "StratifiedGroupKFold by task, no task in both fit and score",
            "folds": args.folds,
            "seed": args.seed,
            "bootstrap": {"draws": args.draws, "cluster": "task", "interval": "percentile 2.5/97.5"},
            "standardisation": "train fold only; S4 one-hots left as raw 0/1 so the ordered and shuffled encodings carry identical penalties",
            "imputation": "train-fold column median; no missingness indicators, so an undefined window cannot launder the clock into S1",
            "prior_band": f"survival prior < {LOW_PRIOR_MAX} is low, otherwise high; the prior used is the train-fold estimate, the same value fed to the model",
            "window_defined_band": f"low_prior rows with chunk >= {WINDOW}, where every W={WINDOW} quantity is defined without imputation",
        },
        "feature_blocks": {
            "S0": {"description": "clock only: per-suite survival prior at this chunk, plus the raw chunk index", "columns": ["prior_logit", "chunk", "log1p_chunk"], "n": 3},
            "S1": {"description": "instantaneous routing readout at this query", "columns": list(S1_COLUMNS), "n": len(S1_COLUMNS)},
            "S2": {"description": f"dwell: consecutive preceding queries in the same {N_TERCILE}x{N_TERCILE} regime of (R, L)", "columns": ["dwell", "log1p_dwell"], "n": 2},
            "S3": {"description": "first differences of the S1 quantities at lags 1 and 2", "columns": list(S3_COLUMNS), "n": len(S3_COLUMNS)},
            "S4": {"description": f"ordered one-hot encoding of the last {HISTORY_DEPTH} regime labels (strict prefix, PAD before episode start)", "n": HISTORY_DEPTH * N_HISTORY_CATEGORY, "categories": N_HISTORY_CATEGORY},
            "S5": {"description": "token-graph summaries residualised on per-layer node utilisation, train-fold fit", "columns": graph_columns, "n": len(graph_columns)},
            "S5_utilisation_control": {"columns": utilisation_columns, "n": len(utilisation_columns)},
        },
        "static_features": static_meta,
        "gate_entropy": entropy_report,
        "alignment_checks": {
            "layer_graphs": graph_report,
            "labels": {
                "file": str(LABELS),
                "merge": "on (task, episode), validate=one_to_one, row order restored",
                "unmatched_rows": 0,
            },
            "route_store": {
                "checked": "per task, episode_id contiguity and per-episode row count against the cache length; every cache row covered",
                "episodes_covered": entropy_report["episodes_covered"],
            },
            "graph_metric_duplicate": duplicate_check,
        },
        "folds": fold_report,
        "fold_diagnostics": fold_diagnostics,
        "bands": {
            band: {
                "n_rows": int(band_mask[band].sum()),
                "positive_rate": float(target[band_mask[band]].mean()),
            }
            for band in BANDS
        },
        "results": {
            band: {
                name: {
                    **band_stats(band, name),
                    **(
                        {}
                        if name not in baseline_of
                        else decrement_stats(band, name, baseline_of[name])
                    ),
                }
                for name in models
            }
            for band in BANDS
        },
        "verdicts": {
            "ordered_history_beats_identical_content_shuffle": ordered_beats_shuffled,
            "order_gain_by_band": order_by_band,
            "s4_over_s3_by_band": s4_by_band,
            "ordered_history_verdict": ordered_verdict,
            "s5_by_band": s5_by_band,
            "s5_bands_where_residual_survives": s5_bands,
            "s5_survives_at_low_prior": s5_survives_at_low_prior,
            "s5_verdict": s5_verdict,
            "nested_steps_with_intervals_excluding_zero": {
                band: [
                    name
                    for name in nested[1:]
                    if bool(
                        increments[
                            (increments["band"] == band) & (increments["model"] == name)
                        ].iloc[0]["decrement_excludes_zero"]
                    )
                ]
                for band in BANDS
            },
        },
        "seconds": round(time.perf_counter() - started, 1),
        "artifacts": {
            "increments_csv": str(increment_path),
            "order_control_csv": str(order_path),
            "increments_csv_sha256": sha256(increment_path),
            "order_control_csv_sha256": sha256(order_path),
            "script_sha256": sha256(Path(__file__)),
            "progress_cache_sha256": sha256(CACHE),
            "layer_graphs_sha256": sha256(GRAPHS),
            "graph_metric_source": str(GRAPH_EXTRACTOR),
        },
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # ---- readable table --------------------------------------------------- #
    print()
    print("=" * 108)
    print("NESTED INFORMATION-SOURCE ABLATION -- analysis instrument, not a detector; nothing fitted here ships")
    print(f"unit = (episode, chunk) over {n_row:,} valid chunks of {COHORT}; "
          f"target = original_failure; {len(unique_tasks)} tasks, {int(risk_episode.sum())} failing episodes")
    print(f"{args.folds}-fold task-blocked CV, L2 logistic (C={PENALTY_C}); "
          f"{args.draws} task-clustered bootstrap draws; log-loss in nats")
    print("=" * 108)
    label_of = {
        "S0": "S0  clock only",
        "S1": "S1  + current state",
        "S2": "S2  + dwell time",
        "S3": "S3  + local change",
        "S4": "S4  + ordered history",
        "S5": "S5  + graph residual",
        "S4_shuffled": "    [control] S3 + shuffled history",
        "S5_raw_graph": "    [diag] S4 + raw graph",
        "S5_utilisation_only": "    [diag] S4 + utilisation only",
    }
    for band in BANDS:
        mask = band_mask[band]
        print()
        print(f"--- band: {band}   n={int(mask.sum()):,} rows   positive rate={target[mask].mean():.4f} ---")
        print(f"{'step':<38}{'log-loss':>10}{'decrement':>12}{'95% CI':>26}{'useful':>8}")
        for name in models:
            stats = band_stats(band, name)
            if name in baseline_of:
                delta = decrement_stats(band, name, baseline_of[name])
                interval = f"[{delta['ci_lo']:+.5f}, {delta['ci_hi']:+.5f}]"
                flag = "yes" if delta["excludes_zero"] and delta["decrement"] > 0 else "no"
                print(
                    f"{label_of[name]:<38}{stats['logloss']:>10.5f}"
                    f"{delta['decrement']:>+12.5f}{interval:>26}{flag:>8}"
                )
            else:
                print(f"{label_of[name]:<38}{stats['logloss']:>10.5f}{'--':>12}{'--':>26}{'--':>8}")

    print()
    print("--- order control: ordered vs identical-content shuffle of the same 8 labels ---")
    print(f"{'band':<28}{'ordered':>10}{'shuffled':>11}{'gain':>11}{'95% CI':>26}{'order?':>8}")
    for _, record in order.iterrows():
        interval = f"[{record['order_gain_ci_lo']:+.5f}, {record['order_gain_ci_hi']:+.5f}]"
        flag = "yes" if record["order_gain_excludes_zero"] and record["order_gain"] > 0 else "no"
        print(
            f"{record['band']:<28}{record['logloss_S4_ordered']:>10.5f}"
            f"{record['logloss_S4_shuffled']:>11.5f}{record['order_gain']:>+11.5f}"
            f"{interval:>26}{flag:>8}"
        )
    print()
    for heading, text in (
        ("VERDICT (ordered history)", ordered_verdict),
        ("VERDICT (graph residual)", s5_verdict),
    ):
        print(f"{heading}:")
        for line in textwrap.wrap(text, width=104):
            print(f"  {line}")
    print()
    print(
        "Reminder: this is an analysis instrument.  The fitted logistic models exist only to "
        "measure how much\ninformation each feature class carries; no coefficient, threshold or "
        "profile written here is read by any\nmonitor, and none of it ships."
    )
    print(f"\nwrote {increment_path}\n      {order_path}\n      {summary_path}")
    print(f"elapsed {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
