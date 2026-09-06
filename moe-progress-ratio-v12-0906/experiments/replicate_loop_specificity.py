#!/usr/bin/env python3
"""Descriptive replication of the circling-cell loop specificity on ``external_8b``.

**This is a descriptive lift over failure-mode labels, not a detector evaluation.**
Nothing here selects a threshold, an operating point, or a rule; no gate is
opened or closed by the numbers below, and no holdout is consumed in the sense
of `docs/`'s sealing protocol.  Two rounds of detector selection have already
failed their pre-registered gates on development and remain failed
(`results/operating_point/selection.json`, `results/loop_operating_point/`);
this script does not revisit them.

What is being replicated is a *mechanism* claim measured on development by
`experiments/diagnose_phenotype_timing.py`: in the low-survival-prior region the
circling cell of the 2x2 phenotype (window path length ``L >= median(L)`` AND
progress ratio ``R < median(R)``) is strongly enriched in ``loop`` failures,
roughly flat on ``static``, and depleted on pure ``phantom grasp``.  If that
specificity is real it should reappear on a second cohort.

Method, mirroring ``diagnose_phenotype_timing.py``:

* the comparison is **chunk-matched within suite**.  Pooling across chunks
  inverts the sign, because a failure by definition runs to the horizon cap and
  therefore contributes disproportionately many late chunks, and late chunks
  have a different phenotype mix.  The pooled numbers are reported alongside
  (``prior_band == "pooled_unmatched"``) precisely so the inversion is visible.
* per ``(suite, chunk)`` cell with at least ``MIN_TARGET`` target episodes and
  at least ``MIN_TIMELY`` timely-success episodes still valid at that chunk,

      lift = P(circling | target, suite, chunk) / P(circling | timely, suite, chunk)

  aggregated across cells weighted by the target count.
* cells are split by the survival prior ``P(failure | still running at chunk)``
  of their suite into low (``< 0.25``) and high (``>= 0.25``).

Uncertainty is a **task-clustered bootstrap** over ``BOOTSTRAP_DRAWS`` draws,
resampling tasks with replacement, following ``clustered_interval`` in
``moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py``: episodes inside a
task share an init-state distribution and are not independent units.  Cell
counts, the survival prior and the band split are all recomputed inside each
draw.  The two cohort-level nuisance constants -- the medians of ``L`` and ``R``
that define the 2x2 -- are held at their full-cohort values; they are estimated
from ~700k (episode, chunk) points each and their resampling noise is
negligible next to the cell counts.

Run:  python experiments/replicate_loop_specificity.py
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

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

CACHE_ROOT = BUNDLE / "results/progress_cache"
LABEL_ROOT = WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
BELIEF_EPISODES = TAXONOMY / "belief_mismatch_episodes.csv.gz"
EVENTS = TAXONOMY / "events.csv.gz"
MOBILITY_ROOT = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
DEFAULT_OUTPUT = BUNDLE / "results/loop_specificity"

SCHEMA = "himoe.progress_ratio_v12.loop_specificity.v1"
JOIN_KEY = ["task", "init_state_id", "flow_noise_seed"]

# (progress cache stem, v4 mobility stem, taxonomy run id, clean-label stem)
COHORTS = (
    ("development_main", "main_reference", "seed1000_1007", "development_main"),
    ("external_8b", "external_8b", "seed1008_1015", "external_8b"),
)

POINTS = ((3, "all"), (4, "all"), (6, "all"), (4, "front"), (6, "back"))
TARGETS = ("loop", "static", "pure_phantom")
BANDS = ("low", "high", "all_matched", "pooled_unmatched")

EPS_LENGTH = 1e-3  # matches diagnose_phenotype_timing.py
PRIOR_CUT = 0.25
MIN_TARGET = 3
MIN_TIMELY = 5
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260906

# Pre-registered counts from the taxonomy report; a mismatch aborts the run.
EXPECTED_COUNTS = {
    "development_main": {
        "fail": 487,
        "loop": 256,
        "static": 135,
        "phantom": 93,
        "pure_phantom": 22,
    },
    "external_8b": {
        "fail": 564,
        "loop": 314,
        "static": 164,
        "phantom": 115,
        "pure_phantom": 32,
    },
}


# --------------------------------------------------------------------------- #
# Pure aggregation helpers -- pinned by tests/test_loop_specificity.py
# --------------------------------------------------------------------------- #


def weighted_lift(lifts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weight-averaged lift over the trailing cell axes, dropping bad cells.

    ``lifts`` and ``weights`` share a shape ``[..., S, Q]``.  Cells whose lift is
    not finite, or whose weight is not positive, contribute to neither the
    numerator nor the denominator.  A batch entry with no usable cell yields NaN
    rather than 0, so "no evidence" never masquerades as "no effect".
    """
    lifts = np.asarray(lifts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if lifts.shape != weights.shape:
        raise ValueError("lifts and weights do not align")
    good = np.isfinite(lifts) & np.isfinite(weights) & (weights > 0)
    numerator = np.where(good, lifts * weights, 0.0).sum(axis=(-2, -1))
    denominator = np.where(good, weights, 0.0).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def cell_lift(
    target_usable: np.ndarray,
    target_circling: np.ndarray,
    timely_usable: np.ndarray,
    timely_circling: np.ndarray,
) -> np.ndarray:
    """Per-cell occupancy ratio ``P(circling | target) / P(circling | timely)``.

    Cells where either denominator vanishes -- no episodes, or no timely episode
    in the circling cell -- return NaN and are dropped downstream, exactly as the
    ``timely_share > 0`` guard in ``diagnose_phenotype_timing.matched_lift``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        target_share = np.where(
            target_usable > 0, target_circling / np.maximum(target_usable, 1), np.nan
        )
        timely_share = np.where(
            timely_usable > 0, timely_circling / np.maximum(timely_usable, 1), np.nan
        )
        return np.where(timely_share > 0, target_share / timely_share, np.nan)


def band_mask(
    alive: np.ndarray,
    fail_alive: np.ndarray,
    band: str,
    prior_cut: float = PRIOR_CUT,
) -> np.ndarray:
    """Which ``(suite, chunk)`` cells fall in a survival-prior band.

    The prior is ``P(failure | still running at chunk q)`` inside the suite, the
    same quantity ``hazard_common.survival_prior`` computes and the same one the
    detector rounds were scored against.  It is the *failure* prior for every
    target, not a per-target prior, so the bands mean the same thing across the
    three targets and the lifts stay comparable.
    """
    alive = np.asarray(alive, dtype=np.float64)
    fail_alive = np.asarray(fail_alive, dtype=np.float64)
    exists = alive > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        prior = np.where(exists, fail_alive / np.maximum(alive, 1), np.nan)
    if band == "low":
        return exists & (prior < prior_cut)
    if band in ("high",):
        return exists & (prior >= prior_cut)
    if band in ("all_matched", "pooled_unmatched"):
        return exists
    raise ValueError(f"unknown prior band: {band}")


def mantel_haenszel(
    target_usable: np.ndarray,
    target_circling: np.ndarray,
    timely_usable: np.ndarray,
    timely_circling: np.ndarray,
    keep: np.ndarray,
) -> np.ndarray:
    """Mantel-Haenszel common risk ratio over the retained strata.

    The pre-registered aggregate is a target-count-weighted **mean of per-cell
    ratios**, and that estimator is dominated by cells whose *timely* share is
    near zero: a cell with 9 circling timely episodes out of 1546 turns into a
    40x per-cell lift and drags the mean up regardless of how many target
    episodes the well-populated cells contribute.  The MH common ratio is the
    textbook stratified alternative, weighting each stratum by
    ``n_target * n_timely / n_total`` instead, so it does not blow up on a thin
    denominator.  It is reported as a robustness column, never as the headline.
    """
    total = target_usable + timely_usable
    good = keep & (total > 0)
    numerator = np.where(good, target_circling * timely_usable / np.maximum(total, 1), 0.0)
    denominator = np.where(good, timely_circling * target_usable / np.maximum(total, 1), 0.0)
    return _ratio(numerator.sum(axis=(-2, -1)), denominator.sum(axis=(-2, -1)))


def band_stats(
    counts: dict[str, np.ndarray],
    band: str,
    min_target: int = MIN_TARGET,
    min_timely: int = MIN_TIMELY,
    prior_cut: float = PRIOR_CUT,
) -> dict[str, np.ndarray]:
    """Aggregate lift for one prior band.

    ``counts`` holds ``[..., S, Q]`` count arrays with keys ``target_usable``,
    ``target_circling``, ``timely_usable``, ``timely_circling``, ``alive`` and
    ``fail_alive``.  Returns ``lift`` (the pre-registered weighted mean of
    per-cell ratios), ``mh_lift`` (the stratified robustness estimator),
    ``cells`` and ``weight``, each shaped ``[...]``.

    ``pooled_unmatched`` is the deliberately confounded comparison: it collapses
    every cell into a single ratio of sums, ignoring the per-cell gates.  It is
    reported only to show the sign inversion that motivates matching.
    """
    include = band_mask(counts["alive"], counts["fail_alive"], band, prior_cut)
    target_usable = np.asarray(counts["target_usable"], dtype=np.float64)
    target_circling = np.asarray(counts["target_circling"], dtype=np.float64)
    timely_usable = np.asarray(counts["timely_usable"], dtype=np.float64)
    timely_circling = np.asarray(counts["timely_circling"], dtype=np.float64)

    if band == "pooled_unmatched":
        take = include.astype(np.float64)
        target_share = _ratio(
            (target_circling * take).sum(axis=(-2, -1)),
            (target_usable * take).sum(axis=(-2, -1)),
        )
        timely_share = _ratio(
            (timely_circling * take).sum(axis=(-2, -1)),
            (timely_usable * take).sum(axis=(-2, -1)),
        )
        lift = _ratio(target_share, timely_share)
        return {
            "lift": lift,
            "mh_lift": lift,
            "cells": (include & (target_usable > 0)).sum(axis=(-2, -1)).astype(np.float64),
            "weight": (target_usable * take).sum(axis=(-2, -1)),
        }

    keep = include & (target_usable >= min_target) & (timely_usable >= min_timely)
    lifts = np.where(
        keep,
        cell_lift(target_usable, target_circling, timely_usable, timely_circling),
        np.nan,
    )
    weights = np.where(keep, target_usable, 0.0)
    usable_cell = keep & np.isfinite(lifts)
    return {
        "lift": weighted_lift(lifts, weights),
        "mh_lift": mantel_haenszel(
            target_usable, target_circling, timely_usable, timely_circling, usable_cell
        ),
        "cells": usable_cell.sum(axis=(-2, -1)).astype(np.float64),
        "weight": np.where(usable_cell, weights, 0.0).sum(axis=(-2, -1)),
    }


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-300), np.nan)


def task_multiplicities(
    n_tasks: int, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``[draws, n_tasks]`` resample counts: draw ``n_tasks`` tasks with replacement."""
    if n_tasks < 1 or draws < 1:
        raise ValueError("n_tasks and draws must be positive")
    picks = rng.integers(0, n_tasks, size=(draws, n_tasks))
    counts = np.zeros((draws, n_tasks), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(draws), n_tasks), picks.ravel()), 1)
    return counts


def suite_sums(
    per_task: np.ndarray,
    suite_index: np.ndarray,
    n_suites: int,
    multiplicities: np.ndarray,
) -> np.ndarray:
    """``[T, Q]`` per-task counts -> ``[draws, S, Q]`` resampled suite counts."""
    per_task = np.asarray(per_task, dtype=np.float64)
    suite_index = np.asarray(suite_index, dtype=int)
    multiplicities = np.asarray(multiplicities, dtype=np.float64)
    if per_task.ndim != 2 or len(suite_index) != len(per_task):
        raise ValueError("per-task counts and suite index do not align")
    if multiplicities.ndim != 2 or multiplicities.shape[1] != len(per_task):
        raise ValueError("multiplicities do not align with the task axis")
    indicator = np.zeros((n_suites, len(per_task)), dtype=np.float64)
    indicator[suite_index, np.arange(len(per_task))] = 1.0
    # [S, T, Q] -> [draws, S, Q]
    return np.tensordot(multiplicities, indicator[:, :, None] * per_task[None, :, :], axes=([1], [1]))


def bootstrap_band_stats(
    per_task: dict[str, np.ndarray],
    suite_index: np.ndarray,
    n_suites: int,
    multiplicities: np.ndarray,
    band: str,
    suites: slice = slice(None),
    min_target: int = MIN_TARGET,
    min_timely: int = MIN_TIMELY,
    prior_cut: float = PRIOR_CUT,
) -> dict[str, np.ndarray]:
    """Task-clustered bootstrap distribution of one band's aggregate lift.

    Every draw recomputes the suite/chunk counts, the survival prior, the band
    membership, the per-cell gates and the weighted aggregate.  Passing an
    all-ones single-row ``multiplicities`` reproduces the point estimate through
    exactly the same code path.  ``suites`` restricts the cells to one suite for
    the per-suite breakdown while still resampling the whole task pool.
    """
    resampled = {
        name: suite_sums(values, suite_index, n_suites, multiplicities)[:, suites, :]
        for name, values in per_task.items()
    }
    return band_stats(resampled, band, min_target, min_timely, prior_cut)


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% percentile interval over the finite draws."""
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high)


# --------------------------------------------------------------------------- #
# Cohort loading and labelling
# --------------------------------------------------------------------------- #


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    return cache["task_names"].astype(str)[cache["task_index"].astype(int)]


def load_cohort(
    cache_stem: str, mobility_stem: str, run_id: str, label_stem: str
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Progress cache plus a row-aligned failure-mode label frame."""
    cache = load_npz(CACHE_ROOT / f"{cache_stem}.npz")
    mobility = load_npz(MOBILITY_ROOT / f"{mobility_stem}.npz")

    if len(mobility["episode"]) != len(cache["episode"]):
        raise ValueError(f"{cache_stem}: the v4 mobility cache has a different row count")
    if not np.array_equal(task_of(mobility), task_of(cache)):
        raise ValueError(f"{cache_stem}: the v4 mobility cache is not row-aligned (task)")
    if not np.array_equal(mobility["episode"], cache["episode"]):
        raise ValueError(f"{cache_stem}: the v4 mobility cache is not row-aligned (episode)")
    if not np.array_equal(mobility["init_state_id"], cache["init_state_id"]):
        raise ValueError(f"{cache_stem}: init_state_id disagrees between the two caches")

    task = task_of(cache)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "suite": pd.Series(task).str.split("/", n=1).str[0],
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
            "init_state_id": cache["init_state_id"].astype(int),
            "flow_noise_seed": mobility["flow_noise_seed"].astype(int),
        }
    )
    if index.duplicated(JOIN_KEY).any():
        raise ValueError(f"{cache_stem}: the cohort key is not unique")

    episodes = pd.read_csv(BELIEF_EPISODES)
    events = pd.read_csv(EVENTS)
    events["task"] = events["task_key"]  # events.csv stores suite and task apart
    taxonomy = episodes.merge(
        events[["run", "task", "episode", "loop_onset", "static_onset"]],
        on=["run", "task", "episode"],
        validate="one_to_one",
    )
    taxonomy = taxonomy[taxonomy["run"] == run_id]
    columns = JOIN_KEY + ["success", "belief_mismatch", "loop_onset", "static_onset"]
    merged = index.merge(
        taxonomy[columns], on=JOIN_KEY, how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["success"].isna().any():
        raise ValueError(f"{cache_stem}: the trap taxonomy does not cover the cohort")
    merged = merged.reset_index(drop=True)

    fail = ~merged["success"].to_numpy(bool)
    merged["fail"] = fail
    merged["loop"] = (merged["loop_onset"].to_numpy(int) >= 0) & fail
    merged["static"] = (merged["static_onset"].to_numpy(int) >= 0) & fail
    merged["phantom"] = merged["belief_mismatch"].to_numpy(bool) & fail
    merged["pure_phantom"] = merged["phantom"] & ~merged["loop"] & ~merged["static"]

    # Cross-check against the timeout-extension label file the detector rounds
    # used, so "failure" means the same thing here as it did there.
    clean = pd.read_csv(LABEL_ROOT / f"{label_stem}_clean_labels.csv")[
        ["task", "episode", "original_failure"]
    ]
    checked = merged[["task", "episode"]].merge(
        clean, on=["task", "episode"], how="left", validate="one_to_one"
    )
    if checked["original_failure"].isna().any():
        raise ValueError(f"{cache_stem}: clean labels do not cover the cohort")
    if not np.array_equal(checked["original_failure"].to_numpy(bool), fail):
        raise ValueError(f"{cache_stem}: taxonomy failure and original_failure disagree")

    observed = {
        "fail": int(fail.sum()),
        "loop": int(merged["loop"].sum()),
        "static": int(merged["static"].sum()),
        "phantom": int(merged["phantom"].sum()),
        "pure_phantom": int(merged["pure_phantom"].sum()),
    }
    if observed != EXPECTED_COUNTS[cache_stem]:
        raise ValueError(
            f"{cache_stem}: label counts {observed} != expected "
            f"{EXPECTED_COUNTS[cache_stem]}"
        )
    return cache, merged


def phenotype_masks(
    cache: dict[str, np.ndarray], window: int, group: str
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """``(usable, circling, medians)``, verbatim the diagnose script's 2x2."""
    adjacent = cache["lag_distance"][:, :, 0, :]
    length = pr.path_length(adjacent, window)
    displacement = cache["lag_distance"][:, :, pr.LAGS.index(window), :]
    ratio = pr.group_ratio(pr.progress_ratio(displacement, length, EPS_LENGTH), group)
    grouped_length = pr.group_ratio(length, group)
    usable = cache["valid"] & np.isfinite(ratio) & np.isfinite(grouped_length)
    median_length = float(np.median(grouped_length[usable]))
    median_ratio = float(np.median(ratio[usable]))
    circling = (grouped_length >= median_length) & (ratio < median_ratio)
    return usable, circling & usable, {"length": median_length, "ratio": median_ratio}


def per_task_counts(
    mask: np.ndarray, task_index: np.ndarray, n_tasks: int
) -> np.ndarray:
    """``[E, Q]`` episode/chunk booleans -> ``[T, Q]`` counts."""
    output = np.zeros((n_tasks, mask.shape[1]), dtype=np.float64)
    for task in range(n_tasks):
        rows = task_index == task
        if rows.any():
            output[task] = mask[rows].sum(axis=0)
    return output


def alive_counts(
    length: np.ndarray, task_index: np.ndarray, n_tasks: int, n_chunks: int
) -> np.ndarray:
    """``[T, Q]`` count of episodes with ``length > chunk`` -- the prior's base."""
    grid = length[:, None] > np.arange(n_chunks)[None, :]
    return per_task_counts(grid, task_index, n_tasks)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def evaluate_cohort(
    cohort: str,
    cache: dict[str, np.ndarray],
    labels: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    task_names = np.unique(labels["task"].to_numpy(str))
    task_lookup = {name: position for position, name in enumerate(task_names)}
    task_index = np.asarray([task_lookup[name] for name in labels["task"]], dtype=int)
    suite_names = np.unique(labels["suite"].to_numpy(str))
    suite_lookup = {name: position for position, name in enumerate(suite_names)}
    suite_of_task = np.asarray(
        [
            suite_lookup[labels["suite"].to_numpy(str)[task_index == position][0]]
            for position in range(len(task_names))
        ],
        dtype=int,
    )

    n_chunks = cache["valid"].shape[1]
    fail = labels["fail"].to_numpy(bool)
    timely = ~fail
    length = labels["length"].to_numpy(int)

    alive = alive_counts(length, task_index, len(task_names), n_chunks)
    fail_alive = alive_counts(
        np.where(fail, length, -1), task_index, len(task_names), n_chunks
    )

    multiplicities = task_multiplicities(len(task_names), draws, rng)
    identity = np.ones((1, len(task_names)), dtype=np.float64)

    rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    medians: dict[str, dict[str, float]] = {}
    for window, group in POINTS:
        usable, circling, median = phenotype_masks(cache, window, group)
        medians[f"W{window}_{group}"] = median
        base = {
            "alive": alive,
            "fail_alive": fail_alive,
            "timely_usable": per_task_counts(
                usable & timely[:, None], task_index, len(task_names)
            ),
            "timely_circling": per_task_counts(
                circling & timely[:, None], task_index, len(task_names)
            ),
        }
        for target in TARGETS:
            flag = labels[target].to_numpy(bool)
            per_task = dict(base)
            per_task["target_usable"] = per_task_counts(
                usable & flag[:, None], task_index, len(task_names)
            )
            per_task["target_circling"] = per_task_counts(
                circling & flag[:, None], task_index, len(task_names)
            )

            point_counts = {
                name: suite_sums(values, suite_of_task, len(suite_names), identity)
                for name, values in per_task.items()
            }
            for band in BANDS:
                point = band_stats(point_counts, band)
                samples = bootstrap_band_stats(
                    per_task, suite_of_task, len(suite_names), multiplicities, band
                )
                low, high = percentile_interval(samples["lift"])
                mh_low, mh_high = percentile_interval(samples["mh_lift"])
                include = band_mask(point_counts["alive"], point_counts["fail_alive"], band)[0]
                if band == "pooled_unmatched":
                    keep = include
                else:
                    keep = (
                        include
                        & (point_counts["target_usable"][0] >= MIN_TARGET)
                        & (point_counts["timely_usable"][0] >= MIN_TIMELY)
                    )
                episodes = episode_coverage(
                    usable & flag[:, None], keep, labels, suite_lookup
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "window": window,
                        "layer_group": group,
                        "target": target,
                        "prior_band": band,
                        "lift": float(point["lift"][0]),
                        "ci_low": low,
                        "ci_high": high,
                        "ci_excludes_one": bool(
                            np.isfinite(low) and np.isfinite(high) and (low > 1.0 or high < 1.0)
                        ),
                        "mh_lift": float(point["mh_lift"][0]),
                        "mh_ci_low": mh_low,
                        "mh_ci_high": mh_high,
                        "mh_ci_excludes_one": bool(
                            np.isfinite(mh_low)
                            and np.isfinite(mh_high)
                            and (mh_low > 1.0 or mh_high < 1.0)
                        ),
                        "n_cells": int(point["cells"][0]),
                        "target_episode_weight": float(point["weight"][0]),
                        "n_target_episodes": int(episodes),
                        "n_target_cohort": int(flag.sum()),
                        "finite_draws": int(np.isfinite(samples["lift"]).sum()),
                        "draws": int(draws),
                    }
                )
                if band not in ("low", "high"):
                    continue
                for position, suite in enumerate(suite_names):
                    window_slice = slice(position, position + 1)
                    suite_point = band_stats(
                        {k: v[:, window_slice, :] for k, v in point_counts.items()}, band
                    )
                    suite_draws = bootstrap_band_stats(
                        per_task,
                        suite_of_task,
                        len(suite_names),
                        multiplicities,
                        band,
                        suites=window_slice,
                    )
                    s_low, s_high = percentile_interval(suite_draws["lift"])
                    suite_rows.append(
                        {
                            "cohort": cohort,
                            "window": window,
                            "layer_group": group,
                            "target": target,
                            "prior_band": band,
                            "suite": str(suite),
                            "lift": float(suite_point["lift"][0]),
                            "ci_low": s_low,
                            "ci_high": s_high,
                            "mh_lift": float(suite_point["mh_lift"][0]),
                            "n_cells": int(suite_point["cells"][0]),
                            "target_episode_weight": float(suite_point["weight"][0]),
                        }
                    )
    meta = {
        "n_tasks": int(len(task_names)),
        "n_suites": int(len(suite_names)),
        "n_episodes": int(len(labels)),
        "suites": [str(name) for name in suite_names],
        "medians": medians,
        "label_counts": EXPECTED_COUNTS[cohort],
    }
    return rows, suite_rows, meta


def episode_coverage(
    target_usable_mask: np.ndarray,
    keep: np.ndarray,
    labels: pd.DataFrame,
    suite_lookup: dict[str, int],
) -> int:
    """Distinct target episodes that contribute to at least one retained cell."""
    suite_row = np.asarray(
        [suite_lookup[name] for name in labels["suite"].to_numpy(str)], dtype=int
    )
    retained = keep[suite_row]  # [E, Q]
    return int((target_usable_mask & retained).any(axis=1).sum())


def comparison_table(frame: pd.DataFrame, band: str) -> pd.DataFrame:
    block = frame[frame["prior_band"] == band]
    wide = block.pivot_table(
        index=["window", "layer_group", "target"],
        columns="cohort",
        values=["lift", "ci_low", "ci_high", "mh_lift", "n_cells"],
        aggfunc="first",
    )
    rows = []
    for (window, group, target), record in wide.iterrows():
        row = {"W": window, "group": group, "target": target}
        for cohort, short in (("development_main", "dev"), ("external_8b", "ext")):
            lift = record[("lift", cohort)]
            low = record[("ci_low", cohort)]
            high = record[("ci_high", cohort)]
            row[f"{short}_lift"] = (
                f"{lift:.2f} [{low:.2f}, {high:.2f}]" if np.isfinite(lift) else "n/a"
            )
            row[f"{short}_mh"] = f"{record[('mh_lift', cohort)]:.2f}"
            row[f"{short}_cells"] = int(record[("n_cells", cohort)])
        rows.append(row)
    order = {target: position for position, target in enumerate(TARGETS)}
    point_order = {point: position for position, point in enumerate(POINTS)}
    table = pd.DataFrame(rows)
    table["_p"] = [point_order[(w, g)] for w, g in zip(table["W"], table["group"])]
    table["_t"] = table["target"].map(order)
    return table.sort_values(["_p", "_t"]).drop(columns=["_p", "_t"])


def verdict(frame: pd.DataFrame, suite_frame: pd.DataFrame) -> dict[str, Any]:
    low = frame[frame["prior_band"] == "low"]

    def pick(cohort: str, target: str, window: int, group: str) -> pd.Series:
        hit = low[
            (low["cohort"] == cohort)
            & (low["target"] == target)
            & (low["window"] == window)
            & (low["layer_group"] == group)
        ]
        return hit.iloc[0]

    per_point = []
    for window, group in POINTS:
        record: dict[str, Any] = {"window": window, "layer_group": group}
        for cohort, short in (("development_main", "dev"), ("external_8b", "ext")):
            values = {target: pick(cohort, target, window, group) for target in TARGETS}
            record[f"{short}_loop_lift"] = values["loop"]["lift"]
            record[f"{short}_loop_ci"] = [values["loop"]["ci_low"], values["loop"]["ci_high"]]
            record[f"{short}_loop_ci_excludes_one"] = bool(values["loop"]["ci_excludes_one"])
            record[f"{short}_loop_ci_above_one"] = bool(
                np.isfinite(values["loop"]["ci_low"]) and values["loop"]["ci_low"] > 1.0
            )
            record[f"{short}_loop_mh_lift"] = values["loop"]["mh_lift"]
            record[f"{short}_loop_mh_ci"] = [
                values["loop"]["mh_ci_low"],
                values["loop"]["mh_ci_high"],
            ]
            record[f"{short}_static_lift"] = values["static"]["lift"]
            record[f"{short}_pure_phantom_lift"] = values["pure_phantom"]["lift"]
            record[f"{short}_loop_beats_static"] = bool(
                values["loop"]["lift"] > values["static"]["lift"]
            )
            record[f"{short}_loop_beats_pure_phantom"] = bool(
                np.isfinite(values["pure_phantom"]["lift"])
                and values["loop"]["lift"] > values["pure_phantom"]["lift"]
            )
        per_point.append(record)

    external_ordering = all(
        row["ext_loop_beats_static"] and row["ext_loop_beats_pure_phantom"]
        for row in per_point
    )
    external_loop_above_one = all(row["ext_loop_ci_above_one"] for row in per_point)

    # Is the enrichment a property of the cohort, or of one suite inside it?
    suite_low = suite_frame[
        (suite_frame["prior_band"] == "low") & (suite_frame["target"] == "loop")
    ]
    per_suite: dict[str, Any] = {}
    for cohort in ("development_main", "external_8b"):
        block = suite_low[suite_low["cohort"] == cohort]
        per_suite[cohort] = {
            str(suite): {
                "median_lift_over_points": float(
                    np.nanmedian(part["lift"].to_numpy(float))
                ),
                "points_above_one": int((part["lift"].to_numpy(float) > 1.0).sum()),
                "points": int(len(part)),
            }
            for suite, part in block.groupby("suite")
        }
    mh_summary = {}
    for cohort, short in (("development_main", "dev"), ("external_8b", "ext")):
        values = np.asarray([row[f"{short}_loop_mh_lift"] for row in per_point])
        mh_summary[cohort] = {
            "low_prior_loop_mh_min": float(np.nanmin(values)),
            "low_prior_loop_mh_max": float(np.nanmax(values)),
            "points_with_mh_above_one": int((values > 1.0).sum()),
            "points": int(len(values)),
        }
    return {
        "per_point": per_point,
        "loop_low_prior_lift_by_suite": per_suite,
        "loop_low_prior_mantel_haenszel": mh_summary,
        "estimator_disagreement": (
            "The pre-registered weighted mean of per-cell ratios and the "
            "stratified Mantel-Haenszel common ratio disagree by roughly an "
            "order of magnitude in the low-prior band. The pattern that "
            "replicates is the ordering loop > static > pure_phantom and the "
            "per-suite structure; the magnitude of the loop enrichment does "
            "not survive a change of aggregator."
        ),
        "external_loop_ci_above_one_everywhere": bool(external_loop_above_one),
        "external_loop_beats_static_and_phantom_everywhere": bool(external_ordering),
        "n_points_external_loop_ci_above_one": int(
            sum(row["ext_loop_ci_above_one"] for row in per_point)
        ),
        "n_points_external_loop_beats_static": int(
            sum(row["ext_loop_beats_static"] for row in per_point)
        ),
        "n_points_external_loop_beats_pure_phantom": int(
            sum(row["ext_loop_beats_pure_phantom"] for row in per_point)
        ),
        "replicates": bool(external_loop_above_one and external_ordering),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, Any]] = []
    suite_rows: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for cache_stem, mobility_stem, run_id, label_stem in COHORTS:
        cache, labels = load_cohort(cache_stem, mobility_stem, run_id, label_stem)
        print(
            f"[{cache_stem}] {len(labels):,} episodes  "
            f"fail={int(labels['fail'].sum())}  loop={int(labels['loop'].sum())}  "
            f"static={int(labels['static'].sum())}  phantom={int(labels['phantom'].sum())}  "
            f"pure_phantom={int(labels['pure_phantom'].sum())}"
        )
        cohort_rows, cohort_suite_rows, cohort_meta = evaluate_cohort(
            cache_stem, cache, labels, args.draws, rng
        )
        rows.extend(cohort_rows)
        suite_rows.extend(cohort_suite_rows)
        meta[cache_stem] = cohort_meta

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "lifts.csv", index=False)
    suite_frame = pd.DataFrame(suite_rows)
    suite_frame.to_csv(args.output / "lifts_by_suite.csv", index=False)

    checks = verdict(frame, suite_frame)
    summary = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_type": "descriptive lift over failure-mode labels, NOT a detector evaluation",
        "notes": [
            "No threshold, operating point or rule is selected here; the two "
            "detector rounds failed their pre-registered gates on development "
            "(17/20 and 19/20 low-prior TP) and remain failed.",
            "external_8b is NOT a pristine holdout: it has been examined "
            "repeatedly across v3-v7 and the HB front/back bundle, so the "
            "external numbers below are a consistency check on an already-seen "
            "cohort, not an out-of-sample confirmation.",
            "The comparison is chunk-matched within suite. The pooled_unmatched "
            "rows are the confounded version and are reported only to show the "
            "sign inversion; failures run to the horizon cap and so contribute "
            "disproportionately many late chunks.",
            "The L and R medians defining the 2x2 are estimated inside each "
            "cohort, so this replicates the pattern, not a transferred threshold.",
            "The survival prior banding uses P(failure | alive at chunk) per "
            "suite for all three targets, so the bands are comparable across "
            "targets.",
            "The pre-registered aggregate is a target-count-weighted mean of "
            "per-cell ratios, which is dominated by cells whose timely share is "
            "near zero (a libero_goal cell with 9/1546 circling timely episodes "
            "yields a 40x per-cell lift). The mh_lift columns and "
            "lifts_by_suite.csv exist so that fragility is visible; read them "
            "before quoting the headline number.",
            "The 2x2 medians are pooled over the whole cohort, not computed per "
            "suite, so the cell definition is confounded with suite even though "
            "the chunk matching is not. That is why the per-suite loop lift "
            "ranges from ~0.5x (libero_long) to ~20x (libero_goal): the circling "
            "cell means something different in each suite.",
        ],
        "method": {
            "points": [{"window": w, "layer_group": g} for w, g in POINTS],
            "targets": list(TARGETS),
            "prior_bands": list(BANDS),
            "eps_length": EPS_LENGTH,
            "prior_cut": PRIOR_CUT,
            "min_target_episodes": MIN_TARGET,
            "min_timely_episodes": MIN_TIMELY,
            "bootstrap": {
                "kind": "task-clustered percentile bootstrap",
                "draws": int(args.draws),
                "seed": BOOTSTRAP_SEED,
                "unit": "task (episodes inside a task share an init-state distribution)",
                "resampled_inside_draw": [
                    "cell counts",
                    "survival prior",
                    "prior band membership",
                    "per-cell gates",
                ],
                "held_fixed": ["cohort L median", "cohort R median"],
                "reference": "moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py::clustered_interval",
            },
        },
        "cohorts": meta,
        "verdict": checks,
        "runtime_seconds": round(time.time() - started, 1),
        "rows": rows,
        "suite_rows": suite_rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 220)
    for band in ("low", "high", "pooled_unmatched"):
        print(f"\n=== circling-cell lift, prior band: {band} "
              f"(descriptive, not a detector) ===")
        print(comparison_table(frame, band).to_string(index=False))

    print("\n=== replication checks (external_8b, low-prior band) ===")
    for row in checks["per_point"]:
        print(
            f"W{row['window']:<2} {row['layer_group']:<5} "
            f"loop {row['ext_loop_lift']:.2f} "
            f"[{row['ext_loop_ci'][0]:.2f}, {row['ext_loop_ci'][1]:.2f}]  "
            f"MH {row['ext_loop_mh_lift']:.2f} "
            f"[{row['ext_loop_mh_ci'][0]:.2f}, {row['ext_loop_mh_ci'][1]:.2f}]  "
            f"static {row['ext_static_lift']:.2f}  "
            f"pure_phantom {row['ext_pure_phantom_lift']:.2f}  "
            f"CI>1 {row['ext_loop_ci_above_one']}  "
            f"loop>static {row['ext_loop_beats_static']}  "
            f"loop>phantom {row['ext_loop_beats_pure_phantom']}"
        )

    print("\n=== loop low-prior lift by suite (is it one suite or all four?) ===")
    block = suite_frame[
        (suite_frame["prior_band"] == "low") & (suite_frame["target"] == "loop")
    ]
    table = block.pivot_table(
        index=["window", "layer_group", "suite"], columns="cohort", values="lift"
    )
    print(table.to_string(float_format=lambda value: f"{value:.2f}"))

    print(f"\nreplicates (all five points): {checks['replicates']}")
    print(f"runtime {summary['runtime_seconds']}s -> {args.output}")


if __name__ == "__main__":
    main()
