#!/usr/bin/env python3
"""Shared machinery for the windowed cross-frame quorum experiments.

Two rule families, both predeclared in results/PREREGISTRATION.json:

    quorum   at least k pool members vote inside a trailing window of W chunks,
             spanning at least f distinct reference frames. Every member is
             thresholded at the same shared sensitivity level s.

    cascade  the same, plus at least one vote from the strict level s_hi. The
             remaining k-1 votes only need the looser level s_lo. This is the
             "sensitive proposes, strict confirms" shape written so that the
             alarm still lands at most W chunks after the first vote.

k=1 is plain OR over the pool. A pairwise AND is k=2, W=inf on a two-member
pool; the family therefore contains the current operating points as special
cases and the comparison is like for like.

Everything is causal: the count at chunk q uses only votes at chunks <= q.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
SURVEY = PROJECT / "moe-hb-front-back-0905"

sys.path.insert(0, str(SURVEY / "experiments"))
sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import LABEL_PATHS, PROFILE_ROOT, load_npz  # noqa: E402
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)


ALARM_ROOT = BUNDLE / "results/alarms"
MODES = ("per_task", "global")
LEVELS = ("selected",) + tuple(f"{q:g}" for q in dev.QUANTILES)
NUMERIC_LEVELS = tuple(f"{q:g}" for q in dev.QUANTILES)

FRAMES = {
    "mobility": "adjacent_query",
    "conditional_query_d1": "adjacent_query",
    "partial_query_d1": "adjacent_query",
    "flow_path": "denoising_steps",
    "flow_endpoint": "denoising_steps",
    "flow_settling_log_ratio": "denoising_steps",
    "action_consensus": "token_graph",
    "state_action_alignment": "token_graph",
    "conditional_energy": "token_graph",
    "conditional_effective_rank": "token_graph",
    "partial_edge_std": "token_graph",
    "expert_load_effective_rank": "expert_load",
}
POOLS = {
    "all12": tuple(sorted(FRAMES)),
    # state_action_alignment and conditional_energy produce near-identical
    # alarm vectors at the selected level; keeping both double counts one view.
    "dedup11": tuple(q for q in sorted(FRAMES) if q != "state_action_alignment"),
}
FRAME_ORDER = ("adjacent_query", "denoising_steps", "token_graph", "expert_load")

# Predeclared grids.
K_VALUES = (1, 2, 3, 4, 5)
W_VALUES = (0, 1, 2, 4, 8, -1)  # -1 encodes an unbounded (latching) window
F_VALUES = (1, 2)
CASCADE_K = (2, 3)

MIN_LOW_PRIOR_PRECISION = 0.60
FPR_CAPS = (0.0005, 0.0010, 0.0015, 0.0030, 0.0050)
OBJECTIVES = {"O1_early": "low_prior_tp", "O2_recall": "tp"}


class Cohort:
    """Alarm arrays, labels and the survival prior for one cohort."""

    def __init__(self, name: str) -> None:
        self.name = name
        graph = load_npz(PROFILE_ROOT / f"{name}.npz")
        labels = pd.read_csv(LABEL_PATHS[name])
        if len(labels) != len(graph["episode"]):
            raise ValueError(f"{name}: labels and cache disagree on episode count")
        task = graph["task_names"].astype(str)[graph["task_index"].astype(int)]
        if not np.array_equal(labels["task"].astype(str).to_numpy(), task):
            raise ValueError(f"{name}: label task order does not match the cache")
        if not np.array_equal(
            labels["episode"].to_numpy(int), graph["episode"].astype(int)
        ):
            raise ValueError(f"{name}: label episode order does not match the cache")

        self.risk = labels["original_failure"].to_numpy(bool)
        self.suite = suite_of(graph)
        self.length = graph["length"].astype(int)
        self.queries = int(graph["valid"].shape[1])
        self.priors = survival_prior(self.suite, self.length, self.risk)
        self.episodes = len(self.risk)

        with np.load(ALARM_ROOT / f"{name}_alarms.npz", allow_pickle=False) as archive:
            self.alarms = {
                key: np.asarray(archive[key], dtype=np.int16)
                for key in archive.files
                if key != "schema"
            }

        suites = sorted(self.priors)
        self.suite_id = np.searchsorted(np.asarray(suites), self.suite)
        table = np.full((len(suites), self.queries), np.nan)
        for position, suite in enumerate(suites):
            for chunk, value in self.priors[suite].items():
                if chunk < self.queries:
                    table[position, chunk] = value
        self.prior_table = table

    def prior_of(self, first: np.ndarray) -> np.ndarray:
        out = np.full(len(first), np.nan)
        fired = first >= 0
        out[fired] = self.prior_table[self.suite_id[fired], first[fired]]
        return out

    def votes(self, pool: str, mode: str, level: str) -> np.ndarray:
        """(n_members, n_episodes) first-alarm chunks, -1 when never."""
        return np.stack(
            [self.alarms[f"{q}|{mode}|{level}"] for q in POOLS[pool]], axis=0
        )

    def score(self, first: np.ndarray) -> dict[str, Any]:
        return score_candidate(first, self.risk, self.prior_of(first))


def _vote_grid(votes: np.ndarray, queries: int) -> np.ndarray:
    """(n_members, n_episodes, n_queries) one-hot indicator of the first alarm."""
    members, episodes = votes.shape
    grid = np.zeros((members, episodes, queries), dtype=bool)
    rows, cols = np.nonzero(votes >= 0)
    grid[rows, cols, votes[rows, cols]] = True
    return grid


def _window_count(grid: np.ndarray, window: int) -> np.ndarray:
    """Votes present in the trailing window of length ``window`` + 1 chunks."""
    cumulative = np.cumsum(grid, axis=-1)
    if window < 0:
        return cumulative
    shifted = np.zeros_like(cumulative)
    if window + 1 < cumulative.shape[-1]:
        shifted[..., window + 1 :] = cumulative[..., : -(window + 1)]
    return cumulative - shifted


def _first_true(condition: np.ndarray) -> np.ndarray:
    fired = condition.any(axis=1)
    out = np.argmax(condition, axis=1).astype(np.int16)
    out[~fired] = -1
    return out


class WindowCache:
    """Windowed member counts and frame counts for one (pool, mode, level)."""

    def __init__(self, cohort: Cohort, pool: str, mode: str, level: str) -> None:
        members = POOLS[pool]
        grid = _vote_grid(cohort.votes(pool, mode, level), cohort.queries)
        self.frame_index = [
            [position for position, q in enumerate(members) if FRAMES[q] == frame]
            for frame in FRAME_ORDER
        ]
        self._grid = grid
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def counts(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        if window not in self._cache:
            per_member = _window_count(self._grid, window)
            total = per_member.sum(axis=0)
            frames = np.zeros_like(total)
            for positions in self.frame_index:
                if positions:
                    frames += (per_member[positions].sum(axis=0) > 0).astype(total.dtype)
            self._cache[window] = (total, frames)
        return self._cache[window]


def quorum_alarm(cache: WindowCache, k: int, window: int, frames: int) -> np.ndarray:
    total, frame_count = cache.counts(window)
    return _first_true((total >= k) & (frame_count >= frames))


def cascade_alarm(
    strict: WindowCache, loose: WindowCache, k: int, window: int, frames: int
) -> np.ndarray:
    strict_total, _ = strict.counts(window)
    loose_total, loose_frames = loose.counts(window)
    return _first_true(
        (strict_total >= 1) & (loose_total >= k) & (loose_frames >= frames)
    )


def pairwise_and(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    both = (left >= 0) & (right >= 0)
    out = np.full(len(left), -1, dtype=np.int16)
    out[both] = np.maximum(left[both], right[both])
    return out


def pool_or(votes: np.ndarray) -> np.ndarray:
    masked = np.where(votes >= 0, votes, np.iinfo(np.int16).max)
    best = masked.min(axis=0)
    return np.where(best == np.iinfo(np.int16).max, -1, best).astype(np.int16)


def enumerate_quorum_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for mode in MODES:
        for pool in POOLS:
            for level in LEVELS:
                for k in K_VALUES:
                    for window in W_VALUES:
                        for frames in F_VALUES:
                            if frames > k:
                                continue
                            if k == 1 and (window != W_VALUES[0] or frames != 1):
                                continue  # OR does not depend on W or f
                            rules.append(
                                {
                                    "family": "quorum",
                                    "mode": mode,
                                    "pool": pool,
                                    "level": level,
                                    "level_loose": level,
                                    "k": k,
                                    "window": window,
                                    "frames": frames,
                                }
                            )
    return rules


def enumerate_cascade_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    for mode in MODES:
        for pool in POOLS:
            for strict_position, strict in enumerate(NUMERIC_LEVELS):
                for loose in NUMERIC_LEVELS[:strict_position]:
                    for k in CASCADE_K:
                        for window in W_VALUES:
                            for frames in F_VALUES:
                                rules.append(
                                    {
                                        "family": "cascade",
                                        "mode": mode,
                                        "pool": pool,
                                        "level": strict,
                                        "level_loose": loose,
                                        "k": k,
                                        "window": window,
                                        "frames": frames,
                                    }
                                )
    return rules


def rule_key(rule: dict[str, Any]) -> str:
    return "|".join(
        str(rule[field])
        for field in ("family", "mode", "pool", "level", "level_loose", "k", "window", "frames")
    )


def evaluate_rules(
    cohort: Cohort, rules: Iterable[dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    caches: dict[tuple[str, str, str], WindowCache] = {}
    or_alarm: dict[tuple[str, str, str], np.ndarray] = {}
    or_tp: dict[tuple[str, str, str], int] = {}
    rows: list[dict[str, Any]] = []
    firsts: dict[str, np.ndarray] = {}

    def cache_for(pool: str, mode: str, level: str) -> WindowCache:
        key = (pool, mode, level)
        if key not in caches:
            caches[key] = WindowCache(cohort, pool, mode, level)
            reference = pool_or(cohort.votes(pool, mode, level))
            or_alarm[key] = reference
            or_tp[key] = int(((reference >= 0) & cohort.risk).sum())
        return caches[key]

    for rule in rules:
        loose = cache_for(rule["pool"], rule["mode"], rule["level_loose"])
        if rule["family"] == "quorum":
            first = quorum_alarm(loose, rule["k"], rule["window"], rule["frames"])
        else:
            strict = cache_for(rule["pool"], rule["mode"], rule["level"])
            first = cascade_alarm(
                strict, loose, rule["k"], rule["window"], rule["frames"]
            )
        key = rule_key(rule)
        firsts[key] = first
        reference = or_alarm[(rule["pool"], rule["mode"], rule["level_loose"])]
        fired = first >= 0
        delay = (first[fired] - reference[fired]).astype(float)
        rows.append(
            {
                "rule": key,
                **rule,
                **cohort.score(first),
                "union_tp": or_tp[(rule["pool"], rule["mode"], rule["level_loose"])],
                "union_coverage_share": (
                    int(((first >= 0) & cohort.risk).sum())
                    / max(or_tp[(rule["pool"], rule["mode"], rule["level_loose"])], 1)
                ),
                "median_confirmation_delay": (
                    float(np.median(delay)) if delay.size else float("nan")
                ),
                "alarms": int(fired.sum()),
            }
        )
    return pd.DataFrame(rows), firsts


__all__ = [
    "ALARM_ROOT",
    "Cohort",
    "FPR_CAPS",
    "FRAMES",
    "LEVELS",
    "LOW_PRIOR",
    "MIN_LOW_PRIOR_PRECISION",
    "MODES",
    "NUMERIC_LEVELS",
    "OBJECTIVES",
    "POOLS",
    "cascade_alarm",
    "enumerate_cascade_rules",
    "enumerate_quorum_rules",
    "evaluate_rules",
    "pairwise_and",
    "pool_or",
    "prior_of",
    "quorum_alarm",
    "rule_key",
    "score_candidate",
]
