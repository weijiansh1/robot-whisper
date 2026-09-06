#!/usr/bin/env python3
"""Per-trajectory complementarity ledger over six MoE-routing alarm channels.

Several alarms have accumulated in this workspace -- v4's layer-median lock rule,
v7's ``relative_freeze``, v7's ``confirmed_turbulence``, their union
(``v7_guard``), this bundle's conjunctive progress-ratio rule, and the union of
the last two.  Each has been published with its own recall and precision.  What
has never been produced is a *per-episode* record of which episodes each one
catches, when, and at what cost.

Low score correlation does not imply useful complementarity and a different
method name certainly does not.  For every ordered pair (base, candidate) this
script produces the only four-way split that bears on a deployment decision, on
risk episodes:

    candidate_only  candidate detects, base misses           -- genuine coverage
    earlier         both detect, candidate >= 2 chunks ahead -- genuine lead
    redundant       both detect, first alarms within 1 chunk -- same detector
    later           both detect, candidate >= 2 chunks late

plus, on timely successes, ``added_false_alarms`` = candidate fires where the
base is silent.  Task-clustered bootstrap intervals are attached to the two
decision-relevant quantities (``candidate_only`` and ``added_false_alarms``).

**Nothing is fused, fitted or selected here.**  Every channel is read at its
already-frozen operating point; the progress-ratio channel FAILED its
pre-registered gate on development (19 low-prior loop true positives against a
floor of 20, ``results/loop_operating_point/selection.json``, ``feasible:
false``) and is carried as a candidate channel, not as a validated detector.

``external_8b`` is NOT a pristine holdout: it has been examined across v3-v7 and
the HB front/back bundle.  Both cohorts are reported; the external numbers are a
consistency check on an already-seen cohort.

Run:  python experiments/build_complementarity_ledger.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))
sys.path.insert(0, str(WORKSPACE / "moe-hb-front-back-0905/experiments"))
sys.path.insert(0, str(WORKSPACE / "moe-v4-0904/experiments"))

import progress_ratio as pr  # noqa: E402
import select_loop_operating_point as loop  # noqa: E402
import select_operating_point_v12 as base_v12  # noqa: E402
from analyze_front_back import (  # noqa: E402
    calibrated_development_alarm,
    calibrated_external_alarm,
    mobility_representations,
    profile_task_map,
)

DEFAULT_OUTPUT = BUNDLE / "results/ledger"

CACHE_ROOT = BUNDLE / "results/progress_cache"
LABEL_ROOT = WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
BELIEF_EPISODES = TAXONOMY / "belief_mismatch_episodes.csv.gz"
EVENTS = TAXONOMY / "events.csv.gz"
MOBILITY_ROOT = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
V4_SEALED = WORKSPACE / "moe-v4-0904/results/cache_new_v4/sealed_first_alarms.npz"
V7_SEALED = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"

SCHEMA = "himoe.progress_ratio_v12.complementarity_ledger.v1"
JOIN_KEY = ["task", "init_state_id", "flow_noise_seed"]

# (ledger cohort, progress-cache stem, v4 mobility stem, taxonomy run, label stem,
#  v7 sealed-array prefix)
COHORTS = (
    ("development_main", "development_main", "main_reference", "seed1000_1007",
     "development_main", "main"),
    ("external_8b", "external_8b", "external_8b", "seed1008_1015",
     "external_8b", "external"),
)

# ---------------------------------------------------------------------------
# Frozen operating points.  None of these is chosen here.
# ---------------------------------------------------------------------------

# v4 (moe-hb-front-back-0905/experiments/evaluate_layer_survival_baseline.py)
V4_LOCK = ("low", 4, 4, 0.75)          # layer-median Hellinger mobility, lock branch
V4_INSTABILITY = ("high", 4, 8, 0.80)  # L5 sustained switching branch
# Counts published in the HB REPORT_ZH for external_8b, as (risk, timely).
V4_PUBLISHED_EXTERNAL = {
    "front_lock": (240, 36),
    "back_lock": (409, 146),
    "all_lock": (382, 76),
    "L5_switching": (34, 5),
    "v4": (410, 81),
}

# progress ratio (moe-progress-ratio-v12-0906/results/loop_operating_point)
PR_WINDOW = 3
PR_GROUP = "all"
PR_CONFIRMATIONS = 2
PR_QUANTILE = 0.025
PR_RATIO_THRESHOLD = 0.2943268120288849
PR_EPS_LENGTH = 0.05766395851969719
PR_ELL_STAR = 0.14589208364486694
PR_REFERENCE_EPISODES = 16_000
# Anchors recorded in results/loop_operating_point/selection.json for that cell.
PR_DEV_ALARM_EPISODES = 94
PR_DEV_LOOP_TP = 35

CHANNELS = (
    "v4",
    "v7_freeze",
    "v7_turbulence",
    "v7_guard",
    "progress_ratio",
    "v7_guard_or_progress_ratio",
)

# Episode-count anchors from analysis_trap_taxonomy; a mismatch aborts the run.
EXPECTED_LABEL_COUNTS = {
    "development_main": {"risk": 487, "loop": 256, "static": 135,
                         "phantom": 93, "pure_phantom": 22},
    "external_8b": {"risk": 564, "loop": 314, "static": 164,
                    "phantom": 115, "pure_phantom": 32},
}

# Environment steps executed per action chunk / router query.  This IS recorded:
# every rollout server log carries "predicted_action_steps": 10, and the
# per-episode records agree exactly across all four suites
# (original_action_steps / n_queries = 300/30 = 520/52 = 280/28 = 220/22 = 10).
STEPS_PER_CHUNK = 10
STEPS_PER_CHUNK_SOURCE = (
    "double-selete/trainfree/results/timeout_extension_plus10/logs/server_*.json "
    "predicted_action_steps=10, cross-checked against episodes/*/*.json "
    "original_action_steps / n_queries = 10 in every suite"
)

LEAD_CHUNKS = 2      # "meaningfully earlier" means at least this many chunks
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260906
SUBSETS = ("all_risk", "loop", "static", "pure_phantom")

# Channels built as the earlier-of other channels.  A pair drawn from a nesting
# relation cannot show base_only or later on the superset side, so its buckets
# are arithmetic, not evidence.  Flagged in pairs.csv so the two kinds of row are
# never averaged together.
NESTED_COMPONENTS = {
    "v7_guard": {"v7_freeze", "v7_turbulence"},
    "v7_guard_or_progress_ratio": {
        "v7_guard", "progress_ratio", "v7_freeze", "v7_turbulence"
    },
}


def structurally_nested(base: str, candidate: str) -> bool:
    """True when one channel is a constructed union containing the other."""
    return (
        candidate in NESTED_COMPONENTS.get(base, ())
        or base in NESTED_COMPONENTS.get(candidate, ())
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Pure decision functions -- pinned by tests/test_ledger.py before use
# --------------------------------------------------------------------------- #


def bucket_masks(
    base_first: np.ndarray, candidate_first: np.ndarray, lead: int = LEAD_CHUNKS
) -> dict[str, np.ndarray]:
    """Partition episodes by how the candidate's first alarm relates to the base's.

    ``-1`` means "never alarmed".  ``lead`` is the number of chunks a candidate
    must gain before the lead is called meaningful; everything strictly inside
    that gap is ``redundant``.  With the pre-registered ``lead = 2``: a gain of
    exactly 2 chunks is ``earlier``, a gain of exactly 1 chunk is ``redundant``,
    and symmetrically on the ``later`` side.

    The six masks are mutually exclusive and cover every episode, so the caller
    can never silently lose an episode between buckets.
    """
    base = np.asarray(base_first, dtype=np.int64)
    candidate = np.asarray(candidate_first, dtype=np.int64)
    if base.shape != candidate.shape:
        raise ValueError("first-alarm arrays do not align")
    if base.ndim != 1:
        raise ValueError(f"first-alarm arrays must be one-dimensional, got {base.shape}")
    if lead < 1:
        raise ValueError("lead must be a positive number of chunks")

    base_hit = base >= 0
    candidate_hit = candidate >= 0
    both = base_hit & candidate_hit
    gain = base - candidate  # > 0 means the candidate alarmed first
    tolerance = lead - 1     # the redundancy band, so the three both-detect
    #                          buckets stay exhaustive by construction
    return {
        "candidate_only": candidate_hit & ~base_hit,
        "base_only": base_hit & ~candidate_hit,
        "earlier": both & (gain >= lead),
        "redundant": both & (np.abs(gain) <= tolerance),
        "later": both & (gain <= -lead),
        "neither": ~base_hit & ~candidate_hit,
    }


def first_or(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Earliest of two first-alarm arrays; ``-1`` only if neither ever fired.

    Byte-for-byte the rule ``moe-v7-0905/method/intrinsic_guard_monitor.first_or``
    uses to build ``v7_guard`` out of freeze and turbulence.
    """
    left = np.asarray(left, dtype=np.int16)
    right = np.asarray(right, dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError("first-alarm arrays do not align")
    return np.where(
        left < 0, right, np.where(right < 0, left, np.minimum(left, right))
    ).astype(np.int16)


def ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def clustered_interval(
    task_code: np.ndarray,
    n_tasks: int,
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws_index: np.ndarray,
) -> tuple[float, float]:
    """Task-clustered percentile bootstrap on a ratio of two episode indicators.

    Episodes inside a task share an init-state distribution and are not
    independent units, so the resampling unit is the task.  This mirrors
    ``moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py::clustered_interval``;
    ``draws_index`` is a pre-drawn ``[draws, n_tasks]`` matrix of task positions
    so that every pair in a cohort sees the same resample, which makes the
    intervals comparable across rows rather than independently noisy.
    """
    numerator = np.asarray(numerator, dtype=bool)
    denominator = np.asarray(denominator, dtype=bool)
    if numerator.shape != denominator.shape or numerator.shape != task_code.shape:
        raise ValueError("indicator arrays do not align")
    task_numerator = np.bincount(task_code[numerator], minlength=n_tasks).astype(float)
    task_denominator = np.bincount(
        task_code[denominator], minlength=n_tasks
    ).astype(float)
    if task_denominator.sum() == 0:
        return float("nan"), float("nan")
    sampled_numerator = task_numerator[draws_index].sum(axis=1)
    sampled_denominator = task_denominator[draws_index].sum(axis=1)
    values = sampled_numerator / np.maximum(sampled_denominator, 1.0)
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Channel 1: v4 -- rebuilt, then anchored against the published counts
# --------------------------------------------------------------------------- #


def build_v4_alarms() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Rebuild the v4 rule on both cohorts and refuse to continue if it drifts.

    ``v4 = all-layer median lock OR L5 sustained switching``.  The external
    branch is calibrated against the pooled development reference exactly as the
    HB bundle does; the development branch uses the cross-fitted per-task
    thresholds v4 itself used.  Both are checked: external against the five
    counts published in the HB REPORT_ZH, development against v4's own sealed
    first-alarm arrays element-by-element.
    """
    main = load_npz(MOBILITY_ROOT / "main_reference.npz")
    extra = load_npz(MOBILITY_ROOT / "extra_reference.npz")
    external = load_npz(MOBILITY_ROOT / "external_8b.npz")
    reference_map = profile_task_map([main, extra])
    reference_representations = {
        id(main): mobility_representations(main),
        id(extra): mobility_representations(extra),
    }
    external_representation = mobility_representations(external)
    main_representation = mobility_representations(main)

    def external_alarm(name: str, head: tuple[str, int, int, float]) -> np.ndarray:
        direction, width, confirmations, quantile = head
        return calibrated_external_alarm(
            external, external_representation[name], reference_map,
            reference_representations, name, direction, width, confirmations, quantile,
        )

    def development_alarm(name: str, head: tuple[str, int, int, float]) -> np.ndarray:
        direction, width, confirmations, quantile = head
        return calibrated_development_alarm(
            main, main_representation[name], direction, width, confirmations, quantile,
        )

    external_branches = {
        "front_lock": external_alarm("front_median", V4_LOCK),
        "back_lock": external_alarm("back_median", V4_LOCK),
        "all_lock": external_alarm("all_median", V4_LOCK),
        "L5_switching": external_alarm("L5", V4_INSTABILITY),
    }
    external_branches["v4"] = first_or(
        external_branches["all_lock"], external_branches["L5_switching"]
    )
    development_branches = {
        "all_lock": development_alarm("all_median", V4_LOCK),
        "L5_switching": development_alarm("L5", V4_INSTABILITY),
    }
    development_branches["v4"] = first_or(
        development_branches["all_lock"], development_branches["L5_switching"]
    )

    external_labels = pd.read_csv(LABEL_ROOT / "external_8b_clean_labels.csv")
    external_risk = external_labels["original_failure"].to_numpy(bool)
    rebuilt = {}
    for name, first in external_branches.items():
        fired = first >= 0
        rebuilt[name] = (int((fired & external_risk).sum()),
                         int((fired & ~external_risk).sum()))
    mismatched = {
        name: {"rebuilt": rebuilt[name], "published": list(published)}
        for name, published in V4_PUBLISHED_EXTERNAL.items()
        if rebuilt[name] != published
    }
    if mismatched:
        raise ValueError(
            "BLOCKED: the v4 rebuild does not reproduce the published external "
            f"counts: {mismatched}"
        )

    sealed = load_npz(V4_SEALED)
    sealed_match = {
        "development_all_lock": bool(
            np.array_equal(development_branches["all_lock"], sealed["main_lock"])
        ),
        "development_L5_switching": bool(
            np.array_equal(development_branches["L5_switching"],
                           sealed["main_instability"])
        ),
        "development_v4": bool(
            np.array_equal(development_branches["v4"], sealed["main_dual"])
        ),
        "external_v4": bool(
            np.array_equal(external_branches["v4"], sealed["external_dual"])
        ),
    }
    if not all(sealed_match.values()):
        raise ValueError(
            f"BLOCKED: the v4 rebuild disagrees with v4's sealed arrays: {sealed_match}"
        )

    anchor = {
        "published_external_counts_risk_timely": plain(V4_PUBLISHED_EXTERNAL),
        "rebuilt_external_counts_risk_timely": plain(rebuilt),
        "external_counts_match": True,
        "development_anchor": (
            "element-wise identity with moe-v4-0904 sealed_first_alarms.npz "
            "(main_lock / main_instability / main_dual); the HB script publishes "
            "counts only for external_8b"
        ),
        "development_sealed_identity": sealed_match,
        "development_alarm_episodes": {
            "all_lock": int((development_branches["all_lock"] >= 0).sum()),
            "L5_switching": int((development_branches["L5_switching"] >= 0).sum()),
            "v4": int((development_branches["v4"] >= 0).sum()),
        },
        "lock_head": {"representation": "all_median", "direction": V4_LOCK[0],
                      "width": V4_LOCK[1], "persistence": V4_LOCK[2],
                      "quantile": V4_LOCK[3]},
        "switching_head": {"representation": "L5", "direction": V4_INSTABILITY[0],
                           "width": V4_INSTABILITY[1],
                           "persistence": V4_INSTABILITY[2],
                           "quantile": V4_INSTABILITY[3]},
    }
    return (
        {"development_main": development_branches["v4"],
         "external_8b": external_branches["v4"]},
        anchor,
    )


# --------------------------------------------------------------------------- #
# Channel 5: the conjunctive progress-ratio rule, at its frozen constants
# --------------------------------------------------------------------------- #


def build_progress_ratio_alarms() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """``L(q,W) >= ell* AND R(q,W) < r*`` for K consecutive chunks.

    The three constants are unlabeled order statistics of the pooled
    16,000-trajectory development reference.  They are recomputed here from that
    reference and checked against the values frozen in
    ``results/loop_operating_point/selection.json`` before being applied to
    either cohort; external_8b never contributes to a threshold.
    """
    main = load_npz(CACHE_ROOT / "development_main.npz")
    extra = load_npz(CACHE_ROOT / "development_extra.npz")
    n_main = len(main["episode"])
    if n_main + len(extra["episode"]) != PR_REFERENCE_EPISODES:
        raise ValueError("the pooled reference is not 16,000 trajectories")

    pooled_valid = np.concatenate(
        (main["valid"].astype(bool), extra["valid"].astype(bool))
    )
    pooled_lag = np.concatenate((main["lag_distance"], extra["lag_distance"]), axis=0)

    pooled_length = pr.path_length(pooled_lag[:, :, 0, :], PR_WINDOW)
    eps_length = float(
        np.quantile(pooled_length[np.isfinite(pooled_length)],
                    base_v12.EPS_LENGTH_QUANTILE)
    )
    pooled_elementwise = pr.progress_ratio(
        pooled_lag[:, :, pr.LAGS.index(PR_WINDOW), :], pooled_length, eps_length
    )
    pooled_ratio = pr.group_ratio(pooled_elementwise, PR_GROUP)
    pooled_grouped_length = pr.group_ratio(pooled_length, PR_GROUP)
    ell_star = loop.length_threshold(pooled_grouped_length, pooled_valid)
    pooled_persistent_ratio = np.where(
        pooled_valid, pr.persistent_low(pooled_ratio, PR_CONFIRMATIONS), np.nan
    )
    ratio_threshold = pr.quantile_lower(
        pr.row_min(pooled_persistent_ratio), PR_QUANTILE
    )

    for name, rebuilt, frozen in (
        ("eps_length", eps_length, PR_EPS_LENGTH),
        ("ell_star", ell_star, PR_ELL_STAR),
        ("ratio_threshold", ratio_threshold, PR_RATIO_THRESHOLD),
    ):
        if not math.isclose(rebuilt, frozen, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"BLOCKED: rebuilt {name}={rebuilt!r} does not match the frozen "
                f"{frozen!r}"
            )
    del pooled_lag, pooled_length, pooled_elementwise, pooled_ratio

    def apply(cache: dict[str, np.ndarray]) -> np.ndarray:
        valid = cache["valid"].astype(bool)
        lag = cache["lag_distance"]
        length = pr.path_length(lag[:, :, 0, :], PR_WINDOW)
        elementwise = pr.progress_ratio(
            lag[:, :, pr.LAGS.index(PR_WINDOW), :], length, eps_length
        )
        persistent_ratio = np.where(
            valid, pr.persistent_low(pr.group_ratio(elementwise, PR_GROUP),
                                     PR_CONFIRMATIONS), np.nan
        )
        persistent_length = loop.persistent_high(
            pr.group_ratio(length, PR_GROUP), PR_CONFIRMATIONS
        )
        return loop.first_conjunctive_alarm(
            persistent_ratio, persistent_length, ratio_threshold, ell_star, valid
        )

    alarms = {
        "development_main": apply(main),
        "external_8b": apply(load_npz(CACHE_ROOT / "external_8b.npz")),
    }
    anchor = {
        "rule": "L(q,W) >= ell* AND R(q,W) < r*, K consecutive chunks",
        "window": PR_WINDOW,
        "layer_group": PR_GROUP,
        "confirmations": PR_CONFIRMATIONS,
        "quantile": PR_QUANTILE,
        "ratio_threshold": ratio_threshold,
        "ell_star": ell_star,
        "eps_length": eps_length,
        "constants_match_frozen_selection": True,
        "calibration_corpus": "pooled 16,000-trajectory development reference",
        "external_used_in_calibration": False,
        "development_alarm_episodes": int((alarms["development_main"] >= 0).sum()),
        "development_alarm_episodes_expected": PR_DEV_ALARM_EPISODES,
        "pre_registered_gate": "FAILED (19 low-prior loop TP against a floor of 20)",
        "status": "candidate channel, not a validated detector",
    }
    return alarms, anchor


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def aligned_labels(
    cache: dict[str, np.ndarray], path: Path, cohort: str
) -> pd.DataFrame:
    """Row-for-row equivalent of ``evaluate_intrinsic_guard_v7.aligned_labels``."""
    task = task_of(cache)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(path)[
        ["task", "episode", "original_failure", "failure",
         "late_success_plus10_queries"]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError(f"{cohort} labels do not align")
    merged["cohort"] = cohort
    merged["suite"] = merged["task"].str.split("/", n=1).str[0]
    return merged.reset_index(drop=True)


def attach_failure_modes(
    labels: pd.DataFrame,
    cache: dict[str, np.ndarray],
    mobility: dict[str, np.ndarray],
    run_id: str,
    cohort: str,
) -> pd.DataFrame:
    """Join the trap taxonomy on ``(task, init_state_id, flow_noise_seed)``.

    Episode ids repeat across runs, so ``(task, episode)`` is not a unique key
    for the taxonomy tables; the init-state / seed triple is.
    """
    for field in ("episode", "init_state_id"):
        if not np.array_equal(mobility[field], cache[field]):
            raise ValueError(f"{cohort}: v4 mobility cache is not row-aligned ({field})")
    if not np.array_equal(task_of(mobility), task_of(cache)):
        raise ValueError(f"{cohort}: v4 mobility cache is not row-aligned (task)")

    keyed = labels.assign(
        init_state_id=cache["init_state_id"].astype(int),
        flow_noise_seed=mobility["flow_noise_seed"].astype(int),
    )
    if keyed.duplicated(JOIN_KEY).any():
        raise ValueError(f"{cohort}: the cohort key is not unique")

    belief = pd.read_csv(BELIEF_EPISODES)
    events = pd.read_csv(EVENTS)
    events["task"] = events["task_key"]  # events.csv stores suite and task apart
    taxonomy = belief.merge(
        events[["run", "task", "episode", "loop_onset", "static_onset"]],
        on=["run", "task", "episode"], validate="one_to_one",
    )
    taxonomy = taxonomy[taxonomy["run"] == run_id]
    columns = JOIN_KEY + ["success", "belief_mismatch", "loop_onset", "static_onset"]
    merged = keyed.merge(
        taxonomy[columns], on=JOIN_KEY, how="left", validate="one_to_one"
    ).sort_values("row").reset_index(drop=True)
    if merged["success"].isna().any():
        raise ValueError(f"{cohort}: the trap taxonomy does not cover the cohort")

    fail = ~merged["success"].to_numpy(bool)
    risk = merged["original_failure"].to_numpy(bool)
    if not np.array_equal(fail, risk):
        raise ValueError(
            f"{cohort}: taxonomy failure and original_failure disagree row-for-row"
        )
    merged["risk"] = risk
    merged["timely"] = ~risk
    merged["loop"] = (merged["loop_onset"].to_numpy(int) >= 0) & fail
    merged["static"] = (merged["static_onset"].to_numpy(int) >= 0) & fail
    merged["phantom"] = merged["belief_mismatch"].to_numpy(bool) & fail
    merged["pure_phantom"] = merged["phantom"] & ~merged["loop"] & ~merged["static"]

    observed = {
        name: int(merged[name].to_numpy(bool).sum())
        for name in ("risk", "loop", "static", "phantom", "pure_phantom")
    }
    if observed != EXPECTED_LABEL_COUNTS[cohort]:
        raise ValueError(
            f"BLOCKED: {cohort} label counts {observed} do not match the expected "
            f"{EXPECTED_LABEL_COUNTS[cohort]}"
        )
    return merged


def survival_prior_matrix(labels: pd.DataFrame, max_chunk: int) -> dict[str, np.ndarray]:
    """``P(risk | still running at chunk q)`` per suite, estimated in-cohort."""
    priors: dict[str, np.ndarray] = {}
    risk = labels["risk"].to_numpy(bool)
    length = labels["length"].to_numpy(int)
    for suite in sorted(labels["suite"].unique()):
        take = labels["suite"].to_numpy(str) == suite
        curve = np.full(max_chunk, np.nan, dtype=np.float64)
        for chunk in range(max_chunk):
            alive = take & (length > chunk)
            if alive.any():
                curve[chunk] = float(risk[alive].mean())
        priors[str(suite)] = curve
    return priors


def matched_priors(
    first: np.ndarray, suite: np.ndarray, priors: dict[str, np.ndarray]
) -> np.ndarray:
    """The survival prior at each episode's own first-alarm chunk; NaN if silent."""
    output = np.full(len(first), np.nan, dtype=np.float64)
    for name, curve in priors.items():
        take = (suite == name) & (first >= 0)
        if take.any():
            output[take] = curve[first[take]]
    return output


# --------------------------------------------------------------------------- #
# Ledger construction
# --------------------------------------------------------------------------- #


def cohort_frame(
    cohort: str, labels: pd.DataFrame, alarms: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    length = labels["length"].to_numpy(int)
    suite = labels["suite"].to_numpy(str)
    priors = survival_prior_matrix(labels, int(length.max()))

    frame = labels[
        ["cohort", "row", "task", "suite", "length", "risk", "timely",
         "loop", "static", "phantom", "pure_phantom"]
    ].copy()
    diagnostics: dict[str, Any] = {}
    for channel in CHANNELS:
        first = np.asarray(alarms[channel], dtype=np.int64)
        if len(first) != len(labels):
            raise ValueError(f"{cohort}/{channel}: alarm array length mismatch")
        fired = first >= 0
        after_horizon = int((fired & (first >= length)).sum())
        if after_horizon:
            raise ValueError(
                f"{cohort}/{channel}: {after_horizon} alarms land at or past the "
                "episode horizon"
            )
        frame[f"{channel}_chunk"] = first
        frame[f"{channel}_prior"] = matched_priors(first, suite, priors)
        diagnostics[channel] = channel_diagnostics(
            first, length, labels["risk"].to_numpy(bool), frame[f"{channel}_prior"]
        )
    return frame, {"channels": diagnostics,
                   "survival_prior": {k: plain(v) for k, v in priors.items()}}


def channel_diagnostics(
    first: np.ndarray, length: np.ndarray, risk: np.ndarray, prior: np.ndarray
) -> dict[str, Any]:
    """Alarm timing, in chunks and in the environment steps already executed."""
    fired = first >= 0

    def timing(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return {"n": 0}
        chunks = first[mask].astype(float)
        lead = (length[mask] - 1 - first[mask]).astype(float)
        quantiles = np.quantile(chunks, (0.25, 0.5, 0.75))
        return {
            "n": int(mask.sum()),
            "chunk_q25": float(quantiles[0]),
            "chunk_median": float(quantiles[1]),
            "chunk_q75": float(quantiles[2]),
            "steps_executed_q25": float(quantiles[0] * STEPS_PER_CHUNK),
            "steps_executed_median": float(quantiles[1] * STEPS_PER_CHUNK),
            "steps_executed_q75": float(quantiles[2] * STEPS_PER_CHUNK),
            "lead_chunks_median": float(np.median(lead)),
            "lead_steps_median": float(np.median(lead) * STEPS_PER_CHUNK),
        }

    prior = np.asarray(prior, dtype=float)
    return {
        "alarms": int(fired.sum()),
        "tp": int((fired & risk).sum()),
        "fp": int((fired & ~risk).sum()),
        "risk_recall": ratio(int((fired & risk).sum()), int(risk.sum())),
        "timely_fpr": ratio(int((fired & ~risk).sum()), int((~risk).sum())),
        "precision": ratio(int((fired & risk).sum()), int(fired.sum())),
        "matched_prior_mean": float(np.nanmean(prior[fired])) if fired.any() else None,
        "low_prior_alarms": int((fired & (prior < 0.25)).sum()),
        "low_prior_tp": int((fired & risk & (prior < 0.25)).sum()),
        "on_risk": timing(fired & risk),
        "on_timely": timing(fired & ~risk),
        "all": timing(fired),
    }


def pair_rows(
    cohort: str,
    frame: pd.DataFrame,
    draws_index: np.ndarray,
) -> list[dict[str, Any]]:
    task = frame["task"].to_numpy(str)
    task_names, task_code = np.unique(task, return_inverse=True)
    n_tasks = len(task_names)
    risk = frame["risk"].to_numpy(bool)
    timely = frame["timely"].to_numpy(bool)
    firsts = {
        channel: frame[f"{channel}_chunk"].to_numpy(np.int64) for channel in CHANNELS
    }
    subsets = {
        "all_risk": risk,
        "loop": frame["loop"].to_numpy(bool),
        "static": frame["static"].to_numpy(bool),
        "pure_phantom": frame["pure_phantom"].to_numpy(bool),
    }

    rows: list[dict[str, Any]] = []
    for base in CHANNELS:
        for candidate in CHANNELS:
            if base == candidate:
                continue
            masks = bucket_masks(firsts[base], firsts[candidate])
            gain = firsts[base] - firsts[candidate]
            both = masks["earlier"] | masks["redundant"] | masks["later"]
            base_hit = firsts[base] >= 0
            candidate_hit = firsts[candidate] >= 0

            # timely-side cost is a property of the pair, not of the subset
            added = timely & candidate_hit & ~base_hit
            added_low, added_high = clustered_interval(
                task_code, n_tasks, added, timely, draws_index
            )
            base_silent_timely = timely & ~base_hit

            for subset_name, subset in subsets.items():
                base_miss = subset & ~base_hit
                base_detect = subset & base_hit
                subset_both = subset & both
                only = subset & masks["candidate_only"]
                low, high = clustered_interval(
                    task_code, n_tasks, only, base_miss, draws_index
                )
                lead_gain = gain[subset_both]
                row = {
                    "cohort": cohort,
                    "subset": subset_name,
                    "base": base,
                    "candidate": candidate,
                    "structurally_nested": structurally_nested(base, candidate),
                    "subset_episodes": int(subset.sum()),
                    "base_detects": int(base_detect.sum()),
                    "base_misses": int(base_miss.sum()),
                    "candidate_detects": int((subset & candidate_hit).sum()),
                    "both_detect": int(subset_both.sum()),
                    "candidate_only": int(only.sum()),
                    "candidate_only_frac_of_base_misses": ratio(
                        int(only.sum()), int(base_miss.sum())
                    ),
                    "candidate_only_ci_low": low,
                    "candidate_only_ci_high": high,
                    "earlier": int((subset & masks["earlier"]).sum()),
                    "earlier_frac_of_base_detects": ratio(
                        int((subset & masks["earlier"]).sum()), int(base_detect.sum())
                    ),
                    "redundant": int((subset & masks["redundant"]).sum()),
                    "redundant_frac_of_base_detects": ratio(
                        int((subset & masks["redundant"]).sum()), int(base_detect.sum())
                    ),
                    "later": int((subset & masks["later"]).sum()),
                    "later_frac_of_base_detects": ratio(
                        int((subset & masks["later"]).sum()), int(base_detect.sum())
                    ),
                    "base_only": int((subset & masks["base_only"]).sum()),
                    "base_only_frac_of_base_detects": ratio(
                        int((subset & masks["base_only"]).sum()), int(base_detect.sum())
                    ),
                    "redundant_frac_of_both_detect": ratio(
                        int((subset & masks["redundant"]).sum()), int(subset_both.sum())
                    ),
                    "median_lead_gain_chunks": (
                        float(np.median(lead_gain)) if lead_gain.size else float("nan")
                    ),
                    "median_lead_gain_steps": (
                        float(np.median(lead_gain)) * STEPS_PER_CHUNK
                        if lead_gain.size else float("nan")
                    ),
                }
                if subset_name == "all_risk":
                    row.update(
                        {
                            "timely_episodes": int(timely.sum()),
                            "base_timely_alarms": int((timely & base_hit).sum()),
                            "candidate_timely_alarms": int(
                                (timely & candidate_hit).sum()
                            ),
                            "added_false_alarms": int(added.sum()),
                            "added_false_alarms_frac_timely": ratio(
                                int(added.sum()), int(timely.sum())
                            ),
                            "added_false_alarms_frac_base_silent_timely": ratio(
                                int(added.sum()), int(base_silent_timely.sum())
                            ),
                            "added_false_alarms_ci_low": added_low,
                            "added_false_alarms_ci_high": added_high,
                        }
                    )
                else:
                    row.update(
                        {
                            "timely_episodes": np.nan,
                            "base_timely_alarms": np.nan,
                            "candidate_timely_alarms": np.nan,
                            "added_false_alarms": np.nan,
                            "added_false_alarms_frac_timely": np.nan,
                            "added_false_alarms_frac_base_silent_timely": np.nan,
                            "added_false_alarms_ci_low": np.nan,
                            "added_false_alarms_ci_high": np.nan,
                        }
                    )
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    v4_alarms, v4_anchor = build_v4_alarms()
    print("v4 rebuild reproduces every published external count and v4's own "
          "sealed development arrays", flush=True)
    progress_alarms, progress_anchor = build_progress_ratio_alarms()
    print(
        f"progress ratio rebuilt at W={PR_WINDOW} group={PR_GROUP} "
        f"K={PR_CONFIRMATIONS} r*={progress_anchor['ratio_threshold']:.16f} "
        f"ell*={progress_anchor['ell_star']:.16f} "
        f"eps={progress_anchor['eps_length']:.16f}",
        flush=True,
    )
    v7 = load_npz(V7_SEALED)

    rng = np.random.default_rng(args.seed)
    episode_frames: list[pd.DataFrame] = []
    pair_records: list[dict[str, Any]] = []
    cohort_summary: dict[str, Any] = {}

    for cohort, cache_stem, mobility_stem, run_id, label_stem, v7_prefix in COHORTS:
        cache = load_npz(CACHE_ROOT / f"{cache_stem}.npz")
        mobility = load_npz(MOBILITY_ROOT / f"{mobility_stem}.npz")
        labels = attach_failure_modes(
            aligned_labels(cache, LABEL_ROOT / f"{label_stem}_clean_labels.csv", cohort),
            cache, mobility, run_id, cohort,
        )
        alarms = {
            "v4": v4_alarms[cohort],
            "v7_freeze": v7[f"{v7_prefix}_freeze"],
            "v7_turbulence": v7[f"{v7_prefix}_turbulence"],
            "v7_guard": v7[f"{v7_prefix}_guard"],
            "progress_ratio": progress_alarms[cohort],
        }
        alarms["v7_guard_or_progress_ratio"] = first_or(
            alarms["v7_guard"], alarms["progress_ratio"]
        )
        if not np.array_equal(
            first_or(alarms["v7_freeze"], alarms["v7_turbulence"]), alarms["v7_guard"]
        ):
            raise ValueError(f"{cohort}: v7_guard is not freeze OR turbulence")

        frame, diagnostics = cohort_frame(cohort, labels, alarms)
        episode_frames.append(frame)

        n_tasks = int(frame["task"].nunique())
        draws_index = rng.integers(0, n_tasks, size=(args.bootstrap, n_tasks))
        pair_records.extend(pair_rows(cohort, frame, draws_index))

        cohort_summary[cohort] = {
            "episodes": int(len(frame)),
            "tasks": n_tasks,
            "suites": sorted(frame["suite"].unique().tolist()),
            "label_counts": {
                name: int(frame[name].to_numpy(bool).sum())
                for name in ("risk", "timely", "loop", "static", "phantom",
                             "pure_phantom")
            },
            "label_counts_match_expected": True,
            "bootstrap_tasks": n_tasks,
            **diagnostics,
        }
        counts = cohort_summary[cohort]["label_counts"]
        print(
            f"{cohort}: episodes={len(frame)} tasks={n_tasks} risk={counts['risk']} "
            f"loop={counts['loop']} static={counts['static']} "
            f"phantom={counts['phantom']} pure_phantom={counts['pure_phantom']}",
            flush=True,
        )
        for channel in CHANNELS:
            block = diagnostics["channels"][channel]
            print(
                f"  {channel:<28} alarms={block['alarms']:>5} tp={block['tp']:>4} "
                f"fp={block['fp']:>4} recall={block['risk_recall']:.3f} "
                f"fpr={block['timely_fpr']:.5f} "
                f"median trigger chunk (risk)={block['on_risk'].get('chunk_median')} "
                f"= {block['on_risk'].get('steps_executed_median')} env steps",
                flush=True,
            )

    # progress-ratio development anchor, now that the failure modes exist
    development = episode_frames[0]
    fired = development["progress_ratio_chunk"].to_numpy() >= 0
    progress_anchor["development_alarm_episodes"] = int(fired.sum())
    progress_anchor["development_loop_tp"] = int(
        (fired & development["loop"].to_numpy(bool)).sum()
    )
    progress_anchor["development_loop_tp_expected"] = PR_DEV_LOOP_TP
    progress_anchor["anchor_matches_frozen_selection"] = bool(
        progress_anchor["development_alarm_episodes"] == PR_DEV_ALARM_EPISODES
        and progress_anchor["development_loop_tp"] == PR_DEV_LOOP_TP
    )
    print(
        f"progress-ratio development anchor: alarm_episodes="
        f"{progress_anchor['development_alarm_episodes']} "
        f"(frozen selection.json records {PR_DEV_ALARM_EPISODES}), loop_tp="
        f"{progress_anchor['development_loop_tp']} (expected {PR_DEV_LOOP_TP})",
        flush=True,
    )

    episodes = pd.concat(episode_frames, ignore_index=True)
    episodes_path = args.output / "episodes.csv"
    episodes.to_csv(episodes_path, index=False)

    pairs = pd.DataFrame(pair_records)
    pairs_path = args.output / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    all_risk = pairs[pairs["subset"] == "all_risk"]
    overlapping = all_risk[all_risk["both_detect"] > 0]
    independent = overlapping[~overlapping["structurally_nested"]]
    mostly_redundant = overlapping[
        overlapping["redundant_frac_of_both_detect"] >= 0.5
    ]
    independent_mostly_redundant = mostly_redundant[
        ~mostly_redundant["structurally_nested"]
    ]
    elapsed = time.perf_counter() - started
    summary = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claim_type": (
            "descriptive per-episode ledger; nothing is fused, fitted or selected"
        ),
        "notes": [
            "external_8b is NOT a pristine holdout: it has been examined across "
            "v3-v7 and the HB front/back bundle. Its numbers are a consistency "
            "check on an already-seen cohort, not out-of-sample confirmation.",
            "The progress-ratio channel failed its pre-registered gate on "
            "development (19 low-prior loop TP against a floor of 20). It is "
            "carried here as a candidate channel, not as a validated detector.",
            "v7_guard is by construction first_or(v7_freeze, v7_turbulence), so "
            "the three v7 rows are not independent detectors and their pairwise "
            "buckets are structurally determined, not empirical findings.",
            "v7_guard_or_progress_ratio is likewise a construction, not a fusion: "
            "it is the elementwise earlier-of the two channels, included so the "
            "cost of the union is visible next to its coverage.",
            "Survival priors are estimated in-cohort, per suite, as "
            "P(risk | still running at chunk q).",
            "Bucket boundaries are pre-registered: a gain of exactly 2 chunks is "
            "'earlier', a gain of exactly 1 chunk is 'redundant'.",
        ],
        "definitions": {
            "candidate_only": "candidate detects, base misses (risk episodes)",
            "earlier": (
                f"both detect and the candidate's first alarm is at least "
                f"{LEAD_CHUNKS} chunks earlier"
            ),
            "redundant": (
                f"both detect and the first alarms are within {LEAD_CHUNKS - 1} "
                "chunk of each other"
            ),
            "later": (
                f"both detect and the candidate is at least {LEAD_CHUNKS} chunks late"
            ),
            "added_false_alarms": (
                "candidate fires and base does not, on timely successes"
            ),
            "lead_chunks": LEAD_CHUNKS,
        },
        "steps_per_chunk": {
            "value": STEPS_PER_CHUNK,
            "recorded": True,
            "source": STEPS_PER_CHUNK_SOURCE,
            "convention": (
                "environment steps already executed at trigger = first-alarm chunk "
                "index x steps per chunk; the chunk that produced the alarm is not "
                "counted as executed"
            ),
        },
        "channels": list(CHANNELS),
        "anchors": {"v4": v4_anchor, "progress_ratio": progress_anchor},
        "cohorts": cohort_summary,
        "bootstrap": {
            "kind": "task-clustered percentile bootstrap",
            "draws": int(args.bootstrap),
            "seed": int(args.seed),
            "unit": "task",
            "quantities": ["candidate_only", "added_false_alarms"],
            "reference": (
                "moe-v7-0905/experiments/evaluate_intrinsic_guard_v7.py::"
                "clustered_interval"
            ),
            "shared_resample_within_cohort": True,
        },
        "headline": {
            "ordered_pairs_per_cohort": int(len(CHANNELS) * (len(CHANNELS) - 1)),
            "pairs_with_any_both_detect": int(len(overlapping)),
            "pairs_mostly_redundant": int(len(mostly_redundant)),
            "structurally_independent_pairs_with_overlap": int(len(independent)),
            "structurally_independent_pairs_mostly_redundant": int(
                len(independent_mostly_redundant)
            ),
            "mostly_redundant_definition": (
                "redundant >= 50% of the episodes both channels detect, on risk "
                "episodes, all_risk subset"
            ),
            "structurally_independent_mostly_redundant_pairs": [
                f"{row.cohort}: base={row.base} candidate={row.candidate} "
                f"({row.redundant_frac_of_both_detect:.0%} of {row.both_detect} "
                "both-detect)"
                for row in independent_mostly_redundant.itertuples()
            ],
            "nested_pairs_note": (
                "the remaining mostly-redundant rows are nesting relations "
                "(v7_guard over its own branches, v7_guard_or_progress_ratio over "
                "everything it unions); their redundancy is arithmetic"
            ),
        },
        "artifacts": {
            "episodes_csv_sha256": sha256(episodes_path),
            "pairs_csv_sha256": sha256(pairs_path),
            "script_sha256": sha256(Path(__file__)),
            "v7_sealed_sha256": sha256(V7_SEALED),
            "v4_sealed_sha256": sha256(V4_SEALED),
        },
        "seconds": round(elapsed, 1),
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        f"\nepisodes.csv rows={len(episodes)}  pairs.csv rows={len(pairs)}\n"
        f"mostly redundant: {len(mostly_redundant)}/{len(overlapping)} ordered "
        f"pairs with any both-detect overlap; restricted to structurally "
        f"independent pairs, {len(independent_mostly_redundant)}/{len(independent)}"
    )
    print(f"elapsed {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
