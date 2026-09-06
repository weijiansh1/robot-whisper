#!/usr/bin/env python3
"""Shared machinery for the two-tier (WATCH / ACT) monitor.

Everything here is a thin wrapper over the frozen v4 / hb-front-back code so the
two-tier study inherits, byte for byte, the same representations, the same
causal trailing mean, the same consecutive-confirmation logic, the same
cross-fitted per-task and pooled global thresholds, and the same survival-prior
scoring. Nothing is refitted.

Scores read routing tensors alone (hb_router_probs -> layer graphs -> the twelve
quantities). Physical failure-mode labels never enter a runtime decision; they
are used for evaluation, and -- only in the explicitly flagged WATCH variant
(b) -- for rule selection on the development split.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

sys.path.insert(0, str(PROJECT / "moe-hb-front-back-0905/experiments"))
sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import (  # noqa: E402
    LABEL_PATHS,
    MOBILITY_PATHS,
    PROFILE_ROOT,
    load_npz,
)
from compare_layers_early import CONFIRMATIONS, WIDTH  # noqa: E402
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)
from survey_reference_frames import FRAMES, quantity_cache  # noqa: E402


RESULTS = BUNDLE / "results"
FRAME_SURVEY = PROJECT / "moe-hb-front-back-0905/results/frame_survey"
EXTERNAL_ALARMS = FRAME_SURVEY / "external_first_alarms.npz"
EXTERNAL_DETECTORS = FRAME_SURVEY / "external_detectors.csv"
DEVELOPMENT_CANDIDATES = FRAME_SURVEY / "development_candidates.csv"
V7_ALARMS = PROJECT / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
PHYSICAL_LABELS = (
    PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"
)

RUN_IDS = {
    "development_main": "right-50x8-20260903",
    "external_8b": "right-50x8b-20260903",
}

MODES = ("per_task", "global")

# Frame of the v7 intrinsic guard. Its four components (freeze, acceleration,
# periodicity, turbulence) are all functions of how the routing distribution at
# query q compares with query q-1..q-k of the same episode, i.e. they treat the
# episode's own recent past as normal. That is the adjacent_query frame. It is
# declared here, before it is used, and it is *not* an independent frame.
V7_FRAME = "adjacent_query"

# Physical failure modes with at least this many risk episodes in a split enter
# the worst-mode / CV statistics. Declared before any rule was scored. Modes
# below the floor are still reported with their n, but a coverage rate over
# n < 10 is noise and would let a single episode dictate the objective.
MODE_FLOOR = 10


# --------------------------------------------------------------------------
# cohorts
# --------------------------------------------------------------------------
def load_cohort(name: str) -> dict[str, Any]:
    """Graph cache + mobility cache + labels + suite/prior scaffolding."""
    graph = load_npz(PROFILE_ROOT / f"{name}.npz")
    mobility = load_npz(MOBILITY_PATHS[name])
    if not np.array_equal(graph["episode"], mobility["episode"]):
        raise ValueError(f"{name}: graph and mobility caches are not aligned")
    out: dict[str, Any] = {
        "name": name,
        "graph": graph,
        "mobility": mobility,
        "valid": graph["valid"].astype(bool),
        "length": graph["length"].astype(int),
        "task_index": graph["task_index"].astype(int),
        "init_state_id": graph["init_state_id"].astype(int),
        "task": graph["task_names"].astype(str)[graph["task_index"].astype(int)],
        "suite": suite_of(graph),
    }
    if name in LABEL_PATHS:
        labels = pd.read_csv(LABEL_PATHS[name])
        if len(labels) != len(graph["episode"]):
            raise ValueError(f"{name}: labels are not row aligned with the cache")
        if not np.array_equal(
            labels["episode"].to_numpy(int), graph["episode"].astype(int)
        ):
            raise ValueError(f"{name}: label episode ids differ from the cache")
        out["labels"] = labels
        out["risk"] = labels["original_failure"].to_numpy(bool)
        out["priors"] = survival_prior(out["suite"], out["length"], out["risk"])
        out["prior_curve"] = out["priors"]
    return out


def physical_modes(cohort: dict[str, Any]) -> np.ndarray:
    """Per-episode primary physical failure reason, '' where none."""
    episodes = pd.read_csv(PHYSICAL_LABELS)
    episodes = episodes[episodes["run_id"] == RUN_IDS[cohort["name"]]]
    key = pd.DataFrame(
        {
            "task_name": [t.split("/", 1)[1] for t in cohort["task"]],
            "episode": cohort["graph"]["episode"].astype(int),
        }
    )
    merged = key.merge(
        episodes[["task_name", "episode_index", "primary_failure_reason"]],
        left_on=["task_name", "episode"],
        right_on=["task_name", "episode_index"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(key):
        raise ValueError("physical-mode join changed row count")
    if merged["episode_index"].isna().any():
        raise ValueError("physical-mode join left unmatched episodes")
    return merged["primary_failure_reason"].fillna("").to_numpy(str)


def mode_counts(mode: np.ndarray, risk: np.ndarray) -> dict[str, int]:
    names, counts = np.unique(mode[risk], return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts, strict=True) if n}


def scored_modes(counts: dict[str, int]) -> list[str]:
    return sorted(name for name, n in counts.items() if n >= MODE_FLOOR)


# --------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------
def oriented(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = dev.trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed


def head_alarms(
    cohorts: dict[str, dict[str, Any]],
    target: str,
    quantity: str,
    representation: str,
    direction: str,
    quantile: float,
    mode: str,
) -> np.ndarray:
    """First-alarm chunk per episode for one head, rebuilt from routing tensors.

    Thresholds always come from the development corpus (main + extra), never
    from the cohort being scored, unless that cohort *is* development -- in
    which case per_task uses v4's leave-one-initial-state-out cross fit and
    global uses the pooled development quantile, exactly as the frame survey
    did.
    """
    reprs: dict[str, np.ndarray] = {}
    for name in ("development_main", "development_extra", target):
        cache = quantity_cache(
            cohorts[name]["graph"], cohorts[name]["mobility"], quantity
        )
        reprs[name] = {k: v for k, (v, _) in dev.representations(cache).items()}[
            representation
        ]

    persistent = dev.persistent_score(oriented(reprs[target], direction), CONFIRMATIONS)
    valid = cohorts[target]["valid"]
    first = np.full(len(persistent), -1, dtype=np.int16)

    if mode == "global":
        pooled = np.concatenate(
            [
                dev.row_max(oriented(reprs[name], direction))
                for name in ("development_main", "development_extra")
            ]
        )
        line = dev.quantile_higher(pooled, quantile)
        return dev.first_query(np.isfinite(persistent) & (persistent > line) & valid)

    if target == "development_main":
        crossfit = dev.crossfit_thresholds(
            dev.row_max(oriented(reprs[target], direction)),
            cohorts[target]["task_index"],
            cohorts[target]["init_state_id"],
        )
        position = list(dev.QUANTILES).index(quantile)
        return dev.first_query(
            np.isfinite(persistent)
            & (persistent > crossfit[:, position][:, None])
            & valid
        )

    reference_task = {
        name: cohorts[name]["task"] for name in ("development_main", "development_extra")
    }
    target_task = cohorts[target]["task"]
    for task in np.unique(target_task):
        take = np.flatnonzero(target_task == task)
        peaks = [
            dev.row_max(oriented(reprs[name][reference_task[name] == task], direction))
            for name in ("development_main", "development_extra")
            if (reference_task[name] == task).any()
        ]
        line = dev.quantile_higher(np.concatenate(peaks), quantile)
        first[take] = dev.first_query(
            np.isfinite(persistent[take]) & (persistent[take] > line) & valid[take]
        )
    return first


# --------------------------------------------------------------------------
# causal combination
# --------------------------------------------------------------------------
def combine(alarms: Sequence[np.ndarray], k: int) -> np.ndarray:
    """k-of-n vote. The combined alarm fires when the k-th component fires.

    Causal by construction: the k-th smallest component alarm chunk is the
    earliest query q at which k components have already fired at chunks <= q.
    k = 1 is OR (min); k = n is AND (max over a full set).
    """
    if k < 1 or k > len(alarms):
        raise ValueError(f"k={k} out of range for {len(alarms)} components")
    stacked = np.stack([np.asarray(a, dtype=np.int32) for a in alarms])
    fired = stacked >= 0
    if fired.shape[0] != len(alarms):
        raise ValueError("stack shape mismatch")
    votes = fired.sum(axis=0)
    filled = np.where(fired, stacked, np.iinfo(np.int32).max)
    kth = np.sort(filled, axis=0)[k - 1]
    return np.where(votes >= k, kth, -1).astype(np.int16)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def evaluate(
    first: np.ndarray,
    cohort: dict[str, Any],
    mode_labels: np.ndarray | None = None,
    scored: Sequence[str] | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """TP/FP/precision/lift/early band + physical-mode coverage for one rule."""
    risk = cohort["risk"]
    prior = prior_of(first, cohort["suite"], cohort["priors"])
    record: dict[str, Any] = dict(score_candidate(first, risk, prior))
    record["n_alarm"] = int((first >= 0).sum())
    record["median_alarm_chunk"] = (
        float(np.median(first[first >= 0])) if record["n_alarm"] else float("nan")
    )
    if mode_labels is None:
        return record
    if scored is None or counts is None:
        raise ValueError("mode coverage needs both the scored list and the counts")
    fired = first >= 0
    rates = []
    for name in scored:
        take = risk & (mode_labels == name)
        n = int(take.sum())
        hit = int((take & fired).sum())
        record[f"mode__{name}__n"] = n
        record[f"mode__{name}__hit"] = hit
        record[f"mode__{name}__coverage"] = hit / n if n else float("nan")
        rates.append(hit / n if n else np.nan)
    rates = np.asarray(rates, dtype=float)
    record["worst_mode_coverage"] = float(np.nanmin(rates))
    record["mean_mode_coverage"] = float(np.nanmean(rates))
    record["mode_cv"] = float(np.nanstd(rates) / np.nanmean(rates))
    record["n_modes_scored"] = int(np.isfinite(rates).sum())
    return record


def suite_breakdown(first: np.ndarray, cohort: dict[str, Any]) -> dict[str, Any]:
    prior = prior_of(first, cohort["suite"], cohort["priors"])
    out: dict[str, Any] = {}
    for name in np.unique(cohort["suite"]):
        take = cohort["suite"] == name
        block = score_candidate(first[take], cohort["risk"][take], prior[take])
        for key in ("tp", "fp", "low_prior_tp", "low_prior_fp", "precision", "lift"):
            out[f"{name}__{key}"] = block[key]
    return out


def plain(value: Any) -> Any:
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def head_frame(key: str) -> str:
    quantity = key.split("|", 1)[0]
    if quantity == "v7_guard":
        return V7_FRAME
    return FRAMES[quantity]


def frames_of(keys: Iterable[str]) -> set[str]:
    return {head_frame(k) for k in keys}
