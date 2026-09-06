#!/usr/bin/env python3
"""Shared assembly for the operational replay ledger.

Every rollout is a sequence of chunks (policy calls). A detector at chunk q sees
chunks 0..q only. The first-alarm arrays stored across the bundles are the output
of exactly that causal replay; ``verify_replay.py`` checks the claim rather than
assuming it.

This module does three things and nothing else:

    1. build the episode index (task, suite, length, horizon cap, outcome,
       physical failure mode) and assert its row alignment against every source;
    2. build the per-suite survival prior P(risk | still running at q);
    3. collect first-alarm arrays for every method, rebuilding the ones that are
       not on disk, and check each against the counts its bundle published.

Nothing derived from an outcome label enters the "knowable at alarm time" block,
with one declared exception: the survival prior is itself an outcome-derived base
rate.  It is carried in the knowable block because that is what the whole corpus
means by "what still running alone would have told you"; it is a per-suite,
per-chunk constant, never an episode-level quantity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

V4 = PROJECT / "moe-v4-0904"
HB = PROJECT / "moe-hb-front-back-0905"
V7 = PROJECT / "moe-v7-0905"
FLOW = PROJECT / "moe-flow-semantics-0906"
CIRCUIT = PROJECT / "moe-circuit-analogy-0906"
TOKGEO = PROJECT / "moe-token-geometry-0906"
STATECH = PROJECT / "moe-state-channel-0906"
COMBO = PROJECT / "moe-combination-rules-0906"
TWOTIER = PROJECT / "moe-two-tier-0906"

LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
PHYSICAL = PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"

sys.path.insert(0, str(HB / "experiments"))
sys.path.insert(0, str(V4 / "experiments"))
sys.path.insert(0, str(COMBO / "experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import (  # noqa: E402
    MOBILITY_PATHS,
    PROFILE_ROOT,
    calibrated_external_alarm,
    load_npz,
    mobility_representations,
    profile_task_map,
)

WIDTH, CONFIRMATIONS = 4, 4
LOW_PRIOR = 0.25

# Horizon caps, declared by the deployment protocol. Every bundle that names them
# names these four.
HORIZON = {
    "libero_goal": 30,
    "libero_long": 52,
    "libero_object": 28,
    "libero_spatial": 22,
}
# Chunk at which the survival prior first reaches 0.25, published identically in
# moe-two-tier-0906/results/manifest.json and moe-hb-front-back-0905.
PRIOR_CROSS_25 = {
    "libero_goal": 18,
    "libero_long": 26,
    "libero_object": 17,
    "libero_spatial": 13,
}

COHORTS = {
    "external_8b": {
        "labels": LABEL_ROOT / "external_8b_clean_labels.csv",
        "mobility": MOBILITY_PATHS["external_8b"],
        "graph": PROFILE_ROOT / "external_8b.npz",
        "physical_run": "right-50x8b-20260903",
        "episodes": 15600,
        "risks": 564,
    },
    "development_main": {
        "labels": LABEL_ROOT / "development_main_clean_labels.csv",
        "mobility": MOBILITY_PATHS["development_main"],
        "graph": PROFILE_ROOT / "development_main.npz",
        "physical_run": "right-50x8-20260903",
        "episodes": 14800,
        "risks": 487,
    },
}

MODE_SHORT = {
    "object_released_or_dropped_before_goal": "dropped",
    "stable_grasp_not_observed": "no_grasp",
    "object_moved_but_goal_unmet": "moved_unmet",
    "goal_predicate_regressed": "regressed",
    "object_released_outside_goal": "released_outside",
    "approached_target_without_observed_contact": "no_contact",
    "timeout_while_holding_target": "timeout_holding",
    "mechanism_threshold_not_reached": "mechanism",
    "no_meaningful_target_progress": "no_progress",
}


# --------------------------------------------------------------------------- #
# index
# --------------------------------------------------------------------------- #
class Cohort:
    """Episode index, survival prior and alignment assertions for one cohort."""

    def __init__(self, name: str) -> None:
        spec = COHORTS[name]
        self.name = name
        self.mobility_cache = load_npz(spec["mobility"])
        self.graph_cache = load_npz(spec["graph"])
        labels = pd.read_csv(spec["labels"])

        cache = self.mobility_cache
        task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]

        # --- row alignment, asserted rather than assumed ----------------------
        assert len(labels) == len(task) == spec["episodes"], (
            f"{name}: episode count mismatch"
        )
        assert np.array_equal(labels["task"].astype(str).to_numpy(), task), (
            f"{name}: label task order != mobility cache order"
        )
        assert np.array_equal(
            labels["episode"].to_numpy(int), cache["episode"].astype(int)
        ), f"{name}: label episode order != mobility cache order"
        assert np.array_equal(
            self.graph_cache["episode"], cache["episode"]
        ), f"{name}: layer-graph cache != mobility cache"
        assert np.array_equal(
            self.graph_cache["task_index"], cache["task_index"]
        ), f"{name}: layer-graph task_index != mobility cache"
        assert np.array_equal(
            labels["episode_length"].to_numpy(int), cache["length"].astype(int)
        ), f"{name}: label episode_length != cache length"
        assert np.array_equal(
            cache["valid"].sum(axis=1), cache["length"].astype(int)
        ), f"{name}: valid mask does not agree with length"

        self.task = task
        self.suite = np.asarray([t.split("/", 1)[0] for t in task])
        self.task_short = np.asarray([t.split("/", 1)[1] for t in task])
        self.episode = cache["episode"].astype(int)
        self.length = cache["length"].astype(int)
        self.valid = cache["valid"].astype(bool)
        self.queries = int(self.valid.shape[1])
        self.n = len(task)

        self.risk = labels["original_failure"].to_numpy(bool)
        self.persistent = labels["failure"].to_numpy(bool)
        self.late = labels["late_success_plus10_queries"].to_numpy(bool)
        assert int(self.risk.sum()) == spec["risks"], f"{name}: risk count mismatch"
        assert not (self.persistent & self.late).any(), "persistent and late overlap"
        assert np.array_equal(self.risk, self.persistent | self.late)

        self.outcome = np.where(
            self.persistent,
            "persistent_failure",
            np.where(self.late, "late_success_plus10", "timely_success"),
        )
        self.horizon = np.asarray([HORIZON[s] for s in self.suite], dtype=int)
        assert (self.length <= self.horizon).all(), f"{name}: length exceeds horizon"

        # risk means "did not finish before the horizon cap". Every risk sits
        # exactly at the cap; a few episodes reach the cap and still succeed on
        # their final chunk, so the implication runs one way only.
        reached_cap = self.length >= self.horizon
        assert (self.length[self.risk] == self.horizon[self.risk]).all(), (
            f"{name}: a risk episode did not reach its horizon cap"
        )
        self.cap_but_success = int((reached_cap & ~self.risk).sum())

        self.priors = self._survival_prior()
        suites = sorted(self.priors)
        self.suite_id = np.searchsorted(np.asarray(suites), self.suite)
        table = np.full((len(suites), self.queries), np.nan)
        for position, suite in enumerate(suites):
            for chunk, value in self.priors[suite].items():
                if chunk < self.queries:
                    table[position, chunk] = value
        self.prior_table = table
        self.suites = suites

        self.physical_mode = self._physical_modes(spec["physical_run"])

    def _survival_prior(self) -> dict[str, dict[int, float]]:
        priors: dict[str, dict[int, float]] = {}
        for suite in np.unique(self.suite):
            take = self.suite == suite
            priors[str(suite)] = {
                chunk: float(self.risk[take][self.length[take] > chunk].mean())
                for chunk in range(int(self.length[take].max()))
            }
        return priors

    def prior_of(self, first: np.ndarray) -> np.ndarray:
        out = np.full(len(first), np.nan)
        fired = first >= 0
        out[fired] = self.prior_table[self.suite_id[fired], first[fired]]
        return out

    def _physical_modes(self, run_id: str) -> np.ndarray:
        table = pd.read_csv(PHYSICAL)
        table = table[table["run_id"] == run_id]
        assert (table["physics_validation_status"] != "failed").all(), (
            "physics_validation did not pass for every row"
        )
        table = table.assign(
            key=table["task_name"].astype(str)
            + "||"
            + table["episode_index"].astype(int).astype(str)
        )
        lookup = dict(
            zip(table["key"], table["primary_failure_reason"].fillna(""), strict=True)
        )
        conf = dict(
            zip(table["key"], table["failure_reason_confidence"].fillna(""), strict=True)
        )
        keys = [
            f"{t}||{e}" for t, e in zip(self.task_short, self.episode, strict=True)
        ]
        missing = [k for k in keys if k not in lookup]
        assert not missing, f"{len(missing)} episodes absent from the physical table"
        modes = np.asarray(
            [MODE_SHORT.get(lookup[k], "") if lookup[k] else "" for k in keys]
        )
        self.mode_confidence = np.asarray([conf[k] for k in keys])
        # every risk carries a mode; no timely success does
        assert (modes[self.risk] != "").all(), "a risk episode has no physical mode"
        assert (modes[~self.risk] == "").all(), "a timely success carries a mode"
        return modes

    def index_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "row": np.arange(self.n),
                "suite": self.suite,
                "task": self.task,
                "episode": self.episode,
                "horizon": self.horizon,
                "length": self.length,
                "risk": self.risk,
                "outcome": self.outcome,
                "physical_mode": self.physical_mode,
                "mode_confidence": self.mode_confidence,
            }
        )


# --------------------------------------------------------------------------- #
# alarm construction primitives
# --------------------------------------------------------------------------- #
def oriented(values: np.ndarray, direction: str, width: int = WIDTH) -> np.ndarray:
    smoothed = dev.trailing_mean(values, width)
    return -smoothed if direction == "low" else smoothed


def quantity_values(
    graph: dict[str, np.ndarray], mobility: dict[str, np.ndarray], quantity: str
) -> dict[str, np.ndarray]:
    """Reproduce survey_reference_frames.quantity_cache."""
    if quantity == "mobility":
        values = np.asarray(mobility["mobility"], dtype=np.float32)
    else:
        names = graph["metric_names"].astype(str).tolist()
        values = np.asarray(
            graph["metrics"][:, :, :, names.index(quantity)], dtype=np.float32
        )
    return {
        "mobility": values,
        "valid": graph["valid"].astype(bool),
        "layer_names": graph["layer_names"],
        "task_names": graph["task_names"],
        "task_index": graph["task_index"],
        "episode": graph["episode"],
        "init_state_id": graph["init_state_id"],
        "length": graph["length"],
    }


def or_alarm(*arrays: np.ndarray) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    masked = np.where(stack >= 0, stack, np.iinfo(np.int16).max)
    best = masked.min(axis=0)
    return np.where(best == np.iinfo(np.int16).max, -1, best).astype(np.int16)


def and_alarm(*arrays: np.ndarray) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    both = (stack >= 0).all(axis=0)
    out = np.full(stack.shape[1], -1, dtype=np.int16)
    out[both] = stack[:, both].max(axis=0)
    return out


def k_of_n_alarm(arrays: list[np.ndarray], k: int) -> np.ndarray:
    """Latching k-of-n: alarm at the chunk where the k-th member has voted."""
    stack = np.stack(arrays, axis=0)
    masked = np.where(stack >= 0, stack, np.iinfo(np.int16).max).astype(np.int32)
    masked.sort(axis=0)
    kth = masked[k - 1]
    return np.where(kth == np.iinfo(np.int16).max, -1, kth).astype(np.int16)


def counts(first: np.ndarray, risk: np.ndarray) -> tuple[int, int]:
    fired = first >= 0
    return int((fired & risk).sum()), int((fired & ~risk).sum())


# --------------------------------------------------------------------------- #
# method registry
# --------------------------------------------------------------------------- #
class Method:
    def __init__(
        self,
        name: str,
        family: str,
        threshold_mode: str,
        runtime_task_identity: bool,
        source: str,
        config: str,
        first: np.ndarray,
        published: tuple[int, int] | None = None,
    ) -> None:
        self.name = name
        self.family = family
        self.threshold_mode = threshold_mode
        self.runtime_task_identity = runtime_task_identity
        self.source = source
        self.config = config
        self.first = np.asarray(first, dtype=np.int16)
        self.published = published


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {k: np.asarray(archive[k]) for k in archive.files}


FRAME_SURVEY_QUANTITIES = (
    "mobility",
    "conditional_query_d1",
    "partial_query_d1",
    "flow_path",
    "flow_endpoint",
    "flow_settling_log_ratio",
    "action_consensus",
    "state_action_alignment",
    "conditional_energy",
    "conditional_effective_rank",
    "partial_edge_std",
    "expert_load_effective_rank",
)
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


def build_layer_rules(cohort_name: str) -> dict[str, np.ndarray]:
    """Rebuild front/back/all lock, L5 switching and the v4 dual regime.

    Follows moe-hb-front-back-0905/experiments/evaluate_layer_survival_baseline.py
    exactly; for the development cohort the same heads are rebuilt with the
    crossfit per-task thresholds that v4 uses on its own calibration split.
    """
    LOCK = ("low", 4, 4, 0.75)
    INSTAB = ("high", 4, 8, 0.80)
    main = load_npz(MOBILITY_PATHS["development_main"])
    extra = load_npz(MOBILITY_PATHS["development_extra"])
    reference_map = profile_task_map([main, extra])
    reference_representations = {
        id(main): mobility_representations(main),
        id(extra): mobility_representations(extra),
    }
    if cohort_name == "external_8b":
        target = load_npz(MOBILITY_PATHS["external_8b"])
        target_repr = mobility_representations(target)

        def alarm(rep: str, head: tuple[str, int, int, float]) -> np.ndarray:
            direction, width, confirmations, quantile = head
            return calibrated_external_alarm(
                target,
                target_repr[rep],
                reference_map,
                reference_representations,
                rep,
                direction,
                width,
                confirmations,
                quantile,
            )

    else:
        target = main
        target_repr = reference_representations[id(main)]
        task_index = target["task_index"].astype(int)
        init = target["init_state_id"].astype(int)
        valid = target["valid"].astype(bool)

        def alarm(rep: str, head: tuple[str, int, int, float]) -> np.ndarray:
            direction, width, confirmations, quantile = head
            values = oriented(target_repr[rep], direction, width)
            thresholds = dev.crossfit_thresholds(
                dev.row_max(values), task_index, init
            )[:, dev.QUANTILES.index(quantile)]
            persistent = dev.persistent_score(values, confirmations)
            trigger = (
                np.isfinite(persistent) & (persistent > thresholds[:, None]) & valid
            )
            return dev.first_query(trigger)

    front = alarm("front_median", LOCK)
    back = alarm("back_median", LOCK)
    every = alarm("all_median", LOCK)
    switching = alarm("L5", INSTAB)
    return {
        "front_lock": front,
        "back_lock": back,
        "all_lock": every,
        "L5_switching": switching,
        "v4_dual_regime": or_alarm(every, switching),
    }


def rebuild_frame_survey(cohort: Cohort, quantity: str, mode: str, spec: dict) -> np.ndarray:
    """Rebuild one frame-survey head from the raw per-chunk score arrays."""
    graphs = {
        n: load_npz(PROFILE_ROOT / f"{n}.npz")
        for n in ("development_main", "development_extra")
    }
    mobs = {n: load_npz(MOBILITY_PATHS[n]) for n in graphs}
    graphs[cohort.name] = cohort.graph_cache
    mobs[cohort.name] = cohort.mobility_cache
    reprs = {
        n: {k: v for k, (v, _) in dev.representations(
            quantity_values(graphs[n], mobs[n], quantity)
        ).items()}
        for n in graphs
    }
    rep, direction, quantile = spec["representation"], spec["direction"], spec["quantile"]
    target = reprs[cohort.name][rep]
    persistent = dev.persistent_score(oriented(target, direction), CONFIRMATIONS)
    first = np.full(len(persistent), -1, dtype=np.int16)
    if mode == "global":
        pooled = np.concatenate(
            [
                dev.row_max(oriented(reprs[n][rep], direction))
                for n in ("development_main", "development_extra")
            ]
        )
        line = dev.quantile_higher(pooled, quantile)
        first = dev.first_query(
            np.isfinite(persistent) & (persistent > line) & cohort.valid
        )
    else:
        ref_task = {
            n: graphs[n]["task_names"].astype(str)[graphs[n]["task_index"].astype(int)]
            for n in ("development_main", "development_extra")
        }
        for task in np.unique(cohort.task):
            take = np.flatnonzero(cohort.task == task)
            peaks = [
                dev.row_max(oriented(reprs[n][rep][ref_task[n] == task], direction))
                for n in ("development_main", "development_extra")
                if (ref_task[n] == task).any()
            ]
            line = dev.quantile_higher(np.concatenate(peaks), quantile)
            first[take] = dev.first_query(
                np.isfinite(persistent[take])
                & (persistent[take] > line)
                & cohort.valid[take]
            )
    return first
