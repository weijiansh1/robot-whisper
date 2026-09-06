#!/usr/bin/env python3
"""Shared plumbing: reuse the published survival/lift protocol verbatim.

Nothing here is re-implemented.  `evaluate_layerwise_alarm_development` (v4)
supplies trailing_mean / persistent_score / row_max / quantile_higher /
first_query / crossfit_thresholds / representations / QUANTILES, and
`select_early_lock` (hb-front-back-0905) supplies survival_prior / prior_of /
score_candidate / LOW_PRIOR / MAX_TIMELY_FPR.  Importing them keeps this
bundle's numbers directly comparable with the published baselines.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))
sys.path.insert(0, str(PROJECT / "moe-hb-front-back-0905/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from compare_layers_early import CONFIRMATIONS, MIN_LOW_PRIOR_PRECISION, WIDTH  # noqa: E402
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)

FEATURE_ROOT = BUNDLE / "results/circuit_features"
GRAPH_ROOT = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
COHORTS = ("development_main", "development_extra", "external_8b")

# The eleven layer views used by the published frame survey, kept identical so
# the candidate grids are the same size and directly comparable.
REPRESENTATIONS = (
    "L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15",
    "front_median", "back_median", "all_median",
)

# Declared before any detector was scored, on mechanistic grounds only (they
# are the scale-free circuit invariants plus the one global connectivity
# magnitude).  No label was consulted to form this split.
PRIMARY_FEATURES = (
    "gap_ratio",              # lambda_2 / lambda_n: near-decoupling of the token network
    "norm_fiedler",           # scale-free algebraic connectivity
    "spectral_erank",         # Laplacian spectral effective rank
    "kirchhoff_efficiency",   # AM-HM identity, 1 iff uniform clique
    "log_kirchhoff",          # total effective resistance (Kirchhoff index)
    "res_cv",                 # dispersion of pairwise effective resistances
    "res_shortcut",           # series-over-end-to-end resistance along the chunk
    "th_cv",                  # dispersion of Thevenin resistance to the state ground
    "th_slope",               # temporal trend of Thevenin resistance across the chunk
    "n_near_zero",            # number of connected components
)


MOBILITY_ROOT = PROJECT / "moe-v4-0904/results/layerwise_mobility"
MOBILITY_FILE = {
    "development_main": "main_reference.npz",
    "development_extra": "extra_reference.npz",
    "external_8b": "external_8b.npz",
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_cohort(cohort: str) -> dict[str, Any]:
    """Circuit features plus the published control quantities, all aligned."""
    circuit = load_npz(FEATURE_ROOT / f"{cohort}.npz")
    graph = load_npz(GRAPH_ROOT / f"{cohort}.npz")
    mobility = load_npz(MOBILITY_ROOT / MOBILITY_FILE[cohort])
    for other in (graph, mobility):
        if not np.array_equal(circuit["episode"], other["episode"]):
            raise ValueError(f"{cohort}: caches are not aligned")
        if not np.array_equal(circuit["task_index"], other["task_index"]):
            raise ValueError(f"{cohort}: task index mismatch")
    return {"circuit": circuit, "graph": graph, "mobility": mobility}


def quantity_list(cohort: dict[str, Any]) -> list[tuple[str, str, int]]:
    """(family, name, index) for every quantity scored, circuit first."""
    out = [("circuit", str(n), i)
           for i, n in enumerate(cohort["circuit"]["feature_names"].astype(str))]
    out += [("published", str(n), i)
            for i, n in enumerate(cohort["graph"]["metric_names"].astype(str))]
    out.append(("published", "mobility", -1))
    return out


def dense_values(cohort: dict[str, Any], family: str, index: int) -> np.ndarray:
    if family == "circuit":
        return dense_feature(cohort["circuit"], index)
    if index < 0:
        return np.asarray(cohort["mobility"]["mobility"], dtype=np.float32)
    return np.asarray(cohort["graph"]["metrics"][:, :, :, index], dtype=np.float32)


def dense_feature(circuit: dict[str, np.ndarray], index: int) -> np.ndarray:
    """Flat valid-only store -> dense [n_episodes, max_query, 8] with NaN gaps."""
    valid = circuit["valid"].astype(bool)
    out = np.full(valid.shape + (8,), np.nan, dtype=np.float32)
    out[circuit["flat_row"].astype(int), circuit["flat_query"].astype(int)] = (
        circuit["features"][:, :, index]
    )
    return out


def quantity_cache(cohort: dict[str, Any], family: str, index: int) -> dict[str, np.ndarray]:
    """Layout expected by dev.representations."""
    circuit = cohort["circuit"]
    return {
        "mobility": dense_values(cohort, family, index),
        "valid": circuit["valid"].astype(bool),
        "layer_names": circuit["layer_names"],
        "task_names": circuit["task_names"],
        "task_index": circuit["task_index"],
        "episode": circuit["episode"],
        "init_state_id": circuit["init_state_id"],
        "length": circuit["length"],
    }


def oriented(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = dev.trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed


__all__ = [
    "dev", "CONFIRMATIONS", "MIN_LOW_PRIOR_PRECISION", "WIDTH", "LOW_PRIOR",
    "MAX_TIMELY_FPR", "prior_of", "score_candidate", "suite_of",
    "survival_prior", "FEATURE_ROOT", "GRAPH_ROOT", "LABEL_PATHS", "COHORTS",
    "REPRESENTATIONS", "PRIMARY_FEATURES", "load_npz", "dense_feature",
    "quantity_cache", "oriented", "load_cohort", "quantity_list", "dense_values",
    "BUNDLE", "PROJECT",
]
