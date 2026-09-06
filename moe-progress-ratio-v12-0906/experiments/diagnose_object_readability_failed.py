#!/usr/bin/env python3
"""Object readability with the episode-outcome confound removed by construction.

The sibling diagnostic ``diagnose_object_readability.py`` asked whether MoE
routing can tell "the object came with me" from "the object stayed behind",
with the gripper closed and the arm executing comparable motion.  It answered
yes.  It has one substantial weakness, and this experiment is about that
weakness only.

The weakness
------------
All 232 ``missed_grasp_then_departure`` (phantom) events sit in **failed**
episodes.  The coupled control pool it matched them against is 96% **successful**
episodes.  So an AUC that looks like "the object stayed behind" could in part be
reading "this episode is going to fail".  The sibling ran a ``failed_controls``
sensitivity that shrank the median action-token gate-entropy effect from 0.280
to 0.157 without eliminating it, but at 10 task clusters and 2,000 bootstrap
draws nothing survived Benjamini-Hochberg -- and the bootstrap p-floor of
1/2000 = 5e-4 was itself above the 0.05/336 = 1.5e-4 that BH needs, so that was
a *resolution* statement, not an evidence statement.

What this experiment changes
----------------------------
* **Both groups sit inside failed episodes.**  Cases are the 232 phantom events
  (100% failed by construction).  Controls are ``coupled_target_motion`` events
  **whose episode also failed** (1,184 of them).  Episode outcome is therefore
  held constant across the contrast and cannot drive any separation.
* **Both runs pooled**, so the design draws on every (run, task) cell in which a
  phantom and a failed-coupled event coexist.
* **Bootstrap unit is the task, not the (run, task) pair.**  The same LIBERO
  task executed under two seed batches is one scene and one object; treating it
  as two independent clusters would understate the interval.  The (run, task)
  variant is computed too and reported as a secondary, less conservative number.
* **10,000 draws.**  The p-floor drops to 1e-4, below the 1.5e-4 that BH over
  336 cells requires, so a cell that deserves to survive BH now can.

Everything else is deliberately identical to the sibling so the two are
comparable cell for cell: the same layers, the same token roles, the same
relative-query grid, the same three soft metrics at ``FINAL_FLOW``, the same
matched-pair AUC, and the same coupled-vs-coupled promotion null.  The pure
helpers are **copied** rather than imported, so the sibling stays byte-for-byte
reproducible and neither file can drift the other.

The question, stated so it can come back negative
-------------------------------------------------
    Once both groups are failing episodes, does the routing still separate
    object-followed from object-stayed, and at what relative query?

  * survives at comparable magnitude -> the effect is object response and the
    sibling headline stands with the confound resolved;
  * collapses to the null -> the earlier separation was episode outcome, and the
    sibling headline must be withdrawn;
  * survives attenuated -> both contributions are real and partly separable, and
    both magnitudes get reported.

The **direction** of the gate-entropy effect gets its own verdict.  An AUC below
0.5 means phantom routing is *sharper* than coupled routing, which is not the
signature of a confused model but of one confidently continuing a transport it
is no longer performing.  Folding that into an absolute-value summary would
throw the mechanistic content away.

The relative query -1 negative control is retained and reported whatever it
says: at -1 the gripper is still open and the object cannot yet have responded,
so anything separating there is residual matching imbalance and caps the
attribution of the later separation.

Nothing is trained or fitted.  CPU only; zarr is read in batches and reduced
immediately.

Outputs (under ``results/object_readability_failed``):
  matched_pairs.csv  one row per matched pair plus one row per unmatched case,
                     with a drop reason -- no case is dropped in silence
  separation.csv     design x scope x cluster unit x layer x token role x
                     relative query x metric: AUC, clustered CI, the
                     coupled-vs-coupled null, the paired test and BH
  summary.json       headline numbers, balance, the side-by-side against the
                     sibling design, provenance, runtime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))

from method import progress_ratio  # noqa: E402

ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_EVENTS = (
    WORKSPACE / "analysis_trap_taxonomy/results/belief_mismatch_events.csv.gz"
)
DEFAULT_OUTPUT = BUNDLE / "results/object_readability_failed"
# Read-only.  The sibling bundle is never written to by this script.
SIBLING_SEPARATION = BUNDLE / "results/object_readability/separation.csv"

SCHEMA = "himoe.object_readability_failed.v1"

RUN_IDS = {
    "seed1000_1007": "right-50x8-20260903",
    "seed1008_1015": "right-50x8b-20260903",
}

LAYER_NAMES = progress_ratio.LAYER_NAMES
N_EXPERTS = progress_ratio.N_EXPERTS
FINAL_FLOW = progress_ratio.FINAL_FLOW
N_FLOW = 10
N_TOKEN = 11
STATE_TOKEN = 0
ACTION_TOKENS = progress_ratio.ACTION
RAW_SHAPE = (len(LAYER_NAMES), N_FLOW, N_TOKEN, N_EXPERTS)

TOKEN_ROLES = ("state", "action")
METRICS = ("hellinger_to_pre", "gate_entropy", "gate_entropy_change")

RELATIVE_QUERIES = (-1, 0, 1, 2, 3, 4, 5)
REFERENCE_RELATIVE = -2

# Same caliper values as the sibling's primary design, so "primary here" and
# "failed_controls there" are the same matched design and the comparison is not
# confounded by the matching itself.
DESIGNS: dict[str, dict[str, Any]] = {
    "primary": {"eef_caliper_m": 0.025, "query_caliper": 3},
    "wide": {"eef_caliper_m": 0.050, "query_caliper": 5},
}
MAX_CONTROLS = 5
MIN_CONTROLS_FOR_NULL = 2

# The bootstrap resampling unit.  ``task`` is the headline: one LIBERO task is
# one scene and one object regardless of which seed batch executed it.
# ``run_task`` splits those into two and is reported as the less conservative
# secondary number.
CLUSTER_UNITS = ("task", "run_task")
PRIMARY_CLUSTER = "task"

# 10,000, not 2,000.  BH over 336 cells needs a p-value that can reach
# 0.05/336 = 1.5e-4; a 2,000-draw bootstrap floors at 5e-4 and cannot.
BOOTSTRAP_DRAWS = 10_000
SEED = 20260906
READ_BATCH = 512

# The band in which the two groups' arm travel actually overlaps, used only to
# describe the design (the calipers, not this band, do the matching).
OVERLAP_QUANTILES = (0.05, 0.95)


# --------------------------------------------------------------------------
# pure helpers -- no I/O, unit tested in tests/test_object_readability_failed.py
# --------------------------------------------------------------------------


def gate_entropy(probability: np.ndarray) -> np.ndarray:
    """Shannon entropy of the last axis normalised by ``log(n_expert)``.

    Uniform -> 1.0, one-hot -> 0.0.
    """
    values = progress_ratio.normalize_probability(probability)
    n_expert = values.shape[-1]
    if n_expert < 2:
        raise ValueError("entropy normalisation needs at least two categories")
    safe = np.clip(values, 1e-12, None)
    entropy = -(values * np.log(safe)).sum(axis=-1)
    return (entropy / np.log(n_expert)).astype(np.float32)


def role_reduce(per_token: np.ndarray, role: str) -> np.ndarray:
    """``[..., 11]`` -> ``[...]`` for one token role.

    ``state`` reads token 0 alone; ``action`` averages tokens 1..10.
    """
    values = np.asarray(per_token, dtype=np.float32)
    if values.shape[-1] != N_TOKEN:
        raise ValueError(f"expected {N_TOKEN} tokens, got {values.shape[-1]}")
    if role == "state":
        return values[..., STATE_TOKEN]
    if role == "action":
        return values[..., ACTION_TOKENS].mean(axis=-1, dtype=np.float32)
    raise ValueError(f"unknown token role {role!r}")


def event_features(route: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Routes ``[8, 11, 32]`` and a reference of the same shape -> features.

    Returns ``[n_metric, n_layer, n_role]`` in the order of ``METRICS`` and
    ``TOKEN_ROLES``.  All-NaN in, all-NaN out.
    """
    route = np.asarray(route, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    expected = (len(LAYER_NAMES), N_TOKEN, N_EXPERTS)
    if route.shape != expected or reference.shape != expected:
        raise ValueError(f"routes must have shape {expected}")
    output = np.full((len(METRICS), len(LAYER_NAMES), len(TOKEN_ROLES)), np.nan, np.float32)
    if not (np.isfinite(route).all() and np.isfinite(reference).all()):
        return output
    hell = progress_ratio.hellinger(route, reference)  # [8, 11]
    entropy = gate_entropy(route)  # [8, 11]
    entropy_reference = gate_entropy(reference)
    per_metric = {
        "hellinger_to_pre": hell,
        "gate_entropy": entropy,
        "gate_entropy_change": entropy - entropy_reference,
    }
    for m, metric in enumerate(METRICS):
        for r, role in enumerate(TOKEN_ROLES):
            output[m, :, r] = role_reduce(per_metric[metric], role)
    return output


def match_events(
    case: pd.DataFrame,
    control: pd.DataFrame,
    *,
    eef_caliper: float,
    query_caliper: float,
    max_controls: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:M nearest-neighbour matching inside ``(run, task)`` under two calipers.

    ``case`` and ``control`` must carry ``event_id``, ``run``, ``task``,
    ``episode``, ``eef_max_displacement_after_m`` and
    ``post_closure_state_query``.

    Rules, all hard:
      * a control shares the case's ``run`` **and** ``task``;
      * a control comes from a different ``episode``;
      * ``|d eef displacement| <= eef_caliper`` and
        ``|d post-closure query| <= query_caliper`` -- the second is mandatory
        because phantom closures sit around query 27 and coupled ones around
        query 12, and position in the episode would otherwise be the
        discriminator;
      * at most ``max_controls`` controls per case, nearest by
        caliper-standardised distance, ties broken by ``event_id``;
      * controls are distinct within a stratum and may recur across strata.

    Returns ``(pairs, unmatched)``; every case appears in exactly one of them.
    """
    if eef_caliper <= 0 or query_caliper <= 0:
        raise ValueError("calipers must be positive")
    if max_controls < 1:
        raise ValueError("max_controls must be positive")

    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    control_by_cell = {key: frame for key, frame in control.groupby(["run", "task"], sort=True)}

    for event in case.sort_values("event_id").to_dict("records"):
        cell = control_by_cell.get((event["run"], event["task"]))
        if cell is None or len(cell) == 0:
            unmatched.append({**event, "n_candidates": 0, "drop_reason": "no_control_in_task"})
            continue
        d_eef = np.abs(
            cell["eef_max_displacement_after_m"].to_numpy(dtype=np.float64)
            - float(event["eef_max_displacement_after_m"])
        )
        d_query = np.abs(
            cell["post_closure_state_query"].to_numpy(dtype=np.float64)
            - float(event["post_closure_state_query"])
        )
        different_episode = cell["episode"].to_numpy() != event["episode"]
        eligible = (d_eef <= eef_caliper) & (d_query <= query_caliper) & different_episode
        n_candidates = int(eligible.sum())
        if n_candidates == 0:
            unmatched.append({**event, "n_candidates": 0, "drop_reason": "caliper_empty"})
            continue
        distance = d_eef / eef_caliper + d_query / query_caliper
        candidate_ids = cell["event_id"].to_numpy()
        index = np.flatnonzero(eligible)
        order = index[np.lexsort((candidate_ids[index], distance[index]))][:max_controls]
        for rank, position in enumerate(order):
            row = cell.iloc[int(position)]
            pairs.append(
                {
                    "stratum_id": int(event["event_id"]),
                    "run": event["run"],
                    "task": event["task"],
                    "run_task": f"{event['run']}|{event['task']}",
                    "phantom_event_id": int(event["event_id"]),
                    "phantom_episode": int(event["episode"]),
                    "phantom_target": event["target"],
                    "phantom_success": bool(event["success"]),
                    "phantom_post_closure_state_query": int(event["post_closure_state_query"]),
                    "phantom_eef_displacement_m": float(event["eef_max_displacement_after_m"]),
                    "phantom_target_displacement_m": float(
                        event["target_max_displacement_after_m"]
                    ),
                    "phantom_repeated_closure": bool(event["repeated_target_closure"]),
                    "control_event_id": int(row["event_id"]),
                    "control_episode": int(row["episode"]),
                    "control_target": row["target"],
                    "control_success": bool(row["success"]),
                    "control_post_closure_state_query": int(row["post_closure_state_query"]),
                    "control_eef_displacement_m": float(row["eef_max_displacement_after_m"]),
                    "control_target_displacement_m": float(
                        row["target_max_displacement_after_m"]
                    ),
                    "control_rank": rank,
                    "abs_eef_difference_m": float(d_eef[position]),
                    "abs_query_difference": int(d_query[position]),
                    "caliper_distance": float(distance[position]),
                    "n_candidates": n_candidates,
                    "n_controls": len(order),
                }
            )
    pair_frame = pd.DataFrame(pairs)
    unmatched_frame = pd.DataFrame(unmatched)
    return pair_frame, unmatched_frame


def stratum_case_score(case: float, controls: np.ndarray) -> float:
    """P(case > control) + 0.5 P(case == control) inside one matched stratum.

    Each stratum weighs the same regardless of how many controls it found, so a
    task with many controls cannot outvote one with few.
    """
    values = np.asarray(controls, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not np.isfinite(case) or len(values) == 0:
        return np.nan
    return float((case > values).mean() + 0.5 * (case == values).mean())


def stratum_promotion_scores(controls: np.ndarray) -> np.ndarray:
    """Failed-coupled versus failed-coupled: promote each control in turn.

    Entry ``j`` is the score control ``j`` obtains against the *other* controls
    of the same stratum.  The average over ``j`` is exactly 0.5 by antisymmetry,
    which is why the reported null promotes a single control rather than
    averaging: all of its information is in the spread.
    """
    values = np.asarray(controls, dtype=np.float64)
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    output = np.full(len(values), np.nan, dtype=np.float64)
    if n_valid < MIN_CONTROLS_FOR_NULL:
        return output
    safe = np.where(finite, values, np.nan)
    greater = (safe[:, None] > safe[None, :]) & finite[None, :]
    equal = (safe[:, None] == safe[None, :]) & finite[None, :]
    # the diagonal is a control against itself: an exact tie worth 0.5
    score = greater.sum(axis=1) + 0.5 * equal.sum(axis=1) - 0.5
    output[finite] = score[finite] / (n_valid - 1)
    return output


def stratum_scores(case: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over a trailing cell axis.

    ``case`` is ``[n_cell]``, ``controls`` is ``[n_control, n_cell]``; returns
    ``(case_score [n_cell], promotion_score [n_control, n_cell])`` with exactly
    the semantics of the two scalar functions applied cell by cell.
    """
    case = np.asarray(case, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 2 or case.shape != controls.shape[1:]:
        raise ValueError("case must be [n_cell] and controls [n_control, n_cell]")
    finite = np.isfinite(controls)
    n_valid = finite.sum(axis=0)

    greater = (case[None, :] > controls) & finite
    equal = (case[None, :] == controls) & finite
    with np.errstate(invalid="ignore", divide="ignore"):
        case_score = (greater.sum(axis=0) + 0.5 * equal.sum(axis=0)) / n_valid
    case_score = np.where(np.isfinite(case) & (n_valid > 0), case_score, np.nan)

    pair_greater = (controls[:, None, :] > controls[None, :, :]) & finite[None, :, :]
    pair_equal = (controls[:, None, :] == controls[None, :, :]) & finite[None, :, :]
    raw = pair_greater.sum(axis=1) + 0.5 * pair_equal.sum(axis=1) - 0.5
    with np.errstate(invalid="ignore", divide="ignore"):
        promotion = raw / (n_valid[None, :] - 1)
    promotion = np.where(finite & (n_valid[None, :] >= MIN_CONTROLS_FOR_NULL), promotion, np.nan)
    return case_score, promotion


def bootstrap_weights(
    cluster: np.ndarray, draws: int, rng: np.random.Generator
) -> np.ndarray:
    """``[draws, n_stratum]`` weights that resample whole clusters.

    Strata of one cluster move together.  With ``cluster`` set to the task, the
    two seed batches of one LIBERO task are a single unit, which is the
    conservative and correct reading: they share a scene, an object and a policy.
    """
    cluster = np.asarray(cluster)
    names, index = np.unique(cluster, return_inverse=True)
    n_cluster = len(names)
    if n_cluster == 0:
        return np.zeros((draws, len(cluster)), dtype=np.float64)
    picks = rng.integers(0, n_cluster, size=(draws, n_cluster))
    counts = np.zeros((draws, n_cluster), dtype=np.float64)
    for draw in range(draws):
        counts[draw] = np.bincount(picks[draw], minlength=n_cluster)
    return counts[:, index]


def weighted_means(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``[draws, n]`` x ``[n]`` -> ``[draws]`` NaN-aware weighted means."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    effective = weights * finite
    denominator = effective.sum(axis=1)
    numerator = effective @ np.where(finite, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        output = numerator / denominator
    output[denominator <= 0] = np.nan
    return output


def weighted_column_means(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """``[draws, n]`` x ``[n, n_cell]`` -> ``[draws, n_cell]``.

    The matrix form of :func:`weighted_means`, one column at a time.  It exists
    because at 10,000 draws the per-cell Python loop dominates the runtime, and
    the two must agree exactly -- a unit test pins that.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or weights.shape[1] != values.shape[0]:
        raise ValueError("weights [draws, n] and values [n, n_cell] do not align")
    finite = np.isfinite(values)
    numerator = weights @ np.where(finite, values, 0.0)
    denominator = weights @ finite.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        output = numerator / denominator
    output[denominator <= 0] = np.nan
    return output


def promoted_column_means(
    weights: np.ndarray, promotion: np.ndarray, promoted: np.ndarray
) -> np.ndarray:
    """The null statistic, in matrix form.

    ``weights`` is ``[draws, n_stratum]``, ``promotion`` is ``[n_stratum,
    n_control, n_cell]`` and ``promoted`` is ``[draws, n_stratum]`` holding the
    control index promoted into the case's seat in that draw.  Returns
    ``[draws, n_cell]``.

    Written as a sum of ``n_control`` matrix products rather than a gather,
    because the gathered array would be ``[draws, n_stratum, n_cell]`` -- half a
    billion float64 at the sizes used here.
    """
    weights = np.asarray(weights, dtype=np.float64)
    promotion = np.asarray(promotion, dtype=np.float64)
    promoted = np.asarray(promoted)
    if promotion.ndim != 3 or weights.shape != promoted.shape:
        raise ValueError("promotion must be [n_stratum, n_control, n_cell]")
    if weights.shape[1] != promotion.shape[0]:
        raise ValueError("weights and promotion disagree on the number of strata")
    n_control = promotion.shape[1]
    finite = np.isfinite(promotion)
    filled = np.where(finite, promotion, 0.0)
    numerator = np.zeros((weights.shape[0], promotion.shape[2]), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for k in range(n_control):
        selected = weights * (promoted == k)
        if not selected.any():
            continue
        numerator += selected @ filled[:, k, :]
        denominator += selected @ finite[:, k, :].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        output = numerator / denominator
    output[denominator <= 0] = np.nan
    return output


def point_mean(values: np.ndarray) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, 0
    return float(values[finite].mean()), int(finite.sum())


def percentile_interval(samples: np.ndarray) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 2:
        return np.nan, np.nan
    lo, hi = np.quantile(samples, (0.025, 0.975))
    return float(lo), float(hi)


def column_intervals(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[draws, n_cell]`` -> two ``[n_cell]`` percentile bounds, NaN-aware.

    Columns with fewer than two finite draws come back NaN, matching
    :func:`percentile_interval`.
    """
    samples = np.asarray(samples, dtype=np.float64)
    finite_count = np.isfinite(samples).sum(axis=0)
    safe = np.where(finite_count >= 2, samples, 0.0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanquantile(safe, 0.025, axis=0)
        hi = np.nanquantile(safe, 0.975, axis=0)
    lo = np.where(finite_count >= 2, lo, np.nan)
    hi = np.where(finite_count >= 2, hi, np.nan)
    return lo, hi


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up over one design x scope x cluster grid.

    336 cells per grid; at a 2.5% per-cell criterion about 8 would beat the null
    by chance.  Neighbouring layers and neighbouring relative queries read almost
    the same routing, so BH is conservative here rather than exact.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return output
    subset = values[finite]
    order = np.argsort(subset)
    ranked = subset[order] * len(subset) / np.arange(1, len(subset) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty_like(subset)
    adjusted[order] = np.minimum(ranked, 1.0)
    output[finite] = adjusted
    return output


# --------------------------------------------------------------------------
# corpus I/O
# --------------------------------------------------------------------------


def load_events(path: Path) -> pd.DataFrame:
    events = pd.read_csv(path)
    overlap = int((events["missed_grasp_then_departure"] & events["coupled_target_motion"]).sum())
    if overlap:
        raise ValueError(f"the two groups overlap on {overlap} events; the contrast is not clean")
    unknown = sorted(set(events["run"].unique()) - set(RUN_IDS))
    if unknown:
        raise ValueError(f"unknown runs in the event table: {unknown}")
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events


def episode_rows(route_path: Path) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """``episode -> global row indices`` with contiguity asserted."""
    group = zarr.open_group(str(route_path), mode="r")
    source = group["hb_router_probs"]
    if tuple(source.shape[1:]) != RAW_SHAPE:
        raise ValueError(f"unexpected HB shape at {route_path}: {source.shape}")
    episode_id = np.asarray(group["episode_id"][:], dtype=np.int64)
    control_step = np.asarray(group["control_step"][:], dtype=np.int64)
    if len(episode_id) != source.shape[0]:
        raise ValueError(f"episode_id/hb_router_probs row mismatch at {route_path}")
    lookup: dict[int, np.ndarray] = {}
    for episode in np.unique(episode_id):
        rows = np.flatnonzero(episode_id == episode)
        if len(rows) > 1 and not np.all(np.diff(rows) == 1):
            raise ValueError(f"non-contiguous episode {episode} in {route_path}")
        steps = control_step[rows]
        if len(steps) > 1 and not np.all(np.diff(steps) == 1):
            raise ValueError(f"control_step skips inside episode {episode} in {route_path}")
        lookup[int(episode)] = rows
    record = {
        "rows": int(len(episode_id)),
        "episodes": int(len(lookup)),
        "dtype": str(source.dtype),
    }
    return lookup, record


def read_routes(route_path: Path, rows: np.ndarray, batch: int) -> np.ndarray:
    """``[n, 8, 11, 32]`` final-flow soft routes for the requested rows."""
    group = zarr.open_group(str(route_path), mode="r")
    source = group["hb_router_probs"]
    output = np.empty((len(rows), len(LAYER_NAMES), N_TOKEN, N_EXPERTS), dtype=np.float32)
    for start in range(0, len(rows), batch):
        stop = min(start + batch, len(rows))
        block = np.asarray(
            source.oindex[rows[start:stop], :, FINAL_FLOW : FINAL_FLOW + 1, :, :],
            dtype=np.float32,
        )
        output[start:stop] = progress_ratio.normalize_probability(block[:, :, 0, :, :])
    return output


def collect_features(
    events: pd.DataFrame,
    needed: np.ndarray,
    route_root: Path,
    batch: int,
    verbose: bool,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Read routing for the requested events and reduce to features immediately."""
    wanted = events[events["event_id"].isin(needed)]
    features: dict[int, np.ndarray] = {}
    zarr_record: dict[str, Any] = {}
    truncated = 0
    total_query_slots = 0
    for (run, task), cell in wanted.groupby(["run", "task"], sort=True):
        route_path = route_root / str(task) / RUN_IDS[str(run)] / "server/routes.zarr"
        if not route_path.is_dir():
            raise FileNotFoundError(route_path)
        lookup, record = episode_rows(route_path)
        zarr_record[f"{run}/{task}"] = record

        requests: list[tuple[int, int, int]] = []  # (event_id, slot, global row)
        slots = (REFERENCE_RELATIVE,) + RELATIVE_QUERIES
        for event in cell.to_dict("records"):
            episode = int(event["episode"])
            if episode not in lookup:
                raise ValueError(f"episode {episode} missing from {route_path}")
            rows = lookup[episode]
            anchor = int(event["post_closure_state_query"])
            for slot, relative in enumerate(slots):
                total_query_slots += 1
                query = anchor + relative
                if query < 0 or query >= len(rows):
                    truncated += 1
                    continue
                requests.append((int(event["event_id"]), slot, int(rows[query])))
        if not requests:
            continue
        unique_rows = np.unique(np.asarray([item[2] for item in requests], dtype=np.int64))
        routes = read_routes(route_path, unique_rows, batch)
        position_of = {int(row): index for index, row in enumerate(unique_rows)}

        buffer: dict[int, np.ndarray] = {}
        for event_id, slot, row in requests:
            store = buffer.setdefault(
                event_id,
                np.full((len(slots), len(LAYER_NAMES), N_TOKEN, N_EXPERTS), np.nan, np.float32),
            )
            store[slot] = routes[position_of[row]]
        for event_id, store in buffer.items():
            reference = store[0]
            block = np.full(
                (len(METRICS), len(LAYER_NAMES), len(TOKEN_ROLES), len(RELATIVE_QUERIES)),
                np.nan,
                np.float32,
            )
            if np.isfinite(reference).all():
                for r_index in range(len(RELATIVE_QUERIES)):
                    block[:, :, :, r_index] = event_features(store[r_index + 1], reference)
            features[event_id] = block
        del routes, buffer
        if verbose:
            print(
                f"  [{run}] {task} rows={record['rows']} events={len(cell)} "
                f"reads={len(unique_rows)}",
                flush=True,
            )
    coverage = {
        "events_read": len(features),
        "query_slots_requested": total_query_slots,
        "query_slots_outside_episode": truncated,
        "zarr": zarr_record,
    }
    return features, coverage


# --------------------------------------------------------------------------
# separation
# --------------------------------------------------------------------------


CELL_SHAPE = (len(METRICS), len(LAYER_NAMES), len(TOKEN_ROLES), len(RELATIVE_QUERIES))
N_CELL = int(np.prod(CELL_SHAPE))


def cell_index(metric: int, layer: int, role: int, relative: int) -> int:
    return int(np.ravel_multi_index((metric, layer, role, relative), CELL_SHAPE))


def cell_frame() -> pd.DataFrame:
    """The flattened cell grid as a frame, in ``reshape(-1)`` order."""
    rows = []
    for metric in METRICS:
        for layer in LAYER_NAMES:
            for role in TOKEN_ROLES:
                for relative in RELATIVE_QUERIES:
                    rows.append(
                        {
                            "layer": layer,
                            "token_role": role,
                            "relative_query": int(relative),
                            "metric": metric,
                        }
                    )
    return pd.DataFrame(rows)


def stratum_table(
    pairs: pd.DataFrame, features: dict[int, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-stratum case score and the promotion scores that build the null."""
    strata = (
        pairs.groupby("stratum_id", sort=True)
        .agg(
            run=("run", "first"),
            task=("task", "first"),
            run_task=("run_task", "first"),
            phantom_event_id=("phantom_event_id", "first"),
            n_controls=("control_event_id", "size"),
        )
        .reset_index()
    )
    case = np.full((len(strata), N_CELL), np.nan, dtype=np.float64)
    promotion = np.full((len(strata), MAX_CONTROLS, N_CELL), np.nan, dtype=np.float64)
    missing = np.full(N_CELL, np.nan, dtype=np.float64)

    control_ids = {
        int(key): frame.sort_values("control_rank")["control_event_id"].to_numpy()
        for key, frame in pairs.groupby("stratum_id", sort=True)
    }
    for s_index, stratum in enumerate(strata.to_dict("records")):
        phantom = features.get(int(stratum["phantom_event_id"]))
        if phantom is None:
            continue
        ids = control_ids[int(stratum["stratum_id"])]
        stack = np.stack(
            [features.get(int(cid), missing).reshape(-1).astype(np.float64) for cid in ids]
        )
        case_score, promotion_score = stratum_scores(phantom.reshape(-1), stack)
        case[s_index] = case_score
        promotion[s_index, : len(ids)] = promotion_score
    return strata, case, promotion


def separation_rows(
    design: str,
    scope: str,
    cluster_unit: str,
    strata: pd.DataFrame,
    case: np.ndarray,
    promotion: np.ndarray,
    pairs: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """AUC, clustered CI and the failed-coupled-vs-failed-coupled null, per cell.

    The null is not the leave-one-out average -- that is identically 0.5 by
    antisymmetry.  Each draw promotes **one** control per stratum into the case's
    seat, so the null statistic has exactly the case statistic's structure and
    its spread is the honest noise floor of the whole pipeline.
    """
    if len(strata) == 0:
        return pd.DataFrame()
    n_strata = len(strata)
    weights = bootstrap_weights(strata[cluster_unit].to_numpy(), draws, rng)
    n_controls = strata["n_controls"].to_numpy(dtype=np.int64)
    promoted = rng.integers(0, np.broadcast_to(n_controls, (draws, n_strata)))
    pair_counts = (
        pairs.groupby("stratum_id", sort=True).size().reindex(strata["stratum_id"]).to_numpy()
    )
    cluster_of = strata[cluster_unit].to_numpy()
    task_of = strata["task"].to_numpy()

    boot_case = weighted_column_means(weights, case)  # [draws, n_cell]
    boot_null = promoted_column_means(weights, promotion, promoted)
    unweighted_null = promoted_column_means(
        np.ones_like(weights), promotion, promoted
    )

    auc_lo, auc_hi = column_intervals(boot_case)
    null_lo, null_hi = column_intervals(boot_null)
    delta = np.abs(boot_case - 0.5) - np.abs(boot_null - 0.5)
    delta_lo, delta_hi = column_intervals(delta)

    finite_delta = np.isfinite(delta)
    n_finite_delta = finite_delta.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_boot = ((delta <= 0.0) & finite_delta).sum(axis=0) / np.maximum(n_finite_delta, 1)
        p_boot = np.maximum(p_boot, 1.0 / np.maximum(n_finite_delta, 1))
    p_boot = np.where(n_finite_delta > 0, p_boot, np.nan)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        null_auc = np.nanmean(unweighted_null, axis=0)
        null_abs = np.nanmean(np.abs(boot_null - 0.5), axis=0)

    finite_case = np.isfinite(case)
    with np.errstate(invalid="ignore"):
        auc = np.nansum(np.where(finite_case, case, 0.0), axis=0) / np.maximum(
            finite_case.sum(axis=0), 1
        )
    auc = np.where(finite_case.any(axis=0), auc, np.nan)

    n_strata_cell = finite_case.sum(axis=0)
    n_pairs_cell = (finite_case * pair_counts[:, None]).sum(axis=0)
    n_tasks_cell = np.array(
        [
            int(pd.unique(task_of[finite_case[:, c]]).size) if finite_case[:, c].any() else 0
            for c in range(N_CELL)
        ]
    )
    n_clusters_cell = np.array(
        [
            int(pd.unique(cluster_of[finite_case[:, c]]).size) if finite_case[:, c].any() else 0
            for c in range(N_CELL)
        ]
    )

    frame = cell_frame()
    frame.insert(0, "cluster_unit", cluster_unit)
    frame.insert(0, "scope", scope)
    frame.insert(0, "design", design)
    frame["n_strata"] = n_strata_cell
    frame["n_pairs"] = n_pairs_cell
    frame["n_tasks"] = n_tasks_cell
    frame["n_clusters"] = n_clusters_cell
    frame["auc"] = auc
    frame["auc_lo"] = auc_lo
    frame["auc_hi"] = auc_hi
    frame["auc_abs"] = np.abs(auc - 0.5)
    frame["null_auc"] = null_auc
    frame["null_lo"] = null_lo
    frame["null_hi"] = null_hi
    frame["null_abs_mean"] = null_abs
    frame["n_strata_null"] = np.isfinite(promotion).any(axis=1).sum(axis=0)
    frame["delta_abs"] = np.abs(auc - 0.5) - null_abs
    frame["delta_abs_lo"] = delta_lo
    frame["delta_abs_hi"] = delta_hi
    frame["p_boot"] = p_boot
    frame["exceeds_null_paired"] = np.isfinite(delta_lo) & (delta_lo > 0.0)
    frame["exceeds_null_disjoint"] = (
        np.isfinite(auc_lo)
        & np.isfinite(null_hi)
        & ((auc_lo > null_hi) | (auc_hi < null_lo))
    )
    return frame


def earliest_exceeding(
    separation: pd.DataFrame, design: str, scope: str, cluster_unit: str
) -> dict[str, Any]:
    """First relative query at which any cell beats the null, per metric/role."""
    frame = separation[
        (separation["design"] == design)
        & (separation["scope"] == scope)
        & (separation["cluster_unit"] == cluster_unit)
    ]
    hit = frame[frame["exceeds_null_paired"]]
    strict = frame[frame["exceeds_null_bh"]] if "exceeds_null_bh" in frame else frame.iloc[:0]
    output: dict[str, Any] = {
        "any": None if hit.empty else int(hit["relative_query"].min()),
        "any_bh_controlled": None if strict.empty else int(strict["relative_query"].min()),
        "any_excluding_negative_control": (
            None
            if hit[hit["relative_query"] >= 0].empty
            else int(hit[hit["relative_query"] >= 0]["relative_query"].min())
        ),
        "by_metric_role": {},
        "by_layer": {},
    }
    for (metric, role), group in frame.groupby(["metric", "token_role"], sort=True):
        winners = group[group["exceeds_null_paired"]]
        output["by_metric_role"][f"{metric}|{role}"] = (
            None if winners.empty else int(winners["relative_query"].min())
        )
    for layer, group in frame.groupby("layer", sort=False):
        winners = group[group["exceeds_null_paired"]]
        output["by_layer"][str(layer)] = (
            None if winners.empty else int(winners["relative_query"].min())
        )
    return output


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def select(
    separation: pd.DataFrame, design: str, scope: str, cluster_unit: str
) -> pd.DataFrame:
    return separation[
        (separation["design"] == design)
        & (separation["scope"] == scope)
        & (separation["cluster_unit"] == cluster_unit)
    ]


def print_grid(
    separation: pd.DataFrame,
    design: str,
    scope: str,
    cluster_unit: str,
    metric: str,
    role: str,
) -> None:
    frame = select(separation, design, scope, cluster_unit)
    frame = frame[(frame["metric"] == metric) & (frame["token_role"] == role)]
    if frame.empty:
        return
    print(
        f"\n  {metric} / {role} token(s)  --  AUC * |null band half-width;"
        "  * = beats the failed-coupled-vs-failed-coupled null"
    )
    print("    layer " + "".join(f"{rel:>13d}" for rel in RELATIVE_QUERIES))
    for layer in LAYER_NAMES:
        cells = []
        for rel in RELATIVE_QUERIES:
            row = frame[(frame["layer"] == layer) & (frame["relative_query"] == rel)]
            if row.empty:
                cells.append(f"{'--':>13}")
                continue
            row = row.iloc[0]
            mark = "*" if row["exceeds_null_paired"] else " "
            band = (row["null_hi"] - row["null_lo"]) / 2.0
            cells.append(f"{row['auc']:>7.3f}{mark}|{band:.02f}")
        print(f"    {layer:<6}" + "".join(cells))


def print_detail(
    separation: pd.DataFrame,
    design: str,
    scope: str,
    cluster_unit: str,
    metric: str,
    role: str,
) -> None:
    frame = select(separation, design, scope, cluster_unit)
    frame = frame[(frame["metric"] == metric) & (frame["token_role"] == role)]
    if frame.empty:
        return
    print(
        f"\n  detail: {metric} / {role}, design={design}, scope={scope}, "
        f"cluster={cluster_unit}"
    )
    print(
        "    layer  rel        AUC [95% CI]              null [95% CI]          "
        "delta|.-0.5| [95% CI]      n   q_BH"
    )
    for layer in LAYER_NAMES:
        for rel in RELATIVE_QUERIES:
            row = frame[(frame["layer"] == layer) & (frame["relative_query"] == rel)]
            if row.empty:
                continue
            row = row.iloc[0]
            print(
                f"    {layer:<6} {rel:>3}  "
                f"{row['auc']:.3f} [{row['auc_lo']:.3f},{row['auc_hi']:.3f}]   "
                f"{row['null_auc']:.3f} [{row['null_lo']:.3f},{row['null_hi']:.3f}]   "
                f"{row['delta_abs']:+.3f} [{row['delta_abs_lo']:+.3f},{row['delta_abs_hi']:+.3f}]  "
                f"{int(row['n_strata']):>4}  {row['q_bh']:.4f}"
            )


def balance_record(pairs: pd.DataFrame, unmatched: pd.DataFrame, n_case: int) -> dict[str, Any]:
    if pairs.empty:
        return {
            "case_total": int(n_case),
            "case_matched": 0,
            "case_unmatched": int(len(unmatched)),
            "unmatched_reasons": (
                unmatched["drop_reason"].value_counts().to_dict() if len(unmatched) else {}
            ),
        }
    per_stratum = pairs.groupby("stratum_id").size()
    reuse = pairs["control_event_id"].value_counts()
    return {
        "case_total": int(n_case),
        "case_matched": int(pairs["stratum_id"].nunique()),
        "case_unmatched": int(len(unmatched)),
        "unmatched_reasons": (
            unmatched["drop_reason"].value_counts().to_dict() if len(unmatched) else {}
        ),
        "pairs": int(len(pairs)),
        "controls_per_case_mean": float(per_stratum.mean()),
        "controls_per_case_median": float(per_stratum.median()),
        "controls_per_case_min": int(per_stratum.min()),
        "controls_per_case_max": int(per_stratum.max()),
        "strata_with_at_least_two_controls": int((per_stratum >= MIN_CONTROLS_FOR_NULL).sum()),
        "distinct_control_events": int(pairs["control_event_id"].nunique()),
        "control_reuse_max": int(reuse.max()),
        "control_reuse_mean": float(reuse.mean()),
        "tasks": int(pairs["task"].nunique()),
        "run_task_cells": int(pairs["run_task"].nunique()),
        "runs": sorted(pairs["run"].unique().tolist()),
        # Held constant by construction: both sides must be failed episodes.
        "case_success_rate": float(
            pairs.groupby("stratum_id")["phantom_success"].first().mean()
        ),
        "control_success_rate": float(pairs["control_success"].mean()),
        "abs_eef_difference_m_mean": float(pairs["abs_eef_difference_m"].mean()),
        "abs_eef_difference_m_max": float(pairs["abs_eef_difference_m"].max()),
        "abs_query_difference_mean": float(pairs["abs_query_difference"].mean()),
        "abs_query_difference_max": int(pairs["abs_query_difference"].max()),
        "signed_eef_difference_m_mean": float(
            (pairs["phantom_eef_displacement_m"] - pairs["control_eef_displacement_m"]).mean()
        ),
        "signed_query_difference_mean": float(
            (
                pairs["phantom_post_closure_state_query"]
                - pairs["control_post_closure_state_query"]
            ).mean()
        ),
        "phantom_eef_displacement_m_median": float(
            pairs.groupby("stratum_id")["phantom_eef_displacement_m"].first().median()
        ),
        "control_eef_displacement_m_median": float(
            pairs.groupby("stratum_id")["control_eef_displacement_m"].mean().median()
        ),
        "phantom_target_displacement_m_median": float(
            pairs.groupby("stratum_id")["phantom_target_displacement_m"].first().median()
        ),
        "control_target_displacement_m_median": float(
            pairs.groupby("stratum_id")["control_target_displacement_m"].mean().median()
        ),
        "phantom_query_median": float(
            pairs.groupby("stratum_id")["phantom_post_closure_state_query"].first().median()
        ),
        "control_query_median": float(
            pairs.groupby("stratum_id")["control_post_closure_state_query"].mean().median()
        ),
        "phantom_repeated_closures": int(
            pairs.groupby("stratum_id")["phantom_repeated_closure"].first().sum()
        ),
    }


CELL_KEYS = ["layer", "token_role", "relative_query", "metric"]


def side_by_side(
    separation: pd.DataFrame, sibling: pd.DataFrame | None, cluster_unit: str
) -> pd.DataFrame:
    """This design's primary grid joined to the sibling's primary and failed_controls.

    Same cells, same estimator; only the control pool, the bootstrap unit and the
    number of draws differ.  Returned wide so a reader can subtract by eye.
    """
    here = select(separation, "primary", "both", cluster_unit)[
        CELL_KEYS + ["auc", "auc_lo", "auc_hi", "null_auc", "p_boot", "q_bh"]
    ].rename(
        columns={
            "auc": "auc_failed_matched",
            "auc_lo": "lo_failed_matched",
            "auc_hi": "hi_failed_matched",
            "null_auc": "null_failed_matched",
            "p_boot": "p_failed_matched",
            "q_bh": "q_failed_matched",
        }
    )
    if sibling is None or sibling.empty:
        return here
    for name, suffix in (("primary", "sibling_primary"), ("failed_controls", "sibling_failed")):
        frame = sibling[(sibling["design"] == name) & (sibling["scope"] == "both")][
            CELL_KEYS + ["auc", "auc_lo", "auc_hi", "null_auc", "p_boot"]
        ].rename(
            columns={
                "auc": f"auc_{suffix}",
                "auc_lo": f"lo_{suffix}",
                "auc_hi": f"hi_{suffix}",
                "null_auc": f"null_{suffix}",
                "p_boot": f"p_{suffix}",
            }
        )
        here = here.merge(frame, on=CELL_KEYS, how="left")
    return here


def median_abs_auc_late_action(frame: pd.DataFrame) -> dict[str, float]:
    """Median |AUC - 0.5| over action tokens at relative queries 1..5, by metric.

    This is the sibling's sensitivity statistic, reproduced exactly so the
    attenuation can be read off directly.
    """
    subset = frame[
        (frame["token_role"] == "action") & (frame["relative_query"].between(1, 5))
    ].copy()
    if subset.empty:
        return {}
    subset["abs_auc"] = (subset["auc"] - 0.5).abs()
    return {
        str(metric): round(float(value), 4)
        for metric, value in subset.groupby("metric")["abs_auc"].median().items()
    }


def signed_median_auc_late_action(frame: pd.DataFrame) -> dict[str, float]:
    """The same cells, but keeping the sign -- the direction verdict needs it."""
    subset = frame[
        (frame["token_role"] == "action") & (frame["relative_query"].between(1, 5))
    ]
    if subset.empty:
        return {}
    return {
        str(metric): round(float(value), 4)
        for metric, value in subset.groupby("metric")["auc"].median().items()
    }


def direction_record(frame: pd.DataFrame) -> dict[str, Any]:
    """Is phantom routing sharper (gate-entropy AUC below 0.5) or noisier?

    Reported separately from the absolute-value summary because the sign is the
    mechanistic content: below 0.5 means the model is not confused but
    confidently continuing a transport it is no longer performing.
    """
    output: dict[str, Any] = {}
    for role in TOKEN_ROLES:
        subset = frame[
            (frame["metric"] == "gate_entropy")
            & (frame["token_role"] == role)
            & (frame["relative_query"] >= 0)
        ]
        if subset.empty or not subset["auc"].notna().any():
            continue
        beating = subset[subset["exceeds_null_paired"]]
        weakest = subset.loc[subset["auc"].idxmin()]
        output[role] = {
            "cells": int(len(subset)),
            "cells_below_half": int((subset["auc"] < 0.5).sum()),
            "median_auc": float(subset["auc"].median()),
            "min_auc": float(subset["auc"].min()),
            "max_auc": float(subset["auc"].max()),
            "cells_beating_null": int(len(beating)),
            "cells_beating_null_below_half": int((beating["auc"] < 0.5).sum())
            if len(beating)
            else 0,
            "min_auc_cell": "{}/rel{}".format(
                weakest["layer"], int(weakest["relative_query"])
            ),
            "min_auc_ci": [float(weakest["auc_lo"]), float(weakest["auc_hi"])],
        }
    return output


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sibling", type=Path, default=SIBLING_SEPARATION)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--batch", type=int, default=READ_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    events = load_events(args.events)
    phantom = events[events["missed_grasp_then_departure"]].copy()
    # The whole point of this experiment: the case group is 100% failed, so the
    # control group is restricted to failed episodes too.  Assert the premise
    # rather than trust it.
    case_success = float(phantom["success"].mean())
    if case_success != 0.0:
        raise ValueError(
            "the case group is not entirely inside failed episodes "
            f"({case_success:.4f} succeeded); the confound-removal premise fails"
        )
    coupled_all = events[
        events["coupled_target_motion"]
        & (events["post_closure_state_query"] >= -REFERENCE_RELATIVE)
    ].copy()
    control = coupled_all[~coupled_all["success"]].copy()
    control_excluded = int(events["coupled_target_motion"].sum()) - len(coupled_all)

    lo_q, hi_q = OVERLAP_QUANTILES
    band = (
        max(
            float(np.quantile(phantom["eef_max_displacement_after_m"], lo_q)),
            float(np.quantile(control["eef_max_displacement_after_m"], lo_q)),
        ),
        min(
            float(np.quantile(phantom["eef_max_displacement_after_m"], hi_q)),
            float(np.quantile(control["eef_max_displacement_after_m"], hi_q)),
        ),
    )
    in_band = lambda frame: int(  # noqa: E731
        (
            (frame["eef_max_displacement_after_m"] >= band[0])
            & (frame["eef_max_displacement_after_m"] <= band[1])
        ).sum()
    )
    pool_cells = set(map(tuple, phantom[["run", "task"]].drop_duplicates().to_numpy())) & set(
        map(tuple, control[["run", "task"]].drop_duplicates().to_numpy())
    )

    print("=" * 96)
    print("OBJECT READABILITY WITH THE EPISODE-OUTCOME CONFOUND REMOVED BY CONSTRUCTION")
    print("=" * 96)
    print(
        f"[design] cases (missed_grasp_then_departure) = {len(phantom)}, "
        f"success rate {case_success:.3f}"
    )
    print(
        f"[design] controls (coupled_target_motion AND episode failed) = {len(control)} "
        f"of {len(coupled_all)} coupled "
        f"(coupled success rate {float(coupled_all['success'].mean()):.3f}; "
        f"{control_excluded} coupled lack a pre-closure reference query)"
    )
    print(
        f"[design] arm-travel overlap band [{band[0]:.3f}, {band[1]:.3f}] m: "
        f"{in_band(phantom)} cases, {in_band(control)} failed-coupled "
        f"({in_band(control) / max(in_band(phantom), 1):.1f} controls per case)"
    )
    print(
        f"[design] pool before matching: {len(pool_cells)} (run, task) cells holding both "
        f"groups, {len({task for _, task in pool_cells})} distinct tasks "
        f"(cases span {phantom['task'].nunique()} tasks over "
        f"{phantom['run'].value_counts().to_dict()})"
    )
    print(
        "[design] NOTE the pool count above is BEFORE the calipers.  What the bootstrap "
        "actually resamples is the count AFTER matching, reported per design below and in "
        "per_grid[*]['n_clusters']; the two are not the same number and only the second is "
        "the power the design has."
    )
    print(
        f"[design] position in episode before matching: case post-closure query median "
        f"{float(phantom['post_closure_state_query'].median()):.1f} vs failed-coupled "
        f"{float(control['post_closure_state_query'].median()):.1f} "
        "-- the query caliper is what removes that"
    )

    designs: dict[str, dict[str, Any]] = {}
    all_pairs: list[pd.DataFrame] = []
    all_unmatched: list[pd.DataFrame] = []
    for name, config in DESIGNS.items():
        pairs, unmatched = match_events(
            phantom,
            control,
            eef_caliper=float(config["eef_caliper_m"]),
            query_caliper=float(config["query_caliper"]),
            max_controls=MAX_CONTROLS,
        )
        pairs = pairs.assign(design=name)
        if len(unmatched):
            unmatched = unmatched.assign(design=name)
        designs[name] = {"config": config, "pairs": pairs, "unmatched": unmatched}
        all_pairs.append(pairs)
        if len(unmatched):
            all_unmatched.append(unmatched)
        print(
            f"[design {name}] caliper eef<={config['eef_caliper_m']} m, "
            f"|d query|<={config['query_caliper']}: matched "
            f"{pairs['stratum_id'].nunique()}/{len(phantom)} cases, {len(pairs)} pairs, "
            f"{pairs['task'].nunique()} tasks, {pairs['run_task'].nunique()} (run, task) cells, "
            f"{len(unmatched)} unmatched "
            f"{unmatched['drop_reason'].value_counts().to_dict() if len(unmatched) else {}}",
            flush=True,
        )

    needed = np.unique(
        np.concatenate(
            [phantom["event_id"].to_numpy()]
            + [frame["control_event_id"].to_numpy() for frame in all_pairs if len(frame)]
        )
    )
    print(
        f"[read] {len(needed)} distinct events, "
        f"{len(needed) * (len(RELATIVE_QUERIES) + 1)} query slots",
        flush=True,
    )
    read_started = time.perf_counter()
    features, coverage = collect_features(
        events, needed, args.route_root, args.batch, not args.quiet
    )
    read_elapsed = time.perf_counter() - read_started
    print(
        f"[read] done in {read_elapsed:.1f}s; {coverage['query_slots_outside_episode']} of "
        f"{coverage['query_slots_requested']} query slots fall outside their episode",
        flush=True,
    )

    frames: list[pd.DataFrame] = []
    scope_records: dict[str, Any] = {}
    headline_data: dict[str, Any] = {}
    boot_started = time.perf_counter()
    for name, bundle in designs.items():
        pairs = bundle["pairs"]
        if pairs.empty:
            continue
        for scope in ("both", *sorted(RUN_IDS)):
            subset = pairs if scope == "both" else pairs[pairs["run"] == scope]
            if subset.empty:
                continue
            strata, case, promotion = stratum_table(subset, features)
            # Inside one run, task and run_task are the same partition, so the
            # second variant would only duplicate rows.
            units = CLUSTER_UNITS if scope == "both" else (PRIMARY_CLUSTER,)
            for unit in units:
                rng = np.random.default_rng(args.seed)
                frames.append(
                    separation_rows(
                        name, scope, unit, strata, case, promotion, subset, args.draws, rng
                    )
                )
            if name == "primary" and scope == "both":
                headline_data = {"strata": strata, "case": case}
                # Strata with a single control contribute no promotion scores at
                # all, so the all-NaN slices here are expected, not a defect.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    leave_one_out = np.nanmean(promotion, axis=1)
                    deviation = float(
                        np.nanmax(np.abs(np.nanmean(leave_one_out, axis=0) - 0.5))
                    )
                if deviation > 1e-9:
                    raise ValueError(
                        "leave-one-out failed-coupled-vs-failed-coupled is not 0.5; the "
                        f"matched-pair estimator is biased (max deviation {deviation:.3e})"
                    )
                scope_records["null_symmetry_max_deviation"] = deviation
            scope_records[f"{name}|{scope}"] = {
                "strata": int(len(strata)),
                "pairs": int(len(subset)),
                "tasks": int(strata["task"].nunique()),
                "run_task_cells": int(strata["run_task"].nunique()),
            }
    separation = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    boot_elapsed = time.perf_counter() - boot_started
    print(f"[bootstrap] {args.draws} draws over all grids in {boot_elapsed:.1f}s", flush=True)

    if not separation.empty:
        separation["q_bh"] = np.nan
        for _, index in separation.groupby(
            ["design", "scope", "cluster_unit"], sort=False
        ).groups.items():
            separation.loc[index, "q_bh"] = bh_adjust(
                separation.loc[index, "p_boot"].to_numpy()
            )
        separation["exceeds_null_bh"] = separation["q_bh"] < 0.05

    pair_frame = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    unmatched_frame = (
        pd.concat(all_unmatched, ignore_index=True) if all_unmatched else pd.DataFrame()
    )
    if len(unmatched_frame):
        unmatched_out = unmatched_frame.rename(
            columns={
                "event_id": "phantom_event_id",
                "episode": "phantom_episode",
                "target": "phantom_target",
                "success": "phantom_success",
                "post_closure_state_query": "phantom_post_closure_state_query",
                "eef_max_displacement_after_m": "phantom_eef_displacement_m",
                "target_max_displacement_after_m": "phantom_target_displacement_m",
                "repeated_target_closure": "phantom_repeated_closure",
            }
        )
        keep = [
            column
            for column in unmatched_out.columns
            if column in set(pair_frame.columns)
            | {"drop_reason", "n_candidates", "design", "run", "task"}
        ]
        unmatched_out = unmatched_out[keep].assign(
            stratum_id=unmatched_out["phantom_event_id"], matched=False
        )
    else:
        unmatched_out = pd.DataFrame()
    matched_pairs = pd.concat(
        [pair_frame.assign(matched=True, drop_reason=""), unmatched_out], ignore_index=True
    )
    matched_pairs.to_csv(args.output / "matched_pairs.csv", index=False)
    separation.to_csv(args.output / "separation.csv", index=False)

    # ---------------------------------------------------------------- report
    for name, bundle in designs.items():
        record = balance_record(bundle["pairs"], bundle["unmatched"], len(phantom))
        designs[name]["balance"] = record
        print(
            f"\n[{name}] matched {record['case_matched']}/{record['case_total']} cases, "
            f"{record.get('pairs', 0)} pairs, {record.get('tasks', 0)} tasks, "
            f"{record.get('run_task_cells', 0)} (run, task) cells; "
            f"unmatched {record['case_unmatched']} {record.get('unmatched_reasons', {})}"
        )
        if record.get("pairs"):
            print(
                f"         controls per case mean {record['controls_per_case_mean']:.2f} "
                f"(median {record['controls_per_case_median']:.0f}, "
                f"min {record['controls_per_case_min']}, max {record['controls_per_case_max']}); "
                f"{record['strata_with_at_least_two_controls']} strata carry >=2 controls "
                "and can contribute to the null"
            )
            print(
                f"         outcome held constant: case success rate "
                f"{record['case_success_rate']:.3f}, control success rate "
                f"{record['control_success_rate']:.3f}"
            )
            print(
                f"         |d eef| mean {record['abs_eef_difference_m_mean'] * 1000:.1f} mm "
                f"(max {record['abs_eef_difference_m_max'] * 1000:.1f} mm); "
                f"|d query| mean {record['abs_query_difference_mean']:.2f} "
                f"(max {record['abs_query_difference_max']}); "
                f"control reuse max {record['control_reuse_max']}x"
            )
            print(
                f"         signed residual: eef "
                f"{record['signed_eef_difference_m_mean'] * 1000:+.1f} mm, query "
                f"{record['signed_query_difference_mean']:+.2f}"
            )
            print(
                f"         target displacement median: case "
                f"{record['phantom_target_displacement_m_median'] * 1000:.2f} mm vs control "
                f"{record['control_target_displacement_m_median'] * 1000:.1f} mm"
            )

    if not separation.empty:
        print(
            f"\n{'-' * 96}\nLAYER x RELATIVE-QUERY GRID  "
            f"(design=primary, scope=both, bootstrap cluster={PRIMARY_CLUSTER})\n{'-' * 96}"
        )
        for metric in METRICS:
            for role in TOKEN_ROLES:
                print_grid(separation, "primary", "both", PRIMARY_CLUSTER, metric, role)
        print_detail(separation, "primary", "both", PRIMARY_CLUSTER, "gate_entropy", "action")
        print_detail(
            separation, "primary", "both", PRIMARY_CLUSTER, "hellinger_to_pre", "action"
        )

    earliest = {}
    for name in designs:
        for scope in ("both", *sorted(RUN_IDS)):
            for unit in (CLUSTER_UNITS if scope == "both" else (PRIMARY_CLUSTER,)):
                if separation.empty or select(separation, name, scope, unit).empty:
                    continue
                earliest[f"{name}|{scope}|{unit}"] = earliest_exceeding(
                    separation, name, scope, unit
                )
    print("\n[earliest relative query beating the failed-vs-failed null]")
    for key, value in earliest.items():
        print(
            f"  {key:<44} any={value['any']}  "
            f"any(rel>=0)={value['any_excluding_negative_control']}  "
            f"BH-controlled={value['any_bh_controlled']}"
        )

    negative_control: dict[str, Any] = {}
    if not separation.empty:
        for name in designs:
            for unit in CLUSTER_UNITS:
                frame = select(separation, name, "both", unit)
                frame = frame[frame["relative_query"] == -1]
                if frame.empty:
                    continue
                negative_control[f"{name}|{unit}"] = {
                    "cells": int(len(frame)),
                    "cells_beating_null_paired": int(frame["exceeds_null_paired"].sum()),
                    "cells_beating_null_bh": int(frame["exceeds_null_bh"].sum()),
                    "max_abs_auc": float(frame["auc"].sub(0.5).abs().max()),
                    "beating_cells": [
                        f"{row['layer']}/{row['token_role']}/{row['metric']} "
                        f"auc={row['auc']:.3f} [{row['auc_lo']:.3f},{row['auc_hi']:.3f}]"
                        for row in frame[frame["exceeds_null_paired"]].to_dict("records")
                    ],
                }
    print(
        "\n[negative control at relative query -1: gripper still open, "
        "the object cannot yet have responded]"
    )
    for name, record in negative_control.items():
        print(
            f"  {name:<24} {record['cells_beating_null_paired']}/{record['cells']} cells beat "
            f"the null ({record['cells_beating_null_bh']} after BH), "
            f"max |AUC-0.5| = {record['max_abs_auc']:.3f}"
        )
        for entry in record["beating_cells"]:
            print(f"      {entry}")

    # ------------------------------------------------------- side by side
    sibling = None
    if args.sibling.is_file():
        sibling = pd.read_csv(args.sibling)
    comparison = side_by_side(separation, sibling, PRIMARY_CLUSTER) if not separation.empty else pd.DataFrame()

    sensitivity: dict[str, Any] = {}
    signed: dict[str, Any] = {}
    if not separation.empty:
        for name in designs:
            frame = select(separation, name, "both", PRIMARY_CLUSTER)
            sensitivity[f"this|{name}"] = median_abs_auc_late_action(frame)
            signed[f"this|{name}"] = signed_median_auc_late_action(frame)
        frame = select(separation, "primary", "both", "run_task")
        sensitivity["this|primary(run_task cluster)"] = median_abs_auc_late_action(frame)
    if sibling is not None:
        for name in ("primary", "failed_controls", "wide"):
            frame = sibling[(sibling["design"] == name) & (sibling["scope"] == "both")]
            if frame.empty:
                continue
            sensitivity[f"sibling|{name}"] = median_abs_auc_late_action(frame)
            signed[f"sibling|{name}"] = signed_median_auc_late_action(frame)

    print(
        "\n[side by side] median |AUC-0.5| over action tokens at relative queries 1..5\n"
        "  (the sibling's own sensitivity statistic, so the attenuation reads directly)"
    )
    header = sorted({metric for record in sensitivity.values() for metric in record})
    print("    design" + "".join(f"{metric:>24}" for metric in header))
    for name, record in sensitivity.items():
        print(
            f"    {name:<32}"
            + "".join(f"{record.get(metric, float('nan')):>24.3f}" for metric in header)
        )
    print("\n  signed median AUC over the same cells (below 0.5 = phantom routing sharper)")
    print("    design" + "".join(f"{metric:>24}" for metric in header))
    for name, record in signed.items():
        print(
            f"    {name:<32}"
            + "".join(f"{record.get(metric, float('nan')):>24.3f}" for metric in header)
        )

    strongest: list[dict[str, Any]] = []
    if not comparison.empty and "auc_sibling_primary" in comparison:
        print(
            "\n[side by side] the strongest cell of each metric x role in THIS design, "
            "with the sibling at the identical cell"
        )
        print(
            "    metric               role    layer rel |   this AUC [CI]         "
            "sib primary   sib failed"
        )
        merged = comparison.copy()
        merged["abs"] = (merged["auc_failed_matched"] - 0.5).abs()
        for (metric, role), group in merged.groupby(["metric", "token_role"], sort=True):
            row = group.loc[group["abs"].idxmax()]
            strongest.append(
                {
                    "metric": str(metric),
                    "token_role": str(role),
                    "layer": str(row["layer"]),
                    "relative_query": int(row["relative_query"]),
                    "this_auc": float(row["auc_failed_matched"]),
                    "this_ci": [float(row["lo_failed_matched"]), float(row["hi_failed_matched"])],
                    "this_null": float(row["null_failed_matched"]),
                    "this_q_bh": float(row["q_failed_matched"]),
                    "sibling_primary_auc": float(row.get("auc_sibling_primary", np.nan)),
                    "sibling_failed_auc": float(row.get("auc_sibling_failed", np.nan)),
                }
            )
            print(
                f"    {metric:<20} {role:<7} {row['layer']:<5} "
                f"{int(row['relative_query']):>3} | "
                f"{row['auc_failed_matched']:.3f} "
                f"[{row['lo_failed_matched']:.3f},{row['hi_failed_matched']:.3f}]  "
                f"{row.get('auc_sibling_primary', float('nan')):>12.3f} "
                f"{row.get('auc_sibling_failed', float('nan')):>12.3f}"
            )

    # How the strongest cell is built out of tasks.  With 10 clusters an AUC can
    # be carried by one or two tasks, and the interval alone does not say which,
    # so the per-task decomposition is printed rather than summarised.
    per_task_record: dict[str, Any] = {}
    if headline_data and not separation.empty:
        head = select(separation, "primary", "both", PRIMARY_CLUSTER)
        best = head.loc[head["auc"].sub(0.5).abs().idxmax()]
        column = cell_index(
            METRICS.index(str(best["metric"])),
            LAYER_NAMES.index(str(best["layer"])),
            TOKEN_ROLES.index(str(best["token_role"])),
            RELATIVE_QUERIES.index(int(best["relative_query"])),
        )
        per_task = (
            pd.DataFrame(
                {
                    "task": headline_data["strata"]["task"].to_numpy(),
                    "score": headline_data["case"][:, column],
                }
            )
            .dropna()
            .groupby("task")["score"]
            .agg(["mean", "size"])
        )
        side = int(((per_task["mean"] - 0.5) * np.sign(best["auc"] - 0.5) > 0).sum())
        per_task_record = {
            "cell": (
                f"{best['layer']}/{best['token_role']}/rel{int(best['relative_query'])}/"
                f"{best['metric']}"
            ),
            "auc": float(best["auc"]),
            "auc_ci": [float(best["auc_lo"]), float(best["auc_hi"])],
            "tasks_on_the_same_side": side,
            "tasks_total": int(len(per_task)),
            "per_task": {
                str(task): {"auc": float(row["mean"]), "n_cases": int(row["size"])}
                for task, row in per_task.iterrows()
            },
        }
        print(
            f"\n[per task] strongest cell {per_task_record['cell']}: AUC "
            f"{best['auc']:.3f} [{best['auc_lo']:.3f},{best['auc_hi']:.3f}] built from "
            f"{side}/{len(per_task)} tasks on the same side of 0.5"
        )
        for task, row in per_task.iterrows():
            print(f"    {str(task)[:66]:<68} {row['mean']:.3f}  n={int(row['size'])}")

    direction = (
        direction_record(select(separation, "primary", "both", PRIMARY_CLUSTER))
        if not separation.empty
        else {}
    )
    print("\n[direction] gate entropy, relative queries 0..5 (AUC < 0.5 = phantom sharper)")
    for role, record in direction.items():
        print(
            f"  {role:<7} {record['cells_below_half']}/{record['cells']} cells below 0.5; "
            f"median {record['median_auc']:.3f}, min {record['min_auc']:.3f} at "
            f"{record['min_auc_cell']} "
            f"[{record['min_auc_ci'][0]:.3f},{record['min_auc_ci'][1]:.3f}]; "
            f"of the {record['cells_beating_null']} cells beating the null, "
            f"{record['cells_beating_null_below_half']} are below 0.5"
        )

    per_grid: dict[str, Any] = {}
    if not separation.empty:
        print("\n[per grid] cells beating the null")
        for name in designs:
            for scope in ("both", *sorted(RUN_IDS)):
                for unit in (CLUSTER_UNITS if scope == "both" else (PRIMARY_CLUSTER,)):
                    frame = select(separation, name, scope, unit)
                    if frame.empty:
                        continue
                    best = frame.loc[frame["auc"].sub(0.5).abs().idxmax()]
                    record = {
                        "cells": int(len(frame)),
                        "cells_beating_null_paired": int(frame["exceeds_null_paired"].sum()),
                        "cells_beating_null_disjoint": int(frame["exceeds_null_disjoint"].sum()),
                        "cells_beating_null_bh": int(frame["exceeds_null_bh"].sum()),
                        "cells_expected_by_chance": round(len(frame) * 0.025, 1),
                        "max_abs_auc": float(frame["auc"].sub(0.5).abs().max()),
                        "min_p_boot": float(frame["p_boot"].min()),
                        "min_q_bh": float(frame["q_bh"].min()),
                        "best_cell": (
                            f"{best['layer']}/{best['token_role']}/"
                            f"rel{int(best['relative_query'])}/{best['metric']}"
                        ),
                        "best_auc": float(best["auc"]),
                        "best_ci": [float(best["auc_lo"]), float(best["auc_hi"])],
                        "null_median_ci_half_width": float(
                            ((frame["null_hi"] - frame["null_lo"]) / 2).median()
                        ),
                        "case_median_ci_half_width": float(
                            ((frame["auc_hi"] - frame["auc_lo"]) / 2).median()
                        ),
                        "n_clusters": int(frame["n_clusters"].max()),
                    }
                    per_grid[f"{name}|{scope}|{unit}"] = record
                    print(
                        f"  {name}|{scope}|{unit:<9} "
                        f"{record['cells_beating_null_paired']:>3}/{record['cells']} paired, "
                        f"{record['cells_beating_null_disjoint']:>3} disjoint, "
                        f"{record['cells_beating_null_bh']:>3} after BH "
                        f"({record['cells_expected_by_chance']:.1f} expected); "
                        f"best {record['best_auc']:.3f} "
                        f"[{record['best_ci'][0]:.3f},{record['best_ci'][1]:.3f}] at "
                        f"{record['best_cell']}; clusters={record['n_clusters']}; "
                        f"min p={record['min_p_boot']:.5f}"
                    )

    # ------------------------------------------------------- the verdict
    verdict: dict[str, Any] = {}
    if not separation.empty and sibling is not None:
        here_late = sensitivity.get("this|primary", {})
        sib_primary_late = sensitivity.get("sibling|primary", {})
        sib_failed_late = sensitivity.get("sibling|failed_controls", {})
        grid = per_grid.get(f"primary|both|{PRIMARY_CLUSTER}", {})
        null_band = grid.get("null_median_ci_half_width", np.nan)
        separates = grid.get("cells_beating_null_disjoint", 0) > 2 * grid.get(
            "cells_expected_by_chance", 0
        )
        retention = {
            metric: (
                round(here_late[metric] / sib_primary_late[metric], 3)
                if sib_primary_late.get(metric)
                else None
            )
            for metric in here_late
            if metric in sib_primary_late
        }
        # Two questions that a single label would blur, so they get two answers.
        #
        # *Magnitude* asks how much of the sibling's effect is still there once
        # episode outcome is held constant.  It is a point-estimate question and
        # the matched design answers it directly.
        #
        # *Evidence* asks whether that magnitude is distinguishable from the
        # failed-vs-failed null with the clusters available.  It is a power
        # question, and losing it is not the same as losing the effect -- the
        # matched design has 10 task clusters, and an interval that includes the
        # null at 10 clusters is not a finding of no effect.
        finite_retention = [value for value in retention.values() if value is not None]
        worst = min(finite_retention) if finite_retention else None
        if worst is None:
            magnitude_row = "unknown"
        elif worst >= 0.85:
            magnitude_row = "comparable_to_the_sibling_primary"
        elif worst >= 0.35:
            magnitude_row = "attenuated_but_present"
        else:
            magnitude_row = "collapsed"
        evidence_row = (
            "beats_the_null_above_chance" if separates else "not_distinguishable_from_the_null"
        )
        if magnitude_row == "collapsed" and not separates:
            row = "collapses_to_the_null"
        elif magnitude_row == "comparable_to_the_sibling_primary" and separates:
            row = "survives_at_comparable_magnitude"
        elif separates:
            row = "survives_attenuated"
        else:
            row = "magnitude_survives_attenuated_but_underpowered_at_this_cluster_count"
        secondary = per_grid.get("primary|both|run_task", {})
        verdict = {
            "row": row,
            "magnitude_row": magnitude_row,
            "evidence_row": evidence_row,
            "rule": (
                "magnitude is the worst per-metric ratio of this design's median |AUC-0.5| "
                "over action tokens at relative queries 1..5 to the sibling primary's: "
                ">=0.85 comparable, >=0.35 attenuated, below that collapsed.  evidence is "
                "whether more than twice the chance number of cells (0.025 x 336 = 8.4) have "
                "non-overlapping case and null intervals.  The two are reported separately "
                "because at 10 task clusters an interval that includes the null is a power "
                "statement, not a finding of no effect."
            ),
            "median_abs_auc_late_action": {
                "this_failed_matched": here_late,
                "sibling_primary": sib_primary_late,
                "sibling_failed_controls": sib_failed_late,
            },
            "retention_vs_sibling_primary": retention,
            "attributable_to_episode_outcome": {
                metric: (round(1.0 - value, 3) if value is not None else None)
                for metric, value in retention.items()
            },
            "attribution_note": (
                "1 - retention is the share of the sibling primary magnitude that does NOT "
                "survive holding episode outcome constant.  It is a decomposition of point "
                "estimates, not a test, and the two designs match different control pools, so "
                "read it as an order of magnitude rather than a coefficient."
            ),
            "cells_beating_null_paired": grid.get("cells_beating_null_paired"),
            "cells_beating_null_disjoint": grid.get("cells_beating_null_disjoint"),
            "cells_beating_null_bh": grid.get("cells_beating_null_bh"),
            "cells_expected_by_chance": grid.get("cells_expected_by_chance"),
            "clusters": grid.get("n_clusters"),
            "max_abs_auc": grid.get("max_abs_auc"),
            "null_median_band_half_width": null_band,
            "first_separating_relative_query": earliest.get(
                f"primary|both|{PRIMARY_CLUSTER}", {}
            ).get("any"),
            "secondary_run_task_cluster": {
                "clusters": secondary.get("n_clusters"),
                "cells_beating_null_paired": secondary.get("cells_beating_null_paired"),
                "cells_beating_null_disjoint": secondary.get("cells_beating_null_disjoint"),
                "cells_beating_null_bh": secondary.get("cells_beating_null_bh"),
                "caveat": (
                    "less conservative by construction, and its negative control is not clean, "
                    "which is itself a reason to prefer the task-clustered headline"
                ),
            },
            "negative_control_clean_primary_cluster": all(
                record["cells_beating_null_paired"] == 0
                for key, record in negative_control.items()
                if key.endswith(f"|{PRIMARY_CLUSTER}")
            ),
            "negative_control_clean_all_variants": all(
                record["cells_beating_null_paired"] == 0
                for record in negative_control.values()
            ),
            "direction_phantom_sharper": bool(
                direction.get("action", {}).get("median_auc", 1.0) < 0.5
            ),
            "direction_note": (
                "gate-entropy AUC below 0.5 means phantom routing is sharper than coupled "
                "routing.  The model is not confused by the object that stayed behind; it is "
                "confidently continuing a transport it is no longer performing."
            ),
            "bh_resolution": (
                f"the bootstrap p-value floors at 1/{args.draws} = {1.0 / args.draws:.1e}, "
                f"below the {0.05 / N_CELL:.2e} that BH over {N_CELL} cells needs, so a BH "
                "rejection here is an evidence statement rather than a resolution artefact, "
                "and a BH non-rejection is no longer explained away by the p-floor"
            ),
        }
        print(f"\n[verdict] {verdict['row']}")
        print(f"  magnitude: {magnitude_row}    evidence: {evidence_row}")
        print(f"  retention of the sibling primary magnitude: {retention}")
        print(
            "  share of the sibling primary magnitude NOT surviving outcome control: "
            f"{verdict['attributable_to_episode_outcome']}"
        )
        print(
            f"  negative control clean at cluster={PRIMARY_CLUSTER}: "
            f"{verdict['negative_control_clean_primary_cluster']}; across all cluster "
            f"variants: {verdict['negative_control_clean_all_variants']}"
        )
        print(
            "  phantom routing sharper (gate-entropy action median AUC < 0.5): "
            f"{verdict['direction_phantom_sharper']}"
        )

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "once both groups are failing episodes, does MoE routing still separate "
            "object-followed from object-stayed, and at what relative query"
        ),
        "why": (
            "the sibling diagnostic matched 232 phantom events, all inside failed episodes, "
            "against a coupled pool that is 96% successful; part of its separation could be "
            "episode outcome.  Here both groups are inside failed episodes, so outcome is "
            "held constant by construction and cannot contribute."
        ),
        "inputs": {
            "events": str(args.events),
            "events_rows": int(len(events)),
            "route_root": str(args.route_root),
            "run_ids": RUN_IDS,
            "sibling_separation": str(args.sibling) if args.sibling.is_file() else None,
        },
        "groups": {
            "cases": int(len(phantom)),
            "case_success_rate": case_success,
            "coupled_all_with_reference": int(len(coupled_all)),
            "coupled_success_rate": float(coupled_all["success"].mean()),
            "controls_failed_coupled": int(len(control)),
            "coupled_without_reference_query": control_excluded,
            "groups_disjoint": True,
            "case_tasks": int(phantom["task"].nunique()),
            "cases_by_run": phantom["run"].value_counts().to_dict(),
            "pool_run_task_cells_with_both_groups": len(pool_cells),
            "pool_distinct_tasks_with_both_groups": len({task for _, task in pool_cells}),
            "eef_overlap_band_m": [band[0], band[1]],
            "cases_in_overlap_band": in_band(phantom),
            "failed_coupled_in_overlap_band": in_band(control),
            "case_post_closure_query_median": float(
                phantom["post_closure_state_query"].median()
            ),
            "control_post_closure_query_median": float(
                control["post_closure_state_query"].median()
            ),
        },
        "readout": {
            "array": "hb_router_probs",
            "final_flow": FINAL_FLOW,
            "layers": list(LAYER_NAMES),
            "token_roles": {"state": [STATE_TOKEN], "action": list(range(1, N_TOKEN))},
            "relative_queries": list(RELATIVE_QUERIES),
            "reference_relative_query": REFERENCE_RELATIVE,
            "metrics": list(METRICS),
            "soft_only": (
                "hard top-k identity is avoided: a companion audit found 21% of "
                "(query, layer, token) triples have p_(4) and p_(5) in the same float16 "
                "code, so the top-k set there is decided by expert numbering"
            ),
        },
        "matching": {
            "designs": {name: bundle["config"] for name, bundle in designs.items()},
            "max_controls": MAX_CONTROLS,
            "stratum": "(run, task)",
            "control_from_different_episode": True,
            "control_pool": "coupled_target_motion AND episode failed",
            "with_replacement_across_strata": True,
            "balance": {name: bundle.get("balance", {}) for name, bundle in designs.items()},
        },
        "null": {
            "construction": (
                "failed-coupled versus failed-coupled inside each matched stratum: every "
                "bootstrap draw promotes one uniformly chosen control per stratum into the "
                "case's seat and scores it against that stratum's remaining controls, so the "
                "null statistic has exactly the case statistic's structure"
            ),
            "why_not_leave_one_out": (
                "averaging over all promotions is exactly 0.5 in every stratum by "
                "antisymmetry; it is used here only as an unbiasedness check"
            ),
            "paired_test": (
                "delta_abs = |AUC_b - 0.5| - |null_b - 0.5| on the same resample b; "
                "exceeds_null_paired is delta_abs_lo > 0"
            ),
        },
        "bootstrap": {
            "draws": int(args.draws),
            "cluster_primary": PRIMARY_CLUSTER,
            "cluster_secondary": "run_task",
            "cluster_rationale": (
                "one LIBERO task under two seed batches is one scene and one object, not two "
                "independent clusters; the run_task variant is the less conservative "
                "secondary reading"
            ),
            "seed": int(args.seed),
            "interval": "percentile 2.5/97.5",
            "p_floor": 1.0 / int(args.draws),
            "bh_threshold_needed": 0.05 / N_CELL,
        },
        "coverage": coverage,
        "scopes": scope_records,
        "per_grid": per_grid,
        "earliest_exceeding_null": earliest,
        "negative_control_relative_minus_one": {
            "meaning": (
                "at relative query -1 the gripper is still open and the object cannot yet "
                "have responded; separation there is residual matching imbalance, not object "
                "readout, and it caps the attribution of the later separation"
            ),
            "by_design_and_cluster": negative_control,
        },
        "direction_gate_entropy": {
            "meaning": (
                "AUC below 0.5 means phantom routing is SHARPER than coupled routing -- the "
                "model is not confused but confidently continuing a transport it is no longer "
                "performing"
            ),
            "by_role": direction,
        },
        "side_by_side_median_abs_auc_late_action": sensitivity,
        "side_by_side_signed_median_auc_late_action": signed,
        "side_by_side_strongest_cells": strongest,
        "strongest_cell_per_task": per_task_record,
        "interpretation": verdict,
        "design_premise_check": {
            "claim": (
                "pooling both runs was expected to give ~28 (run, task) clusters and ~17 "
                "distinct tasks, against the 10 clusters of the sibling's failed_controls "
                "sensitivity"
            ),
            "pool_run_task_cells": len(pool_cells),
            "pool_distinct_tasks": len({task for _, task in pool_cells}),
            "matched_run_task_cells_primary": int(
                designs["primary"]["pairs"]["run_task"].nunique()
            )
            if not designs["primary"]["pairs"].empty
            else 0,
            "matched_distinct_tasks_primary": int(
                designs["primary"]["pairs"]["task"].nunique()
            )
            if not designs["primary"]["pairs"].empty
            else 0,
            "matched_distinct_tasks_wide": int(designs["wide"]["pairs"]["task"].nunique())
            if not designs["wide"]["pairs"].empty
            else 0,
            "finding": (
                "the pool figures are pre-caliper and do not survive matching.  The primary "
                "design here reproduces the sibling's failed_controls design exactly -- same "
                "155 strata, 446 pairs, 10 task clusters -- because the sibling already pooled "
                "both runs at scope=both and used the same calipers.  The power this "
                "experiment adds is therefore NOT extra clusters: it is 10,000 draws instead "
                "of 2,000 (so BH is no longer resolution-limited), the (run, task) secondary "
                "clustering, the wide-caliper variant on the failed-only pool (13 task "
                "clusters), and the relative-query -1 negative control on this pool."
            ),
        },
        "sample_caveat": (
            "the matched design is small and clustered: report the intervals, never the point "
            "estimate alone.  See per_grid[*]['n_clusters'] for the number of bootstrap "
            "clusters actually carrying each grid."
        ),
        "runtime_seconds": round(elapsed, 1),
        "outputs": {
            "matched_pairs": str(args.output / "matched_pairs.csv"),
            "separation": str(args.output / "separation.csv"),
        },
    }
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"\n[done] {elapsed:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
