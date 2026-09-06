#!/usr/bin/env python3
"""Can the MoE routing interface tell "the object came with me" from "it stayed"?

An online monitor in this workspace may read only routing.  Evidence about the
physical world can still reach it, but only along

    world change -> new observation -> internal representation -> MoE reading

and success at any one of those three gates does not imply the next.  This
diagnostic tests the **third** gate with the first two held as fixed as the
existing corpus allows.  The contrast is deliberately narrow:

    with the gripper closed and the arm executing comparable motion, does the
    routing distinguish a coupled object from an object that stayed behind?

That is sharper than success-vs-failure, because the arm behaviour is matched
by design and only the object's response differs.

Design
------
* **Groups.**  From ``analysis_trap_taxonomy`` closure events:
  ``missed_grasp_then_departure`` (phantom; the target never moves) against
  ``coupled_target_motion`` (the target comes along).  The two flags are
  mutually exclusive and that is asserted.
* **Matching.**  Strictly within ``(run, task)`` -- different tasks have
  different objects and scenes, so a cross-task control would confound object
  identity with the visual scene.  Hard calipers on ``eef_max_displacement_
  after_m`` (matched arm travel) *and* on ``post_closure_state_query`` (matched
  position in the episode; without it the phantom group sits at query ~27 and
  the coupled group at ~11, and that confound alone could separate routing).
  Up to ``MAX_CONTROLS`` nearest controls per phantom, from a different
  episode.  Controls may serve more than one phantom -- matching without
  replacement exhausted the caliper neighbourhood and silently dropped 40 more
  phantoms, which is worse than the reuse it avoids.  Reuse is reported.
  Unmatched phantoms are written out with a reason, never dropped in silence.
* **Null.**  The identical pipeline on coupled-versus-coupled: inside each
  matched stratum a control takes the phantom's place and is scored against the
  remaining controls.  This bounds how much routing difference arises from
  irrelevant variation and substitutes for the lighting/background control that
  would need re-rendering.  **Every AUC here must be read against this null,
  not against 0.5.**
* **Readout.**  Raw ``hb_router_probs``, ``(rows, 8, 10, 11, 32)`` =
  ``(row, layer, denoising_step, token, expert)``, at ``FINAL_FLOW = 9``.
  Soft distributions only: a companion audit found 21% of (query, layer, token)
  triples have ``p_(4)`` and ``p_(5)`` in the same float16 code, so a hard
  top-4 identity feature would largely measure expert numbering.
* **Resolution.**  Per layer, per token role (state token 0 against action
  tokens 1..10), per relative query -1..+5 around ``post_closure_state_query``.
  The first relative query whose separation exceeds the null is the estimate of
  first discernible evidence in this interface; a monitor cannot alarm earlier.

Metrics (all soft, all at FINAL_FLOW)
-------------------------------------
``hellinger_to_pre``     Hellinger between the route at the relative query and
                         the route of the *same episode* at ``REFERENCE_
                         RELATIVE`` (two queries before the post-closure
                         state, i.e. strictly before the closing action).  A
                         within-episode change measure, so absolute task scale
                         cancels.
``gate_entropy``         Shannon entropy of the route, normalised by log 32.
``gate_entropy_change``  ``gate_entropy`` minus its value at the reference
                         query -- the within-episode version of the same
                         reading.

Nothing is trained or fitted.  CPU only; zarr is read in batches and reduced
immediately.

Outputs (under ``results/object_readability``):
  matched_pairs.csv  the matched design, one row per pair plus one row per
                     unmatched phantom, with match quality
  separation.csv     design x scope x layer x token role x relative query x
                     metric: AUC, task-clustered CI, n, and the paired
                     coupled-vs-coupled null
  summary.json       headline numbers, balance, provenance, runtime
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
sys.path.insert(0, str(BUNDLE))

from method import progress_ratio  # noqa: E402

ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_EVENTS = (
    WORKSPACE / "analysis_trap_taxonomy/results/belief_mismatch_events.csv.gz"
)
DEFAULT_OUTPUT = BUNDLE / "results/object_readability"

SCHEMA = "himoe.object_readability.v1"

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

# Relative to post_closure_state_query.  -1 is the query at which the closing
# action was emitted, 0 the first state that reflects it.
RELATIVE_QUERIES = (-1, 0, 1, 2, 3, 4, 5)
REFERENCE_RELATIVE = -2

# Caliper on eef_max_displacement_after_m: 0.2 x the pooled SD of that column
# over all closure events (0.1255 m), the conventional Rosenbaum-Rubin choice.
# Caliper on post_closure_state_query in whole queries.
DESIGNS: dict[str, dict[str, Any]] = {
    "primary": {"eef_caliper_m": 0.025, "query_caliper": 3, "failed_controls": False},
    "wide": {"eef_caliper_m": 0.050, "query_caliper": 5, "failed_controls": False},
    # Every phantom event belongs to a failed episode, so a separation found
    # against a mostly-successful control pool could be reading "this episode
    # fails" rather than "the object stayed behind".  This design restricts the
    # controls to coupled closures inside *failed* episodes, at the primary
    # caliper: the object still comes along, the episode still fails.
    "failed_controls": {"eef_caliper_m": 0.025, "query_caliper": 3, "failed_controls": True},
}
MAX_CONTROLS = 5
MIN_CONTROLS_FOR_NULL = 2

BOOTSTRAP_DRAWS = 2000
SEED = 20260906
READ_BATCH = 512


# --------------------------------------------------------------------------
# pure helpers -- no I/O, unit tested in tests/test_object_readability.py
# --------------------------------------------------------------------------


def gate_entropy(probability: np.ndarray) -> np.ndarray:
    """Shannon entropy of the last axis normalised by ``log(n_expert)``.

    Uniform -> 1.0, one-hot -> 0.0, so the reading does not depend on how many
    experts the layer happens to have.
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

    ``state`` reads token 0 alone; ``action`` averages tokens 1..10.  The two
    are reported separately because a pooled score cannot say *where* evidence
    lives.
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
    ``TOKEN_ROLES``.  ``route`` may be all-NaN when the query does not exist in
    the episode, and the features are then NaN.
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
    phantom: pd.DataFrame,
    control: pd.DataFrame,
    *,
    eef_caliper: float,
    query_caliper: float,
    max_controls: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1:M nearest-neighbour matching inside ``(run, task)`` under two calipers.

    ``phantom`` and ``control`` must carry ``event_id``, ``run``, ``task``,
    ``episode``, ``eef_max_displacement_after_m`` and
    ``post_closure_state_query``.

    Rules, all of them hard:
      * a control must share the phantom's ``run`` **and** ``task``;
      * a control must come from a different ``episode``;
      * ``|d eef displacement| <= eef_caliper`` and
        ``|d post-closure query| <= query_caliper``;
      * at most ``max_controls`` controls per phantom, chosen by the smallest
        caliper-standardised distance, ties broken by ``event_id`` so the
        result is deterministic;
      * controls are distinct within a stratum and may be reused across strata.

    Returns ``(pairs, unmatched)``.  Every phantom appears in exactly one of
    the two; none is dropped in silence.
    """
    if eef_caliper <= 0 or query_caliper <= 0:
        raise ValueError("calipers must be positive")
    if max_controls < 1:
        raise ValueError("max_controls must be positive")

    pairs: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []
    control_by_cell = {key: frame for key, frame in control.groupby(["run", "task"], sort=True)}

    for event in phantom.sort_values("event_id").to_dict("records"):
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
                    "phantom_event_id": int(event["event_id"]),
                    "phantom_episode": int(event["episode"]),
                    "phantom_target": event["target"],
                    "phantom_post_closure_state_query": int(event["post_closure_state_query"]),
                    "phantom_eef_displacement_m": float(event["eef_max_displacement_after_m"]),
                    "phantom_target_displacement_m": float(event["target_max_displacement_after_m"]),
                    "phantom_repeated_closure": bool(event["repeated_target_closure"]),
                    "control_event_id": int(row["event_id"]),
                    "control_episode": int(row["episode"]),
                    "control_target": row["target"],
                    "control_post_closure_state_query": int(row["post_closure_state_query"]),
                    "control_eef_displacement_m": float(row["eef_max_displacement_after_m"]),
                    "control_target_displacement_m": float(row["target_max_displacement_after_m"]),
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

    This is the matched-pair AUC contribution of one phantom: every stratum
    carries equal weight regardless of how many controls it found, so a task
    with many controls cannot outvote one with few.
    """
    values = np.asarray(controls, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not np.isfinite(case) or len(values) == 0:
        return np.nan
    return float((case > values).mean() + 0.5 * (case == values).mean())


def stratum_promotion_scores(controls: np.ndarray) -> np.ndarray:
    """Coupled-versus-coupled: promote each control to the phantom's place.

    ``controls`` is ``[n_control]``; the return is ``[n_control]`` where entry
    ``j`` is the score control ``j`` obtains against the *other* controls of the
    same stratum.  Entries whose control is missing, and every entry when fewer
    than ``MIN_CONTROLS_FOR_NULL`` controls survive, are NaN.

    Note that the *average* of these entries is exactly 0.5 for any stratum, by
    the antisymmetry of the pairwise comparison.  The null therefore has no
    information in its centre; all of its information is in the spread produced
    by promoting **one** control, which is why the reported null draws a single
    promotion per stratum rather than enumerating them.
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
    # the diagonal is a control compared with itself: an exact tie worth 0.5
    score = greater.sum(axis=1) + 0.5 * equal.sum(axis=1) - 0.5
    output[finite] = score[finite] / (n_valid - 1)
    return output


def stratum_scores(case: np.ndarray, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over a trailing cell axis.

    ``case`` is ``[n_cell]`` and ``controls`` is ``[n_control, n_cell]``.
    Returns ``(case_score [n_cell], promotion_score [n_control, n_cell])`` with
    exactly the semantics of :func:`stratum_case_score` and
    :func:`stratum_promotion_scores` applied cell by cell.
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
    """``[draws, n_stratum]`` resampling weights that resample whole clusters.

    Strata inside one task are drawn together, because episodes of one task
    share a scene, an object and a policy behaviour and are not independent.
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
        # control_step is a global counter across the run, so only its
        # within-episode increments carry the query index.  Requiring unit
        # increments is what makes "row position == query index" safe.
        steps = control_step[rows]
        if len(steps) > 1 and not np.all(np.diff(steps) == 1):
            raise ValueError(
                f"control_step skips inside episode {episode} in {route_path}"
            )
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
    """Read routing for the requested events and reduce to features immediately.

    ``needed`` are ``event_id`` values.  Returns ``event_id -> [n_metric,
    n_layer, n_role, n_relative]`` and a provenance/coverage record.
    """
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


def stratum_table(
    pairs: pd.DataFrame, features: dict[int, np.ndarray]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-stratum case score and the promotion scores that build the null.

    Returns ``(index_frame, case, promotion)`` with ``case`` of shape
    ``[n_stratum, n_cell]`` and ``promotion`` of shape
    ``[n_stratum, MAX_CONTROLS, n_cell]``; cells are the flattened
    ``(metric, layer, token role, relative query)`` grid.
    """
    strata = (
        pairs.groupby("stratum_id", sort=True)
        .agg(
            run=("run", "first"),
            task=("task", "first"),
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
            [
                features.get(int(cid), missing).reshape(-1).astype(np.float64)
                for cid in ids
            ]
        )
        case_score, promotion_score = stratum_scores(phantom.reshape(-1), stack)
        case[s_index] = case_score
        promotion[s_index, : len(ids)] = promotion_score
    return strata, case, promotion


def separation_rows(
    design: str,
    scope: str,
    strata: pd.DataFrame,
    case: np.ndarray,
    promotion: np.ndarray,
    pairs: pd.DataFrame,
    draws: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """AUC, task-clustered CI and the coupled-vs-coupled null, cell by cell.

    The null is not the leave-one-out average -- that is identically 0.5 by
    antisymmetry and carries no information.  Instead, each bootstrap draw picks
    **one** control per stratum at random to stand in for the phantom, so the
    null statistic has exactly the case statistic's structure and its spread is
    the honest noise floor of the whole pipeline.
    """
    if len(strata) == 0:
        return []
    n_strata = len(strata)
    weights = bootstrap_weights(strata["task"].to_numpy(), draws, rng)
    n_controls = strata["n_controls"].to_numpy(dtype=np.int64)
    promoted = rng.integers(0, np.broadcast_to(n_controls, (draws, n_strata)))
    pair_counts = (
        pairs.groupby("stratum_id", sort=True).size().reindex(strata["stratum_id"]).to_numpy()
    )
    task_of = strata["task"].to_numpy()
    stratum_axis = np.arange(n_strata)[None, :]

    rows: list[dict[str, Any]] = []
    for m, metric in enumerate(METRICS):
        for layer_index, layer in enumerate(LAYER_NAMES):
            for role_index, role in enumerate(TOKEN_ROLES):
                for rel_index, relative in enumerate(RELATIVE_QUERIES):
                    column = cell_index(m, layer_index, role_index, rel_index)
                    case_v = case[:, column]
                    promotion_v = promotion[:, :, column]  # [n_stratum, MAX_CONTROLS]

                    auc, n_case = point_mean(case_v)
                    boot_case = weighted_means(weights, case_v)
                    auc_lo, auc_hi = percentile_interval(boot_case)

                    # one promoted control per stratum per draw
                    null_v = promotion_v[stratum_axis, promoted]  # [draws, n_stratum]
                    finite_null = np.isfinite(null_v)
                    effective = weights * finite_null
                    denominator = effective.sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        boot_null = (effective * np.where(finite_null, null_v, 0.0)).sum(
                            axis=1
                        ) / denominator
                    boot_null[denominator <= 0] = np.nan
                    unweighted = np.where(finite_null, null_v, 0.0).sum(
                        axis=1
                    ) / np.maximum(finite_null.sum(axis=1), 1)
                    unweighted[finite_null.sum(axis=1) == 0] = np.nan
                    null_auc = float(np.nanmean(unweighted)) if np.isfinite(unweighted).any() else np.nan
                    null_lo, null_hi = percentile_interval(boot_null)

                    delta = np.abs(boot_case - 0.5) - np.abs(boot_null - 0.5)
                    delta_lo, delta_hi = percentile_interval(delta)
                    finite_delta = delta[np.isfinite(delta)]
                    p_boot = (
                        max(float((finite_delta <= 0.0).mean()), 1.0 / max(len(finite_delta), 1))
                        if len(finite_delta)
                        else np.nan
                    )
                    with np.errstate(invalid="ignore"):
                        null_abs = (
                            float(np.nanmean(np.abs(boot_null - 0.5)))
                            if np.isfinite(boot_null).any()
                            else np.nan
                        )
                    delta_point = abs(auc - 0.5) - null_abs if np.isfinite(auc) else np.nan

                    finite_case = np.isfinite(case_v)
                    rows.append(
                        {
                            "design": design,
                            "scope": scope,
                            "layer": layer,
                            "token_role": role,
                            "relative_query": int(relative),
                            "metric": metric,
                            "n_strata": int(n_case),
                            "n_pairs": int(pair_counts[finite_case].sum())
                            if finite_case.any()
                            else 0,
                            "n_tasks": int(pd.unique(task_of[finite_case]).size)
                            if finite_case.any()
                            else 0,
                            "auc": auc,
                            "auc_lo": auc_lo,
                            "auc_hi": auc_hi,
                            "auc_abs": abs(auc - 0.5) if np.isfinite(auc) else np.nan,
                            "null_auc": null_auc,
                            "null_lo": null_lo,
                            "null_hi": null_hi,
                            "null_abs_mean": null_abs,
                            "n_strata_null": int(np.isfinite(promotion_v).any(axis=1).sum()),
                            "delta_abs": delta_point,
                            "delta_abs_lo": delta_lo,
                            "delta_abs_hi": delta_hi,
                            "p_boot": p_boot,
                            "exceeds_null_paired": bool(np.isfinite(delta_lo) and delta_lo > 0.0),
                            "exceeds_null_disjoint": bool(
                                np.isfinite(auc_lo)
                                and np.isfinite(null_hi)
                                and (auc_lo > null_hi or auc_hi < null_lo)
                            ),
                        }
                    )
    return rows


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up over one design x scope grid.

    336 cells are reported per grid, and at a 2.5% per-cell criterion roughly 8
    would beat the null by chance alone.  The cells are strongly correlated
    (neighbouring layers and neighbouring relative queries read almost the same
    routing), so BH is conservative here rather than exact, but it keeps the
    headline count from being read as 336 independent tests.
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


def earliest_exceeding(separation: pd.DataFrame, design: str, scope: str) -> dict[str, Any]:
    """First relative query at which any cell beats the null, per metric/role."""
    frame = separation[(separation["design"] == design) & (separation["scope"] == scope)]
    hit = frame[frame["exceeds_null_paired"]]
    strict = frame[frame["exceeds_null_bh"]] if "exceeds_null_bh" in frame else frame.iloc[:0]
    output: dict[str, Any] = {
        "any": None if hit.empty else int(hit["relative_query"].min()),
        "any_bh_controlled": None if strict.empty else int(strict["relative_query"].min()),
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


def print_grid(separation: pd.DataFrame, design: str, scope: str, metric: str, role: str) -> None:
    frame = separation[
        (separation["design"] == design)
        & (separation["scope"] == scope)
        & (separation["metric"] == metric)
        & (separation["token_role"] == role)
    ]
    if frame.empty:
        return
    print(
        f"\n  {metric} / {role} token(s)  --  AUC * |null band half-width;"
        "  * = beats the coupled-vs-coupled null"
    )
    header = "    layer " + "".join(f"{rel:>13d}" for rel in RELATIVE_QUERIES)
    print(header)
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


def print_detail(separation: pd.DataFrame, design: str, scope: str, metric: str, role: str) -> None:
    frame = separation[
        (separation["design"] == design)
        & (separation["scope"] == scope)
        & (separation["metric"] == metric)
        & (separation["token_role"] == role)
    ]
    if frame.empty:
        return
    print(f"\n  detail: {metric} / {role}, scope={scope}, design={design}")
    print(
        "    layer  rel        AUC [95% CI]              null [95% CI]          "
        "delta|.-0.5| [95% CI]      n"
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
                f"{int(row['n_strata']):>4}"
            )


def balance_record(pairs: pd.DataFrame, unmatched: pd.DataFrame, n_phantom: int) -> dict[str, Any]:
    if pairs.empty:
        return {
            "phantom_total": int(n_phantom),
            "phantom_matched": 0,
            "phantom_unmatched": int(len(unmatched)),
        }
    per_stratum = pairs.groupby("stratum_id").size()
    reuse = pairs["control_event_id"].value_counts()
    return {
        "phantom_total": int(n_phantom),
        "phantom_matched": int(pairs["stratum_id"].nunique()),
        "phantom_unmatched": int(len(unmatched)),
        "unmatched_reasons": (
            unmatched["drop_reason"].value_counts().to_dict() if len(unmatched) else {}
        ),
        "pairs": int(len(pairs)),
        "controls_per_phantom_mean": float(per_stratum.mean()),
        "controls_per_phantom_min": int(per_stratum.min()),
        "distinct_control_events": int(pairs["control_event_id"].nunique()),
        "control_reuse_max": int(reuse.max()),
        "control_reuse_mean": float(reuse.mean()),
        "tasks": int(pairs["task"].nunique()),
        "runs": sorted(pairs["run"].unique().tolist()),
        "abs_eef_difference_m_mean": float(pairs["abs_eef_difference_m"].mean()),
        "abs_eef_difference_m_max": float(pairs["abs_eef_difference_m"].max()),
        "abs_query_difference_mean": float(pairs["abs_query_difference"].mean()),
        "abs_query_difference_max": int(pairs["abs_query_difference"].max()),
        # The absolute differences say the caliper held; the signed ones say
        # whether it held symmetrically.  Phantom closures sit late in their
        # episodes, so the eligible controls tend to lie on the earlier side and
        # a residual signed gap survives.  The relative-query -1 negative
        # control is what shows that gap does not by itself separate routing.
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
        # Control medians are taken over the per-stratum mean, so a phantom with
        # five controls does not weigh five times as much as one with a single
        # control -- otherwise the comparison with the phantom column is unfair.
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


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--route-root", type=Path, default=ROUTE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--batch", type=int, default=READ_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)

    events = load_events(args.events)
    phantom = events[events["missed_grasp_then_departure"]].copy()
    control = events[
        events["coupled_target_motion"]
        & (events["post_closure_state_query"] >= -REFERENCE_RELATIVE)
    ].copy()
    control_excluded = int(events["coupled_target_motion"].sum()) - len(control)

    print(
        f"[design] phantom={len(phantom)} coupled={len(control)} "
        f"(coupled without a pre-closure reference query: {control_excluded})",
        flush=True,
    )

    designs: dict[str, dict[str, Any]] = {}
    all_pairs: list[pd.DataFrame] = []
    all_unmatched: list[pd.DataFrame] = []
    for name, config in DESIGNS.items():
        pool = control[~control["success"]] if config.get("failed_controls") else control
        pairs, unmatched = match_events(
            phantom,
            pool,
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
            f"|d query|<={config['query_caliper']}: "
            f"matched {pairs['stratum_id'].nunique()}/{len(phantom)} phantom events, "
            f"{len(pairs)} pairs, {len(unmatched)} unmatched",
            flush=True,
        )

    needed = np.unique(
        np.concatenate(
            [phantom["event_id"].to_numpy()]
            + [frame["control_event_id"].to_numpy() for frame in all_pairs if len(frame)]
        )
    )
    print(f"[read] {len(needed)} distinct events, "
          f"{len(needed) * (len(RELATIVE_QUERIES) + 1)} query slots", flush=True)
    read_started = time.perf_counter()
    features, coverage = collect_features(
        events, needed, args.route_root, args.batch, not args.quiet
    )
    read_elapsed = time.perf_counter() - read_started
    print(
        f"[read] done in {read_elapsed:.1f}s; "
        f"{coverage['query_slots_outside_episode']} of "
        f"{coverage['query_slots_requested']} query slots fall outside their episode",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    scope_records: dict[str, Any] = {}
    headline_data: dict[str, Any] = {}
    for name, bundle in designs.items():
        pairs = bundle["pairs"]
        if pairs.empty:
            continue
        for scope in ("both", *sorted(RUN_IDS)):
            subset = pairs if scope == "both" else pairs[pairs["run"] == scope]
            if subset.empty:
                continue
            strata, case, promotion = stratum_table(subset, features)
            rng = np.random.default_rng(args.seed)
            rows.extend(
                separation_rows(
                    name, scope, strata, case, promotion, subset, args.draws, rng
                )
            )
            if name == "primary" and scope == "both":
                headline_data = {"strata": strata, "case": case}
                leave_one_out = np.nanmean(promotion, axis=1)
                deviation = np.nanmax(np.abs(np.nanmean(leave_one_out, axis=0) - 0.5))
                if deviation > 1e-9:
                    raise ValueError(
                        "leave-one-out coupled-vs-coupled is not 0.5; the matched-pair "
                        f"estimator is biased (max deviation {deviation:.3e})"
                    )
                scope_records["null_symmetry_max_deviation"] = float(deviation)
            scope_records[f"{name}|{scope}"] = {
                "strata": int(len(strata)),
                "pairs": int(len(subset)),
                "tasks": int(strata["task"].nunique()),
            }
    separation = pd.DataFrame(rows)
    if not separation.empty:
        separation["q_bh"] = np.nan
        for _, index in separation.groupby(["design", "scope"], sort=False).groups.items():
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
    print("\n" + "=" * 96)
    print("OBJECT-READABILITY OF THE MoE ROUTING INTERFACE")
    print("phantom (target never moves) vs coupled (target comes along), arm travel matched")
    print("=" * 96)
    for name, bundle in designs.items():
        record = balance_record(bundle["pairs"], bundle["unmatched"], len(phantom))
        designs[name]["balance"] = record
        print(
            f"\n[{name}] matched {record['phantom_matched']}/{record['phantom_total']} phantom, "
            f"{record.get('pairs', 0)} pairs, {record.get('tasks', 0)} tasks; "
            f"unmatched {record['phantom_unmatched']} {record.get('unmatched_reasons', {})}"
        )
        if record.get("pairs"):
            print(
                f"         |d eef| mean {record['abs_eef_difference_m_mean']*1000:.1f} mm "
                f"(max {record['abs_eef_difference_m_max']*1000:.1f} mm); "
                f"|d query| mean {record['abs_query_difference_mean']:.2f} "
                f"(max {record['abs_query_difference_max']}); "
                f"control reuse max {record['control_reuse_max']}x"
            )
            print(
                f"         signed residual: eef "
                f"{record['signed_eef_difference_m_mean']*1000:+.1f} mm, query "
                f"{record['signed_query_difference_mean']:+.2f}"
            )
            print(
                f"         target displacement median: phantom "
                f"{record['phantom_target_displacement_m_median']*1000:.2f} mm vs control "
                f"{record['control_target_displacement_m_median']*1000:.1f} mm"
            )

    headline_design = "primary"
    if not separation.empty:
        for metric in METRICS:
            for role in TOKEN_ROLES:
                print_grid(separation, headline_design, "both", metric, role)
        print_detail(separation, headline_design, "both", "hellinger_to_pre", "action")

    earliest = {
        f"{name}|{scope}": earliest_exceeding(separation, name, scope)
        for name in designs
        for scope in ("both", *sorted(RUN_IDS))
        if not separation.empty
        and not separation[
            (separation["design"] == name) & (separation["scope"] == scope)
        ].empty
    }

    print("\n[earliest relative query beating the coupled-vs-coupled null]")
    for key, value in earliest.items():
        print(
            f"  {key:<30} any={value['any']}  "
            f"BH-controlled={value['any_bh_controlled']}"
        )

    # Negative control.  At relative query -1 the gripper is still open and the
    # object cannot yet have responded, so both routes read a pre-closure world.
    # Anything that separates there is a residual difference between the groups
    # -- most plausibly approach quality, since a closure that is about to miss
    # is reached along a different trajectory -- and it caps how much of the
    # later separation can be attributed to the object's response.
    negative_control: dict[str, Any] = {}
    if not separation.empty:
        for design_name in designs:
            frame = separation[
                (separation["design"] == design_name)
                & (separation["scope"] == "both")
                & (separation["relative_query"] == -1)
            ]
            if frame.empty:
                continue
            negative_control[design_name] = {
                "cells": int(len(frame)),
                "cells_beating_null_paired": int(frame["exceeds_null_paired"].sum()),
                "max_abs_auc": float(frame["auc"].sub(0.5).abs().max()),
                "beating_cells": [
                    f"{row['layer']}/{row['token_role']}/{row['metric']}"
                    f" auc={row['auc']:.3f}"
                    for row in frame[frame["exceeds_null_paired"]].to_dict("records")
                ],
            }
    print("\n[negative control at relative query -1: gripper still open, object cannot have moved]")
    for name, record in negative_control.items():
        print(
            f"  {name:<16} {record['cells_beating_null_paired']}/{record['cells']} cells "
            f"beat the null, max |AUC-0.5| = {record['max_abs_auc']:.3f}"
        )
        for entry in record["beating_cells"]:
            print(f"      {entry}")

    if not separation.empty:
        head = separation[
            (separation["design"] == headline_design) & (separation["scope"] == "both")
        ]
        n_cells = len(head)
        n_beat = int(head["exceeds_null_paired"].sum())
        max_row = head.loc[head["auc"].sub(0.5).abs().idxmax()]
        print(
            f"\n[headline] design={headline_design} scope=both: "
            f"{n_beat}/{n_cells} cells beat the null (paired test); "
            f"largest |AUC-0.5| = {abs(max_row['auc']-0.5):.3f} at "
            f"{max_row['layer']}/{max_row['token_role']}/rel{int(max_row['relative_query'])}/"
            f"{max_row['metric']} (AUC {max_row['auc']:.3f} "
            f"[{max_row['auc_lo']:.3f},{max_row['auc_hi']:.3f}], "
            f"null {max_row['null_auc']:.3f} "
            f"[{max_row['null_lo']:.3f},{max_row['null_hi']:.3f}])"
        )
        null_spread = head["null_auc"].sub(0.5).abs().max()
        print(
            f"[headline] coupled-vs-coupled null centres within "
            f"{null_spread:.3f} of 0.5 across all {n_cells} cells; "
            f"null CI half-width median "
            f"{float(((head['null_hi']-head['null_lo'])/2).median()):.3f}, "
            f"max {float(((head['null_hi']-head['null_lo'])/2).max()):.3f}; "
            f"case CI half-width median "
            f"{float(((head['auc_hi']-head['auc_lo'])/2).median()):.3f}"
        )
        for design_name in designs:
            frame = separation[
                (separation["design"] == design_name) & (separation["scope"] == "both")
            ]
            if frame.empty:
                continue
            best = frame.loc[frame["auc"].sub(0.5).abs().idxmax()]
            print(
                f"[{design_name:<16}] {int(frame['exceeds_null_paired'].sum()):>3}/{len(frame)} "
                f"cells beat the null "
                f"({int(frame['exceeds_null_bh'].sum())} after BH, "
                f"{len(frame) * 0.025:.0f} expected by chance); best AUC {best['auc']:.3f} "
                f"[{best['auc_lo']:.3f},{best['auc_hi']:.3f}] at {best['layer']}/"
                f"{best['token_role']}/rel{int(best['relative_query'])}/{best['metric']}"
            )

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "with the gripper closed and arm travel matched, does MoE routing separate "
            "a coupled object from an object that stayed behind"
        ),
        "inputs": {
            "events": str(args.events),
            "events_rows": int(len(events)),
            "route_root": str(args.route_root),
            "run_ids": RUN_IDS,
        },
        "groups": {
            "phantom": int(len(phantom)),
            "coupled": int(len(control)),
            "coupled_without_reference_query": control_excluded,
            "groups_disjoint": True,
            "phantom_tasks": int(phantom["task"].nunique()),
            "phantom_by_run": phantom["run"].value_counts().to_dict(),
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
                "hard top-4 identity is avoided: a companion audit found 21% of "
                "(query, layer, token) triples have p_(4) and p_(5) in the same float16 "
                "code, so the top-4 set there is decided by expert numbering"
            ),
        },
        "matching": {
            "designs": {name: bundle["config"] for name, bundle in designs.items()},
            "max_controls": MAX_CONTROLS,
            "stratum": "(run, task)",
            "control_from_different_episode": True,
            "with_replacement_across_strata": True,
            "balance": {name: bundle.get("balance", {}) for name, bundle in designs.items()},
        },
        "null": {
            "construction": (
                "coupled vs coupled inside each matched stratum: every bootstrap draw promotes "
                "one uniformly chosen control per stratum to the phantom's place and scores it "
                "against that stratum's remaining controls, so the null statistic has exactly "
                "the case statistic's structure"
            ),
            "why_not_leave_one_out": (
                "averaging over all promotions gives exactly 0.5 in every stratum by the "
                "antisymmetry of the pairwise comparison; that average carries no information "
                "and is used here only as an unbiasedness check on the estimator"
            ),
            "null_lo_hi": (
                "2.5/97.5 percentiles over draws that combine task resampling with the random "
                "promotion -- the band of AUC that irrelevant variation alone produces"
            ),
            "paired_test": (
                "delta_abs = |AUC_b - 0.5| - |null_b - 0.5| on the same task resample b; "
                "exceeds_null_paired is delta_abs_lo > 0"
            ),
            "disjoint_test": (
                "exceeds_null_disjoint is the stricter, non-paired criterion that the case and "
                "null intervals do not overlap"
            ),
        },
        "bootstrap": {
            "draws": int(args.draws),
            "cluster": "task",
            "seed": int(args.seed),
            "interval": "percentile 2.5/97.5",
        },
        "coverage": coverage,
        "scopes": scope_records,
        "earliest_exceeding_null": earliest,
        "negative_control_relative_minus_one": {
            "meaning": (
                "at relative query -1 the gripper is still open and the object cannot yet have "
                "responded; separation there is a residual group difference, not object readout, "
                "and it caps the attribution of the later separation"
            ),
            "by_design": negative_control,
        },
        "sample_caveat": (
            "232 phantom events over 18 tasks, heavily concentrated (108 in "
            "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove). This is a small, "
            "clustered sample and the task-clustered intervals are wide accordingly."
        ),
        "runtime_seconds": round(elapsed, 1),
        "outputs": {
            "matched_pairs": str(args.output / "matched_pairs.csv"),
            "separation": str(args.output / "separation.csv"),
        },
    }
    if not separation.empty:
        head = separation[
            (separation["design"] == headline_design) & (separation["scope"] == "both")
        ]
        max_row = head.loc[head["auc"].sub(0.5).abs().idxmax()]
        summary["headline"] = {
            "design": headline_design,
            "scope": "both",
            "cells": int(len(head)),
            "cells_beating_null_paired": int(head["exceeds_null_paired"].sum()),
            "cells_beating_null_disjoint": int(head["exceeds_null_disjoint"].sum()),
            "max_abs_auc_cell": {
                "layer": str(max_row["layer"]),
                "token_role": str(max_row["token_role"]),
                "relative_query": int(max_row["relative_query"]),
                "metric": str(max_row["metric"]),
                "auc": float(max_row["auc"]),
                "auc_ci": [float(max_row["auc_lo"]), float(max_row["auc_hi"])],
                "null_auc": float(max_row["null_auc"]),
                "null_ci": [float(max_row["null_lo"]), float(max_row["null_hi"])],
                "delta_abs_ci": [
                    float(max_row["delta_abs_lo"]),
                    float(max_row["delta_abs_hi"]),
                ],
            },
            "null_max_abs_deviation": float(head["null_auc"].sub(0.5).abs().max()),
            "null_median_ci_half_width": float(
                ((head["null_hi"] - head["null_lo"]) / 2).median()
            ),
            "null_max_ci_half_width": float(((head["null_hi"] - head["null_lo"]) / 2).max()),
            "case_median_ci_half_width": float(((head["auc_hi"] - head["auc_lo"]) / 2).median()),
            "case_max_ci_half_width": float(((head["auc_hi"] - head["auc_lo"]) / 2).max()),
        }
        summary["headline"]["cells_beating_null_bh"] = int(head["exceeds_null_bh"].sum())
        summary["headline"]["cells_expected_by_chance"] = round(len(head) * 0.025, 1)
        summary["per_design_both"] = {
            name: {
                "cells": int(len(frame)),
                "cells_beating_null_paired": int(frame["exceeds_null_paired"].sum()),
                "cells_beating_null_disjoint": int(frame["exceeds_null_disjoint"].sum()),
                "cells_beating_null_bh": int(frame["exceeds_null_bh"].sum()),
                "cells_expected_by_chance": round(len(frame) * 0.025, 1),
                "max_abs_auc": float(frame["auc"].sub(0.5).abs().max()),
            }
            for name in designs
            for frame in [
                separation[(separation["design"] == name) & (separation["scope"] == "both")]
            ]
            if not frame.empty
        }
        if headline_data:
            strata = headline_data["strata"]
            column = cell_index(
                METRICS.index(str(max_row["metric"])),
                LAYER_NAMES.index(str(max_row["layer"])),
                TOKEN_ROLES.index(str(max_row["token_role"])),
                RELATIVE_QUERIES.index(int(max_row["relative_query"])),
            )
            values = headline_data["case"][:, column]
            per_task = (
                pd.DataFrame({"task": strata["task"].to_numpy(), "score": values})
                .dropna()
                .groupby("task")["score"]
                .agg(["mean", "size"])
            )
            summary["headline"]["max_abs_auc_cell"]["per_task"] = {
                str(task): {"auc": float(row["mean"]), "n_phantom": int(row["size"])}
                for task, row in per_task.iterrows()
            }
            summary["headline"]["max_abs_auc_cell"]["tasks_on_the_same_side"] = int(
                ((per_task["mean"] - 0.5) * np.sign(max_row["auc"] - 0.5) > 0).sum()
            )
            summary["headline"]["max_abs_auc_cell"]["tasks_total"] = int(len(per_task))

        # A sensitivity that decides whether the separation is about the object
        # or about the outcome label.  Point estimates, not tests: the
        # failure-matched design has fewer strata and its intervals are wider,
        # so a lost significance test there is a power statement, while a
        # shrunken point estimate would be a confounding statement.
        late_action = separation[
            (separation["scope"] == "both")
            & (separation["token_role"] == "action")
            & (separation["relative_query"].between(1, 5))
        ].copy()
        late_action["abs_auc"] = (late_action["auc"] - 0.5).abs()
        sensitivity = {
            str(name): {
                str(metric): round(float(value), 4)
                for metric, value in group.groupby("metric")["abs_auc"].median().items()
            }
            for name, group in late_action.groupby("design")
        }

        # Which row of the interpretation frame holds.  The rule uses the
        # disjoint-interval count, which does not depend on the bootstrap's
        # p-value resolution, and states the threshold explicitly so the reading
        # cannot drift with the numbers.
        head_primary = summary["per_design_both"]["primary"]
        null_band = summary["headline"]["null_median_ci_half_width"]
        expected = head_primary["cells_expected_by_chance"]
        separates = head_primary["cells_beating_null_disjoint"] > 2 * expected
        null_comparable = head_primary["max_abs_auc"] <= null_band
        first_hit = earliest.get("primary|both", {}).get("any")
        if not separates:
            verdict = "routing_barely_separates"
        elif null_comparable:
            verdict = "separation_large_but_null_comparably_large"
        elif first_hit is not None and first_hit >= 3:
            verdict = "separation_only_at_large_positive_relative_query"
        else:
            verdict = "interface_carries_object_interaction_evidence"
        failed = summary["per_design_both"].get("failed_controls", {})
        summary["interpretation"] = {
            "row": verdict,
            "rule": (
                "carries evidence: more than twice the chance number of cells have case and "
                "null intervals that do not overlap, and the largest |AUC-0.5| exceeds the "
                "median null band half-width, and the first separating relative query is < 3; "
                "late-only if that first query is >= 3; null-comparable if the largest "
                "|AUC-0.5| does not exceed the null band; otherwise barely separates"
            ),
            "first_separating_relative_query": first_hit,
            "max_abs_auc": head_primary["max_abs_auc"],
            "null_median_band_half_width": null_band,
            "negative_control_clean": all(
                record["cells_beating_null_paired"] == 0
                for record in negative_control.values()
            ),
            "failure_matched_sensitivity_median_abs_auc": sensitivity,
            "survives_failed_only_controls": bool(
                failed.get("cells_beating_null_disjoint", 0)
                > 2 * failed.get("cells_expected_by_chance", 0)
            ),
            "multiplicity_caveat": (
                f"the bootstrap p-value cannot fall below 1/{args.draws} = "
                f"{1.0 / args.draws:.2g}, so Benjamini-Hochberg over 336 correlated cells is "
                "resolution-limited and is reported as supporting evidence only; the "
                "disjoint-interval count carries the decision"
            ),
        }
        print(f"\n[interpretation] {verdict}")
        print(
            f"  negative control clean: {summary['interpretation']['negative_control_clean']}; "
            f"survives failure-matched controls: "
            f"{summary['interpretation']['survives_failed_only_controls']}"
        )
        print("  median |AUC-0.5| over action tokens at relative queries 1..5:")
        for name, record in sensitivity.items():
            joined = "  ".join(f"{metric}={value:.3f}" for metric, value in record.items())
            print(f"    {name:<16} {joined}")
    with (args.output / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)

    print(f"\n[done] {elapsed:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
