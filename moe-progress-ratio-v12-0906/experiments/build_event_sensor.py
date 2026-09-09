#!/usr/bin/env python3
"""Routing as an *event sensor*: closure detection, then coupling failure.

Every attempt in this bundle to make routing predict episode outcome has
failed, and the clock ceiling explains why the comparison was ill-posed:
``risk`` is "did not finish before the horizon cap", so every risk episode has
length exactly the cap and the rule "alarm if still running at chunk T" has
recall 1.000 at 0.42%/0.37% timely FPR.  Recall and precision cannot separate
monitors here.  What survives is *lead time at matched false-alarm rate*.

Separately, routing was shown to carry a **local event readout**: with the
gripper closed, arm motion matched, same task and query index matched, routing
tells "the object came along" from "the object stayed behind" at AUC 0.65-0.84,
with the pre-closure negative control at chance.  This script stops asking
routing to predict outcome and asks it to report that event, with a latency and
a false-event rate.

Two stages, both reading routing only at run time:

1. closure detection  - is this rollout at or past gripper closure
2. coupling failure   - given closure, did the object fail to come along

and two timing variants, because the difference is the honest cost of stage 1:

* **oracle timing**  - stage 2 evaluated from the true closure query.
* **routing timing** - stage 1 predicts the closure query, stage 2 runs there.

Windows are deliberately small.  An earlier related script carried window-8
features, which are NaN before chunk 8; a dropna then silently deleted every
row before chunk 8, and grasp closure has median chunk 5.  Nothing wider than
window 2 enters the feature block and every row count is printed.

The operating point is a **fixed false-event rate on truly coupled closures**,
never an outcome-derived precision.  Because phantom closures happen late
(median closure chunk 26.5 for cases against 5 for controls) the closure index
alone is a strong nuisance predictor; it is scored side by side throughout as
``closure_index`` so the routing number is never read on its own.

Run:  python experiments/build_event_sensor.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE))

from method import progress_ratio  # noqa: E402

SCHEMA = "himoe.event_sensor.v1"
ROUTE_ROOT = WORKSPACE / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
EVENTS = WORKSPACE / "analysis_trap_taxonomy/results/belief_mismatch_events.csv.gz"
MOBILITY = WORKSPACE / "moe-v4-0904/results/layerwise_mobility"
DEV_ENTROPY = BUNDLE / "results/ablation/gate_entropy.npz"
DEFAULT_OUTPUT = BUNDLE / "results/event_sensor"

KEY = ["task", "init_state_id", "flow_noise_seed"]
RUN_COHORT = {"seed1000_1007": "development_main", "seed1008_1015": "external_8b"}
COHORTS = {
    "development_main": {
        "cache": BUNDLE / "results/progress_cache/development_main.npz",
        "mobility": MOBILITY / "main_reference.npz",
        "run_id": "right-50x8-20260903",
    },
    "external_8b": {
        "cache": BUNDLE / "results/progress_cache/external_8b.npz",
        "mobility": MOBILITY / "external_8b.npz",
        "run_id": "right-50x8b-20260903",
    },
}

# The clock rule fires at chunk T; these are the per-suite thresholds at 0.5%
# timely FPR, carried over verbatim from the clock-ceiling measurement.
CLOCK_T = {
    "development_main": {
        "libero_goal": 21,
        "libero_long": 44,
        "libero_object": 20,
        "libero_spatial": 14,
    },
    "external_8b": {
        "libero_goal": 21,
        "libero_long": 44,
        "libero_object": 20,
        "libero_spatial": 15,
    },
}

# One action query is a chunk of 10 environment steps: the rollout logs store
# ``actions`` with shape [query, 10, 7] per episode.
CHUNK_TO_STEP = 10

# Relative queries the stage-2 sensor is allowed to look at, counted from the
# closure query.  0..3 keeps 100% of phantom closures and 99.7% of coupled
# closures inside the episode; 0..4 would already drop 8% of the controls.
SCAN = 4
MIN_QUERY = 2  # window-2 features need q >= 2; the earliest closure is q = 3

TARGET_FER = (0.005, 0.01, 0.02)
DRAWS = 2000
SEED = 20260906
EPS_LENGTH = 1e-6

LAYERS = list(progress_ratio.LAYER_NAMES)
LAGS = list(progress_ratio.LAGS)
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")


# --------------------------------------------------------------------------- #
# pure pieces - unit tested in tests/test_event_sensor.py
# --------------------------------------------------------------------------- #


def check_threshold(threshold: float) -> float:
    """A threshold may be ``+inf`` -- the operating point where nothing fires.

    ``threshold_at_false_event_rate`` returns ``+inf`` when even one triggering
    control would blow the budget, and that has to stay a legal, silent sensor
    rather than an exception.  ``NaN`` and ``-inf`` are still errors.
    """
    value = float(threshold)
    if np.isnan(value) or value == -np.inf:
        raise ValueError(f"threshold must be finite or +inf, got {threshold}")
    return value


def first_trigger(scores: np.ndarray, threshold: float) -> int:
    """Index of the first score at or above ``threshold``; -1 if never.

    A sensor commits at the first crossing, so this is a scan and not an
    argmax.  ``-1`` is the never-reported case and every downstream helper has
    to keep carrying it rather than silently dropping the row.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"scores must be one dimensional, got {values.shape}")
    threshold = check_threshold(threshold)
    fired = np.isfinite(values) & (values >= threshold)
    if not fired.any():
        return -1
    return int(np.argmax(fired))


def first_trigger_rows(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Row-wise ``first_trigger`` over a [N, K] score block."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"scores must be [episode, step], got {values.shape}")
    threshold = check_threshold(threshold)
    fired = np.isfinite(values) & (values >= threshold)
    position = np.argmax(fired, axis=1)
    return np.where(fired.any(axis=1), position, -1)


def report_query(scan: np.ndarray, scores: np.ndarray, threshold: float) -> int:
    """Absolute query the sensor reports at; -1 if it never reports."""
    queries = np.asarray(scan, dtype=int)
    if queries.shape != np.asarray(scores).shape:
        raise ValueError("scan queries and scores do not align")
    position = first_trigger(scores, threshold)
    return -1 if position < 0 else int(queries[position])


def report_queries(scan: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Row-wise ``report_query`` over [N, K] scan queries and scores."""
    queries = np.asarray(scan, dtype=int)
    if queries.shape != np.asarray(scores).shape:
        raise ValueError("scan queries and scores do not align")
    position = first_trigger_rows(scores, threshold)
    return np.where(position < 0, -1, queries[np.arange(len(queries)), position])


def latency(report: int, closure: int) -> float:
    """Report query minus closure query; NaN when the sensor never reported.

    Negative latency is meaningful under routing timing: stage 1 can place the
    scan before the true closure, and the sensor can then fire early.
    """
    if int(report) < 0:
        return float("nan")
    return float(int(report) - int(closure))


def latencies(reports: np.ndarray, closures: np.ndarray) -> np.ndarray:
    """Vectorised ``latency``; never-reported rows stay NaN rather than 0."""
    reports = np.asarray(reports, dtype=int)
    closures = np.asarray(closures, dtype=int)
    if reports.shape != closures.shape:
        raise ValueError("reports and closures do not align")
    return np.where(reports < 0, np.nan, (reports - closures).astype(float))


def threshold_at_false_event_rate(control_scores: np.ndarray, target: float) -> float:
    """Lowest threshold whose trigger rate on truly coupled closures is <= target.

    Lowest, because the sensor is monotone in the threshold: any larger value
    triggers on a subset.  Candidates are the observed scores themselves plus
    ``+inf`` for the case where even the single largest control score already
    exceeds the budget, in which case nothing may fire at all.
    """
    values = np.asarray(control_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("control scores must be a non-empty one dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("control scores must be finite")
    if not 0.0 <= target <= 1.0:
        raise ValueError("target false-event rate must lie in [0, 1]")
    ordered = np.unique(values)
    total = float(values.size)
    for candidate in ordered:
        if float((values >= candidate).sum()) / total <= target:
            return float(candidate)
    return float("inf")


def lead_over_clock(report: int, clock: int) -> tuple[bool, int]:
    """Does the sensor land strictly before the clock, and by how many chunks.

    ``report == clock`` is *not* a lead: the clock has already fired at T, so
    matching it buys nothing.  Never-reported (-1) is not a lead either.
    """
    report = int(report)
    clock = int(clock)
    if report < 0 or report >= clock:
        return False, 0
    return True, clock - report


def leads_over_clock(reports: np.ndarray, clocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``lead_over_clock``; ``report == clock`` is not a lead."""
    reports = np.asarray(reports, dtype=int)
    clocks = np.asarray(clocks, dtype=int)
    if reports.shape != clocks.shape:
        raise ValueError("reports and clocks do not align")
    earlier = (reports >= 0) & (reports < clocks)
    return earlier, np.where(earlier, clocks - reports, 0)


def clustered_interval(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator, draws: int = DRAWS
) -> tuple[float, float]:
    """Task-clustered bootstrap percentile interval for a mean.

    232 phantom events sit on 18 tasks and 72% of them on ``libero_long``, so
    an i.i.d. interval would be far too tight.  Tasks are resampled, not rows.
    """
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.shape != clusters.shape:
        raise ValueError("values and clusters do not align")
    if values.size == 0:
        return float("nan"), float("nan")
    names = np.unique(clusters)
    index = {name: np.flatnonzero(clusters == name) for name in names}
    samples = np.empty(draws)
    for draw in range(draws):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = values[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as handle:
        return {name: handle[name] for name in handle.files}


def load_cohort(name: str) -> dict[str, Any]:
    """Progress cache plus the row-aligned mobility file that carries the seeds.

    ``init_state_id`` / ``flow_noise_seed`` live in the v4 mobility file, which
    is only usable if it is the *same rows* as the progress cache.  Every shared
    identity column is compared element-for-element before anything is joined.
    """
    spec = COHORTS[name]
    cache = load_npz(spec["cache"])
    mobility = load_npz(spec["mobility"])
    checked = []
    for column in (
        "task_names",
        "task_index",
        "episode",
        "init_state_id",
        "length",
        "valid",
        "layer_names",
    ):
        same = bool(np.array_equal(np.asarray(cache[column]), np.asarray(mobility[column])))
        checked.append({"array": column, "identical": same})
        if not same:
            raise ValueError(f"{name}: mobility {column} does not align with the progress cache")
    if str(cache["run_id"]) != str(mobility["run_id"]) != spec["run_id"]:
        raise ValueError(f"{name}: run_id disagreement")
    if not np.array_equal(np.asarray(cache["lags"]).astype(int), np.asarray(LAGS)):
        raise ValueError(f"{name}: cache lags {cache['lags']} are not {LAGS}")

    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    index = pd.DataFrame(
        {
            "row": np.arange(len(task_index)),
            "task": task_names[task_index],
            "init_state_id": cache["init_state_id"].astype(int),
            "flow_noise_seed": mobility["flow_noise_seed"].astype(int),
            "length": cache["length"].astype(int),
            "episode": cache["episode"].astype(int),
        }
    )
    index["suite"] = index["task"].str.split("/", n=1).str[0]
    if index.duplicated(KEY).any():
        raise ValueError(f"{name}: (task, init_state_id, flow_noise_seed) is not unique")
    return {
        "name": name,
        "run_id": spec["run_id"],
        "cache": cache,
        "mobility": mobility["mobility"],
        "index": index,
        "alignment": checked,
    }


def gate_entropy(cohort: dict[str, Any], output: Path, refresh: bool) -> tuple[np.ndarray, dict]:
    """[E, Q, 8] normalised gate entropy of the final-flow action routes.

    ``development_main`` already has this cached under ``results/ablation``.
    ``external_8b`` does not, so it is computed here from the raw route store
    exactly the same way -- ``hb_entropy``, final flow step 9, action tokens
    1..10, mean over tokens, divided by log 32 -- and cached alongside this
    experiment's own outputs rather than written back into another experiment's
    directory.
    """
    cache = cohort["cache"]
    expected = (len(cache["episode"]), cache["valid"].shape[1], len(LAYERS))
    if cohort["name"] == "development_main":
        stored = load_npz(DEV_ENTROPY)
        if tuple(stored["entropy"].shape) != expected:
            raise ValueError("cached development gate entropy has the wrong shape")
        if not np.array_equal(stored["episode"], cache["episode"]) or not np.array_equal(
            stored["task_index"], cache["task_index"]
        ):
            raise ValueError("cached development gate entropy does not align row-for-row")
        report = json.loads(str(stored["provenance"]))
        report["origin"] = f"reused {DEV_ENTROPY}"
        return stored["entropy"], report

    destination = output / f"gate_entropy_{cohort['name']}.npz"
    if destination.is_file() and not refresh:
        stored = load_npz(destination)
        if tuple(stored["entropy"].shape) == expected and np.array_equal(
            stored["episode"], cache["episode"]
        ):
            report = json.loads(str(stored["provenance"]))
            report["origin"] = f"reused {destination}"
            return stored["entropy"], report

    import zarr  # imported here so the pure helpers stay importable without zarr

    task_names = cache["task_names"].astype(str)
    task_index = cache["task_index"].astype(int)
    episodes = cache["episode"].astype(int)
    lengths = cache["length"].astype(int)
    run_id = cohort["run_id"]
    entropy = np.full(expected, np.nan, dtype=np.float32)
    normaliser = float(np.log(progress_ratio.N_EXPERTS))
    report = {
        "origin": f"computed from zarr into {destination}",
        "source": "hb_entropy in <route_root>/<task>/<run_id>/server/routes.zarr",
        "route_root": str(ROUTE_ROOT),
        "run_id": run_id,
        "reduction": "final flow step 9, action tokens 1..10, mean over tokens, / log(32)",
        "tasks": int(len(task_names)),
        "episodes_covered": 0,
        "route_distance_check_rows": 0,
        "max_abs_route_distance_delta": None,
    }
    started = time.perf_counter()
    checks: list[float] = []
    rng = np.random.default_rng(SEED)
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
        per_row = logged[
            :, :, progress_ratio.FINAL_FLOW, progress_ratio.ACTION
        ].mean(axis=-1) / normaliser
        lookup = {int(e): int(r) for r, e in zip(rows, episodes[rows])}
        seen: set[int] = set()
        offsets: dict[int, int] = {}
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
            offsets[row] = int(positions[0])
            seen.add(row)
        missing = sorted(set(lookup.values()) - seen)
        if missing:
            raise ValueError(f"{len(missing)} cache rows of {task} absent from {path}")
        report["episodes_covered"] += len(seen)

        # Anchor the cached lag distances to the raw simplex on a sample: the
        # stage-2 reference feature reads them as Hellinger distances against
        # the episode's own pre-closure route, so they must *be* that.
        if position < 3:
            sample_rows = rng.choice(rows, size=min(8, rows.size), replace=False)
            for row in sample_rows:
                length = int(lengths[row])
                if length <= 5:
                    continue
                start = offsets[int(row)]
                block = np.asarray(
                    group["hb_router_probs"][start : start + length], dtype=np.float32
                )
                routes = progress_ratio.action_route(block)
                for lag_position, lag in enumerate(LAGS[:4]):
                    query = int(rng.integers(lag, length))
                    recomputed = progress_ratio.route_distance(
                        routes[query], routes[query - lag]
                    )
                    stored_value = cache["lag_distance"][int(row), query, lag_position]
                    checks.append(float(np.abs(recomputed - stored_value).max()))
    report["route_distance_check_rows"] = len(checks)
    report["max_abs_route_distance_delta"] = float(max(checks)) if checks else None
    if checks and max(checks) > 1e-4:
        raise ValueError("cached lag distances disagree with the raw router probabilities")

    valid = cache["valid"].astype(bool)
    if not np.isfinite(entropy[valid]).all():
        raise ValueError("gate entropy has holes inside the valid mask")
    report["seconds"] = round(time.perf_counter() - started, 1)
    report["min"] = float(entropy[valid].min())
    report["max"] = float(entropy[valid].max())
    report["mean"] = float(entropy[valid].mean())
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        entropy=entropy,
        episode=cache["episode"],
        task_index=cache["task_index"],
        provenance=np.asarray(json.dumps(report)),
    )
    return entropy, report


def build_channels(cohort: dict[str, Any], entropy: np.ndarray) -> dict[str, np.ndarray]:
    """Every per-(episode, query, layer) channel the sensor is allowed to read.

    Nothing wider than window 2 appears here.  ``adjacent`` is the window-1
    displacement, which is the same quantity the v4 mobility file stores; the
    agreement is asserted so the two provenances cannot drift apart.
    """
    lag_distance = cohort["cache"]["lag_distance"]
    adjacent = lag_distance[:, :, LAGS.index(1), :]
    valid = cohort["cache"]["valid"].astype(bool)
    delta = np.abs(
        np.where(valid[:, :, None], adjacent, 0.0)
        - np.where(valid[:, :, None], cohort["mobility"], 0.0)
    ).max()
    if delta > 1e-4:
        raise ValueError(f"adjacent distance disagrees with the v4 mobility file by {delta}")
    displacement = lag_distance[:, :, LAGS.index(2), :]
    length = progress_ratio.path_length(adjacent, 2)
    return {
        "mobility": adjacent,
        "entropy": entropy,
        "disp2": displacement,
        "path2": length,
        "ratio2": progress_ratio.progress_ratio(displacement, length, EPS_LENGTH),
        "lag_distance": lag_distance,
    }


def build_episodes(cohort: dict[str, Any], events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse closure events onto cohort rows.

    An episode can close on the target more than once, so the events table is
    not unique on KEY and 150 keys carry both a phantom and a coupled closure.
    Routing timing predicts one closure per rollout, so the unit here is the
    episode: an episode with any phantom closure is a case at its first phantom
    closure, and its coupled closures are removed from the control pool rather
    than counted as false events.
    """
    index = cohort["index"]
    known = set(index["task"].unique())
    block = events[events["cohort"] == cohort["name"]].copy()
    outside = sorted(set(block["task"].unique()) - known)
    dropped = block[~block["task"].isin(known)]
    report = {
        "events_in_run": int(len(block)),
        "case_events_in_run": int(block["missed_grasp_then_departure"].sum()),
        "control_events_in_run": int(block["coupled_target_motion"].sum()),
        "tasks_absent_from_cohort": outside,
        "events_dropped_task_absent": int(len(dropped)),
        "case_events_dropped_task_absent": int(dropped["missed_grasp_then_departure"].sum()),
    }
    block = block[block["task"].isin(known)]
    merged = block.merge(index, on=KEY, how="left", validate="many_to_one")
    if merged["row"].isna().any():
        raise ValueError(f"{cohort['name']}: {int(merged['row'].isna().sum())} events did not join")
    merged["row"] = merged["row"].astype(int)

    cases = merged[merged["missed_grasp_then_departure"]]
    controls = merged[merged["coupled_target_motion"]]
    report["case_events_joined"] = int(len(cases))
    report["control_events_joined"] = int(len(controls))

    case_rows = cases.groupby("row")["closure_action_query"].min()
    control_rows = controls.groupby("row")["closure_action_query"].min()
    report["case_episodes"] = int(len(case_rows))
    report["control_episodes_before_case_removal"] = int(len(control_rows))
    overlap = sorted(set(control_rows.index) & set(case_rows.index))
    report["control_episodes_removed_share_case_episode"] = int(len(overlap))
    control_rows = control_rows.drop(index=overlap)

    frame = pd.concat(
        [
            pd.DataFrame({"row": case_rows.index, "closure": case_rows.to_numpy(), "case": 1}),
            pd.DataFrame(
                {"row": control_rows.index, "closure": control_rows.to_numpy(), "case": 0}
            ),
        ],
        ignore_index=True,
    )
    frame = frame.merge(index, on="row", how="left")
    frame["cohort"] = cohort["name"]
    if int(frame["closure"].min()) < MIN_QUERY + 1:
        raise ValueError("a closure lands before the earliest query with window-2 features")

    inside = (frame["closure"] + SCAN - 1) <= (frame["length"] - 1)
    report["episodes_before_scan_fit"] = int(len(frame))
    report["case_episodes_dropped_scan_past_end"] = int((~inside & (frame["case"] == 1)).sum())
    report["control_episodes_dropped_scan_past_end"] = int((~inside & (frame["case"] == 0)).sum())
    frame = frame[inside].reset_index(drop=True)
    report["case_episodes_kept"] = int((frame["case"] == 1).sum())
    report["control_episodes_kept"] = int((frame["case"] == 0).sum())
    return frame, report


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

STAGE1_CHANNELS = ("mobility", "entropy", "disp2", "path2", "ratio2")


def stage1_names() -> list[str]:
    return [f"{channel}|{layer}" for channel in STAGE1_CHANNELS for layer in LAYERS]


def stage2_names() -> list[str]:
    names = [f"{channel}|{layer}" for channel in STAGE1_CHANNELS for layer in LAYERS]
    names += [f"refhell|{layer}" for layer in LAYERS]
    names += [f"dmobility|{layer}" for layer in LAYERS]
    names += [f"dentropy|{layer}" for layer in LAYERS]
    names.append("relative_query")
    return names


def stage1_features(channels: dict[str, np.ndarray], rows: np.ndarray, queries: np.ndarray):
    """[N, 40] instantaneous mobility, gate entropy and the window-2 block."""
    return np.concatenate(
        [channels[channel][rows, queries, :] for channel in STAGE1_CHANNELS], axis=1
    ).astype(np.float32)


def stage2_features(
    channels: dict[str, np.ndarray],
    rows: np.ndarray,
    queries: np.ndarray,
    reference: np.ndarray,
    relative: np.ndarray,
):
    """[N, 65] the stage-1 block plus the reference comparison.

    ``refhell`` is ``route_distance(action_route(P_q), action_route(P_r))`` per
    layer against the query before closure, which was the strongest single
    feature in the matched-pose experiment.  It is read out of the cached lag
    distances, which is exact because ``q - r`` is 1..4 for a 0..3 scan and the
    cache stores lags (1, 2, 3, 4, 6, 8, 10, 12); the identity is re-derived
    from the raw router probabilities in ``gate_entropy``.
    """
    gap = queries - reference
    if gap.min() < 1 or not set(np.unique(gap)).issubset(set(LAGS)):
        raise ValueError(f"reference gaps {sorted(set(np.unique(gap)))} are not cached lags")
    lag_position = np.array([LAGS.index(int(value)) for value in gap])
    refhell = channels["lag_distance"][rows, queries, lag_position, :]
    base = stage1_features(channels, rows, queries)
    return np.concatenate(
        [
            base,
            refhell,
            channels["mobility"][rows, queries, :] - channels["mobility"][rows, reference, :],
            channels["entropy"][rows, queries, :] - channels["entropy"][rows, reference, :],
            relative.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def fit_logistic(features: np.ndarray, labels: np.ndarray):
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs"
    ).fit(scaler.transform(features), labels)
    return scaler, model


def score_logistic(fitted, features: np.ndarray) -> np.ndarray:
    scaler, model = fitted
    return model.predict_proba(scaler.transform(features))[:, 1]


def stratified_auc(labels: np.ndarray, scores: np.ndarray, strata: np.ndarray) -> tuple[float, int]:
    """AUC within strata of the closure query, pooled by stratum size.

    Phantom closures happen at median chunk 27-30 and coupled closures at
    chunk 5, so an unstratified AUC mostly measures *when* the gripper closed.
    Stratifying on the closure query is the sensor-side analogue of the
    query-index matching used in the matched-pose experiment: it asks what
    routing still separates once lateness is held fixed.
    """
    labels = np.asarray(labels)
    total, weight = 0.0, 0
    for value in np.unique(strata):
        mask = strata == value
        if len(np.unique(labels[mask])) < 2:
            continue
        total += roc_auc_score(labels[mask], scores[mask]) * int(mask.sum())
        weight += int(mask.sum())
    return (total / weight if weight else float("nan")), weight


# --------------------------------------------------------------------------- #
# stage 1: closure detection
# --------------------------------------------------------------------------- #


def stage1_table(episodes: pd.DataFrame, cohorts: dict[str, dict]) -> pd.DataFrame:
    """One row per (episode, query >= 2) with the at-or-past-closure label."""
    frames = []
    for name, cohort in cohorts.items():
        block = episodes[episodes["cohort"] == name]
        rows, queries = [], []
        for row, length in zip(block["row"].to_numpy(), block["length"].to_numpy()):
            span = np.arange(MIN_QUERY, int(length))
            rows.append(np.full(span.size, row))
            queries.append(span)
        rows = np.concatenate(rows)
        queries = np.concatenate(queries)
        closure = block.set_index("row")["closure"]
        frames.append(
            pd.DataFrame(
                {
                    "cohort": name,
                    "row": rows,
                    "query": queries,
                    "closure": closure.reindex(rows).to_numpy(),
                    "label": (queries >= closure.reindex(rows).to_numpy()).astype(int),
                }
            )
        )
    table = pd.concat(frames, ignore_index=True)
    lookup = episodes.set_index(["cohort", "row"])[["task", "suite", "case", "length"]]
    joined = table.merge(
        lookup.reset_index(), on=["cohort", "row"], how="left", validate="many_to_one"
    )
    return joined


def dense_probability(
    probability: np.ndarray, queries: np.ndarray, rows: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scatter per-query probabilities into a [episode, query] block.

    Queries the sensor may not use -- anything before the earliest possible
    closure at query ``MIN_QUERY + 1`` -- are filled with ``-inf`` so they can
    never trigger, and absent queries past the episode end likewise.
    """
    unique, inverse = np.unique(rows, return_inverse=True)
    dense = np.full((len(unique), horizon), -np.inf, dtype=np.float64)
    usable = queries >= MIN_QUERY + 1
    dense[inverse[usable], queries[usable]] = probability[usable]
    return dense, unique, np.arange(horizon)


def predict_closure(
    dense: np.ndarray, unique_rows: np.ndarray, grid: np.ndarray, threshold: float, fallback: int
) -> np.ndarray:
    """First query whose closure probability crosses; ``fallback`` when never.

    A rollout that never crosses has to guess something, and the honest guess
    for a deployed sensor is the training-median closure chunk.
    """
    position = first_trigger_rows(dense, threshold)
    return np.where(position < 0, fallback, grid[position])


def run_stage1(
    table: pd.DataFrame, channels: dict[str, dict[str, np.ndarray]], episodes: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, int], int], dict]:
    """Leave-one-suite-out closure detection, with the constant-chunk baseline."""
    features = {
        name: stage1_features(
            channels[name],
            table.loc[table["cohort"] == name, "row"].to_numpy(),
            table.loc[table["cohort"] == name, "query"].to_numpy(),
        )
        for name in channels
    }
    matrix = np.zeros((len(table), len(stage1_names())), dtype=np.float32)
    for name in channels:
        matrix[(table["cohort"] == name).to_numpy()] = features[name]
    if not np.isfinite(matrix).all():
        raise ValueError("stage-1 features contain non-finite entries")

    labels = table["label"].to_numpy()
    suite = table["suite"].to_numpy()
    grid = np.round(np.arange(0.20, 0.86, 0.02), 3)
    records: list[dict] = []
    predicted: dict[tuple[str, int], int] = {}
    chosen: dict[str, dict] = {}

    length_lookup = episodes.set_index(["cohort", "row"])["length"]
    for held in SUITES:
        train = suite != held
        test = suite == held
        if not test.any():
            continue
        fitted = fit_logistic(matrix[train], labels[train])

        # The crossing threshold is picked on the training suites only, out of
        # fold across them, so it is not tuned on the suite it is scored on.
        inner_probability = np.full(int(train.sum()), np.nan)
        train_suite = suite[train]
        for inner in SUITES:
            if inner == held:
                continue
            inner_test = train_suite == inner
            if not inner_test.any():
                continue
            inner_train = ~inner_test
            inner_fit = fit_logistic(matrix[train][inner_train], labels[train][inner_train])
            inner_probability[inner_test] = score_logistic(inner_fit, matrix[train][inner_test])
        usable = np.isfinite(inner_probability)
        train_table = table[train][usable].reset_index(drop=True)
        train_median = int(np.median(episodes.loc[episodes["suite"] != held, "closure"]))
        # One episode-by-query block per cohort keyed on (cohort, row), so the
        # threshold sweep is a scan over a dense array rather than a Python loop.
        horizon = int(table["query"].max()) + 1
        train_row_id = pd.factorize(
            train_table["cohort"].astype(str) + ":" + train_table["row"].astype(str)
        )[0]
        train_dense, train_unique, train_grid = dense_probability(
            inner_probability[usable],
            train_table["query"].to_numpy(),
            train_row_id,
            horizon,
        )
        train_truth = (
            train_table.assign(row_id=train_row_id).groupby("row_id")["closure"].first()
        ).reindex(train_unique).to_numpy()
        best = None
        for threshold in grid:
            guess = predict_closure(train_dense, train_unique, train_grid, float(threshold), train_median)
            score = float(np.abs(guess - train_truth).mean())
            if best is None or score < best[1]:
                best = (float(threshold), score)
        chosen[held] = {"threshold": best[0], "train_mae_chunks": round(best[1], 3),
                        "train_median_closure": train_median}

        probability = score_logistic(fitted, matrix[test])
        held_table = table[test].reset_index(drop=True)
        for cohort_name in sorted(held_table["cohort"].unique()):
            mask = (held_table["cohort"] == cohort_name).to_numpy()
            block = held_table[mask].reset_index(drop=True)
            dense, unique, dense_grid = dense_probability(
                probability[mask], block["query"].to_numpy(), block["row"].to_numpy(), horizon
            )
            raw = predict_closure(
                dense, unique, dense_grid, chosen[held]["threshold"], train_median
            )
            for row, value in zip(unique, raw):
                cap = int(length_lookup[(cohort_name, int(row))]) - SCAN
                predicted[(cohort_name, int(row))] = int(
                    np.clip(value, MIN_QUERY + 1, max(cap, MIN_QUERY + 1))
                )
            truth = block.groupby("row")["closure"].first()
            case_rows = set(
                episodes.loc[
                    (episodes["cohort"] == cohort_name)
                    & (episodes["suite"] == held)
                    & (episodes["case"] == 1),
                    "row",
                ]
            )
            for population, subset in (
                ("all", list(truth.index)),
                ("phantom", [r for r in truth.index if int(r) in case_rows]),
                ("coupled", [r for r in truth.index if int(r) not in case_rows]),
            ):
                if not subset:
                    continue
                error = np.array(
                    [predicted[(cohort_name, int(r))] - int(truth[r]) for r in subset]
                )
                baseline = np.array([train_median - int(truth[r]) for r in subset])
                records.append(
                    {
                        "held_out_suite": held,
                        "cohort": cohort_name,
                        "population": population,
                        "episodes": int(len(subset)),
                        "query_rows": int(mask.sum()) if population == "all" else -1,
                        "auc_at_or_past_closure": float(
                            roc_auc_score(block["label"].to_numpy(), probability[mask])
                        )
                        if population == "all" and block["label"].nunique() > 1
                        else float("nan"),
                        "threshold": chosen[held]["threshold"],
                        "median_signed_error_chunks": float(np.median(error)),
                        "mae_chunks": float(np.abs(error).mean()),
                        "share_within_2": float((np.abs(error) <= 2).mean()),
                        "baseline_median_signed_error_chunks": float(np.median(baseline)),
                        "baseline_mae_chunks": float(np.abs(baseline).mean()),
                        "baseline_share_within_2": float((np.abs(baseline) <= 2).mean()),
                    }
                )
    return pd.DataFrame(records), predicted, chosen


# --------------------------------------------------------------------------- #
# stage 2: coupling failure
# --------------------------------------------------------------------------- #


def scan_matrix(
    episodes: pd.DataFrame, channels: dict[str, dict[str, np.ndarray]], anchor: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """[N, SCAN, F] features and [N, SCAN] absolute queries for a given anchor."""
    features = np.zeros((len(episodes), SCAN, len(stage2_names())), dtype=np.float32)
    queries = np.zeros((len(episodes), SCAN), dtype=int)
    cohort_column = episodes["cohort"].to_numpy()
    rows = episodes["row"].to_numpy()
    for name in channels:
        mask = cohort_column == name
        if not mask.any():
            continue
        for step in range(SCAN):
            query = anchor[mask] + step
            reference = anchor[mask] - 1
            queries[mask, step] = query
            features[mask, step] = stage2_features(
                channels[name], rows[mask], query, reference, np.full(mask.sum(), step)
            )
    if not np.isfinite(features).all():
        raise ValueError("stage-2 features contain non-finite entries")
    return features, queries


def evaluate_operating_point(
    episodes: pd.DataFrame,
    scores: np.ndarray,
    queries: np.ndarray,
    true_closure: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
    label: str,
) -> list[dict]:
    """Detection, latency, absolute report chunk and lead over the clock."""
    case = episodes["case"].to_numpy() == 1
    reports = report_queries(queries, scores, threshold)
    clock = np.array(
        [CLOCK_T[c][s] for c, s in zip(episodes["cohort"], episodes["suite"])]
    )
    earlier, lead = leads_over_clock(reports, clock)
    lat = latencies(reports, true_closure)

    out = []
    for cohort_name in ["both_cohorts"] + sorted(episodes["cohort"].unique()):
        sel = (
            np.ones(len(episodes), dtype=bool)
            if cohort_name == "both_cohorts"
            else (episodes["cohort"].to_numpy() == cohort_name)
        )
        case_sel = sel & case
        control_sel = sel & ~case
        detected = case_sel & (reports >= 0)
        false_event = control_sel & (reports >= 0)
        tasks = episodes["task"].to_numpy()
        if case_sel.sum():
            low, high = clustered_interval(
                (reports[case_sel] >= 0).astype(float), tasks[case_sel], rng
            )
        else:
            low, high = float("nan"), float("nan")
        if detected.sum():
            share_low, share_high = clustered_interval(
                earlier[detected].astype(float), tasks[detected], rng
            )
        else:
            share_low, share_high = float("nan"), float("nan")
        out.append(
            {
                "scope": cohort_name,
                "variant": label,
                "phantom_episodes": int(case_sel.sum()),
                "coupled_episodes": int(control_sel.sum()),
                "detected": int(detected.sum()),
                "detection_rate": float(detected.sum() / case_sel.sum()) if case_sel.sum() else float("nan"),
                "detection_rate_ci95_low": low,
                "detection_rate_ci95_high": high,
                "false_events": int(false_event.sum()),
                "realised_false_event_rate": float(false_event.sum() / control_sel.sum())
                if control_sel.sum()
                else float("nan"),
                "median_latency_chunks": float(np.nanmedian(lat[detected])) if detected.sum() else float("nan"),
                "p25_latency_chunks": float(np.nanpercentile(lat[detected], 25)) if detected.sum() else float("nan"),
                "p75_latency_chunks": float(np.nanpercentile(lat[detected], 75)) if detected.sum() else float("nan"),
                "median_report_chunk": float(np.median(reports[detected])) if detected.sum() else float("nan"),
                "median_closure_chunk": float(np.median(true_closure[case_sel])) if case_sel.sum() else float("nan"),
                "earlier_than_clock": int(earlier[case_sel].sum()),
                "share_earlier_of_detected": float(earlier[detected].mean()) if detected.sum() else float("nan"),
                "share_earlier_ci95_low": share_low,
                "share_earlier_ci95_high": share_high,
                "share_earlier_of_all_phantom": float(earlier[case_sel].mean()) if case_sel.sum() else float("nan"),
                "median_lead_chunks": float(np.median(lead[case_sel & earlier])) if (case_sel & earlier).any() else float("nan"),
                "median_lead_env_steps": float(CHUNK_TO_STEP * np.median(lead[case_sel & earlier]))
                if (case_sel & earlier).any()
                else float("nan"),
                "median_clock_chunk": float(np.median(clock[case_sel])) if case_sel.sum() else float("nan"),
                "latency_histogram": {
                    str(int(value)): int(count)
                    for value, count in zip(
                        *np.unique(lat[detected].astype(int), return_counts=True)
                    )
                }
                if detected.sum()
                else {},
                "report_chunk_histogram": {
                    str(int(value)): int(count)
                    for value, count in zip(*np.unique(reports[detected], return_counts=True))
                }
                if detected.sum()
                else {},
            }
        )
    return out


def run_stage2(
    episodes: pd.DataFrame,
    channels: dict[str, dict[str, np.ndarray]],
    predicted: dict[tuple[str, int], int],
) -> tuple[pd.DataFrame, dict]:
    """Leave-one-suite-out coupling sensor, oracle and routing timing."""
    closure = episodes["closure"].to_numpy()
    oracle_features, oracle_queries = scan_matrix(episodes, channels, closure)
    guess = np.array(
        [predicted[(c, int(r))] for c, r in zip(episodes["cohort"], episodes["row"])]
    )
    routing_features, routing_queries = scan_matrix(episodes, channels, guess)

    labels = episodes["case"].to_numpy()
    suite = episodes["suite"].to_numpy()
    flat = oracle_features.reshape(-1, oracle_features.shape[-1])
    flat_labels = np.repeat(labels, SCAN)
    flat_suite = np.repeat(suite, SCAN)

    rng = np.random.default_rng(SEED)
    records: list[dict] = []
    diagnostics: dict[str, Any] = {"per_suite": {}, "closure_index_auc": {}}

    for held in SUITES:
        train = flat_suite != held
        test = suite == held
        if not test.any():
            continue
        fitted = fit_logistic(flat[train], flat_labels[train])

        # Out-of-fold scores across the *training* suites give the threshold,
        # so the operating point is never calibrated on the suite it is used on.
        inner = np.full(int(train.sum()), np.nan)
        train_suite = flat_suite[train]
        for other in SUITES:
            if other == held:
                continue
            inner_test = train_suite == other
            if not inner_test.any():
                continue
            inner_fit = fit_logistic(flat[train][~inner_test], flat_labels[train][~inner_test])
            inner[inner_test] = score_logistic(inner_fit, flat[train][inner_test])
        train_episodes = episodes[episodes["suite"] != held].reset_index(drop=True)
        inner_by_episode = inner.reshape(len(train_episodes), SCAN)
        control = train_episodes["case"].to_numpy() == 0
        control_max = np.nanmax(inner_by_episode[control], axis=1)

        held_episodes = episodes[test].reset_index(drop=True)
        held_index = np.flatnonzero(test)
        oracle_scores = score_logistic(fitted, oracle_features[held_index].reshape(-1, flat.shape[1])).reshape(-1, SCAN)
        routing_scores = score_logistic(fitted, routing_features[held_index].reshape(-1, flat.shape[1])).reshape(-1, SCAN)

        held_control = held_episodes["case"].to_numpy() == 0
        held_control_max = np.nanmax(oracle_scores[held_control], axis=1)
        held_routing_control_max = np.nanmax(routing_scores[held_control], axis=1)
        held_auc = (
            float(roc_auc_score(held_episodes["case"].to_numpy(), np.nanmax(oracle_scores, axis=1)))
            if held_episodes["case"].nunique() > 1
            else float("nan")
        )
        index_auc = (
            float(roc_auc_score(held_episodes["case"].to_numpy(), held_episodes["closure"].to_numpy()))
            if held_episodes["case"].nunique() > 1
            else float("nan")
        )
        stratified, stratum_weight = stratified_auc(
            held_episodes["case"].to_numpy(),
            np.nanmax(oracle_scores, axis=1),
            held_episodes["closure"].to_numpy(),
        )
        diagnostics["per_suite"][held] = {
            "episode_auc_oracle_timing": held_auc,
            "closure_index_auc": index_auc,
            "closure_stratified_auc_oracle_timing": stratified,
            "closure_stratified_episodes": stratum_weight,
            "phantom_episodes": int((~held_control).sum()),
            "coupled_episodes": int(held_control.sum()),
            "median_phantom_closure": float(
                np.median(held_episodes.loc[~held_control, "closure"])
            )
            if (~held_control).any()
            else float("nan"),
            "median_coupled_closure": float(
                np.median(held_episodes.loc[held_control, "closure"])
            ),
            # The single number that explains the routing-timing collapse: how
            # often the predicted scan window even contains the true closure.
            "routing_scan_covers_closure_phantom": float(
                np.mean(
                    (
                        held_episodes.loc[~held_control, "closure"].to_numpy()
                        >= routing_queries[held_index][~held_control, 0]
                    )
                    & (
                        held_episodes.loc[~held_control, "closure"].to_numpy()
                        <= routing_queries[held_index][~held_control, -1]
                    )
                )
            )
            if (~held_control).any()
            else float("nan"),
            "routing_scan_covers_closure_coupled": float(
                np.mean(
                    (
                        held_episodes.loc[held_control, "closure"].to_numpy()
                        >= routing_queries[held_index][held_control, 0]
                    )
                    & (
                        held_episodes.loc[held_control, "closure"].to_numpy()
                        <= routing_queries[held_index][held_control, -1]
                    )
                )
            ),
        }

        for target in TARGET_FER:
            portable = threshold_at_false_event_rate(control_max, target)
            local = threshold_at_false_event_rate(held_control_max, target)
            routing_local = threshold_at_false_event_rate(held_routing_control_max, target)
            # The closure index itself is a nuisance predictor of the same
            # event; scored the same way it says how much of the sensor is the
            # clock in disguise.
            index_scores = np.repeat(
                held_episodes["closure"].to_numpy().reshape(-1, 1).astype(float), SCAN, axis=1
            )
            index_threshold = threshold_at_false_event_rate(
                train_episodes.loc[control, "closure"].to_numpy().astype(float), target
            )
            index_local = threshold_at_false_event_rate(
                held_episodes.loc[held_control, "closure"].to_numpy().astype(float), target
            )
            for name, scores, queries, anchor_threshold in (
                ("oracle_timing", oracle_scores, oracle_queries[held_index], portable),
                ("routing_timing", routing_scores, routing_queries[held_index], portable),
                ("oracle_timing_local_calibration", oracle_scores, oracle_queries[held_index], local),
                (
                    "routing_timing_local_calibration",
                    routing_scores,
                    routing_queries[held_index],
                    routing_local,
                ),
                ("closure_index_baseline", index_scores, oracle_queries[held_index], index_threshold),
                (
                    "closure_index_local_calibration",
                    index_scores,
                    oracle_queries[held_index],
                    index_local,
                ),
            ):
                for entry in evaluate_operating_point(
                    held_episodes,
                    scores,
                    queries,
                    held_episodes["closure"].to_numpy(),
                    anchor_threshold,
                    rng,
                    name,
                ):
                    entry.update(
                        {
                            "held_out_suite": held,
                            "target_false_event_rate": target,
                            "threshold": anchor_threshold,
                            "calibration": "held_out_suite"
                            if name.endswith("local_calibration")
                            else "train_suites_out_of_fold",
                            "train_calibration_false_event_rate": float(
                                (control_max >= anchor_threshold).mean()
                            )
                            if name in ("oracle_timing", "routing_timing")
                            else float("nan"),
                        }
                    )
                    records.append(entry)
    return pd.DataFrame(records), diagnostics


def pool_over_suites(table: pd.DataFrame) -> pd.DataFrame:
    """Recombine the held-out folds into one number per variant/scope/target."""
    out = []
    for (scope, variant, target), block in table.groupby(
        ["scope", "variant", "target_false_event_rate"], sort=True
    ):
        phantom = int(block["phantom_episodes"].sum())
        coupled = int(block["coupled_episodes"].sum())
        detected = int(block["detected"].sum())
        false_events = int(block["false_events"].sum())
        earlier = int(block["earlier_than_clock"].sum())
        out.append(
            {
                "scope": scope,
                "variant": variant,
                "target_false_event_rate": target,
                "phantom_episodes": phantom,
                "coupled_episodes": coupled,
                "detected": detected,
                "detection_rate": detected / phantom if phantom else float("nan"),
                "false_events": false_events,
                "realised_false_event_rate": false_events / coupled if coupled else float("nan"),
                "earlier_than_clock": earlier,
                "share_earlier_of_detected": earlier / detected if detected else float("nan"),
                "share_earlier_of_all_phantom": earlier / phantom if phantom else float("nan"),
            }
        )
    return pd.DataFrame(out)


def pooled_detail(
    episodes: pd.DataFrame, table: pd.DataFrame
) -> pd.DataFrame:
    """Per-suite detail is in ``table``; this adds the phantom-share context."""
    rows = []
    for cohort_name, block in episodes.groupby("cohort"):
        rows.append(
            {
                "cohort": cohort_name,
                "phantom_episodes_analysed": int((block["case"] == 1).sum()),
                "coupled_episodes_analysed": int((block["case"] == 0).sum()),
                "median_phantom_closure_chunk": float(
                    block.loc[block["case"] == 1, "closure"].median()
                ),
                "median_coupled_closure_chunk": float(
                    block.loc[block["case"] == 0, "closure"].median()
                ),
                "tasks_with_phantom": int(block.loc[block["case"] == 1, "task"].nunique()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


SHARE_CONTEXT = {
    "development_main": {
        "failures": 487,
        "phantom_grasp": 93,
        "phantom_neither_loop_nor_static": 22,
    },
    "external_8b": {
        "failures": 564,
        "phantom_grasp": 115,
        "phantom_neither_loop_nor_static": 32,
    },
    "note": (
        "A sensor for the phantom-grasp event addresses at most this share of "
        "failures: 93/487 development and 115/564 external, of which 22 and 32 "
        "show neither loop nor static."
    ),
    "external_8b_status": (
        "external_8b is not a pristine holdout; it has been examined across "
        "v3-v7 and repeatedly in this bundle."
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--refresh-entropy", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    events = pd.read_csv(EVENTS)
    events["cohort"] = events["run"].map(RUN_COHORT)
    if events["cohort"].isna().any():
        raise ValueError("unmapped run in the events table")

    cohorts, channels, joins, entropy_reports = {}, {}, {}, {}
    frames = []
    for name in COHORTS:
        cohort = load_cohort(name)
        entropy, report = gate_entropy(cohort, args.output, args.refresh_entropy)
        cohorts[name] = cohort
        entropy_reports[name] = report
        channels[name] = build_channels(cohort, entropy)
        frame, join_report = build_episodes(cohort, events)
        joins[name] = join_report
        frames.append(frame)
    episodes = pd.concat(frames, ignore_index=True)

    table = stage1_table(episodes, cohorts)
    closure_table, predicted, thresholds = run_stage1(table, channels, episodes)
    closure_table.to_csv(args.output / "closure_detection.csv", index=False)

    sensor_table, diagnostics = run_stage2(episodes, channels, predicted)
    sensor_table.to_csv(args.output / "coupling_sensor.csv", index=False)
    pooled = pool_over_suites(sensor_table)
    context = pooled_detail(episodes, sensor_table)

    guess_error = np.array(
        [predicted[(c, int(r))] - int(k) for c, r, k in zip(episodes["cohort"], episodes["row"], episodes["closure"])]
    )
    bridge = {}
    for cohort_name in list(COHORTS) + ["both_cohorts"]:
        for population, want in (("phantom", 1), ("coupled", 0)):
            mask = (episodes["case"].to_numpy() == want) & (
                np.ones(len(episodes), dtype=bool)
                if cohort_name == "both_cohorts"
                else (episodes["cohort"].to_numpy() == cohort_name)
            )
            if not mask.any():
                continue
            error = guess_error[mask]
            bridge[f"{cohort_name}|{population}"] = {
                "episodes": int(mask.sum()),
                "median_signed_error_chunks": float(np.median(error)),
                "mae_chunks": float(np.abs(error).mean()),
                "share_within_2": float((np.abs(error) <= 2).mean()),
                "share_scan_covers_closure": float(((error <= 0) & (error > -SCAN)).mean()),
            }
    summary = {
        "schema": SCHEMA,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scan_relative_queries": list(range(SCAN)),
        "chunk_to_env_steps": CHUNK_TO_STEP,
        "clock_thresholds": CLOCK_T,
        "target_false_event_rates": list(TARGET_FER),
        "bootstrap": {"draws": DRAWS, "unit": "task", "seed": SEED},
        "feature_names": {"stage1": stage1_names(), "stage2": stage2_names()},
        "alignment": {name: cohorts[name]["alignment"] for name in cohorts},
        "gate_entropy": entropy_reports,
        "joins": joins,
        "row_counts": {
            "stage1_query_rows": int(len(table)),
            "stage1_positive_rows": int(table["label"].sum()),
            "stage2_episode_rows": int(len(episodes)),
            "stage2_scan_rows": int(len(episodes) * SCAN),
            "phantom_episodes": int((episodes["case"] == 1).sum()),
            "coupled_episodes": int((episodes["case"] == 0).sum()),
            "per_cohort": {
                name: {
                    "phantom": int(((episodes["cohort"] == name) & (episodes["case"] == 1)).sum()),
                    "coupled": int(((episodes["cohort"] == name) & (episodes["case"] == 0)).sum()),
                }
                for name in COHORTS
            },
            "no_window_wider_than_2": True,
        },
        "stage1_thresholds": thresholds,
        "stage1_closure_error_chunks": {
            "median_signed": float(np.median(guess_error)),
            "mae": float(np.abs(guess_error).mean()),
            "share_within_2": float((np.abs(guess_error) <= 2).mean()),
        },
        "stage1_to_stage2_bridge": bridge,
        "stage2_diagnostics": diagnostics,
        "closure_detection": closure_table.to_dict("records"),
        "coupling_sensor": sensor_table.to_dict("records"),
        "pooled": pooled.to_dict("records"),
        "context": context.to_dict("records"),
        "share_of_failures": SHARE_CONTEXT,
    }
    summary["seconds"] = round(time.perf_counter() - started, 1)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    pd.set_option("display.width", 240)
    fmt = lambda value: f"{value:.4f}"  # noqa: E731

    print("=" * 110)
    print("row counts (no feature wider than window 2; nothing was dropped by a dropna)")
    print(json.dumps(summary["row_counts"], indent=1))
    print("\njoin ledger")
    print(json.dumps(joins, indent=1))

    print("\n" + "=" * 110)
    print("STAGE 1  closure detection, leave-one-suite-out")
    print(
        closure_table[
            [
                "held_out_suite",
                "cohort",
                "population",
                "episodes",
                "auc_at_or_past_closure",
                "median_signed_error_chunks",
                "mae_chunks",
                "share_within_2",
                "baseline_mae_chunks",
                "baseline_share_within_2",
            ]
        ].to_string(index=False, float_format=fmt)
    )

    print("\nwhat stage 1 hands to stage 2: does the predicted scan window contain the closure")
    print(
        pd.DataFrame(
            [{"scope|population": key, **value} for key, value in bridge.items()]
        ).to_string(index=False, float_format=fmt)
    )

    print("\n" + "=" * 110)
    print("STAGE 2  coupling failure sensor, per held-out suite")
    for target in TARGET_FER:
        print(f"\n--- fixed false-event rate on truly coupled closures = {target:.1%} ---")
        block = sensor_table[
            (sensor_table["target_false_event_rate"] == target)
            & (sensor_table["scope"] == "both_cohorts")
        ]
        print(
            block[
                [
                    "held_out_suite",
                    "variant",
                    "phantom_episodes",
                    "detected",
                    "detection_rate",
                    "detection_rate_ci95_low",
                    "detection_rate_ci95_high",
                    "train_calibration_false_event_rate",
                    "realised_false_event_rate",
                    "median_latency_chunks",
                    "p25_latency_chunks",
                    "p75_latency_chunks",
                    "median_report_chunk",
                    "share_earlier_of_detected",
                    "share_earlier_ci95_low",
                    "share_earlier_ci95_high",
                    "median_lead_chunks",
                    "median_lead_env_steps",
                ]
            ].to_string(index=False, float_format=fmt)
        )

    print("\n" + "=" * 110)
    print("STAGE 2  pooled over held-out suites")
    print(pooled.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 110)
    print("per-cohort detail at each operating point (pooled over held-out suites)")
    for cohort_name in COHORTS:
        block = sensor_table[sensor_table["scope"] == cohort_name]
        rolled = pool_over_suites(block)
        print(f"\n{cohort_name}")
        print(rolled.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 110)
    print("how much of the separation is the closure chunk rather than routing")
    print(
        pd.DataFrame(
            [{"held_out_suite": key, **value} for key, value in diagnostics["per_suite"].items()]
        ).to_string(index=False, float_format=fmt)
    )

    print("\n" + "=" * 110)
    print("context")
    print(context.to_string(index=False, float_format=fmt))
    print(json.dumps(SHARE_CONTEXT, indent=1))
    print(f"\nseconds: {summary['seconds']}")


if __name__ == "__main__":
    main()
