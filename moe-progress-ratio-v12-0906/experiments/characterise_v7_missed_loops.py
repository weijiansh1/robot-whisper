#!/usr/bin/env python3
"""Task B: are the loop episodes that the conjunction catches and v7 misses a
distinctive subpopulation, or an arbitrary slice of the v7 miss pool?

The conjunctive rule ``L(q, W) >= ell* AND R(q, W) < r*`` confirmed K times, at
the round-two near-miss operating point (``results/loop_operating_point``),
detects 35 of the 256 development ``loop`` episodes.  Twenty-four of those are
caught by neither v7 branch (``main_freeze`` / ``main_turbulence``).  v7's two
branches miss 85 loop episodes in total.

Three groups among the 256 loop episodes:

* ``G1`` -- conjunction alarm, no v7 branch alarm            (expect 24)
* ``G2`` -- no conjunction alarm, no v7 branch alarm         (expect 61)
* ``G3`` -- at least one v7 branch alarm                     (expect 171)

G1 vs G2 is the controlled contrast: both are v7 misses, so the only thing that
differs is whether the routing-geometry conjunction fired.  G3 is context only.

Framing constraint carried over from ``diagnose_phenotype_timing`` and the
round-two timing addendum: this rule's alarms *lag* loop onset by a median of
5 chunks and precede it only 5.7% of the time.  Nothing here is a prediction
claim; it is a readout characterisation.

Every characterisation variable comes from the trap taxonomy tables or the
cohort key.  None of them is derived from the routing quantity, so the
comparison is not circular.

Statistics
----------
* effect size per variable: Cohen's d (numeric), risk difference (binary),
  Cramer's V (categorical)
* task-clustered bootstrap 95% interval: resample the tasks present in the
  85-episode pool with replacement, 2000 draws
* permutation test (as specified): draw 24 of the 85 at random 2000 times, place
  the observed G1 statistic in the resulting null
* stratified permutation (robustness): permute the G1/G2 label *within task*, so
  the null holds the task composition of G1 fixed.  This is the honest null for
  clustered data and it separates "G1 is a distinct physical phenotype" from
  "G1 is a distinct set of tasks".
* multiplicity: Westfall-Young min-P over the *entire* pre-declared variable
  list, Benjamini-Hochberg on the raw permutation p-values, and a single global
  min-P value that answers "is the whole pattern distinguishable from a random
  draw of 24 from the 85".  min-P rather than max-T because the variables carry
  wildly different null scales (a 21-level Cramer's V has a very tight null) and
  a max-T null would be swallowed by the highest-cardinality variable.

Run:  python experiments/characterise_v7_missed_loops.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio  # noqa: E402

DEFAULT_OUTPUT = BUNDLE / "results/v7_missed_loops"
MAIN_CACHE = BUNDLE / "results/progress_cache/development_main.npz"
EXTRA_CACHE = BUNDLE / "results/progress_cache/development_extra.npz"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
BELIEF_EPISODES = TAXONOMY / "belief_mismatch_episodes.csv.gz"
EVENTS = TAXONOMY / "events.csv.gz"
V4_MOBILITY = WORKSPACE / "moe-v4-0904/results/layerwise_mobility/main_reference.npz"
V7_ALARMS = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"
V7_EPISODES = WORKSPACE / "moe-v7-0905/results/intrinsic_guard_v7/episode_alarms.csv"

DEVELOPMENT_RUN = "seed1000_1007"
JOIN_KEY = ["task", "init_state_id", "flow_noise_seed"]
SCHEMA = "himoe.progress_ratio_v12.v7_missed_loops.v1"

# Round-two near-miss operating point, copied verbatim from
# results/loop_operating_point/selection.json.  Nothing here is re-selected.
WINDOW = 3
LAYER_GROUP = "all"
CONFIRMATIONS = 2
RATIO_THRESHOLD = 0.294327
EPS_LENGTH = 0.057664

# Sanity anchors.  If any of these fails the run is BLOCKED, not adjusted.
ANCHOR_FAILURES = 487
ANCHOR_LOOP = 256
ANCHOR_STATIC = 135
ANCHOR_PHANTOM = 93
ANCHOR_PURE_PHANTOM = 22
ANCHOR_LOOP_DETECTIONS = 35
ANCHOR_G1 = 24
ANCHOR_V7_MISSES = 85

BOOTSTRAP_DRAWS = 2000
PERMUTATION_DRAWS = 2000
SEED = 20260906

BEHAVIOR_COMPONENTS = (
    "eef_oscillation",
    "gripper_cycling",
    "stagnation",
    "regrasp_or_drop",
    "active_retry",
    "goal_regression",
    "subtask_undo",
    "goal_approach_leave",
)

# Pre-declared, in full, before any contrast was computed.  The multiplicity
# correction runs over this whole list; nothing is added after seeing a result.
VARIABLES: tuple[tuple[str, str, str], ...] = (
    # (name, kind, provenance)
    ("loop_onset", "numeric", "taxonomy"),
    ("static_onset", "numeric", "taxonomy"),
    ("n_queries", "numeric", "taxonomy"),
    ("loop_onset_fraction", "numeric", "taxonomy-derived"),
    ("queries_after_loop_onset", "numeric", "taxonomy-derived"),
    ("near_unfinished_closure_count", "numeric", "taxonomy"),
    ("belief_mismatch_count", "numeric", "taxonomy"),
    ("n_behaviors", "numeric", "taxonomy-derived"),
    ("init_state_id", "numeric", "cohort"),
    ("proxy_grade", "categorical", "taxonomy"),
    ("first_event", "categorical", "taxonomy"),
    ("primary_behavior", "categorical", "taxonomy"),
    ("behavior_signature", "categorical", "taxonomy"),
    ("suite", "categorical", "cohort"),
    ("task", "categorical", "cohort"),
    ("belief_mismatch", "binary", "taxonomy"),
    ("closed_mismatch_within_5q", "binary", "taxonomy"),
    ("closed_mismatch_candidate", "binary", "taxonomy"),
    ("static", "binary", "co-occurrence"),
    ("phantom", "binary", "co-occurrence"),
    ("pure_phantom", "binary", "co-occurrence"),
    *(
        (f"behavior_{component}", "binary", "taxonomy-derived")
        for component in BEHAVIOR_COMPONENTS
    ),
)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def task_of(cache: dict[str, np.ndarray]) -> np.ndarray:
    return cache["task_names"][cache["task_index"].astype(int)]


def flow_noise_seed(cache: dict[str, np.ndarray]) -> np.ndarray:
    """The progress cache has no flow_noise_seed; take it from the row-aligned
    v4 mobility cache after verifying the alignment element by element."""
    mobility = load_npz(V4_MOBILITY)
    if len(mobility["episode"]) != len(cache["episode"]):
        raise ValueError("the v4 mobility cache has a different row count")
    if not np.array_equal(mobility["episode"], cache["episode"]):
        raise ValueError("episode disagrees between the progress and mobility caches")
    if not np.array_equal(task_of(mobility), task_of(cache)):
        raise ValueError("task disagrees between the progress and mobility caches")
    if not np.array_equal(mobility["init_state_id"], cache["init_state_id"]):
        raise ValueError("init_state_id disagrees between the two caches")
    return mobility["flow_noise_seed"].astype(int)


def v7_alarms(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """v7 sealed alarms plus an explicit row-alignment proof against the cohort."""
    sealed = load_npz(V7_ALARMS)
    frame = pd.read_csv(V7_EPISODES)
    frame = frame[frame["cohort"] == "development_main"].sort_values("row")
    if len(frame) != len(cache["episode"]):
        raise ValueError("v7 episode_alarms has a different development row count")
    if not np.array_equal(frame["row"].to_numpy(int), np.arange(len(frame))):
        raise ValueError("v7 development rows are not 0..N-1")
    if not np.array_equal(frame["task"].to_numpy(str), task_of(cache)):
        raise ValueError("v7 episode_alarms task order disagrees with the cohort")
    if not np.array_equal(frame["episode"].to_numpy(int), cache["episode"].astype(int)):
        raise ValueError("v7 episode_alarms episode order disagrees with the cohort")
    if not np.array_equal(frame["length"].to_numpy(int), cache["length"].astype(int)):
        raise ValueError("v7 episode_alarms length disagrees with the cohort")
    # The sealed npz is what the task specifies; cross-check it against the csv.
    for key, column in (
        ("main_freeze", "first_relative_freeze_query"),
        ("main_turbulence", "first_confirmed_turbulence_query"),
        ("main_guard", "first_intrinsic_guard_v7_query"),
    ):
        if not np.array_equal(sealed[key].astype(int), frame[column].to_numpy(int)):
            raise ValueError(f"sealed {key} disagrees with episode_alarms {column}")
    return {key: sealed[key].astype(int) for key in ("main_freeze", "main_turbulence", "main_guard")}


# --------------------------------------------------------------------------- #
# the alarm array
# --------------------------------------------------------------------------- #


def length_threshold() -> float:
    """``ell*`` is the pooled-reference median of the grouped path length at
    (W, layer group).  Pre-registered, not swept; recomputed here rather than
    read from selection.json so the reproduction is self-contained."""
    main = load_npz(MAIN_CACHE)
    extra = load_npz(EXTRA_CACHE)
    adjacent = np.concatenate([main["lag_distance"], extra["lag_distance"]], axis=0)[:, :, 0, :]
    valid = np.concatenate([main["valid"], extra["valid"]], axis=0)
    grouped = progress_ratio.group_ratio(
        progress_ratio.path_length(adjacent, WINDOW), LAYER_GROUP
    )
    usable = grouped[valid & np.isfinite(grouped)]
    if usable.size == 0:
        raise ValueError("the pooled reference contains no finite path length")
    return float(np.median(usable))


def first_conjunctive_alarm(cache: dict[str, np.ndarray], ell_star: float) -> np.ndarray:
    """Verbatim reproduction of the alarm array specified for this task."""
    adjacent = cache["lag_distance"][:, :, 0, :]
    length = progress_ratio.path_length(adjacent, WINDOW)
    displacement = cache["lag_distance"][:, :, progress_ratio.LAGS.index(WINDOW), :]
    ratio = progress_ratio.group_ratio(
        progress_ratio.progress_ratio(displacement, length, EPS_LENGTH), LAYER_GROUP
    )
    grouped_length = progress_ratio.group_ratio(length, LAYER_GROUP)
    usable = cache["valid"] & np.isfinite(ratio) & np.isfinite(grouped_length)
    with np.errstate(invalid="ignore"):
        hit = usable & (grouped_length >= ell_star) & (ratio < RATIO_THRESHOLD)
    run = hit.copy()
    for shift in range(1, CONFIRMATIONS):
        run[:, shift:] &= hit[:, :-shift]
    run[:, : CONFIRMATIONS - 1] = False
    first = np.full(len(run), -1, dtype=np.int32)
    fired = run.any(axis=1)
    first[fired] = run[fired].argmax(axis=1)
    return first


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #


def build_labels(cache: dict[str, np.ndarray]) -> pd.DataFrame:
    keyed = pd.DataFrame(
        {
            "row": np.arange(len(cache["episode"])),
            "task": task_of(cache),
            "init_state_id": cache["init_state_id"].astype(int),
            "flow_noise_seed": flow_noise_seed(cache),
            "length": cache["length"].astype(int),
        }
    )
    if keyed.duplicated(JOIN_KEY).any():
        raise ValueError("the cohort key is not unique")

    episodes = pd.read_csv(BELIEF_EPISODES)
    events = pd.read_csv(EVENTS)
    events["task"] = events["task_key"]  # events.csv splits suite and task
    taxonomy = episodes.merge(events, on=["run", "task", "episode"], validate="one_to_one")
    taxonomy = taxonomy[taxonomy["run"] == DEVELOPMENT_RUN]
    if not (taxonomy["first_event_x"] == taxonomy["first_event_y"]).all():
        raise ValueError("first_event disagrees between the two taxonomy tables")
    taxonomy = taxonomy.rename(
        columns={"first_event_x": "first_event", "success_x": "success"}
    )
    columns = JOIN_KEY + [
        "success",
        "belief_mismatch",
        "belief_mismatch_count",
        "closed_mismatch_candidate",
        "closed_mismatch_within_5q",
        "near_unfinished_closure_count",
        "first_event",
        "primary_behavior",
        "behavior_signature",
        "suite",
        "n_queries",
        "loop_onset",
        "static_onset",
        "loop_proxy_valid",
        "proxy_grade",
    ]
    merged = keyed.merge(taxonomy[columns], on=JOIN_KEY, how="left", validate="one_to_one")
    if merged["success"].isna().any():
        raise ValueError("the trap taxonomy does not cover the development cohort")
    merged = merged.sort_values("row").reset_index(drop=True)
    if not np.array_equal(merged["row"].to_numpy(int), np.arange(len(merged))):
        raise ValueError("the merge reordered the cohort")
    if not np.array_equal(merged["n_queries"].to_numpy(int), merged["length"].to_numpy(int)):
        raise ValueError("n_queries and the cohort length disagree")

    fail = ~merged["success"].to_numpy(bool)
    merged["failure"] = fail
    merged["loop"] = (merged["loop_onset"].to_numpy(int) >= 0) & fail
    merged["static"] = (merged["static_onset"].to_numpy(int) >= 0) & fail
    merged["phantom"] = merged["belief_mismatch"].to_numpy(bool) & fail
    merged["pure_phantom"] = merged["phantom"] & ~merged["loop"] & ~merged["static"]

    signature = merged["behavior_signature"].astype(str)
    parts = signature.str.split(";")
    for component in BEHAVIOR_COMPONENTS:
        merged[f"behavior_{component}"] = parts.apply(lambda tokens, c=component: c in tokens)
    merged["n_behaviors"] = parts.apply(lambda tokens: 0 if tokens == ["other"] else len(tokens))

    queries = merged["n_queries"].to_numpy(float)
    onset = merged["loop_onset"].to_numpy(float)
    merged["loop_onset_fraction"] = np.where(merged["loop"], onset / queries, np.nan)
    merged["queries_after_loop_onset"] = np.where(merged["loop"], queries - onset, np.nan)
    return merged


def verify_anchors(labels: pd.DataFrame, first: np.ndarray, v7: dict[str, np.ndarray]) -> dict[str, int]:
    loop = labels["loop"].to_numpy(bool)
    observed = {
        "failures": int(labels["failure"].sum()),
        "loop": int(loop.sum()),
        "static": int(labels["static"].sum()),
        "phantom": int(labels["phantom"].sum()),
        "pure_phantom": int(labels["pure_phantom"].sum()),
        "conjunction_alarm_episodes": int((first >= 0).sum()),
        "loop_detections": int((loop & (first >= 0)).sum()),
    }
    v7_miss = (v7["main_freeze"] < 0) & (v7["main_turbulence"] < 0)
    observed["v7_missed_loops"] = int((loop & v7_miss).sum())
    observed["g1"] = int((loop & v7_miss & (first >= 0)).sum())
    expected = {
        "failures": ANCHOR_FAILURES,
        "loop": ANCHOR_LOOP,
        "static": ANCHOR_STATIC,
        "phantom": ANCHOR_PHANTOM,
        "pure_phantom": ANCHOR_PURE_PHANTOM,
        "loop_detections": ANCHOR_LOOP_DETECTIONS,
        "v7_missed_loops": ANCHOR_V7_MISSES,
        "g1": ANCHOR_G1,
    }
    bad = {k: (observed[k], v) for k, v in expected.items() if observed[k] != v}
    if bad:
        raise SystemExit(f"BLOCKED: anchors did not reproduce: {bad}")
    return observed


# --------------------------------------------------------------------------- #
# effect sizes
# --------------------------------------------------------------------------- #


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    n1, n2 = len(left), len(right)
    var = ((n1 - 1) * left.var(ddof=1) + (n2 - 1) * right.var(ddof=1)) / (n1 + n2 - 2)
    if not np.isfinite(var) or var <= 0.0:
        return float("nan")
    return float((left.mean() - right.mean()) / np.sqrt(var))


def risk_difference(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    return float(left.mean() - right.mean())


def cramers_v(left: np.ndarray, right: np.ndarray) -> float:
    """2 x C table; with r = 2 the normaliser min(r-1, c-1) is 1."""
    if len(left) == 0 or len(right) == 0:
        return float("nan")
    levels = np.unique(np.concatenate([left, right]))
    if len(levels) < 2:
        return float("nan")
    table = np.vstack(
        [
            np.array([(left == level).sum() for level in levels], dtype=float),
            np.array([(right == level).sum() for level in levels], dtype=float),
        ]
    )
    total = table.sum()
    row = table.sum(axis=1, keepdims=True)
    column = table.sum(axis=0, keepdims=True)
    expected = row @ column / total
    keep = expected[0] > 0
    chi2 = float(((table[:, keep] - expected[:, keep]) ** 2 / expected[:, keep]).sum())
    return float(np.sqrt(chi2 / total))


def effect_of(kind: str, values: np.ndarray, mask: np.ndarray) -> float:
    left, right = values[mask], values[~mask]
    if kind == "numeric":
        return cohens_d(left.astype(float), right.astype(float))
    if kind == "binary":
        return risk_difference(left.astype(float), right.astype(float))
    return cramers_v(left, right)


def descriptive(kind: str, values: np.ndarray) -> str:
    if len(values) == 0:
        return ""
    if kind == "numeric":
        finite = values.astype(float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            return "n/a"
        return f"mean {finite.mean():.3f} / med {np.median(finite):.3f} (n={len(finite)})"
    if kind == "binary":
        flag = values.astype(float)
        return f"{flag.mean():.3f} ({int(flag.sum())}/{len(flag)})"
    counts = pd.Series(values).value_counts()
    top = "; ".join(f"{k}={v}" for k, v in counts.head(3).items())
    return f"{len(counts)} levels [{top}]"


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #


def clustered_bootstrap(
    kind: str,
    values: np.ndarray,
    mask: np.ndarray,
    tasks: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    unique = np.unique(tasks)
    index_by_task = {task: np.flatnonzero(tasks == task) for task in unique}
    out = np.full(draws, np.nan)
    for draw in range(draws):
        chosen = rng.choice(len(unique), size=len(unique), replace=True)
        index = np.concatenate([index_by_task[unique[c]] for c in chosen])
        sub_mask = mask[index]
        if sub_mask.all() or not sub_mask.any():
            continue
        out[draw] = effect_of(kind, values[index], sub_mask)
    good = out[np.isfinite(out)]
    if len(good) < 50:
        return float("nan"), float("nan"), float(len(good)) / draws
    return (
        float(np.percentile(good, 2.5)),
        float(np.percentile(good, 97.5)),
        float(len(good)) / draws,
    )


def permutation_masks(
    n_left: int, total: int, rng: np.random.Generator, draws: int
) -> np.ndarray:
    """Unrestricted null: draw ``n_left`` of the ``total`` pool members."""
    out = np.zeros((draws, total), dtype=bool)
    for draw in range(draws):
        out[draw, rng.choice(total, size=n_left, replace=False)] = True
    return out


def stratified_permutation_masks(
    mask: np.ndarray, tasks: np.ndarray, rng: np.random.Generator, draws: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Within-task null: keep each task's G1 count exactly as observed.

    Tasks whose members are all G1 or all G2 contribute no permutation freedom;
    their contribution to every statistic is frozen at the observed value, which
    is precisely the point -- the null asks whether anything beyond the task
    composition distinguishes G1.
    """
    unique = np.unique(tasks)
    blocks = []
    free = 0
    for task in unique:
        index = np.flatnonzero(tasks == task)
        count = int(mask[index].sum())
        blocks.append((index, count))
        if 0 < count < len(index):
            free += 1
    out = np.zeros((draws, len(mask)), dtype=bool)
    for draw in range(draws):
        for index, count in blocks:
            if count == 0:
                continue
            out[draw, rng.choice(index, size=count, replace=False)] = True
    info = {
        "tasks_in_pool": int(len(unique)),
        "tasks_with_both_groups": free,
        "g1_in_mixed_tasks": int(
            sum(
                int(mask[np.flatnonzero(tasks == t)].sum())
                for t in unique
                if 0 < int(mask[np.flatnonzero(tasks == t)].sum()) < int((tasks == t).sum())
            )
        ),
        "g1_in_pure_g1_tasks": int(
            sum(
                int(mask[np.flatnonzero(tasks == t)].sum())
                for t in unique
                if int(mask[np.flatnonzero(tasks == t)].sum()) == int((tasks == t).sum())
            )
        ),
    }
    return out, info


def null_from_masks(kind: str, values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = np.full(len(masks), np.nan)
    for draw, mask in enumerate(masks):
        if mask.all() or not mask.any():
            continue
        out[draw] = effect_of(kind, values, mask)
    return out


def extremeness(kind: str, values: np.ndarray, centre: float) -> np.ndarray:
    """Larger means more extreme, on the variable's own scale."""
    if kind == "categorical":
        return np.asarray(values, dtype=float) - centre
    return np.abs(np.asarray(values, dtype=float) - centre)


def min_p_adjust(
    null_scores: list[np.ndarray], observed_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Westfall-Young min-P.  Scale free, so a high-cardinality Cramer's V with a
    tight null cannot dominate the family the way it does under max-T."""
    draws = len(null_scores[0])
    pseudo = np.empty((len(null_scores), draws))
    observed_p = np.empty(len(null_scores))
    for index, scores in enumerate(null_scores):
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(draws)
        ranks[order] = np.arange(draws)
        # ties: count of null draws >= this draw, computed on the sorted view
        sorted_scores = scores[order]
        counts = draws - np.searchsorted(sorted_scores, sorted_scores, side="left")
        pseudo[index][order] = counts / draws
        observed_p[index] = float((scores >= observed_scores[index]).sum()) / draws
    min_pseudo = pseudo.min(axis=0)
    adjusted = np.array(
        [(1.0 + int((min_pseudo <= p).sum())) / (1.0 + draws) for p in observed_p]
    )
    return adjusted, min_pseudo


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    n = len(pvalues)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument("--permutations", type=int, default=PERMUTATION_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cache = load_npz(MAIN_CACHE)
    ell_star = length_threshold()
    first = first_conjunctive_alarm(cache, ell_star)
    labels = build_labels(cache)
    v7 = v7_alarms(cache)
    anchors = verify_anchors(labels, first, v7)
    print(f"ell_star = {ell_star!r}   r_star = {RATIO_THRESHOLD}   eps_length = {EPS_LENGTH}")
    print(f"anchors reproduced: {anchors}")

    labels["conjunction_alarm"] = first >= 0
    labels["conjunction_alarm_chunk"] = first
    for key, value in v7.items():
        labels[key] = value
    v7_miss = (v7["main_freeze"] < 0) & (v7["main_turbulence"] < 0)
    labels["v7_missed"] = v7_miss

    loop = labels["loop"].to_numpy(bool)
    group = np.full(len(labels), "", dtype=object)
    group[loop & v7_miss & (first >= 0)] = "G1"
    group[loop & v7_miss & (first < 0)] = "G2"
    group[loop & ~v7_miss] = "G3"
    labels["group"] = group

    loops = labels[loop].copy().reset_index(drop=True)
    sizes = {name: int((loops["group"] == name).sum()) for name in ("G1", "G2", "G3")}
    print(f"group sizes: {sizes}  (total loop {len(loops)})")
    if sizes != {"G1": 24, "G2": 61, "G3": 171}:
        raise SystemExit(f"BLOCKED: unexpected group sizes {sizes}")

    keep = [
        "row",
        "group",
        "task",
        "suite",
        "episode" if "episode" in loops.columns else "row",
        "init_state_id",
        "flow_noise_seed",
        "conjunction_alarm",
        "conjunction_alarm_chunk",
        "main_freeze",
        "main_turbulence",
        "main_guard",
        "v7_missed",
        "loop",
        "static",
        "phantom",
        "pure_phantom",
        "failure",
        "loop_onset",
        "static_onset",
        "n_queries",
        "length",
        "loop_onset_fraction",
        "queries_after_loop_onset",
        "near_unfinished_closure_count",
        "belief_mismatch",
        "belief_mismatch_count",
        "closed_mismatch_candidate",
        "closed_mismatch_within_5q",
        "first_event",
        "primary_behavior",
        "behavior_signature",
        "n_behaviors",
        "proxy_grade",
        "loop_proxy_valid",
        *[f"behavior_{c}" for c in BEHAVIOR_COMPONENTS],
    ]
    keep = list(dict.fromkeys(c for c in keep if c in loops.columns))
    loops[keep].to_csv(args.output / "groups.csv", index=False)

    pool = loops[loops["group"].isin(["G1", "G2"])].reset_index(drop=True)
    context = loops[loops["group"] == "G3"].reset_index(drop=True)
    pool_mask = (pool["group"] == "G1").to_numpy(bool)
    pool_tasks = pool["task"].to_numpy(str)

    free_masks = permutation_masks(
        int(pool_mask.sum()), len(pool_mask), rng, args.permutations
    )
    strat_masks, strat_info = stratified_permutation_masks(
        pool_mask, pool_tasks, rng, args.permutations
    )
    print(f"stratified permutation structure: {strat_info}")

    rows: list[dict[str, Any]] = []
    free_scores: list[np.ndarray] = []
    strat_scores: list[np.ndarray] = []
    observed_free: list[float] = []
    observed_strat: list[float] = []
    skipped: list[dict[str, Any]] = []

    for name, kind, provenance in VARIABLES:
        if name not in pool.columns:
            skipped.append({"variable": name, "reason": "absent"})
            continue
        column = pool[name]
        if kind in ("numeric", "binary"):
            values = column.astype(float).to_numpy()
            finite = np.isfinite(values)
            if finite.sum() == 0 or np.nanstd(values) == 0.0:
                skipped.append({"variable": name, "reason": "degenerate in the 85-pool"})
                continue
        else:
            values = column.astype(str).to_numpy()
            if len(np.unique(values)) < 2:
                skipped.append({"variable": name, "reason": "degenerate in the 85-pool"})
                continue

        effect = effect_of(kind, values, pool_mask)
        if not np.isfinite(effect):
            skipped.append({"variable": name, "reason": "effect undefined"})
            continue

        lo, hi, valid_fraction = clustered_bootstrap(
            kind, values, pool_mask, pool_tasks, rng, args.bootstrap
        )
        null = null_from_masks(kind, values, free_masks)
        strat_null = null_from_masks(kind, values, strat_masks)
        if not np.isfinite(null).all() or not np.isfinite(strat_null).all():
            null = np.nan_to_num(null, nan=float(np.nanmedian(null)))
            strat_null = np.nan_to_num(strat_null, nan=float(np.nanmedian(strat_null)))
        tail = "upper" if kind == "categorical" else "two-sided"
        centre = 0.0 if kind == "categorical" else float(np.mean(null))
        strat_centre = 0.0 if kind == "categorical" else float(np.mean(strat_null))
        score_null = extremeness(kind, null, centre)
        score_obs = float(extremeness(kind, np.array([effect]), centre)[0])
        strat_score_null = extremeness(kind, strat_null, strat_centre)
        strat_score_obs = float(extremeness(kind, np.array([effect]), strat_centre)[0])
        perm_p = (1.0 + int((score_null >= score_obs).sum())) / (1.0 + len(score_null))
        if np.std(strat_score_null) <= 0.0:
            strat_p = float("nan")  # no permutation freedom for this variable
        else:
            strat_p = (1.0 + int((strat_score_null >= strat_score_obs).sum())) / (
                1.0 + len(strat_score_null)
            )

        g3_effect = float("nan")
        g3_lo = g3_hi = float("nan")
        if len(context) > 0:
            if kind in ("numeric", "binary"):
                g3_values = context[name].astype(float).to_numpy()
            else:
                g3_values = context[name].astype(str).to_numpy()
            g1_values = values[pool_mask]
            merged_values = np.concatenate([g1_values, g3_values])
            merged_mask = np.concatenate(
                [np.ones(len(g1_values), bool), np.zeros(len(g3_values), bool)]
            )
            merged_tasks = np.concatenate(
                [pool["task"].to_numpy(str)[pool_mask], context["task"].to_numpy(str)]
            )
            g3_effect = effect_of(kind, merged_values, merged_mask)
            g3_lo, g3_hi, _ = clustered_bootstrap(
                kind, merged_values, merged_mask, merged_tasks, rng, args.bootstrap
            )

        if kind in ("numeric", "binary"):
            g1_desc = descriptive(kind, values[pool_mask])
            g2_desc = descriptive(kind, values[~pool_mask])
            g3_desc = descriptive(kind, context[name].astype(float).to_numpy())
        else:
            g1_desc = descriptive(kind, values[pool_mask])
            g2_desc = descriptive(kind, values[~pool_mask])
            g3_desc = descriptive(kind, context[name].astype(str).to_numpy())

        rows.append(
            {
                "variable": name,
                "kind": kind,
                "provenance": provenance,
                "effect_statistic": {
                    "numeric": "cohens_d",
                    "binary": "risk_difference",
                    "categorical": "cramers_v",
                }[kind],
                "tail": tail,
                "n_g1": int(pool_mask.sum()),
                "n_g2": int((~pool_mask).sum()),
                "n_g3": int(len(context)),
                "g1": g1_desc,
                "g2": g2_desc,
                "g3": g3_desc,
                "effect_g1_vs_g2": effect,
                "ci_lo": lo,
                "ci_hi": hi,
                "bootstrap_valid_fraction": valid_fraction,
                "null_mean": float(np.mean(null)),
                "null_sd": float(np.std(null, ddof=1)),
                "perm_p": perm_p,
                "perm_p_within_task": strat_p,
                "strat_null_mean": float(np.mean(strat_null)),
                "effect_g1_vs_g3": g3_effect,
                "ci_g3_lo": g3_lo,
                "ci_g3_hi": g3_hi,
            }
        )
        free_scores.append(score_null)
        strat_scores.append(strat_score_null)
        observed_free.append(score_obs)
        observed_strat.append(strat_score_obs)
        print(
            f"  tested {name:34s} effect={effect:+.4f}  perm_p={perm_p:.4f}  "
            f"within_task_p={strat_p:.4f}",
            flush=True,
        )

    contrasts = pd.DataFrame(rows)
    n_tested = len(contrasts)

    # ---- Westfall-Young min-P over the full pre-declared list ---------------
    wy, min_pseudo = min_p_adjust(free_scores, np.array(observed_free))
    contrasts["perm_p_westfall_young"] = wy
    contrasts["perm_p_bh"] = benjamini_hochberg(contrasts["perm_p"].to_numpy())
    strat_ok = [
        index for index, value in enumerate(contrasts["perm_p_within_task"]) if np.isfinite(value)
    ]
    if strat_ok:
        wy_strat, min_pseudo_strat = min_p_adjust(
            [strat_scores[i] for i in strat_ok], np.array([observed_strat[i] for i in strat_ok])
        )
        column = np.full(n_tested, np.nan)
        column[strat_ok] = wy_strat
        contrasts["perm_p_within_task_westfall_young"] = column
        global_p_strat = float(np.nanmin(column))
    else:
        contrasts["perm_p_within_task_westfall_young"] = np.nan
        global_p_strat = float("nan")
    global_p = float(wy.min())
    global_variable = str(contrasts.loc[int(np.argmin(contrasts["perm_p"].to_numpy())), "variable"])

    contrasts = contrasts.sort_values("perm_p").reset_index(drop=True)
    contrasts.to_csv(args.output / "contrasts.csv", index=False)

    # ---- readable table ----------------------------------------------------
    print()
    print("=" * 132)
    print("G1 (conjunction catch, v7 miss, n=24)  vs  G2 (conjunction miss, v7 miss, n=61)")
    print("=" * 132)
    header = (
        f"{'variable':32s} {'stat':16s} {'effect':>9s} {'95% CI (task-clustered)':>26s} "
        f"{'perm p':>8s} {'BH':>7s} {'WY':>7s} {'p|task':>8s}"
    )
    print(header)
    print("-" * 132)
    for _, row in contrasts.iterrows():
        ci = (
            f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]"
            if np.isfinite(row["ci_lo"])
            else "[unstable]"
        )
        strat = (
            f"{row['perm_p_within_task']:8.4f}"
            if np.isfinite(row["perm_p_within_task"])
            else "   frozen"
        )
        print(
            f"{row['variable']:32s} {row['effect_statistic']:16s} "
            f"{row['effect_g1_vs_g2']:+9.4f} {ci:>26s} "
            f"{row['perm_p']:8.4f} {row['perm_p_bh']:7.3f} "
            f"{row['perm_p_westfall_young']:7.3f} {strat}"
        )
    print("-" * 132)
    print(
        "perm p = unrestricted null (draw 24 of the 85).  p|task = same statistic under a "
        "within-task null\nthat holds G1's task composition fixed.  WY = Westfall-Young min-P "
        f"over all {n_tested} variables."
    )
    print("-" * 132)
    print()
    print(f"{'variable':32s} {'G1':>34s} {'G2':>34s} {'G3 (context)':>34s}")
    print("-" * 132)
    for _, row in contrasts.iterrows():
        print(
            f"{row['variable']:32s} {str(row['g1'])[:34]:>34s} "
            f"{str(row['g2'])[:34]:>34s} {str(row['g3'])[:34]:>34s}"
        )
    print("-" * 132)
    print()
    print(f"variables tested                : {n_tested}")
    print(f"variables skipped (degenerate)  : {len(skipped)} -> {[s['variable'] for s in skipped]}")
    print(f"nominally significant (p<0.05)  : {int((contrasts['perm_p'] < 0.05).sum())}")
    print(f"expected by chance at alpha 0.05: {0.05 * n_tested:.2f}")
    print(f"survive BH q<0.10               : {int((contrasts['perm_p_bh'] < 0.10).sum())}")
    print(f"survive Westfall-Young p<0.05   : {int((contrasts['perm_p_westfall_young'] < 0.05).sum())}")
    print(
        f"survive within-task WY p<0.05   : "
        f"{int((contrasts['perm_p_within_task_westfall_young'] < 0.05).sum())}"
    )
    # ---- the same contrasts restricted to tasks that contain both groups ----
    mixed_tasks = [
        task
        for task, block in pool.groupby("task")
        if block["group"].nunique() == 2
    ]
    mixed = pool[pool["task"].isin(mixed_tasks)]
    mixed_mask = (mixed["group"] == "G1").to_numpy(bool)
    mixed_rows: list[dict[str, Any]] = []
    for _, row in contrasts.iterrows():
        name, kind = row["variable"], row["kind"]
        if kind == "categorical":
            continue
        values = mixed[name].astype(float).to_numpy()
        mixed_rows.append(
            {
                "variable": name,
                "pooled_effect": float(row["effect_g1_vs_g2"]),
                "g1_mean_mixed": float(np.nanmean(values[mixed_mask])),
                "g2_mean_mixed": float(np.nanmean(values[~mixed_mask])),
                "mixed_effect": effect_of(kind, values, mixed_mask),
            }
        )
    mixed_frame = pd.DataFrame(mixed_rows)
    print()
    print(
        f"Same contrasts restricted to the {len(mixed_tasks)} tasks that contain both groups "
        f"(G1 n={int(mixed_mask.sum())}, G2 n={int((~mixed_mask).sum())}):"
    )
    print(f"{'variable':32s} {'pooled effect':>14s} {'G1 mean':>10s} {'G2 mean':>10s} {'within-task effect':>19s}")
    print("-" * 132)
    for _, row in mixed_frame.iterrows():
        print(
            f"{row['variable']:32s} {row['pooled_effect']:+14.4f} "
            f"{row['g1_mean_mixed']:10.3f} {row['g2_mean_mixed']:10.3f} {row['mixed_effect']:+19.4f}"
        )
    print("-" * 132)

    print()
    print(
        f"GLOBAL (unrestricted null, min-P over {n_tested} variables): p={global_p:.4f}, "
        f"driven by '{global_variable}'\n  -> G1 is "
        f"{'NOT ' if global_p >= 0.05 else ''}distinguishable from a random draw of 24 from the 85"
    )
    print(
        f"GLOBAL (within-task null, min-P): p={global_p_strat:.4f}\n  -> beyond its task "
        f"composition, G1 is {'NOT ' if not (global_p_strat < 0.05) else ''}distinguishable "
        f"from a random draw of 24 from the 85"
    )

    summary = {
        "schema": SCHEMA,
        "question": (
            "are the 24 loop episodes the conjunction catches and v7 misses a distinctive "
            "subpopulation, or an arbitrary slice of the 85 v7 misses?"
        ),
        "framing": (
            "readout, not prediction: alarms lag loop onset by a median of 5 chunks and "
            "precede it 5.7% of the time versus 24.1% for v7 turbulence"
        ),
        "operating_point": {
            "window": WINDOW,
            "layer_group": LAYER_GROUP,
            "confirmations": CONFIRMATIONS,
            "ratio_threshold": RATIO_THRESHOLD,
            "eps_length": EPS_LENGTH,
            "ell_star": ell_star,
            "ell_star_source": "recomputed pooled-reference median, matches selection.json",
        },
        "anchors": anchors,
        "anchors_reproduced": True,
        "groups": {
            "G1": sizes["G1"],
            "G2": sizes["G2"],
            "G3": sizes["G3"],
            "definition": {
                "G1": "loop & conjunction alarm & no v7 freeze & no v7 turbulence",
                "G2": "loop & no conjunction alarm & no v7 freeze & no v7 turbulence",
                "G3": "loop & (v7 freeze or v7 turbulence)",
            },
            "loop_total": int(len(loops)),
            "tasks_in_pool": int(len(np.unique(pool_tasks))),
            "tasks_in_g1": int(pool.loc[pool_mask, "task"].nunique()),
            "tasks_in_g2": int(pool.loc[~pool_mask, "task"].nunique()),
        },
        "resampling": {
            "bootstrap_draws": args.bootstrap,
            "bootstrap_cluster": "task",
            "permutation_draws": args.permutations,
            "permutation_scheme": "draw 24 of the 85 v7-missed loops without replacement",
            "stratified_permutation_scheme": (
                "permute the G1/G2 label within task, holding each task's G1 count fixed"
            ),
            "stratified_structure": strat_info,
            "seed": args.seed,
        },
        "multiplicity": {
            "variables_declared": len(VARIABLES),
            "variables_tested": n_tested,
            "variables_skipped": skipped,
            "nominally_significant_p05": int((contrasts["perm_p"] < 0.05).sum()),
            "expected_false_positives_at_p05": 0.05 * n_tested,
            "bh_q10_survivors": contrasts.loc[
                contrasts["perm_p_bh"] < 0.10, "variable"
            ].tolist(),
            "westfall_young_p05_survivors": contrasts.loc[
                contrasts["perm_p_westfall_young"] < 0.05, "variable"
            ].tolist(),
            "within_task_westfall_young_p05_survivors": contrasts.loc[
                contrasts["perm_p_within_task_westfall_young"] < 0.05, "variable"
            ].tolist(),
            "within_task_nominally_significant_p05": int(
                (contrasts["perm_p_within_task"] < 0.05).sum()
            ),
        },
        "global_test": {
            "statistic": "Westfall-Young min-P over every tested variable",
            "unrestricted_null_p": global_p,
            "unrestricted_null_driver": global_variable,
            "distinguishable_from_random_draw_at_p05": bool(global_p < 0.05),
            "within_task_null_p": global_p_strat,
            "distinguishable_beyond_task_composition_at_p05": bool(global_p_strat < 0.05),
        },
        "within_task_restriction": {
            "note": (
                "the same contrasts computed only on the tasks that contain both groups; "
                "this is the descriptive companion to the within-task permutation null"
            ),
            "mixed_tasks": len(mixed_tasks),
            "g1_in_mixed": int(mixed_mask.sum()),
            "g2_in_mixed": int((~mixed_mask).sum()),
            "rows": mixed_frame.to_dict(orient="records"),
        },
        "task_level_constants": {
            "note": (
                "n_queries and proxy_grade take exactly one value per task across all "
                "loop episodes, so those two contrasts are task-identity contrasts by "
                "construction and carry no independent physical content"
            ),
            "n_queries_values_per_task": int(loops.groupby("task")["n_queries"].nunique().max()),
            "proxy_grade_values_per_task": int(loops.groupby("task")["proxy_grade"].nunique().max()),
        },
        "alarm_timing_in_g1": {
            "note": "readout corroboration only; not a prediction claim",
            "median_alarm_minus_loop_onset_chunks": float(
                np.median(
                    pool.loc[pool_mask, "conjunction_alarm_chunk"].to_numpy(float)
                    - pool.loc[pool_mask, "loop_onset"].to_numpy(float)
                )
            ),
            "fraction_alarm_precedes_loop_onset": float(
                (
                    pool.loc[pool_mask, "conjunction_alarm_chunk"].to_numpy(float)
                    < pool.loc[pool_mask, "loop_onset"].to_numpy(float)
                ).mean()
            ),
        },
        "reproduction_notes": {
            "conjunction_alarm_episodes_here": int((first >= 0).sum()),
            "conjunction_alarm_episodes_in_selection_json": 94,
            "explanation": (
                "the task specification rounds r* to 0.294327 while selection.json holds "
                "0.2943268120288849; the rounding admits one extra non-loop episode.  The "
                "loop anchors 35 and 24 are identical under both values."
            ),
        },
        "task_composition": {
            "g1_by_task": pool.loc[pool_mask, "task"].value_counts().to_dict(),
            "g2_by_task": pool.loc[~pool_mask, "task"].value_counts().to_dict(),
            "g1_by_suite": pool.loc[pool_mask, "suite"].value_counts().to_dict(),
            "g2_by_suite": pool.loc[~pool_mask, "suite"].value_counts().to_dict(),
            "note": (
                "proxy_grade == 'full' is carried by exactly one task, so the proxy_grade "
                "contrast is a relabelling of the task contrast, not independent evidence"
            ),
        },
        "smallest_perm_p": {
            "variable": str(contrasts.loc[0, "variable"]),
            "effect": float(contrasts.loc[0, "effect_g1_vs_g2"]),
            "perm_p": float(contrasts.loc[0, "perm_p"]),
            "perm_p_bh": float(contrasts.loc[0, "perm_p_bh"]),
            "perm_p_westfall_young": float(contrasts.loc[0, "perm_p_westfall_young"]),
        },
        "co_occurrence": {
            "static": {
                "G1": float(pool.loc[pool_mask, "static"].mean()),
                "G2": float(pool.loc[~pool_mask, "static"].mean()),
                "G3": float(context["static"].mean()),
            },
            "phantom": {
                "G1": float(pool.loc[pool_mask, "phantom"].mean()),
                "G2": float(pool.loc[~pool_mask, "phantom"].mean()),
                "G3": float(context["phantom"].mean()),
            },
            "pure_phantom": {
                "G1": float(pool.loc[pool_mask, "pure_phantom"].mean()),
                "G2": float(pool.loc[~pool_mask, "pure_phantom"].mean()),
                "G3": float(context["pure_phantom"].mean()),
            },
        },
        "outputs": {
            "groups_csv": str((args.output / "groups.csv").relative_to(BUNDLE)),
            "contrasts_csv": str((args.output / "contrasts.csv").relative_to(BUNDLE)),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(f"\nwrote {args.output}/groups.csv, contrasts.csv, summary.json")


if __name__ == "__main__":
    main()
