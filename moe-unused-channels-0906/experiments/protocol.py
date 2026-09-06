#!/usr/bin/env python3
"""Shared loaders and the frozen detector protocol, re-used not reimplemented.

Everything scoring-related is imported from the established bundles:

  moe-v4-0904/experiments/evaluate_layerwise_alarm_development.py
      trailing_mean, persistent_score, row_max, quantile_higher, first_query,
      crossfit_thresholds, representations, QUANTILES
  moe-hb-front-back-0905/experiments/select_early_lock.py
      survival_prior, prior_of, score_candidate, suite_of, MAX_TIMELY_FPR
  moe-hb-front-back-0905/experiments/compare_layers_early.py
      WIDTH, CONFIRMATIONS, MIN_LOW_PRIOR_PRECISION

The only thing defined here is the plumbing this bundle needs: paths, the anchor
assertions, and a generic "score one quantity through the frozen sweep" helper
that takes an arbitrary (episode, chunk, layer) array instead of the mobility
cache.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent

sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))
sys.path.insert(0, str(PROJECT / "moe-hb-front-back-0905/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from compare_layers_early import (  # noqa: E402
    CONFIRMATIONS,
    MIN_LOW_PRIOR_PRECISION,
    WIDTH,
)
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)

ROUTE_ROOT = PROJECT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
MOBILITY_ROOT = PROJECT / "moe-v4-0904/results/layerwise_mobility"
GRAPH_ROOT = PROJECT / "moe-hb-front-back-0905/results/layer_graphs"
LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
FRAME_ALARMS = (
    PROJECT / "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz"
)
STEP_PROFILES = PROJECT / "moe-flow-semantics-0906/results/step_profiles"
RESULTS = BUNDLE / "results"

COHORTS = ("development_main", "development_extra", "external_8b")
MOBILITY_PATHS = {
    "development_main": MOBILITY_ROOT / "main_reference.npz",
    "development_extra": MOBILITY_ROOT / "extra_reference.npz",
    "external_8b": MOBILITY_ROOT / "external_8b.npz",
}
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
REPRESENTATIONS = (
    "L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15",
    "front_median", "back_median", "all_median",
)
MODES = ("per_task", "global")
SEED = 20260906

# Declared in the brief, asserted before anything else runs.
ANCHORS = (
    # quantity, representation, direction, quantile, mode, tp, fp, lift
    ("mobility", "L12", "low", 0.975, "global", 195, 17, 1.7641),
    ("mobility", "L2", "low", 0.700, "per_task", 272, 57, None),
    ("expert_load_effective_rank", "L3", "low", 0.850, "per_task", 370, 93, 1.5479),
)
CAPS = {"libero_goal": 30, "libero_long": 52, "libero_object": 28, "libero_spatial": 22}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(payload), indent=2, sort_keys=True) + "\n", "utf-8")


# --------------------------------------------------------------------------- #
# cohort assembly


def cohort_frames() -> dict[str, dict[str, Any]]:
    """Row-aligned metadata for the three cohorts, with the alignment asserted."""
    output: dict[str, dict[str, Any]] = {}
    for cohort in COHORTS:
        mobility = load_npz(MOBILITY_PATHS[cohort])
        graph_name = {"development_extra": "development_extra"}.get(cohort, cohort)
        graph = load_npz(GRAPH_ROOT / f"{graph_name}.npz")
        for key in ("episode", "task_index", "length"):
            if not np.array_equal(graph[key], mobility[key]):
                raise AssertionError(f"{cohort}: graph/mobility disagree on {key}")
        if not np.array_equal(
            graph["task_names"].astype(str), mobility["task_names"].astype(str)
        ):
            raise AssertionError(f"{cohort}: graph/mobility task_names disagree")
        task = mobility["task_names"].astype(str)[mobility["task_index"].astype(int)]
        entry: dict[str, Any] = {
            "cohort": cohort,
            "run_id": str(mobility["run_id"]),
            "mobility": mobility,
            "graph": graph,
            "task": task,
            "suite": np.asarray([name.split("/", 1)[0] for name in task]),
            "episode": mobility["episode"].astype(int),
            "length": mobility["length"].astype(int),
            "task_index": mobility["task_index"].astype(int),
            "init_state_id": mobility["init_state_id"].astype(int),
            "valid": mobility["valid"].astype(bool),
            "task_names": mobility["task_names"].astype(str),
        }
        if cohort in LABEL_PATHS:
            labels = pd.read_csv(LABEL_PATHS[cohort]).reset_index(drop=True)
            if len(labels) != len(task):
                raise AssertionError(f"{cohort}: label row count")
            if not np.array_equal(labels["task"].astype(str).to_numpy(), task):
                raise AssertionError(f"{cohort}: label task alignment")
            if not np.array_equal(
                labels["episode"].to_numpy(dtype=int), entry["episode"]
            ):
                raise AssertionError(f"{cohort}: label episode alignment")
            entry["labels"] = labels
            entry["risk"] = labels["original_failure"].to_numpy(bool)
            entry["priors"] = survival_prior(
                entry["suite"], entry["length"], entry["risk"]
            )
            at_cap = np.asarray(
                [entry["length"][i] == CAPS[entry["suite"][i]] for i in range(len(task))]
            )
            entry["at_cap"] = at_cap
        output[cohort] = entry
    if int(output["development_main"]["risk"].sum()) != 487:
        raise AssertionError("development_main risk count is not 487")
    if int(output["external_8b"]["risk"].sum()) != 564:
        raise AssertionError("external_8b risk count is not 564")
    if len(output["development_main"]["episode"]) != 14800:
        raise AssertionError("development_main is not 14800 episodes")
    if len(output["external_8b"]["episode"]) != 15600:
        raise AssertionError("external_8b is not 15600 episodes")
    return output


def layer_cache(frame: dict[str, Any], values: np.ndarray) -> dict[str, np.ndarray]:
    """Wrap an (episode, chunk, layer) array in the layout dev.representations wants."""
    return {
        "mobility": np.asarray(values, dtype=np.float32),
        "valid": frame["valid"],
        "layer_names": np.asarray(REPRESENTATIONS[:8]),
        "task_names": frame["task_names"],
        "task_index": frame["task_index"],
        "episode": frame["episode"],
        "init_state_id": frame["init_state_id"],
        "length": frame["length"],
    }


def quantity_values(frame: dict[str, Any], quantity: str) -> np.ndarray:
    """The frozen (episode, chunk, layer) array for an established quantity."""
    if quantity == "mobility":
        return np.asarray(frame["mobility"]["mobility"], dtype=np.float32)
    graph = frame["graph"]
    names = graph["metric_names"].astype(str).tolist()
    return np.asarray(
        graph["metrics"][:, :, :, names.index(quantity)], dtype=np.float32
    )


# --------------------------------------------------------------------------- #
# the frozen sweep, applied to an arbitrary quantity


def oriented(values: np.ndarray, direction: str, width: int = WIDTH) -> np.ndarray:
    smoothed = dev.trailing_mean(values, width)
    return -smoothed if direction == "low" else smoothed


def development_alarm(
    frame: dict[str, Any],
    values: np.ndarray,
    direction: str,
    quantile: float,
    mode: str,
    pooled_peak: np.ndarray | None = None,
) -> np.ndarray:
    signal = oriented(values, direction)
    persistent = dev.persistent_score(signal, CONFIRMATIONS)
    if mode == "per_task":
        line = dev.crossfit_thresholds(
            dev.row_max(signal), frame["task_index"], frame["init_state_id"]
        )[:, dev.QUANTILES.index(quantile)][:, None]
    else:
        line = dev.quantile_higher(pooled_peak, quantile)
    return dev.first_query(
        np.isfinite(persistent) & (persistent > line) & frame["valid"]
    )


def external_alarm(
    external: dict[str, Any],
    external_values: np.ndarray,
    reference: list[tuple[dict[str, Any], np.ndarray]],
    direction: str,
    quantile: float,
    mode: str,
) -> np.ndarray:
    """Replay a development-selected head on external.

    `reference` is a list of (frame, values) pairs from development only.
    """
    persistent = dev.persistent_score(oriented(external_values, direction), CONFIRMATIONS)
    first = np.full(len(external_values), -1, dtype=np.int16)
    if mode == "global":
        pooled = np.concatenate(
            [dev.row_max(oriented(values, direction)) for _, values in reference]
        )
        line = dev.quantile_higher(pooled, quantile)
        return dev.first_query(
            np.isfinite(persistent) & (persistent > line) & external["valid"]
        )
    for task in np.unique(external["task"]):
        take = np.flatnonzero(external["task"] == task)
        peaks = [
            dev.row_max(oriented(values[frame["task"] == task], direction))
            for frame, values in reference
            if (frame["task"] == task).any()
        ]
        if not peaks:
            raise AssertionError(f"external task with no reference: {task}")
        line = dev.quantile_higher(np.concatenate(peaks), quantile)
        first[take] = dev.first_query(
            np.isfinite(persistent[take])
            & (persistent[take] > line)
            & external["valid"][take]
        )
    return first


def sweep_development(
    frames: dict[str, dict[str, Any]],
    quantity: str,
    values: dict[str, np.ndarray],
    representations: tuple[str, ...] = REPRESENTATIONS,
) -> pd.DataFrame:
    """Score one quantity across representations x directions x quantiles x modes."""
    main = frames["development_main"]
    reprs = {
        cohort: {
            name: block
            for name, (block, _) in dev.representations(
                layer_cache(frames[cohort], values[cohort])
            ).items()
        }
        for cohort in ("development_main", "development_extra")
    }
    rows: list[dict[str, Any]] = []
    for representation in representations:
        for direction in ("low", "high"):
            pooled_peak = np.concatenate(
                [
                    dev.row_max(oriented(reprs[c][representation], direction))
                    for c in ("development_main", "development_extra")
                ]
            )
            for quantile in dev.QUANTILES:
                for mode in MODES:
                    first = development_alarm(
                        main,
                        reprs["development_main"][representation],
                        direction,
                        quantile,
                        mode,
                        pooled_peak,
                    )
                    rows.append(
                        {
                            "quantity": quantity,
                            "representation": representation,
                            "direction": direction,
                            "quantile": quantile,
                            "mode": mode,
                            **score_candidate(
                                first,
                                main["risk"],
                                prior_of(first, main["suite"], main["priors"]),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def select_head(candidates: pd.DataFrame, mode: str) -> pd.Series | None:
    eligible = candidates[
        (candidates["mode"] == mode)
        & (candidates["timely_fpr"] <= MAX_TIMELY_FPR)
        & (candidates["low_prior_precision"] >= MIN_LOW_PRIOR_PRECISION)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
    ).iloc[0]


# --------------------------------------------------------------------------- #
# anchors


def assert_anchors(frames: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    external = frames["external_8b"]
    records: list[dict[str, Any]] = []
    for quantity, representation, direction, quantile, mode, tp, fp, lift in ANCHORS:
        reference = [
            (
                frames[cohort],
                {
                    name: block
                    for name, (block, _) in dev.representations(
                        layer_cache(frames[cohort], quantity_values(frames[cohort], quantity))
                    ).items()
                }[representation],
            )
            for cohort in ("development_main", "development_extra")
        ]
        external_values = {
            name: block
            for name, (block, _) in dev.representations(
                layer_cache(external, quantity_values(external, quantity))
            ).items()
        }[representation]
        first = external_alarm(
            external, external_values, reference, direction, quantile, mode
        )
        score = score_candidate(
            first, external["risk"], prior_of(first, external["suite"], external["priors"])
        )
        name = f"{quantity}|{representation}|{direction}|q{quantile:g}|{mode}"
        if int(score["tp"]) != tp or int(score["fp"]) != fp:
            raise AssertionError(
                f"anchor {name}: expected {tp} TP / {fp} FP, got "
                f"{score['tp']} / {score['fp']}"
            )
        if lift is not None and abs(float(score["lift"]) - lift) > 5e-5:
            raise AssertionError(
                f"anchor {name}: expected lift {lift}, got {score['lift']:.6f}"
            )
        records.append({"anchor": name, "tp": tp, "fp": fp, "lift": score["lift"]})
        print(f"  anchor OK  {name}: {tp} TP / {fp} FP  lift {score['lift']:.4f}", flush=True)
    return records


# --------------------------------------------------------------------------- #
# stratified survival-conditioned AUC (moe-audit-0906 estimator, verbatim logic)

CHUNKS = (4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32)
MIN_STRATUM = 30


def mann_whitney(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    n1, n0 = len(positive), len(negative)
    if n1 == 0 or n0 == 0:
        return float("nan"), 0.0
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    index = 0
    while index < len(sorted_values):
        stop = index
        while stop + 1 < len(sorted_values) and sorted_values[stop + 1] == sorted_values[index]:
            stop += 1
        ranks[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0), float(n1 * n0)


def stratified_auc(
    values: np.ndarray, risk: np.ndarray, strata: np.ndarray | None, alive: np.ndarray
) -> tuple[float, float]:
    numerator = denominator = 0.0
    keys = [None] if strata is None else np.unique(strata[alive])
    for key in keys:
        mask = alive if key is None else (alive & (strata == key))
        mask = mask & np.isfinite(values)
        positive, negative = values[mask & risk], values[mask & ~risk]
        if not len(positive) or not len(negative):
            continue
        if strata is not None and (len(positive) + len(negative)) < MIN_STRATUM:
            continue
        area, weight = mann_whitney(positive, negative)
        if np.isfinite(area):
            numerator += area * weight
            denominator += weight
    return (numerator / denominator if denominator else float("nan")), denominator


def survival_auc_rows(
    frame: dict[str, Any], quantity: str, values: np.ndarray, layer_names: tuple[str, ...]
) -> list[dict[str, Any]]:
    """values is (episode, chunk, layer); W4 trailing mean, then AUC at each chunk."""
    masked = np.where(frame["valid"][:, :, None], values, np.nan)
    smoothed = np.stack(
        [dev.trailing_mean(masked[:, :, k], WIDTH) for k in range(values.shape[2])],
        axis=2,
    )
    rows: list[dict[str, Any]] = []
    risk = frame["risk"]
    for position, layer in enumerate(layer_names):
        for chunk in CHUNKS:
            if chunk >= values.shape[1]:
                continue
            alive = frame["length"] > chunk
            if alive.sum() < 50:
                continue
            column = smoothed[:, chunk, position]
            pooled, _ = stratified_auc(column, risk, None, alive)
            suite, _ = stratified_auc(column, risk, frame["suite"], alive)
            task, _ = stratified_auc(column, risk, frame["task"], alive)
            rows.append(
                {
                    "cohort": frame["cohort"],
                    "quantity": quantity,
                    "layer": layer,
                    "chunk": chunk,
                    "alive": int(alive.sum()),
                    "alive_risk": int((alive & risk).sum()),
                    "prior": float((alive & risk).sum() / alive.sum()),
                    "auc_pooled": pooled,
                    "auc_within_suite": suite,
                    "auc_within_task": task,
                }
            )
    return rows


def dependence_ratio(
    left: np.ndarray, right: np.ndarray, timely: np.ndarray
) -> float:
    """Observed co-occurrence over the product of marginals, timely episodes only."""
    a = (left >= 0) & timely
    b = (right >= 0) & timely
    n = int(timely.sum())
    expected = a.sum() * b.sum() / n
    return float((a & b).sum() / expected) if expected > 0 else float("nan")


def overlap_share(left: np.ndarray, right: np.ndarray, timely: np.ndarray) -> float:
    """Raw share of the smaller false-alarm set that the other also flags."""
    a = (left >= 0) & timely
    b = (right >= 0) & timely
    smaller = min(int(a.sum()), int(b.sum()))
    return float((a & b).sum() / smaller) if smaller else float("nan")
