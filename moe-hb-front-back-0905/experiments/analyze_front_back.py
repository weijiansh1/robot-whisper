#!/usr/bin/env python3
"""Evaluate layer-resolved HB structure and its deadline-risk value.

Feature construction and threshold calibration are outcome-blind.  Outcomes are
loaded only after all fixed layer/group scores have been constructed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
RESULTS = BUNDLE / "results"
PROFILE_ROOT = RESULTS / "layer_graphs"
V4 = PROJECT / "moe-v4-0904"
MOBILITY_ROOT = V4 / "results/layerwise_mobility"
LABEL_ROOT = PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
ALARM_TABLE = V4 / "results/cache_new_v4/episode_alarms.csv"
SEALED = V4 / "results/cache_new_v4/sealed_first_alarms.npz"
OLD_TRANSITION = (
    PROJECT
    / "moe-feedback-responsiveness-0905/results/fusion/transition_scores.csv"
)

sys.path.insert(0, str(V4 / "experiments"))
import evaluate_layerwise_alarm_development as dev  # noqa: E402


MOBILITY_PATHS = {
    "development_main": MOBILITY_ROOT / "main_reference.npz",
    "development_extra": MOBILITY_ROOT / "extra_reference.npz",
    "external_8b": MOBILITY_ROOT / "external_8b.npz",
    "legacy_main16x32": V4 / "results/cache16x32_v4/layerwise_mobility.npz",
}
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
REPRESENTATIONS = (
    "L2",
    "L3",
    "L4",
    "L5",
    "L12",
    "L13",
    "L14",
    "L15",
    "front_median",
    "back_median",
    "all_median",
)
PRIMARY_GRAPH_METRICS = (
    "conditional_query_d1",
    "partial_query_d1",
    "conditional_energy",
    "conditional_effective_rank",
    "flow_settling_log_ratio",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "analysis")
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


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
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def safe_auc(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if finite.sum() == 0 or len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(roc_auc_score(target[finite], score[finite]))


def safe_ap(target: np.ndarray, score: np.ndarray) -> float:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    if finite.sum() == 0 or len(np.unique(target[finite])) < 2:
        return float("nan")
    return float(average_precision_score(target[finite], score[finite]))


def bootstrap_auc(
    target: np.ndarray, score: np.ndarray, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    target = np.asarray(target, dtype=int)
    score = np.asarray(score, dtype=float)
    finite = np.isfinite(score)
    target, score = target[finite], score[finite]
    positive = np.flatnonzero(target == 1)
    negative = np.flatnonzero(target == 0)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        take = np.r_[
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ]
        values[draw] = roc_auc_score(target[take], score[take])
    return tuple(np.quantile(values, (0.025, 0.975)).tolist())


def profile_task_map(
    profiles: list[dict[str, np.ndarray]],
) -> dict[str, tuple[dict[str, np.ndarray], np.ndarray]]:
    output: dict[str, tuple[dict[str, np.ndarray], np.ndarray]] = {}
    for profile in profiles:
        names = profile["task_names"].astype(str)
        task_index = profile["task_index"].astype(int)
        for position, task in enumerate(names):
            if task in output:
                raise ValueError(f"duplicate reference task: {task}")
            output[task] = (profile, np.flatnonzero(task_index == position))
    return output


def task_rows(profile: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    names = profile["task_names"].astype(str)
    task_index = profile["task_index"].astype(int)
    layers = profile["layer_names"].astype(str)
    metrics = profile["metric_names"].astype(str)
    values = profile["metrics"].astype(np.float32, copy=False)
    valid = profile["valid"].astype(bool)
    rows: list[dict[str, Any]] = []
    for task_position, task in enumerate(names):
        take = np.flatnonzero(task_index == task_position)
        for metric_position, metric in enumerate(metrics):
            for layer_position, layer in enumerate(layers):
                sample = values[take, :, layer_position, metric_position]
                sample = sample[valid[take] & np.isfinite(sample)]
                rows.append(
                    {
                        "cohort": str(profile["cohort"]),
                        "task": task,
                        "layer": layer,
                        "metric": metric,
                        "queries": len(sample),
                        "median": float(np.median(sample)),
                        "q25": float(np.quantile(sample, 0.25)),
                        "q75": float(np.quantile(sample, 0.75)),
                    }
                )
    return rows


def structure_tables(
    reference_profiles: list[dict[str, np.ndarray]],
    external: dict[str, np.ndarray],
    legacy: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    task_table = pd.DataFrame(
        [
            row
            for profile in reference_profiles + [external, legacy]
            for row in task_rows(profile)
        ]
    )
    task_table["cohort_group"] = np.select(
        (
            task_table["cohort"].str.startswith("development"),
            task_table["cohort"].eq("external_8b"),
        ),
        ("reference_50x8", "external_8b"),
        default="legacy_main16x32",
    )
    layer_table = (
        task_table.groupby(["cohort_group", "layer", "metric"], sort=True)
        .agg(
            tasks=("task", "nunique"),
            queries=("queries", "sum"),
            task_median=("median", "median"),
            task_q25=("median", lambda value: float(np.quantile(value, 0.25))),
            task_q75=("median", lambda value: float(np.quantile(value, 0.75))),
        )
        .reset_index()
    )

    maps = {
        "reference_50x8": profile_task_map(reference_profiles),
        "external_8b": profile_task_map([external]),
        "legacy_main16x32": profile_task_map([legacy]),
    }
    edge_rows: list[dict[str, Any]] = []
    comparisons = (
        ("reference_50x8", "external_8b"),
        ("reference_50x8", "legacy_main16x32"),
        ("external_8b", "legacy_main16x32"),
    )
    for left_name, right_name in comparisons:
        left_map, right_map = maps[left_name], maps[right_name]
        common = sorted(set(left_map) & set(right_map))
        for edge_name in ("conditional_edge_mean", "partial_edge_mean"):
            for layer_position, layer in enumerate(external["layer_names"].astype(str)):
                left_flat = []
                right_flat = []
                within_task = []
                for task in common:
                    left_profile, left_take = left_map[task]
                    right_profile, right_take = right_map[task]
                    left_position = int(left_profile["task_index"][left_take[0]])
                    right_position = int(right_profile["task_index"][right_take[0]])
                    left = left_profile[edge_name][left_position, layer_position].astype(float)
                    right = right_profile[edge_name][right_position, layer_position].astype(float)
                    left_flat.append(left)
                    right_flat.append(right)
                    within_task.append(float(spearmanr(left, right).statistic))
                rho = float(
                    spearmanr(np.concatenate(left_flat), np.concatenate(right_flat)).statistic
                )
                edge_rows.append(
                    {
                        "comparison": f"{left_name}_vs_{right_name}",
                        "edge_view": edge_name,
                        "layer": layer,
                        "common_tasks": len(common),
                        "pooled_task_edge_spearman": rho,
                        "median_within_task_edge_spearman": float(np.nanmedian(within_task)),
                        "q25_within_task_edge_spearman": float(np.nanquantile(within_task, 0.25)),
                        "q75_within_task_edge_spearman": float(np.nanquantile(within_task, 0.75)),
                    }
                )
    return task_table, layer_table, pd.DataFrame(edge_rows)


def front_back_contrasts(
    task_table: pd.DataFrame, draws: int, rng: np.random.Generator
) -> pd.DataFrame:
    definitions = {
        "flow_path": "front_over_back",
        "conditional_energy": "front_over_back",
        "conditional_query_d1": "front_over_back",
        "partial_edge_std": "back_over_front",
        "conditional_effective_rank": "back_over_front",
    }
    rows = []
    for (cohort, metric), block in task_table[
        task_table["metric"].isin(definitions)
    ].groupby(["cohort_group", "metric"], sort=True):
        pivot = block.pivot(index="task", columns="layer", values="median")
        front = pivot[["L2", "L3", "L4", "L5"]].mean(axis=1).to_numpy()
        back = pivot[["L12", "L13", "L14", "L15"]].mean(axis=1).to_numpy()
        if definitions[metric] == "front_over_back":
            ratio = (front + 1e-12) / (back + 1e-12)
        else:
            ratio = (back + 1e-12) / (front + 1e-12)
        sampled = ratio[rng.integers(0, len(ratio), size=(draws, len(ratio)))]
        bootstrap = np.median(sampled, axis=1)
        rows.append(
            {
                "cohort_group": cohort,
                "metric": metric,
                "orientation": definitions[metric],
                "tasks": len(ratio),
                "median_expected_ratio": float(np.median(ratio)),
                "q25_expected_ratio": float(np.quantile(ratio, 0.25)),
                "q75_expected_ratio": float(np.quantile(ratio, 0.75)),
                "bootstrap_median_95_low": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_median_95_high": float(np.quantile(bootstrap, 0.975)),
                "tasks_with_expected_direction": int((ratio > 1.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def mobility_representations(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    available = dev.representations(cache)
    return {name: available[name][0] for name in REPRESENTATIONS}


def oriented_score(values: np.ndarray, direction: str, width: int) -> np.ndarray:
    smoothed = dev.trailing_mean(values, width)
    return -smoothed if direction == "low" else smoothed


def calibrated_development_alarm(
    cache: dict[str, np.ndarray],
    values: np.ndarray,
    direction: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> np.ndarray:
    oriented = oriented_score(values, direction, width)
    thresholds = dev.crossfit_thresholds(
        dev.row_max(oriented),
        cache["task_index"].astype(int),
        cache["init_state_id"].astype(int),
    )[:, dev.QUANTILES.index(quantile)]
    persistent = dev.persistent_score(oriented, confirmations)
    trigger = np.isfinite(persistent) & (persistent > thresholds[:, None]) & cache["valid"]
    return dev.first_query(trigger)


def calibrated_external_alarm(
    external: dict[str, np.ndarray],
    external_values: np.ndarray,
    reference_map: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    reference_representations: dict[int, dict[str, np.ndarray]],
    representation: str,
    direction: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> np.ndarray:
    persistent = dev.persistent_score(
        oriented_score(external_values, direction, width), confirmations
    )
    first = np.full(len(external_values), -1, dtype=np.int16)
    task_names = external["task_names"].astype(str)
    task_index = external["task_index"].astype(int)
    for task_position, task in enumerate(task_names):
        take = np.flatnonzero(task_index == task_position)
        reference, reference_take = reference_map[task]
        reference_values = reference_representations[id(reference)][representation]
        reference_peak = dev.row_max(
            oriented_score(reference_values[reference_take], direction, width)
        )
        threshold = dev.quantile_higher(reference_peak, quantile)
        trigger = (
            np.isfinite(persistent[take])
            & (persistent[take] > threshold)
            & external["valid"][take]
        )
        first[take] = dev.first_query(trigger)
    return first


def aligned_labels(cache: dict[str, np.ndarray], path: Path) -> pd.DataFrame:
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    labels = pd.read_csv(path).reset_index(drop=True)
    if len(labels) != len(task):
        raise ValueError(f"label length mismatch: {path}")
    if not np.array_equal(labels["task"].astype(str), task):
        raise ValueError(f"label task mismatch: {path}")
    if not np.array_equal(labels["episode"].to_numpy(dtype=int), cache["episode"].astype(int)):
        raise ValueError(f"label episode mismatch: {path}")
    return labels


def detector_metric(
    cohort: str, detector: str, first: np.ndarray, labels: pd.DataFrame
) -> dict[str, Any]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    persistent = labels["failure"].to_numpy(dtype=bool)
    timely = ~risk
    lead = labels["episode_length"].to_numpy(dtype=int) - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    return {
        "cohort": cohort,
        "detector": detector,
        "episodes": len(labels),
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "late_n": int(late.sum()),
        "persistent_n": int(persistent.sum()),
        "tp": tp,
        "fp": fp,
        "risk_recall": tp / risk.sum(),
        "timely_fpr": fp / timely.sum(),
        "precision": tp / max(tp + fp, 1),
        "late_recall": (
            int((alarm & late).sum()) / late.sum() if late.sum() else float("nan")
        ),
        "persistent_recall": (
            int((alarm & persistent).sum()) / persistent.sum()
            if persistent.sum()
            else float("nan")
        ),
        "early4_recall": int((alarm & risk & (lead >= 4)).sum()) / risk.sum(),
        "median_lead": float(np.median(lead[alarm & risk])) if tp else float("nan"),
    }


def stage1_tables(
    main: dict[str, np.ndarray],
    extra: dict[str, np.ndarray],
    external: dict[str, np.ndarray],
    legacy: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    main_repr = mobility_representations(main)
    extra_repr = mobility_representations(extra)
    external_repr = mobility_representations(external)
    legacy_repr = mobility_representations(legacy)
    reference_map = profile_task_map([main, extra])
    reference_representations = {id(main): main_repr, id(extra): extra_repr}
    firsts: dict[str, dict[str, np.ndarray]] = {
        "development_main": {},
        "external_8b": {},
        "legacy_main16x32": {},
    }
    for representation in REPRESENTATIONS:
        for head, direction, width, confirmations, quantile in (
            ("lock", "low", 4, 4, 0.75),
            ("instability", "high", 4, 8, 0.80),
        ):
            name = f"{head}|{representation}"
            firsts["development_main"][name] = calibrated_development_alarm(
                main,
                main_repr[representation],
                direction,
                width,
                confirmations,
                quantile,
            )
            firsts["external_8b"][name] = calibrated_external_alarm(
                external,
                external_repr[representation],
                reference_map,
                reference_representations,
                representation,
                direction,
                width,
                confirmations,
                quantile,
            )
            firsts["legacy_main16x32"][name] = calibrated_external_alarm(
                legacy,
                legacy_repr[representation],
                reference_map,
                reference_representations,
                representation,
                direction,
                width,
                confirmations,
                quantile,
            )
        for cohort in firsts:
            lock = firsts[cohort][f"lock|{representation}"]
            instability = firsts[cohort][f"instability|{representation}"]
            firsts[cohort][f"dual_same|{representation}"] = np.where(
                lock < 0,
                instability,
                np.where(instability < 0, lock, np.minimum(lock, instability)),
            ).astype(np.int16)

    for cohort in firsts:
        lock = firsts[cohort]["lock|all_median"]
        instability = firsts[cohort]["instability|L5"]
        firsts[cohort]["v4_exact|all_lock+L5_instability"] = np.where(
            lock < 0,
            instability,
            np.where(instability < 0, lock, np.minimum(lock, instability)),
        ).astype(np.int16)

    sealed = load_npz(SEALED)
    checks = (
        ("development_main", "lock|all_median", "main_lock"),
        ("development_main", "instability|L5", "main_instability"),
        ("development_main", "v4_exact|all_lock+L5_instability", "main_dual"),
        ("external_8b", "lock|all_median", "external_lock"),
        ("external_8b", "instability|L5", "external_instability"),
        ("external_8b", "v4_exact|all_lock+L5_instability", "external_dual"),
    )
    for cohort, detector, key in checks:
        if not np.array_equal(firsts[cohort][detector], sealed[key].astype(np.int16)):
            mismatch = int(np.sum(firsts[cohort][detector] != sealed[key]))
            raise AssertionError(f"sealed v4 mismatch: {cohort}/{detector}: {mismatch}")

    legacy_sealed = pd.read_csv(V4 / "results/cache16x32_v4/episode_alarms.csv")
    legacy_checks = (
        ("lock|all_median", "first_lock_layer_median_q75_k4_query"),
        ("instability|L5", "first_instability_L5_q80_k8_query"),
        (
            "v4_exact|all_lock+L5_instability",
            "first_dual_regime_or_query",
        ),
    )
    for detector, column in legacy_checks:
        expected = legacy_sealed[column].to_numpy(dtype=np.int16)
        observed = firsts["legacy_main16x32"][detector]
        if not np.array_equal(observed, expected):
            mismatch = int(np.sum(observed != expected))
            raise AssertionError(f"legacy sealed v4 mismatch: {detector}: {mismatch}")

    labels = {
        "development_main": aligned_labels(main, LABEL_PATHS["development_main"]),
        "external_8b": aligned_labels(external, LABEL_PATHS["external_8b"]),
        "legacy_main16x32": legacy_sealed.assign(
            original_failure=legacy_sealed["failure"].astype(bool),
            late_success_plus10_queries=False,
            episode_length=legacy_sealed["length"],
        ),
    }
    rows = [
        detector_metric(cohort, detector, first, labels[cohort])
        for cohort in firsts
        for detector, first in firsts[cohort].items()
    ]
    return pd.DataFrame(rows), firsts


def empirical_rank_columns(reference: np.ndarray, current: np.ndarray) -> np.ndarray:
    output = np.full(current.shape, np.nan, dtype=np.float32)
    for layer in range(current.shape[1]):
        for metric in range(current.shape[2]):
            ref = np.sort(reference[:, layer, metric][np.isfinite(reference[:, layer, metric])])
            values = current[:, layer, metric]
            finite = np.isfinite(values)
            if not len(ref):
                continue
            lower = np.searchsorted(ref, values[finite], side="left")
            upper = np.searchsorted(ref, values[finite], side="right")
            output[finite, layer, metric] = (lower + upper) / (2.0 * len(ref))
    return output


def rank_profile(
    evaluation: dict[str, np.ndarray],
    reference_map: dict[str, tuple[dict[str, np.ndarray], np.ndarray]],
    metric_names: tuple[str, ...],
) -> np.ndarray:
    all_names = evaluation["metric_names"].astype(str).tolist()
    positions = [all_names.index(name) for name in metric_names]
    output = np.full(
        (*evaluation["valid"].shape, 8, len(metric_names)), np.nan, dtype=np.float32
    )
    evaluation_tasks = evaluation["task_names"].astype(str)
    evaluation_index = evaluation["task_index"].astype(int)
    for task_position, task in enumerate(evaluation_tasks):
        take = np.flatnonzero(evaluation_index == task_position)
        reference, reference_take = reference_map[task]
        reference_names = reference["metric_names"].astype(str).tolist()
        reference_positions = [reference_names.index(name) for name in metric_names]
        max_query = output.shape[1]
        for query in range(max_query):
            current_valid = evaluation["valid"][take, query]
            reference_valid = reference["valid"][reference_take, query]
            if not current_valid.any() or not reference_valid.any():
                continue
            current_rows = take[current_valid]
            current = evaluation["metrics"][current_rows, query][:, :, positions]
            reference_values = reference["metrics"][reference_take[reference_valid], query][
                :, :, reference_positions
            ]
            output[current_rows, query] = empirical_rank_columns(reference_values, current)
    return output


def window_value(ranks: np.ndarray, row: int, query: int, width: int = 4) -> np.ndarray:
    start = max(0, query - width + 1)
    window = ranks[row, start : query + 1]
    count = np.isfinite(window).sum(axis=0)
    total = np.nansum(window, axis=0)
    output = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=output, where=count > 0)
    return output


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else float("nan")


def stage2_rows(
    cohort: str,
    profile: dict[str, np.ndarray],
    ranks: np.ndarray,
    first: np.ndarray,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_names = profile["layer_names"].astype(str)
    task = profile["task_names"].astype(str)[profile["task_index"].astype(int)]
    risk = labels["original_failure"].to_numpy(dtype=bool)
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    records: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for row in np.flatnonzero(risk & (first >= 0)):
        alarm = int(first[row])
        terminal = int(profile["length"][row]) - 1
        alarm_value = window_value(ranks, row, alarm)
        terminal_value = window_value(ranks, row, terminal)
        release = terminal_value - alarm_value
        base = {
            "cohort": cohort,
            "row": int(row),
            "task": task[row],
            "episode": int(profile["episode"][row]),
            "outcome": "late" if late[row] else "persistent",
            "alarm_query": alarm,
            "terminal_query": terminal,
        }
        record = dict(base)
        for metric_position, metric in enumerate(PRIMARY_GRAPH_METRICS):
            for layer_position, layer in enumerate(layer_names):
                record[f"{metric}|{layer}"] = release[layer_position, metric_position]
            front = finite_mean(release[:4, metric_position])
            back = finite_mean(release[4:, metric_position])
            record[f"{metric}|front_mean"] = front
            record[f"{metric}|back_mean"] = back
            record[f"{metric}|all_mean"] = finite_mean(release[:, metric_position])
            record[f"{metric}|back_minus_front"] = back - front
        records.append(record)

        for offset_label, query in (
            ("+1", alarm + 1),
            ("+2", alarm + 2),
            ("+4", alarm + 4),
            ("+8", alarm + 8),
            ("terminal", terminal),
        ):
            if query > terminal:
                continue
            current = window_value(ranks, row, query)
            delta = current - alarm_value
            for layer_position, layer in enumerate(layer_names):
                curves.append(
                    {
                        **base,
                        "offset": offset_label,
                        "layer": layer,
                        "score": delta[layer_position, 0],
                    }
                )
            curves.extend(
                {
                    **base,
                    "offset": offset_label,
                    "layer": group,
                    "score": value,
                }
                for group, value in (
                    ("front_mean", finite_mean(delta[:4, 0])),
                    ("back_mean", finite_mean(delta[4:, 0])),
                    ("all_mean", finite_mean(delta[:, 0])),
                    (
                        "back_minus_front",
                        finite_mean(delta[4:, 0]) - finite_mean(delta[:4, 0]),
                    ),
                )
            )
    return pd.DataFrame(records), pd.DataFrame(curves)


def stage2_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    metadata = {
        "cohort",
        "row",
        "task",
        "episode",
        "outcome",
        "alarm_query",
        "terminal_query",
    }
    rows = []
    target = scores["outcome"].eq("late").astype(int).to_numpy()
    for feature in sorted(set(scores.columns) - metadata):
        rows.append(
            {
                "cohort": scores["cohort"].iloc[0],
                "feature": feature,
                "n": len(scores),
                "finite_n": int(np.isfinite(scores[feature]).sum()),
                "late_n": int(target.sum()),
                "prevalence": float(target.mean()),
                "auc_high_means_late": safe_auc(target, scores[feature]),
                "average_precision": safe_ap(target, scores[feature]),
            }
        )
    return pd.DataFrame(rows)


def curve_metrics(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, offset, layer), block in curves.groupby(
        ["cohort", "offset", "layer"], sort=True
    ):
        target = block["outcome"].eq("late").astype(int).to_numpy()
        rows.append(
            {
                "cohort": cohort,
                "offset": offset,
                "layer": layer,
                "n": len(block),
                "late_n": int(target.sum()),
                "auc_high_means_late": safe_auc(target, block["score"]),
                "average_precision": safe_ap(target, block["score"]),
            }
        )
    return pd.DataFrame(rows)


def fusion_audit(scores: pd.DataFrame) -> pd.DataFrame:
    if not OLD_TRANSITION.is_file():
        return pd.DataFrame()
    old = pd.read_csv(OLD_TRANSITION)
    keys = ["cohort", "row", "task", "outcome"]
    merged = old.merge(scores, on=keys, validate="one_to_one")
    merged["layer_front_absolute_restart"] = merged[
        "conditional_query_d1|front_mean"
    ]
    merged["layer_back_topology_restart"] = merged["partial_query_d1|back_mean"]
    merged["layer_back_rank_restart"] = merged[
        "conditional_effective_rank|back_mean"
    ]
    merged["layer_structured_equal3"] = merged[
        [
            "layer_front_absolute_restart",
            "layer_back_topology_restart",
            "layer_back_rank_restart",
        ]
    ].mean(axis=1)
    merged["coupled_plus_back_topology"] = merged[
        ["graph_conditional_release", "layer_back_topology_restart"]
    ].mean(axis=1)
    merged["coupled_plus_back_rank"] = merged[
        ["graph_conditional_release", "layer_back_rank_restart"]
    ].mean(axis=1)
    features = (
        "graph_conditional_release",
        "layer_front_absolute_restart",
        "layer_back_topology_restart",
        "layer_back_rank_restart",
        "layer_structured_equal3",
        "coupled_plus_back_topology",
        "coupled_plus_back_rank",
    )
    rows = []
    for cohort, block in merged.groupby("cohort", sort=True):
        target = block["outcome"].eq("late").astype(int).to_numpy()
        for feature in features:
            rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "status": "exploratory_posthoc_layer_fusion",
                    "n": len(block),
                    "late_n": int(target.sum()),
                    "auc": safe_auc(target, block[feature]),
                    "average_precision": safe_ap(target, block[feature]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph = {
        cohort: load_npz(PROFILE_ROOT / f"{cohort}.npz")
        for cohort in (
            "development_main",
            "development_extra",
            "external_8b",
            "legacy_main16x32",
        )
    }
    mobility = {cohort: load_npz(path) for cohort, path in MOBILITY_PATHS.items()}

    task_structure, layer_structure, edge_replication = structure_tables(
        [graph["development_main"], graph["development_extra"]],
        graph["external_8b"],
        graph["legacy_main16x32"],
    )
    rng = np.random.default_rng(args.seed)
    contrasts = front_back_contrasts(task_structure, args.bootstrap, rng)
    task_structure.to_csv(args.output / "task_layer_structure.csv", index=False)
    layer_structure.to_csv(args.output / "layer_structure.csv", index=False)
    edge_replication.to_csv(args.output / "edge_replication.csv", index=False)
    contrasts.to_csv(args.output / "front_back_contrasts.csv", index=False)

    stage1, firsts = stage1_tables(
        mobility["development_main"],
        mobility["development_extra"],
        mobility["external_8b"],
        mobility["legacy_main16x32"],
    )
    stage1.to_csv(args.output / "stage1_layer_detectors.csv", index=False)

    reference_map = profile_task_map(
        [graph["development_main"], graph["development_extra"]]
    )
    development_ranks = rank_profile(
        graph["development_main"], reference_map, PRIMARY_GRAPH_METRICS
    )
    external_ranks = rank_profile(graph["external_8b"], reference_map, PRIMARY_GRAPH_METRICS)
    labels = {
        "development_main": aligned_labels(
            mobility["development_main"], LABEL_PATHS["development_main"]
        ),
        "external_8b": aligned_labels(
            mobility["external_8b"], LABEL_PATHS["external_8b"]
        ),
    }
    score_parts = []
    curve_parts = []
    for cohort, ranks in (
        ("development_main", development_ranks),
        ("external_8b", external_ranks),
    ):
        scores, curves = stage2_rows(
            cohort,
            graph[cohort],
            ranks,
            firsts[cohort]["v4_exact|all_lock+L5_instability"],
            labels[cohort],
        )
        score_parts.append(scores)
        curve_parts.append(curves)
    scores = pd.concat(score_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)
    metrics = pd.concat(
        [stage2_metrics(block) for _, block in scores.groupby("cohort", sort=True)],
        ignore_index=True,
    )
    curve_summary = curve_metrics(curves)
    fusion = fusion_audit(scores)
    scores.to_csv(args.output / "stage2_scores.csv", index=False)
    metrics.to_csv(args.output / "stage2_layer_metrics.csv", index=False)
    curve_summary.to_csv(args.output / "stage2_response_curve.csv", index=False)
    if len(fusion):
        fusion.to_csv(args.output / "fusion_audit.csv", index=False)

    primary_feature = "conditional_query_d1|all_mean"
    external_scores = scores[scores["cohort"] == "external_8b"]
    external_target = external_scores["outcome"].eq("late").astype(int).to_numpy()
    external_primary = external_scores[primary_feature].to_numpy(dtype=float)
    v4_external = stage1[
        (stage1["cohort"] == "external_8b")
        & (stage1["detector"] == "v4_exact|all_lock+L5_instability")
    ].iloc[0]
    summary = {
        "schema": "himoe.hb_front_back_analysis.v1",
        "trained_model": False,
        "q_minus_2_used": False,
        "data": {
            "plus10_labeled_episodes": int(
                len(mobility["development_main"]["episode"])
                + len(mobility["external_8b"]["episode"])
            ),
            "stage1_labeled_episodes": int(
                len(mobility["development_main"]["episode"])
                + len(mobility["external_8b"]["episode"])
                + len(mobility["legacy_main16x32"]["episode"])
            ),
            "reference_only_episodes": int(
                len(mobility["development_extra"]["episode"])
            ),
            "legacy_structure_only_episodes": int(
                len(graph["legacy_main16x32"]["episode"])
            ),
            "cohort_task_pairs": int(
                len(mobility["development_main"]["task_names"])
                + len(mobility["external_8b"]["task_names"])
            ),
            "unique_task_names": int(
                len(
                    set(mobility["development_main"]["task_names"].astype(str))
                    | set(mobility["external_8b"]["task_names"].astype(str))
                )
            ),
            "valid_queries": int(
                mobility["development_main"]["valid"].sum()
                + mobility["development_extra"]["valid"].sum()
                + mobility["external_8b"]["valid"].sum()
                + graph["legacy_main16x32"]["valid"].sum()
            ),
        },
        "v4_exact_external": v4_external.to_dict(),
        "stage2_primary": {
            "feature": primary_feature,
            "definition": "task/query-ranked state-conditioned graph response, terminal W4 minus alarm W4, averaged over all layers",
            "n": len(external_scores),
            "finite_n": int(np.isfinite(external_primary).sum()),
            "late_n": int(external_target.sum()),
            "auc": safe_auc(external_target, external_primary),
            "auc_ci95": bootstrap_auc(
                external_target, external_primary, args.bootstrap, rng
            ),
            "average_precision": safe_ap(external_target, external_primary),
            "prevalence": float(external_target.mean()),
        },
        "artifacts": {
            "layer_structure": "layer_structure.csv",
            "edge_replication": "edge_replication.csv",
            "front_back_contrasts": "front_back_contrasts.csv",
            "stage1": "stage1_layer_detectors.csv",
            "stage2": "stage2_layer_metrics.csv",
            "curve": "stage2_response_curve.csv",
            "fusion_audit": "fusion_audit.csv",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
