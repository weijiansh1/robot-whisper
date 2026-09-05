#!/usr/bin/env python3
"""Development-only sweep of causal train-free layerwise MoE alarms.

This script intentionally reads no external-8B outcomes.  Each alarm threshold is
calibrated without outcome labels by leaving all eight seeds of one initial state
out and using the other 392 trajectories from the same task.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
BUNDLE = HERE.parent
CANONICAL = WORKSPACE / "double-selete/trainfree"
DEFAULT_LAYER_CACHE = BUNDLE / "results/layerwise_mobility/main_reference.npz"
DEFAULT_LABELS = (
    CANONICAL / "results/timeout_extension_plus10/development_main_clean_labels.csv"
)
DEFAULT_OUTPUT = BUNDLE / "results/development_sweep"

WIDTHS = (1, 2, 4)
CONFIRMATIONS = (1, 2, 4, 6, 8)
QUANTILES = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.925,
    0.95,
    0.96,
    0.975,
    0.98,
    0.99,
    0.995,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-cache", type=Path, default=DEFAULT_LAYER_CACHE)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def trailing_mean(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.pad(np.cumsum(filled, axis=1), ((0, 0), (1, 0)))
    count = np.pad(np.cumsum(finite, axis=1), ((0, 0), (1, 0)))
    output[:, width - 1 :] = (csum[:, width:] - csum[:, :-width]) / width
    complete = count[:, width:] - count[:, :-width] == width
    output[:, width - 1 :][~complete] = np.nan
    return output


def persistent_score(oriented: np.ndarray, count: int) -> np.ndarray:
    """Return the minimum oriented anomaly score in each causal window."""
    oriented = np.asarray(oriented, dtype=np.float32)
    output = np.full_like(oriented, np.nan)
    for query in range(count - 1, oriented.shape[1]):
        window = oriented[:, query - count + 1 : query + 1]
        complete = np.isfinite(window).all(axis=1)
        output[complete, query] = window[complete].min(axis=1)
    return output


def row_max(values: np.ndarray) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float32)
    good = np.isfinite(values).any(axis=1)
    output[good] = np.nanmax(values[good], axis=1)
    return output


def quantile_higher(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 32:
        return float("nan")
    return float(np.quantile(finite, quantile, method="higher"))


def first_query(trigger: np.ndarray) -> np.ndarray:
    fired = trigger.any(axis=1)
    output = np.argmax(trigger, axis=1).astype(np.int16)
    output[~fired] = -1
    return output


def build_index(cache: dict[str, np.ndarray], labels_path: Path) -> pd.DataFrame:
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
            "length": cache["length"].astype(int),
        }
    )
    labels = pd.read_csv(labels_path)[
        [
            "task",
            "episode",
            "original_failure",
            "failure",
            "late_success_plus10_queries",
        ]
    ]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError("development outcomes do not align with layer cache")
    return merged.reset_index(drop=True)


def aggregate(values: np.ndarray, indices: slice, operation: str) -> np.ndarray:
    selected = values[:, :, indices]
    if operation == "mean":
        return selected.mean(axis=2)
    if operation == "min":
        return selected.min(axis=2)
    if operation == "max":
        return selected.max(axis=2)
    if operation == "median":
        return np.median(selected, axis=2)
    if operation == "std":
        return selected.std(axis=2)
    raise KeyError(operation)


def representations(cache: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, str]]:
    mobility = np.asarray(cache["mobility"], dtype=np.float32)
    valid = np.asarray(cache["valid"], dtype=bool)
    filled = np.where(valid[:, :, None], mobility, 0.0)
    layer_names = cache["layer_names"].astype(str)
    output: dict[str, tuple[np.ndarray, str]] = {}

    for position, name in enumerate(layer_names):
        output[str(name)] = (filled[:, :, position], "mobility")
    for group, indices in (
        ("front", slice(0, 4)),
        ("back", slice(4, 8)),
        ("all", slice(0, 8)),
    ):
        for operation in ("mean", "min", "max", "median", "std"):
            output[f"{group}_{operation}"] = (
                aggregate(filled, indices, operation),
                "dispersion" if operation == "std" else "mobility",
            )

    output["front_back_mean_gap"] = (
        np.abs(
            aggregate(filled, slice(0, 4), "mean")
            - aggregate(filled, slice(4, 8), "mean")
        ),
        "dispersion",
    )
    for name, (values, family) in list(output.items()):
        values = values.astype(np.float32, copy=True)
        values[~valid] = np.nan
        output[name] = (values, family)
        if family == "mobility":
            delta = np.full_like(values, np.nan)
            delta[:, 1:] = np.abs(values[:, 1:] - values[:, :-1])
            delta[~valid] = np.nan
            output[f"delta_{name}"] = (delta, "derivative")
    return output


def crossfit_thresholds(
    calibration_peak: np.ndarray,
    task_index: np.ndarray,
    init_state: np.ndarray,
) -> np.ndarray:
    output = np.full((len(calibration_peak), len(QUANTILES)), np.nan, np.float32)
    for task_position in np.unique(task_index):
        take = np.flatnonzero(task_index == task_position)
        task_init = init_state[take]
        if len(take) != 400:
            raise ValueError(f"task {task_position} has {len(take)} episodes")
        for held_out in np.unique(task_init):
            test = task_init == held_out
            reference = ~test
            if int(reference.sum()) != 392 or int(test.sum()) != 8:
                raise ValueError("expected 50 initial states with eight seeds each")
            finite = calibration_peak[take[reference]]
            finite = np.sort(finite[np.isfinite(finite)])
            if len(finite) < 32:
                continue
            positions = [
                int(np.ceil(quantile * (len(finite) - 1))) for quantile in QUANTILES
            ]
            output[take[test]] = finite[positions]
    if not np.isfinite(output).all():
        raise ValueError("non-finite cross-fitted threshold")
    return output


def metric_row(
    representation: str,
    family: str,
    direction: str,
    width: int,
    confirmations: int,
    quantile: float,
    first: np.ndarray,
    labels: pd.DataFrame,
    task_index: np.ndarray,
) -> dict[str, Any]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(dtype=bool)
    timely = ~risk
    late = labels["late_success_plus10_queries"].to_numpy(dtype=bool)
    persistent = labels["failure"].to_numpy(dtype=bool)
    length = labels["length"].to_numpy(dtype=int)
    lead = length - 1 - first
    tp = int((alarm & risk).sum())
    fp = int((alarm & timely).sum())
    n_tasks = int(task_index.max()) + 1
    risk_by_task = np.bincount(task_index[risk], minlength=n_tasks)
    tp_by_task = np.bincount(task_index[alarm & risk], minlength=n_tasks)
    risk_tasks = risk_by_task > 0
    task_recall = tp_by_task[risk_tasks] / risk_by_task[risk_tasks]
    return {
        "representation": representation,
        "family": family,
        "direction": direction,
        "width": width,
        "confirmations": confirmations,
        "quantile": quantile,
        "risk_n": int(risk.sum()),
        "timely_n": int(timely.sum()),
        "tp": tp,
        "fp": fp,
        "recall": tp / risk.sum(),
        "timely_fpr": fp / timely.sum(),
        "precision": tp / max(tp + fp, 1),
        "late_recall": int((alarm & late).sum()) / late.sum(),
        "persistent_recall": int((alarm & persistent).sum()) / persistent.sum(),
        "risk_task_macro_recall": float(np.mean(task_recall)),
        "risk_tasks_detected": int((tp_by_task[risk_tasks] > 0).sum()),
        "risk_tasks": int(risk_tasks.sum()),
        "detected_risk_lead_median": (
            float(np.median(lead[alarm & risk])) if tp else float("nan")
        ),
        "early4_recall": float((alarm & risk & (lead >= 4)).sum() / risk.sum()),
    }


def pareto_mask(frame: pd.DataFrame) -> np.ndarray:
    precision = frame["precision"].to_numpy()
    recall = frame["recall"].to_numpy()
    dominated = np.zeros(len(frame), dtype=bool)
    for position in range(len(frame)):
        dominated[position] = bool(
            np.any(
                (precision >= precision[position])
                & (recall >= recall[position])
                & ((precision > precision[position]) | (recall > recall[position]))
            )
        )
    return ~dominated


def main() -> None:
    args = parse_args()
    cache = load_npz(args.layer_cache)
    labels = build_index(cache, args.labels)
    valid = np.asarray(cache["valid"], dtype=bool)
    task_index = np.asarray(cache["task_index"], dtype=int)
    init_state = np.asarray(cache["init_state_id"], dtype=int)
    rows: list[dict[str, Any]] = []

    all_representations = representations(cache)
    for position, (name, (values, family)) in enumerate(all_representations.items()):
        directions = (
            ("high",) if family in {"dispersion", "derivative"} else ("low", "high")
        )
        for width in WIDTHS:
            smoothed = trailing_mean(values, width)
            for direction in directions:
                oriented = -smoothed if direction == "low" else smoothed
                oriented[~valid] = np.nan
                thresholds = crossfit_thresholds(
                    row_max(oriented), task_index, init_state
                )
                for confirmations in CONFIRMATIONS:
                    score = persistent_score(oriented, confirmations)
                    trigger = (
                        np.isfinite(score[:, :, None])
                        & (score[:, :, None] > thresholds[:, None, :])
                        & valid[:, :, None]
                    )
                    for quantile_position, quantile in enumerate(QUANTILES):
                        first = first_query(trigger[:, :, quantile_position])
                        rows.append(
                            metric_row(
                                name,
                                family,
                                direction,
                                width,
                                confirmations,
                                quantile,
                                first,
                                labels,
                                task_index,
                            )
                        )
        print(
            f"[development {position + 1}/{len(all_representations)}] {name}",
            flush=True,
        )

    metrics = pd.DataFrame(rows)
    metrics["pareto"] = pareto_mask(metrics)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output / "candidate_metrics.csv", index=False)

    baseline = metrics[
        (metrics["representation"] == "back_mean")
        & (metrics["direction"] == "low")
        & (metrics["width"] == 4)
        & (metrics["confirmations"] == 2)
        & np.isclose(metrics["quantile"], 0.95)
    ].iloc[0]
    if int(baseline["tp"]) != 234 or int(baseline["fp"]) != 88:
        raise AssertionError(
            "back-mean baseline did not reproduce the sealed detector: "
            f"TP={baseline['tp']}, FP={baseline['fp']}"
        )

    high_both = metrics[(metrics["recall"] >= 0.60) & (metrics["precision"] >= 0.75)]
    best_precision = metrics[metrics["recall"] >= 0.50].nlargest(
        20, ["precision", "recall"]
    )
    best_recall = metrics[metrics["precision"] >= 0.75].nlargest(
        20, ["recall", "precision"]
    )
    pareto = metrics[metrics["pareto"]].sort_values("recall")
    best_precision.to_csv(args.output / "best_precision_at_recall50.csv", index=False)
    best_recall.to_csv(args.output / "best_recall_at_precision75.csv", index=False)
    pareto.to_csv(args.output / "pareto.csv", index=False)
    summary = {
        "schema": "himoe.layerwise_alarm_development.v1",
        "external_outcomes_read": False,
        "episodes": len(labels),
        "risk_n": int(labels["original_failure"].sum()),
        "candidate_rows": len(metrics),
        "baseline": baseline.to_dict(),
        "candidates_recall60_precision75": len(high_both),
        "best_precision_at_recall50": best_precision.iloc[0].to_dict(),
        "best_recall_at_precision75": (
            best_recall.iloc[0].to_dict() if len(best_recall) else None
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plain(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
