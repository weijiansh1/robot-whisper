#!/usr/bin/env python3
"""Select a spatial-only causal guard with a two-GPU development sweep."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
BASE = WORKSPACE / "moe-v4-0904"
BASE_EXPERIMENTS = BASE / "experiments"
sys.path.insert(0, str(BASE_EXPERIMENTS))

import evaluate_dual_regime_v4 as v4  # noqa: E402
import evaluate_layerwise_alarm_development as dev  # noqa: E402
import evaluate_layerwise_persistence_v3 as v3  # noqa: E402
import evaluate_long_guard_v5 as long_guard  # noqa: E402


PROTOCOL = BUNDLE / "method/ONLINE_SPATIAL_GUARD_V6_PROTOCOL.md"
DEFAULT_LAYER_ROOT = BASE / "results/layerwise_mobility"
DEFAULT_LABEL_ROOT = (
    WORKSPACE / "double-selete/trainfree/results/timeout_extension_plus10"
)
DEFAULT_V4_RESULT = BASE / "results/cache_new_v4"
DEFAULT_LONG_RESULT = BASE / "results/long_guard_v5"
DEFAULT_LEAD_RESULT = BASE / "results/lead_guard_v5"
DEFAULT_OUTPUT = BUNDLE / "results/spatial_guard_v6"

SPATIAL_SUITE = "libero_spatial"
# W8 is excluded: at least one short-rollout task has fewer than 32 finite
# leave-one-initial-state calibration peaks for every representation at W8.
WIDTHS = (1, 2, 3, 4, 6)
CONFIRMATIONS = (1, 2, 3, 4, 6, 8, 10, 12)
QUANTILES = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.725,
    0.75,
    0.775,
    0.80,
    0.825,
    0.85,
    0.875,
    0.90,
    0.925,
    0.95,
    0.96,
    0.975,
    0.98,
    0.99,
    0.995,
)
OVERALL_FPR_LIMIT = 0.0025
PRECISION_MIN = 0.90
RECALL_MIN = 0.75
SUITE_FPR_LIMIT = 0.01
TASK_FPR_LIMIT = 0.025
SHORTLIST_SIZE = 512
LAYER_NAMES = (
    "L2",
    "L3",
    "L4",
    "L5",
    "L12",
    "L13",
    "L14",
    "L15",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-root", type=Path, default=DEFAULT_LAYER_ROOT)
    parser.add_argument("--label-root", type=Path, default=DEFAULT_LABEL_ROOT)
    parser.add_argument("--v4-result", type=Path, default=DEFAULT_V4_RESULT)
    parser.add_argument("--long-result", type=Path, default=DEFAULT_LONG_RESULT)
    parser.add_argument("--lead-result", type=Path, default=DEFAULT_LEAD_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--devices",
        default="0,1",
        help="Logical CUDA indices after CUDA_VISIBLE_DEVICES is applied.",
    )
    return parser.parse_args()


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def representation_specs() -> list[tuple[str, str]]:
    mobility = list(LAYER_NAMES)
    dispersion: list[str] = []
    for group in ("front", "back", "all"):
        for operation in ("mean", "min", "max", "median"):
            mobility.append(f"{group}_{operation}")
        dispersion.append(f"{group}_std")
    dispersion.append("front_back_mean_gap")
    specs = [(name, direction) for name in mobility for direction in ("low", "high")]
    specs.extend((name, "high") for name in dispersion)
    specs.extend((f"delta_{name}", "high") for name in mobility)
    if len(specs) != 64:
        raise AssertionError(f"expected 64 oriented representations, got {len(specs)}")
    return specs


def _median(values: torch.Tensor) -> torch.Tensor:
    ordered = values.sort(dim=2).values
    count = ordered.shape[2]
    if count % 2:
        return ordered[:, :, count // 2]
    return (ordered[:, :, count // 2 - 1] + ordered[:, :, count // 2]) / 2


def gpu_representation(
    mobility: torch.Tensor, valid: torch.Tensor, name: str
) -> torch.Tensor:
    if name.startswith("delta_"):
        base = gpu_representation(mobility, valid, name.removeprefix("delta_"))
        output = torch.full_like(base, torch.nan)
        output[:, 1:] = (base[:, 1:] - base[:, :-1]).abs()
        return output.masked_fill(~valid, torch.nan)
    if name in LAYER_NAMES:
        output = mobility[:, :, LAYER_NAMES.index(name)]
    elif name == "front_back_mean_gap":
        output = (mobility[:, :, :4].mean(dim=2) - mobility[:, :, 4:].mean(dim=2)).abs()
    else:
        group, operation = name.split("_", 1)
        selected = mobility[
            :, :, {"front": slice(0, 4), "back": slice(4, 8), "all": slice(0, 8)}[group]
        ]
        if operation == "mean":
            output = selected.mean(dim=2)
        elif operation == "min":
            output = selected.amin(dim=2)
        elif operation == "max":
            output = selected.amax(dim=2)
        elif operation == "median":
            output = _median(selected)
        elif operation == "std":
            output = selected.std(dim=2, correction=0)
        else:
            raise KeyError(name)
    return output.masked_fill(~valid, torch.nan)


def gpu_trailing_mean(values: torch.Tensor, width: int) -> torch.Tensor:
    finite = values.isfinite()
    filled = torch.where(finite, values, 0.0)
    csum = torch.nn.functional.pad(filled.cumsum(dim=1), (1, 0))
    count = torch.nn.functional.pad(finite.to(torch.int16).cumsum(dim=1), (1, 0))
    output = torch.full_like(values, torch.nan)
    total = csum[:, width:] - csum[:, :-width]
    complete = count[:, width:] - count[:, :-width] == width
    output[:, width - 1 :] = torch.where(complete, total / width, torch.nan)
    return output


def gpu_persistent_score(oriented: torch.Tensor, count: int) -> torch.Tensor:
    output = torch.full_like(oriented, torch.nan)
    output[:, count - 1 :] = oriented.unfold(1, count, 1).amin(dim=2)
    return output


def gpu_crossfit_thresholds(
    peaks: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    if peaks.numel() % 400:
        raise ValueError("development cache must contain 400 episodes per task")
    task_count = peaks.numel() // 400
    grid = peaks.reshape(task_count, 50, 8)
    expanded = grid[:, None, :, :].expand(task_count, 50, 50, 8)
    keep = (~torch.eye(50, dtype=torch.bool, device=peaks.device))[None, :, :, None]
    keep = keep.expand(task_count, 50, 50, 8)
    reference = expanded.masked_select(keep).reshape(task_count, 50, 392)
    finite = reference.isfinite()
    finite_count = finite.sum(dim=2)
    if int(finite_count.min().item()) < 32:
        raise ValueError("fewer than 32 finite calibration peaks")
    ordered = torch.where(finite, reference, torch.inf).sort(dim=2).values
    positions = torch.ceil(
        quantiles[None, None, :] * (finite_count[:, :, None] - 1)
    ).to(torch.int64)
    threshold_grid = ordered.gather(2, positions)
    return (
        threshold_grid[:, :, None, :]
        .expand(task_count, 50, 8, -1)
        .reshape(peaks.numel(), -1)
    )


def gpu_first_query(
    persistent: torch.Tensor, thresholds: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    trigger = (
        persistent[:, :, None].isfinite()
        & (persistent[:, :, None] > thresholds[:, None, :])
        & valid[:, :, None]
    )
    fired = trigger.any(dim=1)
    first = trigger.to(torch.int8).argmax(dim=1).to(torch.int16)
    return torch.where(fired, first, -1)


def gpu_first_or(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    sentinel = torch.iinfo(torch.int16).max
    left_available = left >= 0
    right_available = right >= 0
    first = torch.minimum(
        torch.where(left_available, left, sentinel),
        torch.where(right_available, right, sentinel),
    )
    return torch.where(left_available | right_available, first, -1)


def _metric_context(
    labels: pd.DataFrame,
    task_index: torch.Tensor,
    suite_index: torch.Tensor,
    spatial: torch.Tensor,
) -> dict[str, Any]:
    device = task_index.device
    risk = torch.as_tensor(
        labels["original_failure"].to_numpy(bool, copy=True), device=device
    )
    timely = ~risk
    lengths = torch.as_tensor(
        labels["length"].to_numpy(np.int16, copy=True), device=device
    )
    task_count = int(task_index.max().item()) + 1
    suite_count = int(suite_index.max().item()) + 1
    timely_task = torch.nn.functional.one_hot(task_index, task_count).to(torch.float32)
    timely_task *= timely[:, None]
    timely_suite = torch.nn.functional.one_hot(suite_index, suite_count).to(
        torch.float32
    )
    timely_suite *= timely[:, None]
    spatial_risk = risk & spatial
    spatial_timely = timely & spatial
    return {
        "risk": risk,
        "timely": timely,
        "lengths": lengths,
        "timely_task": timely_task,
        "timely_suite": timely_suite,
        "task_denominator": timely_task.sum(dim=0).clamp_min(1),
        "suite_denominator": timely_suite.sum(dim=0).clamp_min(1),
        "spatial_risk": spatial_risk,
        "spatial_timely": spatial_timely,
        "risk_n": int(risk.sum().item()),
        "timely_n": int(timely.sum().item()),
        "spatial_risk_n": int(spatial_risk.sum().item()),
        "spatial_timely_n": int(spatial_timely.sum().item()),
    }


def _metric_vectors(
    first: torch.Tensor, context: dict[str, Any]
) -> dict[str, np.ndarray]:
    risk = context["risk"]
    timely = context["timely"]
    lengths = context["lengths"]
    alarm = first >= 0
    lead = lengths[:, None] - 1 - first
    tp = (alarm & risk[:, None]).sum(dim=0)
    fp = (alarm & timely[:, None]).sum(dim=0)
    early4 = (alarm & risk[:, None] & (lead >= 4)).sum(dim=0)
    early8 = (alarm & risk[:, None] & (lead >= 8)).sum(dim=0)
    alarm_float = alarm.transpose(0, 1).to(torch.float32)
    task_fp = alarm_float @ context["timely_task"]
    suite_fp = alarm_float @ context["timely_suite"]
    spatial_risk = context["spatial_risk"]
    spatial_timely = context["spatial_timely"]
    spatial_tp = (alarm & spatial_risk[:, None]).sum(dim=0)
    spatial_fp = (alarm & spatial_timely[:, None]).sum(dim=0)
    return {
        "tp": tp.cpu().numpy(),
        "fp": fp.cpu().numpy(),
        "early4": early4.cpu().numpy(),
        "early8": early8.cpu().numpy(),
        "max_task_fpr": (task_fp / context["task_denominator"])
        .amax(dim=1)
        .cpu()
        .numpy(),
        "max_suite_fpr": (suite_fp / context["suite_denominator"])
        .amax(dim=1)
        .cpu()
        .numpy(),
        "spatial_tp": spatial_tp.cpu().numpy(),
        "spatial_fp": spatial_fp.cpu().numpy(),
        "risk_n": np.asarray(context["risk_n"]),
        "timely_n": np.asarray(context["timely_n"]),
        "spatial_risk_n": np.asarray(context["spatial_risk_n"]),
        "spatial_timely_n": np.asarray(context["spatial_timely_n"]),
    }


def _rows_for_grid(
    representation: str,
    direction: str,
    width: int,
    confirmations: int,
    first: torch.Tensor,
    metric_context: dict[str, Any],
    device_index: int,
) -> list[dict[str, Any]]:
    values = _metric_vectors(first, metric_context)
    risk_n = int(values["risk_n"])
    timely_n = int(values["timely_n"])
    spatial_risk_n = int(values["spatial_risk_n"])
    spatial_timely_n = int(values["spatial_timely_n"])
    rows: list[dict[str, Any]] = []
    for position, quantile in enumerate(QUANTILES):
        tp = int(values["tp"][position])
        fp = int(values["fp"][position])
        row = {
            "representation": representation,
            "direction": direction,
            "width": width,
            "confirmations": confirmations,
            "quantile": quantile,
            "tp": tp,
            "fp": fp,
            "risk_recall": tp / risk_n,
            "early4_risk_recall": int(values["early4"][position]) / risk_n,
            "early8_risk_recall": int(values["early8"][position]) / risk_n,
            "precision": tp / max(tp + fp, 1),
            "timely_fpr": fp / timely_n,
            "max_suite_fpr": float(values["max_suite_fpr"][position]),
            "max_task_fpr": float(values["max_task_fpr"][position]),
            "spatial_tp": int(values["spatial_tp"][position]),
            "spatial_fp": int(values["spatial_fp"][position]),
            "spatial_recall": int(values["spatial_tp"][position]) / spatial_risk_n,
            "spatial_fpr": int(values["spatial_fp"][position]) / spatial_timely_n,
            "logical_gpu": device_index,
        }
        row["eligible"] = bool(
            row["timely_fpr"] <= OVERALL_FPR_LIMIT
            and row["precision"] >= PRECISION_MIN
            and row["risk_recall"] >= RECALL_MIN
            and row["max_suite_fpr"] <= SUITE_FPR_LIMIT
            and row["max_task_fpr"] <= TASK_FPR_LIMIT
        )
        rows.append(row)
    return rows


def gpu_worker(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    device_index = int(payload["device"])
    torch.cuda.set_device(device_index)
    torch.set_float32_matmul_precision("high")
    device = torch.device(f"cuda:{device_index}")
    cache = dev.load_npz(Path(payload["layer_root"]) / "main_reference.npz")
    labels = v3.aligned_labels(
        cache,
        Path(payload["label_root"]) / "development_main_clean_labels.csv",
        "development_main",
    )
    with np.load(
        Path(payload["long_result"]) / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        base_first_np = np.asarray(archive["main_hybrid"], dtype=np.int16)
    with np.load(
        Path(payload["v4_result"]) / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        instability_np = np.asarray(archive["main_instability"], dtype=np.int16)

    mobility = torch.as_tensor(cache["mobility"], device=device)
    valid = torch.as_tensor(cache["valid"], device=device)
    task_index = torch.as_tensor(cache["task_index"].astype(np.int64), device=device)
    suite_codes = pd.factorize(labels["suite"], sort=True)[0]
    suite_index = torch.as_tensor(suite_codes, dtype=torch.int64, device=device)
    spatial = torch.as_tensor(
        labels["suite"].eq(SPATIAL_SUITE).to_numpy(copy=True), device=device
    )
    base_first = torch.as_tensor(base_first_np, device=device)
    instability = torch.as_tensor(instability_np, device=device)[:, None]
    quantiles = torch.as_tensor(QUANTILES, dtype=torch.float32, device=device)
    metric_context = _metric_context(labels, task_index, suite_index, spatial)

    rows: list[dict[str, Any]] = []
    specs = [tuple(item) for item in payload["specs"]]
    for spec_position, (representation, direction) in enumerate(specs):
        values = gpu_representation(mobility, valid, representation)
        for width in WIDTHS:
            instant = gpu_trailing_mean(values, width)
            if direction == "low":
                instant = -instant
            peaks = torch.where(instant.isfinite(), instant, -torch.inf).amax(dim=1)
            thresholds = gpu_crossfit_thresholds(peaks, quantiles)
            for confirmations in CONFIRMATIONS:
                score = gpu_persistent_score(instant, confirmations)
                candidate = gpu_first_query(score, thresholds, valid)
                candidate_or = gpu_first_or(candidate, instability.expand_as(candidate))
                combined = torch.where(
                    spatial[:, None], candidate_or, base_first[:, None]
                )
                rows.extend(
                    _rows_for_grid(
                        representation,
                        direction,
                        width,
                        confirmations,
                        combined,
                        metric_context,
                        device_index,
                    )
                )
        print(
            f"[gpu {device_index} {spec_position + 1}/{len(specs)}] {representation}:{direction}",
            flush=True,
        )
    torch.cuda.synchronize(device)
    return {
        "rows": rows,
        "runtime": {
            "logical_gpu": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "candidate_rows": len(rows),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device_index)),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def np_crossfit_threshold(
    peaks: np.ndarray, task_index: np.ndarray, init_state: np.ndarray, quantile: float
) -> np.ndarray:
    output = np.full(len(peaks), np.nan, np.float32)
    for task in np.unique(task_index):
        take = np.flatnonzero(task_index == task)
        task_init = init_state[take]
        for held_out in np.unique(task_init):
            test = task_init == held_out
            reference = peaks[take[~test]]
            reference = np.sort(reference[np.isfinite(reference)])
            if len(reference) < 32:
                raise ValueError("fewer than 32 finite calibration peaks")
            position = int(np.ceil(quantile * (len(reference) - 1)))
            output[take[test]] = reference[position]
    if not np.isfinite(output).all():
        raise ValueError("incomplete CPU cross-fitted threshold")
    return output


def np_instant(
    representations: dict[str, tuple[np.ndarray, str]],
    cache: dict[str, np.ndarray],
    representation: str,
    direction: str,
    width: int,
) -> np.ndarray:
    values = representations[representation][0]
    instant = dev.trailing_mean(values, width)
    if direction == "low":
        instant = -instant
    instant[~cache["valid"].astype(bool)] = np.nan
    return instant


def metric_constraints(first: np.ndarray, labels: pd.DataFrame) -> dict[str, Any]:
    row = long_guard.metric_row("development_main", "candidate", first, labels)
    suite_rows = long_guard.group_rows(
        "development_main", "candidate", first, labels, "suite"
    )
    task_rows = long_guard.group_rows(
        "development_main", "candidate", first, labels, "task"
    )
    spatial_row = next(item for item in suite_rows if item["group"] == SPATIAL_SUITE)
    row.update(
        {
            "max_suite_fpr": max(item["timely_fpr"] for item in suite_rows),
            "max_task_fpr": max(item["timely_fpr"] for item in task_rows),
            "spatial_tp": spatial_row["tp"],
            "spatial_fp": spatial_row["fp"],
            "spatial_recall": spatial_row["risk_recall"],
            "spatial_fpr": spatial_row["timely_fpr"],
        }
    )
    row["eligible"] = bool(
        row["timely_fpr"] <= OVERALL_FPR_LIMIT
        and row["precision"] >= PRECISION_MIN
        and row["risk_recall"] >= RECALL_MIN
        and row["max_suite_fpr"] <= SUITE_FPR_LIMIT
        and row["max_task_fpr"] <= TASK_FPR_LIMIT
    )
    return row


RANK_COLUMNS = (
    "early4_risk_recall",
    "risk_recall",
    "precision",
    "timely_fpr",
    "max_task_fpr",
    "confirmations",
    "width",
    "quantile",
    "representation",
    "direction",
)
RANK_ASCENDING = (False, False, False, True, True, False, False, False, True, True)


def sorted_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(list(RANK_COLUMNS), ascending=list(RANK_ASCENDING))


def cpu_validate_shortlist(
    shortlist: pd.DataFrame,
    cache: dict[str, np.ndarray],
    labels: pd.DataFrame,
    base_first: np.ndarray,
    instability: np.ndarray,
) -> tuple[
    pd.DataFrame, dict[tuple[str, str, int, int, float], tuple[np.ndarray, np.ndarray]]
]:
    all_representations = dev.representations(cache)
    instant_cache: dict[tuple[str, str, int], np.ndarray] = {}
    threshold_cache: dict[tuple[str, str, int, float], np.ndarray] = {}
    score_cache: dict[tuple[str, str, int, int], np.ndarray] = {}
    alarm_cache: dict[
        tuple[str, str, int, int, float], tuple[np.ndarray, np.ndarray]
    ] = {}
    spatial = labels["suite"].eq(SPATIAL_SUITE).to_numpy()
    rows: list[dict[str, Any]] = []
    for gpu_rank, candidate in enumerate(shortlist.itertuples(index=False), start=1):
        representation = str(candidate.representation)
        direction = str(candidate.direction)
        width = int(candidate.width)
        confirmations = int(candidate.confirmations)
        quantile = float(candidate.quantile)
        instant_key = (representation, direction, width)
        if instant_key not in instant_cache:
            instant_cache[instant_key] = np_instant(
                all_representations, cache, representation, direction, width
            )
        threshold_key = (*instant_key, quantile)
        if threshold_key not in threshold_cache:
            threshold_cache[threshold_key] = np_crossfit_threshold(
                dev.row_max(instant_cache[instant_key]),
                cache["task_index"].astype(int),
                cache["init_state_id"].astype(int),
                quantile,
            )
        score_key = (*instant_key, confirmations)
        if score_key not in score_cache:
            score_cache[score_key] = dev.persistent_score(
                instant_cache[instant_key], confirmations
            )
        key = (representation, direction, width, confirmations, quantile)
        selected_threshold = threshold_cache[threshold_key]
        selected_first = v4.first_from_threshold(
            score_cache[score_key], cache["valid"].astype(bool), selected_threshold
        )
        combined = base_first.copy()
        combined[spatial] = v4.first_or(selected_first[spatial], instability[spatial])
        metrics = metric_constraints(combined, labels)
        metrics.update(
            {
                "representation": representation,
                "direction": direction,
                "width": width,
                "confirmations": confirmations,
                "quantile": quantile,
                "gpu_rank": gpu_rank,
            }
        )
        rows.append(metrics)
        alarm_cache[key] = (selected_first, selected_threshold)
    return pd.DataFrame(rows), alarm_cache


def external_selected_head(
    external: dict[str, np.ndarray],
    main_reference: dict[str, np.ndarray],
    extra_reference: dict[str, np.ndarray],
    representation: str,
    direction: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    external_representations = dev.representations(external)
    instant = np_instant(
        external_representations, external, representation, direction, width
    )
    score = dev.persistent_score(instant, confirmations)
    valid = external["valid"].astype(bool)
    task_index = external["task_index"].astype(int)
    first = np.full(len(score), -1, np.int16)
    thresholds = np.full(len(score), np.nan, np.float32)
    rows: list[dict[str, Any]] = []
    representation_cache: dict[int, dict[str, tuple[np.ndarray, str]]] = {}
    instant_cache: dict[int, np.ndarray] = {}
    for task_position, task in enumerate(external["task_names"].astype(str)):
        test = np.flatnonzero(task_index == task_position)
        reference, take = v3.reference_for_task(task, main_reference, extra_reference)
        key = id(reference)
        if key not in representation_cache:
            representation_cache[key] = dev.representations(reference)
            instant_cache[key] = np_instant(
                representation_cache[key], reference, representation, direction, width
            )
        threshold = dev.quantile_higher(dev.row_max(instant_cache[key][take]), quantile)
        thresholds[test] = threshold
        first[test] = v4.first_from_threshold(score[test], valid[test], threshold)
        rows.append(
            {
                "task": task,
                "representation": representation,
                "direction": direction,
                "width": width,
                "confirmations": confirmations,
                "quantile": quantile,
                "threshold": threshold,
                "reference_episodes": len(take),
                "outcomes_used": False,
                "active_in_profile": task.startswith(f"{SPATIAL_SUITE}/"),
            }
        )
    if not np.isfinite(thresholds).all():
        raise ValueError("incomplete external threshold assignment")
    return first, thresholds, rows


def save_profile(
    output: Path,
    long_result: Path,
    threshold_rows: list[dict[str, Any]],
    representation: str,
    direction: str,
    width: int,
    confirmations: int,
    quantile: float,
) -> None:
    with np.load(
        long_result / "deployment_profiles.npz", allow_pickle=False
    ) as archive:
        base = {name: np.asarray(archive[name]) for name in archive.files}
    tasks = base["task_names"].astype(str)
    spatial = np.char.startswith(tasks, f"{SPATIAL_SUITE}/")
    selected_thresholds = np.asarray(
        [row["threshold"] for row in threshold_rows], dtype=np.float32
    )
    lock_thresholds = base["lock_thresholds"].astype(np.float32, copy=True)
    # The v5 array is <U10 ("all_median"); use headroom for longer v6 names.
    lock_representations = base["lock_representations"].astype("<U64", copy=True)
    lock_widths = base["lock_widths"].astype(np.int16, copy=True)
    lock_confirmations = base["lock_confirmations"].astype(np.int16, copy=True)
    lock_quantiles = base["lock_quantiles"].astype(np.float32, copy=True)
    lock_directions = np.full(len(tasks), "low", dtype="<U4")
    lock_thresholds[spatial] = selected_thresholds[spatial]
    lock_representations[spatial] = representation
    lock_widths[spatial] = width
    lock_confirmations[spatial] = confirmations
    lock_quantiles[spatial] = quantile
    lock_directions[spatial] = direction
    np.savez_compressed(
        output / "deployment_profiles.npz",
        schema=np.asarray("himoe.spatial_guard_v6.profile.v1"),
        task_names=tasks,
        lock_thresholds=lock_thresholds,
        instability_thresholds=base["instability_thresholds"],
        lock_representations=lock_representations,
        lock_directions=lock_directions,
        lock_widths=lock_widths,
        lock_confirmations=lock_confirmations,
        lock_quantiles=lock_quantiles,
        instability_layers=base["instability_layers"],
        instability_widths=base["instability_widths"],
        instability_confirmations=base["instability_confirmations"],
        instability_quantiles=base["instability_quantiles"],
    )


def main() -> None:
    args = parse_args()
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if len(devices) != 2:
        raise ValueError("v6 sweep requires exactly two logical CUDA devices")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(devices):
        raise RuntimeError(
            f"requested CUDA devices {devices}, available count={torch.cuda.device_count()}"
        )

    main_cache = dev.load_npz(args.layer_root / "main_reference.npz")
    extra_cache = dev.load_npz(args.layer_root / "extra_reference.npz")
    external_cache = dev.load_npz(args.layer_root / "external_8b.npz")
    development_labels = v3.aligned_labels(
        main_cache,
        args.label_root / "development_main_clean_labels.csv",
        "development_main",
    )
    with np.load(
        args.v4_result / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        main_v4 = np.asarray(archive["main_dual"], dtype=np.int16)
        external_v4 = np.asarray(archive["external_dual"], dtype=np.int16)
        main_instability = np.asarray(archive["main_instability"], dtype=np.int16)
        external_instability = np.asarray(
            archive["external_instability"], dtype=np.int16
        )
    with np.load(
        args.long_result / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        main_long = np.asarray(archive["main_hybrid"], dtype=np.int16)
        external_long = np.asarray(archive["external_hybrid"], dtype=np.int16)
        main_long_lock = np.asarray(archive["main_deployed_lock"], dtype=np.int16)
        external_long_lock = np.asarray(
            archive["external_deployed_lock"], dtype=np.int16
        )
    with np.load(
        args.lead_result / "sealed_first_alarms.npz", allow_pickle=False
    ) as archive:
        main_lead = np.asarray(archive["main_hybrid"], dtype=np.int16)
        external_lead = np.asarray(archive["external_hybrid"], dtype=np.int16)

    specs = representation_specs()
    partitions = [specs[position :: len(devices)] for position in range(len(devices))]
    payloads = [
        {
            "device": device,
            "specs": partitions[position],
            "layer_root": str(args.layer_root),
            "label_root": str(args.label_root),
            "v4_result": str(args.v4_result),
            "long_result": str(args.long_result),
        }
        for position, device in enumerate(devices)
    ]
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=len(devices), mp_context=context
    ) as executor:
        worker_results = list(executor.map(gpu_worker, payloads))
    gpu_rows = [row for result in worker_results for row in result["rows"]]
    gpu_runtime = [result["runtime"] for result in worker_results]
    gpu_candidates = pd.DataFrame(gpu_rows)
    expected = len(specs) * len(WIDTHS) * len(CONFIRMATIONS) * len(QUANTILES)
    if len(gpu_candidates) != expected:
        raise AssertionError(
            f"expected {expected} candidates, got {len(gpu_candidates)}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    gpu_candidates.to_csv(args.output / "gpu_development_candidates.csv", index=False)

    eligible_gpu = sorted_candidates(gpu_candidates[gpu_candidates["eligible"]])
    if eligible_gpu.empty:
        raise RuntimeError("GPU sweep produced no eligible candidate")
    shortlist = eligible_gpu.head(SHORTLIST_SIZE).reset_index(drop=True)
    validated, alarm_cache = cpu_validate_shortlist(
        shortlist,
        main_cache,
        development_labels,
        main_long,
        main_instability,
    )
    validated.to_csv(args.output / "cpu_validated_shortlist.csv", index=False)
    eligible_cpu = sorted_candidates(validated[validated["eligible"]])
    if eligible_cpu.empty:
        raise RuntimeError("no GPU-shortlisted candidate passed CPU validation")
    selected = eligible_cpu.iloc[0]
    selection_key = (
        str(selected["representation"]),
        str(selected["direction"]),
        int(selected["width"]),
        int(selected["confirmations"]),
        float(selected["quantile"]),
    )
    main_selected_lock, main_selected_threshold = alarm_cache[selection_key]
    spatial_main = development_labels["suite"].eq(SPATIAL_SUITE).to_numpy()
    main_deployed_lock = main_long_lock.copy()
    main_deployed_lock[spatial_main] = main_selected_lock[spatial_main]
    main_hybrid = main_long.copy()
    main_hybrid[spatial_main] = v4.first_or(
        main_selected_lock[spatial_main], main_instability[spatial_main]
    )

    selection = {
        "schema": "himoe.spatial_guard_v6.selection.v1",
        "status": "selected",
        "external_outcomes_read_by_selection": False,
        "candidate_count": len(gpu_candidates),
        "gpu_eligible_count": int(gpu_candidates["eligible"].sum()),
        "cpu_shortlist_count": len(validated),
        "cpu_eligible_count": int(validated["eligible"].sum()),
        "constraints": {
            "overall_fpr_max": OVERALL_FPR_LIMIT,
            "precision_min": PRECISION_MIN,
            "recall_min": RECALL_MIN,
            "suite_fpr_max": SUITE_FPR_LIMIT,
            "task_fpr_max": TASK_FPR_LIMIT,
        },
        "ranking": list(RANK_COLUMNS),
        "selected": selected.to_dict(),
    }
    (args.output / "selection.json").write_text(
        json.dumps(plain(selection), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    representation, direction, width, confirmations, quantile = selection_key
    external_selected_lock, external_selected_threshold, threshold_rows = (
        external_selected_head(
            external_cache,
            main_cache,
            extra_cache,
            representation,
            direction,
            width,
            confirmations,
            quantile,
        )
    )
    external_episode_tasks = external_cache["task_names"].astype(str)[
        external_cache["task_index"].astype(int)
    ]
    spatial_external = np.char.startswith(external_episode_tasks, f"{SPATIAL_SUITE}/")
    external_deployed_lock = external_long_lock.copy()
    external_deployed_lock[spatial_external] = external_selected_lock[spatial_external]
    external_hybrid = external_long.copy()
    external_hybrid[spatial_external] = v4.first_or(
        external_selected_lock[spatial_external],
        external_instability[spatial_external],
    )
    pd.DataFrame(threshold_rows).to_csv(
        args.output / "external_outcome_blind_thresholds.csv", index=False
    )
    save_profile(
        args.output,
        args.long_result,
        threshold_rows,
        representation,
        direction,
        width,
        confirmations,
        quantile,
    )
    np.savez_compressed(
        args.output / "sealed_first_alarms.npz",
        schema=np.asarray("himoe.spatial_guard_v6.alarms.v1"),
        representation=np.asarray(representation),
        direction=np.asarray(direction),
        width=np.asarray(width),
        confirmations=np.asarray(confirmations),
        quantile=np.asarray(quantile),
        main_selected_lock=main_selected_lock,
        main_deployed_lock=main_deployed_lock,
        main_hybrid=main_hybrid,
        main_selected_threshold=main_selected_threshold,
        external_selected_lock=external_selected_lock,
        external_deployed_lock=external_deployed_lock,
        external_hybrid=external_hybrid,
        external_selected_threshold=external_selected_threshold,
    )

    # External outcomes are intentionally loaded only after the rule and alarms are sealed.
    external_labels = v3.aligned_labels(
        external_cache,
        args.label_root / "external_8b_clean_labels.csv",
        "external_8b",
    )
    frames: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for cohort, labels, baseline, long_alarm, lead_alarm, v6_alarm in (
        (
            "development_main",
            development_labels,
            main_v4,
            main_long,
            main_lead,
            main_hybrid,
        ),
        (
            "external_8b",
            external_labels,
            external_v4,
            external_long,
            external_lead,
            external_hybrid,
        ),
    ):
        frames.append(
            long_guard.tables(
                cohort,
                labels,
                {
                    "v4_dual_regime_or": baseline,
                    "long_guard_v5": long_alarm,
                    "lead_guard_v5": lead_alarm,
                    "spatial_guard_v6": v6_alarm,
                },
            )
        )
    overall = pd.concat([frame[0] for frame in frames], ignore_index=True)
    by_suite = pd.concat([frame[1] for frame in frames], ignore_index=True)
    by_task = pd.concat([frame[2] for frame in frames], ignore_index=True)
    overall.to_csv(args.output / "outcome_metrics.csv", index=False)
    by_suite.to_csv(args.output / "outcome_metrics_by_suite.csv", index=False)
    by_task.to_csv(args.output / "outcome_metrics_by_task.csv", index=False)
    pd.concat(
        [
            development_labels.assign(
                first_v4_query=main_v4,
                first_long_guard_v5_query=main_long,
                first_lead_guard_v5_query=main_lead,
                first_spatial_guard_v6_query=main_hybrid,
            ),
            external_labels.assign(
                first_v4_query=external_v4,
                first_long_guard_v5_query=external_long,
                first_lead_guard_v5_query=external_lead,
                first_spatial_guard_v6_query=external_hybrid,
            ),
        ],
        ignore_index=True,
    ).to_csv(args.output / "episode_alarms.csv", index=False)

    summary = {
        "schema": "himoe.spatial_guard_v6.evaluation.v1",
        "status": (
            "post-hoc suite specialization on previously inspected external 8B; "
            "requires pristine confirmation"
        ),
        "runtime_moe_only": True,
        "runtime_query_causal": True,
        "runtime_outcome_access": False,
        "threshold_calibration_outcome_free": True,
        "method_selection_used_development_outcomes": True,
        "external_outcomes_loaded_after_alarm_seal_in_evaluator": True,
        "external_cohort_pristine_holdout": False,
        "gpu_sweep": {
            "visible_physical_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "logical_devices": devices,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "workers": gpu_runtime,
        },
        "selection": selection,
        "metrics": overall.to_dict(orient="records"),
        "artifacts": {
            "evaluator_sha256": sha256(Path(__file__)),
            "protocol_sha256": sha256(PROTOCOL),
            "selection_sha256": sha256(args.output / "selection.json"),
            "sealed_first_alarms_sha256": sha256(
                args.output / "sealed_first_alarms.npz"
            ),
            "deployment_profiles_sha256": sha256(
                args.output / "deployment_profiles.npz"
            ),
            "outcome_metrics_sha256": sha256(args.output / "outcome_metrics.csv"),
        },
    }
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        overall[
            [
                "cohort",
                "detector",
                "tp",
                "fp",
                "risk_recall",
                "early4_risk_recall",
                "precision",
                "timely_fpr",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
