#!/usr/bin/env python3
"""Apply the frozen A/B near-Trap probability tables to cache_new routes.

The route-only phase never opens client summaries. It writes and hashes all
per-query probabilities before endpoint outcomes are loaded for a separate,
descriptive audit. cache_new has no common query-level Trap onset, so endpoint
metrics in this script are not interpreted as H=2 Trap calibration metrics.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import zarr

import analyze_trainfree_signal_matrix as signal_matrix
import analyze_trainfree_trap_probability as probability


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/trainfree_trap_probability_cache_new.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/trainfree_trap_probability/cache_new_40task"
SCHEMA = "himoe.trainfree_trap_probability_cache_new.v1"
CALIBRATORS = ("A_dense", "B_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def resolve_workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def discover_runs(cache_root: Path, run_id: str) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    pattern = f"libero_*/*/{run_id}/client/summaries.json"
    for summary_path in sorted(cache_root.glob(pattern)):
        run = summary_path.parents[1]
        meta_path = run / "meta.json"
        route_path = run / "server/routes.zarr"
        if not meta_path.exists() or not route_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        if (
            meta.get("status") == "complete"
            and sampling.get("complete") is True
            and int(sampling.get("actual_episodes", -1))
            == int(sampling.get("designed_episodes", -2))
        ):
            task = str(run.relative_to(cache_root).parent)
            runs.append((task, run))
    if not runs:
        raise RuntimeError(f"no complete {run_id!r} runs below {cache_root}")
    return runs


def load_calibrators(
    config: dict[str, Any], probability_config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    table = pd.read_csv(resolve_workspace(config["calibration_table"]))
    minimum_query = int(config["minimum_query"])
    edges = np.asarray(probability_config["score_bins"], np.float64)
    cap = int(probability_config["persistence_cap"])
    calibrators: dict[str, dict[str, Any]] = {}
    for name, item in config["source_calibrators"].items():
        artifact = resolve_workspace(item["route_artifact"])
        with np.load(artifact) as archive:
            valid = np.asarray(archive["valid"], bool)
            valid[:, :minimum_query] = False
            references = {
                feature: np.sort(np.asarray(archive[feature], np.float64)[valid])
                for feature in ("late_flow", "acceleration", "periodicity")
            }
        selected = table[
            (table["direction"] == item["calibration_direction"])
            & (table["event_kind"] == config["calibration_event_kind"])
            & (table["horizon"] == int(config["calibration_horizon_queries"]))
        ]
        if len(selected) != (len(edges) - 1) * (cap + 1):
            raise ValueError(f"incomplete calibration table for {name}")
        matrices = {
            key: np.full((len(edges) - 1, cap + 1), np.nan, np.float64)
            for key in ("probability", "ci_low", "ci_high", "support", "events")
        }
        for row in selected.itertuples(index=False):
            index = (int(row.score_bin), int(row.persistence))
            matrices["probability"][index] = float(row.posterior_mean)
            matrices["ci_low"][index] = float(row.ci_low)
            matrices["ci_high"][index] = float(row.ci_high)
            matrices["support"][index] = float(row.mapped_support)
            matrices["events"][index] = float(row.mapped_events)
        if any(np.isnan(values).any() for values in matrices.values()):
            raise ValueError(f"non-finite calibration matrix for {name}")
        calibrators[name] = {
            "references": references,
            "mapping": matrices,
            "artifact_sha256": sha256_file(artifact),
            "direction": item["calibration_direction"],
        }
    if tuple(calibrators) != CALIBRATORS:
        raise ValueError(f"expected calibrators {CALIBRATORS}, got {tuple(calibrators)}")
    return calibrators


def percentile(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(values.shape, np.nan, np.float64)
    good = np.isfinite(values)
    result[good] = (
        np.searchsorted(sorted_reference, values[good], side="right")
        / len(sorted_reference)
    )
    return result


def route_scalars(router: Any, block: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total = int(router.shape[0])
    late_flow = np.empty(total, np.float32)
    acceleration = np.empty(total, np.float32)
    final_action = np.empty((total, 4, 10, 32), np.float32)
    for start in range(0, total, block):
        stop = min(start + block, total)
        routes = np.asarray(router[start:stop, 4:8, :, :, :], np.float32)
        routes = signal_matrix.normalize_probability(routes)
        action = routes[:, :, :, 1:11, :]
        late = action[:, :, signal_matrix.FLOW_LATE_START :]
        late_flow[start:stop] = (
            1.0
            - signal_matrix.weighted_jaccard(
                late[:, :, 1:], late[:, :, :-1]
            )
        ).mean((1, 2, 3))
        root = np.sqrt(action)
        curve = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        acceleration[start:stop] = (
            np.linalg.norm(curve, axis=-1).mean((1, 2, 3)) / np.sqrt(2.0)
        )
        final_action[start:stop] = action[:, :, signal_matrix.DENOISE_FINAL]
    return late_flow, acceleration, final_action


def task_cache_path(cache_dir: Path, task: str) -> Path:
    token = hashlib.sha1(task.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{token}.npz"


def extract_task_probabilities(
    task: str,
    run_string: str,
    cache_string: str,
    calibrators: dict[str, dict[str, Any]],
    probability_config: dict[str, Any],
    minimum_query: int,
    baseline: tuple[int, int],
    rebuild: bool,
) -> dict[str, Any]:
    run = Path(run_string)
    cache_path = task_cache_path(Path(cache_string), task)
    if cache_path.exists() and not rebuild:
        with np.load(cache_path) as archive:
            if str(archive["schema"].item()) == SCHEMA and str(archive["task"].item()) == task:
                return {
                    "task": task,
                    "cache": str(cache_path),
                    "route_rows": int(archive["route_rows"].item()),
                    "eligible_rows": int(len(archive["episode"])),
                    "episodes": int(len(np.unique(archive["all_episode_ids"]))),
                    "cache_reused": True,
                }

    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    episode_ids = np.asarray(group["episode_id"][:], np.int32)
    control_steps = np.asarray(group["control_step"][:], np.int32)
    if len(control_steps) > 1 and np.any(np.diff(control_steps) != 1):
        raise ValueError(f"non-contiguous global control steps: {task}")
    late_flow, acceleration, final_action = route_scalars(group["hb_router_probs"])
    unique_episodes = np.unique(episode_ids)
    output: dict[str, list[np.ndarray]] = {
        "episode": [],
        "query": [],
        "route_length": [],
    }
    for name in CALIBRATORS:
        for field in (
            "score",
            "persistence",
            "score_bin",
            "probability",
            "ci_low",
            "ci_high",
            "calibration_support",
            "calibration_events",
        ):
            output[f"{field}_{name}"] = []

    edges = np.asarray(probability_config["score_bins"], np.float64)
    threshold = float(probability_config["persistence_score_threshold"])
    cap = int(probability_config["persistence_cap"])
    for episode in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode)
        if len(indices) > 1 and not np.all(np.diff(indices) == 1):
            raise ValueError(f"non-contiguous rows: {task}/{episode}")
        length = len(indices)
        late_w4 = signal_matrix.rolling_mean(
            late_flow[indices][None, :], signal_matrix.ALIGN_WINDOW
        )[0]
        acceleration_w4 = signal_matrix.rolling_mean(
            acceleration[indices][None, :], signal_matrix.ALIGN_WINDOW
        )[0]
        actions = final_action[indices].reshape(length, 40, 32)
        lag_similarity = np.full((length, len(signal_matrix.LAGS)), np.nan)
        for lag_index, lag in enumerate(signal_matrix.LAGS):
            if length > lag:
                lag_similarity[lag:, lag_index] = signal_matrix.weighted_jaccard(
                    actions[lag:], actions[:-lag]
                ).mean(axis=1)
        periodicity_w4 = signal_matrix.periodicity_from_lags(
            lag_similarity[None, :, :], signal_matrix.ALIGN_WINDOW
        )[0]
        start, stop = baseline
        normalized = {
            "late_flow": probability.robust_self_normalize(
                late_w4[None, :], start, stop, True
            )[0],
            "acceleration": probability.robust_self_normalize(
                acceleration_w4[None, :], start, stop, True
            )[0],
            "periodicity": probability.robust_self_normalize(
                periodicity_w4[None, :], start, stop, False
            )[0],
        }
        calibrator_values: dict[str, dict[str, np.ndarray]] = {}
        common_valid = np.arange(length) >= minimum_query
        for name in CALIBRATORS:
            item = calibrators[name]
            transformed = {
                feature: percentile(item["references"][feature], values)
                for feature, values in normalized.items()
            }
            loop = np.minimum(transformed["late_flow"], transformed["acceleration"])
            score = np.maximum(loop, transformed["periodicity"])
            valid = common_valid & np.isfinite(score)
            run_length = probability.persistence(
                score[None, :], np.isfinite(score)[None, :], threshold, cap
            )[0]
            score_bins = np.clip(
                np.searchsorted(edges, score, side="right") - 1,
                0,
                len(edges) - 2,
            ).astype(np.int8)
            mapping = item["mapping"]
            calibrator_values[name] = {
                "valid": valid,
                "score": score,
                "persistence": run_length,
                "score_bin": score_bins,
                "probability": mapping["probability"][score_bins, run_length],
                "ci_low": mapping["ci_low"][score_bins, run_length],
                "ci_high": mapping["ci_high"][score_bins, run_length],
                "calibration_support": mapping["support"][score_bins, run_length],
                "calibration_events": mapping["events"][score_bins, run_length],
            }
        selected = calibrator_values[CALIBRATORS[0]]["valid"]
        if not np.array_equal(selected, calibrator_values[CALIBRATORS[1]]["valid"]):
            raise AssertionError(f"calibrator validity mismatch: {task}/{episode}")
        query = np.flatnonzero(selected).astype(np.int16)
        if not len(query):
            continue
        output["episode"].append(np.full(len(query), episode, np.int32))
        output["query"].append(query)
        output["route_length"].append(np.full(len(query), length, np.int16))
        for name in CALIBRATORS:
            values = calibrator_values[name]
            for field in (
                "score",
                "persistence",
                "score_bin",
                "probability",
                "ci_low",
                "ci_high",
                "calibration_support",
                "calibration_events",
            ):
                output[f"{field}_{name}"].append(values[field][selected])

    arrays: dict[str, np.ndarray] = {
        "schema": np.asarray(SCHEMA),
        "task": np.asarray(task),
        "run": np.asarray(str(run)),
        "training": np.asarray(False),
        "target_labels_loaded": np.asarray(False),
        "route_rows": np.asarray(len(episode_ids)),
        "all_episode_ids": unique_episodes,
    }
    for key, chunks in output.items():
        if chunks:
            arrays[key] = np.concatenate(chunks)
        else:
            arrays[key] = np.asarray([], dtype=np.int32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    return {
        "task": task,
        "cache": str(cache_path),
        "route_rows": len(episode_ids),
        "eligible_rows": len(arrays["episode"]),
        "episodes": len(unique_episodes),
        "cache_reused": False,
    }


def load_prediction_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for record in sorted(records, key=lambda item: item["task"]):
        with np.load(record["cache"]) as archive:
            frame = pd.DataFrame(
                {
                    key: np.asarray(archive[key])
                    for key in archive.files
                    if key
                    not in {
                        "schema",
                        "task",
                        "run",
                        "training",
                        "target_labels_loaded",
                        "route_rows",
                        "all_episode_ids",
                    }
                }
            )
        frame.insert(0, "task", record["task"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_outcomes(runs: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, run in runs:
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        horizon_queries = int(math.ceil(int(meta["max_steps"]) / 10.0))
        summaries = json.loads(
            (run / "client/summaries.json").read_text(encoding="utf-8")
        )
        for item in summaries:
            rows.append(
                {
                    "task": task,
                    "episode": int(item["episode_index"]),
                    "success": bool(item["success"]),
                    "failure": not bool(item["success"]),
                    "inference_calls": int(item["inference_calls"]),
                    "horizon_queries": horizon_queries,
                    "init_state_id": int(item["init_state_id"]),
                    "flow_noise_seed": int(item["flow_noise_seed"]),
                }
            )
    return pd.DataFrame(rows)


def binary_rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, np.float64)
    if not np.any(labels) or np.all(labels):
        return {"roc_auc": np.nan, "average_precision": np.nan}
    ranks = rankdata(scores, method="average")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    auc = (
        ranks[labels].sum() - positives * (positives + 1) / 2
    ) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    precision_at = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    average_precision = (precision_at * sorted_labels).sum() / positives
    return {"roc_auc": float(auc), "average_precision": float(average_precision)}


def stratified_rank_metrics(
    frame: pd.DataFrame, score_column: str
) -> dict[str, float | int]:
    aucs: list[float] = []
    pair_counts: list[int] = []
    for _, group in frame.groupby("task", sort=False):
        labels = group["failure"].to_numpy(bool)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            continue
        auc = binary_rank_metrics(labels, group[score_column].to_numpy())["roc_auc"]
        aucs.append(auc)
        pair_counts.append(positives * negatives)
    weights = np.asarray(pair_counts, np.float64)
    return {
        "eligible_tasks": len(aucs),
        "macro_task_roc_auc": float(np.mean(aucs)) if aucs else np.nan,
        "pair_weighted_within_task_roc_auc": (
            float(np.average(aucs, weights=weights)) if aucs else np.nan
        ),
    }


def clock_baselines(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failure = outcomes["failure"].to_numpy(bool)
    success = ~failure
    length = outcomes["inference_calls"].to_numpy(int)
    horizon = outcomes["horizon_queries"].to_numpy(int)
    for phase in np.linspace(0.1, 1.0, 91):
        alarm_query = np.ceil(phase * horizon).astype(int) - 1
        alarm = length > alarm_query
        tp = int(np.sum(alarm & failure))
        fp = int(np.sum(alarm & success))
        rows.append(
            {
                "horizon_phase": float(phase),
                "alarm_query_min": int(alarm_query.min()),
                "alarm_query_max": int(alarm_query.max()),
                "tp": tp,
                "fp": fp,
                "failure_recall": tp / max(int(failure.sum()), 1),
                "success_false_alarm_rate": fp / max(int(success.sum()), 1),
                "endpoint_precision": tp / max(tp + fp, 1),
            }
        )
    return pd.DataFrame(rows)


def matched_clock(clock: pd.DataFrame, false_alarm_rate: float) -> dict[str, Any]:
    eligible = clock[
        clock["success_false_alarm_rate"] <= false_alarm_rate + 1e-15
    ]
    if len(eligible) == 0:
        selected = clock.sort_values("success_false_alarm_rate").iloc[0]
    else:
        selected = eligible.sort_values(
            ["failure_recall", "horizon_phase"], ascending=[False, True]
        ).iloc[0]
    return plain(selected.to_dict())


def quantiles(values: pd.Series) -> dict[str, float]:
    points = values.quantile([0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "count": int(values.notna().sum()),
        "mean": float(values.mean()),
        "min": float(points.loc[0.0]),
        "median": float(points.loc[0.5]),
        "p90": float(points.loc[0.9]),
        "p95": float(points.loc[0.95]),
        "p99": float(points.loc[0.99]),
        "max": float(points.loc[1.0]),
    }


def evaluate(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    query = predictions.merge(
        outcomes[["task", "episode", "success", "failure", "inference_calls"]],
        on=["task", "episode"],
        validate="many_to_one",
    )
    if not np.array_equal(query["route_length"], query["inference_calls"]):
        raise ValueError("route/client lengths differ")
    query["phase"] = query["query"] / np.maximum(query["inference_calls"] - 1, 1)
    bins = [0.0, 0.4, 0.6, 0.8, 1.000001]
    labels = ["eligible_to_40", "40_60", "60_80", "80_100"]
    query["phase_bin"] = pd.cut(
        query["phase"], bins=bins, labels=labels, include_lowest=True
    )

    summary: dict[str, Any] = {
        "endpoint_target_warning": config["evaluation_warning"],
        "total_episodes": len(outcomes),
        "total_successes": int(outcomes["success"].sum()),
        "total_failures": int(outcomes["failure"].sum()),
        "total_route_queries": int(outcomes["inference_calls"].sum()),
        "eligible_query_rows": len(predictions),
        "calibrators": {},
    }
    clock = clock_baselines(outcomes)
    episode_output = outcomes.copy()
    threshold_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    probability_frequency_rows: list[dict[str, Any]] = []
    fixed_query_rows: list[dict[str, Any]] = []
    for calibrator in CALIBRATORS:
        column = f"probability_{calibrator}"
        grouped = query.groupby(["task", "episode"], sort=False)[column]
        episode = grouped.agg(
            probability_first="first",
            probability_mean="mean",
            probability_max="max",
            scored_queries="size",
        ).reset_index()
        renamed = {
            key: f"{key}_{calibrator}"
            for key in (
                "probability_first",
                "probability_mean",
                "probability_max",
                "scored_queries",
            )
        }
        episode_output = episode_output.merge(
            episode.rename(columns=renamed),
            on=["task", "episode"],
            how="left",
            validate="one_to_one",
        )
        evaluated = outcomes.merge(
            episode, on=["task", "episode"], how="left", validate="one_to_one"
        )
        scoreable = evaluated["probability_max"].notna()
        scored = evaluated[scoreable]
        success_rows = query.loc[query["success"], column]
        failure_rows = query.loc[query["failure"], column]
        success_episode = scored.loc[scored["success"], "probability_max"]
        failure_episode = scored.loc[scored["failure"], "probability_max"]
        first_metrics = binary_rank_metrics(
            scored["failure"].to_numpy(), scored["probability_first"].to_numpy()
        )
        max_metrics = binary_rank_metrics(
            scored["failure"].to_numpy(), scored["probability_max"].to_numpy()
        )
        duration_metrics = binary_rank_metrics(
            scored["failure"].to_numpy(), scored["inference_calls"].to_numpy()
        )
        item = {
            "scoreable_episodes": int(scoreable.sum()),
            "scoreable_successes": int((scoreable & evaluated["success"]).sum()),
            "scoreable_failures": int((scoreable & evaluated["failure"]).sum()),
            "unscoreable_successes_before_query_12": int(
                (~scoreable & evaluated["success"]).sum()
            ),
            "unscoreable_failures_before_query_12": int(
                (~scoreable & evaluated["failure"]).sum()
            ),
            "success_query_probability": quantiles(success_rows),
            "failure_query_probability": quantiles(failure_rows),
            "success_episode_max_probability": quantiles(success_episode),
            "failure_episode_max_probability": quantiles(failure_episode),
            "endpoint_failure_ranking_at_first_eligible_query": first_metrics,
            "endpoint_failure_ranking_by_episode_max": max_metrics,
            "endpoint_duration_ranking_control": duration_metrics,
            "within_task_first_query_ranking": stratified_rank_metrics(
                scored, "probability_first"
            ),
            "within_task_episode_max_ranking": stratified_rank_metrics(
                scored, "probability_max"
            ),
            "within_task_duration_ranking_control": stratified_rank_metrics(
                scored, "inference_calls"
            ),
        }
        calibrator_fixed_rows: list[dict[str, Any]] = []
        for query_index, query_group in query.groupby("query", sort=True):
            pooled = binary_rank_metrics(
                query_group["failure"].to_numpy(), query_group[column].to_numpy()
            )
            stratified = stratified_rank_metrics(query_group, column)
            fixed_row = {
                "calibrator": calibrator,
                "query": int(query_index),
                "surviving_episodes": len(query_group),
                "surviving_successes": int(query_group["success"].sum()),
                "surviving_failures": int(query_group["failure"].sum()),
                "pooled_roc_auc": pooled["roc_auc"],
                "pooled_average_precision": pooled["average_precision"],
                **stratified,
            }
            fixed_query_rows.append(fixed_row)
            calibrator_fixed_rows.append(fixed_row)
        item["fixed_query_key_points"] = {
            str(row["query"]): row
            for row in calibrator_fixed_rows
            if row["query"] in {12, 16, 20, 24, 28, 30, 32, 36, 40, 44, 48}
        }
        for threshold in map(float, config["reported_thresholds"]):
            alarm = scored["probability_max"] >= threshold
            failure = scored["failure"].to_numpy(bool)
            alarm_values = alarm.to_numpy(bool)
            tp = int(np.sum(alarm_values & failure))
            fp = int(np.sum(alarm_values & ~failure))
            fn = int(np.sum(~alarm_values & failure))
            tn = int(np.sum(~alarm_values & ~failure))
            row = {
                "calibrator": calibrator,
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "failure_recall_scoreable": tp / max(tp + fn, 1),
                "success_alarm_rate_scoreable": fp / max(fp + tn, 1),
                "failure_recall_all": tp / max(int(outcomes["failure"].sum()), 1),
                "success_alarm_rate_all": fp / max(int(outcomes["success"].sum()), 1),
                "endpoint_precision": tp / max(tp + fp, 1),
            }
            unseen = scored[scored["task"] != config["source_overlap_task"]]
            unseen_alarm = unseen["probability_max"].to_numpy() >= threshold
            unseen_failure = unseen["failure"].to_numpy(bool)
            unseen_tp = int(np.sum(unseen_alarm & unseen_failure))
            unseen_fp = int(np.sum(unseen_alarm & ~unseen_failure))
            row.update(
                {
                    "unseen_task_failures": int(unseen_failure.sum()),
                    "unseen_task_successes": int((~unseen_failure).sum()),
                    "unseen_task_tp": unseen_tp,
                    "unseen_task_fp": unseen_fp,
                    "unseen_task_failure_recall": unseen_tp
                    / max(int(unseen_failure.sum()), 1),
                    "unseen_task_success_alarm_rate": unseen_fp
                    / max(int((~unseen_failure).sum()), 1),
                    "unseen_task_endpoint_precision": unseen_tp
                    / max(unseen_tp + unseen_fp, 1),
                }
            )
            alarm_queries = query.loc[
                query[column] >= threshold, ["task", "episode", "query"]
            ]
            first_alarm = (
                alarm_queries.groupby(["task", "episode"], as_index=False)["query"]
                .min()
                .merge(
                    outcomes[
                        [
                            "task",
                            "episode",
                            "failure",
                            "inference_calls",
                            "horizon_queries",
                        ]
                    ],
                    on=["task", "episode"],
                    validate="one_to_one",
                )
            )
            detected_failures = first_alarm[first_alarm["failure"]].copy()
            if len(detected_failures):
                detected_failures["horizon_phase"] = detected_failures[
                    "query"
                ] / np.maximum(detected_failures["horizon_queries"] - 1, 1)
                detected_failures["lead_to_episode_end"] = (
                    detected_failures["inference_calls"]
                    - 1
                    - detected_failures["query"]
                )
                task_counts = detected_failures["task"].value_counts()
                row.update(
                    {
                        "detected_failure_first_alarm_query_median": float(
                            detected_failures["query"].median()
                        ),
                        "detected_failure_first_alarm_horizon_phase_median": float(
                            detected_failures["horizon_phase"].median()
                        ),
                        "detected_failure_lead_to_episode_end_median": float(
                            detected_failures["lead_to_episode_end"].median()
                        ),
                        "detected_failure_tasks": int(len(task_counts)),
                        "largest_task_share_of_detected_failures": float(
                            task_counts.iloc[0] / len(detected_failures)
                        ),
                    }
                )
            else:
                row.update(
                    {
                        "detected_failure_first_alarm_query_median": np.nan,
                        "detected_failure_first_alarm_horizon_phase_median": np.nan,
                        "detected_failure_lead_to_episode_end_median": np.nan,
                        "detected_failure_tasks": 0,
                        "largest_task_share_of_detected_failures": np.nan,
                    }
                )
            clock_match = matched_clock(clock, row["success_alarm_rate_all"])
            row.update(
                {
                    "matched_clock_horizon_phase": clock_match["horizon_phase"],
                    "matched_clock_failure_recall": clock_match["failure_recall"],
                    "matched_clock_success_false_alarm_rate": clock_match[
                        "success_false_alarm_rate"
                    ],
                    "matched_clock_endpoint_precision": clock_match[
                        "endpoint_precision"
                    ],
                }
            )
            threshold_rows.append(row)
        item["thresholds"] = [
            row for row in threshold_rows if row["calibrator"] == calibrator
        ]
        summary["calibrators"][calibrator] = item

        for outcome_name, outcome_value in (("success", True), ("failure", False)):
            values = query.loc[query["success"] == outcome_value, column]
            distribution_rows.append(
                {
                    "calibrator": calibrator,
                    "endpoint_outcome": outcome_name,
                    **quantiles(values),
                }
            )
        for (phase_bin, success), group in query.groupby(
            ["phase_bin", "success"], observed=True
        ):
            values = group[column]
            phase_rows.append(
                {
                    "calibrator": calibrator,
                    "phase_bin": str(phase_bin),
                    "endpoint_outcome": "success" if success else "failure",
                    **quantiles(values),
                }
            )
        for task, task_group in scored.groupby("task", sort=True):
            y = task_group["failure"].to_numpy(bool)
            task_metric = binary_rank_metrics(y, task_group["probability_max"].to_numpy())
            first_metric = binary_rank_metrics(
                y, task_group["probability_first"].to_numpy()
            )
            duration_metric = binary_rank_metrics(
                y, task_group["inference_calls"].to_numpy()
            )
            task_rows.append(
                {
                    "calibrator": calibrator,
                    "task": task,
                    "scoreable_episodes": len(task_group),
                    "scoreable_successes": int((~y).sum()),
                    "scoreable_failures": int(y.sum()),
                    "episode_max_roc_auc": task_metric["roc_auc"],
                    "episode_max_average_precision": task_metric[
                        "average_precision"
                    ],
                    "first_query_roc_auc": first_metric["roc_auc"],
                    "duration_roc_auc": duration_metric["roc_auc"],
                    "success_max_probability_mean": float(
                        task_group.loc[~y, "probability_max"].mean()
                    ),
                    "failure_max_probability_mean": float(
                        task_group.loc[y, "probability_max"].mean()
                    )
                    if np.any(y)
                    else np.nan,
                }
            )
        frequency = (
            query.loc[query["success"], column]
            .value_counts(dropna=False)
            .sort_index()
        )
        for value, count in frequency.items():
            probability_frequency_rows.append(
                {
                    "calibrator": calibrator,
                    "probability": float(value),
                    "successful_query_rows": int(count),
                }
            )

    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    episode_output.to_csv(table_dir / "episode_outcome_audit.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(
        table_dir / "endpoint_threshold_audit.csv", index=False
    )
    pd.DataFrame(phase_rows).to_csv(
        table_dir / "endpoint_phase_probability.csv", index=False
    )
    pd.DataFrame(distribution_rows).to_csv(
        table_dir / "endpoint_probability_distribution.csv", index=False
    )
    pd.DataFrame(task_rows).to_csv(
        table_dir / "endpoint_task_ranking.csv", index=False
    )
    pd.DataFrame(probability_frequency_rows).to_csv(
        table_dir / "successful_probability_values.csv", index=False
    )
    pd.DataFrame(fixed_query_rows).to_csv(
        table_dir / "endpoint_fixed_query_ranking.csv", index=False
    )
    clock.to_csv(table_dir / "endpoint_clock_baselines.csv", index=False)
    summary["clock_baselines"] = {
        "rows": len(clock),
        "warning": "query clock uses no MoE and exploits endpoint duration; it is a confound control, not an online failure oracle",
    }
    summary["source_overlap_task"] = config["source_overlap_task"]
    return summary


def plot_audit(summary: dict[str, Any], phase: pd.DataFrame, output: Path) -> None:
    colors = {"A_dense": "#237a57", "B_proxy": "#b94b3c"}
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    x = np.arange(len(CALIBRATORS))
    median = [
        summary["calibrators"][name]["success_query_probability"]["median"]
        for name in CALIBRATORS
    ]
    p90 = [
        summary["calibrators"][name]["success_query_probability"]["p90"]
        for name in CALIBRATORS
    ]
    axes[0].bar(x - 0.17, median, 0.34, color="#3f7d5d", label="median")
    axes[0].bar(x + 0.17, p90, 0.34, color="#d19a43", label="p90")
    axes[0].set_xticks(x, ["A dense", "B proxy"])
    axes[0].set_ylabel("Predicted near-Trap probability")
    axes[0].set_title("Successful query rows")
    axes[0].legend(frameon=False)

    order = ["eligible_to_40", "40_60", "60_80", "80_100"]
    for name in CALIBRATORS:
        selected = phase[
            (phase["calibrator"] == name)
            & (phase["endpoint_outcome"] == "success")
        ].set_index("phase_bin").reindex(order)
        axes[1].plot(
            np.arange(len(order)),
            selected["mean"],
            marker="o",
            color=colors[name],
            label=name,
        )
    axes[1].set_xticks(np.arange(len(order)), ["≤40%", "40–60%", "60–80%", "80–100%"])
    axes[1].set_ylabel("Mean probability")
    axes[1].set_title("Successful trajectories by phase")
    axes[1].legend(frameon=False)

    width = 0.19
    first_auc = [
        summary["calibrators"][name][
            "endpoint_failure_ranking_at_first_eligible_query"
        ]["roc_auc"]
        for name in CALIBRATORS
    ]
    max_auc = [
        summary["calibrators"][name]["endpoint_failure_ranking_by_episode_max"][
            "roc_auc"
        ]
        for name in CALIBRATORS
    ]
    duration_auc = [
        summary["calibrators"][name]["endpoint_duration_ranking_control"][
            "roc_auc"
        ]
        for name in CALIBRATORS
    ]
    within_task_first_auc = [
        summary["calibrators"][name]["within_task_first_query_ranking"][
            "pair_weighted_within_task_roc_auc"
        ]
        for name in CALIBRATORS
    ]
    axes[2].bar(
        x - 1.5 * width, first_auc, width, color="#3f7d5d", label="first q pooled"
    )
    axes[2].bar(
        x - 0.5 * width,
        within_task_first_auc,
        width,
        color="#4b78a8",
        label="first q within-task",
    )
    axes[2].bar(
        x + 0.5 * width, max_auc, width, color="#d19a43", label="episode max"
    )
    axes[2].bar(
        x + 1.5 * width, duration_auc, width, color="#777777", label="duration"
    )
    axes[2].set_xticks(x, ["A dense", "B proxy"])
    axes[2].set_ylim(0.45, 1.0)
    axes[2].set_ylabel("Endpoint failure AUROC")
    axes[2].set_title("Descriptive endpoint ranking")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    figure.tight_layout()
    path = output / "figures/cache_new_probability_audit.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_report(summary: dict[str, Any]) -> str:
    a = summary["calibrators"]["A_dense"]
    b = summary["calibrators"]["B_proxy"]
    a25 = next(row for row in a["thresholds"] if row["threshold"] == 0.25)
    b25 = next(row for row in b["thresholds"] if row["threshold"] == 0.25)
    b50 = next(row for row in b["thresholds"] if row["threshold"] == 0.5)
    return f"""# cache_new 40-task Trap 概率压力测试

## 核心结果

冻结的 A/B 概率表已直接应用于 `cache_new` 的 40 个完整任务、{summary['total_episodes']:,} 条轨迹和 {summary['total_route_queries']:,} 次推理。预测阶段只读取 HB MoE routing；逐 query 概率写盘并记录 SHA256 后才加载最终 success/failure。

该语料没有统一的逐 query Trap onset，因此本实验只能回答成功轨迹沿途会得到什么数值，以及概率与最终 outcome 是否相关。它不能评价原定义的 `P(未来两 query 内 Trap onset)` 是否校准。

| source 概率表 | 可评分成功 / 全部成功 | 成功 query 中位 / p90 / max | 成功 episode-max 中位 / p90 | q12 pooled / within-task AUROC | episode-max AUROC | 75% 报警 |
|---|---:|---:|---:|---:|---:|---:|
| A dense | {a['scoreable_successes']:,}/{summary['total_successes']:,} | {a['success_query_probability']['median']:.2%} / {a['success_query_probability']['p90']:.2%} / {a['success_query_probability']['max']:.2%} | {a['success_episode_max_probability']['median']:.2%} / {a['success_episode_max_probability']['p90']:.2%} | {a['endpoint_failure_ranking_at_first_eligible_query']['roc_auc']:.3f} / {a['within_task_first_query_ranking']['pair_weighted_within_task_roc_auc']:.3f} | {a['endpoint_failure_ranking_by_episode_max']['roc_auc']:.3f} | 0 |
| B proxy | {b['scoreable_successes']:,}/{summary['total_successes']:,} | {b['success_query_probability']['median']:.2%} / {b['success_query_probability']['p90']:.2%} / {b['success_query_probability']['max']:.2%} | {b['success_episode_max_probability']['median']:.2%} / {b['success_episode_max_probability']['p90']:.2%} | {b['endpoint_failure_ranking_at_first_eligible_query']['roc_auc']:.3f} / {b['within_task_first_query_ranking']['pair_weighted_within_task_roc_auc']:.3f} | {b['endpoint_failure_ranking_by_episode_max']['roc_auc']:.3f} | 0 |

方法从 q12 才能输出，因为 q4--q11 用于 episode 自身基线。共有 {a['unscoreable_successes_before_query_12']:,} 条成功轨迹在 q12 前已经完成，不能被当前方法评分。这是部署延迟，不应把它们计作“低风险预测正确”。

## 25% 阈值

- A dense：scoreable failure recall {a25['failure_recall_scoreable']:.2%}，scoreable success alarm rate {a25['success_alarm_rate_scoreable']:.2%}，endpoint precision {a25['endpoint_precision']:.2%}。
- B proxy：scoreable failure recall {b25['failure_recall_scoreable']:.2%}，scoreable success alarm rate {b25['success_alarm_rate_scoreable']:.2%}，endpoint precision {b25['endpoint_precision']:.2%}。

B proxy 的 50% 状态看起来更保守：命中 {b50['tp']}/{summary['total_failures']} 个最终失败，并在全部成功中报警 {b50['fp']}/{summary['total_successes']}，endpoint precision 为 {b50['endpoint_precision']:.2%}。排除与 A/B source 相同的 moka-pot 任务后仍为 recall {b50['unseen_task_failure_recall']:.2%}、scoreable-success alarm {b50['unseen_task_success_alarm_rate']:.2%}、precision {b50['unseen_task_endpoint_precision']:.2%}。首次命中的中位位置已经是 horizon 的 {b50['detected_failure_first_alarm_horizon_phase_median']:.1%}，只覆盖 {b50['detected_failure_tasks']} 个任务，且最大单任务贡献 {b50['largest_task_share_of_detected_failures']:.1%} 的命中。相同或更低成功误报率的固定 phase-{b50['matched_clock_horizon_phase']:.2f} 时钟可覆盖 {b50['matched_clock_failure_recall']:.2%} 的失败，而该 MoE 阈值只有 {b50['failure_recall_all']:.2%}；时钟还略早于 MoE 的中位报警位置。因此不能把它当成已经成立的通用 detector。

这些是与最终失败的描述性关联，不是 Trap-onset precision。75% 仍然必然零报警，因为冻结 A/B onset-hazard 表本身的最大值只有 35.8% 和 51.3%。

## 解释

pooled q12 AUC 看似有 {a['endpoint_failure_ranking_at_first_eligible_query']['roc_auc']:.3f}/{b['endpoint_failure_ranking_at_first_eligible_query']['roc_auc']:.3f}，但在每个任务内部配对后只剩 {a['within_task_first_query_ranking']['pair_weighted_within_task_roc_auc']:.3f}/{b['within_task_first_query_ranking']['pair_weighted_within_task_roc_auc']:.3f}。这说明早期 pooled 分离相当一部分来自任务构成。episode-max AUC 又因失败轨迹更长、碰到高状态的机会更多而升高；最终 duration 自身的 pooled AUROC 为 {a['endpoint_duration_ranking_control']['roc_auc']:.3f}，同任务配对后接近 {a['within_task_duration_ranking_control']['pair_weighted_within_task_roc_auc']:.3f}。所以不能用 endpoint outcome 把这个概率重新命名为“失败概率”。

最准确的结论仍然是：A/B 单任务 onset 频率表不能直接作为 40-task 通用概率标尺；`cache_new` 更适合检查成功轨迹的概率背景和分布漂移。真正验证近两步报警仍需要为新任务构造独立的 query-level Trap onset。
"""


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    probability_config_path = resolve_workspace(config["probability_config"])
    probability_config = json.loads(
        probability_config_path.read_text(encoding="utf-8")
    )
    cache_root = resolve_workspace(config["cache_root"])
    runs = discover_runs(cache_root, config["run_id"])
    if args.limit_tasks is not None:
        runs = runs[: args.limit_tasks]
    elif len(runs) != int(config["expected_complete_tasks"]):
        raise RuntimeError(f"expected {config['expected_complete_tasks']} tasks, got {len(runs)}")
    calibrators = load_calibrators(config, probability_config)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "route_only_tasks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline = tuple(map(int, config["self_reference_queries"]))

    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                extract_task_probabilities,
                task,
                str(run),
                str(cache_dir),
                calibrators,
                probability_config,
                int(config["minimum_query"]),
                baseline,
                args.rebuild,
            ): task
            for task, run in runs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            print(
                f"[route-only {completed}/{len(futures)}] {record['task']}: "
                f"rows={record['route_rows']}, eligible={record['eligible_rows']}, "
                f"reused={record['cache_reused']}",
                flush=True,
            )

    predictions = load_prediction_frame(records).sort_values(
        ["task", "episode", "query"], kind="stable"
    )
    label_free_path = output / "cache_new_probabilities_label_free.csv.gz"
    predictions.to_csv(
        label_free_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    forbidden = {"success", "failure", "outcome", "label", "onset"}
    if not forbidden.isdisjoint(predictions.columns):
        raise AssertionError("label leaked into route-only prediction table")
    prediction_sha = sha256_file(label_free_path)
    route_manifest = {
        "schema": SCHEMA,
        "phase": "route_only_complete_before_outcome_load",
        "training": False,
        "target_labels_loaded": False,
        "tasks": len(runs),
        "episodes": int(sum(record["episodes"] for record in records)),
        "route_queries": int(sum(record["route_rows"] for record in records)),
        "eligible_query_rows": len(predictions),
        "prediction_file": str(label_free_path),
        "prediction_sha256_before_outcome_load": prediction_sha,
        "source_calibrators": {
            name: {
                "direction": item["direction"],
                "route_artifact_sha256": item["artifact_sha256"],
            }
            for name, item in calibrators.items()
        },
        "task_caches": sorted(records, key=lambda item: item["task"]),
    }
    write_json(output / "route_only_manifest.json", route_manifest)

    # Phase 2 begins only after the complete target prediction file is on disk.
    outcomes = load_outcomes(runs)
    if len(outcomes) != int(sum(record["episodes"] for record in records)):
        raise ValueError("route/outcome episode count mismatch")
    summary = evaluate(predictions, outcomes, config, output)
    summary.update(
        {
            "schema": SCHEMA,
            "status": "complete",
            "training": False,
            "gradient_optimization": False,
            "learned_feature_weights": False,
            "source_labels_used_for_frozen_calibration": True,
            "cache_new_outcomes_used_by_probability": False,
            "target_probabilities_written_before_outcomes": True,
            "task_identity_used_by_probability": False,
            "runtime_action_values_used": False,
            "runtime_physical_state_used": False,
            "runtime_outcome_used": False,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "probability_config_sha256": sha256_file(probability_config_path),
            "route_only_manifest": route_manifest,
            "prediction_file_sha256": prediction_sha,
            "complete_tasks": len(runs),
            "run_id": config["run_id"],
        }
    )
    phase = pd.read_csv(output / "tables/endpoint_phase_probability.csv")
    plot_audit(summary, phase, output)
    write_json(output / "summary.json", summary)
    (output / "REPORT_ZH.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(plain(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
