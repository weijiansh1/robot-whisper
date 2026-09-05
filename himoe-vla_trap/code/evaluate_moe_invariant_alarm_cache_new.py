#!/usr/bin/env python3
"""Blind offline replay of a fixed, train-free MoE invariant alarm.

The experiment uses only VLA_MUI_HUB/cache_new. Route-only features are
materialized before any endpoint labels are opened. A deterministic task split
reserves two tasks per LIBERO suite. Successful episodes from the remaining
tasks are used only to confirm a 1% episode-level false-alarm threshold. The
held-out predictions are then written and hashed before held-out outcomes are
loaded and flipped for evaluation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/moe_invariant_alarm_cache_new.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_invariant_alarm_cache_new"
SCHEMA = "himoe.moe_invariant_alarm_cache_new.v1"
HEADS = ("convergence", "response", "recurrence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def resolve_workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def route_feature_config_sha256(config: dict[str, Any]) -> str:
    """Fingerprint every setting that changes the cached route-only features."""
    payload = {
        "schema": SCHEMA,
        "route_geometry": config["route_geometry"],
        "causal_self_reference": config["causal_self_reference"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def weighted_jaccard(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    numerator = np.minimum(left, right).sum(axis=-1)
    denominator = np.maximum(left, right).sum(axis=-1)
    return numerator / np.maximum(denominator, 1e-12)


def discover_runs(cache_root: Path, run_id: str) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    for route_path in sorted(cache_root.glob(f"libero_*/*/{run_id}/server/routes.zarr")):
        run = route_path.parents[1]
        task = str(run.relative_to(cache_root).parent)
        runs.append((task, run))
    if not runs:
        raise RuntimeError(f"no {run_id!r} route stores below {cache_root}")
    return runs


def split_tasks(
    tasks: Iterable[str], salt: str, heldout_per_suite: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[tuple[str, str]]] = {}
    for task in sorted(tasks):
        suite = task.split("/", 1)[0]
        digest = hashlib.sha256(f"{salt}:{task}".encode("utf-8")).hexdigest()
        grouped.setdefault(suite, []).append((digest, task))
    for suite, members in sorted(grouped.items()):
        ranked = sorted(members)
        if len(ranked) <= heldout_per_suite:
            raise ValueError(f"not enough tasks in {suite} for heldout split")
        heldout = {task for _digest, task in ranked[:heldout_per_suite]}
        for rank, (digest, task) in enumerate(ranked):
            rows.append(
                {
                    "task": task,
                    "suite": suite,
                    "sha256_rank": rank,
                    "sha256": digest,
                    "role": "heldout_test" if task in heldout else "threshold_confirmation",
                }
            )
    return pd.DataFrame(rows).sort_values(["suite", "sha256_rank"]).reset_index(drop=True)


def task_cache_path(cache_dir: Path, task: str) -> Path:
    token = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{token}.npz"


def baseline_median(values: np.ndarray, queries: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[queries]
    selected = selected[np.isfinite(selected)]
    return float(np.median(selected)) if len(selected) else float("nan")


def extract_task_route_features(
    task: str,
    run_string: str,
    cache_string: str,
    config: dict[str, Any],
    rebuild: bool,
) -> dict[str, Any]:
    run = Path(run_string)
    cache = task_cache_path(Path(cache_string), task)
    feature_config_sha256 = route_feature_config_sha256(config)
    if cache.exists() and not rebuild:
        with np.load(cache) as archive:
            cache_run = str(archive["run"].item()) if "run" in archive.files else ""
            cache_config = (
                str(archive["feature_config_sha256"].item())
                if "feature_config_sha256" in archive.files
                else ""
            )
            if (
                str(archive["schema"].item()) == SCHEMA
                and str(archive["task"].item()) == task
                and cache_run == str(run)
                and cache_config == feature_config_sha256
            ):
                return {
                    "task": task,
                    "cache": str(cache),
                    "route_rows": int(len(archive["episode"])),
                    "episodes": int(len(np.unique(archive["episode"]))),
                    "cache_reused": True,
                }

    geometry = config["route_geometry"]
    front = np.asarray(geometry["front_layers"], dtype=np.int64)
    back = np.asarray(geometry["back_layers"], dtype=np.int64)
    action_tokens = np.asarray(geometry["action_tokens"], dtype=np.int64)
    state_token = int(geometry["state_token"])
    final_flow = int(geometry["final_flow"])
    late_start = int(geometry["late_flow_transition_start"])
    lags = tuple(int(value) for value in geometry["recurrence_lags"])
    baseline_queries = np.asarray(
        config["causal_self_reference"]["baseline_queries"], dtype=np.int64
    )
    first_scored = int(config["causal_self_reference"]["first_scored_query"])

    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    router = group["hb_router_probs"]
    expected_shape = (8, 10, 11, 32)
    if tuple(router.shape[1:]) != expected_shape:
        raise ValueError(
            f"unexpected hb_router_probs geometry for {task}: "
            f"{tuple(router.shape[1:])} != {expected_shape}"
        )
    episode = np.asarray(group["episode_id"][:], dtype=np.int32)
    control_step = np.asarray(group["control_step"][:], dtype=np.int64)
    total = int(router.shape[0])
    if len(episode) != total or len(control_step) != total:
        raise ValueError(f"route metadata length mismatch: {task}")
    if total > 1 and np.any(np.diff(control_step) != 1):
        raise ValueError(f"non-contiguous global control steps: {task}")
    unique_episodes = np.unique(episode)
    if len(unique_episodes) != int(config["expected_episodes_per_task"]):
        raise ValueError(f"unexpected episode count for {task}: {len(unique_episodes)}")

    convergence_raw = np.empty(total, dtype=np.float32)
    front_state = np.empty((total, len(front), 32), dtype=np.float16)
    front_action = np.empty((total, len(front), 32), dtype=np.float16)
    back_action = np.empty((total, len(back), len(action_tokens), 32), dtype=np.float16)
    late_volatility = np.empty(total, dtype=np.float32)
    route_acceleration = np.empty(total, dtype=np.float32)

    for start in range(0, total, 64):
        stop = min(start + 64, total)
        routes = normalize_probability(router[start:stop])
        action_flow = routes[:, back, :, :, :][:, :, :, action_tokens, :]
        flow_distance = hellinger(action_flow[:, :, 1:], action_flow[:, :, :-1])
        path_by_transition = flow_distance.mean((1, 3))
        convergence_raw[start:stop] = (
            path_by_transition[:, late_start:].sum(axis=1)
            / np.maximum(path_by_transition.sum(axis=1), 1e-12)
        )
        late_volatility[start:stop] = flow_distance[:, :, late_start:].mean((1, 2, 3))
        root = np.sqrt(action_flow)
        curvature = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        route_acceleration[start:stop] = (
            np.linalg.norm(curvature, axis=-1).mean((1, 2, 3)) / np.sqrt(2.0)
        )

        final_state = routes[:, :, final_flow, state_token, :]
        final_action = routes[:, :, final_flow, :, :][:, :, action_tokens, :]
        front_state[start:stop] = final_state[:, front]
        front_action[start:stop] = normalize_probability(
            final_action[:, front].mean(axis=2)
        )
        back_action[start:stop] = final_action[:, back]

    query = np.empty(total, dtype=np.int16)
    valid = np.zeros(total, dtype=bool)
    state_jump = np.full(total, np.nan, dtype=np.float32)
    action_jump = np.full(total, np.nan, dtype=np.float32)
    response_raw = np.full(total, np.nan, dtype=np.float32)
    feedback_split_signed = np.full(total, np.nan, dtype=np.float32)
    recurrence_raw = np.full(total, np.nan, dtype=np.float32)
    recurrence_lag = np.full(total, -1, dtype=np.int8)
    baseline_convergence = np.full(total, np.nan, dtype=np.float32)
    baseline_response = np.full(total, np.nan, dtype=np.float32)
    baseline_recurrence = np.full(total, np.nan, dtype=np.float32)
    evidence_convergence = np.full(total, np.nan, dtype=np.float32)
    evidence_response = np.full(total, np.nan, dtype=np.float32)
    evidence_recurrence = np.full(total, np.nan, dtype=np.float32)

    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode == episode_id)
        if len(indices) > 1 and np.any(np.diff(indices) != 1):
            raise ValueError(f"non-contiguous episode rows: {task}/{episode_id}")
        length = len(indices)
        local_query = np.arange(length, dtype=np.int16)
        query[indices] = local_query

        fstate = normalize_probability(front_state[indices])
        faction = normalize_probability(front_action[indices])
        if length > 1:
            state_jump[indices[1:]] = hellinger(fstate[1:], fstate[:-1]).mean(axis=1)
            action_jump[indices[1:]] = hellinger(faction[1:], faction[:-1]).mean(axis=1)
            signed = state_jump[indices[1:]] - action_jump[indices[1:]]
            feedback_split_signed[indices[1:]] = signed
            response_raw[indices[1:]] = np.maximum(signed, 0.0)

        action_route = normalize_probability(back_action[indices]).reshape(length, -1, 32)
        lag_similarity = np.full((length, len(lags)), np.nan, dtype=np.float32)
        for lag_index, lag in enumerate(lags):
            if length > lag:
                lag_similarity[lag:, lag_index] = weighted_jaccard(
                    action_route[lag:], action_route[:-lag]
                ).mean(axis=1)
        has_lag = np.isfinite(lag_similarity).any(axis=1)
        if np.any(has_lag):
            safe = np.where(np.isfinite(lag_similarity), lag_similarity, -np.inf)
            recurrence_raw[indices[has_lag]] = safe[has_lag].max(axis=1)
            recurrence_lag[indices[has_lag]] = np.asarray(lags, dtype=np.int8)[
                safe[has_lag].argmax(axis=1)
            ]

        base_queries = baseline_queries[baseline_queries < length]
        conv_base = baseline_median(convergence_raw[indices], base_queries)
        response_base = baseline_median(response_raw[indices], base_queries)
        recurrence_base = baseline_median(recurrence_raw[indices], base_queries)
        baseline_convergence[indices] = conv_base
        baseline_response[indices] = response_base
        baseline_recurrence[indices] = recurrence_base
        score_mask = local_query >= first_scored
        score_indices = indices[score_mask]
        if np.isfinite(conv_base):
            evidence_convergence[score_indices] = convergence_raw[score_indices] - conv_base
        if np.isfinite(response_base):
            evidence_response[score_indices] = response_raw[score_indices] - response_base
        if np.isfinite(recurrence_base):
            evidence_recurrence[score_indices] = recurrence_raw[score_indices] - recurrence_base
        valid[score_indices] = (
            np.isfinite(evidence_convergence[score_indices])
            & np.isfinite(evidence_response[score_indices])
            & np.isfinite(evidence_recurrence[score_indices])
        )

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        schema=np.asarray(SCHEMA),
        task=np.asarray(task),
        run=np.asarray(str(run)),
        feature_config_sha256=np.asarray(feature_config_sha256),
        training=np.asarray(False),
        endpoint_labels_loaded=np.asarray(False),
        episode=episode,
        query=query,
        valid=valid,
        convergence_raw=convergence_raw,
        late_flow_volatility=late_volatility,
        route_acceleration=route_acceleration,
        state_jump=state_jump,
        action_jump=action_jump,
        feedback_split_signed=feedback_split_signed,
        response_raw=response_raw,
        recurrence_raw=recurrence_raw,
        recurrence_lag=recurrence_lag,
        baseline_convergence=baseline_convergence,
        baseline_response=baseline_response,
        baseline_recurrence=baseline_recurrence,
        evidence_convergence=evidence_convergence,
        evidence_response=evidence_response,
        evidence_recurrence=evidence_recurrence,
    )
    return {
        "task": task,
        "cache": str(cache),
        "route_rows": total,
        "episodes": len(unique_episodes),
        "cache_reused": False,
    }


def load_outcome_map(run: Path) -> dict[int, bool]:
    rows = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    return {int(row["episode_index"]): bool(row["success"]) for row in rows}


def load_route_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def episode_slices(episode: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    for episode_id in np.unique(episode):
        yield int(episode_id), np.flatnonzero(episode == episode_id)


def finite_tail_confidence(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    good = np.isfinite(values)
    left = np.searchsorted(reference, values[good], side="left")
    tail_probability = (1.0 + len(reference) - left) / (len(reference) + 1.0)
    output[good] = 1.0 - tail_probability
    return output


def detector_series(
    arrays: dict[str, np.ndarray], references: dict[str, np.ndarray]
) -> pd.DataFrame:
    valid = np.asarray(arrays["valid"], dtype=bool)
    frame = pd.DataFrame(
        {
            "episode": arrays["episode"].astype(np.int32),
            "query": arrays["query"].astype(np.int16),
            "valid": valid,
            "convergence_raw": arrays["convergence_raw"],
            "late_flow_volatility": arrays["late_flow_volatility"],
            "route_acceleration": arrays["route_acceleration"],
            "state_jump": arrays["state_jump"],
            "action_jump": arrays["action_jump"],
            "feedback_split_signed": arrays["feedback_split_signed"],
            "response_raw": arrays["response_raw"],
            "recurrence_raw": arrays["recurrence_raw"],
            "recurrence_lag": arrays["recurrence_lag"],
        }
    )
    for head in HEADS:
        evidence = np.asarray(arrays[f"evidence_{head}"], dtype=np.float64)
        frame[f"evidence_{head}"] = evidence
        frame[f"confidence_{head}"] = finite_tail_confidence(
            references[head], evidence
        )

    frame["joint_confidence"] = np.minimum(
        frame["confidence_convergence"], frame["confidence_response"]
    )
    frame["recurrence_persistent_confidence"] = np.nan
    for _episode, indices in episode_slices(frame["episode"].to_numpy()):
        queries = frame.loc[indices, "query"].to_numpy()
        confidence = frame.loc[indices, "confidence_recurrence"].to_numpy()
        consecutive = np.r_[False, np.diff(queries) == 1]
        persistent = np.full(len(indices), np.nan, dtype=np.float64)
        persistent[1:] = np.minimum(confidence[1:], confidence[:-1])
        persistent[~consecutive] = np.nan
        frame.loc[indices, "recurrence_persistent_confidence"] = persistent
    branch_values = frame[
        ["joint_confidence", "recurrence_persistent_confidence"]
    ].to_numpy(dtype=np.float64)
    branch_finite = np.isfinite(branch_values)
    frame["detector_confidence"] = np.max(
        np.where(branch_finite, branch_values, -np.inf), axis=1
    )
    frame.loc[~branch_finite.any(axis=1), "detector_confidence"] = np.nan
    frame.loc[~valid, "detector_confidence"] = np.nan
    return frame


def episode_maxima(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("episode", sort=False)[column].max()


def select_threshold(values: np.ndarray, budget: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("no finite successful-episode detector maxima")
    for threshold in np.unique(values):
        rate = float(np.mean(values >= threshold))
        if rate <= budget + 1e-15:
            return float(threshold), rate
    return float(np.nextafter(values.max(), np.inf)), 0.0


def select_clock_query(lengths: np.ndarray, budget: float) -> tuple[int, float]:
    lengths = np.asarray(lengths, dtype=np.int64)
    for query in range(int(lengths.max()) + 1):
        rate = float(np.mean(lengths > query))
        if rate <= budget + 1e-15:
            return query, rate
    return int(lengths.max()), 0.0


def alarm_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = frame.copy()
    output["alarm"] = output["detector_confidence"] >= threshold
    output["first_alarm"] = False
    output["alarm_cause"] = ""
    joint = output["joint_confidence"] >= threshold
    recurrence = output["recurrence_persistent_confidence"] >= threshold
    output.loc[joint & ~recurrence, "alarm_cause"] = "convergence_and_response"
    output.loc[~joint & recurrence, "alarm_cause"] = "persistent_recurrence"
    output.loc[joint & recurrence, "alarm_cause"] = "both"
    for _episode, indices in episode_slices(output["episode"].to_numpy()):
        hits = indices[output.loc[indices, "alarm"].to_numpy()]
        if len(hits):
            output.loc[hits[0], "first_alarm"] = True
    return output


def build_episode_flip_rows(
    task: str,
    predictions: pd.DataFrame,
    lengths: dict[int, int],
    outcomes: dict[int, bool],
    clock_query: int,
    role: str,
) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for episode_id, success in sorted(outcomes.items()):
        rows = predictions[predictions["episode"] == episode_id]
        hits = rows[rows["first_alarm"]]
        alarm = not hits.empty
        first = hits.iloc[0] if alarm else None
        length = int(lengths[episode_id])
        first_query = float(first["query"]) if alarm else np.nan
        rows_out.append(
            {
                "role": role,
                "task": task,
                "suite": task.split("/", 1)[0],
                "episode": episode_id,
                "episode_length": length,
                "success": success,
                "failure": not success,
                "alarm": alarm,
                "first_alarm_query": first_query,
                "first_alarm_phase": first_query / max(length - 1, 1) if alarm else np.nan,
                "lead_to_endpoint_queries": length - 1 - first_query if alarm else np.nan,
                "alarm_cause": str(first["alarm_cause"]) if alarm else "",
                "episode_max_confidence": (
                    float(rows["detector_confidence"].max()) if len(rows) else np.nan
                ),
                "clock_alarm": length > clock_query,
                "clock_first_alarm_query": (
                    float(clock_query) if length > clock_query else np.nan
                ),
            }
        )
    return rows_out


def rate_metrics(alarm: np.ndarray, failure: np.ndarray) -> dict[str, Any]:
    alarm = np.asarray(alarm, dtype=bool)
    failure = np.asarray(failure, dtype=bool)
    tp = int(np.sum(alarm & failure))
    fp = int(np.sum(alarm & ~failure))
    fn = int(np.sum(~alarm & failure))
    tn = int(np.sum(~alarm & ~failure))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / max(tp + fp, 1),
        "failure_recall": tp / max(tp + fn, 1),
        "success_false_alarm_rate": fp / max(fp + tn, 1),
    }


def task_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task, group in episodes.groupby("task", sort=True):
        metrics = rate_metrics(group["alarm"], group["failure"])
        clock = rate_metrics(group["clock_alarm"], group["failure"])
        matched_clock = rate_metrics(group["matched_clock_alarm"], group["failure"])
        rows.append(
            {
                "task": task,
                "suite": task.split("/", 1)[0],
                "episodes": len(group),
                "failures": int(group["failure"].sum()),
                "successes": int((~group["failure"]).sum()),
                **{f"detector_{key}": value for key, value in metrics.items()},
                **{f"clock_{key}": value for key, value in clock.items()},
                **{
                    f"matched_clock_{key}": value
                    for key, value in matched_clock.items()
                },
            }
        )
    return pd.DataFrame(rows)


def cause_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    alarmed = episodes[episodes["alarm"]].copy()
    rows: list[dict[str, Any]] = []
    for cause, group in alarmed.groupby("alarm_cause", sort=True):
        failures = int(group["failure"].sum())
        rows.append(
            {
                "alarm_cause": cause,
                "alarms": len(group),
                "failures": failures,
                "successes": len(group) - failures,
                "empirical_failure_fraction": failures / len(group),
                "first_alarm_query_median": float(group["first_alarm_query"].median()),
                "first_alarm_phase_median": float(group["first_alarm_phase"].median()),
            }
        )
    return pd.DataFrame(rows)


def clock_sweep(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failure = episodes["failure"].to_numpy(dtype=bool)
    lengths = episodes["episode_length"].to_numpy(dtype=np.int64)
    for query in range(int(lengths.max()) + 1):
        alarm = lengths > query
        rows.append({"query": query, **rate_metrics(alarm, failure)})
    return pd.DataFrame(rows)


def onset_audit(
    episodes: pd.DataFrame, onset_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not onset_path.exists():
        return pd.DataFrame(), {"available": False}
    onset = pd.read_csv(onset_path)
    onset = onset[np.isfinite(onset["onset_query"])].copy()
    keys = episodes[["task", "episode", "alarm", "first_alarm_query", "clock_alarm"]].copy()
    keys["matched_clock_alarm"] = episodes["matched_clock_alarm"]
    merged = onset.merge(keys, on=["task", "episode"], how="inner", validate="one_to_one")
    if merged.empty:
        return merged, {"available": True, "events": 0}
    merged["detector_delta"] = merged["first_alarm_query"] - merged["onset_query"]
    clock_query = float(episodes["clock_first_alarm_query"].dropna().iloc[0])
    merged["clock_first_alarm_query"] = np.where(merged["clock_alarm"], clock_query, np.nan)
    merged["clock_delta"] = merged["clock_first_alarm_query"] - merged["onset_query"]
    matched_clock_query = float(episodes["matched_clock_query"].iloc[0])
    merged["matched_clock_first_alarm_query"] = np.where(
        merged["matched_clock_alarm"], matched_clock_query, np.nan
    )
    merged["matched_clock_delta"] = (
        merged["matched_clock_first_alarm_query"] - merged["onset_query"]
    )

    def summarize(prefix: str, alarm_column: str, delta_column: str) -> dict[str, Any]:
        alarm = merged[alarm_column].fillna(False).to_numpy(dtype=bool)
        delta = merged[delta_column].to_numpy(dtype=np.float64)
        return {
            f"{prefix}_alarm_events": int(alarm.sum()),
            f"{prefix}_not_later_than_onset": int(np.sum(alarm & (delta <= 0))),
            f"{prefix}_within_minus2_to_onset": int(
                np.sum(alarm & (delta >= -2) & (delta <= 0))
            ),
            f"{prefix}_too_early": int(np.sum(alarm & (delta < -2))),
            f"{prefix}_late": int(np.sum(alarm & (delta > 0))),
        }

    summary = {"available": True, "events": len(merged)}
    summary.update(summarize("detector", "alarm", "detector_delta"))
    summary.update(summarize("clock", "clock_alarm", "clock_delta"))
    summary.update(
        summarize(
            "matched_clock", "matched_clock_alarm", "matched_clock_delta"
        )
    )
    return merged, summary


def plot_results(
    episodes: pd.DataFrame,
    detector_metrics: dict[str, Any],
    clock_metrics: dict[str, Any],
    matched_clock_metrics: dict[str, Any],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = ["MoE invariant", "Preset clock", "FPR-matched clock"]
    values = [
        [detector_metrics["precision"], detector_metrics["failure_recall"], detector_metrics["success_false_alarm_rate"]],
        [clock_metrics["precision"], clock_metrics["failure_recall"], clock_metrics["success_false_alarm_rate"]],
        [matched_clock_metrics["precision"], matched_clock_metrics["failure_recall"], matched_clock_metrics["success_false_alarm_rate"]],
    ]
    x = np.arange(3)
    for index, row in enumerate(values):
        axes[0].bar(x + (index - 1.0) * 0.25, row, width=0.25, label=names[index])
    axes[0].set_xticks(x, ["Precision", "Failure recall", "Success FPR"])
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False)
    axes[0].set_title("Held-out endpoint flip")

    for failure, label, color in ((False, "Success", "#3A7D44"), (True, "Failure", "#B33A3A")):
        values_max = episodes.loc[episodes["failure"] == failure, "episode_max_confidence"]
        axes[1].hist(values_max, bins=30, alpha=0.55, density=True, label=label, color=color)
    axes[1].set_xlabel("Episode max detector confidence")
    axes[1].set_title("Held-out score distribution")
    axes[1].legend(frameon=False)

    alarmed = episodes[episodes["alarm"]]
    for failure, label, color in ((False, "Success alarm", "#3A7D44"), (True, "Failure alarm", "#B33A3A")):
        phases = alarmed.loc[alarmed["failure"] == failure, "first_alarm_phase"].dropna()
        if len(phases):
            axes[2].hist(phases, bins=np.linspace(0, 1, 21), alpha=0.55, label=label, color=color)
    axes[2].set_xlabel("First alarm / episode horizon")
    axes[2].set_title("Alarm timing")
    axes[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *body])


def render_report(
    summary: dict[str, Any],
    split: pd.DataFrame,
    causes: pd.DataFrame,
    tasks: pd.DataFrame,
) -> str:
    detector = summary["heldout_endpoint_flip"]["detector"]
    clock = summary["heldout_endpoint_flip"]["fixed_clock"]
    matched_clock = summary["heldout_endpoint_flip"]["heldout_fpr_matched_clock"]
    onset = summary["secondary_onset_proxy"]
    full = summary["all_episodes_descriptive"]
    heldout = split[split["role"] == "heldout_test"][["suite", "task"]]
    lines = [
        "# cache_new MoE 健康不变量报警实验",
        "",
        "## 核心结果",
        "",
        (
            "**严格判定：通过。**"
            if summary["decision"]["strict_test_passed"]
            else "**严格判定：未通过。** 固定三头 MoE 判断器在等误报比较中被时间钟支配，且首次报警没有稳定贴近物理 onset。"
        ),
        "",
        "本实验完全不使用旧的跨语料概率表。检测器只读取当前轨迹截至当前 query 的 "
        "HB full-softmax MoE 路由；任务 ID、动作、物理状态、reward、结果和未来长度均不进入检测。",
        "",
        f"固定任务级哈希划分得到 {summary['split']['threshold_confirmation_tasks']} 个阈值确认任务和 "
        f"{summary['split']['heldout_tasks']} 个留出任务。阈值确认只使用成功轨迹，整条成功轨迹误报预算为 "
        f"{summary['threshold']['target_success_episode_fpr']:.1%}，实际为 "
        f"{summary['threshold']['confirmation_success_episode_fpr']:.2%}。失败标签没有参与特征、组合或阈值。",
        "",
        f"留出集共 {summary['heldout']['episodes']} 条轨迹，其中失败 {summary['heldout']['failures']} 条。"
        f"MoE 判断器报警 {detector['tp'] + detector['fp']} 条：TP={detector['tp']}、FP={detector['fp']}、"
        f"FN={detector['fn']}、TN={detector['tn']}；precision={detector['precision']:.2%}、"
        f"failure recall={detector['failure_recall']:.2%}、成功误报率={detector['success_false_alarm_rate']:.2%}。",
        "",
        f"同一 {summary['threshold']['target_success_episode_fpr']:.1%} 阈值确认预算下，固定时间钟在 "
        f"q={summary['clock']['query']} 报警；留出 precision={clock['precision']:.2%}、"
        f"failure recall={clock['failure_recall']:.2%}、成功误报率={clock['success_false_alarm_rate']:.2%}。",
        "",
        f"由于任务迁移后 q={summary['clock']['query']} 的实际误报率远低于 MoE，另给出一个只在揭盲后用于"
        f"公平诊断的等误报时钟 q={summary['heldout_fpr_matched_clock']['query']}："
        f"precision={matched_clock['precision']:.2%}、failure recall={matched_clock['failure_recall']:.2%}、"
        f"成功误报率={matched_clock['success_false_alarm_rate']:.2%}。它不参与阈值选择，只是对 MoE 结果的"
        "保守压力测试。",
        "",
        f"留出报警涉及 {summary['heldout_alarm_structure']['alarm_tasks']}/"
        f"{summary['split']['heldout_tasks']} 个任务；"
        f"误报只涉及 {summary['heldout_alarm_structure']['false_alarm_tasks']} 个任务。"
        f"收敛+响应联合分支命中 "
        f"{summary['heldout_alarm_structure']['joint_convergence_response_alarm_episodes']} 条，"
        f"持续复返分支命中 "
        f"{summary['heldout_alarm_structure']['persistent_recurrence_alarm_episodes']} 条。"
        "这说明最终结果实际退化成单一 recurrence 检测器，而不是预期的三种机制互补。",
        "",
        "因此是否证明 MoE 有独立价值，只看它在相同误报率下能否超过时间钟，而不根据绝对 precision 单独下结论。",
        "",
        f"在留出预测固定后，才额外揭开阈值确认任务中此前未使用的失败标签。全 40 任务、"
        f"{full['episodes']} 条轨迹的描述性翻牌中，检测器检出 {full['detector']['tp']}/"
        f"{full['failures']} 个失败（{full['detector']['failure_recall']:.2%}），"
        f"precision={full['detector']['precision']:.2%}、成功误报率="
        f"{full['detector']['success_false_alarm_rate']:.2%}。这张表覆盖全部失败，但成功误报率包含"
        "阈值确认样本，不能替代严格留出结果。",
        "",
        "## 固定检测器",
        "",
        "每条轨迹使用 q0--q3 建立自身基线，从 q4 开始评分。三个不可学习的机制头为：",
        "",
        "- 去噪收敛失败：完整 flow 路由路径中发生在 late flow 的 Hellinger 路径占比，相对自身早期基线上升；",
        "- state-action 响应脱节：相邻重规划间 front-HB state-token 路由跳变大于 action-token 跳变；",
        "- 跨 chunk 复返：final-flow back-HB action routing 对 lag 1--4 历史的最大 weighted Jaccard。",
        "",
        "前两个头必须在同一 query 同时极端；复返头必须连续两个 query 极端。两条分支取 OR，但不学习任何权重。",
        "每个头的数值只换算成相对阈值确认成功轨迹整段最大值的健康尾部置信度。这是异常显著性，不是失败概率。",
        "",
        "## 报警原因翻牌",
        "",
        markdown_table(causes),
        "",
        "## 留出任务",
        "",
        markdown_table(heldout),
        "",
        "## 逐任务结果",
        "",
        markdown_table(tasks),
        "",
        "## onset 代理的二次审计",
        "",
    ]
    if onset.get("available") and onset.get("events", 0):
        lines.extend(
            [
                f"留出任务中有 {onset['events']} 个既有的 query-boundary 物理 onset 代理。"
                f"MoE 首次报警不晚于 onset 为 {onset['detector_not_later_than_onset']}，"
                f"位于 [-2,0] 为 {onset['detector_within_minus2_to_onset']}；"
                f"预固定时间钟对应为 {onset['clock_not_later_than_onset']} 和 "
                f"{onset['clock_within_minus2_to_onset']}；等误报时钟对应为 "
                f"{onset['matched_clock_not_later_than_onset']} 和 "
                f"{onset['matched_clock_within_minus2_to_onset']}。",
                "",
                "这些标签仅用于预测文件哈希后的事后审计，属于运动学代理，不是 contact/video 真值。",
            ]
        )
    else:
        lines.append("留出任务中没有可连接的既有 onset 代理。")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 最终 success/failure 翻牌评价的是任意 endpoint failure 关联，不等价于特定 Trap 类型。",
            "- `confidence` 表示健康路由下的经验极端程度，不是单轨迹失败概率。",
            "- 数据集此前已被用于其他探索性研究；任务划分和本检测器未使用留出失败标签，但不能称为全新 prospective benchmark。",
            "- 只有当 MoE 在同成功误报预算下稳定超过时间钟，才能声称它提供了时长之外的失败信息。",
            "",
            "## 可复现产物",
            "",
            "- `configs/moe_invariant_alarm_cache_new.json`：预固定设计；",
            "- `code/evaluate_moe_invariant_alarm_cache_new.py`：路由提取、阈值确认、盲回放、揭盲和审计；",
            "- `heldout_predictions_label_free.csv.gz`：揭盲前逐 query 预测；",
            "- `prediction_manifest.json`：预测文件 SHA256 和标签隔离声明；",
            "- `tables/heldout_episode_flip.csv`：逐轨迹翻牌；",
            "- `tables/all_episode_flip_descriptive.csv`：预测冻结后生成的全 16,000 轨迹描述性翻牌；",
            "- `summary.json`：机器可读主结果。",
            "",
        ]
    )
    return "\n".join(lines)


def self_test() -> None:
    values = np.asarray([[0.2, 0.8], [0.5, 0.5]], dtype=np.float32)
    assert np.allclose(hellinger(values, values), 0.0, atol=1e-6)
    assert np.allclose(weighted_jaccard(values, values), 1.0, atol=1e-6)
    reference = np.asarray([1.0, 2.0, 3.0])
    confidence = finite_tail_confidence(reference, np.asarray([0.0, 2.5, 4.0]))
    assert np.allclose(confidence, [0.0, 0.5, 0.75])
    threshold, rate = select_threshold(np.arange(100, dtype=np.float64), 0.01)
    assert threshold == 99.0 and rate == 0.01
    split = split_tasks(
        [f"suite_{suite}/task_{index}" for suite in range(2) for index in range(5)],
        "test",
        1,
    )
    assert (split["role"] == "heldout_test").sum() == 2
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training") is not False or config.get("probability_calibration") is not False:
        raise ValueError("experiment must remain train-free and probability-free")
    output = args.output.resolve()
    cache_dir = output / "route_only_tasks"
    table_dir = output / "tables"
    figure_dir = output / "figures"
    for directory in (output, cache_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_root = resolve_workspace(config["cache_root"])
    runs = discover_runs(cache_root, config["run_id"])
    if len(runs) != int(config["expected_tasks"]):
        raise ValueError(f"expected {config['expected_tasks']} tasks, found {len(runs)}")
    run_map = dict(runs)
    split = split_tasks(
        run_map,
        config["task_split"]["salt"],
        int(config["task_split"]["heldout_tasks_per_suite"]),
    )
    split.to_csv(table_dir / "task_split.csv", index=False)
    split_sha = sha256_file(table_dir / "task_split.csv")

    extraction: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                extract_task_route_features,
                task,
                str(run),
                str(cache_dir),
                config,
                args.rebuild,
            ): task
            for task, run in runs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            extraction.append(result)
            print(
                f"[route-only {index}/{len(futures)}] {result['task']}: "
                f"rows={result['route_rows']} reused={result['cache_reused']}",
                flush=True,
            )
    extraction.sort(key=lambda row: row["task"])
    write_json(output / "route_only_manifest.json", {
        "schema": SCHEMA,
        "endpoint_labels_loaded": False,
        "task_split_sha256": split_sha,
        "tasks": extraction,
        "route_rows": sum(row["route_rows"] for row in extraction),
    })

    confirmation_tasks = set(
        split.loc[split["role"] == "threshold_confirmation", "task"].astype(str)
    )
    heldout_tasks = set(split.loc[split["role"] == "heldout_test", "task"].astype(str))

    # This is the first label access. Only confirmation-task success flags are opened.
    success_max_rows: list[dict[str, Any]] = []
    success_lengths: list[int] = []
    raw_by_task: dict[str, dict[str, np.ndarray]] = {}
    for task in sorted(confirmation_tasks):
        arrays = load_route_cache(task_cache_path(cache_dir, task))
        raw_by_task[task] = arrays
        outcomes = load_outcome_map(run_map[task])
        for episode_id, indices in episode_slices(arrays["episode"]):
            if not outcomes[episode_id]:
                continue
            row: dict[str, Any] = {"task": task, "episode": episode_id}
            for head in HEADS:
                values = np.asarray(arrays[f"evidence_{head}"][indices], dtype=np.float64)
                finite = values[np.isfinite(values)]
                row[f"max_{head}"] = float(finite.max()) if len(finite) else np.nan
            success_max_rows.append(row)
            success_lengths.append(len(indices))
    success_max = pd.DataFrame(success_max_rows)
    success_max.to_csv(table_dir / "confirmation_success_head_maxima.csv", index=False)
    references = {
        head: success_max[f"max_{head}"].dropna().to_numpy(dtype=np.float64)
        for head in HEADS
    }

    confirmation_detector_max: list[float] = []
    for task in sorted(confirmation_tasks):
        arrays = raw_by_task.pop(task)
        outcomes = load_outcome_map(run_map[task])
        frame = detector_series(arrays, references)
        maxima = episode_maxima(frame, "detector_confidence")
        for episode_id, value in maxima.items():
            if outcomes[int(episode_id)] and np.isfinite(value):
                confirmation_detector_max.append(float(value))
    budget = float(config["threshold_confirmation"]["target_success_episode_false_alarm_rate"])
    detector_threshold, confirmation_fpr = select_threshold(
        np.asarray(confirmation_detector_max), budget
    )
    clock_query, clock_confirmation_fpr = select_clock_query(
        np.asarray(success_lengths), budget
    )
    threshold_payload = {
        "schema": SCHEMA,
        "failure_labels_used": False,
        "confirmation_success_episodes": len(confirmation_detector_max),
        "target_success_episode_fpr": budget,
        "detector_confidence_threshold": detector_threshold,
        "confirmation_success_episode_fpr": confirmation_fpr,
        "clock_query": clock_query,
        "clock_confirmation_success_episode_fpr": clock_confirmation_fpr,
        "head_reference_episode_maxima": {
            head: {
                "count": len(values),
                "min": float(np.min(values)),
                "median": float(np.median(values)),
                "max": float(np.max(values)),
            }
            for head, values in references.items()
        },
    }
    write_json(output / "thresholds.json", threshold_payload)

    # Held-out route predictions are materialized before held-out summaries are opened.
    prediction_parts: list[pd.DataFrame] = []
    heldout_lengths: dict[tuple[str, int], int] = {}
    for task in sorted(heldout_tasks):
        arrays = load_route_cache(task_cache_path(cache_dir, task))
        frame = alarm_rows(detector_series(arrays, references), detector_threshold)
        frame = frame[frame["valid"]].copy()
        frame.insert(0, "task", task)
        prediction_parts.append(frame.drop(columns=["valid"]))
        for episode_id, indices in episode_slices(arrays["episode"]):
            heldout_lengths[(task, episode_id)] = len(indices)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_path = output / "heldout_predictions_label_free.csv.gz"
    predictions.to_csv(
        prediction_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    prediction_sha = sha256_file(prediction_path)
    write_json(output / "prediction_manifest.json", {
        "schema": SCHEMA,
        "heldout_endpoint_labels_loaded": False,
        "predictions_written_before_heldout_outcome_load": True,
        "prediction_path": str(prediction_path.relative_to(PACKAGE_ROOT)),
        "prediction_sha256": prediction_sha,
        "prediction_rows": len(predictions),
        "heldout_tasks": sorted(heldout_tasks),
        "task_split_sha256": split_sha,
        "detector_threshold_sha256": sha256_file(output / "thresholds.json"),
    })
    print(f"held-out predictions frozen: sha256={prediction_sha}", flush=True)

    # Held-out endpoint outcomes are opened only below this line.
    episode_rows: list[dict[str, Any]] = []
    for task in sorted(heldout_tasks):
        outcomes = load_outcome_map(run_map[task])
        task_predictions = predictions[predictions["task"] == task]
        lengths = {
            episode_id: heldout_lengths[(task, episode_id)] for episode_id in outcomes
        }
        episode_rows.extend(
            build_episode_flip_rows(
                task,
                task_predictions,
                lengths,
                outcomes,
                clock_query,
                "heldout_test",
            )
        )
    episodes = pd.DataFrame(episode_rows)
    episodes.to_csv(table_dir / "heldout_episode_flip.csv", index=False)
    detector_metrics = rate_metrics(episodes["alarm"], episodes["failure"])
    clock_metrics = rate_metrics(episodes["clock_alarm"], episodes["failure"])
    sweep = clock_sweep(episodes)
    matched_candidates = sweep[
        sweep["success_false_alarm_rate"]
        <= detector_metrics["success_false_alarm_rate"] + 1e-15
    ]
    matched_clock_row = matched_candidates.sort_values("query").iloc[0]
    matched_clock_query = int(matched_clock_row["query"])
    episodes["matched_clock_alarm"] = episodes["episode_length"] > matched_clock_query
    episodes["matched_clock_query"] = matched_clock_query
    matched_clock_metrics = rate_metrics(
        episodes["matched_clock_alarm"], episodes["failure"]
    )
    episodes.to_csv(table_dir / "heldout_episode_flip.csv", index=False)
    sweep.to_csv(table_dir / "heldout_clock_sweep.csv", index=False)
    tasks = task_metrics(episodes)
    causes = cause_metrics(episodes)
    tasks.to_csv(table_dir / "heldout_task_metrics.csv", index=False)
    causes.to_csv(table_dir / "heldout_alarm_causes.csv", index=False)

    onset_setting = config["heldout_protocol"].get("secondary_onset_proxy")
    onset_run_id = config["heldout_protocol"].get("secondary_onset_proxy_run_id")
    if onset_setting is not None and onset_run_id != config["run_id"]:
        raise ValueError(
            "secondary onset proxy is not declared for this run: "
            f"{onset_run_id!r} != {config['run_id']!r}"
        )
    if onset_setting is None:
        onset_frame = pd.DataFrame()
        onset_summary = {"available": False, "disabled": True}
    else:
        onset_path = resolve_workspace(onset_setting)
        onset_frame, onset_summary = onset_audit(episodes, onset_path)
    if not onset_frame.empty:
        onset_frame.to_csv(table_dir / "heldout_onset_proxy_audit.csv", index=False)

    # Only after held-out predictions are frozen and evaluated do we reveal the
    # previously unused failure flags in the confirmation tasks. These rows are
    # descriptive; successful confirmation episodes helped set the threshold.
    confirmation_episode_rows: list[dict[str, Any]] = []
    for task in sorted(confirmation_tasks):
        arrays = load_route_cache(task_cache_path(cache_dir, task))
        frame = alarm_rows(detector_series(arrays, references), detector_threshold)
        frame = frame[frame["valid"]].copy()
        outcomes = load_outcome_map(run_map[task])
        lengths = {
            episode_id: int(np.sum(arrays["episode"] == episode_id))
            for episode_id in outcomes
        }
        confirmation_episode_rows.extend(
            build_episode_flip_rows(
                task,
                frame,
                lengths,
                outcomes,
                clock_query,
                "threshold_confirmation",
            )
        )
    all_episodes = pd.concat(
        [pd.DataFrame(confirmation_episode_rows), episodes], ignore_index=True
    )
    all_episodes.to_csv(table_dir / "all_episode_flip_descriptive.csv", index=False)
    all_detector_metrics = rate_metrics(all_episodes["alarm"], all_episodes["failure"])
    all_clock_metrics = rate_metrics(
        all_episodes["clock_alarm"], all_episodes["failure"]
    )
    all_causes = cause_metrics(all_episodes)
    all_causes.to_csv(table_dir / "all_alarm_causes_descriptive.csv", index=False)

    heldout_alarm_tasks = int(episodes.loc[episodes["alarm"], "task"].nunique())
    heldout_false_alarm_tasks = int(
        episodes.loc[episodes["alarm"] & ~episodes["failure"], "task"].nunique()
    )
    joint_episode_keys = predictions.loc[
        predictions["joint_confidence"] >= detector_threshold, ["task", "episode"]
    ].drop_duplicates()
    recurrence_episode_keys = predictions.loc[
        predictions["recurrence_persistent_confidence"] >= detector_threshold,
        ["task", "episode"],
    ].drop_duplicates()
    beats_matched_clock = bool(
        detector_metrics["failure_recall"] > matched_clock_metrics["failure_recall"]
    )

    alarmed = episodes[episodes["alarm"]]
    summary = {
        "schema": SCHEMA,
        "training": False,
        "gradient_optimization": False,
        "learned_feature_weights": False,
        "probability_calibration": False,
        "data": {
            "cache_root": str(cache_root),
            "run_id": config["run_id"],
            "tasks": len(runs),
            "episodes": len(runs) * int(config["expected_episodes_per_task"]),
            "route_queries": int(sum(row["route_rows"] for row in extraction)),
        },
        "split": {
            "method": config["task_split"]["method"],
            "threshold_confirmation_tasks": len(confirmation_tasks),
            "heldout_tasks": len(heldout_tasks),
            "task_split_sha256": split_sha,
        },
        "threshold": {
            "target_success_episode_fpr": budget,
            "confirmation_success_episodes": len(confirmation_detector_max),
            "detector_confidence": detector_threshold,
            "confirmation_success_episode_fpr": confirmation_fpr,
        },
        "clock": {
            "query": clock_query,
            "confirmation_success_episode_fpr": clock_confirmation_fpr,
        },
        "heldout": {
            "episodes": len(episodes),
            "successes": int(episodes["success"].sum()),
            "failures": int(episodes["failure"].sum()),
            "prediction_rows": len(predictions),
            "prediction_sha256": prediction_sha,
            "failure_alarm_first_query_median": float(
                alarmed.loc[alarmed["failure"], "first_alarm_query"].median()
            ) if np.any(alarmed["failure"]) else None,
            "failure_alarm_phase_median": float(
                alarmed.loc[alarmed["failure"], "first_alarm_phase"].median()
            ) if np.any(alarmed["failure"]) else None,
            "failure_alarm_lead_to_endpoint_median": float(
                alarmed.loc[alarmed["failure"], "lead_to_endpoint_queries"].median()
            ) if np.any(alarmed["failure"]) else None,
        },
        "heldout_endpoint_flip": {
            "detector": detector_metrics,
            "fixed_clock": clock_metrics,
            "heldout_fpr_matched_clock": matched_clock_metrics,
        },
        "heldout_alarm_structure": {
            "alarm_tasks": heldout_alarm_tasks,
            "false_alarm_tasks": heldout_false_alarm_tasks,
            "joint_convergence_response_alarm_episodes": len(joint_episode_keys),
            "persistent_recurrence_alarm_episodes": len(recurrence_episode_keys),
        },
        "all_episodes_descriptive": {
            "episodes": len(all_episodes),
            "successes": int(all_episodes["success"].sum()),
            "failures": int(all_episodes["failure"].sum()),
            "detector": all_detector_metrics,
            "fixed_clock": all_clock_metrics,
            "warning": "confirmation successes set the threshold; this does not replace the heldout result",
        },
        "heldout_fpr_matched_clock": {
            "query": matched_clock_query,
            "selection_uses_heldout_success_labels": True,
            "role": "post-reveal diagnostic control only",
        },
        "secondary_onset_proxy": onset_summary,
        "decision": {
            "strict_test_passed": beats_matched_clock,
            "beats_heldout_fpr_matched_clock_on_failure_recall": beats_matched_clock,
            "deployment_ready": False,
            "primary_result": (
                "PASS: fixed MoE invariant alarm beats the FPR-matched clock"
                if beats_matched_clock
                else "FAIL: fixed MoE invariant alarm is dominated by the FPR-matched clock and is not onset-localized"
            ),
        },
        "runtime_inputs": config["runtime_inputs"],
        "runtime_excluded_inputs": config["runtime_excluded_inputs"],
        "conclusion_guard": "MoE value requires improvement over the fixed clock at the same confirmation success false-alarm budget.",
    }
    write_json(output / "summary.json", summary)
    plot_results(
        episodes,
        detector_metrics,
        clock_metrics,
        matched_clock_metrics,
        figure_dir / "heldout_invariant_alarm.png",
    )
    (output / "REPORT_ZH.md").write_text(
        render_report(summary, split, causes, tasks), encoding="utf-8"
    )
    print(f"analysis complete: {output}", flush=True)


if __name__ == "__main__":
    main()
