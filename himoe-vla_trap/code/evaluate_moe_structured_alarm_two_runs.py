#!/usr/bin/env python3
"""Train-free structured MoE alarm with two-way frozen replay.

The detector keeps layer, flow, token, full-softmax, selected-support, and
selected-weight views until mechanism-level aggregation. Healthy-reference,
threshold-confirmation, and held-out tasks are disjoint. For each direction,
target predictions are materialized and hashed before target outcomes are read.
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
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/moe_structured_alarm_two_runs.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_structured_alarm_two_runs"
SCHEMA = "himoe.moe_structured_alarm_two_runs.v1"
HEADS = ("lock_in", "instability", "state_action_decoupling")
RULES = ("combined", *HEADS)


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
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
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


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalize_probability(left)
    right = normalize_probability(right)
    affinity = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - affinity, 0.0, 1.0))


def support_jaccard_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left)
    right = np.asarray(right)
    matches = left[..., :, None] == right[..., None, :]
    intersection = matches.any(axis=-1).sum(axis=-1).astype(np.float32)
    k = float(left.shape[-1])
    return intersection / np.maximum(2.0 * k - intersection, 1e-12)


def selected_jaccard_similarity(
    left_ids: np.ndarray,
    left_weight: np.ndarray,
    right_ids: np.ndarray,
    right_weight: np.ndarray,
) -> np.ndarray:
    left_weight = normalize_probability(left_weight)
    right_weight = normalize_probability(right_weight)
    matches = np.asarray(left_ids)[..., :, None] == np.asarray(right_ids)[..., None, :]
    overlap = np.minimum(left_weight[..., :, None], right_weight[..., None, :])
    intersection = (overlap * matches).sum(axis=(-2, -1))
    return intersection / np.maximum(2.0 - intersection, 1e-12)


def discover_runs(cache_root: Path, run_id: str) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for route_path in sorted(cache_root.glob(f"libero_*/*/{run_id}/server/routes.zarr")):
        run = route_path.parents[1]
        task = str(run.relative_to(cache_root).parent)
        summary_path = run / "client/capture_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            sampling = summary.get("sampling", {})
            if sampling and int(sampling.get("actual_episodes", -1)) != int(
                sampling.get("designed_episodes", -2)
            ):
                continue
        rows.append((task, run))
    if not rows:
        raise RuntimeError(f"no complete run {run_id!r} below {cache_root}")
    return rows


def partition_tasks(tasks: Iterable[str], config: dict[str, Any]) -> pd.DataFrame:
    setting = config["task_partition"]
    heldout_count = int(setting["heldout_test_per_suite"])
    threshold_count = int(setting["threshold_confirmation_per_suite"])
    reference_count = int(setting["healthy_reference_per_suite"])
    grouped: dict[str, list[tuple[str, str]]] = {}
    for task in sorted(tasks):
        suite = task.split("/", 1)[0]
        digest = hashlib.sha256(f"{setting['salt']}:{task}".encode()).hexdigest()
        grouped.setdefault(suite, []).append((digest, task))
    output: list[dict[str, Any]] = []
    for suite, members in sorted(grouped.items()):
        ranked = sorted(members)
        expected = heldout_count + threshold_count + reference_count
        if len(ranked) != expected:
            raise ValueError(f"{suite}: expected {expected} tasks, found {len(ranked)}")
        for rank, (digest, task) in enumerate(ranked):
            if rank < heldout_count:
                role = "heldout_test"
            elif rank < heldout_count + threshold_count:
                role = "threshold_confirmation"
            else:
                role = "healthy_reference"
            output.append(
                {
                    "task": task,
                    "suite": suite,
                    "sha256_rank": rank,
                    "sha256": digest,
                    "role": role,
                }
            )
    return pd.DataFrame(output).sort_values(["suite", "sha256_rank"]).reset_index(drop=True)


def feature_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "schema": SCHEMA,
        "position_grid": config["position_grid"],
        "causal_self_reference": config["causal_self_reference"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_cache_path(cache_root: Path, run_name: str, task: str) -> Path:
    token = hashlib.sha256(task.encode()).hexdigest()[:16]
    return cache_root / run_name / f"{token}.npz"


def groups_from_config(config: dict[str, Any]) -> tuple[list[np.ndarray], ...]:
    grid = config["position_grid"]
    layers = [np.asarray(values, dtype=np.int64) for values in grid["layer_groups"].values()]
    probability_flows = [
        np.asarray(values, dtype=np.int64)
        for values in grid["flow_groups"]["probability"].values()
    ]
    transition_flows = [
        np.asarray(values, dtype=np.int64)
        for values in grid["flow_groups"]["transition"].values()
    ]
    acceleration_flows = [
        np.asarray(values, dtype=np.int64)
        for values in grid["flow_groups"]["acceleration"].values()
    ]
    tokens = [np.asarray(values, dtype=np.int64) for values in grid["token_groups"].values()]
    action_tokens = [values - 1 for values in tokens[1:]]
    return layers, probability_flows, transition_flows, acceleration_flows, tokens, action_tokens


def aggregate_cells(
    values: np.ndarray,
    layer_groups: list[np.ndarray],
    flow_groups: list[np.ndarray],
    token_groups: list[np.ndarray],
) -> np.ndarray:
    output = np.empty(
        (len(values), len(layer_groups), len(flow_groups), len(token_groups)),
        dtype=np.float32,
    )
    for layer_index, layers in enumerate(layer_groups):
        by_layer = np.take(values, layers, axis=1)
        for flow_index, flows in enumerate(flow_groups):
            by_flow = np.take(by_layer, flows, axis=2)
            for token_index, tokens in enumerate(token_groups):
                output[:, layer_index, flow_index, token_index] = np.take(
                    by_flow, tokens, axis=3
                ).mean(axis=(1, 2, 3))
    return output


def aggregate_layer_token(
    values: np.ndarray,
    layer_groups: list[np.ndarray],
    token_groups: list[np.ndarray],
) -> np.ndarray:
    output = np.empty((len(values), len(layer_groups), len(token_groups)), dtype=np.float32)
    for layer_index, layers in enumerate(layer_groups):
        by_layer = np.take(values, layers, axis=1)
        for token_index, tokens in enumerate(token_groups):
            output[:, layer_index, token_index] = np.take(
                by_layer, tokens, axis=2
            ).mean(axis=(1, 2))
    return output


def aggregate_state_action(
    values: np.ndarray,
    layer_groups: list[np.ndarray],
    flow_groups: list[np.ndarray],
    action_token_groups: list[np.ndarray],
) -> np.ndarray:
    return aggregate_cells(values, layer_groups, flow_groups, action_token_groups)


def aggregate_layer_flow(
    values: np.ndarray,
    layer_groups: list[np.ndarray],
    flow_groups: list[np.ndarray],
) -> np.ndarray:
    output = np.empty((len(values), len(layer_groups), len(flow_groups)), dtype=np.float32)
    for layer_index, layers in enumerate(layer_groups):
        by_layer = np.take(values, layers, axis=1)
        for flow_index, flows in enumerate(flow_groups):
            output[:, layer_index, flow_index] = np.take(
                by_layer, flows, axis=2
            ).mean(axis=(1, 2))
    return output


def quantile_flat(values: np.ndarray, quantile: float) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    output = np.full(len(flattened), np.nan, dtype=np.float32)
    valid = np.isfinite(flattened).any(axis=1)
    if np.any(valid):
        output[valid] = np.nanquantile(
            flattened[valid], quantile, axis=1
        ).astype(np.float32)
    return output


def component_names(lags: list[int]) -> list[str]:
    names = [
        "instability_soft_flow",
        "instability_acceleration",
        "instability_selected_flow_switch",
        "instability_chunk_selected_switch",
        "decoupling_front_state_action_gap",
        "decoupling_back_state_action_gap",
        "decoupling_chunk_jump_imbalance",
        "decoupling_action_span_selected_gap",
        "lock_low_flow_support_switch",
        "lock_low_action_span_support_gap",
    ]
    for lag in lags:
        names.extend(
            [
                f"lock_soft_recurrence_lag{lag}",
                f"lock_support_recurrence_lag{lag}",
                f"lock_selected_recurrence_lag{lag}",
            ]
        )
    return names


def extract_task_features(
    task: str,
    run_name: str,
    run_string: str,
    cache_root_string: str,
    config: dict[str, Any],
    rebuild: bool,
) -> dict[str, Any]:
    run = Path(run_string)
    cache = task_cache_path(Path(cache_root_string), run_name, task)
    fingerprint = feature_fingerprint(config)
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=False) as archive:
            if (
                str(archive["schema"].item()) == SCHEMA
                and str(archive["task"].item()) == task
                and str(archive["run"].item()) == str(run)
                and str(archive["feature_fingerprint"].item()) == fingerprint
            ):
                return {
                    "run_name": run_name,
                    "task": task,
                    "cache": str(cache),
                    "route_rows": int(len(archive["episode"])),
                    "episodes": int(len(np.unique(archive["episode"]))),
                    "reused": True,
                }

    layer_groups, probability_flows, transition_flows, acceleration_flows, token_groups, action_groups = groups_from_config(config)
    lags = [int(value) for value in config["position_grid"]["recurrence_lags"]]
    names = component_names(lags)
    group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    required = ("hb_router_probs", "hb_expert_ids", "hb_selected_prob")
    if any(name not in group for name in required):
        raise ValueError(f"missing structured route field: {task}/{run_name}")
    router = group["hb_router_probs"]
    expert_ids = group["hb_expert_ids"]
    selected_probability = group["hb_selected_prob"]
    if tuple(router.shape[1:]) != (8, 10, 11, 32):
        raise ValueError(f"unexpected route geometry: {task}/{run_name} {router.shape}")
    if tuple(expert_ids.shape[1:]) != (8, 10, 11, 4):
        raise ValueError(f"unexpected expert-ID geometry: {task}/{run_name}")

    episode = np.asarray(group["episode_id"][:], dtype=np.int32)
    control_step = np.asarray(group["control_step"][:], dtype=np.int64)
    total = len(episode)
    if total != router.shape[0] or np.any(np.diff(control_step) != 1):
        raise ValueError(f"route metadata mismatch: {task}/{run_name}")
    unique_episodes = np.unique(episode)
    if len(unique_episodes) != int(config["expected_episodes_per_task"]):
        raise ValueError(f"unexpected episode count: {task}/{run_name}")

    cell_shape = (total, 2, 3, 4)
    flow_soft = np.empty(cell_shape, dtype=np.float16)
    flow_support = np.empty(cell_shape, dtype=np.float16)
    flow_selected = np.empty(cell_shape, dtype=np.float16)
    acceleration = np.empty(cell_shape, dtype=np.float16)
    entropy = np.empty(cell_shape, dtype=np.float16)
    margin = np.empty(cell_shape, dtype=np.float16)
    state_action_soft = np.empty((total, 2, 3, 3), dtype=np.float16)
    state_action_support = np.empty((total, 2, 3, 3), dtype=np.float16)
    state_action_selected = np.empty((total, 2, 3, 3), dtype=np.float16)
    action_span_soft = np.empty((total, 2, 3), dtype=np.float16)
    action_span_support = np.empty((total, 2, 3), dtype=np.float16)
    action_span_selected = np.empty((total, 2, 3), dtype=np.float16)
    chunk_soft = np.full(cell_shape, np.nan, dtype=np.float16)
    chunk_support = np.full(cell_shape, np.nan, dtype=np.float16)
    chunk_selected = np.full(cell_shape, np.nan, dtype=np.float16)
    lag_soft = np.full((total, len(lags), 2, 4), np.nan, dtype=np.float16)
    lag_support = np.full_like(lag_soft, np.nan)
    lag_selected = np.full_like(lag_soft, np.nan)

    history_probability: np.ndarray | None = None
    history_ids: np.ndarray | None = None
    history_selected: np.ndarray | None = None
    history_episode: np.ndarray | None = None
    max_lag = max(lags)

    for start in range(0, total, 32):
        stop = min(start + 32, total)
        probability = normalize_probability(router[start:stop])
        ids = np.asarray(expert_ids[start:stop], dtype=np.uint8)
        selected = np.asarray(selected_probability[start:stop], dtype=np.float32)

        soft_distance = hellinger(probability[:, :, 1:], probability[:, :, :-1])
        support_distance = 1.0 - support_jaccard_similarity(ids[:, :, 1:], ids[:, :, :-1])
        selected_distance = 1.0 - selected_jaccard_similarity(
            ids[:, :, 1:], selected[:, :, 1:], ids[:, :, :-1], selected[:, :, :-1]
        )
        root = np.sqrt(probability)
        curvature = root[:, :, 2:] - 2.0 * root[:, :, 1:-1] + root[:, :, :-2]
        curvature = np.linalg.norm(curvature, axis=-1) / np.sqrt(2.0)
        flow_soft[start:stop] = aggregate_cells(
            soft_distance, layer_groups, transition_flows, token_groups
        )
        flow_support[start:stop] = aggregate_cells(
            support_distance, layer_groups, transition_flows, token_groups
        )
        flow_selected[start:stop] = aggregate_cells(
            selected_distance, layer_groups, transition_flows, token_groups
        )
        acceleration[start:stop] = aggregate_cells(
            curvature, layer_groups, acceleration_flows, token_groups
        )

        route_entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(axis=-1)
        top_two = np.partition(probability, -2, axis=-1)[..., -2:]
        route_margin = top_two.max(axis=-1) - top_two.min(axis=-1)
        entropy[start:stop] = aggregate_cells(
            route_entropy, layer_groups, probability_flows, token_groups
        )
        margin[start:stop] = aggregate_cells(
            route_margin, layer_groups, probability_flows, token_groups
        )

        state_probability = probability[:, :, :, 0]
        action_probability = probability[:, :, :, 1:]
        state_ids = ids[:, :, :, 0]
        action_ids = ids[:, :, :, 1:]
        state_selected = selected[:, :, :, 0]
        action_selected = selected[:, :, :, 1:]
        sa_soft = hellinger(state_probability[..., None, :], action_probability)
        sa_support = 1.0 - support_jaccard_similarity(
            state_ids[..., None, :], action_ids
        )
        sa_selected = 1.0 - selected_jaccard_similarity(
            state_ids[..., None, :],
            state_selected[..., None, :],
            action_ids,
            action_selected,
        )
        state_action_soft[start:stop] = aggregate_state_action(
            sa_soft, layer_groups, probability_flows, action_groups
        )
        state_action_support[start:stop] = aggregate_state_action(
            sa_support, layer_groups, probability_flows, action_groups
        )
        state_action_selected[start:stop] = aggregate_state_action(
            sa_selected, layer_groups, probability_flows, action_groups
        )

        first_probability = probability[:, :, :, 1:4]
        last_probability = probability[:, :, :, 8:11]
        span_soft = hellinger(
            first_probability[..., :, None, :], last_probability[..., None, :, :]
        ).mean(axis=(-2, -1))
        first_ids = ids[:, :, :, 1:4]
        last_ids = ids[:, :, :, 8:11]
        first_selected = selected[:, :, :, 1:4]
        last_selected = selected[:, :, :, 8:11]
        span_support = 1.0 - support_jaccard_similarity(
            first_ids[..., :, None, :], last_ids[..., None, :, :]
        ).mean(axis=(-2, -1))
        span_selected = 1.0 - selected_jaccard_similarity(
            first_ids[..., :, None, :],
            first_selected[..., :, None, :],
            last_ids[..., None, :, :],
            last_selected[..., None, :, :],
        ).mean(axis=(-2, -1))
        action_span_soft[start:stop] = aggregate_layer_flow(
            span_soft, layer_groups, probability_flows
        )
        action_span_support[start:stop] = aggregate_layer_flow(
            span_support, layer_groups, probability_flows
        )
        action_span_selected[start:stop] = aggregate_layer_flow(
            span_selected, layer_groups, probability_flows
        )

        current_episode = episode[start:stop]
        if history_probability is None:
            combined_probability = probability
            combined_ids = ids
            combined_selected = selected
            combined_episode = current_episode
            history_length = 0
        else:
            combined_probability = np.concatenate([history_probability, probability])
            combined_ids = np.concatenate([history_ids, ids])
            combined_selected = np.concatenate([history_selected, selected])
            combined_episode = np.concatenate([history_episode, current_episode])
            history_length = len(history_episode)
        positions = history_length + np.arange(stop - start)

        previous = positions - 1
        valid_previous = (previous >= 0) & (
            combined_episode[positions] == combined_episode[np.maximum(previous, 0)]
        )
        if np.any(valid_previous):
            take = np.flatnonzero(valid_previous)
            current_index = positions[take]
            previous_index = previous[take]
            chunk_soft[start + take] = aggregate_cells(
                hellinger(
                    combined_probability[current_index],
                    combined_probability[previous_index],
                ),
                layer_groups,
                probability_flows,
                token_groups,
            )
            chunk_support[start + take] = aggregate_cells(
                1.0
                - support_jaccard_similarity(
                    combined_ids[current_index], combined_ids[previous_index]
                ),
                layer_groups,
                probability_flows,
                token_groups,
            )
            chunk_selected[start + take] = aggregate_cells(
                1.0
                - selected_jaccard_similarity(
                    combined_ids[current_index],
                    combined_selected[current_index],
                    combined_ids[previous_index],
                    combined_selected[previous_index],
                ),
                layer_groups,
                probability_flows,
                token_groups,
            )

        for lag_index, lag in enumerate(lags):
            previous = positions - lag
            valid_lag = (previous >= 0) & (
                combined_episode[positions] == combined_episode[np.maximum(previous, 0)]
            )
            if not np.any(valid_lag):
                continue
            take = np.flatnonzero(valid_lag)
            current_index = positions[take]
            previous_index = previous[take]
            current_probability = combined_probability[current_index, :, -1]
            previous_probability = combined_probability[previous_index, :, -1]
            current_ids = combined_ids[current_index, :, -1]
            previous_ids = combined_ids[previous_index, :, -1]
            current_selected = combined_selected[current_index, :, -1]
            previous_selected = combined_selected[previous_index, :, -1]
            lag_soft[start + take, lag_index] = aggregate_layer_token(
                1.0 - hellinger(current_probability, previous_probability),
                layer_groups,
                token_groups,
            )
            lag_support[start + take, lag_index] = aggregate_layer_token(
                support_jaccard_similarity(current_ids, previous_ids),
                layer_groups,
                token_groups,
            )
            lag_selected[start + take, lag_index] = aggregate_layer_token(
                selected_jaccard_similarity(
                    current_ids,
                    current_selected,
                    previous_ids,
                    previous_selected,
                ),
                layer_groups,
                token_groups,
            )

        history_probability = combined_probability[-max_lag:].copy()
        history_ids = combined_ids[-max_lag:].copy()
        history_selected = combined_selected[-max_lag:].copy()
        history_episode = combined_episode[-max_lag:].copy()

    raw = np.full((total, len(names)), np.nan, dtype=np.float32)
    name_index = {name: index for index, name in enumerate(names)}
    raw[:, name_index["instability_soft_flow"]] = quantile_flat(
        flow_soft[:, :, 2, 1:], 0.75
    )
    raw[:, name_index["instability_acceleration"]] = quantile_flat(
        acceleration[:, :, 2, 1:], 0.75
    )
    raw[:, name_index["instability_selected_flow_switch"]] = quantile_flat(
        flow_selected[:, :, 2, 1:], 0.75
    )
    raw[:, name_index["instability_chunk_selected_switch"]] = quantile_flat(
        chunk_selected[:, :, 2, 1:], 0.75
    )
    raw[:, name_index["decoupling_front_state_action_gap"]] = quantile_flat(
        state_action_soft[:, 0, 2, :], 0.75
    )
    raw[:, name_index["decoupling_back_state_action_gap"]] = quantile_flat(
        state_action_soft[:, 1, 2, :], 0.75
    )
    state_jump = np.asarray(chunk_soft[:, :, 2, 0], dtype=np.float32)
    action_values = np.asarray(chunk_soft[:, :, 2, 1:], dtype=np.float32)
    action_jump = np.full(action_values.shape[:2], np.nan, dtype=np.float32)
    valid_action = np.isfinite(action_values).any(axis=-1)
    action_jump[valid_action] = np.nanmedian(
        action_values[valid_action], axis=-1
    )
    raw[:, name_index["decoupling_chunk_jump_imbalance"]] = quantile_flat(
        np.abs(state_jump - action_jump), 0.75
    )
    raw[:, name_index["decoupling_action_span_selected_gap"]] = quantile_flat(
        action_span_selected[:, :, 2], 0.75
    )
    raw[:, name_index["lock_low_flow_support_switch"]] = quantile_flat(
        flow_support[:, :, 2, 1:], 0.25
    )
    raw[:, name_index["lock_low_action_span_support_gap"]] = quantile_flat(
        action_span_support[:, :, 2], 0.25
    )
    for lag_index, lag in enumerate(lags):
        raw[:, name_index[f"lock_soft_recurrence_lag{lag}"]] = quantile_flat(
            lag_soft[:, lag_index, :, 1:], 0.75
        )
        raw[:, name_index[f"lock_support_recurrence_lag{lag}"]] = quantile_flat(
            lag_support[:, lag_index, :, 1:], 0.75
        )
        raw[:, name_index[f"lock_selected_recurrence_lag{lag}"]] = quantile_flat(
            lag_selected[:, lag_index, :, 1:], 0.75
        )

    query = np.empty(total, dtype=np.int16)
    evidence = np.full_like(raw, np.nan)
    ordinary_names = names[:10]
    low_names = {
        "lock_low_flow_support_switch",
        "lock_low_action_span_support_gap",
    }
    baseline_queries = np.asarray(
        config["causal_self_reference"]["ordinary_baseline_queries"], dtype=np.int64
    )
    ordinary_first = int(
        config["causal_self_reference"]["ordinary_first_scored_query"]
    )
    recurrence_observations = int(
        config["causal_self_reference"]["recurrence_baseline_observations_per_lag"]
    )
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode == episode_id)
        if len(indices) > 1 and np.any(np.diff(indices) != 1):
            raise ValueError(f"non-contiguous episode: {task}/{run_name}/{episode_id}")
        local_query = np.arange(len(indices), dtype=np.int16)
        query[indices] = local_query
        ordinary_base = baseline_queries[baseline_queries < len(indices)]
        score = indices[local_query >= ordinary_first]
        for name in ordinary_names:
            column = name_index[name]
            baseline = np.nanmedian(raw[indices[ordinary_base], column])
            if not np.isfinite(baseline):
                continue
            if name in low_names:
                evidence[score, column] = baseline - raw[score, column]
            else:
                evidence[score, column] = raw[score, column] - baseline
        for lag in lags:
            base_query = np.arange(lag, lag + recurrence_observations)
            base_query = base_query[base_query < len(indices)]
            score = indices[local_query >= lag + recurrence_observations]
            for view in ("soft", "support", "selected"):
                name = f"lock_{view}_recurrence_lag{lag}"
                column = name_index[name]
                baseline = np.nanmedian(raw[indices[base_query], column])
                if np.isfinite(baseline):
                    evidence[score, column] = raw[score, column] - baseline

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        schema=np.asarray(SCHEMA),
        task=np.asarray(task),
        run=np.asarray(str(run)),
        run_name=np.asarray(run_name),
        feature_fingerprint=np.asarray(fingerprint),
        training=np.asarray(False),
        endpoint_labels_loaded=np.asarray(False),
        episode=episode,
        query=query,
        component_names=np.asarray(names),
        component_raw=raw,
        component_evidence=evidence,
        flow_soft_mobility=flow_soft,
        flow_support_switch=flow_support,
        flow_selected_switch=flow_selected,
        flow_acceleration=acceleration,
        gate_entropy=entropy,
        top1_top2_margin=margin,
        state_action_soft_gap=state_action_soft,
        state_action_support_gap=state_action_support,
        state_action_selected_gap=state_action_selected,
        action_span_soft_gap=action_span_soft,
        action_span_support_gap=action_span_support,
        action_span_selected_gap=action_span_selected,
        chunk_soft_jump=chunk_soft,
        chunk_support_switch=chunk_support,
        chunk_selected_switch=chunk_selected,
        lag_soft_recurrence=lag_soft,
        lag_support_recurrence=lag_support,
        lag_selected_recurrence=lag_selected,
    )
    return {
        "run_name": run_name,
        "task": task,
        "cache": str(cache),
        "route_rows": total,
        "episodes": len(unique_episodes),
        "reused": False,
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def load_outcomes(run: Path) -> dict[int, bool]:
    rows = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    return {int(row["episode_index"]): bool(row["success"]) for row in rows}


def episode_indices(episode: np.ndarray) -> Iterable[tuple[int, np.ndarray]]:
    for episode_id in np.unique(episode):
        yield int(episode_id), np.flatnonzero(episode == episode_id)


def query_bins(query: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges[1:], np.asarray(query), side="right").astype(np.int8)


def build_healthy_reference(
    cache_root: Path,
    run_name: str,
    run_map: dict[str, Path],
    reference_tasks: list[str],
    config: dict[str, Any],
) -> tuple[dict[tuple[int, int], np.ndarray], dict[str, Any]]:
    names = component_names(
        [int(value) for value in config["position_grid"]["recurrence_lags"]]
    )
    edges = np.asarray(config["healthy_cdf_query_bins"], dtype=np.int64)
    collected: dict[tuple[int, int], list[np.ndarray]] = {
        (component, bin_index): []
        for component in range(len(names))
        for bin_index in range(len(edges) - 1)
    }
    success_episodes = 0
    success_rows = 0
    for task in reference_tasks:
        arrays = load_cache(task_cache_path(cache_root, run_name, task))
        outcomes = load_outcomes(run_map[task])
        success = np.asarray([episode for episode, value in outcomes.items() if value])
        success_episodes += len(success)
        selected = np.isin(arrays["episode"], success)
        evidence = np.asarray(arrays["component_evidence"], dtype=np.float32)
        bins = query_bins(arrays["query"], edges)
        success_rows += int(selected.sum())
        for bin_index in range(len(edges) - 1):
            rows = selected & (bins == bin_index)
            if not np.any(rows):
                continue
            for component in range(len(names)):
                values = evidence[rows, component]
                values = values[np.isfinite(values)]
                if len(values):
                    collected[(component, bin_index)].append(values)
    reference: dict[tuple[int, int], np.ndarray] = {}
    global_values: dict[int, np.ndarray] = {}
    counts: dict[str, int] = {}
    for component, name in enumerate(names):
        parts = [
            value
            for bin_index in range(len(edges) - 1)
            for value in collected[(component, bin_index)]
        ]
        global_values[component] = (
            np.sort(np.concatenate(parts).astype(np.float32))
            if parts
            else np.empty(0, dtype=np.float32)
        )
        counts[f"{name}:all"] = len(global_values[component])
        for bin_index in range(len(edges) - 1):
            values = collected[(component, bin_index)]
            joined = (
                np.sort(np.concatenate(values).astype(np.float32))
                if values
                else np.empty(0, dtype=np.float32)
            )
            reference[(component, bin_index)] = joined
            counts[f"{name}:bin{bin_index}"] = len(joined)
        reference[(component, -1)] = global_values[component]
    return reference, {
        "tasks": len(reference_tasks),
        "success_episodes": success_episodes,
        "success_route_rows": success_rows,
        "counts": counts,
    }


def empirical_upper_confidence(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    good = np.isfinite(values)
    if not len(reference) or not np.any(good):
        return output
    left = np.searchsorted(reference, values[good], side="left")
    output[good] = left / (len(reference) + 1.0)
    return output


def kth_highest(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    ordered = np.sort(safe, axis=-1)
    output = ordered[..., -k].astype(np.float32)
    output[finite.sum(axis=-1) < k] = np.nan
    return output


def finite_max(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    output = np.max(np.where(finite, values, -np.inf), axis=-1).astype(np.float32)
    output[~finite.any(axis=-1)] = np.nan
    return output


def persistent_two_of_three(
    values: np.ndarray, episode: np.ndarray, query: np.ndarray
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype=np.float32)
    for _episode, indices in episode_indices(episode):
        local_query = query[indices]
        local_values = values[indices]
        for offset in range(2, len(indices)):
            if np.array_equal(
                local_query[offset - 2 : offset + 1],
                np.arange(local_query[offset] - 2, local_query[offset] + 1),
            ):
                output[indices[offset]] = kth_highest(
                    local_values[offset - 2 : offset + 1][None], 2
                )[0]
    return output


def score_cache(
    arrays: dict[str, np.ndarray],
    reference: dict[tuple[int, int], np.ndarray],
    config: dict[str, Any],
) -> pd.DataFrame:
    names = [str(value) for value in arrays["component_names"]]
    name_index = {name: index for index, name in enumerate(names)}
    evidence = np.asarray(arrays["component_evidence"], dtype=np.float32)
    query = np.asarray(arrays["query"], dtype=np.int16)
    episode = np.asarray(arrays["episode"], dtype=np.int32)
    edges = np.asarray(config["healthy_cdf_query_bins"], dtype=np.int64)
    bins = query_bins(query, edges)
    minimum = int(config["minimum_reference_values_per_bin"])
    confidence = np.full_like(evidence, np.nan)
    for component in range(len(names)):
        for bin_index in range(len(edges) - 1):
            rows = bins == bin_index
            values = reference[(component, bin_index)]
            if len(values) < minimum:
                values = reference[(component, -1)]
            confidence[rows, component] = empirical_upper_confidence(
                values, evidence[rows, component]
            )

    instability_components = [
        name_index["instability_soft_flow"],
        name_index["instability_acceleration"],
        name_index["instability_selected_flow_switch"],
        name_index["instability_chunk_selected_switch"],
    ]
    decoupling_components = [
        name_index["decoupling_front_state_action_gap"],
        name_index["decoupling_back_state_action_gap"],
        name_index["decoupling_chunk_jump_imbalance"],
        name_index["decoupling_action_span_selected_gap"],
    ]
    instability = kth_highest(confidence[:, instability_components], 3)
    decoupling_gap = finite_max(confidence[:, decoupling_components[:2]])
    decoupling_confirmation = finite_max(confidence[:, decoupling_components[2:]])
    decoupling = np.minimum(decoupling_gap, decoupling_confirmation)
    lags = [int(value) for value in config["position_grid"]["recurrence_lags"]]
    lock_by_lag = np.full((len(query), len(lags)), np.nan, dtype=np.float32)
    low_components = [
        name_index["lock_low_flow_support_switch"],
        name_index["lock_low_action_span_support_gap"],
    ]
    for lag_index, lag in enumerate(lags):
        recurrence_components = [
            name_index[f"lock_soft_recurrence_lag{lag}"],
            name_index[f"lock_support_recurrence_lag{lag}"],
            name_index[f"lock_selected_recurrence_lag{lag}"],
        ]
        recurrence_confirmation = kth_highest(
            confidence[:, recurrence_components], 2
        )
        collapse_confirmation = finite_max(confidence[:, low_components])
        lock_by_lag[:, lag_index] = np.minimum(
            recurrence_confirmation, collapse_confirmation
        )
    lock_finite = np.isfinite(lock_by_lag)
    lock = np.max(np.where(lock_finite, lock_by_lag, -np.inf), axis=1)
    lock[~lock_finite.any(axis=1)] = np.nan
    winning_lag = np.full(len(query), -1, dtype=np.int8)
    has_lock = lock_finite.any(axis=1)
    winning_lag[has_lock] = np.asarray(lags, dtype=np.int8)[
        np.argmax(np.where(lock_finite, lock_by_lag, -np.inf), axis=1)[has_lock]
    ]

    raw_heads = {
        "lock_in": lock,
        "instability": instability,
        "state_action_decoupling": decoupling,
    }
    persistent = {
        name: persistent_two_of_three(values, episode, query)
        for name, values in raw_heads.items()
    }
    persistent_matrix = np.column_stack([persistent[name] for name in HEADS])
    finite = np.isfinite(persistent_matrix)
    combined = np.max(np.where(finite, persistent_matrix, -np.inf), axis=1)
    combined[~finite.any(axis=1)] = np.nan
    frame = pd.DataFrame(
        {
            "episode": episode,
            "query": query,
            "lock_in_raw": lock,
            "lock_in_lag": winning_lag,
            "instability_raw": instability,
            "state_action_decoupling_raw": decoupling,
            "lock_in": persistent["lock_in"],
            "instability": persistent["instability"],
            "state_action_decoupling": persistent["state_action_decoupling"],
            "combined": combined,
        }
    )
    return frame


def select_threshold(maxima: np.ndarray, budget: float) -> tuple[float, int, int]:
    maxima = np.asarray(maxima, dtype=np.float64)
    denominator = len(maxima)
    finite = maxima[np.isfinite(maxima)]
    if not denominator or not len(finite):
        raise ValueError("no successful-episode scores for threshold confirmation")
    values, counts = np.unique(finite, return_counts=True)
    counts_at_or_above = np.cumsum(counts[::-1])[::-1]
    allowed = int(math.floor(budget * denominator + 1e-12))
    candidates = np.flatnonzero(counts_at_or_above <= allowed)
    if not len(candidates):
        return float(np.nextafter(values[-1], np.inf)), 0, denominator
    index = int(candidates[0])
    return float(values[index]), int(counts_at_or_above[index]), denominator


def score_episode_maxima(
    frame: pd.DataFrame,
    episode_ids: Iterable[int],
) -> dict[str, list[float]]:
    maxima = {rule: [] for rule in RULES}
    grouped = {int(key): group for key, group in frame.groupby("episode", sort=False)}
    for episode_id in episode_ids:
        group = grouped.get(int(episode_id))
        for rule in RULES:
            if group is None:
                maxima[rule].append(float("-inf"))
                continue
            values = group[rule].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            maxima[rule].append(float(values.max()) if len(values) else float("-inf"))
    return maxima


def select_clock_query(lengths: np.ndarray, budget: float, minimum: int = 0) -> tuple[int, int]:
    lengths = np.asarray(lengths, dtype=np.int64)
    allowed = int(math.floor(budget * len(lengths) + 1e-12))
    for query in range(minimum, int(lengths.max()) + 1):
        alarms = int(np.sum(lengths > query))
        if alarms <= allowed:
            return query, alarms
    return int(lengths.max()), 0


def add_alarm_columns(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    for rule in RULES:
        output[f"alarm_{rule}"] = output[rule] >= thresholds[rule]
        output[f"first_alarm_{rule}"] = False
    for _episode, indices in episode_indices(output["episode"].to_numpy()):
        for rule in RULES:
            hits = indices[output.loc[indices, f"alarm_{rule}"].to_numpy(dtype=bool)]
            if len(hits):
                output.loc[hits[0], f"first_alarm_{rule}"] = True
    return output


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


def matched_clock(episodes: pd.DataFrame, fp_budget: int) -> tuple[int, dict[str, Any]]:
    maximum = int(episodes["episode_length"].max())
    failure = episodes["failure"].to_numpy(dtype=bool)
    lengths = episodes["episode_length"].to_numpy(dtype=np.int64)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for query in range(maximum + 1):
        metrics = rate_metrics(lengths > query, failure)
        if metrics["fp"] <= fp_budget:
            candidates.append((query, metrics))
    if not candidates:
        query = maximum
        return query, rate_metrics(lengths > query, failure)
    return max(candidates, key=lambda item: (item[1]["tp"], -item[0]))


def build_episode_table(
    predictions: pd.DataFrame,
    cache_root: Path,
    run_name: str,
    run_map: dict[str, Path],
    partition: pd.DataFrame,
    clock_query: int,
) -> pd.DataFrame:
    role_by_task = partition.set_index("task")["role"].to_dict()
    rows: list[dict[str, Any]] = []
    for task, run in sorted(run_map.items()):
        arrays = load_cache(task_cache_path(cache_root, run_name, task))
        outcomes = load_outcomes(run)
        task_predictions = predictions[predictions["task"] == task]
        grouped = {
            int(key): group for key, group in task_predictions.groupby("episode", sort=False)
        }
        for episode_id, indices in episode_indices(arrays["episode"]):
            group = grouped.get(episode_id)
            row: dict[str, Any] = {
                "task": task,
                "suite": task.split("/", 1)[0],
                "role": role_by_task[task],
                "episode": episode_id,
                "episode_length": len(indices),
                "success": outcomes[episode_id],
                "failure": not outcomes[episode_id],
                "clock_alarm": len(indices) > clock_query,
                "clock_query": clock_query,
            }
            for rule in RULES:
                hits = (
                    group[group[f"first_alarm_{rule}"]]
                    if group is not None
                    else pd.DataFrame()
                )
                alarm = not hits.empty
                row[f"alarm_{rule}"] = alarm
                row[f"first_alarm_query_{rule}"] = (
                    int(hits.iloc[0]["query"]) if alarm else np.nan
                )
                if rule == "combined" and alarm:
                    first = hits.iloc[0]
                    head_values = {head: float(first[head]) for head in HEADS}
                    row["alarm_cause"] = max(head_values, key=head_values.get)
                    row["first_alarm_phase"] = int(first["query"]) / max(len(indices) - 1, 1)
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_onsets(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    onset_path: Path,
    matched_clock_query: int,
    minimum_query: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    onset = pd.read_csv(onset_path)
    onset = onset[onset["onset_query"].notna()].copy()
    heldout = episodes[episodes["role"] == "heldout_test"]
    onset = onset.merge(
        heldout[["task", "episode", "failure"]],
        on=["task", "episode"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_evaluated"),
    )
    onset = onset[onset["onset_query"] >= minimum_query].copy()
    records: list[dict[str, Any]] = []
    for event in onset.itertuples(index=False):
        group = predictions[
            (predictions["task"] == event.task)
            & (predictions["episode"] == int(event.episode))
        ]
        row: dict[str, Any] = {
            "task": event.task,
            "episode": int(event.episode),
            "onset_query": int(event.onset_query),
            "onset_type": event.onset_type,
        }
        for rule in RULES:
            queries = group.loc[group[f"alarm_{rule}"], "query"].to_numpy(dtype=int)
            first = int(queries[0]) if len(queries) else None
            row[f"first_alarm_query_{rule}"] = first
            row[f"delta_{rule}"] = None if first is None else first - int(event.onset_query)
            row[f"near_active_{rule}"] = bool(
                np.any((queries >= event.onset_query - 2) & (queries <= event.onset_query))
            )
        for name, query in (("clock", int(heldout["clock_query"].iloc[0])), ("matched_clock", matched_clock_query)):
            active = int(event.episode_length) > query
            row[f"first_alarm_query_{name}"] = query if active else None
            row[f"delta_{name}"] = query - int(event.onset_query) if active else None
            row[f"near_active_{name}"] = bool(
                active and event.onset_query - 2 <= query <= event.onset_query
            )
        records.append(row)
    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {"eligible_events": len(frame), "minimum_onset_query": minimum_query}
    for rule in (*RULES, "clock", "matched_clock"):
        delta = pd.to_numeric(frame[f"delta_{rule}"], errors="coerce")
        summary[rule] = {
            "alarm_events": int(delta.notna().sum()),
            "not_later_than_onset": int((delta <= 0).sum()),
            "near_first_minus2_to_0": int(delta.between(-2, 0).sum()),
            "near_active_minus2_to_0": int(frame[f"near_active_{rule}"].sum()),
            "too_early": int((delta < -2).sum()),
            "late": int((delta > 0).sum()),
        }
    return frame, summary


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


def render_report(summary: dict[str, Any], metrics: pd.DataFrame) -> str:
    combined = summary["combined_heldout"]
    clock = summary["combined_heldout_clock"]
    lines = [
        "# 结构化 MoE 三机制报警双向冻结实验",
        "",
        "## 核心结果",
        "",
        (
            f"两次跨 seed、跨任务留出合计 {combined['episodes']} 条轨迹，失败 "
            f"{combined['failures']} 条。结构化 MoE 为 TP={combined['tp']}、FP={combined['fp']}、"
            f"FN={combined['fn']}、TN={combined['tn']}。预冻结时间钟为 TP={clock['tp']}、"
            f"FP={clock['fp']}、FN={clock['fn']}、TN={clock['tn']}。"
        ),
        "",
        "判断器没有训练权重。每个方向用 24 个健康参考任务建立经验 CDF，另用 8 个任务的成功轨迹"
        "分别确认 1% 整轨迹误报阈值，并取八个逐任务阈值的最大值，再将规则冻结到另一 seed 组的 "
        "8 个未参与参考或阈值确认的任务。"
        "目标预测先落盘并哈希，随后才读取目标 success/failure。",
        "",
        "## 使用的 MoE 结构",
        "",
        "- 2 个 layer stage x 3 个 flow stage x 4 个 token group，共 24 个位置 cell；",
        "- full-softmax Hellinger、实际 top-4 expert support Jaccard、实际 selected-weight Jaccard；",
        "- lag 1--4 分开建立完全对称的自基线；",
        "- 锁死、抖动、state-action 脱节三个机制头均要求多项证据共同异常，并要求最近 3 次中至少 2 次持续。",
        "",
        "gate entropy 与 top1-top2 margin 被保存为上下文，但没有单独触发报警。AS 没有进入判断器。",
        "",
        "## 双向留出与逐头消融",
        "",
        markdown_table(metrics),
        "",
        "## 解释边界",
        "",
        "- endpoint failure 包含多种失败，不等于单一 Trap 类型；",
        "- 健康 CDF 和阈值只使用成功轨迹，但特征分组来自此前机制探索，因此属于严格回放，不是全新 prospective 发现；",
        "- 判断器运行时只读取 MoE 路由和当前 episode 历史，物理 onset 只在预测冻结后用于时序审计；",
        "- 是否优于旧规则必须同时比较 TP/FP、等误报时间钟以及首次报警相对 onset 的位置。",
        "",
    ]
    return "\n".join(lines)


def self_test() -> None:
    probability = np.asarray([[0.6, 0.2, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])
    assert np.allclose(hellinger(probability, probability), 0.0, atol=1e-6)
    ids = np.asarray([[1, 2, 3, 4], [1, 2, 5, 6]])
    assert np.allclose(support_jaccard_similarity(ids, ids), 1.0)
    assert np.isclose(support_jaccard_similarity(ids[:1], ids[1:])[0], 2 / 6)
    weight = np.ones_like(ids, dtype=np.float32)
    assert np.allclose(selected_jaccard_similarity(ids, weight, ids, weight), 1.0)
    values = np.asarray([[0.1, 0.9, 0.8, 0.7], [np.nan, 0.9, 0.8, 0.7]])
    assert np.allclose(kth_highest(values, 3), [0.7, 0.7])
    threshold, alarms, total = select_threshold(
        np.arange(100, dtype=np.float64), 0.01
    )
    assert threshold == 99 and alarms == 1 and total == 100
    print("self-test passed")


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("training") is not False or config.get("learned_feature_weights") is not False:
        raise ValueError("experiment must remain train-free")
    output = args.output.resolve()
    cache_root = output / "route_only_tasks"
    table_root = output / "tables"
    prediction_root = output / "predictions_label_free"
    for path in (output, cache_root, table_root, prediction_root, output / "figures"):
        path.mkdir(parents=True, exist_ok=True)

    raw_root = resolve_workspace(config["cache_root"])
    run_settings = {str(row["name"]): row for row in config["runs"]}
    run_maps: dict[str, dict[str, Path]] = {}
    for name, setting in run_settings.items():
        runs = discover_runs(raw_root, str(setting["run_id"]))
        if len(runs) != int(config["expected_tasks_per_run"]):
            raise ValueError(f"{name}: expected 40 tasks, found {len(runs)}")
        run_maps[name] = dict(runs)
    tasks = sorted(next(iter(run_maps.values())))
    if any(sorted(run_map) != tasks for run_map in run_maps.values()):
        raise ValueError("two runs do not contain identical task sets")
    partition = partition_tasks(tasks, config)
    partition.to_csv(table_root / "task_partition.csv", index=False)

    extraction: list[dict[str, Any]] = []
    jobs = [
        (task, run_name, run)
        for run_name, run_map in run_maps.items()
        for task, run in run_map.items()
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                extract_task_features,
                task,
                run_name,
                str(run),
                str(cache_root),
                config,
                args.rebuild,
            ): (run_name, task)
            for task, run_name, run in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            extraction.append(result)
            print(
                f"[extract {index}/{len(futures)}] {result['run_name']} {result['task']} "
                f"rows={result['route_rows']} reused={result['reused']}",
                flush=True,
            )
    extraction.sort(key=lambda row: (row["run_name"], row["task"]))
    write_json(
        output / "route_only_manifest.json",
        {
            "schema": SCHEMA,
            "training": False,
            "endpoint_labels_loaded": False,
            "config_sha256": sha256_file(args.config),
            "task_partition_sha256": sha256_file(table_root / "task_partition.csv"),
            "feature_fingerprint": feature_fingerprint(config),
            "route_rows": sum(row["route_rows"] for row in extraction),
            "tasks": extraction,
        },
    )

    reference_tasks = partition.loc[
        partition["role"] == "healthy_reference", "task"
    ].astype(str).tolist()
    threshold_tasks = partition.loc[
        partition["role"] == "threshold_confirmation", "task"
    ].astype(str).tolist()
    heldout_tasks = set(
        partition.loc[partition["role"] == "heldout_test", "task"].astype(str)
    )
    run_names = list(run_settings)
    directions = [(run_names[0], run_names[1]), (run_names[1], run_names[0])]
    metric_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    all_onset_rows: list[pd.DataFrame] = []

    for source_name, target_name in directions:
        direction = f"{source_name}_to_{target_name}"
        print(f"[{direction}] building healthy reference", flush=True)
        reference, reference_summary = build_healthy_reference(
            cache_root,
            source_name,
            run_maps[source_name],
            reference_tasks,
            config,
        )
        reference_payload: dict[str, np.ndarray] = {
            "schema": np.asarray(SCHEMA),
            "component_names": np.asarray(component_names([1, 2, 3, 4])),
            "query_bin_edges": np.asarray(config["healthy_cdf_query_bins"]),
        }
        for (component, bin_index), values in reference.items():
            reference_payload[f"component_{component}_bin_{bin_index}"] = values
        reference_path = output / f"healthy_reference_{direction}.npz"
        np.savez_compressed(reference_path, **reference_payload)

        threshold_maxima = {rule: [] for rule in RULES}
        threshold_maxima_by_task = {rule: {} for rule in RULES}
        threshold_success_lengths: list[int] = []
        threshold_success_episodes = 0
        for task in threshold_tasks:
            arrays = load_cache(task_cache_path(cache_root, source_name, task))
            frame = score_cache(arrays, reference, config)
            outcomes = load_outcomes(run_maps[source_name][task])
            successes = [episode for episode, success in outcomes.items() if success]
            threshold_success_episodes += len(successes)
            maxima = score_episode_maxima(frame, successes)
            for rule in RULES:
                threshold_maxima[rule].extend(maxima[rule])
                threshold_maxima_by_task[rule][task] = maxima[rule]
            lengths = {
                episode_id: len(indices)
                for episode_id, indices in episode_indices(arrays["episode"])
            }
            threshold_success_lengths.extend(lengths[episode] for episode in successes)

        budget = float(
            config["threshold_confirmation"]["target_success_episode_false_alarm_rate"]
        )
        thresholds: dict[str, float] = {}
        threshold_counts: dict[str, Any] = {}
        task_threshold_rows: list[dict[str, Any]] = []
        for rule in RULES:
            pooled_threshold, pooled_alarms, pooled_total = select_threshold(
                np.asarray(threshold_maxima[rule]), budget
            )
            task_thresholds: list[float] = []
            for task in threshold_tasks:
                task_threshold, task_alarms, task_total = select_threshold(
                    np.asarray(threshold_maxima_by_task[rule][task]), budget
                )
                task_thresholds.append(task_threshold)
                task_threshold_rows.append(
                    {
                        "direction": direction,
                        "rule": rule,
                        "task": task,
                        "threshold": task_threshold,
                        "success_alarm_episodes": task_alarms,
                        "success_episodes": task_total,
                    }
                )
            threshold = max(task_thresholds)
            robust_values = np.asarray(threshold_maxima[rule], dtype=np.float64)
            alarms = int(np.sum(robust_values >= threshold))
            total = len(robust_values)
            thresholds[rule] = threshold
            threshold_counts[rule] = {
                "threshold": threshold,
                "success_alarm_episodes": alarms,
                "success_episodes": total,
                "selection": "maximum_task_specific_threshold",
                "pooled_threshold_for_audit": pooled_threshold,
                "pooled_success_alarm_episodes_for_audit": pooled_alarms,
                "pooled_success_episodes_for_audit": pooled_total,
            }
        pd.DataFrame(task_threshold_rows).to_csv(
            table_root / f"task_specific_thresholds_{direction}.csv", index=False
        )
        clock_query, clock_alarms = select_clock_query(
            np.asarray(threshold_success_lengths), budget, minimum=6
        )
        threshold_payload = {
            "schema": SCHEMA,
            "training": False,
            "failure_labels_used": False,
            "source_run": source_name,
            "target_run": target_name,
            "reference": reference_summary,
            "threshold_confirmation_tasks": threshold_tasks,
            "threshold_confirmation_success_episodes": threshold_success_episodes,
            "target_success_episode_fpr": budget,
            "threshold_selection": "maximum_task_specific_threshold",
            "rules": threshold_counts,
            "clock_query": clock_query,
            "clock_success_alarm_episodes": clock_alarms,
            "healthy_reference_sha256": sha256_file(reference_path),
        }
        threshold_path = output / f"thresholds_{direction}.json"
        write_json(threshold_path, threshold_payload)

        prediction_parts: list[pd.DataFrame] = []
        for task in tasks:
            arrays = load_cache(task_cache_path(cache_root, target_name, task))
            frame = add_alarm_columns(score_cache(arrays, reference, config), thresholds)
            frame = frame[np.isfinite(frame["combined"])].copy()
            frame.insert(0, "task", task)
            prediction_parts.append(frame)
        predictions = pd.concat(prediction_parts, ignore_index=True)
        prediction_path = prediction_root / f"{direction}.csv.gz"
        predictions.to_csv(
            prediction_path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        prediction_sha = sha256_file(prediction_path)
        write_json(
            prediction_root / f"{direction}_manifest.json",
            {
                "schema": SCHEMA,
                "training": False,
                "target_outcomes_loaded": False,
                "predictions_written_before_target_outcomes": True,
                "source_run": source_name,
                "target_run": target_name,
                "rows": len(predictions),
                "prediction_sha256": prediction_sha,
                "threshold_sha256": sha256_file(threshold_path),
                "heldout_tasks": sorted(heldout_tasks),
            },
        )
        print(f"[{direction}] predictions frozen {prediction_sha}", flush=True)

        episodes = build_episode_table(
            predictions,
            cache_root,
            target_name,
            run_maps[target_name],
            partition,
            clock_query,
        )
        episodes.insert(0, "direction", direction)
        episodes.to_csv(table_root / f"episode_flip_{direction}.csv", index=False)
        heldout = episodes[episodes["role"] == "heldout_test"].copy()
        direction_metrics: dict[str, Any] = {}
        for scope_name, scope in (("heldout", heldout), ("all_target", episodes)):
            for rule in RULES:
                values = rate_metrics(scope[f"alarm_{rule}"], scope["failure"])
                metric_rows.append(
                    {
                        "direction": direction,
                        "scope": scope_name,
                        "rule": rule,
                        "episodes": len(scope),
                        "failures": int(scope["failure"].sum()),
                        **values,
                    }
                )
                direction_metrics[f"{scope_name}_{rule}"] = values
        clock_metrics = rate_metrics(heldout["clock_alarm"], heldout["failure"])
        metric_rows.append(
            {
                "direction": direction,
                "scope": "heldout",
                "rule": "prefrozen_clock",
                "episodes": len(heldout),
                "failures": int(heldout["failure"].sum()),
                **clock_metrics,
            }
        )
        combined_metrics = direction_metrics["heldout_combined"]
        matched_query, matched_metrics = matched_clock(heldout, combined_metrics["fp"])
        metric_rows.append(
            {
                "direction": direction,
                "scope": "heldout",
                "rule": "posthoc_fp_bounded_clock",
                "episodes": len(heldout),
                "failures": int(heldout["failure"].sum()),
                **matched_metrics,
            }
        )

        onset_summary: dict[str, Any] = {"available": False}
        onset_setting = run_settings[target_name].get("posthoc_onset_proxy")
        if onset_setting:
            onset_frame, onset_summary = evaluate_onsets(
                predictions,
                episodes,
                resolve_workspace(onset_setting),
                matched_query,
            )
            onset_frame.insert(0, "direction", direction)
            all_onset_rows.append(onset_frame)
        summaries[direction] = {
            "source_run": source_name,
            "target_run": target_name,
            "prediction_sha256": prediction_sha,
            "thresholds": threshold_counts,
            "clock_query": clock_query,
            "matched_clock_query": matched_query,
            "heldout_episodes": len(heldout),
            "heldout_failures": int(heldout["failure"].sum()),
            "heldout_metrics": direction_metrics,
            "heldout_clock": clock_metrics,
            "heldout_matched_clock": matched_metrics,
            "onset": onset_summary,
        }

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_root / "direction_and_head_metrics.csv", index=False)
    if all_onset_rows:
        pd.concat(all_onset_rows, ignore_index=True).to_csv(
            table_root / "heldout_onset_timing.csv", index=False
        )

    primary = metrics[(metrics["scope"] == "heldout") & (metrics["rule"] == "combined")]
    primary_clock = metrics[
        (metrics["scope"] == "heldout") & (metrics["rule"] == "prefrozen_clock")
    ]

    def combine_rows(frame: pd.DataFrame) -> dict[str, Any]:
        counts = {
            key: int(frame[key].sum())
            for key in ("episodes", "failures", "tp", "fp", "fn", "tn")
        }
        counts.update(
            {
                "precision": counts["tp"] / max(counts["tp"] + counts["fp"], 1),
                "failure_recall": counts["tp"] / max(counts["failures"], 1),
                "success_false_alarm_rate": counts["fp"]
                / max(counts["fp"] + counts["tn"], 1),
            }
        )
        return counts

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "gradient_optimization": False,
        "learned_feature_weights": False,
        "physical_inputs_to_detector": False,
        "failure_labels_used_for_features_or_thresholds": False,
        "runs": run_names,
        "route_queries": sum(row["route_rows"] for row in extraction),
        "task_partition": partition["role"].value_counts().to_dict(),
        "directions": summaries,
        "combined_heldout": combine_rows(primary),
        "combined_heldout_clock": combine_rows(primary_clock),
    }
    write_json(output / "summary.json", summary)
    (output / "REPORT_ZH.md").write_text(
        render_report(summary, metrics[metrics["scope"] == "heldout"]), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    heldout_metrics = metrics[metrics["scope"] == "heldout"].copy()
    for direction_index, direction in enumerate(summaries):
        selected = heldout_metrics[heldout_metrics["direction"] == direction]
        selected = selected.set_index("rule").reindex(
            ["combined", *HEADS, "prefrozen_clock", "posthoc_fp_bounded_clock"]
        )
        x = np.arange(len(selected)) + (direction_index - 0.5) * 0.35
        axes[0].bar(x, selected["tp"], width=0.35, label=direction)
        axes[1].bar(x, selected["fp"], width=0.35, label=direction)
    labels = ["combined", "lock", "instability", "decoupling", "clock", "FP-clock"]
    for axis, title in zip(axes, ("Held-out failures detected", "Held-out success alarms")):
        axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "figures/structured_alarm_metrics.png", dpi=180)
    plt.close(figure)
    print(json.dumps(plain(summary["combined_heldout"]), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
