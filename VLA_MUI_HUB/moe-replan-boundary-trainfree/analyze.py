#!/usr/bin/env python3
"""Train-free analysis of routing and action continuity at replan boundaries.

For every within-episode q -> q+1 transition, compare q:T10 with q+1:T1.
The primary route-change vector has ten fixed coordinates, one per denoising
step.  Each coordinate is the Hellinger distance between either actual Top-4
combine weights or full 32-way router distributions, averaged over the eight
HB layers.  No projection, classifier, regression, learned threshold, or
fitted feature weight is used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.stats import rankdata


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent
DEFAULT_OUTPUT = HERE / "results"

N_LAYERS = 8
N_DENOISE = 10
N_SUFFIX = 11
N_EXPERTS = 32
TOP_K = 4
ACTION_TOKENS = 10
ACTION_DIM = 7
ACTION_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")
PREFIXES = (4, 8)
ROUTE_TYPES = {
    "boundary_t10_to_next_t1": (10, 1, True),
    "same_t1": (1, 1, True),
    "same_t10": (10, 10, True),
    "within_old_t9_to_t10": (9, 10, False),
    "within_new_t1_to_t2": (1, 2, "next"),
}
ROUTE_SCORE_NAMES = (
    "selected_raw",
    "selected_minus_same",
    "selected_minus_edge",
    "soft_raw",
    "soft_minus_same",
    "soft_minus_edge",
)
FAILURE_FEATURES = tuple(
    f"{score}_h{prefix}" for score in ROUTE_SCORE_NAMES for prefix in PREFIXES
) + tuple(f"action_h{prefix}" for prefix in PREFIXES)


@dataclass(frozen=True)
class RunInfo:
    key: str
    suite: str
    task: str
    path: Path
    relative_path: str
    checkpoint_sha256: str
    action_std: np.ndarray


@dataclass
class RunData:
    info: RunInfo
    summaries: list[dict[str, Any]]
    episode_id: np.ndarray
    episode_step: np.ndarray
    control_step: np.ndarray
    pair_starts: np.ndarray
    pair_episode: np.ndarray
    pair_query: np.ndarray
    states: np.ndarray
    seeds: np.ndarray
    failure: np.ndarray
    route: dict[str, np.ndarray]
    action: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default="right-16x32")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_runs(hub: Path, run_id: str) -> list[RunInfo]:
    root = hub / "cache" / "HiMoE-VLA"
    result = []
    for path in sorted(root.glob(f"libero_*/*/{run_id}")):
        required = (
            path / "meta.json",
            path / "client" / "summaries.json",
            path / "client" / "server_metadata.json",
            path / "server" / "routes.zarr",
        )
        if not all(item.exists() for item in required):
            continue
        meta = read_json(path / "meta.json")
        if not bool(meta.get("sampling", {}).get("complete", False)):
            continue
        server = read_json(path / "client" / "server_metadata.json")
        if int(server["n_action_steps"]) != ACTION_TOKENS:
            raise ValueError(f"{path}: expected ten executed action steps")
        suite = str(meta["suite"])
        task = str(meta["task_name"])
        result.append(
            RunInfo(
                key=f"{suite}/{task}",
                suite=suite,
                task=task,
                path=path,
                relative_path=str(path.relative_to(hub)),
                checkpoint_sha256=str(server["checkpoint_sha256"]),
                action_std=np.asarray(server["normalization_action_std"], np.float32),
            )
        )
    if not result:
        raise ValueError(f"no complete {run_id!r} runs found")
    return result


def load_episode_axes(
    run: RunInfo, summaries: list[dict[str, Any]], store: zarr.Group
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episode_id = np.asarray(store["episode_id"][:], np.int32)
    control_step = np.asarray(store["control_step"][:], np.int32)
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries], np.int32)
    expected = np.repeat(np.arange(len(summaries), dtype=np.int32), lengths)
    if not np.array_equal(episode_id, expected):
        raise ValueError(f"{run.key}: server/client episode alignment failed")
    if not np.array_equal(control_step, np.arange(len(expected), dtype=np.int32)):
        raise ValueError(f"{run.key}: control_step is not contiguous")
    episode_step = np.concatenate(
        [np.arange(length, dtype=np.int16) for length in lengths]
    )
    return episode_id, episode_step, control_step


def load_actions(
    run: RunInfo, summaries: list[dict[str, Any]], expected_episode: np.ndarray
) -> np.ndarray:
    parts = []
    for row in summaries:
        episode = int(row["episode_index"])
        path = run.path / "client" / f"episode_{episode:02d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            actions = np.asarray(payload["actions"], np.float32)
        expected_shape = (int(row["inference_calls"]), ACTION_TOKENS, ACTION_DIM)
        if actions.shape != expected_shape:
            raise ValueError(f"{path}: actions {actions.shape} != {expected_shape}")
        parts.append(actions)
    result = np.concatenate(parts, axis=0)
    if len(result) != len(expected_episode):
        raise ValueError(f"{run.key}: action/route row count differs")
    return result


def normalized_dense(ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, np.int64)
    selected = np.maximum(np.asarray(selected, np.float32), 0.0)
    if ids.shape != selected.shape or ids.shape[-1] != TOP_K:
        raise ValueError("invalid selected route tensors")
    total = selected.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("zero-mass Top-4 route site")
    weights = selected / total
    if np.any((ids < 0) | (ids >= N_EXPERTS)):
        raise ValueError("expert id outside [0, 31]")
    if np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0):
        raise ValueError("duplicate expert inside Top-4")
    dense = np.zeros((*ids.shape[:-1], N_EXPERTS), np.float32)
    np.put_along_axis(dense, ids, weights, axis=-1)
    return dense


def compare_routes(
    ids_a: np.ndarray,
    prob_a: np.ndarray,
    ids_b: np.ndarray,
    prob_b: np.ndarray,
) -> dict[str, np.ndarray]:
    a = normalized_dense(ids_a, prob_a)
    b = normalized_dense(ids_b, prob_b)
    hellinger_site = np.sqrt(
        np.maximum(0.5 * np.square(np.sqrt(a) - np.sqrt(b)).sum(axis=-1), 0.0)
    )
    cosine_site = np.sum(a * b, axis=-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12
    )
    equality = ids_a[..., :, None] == ids_b[..., None, :]
    intersection = equality.any(axis=-1).sum(axis=-1)
    jaccard_site = intersection / np.maximum(2 * TOP_K - intersection, 1)
    exact_site = intersection == TOP_K
    return {
        "hellinger10": hellinger_site.mean(axis=1),
        "cosine10": cosine_site.mean(axis=1),
        "top4_jaccard10": jaccard_site.mean(axis=1),
        "top4_exact_fraction10": exact_site.mean(axis=1),
        "whole_top4_exact": exact_site.all(axis=(1, 2)),
    }


def compare_soft_routes(prob_a: np.ndarray, prob_b: np.ndarray) -> np.ndarray:
    a = np.maximum(np.asarray(prob_a, np.float32), 0.0)
    b = np.maximum(np.asarray(prob_b, np.float32), 0.0)
    if a.shape != b.shape or a.shape[-1] != N_EXPERTS:
        raise ValueError("invalid full router probability tensors")
    total_a = a.sum(axis=-1, keepdims=True)
    total_b = b.sum(axis=-1, keepdims=True)
    if np.any(total_a <= 0.0) or np.any(total_b <= 0.0):
        raise ValueError("zero-mass full router site")
    a /= total_a
    b /= total_b
    site = np.sqrt(
        np.maximum(0.5 * np.square(np.sqrt(a) - np.sqrt(b)).sum(axis=-1), 0.0)
    )
    return site.mean(axis=1)


def extract_route_transitions(
    run: RunInfo,
    store: zarr.Group,
    episode_id: np.ndarray,
    pair_starts: np.ndarray,
    batch_size: int,
) -> dict[str, np.ndarray]:
    pair_lookup = np.full(len(episode_id), -1, np.int64)
    pair_lookup[pair_starts] = np.arange(len(pair_starts), dtype=np.int64)
    result: dict[str, np.ndarray] = {}
    for name in ROUTE_TYPES:
        result[f"{name}_hellinger10"] = np.full(
            (len(pair_starts), N_DENOISE), np.nan, np.float32
        )
        result[f"soft_{name}_hellinger10"] = np.full(
            (len(pair_starts), N_DENOISE), np.nan, np.float32
        )
    for metric in (
        "cosine10",
        "top4_jaccard10",
        "top4_exact_fraction10",
    ):
        result[f"boundary_t10_to_next_t1_{metric}"] = np.full(
            (len(pair_starts), N_DENOISE), np.nan, np.float32
        )
    result["boundary_t10_to_next_t1_whole_top4_exact"] = np.zeros(
        len(pair_starts), np.bool_
    )

    ids_array = store["hb_expert_ids"]
    prob_array = store["hb_selected_prob"]
    soft_array = store["hb_router_probs"]
    previous_ids: np.ndarray | None = None
    previous_prob: np.ndarray | None = None
    previous_soft: np.ndarray | None = None
    previous_episode: int | None = None
    total = len(episode_id)

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ids = np.asarray(ids_array[start:stop], np.uint8)
        prob = np.asarray(prob_array[start:stop], np.float32)
        soft = np.asarray(soft_array[start:stop], np.float32)
        if ids.shape[1:] != (N_LAYERS, N_DENOISE, N_SUFFIX, TOP_K):
            raise ValueError(f"{run.key}: unexpected route shape {ids.shape}")
        if soft.shape[1:] != (N_LAYERS, N_DENOISE, N_SUFFIX, N_EXPERTS):
            raise ValueError(f"{run.key}: unexpected soft-route shape {soft.shape}")
        if previous_ids is None:
            joined_ids = ids
            joined_prob = prob
            joined_soft = soft
            joined_episode = episode_id[start:stop]
            base = start
        else:
            joined_ids = np.concatenate((previous_ids[None], ids), axis=0)
            joined_prob = np.concatenate((previous_prob[None], prob), axis=0)
            joined_soft = np.concatenate((previous_soft[None], soft), axis=0)
            joined_episode = np.r_[previous_episode, episode_id[start:stop]]
            base = start - 1

        valid = np.flatnonzero(joined_episode[1:] == joined_episode[:-1])
        destination = pair_lookup[base + valid]
        if len(valid) and np.any(destination < 0):
            raise ValueError(f"{run.key}: adjacent-pair lookup failed")
        for name, (token_a, token_b, row_mode) in ROUTE_TYPES.items():
            if row_mode is True:
                row_a, row_b = valid, valid + 1
            elif row_mode == "next":
                row_a, row_b = valid + 1, valid + 1
            else:
                row_a, row_b = valid, valid
            measured = compare_routes(
                joined_ids[row_a, :, :, token_a, :],
                joined_prob[row_a, :, :, token_a, :],
                joined_ids[row_b, :, :, token_b, :],
                joined_prob[row_b, :, :, token_b, :],
            )
            result[f"{name}_hellinger10"][destination] = measured["hellinger10"]
            result[f"soft_{name}_hellinger10"][destination] = compare_soft_routes(
                joined_soft[row_a, :, :, token_a, :],
                joined_soft[row_b, :, :, token_b, :],
            )
            if name == "boundary_t10_to_next_t1":
                for metric in (
                    "cosine10",
                    "top4_jaccard10",
                    "top4_exact_fraction10",
                ):
                    result[f"{name}_{metric}"][destination] = measured[metric]
                result[f"{name}_whole_top4_exact"][destination] = measured[
                    "whole_top4_exact"
                ]

        previous_ids = ids[-1].copy()
        previous_prob = prob[-1].copy()
        previous_soft = soft[-1].copy()
        previous_episode = int(episode_id[stop - 1])
        print(f"  routes {run.key}: {stop}/{total}", flush=True)

    for name, values in result.items():
        if values.dtype != np.bool_ and not np.all(np.isfinite(values)):
            raise ValueError(f"{run.key}: incomplete {name}")
    return result


def rms(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def action_metrics(
    actions: np.ndarray, pair_starts: np.ndarray, action_std: np.ndarray
) -> dict[str, np.ndarray]:
    old = actions[pair_starts, 9]
    new = actions[pair_starts + 1, 0]
    delta = new - old
    old_inside = actions[pair_starts, 9] - actions[pair_starts, 8]
    new_inside = actions[pair_starts + 1, 1] - actions[pair_starts + 1, 0]

    def arm_raw(values: np.ndarray) -> np.ndarray:
        return rms(values[:, :6])

    def arm_scaled(values: np.ndarray) -> np.ndarray:
        return rms(values[:, :6] / action_std[:6])

    return {
        "delta7": delta.astype(np.float32),
        "abs_delta7": np.abs(delta).astype(np.float32),
        "arm_rms": arm_raw(delta),
        "arm_standardized_rms": arm_scaled(delta),
        "translation_rms": rms(delta[:, :3]),
        "rotation_rms": rms(delta[:, 3:6]),
        "gripper_abs_delta": np.abs(delta[:, 6]),
        "gripper_flip": (np.signbit(old[:, 6]) != np.signbit(new[:, 6])),
        "within_old_t9_t10_arm_rms": arm_raw(old_inside),
        "within_new_t1_t2_arm_rms": arm_raw(new_inside),
        "within_old_t9_t10_arm_standardized_rms": arm_scaled(old_inside),
        "within_new_t1_t2_arm_standardized_rms": arm_scaled(new_inside),
    }


def load_run(run: RunInfo, batch_size: int) -> RunData:
    summaries = sorted(
        read_json(run.path / "client" / "summaries.json"),
        key=lambda row: int(row["episode_index"]),
    )
    if [int(row["episode_index"]) for row in summaries] != list(range(len(summaries))):
        raise ValueError(f"{run.key}: non-contiguous episode indices")
    store = zarr.open_group(str(run.path / "server" / "routes.zarr"), mode="r")
    episode_id, episode_step, control_step = load_episode_axes(run, summaries, store)
    pair_starts = np.flatnonzero(episode_id[1:] == episode_id[:-1]).astype(np.int64)
    actions = load_actions(run, summaries, episode_id)
    route = extract_route_transitions(run, store, episode_id, pair_starts, batch_size)
    action = action_metrics(actions, pair_starts, run.action_std)
    states = np.asarray([int(row["init_state_id"]) for row in summaries], np.int32)
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries], np.int32)
    failure = np.asarray([not bool(row["success"]) for row in summaries], np.bool_)
    pair_episode = episode_id[pair_starts]
    pair_query = episode_step[pair_starts]
    if int(pair_query.max()) + 1 >= np.iinfo(np.int16).max:
        raise ValueError("query index exceeds int16")
    return RunData(
        info=run,
        summaries=summaries,
        episode_id=episode_id,
        episode_step=episode_step,
        control_step=control_step,
        pair_starts=pair_starts,
        pair_episode=pair_episode,
        pair_query=pair_query,
        states=states,
        seeds=seeds,
        failure=failure,
        route=route,
        action=action,
    )


def transition_matrix(run: RunData, values: np.ndarray, prefix: int) -> np.ndarray:
    tail = values.shape[1:]
    matrix = np.full((len(run.summaries), prefix, *tail), np.nan, np.float32)
    selected = run.pair_query < prefix
    matrix[run.pair_episode[selected], run.pair_query[selected]] = values[selected]
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{run.info.key}: incomplete q0-q{prefix - 1} prefix")
    return matrix


def episode_grid(
    run: RunData, episode_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.unique(run.states)
    seeds = np.unique(run.seeds)
    if len(states) != 16 or len(seeds) != 32:
        raise ValueError(f"{run.info.key}: expected 16 states x 32 seeds")
    grid = np.full(
        (len(states), len(seeds), *episode_values.shape[1:]), np.nan, np.float32
    )
    state_index = {int(value): index for index, value in enumerate(states)}
    seed_index = {int(value): index for index, value in enumerate(seeds)}
    for episode, (state, seed) in enumerate(zip(run.states, run.seeds)):
        slot = (state_index[int(state)], seed_index[int(seed)])
        if np.all(np.isfinite(grid[slot])):
            raise ValueError(f"{run.info.key}: duplicate state/seed cell {slot}")
        grid[slot] = episode_values[episode]
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{run.info.key}: incomplete state/seed grid")
    return grid, states, seeds


def rank_standardize_seed_axis(grid: np.ndarray) -> np.ndarray:
    ranks = rankdata(grid, axis=1, method="average").astype(np.float64)
    center = ranks.mean(axis=1, keepdims=True)
    scale = ranks.std(axis=1, keepdims=True)
    return (ranks - center) / np.maximum(scale, 1e-12)


def route_score_vectors(run: RunData) -> dict[str, np.ndarray]:
    result = {}
    for score_prefix, field_prefix in (("selected", ""), ("soft", "soft_")):
        boundary = run.route[f"{field_prefix}boundary_t10_to_next_t1_hellinger10"]
        same_position = 0.5 * (
            run.route[f"{field_prefix}same_t1_hellinger10"]
            + run.route[f"{field_prefix}same_t10_hellinger10"]
        )
        local_edge = 0.5 * (
            run.route[f"{field_prefix}within_old_t9_to_t10_hellinger10"]
            + run.route[f"{field_prefix}within_new_t1_to_t2_hellinger10"]
        )
        result[f"{score_prefix}_raw"] = boundary
        result[f"{score_prefix}_minus_same"] = boundary - same_position
        result[f"{score_prefix}_minus_edge"] = boundary - local_edge
    return result


def association_test(
    runs: list[RunData], permutations: int, seed: int
) -> dict[str, Any]:
    route_grids = []
    action_grids = []
    scalar_route_grids: dict[str, list[np.ndarray]] = {
        name: [] for name in ROUTE_SCORE_NAMES
    }
    scalar_action_grids = []
    by_task: dict[str, dict[str, float]] = {name: {} for name in ROUTE_SCORE_NAMES}
    common_seeds: np.ndarray | None = None

    for run in runs:
        route_scores = route_score_vectors(run)
        route_episode = transition_matrix(
            run,
            route_scores["selected_raw"],
            8,
        )
        action_episode = transition_matrix(run, run.action["abs_delta7"], 8)
        action_scalar = transition_matrix(run, run.action["arm_rms"], 8)
        route_grid, _, seeds = episode_grid(run, route_episode)
        action_grid, _, action_seeds = episode_grid(run, action_episode)
        scalar_action_grid, _, scalar_seeds = episode_grid(run, action_scalar)
        if not np.array_equal(seeds, action_seeds) or not np.array_equal(
            seeds, scalar_seeds
        ):
            raise ValueError(f"{run.info.key}: seed grids disagree")
        if common_seeds is None:
            common_seeds = seeds
        elif not np.array_equal(common_seeds, seeds):
            raise ValueError("flow-noise seed columns differ across tasks")

        route_rank = rank_standardize_seed_axis(route_grid)
        action_rank = rank_standardize_seed_axis(action_grid)
        scalar_action_rank = rank_standardize_seed_axis(scalar_action_grid)
        route_grids.append(route_rank)
        action_grids.append(action_rank)
        scalar_action_grids.append(scalar_action_rank)
        for name, values in route_scores.items():
            route_scalar_episode = transition_matrix(run, values.mean(axis=-1), 8)
            route_scalar_grid, _, score_seeds = episode_grid(run, route_scalar_episode)
            if not np.array_equal(seeds, score_seeds):
                raise ValueError(f"{run.info.key}: route-score seed grid disagrees")
            scalar_route_rank = rank_standardize_seed_axis(route_scalar_grid)
            scalar_route_grids[name].append(scalar_route_rank)
            by_task[name][run.info.key] = float(
                np.mean(scalar_route_rank.reshape(-1) * scalar_action_rank.reshape(-1))
            )

    route_flat = np.concatenate([grid.reshape(-1, N_DENOISE) for grid in route_grids])
    action_flat = np.concatenate(
        [grid.reshape(-1, ACTION_DIM) for grid in action_grids]
    )
    correlation = route_flat.T @ action_flat / len(route_flat)
    scalar_route = {
        name: np.concatenate([grid.reshape(-1) for grid in grids])
        for name, grids in scalar_route_grids.items()
    }
    scalar_action = np.concatenate([grid.reshape(-1) for grid in scalar_action_grids])
    scalar_correlation = np.asarray(
        [np.mean(scalar_route[name] * scalar_action) for name in ROUTE_SCORE_NAMES],
        np.float64,
    )

    rng = np.random.default_rng(seed)
    null_max = np.empty(permutations, np.float32)
    null_scalar = np.empty((permutations, len(ROUTE_SCORE_NAMES)), np.float32)
    seed_columns = len(common_seeds)
    for draw in range(permutations):
        order = rng.permutation(seed_columns)
        permuted_action = np.concatenate(
            [grid[:, order].reshape(-1, ACTION_DIM) for grid in action_grids]
        )
        null_matrix = route_flat.T @ permuted_action / len(route_flat)
        null_max[draw] = np.max(np.abs(null_matrix))
        permuted_scalar = np.concatenate(
            [grid[:, order].reshape(-1) for grid in scalar_action_grids]
        )
        for index, name in enumerate(ROUTE_SCORE_NAMES):
            null_scalar[draw, index] = np.mean(scalar_route[name] * permuted_scalar)

    corrected_p = (
        1 + np.sum(null_max[:, None, None] >= np.abs(correlation)[None, :, :], axis=0)
    ) / (permutations + 1)
    scalar_p = (
        1 + np.sum(np.abs(null_scalar) >= np.abs(scalar_correlation)[None, :], axis=0)
    ) / (permutations + 1)
    scalar_null_max = np.max(np.abs(null_scalar), axis=1)
    scalar_corrected_p = (
        1
        + np.sum(
            scalar_null_max[:, None] >= np.abs(scalar_correlation)[None, :],
            axis=0,
        )
    ) / (permutations + 1)
    scalar_results = {
        name: {
            "correlation": scalar_correlation[index],
            "two_sided_p": scalar_p[index],
            "six_score_maxT_p": scalar_corrected_p[index],
            "by_task": by_task[name],
            "null_95_interval": np.quantile(null_scalar[:, index], [0.025, 0.975]),
        }
        for index, name in enumerate(ROUTE_SCORE_NAMES)
    }
    maximum = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    return {
        "scope": "q0-q7, rank residualized within task x init-state x query",
        "permutation": (
            "same flow-noise seed-column permutation shared by every task/state/query"
        ),
        "observations": int(len(route_flat)),
        "scalar_route_scores_vs_arm_rms": scalar_results,
        "selected_denoise_by_action_dimension": {
            "correlation": correlation,
            "maxT_corrected_p": corrected_p,
            "maximum_absolute_cell": {
                "denoise_step": int(maximum[0]),
                "action_dimension": int(maximum[1]),
                "correlation": float(correlation[maximum]),
                "maxT_p": float(corrected_p[maximum]),
            },
            "null_max_abs_95": float(np.quantile(null_max, 0.95)),
        },
    }


def binary_auc_from_ranks(labels: np.ndarray, ranks: np.ndarray) -> tuple[float, int]:
    labels = np.asarray(labels, np.bool_)
    positive = int(labels.sum())
    negative = len(labels) - positive
    pairs = positive * negative
    if not pairs:
        return math.nan, 0
    value = (float(ranks[labels].sum()) - positive * (positive + 1) / 2.0) / pairs
    return value, pairs


def failure_feature_grids(run: RunData) -> tuple[dict[str, np.ndarray], np.ndarray]:
    features: dict[str, np.ndarray] = {}
    route_scores = route_score_vectors(run)
    for prefix in PREFIXES:
        action = transition_matrix(run, run.action["arm_rms"], prefix).mean(axis=1)
        for name, values in route_scores.items():
            route = transition_matrix(run, values.mean(axis=-1), prefix).mean(axis=1)
            features[f"{name}_h{prefix}"], _, seeds = episode_grid(run, route)
        features[f"action_h{prefix}"], _, action_seeds = episode_grid(run, action)
        if not np.array_equal(seeds, action_seeds):
            raise ValueError(f"{run.info.key}: failure feature seed grids disagree")
    failure_grid, _, failure_seeds = episode_grid(run, run.failure.astype(np.float32))
    if not np.array_equal(seeds, failure_seeds):
        raise ValueError(f"{run.info.key}: failure label seed grid disagrees")
    return features, failure_grid.astype(np.bool_)


def auc_from_records(
    records: list[dict[str, Any]], seed_order: np.ndarray | None = None
) -> tuple[np.ndarray, int, int]:
    numerator = np.zeros(len(FAILURE_FEATURES), np.float64)
    denominator = 0
    informative = 0
    for record in records:
        labels = record["labels"]
        if seed_order is not None:
            labels = labels[seed_order]
        pairs_here = int(labels.sum()) * int((~labels).sum())
        if not pairs_here:
            continue
        for feature_index, feature in enumerate(FAILURE_FEATURES):
            auc, pairs = binary_auc_from_ranks(labels, record["ranks"][feature])
            if pairs != pairs_here:
                raise ValueError("AUC pair count differs across features")
            numerator[feature_index] += auc * pairs
        denominator += pairs_here
        informative += 1
    if not denominator:
        return np.full(len(FAILURE_FEATURES), np.nan), informative, denominator
    return numerator / denominator, informative, denominator


def failure_test(runs: list[RunData], permutations: int, seed: int) -> dict[str, Any]:
    records = []
    task_records: dict[str, list[dict[str, Any]]] = {}
    common_seeds: np.ndarray | None = None
    failures = {}
    for run in runs:
        features, labels = failure_feature_grids(run)
        seeds = np.unique(run.seeds)
        if common_seeds is None:
            common_seeds = seeds
        elif not np.array_equal(common_seeds, seeds):
            raise ValueError("failure-test seed columns differ across tasks")
        task_records[run.info.key] = []
        failures[run.info.key] = int(run.failure.sum())
        for state_index in range(labels.shape[0]):
            item = {
                "task": run.info.key,
                "labels": labels[state_index],
                "ranks": {
                    name: rankdata(values[state_index], method="average")
                    for name, values in features.items()
                },
            }
            records.append(item)
            task_records[run.info.key].append(item)

    observed, informative, pairs = auc_from_records(records)
    by_task = {}
    candidate_feature = "soft_raw_h8"
    candidate_index = FAILURE_FEATURES.index(candidate_feature)
    for task, selected in task_records.items():
        auc, states, task_pairs = auc_from_records(selected)
        state_auc = []
        informative_records = []
        for record in selected:
            value, state_pairs = binary_auc_from_ranks(
                record["labels"], record["ranks"][candidate_feature]
            )
            if state_pairs:
                state_auc.append(value)
                informative_records.append(record)
        leave_one_state_out = []
        for dropped in range(len(informative_records)):
            remaining = [
                record
                for index, record in enumerate(informative_records)
                if index != dropped
            ]
            value = auc_from_records(remaining)[0][candidate_index]
            if np.isfinite(value):
                leave_one_state_out.append(value)
        by_task[task] = {
            "failures": failures[task],
            "informative_states": states,
            "success_failure_pairs": task_pairs,
            "auc_higher_means_failure": {
                name: auc[index] for index, name in enumerate(FAILURE_FEATURES)
            },
            "soft_raw_h8_state_robustness": {
                "states_above_chance": int(np.sum(np.asarray(state_auc) > 0.5)),
                "state_auc_median": (np.median(state_auc) if state_auc else math.nan),
                "leave_one_state_out_min": (
                    np.min(leave_one_state_out) if leave_one_state_out else math.nan
                ),
                "leave_one_state_out_max": (
                    np.max(leave_one_state_out) if leave_one_state_out else math.nan
                ),
            },
        }

    leave_one_task_out = {}
    for held_out in task_records:
        selected = [record for record in records if record["task"] != held_out]
        auc, states, task_pairs = auc_from_records(selected)
        leave_one_task_out[held_out] = {
            "informative_task_state_strata": states,
            "success_failure_pairs": task_pairs,
            "auc_higher_means_failure": {
                name: auc[index] for index, name in enumerate(FAILURE_FEATURES)
            },
        }

    robustness = {}
    for feature in FAILURE_FEATURES:
        task_auc = np.asarray(
            [
                item["auc_higher_means_failure"][feature]
                for item in by_task.values()
                if np.isfinite(item["auc_higher_means_failure"][feature])
            ],
            np.float64,
        )
        loo_auc = {
            task: item["auc_higher_means_failure"][feature]
            for task, item in leave_one_task_out.items()
        }
        robustness[feature] = {
            "informative_tasks": len(task_auc),
            "tasks_above_chance": int(np.sum(task_auc > 0.5)),
            "task_auc_median": np.median(task_auc),
            "task_auc_min": np.min(task_auc),
            "task_auc_max": np.max(task_auc),
            "leave_one_task_out_auc": loo_auc,
            "leave_one_task_out_min": np.nanmin(list(loo_auc.values())),
            "leave_one_task_out_max": np.nanmax(list(loo_auc.values())),
        }

    rng = np.random.default_rng(seed)
    null = np.empty((permutations, len(FAILURE_FEATURES)), np.float32)
    for draw in range(permutations):
        order = rng.permutation(len(common_seeds))
        null[draw] = auc_from_records(records, order)[0]
    null_max = np.max(np.abs(null - 0.5), axis=1)
    corrected_p = (
        1 + np.sum(null_max[:, None] >= np.abs(observed - 0.5)[None, :], axis=0)
    ) / (permutations + 1)
    return {
        "method": (
            "direct pair-weighted within-task/initial-state AUC; no fitted model"
        ),
        "permutation": (
            f"two-sided {len(FAILURE_FEATURES)}-test maxT with one shared "
            "flow-noise seed-column permutation"
        ),
        "informative_task_state_strata": informative,
        "success_failure_pairs": pairs,
        "auc_higher_means_failure": {
            name: observed[index] for index, name in enumerate(FAILURE_FEATURES)
        },
        "maxT_p": {
            name: corrected_p[index] for index, name in enumerate(FAILURE_FEATURES)
        },
        "null_95_interval": {
            name: np.quantile(null[:, index], [0.025, 0.975])
            for index, name in enumerate(FAILURE_FEATURES)
        },
        "by_task": by_task,
        "leave_one_task_out": leave_one_task_out,
        "robustness_by_score": robustness,
    }


def describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
    }


def summarize_run(run: RunData) -> dict[str, Any]:
    route_mean = {
        name: run.route[f"{name}_hellinger10"].mean(axis=-1) for name in ROUTE_TYPES
    }
    soft_route_mean = {
        name: run.route[f"soft_{name}_hellinger10"].mean(axis=-1)
        for name in ROUTE_TYPES
    }
    boundary = route_mean["boundary_t10_to_next_t1"]
    route_reference = 0.5 * (route_mean["same_t1"] + route_mean["same_t10"])
    action_reference = 0.5 * (
        run.action["within_old_t9_t10_arm_rms"] + run.action["within_new_t1_t2_arm_rms"]
    )
    prefix = run.pair_query < 8
    route_rank = rankdata(boundary[prefix])
    action_rank = rankdata(run.action["arm_rms"][prefix])
    correlation = float(np.corrcoef(route_rank, action_rank)[0, 1])
    return {
        "chunks": len(run.episode_id),
        "episodes": len(run.summaries),
        "successes": int((~run.failure).sum()),
        "failures": int(run.failure.sum()),
        "adjacent_boundaries": len(run.pair_starts),
        "route": {
            "hellinger_by_type": {
                name: describe(values) for name, values in route_mean.items()
            },
            "soft_hellinger_by_type": {
                name: describe(values) for name, values in soft_route_mean.items()
            },
            "boundary_to_same_position_mean_ratio": float(
                boundary.mean() / route_reference.mean()
            ),
            "soft_boundary_to_same_position_mean_ratio": float(
                soft_route_mean["boundary_t10_to_next_t1"].mean()
                / (
                    0.5
                    * (
                        soft_route_mean["same_t1"].mean()
                        + soft_route_mean["same_t10"].mean()
                    )
                )
            ),
            "boundary_hellinger_by_denoise": run.route[
                "boundary_t10_to_next_t1_hellinger10"
            ].mean(axis=0),
            "soft_boundary_hellinger_by_denoise": run.route[
                "soft_boundary_t10_to_next_t1_hellinger10"
            ].mean(axis=0),
            "boundary_cosine": describe(
                run.route["boundary_t10_to_next_t1_cosine10"].mean(axis=-1)
            ),
            "boundary_top4_jaccard": describe(
                run.route["boundary_t10_to_next_t1_top4_jaccard10"].mean(axis=-1)
            ),
            "boundary_site_top4_exact_rate": float(
                run.route["boundary_t10_to_next_t1_top4_exact_fraction10"].mean()
            ),
            "boundary_whole_top4_exact_matches": int(
                run.route["boundary_t10_to_next_t1_whole_top4_exact"].sum()
            ),
        },
        "action": {
            "boundary_arm_rms": describe(run.action["arm_rms"]),
            "boundary_arm_standardized_rms": describe(
                run.action["arm_standardized_rms"]
            ),
            "within_old_t9_t10_arm_rms": describe(
                run.action["within_old_t9_t10_arm_rms"]
            ),
            "within_new_t1_t2_arm_rms": describe(
                run.action["within_new_t1_t2_arm_rms"]
            ),
            "boundary_to_within_mean_ratio": float(
                run.action["arm_rms"].mean() / action_reference.mean()
            ),
            "mean_abs_delta_by_dimension": run.action["abs_delta7"].mean(axis=0),
            "gripper_flip_rate": float(run.action["gripper_flip"].mean()),
        },
        "q0_q7_unadjusted_spearman_route_vs_action": correlation,
    }


def aggregate_description(
    runs: list[RunData], summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    def task_mean(path: tuple[str, ...]) -> float:
        values = []
        for item in summaries.values():
            current: Any = item
            for key in path:
                current = current[key]
            values.append(float(current))
        return float(np.mean(values))

    whole_exact = sum(
        int(run.route["boundary_t10_to_next_t1_whole_top4_exact"].sum()) for run in runs
    )
    return {
        "aggregation": "equal weight per task for scalar means",
        "route_boundary_hellinger_mean": task_mean(
            (
                "route",
                "hellinger_by_type",
                "boundary_t10_to_next_t1",
                "mean",
            )
        ),
        "route_boundary_to_same_position_ratio": task_mean(
            ("route", "boundary_to_same_position_mean_ratio")
        ),
        "soft_route_boundary_hellinger_mean": task_mean(
            (
                "route",
                "soft_hellinger_by_type",
                "boundary_t10_to_next_t1",
                "mean",
            )
        ),
        "soft_route_boundary_to_same_position_ratio": task_mean(
            ("route", "soft_boundary_to_same_position_mean_ratio")
        ),
        "route_boundary_top4_jaccard_mean": task_mean(
            ("route", "boundary_top4_jaccard", "mean")
        ),
        "route_boundary_site_top4_exact_rate": task_mean(
            ("route", "boundary_site_top4_exact_rate")
        ),
        "route_boundary_whole_top4_exact_matches": whole_exact,
        "route_boundary_whole_top4_comparisons": int(
            sum(len(run.pair_starts) for run in runs)
        ),
        "action_boundary_arm_rms_mean": task_mean(
            ("action", "boundary_arm_rms", "mean")
        ),
        "action_boundary_arm_standardized_rms_mean": task_mean(
            ("action", "boundary_arm_standardized_rms", "mean")
        ),
        "action_boundary_to_within_ratio": task_mean(
            ("action", "boundary_to_within_mean_ratio")
        ),
        "action_gripper_flip_rate": task_mean(("action", "gripper_flip_rate")),
        "route_hellinger10": np.mean(
            [
                item["route"]["boundary_hellinger_by_denoise"]
                for item in summaries.values()
            ],
            axis=0,
        ),
        "soft_route_hellinger10": np.mean(
            [
                item["route"]["soft_boundary_hellinger_by_denoise"]
                for item in summaries.values()
            ],
            axis=0,
        ),
    }


def save_transition_data(output: Path, runs: list[RunData]) -> None:
    arrays: dict[str, list[np.ndarray]] = {
        "run_index": [],
        "episode_id": [],
        "from_query": [],
        "from_control_step": [],
        "to_control_step": [],
        "failure": [],
    }
    route_fields = sorted({name for run in runs for name in run.route})
    action_fields = sorted({name for run in runs for name in run.action})
    for name in route_fields + action_fields:
        arrays[name] = []

    for run_index, run in enumerate(runs):
        count = len(run.pair_starts)
        arrays["run_index"].append(np.full(count, run_index, np.int8))
        arrays["episode_id"].append(run.pair_episode)
        arrays["from_query"].append(run.pair_query)
        arrays["from_control_step"].append(run.control_step[run.pair_starts])
        arrays["to_control_step"].append(run.control_step[run.pair_starts + 1])
        arrays["failure"].append(run.failure[run.pair_episode])
        for name in route_fields:
            arrays[name].append(run.route[name])
        for name in action_fields:
            arrays[name].append(run.action[name])

    payload = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
    payload["run_names"] = np.asarray([run.info.key for run in runs])
    payload["denoise_steps"] = np.arange(N_DENOISE, dtype=np.int8)
    payload["action_dimensions"] = np.asarray(ACTION_NAMES)
    np.savez_compressed(output / "transitions.npz", **payload)


def save_episode_data(output: Path, runs: list[RunData]) -> None:
    run_index = []
    episode_id = []
    state = []
    seed = []
    failure = []
    features = {name: [] for name in FAILURE_FEATURES}
    for index, run in enumerate(runs):
        route_scores = route_score_vectors(run)
        for prefix in PREFIXES:
            for name, values in route_scores.items():
                features[f"{name}_h{prefix}"].append(
                    transition_matrix(run, values.mean(axis=-1), prefix).mean(axis=1)
                )
            features[f"action_h{prefix}"].append(
                transition_matrix(run, run.action["arm_rms"], prefix).mean(axis=1)
            )
        count = len(run.summaries)
        run_index.append(np.full(count, index, np.int8))
        episode_id.append(np.arange(count, dtype=np.int16))
        state.append(run.states)
        seed.append(run.seeds)
        failure.append(run.failure)
    np.savez_compressed(
        output / "episode_features.npz",
        run_index=np.concatenate(run_index),
        episode_id=np.concatenate(episode_id),
        init_state_id=np.concatenate(state),
        flow_noise_seed=np.concatenate(seed),
        failure=np.concatenate(failure),
        run_names=np.asarray([run.info.key for run in runs]),
        **{name: np.concatenate(parts) for name, parts in features.items()},
    )


def make_plot(output: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(summary["tasks"])
    labels = [
        "goal-middle",
        "goal-top+bowl",
        "long-moka",
        "spatial-ramekin",
        "spatial-stove",
    ]
    x = np.arange(len(tasks))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)

    action_boundary = [
        summary["tasks"][task]["action"]["boundary_arm_rms"]["mean"] for task in tasks
    ]
    action_within = [
        0.5
        * (
            summary["tasks"][task]["action"]["within_old_t9_t10_arm_rms"]["mean"]
            + summary["tasks"][task]["action"]["within_new_t1_t2_arm_rms"]["mean"]
        )
        for task in tasks
    ]
    width = 0.36
    axes[0, 0].bar(x - width / 2, action_boundary, width, label="T10 -> next T1")
    axes[0, 0].bar(x + width / 2, action_within, width, label="within-chunk edge")
    axes[0, 0].set_title("Arm-command change")
    axes[0, 0].set_ylabel("Raw command RMS")
    axes[0, 0].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 0].legend()

    route_boundary = [
        summary["tasks"][task]["route"]["hellinger_by_type"]["boundary_t10_to_next_t1"][
            "mean"
        ]
        for task in tasks
    ]
    route_same = [
        0.5
        * (
            summary["tasks"][task]["route"]["hellinger_by_type"]["same_t1"]["mean"]
            + summary["tasks"][task]["route"]["hellinger_by_type"]["same_t10"]["mean"]
        )
        for task in tasks
    ]
    axes[0, 1].bar(x - width / 2, route_boundary, width, label="T10 -> next T1")
    axes[0, 1].bar(x + width / 2, route_same, width, label="same-position")
    axes[0, 1].set_title("Actual combine-route change")
    axes[0, 1].set_ylabel("Hellinger distance")
    axes[0, 1].set_xticks(x, labels, rotation=22, ha="right")
    axes[0, 1].legend()

    association = summary["association"]["scalar_route_scores_vs_arm_rms"]
    score_labels = {
        "selected_raw": "selected raw",
        "selected_minus_same": "selected - same",
        "selected_minus_edge": "selected - edge",
        "soft_raw": "soft raw",
        "soft_minus_same": "soft - same",
        "soft_minus_edge": "soft - edge",
    }
    score_colors = ("#4c78a8", "#9ecae9", "#2c7fb8", "#f58518", "#ffbf79", "#e45756")
    score_width = 0.12
    for index, name in enumerate(ROUTE_SCORE_NAMES):
        task_corr = [association[name]["by_task"][task] for task in tasks]
        offset = (index - (len(ROUTE_SCORE_NAMES) - 1) / 2) * score_width
        axes[1, 0].bar(
            x + offset,
            task_corr,
            score_width,
            label=score_labels[name],
            color=score_colors[index],
        )
    axes[1, 0].axhline(0.0, color="black", linewidth=1)
    axes[1, 0].set_title("Fixed route-score/action correlation, q0-q7")
    axes[1, 0].set_ylabel("Within-state/query correlation")
    axes[1, 0].set_xticks(x, labels, rotation=22, ha="right")
    axes[1, 0].legend(fontsize=7, ncols=2)

    failure = summary["failure_test"]
    names = list(FAILURE_FEATURES)
    auc = [failure["auc_higher_means_failure"][name] for name in names]
    p = [failure["maxT_p"][name] for name in names]
    colors = ["#e45756" if value < 0.05 else "#72b7b2" for value in p]
    axes[1, 1].bar(np.arange(len(names)), auc, color=colors)
    axes[1, 1].axhline(0.5, color="black", linewidth=1, linestyle="--")
    lower = max(0.0, min([0.5, *auc]) - 0.06)
    upper = min(1.0, max([0.5, *auc]) + 0.06)
    axes[1, 1].set_ylim(lower, upper)
    axes[1, 1].set_title("Direct early failure AUC (red: maxT p < .05)")
    axes[1, 1].set_ylabel("AUC, higher score -> failure")
    axes[1, 1].set_xticks(np.arange(len(names)), names, rotation=22, ha="right")
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(output: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    association = summary["association"]
    scalar_scores = association["scalar_route_scores_vs_arm_rms"]
    raw_scalar = scalar_scores["selected_raw"]
    soft_scalar = scalar_scores["soft_raw"]
    strongest_name = max(
        ROUTE_SCORE_NAMES,
        key=lambda name: abs(scalar_scores[name]["correlation"]),
    )
    cell = association["selected_denoise_by_action_dimension"]["maximum_absolute_cell"]
    failure = summary["failure_test"]
    auc = failure["auc_higher_means_failure"]
    pvalues = failure["maxT_p"]
    candidate_robustness = failure["robustness_by_score"]["soft_raw_h8"]
    weakest_held_out = min(
        candidate_robustness["leave_one_task_out_auc"],
        key=candidate_robustness["leave_one_task_out_auc"].get,
    )
    ten = aggregate["route_hellinger10"]
    soft_ten = aggregate["soft_route_hellinger10"]
    lines = [
        "# Train-free 的 T10 → 下一 chunk T1 检查",
        "",
        "## 核心结论",
        "",
        f"- 重规划边界确实有动作跳变：六维机械臂命令 RMS 为 **{aggregate['action_boundary_arm_rms_mean']:.4f}**，是 chunk 内边缘相邻动作的 **{aggregate['action_boundary_to_within_ratio']:.2f}×**。按 checkpoint 已有动作尺度换算后为 {aggregate['action_boundary_arm_standardized_rms_mean']:.3f} 个标准差。",
        f"- MoE 在边界也明显重排：Top-4 Jaccard 均值 **{aggregate['route_boundary_top4_jaccard_mean']:.3f}**，完整 80-site Top-4 一致为 **{aggregate['route_boundary_whole_top4_exact_matches']}/{aggregate['route_boundary_whole_top4_comparisons']}**。",
        f"- 但两种跳变没有稳定耦合。控制任务、初始状态和 query 后，实际 selected-route 与 arm jump 的秩相关为 **r={raw_scalar['correlation']:.3f}**，六种固定分数 maxT `p={raw_scalar['six_score_maxT_p']:.4f}`。",
        f"- 完整 32-way soft router 的原始距离略强（**r={soft_scalar['correlation']:.3f}**），但六分数 maxT `p={soft_scalar['six_score_maxT_p']:.4f}`；绝对相关最大的固定标量就是 `{strongest_name}`，仍未达到显著。",
        f"- 逐 denoise × 逐动作维的 70 格扫描中，最大绝对相关为 **r={cell['correlation']:.3f}**（D{cell['denoise_step'] + 1} / {ACTION_NAMES[cell['action_dimension']]}），maxT `p={cell['maxT_p']:.4f}`。",
        f"- 失败侧出现一个候选：前 8 个边界的 `soft_raw_h8` 直接 AUC **{auc['soft_raw_h8']:.3f}**，{len(FAILURE_FEATURES)} 分数 maxT `p={pvalues['soft_raw_h8']:.4f}`；4 个有失败的任务均为正方向。",
        f"- 但它还不是通用告警：去掉 `{weakest_held_out}` 后 AUC 降到 **{candidate_robustness['leave_one_task_out_min']:.3f}**，说明主要增益由单个任务贡献。",
        "",
        "## 方法（无训练）",
        "",
        "每个边界的主向量为 10 维：`[d0, ..., d9]`。第 d 维是旧 T10 与新 T1 路由在该 denoise step 的 Hellinger 距离，并对 8 个 HB 层平均。`selected_*` 使用实际 Top-4 combine weights；`soft_*` 使用完整 32-way router probabilities。没有 PCA、回归、分类器、拟合阈值或学习权重。",
        "两种表示各自固定两个无参数对照：`*_minus_same = boundary - mean(same T1, same T10)`，去掉同位置跨 chunk 的变化；`*_minus_edge = boundary - mean(old T9→T10, new T1→T2)`，去掉 chunk 两端局部 token 变化。它们没有从成败标签中估计系数。",
        "动作侧直接使用 `new T1 - old T10` 的 7 维命令差；主统计使用前六维 raw RMS，checkpoint 元数据中的既有 action std 只作为辅助量纲。",
        "相关性固定使用所有任务共有的 q0-q7，并在 task × init-state × query 内做秩变换。置换时对所有任务和初始状态使用同一个 flow-noise seed 列置换，保留采样网格依赖。",
        "",
        "## 10 维路由边界向量的总体均值",
        "",
        "```text",
        "[" + ", ".join(f"{float(value):.4f}" for value in ten) + "]",
        "```",
        "",
        "完整 soft-router 的对应均值：",
        "",
        "```text",
        "[" + ", ".join(f"{float(value):.4f}" for value in soft_ten) + "]",
        "```",
        "",
        "## 固定路由标量与动作跳变",
        "",
        "| route score | rank correlation | six-score maxT p |",
        "|---|---:|---:|",
    ]
    for name in ROUTE_SCORE_NAMES:
        item = scalar_scores[name]
        lines.append(
            f"| `{name}` | {item['correlation']:+.3f} | "
            f"{item['six_score_maxT_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 各任务",
            "",
            "| task | selected H | soft H | same-position selected H | Top-4 Jaccard | arm RMS | boundary/within | selected r | soft r |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, item in summary["tasks"].items():
        route = item["route"]
        same = 0.5 * (
            route["hellinger_by_type"]["same_t1"]["mean"]
            + route["hellinger_by_type"]["same_t10"]["mean"]
        )
        lines.append(
            f"| `{task}` | {route['hellinger_by_type']['boundary_t10_to_next_t1']['mean']:.3f} | "
            f"{route['soft_hellinger_by_type']['boundary_t10_to_next_t1']['mean']:.3f} | "
            f"{same:.3f} | {route['boundary_top4_jaccard']['mean']:.3f} | "
            f"{item['action']['boundary_arm_rms']['mean']:.4f} | "
            f"{item['action']['boundary_to_within_mean_ratio']:.2f}× | "
            f"{raw_scalar['by_task'][task]:+.3f} | "
            f"{soft_scalar['by_task'][task]:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## 固定标量的失败 AUC",
            "",
            "`route_*_h4/h8` 是前 4/8 个边界的相应固定 route score 均值；`action_h4/h8` 是同一前缀的 arm RMS 均值。AUC > 0.5 表示值越大越偏失败。",
            "",
            f"| fixed score | AUC | {len(FAILURE_FEATURES)}-test maxT p |",
            "|---|---:|---:|",
        ]
    )
    for name in FAILURE_FEATURES:
        lines.append(f"| `{name}` | {auc[name]:.3f} | {pvalues[name]:.4f} |")
    lines.extend(
        [
            "",
            f"`soft_raw_h8` 在 {candidate_robustness['tasks_above_chance']}/{candidate_robustness['informative_tasks']} 个可评估任务中 AUC > 0.5，逐任务中位数为 {candidate_robustness['task_auc_median']:.3f}；leave-one-task-out 范围为 {candidate_robustness['leave_one_task_out_min']:.3f}–{candidate_robustness['leave_one_task_out_max']:.3f}。这说明方向一致，但效应强度尚未跨任务稳定。",
            "",
            "| task | failures | soft_raw_h8 AUC | state directions | state median | leave-one-state range |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task, item in failure["by_task"].items():
        if not item["informative_states"]:
            continue
        audit = item["soft_raw_h8_state_robustness"]
        lines.append(
            f"| `{task}` | {item['failures']} | "
            f"{item['auc_higher_means_failure']['soft_raw_h8']:.3f} | "
            f"{audit['states_above_chance']}/{item['informative_states']} | "
            f"{audit['state_auc_median']:.3f} | "
            f"{audit['leave_one_state_out_min']:.3f}–{audit['leave_one_state_out_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "该 AUC 是 task/初始状态内的 success-failure pair 加权结果，没有训练预测器；全成功的 middle-drawer 任务自动不参与 AUC。它仍是同一批离线数据上的探索性检验，不是独立验证集性能。",
            "",
            "## 解释",
            "",
            "重规划会同时重置 action-token 的相对位置（T10 回到 T1）、更新图像/状态并注入下一次 flow 初值，所以路由重排很大并不必然造成动作跳变。当前结果表明二者更多是并行发生，MoE 路由不能稳定替代动作连续性指标。",
            "失败预测是另一件事：`soft_raw_h8` 可能在累积早期状态偏离，而不是测量 T10→T1 的动作跳变。它可以作为无需训练的候选排序分数继续做 held-out task/checkpoint 验证；若要变成二值报警器，仍需要一个不从当前评估集调出来的阈值。",
            "",
            "## 产物",
            "",
            f"- `transitions.npz`: {summary['corpus']['replan_boundaries']:,} 个边界的 selected/soft route 10D、原始 7D action delta 和固定对照。",
            "- `episode_features.npz`: 固定前 4/8 个边界的 train-free episode 标量。",
            "- `summary.json`: 完整相关矩阵、maxT p、逐任务结果。",
            "- `overview.png`: 动作/路由边界、相关性和失败 AUC 总览。",
        ]
    )
    (output / "report.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_test() -> None:
    ids_a = np.zeros((2, N_LAYERS, N_DENOISE, TOP_K), np.uint8)
    ids_a[...] = np.arange(TOP_K, dtype=np.uint8)
    prob = np.zeros_like(ids_a, np.float32)
    prob[...] = np.asarray([1.0, 2.0, 3.0, 4.0], np.float32)
    same = compare_routes(ids_a, prob, ids_a.copy(), prob.copy())
    np.testing.assert_allclose(same["hellinger10"], 0.0)
    np.testing.assert_allclose(same["cosine10"], 1.0)
    np.testing.assert_allclose(same["top4_jaccard10"], 1.0)
    assert np.all(same["whole_top4_exact"])

    ids_b = ids_a + TOP_K
    different = compare_routes(ids_a, prob, ids_b, prob)
    np.testing.assert_allclose(different["hellinger10"], 1.0)
    np.testing.assert_allclose(different["cosine10"], 0.0)
    np.testing.assert_allclose(different["top4_jaccard10"], 0.0)
    assert not np.any(different["whole_top4_exact"])

    soft_a = np.zeros((2, N_LAYERS, N_DENOISE, N_EXPERTS), np.float32)
    soft_a[..., 0] = 1.0
    soft_b = np.zeros_like(soft_a)
    soft_b[..., 1] = 1.0
    np.testing.assert_allclose(compare_soft_routes(soft_a, soft_a), 0.0)
    np.testing.assert_allclose(compare_soft_routes(soft_a, soft_b), 1.0)

    labels = np.asarray([False, False, True, True])
    score = np.asarray([1.0, 2.0, 3.0, 4.0])
    auc, pairs = binary_auc_from_ranks(labels, rankdata(score))
    assert auc == 1.0 and pairs == 4
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.batch_size <= 0 or args.permutations <= 0:
        raise ValueError("batch-size and permutations must be positive")
    hub = args.hub.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    infos = discover_runs(hub, args.run_id)
    print(f"discovered {len(infos)} complete runs", flush=True)
    runs = [load_run(info, args.batch_size) for info in infos]

    task_summaries = {run.info.key: summarize_run(run) for run in runs}
    association = association_test(runs, args.permutations, args.seed)
    failure = failure_test(runs, args.permutations, args.seed + 1)
    summary = {
        "schema_version": "himoe-replan-boundary-trainfree/3",
        "method": {
            "trained_components": False,
            "learned_projection": False,
            "learned_predictor": False,
            "fitted_threshold": False,
            "route_vector": (
                "10 denoise-step Hellinger distances, each averaged over 8 HB layers"
            ),
            "route_distributions": {
                "selected": "actual normalized Top-4 combine weights",
                "soft": "full normalized 32-way router probabilities",
            },
            "fixed_route_scores": {
                name: (
                    "boundary"
                    if name.endswith("_raw")
                    else "boundary - mean(same-position T1, same-position T10)"
                    if name.endswith("_minus_same")
                    else "boundary - mean(old T9-to-T10, new T1-to-T2)"
                )
                for name in ROUTE_SCORE_NAMES
            },
            "action_vector": "raw next-T1 minus old-T10 7D command",
            "common_prefix_boundaries": 8,
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "corpus": {
            "hub": str(hub),
            "run_id": args.run_id,
            "runs": len(runs),
            "episodes": sum(len(run.summaries) for run in runs),
            "chunks": sum(len(run.episode_id) for run in runs),
            "replan_boundaries": sum(len(run.pair_starts) for run in runs),
        },
        "tasks": task_summaries,
        "aggregate": aggregate_description(runs, task_summaries),
        "association": association,
        "failure_test": failure,
    }

    save_transition_data(output, runs)
    save_episode_data(output, runs)
    serializable = plain(summary)
    (output / "summary.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output, serializable)
    make_plot(output, serializable)
    print(f"wrote train-free analysis to {output}", flush=True)


if __name__ == "__main__":
    main()
