#!/usr/bin/env python3
"""Compare successful and failed rollout MoE trends, one query at a time.

Every model at query k receives only HB-MoE computation from the ten denoise
rounds inside query k. No feature crosses query boundaries. Selected expert
outputs are reconstructed from recorded IDs/weights, fp16 router inputs, and
checkpoint MLPs. The fixed k=0..8 cohort avoids outcome-dependent termination.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import zarr
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_expert_activation_v2 import _layer_weights, _mlp


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/moe-rollout-trend"
LAYERS = (2, 5, 12, 15)
N_CHUNKS = 9
N_DENOISE = 10
N_TOKENS = 11
WIDTH = 1024
N_GROUPS = 2  # state token, mean action token
PROJECTED_DIM = 64
FEATURE_VERSION = 2
PCA_COMPONENTS = 12
LOGISTIC_C = 0.1
BOOTSTRAPS = 5000
BRANCHES = ("routed", "shared", "hidden")
SCALAR_NAMES = (
    "expert_mass",
    "routed_rms",
    "disagreement",
    "cancellation",
    "expert_cv",
    "routed_over_shared",
    "routed_shared_conflict",
)
MODEL_FAMILIES = {
    "routed_identity": ("routed",),
    "routed_scalar": ("scalar",),
    "routed_full": ("routed", "scalar"),
    "shared_identity": ("shared",),
    "hidden_identity": ("hidden",),
    "base": ("hidden", "shared"),
    "base_routed": ("hidden", "shared", "routed", "scalar"),
}
DESCRIPTIVE_METRICS = (
    "routed_rms",
    "cancellation",
    "shared_conflict",
    "within_query_path",
    "d0_to_d9_commitment",
)


@dataclass(frozen=True)
class TrendTask:
    task: str
    episodes: np.ndarray
    scenes: np.ndarray
    seeds: np.ndarray
    failure: np.ndarray
    routed: np.ndarray
    shared: np.ndarray
    hidden: np.ndarray
    scalar: np.ndarray
    routed_commitment: np.ndarray
    shared_commitment: np.ndarray
    hidden_commitment: np.ndarray
    routed_adjacent: np.ndarray
    shared_adjacent: np.ndarray
    hidden_adjacent: np.ndarray
    validation: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--task", action="append", help="suite/task; repeatable")
    parser.add_argument("--max-chunks", type=int, default=N_CHUNKS)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--force-compact", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--layer-shard", type=int, choices=LAYERS)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="regenerate the strict summary, report, and plot from summary.json",
    )
    return parser.parse_args()


def discover_runs(cache_root: pathlib.Path, requested: set[str] | None) -> list[pathlib.Path]:
    runs = []
    for summary_path in sorted(cache_root.glob("**/right-16x32/client/summaries.json")):
        run = summary_path.parents[1]
        if not (run / "server/routes.zarr").exists() or not (
            run / "server/hidden.zarr"
        ).exists():
            continue
        task = str(run.relative_to(cache_root).parent)
        if requested is not None and task not in requested:
            continue
        rows = json.loads(summary_path.read_text())
        outcomes = [bool(row["success"]) for row in rows]
        if any(outcomes) and not all(outcomes):
            runs.append(run)
    found = {str(run.relative_to(cache_root).parent) for run in runs}
    if requested is not None and found != requested:
        raise ValueError("requested variable-outcome tasks not found: %s" % sorted(requested - found))
    if not runs:
        raise RuntimeError("no variable-outcome right-16x32 runs found")
    return runs


def _cache_path(out_dir: pathlib.Path, task: str) -> pathlib.Path:
    return out_dir / "compact" / (task.replace("/", "__") + ".npz")


def _shard_path(out_dir: pathlib.Path, task: str, layer: int) -> pathlib.Path:
    return out_dir / "shards" / (
        task.replace("/", "__") + "__layer%02d.npz" % layer
    )


def _projection(seed: int, projected_dim: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, projected_dim, size=WIDTH)
    sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=WIDTH)
    return sparse.csr_matrix(
        (sign, (np.arange(WIDTH), bucket)), shape=(WIDTH, projected_dim)
    )


def project_group_vectors(
    values: np.ndarray, projection: sparse.csr_matrix
) -> np.ndarray:
    """Project [N,D,G,W] directions while preserving denoise and token group."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 4 or values.shape[1:3] != (N_DENOISE, N_GROUPS):
        raise ValueError("unexpected grouped-vector shape %s" % (values.shape,))
    projected = np.asarray(values.reshape(-1, WIDTH) @ projection, dtype=np.float32)
    projected /= np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-12)
    return projected.reshape(*values.shape[:-1], projection.shape[1]).astype(np.float16)


def vector_dynamics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cosine-to-d9 and adjacent cosine distance for [N,D,G,W]."""
    values = np.asarray(values, dtype=np.float32)
    normalized = values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)
    commitment = np.sum(normalized * normalized[:, -1:, :, :], axis=-1)
    adjacent = 1.0 - np.sum(normalized[:, 1:] * normalized[:, :-1], axis=-1)
    return (
        np.clip(commitment, -1.0, 1.0).astype(np.float16),
        np.clip(adjacent, 0.0, 2.0).astype(np.float16),
    )


def _group_vectors(values, n_rows: int):
    values = values.reshape(n_rows, N_DENOISE, N_TOKENS, WIDTH)
    return np.stack(
        (
            values[:, :, 0].float().numpy(),
            values[:, :, 1:].mean(dim=2).float().numpy(),
        ),
        axis=2,
    ).astype(np.float32, copy=False)


def _compute_query_features(
    torch,
    hidden: np.ndarray,
    ids_np: np.ndarray,
    raw_np: np.ndarray,
    experts,
    shared_weights,
    gate_weight,
    projection: sparse.csr_matrix,
) -> dict:
    """Reconstruct one layer/query and return only compact arrays."""
    n_rows = len(hidden)
    alpha_np = raw_np / np.maximum(raw_np.sum(axis=-1, keepdims=True), 1e-20)
    x = (
        torch.from_numpy(np.ascontiguousarray(hidden))
        .reshape(-1, WIDTH)
        .to(dtype=torch.bfloat16)
    )
    ids = torch.from_numpy(np.ascontiguousarray(ids_np).reshape(-1, 4))
    alpha = torch.from_numpy(np.ascontiguousarray(alpha_np).reshape(-1, 4)).float()
    raw = torch.from_numpy(np.ascontiguousarray(raw_np).reshape(-1, 4)).float()
    token_count = len(x)
    mlp_chunk = 8192
    with torch.inference_mode():
        routed = torch.zeros((token_count, WIDTH), dtype=torch.float32)
        first_moment = torch.zeros(token_count, dtype=torch.float32)
        second_moment = torch.zeros(token_count, dtype=torch.float32)
        for expert_id, expert_weights in enumerate(experts):
            matches = torch.nonzero(ids == expert_id, as_tuple=False)
            if matches.numel() == 0:
                continue
            token_index = matches[:, 0]
            slot_index = matches[:, 1]
            for lo in range(0, len(token_index), mlp_chunk):
                ti = token_index[lo : lo + mlp_chunk]
                si = slot_index[lo : lo + mlp_chunk]
                expert_output = _mlp(torch, x[ti], expert_weights).float()
                weight = alpha[ti, si]
                rms = expert_output.square().mean(dim=-1).sqrt()
                routed.index_add_(0, ti, expert_output * weight[:, None])
                first_moment.index_add_(0, ti, rms * weight)
                second_moment.index_add_(0, ti, rms.square() * weight)

        shared = torch.empty((token_count, WIDTH), dtype=torch.float32)
        for lo in range(0, token_count, mlp_chunk):
            shared[lo : lo + mlp_chunk] = _mlp(
                torch, x[lo : lo + mlp_chunk], shared_weights
            ).float()
        routed_rms = routed.square().mean(dim=-1).sqrt()
        shared_rms = shared.square().mean(dim=-1).sqrt()
        disagreement = 1.0 - routed_rms.square() / second_moment.clamp_min(1e-12)
        cancellation = 1.0 - routed_rms / first_moment.clamp_min(1e-12)
        expert_cv = (
            (second_moment - first_moment.square()).clamp_min(0).sqrt()
            / first_moment.clamp_min(1e-12)
        )
        routed_over_shared = routed_rms / shared_rms.clamp_min(1e-12)
        conflict = 1.0 - (routed * shared).sum(dim=-1) / (
            routed.norm(dim=-1) * shared.norm(dim=-1)
        ).clamp_min(1e-12)
        scalar_token = torch.stack(
            (
                first_moment,
                routed_rms,
                disagreement,
                cancellation,
                expert_cv,
                routed_over_shared,
                conflict,
            ),
            dim=-1,
        ).reshape(n_rows, N_DENOISE, N_TOKENS, len(SCALAR_NAMES))
        scalar = torch.cat(
            (
                scalar_token[:, :, 0],
                scalar_token[:, :, 1:].mean(dim=2),
                scalar_token[:, :, 1:].std(dim=2, correction=0),
            ),
            dim=-1,
        ).numpy()
        grouped = {
            "routed": _group_vectors(routed, n_rows),
            "shared": _group_vectors(shared, n_rows),
            "hidden": _group_vectors(x, n_rows),
        }
        output = {"scalar": scalar}
        for branch, value in grouped.items():
            output[branch] = project_group_vectors(value, projection)
            output[branch + "_commitment"], output[branch + "_adjacent"] = (
                vector_dynamics(value)
            )
        logits = torch.nn.functional.linear(x, gate_weight)
        probabilities = logits.float().softmax(dim=-1)
        predicted = probabilities.topk(4, dim=-1, sorted=False).indices
        output["top4_set_match"] = float(
            (
                predicted.sort(dim=-1).values == ids.sort(dim=-1).values
            ).all(dim=-1).float().mean()
        )
        output["selected_probability_mae"] = float(
            (probabilities.gather(1, ids).float() - raw).abs().mean()
        )
    return output


def _row_grid(
    summaries: list[dict], max_chunks: int, episode_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    failure = np.asarray([not bool(row["success"]) for row in summaries])
    counts = np.asarray([int(row["inference_calls"]) for row in summaries])
    if counts.min() < max_chunks:
        raise ValueError("fixed cohort is not active through k=%d" % (max_chunks - 1))
    offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    grid = offsets[:, None] + np.arange(max_chunks, dtype=np.int64)[None, :]
    if len(episode_axis) != int(counts.sum()):
        raise ValueError("summary and server row counts differ")
    if not np.all(episode_axis[grid] == episodes[:, None]):
        raise ValueError("episode/query row lookup failed")
    return episodes, scenes, seeds, failure, grid


def _load_cached(path: pathlib.Path, max_chunks: int, projected_dim: int) -> TrendTask | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if (
            int(stored["feature_version"]) != FEATURE_VERSION
            or int(stored["max_chunks"]) != max_chunks
            or int(stored["projected_dim"]) != projected_dim
        ):
            return None
        kwargs = {
            name: np.asarray(stored[name])
            for name in (
                "episodes",
                "scenes",
                "seeds",
                "routed",
                "shared",
                "hidden",
                "scalar",
                "routed_commitment",
                "shared_commitment",
                "hidden_commitment",
                "routed_adjacent",
                "shared_adjacent",
                "hidden_adjacent",
            )
        }
        return TrendTask(
            task=str(stored["task"]),
            failure=np.asarray(stored["failure"], dtype=bool),
            validation=json.loads(str(stored["validation_json"])),
            **kwargs,
        )


def extract_layer_shard(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    out_dir: pathlib.Path,
    layer: int,
    max_chunks: int,
    projected_dim: int,
    threads: int,
    seed: int,
    force: bool,
) -> pathlib.Path:
    import torch

    task = str(run.relative_to(cache_root).parent)
    path = _shard_path(out_dir, task, layer)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as stored:
            if (
                int(stored["feature_version"]) == FEATURE_VERSION
                and int(stored["max_chunks"]) == max_chunks
                and int(stored["projected_dim"]) == projected_dim
                and int(stored["layer"]) == layer
            ):
                print("%s layer %d: loading existing shard" % (task, layer), flush=True)
                return path

    torch.set_num_threads(threads)
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden_group = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:])
    if not np.array_equal(route_episode, np.asarray(hidden_group["episode_id"][:])):
        raise ValueError("route/hidden episode axes differ")
    episodes, scenes, seeds, failure, row_grid = _row_grid(
        summaries, max_chunks, route_episode
    )
    metadata = json.loads((run / "client/server_metadata.json").read_text())
    captured_layers = tuple(int(v) for v in metadata["routing_hb_layer_indices"])
    captured_axis = captured_layers.index(layer)
    checkpoint = pathlib.Path(metadata["checkpoint"]) / "pytorch_model.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    experts, shared_weights, gate_weight = _layer_weights(
        state, layer, "cpu", torch
    )
    projection = _projection(seed + 17, projected_dim)
    n_rows = len(episodes)
    vector_shape = (n_rows, max_chunks, N_DENOISE, N_GROUPS, projected_dim)
    dynamics_shape = (n_rows, max_chunks, N_DENOISE, N_GROUPS)
    adjacent_shape = (n_rows, max_chunks, N_DENOISE - 1, N_GROUPS)
    vectors = {branch: np.empty(vector_shape, np.float16) for branch in BRANCHES}
    commitments = {
        branch: np.empty(dynamics_shape, np.float16) for branch in BRANCHES
    }
    adjacencies = {
        branch: np.empty(adjacent_shape, np.float16) for branch in BRANCHES
    }
    scalar = np.empty(
        (n_rows, max_chunks, N_DENOISE, 3 * len(SCALAR_NAMES)), np.float32
    )
    validation = []
    started = time.perf_counter()
    for chunk_index in range(max_chunks):
        rows = row_grid[:, chunk_index]
        selection = (rows, [captured_axis], slice(None), slice(None), slice(None))
        hidden = np.asarray(
            hidden_group["hb_hidden"].get_orthogonal_selection(selection),
            dtype=np.float16,
        )[:, 0]
        ids = np.asarray(
            route_group["hb_expert_ids"].get_orthogonal_selection(selection),
            dtype=np.int64,
        )[:, 0]
        raw = np.asarray(
            route_group["hb_selected_prob"].get_orthogonal_selection(selection),
            dtype=np.float32,
        )[:, 0]
        result = _compute_query_features(
            torch,
            hidden,
            ids,
            raw,
            experts,
            shared_weights,
            gate_weight,
            projection,
        )
        scalar[:, chunk_index] = result["scalar"]
        for branch in BRANCHES:
            vectors[branch][:, chunk_index] = result[branch]
            commitments[branch][:, chunk_index] = result[branch + "_commitment"]
            adjacencies[branch][:, chunk_index] = result[branch + "_adjacent"]
        validation.append(
            {
                "chunk": chunk_index,
                "layer": layer,
                "top4_set_match": result["top4_set_match"],
                "selected_probability_mae": result["selected_probability_mae"],
            }
        )
        del hidden, ids, raw, result
        gc.collect()
        print(
            "%s layer %d: query %d/%d (%.1fs)"
            % (task, layer, chunk_index, max_chunks - 1, time.perf_counter() - started),
            flush=True,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_version=np.asarray(FEATURE_VERSION),
        max_chunks=np.asarray(max_chunks),
        projected_dim=np.asarray(projected_dim),
        layer=np.asarray(layer),
        task=np.asarray(task),
        episodes=episodes,
        scenes=scenes,
        seeds=seeds,
        failure=failure,
        scalar=scalar,
        validation_json=json.dumps(validation),
        **vectors,
        **{branch + "_commitment": commitments[branch] for branch in BRANCHES},
        **{branch + "_adjacent": adjacencies[branch] for branch in BRANCHES},
    )
    print("%s layer %d: wrote %s" % (task, layer, path), flush=True)
    return path


def assemble_task_shards(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    out_dir: pathlib.Path,
    max_chunks: int,
    projected_dim: int,
) -> TrendTask:
    task = str(run.relative_to(cache_root).parent)
    cache = _cache_path(out_dir, task)
    shards = []
    for layer in LAYERS:
        path = _shard_path(out_dir, task, layer)
        if not path.exists():
            raise FileNotFoundError("missing layer shard %s" % path)
        shards.append(np.load(path, allow_pickle=False))
    try:
        reference = shards[0]
        for layer, stored in zip(LAYERS, shards):
            if (
                int(stored["feature_version"]) != FEATURE_VERSION
                or int(stored["max_chunks"]) != max_chunks
                or int(stored["projected_dim"]) != projected_dim
                or int(stored["layer"]) != layer
                or not np.array_equal(stored["episodes"], reference["episodes"])
                or not np.array_equal(stored["failure"], reference["failure"])
            ):
                raise ValueError("incompatible layer shard for %s layer %d" % (task, layer))
        arrays = {
            branch: np.stack([np.asarray(stored[branch]) for stored in shards], axis=3)
            for branch in BRANCHES
        }
        arrays["scalar"] = np.stack(
            [np.asarray(stored["scalar"]) for stored in shards], axis=3
        )
        for branch in BRANCHES:
            arrays[branch + "_commitment"] = np.stack(
                [np.asarray(stored[branch + "_commitment"]) for stored in shards],
                axis=3,
            )
            arrays[branch + "_adjacent"] = np.stack(
                [np.asarray(stored[branch + "_adjacent"]) for stored in shards],
                axis=3,
            )
        validation = []
        for stored in shards:
            validation.extend(json.loads(str(stored["validation_json"])))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            feature_version=np.asarray(FEATURE_VERSION),
            max_chunks=np.asarray(max_chunks),
            projected_dim=np.asarray(projected_dim),
            task=np.asarray(task),
            episodes=np.asarray(reference["episodes"]),
            scenes=np.asarray(reference["scenes"]),
            seeds=np.asarray(reference["seeds"]),
            failure=np.asarray(reference["failure"]),
            validation_json=json.dumps(validation),
            **arrays,
        )
    finally:
        for stored in shards:
            stored.close()
    result = _load_cached(cache, max_chunks, projected_dim)
    if result is None:
        raise RuntimeError("assembled compact cache failed validation")
    print("%s: assembled %s" % (task, cache), flush=True)
    return result


def extract_task(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    out_dir: pathlib.Path,
    max_chunks: int,
    projected_dim: int,
    threads: int,
    seed: int,
    force: bool,
) -> TrendTask:
    task = str(run.relative_to(cache_root).parent)
    cache = _cache_path(out_dir, task)
    if not force:
        cached = _load_cached(cache, max_chunks, projected_dim)
        if cached is not None:
            print("%s: loading compact rollout-trend cache" % task, flush=True)
            return cached

    import torch

    torch.set_num_threads(threads)
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden_group = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:])
    hidden_episode = np.asarray(hidden_group["episode_id"][:])
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError("route/hidden episode axes differ")
    episodes, scenes, seeds, failure, row_grid = _row_grid(
        summaries, max_chunks, route_episode
    )
    if len(episodes) != 512 or len(np.unique(scenes)) != 16 or len(np.unique(seeds)) != 32:
        raise ValueError("expected complete 16-state x 32-seed grid")

    metadata = json.loads((run / "client/server_metadata.json").read_text())
    captured_layers = tuple(int(v) for v in metadata["routing_hb_layer_indices"])
    layer_axes = tuple(captured_layers.index(layer) for layer in LAYERS)
    checkpoint = pathlib.Path(metadata["checkpoint"]) / "pytorch_model.pth"
    checkpoint_state = torch.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    projection = _projection(seed + 17, projected_dim)
    n_rows = len(episodes)
    vec_shape = (
        n_rows,
        max_chunks,
        N_DENOISE,
        len(LAYERS),
        N_GROUPS,
        projected_dim,
    )
    dynamics_shape = (
        n_rows,
        max_chunks,
        N_DENOISE,
        len(LAYERS),
        N_GROUPS,
    )
    adjacent_shape = (
        n_rows,
        max_chunks,
        N_DENOISE - 1,
        len(LAYERS),
        N_GROUPS,
    )
    routed_result = np.empty(vec_shape, dtype=np.float16)
    shared_result = np.empty(vec_shape, dtype=np.float16)
    hidden_result = np.empty(vec_shape, dtype=np.float16)
    scalar_result = np.empty(
        (n_rows, max_chunks, N_DENOISE, len(LAYERS), 3 * len(SCALAR_NAMES)),
        dtype=np.float32,
    )
    commitments = {
        branch: np.empty(dynamics_shape, dtype=np.float16) for branch in BRANCHES
    }
    adjacencies = {
        branch: np.empty(adjacent_shape, dtype=np.float16) for branch in BRANCHES
    }
    vectors = {
        "routed": routed_result,
        "shared": shared_result,
        "hidden": hidden_result,
    }
    validation: list[dict] = []
    started = time.perf_counter()
    mlp_chunk = 8192

    for output_layer_axis, (layer, captured_axis) in enumerate(zip(LAYERS, layer_axes)):
        experts, shared_weights, gate_weight = _layer_weights(
            checkpoint_state, int(layer), "cpu", torch
        )
        experts = [tuple(weight.float() for weight in triplet) for triplet in experts]
        shared_weights = tuple(weight.float() for weight in shared_weights)
        gate_weight = gate_weight.float()
        for chunk_index in range(max_chunks):
            rows = row_grid[:, chunk_index]
            selection = (rows, [captured_axis], slice(None), slice(None), slice(None))
            hidden = np.asarray(
                hidden_group["hb_hidden"].get_orthogonal_selection(selection),
                dtype=np.float16,
            )[:, 0]
            ids_np = np.asarray(
                route_group["hb_expert_ids"].get_orthogonal_selection(selection),
                dtype=np.int64,
            )[:, 0]
            raw_np = np.asarray(
                route_group["hb_selected_prob"].get_orthogonal_selection(selection),
                dtype=np.float32,
            )[:, 0]
            alpha_np = raw_np / np.maximum(raw_np.sum(axis=-1, keepdims=True), 1e-20)

            x = torch.from_numpy(np.ascontiguousarray(hidden)).reshape(-1, WIDTH).float()
            ids = torch.from_numpy(np.ascontiguousarray(ids_np).reshape(-1, 4))
            alpha = torch.from_numpy(np.ascontiguousarray(alpha_np).reshape(-1, 4)).float()
            raw = torch.from_numpy(np.ascontiguousarray(raw_np).reshape(-1, 4)).float()
            token_count = len(x)
            with torch.inference_mode():
                routed = torch.zeros((token_count, WIDTH), dtype=torch.float32)
                first_moment = torch.zeros(token_count, dtype=torch.float32)
                second_moment = torch.zeros(token_count, dtype=torch.float32)
                for expert_id, expert_weights in enumerate(experts):
                    matches = torch.nonzero(ids == expert_id, as_tuple=False)
                    if matches.numel() == 0:
                        continue
                    token_index = matches[:, 0]
                    slot_index = matches[:, 1]
                    for lo in range(0, len(token_index), mlp_chunk):
                        ti = token_index[lo : lo + mlp_chunk]
                        si = slot_index[lo : lo + mlp_chunk]
                        expert_output = _mlp(torch, x[ti], expert_weights)
                        weight = alpha[ti, si]
                        rms = expert_output.square().mean(dim=-1).sqrt()
                        routed.index_add_(0, ti, expert_output * weight[:, None])
                        first_moment.index_add_(0, ti, rms * weight)
                        second_moment.index_add_(0, ti, rms.square() * weight)

                shared = torch.empty((token_count, WIDTH), dtype=torch.float32)
                for lo in range(0, token_count, mlp_chunk):
                    shared[lo : lo + mlp_chunk] = _mlp(
                        torch, x[lo : lo + mlp_chunk], shared_weights
                    )

                routed_rms = routed.square().mean(dim=-1).sqrt()
                shared_rms = shared.square().mean(dim=-1).sqrt()
                disagreement = 1.0 - routed_rms.square() / second_moment.clamp_min(1e-12)
                cancellation = 1.0 - routed_rms / first_moment.clamp_min(1e-12)
                expert_cv = (
                    (second_moment - first_moment.square()).clamp_min(0).sqrt()
                    / first_moment.clamp_min(1e-12)
                )
                routed_over_shared = routed_rms / shared_rms.clamp_min(1e-12)
                conflict = 1.0 - (routed * shared).sum(dim=-1) / (
                    routed.norm(dim=-1) * shared.norm(dim=-1)
                ).clamp_min(1e-12)
                scalar_token = torch.stack(
                    (
                        first_moment,
                        routed_rms,
                        disagreement,
                        cancellation,
                        expert_cv,
                        routed_over_shared,
                        conflict,
                    ),
                    dim=-1,
                ).reshape(n_rows, N_DENOISE, N_TOKENS, len(SCALAR_NAMES))
                scalar_group = torch.cat(
                    (
                        scalar_token[:, :, 0],
                        scalar_token[:, :, 1:].mean(dim=2),
                        scalar_token[:, :, 1:].std(dim=2, correction=0),
                    ),
                    dim=-1,
                )
                scalar_result[:, chunk_index, :, output_layer_axis] = scalar_group.numpy()

                grouped = {
                    "routed": _group_vectors(routed, n_rows),
                    "shared": _group_vectors(shared, n_rows),
                    "hidden": _group_vectors(x, n_rows),
                }
                for branch, value in grouped.items():
                    vectors[branch][
                        :, chunk_index, :, output_layer_axis
                    ] = project_group_vectors(value, projection)
                    commitment, adjacent = vector_dynamics(value)
                    commitments[branch][
                        :, chunk_index, :, output_layer_axis
                    ] = commitment
                    adjacencies[branch][
                        :, chunk_index, :, output_layer_axis
                    ] = adjacent

                logits = torch.nn.functional.linear(x, gate_weight)
                probabilities = logits.softmax(dim=-1)
                predicted = probabilities.topk(4, dim=-1, sorted=False).indices
                set_match = (
                    predicted.sort(dim=-1).values == ids.sort(dim=-1).values
                ).all(dim=-1).float().mean()
                probability_mae = (
                    probabilities.gather(1, ids).float() - raw
                ).abs().mean()
                validation.append(
                    {
                        "chunk": chunk_index,
                        "layer": int(layer),
                        "top4_set_match": float(set_match),
                        "selected_probability_mae": float(probability_mae),
                    }
                )

            del hidden, ids_np, raw_np, alpha_np, x, ids, alpha, raw
            del routed, shared, grouped, scalar_token, scalar_group
            gc.collect()
            print(
                "%s: layer %d query %d/%d (%.1fs)"
                % (task, layer, chunk_index, max_chunks - 1, time.perf_counter() - started),
                flush=True,
            )
        del experts, shared_weights, gate_weight
        gc.collect()

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        feature_version=np.asarray(FEATURE_VERSION),
        max_chunks=np.asarray(max_chunks),
        projected_dim=np.asarray(projected_dim),
        task=np.asarray(task),
        episodes=episodes,
        scenes=scenes,
        seeds=seeds,
        failure=failure,
        routed=routed_result,
        shared=shared_result,
        hidden=hidden_result,
        scalar=scalar_result,
        routed_commitment=commitments["routed"],
        shared_commitment=commitments["shared"],
        hidden_commitment=commitments["hidden"],
        routed_adjacent=adjacencies["routed"],
        shared_adjacent=adjacencies["shared"],
        hidden_adjacent=adjacencies["hidden"],
        validation_json=json.dumps(validation),
    )
    print("%s: wrote %s" % (task, cache), flush=True)
    result = _load_cached(cache, max_chunks, projected_dim)
    if result is None:
        raise RuntimeError("new compact cache failed validation")
    return result


def feature_blocks(data: TrendTask, chunk: int) -> dict[str, np.ndarray]:
    blocks = {}
    for branch in BRANCHES:
        identity = getattr(data, branch)[:, chunk].reshape(len(data.failure), -1)
        commitment = getattr(data, branch + "_commitment")[:, chunk].reshape(
            len(data.failure), -1
        )
        adjacent = getattr(data, branch + "_adjacent")[:, chunk].reshape(
            len(data.failure), -1
        )
        blocks[branch] = np.column_stack((identity, commitment, adjacent))
    blocks["scalar"] = np.column_stack(
        (
            data.scalar[:, chunk].reshape(len(data.failure), -1),
            data.routed_commitment[:, chunk].reshape(len(data.failure), -1),
            data.routed_adjacent[:, chunk].reshape(len(data.failure), -1),
        )
    )
    return blocks


def _reduce_block(
    train: np.ndarray, test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    keep = train.std(axis=0) > 1e-9
    if not np.any(keep):
        raise ValueError("feature block is constant")
    scaler = StandardScaler().fit(train[:, keep])
    x_train = scaler.transform(train[:, keep])
    x_test = scaler.transform(test[:, keep])
    count = min(PCA_COMPONENTS, x_train.shape[1], x_train.shape[0] - 1)
    if count < x_train.shape[1]:
        pca = PCA(
            n_components=count, svd_solver="randomized", random_state=seed
        ).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test


def seed_folds(seeds: np.ndarray) -> np.ndarray:
    unique = np.sort(np.unique(seeds))
    if len(unique) != 32:
        raise ValueError("expected 32 common noise seeds")
    rank = {int(seed): axis for axis, seed in enumerate(unique)}
    return np.asarray([rank[int(seed)] // 8 for seed in seeds], dtype=np.int8)


def evaluation_splits(data: TrendTask, evaluation: str):
    unique_scenes = np.unique(data.scenes)
    if evaluation == "seed_heldout":
        folds = seed_folds(data.seeds)
        splits = [(folds != fold, folds == fold) for fold in range(4)]
        scene_axis = {int(scene): axis for axis, scene in enumerate(unique_scenes)}
        strata = np.asarray(
            [scene_axis[int(scene)] * 4 + int(fold) for scene, fold in zip(data.scenes, folds)]
        )
        onehot = (data.scenes[:, None] == unique_scenes[None, :]).astype(np.float64)
        return splits, strata, onehot
    if evaluation == "scene_loso":
        splits = [
            (data.scenes != scene, data.scenes == scene) for scene in unique_scenes
        ]
        return splits, data.scenes.astype(np.int64), None
    raise ValueError("unknown evaluation %s" % evaluation)


def crossfit_query(
    data: TrendTask, chunk: int, evaluation: str, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    blocks = feature_blocks(data, chunk)
    splits, strata, onehot = evaluation_splits(data, evaluation)
    labels = data.failure.astype(np.int8)
    predictions = {
        family: np.full(len(labels), np.nan, dtype=np.float64)
        for family in MODEL_FAMILIES
    }
    manifold = {
        family: np.full(len(labels), np.nan, dtype=np.float64)
        for family in MODEL_FAMILIES
    }
    for fold, (train, test) in enumerate(splits):
        if len(np.unique(labels[train])) != 2:
            raise ValueError("one-class training fold")
        reduced = {
            name: _reduce_block(
                value[train], value[test], seed + chunk * 1000 + fold * 10 + axis
            )
            for axis, (name, value) in enumerate(blocks.items())
        }
        for family, names in MODEL_FAMILIES.items():
            train_parts = [reduced[name][0] for name in names]
            test_parts = [reduced[name][1] for name in names]
            if onehot is not None:
                train_parts.insert(0, onehot[train])
                test_parts.insert(0, onehot[test])
            x_train = np.column_stack(train_parts)
            x_test = np.column_stack(test_parts)
            scaler = StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            x_test = scaler.transform(x_test)
            model = LogisticRegression(
                C=LOGISTIC_C,
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",
            ).fit(x_train, labels[train])
            predictions[family][test] = model.predict_proba(x_test)[:, 1]

            if evaluation == "seed_heldout":
                train_indices = np.flatnonzero(train)
                test_indices = np.flatnonzero(test)
                for scene in np.unique(data.scenes):
                    success_local = (
                        (data.scenes[train_indices] == scene)
                        & ~data.failure[train_indices]
                    )
                    test_local = data.scenes[test_indices] == scene
                    if success_local.sum() < 3 or not np.any(test_local):
                        continue
                    center = x_train[success_local].mean(axis=0)
                    manifold[family][test_indices[test_local]] = np.sqrt(
                        np.mean(np.square(x_test[test_local] - center), axis=1)
                    )
    if any(np.any(~np.isfinite(score)) for score in predictions.values()):
        raise RuntimeError("cross-fitted classifier predictions are incomplete")
    return predictions, manifold, strata


def conditional_scene_stats(
    labels: np.ndarray,
    scores: np.ndarray,
    scenes: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    unique_scenes = np.unique(scenes)
    wins = np.zeros(len(unique_scenes), dtype=np.float64)
    pairs = np.zeros(len(unique_scenes), dtype=np.int64)
    for axis, scene in enumerate(unique_scenes):
        scene_mask = scenes == scene
        for stratum in np.unique(strata[scene_mask]):
            mask = scene_mask & (strata == stratum) & np.isfinite(scores)
            positive = scores[mask & labels]
            negative = scores[mask & ~labels]
            if not len(positive) or not len(negative):
                continue
            wins[axis] += float((positive[:, None] > negative[None, :]).sum())
            wins[axis] += 0.5 * float((positive[:, None] == negative[None, :]).sum())
            pairs[axis] += len(positive) * len(negative)
    return wins, pairs


def _auc(wins: np.ndarray, pairs: np.ndarray) -> float:
    return float(wins.sum() / pairs.sum()) if pairs.sum() else float("nan")


def _bootstrap_auc(
    wins: np.ndarray, pairs: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    sampled_pairs = pairs[draws].sum(axis=1)
    return np.divide(
        wins[draws].sum(axis=1),
        sampled_pairs,
        out=np.full(len(draws), np.nan),
        where=sampled_pairs > 0,
    )


def evaluate_task(
    data: TrendTask, bootstrap: int, seed: int
) -> tuple[list[dict], list[dict], dict, dict]:
    rows = []
    manifold_rows = []
    classifier_boots = {}
    manifold_boots = {}
    rng = np.random.default_rng(seed)
    n_scenes = len(np.unique(data.scenes))
    draws = rng.integers(0, n_scenes, size=(bootstrap, n_scenes))
    for evaluation in ("seed_heldout", "scene_loso"):
        for chunk in range(data.routed.shape[1]):
            predictions, manifold, strata = crossfit_query(
                data, chunk, evaluation, seed
            )
            for family, score in predictions.items():
                wins, pairs = conditional_scene_stats(
                    data.failure, score, data.scenes, strata
                )
                boot = _bootstrap_auc(wins, pairs, draws)
                valid = boot[np.isfinite(boot)]
                classifier_boots[(evaluation, chunk, family)] = boot
                rows.append(
                    {
                        "task": data.task,
                        "evaluation": evaluation,
                        "chunk": chunk,
                        "family": family,
                        "conditional_auc": _auc(wins, pairs),
                        "ci_low": float(np.percentile(valid, 2.5)),
                        "ci_high": float(np.percentile(valid, 97.5)),
                        "pairs": int(pairs.sum()),
                        "mixed_scenes": int(np.sum(pairs > 0)),
                    }
                )
            if evaluation == "seed_heldout":
                for family, score in manifold.items():
                    wins, pairs = conditional_scene_stats(
                        data.failure, score, data.scenes, strata
                    )
                    boot = _bootstrap_auc(wins, pairs, draws)
                    valid = boot[np.isfinite(boot)]
                    manifold_boots[(chunk, family)] = boot
                    manifold_rows.append(
                        {
                            "task": data.task,
                            "chunk": chunk,
                            "family": family,
                            "conditional_auc": _auc(wins, pairs),
                            "ci_low": float(np.percentile(valid, 2.5)),
                            "ci_high": float(np.percentile(valid, 97.5)),
                            "pairs": int(pairs.sum()),
                            "mixed_scenes": int(np.sum(pairs > 0)),
                        }
                    )
            print(
                "%s: %s query %d" % (data.task, evaluation, chunk), flush=True
            )
    return rows, manifold_rows, classifier_boots, manifold_boots


def descriptive_values(data: TrendTask) -> dict[str, np.ndarray]:
    # Scalar layout is state, action mean, action std; each group has seven values.
    action_offset = len(SCALAR_NAMES)
    scalar = data.scalar
    return {
        "routed_rms": scalar[..., action_offset + 1].mean(axis=(2, 3)),
        "cancellation": scalar[..., action_offset + 3].mean(axis=(2, 3)),
        "shared_conflict": scalar[..., action_offset + 6].mean(axis=(2, 3)),
        "within_query_path": data.routed_adjacent[..., 1].astype(np.float32).mean(
            axis=(2, 3)
        ),
        "d0_to_d9_commitment": data.routed_commitment[:, :, 0, :, 1]
        .astype(np.float32)
        .mean(axis=2),
    }


def descriptive_rows(data: TrendTask) -> tuple[list[dict], dict]:
    rows = []
    per_scene = {}
    for metric, matrix in descriptive_values(data).items():
        task_mean = matrix.mean()
        task_std = matrix.std()
        standardized = (matrix - task_mean) / max(task_std, 1e-12)
        for chunk in range(matrix.shape[1]):
            gaps = []
            for scene in np.unique(data.scenes):
                mask = data.scenes == scene
                positive = standardized[mask & data.failure, chunk]
                negative = standardized[mask & ~data.failure, chunk]
                gaps.append(
                    float(positive.mean() - negative.mean())
                    if len(positive) and len(negative)
                    else float("nan")
                )
            values = np.asarray(gaps, dtype=np.float64)
            per_scene[(metric, chunk)] = values
            rows.append(
                {
                    "task": data.task,
                    "metric": metric,
                    "chunk": chunk,
                    "failure_minus_success_z": float(np.nanmean(values)),
                    "mixed_scenes": int(np.isfinite(values).sum()),
                    "per_scene": values.tolist(),
                }
            )
    return rows, per_scene


def macro_results(
    tasks: list[TrendTask],
    classifier_rows: list[dict],
    manifold_rows: list[dict],
    classifier_boots: dict,
    manifold_boots: dict,
) -> tuple[list[dict], list[dict], dict]:
    task_names = [task.task for task in tasks]
    classifier_lookup = {
        (row["task"], row["evaluation"], row["chunk"], row["family"]): row
        for row in classifier_rows
    }
    manifold_lookup = {
        (row["task"], row["chunk"], row["family"]): row
        for row in manifold_rows
    }
    classifier_macro = []
    manifold_macro = []
    trends = {"classifier": {}, "manifold": {}}
    for evaluation in ("seed_heldout", "scene_loso"):
        for family in MODEL_FAMILIES:
            family_boot = {}
            points = {}
            for chunk in range(tasks[0].routed.shape[1]):
                values = [
                    classifier_lookup[(task, evaluation, chunk, family)]["conditional_auc"]
                    for task in task_names
                ]
                boot = np.nanmean(
                    [classifier_boots[(task, evaluation, chunk, family)] for task in task_names],
                    axis=0,
                )
                valid = boot[np.isfinite(boot)]
                family_boot[chunk] = boot
                points[chunk] = values
                classifier_macro.append(
                    {
                        "evaluation": evaluation,
                        "family": family,
                        "chunk": chunk,
                        "macro_auc": float(np.mean(values)),
                        "ci_low": float(np.percentile(valid, 2.5)),
                        "ci_high": float(np.percentile(valid, 97.5)),
                        "tasks_above_chance": int(sum(value > 0.5 for value in values)),
                    }
                )
            contrast = family_boot[8] - family_boot[0]
            contrast = contrast[np.isfinite(contrast)]
            per_task = [points[8][i] - points[0][i] for i in range(len(task_names))]
            trends["classifier"][evaluation + ":" + family] = {
                "estimate": float(np.mean(per_task)),
                "ci95": [
                    float(np.percentile(contrast, 2.5)),
                    float(np.percentile(contrast, 97.5)),
                ],
                "per_task": per_task,
                "tasks_positive": int(sum(value > 0 for value in per_task)),
            }

    for family in MODEL_FAMILIES:
        family_boot = {}
        points = {}
        for chunk in range(tasks[0].routed.shape[1]):
            values = [
                manifold_lookup[(task, chunk, family)]["conditional_auc"]
                for task in task_names
            ]
            boot = np.nanmean(
                [manifold_boots[(task, chunk, family)] for task in task_names], axis=0
            )
            valid = boot[np.isfinite(boot)]
            family_boot[chunk] = boot
            points[chunk] = values
            manifold_macro.append(
                {
                    "family": family,
                    "chunk": chunk,
                    "macro_auc": float(np.mean(values)),
                    "ci_low": float(np.percentile(valid, 2.5)),
                    "ci_high": float(np.percentile(valid, 97.5)),
                    "tasks_above_chance": int(sum(value > 0.5 for value in values)),
                }
            )
        contrast = family_boot[8] - family_boot[0]
        contrast = contrast[np.isfinite(contrast)]
        per_task = [points[8][i] - points[0][i] for i in range(len(task_names))]
        trends["manifold"][family] = {
            "estimate": float(np.mean(per_task)),
            "ci95": [
                float(np.percentile(contrast, 2.5)),
                float(np.percentile(contrast, 97.5)),
            ],
            "per_task": per_task,
            "tasks_positive": int(sum(value > 0 for value in per_task)),
        }
    return classifier_macro, manifold_macro, trends


def macro_descriptive(
    tasks: list[TrendTask], rows: list[dict], scene_values: dict, bootstrap: int, seed: int
) -> tuple[list[dict], dict]:
    task_names = [task.task for task in tasks]
    lookup = {(row["task"], row["metric"], row["chunk"]): row for row in rows}
    rng = np.random.default_rng(seed)
    macro = []
    trends = {}
    for metric in DESCRIPTIVE_METRICS:
        boots = {}
        points = {}
        paired_draws = {}
        for task in task_names:
            scene = np.asarray(scene_values[(task, metric, 0)], dtype=np.float64)
            valid_axis = np.flatnonzero(np.isfinite(scene))
            paired_draws[task] = (
                valid_axis,
                rng.integers(0, len(valid_axis), size=(bootstrap, len(valid_axis))),
            )
        for chunk in range(tasks[0].routed.shape[1]):
            task_boot = []
            values = []
            for task in task_names:
                scene = np.asarray(scene_values[(task, metric, chunk)], dtype=np.float64)
                valid_axis, draws = paired_draws[task]
                if not np.all(np.isfinite(scene[valid_axis])):
                    raise ValueError("mixed-scene support changed across fixed-cohort chunks")
                task_boot.append(scene[valid_axis][draws].mean(axis=1))
                values.append(lookup[(task, metric, chunk)]["failure_minus_success_z"])
            boot = np.mean(task_boot, axis=0)
            boots[chunk] = boot
            points[chunk] = values
            macro.append(
                {
                    "metric": metric,
                    "chunk": chunk,
                    "failure_minus_success_z": float(np.mean(values)),
                    "ci_low": float(np.percentile(boot, 2.5)),
                    "ci_high": float(np.percentile(boot, 97.5)),
                }
            )
        contrast = boots[8] - boots[0]
        per_task = [points[8][i] - points[0][i] for i in range(len(task_names))]
        trends[metric] = {
            "estimate": float(np.mean(per_task)),
            "ci95": [
                float(np.percentile(contrast, 2.5)),
                float(np.percentile(contrast, 97.5)),
            ],
            "per_task": per_task,
            "tasks_positive": int(sum(value > 0 for value in per_task)),
        }
    return macro, trends


def _write_csv(path: pathlib.Path, rows: list[dict], exclude: set[str] | None = None):
    exclude = exclude or set()
    clean = [{k: v for k, v in row.items() if k not in exclude} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)


def render_report(summary: dict) -> str:
    classifier = {
        (row["evaluation"], row["family"], row["chunk"]): row
        for row in summary["classifier_macro"]
    }
    manifold = {
        (row["family"], row["chunk"]): row for row in summary["manifold_macro"]
    }
    descriptive = {
        (row["metric"], row["chunk"]): row for row in summary["descriptive_macro"]
    }
    trends = summary["trends"]
    routed_classifier_trend = trends["classifier"]["seed_heldout:routed_full"]
    routed_loso_trend = trends["classifier"]["scene_loso:routed_full"]
    routed_manifold_trend = trends["manifold"]["routed_full"]

    def c(evaluation: str, family: str, chunk: int) -> float:
        return classifier[(evaluation, family, chunk)]["macro_auc"]

    lines = [
        "# Success/failure MoE trend across single chunks",
        "",
        "## Bottom line",
        "",
        "Each point uses only one query's internal ten-denoise computation. "
        "There is no robust k0 precursor. At k1, routed direction alone has a weak "
        "same-state signal (AUC %.3f, 95%% CI [%.3f, %.3f], %d/4 tasks above "
        "chance), but routed-full remains at %.3f and the direction signal is %.3f "
        "under unseen-initial-state evaluation."
        % (
            c("seed_heldout", "routed_identity", 1),
            classifier[("seed_heldout", "routed_identity", 1)]["ci_low"],
            classifier[("seed_heldout", "routed_identity", 1)]["ci_high"],
            classifier[("seed_heldout", "routed_identity", 1)][
                "tasks_above_chance"
            ],
            c("seed_heldout", "routed_full", 1),
            c("scene_loso", "routed_identity", 1),
        ),
        "",
        "The primary routed-full separation appears at k2/k3, disappears at k4, "
        "and becomes consistently strong only from k6 through k8 (%.3f, %.3f, "
        "%.3f)."
        % (
            c("seed_heldout", "routed_full", 6),
            c("seed_heldout", "routed_full", 7),
            c("seed_heldout", "routed_full", 8),
        ),
        "",
        "Within the same initial states, routed-full AUC changes from %.3f at k0 "
        "to %.3f at k8 (delta %+.3f, 95%% CI [%+.3f, %+.3f], %d/4 task "
        "contrasts positive)."
        % (
            c("seed_heldout", "routed_full", 0),
            c("seed_heldout", "routed_full", 8),
            routed_classifier_trend["estimate"],
            routed_classifier_trend["ci95"][0],
            routed_classifier_trend["ci95"][1],
            routed_classifier_trend["tasks_positive"],
        ),
        "",
        "Distance from the successful routed-computation manifold changes from %.3f "
        "to %.3f (delta %+.3f, 95%% CI [%+.3f, %+.3f], %d/4 tasks positive)."
        % (
            manifold[("routed_full", 0)]["macro_auc"],
            manifold[("routed_full", 8)]["macro_auc"],
            routed_manifold_trend["estimate"],
            routed_manifold_trend["ci95"][0],
            routed_manifold_trend["ci95"][1],
            routed_manifold_trend["tasks_positive"],
        ),
        "",
        "This is not a routed-expert-specific effect: at k8 hidden and base AUC "
        "are %.3f and %.3f, while adding routed features to hidden+shared changes "
        "AUC from %.3f to %.3f. The secondary unseen-initial-state test is also "
        "not stable for routed-full (k0 %.3f, k8 %.3f; delta %+.3f, 95%% CI "
        "[%+.3f, %+.3f], %d/4 task contrasts positive)."
        % (
            c("seed_heldout", "hidden_identity", 8),
            c("seed_heldout", "base", 8),
            c("seed_heldout", "base", 8),
            c("seed_heldout", "base_routed", 8),
            c("scene_loso", "routed_full", 0),
            c("scene_loso", "routed_full", 8),
            routed_loso_trend["estimate"],
            routed_loso_trend["ci95"][0],
            routed_loso_trend["ci95"][1],
            routed_loso_trend["tasks_positive"],
        ),
        "",
        "## Scope",
        "",
        "- Fixed 2048-rollout cohort: four tasks, 16 initial states x 32 seeds per task.",
        "- Queries k0..k8 are evaluated separately; no predictor reads another chunk.",
        "- All rollouts are active through k8. k9 is excluded to avoid success-dependent termination.",
        "- Label: eventual failure, not annotated trap onset.",
        "- Primary evaluation holds out complete seed groups and compares scores only within the same initial state and fold.",
        "- Routed contributions use recorded top-4 IDs/weights and checkpoint expert outputs at layers 2/5/12/15.",
        "",
        "## Primary trends (k8 minus k0)",
        "",
        "| readout | representation | delta | 95% CI | positive tasks |",
        "|---|---|---:|---:|---:|",
    ]
    for kind, family, label in (
        ("classifier", "routed_full", "failure classifier"),
        ("classifier", "hidden_identity", "failure classifier"),
        ("classifier", "shared_identity", "failure classifier"),
        ("classifier", "base", "failure classifier"),
        ("classifier", "base_routed", "failure classifier"),
        ("manifold", "routed_full", "success-manifold distance"),
        ("manifold", "hidden_identity", "success-manifold distance"),
        ("manifold", "shared_identity", "success-manifold distance"),
        ("manifold", "base", "success-manifold distance"),
        ("manifold", "base_routed", "success-manifold distance"),
    ):
        key = "seed_heldout:" + family if kind == "classifier" else family
        row = trends[kind][key]
        lines.append(
            "| %s | %s | %+.3f | [%+.3f, %+.3f] | %d/4 |"
            % (label, family, row["estimate"], row["ci95"][0], row["ci95"][1], row["tasks_positive"])
        )
    lines.extend(
        [
            "",
            "## Single-chunk curves",
            "",
            "| k | routed direction | routed full | hidden | base | base+routed | routed manifold | hidden manifold |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for chunk in range(summary["max_chunks"]):
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                chunk,
                classifier[("seed_heldout", "routed_identity", chunk)][
                    "macro_auc"
                ],
                classifier[("seed_heldout", "routed_full", chunk)]["macro_auc"],
                classifier[("seed_heldout", "hidden_identity", chunk)]["macro_auc"],
                classifier[("seed_heldout", "base", chunk)]["macro_auc"],
                classifier[("seed_heldout", "base_routed", chunk)]["macro_auc"],
                manifold[("routed_full", chunk)]["macro_auc"],
                manifold[("hidden_identity", chunk)]["macro_auc"],
            )
        )
    lines.extend(
        [
            "",
            "## Secondary unseen-state check",
            "",
            "This classifier is trained on 15 initial states and evaluated on the "
            "held-out state. It is a harder generalization test than the repeated-"
            "rollout comparison above.",
            "",
            "| k | routed direction | routed full | routed scalar | hidden | base | base+routed |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for chunk in range(summary["max_chunks"]):
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                chunk,
                c("scene_loso", "routed_identity", chunk),
                c("scene_loso", "routed_full", chunk),
                c("scene_loso", "routed_scalar", chunk),
                c("scene_loso", "hidden_identity", chunk),
                c("scene_loso", "base", chunk),
                c("scene_loso", "base_routed", chunk),
            )
        )
    lines.extend(
        [
            "",
            "## Direct success/failure computation gaps",
            "",
            "Positive values mean the metric is higher in failed rollouts after task-wide standardization.",
            "",
            "| metric | k0 gap | k4 gap | k8 gap | k8-k0 (95% CI) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric in DESCRIPTIVE_METRICS:
        trend = summary["descriptive_trends"][metric]
        lines.append(
            "| %s | %+.3f | %+.3f | %+.3f | %+.3f [%+.3f, %+.3f] |"
            % (
                metric,
                descriptive[(metric, 0)]["failure_minus_success_z"],
                descriptive[(metric, 4)]["failure_minus_success_z"],
                descriptive[(metric, 8)]["failure_minus_success_z"],
                trend["estimate"],
                trend["ci95"][0],
                trend["ci95"][1],
            )
        )
    lines.extend(
        [
            "",
            "## Per-task routed trend",
            "",
            "| task | classifier k0 | classifier k8 | delta | manifold k0 | manifold k8 | delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    class_by_task = {
        (r["task"], r["chunk"]): r
        for r in summary["classifier_by_task"]
        if r["evaluation"] == "seed_heldout" and r["family"] == "routed_full"
    }
    manifold_by_task = {
        (r["task"], r["chunk"]): r
        for r in summary["manifold_by_task"]
        if r["family"] == "routed_full"
    }
    for task in summary["tasks"]:
        c0 = class_by_task[(task, 0)]["conditional_auc"]
        c8 = class_by_task[(task, 8)]["conditional_auc"]
        m0 = manifold_by_task[(task, 0)]["conditional_auc"]
        m8 = manifold_by_task[(task, 8)]["conditional_auc"]
        lines.append(
            "| %s | %.3f | %.3f | %+.3f | %.3f | %.3f | %+.3f |"
            % (task, c0, c8, c8 - c0, m0, m8, m8 - m0)
        )
    validation = [row for rows in summary["validation"].values() for row in rows]
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A growing curve means failed and successful closed-loop trajectories become more separable in single-query MoE computation. It does not show that an earlier MoE state caused the later separation, because later observations are descendants of earlier actions.",
            "",
            "Hidden/shared controls test routed specificity. If they grow equally or more, the defensible interpretation is general model-state divergence rather than an extra routed-expert precursor.",
            "",
            "Contributions are reconstructed from fp16 hidden captures and are not runtime-exact. Recorded IDs and weights are authoritative; selected-probability reconstruction has maximum MAE %.2g and the fp16 gate audit recovers at least %.1f%% of top-4 sets."
            % (
                max(row["selected_probability_mae"] for row in validation),
                100.0 * min(row["top4_set_match"] for row in validation),
            ),
            "",
            "This is exploratory because earlier query-0 contribution and query-0..4 hidden/router results were already inspected.",
            "",
            "The supported conclusion is therefore trajectory-level internal-state "
            "divergence after several closed-loop interactions. These offline data "
            "do not support the stronger claim that the initial MoE state is already "
            "wrong, uniquely predicts failure, or causally pushes the system into it.",
            "",
        ]
    )
    return "\n".join(lines)


def make_plot(summary: dict, path: pathlib.Path):
    import matplotlib.pyplot as plt

    classifier = {
        (row["evaluation"], row["family"], row["chunk"]): row
        for row in summary["classifier_macro"]
    }
    manifold = {
        (row["family"], row["chunk"]): row for row in summary["manifold_macro"]
    }
    x = np.arange(summary["max_chunks"])
    colors = {
        "routed_full": "#C44536",
        "hidden_identity": "#2A9D8F",
        "shared_identity": "#4C566A",
        "base": "#7A5195",
        "base_routed": "#E09F3E",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()
    for family in colors:
        axes[0].plot(
            x,
            [classifier[("seed_heldout", family, int(k))]["macro_auc"] for k in x],
            marker="o",
            label=family,
            color=colors[family],
        )
        axes[1].plot(
            x,
            [manifold[(family, int(k))]["macro_auc"] for k in x],
            marker="o",
            label=family,
            color=colors[family],
        )
    axes[0].plot(
        x,
        [classifier[("seed_heldout", "routed_identity", int(k))]["macro_auc"] for k in x],
        marker="o",
        linestyle="--",
        label="routed_direction",
        color="#D97B66",
    )
    for axis in axes[:2]:
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel="control query k", ylabel="conditional AUC", xticks=x)
        axis.legend(frameon=False, fontsize=7)
    axes[0].set_title("Single-chunk failure decoding")
    axes[1].set_title("Distance from success manifold")
    base = np.asarray(
        [classifier[("seed_heldout", "base", int(k))]["macro_auc"] for k in x]
    )
    full = np.asarray(
        [classifier[("seed_heldout", "base_routed", int(k))]["macro_auc"] for k in x]
    )
    axes[2].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[2].plot(x, full - base, marker="o", color="#C44536")
    axes[2].set(
        xlabel="control query k",
        ylabel="conditional AUC delta",
        xticks=x,
        title="Add routed to hidden + shared",
    )
    for family, color, linestyle in (
        ("routed_identity", "#D97B66", "--"),
        ("routed_full", "#C44536", "-"),
        ("routed_scalar", "#E09F3E", "-"),
    ):
        axes[3].plot(
            x,
            [classifier[("scene_loso", family, int(k))]["macro_auc"] for k in x],
            marker="o",
            label=family,
            color=color,
            linestyle=linestyle,
        )
    axes[3].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[3].set(
        xlabel="control query k",
        ylabel="conditional AUC",
        xticks=x,
        title="Leave-one-initial-state-out",
    )
    axes[3].legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary_artifacts(summary: dict, out_dir: pathlib.Path) -> None:
    # Per-scene descriptive arrays contain NaN for single-outcome scenes and are
    # only needed while bootstrapping. Keep the public summary strict JSON.
    summary["descriptive_by_task"] = [
        {key: value for key, value in row.items() if key != "per_scene"}
        for row in summary["descriptive_by_task"]
    ]
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (out_dir / "report.md").write_text(render_report(summary))
    make_plot(summary, out_dir / "moe_rollout_trend.png")


def main() -> None:
    args = parse_args()
    if args.max_chunks != N_CHUNKS:
        raise ValueError("this frozen analysis requires exactly nine queries")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        summary_path = args.out_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        write_summary_artifacts(summary, args.out_dir)
        print("regenerated %s" % args.out_dir, flush=True)
        return
    requested = set(args.task) if args.task else None
    runs = discover_runs(args.cache_root, requested)
    if args.layer_shard is not None:
        if len(runs) != 1:
            raise ValueError("--layer-shard requires exactly one --task")
        extract_layer_shard(
            runs[0],
            args.cache_root,
            args.out_dir,
            args.layer_shard,
            args.max_chunks,
            PROJECTED_DIM,
            args.threads,
            args.seed,
            args.force_compact,
        )
        return
    if args.assemble_only:
        for run in runs:
            assemble_task_shards(
                run,
                args.cache_root,
                args.out_dir,
                args.max_chunks,
                PROJECTED_DIM,
            )
        return
    tasks = [
        extract_task(
            run,
            args.cache_root,
            args.out_dir,
            args.max_chunks,
            PROJECTED_DIM,
            args.threads,
            args.seed,
            args.force_compact,
        )
        for run in runs
    ]
    if args.extract_only:
        return
    if len(tasks) != 4:
        raise ValueError("macro analysis requires all four variable-outcome tasks")

    classifier_rows = []
    manifold_rows = []
    descriptive_output = []
    classifier_boots = {}
    manifold_boots = {}
    descriptive_scene = {}
    for task_axis, task in enumerate(tasks):
        rows, manifold, class_boot, mani_boot = evaluate_task(
            task, args.bootstrap, args.seed + task_axis * 10000
        )
        classifier_rows.extend(rows)
        manifold_rows.extend(manifold)
        for key, value in class_boot.items():
            classifier_boots[(task.task, *key)] = value
        for key, value in mani_boot.items():
            manifold_boots[(task.task, *key)] = value
        rows, scene = descriptive_rows(task)
        descriptive_output.extend(rows)
        for key, value in scene.items():
            descriptive_scene[(task.task, *key)] = value

    classifier_macro, manifold_macro, trends = macro_results(
        tasks,
        classifier_rows,
        manifold_rows,
        classifier_boots,
        manifold_boots,
    )
    descriptive_macro, descriptive_trends = macro_descriptive(
        tasks,
        descriptive_output,
        descriptive_scene,
        args.bootstrap,
        args.seed + 900000,
    )
    summary = {
        "analysis": "success/failure single-chunk MoE rollout trend",
        "exploratory": True,
        "feature_version": FEATURE_VERSION,
        "tasks": [task.task for task in tasks],
        "max_chunks": args.max_chunks,
        "layers": list(LAYERS),
        "projected_dim": PROJECTED_DIM,
        "pca_components": PCA_COMPONENTS,
        "logistic_c": LOGISTIC_C,
        "bootstrap": args.bootstrap,
        "validation": {task.task: task.validation for task in tasks},
        "classifier_macro": classifier_macro,
        "manifold_macro": manifold_macro,
        "trends": trends,
        "descriptive_macro": descriptive_macro,
        "descriptive_trends": descriptive_trends,
        "classifier_by_task": classifier_rows,
        "manifold_by_task": manifold_rows,
        "descriptive_by_task": descriptive_output,
    }
    _write_csv(args.out_dir / "classifier.csv", classifier_rows)
    _write_csv(args.out_dir / "manifold.csv", manifold_rows)
    _write_csv(args.out_dir / "descriptive.csv", descriptive_output, {"per_scene"})
    write_summary_artifacts(summary, args.out_dir)
    print("wrote %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
