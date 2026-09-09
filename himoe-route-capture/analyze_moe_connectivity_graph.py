"""Candidate-connectivity analysis for the HB5/d0 same-state K=32 corpus.

The confirmatory endpoints follow ``MOE_CONNECTIVITY_GRAPH_DESIGN.md``.  The
primary graph is built from the full 32-way router distribution, not from the
storage order of the selected expert IDs.  All endpoint permutations use the
same seed-column permutation in every state and every task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata


TASKS = (
    "goal-middle",
    "goal-top",
    "long-t08",
    "spatial-ramekin",
    "spatial-stove",
)
N_SEEDS = 32
N_SCENES = 16
MAX_K = 8
ACTION_TOKEN_SLICE = slice(1, 11)
EXPECTED_SUCCESSES = {
    "goal-middle": 512, "goal-top": 470, "long-t08": 296,
    "spatial-ramekin": 500, "spatial-stove": 475,
}
EXPECTED_MIXED = {
    "goal-middle": 0, "goal-top": 7, "long-t08": 13,
    "spatial-ramekin": 8, "spatial-stove": 12,
}
EXPECTED_BALANCED = {
    "goal-middle": 0, "goal-top": 3, "long-t08": 10,
    "spatial-ramekin": 1, "spatial-stove": 2,
}


@dataclass
class TaskData:
    name: str
    scenes: np.ndarray
    seeds: np.ndarray
    actions: np.ndarray
    success: np.ndarray
    router_probs: np.ndarray
    hidden: np.ndarray
    expert_ids: np.ndarray
    selected_prob: np.ndarray
    action_std: np.ndarray
    proxies: dict[str, np.ndarray]
    validation: dict[str, object]


def _finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains non-finite values")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_normalization_stats_path(metadata_path: Path, recorded_path: str) -> Path:
    recorded = Path(recorded_path)
    portable = Path(metadata_path).with_name("normalization_stats.json")
    if recorded.exists():
        return recorded
    if portable.exists():
        return portable
    raise FileNotFoundError(
        f"neither recorded nor portable normalization stats exist: {recorded}, {portable}"
    )


def hellinger_distance(probabilities: np.ndarray) -> np.ndarray:
    """Pairwise root-mean Hellinger distance over all aligned routing sites."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim < 2:
        raise ValueError("probabilities need candidate and expert axes")
    if np.any(~np.isfinite(p)):
        raise ValueError("probabilities contain non-finite values")
    p = np.maximum(p, 0.0)
    mass = p.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("routing site has no positive probability mass")
    root = np.sqrt(p / mass)
    n_sites = int(np.prod(root.shape[1:-1])) or 1
    flat = root.reshape(root.shape[0], -1)
    affinity = (flat @ flat.T) / n_sites
    distance2 = np.maximum(1.0 - affinity, 0.0)
    distance = np.sqrt(distance2)
    np.fill_diagonal(distance, 0.0)
    return distance


def pairwise_rms(values: np.ndarray) -> np.ndarray:
    """Pairwise RMS distance over every axis except the candidate axis."""
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    if x.shape[1] == 0:
        raise ValueError("values have no feature coordinates")
    if np.any(~np.isfinite(x)):
        raise ValueError("values contain non-finite coordinates")
    squared_norm = np.einsum("ij,ij->i", x, x)
    distance2 = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * (x @ x.T)
    ) / x.shape[1]
    distance = np.sqrt(np.maximum(distance2, 0.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def topk_jaccard_distance(expert_ids: np.ndarray) -> np.ndarray:
    """Mean order-invariant top-k Jaccard distance across aligned sites."""
    ids = np.asarray(expert_ids)
    if ids.ndim == 2:
        ids = ids[:, None, :]
    n, n_sites, _ = ids.shape
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            distances = []
            for site in range(n_sites):
                left = set(int(v) for v in ids[i, site])
                right = set(int(v) for v in ids[j, site])
                distances.append(1.0 - len(left & right) / len(left | right))
            out[i, j] = out[j, i] = float(np.mean(distances))
    return out


def selected_probability_tensor(
    expert_ids: np.ndarray, selected_prob: np.ndarray, n_experts: int = 32
) -> np.ndarray:
    ids = np.asarray(expert_ids, dtype=np.int64)
    weights = np.asarray(selected_prob, dtype=np.float64)
    if ids.shape != weights.shape:
        raise ValueError("selected IDs and probabilities have different shapes")
    out = np.zeros(ids.shape[:-1] + (n_experts,), dtype=np.float64)
    np.put_along_axis(out, ids, weights, axis=-1)
    mass = out.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("selected expert mass is zero")
    return out / mass


def _target_rank_matrix(distance: np.ndarray) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64)
    n = len(d)
    if d.shape != (n, n) or n < 3:
        raise ValueError("distance must be a square matrix with at least 3 nodes")
    ranks = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        keep = np.arange(n) != i
        ranks[i, keep] = (rankdata(d[i, keep], method="average") - 1.0) / (n - 2)
    return ranks


def _neighbor_order(distance: np.ndarray, max_k: int) -> np.ndarray:
    d = np.asarray(distance, dtype=np.float64).copy()
    if d.shape[0] != d.shape[1] or max_k >= len(d):
        raise ValueError("invalid distance matrix or max_k")
    np.fill_diagonal(d, np.inf)
    # Input candidates are in ascending seed order; stable sorting therefore
    # implements the frozen ascending-seed tie break.
    return np.argsort(d, axis=1, kind="stable")[:, :max_k]


def auk_curve(
    source_distance: np.ndarray, target_distance: np.ndarray, max_k: int = MAX_K
) -> np.ndarray:
    neighbors = _neighbor_order(source_distance, max_k)
    target_ranks = _target_rank_matrix(target_distance)
    selected = target_ranks[np.arange(len(neighbors))[:, None], neighbors]
    cumulative = np.cumsum(selected, axis=1) / np.arange(1, max_k + 1)
    return 1.0 - 2.0 * cumulative.mean(axis=0)


def auk8(
    source_distance: np.ndarray, target_distance: np.ndarray, max_k: int = MAX_K
) -> float:
    return float(np.mean(auk_curve(source_distance, target_distance, max_k)))


def tie_aware_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    raw_labels = np.asarray(labels)
    score = np.asarray(scores, dtype=np.float64)
    if raw_labels.ndim != 1 or score.ndim != 1 or len(raw_labels) != len(score):
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(score)):
        raise ValueError("scores contain non-finite values")
    if not np.all(np.isin(raw_labels, (0, 1, False, True))):
        raise ValueError("labels must be binary")
    y = raw_labels.astype(bool)
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    if n_positive == 0 or n_negative == 0:
        raise ValueError("AUC requires both label classes")
    ranks = rankdata(score, method="average")
    wins = ranks[y].sum() - n_positive * (n_positive + 1) / 2.0
    return float(wins / (n_positive * n_negative))


def common_seed_permute(
    arrays: Iterable[np.ndarray], permutation: np.ndarray
) -> list[np.ndarray]:
    """Apply one seed-column permutation to aligned ``[pool, seed, ...]`` arrays."""
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.ndim != 1 or not np.array_equal(
        np.sort(permutation), np.arange(len(permutation))
    ):
        raise ValueError("permutation must contain every seed-column index once")
    result = []
    for value in arrays:
        array = np.asarray(value)
        if array.ndim < 2 or array.shape[1] != len(permutation):
            raise ValueError("each array must have seed as axis 1")
        result.append(np.take(array, permutation, axis=1))
    return result


def _mst_edges(distance: np.ndarray) -> list[tuple[int, int, float]]:
    """Deterministic Kruskal MST, with lexicographic tie breaking."""
    d = np.asarray(distance, dtype=np.float64)
    n = len(d)
    candidates = sorted(
        (float(d[i, j]), i, j) for i in range(n) for j in range(i + 1, n)
    )
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges = []
    for weight, left, right in candidates:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        parent[root_right] = root_left
        edges.append((left, right, weight))
        if len(edges) == n - 1:
            break
    if len(edges) != n - 1:
        raise ValueError("distance graph is disconnected or non-finite")
    return edges


def balanced_mst_cut(distance: np.ndarray, min_size: int = 4) -> np.ndarray:
    """Remove the largest MST edge whose two components both meet ``min_size``."""
    edges = _mst_edges(distance)
    n = len(distance)
    choices = []
    for edge_index, (left, right, weight) in enumerate(edges):
        adjacency = [[] for _ in range(n)]
        for other_index, (a, b, _) in enumerate(edges):
            if other_index == edge_index:
                continue
            adjacency[a].append(b)
            adjacency[b].append(a)
        seen = {left}
        stack = [left]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        size = len(seen)
        if min(size, n - size) >= min_size:
            choices.append((weight, -abs(n - 2 * size), -left, -right, seen))
    if not choices:
        raise ValueError("MST has no cut satisfying min_size")
    chosen = max(choices)[-1]
    labels = np.asarray([0 if i in chosen else 1 for i in range(n)], dtype=np.int8)
    return labels


def _reshape_by_scene_seed(
    value: np.ndarray,
    scene_id: np.ndarray,
    seed_id: np.ndarray,
    scenes: np.ndarray,
    seeds: np.ndarray,
) -> np.ndarray:
    lookup = {(int(s), int(k)): i for i, (s, k) in enumerate(zip(scene_id, seed_id))}
    if len(lookup) != len(scene_id):
        raise ValueError("duplicate (scene, seed) row")
    try:
        index = np.asarray([[lookup[(int(s), int(k))] for k in seeds] for s in scenes])
    except KeyError as exc:
        raise ValueError(f"missing scene/seed pair {exc}") from exc
    if index.size != len(scene_id):
        raise ValueError("unexpected rows outside the canonical scene/seed grid")
    return np.asarray(value)[index]


def load_task(slice_root: Path, proxy_root: Path, name: str) -> TaskData:
    source = slice_root / name / "first_inference_hb5_d0.npz"
    if not source.exists():
        raise FileNotFoundError(source)
    manifest = json.loads((slice_root / name / "source_manifest.json").read_text())
    selection = manifest.get("selection", {})
    if (
        manifest.get("alias") != name
        or selection.get("hb_layer") != 5
        or selection.get("denoise_step") != 0
        or manifest.get("candidate_count") != N_SCENES * N_SEEDS
    ):
        raise ValueError(f"{name}: source manifest does not describe the frozen slice")
    archive_sha256 = _sha256(source)
    if archive_sha256 != manifest.get("archive_sha256"):
        raise ValueError(f"{name}: first-inference archive SHA256 mismatch")
    with np.load(source) as data:
        scenes = np.unique(data["init_state_id"]).astype(np.int64)
        seeds = np.unique(data["flow_noise_seed"]).astype(np.int64)
        if len(scenes) != N_SCENES or len(seeds) != N_SEEDS:
            raise ValueError(f"{name}: expected {N_SCENES} scenes x {N_SEEDS} seeds")
        expected_scenes = np.asarray([0, 3, 7, 10, 13, 16, 20, 23, 26, 29, 33, 36, 39, 42, 46, 49])
        if not np.array_equal(scenes, expected_scenes):
            raise ValueError(f"{name}: unexpected scene IDs")
        if not np.array_equal(seeds, np.arange(1000, 1032)):
            raise ValueError(f"{name}: expected flow-noise seeds 1000 through 1031")
        kwargs = dict(
            scene_id=data["init_state_id"], seed_id=data["flow_noise_seed"],
            scenes=scenes, seeds=seeds,
        )
        actions = _reshape_by_scene_seed(data["actions"], **kwargs).astype(np.float32)
        success = _reshape_by_scene_seed(data["success"], **kwargs).astype(bool)
        router = _reshape_by_scene_seed(data["hb_router_probs"], **kwargs)[
            :, :, ACTION_TOKEN_SLICE, :
        ].astype(np.float32)
        hidden = _reshape_by_scene_seed(data["hb_hidden"], **kwargs)[
            :, :, ACTION_TOKEN_SLICE, :
        ].astype(np.float32)
        ids = _reshape_by_scene_seed(data["hb_expert_ids"], **kwargs)[
            :, :, ACTION_TOKEN_SLICE, :
        ].astype(np.int64)
        selected = _reshape_by_scene_seed(data["hb_selected_prob"], **kwargs)[
            :, :, ACTION_TOKEN_SLICE, :
        ].astype(np.float32)
        state = _reshape_by_scene_seed(data["state"], **kwargs)
        sim_state = _reshape_by_scene_seed(data["sim_state"], **kwargs)

    expected_shapes = {
        "actions": (N_SCENES, N_SEEDS, 10, 7),
        "router": (N_SCENES, N_SEEDS, 10, 32),
        "hidden": (N_SCENES, N_SEEDS, 10, 1024),
        "ids": (N_SCENES, N_SEEDS, 10, 4),
        "selected": (N_SCENES, N_SEEDS, 10, 4),
    }
    actual_shapes = {
        "actions": actions.shape, "router": router.shape, "hidden": hidden.shape,
        "ids": ids.shape, "selected": selected.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"{name}: frozen array shape mismatch: {actual_shapes}")

    for key, value in {
        "actions": actions,
        "router": router,
        "hidden": hidden,
        "selected": selected,
        "state": state,
        "sim_state": sim_state,
    }.items():
        _finite(f"{name}/{key}", value)
    if np.any(np.ptp(state, axis=1) != 0) or np.any(np.ptp(sim_state, axis=1) != 0):
        raise ValueError(f"{name}: observation is not exactly fixed inside a pool")
    router_mass = np.maximum(router, 0.0).sum(axis=-1)
    if np.any(router_mass <= 0):
        raise ValueError(f"{name}: non-positive router mass")
    gathered = np.take_along_axis(router, ids, axis=-1)
    selected_max_abs_error = float(np.max(np.abs(gathered - selected)))
    if not np.array_equal(gathered, selected):
        raise ValueError(f"{name}: selected probability alignment failed")
    if (
        np.any(ids < 0) or np.any(ids >= 32)
        or np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0)
    ):
        raise ValueError(f"{name}: invalid or repeated expert IDs in a top-4 site")
    normalized_mass_error = float(np.max(np.abs(
        (np.maximum(router, 0.0) / router_mass[..., None]).sum(axis=-1) - 1.0
    )))

    metadata_path = slice_root / name / "metadata/client/server_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("checkpoint_sha256") != manifest.get("source_checkpoint_sha256"):
        raise ValueError(f"{name}: checkpoint SHA256 differs from source manifest")
    if metadata.get("normalization_stats_loaded") is not True:
        raise ValueError(f"{name}: normalization stats were not recorded as loaded")
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    if action_std.shape != (7,) or np.any(~np.isfinite(action_std)) or np.any(action_std <= 0):
        raise ValueError(f"{name}: invalid action standard deviation")
    recorded_stats_path = Path(metadata["normalization_stats_path"])
    stats_path = resolve_normalization_stats_path(metadata_path, str(recorded_stats_path))
    stats_sha256 = _sha256(stats_path)
    if stats_sha256 != metadata["normalization_stats_sha256"]:
        raise ValueError(f"{name}: normalization stats SHA256 mismatch")
    stats_std = np.asarray(json.loads(stats_path.read_text())["actions"]["std"])
    if not np.array_equal(stats_std, action_std):
        raise ValueError(f"{name}: embedded action std differs from frozen stats file")

    proxy_path = proxy_root / name / "candidate_proxy_values.npz"
    proxies: dict[str, np.ndarray] = {}
    if proxy_path.exists():
        proxy_summary_path = proxy_path.with_name("summary.json")
        proxy_summary = json.loads(proxy_summary_path.read_text())
        if (
            proxy_summary.get("layer") != 5
            or proxy_summary.get("denoise") != 0
            or proxy_summary.get("status") != "retrospective_proxy_not_runtime_causal"
            or proxy_summary.get("checkpoint_sha256") != metadata.get("checkpoint_sha256")
        ):
            raise ValueError(f"{name}: proxy summary does not match the frozen cell/checkpoint")
        with np.load(proxy_path) as proxy:
            if not np.array_equal(proxy["scene_ids"], scenes):
                raise ValueError(f"{name}: proxy scene IDs do not align")
            if not np.array_equal(proxy["seed_ids"], seeds):
                raise ValueError(f"{name}: proxy seed IDs do not align")
            for key in (
                "block_sensitivity", "routed_sensitivity", "input_sensitivity",
                "disagreement", "cancellation", "conflict",
            ):
                proxies[key] = np.asarray(proxy[key], dtype=np.float64)
                if proxies[key].shape != (N_SCENES, N_SEEDS):
                    raise ValueError(f"{name}: proxy/{key} shape mismatch")
                _finite(f"{name}/proxy/{key}", proxies[key])

    success_count = int(success.sum())
    mixed_count = int(np.sum((success.sum(axis=1) > 0) & (success.sum(axis=1) < N_SEEDS)))
    balanced_count = int(np.sum((success.sum(axis=1) >= 4) & (success.sum(axis=1) <= 28)))
    if (
        success_count != EXPECTED_SUCCESSES[name]
        or mixed_count != EXPECTED_MIXED[name]
        or balanced_count != EXPECTED_BALANCED[name]
    ):
        raise ValueError(f"{name}: frozen success/mixed/balanced counts changed")

    validation = {
        "rows": int(actions.shape[0] * actions.shape[1]),
        "scenes": int(len(scenes)),
        "seeds": int(len(seeds)),
        "seed_min": int(seeds.min()),
        "seed_max": int(seeds.max()),
        "exact_observation_within_pool": True,
        "router_mass_before_normalization_min": float(router_mass.min()),
        "router_mass_before_normalization_max": float(router_mass.max()),
        "router_normalized_mass_max_abs_error": normalized_mass_error,
        "selected_probability_max_abs_error": selected_max_abs_error,
        "normalization_stats_sha256": stats_sha256,
        "normalization_stats_source": (
            "recorded_checkpoint_path" if stats_path == recorded_stats_path else "portable_hash_matched_copy"
        ),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "archive_sha256": archive_sha256,
        "successes": success_count,
        "mixed_pools": mixed_count,
        "balanced16_pools": balanced_count,
        "proxy_aligned": bool(proxies),
    }
    return TaskData(
        name=name, scenes=scenes, seeds=seeds, actions=actions, success=success,
        router_probs=router, hidden=hidden, expert_ids=ids,
        selected_prob=selected, action_std=action_std, proxies=proxies,
        validation=validation,
    )


def _task_macro(pool_values: np.ndarray, task_index: np.ndarray) -> float:
    values = np.asarray(pool_values, dtype=np.float64)
    return float(np.mean([values[task_index == task].mean() for task in np.unique(task_index)]))


def _task_effects(pool_values: np.ndarray, task_index: np.ndarray) -> dict[str, float]:
    return {
        TASKS[int(task)]: float(np.mean(pool_values[task_index == task]))
        for task in np.unique(task_index)
    }


def _leave_one_task_out(pool_values: np.ndarray, task_index: np.ndarray) -> dict[str, float]:
    return {
        TASKS[int(task)]: _task_macro(pool_values[task_index != task], task_index[task_index != task])
        for task in np.unique(task_index)
    }


def _success_curve(labels: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    votes = np.cumsum(labels[neighbors], axis=1) / np.arange(1, neighbors.shape[1] + 1)
    return np.asarray([tie_aware_auc(labels, votes[:, k]) - 0.5 for k in range(votes.shape[1])])


def _graph_diagnostics(distance: np.ndarray, neighbors: np.ndarray) -> dict[str, float]:
    n = len(distance)
    directed = np.zeros((n, n), dtype=bool)
    directed[np.arange(n)[:, None], neighbors[:, :4]] = True
    union = directed | directed.T
    mutual = directed & directed.T

    def components(adjacency: np.ndarray) -> tuple[int, int]:
        unseen = set(range(n))
        sizes = []
        while unseen:
            start = unseen.pop()
            seen = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                found = set(np.flatnonzero(adjacency[node])) & unseen
                unseen -= found
                seen |= found
                stack.extend(found)
            sizes.append(len(seen))
        return len(sizes), max(sizes)

    union_components, union_largest = components(union)
    mutual_components, mutual_largest = components(mutual)
    degree = union.sum(axis=1).astype(np.float64)
    inverse_root = np.zeros_like(degree)
    inverse_root[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    laplacian = np.eye(n) - inverse_root[:, None] * union * inverse_root[None, :]
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return {
        "union_components_k4": int(union_components),
        "union_largest_k4": int(union_largest),
        "mutual_components_k4": int(mutual_components),
        "mutual_largest_k4": int(mutual_largest),
        "union_lambda2_k4": float(eigenvalues[1]),
        "route_neighbor_radius_k4": float(np.mean(distance[np.arange(n)[:, None], neighbors[:, :4]])),
    }


def _permutation_action(
    target_ranks: np.ndarray,
    neighbors: np.ndarray,
    task_index: np.ndarray,
    permutations: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    n_draws, n = permutations.shape
    n_pools = len(target_ranks)
    result = np.empty(n_draws, dtype=np.float64)
    pool_index = np.arange(n_pools)[None, :, None, None]
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        pi = permutations[start:stop]
        mapped_neighbors = np.take(pi, neighbors, axis=1)
        values = target_ranks[
            pool_index,
            pi[:, None, :, None],
            mapped_neighbors,
        ]
        cumulative = np.cumsum(values, axis=-1) / np.arange(1, neighbors.shape[-1] + 1)
        pool_auk = (1.0 - 2.0 * cumulative.mean(axis=2)).mean(axis=-1)
        task_means = np.stack(
            [pool_auk[:, task_index == task].mean(axis=1) for task in np.unique(task_index)],
            axis=1,
        )
        result[start:stop] = task_means.mean(axis=1)
    return result


def _permutation_success_curves(
    labels: np.ndarray,
    neighbors: np.ndarray,
    task_index: np.ndarray,
    permutations: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    n_draws = len(permutations)
    n_pools, n = labels.shape
    result = np.empty((n_draws, neighbors.shape[-1]), dtype=np.float64)
    pool_index = np.arange(n_pools)[None, :, None, None]
    n_positive = labels.sum(axis=1)[None, :, None]
    n_negative = n - n_positive
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        pi = permutations[start:stop]
        permuted = np.stack([labels[:, order] for order in pi], axis=0)
        neighbor_labels = permuted[:, pool_index, neighbors[None, :, :, :]].squeeze(1)
        score = np.cumsum(neighbor_labels, axis=-1) / np.arange(1, neighbors.shape[-1] + 1)
        score = np.moveaxis(score, -1, -2)
        ranks = rankdata(score, method="average", axis=-1)
        rank_sum = np.sum(ranks * permuted[:, :, None, :], axis=-1)
        auc = (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
        pool_curves = auc - 0.5
        task_means = np.stack(
            [pool_curves[:, task_index == task].mean(axis=1) for task in np.unique(task_index)],
            axis=1,
        )
        result[start:stop] = task_means.mean(axis=1)
    return result


def _permutation_success(
    labels: np.ndarray,
    neighbors: np.ndarray,
    task_index: np.ndarray,
    permutations: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    return _permutation_success_curves(
        labels, neighbors, task_index, permutations, batch_size
    ).mean(axis=1)


def _permutation_mst_action(
    target_ranks: np.ndarray,
    edge_nodes: np.ndarray,
    task_index: np.ndarray,
    permutations: np.ndarray,
    batch_size: int = 128,
) -> np.ndarray:
    n_draws = len(permutations)
    n_pools = len(target_ranks)
    result = np.empty(n_draws, dtype=np.float64)
    pool_index = np.arange(n_pools)[None, :, None]
    left, right = edge_nodes[:, :, 0], edge_nodes[:, :, 1]
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        pi = permutations[start:stop]
        mapped_left = np.take(pi, left, axis=1)
        mapped_right = np.take(pi, right, axis=1)
        forward = target_ranks[pool_index, mapped_left, mapped_right]
        reverse = target_ranks[pool_index, mapped_right, mapped_left]
        pool_gain = 1.0 - 2.0 * np.mean(np.concatenate([forward, reverse], axis=2), axis=2)
        task_means = np.stack(
            [pool_gain[:, task_index == task].mean(axis=1) for task in np.unique(task_index)],
            axis=1,
        )
        result[start:stop] = task_means.mean(axis=1)
    return result


def _permutation_mst_success(
    labels: np.ndarray,
    edge_nodes: np.ndarray,
    expected_same: np.ndarray,
    task_index: np.ndarray,
    permutations: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    n_draws = len(permutations)
    n_pools = len(labels)
    result = np.empty(n_draws, dtype=np.float64)
    left, right = edge_nodes[:, :, 0], edge_nodes[:, :, 1]
    for start in range(0, n_draws, batch_size):
        stop = min(start + batch_size, n_draws)
        pi = permutations[start:stop]
        permuted = np.stack([labels[:, order] for order in pi], axis=0)
        batch = len(pi)
        left_label = np.take_along_axis(
            permuted, np.broadcast_to(left[None], (batch,) + left.shape), axis=2
        )
        right_label = np.take_along_axis(
            permuted, np.broadcast_to(right[None], (batch,) + right.shape), axis=2
        )
        pool_gain = np.mean(left_label == right_label, axis=2) - expected_same[None, :]
        task_means = np.stack(
            [pool_gain[:, task_index == task].mean(axis=1) for task in np.unique(task_index)],
            axis=1,
        )
        result[start:stop] = task_means.mean(axis=1)
    return result


def _permutation_p(observed: float, null: np.ndarray) -> float:
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def _studentized_max_t(
    observed: dict[str, float], nulls: dict[str, np.ndarray]
) -> tuple[dict[str, float], dict[str, float]]:
    names = list(observed)
    centers = {name: float(np.mean(nulls[name])) for name in names}
    scales = {name: float(np.std(nulls[name], ddof=1)) for name in names}
    if any(scales[name] <= 0 for name in names):
        raise ValueError("degenerate permutation null")
    null_z = np.column_stack(
        [(nulls[name] - centers[name]) / scales[name] for name in names]
    )
    max_null = null_z.max(axis=1)
    observed_z = {name: (observed[name] - centers[name]) / scales[name] for name in names}
    adjusted = {
        name: float((1 + np.count_nonzero(max_null >= observed_z[name])) / (len(max_null) + 1))
        for name in names
    }
    return observed_z, adjusted


def _rank_z(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average")
    ranks -= ranks.mean()
    norm = np.linalg.norm(ranks)
    return ranks / norm if norm > 0 else np.zeros_like(ranks)


def _pool_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std()
    return (values - values.mean()) / scale if scale > 0 else np.zeros_like(values)


def _exploratory_correlations(
    pools: list[dict[str, object]], task_index: np.ndarray
) -> dict[str, object]:
    predictor_names = (
        "route_radius", "hidden_radius", "disagreement", "cancellation",
        "conflict", "dcq", "route_plus_dcq",
    )
    outcome_names = (
        "block_sensitivity", "routed_sensitivity", "input_sensitivity",
        "local_action_distortion",
    )
    per_pool = np.full((len(pools), len(predictor_names), len(outcome_names)), np.nan)
    smooth_names = (
        "disagreement", "cancellation", "conflict", "dcq",
        "block_sensitivity", "routed_sensitivity", "input_sensitivity",
    )
    smooth_route = np.full((len(pools), len(smooth_names)), np.nan)
    smooth_hidden = np.full_like(smooth_route, np.nan)
    for pool_index, pool in enumerate(pools):
        route = pool["distances"]["route"]
        hidden = pool["distances"]["hidden"]
        action = pool["distances"]["action"]
        neighbors = pool["neighbors"]["route"][:, :4]
        rows = np.arange(N_SEEDS)[:, None]
        route_radius = np.mean(route[rows, neighbors], axis=1)
        hidden_radius = np.mean(hidden[rows, neighbors], axis=1)
        local_action = np.mean(action[rows, neighbors], axis=1) / np.maximum(
            np.sum(action, axis=1) / (N_SEEDS - 1), 1e-12
        )
        proxy = pool["proxies"]
        if not proxy:
            continue
        dcq = np.mean(
            np.stack([_pool_z(proxy[key]) for key in ("disagreement", "cancellation", "conflict")]),
            axis=0,
        )
        combined = 0.5 * (_pool_z(route_radius) + _pool_z(dcq))
        predictors = {
            "route_radius": route_radius,
            "hidden_radius": hidden_radius,
            "disagreement": proxy["disagreement"],
            "cancellation": proxy["cancellation"],
            "conflict": proxy["conflict"],
            "dcq": dcq,
            "route_plus_dcq": combined,
        }
        outcomes = {
            "block_sensitivity": proxy["block_sensitivity"],
            "routed_sensitivity": proxy["routed_sensitivity"],
            "input_sensitivity": proxy["input_sensitivity"],
            "local_action_distortion": local_action,
        }
        smooth_values = {**{key: proxy[key] for key in smooth_names if key != "dcq"}, "dcq": dcq}
        for scalar_index, scalar_name in enumerate(smooth_names):
            scalar = smooth_values[scalar_name]
            scalar_distance = np.abs(scalar[:, None] - scalar[None, :])
            smooth_route[pool_index, scalar_index] = auk8(route, scalar_distance)
            smooth_hidden[pool_index, scalar_index] = auk8(hidden, scalar_distance)
        for i, pred_name in enumerate(predictor_names):
            for j, outcome_name in enumerate(outcome_names):
                per_pool[pool_index, i, j] = float(
                    np.dot(_rank_z(predictors[pred_name]), _rank_z(outcomes[outcome_name]))
                )
    task_macro = np.stack(
        [np.nanmean(per_pool[task_index == task], axis=0) for task in np.unique(task_index)]
    )
    macro = np.nanmean(task_macro, axis=0)
    smooth_route_task = np.stack([
        np.nanmean(smooth_route[task_index == task], axis=0) for task in np.unique(task_index)
    ])
    smooth_hidden_task = np.stack([
        np.nanmean(smooth_hidden[task_index == task], axis=0) for task in np.unique(task_index)
    ])
    pred = {name: index for index, name in enumerate(predictor_names)}
    outcome = {name: index for index, name in enumerate(outcome_names)}
    local_column = macro[:, outcome["local_action_distortion"]]
    local_best = int(np.argmax(np.abs(local_column)))
    return {
        "note": (
            "exploratory effect sizes only; no p-values. D/C/Q and sensitivity "
            "share reconstructed expert-output norms, so their correlations are not independent validation"
        ),
        "predictors": list(predictor_names),
        "outcomes": list(outcome_names),
        "macro_mean_pool_spearman": macro.tolist(),
        "per_task_mean_pool_spearman": {
            TASKS[int(task)]: task_macro[pos].tolist()
            for pos, task in enumerate(np.unique(task_index))
        },
        "key_comparisons": {
            "conflict_to_block_sensitivity": float(macro[pred["conflict"], outcome["block_sensitivity"]]),
            "dcq_to_routed_sensitivity": float(macro[pred["dcq"], outcome["routed_sensitivity"]]),
            "route_plus_dcq_to_routed_sensitivity": float(macro[pred["route_plus_dcq"], outcome["routed_sensitivity"]]),
            "route_plus_dcq_minus_dcq_for_routed_sensitivity": float(
                macro[pred["route_plus_dcq"], outcome["routed_sensitivity"]]
                - macro[pred["dcq"], outcome["routed_sensitivity"]]
            ),
            "route_radius_to_input_sensitivity": float(macro[pred["route_radius"], outcome["input_sensitivity"]]),
            "largest_abs_local_action_distortion_predictor": predictor_names[local_best],
            "largest_abs_local_action_distortion_rho": float(local_column[local_best]),
        },
        "graph_smoothness_auk8": {
            "scalars": list(smooth_names),
            "route_macro": np.nanmean(smooth_route_task, axis=0).tolist(),
            "hidden_macro": np.nanmean(smooth_hidden_task, axis=0).tolist(),
            "route_minus_hidden": (
                np.nanmean(smooth_route_task, axis=0) - np.nanmean(smooth_hidden_task, axis=0)
            ).tolist(),
            "route_per_task": {
                TASKS[int(task)]: smooth_route_task[pos].tolist()
                for pos, task in enumerate(np.unique(task_index))
            },
            "hidden_per_task": {
                TASKS[int(task)]: smooth_hidden_task[pos].tolist()
                for pos, task in enumerate(np.unique(task_index))
            },
        },
    }


def _classical_mds(distance: np.ndarray) -> np.ndarray:
    n = len(distance)
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ np.square(distance) @ centering
    values, vectors = np.linalg.eigh(gram)
    take = np.argsort(values)[-2:][::-1]
    return vectors[:, take] * np.sqrt(np.maximum(values[take], 0.0))


def _make_figure(
    out: Path,
    pools: list[dict[str, object]],
    source_names: list[str],
    source_curves: dict[str, np.ndarray],
    action_observed: float,
    action_null: np.ndarray,
    success_curve: np.ndarray,
    success_null_curve: np.ndarray,
    success_observed: float,
    success_null: np.ndarray,
    exploratory: dict[str, object],
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), constrained_layout=True)

    example = next(pool for pool in pools if pool["task"] == "long-t08" and pool["scene"] == 0)
    xy = _classical_mds(example["distances"]["route"])
    neighbors = example["neighbors"]["route"][:, :4]
    union_edges = {tuple(sorted((i, int(j)))) for i in range(N_SEEDS) for j in neighbors[i]}
    ax = axes[0, 0]
    for i, j in union_edges:
        ax.plot(xy[[i, j], 0], xy[[i, j], 1], color="#c4c8cc", lw=0.55, zorder=1)
    colors = np.where(example["success"], "#167d55", "#c4493d")
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=28, edgecolor="white", linewidth=0.4, zorder=2)
    ax.set_title("Fixed example: long-t08 / state 0 / route k=4")
    ax.set_xlabel("route MDS 1")
    ax.set_ylabel("route MDS 2")

    ax = axes[0, 1]
    palette = {
        "route": "#1769aa", "hidden": "#30343b", "noise7": "#d17a00",
        "noise24": "#a44a9f", "top4_weighted": "#2a9d8f", "top4_jaccard": "#8b6f47",
    }
    for name in source_names:
        ax.plot(np.arange(1, MAX_K + 1), source_curves[name], marker="o", ms=3,
                lw=1.4, label=name, color=palette.get(name))
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Action-neighbor gain (task macro)")
    ax.set_xlabel("k")
    ax.set_ylabel("G_action(k)")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[0, 2]
    ax.hist(action_null, bins=45, color="#b7c7d6", edgecolor="none")
    ax.axvline(action_observed, color="#1769aa", lw=2)
    ax.set_title("Common-seed null: action AUK8")
    ax.set_xlabel("T_action")

    ax = axes[1, 0]
    ax.plot(np.arange(1, MAX_K + 1), success_curve, marker="o", color="#167d55", label="observed")
    ax.plot(np.arange(1, MAX_K + 1), success_null_curve, ls="--", color="#777", label="null mean")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("Success-neighbor gain (mixed40 macro)")
    ax.set_xlabel("k")
    ax.set_ylabel("AUC - 0.5")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    ax.hist(success_null, bins=45, color="#bfd6cc", edgecolor="none")
    ax.axvline(success_observed, color="#167d55", lw=2)
    ax.set_title("Common-seed null: success AUK8")
    ax.set_xlabel("T_success")

    ax = axes[1, 2]
    matrix = np.asarray(exploratory["macro_mean_pool_spearman"])
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(len(exploratory["outcomes"])), exploratory["outcomes"], rotation=35, ha="right")
    ax.set_yticks(range(len(exploratory["predictors"])), exploratory["predictors"])
    ax.set_title("Exploratory within-pool Spearman")
    fig.colorbar(image, ax=ax, shrink=0.8)

    fig.savefig(out / "overview.png", dpi=180)
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_report(out: Path, summary: dict[str, object]) -> None:
    action = summary["confirmatory"]["action"]
    success = summary["confirmatory"]["success"]
    baselines = summary["baselines"]
    mst = summary["secondary_mst"]
    exploratory = summary["exploratory_expert_output"]
    key = exploratory["key_comparisons"]
    smooth = exploratory["graph_smoothness_auk8"]
    smooth_index = {name: index for index, name in enumerate(smooth["scalars"])}
    action_ok = action["p_maxT"] < 0.05 and action["excess_over_null_mean"] > 0
    success_ok = success["p_maxT"] < 0.05 and success["excess_over_null_mean"] > 0
    if action_ok:
        action_text = "路由邻接与最终完整动作块的局部几何存在经校正的正联系"
    else:
        action_text = "没有检出路由邻接与最终动作块几何的稳定正联系"
    if success_ok:
        success_text = "成功标签在路由图上存在经校正的局部聚集"
    else:
        success_text = "没有检出成功标签在路由图上的稳定局部聚集"

    hidden = baselines["action"]["hidden"]["observed"]
    route = action["observed"]
    if action_ok and route > hidden:
        boundary = "本数据上 full-route 效应还高于完整 HB hidden 的同口径基线，但该差值未列入确认性检验，不能表述成额外信息。"
    elif action_ok:
        boundary = "完整 hidden 基线至少同样强，因此路由最多是结构化、较便宜的监测接口，不是 hidden 之外的新信息。"
    else:
        boundary = "路由图主效应未通过，不能据此建立候选裁剪器。"

    text = f"""# MoE 候选连通图结果

## 结论

- 动作端点：{action_text}。`T_action={action['observed']:.5f}`，未校正
  `p={action['p_unadjusted']:.5g}`，两端点 maxT 校正后
  `p={action['p_maxT']:.5g}`；相对共同置换零均值的超额为
  `{action['excess_over_null_mean']:.5f}`。
- 成败端点：{success_text}。`T_success={success['observed']:.5f}`，未校正
  `p={success['p_unadjusted']:.5g}`，maxT 校正后
  `p={success['p_maxT']:.5g}`；相对固定类别数零分布的超额为
  `{success['excess_over_null_mean']:.5f}`。
- {boundary}

这两个问题必须分开：动作图有效，只能支持“相似计算路由对应相似动作候选”；只有成败图也有效，才可能进一步讨论好/坏盆地。即使二者都有效，这仍是关联证据，不是换专家的因果证据。

`T_success` 的原始值为负不等于显著“反盆地”：固定一个池里的成功数、并用不含自己的邻居投票时，随机标签零均值本来就是 `{success['null_mean']:.5f}`，不是 0。因此判断依据是共同置换及其超额，而不是单看 `AUC-0.5` 的符号。

## 数据与检验

五个任务各有 16 个严格相同 observation 的状态池，每池共用 32 个 flow-noise seed，共 80 张候选图、2,560 个候选。动作端点使用全部 80 池；成败端点仅使用同时含成功和失败的 40 池。主图由 HB5/d0 十个 action token 的完整 32-way router probability 的 Hellinger 距离构成，未使用 `expert_ids[...,0]`。

统计量固定为 `k=1..8` 的 AUK8。零分布使用 {summary['permutation']['draws']:,} 次共同 seed-column 置换：每次在所有任务和状态上使用同一列置换。动作与成败两个主端点再做 studentized maxT 家族校正。

## 每任务效应

| task | action AUK8 | success AUK8 | mixed pools |
|---|---:|---:|---:|
"""
    for task in TASKS:
        action_value = action["per_task"].get(task, float("nan"))
        success_value = success["per_task"].get(task, float("nan"))
        mixed = summary["validation"][task]["mixed_pools"]
        success_cell = "NA" if math.isnan(success_value) else f"{success_value:.5f}"
        text += f"| {task} | {action_value:.5f} | {success_cell} | {mixed} |\n"

    text += f"""

## 基线边界

下面所有数值都使用完全相同的 AUK8 定义。它们是效应量比较，不是额外的显著性家族。

| source graph | action AUK8 | route minus source |
|---|---:|---:|
"""
    for name, item in baselines["action"].items():
        text += f"| {name} | {item['observed']:.5f} | {route - item['observed']:.5f} |\n"

    text += f"""

## 连通图与 expert-output 探索

`per_pool.csv` 包含每张图的 k=4 union/mutual component、最大连通分量、归一化 Laplacian `lambda2`、MST 和 balanced-cut 描述量。连通分量本身只描述图的形态，不是成败检验。

MST 是另一个不需要选择阈值的次要口径：动作 `p={mst['action']['p_unadjusted']:.5g}`，mixed40 成败 `p={mst['success_mixed40']['p_unadjusted']:.5g}`。它不能替换失败的 AUK8 主端点。balanced16 的 success AUK8 为 `{success['balanced16_descriptive']['observed']:.5f}`，仅作固定敏感性描述。

`disagreement/cancellation/conflict` 来自离线重建的真实 expert-output proxy，而不是 router ID。池内相关显示 `DCQ -> routed sensitivity` 为 `{key['dcq_to_routed_sensitivity']:.3f}`，但固定的 `route + DCQ` 组合只有 `{key['route_plus_dcq_to_routed_sensitivity']:.3f}`，增量 `{key['route_plus_dcq_minus_dcq_for_routed_sensitivity']:.3f}`；连通图没有改善这个张力信号。与局部 action distortion 的最大绝对相关也只有 `{key['largest_abs_local_action_distortion_rho']:.3f}`。

直接把 expert 标量当作图上的 target 后，route graph 的 smoothness AUK8 为：D `{smooth['route_macro'][smooth_index['disagreement']]:.3f}`、C `{smooth['route_macro'][smooth_index['cancellation']]:.3f}`、Q `{smooth['route_macro'][smooth_index['conflict']]:.3f}`；对应 hidden graph 是 `{smooth['hidden_macro'][smooth_index['disagreement']]:.3f}`、`{smooth['hidden_macro'][smooth_index['cancellation']]:.3f}`、`{smooth['hidden_macro'][smooth_index['conflict']]:.3f}`。这说明 full router 能组织动作候选，但并没有同样清楚地组织 expert-output 张力。

这些都是探索性效应量，没有 p 值，而且 D/C/Q 与 sensitivity 共用重建 expert 输出的范数，不能视作独立验证。下一轮若继续，应在独立 observation 上预注册一个固定张力分数，并直接记录 runtime expert vectors 与 intervention sensitivity。

## 限制

结果只覆盖 HB5/d0、每个 episode 的首次 inference、五个任务和三个 checkpoint suite。图与动作/结果的关系是观察性几何，不是 expert 替换的因果效应；当前数据也没有完整 runtime expert vector，不能构造真正的 signed expert-conflict graph。

## 旧结果撤回

所有把 `hb_expert_ids[...,0]` 当作 top-1 的旧 adjacency-MI 或候选图数字均无效，因为存储的 top-k 是未排序集合。这里的主结果只使用完整 router probability；top-4 控制也是顺序不变的。

## 文件

- `summary.json`：所有汇总效应、p 值、验证和解释边界；
- `per_pool.csv`：80 个状态池的逐图结果；
- `permutation_nulls.npz`：共同置换索引与两个主零分布；
- `overview.png`：固定示例图、完整 k 曲线、零分布与探索性相关矩阵。
"""
    (out / "REPORT.md").write_text(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slice-root",
        default="/home/jovyan/work/himoe-vla-moe-results-bundle-2026-08-23-max500mb/raw/hub-first-inference",
    )
    parser.add_argument(
        "--proxy-root", default="analysis/expert-activation-hidden-matched"
    )
    parser.add_argument("--out", default="analysis/moe-connectivity-graph")
    parser.add_argument("--permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args()

    slice_root = Path(args.slice_root).resolve()
    proxy_root = Path(args.proxy_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    tasks = [load_task(slice_root, proxy_root, name) for name in TASKS]
    canonical_seeds = tasks[0].seeds
    if any(not np.array_equal(task.seeds, canonical_seeds) for task in tasks[1:]):
        raise ValueError("tasks do not share the same seed columns")

    noise24 = np.stack([
        np.random.default_rng(int(seed)).standard_normal((10, 24)).astype(np.float32)
        for seed in canonical_seeds
    ])
    noise_distances = {
        "noise7": pairwise_rms(noise24[:, :, :7]),
        "noise24": pairwise_rms(noise24),
    }

    pools: list[dict[str, object]] = []
    task_index = []
    for task_number, task in enumerate(tasks):
        for scene_number, scene in enumerate(task.scenes):
            route = hellinger_distance(task.router_probs[scene_number])
            hidden = pairwise_rms(task.hidden[scene_number])
            action = pairwise_rms(task.actions[scene_number] / task.action_std[None, None, :])
            selected_tensor = selected_probability_tensor(
                task.expert_ids[scene_number], task.selected_prob[scene_number]
            )
            distances = {
                "route": route,
                "hidden": hidden,
                "noise7": noise_distances["noise7"],
                "noise24": noise_distances["noise24"],
                "top4_weighted": hellinger_distance(selected_tensor),
                "top4_jaccard": topk_jaccard_distance(task.expert_ids[scene_number]),
                "action": action,
            }
            neighbors = {
                key: _neighbor_order(value, MAX_K)
                for key, value in distances.items() if key != "action"
            }
            proxy_values = {
                key: value[scene_number] for key, value in task.proxies.items()
            }
            pools.append({
                "task": task.name,
                "task_number": task_number,
                "scene": int(scene),
                "seeds": task.seeds,
                "success": task.success[scene_number],
                "distances": distances,
                "neighbors": neighbors,
                "proxies": proxy_values,
            })
            task_index.append(task_number)
    task_index = np.asarray(task_index, dtype=np.int64)

    source_names = ["route", "hidden", "noise7", "noise24", "top4_weighted", "top4_jaccard"]
    pool_action_curves = {
        source: np.stack([
            auk_curve(pool["distances"][source], pool["distances"]["action"], MAX_K)
            for pool in pools
        ]) for source in source_names
    }
    pool_action_auk = {source: curves.mean(axis=1) for source, curves in pool_action_curves.items()}
    source_curves = {
        source: np.mean([
            curves[task_index == task].mean(axis=0) for task in np.unique(task_index)
        ], axis=0)
        for source, curves in pool_action_curves.items()
    }
    source_observed = {
        source: _task_macro(values, task_index) for source, values in pool_action_auk.items()
    }

    success_all = np.stack([pool["success"] for pool in pools])
    success_counts = success_all.sum(axis=1)
    mixed = (success_counts > 0) & (success_counts < N_SEEDS)
    balanced = (success_counts >= 4) & (success_counts <= 28)
    if int(mixed.sum()) != 40 or int(balanced.sum()) != 16:
        raise ValueError("frozen mixed40/balanced16 pool counts changed")
    mixed_task_index = task_index[mixed]
    mixed_neighbors = np.stack([pool["neighbors"]["route"] for pool in pools])[mixed]
    mixed_labels = success_all[mixed]
    pool_success_curves = np.stack([
        _success_curve(labels, neighbors)
        for labels, neighbors in zip(mixed_labels, mixed_neighbors)
    ])
    pool_success_auk = pool_success_curves.mean(axis=1)
    success_curve_macro = np.mean([
        pool_success_curves[mixed_task_index == task].mean(axis=0)
        for task in np.unique(mixed_task_index)
    ], axis=0)
    success_observed = _task_macro(pool_success_auk, mixed_task_index)
    balanced_neighbors = np.stack([pool["neighbors"]["route"] for pool in pools])[balanced]
    balanced_labels = success_all[balanced]
    balanced_task_index = task_index[balanced]
    balanced_success_curves = np.stack([
        _success_curve(labels, neighbors)
        for labels, neighbors in zip(balanced_labels, balanced_neighbors)
    ])
    balanced_success_auk = balanced_success_curves.mean(axis=1)

    rng = np.random.default_rng(args.seed)
    permutations = np.stack([rng.permutation(N_SEEDS) for _ in range(args.permutations)])
    target_ranks = np.stack([_target_rank_matrix(pool["distances"]["action"]) for pool in pools])
    route_neighbors = np.stack([pool["neighbors"]["route"] for pool in pools])
    action_null = _permutation_action(
        target_ranks, route_neighbors, task_index, permutations
    )
    success_null_curves = _permutation_success_curves(
        mixed_labels, mixed_neighbors, mixed_task_index, permutations
    )
    success_null = success_null_curves.mean(axis=1)
    primary_observed = {"action": source_observed["route"], "success": success_observed}
    primary_null = {"action": action_null, "success": success_null}
    observed_z, adjusted_p = _studentized_max_t(primary_observed, primary_null)

    # Per-pool MST and graph morphology are secondary. Balanced cuts are
    # retained as descriptive basin diagnostics.
    per_pool_rows = []
    all_mst_edges = []
    mst_expected_same = []
    for pool_number, pool in enumerate(pools):
        route = pool["distances"]["route"]
        action_ranks = _target_rank_matrix(pool["distances"]["action"])
        edges = _mst_edges(route)
        all_mst_edges.append(np.asarray([(i, j) for i, j, _ in edges], dtype=np.int64))
        directed_edge_ranks = [value for i, j, _ in edges for value in (action_ranks[i, j], action_ranks[j, i])]
        mst_action = 1.0 - 2.0 * float(np.mean(directed_edge_ranks))
        labels = pool["success"].astype(bool)
        expected_same = (
            labels.sum() * (labels.sum() - 1)
            + (~labels).sum() * ((~labels).sum() - 1)
        ) / (N_SEEDS * (N_SEEDS - 1))
        mst_expected_same.append(float(expected_same))
        mst_success = float(np.mean([labels[i] == labels[j] for i, j, _ in edges]) - expected_same)
        try:
            cut = balanced_mst_cut(route, min_size=4)
            between = pool["distances"]["action"][cut[:, None] != cut[None, :]]
            within_mask = (cut[:, None] == cut[None, :]) & ~np.eye(N_SEEDS, dtype=bool)
            within = pool["distances"]["action"][within_mask]
            cut_action_ratio = float(between.mean() / within.mean())
            cut_success_gap = float(abs(labels[cut == 0].mean() - labels[cut == 1].mean()))
            cut_min_size = int(min(np.sum(cut == 0), np.sum(cut == 1)))
        except ValueError:
            cut_action_ratio = float("nan")
            cut_success_gap = float("nan")
            cut_min_size = 0
        row = {
            "task": pool["task"], "scene": pool["scene"],
            "successes": int(labels.sum()), "mixed": bool(mixed[pool_number]),
            "balanced16": bool(balanced[pool_number]),
            **{f"action_auk8_{name}": float(pool_action_auk[name][pool_number]) for name in source_names},
            "success_auk8_route": float(pool_success_auk[np.flatnonzero(mixed == True).tolist().index(pool_number)]) if mixed[pool_number] else float("nan"),
            "mst_action_gain": mst_action, "mst_success_gain": mst_success,
            "balanced_cut_action_between_within_ratio": cut_action_ratio,
            "balanced_cut_success_gap": cut_success_gap,
            "balanced_cut_min_size": cut_min_size,
            **_graph_diagnostics(route, pool["neighbors"]["route"]),
        }
        per_pool_rows.append(row)

    all_mst_edges = np.stack(all_mst_edges)
    mst_expected_same = np.asarray(mst_expected_same)
    mst_action_values = np.asarray([row["mst_action_gain"] for row in per_pool_rows])
    mst_success_values = np.asarray([row["mst_success_gain"] for row in per_pool_rows])
    mst_action_observed = _task_macro(mst_action_values, task_index)
    mst_success_observed = _task_macro(mst_success_values[mixed], mixed_task_index)
    mst_action_null = _permutation_mst_action(
        target_ranks, all_mst_edges, task_index, permutations
    )
    mst_success_null = _permutation_mst_success(
        success_all[mixed], all_mst_edges[mixed], mst_expected_same[mixed],
        mixed_task_index, permutations,
    )

    exploratory = _exploratory_correlations(pools, task_index)
    success_per_task = _task_effects(pool_success_auk, mixed_task_index)
    success_loo = _leave_one_task_out(pool_success_auk, mixed_task_index)
    summary = {
        "analysis": "same-state K=32 MoE candidate-connectivity graph",
        "design": str((Path(__file__).parent / "MOE_CONNECTIVITY_GRAPH_DESIGN.md").resolve()),
        "data": {
            "slice_root": str(slice_root), "proxy_root": str(proxy_root),
            "tasks": list(TASKS), "pools": len(pools), "candidates": len(pools) * N_SEEDS,
            "mixed_pools": int(mixed.sum()), "balanced16_pools": int(balanced.sum()),
            "cell": "HB5/d0 first inference", "tokens": "action positions 1:11",
        },
        "validation": {task.name: task.validation for task in tasks},
        "permutation": {
            "draws": int(args.permutations), "seed": int(args.seed),
            "scheme": "one common 32-seed column permutation per draw across all 80 pools",
        },
        "confirmatory": {
            "action": {
                "observed": source_observed["route"],
                "excess_over_null_mean": source_observed["route"] - float(action_null.mean()),
                "curve_k1_to_k8": source_curves["route"].tolist(),
                "per_task": _task_effects(pool_action_auk["route"], task_index),
                "leave_one_task_out": _leave_one_task_out(pool_action_auk["route"], task_index),
                "null_mean": float(action_null.mean()), "null_sd": float(action_null.std(ddof=1)),
                "z_permutation": observed_z["action"],
                "p_unadjusted": _permutation_p(source_observed["route"], action_null),
                "p_maxT": adjusted_p["action"],
            },
            "success": {
                "observed": success_observed,
                "excess_over_null_mean": success_observed - float(success_null.mean()),
                "curve_k1_to_k8": success_curve_macro.tolist(),
                "null_curve_mean_k1_to_k8": success_null_curves.mean(axis=0).tolist(),
                "excess_curve_k1_to_k8": (
                    success_curve_macro - success_null_curves.mean(axis=0)
                ).tolist(),
                "per_task": success_per_task,
                "leave_one_task_out": success_loo,
                "null_mean": float(success_null.mean()), "null_sd": float(success_null.std(ddof=1)),
                "z_permutation": observed_z["success"],
                "p_unadjusted": _permutation_p(success_observed, success_null),
                "p_maxT": adjusted_p["success"],
                "balanced16_descriptive": {
                    "pools": int(balanced.sum()),
                    "observed": _task_macro(balanced_success_auk, balanced_task_index),
                    "curve_k1_to_k8": np.mean([
                        balanced_success_curves[balanced_task_index == task].mean(axis=0)
                        for task in np.unique(balanced_task_index)
                    ], axis=0).tolist(),
                    "per_task": _task_effects(balanced_success_auk, balanced_task_index),
                },
            },
        },
        "baselines": {
            "action": {
                source: {
                    "observed": source_observed[source],
                    "route_minus_source": source_observed["route"] - source_observed[source],
                    "curve_k1_to_k8": source_curves[source].tolist(),
                    "per_task": _task_effects(pool_action_auk[source], task_index),
                }
                for source in source_names if source != "route"
            }
        },
        "graph_morphology": {
            key: float(np.nanmean([row[key] for row in per_pool_rows]))
            for key in (
                "union_components_k4", "union_largest_k4", "mutual_components_k4",
                "mutual_largest_k4", "union_lambda2_k4", "route_neighbor_radius_k4",
                "mst_action_gain", "mst_success_gain",
                "balanced_cut_action_between_within_ratio", "balanced_cut_success_gap",
            )
        },
        "secondary_mst": {
            "action": {
                "observed": mst_action_observed,
                "null_mean": float(mst_action_null.mean()),
                "excess_over_null_mean": mst_action_observed - float(mst_action_null.mean()),
                "p_unadjusted": _permutation_p(mst_action_observed, mst_action_null),
                "per_task": _task_effects(mst_action_values, task_index),
            },
            "success_mixed40": {
                "observed": mst_success_observed,
                "null_mean": float(mst_success_null.mean()),
                "excess_over_null_mean": mst_success_observed - float(mst_success_null.mean()),
                "p_unadjusted": _permutation_p(mst_success_observed, mst_success_null),
                "per_task": _task_effects(mst_success_values[mixed], mixed_task_index),
            },
            "note": "secondary; p-values are not part of the two-endpoint maxT family",
        },
        "exploratory_expert_output": exploratory,
        "withdrawal": "all prior ids[...,0]-as-top1 adjacency results are invalid; top-k storage is unsorted",
    }
    summary = _jsonable(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    with (out / "per_pool.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_pool_rows[0]))
        writer.writeheader()
        writer.writerows(per_pool_rows)
    np.savez_compressed(
        out / "permutation_nulls.npz", permutations=permutations.astype(np.uint8),
        action_null=action_null, success_null=success_null,
        success_null_curves=success_null_curves,
        mst_action_null=mst_action_null, mst_success_null=mst_success_null,
    )
    _make_figure(
        out, pools, source_names, source_curves,
        source_observed["route"], action_null,
        success_curve_macro, success_null_curves.mean(axis=0),
        success_observed, success_null, exploratory,
    )
    _write_report(out, summary)

    print(json.dumps({
        "out": str(out),
        "action": summary["confirmatory"]["action"],
        "success": summary["confirmatory"]["success"],
        "baselines": {name: item["observed"] for name, item in summary["baselines"]["action"].items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
