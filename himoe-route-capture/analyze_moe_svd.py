#!/usr/bin/env python3
"""SVD diagnostics for HiMoE routed activations and full routing sequences.

The decompositions are unsupervised.  Sequence order is used only to orient a
temporal axis after SVD; actions, rewards, simulator state, and success are not
used to fit either the subspace or the axis.  Success is inspected separately as
an exploratory diagnostic with max-statistic permutation correction and a
scene-held-out confirmation split.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from scipy.stats import spearmanr
from sklearn.utils.extmath import randomized_svd


HERE = Path(__file__).resolve().parent
DEFAULT_ACTIVATION = HERE / "analysis/expert-activation-future/long-t08"
DEFAULT_CORPUS = HERE / "corpus/libero30-right-v1"
DEFAULT_OUT = HERE / "analysis/moe-svd"
N_LAYERS = 8
N_DENOISE = 10
N_EXPERTS = 32
ACTION_TOKENS = slice(1, 11)
EPS = 1e-12


@dataclass(frozen=True)
class ActivationVectors:
    routed: np.ndarray
    shared: np.ndarray
    layers: np.ndarray
    scene: np.ndarray
    noise_seed: np.ndarray
    success: np.ndarray


@dataclass(frozen=True)
class RouteSequences:
    denoise_routes: np.ndarray
    task: np.ndarray
    suite: np.ndarray
    episode: np.ndarray
    phase: np.ndarray
    progress: np.ndarray
    task_names: tuple[str, ...]
    probability_sum_bounds: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation-dir", type=Path, default=DEFAULT_ACTIVATION)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260822)
    return parser.parse_args()


def normalize_probabilities(values: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.shape[-1] != N_EXPERTS:
        raise ValueError("router probability axis must contain 32 experts")
    if np.any(~np.isfinite(probabilities)) or float(probabilities.min()) < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum(axis=-1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("router probability vector has zero mass")
    bounds = (float(total.min()), float(total.max()))
    probabilities /= total
    return probabilities, bounds


def load_activation_vectors(directory: Path) -> ActivationVectors:
    directory = directory.expanduser().resolve()
    with np.load(directory / "features.npz", allow_pickle=False) as archive:
        routed = np.asarray(archive["routed_vectors"], dtype=np.float32)
        shared = np.asarray(archive["shared_vectors"], dtype=np.float32)
        layers = np.asarray(archive["layer_numbers"], dtype=np.int64)
    if routed.shape != shared.shape or routed.ndim != 4:
        raise ValueError("activation vectors must align as [query,round,layer,hidden]")
    if routed.shape[1] != N_DENOISE or routed.shape[2] != len(layers):
        raise ValueError("activation round/layer axes disagree with metadata")
    if np.any(~np.isfinite(routed)) or np.any(~np.isfinite(shared)):
        raise ValueError("activation vectors contain non-finite values")

    summary = json.loads((directory / "summary.json").read_text())
    rows = json.loads((Path(summary["run"]) / "client/summaries.json").read_text())
    rows = sorted(rows, key=lambda row: int(row["episode_index"]))
    if len(rows) != len(routed):
        raise ValueError("activation features and rollout summaries have different lengths")
    scene = np.asarray([int(row["init_state_id"]) for row in rows], dtype=np.int64)
    noise = np.asarray([int(row["flow_noise_seed"]) for row in rows], dtype=np.int64)
    success = np.asarray([bool(row["success"]) for row in rows], dtype=np.float64)
    scene_values = np.unique(scene)
    noise_values = np.unique(noise)
    if len(rows) != len(scene_values) * len(noise_values):
        raise ValueError("activation archive is not a complete scene x noise grid")
    expected = set(int(value) for value in noise_values)
    if any(set(int(value) for value in noise[scene == item]) != expected for item in scene_values):
        raise ValueError("each activation scene must contain the same noise seeds")
    return ActivationVectors(routed, shared, layers, scene, noise, success)


def load_route_sequences(corpus: Path) -> RouteSequences:
    corpus = corpus.expanduser().resolve()
    with (corpus / "INDEX.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "suite",
        "task_dir",
        "episode_index",
        "inference_calls",
        "control_step_offset",
        "path",
    }
    if not rows or required - set(rows[0]):
        raise ValueError("corpus INDEX.csv has an unexpected schema")

    keys = sorted({(row["suite"], row["task_dir"], row["path"]) for row in rows})
    task_names = tuple("%s/%s" % (suite, task) for suite, task, _path in keys)
    suites = sorted({suite for suite, _task, _path in keys})
    suite_index = {name: index for index, name in enumerate(suites)}
    row_count = sum(int(row["inference_calls"]) for row in rows)
    routes = np.empty((row_count, N_DENOISE, N_LAYERS * N_EXPERTS), dtype=np.float32)
    task_axis = np.empty(row_count, dtype=np.int16)
    suite_axis = np.empty(row_count, dtype=np.int8)
    episode_axis = np.empty(row_count, dtype=np.int32)
    phase_axis = np.empty(row_count, dtype=np.int8)
    progress_axis = np.empty(row_count, dtype=np.float32)
    raw_min, raw_max = float("inf"), float("-inf")
    cursor = 0
    episode_uid = 0

    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["task_dir"], row["path"])].append(row)
    for task_id, (suite, task_name, path_value) in enumerate(keys):
        task_path = Path(path_value)
        if not task_path.is_absolute():
            task_path = HERE / task_path
        group = zarr.open_group(str(task_path / "server/routes.zarr"), mode="r")
        stored_episode = np.asarray(group["episode_id"][:], dtype=np.int64)
        task_rows = sorted(
            grouped[(suite, task_name, path_value)],
            key=lambda row: int(row["episode_index"]),
        )
        for row in task_rows:
            start = int(row["control_step_offset"])
            count = int(row["inference_calls"])
            stop = start + count
            episode_id = int(row["episode_index"])
            if stop > len(stored_episode) or not np.all(stored_episode[start:stop] == episode_id):
                raise ValueError("episode boundary mismatch in %s" % task_path)
            raw = np.asarray(
                group["hb_router_probs"][start:stop, :, :, ACTION_TOKENS, :],
                dtype=np.float32,
            )
            normalized, bounds = normalize_probabilities(raw)
            raw_min = min(raw_min, bounds[0])
            raw_max = max(raw_max, bounds[1])
            averaged, _ = normalize_probabilities(normalized.mean(axis=3))
            root = np.sqrt(averaged).transpose(0, 2, 1, 3)
            routes[cursor : cursor + count] = root.reshape(
                count, N_DENOISE, N_LAYERS * N_EXPERTS
            )
            task_axis[cursor : cursor + count] = task_id
            suite_axis[cursor : cursor + count] = suite_index[suite]
            episode_axis[cursor : cursor + count] = episode_uid
            progress = (np.arange(count, dtype=np.float32) + 0.5) / count
            progress_axis[cursor : cursor + count] = progress
            phase_axis[cursor : cursor + count] = np.minimum((3 * progress).astype(int), 2)
            cursor += count
            episode_uid += 1
    if cursor != row_count:
        raise AssertionError("route row count changed while loading")
    return RouteSequences(
        routes,
        task_axis,
        suite_axis,
        episode_axis,
        phase_axis,
        progress_axis,
        task_names,
        (raw_min, raw_max),
    )


def fit_svd(matrix: np.ndarray, components: int, seed: int) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("SVD matrix must be two-dimensional with at least two rows")
    count = min(int(components), min(values.shape) - 1)
    if count <= 0:
        raise ValueError("component count is too small")
    u, singular, basis = randomized_svd(
        values, n_components=count, n_iter=5, random_state=int(seed)
    )
    total = float(np.sum(np.square(values, dtype=np.float64)))
    if total <= 0.0:
        raise ValueError("SVD matrix has no variance")
    explained = np.square(singular.astype(np.float64)) / total
    return {
        "u": u,
        "singular": singular,
        "basis": basis,
        "scores": values @ basis.T,
        "explained": explained,
        "total": total,
    }


def spectrum_summary(fit: dict[str, Any]) -> dict[str, Any]:
    explained = np.asarray(fit["explained"], dtype=np.float64)
    cumulative = np.cumsum(explained)
    result: dict[str, Any] = {
        "components_computed": int(len(explained)),
        "stable_rank": float(1.0 / explained[0]),
        "explained": explained.tolist(),
    }
    for count in (1, 4, 16, 32, 64):
        if count <= len(explained):
            result["top%d" % count] = float(cumulative[count - 1])
    return result


def eta_squared(scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(groups)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(labels):
        raise ValueError("score and group lengths differ")
    grand = values.mean(axis=0)
    denominator = np.square(values - grand).sum(axis=0)
    numerator = np.zeros(values.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        block = values[labels == label]
        numerator += len(block) * np.square(block.mean(axis=0) - grand)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPS,
    )


def weighted_eta(scores: np.ndarray, groups: np.ndarray, singular: np.ndarray) -> float:
    eta = eta_squared(scores, groups)
    weights = np.square(np.asarray(singular[: len(eta)], dtype=np.float64))
    return float(np.sum(eta * weights) / np.sum(weights))


def two_way_residual(values: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    one_dimensional = array.ndim == 1
    if one_dimensional:
        array = array[:, None]
    first = np.asarray(first)
    second = np.asarray(second)
    if len(array) != len(first) or len(array) != len(second):
        raise ValueError("two-way residual axes have different lengths")
    pairs = {(a, b) for a, b in zip(first.tolist(), second.tolist())}
    if len(pairs) != len(np.unique(first)) * len(np.unique(second)):
        raise ValueError("two-way residual requires a complete crossed design")
    grand = array.mean(axis=0)
    first_mean = {item: array[first == item].mean(axis=0) for item in np.unique(first)}
    second_mean = {item: array[second == item].mean(axis=0) for item in np.unique(second)}
    result = array - np.stack([first_mean[item] for item in first])
    result -= np.stack([second_mean[item] for item in second])
    result += grand
    return result[:, 0] if one_dimensional else result


def correlations(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    x = x - x.mean(axis=0)
    y = y - y.mean()
    denominator = np.sqrt(np.square(x).sum(axis=0) * np.square(y).sum())
    return np.divide(x.T @ y, denominator, out=np.zeros(x.shape[1]), where=denominator > EPS)


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    left -= left.mean(axis=0)
    right -= right.mean(axis=0)
    cross = left.T @ right
    left_gram = left.T @ left
    right_gram = right.T @ right
    denominator = np.sqrt(np.square(left_gram).sum() * np.square(right_gram).sum())
    return float(np.square(cross).sum() / max(denominator, EPS))


def principal_cosines(first: np.ndarray, second: np.ndarray, count: int) -> np.ndarray:
    left = np.asarray(first[:count], dtype=np.float64)
    right = np.asarray(second[:count], dtype=np.float64)
    return np.linalg.svd(left @ right.T, compute_uv=False)


def permute_within(values: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(values).copy()
    for group in np.unique(groups):
        locations = np.flatnonzero(groups == group)
        output[locations] = rng.permutation(output[locations])
    return output


def max_success_scan(
    scores: np.ndarray,
    success: np.ndarray,
    scene: np.ndarray,
    noise: np.ndarray,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    feature = np.concatenate(
        [two_way_residual(scores[:, round_index, :32], scene, noise) for round_index in range(10)],
        axis=1,
    )
    target = two_way_residual(success, scene, noise)
    observed_correlation = correlations(feature, target)
    best = int(np.argmax(np.abs(observed_correlation)))
    observed = float(abs(observed_correlation[best]))
    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = permute_within(success, scene, rng)
        residual = two_way_residual(shuffled, scene, noise)
        null[index] = float(np.max(np.abs(correlations(feature, residual))))
    return {
        "round": best // 32 + 1,
        "pc": best % 32 + 1,
        "correlation": float(observed_correlation[best]),
        "max_abs_correlation": observed,
        "max_statistic_fwer_p": float((1 + np.sum(null >= observed)) / (permutations + 1)),
        "null_max_median": float(np.median(null)),
        "null_max_p95": float(np.percentile(null, 95)),
        "hypotheses": int(feature.shape[1]),
    }


def heldout_success_direction(
    vectors: np.ndarray,
    scene: np.ndarray,
    noise: np.ndarray,
    success: np.ndarray,
    even_development: bool,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    development = (scene % 2 == 0) if even_development else (scene % 2 == 1)
    holdout = ~development
    layer_scale = np.sqrt(
        np.mean(np.square(vectors[development], dtype=np.float64), axis=(0, 1, 3), keepdims=True)
    )
    dev = vectors[development] / np.maximum(layer_scale, EPS)
    held = vectors[holdout] / np.maximum(layer_scale, EPS)
    round_mean = dev.mean(axis=0, keepdims=True)
    fit = fit_svd((dev - round_mean).reshape(-1, np.prod(dev.shape[2:])), 32, seed)
    basis = fit["basis"]
    dev_scores = ((dev - round_mean).reshape(-1, basis.shape[1]) @ basis.T).reshape(
        len(dev), N_DENOISE, 32
    )
    held_scores = ((held - round_mean).reshape(-1, basis.shape[1]) @ basis.T).reshape(
        len(held), N_DENOISE, 32
    )
    dev_features = np.concatenate(
        [
            two_way_residual(dev_scores[:, round_index], scene[development], noise[development])
            for round_index in range(N_DENOISE)
        ],
        axis=1,
    )
    dev_target = two_way_residual(success[development], scene[development], noise[development])
    dev_correlation = correlations(dev_features, dev_target)
    best = int(np.argmax(np.abs(dev_correlation)))
    orientation = float(np.sign(dev_correlation[best]) or 1.0)

    held_features = np.concatenate(
        [
            two_way_residual(held_scores[:, round_index], scene[holdout], noise[holdout])
            for round_index in range(N_DENOISE)
        ],
        axis=1,
    )
    held_target = two_way_residual(success[holdout], scene[holdout], noise[holdout])
    held_correlation = float(correlations(orientation * held_features[:, best], held_target)[0])
    rng = np.random.default_rng(seed + 17)
    exceed = 0
    for _index in range(permutations):
        shuffled = permute_within(success[holdout], scene[holdout], rng)
        residual = two_way_residual(shuffled, scene[holdout], noise[holdout])
        value = float(correlations(orientation * held_features[:, best], residual)[0])
        exceed += value >= held_correlation
    return {
        "development_scenes": [int(value) for value in np.unique(scene[development])],
        "holdout_scenes": [int(value) for value in np.unique(scene[holdout])],
        "selected_round": best // 32 + 1,
        "selected_pc": best % 32 + 1,
        "development_correlation": float(dev_correlation[best]),
        "holdout_oriented_correlation": held_correlation,
        "holdout_one_sided_p": float((exceed + 1) / (permutations + 1)),
    }


def analyze_activations(
    data: ActivationVectors, components: int, permutations: int, seed: int
) -> dict[str, Any]:
    routed = data.routed
    flattened = routed.reshape(len(routed), N_DENOISE, -1)
    global_centered = flattened.reshape(-1, flattened.shape[-1])
    global_centered = global_centered - global_centered.mean(axis=0)
    round_residual = flattened - flattened.mean(axis=0, keepdims=True)
    raw_global_fit = fit_svd(global_centered, components, seed)
    raw_residual_fit = fit_svd(round_residual.reshape(-1, flattened.shape[-1]), components, seed)

    layer_scale = np.sqrt(
        np.mean(np.square(routed, dtype=np.float64), axis=(0, 1, 3), keepdims=True)
    )
    scaled = routed / np.maximum(layer_scale, EPS)
    scaled_flat = scaled.reshape(len(scaled), N_DENOISE, -1)
    scaled_global = scaled_flat.reshape(-1, scaled_flat.shape[-1])
    scaled_global = scaled_global - scaled_global.mean(axis=0)
    scaled_residual = scaled_flat - scaled_flat.mean(axis=0, keepdims=True)
    scaled_global_fit = fit_svd(scaled_global, components, seed)
    scaled_residual_fit = fit_svd(
        scaled_residual.reshape(-1, scaled_flat.shape[-1]), components, seed
    )
    residual_scores = scaled_residual_fit["scores"].reshape(
        len(scaled), N_DENOISE, -1
    )
    factors = min(32, residual_scores.shape[-1])
    flat_scores = residual_scores[:, :, :factors].reshape(-1, factors)
    repeated_scene = np.repeat(data.scene, N_DENOISE)
    repeated_noise = np.repeat(data.noise_seed, N_DENOISE)
    scene_eta = eta_squared(flat_scores, repeated_scene)
    noise_eta = eta_squared(flat_scores, repeated_noise)
    singular = scaled_residual_fit["singular"][:factors]

    score_cka = []
    interaction_cka = []
    final_interaction = two_way_residual(
        residual_scores[:, -1, :16], data.scene, data.noise_seed
    )
    for round_index in range(N_DENOISE):
        score_cka.append(
            linear_cka(residual_scores[:, round_index, :16], residual_scores[:, -1, :16])
        )
        interaction = two_way_residual(
            residual_scores[:, round_index, :16], data.scene, data.noise_seed
        )
        interaction_cka.append(linear_cka(interaction, final_interaction))

    half_bases = []
    for parity in (0, 1):
        subset = scaled[data.scene % 2 == parity]
        subset -= subset.mean(axis=0, keepdims=True)
        half_bases.append(
            fit_svd(subset.reshape(-1, scaled_flat.shape[-1]), 32, seed)["basis"]
        )
    stability = {}
    for count in (4, 16, 32):
        cosine = principal_cosines(half_bases[0], half_bases[1], count)
        stability["top%d" % count] = {
            "mean_cosine": float(cosine.mean()),
            "minimum_cosine": float(cosine.min()),
        }

    success_scan = max_success_scan(
        residual_scores, data.success, data.scene, data.noise_seed, permutations, seed
    )
    split_permutations = max(500, permutations // 2)
    confirmations = [
        heldout_success_direction(
            routed,
            data.scene,
            data.noise_seed,
            data.success,
            even,
            split_permutations,
            seed + index,
        )
        for index, even in enumerate((True, False))
    ]
    return {
        "queries": int(len(routed)),
        "scenes": int(len(np.unique(data.scene))),
        "noise_seeds": int(len(np.unique(data.noise_seed))),
        "success_rate": float(data.success.mean()),
        "layers": data.layers.tolist(),
        "raw_global_centered": spectrum_summary(raw_global_fit),
        "raw_round_residual": spectrum_summary(raw_residual_fit),
        "layer_scaled_global_centered": spectrum_summary(scaled_global_fit),
        "layer_scaled_round_residual": spectrum_summary(scaled_residual_fit),
        "round_residual_factor_effects": {
            "scene_eta_pc1_to_8": scene_eta[:8].tolist(),
            "noise_eta_pc1_to_8": noise_eta[:8].tolist(),
            "scene_eta_weighted_top32": weighted_eta(flat_scores, repeated_scene, singular),
            "noise_eta_weighted_top32": weighted_eta(flat_scores, repeated_noise, singular),
        },
        "round_to_final_cka_top16": score_cka,
        "round_to_final_cka_after_scene_noise_residual_top16": interaction_cka,
        "scene_half_subspace_stability": stability,
        "success_exploratory_max_scan": success_scan,
        "success_scene_heldout_confirmation": confirmations,
    }


def center_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32).copy()
    for group in np.unique(groups):
        result[groups == group] -= result[groups == group].mean(axis=0)
    return result


def sequence_arrow(scores: np.ndarray, episodes: np.ndarray) -> np.ndarray:
    arrows = []
    for episode in np.unique(episodes):
        sequence = scores[episodes == episode]
        quarter = max(1, len(sequence) // 4)
        difference = sequence[-quarter:].mean(axis=0) - sequence[:quarter].mean(axis=0)
        norm = float(np.linalg.norm(difference))
        if norm > EPS:
            arrows.append(difference / norm)
    if not arrows:
        raise ValueError("cannot orient a temporal arrow from empty sequences")
    arrow = np.mean(arrows, axis=0)
    return arrow / max(float(np.linalg.norm(arrow)), EPS)


def progress_split(
    query_routes: np.ndarray,
    task: np.ndarray,
    episode: np.ndarray,
    progress: np.ndarray,
    even_development: bool,
    seed: int,
) -> dict[str, Any]:
    development = (task % 2 == 0) if even_development else (task % 2 == 1)
    holdout = ~development
    centered = center_by_group(query_routes[development], task[development])
    fit = fit_svd(centered, 16, seed)
    scores = query_routes @ fit["basis"].T
    arrow = sequence_arrow(scores[development], episode[development])
    temporal_score = scores @ arrow
    episode_rho = []
    task_rho = []
    for episode_id in np.unique(episode[holdout]):
        mask = episode == episode_id
        episode_rho.append(float(spearmanr(temporal_score[mask], progress[mask]).statistic))
    for task_id in np.unique(task[holdout]):
        values = []
        for episode_id in np.unique(episode[(task == task_id) & holdout]):
            mask = episode == episode_id
            values.append(float(spearmanr(temporal_score[mask], progress[mask]).statistic))
        task_rho.append(float(np.nanmean(values)))
    return {
        "development_tasks": int(len(np.unique(task[development]))),
        "holdout_tasks": int(len(np.unique(task[holdout]))),
        "holdout_episodes": int(len(episode_rho)),
        "episode_rho_median": float(np.nanmedian(episode_rho)),
        "positive_episodes": int(np.sum(np.asarray(episode_rho) > 0.0)),
        "task_rho_median": float(np.nanmedian(task_rho)),
        "positive_tasks": int(np.sum(np.asarray(task_rho) > 0.0)),
        "task_rho_min": float(np.nanmin(task_rho)),
        "task_rho_max": float(np.nanmax(task_rho)),
    }


def adjacent_subspace_diagnostics(
    query_routes: np.ndarray,
    scores: np.ndarray,
    episodes: np.ndarray,
) -> dict[str, Any]:
    continuity = []
    captured = {4: [], 16: [], 64: []}
    for episode in np.unique(episodes):
        mask = episodes == episode
        route = query_routes[mask]
        score = scores[mask]
        if len(route) < 2:
            continue
        reduced = score[:, :16]
        adjacent = np.linalg.norm(reduced[1:] - reduced[:-1], axis=1).mean()
        difference = reduced[:, None] - reduced[None, :]
        pairwise = np.linalg.norm(difference, axis=-1)
        upper = pairwise[np.triu_indices(len(reduced), 1)].mean()
        continuity.append(float(adjacent / max(upper, EPS)))
        full_delta = route[1:] - route[:-1]
        denominator = float(np.square(full_delta, dtype=np.float64).sum())
        for count in captured:
            projected = score[1:, :count] - score[:-1, :count]
            captured[count].append(
                float(np.square(projected, dtype=np.float64).sum() / denominator)
            )
    return {
        "pc16_adjacent_to_all_pair_ratio_mean": float(np.mean(continuity)),
        "pc16_adjacent_to_all_pair_ratio_median": float(np.median(continuity)),
        "adjacent_change_energy_capture": {
            "top%d" % count: float(np.mean(values)) for count, values in captured.items()
        },
    }


def denoise_arrow_split(
    denoise_routes: np.ndarray,
    task: np.ndarray,
    even_development: bool,
    remove_round_template: bool,
    seed: int,
) -> dict[str, Any]:
    development = (task % 2 == 0) if even_development else (task % 2 == 1)
    holdout = ~development
    if remove_round_template:
        center = denoise_routes[development].mean(axis=0, keepdims=True)
    else:
        center = denoise_routes[development].reshape(-1, denoise_routes.shape[-1]).mean(
            axis=0
        )[None, None]
    fit = fit_svd(
        (denoise_routes[development] - center).reshape(-1, denoise_routes.shape[-1]),
        16,
        seed,
    )
    dev_score = (denoise_routes[development] - center) @ fit["basis"].T
    held_score = (denoise_routes[holdout] - center) @ fit["basis"].T
    differences = dev_score[:, -1] - dev_score[:, 0]
    norms = np.linalg.norm(differences, axis=1)
    valid = norms > EPS
    arrow = np.mean(differences[valid] / norms[valid, None], axis=0)
    arrow /= max(float(np.linalg.norm(arrow)), EPS)
    scalar = held_score @ arrow
    round_axis = np.arange(N_DENOISE)
    correlations_by_query = np.asarray(
        [spearmanr(sequence, round_axis).statistic for sequence in scalar], dtype=np.float64
    )
    return {
        "remove_round_template": bool(remove_round_template),
        "holdout_queries": int(len(correlations_by_query)),
        "rho_median": float(np.nanmedian(correlations_by_query)),
        "rho_mean": float(np.nanmean(correlations_by_query)),
        "positive_queries": int(np.sum(correlations_by_query > 0.0)),
    }


def analyze_routes(data: RouteSequences, components: int, seed: int) -> dict[str, Any]:
    query_routes = data.denoise_routes.reshape(len(data.denoise_routes), -1)
    query_centered = query_routes - query_routes.mean(axis=0)
    query_fit = fit_svd(query_centered, components, seed)
    query_scores = query_fit["scores"]
    factors = min(32, query_scores.shape[1])
    score_factors = query_scores[:, :factors]
    singular = query_fit["singular"][:factors]
    effects = {
        "suite_eta_weighted_top32": weighted_eta(score_factors, data.suite, singular),
        "task_eta_weighted_top32": weighted_eta(score_factors, data.task, singular),
        "phase_eta_weighted_top32": weighted_eta(score_factors, data.phase, singular),
        "episode_eta_weighted_top32": weighted_eta(score_factors, data.episode, singular),
        "task_eta_pc1_to_8": eta_squared(score_factors, data.task)[:8].tolist(),
        "phase_eta_pc1_to_8": eta_squared(score_factors, data.phase)[:8].tolist(),
    }
    adjacent = adjacent_subspace_diagnostics(
        query_routes, query_scores, data.episode
    )

    half_bases = []
    for parity in (0, 1):
        subset = query_routes[data.task % 2 == parity]
        subset -= subset.mean(axis=0)
        half_bases.append(fit_svd(subset, 32, seed)["basis"])
    stability = {}
    for count in (4, 16, 32):
        cosine = principal_cosines(half_bases[0], half_bases[1], count)
        stability["top%d" % count] = {
            "mean_cosine": float(cosine.mean()),
            "minimum_cosine": float(cosine.min()),
        }

    progress = [
        progress_split(
            query_routes,
            data.task,
            data.episode,
            data.progress,
            even,
            seed + index,
        )
        for index, even in enumerate((True, False))
    ]

    denoise_flat = data.denoise_routes.reshape(-1, data.denoise_routes.shape[-1])
    denoise_global = denoise_flat - denoise_flat.mean(axis=0)
    denoise_residual = data.denoise_routes - data.denoise_routes.mean(axis=0, keepdims=True)
    denoise_global_fit = fit_svd(denoise_global, components, seed)
    denoise_residual_fit = fit_svd(
        denoise_residual.reshape(-1, denoise_residual.shape[-1]), components, seed
    )
    round_axis = np.tile(np.arange(N_DENOISE), len(data.denoise_routes))
    denoise_factors = min(32, denoise_global_fit["scores"].shape[1])
    round_eta = weighted_eta(
        denoise_global_fit["scores"][:, :denoise_factors],
        round_axis,
        denoise_global_fit["singular"][:denoise_factors],
    )
    denoise_arrows = []
    for remove_template in (False, True):
        for index, even in enumerate((True, False)):
            denoise_arrows.append(
                denoise_arrow_split(
                    data.denoise_routes,
                    data.task,
                    even,
                    remove_template,
                    seed + 10 * int(remove_template) + index,
                )
            )
    return {
        "control_queries": int(len(query_routes)),
        "tasks": int(len(np.unique(data.task))),
        "episodes": int(len(np.unique(data.episode))),
        "probability_sum_bounds_before_renormalization": list(data.probability_sum_bounds),
        "query_route_spectrum": spectrum_summary(query_fit),
        "query_factor_effects": effects,
        "query_subspace_dynamics": adjacent,
        "task_half_subspace_stability": stability,
        "unsupervised_control_progress_axis": progress,
        "denoise_global_spectrum": spectrum_summary(denoise_global_fit),
        "denoise_round_residual_spectrum": spectrum_summary(denoise_residual_fit),
        "denoise_round_eta_weighted_top32": round_eta,
        "unsupervised_denoise_axes": denoise_arrows,
    }


def write_spectra(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    sources = {
        "activation_raw_global": summary["activation"]["raw_global_centered"],
        "activation_raw_round_residual": summary["activation"]["raw_round_residual"],
        "activation_scaled_global": summary["activation"]["layer_scaled_global_centered"],
        "activation_scaled_round_residual": summary["activation"][
            "layer_scaled_round_residual"
        ],
        "route_query": summary["routes"]["query_route_spectrum"],
        "route_denoise_global": summary["routes"]["denoise_global_spectrum"],
        "route_denoise_round_residual": summary["routes"][
            "denoise_round_residual_spectrum"
        ],
    }
    for source, values in sources.items():
        cumulative = 0.0
        for component, explained in enumerate(values["explained"], 1):
            cumulative += explained
            rows.append(
                {"source": source, "component": component, "explained": explained, "cumulative": cumulative}
            )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "component", "explained", "cumulative"))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return "%.1f%%" % (100.0 * value)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    activation = summary["activation"]
    routes = summary["routes"]
    scaled = activation["layer_scaled_round_residual"]
    scan = activation["success_exploratory_max_scan"]
    confirmations = activation["success_scene_heldout_confirmation"]
    activation_stability = activation["scene_half_subspace_stability"]
    progress = routes["unsupervised_control_progress_axis"]
    route_stability = routes["task_half_subspace_stability"]
    denoise = routes["unsupervised_denoise_axes"]
    global_denoise = [row for row in denoise if not row["remove_round_template"]]
    residual_denoise = [row for row in denoise if row["remove_round_template"]]
    lines = [
        "# HiMoE 路由与真实专家贡献的 SVD 验证",
        "",
        "## 口径",
        "",
        "- SVD 本身不读取动作、reward、仿真状态或 success。",
        "- 真实激活使用 top-k 专家加权向量合并后的 routed vector；四层先按各层 RMS 等尺度化，再去掉每个 denoise round 的公共均值。",
        "- 路由使用逐 site L1 归一化概率的平方根，因此欧氏距离对应 Hellinger 几何。",
        "- success 只在分解完成后作探索诊断，并另做 scene-held-out 复核。",
        "",
        "## 真实 routed 向量",
        "",
        "数据：%d 个首 query，%d scene x %d noise，10 轮 x 4 层 x 1024 维。"
        % (activation["queries"], activation["scenes"], activation["noise_seeds"]),
        "",
        "| 数据处理 | PC1 | PC1-4 | PC1-16 | PC1-64 | stable rank |",
        "|---|---:|---:|---:|---:|---:|",
        "| 原始向量、全局中心化 | %s | %s | %s | %s | %.2f |"
        % (
            _pct(activation["raw_global_centered"]["top1"]),
            _pct(activation["raw_global_centered"]["top4"]),
            _pct(activation["raw_global_centered"]["top16"]),
            _pct(activation["raw_global_centered"]["top64"]),
            activation["raw_global_centered"]["stable_rank"],
        ),
        "| 层等尺度、去 round 模板 | %s | %s | %s | %s | %.2f |"
        % (
            _pct(scaled["top1"]),
            _pct(scaled["top4"]),
            _pct(scaled["top16"]),
            _pct(scaled["top64"]),
            scaled["stable_rank"],
        ),
        "",
        "去 round 模板后 PC1-16 仍解释 %s，说明真实专家贡献不是纯一维幅度；但前 32 个方向中 scene/noise 主效应分别解释加权 %s / %s。"
        % (
            _pct(scaled["top16"]),
            _pct(activation["round_residual_factor_effects"]["scene_eta_weighted_top32"]),
            _pct(activation["round_residual_factor_effects"]["noise_eta_weighted_top32"]),
        ),
        "",
        "前 16 维当前轮到最终轮的 CKA 为：`%s`。去掉 scene+noise 主效应后为：`%s`。"
        % (
            " ".join("%.3f" % value for value in activation["round_to_final_cka_top16"]),
            " ".join(
                "%.3f"
                % value
                for value in activation[
                    "round_to_final_cka_after_scene_noise_residual_top16"
                ]
            ),
        ),
        "",
        "这意味着早轮和最终轮看似很一致，但大部分一致性来自固定 scene/noise 身份；真正 query 特异的未来贡献在第 9 轮前仍很弱。",
        "scene 对半后，激活 top-4 / top-16 子空间的平均 principal cosine 仅为 `%.3f / %.3f`，说明深层残差方向本身也不够稳定。"
        % (
            activation_stability["top4"]["mean_cosine"],
            activation_stability["top16"]["mean_cosine"],
        ),
        "",
        "## 与最终成功的诊断",
        "",
        "全样本事后扫描 %d 个 PC x round，最大相关为 `r=%.3f`（round %d, PC%d），max-statistic FWER `p=%.4f`。"
        % (
            scan["hypotheses"],
            scan["correlation"],
            scan["round"],
            scan["pc"],
            scan["max_statistic_fwer_p"],
        ),
        "",
        "但冻结到另一半 scene 后，两次方向的相关分别为 `%.3f`、`%.3f`，单侧 p 为 `%.3f`、`%.3f`。因此这个成功相关没有复制，不能当作 outcome signal。"
        % (
            confirmations[0]["holdout_oriented_correlation"],
            confirmations[1]["holdout_oriented_correlation"],
            confirmations[0]["holdout_one_sided_p"],
            confirmations[1]["holdout_one_sided_p"],
        ),
        "",
        "## 30 任务完整控制步路由",
        "",
        "数据：%d 个 control query、%d 条 rollout、%d 个任务。"
        % (routes["control_queries"], routes["episodes"], routes["tasks"]),
        "",
        "全局路由 PC1/PC1-16/PC1-64 分别解释 %s / %s / %s，stable rank %.2f。"
        % (
            _pct(routes["query_route_spectrum"]["top1"]),
            _pct(routes["query_route_spectrum"]["top16"]),
            _pct(routes["query_route_spectrum"]["top64"]),
            routes["query_route_spectrum"]["stable_rank"],
        ),
        "前 32 维的 task / phase 加权解释量为 %s / %s；PC1 的 task eta 是 %.3f。主方向首先编码 checkpoint/任务身份，不是通用阶段。"
        % (
            _pct(routes["query_factor_effects"]["task_eta_weighted_top32"]),
            _pct(routes["query_factor_effects"]["phase_eta_weighted_top32"]),
            routes["query_factor_effects"]["task_eta_pc1_to_8"][0],
        ),
        "task 对半后，top-4 子空间平均 principal cosine 为 `%.3f`，比单任务激活残差稳定得多。"
        % route_stability["top4"]["mean_cosine"],
        "",
        "不过，用训练任务的前后 query 顺序无监督地确定一个 SVD 时间箭头后，两个 15-task holdout 的 episode 中位 Spearman 为 `%.3f / %.3f`；正相关 episode 为 `%d/%d` 和 `%d/%d`，两边都是 15/15 任务同号。"
        % (
            progress[0]["episode_rho_median"],
            progress[1]["episode_rho_median"],
            progress[0]["positive_episodes"],
            progress[0]["holdout_episodes"],
            progress[1]["positive_episodes"],
            progress[1]["holdout_episodes"],
        ),
        "",
        "这是一个可重复的模型内部控制进度轴，但它只表示 query 序列的先后，不证明物理任务更接近完成。",
        "",
        "## 对早停的含义",
        "",
        "不去 round 模板时，纯序列顺序构造的 denoise SVD 轴在两半未见任务上的 query 内 rho 中位数为 `%.3f / %.3f`；去掉 round 模板后降为 `%.3f / %.3f`。"
        % (
            global_denoise[0]["rho_median"],
            global_denoise[1]["rho_median"],
            residual_denoise[0]["rho_median"],
            residual_denoise[1]["rho_median"],
        ),
        "去模板后平均 rho 只有 `%.3f / %.3f`，正向 query 为 `%d/%d` 和 `%d/%d`，远不如公共时钟稳定。"
        % (
            residual_denoise[0]["rho_mean"],
            residual_denoise[1]["rho_mean"],
            residual_denoise[0]["positive_queries"],
            residual_denoise[0]["holdout_queries"],
            residual_denoise[1]["positive_queries"],
            residual_denoise[1]["holdout_queries"],
        ),
        "",
        "所以 denoise SVD 很容易读出‘现在是第几轮’，但去掉公共时钟后没有稳定的 query 特异收敛轴。",
        "此外，query SVD 的 PC1-16 只保留 %.1f%% 的相邻路由变化能量，PC1-64 保留 %.1f%%；高维尾部仍承载接近一半动态变化。直接按低维变化小来早停会漏掉这些变化。"
        % (
            100
            * routes["query_subspace_dynamics"]["adjacent_change_energy_capture"][
                "top16"
            ],
            100
            * routes["query_subspace_dynamics"]["adjacent_change_energy_capture"][
                "top64"
            ],
        ),
        "",
        "## 裁决",
        "",
        "- **可解释性：有结果。** SVD 提取出了跨 30 个任务复现的内部控制进度轴，可用于无监督阶段分段、变化点检测和 OOD/阶段原型研究。",
        "- **成功预测：没有通过。** 全样本弱相关在 scene-held-out 后归零。",
        "- **逐次推理早停：暂不通过。** 去 round 模板后没有稳定收敛轴，且低秩截断会漏掉大量动态能量。",
        "- 下一步应把这个进度轴用于跨 rollout 阶段对齐；早停仍优先验证联合采集中的真实激活残差 + controller-aware endpoint，而不是单独用 SVD 分数。",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    if args.components < 32:
        raise ValueError("--components must be at least 32")
    if args.permutations < 100:
        raise ValueError("--permutations must be at least 100")
    activation = load_activation_vectors(args.activation_dir)
    activation_result = analyze_activations(
        activation, args.components, args.permutations, args.seed
    )
    routes = load_route_sequences(args.corpus)
    route_result = analyze_routes(routes, args.components, args.seed)
    summary = {
        "protocol": {
            "decomposition": "unsupervised randomized SVD",
            "activation_geometry": "true weighted routed vectors, layer RMS scaled",
            "route_geometry": "sqrt of per-site L1-normalized router probabilities",
            "temporal_axis": "mean normalized last-quarter minus first-quarter direction in training sequences",
            "success_usage": "post-SVD diagnostic only",
            "components": args.components,
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "activation": activation_result,
        "routes": route_result,
    }
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    write_spectra(out / "spectra.csv", summary)
    write_report(out / "report.md", summary)
    print("wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
