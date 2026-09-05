#!/usr/bin/env python3
"""Evaluate whether structured HB-MoE state predicts a later stasis trap.

The target is constructed only from future simulator object poses.  Every
predictor is causal: it uses HB router probabilities or the HB-MoE input hidden
state no later than the evaluated control step.  Generalization holds out both
initial-state groups and flow-noise-seed groups.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
HUB = HERE.parent
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
RUN = HUB / "cache" / "HiMoE-VLA" / "libero_long" / TASK / "right-16x32"

HORIZONS = (7, 12, 20, 27, 34)
EARLY_HORIZONS = (7, 12)
HISTORY = 7
N_EXPERTS = 32
PROJECTED_DIM = 384
PCA_PER_BLOCK = 12
RIDGE_ALPHA = 30.0
GOAL_RADIUS_M = 0.05
PROGRESS_EPS_M = 0.01
BOOTSTRAPS = 2_000

BLOCK_NAMES = (
    "route_identity",
    "route_geometry",
    "route_temporal",
    "hidden_identity",
    "hidden_structure",
    "hidden_temporal",
    "route_state",
    "route_action_1_3",
    "route_action_4_7",
    "route_action_8_10",
    "proprio",
    "sim",
)

FAMILIES = {
    "route_current": ("route_identity", "route_geometry"),
    "route_history": ("route_identity", "route_geometry", "route_temporal"),
    "hidden_current": ("hidden_identity", "hidden_structure"),
    "hidden_history": ("hidden_identity", "hidden_structure", "hidden_temporal"),
    "moe_current": (
        "route_identity",
        "route_geometry",
        "hidden_identity",
        "hidden_structure",
    ),
    "moe_history": (
        "route_identity",
        "route_geometry",
        "route_temporal",
        "hidden_identity",
        "hidden_structure",
        "hidden_temporal",
    ),
    "proprio_history": ("proprio",),
    "sim_history": ("sim",),
    "sim_plus_moe": (
        "sim",
        "route_identity",
        "route_geometry",
        "route_temporal",
        "hidden_identity",
        "hidden_structure",
        "hidden_temporal",
    ),
    "route_state": ("route_state",),
    "route_action_1_3": ("route_action_1_3",),
    "route_action_4_7": ("route_action_4_7",),
    "route_action_8_10": ("route_action_8_10",),
}


@dataclass(frozen=True)
class Episode:
    index: int
    state: int
    noise_seed: int
    failure: int
    offset: int
    length: int


@dataclass
class FoldModel:
    test_index: np.ndarray
    train_index: np.ndarray
    test_map: np.ndarray
    train_map: np.ndarray


class CountSketch:
    """A deterministic, data-independent signed hash projection."""

    def __init__(self, input_dim: int, output_dim: int, seed: int):
        rng = np.random.default_rng(seed)
        self.output_dim = output_dim
        self.bucket = rng.integers(0, output_dim, size=input_dim, dtype=np.int32)
        self.sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=input_dim)

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32).ravel()
        if len(values) != len(self.bucket):
            raise ValueError(f"CountSketch expected {len(self.bucket)} values, got {len(values)}")
        return np.bincount(
            self.bucket,
            weights=values * self.sign,
            minlength=self.output_dim,
        ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--cache", type=Path, default=HERE / "results" / "early_structure_features.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser.parse_args()


def load_episodes(run: Path) -> list[Episode]:
    rows = sorted(
        json.loads((run / "client" / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    lengths = np.asarray([row["inference_calls"] for row in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    episodes = [
        Episode(
            index=int(row["episode_index"]),
            state=int(row["init_state_id"]),
            noise_seed=int(row["flow_noise_seed"]),
            failure=int(not row["success"]),
            offset=int(offset),
            length=int(length),
        )
        for row, offset, length in zip(rows, offsets, lengths)
    ]
    if len(episodes) != 512:
        raise ValueError(f"expected 512 episodes, found {len(episodes)}")
    if min(episode.length for episode in episodes) <= max(HORIZONS):
        raise ValueError("not every episode has the common t0--t34 prefix")
    return episodes


def episode_path(run: Path, episode: Episode) -> Path:
    return run / "client" / f"episode_{episode.index:02d}.npz"


def normalize_probability(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    return values / np.maximum(values.sum(axis=-1, keepdims=True), 1e-12)


def normalize_vector(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norm, 1e-12), norm[..., 0]


def linear_slope(values: np.ndarray, axis: int) -> np.ndarray:
    count = values.shape[axis]
    if count < 2:
        return np.zeros(np.delete(values.shape, axis), dtype=np.float32)
    coordinate = np.linspace(-1.0, 1.0, count, dtype=np.float32)
    shape = [1] * values.ndim
    shape[axis] = count
    return np.sum(values * coordinate.reshape(shape), axis=axis) / float(np.square(coordinate).sum())


def summarize_axis(values: np.ndarray, axis: int) -> np.ndarray:
    return np.concatenate(
        [
            np.take(values, 0, axis=axis).ravel(),
            np.take(values, -1, axis=axis).ravel(),
            values.mean(axis=axis).ravel(),
            linear_slope(values, axis).ravel(),
        ]
    ).astype(np.float32)


def summarize_time_flow(values: np.ndarray) -> np.ndarray:
    """Summarize [time, layer, flow, token-like...] without merging layer/token."""
    time_summary = np.stack(
        [values[-1], values.mean(axis=0), linear_slope(values, 0)], axis=0
    )
    output = []
    for item in time_summary:
        output.append(summarize_axis(item, 1))
    return np.concatenate(output).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an, _ = normalize_vector(a)
    bn, _ = normalize_vector(b)
    return np.maximum(1.0 - np.sum(an * bn, axis=-1), 0.0)


def hellinger_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.5 * np.square(np.sqrt(a) - np.sqrt(b)).sum(axis=-1), 0.0))


def weighted_jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.minimum(a, b).sum(axis=-1) / np.maximum(
        np.maximum(a, b).sum(axis=-1), 1e-12
    )


def top4_mask(probability: np.ndarray) -> np.ndarray:
    ids = np.argpartition(probability, -4, axis=-1)[..., -4:]
    mask = np.zeros(probability.shape, dtype=np.bool_)
    np.put_along_axis(mask, ids, True, axis=-1)
    return mask


def jaccard_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    intersection = np.logical_and(a, b).sum(axis=-1)
    union = np.logical_or(a, b).sum(axis=-1)
    return 1.0 - intersection / np.maximum(union, 1)


def route_identity(probability: np.ndarray, token_slice: slice = slice(None)) -> np.ndarray:
    # probability: [layer, flow, token, expert]
    p = probability[:, :, token_slice, :]
    return np.concatenate(
        [p[:, -1].ravel(), p.mean(axis=1).ravel(), linear_slope(p, 1).ravel()]
    ).astype(np.float32)


def route_geometry(probability: np.ndarray) -> np.ndarray:
    p = probability
    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1) / math.log(N_EXPERTS)
    top4 = np.partition(p, -4, axis=-1)[..., -4:].sum(axis=-1)
    maximum = p.max(axis=-1)
    output = [summarize_axis(x, 1) for x in (entropy, top4, maximum)]

    hard = top4_mask(p)
    pair_sets = (
        (p[:, :, 1:-1], p[:, :, 2:]),
        (p[:, :, 0:1], p[:, :, 1:]),
    )
    hard_sets = (
        (hard[:, :, 1:-1], hard[:, :, 2:]),
        (hard[:, :, 0:1], hard[:, :, 1:]),
    )
    for (a, b), (ha, hb) in zip(pair_sets, hard_sets):
        for metric in (
            cosine_distance(a, b),
            hellinger_distance(a, b),
            weighted_jaccard_distance(a, b),
            jaccard_distance(ha, hb),
        ):
            output.append(summarize_axis(metric, 1))

    action = np.sqrt(p[:, :, 1:])
    gram = np.einsum("lfue,lfve->lfuv", action, action, optimize=True)
    eigenvalue = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    eigenvalue /= np.maximum(eigenvalue.sum(axis=-1, keepdims=True), 1e-12)
    effective_rank = np.exp(
        -np.sum(eigenvalue * np.log(np.maximum(eigenvalue, 1e-12)), axis=-1)
    ) / action.shape[2]
    upper = np.triu_indices(action.shape[2], 1)
    mean_similarity = gram[..., upper[0], upper[1]].mean(axis=-1)
    output.extend(
        [
            summarize_axis(eigenvalue, 1),
            summarize_axis(effective_rank, 1),
            summarize_axis(mean_similarity, 1),
        ]
    )
    return np.concatenate(output).astype(np.float32)


def route_temporal(history: np.ndarray, token_slice: slice = slice(None)) -> np.ndarray:
    # history: [control, layer, flow, token, expert]
    p = history[:, :, :, token_slice, :]
    hard = top4_mask(p)
    metrics = (
        cosine_distance(p[1:], p[:-1]),
        hellinger_distance(p[1:], p[:-1]),
        weighted_jaccard_distance(p[1:], p[:-1]),
        jaccard_distance(hard[1:], hard[:-1]),
    )
    output = [summarize_time_flow(metric) for metric in metrics]
    velocity = p[1:] - p[:-1]
    if len(velocity) >= 2:
        vn, vnorm = normalize_vector(velocity)
        momentum = np.sum(vn[1:] * vn[:-1], axis=-1)
        momentum[(vnorm[1:] <= 1e-12) | (vnorm[:-1] <= 1e-12)] = 0.0
        output.append(summarize_time_flow(momentum))
    return np.concatenate(output).astype(np.float32)


def hidden_identity(hidden: np.ndarray) -> np.ndarray:
    # hidden: [layer, flow, token, channel]
    normalized, _ = normalize_vector(hidden)
    return np.concatenate(
        [
            normalized[:, -1].ravel(),
            normalized.mean(axis=1).ravel(),
            linear_slope(normalized, 1).ravel(),
        ]
    ).astype(np.float32)


def hidden_structure(hidden: np.ndarray) -> np.ndarray:
    normalized, norm = normalize_vector(hidden)
    gram = np.einsum("lfud,lfvd->lfuv", normalized, normalized, optimize=True)
    upper = np.triu_indices(gram.shape[-1], 1)
    off_diagonal = gram[..., upper[0], upper[1]]
    eigenvalue = np.maximum(np.linalg.eigvalsh(gram), 0.0)
    eigenvalue /= np.maximum(eigenvalue.sum(axis=-1, keepdims=True), 1e-12)
    effective_rank = np.exp(
        -np.sum(eigenvalue * np.log(np.maximum(eigenvalue, 1e-12)), axis=-1)
    ) / gram.shape[-1]
    flow_distance = cosine_distance(normalized[:, 1:], normalized[:, :-1])
    return np.concatenate(
        [
            summarize_axis(off_diagonal, 1),
            summarize_axis(eigenvalue, 1),
            summarize_axis(effective_rank, 1),
            summarize_axis(norm / math.sqrt(hidden.shape[-1]), 1),
            summarize_axis(flow_distance, 1),
        ]
    ).astype(np.float32)


def hidden_temporal(history: np.ndarray) -> np.ndarray:
    normalized, norm = normalize_vector(history)
    direction_distance = cosine_distance(normalized[1:], normalized[:-1])
    log_norm_change = np.abs(
        np.log(np.maximum(norm[1:], 1e-12)) - np.log(np.maximum(norm[:-1], 1e-12))
    )
    output = [
        summarize_time_flow(direction_distance),
        summarize_time_flow(log_norm_change),
    ]
    velocity = normalized[1:] - normalized[:-1]
    if len(velocity) >= 2:
        vn, vnorm = normalize_vector(velocity)
        momentum = np.sum(vn[1:] * vn[:-1], axis=-1)
        momentum[(vnorm[1:] <= 1e-12) | (vnorm[:-1] <= 1e-12)] = 0.0
        output.append(summarize_time_flow(momentum))
    return np.concatenate(output).astype(np.float32)


def physical_history(history: np.ndarray) -> np.ndarray:
    current = history[-1]
    mean = history.mean(axis=0)
    slope = linear_slope(history, 0)
    velocity = history[-1] - history[-2]
    acceleration = history[-1] - 2.0 * history[-2] + history[-3]
    return np.concatenate([current, mean, slope, velocity, acceleration]).astype(np.float32)


def make_sketch(name: str, vector: np.ndarray, sketches: dict[str, CountSketch], seed: int) -> np.ndarray:
    if name not in sketches:
        sketches[name] = CountSketch(len(vector), PROJECTED_DIM, seed)
    return sketches[name].transform(vector)


def build_targets(
    episodes: list[Episode], sim_by_episode: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    terminal = []
    for episode, sim in zip(episodes, sim_by_episode):
        if not episode.failure:
            terminal.append((episode.state, episode.noise_seed, np.r_[sim[-1, 10:13], sim[-1, 17:20]]))

    terminal_distance = np.empty(len(episodes), dtype=np.float32)
    onset = np.full(len(episodes), -1, dtype=np.int16)
    sensitivity: dict[str, list[int]] = {}
    distance_series = []
    for episode, sim in zip(episodes, sim_by_episode):
        reference = np.stack(
            [
                pose
                for state, seed, pose in terminal
                if state != episode.state and seed != episode.noise_seed
            ]
        )
        pose = np.c_[sim[:, 10:13], sim[:, 17:20]]
        distance = cdist(pose, reference).min(axis=1).astype(np.float32)
        distance_series.append(distance)
        terminal_distance[episode.index] = distance[-1]
        if episode.failure:
            best = np.minimum.accumulate(distance)
            future_best = np.minimum.accumulate(distance[::-1])[::-1]
            eligible = np.flatnonzero(
                (best - future_best <= PROGRESS_EPS_M) & (best > GOAL_RADIUS_M)
            )
            if len(eligible):
                onset[episode.index] = int(eligible[0])

    for epsilon in (0.005, 0.01, 0.02):
        values = []
        for episode, distance in zip(episodes, distance_series):
            if not episode.failure:
                continue
            best = np.minimum.accumulate(distance)
            future_best = np.minimum.accumulate(distance[::-1])[::-1]
            eligible = np.flatnonzero(
                (best - future_best <= epsilon) & (best > GOAL_RADIUS_M)
            )
            if len(eligible):
                values.append(int(eligible[0]))
        sensitivity[f"epsilon_{epsilon:.3f}"] = values

    failure = np.asarray([episode.failure for episode in episodes], dtype=np.int8)
    stasis = (onset >= 0).astype(np.int8)
    included = (failure == 0) | (stasis == 1)
    values = onset[onset >= 0]
    metadata = {
        "goal_radius_m": GOAL_RADIUS_M,
        "progress_epsilon_m": PROGRESS_EPS_M,
        "n_success": int((failure == 0).sum()),
        "n_failure": int(failure.sum()),
        "n_stasis_trap": int(stasis.sum()),
        "n_ambiguous_failure_excluded": int(((failure == 1) & (stasis == 0)).sum()),
        "onset_quantiles": {
            str(q): float(np.quantile(values, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "terminal_distance_failure_quantiles_m": {
            str(q): float(np.quantile(terminal_distance[failure == 1], q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "sensitivity_onsets": {
            key: {
                "n": len(items),
                "median": float(np.median(items)),
                "q10": float(np.quantile(items, 0.1)),
                "q90": float(np.quantile(items, 0.9)),
            }
            for key, items in sensitivity.items()
        },
    }
    return stasis, included, onset, metadata


def extract_cache(run: Path, episodes: list[Episode], cache_path: Path, seed: int) -> dict[str, np.ndarray]:
    route_store = zarr.open(str(run / "server" / "routes.zarr"), mode="r")
    hidden_store = zarr.open(str(run / "server" / "hidden.zarr"), mode="r")
    route_episode = np.asarray(route_store["episode_id"][:], dtype=np.int32)
    hidden_episode = np.asarray(hidden_store["episode_id"][:], dtype=np.int32)
    if not np.array_equal(route_episode, hidden_episode):
        raise ValueError("route and hidden episode axes differ")

    count = len(episodes)
    horizon_count = len(HORIZONS)
    arrays = {
        name: np.empty((count, horizon_count, PROJECTED_DIM), dtype=np.float32)
        for name in BLOCK_NAMES
        if name not in ("proprio", "sim")
    }
    arrays["proprio"] = np.empty((count, horizon_count, 8 * 5), dtype=np.float32)
    arrays["sim"] = np.empty((count, horizon_count, 46 * 5), dtype=np.float32)
    sketches: dict[str, CountSketch] = {}
    sketch_seed = {name: seed + 1009 * (i + 1) for i, name in enumerate(BLOCK_NAMES)}

    for position, episode in enumerate(episodes, start=1):
        stop = episode.offset + max(HORIZONS) + 1
        if not np.all(route_episode[episode.offset:stop] == episode.index):
            raise ValueError(f"episode boundary mismatch for {episode.index}")
        route = normalize_probability(
            np.asarray(route_store["hb_router_probs"][episode.offset:stop], dtype=np.float32)
        )
        hidden = np.asarray(hidden_store["hb_hidden"][episode.offset:stop], dtype=np.float32)
        with np.load(episode_path(run, episode), allow_pickle=False) as payload:
            proprio = np.asarray(payload["state"][: max(HORIZONS) + 1], dtype=np.float32)
            sim = np.asarray(payload["sim_state"][: max(HORIZONS) + 1, 1:], dtype=np.float32)

        for h_index, horizon in enumerate(HORIZONS):
            lo = horizon - HISTORY + 1
            route_history = route[lo : horizon + 1]
            hidden_history_values = hidden[lo : horizon + 1]
            current_route = route[horizon]
            current_hidden = hidden[horizon]
            raw = {
                "route_identity": route_identity(current_route),
                "route_geometry": route_geometry(current_route),
                "route_temporal": route_temporal(route_history),
                "hidden_identity": hidden_identity(current_hidden),
                "hidden_structure": hidden_structure(current_hidden),
                "hidden_temporal": hidden_temporal(hidden_history_values),
                "route_state": np.concatenate(
                    [route_identity(current_route, slice(0, 1)), route_temporal(route_history, slice(0, 1))]
                ),
                "route_action_1_3": np.concatenate(
                    [route_identity(current_route, slice(1, 4)), route_temporal(route_history, slice(1, 4))]
                ),
                "route_action_4_7": np.concatenate(
                    [route_identity(current_route, slice(4, 8)), route_temporal(route_history, slice(4, 8))]
                ),
                "route_action_8_10": np.concatenate(
                    [route_identity(current_route, slice(8, 11)), route_temporal(route_history, slice(8, 11))]
                ),
            }
            for name, vector in raw.items():
                arrays[name][episode.index, h_index] = make_sketch(
                    name, vector, sketches, sketch_seed[name]
                )
            arrays["proprio"][episode.index, h_index] = physical_history(
                proprio[lo : horizon + 1]
            )
            arrays["sim"][episode.index, h_index] = physical_history(sim[lo : horizon + 1])

        print(f"extracted {position:03d}/{count} episode {episode.index}", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        **arrays,
        episode=np.asarray([episode.index for episode in episodes], dtype=np.int32),
        state=np.asarray([episode.state for episode in episodes], dtype=np.int16),
        noise_seed=np.asarray([episode.noise_seed for episode in episodes], dtype=np.int16),
        failure=np.asarray([episode.failure for episode in episodes], dtype=np.int8),
        horizons=np.asarray(HORIZONS, dtype=np.int16),
    )
    return arrays


def load_or_extract(
    run: Path, episodes: list[Episode], cache_path: Path, rebuild: bool, seed: int
) -> dict[str, np.ndarray]:
    if rebuild or not cache_path.exists():
        return extract_cache(run, episodes, cache_path, seed)
    with np.load(cache_path, allow_pickle=False) as payload:
        if tuple(payload["horizons"].tolist()) != HORIZONS:
            raise ValueError("cached horizons do not match")
        if not np.array_equal(payload["episode"], np.arange(len(episodes))):
            raise ValueError("cached episode order does not match")
        return {name: np.asarray(payload[name], dtype=np.float32) for name in BLOCK_NAMES}


def load_sim(run: Path, episodes: list[Episode]) -> list[np.ndarray]:
    output = []
    for episode in episodes:
        with np.load(episode_path(run, episode), allow_pickle=False) as payload:
            output.append(np.asarray(payload["sim_state"], dtype=np.float32))
    return output


def double_holdout_folds(
    state: np.ndarray, seed: np.ndarray, included: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, dict[str, list[int]]]]:
    state_groups = [np.asarray(group) for group in np.array_split(np.unique(state), 4)]
    seed_groups = [np.asarray(group) for group in np.array_split(np.unique(seed), 4)]
    folds = []
    for held_states in state_groups:
        for held_seeds in seed_groups:
            test = included & np.isin(state, held_states) & np.isin(seed, held_seeds)
            train = included & ~np.isin(state, held_states) & ~np.isin(seed, held_seeds)
            folds.append(
                (
                    np.flatnonzero(train),
                    np.flatnonzero(test),
                    {"states": held_states.tolist(), "seeds": held_seeds.tolist()},
                )
            )
    test_count = np.zeros(len(state), dtype=np.int8)
    for _, test, _ in folds:
        test_count[test] += 1
    if not np.all(test_count[included] == 1):
        raise ValueError("double-holdout folds do not cover every included episode once")
    return folds


def pca_block(
    train: np.ndarray, test: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    keep = scale > 1e-7
    if not np.any(keep):
        raise ValueError("constant feature block")
    train_scaled = (train[:, keep] - mean[keep]) / scale[keep]
    test_scaled = (test[:, keep] - mean[keep]) / scale[keep]
    components = min(PCA_PER_BLOCK, len(train) - 1, int(keep.sum()))
    pca = PCA(
        n_components=components,
        whiten=True,
        svd_solver="randomized",
        random_state=random_state,
    )
    return pca.fit_transform(train_scaled).astype(np.float32), pca.transform(test_scaled).astype(np.float32)


def build_fold_models(
    blocks: dict[str, np.ndarray],
    family: tuple[str, ...],
    horizon_index: int,
    folds: list[tuple[np.ndarray, np.ndarray, dict[str, list[int]]]],
    seed: int,
) -> list[FoldModel]:
    models = []
    for fold_index, (train_index, test_index, _) in enumerate(folds):
        train_parts = []
        test_parts = []
        for block_index, name in enumerate(family):
            train, test = pca_block(
                blocks[name][train_index, horizon_index],
                blocks[name][test_index, horizon_index],
                seed + 101 * fold_index + 7919 * block_index,
            )
            train_parts.append(train)
            test_parts.append(test)
        x_train = np.concatenate(train_parts, axis=1).astype(np.float64)
        x_test = np.concatenate(test_parts, axis=1).astype(np.float64)
        gram = x_train.T @ x_train
        inverse = np.linalg.solve(
            gram + RIDGE_ALPHA * np.eye(gram.shape[0]), x_train.T
        )
        models.append(
            FoldModel(
                test_index=test_index,
                train_index=train_index,
                test_map=x_test @ inverse,
                train_map=x_train @ inverse,
            )
        )
    return models


def predict(models: list[FoldModel], labels: np.ndarray) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for model in models:
        centered = labels[model.train_index].astype(np.float64)
        centered -= centered.mean()
        test_score = model.test_map @ centered
        train_score = model.train_map @ centered
        scale = max(float(train_score.std()), 1e-12)
        scores[model.test_index] = test_score / scale
    return scores


def state_auc(
    labels: np.ndarray, scores: np.ndarray, state: np.ndarray, included: np.ndarray
) -> tuple[float, float, dict[int, float]]:
    aucs = {}
    numerator = 0.0
    denominator = 0
    for value in np.unique(state[included]):
        select = included & (state == value)
        y = labels[select]
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, scores[select]))
        aucs[int(value)] = auc
        pairs = int(y.sum()) * int((1 - y).sum())
        numerator += auc * pairs
        denominator += pairs
    if denominator == 0:
        raise ValueError("no within-state positive/negative pairs")
    return numerator / denominator, float(np.mean(list(aucs.values()))), aucs


def bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    state: np.ndarray,
    included: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    informative = []
    for value in np.unique(state[included]):
        index = np.flatnonzero(included & (state == value))
        if len(np.unique(labels[index])) == 2:
            informative.append(index)
    values = []
    while len(values) < BOOTSTRAPS:
        selected = []
        for index in informative:
            selected.append(rng.choice(index, size=len(index), replace=True))
        sample = np.concatenate(selected)
        sample_included = np.zeros(len(labels), dtype=bool)
        # Duplicates matter, so evaluate directly by state rather than through a mask.
        numerator = 0.0
        denominator = 0
        for index in selected:
            y = labels[index]
            if len(np.unique(y)) < 2:
                continue
            auc = roc_auc_score(y, scores[index])
            pairs = int(y.sum()) * int((1 - y).sum())
            numerator += auc * pairs
            denominator += pairs
        if denominator:
            values.append(numerator / denominator)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def permute_within_state(
    labels: np.ndarray,
    state: np.ndarray,
    included: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = labels.copy()
    for value in np.unique(state[included]):
        index = np.flatnonzero(included & (state == value))
        output[index] = rng.permutation(output[index])
    return output


def evaluate(
    blocks: dict[str, np.ndarray],
    labels: np.ndarray,
    state: np.ndarray,
    noise_seed: np.ndarray,
    included: np.ndarray,
    permutations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[tuple[str, int], np.ndarray]]:
    folds = double_holdout_folds(state, noise_seed, included)
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}
    predictions: dict[tuple[str, int], np.ndarray] = {}
    model_cache: dict[tuple[str, int], list[FoldModel]] = {}

    for family_name, family_blocks in FAMILIES.items():
        results[family_name] = {}
        horizons = HORIZONS if family_name not in {
            "route_state",
            "route_action_1_3",
            "route_action_4_7",
            "route_action_8_10",
        } else EARLY_HORIZONS
        for horizon in horizons:
            h_index = HORIZONS.index(horizon)
            models = build_fold_models(
                blocks, family_blocks, h_index, folds, seed + 100_003 * h_index
            )
            model_cache[(family_name, horizon)] = models
            scores = predict(models, labels)
            predictions[(family_name, horizon)] = scores
            pair_auc, macro_auc, by_state = state_auc(labels, scores, state, included)
            ci = bootstrap_auc(labels, scores, state, included, rng)
            pooled = float(roc_auc_score(labels[included], scores[included]))
            results[family_name][str(horizon)] = {
                "pair_weighted_within_state_auc": pair_auc,
                "macro_state_auc": macro_auc,
                "pooled_auc": pooled,
                "bootstrap_95_ci": list(ci),
                "by_state_auc": {str(key): value for key, value in by_state.items()},
            }
            print(
                f"{family_name:22s} t{horizon:02d}: within-state AUC "
                f"{pair_auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",
                flush=True,
            )

    primary = [("moe_history", horizon) for horizon in EARLY_HORIZONS]
    observed = max(
        results[family][str(horizon)]["pair_weighted_within_state_auc"]
        for family, horizon in primary
    )
    null = np.empty(permutations, dtype=np.float64)
    raw_null = {horizon: np.empty(permutations, dtype=np.float64) for horizon in EARLY_HORIZONS}
    for permutation in range(permutations):
        shuffled = permute_within_state(labels, state, included, rng)
        values = []
        for family, horizon in primary:
            score = predict(model_cache[(family, horizon)], shuffled)
            auc, _, _ = state_auc(shuffled, score, state, included)
            raw_null[horizon][permutation] = auc
            values.append(auc)
        null[permutation] = max(values)
        if (permutation + 1) % 50 == 0:
            print(f"permutation {permutation + 1:04d}/{permutations}", flush=True)
    p_value = float((1 + np.sum(null >= observed)) / (permutations + 1))
    results["primary_permutation"] = {
        "family": "moe_history",
        "horizons": list(EARLY_HORIZONS),
        "statistic": "max pair-weighted within-state AUC",
        "observed": observed,
        "permutations": permutations,
        "maxT_p_value": p_value,
        "null_quantiles": {
            str(q): float(np.quantile(null, q)) for q in (0.5, 0.9, 0.95, 0.99)
        },
        "per_horizon_uncorrected_p": {
            str(horizon): float(
                (1 + np.sum(raw_null[horizon] >= results["moe_history"][str(horizon)]["pair_weighted_within_state_auc"]))
                / (permutations + 1)
            )
            for horizon in EARLY_HORIZONS
        },
    }
    results["folds"] = [metadata for _, _, metadata in folds]
    return results, predictions


def paired_bootstrap_difference(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    state: np.ndarray,
    included: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = state_auc(labels, left, state, included)[0] - state_auc(labels, right, state, included)[0]
    informative = []
    for value in np.unique(state[included]):
        index = np.flatnonzero(included & (state == value))
        if len(np.unique(labels[index])) == 2:
            informative.append(index)
    values = []
    while len(values) < BOOTSTRAPS:
        numerator_left = numerator_right = 0.0
        denominator = 0
        for index in informative:
            sample = rng.choice(index, size=len(index), replace=True)
            y = labels[sample]
            if len(np.unique(y)) < 2:
                continue
            pairs = int(y.sum()) * int((1 - y).sum())
            numerator_left += roc_auc_score(y, left[sample]) * pairs
            numerator_right += roc_auc_score(y, right[sample]) * pairs
            denominator += pairs
        if denominator:
            values.append((numerator_left - numerator_right) / denominator)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(point), float(low), float(high)


def render_plot(
    output: Path,
    evaluation: dict[str, Any],
    onset: np.ndarray,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = {
        "route_history": "#2878b5",
        "hidden_history": "#cf4b32",
        "moe_history": "#1b7f5b",
        "proprio_history": "#7d5ba6",
        "sim_history": "#555555",
    }
    for family, color in colors.items():
        values = [evaluation[family][str(h)]["pair_weighted_within_state_auc"] for h in HORIZONS]
        low = [evaluation[family][str(h)]["bootstrap_95_ci"][0] for h in HORIZONS]
        high = [evaluation[family][str(h)]["bootstrap_95_ci"][1] for h in HORIZONS]
        axes[0].plot(HORIZONS, values, marker="o", label=family, color=color)
        axes[0].fill_between(HORIZONS, low, high, color=color, alpha=0.10)
    axes[0].axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    axes[0].set(title="Stasis-trap prediction", xlabel="control step", ylabel="within-state AUC", ylim=(0.35, 1.0))
    axes[0].legend(fontsize=8, loc="lower right")

    role_names = ("route_state", "route_action_1_3", "route_action_4_7", "route_action_8_10")
    x = np.arange(len(role_names))
    width = 0.36
    for offset, horizon, color in ((-width / 2, 7, "#4c78a8"), (width / 2, 12, "#f58518")):
        values = [evaluation[name][str(horizon)]["pair_weighted_within_state_auc"] for name in role_names]
        axes[1].bar(x + offset, values, width, label=f"t{horizon}", color=color)
    axes[1].axhline(0.5, color="#999999", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, ("state", "a1-3", "a4-7", "a8-10"))
    axes[1].set(title="Route token-role ablation", ylabel="within-state AUC", ylim=(0.35, 0.8))
    axes[1].legend()

    values = onset[onset >= 0]
    axes[2].hist(values, bins=np.arange(values.min() - 0.5, values.max() + 1.5), color="#3b8c6e")
    axes[2].axvline(np.median(values), color="#222222", linestyle="--", label=f"median t{int(np.median(values))}")
    for horizon in EARLY_HORIZONS:
        axes[2].axvline(horizon, color="#bd3c3c", alpha=0.75)
    axes[2].set(title="Behavior-defined trap onset", xlabel="control step", ylabel="failure rollouts")
    axes[2].legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_report(
    output: Path,
    target: dict[str, Any],
    evaluation: dict[str, Any],
    increments: dict[str, Any],
) -> None:
    early = evaluation["primary_permutation"]
    lines = [
        "# 结构化 MoE 早期陷入信号实验",
        "",
        "## 设计",
        "",
        f"- 数据：scene8 同任务 16 个初始状态 × 32 个 flow-noise seed，共 512 条轨迹。",
        f"- 行为标签：成功终点集合的留出最近邻距离；若历史最佳距离仍大于 {GOAL_RADIUS_M:.2f} m，且未来再不能改善 {PROGRESS_EPS_M:.2f} m，则记为停滞型陷入起点。",
        f"- 覆盖：{target['n_stasis_trap']}/{target['n_failure']} 条失败为停滞型陷入；{target['n_ambiguous_failure_excluded']} 条非同类失败从主检验排除。",
        f"- 陷入点：中位 t{target['onset_quantiles']['0.5']:.0f}，10%–90% 为 t{target['onset_quantiles']['0.1']:.0f}–t{target['onset_quantiles']['0.9']:.0f}。因此 t7/t12 均在所有主标签陷入点之前。",
        "- MoE 输入：HB route identity、token 几何、跨控制步 cosine/JA/WJ、速度方向动量；HB hidden identity、token Gram/effective-rank、跨控制步动量。历史窗为 7，绝不读取当前截面之后的数据。",
        "- 验证：4×4 init-state/seed 双重留出；主指标在同一 init state 内形成正负样本对，再做 pair-weighted AUC。",
        "",
        "## 主结果",
        "",
        "| 特征 | t7 | t12 | t20 | t27 | t34 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for family in ("route_history", "hidden_history", "moe_current", "moe_history", "proprio_history", "sim_history", "sim_plus_moe"):
        values = []
        for horizon in HORIZONS:
            row = evaluation[family][str(horizon)]
            values.append(
                f"{row['pair_weighted_within_state_auc']:.3f} [{row['bootstrap_95_ci'][0]:.3f}, {row['bootstrap_95_ci'][1]:.3f}]"
            )
        lines.append(f"| {family} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            f"预先指定的早期检验是 `moe_history` 在 t7/t12 的最大 AUC；within-state 重排 {early['permutations']} 次后的 maxT p={early['maxT_p_value']:.4f}。",
            "",
            "## Token 位置消融",
            "",
            "| route token 组 | t7 | t12 |",
            "|---|---:|---:|",
        ]
    )
    for family in ("route_state", "route_action_1_3", "route_action_4_7", "route_action_8_10"):
        lines.append(
            f"| {family} | {evaluation[family]['7']['pair_weighted_within_state_auc']:.3f} | {evaluation[family]['12']['pair_weighted_within_state_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## MoE 对物理状态的增量",
            "",
            "| control step | AUC(sim+MoE) − AUC(sim) | paired bootstrap 95% CI |",
            "|---:|---:|---:|",
        ]
    )
    for horizon in HORIZONS:
        row = increments[str(horizon)]
        lines.append(
            f"| t{horizon} | {row['difference']:+.3f} | [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}] |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这是离线预测/表征实验，不是 MoE 因果干预。",
            "- `hb_hidden` 是进入 HB expert MLP 前的 contextualized token state；缓存没有 expert output/contribution，因此本实验不能声称专家贡献本身预示陷入。",
            "- t20 及以后对多数停滞失败已接近或越过行为陷入点，只能作为阶段/阳性对照；真正的 precursor 结论只看 t7/t12。",
            "- 多个特征族和 token 消融属于探索性读数；正式显著性只给预先指定的 t7/t12 `moe_history` maxT 检验。",
            "",
        ]
    )
    output.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes(args.run)
    sim = load_sim(args.run, episodes)
    labels, included, onset, target = build_targets(episodes, sim)
    print(json.dumps(target, indent=2), flush=True)
    blocks = load_or_extract(args.run, episodes, args.cache, args.rebuild_cache, args.seed)
    state = np.asarray([episode.state for episode in episodes], dtype=np.int16)
    noise_seed = np.asarray([episode.noise_seed for episode in episodes], dtype=np.int16)
    evaluation, predictions = evaluate(
        blocks, labels, state, noise_seed, included, args.permutations, args.seed
    )

    rng = np.random.default_rng(args.seed + 77)
    increments = {}
    for horizon in HORIZONS:
        difference, low, high = paired_bootstrap_difference(
            labels,
            predictions[("sim_plus_moe", horizon)],
            predictions[("sim_history", horizon)],
            state,
            included,
            rng,
        )
        increments[str(horizon)] = {"difference": difference, "ci": [low, high]}

    summary = {
        "task": TASK,
        "horizons": list(HORIZONS),
        "history": HISTORY,
        "target": target,
        "evaluation": evaluation,
        "sim_moe_increment": increments,
        "method": {
            "projection": f"data-independent CountSketch to {PROJECTED_DIM} dimensions per block",
            "fold_transform": f"train-only scaling + whitened PCA({PCA_PER_BLOCK}) per block",
            "predictor": f"linear ridge alpha={RIDGE_ALPHA}",
            "cross_validation": "16 state-group x seed-group double-holdout intersections",
            "metric": "pair-weighted within-initial-state AUC",
        },
    }
    (args.output_dir / "early_structure_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        args.output_dir / "early_structure_scores.npz",
        label=labels,
        included=included,
        onset=onset,
        state=state,
        noise_seed=noise_seed,
        **{
            f"score__{family}__t{horizon}": score
            for (family, horizon), score in predictions.items()
        },
    )
    render_plot(args.output_dir / "early_structure.png", evaluation, onset)
    write_report(args.output_dir / "early_structure_report.md", target, evaluation, increments)
    print(f"wrote {args.output_dir / 'early_structure_report.md'}", flush=True)


if __name__ == "__main__":
    main()
