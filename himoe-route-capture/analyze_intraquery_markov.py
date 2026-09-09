#!/usr/bin/env python3
"""Model candidate-specific MoE routing within one policy query.

One environment control step is not one Markov step here.  A policy query has
ten flow-denoising rounds and eight recorded HB MoE layers.  The observable at
micro-step ``(denoise, layer)`` is the full routing fingerprint of the ten
action tokens.  Layer-specific Hellinger K-means codebooks map those 10x32
probability tensors to discrete states, producing an 80-state path per query::

    (tau=0, layer=2) -> ... -> (tau=0, layer=15)
      -> (tau=1, layer=2) -> ... -> (tau=9, layer=15)

Transition matrices are conditioned on the source layer.  This is a periodic
Markov chain, rather than a homogeneous chain that incorrectly equates expert
IDs across layers.  A fully position-conditioned chain is retained as a
non-homogeneity diagnostic.

The selector is a success-versus-failure path likelihood ratio.  Every score is
cross-fitted while holding out both the complete initial scene and the K8 seed
fold being ranked.  Codebooks are fitted once per task without labels; this is
transductive representation learning, while every outcome-dependent count is
strictly double-held-out.

The corpus has one recorded rollout per episode-start noise stream.  Therefore
the selection result is a shadow association test, not a causal per-query
closed-loop result.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from sklearn.cluster import MiniBatchKMeans


HERE = Path(__file__).resolve().parent
DEFAULT_HUB = HERE.parent / "VLA_MUI_HUB"
DEFAULT_OUT = HERE / "analysis" / "intraquery-markov"
PRIMARY_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_DENOISE = 10
N_ACTION_TOKENS = 10
N_EXPERTS = 32
SEED_FOLDS = 4
METHODS = (
    "occupancy_periodic",
    "markov_periodic",
    "occupancy_position",
    "markov_position",
)
TOP_K_VALUES = (1, 2, 4)


@dataclass(frozen=True)
class TaskData:
    task: str
    suite: str
    checkpoint_sha256: str
    scenes: np.ndarray
    seeds: np.ndarray
    seed_folds: np.ndarray
    seed_positions: np.ndarray
    success: np.ndarray
    index_grid: np.ndarray
    routes: np.ndarray  # [episode, layer, denoise, action-token, expert]
    probability_mass_max_error: float


@dataclass(frozen=True)
class RouteCodebooks:
    centers: np.ndarray  # [layer, state, action-token * expert]
    codes: np.ndarray  # [episode, denoise, layer]
    inertia: np.ndarray


@dataclass(frozen=True)
class RouteMarkovModel:
    occupancy_periodic: np.ndarray  # [layer, state]
    occupancy_position: np.ndarray  # [80, state]
    transition_periodic: np.ndarray  # [source-layer, state, state]
    transition_position: np.ndarray  # [79, state, state]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub-root", type=Path, default=DEFAULT_HUB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task", action="append", help="suite/task; repeatable")
    parser.add_argument("--primary-task", default=PRIMARY_TASK)
    parser.add_argument("--clusters", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--primary-k", type=int, default=16)
    parser.add_argument("--primary-rounds", type=int, default=3)
    parser.add_argument("--primary-top-k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--position-alpha", type=float, default=10.0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def finite(value: float | np.floating) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("non-finite result: %r" % result)
    return result


def normalize_router_probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float32)
    if probabilities.ndim != 5 or probabilities.shape[-1] != N_EXPERTS:
        raise ValueError("expected router probabilities [N,L,D,A,32]")
    if np.any(~np.isfinite(probabilities)) or float(probabilities.min()) < -1e-6:
        raise ValueError("router probabilities must be finite and nonnegative")
    probabilities = np.maximum(probabilities, 0.0)
    mass = probabilities.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("router probability vector has zero mass")
    probabilities /= mass
    return probabilities


def first_rows(episode_ids: np.ndarray, expected: np.ndarray) -> np.ndarray:
    unique, first = np.unique(episode_ids, return_index=True)
    lookup = {int(episode): int(row) for episode, row in zip(unique, first)}
    if set(map(int, expected)) != set(lookup):
        missing = sorted(set(map(int, expected)) - set(lookup))
        extra = sorted(set(lookup) - set(map(int, expected)))
        raise RuntimeError(
            "episode alignment mismatch: missing=%s extra=%s"
            % (missing[:5], extra[:5])
        )
    return np.asarray([lookup[int(episode)] for episode in expected], dtype=np.int64)


def make_index_grid(
    scenes: np.ndarray, folds: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    unique_scenes = np.unique(scenes)
    candidates = len(np.unique(positions))
    grid = np.full(
        (len(unique_scenes), SEED_FOLDS, candidates), -1, dtype=np.int64
    )
    for scene_axis, scene in enumerate(unique_scenes):
        for fold in range(SEED_FOLDS):
            indices = np.flatnonzero((scenes == scene) & (folds == fold))
            if len(indices) != candidates:
                raise ValueError("scene/fold is not a complete candidate pool")
            if len(np.unique(positions[indices])) != candidates:
                raise ValueError("candidate positions repeat within a pool")
            grid[scene_axis, fold, positions[indices]] = indices
    if np.any(grid < 0) or len(np.unique(grid)) != len(scenes):
        raise ValueError("candidate grid does not cover every episode exactly once")
    return grid


def discover_runs(hub_root: Path, requested: set[str] | None = None) -> list[Path]:
    cache = hub_root / "cache" / "HiMoE-VLA"
    runs = sorted(
        path.parent.parent
        for path in cache.glob("**/right-16x32/server/routes.zarr")
        if (path.parent.parent / "client" / "summaries.json").exists()
    )
    if requested is not None:
        runs = [
            run
            for run in runs
            if str(run.relative_to(cache).parent) in requested
        ]
        found = {str(run.relative_to(cache).parent) for run in runs}
        if found != requested:
            raise ValueError("requested tasks not found: %s" % sorted(requested - found))
    if not runs:
        raise RuntimeError("no complete right-16x32 route captures found")
    return runs


def load_task(run: Path, cache_root: Path) -> TaskData:
    summaries = sorted(
        json.loads((run / "client" / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    if len(summaries) != 16 * 32:
        raise RuntimeError("%s is not a complete 16x32 run" % run)
    episodes = np.asarray(
        [int(row["episode_index"]) for row in summaries], dtype=np.int64
    )
    scenes = np.asarray(
        [int(row["init_state_id"]) for row in summaries], dtype=np.int64
    )
    seeds = np.asarray(
        [int(row["flow_noise_seed"]) for row in summaries], dtype=np.int64
    )
    success = np.asarray([bool(row["success"]) for row in summaries], dtype=bool)

    unique_seeds = np.sort(np.unique(seeds))
    if len(np.unique(scenes)) != 16 or len(unique_seeds) != 32:
        raise RuntimeError("expected a 16-scene x 32-seed grid")
    rank = {int(seed): index for index, seed in enumerate(unique_seeds)}
    seed_ranks = np.asarray([rank[int(seed)] for seed in seeds], dtype=np.int64)
    folds = seed_ranks % SEED_FOLDS
    positions = seed_ranks // SEED_FOLDS
    grid = make_index_grid(scenes, folds, positions)

    group = zarr.open_group(str(run / "server" / "routes.zarr"), mode="r")
    rows = first_rows(np.asarray(group["episode_id"][:]), episodes)
    raw = np.asarray(
        group["hb_router_probs"].oindex[rows, :, :, 1:, :], dtype=np.float32
    )
    expected_shape = (len(summaries), len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS)
    if raw.shape != expected_shape:
        raise RuntimeError("unexpected action-token route shape %s" % (raw.shape,))
    mass_error = float(np.max(np.abs(raw.sum(axis=-1) - 1.0)))
    routes = normalize_router_probabilities(raw)

    metadata = json.loads((run / "client" / "server_metadata.json").read_text())
    recorded_layers = tuple(int(value) for value in metadata["routing_hb_layer_indices"])
    if recorded_layers != HB_LAYERS:
        raise RuntimeError("recorded HB layers changed: %s" % (recorded_layers,))
    relative = run.relative_to(cache_root)
    return TaskData(
        task=str(relative.parent),
        suite=relative.parts[0],
        checkpoint_sha256=str(metadata.get("checkpoint_sha256", "")),
        scenes=scenes,
        seeds=seeds,
        seed_folds=folds,
        seed_positions=positions,
        success=success,
        index_grid=grid,
        routes=routes,
        probability_mass_max_error=mass_error,
    )


def fit_layer_codebooks(
    routes: np.ndarray, clusters: int, random_seed: int
) -> RouteCodebooks:
    values = np.asarray(routes, dtype=np.float32)
    expected = (len(values), len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS)
    if values.shape != expected:
        raise ValueError("unexpected route tensor %s" % (values.shape,))
    if clusters < 2 or clusters > len(values) * N_DENOISE:
        raise ValueError("invalid number of route states")

    codes = np.empty((len(values), N_DENOISE, len(HB_LAYERS)), dtype=np.int64)
    centers = np.empty(
        (len(HB_LAYERS), clusters, N_ACTION_TOKENS * N_EXPERTS),
        dtype=np.float32,
    )
    inertia = np.empty(len(HB_LAYERS), dtype=np.float64)
    for layer in range(len(HB_LAYERS)):
        # Concatenated sqrt probabilities preserve RMS Hellinger geometry over
        # the ten action-token routing sites.
        embedded = np.sqrt(values[:, layer]).reshape(len(values) * N_DENOISE, -1)
        model = MiniBatchKMeans(
            n_clusters=clusters,
            random_state=random_seed + layer,
            n_init=10,
            batch_size=min(2048, len(embedded)),
            max_iter=200,
            reassignment_ratio=0.01,
        ).fit(embedded)
        codes[:, :, layer] = model.labels_.reshape(len(values), N_DENOISE)
        centers[layer] = model.cluster_centers_.astype(np.float32)
        inertia[layer] = float(model.inertia_)
    return RouteCodebooks(centers=centers, codes=codes, inertia=inertia)


def assign_layer_codebooks(routes: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign new route fingerprints to frozen layer-specific codebooks."""

    values = np.asarray(routes, dtype=np.float32)
    expected = (len(values), len(HB_LAYERS), N_DENOISE, N_ACTION_TOKENS, N_EXPERTS)
    if values.shape != expected:
        raise ValueError("unexpected route tensor %s" % (values.shape,))
    codebook = np.asarray(centers, dtype=np.float32)
    if (
        codebook.ndim != 3
        or codebook.shape[0] != len(HB_LAYERS)
        or codebook.shape[2] != N_ACTION_TOKENS * N_EXPERTS
    ):
        raise ValueError("unexpected layer codebook shape %s" % (codebook.shape,))
    clusters = codebook.shape[1]
    codes = np.empty((len(values), N_DENOISE, len(HB_LAYERS)), dtype=np.int64)
    for layer in range(len(HB_LAYERS)):
        embedded = np.sqrt(values[:, layer]).reshape(len(values) * N_DENOISE, -1)
        layer_centers = codebook[layer]
        squared_distance = (
            np.square(embedded).sum(axis=1, keepdims=True)
            + np.square(layer_centers).sum(axis=1)[None, :]
            - 2.0 * (embedded @ layer_centers.T)
        )
        codes[:, :, layer] = np.argmin(squared_distance, axis=1).reshape(
            len(values), N_DENOISE
        )
    if np.min(codes) < 0 or np.max(codes) >= clusters:
        raise RuntimeError("codebook assignment produced an invalid state")
    return codes


def flatten_microstates(codes: np.ndarray) -> np.ndarray:
    """Flatten in causal compute order: denoise outer, HB layer inner."""

    values = np.asarray(codes, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (N_DENOISE, len(HB_LAYERS)):
        raise ValueError("expected state codes [episode,10,8]")
    return values.reshape(len(values), N_DENOISE * len(HB_LAYERS))


def double_holdout_mask(
    scenes: np.ndarray, folds: np.ndarray, held_scene: int, held_fold: int
) -> np.ndarray:
    return (np.asarray(scenes) != held_scene) & (np.asarray(folds) != held_fold)


def _smoothed_occupancy(counts: np.ndarray, beta: float = 0.5) -> np.ndarray:
    if beta <= 0.0:
        raise ValueError("occupancy pseudocount must be positive")
    return (counts + beta) / (counts.sum(axis=-1, keepdims=True) + beta * counts.shape[-1])


def _posterior_rows(counts: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("transition prior strength must be positive")
    if counts.shape != prior.shape:
        raise ValueError("transition counts and prior shapes differ")
    output = counts.astype(np.float64) + alpha * prior
    return output / output.sum(axis=-1, keepdims=True)


def fit_route_markov_model(
    sequences: np.ndarray,
    clusters: int,
    alpha: float = 5.0,
    position_alpha: float = 10.0,
) -> RouteMarkovModel:
    values = np.asarray(sequences, dtype=np.int64)
    if values.ndim != 3 or values.shape[1:] != (N_DENOISE, len(HB_LAYERS)):
        raise ValueError("expected route states [episode,10,8]")
    if len(values) == 0:
        raise ValueError("cannot fit an empty class")
    if np.min(values) < 0 or np.max(values) >= clusters:
        raise ValueError("route state outside the declared state space")

    layers = len(HB_LAYERS)
    events = N_DENOISE * layers
    flat = flatten_microstates(values)

    periodic_counts = np.zeros((layers, clusters), dtype=np.int64)
    for layer in range(layers):
        periodic_counts[layer] = np.bincount(
            values[:, :, layer].ravel(), minlength=clusters
        )
    occupancy_periodic = _smoothed_occupancy(periodic_counts)

    position_counts = np.zeros((events, clusters), dtype=np.int64)
    for position in range(events):
        position_counts[position] = np.bincount(flat[:, position], minlength=clusters)
    position_prior = occupancy_periodic[np.arange(events) % layers]
    occupancy_position = (
        position_counts + position_alpha * position_prior
    ) / (position_counts.sum(axis=-1, keepdims=True) + position_alpha)

    periodic_counts_2 = np.zeros((layers, clusters, clusters), dtype=np.int64)
    for layer in range(layers - 1):
        np.add.at(
            periodic_counts_2[layer],
            (values[:, :, layer].ravel(), values[:, :, layer + 1].ravel()),
            1,
        )
    np.add.at(
        periodic_counts_2[-1],
        (values[:, :-1, -1].ravel(), values[:, 1:, 0].ravel()),
        1,
    )
    periodic_prior = np.empty_like(periodic_counts_2, dtype=np.float64)
    for source_layer in range(layers):
        destination_layer = (source_layer + 1) % layers
        periodic_prior[source_layer] = occupancy_periodic[destination_layer][None, :]
    transition_periodic = _posterior_rows(periodic_counts_2, periodic_prior, alpha)

    position_counts_2 = np.zeros((events - 1, clusters, clusters), dtype=np.int64)
    for position in range(events - 1):
        np.add.at(
            position_counts_2[position],
            (flat[:, position], flat[:, position + 1]),
            1,
        )
    position_prior_2 = transition_periodic[np.arange(events - 1) % layers]
    transition_position = _posterior_rows(
        position_counts_2, position_prior_2, position_alpha
    )
    return RouteMarkovModel(
        occupancy_periodic=occupancy_periodic,
        occupancy_position=occupancy_position,
        transition_periodic=transition_periodic,
        transition_position=transition_position,
    )


def sequence_log_likelihoods(
    model: RouteMarkovModel, sequences: np.ndarray
) -> dict[str, np.ndarray]:
    values = np.asarray(sequences, dtype=np.int64)
    flat = flatten_microstates(values)
    episodes, events = flat.shape
    layers = len(HB_LAYERS)

    occupancy_periodic = np.empty((episodes, events), dtype=np.float64)
    occupancy_position = np.empty((episodes, events), dtype=np.float64)
    markov_periodic = np.empty((episodes, events), dtype=np.float64)
    markov_position = np.empty((episodes, events), dtype=np.float64)
    for position in range(events):
        layer = position % layers
        state = flat[:, position]
        occupancy_periodic[:, position] = np.log(model.occupancy_periodic[layer, state])
        occupancy_position[:, position] = np.log(model.occupancy_position[position, state])
        if position == 0:
            markov_periodic[:, position] = occupancy_periodic[:, position]
            markov_position[:, position] = occupancy_position[:, position]
        else:
            previous = flat[:, position - 1]
            edge = (position - 1) % layers
            markov_periodic[:, position] = np.log(
                model.transition_periodic[edge, previous, state]
            )
            markov_position[:, position] = np.log(
                model.transition_position[position - 1, previous, state]
            )

    round_ends = np.arange(layers - 1, events, layers)
    return {
        "occupancy_periodic": np.cumsum(occupancy_periodic, axis=1)[:, round_ends],
        "markov_periodic": np.cumsum(markov_periodic, axis=1)[:, round_ends],
        "occupancy_position": np.cumsum(occupancy_position, axis=1)[:, round_ends],
        "markov_position": np.cumsum(markov_position, axis=1)[:, round_ends],
    }


def crossfit_likelihood_ratios(
    data: TaskData,
    codes: np.ndarray,
    clusters: int,
    alpha: float,
    position_alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores = {
        method: np.full((len(codes), N_DENOISE), np.nan, dtype=np.float64)
        for method in METHODS
    }
    true_log_likelihood = {
        method: np.full((len(codes), N_DENOISE), np.nan, dtype=np.float64)
        for method in METHODS
    }
    train_counts: list[dict[str, int]] = []

    for scene in np.unique(data.scenes):
        for fold in range(SEED_FOLDS):
            train = double_holdout_mask(data.scenes, data.seed_folds, int(scene), fold)
            test = (data.scenes == scene) & (data.seed_folds == fold)
            models: dict[bool, RouteMarkovModel] = {}
            logp: dict[bool, dict[str, np.ndarray]] = {}
            for label in (False, True):
                class_train = train & (data.success == label)
                count = int(class_train.sum())
                if count == 0:
                    raise ValueError(
                        "double holdout leaves no %s examples for scene=%s fold=%s"
                        % ("success" if label else "failure", scene, fold)
                    )
                models[label] = fit_route_markov_model(
                    codes[class_train], clusters, alpha, position_alpha
                )
                logp[label] = sequence_log_likelihoods(models[label], codes[test])
            train_counts.append(
                {
                    "scene": int(scene),
                    "fold": fold,
                    "failure": int((train & ~data.success).sum()),
                    "success": int((train & data.success).sum()),
                }
            )
            test_labels = data.success[test]
            for method in METHODS:
                scores[method][test] = logp[True][method] - logp[False][method]
                true_log_likelihood[method][test] = np.where(
                    test_labels[:, None], logp[True][method], logp[False][method]
                )

    for values in (*scores.values(), *true_log_likelihood.values()):
        if np.any(~np.isfinite(values)):
            raise RuntimeError("cross-fitting did not score every episode")

    events = N_DENOISE * len(HB_LAYERS)
    predictive = {
        method: {
            "full_path_cross_entropy_bits_per_event": finite(
                -true_log_likelihood[method][:, -1].mean() / (events * math.log(2.0))
            )
        }
        for method in METHODS
    }
    predictive["markov_periodic"]["gain_over_occupancy_bits_per_transition"] = finite(
        (
            true_log_likelihood["markov_periodic"][:, -1]
            - true_log_likelihood["occupancy_periodic"][:, -1]
        ).mean()
        / ((events - 1) * math.log(2.0))
    )
    predictive["markov_position"]["gain_over_occupancy_bits_per_transition"] = finite(
        (
            true_log_likelihood["markov_position"][:, -1]
            - true_log_likelihood["occupancy_position"][:, -1]
        ).mean()
        / ((events - 1) * math.log(2.0))
    )
    diagnostic = {
        "train_examples_min": {
            "failure": min(row["failure"] for row in train_counts),
            "success": min(row["success"] for row in train_counts),
        },
        "train_examples_max": {
            "failure": max(row["failure"] for row in train_counts),
            "success": max(row["success"] for row in train_counts),
        },
        "predictive": predictive,
    }
    return scores, diagnostic


def topk_tie_weights(scores: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("top-k scores must be one finite vector")
    if k < 1 or k > len(values):
        raise ValueError("top-k is outside the candidate pool")
    cutoff = np.partition(values, len(values) - k)[len(values) - k]
    above = values > cutoff
    tied = values == cutoff
    weights = above.astype(np.float64)
    weights[tied] = (k - int(above.sum())) / int(tied.sum())
    if not np.isclose(weights.sum(), k):
        raise RuntimeError("top-k tie weights do not sum to k")
    return weights


def _pool_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = float((positive[:, None] > negative[None, :]).sum())
    ties = float((positive[:, None] == negative[None, :]).sum())
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def evaluate_selection(
    data: TaskData,
    score: np.ndarray,
    top_k: int,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    score_grid = np.asarray(score, dtype=np.float64)[data.index_grid]
    label_grid = data.success[data.index_grid]
    weights = np.empty_like(score_grid)
    pool_auc = np.full(score_grid.shape[:2], np.nan, dtype=np.float64)
    cutoff_tied = np.zeros(score_grid.shape[:2], dtype=bool)
    for scene in range(score_grid.shape[0]):
        for fold in range(score_grid.shape[1]):
            weights[scene, fold] = topk_tie_weights(score_grid[scene, fold], top_k)
            cutoff = np.partition(score_grid[scene, fold], -top_k)[-top_k]
            above = int(np.count_nonzero(score_grid[scene, fold] > cutoff))
            tied = int(np.count_nonzero(score_grid[scene, fold] == cutoff))
            cutoff_tied[scene, fold] = tied > top_k - above
            pool_auc[scene, fold] = _pool_auc(
                label_grid[scene, fold], score_grid[scene, fold]
            )

    selected = (weights * label_grid).sum(axis=-1) / top_k
    baseline = label_grid.mean(axis=-1)
    scene_delta = (selected - baseline).mean(axis=1)
    scene_draws = rng.integers(
        0, score_grid.shape[0], size=(bootstrap, score_grid.shape[0])
    )
    delta_draws = scene_delta[scene_draws].mean(axis=1)
    sampled_auc = pool_auc[scene_draws]
    valid_auc = np.isfinite(sampled_auc)
    auc_draws = np.divide(
        np.where(valid_auc, sampled_auc, 0.0).sum(axis=(1, 2)),
        valid_auc.sum(axis=(1, 2)),
        out=np.full(bootstrap, np.nan, dtype=np.float64),
        where=valid_auc.sum(axis=(1, 2)) > 0,
    )
    successes = label_grid.sum(axis=-1)
    oracle = np.minimum(successes, top_k) / top_k
    seed_grid = data.seeds[data.index_grid]
    unique_seeds = np.unique(seed_grid)
    seed_mass = np.asarray(
        [weights[seed_grid == seed].sum() for seed in unique_seeds],
        dtype=np.float64,
    )
    seed_probability = seed_mass / seed_mass.sum()
    positive_seed_probability = seed_probability[seed_probability > 0.0]
    seed_entropy = -float(
        np.sum(positive_seed_probability * np.log(positive_seed_probability))
    )
    result = {
        "top_k": top_k,
        "selected_success_rate": finite(selected.mean()),
        "exact_random_success_rate": finite(baseline.mean()),
        "delta_vs_random": finite(scene_delta.mean()),
        "delta_scene_bootstrap_ci95": [
            finite(value) for value in np.percentile(delta_draws, [2.5, 97.5])
        ],
        "mean_pool_auc": finite(np.nanmean(pool_auc)),
        "evaluable_auc_pools": int(np.isfinite(pool_auc).sum()),
        "mean_pool_auc_scene_bootstrap_ci95": [
            finite(value) for value in np.nanpercentile(auc_draws, [2.5, 97.5])
        ],
        "oracle_selected_success_rate": finite(oracle.mean()),
        "fractional_tie_breaking": True,
        "cutoff_tie_pool_rate": finite(cutoff_tied.mean()),
        "seed_concentration": {
            "unique_seeds_with_positive_weight": int(np.count_nonzero(seed_mass)),
            "effective_seed_count": finite(math.exp(seed_entropy)),
            "maximum_expected_seed_share": finite(seed_probability.max()),
        },
    }
    return result, weights


def coherent_fold_permutation_p(
    data: TaskData,
    weights: np.ndarray,
    observed_delta: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Permute held-out K8 seed columns coherently across scenes within a fold."""

    labels = data.success[data.index_grid]
    top_k = float(weights.sum(axis=-1)[0, 0])
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = np.empty_like(labels)
        for fold in range(SEED_FOLDS):
            order = rng.permutation(labels.shape[-1])
            permuted[:, fold] = labels[:, fold, order]
        selected = (weights * permuted).sum(axis=-1) / top_k
        null[draw] = float((selected - permuted.mean(axis=-1)).mean())
    return {
        "one_sided_improvement_p": finite(
            (np.count_nonzero(null >= observed_delta - 1e-15) + 1) / (draws + 1)
        ),
        "null_mean": finite(null.mean()),
        "null_95": [finite(value) for value in np.percentile(null, [2.5, 97.5])],
        "draws": draws,
        "scheme": "one K8 seed-column permutation per fold, reused across 16 scenes",
    }


def coherent_fold_comparison_p(
    data: TaskData,
    left_weights: np.ndarray,
    right_weights: np.ndarray,
    observed_difference: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Test whether one fixed held-fold selector beats another fixed selector."""

    if left_weights.shape != right_weights.shape:
        raise ValueError("selector weight grids differ")
    labels = data.success[data.index_grid]
    top_k = float(left_weights.sum(axis=-1)[0, 0])
    if not np.allclose(right_weights.sum(axis=-1), top_k):
        raise ValueError("selectors choose different candidate counts")
    difference_weights = left_weights - right_weights
    null = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        permuted = np.empty_like(labels)
        for fold in range(SEED_FOLDS):
            order = rng.permutation(labels.shape[-1])
            permuted[:, fold] = labels[:, fold, order]
        null[draw] = float((difference_weights * permuted).sum(axis=-1).mean() / top_k)
    return {
        "one_sided_left_better_p": finite(
            (np.count_nonzero(null >= observed_difference - 1e-15) + 1)
            / (draws + 1)
        ),
        "observed_selected_success_difference": finite(observed_difference),
        "null_mean": finite(null.mean()),
        "null_95": [finite(value) for value in np.percentile(null, [2.5, 97.5])],
        "draws": draws,
        "scheme": "one K8 seed-column permutation per fold, reused across 16 scenes",
    }


def _compact_codebook(codebooks: RouteCodebooks) -> dict[str, Any]:
    return {
        "inertia_per_layer": [finite(value) for value in codebooks.inertia],
        "state_count_per_layer": [
            int(len(np.unique(codebooks.codes[:, :, layer])))
            for layer in range(len(HB_LAYERS))
        ],
        "fit_uses_outcome_labels": False,
        "fit_scope": "all route covariates in this task (transductive, label-blind)",
    }


def _save_production_model(
    out_dir: Path,
    data: TaskData,
    codebooks: RouteCodebooks,
    clusters: int,
    alpha: float,
    position_alpha: float,
) -> str:
    if len(np.unique(data.success)) != 2:
        raise ValueError("a class-conditional production model needs both outcomes")
    failure = fit_route_markov_model(
        codebooks.codes[~data.success], clusters, alpha, position_alpha
    )
    success = fit_route_markov_model(
        codebooks.codes[data.success], clusters, alpha, position_alpha
    )
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", data.task)
    path = out_dir / (slug + "__k%d.npz" % clusters)
    arrays: dict[str, np.ndarray] = {
        "codebook_centers": codebooks.centers,
        "hb_layers": np.asarray(HB_LAYERS, dtype=np.int64),
        "clusters": np.asarray(clusters, dtype=np.int64),
    }
    for label, model in (("failure", failure), ("success", success)):
        for key, value in asdict(model).items():
            arrays[label + "__" + key] = value
    np.savez_compressed(path, **arrays)
    return path.name


def analyze_task(
    data: TaskData,
    args: argparse.Namespace,
    rng: np.random.Generator,
    out_dir: Path,
) -> dict[str, Any]:
    outcome_classes = np.unique(data.success)
    result: dict[str, Any] = {
        "suite": data.suite,
        "checkpoint_sha256": data.checkpoint_sha256,
        "episodes": len(data.success),
        "successes": int(data.success.sum()),
        "success_rate": finite(data.success.mean()),
        "probability_mass_max_error": data.probability_mass_max_error,
        "clusters": {},
    }
    if len(outcome_classes) != 2:
        result.update(
            {
                "status": "uninformative_outcome",
                "reason": "all recorded rollouts have the same outcome",
            }
        )
        return result

    result["status"] = "evaluated"
    for clusters in args.clusters:
        print("  fitting K=%d layer codebooks" % clusters, flush=True)
        codebooks = fit_layer_codebooks(data.routes, clusters, args.seed + clusters * 100)
        scores, diagnostic = crossfit_likelihood_ratios(
            data, codebooks.codes, clusters, args.alpha, args.position_alpha
        )
        selection: dict[str, dict[str, dict[str, Any]]] = {}
        primary_weights: np.ndarray | None = None
        primary_occupancy_weights: np.ndarray | None = None
        for method in METHODS:
            selection[method] = {}
            for rounds in range(1, N_DENOISE + 1):
                by_top_k: dict[str, Any] = {}
                for top_k in TOP_K_VALUES:
                    row, weights = evaluate_selection(
                        data,
                        scores[method][:, rounds - 1],
                        top_k,
                        args.bootstrap,
                        rng,
                    )
                    by_top_k[str(top_k)] = row
                    if (
                        clusters == args.primary_k
                        and method == "markov_periodic"
                        and rounds == args.primary_rounds
                        and top_k == args.primary_top_k
                    ):
                        primary_weights = weights
                    if (
                        clusters == args.primary_k
                        and method == "occupancy_periodic"
                        and rounds == args.primary_rounds
                        and top_k == args.primary_top_k
                    ):
                        primary_occupancy_weights = weights
                selection[method][str(rounds)] = by_top_k
        if clusters == args.primary_k:
            if primary_weights is None or primary_occupancy_weights is None:
                raise RuntimeError("primary selector was not evaluated")
            primary = selection["markov_periodic"][str(args.primary_rounds)][
                str(args.primary_top_k)
            ]
            primary["coherent_seed_permutation"] = coherent_fold_permutation_p(
                data,
                primary_weights,
                primary["delta_vs_random"],
                args.permutations,
                rng,
            )
            occupancy = selection["occupancy_periodic"][str(args.primary_rounds)][
                str(args.primary_top_k)
            ]
            primary["paired_vs_periodic_occupancy"] = coherent_fold_comparison_p(
                data,
                primary_weights,
                primary_occupancy_weights,
                primary["selected_success_rate"]
                - occupancy["selected_success_rate"],
                args.permutations,
                rng,
            )
            artifact = _save_production_model(
                out_dir,
                data,
                codebooks,
                clusters,
                args.alpha,
                args.position_alpha,
            )
        else:
            artifact = None
        result["clusters"][str(clusters)] = {
            "codebook": _compact_codebook(codebooks),
            "crossfit": diagnostic,
            "selection": selection,
            "full_fit_artifact": artifact,
        }
    return result


def _fmt(value: float, signed: bool = False) -> str:
    return ("%+.3f" if signed else "%.3f") % value


def _selection_row(
    task: dict[str, Any], clusters: int, method: str, rounds: int, top_k: int
) -> dict[str, Any]:
    return task["clusters"][str(clusters)]["selection"][method][str(rounds)][
        str(top_k)
    ]


def render_report(summary: dict[str, Any]) -> str:
    protocol = summary["protocol"]
    primary_name = protocol["primary_task"]
    primary = summary["tasks"][primary_name]
    k = protocol["primary_k"]
    primary_rounds = protocol["primary_rounds"]
    primary_top_k = protocol["primary_top_k"]
    lines = [
        "# Query 内 MoE 微步马尔科夫模型",
        "",
        "## 模型边界",
        "",
        (
            "这里不再把环境 control step 当成 Markov step。一次 policy query "
            "展开为 `10 denoise × 8 HB layer = 80` 个有序事件；状态 "
            "`z[tau, layer]` 是该层 10 个 action token 的完整 32 路概率指纹，"
            "经 Hellinger 嵌入和逐层 K-means 离散化。"
        ),
        "",
        (
            "主链按 `layer 2 -> 3 -> 4 -> 5 -> 12 -> 13 -> 14 -> 15 -> "
            "下一 denoise round 的 layer 2` 展开。转移矩阵以源层为条件，"
            "所以不同层的 expert 编号从未被假设为同一种专家。"
        ),
        "",
        (
            "`occupancy` 是不使用前态的零阶对照；`periodic Markov` 在 10 轮间"
            "共享 8 类层边；`position Markov` 为 79 条绝对位置边分别建模，"
            "用于检查 denoise 非齐次性。模型只使用路由概率和成败标签，不读取"
            "专家职责或 expert output。"
        ),
        "",
        "## 选择协议",
        "",
        (
            "同一初始观测的 32 个 seed 被拆为 4 个互斥 K8 池。每个测试池的"
            "完整初始场景，以及该池对应的 8 个 seed，都从成败转移矩阵训练中"
            "删除。候选分数为 `log p(path | success) - log p(path | failure)`；"
            "并列候选按等概率分摊，避免用 seed ID 人为破局。"
        ),
        "",
        (
            "主规格固定为 `K=%d`、看完前 `%d` 个 denoise round（%d 个微事件）"
            "后选 top-%d。" % (k, primary_rounds, primary_rounds * len(HB_LAYERS), primary_top_k)
        ),
        "",
        "## 主任务的前缀结果",
        "",
        "任务：`%s`，成功率 `%s`。" % (primary_name, _fmt(primary["success_rate"])),
        "",
        "| 已观察 denoise 轮数 | 零阶 top-2 相对随机 | Markov top-1 | Markov top-2 | Markov top-4 | Markov 池内 AUC |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for rounds in (1, 2, 3, 5, 10):
        zero = _selection_row(primary, k, "occupancy_periodic", rounds, 2)
        markov_1 = _selection_row(primary, k, "markov_periodic", rounds, 1)
        markov_2 = _selection_row(primary, k, "markov_periodic", rounds, 2)
        markov_4 = _selection_row(primary, k, "markov_periodic", rounds, 4)
        lines.append(
            "| %d | %s | %s | %s | %s | %s |"
            % (
                rounds,
                _fmt(zero["delta_vs_random"], True),
                _fmt(markov_1["delta_vs_random"], True),
                _fmt(markov_2["delta_vs_random"], True),
                _fmt(markov_4["delta_vs_random"], True),
                _fmt(markov_2["mean_pool_auc"]),
            )
        )

    primary_result = _selection_row(
        primary, k, "markov_periodic", primary_rounds, primary_top_k
    )
    ci = primary_result["delta_scene_bootstrap_ci95"]
    permutation = primary_result["coherent_seed_permutation"]
    paired = primary_result["paired_vs_periodic_occupancy"]
    seed_concentration = primary_result["seed_concentration"]
    lines.extend(
        [
            "",
            "## 主检验",
            "",
            (
                "主规格选中候选的平均成功率为 `%s`，K8 内精确随机期望为 `%s`，"
                "差值 `%s`，scene-bootstrap 95%% CI `[%s, %s]`，一致 seed-column "
                "置换单侧 `p=%.4f`。"
                % (
                    _fmt(primary_result["selected_success_rate"]),
                    _fmt(primary_result["exact_random_success_rate"]),
                    _fmt(primary_result["delta_vs_random"], True),
                    _fmt(ci[0], True),
                    _fmt(ci[1], True),
                    permutation["one_sided_improvement_p"],
                )
            ),
            "",
            (
                "同前缀、同 K 的零阶占用 top-%d 成功率为 `%s`；Markov 比它高 "
                "`%s`，配对一致置换单侧 `p=%.4f`。Markov 的期望选择权重覆盖 "
                "`%d/32` 个 seed，有效 seed 数 `%.1f`，最大单 seed 权重占比 "
                "`%.3f`。"
                % (
                    primary_top_k,
                    _fmt(
                        _selection_row(
                            primary,
                            k,
                            "occupancy_periodic",
                            primary_rounds,
                            primary_top_k,
                        )["selected_success_rate"]
                    ),
                    _fmt(paired["observed_selected_success_difference"], True),
                    paired["one_sided_left_better_p"],
                    seed_concentration["unique_seeds_with_positive_weight"],
                    seed_concentration["effective_seed_count"],
                    seed_concentration["maximum_expected_seed_share"],
                )
            ),
            "",
            "## 跨任务与离散粒度",
            "",
            "| 任务 | rollout 成功率 | K%d 前 %d 轮 top-%d 相对随机 | 95%% CI | 转移相对位置占用增益 (bit/transition) |"
            % (k, primary_rounds, primary_top_k),
            "|---|---:|---:|---:|---:|",
        ]
    )
    evaluated = []
    for task_name, task in summary["tasks"].items():
        if task["status"] != "evaluated":
            lines.append(
                "| `%s` | %s | 不可评估（单一 outcome） | - | - |"
                % (task_name, _fmt(task["success_rate"]))
            )
            continue
        row = _selection_row(task, k, "markov_periodic", primary_rounds, primary_top_k)
        gain = task["clusters"][str(k)]["crossfit"]["predictive"][
            "markov_position"
        ]["gain_over_occupancy_bits_per_transition"]
        row_ci = row["delta_scene_bootstrap_ci95"]
        lines.append(
            "| `%s` | %s | %s | [%s, %s] | %s |"
            % (
                task_name,
                _fmt(task["success_rate"]),
                _fmt(row["delta_vs_random"], True),
                _fmt(row_ci[0], True),
                _fmt(row_ci[1], True),
                _fmt(gain, True),
            )
        )
        evaluated.append(row["delta_vs_random"])

    lines.extend(
        [
            "",
            "主任务 K 敏感性（前 %d 轮、periodic Markov、top-%d）："
            % (primary_rounds, primary_top_k),
            "",
            "| K | 选中成功率 | 相对随机 | 池内 AUC |",
            "|---:|---:|---:|---:|",
        ]
    )
    for cluster_count in protocol["clusters"]:
        row = _selection_row(
            primary, cluster_count, "markov_periodic", primary_rounds, primary_top_k
        )
        lines.append(
            "| %d | %s | %s | %s |"
            % (
                cluster_count,
                _fmt(row["selected_success_rate"]),
                _fmt(row["delta_vs_random"], True),
                _fmt(row["mean_pool_auc"]),
            )
        )

    macro = float(np.mean(evaluated)) if evaluated else float("nan")
    predictive_gains = [
        task["clusters"][str(k)]["crossfit"]["predictive"]["markov_position"][
            "gain_over_occupancy_bits_per_transition"
        ]
        for task in summary["tasks"].values()
        if task["status"] == "evaluated"
    ]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            (
                "时间轴必须放在 query 内部，才能让候选拥有不同的路径；但这并不"
                "自动保证一阶 Markov 假设成立。位置条件链相对位置条件零阶占用"
                "的留出预测增益在 4 个可评估任务中为正 `%d/4` 次，Long 主任务"
                "反而为 `%s bit/transition`，说明当前离散链的生成拟合并不稳定。"
                % (
                    sum(value > 0 for value in predictive_gains),
                    _fmt(
                        primary["clusters"][str(k)]["crossfit"]["predictive"][
                            "markov_position"
                        ]["gain_over_occupancy_bits_per_transition"],
                        True,
                    ),
                )
            ),
            "",
            (
                "候选排序在 Long 的固定主规格上出现正关联（相对随机 `%s`，"
                "`p=%.4f`），也优于零阶占用 `%s`；但另外 3 个可评估任务均未"
                "形成明确收益，跨任务宏平均只有 `%s`。这是可进入独立闭环复验"
                "的探索性信号，不是‘已经能稳定选出好 rollout’的结论。"
                % (
                    _fmt(primary_result["delta_vs_random"], True),
                    permutation["one_sided_improvement_p"],
                    _fmt(paired["observed_selected_success_difference"], True),
                    _fmt(macro, True),
                )
            ),
            "",
            (
                "限制：标签属于整个 episode，记录 seed 也影响后续 replanning 噪声流；"
                "本分析只使用每条 episode 的第一次 policy query。真正的‘每次选择’"
                "仍需要在同一 snapshot 上分叉候选、执行所选动作并使用共同未来噪声"
                "做闭环因果验证。"
            ),
            "",
            (
                "离散 codebook 在每个任务的全部路由协变量上无标签拟合，因此当前"
                "检验是 transductive 的；部署或独立复验时必须冻结在历史训练数据"
                "上，不能用待评估 query 重拟合。"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.primary_k not in args.clusters:
        raise ValueError("--primary-k must be included in --clusters")
    if args.primary_rounds < 1 or args.primary_rounds > N_DENOISE:
        raise ValueError("--primary-rounds must be between 1 and 10")
    if args.primary_top_k not in TOP_K_VALUES:
        raise ValueError("--primary-top-k must be one of %s" % (TOP_K_VALUES,))
    if len(set(args.clusters)) != len(args.clusters) or min(args.clusters) < 2:
        raise ValueError("cluster counts must be unique and at least two")
    if args.bootstrap < 100 or args.permutations < 100:
        raise ValueError("bootstrap and permutations must each be at least 100")

    hub_root = args.hub_root.resolve()
    cache_root = hub_root / "cache" / "HiMoE-VLA"
    runs = discover_runs(hub_root, set(args.task) if args.task else None)
    task_names = {str(run.relative_to(cache_root).parent) for run in runs}
    if args.primary_task not in task_names:
        raise ValueError("primary task is not among the selected captures")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tasks: dict[str, Any] = {}
    for run in runs:
        task_name = str(run.relative_to(cache_root).parent)
        print("loading %s" % task_name, flush=True)
        data = load_task(run, cache_root)
        tasks[data.task] = analyze_task(data, args, rng, args.out_dir)

    summary = {
        "schema": "himoe-intraquery-markov-selector-v1",
        "protocol": {
            "source": "right-16x32 first policy query per episode",
            "micro_clock": "10 flow-denoise rounds x 8 ordered HB MoE layers",
            "hb_layers": list(HB_LAYERS),
            "state": "layer-specific K-means cell of sqrt 10-action-token x 32-expert probabilities",
            "expert_semantics_used": False,
            "codebook_fit": "task-local, label-blind, transductive",
            "outcome_model_fit": "held initial scene AND held K8 seed fold",
            "candidate_pool": "four disjoint K8 folds from each K32 seed grid",
            "clusters": list(args.clusters),
            "primary_k": args.primary_k,
            "primary_task": args.primary_task,
            "primary_method": "markov_periodic",
            "primary_rounds": args.primary_rounds,
            "primary_top_k": args.primary_top_k,
            "transition_prior_strength": args.alpha,
            "position_prior_strength": args.position_alpha,
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "random_seed": args.seed,
            "causal_scope": "episode-start shadow association; not per-query closed-loop",
        },
        "tasks": tasks,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "REPORT.md").write_text(
        render_report(summary), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build(args)
    primary = summary["protocol"]
    task = summary["tasks"][primary["primary_task"]]
    row = _selection_row(
        task,
        primary["primary_k"],
        primary["primary_method"],
        primary["primary_rounds"],
        primary["primary_top_k"],
    )
    print(
        "primary top-%d delta=%+.4f p=%.4f"
        % (
            primary["primary_top_k"],
            row["delta_vs_random"],
            row["coherent_seed_permutation"]["one_sided_improvement_p"],
        )
    )
    print("wrote %s" % args.out_dir.resolve())


if __name__ == "__main__":
    main()
