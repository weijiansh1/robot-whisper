#!/usr/bin/env python3
"""Screen early, single-query HB-MoE features for rollout outcome signal.

The corpus contains one closed-loop rollout for each initial-state/noise-seed
pair.  It is not a tree of continuations from later snapshots.  Accordingly,
this analysis treats ``(episode, control_query)`` as the unit and never uses a
previous or future query as an input feature.  Query indices are evaluated
separately.

For every query, fixed scalar descriptors summarize only the ten denoising
rounds inside that query:

* HB router entropy, confidence, route dispersion, and route persistence;
* HB router-input hidden-state size and within-query flow displacement.

Final failure is the label.  Models are cross-fitted by leaving out a complete
initial state, and AUC comparisons are made only between successful and failed
rollouts sharing that initial state.  The stringent control contains current
simulator state, proprioception, and the complete emitted action chunk.

This is an exploratory decodability screen.  ``hb_hidden`` is the input to an
HB MoE block, not the selected experts' output contribution; any positive
result should be followed by checkpoint-based contribution reconstruction and
an independent confirmation set.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
DEFAULT_OUT_DIR = HERE / "analysis/single-chunk-early-signal"
HB_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
N_DENOISE = 10
N_TOKENS = 11
N_EXPERTS = 32
FEATURE_VERSION = 1
EVALUATIONS = ("scene_loso", "seed_heldout")
MODEL_BLOCKS = {
    "control": ("control",),
    "router": ("router",),
    "hidden": ("hidden",),
    "moe": ("router", "hidden"),
    "control_router": ("control", "router"),
    "control_hidden": ("control", "hidden"),
    "control_moe": ("control", "router", "hidden"),
}


@dataclass(frozen=True)
class TaskFeatures:
    task: str
    run: pathlib.Path
    episodes: np.ndarray
    scenes: np.ndarray
    seeds: np.ndarray
    failure: np.ndarray
    control: np.ndarray
    router: np.ndarray
    hidden: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--task", action="append", help="suite/task; repeatable")
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--episode-batch", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=12)
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def discover_runs(
    cache_root: pathlib.Path, requested: set[str] | None = None
) -> list[pathlib.Path]:
    runs = sorted(
        path.parents[1]
        for path in cache_root.glob("**/right-16x32/client/summaries.json")
        if (path.parents[1] / "server/routes.zarr").exists()
        and (path.parents[1] / "server/hidden.zarr").exists()
    )
    if requested is not None:
        runs = [
            run
            for run in runs
            if str(run.relative_to(cache_root).parent) in requested
        ]
        found = {str(run.relative_to(cache_root).parent) for run in runs}
        if found != requested:
            raise ValueError("requested tasks not found: %s" % sorted(requested - found))
    if not runs:
        raise RuntimeError("no complete right-16x32 captures found")
    return runs


def episode_path(client: pathlib.Path, episode: int) -> pathlib.Path:
    return client / ("episode_%02d.npz" % episode)


def _sequence_summary(values: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, and normalized linear slope."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 2:
        raise ValueError("expected [sample, layer, flow-step] sequence")
    time_axis = np.linspace(-1.0, 1.0, values.shape[-1], dtype=np.float32)
    slope = np.sum(values * time_axis, axis=-1) / np.sum(time_axis * time_axis)
    result = np.stack(
        (values.mean(axis=-1), values.std(axis=-1), slope), axis=-1
    )
    return result.reshape(len(values), -1)


def _transition_summary(values: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, final value, and maximum."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("expected [sample, layer, flow-transition] sequence")
    result = np.stack(
        (
            values.mean(axis=-1),
            values.std(axis=-1),
            values[..., -1],
            values.max(axis=-1),
        ),
        axis=-1,
    )
    return result.reshape(len(values), -1)


def _rms(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values, dtype=np.float32), axis=axis))


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.sqrt(
        np.sum(left * left, axis=-1) * np.sum(right * right, axis=-1)
    )
    return 1.0 - numerator / np.maximum(denominator, 1e-12)


def router_features(probabilities: np.ndarray, expert_ids: np.ndarray) -> np.ndarray:
    """Fixed descriptors of one query's 10-step router trajectory."""
    p = np.asarray(probabilities, dtype=np.float32)
    ids = np.asarray(expert_ids)
    expected = (len(p), len(HB_LAYERS), N_DENOISE, N_TOKENS, N_EXPERTS)
    if p.shape != expected:
        raise ValueError("unexpected router shape %s; expected %s" % (p.shape, expected))
    if ids.shape != expected[:-1] + (4,):
        raise ValueError("unexpected expert-id shape %s" % (ids.shape,))
    if np.any(~np.isfinite(p)) or np.any(p < -1e-6):
        raise ValueError("router probabilities must be finite and nonnegative")

    p = np.maximum(p, 0.0)
    mass = p.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("router probability vector has zero mass")
    p /= mass

    entropy = -np.sum(p * np.log(np.maximum(p, 1e-12)), axis=-1)
    top1 = p.max(axis=-1)
    top4 = np.partition(p, -4, axis=-1)[..., -4:].sum(axis=-1)
    action_mean = p[..., 1:, :].mean(axis=3)
    action_dispersion = np.sqrt(
        0.5
        * np.sum(
            np.square(
                np.sqrt(p[..., 1:, :]) - np.sqrt(action_mean[..., None, :])
            ),
            axis=-1,
        )
    ).mean(axis=-1)
    state_action_gap = np.sqrt(
        0.5
        * np.sum(
            np.square(np.sqrt(p[..., 0, :]) - np.sqrt(action_mean)), axis=-1
        )
    )

    sequences = (
        entropy[..., 0],
        entropy[..., 1:].mean(axis=-1),
        top1[..., 0],
        top1[..., 1:].mean(axis=-1),
        top4[..., 0],
        top4[..., 1:].mean(axis=-1),
        action_dispersion,
        state_action_gap,
    )

    root = np.sqrt(p)
    hellinger = np.sqrt(
        0.5 * np.sum(np.square(root[:, :, 1:] - root[:, :, :-1]), axis=-1)
    )
    current = ids[:, :, 1:, :, :, None]
    previous = ids[:, :, :-1, :, None, :]
    hard_retention = (current == previous).any(axis=-1).mean(axis=-1)
    transitions = (
        hellinger[..., 0],
        hellinger[..., 1:].mean(axis=-1),
        hard_retention[..., 0],
        hard_retention[..., 1:].mean(axis=-1),
    )
    result = np.column_stack(
        [*(_sequence_summary(value) for value in sequences),
         *(_transition_summary(value) for value in transitions)]
    ).astype(np.float32, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError("non-finite router feature")
    return result


def hidden_features(hidden: np.ndarray) -> np.ndarray:
    """Fixed descriptors of one query's HB router-input hidden trajectory."""
    h = np.asarray(hidden, dtype=np.float32)
    if h.ndim != 5 or h.shape[1:4] != (
        len(HB_LAYERS),
        N_DENOISE,
        N_TOKENS,
    ):
        raise ValueError("unexpected hidden shape %s" % (h.shape,))
    state = h[..., 0, :]
    action = h[..., 1:, :]
    action_mean = action.mean(axis=3)

    sequences = (
        _rms(state, axis=-1),
        _rms(action, axis=(-1, -2)),
        _rms(action - action_mean[..., None, :], axis=(-1, -2)),
        _cosine_distance(state, action_mean),
        _rms(state - action_mean, axis=-1),
    )

    state_step = state[:, :, 1:] - state[:, :, :-1]
    action_step = action_mean[:, :, 1:] - action_mean[:, :, :-1]
    transitions = (
        _rms(state_step, axis=-1),
        _rms(action_step, axis=-1),
        _cosine_distance(state[:, :, 1:], state[:, :, :-1]),
        _cosine_distance(action_mean[:, :, 1:], action_mean[:, :, :-1]),
    )

    path_values: list[np.ndarray] = []
    for trajectory, step in ((state, state_step), (action_mean, action_step)):
        step_size = _rms(step, axis=-1)
        path_length = step_size.sum(axis=-1)
        endpoint = _rms(trajectory[:, :, -1] - trajectory[:, :, 0], axis=-1)
        straightness = endpoint / np.maximum(path_length, 1e-12)
        path_values.extend((endpoint, path_length, straightness))

    result = np.column_stack(
        [*(_sequence_summary(value) for value in sequences),
         *(_transition_summary(value) for value in transitions),
         *path_values]
    ).astype(np.float32, copy=False)
    if np.any(~np.isfinite(result)):
        raise ValueError("non-finite hidden feature")
    return result


def _task_name(run: pathlib.Path, cache_root: pathlib.Path) -> str:
    return str(run.relative_to(cache_root).parent)


def _cache_path(out_dir: pathlib.Path, task: str) -> pathlib.Path:
    return out_dir / "features" / (task.replace("/", "__") + ".npz")


def extract_task(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    out_dir: pathlib.Path,
    max_chunks: int,
    episode_batch: int,
    force: bool,
) -> TaskFeatures | None:
    task = _task_name(run, cache_root)
    summaries = sorted(
        json.loads((run / "client/summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    failure = np.asarray([not bool(row["success"]) for row in summaries])
    if not np.any(failure) or np.all(failure):
        print("%s: skipped (no outcome variation)" % task, flush=True)
        return None
    if min(int(row["inference_calls"]) for row in summaries) < max_chunks:
        raise ValueError("%s has fewer than %d queries" % (task, max_chunks))

    cache_path = _cache_path(out_dir, task)
    if cache_path.exists() and not force:
        with np.load(cache_path, allow_pickle=False) as cached:
            version = int(cached["feature_version"])
            chunks = int(cached["max_chunks"])
            if version == FEATURE_VERSION and chunks == max_chunks:
                print("%s: loading cached features" % task, flush=True)
                return TaskFeatures(
                    task=task,
                    run=run,
                    episodes=np.asarray(cached["episodes"]),
                    scenes=np.asarray(cached["scenes"]),
                    seeds=np.asarray(cached["seeds"]),
                    failure=np.asarray(cached["failure"], dtype=bool),
                    control=np.asarray(cached["control"], dtype=np.float32),
                    router=np.asarray(cached["router"], dtype=np.float32),
                    hidden=np.asarray(cached["hidden"], dtype=np.float32),
                )

    episodes = np.asarray([int(row["episode_index"]) for row in summaries])
    scenes = np.asarray([int(row["init_state_id"]) for row in summaries])
    seeds = np.asarray([int(row["flow_noise_seed"]) for row in summaries])
    counts = np.asarray([int(row["inference_calls"]) for row in summaries])
    offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    row_grid = offsets[:, None] + np.arange(max_chunks, dtype=np.int64)[None, :]

    route_group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    hidden_group = zarr.open_group(str(run / "server/hidden.zarr"), mode="r")
    route_episode = np.asarray(route_group["episode_id"][:])
    hidden_episode = np.asarray(hidden_group["episode_id"][:])
    route_step = np.asarray(route_group["control_step"][:])
    hidden_step = np.asarray(hidden_group["control_step"][:])
    if not np.array_equal(route_episode, hidden_episode) or not np.array_equal(
        route_step, hidden_step
    ):
        raise ValueError("route/hidden row alignment failed for %s" % task)
    if len(route_episode) != int(counts.sum()):
        raise ValueError("summary/server row count mismatch for %s" % task)
    if not np.all(route_episode[row_grid] == episodes[:, None]):
        raise ValueError("episode row lookup failed for %s" % task)
    # control_step is the server-global query counter, not an episode-local k.
    if not np.array_equal(route_step, np.arange(len(route_step))):
        raise ValueError("global control_step ordering failed for %s" % task)

    metadata = json.loads((run / "client/server_metadata.json").read_text())
    layers = tuple(int(value) for value in metadata["routing_hb_layer_indices"])
    if layers != HB_LAYERS:
        raise ValueError("recorded HB layers changed: %s" % (layers,))

    controls: list[np.ndarray] = []
    for episode in episodes:
        with np.load(episode_path(run / "client", int(episode)), allow_pickle=False) as record:
            state = np.asarray(record["state"][:max_chunks], dtype=np.float32)
            action = np.asarray(record["actions"][:max_chunks], dtype=np.float32)
            sim = np.asarray(record["sim_state"][:max_chunks], dtype=np.float32)
        controls.append(
            np.column_stack((state, action.reshape(max_chunks, -1), sim))
        )
    control = np.stack(controls)
    for scene in np.unique(scenes):
        index = np.flatnonzero(scenes == scene)
        if not np.allclose(control[index, 0, :8], control[index[0], 0, :8]):
            raise ValueError("query-0 proprioception varies within scene %s" % scene)
        if not np.allclose(control[index, 0, 78:], control[index[0], 0, 78:]):
            raise ValueError("query-0 simulator state varies within scene %s" % scene)

    n_episodes = len(episodes)
    router_result: np.ndarray | None = None
    hidden_result: np.ndarray | None = None
    started = time.perf_counter()
    for lo in range(0, n_episodes, episode_batch):
        hi = min(lo + episode_batch, n_episodes)
        rows = row_grid[lo:hi].reshape(-1)
        p = np.asarray(
            route_group["hb_router_probs"].oindex[rows, :, :, :, :],
            dtype=np.float32,
        )
        ids = np.asarray(route_group["hb_expert_ids"].oindex[rows, :, :, :, :])
        route_batch = router_features(p, ids).reshape(hi - lo, max_chunks, -1)
        if router_result is None:
            router_result = np.empty(
                (n_episodes, max_chunks, route_batch.shape[-1]), dtype=np.float32
            )
        router_result[lo:hi] = route_batch
        del p, ids, route_batch

        h = np.asarray(
            hidden_group["hb_hidden"].oindex[rows, :, :, :, :], dtype=np.float32
        )
        hidden_batch = hidden_features(h).reshape(hi - lo, max_chunks, -1)
        if hidden_result is None:
            hidden_result = np.empty(
                (n_episodes, max_chunks, hidden_batch.shape[-1]), dtype=np.float32
            )
        hidden_result[lo:hi] = hidden_batch
        del h, hidden_batch
        if hi % 32 == 0 or hi == n_episodes:
            print(
                "%s: features %d/%d episodes (%.1fs)"
                % (task, hi, n_episodes, time.perf_counter() - started),
                flush=True,
            )
        gc.collect()

    assert router_result is not None and hidden_result is not None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        feature_version=np.asarray(FEATURE_VERSION),
        max_chunks=np.asarray(max_chunks),
        episodes=episodes,
        scenes=scenes,
        seeds=seeds,
        failure=failure,
        control=control,
        router=router_result,
        hidden=hidden_result,
    )
    print("%s: cached %s" % (task, cache_path), flush=True)
    return TaskFeatures(
        task=task,
        run=run,
        episodes=episodes,
        scenes=scenes,
        seeds=seeds,
        failure=failure,
        control=control,
        router=router_result,
        hidden=hidden_result,
    )


def _reduce_block(
    train: np.ndarray,
    test: np.ndarray,
    components: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    keep = np.std(train, axis=0) > 1e-9
    if not np.any(keep):
        raise ValueError("feature block is constant in training data")
    scaler = StandardScaler().fit(train[:, keep])
    x_train = scaler.transform(train[:, keep])
    x_test = scaler.transform(test[:, keep])
    count = min(components, x_train.shape[1], x_train.shape[0] - 1)
    if count < x_train.shape[1]:
        pca = PCA(
            n_components=count,
            svd_solver="randomized",
            random_state=random_state,
        ).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test


def conditional_pair_stats(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return group ids, concordant wins, and positive-negative pair counts."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    unique = np.unique(groups)
    wins = np.zeros(len(unique), dtype=np.float64)
    pairs = np.zeros(len(unique), dtype=np.int64)
    for axis, group in enumerate(unique):
        index = groups == group
        positive = scores[index & labels]
        negative = scores[index & ~labels]
        if not len(positive) or not len(negative):
            continue
        wins[axis] = float((positive[:, None] > negative[None, :]).sum())
        wins[axis] += 0.5 * float((positive[:, None] == negative[None, :]).sum())
        pairs[axis] = len(positive) * len(negative)
    return unique, wins, pairs


def _auc_from_stats(wins: np.ndarray, pairs: np.ndarray) -> float:
    return float(wins.sum() / pairs.sum()) if pairs.sum() else float("nan")


def cross_fitted_predictions(
    data: TaskFeatures,
    chunk: int,
    components: int,
    logistic_c: float,
    seed: int,
    evaluation: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    labels = data.failure.astype(np.int8)
    blocks = {
        "control": data.control[:, chunk],
        "router": data.router[:, chunk],
        "hidden": data.hidden[:, chunk],
    }
    predictions = {
        name: np.full(len(labels), np.nan, dtype=np.float64) for name in MODEL_BLOCKS
    }
    unique_scenes = np.unique(data.scenes)
    scene_onehot = (data.scenes[:, None] == unique_scenes[None, :]).astype(np.float64)
    if evaluation == "scene_loso":
        splits = [
            (data.scenes != held_scene, data.scenes == held_scene)
            for held_scene in unique_scenes
        ]
        strata = data.scenes.astype(np.int64)
        add_scene_context = False
    elif evaluation == "seed_heldout":
        unique_seeds = np.sort(np.unique(data.seeds))
        if len(unique_seeds) != 32:
            raise ValueError("seed-held-out evaluation expects 32 unique seeds")
        seed_rank = {int(value): axis for axis, value in enumerate(unique_seeds)}
        seed_fold = np.asarray([seed_rank[int(value)] % 4 for value in data.seeds])
        for scene in unique_scenes:
            if set(map(int, data.seeds[data.scenes == scene])) != set(
                map(int, unique_seeds)
            ):
                raise ValueError("scene %s does not contain every seed" % scene)
        splits = [(seed_fold != fold, seed_fold == fold) for fold in range(4)]
        scene_axis = {int(value): axis for axis, value in enumerate(unique_scenes)}
        strata = np.asarray(
            [scene_axis[int(scene)] * 4 + fold for scene, fold in zip(data.scenes, seed_fold)]
        )
        add_scene_context = True
    else:
        raise ValueError("unknown evaluation %s" % evaluation)

    for fold, (train, test) in enumerate(splits):
        if len(np.unique(labels[train])) != 2:
            raise ValueError("training set has one outcome in %s fold %d" % (evaluation, fold))
        reduced = {
            name: _reduce_block(
                values[train],
                values[test],
                components,
                random_state=seed + chunk * 100 + fold,
            )
            for name, values in blocks.items()
        }
        for model_name, names in MODEL_BLOCKS.items():
            train_parts = [reduced[name][0] for name in names]
            test_parts = [reduced[name][1] for name in names]
            if add_scene_context:
                train_parts.insert(0, scene_onehot[train])
                test_parts.insert(0, scene_onehot[test])
            x_train = np.column_stack(train_parts)
            x_test = np.column_stack(test_parts)
            scaler = StandardScaler().fit(x_train)
            model = LogisticRegression(
                C=logistic_c,
                solver="lbfgs",
                max_iter=3000,
                class_weight="balanced",
                random_state=seed,
            ).fit(scaler.transform(x_train), labels[train])
            predictions[model_name][test] = model.predict_proba(
                scaler.transform(x_test)
            )[:, 1]
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("cross-fitting did not produce complete predictions")
    return predictions, strata


def conditional_scene_pair_stats(
    labels: np.ndarray,
    scores: np.ndarray,
    scenes: np.ndarray,
    strata: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate conditional concordance counts to scene bootstrap units."""
    unique_scenes = np.unique(scenes)
    wins = np.zeros(len(unique_scenes), dtype=np.float64)
    pairs = np.zeros(len(unique_scenes), dtype=np.int64)
    for axis, scene in enumerate(unique_scenes):
        index = scenes == scene
        _, stratum_wins, stratum_pairs = conditional_pair_stats(
            labels[index], scores[index], strata[index]
        )
        wins[axis] = stratum_wins.sum()
        pairs[axis] = stratum_pairs.sum()
    return unique_scenes, wins, pairs


def evaluate_task(
    data: TaskFeatures,
    components: int,
    logistic_c: float,
    bootstrap: int,
    seed: int,
    evaluation: str,
) -> tuple[list[dict], list[dict], dict[tuple[int, str], np.ndarray]]:
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    bootstraps: dict[tuple[int, str], np.ndarray] = {}
    unique_scenes = np.unique(data.scenes)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_scenes), size=(bootstrap, len(unique_scenes)))

    for chunk in range(data.control.shape[1]):
        predictions, strata = cross_fitted_predictions(
            data, chunk, components, logistic_c, seed, evaluation
        )
        stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, scores in predictions.items():
            groups, wins, pairs = conditional_scene_pair_stats(
                data.failure, scores, data.scenes, strata
            )
            if not np.array_equal(groups, unique_scenes):
                raise RuntimeError("scene ordering changed")
            stats[name] = wins, pairs
            sampled_wins = wins[draws].sum(axis=1)
            sampled_pairs = pairs[draws].sum(axis=1)
            boot = np.divide(
                sampled_wins,
                sampled_pairs,
                out=np.full(bootstrap, np.nan),
                where=sampled_pairs > 0,
            )
            bootstraps[(chunk, name)] = boot

        baseline_boot = bootstraps[(chunk, "control")]
        for name, scores in predictions.items():
            wins, pairs = stats[name]
            auc = _auc_from_stats(wins, pairs)
            valid_boot = bootstraps[(chunk, name)]
            valid_boot = valid_boot[np.isfinite(valid_boot)]
            delta_boot = bootstraps[(chunk, name)] - baseline_boot
            delta_boot = delta_boot[np.isfinite(delta_boot)]
            metric_rows.append(
                {
                    "evaluation": evaluation,
                    "task": data.task,
                    "chunk": chunk,
                    "model": name,
                    "failures": int(data.failure.sum()),
                    "mixed_scenes": int(np.sum(pairs > 0)),
                    "conditional_pairs": int(pairs.sum()),
                    "conditional_auc": auc,
                    "auc_ci_low": float(np.percentile(valid_boot, 2.5)),
                    "auc_ci_high": float(np.percentile(valid_boot, 97.5)),
                    "delta_vs_control": auc - _auc_from_stats(*stats["control"]),
                    "delta_ci_low": float(np.percentile(delta_boot, 2.5)),
                    "delta_ci_high": float(np.percentile(delta_boot, 97.5)),
                    "global_roc_auc": float(roc_auc_score(data.failure, scores)),
                    "global_average_precision": float(
                        average_precision_score(data.failure, scores)
                    ),
                }
            )
            for episode, scene, label, score in zip(
                data.episodes, data.scenes, data.failure, scores
            ):
                prediction_rows.append(
                    {
                        "evaluation": evaluation,
                        "task": data.task,
                        "episode": int(episode),
                        "init_state_id": int(scene),
                        "chunk": chunk,
                        "failure": int(label),
                        "model": name,
                        "score": float(score),
                    }
                )
        print(
            "%s: %s evaluated chunk %d" % (data.task, evaluation, chunk),
            flush=True,
        )
    return metric_rows, prediction_rows, bootstraps


def macro_summary(
    metric_rows: list[dict],
    task_bootstraps: dict[tuple[str, str, int, str], np.ndarray],
    max_chunks: int,
) -> list[dict]:
    tasks = sorted({row["task"] for row in metric_rows})
    by_key = {
        (row["evaluation"], row["task"], row["chunk"], row["model"]): row
        for row in metric_rows
    }
    output = []
    for evaluation in EVALUATIONS:
        for chunk in range(max_chunks):
            baseline_boot = np.nanmean(
                [
                    task_bootstraps[(evaluation, task, chunk, "control")]
                    for task in tasks
                ],
                axis=0,
            )
            for model in MODEL_BLOCKS:
                rows = [
                    by_key[(evaluation, task, chunk, model)] for task in tasks
                ]
                auc_boot = np.nanmean(
                    [
                        task_bootstraps[(evaluation, task, chunk, model)]
                        for task in tasks
                    ],
                    axis=0,
                )
                valid_auc = auc_boot[np.isfinite(auc_boot)]
                delta_boot = auc_boot - baseline_boot
                valid_delta = delta_boot[np.isfinite(delta_boot)]
                output.append(
                    {
                        "evaluation": evaluation,
                        "chunk": chunk,
                        "model": model,
                        "macro_conditional_auc": float(
                            np.mean([row["conditional_auc"] for row in rows])
                        ),
                        "auc_ci_low": float(np.percentile(valid_auc, 2.5)),
                        "auc_ci_high": float(np.percentile(valid_auc, 97.5)),
                        "macro_delta_vs_control": float(
                            np.mean([row["delta_vs_control"] for row in rows])
                        ),
                        "delta_ci_low": float(np.percentile(valid_delta, 2.5)),
                        "delta_ci_high": float(np.percentile(valid_delta, 97.5)),
                        "tasks_delta_positive": int(
                            sum(row["delta_vs_control"] > 0 for row in rows)
                        ),
                        "tasks": len(tasks),
                    }
                )
    return output


def _write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict) -> str:
    macro = {
        (row["evaluation"], row["chunk"], row["model"]): row
        for row in summary["macro_metrics"]
    }
    lines = [
        "# Early single-query outcome signal",
        "",
        "## Scope",
        "",
        "- Unit: one rollout at one control query. Features never cross query boundaries.",
        "- Queries 0 through %d are evaluated independently." % (summary["max_chunks"] - 1),
        "- Query 0 compares exact shared initial states. Later queries share only the rollout origin; they are not tree-sampled counterfactuals.",
        "- Positive label: eventual rollout failure. The all-success task is excluded.",
        "- Evaluations: (1) leave one complete initial state out; (2) hold out complete noise seeds while retaining other seeds from the same state.",
        "- Conditional AUC always counts only failure/success pairs from the same initial state and the same fitted fold.",
        "- Control: current simulator state, proprioception, and all 10x7 emitted action values.",
        "- Each control/router/hidden block is standardized and reduced to %d train-only PCA components; logistic C=%g is fixed." % (summary["pca_components"], summary["logistic_c"]),
        "- Uncertainty: initial-state bootstrap, conditional on these tasks and recorded rollouts.",
        "",
        "`hidden` means the HB router-input representation. It is not the selected experts' output contribution.",
        "",
    ]
    for evaluation, title in (
        ("scene_loso", "Unseen-initial-state result"),
        ("seed_heldout", "Same-initial-state, held-out-seed result"),
    ):
        lines.extend(
            [
                "",
                "## %s" % title,
                "",
                "AUC 0.5 is chance. Delta is `control + router + hidden` minus the control AUC.",
                "",
                "| chunk | control AUC | hidden-only AUC | MoE-only AUC | control+MoE AUC | delta | tasks with positive delta |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for chunk in range(summary["max_chunks"]):
            control = macro[(evaluation, chunk, "control")]
            hidden = macro[(evaluation, chunk, "hidden")]
            moe = macro[(evaluation, chunk, "moe")]
            combined = macro[(evaluation, chunk, "control_moe")]
            lines.append(
                "| %d | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %+.3f [%.3f, %.3f] | %d/%d |"
                % (
                    chunk,
                    control["macro_conditional_auc"],
                    control["auc_ci_low"],
                    control["auc_ci_high"],
                    hidden["macro_conditional_auc"],
                    hidden["auc_ci_low"],
                    hidden["auc_ci_high"],
                    moe["macro_conditional_auc"],
                    moe["auc_ci_low"],
                    moe["auc_ci_high"],
                    combined["macro_conditional_auc"],
                    combined["auc_ci_low"],
                    combined["auc_ci_high"],
                    combined["macro_delta_vs_control"],
                    combined["delta_ci_low"],
                    combined["delta_ci_high"],
                    combined["tasks_delta_positive"],
                    combined["tasks"],
                )
            )

    unseen_hidden = macro[("scene_loso", 0, "hidden")]
    same_hidden = macro[("seed_heldout", 0, "hidden")]
    same_combined = macro[("seed_heldout", 0, "control_moe")]
    lines.extend(
        [
            "",
            "## Bottom line",
            "",
            "The fixed screen does not show a stable early HB-MoE outcome signal. The only repeatable hint is query-0 `hidden` alone: unseen-state macro AUC %.3f [%.3f, %.3f] and same-state held-out-seed macro AUC %.3f [%.3f, %.3f]. Both intervals include chance. After adding router features and the state/action control, the same-state AUC is %.3f and the incremental delta is %+.3f [%.3f, %.3f], so the hint is not incremental under this model. No later query has a consistently positive increment."
            % (
                unseen_hidden["macro_conditional_auc"],
                unseen_hidden["auc_ci_low"],
                unseen_hidden["auc_ci_high"],
                same_hidden["macro_conditional_auc"],
                same_hidden["auc_ci_low"],
                same_hidden["auc_ci_high"],
                same_combined["macro_conditional_auc"],
                same_combined["macro_delta_vs_control"],
                same_combined["delta_ci_low"],
                same_combined["delta_ci_high"],
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Per-task control+MoE result",
            "",
            "| evaluation | task | chunk | failures | mixed states | control AUC | hidden AUC | control+MoE AUC | delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    metrics = {
        (row["evaluation"], row["task"], row["chunk"], row["model"]): row
        for row in summary["task_metrics"]
    }
    for evaluation in EVALUATIONS:
        for task in summary["tasks"]:
            for chunk in range(summary["max_chunks"]):
                control = metrics[(evaluation, task, chunk, "control")]
                hidden = metrics[(evaluation, task, chunk, "hidden")]
                combined = metrics[(evaluation, task, chunk, "control_moe")]
                lines.append(
                    "| %s | %s | %d | %d | %d | %.3f | %.3f | %.3f | %+.3f [%.3f, %.3f] |"
                    % (
                        evaluation,
                        task,
                        chunk,
                        combined["failures"],
                        combined["mixed_scenes"],
                        control["conditional_auc"],
                        hidden["conditional_auc"],
                        combined["conditional_auc"],
                        combined["delta_vs_control"],
                        combined["delta_ci_low"],
                        combined["delta_ci_high"],
                    )
                )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This screen uses eventual outcome labels, not annotated trap-onset labels. A positive result establishes only early outcome decodability under the recorded policy. It does not identify a trap mechanism, recoverability, or a causal expert contribution. The five chunk indices and seven model families are exploratory and are not multiplicity-corrected.",
            "",
        ]
    )
    return "\n".join(lines)


def make_plot(summary: dict, output: pathlib.Path, evaluation: str) -> None:
    import matplotlib.pyplot as plt

    macro = {
        (row["chunk"], row["model"]): row
        for row in summary["macro_metrics"]
        if row["evaluation"] == evaluation
    }
    chunks = np.arange(summary["max_chunks"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for name, label, color in (
        ("control", "sim/proprio + action", "#4C566A"),
        ("moe", "HB internal only", "#2A9D8F"),
        ("control_moe", "control + HB", "#C44536"),
    ):
        values = np.asarray([macro[(int(k), name)]["macro_conditional_auc"] for k in chunks])
        low = np.asarray([macro[(int(k), name)]["auc_ci_low"] for k in chunks])
        high = np.asarray([macro[(int(k), name)]["auc_ci_high"] for k in chunks])
        axes[0].plot(chunks, values, marker="o", label=label, color=color)
        axes[0].fill_between(chunks, low, high, alpha=0.12, color=color)
    axes[0].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[0].set(xlabel="control query k", ylabel="macro conditional AUC", xticks=chunks)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(evaluation.replace("_", " "))

    combined = [macro[(int(k), "control_moe")] for k in chunks]
    delta = np.asarray([row["macro_delta_vs_control"] for row in combined])
    low = np.asarray([row["delta_ci_low"] for row in combined])
    high = np.asarray([row["delta_ci_high"] for row in combined])
    axes[1].errorbar(
        chunks,
        delta,
        yerr=np.vstack((delta - low, high - delta)),
        marker="o",
        capsize=3,
        color="#C44536",
    )
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set(
        xlabel="control query k",
        ylabel="AUC delta: control + HB minus control",
        xticks=chunks,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.max_chunks < 1 or args.episode_batch < 1:
        raise ValueError("max-chunks and episode-batch must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.task) if args.task else None
    runs = discover_runs(args.cache_root, requested)
    datasets = []
    for run in runs:
        data = extract_task(
            run,
            args.cache_root,
            args.out_dir,
            args.max_chunks,
            args.episode_batch,
            args.force_features,
        )
        if data is not None:
            datasets.append(data)
    if not datasets:
        raise RuntimeError("no task has both success and failure outcomes")

    task_rows: list[dict] = []
    prediction_rows: list[dict] = []
    task_bootstraps: dict[tuple[str, str, int, str], np.ndarray] = {}
    for evaluation_index, evaluation in enumerate(EVALUATIONS):
        for task_index, data in enumerate(datasets):
            rows, predictions, boots = evaluate_task(
                data,
                args.pca_components,
                args.logistic_c,
                args.bootstrap,
                args.seed + evaluation_index * 100000 + task_index * 10000,
                evaluation,
            )
            task_rows.extend(rows)
            prediction_rows.extend(predictions)
            for (chunk, model), values in boots.items():
                task_bootstraps[(evaluation, data.task, chunk, model)] = values

    macro = macro_summary(task_rows, task_bootstraps, args.max_chunks)
    summary = {
        "analysis": "single-query early rollout-outcome screen",
        "feature_version": FEATURE_VERSION,
        "cache_root": str(args.cache_root.resolve()),
        "tasks": [data.task for data in datasets],
        "max_chunks": args.max_chunks,
        "pca_components": args.pca_components,
        "logistic_c": args.logistic_c,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "evaluations": list(EVALUATIONS),
        "router_feature_count": int(datasets[0].router.shape[-1]),
        "hidden_feature_count": int(datasets[0].hidden.shape[-1]),
        "task_metrics": task_rows,
        "macro_metrics": macro,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(args.out_dir / "metrics.csv", task_rows)
    _write_csv(args.out_dir / "predictions.csv", prediction_rows)
    (args.out_dir / "report.md").write_text(render_report(summary))
    for evaluation in EVALUATIONS:
        make_plot(
            summary,
            args.out_dir / ("auc_by_chunk_%s.png" % evaluation),
            evaluation,
        )
    legacy_plot = args.out_dir / "auc_by_chunk.png"
    if legacy_plot.exists():
        legacy_plot.unlink()
    print("wrote %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
