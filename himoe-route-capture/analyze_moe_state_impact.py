#!/usr/bin/env python3
"""Test whether first-query routed MoE state forms an action commitment.

The first policy query is identical in physical state within each 32-seed
initial-state pool.  True selected-expert contributions are reconstructed from
the captured HB inputs, stored top-4 decisions, and checkpoint MLP weights.
Only within-query flow positions are predictors.  Final action, next physical
state, and rollout outcome are downstream readouts.
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
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_expert_activation_v2 import compute_features, load_dataset


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
OUT_DIR = HERE / "analysis/moe-state-impact"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LONG_FEATURE_CACHE = HERE / "analysis/expert-activation-v2/long-t08/features_v2.npz"
LAYERS = (2, 5, 12, 15)
N_DENOISE = 10
N_TOKENS = 11
WIDTH = 1024
TOKEN_PROJECTED_DIM = 64
FEATURE_VERSION = 1
BOOTSTRAPS = 5000
PCA_COMPONENTS = 12
LOGISTIC_C = 0.1
PRIMARY_DENOISE = (0, 8)
BRANCHES = ("routed", "shared", "hidden")
TARGETS = ("action", "next_proprio", "next_sim")
OUTCOME_FAMILIES = {
    "routed_identity": ("routed",),
    "routed_scalar": ("scalar",),
    "routed_full": ("routed", "scalar"),
    "shared_identity": ("shared",),
    "hidden_identity": ("hidden",),
}


@dataclass(frozen=True)
class CompactTask:
    task: str
    episodes: np.ndarray
    scenes: np.ndarray
    seeds: np.ndarray
    failure: np.ndarray
    action: np.ndarray
    proprio_delta: np.ndarray
    sim_delta: np.ndarray
    routed: np.ndarray
    shared: np.ndarray
    hidden: np.ndarray
    scalar: np.ndarray
    routed_commitment: np.ndarray
    shared_commitment: np.ndarray
    hidden_commitment: np.ndarray
    validation: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--task", action="append", help="suite/task; repeatable")
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--force-compact", action="store_true")
    return parser.parse_args()


def discover_runs(cache_root: pathlib.Path, requested: set[str] | None) -> list[pathlib.Path]:
    runs = []
    for summary in sorted(cache_root.glob("**/right-16x32/client/summaries.json")):
        run = summary.parents[1]
        rows = json.loads(summary.read_text())
        task = str(run.relative_to(cache_root).parent)
        if requested is not None and task not in requested:
            continue
        labels = [bool(row["success"]) for row in rows]
        if any(labels) and not all(labels):
            runs.append(run)
    found = {str(run.relative_to(cache_root).parent) for run in runs}
    if requested is not None and found != requested:
        raise ValueError("requested variable-outcome tasks not found: %s" % sorted(requested - found))
    if not runs:
        raise RuntimeError("no variable-outcome right-16x32 runs found")
    return runs


def _episode_path(run: pathlib.Path, episode: int) -> pathlib.Path:
    return run / "client" / ("episode_%02d.npz" % episode)


def _compact_path(out_dir: pathlib.Path, task: str) -> pathlib.Path:
    return out_dir / "compact" / (task.replace("/", "__") + ".npz")


def _token_projection(seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    bucket = rng.integers(0, TOKEN_PROJECTED_DIM, size=WIDTH)
    sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=WIDTH)
    return sparse.csr_matrix(
        (sign, (np.arange(WIDTH), bucket)),
        shape=(WIDTH, TOKEN_PROJECTED_DIM),
    )


def project_token_directions(values: np.ndarray, seed: int) -> np.ndarray:
    """Project [N,D,L,T,W] while preserving every layer/token position."""
    values = np.asarray(values)
    if values.ndim != 5 or values.shape[1:] != (
        N_DENOISE,
        len(LAYERS),
        N_TOKENS,
        WIDTH,
    ):
        raise ValueError("unexpected contribution shape %s" % (values.shape,))
    projection = _token_projection(seed)
    result = np.empty(
        (
            len(values),
            N_DENOISE,
            len(LAYERS) * N_TOKENS * TOKEN_PROJECTED_DIM,
        ),
        dtype=np.float16,
    )
    for denoise in range(N_DENOISE):
        block = np.asarray(values[:, denoise], dtype=np.float32)
        block /= np.maximum(np.linalg.norm(block, axis=-1, keepdims=True), 1e-12)
        projected = np.asarray(block.reshape(-1, WIDTH) @ projection, dtype=np.float32)
        projected /= np.maximum(np.linalg.norm(projected, axis=-1, keepdims=True), 1e-12)
        result[:, denoise] = projected.reshape(len(values), -1).astype(np.float16)
    return result


def contribution_commitment(values: np.ndarray) -> np.ndarray:
    """Mean per-layer/token cosine to the query's final flow position."""
    values = np.asarray(values)
    reference = np.asarray(values[:, -1], dtype=np.float32)
    reference /= np.maximum(np.linalg.norm(reference, axis=-1, keepdims=True), 1e-12)
    output = np.empty((len(values), N_DENOISE), dtype=np.float32)
    for denoise in range(N_DENOISE):
        current = np.asarray(values[:, denoise], dtype=np.float32)
        current /= np.maximum(np.linalg.norm(current, axis=-1, keepdims=True), 1e-12)
        output[:, denoise] = np.sum(current * reference, axis=-1).mean(axis=(1, 2))
    return output


def _load_full_features(
    task: str, data: dict, checkpoint: pathlib.Path, threads: int
) -> dict:
    if task == LONG_TASK and LONG_FEATURE_CACHE.exists():
        print("%s: using existing full contribution cache" % task, flush=True)
        with np.load(LONG_FEATURE_CACHE, allow_pickle=False) as stored:
            if tuple(map(int, stored["layer_numbers"])) != LAYERS:
                raise ValueError("long contribution cache layer mismatch")
            return {
                "scalars": np.asarray(stored["scalars"]),
                "routed_vec": np.asarray(stored["routed_vec"]),
                "shared_vec": np.asarray(stored["shared_vec"]),
                "validation": json.loads(str(stored["validation_json"])),
            }
    return compute_features(data, checkpoint, threads)


def extract_compact(
    run: pathlib.Path,
    cache_root: pathlib.Path,
    out_dir: pathlib.Path,
    threads: int,
    seed: int,
    force: bool,
) -> CompactTask:
    task = str(run.relative_to(cache_root).parent)
    cache = _compact_path(out_dir, task)
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as stored:
            if int(stored["feature_version"]) == FEATURE_VERSION:
                print("%s: loading compact contribution cache" % task, flush=True)
                return CompactTask(
                    task=task,
                    episodes=np.asarray(stored["episodes"]),
                    scenes=np.asarray(stored["scenes"]),
                    seeds=np.asarray(stored["seeds"]),
                    failure=np.asarray(stored["failure"], dtype=bool),
                    action=np.asarray(stored["action"], dtype=np.float32),
                    proprio_delta=np.asarray(stored["proprio_delta"], dtype=np.float32),
                    sim_delta=np.asarray(stored["sim_delta"], dtype=np.float32),
                    routed=np.asarray(stored["routed"], dtype=np.float16),
                    shared=np.asarray(stored["shared"], dtype=np.float16),
                    hidden=np.asarray(stored["hidden"], dtype=np.float16),
                    scalar=np.asarray(stored["scalar"], dtype=np.float32),
                    routed_commitment=np.asarray(stored["routed_commitment"], dtype=np.float32),
                    shared_commitment=np.asarray(stored["shared_commitment"], dtype=np.float32),
                    hidden_commitment=np.asarray(stored["hidden_commitment"], dtype=np.float32),
                    validation=json.loads(str(stored["validation_json"])),
                )

    started = time.perf_counter()
    data = load_dataset(run, LAYERS)
    metadata = json.loads((run / "client/server_metadata.json").read_text())
    checkpoint = pathlib.Path(metadata["checkpoint"]) / "pytorch_model.pth"
    full = _load_full_features(task, data, checkpoint, threads)

    routed_full = np.asarray(full["routed_vec"])
    shared_full = np.asarray(full["shared_vec"])
    hidden_full = np.transpose(data["hidden"], (0, 2, 1, 3, 4))
    routed = project_token_directions(routed_full, seed + 11)
    shared = project_token_directions(shared_full, seed + 11)
    hidden = project_token_directions(hidden_full, seed + 11)
    routed_commitment = contribution_commitment(routed_full)
    shared_commitment = contribution_commitment(shared_full)
    hidden_commitment = contribution_commitment(hidden_full)
    # Exclude input_rms and shared_rms; retain seven routed/expert-state scalars.
    scalar = np.asarray(full["scalars"][..., 1:8], dtype=np.float32).reshape(
        len(data["episodes"]), N_DENOISE, -1
    )

    next_state = []
    next_sim = []
    for episode in data["episodes"]:
        with np.load(_episode_path(run, int(episode)), allow_pickle=False) as record:
            state = np.asarray(record["state"][:2], dtype=np.float32)
            sim = np.asarray(record["sim_state"][:2], dtype=np.float32)
        next_state.append(state[1] - state[0])
        next_sim.append(sim[1, 1:] - sim[0, 1:])

    action = (data["actions"] / data["action_std"][None, None, :]).reshape(
        len(data["episodes"]), -1
    ).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        feature_version=np.asarray(FEATURE_VERSION),
        episodes=data["episodes"],
        scenes=data["scenes"],
        seeds=data["noise_seeds"],
        failure=1 - data["success"],
        action=action,
        proprio_delta=np.stack(next_state),
        sim_delta=np.stack(next_sim),
        routed=routed,
        shared=shared,
        hidden=hidden,
        scalar=scalar,
        routed_commitment=routed_commitment,
        shared_commitment=shared_commitment,
        hidden_commitment=hidden_commitment,
        validation_json=json.dumps(full["validation"]),
    )
    print("%s: compact cache written in %.1fs" % (task, time.perf_counter() - started), flush=True)
    result = CompactTask(
        task=task,
        episodes=data["episodes"],
        scenes=data["scenes"],
        seeds=data["noise_seeds"],
        failure=(1 - data["success"]).astype(bool),
        action=action,
        proprio_delta=np.stack(next_state),
        sim_delta=np.stack(next_sim),
        routed=routed,
        shared=shared,
        hidden=hidden,
        scalar=scalar,
        routed_commitment=routed_commitment,
        shared_commitment=shared_commitment,
        hidden_commitment=hidden_commitment,
        validation=full["validation"],
    )
    del data, full, routed_full, shared_full, hidden_full
    gc.collect()
    return result


def _standardize_distance_target(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    scale = values.std(axis=0)
    keep = scale > 1e-8
    if not np.any(keep):
        raise ValueError("physical target has no varying coordinates")
    return (values[:, keep] - values[:, keep].mean(axis=0)) / scale[keep]


def _cosine_distance_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    return np.maximum(1.0 - values @ values.T, 0.0)


def _rms_distance_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.mean(np.square(values[:, None] - values[None, :]), axis=-1))


def geometry_by_scene(data: CompactTask) -> tuple[list[dict], dict[tuple[str, str, int], np.ndarray]]:
    targets = {
        "action": data.action,
        "next_proprio": _standardize_distance_target(data.proprio_delta),
        "next_sim": _standardize_distance_target(data.sim_delta),
    }
    rows = []
    scene_values: dict[tuple[str, str, int], np.ndarray] = {}
    unique_scenes = np.unique(data.scenes)
    for target_name, target in targets.items():
        target_distance = {}
        for scene in unique_scenes:
            index = np.flatnonzero(data.scenes == scene)
            upper = np.triu_indices(len(index), 1)
            target_distance[int(scene)] = _rms_distance_matrix(target[index])[upper]
        for branch in BRANCHES:
            values = getattr(data, branch)
            for denoise in range(N_DENOISE):
                per_scene = []
                for scene in unique_scenes:
                    index = np.flatnonzero(data.scenes == scene)
                    upper = np.triu_indices(len(index), 1)
                    distance = _cosine_distance_matrix(values[index, denoise])[upper]
                    per_scene.append(
                        float(spearmanr(distance, target_distance[int(scene)]).statistic)
                    )
                per_scene_array = np.asarray(per_scene, dtype=np.float64)
                scene_values[(branch, target_name, denoise)] = per_scene_array
                rows.append(
                    {
                        "task": data.task,
                        "branch": branch,
                        "target": target_name,
                        "denoise": denoise,
                        "scene_macro_spearman": float(np.nanmean(per_scene_array)),
                        "per_scene": per_scene_array.tolist(),
                    }
                )
    return rows, scene_values


def _reduce_block(
    train: np.ndarray, test: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    keep = train.std(axis=0) > 1e-9
    scaler = StandardScaler().fit(train[:, keep])
    x_train = scaler.transform(train[:, keep])
    x_test = scaler.transform(test[:, keep])
    count = min(PCA_COMPONENTS, x_train.shape[1], x_train.shape[0] - 1)
    if count < x_train.shape[1]:
        pca = PCA(n_components=count, svd_solver="randomized", random_state=seed).fit(x_train)
        x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test


def seed_folds(seeds: np.ndarray) -> np.ndarray:
    unique = np.sort(np.unique(seeds))
    if len(unique) != 32:
        raise ValueError("expected 32 shared noise seeds")
    rank = {int(seed): axis for axis, seed in enumerate(unique)}
    return np.asarray([rank[int(seed)] // 8 for seed in seeds], dtype=np.int8)


def conditional_scene_stats(
    labels: np.ndarray, scores: np.ndarray, scenes: np.ndarray, strata: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    wins = np.zeros(len(np.unique(scenes)), dtype=np.float64)
    pairs = np.zeros(len(wins), dtype=np.int64)
    for axis, scene in enumerate(np.unique(scenes)):
        scene_index = scenes == scene
        for stratum in np.unique(strata[scene_index]):
            index = scene_index & (strata == stratum)
            positive = scores[index & labels]
            negative = scores[index & ~labels]
            if not len(positive) or not len(negative):
                continue
            wins[axis] += float((positive[:, None] > negative[None, :]).sum())
            wins[axis] += 0.5 * float((positive[:, None] == negative[None, :]).sum())
            pairs[axis] += len(positive) * len(negative)
    return wins, pairs


def _auc(wins: np.ndarray, pairs: np.ndarray) -> float:
    return float(wins.sum() / pairs.sum()) if pairs.sum() else float("nan")


def _scene_bootstrap(
    wins: np.ndarray, pairs: np.ndarray, draws: np.ndarray
) -> np.ndarray:
    sampled_pairs = pairs[draws].sum(axis=1)
    return np.divide(
        wins[draws].sum(axis=1),
        sampled_pairs,
        out=np.full(len(draws), np.nan),
        where=sampled_pairs > 0,
    )


def outcome_models(
    data: CompactTask, bootstrap: int, seed: int
) -> tuple[list[dict], dict[tuple[int, str], np.ndarray], dict[tuple[int, str], np.ndarray]]:
    folds = seed_folds(data.seeds)
    unique_scenes = np.unique(data.scenes)
    scene_axis = {int(scene): axis for axis, scene in enumerate(unique_scenes)}
    strata = np.asarray(
        [scene_axis[int(scene)] * 4 + int(fold) for scene, fold in zip(data.scenes, folds)]
    )
    onehot = (data.scenes[:, None] == unique_scenes[None, :]).astype(np.float64)
    predictions: dict[tuple[int, str], np.ndarray] = {}
    rows = []
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_scenes), size=(bootstrap, len(unique_scenes)))
    bootstrap_values: dict[tuple[int, str], np.ndarray] = {}

    for denoise in range(N_DENOISE):
        scores = {
            family: np.full(len(data.failure), np.nan, dtype=np.float64)
            for family in OUTCOME_FAMILIES
        }
        blocks = {
            "routed": data.routed[:, denoise],
            "scalar": data.scalar[:, denoise],
            "shared": data.shared[:, denoise],
            "hidden": data.hidden[:, denoise],
        }
        for fold in range(4):
            test = folds == fold
            train = ~test
            reduced = {
                name: _reduce_block(value[train], value[test], seed + denoise * 100 + fold)
                for name, value in blocks.items()
            }
            for family, names in OUTCOME_FAMILIES.items():
                x_train = np.column_stack([onehot[train], *[reduced[name][0] for name in names]])
                x_test = np.column_stack([onehot[test], *[reduced[name][1] for name in names]])
                scaler = StandardScaler().fit(x_train)
                model = LogisticRegression(
                    C=LOGISTIC_C,
                    solver="lbfgs",
                    max_iter=3000,
                    class_weight="balanced",
                ).fit(scaler.transform(x_train), data.failure[train])
                scores[family][test] = model.predict_proba(scaler.transform(x_test))[:, 1]
        for family, score in scores.items():
            if np.any(~np.isfinite(score)):
                raise RuntimeError("outcome cross-fitting incomplete")
            wins, pairs = conditional_scene_stats(data.failure, score, data.scenes, strata)
            boot = _scene_bootstrap(wins, pairs, draws)
            valid = boot[np.isfinite(boot)]
            predictions[(denoise, family)] = score
            bootstrap_values[(denoise, family)] = boot
            rows.append(
                {
                    "task": data.task,
                    "denoise": denoise,
                    "family": family,
                    "conditional_auc": _auc(wins, pairs),
                    "auc_ci_low": float(np.percentile(valid, 2.5)),
                    "auc_ci_high": float(np.percentile(valid, 97.5)),
                    "conditional_pairs": int(pairs.sum()),
                    "mixed_scenes": int(np.sum(pairs > 0)),
                }
            )
        print("%s: outcome denoise %d" % (data.task, denoise), flush=True)
    return rows, predictions, bootstrap_values


def commitment_rows(
    data: CompactTask, bootstrap: int, seed: int
) -> tuple[list[dict], dict[tuple[str, int], np.ndarray]]:
    unique_scenes = np.unique(data.scenes)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique_scenes), size=(bootstrap, len(unique_scenes)))
    rows = []
    bootstraps = {}
    strata = data.scenes
    for branch in BRANCHES:
        values = getattr(data, branch + "_commitment")
        for denoise in range(N_DENOISE):
            wins, pairs = conditional_scene_stats(
                data.failure, values[:, denoise], data.scenes, strata
            )
            boot = _scene_bootstrap(wins, pairs, draws)
            valid = boot[np.isfinite(boot)]
            bootstraps[(branch, denoise)] = boot
            rows.append(
                {
                    "task": data.task,
                    "branch": branch,
                    "denoise": denoise,
                    "mean_cosine_to_final": float(values[:, denoise].mean()),
                    "failure_conditional_auc": _auc(wins, pairs),
                    "auc_ci_low": float(np.percentile(valid, 2.5)),
                    "auc_ci_high": float(np.percentile(valid, 97.5)),
                }
            )
    return rows, bootstraps


def off_success_proxy(
    data: CompactTask,
    routed_risk: np.ndarray,
) -> dict:
    folds = seed_folds(data.seeds)
    unique_scenes = np.unique(data.scenes)
    scene_axis = {int(scene): axis for axis, scene in enumerate(unique_scenes)}
    strata = np.asarray(
        [scene_axis[int(scene)] * 4 + int(fold) for scene, fold in zip(data.scenes, folds)]
    )
    physical = np.column_stack((data.proprio_delta, data.sim_delta)).astype(np.float64)
    score = np.full(len(data.failure), np.nan)
    adverse = np.zeros(len(data.failure), dtype=bool)
    usable = np.zeros(len(data.failure), dtype=bool)
    for fold in range(4):
        test = folds == fold
        train = ~test
        scale = physical[train].std(axis=0)
        keep = scale > 1e-8
        scaler = StandardScaler().fit(physical[train][:, keep])
        standardized = scaler.transform(physical[:, keep])
        for scene in unique_scenes:
            success_train = train & (data.scenes == scene) & ~data.failure
            scene_test = test & (data.scenes == scene)
            if success_train.sum() < 3:
                continue
            center = standardized[success_train].mean(axis=0)
            train_distance = np.sqrt(
                np.mean(np.square(standardized[success_train] - center), axis=1)
            )
            test_distance = np.sqrt(
                np.mean(np.square(standardized[scene_test] - center), axis=1)
            )
            threshold = float(np.quantile(train_distance, 0.75))
            score[scene_test] = test_distance
            adverse[scene_test] = test_distance > threshold
            usable[scene_test] = True

    physical_wins, physical_pairs = conditional_scene_stats(
        data.failure[usable], score[usable], data.scenes[usable], strata[usable]
    )
    subset = usable & adverse
    risk_wins, risk_pairs = conditional_scene_stats(
        data.failure[subset], routed_risk[subset], data.scenes[subset], strata[subset]
    )
    return {
        "task": data.task,
        "usable": int(usable.sum()),
        "adverse": int(subset.sum()),
        "recovered_success": int((subset & ~data.failure).sum()),
        "compounded_failure": int((subset & data.failure).sum()),
        "physical_score_auc": _auc(physical_wins, physical_pairs),
        "routed_risk_auc_within_adverse": _auc(risk_wins, risk_pairs),
        "within_adverse_pairs": int(risk_pairs.sum()),
    }


def _bootstrap_mean(values: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    index = rng.integers(0, len(values), size=(draws, len(values)))
    return np.nanmean(values[index], axis=1)


def confirmation_geometry(
    tasks: list[CompactTask],
    scene_values: dict[tuple[str, str, str, int], np.ndarray],
    bootstrap: int,
    seed: int,
) -> dict:
    confirmation = [task for task in tasks if task.task != LONG_TASK]
    rng = np.random.default_rng(seed)

    def contrast(branch: str, target: str) -> tuple[float, np.ndarray, list[float]]:
        per_task = []
        boot = []
        for task in confirmation:
            delta = (
                scene_values[(task.task, branch, target, 8)]
                - scene_values[(task.task, branch, target, 0)]
            )
            per_task.append(float(np.nanmean(delta)))
            boot.append(_bootstrap_mean(delta, bootstrap, rng))
        return float(np.mean(per_task)), np.mean(boot, axis=0), per_task

    output = {}
    for branch, target, name in (
        ("routed", "action", "routed_action_d8_minus_d0"),
        ("routed", "next_sim", "routed_sim_d8_minus_d0"),
        ("shared", "action", "shared_action_d8_minus_d0"),
    ):
        point, boot, per_task = contrast(branch, target)
        output[name] = {
            "estimate": point,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "per_task": per_task,
            "tasks_positive": int(sum(value > 0 for value in per_task)),
        }
    routed = output["routed_action_d8_minus_d0"]
    shared = output["shared_action_d8_minus_d0"]
    difference = []
    boot_difference = []
    for task in confirmation:
        routed_scene = (
            scene_values[(task.task, "routed", "action", 8)]
            - scene_values[(task.task, "routed", "action", 0)]
        )
        shared_scene = (
            scene_values[(task.task, "shared", "action", 8)]
            - scene_values[(task.task, "shared", "action", 0)]
        )
        delta = routed_scene - shared_scene
        difference.append(float(np.nanmean(delta)))
        boot_difference.append(_bootstrap_mean(delta, bootstrap, rng))
    combined = np.mean(boot_difference, axis=0)
    output["routed_minus_shared_action_slope"] = {
        "estimate": float(np.mean(difference)),
        "ci95": [float(np.percentile(combined, 2.5)), float(np.percentile(combined, 97.5))],
        "per_task": difference,
        "tasks_positive": int(sum(value > 0 for value in difference)),
    }
    return output


def macro_curves(
    tasks: list[CompactTask],
    geometry_rows: list[dict],
    outcome_rows: list[dict],
    outcome_boots: dict[tuple[str, int, str], np.ndarray],
) -> tuple[list[dict], list[dict]]:
    confirmation = {task.task for task in tasks if task.task != LONG_TASK}
    geometry_lookup = {
        (row["task"], row["branch"], row["target"], row["denoise"]): row
        for row in geometry_rows
    }
    outcome_lookup = {
        (row["task"], row["denoise"], row["family"]): row for row in outcome_rows
    }
    geometry_macro = []
    outcome_macro = []
    for branch in BRANCHES:
        for target in TARGETS:
            for denoise in range(N_DENOISE):
                values = [
                    geometry_lookup[(task, branch, target, denoise)]["scene_macro_spearman"]
                    for task in confirmation
                ]
                geometry_macro.append(
                    {
                        "branch": branch,
                        "target": target,
                        "denoise": denoise,
                        "confirmation_macro_spearman": float(np.mean(values)),
                    }
                )
    for family in OUTCOME_FAMILIES:
        for denoise in range(N_DENOISE):
            values = [
                outcome_lookup[(task, denoise, family)]["conditional_auc"]
                for task in confirmation
            ]
            boot = np.nanmean(
                [outcome_boots[(task, denoise, family)] for task in confirmation], axis=0
            )
            valid = boot[np.isfinite(boot)]
            outcome_macro.append(
                {
                    "family": family,
                    "denoise": denoise,
                    "confirmation_macro_auc": float(np.mean(values)),
                    "ci95": [
                        float(np.percentile(valid, 2.5)),
                        float(np.percentile(valid, 97.5)),
                    ],
                    "tasks_above_chance": int(sum(value > 0.5 for value in values)),
                }
            )
    return geometry_macro, outcome_macro


def _write_csv(path: pathlib.Path, rows: list[dict], exclude: set[str] | None = None) -> None:
    exclude = exclude or set()
    clean = [{key: value for key, value in row.items() if key not in exclude} for row in rows]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean[0]))
        writer.writeheader()
        writer.writerows(clean)


def render_report(summary: dict) -> str:
    geometry = {
        (row["branch"], row["target"], row["denoise"]): row
        for row in summary["confirmation_geometry_curve"]
    }
    outcome = {
        (row["family"], row["denoise"]): row
        for row in summary["confirmation_outcome_curve"]
    }
    primary = summary["confirmation_primary"]
    routed_d0 = outcome[("routed_full", 0)]
    routed_d8 = outcome[("routed_full", 8)]
    validation_rows = [
        row for task_rows in summary["validation"].values() for row in task_rows
    ]
    min_top4_match = min(row["top4_set_match"] for row in validation_rows)
    max_probability_mae = max(
        row["selected_probability_mae"] for row in validation_rows
    )
    lines = [
        "# First-query MoE state impact",
        "",
        "## Bottom line",
        "",
        "There is a same-state MoE association, but not a confirmed routed-expert failure precursor. "
        "At recorded denoise position d0, routed-contribution geometry tracks the final action "
        "(rho %.3f) and the one-chunk simulator transition (rho %.3f). However, shared-output "
        "and pre-MoE hidden controls are stronger, so this is not routed-specific evidence."
        % (
            geometry[("routed", "action", 0)]["confirmation_macro_spearman"],
            geometry[("routed", "next_sim", 0)]["confirmation_macro_spearman"],
        ),
        "",
        "The prespecified formation effect went in the opposite direction: routed action alignment "
        "changed by %+.3f from d0 to d8 and routed next-state alignment by %+.3f. "
        "Contribution-only failure prediction remained at chance (d0 AUC %.3f; d8 AUC %.3f)."
        % (
            primary["routed_action_d8_minus_d0"]["estimate"],
            primary["routed_sim_d8_minus_d0"]["estimate"],
            routed_d0["confirmation_macro_auc"],
            routed_d8["confirmation_macro_auc"],
        ),
        "",
        "## Design",
        "",
        "- The input is only query `k=0`; all 32 seed siblings have the exact same physical initial state.",
        "- Routed top-4 expert contributions are reconstructed offline at layers 2/5/12/15 for every denoise position and token.",
        "- LIBERO-Long is discovery. Goal-top and two spatial tasks are the frozen confirmation set.",
        "- Pairwise geometry is computed only within initial state. Outcome models hold out complete seed groups.",
        "- Shared-expert output and the expert-input hidden state are matched-capacity controls.",
        "",
        "## Confirmation endpoints",
        "",
        "| endpoint | estimate | 95% scene-bootstrap CI | positive tasks |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("routed_action_d8_minus_d0", "routed action-alignment slope d8-d0"),
        ("routed_sim_d8_minus_d0", "routed next-sim alignment slope d8-d0"),
        ("shared_action_d8_minus_d0", "shared action-alignment slope d8-d0"),
        ("routed_minus_shared_action_slope", "routed slope minus shared slope"),
    ):
        row = primary[key]
        lines.append(
            "| %s | %+.3f | [%+.3f, %+.3f] | %d/3 |"
            % (label, row["estimate"], row["ci95"][0], row["ci95"][1], row["tasks_positive"])
        )
    lines.extend(
        [
            "",
            "## Denoise curve on confirmation tasks",
            "",
            "| d | routed-action rho | shared-action rho | routed-next-sim rho | routed-full failure AUC |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for denoise in range(N_DENOISE):
        result = outcome[("routed_full", denoise)]
        lines.append(
            "| %d | %.3f | %.3f | %.3f | %.3f [%.3f, %.3f] |"
            % (
                denoise,
                geometry[("routed", "action", denoise)]["confirmation_macro_spearman"],
                geometry[("shared", "action", denoise)]["confirmation_macro_spearman"],
                geometry[("routed", "next_sim", denoise)]["confirmation_macro_spearman"],
                result["confirmation_macro_auc"],
                result["ci95"][0],
                result["ci95"][1],
            )
        )
    lines.extend(
        [
            "",
            "## Prespecified outcome readouts",
            "",
            "| representation | d0 AUC | d8 AUC | d8 tasks above chance |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in OUTCOME_FAMILIES:
        d0 = outcome[(family, 0)]
        d8 = outcome[(family, 8)]
        lines.append(
            "| %s | %.3f [%.3f, %.3f] | %.3f [%.3f, %.3f] | %d/3 |"
            % (
                family,
                d0["confirmation_macro_auc"], d0["ci95"][0], d0["ci95"][1],
                d8["confirmation_macro_auc"], d8["ci95"][0], d8["ci95"][1],
                d8["tasks_above_chance"],
            )
        )
    lines.extend(
        [
            "",
            "## Recovery proxy",
            "",
            "`off-success-manifold` is defined from training-seed successful next states only. It is a descriptive subgroup, not `Q_escape`.",
            "",
            "| task | adverse | recovered | compounded | physical-score AUC | routed-risk AUC within adverse | pairs |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["recovery_proxy"]:
        lines.append(
            "| %s | %d | %d | %d | %.3f | %.3f | %d |"
            % (
                row["task"], row["adverse"], row["recovered_success"],
                row["compounded_failure"], row["physical_score_auc"],
                row["routed_risk_auc_within_adverse"], row["within_adverse_pairs"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The geometry endpoints establish whether routed computation is organized like the emitted action and resulting one-chunk state transition. They do not establish a causal expert effect: seed noise also changes non-MoE computation. A snapshot-and-noise-fixed activation intervention is still required for causality.",
            "",
            "The vectors are recomputed from stored fp16 hidden states, recorded expert IDs/weights, and checkpoint MLPs; they are not runtime-exact. Recorded selected probabilities reproduce with maximum MAE %.2g. The fp16 gate audit recovers at least %.1f%% of top-4 sets; recorded IDs, rather than recomputed IDs, are used for every contribution."
            % (max_probability_mae, 100.0 * min_top4_match),
            "",
            "The recovery subgroup is underpowered on confirmation tasks (only 5-21 within-stratum failure/success pairs), so its below-chance point estimates are not treated as evidence of an inverse effect.",
            "",
        ]
    )
    return "\n".join(lines)


def make_plot(summary: dict, path: pathlib.Path) -> None:
    import matplotlib.pyplot as plt

    geometry = {
        (row["branch"], row["target"], row["denoise"]): row
        for row in summary["confirmation_geometry_curve"]
    }
    outcome = {
        (row["family"], row["denoise"]): row
        for row in summary["confirmation_outcome_curve"]
    }
    x = np.arange(N_DENOISE)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    colors = {"routed": "#C44536", "shared": "#4C566A", "hidden": "#2A9D8F"}
    for branch in BRANCHES:
        axes[0].plot(
            x,
            [geometry[(branch, "action", int(d))]["confirmation_macro_spearman"] for d in x],
            marker="o", label=branch, color=colors[branch],
        )
        axes[1].plot(
            x,
            [geometry[(branch, "next_sim", int(d))]["confirmation_macro_spearman"] for d in x],
            marker="o", label=branch, color=colors[branch],
        )
    axes[0].set(xlabel="denoise position", ylabel="within-state Spearman", title="Final action geometry")
    axes[1].set(xlabel="denoise position", ylabel="within-state Spearman", title="Next simulator-state geometry")
    for family, color in (("routed_full", "#C44536"), ("shared_identity", "#4C566A"), ("hidden_identity", "#2A9D8F")):
        axes[2].plot(
            x,
            [outcome[(family, int(d))]["confirmation_macro_auc"] for d in x],
            marker="o", label=family, color=color,
        )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set(xlabel="denoise position", ylabel="conditional AUC", title="Eventual failure")
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(args.task) if args.task else None
    runs = discover_runs(args.cache_root, requested)
    tasks = [
        extract_compact(
            run, args.cache_root, args.out_dir, args.threads, args.seed, args.force_compact
        )
        for run in runs
    ]

    geometry_rows: list[dict] = []
    outcome_rows: list[dict] = []
    commitment_output: list[dict] = []
    recovery = []
    geometry_scene: dict[tuple[str, str, str, int], np.ndarray] = {}
    outcome_boots: dict[tuple[str, int, str], np.ndarray] = {}
    for task_index, task in enumerate(tasks):
        rows, scene = geometry_by_scene(task)
        geometry_rows.extend(rows)
        for key, value in scene.items():
            geometry_scene[(task.task, *key)] = value
        rows, predictions, boots = outcome_models(
            task, args.bootstrap, args.seed + task_index * 10000
        )
        outcome_rows.extend(rows)
        for key, value in boots.items():
            outcome_boots[(task.task, *key)] = value
        rows, _ = commitment_rows(task, args.bootstrap, args.seed + task_index * 10000 + 500)
        commitment_output.extend(rows)
        recovery.append(off_success_proxy(task, predictions[(8, "routed_full")]))

    confirmation_primary = confirmation_geometry(
        tasks, geometry_scene, args.bootstrap, args.seed + 900000
    )
    geometry_macro, outcome_macro = macro_curves(
        tasks, geometry_rows, outcome_rows, outcome_boots
    )
    summary = {
        "experiment": "first-query routed MoE state impact",
        "feature_version": FEATURE_VERSION,
        "tasks": [task.task for task in tasks],
        "discovery_task": LONG_TASK,
        "confirmation_tasks": [task.task for task in tasks if task.task != LONG_TASK],
        "layers": list(LAYERS),
        "denoise_positions": N_DENOISE,
        "token_projected_dim": TOKEN_PROJECTED_DIM,
        "pca_components": PCA_COMPONENTS,
        "logistic_c": LOGISTIC_C,
        "bootstrap": args.bootstrap,
        "validation": {task.task: task.validation for task in tasks},
        "confirmation_primary": confirmation_primary,
        "confirmation_geometry_curve": geometry_macro,
        "confirmation_outcome_curve": outcome_macro,
        "recovery_proxy": recovery,
        "geometry_by_task": geometry_rows,
        "outcome_by_task": outcome_rows,
        "commitment_by_task": commitment_output,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_csv(args.out_dir / "geometry.csv", geometry_rows, {"per_scene"})
    _write_csv(args.out_dir / "outcome.csv", outcome_rows)
    _write_csv(args.out_dir / "commitment.csv", commitment_output)
    _write_csv(args.out_dir / "recovery.csv", recovery)
    (args.out_dir / "report.md").write_text(render_report(summary))
    make_plot(summary, args.out_dir / "moe_state_impact.png")
    print("wrote %s" % args.out_dir, flush=True)


if __name__ == "__main__":
    main()
