#!/usr/bin/env python3
"""Test whether early within-query MoE denoising predicts eventual timeout.

The primary corpus contains 32 flow-noise rollouts for each of 16 initial
states in five LIBERO tasks.  The analysis keeps the first four absolute
policy queries and all ten within-query denoising forwards.  Outcome contrasts
are made within task x initial-state strata.  A second, independent contrast
uses the rolling-star K=16 branches within exact simulator snapshots.

No episode length, relative phase, remaining time, terminal feature, hard
expert ID, or hidden state is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import analyze_initial_state_early_moe_signal as initial
import analyze_rolling_star_experiment as rolling
import analyze_state_route_layer_denoise_effects as layer


HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
DEFAULT_ROLLING = HERE / "runs/rolling-star-a100-long-t08-k16-20260828"
DEFAULT_OUT = HERE / "analysis/early-chunk-denoise-timeout-20260829"

SEED = 20260829
QUERIES = (0, 1, 2, 3)
N_DENOISE = 10
N_ACTION_TOKENS = 10
N_EXPERTS = 32
MAX_SIM_STATE = 92
GEOMETRY_WIDTH = 8 + MAX_SIM_STATE
CONTROL_WIDTH = GEOMETRY_WIDTH + 70
LAYER_GROUPS = {
    "front_2_5": (0, 1, 2, 3),
    "back_12_15": (4, 5, 6, 7),
}

TASKS = (
    "libero_goal/open_the_middle_drawer_of_the_cabinet",
    "libero_goal/open_the_top_drawer_and_put_the_bowl_inside",
    "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    "libero_spatial/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "libero_spatial/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
)
SHORT_TASK = {
    TASKS[0]: "middle_drawer",
    TASKS[1]: "top_drawer_then_bowl",
    TASKS[2]: "two_moka_pots",
    TASKS[3]: "bowl_from_ramekin",
    TASKS[4]: "bowl_from_stove",
}
VARIABLE_TASKS = TASKS[1:]

DENOISE_WINDOWS = {
    "q0_d0_d2": ((0,), (0, 1, 2)),
    "q0_d3_d6": ((0,), (3, 4, 5, 6)),
    "q0_d7_d9": ((0,), (7, 8, 9)),
    "q0_all_d": ((0,), tuple(range(10))),
    "q0_q1_all_d": ((0, 1), tuple(range(10))),
    "q0_q2_all_d": ((0, 1, 2), tuple(range(10))),
    "q0_q3_all_d": ((0, 1, 2, 3), tuple(range(10))),
}


@dataclass(frozen=True)
class RouteFeature:
    key: str
    family: str
    query: int
    layer_group: str
    metric: str
    denoise_step: int
    feature_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rolling-root", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
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
    return value


def normalize(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid routing probabilities")
    return result / mass


def entropy(probability: np.ndarray) -> np.ndarray:
    return -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=-1
    ) / np.log(probability.shape[-1])


def top4_mass(probability: np.ndarray) -> np.ndarray:
    return np.partition(probability, -4, axis=-1)[..., -4:].sum(axis=-1)


def p4p5_gap(probability: np.ndarray) -> np.ndarray:
    top5 = np.sort(np.partition(probability, -5, axis=-1)[..., -5:], axis=-1)
    return top5[..., 1] - top5[..., 0]


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5
        * np.sum(
            np.square(np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))),
            axis=-1,
        )
    )


def extract_route_features(
    routes: np.ndarray,
) -> tuple[np.ndarray, list[RouteFeature], np.ndarray, list[str], float]:
    """Convert [episode, query, 8, 10, 11, 32] routes to invariant cells."""
    if routes.shape[1:] != (len(QUERIES), 8, 10, 11, 32):
        raise ValueError(f"unexpected route tensor {routes.shape}")
    columns: list[np.ndarray] = []
    metadata: list[RouteFeature] = []
    state_columns: list[np.ndarray] = []
    state_keys: list[str] = []
    state_span = 0.0

    for query_axis, query in enumerate(QUERIES):
        state_all = normalize(routes[:, query_axis, :, :, 0, :])
        state_span = max(
            state_span,
            float(np.max(np.abs(state_all - state_all[:, :, :1, :]))),
        )
        for group_name, layer_axes in LAYER_GROUPS.items():
            state = state_all[:, layer_axes, 0, :]
            state_metrics = {
                "entropy": entropy(state).mean(axis=1),
                "top1": state.max(axis=-1).mean(axis=1),
                "top4": top4_mass(state).mean(axis=1),
                "p4p5_gap": p4p5_gap(state).mean(axis=1),
            }
            for metric, values in state_metrics.items():
                state_columns.append(values)
                state_keys.append(f"q{query}|state|{group_name}|{metric}")

            action = normalize(
                routes[:, query_axis, layer_axes, :, 1:, :]
            )
            token_center = action.mean(axis=3, keepdims=True)
            level_metrics = {
                "entropy": entropy(action).mean(axis=(1, 3)),
                "top1": action.max(axis=-1).mean(axis=(1, 3)),
                "top4": top4_mass(action).mean(axis=(1, 3)),
                "p4p5_gap": p4p5_gap(action).mean(axis=(1, 3)),
                "token_dispersion": hellinger(action, token_center).mean(axis=(1, 3)),
            }
            for metric, values in level_metrics.items():
                for denoise in range(N_DENOISE):
                    key = f"q{query}|action|{group_name}|{metric}|d{denoise}"
                    columns.append(values[:, denoise])
                    metadata.append(
                        RouteFeature(
                            key=key,
                            family=f"q{query}_level",
                            query=query,
                            layer_group=group_name,
                            metric=metric,
                            denoise_step=denoise,
                            feature_kind="level",
                        )
                    )
            motion = hellinger(action[:, :, 1:], action[:, :, :-1]).mean(axis=(1, 3))
            for transition in range(1, N_DENOISE):
                key = f"q{query}|action|{group_name}|denoise_motion|d{transition - 1}_to_d{transition}"
                columns.append(motion[:, transition - 1])
                metadata.append(
                    RouteFeature(
                        key=key,
                        family=f"q{query}_motion",
                        query=query,
                        layer_group=group_name,
                        metric="denoise_motion",
                        denoise_step=transition,
                        feature_kind="motion",
                    )
                )
    return (
        np.column_stack(columns).astype(np.float32),
        metadata,
        np.column_stack(state_columns).astype(np.float32),
        state_keys,
        state_span,
    )


def route_row_indices(
    store: zarr.Group, episodes: list[int]
) -> np.ndarray:
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    indices = []
    for episode in episodes:
        rows = np.flatnonzero(episode_ids == episode)[: len(QUERIES)]
        if len(rows) != len(QUERIES):
            raise ValueError(f"episode {episode} is missing q0-q3")
        indices.extend(rows.tolist())
    return np.asarray(indices, dtype=np.int64)


def load_old_task(
    cache_root: Path, task: str
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[RouteFeature],
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    run = cache_root / task / "right-16x32"
    summaries = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    if len(summaries) != 512:
        raise ValueError(f"{task}: expected 512 summaries, got {len(summaries)}")
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    episodes = [int(row["episode_index"]) for row in summaries]
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    indices = route_row_indices(store, episodes)
    routes = np.asarray(store["hb_router_probs"].oindex[indices]).reshape(
        len(episodes), len(QUERIES), 8, 10, 11, 32
    )
    route_features, route_meta, state_features, state_keys, state_span = (
        extract_route_features(routes)
    )
    del routes

    controls = np.zeros(
        (len(episodes), len(QUERIES), CONTROL_WIDTH), dtype=np.float32
    )
    geometry = np.zeros(
        (len(episodes), len(QUERIES), GEOMETRY_WIDTH), dtype=np.float32
    )
    action = np.empty((len(episodes), len(QUERIES), 70), dtype=np.float32)
    for row, episode in enumerate(episodes):
        path = run / "client" / f"episode_{episode:02d}.npz"
        with np.load(path, allow_pickle=False) as data:
            policy_state = np.asarray(data["state"][: len(QUERIES)], dtype=np.float32)
            sim_state = np.asarray(data["sim_state"][: len(QUERIES)], dtype=np.float32)
            action_chunk = np.asarray(data["actions"][: len(QUERIES)], dtype=np.float32).reshape(
                len(QUERIES), -1
            )
        if (
            policy_state.shape != (4, 8)
            or sim_state.shape[0] != 4
            or sim_state.shape[1] > MAX_SIM_STATE
            or action_chunk.shape != (4, 70)
        ):
            raise ValueError(f"{path}: unexpected early control shape")
        geometry[row, :, :8] = policy_state
        geometry[row, :, 8 : 8 + sim_state.shape[1]] = sim_state
        action[row] = action_chunk
        controls[row, :, :GEOMETRY_WIDTH] = geometry[row]
        controls[row, :, GEOMETRY_WIDTH:] = action_chunk

    frame = pd.DataFrame(
        {
            "source": "right16x32",
            "task": task,
            "short_task": SHORT_TASK[task],
            "episode": episodes,
            "init_state_id": [int(row["init_state_id"]) for row in summaries],
            "flow_noise_seed": [int(row["flow_noise_seed"]) for row in summaries],
            "failure": [not bool(row["success"]) for row in summaries],
            "inference_calls": [int(row["inference_calls"]) for row in summaries],
        }
    )
    frame["stratum"] = frame["task"] + "|i" + frame["init_state_id"].astype(str)
    return (
        frame,
        route_features,
        route_meta,
        state_features,
        state_keys,
        controls,
        geometry,
        action,
        state_span,
    )


def load_old_corpus(
    cache_root: Path,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[RouteFeature],
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
    frames = []
    routes = []
    states = []
    controls = []
    geometries = []
    actions = []
    route_meta: list[RouteFeature] | None = None
    state_keys: list[str] | None = None
    spans: dict[str, float] = {}
    for task in TASKS:
        loaded = load_old_task(cache_root, task)
        frame, route, meta, state, keys, control, geometry, action, span = loaded
        if route_meta is None:
            route_meta = meta
            state_keys = keys
        elif route_meta != meta or state_keys != keys:
            raise ValueError("route feature schemas differ across tasks")
        frames.append(frame)
        routes.append(route)
        states.append(state)
        controls.append(control)
        geometries.append(geometry)
        actions.append(action)
        spans[SHORT_TASK[task]] = span
        print(f"loaded old corpus: {SHORT_TASK[task]}", flush=True)
    assert route_meta is not None and state_keys is not None
    return (
        pd.concat(frames, ignore_index=True),
        np.vstack(routes),
        route_meta,
        np.vstack(states),
        state_keys,
        np.vstack(controls),
        np.vstack(geometries),
        np.vstack(actions),
        spans,
    )


def load_rolling_corpus(
    run_root: Path,
) -> tuple[pd.DataFrame, np.ndarray, list[RouteFeature], np.ndarray, list[str], float]:
    candidates, _ = rolling.discover_candidates(run_root)
    store = zarr.open_group(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    by_episode = {
        candidate.episode_id: np.flatnonzero(episode_ids == candidate.episode_id)[: len(QUERIES)]
        for candidate in candidates
    }
    if any(len(index) != len(QUERIES) for index in by_episode.values()):
        raise ValueError("rolling candidate missing q0-q3 route rows")
    indices = np.concatenate([by_episode[candidate.episode_id] for candidate in candidates])
    routes = np.asarray(store["hb_router_probs"].oindex[indices]).reshape(
        len(candidates), len(QUERIES), 8, 10, 11, 32
    )
    route_features, route_meta, state_features, state_keys, state_span = (
        extract_route_features(routes)
    )
    frame = pd.DataFrame(
        {
            "source": "rolling_star",
            "task": "two_moka_pots",
            "short_task": "two_moka_pots",
            "episode": [item.episode_id for item in candidates],
            "worker": [item.worker for item in candidates],
            "snapshot": [item.snapshot for item in candidates],
            "candidate": [item.candidate for item in candidates],
            "snapshot_key": [item.snapshot_key for item in candidates],
            "failure": [not item.success for item in candidates],
            "inference_calls": [item.inference_calls for item in candidates],
        }
    )
    frame["stratum"] = frame["snapshot_key"]
    return frame, route_features, route_meta, state_features, state_keys, state_span


def to_layer_metadata(metadata: list[RouteFeature]) -> list[layer.FeatureMeta]:
    return [
        layer.FeatureMeta(
            key=item.key,
            family=item.family,
            token_role="action",
            scope="early_query_denoise",
            metric=item.metric,
            layer_group=item.layer_group,
            denoise_step=item.denoise_step,
            relative_query=item.query,
        )
        for item in metadata
    ]


def mixed_mask(frame: pd.DataFrame) -> np.ndarray:
    mixed = frame.groupby("stratum")["failure"].nunique()
    return frame["stratum"].isin(mixed.index[mixed == 2]).to_numpy()


def effect_scan(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[RouteFeature],
    permutations: int,
    seed: int,
    bootstraps: int,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mask = mixed_mask(frame)
    subset = frame[mask].reset_index(drop=True)
    selected = values[mask]
    labels = subset["failure"].to_numpy(dtype=np.int64)
    groups, group_names = pd.factorize(subset["stratum"], sort=True)
    beta, residual_sd, effect, _ = layer.fixed_effect_shift(selected, labels, groups)
    raw_p, family_p, global_p = layer.permutation_scan(
        selected,
        labels,
        groups,
        to_layer_metadata(metadata),
        permutations,
        seed,
    )
    top = np.argsort(-np.abs(effect))[: min(40, len(effect))]
    bootstrap = layer.bootstrap_top_effects(
        selected, labels, groups, top, bootstraps, seed + 1
    )
    ci_low = np.full(len(effect), np.nan)
    ci_high = np.full(len(effect), np.nan)
    ci_low[top] = np.quantile(bootstrap, 0.025, axis=0)
    ci_high[top] = np.quantile(bootstrap, 0.975, axis=0)
    rows = []
    for index, item in enumerate(metadata):
        rows.append(
            {
                "dataset": dataset,
                **asdict(item),
                "failure_mean": float(selected[labels == 1, index].mean()),
                "success_mean": float(selected[labels == 0, index].mean()),
                "adjusted_difference": beta[index],
                "residual_sd": residual_sd[index],
                "effect_sigma": effect[index],
                "effect_ci95_low": ci_low[index],
                "effect_ci95_high": ci_high[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_family": family_p[index],
                "permutation_p_max_global": global_p[index],
            }
        )
    return pd.DataFrame(rows), {
        "dataset": dataset,
        "episodes": len(subset),
        "failures": int(labels.sum()),
        "successes": int((labels == 0).sum()),
        "mixed_strata": int(len(group_names)),
        "features": len(metadata),
        "permutations": permutations,
        "bootstraps": bootstraps,
    }


def task_effects(
    frame: pd.DataFrame, values: np.ndarray, metadata: list[RouteFeature]
) -> pd.DataFrame:
    rows = []
    for task in VARIABLE_TASKS:
        task_mask = frame["task"].eq(task).to_numpy()
        task_frame = frame[task_mask].reset_index(drop=True)
        mask = mixed_mask(task_frame)
        subset = task_frame[mask]
        selected = values[task_mask][mask]
        labels = subset["failure"].to_numpy(dtype=np.int64)
        groups, _ = pd.factorize(subset["stratum"], sort=True)
        beta, _, effect, _ = layer.fixed_effect_shift(selected, labels, groups)
        for index, item in enumerate(metadata):
            rows.append(
                {
                    "task": task,
                    "short_task": SHORT_TASK[task],
                    "key": item.key,
                    "query": item.query,
                    "layer_group": item.layer_group,
                    "metric": item.metric,
                    "denoise_step": item.denoise_step,
                    "adjusted_difference": beta[index],
                    "effect_sigma": effect[index],
                    "episodes": len(subset),
                    "failures": int(labels.sum()),
                    "mixed_initial_states": int(len(np.unique(groups))),
                }
            )
    return pd.DataFrame(rows)


def window_columns(metadata: list[RouteFeature], window: str) -> np.ndarray:
    queries, denoise_steps = DENOISE_WINDOWS[window]
    denoise_set = set(denoise_steps)
    selected = []
    for index, item in enumerate(metadata):
        if item.query not in queries:
            continue
        if item.feature_kind == "level" and item.denoise_step in denoise_set:
            selected.append(index)
        elif item.feature_kind == "motion":
            if item.denoise_step in denoise_set and item.denoise_step - 1 in denoise_set:
                selected.append(index)
    if not selected:
        raise ValueError(f"window {window} selected no route features")
    return np.asarray(selected, dtype=np.int64)


def assign_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["heldout_seed_fold"] = -1
    result["heldout_init_fold"] = -1
    for task in VARIABLE_TASKS:
        mask = result["task"].eq(task)
        seeds = sorted(result.loc[mask, "flow_noise_seed"].unique())
        states = sorted(result.loc[mask, "init_state_id"].unique())
        if len(seeds) != 32 or len(states) != 16:
            raise ValueError(f"{task}: expected 32 seeds and 16 states")
        seed_fold = {value: index // 4 for index, value in enumerate(seeds)}
        init_fold = {value: index // 2 for index, value in enumerate(states)}
        result.loc[mask, "heldout_seed_fold"] = result.loc[mask, "flow_noise_seed"].map(seed_fold)
        result.loc[mask, "heldout_init_fold"] = result.loc[mask, "init_state_id"].map(init_fold)
    return result


def reduce_block(
    train: np.ndarray, test: np.ndarray, components: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    count = min(components, train.shape[0] - 1, train.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=seed)
    return pca.fit_transform(train_scaled), pca.transform(test_scaled)


def early_outcome_predictions(
    frame: pd.DataFrame,
    route_values: np.ndarray,
    metadata: list[RouteFeature],
    controls: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    variable_mask = frame["task"].isin(VARIABLE_TASKS).to_numpy()
    frame = assign_folds(frame[variable_mask].reset_index(drop=True))
    route_values = route_values[variable_mask]
    controls = controls[variable_mask]
    rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(VARIABLE_TASKS):
        task_mask = frame["task"].eq(task).to_numpy()
        for split in ("heldout_seed", "heldout_init"):
            fold_column = f"{split}_fold"
            include_init = split == "heldout_seed"
            for fold in range(8):
                test_index = np.flatnonzero(task_mask & frame[fold_column].eq(fold).to_numpy())
                train_index = np.flatnonzero(task_mask & ~frame[fold_column].eq(fold).to_numpy())
                y_train = frame.loc[train_index, "failure"].to_numpy(dtype=bool)
                train_states = frame.loc[train_index, "init_state_id"].to_numpy(dtype=np.int64)
                test_states = frame.loc[test_index, "init_state_id"].to_numpy(dtype=np.int64)
                baseline_train, baseline_test = initial.prior_scores(
                    y_train, train_states, test_states, include_init
                )
                for window_index, window in enumerate(DENOISE_WINDOWS):
                    route_columns = window_columns(metadata, window)
                    route_train, route_test = reduce_block(
                        route_values[train_index][:, route_columns],
                        route_values[test_index][:, route_columns],
                        12,
                        seed + task_index * 1000 + fold * 20 + window_index,
                    )
                    query_count = max(DENOISE_WINDOWS[window][0]) + 1
                    control_train, control_test = reduce_block(
                        controls[train_index, :query_count].reshape(len(train_index), -1),
                        controls[test_index, :query_count].reshape(len(test_index), -1),
                        12,
                        seed + 50000 + task_index * 1000 + fold * 20 + window_index,
                    )
                    models = {
                        "baseline": (None, None),
                        "route": (route_train, route_test),
                        "controls": (control_train, control_test),
                        "controls_route": (
                            np.column_stack((control_train, route_train)),
                            np.column_stack((control_test, route_test)),
                        ),
                    }
                    for model, blocks in models.items():
                        if blocks[0] is None:
                            scores = baseline_test
                        else:
                            _, scores, _ = initial.offset_logistic_scores(
                                blocks[0],
                                y_train,
                                blocks[1],
                                baseline_train,
                                baseline_test,
                                0.1,
                            )
                        for position, row_index in enumerate(test_index):
                            row = frame.iloc[row_index]
                            rows.append(
                                {
                                    "task": task,
                                    "short_task": SHORT_TASK[task],
                                    "split": split,
                                    "fold": fold,
                                    "window": window,
                                    "model": model,
                                    "episode": int(row["episode"]),
                                    "init_state_id": int(row["init_state_id"]),
                                    "flow_noise_seed": int(row["flow_noise_seed"]),
                                    "truth": bool(row["failure"]),
                                    "score": float(scores[position]),
                                }
                            )
    return pd.DataFrame(rows)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "window", "model"]
    for values, group in predictions.groupby(keys, sort=False):
        per_task = []
        for _, task_group in group.groupby("task"):
            truth = task_group["truth"].to_numpy(dtype=float)
            score = task_group["score"].to_numpy(dtype=float)
            per_task.append(float(np.mean(np.square(score - truth))))
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "episodes": len(group),
                "failures": int(group["truth"].sum()),
                "brier_task_macro": float(np.mean(per_task)),
                "brier_task_min": float(np.min(per_task)),
                "brier_task_max": float(np.max(per_task)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_prediction_deltas(
    predictions: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comparisons = {
        "route_vs_baseline": ("baseline", "route"),
        "controls_vs_baseline": ("baseline", "controls"),
        "route_beyond_controls": ("controls", "controls_route"),
    }
    rows = []
    index_columns = [
        "task",
        "short_task",
        "split",
        "window",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "truth",
    ]
    wide = predictions.pivot(index=index_columns, columns="model", values="score").reset_index()
    for (split, window), group in wide.groupby(["split", "window"], sort=False):
        truth = group["truth"].to_numpy(dtype=float)
        tasks = group["task"].to_numpy()
        states = group["init_state_id"].to_numpy()
        cluster_keys = np.asarray([f"{task}|{state}" for task, state in zip(tasks, states, strict=True)])
        task_clusters = {
            task: np.unique(cluster_keys[tasks == task]) for task in np.unique(tasks)
        }
        for comparison, (baseline, candidate) in comparisons.items():
            delta = np.square(group[baseline].to_numpy() - truth) - np.square(
                group[candidate].to_numpy() - truth
            )
            point_by_task = [float(delta[tasks == task].mean()) for task in np.unique(tasks)]
            point = float(np.mean(point_by_task))
            samples = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                task_values = []
                for task, clusters in task_clusters.items():
                    picked = rng.choice(clusters, size=len(clusters), replace=True)
                    values = [delta[cluster_keys == cluster] for cluster in picked]
                    task_values.append(float(np.concatenate(values).mean()))
                samples[draw] = float(np.mean(task_values))
            rows.append(
                {
                    "split": split,
                    "window": window,
                    "comparison": comparison,
                    "brier_gain": point,
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "tasks_positive": int(np.sum(np.asarray(point_by_task) > 0)),
                    "tasks_total": len(point_by_task),
                }
            )
    return pd.DataFrame(rows)


def difficulty_frame(
    frame: pd.DataFrame,
    route_values: np.ndarray,
    state_values: np.ndarray,
    geometry: np.ndarray,
    action: np.ndarray,
    metadata: list[RouteFeature],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    q0_columns = np.asarray([index for index, item in enumerate(metadata) if item.query == 0])
    rows = []
    feature_blocks: dict[str, list[np.ndarray]] = {
        "moe_single": [],
        "moe_cloud": [],
        "geometry": [],
        "action_single": [],
        "action_cloud": [],
    }
    for (task, init_state), group in frame.groupby(["task", "init_state_id"], sort=True):
        index = group.index.to_numpy()[np.argsort(group["flow_noise_seed"].to_numpy())]
        if len(index) != 32:
            raise ValueError(f"{task} init {init_state}: expected 32 rollouts")
        feature_index = index[:16]
        target_index = index[16:]
        canonical = feature_index[0]
        q0_route = route_values[feature_index][:, q0_columns]
        q0_state = state_values[canonical, :8]
        q0_action = action[feature_index, 0]
        feature_blocks["moe_single"].append(np.r_[q0_state, route_values[canonical, q0_columns]])
        feature_blocks["moe_cloud"].append(
            np.r_[q0_state, q0_route.mean(axis=0), q0_route.std(axis=0)]
        )
        feature_blocks["geometry"].append(geometry[index[0], 0])
        feature_blocks["action_single"].append(action[canonical, 0])
        feature_blocks["action_cloud"].append(
            np.r_[q0_action.mean(axis=0), q0_action.std(axis=0)]
        )
        rows.append(
            {
                "task": task,
                "short_task": SHORT_TASK[task],
                "init_state_id": int(init_state),
                "feature_rollouts": len(feature_index),
                "target_rollouts": len(target_index),
                "failure_rate": float(frame.loc[target_index, "failure"].mean()),
                "feature_half_failure_rate": float(
                    frame.loc[feature_index, "failure"].mean()
                ),
                "all32_failure_rate": float(frame.loc[index, "failure"].mean()),
                "canonical_seed": int(frame.loc[canonical, "flow_noise_seed"]),
            }
        )
    result_blocks = {
        key: np.vstack(values).astype(np.float32) for key, values in feature_blocks.items()
    }
    return pd.DataFrame(rows), result_blocks


def difficulty_models(
    frame: pd.DataFrame, blocks: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_blocks = {
        "task_mean": (),
        "geometry": ("geometry",),
        "moe_single": ("moe_single",),
        "moe_cloud": ("moe_cloud",),
        "action_cloud": ("action_cloud",),
        "geometry_action": ("geometry", "action_cloud"),
        "geometry_moe": ("geometry", "moe_cloud"),
        "geometry_action_moe": ("geometry", "action_cloud", "moe_cloud"),
    }
    rows = []
    for task_index, task in enumerate(TASKS):
        task_indices = np.flatnonzero(frame["task"].eq(task).to_numpy())
        for test_index in task_indices:
            train_index = task_indices[task_indices != test_index]
            y_train = frame.loc[train_index, "failure_rate"].to_numpy(dtype=float)
            truth = float(frame.loc[test_index, "failure_rate"])
            for model_index, (model, names) in enumerate(model_blocks.items()):
                if not names or np.allclose(y_train, y_train[0]):
                    score = float(y_train.mean())
                else:
                    train_parts = []
                    test_parts = []
                    for block_index, name in enumerate(names):
                        train_reduced, test_reduced = reduce_block(
                            blocks[name][train_index],
                            blocks[name][[test_index]],
                            6,
                            SEED + task_index * 1000 + model_index * 50 + block_index,
                        )
                        train_parts.append(train_reduced)
                        test_parts.append(test_reduced)
                    x_train = np.column_stack(train_parts)
                    x_test = np.column_stack(test_parts)
                    model_fit = RidgeCV(alphas=np.asarray([0.1, 1.0, 10.0, 100.0]))
                    model_fit.fit(x_train, y_train)
                    score = float(np.clip(model_fit.predict(x_test)[0], 0.0, 1.0))
                rows.append(
                    {
                        "task": task,
                        "short_task": SHORT_TASK[task],
                        "init_state_id": int(frame.loc[test_index, "init_state_id"]),
                        "truth": truth,
                        "model": model,
                        "score": score,
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = []
    for model, group in predictions.groupby("model", sort=False):
        task_mse = []
        for _, task_group in group.groupby("task"):
            task_mse.append(
                float(np.mean(np.square(task_group["score"] - task_group["truth"])))
            )
        metrics.append(
            {
                "model": model,
                "initial_states": len(group),
                "mse_task_macro": float(np.mean(task_mse)),
                "mse_task_min": float(np.min(task_mse)),
                "mse_task_max": float(np.max(task_mse)),
            }
        )
    return predictions, pd.DataFrame(metrics)


def difficulty_feature_inputs(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[RouteFeature],
) -> tuple[
    list[RouteFeature], dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray
]:
    q0_columns = np.asarray([index for index, item in enumerate(metadata) if item.query == 0])
    q0_meta = [metadata[index] for index in q0_columns]
    feature_rows: dict[str, list[np.ndarray]] = {
        "single_seed_a_to_other31": [],
        "single_seed_b_to_other31": [],
        "cloud_mean_16a_to_16b": [],
        "cloud_mean_16b_to_16a": [],
    }
    targets: dict[str, list[float]] = {key: [] for key in feature_rows}
    task_rows = []
    for _, group in frame.groupby(["task", "init_state_id"], sort=True):
        index = group.index.to_numpy()[np.argsort(group["flow_noise_seed"].to_numpy())]
        if len(index) != 32:
            raise ValueError("difficulty scan requires exactly 32 seeds per initial state")
        half_a = index[:16]
        half_b = index[16:]
        seed_a = half_a[0]
        seed_b = half_b[0]
        feature_rows["single_seed_a_to_other31"].append(values[seed_a, q0_columns])
        feature_rows["single_seed_b_to_other31"].append(values[seed_b, q0_columns])
        feature_rows["cloud_mean_16a_to_16b"].append(
            values[half_a][:, q0_columns].mean(axis=0)
        )
        feature_rows["cloud_mean_16b_to_16a"].append(
            values[half_b][:, q0_columns].mean(axis=0)
        )
        targets["single_seed_a_to_other31"].append(
            float(frame.loc[index[index != seed_a], "failure"].mean())
        )
        targets["single_seed_b_to_other31"].append(
            float(frame.loc[index[index != seed_b], "failure"].mean())
        )
        targets["cloud_mean_16a_to_16b"].append(
            float(frame.loc[half_b, "failure"].mean())
        )
        targets["cloud_mean_16b_to_16a"].append(
            float(frame.loc[half_a, "failure"].mean())
        )
        task_rows.append(str(group["task"].iloc[0]))
    representations = {
        key: (
            np.vstack(rows).astype(np.float64),
            np.asarray(targets[key], dtype=np.float64),
        )
        for key, rows in feature_rows.items()
    }
    return q0_meta, representations, np.asarray(task_rows)


def continuous_difficulty_scan(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[RouteFeature],
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    q0_meta, representations, tasks = difficulty_feature_inputs(frame, values, metadata)
    rng = np.random.default_rng(seed)
    rows = []
    for representation, (matrix, target) in representations.items():
        y_residual = layer.residualize_by_group(target[:, None], tasks)[:, 0]
        y_scale = float(np.sqrt(np.sum(np.square(y_residual))))
        permuted_y = np.empty((permutations, len(target)), dtype=np.float32)
        for draw in range(permutations):
            permuted = target.copy()
            for task in np.unique(tasks):
                selected = np.flatnonzero(tasks == task)
                permuted[selected] = rng.permutation(permuted[selected])
            permuted_y[draw] = layer.residualize_by_group(
                permuted[:, None], tasks
            )[:, 0]
        null_y_scale = np.sqrt(np.sum(np.square(permuted_y), axis=1, keepdims=True))
        x_residual = layer.residualize_by_group(matrix, tasks)
        x_scale = np.sqrt(np.sum(np.square(x_residual), axis=0))
        observed = (y_residual @ x_residual) / np.maximum(
            x_scale * y_scale, 1e-12
        )
        null = (permuted_y @ x_residual) / np.maximum(
            null_y_scale * x_scale[None, :], 1e-12
        )
        absolute = np.abs(null)
        raw_p = (1 + np.sum(absolute >= np.abs(observed)[None, :], axis=0)) / (
            permutations + 1
        )
        max_p = (
            1
            + np.sum(
                absolute.max(axis=1)[:, None] >= np.abs(observed)[None, :], axis=0
            )
        ) / (permutations + 1)
        for index, item in enumerate(q0_meta):
            rows.append(
                {
                    "representation": representation,
                    **asdict(item),
                    "difficulty_correlation_task_residual": observed[index],
                    "permutation_p_raw": raw_p[index],
                    "permutation_p_max_global": max_p[index],
                }
            )
    result = pd.DataFrame(rows)
    result["permutation_p_across_4_representations"] = np.minimum(
        1.0, result["permutation_p_max_global"] * len(representations)
    )
    return result


def difficulty_task_correlations(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[RouteFeature],
    effects: pd.DataFrame,
) -> pd.DataFrame:
    q0_meta, representations, tasks = difficulty_feature_inputs(frame, values, metadata)
    key_to_column = {item.key: index for index, item in enumerate(q0_meta)}
    rows = []
    for representation, (matrix, target) in representations.items():
        selected_effects = effects[effects["representation"].eq(representation)]
        strongest = selected_effects.loc[
            selected_effects["difficulty_correlation_task_residual"].abs().idxmax()
        ]
        key = str(strongest["key"])
        feature = matrix[:, key_to_column[key]]
        for task in TASKS:
            selected = tasks == task
            x = feature[selected]
            y = target[selected]
            if np.std(x) <= 0.0 or np.std(y) <= 0.0:
                pearson = np.nan
                rank = np.nan
            else:
                pearson = float(np.corrcoef(x, y)[0, 1])
                rank = float(spearmanr(x, y).statistic)
            rows.append(
                {
                    "representation": representation,
                    "selected_key": key,
                    "task": task,
                    "short_task": SHORT_TASK[task],
                    "initial_states": int(selected.sum()),
                    "failure_rate_mean": float(y.mean()),
                    "failure_rate_sd": float(y.std(ddof=1)),
                    "pearson": pearson,
                    "spearman": rank,
                }
            )
    return pd.DataFrame(rows)


def frozen_validation(
    old_effects: pd.DataFrame,
    rolling_effects: pd.DataFrame,
    old_task_effects: pd.DataFrame,
) -> pd.DataFrame:
    long_effect = old_task_effects[old_task_effects["short_task"].eq("two_moka_pots")]
    selected_keys = []
    for query in QUERIES:
        part = long_effect[long_effect["query"].eq(query)].copy()
        selected_keys.extend(
            part.sort_values("effect_sigma", key=np.abs, ascending=False).head(3)["key"].tolist()
        )
    old = old_effects.set_index("key")
    latest = rolling_effects.set_index("key")
    rows = []
    for key in selected_keys:
        old_long_row = long_effect[long_effect["key"].eq(key)].iloc[0]
        rows.append(
            {
                "key": key,
                "query": int(old_long_row["query"]),
                "old_long_effect_sigma": float(old_long_row["effect_sigma"]),
                "old_pooled_effect_sigma": float(old.loc[key, "effect_sigma"]),
                "rolling_effect_sigma": float(latest.loc[key, "effect_sigma"]),
                "direction_match_old_long": bool(
                    np.sign(old_long_row["effect_sigma"])
                    == np.sign(latest.loc[key, "effect_sigma"])
                ),
                "rolling_p_raw": float(latest.loc[key, "permutation_p_raw"]),
                "rolling_bonferroni_p_12": float(
                    min(1.0, 12 * latest.loc[key, "permutation_p_raw"])
                ),
            }
        )
    return pd.DataFrame(rows)


def load_prior_state_evidence(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis = run_root / "analysis"
    quartile_path = analysis / "early_route_quartile_effects.csv"
    frozen_path = analysis / "frozen_signal_validation_family.csv"
    if not quartile_path.is_file() or not frozen_path.is_file():
        raise FileNotFoundError("rolling-star state-token validation artifacts are missing")
    quartile = pd.read_csv(quartile_path)
    quartile = quartile[
        quartile["prefix_queries"].eq(1) & quartile["token_family"].eq("state")
    ].sort_values(["permutation_p_maxT", "permutation_p_raw"])
    frozen = pd.read_csv(frozen_path)
    frozen = frozen[frozen["signal"].eq("q0_state_front_d1_top1_mass")].copy()
    if quartile.empty or set(frozen["split"]) != {
        "discovery",
        "transition_excluded",
        "validation",
    }:
        raise ValueError("rolling-star state-token evidence is incomplete")
    return quartile.reset_index(drop=True), frozen.reset_index(drop=True)


def query_summary(
    old_effects: pd.DataFrame,
    rolling_effects: pd.DataFrame,
    old_task_effects: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for query in QUERIES:
        old = old_effects[old_effects["query"].eq(query)]
        latest = rolling_effects[rolling_effects["query"].eq(query)]
        old_long = old_task_effects[
            old_task_effects["query"].eq(query)
            & old_task_effects["short_task"].eq("two_moka_pots")
        ].set_index("key")
        aligned = latest.set_index("key").loc[old_long.index]
        rho = spearmanr(
            old_long["effect_sigma"].to_numpy(),
            aligned["effect_sigma"].to_numpy(),
        ).statistic
        rows.append(
            {
                "query": query,
                "old_pooled_max_abs_effect": float(old["effect_sigma"].abs().max()),
                "old_pooled_min_family_p": float(old["permutation_p_max_family"].min()),
                "old_pooled_min_global_p": float(old["permutation_p_max_global"].min()),
                "rolling_max_abs_effect": float(latest["effect_sigma"].abs().max()),
                "rolling_min_family_p": float(latest["permutation_p_max_family"].min()),
                "rolling_min_global_p": float(latest["permutation_p_max_global"].min()),
                "long_vs_rolling_effect_spearman": float(rho),
            }
        )
    return pd.DataFrame(rows)


def plot_effect_heatmap(effects: pd.DataFrame, path: Path) -> None:
    metrics = ("entropy", "top1", "token_dispersion", "denoise_motion")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), metrics, strict=True):
        matrix = np.full((len(QUERIES) * 2, N_DENOISE), np.nan)
        selected = effects[effects["metric"].eq(metric)]
        for _, row in selected.iterrows():
            group_axis = 0 if row["layer_group"] == "front_2_5" else 1
            matrix[int(row["query"]) * 2 + group_axis, int(row["denoise_step"])] = row[
                "effect_sigma"
            ]
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-0.8, vmax=0.8)
        axis.set_title(metric)
        axis.set_xlabel("within-query denoise step")
        axis.set_ylabel("outer query / layer group")
        axis.set_xticks(range(N_DENOISE))
        axis.set_yticks(
            range(len(QUERIES) * 2),
            labels=[f"q{query} {'front' if group == 0 else 'back'}" for query in QUERIES for group in range(2)],
        )
    fig.colorbar(image, ax=axes, shrink=0.8, label="failure - success effect (SD)")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_prediction_deltas(deltas: pd.DataFrame, path: Path) -> None:
    selected = deltas[
        deltas["comparison"].isin(("route_vs_baseline", "route_beyond_controls"))
        & deltas["split"].eq("heldout_seed")
    ].copy()
    windows = list(DENOISE_WINDOWS)
    fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    offsets = {"route_vs_baseline": -0.12, "route_beyond_controls": 0.12}
    colors = {"route_vs_baseline": "#3b6ea8", "route_beyond_controls": "#c44e52"}
    for comparison in offsets:
        part = selected[selected["comparison"].eq(comparison)].set_index("window").loc[windows]
        x = np.arange(len(windows)) + offsets[comparison]
        y = part["brier_gain"].to_numpy()
        low = part["ci95_low"].to_numpy()
        high = part["ci95_high"].to_numpy()
        axis.errorbar(
            x,
            y,
            yerr=np.vstack((y - low, high - y)),
            fmt="o",
            capsize=3,
            color=colors[comparison],
            label=comparison,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(len(windows)), labels=windows, rotation=25, ha="right")
    axis.set_ylabel("OOF Brier gain")
    axis.set_title("Early MoE outcome increment on held-out flow seeds")
    axis.legend(frameon=False)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_report(summary: dict[str, Any], out_dir: Path) -> str:
    query = pd.read_csv(out_dir / "query_summary.csv")
    old_effects = pd.read_csv(out_dir / "old_effects.csv").set_index("key")
    rolling_effects = pd.read_csv(out_dir / "rolling_effects.csv").set_index("key")
    deltas = pd.read_csv(out_dir / "prediction_deltas.csv")
    difficulty = pd.read_csv(out_dir / "difficulty_model_metrics.csv")
    difficulty_predictions = pd.read_csv(out_dir / "difficulty_predictions.csv")
    difficulty_tasks = pd.read_csv(out_dir / "difficulty_task_correlations.csv")
    validation = pd.read_csv(out_dir / "frozen_validation.csv")
    diff_effects = pd.read_csv(out_dir / "difficulty_effects.csv")
    state_quartile = pd.read_csv(out_dir / "rolling_prior_state_q0_effects.csv")
    state_frozen = pd.read_csv(out_dir / "rolling_prior_state_frozen_validation.csv")

    seed_route = deltas[
        deltas["split"].eq("heldout_seed")
        & deltas["comparison"].eq("route_vs_baseline")
    ].set_index("window")
    seed_increment = deltas[
        deltas["split"].eq("heldout_seed")
        & deltas["comparison"].eq("route_beyond_controls")
    ].set_index("window")
    init_route = deltas[
        deltas["split"].eq("heldout_init")
        & deltas["comparison"].eq("route_vs_baseline")
    ].set_index("window")
    difficulty_map = difficulty.set_index("model")
    strongest_difficulty = {}
    for representation in diff_effects["representation"].unique():
        part = diff_effects[diff_effects["representation"].eq(representation)]
        strongest_difficulty[representation] = part.loc[
            part["difficulty_correlation_task_residual"].abs().idxmax()
        ]
    state_best = state_quartile.iloc[0]
    state_validation = state_frozen[state_frozen["split"].eq("validation")].iloc[0]
    task_mse = (
        difficulty_predictions.assign(
            squared_error=lambda frame: np.square(frame["score"] - frame["truth"])
        )
        .groupby(["short_task", "model"])["squared_error"]
        .mean()
        .unstack()
    )

    def delta_row(table: pd.DataFrame, window: str) -> str:
        row = table.loc[window]
        return f"{row['brier_gain']:+.5f} [{row['ci95_low']:+.5f},{row['ci95_high']:+.5f}]"

    lines = [
        "# 前四个 chunk 的 MoE 去噪轨迹与最终超时",
        "",
        "## 直接结论",
        "",
        "**最像“模型在第一次推理里读到了难度”的线索，确实落在 q0 的早期去噪；但它只在探索数据中出现，没有通过后续冻结验证。action-token 路由也有小差异，却没有形成跨初态、跨任务、跨 worker 的稳定超时信号。**",
        "",
        "严格区分后，答案分成两半：rolling-star 的 q0 state-token 前层 d0-d3 有过一个很像你直觉的 discovery 峰；旧五任务里可跨数据比较的 action-token 路由则没有“d0-d2 稳定优于后段”的顺序。加入 q1-q3 后也没有带来折外收益；此时即使有差异，也已经混入执行后的状态分叉。",
        "",
        "## 数据和口径",
        "",
        f"- 旧五任务：{summary['old_corpus']['episodes']} rollout，其中有成败变化的四任务共 {summary['old_corpus']['variable_task_episodes']} 条、{summary['old_corpus']['variable_task_failures']} 次超时；每个 task x initial-state 有 32 个 flow seeds。",
        f"- 同初态成败主检验：{summary['old_effect_scan']['episodes']} 条、{summary['old_effect_scan']['mixed_strata']} 个 mixed task x init strata。",
        f"- rolling-star：总共 {summary['rolling_corpus']['branches']} 条分支；主 sibling 对照只用 {summary['rolling_effect_scan']['episodes']} 条、{summary['rolling_effect_scan']['mixed_strata']} 个成败混合快照。",
        "- 只使用共同绝对 q0-q3，以及每个 query 内 d0-d9 的 soft action routing；不使用 episode length、remaining time、相对尾段或 hard expert ID。",
        f"- state token 在 denoise 轴最大差异为 old `{summary['integrity']['old_state_denoise_max']:.8f}`、rolling `{summary['integrity']['rolling_state_denoise_max']:.8f}`。旧语料的 state 路由在 d 轴完全退化，所以跨语料的去噪定位只能比较 action token；rolling 的 state 结果单列。",
        "",
        "## 最像直觉的 state-token 线索",
        "",
        "rolling-star 既有的快照内四分位扫描中，最强 q0 state cell 是 `%s / %s / d%d / %s`：高值四分位最终超时率 %.1f%%，低值四分位 %.1f%%，差 %.1f 个百分点；全 280-cell max-T `p=%.4f`。"
        % (
            state_best["token_family"],
            state_best["layer_group"],
            state_best["denoise_step"],
            state_best["metric"],
            100 * state_best["top_quartile_failure_rate"],
            100 * state_best["bottom_quartile_failure_rate"],
            100 * state_best["failure_rate_difference"],
            state_best["permutation_p_maxT"],
        ),
        "",
        "这条峰的位置支持“前层、q0、早期 d”的定位，但统计强度只到边缘。更关键的是，当时冻结的同族 `q0 state / front / d1 / top1` 在后来 5 个快照上，高低四分位超时率分别为 %.1f%% 和 %.1f%%，差值 %.1f 个百分点，Bonferroni `p=%.4f`。所以它是一个没有复现的线索，不是已成立的早期难度读数。"
        % (
            100 * state_validation["top_quartile_failure_rate"],
            100 * state_validation["bottom_quartile_failure_rate"],
            100 * state_validation["quartile_failure_rate_difference"],
            state_validation["primary_p_bonferroni"],
        ),
        "",
        "## action-token 的 q0 去噪带",
        "",
        "Brier gain 为正才有用。`heldout_seed` 已知道初态的训练难度先验；`heldout_init` 完整留出初态。",
        "",
        "| window | route vs initial-state prior | route beyond state/action controls | route on unseen init vs task prior |",
        "| --- | ---: | ---: | ---: |",
    ]
    for window in ("q0_d0_d2", "q0_d3_d6", "q0_d7_d9", "q0_all_d"):
        lines.append(
            f"| {window} | {delta_row(seed_route, window)} | {delta_row(seed_increment, window)} | {delta_row(init_route, window)} |"
        )
    lines.extend(
        [
            "",
            "如果直觉成立，最早的 d0-d2 应稳定优于中后段，并在 unseen init 上保留正增量。实际没有看到这个顺序。",
            "",
            "## 加入前几个 chunk",
            "",
            "| prefix | route vs initial-state prior | route beyond state/action controls |",
            "| --- | ---: | ---: |",
        ]
    )
    for window in ("q0_all_d", "q0_q1_all_d", "q0_q2_all_d", "q0_q3_all_d"):
        lines.append(
            f"| {window} | {delta_row(seed_route, window)} | {delta_row(seed_increment, window)} |"
        )
    lines.extend(
        [
            "",
            "q1-q3 已经执行过前面的动作块，因此这里若出现信号，只能叫早期轨迹读出。必须看 `route beyond controls`，不能把 route-only 的改善直接解释成模型内部提前知道难度。",
            "",
            "## 单特征差异与跨数据复现",
            "",
            "| query | old pooled max | old global p(min) | rolling max | rolling global p(min) | Long-old vs rolling effect rho |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in query.iterrows():
        lines.append(
            "| q%d | %.3f | %.4f | %.3f | %.4f | %+.3f |"
            % (
                row["query"],
                row["old_pooled_max_abs_effect"],
                row["old_pooled_min_global_p"],
                row["rolling_max_abs_effect"],
                row["rolling_min_global_p"],
                row["long_vs_rolling_effect_spearman"],
            )
        )
    matches = int(validation["direction_match_old_long"].sum())
    validated = int((validation["rolling_bonferroni_p_12"] < 0.05).sum())
    lines.extend(
        [
            "",
            f"从旧 Long 每个 q 冻结三个最强 cell（共 12 个）到 rolling-star：方向一致 {matches}/12，Bonferroni 后通过 {validated}/12。这个验证优先于任一数据集里事后挑出的漂亮 cell。",
            "",
            "![早期 query x denoise effect](old_effect_heatmap.png)",
            "",
            "## 初态难度，而不是单条 rollout 命运",
            "",
            "这里避免同批噪声泄漏：前 16 个 seeds 的 q0 特征只预测后 16 个 seeds 的失败率。下面是真正 leave-one-initial-state-out 的 MSE；`moe_single` 只用前半固定第一条的一次 q0，`moe_cloud` 使用前半 16 个 q0 的均值和方差。",
            "",
            "| model | task-macro MSE |",
            "| --- | ---: |",
        ]
    )
    for model in (
        "task_mean",
        "geometry",
        "moe_single",
        "moe_cloud",
        "action_cloud",
        "geometry_action",
        "geometry_moe",
        "geometry_action_moe",
    ):
        lines.append(f"| {model} | {difficulty_map.loc[model, 'mse_task_macro']:.5f} |")
    cloud_task_rows = difficulty_tasks[
        difficulty_tasks["representation"].eq("cloud_mean_16a_to_16b")
        & difficulty_tasks["pearson"].notna()
    ]
    single_task_rows = difficulty_tasks[
        difficulty_tasks["representation"].eq("single_seed_a_to_other31")
        & difficulty_tasks["pearson"].notna()
    ]
    cloud_primary = strongest_difficulty["cloud_mean_16a_to_16b"]
    cloud_reverse = strongest_difficulty["cloud_mean_16b_to_16a"]
    single_primary = strongest_difficulty["single_seed_a_to_other31"]
    single_reverse = strongest_difficulty["single_seed_b_to_other31"]
    difficulty_key = "q0|action|back_12_15|top1|d5"
    long_primary = difficulty_tasks[
        difficulty_tasks["representation"].eq("cloud_mean_16a_to_16b")
        & difficulty_tasks["short_task"].eq("two_moka_pots")
    ].iloc[0]
    long_reverse = difficulty_tasks[
        difficulty_tasks["representation"].eq("cloud_mean_16b_to_16a")
        & difficulty_tasks["short_task"].eq("two_moka_pots")
    ].iloc[0]
    lines.extend(
        [
            "",
            "前 16 个 q0 预测后 16 个结果时，最强 cell 是 `%s`，task-residual correlation `%+.3f`，全 q0 max-T `p=%.4f`，再校正 4 种 representation 后 `p=%.4f`；在 %d/%d 个有难度变化的任务里方向为正。反向用后 16 预测前 16 时，最强 cell 变为 `%s`，correlation `%+.3f`，max-T `p=%.4f`，四重校正 `p=%.4f`。"
            % (
                cloud_primary["key"],
                cloud_primary["difficulty_correlation_task_residual"],
                cloud_primary["permutation_p_max_global"],
                cloud_primary["permutation_p_across_4_representations"],
                int((cloud_task_rows["pearson"] > 0).sum()),
                len(cloud_task_rows),
                cloud_reverse["key"],
                cloud_reverse["difficulty_correlation_task_residual"],
                cloud_reverse["permutation_p_max_global"],
                cloud_reverse["permutation_p_across_4_representations"],
            ),
            "固定只看一次 q0、并用其余 31 条估计难度时，前半 canonical 的最强 cell 是 `%s`，correlation `%+.3f`，max-T `p=%.4f`，四重校正 `p=%.4f`；任务内方向为正 %d/%d。换成后半 canonical 后，最强 cell 为 `%s`，correlation `%+.3f`，max-T `p=%.4f`，四重校正 `p=%.4f`。单次形式才对应在线可用口径。"
            % (
                single_primary["key"],
                single_primary["difficulty_correlation_task_residual"],
                single_primary["permutation_p_max_global"],
                single_primary["permutation_p_across_4_representations"],
                int((single_task_rows["pearson"] > 0).sum()),
                len(single_task_rows),
                single_reverse["key"],
                single_reverse["difficulty_correlation_task_residual"],
                single_reverse["permutation_p_max_global"],
                single_reverse["permutation_p_across_4_representations"],
            ),
            "",
            "预测门槛更直接：`moe_single` 相对 task mean 的 task-macro MSE 改善为 `%+.5f`，`moe_cloud` 为 `%+.5f`（正数才有用）。在 Long moka-pot 单任务里 `moe_single` 从 `%.5f` 降到 `%.5f`，但其他任务的退化抵消了它。"
            % (
                difficulty_map.loc["task_mean", "mse_task_macro"]
                - difficulty_map.loc["moe_single", "mse_task_macro"],
                difficulty_map.loc["task_mean", "mse_task_macro"]
                - difficulty_map.loc["moe_cloud", "mse_task_macro"],
                task_mse.loc["two_moka_pots", "task_mean"],
                task_mse.loc["two_moka_pots", "moe_single"],
            ),
            "",
            "最重要的区分：同一个 `%s` 对 Long 初态难度的正反 half 相关分别为 `%+.3f`、`%+.3f`，但在同一初态/快照内比较“这一条最终超时还是成功”时，old effect 只有 `%+.3f SD`、rolling effect `%+.3f SD`，两边全局 max-T 都是 `p=%.1f`。所以它目前读到的是初态/任务难度，不是单条 branch 的命运。"
            % (
                difficulty_key,
                long_primary["pearson"],
                long_reverse["pearson"],
                old_effects.loc[difficulty_key, "effect_sigma"],
                rolling_effects.loc[difficulty_key, "effect_sigma"],
                max(
                    old_effects.loc[difficulty_key, "permutation_p_max_global"],
                    rolling_effects.loc[difficulty_key, "permutation_p_max_global"],
                ),
            ),
            "",
            "## 判断",
            "",
            "1. 有一个位置上很符合直觉的线索：rolling q0 state-token、前层、d0-d3；但 frozen validation 明确没有复现。",
            "2. q0 action routing 在最终成功/超时之间有小而任务依赖的差异；没有证据说明信息稳定集中在 d0-d2。",
            "3. q1-q3 没有增加折外预测收益；即使未来看到差异，也应先解释为执行后状态/动作分叉。",
            "4. 多次平均后的初态难度相关，不等于一次推理能读出；而且正反 half 必须复现。当前单次 q0 没有跨任务胜过 task mean。",
            "5. 没有稳定单 expert 结论；本分析只使用 expert-permutation-invariant soft-routing summaries。",
            "",
            "## 解释限制",
            "",
            "- 旧语料的同一个 flow seed 同时影响 q0 和后续 replanning；这是预测关联，不是 q0 action 的因果 value。",
            "- `moe_cloud` 需要 16 次 q0 forward，不能冒充一次推理即可得到的信号。",
            "- rolling-star 有 352 条 branch，但独立 worker 只有 4 个；snapshot 内 sibling 通过分层处理，不能当成 352 个独立初态。",
            "- 本分析在既有数据上提出，属于探索性定位。确认实验应冻结一个窗口和一个分数，再收新 worker/初态。",
            "",
            "## 工件",
            "",
            "- `old_effects.csv`、`old_task_effects.csv`、`rolling_effects.csv`：逐 cell 差异。",
            "- `prediction_metrics.csv`、`prediction_deltas.csv`、`predictions.csv.gz`：折外预测。",
            "- `difficulty_effects.csv`、`difficulty_task_correlations.csv`、`difficulty_model_metrics.csv`：单次/云平均初态难度分析。",
            "- `rolling_prior_state_q0_effects.csv`、`rolling_prior_state_frozen_validation.csv`：既有 state-token discovery 与冻结验证。",
            "- `frozen_validation.csv`：旧 Long 到 rolling-star 的冻结验证。",
            "- `summary.json`、`checksums.sha256`：审计与校验。",
            "",
            "![Brier increments](prediction_brier_gains.png)",
            "",
        ]
    )
    return "\n".join(lines)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    routes = rng.uniform(0.01, 1.0, size=(6, 4, 8, 10, 11, 32)).astype(np.float16)
    routes /= routes.sum(axis=-1, keepdims=True)
    values, metadata, state, keys, span = extract_route_features(routes)
    assert values.shape == (6, 472)
    assert len(metadata) == 472
    assert state.shape == (6, 32) and len(keys) == 32
    assert span > 0
    assert len(window_columns(metadata, "q0_d0_d2")) == 34
    groups = np.repeat(np.arange(3), 4)
    labels = np.tile([0, 0, 1, 1], 3)
    x = labels[:, None] + rng.normal(0, 0.05, size=(12, 1))
    _, _, effect, _ = layer.fixed_effect_shift(x, labels, groups)
    assert effect[0] > 5
    print("self-test passed: route_features=472 state_features=32")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    (
        old_frame,
        old_route,
        route_meta,
        old_state,
        state_keys,
        old_controls,
        old_geometry,
        old_action,
        old_state_spans,
    ) = load_old_corpus(args.cache_root)
    (
        rolling_frame,
        rolling_route,
        rolling_meta,
        rolling_state,
        rolling_state_keys,
        rolling_state_span,
    ) = load_rolling_corpus(args.rolling_root)
    if rolling_meta != route_meta or rolling_state_keys != state_keys:
        raise ValueError("old and rolling route feature schemas differ")

    variable_mask = old_frame["task"].isin(VARIABLE_TASKS).to_numpy()
    old_effects, old_scan = effect_scan(
        old_frame[variable_mask].reset_index(drop=True),
        old_route[variable_mask],
        route_meta,
        args.permutations,
        args.seed,
        args.bootstraps,
        "right16x32_pooled",
    )
    rolling_effects, rolling_scan = effect_scan(
        rolling_frame,
        rolling_route,
        route_meta,
        args.permutations,
        args.seed + 100,
        args.bootstraps,
        "rolling_star",
    )
    old_by_task = task_effects(old_frame, old_route, route_meta)
    frozen = frozen_validation(old_effects, rolling_effects, old_by_task)
    q_summary = query_summary(old_effects, rolling_effects, old_by_task)

    predictions = early_outcome_predictions(
        old_frame, old_route, route_meta, old_controls, args.seed + 200
    )
    metrics = prediction_metrics(predictions)
    deltas = bootstrap_prediction_deltas(
        predictions, args.bootstraps, args.seed + 300
    )

    pool_frame, pool_blocks = difficulty_frame(
        old_frame,
        old_route,
        old_state,
        old_geometry,
        old_action,
        route_meta,
    )
    difficulty_predictions, difficulty_metrics = difficulty_models(pool_frame, pool_blocks)
    difficulty_effects = continuous_difficulty_scan(
        old_frame,
        old_route,
        route_meta,
        args.permutations,
        args.seed + 400,
    )
    difficulty_task_effects = difficulty_task_correlations(
        old_frame, old_route, route_meta, difficulty_effects
    )
    prior_state_effects, prior_state_frozen = load_prior_state_evidence(args.rolling_root)

    old_effects.to_csv(args.out_dir / "old_effects.csv", index=False)
    old_by_task.to_csv(args.out_dir / "old_task_effects.csv", index=False)
    rolling_effects.to_csv(args.out_dir / "rolling_effects.csv", index=False)
    frozen.to_csv(args.out_dir / "frozen_validation.csv", index=False)
    q_summary.to_csv(args.out_dir / "query_summary.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(args.out_dir / "prediction_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "prediction_deltas.csv", index=False)
    pool_frame.to_csv(args.out_dir / "difficulty_cohort.csv", index=False)
    difficulty_predictions.to_csv(args.out_dir / "difficulty_predictions.csv", index=False)
    difficulty_metrics.to_csv(args.out_dir / "difficulty_model_metrics.csv", index=False)
    difficulty_effects.to_csv(args.out_dir / "difficulty_effects.csv", index=False)
    difficulty_task_effects.to_csv(
        args.out_dir / "difficulty_task_correlations.csv", index=False
    )
    prior_state_effects.to_csv(
        args.out_dir / "rolling_prior_state_q0_effects.csv", index=False
    )
    prior_state_frozen.to_csv(
        args.out_dir / "rolling_prior_state_frozen_validation.csv", index=False
    )

    plot_effect_heatmap(old_effects, args.out_dir / "old_effect_heatmap.png")
    plot_prediction_deltas(deltas, args.out_dir / "prediction_brier_gains.png")

    summary = {
        "schema": "himoe.early_chunk_denoise_timeout.v1",
        "protocol": {
            "outer_queries": list(QUERIES),
            "within_query_denoise_steps": N_DENOISE,
            "features": len(route_meta),
            "denoise_windows": DENOISE_WINDOWS,
            "outcome": "eventual timeout versus success",
            "forbidden_features": [
                "episode_length",
                "remaining_time",
                "relative_phase",
                "terminal_window",
                "hard_expert_id",
                "hidden_state",
            ],
            "effect": "failure-minus-success fixed-stratum shift / residual SD",
            "prediction": "task-local offset logistic, training-fold scaler/PCA, C=0.1",
        },
        "old_corpus": {
            "episodes": len(old_frame),
            "failures": int(old_frame["failure"].sum()),
            "variable_task_episodes": int(variable_mask.sum()),
            "variable_task_failures": int(old_frame.loc[variable_mask, "failure"].sum()),
            "tasks": len(TASKS),
            "initial_states": int(old_frame.groupby(["task", "init_state_id"]).ngroups),
        },
        "rolling_corpus": {
            "branches": len(rolling_frame),
            "failures": int(rolling_frame["failure"].sum()),
            "successes": int((~rolling_frame["failure"]).sum()),
            "snapshots": int(rolling_frame["snapshot_key"].nunique()),
            "workers": int(rolling_frame["worker"].nunique()),
        },
        "old_effect_scan": old_scan,
        "rolling_effect_scan": rolling_scan,
        "integrity": {
            "old_state_denoise_max": max(old_state_spans.values()),
            "old_state_denoise_by_task": old_state_spans,
            "rolling_state_denoise_max": rolling_state_span,
            "old_and_rolling_feature_schema_equal": rolling_meta == route_meta,
            "old_min_queries": int(old_frame["inference_calls"].min()),
            "rolling_min_queries": int(rolling_frame["inference_calls"].min()),
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
        },
        "artifacts": {},
    }
    report = render_report(summary, args.out_dir)
    (args.out_dir / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(plain(summary), indent=2) + "\n", encoding="utf-8")

    artifacts = [
        path
        for path in sorted(args.out_dir.iterdir())
        if path.is_file() and path.name not in {"checksums.sha256", "summary.json"}
    ]
    checksums = {path.name: sha256(path) for path in artifacts}
    summary["artifacts"] = checksums
    summary_path.write_text(json.dumps(plain(summary), indent=2) + "\n", encoding="utf-8")
    checksums[summary_path.name] = sha256(summary_path)
    with (args.out_dir / "checksums.sha256").open("w", encoding="utf-8") as handle:
        for name, digest in sorted(checksums.items()):
            handle.write(f"{digest}  {name}\n")
    print(json.dumps(plain(summary), indent=2))


if __name__ == "__main__":
    main()
