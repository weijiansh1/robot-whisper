#!/usr/bin/env python3
"""Test whether early chunk-token MoE structure contains outcome information.

The primary analysis is conditional on initial state in the 16x32 Long corpus.
It never uses episode length, remaining time, terminal queries, actions, hidden
states, or simulator state.  Goal/Object/Spatial are a secondary task-stratified
check because those corpora have only one rollout per initial state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

import analyze_chunk_token_routing as token
import analyze_early_chunk_denoise_timeout as early
import analyze_initial_state_early_moe_signal as initial
import analyze_state_route_layer_denoise_effects as layer


HERE = Path(__file__).resolve().parent
DEFAULT_LONG = token.suite.DEFAULT_LONG
DEFAULT_GOS = token.suite.DEFAULT_CORPUS
DEFAULT_OUT = HERE / "analysis/chunk-token-outcomes-20260829"
SEED = 20260829
PRIMARY_SCORE_NAMES = (
    "top1_u_shape",
    "entropy_u_shape",
    "soft_change_mean",
    "soft_front_minus_back",
    "soft_early_minus_late",
    "soft_edge_minus_middle",
    "top1_switch_mean",
    "top4_turnover_mean",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--long-run", type=Path, default=DEFAULT_LONG)
    parser.add_argument("--gos-corpus", type=Path, default=DEFAULT_GOS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--primary-permutations", type=int, default=10000)
    parser.add_argument("--cell-permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--prediction-bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def episode_profiles(routes: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    if routes.shape[1:] != (4, 8, 10, 11, 32):
        raise ValueError(f"unexpected route tensor {routes.shape}")
    profiles: dict[tuple[str, str], np.ndarray] = {}
    for group_name, layer_axes in token.LAYER_GROUPS.items():
        selected = np.take(routes, layer_axes, axis=2)
        action = token.suite.early.normalize(selected[:, :, :, :, 1:, :])
        profiles[("top1", group_name)] = action.max(axis=-1).mean(axis=2)
        profiles[("entropy", group_name)] = token.suite.early.entropy(action).mean(
            axis=2
        )
        profiles[("adjacent_hellinger", group_name)] = token.suite.early.hellinger(
            action[..., 1:, :], action[..., :-1, :]
        ).mean(axis=2)
        top1_expert = np.argmax(action, axis=-1)
        profiles[("top1_switch_rate", group_name)] = (
            top1_expert[..., 1:] != top1_expert[..., :-1]
        ).mean(axis=2)
        top4 = np.argpartition(action, -4, axis=-1)[..., -4:]
        left = top4[..., 1:, :]
        right = top4[..., :-1, :]
        overlap = (left[..., :, None] == right[..., None, :]).any(axis=-1).sum(
            axis=-1
        )
        profiles[("top4_turnover", group_name)] = (1.0 - overlap / 4.0).mean(
            axis=2
        )
    return profiles


def token_feature_matrix(
    profiles: dict[tuple[str, str], np.ndarray],
    schema: list[token.TokenFeature],
) -> np.ndarray:
    centered: dict[tuple[str, str], np.ndarray] = {}
    for metric in ("top1", "entropy"):
        for group_name in token.LAYER_GROUPS:
            raw = profiles[(metric, group_name)]
            centered[(f"{metric}_centered", group_name)] = raw - raw.mean(
                axis=-1, keepdims=True
            )
    columns = []
    for item in schema:
        if item.metric in token.ADJACENT_METRICS:
            values = profiles[(item.metric, item.layer_group)]
            token_axis = item.token_position - 2
        else:
            values = centered[(item.metric, item.layer_group)]
            token_axis = item.token_position - 1
        columns.append(values[:, item.query, item.denoise_step, token_axis])
    return np.column_stack(columns).astype(np.float32)


def score_block(
    profiles: dict[tuple[str, str], np.ndarray], queries: tuple[int, ...]
) -> dict[str, np.ndarray]:
    groups = tuple(token.LAYER_GROUPS)
    top1 = np.stack([profiles[("top1", group)] for group in groups], axis=2)
    entropy = np.stack([profiles[("entropy", group)] for group in groups], axis=2)
    adjacent = np.stack(
        [profiles[("adjacent_hellinger", group)] for group in groups], axis=2
    )
    switch = np.stack(
        [profiles[("top1_switch_rate", group)] for group in groups], axis=2
    )
    turnover = np.stack(
        [profiles[("top4_turnover", group)] for group in groups], axis=2
    )
    top1 = top1[:, queries]
    entropy = entropy[:, queries]
    adjacent = adjacent[:, queries]
    switch = switch[:, queries]
    turnover = turnover[:, queries]
    top1_ends = top1[..., [0, 9]].mean(axis=-1)
    top1_middle = top1[..., 3:7].mean(axis=-1)
    entropy_ends = entropy[..., [0, 9]].mean(axis=-1)
    entropy_middle = entropy[..., 3:7].mean(axis=-1)
    return {
        "top1_u_shape": (top1_ends - top1_middle).mean(axis=(1, 2, 3)),
        "entropy_u_shape": (entropy_middle - entropy_ends).mean(axis=(1, 2, 3)),
        "soft_change_mean": adjacent.mean(axis=(1, 2, 3, 4)),
        "soft_front_minus_back": (
            adjacent[:, :, 0].mean(axis=(1, 2, 3))
            - adjacent[:, :, 1].mean(axis=(1, 2, 3))
        ),
        "soft_early_minus_late": (
            adjacent[..., :3, :].mean(axis=(1, 2, 3, 4))
            - adjacent[..., 7:, :].mean(axis=(1, 2, 3, 4))
        ),
        "soft_edge_minus_middle": (
            adjacent[..., [0, 8]].mean(axis=(1, 2, 3, 4))
            - adjacent[..., 2:7].mean(axis=(1, 2, 3, 4))
        ),
        "top1_switch_mean": switch.mean(axis=(1, 2, 3, 4)),
        "top4_turnover_mean": turnover.mean(axis=(1, 2, 3, 4)),
    }


def structure_matrices(
    profiles: dict[tuple[str, str], np.ndarray]
) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    primary: dict[str, np.ndarray] = {}
    for prefix, queries in (("q0", (0,)), ("q0_q3_mean", (0, 1, 2, 3))):
        for name, values in score_block(profiles, queries).items():
            primary[f"{prefix}|{name}"] = values
    adjacent_back = profiles[("adjacent_hellinger", "back_12_15")]
    primary["q0|soft_tail_d5_back"] = adjacent_back[:, 0, 5, 8] - adjacent_back[
        :, 0, 5, :8
    ].mean(axis=-1)

    predictor: dict[str, np.ndarray] = {}
    for query in range(4):
        for name, values in score_block(profiles, (query,)).items():
            predictor[f"q{query}|{name}"] = values
    predictor["q0|soft_tail_d5_back"] = primary["q0|soft_tail_d5_back"]
    return (
        np.column_stack(list(primary.values())).astype(np.float32),
        list(primary),
        np.column_stack(list(predictor.values())).astype(np.float32),
        list(predictor),
    )


def route_rows(store: zarr.Group, episodes: list[int]) -> np.ndarray:
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    rows = []
    for episode in episodes:
        found = np.flatnonzero(episode_ids == episode)[:4]
        if len(found) != 4:
            raise ValueError(f"episode {episode}: missing q0-q3")
        rows.extend(found.tolist())
    return np.asarray(rows, dtype=np.int64)


def load_long(
    run: Path, batch_size: int = 32
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    list[token.TokenFeature],
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    list[early.RouteFeature],
]:
    summaries = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    if len(summaries) != 512:
        raise ValueError(f"expected 512 Long summaries, got {len(summaries)}")
    frame = pd.DataFrame(
        {
            "episode": [int(row["episode_index"]) for row in summaries],
            "init_state_id": [int(row["init_state_id"]) for row in summaries],
            "flow_noise_seed": [int(row["flow_noise_seed"]) for row in summaries],
            "success": [bool(row["success"]) for row in summaries],
        }
    )
    frame["failure"] = ~frame["success"]
    frame["stratum"] = "i" + frame["init_state_id"].astype(str)
    schema = token.feature_schema()
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    stored_episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    row_lookup: dict[int, np.ndarray] = {}
    for episode in frame["episode"].astype(int):
        found = np.flatnonzero(stored_episode_ids == episode)[:4]
        if len(found) != 4:
            raise ValueError(f"episode {episode}: missing q0-q3")
        row_lookup[episode] = found
    token_chunks = []
    primary_chunks = []
    predictor_chunks = []
    aggregate_chunks = []
    primary_keys: list[str] | None = None
    predictor_keys: list[str] | None = None
    aggregate_meta: list[early.RouteFeature] | None = None
    for start in range(0, len(frame), batch_size):
        stop = min(start + batch_size, len(frame))
        episodes = frame.iloc[start:stop]["episode"].astype(int).tolist()
        rows = np.concatenate([row_lookup[episode] for episode in episodes])
        routes = np.asarray(store["hb_router_probs"].oindex[rows]).reshape(
            len(episodes), 4, 8, 10, 11, 32
        )
        profiles = episode_profiles(routes)
        token_chunks.append(token_feature_matrix(profiles, schema))
        primary, keys, predictor, predictor_names = structure_matrices(profiles)
        primary_chunks.append(primary)
        predictor_chunks.append(predictor)
        aggregate, meta, _, _, _ = early.extract_route_features(routes)
        aggregate_chunks.append(aggregate)
        if primary_keys is None:
            primary_keys = keys
            predictor_keys = predictor_names
            aggregate_meta = meta
        elif primary_keys != keys or predictor_keys != predictor_names or aggregate_meta != meta:
            raise ValueError("feature schema changed between Long batches")
        print(f"Long routes {stop}/{len(frame)}", flush=True)
    assert primary_keys is not None and predictor_keys is not None
    assert aggregate_meta is not None
    return (
        frame,
        np.vstack(token_chunks),
        schema,
        np.vstack(primary_chunks),
        primary_keys,
        np.vstack(predictor_chunks),
        predictor_keys,
        np.vstack(aggregate_chunks),
        aggregate_meta,
    )


def load_gos(
    corpus: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    index = token.suite.read_index(corpus)
    schema = token.feature_schema()
    frames = []
    token_chunks = []
    primary_chunks = []
    primary_keys: list[str] | None = None
    for (suite_name, task_dir), group in index.groupby(
        ["suite", "task_dir"], sort=True
    ):
        group = group.sort_values("episode_index").reset_index(drop=True)
        task_path = corpus / f"libero_{suite_name}" / task_dir
        store = zarr.open_group(str(task_path / "server/routes.zarr"), mode="r")
        episodes = group["episode_index"].astype(int).tolist()
        rows = route_rows(store, episodes)
        routes = np.asarray(store["hb_router_probs"].oindex[rows]).reshape(
            len(group), 4, 8, 10, 11, 32
        )
        profiles = episode_profiles(routes)
        token_chunks.append(token_feature_matrix(profiles, schema))
        primary, keys, _, _ = structure_matrices(profiles)
        primary_chunks.append(primary)
        if primary_keys is None:
            primary_keys = keys
        elif primary_keys != keys:
            raise ValueError("GOS primary score schema changed")
        frame = pd.DataFrame(
            {
                "suite": suite_name,
                "task": suite_name + "/" + task_dir,
                "episode": group["episode_index"].astype(int),
                "init_state_id": group["init_state_id"].astype(int),
                "success": group["success"].astype(bool),
            }
        )
        frame["failure"] = ~frame["success"]
        frame["stratum"] = frame["task"]
        frames.append(frame)
        print(f"GOS loaded {suite_name}/{task_dir}", flush=True)
    assert primary_keys is not None
    return (
        pd.concat(frames, ignore_index=True),
        np.vstack(token_chunks),
        np.vstack(primary_chunks),
        primary_keys,
    )


def score_metadata(keys: list[str]) -> list[layer.FeatureMeta]:
    result = []
    for key in keys:
        parts = key.split("|")
        result.append(
            layer.FeatureMeta(
                key=key,
                family=parts[-1],
                token_role="action",
                scope="chunk_position_summary",
                metric=parts[-1],
                relative_query=0 if parts[0] == "q0" else None,
            )
        )
    return result


def cell_metadata(schema: list[token.TokenFeature]) -> list[layer.FeatureMeta]:
    return [
        layer.FeatureMeta(
            key=item.key,
            family=item.metric,
            token_role="action",
            scope="chunk_token_cell",
            metric=item.metric,
            layer_group=item.layer_group,
            denoise_step=item.denoise_step,
            relative_query=item.query,
        )
        for item in schema
    ]


def mixed_subset(
    frame: pd.DataFrame, values: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    mixed = frame.groupby("stratum")["failure"].nunique()
    mask = frame["stratum"].isin(mixed.index[mixed == 2]).to_numpy()
    subset = frame.loc[mask].reset_index(drop=True)
    labels = subset["failure"].to_numpy(dtype=np.int64)
    groups, names = pd.factorize(subset["stratum"], sort=True)
    return subset, values[mask], labels, groups


def effect_scan(
    frame: pd.DataFrame,
    values: np.ndarray,
    metadata: list[layer.FeatureMeta],
    permutations: int,
    bootstraps: int,
    seed: int,
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset, selected, labels, groups = mixed_subset(frame, values)
    beta, residual_sd, effect, _ = layer.fixed_effect_shift(selected, labels, groups)
    raw_p, family_p, global_p = layer.permutation_scan(
        selected, labels, groups, metadata, permutations, seed
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
                "adjusted_difference": float(beta[index]),
                "residual_sd": float(residual_sd[index]),
                "effect_sigma": float(effect[index]),
                "effect_ci95_low": float(ci_low[index]),
                "effect_ci95_high": float(ci_high[index]),
                "permutation_p_raw": float(raw_p[index]),
                "permutation_p_max_family": float(family_p[index]),
                "permutation_p_max_global": float(global_p[index]),
            }
        )
    audit = {
        "dataset": dataset,
        "episodes": len(subset),
        "failures": int(labels.sum()),
        "successes": int((labels == 0).sum()),
        "mixed_strata": int(len(np.unique(groups))),
        "features": len(metadata),
        "permutations": permutations,
        "bootstraps": bootstraps,
    }
    return pd.DataFrame(rows), audit


def effect_vector(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    _, selected, labels, groups = mixed_subset(frame, values)
    _, _, effect, _ = layer.fixed_effect_shift(selected, labels, groups)
    return effect


def assign_folds(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    seeds = sorted(result["flow_noise_seed"].unique())
    states = sorted(result["init_state_id"].unique())
    if len(seeds) != 32 or len(states) != 16:
        raise ValueError("prediction expects 32 seeds and 16 initial states")
    seed_fold = {value: index // 4 for index, value in enumerate(seeds)}
    state_fold = {value: index // 2 for index, value in enumerate(states)}
    result["heldout_seed_fold"] = result["flow_noise_seed"].map(seed_fold)
    result["heldout_init_fold"] = result["init_state_id"].map(state_fold)
    return result


def standardize_block(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def robust_offset_scores(
    train: np.ndarray,
    y_train: np.ndarray,
    test: np.ndarray,
    train_baseline: np.ndarray,
    test_baseline: np.ndarray,
    c_value: float,
) -> np.ndarray:
    try:
        return initial.offset_logistic_scores(
            train,
            y_train,
            test,
            train_baseline,
            test_baseline,
            c_value,
        )[1]
    except RuntimeError:
        epsilon = 1e-6
        train_offset = np.log(
            np.clip(train_baseline, epsilon, 1 - epsilon)
            / np.clip(1 - train_baseline, epsilon, 1 - epsilon)
        )
        test_offset = np.log(
            np.clip(test_baseline, epsilon, 1 - epsilon)
            / np.clip(1 - test_baseline, epsilon, 1 - epsilon)
        )
        train = np.asarray(train, dtype=np.float64)
        test = np.asarray(test, dtype=np.float64)
        labels = y_train.astype(np.float64)

        def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
            logits = train_offset + train @ coefficients
            probability = expit(logits)
            loss = float(
                np.sum(np.logaddexp(0.0, logits) - labels * logits)
                + 0.5 * np.dot(coefficients, coefficients) / c_value
            )
            gradient = train.T @ (probability - labels) + coefficients / c_value
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(train.shape[1], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": 2000,
                "maxls": 100,
                "ftol": 1e-9,
                "gtol": 1e-6,
            },
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise RuntimeError(f"offset logistic retry failed: {result.message}")
        return expit(test_offset + test @ result.x)


def shifted_sibling_features(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    for _, group in frame.groupby("init_state_id", sort=True):
        indices = group.sort_values("flow_noise_seed").index.to_numpy(dtype=np.int64)
        result[indices] = values[np.roll(indices, -1)]
    return result


def outcome_predictions(
    frame: pd.DataFrame,
    structures: np.ndarray,
    structure_keys: list[str],
    aggregates: np.ndarray,
    aggregate_meta: list[early.RouteFeature],
    cells: np.ndarray,
    cell_schema: list[token.TokenFeature],
    seed: int,
) -> pd.DataFrame:
    frame = assign_folds(frame)
    q0_structure = np.asarray(
        [index for index, key in enumerate(structure_keys) if key.startswith("q0|")]
    )
    q0_aggregate = np.asarray(
        [index for index, item in enumerate(aggregate_meta) if item.query == 0]
    )
    q0_cells = np.asarray(
        [index for index, item in enumerate(cell_schema) if item.query == 0]
    )
    all_cells = np.arange(len(cell_schema), dtype=np.int64)
    rows = []
    for split in ("heldout_seed", "heldout_init"):
        include_state = split == "heldout_seed"
        fold_column = f"{split}_fold"
        for fold in range(8):
            test_index = np.flatnonzero(frame[fold_column].eq(fold).to_numpy())
            train_index = np.flatnonzero(~frame[fold_column].eq(fold).to_numpy())
            y_train = frame.loc[train_index, "failure"].to_numpy(dtype=bool)
            train_frame = frame.iloc[train_index].reset_index(drop=True)
            test_frame = frame.iloc[test_index].reset_index(drop=True)
            shifted_structure_train = shifted_sibling_features(
                train_frame, structures[train_index]
            )
            shifted_structure_test = shifted_sibling_features(
                test_frame, structures[test_index]
            )
            shifted_cell_train = shifted_sibling_features(
                train_frame, cells[train_index]
            )
            shifted_cell_test = shifted_sibling_features(
                test_frame, cells[test_index]
            )
            train_states = frame.loc[train_index, "init_state_id"].to_numpy(dtype=int)
            test_states = frame.loc[test_index, "init_state_id"].to_numpy(dtype=int)
            global_train, global_test = initial.prior_scores(
                y_train, train_states, test_states, False
            )
            state_train, state_test = initial.prior_scores(
                y_train, train_states, test_states, include_state
            )
            structure_q0_train, structure_q0_test = standardize_block(
                structures[train_index][:, q0_structure],
                structures[test_index][:, q0_structure],
            )
            structure_all_train, structure_all_test = standardize_block(
                structures[train_index], structures[test_index]
            )
            shifted_q0_train, shifted_q0_test = standardize_block(
                shifted_structure_train[:, q0_structure],
                shifted_structure_test[:, q0_structure],
            )
            shifted_all_train, shifted_all_test = standardize_block(
                shifted_structure_train, shifted_structure_test
            )
            aggregate_q0_train, aggregate_q0_test = early.reduce_block(
                aggregates[train_index][:, q0_aggregate],
                aggregates[test_index][:, q0_aggregate],
                12,
                seed + fold + (0 if split == "heldout_seed" else 100),
            )
            aggregate_all_train, aggregate_all_test = early.reduce_block(
                aggregates[train_index],
                aggregates[test_index],
                12,
                seed + fold + (1000 if split == "heldout_seed" else 1100),
            )
            nested_blocks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            selected_features: dict[str, str] = {}
            for scope, candidates in (("q0", q0_cells), ("q0_q3", all_cells)):
                aligned_effect = effect_vector(
                    train_frame, cells[train_index][:, candidates]
                )
                aligned_index = int(candidates[np.argmax(np.abs(aligned_effect))])
                aligned_train, aligned_test = standardize_block(
                    cells[train_index, aligned_index : aligned_index + 1],
                    cells[test_index, aligned_index : aligned_index + 1],
                )
                aligned_model = f"nested_best_{scope}_cell"
                nested_blocks[aligned_model] = (aligned_train, aligned_test)
                selected_features[aligned_model] = cell_schema[aligned_index].key

                shifted_effect = effect_vector(
                    train_frame, shifted_cell_train[:, candidates]
                )
                shifted_index = int(candidates[np.argmax(np.abs(shifted_effect))])
                shifted_train, shifted_test = standardize_block(
                    shifted_cell_train[:, shifted_index : shifted_index + 1],
                    shifted_cell_test[:, shifted_index : shifted_index + 1],
                )
                shifted_model = f"nested_best_{scope}_cell_shifted_sibling"
                nested_blocks[shifted_model] = (shifted_train, shifted_test)
                selected_features[shifted_model] = cell_schema[shifted_index].key
            model_blocks = {
                "structure_q0": (structure_q0_train, structure_q0_test),
                "structure_q0_q3": (structure_all_train, structure_all_test),
                "structure_q0_shifted_sibling": (
                    shifted_q0_train,
                    shifted_q0_test,
                ),
                "structure_q0_q3_shifted_sibling": (
                    shifted_all_train,
                    shifted_all_test,
                ),
                "aggregate_q0": (aggregate_q0_train, aggregate_q0_test),
                "aggregate_q0_plus_structure": (
                    np.column_stack((aggregate_q0_train, structure_q0_train)),
                    np.column_stack((aggregate_q0_test, structure_q0_test)),
                ),
                "aggregate_q0_q3": (aggregate_all_train, aggregate_all_test),
                "aggregate_q0_q3_plus_structure": (
                    np.column_stack((aggregate_all_train, structure_all_train)),
                    np.column_stack((aggregate_all_test, structure_all_test)),
                ),
                **nested_blocks,
            }
            scores = {
                "global_prior": global_test,
                "state_prior": state_test,
            }
            for model, (train_values, test_values) in model_blocks.items():
                score = robust_offset_scores(
                    train_values,
                    y_train,
                    test_values,
                    state_train,
                    state_test,
                    0.1,
                )
                scores[model] = score
            for position, row_index in enumerate(test_index):
                item = frame.iloc[row_index]
                for model, score in scores.items():
                    rows.append(
                        {
                            "split": split,
                            "fold": fold,
                            "episode": int(item["episode"]),
                            "init_state_id": int(item["init_state_id"]),
                            "flow_noise_seed": int(item["flow_noise_seed"]),
                            "truth_failure": bool(item["failure"]),
                            "model": model,
                            "score_failure": float(score[position]),
                            "selected_feature": selected_features.get(model, ""),
                        }
                    )
    return pd.DataFrame(rows)


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, model), group in predictions.groupby(["split", "model"], sort=False):
        truth = group["truth_failure"].to_numpy(dtype=float)
        score = group["score_failure"].to_numpy(dtype=float)
        rows.append(
            {
                "split": split,
                "model": model,
                "episodes": len(group),
                "failures": int(truth.sum()),
                "brier": float(np.mean(np.square(score - truth))),
            }
        )
    return pd.DataFrame(rows)


def prediction_deltas(
    predictions: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    comparisons = {
        "state_vs_structure_q0": ("state_prior", "structure_q0"),
        "state_vs_structure_q0_q3": ("state_prior", "structure_q0_q3"),
        "aggregate_q0_structure_increment": (
            "aggregate_q0",
            "aggregate_q0_plus_structure",
        ),
        "aggregate_q0_q3_structure_increment": (
            "aggregate_q0_q3",
            "aggregate_q0_q3_plus_structure",
        ),
        "q1_q3_increment_over_q0_structure": (
            "structure_q0",
            "structure_q0_q3",
        ),
        "initial_state_prior_vs_global": ("global_prior", "state_prior"),
        "shifted_sibling_vs_aligned_q0": (
            "structure_q0_shifted_sibling",
            "structure_q0",
        ),
        "shifted_sibling_vs_aligned_q0_q3": (
            "structure_q0_q3_shifted_sibling",
            "structure_q0_q3",
        ),
        "state_vs_nested_best_q0_cell": ("state_prior", "nested_best_q0_cell"),
        "state_vs_nested_best_q0_q3_cell": (
            "state_prior",
            "nested_best_q0_q3_cell",
        ),
        "nested_q1_q3_increment_over_q0": (
            "nested_best_q0_cell",
            "nested_best_q0_q3_cell",
        ),
        "shifted_sibling_vs_aligned_nested_q0": (
            "nested_best_q0_cell_shifted_sibling",
            "nested_best_q0_cell",
        ),
        "shifted_sibling_vs_aligned_nested_q0_q3": (
            "nested_best_q0_q3_cell_shifted_sibling",
            "nested_best_q0_q3_cell",
        ),
    }
    index = [
        "split",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "truth_failure",
    ]
    wide = predictions.pivot(index=index, columns="model", values="score_failure").reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for split, group in wide.groupby("split", sort=False):
        truth = group["truth_failure"].to_numpy(dtype=float)
        states = group["init_state_id"].to_numpy(dtype=int)
        unique_states = np.unique(states)
        for name, (baseline, candidate) in comparisons.items():
            delta = np.square(group[baseline].to_numpy() - truth) - np.square(
                group[candidate].to_numpy() - truth
            )
            samples = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                picked = rng.choice(unique_states, size=len(unique_states), replace=True)
                samples[draw] = np.mean(
                    np.concatenate([delta[states == state] for state in picked])
                )
            rows.append(
                {
                    "split": split,
                    "comparison": name,
                    "brier_gain": float(delta.mean()),
                    "ci95_low": float(np.quantile(samples, 0.025)),
                    "ci95_high": float(np.quantile(samples, 0.975)),
                    "states_positive": int(
                        sum(delta[states == state].mean() > 0 for state in unique_states)
                    ),
                    "states_total": len(unique_states),
                    "bootstrap_draws": draws,
                }
            )
    return pd.DataFrame(rows)


def plot_structure_effects(
    long_effects: pd.DataFrame, gos_effects: pd.DataFrame, path: Path
) -> None:
    selected = long_effects.copy().sort_values("effect_sigma")
    positions = np.arange(len(selected))
    gos = gos_effects.set_index("key").loc[selected["key"]]
    fig, axis = plt.subplots(figsize=(9.5, 7.2))
    axis.scatter(
        selected["effect_sigma"], positions + 0.12, label="Long: within initial state", color="#0072B2"
    )
    axis.scatter(
        gos["effect_sigma"], positions - 0.12, label="G/O/S: within task", color="#D55E00"
    )
    axis.axvline(0, color="#777777", linewidth=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(selected["key"])
    axis.set_xlabel("Failure minus success effect (residual SD)")
    axis.set_title("Outcome effect of frozen chunk-position summaries")
    axis.legend(frameon=False)
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_cell_heatmap(effects: pd.DataFrame, metric: str, path: Path) -> None:
    selected = effects[effects["metric"].eq(metric)]
    adjacent = metric in token.ADJACENT_METRICS
    token_positions = range(2, 11) if adjacent else range(1, 11)
    limit = float(np.nanmax(np.abs(selected["effect_sigma"])))
    fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.5), sharex=True, sharey=True)
    image = None
    for row, group_name in enumerate(token.LAYER_GROUPS):
        for query in range(4):
            cell = selected[
                selected["layer_group"].eq(group_name)
                & selected["relative_query"].eq(query)
            ]
            ordered = []
            for denoise in range(10):
                row_values = []
                for ending_token in token_positions:
                    if adjacent:
                        key = (
                            f"q{query}|{group_name}|{metric}|d{denoise}|"
                            f"t{ending_token - 1}_to_t{ending_token}"
                        )
                    else:
                        key = f"q{query}|{group_name}|{metric}|d{denoise}|t{ending_token}"
                    row_values.append(float(cell.loc[cell["key"].eq(key), "effect_sigma"].iloc[0]))
                ordered.append(row_values)
            image = axes[row, query].imshow(
                np.asarray(ordered),
                origin="lower",
                aspect="auto",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axes[row, query].set_title(f"q{query}")
            if query == 0:
                axes[row, query].set_ylabel(f"{group_name}\nDenoise")
            if row == 1:
                axes[row, query].set_xlabel("Ending token")
                axes[row, query].set_xticks(range(len(token_positions)))
                axes[row, query].set_xticklabels(token_positions)
    assert image is not None
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02, label="failure - success effect (SD)")
    fig.suptitle(f"Long within-state {metric} effects", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.10, top=0.91, wspace=0.10, hspace=0.18)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_brier(deltas: pd.DataFrame, path: Path) -> None:
    order = [
        "state_vs_structure_q0",
        "state_vs_structure_q0_q3",
        "aggregate_q0_structure_increment",
        "aggregate_q0_q3_structure_increment",
        "q1_q3_increment_over_q0_structure",
        "shifted_sibling_vs_aligned_q0",
        "shifted_sibling_vs_aligned_q0_q3",
        "state_vs_nested_best_q0_cell",
        "state_vs_nested_best_q0_q3_cell",
        "shifted_sibling_vs_aligned_nested_q0",
        "shifted_sibling_vs_aligned_nested_q0_q3",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.4), sharey=True)
    for axis, split in zip(axes, ("heldout_seed", "heldout_init"), strict=True):
        selected = deltas[
            deltas["split"].eq(split) & deltas["comparison"].isin(order)
        ].set_index("comparison").loc[order]
        positions = np.arange(len(selected))
        values = selected["brier_gain"].to_numpy()
        errors = np.vstack(
            (
                values - selected["ci95_low"].to_numpy(),
                selected["ci95_high"].to_numpy() - values,
            )
        )
        axis.errorbar(values, positions, xerr=errors, fmt="o", color="#0072B2", capsize=3)
        axis.axvline(0, color="#777777", linewidth=0.8)
        axis.set_title(split)
        axis.set_xlabel("Brier gain (positive is useful)")
        axis.grid(axis="x", alpha=0.2)
        axis.set_yticks(positions)
        axis.set_yticklabels(order)
    fig.suptitle("Out-of-fold value of chunk-position structure", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def report_table_effects(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| score | effect (SD) | 95% CI | raw p | global max-T p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in frame.sort_values("key").itertuples(index=False):
        lines.append(
            f"| `{row.key}` | {row.effect_sigma:+.3f} | "
            f"[{row.effect_ci95_low:+.3f},{row.effect_ci95_high:+.3f}] | "
            f"{row.permutation_p_raw:.4f} | {row.permutation_p_max_global:.4f} |"
        )
    return lines


def write_report(
    out: Path,
    primary_long: pd.DataFrame,
    full_cells: pd.DataFrame,
    discovery_cells: pd.DataFrame,
    frozen: pd.DataFrame,
    prediction_metric: pd.DataFrame,
    deltas: pd.DataFrame,
    primary_gos: pd.DataFrame,
    audits: dict[str, Any],
    effect_rho: float,
    split_cell_effects: pd.DataFrame,
    winner_state_effects: pd.DataFrame,
    selections: pd.DataFrame,
    discovery_validation_rho: float,
    long_gos_cell_rho: float,
) -> None:
    best_primary = primary_long.iloc[
        int(np.argmax(np.abs(primary_long["effect_sigma"].to_numpy())))
    ]
    best_cell = full_cells.iloc[
        int(np.argmax(np.abs(full_cells["effect_sigma"].to_numpy())))
    ]
    best_discovery = discovery_cells.iloc[
        int(np.argmax(np.abs(discovery_cells["effect_sigma"].to_numpy())))
    ]
    split_winner = split_cell_effects.set_index("key").loc[best_cell["key"]]
    same_state_direction = int(
        np.sum(
            np.sign(winner_state_effects["failure_minus_success"].to_numpy())
            == np.sign(best_cell["effect_sigma"])
        )
    )
    delta_index = deltas.set_index(["split", "comparison"])
    nested_seed = delta_index.loc[("heldout_seed", "state_vs_nested_best_q0_cell")]
    nested_shift = delta_index.loc[
        ("heldout_seed", "shifted_sibling_vs_aligned_nested_q0")
    ]
    summary_seed = delta_index.loc[("heldout_seed", "state_vs_structure_q0")]
    unseen_init = delta_index.loc[("heldout_init", "state_vs_structure_q0")]
    seed_selections = selections[
        selections["split"].eq("heldout_seed")
        & selections["model"].eq("nested_best_q0_cell")
    ]["selected_feature"].value_counts()
    selection_text = "; ".join(
        f"`{key}` {count}/8" for key, count in seed_selections.items()
    )
    gos_full_winner = frozen[
        frozen["dataset"].eq("gos_frozen_long_full_winner")
    ].iloc[0]
    lines = [
        "# Chunk 逐 token 路由是否含有成败信息",
        "",
        "## 结论",
        "",
        "**有一个局部 q0 候选，但广义 chunk 结构还不能稳定预测单条分支成败。**",
        "上一轮冻结的 U 形、前后层差、去噪稳定化等 17 个结构分数没有一个通过全局 0.05；"
        f"最强 `q0 soft-tail` 只有 `p={best_primary['permutation_p_max_global']:.4f}`。",
        f"全位置扫描找到 `{best_cell['key']}`，效应 {best_cell['effect_sigma']:+.3f} SD、"
        f"3,760 格 max-T `p={best_cell['permutation_p_max_global']:.4f}`。训练折内重新选点后，"
        f"q0 cell 相对初态先验的 Brier gain 为 {nested_seed['brier_gain']:+.5f} "
        f"[{nested_seed['ci95_low']:+.5f},{nested_seed['ci95_high']:+.5f}]；相对错配 sibling 为 "
        f"{nested_shift['brier_gain']:+.5f} "
        f"[{nested_shift['ci95_low']:+.5f},{nested_shift['ci95_high']:+.5f}]。",
        f"预定义 q0 总结构相对初态先验只有 {summary_seed['brier_gain']:+.5f} "
        f"[{summary_seed['ci95_low']:+.5f},{summary_seed['ci95_high']:+.5f}]；在未见初态上趋势为 "
        f"{unseen_init['brier_gain']:+.5f} "
        f"[{unseen_init['ci95_low']:+.5f},{unseen_init['ci95_high']:+.5f}]。",
        "",
        "本报告区分统计差异、冻结复验和折外预测。所有主检验只使用 q0-q3；没有使用 rollout",
        "长度、remaining time、terminal window、动作、hidden state 或 simulator state。",
        "",
        "## 主数据与正负对照",
        "",
        "- Long 共 512 条 rollout、216 失败、296 成功；16 个初态中 13 个同时有成败。",
        f"- 同初态主检验使用 {audits['primary_long']['episodes']} 条、"
        f"{audits['primary_long']['mixed_strata']} 个 mixed 初态。统计单位是初态，标签只在初态内置换。",
        "- heldout-seed 预测把 32 个 noise seeds 分成 8 折；测试 seed 在训练中从未出现，但训练折可估计初态先验。",
        "- heldout-init 完整留出两个初态，检验结构能否迁移到未见场景。",
        "- 正对照是 heldout-seed 的 initial-state prior；负对照是初态内标签置换的 max-T 零分布。",
        "- 第二个负对照把每条分支的结构替换成同初态下下一个 noise seed 的 sibling 路由；它保留初态，破坏 branch 对齐。",
        "",
        "## 无监督发现的结构分数",
        "",
        "下面这些分数由上一轮不看标签的 token 曲线定义。正 effect 表示失败分支的该结构更强。",
        "",
        *report_table_effects(primary_long),
        "",
        f"绝对效应最大的是 `{best_primary['key']}`：{best_primary['effect_sigma']:+.3f} SD，"
        f"17 分数全局 max-T `p={best_primary['permutation_p_max_global']:.4f}`。",
        "",
        "## 逐位置全扫描与冻结复验",
        "",
        f"全 416 条 mixed-state 分支扫描 3,760 个 cell，最大为 `{best_cell['key']}`："
        f"{best_cell['effect_sigma']:+.3f} SD，全局 max-T `p={best_cell['permutation_p_max_global']:.4f}`。",
        f"前 16 seeds 的 discovery 最大 cell 是 `{best_discovery['key']}`："
        f"{best_discovery['effect_sigma']:+.3f} SD，全局 `p={best_discovery['permutation_p_max_global']:.4f}`。",
        "冻结结果：",
        "",
        "| dataset | cell | effect (SD) | raw p | global p |",
        "|---|---|---:|---:|---:|",
    ]
    for row in frozen.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | `{row.key}` | {row.effect_sigma:+.3f} | "
            f"{row.permutation_p_raw:.4f} | {row.permutation_p_max_global:.4f} |"
        )
    lines += [
        "",
        f"全样本赢家 `{best_cell['key']}` 在 seed A/B 的描述性效应分别为 "
        f"{split_winner['seed_A_effect']:+.3f}/{split_winner['seed_B_effect']:+.3f} SD，"
        f"13 个 mixed 初态中 {same_state_direction}/13 与总体同方向。这个拆半发生在全样本选点之后，"
        "所以是稳定性诊断，不是独立冻结检验。G/O/S 对该 Long 赢家另作独立 task 内方向检查。",
        f"heldout-seed 每折训练内的 q0 选点频次为：{selection_text}。",
        f"但 3,760 个 cell 的 seed A/B 效应排名相关只有 `rho={discovery_validation_rho:+.3f}`，"
        "说明除这个局部簇外，整张效应图并不稳定。",
        "",
        "## 折外预测：结构有没有实际增量",
        "",
        "Brier gain 为正才有用。`state_vs_structure_q0` 是第一次推理的逐 token 结构相对初态先验；",
        "`aggregate_*_structure_increment` 检验逐 token 位置是否超过旧的 token-平均路由量。",
        "`shifted_sibling_vs_aligned_*` 为正才表示真实 branch 路由优于同初态错配路由。",
        "`nested_best_*_cell` 的位置选择完全发生在训练折内，因此是对 3,760 格探索峰的无泄漏预测检查。",
        "",
        "| split | comparison | Brier gain | 95% state-bootstrap CI | positive states |",
        "|---|---|---:|---:|---:|",
    ]
    for row in deltas.itertuples(index=False):
        lines.append(
            f"| {row.split} | `{row.comparison}` | {row.brier_gain:+.5f} | "
            f"[{row.ci95_low:+.5f},{row.ci95_high:+.5f}] | "
            f"{row.states_positive}/{row.states_total} |"
        )
    lines += [
        "",
        "模型的绝对折外 Brier：",
        "",
        "| split | model | Brier |",
        "|---|---|---:|",
    ]
    for row in prediction_metric.sort_values(["split", "brier"]).itertuples(index=False):
        lines.append(f"| {row.split} | `{row.model}` | {row.brier:.5f} |")
    lines += [
        "",
        "## Goal/Object/Spatial 外部方向检查",
        "",
        f"G/O/S 有 {audits['primary_gos']['episodes']} 条来自 mixed tasks 的 rollout、"
        f"{audits['primary_gos']['failures']} 次失败。这里只有 task 固定效应，不能控制初态。",
        f"17 个结构效应与 Long 的 Spearman `rho={effect_rho:+.3f}`。不同 checkpoint、低失败率和"
        "单次初态意味着这只能检查方向，不能替代 Long 的 sibling 正负对照。",
        f"更细的 3,760-cell Long/GOS 效应相关只有 `rho={long_gos_cell_rho:+.3f}`，且 Long 赢家"
        f"在 G/O/S 中方向相反、`p={gos_full_winner['permutation_p_raw']:.4f}`，所以目前应把它视为 Long 特定候选。",
        "",
        "## 解释边界",
        "",
        "即使某个 q1-q3 结构能预测，它也已经包含前几个 action chunk 执行后的状态分叉；只有 q0",
        "可以称为执行前信号。binary failure 在这个 Long 数据里等同最终未成功并跑满时限，但本分析",
        "没有把时长输入模型。高 hard-expert switch 仍可能来自近似并列专家换序，因此主解释优先看",
        "soft-route Hellinger 和折外 Brier，而不是专家 ID。",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (
        long_frame,
        long_cells,
        cell_schema,
        long_primary,
        primary_keys,
        long_predictor,
        predictor_keys,
        long_aggregate,
        aggregate_meta,
    ) = load_long(args.long_run)

    primary_meta = score_metadata(primary_keys)
    cells_meta = cell_metadata(cell_schema)
    primary_long, primary_long_audit = effect_scan(
        long_frame,
        long_primary,
        primary_meta,
        args.primary_permutations,
        args.bootstraps,
        args.seed,
        "long_full_within_init",
    )
    full_cells, full_cells_audit = effect_scan(
        long_frame,
        long_cells,
        cells_meta,
        args.cell_permutations,
        args.bootstraps,
        args.seed + 10,
        "long_full_within_init",
    )

    half_a = long_frame["flow_noise_seed"].lt(1016).to_numpy()
    half_b = ~half_a
    discovery_cells, discovery_audit = effect_scan(
        long_frame.loc[half_a].reset_index(drop=True),
        long_cells[half_a],
        cells_meta,
        args.cell_permutations,
        args.bootstraps,
        args.seed + 20,
        "long_seed_A_discovery",
    )
    discovery_index = int(
        np.argmax(np.abs(discovery_cells["effect_sigma"].to_numpy()))
    )
    selected_meta = [cells_meta[discovery_index]]
    validation_cell, validation_audit = effect_scan(
        long_frame.loc[half_b].reset_index(drop=True),
        long_cells[half_b, discovery_index : discovery_index + 1],
        selected_meta,
        args.primary_permutations,
        args.bootstraps,
        args.seed + 30,
        "long_seed_B_frozen",
    )
    validation_effects = effect_vector(
        long_frame.loc[half_b].reset_index(drop=True), long_cells[half_b]
    )
    split_cell_effects = pd.DataFrame(
        {
            "key": [item.key for item in cell_schema],
            "full_effect": full_cells["effect_sigma"].to_numpy(),
            "seed_A_effect": discovery_cells["effect_sigma"].to_numpy(),
            "seed_B_effect": validation_effects,
        }
    )
    discovery_validation_rho = float(
        spearmanr(discovery_cells["effect_sigma"], validation_effects).statistic
    )
    full_winner_index = int(
        np.argmax(np.abs(full_cells["effect_sigma"].to_numpy()))
    )
    winner_state_rows = []
    for init_state, group in long_frame.groupby("init_state_id", sort=True):
        if group["failure"].nunique() != 2:
            continue
        indices = group.index.to_numpy(dtype=np.int64)
        failures = group["failure"].to_numpy(dtype=bool)
        values = long_cells[indices, full_winner_index]
        winner_state_rows.append(
            {
                "init_state_id": int(init_state),
                "key": cell_schema[full_winner_index].key,
                "failures": int(failures.sum()),
                "successes": int((~failures).sum()),
                "failure_minus_success": float(
                    values[failures].mean() - values[~failures].mean()
                ),
            }
        )
    winner_state_effects = pd.DataFrame(winner_state_rows)

    predictions = outcome_predictions(
        long_frame,
        long_predictor,
        predictor_keys,
        long_aggregate,
        aggregate_meta,
        long_cells,
        cell_schema,
        args.seed + 100,
    )
    metrics = prediction_metrics(predictions)
    deltas = prediction_deltas(
        predictions, args.prediction_bootstraps, args.seed + 200
    )
    selections = (
        predictions[predictions["selected_feature"].ne("")][
            ["split", "fold", "model", "selected_feature"]
        ]
        .drop_duplicates()
        .sort_values(["split", "model", "fold"])
        .reset_index(drop=True)
    )

    gos_frame, gos_cells, gos_primary, gos_keys = load_gos(args.gos_corpus)
    if gos_keys != primary_keys:
        raise ValueError("Long and GOS primary score schemas differ")
    primary_gos, primary_gos_audit = effect_scan(
        gos_frame,
        gos_primary,
        primary_meta,
        args.primary_permutations,
        args.bootstraps,
        args.seed + 300,
        "gos_within_task",
    )
    gos_selected, gos_selected_audit = effect_scan(
        gos_frame,
        gos_cells[:, discovery_index : discovery_index + 1],
        selected_meta,
        args.primary_permutations,
        args.bootstraps,
        args.seed + 310,
        "gos_frozen_discovery_cell",
    )
    full_winner_meta = [cells_meta[full_winner_index]]
    gos_full_winner, gos_full_winner_audit = effect_scan(
        gos_frame,
        gos_cells[:, full_winner_index : full_winner_index + 1],
        full_winner_meta,
        args.primary_permutations,
        args.bootstraps,
        args.seed + 320,
        "gos_frozen_long_full_winner",
    )
    effect_rho = float(
        spearmanr(primary_long["effect_sigma"], primary_gos["effect_sigma"]).statistic
    )
    gos_cell_effects = effect_vector(gos_frame, gos_cells)
    long_gos_cell_rho = float(
        spearmanr(full_cells["effect_sigma"], gos_cell_effects).statistic
    )

    frozen = pd.concat(
        (
            discovery_cells.iloc[[discovery_index]],
            validation_cell,
            gos_selected,
            gos_full_winner,
        ),
        ignore_index=True,
    )
    cohort_long = long_frame.copy()
    cohort_gos = gos_frame.copy()
    primary_values_long = pd.DataFrame(long_primary, columns=primary_keys)
    primary_values_long.insert(0, "episode", long_frame["episode"].to_numpy())
    primary_values_gos = pd.DataFrame(gos_primary, columns=primary_keys)
    primary_values_gos.insert(0, "task", gos_frame["task"].to_numpy())
    primary_values_gos.insert(0, "episode", gos_frame["episode"].to_numpy())

    cohort_long.to_csv(args.out / "cohort_long.csv", index=False)
    cohort_gos.to_csv(args.out / "cohort_gos.csv", index=False)
    primary_values_long.to_csv(args.out / "structure_scores_long.csv.gz", index=False)
    primary_values_gos.to_csv(args.out / "structure_scores_gos.csv.gz", index=False)
    primary_long.to_csv(args.out / "primary_structure_effects_long.csv", index=False)
    primary_gos.to_csv(args.out / "primary_structure_effects_gos.csv", index=False)
    full_cells.to_csv(args.out / "token_cell_effects_long.csv", index=False)
    discovery_cells.to_csv(args.out / "token_cell_effects_discovery.csv", index=False)
    split_cell_effects.to_csv(args.out / "token_cell_split_half_effects.csv", index=False)
    winner_state_effects.to_csv(args.out / "full_winner_state_effects.csv", index=False)
    frozen.to_csv(args.out / "frozen_cell_validation.csv", index=False)
    predictions.to_csv(args.out / "prediction_oof.csv.gz", index=False)
    metrics.to_csv(args.out / "prediction_metrics.csv", index=False)
    deltas.to_csv(args.out / "prediction_deltas.csv", index=False)
    selections.to_csv(args.out / "nested_cell_selections.csv", index=False)

    plot_structure_effects(
        primary_long, primary_gos, args.out / "structure_effects.png"
    )
    plot_cell_heatmap(
        full_cells,
        "adjacent_hellinger",
        args.out / "cell_effect_heatmap.png",
    )
    plot_cell_heatmap(
        full_cells,
        "entropy_centered",
        args.out / "entropy_position_effect_heatmap.png",
    )
    plot_brier(deltas, args.out / "prediction_brier_gains.png")

    audits = {
        "primary_long": primary_long_audit,
        "full_cells": full_cells_audit,
        "discovery": discovery_audit,
        "validation": validation_audit,
        "primary_gos": primary_gos_audit,
        "gos_selected_cell": gos_selected_audit,
        "gos_full_winner": gos_full_winner_audit,
    }
    write_report(
        args.out,
        primary_long,
        full_cells,
        discovery_cells,
        frozen,
        metrics,
        deltas,
        primary_gos,
        audits,
        effect_rho,
        split_cell_effects,
        winner_state_effects,
        selections,
        discovery_validation_rho,
        long_gos_cell_rho,
    )
    strongest_primary = primary_long.iloc[
        int(np.argmax(np.abs(primary_long["effect_sigma"].to_numpy())))
    ]
    strongest_cell = full_cells.iloc[
        int(np.argmax(np.abs(full_cells["effect_sigma"].to_numpy())))
    ]
    summary = {
        "schema": "himoe.chunk_token_outcomes.v1",
        "outcome": "eventual success versus timeout failure",
        "forbidden_inputs_used": False,
        "queries": [0, 1, 2, 3],
        "denoise_steps": 10,
        "action_token_positions": 10,
        "primary_structure_scores": len(primary_keys),
        "token_cells": len(cell_schema),
        "audits": audits,
        "strongest_primary": plain(strongest_primary.to_dict()),
        "strongest_full_cell": plain(strongest_cell.to_dict()),
        "discovery_validation_effect_rho": discovery_validation_rho,
        "long_gos_primary_effect_rho": effect_rho,
        "long_gos_cell_effect_rho": long_gos_cell_rho,
        "prediction_brier_deltas": plain(deltas.to_dict(orient="records")),
        "nested_cell_selections": plain(selections.to_dict(orient="records")),
    }
    (args.out / "summary.json").write_text(
        json.dumps(plain(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifacts = sorted(
        path for path in args.out.iterdir() if path.name != "checksums.sha256"
    )
    (args.out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="ascii",
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
