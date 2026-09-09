#!/usr/bin/env python3
"""Measure success/failure information in MoE routes over the full rollout.

The primary estimand is out-of-fold ROC AUC within an initial state: can a
route score rank a failed noise-seed rollout above a successful sibling?  Every
AUC is computed at a fixed policy query.  Episode length, remaining time,
terminal windows, actions, hidden state, and simulator state are never model
inputs.

Queries q0-q34 are observed for all 512 Long rollouts.  Later queries are
reported separately as online risk sets containing only branches still active
at that query.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import analyze_chunk_token_outcomes as previous
import analyze_initial_state_early_moe_signal as initial
import analyze_state_route_layer_denoise_effects as route_effects


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = (
    HERE.parent
    / "VLA_MUI_HUB/cache/HiMoE-VLA/libero_long"
    / "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/right-16x32"
)
DEFAULT_OUT = HERE / "analysis/full-chunk-outcome-auc-20260829"
SEED = 20260829
LAYER_GROUPS = {
    "front_2_5": np.arange(0, 4, dtype=np.int64),
    "back_12_15": np.arange(4, 8, dtype=np.int64),
}
CURRENT_FAMILIES = (
    "top1_centered",
    "entropy_centered",
    "adjacent_hellinger",
    "top1_switch_rate",
    "top4_turnover",
)
TEMPORAL_FAMILIES = (
    "query_hellinger",
    "query_top4_turnover",
    "nonlocal_return_hellinger",
)
COMMON_QUERY_MAX = 34
RISK_QUERY_MAX = 46
SHIFT_OFFSETS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--rebuild-cache", action="store_true")
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


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = probability.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(probability)):
        raise ValueError("invalid router probability")
    return probability / mass


def entropy(probability: np.ndarray) -> np.ndarray:
    return -np.sum(
        probability * np.log(np.maximum(probability, 1e-12)), axis=-1
    ) / np.log(probability.shape[-1])


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        np.maximum(
            0.0,
            0.5
            * np.sum(
                np.square(
                    np.sqrt(np.maximum(left, 0.0))
                    - np.sqrt(np.maximum(right, 0.0))
                ),
                axis=-1,
            ),
        )
    )


def top4_turnover(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_top = np.argpartition(left, -4, axis=-1)[..., -4:]
    right_top = np.argpartition(right, -4, axis=-1)[..., -4:]
    overlap = (
        left_top[..., :, None] == right_top[..., None, :]
    ).any(axis=-1).sum(axis=-1)
    return 1.0 - overlap.astype(np.float32) / 4.0


def schemas() -> dict[str, tuple[list[str], list[str], np.ndarray]]:
    aggregate_names: list[str] = []
    aggregate_families: list[str] = []
    for metric in CURRENT_FAMILIES:
        display_metric = {
            "top1_centered": "top1",
            "entropy_centered": "entropy",
        }.get(metric, metric)
        for group in LAYER_GROUPS:
            for denoise in range(10):
                aggregate_names.append(
                    f"{display_metric}|{group}|d{denoise}|token_mean"
                )
                aggregate_families.append(metric)

    cell_names: list[str] = []
    cell_families: list[str] = []
    cell_tokens: list[int] = []
    for metric in CURRENT_FAMILIES[:2]:
        for group in LAYER_GROUPS:
            for denoise in range(10):
                for token in range(1, 11):
                    cell_names.append(f"{metric}|{group}|d{denoise}|t{token}")
                    cell_families.append(metric)
                    cell_tokens.append(token)
    for metric in CURRENT_FAMILIES[2:]:
        for group in LAYER_GROUPS:
            for denoise in range(10):
                for token in range(2, 11):
                    cell_names.append(
                        f"{metric}|{group}|d{denoise}|t{token - 1}_to_t{token}"
                    )
                    cell_families.append(metric)
                    cell_tokens.append(token)

    temporal_names: list[str] = []
    temporal_families: list[str] = []
    temporal_tokens: list[int] = []
    temporal_aggregate_names: list[str] = []
    temporal_aggregate_families: list[str] = []
    for metric in TEMPORAL_FAMILIES:
        for group in LAYER_GROUPS:
            for denoise in range(10):
                temporal_aggregate_names.append(
                    f"{metric}|{group}|d{denoise}|token_mean"
                )
                temporal_aggregate_families.append(metric)
                for token in range(1, 11):
                    temporal_names.append(f"{metric}|{group}|d{denoise}|t{token}")
                    temporal_families.append(metric)
                    temporal_tokens.append(token)
    return {
        "aggregate": (
            aggregate_names,
            aggregate_families,
            np.zeros(len(aggregate_names), dtype=np.int16),
        ),
        "cell": (
            cell_names,
            cell_families,
            np.asarray(cell_tokens, dtype=np.int16),
        ),
        "temporal": (
            temporal_names,
            temporal_families,
            np.asarray(temporal_tokens, dtype=np.int16),
        ),
        "temporal_aggregate": (
            temporal_aggregate_names,
            temporal_aggregate_families,
            np.zeros(len(temporal_aggregate_names), dtype=np.int16),
        ),
    }


def current_route_features(
    raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return token-averaged and token-position features for route rows."""
    probability = normalize_probability(raw)
    if probability.ndim != 5 or probability.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"unexpected route block {probability.shape}")
    action = probability[..., 1:, :]
    metric_groups: dict[str, list[np.ndarray]] = {
        family: [] for family in CURRENT_FAMILIES
    }
    for layer_axes in LAYER_GROUPS.values():
        selected = action[:, layer_axes]
        top1 = selected.max(axis=-1).mean(axis=1)
        ent = entropy(selected).mean(axis=1)
        adjacent = hellinger(selected[..., 1:, :], selected[..., :-1, :]).mean(
            axis=1
        )
        switch = (
            np.argmax(selected[..., 1:, :], axis=-1)
            != np.argmax(selected[..., :-1, :], axis=-1)
        ).mean(axis=1)
        turnover = top4_turnover(
            selected[..., 1:, :], selected[..., :-1, :]
        ).mean(axis=1)
        metric_groups["top1_centered"].append(top1)
        metric_groups["entropy_centered"].append(ent)
        metric_groups["adjacent_hellinger"].append(adjacent)
        metric_groups["top1_switch_rate"].append(switch)
        metric_groups["top4_turnover"].append(turnover)

    aggregate_parts = []
    cell_parts = []
    for metric in CURRENT_FAMILIES:
        values = np.stack(metric_groups[metric], axis=1)
        aggregate_parts.append(values.mean(axis=-1).reshape(len(raw), -1))
        if metric in CURRENT_FAMILIES[:2]:
            values = values - values.mean(axis=-1, keepdims=True)
        cell_parts.append(values.reshape(len(raw), -1))
    return (
        np.concatenate(aggregate_parts, axis=1).astype(np.float32),
        np.concatenate(cell_parts, axis=1).astype(np.float32),
    )


def temporal_route_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cross-query changes, preserving each action-token position."""
    probability = normalize_probability(raw)
    action = probability[..., 1:, :]
    queries = len(action)
    metric_groups: dict[str, list[np.ndarray]] = {
        family: [] for family in TEMPORAL_FAMILIES
    }
    for layer_axes in LAYER_GROUPS.values():
        selected = action[:, layer_axes]
        query_change = np.zeros((queries, 10, 10), dtype=np.float32)
        hard_change = np.zeros_like(query_change)
        nonlocal_return = np.zeros_like(query_change)
        if queries > 1:
            query_change[1:] = hellinger(selected[1:], selected[:-1]).mean(axis=1)
            hard_change[1:] = top4_turnover(selected[1:], selected[:-1]).mean(
                axis=1
            )
        for query in range(2, queries):
            start = max(0, query - 8)
            candidates = [
                hellinger(selected[query], selected[past]).mean(axis=0)
                for past in range(start, query - 1)
            ]
            nonlocal_return[query] = np.min(np.stack(candidates), axis=0)
        metric_groups["query_hellinger"].append(query_change)
        metric_groups["query_top4_turnover"].append(hard_change)
        metric_groups["nonlocal_return_hellinger"].append(nonlocal_return)

    aggregate_parts = []
    position_parts = []
    for metric in TEMPORAL_FAMILIES:
        values = np.stack(metric_groups[metric], axis=1)
        aggregate_parts.append(values.mean(axis=-1).reshape(queries, -1))
        position_parts.append(values.reshape(queries, -1))
    return (
        np.concatenate(aggregate_parts, axis=1).astype(np.float32),
        np.concatenate(position_parts, axis=1).astype(np.float32),
    )


def read_frame(run: Path) -> pd.DataFrame:
    summaries = json.loads((run / "client/summaries.json").read_text(encoding="utf-8"))
    summaries = sorted(summaries, key=lambda row: int(row["episode_index"]))
    frame = pd.DataFrame(
        {
            "episode": [int(row["episode_index"]) for row in summaries],
            "init_state_id": [int(row["init_state_id"]) for row in summaries],
            "flow_noise_seed": [int(row["flow_noise_seed"]) for row in summaries],
            "success": [bool(row["success"]) for row in summaries],
            "inference_calls": [int(row["inference_calls"]) for row in summaries],
        }
    )
    frame["failure"] = ~frame["success"]
    if len(frame) != 512 or frame["init_state_id"].nunique() != 16:
        raise ValueError("expected the Long 16x32 cohort")
    counts = frame.groupby("init_state_id")["flow_noise_seed"].nunique()
    if not counts.eq(32).all():
        raise ValueError("each initial state must have 32 distinct seeds")
    return previous.assign_folds(frame)


def extract_feature_cache(
    run: Path, frame: pd.DataFrame, cache_path: Path
) -> dict[str, np.ndarray]:
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    episode_rows = np.asarray(store["episode_id"][:], dtype=np.int64)
    control_rows = np.asarray(store["control_step"][:], dtype=np.int64)
    raw = np.asarray(store["hb_router_probs"][:])
    expected = int(frame["inference_calls"].sum())
    if len(raw) != expected or len(episode_rows) != expected:
        raise ValueError("route-row count differs from recorded inference calls")

    schema = schemas()
    aggregate_dim = len(schema["aggregate"][0])
    cell_dim = len(schema["cell"][0])
    aggregate_rows = np.empty((len(raw), aggregate_dim), dtype=np.float32)
    cell_rows = np.empty((len(raw), cell_dim), dtype=np.float32)
    block = 16
    for start in range(0, len(raw), block):
        stop = min(start + block, len(raw))
        aggregate_rows[start:stop], cell_rows[start:stop] = current_route_features(
            raw[start:stop]
        )
        if start % 1024 == 0:
            print(f"current route rows {stop}/{len(raw)}", flush=True)

    max_query = int(frame["inference_calls"].max())
    current_aggregate = np.full(
        (len(frame), max_query, aggregate_dim), np.nan, dtype=np.float32
    )
    current_cell = np.full(
        (len(frame), max_query, cell_dim), np.nan, dtype=np.float32
    )
    temporal_aggregate = np.full(
        (len(frame), max_query, len(schema["temporal_aggregate"][0])),
        np.nan,
        dtype=np.float32,
    )
    temporal_cell = np.full(
        (len(frame), max_query, len(schema["temporal"][0])),
        np.nan,
        dtype=np.float32,
    )
    for row, item in frame.iterrows():
        episode = int(item["episode"])
        indices = np.flatnonzero(episode_rows == episode)
        indices = indices[np.argsort(control_rows[indices], kind="stable")]
        queries = int(item["inference_calls"])
        if len(indices) != queries:
            raise ValueError(f"episode {episode}: expected {queries} route rows")
        current_aggregate[row, :queries] = aggregate_rows[indices]
        current_cell[row, :queries] = cell_rows[indices]
        temporal_aggregate[row, :queries], temporal_cell[row, :queries] = (
            temporal_route_features(raw[indices])
        )
        if (row + 1) % 32 == 0:
            print(f"temporal episodes {row + 1}/{len(frame)}", flush=True)

    arrays = {
        "current_aggregate": current_aggregate,
        "current_cell": current_cell,
        "temporal_aggregate": temporal_aggregate,
        "temporal_cell": temporal_cell,
    }
    np.savez_compressed(
        cache_path,
        **{key: value.astype(np.float16) for key, value in arrays.items()},
        episode=frame["episode"].to_numpy(dtype=np.int32),
        inference_calls=frame["inference_calls"].to_numpy(dtype=np.int16),
    )
    return arrays


def load_features(
    run: Path, frame: pd.DataFrame, cache_path: Path, rebuild: bool
) -> dict[str, np.ndarray]:
    if rebuild or not cache_path.is_file():
        return extract_feature_cache(run, frame, cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["episode"], frame["episode"].to_numpy(dtype=np.int32)
        ):
            raise ValueError("feature cache episode order mismatch")
        if not np.array_equal(
            archive["inference_calls"],
            frame["inference_calls"].to_numpy(dtype=np.int16),
        ):
            raise ValueError("feature cache length mismatch")
        return {
            key: np.asarray(archive[key], dtype=np.float32)
            for key in (
                "current_aggregate",
                "current_cell",
                "temporal_aggregate",
                "temporal_cell",
            )
        }


def feature_banks(
    features: dict[str, np.ndarray], query: int
) -> dict[str, tuple[np.ndarray, list[str], list[str], np.ndarray]]:
    schema = schemas()
    aggregate = features["current_aggregate"][:, query]
    aggregate_temporal = features["temporal_aggregate"][:, query]
    cell = features["current_cell"][:, query]
    temporal = features["temporal_cell"][:, query]
    aggregate_names, aggregate_families, aggregate_tokens = schema["aggregate"]
    temporal_aggregate_names, temporal_aggregate_families, temporal_aggregate_tokens = (
        schema["temporal_aggregate"]
    )
    cell_names, cell_families, cell_tokens = schema["cell"]
    temporal_names, temporal_families, temporal_tokens = schema["temporal"]
    return {
        "aggregate_current": (
            aggregate,
            [f"aggregate/{name}" for name in aggregate_names],
            aggregate_families,
            aggregate_tokens,
        ),
        "aggregate_history": (
            np.c_[aggregate, aggregate_temporal],
            [f"aggregate/{name}" for name in aggregate_names]
            + [f"aggregate/{name}" for name in temporal_aggregate_names],
            aggregate_families + temporal_aggregate_families,
            np.r_[aggregate_tokens, temporal_aggregate_tokens],
        ),
        "token_current": (
            np.c_[aggregate, cell],
            [f"aggregate/{name}" for name in aggregate_names]
            + [f"position/{name}" for name in cell_names],
            aggregate_families + cell_families,
            np.r_[aggregate_tokens, cell_tokens],
        ),
        "token_history": (
            np.c_[aggregate, aggregate_temporal, cell, temporal],
            [f"aggregate/{name}" for name in aggregate_names]
            + [f"aggregate/{name}" for name in temporal_aggregate_names]
            + [f"position/{name}" for name in cell_names]
            + [f"position/{name}" for name in temporal_names],
            aggregate_families
            + temporal_aggregate_families
            + cell_families
            + temporal_families,
            np.r_[
                aggregate_tokens,
                temporal_aggregate_tokens,
                cell_tokens,
                temporal_tokens,
            ],
        ),
        "token_only_history": (
            np.c_[cell, temporal],
            [f"position/{name}" for name in cell_names]
            + [f"position/{name}" for name in temporal_names],
            cell_families + temporal_families,
            np.r_[cell_tokens, temporal_tokens],
        ),
    }


def within_state_scores(
    matrix: np.ndarray, labels: np.ndarray, states: np.ndarray
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.float64)
    centered = np.empty_like(values)
    label_centered = np.empty_like(outcome)
    for state in np.unique(states):
        selected = states == state
        centered[selected] = values[selected] - values[selected].mean(
            axis=0, keepdims=True
        )
        label_centered[selected] = outcome[selected] - outcome[selected].mean()
    numerator = centered.T @ label_centered
    denominator = np.sqrt(
        np.sum(np.square(centered), axis=0)
        * np.sum(np.square(label_centered))
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def select_by_family(
    matrix: np.ndarray,
    labels: np.ndarray,
    states: np.ndarray,
    families: list[str],
    allowed: np.ndarray | None = None,
) -> list[int]:
    score = within_state_scores(matrix, labels, states)
    family_array = np.asarray(families)
    eligible = np.ones(matrix.shape[1], dtype=bool) if allowed is None else allowed
    selected = []
    for family in dict.fromkeys(families):
        indices = np.flatnonzero(eligible & (family_array == family))
        if len(indices):
            selected.append(int(indices[np.argmax(np.abs(score[indices]))]))
    return selected


def shift_indices(frame: pd.DataFrame, offset: int) -> np.ndarray:
    result = np.empty(len(frame), dtype=np.int64)
    for _, group in frame.groupby("init_state_id", sort=True):
        indices = group.sort_values("flow_noise_seed").index.to_numpy(dtype=np.int64)
        result[indices] = np.roll(indices, -offset)
    return result


def predict_selected(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    y_train: np.ndarray,
    train_states: np.ndarray,
    train_prior: np.ndarray,
    test_prior: np.ndarray,
    families: list[str],
    shifted_test: list[np.ndarray],
    *,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    selected = select_by_family(
        train_matrix, y_train, train_states, families, allowed=allowed
    )
    if not selected:
        return test_prior.copy(), [test_prior.copy() for _ in shifted_test], []
    columns = np.asarray(selected, dtype=np.int64)
    scaler = StandardScaler()
    train = scaler.fit_transform(train_matrix[:, columns])
    test_blocks = [test_matrix[:, columns]] + [item[:, columns] for item in shifted_test]
    sizes = [len(item) for item in test_blocks]
    stacked = scaler.transform(np.vstack(test_blocks))
    stacked_prior = np.tile(test_prior, len(test_blocks))
    scores = previous.robust_offset_scores(
        train,
        y_train,
        stacked,
        train_prior,
        stacked_prior,
        0.1,
    )
    split = np.cumsum(sizes)[:-1]
    blocks = np.split(scores, split)
    return blocks[0], blocks[1:], selected


def predict_best_cell(
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    labels: np.ndarray,
    states: np.ndarray,
    test_prior: np.ndarray,
    allowed: np.ndarray,
) -> tuple[np.ndarray, int | None]:
    """Orient one training-selected coordinate; used only for token heatmaps."""
    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        return test_prior.copy(), None
    effect = within_state_scores(train_matrix[:, candidates], labels, states)
    local = int(np.argmax(np.abs(effect)))
    selected = int(candidates[local])
    scaler = StandardScaler()
    train = scaler.fit_transform(train_matrix[:, selected : selected + 1])[:, 0]
    test = scaler.transform(test_matrix[:, selected : selected + 1])[:, 0]
    direction = float(np.sign(np.dot(train, labels - labels.mean())))
    if direction == 0.0:
        return test_prior.copy(), selected
    epsilon = 1e-6
    prior_logit = np.log(
        np.clip(test_prior, epsilon, 1.0 - epsilon)
        / np.clip(1.0 - test_prior, epsilon, 1.0 - epsilon)
    )
    score = 1.0 / (1.0 + np.exp(-(prior_logit + direction * test)))
    return score, selected


def append_prediction_rows(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    split: str,
    fold: int,
    query: int,
    model: str,
    variant: str,
) -> None:
    for position, row_index in enumerate(indices):
        item = frame.iloc[row_index]
        rows.append(
            {
                "split": split,
                "fold": fold,
                "query": query,
                "cohort": "common" if query <= COMMON_QUERY_MAX else "risk_set",
                "model": model,
                "variant": variant,
                "episode": int(item["episode"]),
                "init_state_id": int(item["init_state_id"]),
                "flow_noise_seed": int(item["flow_noise_seed"]),
                "failure": bool(item["failure"]),
                "score_failure": float(scores[position]),
            }
        )


def run_oof(
    frame: pd.DataFrame,
    features: dict[str, np.ndarray],
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_frames = []
    selection_frames = []
    for query in range(RISK_QUERY_MAX + 1):
        prediction_path = checkpoint_dir / f"predictions_q{query:02d}.csv.gz"
        selection_path = checkpoint_dir / f"selections_q{query:02d}.csv.gz"
        if prediction_path.is_file() and selection_path.is_file():
            prediction_frames.append(pd.read_csv(prediction_path))
            selection_frames.append(pd.read_csv(selection_path))
            print(f"OOF query {query}/{RISK_QUERY_MAX} cached", flush=True)
            continue
        prediction_rows: list[dict[str, Any]] = []
        selection_rows: list[dict[str, Any]] = []
        alive = frame["inference_calls"].gt(query).to_numpy()
        banks = feature_banks(features, query)
        for split in ("heldout_seed", "heldout_init"):
            fold_column = f"{split}_fold"
            include_state = split == "heldout_seed"
            for fold in range(8):
                test_index = np.flatnonzero(
                    alive & frame[fold_column].eq(fold).to_numpy()
                )
                train_index = np.flatnonzero(
                    alive & ~frame[fold_column].eq(fold).to_numpy()
                )
                if not len(test_index) or not len(train_index):
                    continue
                y_train = frame.loc[train_index, "failure"].to_numpy(dtype=bool)
                train_states = frame.loc[
                    train_index, "init_state_id"
                ].to_numpy(dtype=int)
                test_states = frame.loc[test_index, "init_state_id"].to_numpy(
                    dtype=int
                )
                train_prior, test_prior = initial.prior_scores(
                    y_train, train_states, test_states, include_state
                )
                append_prediction_rows(
                    prediction_rows,
                    frame,
                    test_index,
                    test_prior,
                    split=split,
                    fold=fold,
                    query=query,
                    model="state_prior",
                    variant="aligned",
                )
                test_frame = frame.iloc[test_index].reset_index(drop=True)
                shifted_indices = [
                    shift_indices(test_frame, offset) for offset in SHIFT_OFFSETS
                ]
                for model in (
                    "aggregate_current",
                    "aggregate_history",
                    "token_current",
                    "token_history",
                ):
                    matrix, names, families, _ = banks[model]
                    shifted = [
                        matrix[test_index][indices] for indices in shifted_indices
                    ]
                    aligned_score, shifted_scores, selected = predict_selected(
                        matrix[train_index],
                        matrix[test_index],
                        y_train,
                        train_states,
                        train_prior,
                        test_prior,
                        families,
                        shifted,
                    )
                    for rank, feature_index in enumerate(selected, start=1):
                        selection_rows.append(
                            {
                                "split": split,
                                "fold": fold,
                                "query": query,
                                "model": model,
                                "rank": rank,
                                "family": families[feature_index],
                                "feature": names[feature_index],
                            }
                        )
                    append_prediction_rows(
                        prediction_rows,
                        frame,
                        test_index,
                        aligned_score,
                        split=split,
                        fold=fold,
                        query=query,
                        model=model,
                        variant="aligned",
                    )
                    for offset, score in zip(
                        SHIFT_OFFSETS, shifted_scores, strict=True
                    ):
                        append_prediction_rows(
                            prediction_rows,
                            frame,
                            test_index,
                            score,
                            split=split,
                            fold=fold,
                            query=query,
                            model=model,
                            variant=f"shift_{offset}",
                        )

                if split != "heldout_seed":
                    continue
                matrix, names, families, tokens = banks["token_only_history"]
                for token in range(1, 11):
                    allowed = tokens == token
                    score, selected = predict_best_cell(
                        matrix[train_index],
                        matrix[test_index],
                        y_train,
                        train_states,
                        test_prior,
                        allowed,
                    )
                    if selected is not None:
                        selection_rows.append(
                            {
                                "split": split,
                                "fold": fold,
                                "query": query,
                                "model": f"token_t{token}_history",
                                "rank": 1,
                                "family": families[selected],
                                "feature": names[selected],
                            }
                        )
                    append_prediction_rows(
                        prediction_rows,
                        frame,
                        test_index,
                        score,
                        split=split,
                        fold=fold,
                        query=query,
                        model=f"token_t{token}_history",
                        variant="aligned",
                    )
        prediction_frame = pd.DataFrame(prediction_rows)
        selection_frame = pd.DataFrame(selection_rows)
        prediction_frame.to_csv(prediction_path, index=False)
        selection_frame.to_csv(selection_path, index=False)
        prediction_frames.append(prediction_frame)
        selection_frames.append(selection_frame)
        print(f"OOF query {query}/{RISK_QUERY_MAX}", flush=True)
    selections = pd.concat(selection_frames, ignore_index=True)
    selections["feature"] = (
        selections["feature"]
        .str.replace(
            "aggregate/top1_centered|", "aggregate/top1|", regex=False
        )
        .str.replace(
            "aggregate/entropy_centered|", "aggregate/entropy|", regex=False
        )
    )
    return pd.concat(prediction_frames, ignore_index=True), selections


def state_auc_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "query", "cohort", "model", "variant"]
    for key, group in predictions.groupby(keys, sort=False):
        for (state, fold), siblings in group.groupby(
            ["init_state_id", "fold"], sort=True
        ):
            labels = siblings["failure"].to_numpy(dtype=bool)
            if len(np.unique(labels)) < 2:
                continue
            score = siblings["score_failure"].to_numpy(dtype=float)
            rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "init_state_id": int(state),
                    "fold": int(fold),
                    "episodes": len(siblings),
                    "failures": int(labels.sum()),
                    "successes": int((~labels).sum()),
                    "pairs": int(labels.sum() * (~labels).sum()),
                    "auc": float(roc_auc_score(labels, score)),
                }
            )
    return pd.DataFrame(rows)


def collapse_state_aucs(state_aucs: pd.DataFrame) -> pd.DataFrame:
    """Combine estimable test folds within state, weighting actual S/F pairs."""
    keys = ["split", "query", "cohort", "model", "variant", "init_state_id"]
    rows = []
    for key, group in state_aucs.groupby(keys, sort=False):
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "episodes": int(group["episodes"].sum()),
                "failures": int(group["failures"].sum()),
                "successes": int(group["successes"].sum()),
                "pairs": int(group["pairs"].sum()),
                "estimable_folds": len(group),
                "auc": float(np.average(group["auc"], weights=group["pairs"])),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(
    values: np.ndarray, draws: int, seed: int
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(draws, len(array)))
    distribution = array[samples].mean(axis=1)
    low, high = np.quantile(distribution, (0.025, 0.975))
    return float(array.mean()), float(low), float(high)


def auc_metrics(
    predictions: pd.DataFrame,
    state_aucs: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    collapsed = collapse_state_aucs(state_aucs)
    rows = []
    keys = ["split", "query", "cohort", "model", "variant"]
    prediction_lookup = {
        key: group for key, group in predictions.groupby(keys, sort=False)
    }
    for axis, (key, group) in enumerate(collapsed.groupby(keys, sort=False)):
        macro, low, high = bootstrap_mean(
            group["auc"].to_numpy(), draws, seed + axis
        )
        prediction = prediction_lookup[key]
        truth = prediction["failure"].to_numpy(dtype=bool)
        score = prediction["score_failure"].to_numpy(dtype=float)
        pooled = (
            float(roc_auc_score(truth, score))
            if len(np.unique(truth)) == 2
            else np.nan
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "episodes": len(prediction),
                "failures": int(truth.sum()),
                "successes": int((~truth).sum()),
                "mixed_states": len(group),
                "macro_within_state_auc": macro,
                "auc_ci95_low": low,
                "auc_ci95_high": high,
                "pair_weighted_within_state_auc": float(
                    np.average(group["auc"], weights=group["pairs"])
                ),
                "pooled_auc": pooled,
            }
        )
    result = pd.DataFrame(rows)

    shifted = state_aucs[state_aucs["variant"].str.startswith("shift_")]
    if not shifted.empty:
        shift_fold = (
            shifted.groupby(
                [
                    "split",
                    "query",
                    "cohort",
                    "model",
                    "init_state_id",
                    "fold",
                ],
                as_index=False,
            )
            .agg(
                episodes=("episodes", "first"),
                failures=("failures", "first"),
                successes=("successes", "first"),
                pairs=("pairs", "first"),
                auc=("auc", "mean"),
            )
        )
        shift_fold["variant"] = "shift_mean"
        shift_state = collapse_state_aucs(shift_fold)
        extra = []
        for axis, (key, group) in enumerate(
            shift_state.groupby(["split", "query", "cohort", "model"], sort=False)
        ):
            macro, low, high = bootstrap_mean(
                group["auc"].to_numpy(), draws, seed + 100000 + axis
            )
            extra.append(
                {
                    "split": key[0],
                    "query": key[1],
                    "cohort": key[2],
                    "model": key[3],
                    "variant": "shift_mean",
                    "episodes": int(group["episodes"].sum()),
                    "failures": int(group["failures"].sum()),
                    "successes": int(group["successes"].sum()),
                    "mixed_states": len(group),
                    "macro_within_state_auc": macro,
                    "auc_ci95_low": low,
                    "auc_ci95_high": high,
                    "pair_weighted_within_state_auc": float(
                        np.average(group["auc"], weights=group["pairs"])
                    ),
                    "pooled_auc": np.nan,
                }
            )
        result = pd.concat((result, pd.DataFrame(extra)), ignore_index=True)
    return result


def auc_deltas(
    state_aucs: pd.DataFrame, draws: int, seed: int
) -> pd.DataFrame:
    aligned = collapse_state_aucs(
        state_aucs[state_aucs["variant"].eq("aligned")]
    )
    shifted = state_aucs[state_aucs["variant"].str.startswith("shift_")]
    shifted_fold = (
        shifted.groupby(
            [
                "split",
                "query",
                "cohort",
                "model",
                "init_state_id",
                "fold",
            ],
            as_index=False,
        )
        .agg(
            episodes=("episodes", "first"),
            failures=("failures", "first"),
            successes=("successes", "first"),
            pairs=("pairs", "first"),
            auc=("auc", "mean"),
        )
    )
    shifted_fold["variant"] = "shift_mean"
    shifted = collapse_state_aucs(shifted_fold).rename(
        columns={"auc": "shift_auc"}
    )
    comparisons = {
        "token_position_increment_current": (
            "aggregate_current",
            "token_current",
        ),
        "token_position_increment_history": (
            "aggregate_history",
            "token_history",
        ),
        "temporal_increment_aggregate": (
            "aggregate_current",
            "aggregate_history",
        ),
        "temporal_increment_token": ("token_current", "token_history"),
        "state_to_token_history": ("state_prior", "token_history"),
    }
    rows = []
    axis = 0
    for split in aligned["split"].unique():
        for query in sorted(aligned["query"].unique()):
            block = aligned[
                aligned["split"].eq(split) & aligned["query"].eq(query)
            ]
            if block.empty:
                continue
            cohort = str(block["cohort"].iloc[0])
            pivot = block.pivot(index="init_state_id", columns="model", values="auc")
            for comparison, (left, right) in comparisons.items():
                if left not in pivot or right not in pivot:
                    continue
                values = (pivot[right] - pivot[left]).dropna().to_numpy()
                mean, low, high = bootstrap_mean(values, draws, seed + axis)
                axis += 1
                rows.append(
                    {
                        "split": split,
                        "query": query,
                        "cohort": cohort,
                        "comparison": comparison,
                        "states": len(values),
                        "auc_gain": mean,
                        "auc_gain_ci95_low": low,
                        "auc_gain_ci95_high": high,
                    }
                )
            token = block[block["model"].eq("token_history")][
                ["init_state_id", "auc"]
            ]
            shift = shifted[
                shifted["split"].eq(split)
                & shifted["query"].eq(query)
                & shifted["model"].eq("token_history")
            ][["init_state_id", "shift_auc"]]
            paired = token.merge(shift, on="init_state_id", validate="one_to_one")
            if not paired.empty:
                values = paired["auc"].to_numpy() - paired["shift_auc"].to_numpy()
                mean, low, high = bootstrap_mean(values, draws, seed + axis)
                axis += 1
                rows.append(
                    {
                        "split": split,
                        "query": query,
                        "cohort": cohort,
                        "comparison": "aligned_vs_shifted_token_history",
                        "states": len(values),
                        "auc_gain": mean,
                        "auc_gain_ci95_low": low,
                        "auc_gain_ci95_high": high,
                    }
                )
    return pd.DataFrame(rows)


def fixed_train_test_prediction(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    matrix: np.ndarray,
    names: list[str],
    families: list[str],
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    y_train = train_frame["failure"].to_numpy(dtype=bool)
    train_states = train_frame["init_state_id"].to_numpy(dtype=int)
    test_states = test_frame["init_state_id"].to_numpy(dtype=int)
    train_prior, test_prior = initial.prior_scores(
        y_train, train_states, test_states, True
    )
    shift_maps = [shift_indices(test_frame, offset) for offset in SHIFT_OFFSETS]
    shifted = [matrix[test_index][indices] for indices in shift_maps]
    aligned, shifted_scores, selected = predict_selected(
        matrix[train_index],
        matrix[test_index],
        y_train,
        train_states,
        train_prior,
        test_prior,
        families,
        shifted,
    )
    rows = []
    score_blocks = [("aligned", aligned)] + [
        (f"shift_{offset}", score)
        for offset, score in zip(SHIFT_OFFSETS, shifted_scores, strict=True)
    ]
    for variant, score in score_blocks:
        for position, (_, item) in enumerate(test_frame.iterrows()):
            rows.append(
                {
                    "variant": variant,
                    "init_state_id": int(item["init_state_id"]),
                    "flow_noise_seed": int(item["flow_noise_seed"]),
                    "failure": bool(item["failure"]),
                    "score_failure": float(score[position]),
                }
            )
    return pd.DataFrame(rows), [names[index] for index in selected]


def discovery_validation(
    frame: pd.DataFrame,
    features: dict[str, np.ndarray],
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    discovery_mask = frame["flow_noise_seed"].lt(1016).to_numpy()
    validation_mask = ~discovery_mask
    discovery = frame.loc[discovery_mask].reset_index(drop=True)
    discovery["mini_fold"] = (
        (discovery["flow_noise_seed"] - 1000) // 4
    ).astype(int)
    curve_rows = []
    for query in range(COMMON_QUERY_MAX + 1):
        matrix, names, families, _ = feature_banks(features, query)["token_history"]
        prediction = np.full(len(discovery), np.nan, dtype=np.float64)
        discovery_full_index = np.flatnonzero(discovery_mask)
        for fold in range(4):
            test_local = np.flatnonzero(discovery["mini_fold"].eq(fold).to_numpy())
            train_local = np.flatnonzero(~discovery["mini_fold"].eq(fold).to_numpy())
            train_full = discovery_full_index[train_local]
            test_full = discovery_full_index[test_local]
            y_train = discovery.loc[train_local, "failure"].to_numpy(dtype=bool)
            train_states = discovery.loc[
                train_local, "init_state_id"
            ].to_numpy(dtype=int)
            test_states = discovery.loc[test_local, "init_state_id"].to_numpy(
                dtype=int
            )
            train_prior, test_prior = initial.prior_scores(
                y_train, train_states, test_states, True
            )
            score, _, _ = predict_selected(
                matrix[train_full],
                matrix[test_full],
                y_train,
                train_states,
                train_prior,
                test_prior,
                families,
                [],
            )
            prediction[test_local] = score
        state_cells = []
        for (state, fold), group in discovery.assign(score=prediction).groupby(
            ["init_state_id", "mini_fold"], sort=True
        ):
            if group["failure"].nunique() == 2:
                failures = int(group["failure"].sum())
                successes = int((~group["failure"]).sum())
                state_cells.append(
                    {
                        "state": int(state),
                        "fold": int(fold),
                        "pairs": failures * successes,
                        "auc": roc_auc_score(group["failure"], group["score"]),
                    }
                )
        cell_frame = pd.DataFrame(state_cells)
        state_values = []
        if not cell_frame.empty:
            for _, group in cell_frame.groupby("state"):
                state_values.append(np.average(group["auc"], weights=group["pairs"]))
        curve_rows.append(
            {
                "query": query,
                "discovery_oof_macro_within_state_auc": float(np.mean(state_values)),
                "mixed_states": len(state_values),
            }
        )
    curve = pd.DataFrame(curve_rows)
    chosen_query = int(
        curve.loc[curve["discovery_oof_macro_within_state_auc"].idxmax(), "query"]
    )
    matrix, names, families, _ = feature_banks(features, chosen_query)["token_history"]
    train_index = np.flatnonzero(discovery_mask)
    test_index = np.flatnonzero(validation_mask)
    validation = frame.loc[validation_mask].reset_index(drop=True)
    predictions, selected = fixed_train_test_prediction(
        discovery,
        validation,
        matrix,
        names,
        families,
        train_index,
        test_index,
    )
    state_rows = []
    for (variant, state), group in predictions.groupby(
        ["variant", "init_state_id"], sort=True
    ):
        if group["failure"].nunique() != 2:
            continue
        state_rows.append(
            {
                "variant": variant,
                "init_state_id": int(state),
                "auc": float(roc_auc_score(group["failure"], group["score_failure"])),
            }
        )
    state_frame = pd.DataFrame(state_rows)
    metrics = []
    for axis, (variant, group) in enumerate(state_frame.groupby("variant")):
        mean, low, high = bootstrap_mean(
            group["auc"].to_numpy(), draws, seed + 800000 + axis
        )
        metrics.append(
            {
                "variant": variant,
                "macro_within_state_auc": mean,
                "auc_ci95_low": low,
                "auc_ci95_high": high,
                "mixed_states": len(group),
            }
        )
    metric_frame = pd.DataFrame(metrics)
    shifted_mean = float(
        metric_frame[metric_frame["variant"].str.startswith("shift_")][
            "macro_within_state_auc"
        ].mean()
    )
    aligned = metric_frame[metric_frame["variant"].eq("aligned")].iloc[0]
    summary = {
        "discovery_seeds": [1000, 1015],
        "validation_seeds": [1016, 1031],
        "chosen_query": chosen_query,
        "discovery_oof_auc": float(
            curve.loc[
                curve["query"].eq(chosen_query),
                "discovery_oof_macro_within_state_auc",
            ].iloc[0]
        ),
        "validation_auc": float(aligned["macro_within_state_auc"]),
        "validation_auc_ci95": [
            float(aligned["auc_ci95_low"]),
            float(aligned["auc_ci95_high"]),
        ],
        "validation_shifted_mean_auc": shifted_mean,
        "selected_features": selected,
    }
    curve["chosen"] = curve["query"].eq(chosen_query)
    return curve, {"summary": summary, "metrics": metric_frame, "predictions": predictions}


def frozen_feature_effects(
    frame: pd.DataFrame,
    features: dict[str, np.ndarray],
    discovery: dict[str, Any],
    draws: int,
    seed: int,
) -> pd.DataFrame:
    summary = discovery["summary"]
    query = int(summary["chosen_query"])
    matrix, names, _, _ = feature_banks(features, query)["token_history"]
    indices = np.asarray(
        [names.index(name) for name in summary["selected_features"]], dtype=np.int64
    )
    validation = frame["flow_noise_seed"].ge(1016).to_numpy()
    selected_frame = frame.loc[validation].reset_index(drop=True)
    values = matrix[validation][:, indices]
    mixed = selected_frame.groupby("init_state_id")["failure"].nunique()
    keep = selected_frame["init_state_id"].isin(mixed.index[mixed.eq(2)]).to_numpy()
    selected_frame = selected_frame.loc[keep].reset_index(drop=True)
    values = values[keep]
    labels = selected_frame["failure"].to_numpy(dtype=bool)
    groups, _ = pd.factorize(selected_frame["init_state_id"], sort=True)
    adjusted, residual_sd, effect, _ = route_effects.fixed_effect_shift(
        values, labels, groups
    )
    bootstrap = route_effects.bootstrap_top_effects(
        values,
        labels,
        groups,
        np.arange(values.shape[1]),
        draws,
        seed,
    )
    low, high = np.quantile(bootstrap, (0.025, 0.975), axis=0)
    return pd.DataFrame(
        {
            "query": query,
            "feature": summary["selected_features"],
            "validation_episodes": len(selected_frame),
            "mixed_states": len(np.unique(groups)),
            "failure_mean": values[labels].mean(axis=0),
            "success_mean": values[~labels].mean(axis=0),
            "adjusted_failure_minus_success": adjusted,
            "residual_sd": residual_sd,
            "effect_sigma": effect,
            "effect_ci95_low": low,
            "effect_ci95_high": high,
        }
    )


def plot_auc_curves(metrics: pd.DataFrame, out: Path) -> None:
    selected = metrics[
        metrics["split"].eq("heldout_seed")
        & metrics["variant"].eq("aligned")
        & metrics["model"].isin(
            (
                "state_prior",
                "aggregate_current",
                "aggregate_history",
                "token_current",
                "token_history",
            )
        )
    ]
    style = {
        "state_prior": ("Initial-state prior", "#777777", "--"),
        "aggregate_current": ("Token mean, current", "#2f6b8a", "-"),
        "aggregate_history": ("Token mean + history", "#69a3bd", "-"),
        "token_current": ("Per-token, current", "#b5473c", "-"),
        "token_history": ("Per-token + history", "#d58b32", "-"),
    }
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    for model, group in selected.groupby("model", sort=False):
        group = group.sort_values("query")
        label, color, linestyle = style[model]
        x = group["query"].to_numpy()
        y = group["macro_within_state_auc"].to_numpy()
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=2)
        if model == "token_history":
            ax.fill_between(
                x,
                group["auc_ci95_low"],
                group["auc_ci95_high"],
                color=color,
                alpha=0.16,
            )
    shifted = metrics[
        metrics["split"].eq("heldout_seed")
        & metrics["variant"].eq("shift_mean")
        & metrics["model"].eq("token_history")
    ].sort_values("query")
    ax.plot(
        shifted["query"],
        shifted["macro_within_state_auc"],
        color="#7d5a91",
        linestyle=":",
        linewidth=2,
        label="Per-token + history, shifted sibling",
    )
    ax.axhline(0.5, color="#222222", linewidth=1, alpha=0.65)
    ax.axvline(COMMON_QUERY_MAX + 0.5, color="#222222", linestyle="--", linewidth=1)
    ax.text(
        COMMON_QUERY_MAX + 0.8,
        0.985,
        "risk set",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
    )
    ax.set(xlabel="Policy query / action chunk", ylabel="Macro within-state ROC AUC")
    ax.set_ylim(0.25, 0.9)
    ax.grid(alpha=0.18)
    ax.legend(ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_token_heatmap(metrics: pd.DataFrame, out: Path) -> None:
    selected = metrics[
        metrics["split"].eq("heldout_seed")
        & metrics["variant"].eq("aligned")
        & metrics["model"].str.match(r"token_t\d+_history")
    ].copy()
    selected["token"] = selected["model"].str.extract(r"token_t(\d+)")[0].astype(int)
    table = selected.pivot(index="token", columns="query", values="macro_within_state_auc")
    table = table.reindex(index=range(1, 11), columns=range(RISK_QUERY_MAX + 1))
    fig, ax = plt.subplots(figsize=(13.2, 4.6))
    image = ax.imshow(
        table.to_numpy(),
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=0.30,
        vmax=0.85,
    )
    ax.axvline(COMMON_QUERY_MAX + 0.5, color="black", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Policy query / action chunk",
        ylabel="Action-token position",
        yticks=np.arange(10),
        yticklabels=np.arange(1, 11),
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.015)
    colorbar.set_label("Nested out-of-fold within-state AUC")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{digits}f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def write_report(
    out: Path,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    discovery: dict[str, Any],
    validation_effects: pd.DataFrame,
    pooled_clock_auc: float,
) -> None:
    primary = metrics[
        metrics["split"].eq("heldout_seed")
        & metrics["variant"].eq("aligned")
        & metrics["model"].eq("token_history")
    ]
    common = primary[primary["query"].le(COMMON_QUERY_MAX)]
    best = common.loc[common["macro_within_state_auc"].idxmax()]
    checkpoints = [0, 4, 8, 12, 16, 20, 24, 28, 32, 34, 36, 38, 40, 42, 44, 46]
    table = metrics[
        metrics["split"].eq("heldout_seed")
        & metrics["variant"].eq("aligned")
        & metrics["query"].isin(checkpoints)
        & metrics["model"].isin(
            ("aggregate_history", "token_current", "token_history")
        )
    ][
        [
            "query",
            "cohort",
            "model",
            "episodes",
            "mixed_states",
            "macro_within_state_auc",
            "auc_ci95_low",
            "auc_ci95_high",
            "pooled_auc",
        ]
    ].copy()
    table.columns = [
        "q",
        "cohort",
        "model",
        "n",
        "mixed states",
        "within-state AUC",
        "CI low",
        "CI high",
        "pooled AUC",
    ]
    frozen = discovery["summary"]
    delta_index = deltas.set_index(["split", "query", "comparison"])
    q24_position = delta_index.loc[
        ("heldout_seed", 24, "token_position_increment_history")
    ]
    q30_temporal = delta_index.loc[
        ("heldout_seed", 30, "temporal_increment_token")
    ]
    q34_position = delta_index.loc[
        ("heldout_seed", 34, "token_position_increment_history")
    ]
    validation_table = validation_effects[
        ["feature", "effect_sigma", "effect_ci95_low", "effect_ci95_high"]
    ].copy()
    validation_table.columns = ["frozen feature", "effect SD", "CI low", "CI high"]
    lines = [
        "# 完整 rollout 的逐 chunk MoE 成败 AUC",
        "",
        "## 结论",
        "",
        (
            f"逐 chunk 分析覆盖 q0-q{RISK_QUERY_MAX}。q0-q{COMMON_QUERY_MAX} 的 512 条分支"
            "全部存活，因此这里的 AUC 不可能靠 episode 长度或 remaining-time 得到。"
        ),
        (
            f"完整样本上，保留 token 位置并加入跨 chunk 变化的探索性最高点是 q{int(best['query'])}："
            f"同初态折外 AUC={best['macro_within_state_auc']:.3f} "
            f"[{best['auc_ci95_low']:.3f},{best['auc_ci95_high']:.3f}]。"
        ),
        (
            f"更严格的 seed 拆半先在 1000-1015 中选择 q{frozen['chosen_query']}，再冻结到 "
            f"1016-1031：验证 AUC={frozen['validation_auc']:.3f} "
            f"[{frozen['validation_auc_ci95'][0]:.3f},{frozen['validation_auc_ci95'][1]:.3f}]；"
            f"错配 sibling 平均 AUC={frozen['validation_shifted_mean_auc']:.3f}。"
        ),
        (
            f"位置解析不是始终有用：q24 保留 token 位置相对 token 均值增加 AUC "
            f"{q24_position['auc_gain']:+.3f} "
            f"[{q24_position['auc_gain_ci95_low']:+.3f},{q24_position['auc_gain_ci95_high']:+.3f}]；"
            f"q34 的增量则为 {q34_position['auc_gain']:+.3f} "
            f"[{q34_position['auc_gain_ci95_low']:+.3f},{q34_position['auc_gain_ci95_high']:+.3f}]。"
        ),
        "",
        "AUC 的主口径是在每个初态内部计算后再平均：0.5 表示同一初态的成功/失败 seed 无法排序。"
        " pooled AUC 另列，只用于观察初态难度；不能拿它代替 branch 级成败信息。",
        "",
        "## 数据与防泄漏口径",
        "",
        f"- 512 条 rollout：{int(frame['success'].sum())} 成功、{int(frame['failure'].sum())} 失败；16 个初态，每个 32 seeds。",
        f"- 最短轨迹有 {int(frame['inference_calls'].min())} 个 query，所以 q0-q{COMMON_QUERY_MAX} 是完整共同前缀。",
        f"- q{COMMON_QUERY_MAX + 1}-q{RISK_QUERY_MAX} 只评价当时仍在运行的分支，明确标成 risk set。",
        "- 每个 q 单独训练和评价；模型输入没有 q 编号、轨迹长度、剩余时间、terminal 标记、动作、hidden state 或 simulator state。",
        "- heldout-seed 为 8 折，每折完整留出 4 个 noise seeds；特征坐标只在训练折内选择。",
        "- shifted sibling 把测试分支的路由换成同初态另一 seed 的路由，但保留原标签。",
        (
            f"- 若错误地把 q0-q{RISK_QUERY_MAX} 风险集混在一起，单用 query 编号就有 AUC={pooled_clock_auc:.3f}；"
            "本报告没有使用这个泄漏口径。"
        ),
        "",
        "## 逐 chunk AUC",
        "",
        markdown_table(table),
        "",
        "模型含义：`aggregate_history` 把 10 个 action token 平均后使用当前路由与跨 query 变化；"
        "`token_current` 保留当前 chunk 的 token 位置；`token_history` 同时保留 token 位置和跨 chunk 路由变化。",
        (
            f"在 q30，给逐 token 当前结构加入历史变化的 AUC 增量为 "
            f"{q30_temporal['auc_gain']:+.3f} "
            f"[{q30_temporal['auc_gain_ci95_low']:+.3f},{q30_temporal['auc_gain_ci95_high']:+.3f}]。"
        ),
        "",
        "## 严格拆半验证",
        "",
        f"发现半部的折外曲线选择 q{frozen['chosen_query']}，发现 AUC={frozen['discovery_oof_auc']:.3f}。",
        f"冻结特征族每族只选一个训练内坐标，共 {len(frozen['selected_features'])} 个。"
        "下表效应完全来自未参与选择的后 16 seeds；正值表示失败更高：",
        "",
        markdown_table(validation_table),
        "",
        "组合方向是：失败分支在前层 d8/d9 的 chunk 内相邻-token Hellinger 和 top-4 turnover 更高，"
        "但后层 d9 的 t1 跨-query Hellinger 与非相邻 return Hellinger 更低。"
        "说人话就是 chunk 内更抖，跨 chunk 却更容易重复旧路由状态。",
        "",
        "## 解释边界",
        "",
        "这里预测的是最终 success/failure，而不是停滞标签。q 越晚，MoE 可以反映已经发生的状态分叉、循环或停滞，"
        "所以晚期 AUC 不是纯初态难度；但它仍是观察性相关，不等于 MoE 已经因果识别出失败原因。",
        f"q{COMMON_QUERY_MAX + 1} 以后成功轨迹陆续退出，风险集越来越小，必须结合 n、mixed states 和置信区间读，不能只看最高 AUC。",
        "",
        "完整数值见 `auc_metrics.csv`、`auc_deltas.csv`、`state_aucs.csv.gz`、"
        "`oof_predictions.csv.gz`、`feature_selections.csv.gz` 和 `discovery_validation.json`。",
        "",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    frame = read_frame(run)
    cache_path = out / "route_feature_cache.npz"
    features = load_features(run, frame, cache_path, args.rebuild_cache)
    predictions, selections = run_oof(frame, features, out / "_oof_checkpoints")
    state_aucs = state_auc_rows(predictions)
    metrics = auc_metrics(
        predictions, state_aucs, args.bootstraps, args.seed
    )
    deltas = auc_deltas(state_aucs, args.bootstraps, args.seed + 200000)
    discovery_curve, discovery = discovery_validation(
        frame, features, args.bootstraps, args.seed
    )
    validation_effects = frozen_feature_effects(
        frame, features, discovery, args.bootstraps, args.seed + 900000
    )

    pooled_rows = []
    for query in range(RISK_QUERY_MAX + 1):
        selected = frame[frame["inference_calls"].gt(query)]
        pooled_rows.append(
            pd.DataFrame(
                {
                    "query": query,
                    "failure": selected["failure"].to_numpy(dtype=bool),
                }
            )
        )
    pooled = pd.concat(pooled_rows, ignore_index=True)
    pooled_clock_auc = float(roc_auc_score(pooled["failure"], pooled["query"]))

    predictions.to_csv(out / "oof_predictions.csv.gz", index=False)
    selections.to_csv(out / "feature_selections.csv.gz", index=False)
    state_aucs.to_csv(out / "state_aucs.csv.gz", index=False)
    metrics.to_csv(out / "auc_metrics.csv", index=False)
    deltas.to_csv(out / "auc_deltas.csv", index=False)
    discovery_curve.to_csv(out / "discovery_curve.csv", index=False)
    discovery["metrics"].to_csv(out / "frozen_validation_metrics.csv", index=False)
    discovery["predictions"].to_csv(
        out / "frozen_validation_predictions.csv", index=False
    )
    validation_effects.to_csv(out / "frozen_validation_feature_effects.csv", index=False)
    write_json(out / "discovery_validation.json", discovery["summary"])
    write_json(
        out / "audit.json",
        {
            "schema": "himoe.full_chunk_outcome_auc.v1",
            "run": str(run),
            "episodes": len(frame),
            "successes": int(frame["success"].sum()),
            "failures": int(frame["failure"].sum()),
            "initial_states": int(frame["init_state_id"].nunique()),
            "seeds": int(frame["flow_noise_seed"].nunique()),
            "common_query_max": COMMON_QUERY_MAX,
            "risk_query_max": RISK_QUERY_MAX,
            "pooled_query_clock_auc_forbidden_control": pooled_clock_auc,
            "model_inputs": [
                "MoE full router probabilities at the fixed current query",
                "MoE route changes to already observed earlier queries",
            ],
            "forbidden_inputs_used": False,
            "forbidden_inputs": [
                "episode length",
                "remaining time",
                "terminal window",
                "query index",
                "action",
                "hidden state",
                "simulator state",
            ],
        },
    )
    plot_auc_curves(metrics, out / "auc_over_chunks.png")
    plot_token_heatmap(metrics, out / "token_position_auc_heatmap.png")
    write_report(
        out,
        frame,
        metrics,
        deltas,
        discovery,
        validation_effects,
        pooled_clock_auc,
    )

    artifacts = [
        path
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
