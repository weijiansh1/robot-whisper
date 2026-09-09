#!/usr/bin/env python3
"""Test whether MoE routing predicts a physical trap before its first onset.

The analysis uses the rolling-star K=16 capture.  Loop and prolonged-static
onsets are defined only from physical trajectories.  At fixed policy-query
landmarks, models predict whether a previously untrapped branch will first
enter a trap in one of three future query bands.  Evaluation is always within
the same snapshot and landmark, so initial state, trunk phase, and elapsed
query count cannot by themselves rank a positive sibling above a negative.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_full_chunk_outcome_auc as route_auc
import analyze_rolling_star_experiment as rolling
from analyze_trap_onset_lead import prefix_control_features
from analyze_trap_onset_sweep import loop_onset_query, per_query_arrays


HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/rolling-star-a100-long-t08-k16-20260828"
DEFAULT_OUT_NAME = "analysis_trap_event_moe_20260829"
LANDMARKS = tuple(range(8, 33, 4))
LEAD_BANDS = {
    "next_1_4": (1, 4),
    "next_5_8": (5, 8),
    "next_9_12": (9, 12),
}
TARGETS = ("trap", "loop", "static")
STATIC_EVENT_ACTIONS = 80
SEED = 20260829
SHIFT_OFFSETS = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path)
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


def static_onset_query(arrays: dict[str, np.ndarray], targets: list[dict]) -> int:
    """First query by which 80 continuous low-motion action steps are observed."""
    eef = np.asarray(arrays["control_eef_position"], dtype=np.float32)
    sim = np.asarray(arrays["control_sim_state"], dtype=np.float32)
    gripper = np.asarray(arrays["control_gripper_qpos"], dtype=np.float32).mean(
        axis=1
    )
    objects = np.stack(
        [
            sim[:, int(target["state_lo"]) : int(target["state_lo"]) + 3]
            for target in targets
        ],
        axis=1,
    )
    eef_step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    object_step = np.linalg.norm(np.diff(objects, axis=0), axis=2).max(axis=1)
    grip_step = np.abs(np.diff(gripper))
    if len(eef_step) < rolling.STATIC_WINDOW:
        return -1
    kernel = np.ones(rolling.STATIC_WINDOW, dtype=np.float64)
    static = (
        (np.convolve(eef_step, kernel, mode="valid") <= rolling.STATIC_EEF_PATH_M)
        & (
            np.convolve(object_step, kernel, mode="valid")
            <= rolling.STATIC_OBJECT_PATH_M
        )
        & (
            np.convolve(grip_step, kernel, mode="valid")
            <= rolling.STATIC_GRIPPER_PATH_M
        )
    )
    required_windows = STATIC_EVENT_ACTIONS - rolling.STATIC_WINDOW + 1
    run = 0
    for window_start, value in enumerate(static):
        run = run + 1 if value else 0
        if run < required_windows:
            continue
        last_action = window_start + rolling.STATIC_WINDOW - 1
        query = np.asarray(arrays["control_query_index"], dtype=np.int64)
        return int(query[min(last_action, len(query) - 1)])
    return -1


def episode_table(
    candidates: list[rolling.Candidate],
    trajectories: dict[int, dict[str, np.ndarray]],
    targets: list[dict],
    references: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[int, dict[str, np.ndarray]]]:
    rows = []
    series_by_episode = {}
    for candidate in candidates:
        arrays = trajectories[candidate.episode_id]
        series = per_query_arrays(arrays, targets, references)
        series_by_episode[candidate.episode_id] = series
        loop, partner = loop_onset_query(
            series["eef"], series["objects"], series["gripper"], series["goal"]
        )
        static = static_onset_query(arrays, targets)
        onset_candidates = [value for value in (loop, static) if value >= 0]
        trap = min(onset_candidates, default=-1)
        improvement = np.nan
        recovered = False
        if trap >= 0:
            future = series["goal"][trap + 1 :]
            improvement = (
                float(series["goal"][trap] - future.min()) if len(future) else 0.0
            )
            recovered = bool(improvement >= 0.035)
        rows.append(
            {
                "worker": candidate.worker,
                "init_state_id": candidate.init_state,
                "snapshot": candidate.snapshot,
                "snapshot_key": candidate.snapshot_key,
                "candidate": candidate.candidate,
                "episode_id": candidate.episode_id,
                "success": candidate.success,
                "n_query": len(series["goal"]),
                "loop_onset": loop,
                "loop_partner": partner,
                "static_onset": static,
                "trap_onset": trap,
                "post_trap_goal_improvement_m": improvement,
                "post_trap_recovered_035m": recovered,
            }
        )
    return pd.DataFrame(rows), series_by_episode


def action_features(action_chunks: np.ndarray, query: int) -> np.ndarray:
    current = np.asarray(action_chunks[query], dtype=np.float32)
    previous = current if query == 0 else np.asarray(action_chunks[query - 1])
    token_norm = np.linalg.norm(current, axis=1)
    adjacent = np.linalg.norm(np.diff(current, axis=0), axis=1)
    delta = current - previous
    return np.r_[
        current.mean(axis=0),
        current.std(axis=0),
        np.abs(current).mean(axis=0),
        delta.mean(axis=0),
        np.abs(delta).mean(axis=0),
        [
            token_norm.mean(),
            token_norm.std(),
            adjacent.mean(),
            adjacent.std(),
            np.linalg.norm(current[-1] - current[0]),
        ],
    ].astype(np.float32)


def control_names() -> list[str]:
    physical = [
        "eef_path",
        "eef_net",
        "eef_efficiency",
        "eef_step_mean",
        "eef_step_std",
        "eef_recent_mean",
        "eef_recent_std",
        "eef_last_step",
        "goal_current",
        "goal_progress",
        "goal_delta_mean",
        "goal_recent_delta",
        "object_path",
        "object_recent_step",
        "gripper_current",
        "gripper_std",
        "gripper_total_change",
    ]
    action = []
    for summary in ("mean", "std", "abs_mean", "delta_mean", "delta_abs_mean"):
        action.extend([f"action_{summary}_{axis}" for axis in range(7)])
    action.extend(
        [
            "action_token_norm_mean",
            "action_token_norm_std",
            "action_adjacent_change_mean",
            "action_adjacent_change_std",
            "action_first_last_change",
        ]
    )
    return ["query_fraction"] + physical + action


def build_landmark_table(
    episodes: pd.DataFrame,
    trajectories: dict[int, dict[str, np.ndarray]],
    series_by_episode: dict[int, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    controls = []
    for item in episodes.itertuples(index=False):
        arrays = trajectories[item.episode_id]
        series = series_by_episode[item.episode_id]
        for query in LANDMARKS:
            if item.n_query <= query:
                continue
            rows.append(
                {
                    "worker": item.worker,
                    "snapshot": item.snapshot,
                    "snapshot_key": item.snapshot_key,
                    "candidate": item.candidate,
                    "episode_id": item.episode_id,
                    "success": item.success,
                    "n_query": item.n_query,
                    "query": query,
                    "stratum": f"{item.snapshot_key}|q{query:02d}",
                    "loop_onset": item.loop_onset,
                    "static_onset": item.static_onset,
                    "trap_onset": item.trap_onset,
                }
            )
            controls.append(
                np.r_[
                    query / 52.0,
                    prefix_control_features(series, query),
                    action_features(arrays["action_chunks"], query),
                ]
            )
    frame = pd.DataFrame(rows)
    frame["base_index"] = np.arange(len(frame), dtype=np.int64)
    # Each worker has at least five snapshots; this holds a whole snapshot out.
    frame["snapshot_fold"] = (
        frame["snapshot"].to_numpy(dtype=int)
        + 2 * frame["worker"].to_numpy(dtype=int)
    ) % 5
    frame["worker_fold"] = frame["worker"].to_numpy(dtype=int)
    return frame, np.stack(controls).astype(np.float32)


def route_schema() -> dict[str, tuple[list[str], list[str]]]:
    schema = route_auc.schemas()
    aggregate_names, aggregate_families, _ = schema["aggregate"]
    cell_names, cell_families, _ = schema["cell"]
    temporal_aggregate_names, temporal_aggregate_families, _ = schema[
        "temporal_aggregate"
    ]
    temporal_names, temporal_families, _ = schema["temporal"]
    return {
        "aggregate_current": (
            [f"aggregate/{name}" for name in aggregate_names],
            aggregate_families,
        ),
        "token_current": (
            [f"aggregate/{name}" for name in aggregate_names]
            + [f"position/{name}" for name in cell_names],
            aggregate_families + cell_families,
        ),
        "token_history": (
            [f"aggregate/{name}" for name in aggregate_names]
            + [f"aggregate/{name}" for name in temporal_aggregate_names]
            + [f"position/{name}" for name in cell_names]
            + [f"position/{name}" for name in temporal_names],
            aggregate_families
            + temporal_aggregate_families
            + cell_families
            + temporal_families,
        ),
    }


def extract_route_cache(run_root: Path, cache_path: Path) -> None:
    store = zarr.open_group(str(run_root / "formal/server/routes.zarr"), mode="r")
    episode = np.asarray(store["episode_id"][:], dtype=np.int32)
    control_step = np.asarray(store["control_step"][:], dtype=np.int32)
    probabilities = store["hb_router_probs"]
    schema = route_auc.schemas()
    current_aggregate = np.empty(
        (len(episode), len(schema["aggregate"][0])), dtype=np.float16
    )
    current_cell = np.empty(
        (len(episode), len(schema["cell"][0])), dtype=np.float16
    )
    temporal_aggregate = np.empty(
        (len(episode), len(schema["temporal_aggregate"][0])), dtype=np.float16
    )
    temporal_cell = np.empty(
        (len(episode), len(schema["temporal"][0])), dtype=np.float16
    )
    block = 32
    for start in range(0, len(episode), block):
        stop = min(start + block, len(episode))
        aggregate, cell = route_auc.current_route_features(
            np.asarray(probabilities[start:stop])
        )
        current_aggregate[start:stop] = aggregate.astype(np.float16)
        current_cell[start:stop] = cell.astype(np.float16)
        if start % 2048 == 0:
            print(f"current route rows {stop}/{len(episode)}", flush=True)
    route_query = np.empty(len(episode), dtype=np.int16)
    for axis, episode_id in enumerate(np.unique(episode)):
        indices = np.flatnonzero(episode == episode_id)
        indices = indices[np.argsort(control_step[indices], kind="stable")]
        raw = np.asarray(probabilities.oindex[indices])
        aggregate, cell = route_auc.temporal_route_features(raw)
        temporal_aggregate[indices] = aggregate.astype(np.float16)
        temporal_cell[indices] = cell.astype(np.float16)
        route_query[indices] = np.arange(len(indices), dtype=np.int16)
        if axis % 32 == 0:
            print(f"temporal episodes {axis + 1}/{len(np.unique(episode))}", flush=True)
    np.savez_compressed(
        cache_path,
        episode_id=episode,
        control_step=control_step,
        route_query=route_query,
        current_aggregate=current_aggregate,
        current_cell=current_cell,
        temporal_aggregate=temporal_aggregate,
        temporal_cell=temporal_cell,
    )


def load_route_cache(
    run_root: Path, cache_path: Path, rebuild: bool
) -> tuple[dict[str, np.ndarray], dict[tuple[int, int], int]]:
    if rebuild or not cache_path.is_file():
        extract_route_cache(run_root, cache_path)
    with np.load(cache_path, allow_pickle=False) as archive:
        episode = np.asarray(archive["episode_id"], dtype=np.int64)
        query = np.asarray(archive["route_query"], dtype=np.int64)
        current_aggregate = np.asarray(archive["current_aggregate"], dtype=np.float32)
        current_cell = np.asarray(archive["current_cell"], dtype=np.float32)
        temporal_aggregate = np.asarray(
            archive["temporal_aggregate"], dtype=np.float32
        )
        temporal_cell = np.asarray(archive["temporal_cell"], dtype=np.float32)
    lookup = {
        (int(episode_id), int(route_query)): index
        for index, (episode_id, route_query) in enumerate(
            zip(episode, query, strict=True)
        )
    }
    banks = {
        "aggregate_current": current_aggregate,
        "token_current": np.c_[current_aggregate, current_cell],
        "token_history": np.c_[
            current_aggregate, temporal_aggregate, current_cell, temporal_cell
        ],
    }
    return banks, lookup


def attach_route_rows(frame: pd.DataFrame, lookup: dict[tuple[int, int], int]) -> None:
    rows = []
    for item in frame.itertuples(index=False):
        key = (int(item.episode_id), int(item.query))
        if key not in lookup:
            raise ValueError(f"missing route row for episode/query {key}")
        rows.append(lookup[key])
    frame["route_row"] = np.asarray(rows, dtype=np.int64)


def event_dataset(base: pd.DataFrame, target: str, low: int, high: int) -> pd.DataFrame:
    onset_column = f"{target}_onset"
    rows = []
    for item in base.itertuples(index=False):
        onset = int(getattr(item, onset_column))
        query = int(item.query)
        if onset >= 0 and onset <= query:
            continue
        # An event before the requested lead band is a competing event, not a negative.
        if onset >= 0 and onset < query + low:
            continue
        positive = bool(query + low <= onset <= query + high)
        observed_through_horizon = item.n_query > query + high
        if not positive and not observed_through_horizon and not item.success:
            continue
        row = item._asdict()
        row["event"] = positive
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_logistic(matrix: np.ndarray, labels: np.ndarray):
    if len(np.unique(labels)) != 2:
        return None
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=3000, random_state=SEED),
    )
    model.fit(matrix, labels)
    return model


def predict_model(model, matrix: np.ndarray, fallback: float) -> np.ndarray:
    if model is None:
        return np.full(len(matrix), fallback, dtype=np.float64)
    return model.predict_proba(matrix)[:, 1]


def select_features(
    matrix: np.ndarray,
    target: np.ndarray,
    strata: np.ndarray,
    families: list[str],
) -> list[int]:
    return route_auc.select_by_family(matrix, target, strata, families)


def sibling_shift_indices(frame: pd.DataFrame, offset: int) -> np.ndarray:
    result = np.empty(len(frame), dtype=np.int64)
    for _, group in frame.groupby("stratum", sort=False):
        indices = group.sort_values("candidate").index.to_numpy(dtype=np.int64)
        result[indices] = np.roll(indices, -offset)
    return result


def append_predictions(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    indices: np.ndarray,
    scores: np.ndarray,
    *,
    target: str,
    band: str,
    split: str,
    fold: int,
    model: str,
    variant: str = "aligned",
) -> None:
    for position, index in enumerate(indices):
        item = frame.iloc[int(index)]
        rows.append(
            {
                "target": target,
                "band": band,
                "split": split,
                "fold": fold,
                "model": model,
                "variant": variant,
                "worker": int(item["worker"]),
                "snapshot_key": item["snapshot_key"],
                "candidate": int(item["candidate"]),
                "episode_id": int(item["episode_id"]),
                "query": int(item["query"]),
                "stratum": item["stratum"],
                "event": bool(item["event"]),
                "score": float(scores[position]),
            }
        )


def run_oof(
    frame: pd.DataFrame,
    controls: np.ndarray,
    route_banks: dict[str, np.ndarray],
    schemas: dict[str, tuple[list[str], list[str]]],
    *,
    target: str,
    band: str,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_column = "snapshot_fold" if split == "heldout_snapshot" else "worker_fold"
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    route_rows = frame["route_row"].to_numpy(dtype=np.int64)
    labels = frame["event"].to_numpy(dtype=bool)
    physical_dim = 1 + 17
    physical = controls[:, :physical_dim]
    for fold in sorted(frame[fold_column].unique()):
        test_index = np.flatnonzero(frame[fold_column].eq(fold).to_numpy())
        train_index = np.flatnonzero(~frame[fold_column].eq(fold).to_numpy())
        y_train = labels[train_index]
        fallback = float((y_train.sum() + 0.5) / (len(y_train) + 1.0))
        append_predictions(
            rows,
            frame,
            test_index,
            np.full(len(test_index), fallback),
            target=target,
            band=band,
            split=split,
            fold=int(fold),
            model="rate_only",
        )
        control_model = fit_logistic(controls[train_index], y_train)
        control_train = predict_model(control_model, controls[train_index], fallback)
        control_test = predict_model(control_model, controls[test_index], fallback)
        append_predictions(
            rows,
            frame,
            test_index,
            control_test,
            target=target,
            band=band,
            split=split,
            fold=int(fold),
            model="physical_action",
        )
        physical_model = fit_logistic(physical[train_index], y_train)
        physical_train = predict_model(
            physical_model, physical[train_index], fallback
        )
        physical_test = predict_model(physical_model, physical[test_index], fallback)
        append_predictions(
            rows,
            frame,
            test_index,
            physical_test,
            target=target,
            band=band,
            split=split,
            fold=int(fold),
            model="physical",
        )
        strata, _ = pd.factorize(frame.iloc[train_index]["stratum"], sort=True)
        residual = y_train.astype(np.float64) - control_train
        physical_residual = y_train.astype(np.float64) - physical_train
        for bank_name, route_matrix in route_banks.items():
            names, families = schemas[bank_name]
            values = route_matrix[route_rows]
            selected_raw = select_features(
                values[train_index], y_train, strata, families
            )
            raw_columns = np.asarray(selected_raw, dtype=np.int64)
            moe_model = fit_logistic(values[train_index][:, raw_columns], y_train)
            moe_test = predict_model(
                moe_model, values[test_index][:, raw_columns], fallback
            )
            append_predictions(
                rows,
                frame,
                test_index,
                moe_test,
                target=target,
                band=band,
                split=split,
                fold=int(fold),
                model=f"moe_{bank_name}",
            )
            selected_joint = select_features(
                values[train_index], residual, strata, families
            )
            joint_columns = np.asarray(selected_joint, dtype=np.int64)
            joint_train = np.c_[controls[train_index], values[train_index][:, joint_columns]]
            joint_test = np.c_[controls[test_index], values[test_index][:, joint_columns]]
            joint_model = fit_logistic(joint_train, y_train)
            joint_score = predict_model(joint_model, joint_test, fallback)
            joint_name = f"physical_action+{bank_name}"
            append_predictions(
                rows,
                frame,
                test_index,
                joint_score,
                target=target,
                band=band,
                split=split,
                fold=int(fold),
                model=joint_name,
            )
            coefficient = None
            if joint_model is not None:
                coefficient = joint_model.named_steps["logisticregression"].coef_[0]
                coefficient = coefficient[-len(joint_columns) :]
            for rank, feature_index in enumerate(joint_columns):
                selections.append(
                    {
                        "target": target,
                        "band": band,
                        "split": split,
                        "fold": int(fold),
                        "model": joint_name,
                        "rank": rank + 1,
                        "family": families[feature_index],
                        "feature": names[feature_index],
                        "standardized_coefficient": (
                            float(coefficient[rank]) if coefficient is not None else np.nan
                        ),
                    }
                )
            selected_physical = select_features(
                values[train_index], physical_residual, strata, families
            )
            physical_columns = np.asarray(selected_physical, dtype=np.int64)
            physical_joint_train = np.c_[
                physical[train_index], values[train_index][:, physical_columns]
            ]
            physical_joint_test = np.c_[
                physical[test_index], values[test_index][:, physical_columns]
            ]
            physical_joint_model = fit_logistic(physical_joint_train, y_train)
            physical_joint_score = predict_model(
                physical_joint_model, physical_joint_test, fallback
            )
            physical_joint_name = f"physical+{bank_name}"
            append_predictions(
                rows,
                frame,
                test_index,
                physical_joint_score,
                target=target,
                band=band,
                split=split,
                fold=int(fold),
                model=physical_joint_name,
            )
            physical_coefficient = None
            if physical_joint_model is not None:
                physical_coefficient = physical_joint_model.named_steps[
                    "logisticregression"
                ].coef_[0][-len(physical_columns) :]
            for rank, feature_index in enumerate(physical_columns):
                selections.append(
                    {
                        "target": target,
                        "band": band,
                        "split": split,
                        "fold": int(fold),
                        "model": physical_joint_name,
                        "rank": rank + 1,
                        "family": families[feature_index],
                        "feature": names[feature_index],
                        "standardized_coefficient": (
                            float(physical_coefficient[rank])
                            if physical_coefficient is not None
                            else np.nan
                        ),
                    }
                )
            test_frame = frame.iloc[test_index].reset_index(drop=True)
            test_values = values[test_index]
            for offset in SHIFT_OFFSETS:
                shifted = sibling_shift_indices(test_frame, offset)
                shifted_joint = np.c_[
                    controls[test_index], test_values[shifted][:, joint_columns]
                ]
                shifted_score = predict_model(joint_model, shifted_joint, fallback)
                append_predictions(
                    rows,
                    frame,
                    test_index,
                    shifted_score,
                    target=target,
                    band=band,
                    split=split,
                    fold=int(fold),
                    model=joint_name,
                    variant=f"shift_{offset}",
                )
                shifted_physical_joint = np.c_[
                    physical[test_index], test_values[shifted][:, physical_columns]
                ]
                shifted_physical_score = predict_model(
                    physical_joint_model, shifted_physical_joint, fallback
                )
                append_predictions(
                    rows,
                    frame,
                    test_index,
                    shifted_physical_score,
                    target=target,
                    band=band,
                    split=split,
                    fold=int(fold),
                    model=physical_joint_name,
                    variant=f"shift_{offset}",
                )
    return pd.DataFrame(rows), pd.DataFrame(selections)


def bootstrap_mean(
    values: np.ndarray, draws: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(array), size=(draws, len(array)))
    distribution = array[samples].mean(axis=1)
    low, high = np.quantile(distribution, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(array.mean()), float(low), float(high)


def metric_tables(
    predictions: pd.DataFrame, draws: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratum_rows = []
    keys = ["target", "band", "split", "model", "variant"]
    for key, group in predictions.groupby(keys, sort=False):
        for stratum, cell in group.groupby("stratum", sort=False):
            labels = cell["event"].to_numpy(dtype=bool)
            if len(np.unique(labels)) != 2:
                continue
            positives = int(labels.sum())
            negatives = int((~labels).sum())
            stratum_rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "snapshot_key": cell["snapshot_key"].iloc[0],
                    "worker": int(cell["worker"].iloc[0]),
                    "query": int(cell["query"].iloc[0]),
                    "stratum": stratum,
                    "positives": positives,
                    "negatives": negatives,
                    "pairs": positives * negatives,
                    "auc": float(roc_auc_score(labels, cell["score"])),
                }
            )
    strata = pd.DataFrame(stratum_rows)
    shifted = strata[strata["variant"].str.startswith("shift_")]
    if not shifted.empty:
        shift_mean = (
            shifted.groupby(
                ["target", "band", "split", "model", "snapshot_key", "worker", "query", "stratum"],
                as_index=False,
            )
            .agg(
                positives=("positives", "first"),
                negatives=("negatives", "first"),
                pairs=("pairs", "first"),
                auc=("auc", "mean"),
            )
        )
        shift_mean["variant"] = "shift_mean"
        strata = pd.concat((strata, shift_mean), ignore_index=True)
    snapshot_rows = []
    for key, group in strata.groupby(keys, sort=False):
        for snapshot, block in group.groupby("snapshot_key", sort=False):
            snapshot_rows.append(
                {
                    **dict(zip(keys, key, strict=True)),
                    "snapshot_key": snapshot,
                    "worker": int(block["worker"].iloc[0]),
                    "mixed_strata": len(block),
                    "pairs": int(block["pairs"].sum()),
                    "auc": float(np.average(block["auc"], weights=block["pairs"])),
                }
            )
    snapshots = pd.DataFrame(snapshot_rows)
    prediction_lookup = {
        key: group for key, group in predictions.groupby(keys, sort=False)
    }
    metric_rows = []
    for axis, (key, group) in enumerate(snapshots.groupby(keys, sort=False)):
        mean, low, high = bootstrap_mean(
            group["auc"].to_numpy(), draws, seed + axis
        )
        prediction = prediction_lookup.get(key)
        metric_rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "rows": len(prediction) if prediction is not None else np.nan,
                "positives": (
                    int(prediction["event"].sum()) if prediction is not None else np.nan
                ),
                "mixed_strata": int(group["mixed_strata"].sum()),
                "mixed_snapshots": len(group),
                "macro_snapshot_auc": mean,
                "auc_ci95_low": low,
                "auc_ci95_high": high,
                "pair_weighted_auc": float(
                    np.average(group["auc"], weights=group["pairs"])
                ),
                "brier": (
                    float(
                        np.mean(
                            (
                                prediction["score"].to_numpy()
                                - prediction["event"].to_numpy(dtype=float)
                            )
                            ** 2
                        )
                    )
                    if prediction is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(metric_rows), strata, snapshots


def delta_table(
    predictions: pd.DataFrame,
    snapshots: pd.DataFrame,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    aligned = snapshots[snapshots["variant"].eq("aligned")]
    shifted = snapshots[snapshots["variant"].eq("shift_mean")]
    comparisons = {
        "action_increment": ("physical", "physical_action"),
        "aggregate_moe_over_physical": (
            "physical",
            "physical+aggregate_current",
        ),
        "token_current_moe_over_physical": (
            "physical",
            "physical+token_current",
        ),
        "token_history_moe_over_physical": (
            "physical",
            "physical+token_history",
        ),
        "aggregate_moe_increment": (
            "physical_action",
            "physical_action+aggregate_current",
        ),
        "token_position_increment": (
            "physical_action+aggregate_current",
            "physical_action+token_current",
        ),
        "token_current_total_moe_increment": (
            "physical_action",
            "physical_action+token_current",
        ),
        "history_increment": (
            "physical_action+token_current",
            "physical_action+token_history",
        ),
        "total_moe_increment": (
            "physical_action",
            "physical_action+token_history",
        ),
    }
    rows = []
    axis = 0
    for (target, band, split), group in aligned.groupby(
        ["target", "band", "split"], sort=False
    ):
        pivot = group.pivot(index="snapshot_key", columns="model", values="auc")
        for comparison, (left, right) in comparisons.items():
            if left not in pivot or right not in pivot:
                continue
            values = (pivot[right] - pivot[left]).dropna().to_numpy()
            mean, low, high = bootstrap_mean(values, draws, seed + axis)
            _, family_low, family_high = bootstrap_mean(
                values, draws, seed + axis, alpha=0.05 / 9.0
            )
            axis += 1
            rows.append(
                {
                    "target": target,
                    "band": band,
                    "split": split,
                    "comparison": comparison,
                    "snapshots": len(values),
                    "auc_gain": mean,
                    "auc_gain_ci95_low": low,
                    "auc_gain_ci95_high": high,
                    "auc_gain_9test_low": family_low,
                    "auc_gain_9test_high": family_high,
                }
            )
        for bank_name in ("aggregate_current", "token_current", "token_history"):
            for control_prefix in ("physical_action", "physical"):
                model = f"{control_prefix}+{bank_name}"
                joint = group[group["model"].eq(model)][["snapshot_key", "auc"]]
                shift = shifted[
                    shifted["target"].eq(target)
                    & shifted["band"].eq(band)
                    & shifted["split"].eq(split)
                    & shifted["model"].eq(model)
                ][["snapshot_key", "auc"]]
                matched = joint.merge(
                    shift, on="snapshot_key", suffixes=("_aligned", "_shift")
                )
                if not len(matched):
                    continue
                values = matched["auc_aligned"].to_numpy() - matched[
                    "auc_shift"
                ].to_numpy()
                mean, low, high = bootstrap_mean(values, draws, seed + axis)
                axis += 1
                prefix = "" if control_prefix == "physical_action" else "physical_"
                rows.append(
                    {
                        "target": target,
                        "band": band,
                        "split": split,
                        "comparison": f"aligned_vs_shifted_{prefix}{bank_name}",
                        "snapshots": len(values),
                        "auc_gain": mean,
                        "auc_gain_ci95_low": low,
                        "auc_gain_ci95_high": high,
                        "auc_gain_9test_low": np.nan,
                        "auc_gain_9test_high": np.nan,
                    }
                )
    # Brier is evaluated on every risk-set row, not just mixed strata.
    aligned_predictions = predictions[predictions["variant"].eq("aligned")]
    for (target, band, split), group in aligned_predictions.groupby(
        ["target", "band", "split"], sort=False
    ):
        table = (
            group.assign(error=lambda x: (x["score"] - x["event"].astype(float)) ** 2)
            .groupby(["snapshot_key", "model"])["error"]
            .mean()
            .unstack("model")
        )
        left = "physical_action"
        right = "physical_action+token_history"
        if left not in table or right not in table:
            continue
        values = (table[left] - table[right]).dropna().to_numpy()
        mean, low, high = bootstrap_mean(values, draws, seed + axis)
        axis += 1
        rows.append(
            {
                "target": target,
                "band": band,
                "split": split,
                "comparison": "total_moe_brier_gain",
                "snapshots": len(values),
                "auc_gain": mean,
                "auc_gain_ci95_low": low,
                "auc_gain_ci95_high": high,
                "auc_gain_9test_low": np.nan,
                "auc_gain_9test_high": np.nan,
            }
        )
    return pd.DataFrame(rows)


def support_table(datasets: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for (target, band), frame in datasets.items():
        mixed = frame.groupby("stratum")["event"].nunique()
        mixed_strata = set(mixed.index[mixed.eq(2)])
        mixed_frame = frame[frame["stratum"].isin(mixed_strata)]
        rows.append(
            {
                "target": target,
                "band": band,
                "risk_rows": len(frame),
                "events": int(frame["event"].sum()),
                "mixed_strata": len(mixed_strata),
                "mixed_snapshots": mixed_frame["snapshot_key"].nunique(),
                "workers_with_mixed_strata": mixed_frame["worker"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def far_worker_audit(snapshot_aucs: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    selected = snapshot_aucs[
        snapshot_aucs["target"].eq("trap")
        & snapshot_aucs["band"].eq("next_9_12")
        & snapshot_aucs["split"].eq("heldout_worker")
        & snapshot_aucs["variant"].eq("aligned")
        & snapshot_aucs["model"].isin(
            ("physical_action", "physical_action+aggregate_current")
        )
    ]
    pivot = selected.pivot(
        index=["snapshot_key", "worker"], columns="model", values="auc"
    ).reset_index()
    pivot["auc_gain"] = (
        pivot["physical_action+aggregate_current"] - pivot["physical_action"]
    )
    worker = (
        pivot.groupby("worker", as_index=False)
        .agg(
            snapshots=("snapshot_key", "nunique"),
            physical_action_auc=("physical_action", "mean"),
            joint_auc=("physical_action+aggregate_current", "mean"),
            auc_gain=("auc_gain", "mean"),
        )
        .sort_values("worker")
    )
    values = worker["auc_gain"].to_numpy(dtype=float)
    observed = float(values.mean())
    signs = np.asarray(
        [
            [1.0 if (mask >> axis) & 1 else -1.0 for axis in range(len(values))]
            for mask in range(2 ** len(values))
        ]
    )
    distribution = (signs * values).mean(axis=1)
    p_value = float(np.mean(distribution >= observed - 1e-12))
    return worker, p_value


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "(no rows)"
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("NA")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.{digits}f}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_auc(metrics: pd.DataFrame, out: Path) -> None:
    selected = metrics[
        metrics["target"].eq("trap")
        & metrics["split"].eq("heldout_snapshot")
        & metrics["variant"].eq("aligned")
        & metrics["model"].isin(
            (
                "physical_action",
                "physical_action+aggregate_current",
                "physical_action+token_current",
                "physical_action+token_history",
            )
        )
    ].copy()
    order = list(LEAD_BANDS)
    selected["band_axis"] = selected["band"].map({value: i for i, value in enumerate(order)})
    style = {
        "physical_action": ("Physical + action", "#666666", "--"),
        "physical_action+aggregate_current": ("+ token mean", "#2f6b8a", "-"),
        "physical_action+token_current": ("+ token positions", "#b5473c", "-"),
        "physical_action+token_history": ("+ token positions + history", "#d58b32", "-"),
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for model, block in selected.groupby("model", sort=False):
        block = block.sort_values("band_axis")
        label, color, linestyle = style[model]
        ax.plot(
            block["band_axis"],
            block["macro_snapshot_auc"],
            marker="o",
            color=color,
            linestyle=linestyle,
            label=label,
        )
        if model == "physical_action+token_history":
            ax.fill_between(
                block["band_axis"],
                block["auc_ci95_low"],
                block["auc_ci95_high"],
                color=color,
                alpha=0.16,
            )
    ax.axhline(0.5, color="#222222", linewidth=1)
    ax.set_xticks(range(len(order)), [value.replace("next_", "") for value in order])
    ax.set(xlabel="Queries before first trap onset", ylabel="Within-snapshot ROC AUC")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def write_report(
    out: Path,
    episodes: pd.DataFrame,
    support: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    selections: pd.DataFrame,
    stratum_aucs: pd.DataFrame,
    worker_audit: pd.DataFrame,
    worker_signflip_p: float,
) -> None:
    models = [
        "physical",
        "physical+aggregate_current",
        "physical+token_history",
        "physical_action",
        "physical_action+aggregate_current",
        "physical_action+token_current",
        "physical_action+token_history",
    ]
    primary = metrics[
        metrics["target"].eq("trap")
        & metrics["split"].eq("heldout_snapshot")
        & metrics["variant"].eq("aligned")
        & metrics["model"].isin(models)
    ][
        [
            "band",
            "model",
            "mixed_snapshots",
            "macro_snapshot_auc",
            "auc_ci95_low",
            "auc_ci95_high",
            "brier",
        ]
    ].copy()
    primary.columns = ["lead", "model", "snapshots", "AUC", "CI low", "CI high", "Brier"]
    sensitivity = metrics[
        metrics["target"].eq("trap")
        & metrics["band"].eq("next_1_4")
        & metrics["variant"].eq("aligned")
        & metrics["model"].isin(
            (
                "physical",
                "physical+token_history",
                "physical_action",
                "physical_action+token_history",
            )
        )
    ][
        ["split", "model", "mixed_snapshots", "macro_snapshot_auc", "auc_ci95_low", "auc_ci95_high"]
    ].copy()
    sensitivity.columns = ["split", "model", "snapshots", "AUC", "CI low", "CI high"]
    subtype = metrics[
        metrics["band"].eq("next_1_4")
        & metrics["split"].eq("heldout_snapshot")
        & metrics["variant"].eq("aligned")
        & metrics["model"].isin(("physical_action", "physical_action+token_history"))
    ][["target", "model", "mixed_snapshots", "macro_snapshot_auc", "auc_ci95_low", "auc_ci95_high"]]
    subtype.columns = ["target", "model", "snapshots", "AUC", "CI low", "CI high"]
    delta = deltas[
        deltas["target"].eq("trap")
        & deltas["band"].eq("next_1_4")
        & deltas["comparison"].isin(
            (
                "total_moe_increment",
                "aligned_vs_shifted_token_history",
                "total_moe_brier_gain",
            )
        )
    ][
        ["split", "comparison", "snapshots", "auc_gain", "auc_gain_ci95_low", "auc_gain_ci95_high"]
    ].copy()
    delta.columns = ["split", "comparison", "snapshots", "gain", "CI low", "CI high"]
    far = deltas[
        deltas["target"].eq("trap")
        & deltas["band"].eq("next_9_12")
        & deltas["comparison"].isin(
            (
                "action_increment",
                "aggregate_moe_over_physical",
                "aggregate_moe_increment",
                "aligned_vs_shifted_aggregate_current",
                "aligned_vs_shifted_physical_aggregate_current",
            )
        )
    ][
        [
            "split",
            "comparison",
            "snapshots",
            "auc_gain",
            "auc_gain_ci95_low",
            "auc_gain_ci95_high",
            "auc_gain_9test_low",
            "auc_gain_9test_high",
        ]
    ].copy()
    far.columns = [
        "split",
        "comparison",
        "snapshots",
        "gain",
        "CI low",
        "CI high",
        "9-test low",
        "9-test high",
    ]
    far_subtype = deltas[
        deltas["band"].eq("next_9_12")
        & deltas["comparison"].eq("aggregate_moe_increment")
    ][
        [
            "target",
            "split",
            "snapshots",
            "auc_gain",
            "auc_gain_ci95_low",
            "auc_gain_ci95_high",
        ]
    ].copy()
    far_subtype.columns = ["target", "split", "snapshots", "gain", "CI low", "CI high"]
    primary_select = selections[
        selections["target"].eq("trap")
        & selections["band"].eq("next_1_4")
        & selections["split"].eq("heldout_snapshot")
        & selections["model"].eq("physical_action+token_history")
    ]
    top = (
        primary_select.groupby(["family", "feature"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            coefficient_mean=("standardized_coefficient", "mean"),
            coefficient_positive=("standardized_coefficient", lambda x: int((x > 0).sum())),
        )
        .sort_values(["folds", "family"], ascending=[False, True])
        .head(12)
    )
    top.columns = ["family", "feature", "folds", "coef mean", "positive folds"]
    far_select = selections[
        selections["target"].eq("trap")
        & selections["band"].eq("next_9_12")
        & selections["split"].eq("heldout_snapshot")
        & selections["model"].eq("physical_action+aggregate_current")
    ]
    far_top = (
        far_select.groupby(["family", "feature"], as_index=False)
        .agg(
            folds=("fold", "nunique"),
            coefficient_mean=("standardized_coefficient", "mean"),
            coefficient_positive=(
                "standardized_coefficient", lambda x: int((x > 0).sum())
            ),
        )
        .sort_values(["folds", "family"], ascending=[False, True])
        .head(10)
    )
    far_top.columns = ["family", "feature", "folds", "coef mean", "positive folds"]
    query_source = stratum_aucs[
        stratum_aucs["target"].eq("trap")
        & stratum_aucs["band"].eq("next_9_12")
        & stratum_aucs["split"].eq("heldout_snapshot")
        & stratum_aucs["variant"].eq("aligned")
        & stratum_aucs["model"].isin(
            ("physical_action", "physical_action+aggregate_current")
        )
    ]
    query_pivot = query_source.pivot(
        index=["snapshot_key", "query", "stratum"], columns="model", values="auc"
    ).reset_index()
    query_pivot["gain"] = (
        query_pivot["physical_action+aggregate_current"]
        - query_pivot["physical_action"]
    )
    query_audit = (
        query_pivot.groupby("query", as_index=False)["gain"]
        .agg(["count", "mean", "median"])
    )
    query_audit.columns = ["q", "mixed strata", "mean gain", "median gain"]
    recovery = episodes[episodes["trap_onset"].ge(0)]
    event_counts = pd.DataFrame(
        [
            {
                "event": "loop onset",
                "branches": int(episodes["loop_onset"].ge(0).sum()),
                "successful": int(episodes.loc[episodes["loop_onset"].ge(0), "success"].sum()),
            },
            {
                "event": "80-action static onset",
                "branches": int(episodes["static_onset"].ge(0).sum()),
                "successful": int(episodes.loc[episodes["static_onset"].ge(0), "success"].sum()),
            },
            {
                "event": "either trap onset",
                "branches": int(episodes["trap_onset"].ge(0).sum()),
                "successful": int(episodes.loc[episodes["trap_onset"].ge(0), "success"].sum()),
            },
            {
                "event": "post-trap recovery >=3.5cm",
                "branches": int(recovery["post_trap_recovered_035m"].sum()),
                "successful": int(recovery.loc[recovery["post_trap_recovered_035m"], "success"].sum()),
            },
        ]
    )
    primary_delta = delta[
        delta["split"].eq("heldout_snapshot")
        & delta["comparison"].eq("total_moe_increment")
    ].iloc[0]
    strict_delta = delta[
        delta["split"].eq("heldout_worker")
        & delta["comparison"].eq("total_moe_increment")
    ].iloc[0]
    far_increment = far[far["comparison"].eq("aggregate_moe_increment")]
    far_uncorrected = bool((far_increment["CI low"] > 0).all())
    far_corrected = bool((far_increment["9-test low"] > 0).all())
    if primary_delta["CI low"] > 0 and strict_delta["gain"] > 0:
        conclusion = (
            "临近 1–4 query 时，MoE 在物理/动作控制之外提供了信息，且严格留出 worker 时仍同方向。"
        )
    elif primary_delta["gain"] > 0:
        conclusion = (
            "临近 1–4 query 时，MoE 只在 snapshot 留出中有正增量，没有通过严格 worker 泛化。"
        )
    else:
        conclusion = (
            "临近 1–4 query 时，MoE 没有超过物理/动作前缀控制，不能作为独立 trap 报警器。"
        )
    if far_corrected:
        early_conclusion = (
            "但 9–12 个 query 提前量的 token-mean 路由在两种拆分及 9 重校正后仍有正增量，形成候选早期信号。"
        )
    elif far_uncorrected:
        early_conclusion = (
            "9–12 个 query 提前量出现了两种拆分同方向的 token-mean 候选信号，但未通过 9 重校正，只能列为探索性发现。"
        )
    else:
        early_conclusion = "更早的 9–12 query 窗口也没有稳定的校正后增量。"
    lines = [
        "# 逐 chunk trap onset 的 MoE 实验",
        "",
        "## 结论",
        "",
        f"**{conclusion}**",
        "",
        early_conclusion,
        "",
        "本实验不再预测最终 success/failure，而是在尚未发生 trap 的风险集中，预测未来固定 query 区间的首次物理事件。",
        "",
        "## 事件覆盖",
        "",
        markdown_table(event_counts),
        "",
        "恢复阈值是 onset 后目标距离至少再改善 3.5 cm。恢复样本过少，不训练恢复分类器。",
        "",
        "## 主结果：同 snapshot、同 q 的未来 trap",
        "",
        markdown_table(primary),
        "",
        "AUC 先在每个 `snapshot x q` 内计算，再按 snapshot 平均；因此初态、主干阶段、q 编号和剩余预算都不能直接产生排序。",
        "`physical` 只读状态轨迹；`physical_action` 再加入当前 action chunk。MoE 模型每个外层训练折内每个特征族只选一个坐标。",
        "",
        "## MoE 增量与 sibling 错配",
        "",
        markdown_table(delta),
        "",
        "`total_moe_increment` 是完整逐 token+历史 MoE 相对物理+动作控制的 AUC 增量；`aligned_vs_shifted_token_history` 把测试分支的 MoE 换成同 snapshot、同 q 的另一个候选。",
        "Brier gain 与 AUC gain 单位不同，表中保留原始数值。",
        "",
        "## 提前 9–12 query 的探索性信号",
        "",
        markdown_table(far),
        "",
        "这里使用 action-token 均值后的当前路由；`9-test` 区间同时校正 3 个提前量 x 3 种 MoE 表示。`action_increment` 可判断路由信息是否只是当前动作计划的替代。该窗口不是主终点，必须按探索性结果解释。",
        "",
        "按决策 q 拆开：",
        "",
        markdown_table(query_audit),
        "",
        "该窗口反复选择的 token-mean 坐标：",
        "",
        markdown_table(far_top),
        "",
        "四个 worker 的严格留出增量：",
        "",
        markdown_table(worker_audit),
        "",
        f"4-worker 精确单侧 sign-flip p={worker_signflip_p:.4f}；只有四个独立初态时，4/4 同方向的最小 p 就是 0.0625。",
        "增量幅度主要由 worker3 提供，其余三个 worker 只有小幅正值；这也是候选信号不能升级为结论的原因。",
        "",
        "远期候选按事件类型拆开：",
        "",
        markdown_table(far_subtype),
        "",
        "正增量主要指向 loop 风险；连续静止的 snapshot 留出增量为负。因此不能把该候选解释成通用的停滞编码。",
        "",
        "## 严格 worker 留出",
        "",
        markdown_table(sensitivity),
        "",
        "worker 对应四个独立初态。worker1 几乎都 trap、worker3 几乎都不 trap，所以该拆分是高域偏移敏感性，不应只看 pooled AUC。",
        "",
        "## Loop 与静止分开",
        "",
        markdown_table(subtype),
        "",
        "Loop onset 使用冻结的非局部物理回返规则；静止 onset 是第一次能够因果确认连续 80 个 action step 低运动的 query。两者标签均不读取 MoE。",
        "",
        "## 主模型反复选择的坐标",
        "",
        markdown_table(top),
        "",
        "特征若只在少数折出现，不能解释为稳定机制；系数正值表示该路由量更高时预测临近 trap。",
        "",
        "## 设计边界",
        "",
        "- 这是观察性预测，不证明 MoE 路由导致或因果识别了 trap。",
        "- 物理标签来自 MuJoCo 状态与成功终态参考，不是人工语义标注。",
        "- 只有一个 Long 任务、四个 worker；跨任务泛化仍需同协议的新采集。",
        "- 当前 route cache 没有隐藏层；结论只涉及保存的 HB-MoE 概率。",
    ]
    (out / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    out = (args.out_dir or run_root / DEFAULT_OUT_NAME).resolve()
    out.mkdir(parents=True, exist_ok=True)
    candidates, discovery_audit = rolling.discover_candidates(run_root)
    targets, layout_audit = rolling.load_layout(run_root)
    trajectories = {
        candidate.episode_id: rolling.load_trajectory(candidate)
        for candidate in candidates
    }
    references = rolling.build_goal_references(candidates, trajectories, targets)
    episodes, series = episode_table(candidates, trajectories, targets, references)
    base, controls = build_landmark_table(episodes, trajectories, series)
    cache_path = out / "route_feature_cache.npz"
    route_banks, lookup = load_route_cache(
        run_root, cache_path, args.rebuild_cache
    )
    attach_route_rows(base, lookup)
    schemas = route_schema()
    if controls.shape[1] != len(control_names()):
        raise ValueError("control feature schema mismatch")

    datasets = {}
    predictions = []
    selections = []
    for target in TARGETS:
        for band, (low, high) in LEAD_BANDS.items():
            frame = event_dataset(base, target, low, high)
            datasets[(target, band)] = frame
            if frame["event"].sum() < 8 or (~frame["event"]).sum() < 8:
                print(f"skip {target}/{band}: insufficient outcomes", flush=True)
                continue
            control = controls[frame["base_index"].to_numpy(dtype=np.int64)]
            for split in ("heldout_snapshot", "heldout_worker"):
                prediction, selection = run_oof(
                    frame,
                    control,
                    route_banks,
                    schemas,
                    target=target,
                    band=band,
                    split=split,
                )
                predictions.append(prediction)
                selections.append(selection)
                print(
                    f"OOF {target}/{band}/{split}: rows={len(frame)} "
                    f"events={int(frame['event'].sum())}",
                    flush=True,
                )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    selection_frame = pd.concat(selections, ignore_index=True)
    metrics, stratum_aucs, snapshot_aucs = metric_tables(
        prediction_frame, args.bootstraps, args.seed
    )
    deltas = delta_table(
        prediction_frame, snapshot_aucs, args.bootstraps, args.seed + 100000
    )
    support = support_table(datasets)
    worker_audit, worker_signflip_p = far_worker_audit(snapshot_aucs)

    episodes.to_csv(out / "event_onsets.csv", index=False)
    base.to_csv(out / "landmark_rows.csv", index=False)
    pd.DataFrame({"feature": control_names()}).to_csv(
        out / "control_feature_dictionary.csv", index=False
    )
    support.to_csv(out / "support.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    deltas.to_csv(out / "deltas.csv", index=False)
    stratum_aucs.to_csv(out / "stratum_aucs.csv.gz", index=False)
    snapshot_aucs.to_csv(out / "snapshot_aucs.csv", index=False)
    prediction_frame.to_csv(out / "oof_predictions.csv.gz", index=False)
    selection_frame.to_csv(out / "feature_selections.csv", index=False)
    worker_audit.to_csv(out / "far_worker_audit.csv", index=False)
    audit = {
        "run_root": str(run_root),
        "branches": len(episodes),
        "landmarks": LANDMARKS,
        "lead_bands": LEAD_BANDS,
        "static_event_actions": STATIC_EVENT_ACTIONS,
        "loop_thresholds": {
            "min_query_lag": rolling.LOOP_MIN_QUERY_LAG,
            "eef_return_m": rolling.LOOP_EEF_RETURN_M,
            "object_return_m": rolling.LOOP_OBJECT_RETURN_M,
            "gripper_return_m": rolling.LOOP_GRIPPER_RETURN_M,
            "eef_path_m": rolling.LOOP_EEF_PATH_M,
        },
        "static_thresholds": {
            "window": rolling.STATIC_WINDOW,
            "eef_path_m": rolling.STATIC_EEF_PATH_M,
            "object_path_m": rolling.STATIC_OBJECT_PATH_M,
            "gripper_path_m": rolling.STATIC_GRIPPER_PATH_M,
        },
        "bootstraps": args.bootstraps,
        "far_worker_signflip_p_one_sided": worker_signflip_p,
        "discovery_audit": discovery_audit,
        "layout_audit": layout_audit,
    }
    write_json(out / "audit.json", audit)
    plot_auc(metrics, out / "trap_auc_by_lead.png")
    write_report(
        out,
        episodes,
        support,
        metrics,
        deltas,
        selection_frame,
        stratum_aucs,
        worker_audit,
        worker_signflip_p,
    )
    outputs = sorted(
        path
        for path in out.iterdir()
        if path.is_file()
        and path.name not in {"checksums.sha256", "route_feature_cache.npz"}
    )
    (out / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
