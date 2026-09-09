#!/usr/bin/env python3
"""Test MoE routing signatures around physically defined long-task failures.

The primary prediction protocol is deliberately prospective: every episode is
cut eight queries after pot 2 first reaches its success-goal proxy, and only
episodes whose physical failure onset is later than that cut are retained.
Routing is evaluated after physical and action baselines with leave-one-initial-
state-out predictions.  Event-aligned curves are descriptive and are reported
separately from the prospective result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from analyze_failure_routing_clusters import (
    normalize_probabilities,
    selected_expert_occupancy,
)


HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
ANALYSIS = HERE / "analysis"
AUDIT_CSV = ANALYSIS / "residual-failure-physical-audit/episode_audit.csv"
EVENT_AUDIT_CSV = ANALYSIS / "failure-event-audit/episode_events.csv"
DYNAMICS_CSV = ANALYSIS / "residual-failure-dynamics/assignments.csv"
EVENT_FEATURES_NPZ = ANALYSIS / "route-change-events/event_features.npz"
FEATURE_CACHE = ANALYSIS / "all-outcome-routing-clusters/feature_cache"
OUT_DIR = ANALYSIS / "failure-moe-signatures"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
PREFIX_AFTER_POT2 = 8
APPROACH_DISTANCE_M = 0.13
LOW_STEP_M = 0.01
LOW_STEP_P80_M = 0.015
LOW_STEP_FRACTION = 0.80
MIN_STASIS_TRANSITIONS = 4
EVENT_OFFSETS = np.arange(-10, 9, dtype=np.int64)
SEED = 20260829
BOOTSTRAPS = 2000

PHYSICAL_SIGNALS = (
    "eef_step_m",
    "eef_pot1_distance_m",
    "eef_pot2_distance_m",
    "pot1_step_m",
    "pot2_step_m",
    "pot1_goal_distance_m",
    "pot2_goal_distance_m",
    "eef_pot1_distance_change_m",
)
ACTION_SIGNALS = (
    "action_translation",
    "action_rotation",
    "action_gripper",
    "action_change",
    "action_recurrence_advantage",
    "action_anchor_advantage",
    "action_gripper_flip",
)
ROUTING_SIGNALS = (
    "route_speed",
    "route_recurrence_advantage",
    "route_recurrence_lag_fraction",
    "route_anchor_advantage",
    "route_entropy",
    "route_top1_mass",
    "expert_speed",
    "expert_entropy",
    "expert_top1_mass",
    "route_layer_synchrony",
)
ALL_SIGNALS = PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS
SUMMARY_STATS = ("last", "mean", "std", "slope")


@dataclass
class EpisodeTape:
    episode: int
    init_state_id: int
    flow_noise_seed: int
    success: bool
    q0: int
    eef: np.ndarray
    pot1: np.ndarray
    pot2: np.ndarray
    actions: np.ndarray
    signals: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--audit-csv", type=pathlib.Path, default=AUDIT_CSV)
    parser.add_argument(
        "--event-audit-csv", type=pathlib.Path, default=EVENT_AUDIT_CSV
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _step(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    if len(values) > 1:
        result[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return result


def _hellinger_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5
        * np.sum(
            np.square(
                np.sqrt(np.maximum(left, 0.0))
                - np.sqrt(np.maximum(right, 0.0))
            ),
            axis=-1,
        )
    )


def pairwise_hellinger(values: np.ndarray) -> np.ndarray:
    """Mean Hellinger distance over all non-expert sites."""
    values = np.asarray(values, dtype=np.float32)
    root = np.sqrt(np.maximum(values, 0.0))
    distance = np.sqrt(
        0.5
        * np.sum(
            np.square(root[:, None, ...] - root[None, :, ...]), axis=-1
        )
    )
    if distance.ndim > 2:
        distance = distance.mean(axis=tuple(range(2, distance.ndim)))
    return distance.astype(np.float32)


def _causal_scale(speed: np.ndarray) -> np.ndarray:
    scale = np.ones(len(speed), dtype=np.float32)
    positive: list[float] = []
    for query in range(1, len(speed)):
        value = float(speed[query])
        if np.isfinite(value) and value > 1e-7:
            positive.append(value)
        scale[query] = max(float(np.median(positive)) if positive else value, 1e-6)
    return scale


def recurrence_signals(
    distance: np.ndarray, speed: np.ndarray, anchor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(speed)
    advantage = np.zeros(n, dtype=np.float32)
    lag = np.zeros(n, dtype=np.float32)
    anchor_advantage = np.zeros(n, dtype=np.float32)
    scale = _causal_scale(speed)
    for query in range(2, n):
        older = distance[query, : query - 1]
        nearest = int(np.argmin(older))
        advantage[query] = np.clip(
            (distance[query, query - 1] - older[nearest]) / scale[query],
            -10.0,
            10.0,
        )
        lag[query] = (query - nearest) / query
        if query > anchor:
            anchor_advantage[query] = np.clip(
                (distance[query, query - 1] - distance[query, anchor])
                / scale[query],
                -10.0,
                10.0,
            )
    return advantage, lag, anchor_advantage


def action_distance(actions: np.ndarray) -> np.ndarray:
    scale = np.asarray([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 1.0])
    normalized = actions / scale[None, None, :]
    flat = normalized.reshape(len(actions), -1)
    return np.sqrt(np.mean(np.square(flat[:, None] - flat[None, :]), axis=-1))


def physical_action_signals(
    eef: np.ndarray,
    pot1: np.ndarray,
    pot2: np.ndarray,
    actions: np.ndarray,
    goals: dict[str, np.ndarray],
    q0: int,
) -> dict[str, np.ndarray]:
    d1 = np.linalg.norm(eef - pot1, axis=1).astype(np.float32)
    d2 = np.linalg.norm(eef - pot2, axis=1).astype(np.float32)
    d1_change = np.zeros(len(eef), dtype=np.float32)
    d1_change[1:] = np.diff(d1)
    chunk_mean = actions.mean(axis=1)
    translation = np.linalg.norm(actions[:, :, :3], axis=2).mean(axis=1)
    rotation = np.linalg.norm(actions[:, :, 3:6], axis=2).mean(axis=1)
    gripper = chunk_mean[:, 6]
    action_pairwise = action_distance(actions)
    action_change = np.zeros(len(actions), dtype=np.float32)
    action_change[1:] = action_pairwise[np.arange(1, len(actions)), np.arange(len(actions) - 1)]
    action_return, _unused_lag, action_anchor = recurrence_signals(
        action_pairwise, action_change, q0
    )
    gripper_sign = np.sign(gripper)
    gripper_flip = np.zeros(len(gripper), dtype=np.float32)
    gripper_flip[1:] = (
        (gripper_sign[1:] != 0)
        & (gripper_sign[:-1] != 0)
        & (gripper_sign[1:] != gripper_sign[:-1])
    )
    return {
        "eef_step_m": _step(eef),
        "eef_pot1_distance_m": d1,
        "eef_pot2_distance_m": d2,
        "pot1_step_m": _step(pot1),
        "pot2_step_m": _step(pot2),
        "pot1_goal_distance_m": np.linalg.norm(pot1 - goals["pot1"], axis=1),
        "pot2_goal_distance_m": np.linalg.norm(pot2 - goals["pot2"], axis=1),
        "eef_pot1_distance_change_m": d1_change,
        "action_translation": translation.astype(np.float32),
        "action_rotation": rotation.astype(np.float32),
        "action_gripper": gripper.astype(np.float32),
        "action_change": action_change,
        "action_recurrence_advantage": action_return,
        "action_anchor_advantage": action_anchor,
        "action_gripper_flip": gripper_flip,
    }


def routing_signals(
    probabilities: np.ndarray, expert_ids: np.ndarray, q0: int
) -> dict[str, np.ndarray]:
    probabilities, _low, _high = normalize_probabilities(probabilities)
    # Average action tokens and denoise steps, retaining the eight layer sites.
    route = probabilities[:, :, :, 1:, :].mean(axis=(2, 3))
    route, _low, _high = normalize_probabilities(route)
    route_distance = pairwise_hellinger(route)
    route_speed = np.zeros(len(route), dtype=np.float32)
    route_speed[1:] = route_distance[
        np.arange(1, len(route)), np.arange(len(route) - 1)
    ]
    route_return, route_lag, route_anchor = recurrence_signals(
        route_distance, route_speed, q0
    )
    entropy = -np.sum(route * np.log(np.maximum(route, 1e-12)), axis=-1)
    entropy = entropy.mean(axis=1) / np.log(route.shape[-1])
    top1 = route.max(axis=-1).mean(axis=1)

    occupancy = selected_expert_occupancy(expert_ids)
    occupancy, _low, _high = normalize_probabilities(occupancy)
    expert_distance = pairwise_hellinger(occupancy)
    expert_speed = np.zeros(len(route), dtype=np.float32)
    expert_speed[1:] = expert_distance[
        np.arange(1, len(route)), np.arange(len(route) - 1)
    ]
    expert_entropy = -np.sum(
        occupancy * np.log(np.maximum(occupancy, 1e-12)), axis=-1
    )
    expert_entropy = expert_entropy.mean(axis=1) / np.log(occupancy.shape[-1])
    expert_top1 = occupancy.max(axis=-1).mean(axis=1)

    layer_speed = np.zeros((len(route), route.shape[1]), dtype=np.float32)
    if len(route) > 1:
        layer_speed[1:] = _hellinger_rows(route[1:], route[:-1])
    synchrony = 1.0 - layer_speed.std(axis=1) / np.maximum(
        layer_speed.mean(axis=1), 1e-6
    )
    synchrony = np.clip(synchrony, -1.0, 1.0)
    return {
        "route_speed": route_speed,
        "route_recurrence_advantage": route_return,
        "route_recurrence_lag_fraction": route_lag,
        "route_anchor_advantage": route_anchor,
        "route_entropy": entropy.astype(np.float32),
        "route_top1_mass": top1.astype(np.float32),
        "expert_speed": expert_speed,
        "expert_entropy": expert_entropy.astype(np.float32),
        "expert_top1_mass": expert_top1.astype(np.float32),
        "route_layer_synchrony": synchrony.astype(np.float32),
    }


def load_long_physical(
    cache_root: pathlib.Path, audit: pd.DataFrame
) -> tuple[dict[int, dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    client = cache_root / LONG_TASK / "right-16x32/client"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    layout = json.loads((client / "sim_layout.json").read_text())
    joints = {row["joint"]: row for row in layout["joints"]}
    slices = {}
    for key, joint_name in (
        ("pot1", "moka_pot_1_joint0"),
        ("pot2", "moka_pot_2_joint0"),
    ):
        start = int(joints[joint_name]["state_lo"])
        slices[key] = slice(start, start + 3)
    physical: dict[int, dict[str, Any]] = {}
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(client / f"episode_{episode:02d}.npz", allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
        physical[episode] = {
            "eef": state[:, :3],
            "pot1": sim[:, slices["pot1"]],
            "pot2": sim[:, slices["pot2"]],
            "actions": actions,
        }
    success_ids = [int(row["episode_index"]) for row in summaries if row["success"]]
    goals = {
        key: np.mean([physical[episode][key][-1] for episode in success_ids], axis=0)
        for key in ("pot1", "pot2")
    }
    long_audit = audit[audit["task"] == LONG_TASK].set_index("episode")
    for episode, tape in physical.items():
        row = long_audit.loc[episode]
        tape["q0"] = int(row["pot2_first_goal_query"])
    return physical, goals, summaries


def sustained_stasis_onset(eef_step: np.ndarray, start: int) -> int | None:
    for onset in range(max(start, 1), len(eef_step) - MIN_STASIS_TRANSITIONS + 1):
        remaining = eef_step[onset + 1 :]
        if len(remaining) < MIN_STASIS_TRANSITIONS:
            continue
        if (
            np.mean(remaining < LOW_STEP_M) >= LOW_STEP_FRACTION
            and np.quantile(remaining, 0.80) < LOW_STEP_P80_M
        ):
            return onset
    return None


def label_long_episodes(
    audit: pd.DataFrame, physical: dict[int, dict[str, Any]]
) -> pd.DataFrame:
    long = audit[audit["task"] == LONG_TASK].copy()
    labels = []
    for _, row in long.iterrows():
        episode = int(row["episode"])
        tape = physical[episode]
        q0 = int(tape["q0"])
        d1 = np.linalg.norm(tape["eef"] - tape["pot1"], axis=1)
        d2 = np.linalg.norm(tape["eef"] - tape["pot2"], axis=1)
        candidates = np.flatnonzero(
            (np.arange(len(d1)) >= q0) & (d1 <= APPROACH_DISTANCE_M)
        )
        approach = int(candidates[0]) if len(candidates) else None
        label = "other_long_failure" if bool(row["failure"]) else "success"
        onset: int | None = approach if not bool(row["failure"]) else None
        if (
            bool(row["failure"])
            and row["long_physical_stage"] == "pot2_only"
            and row["eef_terminal_basin"] == "pot1"
            and approach is not None
        ):
            onset = sustained_stasis_onset(_step(tape["eef"]), approach)
            if onset is not None:
                label = "stagnation_core"
        elif (
            bool(row["failure"])
            and row["long_physical_stage"] == "pot2_only"
            and row["eef_terminal_basin"] == "pot2"
            and approach is not None
        ):
            returned = np.flatnonzero(
                (np.arange(len(d1)) > approach) & (d2 < d1)
            )
            if len(returned):
                onset = int(returned[0])
                label = "active_return"
        cut = q0 + PREFIX_AFTER_POT2
        labels.append(
            {
                "task": LONG_TASK,
                "episode": episode,
                "init_state_id": int(row["init_state_id"]),
                "flow_noise_seed": int(row["flow_noise_seed"]),
                "outcome": row["outcome"],
                "physical_type": label,
                "pot2_anchor_query": q0,
                "pot1_approach_query": approach,
                "physical_onset_query": onset,
                "prospective_cut_query": cut,
                "cut_before_onset": bool(onset is not None and cut < onset),
                "episode_length": int(row["episode_length"]),
            }
        )
    return pd.DataFrame(labels)


def load_tapes(
    cache_root: pathlib.Path,
    physical: dict[int, dict[str, Any]],
    goals: dict[str, np.ndarray],
    summaries: list[dict[str, Any]],
) -> dict[int, EpisodeTape]:
    run = cache_root / LONG_TASK / "right-16x32"
    route = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries])
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    tapes: dict[int, EpisodeTape] = {}
    for index, row in enumerate(summaries):
        episode = int(row["episode_index"])
        start = int(offsets[index])
        stop = start + int(lengths[index])
        raw = np.asarray(route["hb_router_probs"][start:stop], dtype=np.float32)
        ids = np.asarray(route["hb_expert_ids"][start:stop], dtype=np.uint8)
        item = physical[episode]
        q0 = int(item["q0"])
        signals = physical_action_signals(
            item["eef"], item["pot1"], item["pot2"], item["actions"], goals, q0
        )
        signals.update(routing_signals(raw, ids, q0))
        tapes[episode] = EpisodeTape(
            episode=episode,
            init_state_id=int(row["init_state_id"]),
            flow_noise_seed=int(row["flow_noise_seed"]),
            success=bool(row["success"]),
            q0=q0,
            eef=item["eef"],
            pot1=item["pot1"],
            pot2=item["pot2"],
            actions=item["actions"],
            signals=signals,
        )
        if (index + 1) % 64 == 0:
            print(f"loaded compact descriptors: {index + 1}/{len(summaries)}", flush=True)
    return tapes


def summarize_window(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    time = np.linspace(0.0, 1.0, len(values))
    slope = float(np.polyfit(time, values, 1)[0]) if len(values) > 1 else 0.0
    return {
        "last": float(values[-1]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "slope": slope,
    }


def prefix_feature_table(labels: pd.DataFrame, tapes: dict[int, EpisodeTape]) -> pd.DataFrame:
    rows = []
    for _, label in labels.iterrows():
        episode = int(label["episode"])
        tape = tapes[episode]
        start = int(label["pot2_anchor_query"])
        stop = int(label["prospective_cut_query"]) + 1
        if stop > len(tape.eef):
            continue
        row = label.to_dict()
        for signal in ALL_SIGNALS:
            summary = summarize_window(tape.signals[signal][start:stop])
            for statistic, value in summary.items():
                row[f"{signal}__{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def block_columns(frame: pd.DataFrame, signals: tuple[str, ...]) -> list[str]:
    columns = [f"{signal}__{stat}" for signal in signals for stat in SUMMARY_STATS]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return columns


def logo_predictions(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    fold = np.full(len(labels), -1, dtype=np.int64)
    splitter = LeaveOneGroupOut()
    for fold_id, (train, test) in enumerate(splitter.split(features, labels, groups)):
        if len(np.unique(labels[train])) < 2:
            raise ValueError("a leave-one-group-out training fold has one class")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(25, 75)),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed + fold_id,
            ),
        )
        model.fit(features[train], labels[train])
        prediction[test] = model.predict_proba(features[test])[:, 1]
        fold[test] = fold_id
    if np.any(~np.isfinite(prediction)) or np.any(fold < 0):
        raise RuntimeError("incomplete LOGO predictions")
    return prediction, fold


def score_predictions(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(labels, prediction)),
        "average_precision": float(average_precision_score(labels, prediction)),
        "balanced_accuracy_at_0.5": float(
            balanced_accuracy_score(labels, prediction >= 0.5)
        ),
    }


def grouped_auc_delta_ci(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, list[float], int]:
    estimate = float(roc_auc_score(labels, left) - roc_auc_score(labels, right))
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(draws):
        selected = rng.choice(unique, size=len(unique), replace=True)
        sample = np.concatenate([indices[group] for group in selected])
        if len(np.unique(labels[sample])) < 2:
            continue
        bootstrap.append(
            roc_auc_score(labels[sample], left[sample])
            - roc_auc_score(labels[sample], right[sample])
        )
    if not bootstrap:
        return estimate, [float("nan"), float("nan")], 0
    return estimate, list(map(float, np.quantile(bootstrap, [0.025, 0.975]))), len(bootstrap)


def prospective_analysis(
    prefix: pd.DataFrame, bootstrap: int, seed: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    cohort = prefix[
        prefix["physical_type"].isin(["stagnation_core", "active_return"])
        & prefix["cut_before_onset"]
    ].copy()
    labels = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    groups = cohort["init_state_id"].to_numpy(dtype=np.int64)
    blocks = {
        "physical": PHYSICAL_SIGNALS,
        "action": ACTION_SIGNALS,
        "physical_action": PHYSICAL_SIGNALS + ACTION_SIGNALS,
        "routing": ROUTING_SIGNALS,
        "joint": PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS,
    }
    results: dict[str, Any] = {}
    predictions = cohort[
        [
            "task",
            "episode",
            "init_state_id",
            "flow_noise_seed",
            "physical_type",
            "pot2_anchor_query",
            "physical_onset_query",
            "prospective_cut_query",
        ]
    ].copy()
    for block, signals in blocks.items():
        columns = block_columns(cohort, signals)
        prediction, fold = logo_predictions(
            cohort[columns].to_numpy(dtype=np.float64), labels, groups, seed
        )
        predictions[f"prediction_{block}"] = prediction
        predictions[f"fold_{block}"] = fold
        results[block] = {
            "features": len(columns),
            **score_predictions(labels, prediction),
        }
    delta, ci, valid = grouped_auc_delta_ci(
        labels,
        predictions["prediction_joint"].to_numpy(),
        predictions["prediction_physical_action"].to_numpy(),
        groups,
        bootstrap,
        seed + 1000,
    )
    results["routing_increment_over_physical_action"] = {
        "delta_roc_auc": delta,
        "init_cluster_bootstrap_ci95": ci,
        "valid_draws": valid,
    }
    results["cohort"] = {
        "n": len(cohort),
        "active_return": int(labels.sum()),
        "stagnation_core": int(len(labels) - labels.sum()),
        "initial_states": int(len(np.unique(groups))),
        "all_cuts_strictly_before_onset": bool(cohort["cut_before_onset"].all()),
        "minimum_onset_minus_cut_queries": int(
            (cohort["physical_onset_query"] - cohort["prospective_cut_query"]).min()
        ),
        "lead_time_queries_by_type": {
            physical_type: {
                "mean": float(
                    (
                        group["physical_onset_query"]
                        - group["prospective_cut_query"]
                    ).mean()
                ),
                "median": float(
                    (
                        group["physical_onset_query"]
                        - group["prospective_cut_query"]
                    ).median()
                ),
                "minimum": int(
                    (
                        group["physical_onset_query"]
                        - group["prospective_cut_query"]
                    ).min()
                ),
            }
            for physical_type, group in cohort.groupby("physical_type")
        },
        "single_class_initial_states": int(
            np.sum(
                pd.crosstab(cohort["init_state_id"], cohort["physical_type"])
                .gt(0)
                .sum(axis=1)
                == 1
            )
        ),
    }
    results["leave_one_seed_out_sensitivity"] = grouped_sensitivity(
        cohort, "flow_noise_seed", seed + 2000
    )
    table = pd.crosstab(cohort["init_state_id"], cohort["physical_type"])
    matched_init = table[(table > 0).all(axis=1)].index
    matched = cohort[cohort["init_state_id"].isin(matched_init)]
    results["matched_initial_states_sensitivity"] = {
        "cohort_n": int(len(matched)),
        "active_return": int((matched["physical_type"] == "active_return").sum()),
        "initial_states": int(len(matched_init)),
        **grouped_sensitivity(matched, "init_state_id", seed + 3000),
    }
    return results, predictions


def grouped_sensitivity(
    cohort: pd.DataFrame, group_column: str, seed: int
) -> dict[str, Any]:
    labels = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    groups = cohort[group_column].to_numpy()
    blocks = {
        "physical": PHYSICAL_SIGNALS,
        "action": ACTION_SIGNALS,
        "physical_action": PHYSICAL_SIGNALS + ACTION_SIGNALS,
        "routing": ROUTING_SIGNALS,
        "joint": PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS,
    }
    result: dict[str, Any] = {"groups": int(len(np.unique(groups)))}
    for block, signals in blocks.items():
        columns = block_columns(cohort, signals)
        prediction, _fold = logo_predictions(
            cohort[columns].to_numpy(dtype=np.float64), labels, groups, seed
        )
        result[block] = score_predictions(labels, prediction)
    result["routing_increment_roc_auc"] = float(
        result["joint"]["roc_auc"] - result["physical_action"]["roc_auc"]
    )
    return result


def _aligned_values(values: np.ndarray, onset: int) -> np.ndarray:
    aligned = np.full(len(EVENT_OFFSETS), np.nan, dtype=np.float32)
    query = onset + EVENT_OFFSETS
    valid = (query >= 0) & (query < len(values))
    aligned[valid] = values[query[valid]]
    return aligned


def _cluster_bootstrap_interval(
    values: np.ndarray, draws: int, seed: int
) -> tuple[float, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    estimate = float(values.mean())
    if len(values) == 1:
        return estimate, [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[selected].mean(axis=1)
    return estimate, list(map(float, np.quantile(sampled, [0.025, 0.975])))


def event_aligned_analysis(
    labels: pd.DataFrame,
    tapes: dict[int, EpisodeTape],
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    selected = labels[
        labels["physical_type"].isin(
            ["stagnation_core", "active_return", "success"]
        )
        & labels["physical_onset_query"].notna()
    ].copy()
    curves: dict[str, np.ndarray] = {}
    episode_rows: list[dict[str, Any]] = []
    reference_mask = (EVENT_OFFSETS >= -10) & (EVENT_OFFSETS <= -7)
    windows = {
        "lead_-6_-2": (EVENT_OFFSETS >= -6) & (EVENT_OFFSETS <= -2),
        "sync_-1_+1": (EVENT_OFFSETS >= -1) & (EVENT_OFFSETS <= 1),
        "after_+2_+6": (EVENT_OFFSETS >= 2) & (EVENT_OFFSETS <= 6),
    }
    for signal in ALL_SIGNALS:
        aligned = np.stack(
            [
                _aligned_values(
                    tapes[int(row["episode"])].signals[signal],
                    int(row["physical_onset_query"]),
                )
                for _, row in selected.iterrows()
            ]
        )
        reference = np.nanmean(aligned[:, reference_mask], axis=1)
        scale = float(np.nanstd(reference))
        if not np.isfinite(scale) or scale < 1e-7:
            scale = float(np.nanstd(aligned))
        scale = max(scale, 1e-7)
        normalized = (aligned - reference[:, None]) / scale
        for physical_type in ("success", "stagnation_core", "active_return"):
            mask = selected["physical_type"].to_numpy() == physical_type
            curves[f"{physical_type}__{signal}"] = normalized[mask].astype(np.float32)
        for row_index, (_, row) in enumerate(selected.iterrows()):
            for window_name, window_mask in windows.items():
                window_values = normalized[row_index, window_mask]
                value = (
                    float(np.nanmean(window_values))
                    if np.any(np.isfinite(window_values))
                    else float("nan")
                )
                episode_rows.append(
                    {
                        "task": LONG_TASK,
                        "episode": int(row["episode"]),
                        "init_state_id": int(row["init_state_id"]),
                        "physical_type": row["physical_type"],
                        "signal": signal,
                        "window": window_name,
                        "reference_standardized_change": value,
                    }
                )
    episode_windows = pd.DataFrame(episode_rows)
    comparisons = (
        ("active_return", "stagnation_core"),
        ("stagnation_core", "success"),
        ("active_return", "success"),
    )
    effect_rows: list[dict[str, Any]] = []
    for comparison_index, (left, right) in enumerate(comparisons):
        for signal_index, signal in enumerate(ALL_SIGNALS):
            earliest = "none"
            local_rows = []
            for window_index, window in enumerate(windows):
                subset = episode_windows[
                    (episode_windows["signal"] == signal)
                    & (episode_windows["window"] == window)
                    & episode_windows["physical_type"].isin([left, right])
                ]
                contrasts = []
                for _init, group in subset.groupby("init_state_id"):
                    if {left, right}.issubset(set(group["physical_type"])):
                        left_mean = group.loc[
                            group["physical_type"] == left,
                            "reference_standardized_change",
                        ].mean()
                        right_mean = group.loc[
                            group["physical_type"] == right,
                            "reference_standardized_change",
                        ].mean()
                        if np.isfinite(left_mean) and np.isfinite(right_mean):
                            contrasts.append(left_mean - right_mean)
                if contrasts:
                    effect, ci = _cluster_bootstrap_interval(
                        np.asarray(contrasts),
                        bootstrap,
                        seed
                        + comparison_index * 10000
                        + signal_index * 100
                        + window_index,
                    )
                else:
                    effect, ci = float("nan"), [float("nan"), float("nan")]
                separated = bool(
                    np.isfinite(effect)
                    and abs(effect) >= 0.20
                    and ((ci[0] > 0.0) or (ci[1] < 0.0))
                )
                if earliest == "none" and separated:
                    earliest = window
                local_rows.append(
                    {
                        "comparison": f"{left}_minus_{right}",
                        "signal": signal,
                        "signal_block": (
                            "physical"
                            if signal in PHYSICAL_SIGNALS
                            else "action"
                            if signal in ACTION_SIGNALS
                            else "routing"
                        ),
                        "window": window,
                        "matched_initial_states": len(contrasts),
                        "effect_sd": effect,
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "separated": separated,
                    }
                )
            for item in local_rows:
                item["earliest_separation"] = earliest
                effect_rows.append(item)
    effects = pd.DataFrame(effect_rows)
    chronology = {}
    main = effects[
        effects["comparison"] == "active_return_minus_stagnation_core"
    ]
    for block in ("physical", "action", "routing"):
        chronology[block] = {
            signal: str(group["earliest_separation"].iloc[0])
            for signal, group in main[main["signal_block"] == block].groupby("signal")
        }
    summary = {
        "alignment": {
            "stagnation_core": "query before the sustained low-EEF transition run",
            "active_return": "first observed query where EEF is again closer to pot2",
            "success": "first query inside the 13 cm pot1 approach neighborhood",
        },
        "windows": {
            "reference": [-10, -7],
            "lead": [-6, -2],
            "sync": [-1, 1],
            "after": [2, 6],
        },
        "effect": "within-episode change from reference; active-minus-stagnation contrast averaged equally across matched init states",
        "chronology_active_return_vs_stagnation": chronology,
    }
    return effects, summary, curves


def make_event_plot(curves: dict[str, np.ndarray], out: pathlib.Path) -> None:
    shown = (
        "eef_step_m",
        "action_change",
        "action_recurrence_advantage",
        "route_speed",
        "route_recurrence_advantage",
        "expert_speed",
    )
    colors = {
        "success": "#54A24B",
        "stagnation_core": "#4C78A8",
        "active_return": "#E45756",
    }
    display = {
        "success": "success approach",
        "stagnation_core": "suffix-low-motion stasis",
        "active_return": "EEF return-to-pot2-side",
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex=True, constrained_layout=True)
    for ax, signal in zip(axes.flat, shown):
        for physical_type in ("success", "stagnation_core", "active_return"):
            values = curves[f"{physical_type}__{signal}"]
            mean = np.nanmean(values, axis=0)
            sem = np.nanstd(values, axis=0) / np.sqrt(
                np.maximum(np.sum(np.isfinite(values), axis=0), 1)
            )
            ax.plot(
                EVENT_OFFSETS,
                mean,
                color=colors[physical_type],
                label=display[physical_type],
            )
            ax.fill_between(
                EVENT_OFFSETS,
                mean - 1.96 * sem,
                mean + 1.96 * sem,
                color=colors[physical_type],
                alpha=0.12,
            )
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--")
        ax.axhline(0, color="#999999", linewidth=0.7)
        ax.set_title(signal)
        ax.set_xlabel("query offset from physical event proxy")
        ax.set_ylabel("change from [-10,-7] (SD)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _third_means(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    cuts = np.array_split(np.arange(values.shape[1]), 3)
    return tuple(np.nanmean(values[:, cut], axis=1) for cut in cuts)  # type: ignore[return-value]


def all_outcome_route_descriptors() -> pd.DataFrame:
    dynamics = pd.read_csv(DYNAMICS_CSV)
    event = np.load(EVENT_FEATURES_NPZ, allow_pickle=False)
    if len(dynamics) != len(event["soft_speed"]):
        raise ValueError("route event row alignment drifted")
    soft = np.asarray(event["soft_speed"], dtype=np.float64)
    hard = np.asarray(event["hard_speed"], dtype=np.float64)
    soft /= np.maximum(np.median(soft, axis=1, keepdims=True), 1e-8)
    soft_thirds = _third_means(soft)
    hard_overlap_thirds = _third_means(1.0 - hard)

    manifest = json.loads((FEATURE_CACHE / "manifest.json").read_text())
    item = manifest["arrays"]["feature_primary_expert_occupancy"]
    sqrt_occupancy = np.load(FEATURE_CACHE / item["file"], mmap_mode="r")
    occupancy = np.square(np.asarray(sqrt_occupancy, dtype=np.float32)).reshape(
        len(dynamics), 10, 8, 32
    )
    occupancy /= np.maximum(occupancy.sum(axis=-1, keepdims=True), 1e-8)
    entropy = -np.sum(
        occupancy * np.log(np.maximum(occupancy, 1e-12)), axis=-1
    ).mean(axis=2) / np.log(32.0)
    entropy_thirds = _third_means(entropy)
    result = dynamics[
        [
            "task",
            "episode",
            "init_state_id",
            "flow_noise_seed",
            "episode_length",
            "failure",
            "consensus_votes",
        ]
    ].copy()
    for prefix, thirds in (
        ("route_speed_norm", soft_thirds),
        ("hard_expert_overlap", hard_overlap_thirds),
        ("hard_expert_entropy", entropy_thirds),
    ):
        for phase, values in zip(("early", "middle", "late"), thirds):
            result[f"{prefix}_{phase}"] = values
    return result


def failure_action_speed(cache_root: pathlib.Path, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        client = cache_root / str(row["task"]) / "right-16x32/client"
        with np.load(
            client / f"episode_{int(row['episode']):02d}.npz", allow_pickle=False
        ) as archive:
            actions = np.asarray(archive["actions"], dtype=np.float32)
        pairwise = action_distance(actions)
        speed = pairwise[np.arange(1, len(actions)), np.arange(len(actions) - 1)]
        target = np.linspace(0.0, 1.0, 9)
        source = np.linspace(0.0, 1.0, len(speed))
        normalized = np.interp(target, source, speed)
        normalized /= max(float(np.median(normalized)), 1e-8)
        early, middle, late = (float(part.mean()) for part in np.array_split(normalized, 3))
        rows.append(
            {
                "task": row["task"],
                "episode": int(row["episode"]),
                "action_speed_norm_early": early,
                "action_speed_norm_middle": middle,
                "action_speed_norm_late": late,
            }
        )
    return pd.DataFrame(rows)


def failure_pattern_analysis(
    cache_root: pathlib.Path, event_audit_csv: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    audit = pd.read_csv(event_audit_csv)
    routes = all_outcome_route_descriptors()
    failure_routes = routes[routes["failure"]].copy()
    frame = audit.merge(
        failure_routes,
        on=[
            "task",
            "episode",
            "init_state_id",
            "flow_noise_seed",
            "episode_length",
        ],
        how="left",
        validate="one_to_one",
    )
    if frame["route_speed_norm_early"].isna().any():
        raise ValueError("failure taxonomy/route merge incomplete")
    frame = frame.merge(
        failure_action_speed(cache_root, frame),
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    )
    descriptor_columns = [
        column
        for column in frame.columns
        if column.startswith(
            ("route_speed_norm_", "hard_expert_", "action_speed_norm_")
        )
    ]
    grouped = (
        frame.groupby(["task", "primary_physical_pattern"], as_index=False)
        .agg(
            n=("episode", "size"),
            **{column: (column, "median") for column in descriptor_columns},
        )
        .sort_values(["task", "n"], ascending=[True, False])
    )

    isolated = routes[
        routes["failure"]
        & (routes["consensus_votes"] < 2)
        & (routes["task"] != LONG_TASK)
    ].copy()
    percentile_metrics = [
        "route_speed_norm_early",
        "route_speed_norm_middle",
        "route_speed_norm_late",
        "hard_expert_overlap_late",
        "hard_expert_entropy_late",
    ]
    isolated_rows = []
    for _, failure in isolated.iterrows():
        success = routes[
            (~routes["failure"])
            & (routes["task"] == failure["task"])
            & (routes["init_state_id"] == failure["init_state_id"])
        ]
        row = failure.to_dict()
        row["same_task_init_successes"] = int(len(success))
        for metric in percentile_metrics:
            row[f"{metric}_success_percentile"] = float(
                np.mean(success[metric].to_numpy() <= float(failure[metric]))
            )
        isolated_rows.append(row)
    isolated_frame = pd.DataFrame(isolated_rows)
    summary = {
        "failures": int(len(frame)),
        "patterns": int(len(grouped)),
        "isolated_other_task_failures": int(len(isolated_frame)),
        "descriptor_scope": "full episode, relative phase; descriptive and post-hoc",
    }
    return frame, grouped, isolated_frame, summary


def save_descriptor_cache(
    tapes: dict[int, EpisodeTape], path: pathlib.Path
) -> None:
    episodes = np.asarray(sorted(tapes), dtype=np.int64)
    lengths = np.asarray([len(tapes[int(ep)].eef) for ep in episodes], dtype=np.int16)
    maximum = int(lengths.max())
    arrays: dict[str, np.ndarray] = {
        "episode": episodes,
        "length": lengths,
        "q0": np.asarray([tapes[int(ep)].q0 for ep in episodes], dtype=np.int16),
    }
    for signal in ALL_SIGNALS:
        values = np.full((len(episodes), maximum), np.nan, dtype=np.float32)
        for row, episode in enumerate(episodes):
            tape = tapes[int(episode)]
            values[row, : len(tape.eef)] = tape.signals[signal]
        arrays[signal] = values
    np.savez_compressed(path, **arrays)


def make_primary_plot(results: dict[str, Any], out: pathlib.Path) -> None:
    names = ["physical", "action", "physical_action", "routing", "joint"]
    labels = ["physical", "action", "phys+action", "routing", "joint"]
    values = [results[name]["roc_auc"] for name in names]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#72B7B2", "#B279A2", "#54A24B"])
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.set_ylim(0.35, 1.0)
    ax.set_ylabel("Leave-one-init-out ROC AUC")
    ax.set_title("Pre-onset EEF return-to-pot2-side vs stasis proxy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def render_report(
    labels: pd.DataFrame,
    results: dict[str, Any],
    event_effects: pd.DataFrame,
    event_summary: dict[str, Any],
    pattern_groups: pd.DataFrame,
    isolated: pd.DataFrame,
) -> str:
    cohort = results["cohort"]
    inc = results["routing_increment_over_physical_action"]
    rows = []
    for name, label in (
        ("physical", "physical"),
        ("action", "action"),
        ("physical_action", "physical + action"),
        ("routing", "routing"),
        ("joint", "physical + action + routing"),
    ):
        value = results[name]
        rows.append(
            f"| {label} | {value['roc_auc']:.3f} | {value['average_precision']:.3f} | "
            f"{value['balanced_accuracy_at_0.5']:.3f} | {value['features']} |"
        )
    counts = labels["physical_type"].value_counts()
    stasis_lead = cohort["lead_time_queries_by_type"]["stagnation_core"]
    return_lead = cohort["lead_time_queries_by_type"]["active_return"]
    seed_sensitivity = results["leave_one_seed_out_sensitivity"]
    matched_sensitivity = results["matched_initial_states_sensitivity"]
    sensitivity_rows = []
    for name, label in (
        ("physical_action", "physical + action"),
        ("routing", "routing"),
        ("joint", "joint"),
    ):
        sensitivity_rows.append(
            f"| {label} | {results[name]['roc_auc']:.3f} | "
            f"{seed_sensitivity[name]['roc_auc']:.3f} | "
            f"{matched_sensitivity[name]['roc_auc']:.3f} |"
        )

    main_event = event_effects[
        event_effects["comparison"] == "active_return_minus_stagnation_core"
    ]
    timing_rows = []
    for signal in ROUTING_SIGNALS:
        subset = main_event[main_event["signal"] == signal]
        effects = {
            row["window"]: row for _, row in subset.iterrows()
        }
        timing_rows.append(
            f"| {signal} | {subset['earliest_separation'].iloc[0]} | "
            f"{effects['lead_-6_-2']['effect_sd']:+.2f} | "
            f"{effects['sync_-1_+1']['effect_sd']:+.2f} | "
            f"{effects['after_+2_+6']['effect_sd']:+.2f} |"
        )
    earliest_counts = (
        main_event.drop_duplicates("signal")
        .groupby(["signal_block", "earliest_separation"])
        .size()
    )
    physical_lead_count = int(earliest_counts.get(("physical", "lead_-6_-2"), 0))
    action_lead_count = int(earliest_counts.get(("action", "lead_-6_-2"), 0))

    wanted_patterns = {
        "bowl_transport_loss_proxy",
        "drawer_subgoal_regression",
        "stalled_at_unfinished_pot1",
        "return_to_completed_pot2_proxy",
        "ramekin_displacement_interference_proxy",
        "bowl_never_transported",
    }
    pattern_rows = []
    pattern_display = {
        "return_to_completed_pot2_proxy": "EEF return-to-pot2-side proxy",
        "ramekin_displacement_interference_proxy": "ramekin-displacement outlier",
        "bowl_never_transported": "no-transport-proxy-detected",
    }
    for _, row in pattern_groups[
        pattern_groups["primary_physical_pattern"].isin(wanted_patterns)
    ].iterrows():
        pattern_rows.append(
            f"| {str(row['task']).split('/')[-1][:22]} | "
            f"{pattern_display.get(row['primary_physical_pattern'], row['primary_physical_pattern'])} | {int(row['n'])} | "
            f"{row['route_speed_norm_early']:.2f}/{row['route_speed_norm_middle']:.2f}/{row['route_speed_norm_late']:.2f} | "
            f"{row['hard_expert_overlap_late']:.3f} | "
            f"{row['hard_expert_entropy_late']:.3f} | "
            f"{row['action_speed_norm_late']:.2f} |"
        )
    isolated_rows = []
    for _, row in isolated.iterrows():
        isolated_rows.append(
            f"| {str(row['task']).split('/')[-1][:24]} | {int(row['episode'])} | "
            f"{int(row['same_task_init_successes'])} | "
            f"{row['route_speed_norm_late_success_percentile']:.2f} | "
            f"{row['hard_expert_overlap_late_success_percentile']:.2f} | "
            f"{row['hard_expert_entropy_late_success_percentile']:.2f} |"
        )
    return f"""# 失败类型的 MoE 时序信号

## 主结论（选择性 fixed-prefix case-control）

**在这套由未来终局筛出的选择性协议下，没有看到 routing 超过 physical+action 的增量；这不能泛化成“routing 普遍没有信息”。**

主问题只比较同一 long moka-pot 任务中的两种物理终局：宽松的 suffix-low-motion stasis proxy 是完成 pot 2、转向 pot 1 后进入持续低 EEF 运动；EEF return-to-pot2-side 是靠近 pot 1 后，EEF 又回到 pot 2 一侧。标签和 onset 只由机器人/物体轨迹定义，不使用路由投票。内部 CSV 为兼容已有代码仍保留 `stagnation_core` / `active_return` 键，显示名不把 EEF 回返解释成策略或 controller 回返。

所有输入统一截到 pot 2 首次到位后的第 {PREFIX_AFTER_POT2} 个 query。最终纳入 {cohort['n']} 条（stasis proxy {cohort['stagnation_core']}，EEF return-to-pot2-side {cohort['active_return']}，{cohort['initial_states']} 个 init）；每条 cut 都早于 query-boundary onset，最近相隔 {cohort['minimum_onset_minus_cut_queries']} 个 query。模型按 init 留一，held-out init 未参与该折的 scaler 或 classifier 拟合。

| feature block | ROC AUC | AP | balanced acc @ 0.5 | dims |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

在 physical+action 上加入 routing 的 OOF AUC 增量是 {inc['delta_roc_auc']:+.3f}，按 init 聚类 bootstrap 95% CI 为 [{inc['init_cluster_bootstrap_ci95'][0]:+.3f}, {inc['init_cluster_bootstrap_ci95'][1]:+.3f}]。因此只能说“此选择性协议下未见增量”，不能说 routing 产生负作用，也不能说它在其他任务、风险集或时间点普遍无信息。

## Fixed-prefix 方法风险

没有发现直接把尾部路由放进特征的硬泄漏：统计只取 `q0:q0+8`，recurrence 只查当前和过去，imputer/scaler/model 都在训练折内拟合，元数据和 onset 没有进模型。但独立方法复核指出以下限制：

1. pot1/pot2 success-goal reference 由全部 296 条成功 rollout 的终点构造，包含 held-out init；因此上游 `q0`、goal-distance 特征和物理标签是 transductive 的。真正 LOGO 需要环境真值 goal 或每折重算 reference。
2. cohort 由未来筛选：成功与 54 条 other-long failure 被排除，线上在 cut 时并不知道 rollout 会落入这两类。这里是前缀上的回顾性 case-control，不是全体 rollout 在线预警。
3. lead time 严重不匹配：stasis proxy 的 onset-cut 均值/中位数为 {stasis_lead['mean']:.2f}/{stasis_lead['median']:.0f} queries；EEF return-to-pot2-side 为 {return_lead['mean']:.2f}/{return_lead['median']:.0f}，最小仍为 {return_lead['minimum']}。两类不是同一预测距离上的风险集。
4. 13 个 init 中 {cohort['single_class_initial_states']} 个只有一个类别；joint 有 100 维但只有 {cohort['n']} 条。即使 LOGO 阻止精确 init 记忆，小组数、类别组成和几何偏移仍限制功效。
5. 最短 1-query 间隔只表示在缓存的 query 边界尚未观察到 onset；cut 处生成的 action chunk 可能正造成下一边界的低运动，不能声称 chunk 内物理事件前预警。

## 稳定性审计

| feature block | leave-one-init-out | leave-one-seed-out | leave-one-init-out, 仅双方都有样本的 init |
|---|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

按 seed 留一仍会在训练中看到同一 init 的其他 seed；它的较高分数不能替代跨 init 验证。双方都有样本的 7 个 init 上共 {matched_sensitivity['cohort_n']} 条，跨 init 结果仍低，说明主表的负结果不是只由单标签 init 造成。两种划分中 routing 对 physical+action 的 AUC 增量分别为 {results['routing_increment_over_physical_action']['delta_roc_auc']:+.3f} 和 {seed_sensitivity['routing_increment_roc_auc']:+.3f}。

## 相对物理 onset 的时序

每条曲线先减去本 episode 的 `[-10,-7]` reference，再按 pooled reference SD 标准化；表内是 matched-init 的 EEF return-to-pot2-side 减 stasis-proxy 动态差。`lead`=`[-6,-2]`，`sync`=`[-1,+1]`，`after`=`[+2,+6]`。只有 |effect|≥0.2 且 init-bootstrap CI 不跨 0 才标为 separation。

| routing signal | earliest separation | lead | sync | after |
|---|---|---:|---:|---:|
{chr(10).join(timing_rows)}

停滞 onset 定义在低运动 transition 开始前的 query，因此 onset query 的 route/action 是对下一 chunk 的计划；EEF return onset 则是第一次已经观察到 EEF 重新更靠 pot 2 的 query，因此同 query routing 同时含有物理状态读出。成功曲线按首次进入 pot 1 的 13 cm 邻域对齐，只是行为参照，不把它称作失败 onset。

两种事件的 lead time 和物理定义并不相同，因此这张对齐表只能描述各自 onset 附近的形态，不能把大量 `lead` 自动解释成共同早期预警。实际上 physical/action 也分别有 {physical_lead_count}/{len(PHYSICAL_SIGNALS)} 和 {action_lead_count}/{len(ACTION_SIGNALS)} 个信号在 lead 窗分开，说明 routing 很可能同时跟随已不同的物理阶段和待执行动作。局部 lead 最多是下一动作前验候选；sync/after 更像状态读出。结合统一 cut 上不跨 init 泛化，不能升级为通用 MoE trap 机制。

## 物理类型覆盖

- suffix-low-motion stasis proxy（内部键 `stagnation_core`）: {int(counts.get('stagnation_core', 0))}。完成 pot 2、接近 pot 1，随后剩余 EEF transitions 中至少 80% 小于 1 cm/query，且 p80 小于 1.5 cm/query。独立 event audit 的严格交集是 109 条 `stalled_at_unfinished_pot1`；123 不能和 109 混称同一个 core。
- EEF return-to-pot2-side（内部键 `active_return`）: {int(counts.get('active_return', 0))}。完成 pot 2、进入 pot 1 的 13 cm 邻域，之后首次重新变成离 pot 2 更近；它只确认 EEF 空间回返，不确认内部策略、controller 或 routing reset。
- `success`: {int(counts.get('success', 0))}。这里只作为下一阶段的事件对照，不参与上面的二分类主结果。
- `other_long_failure`: {int(counts.get('other_long_failure', 0))}。不满足上述干净定义，主结果不强行归类。

## 全 307 失败的描述表

下面使用完整 episode 的 10-bin 相对相位，只回答“终局类型伴随什么动态”，不能用于 pre-onset 预测。route/action speed 除以各 episode 中位数；hard overlap 是相邻相位 hard-expert occupancy 的 `1-Hellinger`；entropy 是归一化 hard occupancy entropy。

| task (short) | physical pattern | n | route speed E/M/L | hard overlap L | hard entropy L | action speed L |
|---|---|---:|---|---:|---:|---:|
{chr(10).join(pattern_rows)}

这一层的类型来自独立物理账本：top-drawer 分 transport-loss proxy / drawer regression，long 分 stall / EEF return-to-pot2-side，ramekin 只称 displacement outlier，stove 分 no-transport-proxy-detected / transport-loss proxy。缓存没有 RGB、contact 或 force，不能把这些代理升级为真实碰撞、抓取、掉落或视觉混淆。小组只报中位数，不做 AUC 或机制宣称。

## 5 条跨任务孤立失败

每条只和同 task、同 init 的成功 sibling 比整段路由百分位；1 表示高于所有 sibling。

| task (short) | episode | success siblings | late route-speed pct | late hard-overlap pct | late hard-entropy pct |
|---|---:|---:|---:|---:|---:|
{chr(10).join(isolated_rows)}

## 如何解释

- `physical` 是当前状态读出；`action` 是已经生成但尚未执行完的下一 chunk 计划。
- `routing` 若只在 onset 同步或之后分开，最多说明内部状态读出了已发生的物理变化；lead 分开也必须先排除 lead-time、物理状态和 action 差异。
- 固定 cut 上 routing 超过 physical+action，才是“对未来恢复动作有额外预后信息”的候选证据；即使如此仍只能叫预测关联。
- 机制结论还需要对 route/expert 或 action 做受控干预，本报告不作因果宣称。

## 产物

- `episode_results.csv`: 逐 episode 物理类型、onset、cut 和 OOF 分数。
- `prefix_features.csv.gz`: cut 前压缩特征，便于复核；不含原始路由张量。
- `query_descriptors.npz`: 每 query 的标量 physical/action/routing 描述符。
- `event_window_effects.csv` / `event_aligned.png`: onset 前、同步、后的 matched-init 动态差。
- `failure_pattern_descriptors.csv` / `failure_pattern_summary.csv`: 全 307 失败的物理 taxonomy 与整段动态。
- `isolated_other_task_failures.csv`: 5 条孤立失败相对成功 sibling 的百分位。
- `summary.json`: 计数、协议和指标。
- `prospective_auc.png`: 主结果图。
"""


def run_self_test() -> None:
    values = np.asarray(
        [
            [[0.8, 0.2]],
            [[0.2, 0.8]],
            [[0.79, 0.21]],
        ],
        dtype=np.float32,
    )
    distance = pairwise_hellinger(values)
    speed = np.r_[0.0, distance[1, 0], distance[2, 1]].astype(np.float32)
    advantage, lag, _anchor = recurrence_signals(distance, speed, 0)
    assert distance.shape == (3, 3)
    assert advantage[2] > 0.0 and lag[2] == 1.0
    steps = np.asarray([0.0, 0.03, 0.02, 0.001, 0.002, 0.003, 0.001])
    assert sustained_stasis_onset(steps, 2) == 2
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(6), 8)
    labels = np.tile(np.r_[np.zeros(4), np.ones(4)], 6).astype(int)
    features = labels[:, None] + rng.normal(0.0, 0.1, size=(len(labels), 2))
    prediction, fold = logo_predictions(features, labels, groups, 0)
    assert np.all(fold >= 0) and roc_auc_score(labels, prediction) > 0.9
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(args.audit_csv)
    physical, goals, summaries = load_long_physical(args.cache_root, audit)
    labels = label_long_episodes(audit, physical)
    print(labels["physical_type"].value_counts().to_string(), flush=True)
    tapes = load_tapes(args.cache_root, physical, goals, summaries)
    prefix = prefix_feature_table(labels, tapes)
    results, predictions = prospective_analysis(prefix, args.bootstrap, args.seed)
    event_effects, event_summary, event_curves = event_aligned_analysis(
        labels, tapes, args.bootstrap, args.seed + 5000
    )
    pattern_frame, pattern_groups, isolated, pattern_summary = failure_pattern_analysis(
        args.cache_root, args.event_audit_csv
    )
    episode_results = labels.merge(
        predictions,
        on=[
            "task",
            "episode",
            "init_state_id",
            "flow_noise_seed",
            "physical_type",
            "pot2_anchor_query",
            "physical_onset_query",
            "prospective_cut_query",
        ],
        how="left",
        validate="one_to_one",
    )
    episode_results.to_csv(args.out_dir / "episode_results.csv", index=False)
    prefix.to_csv(args.out_dir / "prefix_features.csv.gz", index=False, compression="gzip")
    save_descriptor_cache(tapes, args.out_dir / "query_descriptors.npz")
    event_effects.to_csv(args.out_dir / "event_window_effects.csv", index=False)
    np.savez_compressed(
        args.out_dir / "event_curves.npz",
        offsets=EVENT_OFFSETS,
        **event_curves,
    )
    pattern_frame.to_csv(
        args.out_dir / "failure_pattern_descriptors.csv.gz",
        index=False,
        compression="gzip",
    )
    pattern_groups.to_csv(args.out_dir / "failure_pattern_summary.csv", index=False)
    isolated.to_csv(args.out_dir / "isolated_other_task_failures.csv", index=False)
    summary = {
        "protocol": {
            "task": LONG_TASK,
            "prefix_after_pot2_queries": PREFIX_AFTER_POT2,
            "event_offsets": EVENT_OFFSETS.tolist(),
            "validation": "LeaveOneGroupOut(init_state_id)",
            "label_source": "physical state only; no routing vote",
            "raw_route_tensor_cached": False,
        },
        "physical_type_counts": {
            str(key): int(value)
            for key, value in labels["physical_type"].value_counts().items()
        },
        "prospective": results,
        "event_aligned": event_summary,
        "failure_patterns": pattern_summary,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    make_primary_plot(results, args.out_dir / "prospective_auc.png")
    make_event_plot(event_curves, args.out_dir / "event_aligned.png")
    (args.out_dir / "report.md").write_text(
        render_report(
            labels,
            results,
            event_effects,
            event_summary,
            pattern_groups,
            isolated,
        )
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
