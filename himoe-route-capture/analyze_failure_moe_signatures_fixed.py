#!/usr/bin/env python3
"""Audited rerun of the fixed-prefix MoE negative result on the long moka-pot task.

The upstream experiment (``analyze_failure_moe_signatures.py``) reported a
negative result: at a cut eight queries after pot 2 first reaches its
success-goal proxy, routing added -0.020 leave-one-init-out ROC AUC on top of
physical+action for discriminating suffix-low-motion stasis from EEF
return-to-pot2-side.  An independent method review listed six defects.  This
script fixes the ones that can be fixed offline and reports what survives:

1. metric audit  - the pooled leave-one-group-out OOF AUC used upstream is
   dominated by *between-init* pairs, so every reported number below 0.5 is a
   pooling artefact rather than an inverted effect.  One-dimensional sentinels
   (lead time, cut index, init code) quantify this.
2. transductive goal reference - the pot1/pot2 success-goal means are refit
   inside every LOGO fold, which also moves ``q0``, the labels and the cohort.
3. cohort selected by the future - success and other_long_failure rollouts are
   added back, as a multiclass and as an online failure-vs-success problem.
4. lead-time mismatch - a lead-time window restriction and a caliper-matched
   subset put both classes at the same prediction distance.
5. token axis - the routing block is split into token 0 (state) and tokens 1-10
   (action) and each is compared against physical+action separately.
6. power - a synthetic injection sweep that moves real routing toward the same
   rollout's routing at an earlier phase (never Gaussian noise) gives the
   minimum detectable routing increment at 80% power.

Everything is offline: no mujoco, no simulator, no RGB.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

HERE = pathlib.Path(__file__).resolve().parent
CACHE_ROOT = HERE.parent / "VLA_MUI_HUB/cache/HiMoE-VLA"
ANALYSIS = HERE / "analysis"
AUDIT_CSV = ANALYSIS / "residual-failure-physical-audit/episode_audit.csv"
OUT_DIR = ANALYSIS / "failure-moe-signatures-fixed"
LONG_TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

PREFIX_AFTER_POT2 = 8
GOAL_DISTANCE_M = 0.05
APPROACH_DISTANCE_M = 0.13
LOW_STEP_M = 0.01
LOW_STEP_P80_M = 0.015
LOW_STEP_FRACTION = 0.80
MIN_STASIS_TRANSITIONS = 4
SEED = 20260829
BOOTSTRAPS = 2000

N_LAYERS = 8
N_DENOISE = 10
N_TOKENS = 11
N_EXPERTS = 32

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
PHYSICAL_NO_GOAL = tuple(s for s in PHYSICAL_SIGNALS if not s.endswith("goal_distance_m"))
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
SUMMARY_STATS = ("last", "mean", "std", "slope")
BLOCKS = {
    "physical": PHYSICAL_SIGNALS,
    "action": ACTION_SIGNALS,
    "physical_action": PHYSICAL_SIGNALS + ACTION_SIGNALS,
    "routing": ROUTING_SIGNALS,
    "joint": PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS,
}


# --------------------------------------------------------------------------
# small numeric helpers (kept identical to the upstream implementation)
# --------------------------------------------------------------------------
def normalize(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid router probabilities")
    return result / mass


def step_norm(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    if len(values) > 1:
        result[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1)
    return result


def pairwise_hellinger(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    root = np.sqrt(np.maximum(values, 0.0))
    distance = np.sqrt(
        0.5 * np.sum(np.square(root[:, None, ...] - root[None, :, ...]), axis=-1)
    )
    if distance.ndim > 2:
        distance = distance.mean(axis=tuple(range(2, distance.ndim)))
    return distance.astype(np.float32)


def hellinger_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(
        0.5
        * np.sum(
            np.square(np.sqrt(np.maximum(left, 0.0)) - np.sqrt(np.maximum(right, 0.0))),
            axis=-1,
        )
    )


def causal_scale(speed: np.ndarray) -> np.ndarray:
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
    scale = causal_scale(speed)
    for query in range(2, n):
        older = distance[query, : query - 1]
        nearest = int(np.argmin(older))
        advantage[query] = np.clip(
            (distance[query, query - 1] - older[nearest]) / scale[query], -10.0, 10.0
        )
        lag[query] = (query - nearest) / query
        if query > anchor:
            anchor_advantage[query] = np.clip(
                (distance[query, query - 1] - distance[query, anchor]) / scale[query],
                -10.0,
                10.0,
            )
    return advantage, lag, anchor_advantage


def anchor_advantage_only(
    distance: np.ndarray, scale: np.ndarray, anchor: int
) -> np.ndarray:
    """Closed form of the anchor term so ``q0`` can be varied per fold cheaply."""
    n = len(scale)
    result = np.zeros(n, dtype=np.float32)
    for query in range(max(2, anchor + 1), n):
        result[query] = np.clip(
            (distance[query, query - 1] - distance[query, anchor]) / scale[query],
            -10.0,
            10.0,
        )
    return result


def action_distance(actions: np.ndarray) -> np.ndarray:
    scale = np.asarray([0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 1.0])
    normalized = actions / scale[None, None, :]
    flat = normalized.reshape(len(actions), -1)
    return np.sqrt(np.mean(np.square(flat[:, None] - flat[None, :]), axis=-1))


def summarize_window(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    time_axis = np.linspace(0.0, 1.0, len(values))
    slope = float(np.polyfit(time_axis, values, 1)[0]) if len(values) > 1 else 0.0
    return {
        "last": float(values[-1]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "slope": slope,
    }


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_client(cache_root: pathlib.Path) -> tuple[dict[int, dict[str, Any]], list[dict]]:
    client = cache_root / LONG_TASK / "right-16x32/client"
    summaries = sorted(
        json.loads((client / "summaries.json").read_text()),
        key=lambda row: int(row["episode_index"]),
    )
    layout = json.loads((client / "sim_layout.json").read_text())
    joints = {row["joint"]: row for row in layout["joints"]}
    slices = {}
    for key, joint_name in (("pot1", "moka_pot_1_joint0"), ("pot2", "moka_pot_2_joint0")):
        lo = int(joints[joint_name]["state_lo"])
        slices[key] = slice(lo, lo + 3)
    physical: dict[int, dict[str, Any]] = {}
    for row in summaries:
        episode = int(row["episode_index"])
        with np.load(client / f"episode_{episode:02d}.npz", allow_pickle=False) as archive:
            state = np.asarray(archive["state"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
            sim = np.asarray(archive["sim_state"], dtype=np.float32)
        physical[episode] = {
            "eef": state[:, :3],
            "pot1": sim[:, slices["pot1"]].astype(np.float64),
            "pot2": sim[:, slices["pot2"]].astype(np.float64),
            "actions": actions,
            "success": bool(row["success"]),
            "init_state_id": int(row["init_state_id"]),
            "flow_noise_seed": int(row["flow_noise_seed"]),
            "length": int(row["inference_calls"]),
        }
    return physical, summaries


def reduce_routes(
    cache_root: pathlib.Path, summaries: list[dict], block: int = 1024
) -> dict[str, np.ndarray]:
    """Per-query [layer, expert] route distributions and hard occupancies.

    Four variants: soft router probabilities and selected-expert occupancies,
    each for the state token (index 0) and for the ten action tokens (1..10).
    """
    run = cache_root / LONG_TASK / "right-16x32"
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    total = int(store["hb_router_probs"].shape[0])
    out = {
        name: np.zeros((total, N_LAYERS, N_EXPERTS), dtype=np.float32)
        for name in ("route_state", "route_action", "occ_state", "occ_action")
    }
    for start in range(0, total, block):
        stop = min(start + block, total)
        probs = np.asarray(store["hb_router_probs"][start:stop], dtype=np.float32)
        ids = np.asarray(store["hb_expert_ids"][start:stop], dtype=np.int64)
        probs = normalize(probs)
        out["route_state"][start:stop] = normalize(probs[:, :, :, 0, :].mean(axis=2))
        out["route_action"][start:stop] = normalize(
            probs[:, :, :, 1:, :].mean(axis=(2, 3))
        )
        for name, token in (("occ_state", slice(0, 1)), ("occ_action", slice(1, N_TOKENS))):
            selected = ids[:, :, :, token, :]
            flat = selected.reshape(len(selected) * N_LAYERS, -1)
            occupancy = np.zeros((len(flat), N_EXPERTS), dtype=np.float32)
            rows = np.repeat(np.arange(len(flat)), flat.shape[1])
            np.add.at(occupancy, (rows, flat.reshape(-1)), 1.0)
            occupancy /= float(flat.shape[1])
            out[name][start:stop] = normalize(
                occupancy.reshape(len(selected), N_LAYERS, N_EXPERTS)
            )
        print(f"reduced routes {stop}/{total}", flush=True)
    lengths = np.asarray([int(row["inference_calls"]) for row in summaries])
    out["episode_offset"] = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
    out["episode_length"] = lengths.astype(np.int64)
    out["episode"] = np.asarray(
        [int(row["episode_index"]) for row in summaries], dtype=np.int64
    )
    return out


# --------------------------------------------------------------------------
# per-episode descriptor cache (everything that does not depend on q0/goal)
# --------------------------------------------------------------------------
class EpisodeCache:
    """Query-level descriptors with the q0- and goal-dependent parts factored out."""

    def __init__(self, item: dict[str, Any], route: np.ndarray, occupancy: np.ndarray):
        eef = item["eef"]
        self.n = len(eef)
        self.pot1 = item["pot1"]
        self.pot2 = item["pot2"]
        self.base: dict[str, np.ndarray] = {
            "eef_step_m": step_norm(eef),
            "eef_pot1_distance_m": np.linalg.norm(eef - item["pot1"], axis=1).astype(np.float32),
            "eef_pot2_distance_m": np.linalg.norm(eef - item["pot2"], axis=1).astype(np.float32),
            "pot1_step_m": step_norm(item["pot1"]),
            "pot2_step_m": step_norm(item["pot2"]),
        }
        d1_change = np.zeros(self.n, dtype=np.float32)
        d1_change[1:] = np.diff(self.base["eef_pot1_distance_m"])
        self.base["eef_pot1_distance_change_m"] = d1_change

        actions = item["actions"]
        self.action_dist = action_distance(actions).astype(np.float32)
        change = np.zeros(self.n, dtype=np.float32)
        change[1:] = self.action_dist[np.arange(1, self.n), np.arange(self.n - 1)]
        self.action_scale = causal_scale(change)
        adv, _lag, _anchor = recurrence_signals(self.action_dist, change, self.n + 1)
        chunk_mean = actions.mean(axis=1)
        gripper = chunk_mean[:, 6].astype(np.float32)
        sign = np.sign(gripper)
        flip = np.zeros(self.n, dtype=np.float32)
        flip[1:] = (
            (sign[1:] != 0) & (sign[:-1] != 0) & (sign[1:] != sign[:-1])
        ).astype(np.float32)
        self.base.update(
            {
                "action_translation": np.linalg.norm(actions[:, :, :3], axis=2)
                .mean(axis=1)
                .astype(np.float32),
                "action_rotation": np.linalg.norm(actions[:, :, 3:6], axis=2)
                .mean(axis=1)
                .astype(np.float32),
                "action_gripper": gripper,
                "action_change": change,
                "action_recurrence_advantage": adv,
                "action_gripper_flip": flip,
            }
        )
        self.set_routing(route, occupancy)

    def set_routing(self, route: np.ndarray, occupancy: np.ndarray) -> None:
        self.route_dist = pairwise_hellinger(route)
        speed = np.zeros(self.n, dtype=np.float32)
        speed[1:] = self.route_dist[np.arange(1, self.n), np.arange(self.n - 1)]
        self.route_scale = causal_scale(speed)
        adv, lag, _anchor = recurrence_signals(self.route_dist, speed, self.n + 1)
        entropy = -np.sum(route * np.log(np.maximum(route, 1e-12)), axis=-1)
        entropy = entropy.mean(axis=1) / np.log(route.shape[-1])
        top1 = route.max(axis=-1).mean(axis=1)

        occ_dist = pairwise_hellinger(occupancy)
        expert_speed = np.zeros(self.n, dtype=np.float32)
        expert_speed[1:] = occ_dist[np.arange(1, self.n), np.arange(self.n - 1)]
        expert_entropy = -np.sum(
            occupancy * np.log(np.maximum(occupancy, 1e-12)), axis=-1
        )
        expert_entropy = expert_entropy.mean(axis=1) / np.log(occupancy.shape[-1])
        expert_top1 = occupancy.max(axis=-1).mean(axis=1)

        layer_speed = np.zeros((self.n, route.shape[1]), dtype=np.float32)
        if self.n > 1:
            layer_speed[1:] = hellinger_rows(route[1:], route[:-1])
        synchrony = 1.0 - layer_speed.std(axis=1) / np.maximum(
            layer_speed.mean(axis=1), 1e-6
        )
        self.base.update(
            {
                "route_speed": speed,
                "route_recurrence_advantage": adv,
                "route_recurrence_lag_fraction": lag,
                "route_entropy": entropy.astype(np.float32),
                "route_top1_mass": top1.astype(np.float32),
                "expert_speed": expert_speed,
                "expert_entropy": expert_entropy.astype(np.float32),
                "expert_top1_mass": expert_top1.astype(np.float32),
                "route_layer_synchrony": np.clip(synchrony, -1.0, 1.0).astype(np.float32),
            }
        )

    def window_features(
        self, q0: int, cut: int, goals: dict[str, np.ndarray], signals: tuple[str, ...]
    ) -> dict[str, float]:
        window = slice(q0, cut + 1)
        out: dict[str, float] = {}
        for signal in signals:
            if signal == "pot1_goal_distance_m":
                values = np.linalg.norm(self.pot1[window] - goals["pot1"], axis=1)
            elif signal == "pot2_goal_distance_m":
                values = np.linalg.norm(self.pot2[window] - goals["pot2"], axis=1)
            elif signal == "route_anchor_advantage":
                values = anchor_advantage_only(self.route_dist, self.route_scale, q0)[window]
            elif signal == "action_anchor_advantage":
                values = anchor_advantage_only(self.action_dist, self.action_scale, q0)[window]
            else:
                values = self.base[signal][window]
            for statistic, value in summarize_window(values).items():
                out[f"{signal}__{statistic}"] = value
        return out


# --------------------------------------------------------------------------
# label derivation (reimplements the upstream chain so the goal can be refit)
# --------------------------------------------------------------------------
def goal_reference(
    physical: dict[int, dict[str, Any]], success_ids: list[int]
) -> dict[str, np.ndarray]:
    return {
        key: np.mean([physical[e][key][-1] for e in success_ids], axis=0)
        for key in ("pot1", "pot2")
    }


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


def derive_labels(
    physical: dict[int, dict[str, Any]], goals: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for episode in sorted(physical):
        item = physical[episode]
        d_goal1 = np.linalg.norm(item["pot1"] - goals["pot1"], axis=1)
        d_goal2 = np.linalg.norm(item["pot2"] - goals["pot2"], axis=1)
        reach2 = np.flatnonzero(d_goal2 <= GOAL_DISTANCE_M)
        q0 = int(reach2[0]) if len(reach2) else -1
        placed1 = bool(d_goal1[-1] <= GOAL_DISTANCE_M)
        placed2 = bool(d_goal2[-1] <= GOAL_DISTANCE_M)
        stage = (
            "both_at_goal"
            if placed1 and placed2
            else "pot2_only"
            if placed2
            else "pot1_only"
            if placed1
            else "neither_at_goal"
        )
        eef = item["eef"]
        d1 = np.linalg.norm(eef - item["pot1"], axis=1)
        d2 = np.linalg.norm(eef - item["pot2"], axis=1)
        basin = "pot1" if float(np.linalg.norm(eef[-1] - item["pot1"][-1])) < float(
            np.linalg.norm(eef[-1] - item["pot2"][-1])
        ) else "pot2"
        failure = not item["success"]
        approach = None
        if q0 >= 0:
            candidates = np.flatnonzero((np.arange(len(d1)) >= q0) & (d1 <= APPROACH_DISTANCE_M))
            approach = int(candidates[0]) if len(candidates) else None
        label = "other_long_failure" if failure else "success"
        onset = approach if not failure else None
        if failure and stage == "pot2_only" and basin == "pot1" and approach is not None:
            candidate = sustained_stasis_onset(step_norm(eef), approach)
            if candidate is not None:
                onset, label = candidate, "stagnation_core"
        elif failure and stage == "pot2_only" and basin == "pot2" and approach is not None:
            returned = np.flatnonzero((np.arange(len(d1)) > approach) & (d2 < d1))
            if len(returned):
                onset, label = int(returned[0]), "active_return"
        cut = q0 + PREFIX_AFTER_POT2 if q0 >= 0 else -1
        rows.append(
            {
                "episode": episode,
                "init_state_id": item["init_state_id"],
                "flow_noise_seed": item["flow_noise_seed"],
                "success": item["success"],
                "episode_length": item["length"],
                "long_physical_stage": stage,
                "eef_terminal_basin": basin,
                "physical_type": label,
                "pot2_anchor_query": q0,
                "pot1_approach_query": -1 if approach is None else approach,
                "physical_onset_query": -1 if onset is None else int(onset),
                "prospective_cut_query": cut,
                "cut_before_onset": bool(onset is not None and 0 <= cut < onset),
                "cut_available": bool(q0 >= 0 and cut + 1 <= item["length"]),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC with mid-ranks for ties (matches sklearn's roc_auc_score)."""
    positives = int(np.count_nonzero(y == 1))
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(s)
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def within_group_auc(
    y: np.ndarray, s: np.ndarray, groups: np.ndarray
) -> tuple[float, int, int]:
    """Pair-weighted mean of the within-group AUCs (the fold-comparable metric)."""
    num = den = 0.0
    used = 0
    for group in np.unique(groups):
        mask = groups == group
        yy = y[mask]
        if len(np.unique(yy)) < 2:
            continue
        weight = float(yy.sum() * (1 - yy).sum())
        num += fast_auc(yy, s[mask]) * weight
        den += weight
        used += 1
    return (num / den if den else float("nan")), used, int(den)


def pair_decomposition(y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    total = float(y.sum() * (1 - y).sum())
    within = 0.0
    for group in np.unique(groups):
        mask = groups == group
        within += float(y[mask].sum() * (1 - y[mask]).sum())
    return {
        "positive_negative_pairs": total,
        "within_init_pairs": within,
        "within_init_pair_fraction": within / total if total else float("nan"),
    }


def score_block(y: np.ndarray, s: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    within, folds, pairs = within_group_auc(y, s, groups)
    pooled = float(roc_auc_score(y, s))
    total = float(y.sum() * (1 - y).sum())
    between = (
        (pooled * total - within * pairs) / (total - pairs)
        if total > pairs and np.isfinite(within)
        else float("nan")
    )
    return {
        "pooled_auc": pooled,
        "within_init_auc": within,
        "between_init_auc": float(between),
        "within_init_folds": folds,
        "within_init_pairs": pairs,
    }


def logo_predictions(
    features: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, penalty: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    fold = np.full(len(y), -1, dtype=np.int64)
    for fold_id, (train, test) in enumerate(LeaveOneGroupOut().split(features, y, groups)):
        if len(np.unique(y[train])) < 2:
            raise ValueError("a leave-one-group-out training fold has one class")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(25, 75)),
            LogisticRegression(
                C=penalty,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed + fold_id,
            ),
        )
        model.fit(features[train], y[train])
        prediction[test] = model.predict_proba(features[test])[:, 1]
        fold[test] = fold_id
    if np.any(~np.isfinite(prediction)):
        raise RuntimeError("incomplete LOGO predictions")
    return prediction, fold


def logo_multiclass(
    features: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> np.ndarray:
    classes = np.unique(y)
    prediction = np.full((len(y), len(classes)), np.nan, dtype=np.float64)
    for fold_id, (train, test) in enumerate(LeaveOneGroupOut().split(features, y, groups)):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RobustScaler(quantile_range=(25, 75)),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=3000, random_state=seed + fold_id
            ),
        )
        model.fit(features[train], y[train])
        local = model.predict_proba(features[test])
        for index, label in enumerate(model.classes_):
            prediction[test, int(np.flatnonzero(classes == label)[0])] = local[:, index]
    prediction = np.nan_to_num(prediction, nan=0.0)
    return prediction


def macro_ovr(y: np.ndarray, proba: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    classes = np.unique(y)
    pooled, within = [], []
    for index, label in enumerate(classes):
        binary = (y == label).astype(np.int64)
        pooled.append(roc_auc_score(binary, proba[:, index]))
        value, _folds, _pairs = within_group_auc(binary, proba[:, index], groups)
        if np.isfinite(value):
            within.append(value)
    return {
        "pooled_macro_ovr_auc": float(np.mean(pooled)),
        "within_init_macro_ovr_auc": float(np.mean(within)) if within else float("nan"),
    }


def init_cluster_ci(
    y: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
    metric: str = "within",
) -> dict[str, Any]:
    def statistic(idx: np.ndarray, grp: np.ndarray) -> float:
        if metric == "within":
            a, _f, _p = within_group_auc(y[idx], left[idx], grp)
            b, _f, _p = within_group_auc(y[idx], right[idx], grp)
        else:
            a = fast_auc(y[idx], left[idx])
            b = fast_auc(y[idx], right[idx])
        return a - b

    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    estimate = statistic(np.arange(len(y)), groups)
    rng = np.random.default_rng(seed)
    draws_out = []
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in picked])
        grp = np.concatenate([np.full(len(index[g]), k) for k, g in enumerate(picked)])
        if len(np.unique(y[idx])) < 2:
            continue
        value = statistic(idx, grp)
        if np.isfinite(value):
            draws_out.append(value)
    if not draws_out:
        return {"delta": estimate, "ci95": [float("nan"), float("nan")], "valid_draws": 0}
    low, high = np.quantile(draws_out, [0.025, 0.975])
    return {
        "delta": float(estimate),
        "ci95": [float(low), float(high)],
        "valid_draws": len(draws_out),
    }


# --------------------------------------------------------------------------
# feature table + block evaluation
# --------------------------------------------------------------------------
def build_features(
    labels: pd.DataFrame,
    caches: dict[int, EpisodeCache],
    goals: dict[str, np.ndarray],
    signals: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for _, row in labels.iterrows():
        episode = int(row["episode"])
        cache = caches[episode]
        q0, cut = int(row["pot2_anchor_query"]), int(row["prospective_cut_query"])
        if q0 < 0 or cut + 1 > cache.n:
            continue
        record = row.to_dict()
        record.update(cache.window_features(q0, cut, goals, signals))
        rows.append(record)
    return pd.DataFrame(rows)


def block_columns(frame: pd.DataFrame, signals: tuple[str, ...]) -> list[str]:
    columns = [f"{s}__{stat}" for s in signals for stat in SUMMARY_STATS]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    return columns


def evaluate_blocks(
    frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    blocks: dict[str, tuple[str, ...]],
    seed: int,
    bootstrap: int,
    reference: str = "physical_action",
    increments: tuple[str, ...] = ("joint", "routing"),
    splitter: str = "logo",
) -> dict[str, Any]:
    predictions: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {}
    for name, signals in blocks.items():
        columns = block_columns(frame, signals)
        prediction, _fold = grouped_predictions(
            frame[columns].to_numpy(dtype=np.float64), y, groups, seed, splitter
        )
        predictions[name] = prediction
        results[name] = {"features": len(columns), **score_block(y, prediction, groups)}
    for name in increments:
        if name not in predictions or reference not in predictions:
            continue
        results[f"increment_{name}_over_{reference}"] = {
            "within_init": init_cluster_ci(
                y, predictions[name], predictions[reference], groups, bootstrap, seed + 991, "within"
            ),
            "pooled": init_cluster_ci(
                y, predictions[name], predictions[reference], groups, bootstrap, seed + 992, "pooled"
            ),
        }
    results["cohort"] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "initial_states": int(len(np.unique(groups))),
        "single_class_initial_states": int(
            sum(
                1
                for g in np.unique(groups)
                if len(np.unique(y[groups == g])) == 1
            )
        ),
        "splitter": splitter,
        **pair_decomposition(y, groups),
    }
    return results, predictions


def grouped_predictions(
    features: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, splitter: str
) -> tuple[np.ndarray, np.ndarray]:
    if splitter == "logo":
        return logo_predictions(features, y, groups, seed)
    if splitter.startswith("sgkf"):
        n_splits = int(splitter.split("-")[1])
        prediction = np.full(len(y), np.nan, dtype=np.float64)
        fold = np.full(len(y), -1, dtype=np.int64)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_id, (train, test) in enumerate(cv.split(features, y, groups)):
            if len(np.unique(y[train])) < 2:
                raise ValueError("stratified group fold has a single-class training set")
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
            model.fit(features[train], y[train])
            prediction[test] = model.predict_proba(features[test])[:, 1]
            fold[test] = fold_id
        if np.any(~np.isfinite(prediction)):
            raise RuntimeError("incomplete grouped predictions")
        return prediction, fold
    raise ValueError(f"unknown splitter {splitter}")


def feasible_splitter(y: np.ndarray, groups: np.ndarray, seed: int) -> str:
    """Prefer strict LOGO; fall back to init-honest StratifiedGroupKFold."""
    unique = np.unique(groups)
    if len(unique) >= 2 and all(
        len(np.unique(y[groups != g])) == 2 and len(np.unique(y)) == 2 for g in unique
    ):
        return "logo"
    for n_splits in (5, 4, 3, 2):
        if len(unique) < n_splits:
            continue
        try:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            if all(
                len(np.unique(y[train])) == 2 and len(np.unique(y[test])) == 2
                for train, test in cv.split(np.zeros((len(y), 1)), y, groups)
            ):
                return f"sgkf-{n_splits}"
        except ValueError:
            continue
    return "infeasible"


# --------------------------------------------------------------------------
# stage 1/2: baseline replication + metric audit + sentinels
# --------------------------------------------------------------------------
def cohort_mask(frame: pd.DataFrame) -> np.ndarray:
    return (
        frame["physical_type"].isin(["stagnation_core", "active_return"])
        & frame["cut_before_onset"]
    ).to_numpy()


def stage_baseline(
    labels: pd.DataFrame,
    caches: dict[int, EpisodeCache],
    goals: dict[str, np.ndarray],
    seed: int,
    bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    frame = build_features(labels, caches, goals, PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS)
    cohort = frame[cohort_mask(frame)].reset_index(drop=True)
    y = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    groups = cohort["init_state_id"].to_numpy()
    results, predictions = evaluate_blocks(
        cohort, y, groups, BLOCKS, seed, bootstrap
    )
    lead = (cohort["physical_onset_query"] - cohort["prospective_cut_query"]).to_numpy()
    results["lead_time"] = {
        physical_type: {
            "mean": float(lead[cohort["physical_type"] == physical_type].mean()),
            "median": float(np.median(lead[cohort["physical_type"] == physical_type])),
            "minimum": int(lead[cohort["physical_type"] == physical_type].min()),
            "maximum": int(lead[cohort["physical_type"] == physical_type].max()),
        }
        for physical_type in ("stagnation_core", "active_return")
    }
    # per fold detail
    fold_rows = []
    for group in np.unique(groups):
        mask = groups == group
        entry = {
            "init_state_id": int(group),
            "n": int(mask.sum()),
            "active_return": int(y[mask].sum()),
        }
        for name in BLOCKS:
            entry[f"auc_{name}"] = (
                fast_auc(y[mask], predictions[name][mask])
                if len(np.unique(y[mask])) == 2
                else float("nan")
            )
            entry[f"mean_score_{name}"] = float(predictions[name][mask].mean())
        fold_rows.append(entry)
    folds = pd.DataFrame(fold_rows)
    return results, cohort, predictions, folds


def stage_seed_split(cohort: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Audit the upstream leave-one-seed-out sensitivity table.

    Upstream reported much higher AUCs under leave-one-seed-out (0.822 / 0.651 /
    0.798).  Seeds are crossed with initial states, so a seed-held-out model has
    seen the other rollouts of the same initial state.  Scoring the *same*
    out-of-fold predictions within initial state separates genuine
    discrimination from initial-state memorisation.
    """
    y = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    seeds = cohort["flow_noise_seed"].to_numpy()
    inits = cohort["init_state_id"].to_numpy()
    out: dict[str, Any] = {"seed_groups": int(len(np.unique(seeds)))}
    predictions: dict[str, np.ndarray] = {}
    for name, block in BLOCKS.items():
        columns = block_columns(cohort, block)
        prediction, _fold = logo_predictions(
            cohort[columns].to_numpy(dtype=np.float64), y, seeds, seed
        )
        predictions[name] = prediction
        out[name] = {
            "pooled_auc": float(roc_auc_score(y, prediction)),
            "within_seed_auc": within_group_auc(y, prediction, seeds)[0],
            "within_init_auc": within_group_auc(y, prediction, inits)[0],
        }
    out["increment_joint_over_physical_action"] = {
        "pooled": out["joint"]["pooled_auc"] - out["physical_action"]["pooled_auc"],
        "within_init": out["joint"]["within_init_auc"]
        - out["physical_action"]["within_init_auc"],
    }
    out["note"] = (
        "the same predictions rescored within initial state; a large gap between "
        "pooled and within-init means the leave-one-seed-out score is initial-state "
        "memorisation, not prefix discrimination"
    )
    return out


def stage_sentinels(
    cohort: pd.DataFrame, seed: int, problem: str = "active_return_vs_stagnation"
) -> pd.DataFrame:
    if problem == "active_return_vs_stagnation":
        y = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    else:
        y = (cohort["physical_type"] != "success").to_numpy(dtype=np.int64)
    groups = cohort["init_state_id"].to_numpy()
    sentinels = {
        "lead_time": (cohort["physical_onset_query"] - cohort["prospective_cut_query"]).to_numpy(float),
        "episode_length": cohort["episode_length"].to_numpy(float),
        "cut_query_index": cohort["prospective_cut_query"].to_numpy(float),
        "pot2_anchor_query": cohort["pot2_anchor_query"].to_numpy(float),
        "init_state_id_raw": cohort["init_state_id"].to_numpy(float),
    }
    if problem != "active_return_vs_stagnation":
        sentinels.pop("lead_time")
    rows = []
    for name, values in sentinels.items():
        marginal = fast_auc(y, values)
        constant = float(np.nanstd(values)) < 1e-12
        if constant:
            rows.append(
                {
                    "problem": problem,
                    "sentinel": name,
                    "constant": True,
                    "marginal_auc": float("nan"),
                    "pooled_logo_auc": float("nan"),
                    "within_init_auc": float("nan"),
                }
            )
            continue
        prediction, _fold = logo_predictions(values[:, None], y, groups, seed)
        within, _folds, _pairs = within_group_auc(y, prediction, groups)
        rows.append(
            {
                "problem": problem,
                "sentinel": name,
                "constant": False,
                "marginal_auc": marginal,
                "pooled_logo_auc": float(roc_auc_score(y, prediction)),
                "within_init_auc": within,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# stage 3: fold-wise goal reference
# --------------------------------------------------------------------------
def stage_fold_goal(
    physical: dict[int, dict[str, Any]],
    caches: dict[int, EpisodeCache],
    seed: int,
    bootstrap: int,
    blocks: dict[str, tuple[str, ...]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, np.ndarray]]:
    """Refit the pot goal means inside every LOGO fold and rebuild the cohort."""
    blocks = blocks or BLOCKS
    signals = tuple(sorted({s for block in blocks.values() for s in block}))
    inits = sorted({item["init_state_id"] for item in physical.values()})
    all_success = [e for e in sorted(physical) if physical[e]["success"]]
    global_goals = goal_reference(physical, all_success)

    fold_tables: dict[int, pd.DataFrame] = {}
    fold_goal_shift: list[dict[str, Any]] = []
    for init in inits:
        train_success = [
            e for e in all_success if physical[e]["init_state_id"] != init
        ]
        goals = goal_reference(physical, train_success)
        fold_goal_shift.append(
            {
                "held_out_init": init,
                "train_successes": len(train_success),
                "pot1_shift_mm": float(
                    1000 * np.linalg.norm(goals["pot1"] - global_goals["pot1"])
                ),
                "pot2_shift_mm": float(
                    1000 * np.linalg.norm(goals["pot2"] - global_goals["pot2"])
                ),
            }
        )
        labels = derive_labels(physical, goals)
        frame = build_features(labels, caches, goals, signals)
        fold_tables[init] = frame

    # cohort membership per fold (defined with that fold's leak-free reference)
    prediction_rows: list[dict[str, Any]] = []
    predictions: dict[str, list[float]] = {name: [] for name in blocks}
    membership: list[dict[str, Any]] = []
    for init in inits:
        frame = fold_tables[init]
        selected = frame[cohort_mask(frame)].reset_index(drop=True)
        train = selected[selected["init_state_id"] != init]
        test = selected[selected["init_state_id"] == init]
        membership.append(
            {
                "held_out_init": init,
                "fold_cohort_n": int(len(selected)),
                "fold_cohort_active_return": int(
                    (selected["physical_type"] == "active_return").sum()
                ),
                "test_n": int(len(test)),
                "test_active_return": int((test["physical_type"] == "active_return").sum()),
            }
        )
        if not len(test):
            continue
        y_train = (train["physical_type"] == "active_return").to_numpy(dtype=np.int64)
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"training fold for init {init} has one class")
        y_test = (test["physical_type"] == "active_return").to_numpy(dtype=np.int64)
        local: dict[str, np.ndarray] = {}
        for name, block in blocks.items():
            columns = block_columns(selected, block)
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                RobustScaler(quantile_range=(25, 75)),
                LogisticRegression(
                    C=0.1, class_weight="balanced", max_iter=3000, random_state=seed + init
                ),
            )
            model.fit(train[columns].to_numpy(dtype=np.float64), y_train)
            local[name] = model.predict_proba(test[columns].to_numpy(dtype=np.float64))[:, 1]
        for row_index in range(len(test)):
            entry = {
                "episode": int(test["episode"].iloc[row_index]),
                "init_state_id": init,
                "physical_type": test["physical_type"].iloc[row_index],
                "label": int(y_test[row_index]),
            }
            for name in blocks:
                entry[f"prediction_{name}"] = float(local[name][row_index])
            prediction_rows.append(entry)
    table = pd.DataFrame(prediction_rows)
    y = table["label"].to_numpy(dtype=np.int64)
    groups = table["init_state_id"].to_numpy()
    results: dict[str, Any] = {}
    scores = {name: table[f"prediction_{name}"].to_numpy() for name in blocks}
    for name in blocks:
        results[name] = {
            "features": len(block_columns(fold_tables[inits[0]], blocks[name])),
            **score_block(y, scores[name], groups),
        }
    for name in ("joint", "routing"):
        if name in scores and "physical_action" in scores:
            results[f"increment_{name}_over_physical_action"] = {
                "within_init": init_cluster_ci(
                    y, scores[name], scores["physical_action"], groups, bootstrap, seed + 991, "within"
                ),
                "pooled": init_cluster_ci(
                    y, scores[name], scores["physical_action"], groups, bootstrap, seed + 992, "pooled"
                ),
            }
    results["cohort"] = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "initial_states": int(len(np.unique(groups))),
        **pair_decomposition(y, groups),
    }
    results["goal_shift_mm"] = {
        "pot1_max": float(max(r["pot1_shift_mm"] for r in fold_goal_shift)),
        "pot2_max": float(max(r["pot2_shift_mm"] for r in fold_goal_shift)),
        "pot1_median": float(np.median([r["pot1_shift_mm"] for r in fold_goal_shift])),
        "pot2_median": float(np.median([r["pot2_shift_mm"] for r in fold_goal_shift])),
    }
    detail = pd.DataFrame(fold_goal_shift).merge(
        pd.DataFrame(membership), on="held_out_init", how="outer"
    )
    return results, detail, table


# --------------------------------------------------------------------------
# stage 4: cohort with success and other_long_failure
# --------------------------------------------------------------------------
def stage_full_cohort(
    labels: pd.DataFrame,
    caches: dict[int, EpisodeCache],
    goals: dict[str, np.ndarray],
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    frame = build_features(
        labels, caches, goals, PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS
    )
    frame = frame[frame["cut_available"]].reset_index(drop=True)
    # exclude rollouts whose physical onset is already at or before the cut
    known_onset = frame["physical_onset_query"] >= 0
    valid = (~known_onset) | (frame["prospective_cut_query"] < frame["physical_onset_query"])
    frame = frame[valid].reset_index(drop=True)
    groups = frame["init_state_id"].to_numpy()
    out: dict[str, Any] = {
        "n": int(len(frame)),
        "class_counts": {
            str(k): int(v) for k, v in frame["physical_type"].value_counts().items()
        },
        "initial_states": int(len(np.unique(groups))),
    }

    # (a) binary: any long failure vs success (the true online early-warning task)
    y = (frame["physical_type"] != "success").to_numpy(dtype=np.int64)
    binary, _pred = evaluate_blocks(frame, y, groups, BLOCKS, seed, bootstrap)
    out["failure_vs_success"] = binary
    out["failure_vs_success_sentinels"] = stage_sentinels(
        frame, seed, problem="failure_vs_success"
    ).to_dict(orient="records")

    # (a2) the same landmark risk set without other_long_failure, whose physical
    # onset is undefined and therefore cannot be checked against the cut
    clean = frame[frame["physical_type"] != "other_long_failure"].reset_index(drop=True)
    y_clean = (clean["physical_type"] != "success").to_numpy(dtype=np.int64)
    clean_results, _pred = evaluate_blocks(
        clean, y_clean, clean["init_state_id"].to_numpy(), BLOCKS, seed, bootstrap
    )
    out["clean_failure_vs_success"] = clean_results
    out["clean_failure_vs_success"]["note"] = (
        "success and the two labelled failure types only; every rollout has a "
        "physical event query strictly later than the cut"
    )

    # (b) 4-class multiclass
    classes = np.unique(frame["physical_type"].to_numpy())
    y_multi = np.searchsorted(classes, frame["physical_type"].to_numpy())
    multi: dict[str, Any] = {"classes": [str(c) for c in classes]}
    proba: dict[str, np.ndarray] = {}
    for name, block in BLOCKS.items():
        columns = block_columns(frame, block)
        proba[name] = logo_multiclass(
            frame[columns].to_numpy(dtype=np.float64), y_multi, groups, seed
        )
        multi[name] = {"features": len(columns), **macro_ovr(y_multi, proba[name], groups)}
    multi["increment_joint_over_physical_action"] = {
        "within_init_macro_ovr": multi["joint"]["within_init_macro_ovr_auc"]
        - multi["physical_action"]["within_init_macro_ovr_auc"],
        "pooled_macro_ovr": multi["joint"]["pooled_macro_ovr_auc"]
        - multi["physical_action"]["pooled_macro_ovr_auc"],
    }
    out["multiclass"] = multi

    # (c) the two-class contrast inside the enlarged, non-selected pool
    pair = frame[frame["physical_type"].isin(["stagnation_core", "active_return"])].reset_index(drop=True)
    y_pair = (pair["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    pair_results, _pred = evaluate_blocks(
        pair, y_pair, pair["init_state_id"].to_numpy(), BLOCKS, seed, bootstrap
    )
    out["two_class_within_full_pool"] = pair_results
    return out


# --------------------------------------------------------------------------
# stage 5: lead-time matching
# --------------------------------------------------------------------------
def _lead_subset_report(
    subset: pd.DataFrame, seed: int, bootstrap: int, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    subset = subset.reset_index(drop=True)
    y = (subset["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    groups = subset["init_state_id"].to_numpy()
    entry: dict[str, Any] = {
        "n": int(len(subset)),
        "active_return": int(y.sum()),
        "stagnation_core": int(len(y) - y.sum()),
        "initial_states": int(len(np.unique(groups))),
        "both_class_initial_states": int(
            sum(1 for g in np.unique(groups) if len(np.unique(y[groups == g])) == 2)
        ),
        "lead_time_mean_by_type": {
            str(k): float(v)
            for k, v in subset.groupby("physical_type")["lead_time"].mean().items()
        },
        **(extra or {}),
        **pair_decomposition(y, groups),
    }
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < 2:
        entry["status"] = "infeasible: fewer than two classes or two initial states"
        return entry
    if entry["within_init_pairs"] == 0:
        entry["status"] = (
            "infeasible: no initial state contains both classes, so there is no "
            "within-init contrast and every AUC would be a between-init comparison"
        )
        return entry
    splitter = feasible_splitter(y, groups, seed)
    entry["splitter"] = splitter
    if splitter == "infeasible":
        entry["status"] = "infeasible: no init-honest split keeps both classes in every fold"
        return entry
    try:
        results, _pred = evaluate_blocks(
            subset, y, groups, BLOCKS, seed, bootstrap, splitter=splitter
        )
        entry["results"] = results
        entry["status"] = "ok"
    except ValueError as error:
        entry["status"] = f"infeasible: {error}"
    return entry


def stage_lead_time(cohort: pd.DataFrame, seed: int, bootstrap: int) -> dict[str, Any]:
    frame = cohort.copy()
    frame["lead_time"] = frame["physical_onset_query"] - frame["prospective_cut_query"]
    out: dict[str, Any] = {
        "lead_time_histogram": {
            str(physical_type): {
                str(int(k)): int(v)
                for k, v in group["lead_time"].value_counts().sort_index().items()
            }
            for physical_type, group in frame.groupby("physical_type")
        },
        "overlap_region": [
            int(max(frame.groupby("physical_type")["lead_time"].min())),
            int(min(frame.groupby("physical_type")["lead_time"].max())),
        ],
    }
    low, high = out["overlap_region"]
    out["lead_window_overlap"] = _lead_subset_report(
        frame[(frame["lead_time"] >= low) & (frame["lead_time"] <= high)],
        seed,
        bootstrap,
        {"window": [low, high]},
    )
    out["lead_window_9_26"] = _lead_subset_report(
        frame[(frame["lead_time"] >= 9) & (frame["lead_time"] <= 26)],
        seed,
        bootstrap,
        {"window": [9, 26]},
    )

    # 1:1 caliper matching on lead time, preferring a partner from the same init
    rng = np.random.default_rng(seed)
    matched_index: list[int] = []
    negatives = frame[frame["physical_type"] == "stagnation_core"]
    used: set[int] = set()
    for index, row in frame[frame["physical_type"] == "active_return"].iterrows():
        candidates = negatives[
            (~negatives.index.isin(used))
            & (np.abs(negatives["lead_time"] - row["lead_time"]) <= 3)
        ]
        same_init = candidates[candidates["init_state_id"] == row["init_state_id"]]
        pool = same_init if len(same_init) else candidates
        if not len(pool):
            continue
        gap = np.abs(pool["lead_time"] - row["lead_time"]).to_numpy()
        best = pool.index.to_numpy()[gap == gap.min()]
        pick = int(rng.choice(best))
        used.add(pick)
        matched_index.extend([index, pick])
    out["caliper_matched"] = _lead_subset_report(
        frame.loc[sorted(set(matched_index))],
        seed,
        bootstrap,
        {"caliper_queries": 3, "matching": "1:1 nearest lead time, same init preferred"},
    )
    return out


# --------------------------------------------------------------------------
# stage 6: token split
# --------------------------------------------------------------------------
def stage_token_split(
    labels: pd.DataFrame,
    physical: dict[int, dict[str, Any]],
    reduced: dict[str, np.ndarray],
    goals: dict[str, np.ndarray],
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    caches_by_variant: dict[str, dict[int, EpisodeCache]] = {}
    for variant, route_key, occ_key in (
        ("action_tokens", "route_action", "occ_action"),
        ("state_token", "route_state", "occ_state"),
    ):
        caches = build_caches(physical, reduced, route_key, occ_key)
        caches_by_variant[variant] = caches
        frame = build_features(
            labels, caches, goals, PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS
        )
        blocks = {
            "physical_action": PHYSICAL_SIGNALS + ACTION_SIGNALS,
            "routing": ROUTING_SIGNALS,
            "joint": PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS,
        }
        cohort = frame[cohort_mask(frame)].reset_index(drop=True)
        y = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
        results, _pred = evaluate_blocks(
            cohort, y, cohort["init_state_id"].to_numpy(), blocks, seed, bootstrap
        )
        out[variant] = {"selective_two_class": results}

        landmark = frame[
            frame["cut_available"]
            & frame["physical_type"].isin(["success", "stagnation_core", "active_return"])
            & (frame["prospective_cut_query"] < frame["physical_onset_query"])
        ].reset_index(drop=True)
        y_landmark = (landmark["physical_type"] != "success").to_numpy(dtype=np.int64)
        landmark_results, _pred = evaluate_blocks(
            landmark,
            y_landmark,
            landmark["init_state_id"].to_numpy(),
            blocks,
            seed,
            bootstrap,
        )
        out[variant]["landmark_failure_vs_success"] = landmark_results
    return out


# --------------------------------------------------------------------------
# stage 7: synthetic injection power
# --------------------------------------------------------------------------
def inject_routing(
    route: np.ndarray,
    occupancy: np.ndarray,
    window: slice,
    source: int,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the window toward the same rollout's routing at an earlier query."""
    new_route = route.copy()
    new_occ = occupancy.copy()
    mix_route = (1.0 - strength) * route[window] + strength * route[source][None]
    mix_occ = (1.0 - strength) * occupancy[window] + strength * occupancy[source][None]
    new_route[window] = normalize(np.maximum(mix_route, 1e-12))
    new_occ[window] = normalize(np.maximum(mix_occ, 1e-12))
    return new_route, new_occ


def stage_power(
    labels: pd.DataFrame,
    physical: dict[int, dict[str, Any]],
    reduced: dict[str, np.ndarray],
    caches: dict[int, EpisodeCache],
    goals: dict[str, np.ndarray],
    seed: int,
    strengths: tuple[float, ...],
    repeats: int,
    bootstrap: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = build_features(
        labels, caches, goals, PHYSICAL_SIGNALS + ACTION_SIGNALS + ROUTING_SIGNALS
    )
    cohort = frame[cohort_mask(frame)].reset_index(drop=True)
    y = (cohort["physical_type"] == "active_return").to_numpy(dtype=np.int64)
    groups = cohort["init_state_id"].to_numpy()
    episodes = cohort["episode"].to_numpy(dtype=np.int64)
    offsets = {int(e): int(o) for e, o in zip(reduced["episode"], reduced["episode_offset"])}
    lengths = {int(e): int(n) for e, n in zip(reduced["episode"], reduced["episode_length"])}
    route_all = reduced["route_action"]
    occ_all = reduced["occ_action"]

    positives = np.flatnonzero(y == 1)
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    routing_columns = block_columns(cohort, ROUTING_SIGNALS)
    joint_columns = block_columns(cohort, BLOCKS["joint"])
    reference_columns = block_columns(cohort, BLOCKS["physical_action"])
    base_reference, _f = logo_predictions(
        cohort[reference_columns].to_numpy(dtype=np.float64), y, groups, seed
    )
    reference_auc, _folds, _pairs = within_group_auc(y, base_reference, groups)

    for strength in strengths:
        detections = 0
        routing_detections = 0
        increments: list[float] = []
        routing_increments: list[float] = []
        routing_aucs: list[float] = []
        for repeat in range(repeats):
            perturbed = cohort.copy()
            updates = {
                column: np.array(perturbed[column].to_numpy(dtype=np.float64), copy=True)
                for column in routing_columns
            }
            for row_index in positives:
                episode = int(episodes[row_index])
                offset, n = offsets[episode], lengths[episode]
                q0 = int(cohort["pot2_anchor_query"].iloc[row_index])
                cut = int(cohort["prospective_cut_query"].iloc[row_index])
                route = route_all[offset : offset + n]
                occupancy = occ_all[offset : offset + n]
                source = int(rng.integers(0, max(q0 - 2, 1)))
                onset = int(rng.integers(q0, cut + 1))
                new_route, new_occ = inject_routing(
                    route, occupancy, slice(onset, cut + 1), source, strength
                )
                cache = EpisodeCache(physical[episode], new_route, new_occ)
                features = cache.window_features(q0, cut, goals, ROUTING_SIGNALS)
                for column, value in features.items():
                    updates[column][row_index] = value
            for column, values in updates.items():
                perturbed[column] = values
            routing_pred, _f = logo_predictions(
                perturbed[routing_columns].to_numpy(dtype=np.float64), y, groups, seed
            )
            joint_pred, _f = logo_predictions(
                perturbed[joint_columns].to_numpy(dtype=np.float64), y, groups, seed
            )
            routing_auc, _folds, _pairs = within_group_auc(y, routing_pred, groups)
            joint_auc, _folds, _pairs = within_group_auc(y, joint_pred, groups)
            increment = joint_auc - reference_auc
            interval = init_cluster_ci(
                y, joint_pred, base_reference, groups, bootstrap, seed + 7000 + repeat, "within"
            )
            detected = bool(np.isfinite(interval["ci95"][0]) and interval["ci95"][0] > 0.0)
            detections += int(detected)
            routing_interval = init_cluster_ci(
                y, routing_pred, base_reference, groups, bootstrap, seed + 8000 + repeat, "within"
            )
            routing_detections += int(
                bool(np.isfinite(routing_interval["ci95"][0]) and routing_interval["ci95"][0] > 0.0)
            )
            increments.append(increment)
            routing_increments.append(routing_auc - reference_auc)
            routing_aucs.append(routing_auc)
        rows.append(
            {
                "strength_lambda": strength,
                "repeats": repeats,
                "power": detections / repeats,
                "power_routing_only": routing_detections / repeats,
                "median_increment": float(np.median(increments)),
                "median_routing_increment": float(np.median(routing_increments)),
                "median_routing_within_auc": float(np.median(routing_aucs)),
                "reference_within_auc": float(reference_auc),
            }
        )
        print(
            f"power lambda={strength:.2f} power(joint)={detections / repeats:.2f} "
            f"power(routing)={routing_detections / repeats:.2f} "
            f"increment={np.median(increments):+.3f} routingAUC={np.median(routing_aucs):.3f}",
            flush=True,
        )
    curve = pd.DataFrame(rows)
    mde = interpolate_mde(curve, 0.80)
    mde_routing = interpolate_mde(
        curve.rename(
            columns={
                "power": "power_joint",
                "power_routing_only": "power",
                "median_increment": "median_increment_joint",
                "median_routing_increment": "median_increment",
            }
        ),
        0.80,
    )
    summary = {
        "reference_within_auc": float(reference_auc),
        "repeats": repeats,
        "bootstrap": bootstrap,
        "mde80_joint_increment": mde["increment"],
        "mde80_joint_lambda": mde["strength"],
        "mde80_routing_only_increment": mde_routing["increment"],
        "mde80_routing_only_lambda": mde_routing["strength"],
        "observed_joint_increment": None,
        "detection_rule": "init-cluster bootstrap 95% CI lower bound of the within-init AUC increment > 0",
        "injection": "window rows interpolated toward the same rollout's routing at a random earlier query (soft router probabilities and hard expert occupancy jointly); never Gaussian noise",
    }
    return summary, curve


def interpolate_mde(curve: pd.DataFrame, target: float) -> dict[str, float]:
    ordered = curve.sort_values("strength_lambda").reset_index(drop=True)
    for index in range(len(ordered) - 1):
        low, high = ordered.iloc[index], ordered.iloc[index + 1]
        if low["power"] < target <= high["power"]:
            span = high["power"] - low["power"]
            weight = (target - low["power"]) / span if span > 0 else 0.0
            return {
                "strength": float(low["strength_lambda"] + weight * (high["strength_lambda"] - low["strength_lambda"])),
                "increment": float(low["median_increment"] + weight * (high["median_increment"] - low["median_increment"])),
            }
    if len(ordered) and ordered["power"].iloc[0] >= target:
        return {
            "strength": float(ordered["strength_lambda"].iloc[0]),
            "increment": float(ordered["median_increment"].iloc[0]),
        }
    return {"strength": float("nan"), "increment": float("nan")}


def stage_positive_control(
    cohort: pd.DataFrame, seed: int
) -> dict[str, Any]:
    """Can the same pipeline read a physical state the routing surely tracks?"""
    target = cohort["eef_step_m__mean"].to_numpy(dtype=np.float64)
    y = (target >= np.median(target)).astype(np.int64)
    groups = cohort["init_state_id"].to_numpy()
    out: dict[str, Any] = {"target": "median split of window mean EEF step"}
    for name in ("routing", "physical_action"):
        columns = block_columns(cohort, BLOCKS[name])
        prediction, _f = logo_predictions(
            cohort[columns].to_numpy(dtype=np.float64), y, groups, seed
        )
        out[name] = score_block(y, prediction, groups)
    return out


# --------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------
def build_caches(
    physical: dict[int, dict[str, Any]],
    reduced: dict[str, np.ndarray],
    route_key: str,
    occ_key: str,
) -> dict[int, EpisodeCache]:
    offsets = {int(e): int(o) for e, o in zip(reduced["episode"], reduced["episode_offset"])}
    lengths = {int(e): int(n) for e, n in zip(reduced["episode"], reduced["episode_length"])}
    caches: dict[int, EpisodeCache] = {}
    for episode in sorted(physical):
        offset, n = offsets[episode], lengths[episode]
        caches[episode] = EpisodeCache(
            physical[episode],
            reduced[route_key][offset : offset + n],
            reduced[occ_key][offset : offset + n],
        )
    return caches


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_self_test() -> None:
    # simplex helpers
    values = np.asarray([[[0.8, 0.2]], [[0.2, 0.8]], [[0.79, 0.21]]], dtype=np.float32)
    distance = pairwise_hellinger(values)
    speed = np.r_[0.0, distance[1, 0], distance[2, 1]].astype(np.float32)
    advantage, lag, anchor = recurrence_signals(distance, speed, 0)
    assert distance.shape == (3, 3)
    assert advantage[2] > 0.0 and lag[2] == 1.0
    scale = causal_scale(speed)
    assert np.allclose(anchor_advantage_only(distance, scale, 0), anchor)

    steps = np.asarray([0.0, 0.03, 0.02, 0.001, 0.002, 0.003, 0.001])
    assert sustained_stasis_onset(steps, 2) == 2

    # fast_auc matches sklearn including ties
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200), 1)
    assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-9

    # within_group_auc: perfectly separable inside groups but reversed between
    y = np.array([0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1])
    s = np.array([0.0, 0.1, 1.0, 1.1])
    within, folds, pairs = within_group_auc(y, s, groups)
    assert within == 1.0 and folds == 2 and pairs == 2
    assert abs(roc_auc_score(y, s) - 0.75) < 1e-9

    # pair decomposition
    decomposition = pair_decomposition(y, groups)
    assert decomposition["within_init_pairs"] == 2 and decomposition["positive_negative_pairs"] == 4

    # LOGO sanity
    groups = np.repeat(np.arange(6), 8)
    y = np.tile(np.r_[np.zeros(4), np.ones(4)], 6).astype(int)
    features = y[:, None] + rng.normal(0.0, 0.1, size=(len(y), 2))
    prediction, fold = logo_predictions(features, y, groups, 0)
    assert np.all(fold >= 0) and roc_auc_score(y, prediction) > 0.9

    # injection stays on the simplex and is monotone in strength
    route = normalize(rng.random((6, 2, 4)).astype(np.float32))
    occupancy = normalize(rng.random((6, 2, 4)).astype(np.float32))
    low, _ = inject_routing(route, occupancy, slice(3, 6), 0, 0.2)
    high, _ = inject_routing(route, occupancy, slice(3, 6), 0, 0.8)
    assert np.allclose(low.sum(-1), 1.0) and np.allclose(high.sum(-1), 1.0)
    assert np.allclose(low[:3], route[:3])
    assert np.linalg.norm(high[3:] - route[3:]) > np.linalg.norm(low[3:] - route[3:])

    # MDE interpolation
    curve = pd.DataFrame(
        {
            "strength_lambda": [0.1, 0.2, 0.3],
            "power": [0.0, 0.6, 1.0],
            "median_increment": [0.01, 0.05, 0.12],
        }
    )
    mde = interpolate_mde(curve, 0.80)
    assert abs(mde["strength"] - 0.25) < 1e-9 and abs(mde["increment"] - 0.085) < 1e-9
    print("self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE_ROOT)
    parser.add_argument("--audit-csv", type=pathlib.Path, default=AUDIT_CSV)
    parser.add_argument("--out-dir", type=pathlib.Path, default=OUT_DIR)
    parser.add_argument("--route-cache", type=pathlib.Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
    parser.add_argument("--power-bootstrap", type=int, default=400)
    parser.add_argument("--power-repeats", type=int, default=12)
    parser.add_argument(
        "--power-strengths",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.35, 0.55, 0.80],
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    physical, summaries = load_client(args.cache_root)
    cache_file = args.route_cache or (args.out_dir / "route_reduced.npz")
    if cache_file.exists():
        with np.load(cache_file) as archive:
            reduced = {key: archive[key] for key in archive.files}
        print(f"loaded reduced route cache {cache_file}", flush=True)
    else:
        reduced = reduce_routes(args.cache_root, summaries)
        np.savez(cache_file, **reduced)
        print(f"wrote reduced route cache {cache_file}", flush=True)

    all_success = [e for e in sorted(physical) if physical[e]["success"]]
    goals = goal_reference(physical, all_success)
    labels = derive_labels(physical, goals)

    # cross-check against the frozen upstream audit table
    audit = pd.read_csv(args.audit_csv)
    audit = audit[audit["task"] == LONG_TASK].set_index("episode")
    check = labels.set_index("episode")
    q0_match = int(
        (
            check["pot2_anchor_query"]
            == audit.loc[check.index, "pot2_first_goal_query"].astype(int)
        ).sum()
    )
    stage_match = int(
        (check["long_physical_stage"] == audit.loc[check.index, "long_physical_stage"]).sum()
    )
    basin_match = int(
        (check["eef_terminal_basin"] == audit.loc[check.index, "eef_terminal_basin"]).sum()
    )
    print(
        f"upstream cross-check: q0 {q0_match}/512  stage {stage_match}/512  basin {basin_match}/512",
        flush=True,
    )
    print(labels["physical_type"].value_counts().to_string(), flush=True)

    caches = build_caches(physical, reduced, "route_action", "occ_action")

    summary: dict[str, Any] = {
        "protocol": {
            "task": LONG_TASK,
            "prefix_after_pot2_queries": PREFIX_AFTER_POT2,
            "validation": "LeaveOneGroupOut(init_state_id)",
            "primary_metric": "pair-weighted mean within-init AUC (pooled AUC also reported)",
        },
        "upstream_cross_check": {
            "pot2_anchor_query_matches": q0_match,
            "long_physical_stage_matches": stage_match,
            "eef_terminal_basin_matches": basin_match,
            "episodes": int(len(check)),
        },
        "physical_type_counts": {
            str(k): int(v) for k, v in labels["physical_type"].value_counts().items()
        },
    }

    print("stage: baseline replication + metric audit", flush=True)
    baseline, cohort, predictions, folds = stage_baseline(
        labels, caches, goals, args.seed, args.bootstrap
    )
    summary["baseline"] = baseline
    folds.to_csv(args.out_dir / "baseline_folds.csv", index=False)
    cohort.to_csv(args.out_dir / "baseline_cohort_features.csv.gz", index=False, compression="gzip")

    sentinels = stage_sentinels(cohort, args.seed)
    sentinels.to_csv(args.out_dir / "sentinels.csv", index=False)
    summary["sentinels"] = sentinels.to_dict(orient="records")
    summary["positive_control"] = stage_positive_control(cohort, args.seed)
    summary["leave_one_seed_out_audit"] = stage_seed_split(cohort, args.seed + 2000)

    print("stage: fold-wise goal reference", flush=True)
    fold_goal, goal_detail, fold_predictions = stage_fold_goal(
        physical, caches, args.seed, args.bootstrap
    )
    fold_goal["test_set_identical_to_baseline_cohort"] = bool(
        sorted(fold_predictions["episode"].tolist()) == sorted(cohort["episode"].tolist())
    )
    summary["fold_goal_reference"] = fold_goal
    goal_detail.to_csv(args.out_dir / "fold_goal_reference.csv", index=False)
    fold_predictions.to_csv(args.out_dir / "fold_goal_predictions.csv", index=False)

    print("stage: full cohort with success and other_long_failure", flush=True)
    summary["full_cohort"] = stage_full_cohort(labels, caches, goals, args.seed, args.bootstrap)

    print("stage: lead-time matching", flush=True)
    summary["lead_time"] = stage_lead_time(cohort, args.seed, args.bootstrap)

    print("stage: token split", flush=True)
    summary["token_split"] = stage_token_split(
        labels, physical, reduced, goals, args.seed, args.bootstrap
    )

    if not args.skip_power:
        print("stage: synthetic injection power", flush=True)
        power, curve = stage_power(
            labels,
            physical,
            reduced,
            caches,
            goals,
            args.seed,
            tuple(args.power_strengths),
            args.power_repeats,
            args.power_bootstrap,
        )
        power["observed_joint_increment"] = summary["baseline"][
            "increment_joint_over_physical_action"
        ]["within_init"]["delta"]
        summary["power"] = power
        curve.to_csv(args.out_dir / "power_curve.csv", index=False)
        make_power_plot(curve, power, args.out_dir / "power_curve.png")

    summary["runtime_seconds"] = round(time.time() - started, 1)
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    print(f"wrote {args.out_dir / 'summary.json'} in {summary['runtime_seconds']}s", flush=True)


def make_power_plot(curve: pd.DataFrame, power: dict[str, Any], out: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(curve["strength_lambda"], curve["power"], marker="o", color="#4C78A8")
    axes[0].axhline(0.8, color="#888888", linestyle="--", linewidth=1)
    axes[0].set_xlabel("injection strength lambda")
    axes[0].set_ylabel("detection rate")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("power vs injection strength")
    axes[1].plot(curve["median_increment"], curve["power"], marker="o", color="#E45756")
    axes[1].axhline(0.8, color="#888888", linestyle="--", linewidth=1)
    if np.isfinite(power.get("mde80_increment", float("nan"))):
        axes[1].axvline(power["mde80_increment"], color="#54A24B", linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("achieved within-init AUC increment")
    axes[1].set_ylabel("detection rate")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("power vs achieved increment")
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
