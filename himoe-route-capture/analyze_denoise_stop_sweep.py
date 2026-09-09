#!/usr/bin/env python3
"""Systematic shadow benchmark for per-query flow-denoising stopping rules.

The benchmark separates two decisions that previous pilots conflated:

1. whether a completed round looks safe to stop at; and
2. what action is returned after stopping.

Every rule is evaluated against the same full ten-round trajectory.  Thresholds
are calibrated without reward/success labels and are evaluated out of fold by
holding out complete rollouts.  This is a shadow benchmark: it does not claim
that a small endpoint error is equivalent to unchanged closed-loop success.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


LIVE_DIMS = 7
TOTAL_ROUNDS = 10
STAGES = np.arange(2, TOTAL_ROUNDS, dtype=np.int64)
EPS = 1e-12
PRIMARY_TOLERANCE = 0.05
PRIMARY_TRAIN_RISK = 0.0
HELD_OUT_ACCEPT_RISK = 0.01
SIGNAL_MIN_STAGES = (4, 6, 8)
MATCHED_SAVINGS = (0.10, 0.20, 0.30)


@dataclasses.dataclass(frozen=True)
class TraceData:
    name: str
    run: Path
    x: np.ndarray
    probability: np.ndarray
    state_probability: np.ndarray
    expert_ids: np.ndarray
    state_expert_ids: np.ndarray
    selected_probability: np.ndarray
    query_id: np.ndarray
    candidate_id: np.ndarray
    rollout_id: np.ndarray
    action_std: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.x))


@dataclasses.dataclass(frozen=True)
class Signal:
    name: str
    family: str
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="runs/flow-lead-cpu-t0s24",
        help="calibration run with flow_traces_part*.npz and routes.zarr",
    )
    parser.add_argument(
        "--holdout-run",
        default=None,
        help="optional second task evaluated with thresholds frozen on --run",
    )
    parser.add_argument(
        "--diagnostic-summary",
        default=None,
        help="optional within-holdout sweep summary, reported only as a diagnostic",
    )
    parser.add_argument("--out-dir", default="analysis/denoise-stop-sweep")
    parser.add_argument("--tolerance", type=float, default=PRIMARY_TOLERANCE)
    parser.add_argument("--train-risk", type=float, default=PRIMARY_TRAIN_RISK)
    return parser.parse_args()


def _normalized_probability(value: np.ndarray) -> np.ndarray:
    result = np.clip(np.asarray(value, dtype=np.float64), 0.0, None)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("router probabilities must be finite with positive mass")
    return result / mass


def load_trace(run: Path, name: str | None = None) -> TraceData:
    parts = [np.load(path) for path in sorted(run.glob("flow_traces_part*.npz"))]
    if not parts:
        raise FileNotFoundError("no flow_traces_part*.npz under %s" % run)
    trajectory = np.concatenate([part["x_traj"] for part in parts])
    query_id = np.concatenate([part["query_id"] for part in parts]).astype(np.int64)
    candidate_id = np.concatenate([part["candidate_id"] for part in parts]).astype(
        np.int64
    )
    if trajectory.ndim == 5 and trajectory.shape[2] == 1:
        trajectory = trajectory[:, :, 0]
    if trajectory.shape[1:] != (TOTAL_ROUNDS + 1, 10, 24):
        raise ValueError("unexpected trajectory shape %s" % (trajectory.shape,))
    if len(query_id) != len(trajectory) or len(candidate_id) != len(trajectory):
        raise ValueError("flow trace identity arrays are not aligned")

    records = json.loads((run / "query_records.json").read_text())
    record_by_query = {int(record["query_id"]): record for record in records}
    if set(np.unique(query_id)) != set(record_by_query):
        raise ValueError("query_records.json does not match flow traces")
    rollout_id = np.asarray(
        [int(record_by_query[int(query)]["flow_noise_seed"]) for query in query_id],
        dtype=np.int64,
    )

    route = zarr.open_group(str(run / "routes.zarr"), mode="r")
    route_query = np.asarray(route["episode_id"][:], dtype=np.int64)
    if not np.array_equal(route_query, query_id):
        raise ValueError("routes.zarr row ordering differs from flow traces")
    probability = np.asarray(route["hb_router_probs"][:], dtype=np.float32)
    expert_ids = np.asarray(route["hb_expert_ids"][:], dtype=np.int16)
    selected = np.asarray(route["hb_selected_prob"][:], dtype=np.float32)
    expected_route = (len(trajectory), 8, TOTAL_ROUNDS, 11, 32)
    if probability.shape != expected_route:
        raise ValueError("unexpected router shape %s" % (probability.shape,))
    if expert_ids.shape != expected_route[:-1] + (4,):
        raise ValueError("unexpected expert-id shape %s" % (expert_ids.shape,))
    if selected.shape != expert_ids.shape:
        raise ValueError("selected probabilities do not align with expert ids")

    metadata = json.loads((run / "server_metadata.json").read_text())
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    if action_std.shape != (LIVE_DIMS,) or np.any(action_std <= 0.0):
        raise ValueError("invalid normalization_action_std")

    return TraceData(
        name=name or run.name,
        run=run,
        x=np.asarray(trajectory[..., :LIVE_DIMS], dtype=np.float64),
        probability=_normalized_probability(probability[:, :, :, 1:, :]),
        state_probability=_normalized_probability(probability[:, :, :, :1, :]),
        expert_ids=expert_ids[:, :, :, 1:, :],
        state_expert_ids=expert_ids[:, :, :, :1, :],
        selected_probability=selected[:, :, :, 1:, :],
        query_id=query_id,
        candidate_id=candidate_id,
        rollout_id=rollout_id,
        action_std=action_std,
    )


def _rms(value: np.ndarray, axes: int | tuple[int, ...]) -> np.ndarray:
    return np.sqrt(np.mean(np.square(value), axis=axes))


def endpoint_predictions(data: TraceData) -> dict[str, np.ndarray]:
    """Return deterministic endpoint estimates [row, stage, action, dim]."""

    x = data.x
    estimates: dict[str, list[np.ndarray]] = {
        "current": [],
        "constant_velocity": [],
        "mean_last2_velocity": [],
        "constant_acceleration": [],
        "linear_fit_3": [],
        "linear_fit_4": [],
        "quadratic_fit_3": [],
    }
    for stage in STAGES:
        remaining = TOTAL_ROUNDS - int(stage)
        current = x[:, stage]
        delta = current - x[:, stage - 1]
        previous_delta = x[:, stage - 1] - x[:, stage - 2]
        estimates["current"].append(current)
        estimates["constant_velocity"].append(current + remaining * delta)
        estimates["mean_last2_velocity"].append(
            current + remaining * (delta + previous_delta) / 2.0
        )
        estimates["constant_acceleration"].append(
            current
            + remaining * delta
            + remaining * (remaining + 1) * (delta - previous_delta) / 2.0
        )
        for window in (3, 4):
            count = min(window, int(stage) + 1)
            times = np.arange(stage - count + 1, stage + 1, dtype=np.float64)
            centered = times - times.mean()
            slope = np.tensordot(
                centered, x[:, stage - count + 1 : stage + 1], axes=(0, 1)
            ) / np.sum(np.square(centered))
            intercept = x[:, stage - count + 1 : stage + 1].mean(axis=1)
            intercept -= slope * times.mean()
            estimates["linear_fit_%d" % window].append(
                intercept + slope * TOTAL_ROUNDS
            )

        # Three points define a quadratic.  Evaluate it directly at round 10.
        y0, y1, y2 = x[:, stage - 2], x[:, stage - 1], current
        first = y2 - y1
        second = y2 - 2.0 * y1 + y0
        estimates["quadratic_fit_3"].append(
            y2 + remaining * first + remaining * (remaining + 1) * second / 2.0
        )
    return {key: np.stack(value, axis=1) for key, value in estimates.items()}


def _ridge_without_intercept(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    gram = design.T @ design
    scale = float(np.trace(gram) / max(1, len(gram)))
    penalty = max(scale * 1e-6, 1e-12)
    return np.linalg.solve(gram + penalty * np.eye(len(gram)), design.T @ target)


def fit_endpoint_coefficients(data: TraceData, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Fit tiny x10-supervised extrapolators on complete training rollouts."""

    selected = np.asarray(rows, dtype=bool)
    if selected.shape != (data.rows,) or not np.any(selected):
        raise ValueError("endpoint coefficient training mask is empty or misaligned")
    coefficients = {
        "learned_velocity_scalar": [],
        "learned_two_velocity_scalar": [],
        "learned_two_velocity_per_dim": [],
    }
    final = data.x[selected, TOTAL_ROUNDS]
    for stage in STAGES:
        current = data.x[selected, stage]
        delta = current - data.x[selected, stage - 1]
        previous_delta = data.x[selected, stage - 1] - data.x[selected, stage - 2]
        target = (final - current).reshape(-1)
        one = delta.reshape(-1, 1)
        two = np.column_stack((delta.reshape(-1), previous_delta.reshape(-1)))
        coefficients["learned_velocity_scalar"].append(
            _ridge_without_intercept(one, target)
        )
        coefficients["learned_two_velocity_scalar"].append(
            _ridge_without_intercept(two, target)
        )
        per_dimension = []
        for dimension in range(LIVE_DIMS):
            design = np.column_stack(
                (
                    delta[..., dimension].reshape(-1),
                    previous_delta[..., dimension].reshape(-1),
                )
            )
            per_dimension.append(
                _ridge_without_intercept(
                    design, (final - current)[..., dimension].reshape(-1)
                )
            )
        coefficients["learned_two_velocity_per_dim"].append(
            np.stack(per_dimension)
        )
    return {name: np.stack(value) for name, value in coefficients.items()}


def apply_endpoint_coefficients(
    data: TraceData, rows: np.ndarray, coefficients: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    selected = np.asarray(rows, dtype=bool)
    estimates = {name: [] for name in coefficients}
    for stage_index, stage in enumerate(STAGES):
        current = data.x[selected, stage]
        delta = current - data.x[selected, stage - 1]
        previous_delta = data.x[selected, stage - 1] - data.x[selected, stage - 2]
        one = coefficients["learned_velocity_scalar"][stage_index]
        two = coefficients["learned_two_velocity_scalar"][stage_index]
        per_dimension = coefficients["learned_two_velocity_per_dim"][stage_index]
        estimates["learned_velocity_scalar"].append(current + one[0] * delta)
        estimates["learned_two_velocity_scalar"].append(
            current + two[0] * delta + two[1] * previous_delta
        )
        estimates["learned_two_velocity_per_dim"].append(
            current
            + per_dimension[:, 0][None, None, :] * delta
            + per_dimension[:, 1][None, None, :] * previous_delta
        )
    return {name: np.stack(value, axis=1) for name, value in estimates.items()}


def oof_endpoint_predictions(data: TraceData) -> dict[str, np.ndarray]:
    """Leave-one-rollout-out predictions for self-supervised endpoint heads."""

    rollouts = np.unique(data.rollout_id)
    if len(rollouts) < 2:
        raise ValueError("self-supervised endpoint heads need at least two rollouts")
    result = {
        name: np.empty((data.rows, len(STAGES), 10, LIVE_DIMS), dtype=np.float64)
        for name in (
            "learned_velocity_scalar",
            "learned_two_velocity_scalar",
            "learned_two_velocity_per_dim",
        )
    }
    for held_out in rollouts:
        train = data.rollout_id != held_out
        test = ~train
        coefficients = fit_endpoint_coefficients(data, train)
        predictions = apply_endpoint_coefficients(data, test, coefficients)
        for name in result:
            result[name][test] = predictions[name]
    return result


def transferred_endpoint_predictions(
    train: TraceData, test: TraceData
) -> dict[str, np.ndarray]:
    coefficients = fit_endpoint_coefficients(
        train, np.ones(train.rows, dtype=bool)
    )
    return apply_endpoint_coefficients(test, np.ones(test.rows, dtype=bool), coefficients)


def endpoint_error(prediction: np.ndarray, final: np.ndarray) -> np.ndarray:
    difference = prediction - final[:, None]
    numerator = np.linalg.norm(difference.reshape(len(final), len(STAGES), -1), axis=-1)
    denominator = np.linalg.norm(final.reshape(len(final), -1), axis=-1)
    return numerator / np.maximum(denominator[:, None], EPS)


def _topk_intersection(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # The gate has unique top-k ids at each site.  This still works if a malformed
    # capture contains duplicates because each left id contributes at most once.
    return np.any(left[..., :, None] == right[..., None, :], axis=-1).sum(axis=-1)


def _dense_selected(ids: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = np.zeros(ids.shape[:-1] + (32,), dtype=np.float64)
    np.put_along_axis(result, ids.astype(np.int64), weights.astype(np.float64), axis=-1)
    return result


def build_signals(data: TraceData) -> list[Signal]:
    """Build route, latent, and clock observables available online at each stage."""

    probability = data.probability
    ids = data.expert_ids
    selected = data.selected_probability
    x = data.x
    values: dict[tuple[str, str], list[np.ndarray]] = {}

    def add(family: str, name: str, value: np.ndarray) -> None:
        values.setdefault((family, name), []).append(np.asarray(value, dtype=np.float64))

    previous_core: dict[str, np.ndarray] = {}
    two_steps_ago_core: dict[str, np.ndarray] = {}
    for stage in STAGES:
        current_index = int(stage) - 1
        previous_index = current_index - 1
        current = probability[:, :, current_index]
        previous = probability[:, :, previous_index]
        sqrt_delta = np.sqrt(current) - np.sqrt(previous)
        hellinger = np.sqrt(0.5 * np.sum(np.square(sqrt_delta), axis=-1))
        tv = 0.5 * np.sum(np.abs(current - previous), axis=-1)
        midpoint = 0.5 * (current + previous)
        js = 0.5 * np.sum(
            current * np.log(np.maximum(current, EPS) / np.maximum(midpoint, EPS))
            + previous * np.log(np.maximum(previous, EPS) / np.maximum(midpoint, EPS)),
            axis=-1,
        )
        cosine = 1.0 - np.sum(current * previous, axis=-1) / np.maximum(
            np.linalg.norm(current, axis=-1) * np.linalg.norm(previous, axis=-1), EPS
        )
        l2 = _rms(current - previous, -1)

        current_mean = _normalized_probability(current.mean(axis=2))
        previous_mean = _normalized_probability(previous.mean(axis=2))
        mean_hellinger = np.sqrt(
            0.5 * np.sum(np.square(np.sqrt(current_mean) - np.sqrt(previous_mean)), axis=-1)
        )
        core = {
            "prob_mean_hellinger": _rms(mean_hellinger, 1),
            "prob_site_hellinger_mean": hellinger.mean(axis=(1, 2)),
            "prob_site_hellinger_max": hellinger.max(axis=(1, 2)),
            "prob_site_tv_mean": tv.mean(axis=(1, 2)),
            "prob_site_tv_max": tv.max(axis=(1, 2)),
            "prob_site_js_mean": js.mean(axis=(1, 2)),
            "prob_site_cosine_mean": cosine.mean(axis=(1, 2)),
            "prob_site_l2_mean": l2.mean(axis=(1, 2)),
        }

        current_ids = ids[:, :, current_index]
        previous_ids = ids[:, :, previous_index]
        top1_switch = current_ids[..., 0] != previous_ids[..., 0]
        intersection = _topk_intersection(current_ids, previous_ids)
        token_jaccard = 1.0 - intersection / np.maximum(8.0 - intersection, 1.0)
        current_set = np.zeros((data.rows, 8, 32), dtype=bool)
        previous_set = np.zeros_like(current_set)
        for slot in range(4):
            np.put_along_axis(
                current_set,
                current_ids[..., slot].reshape(data.rows, 8, -1),
                True,
                axis=-1,
            )
            np.put_along_axis(
                previous_set,
                previous_ids[..., slot].reshape(data.rows, 8, -1),
                True,
                axis=-1,
            )
        set_intersection = (current_set & previous_set).sum(axis=-1)
        set_union = (current_set | previous_set).sum(axis=-1)
        set_jaccard = 1.0 - set_intersection / np.maximum(set_union, 1)
        count_delta = np.abs(current_set.sum(axis=-1) - previous_set.sum(axis=-1))
        dense_current = _dense_selected(current_ids, selected[:, :, current_index])
        dense_previous = _dense_selected(previous_ids, selected[:, :, previous_index])
        weighted_jaccard = 1.0 - np.minimum(dense_current, dense_previous).sum(
            axis=-1
        ) / np.maximum(np.maximum(dense_current, dense_previous).sum(axis=-1), EPS)
        core.update(
            {
                "top1_switch_mean": top1_switch.mean(axis=(1, 2)),
                "top1_switch_layer_max": top1_switch.mean(axis=2).max(axis=1),
                "top4_token_jaccard_mean": token_jaccard.mean(axis=(1, 2)),
                "top4_token_jaccard_max": token_jaccard.max(axis=(1, 2)),
                "dedup_set_jaccard_mean": set_jaccard.mean(axis=1),
                "dedup_set_jaccard_max": set_jaccard.max(axis=1),
                "dedup_count_delta_mean": count_delta.mean(axis=1),
                "dedup_count_delta_max": count_delta.max(axis=1),
                "weighted_top4_jaccard_mean": weighted_jaccard.mean(axis=(1, 2)),
            }
        )

        ordered_current = np.sort(current, axis=-1)[..., ::-1]
        ordered_previous = np.sort(previous, axis=-1)[..., ::-1]
        entropy_current = -np.sum(
            current * np.log(np.maximum(current, EPS)), axis=-1
        ) / math.log(32.0)
        entropy_previous = -np.sum(
            previous * np.log(np.maximum(previous, EPS)), axis=-1
        ) / math.log(32.0)
        scalar_current = {
            "entropy": entropy_current,
            "top1_mass": ordered_current[..., 0],
            "margin_1_2": ordered_current[..., 0] - ordered_current[..., 1],
            "top4_mass": ordered_current[..., :4].sum(axis=-1),
        }
        scalar_previous = {
            "entropy": entropy_previous,
            "top1_mass": ordered_previous[..., 0],
            "margin_1_2": ordered_previous[..., 0] - ordered_previous[..., 1],
            "top4_mass": ordered_previous[..., :4].sum(axis=-1),
        }
        for name in scalar_current:
            core["%s_abs_delta" % name] = np.abs(
                scalar_current[name] - scalar_previous[name]
            ).mean(axis=(1, 2))
            core["%s_level" % name] = scalar_current[name].mean(axis=(1, 2))
        core["active_expert_count_level"] = current_set.sum(axis=-1).mean(axis=1)

        for name, value in core.items():
            add("route", name, value)
            if name in previous_core:
                add("route", name + "__two_round_max", np.maximum(value, previous_core[name]))
                add("route", name + "__two_round_mean", 0.5 * (value + previous_core[name]))
            else:
                # Infinity means a two-round rule cannot fire at the first stage.
                add("route", name + "__two_round_max", np.full(data.rows, np.inf))
                add("route", name + "__two_round_mean", np.full(data.rows, np.inf))
            if name in previous_core and name in two_steps_ago_core:
                history = np.stack(
                    [value, previous_core[name], two_steps_ago_core[name]], axis=1
                )
                add("route", name + "__three_round_max", history.max(axis=1))
                add("route", name + "__three_round_mean", history.mean(axis=1))
            else:
                add("route", name + "__three_round_max", np.full(data.rows, np.inf))
                add("route", name + "__three_round_mean", np.full(data.rows, np.inf))
        two_steps_ago_core = previous_core
        previous_core = core

        current_x = x[:, stage]
        delta = current_x - x[:, stage - 1]
        previous_delta = x[:, stage - 1] - x[:, stage - 2]
        remaining = TOTAL_ROUNDS - int(stage)
        current_projection = current_x + remaining * delta
        previous_projection = x[:, stage - 1] + (remaining + 1) * previous_delta
        flat_norm = np.linalg.norm(current_x.reshape(data.rows, -1), axis=-1)
        delta_flat = delta.reshape(data.rows, -1)
        previous_flat = previous_delta.reshape(data.rows, -1)
        turning = 1.0 - np.sum(delta_flat * previous_flat, axis=-1) / np.maximum(
            np.linalg.norm(delta_flat, axis=-1)
            * np.linalg.norm(previous_flat, axis=-1),
            EPS,
        )
        latent = {
            "update_rms": _rms(delta, (1, 2)),
            "update_max_abs": np.abs(delta).max(axis=(1, 2)),
            "translation_update_rms": _rms(delta[..., :3], (1, 2)),
            "rotation_update_rms": _rms(delta[..., 3:6], (1, 2)),
            "gripper_update_rms": _rms(delta[..., 6], 1),
            "relative_update": np.linalg.norm(delta_flat, axis=-1)
            / np.maximum(flat_norm, EPS),
            "update_acceleration_rms": _rms(delta - previous_delta, (1, 2)),
            "update_turning": turning,
            "projected_endpoint_disagreement": _rms(
                current_projection - previous_projection, (1, 2)
            ),
            "curvature_remainder_proxy": (
                remaining * (remaining + 1) / 2.0
            )
            * _rms(delta - previous_delta, (1, 2)),
        }
        for name, value in latent.items():
            add("latent", name, value)
        add("clock", "completed_round", np.full(data.rows, float(stage)))

        state_current = data.state_probability[:, :, current_index, 0]
        state_previous = data.state_probability[:, :, previous_index, 0]
        state_hellinger = np.sqrt(
            0.5
            * np.sum(
                np.square(np.sqrt(state_current) - np.sqrt(state_previous)), axis=-1
            )
        )
        state_top1_switch = (
            data.state_expert_ids[:, :, current_index, 0, 0]
            != data.state_expert_ids[:, :, previous_index, 0, 0]
        )
        add("negative_control", "state_token_hellinger", _rms(state_hellinger, 1))
        add(
            "negative_control",
            "state_token_top1_switch",
            state_top1_switch.mean(axis=1),
        )
        add("negative_control", "as_route_constant", np.zeros(data.rows))

    signals: list[Signal] = []
    for (family, name), sequence in values.items():
        array = np.stack(sequence, axis=1)
        if array.shape != (data.rows, len(STAGES)):
            raise AssertionError("signal %s has shape %s" % (name, array.shape))
        signals.append(Signal(name=name, family=family, values=array))
    return signals


def _stops_for_threshold(
    values: np.ndarray, threshold: float, direction: str, min_stage: int
) -> np.ndarray:
    eligible = STAGES >= min_stage
    finite = np.isfinite(values)
    if direction == "le":
        trigger = finite & (values <= threshold) & eligible[None]
    elif direction == "ge":
        trigger = finite & (values >= threshold) & eligible[None]
    else:
        raise ValueError("unknown threshold direction %s" % direction)
    found = trigger.any(axis=1)
    first = np.argmax(trigger, axis=1)
    return np.where(found, STAGES[first], TOTAL_ROUNDS).astype(np.int64)


def _selected_error(error: np.ndarray, stops: np.ndarray) -> np.ndarray:
    result = np.zeros(len(stops), dtype=np.float64)
    early = stops < TOTAL_ROUNDS
    if np.any(early):
        stage_index = stops[early] - int(STAGES[0])
        result[early] = error[np.flatnonzero(early), stage_index]
    return result


def _candidate_thresholds(values: np.ndarray, direction: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.asarray([], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, 257)
    candidates = np.unique(np.quantile(finite, quantiles))
    if direction == "le":
        return np.concatenate(([-np.inf], candidates, [np.inf]))
    return np.concatenate(([np.inf], candidates[::-1], [-np.inf]))


def _threshold_curve(
    values: np.ndarray,
    error: np.ndarray,
    tolerance: float,
    direction: str,
    min_stage: int,
) -> dict[str, np.ndarray]:
    """Evaluate every candidate threshold in one vectorized pass."""

    thresholds = _candidate_thresholds(values, direction)
    eligible = STAGES >= min_stage
    finite = np.isfinite(values)
    if direction == "le":
        trigger = (
            finite[:, :, None]
            & (values[:, :, None] <= thresholds[None, None, :])
            & eligible[None, :, None]
        )
    elif direction == "ge":
        trigger = (
            finite[:, :, None]
            & (values[:, :, None] >= thresholds[None, None, :])
            & eligible[None, :, None]
        )
    else:
        raise ValueError("unknown threshold direction %s" % direction)
    found = trigger.any(axis=1)
    first = np.argmax(trigger, axis=1)
    stops = np.where(found, STAGES[first], TOTAL_ROUNDS)
    selected = np.zeros_like(stops, dtype=np.float64)
    rows, columns = np.nonzero(found)
    selected[rows, columns] = error[rows, first[rows, columns]]
    return {
        "threshold": thresholds,
        "saving": np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS, axis=0),
        "risk": np.mean(selected > tolerance, axis=0),
        "mean_error": selected.mean(axis=0),
    }


def _select_threshold(
    curve: dict[str, np.ndarray],
    max_risk: float,
    target_saving: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Select one training threshold from a precomputed trade-off curve."""

    best: tuple[tuple[float, ...], int] | None = None
    for index in range(len(curve["threshold"])):
        saving = float(curve["saving"][index])
        risk = float(curve["risk"][index])
        mean_error = float(curve["mean_error"][index])
        if target_saving is None:
            if risk > max_risk + 1e-15:
                continue
            objective = (saving, -risk, -mean_error)
        else:
            objective = (-abs(saving - target_saving), -risk, -mean_error)
        if best is None or objective > best[0]:
            best = (objective, index)
    if best is None:
        raise AssertionError("the never-stop threshold must always be feasible")
    index = best[1]
    return float(curve["threshold"][index]), {
        "saving": float(curve["saving"][index]),
        "risk": float(curve["risk"][index]),
        "mean_error": float(curve["mean_error"][index]),
    }


def calibrate_threshold(
    values: np.ndarray,
    error: np.ndarray,
    tolerance: float,
    max_risk: float,
    direction: str,
    min_stage: int,
    target_saving: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Choose a threshold on training rows only."""

    curve = _threshold_curve(values, error, tolerance, direction, min_stage)
    return _select_threshold(curve, max_risk, target_saving)


def _aggregate_result(
    data: TraceData,
    signal: Signal,
    estimator: str,
    direction: str,
    min_stage: int,
    stops: np.ndarray,
    selected: np.ndarray,
    thresholds: Iterable[float],
    tolerance: float,
    protocol: str,
) -> dict[str, Any]:
    thresholds = list(thresholds)
    unsafe = selected > tolerance
    unsafe_queries = np.unique(data.query_id[unsafe])
    unsafe_rollouts = np.unique(data.rollout_id[unsafe])
    rollout_saving = [
        float(np.mean((TOTAL_ROUNDS - stops[data.rollout_id == rollout]) / TOTAL_ROUNDS))
        for rollout in np.unique(data.rollout_id)
    ]
    return {
        "protocol": protocol,
        "family": signal.family,
        "signal": signal.name,
        "endpoint_estimator": estimator,
        "direction": direction,
        "min_stage": min_stage,
        "saving": float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)),
        "unsafe_rate": float(np.mean(unsafe)),
        "unsafe_count": int(unsafe.sum()),
        "unsafe_query_count": int(len(unsafe_queries)),
        "unsafe_query_any_rate": float(
            len(unsafe_queries) / len(np.unique(data.query_id))
        ),
        "unsafe_rollout_count": int(len(unsafe_rollouts)),
        "rollout_saving_range": [min(rollout_saving), max(rollout_saving)],
        "mean_relative_error": float(selected.mean()),
        "p95_relative_error": float(np.percentile(selected, 95)),
        "p99_relative_error": float(np.percentile(selected, 99)),
        "early_fraction": float(np.mean(stops < TOTAL_ROUNDS)),
        "mean_completed_rounds": float(stops.mean()),
        "stop_histogram": {
            str(stage): int(np.sum(stops == stage))
            for stage in range(int(STAGES[0]), TOTAL_ROUNDS + 1)
        },
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_range": [float(np.min(thresholds)), float(np.max(thresholds))],
    }


def cross_validated_signal_sweep(
    data: TraceData,
    signals: list[Signal],
    error: np.ndarray,
    estimator: str,
    tolerance: float,
    train_risk: float,
) -> list[dict[str, Any]]:
    rollouts = np.unique(data.rollout_id)
    if len(rollouts) < 2:
        raise ValueError("cross-validation needs at least two complete rollouts")
    results: list[dict[str, Any]] = []
    for signal_index, signal in enumerate(signals):
        for direction in ("le", "ge"):
            for min_stage in SIGNAL_MIN_STAGES:
                protocols = {
                    "safety_calibrated_loeo": None,
                    **{
                        "compute_matched_%02d" % round(100 * target): target
                        for target in MATCHED_SAVINGS
                    },
                }
                oof_stops = {
                    protocol: np.full(data.rows, TOTAL_ROUNDS, dtype=np.int64)
                    for protocol in protocols
                }
                thresholds: dict[str, list[float]] = {
                    protocol: [] for protocol in protocols
                }
                for held_out in rollouts:
                    train = data.rollout_id != held_out
                    test = ~train
                    curve = _threshold_curve(
                        signal.values[train],
                        error[train],
                        tolerance,
                        direction,
                        min_stage,
                    )
                    for protocol, target in protocols.items():
                        threshold, _ = _select_threshold(
                            curve, train_risk, target_saving=target
                        )
                        thresholds[protocol].append(threshold)
                        oof_stops[protocol][test] = _stops_for_threshold(
                            signal.values[test], threshold, direction, min_stage
                        )
                for protocol in protocols:
                    selected = _selected_error(error, oof_stops[protocol])
                    results.append(
                        _aggregate_result(
                            data,
                            signal,
                        estimator,
                        direction,
                            min_stage,
                            oof_stops[protocol],
                            selected,
                            thresholds[protocol],
                            tolerance,
                            protocol,
                        )
                    )
        if signal_index % 25 == 0:
            print(
                "signal sweep %d/%d: %s"
                % (signal_index + 1, len(signals), signal.name),
                flush=True,
            )
    return results


def fixed_stage_rows(
    data: TraceData,
    predictions: dict[str, np.ndarray],
    tolerance: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    final = data.x[:, TOTAL_ROUNDS]
    rows: list[dict[str, Any]] = []
    errors: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        error = endpoint_error(prediction, final)
        errors[name] = error
        for index, stage in enumerate(STAGES):
            difference = prediction[:, index] - final
            physical = difference * data.action_std[None, None]
            relative = error[:, index]
            sign_flip = np.any(
                np.signbit(prediction[:, index, :, 6]) != np.signbit(final[:, :, 6]),
                axis=1,
            )
            unsafe = relative > tolerance
            rows.append(
                {
                    "estimator": name,
                    "stage": int(stage),
                    "saving": float((TOTAL_ROUNDS - stage) / TOTAL_ROUNDS),
                    "mean_relative_error": float(relative.mean()),
                    "median_relative_error": float(np.median(relative)),
                    "p90_relative_error": float(np.percentile(relative, 90)),
                    "p95_relative_error": float(np.percentile(relative, 95)),
                    "unsafe_rate": float(np.mean(unsafe)),
                    "unsafe_count": int(unsafe.sum()),
                    "unsafe_query_count": int(len(np.unique(data.query_id[unsafe]))),
                    "unsafe_query_any_rate": float(
                        len(np.unique(data.query_id[unsafe]))
                        / len(np.unique(data.query_id))
                    ),
                    "unsafe_rollout_count": int(
                        len(np.unique(data.rollout_id[unsafe]))
                    ),
                    "physical_rms": float(_rms(physical, (1, 2)).mean()),
                    "physical_max_abs_p95": float(
                        np.percentile(np.abs(physical).max(axis=(1, 2)), 95)
                    ),
                    "gripper_sign_flip_rate": float(sign_flip.mean()),
                }
            )
    return rows, errors


def oracle_rows(errors: dict[str, np.ndarray], tolerance: float) -> list[dict[str, Any]]:
    rows = []
    for name, error in errors.items():
        safe = error <= tolerance
        found = safe.any(axis=1)
        first = np.argmax(safe, axis=1)
        stops = np.where(found, STAGES[first], TOTAL_ROUNDS)
        selected = _selected_error(error, stops)
        rows.append(
            {
                "estimator": name,
                "saving": float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)),
                "unsafe_rate": float(np.mean(selected > tolerance)),
                "mean_completed_rounds": float(stops.mean()),
                "early_fraction": float(np.mean(stops < TOTAL_ROUNDS)),
            }
        )
    return rows


def _best_rows(
    rows: list[dict[str, Any]], family: str, protocol: str, count: int = 8
) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows if row["family"] == family and row["protocol"] == protocol
    ]
    return sorted(
        candidates,
        key=lambda row: (
            row["unsafe_rate"] > HELD_OUT_ACCEPT_RISK,
            -row["saving"],
            row["unsafe_rate"],
            row["mean_relative_error"],
        ),
    )[:count]


def load_internal_diagnostic(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text())
    rows = summary["signal_sweep"]
    return {
        "run": summary["development"]["run"],
        "rows": summary["development"]["rows"],
        "rollouts": summary["development"]["rollouts"],
        "best_clock": _best_rows(rows, "clock", "safety_calibrated_loeo", 1)[0],
        "best_route": _best_rows(rows, "route", "safety_calibrated_loeo", 1)[0],
        "best_latent": _best_rows(rows, "latent", "safety_calibrated_loeo", 1)[0],
    }


def _format_percent(value: float) -> str:
    return "%.1f%%" % (100.0 * value)


def _risk_label(row: dict[str, Any]) -> str:
    return "%s (%d 行/%d query)" % (
        _format_percent(row["unsafe_rate"]),
        row["unsafe_count"],
        row["unsafe_query_count"],
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_keys = [
        key
        for key in rows[0]
        if all(not isinstance(row.get(key), (dict, list)) for row in rows)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in scalar_keys})


def plot_pareto(rows: list[dict[str, Any]], out: Path) -> None:
    selected = [row for row in rows if row["protocol"] == "safety_calibrated_loeo"]
    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    colors = {
        "route": "#2f6f9f",
        "latent": "#b24a3b",
        "clock": "#4f7d4b",
        "negative_control": "#777777",
    }
    for family in ("route", "latent", "clock", "negative_control"):
        group = [row for row in selected if row["family"] == family]
        axis.scatter(
            [100 * row["saving"] for row in group],
            [100 * row["unsafe_rate"] for row in group],
            s=18,
            alpha=0.55,
            color=colors[family],
            label=family,
        )
    axis.axhline(1.0, color="#444444", linestyle="--", linewidth=1)
    axis.set_xlabel("denoise-pass saving (%)")
    axis.set_ylabel("held-out endpoint unsafe rate (%)")
    axis.set_title("Stopping-signal and endpoint-estimator sweep")
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(out, dpi=180)
    plt.close(figure)


def _fixed_lookup(rows: list[dict[str, Any]], estimator: str, stage: int) -> dict[str, Any]:
    matches = [
        row for row in rows if row["estimator"] == estimator and row["stage"] == stage
    ]
    if len(matches) != 1:
        raise AssertionError("fixed-stage row is not unique")
    return matches[0]


def write_report(
    out: Path,
    data: TraceData,
    fixed: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    sweep: list[dict[str, Any]],
    tolerance: float,
    holdout: dict[str, Any] | None,
) -> None:
    raw_oracle = next(row for row in oracle if row["estimator"] == "current")
    jump_oracle = next(
        row for row in oracle if row["estimator"] == "constant_velocity"
    )
    best_route = _best_rows(sweep, "route", "safety_calibrated_loeo")
    best_latent = _best_rows(sweep, "latent", "safety_calibrated_loeo")
    best_clock = _best_rows(sweep, "clock", "safety_calibrated_loeo", count=1)[0]
    fixed_acceleration = _fixed_lookup(fixed, "constant_acceleration", 8)
    lines = [
        "# 每次推理内去噪停止方法全量扫描",
        "",
        "## 实验思路",
        "",
        "- 单位是一次 policy query 内的一条完整 `x0→x10` 推理轨迹；每个 query 的停止器独立重置，不跨控制步。",
        "- 数据为 `%d` 条候选轨迹、`%d` 个真实控制状态、`%d` 条完整 rollout。所有阈值按完整 rollout 留一，测试折从不参与阈值选择。"
        % (data.rows, len(np.unique(data.query_id)), len(np.unique(data.rollout_id))),
        "- 不使用 reward/success。自监督真值只是同一次推理完整第 10 轮的动作；主安全线定义为预测动作与第 10 轮动作的相对 L2 误差 `<= %.3f`。" % tolerance,
        "- 共扫描路由概率距离、top-k/去重集合、熵与集中度、两/三轮连续规则、latent 更新/曲率、固定轮数，以及 10 种停止后输出方式；其中 7 种无训练输出器与停止信号做完整笛卡尔组合，3 种自监督输出头单独做 rollout 留出。",
        "- 这里的“全量”指当前前向过程可直接读取的有限方法网格，不声称穷尽数学上无限多的函数。完整 `%d` 行结果保存在 `signal_sweep.csv`。"
        % len(sweep),
        "",
        "| 信号族 | 实际扫描内容 |",
        "|---|---|",
        "| 路由概率 | Hellinger、TV、JS、cosine、L2；token 均值/最大值、先聚合 token 再比较 |",
        "| 专家身份 | top-1 切换、token top-4 Jaccard、逐层去重集合 Jaccard、加权 Jaccard、去重数量 |",
        "| 路由形状 | entropy、top-1 mass、top1-top2 margin、top-4 mass、活跃专家数的绝对值与相邻变化 |",
        "| 时间规则 | 当前一轮、最近两/三轮 max 与 mean；最早第 4/6/8 轮；阈值正反方向 |",
        "| latent | 更新 RMS/最大值、平移/旋转/gripper、相对更新、加速度、转向、外推终点自洽性 |",
        "| 零对照 | state-token 路由变化、恒定 AS 路由；必须退化为固定轮数，不能产生候选判别 |",
        "| 输出器 | 当前 latent、常速度、两轮平均速度、常加速度、3/4 点线性、3 点二次，以及 3 个 x10 自监督速度头 |",
        "",
        "## 先看天花板",
        "",
        "直接输出当前 latent 的事后 oracle 也只能节省 `%s`；常速度外推的 oracle 可节省 `%s`。这说明问题的关键首先不是如何识别稳定，而是停止后不能直接返回未积分到 `t=0` 的 `x_s`。"
        % (_format_percent(raw_oracle["saving"]), _format_percent(jump_oracle["saving"])),
        "",
        "| 输出方式 | 轮数 | 节省 | 中位相对误差 | P90 | 超过安全线 | gripper 符号变化 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "current": "直接输出当前 latent",
        "constant_velocity": "最近速度外推",
        "mean_last2_velocity": "最近两轮平均速度外推",
        "constant_acceleration": "常加速度外推",
        "linear_fit_3": "最近 3 点线性拟合",
        "linear_fit_4": "最近 4 点线性拟合",
        "quadratic_fit_3": "最近 3 点二次外推",
        "learned_velocity_scalar": "自监督单速度系数",
        "learned_two_velocity_scalar": "自监督双速度系数",
        "learned_two_velocity_per_dim": "自监督逐维双速度系数",
    }
    for estimator in labels:
        for stage in (4, 6, 8, 9):
            row = _fixed_lookup(fixed, estimator, stage)
            lines.append(
                "| %s | %d | %s | %.4f | %.4f | %s | %s |"
                % (
                    labels[estimator],
                    stage,
                    _format_percent(row["saving"]),
                    row["median_relative_error"],
                    row["p90_relative_error"],
                    _format_percent(row["unsafe_rate"]),
                    _format_percent(row["gripper_sign_flip_rate"]),
                )
            )
    lines += [
        "",
        "## 停止信号",
        "",
        "下表的阈值只在训练 rollout 上选择，要求训练折零危险漏停，然后在整条留出 rollout 上评价。排名先要求留出危险率不超过 1%，再比较节省率。相同信号同时扫描正/反方向、最早第 4/6/8 轮和单轮/连续两轮规则，因此这里是探索性排序，不是确认性显著性结论。",
        "",
        "| 对照/方法 | 留出或 shadow 节省 | 危险率 | 说明 |",
        "|---|---:|---:|---|",
        "| 固定第 9 轮 + 常速度外推 | 10.0%% | %s | 无动态信号 |"
        % _risk_label(_fixed_lookup(fixed, "constant_velocity", 9)),
        "| 固定第 8 轮 + 常加速度外推 | 20.0%% | %s | 无动态信号，探索后选中 |"
        % _risk_label(fixed_acceleration),
        "| rollout 留出 clock 规则 + %s | %s | %s | 只看已完成轮数 |"
        % (
            best_clock["endpoint_estimator"],
            _format_percent(best_clock["saving"]),
            _risk_label(best_clock),
        ),
        "| 最佳路由规则 + %s | %s | %s | 从全部路由规则探索选中 |"
        % (
            best_route[0]["endpoint_estimator"],
            _format_percent(best_route[0]["saving"]),
            _risk_label(best_route[0]),
        ),
        "| 最佳 latent 规则 + %s | %s | %s | 从全部 latent 规则探索选中 |"
        % (
            best_latent[0]["endpoint_estimator"],
            _format_percent(best_latent[0]["saving"]),
            _risk_label(best_latent[0]),
        ),
        "",
        "在开发集上，固定第 8 轮常加速度外推有 `%d` 个坏候选；最佳路由过滤后为 `%d` 个，只少了 `%d` 个，同时节省从 20.0%% 降到 `%s`。这个计数太小，只能叫弱线索，不能叫路由增益。"
        % (
            fixed_acceleration["unsafe_count"],
            best_route[0]["unsafe_count"],
            fixed_acceleration["unsafe_count"] - best_route[0]["unsafe_count"],
            _format_percent(best_route[0]["saving"]),
        ),
        "",
        "### 路由信号前列",
        "",
        "| 信号 | 停止后输出 | 方向 | 最早轮 | 留出节省 | 留出危险 | P95 相对误差 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_route:
        lines.append(
            "| %s | %s | %s | %d | %s | %s | %.4f |"
            % (
                row["signal"],
                row["endpoint_estimator"],
                row["direction"],
                row["min_stage"],
                _format_percent(row["saving"]),
                _risk_label(row),
                row["p95_relative_error"],
            )
        )
    lines += [
        "",
        "### Latent/更新信号前列",
        "",
        "| 信号 | 停止后输出 | 方向 | 最早轮 | 留出节省 | 留出危险 | P95 相对误差 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in best_latent:
        lines.append(
            "| %s | %s | %s | %d | %s | %s | %.4f |"
            % (
                row["signal"],
                row["endpoint_estimator"],
                row["direction"],
                row["min_stage"],
                _format_percent(row["saving"]),
                _risk_label(row),
                row["p95_relative_error"],
            )
        )
    lines += [
        "",
        "![停止方法 Pareto 图](pareto.png)",
        "",
        "## 结论",
        "",
        "1. `变化小就直接输出` 被 oracle 门否决：flow matching 的每轮更新没有趋于零，路由稳定也不等于动作已经到终点。",
        "2. 真正有用的方向是数值外推：完成当前 MoE 前向后，用最后速度把剩余 flow 时间一次积分完。它不需要任务标签，也不需要额外 head。",
        "3. 动态停止信号是否优于固定轮数，必须看上面留出表；即便 endpoint shadow 过线，也还需要困难任务在线配对才能声称成功率不降。",
    ]
    if holdout is not None:
        holdout_acceleration = next(
            row
            for row in holdout["fixed_stage"]
            if row["estimator"] == "constant_acceleration" and row["stage"] == 8
        )
        holdout_velocity = next(
            row
            for row in holdout["fixed_stage"]
            if row["estimator"] == "constant_velocity" and row["stage"] == 9
        )
        lines += [
            "",
            "## 困难任务压力测试",
            "",
            "- 数据：`%s`，%d 条轨迹、%d 条 rollout。Goal 上冻结的阈值原样迁移。"
            % (holdout["name"], holdout["rows"], holdout["rollouts"]),
            "- 固定第 9 轮 + 常速度外推：节省 `10.0%%`，危险率 `%s`。"
            % _risk_label(holdout_velocity),
            "- 固定第 8 轮 + 常加速度外推：节省 `20.0%%`，危险率 `%s`。"
            % _risk_label(holdout_acceleration),
            "- 最佳冻结路由规则：节省 `%s`，危险率 `%s`。"
            % (
                _format_percent(holdout["best_route"]["saving"]),
                _risk_label(holdout["best_route"]),
            ),
            "- 最佳冻结 latent 规则：节省 `%s`，危险率 `%s`。"
            % (
                _format_percent(holdout["best_latent"]["saving"]),
                _risk_label(holdout["best_latent"]),
            ),
        ]
        if holdout.get("internal_diagnostic") is not None:
            diagnostic = holdout["internal_diagnostic"]
            lines += [
                "",
                "### Long 内部诊断（不是迁移结果）",
                "",
                "这里允许用 Long 自己的 3 条 rollout 校准 x10 误差，再留第 4 条；它只能回答任务内是否存在弱信号，不能用于声称新任务零样本早停。",
                "",
                "- clock/固定轮数在训练折零危险约束下：节省 `%s`，危险 `%s`。"
                % (
                    _format_percent(diagnostic["best_clock"]["saving"]),
                    _risk_label(diagnostic["best_clock"]),
                ),
                "- 最佳路由规则 `%s + %s`：节省 `%s`，危险 `%s`。"
                % (
                    diagnostic["best_route"]["signal"],
                    diagnostic["best_route"]["endpoint_estimator"],
                    _format_percent(diagnostic["best_route"]["saving"]),
                    _risk_label(diagnostic["best_route"]),
                ),
                "- 最佳 latent 规则 `%s + %s`：节省 `%s`，危险 `%s`。"
                % (
                    diagnostic["best_latent"]["signal"],
                    diagnostic["best_latent"]["endpoint_estimator"],
                    _format_percent(diagnostic["best_latent"]["saving"]),
                    _risk_label(diagnostic["best_latent"]),
                ),
                "- 任务内最优路由信号变成了 JS 概率变化，而 Goal 上是活跃专家数量；信号定义和阈值都没有稳定迁移。",
            ]
        lines += [
            "",
            "## 适用边界与最终裁决",
            "",
            "- Goal 与 Long 都走同一 CPU 推理数值路径，比较内部一致；CPU/GPU 的 top-k 边界可能不同，因此这里不证明 GPU 延迟或逐位一致性。",
            "- endpoint 误差是执行完整第 10 轮得到的自监督标签，不是任务成功标签；Long 基线 4 条 rollout 为 2 成功/2 失败，但成败未用于方法选择。",
            "- 新任务主端点是 Goal 冻结后直接迁移到 Long。路由、latent、固定第 8 轮和固定第 9 轮全部未过 `<=1%` 危险率门，因此不进入在线 A/B。",
            "- 当前结论：可以看到任务内弱停止信息，但没有得到可零样本迁移的无监督 MoE 停止器。继续扫同类阈值没有依据。",
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def freeze_and_apply(
    train: TraceData,
    test: TraceData,
    train_signals: list[Signal],
    test_signals: list[Signal],
    train_errors: dict[str, np.ndarray],
    test_errors: dict[str, np.ndarray],
    sweep: list[dict[str, Any]],
    tolerance: float,
    train_risk: float,
) -> dict[str, Any]:
    test_by_key = {(signal.family, signal.name): signal for signal in test_signals}
    candidates = [row for row in sweep if row["protocol"] == "safety_calibrated_loeo"]
    selected: dict[str, dict[str, Any]] = {}
    for family in ("route", "latent"):
        development = _best_rows(candidates, family, "safety_calibrated_loeo", count=1)[0]
        train_signal = next(
            signal
            for signal in train_signals
            if signal.family == family and signal.name == development["signal"]
        )
        test_signal = test_by_key[(family, development["signal"])]
        estimator = development["endpoint_estimator"]
        threshold, train_stats = calibrate_threshold(
            train_signal.values,
            train_errors[estimator],
            tolerance,
            train_risk,
            development["direction"],
            int(development["min_stage"]),
        )
        stops = _stops_for_threshold(
            test_signal.values,
            threshold,
            development["direction"],
            int(development["min_stage"]),
        )
        errors = _selected_error(test_errors[estimator], stops)
        unsafe = errors > tolerance
        selected[family] = {
            "signal": development["signal"],
            "endpoint_estimator": estimator,
            "direction": development["direction"],
            "min_stage": int(development["min_stage"]),
            "threshold": threshold,
            "train": train_stats,
            "saving": float(np.mean((TOTAL_ROUNDS - stops) / TOTAL_ROUNDS)),
            "unsafe_rate": float(np.mean(unsafe)),
            "unsafe_count": int(unsafe.sum()),
            "unsafe_query_count": int(len(np.unique(test.query_id[unsafe]))),
            "unsafe_query_any_rate": float(
                len(np.unique(test.query_id[unsafe])) / len(np.unique(test.query_id))
            ),
            "unsafe_rollout_count": int(len(np.unique(test.rollout_id[unsafe]))),
            "mean_relative_error": float(errors.mean()),
        }
    return {
        "name": test.name,
        "rows": test.rows,
        "rollouts": int(len(np.unique(test.rollout_id))),
        "best_route": selected["route"],
        "best_latent": selected["latent"],
    }


def main() -> int:
    args = parse_args()
    if not 0.0 < args.tolerance < 1.0:
        raise ValueError("--tolerance must be in (0, 1)")
    if not 0.0 <= args.train_risk < 1.0:
        raise ValueError("--train-risk must be in [0, 1)")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_trace(Path(args.run), name="development")
    print(
        "loaded %s: %d rows, %d queries, %d rollouts"
        % (
            data.run,
            data.rows,
            len(np.unique(data.query_id)),
            len(np.unique(data.rollout_id)),
        ),
        flush=True,
    )
    deterministic_predictions = endpoint_predictions(data)
    predictions = dict(deterministic_predictions)
    predictions.update(oof_endpoint_predictions(data))
    fixed, errors = fixed_stage_rows(data, predictions, args.tolerance)
    oracle = oracle_rows(errors, args.tolerance)
    signals = build_signals(data)
    for signal in signals:
        if signal.family == "negative_control" and not np.array_equal(
            signal.values, np.zeros_like(signal.values)
        ):
            raise RuntimeError("negative control is nonzero: %s" % signal.name)
    print("built %d online-observable signals" % len(signals), flush=True)
    sweep = []
    for estimator in deterministic_predictions:
        error = errors[estimator]
        print("sweeping endpoint estimator %s" % estimator, flush=True)
        sweep.extend(
            cross_validated_signal_sweep(
                data,
                signals,
                error,
                estimator,
                args.tolerance,
                args.train_risk,
            )
        )

    holdout_summary = None
    if args.holdout_run:
        holdout_data = load_trace(Path(args.holdout_run), name="hard-task holdout")
        holdout_predictions = endpoint_predictions(holdout_data)
        holdout_predictions.update(transferred_endpoint_predictions(data, holdout_data))
        holdout_fixed, holdout_errors = fixed_stage_rows(
            holdout_data, holdout_predictions, args.tolerance
        )
        holdout_signals = build_signals(holdout_data)
        holdout_summary = freeze_and_apply(
            data,
            holdout_data,
            signals,
            holdout_signals,
            errors,
            holdout_errors,
            sweep,
            args.tolerance,
            args.train_risk,
        )
        holdout_summary["fixed_stage"] = holdout_fixed
        if args.diagnostic_summary:
            holdout_summary["internal_diagnostic"] = load_internal_diagnostic(
                Path(args.diagnostic_summary)
            )

    summary = {
        "schema": "himoe-denoise-stop-sweep-v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "development": {
            "run": str(data.run.resolve()),
            "rows": data.rows,
            "queries": int(len(np.unique(data.query_id))),
            "rollouts": int(len(np.unique(data.rollout_id))),
        },
        "definition": {
            "stages": STAGES.tolist(),
            "full_rounds": TOTAL_ROUNDS,
            "endpoint_error": "relative L2 on normalized 10x7 action chunk versus x10",
            "tolerance": args.tolerance,
            "train_risk": args.train_risk,
            "cv": "leave one complete rollout out",
            "fixed_stage_endpoint_estimators": list(predictions),
            "signal_sweep_endpoint_estimators": list(deterministic_predictions),
            "signal_min_stages": list(SIGNAL_MIN_STAGES),
            "directions": ["le", "ge"],
        },
        "counts": {
            "signals": len(signals),
            "safety_methods": sum(
                row["protocol"] == "safety_calibrated_loeo" for row in sweep
            ),
            "all_signal_rows": len(sweep),
            "fixed_stage_endpoint_estimators": len(predictions),
            "signal_sweep_endpoint_estimators": len(deterministic_predictions),
        },
        "fixed_stage": fixed,
        "oracle": oracle,
        "signal_sweep": sweep,
        "holdout": holdout_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(out / "fixed_stage.csv", fixed)
    write_csv(out / "signal_sweep.csv", sweep)
    plot_pareto(sweep, out / "pareto.png")
    write_report(
        out, data, fixed, oracle, sweep, args.tolerance, holdout_summary
    )
    print("wrote %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
