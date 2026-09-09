#!/usr/bin/env python3
"""Leakage-safe HB5/d0 runtime-vector readout for activation-flow v3.

This is an inference pipeline, not a capture path.  It deliberately rejects
single-pool/K1 smoke data and only reports held-out state-and-seed predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import zarr
from scipy.stats import spearmanr


FORMAT = "himoe_hb_activation_flow_v3"
N_ACTION_TOKENS = 10
N_LIVE_DIMS = 7
N_EXPERTS = 32
PROBE_LAYER = 5
PROBE_DENOISE = 0
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_PROJECTION_SEEDS = (20260823, 20260824, 20260825, 20260826, 20260827)
EXPECTED_TASKS = frozenset(
    {
        "libero_goal:0",
        "libero_goal:3",
        "libero_10:8",
        "libero_spatial:5",
        "libero_spatial:7",
    }
)
N_STATES_PER_TASK = 8
N_SEEDS_PER_TASK = 16
X0_REPEAT_ATOL = 1e-6
WEIGHT_SUM_ERROR_LIMIT = 0.02
BOOTSTRAP_DRAWS = 5000

BASELINE_TIERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("b0_x0_x1", ("x0", "x1")),
    ("b0_plus_hidden", ("x0", "x1", "hb5_input_hidden")),
    (
        "b0_plus_hidden_shared",
        ("x0", "x1", "hb5_input_hidden", "hb5_shared_output"),
    ),
    (
        "b1_operational",
        (
            "x0",
            "x1",
            "hb5_input_hidden",
            "hb5_shared_output",
            "hb5_router_probs",
            "hb5_selected_id_one_hot",
            "hb5_selected_weight_by_id",
            "hb5_actual_merged_routed",
        ),
    ),
)
MOE_VECTOR_FAMILY = "hb5_selected_raw_by_expert_id"
MOE_GEOMETRY_FAMILY = "hb5_selected_raw_rms_by_expert_id"
MOE_CONTRIBUTION_FAMILY = "hb5_weighted_contribution_by_expert_id"
ID_AWARE_EXPERT_FAMILIES = (
    MOE_VECTOR_FAMILY,
    MOE_GEOMETRY_FAMILY,
    MOE_CONTRIBUTION_FAMILY,
)
INVARIANT_GEOMETRY_FAMILY = "hb5_expert_geometry_permutation_invariant"
S_LOO_FAMILY = "hb5_drop_one_s_loo_absolute_permutation_invariant"
INVARIANT_EXPERT_FAMILIES = (S_LOO_FAMILY,)
GRAM_EXPERT_FAMILIES = (INVARIANT_GEOMETRY_FAMILY,)
ROUTED_RMS_CONTROL_FAMILY = "hb5_runtime_routed_rms_exact_control"


class AdmissionError(ValueError):
    """The capture does not support the requested scientific inference."""


@dataclass(frozen=True)
class RuntimeVectorData:
    x_traj: np.ndarray
    input_hidden: np.ndarray
    shared_output: np.ndarray
    router_probs: np.ndarray
    selected_ids: np.ndarray
    selected_weights: np.ndarray
    selected_raw: np.ndarray
    state_group: np.ndarray
    seed_group: np.ndarray
    task_group: np.ndarray
    checkpoint_group: np.ndarray
    sample_id: np.ndarray

    @property
    def target(self) -> np.ndarray:
        return future_path_energy_target(self.x_traj)

    @property
    def secondary_target(self) -> np.ndarray:
        return remaining_correction_target(self.x_traj)


@dataclass(frozen=True)
class GroupFold:
    train: np.ndarray
    test: np.ndarray
    held_states: tuple[str, ...]
    held_seeds: tuple[str, ...]


def remaining_correction_target(x_traj: np.ndarray) -> np.ndarray:
    """Secondary target: flattened live-7 ``x10 - x1`` net correction."""
    x = np.asarray(x_traj)
    if x.ndim != 4 or x.shape[1] != 11:
        raise ValueError("x_traj must be [sample,11,10,dim]")
    if x.shape[2] != N_ACTION_TOKENS or x.shape[3] < N_LIVE_DIMS:
        raise ValueError("x_traj must contain all ten action tokens and live7")
    target = x[:, 10, :, :N_LIVE_DIMS] - x[:, 1, :, :N_LIVE_DIMS]
    return np.asarray(target, dtype=np.float64).reshape(len(x), 70)


def future_path_energy_target(x_traj: np.ndarray) -> np.ndarray:
    """Primary target: live-7 RMS path energy over transitions tau=1..9."""
    x = np.asarray(x_traj)
    if (
        x.ndim != 4
        or x.shape[1] != 11
        or x.shape[2] != N_ACTION_TOKENS
        or x.shape[3] < N_LIVE_DIMS
    ):
        raise ValueError("x_traj must be [sample,11,10,dim>=7]")
    step = x[:, 2:11, :, :N_LIVE_DIMS] - x[:, 1:10, :, :N_LIVE_DIMS]
    return np.sqrt(np.mean(np.square(step), axis=(1, 2, 3)))[:, None]


def immediate_control_target(x_traj: np.ndarray) -> np.ndarray:
    """Instrumentation-only positive control: flattened live-7 ``x1 - x0``."""
    x = np.asarray(x_traj)
    if (
        x.ndim != 4
        or x.shape[1] != 11
        or x.shape[2] != N_ACTION_TOKENS
        or x.shape[3] < N_LIVE_DIMS
    ):
        raise ValueError("x_traj must be [sample,11,10,dim>=7]")
    value = x[:, 1, :, :N_LIVE_DIMS] - x[:, 0, :, :N_LIVE_DIMS]
    return np.asarray(value, dtype=np.float64).reshape(len(x), 70)


def scatter_selected_by_expert(
    expert_ids: np.ndarray,
    values: np.ndarray,
    n_experts: int = N_EXPERTS,
) -> np.ndarray:
    """Scatter selected scalar values into expert-ID coordinates, summing repeats."""
    ids = np.asarray(expert_ids, dtype=np.int64)
    value = np.asarray(values)
    if ids.shape != value.shape or ids.ndim != 3:
        raise ValueError("expert IDs and values must share [sample,token,slot]")
    if np.any(ids < 0) or np.any(ids >= n_experts):
        raise ValueError("expert ID is outside the routed-expert range")
    output = np.zeros((*ids.shape[:2], n_experts), dtype=value.dtype)
    sample, token, _slot = np.indices(ids.shape)
    np.add.at(output, (sample, token, ids), value)
    return output


def scatter_expert_vectors(
    expert_ids: np.ndarray,
    raw_vectors: np.ndarray,
    n_experts: int = N_EXPERTS,
) -> np.ndarray:
    """Represent raw vectors by token and expert identity, never by top-k slot."""
    ids = np.asarray(expert_ids, dtype=np.int64)
    raw = np.asarray(raw_vectors)
    if ids.ndim != 3 or raw.shape[:3] != ids.shape or raw.ndim != 4:
        raise ValueError("raw vectors must align as [sample,token,slot,hidden]")
    if np.any(ids < 0) or np.any(ids >= n_experts):
        raise ValueError("expert ID is outside the routed-expert range")
    output = np.zeros((*ids.shape[:2], n_experts, raw.shape[-1]), dtype=raw.dtype)
    for sample in range(ids.shape[0]):
        for token in range(ids.shape[1]):
            np.add.at(output[sample, token], ids[sample, token], raw[sample, token])
    return output


def baseline_tiers() -> dict[str, tuple[str, ...]]:
    """Return an insertion-ordered copy of the frozen nested baselines."""
    return dict(BASELINE_TIERS)


def _stable_namespace(name: str, projection_seed: int) -> int:
    digest = hashlib.sha256(f"{projection_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _hash_coordinates(
    coordinates: np.ndarray, width: int, namespace: int
) -> tuple[np.ndarray, np.ndarray]:
    if width < 1:
        raise ValueError("projection width must be positive")
    value = np.asarray(coordinates, dtype=np.uint64) + np.uint64(namespace)
    with np.errstate(over="ignore"):
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    bucket = np.asarray(value % np.uint64(width), dtype=np.int64)
    sign = np.where((value >> np.uint64(63)) == 0, 1.0, -1.0)
    return bucket, sign


def countsketch_dense(
    values: np.ndarray, width: int, namespace: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    bucket, sign = _hash_coordinates(np.arange(array.shape[1]), width, namespace)
    output = np.zeros((len(array), width), dtype=np.float64)
    scale = np.sqrt(max(array.shape[1], 1))
    for sample in range(len(array)):
        np.add.at(output[sample], bucket, array[sample] * sign / scale)
    return output


def countsketch_expert_raw(
    expert_ids: np.ndarray,
    raw_vectors: np.ndarray,
    width: int,
    namespace: int,
    n_experts: int = N_EXPERTS,
) -> np.ndarray:
    """Sketch raw vectors using ``(token, expert_id, hidden)`` coordinates."""
    ids = np.asarray(expert_ids, dtype=np.int64)
    raw = np.asarray(raw_vectors, dtype=np.float64)
    if ids.ndim != 3 or raw.ndim != 4 or raw.shape[:3] != ids.shape:
        raise ValueError("raw vectors must align as [sample,token,slot,hidden]")
    if np.any(ids < 0) or np.any(ids >= n_experts):
        raise ValueError("expert ID is outside the routed-expert range")
    output = np.zeros((len(ids), width), dtype=np.float64)
    hidden = raw.shape[-1]
    hidden_axis = np.arange(hidden, dtype=np.int64)
    scale = np.sqrt(ids.shape[1] * ids.shape[2] * hidden)
    for token in range(ids.shape[1]):
        for slot in range(ids.shape[2]):
            coordinate = (
                (token * n_experts + ids[:, token, slot, None]) * hidden
                + hidden_axis[None]
            )
            bucket, sign = _hash_coordinates(coordinate, width, namespace)
            for sample in range(len(ids)):
                np.add.at(
                    output[sample],
                    bucket[sample],
                    raw[sample, token, slot] * sign[sample] / scale,
                )
    return output


def expert_output_blocks(
    raw_vectors: np.ndarray, selected_weights: np.ndarray
) -> dict[str, np.ndarray]:
    """Construct frozen A/B/C expert features before any random projection."""
    raw = np.asarray(raw_vectors, dtype=np.float64)
    weight = np.asarray(selected_weights, dtype=np.float64)
    if raw.ndim != 4 or weight.shape != raw.shape[:3]:
        raise ValueError("raw vectors and weights must align [sample,token,slot,hidden]")
    raw_rms = np.sqrt(np.mean(np.square(raw), axis=-1))
    weight_sum = weight.sum(axis=-1, keepdims=True)
    if np.any(weight_sum <= 0.0):
        raise AdmissionError("selected expert weights have zero total mass")
    normalized_weight = weight / weight_sum
    contribution = weight[..., None] * raw
    normalized_contribution = normalized_weight[..., None] * raw
    routed = np.sum(contribution, axis=2)
    normalized_routed = np.sum(normalized_contribution, axis=2)
    hidden_width = raw.shape[-1]
    raw_gram = np.einsum("ntkh,ntlh->ntkl", raw, raw, optimize=True) / hidden_width
    contribution_gram = (
        np.einsum(
            "ntkh,ntlh->ntkl",
            normalized_contribution,
            normalized_contribution,
            optimize=True,
        )
        / hidden_width
    )
    routed_rms = np.sqrt(np.mean(np.square(normalized_routed), axis=-1))
    c_abs = np.maximum(
        np.sum(normalized_weight * raw_rms, axis=-1) - routed_rms, 0.0
    )
    denominator = 1.0 - normalized_weight
    if np.any(denominator <= 1e-6):
        raise AdmissionError("drop-one sensitivity is undefined for expert weight near one")
    drop_one = (
        (normalized_routed[:, :, None] - normalized_contribution)
        / denominator[..., None]
        - normalized_routed[:, :, None]
    )
    drop_one_rms = np.sqrt(np.mean(np.square(drop_one), axis=-1))
    s_loo_squared = np.sum(normalized_weight * np.square(drop_one_rms), axis=-1)
    s_loo = np.sqrt(s_loo_squared)
    s_loo_features = np.concatenate(
        (
            s_loo,
            np.sqrt(np.mean(s_loo_squared, axis=-1, keepdims=True)),
        ),
        axis=1,
    )
    gram_geometry = np.concatenate(
        (
            np.maximum(np.linalg.eigvalsh(raw_gram), 0.0)[..., ::-1],
            np.maximum(np.linalg.eigvalsh(contribution_gram), 0.0)[..., ::-1],
            routed_rms[..., None],
            c_abs[..., None],
            s_loo[..., None],
        ),
        axis=-1,
    )
    return {
        "raw_rms": raw_rms,
        "contribution": contribution,
        "normalized_contribution": normalized_contribution,
        "routed": routed,
        "normalized_routed": normalized_routed,
        "normalized_weights": normalized_weight,
        "drop_one": drop_one,
        "s_loo_squared": s_loo_squared,
        "s_loo": s_loo,
        "s_loo_features": s_loo_features,
        "gram_geometry": gram_geometry,
    }


def project_feature_families(
    data: RuntimeVectorData, width: int, projection_seed: int
) -> dict[str, np.ndarray]:
    selected_one = scatter_selected_by_expert(
        data.selected_ids, np.ones_like(data.selected_weights)
    )
    selected_weight = scatter_selected_by_expert(
        data.selected_ids, data.selected_weights
    )
    expert = expert_output_blocks(data.selected_raw, data.selected_weights)
    raw_rms = expert["raw_rms"]
    contribution = expert["contribution"]
    routed_vector = expert["routed"]
    runtime_routed_rms = np.sqrt(np.mean(np.square(routed_vector), axis=-1))
    routed_rms_control = np.concatenate(
        (
            runtime_routed_rms,
            np.sqrt(
                np.mean(np.square(runtime_routed_rms), axis=-1, keepdims=True)
            ),
        ),
        axis=1,
    )
    dense = {
        "x0": data.x_traj[:, 0],
        "x1": data.x_traj[:, 1],
        "hb5_input_hidden": data.input_hidden,
        "hb5_shared_output": data.shared_output,
        "hb5_router_probs": data.router_probs,
        "hb5_selected_id_one_hot": selected_one,
        "hb5_selected_weight_by_id": selected_weight,
        "hb5_actual_merged_routed": routed_vector,
        MOE_GEOMETRY_FAMILY: scatter_selected_by_expert(
            data.selected_ids,
            raw_rms,
        ),
        INVARIANT_GEOMETRY_FAMILY: expert["gram_geometry"],
        S_LOO_FAMILY: expert["s_loo_features"],
        ROUTED_RMS_CONTROL_FAMILY: routed_rms_control,
    }
    exact_families = {
        INVARIANT_GEOMETRY_FAMILY,
        S_LOO_FAMILY,
        ROUTED_RMS_CONTROL_FAMILY,
    }
    projected = {
        name: countsketch_dense(
            value, width, _stable_namespace(name, projection_seed)
        )
        for name, value in dense.items()
        if name not in exact_families
    }
    projected[INVARIANT_GEOMETRY_FAMILY] = np.asarray(
        dense[INVARIANT_GEOMETRY_FAMILY], dtype=np.float64
    ).reshape(len(data.x_traj), -1)
    projected[S_LOO_FAMILY] = np.asarray(
        dense[S_LOO_FAMILY], dtype=np.float64
    ).reshape(len(data.x_traj), -1)
    projected[ROUTED_RMS_CONTROL_FAMILY] = np.asarray(
        dense[ROUTED_RMS_CONTROL_FAMILY], dtype=np.float64
    ).reshape(len(data.x_traj), -1)
    projected[MOE_VECTOR_FAMILY] = countsketch_expert_raw(
        data.selected_ids,
        data.selected_raw,
        width,
        _stable_namespace(MOE_VECTOR_FAMILY, projection_seed),
    )
    projected[MOE_CONTRIBUTION_FAMILY] = countsketch_expert_raw(
        data.selected_ids,
        contribution,
        width,
        _stable_namespace(MOE_CONTRIBUTION_FAMILY, projection_seed),
    )
    return projected


def state_seed_disjoint_folds(
    state_group: np.ndarray,
    seed_group: np.ndarray,
    n_splits: int,
    split_seed: int,
) -> list[GroupFold]:
    """Cartesian folds whose train rows share no state or seed with test rows."""
    states = np.asarray(state_group).astype(str)
    seeds = np.asarray(seed_group).astype(str)
    if states.ndim != 1 or seeds.shape != states.shape:
        raise ValueError("state_group and seed_group must be aligned vectors")
    unique_states = np.unique(states)
    unique_seeds = np.unique(seeds)
    if n_splits < 2 or len(unique_states) < n_splits or len(unique_seeds) < n_splits:
        raise ValueError("not enough state and seed groups for requested folds")
    rng = np.random.default_rng(split_seed)
    state_bins = np.array_split(rng.permutation(unique_states), n_splits)
    seed_bins = np.array_split(rng.permutation(unique_seeds), n_splits)
    folds = []
    coverage = np.zeros(len(states), dtype=np.int64)
    for held_states_array in state_bins:
        for held_seeds_array in seed_bins:
            test = np.flatnonzero(
                np.isin(states, held_states_array) & np.isin(seeds, held_seeds_array)
            )
            train = np.flatnonzero(
                ~np.isin(states, held_states_array) & ~np.isin(seeds, held_seeds_array)
            )
            if not len(test):
                continue
            if not len(train):
                raise AdmissionError("a state+seed fold has no training rows")
            coverage[test] += 1
            folds.append(
                GroupFold(
                    train=train,
                    test=test,
                    held_states=tuple(map(str, held_states_array)),
                    held_seeds=tuple(map(str, held_seeds_array)),
                )
            )
    if not np.all(coverage == 1):
        raise RuntimeError("state+seed Cartesian folds did not cover every row once")
    return folds


def repeated_x0_max_error(data: RuntimeVectorData) -> float:
    seeds = np.asarray(data.seed_group).astype(str)
    max_error = 0.0
    for seed in np.unique(seeds):
        repeated = np.asarray(data.x_traj[seeds == seed, 0], dtype=np.float64)
        max_error = max(max_error, float(np.max(np.abs(repeated - repeated[:1]))))
    return max_error


def validate_admission(
    data: RuntimeVectorData, outer_splits: int = 3
) -> dict[str, Any]:
    n = len(data.x_traj)
    arrays = (
        data.input_hidden,
        data.shared_output,
        data.router_probs,
        data.selected_ids,
        data.selected_weights,
        data.selected_raw,
        data.state_group,
        data.seed_group,
        data.task_group,
        data.checkpoint_group,
        data.sample_id,
    )
    if n < 1 or any(len(value) != n for value in arrays):
        raise AdmissionError("runtime-vector arrays do not share a non-empty sample axis")
    states = np.asarray(data.state_group).astype(str)
    seeds = np.asarray(data.seed_group).astype(str)
    n_states, n_seeds = len(np.unique(states)), len(np.unique(seeds))
    required = max(3, outer_splits)
    if n_states < required or n_seeds < required:
        raise AdmissionError(
            "scientific inference requires at least "
            f"{required} independent state pools and {required} seed groups; "
            f"found {n_states} state pool(s), {n_seeds} seed group(s), {n} row(s). "
            "K1/v3 smoke captures are schema checks only."
        )
    state_seed_counts = {
        state: len(np.unique(seeds[states == state])) for state in np.unique(states)
    }
    seed_state_counts = {
        seed: len(np.unique(states[seeds == seed])) for seed in np.unique(seeds)
    }
    if min(state_seed_counts.values()) < required or min(seed_state_counts.values()) < required:
        raise AdmissionError(
            "every state must cross at least three seed groups and every seed must "
            "cross at least three states for state+seed-disjoint inference"
        )
    if not np.all(np.isfinite(data.target)):
        raise AdmissionError("primary future-path-energy target contains non-finite values")
    if not np.all(np.isfinite(data.secondary_target)):
        raise AdmissionError("secondary x10-x1 target contains non-finite values")
    for name in (
        "input_hidden",
        "shared_output",
        "router_probs",
        "selected_weights",
        "selected_raw",
    ):
        if not np.all(np.isfinite(getattr(data, name))):
            raise AdmissionError(f"{name} contains non-finite values")
    if np.any(data.selected_ids < 0) or np.any(data.selected_ids >= N_EXPERTS):
        raise AdmissionError("selected expert ID is outside 0..31")
    weight_sum_error = float(
        np.max(np.abs(np.sum(data.selected_weights, axis=-1) - 1.0))
    )
    if weight_sum_error > WEIGHT_SUM_ERROR_LIMIT:
        raise AdmissionError(
            f"selected weight-sum error {weight_sum_error:.3g} exceeds "
            f"{WEIGHT_SUM_ERROR_LIMIT:.3g}"
        )
    max_x0_repeat_error = repeated_x0_max_error(data)
    if max_x0_repeat_error > X0_REPEAT_ATOL:
        raise AdmissionError(
            "seed_group does not identify a repeated flow-noise draw: "
            f"max x0 disagreement {max_x0_repeat_error:.3g} exceeds "
            f"{X0_REPEAT_ATOL:.1g}"
        )
    folds = state_seed_disjoint_folds(states, seeds, outer_splits, 0)
    for fold in folds:
        train_states = states[fold.train]
        train_seeds = seeds[fold.train]
        inner_splits = min(
            max(2, outer_splits - 1),
            len(np.unique(train_states)),
            len(np.unique(train_seeds)),
        )
        if inner_splits < 2:
            raise AdmissionError("outer training data cannot support nested group-disjoint tuning")
        state_seed_disjoint_folds(train_states, train_seeds, inner_splits, 1)
    return {
        "admitted": True,
        "rows": n,
        "state_groups": n_states,
        "seed_groups": n_seeds,
        "outer_folds": len(folds),
        "minimum_seed_groups_per_state": min(state_seed_counts.values()),
        "minimum_state_groups_per_seed": min(seed_state_counts.values()),
        "max_x0_repeat_error_within_seed_group": max_x0_repeat_error,
        "x0_repeat_atol": X0_REPEAT_ATOL,
        "max_abs_runtime_weight_sum_minus_one": weight_sum_error,
        "weight_sum_error_limit": WEIGHT_SUM_ERROR_LIMIT,
        "expert_probability_geometry_uses_renormalized_weights": True,
    }


def _design(
    projected: dict[str, np.ndarray],
    feature_blocks: Sequence[Sequence[str]],
    train: np.ndarray,
    apply: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not feature_blocks or any(not block for block in feature_blocks):
        raise ValueError("every readout feature block must be non-empty")
    train_blocks, apply_blocks = [], []
    for block in feature_blocks:
        train_parts, apply_parts = [], []
        for family in block:
            value = np.asarray(projected[family], dtype=np.float64)
            mean = value[train].mean(axis=0)
            scale = value[train].std(axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            train_parts.append((value[train] - mean) / scale)
            apply_parts.append((value[apply] - mean) / scale)
        normalizer = np.sqrt(len(block))
        train_blocks.append(sum(train_parts) / normalizer)
        apply_blocks.append(sum(apply_parts) / normalizer)
    return np.concatenate(train_blocks, axis=1), np.concatenate(apply_blocks, axis=1)


def _ridge_predict(
    projected: dict[str, np.ndarray],
    feature_blocks: Sequence[Sequence[str]],
    targets: np.ndarray,
    train: np.ndarray,
    apply: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_train, x_apply = _design(projected, feature_blocks, train, apply)
    y_train = np.asarray(targets[train], dtype=np.float64)
    y_mean = y_train.mean(axis=0)
    centered_target = y_train - y_mean
    if len(train) < x_train.shape[1]:
        kernel = x_train @ x_train.T / len(train)
        dual = np.linalg.solve(
            kernel + float(alpha) * np.eye(len(train)), centered_target
        )
        return (x_apply @ x_train.T / len(train)) @ dual + y_mean
    gram = x_train.T @ x_train / len(train)
    rhs = x_train.T @ centered_target / len(train)
    coefficient = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), rhs)
    return x_apply @ coefficient + y_mean


def cross_validated_predictions(
    projected: dict[str, np.ndarray],
    feature_blocks: Sequence[Sequence[str]],
    targets: np.ndarray,
    state_group: np.ndarray,
    seed_group: np.ndarray,
    outer_splits: int,
    split_seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    outer = state_seed_disjoint_folds(
        state_group, seed_group, outer_splits, split_seed
    )
    prediction = np.full_like(targets, np.nan, dtype=np.float64)
    prediction_fold_id = np.full(len(targets), -1, dtype=np.int64)
    fold_rows = []
    for fold_index, fold in enumerate(outer):
        train_states = np.asarray(state_group)[fold.train]
        train_seeds = np.asarray(seed_group)[fold.train]
        inner_count = min(
            max(2, outer_splits - 1),
            len(np.unique(train_states)),
            len(np.unique(train_seeds)),
        )
        inner_local = state_seed_disjoint_folds(
            train_states, train_seeds, inner_count, split_seed + 1009 + fold_index
        )
        losses = []
        for alpha in RIDGE_ALPHAS:
            squared_error = 0.0
            elements = 0
            for inner in inner_local:
                inner_train = fold.train[inner.train]
                inner_test = fold.train[inner.test]
                inner_prediction = _ridge_predict(
                    projected,
                    feature_blocks,
                    targets,
                    inner_train,
                    inner_test,
                    alpha,
                )
                squared_error += float(
                    np.sum(np.square(inner_prediction - targets[inner_test]))
                )
                elements += int(np.prod(targets[inner_test].shape))
            losses.append(squared_error / elements)
        selected_alpha = float(RIDGE_ALPHAS[int(np.argmin(losses))])
        prediction[fold.test] = _ridge_predict(
            projected,
            feature_blocks,
            targets,
            fold.train,
            fold.test,
            selected_alpha,
        )
        prediction_fold_id[fold.test] = fold_index
        fold_rows.append(
            {
                "fold": fold_index,
                "train_rows": len(fold.train),
                "test_rows": len(fold.test),
                "held_states": list(fold.held_states),
                "held_seeds": list(fold.held_seeds),
                "selected_alpha": selected_alpha,
            }
        )
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-fitting did not produce one finite prediction per row")
    if np.any(prediction_fold_id < 0):
        raise RuntimeError("cross-fitting did not assign every row to an outer fold")
    return prediction, fold_rows, prediction_fold_id


def _pairwise_rms(value: np.ndarray) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(len(value), -1)
    delta = flat[:, None] - flat[None, :]
    return np.sqrt(np.mean(np.square(delta), axis=-1))


def _pam(distance: np.ndarray, keep: int) -> np.ndarray:
    matrix = np.asarray(distance, dtype=np.float64)
    if keep < 1 or keep > len(matrix):
        raise ValueError("PAM keep must fit the candidate count")
    medoids: list[int] = []
    current = np.full(len(matrix), np.inf)
    for _ in range(keep):
        costs = [
            (float(np.minimum(current, matrix[:, candidate]).sum()), candidate)
            for candidate in range(len(matrix))
            if candidate not in medoids
        ]
        _cost, selected = min(costs)
        medoids.append(selected)
        current = np.minimum(current, matrix[:, selected])
    while True:
        base = float(np.min(matrix[:, medoids], axis=1).sum())
        best_cost = base
        best_swap: tuple[int, int] | None = None
        for old in medoids:
            for candidate in range(len(matrix)):
                if candidate in medoids:
                    continue
                proposed = medoids.copy()
                proposed[proposed.index(old)] = candidate
                cost = float(np.min(matrix[:, proposed], axis=1).sum())
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best_swap = (old, candidate)
        if best_swap is None:
            break
        medoids[medoids.index(best_swap[0])] = best_swap[1]
    return np.asarray(sorted(medoids), dtype=np.int64)


def k8_geometry_coverage(
    prediction: np.ndarray,
    target: np.ndarray,
    state_group: np.ndarray,
    prediction_fold_id: np.ndarray,
) -> dict[str, Any]:
    states = np.asarray(state_group).astype(str)
    fold_ids = np.asarray(prediction_fold_id, dtype=np.int64)
    ratios, selected_coverages, target_pam_coverages = [], [], []
    per_state = {}
    for state in np.unique(states):
        index = np.flatnonzero(states == state)
        if len(index) < 8:
            continue
        predicted_distance = _pairwise_rms(prediction[index])
        target_distance = _pairwise_rms(target[index])
        local_fold = fold_ids[index]
        groups = [np.flatnonzero(local_fold == fold) for fold in np.unique(local_fold)]
        sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
        raw_allocation = 8.0 * sizes / sizes.sum()
        allocation = np.floor(raw_allocation).astype(np.int64)
        remaining = 8 - int(allocation.sum())
        remainder_order = np.argsort(
            -(raw_allocation - allocation), kind="stable"
        )
        allocation[remainder_order[:remaining]] += 1
        if np.any(allocation > sizes):
            raise RuntimeError("stratified K8 allocation exceeds a seed-fold block")
        selected_parts, target_pam_parts = [], []
        for group, keep in zip(groups, allocation, strict=True):
            if keep == 0:
                continue
            selected_parts.append(group[_pam(predicted_distance[np.ix_(group, group)], int(keep))])
            target_pam_parts.append(group[_pam(target_distance[np.ix_(group, group)], int(keep))])
        selected = np.concatenate(selected_parts)
        target_pam = np.concatenate(target_pam_parts)
        selected_coverage = float(np.min(target_distance[:, selected], axis=1).mean())
        target_pam_coverage = float(
            np.min(target_distance[:, target_pam], axis=1).mean()
        )
        selected_coverages.append(selected_coverage)
        target_pam_coverages.append(target_pam_coverage)
        ratios.append(
            1.0
            if target_pam_coverage <= 1e-15
            else selected_coverage / target_pam_coverage
        )
        per_state[state] = {
            "seed_fold_sizes": sizes.tolist(),
            "stratified_k8_allocation": allocation.tolist(),
            "selected_target_coverage": selected_coverage,
            "target_pam_coverage": target_pam_coverage,
        }
    return {
        "states_with_at_least_k8": len(ratios),
        "macro_selected_to_target_pam_coverage_ratio": (
            float(np.mean(ratios)) if ratios else None
        ),
        "macro_selected_target_coverage": (
            float(np.mean(selected_coverages)) if selected_coverages else None
        ),
        "macro_target_pam_coverage": (
            float(np.mean(target_pam_coverages)) if target_pam_coverages else None
        ),
        "reference": "deterministic PAM BUILD+SWAP on actual final-action geometry; not exact oracle",
        "prediction_comparison_rule": (
            "largest-remainder K8 allocation across held-seed-fold blocks; PAM "
            "selection only within rows predicted by the same model"
        ),
        "per_state": per_state,
    }


def geometry_spearman(
    prediction: np.ndarray,
    target: np.ndarray,
    state_group: np.ndarray,
    prediction_fold_id: np.ndarray,
) -> dict[str, Any]:
    predicted_distances, target_distances = [], []
    per_state: dict[str, float | None] = {}
    states = np.asarray(state_group).astype(str)
    fold_ids = np.asarray(prediction_fold_id, dtype=np.int64)
    for state in np.unique(states):
        index = np.flatnonzero(states == state)
        state_predicted, state_actual = [], []
        for fold in np.unique(fold_ids[index]):
            block = index[fold_ids[index] == fold]
            if len(block) < 2:
                continue
            upper = np.triu_indices(len(block), 1)
            state_predicted.append(_pairwise_rms(prediction[block])[upper])
            state_actual.append(_pairwise_rms(target[block])[upper])
        if not state_predicted:
            per_state[state] = None
            continue
        predicted = np.concatenate(state_predicted)
        actual = np.concatenate(state_actual)
        correlation = (
            float("nan")
            if np.ptp(predicted) <= 1e-15 or np.ptp(actual) <= 1e-15
            else float(spearmanr(predicted, actual).statistic)
        )
        per_state[state] = correlation if np.isfinite(correlation) else None
        predicted_distances.append(predicted)
        target_distances.append(actual)
    valid = [value for value in per_state.values() if value is not None]
    if not predicted_distances:
        pooled = None
    else:
        predicted_all = np.concatenate(predicted_distances)
        target_all = np.concatenate(target_distances)
        pooled_value = (
            float("nan")
            if np.ptp(predicted_all) <= 1e-15 or np.ptp(target_all) <= 1e-15
            else float(spearmanr(predicted_all, target_all).statistic)
        )
        pooled = pooled_value if np.isfinite(pooled_value) else None
    return {
        "pooled_within_state": pooled,
        "macro_state": float(np.mean(valid)) if valid else None,
        "per_state": per_state,
        "comparison_rule": "pairwise distances only within the same held-seed-fold model",
    }


def standalone_mechanism_correlations(data: RuntimeVectorData) -> dict[str, Any]:
    """Untrained within-state monotonic checks; never substitutes for B1 increments."""
    expert = expert_output_blocks(data.selected_raw, data.selected_weights)
    s_loo_chunk = expert["s_loo_features"][:, -1]
    runtime_routed_rms = np.sqrt(np.mean(np.square(expert["routed"]), axis=-1))
    routed_chunk = np.sqrt(np.mean(np.square(runtime_routed_rms), axis=-1))
    velocity = (
        data.x_traj[:, 1, :, :N_LIVE_DIMS]
        - data.x_traj[:, 0, :, :N_LIVE_DIMS]
    )
    velocity_chunk = np.sqrt(np.mean(np.square(velocity), axis=(1, 2)))
    target = data.target[:, 0]
    features = {
        "s_loo_chunk": s_loo_chunk,
        "runtime_merged_routed_rms_chunk": routed_chunk,
        "d0_velocity_live7_rms": velocity_chunk,
    }
    states = data.state_group.astype(str)
    result = {}
    for name, value in features.items():
        per_state = {}
        for state in np.unique(states):
            index = np.flatnonzero(states == state)
            correlation = (
                None
                if np.ptp(value[index]) <= 1e-15 or np.ptp(target[index]) <= 1e-15
                else float(spearmanr(value[index], target[index]).statistic)
            )
            per_state[state] = correlation
        valid = [correlation for correlation in per_state.values() if correlation is not None]
        result[name] = {
            "per_state_spearman": per_state,
            "equal_state_macro_spearman": float(np.mean(valid)) if valid else None,
        }
    return {
        "status": "standalone untrained mechanism description; not a B1 increment",
        "target": "primary future-path energy",
        "features": result,
        "permutation_test": "not implemented and not claimed",
    }


def evaluate_feature_set(
    projected: dict[str, np.ndarray],
    feature_blocks: Sequence[Sequence[str]],
    data: RuntimeVectorData,
    outer_splits: int,
    split_seed: int,
    targets: np.ndarray | None = None,
) -> dict[str, Any]:
    evaluated_targets = data.target if targets is None else np.asarray(targets)
    prediction, folds, prediction_fold_id = cross_validated_predictions(
        projected,
        feature_blocks,
        evaluated_targets,
        data.state_group,
        data.seed_group,
        outer_splits,
        split_seed,
    )
    block_widths = []
    for block in feature_blocks:
        widths = {int(projected[family].shape[1]) for family in block}
        if len(widths) != 1:
            raise ValueError(f"families within one feature block differ in width: {block}")
        block_widths.append(widths.pop())
    per_state_mse = {
        str(state): float(
            np.mean(
                np.square(
                    prediction[np.asarray(data.state_group).astype(str) == str(state)]
                    - evaluated_targets[
                        np.asarray(data.state_group).astype(str) == str(state)
                    ]
                )
            )
        )
        for state in np.unique(np.asarray(data.state_group).astype(str))
    }
    row_mse = np.mean(np.square(prediction - evaluated_targets), axis=1)
    states_axis = np.asarray(data.state_group).astype(str)
    seeds_axis = np.asarray(data.seed_group).astype(str)
    paired_error_grid = {
        state: {
            seed: float(np.mean(row_mse[(states_axis == state) & (seeds_axis == seed)]))
            for seed in np.unique(seeds_axis)
        }
        for state in np.unique(states_axis)
    }
    result = {
        "feature_blocks": [list(block) for block in feature_blocks],
        "block_widths": block_widths,
        "total_readout_width": int(sum(block_widths)),
        "held_out_rmse": float(
            np.sqrt(np.mean(np.square(prediction - evaluated_targets)))
        ),
        "held_out_geometry_spearman": geometry_spearman(
            prediction, evaluated_targets, data.state_group, prediction_fold_id
        ),
        "per_state_paired_mse": per_state_mse,
        "paired_squared_error_state_by_candidate": paired_error_grid,
        "folds": folds,
        "outer_fold_id_by_sample": prediction_fold_id.tolist(),
    }
    if evaluated_targets.ndim == 2 and evaluated_targets.shape[1] == 70:
        result["secondary_correction_geometry_spearman"] = result[
            "held_out_geometry_spearman"
        ]
        predicted_final = (
            data.x_traj[:, 1, :, :N_LIVE_DIMS].reshape(len(data.x_traj), 70)
            + prediction
        )
        actual_final = data.x_traj[:, 10, :, :N_LIVE_DIMS].reshape(
            len(data.x_traj), 70
        )
        result["held_out_geometry_spearman"] = geometry_spearman(
            predicted_final, actual_final, data.state_group, prediction_fold_id
        )
        target_net_rms = np.sqrt(np.mean(np.square(evaluated_targets), axis=1))
        predicted_net_rms = np.sqrt(np.mean(np.square(prediction), axis=1))
        result["secondary_net_rms_rmse"] = float(
            np.sqrt(np.mean(np.square(predicted_net_rms - target_net_rms)))
        )
        result["secondary_k8_geometry_coverage"] = k8_geometry_coverage(
            predicted_final,
            actual_final,
            data.state_group,
            prediction_fold_id,
        )
    return result


def _require_shape(name: str, value: np.ndarray, suffix: tuple[int, ...]) -> None:
    if value.shape[1:] != suffix:
        raise ValueError(f"{name} has shape {value.shape}, expected [N,{','.join(map(str, suffix))}]")


def _load_one_run(run_dir: Path, store_path: Path | None) -> RuntimeVectorData:
    config = json.loads((run_dir / "experiment_config.json").read_text())
    records = json.loads((run_dir / "query_records.json").read_text())
    metadata = json.loads((run_dir / "server_metadata.json").read_text())
    actual_store = store_path or run_dir / "activation_flow.zarr"
    root = zarr.open_group(str(actual_store), mode="r")
    attrs = dict(root.attrs)
    if attrs.get("format") != FORMAT:
        raise ValueError(f"{actual_store} is not {FORMAT}")
    if int(attrs.get("probe_hb_layer", -1)) != PROBE_LAYER or int(
        attrs.get("probe_denoise", -1)
    ) != PROBE_DENOISE:
        raise ValueError("v3 vector analysis is frozen to HB5/d0")
    layers = [int(value) for value in attrs["hb_layers"]]
    if PROBE_LAYER not in layers:
        raise ValueError("HB5 is absent from capture metadata")
    layer_axis = layers.index(PROBE_LAYER)

    store_query = np.asarray(root["query_id"][:], dtype=np.int64)
    store_candidate = np.asarray(root["candidate_id"][:], dtype=np.int64)
    pair_to_row: dict[tuple[int, int], int] = {}
    for row, pair in enumerate(zip(store_query, store_candidate, strict=True)):
        key = (int(pair[0]), int(pair[1]))
        if key in pair_to_row:
            raise ValueError(f"duplicate query/candidate identity in {actual_store}: {key}")
        pair_to_row[key] = row

    selected_rows, states, seeds, tasks, checkpoints, sample_ids = [], [], [], [], [], []
    task_name = f"{config.get('benchmark')}:{config.get('task_id')}"
    checkpoint_name = str(metadata.get("checkpoint_sha256", metadata.get("checkpoint")))
    state_name = (
        f"{config.get('benchmark')}:{config.get('task_id')}:"
        f"init{config.get('init_state_id')}"
    )
    seen_queries: set[int] = set()
    for record in records:
        query = int(record["query_id"])
        if query in seen_queries:
            raise ValueError(f"query_records repeats query_id {query}")
        seen_queries.add(query)
        flow_noise_seed = str(record["flow_noise_seed"])
        for candidate in range(int(record["n_candidates"])):
            key = (query, candidate)
            if key not in pair_to_row:
                raise ValueError(f"capture store is missing query/candidate {key}")
            selected_rows.append(pair_to_row[key])
            states.append(state_name)
            seeds.append(f"{flow_noise_seed}:candidate{candidate}")
            tasks.append(task_name)
            checkpoints.append(checkpoint_name)
            sample_ids.append(f"{run_dir.resolve()}:{query}:{candidate}")
    rows = np.asarray(selected_rows, dtype=np.int64)
    ids = np.asarray(
        root["hb_selected_expert_id"].oindex[rows, layer_axis, PROBE_DENOISE],
        dtype=np.int64,
    )
    weights = np.asarray(
        root["hb_selected_expert_weight"].oindex[
            rows, layer_axis, PROBE_DENOISE
        ],
        dtype=np.float64,
    )
    router_probs = np.asarray(
        root["hb_probe_router_probs"].oindex[rows], dtype=np.float64
    )
    router_probs = np.maximum(router_probs, 0.0)
    router_mass = router_probs.sum(axis=-1, keepdims=True)
    if np.any(router_mass <= 0.0):
        raise ValueError("HB5/d0 full router probabilities have zero mass")
    router_probs /= router_mass
    data = RuntimeVectorData(
        x_traj=np.asarray(root["x_traj"].oindex[rows], dtype=np.float64),
        input_hidden=np.asarray(root["hb_probe_input_hidden"].oindex[rows], dtype=np.float64),
        shared_output=np.asarray(root["hb_probe_shared_output"].oindex[rows], dtype=np.float64),
        router_probs=router_probs,
        selected_ids=ids,
        selected_weights=weights,
        selected_raw=np.asarray(root["hb_probe_selected_expert_raw"].oindex[rows], dtype=np.float64),
        state_group=np.asarray(states),
        seed_group=np.asarray(seeds),
        task_group=np.asarray(tasks),
        checkpoint_group=np.asarray(checkpoints),
        sample_id=np.asarray(sample_ids),
    )
    _require_shape("x_traj", data.x_traj, (11, 10, 24))
    _require_shape("input_hidden", data.input_hidden, (10, 1024))
    _require_shape("shared_output", data.shared_output, (10, 1024))
    _require_shape("router_probs", data.router_probs, (10, 32))
    _require_shape("selected_ids", data.selected_ids, (10, 4))
    _require_shape("selected_weights", data.selected_weights, (10, 4))
    _require_shape("selected_raw", data.selected_raw, (10, 4, 1024))
    if np.any(ids < 0) or np.any(ids >= N_EXPERTS):
        raise ValueError("selected expert ID is outside 0..31")
    for name in (
        "x_traj",
        "input_hidden",
        "shared_output",
        "router_probs",
        "selected_weights",
        "selected_raw",
    ):
        if not np.all(np.isfinite(getattr(data, name))):
            raise ValueError(f"v3 vector capture has non-finite {name}")
    if np.any(data.selected_weights < 0.0):
        raise ValueError("selected expert weights contain negative values")
    return data


def concatenate_data(parts: Iterable[RuntimeVectorData]) -> RuntimeVectorData:
    rows = list(parts)
    if not rows:
        raise ValueError("at least one run is required")
    names = RuntimeVectorData.__dataclass_fields__
    return RuntimeVectorData(
        **{name: np.concatenate([getattr(row, name) for row in rows]) for name in names}
    )


def load_run_store_pairs(
    run_store_pairs: Sequence[tuple[Path, Path | None]],
) -> RuntimeVectorData:
    if not run_store_pairs:
        raise ValueError("at least one client run/store pair is required")
    owners_by_store: dict[str, dict[int, Path]] = {}
    for run_dir, store_path in run_store_pairs:
        if store_path is not None:
            store_key = str(store_path.resolve())
            owner_by_query = owners_by_store.setdefault(store_key, {})
            records = json.loads((run_dir / "query_records.json").read_text())
            for record in records:
                query = int(record["query_id"])
                if query in owner_by_query:
                    raise ValueError(
                        "shared-store query_id collision between "
                        f"{owner_by_query[query]} and {run_dir}: {query}; "
                        "use non-overlapping rollout_flow_lead.py --query-base values"
                    )
                owner_by_query[query] = run_dir
    return concatenate_data(
        _load_one_run(run_dir, store_path)
        for run_dir, store_path in run_store_pairs
    )


def load_runs(run_dirs: Sequence[Path], store_path: Path | None = None) -> RuntimeVectorData:
    return load_run_store_pairs([(run_dir, store_path) for run_dir in run_dirs])


def discover_grid_run_stores(grid_root: Path) -> list[tuple[Path, Path]]:
    suite_by_benchmark = {
        "libero_goal": "goal",
        "libero_spatial": "spatial",
        "libero_10": "long",
    }
    pairs = []
    for config_path in sorted(grid_root.rglob("experiment_config.json")):
        run_dir = config_path.parent
        if not (run_dir / "query_records.json").exists():
            continue
        config = json.loads(config_path.read_text())
        benchmark = str(config.get("benchmark"))
        if benchmark not in suite_by_benchmark:
            raise ValueError(f"unsupported grid benchmark in {config_path}: {benchmark}")
        store_path = (
            grid_root
            / "server"
            / suite_by_benchmark[benchmark]
            / "activation_flow.zarr"
        )
        if not store_path.exists():
            raise ValueError(f"grid store is missing for {config_path}: {store_path}")
        pairs.append((run_dir, store_path))
    if not pairs:
        raise ValueError(f"no client runtime-vector runs found under {grid_root}")
    return pairs


def load_grid_root(grid_root: Path) -> RuntimeVectorData:
    return load_run_store_pairs(discover_grid_run_stores(grid_root))


def subset_data(data: RuntimeVectorData, indices: np.ndarray) -> RuntimeVectorData:
    index = np.asarray(indices, dtype=np.int64)
    return RuntimeVectorData(
        **{
            name: np.asarray(getattr(data, name))[index]
            for name in RuntimeVectorData.__dataclass_fields__
        }
    )


def _one_metric_delta(added: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    baseline_spearman = baseline["held_out_geometry_spearman"][
        "pooled_within_state"
    ]
    added_spearman = added["held_out_geometry_spearman"]["pooled_within_state"]
    return {
        "held_out_rmse_delta": added["held_out_rmse"] - baseline["held_out_rmse"],
        "held_out_pooled_geometry_spearman_delta": (
            None
            if baseline_spearman is None or added_spearman is None
            else added_spearman - baseline_spearman
        ),
    }


def _evaluate_endpoints(
    projected: dict[str, np.ndarray],
    feature_blocks: Sequence[Sequence[str]],
    data: RuntimeVectorData,
    outer_splits: int,
    split_seed: int,
) -> dict[str, Any]:
    return {
        "primary_future_path_energy": evaluate_feature_set(
            projected, feature_blocks, data, outer_splits, split_seed
        ),
        "secondary_x10_minus_x1_live10x7": evaluate_feature_set(
            projected,
            feature_blocks,
            data,
            outer_splits,
            split_seed,
            data.secondary_target,
        ),
    }


def _endpoint_deltas(
    added: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    return {
        endpoint: _one_metric_delta(added[endpoint], baseline[endpoint])
        for endpoint in baseline
    }


def _analyze_task(
    data: RuntimeVectorData,
    width: int,
    outer_splits: int,
    split_seed: int,
    projection_seeds: Sequence[int],
) -> dict[str, Any]:
    admission = validate_admission(data, outer_splits)
    operational_baseline = dict(BASELINE_TIERS)["b1_operational"]
    seed_runs = []
    reference_projected: dict[str, np.ndarray] | None = None
    for projection_seed in projection_seeds:
        projected = project_feature_families(data, width, projection_seed)
        if reference_projected is None:
            reference_projected = projected
        b1 = _evaluate_endpoints(
            projected, (operational_baseline,), data, outer_splits, split_seed
        )
        matched_control = _evaluate_endpoints(
            projected,
            (operational_baseline, (ROUTED_RMS_CONTROL_FAMILY,)),
            data,
            outer_splits,
            split_seed,
        )
        id_aware = _evaluate_endpoints(
            projected,
            (operational_baseline, ID_AWARE_EXPERT_FAMILIES),
            data,
            outer_splits,
            split_seed,
        )
        invariant = _evaluate_endpoints(
            projected,
            (operational_baseline, INVARIANT_EXPERT_FAMILIES),
            data,
            outer_splits,
            split_seed,
        )
        invariant_gram = _evaluate_endpoints(
            projected,
            (
                operational_baseline,
                INVARIANT_EXPERT_FAMILIES,
                GRAM_EXPERT_FAMILIES,
            ),
            data,
            outer_splits,
            split_seed,
        )
        seed_runs.append(
            {
                "projection_seed": int(projection_seed),
                "b1_operational": b1,
                "b1_plus_matched_routed_rms_control": matched_control,
                "b1_plus_id_aware_contribution_secondary": id_aware,
                "b1_plus_s_loo_primary_commonality": invariant,
                "b1_plus_s_loo_gram_secondary": invariant_gram,
                "deltas_vs_b1": {
                    "id_aware_contribution_secondary": _endpoint_deltas(
                        id_aware, b1
                    ),
                    "s_loo_primary_commonality": _endpoint_deltas(invariant, b1),
                    "s_loo_gram_secondary": _endpoint_deltas(invariant_gram, b1),
                    "matched_routed_rms_control": _endpoint_deltas(
                        matched_control, b1
                    ),
                },
            }
        )

    assert reference_projected is not None
    reference_feature_sets = {
        name: evaluate_feature_set(
            reference_projected, (families,), data, outer_splits, split_seed
        )
        for name, families in BASELINE_TIERS
    }
    reference_feature_sets["id_aware_expert_only_diagnostic"] = evaluate_feature_set(
        reference_projected,
        (ID_AWARE_EXPERT_FAMILIES,),
        data,
        outer_splits,
        split_seed,
    )
    positive_control_target = immediate_control_target(data.x_traj)
    positive_control = {
        "status": "instrumentation only; excluded from the primary comparison",
        "projection_seed": int(projection_seeds[0]),
        "target": "flatten10x7(live7(x1 - x0))",
        "b1_operational": evaluate_feature_set(
            reference_projected,
            (operational_baseline,),
            data,
            outer_splits,
            split_seed,
            positive_control_target,
        ),
        "b1_plus_s_loo_gram": evaluate_feature_set(
            reference_projected,
            (
                operational_baseline,
                INVARIANT_EXPERT_FAMILIES,
                GRAM_EXPERT_FAMILIES,
            ),
            data,
            outer_splits,
            split_seed,
            positive_control_target,
        ),
    }
    return {
        "admission": admission,
        "checkpoint": str(np.unique(data.checkpoint_group)[0]),
        "standalone_mechanism": standalone_mechanism_correlations(data),
        "projection_seed_runs": seed_runs,
        "reference_projection_diagnostics": {
            "projection_seed": int(projection_seeds[0]),
            "feature_sets": reference_feature_sets,
            "instrumentation_positive_control": positive_control,
        },
    }


def _macro_metric(
    task_results: list[dict[str, Any]], arm: str, endpoint: str
) -> dict[str, Any]:
    rows = [result[arm][endpoint] for result in task_results]
    rmse = [row["held_out_rmse"] for row in rows]
    spearman = [
        row["held_out_geometry_spearman"]["pooled_within_state"] for row in rows
    ]
    return {
        "equal_task_macro_held_out_rmse": float(np.mean(rmse)),
        "equal_task_macro_held_out_geometry_spearman": (
            None if any(value is None for value in spearman) else float(np.mean(spearman))
        ),
        "per_task_held_out_rmse": list(map(float, rmse)),
        "per_task_held_out_geometry_spearman": spearman,
    }


def _robustness_summary(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in (
        "matched_routed_rms_control",
        "id_aware_contribution_secondary",
        "s_loo_primary_commonality",
        "s_loo_gram_secondary",
    ):
        result[arm] = {}
        for endpoint in (
            "primary_future_path_energy",
            "secondary_x10_minus_x1_live10x7",
        ):
            rmse_delta = np.asarray(
                [
                    row["deltas_vs_b1"][arm][endpoint]["held_out_rmse_delta"]
                    for row in seed_rows
                ]
            )
            spearman_values = [
                row["deltas_vs_b1"][arm][endpoint][
                    "held_out_pooled_geometry_spearman_delta"
                ]
                for row in seed_rows
            ]
            result[arm][endpoint] = {
                "rmse_delta_per_projection_seed": rmse_delta.tolist(),
                "rmse_delta_median": float(np.median(rmse_delta)),
                "rmse_delta_min": float(np.min(rmse_delta)),
                "rmse_delta_max": float(np.max(rmse_delta)),
                "rmse_improves_all_projection_seeds": bool(
                    np.all(rmse_delta < 0.0)
                ),
                "geometry_spearman_delta_per_projection_seed": spearman_values,
                "geometry_spearman_delta_median": (
                    None
                    if any(value is None for value in spearman_values)
                    else float(np.median(spearman_values))
                ),
            }
    return result


def bootstrap_axis_indices(
    n_tasks: int,
    n_states: int,
    n_candidates: int,
    draws: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw task-local states and one common candidate-column sample per draw."""
    rng = np.random.default_rng(seed)
    candidate_indices = np.empty((draws, n_candidates), dtype=np.int64)
    state_indices = np.empty((draws, n_tasks, n_states), dtype=np.int64)
    for draw in range(draws):
        candidate_indices[draw] = rng.integers(0, n_candidates, n_candidates)
        for task in range(n_tasks):
            state_indices[draw, task] = rng.integers(0, n_states, n_states)
    return state_indices, candidate_indices


def _preregistered_decision(
    task_seed_results: list[dict[str, Any]], bootstrap_seed: int, task_order: Sequence[str]
) -> dict[str, Any]:
    endpoint = "primary_future_path_energy"
    baseline_rows = [row["b1_operational"][endpoint] for row in task_seed_results]
    full_rows = [
        row["b1_plus_s_loo_primary_commonality"][endpoint]
        for row in task_seed_results
    ]
    control_rows = [
        row["b1_plus_matched_routed_rms_control"][endpoint]
        for row in task_seed_results
    ]
    task_relative_improvements = []
    task_vs_control_improvements = []
    error_grid_triplets = []
    for baseline, full, control in zip(
        baseline_rows, full_rows, control_rows, strict=True
    ):
        baseline_grid = baseline["paired_squared_error_state_by_candidate"]
        full_grid = full["paired_squared_error_state_by_candidate"]
        control_grid = control["paired_squared_error_state_by_candidate"]
        states = sorted(baseline_grid)
        seeds = sorted(baseline_grid[states[0]])
        if states != sorted(full_grid) or states != sorted(control_grid):
            raise RuntimeError("B1/full/control paired-state identities differ")
        baseline_mse = np.asarray(
            [[baseline_grid[state][seed] for seed in seeds] for state in states]
        )
        full_mse = np.asarray(
            [[full_grid[state][seed] for seed in seeds] for state in states]
        )
        control_mse = np.asarray(
            [[control_grid[state][seed] for seed in seeds] for state in states]
        )
        baseline_rmse = float(np.sqrt(np.mean(baseline_mse)))
        full_rmse = float(np.sqrt(np.mean(full_mse)))
        control_rmse = float(np.sqrt(np.mean(control_mse)))
        if baseline_rmse <= 1e-15 or control_rmse <= 1e-15:
            raise AdmissionError(
                "relative-effect gate is undefined for zero B1/control RMSE"
            )
        task_relative_improvements.append((baseline_rmse - full_rmse) / baseline_rmse)
        task_vs_control_improvements.append(
            (control_rmse - full_rmse) / control_rmse
        )
        error_grid_triplets.append((baseline_mse, full_mse, control_mse))

    state_draws, common_candidate_draws = bootstrap_axis_indices(
        len(error_grid_triplets),
        N_STATES_PER_TASK,
        N_SEEDS_PER_TASK,
        BOOTSTRAP_DRAWS,
        bootstrap_seed,
    )
    bootstrap_macro = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        draw_effects = []
        sampled_candidates = common_candidate_draws[draw]
        for task_index, (baseline_mse, full_mse, _control_mse) in enumerate(
            error_grid_triplets
        ):
            sampled_states = state_draws[draw, task_index]
            sampled = np.ix_(sampled_states, sampled_candidates)
            baseline_rmse = np.sqrt(np.mean(baseline_mse[sampled]))
            full_rmse = np.sqrt(np.mean(full_mse[sampled]))
            draw_effects.append((baseline_rmse - full_rmse) / baseline_rmse)
        bootstrap_macro[draw] = np.mean(draw_effects)
    macro = float(np.mean(task_relative_improvements))
    ci_low, ci_high = np.quantile(bootstrap_macro, (0.025, 0.975))
    direction_gate = bool(np.all(np.asarray(task_relative_improvements) > 0.0))
    effect_gate = macro >= 0.02
    ci_gate = float(ci_low) > 0.0
    matched_control_macro = float(np.mean(task_vs_control_improvements))
    matched_control_gate = bool(
        np.all(np.asarray(task_vs_control_improvements) > 0.0)
        and matched_control_macro > 0.0
    )
    secondary_endpoint = "secondary_x10_minus_x1_live10x7"
    k8_relative_improvements = []
    for row in task_seed_results:
        baseline_coverage = row["b1_operational"][secondary_endpoint][
            "secondary_k8_geometry_coverage"
        ]["macro_selected_target_coverage"]
        full_coverage = row["b1_plus_s_loo_primary_commonality"][
            secondary_endpoint
        ]["secondary_k8_geometry_coverage"]["macro_selected_target_coverage"]
        if (
            baseline_coverage is None
            or full_coverage is None
            or baseline_coverage <= 1e-15
        ):
            k8_relative_improvements.append(None)
        else:
            k8_relative_improvements.append(
                (baseline_coverage - full_coverage) / baseline_coverage
            )
    valid_k8 = all(value is not None for value in k8_relative_improvements)
    k8_macro = (
        float(np.mean(k8_relative_improvements)) if valid_k8 else None
    )
    k8_all_positive = bool(
        valid_k8 and np.all(np.asarray(k8_relative_improvements) > 0.0)
    )
    k8_effect_gate = bool(k8_macro is not None and k8_macro >= 0.01)
    return {
        "effect": "(RMSE_B1 - RMSE_B1_plus_S_LOO) / RMSE_B1",
        "task_order": list(task_order),
        "per_task_relative_improvement": task_relative_improvements,
        "tasks_improved": int(np.sum(np.asarray(task_relative_improvements) > 0.0)),
        "required_tasks_improved": len(EXPECTED_TASKS),
        "equal_task_macro_relative_improvement": macro,
        "required_macro_relative_improvement": 0.02,
        "task_stratified_state_candidate_bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": bootstrap_seed,
            "confidence_interval_95": [float(ci_low), float(ci_high)],
            "resampling_unit": (
                "state and repeated-noise candidate axes independently within task; "
                "one common candidate-column draw reused across all tasks; task "
                "effects equally averaged"
            ),
        },
        "matched_exact11_control": {
            "control": "runtime merged-routed RMS per token plus chunk RMS",
            "per_task_relative_improvement_s_loo_over_control": (
                task_vs_control_improvements
            ),
            "equal_task_macro_relative_improvement_s_loo_over_control": (
                matched_control_macro
            ),
            "all_tasks_s_loo_better_than_control": matched_control_gate,
        },
        "gates": {
            "all_tasks_positive": direction_gate,
            "macro_at_least_two_percent": effect_gate,
            "bootstrap_ci_lower_above_zero": ci_gate,
            "s_loo_beats_matched_control": matched_control_gate,
            "projection_seed_pass": (
                direction_gate and effect_gate and ci_gate and matched_control_gate
            ),
        },
        "pruning_k8_gate": {
            "effect": (
                "(final-action K8 coverage_B1 - coverage_B1_plus_S_LOO) / "
                "coverage_B1; lower coverage is better"
            ),
            "per_task_relative_improvement": k8_relative_improvements,
            "equal_task_macro_relative_improvement": k8_macro,
            "required_macro_relative_improvement": 0.01,
            "all_tasks_positive": k8_all_positive,
            "macro_at_least_one_percent": k8_effect_gate,
            "status": (
                "unresolved crossfit-ensemble diagnostic; not a deployable K16 "
                "candidate cloud and not part of the preregistered gate"
            ),
            "descriptive_threshold_met": k8_all_positive and k8_effect_gate,
            "projection_seed_pruning_pass": None,
        },
    }


def analyze(
    data: RuntimeVectorData,
    width: int,
    outer_splits: int,
    split_seed: int,
    projection_seeds: Sequence[int],
    sensitivity_widths: Sequence[int] = (),
) -> dict[str, Any]:
    if sensitivity_widths and width != 128:
        raise AdmissionError(
            "the preregistered primary readout width is 128; use "
            "--sensitivity-width for descriptive alternatives"
        )
    seeds = tuple(map(int, projection_seeds))
    if seeds != DEFAULT_PROJECTION_SEEDS:
        raise AdmissionError(
            "the preregistered decision requires the exact frozen projection seeds "
            f"{DEFAULT_PROJECTION_SEEDS}; overrides are sensitivity-only"
        )
    global_x0_repeat_error = repeated_x0_max_error(data)
    if global_x0_repeat_error > X0_REPEAT_ATOL:
        raise AdmissionError(
            "repeated candidate identity fails across-state/task/checkpoint x0 "
            f"audit: {global_x0_repeat_error:.3g} > {X0_REPEAT_ATOL:.1g}"
        )
    task_names = np.unique(data.task_group.astype(str))
    for task in task_names:
        task_data = subset_data(data, np.flatnonzero(data.task_group.astype(str) == task))
        validate_admission(task_data, outer_splits)
        n_states = len(np.unique(task_data.state_group.astype(str)))
        n_seeds = len(np.unique(task_data.seed_group.astype(str)))
        if n_states != N_STATES_PER_TASK or n_seeds != N_SEEDS_PER_TASK:
            raise AdmissionError(
                f"task {task} needs exactly {N_STATES_PER_TASK} states and "
                f"{N_SEEDS_PER_TASK} repeated-noise candidates for the frozen "
                f"bootstrap/K8 protocol; found {n_states} and {n_seeds}"
            )
        states = task_data.state_group.astype(str)
        repeated_seeds = task_data.seed_group.astype(str)
        cell_counts = np.asarray(
            [
                np.sum((states == state) & (repeated_seeds == seed))
                for state in np.unique(states)
                for seed in np.unique(repeated_seeds)
            ]
        )
        if not np.all(cell_counts == 1):
            raise AdmissionError(
                f"task {task} is not an exact 8x16 crossed state/candidate grid"
            )
        if len(np.unique(task_data.checkpoint_group.astype(str))) != 1:
            raise AdmissionError(
                f"task {task} spans multiple checkpoints; expert coordinates "
                "must never be pooled across checkpoints"
            )
    if set(map(str, task_names)) != set(EXPECTED_TASKS):
        raise AdmissionError(
            "cross-task inference requires exactly "
            f"{sorted(EXPECTED_TASKS)}; found {list(map(str, task_names))}"
        )
    per_task: dict[str, dict[str, Any]] = {}
    for task in task_names:
        task_data = subset_data(data, np.flatnonzero(data.task_group.astype(str) == task))
        checkpoints = np.unique(task_data.checkpoint_group.astype(str))
        if len(checkpoints) != 1:
            raise AdmissionError(
                f"task {task} spans {len(checkpoints)} checkpoints; expert coordinates "
                "must never be pooled across checkpoints"
            )
        per_task[str(task)] = _analyze_task(
            task_data, width, outer_splits, split_seed, seeds
        )

    mechanism_macro = {}
    for feature in (
        "s_loo_chunk",
        "runtime_merged_routed_rms_chunk",
        "d0_velocity_live7_rms",
    ):
        task_values = [
            per_task[str(task)]["standalone_mechanism"]["features"][feature][
                "equal_state_macro_spearman"
            ]
            for task in task_names
        ]
        mechanism_macro[feature] = {
            "per_task_equal_state_macro_spearman": task_values,
            "equal_task_macro_spearman": (
                None
                if any(value is None for value in task_values)
                else float(np.mean(task_values))
            ),
        }

    macro_seed_rows = []
    for seed_index, projection_seed in enumerate(seeds):
        task_seed_results = [
            per_task[task]["projection_seed_runs"][seed_index]
            for task in map(str, task_names)
        ]
        arm_names = {
            "b1_operational": "b1_operational",
            "b1_plus_matched_routed_rms_control": (
                "b1_plus_matched_routed_rms_control"
            ),
            "b1_plus_id_aware_contribution_secondary": (
                "b1_plus_id_aware_contribution_secondary"
            ),
            "b1_plus_s_loo_primary_commonality": (
                "b1_plus_s_loo_primary_commonality"
            ),
            "b1_plus_s_loo_gram_secondary": "b1_plus_s_loo_gram_secondary",
        }
        endpoints = (
            "primary_future_path_energy",
            "secondary_x10_minus_x1_live10x7",
        )
        macro_row: dict[str, Any] = {"projection_seed": projection_seed}
        for output_name, source_name in arm_names.items():
            macro_row[output_name] = {
                endpoint: _macro_metric(task_seed_results, source_name, endpoint)
                for endpoint in endpoints
            }
        macro_row["deltas_vs_b1"] = {}
        for arm in (
            "matched_routed_rms_control",
            "id_aware_contribution_secondary",
            "s_loo_primary_commonality",
            "s_loo_gram_secondary",
        ):
            macro_row["deltas_vs_b1"][arm] = {}
            for endpoint in endpoints:
                macro_row["deltas_vs_b1"][arm][endpoint] = {}
                for key in (
                    "held_out_rmse_delta",
                    "held_out_pooled_geometry_spearman_delta",
                ):
                    values = [
                        row["deltas_vs_b1"][arm][endpoint][key]
                        for row in task_seed_results
                    ]
                    macro_row["deltas_vs_b1"][arm][endpoint][key] = (
                        None
                        if any(value is None for value in values)
                        else float(np.mean(values))
                    )
        macro_row["preregistered_decision"] = _preregistered_decision(
            task_seed_results,
            split_seed + int(projection_seed),
            list(map(str, task_names)),
        )
        macro_seed_rows.append(macro_row)
    summary = {
        "protocol": "runtime-vector-remaining-correction-v3",
        "scientific_status": (
            "task-local held-out inference with equal-task macro aggregation; "
            "no causal claim"
        ),
        "target": {
            "primary": (
                "sqrt(sum_{tau=1..9,token,live7}(x[tau+1]-x[tau])^2 / "
                "(9*10*7))"
            ),
            "primary_width": 1,
            "secondary": "flatten10x7(live7(x10 - x1))",
            "secondary_width": 70,
            "secondary_reports": "vector RMSE, geometry Spearman, net-RMS RMSE, K8",
            "activation_clock": (
                "HB5/d0 activation is computed at x0 input; x1 becomes observable "
                "after d0 completes and before the operational decision"
            ),
            "positive_control_only": "flatten10x7(live7(x1 - x0))",
        },
        "admission": {
            "admitted": True,
            "task_count": len(task_names),
            "expected_tasks": sorted(EXPECTED_TASKS),
            "tasks": list(map(str, task_names)),
            "checkpoint_count": len(np.unique(data.checkpoint_group.astype(str))),
            "max_x0_repeat_error_across_states_tasks_checkpoints": (
                global_x0_repeat_error
            ),
            "states_per_task": N_STATES_PER_TASK,
            "seed_candidates_per_task": N_SEEDS_PER_TASK,
        },
        "split": {
            "scheme": "nested Cartesian state+seed group-disjoint CV",
            "fit_scope": "one task and one checkpoint only",
            "seed_identity": "(flow_noise_seed, candidate_id), audited by repeated x0",
            "outer_splits_per_axis": outer_splits,
            "split_seed": split_seed,
            "target_preprocessing": "training-fold mean only",
        },
        "readout": {
            "type": "linear ridge over deterministic signed CountSketch",
            "fixed_width_per_feature_block": width,
            "projection_seeds": list(seeds),
            "feature_standardization": "training fold only, per family",
            "strict_nesting": (
                "B1 sketch is frozen verbatim as the first block; independent expert "
                "blocks are appended, never mixed into or substituted for B1"
            ),
            "alpha_selection": "independent nested-CV selection for every arm and seed",
            "projection_inference_guard": (
                "report all projection seeds; no single-sketch conclusion"
            ),
        },
        "arm_roles": {
            "id_aware_contribution_secondary": "task-local extraction probe only",
            "s_loo_primary_commonality": (
                "primary invariant S_LOO,t plus chunk-RMS expert block"
            ),
            "s_loo_gram_secondary": (
                "secondary invariant Gv/Gc eigenvalues plus R/C/S_LOO block"
            ),
        },
        "per_task": per_task,
        "standalone_mechanism_equal_task_macro": {
            "status": "descriptive only; does not replace B1 incremental gates",
            "features": mechanism_macro,
        },
        "equal_task_macro_by_projection_seed": macro_seed_rows,
        "projection_seed_robustness": _robustness_summary(macro_seed_rows),
        "preregistered_decision": {
            "rule": (
                "every projection seed must pass 5/5 task direction, >=2% equal-task "
                "macro relative RMSE improvement, bootstrap CI lower > 0, and "
                "S_LOO must beat the matched exact11 routed-RMS control in 5/5 tasks"
            ),
            "all_projection_seeds_pass": bool(
                all(
                    row["preregistered_decision"]["gates"]["projection_seed_pass"]
                    for row in macro_seed_rows
                )
            ),
            "worst_projection_seed_macro_relative_improvement": float(
                min(
                    row["preregistered_decision"][
                        "equal_task_macro_relative_improvement"
                    ]
                    for row in macro_seed_rows
                )
            ),
            "deployable_pruning_k8_gate": None,
            "worst_projection_seed_k8_macro_relative_improvement": (
                None
                if any(
                    row["preregistered_decision"]["pruning_k8_gate"][
                        "equal_task_macro_relative_improvement"
                    ]
                    is None
                    for row in macro_seed_rows
                )
                else float(
                    min(
                        row["preregistered_decision"]["pruning_k8_gate"][
                            "equal_task_macro_relative_improvement"
                        ]
                        for row in macro_seed_rows
                    )
                )
            ),
            "pruning_claim_rule": (
                "unresolved until an independent 16-candidate train-noise grid "
                "permits one deployable model to score the complete K16 cloud"
            ),
        },
        "statistics": {
            "p_values": "not computed",
            "common_seed_cross_task_permutation": "not implemented and not claimed",
        },
    }
    if sensitivity_widths:
        summary["capacity_sensitivity"] = {
            "status": (
                "descriptive only; the preregistered decision uses width 128 and "
                "widths are never selected by performance"
            ),
            "runs": {},
        }
        for sensitivity_width in sensitivity_widths:
            if sensitivity_width == width:
                continue
            sensitivity = analyze(
                data,
                int(sensitivity_width),
                outer_splits,
                split_seed,
                seeds,
                (),
            )
            summary["capacity_sensitivity"]["runs"][str(sensitivity_width)] = {
                "preregistered_gate_applied": False,
                "decision_metrics_for_description": sensitivity[
                    "preregistered_decision"
                ],
                "projection_seed_robustness": sensitivity[
                    "projection_seed_robustness"
                ],
            }
    return summary


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, action="append")
    source.add_argument(
        "--grid-root",
        type=Path,
        help="discover clients and server/{goal,spatial,long} stores",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="optional shared activation_flow.zarr selected by unique query IDs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "analysis" / "runtime-vector-v3" / "summary.json",
    )
    parser.add_argument("--readout-width", type=int, default=128)
    parser.add_argument(
        "--sensitivity-width",
        type=int,
        action="append",
        dest="sensitivity_widths",
        help="descriptive capacity sensitivity; defaults to 64 and 256",
    )
    parser.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="run only the preregistered width-128 analysis",
    )
    parser.add_argument("--outer-splits", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=20260823)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.grid_root is not None:
            if args.store is not None:
                raise ValueError("--store cannot be combined with --grid-root")
            data = load_grid_root(args.grid_root)
        else:
            data = load_runs(args.run, args.store)
        summary = analyze(
            data,
            args.readout_width,
            args.outer_splits,
            args.split_seed,
            DEFAULT_PROJECTION_SEEDS,
            ()
            if args.no_sensitivity
            else (args.sensitivity_widths or (64, 256)),
        )
    except AdmissionError as error:
        print(f"REFUSED: {error}")
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
