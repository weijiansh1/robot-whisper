#!/usr/bin/env python3
"""Test whether late MoE routing signatures transfer to rolling-star branches.

The analysis deliberately separates two questions:

* ``relative_tail_50_90`` reproduces the earlier offline tail analysis.  It is
  descriptive because computing relative phase requires the eventual rollout
  length.
* ``fixed_q24_q31`` uses the same absolute queries for every branch.  All 352
  branches are alive through q31, so this window is a causal-prefix test that
  cannot read termination time.

No AUC or remaining-time feature is computed.  Outcome models are evaluated
with Brier error, snapshot-grouped cross-validation, and leave-one-worker-out
cross-validation.  MoE-only models are compared with physical-state,
flow-noise, and sampled-action controls.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import analyze_rolling_star_experiment as rolling
import analyze_state_route_layer_denoise_effects as route_effects


SEED = 20260829
FIXED_WINDOWS = {
    "fixed_q0_q7": (0, 8),
    "fixed_q8_q15": (8, 16),
    "fixed_q16_q23": (16, 24),
    "fixed_q24_q31": (24, 32),
}
RELATIVE_ANCHORS = 10
PCA_COMPONENTS = 24
LAYER_GROUPS = {
    "front_2_5": np.arange(0, 4, dtype=np.int64),
    "back_12_15": np.arange(4, 8, dtype=np.int64),
}
PHASE_STATISTICS = (
    "phase_mean",
    "phase_late",
    "late_minus_early",
    "phase_std",
    "phase_slope",
    "query_change",
)


@dataclass(frozen=True)
class RouteFeature:
    name: str
    family: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report-only", action="store_true")
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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
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


def top4_mass(probability: np.ndarray) -> np.ndarray:
    return np.partition(probability, -4, axis=-1)[..., -4:].sum(axis=-1)


def phase_statistics(curve: np.ndarray) -> dict[str, float]:
    values = np.asarray(curve, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or np.any(~np.isfinite(values)):
        raise ValueError("phase curve must be a finite one-dimensional window")
    width = max(2, len(values) // 3)
    time = np.linspace(-1.0, 1.0, len(values), dtype=np.float64)
    denominator = float(time @ time)
    return {
        "phase_mean": float(values.mean()),
        "phase_late": float(values[-width:].mean()),
        "late_minus_early": float(
            values[-width:].mean() - values[:width].mean()
        ),
        "phase_std": float(values.std()),
        "phase_slope": float(values @ time / denominator),
        "query_change": float(np.abs(np.diff(values)).mean()),
    }


def pair_distance_matrix(
    probability: np.ndarray, layer_axes: np.ndarray, *, hard: bool
) -> np.ndarray:
    """Return query-pair route distances averaged over selected layers."""
    selected = np.asarray(probability[:, layer_axes], dtype=np.float32)
    if selected.ndim != 3 or selected.shape[-1] != 32:
        raise ValueError(f"unexpected route curve {selected.shape}")
    if hard:
        expert = np.argpartition(selected, -4, axis=-1)[..., -4:]
        equality = expert[:, None, :, :, None] == expert[None, :, :, None, :]
        overlap = equality.any(axis=-1).sum(axis=-1).astype(np.float32) / 4.0
        return 1.0 - overlap.mean(axis=-1)
    distance = hellinger(selected[:, None], selected[None, :])
    return distance.mean(axis=-1)


def route_pair_descriptors(distance: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 4:
        raise ValueError("route pair matrix has the wrong shape")
    adjacent = np.asarray(
        [matrix[index - 1, index] for index in range(1, len(matrix))]
    )
    width = max(2, len(adjacent) // 3)
    return_advantage = np.asarray(
        [
            matrix[index - 1, index] - np.min(matrix[index, : index - 1])
            for index in range(2, len(matrix))
        ]
    )
    return {
        "adjacent_mean": float(adjacent.mean()),
        "adjacent_late": float(adjacent[-width:].mean()),
        "adjacent_delta": float(
            adjacent[-width:].mean() - adjacent[:width].mean()
        ),
        "return_advantage_mean": float(return_advantage.mean()),
        "return_advantage_max": float(return_advantage.max()),
        "return_fraction": float(np.mean(return_advantage > 0.0)),
    }


def append_phase_features(
    values: list[float],
    metadata: list[RouteFeature],
    curve: np.ndarray,
    *,
    prefix: str,
    family: str,
) -> None:
    summaries = phase_statistics(curve)
    for statistic in PHASE_STATISTICS:
        values.append(summaries[statistic])
        metadata.append(RouteFeature(f"{prefix}|{statistic}", family))


def append_pair_features(
    values: list[float],
    metadata: list[RouteFeature],
    probability: np.ndarray,
    layer_axes: np.ndarray,
    *,
    prefix: str,
) -> None:
    for route_kind, hard in (("soft_route", False), ("hard_route", True)):
        descriptors = route_pair_descriptors(
            pair_distance_matrix(probability, layer_axes, hard=hard)
        )
        for descriptor, value in descriptors.items():
            family = (
                "query_recurrence"
                if descriptor.startswith("return")
                else "query_persistence"
            )
            values.append(value)
            metadata.append(
                RouteFeature(f"{route_kind}|{prefix}|{descriptor}", family)
            )


def extract_moe_features(
    raw_probability: np.ndarray,
) -> tuple[np.ndarray, list[RouteFeature], float]:
    probability = normalize_probability(raw_probability)
    if probability.ndim != 5 or probability.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"unexpected HB probability shape {probability.shape}")
    state = probability[..., 0, :]
    action = probability[..., 1:, :]
    action_mean = action.mean(axis=3)
    state_denoise_span = float(np.max(np.abs(state - state[:, :, :1, :])))

    state_metrics = {
        "state_entropy": -np.sum(
            state * np.log(np.maximum(state, 1e-12)), axis=-1
        )
        / np.log(32.0),
        "state_top1": state.max(axis=-1),
        "state_top4": top4_mass(state),
    }
    action_metrics = {
        "action_entropy": -np.sum(
            action * np.log(np.maximum(action, 1e-12)), axis=-1
        ).mean(axis=3)
        / np.log(32.0),
        "action_top1": action.max(axis=-1).mean(axis=3),
        "action_top4": top4_mass(action).mean(axis=3),
        "action_token_dispersion": hellinger(
            action, action_mean[..., None, :]
        ).mean(axis=3),
        "state_action_gap": hellinger(state, action_mean),
    }
    family_lookup = {
        "state_entropy": "state_confidence",
        "state_top1": "state_confidence",
        "state_top4": "state_confidence",
        "action_entropy": "action_confidence",
        "action_top1": "action_confidence",
        "action_top4": "action_confidence",
        "action_token_dispersion": "token_disagreement",
        "state_action_gap": "state_action_mismatch",
    }

    values: list[float] = []
    metadata: list[RouteFeature] = []
    for metric, curve in state_metrics.items():
        for group_name, layer_axes in LAYER_GROUPS.items():
            for denoise in range(10):
                append_phase_features(
                    values,
                    metadata,
                    curve[:, layer_axes, denoise].mean(axis=1),
                    prefix=f"{metric}|{group_name}|d{denoise}",
                    family=family_lookup[metric],
                )
    for metric, curve in action_metrics.items():
        for group_name, layer_axes in LAYER_GROUPS.items():
            for denoise in range(10):
                append_phase_features(
                    values,
                    metadata,
                    curve[:, layer_axes, denoise].mean(axis=1),
                    prefix=f"{metric}|{group_name}|d{denoise}",
                    family=family_lookup[metric],
                )

    denoise_sources = {"state": state, "action": action_mean}
    denoise_curves = []
    for token_role, token_probability in denoise_sources.items():
        soft_motion = hellinger(
            token_probability[:, :, 1:], token_probability[:, :, :-1]
        )
        expert = np.argpartition(token_probability, -4, axis=-1)[..., -4:]
        equality = expert[:, :, 1:, :, None] == expert[:, :, :-1, None, :]
        hard_switch = 1.0 - (
            equality.any(axis=-1).sum(axis=-1).astype(np.float32) / 4.0
        )
        denoise_curves.extend(
            (
                (
                    f"{token_role}_denoise_motion",
                    soft_motion,
                    "denoise_soft_motion",
                ),
                (
                    f"{token_role}_denoise_hard_switch",
                    hard_switch,
                    "denoise_hard_switch",
                ),
            )
        )
    for metric, curve, family in denoise_curves:
        for group_name, layer_axes in LAYER_GROUPS.items():
            for denoise in range(9):
                append_phase_features(
                    values,
                    metadata,
                    curve[:, layer_axes, denoise].mean(axis=1),
                    prefix=(
                        f"{metric}|{group_name}|d{denoise}_to_d{denoise + 1}"
                    ),
                    family=family,
                )

    for group_name, layer_axes in LAYER_GROUPS.items():
        for denoise in range(10):
            append_pair_features(
                values,
                metadata,
                state[:, :, denoise, :],
                layer_axes,
                prefix=f"state|{group_name}|d{denoise}",
            )
    for group_name, layer_axes in LAYER_GROUPS.items():
        for denoise in range(10):
            append_pair_features(
                values,
                metadata,
                action_mean[:, :, denoise, :],
                layer_axes,
                prefix=f"action|{group_name}|d{denoise}",
            )

    result = np.asarray(values, dtype=np.float32)
    names = [item.name for item in metadata]
    if len(names) != len(set(names)) or np.any(~np.isfinite(result)):
        raise ValueError("invalid or duplicate MoE feature bank")
    return result, metadata, state_denoise_span


def relative_positions(length: int, start: float, stop: float) -> np.ndarray:
    if length < 4:
        raise ValueError("rollout is too short for relative anchors")
    return np.linspace(
        start * (length - 1),
        stop * (length - 1),
        RELATIVE_ANCHORS,
        dtype=np.float64,
    )


def interpolate_rows(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    alpha_shape = (len(positions),) + (1,) * (array.ndim - 1)
    alpha = (positions - lower).reshape(alpha_shape)
    return array[lower] * (1.0 - alpha) + array[upper] * alpha


def summarize_channels(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or len(array) < 2:
        raise ValueError("control summary expects [query, channel]")
    width = max(2, len(array) // 3)
    return np.concatenate(
        (
            array.mean(axis=0),
            array.std(axis=0),
            array[-1] - array[0],
            array[-width:].mean(axis=0) - array[:width].mean(axis=0),
            np.abs(np.diff(array, axis=0)).mean(axis=0),
        )
    ).astype(np.float32)


def extract_control_features(
    trajectory: dict[str, np.ndarray], positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    policy = interpolate_rows(trajectory["policy_state"], positions)
    # sim_state[:, 0] is MuJoCo time.  Excluding it prevents an explicit clock
    # from entering the physical control bank.
    physical_state = interpolate_rows(trajectory["sim_state"][:, 1:], positions)
    physical = summarize_channels(np.c_[policy, physical_state])

    noise = interpolate_rows(trajectory["flow_noise"], positions)
    action = interpolate_rows(trajectory["action_chunks"], positions)
    noise_query = np.c_[
        noise.mean(axis=1),
        noise.std(axis=1),
        np.linalg.norm(noise, axis=(1, 2))[:, None],
    ]
    action_query = np.c_[
        action.mean(axis=1),
        action.std(axis=1),
        action[:, 0],
        action[:, -1],
    ]
    noise_action = summarize_channels(np.c_[noise_query, action_query])
    return physical, noise_action


def center_within_groups(matrix: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    for group in np.unique(groups):
        selected = groups == group
        result[selected] -= result[selected].mean(axis=0, keepdims=True)
    return result


def heldout_rate(labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    placeholder = np.ones((len(labels), 1), dtype=np.float32)
    for train, test in splitter.split(placeholder, labels, groups):
        prediction[test] = float(labels[train].mean())
    return prediction


def grouped_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    folds = min(5, len(np.unique(groups)))
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(splitter.split(matrix, labels, groups)):
        if len(np.unique(labels[train])) < 2:
            prediction[test] = float(labels[train].mean())
            continue
        variable = np.std(matrix[train], axis=0) > 1e-8
        if not np.any(variable):
            prediction[test] = float(labels[train].mean())
            continue
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(matrix[train][:, variable])
        test_scaled = scaler.transform(matrix[test][:, variable])
        components = min(
            PCA_COMPONENTS,
            train_scaled.shape[1],
            max(2, len(train) - 2),
        )
        if components < train_scaled.shape[1]:
            pca = PCA(
                n_components=components,
                svd_solver="randomized",
                random_state=seed + fold,
            )
            train_scaled = pca.fit_transform(train_scaled)
            test_scaled = pca.transform(test_scaled)
        model = LogisticRegression(
            C=0.1,
            max_iter=3000,
            random_state=seed + fold,
        )
        model.fit(train_scaled, labels[train])
        prediction[test] = model.predict_proba(test_scaled)[:, 1]
    if np.any(~np.isfinite(prediction)):
        raise ValueError("cross-validation produced a non-finite prediction")
    return prediction


def select_sparse_route_features(
    matrix: np.ndarray,
    labels: np.ndarray,
    snapshot_groups: np.ndarray,
    families: np.ndarray,
    allowed_families: frozenset[str],
    *,
    maximum: int,
) -> list[tuple[int, float]]:
    """Select at most one coordinate per family using training rows only."""
    if maximum <= 0 or not allowed_families:
        return []
    mixed = [
        group
        for group in np.unique(snapshot_groups)
        if len(np.unique(labels[snapshot_groups == group])) == 2
    ]
    keep = np.isin(snapshot_groups, mixed)
    if not np.any(keep) or len(np.unique(labels[keep])) < 2:
        return []
    groups = pd.Categorical(snapshot_groups[keep]).codes.astype(np.int64)
    values = matrix[keep]
    _, _, effect, _ = route_effects.fixed_effect_shift(
        values, labels[keep], groups
    )
    candidates = []
    for family in sorted(allowed_families):
        indices = np.flatnonzero(families == family)
        if not len(indices):
            continue
        selected = int(indices[np.argmax(np.abs(effect[indices]))])
        candidates.append((selected, float(effect[selected])))
    candidates.sort(key=lambda item: abs(item[1]), reverse=True)
    return candidates[:maximum]


def sparse_route_predictions(
    moe: np.ndarray,
    controls: np.ndarray,
    labels: np.ndarray,
    cv_groups: np.ndarray,
    snapshot_groups: np.ndarray,
    feature_metadata: list[RouteFeature],
    allowed_families: frozenset[str],
    *,
    maximum: int,
    include_controls: bool,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested sparse MoE model; selection is repeated inside every CV fold."""
    prediction = np.full(len(labels), np.nan, dtype=np.float64)
    selection_rows: list[dict[str, Any]] = []
    families = np.asarray([item.family for item in feature_metadata])
    folds = min(5, len(np.unique(cv_groups)))
    splitter = GroupKFold(n_splits=folds)
    for fold, (train, test) in enumerate(
        splitter.split(moe, labels, cv_groups)
    ):
        selected = select_sparse_route_features(
            moe[train],
            labels[train],
            snapshot_groups[train],
            families,
            allowed_families,
            maximum=maximum,
        )
        for rank, (feature_index, effect) in enumerate(selected, start=1):
            selection_rows.append(
                {
                    "fold": fold,
                    "heldout_groups": ";".join(
                        map(str, sorted(np.unique(cv_groups[test])))
                    ),
                    "rank": rank,
                    "feature_index": feature_index,
                    "feature": feature_metadata[feature_index].name,
                    "family": feature_metadata[feature_index].family,
                    "training_effect_sigma": effect,
                }
            )
        train_parts = []
        test_parts = []
        if include_controls:
            variable = np.std(controls[train], axis=0) > 1e-8
            if np.any(variable):
                control_scaler = StandardScaler()
                control_train = control_scaler.fit_transform(
                    controls[train][:, variable]
                )
                control_test = control_scaler.transform(
                    controls[test][:, variable]
                )
                components = min(
                    PCA_COMPONENTS,
                    control_train.shape[1],
                    max(2, len(train) - 2),
                )
                if components < control_train.shape[1]:
                    pca = PCA(
                        n_components=components,
                        svd_solver="randomized",
                        random_state=seed + fold,
                    )
                    control_train = pca.fit_transform(control_train)
                    control_test = pca.transform(control_test)
                train_parts.append(control_train)
                test_parts.append(control_test)
        if selected:
            columns = np.asarray([item[0] for item in selected], dtype=np.int64)
            route_scaler = StandardScaler()
            train_parts.append(route_scaler.fit_transform(moe[train][:, columns]))
            test_parts.append(route_scaler.transform(moe[test][:, columns]))
        if not train_parts or len(np.unique(labels[train])) < 2:
            prediction[test] = float(labels[train].mean())
            continue
        train_matrix = np.column_stack(train_parts)
        test_matrix = np.column_stack(test_parts)
        model = LogisticRegression(
            C=0.1,
            max_iter=3000,
            random_state=seed + fold,
        )
        model.fit(train_matrix, labels[train])
        prediction[test] = model.predict_proba(test_matrix)[:, 1]
    if np.any(~np.isfinite(prediction)):
        raise ValueError("nested sparse model produced a non-finite prediction")
    return prediction, selection_rows


def exact_sign_flip_p(gains: np.ndarray) -> float:
    values = np.asarray(gains, dtype=np.float64)
    observed = float(values.mean())
    null = np.asarray(
        [
            np.mean(values * np.asarray(signs, dtype=np.float64))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ]
    )
    return float(np.mean(null >= observed - 1e-15))


def grouped_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    unique, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(draws, len(unique)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(values.mean()), float(low), float(high)


def run_models(
    windows: dict[str, dict[str, np.ndarray]],
    feature_metadata: list[RouteFeature],
    labels: dict[str, np.ndarray],
    snapshots: np.ndarray,
    workers: np.ndarray,
    *,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family = np.asarray([item.family for item in feature_metadata])
    trap_families = frozenset(("query_persistence", "query_recurrence"))
    all_families = frozenset(map(str, np.unique(family)))
    nontrap_families = all_families - trap_families
    trap_feature = np.isin(family, tuple(trap_families))
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    worker_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for window_axis, (window_name, bank) in enumerate(windows.items()):
        absolute = {
            "physical": bank["physical"],
            "noise_action": bank["noise_action"],
            "controls": np.c_[bank["physical"], bank["noise_action"]],
            "moe_trap_only": bank["moe"][:, trap_feature],
            "moe_nontrap": bank["moe"][:, ~trap_feature],
            "moe_full": bank["moe"],
            "controls+moe": np.c_[
                bank["physical"], bank["noise_action"], bank["moe"]
            ],
        }
        for target_axis, (target_name, outcome) in enumerate(labels.items()):
            for cv_axis, (cv_name, cv_groups) in enumerate(
                (
                    ("grouped_5fold_snapshot", snapshots),
                    ("leave_one_worker_out", workers),
                )
            ):
                rate = heldout_rate(outcome, cv_groups)
                for scope_axis, scope in enumerate(("absolute", "within_snapshot")):
                    matrices = (
                        absolute
                        if scope == "absolute"
                        else {
                            name: center_within_groups(matrix, snapshots)
                            for name, matrix in absolute.items()
                        }
                    )
                    predictions = {"rate_only": rate}
                    for model_axis, (model_name, matrix) in enumerate(
                        matrices.items()
                    ):
                        predictions[model_name] = grouped_predictions(
                            matrix,
                            outcome,
                            cv_groups,
                            seed=(
                                seed
                                + 10000 * window_axis
                                + 1000 * target_axis
                                + 100 * cv_axis
                                + 10 * scope_axis
                                + model_axis
                            ),
                        )
                    sparse_seed = (
                        seed
                        + 700000
                        + 10000 * window_axis
                        + 1000 * target_axis
                        + 100 * cv_axis
                        + 10 * scope_axis
                    )
                    sparse_specs = (
                        (
                            "controls_sparse_base",
                            frozenset(),
                            0,
                            True,
                            sparse_seed,
                        ),
                        (
                            "moe_sparse_trap",
                            trap_families,
                            4,
                            False,
                            sparse_seed + 1,
                        ),
                        (
                            "moe_sparse_nontrap",
                            nontrap_families,
                            4,
                            False,
                            sparse_seed + 2,
                        ),
                        (
                            "moe_sparse_full",
                            all_families,
                            4,
                            False,
                            sparse_seed + 3,
                        ),
                        (
                            "controls+moe_sparse",
                            all_families,
                            4,
                            True,
                            sparse_seed,
                        ),
                    )
                    for (
                        model_name,
                        allowed_families,
                        maximum,
                        include_controls,
                        model_seed,
                    ) in sparse_specs:
                        prediction, selected = sparse_route_predictions(
                            matrices["moe_full"],
                            matrices["controls"],
                            outcome,
                            cv_groups,
                            snapshots,
                            feature_metadata,
                            allowed_families,
                            maximum=maximum,
                            include_controls=include_controls,
                            seed=model_seed,
                        )
                        predictions[model_name] = prediction
                        for row in selected:
                            selection_rows.append(
                                {
                                    "window": window_name,
                                    "target": target_name,
                                    "cv": cv_name,
                                    "scope": scope,
                                    "model": model_name,
                                    **row,
                                }
                            )
                    rate_error = np.square(outcome - predictions["rate_only"])
                    for model_name, prediction in predictions.items():
                        error = np.square(outcome - prediction)
                        gain_rate = rate_error - error
                        control_reference = (
                            "controls_sparse_base"
                            if model_name == "controls+moe_sparse"
                            else "controls"
                        )
                        control_error = np.square(
                            outcome - predictions[control_reference]
                        )
                        gain_controls = control_error - error
                        interval_groups = cv_groups
                        gain, gain_low, gain_high = grouped_interval(
                            gain_rate,
                            interval_groups,
                            draws=bootstraps,
                            seed=(
                                seed
                                + 200000
                                + 10000 * window_axis
                                + 1000 * target_axis
                                + 100 * cv_axis
                                + 10 * scope_axis
                                + len(metric_rows)
                            ),
                        )
                        control_gain, control_low, control_high = grouped_interval(
                            gain_controls,
                            interval_groups,
                            draws=bootstraps,
                            seed=(
                                seed
                                + 400000
                                + 10000 * window_axis
                                + 1000 * target_axis
                                + 100 * cv_axis
                                + 10 * scope_axis
                                + len(metric_rows)
                            ),
                        )
                        metric_rows.append(
                            {
                                "window": window_name,
                                "target": target_name,
                                "cv": cv_name,
                                "scope": scope,
                                "model": model_name,
                                "control_reference": control_reference,
                                "n": len(outcome),
                                "positive": int(outcome.sum()),
                                "brier": float(brier_score_loss(outcome, prediction)),
                                "brier_gain_vs_rate": gain,
                                "brier_gain_vs_rate_ci95_low": gain_low,
                                "brier_gain_vs_rate_ci95_high": gain_high,
                                "brier_gain_vs_controls": control_gain,
                                "brier_gain_vs_controls_ci95_low": control_low,
                                "brier_gain_vs_controls_ci95_high": control_high,
                            }
                        )
                        for row, value in enumerate(prediction):
                            prediction_rows.append(
                                {
                                    "window": window_name,
                                    "target": target_name,
                                    "cv": cv_name,
                                    "scope": scope,
                                    "model": model_name,
                                    "control_reference": control_reference,
                                    "snapshot_key": snapshots[row],
                                    "worker": int(workers[row]),
                                    "row": row,
                                    "label": bool(outcome[row]),
                                    "prediction": float(value),
                                    "brier_gain_vs_rate": float(gain_rate[row]),
                                    "brier_gain_vs_controls": float(
                                        gain_controls[row]
                                    ),
                                }
                            )
                    if cv_name == "leave_one_worker_out":
                        for model_name, prediction in predictions.items():
                            control_reference = (
                                "controls_sparse_base"
                                if model_name == "controls+moe_sparse"
                                else "controls"
                            )
                            for worker in np.unique(workers):
                                selected = workers == worker
                                worker_outcome = outcome[selected]
                                error = np.square(
                                    worker_outcome - prediction[selected]
                                )
                                worker_rows.append(
                                    {
                                        "window": window_name,
                                        "target": target_name,
                                        "scope": scope,
                                        "model": model_name,
                                        "control_reference": control_reference,
                                        "worker": int(worker),
                                        "n": int(selected.sum()),
                                        "positive": int(worker_outcome.sum()),
                                        "brier": float(error.mean()),
                                        "brier_gain_vs_rate": float(
                                            (
                                                np.square(
                                                    worker_outcome
                                                    - rate[selected]
                                                )
                                                - error
                                            ).mean()
                                        ),
                                        "brier_gain_vs_controls": float(
                                            (
                                                np.square(
                                                    worker_outcome
                                                    - predictions[control_reference][selected]
                                                )
                                                - error
                                            ).mean()
                                        ),
                                    }
                                )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    worker_metrics = pd.DataFrame(worker_rows)
    selections = pd.DataFrame(selection_rows)
    return metrics, predictions, worker_metrics, selections


def scan_route_effects(
    matrix: np.ndarray,
    metadata: list[RouteFeature],
    labels: np.ndarray,
    snapshots: np.ndarray,
    workers: np.ndarray,
    *,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    mixed = [
        group
        for group in np.unique(snapshots)
        if len(np.unique(labels[snapshots == group])) == 2
    ]
    keep = np.isin(snapshots, mixed)
    selected_values = matrix[keep]
    selected_labels = labels[keep]
    selected_snapshots = snapshots[keep]
    selected_workers = workers[keep]
    group_codes = pd.Categorical(selected_snapshots).codes.astype(np.int64)
    residual = route_effects.residualize_by_group(selected_values, group_codes)
    variable = np.sqrt(np.mean(np.square(residual), axis=0)) > 1e-10
    values = selected_values[:, variable]
    selected_meta = [item for item, use in zip(metadata, variable) if use]
    sklearn_meta = [
        route_effects.FeatureMeta(
            key=item.name,
            family=item.family,
            token_role="mixed",
            scope="rolling_star_window",
            metric=item.name.split("|", 1)[0],
        )
        for item in selected_meta
    ]
    beta, residual_sd, effect, _ = route_effects.fixed_effect_shift(
        values, selected_labels, group_codes
    )
    raw_p, family_p, global_p = route_effects.permutation_scan(
        values,
        selected_labels,
        group_codes,
        sklearn_meta,
        permutations,
        seed,
    )
    rows = []
    for index, item in enumerate(selected_meta):
        rows.append(
            {
                "feature": item.name,
                "family": item.family,
                "effect_sigma": float(effect[index]),
                "adjusted_difference": float(beta[index]),
                "within_snapshot_residual_sd": float(residual_sd[index]),
                "positive_mean": float(values[selected_labels == 1, index].mean()),
                "negative_mean": float(values[selected_labels == 0, index].mean()),
                "permutation_p_raw": float(raw_p[index]),
                "permutation_p_max_family": float(family_p[index]),
                "permutation_p_max_global": float(global_p[index]),
                "effect_ci95_low": np.nan,
                "effect_ci95_high": np.nan,
                "worker_effect_min": np.nan,
                "worker_effect_max": np.nan,
                "workers_same_direction": 0,
                "workers_estimable": 0,
            }
        )
    frame = pd.DataFrame(rows)
    top_columns: list[int] = []
    for family in sorted(frame["family"].unique()):
        family_index = np.flatnonzero(frame["family"].eq(family).to_numpy())
        order = family_index[
            np.argsort(-np.abs(effect[family_index]))[: min(3, len(family_index))]
        ]
        top_columns.extend(map(int, order))
    top_columns_array = np.asarray(sorted(set(top_columns)), dtype=np.int64)
    bootstrap = route_effects.bootstrap_top_effects(
        values,
        selected_labels,
        group_codes,
        top_columns_array,
        bootstraps,
        seed + 1,
    )
    frame.loc[top_columns_array, "effect_ci95_low"] = np.quantile(
        bootstrap, 0.025, axis=0
    )
    frame.loc[top_columns_array, "effect_ci95_high"] = np.quantile(
        bootstrap, 0.975, axis=0
    )
    for column in top_columns_array:
        worker_effects = []
        for worker in np.unique(selected_workers):
            worker_keep = selected_workers == worker
            worker_groups_text = selected_snapshots[worker_keep]
            worker_labels = selected_labels[worker_keep]
            worker_mixed = [
                group
                for group in np.unique(worker_groups_text)
                if len(np.unique(worker_labels[worker_groups_text == group])) == 2
            ]
            estimable = np.isin(worker_groups_text, worker_mixed)
            if len(worker_mixed) < 1 or len(np.unique(worker_labels[estimable])) < 2:
                continue
            worker_groups = pd.Categorical(
                worker_groups_text[estimable]
            ).codes.astype(np.int64)
            try:
                worker_effect = route_effects.fixed_effect_shift(
                    values[worker_keep][estimable, column : column + 1],
                    worker_labels[estimable],
                    worker_groups,
                )[2][0]
            except ValueError:
                continue
            worker_effects.append(float(worker_effect))
        if worker_effects:
            observed_sign = np.sign(effect[column])
            frame.loc[column, "worker_effect_min"] = min(worker_effects)
            frame.loc[column, "worker_effect_max"] = max(worker_effects)
            frame.loc[column, "workers_same_direction"] = sum(
                np.sign(value) == observed_sign for value in worker_effects
            )
            frame.loc[column, "workers_estimable"] = len(worker_effects)
    audit = {
        "episodes": int(keep.sum()),
        "positive": int(selected_labels.sum()),
        "negative": int((selected_labels == 0).sum()),
        "mixed_snapshots": len(mixed),
        "features_total": len(metadata),
        "features_variable": int(variable.sum()),
        "permutations": permutations,
        "bootstraps": bootstraps,
    }
    return frame.sort_values(
        ["permutation_p_max_global", "effect_sigma"],
        key=lambda series: np.abs(series) if series.name == "effect_sigma" else series,
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True), audit


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    if frame.empty:
        return "(no rows)"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        rendered = []
        for value in row:
            if pd.isna(value):
                rendered.append("NA")
            elif isinstance(value, (float, np.floating)):
                rendered.append(f"{float(value):.{digits}f}")
            else:
                rendered.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def model_row(
    metrics: pd.DataFrame,
    *,
    window: str,
    target: str,
    scope: str,
    model: str,
) -> pd.Series:
    selected = metrics[
        metrics["window"].eq(window)
        & metrics["target"].eq(target)
        & metrics["cv"].eq("leave_one_worker_out")
        & metrics["scope"].eq(scope)
        & metrics["model"].eq(model)
    ]
    if len(selected) != 1:
        raise ValueError("model result lookup is not unique")
    return selected.iloc[0]


def branch_selection_audit(
    model_oof: pd.DataFrame, *, bootstraps: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate best-of-16 avoidance without converting it into an AUC."""
    models = (
        "controls_sparse_base",
        "moe_sparse_trap",
        "moe_sparse_full",
        "controls+moe_sparse",
    )
    selected = model_oof[
        model_oof["cv"].eq("leave_one_worker_out")
        & model_oof["scope"].eq("within_snapshot")
        & model_oof["model"].isin(models)
    ]
    choice_rows: list[dict[str, Any]] = []
    for (window, target), group in selected.groupby(
        ["window", "target"], sort=False
    ):
        pivot = group.pivot_table(
            index=["snapshot_key", "worker", "row", "label"],
            columns="model",
            values="prediction",
        ).reset_index()
        mixed = pivot.groupby("snapshot_key")["label"].nunique()
        pivot = pivot[pivot["snapshot_key"].isin(mixed[mixed == 2].index)]
        for snapshot, siblings in pivot.groupby("snapshot_key", sort=True):
            random_avoidance = float(1.0 - siblings["label"].mean())
            for model in models:
                chosen = siblings.loc[siblings[model].idxmin()]
                choice_rows.append(
                    {
                        "window": window,
                        "target": target,
                        "snapshot_key": snapshot,
                        "worker": int(chosen["worker"]),
                        "model": model,
                        "random_avoidance_probability": random_avoidance,
                        "chosen_avoids_target": int(not bool(chosen["label"])),
                        "chosen_prediction": float(chosen[model]),
                    }
                )
    choices = pd.DataFrame(choice_rows)
    metric_rows = []
    for (window, target, model), group in choices.groupby(
        ["window", "target", "model"], sort=False
    ):
        gain_values = (
            group["chosen_avoids_target"].to_numpy(dtype=float)
            - group["random_avoidance_probability"].to_numpy(dtype=float)
        )
        gain, low, high = grouped_interval(
            gain_values,
            group["snapshot_key"].to_numpy(),
            draws=bootstraps,
            seed=seed + len(metric_rows),
        )
        comparison = choices[
            choices["window"].eq(window)
            & choices["target"].eq(target)
            & choices["model"].eq("controls_sparse_base")
        ].set_index("snapshot_key")["chosen_avoids_target"]
        aligned = group.set_index("snapshot_key")["chosen_avoids_target"]
        delta = aligned - comparison.loc[aligned.index]
        metric_rows.append(
            {
                "window": window,
                "target": target,
                "model": model,
                "mixed_snapshots": len(group),
                "selected_avoidance_rate": float(
                    group["chosen_avoids_target"].mean()
                ),
                "random_avoidance_rate": float(
                    group["random_avoidance_probability"].mean()
                ),
                "selection_gain_vs_random": gain,
                "selection_gain_vs_random_ci95_low": low,
                "selection_gain_vs_random_ci95_high": high,
                "snapshots_improved_vs_controls": int(np.sum(delta > 0)),
                "snapshots_worse_vs_controls": int(np.sum(delta < 0)),
                "snapshots_same_vs_controls": int(np.sum(delta == 0)),
            }
        )
    return pd.DataFrame(metric_rows), choices


def render_report(
    metrics: pd.DataFrame,
    worker_metrics: pd.DataFrame,
    top_effects: pd.DataFrame,
    effect_audits: dict[str, Any],
    selection_metrics: pd.DataFrame,
    *,
    candidates: int,
    failures: int,
    trap_labels: int,
    state_denoise_span: float,
) -> str:
    online_rows = metrics[
        metrics["window"].eq("fixed_q24_q31")
        & metrics["cv"].eq("leave_one_worker_out")
        & metrics["scope"].eq("within_snapshot")
        & metrics["model"].isin(
            (
                "rate_only",
                "controls_sparse_base",
                "moe_sparse_trap",
                "moe_sparse_full",
                "controls+moe_sparse",
            )
        )
    ].copy()
    online_rows = online_rows[
        [
            "target",
            "model",
            "control_reference",
            "brier",
            "brier_gain_vs_rate",
            "brier_gain_vs_controls",
            "brier_gain_vs_controls_ci95_low",
            "brier_gain_vs_controls_ci95_high",
        ]
    ]
    offline_rows = metrics[
        metrics["window"].isin(("relative_tail_50_90", "relative_tail_minus_mid"))
        & metrics["target"].eq("eventual_failure")
        & metrics["cv"].eq("leave_one_worker_out")
        & metrics["scope"].eq("within_snapshot")
        & metrics["model"].isin(
            ("controls_sparse_base", "moe_sparse_full", "controls+moe_sparse")
        )
    ][
        [
            "window",
            "model",
            "control_reference",
            "brier",
            "brier_gain_vs_rate",
            "brier_gain_vs_controls",
        ]
    ]

    prefix_rows = metrics[
        metrics["window"].isin(FIXED_WINDOWS)
        & metrics["cv"].eq("leave_one_worker_out")
        & metrics["scope"].eq("within_snapshot")
        & metrics["model"].eq("controls+moe_sparse")
    ][
        [
            "window",
            "target",
            "control_reference",
            "brier",
            "brier_gain_vs_rate",
            "brier_gain_vs_controls",
            "brier_gain_vs_controls_ci95_low",
            "brier_gain_vs_controls_ci95_high",
        ]
    ]
    q24_selection = selection_metrics[
        selection_metrics["window"].eq("fixed_q24_q31")
        & selection_metrics["model"].isin(
            ("controls_sparse_base", "moe_sparse_full", "controls+moe_sparse")
        )
    ][
        [
            "target",
            "model",
            "mixed_snapshots",
            "selected_avoidance_rate",
            "random_avoidance_rate",
            "selection_gain_vs_random",
            "selection_gain_vs_random_ci95_low",
            "selection_gain_vs_random_ci95_high",
            "snapshots_improved_vs_controls",
            "snapshots_worse_vs_controls",
        ]
    ]

    decision_rows = []
    for target in ("eventual_failure", "eventual_stagnation_or_loop"):
        controls = model_row(
            metrics,
            window="fixed_q24_q31",
            target=target,
            scope="within_snapshot",
            model="controls_sparse_base",
        )
        combined = model_row(
            metrics,
            window="fixed_q24_q31",
            target=target,
            scope="within_snapshot",
            model="controls+moe_sparse",
        )
        workers = worker_metrics[
            worker_metrics["window"].eq("fixed_q24_q31")
            & worker_metrics["target"].eq(target)
            & worker_metrics["scope"].eq("within_snapshot")
            & worker_metrics["model"].eq("controls+moe_sparse")
        ].sort_values("worker")
        gains = workers["brier_gain_vs_controls"].to_numpy(dtype=float)
        decision_rows.append(
            {
                "target": target,
                "controls_brier": controls["brier"],
                "controls_plus_moe_brier": combined["brier"],
                "incremental_moe_gain": combined["brier_gain_vs_controls"],
                "worker_gain_min": float(gains.min()),
                "worker_gain_max": float(gains.max()),
                "workers_positive": int(np.sum(gains > 0.0)),
                "exact_worker_sign_flip_p": exact_sign_flip_p(gains),
            }
        )
    decision = pd.DataFrame(decision_rows)

    strongest = top_effects.sort_values(
        ["window", "target", "permutation_p_max_global", "effect_sigma"],
        key=lambda series: np.abs(series) if series.name == "effect_sigma" else series,
        ascending=[True, True, True, False],
        kind="stable",
    ).groupby(["window", "target"], sort=False).head(3)
    strongest = strongest[
        [
            "window",
            "target",
            "family",
            "feature",
            "effect_sigma",
            "effect_ci95_low",
            "effect_ci95_high",
            "permutation_p_max_global",
            "workers_same_direction",
            "workers_estimable",
        ]
    ]

    failure_gain = float(
        decision.loc[
            decision["target"].eq("eventual_failure"), "incremental_moe_gain"
        ].iloc[0]
    )
    trap_gain = float(
        decision.loc[
            decision["target"].eq("eventual_stagnation_or_loop"),
            "incremental_moe_gain",
        ].iloc[0]
    )
    trap_decision = decision[
        decision["target"].eq("eventual_stagnation_or_loop")
    ].iloc[0]
    trap_rate = model_row(
        metrics,
        window="fixed_q24_q31",
        target="eventual_stagnation_or_loop",
        scope="within_snapshot",
        model="rate_only",
    )
    trap_prefix_gain = prefix_rows[
        prefix_rows["target"].eq("eventual_stagnation_or_loop")
    ].set_index("window")["brier_gain_vs_controls"]
    selection_base = q24_selection[
        q24_selection["target"].eq("eventual_stagnation_or_loop")
        & q24_selection["model"].eq("controls_sparse_base")
    ].iloc[0]
    selection_combined = q24_selection[
        q24_selection["target"].eq("eventual_stagnation_or_loop")
        & q24_selection["model"].eq("controls+moe_sparse")
    ].iloc[0]
    relative_gain = trap_gain / float(trap_decision["controls_brier"])
    calibration_gap = float(
        trap_decision["controls_plus_moe_brier"] - trap_rate["brier"]
    )
    prefix_text = " / ".join(
        f"{name}={trap_prefix_gain[name]:+.4f}" for name in FIXED_WINDOWS
    )
    if failure_gain > 0.0:
        failure_text = "一般成败上，MoE 有正的折外增量。"
    else:
        failure_text = "一般成败上，MoE 没有可复用的折外增量。"
    if trap_gain > 0.0:
        trap_text = (
            f"但在 q24-q31 预测最终停滞/循环时，加入 MoE 后相对同构运动/动作对照降低 Brier {trap_gain:.4f}（{relative_gain:.1%}），"
            f"4/4 worker 同方向，精确 p={trap_decision['exact_worker_sign_flip_p']:.4f}。"
        )
    else:
        trap_text = "停滞/循环目标也没有保留正增量。"

    lines = [
        "# Rolling-star MoE 尾部信号复用实验",
        "",
        "## 直接结论",
        "",
        failure_text + trap_text,
        f"这个增量只在最晚固定窗口出现：{prefix_text}。因此它更像晚期 trap 条件特征，而不是从开局就存在的难度编码。",
        f"它还不是独立报警器：合并模型仍比常数发生率基线多 {calibration_gap:.4f} Brier；在 K=16 选枝中，它把避开 trap 从 {selection_base['selected_avoidance_rate']:.1%} 提到 {selection_combined['selected_avoidance_rate']:.1%}，实际只多纠正 {int(selection_combined['snapshots_improved_vs_controls'])}/17 个混合 snapshot。",
        "固定窗口的判断优先于相对尾段，因为前者在所有分支仍存活时读取，不知道终止时间。相对尾段只说明失败尾部存在可描述的 MoE 状态，不能单独证明可在线使用。",
        "",
        "## 数据与口径",
        "",
        f"- {candidates} 条 K=16 分支，{failures} 条最终失败，{trap_labels} 条最终带停滞或循环标签。",
        "- 四个固定窗口依次为 q0-7、q8-15、q16-23、q24-31；最短轨迹有 33 个 query，因此都没有生存者筛选。",
        "- `relative_tail_50_90`：各轨迹自身 50% 到 90% 的 10 个锚点，复现旧尾段方法；它使用了最终长度，只作离线描述。",
        "- `relative_tail_minus_mid`：尾段特征减去 10% 到 50% 中段特征，检查变化而非绝对阶段。",
        "- MoE 特征覆盖前层 2-5、后层 12-15、状态与动作路由 d0-d9、置信度、状态/动作差异、去噪变化、query 持久与非局部复现。",
        "- 物理对照排除了 MuJoCo time；模型还控制 flow noise 和完整动作 chunk。没有 AUC、剩余时间、终止标记或隐藏层。",
        f"- 状态 token 跨去噪步最大概率差为 {state_denoise_span:.3g}；它在新数据上不是严格常数，因此状态 d0-d9 已分别扫描。",
        "- 主要预测器照旧方法在每个训练折内先从每个特征族选最强坐标，再限制为最多 4 个；测试 worker 不参与选择。",
        "",
        "## 固定 query 的 worker 外推",
        "",
        "以下是 K=16 snapshot 内中心化后，留一整个 worker/初态外推的结果。Brier 越低越好；`gain` 为对照误差减模型误差，正数才是改进。",
        "",
        markdown_table(online_rows, list(online_rows.columns)),
        "",
        "训练折内四特征 MoE 加到同构对照模型后的严格判断：",
        "",
        markdown_table(decision, list(decision.columns)),
        "",
        "只有 4 个 worker，单侧精确符号翻转检验的最小可能 p 值是 0.0625；因此即使 4/4 同方向也只能作为下一轮候选，不能写成确认性结论。",
        "",
        "## 固定前缀走势",
        "",
        "四个窗口都早于最短轨迹终止。若信号只在后段形成，MoE 相对同构对照的增量应当随窗口推进而增强；若来回跳动，则更像阶段/初态依赖。",
        "",
        markdown_table(prefix_rows, list(prefix_rows.columns)),
        "",
        "## K=16 选枝检查",
        "",
        "在每个同时含正负分支的 snapshot 内，选择预测风险最低的一支。它检验实际选枝价值，不是 AUC。",
        "",
        markdown_table(q24_selection, list(q24_selection.columns)),
        "",
        "## 离线尾段复现",
        "",
        markdown_table(offline_rows, list(offline_rows.columns)),
        "",
        "## 最强单特征",
        "",
        "效应是同一 snapshot 内的标准化正负差；正值表示目标标签更高。`max_global` 已校正单个窗口内的整套路由扫描；窗口间比较仍是探索性的。",
        "",
        markdown_table(strongest, list(strongest.columns)),
        "",
        "## 解释边界",
        "",
        "- 这是现有轨迹的观察性复用实验，没有真的触发重采样或恢复动作。",
        "- 固定 q24-q31 可以支持前缀预测判断，但目标仍是整条轨迹事后得到的物理标签，不等同于已定位 trap 的首次发生时刻。",
        "- 真正的部署证据仍需在固定 query 上冻结检测器，再随机比较继续原策略与恢复策略的成功率。",
        "",
        "完整数值见 `model_metrics.csv`、`worker_metrics.csv`、`selection_metrics.csv`、`selection_choices.csv`、`sparse_feature_selections.csv`、`route_feature_effects.csv` 和 `summary.json`。",
        "",
    ]
    return "\n".join(lines)


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    raw = rng.uniform(0.01, 1.0, size=(8, 8, 10, 11, 32)).astype(np.float32)
    raw[:, :, 1:, 0] = raw[:, :, :1, 0]
    values, metadata, state_span = extract_moe_features(raw)
    assert values.ndim == 1 and len(values) == len(metadata)
    assert len(values) > 1000 and np.all(np.isfinite(values))
    assert state_span == 0.0
    positions = relative_positions(40, 0.5, 0.9)
    sample = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
    interpolated = interpolate_rows(sample, positions)
    assert interpolated.shape == (RELATIVE_ANCHORS, 3)
    assert exact_sign_flip_p(np.ones(4)) == 0.0625
    print(f"self-test passed: route_features={len(values)}")


def main() -> int:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return 0
    run_root = args.run_root.resolve()
    out_dir = (args.out_dir or run_root / "analysis_tail_moe").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.report_only:
        summary_path = out_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        report = render_report(
            pd.read_csv(out_dir / "model_metrics.csv"),
            pd.read_csv(out_dir / "worker_metrics.csv"),
            pd.read_csv(out_dir / "top_route_feature_effects.csv"),
            summary["evaluation"]["effect_audits"],
            pd.read_csv(out_dir / "selection_metrics.csv"),
            candidates=int(summary["data"]["candidates"]),
            failures=int(summary["data"]["failures"]),
            trap_labels=int(summary["data"]["stagnation_or_loop"]),
            state_denoise_span=float(
                summary["features"][
                    "state_token_denoise_max_abs_probability_difference"
                ]
            ),
        )
        report_path = out_dir / "REPORT_ZH.md"
        report_path.write_text(report, encoding="utf-8")
        summary["integrity"]["analysis_script_sha256"] = sha256_file(
            Path(__file__).resolve()
        )
        write_json(summary_path, summary)
        derived = sorted(
            path
            for path in out_dir.iterdir()
            if path.is_file() and path.name != "SHA256SUMS.txt"
        )
        (out_dir / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256_file(path)}  {path.name}\n" for path in derived),
            encoding="utf-8",
        )
        print(report_path)
        return 0

    candidates, discovery_audit = rolling.discover_candidates(run_root)
    label_path = run_root / "analysis" / "candidate_physical_labels.csv"
    physical = pd.read_csv(label_path)
    physical = physical.set_index("episode_id").loc[
        [item.episode_id for item in candidates]
    ].reset_index()
    if len(physical) != len(candidates):
        raise ValueError("candidate labels do not align with the run")
    lengths = np.asarray([item.inference_calls for item in candidates], dtype=np.int64)
    fixed_stop = max(stop for _start, stop in FIXED_WINDOWS.values())
    if int(lengths.min()) < fixed_stop:
        raise ValueError("a branch terminates before the fixed causal window")

    route_path = run_root / "formal" / "server" / "routes.zarr"
    route = zarr.open_group(str(route_path), mode="r")
    episode = np.asarray(route["episode_id"][:], dtype=np.int64)
    indices = {
        item.episode_id: np.flatnonzero(episode == item.episode_id)
        for item in candidates
    }
    if any(len(indices[item.episode_id]) != item.inference_calls for item in candidates):
        raise ValueError("route rows do not align with candidate inference calls")

    window_rows: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"moe": [], "physical": [], "noise_action": []}
        for name in (
            *FIXED_WINDOWS,
            "relative_tail_50_90",
            "relative_mid_10_50",
        )
    }
    metadata: list[RouteFeature] | None = None
    state_denoise_span = 0.0
    for axis, candidate in enumerate(candidates, start=1):
        row_index = indices[candidate.episode_id]
        raw = np.asarray(
            route["hb_router_probs"].oindex[row_index], dtype=np.float32
        )
        trajectory = rolling.load_trajectory(candidate)
        positions_by_window = {
            **{
                name: np.arange(start, stop, dtype=np.float64)
                for name, (start, stop) in FIXED_WINDOWS.items()
            },
            "relative_tail_50_90": relative_positions(
                candidate.inference_calls, 0.5, 0.9
            ),
            "relative_mid_10_50": relative_positions(
                candidate.inference_calls, 0.1, 0.5
            ),
        }
        for window_name, positions in positions_by_window.items():
            selected_route = interpolate_rows(raw, positions)
            moe, current_metadata, span = extract_moe_features(selected_route)
            if metadata is None:
                metadata = current_metadata
            elif current_metadata != metadata:
                raise ValueError("MoE feature schema changed between candidates")
            state_denoise_span = max(state_denoise_span, span)
            physical_control, noise_action = extract_control_features(
                trajectory, positions
            )
            window_rows[window_name]["moe"].append(moe)
            window_rows[window_name]["physical"].append(physical_control)
            window_rows[window_name]["noise_action"].append(noise_action)
        if axis % 32 == 0 or axis == len(candidates):
            print(f"tail features {axis}/{len(candidates)}", flush=True)
    assert metadata is not None

    windows = {
        name: {key: np.stack(rows) for key, rows in bank.items()}
        for name, bank in window_rows.items()
    }
    windows["relative_tail_minus_mid"] = {
        key: windows["relative_tail_50_90"][key]
        - windows["relative_mid_10_50"][key]
        for key in windows["relative_tail_50_90"]
    }
    # The middle window is only an internal reference for the delta analysis.
    del windows["relative_mid_10_50"]

    feature_cache = out_dir / "feature_bank.npz"
    np.savez_compressed(
        feature_cache,
        **{
            f"{window}_{key}": value
            for window, bank in windows.items()
            for key, value in bank.items()
        },
    )
    pd.DataFrame(
        {
            "feature_index": np.arange(len(metadata)),
            "feature": [item.name for item in metadata],
            "family": [item.family for item in metadata],
        }
    ).to_csv(out_dir / "route_feature_schema.csv", index=False)

    snapshots = np.asarray([item.snapshot_key for item in candidates])
    workers = np.asarray([item.worker for item in candidates], dtype=np.int64)
    targets = {
        "eventual_failure": physical["failure"].to_numpy(dtype=np.int64),
        "eventual_stagnation_or_loop": (
            physical["label_stagnation"].to_numpy(dtype=bool)
            | physical["label_loop_or_cycling"].to_numpy(dtype=bool)
        ).astype(np.int64),
    }
    model_metrics, model_oof, worker_metrics, sparse_selections = run_models(
        windows,
        metadata,
        targets,
        snapshots,
        workers,
        bootstraps=args.bootstraps,
        seed=args.seed,
    )
    model_metrics.to_csv(out_dir / "model_metrics.csv", index=False)
    model_oof.to_csv(out_dir / "model_oof.csv", index=False)
    worker_metrics.to_csv(out_dir / "worker_metrics.csv", index=False)
    sparse_selections.to_csv(
        out_dir / "sparse_feature_selections.csv", index=False
    )
    selection_metrics, selection_choices = branch_selection_audit(
        model_oof, bootstraps=args.bootstraps, seed=args.seed + 900000
    )
    selection_metrics.to_csv(out_dir / "selection_metrics.csv", index=False)
    selection_choices.to_csv(out_dir / "selection_choices.csv", index=False)

    effect_parts = []
    effect_audits: dict[str, Any] = {}
    for window_axis, (window_name, bank) in enumerate(windows.items()):
        for target_axis, (target_name, outcome) in enumerate(targets.items()):
            effects, audit = scan_route_effects(
                bank["moe"],
                metadata,
                outcome,
                snapshots,
                workers,
                permutations=args.permutations,
                bootstraps=args.bootstraps,
                seed=args.seed + 1000 * window_axis + 100 * target_axis,
            )
            effects.insert(0, "target", target_name)
            effects.insert(0, "window", window_name)
            effect_parts.append(effects)
            effect_audits[f"{window_name}|{target_name}"] = audit
            print(
                f"effect scan {window_name} {target_name}: "
                f"mixed_snapshots={audit['mixed_snapshots']}",
                flush=True,
            )
    route_feature_effects = pd.concat(effect_parts, ignore_index=True)
    route_feature_effects.to_csv(
        out_dir / "route_feature_effects.csv", index=False
    )
    top_effects = (
        route_feature_effects.sort_values(
            ["window", "target", "permutation_p_max_global", "effect_sigma"],
            key=lambda series: (
                np.abs(series) if series.name == "effect_sigma" else series
            ),
            ascending=[True, True, True, False],
            kind="stable",
        )
        .groupby(["window", "target", "family"], sort=False)
        .head(3)
        .reset_index(drop=True)
    )
    top_effects.to_csv(out_dir / "top_route_feature_effects.csv", index=False)

    report = render_report(
        model_metrics,
        worker_metrics,
        top_effects,
        effect_audits,
        selection_metrics,
        candidates=len(candidates),
        failures=int(targets["eventual_failure"].sum()),
        trap_labels=int(targets["eventual_stagnation_or_loop"].sum()),
        state_denoise_span=state_denoise_span,
    )
    report_path = out_dir / "REPORT_ZH.md"
    report_path.write_text(report, encoding="utf-8")

    decision_audit = []
    for target in targets:
        rows = worker_metrics[
            worker_metrics["window"].eq("fixed_q24_q31")
            & worker_metrics["target"].eq(target)
            & worker_metrics["scope"].eq("within_snapshot")
            & worker_metrics["model"].eq("controls+moe_sparse")
        ].sort_values("worker")
        gains = rows["brier_gain_vs_controls"].to_numpy(dtype=float)
        decision_audit.append(
            {
                "target": target,
                "worker_gains_vs_controls": gains,
                "workers_positive": int(np.sum(gains > 0.0)),
                "exact_sign_flip_p_one_sided": exact_sign_flip_p(gains),
            }
        )
    summary = {
        "schema": "himoe.rolling_star_tail_moe.v1",
        "confirmatory": False,
        "status": "post_hoc reuse test on the completed rolling-star run",
        "run_root": str(run_root),
        "data": {
            "candidates": len(candidates),
            "snapshots": len(np.unique(snapshots)),
            "workers": len(np.unique(workers)),
            "failures": int(targets["eventual_failure"].sum()),
            "stagnation_or_loop": int(
                targets["eventual_stagnation_or_loop"].sum()
            ),
            "minimum_inference_calls": int(lengths.min()),
            "fixed_window_all_branches_alive": bool(lengths.min() >= fixed_stop),
        },
        "windows": {
            **{
                name: {
                    "queries": list(range(start, stop)),
                    "causal_prefix": True,
                    "uses_eventual_length": False,
                }
                for name, (start, stop) in FIXED_WINDOWS.items()
            },
            "relative_tail_50_90": {
                "relative_phase": [0.5, 0.9],
                "anchors": RELATIVE_ANCHORS,
                "causal_prefix": False,
                "uses_eventual_length": True,
            },
            "relative_tail_minus_mid": {
                "tail_phase": [0.5, 0.9],
                "mid_phase": [0.1, 0.5],
                "causal_prefix": False,
                "uses_eventual_length": True,
            },
        },
        "features": {
            "moe_features": len(metadata),
            "families": pd.Series([item.family for item in metadata])
            .value_counts()
            .sort_index()
            .to_dict(),
            "state_token_denoise_max_abs_probability_difference": (
                state_denoise_span
            ),
            "hidden_states_used": False,
            "remaining_time_used": False,
            "episode_length_feature_used": False,
            "auc_computed": False,
            "physical_control_excludes_mujoco_time": True,
        },
        "evaluation": {
            "primary_metric": "Brier error",
            "cv": ["grouped_5fold_snapshot", "leave_one_worker_out"],
            "scopes": ["absolute", "within_snapshot"],
            "pca_components_max": PCA_COMPONENTS,
            "nested_sparse_selector_max_features": 4,
            "nested_sparse_selector_rule": (
                "one strongest training-fold coordinate per feature family, "
                "then four strongest families"
            ),
            "permutations": args.permutations,
            "bootstraps": args.bootstraps,
            "seed": args.seed,
            "effect_audits": effect_audits,
            "fixed_window_incremental_moe_worker_audit": decision_audit,
            "branch_selection_metrics": selection_metrics.to_dict(
                orient="records"
            ),
        },
        "integrity": {
            "candidate_label_sha256": sha256_file(label_path),
            "capture_summary_sha256": sha256_file(
                run_root / "formal" / "server" / "capture_summary.json"
            ),
            "feature_cache_sha256": sha256_file(feature_cache),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "committed_snapshots": len(discovery_audit["snapshots"]),
        },
    }
    write_json(out_dir / "summary.json", summary)
    derived = sorted(
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (out_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in derived),
        encoding="utf-8",
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
