#!/usr/bin/env python3
"""Build a failure-mode atlas from MoE routing dynamics without timeout features.

The statistical unit is a rollout.  The primary comparisons use failed
rollouts only and permute labels inside task x initial-state strata, so all
compared episodes have the same task-specific horizon.  Routing features come
from relative phase 0.5--0.9; episode length, remaining time, final query, and
success/failure are absent from the feature matrix.

The experiment asks two questions:

1. Is the common repeated routing response a fixed-point/persistence signal or
   a genuine nonlocal loop?
2. After persistence features are accounted for, do confidence, denoising,
   token-disagreement, and state/action-mismatch features identify additional
   physical failure modes?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import analyze_state_route_layer_denoise_effects as layer_effects


HERE = pathlib.Path(__file__).resolve().parent
ANALYSIS = HERE / "analysis"
FEATURE_CACHE = ANALYSIS / "all-outcome-routing-clusters/feature_cache"
TAXONOMY = ANALYSIS / "failure-behavior-taxonomy/episode_features_labels.csv"
EVENTS = ANALYSIS / "failure-event-audit/episode_events.csv"
PERIODICITY = ANALYSIS / "failure-rollout-periodicity/episode_metrics.csv"
DEFAULT_OUT = ANALYSIS / "residual-failure-moe-atlas-20260828"

SEED = 20260828
PERMUTATIONS = 2000
BOOTSTRAPS = 2000
PHASE_BINS = 10
N_DENOISE = 10
CONTROL_QUANTILE = 0.90
SUCCESS_SHRINKAGE = 8.0

LAYER_GROUPS = {
    "front_2_5": np.arange(0, 4),
    "back_12_15": np.arange(4, 8),
}
SEQUENCE_METRICS = (
    "state_entropy",
    "action_entropy",
    "state_top1",
    "action_top1",
    "state_top4",
    "action_top4",
    "action_token_dispersion",
    "state_action_gap",
)
TRANSITION_METRICS = (
    "state_denoise_motion",
    "action_denoise_motion",
    "state_hard_switch",
    "action_hard_switch",
)
SEQUENCE_STATS = ("denoise_mean", "denoise_std", "denoise_slope")
TRANSITION_STATS = (
    "denoise_mean",
    "denoise_std",
    "denoise_final",
    "denoise_max",
)
PHASE_STATS = (
    "phase_mean",
    "phase_late",
    "late_minus_mid",
    "phase_std",
    "phase_slope",
    "query_change",
)

METRIC_FAMILY = {
    "state_entropy": "state_confidence",
    "state_top1": "state_confidence",
    "state_top4": "state_confidence",
    "action_entropy": "action_confidence",
    "action_top1": "action_confidence",
    "action_top4": "action_confidence",
    "action_token_dispersion": "token_disagreement",
    "state_action_gap": "state_action_mismatch",
    "state_denoise_motion": "denoise_soft_motion",
    "action_denoise_motion": "denoise_soft_motion",
    "state_hard_switch": "denoise_hard_switch",
    "action_hard_switch": "denoise_hard_switch",
}

TRAP_FAMILIES = frozenset(("query_persistence", "query_recurrence"))
DYNAMICS_FAMILIES = TRAP_FAMILIES | frozenset(
    ("denoise_soft_motion", "denoise_hard_switch")
)
ALL_FAMILIES = frozenset(METRIC_FAMILY.values()) | TRAP_FAMILIES
NONTRAP_FAMILIES = ALL_FAMILIES - TRAP_FAMILIES
BLOCKS = {
    "trap_only": TRAP_FAMILIES,
    "trap_plus_denoise": DYNAMICS_FAMILIES,
    "nontrap_only": NONTRAP_FAMILIES,
    "full_invariant": ALL_FAMILIES,
}
CALIBRATIONS = ("other_failure_q90", "dual_control_q90")

PRIMARY_LABELS = (
    "label_stagnation",
    "label_regrasp_or_drop",
    "label_goal_regression",
    "residual_none",
    "label_active_return",
)
EXPLORATORY_LABELS = (
    "label_gripper_cycling",
    "label_subtask_undo",
)
LABEL_NAMES = {
    "label_stagnation": "stagnation",
    "label_regrasp_or_drop": "regrasp_or_drop",
    "label_goal_regression": "goal_regression",
    "residual_none": "residual_none",
    "label_active_return": "active_return",
    "label_gripper_cycling": "gripper_cycling",
    "label_subtask_undo": "subtask_undo",
}


@dataclass(frozen=True)
class FeatureBank:
    values: np.ndarray
    names: list[str]
    families: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", type=pathlib.Path, default=FEATURE_CACHE)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAPS)
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
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def load_manifest_array(
    cache: pathlib.Path, manifest: dict[str, Any], key: str
) -> np.ndarray:
    entry = manifest["arrays"][key]
    array = np.load(cache / entry["file"], mmap_mode="r")
    if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
        raise ValueError(f"cached array metadata mismatch: {key}")
    return array


def load_cached_inputs(
    cache: pathlib.Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads((cache / "manifest.json").read_text())

    def arrays(key: str) -> np.ndarray:
        return load_manifest_array(cache, manifest, key)

    metadata = pd.DataFrame(
        {
            "task": np.asarray(arrays("meta_task")).astype(str),
            "episode": np.asarray(arrays("meta_episode"), dtype=np.int64),
            "init_state_id": np.asarray(
                arrays("meta_init_state_id"), dtype=np.int64
            ),
            "flow_noise_seed": np.asarray(
                arrays("meta_flow_noise_seed"), dtype=np.int64
            ),
            "episode_length": np.asarray(
                arrays("meta_episode_length"), dtype=np.int64
            ),
            "is_failure": np.asarray(arrays("meta_failure"), dtype=bool),
        }
    )
    geometry = np.asarray(arrays("feature_truncate90_geometry"), dtype=np.float32)
    recurrence = np.asarray(
        arrays("feature_truncate90_recurrence"), dtype=np.float32
    )
    if len(metadata) != 2560 or int(metadata["is_failure"].sum()) != 307:
        raise ValueError("unexpected all-outcome cache coverage")
    return metadata, geometry, recurrence, manifest


def phase_summaries(curve: np.ndarray) -> dict[str, np.ndarray]:
    if curve.ndim != 2 or curve.shape[1] != PHASE_BINS:
        raise ValueError("phase curve must have ten bins")
    time = np.linspace(-1.0, 1.0, PHASE_BINS, dtype=np.float32)
    return {
        "phase_mean": curve.mean(axis=1),
        "phase_late": curve[:, -3:].mean(axis=1),
        "late_minus_mid": curve[:, -3:].mean(axis=1)
        - curve[:, :3].mean(axis=1),
        "phase_std": curve.std(axis=1),
        "phase_slope": (curve * time).sum(axis=1) / float(time @ time),
        "query_change": np.abs(np.diff(curve, axis=1)).mean(axis=1),
    }


def add_geometry_block(
    columns: list[np.ndarray],
    names: list[str],
    families: list[str],
    metric: str,
    block: np.ndarray,
    statistics: tuple[str, ...],
) -> None:
    family = METRIC_FAMILY[metric]
    for statistic_axis, statistic in enumerate(statistics):
        for group, layer_axes in LAYER_GROUPS.items():
            curve = block[:, :, layer_axes, statistic_axis].mean(axis=2)
            for phase_statistic, values in phase_summaries(curve).items():
                columns.append(values)
                names.append(
                    "|".join((metric, group, statistic, phase_statistic))
                )
                families.append(family)


def build_feature_bank(
    geometry: np.ndarray, recurrence: np.ndarray
) -> FeatureBank:
    n = len(geometry)
    geometry = geometry.reshape(n, PHASE_BINS, 320)
    recurrence = recurrence.reshape(n, 45, 24)
    columns: list[np.ndarray] = []
    names: list[str] = []
    families: list[str] = []
    offset = 0
    for metric in SEQUENCE_METRICS:
        width = 8 * len(SEQUENCE_STATS)
        block = geometry[:, :, offset : offset + width].reshape(
            n, PHASE_BINS, 8, len(SEQUENCE_STATS)
        )
        offset += width
        add_geometry_block(
            columns, names, families, metric, block, SEQUENCE_STATS
        )
    for metric in TRANSITION_METRICS:
        width = 8 * len(TRANSITION_STATS)
        block = geometry[:, :, offset : offset + width].reshape(
            n, PHASE_BINS, 8, len(TRANSITION_STATS)
        )
        offset += width
        if "hard_switch" in metric:
            block = 1.0 - block
        add_geometry_block(
            columns, names, families, metric, block, TRANSITION_STATS
        )
    if offset != geometry.shape[-1]:
        raise ValueError("geometry descriptor width changed")

    phase_pairs = list(combinations(range(PHASE_BINS), 2))
    for kind_axis, kind in enumerate(
        ("soft_route", "soft_denoise_spread", "hard_route")
    ):
        block = recurrence[:, :, kind_axis * 8 : (kind_axis + 1) * 8]
        for group, layer_axes in LAYER_GROUPS.items():
            distance = block[:, :, layer_axes].mean(axis=2)
            matrix = np.full(
                (n, PHASE_BINS, PHASE_BINS), np.nan, dtype=np.float32
            )
            for pair_axis, (left, right) in enumerate(phase_pairs):
                matrix[:, left, right] = distance[:, pair_axis]
                matrix[:, right, left] = distance[:, pair_axis]
            adjacent = np.stack(
                [matrix[:, query - 1, query] for query in range(1, PHASE_BINS)],
                axis=1,
            )
            return_advantage = np.stack(
                [
                    matrix[:, query - 1, query]
                    - np.nanmin(matrix[:, query, : query - 1], axis=1)
                    for query in range(2, PHASE_BINS)
                ],
                axis=1,
            )
            descriptors = {
                "adjacent_mean": adjacent.mean(axis=1),
                "adjacent_late": adjacent[:, -3:].mean(axis=1),
                "adjacent_delta": adjacent[:, -3:].mean(axis=1)
                - adjacent[:, :3].mean(axis=1),
                "return_advantage_mean": return_advantage.mean(axis=1),
                "return_advantage_max": return_advantage.max(axis=1),
                "return_fraction": (return_advantage > 0.0).mean(axis=1),
            }
            for descriptor, values in descriptors.items():
                columns.append(values)
                names.append("|".join((kind, group, descriptor)))
                families.append(
                    "query_recurrence"
                    if descriptor.startswith("return")
                    else "query_persistence"
                )
    values = np.column_stack(columns).astype(np.float32)
    if values.shape[1] != 516 or np.any(~np.isfinite(values)):
        raise ValueError(f"unexpected feature bank shape/values: {values.shape}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate interpretable feature name")
    return FeatureBank(values=values, names=names, families=families)


def interpolate_middle_phase(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("per-denoise phase interpolation needs [query, value]")
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.5, 0.9, PHASE_BINS, dtype=np.float64)
    upper = np.searchsorted(source, target, side="left")
    upper = np.clip(upper, 1, len(source) - 1)
    lower = upper - 1
    alpha = ((target - source[lower]) / (source[upper] - source[lower])).astype(
        np.float32
    )
    return values[lower] * (1.0 - alpha[:, None]) + values[upper] * alpha[
        :, None
    ]


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
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


def per_denoise_episode_features(
    raw_probability: np.ndarray,
) -> tuple[np.ndarray, list[str], list[str], float]:
    probability = np.maximum(np.asarray(raw_probability, dtype=np.float32), 0.0)
    if probability.ndim != 5 or probability.shape[1:] != (8, 10, 11, 32):
        raise ValueError(f"unexpected router probability shape: {probability.shape}")
    mass = probability.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(probability)):
        raise ValueError("invalid router probability")
    probability /= mass
    state = probability[..., 0, :]
    action = probability[..., 1:, :]
    action_mean = action.mean(axis=3)
    entropy = -np.sum(
        action * np.log(np.maximum(action, 1e-12)), axis=-1
    ).mean(axis=3)
    top1 = action.max(axis=-1).mean(axis=3)
    top4 = np.partition(action, -4, axis=-1)[..., -4:].sum(axis=-1).mean(axis=3)
    token_dispersion = hellinger(action, action_mean[..., None, :]).mean(axis=3)
    state_action_gap = hellinger(state, action_mean)
    denoise_motion = hellinger(action_mean[:, :, 1:], action_mean[:, :, :-1])
    metrics = {
        "action_entropy_at_d": (entropy, "action_confidence"),
        "action_top1_at_d": (top1, "action_confidence"),
        "action_top4_at_d": (top4, "action_confidence"),
        "action_token_dispersion_at_d": (
            token_dispersion,
            "token_disagreement",
        ),
        "state_action_gap_at_d": (
            state_action_gap,
            "state_action_mismatch",
        ),
    }
    columns: list[float] = []
    names: list[str] = []
    families: list[str] = []
    for metric, (curve, family) in metrics.items():
        for group, layer_axes in LAYER_GROUPS.items():
            anchors = interpolate_middle_phase(curve[:, layer_axes].mean(axis=1))
            for denoise in range(N_DENOISE):
                summaries = phase_summaries(anchors[:, denoise][None, :])
                for statistic, value in summaries.items():
                    columns.append(float(value[0]))
                    names.append(
                        "|".join((metric, group, f"d{denoise}", statistic))
                    )
                    families.append(family)
    for group, layer_axes in LAYER_GROUPS.items():
        anchors = interpolate_middle_phase(
            denoise_motion[:, layer_axes].mean(axis=1)
        )
        for denoise in range(N_DENOISE - 1):
            summaries = phase_summaries(anchors[:, denoise][None, :])
            for statistic, value in summaries.items():
                columns.append(float(value[0]))
                names.append(
                    "|".join(
                        (
                            "action_denoise_motion_at_i",
                            group,
                            f"d{denoise}_to_d{denoise + 1}",
                            statistic,
                        )
                    )
                )
                families.append("denoise_soft_motion")
    state_denoise_span = float(
        np.max(np.abs(state - state[:, :, :1, :]))
    )
    values = np.asarray(columns, dtype=np.float32)
    if values.shape != (708,) or np.any(~np.isfinite(values)):
        raise ValueError(f"unexpected per-denoise features: {values.shape}")
    return values, names, families, state_denoise_span


def extract_failure_per_denoise_bank(
    labels: pd.DataFrame, manifest: dict[str, Any]
) -> tuple[FeatureBank, dict[str, Any]]:
    cache_root = pathlib.Path(manifest["cache_root"])
    failure = labels[labels["is_failure"]].reset_index(drop=True)
    run_data: dict[str, tuple[Any, np.ndarray]] = {}
    for task, task_frame in labels.groupby("task", sort=False):
        ordered = task_frame.sort_values("episode")
        episodes = ordered["episode"].to_numpy(dtype=int)
        if not np.array_equal(episodes, np.arange(len(ordered))):
            raise ValueError(f"episode order changed for {task}")
        lengths = ordered["episode_length"].to_numpy(dtype=int)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]].astype(np.int64)
        route = zarr.open_group(
            str(cache_root / task / "right-16x32/server/routes.zarr"),
            mode="r",
        )
        run_data[task] = (route, offsets)
    rows = []
    names: list[str] | None = None
    families: list[str] | None = None
    state_denoise_span = 0.0
    for position, row in enumerate(failure.itertuples(index=False), start=1):
        route, offsets = run_data[row.task]
        start = int(offsets[row.episode])
        stop = start + int(row.episode_length)
        raw = np.asarray(route["hb_router_probs"][start:stop], dtype=np.float32)
        values, current_names, current_families, current_span = (
            per_denoise_episode_features(raw)
        )
        if names is None:
            names = current_names
            families = current_families
        elif current_names != names or current_families != families:
            raise ValueError("per-denoise feature schema changed between episodes")
        rows.append(values)
        state_denoise_span = max(state_denoise_span, current_span)
        if position % 50 == 0:
            print(
                f"extracted per-denoise failures: {position}/{len(failure)}",
                flush=True,
            )
    assert names is not None and families is not None
    matrix = np.stack(rows).astype(np.float32)
    return (
        FeatureBank(matrix, names, families),
        {
            "failure_episodes": len(failure),
            "features": matrix.shape[1],
            "state_denoise_max_abs_probability_difference": state_denoise_span,
            "denoise_order": "d0 noisiest at flow time 1.0; d9 last recorded forward at 0.1",
        },
    )


def load_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    taxonomy = pd.read_csv(TAXONOMY)
    taxonomy = taxonomy.rename(columns={"failure": "taxonomy_failure"})
    rule_labels = [column for column in taxonomy if column.startswith("label_")]
    keep = [
        "task",
        "episode",
        "init_state_id",
        "taxonomy_failure",
        "primary_behavior",
        *rule_labels,
    ]
    frame = metadata.reset_index(names="cache_row").merge(
        taxonomy[keep],
        on=["task", "episode", "init_state_id"],
        how="left",
        validate="one_to_one",
    )
    if frame["taxonomy_failure"].isna().any() or not np.array_equal(
        frame["is_failure"].to_numpy(), frame["taxonomy_failure"].to_numpy()
    ):
        raise ValueError("behavior taxonomy does not align with routing cache")
    events = pd.read_csv(EVENTS)[
        ["task", "episode", "primary_physical_pattern", "late_motion_class"]
    ]
    frame = frame.merge(
        events,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    )
    failure = frame["is_failure"]
    frame.loc[failure, "residual_none"] = ~frame.loc[
        failure, rule_labels
    ].any(axis=1)
    frame.loc[~failure, "residual_none"] = False
    frame["residual_none"] = frame["residual_none"].astype(bool)
    frame["label_active_return"] = frame["primary_physical_pattern"].eq(
        "return_to_completed_pot2_proxy"
    )
    frame["stratum"] = (
        frame["task"] + "|init=" + frame["init_state_id"].astype(str)
    )
    return frame


def success_reference_z(
    values: np.ndarray,
    labels: pd.DataFrame,
    reference_mask: np.ndarray | None = None,
) -> np.ndarray:
    if reference_mask is None:
        reference_mask = np.ones(len(labels), dtype=bool)
    if reference_mask.shape != (len(labels),):
        raise ValueError("success reference mask has the wrong shape")
    is_failure = labels["is_failure"].to_numpy(dtype=bool)
    usable_success = reference_mask & ~is_failure
    result = np.zeros_like(values, dtype=np.float32)
    for task in labels["task"].unique():
        task_index = np.flatnonzero(labels["task"].eq(task).to_numpy())
        task_success = task_index[usable_success[task_index]]
        if len(task_success) < 2:
            raise ValueError(f"no success reference for {task}")
        task_mean = values[task_success].mean(axis=0)
        task_var = values[task_success].var(axis=0, ddof=1)
        for init_state in labels.iloc[task_index]["init_state_id"].unique():
            cell_index = task_index[
                labels.iloc[task_index]["init_state_id"].to_numpy() == init_state
            ]
            cell_success = cell_index[usable_success[cell_index]]
            weight = len(cell_success) / (len(cell_success) + SUCCESS_SHRINKAGE)
            if len(cell_success):
                cell_mean = values[cell_success].mean(axis=0)
            else:
                cell_mean = task_mean
            if len(cell_success) > 1:
                cell_var = values[cell_success].var(axis=0, ddof=1)
            else:
                cell_var = task_var
            center = weight * cell_mean + (1.0 - weight) * task_mean
            variance = (
                weight * cell_var
                + (1.0 - weight) * task_var
                + weight * (1.0 - weight) * np.square(cell_mean - task_mean)
            )
            scale = np.sqrt(np.maximum(variance, 0.05 * task_var + 1e-12))
            result[cell_index] = (values[cell_index] - center) / scale
    if np.any(~np.isfinite(result)):
        raise ValueError("non-finite success-reference feature")
    return np.clip(result, -20.0, 20.0)


def split_success_reference_fold(
    values: np.ndarray,
    labels: pd.DataFrame,
    held_out: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    held_success = np.flatnonzero(
        labels["stratum"].eq(held_out).to_numpy()
        & ~labels["is_failure"].to_numpy(dtype=bool)
    )
    if len(held_success) < 2:
        raise ValueError(f"need two held-out successes for {held_out}")
    stable = int.from_bytes(
        hashlib.sha256(held_out.encode("utf-8")).digest()[:8], "little"
    )
    rng = np.random.default_rng(seed + stable)
    shuffled = rng.permutation(held_success)
    calibration_count = (len(shuffled) + 1) // 2
    evaluation_rows = np.sort(shuffled[calibration_count:])
    reference_mask = np.ones(len(labels), dtype=bool)
    reference_mask[evaluation_rows] = False
    return (
        success_reference_z(values, labels, reference_mask),
        evaluation_rows,
    )


def mixed_group_subset(
    frame: pd.DataFrame, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome = frame[label].to_numpy(dtype=np.int64)
    group_text = frame["stratum"].to_numpy()
    mixed = [
        group
        for group in np.unique(group_text)
        if np.unique(outcome[group_text == group]).size == 2
    ]
    keep = np.isin(group_text, mixed)
    groups = pd.Categorical(group_text[keep]).codes.astype(np.int64)
    return keep, outcome[keep], groups


def leave_one_group_effects(
    values: np.ndarray, outcome: np.ndarray, groups: np.ndarray, column: int
) -> tuple[float, float, int, int]:
    full = layer_effects.fixed_effect_shift(
        values[:, [column]], outcome, groups
    )[2][0]
    effects = []
    for group in np.unique(groups):
        keep = groups != group
        if np.unique(groups[keep]).size < 2:
            continue
        effects.append(
            float(
                layer_effects.fixed_effect_shift(
                    values[keep][:, [column]], outcome[keep], groups[keep]
                )[2][0]
            )
        )
    if not effects:
        return np.nan, np.nan, 0, 0
    same = sum(np.sign(value) == np.sign(full) for value in effects)
    return min(effects), max(effects), same, len(effects)


def scan_labels(
    bank: FeatureBank,
    failure: pd.DataFrame,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_meta = [
        layer_effects.FeatureMeta(
            key=name,
            family=family,
            token_role="mixed",
            scope="episode_0.5_0.9",
            metric=name,
        )
        for name, family in zip(bank.names, bank.families)
    ]
    effect_rows = []
    signature_rows = []
    support_rows = []
    labels = PRIMARY_LABELS + EXPLORATORY_LABELS
    for label_axis, label in enumerate(labels):
        keep, outcome, groups = mixed_group_subset(failure, label)
        values = bank.values[failure.loc[keep, "cache_row"].to_numpy(dtype=int)]
        within = layer_effects.residualize_by_group(values, groups)
        variable = np.sqrt(np.mean(np.square(within), axis=0)) > 1e-10
        selected_values = values[:, variable]
        selected_meta = [meta for meta, use in zip(feature_meta, variable) if use]
        beta, residual_sd, effect, _ = layer_effects.fixed_effect_shift(
            selected_values, outcome, groups
        )
        raw_p, family_p, global_p = layer_effects.permutation_scan(
            selected_values,
            outcome,
            groups,
            selected_meta,
            permutations,
            seed + label_axis * 100,
        )
        support_rows.append(
            {
                "label": label,
                "mode": LABEL_NAMES[label],
                "primary": label in PRIMARY_LABELS,
                "positive_total": int(failure[label].sum()),
                "episodes_in_mixed_strata": len(outcome),
                "positives_in_mixed_strata": int(outcome.sum()),
                "controls_in_mixed_strata": int((outcome == 0).sum()),
                "mixed_strata": int(np.unique(groups).size),
            }
        )
        index_lookup = np.flatnonzero(variable)
        for axis, (meta, original_index) in enumerate(
            zip(selected_meta, index_lookup)
        ):
            effect_rows.append(
                {
                    "label": label,
                    "mode": LABEL_NAMES[label],
                    "primary": label in PRIMARY_LABELS,
                    "feature": meta.key,
                    "family": meta.family,
                    "feature_index": int(original_index),
                    "effect_sigma": effect[axis],
                    "adjusted_difference": beta[axis],
                    "within_stratum_residual_sd": residual_sd[axis],
                    "target_mean": selected_values[outcome == 1, axis].mean(),
                    "control_mean": selected_values[outcome == 0, axis].mean(),
                    "permutation_p_raw": raw_p[axis],
                    "permutation_p_max_family": family_p[axis],
                    "permutation_p_max_all_features": global_p[axis],
                    "primary_label_fwer_p": min(
                        1.0, global_p[axis] * len(PRIMARY_LABELS)
                    )
                    if label in PRIMARY_LABELS
                    else np.nan,
                }
            )
        current = pd.DataFrame(effect_rows)
        current = current[current["label"] == label]
        for family, family_frame in current.groupby("family"):
            row = family_frame.loc[
                family_frame["effect_sigma"].abs().idxmax()
            ].to_dict()
            original_index = int(row["feature_index"])
            lo, hi, same, folds = leave_one_group_effects(
                values, outcome, groups, original_index
            )
            row.update(
                {
                    "loso_effect_min": lo,
                    "loso_effect_max": hi,
                    "loso_same_direction": same,
                    "loso_folds": folds,
                }
            )
            signature_rows.append(row)
    return (
        pd.DataFrame(effect_rows),
        pd.DataFrame(signature_rows),
        pd.DataFrame(support_rows),
    )


def select_fold_features(
    values: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    families: np.ndarray,
    allowed: frozenset[str],
    maximum: int = 4,
) -> list[tuple[int, float, float]]:
    _, _, effect, _ = layer_effects.fixed_effect_shift(values, outcome, groups)
    candidates = []
    for family in sorted(allowed):
        indices = np.flatnonzero(families == family)
        if not len(indices):
            continue
        selected = int(indices[np.argmax(np.abs(effect[indices]))])
        direction = float(np.sign(effect[selected]))
        candidates.append((selected, direction if direction else 1.0, abs(effect[selected])))
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:maximum]


def group_bootstrap_net(
    truth: np.ndarray,
    trigger: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([by_group[group] for group in sampled])
        if not np.any(truth[index]) or np.all(truth[index]):
            continue
        result.append(
            float(trigger[index][truth[index]].mean())
            - float(trigger[index][~truth[index]].mean())
        )
    if not result:
        return np.nan, np.nan
    return tuple(np.quantile(result, [0.025, 0.975]))


def crossfit_coverage(
    values: np.ndarray,
    bank: FeatureBank,
    labels: pd.DataFrame,
    failure: pd.DataFrame,
    bootstrap: int,
    seed: int,
    normalization: str = "split_success_reference",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if normalization not in ("split_success_reference", "none"):
        raise ValueError(f"unknown normalization: {normalization}")
    family_array = np.asarray(bank.families)
    summaries = []
    predictions = []
    selections = []
    fold_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label_axis, label in enumerate(PRIMARY_LABELS):
        failure_truth = failure[label].to_numpy(dtype=bool)
        failure_groups = failure["stratum"].to_numpy()
        mixed = [
            group
            for group in np.unique(failure_groups)
            if np.unique(failure_truth[failure_groups == group]).size == 2
        ]
        identifiable = np.isin(failure_groups, mixed)
        for block_axis, (block, allowed) in enumerate(BLOCKS.items()):
            failure_score = np.full(len(failure), np.nan, dtype=np.float64)
            failure_trigger = {
                calibration: np.zeros(len(failure), dtype=bool)
                for calibration in CALIBRATIONS
            }
            success_records: dict[str, list[tuple[int, float, bool]]] = {
                calibration: [] for calibration in CALIBRATIONS
            }
            for held_out in mixed:
                if normalization == "split_success_reference":
                    if held_out not in fold_inputs:
                        fold_inputs[held_out] = split_success_reference_fold(
                            values, labels, held_out, seed
                        )
                    normalized, held_success_rows = fold_inputs[held_out]
                else:
                    normalized = values
                    held_success_rows = labels.loc[
                        labels["stratum"].eq(held_out)
                        & ~labels["is_failure"],
                        "cache_row",
                    ].to_numpy(dtype=int)
                train = identifiable & (failure_groups != held_out)
                test = failure_groups == held_out
                train_groups = pd.Categorical(failure_groups[train]).codes
                train_rows = failure.loc[train, "cache_row"].to_numpy(dtype=int)
                chosen = select_fold_features(
                    normalized[train_rows],
                    failure_truth[train].astype(np.int64),
                    train_groups,
                    family_array,
                    allowed,
                    maximum=4,
                )
                feature_index = np.asarray([item[0] for item in chosen], dtype=int)
                direction = np.asarray([item[1] for item in chosen], dtype=float)
                train_control = train & ~failure_truth
                control_rows = failure.loc[
                    train_control, "cache_row"
                ].to_numpy(dtype=int)
                center = normalized[control_rows][:, feature_index].mean(axis=0)
                scale = normalized[control_rows][:, feature_index].std(
                    axis=0, ddof=1
                )
                scale = np.maximum(scale, 1e-6)

                def score_rows(rows: np.ndarray) -> np.ndarray:
                    return (
                        (normalized[rows][:, feature_index] - center)
                        / scale
                        * direction
                    ).mean(axis=1)

                control_score = score_rows(control_rows)
                failure_threshold = float(
                    np.quantile(control_score, CONTROL_QUANTILE, method="higher")
                )
                training_strata = [group for group in mixed if group != held_out]
                training_success_rows = labels.loc[
                    labels["stratum"].isin(training_strata)
                    & ~labels["is_failure"],
                    "cache_row",
                ].to_numpy(dtype=int)
                if not len(training_success_rows):
                    raise ValueError("cross-fit fold has no success controls")
                training_success_score = score_rows(training_success_rows)
                success_threshold = float(
                    np.quantile(
                        training_success_score,
                        CONTROL_QUANTILE,
                        method="higher",
                    )
                )
                thresholds = {
                    "other_failure_q90": failure_threshold,
                    "dual_control_q90": max(
                        failure_threshold, success_threshold
                    ),
                }
                test_failure_axis = np.flatnonzero(test)
                test_rows = failure.loc[test, "cache_row"].to_numpy(dtype=int)
                held_score = score_rows(test_rows)
                failure_score[test_failure_axis] = held_score
                for calibration, threshold in thresholds.items():
                    failure_trigger[calibration][test_failure_axis] = (
                        held_score > threshold
                    )
                if len(held_success_rows):
                    success_score = score_rows(held_success_rows)
                    for calibration, threshold in thresholds.items():
                        success_records[calibration].extend(
                            zip(
                                held_success_rows,
                                success_score,
                                success_score > threshold,
                            )
                        )
                for selected, sign, training_effect in chosen:
                    selections.append(
                        {
                            "label": label,
                            "mode": LABEL_NAMES[label],
                            "block": block,
                            "held_out_stratum": held_out,
                            "feature": bank.names[selected],
                            "family": bank.families[selected],
                            "direction": sign,
                            "training_effect_abs": training_effect,
                        }
                    )
            for calibration_axis, calibration in enumerate(CALIBRATIONS):
                truth = failure_truth[identifiable]
                trigger = failure_trigger[calibration][identifiable]
                group_codes = pd.Categorical(failure_groups[identifiable]).codes
                coverage = float(trigger[truth].mean())
                control_rate = float(trigger[~truth].mean())
                ci_low, ci_high = group_bootstrap_net(
                    truth,
                    trigger,
                    group_codes,
                    bootstrap,
                    seed
                    + label_axis * 100
                    + block_axis * 10
                    + calibration_axis,
                )
                current_success = success_records[calibration]
                success_rate = (
                    float(np.mean([item[2] for item in current_success]))
                    if current_success
                    else np.nan
                )
                worst_control = max(control_rate, success_rate)
                summaries.append(
                    {
                        "label": label,
                        "mode": LABEL_NAMES[label],
                        "block": block,
                        "calibration": calibration,
                        "target_n": int(truth.sum()),
                        "target_coverage": coverage,
                        "other_failure_n": int((~truth).sum()),
                        "other_failure_trigger_rate": control_rate,
                        "net_coverage": coverage - control_rate,
                        "net_coverage_ci_low": ci_low,
                        "net_coverage_ci_high": ci_high,
                        "success_n": len(current_success),
                        "success_trigger_rate": success_rate,
                        "worst_control_trigger_rate": worst_control,
                        "operational_net_coverage": coverage - worst_control,
                        "mixed_strata": len(mixed),
                    }
                )
                for axis in np.flatnonzero(identifiable):
                    row = failure.iloc[axis]
                    predictions.append(
                        {
                            "label": label,
                            "mode": LABEL_NAMES[label],
                            "block": block,
                            "calibration": calibration,
                            "task": row["task"],
                            "episode": int(row["episode"]),
                            "init_state_id": int(row["init_state_id"]),
                            "truth_class": "target"
                            if failure_truth[axis]
                            else "other_failure",
                            "score": failure_score[axis],
                            "trigger": failure_trigger[calibration][axis],
                        }
                    )
                for cache_row, score, trigger_value in current_success:
                    row = labels.iloc[int(cache_row)]
                    predictions.append(
                        {
                            "label": label,
                            "mode": LABEL_NAMES[label],
                            "block": block,
                            "calibration": calibration,
                            "task": row["task"],
                            "episode": int(row["episode"]),
                            "init_state_id": int(row["init_state_id"]),
                            "truth_class": "success",
                            "score": score,
                            "trigger": bool(trigger_value),
                        }
                    )
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(predictions),
        pd.DataFrame(selections),
    )


def union_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for calibration in CALIBRATIONS:
        for block in BLOCKS:
            selected = predictions[
                (predictions["block"] == block)
                & (predictions["calibration"] == calibration)
                & (predictions["truth_class"] == "target")
            ]
            grouped = selected.groupby(["task", "episode"], sort=False)[
                "trigger"
            ].max()
            success = predictions[
                (predictions["block"] == block)
                & (predictions["calibration"] == calibration)
                & (predictions["truth_class"] == "success")
            ]
            success_grouped = success.groupby(
                ["task", "episode"], sort=False
            )["trigger"].max()
            rows.append(
                {
                    "block": block,
                    "calibration": calibration,
                    "identifiable_failure_episodes": len(grouped),
                    "covered_failure_episodes": int(grouped.sum()),
                    "coverage": float(grouped.mean()),
                    "supported_success_episodes": len(success_grouped),
                    "success_union_trigger_rate": float(success_grouped.mean()),
                }
            )
    return pd.DataFrame(rows)


def episode_coverage_status(
    predictions: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = predictions[
        (predictions["calibration"] == "dual_control_q90")
        & (predictions["truth_class"] == "target")
        & predictions["block"].isin(
            ("trap_only", "nontrap_only", "full_invariant")
        )
    ]
    trigger = (
        selected.groupby(["task", "episode", "block"])["trigger"]
        .max()
        .unstack("block", fill_value=False)
        .reset_index()
    )
    for block in ("trap_only", "nontrap_only", "full_invariant"):
        if block not in trigger:
            trigger[block] = False
    failure = labels[labels["is_failure"]][
        [
            "task",
            "episode",
            "init_state_id",
            "primary_behavior",
            "primary_physical_pattern",
        ]
    ]
    episode = trigger.merge(
        failure,
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    )
    episode["coverage_status"] = np.select(
        [
            episode["trap_only"] & episode["full_invariant"],
            episode["trap_only"] & ~episode["full_invariant"],
            ~episode["trap_only"] & episode["full_invariant"],
        ],
        ["both", "trap_only", "full_only"],
        default="uncovered",
    )
    episode["nontrap_coverage_status"] = np.select(
        [
            episode["trap_only"] & episode["nontrap_only"],
            episode["trap_only"] & ~episode["nontrap_only"],
            ~episode["trap_only"] & episode["nontrap_only"],
        ],
        ["both", "trap_only", "nontrap_only"],
        default="uncovered",
    )
    breakdown = (
        episode.groupby(
            ["coverage_status", "primary_physical_pattern"], dropna=False
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["coverage_status", "n"], ascending=[True, False])
    )
    nontrap_breakdown = (
        episode.groupby(
            ["nontrap_coverage_status", "primary_physical_pattern"],
            dropna=False,
        )
        .size()
        .rename("n")
        .reset_index()
        .sort_values(
            ["nontrap_coverage_status", "n"], ascending=[True, False]
        )
    )
    return episode, breakdown, nontrap_breakdown


def physical_mode_support(labels: pd.DataFrame) -> pd.DataFrame:
    failure = labels[labels["is_failure"]].copy()
    rows = []
    for (task, pattern), frame in failure.groupby(
        ["task", "primary_physical_pattern"]
    ):
        task_frame = failure[failure["task"] == task]
        mixed = 0
        positives_in_mixed = 0
        episodes_in_mixed = 0
        for _, group in task_frame.groupby("init_state_id"):
            outcome = group["primary_physical_pattern"].eq(pattern)
            if outcome.nunique() == 2:
                mixed += 1
                positives_in_mixed += int(outcome.sum())
                episodes_in_mixed += len(group)
        rows.append(
            {
                "task": task,
                "physical_pattern": pattern,
                "n": len(frame),
                "initial_states": frame["init_state_id"].nunique(),
                "mixed_initial_states_vs_other_failure": mixed,
                "positives_in_mixed_states": positives_in_mixed,
                "episodes_in_mixed_states": episodes_in_mixed,
                "cross_init_status": "estimable"
                if mixed >= 3 and positives_in_mixed >= 8
                else "descriptive_only",
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "physical_pattern"])


def physical_mode_descriptive(
    bank: FeatureBank, labels: pd.DataFrame, support: pd.DataFrame
) -> pd.DataFrame:
    failure = labels[labels["is_failure"]].reset_index(drop=True)
    rows = []
    for item in support.itertuples(index=False):
        task_frame = failure[failure["task"] == item.task].reset_index(drop=True)
        outcome = task_frame["primary_physical_pattern"].eq(
            item.physical_pattern
        ).to_numpy(dtype=np.int64)
        group_text = task_frame["stratum"].to_numpy()
        mixed = [
            group
            for group in np.unique(group_text)
            if np.unique(outcome[group_text == group]).size == 2
        ]
        if not mixed:
            continue
        keep = np.isin(group_text, mixed)
        groups = pd.Categorical(group_text[keep]).codes
        values = bank.values[
            task_frame.loc[keep, "cache_row"].to_numpy(dtype=int)
        ]
        _, _, effect, _ = layer_effects.fixed_effect_shift(
            values, outcome[keep], groups
        )
        for family in sorted(ALL_FAMILIES):
            indices = np.flatnonzero(np.asarray(bank.families) == family)
            selected = int(indices[np.argmax(np.abs(effect[indices]))])
            rows.append(
                {
                    "task": item.task,
                    "physical_pattern": item.physical_pattern,
                    "n": item.n,
                    "mixed_initial_states": len(mixed),
                    "cross_init_status": item.cross_init_status,
                    "family": family,
                    "feature": bank.names[selected],
                    "effect_sigma": effect[selected],
                }
            )
    return pd.DataFrame(rows)


def loop_audit(labels: pd.DataFrame) -> dict[str, int]:
    periodicity = pd.read_csv(PERIODICITY)
    late = periodicity[
        (periodicity["window"] == "failure_late") & periodicity["failure"]
    ]
    if len(late) != 307:
        raise ValueError("periodicity failure-late coverage changed")
    return {
        "failures": 307,
        "taxonomy_eef_oscillation": int(
            labels.loc[labels["is_failure"], "label_eef_oscillation"].sum()
        ),
        "late_route_cycle_candidates": int(late["route_cycle_candidate"].sum()),
        "late_route_plus_state_cycle_candidates": int(
            late["route_state_cycle_candidate"].sum()
        ),
        "late_joint_route_state_action_loop_candidates": int(
            late["joint_loop_candidate"].sum()
        ),
    }


def build_denoise_profiles(
    effects: pd.DataFrame, per_denoise_signatures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = per_denoise_signatures[
        per_denoise_signatures["label"].isin(PRIMARY_LABELS)
    ]
    top = primary.loc[
        primary.groupby("label")["effect_sigma"].apply(
            lambda values: values.abs().idxmax()
        )
    ]
    rows = []
    summaries = []
    for selected in top.itertuples(index=False):
        metric, group, _selected_step, statistic = selected.feature.split("|")
        candidates = effects[effects["label"].eq(selected.label)].copy()
        components = candidates["feature"].str.split("|", expand=True)
        candidates = candidates[
            components[0].eq(metric)
            & components[1].eq(group)
            & components[3].eq(statistic)
        ].copy()
        candidates["denoise_step"] = candidates["feature"].str.extract(
            r"\|d(\d+)"
        )[0].astype(int)
        candidates = candidates.sort_values("denoise_step")
        for candidate in candidates.itertuples(index=False):
            rows.append(
                {
                    "mode": selected.mode,
                    "label": selected.label,
                    "metric": metric,
                    "layer_group": group,
                    "phase_statistic": statistic,
                    "denoise_location": candidate.feature.split("|")[2],
                    "denoise_step": int(
                        re.search(r"d(\d+)", candidate.feature).group(1)
                    ),
                    "effect_sigma": candidate.effect_sigma,
                    "permutation_p_max_all_features": (
                        candidate.permutation_p_max_all_features
                    ),
                    "primary_label_fwer_p": candidate.primary_label_fwer_p,
                }
            )
        peak = candidates.loc[candidates["effect_sigma"].abs().idxmax()]
        median_abs = float(candidates["effect_sigma"].abs().median())
        summaries.append(
            {
                "mode": selected.mode,
                "metric": metric,
                "layer_group": group,
                "phase_statistic": statistic,
                "peak_location": peak["feature"].split("|")[2],
                "peak_effect_sigma": peak["effect_sigma"],
                "effect_min": candidates["effect_sigma"].min(),
                "effect_max": candidates["effect_sigma"].max(),
                "median_abs_effect": median_abs,
                "peak_to_median_abs": abs(peak["effect_sigma"])
                / max(median_abs, 1e-12),
                "direction_consistent": bool(
                    np.all(
                        np.sign(candidates["effect_sigma"])
                        == np.sign(peak["effect_sigma"])
                    )
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summaries).sort_values("mode")


def markdown(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(value) else f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def render_report(
    support: pd.DataFrame,
    signatures: pd.DataFrame,
    per_denoise_signatures: pd.DataFrame,
    denoise_profile_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    union: pd.DataFrame,
    coverage_breakdown: pd.DataFrame,
    nontrap_breakdown: pd.DataFrame,
    no_reference_coverage: pd.DataFrame,
    no_reference_union: pd.DataFrame,
    mode_support: pd.DataFrame,
    mode_descriptive: pd.DataFrame,
    loops: dict[str, int],
    coverage_feature_count: int,
    formal_feature_count: int,
    denoise_audit: dict[str, Any],
) -> str:
    primary_signature = signatures[signatures["label"].isin(PRIMARY_LABELS)].copy()
    primary_signature = primary_signature.loc[
        primary_signature.groupby("label")["effect_sigma"].apply(
            lambda values: values.abs().idxmax()
        )
    ].copy()
    primary_signature["mode"] = primary_signature["label"].map(LABEL_NAMES)
    primary_signature = primary_signature.sort_values("mode")
    primary_denoise = per_denoise_signatures[
        per_denoise_signatures["label"].isin(PRIMARY_LABELS)
    ].copy()
    primary_denoise = primary_denoise.loc[
        primary_denoise.groupby("label")["effect_sigma"].apply(
            lambda values: values.abs().idxmax()
        )
    ].sort_values("mode")
    coverage_display = coverage.copy()
    coverage_display["coverage_minus_other_failure"] = coverage_display[
        "net_coverage"
    ]
    full = coverage_display[
        (coverage_display["block"] == "full_invariant")
        & (coverage_display["calibration"] == "dual_control_q90")
    ]
    no_reference_full = no_reference_coverage[
        (no_reference_coverage["block"] == "full_invariant")
        & (no_reference_coverage["calibration"] == "dual_control_q90")
    ]
    no_reference_union_display = no_reference_union[
        no_reference_union["calibration"] == "dual_control_q90"
    ]
    estimable_modes = mode_descriptive[
        mode_descriptive["cross_init_status"] == "estimable"
    ]
    physical_top = estimable_modes.loc[
        estimable_modes.groupby(["task", "physical_pattern"])[
            "effect_sigma"
        ].apply(lambda values: values.abs().idxmax())
    ].copy()
    physical_top["task"] = physical_top["task"].str.split("/").str[-1]
    support_display = support.copy()
    dual_union = union[union["calibration"] == "dual_control_q90"].set_index(
        "block"
    )
    trap_union = dual_union.loc["trap_only"]
    nontrap_union = dual_union.loc["nontrap_only"]
    full_union = dual_union.loc["full_invariant"]
    status_totals = coverage_breakdown.groupby("coverage_status")["n"].sum()
    added_by_full = coverage_breakdown[
        coverage_breakdown["coverage_status"] == "full_only"
    ].copy()
    added_by_full["view"] = "added_by_full"
    missed_by_full = (
        coverage_breakdown[
            coverage_breakdown["coverage_status"].isin(
                ("trap_only", "uncovered")
            )
        ]
        .groupby("primary_physical_pattern", as_index=False)["n"]
        .sum()
    )
    missed_by_full["view"] = "missed_by_full"
    added_and_missed = pd.concat(
        [added_by_full, missed_by_full], ignore_index=True
    ).sort_values(["view", "n"], ascending=[True, False])
    identifiable = int(full_union["identifiable_failure_episodes"])
    unidentifiable = loops["failures"] - identifiable
    missed_count = int(
        status_totals.get("trap_only", 0) + status_totals.get("uncovered", 0)
    )
    nontrap_status_totals = nontrap_breakdown.groupby(
        "nontrap_coverage_status"
    )["n"].sum()
    added_by_nontrap = nontrap_breakdown[
        nontrap_breakdown["nontrap_coverage_status"] == "nontrap_only"
    ].copy()
    added_by_nontrap["view"] = "added_by_nontrap"
    missed_by_nontrap = (
        nontrap_breakdown[
            nontrap_breakdown["nontrap_coverage_status"].isin(
                ("trap_only", "uncovered")
            )
        ]
        .groupby("primary_physical_pattern", as_index=False)["n"]
        .sum()
    )
    missed_by_nontrap["view"] = "missed_by_nontrap"
    nontrap_added_and_missed = pd.concat(
        [added_by_nontrap, missed_by_nontrap], ignore_index=True
    ).sort_values(["view", "n"], ascending=[True, False])
    lines = [
        "# Residual-failure MoE atlas",
        "",
        "## Direct result",
        "",
        f"The repeated response is overwhelmingly persistence rather than a literal loop: {loops['late_route_cycle_candidates']}/{loops['failures']} failures pass the late route-cycle rule, {loops['late_route_plus_state_cycle_candidates']} also pass the physical-state rule, and {loops['late_joint_route_state_action_loop_candidates']} passes the full route+state+action rule. The independent physical oscillation label is {loops['taxonomy_eef_oscillation']}/{loops['failures']}.",
        "",
        "The fixed-point family is strongest for stagnation, but it is not the whole signal. Regrasp/drop, goal regression, active return, and the previously unruled residual cohort peak in different confidence, denoising, token-disagreement, or state/action-mismatch families. The cross-fitted coverage table below quantifies what those extra families add under both failure-specific and dual failure/success control thresholds, never an episode-timeout threshold.",
        "",
        "## Protocol",
        "",
        f"Features use only relative phase 0.5 through 0.9. The {formal_feature_count}-dimensional formal bank is expert-permutation invariant. It contains the {coverage_feature_count} frozen front/back summaries of state/action confidence, action-token dispersion, state/action gap, within-query denoising motion and expert retention, adjacent-query persistence and nonlocal recurrence, plus {denoise_audit['features']} raw-route features that resolve action routing at d0 through d9 and each adjacent denoise transition. Episode length, remaining time, the final 10% of the rollout, physical state, action values, and labels are absent from the feature matrix.",
        "",
        f"All formal effects compare failures with other failures inside task x initial-state strata. Therefore every comparison is also within one task-specific failure horizon. Effects are target-minus-control differences in within-stratum residual standard deviations. Primary-label p-values use max-T over all {formal_feature_count} features and then correct across the five prespecified modes.",
        "",
        f"State-token routing is identical over all ten denoise forwards (maximum raw probability difference {denoise_audit['state_denoise_max_abs_probability_difference']:.1g}), so a state d_i scan would only duplicate the same feature ten times. The per-step extension therefore resolves action-token routing only. {denoise_audit['denoise_order']}.",
        "",
        "The primary coverage experiment permits nominal calibration for a known initial state, but avoids reusing its evaluation controls: a deterministic half of held-out-state successes sets its shrunken success reference, while the other half alone estimates held-out success triggering. A second zero-reference sensitivity run uses raw routing features and all held-out successes, so it has no held-out-state calibration at all.",
        "",
        "## Loop control",
        "",
        *markdown(
            pd.DataFrame([loops]),
            [
                "failures",
                "taxonomy_eef_oscillation",
                "late_route_cycle_candidates",
                "late_route_plus_state_cycle_candidates",
                "late_joint_route_state_action_loop_candidates",
            ],
        ),
        "",
        "## Mode support",
        "",
        *markdown(
            support_display,
            [
                "mode",
                "primary",
                "positive_total",
                "positives_in_mixed_strata",
                "controls_in_mixed_strata",
                "mixed_strata",
            ],
        ),
        "",
        "## Strongest invariant signature per primary mode",
        "",
        *markdown(
            primary_signature,
            [
                "mode",
                "family",
                "feature",
                "effect_sigma",
                "permutation_p_max_all_features",
                "primary_label_fwer_p",
                "loso_effect_min",
                "loso_effect_max",
            ],
        ),
        "",
        "## Strongest denoise-step localization per primary mode",
        "",
        "These rows are selected only from features that preserve an individual action-routing denoise step or transition. Their p-values still use the joint formal max-T scan, not a smaller post-hoc search.",
        "",
        *markdown(
            primary_denoise,
            [
                "mode",
                "family",
                "feature",
                "effect_sigma",
                "permutation_p_max_all_features",
                "primary_label_fwer_p",
            ],
        ),
        "",
        "A peak alone can be misleading, so the next table decomposes the selected metric across every denoise step. `peak_to_median_abs` near one means a trajectory-wide offset; a large value means genuine step localization.",
        "",
        *markdown(
            denoise_profile_summary,
            [
                "mode",
                "metric",
                "layer_group",
                "phase_statistic",
                "peak_location",
                "peak_effect_sigma",
                "effect_min",
                "effect_max",
                "peak_to_median_abs",
                "direction_consistent",
            ],
        ),
        "",
        "## Cross-fitted coverage",
        "",
        f"For each held-out task x initial-state stratum, features and directions are selected only on the other strata. This detector table deliberately stays on the frozen {coverage_feature_count}-feature all-outcome cache; the raw per-step extension is used for localization, not silently added without matching success controls. `other_failure_q90` uses the training 90th percentile of other-failure scores. The stricter `dual_control_q90` uses the larger of the other-failure and success 90th percentiles. `net_coverage` is target coverage minus held-out other-failure trigger rate. Success rates below use only the disjoint held-out evaluation half.",
        "",
        *markdown(
            coverage_display,
            [
                "mode",
                "block",
                "calibration",
                "target_n",
                "target_coverage",
                "other_failure_trigger_rate",
                "net_coverage",
                "net_coverage_ci_low",
                "net_coverage_ci_high",
                "success_trigger_rate",
                "operational_net_coverage",
            ],
        ),
        "",
        "Full invariant block with dual controls only:",
        "",
        *markdown(
            full,
            [
                "mode",
                "target_coverage",
                "other_failure_trigger_rate",
                "coverage_minus_other_failure",
                "success_trigger_rate",
                "operational_net_coverage",
            ],
        ),
        "",
        "Union coverage counts an identifiable failed episode once when any detector for one of its true primary labels fires.",
        "",
        *markdown(
            union,
            [
                "block",
                "calibration",
                "identifiable_failure_episodes",
                "covered_failure_episodes",
                "coverage",
                "supported_success_episodes",
                "success_union_trigger_rate",
            ],
        ),
        "",
        f"Under dual controls, the trap-only union covers {int(trap_union['covered_failure_episodes'])}/{int(trap_union['identifiable_failure_episodes'])} identifiable failures ({trap_union['coverage']:.1%}) with a {trap_union['success_union_trigger_rate']:.1%} union trigger rate on supported successes. The full invariant union covers {int(full_union['covered_failure_episodes'])}/{identifiable} ({full_union['coverage']:.1%}) at {full_union['success_union_trigger_rate']:.1%}: a net {int(full_union['covered_failure_episodes'] - trap_union['covered_failure_episodes'])} additional failures while aggregate success triggering remains essentially unchanged.",
        "",
        f"The requested trap-removal ablation is `nontrap_only`: both query-persistence and recurrence families are excluded. It still covers {int(nontrap_union['covered_failure_episodes'])}/{identifiable} failures ({nontrap_union['coverage']:.1%}) with {nontrap_union['success_union_trigger_rate']:.1%} success triggering.",
        "",
        "Each block retrains a four-feature selector. Therefore `full_invariant` is not the set-theoretic union of the fitted nontrap and trap detectors: allowing trap families can displace a useful nontrap family from that four-feature budget. The nontrap block outperforming the full block is evidence of selector interference under fixed capacity, not evidence that adding a feature can intrinsically destroy information.",
        "",
        "## Trap-removal overlap",
        "",
        f"Against trap-only, the nontrap detector overlap is: both {int(nontrap_status_totals.get('both', 0))}, nontrap-only {int(nontrap_status_totals.get('nontrap_only', 0))}, trap-only {int(nontrap_status_totals.get('trap_only', 0))}, and neither {int(nontrap_status_totals.get('uncovered', 0))}.",
        "",
        *markdown(
            nontrap_added_and_missed,
            ["view", "primary_physical_pattern", "n"],
        ),
        "",
        "## What the extra families add, and what remains",
        "",
        f"The episode-level overlap is: both blocks {int(status_totals.get('both', 0))}, full-only {int(status_totals.get('full_only', 0))}, trap-only {int(status_totals.get('trap_only', 0))}, and neither {int(status_totals.get('uncovered', 0))}. Thus the full block leaves {missed_count}/{identifiable} identifiable failures untriggered. Another {unidentifiable}/{loops['failures']} failures are not in this coverage denominator because their primary mode lacks within-initial-state positive/control support.",
        "",
        *markdown(
            added_and_missed,
            ["view", "primary_physical_pattern", "n"],
        ),
        "",
        "## Zero-reference sensitivity",
        "",
        "This run removes even nominal success calibration from the held-out initial state. It is deliberately harsher: raw routing levels must transfer to an unseen state without a local baseline.",
        "",
        *markdown(
            no_reference_union_display,
            [
                "block",
                "identifiable_failure_episodes",
                "covered_failure_episodes",
                "coverage",
                "supported_success_episodes",
                "success_union_trigger_rate",
            ],
        ),
        "",
        *markdown(
            no_reference_full,
            [
                "mode",
                "target_coverage",
                "other_failure_trigger_rate",
                "success_trigger_rate",
                "operational_net_coverage",
            ],
        ),
        "",
        "## Task-specific physical modes",
        "",
        "The table reports the strongest descriptive invariant feature only for physical patterns supported by at least three mixed initial states and eight positives in those states.",
        "",
        *markdown(
            physical_top,
            [
                "task",
                "physical_pattern",
                "n",
                "mixed_initial_states",
                "cross_init_status",
                "family",
                "feature",
                "effect_sigma",
            ],
        ),
        "",
        "The full support audit is in `physical_mode_support.csv`. In particular, ramekin interference and stove never-transported are mostly initial-state-separated from their within-task failure controls; they remain strong outcome-vs-success observations in the earlier anchor analysis, but this dataset cannot establish a cross-initial-state mode-specific signature for them.",
        "",
        "## Interpretation limits",
        "",
        "- This is an observational state-reading experiment, not a causal intervention and not necessarily an early-warning experiment.",
        "- Physical labels are kinematic proxies. Multi-label overlap is retained rather than forced into one mechanism.",
        "- The primary cross-fit holds out all failures in an initial-state stratum but uses half of that stratum's successes for nominal calibration; the zero-reference run is the stricter unseen-state sensitivity. Both remain conditional on these tasks and checkpoints.",
        "- A missing routing signature does not prove that the VLA lacks the information; expert hidden values, attention, visual tokens, and sampled action geometry are outside this routing-only bank.",
        "",
    ]
    return "\n".join(lines)


def plot_results(
    signatures: pd.DataFrame,
    coverage: pd.DataFrame,
    denoise_profiles: pd.DataFrame,
    out_dir: pathlib.Path,
) -> None:
    modes = [LABEL_NAMES[label] for label in PRIMARY_LABELS]
    families = sorted(ALL_FAMILIES)
    matrix = np.full((len(modes), len(families)), np.nan)
    for mode_axis, mode in enumerate(modes):
        for family_axis, family in enumerate(families):
            selected = signatures[
                (signatures["mode"] == mode) & (signatures["family"] == family)
            ]
            if len(selected):
                matrix[mode_axis, family_axis] = selected.iloc[0]["effect_sigma"]
    fig, axis = plt.subplots(figsize=(11, 4.8))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-3.5, vmax=3.5, aspect="auto")
    axis.set_xticks(range(len(families)), labels=families, rotation=28, ha="right")
    axis.set_yticks(range(len(modes)), labels=modes)
    axis.set_title("Strongest within-family target-minus-other-failure effect")
    for row in range(len(modes)):
        for column in range(len(families)):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=axis, label="within-stratum effect sigma")
    fig.tight_layout()
    fig.savefig(out_dir / "mode_family_effects.png", dpi=160)
    plt.close(fig)

    coverage = coverage[coverage["calibration"] == "dual_control_q90"]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    width = 0.18
    x = np.arange(len(modes))
    colors = ("#4c78a8", "#59a14f", "#f28e2b", "#e15759")
    for block_axis, (block, color) in enumerate(zip(BLOCKS, colors)):
        selected = coverage.set_index(["mode", "block"])
        values = [selected.loc[(mode, block), "net_coverage"] for mode in modes]
        axis.bar(
            x + (block_axis - (len(BLOCKS) - 1) / 2) * width,
            values,
            width=width,
            label=block,
            color=color,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels=modes, rotation=20, ha="right")
    axis.set_ylabel("coverage - other-failure trigger rate")
    axis.set_title("Held-out initial-state mode coverage")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "crossfit_net_coverage.png", dpi=160)
    plt.close(fig)

    modes = [LABEL_NAMES[label] for label in PRIMARY_LABELS]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharex=False)
    for axis, mode in zip(axes.flat, modes):
        selected = denoise_profiles[denoise_profiles["mode"] == mode]
        axis.plot(
            selected["denoise_step"],
            selected["effect_sigma"],
            marker="o",
            color="#4c78a8",
            linewidth=1.8,
        )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(mode)
        axis.set_xticks(selected["denoise_step"])
        axis.set_xticklabels(selected["denoise_location"], rotation=40, ha="right")
        axis.set_ylabel("effect sigma")
    axes.flat[-1].axis("off")
    fig.suptitle("Denoise profile of each mode's strongest step-resolved signal")
    fig.tight_layout()
    fig.savefig(out_dir / "denoise_effect_profiles.png", dpi=160)
    plt.close(fig)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_self_test() -> None:
    rng = np.random.default_rng(7)
    geometry = rng.uniform(size=(12, PHASE_BINS * 320)).astype(np.float32)
    recurrence = rng.uniform(size=(12, 45 * 24)).astype(np.float32)
    bank = build_feature_bank(geometry, recurrence)
    assert bank.values.shape == (12, 516)
    assert len(set(bank.names)) == 516
    assert set(bank.families) == ALL_FAMILIES
    curve = np.tile(np.arange(PHASE_BINS, dtype=np.float32), (2, 1))
    summaries = phase_summaries(curve)
    assert np.allclose(summaries["late_minus_mid"], 7.0)
    raw = rng.uniform(size=(12, 8, 10, 11, 32)).astype(np.float32)
    raw[:, :, :, 0] = raw[:, :, :1, 0]
    values, names, families, state_span = per_denoise_episode_features(raw)
    assert values.shape == (708,)
    assert len(names) == len(set(names)) == len(families) == 708
    assert state_span == 0.0
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("loading frozen route feature cache", flush=True)
    metadata, geometry, recurrence, manifest = load_cached_inputs(
        args.feature_cache
    )
    bank = build_feature_bank(geometry, recurrence)
    labels = load_labels(metadata)
    if not np.array_equal(labels["cache_row"].to_numpy(), np.arange(len(labels))):
        raise ValueError("cache row order changed during label merge")
    failure = labels[labels["is_failure"]].reset_index(drop=True)
    print("extracting failure action routes by denoise step", flush=True)
    per_denoise_bank, denoise_audit = extract_failure_per_denoise_bank(
        labels, manifest
    )
    formal_failure = failure.copy()
    formal_failure["cache_row"] = np.arange(len(formal_failure))
    failure_bank = FeatureBank(
        values=np.column_stack(
            [
                bank.values[failure["cache_row"].to_numpy(dtype=int)],
                per_denoise_bank.values,
            ]
        ),
        names=bank.names + per_denoise_bank.names,
        families=bank.families + per_denoise_bank.families,
    )

    print("formal within-failure max-T scans", flush=True)
    effects, signatures, support = scan_labels(
        failure_bank, formal_failure, args.permutations, args.seed
    )
    per_denoise_effects = effects[
        effects["feature"].str.contains("_at_d|_at_i", regex=True)
    ]
    per_denoise_signatures = per_denoise_effects.loc[
        per_denoise_effects.groupby(["label", "family"])[
            "effect_sigma"
        ].apply(lambda values: values.abs().idxmax())
    ].copy()
    denoise_profiles, denoise_profile_summary = build_denoise_profiles(
        effects, per_denoise_signatures
    )
    print("split-reference normalization and cross-fitting", flush=True)
    coverage, predictions, selections = crossfit_coverage(
        bank.values,
        bank,
        labels,
        failure,
        args.bootstrap,
        args.seed + 10000,
    )
    union = union_coverage(predictions)
    episode_status, coverage_breakdown, nontrap_breakdown = (
        episode_coverage_status(predictions, labels)
    )
    print("zero-reference sensitivity cross-fitting", flush=True)
    no_reference_coverage, no_reference_predictions, no_reference_selections = (
        crossfit_coverage(
            bank.values,
            bank,
            labels,
            failure,
            args.bootstrap,
            args.seed + 20000,
            normalization="none",
        )
    )
    no_reference_union = union_coverage(no_reference_predictions)
    mode_support = physical_mode_support(labels)
    mode_descriptive = physical_mode_descriptive(
        failure_bank, formal_failure, mode_support
    )
    loops = loop_audit(labels)

    effects.to_csv(args.out_dir / "feature_effects.csv", index=False)
    signatures.to_csv(args.out_dir / "mode_signatures.csv", index=False)
    per_denoise_signatures.to_csv(
        args.out_dir / "per_denoise_mode_signatures.csv", index=False
    )
    denoise_profiles.to_csv(
        args.out_dir / "denoise_effect_profiles.csv", index=False
    )
    denoise_profile_summary.to_csv(
        args.out_dir / "denoise_effect_profile_summary.csv", index=False
    )
    support.to_csv(args.out_dir / "mode_support.csv", index=False)
    coverage.to_csv(args.out_dir / "crossfit_coverage.csv", index=False)
    predictions.to_csv(args.out_dir / "crossfit_predictions.csv", index=False)
    selections.to_csv(args.out_dir / "crossfit_selected_features.csv", index=False)
    union.to_csv(args.out_dir / "union_coverage.csv", index=False)
    episode_status.to_csv(
        args.out_dir / "episode_coverage_status.csv", index=False
    )
    coverage_breakdown.to_csv(
        args.out_dir / "coverage_breakdown.csv", index=False
    )
    nontrap_breakdown.to_csv(
        args.out_dir / "nontrap_coverage_breakdown.csv", index=False
    )
    no_reference_coverage.to_csv(
        args.out_dir / "no_reference_crossfit_coverage.csv", index=False
    )
    no_reference_predictions.to_csv(
        args.out_dir / "no_reference_crossfit_predictions.csv", index=False
    )
    no_reference_selections.to_csv(
        args.out_dir / "no_reference_crossfit_selected_features.csv",
        index=False,
    )
    no_reference_union.to_csv(
        args.out_dir / "no_reference_union_coverage.csv", index=False
    )
    mode_support.to_csv(args.out_dir / "physical_mode_support.csv", index=False)
    mode_descriptive.to_csv(
        args.out_dir / "physical_mode_descriptive.csv", index=False
    )
    pd.DataFrame(
        {"feature": failure_bank.names, "family": failure_bank.families}
    ).to_csv(args.out_dir / "feature_dictionary.csv", index=False)
    pd.DataFrame(
        {"feature": bank.names, "family": bank.families}
    ).to_csv(args.out_dir / "coverage_feature_dictionary.csv", index=False)
    np.savez_compressed(
        args.out_dir / "per_denoise_failure_features.npz",
        values=per_denoise_bank.values,
        task=failure["task"].to_numpy(),
        episode=failure["episode"].to_numpy(dtype=np.int32),
        feature=np.asarray(per_denoise_bank.names),
        family=np.asarray(per_denoise_bank.families),
    )
    report = render_report(
        support,
        signatures,
        per_denoise_signatures,
        denoise_profile_summary,
        coverage,
        union,
        coverage_breakdown,
        nontrap_breakdown,
        no_reference_coverage,
        no_reference_union,
        mode_support,
        mode_descriptive,
        loops,
        bank.values.shape[1],
        failure_bank.values.shape[1],
        denoise_audit,
    )
    (args.out_dir / "report.md").write_text(report)
    plot_results(signatures, coverage, denoise_profiles, args.out_dir)
    summary = {
        "schema": "residual-failure-moe-atlas/1",
        "date": "2026-08-28",
        "protocol": {
            "feature_window": "relative phase 0.5 through 0.9",
            "formal_feature_count": failure_bank.values.shape[1],
            "coverage_feature_count": bank.values.shape[1],
            "per_denoise_feature_count": per_denoise_bank.values.shape[1],
            "feature_families": sorted(ALL_FAMILIES),
            "effect": "target minus other failure, fixed task x init effect / within-stratum residual SD",
            "multiplicity": f"max-T over {failure_bank.values.shape[1]} features; five-label FWER for prespecified modes",
            "crossfit": "leave one task x init stratum out; split nominal calibration/evaluation successes; training control q90 thresholds",
            "sensitivity": "raw features with no held-out initial-state reference",
            "forbidden_features": [
                "remaining_time",
                "episode_length",
                "absolute_query_index",
                "final_10_percent",
                "physical_state",
                "actions",
                "outcome",
            ],
        },
        "input_manifest_extractor_sha256": manifest["extractor_sha256"],
        "episodes": len(labels),
        "failures": int(labels["is_failure"].sum()),
        "successes": int((~labels["is_failure"]).sum()),
        "loop_audit": loops,
        "denoise_audit": denoise_audit,
        "mode_support": support.to_dict(orient="records"),
        "per_denoise_signatures": per_denoise_signatures.to_dict(
            orient="records"
        ),
        "denoise_profile_summary": denoise_profile_summary.to_dict(
            orient="records"
        ),
        "coverage": coverage.to_dict(orient="records"),
        "union_coverage": union.to_dict(orient="records"),
        "no_reference_coverage": no_reference_coverage.to_dict(
            orient="records"
        ),
        "no_reference_union_coverage": no_reference_union.to_dict(
            orient="records"
        ),
        "coverage_status": (
            coverage_breakdown.groupby("coverage_status")["n"]
            .sum()
            .to_dict()
        ),
        "nontrap_coverage_status": (
            nontrap_breakdown.groupby("nontrap_coverage_status")["n"]
            .sum()
            .to_dict()
        ),
        "coverage_breakdown": coverage_breakdown.to_dict(orient="records"),
        "nontrap_coverage_breakdown": nontrap_breakdown.to_dict(
            orient="records"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    artifacts = sorted(
        path
        for path in args.out_dir.iterdir()
        if path.is_file() and path.name != "completion.json"
    )
    completion = {
        "status": "complete",
        "script": pathlib.Path(__file__).name,
        "artifacts_sha256": {
            path.name: file_sha256(path) for path in artifacts
        },
    }
    (args.out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
