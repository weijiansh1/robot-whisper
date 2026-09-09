#!/usr/bin/env python3
"""Audit success/failure routing shifts by layer and denoise step.

The primary contrast is active-return failure versus success after removing all
stagnation-core episodes.  Every failure window ends before its annotated
physical event.  Effects are fixed-initial-state standardized mean shifts, not
classifier AUCs.  Within-init label permutations provide cellwise and max-T
familywise p-values over the full feature scan.

State-token routing is verified across the denoise axis.  Action-token routing
is analyzed at every HB layer and denoise step.  No hard expert IDs are read.
"""

from __future__ import annotations

import argparse
import hashlib
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

import analyze_failure_moe_signatures_fixed as fixed


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis/state-route-layer-denoise-effects-20260828"
SEED = 20260828
PERMUTATIONS = 2000
BOOTSTRAPS = 2000
BOOTSTRAP_TOP_PER_FAMILY = 20

LAYER_NUMBERS = (2, 3, 4, 5, 12, 13, 14, 15)
LAYER_GROUPS = {
    "front_2_5": (0, 1, 2, 3),
    "back_12_15": (4, 5, 6, 7),
}
N_DENOISE = 10
N_ACTION_TOKENS = 10
N_EXPERTS = 32
SUMMARY_METRICS = (
    "entropy_mean",
    "entropy_std_query",
    "top1_mean",
    "top1_std_query",
    "p4p5_gap_mean",
    "query_speed_mean",
)
ACTION_METRICS = SUMMARY_METRICS + (
    "token_dispersion_mean",
    "denoise_delta_mean",
)


@dataclass(frozen=True)
class FeatureMeta:
    key: str
    family: str
    token_role: str
    scope: str
    metric: str
    layer: int | None = None
    layer_group: str | None = None
    denoise_step: int | None = None
    relative_query: int | None = None
    expert: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=fixed.CACHE_ROOT)
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


def normalize_probabilities(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    mass = result.sum(axis=-1, keepdims=True)
    if np.any(mass <= 0.0) or np.any(~np.isfinite(result)):
        raise ValueError("invalid router probabilities")
    return result / mass


def normalized_entropy(probability: np.ndarray) -> np.ndarray:
    return (
        -np.sum(probability * np.log(np.maximum(probability, 1e-12)), axis=-1)
        / np.log(probability.shape[-1])
    )


def top1_mass(probability: np.ndarray) -> np.ndarray:
    return probability.max(axis=-1)


def p4p5_gap(probability: np.ndarray) -> np.ndarray:
    top5 = np.sort(
        np.partition(probability, -5, axis=-1)[..., -5:], axis=-1
    )
    return top5[..., 1] - top5[..., 0]


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


def clean_cohort(
    cache_root: pathlib.Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    physical, summaries = fixed.load_client(cache_root)
    successes = [episode for episode, row in physical.items() if row["success"]]
    goals = fixed.goal_reference(physical, successes)
    labels = fixed.derive_labels(physical, goals)
    valid = labels["cut_available"].to_numpy(dtype=bool).copy()
    known = labels["physical_onset_query"].to_numpy() >= 0
    valid &= (~known) | (
        labels["prospective_cut_query"].to_numpy()
        < labels["physical_onset_query"].to_numpy()
    )
    valid &= labels["physical_type"].isin(
        ["success", "stagnation_core", "active_return"]
    ).to_numpy()
    cohort = labels[valid].sort_values("episode").reset_index(drop=True)
    if len(cohort) != 444:
        raise ValueError(f"expected clean-444 cohort, found {len(cohort)}")
    window = (
        cohort["prospective_cut_query"].to_numpy()
        - cohort["pot2_anchor_query"].to_numpy()
        + 1
    )
    if not np.all(window == fixed.PREFIX_AFTER_POT2 + 1):
        raise ValueError("cohort windows do not have the expected fixed width")
    return cohort, summaries


class FeatureCollector:
    def __init__(self) -> None:
        self.meta: dict[str, FeatureMeta] = {}

    def put(
        self,
        row: dict[str, float],
        value: float,
        *,
        family: str,
        token_role: str,
        scope: str,
        metric: str,
        layer: int | None = None,
        layer_group: str | None = None,
        denoise_step: int | None = None,
        relative_query: int | None = None,
        expert: int | None = None,
    ) -> None:
        parts = [family, token_role, scope, metric]
        for name, item in (
            ("L", layer),
            ("G", layer_group),
            ("D", denoise_step),
            ("Q", relative_query),
            ("E", expert),
        ):
            if item is not None:
                parts.append(f"{name}{item}")
        key = "|".join(map(str, parts))
        meta = FeatureMeta(
            key=key,
            family=family,
            token_role=token_role,
            scope=scope,
            metric=metric,
            layer=layer,
            layer_group=layer_group,
            denoise_step=denoise_step,
            relative_query=relative_query,
            expert=expert,
        )
        previous = self.meta.setdefault(key, meta)
        if previous != meta:
            raise RuntimeError(f"feature metadata collision for {key}")
        row[key] = float(value)


def state_layer_metrics(probability: np.ndarray) -> dict[str, np.ndarray]:
    # Input [query, layer, expert], outputs [layer].
    entropy = normalized_entropy(probability)
    top1 = top1_mass(probability)
    gap = p4p5_gap(probability)
    speed = hellinger(probability[1:], probability[:-1])
    return {
        "entropy_mean": entropy.mean(axis=0),
        "entropy_std_query": entropy.std(axis=0),
        "top1_mean": top1.mean(axis=0),
        "top1_std_query": top1.std(axis=0),
        "p4p5_gap_mean": gap.mean(axis=0),
        "query_speed_mean": speed.mean(axis=0),
    }


def action_layer_denoise_metrics(probability: np.ndarray) -> dict[str, np.ndarray]:
    # Input [query, layer, denoise, action_token, expert], outputs [layer, denoise].
    entropy = normalized_entropy(probability)
    top1 = top1_mass(probability)
    gap = p4p5_gap(probability)
    entropy_query = entropy.mean(axis=3)
    top1_query = top1.mean(axis=3)
    query_route = probability.mean(axis=3)
    query_speed = hellinger(query_route[1:], query_route[:-1])
    token_center = probability.mean(axis=3, keepdims=True)
    token_dispersion = hellinger(probability, token_center)
    denoise_delta = np.full(
        (probability.shape[1], probability.shape[2]), np.nan, dtype=np.float32
    )
    denoise_delta[:, 1:] = hellinger(
        probability[:, :, 1:], probability[:, :, :-1]
    ).mean(axis=(0, 3))
    return {
        "entropy_mean": entropy.mean(axis=(0, 3)),
        "entropy_std_query": entropy_query.std(axis=0),
        "top1_mean": top1.mean(axis=(0, 3)),
        "top1_std_query": top1_query.std(axis=0),
        "p4p5_gap_mean": gap.mean(axis=(0, 3)),
        "query_speed_mean": query_speed.mean(axis=0),
        "token_dispersion_mean": token_dispersion.mean(axis=(0, 3)),
        "denoise_delta_mean": denoise_delta,
    }


def extract_features(
    cache_root: pathlib.Path,
    cohort: pd.DataFrame,
    summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[FeatureMeta], dict[str, Any]]:
    run = cache_root / fixed.LONG_TASK / "right-16x32"
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    probability_store = store["hb_router_probs"]
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    control_steps = np.asarray(store["control_step"][:], dtype=np.int64)
    offsets = {
        int(episode): int(index)
        for index, episode in enumerate(episode_ids)
        if index == 0 or episode_ids[index - 1] != episode
    }
    expected_lengths = {
        int(row["episode_index"]): int(row["inference_calls"]) for row in summaries
    }
    for episode, start in offsets.items():
        stop = start + expected_lengths[episode]
        if not np.all(episode_ids[start:stop] == episode):
            raise ValueError(f"episode alignment failed for episode {episode}")
        if not np.array_equal(control_steps[start:stop], np.arange(start, stop)):
            raise ValueError(f"route alignment failed for episode {episode}")

    collector = FeatureCollector()
    rows: list[dict[str, Any]] = []
    state_denoise_max = 0.0
    for axis, item in enumerate(cohort.itertuples(index=False), start=1):
        q0 = int(item.pot2_anchor_query)
        cut = int(item.prospective_cut_query)
        start = offsets[int(item.episode)] + q0
        stop = offsets[int(item.episode)] + cut + 1
        raw = np.asarray(probability_store[start:stop], dtype=np.float32)
        probability = normalize_probabilities(raw)
        state_all = probability[:, :, :, 0, :]
        state_denoise_max = max(
            state_denoise_max,
            float(np.max(np.abs(state_all - state_all[:, :, :1, :]))),
        )
        state = state_all[:, :, 0, :]
        action = probability[:, :, :, 1:, :]
        row: dict[str, Any] = {
            "episode": int(item.episode),
            "init_state_id": int(item.init_state_id),
            "physical_type": str(item.physical_type),
            "q0": q0,
            "cut": cut,
        }

        state_metrics = state_layer_metrics(state)
        for layer_axis, layer in enumerate(LAYER_NUMBERS):
            for metric, values in state_metrics.items():
                collector.put(
                    row,
                    values[layer_axis],
                    family="state_layer_summary",
                    token_role="state",
                    scope="layer",
                    metric=metric,
                    layer=layer,
                )
            expert_mean = state[:, layer_axis].mean(axis=0)
            for expert, value in enumerate(expert_mean):
                collector.put(
                    row,
                    value,
                    family="state_expert_probability",
                    token_role="state",
                    scope="layer_expert",
                    metric="mean_probability",
                    layer=layer,
                    expert=expert,
                )

        state_entropy = normalized_entropy(state)
        state_top1 = top1_mass(state)
        state_gap = p4p5_gap(state)
        for group_name, layer_axes in LAYER_GROUPS.items():
            for metric, values in state_metrics.items():
                collector.put(
                    row,
                    np.mean(values[list(layer_axes)]),
                    family="state_group_summary",
                    token_role="state",
                    scope="layer_group",
                    metric=metric,
                    layer_group=group_name,
                )
            for relative_query in range(len(state)):
                for metric, values in (
                    ("entropy", state_entropy),
                    ("top1_mass", state_top1),
                    ("p4p5_gap", state_gap),
                ):
                    collector.put(
                        row,
                        np.mean(values[relative_query, list(layer_axes)]),
                        family="state_group_relative_query",
                        token_role="state",
                        scope="layer_group_query",
                        metric=metric,
                        layer_group=group_name,
                        relative_query=relative_query,
                    )

        action_metrics = action_layer_denoise_metrics(action)
        for layer_axis, layer in enumerate(LAYER_NUMBERS):
            for denoise in range(N_DENOISE):
                for metric, values in action_metrics.items():
                    value = values[layer_axis, denoise]
                    if not np.isfinite(value):
                        continue
                    collector.put(
                        row,
                        value,
                        family="action_layer_denoise",
                        token_role="action",
                        scope="layer_denoise",
                        metric=metric,
                        layer=layer,
                        denoise_step=denoise,
                    )
                expert_mean = action[:, layer_axis, denoise].mean(axis=(0, 1))
                for expert, value in enumerate(expert_mean):
                    collector.put(
                        row,
                        value,
                        family="action_expert_probability",
                        token_role="action",
                        scope="layer_denoise_expert",
                        metric="mean_probability",
                        layer=layer,
                        denoise_step=denoise,
                        expert=expert,
                    )
        for group_name, layer_axes in LAYER_GROUPS.items():
            for denoise in range(N_DENOISE):
                for metric, values in action_metrics.items():
                    selected = values[list(layer_axes), denoise]
                    selected = selected[np.isfinite(selected)]
                    if not len(selected):
                        continue
                    collector.put(
                        row,
                        np.mean(selected),
                        family="action_group_denoise",
                        token_role="action",
                        scope="layer_group_denoise",
                        metric=metric,
                        layer_group=group_name,
                        denoise_step=denoise,
                    )
        rows.append(row)
        if axis % 64 == 0 or axis == len(cohort):
            print(f"routing windows {axis}/{len(cohort)}", flush=True)

    frame = pd.DataFrame(rows)
    feature_meta = [collector.meta[key] for key in collector.meta]
    feature_keys = [meta.key for meta in feature_meta]
    if frame[feature_keys].isna().any().any():
        raise RuntimeError("non-finite extracted routing feature")
    return frame, feature_meta, {
        "state_token_denoise_max_abs_probability_difference": state_denoise_max,
        "hard_expert_ids_read": False,
        "route_store": str(run / "server/routes.zarr"),
        "feature_count": len(feature_meta),
    }


def residualize_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for group in np.unique(groups):
        selected = groups == group
        result[selected] -= result[selected].mean(axis=0, keepdims=True)
    return result


def fixed_effect_shift(
    values: np.ndarray, labels: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_residual = residualize_by_group(values, groups)
    y_residual = residualize_by_group(labels[:, None], groups)[:, 0]
    denominator = float(y_residual @ y_residual)
    if denominator <= 0.0:
        raise ValueError("outcome is not identified within groups")
    beta = (y_residual @ x_residual) / denominator
    residual = x_residual - y_residual[:, None] * beta[None, :]
    dof = max(1, len(labels) - len(np.unique(groups)) - 1)
    residual_sd = np.sqrt(np.sum(residual * residual, axis=0) / dof)
    effect = beta / np.maximum(residual_sd, 1e-15)
    return beta, residual_sd, effect, x_residual


def permutation_scan(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    metadata: list[FeatureMeta],
    permutations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, _, observed, x_residual = fixed_effect_shift(values, labels, groups)
    y_group_mean = np.zeros(len(labels), dtype=np.float32)
    group_indices = []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        group_indices.append(selected)
        y_group_mean[selected] = float(labels[selected].mean())
    rng = np.random.default_rng(seed)
    permuted = np.empty((permutations, len(labels)), dtype=np.float32)
    for draw in range(permutations):
        row = labels.astype(np.float32).copy()
        for selected in group_indices:
            row[selected] = rng.permutation(row[selected])
        permuted[draw] = row - y_group_mean
    denominator = np.sum(np.square(permuted), axis=1, keepdims=True)
    dot = permuted @ x_residual.astype(np.float32)
    beta = dot / np.maximum(denominator, 1e-12)
    x_sum_square = np.sum(np.square(x_residual), axis=0, keepdims=True)
    sse = x_sum_square - np.square(dot) / np.maximum(denominator, 1e-12)
    dof = max(1, len(labels) - len(np.unique(groups)) - 1)
    residual_sd = np.sqrt(np.maximum(sse / dof, 1e-30))
    permuted_effect = beta / residual_sd
    absolute = np.abs(permuted_effect)
    observed_absolute = np.abs(observed)
    raw_p = (1.0 + np.sum(absolute >= observed_absolute[None, :], axis=0)) / (
        permutations + 1.0
    )
    global_max = absolute.max(axis=1)
    global_p = (1.0 + np.sum(global_max[:, None] >= observed_absolute, axis=0)) / (
        permutations + 1.0
    )
    family_p = np.ones(len(metadata), dtype=np.float64)
    families: dict[str, list[int]] = {}
    for index, meta in enumerate(metadata):
        families.setdefault(meta.family, []).append(index)
    for indices in families.values():
        family_max = absolute[:, indices].max(axis=1)
        family_p[indices] = (
            1.0
            + np.sum(
                family_max[:, None] >= observed_absolute[indices][None, :], axis=0
            )
        ) / (permutations + 1.0)
    return raw_p, family_p, global_p


def bootstrap_top_effects(
    values: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    selected_columns: np.ndarray,
    draws: int,
    seed: int,
) -> np.ndarray:
    unique = np.unique(groups)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    result = np.empty((draws, len(selected_columns)), dtype=np.float64)
    selected_values = values[:, selected_columns]
    for draw in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        indices = []
        boot_groups = []
        for new_group, old_group in enumerate(picked):
            index = by_group[old_group]
            indices.append(index)
            boot_groups.append(np.full(len(index), new_group, dtype=np.int64))
        index = np.concatenate(indices)
        boot_group = np.concatenate(boot_groups)
        _, _, effect, _ = fixed_effect_shift(
            selected_values[index], labels[index], boot_group
        )
        result[draw] = effect
    return result


def contrast_effects(
    frame: pd.DataFrame,
    metadata: list[FeatureMeta],
    positive_types: tuple[str, ...],
    negative_types: tuple[str, ...],
    contrast: str,
    permutations: int,
    bootstraps: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed = set(positive_types) | set(negative_types)
    selected = frame["physical_type"].isin(allowed).to_numpy()
    subset = frame[selected].reset_index(drop=True)
    labels = subset["physical_type"].isin(positive_types).to_numpy(dtype=np.int64)
    groups = subset["init_state_id"].to_numpy(dtype=np.int64)
    mixed = [
        group
        for group in np.unique(groups)
        if np.unique(labels[groups == group]).size == 2
    ]
    identified = np.isin(groups, mixed)
    subset = subset[identified].reset_index(drop=True)
    labels = labels[identified]
    groups = groups[identified]
    keys = [meta.key for meta in metadata]
    values = subset[keys].to_numpy(dtype=np.float64)
    beta, residual_sd, effect, _ = fixed_effect_shift(values, labels, groups)
    raw_p, family_p, global_p = permutation_scan(
        values, labels, groups, metadata, permutations, seed
    )
    positive_mean = values[labels == 1].mean(axis=0)
    negative_mean = values[labels == 0].mean(axis=0)
    rows = []
    for index, meta in enumerate(metadata):
        rows.append(
            {
                "contrast": contrast,
                **meta.__dict__,
                "positive_mean": positive_mean[index],
                "success_mean": negative_mean[index],
                "adjusted_difference": beta[index],
                "within_init_residual_sd": residual_sd[index],
                "effect_sigma": effect[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_family": family_p[index],
                "permutation_p_max_global": global_p[index],
                "effect_ci95_low": np.nan,
                "effect_ci95_high": np.nan,
            }
        )
    effects = pd.DataFrame(rows)
    top_parts = []
    for family in sorted({meta.family for meta in metadata}):
        family_indices = np.asarray(
            [index for index, meta in enumerate(metadata) if meta.family == family]
        )
        family_top = min(BOOTSTRAP_TOP_PER_FAMILY, len(family_indices))
        order = np.argsort(-np.abs(effect[family_indices]))[:family_top]
        top_parts.append(family_indices[order])
    top = np.unique(np.concatenate(top_parts))
    bootstrap = bootstrap_top_effects(
        values, labels, groups, top, bootstraps, seed + 1
    )
    effects.loc[top, "effect_ci95_low"] = np.quantile(bootstrap, 0.025, axis=0)
    effects.loc[top, "effect_ci95_high"] = np.quantile(bootstrap, 0.975, axis=0)
    return effects, {
        "contrast": contrast,
        "positive_types": positive_types,
        "negative_types": negative_types,
        "episodes": len(subset),
        "positive": int(labels.sum()),
        "success": int((labels == 0).sum()),
        "mixed_initial_states": len(mixed),
        "permutations": permutations,
        "bootstrap_draws": bootstraps,
        "bootstrap_top_per_family": BOOTSTRAP_TOP_PER_FAMILY,
    }


def format_cell(row: pd.Series) -> str:
    location = []
    if pd.notna(row["layer"]):
        location.append(f"L{int(row['layer'])}")
    if pd.notna(row["layer_group"]):
        location.append(str(row["layer_group"]))
    if pd.notna(row["denoise_step"]):
        location.append(f"d{int(row['denoise_step'])}")
    if pd.notna(row["relative_query"]):
        location.append(f"q+{int(row['relative_query'])}")
    if pd.notna(row["expert"]):
        location.append(f"e{int(row['expert'])}")
    return "/".join(location) or "aggregate"


def strongest_table(frame: pd.DataFrame, family: str, count: int = 10) -> list[str]:
    selected = frame[frame["family"] == family].copy()
    selected = selected.sort_values("effect_sigma", key=np.abs, ascending=False).head(count)
    lines = [
        "| location | metric | effect sigma | init-bootstrap 95% | raw shift | max-T family p | max-T global p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in selected.iterrows():
        lines.append(
            "| %s | %s | %+.3f | [%+.3f, %+.3f] | %+.6g | %.4f | %.4f |"
            % (
                format_cell(row),
                row["metric"],
                row["effect_sigma"],
                row["effect_ci95_low"],
                row["effect_ci95_high"],
                row["adjusted_difference"],
                row["permutation_p_max_family"],
                row["permutation_p_max_global"],
            )
        )
    return lines


def render_report(
    effects: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# State/action soft-routing layer and denoise effect audit",
        "",
        "## Scope",
        "",
        "Effects are fixed-initial-state standardized mean shifts. Positive values mean the failure class is higher than success. No classifier AUC and no hard expert ID is used.",
        "",
        f"State-token maximum probability difference across denoise steps: `{summary['extraction']['state_token_denoise_max_abs_probability_difference']:.8f}`.",
        "Therefore state-token denoise steps are exact duplicates in this cache; state results are reported once per layer. Action-token routing is reported at denoise steps 0-9.",
        "Denoise `d0` is the first, noisiest forward at flow time 1.0; `d9` is the last recorded forward at flow time 0.1 before the Euler update reaches 0.0.",
        "",
    ]
    for contrast in summary["contrasts"]:
        name = contrast["contrast"]
        selected = effects[effects["contrast"] == name]
        lines.extend(
            [
                f"## {name}",
                "",
                "Episodes `%d` = positive `%d` + success `%d`; mixed initial states `%d`."
                % (
                    contrast["episodes"],
                    contrast["positive"],
                    contrast["success"],
                    contrast["mixed_initial_states"],
                ),
                "",
                "### State layers",
                "",
                *strongest_table(selected, "state_layer_summary", 12),
                "",
                "### State front/back and relative-query scan",
                "",
                *strongest_table(selected, "state_group_summary", 8),
                "",
                *strongest_table(selected, "state_group_relative_query", 8),
                "",
                "### Action layer x denoise",
                "",
                *strongest_table(selected, "action_layer_denoise", 15),
                "",
                "### Action front/back x denoise",
                "",
                *strongest_table(selected, "action_group_denoise", 15),
                "",
                "### Soft expert-probability shifts",
                "",
                *strongest_table(selected, "state_expert_probability", 8),
                "",
                *strongest_table(selected, "action_expert_probability", 8),
                "",
            ]
        )
    lines.extend(
        [
            "## Statistical interpretation",
            "",
            "- `effect sigma` is the class shift divided by within-init residual SD.",
            "- Labels are permuted within initial state. `max-T family p` corrects the full search inside one feature family; `max-T global p` corrects all scanned cells in the contrast.",
            "- Bootstrap intervals are written for the 20 largest absolute effects in each feature family and resample initial states; they do not account for choosing this analysis after inspecting earlier results.",
            "- The active-return contrast is exploratory and has only 39 positive episodes across nine mixed initial states.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_action_heatmaps(effects: pd.DataFrame, out_dir: pathlib.Path) -> None:
    contrast = "active_return_vs_success"
    selected = effects[
        (effects["contrast"] == contrast)
        & (effects["family"] == "action_layer_denoise")
    ]
    metrics = (
        "entropy_mean",
        "entropy_std_query",
        "top1_mean",
        "top1_std_query",
        "p4p5_gap_mean",
        "query_speed_mean",
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for axis, metric in zip(axes.ravel(), metrics):
        matrix = np.full((len(LAYER_NUMBERS), N_DENOISE), np.nan)
        rows = selected[selected["metric"] == metric]
        for _, row in rows.iterrows():
            layer_axis = LAYER_NUMBERS.index(int(row["layer"]))
            matrix[layer_axis, int(row["denoise_step"])] = row["effect_sigma"]
        image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
        axis.set_title(metric)
        axis.set_xlabel("denoise step")
        axis.set_ylabel("HB layer")
        axis.set_xticks(range(N_DENOISE))
        axis.set_yticks(range(len(LAYER_NUMBERS)), labels=LAYER_NUMBERS)
    fig.colorbar(image, ax=axes, shrink=0.8, label="effect sigma")
    fig.savefig(out_dir / "active_return_action_layer_denoise_effects.png", dpi=160)
    plt.close(fig)


def run_self_test() -> None:
    probability = normalize_probabilities(
        np.asarray([[0.2, 0.3, 0.5], [2.0, 3.0, 5.0]], dtype=np.float32)
    )
    assert np.allclose(probability.sum(axis=-1), 1.0)
    assert np.allclose(probability[0], probability[1])
    groups = np.repeat(np.arange(4), 8)
    labels = np.tile(np.r_[np.zeros(4), np.ones(4)], 4).astype(np.int64)
    rng = np.random.default_rng(7)
    values = (1.5 * labels + np.repeat(np.arange(4), 8) + rng.normal(0, 0.2, 32))[:, None]
    beta, _, effect, _ = fixed_effect_shift(values, labels, groups)
    assert beta[0] > 1.0 and effect[0] > 3.0
    metadata = [FeatureMeta("x", "test", "state", "unit", "x")]
    raw, family, global_p = permutation_scan(values, labels, groups, metadata, 99, 8)
    assert raw.shape == family.shape == global_p.shape == (1,)
    print("self-test passed")


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cohort, summaries = clean_cohort(args.cache_root)
    feature_frame, metadata, extraction = extract_features(
        args.cache_root, cohort, summaries
    )
    contrasts = (
        (("active_return",), ("success",), "active_return_vs_success"),
        (("stagnation_core",), ("success",), "stagnation_vs_success"),
        (
            ("stagnation_core", "active_return"),
            ("success",),
            "any_failure_vs_success",
        ),
    )
    effect_frames = []
    contrast_summary = []
    for axis, (positive, negative, name) in enumerate(contrasts):
        print(f"effect scan {name}", flush=True)
        frame, info = contrast_effects(
            feature_frame,
            metadata,
            positive,
            negative,
            name,
            args.permutations,
            args.bootstrap,
            args.seed + axis * 1000,
        )
        effect_frames.append(frame)
        contrast_summary.append(info)
    effects = pd.concat(effect_frames, ignore_index=True)
    effects.to_csv(args.out_dir / "effects.csv", index=False)
    cohort[[
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "physical_type",
        "pot2_anchor_query",
        "prospective_cut_query",
        "physical_onset_query",
    ]].to_csv(args.out_dir / "cohort.csv", index=False)
    summary = {
        "date": "2026-08-28",
        "protocol": {
            "window": "first pot2 goal-proxy query through q0+8",
            "outcome_window": "all annotated failure events occur after the cut",
            "denoise_order": "d0: flow time 1.0 (noisiest); d9: flow time 0.1 (last forward before reaching 0.0)",
            "effect": "fixed-init adjusted difference / within-init residual SD",
            "permutation": "labels permuted within initial state",
            "multiplicity": "max absolute permutation effect, per family and global",
            "hard_expert_ids_used": False,
        },
        "cohort": {
            "episodes": len(cohort),
            "class_counts": cohort["physical_type"].value_counts().to_dict(),
            "initial_states": int(cohort["init_state_id"].nunique()),
        },
        "extraction": extraction,
        "contrasts": contrast_summary,
        "feature_families": effects.groupby("family")["key"].nunique().to_dict(),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(render_report(effects, summary))
    plot_action_heatmaps(effects, args.out_dir)
    artifacts = [
        args.out_dir / "cohort.csv",
        args.out_dir / "effects.csv",
        args.out_dir / "summary.json",
        args.out_dir / "report.md",
        args.out_dir / "active_return_action_layer_denoise_effects.png",
    ]
    completion = {
        "status": "complete",
        "script": pathlib.Path(__file__).name,
        "artifacts_sha256": {path.name: file_sha256(path) for path in artifacts},
    }
    (args.out_dir / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n"
    )
    print(f"wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
