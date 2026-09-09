#!/usr/bin/env python3
"""Test layer/denoise routing effects on other captured manipulation tasks.

The analysis has two fixed observation points:

* query 0, before any policy-controlled motion;
* five queries starting at a task-specific physical stage anchor.

All outcome effects are estimated within initial state.  Scene8 aggregate
features are frozen before inspecting the three validation tasks.  Expert IDs
are transferred only between the two spatial tasks, which use the same model
checkpoint; expert indices are not compared across checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import analyze_failure_event_audit as events
import analyze_state_route_layer_denoise_effects as layer


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "analysis/cross-task-layer-denoise-transfer-20260828"
SEED = 20260828
PERMUTATIONS = 2000
PHASE_WINDOW = 5
PREANCHOR_WINDOW = 4

LONG = events.LONG_TASK
TOP = events.TOP_DRAWER_TASK
RAMEKIN = events.RAMEKIN_TASK
STOVE = events.STOVE_TASK
TASK_ORDER = (LONG, TOP, RAMEKIN, STOVE)
VALIDATION_TASKS = (TOP, RAMEKIN, STOVE)
SHORT_TASK = {
    LONG: "two_moka_pots",
    TOP: "top_drawer_then_bowl",
    RAMEKIN: "bowl_from_ramekin",
    STOVE: "bowl_from_stove",
}

AGGREGATE_PHASE_FAMILIES = (
    "state_layer_summary",
    "state_group_summary",
    "action_layer_denoise",
    "action_group_denoise",
)
EXPERT_PHASE_FAMILIES = (
    "state_expert_probability",
    "action_expert_probability",
)
Q0_FAMILIES = (
    "q0_action_layer_denoise",
    "q0_action_group_denoise",
    "q0_action_expert_probability",
)

TEMPORAL_PROFILE_TASKS = (TOP, RAMEKIN, STOVE)


@dataclass(frozen=True)
class TaskInfo:
    task: str
    short_task: str
    checkpoint_sha256: str
    anchor_definition: str
    episodes: int
    successes: int
    failures: int
    mixed_initial_states: int
    anchor_min: int
    anchor_median: float
    anchor_max: int
    anchor_effect_sigma: float
    anchor_permutation_p: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=pathlib.Path, default=events.CACHE_ROOT)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
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


def target_detail(details: str, name: str) -> dict[str, Any]:
    for item in json.loads(details):
        if item["name"] == name:
            return item
    raise ValueError(f"target {name} is absent from event details")


def add_anchor(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    result = frame.copy()
    if task == LONG:
        result["anchor_query"] = result["target_details"].map(
            lambda value: target_detail(value, "moka_pot_2_joint0")[
                "goal_first_query"
            ]
        )
        definition = "first query where moka pot 2 reaches its goal"
    elif task == TOP:
        result["anchor_query"] = result["drawer_goal_first_query"]
        definition = "first query where the top drawer reaches the open threshold"
    elif task in (RAMEKIN, STOVE):
        result["anchor_query"] = result["target_details"].map(
            lambda value: target_detail(value, "akita_black_bowl_1_joint0")[
                "near_close_onsets"
            ][0]
        )
        definition = "first target-near close-command bout"
    else:
        raise ValueError(task)
    valid = result["anchor_query"].notna() & (
        result["anchor_query"] + PHASE_WINDOW <= result["episode_length"]
    )
    result = result[valid].copy()
    result["anchor_query"] = result["anchor_query"].astype(np.int64)
    if len(result) != 512:
        raise ValueError(f"{task}: phase window retained {len(result)}, expected 512")
    return result.sort_values("episode").reset_index(drop=True), definition


def mixed_subset(frame: pd.DataFrame) -> pd.DataFrame:
    mixed = [
        group
        for group, subset in frame.groupby("init_state_id")
        if subset["success"].nunique() == 2
    ]
    return frame[frame["init_state_id"].isin(mixed)].reset_index(drop=True)


def anchor_effect(
    frame: pd.DataFrame, permutations: int, seed: int
) -> tuple[float, float, int]:
    subset = mixed_subset(frame)
    labels = (~subset["success"]).to_numpy(dtype=np.int64)
    groups = subset["init_state_id"].to_numpy(dtype=np.int64)
    values = subset[["anchor_query"]].to_numpy(dtype=np.float64)
    _, _, effect, _ = layer.fixed_effect_shift(values, labels, groups)
    meta = [
        layer.FeatureMeta(
            "anchor_query", "anchor_sentinel", "physical", "stage", "query"
        )
    ]
    raw_p, _, _ = layer.permutation_scan(
        values, labels, groups, meta, permutations, seed
    )
    return float(effect[0]), float(raw_p[0]), int(subset["init_state_id"].nunique())


def load_task_frame(
    cache_root: pathlib.Path,
    task: str,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, TaskInfo]:
    frame, _ = events.process_task(cache_root, task)
    frame, definition = add_anchor(frame, task)
    effect, p_value, mixed = anchor_effect(frame, permutations, seed)
    run = cache_root / task / "right-16x32"
    server_meta = json.loads((run / "client/server_metadata.json").read_text())
    anchor = frame["anchor_query"].to_numpy(dtype=np.int64)
    info = TaskInfo(
        task=task,
        short_task=SHORT_TASK[task],
        checkpoint_sha256=str(server_meta["checkpoint_sha256"]),
        anchor_definition=definition,
        episodes=len(frame),
        successes=int(frame["success"].sum()),
        failures=int((~frame["success"]).sum()),
        mixed_initial_states=mixed,
        anchor_min=int(anchor.min()),
        anchor_median=float(np.median(anchor)),
        anchor_max=int(anchor.max()),
        anchor_effect_sigma=effect,
        anchor_permutation_p=p_value,
    )
    return frame, info


def action_snapshot_metrics(probability: np.ndarray) -> dict[str, np.ndarray]:
    """Input [layer, denoise, action-token, expert], output [layer, denoise]."""
    entropy = layer.normalized_entropy(probability)
    top1 = layer.top1_mass(probability)
    gap = layer.p4p5_gap(probability)
    center = probability.mean(axis=2, keepdims=True)
    dispersion = layer.hellinger(probability, center)
    return {
        "entropy_mean": entropy.mean(axis=2),
        "top1_mean": top1.mean(axis=2),
        "p4p5_gap_mean": gap.mean(axis=2),
        "token_dispersion_mean": dispersion.mean(axis=2),
    }


def phase_feature_row(
    probability: np.ndarray,
    collector: layer.FeatureCollector,
    row: dict[str, Any],
) -> None:
    state = probability[:, :, 0, 0, :]
    action = probability[:, :, :, 1:, :]
    state_metrics = layer.state_layer_metrics(state)
    for layer_axis, layer_number in enumerate(layer.LAYER_NUMBERS):
        for metric, values in state_metrics.items():
            collector.put(
                row,
                values[layer_axis],
                family="state_layer_summary",
                token_role="state",
                scope="layer",
                metric=metric,
                layer=layer_number,
            )
        for expert, value in enumerate(state[:, layer_axis].mean(axis=0)):
            collector.put(
                row,
                value,
                family="state_expert_probability",
                token_role="state",
                scope="layer_expert",
                metric="mean_probability",
                layer=layer_number,
                expert=expert,
            )
    for group_name, layer_axes in layer.LAYER_GROUPS.items():
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

    action_metrics = layer.action_layer_denoise_metrics(action)
    for layer_axis, layer_number in enumerate(layer.LAYER_NUMBERS):
        for denoise in range(layer.N_DENOISE):
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
                    layer=layer_number,
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
                    layer=layer_number,
                    denoise_step=denoise,
                    expert=expert,
                )
    for group_name, layer_axes in layer.LAYER_GROUPS.items():
        for denoise in range(layer.N_DENOISE):
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


def q0_feature_row(
    probability: np.ndarray,
    collector: layer.FeatureCollector,
    row: dict[str, Any],
) -> None:
    action = probability[:, :, 1:, :]
    metrics = action_snapshot_metrics(action)
    for layer_axis, layer_number in enumerate(layer.LAYER_NUMBERS):
        for denoise in range(layer.N_DENOISE):
            for metric, values in metrics.items():
                collector.put(
                    row,
                    values[layer_axis, denoise],
                    family="q0_action_layer_denoise",
                    token_role="action",
                    scope="layer_denoise",
                    metric=metric,
                    layer=layer_number,
                    denoise_step=denoise,
                )
            for expert, value in enumerate(
                action[layer_axis, denoise].mean(axis=0)
            ):
                collector.put(
                    row,
                    value,
                    family="q0_action_expert_probability",
                    token_role="action",
                    scope="layer_denoise_expert",
                    metric="mean_probability",
                    layer=layer_number,
                    denoise_step=denoise,
                    expert=expert,
                )
    for group_name, layer_axes in layer.LAYER_GROUPS.items():
        for denoise in range(layer.N_DENOISE):
            for metric, values in metrics.items():
                collector.put(
                    row,
                    np.mean(values[list(layer_axes), denoise]),
                    family="q0_action_group_denoise",
                    token_role="action",
                    scope="layer_group_denoise",
                    metric=metric,
                    layer_group=group_name,
                    denoise_step=denoise,
                )


def route_offsets(
    run: pathlib.Path, frame: pd.DataFrame
) -> tuple[Any, np.ndarray, np.ndarray]:
    store = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
    episode_ids = np.asarray(store["episode_id"][:], dtype=np.int64)
    control_steps = np.asarray(store["control_step"][:], dtype=np.int64)
    counts = frame.sort_values("episode")["episode_length"].to_numpy(dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)[:-1]].astype(np.int64)
    expected_episode = np.repeat(np.arange(len(frame)), counts)
    if not np.array_equal(episode_ids, expected_episode):
        raise ValueError(f"{run}: episode alignment failed")
    if not np.array_equal(control_steps, np.arange(len(control_steps))):
        raise ValueError(f"{run}: global control-step alignment failed")
    return store, offsets, counts


def extract_task_features(
    cache_root: pathlib.Path,
    task: str,
    cohort: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[layer.FeatureMeta],
    pd.DataFrame,
    list[layer.FeatureMeta],
    pd.DataFrame,
    list[layer.FeatureMeta],
    dict[str, float],
]:
    run = cache_root / task / "right-16x32"
    store, offsets, _ = route_offsets(run, cohort)
    routes = store["hb_router_probs"]
    pre_collector = layer.FeatureCollector()
    phase_collector = layer.FeatureCollector()
    q0_collector = layer.FeatureCollector()
    pre_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    q0_rows: list[dict[str, Any]] = []
    state_denoise_span = 0.0
    q0_state_by_group: dict[int, list[np.ndarray]] = {}
    for axis, item in enumerate(cohort.itertuples(index=False), start=1):
        episode = int(item.episode)
        anchor = int(item.anchor_query)
        base = int(offsets[episode])
        pre_start = base + anchor - PREANCHOR_WINDOW + 1
        if pre_start < base:
            raise ValueError(f"{task}/episode {episode}: pre-anchor window unavailable")
        pre = layer.normalize_probabilities(
            np.asarray(routes[pre_start : base + anchor + 1], dtype=np.float32)
        )
        phase_raw = np.asarray(
            routes[base + anchor : base + anchor + PHASE_WINDOW], dtype=np.float32
        )
        phase = layer.normalize_probabilities(phase_raw)
        pre_state_all = pre[:, :, :, 0, :]
        state_all = phase[:, :, :, 0, :]
        state_denoise_span = max(
            state_denoise_span,
            float(np.max(np.abs(pre_state_all - pre_state_all[:, :, :1, :]))),
            float(np.max(np.abs(state_all - state_all[:, :, :1, :]))),
        )
        common = {
            "task": task,
            "episode": episode,
            "init_state_id": int(item.init_state_id),
            "flow_noise_seed": int(item.flow_noise_seed),
            "success": bool(item.success),
            "anchor_query": anchor,
        }
        pre_row = dict(common)
        phase_feature_row(pre, pre_collector, pre_row)
        pre_rows.append(pre_row)
        phase_row = dict(common)
        phase_feature_row(phase, phase_collector, phase_row)
        phase_rows.append(phase_row)

        q0 = layer.normalize_probabilities(
            np.asarray(routes[base], dtype=np.float32)
        )
        q0_row = dict(common)
        q0_feature_row(q0, q0_collector, q0_row)
        q0_rows.append(q0_row)
        q0_state_by_group.setdefault(int(item.init_state_id), []).append(
            q0[:, 0, 0, :].reshape(-1)
        )
        if axis % 128 == 0 or axis == len(cohort):
            print(f"{SHORT_TASK[task]} feature windows {axis}/{len(cohort)}", flush=True)

    q0_state_span = 0.0
    for values in q0_state_by_group.values():
        array = np.stack(values)
        q0_state_span = max(q0_state_span, float(np.max(np.ptp(array, axis=0))))
    pre_frame = pd.DataFrame(pre_rows)
    phase_frame = pd.DataFrame(phase_rows)
    q0_frame = pd.DataFrame(q0_rows)
    pre_meta = list(pre_collector.meta.values())
    phase_meta = list(phase_collector.meta.values())
    q0_meta = list(q0_collector.meta.values())
    for frame, metadata in (
        (pre_frame, pre_meta),
        (phase_frame, phase_meta),
        (q0_frame, q0_meta),
    ):
        keys = [meta.key for meta in metadata]
        if frame[keys].isna().any().any():
            raise ValueError(f"{task}: non-finite extracted feature")
    return pre_frame, pre_meta, phase_frame, phase_meta, q0_frame, q0_meta, {
        "state_denoise_max_abs_probability_difference": state_denoise_span,
        "q0_state_within_init_max_abs_probability_span": q0_state_span,
    }


def temporal_profile_series(
    task: str, probability: np.ndarray
) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """Decompose selected phase effects around the physical anchor.

    Query-speed rows are indexed by the later query in each transition.  These
    profiles are descriptive localizations of effects selected by the phase
    scan, not a second confirmatory feature search.
    """
    relative_query = np.arange(-3, 5, dtype=np.int64)
    layer_axis = {
        layer_number: axis
        for axis, layer_number in enumerate(layer.LAYER_NUMBERS)
    }
    state = probability[:, :, 0, 0, :]
    action = probability[:, :, :, 1:, :]
    if task == TOP:
        return {
            "state_L14_E21_probability": (
                "query",
                relative_query,
                state[:, layer_axis[14], 21],
            )
        }
    if task == RAMEKIN:
        return {
            "state_L4_top1": (
                "query",
                relative_query,
                layer.top1_mass(state[:, layer_axis[4]]),
            ),
            "action_L13_d9_query_speed": (
                "transition_endpoint",
                relative_query[1:],
                layer.hellinger(
                    action[1:, layer_axis[13], 9].mean(axis=1),
                    action[:-1, layer_axis[13], 9].mean(axis=1),
                ),
            ),
            "action_L15_d3_E21_probability": (
                "query",
                relative_query,
                action[:, layer_axis[15], 3, :, 21].mean(axis=1),
            ),
            "action_L15_d3_E7_probability": (
                "query",
                relative_query,
                action[:, layer_axis[15], 3, :, 7].mean(axis=1),
            ),
        }
    if task == STOVE:
        back = list(layer.LAYER_GROUPS["back_12_15"])
        return {
            "state_L2_query_speed": (
                "transition_endpoint",
                relative_query[1:],
                layer.hellinger(
                    state[1:, layer_axis[2]], state[:-1, layer_axis[2]]
                ),
            ),
            "action_L15_d0_top1": (
                "query",
                relative_query,
                layer.top1_mass(action[:, layer_axis[15], 0]).mean(axis=1),
            ),
            "action_back_d7_top1": (
                "query",
                relative_query,
                layer.top1_mass(action[:, back, 7]).mean(axis=(1, 2)),
            ),
            "action_L2_d8_E9_probability": (
                "query",
                relative_query,
                action[:, layer_axis[2], 8, :, 9].mean(axis=1),
            ),
        }
    raise ValueError(task)


def temporal_localization(
    cache_root: pathlib.Path,
    cohorts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    observations = []
    for task in TEMPORAL_PROFILE_TASKS:
        cohort = cohorts[task]
        run = cache_root / task / "right-16x32"
        store, offsets, _ = route_offsets(run, cohort)
        routes = store["hb_router_probs"]
        for item in cohort.itertuples(index=False):
            base = int(offsets[int(item.episode)])
            anchor = int(item.anchor_query)
            probability = layer.normalize_probabilities(
                np.asarray(
                    routes[base + anchor - 3 : base + anchor + 5],
                    dtype=np.float32,
                )
            )
            for profile, (kind, relative, values) in temporal_profile_series(
                task, probability
            ).items():
                for query, value in zip(relative, values):
                    observations.append(
                        {
                            "task": task,
                            "short_task": SHORT_TASK[task],
                            "profile": profile,
                            "profile_kind": kind,
                            "relative_query": int(query),
                            "episode": int(item.episode),
                            "init_state_id": int(item.init_state_id),
                            "success": bool(item.success),
                            "value": float(value),
                        }
                    )
    long = pd.DataFrame(observations)
    rows = []
    for identifiers, frame in long.groupby(
        ["task", "short_task", "profile", "profile_kind", "relative_query"],
        sort=False,
    ):
        subset = mixed_subset(frame)
        labels = (~subset["success"]).to_numpy(dtype=np.int64)
        groups = subset["init_state_id"].to_numpy(dtype=np.int64)
        values = subset[["value"]].to_numpy(dtype=np.float64)
        beta, residual_sd, effect, _ = layer.fixed_effect_shift(
            values, labels, groups
        )
        loso = []
        for group in np.unique(groups):
            keep = groups != group
            loso.append(
                float(
                    layer.fixed_effect_shift(
                        values[keep], labels[keep], groups[keep]
                    )[2][0]
                )
            )
        rows.append(
            {
                "task": identifiers[0],
                "short_task": identifiers[1],
                "profile": identifiers[2],
                "profile_kind": identifiers[3],
                "relative_query": identifiers[4],
                "failure_mean": float(values[labels == 1].mean()),
                "success_mean": float(values[labels == 0].mean()),
                "adjusted_difference": float(beta[0]),
                "within_init_residual_sd": float(residual_sd[0]),
                "effect_sigma": float(effect[0]),
                "loso_effect_min": min(loso),
                "loso_effect_max": max(loso),
                "loso_same_direction": sum(
                    np.sign(value) == np.sign(effect[0]) for value in loso
                ),
                "loso_folds": len(loso),
            }
        )
    return pd.DataFrame(rows)


def scan_effects(
    frame: pd.DataFrame,
    metadata: list[layer.FeatureMeta],
    task: str,
    window: str,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subset = mixed_subset(frame)
    labels = (~subset["success"]).to_numpy(dtype=np.int64)
    groups = subset["init_state_id"].to_numpy(dtype=np.int64)
    keys = [meta.key for meta in metadata]
    values = subset[keys].to_numpy(dtype=np.float64)
    within = layer.residualize_by_group(values, groups)
    variable = np.sqrt(np.mean(np.square(within), axis=0)) > 1e-10
    values = values[:, variable]
    selected_meta = [meta for meta, keep in zip(metadata, variable) if keep]
    beta, residual_sd, effect, _ = layer.fixed_effect_shift(values, labels, groups)
    raw_p, family_p, global_p = layer.permutation_scan(
        values, labels, groups, selected_meta, permutations, seed
    )
    success_mean = values[labels == 0].mean(axis=0)
    failure_mean = values[labels == 1].mean(axis=0)
    rows = []
    for index, meta in enumerate(selected_meta):
        rows.append(
            {
                "task": task,
                "short_task": SHORT_TASK[task],
                "window": window,
                **asdict(meta),
                "failure_mean": failure_mean[index],
                "success_mean": success_mean[index],
                "adjusted_difference": beta[index],
                "within_init_residual_sd": residual_sd[index],
                "effect_sigma": effect[index],
                "permutation_p_raw": raw_p[index],
                "permutation_p_max_family": family_p[index],
                "permutation_p_max_global": global_p[index],
            }
        )
    return pd.DataFrame(rows), {
        "task": task,
        "window": window,
        "episodes_in_mixed_states": len(subset),
        "failures": int(labels.sum()),
        "successes": int((labels == 0).sum()),
        "mixed_initial_states": int(subset["init_state_id"].nunique()),
        "features": len(selected_meta),
    }


def format_location(row: pd.Series) -> str:
    values = []
    if pd.notna(row.get("layer")):
        values.append(f"L{int(row['layer'])}")
    if pd.notna(row.get("layer_group")):
        values.append(str(row["layer_group"]))
    if pd.notna(row.get("denoise_step")):
        values.append(f"d{int(row['denoise_step'])}")
    if pd.notna(row.get("expert")):
        values.append(f"e{int(row['expert'])}")
    return "/".join(values) or "aggregate"


def select_top_by_family(
    effects: pd.DataFrame,
    task: str,
    window: str,
    families: tuple[str, ...],
    selection_name: str,
) -> pd.DataFrame:
    selected = effects[(effects["task"] == task) & (effects["window"] == window)]
    rows = []
    for family in families:
        family_rows = selected[selected["family"] == family]
        if not len(family_rows):
            continue
        row = family_rows.loc[family_rows["effect_sigma"].abs().idxmax()].copy()
        row["selection"] = selection_name
        row["discovery_task"] = task
        rows.append(row)
    return pd.DataFrame(rows)


def prior_active_keys(metadata: list[layer.FeatureMeta]) -> pd.DataFrame:
    specifications = (
        ("state_layer_summary", "entropy_std_query", 4, None, None),
        ("state_layer_summary", "query_speed_mean", 14, None, None),
        ("action_group_denoise", "entropy_std_query", None, "back_12_15", 2),
        ("action_group_denoise", "top1_std_query", None, "back_12_15", 9),
        ("action_layer_denoise", "entropy_mean", 15, None, 0),
    )
    rows = []
    for family, metric, layer_number, group, denoise in specifications:
        match = [
            meta
            for meta in metadata
            if meta.family == family
            and meta.metric == metric
            and meta.layer == layer_number
            and meta.layer_group == group
            and meta.denoise_step == denoise
        ]
        if len(match) != 1:
            raise ValueError(f"prior feature lookup failed: {specifications}")
        rows.append(
            {
                "selection": "prior_active_return_9q",
                "discovery_task": LONG,
                "key": match[0].key,
                "family": family,
                "metric": metric,
                "expected_direction": -1.0 if "std" in metric or "speed" in metric else 1.0,
            }
        )
    return pd.DataFrame(rows)


def leave_one_state_out(
    frame: pd.DataFrame, key: str
) -> tuple[float, float, int, int]:
    subset = mixed_subset(frame)
    labels = (~subset["success"]).to_numpy(dtype=np.int64)
    groups = subset["init_state_id"].to_numpy(dtype=np.int64)
    values = subset[[key]].to_numpy(dtype=np.float64)
    full_effect = layer.fixed_effect_shift(values, labels, groups)[2][0]
    effects = []
    for group in np.unique(groups):
        keep = groups != group
        remaining = groups[keep]
        if len(np.unique(remaining)) < 2:
            continue
        effects.append(
            layer.fixed_effect_shift(values[keep], labels[keep], remaining)[2][0]
        )
    same = sum(np.sign(value) == np.sign(full_effect) for value in effects)
    return float(np.min(effects)), float(np.max(effects)), same, len(effects)


def build_transfer(
    effects: pd.DataFrame,
    frames: dict[tuple[str, str], pd.DataFrame],
    selected: pd.DataFrame,
    validation_tasks: tuple[str, ...],
    family_size: int,
) -> pd.DataFrame:
    rows = []
    for _, discovery in selected.iterrows():
        discovery_effect = float(discovery.get("effect_sigma", np.nan))
        expected_value = discovery.get("expected_direction", np.nan)
        expected = (
            float(expected_value)
            if pd.notna(expected_value)
            else float(np.sign(discovery_effect))
        )
        window_value = discovery.get("window", np.nan)
        window = "phase5" if pd.isna(window_value) else str(window_value)
        for task in validation_tasks:
            match = effects[
                (effects["task"] == task)
                & (effects["window"] == window)
                & (effects["key"] == discovery["key"])
            ]
            if len(match) != 1:
                raise ValueError(f"transfer feature absent: {task}/{discovery['key']}")
            row = match.iloc[0]
            lo, hi, stable, folds = leave_one_state_out(
                frames[(task, window)], str(discovery["key"])
            )
            rows.append(
                {
                    "selection": discovery["selection"],
                    "discovery_task": discovery["discovery_task"],
                    "validation_task": task,
                    "short_validation_task": SHORT_TASK[task],
                    "window": window,
                    "key": discovery["key"],
                    "family": discovery["family"],
                    "location": format_location(row),
                    "metric": discovery["metric"],
                    "expected_direction": expected,
                    "effect_sigma": row["effect_sigma"],
                    "direction_match": bool(np.sign(row["effect_sigma"]) == np.sign(expected)),
                    "adjusted_difference": row["adjusted_difference"],
                    "permutation_p_raw": row["permutation_p_raw"],
                    "validation_bonferroni_p": min(
                        1.0, float(row["permutation_p_raw"]) * family_size
                    ),
                    "loso_effect_min": lo,
                    "loso_effect_max": hi,
                    "loso_same_direction": stable,
                    "loso_folds": folds,
                }
            )
    return pd.DataFrame(rows)


def task_top_table(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    aggregate = set(AGGREGATE_PHASE_FAMILIES) | {
        "q0_action_layer_denoise",
        "q0_action_group_denoise",
    }
    for (task, window, family), subset in effects.groupby(
        ["task", "window", "family"]
    ):
        if family not in aggregate:
            continue
        rows.append(subset.loc[subset["effect_sigma"].abs().idxmax()])
    result = pd.DataFrame(rows).reset_index(drop=True)
    result["location"] = result.apply(format_location, axis=1)
    return result


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


def temporal_peak_summary(temporal: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, profile), frame in temporal.groupby(
        ["short_task", "profile"], sort=False
    ):
        before = frame[frame["relative_query"] <= 0]
        after = frame[frame["relative_query"] > 0]
        before_peak = before.loc[before["effect_sigma"].abs().idxmax()]
        after_peak = after.loc[after["effect_sigma"].abs().idxmax()]
        rows.append(
            {
                "task": task,
                "profile": profile,
                "pre_or_anchor_query": int(before_peak["relative_query"]),
                "pre_or_anchor_effect_sigma": before_peak["effect_sigma"],
                "post_query": int(after_peak["relative_query"]),
                "post_effect_sigma": after_peak["effect_sigma"],
                "post_loso_min": after_peak["loso_effect_min"],
                "post_loso_max": after_peak["loso_effect_max"],
            }
        )
    return pd.DataFrame(rows)


def render_report(
    task_info: pd.DataFrame,
    task_top: pd.DataFrame,
    transfer: pd.DataFrame,
    spatial_expert: pd.DataFrame,
    temporal: pd.DataFrame,
    audits: dict[str, Any],
) -> str:
    info = task_info.copy()
    info["checkpoint"] = info["checkpoint_sha256"].str[:12]
    top = task_top.copy()
    top["task"] = top["short_task"]
    top = top.sort_values(["window", "task", "family"])
    transfer_display = transfer.copy()
    transfer_display["task"] = transfer_display["short_validation_task"]
    transfer_display["match"] = transfer_display["direction_match"].map(
        {True: "yes", False: "no"}
    )
    spatial_display = spatial_expert.copy()
    if len(spatial_display):
        spatial_display["task"] = spatial_display["short_validation_task"]
        spatial_display["match"] = spatial_display["direction_match"].map(
            {True: "yes", False: "no"}
        )
    temporal_peaks = temporal_peak_summary(temporal)
    lines = [
        "# Cross-task layer and denoise routing transfer",
        "",
        "## Direct result",
        "",
        "No aggregate routing feature survives the task-wide max-T scan in any validation task at the first policy query or in the strict pre-anchor window. Scene8-selected directions also conflict across validation tasks, and selected expert probabilities do not transfer robustly between the two tasks sharing the Spatial checkpoint.",
        "",
        "The strongest validation-task effects occur after the interaction anchor. In the stove task, selected effects are small through the anchor and grow at relative queries +2 to +4. In the ramekin task, the clearest divergence also starts at +2. This timing supports concurrent physical-state encoding rather than a generic early failure code.",
        "",
        "## Cohorts",
        "",
        "Every task contributes all 512 rollouts. `preanchor4` contains the last four decisions ending at the physical anchor; `phase5` contains five queries starting at that anchor. No rollout is removed because it ends early. Effects are failure minus success within initial state, measured in within-init standard deviations.",
        "",
        *markdown(
            info,
            [
                "short_task",
                "successes",
                "failures",
                "mixed_initial_states",
                "anchor_min",
                "anchor_median",
                "anchor_max",
                "anchor_effect_sigma",
                "anchor_permutation_p",
                "checkpoint",
            ],
        ),
        "",
        "Expert IDs are not compared across the Long, Goal, and Spatial checkpoints. The two spatial tasks share one checkpoint, so their expert indices are directly comparable.",
        "",
        "## Frozen Scene8 aggregate transfer",
        "",
        "The rows below were selected on Scene8 or fixed by the previous active-return analysis before examining validation-task effects. `match` means the failure-success direction agrees; Bonferroni p corrects the fixed validation family.",
        "",
        *markdown(
            transfer_display,
            [
                "selection",
                "task",
                "window",
                "family",
                "location",
                "metric",
                "effect_sigma",
                "match",
                "validation_bonferroni_p",
                "loso_effect_min",
                "loso_effect_max",
            ],
        ),
        "",
        "## Spatial-checkpoint expert transfer",
        "",
    ]
    if len(spatial_display):
        lines.extend(
            markdown(
                spatial_display,
                [
                    "selection",
                    "task",
                    "window",
                    "family",
                    "location",
                    "effect_sigma",
                    "match",
                    "validation_bonferroni_p",
                    "loso_effect_min",
                    "loso_effect_max",
                ],
            )
        )
    else:
        lines.append("No estimable spatial expert transfer.")
    lines.extend(
        [
            "",
            "## Temporal localization",
            "",
            "These rows decompose already-selected phase effects; they are descriptive, not a second confirmatory search. Query 0 is the anchor decision before its action is executed. For query-speed profiles, the relative-query index names the later endpoint of the transition.",
            "",
            *markdown(
                temporal_peaks,
                [
                    "task",
                    "profile",
                    "pre_or_anchor_query",
                    "pre_or_anchor_effect_sigma",
                    "post_query",
                    "post_effect_sigma",
                    "post_loso_min",
                    "post_loso_max",
                ],
            ),
            "",
            "## Task-specific exploratory peaks",
            "",
            "These are selected separately inside each task and are not transfer evidence. Family/global max-T p-values account for scanning the displayed feature family/all features in that task-window.",
            "",
            *markdown(
                top,
                [
                    "task",
                    "window",
                    "family",
                    "location",
                    "metric",
                    "effect_sigma",
                    "permutation_p_max_family",
                    "permutation_p_max_global",
                ],
            ),
            "",
            "## Integrity",
            "",
            f"State-token probability maximum difference over denoise steps: `{audits['state_denoise_max_abs_probability_difference']:.8f}`.",
            f"Query-0 state-token maximum within-init probability span: `{audits['q0_state_within_init_max_abs_probability_span']:.8f}`.",
            "Query-0 therefore tests action routing only. The analysis uses soft routing probabilities rather than argmax expert labels, and uses no remaining time, episode length, or future-event timing as a feature.",
            "",
            "The ramekin task has only 12 failures, so its effect estimates are visibly less stable. All findings remain conditional on these four tasks and three checkpoint families.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_transfer(transfer: pd.DataFrame, out_dir: pathlib.Path) -> None:
    selected = transfer[transfer["selection"] == "scene8_phase5_top_per_family"].copy()
    if not len(selected):
        return
    selected["label"] = selected["family"] + "\n" + selected["location"] + "\n" + selected["metric"]
    labels = list(dict.fromkeys(selected["label"]))
    tasks = [SHORT_TASK[task] for task in VALIDATION_TASKS]
    matrix = np.full((len(labels), len(tasks)), np.nan)
    for _, row in selected.iterrows():
        matrix[labels.index(row["label"]), tasks.index(row["short_validation_task"])] = row[
            "effect_sigma"
        ]
    fig, axis = plt.subplots(figsize=(8, max(3.5, 0.8 * len(labels))))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto")
    axis.set_xticks(range(len(tasks)), labels=tasks, rotation=20, ha="right")
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title("Frozen Scene8 phase-5 aggregate features")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            axis.text(x, y, f"{matrix[y, x]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="failure - success effect sigma")
    fig.tight_layout()
    fig.savefig(out_dir / "scene8_aggregate_transfer.png", dpi=160)
    plt.close(fig)


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_self_test() -> None:
    probability = np.full((8, 10, 10, 32), 1.0 / 32, dtype=np.float32)
    metrics = action_snapshot_metrics(probability)
    assert metrics["entropy_mean"].shape == (8, 10)
    assert np.allclose(metrics["entropy_mean"], 1.0)
    assert np.allclose(metrics["top1_mean"], 1.0 / 32)
    frame = pd.DataFrame(
        {
            "success": [True, False] * 4,
            "init_state_id": np.repeat(np.arange(4), 2),
        }
    )
    assert len(mixed_subset(frame)) == 8
    print("self-test passed")


def main() -> None:
    args = parse_args()
    if args.self_test:
        run_self_test()
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cohorts: dict[str, pd.DataFrame] = {}
    task_infos = []
    for index, task in enumerate(TASK_ORDER):
        print(f"physical cohort {SHORT_TASK[task]}", flush=True)
        cohort, info = load_task_frame(
            args.cache_root, task, args.permutations, args.seed + index * 100
        )
        cohorts[task] = cohort
        task_infos.append(info)

    frames: dict[tuple[str, str], pd.DataFrame] = {}
    metadata_by_window: dict[str, list[layer.FeatureMeta]] = {}
    extraction_audits = []
    effect_parts = []
    scan_summaries = []
    for index, task in enumerate(TASK_ORDER):
        pre, pre_meta, phase, phase_meta, q0, q0_meta, audit = (
            extract_task_features(args.cache_root, task, cohorts[task])
        )
        frames[(task, "preanchor4")] = pre
        frames[(task, "phase5")] = phase
        frames[(task, "query0")] = q0
        metadata_by_window.setdefault("preanchor4", pre_meta)
        metadata_by_window.setdefault("phase5", phase_meta)
        metadata_by_window.setdefault("query0", q0_meta)
        extraction_audits.append({"task": task, **audit})
        for window, frame, metadata, offset in (
            ("preanchor4", pre, pre_meta, 0),
            ("phase5", phase, phase_meta, 25),
            ("query0", q0, q0_meta, 50),
        ):
            print(f"effect scan {SHORT_TASK[task]} {window}", flush=True)
            effect, scan = scan_effects(
                frame,
                metadata,
                task,
                window,
                args.permutations,
                args.seed + index * 1000 + offset,
            )
            effect_parts.append(effect)
            scan_summaries.append(scan)
    effects = pd.concat(effect_parts, ignore_index=True)

    scene8_phase = select_top_by_family(
        effects,
        LONG,
        "phase5",
        AGGREGATE_PHASE_FAMILIES,
        "scene8_phase5_top_per_family",
    )
    scene8_pre = select_top_by_family(
        effects,
        LONG,
        "preanchor4",
        AGGREGATE_PHASE_FAMILIES,
        "scene8_preanchor4_top_per_family",
    )
    scene8_q0 = select_top_by_family(
        effects,
        LONG,
        "query0",
        ("q0_action_layer_denoise", "q0_action_group_denoise"),
        "scene8_query0_top_per_family",
    )
    prior = prior_active_keys(metadata_by_window["phase5"])
    frozen = pd.concat(
        [scene8_pre, scene8_phase, scene8_q0, prior],
        ignore_index=True,
        sort=False,
    )
    frozen_count = len(frozen) * len(VALIDATION_TASKS)
    transfer = build_transfer(
        effects, frames, frozen, VALIDATION_TASKS, frozen_count
    )

    spatial_phase = select_top_by_family(
        effects,
        STOVE,
        "phase5",
        EXPERT_PHASE_FAMILIES,
        "stove_phase5_expert_top",
    )
    spatial_pre = select_top_by_family(
        effects,
        STOVE,
        "preanchor4",
        EXPERT_PHASE_FAMILIES,
        "stove_preanchor4_expert_top",
    )
    spatial_q0 = select_top_by_family(
        effects,
        STOVE,
        "query0",
        ("q0_action_expert_probability",),
        "stove_query0_expert_top",
    )
    spatial_selected = pd.concat(
        [spatial_pre, spatial_phase, spatial_q0], ignore_index=True, sort=False
    )
    spatial_expert = build_transfer(
        effects,
        frames,
        spatial_selected,
        (RAMEKIN,),
        len(spatial_selected),
    )
    task_top = task_top_table(effects)
    temporal = temporal_localization(args.cache_root, cohorts)

    effects.to_csv(args.out_dir / "effects.csv", index=False)
    transfer.to_csv(args.out_dir / "frozen_transfer.csv", index=False)
    spatial_expert.to_csv(args.out_dir / "spatial_expert_transfer.csv", index=False)
    task_top.to_csv(args.out_dir / "task_specific_peaks.csv", index=False)
    temporal.to_csv(args.out_dir / "temporal_localization.csv", index=False)
    pd.DataFrame([asdict(info) for info in task_infos]).to_csv(
        args.out_dir / "task_cohorts.csv", index=False
    )
    cohort_rows = []
    for task, frame in cohorts.items():
        cohort_rows.append(
            frame[
                [
                    "episode",
                    "init_state_id",
                    "flow_noise_seed",
                    "success",
                    "episode_length",
                    "anchor_query",
                ]
            ].assign(task=task)
        )
    pd.concat(cohort_rows, ignore_index=True).to_csv(
        args.out_dir / "episodes.csv", index=False
    )

    audits = {
        "state_denoise_max_abs_probability_difference": max(
            item["state_denoise_max_abs_probability_difference"]
            for item in extraction_audits
        ),
        "q0_state_within_init_max_abs_probability_span": max(
            item["q0_state_within_init_max_abs_probability_span"]
            for item in extraction_audits
        ),
        "by_task": extraction_audits,
    }
    task_info_frame = pd.DataFrame([asdict(info) for info in task_infos])
    summary = {
        "date": "2026-08-28",
        "protocol": {
            "preanchor_window": "four queries q_anchor-3 through q_anchor; ends before the anchor action is executed",
            "phase_window": "five queries q_anchor through q_anchor+4",
            "query0": "single first policy query; action routing only",
            "temporal_localization": "descriptive decomposition from q_anchor-3 through q_anchor+4 of features selected by the phase scan",
            "effect": "failure-minus-success fixed-init difference / within-init residual SD",
            "multiplicity": "within-task permutation max-T for scans; Bonferroni for frozen validation rows",
            "hard_top1_expert_assignments_used": False,
            "expert_transfer": "only between the two tasks sharing the Spatial checkpoint",
        },
        "task_info": [asdict(info) for info in task_infos],
        "scan_summaries": scan_summaries,
        "audits": audits,
        "frozen_validation_rows": len(transfer),
        "spatial_expert_validation_rows": len(spatial_expert),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "report.md").write_text(
        render_report(
            task_info_frame,
            task_top,
            transfer,
            spatial_expert,
            temporal,
            audits,
        )
    )
    plot_transfer(transfer, args.out_dir)
    artifacts = [
        args.out_dir / "effects.csv",
        args.out_dir / "frozen_transfer.csv",
        args.out_dir / "spatial_expert_transfer.csv",
        args.out_dir / "task_specific_peaks.csv",
        args.out_dir / "temporal_localization.csv",
        args.out_dir / "task_cohorts.csv",
        args.out_dir / "episodes.csv",
        args.out_dir / "summary.json",
        args.out_dir / "report.md",
        args.out_dir / "scene8_aggregate_transfer.png",
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
