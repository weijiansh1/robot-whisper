#!/usr/bin/env python3
"""Analyze same-noise input-version counterfactual MoE captures.

The primary train-free score compares how much the back-layer action-token
routing changes when the input version changes with how much it changes under
flow-noise resampling at a fixed input.  The natural decision boundary is one:
below one, latent sampling moves the action route more than new feedback does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from itertools import combinations
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from moe_self_reference_selector import (  # noqa: E402
    SelfReferenceConfig,
    SelfReferenceCouplingCollapseAlarm,
)


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/input_version_counterfactual.json"
DEFAULT_CAPTURE = PACKAGE_ROOT / "results/input_version_counterfactual/formal_capture"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/input_version_counterfactual/analysis"
SCHEMA = "himoe.input_version_counterfactual.analysis.v1"


def normalized(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = normalized(left)
    right = normalized(right)
    coefficient = np.sqrt(left * right).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def subset_distance(
    left: np.ndarray,
    right: np.ndarray,
    layers: Sequence[int],
    flows: slice,
    tokens: slice,
) -> float:
    distance = hellinger(
        left[np.asarray(layers), flows, tokens, :],
        right[np.asarray(layers), flows, tokens, :],
    )
    return float(np.mean(distance))


def effects(
    previous: np.ndarray,
    current: np.ndarray,
    layers: Sequence[int],
    flows: slice,
    tokens: slice,
    draws: Sequence[int],
    epsilon: float,
) -> Dict[str, float]:
    draws = list(map(int, draws))
    observation_values = [
        subset_distance(previous[draw], current[draw], layers, flows, tokens)
        for draw in draws
    ]
    noise_values: List[float] = []
    for source in (previous, current):
        for first, second in combinations(draws, 2):
            noise_values.append(
                subset_distance(source[first], source[second], layers, flows, tokens)
            )
    observation = float(np.median(observation_values))
    noise = float(np.median(noise_values))
    ratio = observation / max(noise, epsilon)
    return {
        "input_version_effect_h": observation,
        "input_version_effect_min_h": float(np.min(observation_values)),
        "input_version_effect_max_h": float(np.max(observation_values)),
        "noise_effect_h": noise,
        "noise_effect_min_h": float(np.min(noise_values)),
        "noise_effect_max_h": float(np.max(noise_values)),
        "input_to_noise_ratio": ratio,
        "log_input_to_noise_ratio": float(np.log(max(ratio, epsilon))),
    }


def route_rows_for_episode(store: zarr.Group, episode_id: int) -> np.ndarray:
    episodes = np.asarray(store["episode_id"][:])
    control_steps = np.asarray(store["control_step"][:])
    rows = np.flatnonzero(episodes == int(episode_id))
    return rows[np.argsort(control_steps[rows])]


def replay_fidelity(
    archive: Dict[str, np.ndarray],
    store: zarr.Group,
    model_layers: Sequence[int],
) -> pd.DataFrame:
    row_cache: Dict[int, np.ndarray] = {}
    output: List[Dict[str, Any]] = []
    for record in range(len(archive["candidate"])):
        episode_id = int(archive["episode_id"][record])
        if episode_id not in row_cache:
            row_cache[episode_id] = route_rows_for_episode(store, episode_id)
        query = int(archive["current_query"][record])
        source_row = int(row_cache[episode_id][query])
        source = normalized(np.asarray(store["hb_router_probs"][source_row], np.float32))
        replay = normalized(archive["routes_current"][record, 0])
        distance = hellinger(source, replay)
        action_rms = float(
            np.sqrt(
                np.mean(
                    (
                        archive["actions_current"][record, 0]
                        - archive["source_actions_current"][record]
                    )
                    ** 2
                )
            )
        )
        for stored_layer, model_layer in enumerate(model_layers):
            output.append(
                {
                    "record_index": record,
                    "candidate": int(archive["candidate"][record]),
                    "outcome": (
                        "failed_grasp"
                        if bool(archive["failed_grasp"][record])
                        else "success"
                    ),
                    "relative_query": int(archive["relative_query"][record]),
                    "current_query": query,
                    "stored_layer": stored_layer,
                    "model_layer": int(model_layer),
                    "route_replay_mean_h": float(np.mean(distance[stored_layer])),
                    "route_replay_max_h": float(np.max(distance[stored_layer])),
                    "route_replay_state_h": float(np.mean(distance[stored_layer, :, 0])),
                    "route_replay_action_h": float(np.mean(distance[stored_layer, :, 1:])),
                    "action_replay_rms": action_rms,
                }
            )
    return pd.DataFrame(output)


def calculate_metrics(
    archive: Dict[str, np.ndarray], config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epsilon = float(config["epsilon"])
    model_layers = list(map(int, config["hb_model_layers"]))
    groups = {
        "front": list(map(int, config["front_stored_layers"])),
        "back": list(map(int, config["back_stored_layers"])),
    }
    flow_sets = {
        "full": slice(None),
        "late": slice(6, None),
        "final": slice(9, 10),
    }
    token_sets = {
        "state": slice(int(config["state_token"]), int(config["state_token"]) + 1),
        "action": slice(*map(int, config["action_token_slice"])),
    }
    n_draws = int(archive["routes_current"].shape[1])
    all_draws = list(range(n_draws))
    minimal_draws = [0, 1]
    group_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    primary_rows: List[Dict[str, Any]] = []

    for record in range(len(archive["candidate"])):
        previous = normalized(archive["routes_previous"][record])
        current = normalized(archive["routes_current"][record])
        identity = {
            "record_index": record,
            "candidate": int(archive["candidate"][record]),
            "episode_id": int(archive["episode_id"][record]),
            "outcome": (
                "failed_grasp" if bool(archive["failed_grasp"][record]) else "success"
            ),
            "relative_query": int(archive["relative_query"][record]),
            "previous_query": int(archive["previous_query"][record]),
            "current_query": int(archive["current_query"][record]),
        }
        metric_lookup: Dict[Tuple[str, str, str, str], Dict[str, float]] = {}
        for group_name, layers in groups.items():
            for flow_name, flows in flow_sets.items():
                for token_name, tokens in token_sets.items():
                    for draw_name, draws in (
                        ("m4", all_draws),
                        ("m2", minimal_draws),
                    ):
                        value = effects(
                            previous,
                            current,
                            layers,
                            flows,
                            tokens,
                            draws,
                            epsilon,
                        )
                        metric_lookup[(group_name, flow_name, token_name, draw_name)] = value
                        group_rows.append(
                            {
                                **identity,
                                "layer_group": group_name,
                                "flow_range": flow_name,
                                "token_class": token_name,
                                "draw_ablation": draw_name,
                                **value,
                            }
                        )

        for stored_layer, model_layer in enumerate(model_layers):
            state = effects(
                previous,
                current,
                [stored_layer],
                slice(None),
                token_sets["state"],
                all_draws,
                epsilon,
            )
            action = effects(
                previous,
                current,
                [stored_layer],
                slice(None),
                token_sets["action"],
                all_draws,
                epsilon,
            )
            layer_rows.append(
                {
                    **identity,
                    "stored_layer": stored_layer,
                    "model_layer": int(model_layer),
                    "layer_group": "front" if stored_layer < 4 else "back",
                    "state_input_effect_h": state["input_version_effect_h"],
                    "state_noise_effect_h": state["noise_effect_h"],
                    "action_input_effect_h": action["input_version_effect_h"],
                    "action_noise_effect_h": action["noise_effect_h"],
                    "action_input_to_noise_ratio": action["input_to_noise_ratio"],
                    "input_transmission_action_over_state": (
                        action["input_version_effect_h"]
                        / max(state["input_version_effect_h"], epsilon)
                    ),
                }
            )

        front_state = metric_lookup[("front", "full", "state", "m4")]
        back_action = metric_lookup[("back", "full", "action", "m4")]
        back_action_m2 = metric_lookup[("back", "full", "action", "m2")]
        transmission = back_action["input_version_effect_h"] / max(
            front_state["input_version_effect_h"], epsilon
        )
        action_ratio = back_action["input_to_noise_ratio"]
        action_ratio_m2 = back_action_m2["input_to_noise_ratio"]
        primary_rows.append(
            {
                **identity,
                "front_state_input_effect_h": front_state["input_version_effect_h"],
                "front_state_noise_effect_h": front_state["noise_effect_h"],
                "back_action_input_effect_h": back_action["input_version_effect_h"],
                "back_action_noise_effect_h": back_action["noise_effect_h"],
                "back_action_input_to_noise_ratio": action_ratio,
                "back_action_noise_dominance": 1.0 / max(action_ratio, epsilon),
                "back_action_log_noise_dominance": float(
                    -np.log(max(action_ratio, epsilon))
                ),
                "input_transmission_back_action_over_front_state": transmission,
                "trainfree_alarm_m4": bool(
                    front_state["input_version_effect_h"] > epsilon
                    and action_ratio < 1.0
                ),
                "back_action_input_to_noise_ratio_m2": action_ratio_m2,
                "trainfree_alarm_m2": bool(
                    front_state["input_version_effect_h"] > epsilon
                    and action_ratio_m2 < 1.0
                ),
            }
        )
    return (
        pd.DataFrame(group_rows),
        pd.DataFrame(layer_rows),
        pd.DataFrame(primary_rows),
    )


def aligned_summary(primary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "front_state_input_effect_h",
        "back_action_input_effect_h",
        "back_action_noise_effect_h",
        "back_action_input_to_noise_ratio",
        "back_action_noise_dominance",
        "input_transmission_back_action_over_front_state",
    ]
    rows: List[Dict[str, Any]] = []
    for relative in sorted(primary.relative_query.unique()):
        subset = primary[primary.relative_query == relative]
        failure = subset[subset.outcome == "failed_grasp"]
        controls = subset[subset.outcome == "success"]
        if len(failure) != 1 or len(controls) == 0:
            continue
        for metric in metrics:
            failed_value = float(failure.iloc[0][metric])
            values = controls[metric].to_numpy(float)
            rows.append(
                {
                    "relative_query": int(relative),
                    "metric": metric,
                    "failed_value": failed_value,
                    "success_mean": float(np.mean(values)),
                    "success_min": float(np.min(values)),
                    "success_max": float(np.max(values)),
                    "failed_over_success_mean": failed_value / max(float(np.mean(values)), 1e-12),
                    "failed_rank_high_of_8": int(1 + np.sum(values < failed_value)),
                    "relation_to_success_range": (
                        "above_all"
                        if failed_value > np.max(values)
                        else "below_all"
                        if failed_value < np.min(values)
                        else "inside"
                    ),
                    "n_success_controls": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def alarm_summary(primary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for relative in sorted(primary.relative_query.unique()):
        subset = primary[primary.relative_query == relative]
        failure = subset[subset.outcome == "failed_grasp"]
        controls = subset[subset.outcome == "success"]
        rows.append(
            {
                "relative_query": int(relative),
                "failed_alarm_m4": bool(failure.trainfree_alarm_m4.iloc[0]),
                "success_alarms_m4": int(controls.trainfree_alarm_m4.sum()),
                "success_controls": int(len(controls)),
                "failed_alarm_m2": bool(failure.trainfree_alarm_m2.iloc[0]),
                "success_alarms_m2": int(controls.trainfree_alarm_m2.sum()),
                "failed_ratio_m4": float(
                    failure.back_action_input_to_noise_ratio.iloc[0]
                ),
                "success_ratio_m4_mean": float(
                    controls.back_action_input_to_noise_ratio.mean()
                ),
                "success_ratio_m4_min": float(
                    controls.back_action_input_to_noise_ratio.min()
                ),
                "success_ratio_m4_max": float(
                    controls.back_action_input_to_noise_ratio.max()
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_self_reference_v3(
    primary: pd.DataFrame, store: zarr.Group
) -> pd.DataFrame:
    config = SelfReferenceConfig.load(
        PACKAGE_ROOT / "configs/self_reference_coupling_collapse_v3.json"
    )
    rows: List[Dict[str, Any]] = []
    for candidate in sorted(primary.candidate.unique()):
        source = primary[primary.candidate == candidate].iloc[0]
        episode_id = int(source.episode_id)
        route_rows = route_rows_for_episode(store, episode_id)
        routes = np.asarray(
            store["hb_router_probs"].oindex[route_rows, :, :, :, :], np.float32
        )
        selector = SelfReferenceCouplingCollapseAlarm(config)
        alarms: List[int] = []
        for route in routes:
            decision = selector.update(route)
            if decision.alarm:
                alarms.append(int(decision.query))
        closure_query = int(source.current_query - source.relative_query)
        rows.append(
            {
                "candidate": int(candidate),
                "episode_id": episode_id,
                "outcome": str(source.outcome),
                "queries": int(len(routes)),
                "closure_query": closure_query,
                "self_reference_v3_alarm_count": len(alarms),
                "self_reference_v3_first_alarm_query": (
                    alarms[0] if alarms else np.nan
                ),
                "self_reference_v3_first_alarm_relative_to_closure": (
                    alarms[0] - closure_query if alarms else np.nan
                ),
                "self_reference_v3_alarm_queries": "|".join(map(str, alarms)),
                "counterfactual_alarm_count_in_window": int(
                    primary[
                        (primary.candidate == candidate)
                        & primary.trainfree_alarm_m4
                    ].shape[0]
                ),
                "counterfactual_first_alarm_relative_in_window": (
                    int(
                        primary[
                            (primary.candidate == candidate)
                            & primary.trainfree_alarm_m4
                        ].relative_query.min()
                    )
                    if bool(
                        (
                            (primary.candidate == candidate)
                            & primary.trainfree_alarm_m4
                        ).any()
                    )
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_primary(primary: pd.DataFrame, output: pathlib.Path) -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 160}
    )
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), constrained_layout=True)
    success = primary[primary.outcome == "success"]
    failure = primary[primary.outcome == "failed_grasp"]
    relatives = np.asarray(sorted(primary.relative_query.unique()), dtype=int)

    def band(ax: Any, metric: str, ylabel: str, threshold: float = None) -> None:
        grouped = success.groupby("relative_query")[metric]
        mean = grouped.mean().reindex(relatives).to_numpy(float)
        low = grouped.min().reindex(relatives).to_numpy(float)
        high = grouped.max().reindex(relatives).to_numpy(float)
        ax.fill_between(relatives, low, high, color="#78a6b8", alpha=0.25, label="7 success range")
        ax.plot(relatives, mean, color="#267187", marker="o", markersize=3, label="success mean")
        aligned_failure = failure.set_index("relative_query").reindex(relatives)
        ax.plot(
            relatives,
            aligned_failure[metric].to_numpy(float),
            color="#b43c3c",
            marker="o",
            linewidth=2,
            label="failed grasp",
        )
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        if threshold is not None:
            ax.axhline(threshold, color="#555555", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Queries from gripper closure")
        ax.set_ylabel(ylabel)

    band(
        axes[0, 0],
        "back_action_input_to_noise_ratio",
        "Input-version effect / noise effect",
        1.0,
    )
    axes[0, 0].set_title("Back-layer action route: feedback vs latent noise")
    axes[0, 0].legend(frameon=False, fontsize=8)
    band(
        axes[0, 1],
        "front_state_input_effect_h",
        "Hellinger distance",
    )
    axes[0, 1].set_title("New input reaches the front state token")
    band(
        axes[1, 0],
        "input_transmission_back_action_over_front_state",
        "Back action / front state response",
    )
    axes[1, 0].set_title("Input-change transmission coefficient")
    band(
        axes[1, 1],
        "back_action_noise_dominance",
        "Noise effect / input-version effect",
        1.0,
    )
    axes[1, 1].set_title("Train-free open-loop dominance score")
    figure.suptitle("Same-noise counterfactual around grasp closure", fontsize=12)
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def plot_layers(layer: pd.DataFrame, output: pathlib.Path) -> None:
    relatives = np.asarray(sorted(layer.relative_query.unique()), dtype=int)
    model_layers = list(dict.fromkeys(layer.model_layer.tolist()))
    failure = layer[layer.outcome == "failed_grasp"]
    success = layer[layer.outcome == "success"]
    matrices = []
    for metric in (
        "action_input_to_noise_ratio",
        "input_transmission_action_over_state",
    ):
        values = np.full((len(model_layers), len(relatives)), np.nan, dtype=float)
        for y, model_layer in enumerate(model_layers):
            for x, relative in enumerate(relatives):
                failed_row = failure[
                    (failure.model_layer == model_layer)
                    & (failure.relative_query == relative)
                ]
                control_rows = success[
                    (success.model_layer == model_layer)
                    & (success.relative_query == relative)
                ]
                if len(failed_row) and len(control_rows):
                    values[y, x] = float(failed_row.iloc[0][metric]) / max(
                        float(control_rows[metric].mean()), 1e-12
                    )
        matrices.append(values)

    figure, axes = plt.subplots(2, 1, figsize=(11, 6.7), constrained_layout=True)
    titles = [
        "Failure / success mean: action input-to-noise ratio",
        "Failure / success mean: action/state transmission",
    ]
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=0.4, vmax=1.6)
        ax.set_xticks(np.arange(len(relatives)), relatives)
        ax.set_yticks(np.arange(len(model_layers)), model_layers)
        ax.set_xlabel("Queries from gripper closure")
        ax.set_ylabel("Model layer")
        ax.set_title(title)
        figure.colorbar(image, ax=ax, label="ratio")
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--capture", type=pathlib.Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    capture = args.capture.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    capture_manifest_path = capture / "manifest.json"
    capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
    if capture_manifest.get("status") != "complete":
        raise RuntimeError("capture is not complete")
    archive_path = capture / "paired_counterfactual_routes.npz"
    with np.load(str(archive_path), allow_pickle=False) as source:
        archive = {name: np.asarray(source[name]) for name in source.files}
    if str(archive["schema"].item()) != "himoe.input_version_counterfactual.capture.v1":
        raise RuntimeError("unsupported capture schema")
    if bool(archive["training"].item()):
        raise RuntimeError("capture unexpectedly declares training")
    if archive["routes_current"].shape != archive["routes_previous"].shape:
        raise RuntimeError("paired route tensors do not align")
    if archive["routes_current"].shape[1:] != (4, 8, 10, 11, 32):
        raise RuntimeError("unexpected paired route shape %s" % (archive["routes_current"].shape,))

    raw_root = pathlib.Path(capture_manifest["raw_run_root"])
    route_store_path = raw_root / "formal/server/routes.zarr"
    store = zarr.open_group(str(route_store_path), mode="r")
    fidelity = replay_fidelity(archive, store, config["hb_model_layers"])
    group, layer, primary = calculate_metrics(archive, config)
    aligned = aligned_summary(primary)
    alarms = alarm_summary(primary)
    selector_comparison = compare_self_reference_v3(primary, store)

    fidelity.to_csv(tables / "replay_fidelity_by_layer.csv", index=False)
    group.to_csv(tables / "counterfactual_group_metrics.csv", index=False)
    layer.to_csv(tables / "counterfactual_layer_metrics.csv", index=False)
    primary.to_csv(tables / "primary_scores.csv", index=False)
    aligned.to_csv(tables / "aligned_failure_vs_success.csv", index=False)
    alarms.to_csv(tables / "trainfree_alarm_by_relative_query.csv", index=False)
    selector_comparison.to_csv(
        tables / "comparison_to_self_reference_v3.csv", index=False
    )
    plot_primary(primary, figures / "counterfactual_feedback_transmission.png")
    plot_layers(layer, figures / "layerwise_feedback_transmission.png")

    failed = primary[primary.outcome == "failed_grasp"].sort_values("relative_query")
    controls = primary[primary.outcome == "success"]
    failed_alarm_relatives = failed.loc[failed.trainfree_alarm_m4, "relative_query"].astype(int).tolist()
    pre_event = primary[primary.relative_query < 0]
    post_event = primary[primary.relative_query >= 0]
    fidelity_record = fidelity.groupby("record_index").agg(
        route_replay_mean_h=("route_replay_mean_h", "mean"),
        route_replay_max_h=("route_replay_max_h", "max"),
        action_replay_rms=("action_replay_rms", "first"),
    )
    front_state_noise = group[
        (group.layer_group == "front")
        & (group.flow_range == "full")
        & (group.token_class == "state")
        & (group.draw_ablation == "m4")
    ]
    target_metrics: Dict[str, Any] = {}
    for relative in (-3, -2, -1, 0, 1, 2, 4, 8):
        row = failed[failed.relative_query == relative]
        if len(row):
            target_metrics[str(relative)] = {
                "back_action_input_to_noise_ratio": float(
                    row.back_action_input_to_noise_ratio.iloc[0]
                ),
                "noise_dominates_action_route": bool(row.trainfree_alarm_m4.iloc[0]),
                "success_control_range": [
                    float(
                        controls[controls.relative_query == relative]
                        .back_action_input_to_noise_ratio.min()
                    ),
                    float(
                        controls[controls.relative_query == relative]
                        .back_action_input_to_noise_ratio.max()
                    ),
                ],
            }

    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "score_uses_physics": False,
        "score_uses_outcome": False,
        "score_uses_normal_trajectory_bank": False,
        "score_uses_task_identity": False,
        "score": (
            "median same-noise adjacent-input Hellinger effect on back-layer "
            "action tokens divided by median fixed-input cross-noise effect"
        ),
        "alarm": "score < 1.0 (flow noise moves the action route more than new input)",
        "records": int(len(primary)),
        "failed_records": int(len(failed)),
        "success_control_records": int(len(controls)),
        "candidates": int(primary.candidate.nunique()),
        "noise_draws": int(archive["routes_current"].shape[1]),
        "inference_calls": int(capture_manifest["inference_calls"]),
        "failed_alarm_relative_queries_m4": failed_alarm_relatives,
        "failed_first_alarm_relative_query_m4": (
            min(failed_alarm_relatives) if failed_alarm_relatives else None
        ),
        "success_control_alarm_rate_all_records_m4": float(
            controls.trainfree_alarm_m4.mean()
        ),
        "success_control_trajectories_with_any_alarm_m4": int(
            controls.groupby("candidate").trainfree_alarm_m4.any().sum()
        ),
        "success_control_trajectories_m4": int(controls.candidate.nunique()),
        "relative_minus_1": {
            "failed_alarm_m4": bool(
                failed[failed.relative_query == -1].trainfree_alarm_m4.iloc[0]
            ),
            "success_control_alarms_m4": int(
                controls[controls.relative_query == -1].trainfree_alarm_m4.sum()
            ),
            "success_controls": int(
                (controls.relative_query == -1).sum()
            ),
        },
        "success_control_alarm_rate_pre_event_m4": float(
            pre_event[pre_event.outcome == "success"].trainfree_alarm_m4.mean()
        ),
        "failed_alarm_rate_pre_event_m4": float(
            pre_event[pre_event.outcome == "failed_grasp"].trainfree_alarm_m4.mean()
        ),
        "success_control_alarm_rate_post_event_m4": float(
            post_event[post_event.outcome == "success"].trainfree_alarm_m4.mean()
        ),
        "failed_alarm_rate_post_event_m4": float(
            post_event[post_event.outcome == "failed_grasp"].trainfree_alarm_m4.mean()
        ),
        "minimal_four_call_m2": {
            "failed_alarm_relative_queries": failed.loc[
                failed.trainfree_alarm_m2, "relative_query"
            ].astype(int).tolist(),
            "success_control_alarm_rate": float(controls.trainfree_alarm_m2.mean()),
            "success_control_trajectories_with_any_alarm": int(
                controls.groupby("candidate").trainfree_alarm_m2.any().sum()
            ),
        },
        "comparison_to_self_reference_v3": {
            "failed_first_alarm_relative_to_closure": int(
                selector_comparison[
                    selector_comparison.outcome == "failed_grasp"
                ].self_reference_v3_first_alarm_relative_to_closure.iloc[0]
            ),
            "failed_counterfactual_first_alarm_relative": int(
                selector_comparison[
                    selector_comparison.outcome == "failed_grasp"
                ].counterfactual_first_alarm_relative_in_window.iloc[0]
            ),
            "successful_siblings_with_v3_alarm": int(
                (
                    selector_comparison[
                        selector_comparison.outcome == "success"
                    ].self_reference_v3_alarm_count
                    > 0
                ).sum()
            ),
            "successful_siblings_with_counterfactual_alarm_in_window": int(
                (
                    selector_comparison[
                        selector_comparison.outcome == "success"
                    ].counterfactual_alarm_count_in_window
                    > 0
                ).sum()
            ),
        },
        "architecture_audit": {
            "front_state_noise_effect_max_h": float(
                front_state_noise.noise_effect_h.max()
            ),
            "front_state_noise_effect_median_h": float(
                front_state_noise.noise_effect_h.median()
            ),
            "interpretation": (
                "The front state token is causally before noisy action tokens; "
                "its route is effectively invariant to flow-noise resampling."
            ),
        },
        "replay_fidelity": {
            "route_mean_h": float(fidelity_record.route_replay_mean_h.mean()),
            "route_p95_mean_h": float(
                fidelity_record.route_replay_mean_h.quantile(0.95)
            ),
            "route_global_max_h": float(fidelity_record.route_replay_max_h.max()),
            "action_rms_mean": float(fidelity_record.action_replay_rms.mean()),
            "action_rms_p95": float(fidelity_record.action_replay_rms.quantile(0.95)),
        },
        "failed_target_timeline": target_metrics,
        "limitations": [
            "One failed-grasp trajectory is compared with seven matched successful siblings.",
            "The paired inference is causal with respect to input version at a fixed noise, but the source input versions come from observational trajectories.",
            "The score requires shadow inference and is evaluated here as a mechanism case study, not a population-level detector benchmark.",
        ],
        "inputs": {
            "config": str(config_path),
            "config_sha256": file_digest(config_path),
            "capture_manifest": str(capture_manifest_path),
            "capture_manifest_sha256": file_digest(capture_manifest_path),
            "capture_npz": str(archive_path),
            "capture_npz_sha256": file_digest(archive_path),
            "source_route_store": str(route_store_path),
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
