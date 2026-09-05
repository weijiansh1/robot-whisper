#!/usr/bin/env python3
"""Decompose adjacent-input MoE response into image and 8-D state effects."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/input_version_counterfactual.json"
DEFAULT_PAIRED = PACKAGE_ROOT / "results/input_version_counterfactual/formal_capture"
DEFAULT_MODALITY = PACKAGE_ROOT / "results/input_version_counterfactual/modality_capture"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/input_version_counterfactual/modality_analysis"
SCHEMA = "himoe.input_modality_counterfactual.analysis.v1"


def normalized(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, np.float32), 0.0)
    return value / np.maximum(value.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left, right = normalized(left), normalized(right)
    return np.sqrt(
        np.clip(1.0 - np.sqrt(left * right).sum(axis=-1), 0.0, 1.0)
    )


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


def distance(
    left: np.ndarray,
    right: np.ndarray,
    layers: Sequence[int],
    flows: slice,
    tokens: slice,
) -> float:
    value = hellinger(
        left[np.asarray(layers), flows, tokens, :],
        right[np.asarray(layers), flows, tokens, :],
    )
    return float(np.mean(value))


def decompose(
    cells: Mapping[str, np.ndarray],
    layers: Sequence[int],
    flows: slice,
    tokens: slice,
    epsilon: float,
) -> Dict[str, float]:
    draws = range(cells["V00"].shape[0])
    total = [
        distance(cells["V00"][draw], cells["V11"][draw], layers, flows, tokens)
        for draw in draws
    ]
    vision_old_state = [
        distance(cells["V00"][draw], cells["V10"][draw], layers, flows, tokens)
        for draw in draws
    ]
    vision_new_state = [
        distance(cells["V01"][draw], cells["V11"][draw], layers, flows, tokens)
        for draw in draws
    ]
    state_old_image = [
        distance(cells["V00"][draw], cells["V01"][draw], layers, flows, tokens)
        for draw in draws
    ]
    state_new_image = [
        distance(cells["V10"][draw], cells["V11"][draw], layers, flows, tokens)
        for draw in draws
    ]
    vision = float(np.median(vision_old_state + vision_new_state))
    state = float(np.median(state_old_image + state_new_image))
    total_value = float(np.median(total))
    noise_values: List[float] = []
    for cell in cells.values():
        for first, second in combinations(range(cell.shape[0]), 2):
            noise_values.append(
                distance(cell[first], cell[second], layers, flows, tokens)
            )
    noise = float(np.median(noise_values))
    denominator = max(vision + state, epsilon)
    return {
        "total_input_effect_h": total_value,
        "vision_marginal_effect_h": vision,
        "state_marginal_effect_h": state,
        "vision_effect_old_state_h": float(np.median(vision_old_state)),
        "vision_effect_new_state_h": float(np.median(vision_new_state)),
        "state_effect_old_image_h": float(np.median(state_old_image)),
        "state_effect_new_image_h": float(np.median(state_new_image)),
        "noise_effect_h": noise,
        "total_input_to_noise_ratio": total_value / max(noise, epsilon),
        "vision_to_noise_ratio": vision / max(noise, epsilon),
        "state_to_noise_ratio": state / max(noise, epsilon),
        "vision_fraction_of_marginal_effect": vision / denominator,
        "state_fraction_of_marginal_effect": state / denominator,
        "total_over_sum_marginal_effect": total_value / denominator,
    }


def calculate(
    paired: Dict[str, np.ndarray],
    modality: Dict[str, np.ndarray],
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = {
        "front": list(map(int, config["front_stored_layers"])),
        "back": list(map(int, config["back_stored_layers"])),
    }
    token_sets = {
        "state": slice(int(config["state_token"]), int(config["state_token"]) + 1),
        "action": slice(*map(int, config["action_token_slice"])),
    }
    flow_sets = {"full": slice(None), "late": slice(6, None), "final": slice(9, 10)}
    epsilon = float(config["epsilon"])
    group_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    primary_rows: List[Dict[str, Any]] = []

    for modality_index, paired_index_value in enumerate(modality["paired_record_index"]):
        paired_index = int(paired_index_value)
        record = records[paired_index]
        identity = {
            "paired_record_index": paired_index,
            "candidate": int(record["candidate"]),
            "episode_id": int(record["episode_id"]),
            "outcome": str(record["outcome"]),
            "relative_query": int(record["relative_query"]),
            "previous_query": int(record["previous_query"]),
            "current_query": int(record["current_query"]),
        }
        cells = {
            "V00": normalized(paired["routes_previous"][paired_index]),
            "V10": normalized(modality["routes_vision_only"][modality_index]),
            "V01": normalized(modality["routes_state_only"][modality_index]),
            "V11": normalized(paired["routes_current"][paired_index]),
        }
        lookup: Dict[Tuple[str, str, str], Dict[str, float]] = {}
        for group_name, layers in groups.items():
            for flow_name, flows in flow_sets.items():
                for token_name, tokens in token_sets.items():
                    value = decompose(cells, layers, flows, tokens, epsilon)
                    lookup[(group_name, flow_name, token_name)] = value
                    group_rows.append(
                        {
                            **identity,
                            "layer_group": group_name,
                            "flow_range": flow_name,
                            "token_class": token_name,
                            **value,
                        }
                    )
        for stored_layer, model_layer in enumerate(config["hb_model_layers"]):
            for token_name, tokens in token_sets.items():
                value = decompose(
                    cells, [stored_layer], slice(None), tokens, epsilon
                )
                layer_rows.append(
                    {
                        **identity,
                        "stored_layer": stored_layer,
                        "model_layer": int(model_layer),
                        "layer_group": "front" if stored_layer < 4 else "back",
                        "token_class": token_name,
                        **value,
                    }
                )
        front_state = lookup[("front", "full", "state")]
        back_action = lookup[("back", "full", "action")]
        primary_rows.append(
            {
                **identity,
                "front_state_total_h": front_state["total_input_effect_h"],
                "front_state_vision_h": front_state["vision_marginal_effect_h"],
                "front_state_proprio_h": front_state["state_marginal_effect_h"],
                "front_state_vision_fraction": front_state[
                    "vision_fraction_of_marginal_effect"
                ],
                "back_action_total_h": back_action["total_input_effect_h"],
                "back_action_vision_h": back_action["vision_marginal_effect_h"],
                "back_action_proprio_h": back_action["state_marginal_effect_h"],
                "back_action_noise_h": back_action["noise_effect_h"],
                "back_action_total_to_noise": back_action[
                    "total_input_to_noise_ratio"
                ],
                "back_action_vision_to_noise": back_action["vision_to_noise_ratio"],
                "back_action_proprio_to_noise": back_action["state_to_noise_ratio"],
                "back_action_vision_fraction": back_action[
                    "vision_fraction_of_marginal_effect"
                ],
                "back_action_total_over_sum_marginal": back_action[
                    "total_over_sum_marginal_effect"
                ],
            }
        )
    return pd.DataFrame(group_rows), pd.DataFrame(layer_rows), pd.DataFrame(primary_rows)


def action_rms(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2, axis=(1, 2)))


def decompose_actions(cells: Mapping[str, np.ndarray]) -> Dict[str, float]:
    total = action_rms(cells["V00"], cells["V11"])
    vision = np.concatenate(
        [
            action_rms(cells["V00"], cells["V10"]),
            action_rms(cells["V01"], cells["V11"]),
        ]
    )
    state = np.concatenate(
        [
            action_rms(cells["V00"], cells["V01"]),
            action_rms(cells["V10"], cells["V11"]),
        ]
    )
    noise: List[float] = []
    for cell in cells.values():
        for first, second in combinations(range(cell.shape[0]), 2):
            noise.append(float(np.sqrt(np.mean((cell[first] - cell[second]) ** 2))))
    total_value = float(np.median(total))
    vision_value = float(np.median(vision))
    state_value = float(np.median(state))
    noise_value = float(np.median(noise))
    return {
        "action_total_input_rms": total_value,
        "action_vision_marginal_rms": vision_value,
        "action_state_marginal_rms": state_value,
        "action_noise_rms": noise_value,
        "action_input_to_noise_ratio": total_value / max(noise_value, 1e-12),
        "action_vision_to_noise_ratio": vision_value / max(noise_value, 1e-12),
        "action_state_to_noise_ratio": state_value / max(noise_value, 1e-12),
    }


def calculate_action_outputs(
    paired: Dict[str, np.ndarray],
    modality: Dict[str, np.ndarray],
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for modality_index, paired_index_value in enumerate(modality["paired_record_index"]):
        paired_index = int(paired_index_value)
        record = records[paired_index]
        cells = {
            "V00": paired["actions_previous"][paired_index],
            "V10": modality["actions_vision_only"][modality_index],
            "V01": modality["actions_state_only"][modality_index],
            "V11": paired["actions_current"][paired_index],
        }
        rows.append(
            {
                "paired_record_index": paired_index,
                "candidate": int(record["candidate"]),
                "episode_id": int(record["episode_id"]),
                "outcome": str(record["outcome"]),
                "relative_query": int(record["relative_query"]),
                **decompose_actions(cells),
            }
        )
    return pd.DataFrame(rows)


def add_policy_state_novelty(
    primary: pd.DataFrame,
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    cache: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for row in primary.itertuples():
        record = records[int(row.paired_record_index)]
        source_path = str(record["source_npz"])
        if source_path not in cache:
            with np.load(source_path, allow_pickle=False) as archive:
                cache[source_path] = np.asarray(archive["policy_state"], np.float32)
        state = cache[source_path]
        previous = state[int(row.previous_query)]
        current = state[int(row.current_query)]
        rows.append(
            {
                "paired_record_index": int(row.paired_record_index),
                "candidate": int(row.candidate),
                "outcome": str(row.outcome),
                "relative_query": int(row.relative_query),
                "policy_state_delta_l2": float(np.linalg.norm(current - previous)),
                "eef_position_delta_m": float(np.linalg.norm(current[:3] - previous[:3])),
                "eef_axis_angle_delta": float(np.linalg.norm(current[3:6] - previous[3:6])),
                "gripper_state_delta": float(np.linalg.norm(current[6:] - previous[6:])),
            }
        )
    return pd.DataFrame(rows)


def aligned_summary(primary: pd.DataFrame, novelty: pd.DataFrame) -> pd.DataFrame:
    merged = primary.merge(
        novelty.drop(columns=["candidate", "outcome", "relative_query"]),
        on="paired_record_index",
        validate="one_to_one",
    )
    metrics = [
        "front_state_vision_h",
        "front_state_proprio_h",
        "back_action_vision_h",
        "back_action_proprio_h",
        "back_action_vision_to_noise",
        "back_action_proprio_to_noise",
        "back_action_vision_fraction",
        "policy_state_delta_l2",
        "eef_position_delta_m",
        "gripper_state_delta",
    ]
    rows: List[Dict[str, Any]] = []
    for relative in sorted(merged.relative_query.unique()):
        subset = merged[merged.relative_query == relative]
        failure = subset[subset.outcome == "failed_grasp"]
        controls = subset[subset.outcome == "success"]
        for metric in metrics:
            failed_value = float(failure.iloc[0][metric])
            values = controls[metric].to_numpy(float)
            rows.append(
                {
                    "relative_query": int(relative),
                    "metric": metric,
                    "failed_value": failed_value,
                    "success_mean": float(values.mean()),
                    "success_min": float(values.min()),
                    "success_max": float(values.max()),
                    "failed_over_success_mean": failed_value / max(float(values.mean()), 1e-12),
                    "failed_rank_high_of_8": int(1 + np.sum(values < failed_value)),
                    "relation_to_success_range": (
                        "above_all"
                        if failed_value > values.max()
                        else "below_all"
                        if failed_value < values.min()
                        else "inside"
                    ),
                }
            )
    return pd.DataFrame(rows)


def action_aligned_summary(action: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "action_total_input_rms",
        "action_vision_marginal_rms",
        "action_state_marginal_rms",
        "action_noise_rms",
        "action_input_to_noise_ratio",
    ]
    rows: List[Dict[str, Any]] = []
    for relative in sorted(action.relative_query.unique()):
        failure = action[
            (action.relative_query == relative) & (action.outcome == "failed_grasp")
        ].iloc[0]
        controls = action[
            (action.relative_query == relative) & (action.outcome == "success")
        ]
        for metric in metrics:
            failed_value = float(failure[metric])
            values = controls[metric].to_numpy(float)
            rows.append(
                {
                    "relative_query": int(relative),
                    "metric": metric,
                    "failed_value": failed_value,
                    "success_mean": float(values.mean()),
                    "success_min": float(values.min()),
                    "success_max": float(values.max()),
                    "failed_rank_high_of_8": int(1 + np.sum(values < failed_value)),
                    "relation_to_success_range": (
                        "above_all"
                        if failed_value > values.max()
                        else "below_all"
                        if failed_value < values.min()
                        else "inside"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot(primary: pd.DataFrame, output: pathlib.Path) -> None:
    success = primary[primary.outcome == "success"]
    failure = primary[primary.outcome == "failed_grasp"]
    relatives = np.asarray(sorted(primary.relative_query.unique()), int)
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)

    def panel(ax: Any, metric: str, title: str, ylabel: str) -> None:
        grouped = success.groupby("relative_query")[metric]
        mean = grouped.mean().reindex(relatives).to_numpy(float)
        low = grouped.min().reindex(relatives).to_numpy(float)
        high = grouped.max().reindex(relatives).to_numpy(float)
        ax.fill_between(relatives, low, high, color="#78a6b8", alpha=0.25)
        ax.plot(relatives, mean, color="#267187", marker="o", label="success mean")
        values = failure.set_index("relative_query").reindex(relatives)[metric]
        ax.plot(relatives, values, color="#b43c3c", marker="o", linewidth=2, label="failed grasp")
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set(title=title, xlabel="Queries from gripper closure", ylabel=ylabel)

    panel(axes[0, 0], "back_action_vision_h", "Image-only effect on back action routes", "Hellinger")
    panel(axes[0, 1], "back_action_proprio_h", "8-D-state-only effect on back action routes", "Hellinger")
    panel(axes[1, 0], "front_state_vision_h", "Image-only effect on front state route", "Hellinger")
    panel(axes[1, 1], "back_action_vision_fraction", "Visual share of marginal action-route response", "Vision / (vision + state)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("2x2 input-modality counterfactual under matched flow noise", fontsize=12)
    figure.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(figure)


def plot_route_vs_action(
    primary: pd.DataFrame, action: pd.DataFrame, output: pathlib.Path
) -> None:
    merged = primary.merge(
        action[
            [
                "paired_record_index",
                "action_total_input_rms",
                "action_noise_rms",
                "action_input_to_noise_ratio",
            ]
        ],
        on="paired_record_index",
        validate="one_to_one",
    )
    success = merged[merged.outcome == "success"]
    failure = merged[merged.outcome == "failed_grasp"]
    relatives = np.asarray(sorted(merged.relative_query.unique()), int)
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    specifications = [
        (
            "back_action_total_to_noise",
            "MoE action-route input / noise effect",
        ),
        ("action_input_to_noise_ratio", "Action-output input / noise effect"),
    ]
    for ax, (metric, title) in zip(axes, specifications):
        grouped = success.groupby("relative_query")[metric]
        mean = grouped.mean().reindex(relatives).to_numpy(float)
        low = grouped.min().reindex(relatives).to_numpy(float)
        high = grouped.max().reindex(relatives).to_numpy(float)
        ax.fill_between(relatives, low, high, color="#78a6b8", alpha=0.25)
        ax.plot(relatives, mean, color="#267187", marker="o", label="success mean")
        values = failure.set_index("relative_query").reindex(relatives)[metric]
        ax.plot(relatives, values, color="#b43c3c", marker="o", linewidth=2, label="failed grasp")
        ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
        ax.set(title=title, xlabel="Queries from gripper closure")
    axes[0].legend(frameon=False, fontsize=8)
    figure.savefig(output, bbox_inches="tight", dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paired", type=pathlib.Path, default=DEFAULT_PAIRED)
    parser.add_argument("--modality", type=pathlib.Path, default=DEFAULT_MODALITY)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    paired_root = args.paired.expanduser().resolve()
    modality_root = args.modality.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = json.loads((paired_root / "records.json").read_text(encoding="utf-8"))[
        "records"
    ]
    paired_path = paired_root / "paired_counterfactual_routes.npz"
    modality_path = modality_root / "modality_counterfactual_routes.npz"
    with np.load(str(paired_path), allow_pickle=False) as source:
        paired = {name: np.asarray(source[name]) for name in source.files}
    with np.load(str(modality_path), allow_pickle=False) as source:
        modality = {name: np.asarray(source[name]) for name in source.files}
    if str(modality["schema"].item()) != "himoe.input_modality_counterfactual.capture.v1":
        raise RuntimeError("unsupported modality capture")
    if bool(modality["training"].item()):
        raise RuntimeError("modality capture unexpectedly declares training")

    group, layer, primary = calculate(paired, modality, records, config)
    action = calculate_action_outputs(paired, modality, records)
    novelty = add_policy_state_novelty(primary, records)
    aligned = aligned_summary(primary, novelty)
    action_aligned = action_aligned_summary(action)
    group.to_csv(tables / "modality_group_metrics.csv", index=False)
    layer.to_csv(tables / "modality_layer_metrics.csv", index=False)
    primary.to_csv(tables / "modality_primary_metrics.csv", index=False)
    action.to_csv(tables / "action_output_counterfactual.csv", index=False)
    novelty.to_csv(tables / "policy_state_novelty_posthoc.csv", index=False)
    aligned.to_csv(tables / "modality_failure_vs_success.csv", index=False)
    action_aligned.to_csv(tables / "action_output_failure_vs_success.csv", index=False)
    plot(primary, figures / "input_modality_decomposition.png")
    plot_route_vs_action(
        primary, action, figures / "route_vs_action_counterfactual.png"
    )

    failure = primary[primary.outcome == "failed_grasp"].set_index("relative_query")
    action_failure = action[action.outcome == "failed_grasp"].set_index(
        "relative_query"
    )
    controls = primary[primary.outcome == "success"]
    timeline: Dict[str, Any] = {}
    for relative in sorted(primary.relative_query.unique()):
        row = failure.loc[int(relative)]
        action_row = action_failure.loc[int(relative)]
        control = controls[controls.relative_query == relative]
        timeline[str(int(relative))] = {
            "front_state_vision_h": float(row.front_state_vision_h),
            "front_state_proprio_h": float(row.front_state_proprio_h),
            "back_action_vision_h": float(row.back_action_vision_h),
            "back_action_proprio_h": float(row.back_action_proprio_h),
            "back_action_vision_fraction": float(row.back_action_vision_fraction),
            "back_action_vision_relation": aligned[
                (aligned.relative_query == relative)
                & (aligned.metric == "back_action_vision_h")
            ].relation_to_success_range.iloc[0],
            "back_action_proprio_relation": aligned[
                (aligned.relative_query == relative)
                & (aligned.metric == "back_action_proprio_h")
            ].relation_to_success_range.iloc[0],
            "success_back_action_vision_range": [
                float(control.back_action_vision_h.min()),
                float(control.back_action_vision_h.max()),
            ],
            "success_back_action_proprio_range": [
                float(control.back_action_proprio_h.min()),
                float(control.back_action_proprio_h.max()),
            ],
            "action_output": {
                "total_input_rms": float(action_row.action_total_input_rms),
                "vision_marginal_rms": float(action_row.action_vision_marginal_rms),
                "state_marginal_rms": float(action_row.action_state_marginal_rms),
                "noise_rms": float(action_row.action_noise_rms),
                "input_to_noise_ratio": float(
                    action_row.action_input_to_noise_ratio
                ),
                "total_input_relation": action_aligned[
                    (action_aligned.relative_query == relative)
                    & (action_aligned.metric == "action_total_input_rms")
                ].relation_to_success_range.iloc[0],
                "noise_relation": action_aligned[
                    (action_aligned.relative_query == relative)
                    & (action_aligned.metric == "action_noise_rms")
                ].relation_to_success_range.iloc[0],
            },
        }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "score_uses_physics": False,
        "score_uses_outcome": False,
        "score_uses_normal_trajectory_bank": False,
        "factorial_design": {
            "V00": "old images + old 8-D state",
            "V10": "new images + old 8-D state",
            "V01": "old images + new 8-D state",
            "V11": "new images + new 8-D state",
            "matched_flow_noise": True,
        },
        "records": int(len(primary)),
        "failed_records": int((primary.outcome == "failed_grasp").sum()),
        "success_control_records": int((primary.outcome == "success").sum()),
        "relative_queries": sorted(map(int, primary.relative_query.unique())),
        "failed_timeline": timeline,
        "limitations": [
            "Image/state hybrids are slightly off-manifold, although only adjacent replanning inputs are mixed.",
            "This is one failed-grasp trajectory with seven same-snapshot success controls.",
            "Physical and outcome fields are used only after route metrics are written, for interpretation.",
        ],
        "inputs": {
            "paired_capture": str(paired_path),
            "paired_capture_sha256": file_digest(paired_path),
            "modality_capture": str(modality_path),
            "modality_capture_sha256": file_digest(modality_path),
            "config": str(config_path),
            "config_sha256": file_digest(config_path),
        },
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
