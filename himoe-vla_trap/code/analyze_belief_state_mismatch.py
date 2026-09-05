#!/usr/bin/env python3
"""Train-free belief/physics mismatch analysis for the failed-grasp case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

import analyze_failed_grasp_moe_dynamics as base


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/belief_state_mismatch.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/belief_state_mismatch"
HB_MODEL_LAYERS = (2, 3, 4, 5, 12, 13, 14, 15)
GROUPS = {"front": tuple(range(4)), "back": tuple(range(4, 8))}


def relation(value: float, controls: np.ndarray) -> str:
    if value > float(np.max(controls)):
        return "above_all"
    if value < float(np.min(controls)):
        return "below_all"
    return "inside"


def control_comparison(value: float, controls: np.ndarray) -> dict[str, Any]:
    controls = np.asarray(controls, np.float64)
    mean = float(controls.mean())
    return {
        "failed_value": float(value),
        "success_mean": mean,
        "success_min": float(controls.min()),
        "success_max": float(controls.max()),
        "failed_over_success_mean": float(value / mean) if mean else np.nan,
        "relation_to_success_range": relation(value, controls),
    }


def loo_distances(values: np.ndarray, distance) -> np.ndarray:
    return np.asarray([
        float(distance(values[index], np.delete(values, index, axis=0).mean(axis=0)))
        for index in range(len(values))
    ])


def route_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(base.hellinger(a, b).mean())


def action_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a - b))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def physical_alignment(
    branches: list[base.Branch], low: int, high: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for branch in branches:
        q0 = branch.event.query
        initial_eef = branch.arrays["policy_state"][q0, :3]
        initial_pot = branch.arrays["sim_state"][q0, 10:13]
        for relative in range(low, high + 1):
            query = q0 + relative
            if not 0 <= query < len(branch.arrays["sim_state"]):
                continue
            eef = branch.arrays["policy_state"][query, :3]
            pot = branch.arrays["sim_state"][query, 10:13]
            eef_displacement = float(np.linalg.norm(eef - initial_eef))
            pot_displacement = float(np.linalg.norm(pot - initial_pot))
            action = branch.arrays["action_chunks"][query]
            rows.append({
                "candidate": branch.candidate,
                "episode_id": branch.episode_id,
                "outcome": "success" if branch.success else "belief_trap",
                "relative_query": relative,
                "query": query,
                "eef_displacement_from_event_m": eef_displacement,
                "pot_displacement_from_event_m": pot_displacement,
                "eef_pot_separation_m": float(np.linalg.norm(eef - pot)),
                "coupling_ratio_pot_over_eef": pot_displacement / max(eef_displacement, 1e-6),
                "gripper_command_mean": float(action[:, 6].mean()),
            })
    frame = pd.DataFrame(rows)
    summaries = []
    for relative in sorted(frame.relative_query.unique()):
        subset = frame[frame.relative_query == relative]
        failed = subset[subset.outcome == "belief_trap"].iloc[0]
        controls = subset[subset.outcome == "success"]
        for metric in (
            "eef_displacement_from_event_m",
            "pot_displacement_from_event_m",
            "eef_pot_separation_m",
            "coupling_ratio_pot_over_eef",
            "gripper_command_mean",
        ):
            summaries.append({
                "relative_query": relative,
                "metric": metric,
                **control_comparison(float(failed[metric]), controls[metric].to_numpy(float)),
            })
    return frame, pd.DataFrame(summaries)


def token_route_comparison(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows = []
    assert failed.route is not None
    for relative in range(low, high + 1):
        failed_query = failed.event.query + relative
        if not 0 <= failed_query < len(failed.route):
            continue
        for group_name, indices in GROUPS.items():
            for token_type in ("state", "action"):
                def select(branch: base.Branch) -> np.ndarray:
                    assert branch.route is not None
                    value = branch.route[branch.event.query + relative, indices, -1]
                    return value[:, 0] if token_type == "state" else value[:, 1:11]

                control_values = np.stack([select(branch) for branch in controls])
                failed_value = select(failed)
                failed_distance = route_distance(failed_value, control_values.mean(axis=0))
                loo = loo_distances(control_values, route_distance)
                rows.append({
                    "relative_query": relative,
                    "layer_group": group_name,
                    "token_type": token_type,
                    **control_comparison(failed_distance, loo),
                })
    return pd.DataFrame(rows)


def flow_token_route_comparison(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows = []
    assert failed.route is not None
    flow_steps = failed.route.shape[2]
    for relative in range(low, high + 1):
        for flow in range(flow_steps):
            for group_name, indices in GROUPS.items():
                for token_type in ("state", "action"):
                    def select(branch: base.Branch) -> np.ndarray:
                        assert branch.route is not None
                        value = branch.route[
                            branch.event.query + relative, indices, flow
                        ]
                        return value[:, 0] if token_type == "state" else value[:, 1:11]

                    control_values = np.stack([select(branch) for branch in controls])
                    failed_value = select(failed)
                    failed_distance = route_distance(
                        failed_value, control_values.mean(axis=0)
                    )
                    loo = loo_distances(control_values, route_distance)
                    rows.append({
                        "relative_query": relative,
                        "flow_step": flow,
                        "layer_group": group_name,
                        "token_type": token_type,
                        **control_comparison(failed_distance, loo),
                    })
    return pd.DataFrame(rows)


def flow_state_action_gap(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows = []
    assert failed.route is not None
    flow_steps = failed.route.shape[2]
    for relative in range(low, high + 1):
        for flow in range(flow_steps):
            for group_name, indices in GROUPS.items():
                values = []
                for branch in [failed, *controls]:
                    assert branch.route is not None
                    p = branch.route[
                        branch.event.query + relative, indices, flow
                    ]
                    state = p[:, 0]
                    action = base.normalized(p[:, 1:11].mean(axis=1))
                    values.append(float(base.hellinger(state, action).mean()))
                rows.append({
                    "relative_query": relative,
                    "flow_step": flow,
                    "layer_group": group_name,
                    **control_comparison(values[0], np.asarray(values[1:])),
                })
    return pd.DataFrame(rows)


def action_position_route_comparison(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows = []
    assert failed.route is not None
    action_positions = failed.route.shape[3] - 1
    for relative in range(low, high + 1):
        for group_name, indices in GROUPS.items():
            for position in range(action_positions):
                def select(branch: base.Branch) -> np.ndarray:
                    assert branch.route is not None
                    return branch.route[
                        branch.event.query + relative, indices, :, position + 1
                    ]

                control_values = np.stack([select(branch) for branch in controls])
                failed_value = select(failed)
                failed_distance = route_distance(
                    failed_value, control_values.mean(axis=0)
                )
                loo = loo_distances(control_values, route_distance)
                rows.append({
                    "relative_query": relative,
                    "layer_group": group_name,
                    "action_position_1based": position + 1,
                    **control_comparison(failed_distance, loo),
                })
    return pd.DataFrame(rows)


def state_action_gap(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    layer_rows = []
    group_rows = []
    assert failed.route is not None
    for relative in range(low, high + 1):
        for stored_layer, model_layer in enumerate(HB_MODEL_LAYERS):
            values = []
            for branch in [failed, *controls]:
                assert branch.route is not None
                p = branch.route[branch.event.query + relative, stored_layer, -1]
                state = p[0]
                action = base.normalized(p[1:11].mean(axis=0))
                values.append(float(base.hellinger(state, action)))
            layer_rows.append({
                "relative_query": relative,
                "stored_layer_index": stored_layer,
                "model_layer": model_layer,
                "layer_group": "front" if stored_layer < 4 else "back",
                **control_comparison(values[0], np.asarray(values[1:])),
            })
        for group_name, indices in GROUPS.items():
            values = []
            for branch in [failed, *controls]:
                assert branch.route is not None
                p = branch.route[branch.event.query + relative, indices, -1]
                state = p[:, 0]
                action = base.normalized(p[:, 1:11].mean(axis=1))
                values.append(float(base.hellinger(state, action).mean()))
            group_rows.append({
                "relative_query": relative,
                "layer_group": group_name,
                **control_comparison(values[0], np.asarray(values[1:])),
            })
    return pd.DataFrame(layer_rows), pd.DataFrame(group_rows)


def action_phase_analysis(
    failed: base.Branch,
    controls: list[base.Branch],
    action_std: np.ndarray,
    low: int,
    high: int,
    phase_low: int,
    phase_high: int,
) -> pd.DataFrame:
    phase_centers: dict[int, np.ndarray] = {}
    for phase in range(phase_low, phase_high + 1):
        values = [
            branch.arrays["action_chunks"][branch.event.query + phase] / action_std
            for branch in controls
            if 0 <= branch.event.query + phase < len(branch.arrays["action_chunks"])
        ]
        if len(values) >= 4:
            phase_centers[phase] = np.mean(values, axis=0)

    rows = []
    for relative in range(low, high + 1):
        failed_query = failed.event.query + relative
        if not 0 <= failed_query < len(failed.arrays["action_chunks"]):
            continue
        failed_action = failed.arrays["action_chunks"][failed_query] / action_std
        controls_at_phase = np.stack([
            branch.arrays["action_chunks"][branch.event.query + relative] / action_std
            for branch in controls
        ])
        same_distance = action_distance(failed_action, controls_at_phase.mean(axis=0))
        loo = loo_distances(controls_at_phase, action_distance)
        phase_distances = {
            phase: action_distance(failed_action, center)
            for phase, center in phase_centers.items()
        }
        nearest_phase = min(phase_distances, key=phase_distances.get)
        center = controls_at_phase.mean(axis=0)
        rows.append({
            "relative_query": relative,
            **control_comparison(same_distance, loo),
            "cosine_to_same_phase_success_center": cosine(failed_action, center),
            "nearest_success_phase": nearest_phase,
            "nearest_success_phase_distance": phase_distances[nearest_phase],
            "phase_advance": nearest_phase - relative,
        })
    return pd.DataFrame(rows)


def route_phase_analysis(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
    phase_low: int,
    phase_high: int,
) -> pd.DataFrame:
    assert failed.route is not None
    centers: dict[tuple[str, str, int], np.ndarray] = {}
    for group_name, indices in GROUPS.items():
        for token_type in ("state", "action"):
            for phase in range(phase_low, phase_high + 1):
                values = []
                for branch in controls:
                    assert branch.route is not None
                    query = branch.event.query + phase
                    if not 0 <= query < len(branch.route):
                        continue
                    p = branch.route[query, indices, -1]
                    values.append(p[:, 0] if token_type == "state" else p[:, 1:11])
                if len(values) >= 4:
                    centers[(group_name, token_type, phase)] = np.mean(values, axis=0)

    rows = []
    for relative in range(low, high + 1):
        query = failed.event.query + relative
        for group_name, indices in GROUPS.items():
            for token_type in ("state", "action"):
                p = failed.route[query, indices, -1]
                value = p[:, 0] if token_type == "state" else p[:, 1:11]
                distances = {
                    phase: route_distance(value, center)
                    for (group, token, phase), center in centers.items()
                    if group == group_name and token == token_type
                }
                nearest = min(distances, key=distances.get)
                rows.append({
                    "relative_query": relative,
                    "layer_group": group_name,
                    "token_type": token_type,
                    "nearest_success_phase": nearest,
                    "nearest_success_phase_distance": distances[nearest],
                    "phase_advance": nearest - relative,
                    "same_phase_distance": distances.get(relative, np.nan),
                })
    return pd.DataFrame(rows)


def cross_chunk_response(
    failed: base.Branch,
    controls: list[base.Branch],
    low: int,
    high: int,
) -> pd.DataFrame:
    rows = []
    for relative in range(low, high + 1):
        for group_name, indices in GROUPS.items():
            values = []
            for branch in [failed, *controls]:
                assert branch.route is not None
                query = branch.event.query + relative
                current = branch.route[query, indices, :, 1:11]
                previous = branch.route[query - 1, indices, :, 1:11]
                values.append(route_distance(current, previous))
            rows.append({
                "relative_query": relative,
                "layer_group": group_name,
                **control_comparison(values[0], np.asarray(values[1:])),
            })
    return pd.DataFrame(rows)


def external_evidence() -> dict[str, Any]:
    adaptation_path = (
        WORKSPACE_ROOT
        / "himoe-route-capture/analysis/post-error-adaptation-20260828/summary.json"
    )
    recovery_path = (
        WORKSPACE_ROOT / "himoe-route-capture/analysis/post-error-recovery-hub/summary.json"
    )
    adaptation = json.loads(adaptation_path.read_text(encoding="utf-8"))
    run = next(
        value for value in adaptation["ladder"]["runs"]
        if value["role"] == "setback_landmark"
    )
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    recurrence = recovery["descriptive_recurrence"]["failed_chunk"]
    return {
        "grasp_fail_ladder": {
            "global_gate": adaptation["gate"]["gate"],
            "status": run["status"],
            "n_used": run["n_used"],
            "n_init_clusters": run["n_clusters"],
            "route_action_minus_physical_action_auc": run["increments"][
                "M_phys+route_action_minus_M_phys+action"
            ],
        },
        "post_loss_recurrence": {
            "global_gate": recovery["gate0"]["status"],
            "terminal_success_n": recurrence["terminal_success_after_proxy"]["n"],
            "terminal_failure_n": recurrence["terminal_failure_after_proxy"]["n"],
            "terminal_success_action_route_similarity": recurrence[
                "terminal_success_after_proxy"
            ]["metrics"]["route_action_hellinger_similarity"]["mean"],
            "terminal_failure_action_route_similarity": recurrence[
                "terminal_failure_after_proxy"
            ]["metrics"]["route_action_hellinger_similarity"]["mean"],
        },
    }


def record_at(frame: pd.DataFrame, **selectors: Any) -> dict[str, Any]:
    subset = frame
    for key, value in selectors.items():
        subset = subset[subset[key] == value]
    if len(subset) != 1:
        raise RuntimeError(f"expected one row for {selectors}, found {len(subset)}")
    return {key: base.jsonable(value) for key, value in subset.iloc[0].to_dict().items()}


def anomaly_axis_profile(
    frame: pd.DataFrame,
    axis: str,
    **selectors: Any,
) -> dict[str, Any]:
    subset = frame
    for key, value in selectors.items():
        subset = subset[subset[key] == value]
    subset = subset.sort_values(axis).copy()
    subset["ratio_to_success_max"] = subset.failed_value / subset.success_max
    peak = subset.loc[subset.ratio_to_success_max.idxmax()]
    above = subset[
        subset.relation_to_success_range == "above_all"
    ][axis].astype(int).tolist()
    return {
        "above_success_range": above,
        "n_above": len(above),
        "n_total": int(len(subset)),
        "peak_axis_value": int(peak[axis]),
        "peak_failed_over_success_max": float(peak.ratio_to_success_max),
    }


def plot_summary(
    physical: pd.DataFrame,
    token_routes: pd.DataFrame,
    group_gaps: pd.DataFrame,
    action_phase: pd.DataFrame,
    figure_path: Path,
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    failed = physical[physical.outcome == "belief_trap"]
    success = physical[physical.outcome == "success"]
    for metric, color, label in [
        ("eef_displacement_from_event_m", "#b43c3c", "EEF: failed"),
        ("pot_displacement_from_event_m", "#7b2cbf", "Pot: failed"),
    ]:
        ax.plot(failed.relative_query, failed[metric] * 100, color=color, linewidth=2, label=label)
    for metric, color, label in [
        ("eef_displacement_from_event_m", "#276678", "EEF: success mean"),
        ("pot_displacement_from_event_m", "#4a8f58", "Pot: success mean"),
    ]:
        mean = success.groupby("relative_query")[metric].mean()
        ax.plot(mean.index, mean.values * 100, color=color, linestyle="--", label=label)
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set(title="Belief/physics decoupling", xlabel="Queries from failed closure", ylabel="Displacement from closure (cm)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    ax.fill_between(
        action_phase.relative_query,
        action_phase.success_min,
        action_phase.success_max,
        color="#8db3c7",
        alpha=0.3,
        label="Success leave-one-out range",
    )
    ax.plot(action_phase.relative_query, action_phase.failed_value, color="#b43c3c", linewidth=2, label="Failed action to success center")
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set(title="Action plan returns to nominal transport", xlabel="Queries from failed closure", ylabel="Normalized action RMS")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    styles = {
        ("front", "state"): ("#247ba0", "-"),
        ("front", "action"): ("#247ba0", "--"),
        ("back", "state"): ("#7b2cbf", "-"),
        ("back", "action"): ("#7b2cbf", "--"),
    }
    for (group, token), (color, line) in styles.items():
        subset = token_routes[(token_routes.layer_group == group) & (token_routes.token_type == token)]
        ratio = subset.failed_value / subset.success_max
        ax.plot(subset.relative_query, ratio, color=color, linestyle=line, marker="o", markersize=3, label=f"{group} {token}")
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set(title="Token routes outside successful phase", xlabel="Queries from failed closure", ylabel="Failed distance / success maximum")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1, 1]
    for group, color in [("front", "#247ba0"), ("back", "#7b2cbf")]:
        subset = group_gaps[group_gaps.layer_group == group]
        ax.fill_between(subset.relative_query, subset.success_min, subset.success_max, color=color, alpha=0.16)
        ax.plot(subset.relative_query, subset.failed_value, color=color, linewidth=2, label=f"{group} failed")
        ax.plot(subset.relative_query, subset.success_mean, color=color, linestyle="--", alpha=0.7, label=f"{group} success mean")
    ax.axvline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set(title="HB state/action routing gap", xlabel="Queries from failed closure", ylabel="Hellinger gap")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def plot_flow_position_summary(
    flow_routes: pd.DataFrame,
    flow_gaps: pd.DataFrame,
    positions: pd.DataFrame,
    figure_path: Path,
) -> None:
    relative = 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    ax = axes[0]
    styles = {
        ("front", "state"): ("#247ba0", "-"),
        ("front", "action"): ("#247ba0", "--"),
        ("back", "state"): ("#7b2cbf", "-"),
        ("back", "action"): ("#7b2cbf", "--"),
    }
    for (group, token), (color, line) in styles.items():
        subset = flow_routes[
            (flow_routes.relative_query == relative)
            & (flow_routes.layer_group == group)
            & (flow_routes.token_type == token)
        ]
        ratio = subset.failed_value / subset.success_max
        ax.plot(
            subset.flow_step, ratio, color=color, linestyle=line, marker="o",
            markersize=3, label=f"{group} {token}",
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set(
        title="q+2 route deviation over flow",
        xlabel="Flow step",
        ylabel="Failed distance / success maximum",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    for group, color in [("front", "#247ba0"), ("back", "#7b2cbf")]:
        subset = flow_gaps[
            (flow_gaps.relative_query == relative)
            & (flow_gaps.layer_group == group)
        ]
        ax.fill_between(
            subset.flow_step.to_numpy(), subset.success_min.to_numpy(),
            subset.success_max.to_numpy(), color=color, alpha=0.15,
        )
        ax.plot(
            subset.flow_step, subset.failed_value, color=color, marker="o",
            markersize=3, label=f"{group} failed",
        )
        ax.plot(
            subset.flow_step, subset.success_mean, color=color, linestyle="--",
            alpha=0.7, label=f"{group} success mean",
        )
    ax.set(
        title="q+2 state/action gap over flow",
        xlabel="Flow step",
        ylabel="Hellinger gap",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[2]
    for group, color in [("front", "#247ba0"), ("back", "#7b2cbf")]:
        subset = positions[
            (positions.relative_query == relative)
            & (positions.layer_group == group)
        ]
        ratio = subset.failed_value / subset.success_max
        ax.plot(
            subset.action_position_1based, ratio, color=color, marker="o",
            markersize=3, label=group,
        )
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set(
        title="q+2 action-position route deviation",
        xlabel="Action position in chunk",
        ylabel="Failed distance / success maximum",
        xticks=range(1, 11),
    )
    ax.legend(frameon=False, fontsize=8)

    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    event_config_path = PACKAGE_ROOT / config["base_event_config"]
    event_config = json.loads(event_config_path.read_text(encoding="utf-8"))
    run_root = (WORKSPACE_ROOT / event_config["run_root"]).resolve()
    snapshot_dir = run_root / "formal" / f"worker{event_config['worker']}" / f"snapshot_{event_config['snapshot']:03d}"
    store_path = run_root / "formal/server/routes.zarr"
    failed, controls = base.load_branches(event_config, snapshot_dir)
    store = zarr.open_group(str(store_path), mode="r")
    base.attach_routes([failed, *controls], store)

    low, high = map(int, config["relative_query_range"])
    phase_low, phase_high = map(int, config["success_phase_range"])
    action_std = np.asarray(config["action_std"], np.float32)
    output = args.output.resolve()
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    physical, physical_summary = physical_alignment([failed, *controls], low, high)
    token_routes = token_route_comparison(failed, controls, low, high)
    flow_token_routes = flow_token_route_comparison(failed, controls, low, high)
    flow_gaps = flow_state_action_gap(failed, controls, low, high)
    action_positions = action_position_route_comparison(
        failed, controls, low, high
    )
    layer_gaps, group_gaps = state_action_gap(failed, controls, low, high)
    action_phase = action_phase_analysis(
        failed, controls, action_std, low, high, phase_low, phase_high
    )
    route_phase = route_phase_analysis(
        failed, controls, low, high, phase_low, phase_high
    )
    chunk_response = cross_chunk_response(failed, controls, low, high)

    tables = {
        "belief_physics_alignment.csv": physical,
        "belief_physics_summary.csv": physical_summary,
        "hb_token_match_to_success.csv": token_routes,
        "hb_flow_token_match_to_success.csv": flow_token_routes,
        "hb_flow_state_action_gap.csv": flow_gaps,
        "hb_action_position_match_to_success.csv": action_positions,
        "hb_state_action_gap_layer.csv": layer_gaps,
        "hb_state_action_gap_group.csv": group_gaps,
        "action_phase_consistency.csv": action_phase,
        "hb_phase_tracking.csv": route_phase,
        "hb_cross_chunk_response.csv": chunk_response,
    }
    for name, frame in tables.items():
        frame.to_csv(table_dir / name, index=False)

    plot_summary(
        physical, token_routes, group_gaps, action_phase,
        figure_dir / "belief_state_mismatch_signature.png",
    )
    plot_flow_position_summary(
        flow_token_routes, flow_gaps, action_positions,
        figure_dir / "belief_state_mismatch_flow_position.png",
    )

    mismatch_low, mismatch_high = map(int, config["post_mismatch_window"])
    summary = {
        "schema": "himoe.belief_state_mismatch.v1",
        "training": False,
        "case": {
            "failed_candidate": failed.candidate,
            "episode_id": failed.episode_id,
            "failed_closure_query": failed.event.query,
            "matched_success_controls": len(controls),
            "belief_state_is_behavioral_inference": True,
        },
        "operational_definition": {
            "physics": "EEF continues transport while target pot does not remain coupled",
            "policy": "action chunks return to the matched successful transport-phase range",
            "moe": "state/action token routing separates relative to same-snapshot successful phases",
        },
        "first_clear_mismatch_query": 2,
        "query_plus_2": {
            "physics_eef_displacement": record_at(
                physical_summary, relative_query=2, metric="eef_displacement_from_event_m"
            ),
            "physics_pot_displacement": record_at(
                physical_summary, relative_query=2, metric="pot_displacement_from_event_m"
            ),
            "action_phase": record_at(action_phase, relative_query=2),
            "front_state_route": record_at(
                token_routes, relative_query=2, layer_group="front", token_type="state"
            ),
            "front_action_route": record_at(
                token_routes, relative_query=2, layer_group="front", token_type="action"
            ),
            "back_state_route": record_at(
                token_routes, relative_query=2, layer_group="back", token_type="state"
            ),
            "back_action_route": record_at(
                token_routes, relative_query=2, layer_group="back", token_type="action"
            ),
            "front_state_action_gap": record_at(
                group_gaps, relative_query=2, layer_group="front"
            ),
        },
        "query_plus_2_axis_profiles": {
            "flow_front_state": anomaly_axis_profile(
                flow_token_routes, "flow_step", relative_query=2,
                layer_group="front", token_type="state",
            ),
            "flow_front_action": anomaly_axis_profile(
                flow_token_routes, "flow_step", relative_query=2,
                layer_group="front", token_type="action",
            ),
            "flow_back_state": anomaly_axis_profile(
                flow_token_routes, "flow_step", relative_query=2,
                layer_group="back", token_type="state",
            ),
            "flow_back_action": anomaly_axis_profile(
                flow_token_routes, "flow_step", relative_query=2,
                layer_group="back", token_type="action",
            ),
            "flow_front_state_action_gap": anomaly_axis_profile(
                flow_gaps, "flow_step", relative_query=2,
                layer_group="front",
            ),
            "flow_back_state_action_gap": anomaly_axis_profile(
                flow_gaps, "flow_step", relative_query=2,
                layer_group="back",
            ),
            "position_front_action": anomaly_axis_profile(
                action_positions, "action_position_1based", relative_query=2,
                layer_group="front",
            ),
            "position_back_action": anomaly_axis_profile(
                action_positions, "action_position_1based", relative_query=2,
                layer_group="back",
            ),
        },
        "persistent_patterns": {
            "action_inside_success_range": action_phase[
                action_phase.relation_to_success_range == "inside"
            ].relative_query.tolist(),
            "front_state_action_gap_above_all": group_gaps[
                (group_gaps.layer_group == "front")
                & (group_gaps.relation_to_success_range == "above_all")
            ].relative_query.tolist(),
            "back_state_route_above_all": token_routes[
                (token_routes.layer_group == "back")
                & (token_routes.token_type == "state")
                & (token_routes.relation_to_success_range == "above_all")
            ].relative_query.tolist(),
            "back_action_route_above_all": token_routes[
                (token_routes.layer_group == "back")
                & (token_routes.token_type == "action")
                & (token_routes.relation_to_success_range == "above_all")
            ].relative_query.tolist(),
            "front_gap_layers_above_all_in_post_window": sorted(
                layer_gaps[
                    layer_gaps.relative_query.between(mismatch_low, mismatch_high)
                    & (layer_gaps.layer_group == "front")
                    & (layer_gaps.relation_to_success_range == "above_all")
                ].model_layer.unique().tolist()
            ),
            "front_gap_above_all_queries_by_layer": {
                str(model_layer): layer_gaps[
                    (layer_gaps.model_layer == model_layer)
                    & (layer_gaps.relative_query.between(mismatch_low, high))
                    & (layer_gaps.relation_to_success_range == "above_all")
                ].relative_query.tolist()
                for model_layer in (2, 3, 4, 5)
            },
        },
        "delayed_response": {
            "front_chunk_jump_plus_1": record_at(
                chunk_response, relative_query=1, layer_group="front"
            ),
            "back_chunk_jump_plus_1": record_at(
                chunk_response, relative_query=1, layer_group="back"
            ),
            "back_chunk_jump_plus_2": record_at(
                chunk_response, relative_query=2, layer_group="back"
            ),
            "back_chunk_jump_plus_3": record_at(
                chunk_response, relative_query=3, layer_group="back"
            ),
        },
        "external_evidence_audit": external_evidence(),
        "interpretation": (
            "The state-facing route changes, but the action plan resumes the nominal "
            "transport phase. This supports stale action belief / state-action update "
            "failure more specifically than a claim that the whole network is unaware."
        ),
        "limitations": [
            "Single failed-grasp case; ranges over seven matched successes are descriptive, not population confidence intervals.",
            "No contact sensor or explicit model belief variable was captured; belief state is inferred from physical/action divergence.",
            "HB action probabilities are close to uniform, so conclusions use soft distances rather than hard expert IDs.",
            "Existing broader grasp-failure analyses failed their support gate and do not establish generalization.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(base.jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "OK",
        "failed_candidate": failed.candidate,
        "first_clear_mismatch_query": 2,
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
