#!/usr/bin/env python3
"""Audit AS/HB MoE dynamics around one failed grasp without training a probe.

The failed event is aligned to successful pot-grasp events from sibling
rollouts of the same simulator snapshot.  Labels select the case and controls;
all reported quantities are direct routing distances or physical measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/failed_grasp_moe_dynamics.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/failed_grasp_moe_dynamics"


@dataclass
class GraspEvent:
    control_step: int
    query: int
    action_position_1based: int
    eef_object_distance_m: float
    aperture_before: float
    aperture_after: float
    subsequent_max_lift_m: float
    lift_control_step: int | None
    lift_query: int | None


@dataclass
class Branch:
    candidate: int
    episode_id: int
    success: bool
    arrays: dict[str, np.ndarray]
    event: GraspEvent
    route: np.ndarray | None = None


def normalized(p: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(p, np.float32), 0.0)
    return p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-12)


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalized(p)
    q = normalized(q)
    coefficient = np.sqrt(p * q).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - coefficient, 0.0, 1.0))


def weighted_jaccard_distance(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = normalized(p)
    q = normalized(q)
    numerator = np.minimum(p, q).sum(axis=-1)
    denominator = np.maximum(p, q).sum(axis=-1)
    return 1.0 - numerator / np.maximum(denominator, 1e-12)


def entropy(p: np.ndarray) -> np.ndarray:
    p = normalized(p)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1) / np.log(p.shape[-1])


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def find_grasp_event(
    arrays: dict[str, np.ndarray],
    pot_slice: slice,
    close_threshold: float,
    near_threshold: float,
    lift_threshold: float,
    require_lift: bool,
) -> GraspEvent | None:
    sim = arrays["control_sim_state"]
    eef = arrays["control_eef_position"]
    grip = arrays["control_gripper_qpos"]
    query_index = arrays["control_query_index"]
    aperture = np.abs(grip).sum(axis=1)
    crossings = np.flatnonzero(
        (aperture[:-1] >= close_threshold) & (aperture[1:] < close_threshold)
    ) + 1

    possible: list[GraspEvent] = []
    for step in crossings:
        if step >= len(query_index):
            continue
        object_xyz = sim[step, pot_slice]
        distance = float(np.linalg.norm(eef[step] - object_xyz))
        if distance >= near_threshold:
            continue
        lift = sim[step:, pot_slice.stop - 1] - object_xyz[-1]
        max_lift = float(np.max(lift))
        lift_candidates = np.flatnonzero(lift >= lift_threshold)
        lift_step = int(step + lift_candidates[0]) if len(lift_candidates) else None
        lift_query = (
            int(query_index[max(lift_step - 1, 0)])
            if lift_step is not None
            else None
        )
        if require_lift and lift_step is None:
            continue
        # Dense state index `step` is the state after action `step - 1`.
        # Map the aperture crossing to the action that caused it.
        action_index = int(step - 1)
        query = int(query_index[action_index])
        query_steps = np.flatnonzero(query_index == query)
        position = int(np.flatnonzero(query_steps == action_index)[0]) + 1
        possible.append(
            GraspEvent(
                control_step=int(step),
                query=query,
                action_position_1based=position,
                eef_object_distance_m=distance,
                aperture_before=float(aperture[step - 1]),
                aperture_after=float(aperture[step]),
                subsequent_max_lift_m=max_lift,
                lift_control_step=lift_step,
                lift_query=lift_query,
            )
        )
    if not possible:
        return None
    return possible[0] if require_lift else possible[-1]


def load_branches(config: dict[str, Any], snapshot_dir: Path) -> tuple[Branch, list[Branch]]:
    pot_slice = slice(*config["pot_state_slice"])
    failed_candidate = int(config["failed_candidate"])
    failed: Branch | None = None
    controls: list[Branch] = []
    for metadata_path in sorted(snapshot_dir.glob("candidate_*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        candidate = int(metadata["candidate"])
        success = bool(metadata["success"])
        if candidate != failed_candidate and not success:
            continue
        arrays = load_archive(metadata_path.with_suffix(".npz"))
        event = find_grasp_event(
            arrays,
            pot_slice=pot_slice,
            close_threshold=float(config["close_aperture_threshold"]),
            near_threshold=float(config["near_object_threshold_m"]),
            lift_threshold=float(config["lift_threshold_m"]),
            require_lift=success,
        )
        if event is None:
            if candidate == failed_candidate:
                raise RuntimeError("failed candidate has no near-object closure event")
            continue
        branch = Branch(
            candidate=candidate,
            episode_id=int(metadata["episode_id"]),
            success=success,
            arrays=arrays,
            event=event,
        )
        if candidate == failed_candidate:
            failed = branch
        elif success:
            controls.append(branch)
    if failed is None:
        raise RuntimeError(f"candidate {failed_candidate} was not loaded")
    if not controls:
        raise RuntimeError("no successful sibling grasp controls were found")
    return failed, controls


def attach_routes(branches: list[Branch], store: zarr.Group) -> None:
    episode = np.asarray(store["episode_id"][:])
    control_step = np.asarray(store["control_step"][:])
    for branch in branches:
        rows = np.flatnonzero(episode == branch.episode_id)
        rows = rows[np.argsort(control_step[rows])]
        expected = len(branch.arrays["action_chunks"])
        if len(rows) != expected:
            raise RuntimeError(
                f"episode {branch.episode_id}: {len(rows)} route rows, expected {expected}"
            )
        # control_step is the worker-wide batching counter, so sibling episodes
        # interleave and an individual episode need not advance by exactly one.
        # Its order still matches that episode's local inference/query order.
        observed_steps = control_step[rows]
        if len(observed_steps) > 1 and not np.all(np.diff(observed_steps) > 0):
            raise RuntimeError(f"episode {branch.episode_id}: unordered route rows")
        branch.route = normalized(
            np.asarray(store["hb_router_probs"].oindex[rows, :, :, :, :])
        )


def layer_metrics(
    p: np.ndarray,
    previous: np.ndarray | None,
    action_tokens: slice,
    late_start: int,
) -> dict[str, float]:
    """Metrics for one layer, p=[flow, suffix, expert]."""
    action = p[:, action_tokens, :]
    flow_distance = weighted_jaccard_distance(action[1:], action[:-1])
    roots = np.sqrt(action)
    acceleration = roots[2:] - 2.0 * roots[1:-1] + roots[:-2]
    final = action[-1]
    ordered = np.partition(final, -2, axis=-1)
    state = p[-1, 0]
    action_mean = normalized(final.mean(axis=0))
    values = {
        "within_flow_wj": float(flow_distance.mean()),
        # flow_distance[i] is the transition i -> i+1.  Restrict both ends to
        # the late-flow set {late_start, ..., F-1}, hence start at index 6.
        "late_flow_wj": float(flow_distance[late_start:].mean()),
        "route_acceleration": float(
            np.linalg.norm(acceleration, axis=-1).mean() / np.sqrt(2.0)
        ),
        "gate_entropy": float(entropy(final).mean()),
        "top12_margin": float((ordered[..., -1] - ordered[..., -2]).mean()),
        "state_action_gap": float(hellinger(state, action_mean)),
    }
    if previous is None:
        values["chunk_jump_full_h"] = np.nan
        values["chunk_jump_d9_h"] = np.nan
    else:
        previous_action = previous[:, action_tokens, :]
        values["chunk_jump_full_h"] = float(hellinger(action, previous_action).mean())
        values["chunk_jump_d9_h"] = float(hellinger(action[-1], previous_action[-1]).mean())
    return values


def mean_metrics(
    route: np.ndarray,
    query: int,
    layer_indices: list[int],
    action_tokens: slice,
    late_start: int,
) -> dict[str, float]:
    records = []
    for layer in layer_indices:
        previous = route[query - 1, layer] if query > 0 else None
        records.append(layer_metrics(route[query, layer], previous, action_tokens, late_start))
    return {
        metric: float(np.nanmean([record[metric] for record in records]))
        for metric in records[0]
    }


def event_reference_distance(
    branch: Branch,
    reference: list[Branch],
    layer_indices: list[int],
    action_tokens: slice,
) -> float:
    assert branch.route is not None
    source = branch.route[branch.event.query, layer_indices, :, action_tokens, :]
    center = np.mean(
        [
            control.route[control.event.query, layer_indices, :, action_tokens, :]
            for control in reference
            if control.route is not None
        ],
        axis=0,
    )
    return float(hellinger(source, center).mean())


def write_physical_alignment(
    branches: list[Branch],
    config: dict[str, Any],
    table_dir: Path,
) -> pd.DataFrame:
    pot_slice = slice(*config["pot_state_slice"])
    eef_slice = slice(*config["eef_state_slice"])
    grip_slice = slice(*config["gripper_state_slice"])
    low, high = map(int, config["alignment_window"])
    event_rows = []
    aligned_rows = []
    for branch in branches:
        event = branch.event
        event_rows.append({
            "candidate": branch.candidate,
            "episode_id": branch.episode_id,
            "outcome": "success" if branch.success else "failed_grasp",
            "closure_control_step": event.control_step,
            "closure_query": event.query,
            "closure_action_position_1based": event.action_position_1based,
            "eef_object_distance_m": event.eef_object_distance_m,
            "aperture_before": event.aperture_before,
            "aperture_after": event.aperture_after,
            "subsequent_max_lift_m": event.subsequent_max_lift_m,
            "lift_control_step": event.lift_control_step,
            "lift_query": event.lift_query,
        })
        query_state = branch.arrays["sim_state"]
        policy_state = branch.arrays["policy_state"]
        base_z = float(query_state[event.query, pot_slice.stop - 1])
        for relative in range(low, high + 1):
            query = event.query + relative
            if not 0 <= query < len(query_state):
                continue
            obj = query_state[query, pot_slice]
            eef = policy_state[query, eef_slice]
            aperture = float(np.abs(policy_state[query, grip_slice]).sum())
            aligned_rows.append({
                "candidate": branch.candidate,
                "episode_id": branch.episode_id,
                "outcome": "success" if branch.success else "failed_grasp",
                "relative_query": relative,
                "query": query,
                "pot_lift_from_event_m": float(obj[-1] - base_z),
                "eef_object_distance_m": float(np.linalg.norm(eef - obj)),
                "gripper_aperture": aperture,
            })
    pd.DataFrame(event_rows).to_csv(table_dir / "physical_event_alignment.csv", index=False)
    aligned = pd.DataFrame(aligned_rows)
    aligned.to_csv(table_dir / "aligned_physical_state.csv", index=False)
    return aligned


def audit_as_routes(
    store: zarr.Group,
    failed: Branch,
    config: dict[str, Any],
    table_dir: Path,
) -> dict[str, Any]:
    probabilities = np.asarray(store["as_probs"][:], np.float32)
    experts = np.asarray(store["as_expert_ids"][:])
    episodes = np.asarray(store["episode_id"][:])
    failed_rows = episodes == failed.episode_id
    model_layers = list(map(int, config["as_model_layers"]))
    rows = []
    for index, model_layer in enumerate(model_layers):
        group = "front" if model_layer <= 1 else "back"
        corpus_span = float(np.ptp(probabilities[:, index, :], axis=0).max())
        episode_span = float(np.ptp(probabilities[failed_rows, index, :], axis=0).max())
        unique_experts = np.unique(experts[:, index]).tolist()
        representative = probabilities[failed_rows, index, :][0]
        rows.append({
            "stored_layer_index": index,
            "model_layer": model_layer,
            "layer_group": group,
            "selected_expert": int(experts[failed_rows, index][0]),
            "prob_expert_0": float(representative[0]),
            "prob_expert_1": float(representative[1]),
            "prob_expert_2": float(representative[2]),
            "failed_episode_max_probability_span": episode_span,
            "whole_corpus_max_probability_span": corpus_span,
            "whole_corpus_selected_experts": "|".join(map(str, unique_experts)),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "as_route_state.csv", index=False)
    unique_rows = int(np.unique(probabilities.reshape(len(probabilities), -1), axis=0).shape[0])
    return {
        "rows_audited": int(len(probabilities)),
        "unique_probability_rows": unique_rows,
        "whole_corpus_max_probability_span": float(
            np.ptp(probabilities, axis=0).max()
        ),
        "failed_episode_max_probability_span": float(
            np.ptp(probabilities[failed_rows], axis=0).max()
        ),
        "selected_experts_by_model_layer": {
            str(row["model_layer"]): row["selected_expert"] for row in rows
        },
    }


def analyze_hb(
    failed: Branch,
    controls: list[Branch],
    config: dict[str, Any],
    table_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branches = [failed, *controls]
    model_layers = list(map(int, config["hb_model_layers"]))
    action_tokens = slice(*config["action_token_slice"])
    late_start = int(config["late_flow_start"])
    groups = {"front": list(range(0, 4)), "back": list(range(4, 8))}

    layer_rows = []
    for branch in branches:
        assert branch.route is not None
        for stored_layer, model_layer in enumerate(model_layers):
            values = layer_metrics(
                branch.route[branch.event.query, stored_layer],
                branch.route[branch.event.query - 1, stored_layer]
                if branch.event.query > 0 else None,
                action_tokens,
                late_start,
            )
            references = controls if not branch.success else [c for c in controls if c is not branch]
            distance = event_reference_distance(
                branch, references, [stored_layer], action_tokens
            )
            layer_rows.append({
                "candidate": branch.candidate,
                "episode_id": branch.episode_id,
                "outcome": "success" if branch.success else "failed_grasp",
                "closure_query": branch.event.query,
                "stored_layer_index": stored_layer,
                "model_layer": model_layer,
                "layer_group": "front" if stored_layer < 4 else "back",
                **values,
                "event_to_success_center_h": distance,
            })
    layer_frame = pd.DataFrame(layer_rows)
    layer_frame.to_csv(table_dir / "hb_event_layer_metrics.csv", index=False)

    group_rows = []
    for branch in branches:
        assert branch.route is not None
        for group_name, layer_indices in groups.items():
            values = mean_metrics(
                branch.route, branch.event.query, layer_indices, action_tokens, late_start
            )
            references = controls if not branch.success else [c for c in controls if c is not branch]
            values["event_to_success_center_h"] = event_reference_distance(
                branch, references, layer_indices, action_tokens
            )
            group_rows.append({
                "candidate": branch.candidate,
                "episode_id": branch.episode_id,
                "outcome": "success" if branch.success else "failed_grasp",
                "closure_query": branch.event.query,
                "layer_group": group_name,
                **values,
            })
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(table_dir / "hb_event_group_metrics.csv", index=False)

    metric_names = [
        "within_flow_wj",
        "late_flow_wj",
        "route_acceleration",
        "gate_entropy",
        "top12_margin",
        "state_action_gap",
        "chunk_jump_full_h",
        "chunk_jump_d9_h",
        "event_to_success_center_h",
    ]
    summary_rows = []
    summary: dict[str, Any] = {}
    for group_name in groups:
        subset = group_frame[group_frame.layer_group == group_name]
        failed_row = subset[subset.outcome == "failed_grasp"].iloc[0]
        success_rows = subset[subset.outcome == "success"]
        summary[group_name] = {}
        for metric in metric_names:
            fail_value = float(failed_row[metric])
            control_values = success_rows[metric].to_numpy(float)
            mean_value = float(np.mean(control_values))
            minimum = float(np.min(control_values))
            maximum = float(np.max(control_values))
            rank_high = int(1 + np.sum(control_values < fail_value))
            relation = "inside"
            if fail_value > maximum:
                relation = "above_all"
            elif fail_value < minimum:
                relation = "below_all"
            record = {
                "layer_group": group_name,
                "metric": metric,
                "failed_value": fail_value,
                "success_mean": mean_value,
                "success_min": minimum,
                "success_max": maximum,
                "failed_minus_success_mean": fail_value - mean_value,
                "failed_over_success_mean": fail_value / mean_value if mean_value else np.nan,
                "failed_rank_high_of_8": rank_high,
                "relation_to_success_range": relation,
                "n_success_controls": len(control_values),
            }
            summary_rows.append(record)
            summary[group_name][metric] = record
    pd.DataFrame(summary_rows).to_csv(table_dir / "hb_event_group_summary.csv", index=False)

    low, high = map(int, config["alignment_window"])
    aligned_rows = []
    for branch in branches:
        assert branch.route is not None
        for relative in range(low, high + 1):
            query = branch.event.query + relative
            if not 0 <= query < len(branch.route):
                continue
            for group_name, layer_indices in groups.items():
                values = mean_metrics(
                    branch.route, query, layer_indices, action_tokens, late_start
                )
                aligned_rows.append({
                    "candidate": branch.candidate,
                    "episode_id": branch.episode_id,
                    "outcome": "success" if branch.success else "failed_grasp",
                    "relative_query": relative,
                    "query": query,
                    "layer_group": group_name,
                    **values,
                })
    aligned = pd.DataFrame(aligned_rows)
    aligned.to_csv(table_dir / "aligned_chunk_dynamics.csv", index=False)

    assert failed.route is not None
    failed_event = failed.route[failed.event.query]
    control_events = np.stack([c.route[c.event.query] for c in controls])
    control_center = control_events.mean(axis=0)
    flow_token_rows = []
    path_rows = []
    for group_name, layer_indices in groups.items():
        matrix = hellinger(
            failed_event[layer_indices, :, action_tokens, :],
            control_center[layer_indices, :, action_tokens, :],
        ).mean(axis=0)
        for flow in range(matrix.shape[0]):
            for token_offset in range(matrix.shape[1]):
                flow_token_rows.append({
                    "layer_group": group_name,
                    "flow_step": flow,
                    "action_token_position": token_offset + 1,
                    "failed_to_success_center_h": float(matrix[flow, token_offset]),
                })
    pd.DataFrame(flow_token_rows).to_csv(
        table_dir / "hb_flow_token_distance.csv", index=False
    )

    for stored_layer, model_layer in enumerate(model_layers):
        group_name = "front" if stored_layer < 4 else "back"
        for flow in range(failed_event.shape[1]):
            source = normalized(failed_event[stored_layer, flow, action_tokens].mean(axis=0))
            center = normalized(control_center[stored_layer, flow, action_tokens].mean(axis=0))
            source_top4 = set(np.argsort(source)[-4:].tolist())
            center_top4 = set(np.argsort(center)[-4:].tolist())
            path_rows.append({
                "stored_layer_index": stored_layer,
                "model_layer": model_layer,
                "layer_group": group_name,
                "flow_step": flow,
                "failed_top1_expert": int(np.argmax(source)),
                "success_center_top1_expert": int(np.argmax(center)),
                "top1_same": bool(np.argmax(source) == np.argmax(center)),
                "top4_jaccard": len(source_top4 & center_top4) / len(source_top4 | center_top4),
                "failed_to_success_center_h": float(hellinger(source, center)),
            })
    paths = pd.DataFrame(path_rows)
    paths.to_csv(table_dir / "hb_instantaneous_expert_paths.csv", index=False)
    return summary, aligned, pd.DataFrame(flow_token_rows), paths


def analyze_actions(
    failed: Branch,
    controls: list[Branch],
    table_dir: Path,
) -> dict[str, Any]:
    branches = [failed, *controls]
    control_chunks = np.stack(
        [branch.arrays["action_chunks"][branch.event.query] for branch in controls]
    )
    center = control_chunks.mean(axis=0)
    rows = []
    token_rows = []
    for branch in branches:
        chunk = branch.arrays["action_chunks"][branch.event.query]
        rows.append({
            "candidate": branch.candidate,
            "episode_id": branch.episode_id,
            "outcome": "success" if branch.success else "failed_grasp",
            "closure_query": branch.event.query,
            "translation_norm_mean": float(np.linalg.norm(chunk[:, :3], axis=1).mean()),
            "rotation_norm_mean": float(np.linalg.norm(chunk[:, 3:6], axis=1).mean()),
            "gripper_command_mean": float(chunk[:, 6].mean()),
            "rms_distance_to_success_center": float(np.sqrt(np.mean((chunk - center) ** 2))),
        })
        for position, action in enumerate(chunk, start=1):
            token_rows.append({
                "candidate": branch.candidate,
                "episode_id": branch.episode_id,
                "outcome": "success" if branch.success else "failed_grasp",
                "closure_query": branch.event.query,
                "action_token_position": position,
                "translation_norm": float(np.linalg.norm(action[:3])),
                "rotation_norm": float(np.linalg.norm(action[3:6])),
                "gripper_command": float(action[6]),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "action_event_summary.csv", index=False)
    token_frame = pd.DataFrame(token_rows)
    token_frame.to_csv(table_dir / "action_token_event.csv", index=False)
    failed_row = frame[frame.outcome == "failed_grasp"].iloc[0]
    success_distance = frame[frame.outcome == "success"]["rms_distance_to_success_center"]
    failed_tokens = token_frame[token_frame.outcome == "failed_grasp"]
    successful_tokens = token_frame[token_frame.outcome == "success"]
    success_gripper = successful_tokens.groupby("action_token_position").gripper_command.mean()
    return {
        "failed_gripper_command_mean": float(failed_row.gripper_command_mean),
        "failed_gripper_command_by_token": failed_tokens.gripper_command.tolist(),
        "success_mean_gripper_command_by_token": success_gripper.tolist(),
        "failed_rms_distance_to_success_center": float(
            failed_row.rms_distance_to_success_center
        ),
        "success_rms_distance_range": [
            float(success_distance.min()), float(success_distance.max())
        ],
    }


def summarize_continuous_hb(aligned: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for group_name in ("front", "back"):
        group = aligned[aligned.layer_group == group_name]
        summary[group_name] = {}
        for relative in range(-3, 4):
            subset = group[group.relative_query == relative]
            summary[group_name][str(relative)] = {}
            for metric in (
                "late_flow_wj", "route_acceleration", "chunk_jump_full_h"
            ):
                failed_value = float(
                    subset[subset.outcome == "failed_grasp"][metric].iloc[0]
                )
                controls = subset[subset.outcome == "success"][metric].to_numpy(float)
                relation = "inside"
                if failed_value > controls.max():
                    relation = "above_all"
                elif failed_value < controls.min():
                    relation = "below_all"
                summary[group_name][str(relative)][metric] = {
                    "failed_value": failed_value,
                    "success_mean": float(controls.mean()),
                    "success_min": float(controls.min()),
                    "success_max": float(controls.max()),
                    "relation_to_success_range": relation,
                }
    return summary


def summarize_expert_paths(
    paths: pd.DataFrame, flow_token: pd.DataFrame
) -> dict[str, Any]:
    by_layer = (
        paths.groupby(["model_layer", "layer_group"])
        .top1_same.mean()
        .reset_index()
    )
    final_flow = (
        flow_token[flow_token.flow_step == flow_token.flow_step.max()]
        .groupby("layer_group")
        .failed_to_success_center_h.mean()
    )
    return {
        "top1_match_fraction_over_10_flow_steps_by_layer": {
            str(int(row.model_layer)): float(row.top1_same)
            for row in by_layer.itertuples()
        },
        "failed_to_success_center_h_at_final_flow": {
            group: float(value) for group, value in final_flow.items()
        },
    }


def plot_results(
    failed: Branch,
    controls: list[Branch],
    physical: pd.DataFrame,
    hb_summary: dict[str, Any],
    aligned: pd.DataFrame,
    flow_token: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
    })
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)

    ax = axes[0, 0]
    success = physical[physical.outcome == "success"]
    failed_physical = physical[physical.outcome == "failed_grasp"]
    grouped = success.groupby("relative_query")["pot_lift_from_event_m"]
    x = np.array(sorted(success.relative_query.unique()))
    mean = grouped.mean().reindex(x).to_numpy()
    low = grouped.min().reindex(x).to_numpy()
    high = grouped.max().reindex(x).to_numpy()
    ax.fill_between(x, low * 100, high * 100, color="#8db3c7", alpha=0.25)
    ax.plot(x, mean * 100, color="#276678", label="Success mean (n=7)")
    ax.plot(
        failed_physical.relative_query,
        failed_physical.pot_lift_from_event_m * 100,
        color="#b43c3c",
        linewidth=2,
        label="Failed grasp",
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set(title="Physical event alignment", xlabel="Queries from gripper closure", ylabel="Pot lift from event (cm)")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    labels = []
    ratios = []
    colors = []
    for group_name, color in [("front", "#477998"), ("back", "#6b4c9a")]:
        for metric, short in [
            ("late_flow_wj", "late volatility"),
            ("route_acceleration", "acceleration"),
            ("chunk_jump_full_h", "chunk jump"),
        ]:
            labels.append(f"{group_name}\n{short}")
            ratios.append(hb_summary[group_name][metric]["failed_over_success_mean"])
            colors.append(color)
    ax.bar(np.arange(len(ratios)), ratios, color=colors, width=0.72)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    ax.set(ylabel="Failed / success-control mean", title="HB state at the closure chunk")

    ax = axes[1, 0]
    palette = {"front": "#247ba0", "back": "#7b2cbf"}
    for group_name in ("front", "back"):
        subset = aligned[aligned.layer_group == group_name]
        successful = subset[subset.outcome == "success"]
        failed_subset = subset[subset.outcome == "failed_grasp"]
        x = np.array(sorted(successful.relative_query.unique()))
        success_mean = successful.groupby("relative_query")["chunk_jump_full_h"].mean().reindex(x)
        ax.plot(x, success_mean, color=palette[group_name], linestyle="--", alpha=0.65, label=f"{group_name} success mean")
        ax.plot(
            failed_subset.relative_query,
            failed_subset.chunk_jump_full_h,
            color=palette[group_name],
            linewidth=2,
            label=f"{group_name} failed",
        )
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set(title="Continuous HB state across chunks", xlabel="Queries from gripper closure", ylabel="Cross-chunk Hellinger jump")
    ax.legend(frameon=False, ncol=2, fontsize=8)

    ax = axes[1, 1]
    profile = flow_token.groupby(["layer_group", "flow_step"])["failed_to_success_center_h"].mean()
    for group_name in ("front", "back"):
        values = profile.loc[group_name]
        ax.plot(values.index, values.values, marker="o", color=palette[group_name], label=group_name)
    ax.set(title="Failed-vs-success HB divergence", xlabel="Flow step", ylabel="Hellinger distance")
    ax.legend(frameon=False)
    fig.suptitle(
        f"Failed grasp candidate {failed.candidate:02d} vs {len(controls)} same-snapshot successes",
        fontsize=12,
    )
    fig.savefig(figure_dir / "failed_grasp_moe_dynamics.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    metric_specs = [
        ("late_flow_wj", "Late-flow weighted-Jaccard distance"),
        ("route_acceleration", "Routing acceleration"),
    ]
    for row, (metric, label) in enumerate(metric_specs):
        for column, group_name in enumerate(("front", "back")):
            ax = axes[row, column]
            subset = aligned[aligned.layer_group == group_name]
            successful = subset[subset.outcome == "success"]
            failed_subset = subset[subset.outcome == "failed_grasp"]
            x = np.array(sorted(successful.relative_query.unique()))
            grouped = successful.groupby("relative_query")[metric]
            mean = grouped.mean().reindex(x).to_numpy()
            low = grouped.min().reindex(x).to_numpy()
            high = grouped.max().reindex(x).to_numpy()
            ax.fill_between(x, low, high, color=palette[group_name], alpha=0.18)
            ax.plot(x, mean, color=palette[group_name], linestyle="--", label="Success mean/range")
            ax.plot(
                failed_subset.relative_query,
                failed_subset[metric],
                color="#b43c3c",
                linewidth=2,
                label="Failed grasp",
            )
            ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
            ax.set(
                title=f"HB {group_name}: {label}",
                xlabel="Queries from gripper closure",
                ylabel=label,
            )
            if row == 0 and column == 0:
                ax.legend(frameon=False)
    fig.savefig(figure_dir / "hb_continuous_flow_dynamics.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    vmax = float(flow_token.failed_to_success_center_h.max())
    image = None
    for ax, group_name in zip(axes, ("front", "back")):
        matrix = (
            flow_token[flow_token.layer_group == group_name]
            .pivot(index="action_token_position", columns="flow_step", values="failed_to_success_center_h")
            .sort_index()
        )
        image = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set(title=f"HB {group_name}", xlabel="Flow step", ylabel="Action-token position")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns)
        ax.set_yticks(range(matrix.shape[0]), matrix.index)
    assert image is not None
    fig.colorbar(image, ax=axes, label="Failed-to-success-center Hellinger")
    fig.savefig(figure_dir / "hb_flow_token_distance_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_root = (WORKSPACE_ROOT / config["run_root"]).resolve()
    snapshot_dir = (
        run_root / "formal" / f"worker{config['worker']}" / f"snapshot_{config['snapshot']:03d}"
    )
    route_store = run_root / "formal/server/routes.zarr"
    output = args.output.resolve()
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    failed, controls = load_branches(config, snapshot_dir)
    branches = [failed, *controls]
    store = zarr.open_group(str(route_store), mode="r")
    attach_routes(branches, store)

    physical = write_physical_alignment(branches, config, table_dir)
    as_summary = audit_as_routes(store, failed, config, table_dir)
    hb_summary, aligned, flow_token, expert_paths = analyze_hb(
        failed, controls, config, table_dir
    )
    action_summary = analyze_actions(failed, controls, table_dir)
    plot_results(
        failed, controls, physical, hb_summary, aligned, flow_token, figure_dir
    )

    success_distances = [c.event.eef_object_distance_m for c in controls]
    summary = {
        "schema": "himoe.failed_grasp_moe_dynamics.v1",
        "training": False,
        "comparison_design": "one failed grasp vs successful siblings from the same simulator snapshot",
        "raw_sources": {
            "snapshot_dir": str(snapshot_dir.relative_to(WORKSPACE_ROOT)),
            "route_store": str(route_store.relative_to(WORKSPACE_ROOT)),
        },
        "architecture": {
            "vision_inputs_per_query": "224x224 base image plus 224x224 wrist image",
            "vision_prefix_recomputed_each_query": True,
            "vision_prefix_fixed_across_the_10_flow_steps": True,
            "per_query_raw_vision_captured": False,
            "hb_router_input": "1024-dimensional hidden state",
            "hb_layers": config["hb_model_layers"],
            "hb_front_layers": config["hb_front_layers"],
            "hb_back_layers": config["hb_back_layers"],
            "as_router_input": "24-dimensional data_mask expanded over suffix tokens",
            "as_layers": config["as_model_layers"],
            "hidden_state_captured": False,
        },
        "failed_event": {
            "candidate": failed.candidate,
            "episode_id": failed.episode_id,
            **jsonable(failed.event.__dict__),
        },
        "success_controls": {
            "n": len(controls),
            "candidates": [c.candidate for c in controls],
            "episodes": [c.episode_id for c in controls],
            "closure_queries": [c.event.query for c in controls],
            "eef_object_distance_range_m": [min(success_distances), max(success_distances)],
        },
        "as_routing": as_summary,
        "hb_event_comparison": hb_summary,
        "hb_continuous_comparison": summarize_continuous_hb(aligned),
        "hb_expert_paths": summarize_expert_paths(expert_paths, flow_token),
        "action_event": action_summary,
        "limitations": [
            "This is a post-hoc single failed-grasp case with seven matched successful controls.",
            "The capture stores router probabilities, not HB/AS hidden vectors or expert outputs.",
            "Per-query raw base/wrist images were not stored; available MP4s contain only compressed base-camera frames for a subset of branches.",
            "Routing differences are correlational and may encode the different grasp geometry.",
            "AS expert outputs may change even though AS routing is exactly constant.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "OK",
        "failed_candidate": failed.candidate,
        "failed_query": failed.event.query,
        "success_controls": len(controls),
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
