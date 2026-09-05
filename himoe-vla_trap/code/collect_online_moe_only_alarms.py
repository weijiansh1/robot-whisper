#!/usr/bin/env python3
"""Collect fixed random episodes with a physics-free MoE online alarm."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
ROUTE_CAPTURE = WORKSPACE_ROOT / "himoe-route-capture"
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-libero-wrist-fix/src"))
sys.path.insert(0, str(ROUTE_CAPTURE))

from himoe_libero_bridge.client import PolicyClient  # noqa: E402
from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402
from himoe_libero_bridge.protocol import FLOW_NOISE_SHAPE  # noqa: E402
from rolling_star_collect import (  # noqa: E402
    _atomic_json,
    _atomic_npz,
    _frame,
    _sha256_file,
    _write_video,
)
from rollout_with_routes import sim_joint_layout  # noqa: E402

from collect_online_belief_alarms import (  # noqa: E402
    FULL_PROBS_KEY,
    draw_label,
    parse_int_list,
    random_noise,
    request_with_full_routes,
)
from moe_only_online_selector import (  # noqa: E402
    HealthySequenceBank,
    MoeOnlyOnlineAlarm,
)
from moe_dynamics_online_selector import (  # noqa: E402
    DynamicsCalibration,
    MoeDynamicsOnlineAlarm,
    SELECTOR_VERSION as DYNAMICS_SELECTOR_VERSION,
    sha256_file as dynamics_sha256_file,
)
from moe_self_reference_selector import (  # noqa: E402
    SELECTOR_VERSION as SELF_REFERENCE_SELECTOR_VERSION,
    SelfReferenceConfig,
    SelfReferenceCouplingCollapseAlarm,
)


SCHEMA = "himoe.online_moe_only_alarm.v1"
DYNAMICS_SCHEMA = "himoe.online_moe_dynamics_alarm.v2"
SELF_REFERENCE_SCHEMA = "himoe.online_self_reference_alarm.v3"
LEGACY_SELECTOR_VERSION = "transition_split_v1"
TARGET_STATE_SLICES = {
    "moka_pot_1_joint0": slice(10, 13),
    "moka_pot_2_joint0": slice(17, 20),
}


def schema_for_selector(version: str) -> str:
    if version == SELF_REFERENCE_SELECTOR_VERSION:
        return SELF_REFERENCE_SCHEMA
    if version == DYNAMICS_SELECTOR_VERSION:
        return DYNAMICS_SCHEMA
    return SCHEMA


def posthoc_grasp_events(
    control_sim_state: np.ndarray,
    control_eef_position: np.ndarray,
    control_gripper_qpos: np.ndarray,
    control_query_index: np.ndarray,
    close_threshold: float = 0.05,
    near_threshold: float = 0.16,
) -> List[Dict[str, Any]]:
    """Describe closures after rollout; never called by the selector."""

    aperture = np.abs(control_gripper_qpos).sum(axis=1)
    crossings = np.flatnonzero(
        (aperture[:-1] >= close_threshold) & (aperture[1:] < close_threshold)
    ) + 1
    events = []
    for step_value in crossings:
        step = int(step_value)
        eef = control_eef_position[step]
        for target, state_slice in TARGET_STATE_SLICES.items():
            pot = control_sim_state[step, state_slice]
            distance = float(np.linalg.norm(eef - pot))
            if distance >= near_threshold:
                continue
            future_eef = control_eef_position[step:]
            future_pot = control_sim_state[step:, state_slice]
            eef_displacement = np.linalg.norm(future_eef - eef, axis=1)
            pot_displacement = np.linalg.norm(future_pot - pot, axis=1)
            separation = np.linalg.norm(future_eef - future_pot, axis=1)
            action_index = step - 1
            query = int(control_query_index[action_index])
            events.append({
                "target": target,
                "query": query,
                "control_step": step,
                "eef_pot_distance_at_closure_m": distance,
                "aperture_before": float(aperture[step - 1]),
                "aperture_after": float(aperture[step]),
                "eef_max_displacement_after_m": float(eef_displacement.max()),
                "pot_max_displacement_after_m": float(pot_displacement.max()),
                "pot_max_lift_after_m": float(
                    (future_pot[:, 2] - pot[2]).max()
                ),
                "eef_pot_max_separation_after_m": float(separation.max()),
                "failed_grasp_geometry": bool(
                    eef_displacement.max() >= 0.10
                    and pot_displacement.max() < 0.01
                    and separation.max() >= 0.15
                ),
                "role": "posthoc_only_not_available_to_selector",
            })
    return events


def annotate_frames(
    frames: Sequence[np.ndarray],
    frame_queries: Sequence[int],
    frame_action_positions: Sequence[int],
    init_state_id: int,
    decisions: Mapping[int, Mapping[str, Any]],
    alarm_queries: Sequence[int],
) -> List[np.ndarray]:
    alarm_set = set(int(value) for value in alarm_queries)
    first_alarm = min(alarm_set) if alarm_set else None
    output = []
    for source, query, action_position in zip(
        frames, frame_queries, frame_action_positions
    ):
        image = np.asarray(source, dtype=np.uint8).copy()
        draw_label(
            image,
            "init=%02d  query=%02d  action=%02d"
            % (init_state_id, query, action_position),
            (14, 28),
            (255, 255, 255),
            0.55,
        )
        decision = decisions.get(int(query))
        next_y = 58
        if query in alarm_set:
            cv2.rectangle(
                image,
                (3, 3),
                (image.shape[1] - 4, image.shape[0] - 4),
                (255, 0, 0),
                thickness=8,
            )
            draw_label(
                image,
                "MoE-ONLY ALARM - chunk still executed",
                (14, next_y),
                (255, 80, 80),
                0.64,
            )
            next_y += 34
        elif decision is not None and bool(decision["raw_reject"]):
            cv2.rectangle(
                image,
                (3, 3),
                (image.shape[1] - 4, image.shape[0] - 4),
                (255, 220, 0),
                thickness=5,
            )
            draw_label(
                image,
                "MoE anomaly %d/2 - no alarm yet"
                % int(decision["consecutive_raw_rejects"]),
                (14, next_y),
                (255, 220, 0),
                0.60,
            )
            next_y += 33
        elif first_alarm is not None and query > first_alarm:
            draw_label(
                image,
                "AFTER ALARM - continuing unchanged",
                (14, next_y),
                (255, 100, 100),
                0.57,
            )
            next_y += 31
        if decision is not None:
            if "planning_churn_ratio" in decision:
                draw_label(
                    image,
                    "self-route: response %.2f<=%.2f  churn %.2f>=%.2f"
                    % (
                        decision["state_response_ratio"],
                        decision["state_response_ratio_max"],
                        decision["planning_churn_ratio"],
                        decision["planning_churn_ratio_min"],
                    ),
                    (14, next_y),
                    (255, 255, 255),
                    0.46,
                )
                next_y += 27
                draw_label(
                    image,
                    "state/action gap %.2f>=%.2f  own prefix only"
                    % (
                        decision["state_action_gap_ratio"],
                        decision["state_action_gap_ratio_min"],
                    ),
                    (14, next_y),
                    (255, 255, 255),
                    0.46,
                )
            else:
                matched = [
                    int(value) for value in decision["matched_reference_queries"]
                ]
                draw_label(
                    image,
                    "matched healthy q=%d..%d  median=%d"
                    % (min(matched), max(matched), int(np.median(matched))),
                    (14, next_y),
                    (255, 255, 255),
                    0.49,
                )
                next_y += 27
                if "normalized_acceleration_excess" in decision:
                    draw_label(
                        image,
                        "route acc front %.4f back %.4f  excess %.3f/%.3f"
                        % (
                            decision["front_route_acceleration"],
                            decision["back_route_acceleration"],
                            decision["normalized_acceleration_excess"],
                            decision["normalized_excess_threshold"],
                        ),
                        (14, next_y),
                        (255, 255, 255),
                        0.46,
                    )
                else:
                    draw_label(
                        image,
                        "ratios gap %.2f  jump %.2f  action %.2f"
                        % (
                            decision["layer5_gap_ratio"],
                            decision["back_chunk_jump_ratio"],
                            decision["front_action_distance_ratio"],
                        ),
                        (14, next_y),
                        (255, 255, 255),
                        0.50,
                    )
        output.append(image)
    return output


def run_episode(
    args: argparse.Namespace,
    client: PolicyClient,
    bank: HealthySequenceBank | None,
    output: pathlib.Path,
    episode_index: int,
    init_state_id: int,
    dynamics_calibration: DynamicsCalibration | None,
    self_reference_config: SelfReferenceConfig | None,
    flow_noise_seed: int | None,
) -> Dict[str, Any]:
    episode_dir = output / (
        "episode_%03d_init_%02d_%s"
        % (
            episode_index,
            init_state_id,
            "random" if flow_noise_seed is None else f"flowseed_{flow_noise_seed}",
        )
    )
    episode_dir.mkdir(parents=True, exist_ok=False)
    environment, observation, task, prompt = _load_task(EpisodeConfig(
        task_suite=args.benchmark,
        task_id=args.task_id,
        init_state_id=init_state_id,
        seed=args.environment_seed,
        host=args.host,
        port=args.port,
        libero_root=args.libero_root,
        output_root=str(episode_dir),
        settle_steps=args.settle_steps,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        render_size=args.render_size,
        fps=args.fps,
        inference_timeout=args.inference_timeout,
    ))
    if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION:
        if self_reference_config is None:
            raise RuntimeError("v3 selector requires self-reference config")
        selector = SelfReferenceCouplingCollapseAlarm(self_reference_config)
        metric_names = selector.metric_names
    elif args.selector_version == DYNAMICS_SELECTOR_VERSION:
        if dynamics_calibration is None:
            raise RuntimeError("v2 selector requires dynamics calibration")
        if bank is None:
            raise RuntimeError("v2 selector requires a healthy route bank")
        selector = MoeDynamicsOnlineAlarm(bank, dynamics_calibration)
        metric_names = selector.metric_names
    else:
        if bank is None:
            raise RuntimeError("legacy selector requires a healthy route bank")
        selector = MoeOnlyOnlineAlarm(
            bank, persistence=args.persistence, max_advance=args.max_reference_advance
        )
        metric_names = (
            "mean_local_match_distance",
            "layer5_gap",
            "back_chunk_jump",
            "front_action_distance",
            "layer5_gap_ratio",
            "back_chunk_jump_ratio",
            "front_action_distance_ratio",
        )
    routes: List[np.ndarray] = []
    noises: List[np.ndarray] = []
    action_chunks: List[np.ndarray] = []
    executed_masks: List[np.ndarray] = []
    policy_states: List[np.ndarray] = []
    query_sim_states: List[np.ndarray] = []
    inference_ms: List[float] = []
    action_steps_at_query: List[int] = []
    raw_rejects: List[bool] = []
    alarm_flags: List[bool] = []
    metrics: List[List[float]] = []
    matched_queries: List[np.ndarray] = []
    decisions: Dict[int, Dict[str, Any]] = {}
    alarm_records: List[Dict[str, Any]] = []

    frames: List[np.ndarray] = []
    frame_queries: List[int] = []
    frame_action_positions: List[int] = []
    control_sim_states: List[np.ndarray] = []
    control_eef: List[np.ndarray] = []
    control_gripper: List[np.ndarray] = []
    control_actions: List[np.ndarray] = []
    control_query_indices: List[int] = []
    control_success: List[bool] = []
    action_steps = 0
    success = False
    started = time.time()
    flow_rng = (
        None if flow_noise_seed is None else np.random.default_rng(flow_noise_seed)
    )
    try:
        for _ in range(args.settle_steps):
            observation, _reward, _done, _info = environment.step(
                LIBERO_DUMMY_ACTION.tolist()
            )
        _atomic_json(episode_dir / "sim_layout.json", sim_joint_layout(environment))
        frames.append(_frame(observation))
        frame_queries.append(-1)
        frame_action_positions.append(0)
        control_sim_states.append(
            np.asarray(environment.get_sim_state(), dtype=np.float32)
        )
        control_eef.append(
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
        )
        control_gripper.append(
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
        )

        max_queries = (args.max_steps + args.replan_steps - 1) // args.replan_steps
        for query in range(max_queries):
            if action_steps >= args.max_steps or success:
                break
            policy_observation = build_policy_observation(observation, prompt)
            noise = (
                random_noise(
                    args.seed, args.task_id, init_state_id, episode_index, query
                )
                if flow_rng is None
                else flow_rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
            )
            actions, route, server_ms = request_with_full_routes(
                client,
                policy_observation,
                noise,
                args.episode_id_base + episode_index,
            )
            decision = selector.update(route)
            decision_dict: Optional[Dict[str, Any]] = None
            if decision is not None:
                decision_dict = decision.to_dict()
                decisions[query] = decision_dict
                if decision.alarm:
                    alarm_record = dict(decision_dict)
                    alarm_record.update({
                        "action_step_before_execution": action_steps,
                        "intervention": "none_chunk_executed_unchanged",
                    })
                    alarm_records.append(alarm_record)
                    if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION:
                        print(
                            "MOE-SELF ALARM episode=%d init=%d q=%d ratios=(response %.3f, churn %.3f, gap %.3f); continuing"
                            % (
                                episode_index,
                                init_state_id,
                                query,
                                decision.state_response_ratio,
                                decision.planning_churn_ratio,
                                decision.state_action_gap_ratio,
                            ),
                            flush=True,
                        )
                    elif args.selector_version == DYNAMICS_SELECTOR_VERSION:
                        print(
                            "MOE-DYNAMICS ALARM episode=%d init=%d q=%d excess=%.4f threshold=%.4f; continuing"
                            % (
                                episode_index,
                                init_state_id,
                                query,
                                decision.normalized_acceleration_excess,
                                decision.normalized_excess_threshold,
                            ),
                            flush=True,
                        )
                    else:
                        print(
                            "MOE-ONLY ALARM episode=%d init=%d q=%d ratios=(%.3f, %.3f, %.3f); continuing"
                            % (
                                episode_index,
                                init_state_id,
                                query,
                                decision.layer5_gap_ratio,
                                decision.back_chunk_jump_ratio,
                                decision.front_action_distance_ratio,
                            ),
                            flush=True,
                        )

            routes.append(route.astype(np.float16))
            noises.append(noise)
            action_chunks.append(actions)
            policy_states.append(
                np.asarray(policy_observation["observation/state"], dtype=np.float32)
            )
            query_sim_states.append(
                np.asarray(environment.get_sim_state(), dtype=np.float32)
            )
            inference_ms.append(server_ms)
            action_steps_at_query.append(action_steps)
            raw_rejects.append(False if decision is None else decision.raw_reject)
            alarm_flags.append(False if decision is None else decision.alarm)
            if decision is None:
                metrics.append([np.nan] * len(metric_names))
                reference_count = 0 if bank is None else len(bank.routes)
                matched_queries.append(
                    np.full(reference_count, -1, dtype=np.int16)
                )
            else:
                metrics.append([float(getattr(decision, name)) for name in metric_names])
                if hasattr(decision, "matched_reference_queries"):
                    matched_queries.append(
                        np.asarray(decision.matched_reference_queries, dtype=np.int16)
                    )
                else:
                    matched_queries.append(np.empty(0, dtype=np.int16))

            take = min(args.replan_steps, len(actions), args.max_steps - action_steps)
            mask = np.zeros((args.replan_steps,), dtype=np.bool_)
            for action_index in range(take):
                observation, _reward, _done, _info = environment.step(
                    actions[action_index].tolist()
                )
                action_steps += 1
                mask[action_index] = True
                success = bool(environment.check_success())
                control_actions.append(actions[action_index].copy())
                control_query_indices.append(query)
                control_success.append(success)
                control_sim_states.append(
                    np.asarray(environment.get_sim_state(), dtype=np.float32)
                )
                control_eef.append(
                    np.asarray(observation["robot0_eef_pos"], dtype=np.float32)
                )
                control_gripper.append(
                    np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)
                )
                frames.append(_frame(observation))
                frame_queries.append(query)
                frame_action_positions.append(action_index + 1)
                if success:
                    break
            executed_masks.append(mask)
            print(
                "episode=%d q=%02d raw=%s alarm=%s success=%s"
                % (
                    episode_index,
                    query,
                    raw_rejects[-1],
                    alarm_flags[-1],
                    success,
                ),
                flush=True,
            )

        control_sim_array = np.stack(control_sim_states).astype(np.float32)
        control_eef_array = np.stack(control_eef).astype(np.float32)
        control_gripper_array = np.stack(control_gripper).astype(np.float32)
        control_query_array = np.asarray(control_query_indices, dtype=np.int32)
        posthoc_events = (
            posthoc_grasp_events(
                control_sim_array,
                control_eef_array,
                control_gripper_array,
                control_query_array,
            )
            if "put_both_moka_pots_on_the_stove" in str(task.name)
            else []
        )
        for alarm in alarm_records:
            alarm["posthoc_nearest_grasp_event_query"] = (
                None
                if not posthoc_events
                else min(
                    posthoc_events,
                    key=lambda event: abs(int(event["query"]) - int(alarm["query"])),
                )["query"]
            )

        alarm_queries = [int(item["query"]) for item in alarm_records]
        annotated = annotate_frames(
            frames,
            frame_queries,
            frame_action_positions,
            init_state_id,
            decisions,
            alarm_queries,
        )
        videos = []
        if alarm_records:
            video_dir = episode_dir / "videos"
            clean_path = video_dir / "alarm_clean_full.mp4"
            annotated_path = video_dir / "alarm_annotated_full.mp4"
            _write_video(clean_path, frames, fps=args.fps)
            _write_video(annotated_path, annotated, fps=args.fps)
            first_alarm_query = alarm_queries[0]
            alarm_frame_indices = [
                index
                for index, query in enumerate(frame_queries)
                if query == first_alarm_query
            ]
            first_alarm_frame = alarm_frame_indices[0]
            left = max(0, first_alarm_frame - args.clip_pre_frames)
            right = min(len(frames), first_alarm_frame + args.clip_post_frames)
            clip_path = video_dir / "alarm_annotated_clip.mp4"
            _write_video(clip_path, annotated[left:right], fps=args.fps)
            for role, path in (
                ("clean_full", clean_path),
                ("annotated_full", annotated_path),
                ("annotated_clip", clip_path),
            ):
                videos.append({
                    "role": role,
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                })

        arrays_path = episode_dir / "trajectory_and_routes.npz"
        _atomic_npz(arrays_path, {
            "hb_router_probs": np.stack(routes).astype(np.float16),
            "flow_noise": np.stack(noises).astype(np.float32),
            "action_chunks": np.stack(action_chunks).astype(np.float32),
            "action_executed": np.stack(executed_masks),
            "policy_state": np.stack(policy_states).astype(np.float32),
            "query_sim_state": np.stack(query_sim_states).astype(np.float32),
            "server_inference_ms": np.asarray(inference_ms, dtype=np.float32),
            "action_steps_at_query": np.asarray(action_steps_at_query, dtype=np.int32),
            "selector_raw_reject": np.asarray(raw_rejects, dtype=np.bool_),
            "selector_alarm": np.asarray(alarm_flags, dtype=np.bool_),
            "selector_metric_names": np.asarray(metric_names),
            "selector_metrics": np.asarray(metrics, dtype=np.float64),
            "matched_reference_queries": np.stack(matched_queries),
            "control_sim_state": control_sim_array,
            "control_eef_position": control_eef_array,
            "control_gripper_qpos": control_gripper_array,
            "control_action": np.asarray(control_actions, dtype=np.float32),
            "control_query_index": control_query_array,
            "control_success_after_step": np.asarray(control_success, dtype=np.bool_),
            "frame_query_index": np.asarray(frame_queries, dtype=np.int16),
            "frame_action_position": np.asarray(frame_action_positions, dtype=np.int8),
        })
        summary = {
            "schema": (
                SELF_REFERENCE_SCHEMA
                if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION
                else (
                    DYNAMICS_SCHEMA
                    if args.selector_version == DYNAMICS_SELECTOR_VERSION
                    else SCHEMA
                )
            ),
            "status": "complete",
            "training": False,
            "selector_version": args.selector_version,
            "selector_inputs": (
                ["current HB router probabilities", "own episode route prefix"]
                if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION
                else [
                    "current HB router probabilities",
                    "previous HB router probabilities",
                    "successful HB route sequences",
                ]
            ),
            "selector_uses_task_identity": False,
            "selector_uses_normal_trajectory_bank": bank is not None,
            "selector_uses_physical_state": False,
            "selector_uses_gripper_event": False,
            "selector_uses_action_values": False,
            "selector_uses_reward_or_success": False,
            "posthoc_physics_computed_after_rollout": True,
            "episode_index": episode_index,
            "episode_id": args.episode_id_base + episode_index,
            "init_state_id": init_state_id,
            "flow_noise_seed": flow_noise_seed,
            "task_name": str(task.name),
            "prompt": prompt,
            "success": success,
            "action_steps": action_steps,
            "inference_calls": len(routes),
            "raw_reject_queries": [
                index for index, value in enumerate(raw_rejects) if value
            ],
            "alarm_queries": alarm_queries,
            "alarm_count": len(alarm_records),
            "continued_after_alarm": bool(
                alarm_records
                and action_steps > int(alarm_records[0]["action_step_before_execution"])
            ),
            "post_alarm_action_steps": (
                0
                if not alarm_records
                else action_steps
                - int(alarm_records[0]["action_step_before_execution"])
            ),
            "alarms": alarm_records,
            "posthoc_near_pot_grasp_events": posthoc_events,
            "videos": videos,
            "trajectory_npz": str(arrays_path.relative_to(output)),
            "trajectory_npz_sha256": _sha256_file(arrays_path),
            "wall_s": round(time.time() - started, 3),
        }
        _atomic_json(episode_dir / "summary.json", summary)
        print(
            "EPISODE COMPLETE index=%d init=%d success=%s alarms=%d steps=%d"
            % (episode_index, init_state_id, success, len(alarm_records), action_steps),
            flush=True,
        )
        return summary
    finally:
        try:
            environment.close()
        except BaseException:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--benchmark", default="libero_10")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--init-state-ids", type=parse_int_list, default=(0, 3, 7, 20))
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--flow-noise-seeds",
        type=parse_int_list,
        help=(
            "Optional ordered per-episode flow-noise seeds. When supplied, "
            "--init-state-ids is cycled in the given order and the seed list "
            "must have --episodes entries."
        ),
    )
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--episode-id-base", type=int, default=1_610_000_000)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clip-pre-frames", type=int, default=80)
    parser.add_argument("--clip-post-frames", type=int, default=160)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--max-reference-advance", type=int, default=2)
    parser.add_argument(
        "--selector-version",
        choices=(
            LEGACY_SELECTOR_VERSION,
            DYNAMICS_SELECTOR_VERSION,
            SELF_REFERENCE_SELECTOR_VERSION,
        ),
        default=LEGACY_SELECTOR_VERSION,
    )
    parser.add_argument("--dynamics-calibration", type=pathlib.Path)
    parser.add_argument(
        "--self-reference-config",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "configs/self_reference_coupling_collapse_v3.json",
    )
    parser.add_argument("--libero-root", required=True)
    parser.add_argument(
        "--healthy-reference",
        type=pathlib.Path,
        default=PACKAGE_ROOT
        / "results/moe_only_online_alarm/full_healthy_route_sequences.npz",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.episodes < 1 or args.replan_steps != 10:
        raise ValueError("requires episodes >= 1 and replan_steps=10")
    if (
        args.selector_version != SELF_REFERENCE_SELECTOR_VERSION
        and args.persistence != 2
    ):
        raise ValueError("legacy and v2 selectors require persistence=2")
    if args.episode_id_base + args.episodes > np.iinfo(np.int32).max:
        raise ValueError("episode IDs exceed int32")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    bank = (
        None
        if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION
        else HealthySequenceBank.load(args.healthy_reference.resolve())
    )
    dynamics_calibration = None
    if args.selector_version == DYNAMICS_SELECTOR_VERSION:
        if args.dynamics_calibration is None:
            raise ValueError("--dynamics-calibration is required for the v2 selector")
        dynamics_calibration = DynamicsCalibration.load(
            args.dynamics_calibration.resolve(),
            args.healthy_reference.resolve(),
            bank,
        )
        if (
            args.persistence != dynamics_calibration.persistence
            or args.max_reference_advance
            != dynamics_calibration.max_reference_advance
        ):
            raise ValueError("CLI persistence/matcher settings differ from frozen calibration")
    self_reference_config = None
    if args.selector_version == SELF_REFERENCE_SELECTOR_VERSION:
        self_reference_config = SelfReferenceConfig.load(
            args.self_reference_config.resolve()
        )
    if args.flow_noise_seeds is not None:
        if len(args.flow_noise_seeds) != args.episodes:
            raise ValueError(
                "ordered --flow-noise-seeds must contain exactly --episodes entries"
            )
        schedule = [
            int(args.init_state_ids[index % len(args.init_state_ids)])
            for index in range(args.episodes)
        ]
        flow_noise_schedule: List[int | None] = list(args.flow_noise_seeds)
    else:
        rng = np.random.default_rng(args.seed)
        schedule = []
        while len(schedule) < args.episodes:
            schedule.extend(int(value) for value in rng.permutation(args.init_state_ids))
        schedule = schedule[: args.episodes]
        flow_noise_schedule = [None] * args.episodes
    _atomic_json(output / "experiment_config.json", {
        "schema": schema_for_selector(args.selector_version),
        "training": False,
        "selector_version": args.selector_version,
        "selector_uses_physical_state": False,
        "selector_uses_gripper_event": False,
        "fixed_episode_count_not_stopped_by_alarm": True,
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "seed": args.seed,
        "environment_seed": args.environment_seed,
        "init_state_schedule": schedule,
        "flow_noise_seed_schedule": flow_noise_schedule,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "persistence": args.persistence,
        "max_reference_advance": args.max_reference_advance,
        "healthy_reference": (
            None if bank is None else str(args.healthy_reference.resolve())
        ),
        "healthy_reference_identities": (
            [] if bank is None else list(bank.identities)
        ),
        "dynamics_calibration": (
            None
            if args.dynamics_calibration is None
            else str(args.dynamics_calibration.resolve())
        ),
        "dynamics_calibration_sha256": (
            None
            if args.dynamics_calibration is None
            else dynamics_sha256_file(args.dynamics_calibration.resolve())
        ),
        "self_reference_config": (
            str(args.self_reference_config.resolve())
            if self_reference_config is not None
            else None
        ),
        "self_reference_config_sha256": (
            dynamics_sha256_file(args.self_reference_config.resolve())
            if self_reference_config is not None
            else None
        ),
        "post_alarm_policy": "continue unchanged to success or horizon",
    })

    summaries = []
    started = time.time()
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        validate_policy_suite(client.metadata, args.benchmark)
        if client.metadata.get("episode_id_key") != "episode_id":
            raise RuntimeError("server does not advertise episode_id")
        if client.metadata.get("full_router_probs_response_key") != FULL_PROBS_KEY:
            raise RuntimeError("server does not return full HB probabilities")
        _atomic_json(output / "server_metadata.json", dict(client.metadata))
        for episode_index, (init_state_id, flow_noise_seed) in enumerate(
            zip(schedule, flow_noise_schedule)
        ):
            summary = run_episode(
                args,
                client,
                bank,
                output,
                episode_index,
                init_state_id,
                dynamics_calibration,
                self_reference_config,
                flow_noise_seed,
            )
            summaries.append(summary)
            _atomic_json(output / "progress.json", {
                "schema": schema_for_selector(args.selector_version),
                "completed_episodes": len(summaries),
                "planned_episodes": args.episodes,
                "alarm_episodes": sum(item["alarm_count"] > 0 for item in summaries),
                "last_episode": summary,
            })
    manifest = {
        "schema": schema_for_selector(args.selector_version),
        "status": "complete",
        "training": False,
        "selector_version": args.selector_version,
        "selector_uses_physical_state": False,
        "online_alarm": True,
        "fixed_episode_count": True,
        "completed_episodes": len(summaries),
        "alarm_episodes": sum(item["alarm_count"] > 0 for item in summaries),
        "successes": sum(bool(item["success"]) for item in summaries),
        "failures": sum(not bool(item["success"]) for item in summaries),
        "episodes": summaries,
        "videos": [video for item in summaries for video in item["videos"]],
        "wall_s": round(time.time() - started, 3),
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete",
        "episodes": len(summaries),
        "alarm_episodes": manifest["alarm_episodes"],
        "successes": manifest["successes"],
        "failures": manifest["failures"],
        "videos": manifest["videos"],
        "output": str(output),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
