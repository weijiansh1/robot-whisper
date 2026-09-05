#!/usr/bin/env python3
"""Run HiMoE online, fire a frozen train-free belief alarm, and keep acting.

The alarm uses only the current/previous HB routing tensors and a frozen bank
of successful, phase-aligned routes.  Robot motion, rewards, success, and
future observations are recorded only for retrospective alarm validation.
"""

from __future__ import annotations

import argparse
import hashlib
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
from himoe_libero_bridge.protocol import (  # noqa: E402
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)
from rolling_star_collect import (  # noqa: E402
    _array_sha256,
    _atomic_json,
    _atomic_npz,
    _frame,
    _sha256_file,
    _write_video,
)
from rollout_with_routes import sim_joint_layout  # noqa: E402
from trainfree_belief_selector import (  # noqa: E402
    HealthyThresholds,
    evaluate_stale_belief,
)


SCHEMA = "himoe.online_belief_alarm.v1"
FULL_PROBS_KEY = "recorder/hb_router_probs"
EXPECTED_ROUTE_SHAPE = (8, 10, 11, 32)
MISSING_RELATIVE_QUERY = np.iinfo(np.int16).min
POT_STATE_SLICE = slice(10, 13)


def parse_int_list(value: str) -> Tuple[int, ...]:
    result = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise argparse.ArgumentTypeError(
            "expected distinct comma-separated non-negative integers"
        )
    return result


def file_digest(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class HealthyReference:
    def __init__(self, path: pathlib.Path) -> None:
        with np.load(str(path), allow_pickle=False) as archive:
            schema = str(archive["schema"].item())
            if schema != "himoe.online_healthy_reference.v1":
                raise RuntimeError("unsupported healthy-reference schema %r" % schema)
            if bool(archive["training"].item()):
                raise RuntimeError("online reference unexpectedly declares training")
            if bool(archive["failure_labels_used"].item()):
                raise RuntimeError("online reference used failure labels")
            relatives = np.asarray(archive["relative_queries"], dtype=np.int16)
            banks = np.asarray(archive["route_banks"], dtype=np.float32)
            thresholds = np.asarray(archive["thresholds"], dtype=np.float64)
            names = [str(item) for item in archive["threshold_names"].tolist()]
            candidates = np.asarray(archive["control_candidates"], dtype=np.int32)
            episode_ids = np.asarray(archive["control_episode_ids"], dtype=np.int64)
        if banks.ndim != 6 or banks.shape[0] != len(relatives):
            raise RuntimeError("healthy route bank must be [P,N,L,F,U,E]")
        if tuple(banks.shape[2:]) != EXPECTED_ROUTE_SHAPE:
            raise RuntimeError("healthy routes have shape %s" % (banks.shape,))
        expected_names = [
            "layer5_gap_upper",
            "back_chunk_jump_upper",
            "front_action_distance_upper",
        ]
        if names != expected_names or thresholds.shape != (len(relatives), 3):
            raise RuntimeError("healthy-reference threshold table is malformed")
        self.path = path.resolve()
        self.sha256 = file_digest(path)
        self.relatives = relatives
        self.banks = banks
        self.thresholds = thresholds
        self.control_candidates = candidates
        self.control_episode_ids = episode_ids
        self.lookup = {int(relative): index for index, relative in enumerate(relatives)}

    def evaluate(
        self, relative: int, current: np.ndarray, previous: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        index = self.lookup.get(int(relative))
        if index is None:
            return None
        row = self.thresholds[index]
        limits = HealthyThresholds(
            layer5_gap_upper=float(row[0]),
            back_chunk_jump_upper=float(row[1]),
            front_action_distance_upper=float(row[2]),
        )
        decision = evaluate_stale_belief(
            current,
            previous,
            self.banks[index],
            limits,
        )
        output = decision.to_dict()
        output.update({
            "relative_query": int(relative),
            "layer5_gap_upper": limits.layer5_gap_upper,
            "back_chunk_jump_upper": limits.back_chunk_jump_upper,
            "front_action_distance_upper": limits.front_action_distance_upper,
        })
        return output


def request_with_full_routes(
    client: PolicyClient,
    policy_observation: Mapping[str, Any],
    noise: np.ndarray,
    episode_id: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    request = dict(policy_observation)
    request[FLOW_NOISE_KEY] = np.ascontiguousarray(noise, dtype=np.float32)
    request["episode_id"] = int(episode_id)
    response = validate_action_response(client.infer(request))
    if response.get(FLOW_NOISE_SHA256_KEY) != _array_sha256(noise):
        raise RuntimeError("server did not acknowledge the exact flow-noise tensor")
    if FULL_PROBS_KEY not in response:
        raise RuntimeError(
            "policy response has no %s; start the server with --return-full-probs"
            % FULL_PROBS_KEY
        )
    actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
    route = np.asarray(response[FULL_PROBS_KEY], dtype=np.float32)
    if actions.shape != (10, 7):
        raise RuntimeError("unexpected action shape %s" % (actions.shape,))
    if route.shape != EXPECTED_ROUTE_SHAPE:
        raise RuntimeError("unexpected route shape %s" % (route.shape,))
    if not np.isfinite(route).all():
        raise RuntimeError("router probabilities contain NaN or infinity")
    if not np.allclose(route.sum(axis=-1), 1.0, atol=2e-3, rtol=2e-3):
        raise RuntimeError("router probabilities do not sum to one")
    return actions, route, float(response.get("server/inference_ms", np.nan))


def random_noise(
    seed: int, task_id: int, init_state_id: int, episode_index: int, query: int
) -> np.ndarray:
    sequence = np.random.SeedSequence(
        [int(seed), int(task_id), int(init_state_id), int(episode_index), int(query)]
    )
    return np.random.default_rng(sequence).standard_normal(FLOW_NOISE_SHAPE).astype(
        np.float32
    )


def gripper_aperture(observation: Mapping[str, Any]) -> float:
    return float(np.abs(np.asarray(observation["robot0_gripper_qpos"])).sum())


def physical_alarm_diagnostic(
    closure_eef: np.ndarray,
    closure_pot: np.ndarray,
    current_eef: np.ndarray,
    current_pot: np.ndarray,
) -> Dict[str, Any]:
    eef_displacement = float(np.linalg.norm(current_eef - closure_eef))
    pot_displacement = float(np.linalg.norm(current_pot - closure_pot))
    separation = float(np.linalg.norm(current_eef - current_pot))
    coupling = pot_displacement / max(eef_displacement, 1e-12)
    return {
        "eef_displacement_from_closure_m": eef_displacement,
        "pot_displacement_from_closure_m": pot_displacement,
        "eef_pot_separation_m": separation,
        "pot_to_eef_displacement_ratio": coupling,
        "failed_grasp_geometry": bool(
            eef_displacement >= 0.03 and coupling < 0.5 and separation >= 0.10
        ),
        "role": "retrospective_only_not_used_by_alarm",
    }


def draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.62,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image,
        (x - 5, y - height - 5),
        (x + width + 5, y + baseline + 5),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def annotate_frames(
    frames: Sequence[np.ndarray],
    frame_queries: Sequence[int],
    frame_action_positions: Sequence[int],
    init_state_id: int,
    closure_query: Optional[int],
    decisions: Mapping[int, Mapping[str, Any]],
    alarm_queries: Sequence[int],
) -> List[np.ndarray]:
    alarms = set(int(query) for query in alarm_queries)
    first_alarm = min(alarms) if alarms else None
    output: List[np.ndarray] = []
    for source, query, position in zip(
        frames, frame_queries, frame_action_positions
    ):
        image = np.asarray(source, dtype=np.uint8).copy()
        draw_label(
            image,
            "init=%02d  query=%02d  action=%02d" % (init_state_id, query, position),
            (14, 28),
            (255, 255, 255),
            0.55,
        )
        next_y = 58
        if closure_query is not None and query >= closure_query:
            draw_label(
                image,
                "near-pot gripper closure: q%02d" % closure_query,
                (14, next_y),
                (255, 220, 0),
                0.55,
            )
            next_y += 30
        if query in alarms:
            cv2.rectangle(
                image,
                (3, 3),
                (image.shape[1] - 4, image.shape[0] - 4),
                (255, 0, 0),
                thickness=8,
            )
            draw_label(
                image,
                "MoE ALARM - chunk still executed",
                (14, next_y),
                (255, 80, 80),
                0.68,
            )
            next_y += 34
        elif first_alarm is not None and query > first_alarm:
            draw_label(
                image,
                "AFTER ALARM - continuing unchanged",
                (14, next_y),
                (255, 100, 100),
                0.58,
            )
            next_y += 31
        decision = decisions.get(int(query))
        if decision is not None:
            draw_label(
                image,
                "MoE ratios gap %.2f  jump %.2f  action %.2f"
                % (
                    decision["layer5_gap_ratio"],
                    decision["back_chunk_jump_ratio"],
                    decision["front_action_distance_ratio"],
                ),
                (14, next_y),
                (255, 255, 255),
                0.52,
            )
        output.append(image)
    return output


def load_replay_noise(path: Optional[pathlib.Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    with np.load(str(path), allow_pickle=False) as archive:
        noise = np.asarray(archive["flow_noise"], dtype=np.float32)
    if noise.ndim != 3 or tuple(noise.shape[1:]) != FLOW_NOISE_SHAPE:
        raise RuntimeError("replay flow_noise has shape %s" % (noise.shape,))
    return noise


def run_episode(
    args: argparse.Namespace,
    client: PolicyClient,
    reference: HealthyReference,
    episode_index: int,
    init_state_id: int,
    replay_noise: Optional[np.ndarray],
    output: pathlib.Path,
) -> Dict[str, Any]:
    episode_dir = output / (
        "episode_%03d_init_%02d_%s"
        % (
            episode_index,
            init_state_id,
            "replay" if replay_noise is not None else "random",
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

    routes: List[np.ndarray] = []
    actions_all: List[np.ndarray] = []
    noises: List[np.ndarray] = []
    executed_masks: List[np.ndarray] = []
    policy_states: List[np.ndarray] = []
    query_sim_states: List[np.ndarray] = []
    action_steps_at_query: List[int] = []
    inference_ms: List[float] = []
    selector_relative: List[int] = []
    selector_reject: List[bool] = []
    selector_metrics: List[List[float]] = []
    decisions: Dict[int, Dict[str, Any]] = {}
    alarms: List[Dict[str, Any]] = []

    frames: List[np.ndarray] = []
    frame_queries: List[int] = []
    frame_action_positions: List[int] = []
    control_sim_states: List[np.ndarray] = []
    control_eef: List[np.ndarray] = []
    control_gripper: List[np.ndarray] = []
    control_actions: List[np.ndarray] = []
    control_query_indices: List[int] = []
    control_success: List[bool] = []

    closure: Optional[Dict[str, Any]] = None
    action_steps = 0
    success = False
    started = time.time()
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
        previous_aperture = gripper_aperture(observation)

        max_queries = (args.max_steps + args.replan_steps - 1) // args.replan_steps
        for query in range(max_queries):
            if action_steps >= args.max_steps or success:
                break
            policy_observation = build_policy_observation(observation, prompt)
            policy_state = np.asarray(
                policy_observation["observation/state"], dtype=np.float32
            )
            current_sim = np.asarray(environment.get_sim_state(), dtype=np.float32)
            if replay_noise is not None and query < len(replay_noise):
                noise = replay_noise[query].copy()
                noise_source = "archived_replay"
            else:
                noise = random_noise(
                    args.seed, args.task_id, init_state_id, episode_index, query
                )
                noise_source = "fresh_random"
            actions, route, server_ms = request_with_full_routes(
                client,
                policy_observation,
                noise,
                args.episode_id_base + episode_index,
            )

            routes.append(route.astype(np.float16))
            actions_all.append(actions)
            noises.append(noise)
            policy_states.append(policy_state)
            query_sim_states.append(current_sim)
            action_steps_at_query.append(action_steps)
            inference_ms.append(server_ms)

            relative = MISSING_RELATIVE_QUERY
            decision: Optional[Dict[str, Any]] = None
            if closure is not None and len(routes) >= 2:
                relative = query - int(closure["query"])
                decision = reference.evaluate(
                    relative,
                    route,
                    routes[-2].astype(np.float32),
                )
            selector_relative.append(relative)
            selector_reject.append(
                bool(decision is not None and decision["reject_stale_chunk"])
            )
            if decision is None:
                selector_metrics.append([np.nan] * 6)
            else:
                decision["query"] = query
                decisions[query] = decision
                selector_metrics.append([
                    float(decision["layer5_gap"]),
                    float(decision["back_chunk_jump"]),
                    float(decision["front_action_distance"]),
                    float(decision["layer5_gap_ratio"]),
                    float(decision["back_chunk_jump_ratio"]),
                    float(decision["front_action_distance_ratio"]),
                ])
                if bool(decision["reject_stale_chunk"]):
                    diagnostic = physical_alarm_diagnostic(
                        np.asarray(closure["eef_xyz"], dtype=np.float32),
                        np.asarray(closure["pot_xyz"], dtype=np.float32),
                        policy_state[:3],
                        current_sim[POT_STATE_SLICE],
                    )
                    alarm = dict(decision)
                    alarm.update({
                        "action_step_before_execution": action_steps,
                        "physical_diagnostic": diagnostic,
                        "intervention": "none_chunk_executed_unchanged",
                    })
                    alarms.append(alarm)
                    print(
                        "ALARM episode=%d init=%d q=%d rel=%+d "
                        "ratios=(%.3f, %.3f, %.3f) failed_grasp_geometry=%s; "
                        "continuing"
                        % (
                            episode_index,
                            init_state_id,
                            query,
                            relative,
                            decision["layer5_gap_ratio"],
                            decision["back_chunk_jump_ratio"],
                            decision["front_action_distance_ratio"],
                            diagnostic["failed_grasp_geometry"],
                        ),
                        flush=True,
                    )

            mask = np.zeros((args.replan_steps,), dtype=np.bool_)
            take = min(args.replan_steps, len(actions), args.max_steps - action_steps)
            for action_index in range(take):
                observation, _reward, _done, _info = environment.step(
                    actions[action_index].tolist()
                )
                mask[action_index] = True
                action_steps += 1
                success = bool(environment.check_success())

                control_actions.append(actions[action_index].copy())
                control_query_indices.append(query)
                control_success.append(success)
                sim_after = np.asarray(environment.get_sim_state(), dtype=np.float32)
                eef_after = np.asarray(
                    observation["robot0_eef_pos"], dtype=np.float32
                )
                gripper_after = np.asarray(
                    observation["robot0_gripper_qpos"], dtype=np.float32
                )
                control_sim_states.append(sim_after)
                control_eef.append(eef_after)
                control_gripper.append(gripper_after)
                frames.append(_frame(observation))
                frame_queries.append(query)
                frame_action_positions.append(action_index + 1)

                aperture_after = float(np.abs(gripper_after).sum())
                if (
                    closure is None
                    and previous_aperture >= args.close_aperture_threshold
                    and aperture_after < args.close_aperture_threshold
                ):
                    pot_after = sim_after[POT_STATE_SLICE].copy()
                    distance = float(np.linalg.norm(eef_after - pot_after))
                    if distance < args.near_object_threshold:
                        closure = {
                            "query": query,
                            "action_position_1based": action_index + 1,
                            "control_step": action_steps,
                            "eef_pot_distance_m": distance,
                            "aperture_before": previous_aperture,
                            "aperture_after": aperture_after,
                            "eef_xyz": eef_after.tolist(),
                            "pot_xyz": pot_after.tolist(),
                        }
                        print(
                            "CLOSURE episode=%d init=%d q=%d action=%d distance=%.3f"
                            % (
                                episode_index,
                                init_state_id,
                                query,
                                action_index + 1,
                                distance,
                            ),
                            flush=True,
                        )
                previous_aperture = aperture_after
                if success:
                    break
            executed_masks.append(mask)
            print(
                "episode=%d q=%02d source=%s rel=%s alarm=%s success=%s"
                % (
                    episode_index,
                    query,
                    noise_source,
                    "NA" if relative == MISSING_RELATIVE_QUERY else "%+d" % relative,
                    selector_reject[-1],
                    success,
                ),
                flush=True,
            )

        alarm_queries = [int(alarm["query"]) for alarm in alarms]
        annotated = annotate_frames(
            frames,
            frame_queries,
            frame_action_positions,
            init_state_id,
            None if closure is None else int(closure["query"]),
            decisions,
            alarm_queries,
        )
        video_records: List[Dict[str, Any]] = []
        if alarms:
            video_dir = episode_dir / "videos"
            clean_path = video_dir / "alarm_clean_full.mp4"
            annotated_path = video_dir / "alarm_annotated_full.mp4"
            _write_video(clean_path, frames, fps=args.fps)
            _write_video(annotated_path, annotated, fps=args.fps)
            first_alarm_query = alarm_queries[0]
            matching = [
                index
                for index, query in enumerate(frame_queries)
                if int(query) == first_alarm_query
            ]
            first_alarm_frame = matching[0] if matching else 0
            left = max(0, first_alarm_frame - args.clip_pre_frames)
            right = min(len(annotated), first_alarm_frame + args.clip_post_frames)
            clip_path = video_dir / "alarm_annotated_clip.mp4"
            _write_video(clip_path, annotated[left:right], fps=args.fps)
            for role, path in (
                ("clean_full", clean_path),
                ("annotated_full", annotated_path),
                ("annotated_clip", clip_path),
            ):
                video_records.append({
                    "role": role,
                    "path": str(path.relative_to(output)),
                    "sha256": _sha256_file(path),
                    "bytes": path.stat().st_size,
                })

        selector_metric_names = np.asarray([
            "layer5_gap",
            "back_chunk_jump",
            "front_action_distance",
            "layer5_gap_ratio",
            "back_chunk_jump_ratio",
            "front_action_distance_ratio",
        ])
        arrays_path = episode_dir / "trajectory_and_routes.npz"
        _atomic_npz(arrays_path, {
            "hb_router_probs": np.stack(routes).astype(np.float16),
            "flow_noise": np.stack(noises).astype(np.float32),
            "action_chunks": np.stack(actions_all).astype(np.float32),
            "action_executed": np.stack(executed_masks),
            "policy_state": np.stack(policy_states).astype(np.float32),
            "query_sim_state": np.stack(query_sim_states).astype(np.float32),
            "action_steps_at_query": np.asarray(action_steps_at_query, dtype=np.int32),
            "server_inference_ms": np.asarray(inference_ms, dtype=np.float32),
            "selector_relative_query": np.asarray(selector_relative, dtype=np.int16),
            "selector_reject": np.asarray(selector_reject, dtype=np.bool_),
            "selector_metric_names": selector_metric_names,
            "selector_metrics": np.asarray(selector_metrics, dtype=np.float64),
            "control_sim_state": np.stack(control_sim_states).astype(np.float32),
            "control_eef_position": np.stack(control_eef).astype(np.float32),
            "control_gripper_qpos": np.stack(control_gripper).astype(np.float32),
            "control_action": np.asarray(control_actions, dtype=np.float32),
            "control_query_index": np.asarray(control_query_indices, dtype=np.int32),
            "control_success_after_step": np.asarray(control_success, dtype=np.bool_),
            "frame_query_index": np.asarray(frame_queries, dtype=np.int16),
            "frame_action_position": np.asarray(frame_action_positions, dtype=np.int8),
        })
        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "episode_index": episode_index,
            "episode_id": args.episode_id_base + episode_index,
            "benchmark": args.benchmark,
            "task_id": args.task_id,
            "task_name": str(task.name),
            "prompt": prompt,
            "init_state_id": init_state_id,
            "noise_mode": "archived_replay" if replay_noise is not None else "fresh_random",
            "success": success,
            "action_steps": action_steps,
            "inference_calls": len(routes),
            "continued_after_alarm": bool(
                alarms and action_steps > int(alarms[0]["action_step_before_execution"])
            ),
            "post_alarm_action_steps": (
                0
                if not alarms
                else action_steps - int(alarms[0]["action_step_before_execution"])
            ),
            "closure": closure,
            "alarms": alarms,
            "alarm_count": len(alarms),
            "videos": video_records,
            "trajectory_npz": str(arrays_path.relative_to(output)),
            "trajectory_npz_sha256": _sha256_file(arrays_path),
            "wall_s": round(time.time() - started, 3),
        }
        _atomic_json(episode_dir / "summary.json", summary)
        print(
            "EPISODE COMPLETE index=%d init=%d success=%s alarms=%d steps=%d"
            % (episode_index, init_state_id, success, len(alarms), action_steps),
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
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--target-alarm-episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--episode-id-base", type=int, default=1_600_000_000)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=520)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clip-pre-frames", type=int, default=60)
    parser.add_argument("--clip-post-frames", type=int, default=140)
    parser.add_argument("--close-aperture-threshold", type=float, default=0.05)
    parser.add_argument("--near-object-threshold", type=float, default=0.16)
    parser.add_argument("--libero-root", required=True)
    parser.add_argument(
        "--healthy-reference",
        type=pathlib.Path,
        default=PACKAGE_ROOT
        / "results/trainfree_belief_selector/online_healthy_reference.npz",
    )
    parser.add_argument(
        "--replay-noise",
        type=pathlib.Path,
        help="use this archive's flow_noise for episode zero; later episodes are fresh",
    )
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    args = parser.parse_args()

    if args.replan_steps != 10:
        raise ValueError("the checkpoint and online reference require replan_steps=10")
    if args.max_episodes <= 0 or args.target_alarm_episodes <= 0:
        raise ValueError("episode and target counts must be positive")
    if args.episode_id_base < 0 or args.episode_id_base + args.max_episodes > np.iinfo(np.int32).max:
        raise ValueError("episode IDs must fit int32")
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite %s" % output)
    output.mkdir(parents=True)
    reference = HealthyReference(args.healthy_reference.resolve())
    replay_noise = load_replay_noise(args.replay_noise)
    if replay_noise is not None and args.init_state_ids[0] != 0:
        raise ValueError("the archived candidate-06 replay requires init state 0 first")

    rng = np.random.default_rng(args.seed)
    init_schedule: List[int] = []
    while len(init_schedule) < args.max_episodes:
        init_schedule.extend(
            int(value) for value in rng.permutation(args.init_state_ids).tolist()
        )
    init_schedule = init_schedule[: args.max_episodes]
    if replay_noise is not None:
        init_schedule[0] = 0

    config = {
        "schema": SCHEMA,
        "training": False,
        "alarm_uses_physics": False,
        "alarm_uses_outcome": False,
        "alarm_inputs": [
            "current HB router probabilities",
            "previous HB router probabilities",
            "frozen phase-matched healthy route bank",
        ],
        "post_alarm_policy": "continue every generated action unchanged until success/horizon",
        "benchmark": args.benchmark,
        "task_id": args.task_id,
        "init_state_schedule": init_schedule,
        "max_episodes": args.max_episodes,
        "target_alarm_episodes": args.target_alarm_episodes,
        "seed": args.seed,
        "environment_seed": args.environment_seed,
        "max_steps": args.max_steps,
        "replan_steps": args.replan_steps,
        "healthy_reference": str(reference.path),
        "healthy_reference_sha256": reference.sha256,
        "replay_noise": None if args.replay_noise is None else str(args.replay_noise.resolve()),
        "replay_noise_sha256": None if args.replay_noise is None else file_digest(args.replay_noise),
    }
    _atomic_json(output / "experiment_config.json", config)

    summaries: List[Dict[str, Any]] = []
    alarm_episodes = 0
    started = time.time()
    with PolicyClient(
        host=args.host,
        port=args.port,
        connect_timeout=600.0,
        inference_timeout=args.inference_timeout,
    ) as client:
        validate_policy_suite(client.metadata, args.benchmark)
        if client.metadata.get("episode_id_key") != "episode_id":
            raise RuntimeError("server does not advertise the episode-id contract")
        if client.metadata.get("full_router_probs_response_key") != FULL_PROBS_KEY:
            raise RuntimeError("server does not return full router probabilities")
        _atomic_json(output / "server_metadata.json", dict(client.metadata))
        for episode_index, init_state_id in enumerate(init_schedule):
            use_replay = replay_noise if episode_index == 0 else None
            summary = run_episode(
                args,
                client,
                reference,
                episode_index,
                init_state_id,
                use_replay,
                output,
            )
            summaries.append(summary)
            alarm_episodes += int(summary["alarm_count"] > 0)
            _atomic_json(output / "progress.json", {
                "schema": SCHEMA,
                "completed_episodes": len(summaries),
                "alarm_episodes": alarm_episodes,
                "target_alarm_episodes": args.target_alarm_episodes,
                "last_episode": summary,
            })
            if alarm_episodes >= args.target_alarm_episodes:
                break

    manifest = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "online_alarm": True,
        "alarm_chunk_was_executed": True,
        "alarm_uses_physics": False,
        "completed_episodes": len(summaries),
        "alarm_episodes": alarm_episodes,
        "target_reached": alarm_episodes >= args.target_alarm_episodes,
        "episodes": summaries,
        "videos": [
            video
            for summary in summaries
            for video in summary.get("videos", [])
        ],
        "wall_s": round(time.time() - started, 3),
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps({
        "status": "complete",
        "completed_episodes": len(summaries),
        "alarm_episodes": alarm_episodes,
        "target_reached": manifest["target_reached"],
        "output": str(output),
        "videos": manifest["videos"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
