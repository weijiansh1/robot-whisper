#!/usr/bin/env python3
"""Deterministically replay saved actions and render MoE-only review videos."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
ROUTE_CAPTURE = WORKSPACE_ROOT / "himoe-route-capture"
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-libero-wrist-fix/src"))
sys.path.insert(0, str(ROUTE_CAPTURE))

from himoe_libero_bridge.libero_runtime import (  # noqa: E402
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
)
from rolling_star_collect import (  # noqa: E402
    _atomic_json,
    _frame,
    _sha256_file,
    _write_video,
)

from collect_online_belief_alarms import draw_label  # noqa: E402


DEFAULT_RUN = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/gpu4_heldout_random_seed20260906"
)
DEFAULT_AUDIT = (
    PACKAGE_ROOT
    / "results/moe_only_online_alarm/failure_type_audit/episode_audit.csv"
)
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/moe_only_online_alarm/review_videos"


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def audit_rows(path: Path, run_role: str) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            int(row["episode_index"]): row
            for row in csv.DictReader(stream)
            if row["run_role"] == run_role
        }


def annotate(
    frame: np.ndarray,
    *,
    query: int,
    action_position: int,
    init_state: int,
    family: str,
    raw_queries: set[int],
    missed_queries: set[int],
) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8).copy()
    draw_label(
        image,
        f"init={init_state:02d} query={query:02d} action={action_position:02d}",
        (14, 28),
        (255, 255, 255),
        0.55,
    )
    draw_label(image, f"posthoc family: {family}", (14, 58), (255, 255, 255), 0.50)
    if query in raw_queries:
        cv2.rectangle(
            image,
            (3, 3),
            (image.shape[1] - 4, image.shape[0] - 4),
            (255, 220, 0),
            thickness=6,
        )
        draw_label(image, "RAW MoE REJECT (not an alarm)", (14, 90), (255, 220, 0), 0.61)
    else:
        draw_label(image, "NO FORMAL MoE ALARM", (14, 90), (255, 255, 255), 0.55)
    if query in missed_queries:
        draw_label(
            image,
            "POSTHOC missed-grasp proxy begins here",
            (14, 122),
            (255, 90, 255),
            0.58,
        )
    return image


def render_episode(
    run_root: Path,
    output: Path,
    episode_index: int,
    audit: dict[str, str],
    libero_root: Path,
) -> dict[str, Any]:
    matches = sorted(run_root.glob(f"episode_{episode_index:03d}_*"))
    if len(matches) != 1:
        raise ValueError(f"episode {episode_index}: expected one directory, found {len(matches)}")
    episode_dir = matches[0]
    summary = json.loads((episode_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_root / "experiment_config.json").read_text(encoding="utf-8"))
    with np.load(episode_dir / "trajectory_and_routes.npz", allow_pickle=False) as archive:
        actions = np.asarray(archive["control_action"], dtype=np.float32)
        query_index = np.asarray(archive["control_query_index"], dtype=np.int32)
        expected_states = np.asarray(archive["control_sim_state"], dtype=np.float32)
    raw_queries = set(json.loads(audit["raw_reject_queries"]))
    missed_queries = set(json.loads(audit["missed_grasp_queries"]))
    family = audit["failure_family"]
    init_state = int(summary["init_state_id"])

    environment, observation, _task, _prompt = _load_task(
        EpisodeConfig(
            task_suite=str(config["benchmark"]),
            task_id=int(config["task_id"]),
            init_state_id=init_state,
            seed=int(config["environment_seed"]),
            host="127.0.0.1",
            port=0,
            libero_root=str(libero_root),
            output_root=str(output),
            settle_steps=10,
            max_steps=int(config["max_steps"]),
            replan_steps=int(config["replan_steps"]),
            render_size=512,
            fps=20,
            inference_timeout=1.0,
        )
    )
    clean_frames: list[np.ndarray] = []
    annotated_frames: list[np.ndarray] = []
    observed_states: list[np.ndarray] = []
    success = False
    try:
        for _ in range(10):
            observation, _reward, _done, _info = environment.step(
                LIBERO_DUMMY_ACTION.tolist()
            )
        initial = _frame(observation)
        clean_frames.append(initial)
        annotated_frames.append(
            annotate(
                initial,
                query=-1,
                action_position=0,
                init_state=init_state,
                family=family,
                raw_queries=raw_queries,
                missed_queries=missed_queries,
            )
        )
        observed_states.append(np.asarray(environment.get_sim_state(), dtype=np.float32))
        previous_query = -1
        action_position = 0
        for step, action in enumerate(actions):
            query = int(query_index[step])
            action_position = action_position + 1 if query == previous_query else 1
            previous_query = query
            observation, _reward, _done, _info = environment.step(action.tolist())
            success = bool(environment.check_success())
            frame = _frame(observation)
            clean_frames.append(frame)
            annotated_frames.append(
                annotate(
                    frame,
                    query=query,
                    action_position=action_position,
                    init_state=init_state,
                    family=family,
                    raw_queries=raw_queries,
                    missed_queries=missed_queries,
                )
            )
            observed_states.append(
                np.asarray(environment.get_sim_state(), dtype=np.float32)
            )
    finally:
        environment.close()

    observed = np.stack(observed_states)
    if observed.shape != expected_states.shape:
        raise ValueError(
            f"episode {episode_index}: replay state shape {observed.shape} != {expected_states.shape}"
        )
    absolute_error = np.abs(observed - expected_states)
    episode_output = output / f"episode_{episode_index:03d}_init_{init_state:02d}_{family}"
    clean_path = episode_output / "clean_full.mp4"
    annotated_path = episode_output / "annotated_full.mp4"
    _write_video(clean_path, clean_frames, fps=20)
    _write_video(annotated_path, annotated_frames, fps=20)
    return {
        "episode_index": episode_index,
        "init_state_id": init_state,
        "failure_family": family,
        "source_success": bool(summary["success"]),
        "replay_success": success,
        "frames": len(clean_frames),
        "max_abs_sim_state_error": float(absolute_error.max()),
        "mean_abs_sim_state_error": float(absolute_error.mean()),
        "raw_reject_queries": sorted(raw_queries),
        "missed_grasp_queries": sorted(missed_queries),
        "formal_alarm_queries": json.loads(audit["formal_alarm_queries"]),
        "videos": [
            {
                "role": "clean_full_posthoc_replay",
                "path": str(clean_path.relative_to(PACKAGE_ROOT)),
                "sha256": _sha256_file(clean_path),
                "bytes": clean_path.stat().st_size,
            },
            {
                "role": "annotated_full_posthoc_replay",
                "path": str(annotated_path.relative_to(PACKAGE_ROOT)),
                "sha256": _sha256_file(annotated_path),
                "bytes": annotated_path.stat().st_size,
            },
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--episodes", type=parse_int_list, default=[1, 3, 7])
    parser.add_argument(
        "--libero-root",
        type=Path,
        default=WORKSPACE_ROOT / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows = audit_rows(args.audit.resolve(), "heldout_gpu4")
    results = [
        render_episode(
            run_root,
            output,
            episode_index,
            rows[episode_index],
            args.libero_root.resolve(),
        )
        for episode_index in args.episodes
    ]
    manifest = {
        "schema": "himoe.moe_only_posthoc_review_videos.v1",
        "training": False,
        "online_alarm_video": False,
        "reason": "No formal persistence-2 alarm fired; these are deterministic posthoc action replays for visual audit.",
        "source_run": str(run_root.relative_to(PACKAGE_ROOT)),
        "episodes": results,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    raise SystemExit(main())
