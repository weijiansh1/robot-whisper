#!/usr/bin/env python3
"""Post-hoc audit of raw image/state novelty for counterfactual input pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Mapping

import cv2
import numpy as np
from PIL import Image


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = pathlib.Path(
    os.environ.get("HIMOE_VLA_WORKSPACE", PACKAGE_ROOT.parent)
).resolve()
sys.path.insert(0, str(WORKSPACE_ROOT / "himoe-libero-wrist-fix/src"))

from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task  # noqa: E402
from himoe_libero_bridge.preprocess import build_policy_observation  # noqa: E402


SCHEMA = "himoe.input_version_novelty.audit.v1"


def image_metrics(previous: np.ndarray, current: np.ndarray) -> Dict[str, float]:
    delta = np.asarray(current, np.float32) - np.asarray(previous, np.float32)
    absolute = np.abs(delta)
    return {
        "mae_u8": float(absolute.mean()),
        "rmse_u8": float(np.sqrt(np.mean(delta ** 2))),
        "changed_pixel_fraction": float(np.any(absolute > 0.0, axis=-1).mean()),
        "changed_gt8_fraction": float(np.any(absolute > 8.0, axis=-1).mean()),
    }


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: pathlib.Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def draw_label(image: np.ndarray, text: str) -> np.ndarray:
    output = np.asarray(image, np.uint8).copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def diff_image(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    difference = np.abs(
        np.asarray(current, np.int16) - np.asarray(previous, np.int16)
    )
    return np.asarray(np.clip(difference * 4, 0, 255), np.uint8)


def relation(value: float, controls: List[float]) -> str:
    if value < min(controls):
        return "below_all"
    if value > max(controls):
        return "above_all"
    return "inside"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "configs/input_version_counterfactual.json",
    )
    parser.add_argument(
        "--paired",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "results/input_version_counterfactual/formal_capture",
    )
    parser.add_argument(
        "--modality",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "results/input_version_counterfactual/modality_capture",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=PACKAGE_ROOT / "results/input_version_counterfactual/modality_analysis",
    )
    parser.add_argument(
        "--libero-root",
        type=pathlib.Path,
        default=WORKSPACE_ROOT
        / "himoe-vla-cache/himoe-libero-bridge/cache/upstream/LIBERO",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    paired_root = args.paired.expanduser().resolve()
    modality_root = args.modality.expanduser().resolve()
    output = args.output.expanduser().resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    records = json.loads((paired_root / "records.json").read_text(encoding="utf-8"))[
        "records"
    ]
    with np.load(
        str(modality_root / "modality_counterfactual_routes.npz"),
        allow_pickle=False,
    ) as archive:
        selected = np.asarray(archive["paired_record_index"], dtype=int)
    paired_manifest = json.loads(
        (paired_root / "manifest.json").read_text(encoding="utf-8")
    )
    raw_root = pathlib.Path(paired_manifest["raw_run_root"])
    snapshot_dir = raw_root / (
        "formal/worker%d/snapshot_%03d"
        % (int(config["worker"]), int(config["snapshot"]))
    )
    source_manifest = json.loads(
        (snapshot_dir / "manifest.json").read_text(encoding="utf-8")
    )
    prompt = str(source_manifest["prompt"])
    environment, _initial, _task, runtime_prompt = _load_task(
        EpisodeConfig(
            libero_root=str(args.libero_root.expanduser().resolve()),
            task_suite=str(config["benchmark"]),
            task_id=int(config["task_id"]),
            init_state_id=int(config["init_state_id"]),
            seed=int(config["environment_seed"]),
            settle_steps=10,
            max_steps=520,
            render_size=512,
        )
    )
    if str(runtime_prompt) != prompt:
        environment.close()
        raise RuntimeError("prompt mismatch")

    branch_cache: Dict[int, Dict[str, np.ndarray]] = {}
    rows: List[Dict[str, Any]] = []
    failure_panels: List[np.ndarray] = []
    try:
        for paired_index in selected:
            record = records[int(paired_index)]
            candidate = int(record["candidate"])
            if candidate not in branch_cache:
                source_path = snapshot_dir / ("candidate_%02d.npz" % candidate)
                with np.load(str(source_path), allow_pickle=False) as archive:
                    branch_cache[candidate] = {
                        "sim_state": np.asarray(archive["sim_state"]),
                        "policy_state": np.asarray(archive["policy_state"]),
                    }
            branch = branch_cache[candidate]
            previous_query = int(record["previous_query"])
            current_query = int(record["current_query"])
            previous = build_policy_observation(
                environment.regenerate_obs_from_state(
                    branch["sim_state"][previous_query]
                ),
                prompt,
            )
            current = build_policy_observation(
                environment.regenerate_obs_from_state(branch["sim_state"][current_query]),
                prompt,
            )
            previous_state = branch["policy_state"][previous_query]
            current_state = branch["policy_state"][current_query]
            base = image_metrics(
                previous["observation/image"], current["observation/image"]
            )
            wrist = image_metrics(
                previous["observation/wrist_image"],
                current["observation/wrist_image"],
            )
            rows.append(
                {
                    "paired_record_index": int(paired_index),
                    "candidate": candidate,
                    "outcome": str(record["outcome"]),
                    "relative_query": int(record["relative_query"]),
                    "previous_query": previous_query,
                    "current_query": current_query,
                    "base_mae_u8": base["mae_u8"],
                    "base_rmse_u8": base["rmse_u8"],
                    "base_changed_pixel_fraction": base["changed_pixel_fraction"],
                    "base_changed_gt8_fraction": base["changed_gt8_fraction"],
                    "wrist_mae_u8": wrist["mae_u8"],
                    "wrist_rmse_u8": wrist["rmse_u8"],
                    "wrist_changed_pixel_fraction": wrist["changed_pixel_fraction"],
                    "wrist_changed_gt8_fraction": wrist["changed_gt8_fraction"],
                    "policy_state_delta_l2": float(
                        np.linalg.norm(current_state - previous_state)
                    ),
                    "eef_position_delta_m": float(
                        np.linalg.norm(current_state[:3] - previous_state[:3])
                    ),
                    "gripper_state_delta": float(
                        np.linalg.norm(current_state[6:] - previous_state[6:])
                    ),
                }
            )
            if str(record["outcome"]) == "failed_grasp":
                relative = int(record["relative_query"])
                panel = np.concatenate(
                    [
                        draw_label(previous["observation/image"], "r=%+d old base" % relative),
                        draw_label(current["observation/image"], "r=%+d new base" % relative),
                        draw_label(
                            diff_image(
                                previous["observation/image"], current["observation/image"]
                            ),
                            "base |diff| x4",
                        ),
                        draw_label(previous["observation/wrist_image"], "old wrist"),
                        draw_label(current["observation/wrist_image"], "new wrist"),
                        draw_label(
                            diff_image(
                                previous["observation/wrist_image"],
                                current["observation/wrist_image"],
                            ),
                            "wrist |diff| x4",
                        ),
                    ],
                    axis=1,
                )
                failure_panels.append(panel)
    finally:
        environment.close()

    write_csv(tables / "input_version_pixel_novelty_posthoc.csv", rows)
    if failure_panels:
        Image.fromarray(np.concatenate(failure_panels, axis=0)).save(
            figures / "failed_input_version_pairs.png"
        )
    metrics = [
        "base_mae_u8",
        "wrist_mae_u8",
        "base_changed_gt8_fraction",
        "wrist_changed_gt8_fraction",
        "policy_state_delta_l2",
        "eef_position_delta_m",
        "gripper_state_delta",
    ]
    route_path = tables / "modality_primary_metrics.csv"
    normalized_rows: List[Dict[str, Any]] = []
    if route_path.exists():
        route_lookup = {
            int(row["paired_record_index"]): row for row in read_csv(route_path)
        }
        for row in rows:
            route = route_lookup[int(row["paired_record_index"])]
            pixel_mae = 0.5 * (
                float(row["base_mae_u8"]) + float(row["wrist_mae_u8"])
            )
            state_delta = float(row["policy_state_delta_l2"])
            vision_route = float(route["back_action_vision_h"])
            proprio_route = float(route["back_action_proprio_h"])
            normalized_rows.append(
                {
                    "paired_record_index": int(row["paired_record_index"]),
                    "candidate": int(row["candidate"]),
                    "outcome": str(row["outcome"]),
                    "relative_query": int(row["relative_query"]),
                    "combined_pixel_mae_u8": pixel_mae,
                    "back_action_vision_effect_h": vision_route,
                    "vision_route_effect_per_pixel_mae": vision_route
                    / max(pixel_mae, 1e-12),
                    "policy_state_delta_l2": state_delta,
                    "back_action_proprio_effect_h": proprio_route,
                    "proprio_route_effect_per_state_l2": proprio_route
                    / max(state_delta, 1e-12),
                }
            )
        write_csv(
            tables / "route_response_per_input_novelty_posthoc.csv",
            normalized_rows,
        )
        metrics.extend(
            [
                "vision_route_effect_per_pixel_mae",
                "proprio_route_effect_per_state_l2",
            ]
        )
    timeline: Dict[str, Any] = {}
    relatives = sorted(set(int(row["relative_query"]) for row in rows))
    for relative in relatives:
        failure = [
            row
            for row in rows
            if row["outcome"] == "failed_grasp"
            and int(row["relative_query"]) == relative
        ][0]
        controls = [
            row
            for row in rows
            if row["outcome"] == "success"
            and int(row["relative_query"]) == relative
        ]
        normalized_failure = None
        normalized_controls: List[Dict[str, Any]] = []
        if normalized_rows:
            normalized_failure = [
                row
                for row in normalized_rows
                if row["outcome"] == "failed_grasp"
                and int(row["relative_query"]) == relative
            ][0]
            normalized_controls = [
                row
                for row in normalized_rows
                if row["outcome"] == "success"
                and int(row["relative_query"]) == relative
            ]
        timeline[str(relative)] = {}
        for metric in metrics:
            source_failure = (
                normalized_failure
                if metric
                in (
                    "vision_route_effect_per_pixel_mae",
                    "proprio_route_effect_per_state_l2",
                )
                else failure
            )
            source_controls = (
                normalized_controls
                if metric
                in (
                    "vision_route_effect_per_pixel_mae",
                    "proprio_route_effect_per_state_l2",
                )
                else controls
            )
            values = [float(row[metric]) for row in source_controls]
            timeline[str(relative)][metric] = {
                "failed_value": float(source_failure[metric]),
                "success_mean": float(np.mean(values)),
                "success_range": [float(min(values)), float(max(values))],
                "relation": relation(float(source_failure[metric]), values),
            }
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "role": "posthoc_confound_audit_not_used_by_alarm",
        "training": False,
        "records": len(rows),
        "failed_records": sum(row["outcome"] == "failed_grasp" for row in rows),
        "success_control_records": sum(row["outcome"] == "success" for row in rows),
        "route_response_normalized_by_raw_input_novelty": bool(normalized_rows),
        "timeline": timeline,
    }
    (output / "input_novelty_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
