#!/usr/bin/env python3
"""Audit and summarize the GPU-4 online train-free alarm experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/online_belief_alarm_gpu4.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/online_belief_alarm"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def closure_geometry(episode_path: Path, episode: dict[str, Any]) -> dict[str, Any]:
    closure = episode.get("closure")
    if closure is None:
        return {
            "closure_eligible": False,
            "pot_max_displacement_m": None,
            "pot_final_displacement_m": None,
            "eef_max_displacement_m": None,
            "eef_pot_max_separation_m": None,
            "pot_max_lift_m": None,
        }
    with np.load(episode_path / "trajectory_and_routes.npz", allow_pickle=False) as archive:
        sim = np.asarray(archive["control_sim_state"], dtype=np.float64)
        eef = np.asarray(archive["control_eef_position"], dtype=np.float64)
    step = int(closure["control_step"])
    pot_origin = sim[step, 10:13]
    eef_origin = eef[step]
    pot_displacement = np.linalg.norm(sim[step:, 10:13] - pot_origin, axis=1)
    eef_displacement = np.linalg.norm(eef[step:] - eef_origin, axis=1)
    separation = np.linalg.norm(eef[step:] - sim[step:, 10:13], axis=1)
    return {
        "closure_eligible": True,
        "pot_max_displacement_m": float(pot_displacement.max()),
        "pot_final_displacement_m": float(pot_displacement[-1]),
        "eef_max_displacement_m": float(eef_displacement.max()),
        "eef_pot_max_separation_m": float(separation.max()),
        "pot_max_lift_m": float((sim[step:, 12] - pot_origin[2]).max()),
    }


def classify_episode(episode: dict[str, Any], geometry: dict[str, Any]) -> str:
    alarm = int(episode["alarm_count"]) > 0
    if not geometry["closure_eligible"]:
        return "out_of_scope_no_near_pot_closure"
    if alarm and bool(episode["success"]):
        return "false_positive_successful_grasp"
    if alarm and not bool(episode["success"]):
        confirmed = any(
            bool(item["physical_diagnostic"]["failed_grasp_geometry"])
            for item in episode["alarms"]
        )
        return (
            "alerted_failed_grasp_geometry_confirmed"
            if confirmed
            else "alerted_failure_without_geometry_confirmation"
        )
    if bool(episode["success"]):
        return "eligible_success_no_alarm"
    return "eligible_failure_no_alarm"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assert config["training"] is False
    output = args.output.resolve()
    table_dir = output / "tables"

    episode_rows: list[dict[str, Any]] = []
    alarm_rows: list[dict[str, Any]] = []
    client_route_parts: list[np.ndarray] = []
    capture_spans: list[dict[str, Any]] = []
    video_count = 0
    capture_start = 0
    episode_id_occurrences: dict[int, int] = {}

    for run_spec in config["runs_in_server_request_order"]:
        run_path = (PACKAGE_ROOT / run_spec["path"]).resolve()
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "complete"
        assert manifest["training"] is False
        assert manifest["online_alarm"] is True
        assert manifest["alarm_chunk_was_executed"] is True
        for episode in manifest["episodes"]:
            episode_path = run_path / Path(episode["trajectory_npz"]).parent
            with np.load(
                run_path / episode["trajectory_npz"], allow_pickle=False
            ) as archive:
                route = np.asarray(archive["hb_router_probs"])
                reject = np.asarray(archive["selector_reject"], dtype=bool)
                executed = np.asarray(archive["action_executed"], dtype=bool)
            assert route.shape[0] == int(episode["inference_calls"])
            assert int(reject.sum()) == int(episode["alarm_count"])
            for alarm in episode["alarms"]:
                query = int(alarm["query"])
                assert reject[query]
                assert executed[query].any()
                assert alarm["intervention"] == "none_chunk_executed_unchanged"

            client_route_parts.append(route)
            capture_end = capture_start + len(route)
            episode_id = int(episode["episode_id"])
            episode_id_occurrences[episode_id] = (
                episode_id_occurrences.get(episode_id, 0) + 1
            )
            capture_spans.append({
                "run_role": run_spec["role"],
                "episode_index": int(episode["episode_index"]),
                "episode_id": episode_id,
                "first_control_step": capture_start,
                "last_control_step": capture_end - 1,
                "rows": len(route),
            })
            capture_start = capture_end

            geometry = closure_geometry(episode_path, episode)
            classification = classify_episode(episode, geometry)
            first_alarm = episode["alarms"][0] if episode["alarms"] else None
            episode_rows.append({
                "run_role": run_spec["role"],
                "episode_index": int(episode["episode_index"]),
                "init_state_id": int(episode["init_state_id"]),
                "noise_mode": episode["noise_mode"],
                "success": bool(episode["success"]),
                "action_steps": int(episode["action_steps"]),
                "closure_query": (
                    None if episode["closure"] is None else episode["closure"]["query"]
                ),
                "alarm_count": int(episode["alarm_count"]),
                "first_alarm_query": None if first_alarm is None else first_alarm["query"],
                "first_alarm_relative_query": (
                    None if first_alarm is None else first_alarm["relative_query"]
                ),
                "continued_after_alarm": bool(episode["continued_after_alarm"]),
                "post_alarm_action_steps": int(episode["post_alarm_action_steps"]),
                "classification": classification,
                **geometry,
            })
            for alarm in episode["alarms"]:
                diagnostic = alarm["physical_diagnostic"]
                alarm_rows.append({
                    "run_role": run_spec["role"],
                    "episode_index": int(episode["episode_index"]),
                    "init_state_id": int(episode["init_state_id"]),
                    "episode_success": bool(episode["success"]),
                    "query": int(alarm["query"]),
                    "relative_query": int(alarm["relative_query"]),
                    "layer5_gap_ratio": float(alarm["layer5_gap_ratio"]),
                    "back_chunk_jump_ratio": float(alarm["back_chunk_jump_ratio"]),
                    "front_action_distance_ratio": float(
                        alarm["front_action_distance_ratio"]
                    ),
                    "eef_displacement_from_closure_m": float(
                        diagnostic["eef_displacement_from_closure_m"]
                    ),
                    "pot_displacement_from_closure_m": float(
                        diagnostic["pot_displacement_from_closure_m"]
                    ),
                    "eef_pot_separation_m": float(
                        diagnostic["eef_pot_separation_m"]
                    ),
                    "failed_grasp_geometry": bool(
                        diagnostic["failed_grasp_geometry"]
                    ),
                    "continued_unchanged": True,
                })
            for video in episode["videos"]:
                video_path = run_path / video["path"]
                assert video_path.stat().st_size == int(video["bytes"])
                assert sha256_file(video_path) == video["sha256"]
                video_count += 1

    client_routes = np.concatenate(client_route_parts, axis=0)
    server_path = (PACKAGE_ROOT / config["server_capture"]).resolve()
    capture_summary = json.loads(
        (server_path / "capture_summary.json").read_text(encoding="utf-8")
    )
    store = zarr.open_group(str(server_path / "routes.zarr"), mode="r")
    server_routes = np.asarray(store["hb_router_probs"][:])
    control_steps = np.asarray(store["control_step"][:])
    assert capture_summary["control_steps"] == len(client_routes) == len(server_routes)
    assert np.array_equal(control_steps, np.arange(len(control_steps)))
    routes_exact = bool(np.array_equal(client_routes, server_routes))
    assert routes_exact
    assert capture_summary["hook_verify_failures"] == []
    assert capture_summary["return_full_probs"] is True

    fresh = [row for row in episode_rows if row["run_role"] == "fresh_random_test"]
    eligible = [row for row in fresh if row["closure_eligible"]]
    fresh_summary = {
        "episodes": len(fresh),
        "successes": sum(bool(row["success"]) for row in fresh),
        "failures": sum(not bool(row["success"]) for row in fresh),
        "closure_eligible_episodes": len(eligible),
        "alarm_episodes": sum(int(row["alarm_count"] > 0) for row in fresh),
        "alarm_success_episodes": sum(
            int(row["alarm_count"] > 0 and row["success"]) for row in fresh
        ),
        "alarm_failure_episodes": sum(
            int(row["alarm_count"] > 0 and not row["success"]) for row in fresh
        ),
        "out_of_scope_no_closure_failures": sum(
            int(not row["closure_eligible"] and not row["success"]) for row in fresh
        ),
        "confirmed_failed_grasp_alarm_episodes": sum(
            row["classification"] == "alerted_failed_grasp_geometry_confirmed"
            for row in fresh
        ),
    }
    duplicate_episode_ids = sorted(
        episode_id
        for episode_id, count in episode_id_occurrences.items()
        if count > 1
    )
    summary = {
        "schema": "himoe.online_belief_alarm_gpu4.summary.v1",
        "training": False,
        "selector_thresholds_changed_after_online_results": False,
        "hardware": config["hardware"],
        "fresh_random_test": fresh_summary,
        "all_runs": {
            "episodes": len(episode_rows),
            "alarm_episodes": sum(row["alarm_count"] > 0 for row in episode_rows),
            "alarm_events": len(alarm_rows),
            "videos": video_count,
        },
        "capture_audit": {
            "server_rows": len(server_routes),
            "client_rows": len(client_routes),
            "client_server_full_routes_exact": routes_exact,
            "route_shape": list(server_routes.shape),
            "control_steps_strictly_sequential": True,
            "hook_verify_failures": capture_summary["hook_verify_failures"],
            "spans": capture_spans,
            "duplicate_episode_ids_across_separate_client_runs": duplicate_episode_ids,
            "duplicate_id_resolution": (
                "global server control_step and declared run order uniquely identify rows"
            ),
        },
        "episode_results": episode_rows,
        "core_result": (
            "The frozen train-free selector produced one confirmed failed-grasp "
            "alarm episode and one successful-grasp false positive in four fresh "
            "random trajectories. Two additional failures never entered the "
            "near-pot closure phase and were outside this selector's scope."
        ),
        "deployment_status": "not_reliable_as_a_standalone_online_alarm",
    }
    write_csv(table_dir / "episode_summary.csv", episode_rows)
    write_csv(table_dir / "alarm_events.csv", alarm_rows)
    (output / "summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "OK",
        "fresh_random_test": fresh_summary,
        "capture_rows": len(server_routes),
        "client_server_routes_exact": routes_exact,
        "videos": video_count,
        "output": str(output / "summary.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
