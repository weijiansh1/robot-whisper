"""Build the action--physics--outcome map for same-snapshot candidate forks.

Two input contracts are supported:

* ``himoe.behavior_forks.v1`` contains full chunk trajectories, aligned semantic
  and raw contact events, and repeated common-random-number continuations.  A run
  is confirmatory only when scales and thresholds carry disjoint calibration
  snapshot hashes and the event/Q capability gates pass.
* the historical ``fork-pilot-n32`` directory contains endpoints and one future
  only.  It is accepted as a clearly labelled endpoint-proxy feasibility screen;
  it can never emit a strict control-equivalence or sensitive-fork claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from behavior_geometry import (
    PairThresholds,
    action_distance_matrix,
    action_rms_distance_matrix,
    classify_pair,
    contact_event_distance_matrix,
    empirical_thresholds,
    exact_event_equal_matrix,
    interval_is_equivalent,
    paired_binary_difference_interval,
    quaternion_distance_matrix,
    snapshot_bootstrap_mean,
    temporal_l2_distance_matrix,
)


FORMAL_SCHEMA = "himoe.behavior_forks.v1"
PHYSICAL_SCALE_FLOORS = {
    "eef_position_m": 1e-3,
    "eef_orientation_rad": 1e-2,
    "arm_qpos_native": 1e-2,
    "arm_qvel_native": 1e-2,
    "gripper_qpos_native": 1e-3,
    "object_translation_m": 1e-3,
    "object_orientation_rad": 1e-2,
    "object_joint_native": 1e-3,
    "object_qvel_native": 1e-3,
    # Legacy layouts cannot always split free-joint position and quaternion.
    "robot_qpos_native": 1e-2,
    "object_qpos_native": 1e-3,
    "eef_position_endpoint_m": 1e-3,
    "eef_orientation_endpoint_rad": 1e-2,
    "gripper_qpos_endpoint_native": 1e-3,
    "arm_qpos_endpoint_native": 1e-2,
    "arm_qvel_endpoint_native": 1e-2,
    "robot_qpos_endpoint_native": 1e-2,
    "object_translation_endpoint_m": 1e-3,
    "object_orientation_endpoint_rad": 1e-2,
    "object_joint_endpoint_native": 1e-3,
    "object_qvel_endpoint_native": 1e-3,
}
PAIR_CLASSES = (
    "command_redundancy",
    "control_equivalence",
    "physics_amplification",
    "critical_microdifference",
    "sensitive_fork",
)
PROXY_CLASSES = (
    "proxy_command_redundancy",
    "proxy_control_equivalence",
    "proxy_physics_amplification",
    "proxy_critical_microdifference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "formal", "legacy"), default="auto")
    parser.add_argument(
        "--legacy-layout",
        type=Path,
        default=Path("runs/objstate-t0s24-client/sim_layout.json"),
    )
    parser.add_argument("--thresholds-in", type=Path)
    parser.add_argument(
        "--physics-scales-in",
        type=Path,
        help="frozen component scales produced by a disjoint calibration run",
    )
    parser.add_argument(
        "--calibration-tags",
        type=Path,
        help="JSON list of snapshot tags used to fit thresholds; evaluation still reports all tags",
    )
    parser.add_argument("--near-quantile", type=float, default=0.2)
    parser.add_argument("--far-quantile", type=float, default=0.8)
    parser.add_argument("--gripper-weight", type=float, default=0.25)
    parser.add_argument("--q-equivalence", type=float, default=0.1)
    parser.add_argument("--q-difference", type=float, default=0.1)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--no-figure", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _snapshot_tag(episode: int, fork_step: int) -> str:
    return f"ep{episode}/t{fork_step}"


def _qpos_partitions(layout: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    nq = int(layout["nq"])
    robot: list[int] = []
    objects: list[int] = []
    for joint in layout["joints"]:
        # state_lo/state_hi index [time]+qpos, so subtract one for qpos-local indices.
        target = robot if bool(joint["is_robot"]) else objects
        target.extend(range(int(joint["state_lo"]) - 1, int(joint["state_hi"]) - 1))
    if sorted(robot + objects) != list(range(nq)):
        raise ValueError("joint layout does not partition every qpos coordinate exactly once")
    return np.asarray(robot, dtype=np.int64), np.asarray(objects, dtype=np.int64)


def _joint_type_name(joint: Mapping[str, Any]) -> str:
    raw = joint.get("joint_type")
    if isinstance(raw, Mapping):
        raw = raw.get("name", raw.get("raw", ""))
    if raw is None:
        raw = joint.get("joint_type_name", "")
    return str(raw).lower()


def _joint_qpos_bounds(joint: Mapping[str, Any]) -> tuple[int, int]:
    if "qpos_lo" in joint and "qpos_hi" in joint:
        return int(joint["qpos_lo"]), int(joint["qpos_hi"])
    return int(joint["state_lo"]) - 1, int(joint["state_hi"]) - 1


def _joint_qvel_bounds(joint: Mapping[str, Any], nq: int) -> tuple[int, int] | None:
    if "qvel_lo" in joint and "qvel_hi" in joint:
        return int(joint["qvel_lo"]), int(joint["qvel_hi"])
    if "qvel_state_lo" in joint and "qvel_state_hi" in joint:
        return int(joint["qvel_state_lo"]) - 1 - nq, int(joint["qvel_state_hi"]) - 1 - nq
    return None


def _layout_groups(layout: Mapping[str, Any]) -> dict[str, np.ndarray | list[slice]]:
    """Separate arm, gripper and object pose coordinates without mixing units."""

    nq = int(layout["nq"])
    arm: list[int] = []
    gripper: list[int] = []
    object_translation: list[int] = []
    task_object_translation: list[int] = []
    object_quaternion: list[slice] = []
    object_joint: list[int] = []
    arm_qvel: list[int] = []
    object_qvel: list[int] = []
    for joint in layout["joints"]:
        lo, hi = _joint_qpos_bounds(joint)
        name = str(joint["joint"])
        if bool(joint["is_robot"]):
            target = gripper if name.startswith("gripper0_") else arm
            target.extend(range(lo, hi))
        else:
            width = hi - lo
            kind = _joint_type_name(joint)
            if "free" in kind or (not kind and width == 7):
                object_translation.extend(range(lo, lo + 3))
                if bool(joint.get("is_task_object", False)):
                    task_object_translation.extend(range(lo, lo + 3))
                object_quaternion.append(slice(lo + 3, lo + 7))
            elif "ball" in kind or (not kind and width == 4):
                object_quaternion.append(slice(lo, hi))
            else:
                object_joint.extend(range(lo, hi))
        velocity = _joint_qvel_bounds(joint, nq)
        if velocity is not None:
            target_velocity = arm_qvel if bool(joint["is_robot"]) else object_qvel
            target_velocity.extend(range(*velocity))
    return {
        "arm": np.asarray(arm, dtype=np.int64),
        "gripper": np.asarray(gripper, dtype=np.int64),
        "object_translation": np.asarray(object_translation, dtype=np.int64),
        "task_object_translation": np.asarray(task_object_translation, dtype=np.int64),
        "object_quaternion": object_quaternion,
        "object_joint": np.asarray(object_joint, dtype=np.int64),
        "arm_qvel": np.asarray(arm_qvel, dtype=np.int64),
        "object_qvel": np.asarray(object_qvel, dtype=np.int64),
    }


def _mean_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("cannot average an empty matrix list")
    return np.mean(np.stack(matrices), axis=0)


def _layout_physical_components(
    sim_states: np.ndarray,
    eef_positions: np.ndarray,
    eef_quaternions: np.ndarray | None,
    gripper_qpos: np.ndarray,
    layout: Mapping[str, Any],
    endpoint: bool = False,
) -> dict[str, np.ndarray]:
    sim = np.asarray(sim_states, dtype=np.float64)
    nq = int(layout["nq"])
    nv = int(layout["nv"])
    if sim.ndim != 3 or sim.shape[-1] < 1 + nq + nv:
        raise ValueError("sim state shape is inconsistent with the joint layout")
    qpos = sim[..., 1 : 1 + nq]
    qvel = sim[..., 1 + nq :]
    groups = _layout_groups(layout)
    suffix = "_endpoint" if endpoint else ""
    components: dict[str, np.ndarray] = {
        f"eef_position{suffix}_m": temporal_l2_distance_matrix(eef_positions),
        f"gripper_qpos{suffix}_native": temporal_l2_distance_matrix(gripper_qpos),
    }
    if eef_quaternions is not None:
        components[f"eef_orientation{suffix}_rad"] = quaternion_distance_matrix(
            eef_quaternions
        )
    arm = np.asarray(groups["arm"])
    if len(arm):
        components[f"arm_qpos{suffix}_native"] = temporal_l2_distance_matrix(
            qpos[..., arm]
        )
    translations = np.asarray(groups["object_translation"])
    if len(translations):
        components[f"object_translation{suffix}_m"] = temporal_l2_distance_matrix(
            qpos[..., translations]
        )
    rotations = list(groups["object_quaternion"])
    if rotations:
        components[f"object_orientation{suffix}_rad"] = _mean_matrices(
            [quaternion_distance_matrix(qpos[..., rotation]) for rotation in rotations]
        )
    object_joint = np.asarray(groups["object_joint"])
    if len(object_joint):
        components[f"object_joint{suffix}_native"] = temporal_l2_distance_matrix(
            qpos[..., object_joint]
        )
    arm_velocity = np.asarray(groups["arm_qvel"])
    if len(arm_velocity):
        components[f"arm_qvel{suffix}_native"] = temporal_l2_distance_matrix(
            qvel[..., arm_velocity]
        )
    object_velocity = np.asarray(groups["object_qvel"])
    if len(object_velocity):
        components[f"object_qvel{suffix}_native"] = temporal_l2_distance_matrix(
            qvel[..., object_velocity]
        )
    return components


def _pairwise_binary_hamming(features: np.ndarray) -> np.ndarray:
    value = np.asarray(features, dtype=np.bool_)
    if value.ndim != 2 or len(value) < 2 or value.shape[1] == 0:
        raise ValueError("event features must have shape [candidate,event], event>=1")
    distance = np.mean(value[:, None] != value[None, :], axis=-1)
    np.fill_diagonal(distance, 0.0)
    return distance


def _coarse_event_matrices(
    arrays: Mapping[str, np.ndarray],
    layout: Mapping[str, Any],
    lift_threshold_m: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Derive contact/grasp/lift/collision/release events from captured primitives."""

    active = np.asarray(arrays["contact_active"], dtype=np.bool_)
    names = [str(value) for value in np.asarray(arrays["contact_pair_names"])]
    chunk_success = np.asarray(arrays["chunk_success"], dtype=np.bool_)
    k, time_count = active.shape[:2]
    geom_rows = layout.get("geom_roles")
    if not isinstance(geom_rows, list) or not geom_rows:
        # Preserve aligned timing and geom identity in pre-role v1 artifacts.
        features = np.concatenate([active, chunk_success[..., None]], axis=-1)
        flat = features.reshape(k, -1)
        labels = [f"raw_contact_tape:{name}" for name in names] + ["success_tape"]
        distance = _pairwise_binary_hamming(flat)
        equal = exact_event_equal_matrix(active, chunk_success)
        return distance, equal, labels, {
            "kind": "raw_contact_tape_fallback",
            "semantic_events_available": False,
            "aligned_time": True,
        }

    role = {str(row["geom"]): row for row in geom_rows}
    robot_object_pair = np.zeros(len(names), dtype=np.bool_)
    gripper_object_pair = np.zeros(len(names), dtype=np.bool_)
    collision_pair = np.zeros(len(names), dtype=np.bool_)
    grasp_keys: list[tuple[str, str] | None] = []
    mapped_pairs = 0
    for index, encoded in enumerate(names):
        parts = encoded.split(" <-> ", 1)
        if len(parts) != 2 or parts[0] not in role or parts[1] not in role:
            grasp_keys.append(None)
            continue
        mapped_pairs += 1
        left, right = role[parts[0]], role[parts[1]]
        roles = {str(left["role"]), str(right["role"])}
        robot_object_pair[index] = roles == {"robot", "object"}
        left_gripper = left.get("subtype") == "gripper" and right.get("role") == "object"
        right_gripper = right.get("subtype") == "gripper" and left.get("role") == "object"
        gripper_object_pair[index] = bool(left_gripper or right_gripper)
        collision_pair[index] = bool(
            roles in ({"robot", "support"}, {"robot"})
        )
        if left_gripper:
            side = left.get("finger_side")
            target = bool(right.get("is_task_object", False))
            grasp_keys.append(
                (str(side), str(right.get("object_body") or right["body"]))
                if side in ("left", "right") and target
                else None
            )
        elif right_gripper:
            side = right.get("finger_side")
            target = bool(left.get("is_task_object", False))
            grasp_keys.append(
                (str(side), str(left.get("object_body") or left["body"]))
                if side in ("left", "right") and target
                else None
            )
        else:
            grasp_keys.append(None)

    robot_object = (
        active[..., robot_object_pair].any(axis=-1)
        if np.any(robot_object_pair)
        else np.zeros((k, time_count), dtype=np.bool_)
    )
    gripper_object = (
        active[..., gripper_object_pair].any(axis=-1)
        if np.any(gripper_object_pair)
        else np.zeros((k, time_count), dtype=np.bool_)
    )
    collision = (
        active[..., collision_pair].any(axis=-1)
        if np.any(collision_pair)
        else np.zeros((k, time_count), dtype=np.bool_)
    )
    grasp = np.zeros((k, time_count), dtype=np.bool_)
    keyed_indices = [index for index, key in enumerate(grasp_keys) if key is not None]
    for candidate in range(k):
        for step in range(time_count):
            by_object: dict[str, set[str]] = defaultdict(set)
            for index in keyed_indices:
                if active[candidate, step, index]:
                    finger_side, object_key = grasp_keys[index]  # type: ignore[misc]
                    by_object[object_key].add(finger_side)
            grasp[candidate, step] = any(
                {"left", "right"}.issubset(sides) for sides in by_object.values()
            )
    release = np.zeros_like(grasp)
    release[:, 1:] = grasp[:, :-1] & ~grasp[:, 1:]

    sim = np.asarray(arrays["sim_states"], dtype=np.float64)
    qpos = sim[..., 1 : 1 + int(layout["nq"])]
    groups = _layout_groups(layout)
    target_translations = np.asarray(groups["task_object_translation"])
    translations = target_translations
    if len(translations):
        xyz = qpos[..., translations].reshape(k, time_count, -1, 3)
        lift = np.any(xyz[..., 2] - xyz[:, :1, :, 2] >= lift_threshold_m, axis=-1)
    else:
        lift = np.zeros((k, time_count), dtype=np.bool_)

    features = np.stack(
        [
            robot_object,
            gripper_object,
            grasp,
            lift,
            collision,
            release,
            chunk_success,
        ],
        axis=-1,
    )
    labels = [
        "robot_object_contact",
        "gripper_object_contact",
        "bilateral_grasp",
        f"object_lift_ge_{lift_threshold_m:g}m",
        "robot_support_or_self_collision",
        "release_after_bilateral_grasp",
        "success",
    ]
    semantic_distance = _pairwise_binary_hamming(features.reshape(k, -1))
    raw_distance = contact_event_distance_matrix(active)
    distance = 0.5 * (semantic_distance + raw_distance)
    semantic_equal = np.all(features[:, None] == features[None, :], axis=(-2, -1))
    raw_equal = exact_event_equal_matrix(active, chunk_success)
    equal = semantic_equal & raw_equal
    return distance, equal, labels, {
        "kind": "semantic_event_tape_v1",
        "semantic_events_available": True,
        "aligned_time": True,
        "raw_contact_exact_in_events_equal": True,
        "contact_role_mapping_fraction": 1.0 if not names else mapped_pairs / len(names),
        "task_object_geom_count": int(
            sum(bool(row.get("is_task_object", False)) for row in geom_rows)
        ),
        "finger_sides": sorted(
            {
                str(row.get("finger_side"))
                for row in geom_rows
                if row.get("finger_side") in ("left", "right")
            }
        ),
        "task_object_translation_coordinates": int(len(target_translations)),
    }


def _upper_rows(candidate_ids: np.ndarray) -> Iterable[tuple[int, int, int, int]]:
    for left in range(len(candidate_ids)):
        for right in range(left + 1, len(candidate_ids)):
            yield left, right, int(candidate_ids[left]), int(candidate_ids[right])


def _formal_manifest(capture: Path) -> tuple[Path, dict[str, Any]]:
    candidates = (capture / "manifest.json", capture / "behavior_manifest.json")
    for path in candidates:
        if path.exists():
            manifest = _load_json(path)
            if manifest.get("schema") == FORMAL_SCHEMA:
                return path, manifest
    raise ValueError(f"{capture} does not contain a {FORMAL_SCHEMA} manifest")


def _snapshot_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    entries = manifest.get("snapshots", manifest.get("artifacts"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("formal manifest has no snapshots")
    failed = [entry for entry in entries if entry.get("status", "complete") != "complete"]
    if failed:
        raise ValueError("formal manifest contains a snapshot that failed its fidelity gate")
    indices = [int(entry.get("snapshot_index", -1)) for entry in entries]
    if any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise ValueError("formal manifest has invalid or duplicate snapshot indices")
    planned = manifest.get("planned_snapshots")
    if isinstance(planned, list):
        planned_indices = [int(entry.get("snapshot_index", -1)) for entry in planned]
        if sorted(planned_indices) != sorted(indices):
            raise ValueError("completed artifacts do not exactly cover the snapshot plan")
    return entries


def _validate_capture_ledger(
    capture: Path,
    manifest: Mapping[str, Any],
    entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    records_name = manifest.get("records_file")
    records_sha256 = manifest.get("records_file_sha256")
    if not isinstance(records_name, str) or not isinstance(records_sha256, str):
        raise ValueError("formal manifest has no checksummed records file")
    path = capture / records_name
    if _sha256(path) != records_sha256:
        raise ValueError("records file checksum mismatch")
    records = _load_json(path)
    if records.get("schema") != FORMAL_SCHEMA:
        raise ValueError("records file schema mismatch")
    queries = records.get("queries")
    snapshots = records.get("snapshots")
    if not isinstance(queries, list) or not isinstance(snapshots, list):
        raise ValueError("records file must contain query and snapshot lists")
    if int(manifest.get("query_count", -1)) != len(queries):
        raise ValueError("manifest query_count disagrees with records")
    config = manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("manifest config is malformed")
    capture_routes = bool(config.get("capture_routes", False))
    query_id_base = int(config.get("query_id_base", 0))
    trace_row_offset = int(config.get("trace_row_offset", 0))
    for ordinal, query in enumerate(queries):
        if int(query.get("local_call_ordinal", -1)) != ordinal:
            raise ValueError("query ledger local ordinals are not contiguous")
        if int(query.get("query_id", -1)) != query_id_base + ordinal:
            raise ValueError("query ledger ids are not contiguous")
        expected_row = trace_row_offset + ordinal if capture_routes else -1
        if int(query.get("trace_row", -2)) != expected_row:
            raise ValueError("query ledger trace rows violate the route-capture contract")
    artifact_by_index = {int(entry["snapshot_index"]): entry for entry in entries}
    record_by_index = {int(entry.get("snapshot_index", -1)): entry for entry in snapshots}
    if len(record_by_index) != len(snapshots) or set(record_by_index) != set(artifact_by_index):
        raise ValueError("records snapshots do not match manifest artifacts")
    covered = np.full(len(queries), -1, dtype=np.int64)
    for index, artifact in artifact_by_index.items():
        record = record_by_index[index]
        for key in ("npz_file", "npz_sha256", "layout_file", "layout_sha256"):
            if record.get(key) != artifact.get(key):
                raise ValueError(f"snapshot {index} records disagree on {key}")
        interval = record.get("query_ordinal_range")
        if not isinstance(interval, list) or len(interval) != 2:
            raise ValueError(f"snapshot {index} has no query ordinal range")
        start, end = int(interval[0]), int(interval[1])
        if start < 0 or end < start or end > len(queries) or np.any(covered[start:end] >= 0):
            raise ValueError("snapshot query ranges are invalid or overlap")
        covered[start:end] = index
    for ordinal, snapshot_index in enumerate(covered):
        query = queries[ordinal]
        if snapshot_index >= 0:
            if int(query.get("artifact_snapshot_index", -1)) != int(snapshot_index):
                raise ValueError("committed query does not identify its snapshot artifact")
        elif query.get("abandoned") is not True:
            raise ValueError("unassigned query ledger rows must be marked abandoned")
    return {
        "path": str(path),
        "sha256": records_sha256,
        "query_count": len(queries),
        "capture_routes": capture_routes,
    }


def _resolve_snapshot_file(capture: Path, entry: Mapping[str, Any]) -> Path:
    name = (
        entry.get("data_file")
        or entry.get("npz")
        or entry.get("physics_file")
        or entry.get("npz_file")
    )
    if not isinstance(name, str):
        raise ValueError("snapshot manifest entry has no data file")
    path = capture / name
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = entry.get("sha256") or entry.get("data_sha256") or entry.get("npz_sha256")
    if expected is not None and _sha256(path) != expected:
        raise ValueError(f"snapshot checksum mismatch: {path}")
    return path


def _resolve_layout(capture: Path, entry: Mapping[str, Any], arrays: Mapping[str, Any]) -> dict:
    name = entry.get("layout_file")
    if isinstance(name, str):
        path = capture / name
        expected = entry.get("layout_sha256")
        if expected is not None and _sha256(path) != expected:
            raise ValueError(f"snapshot layout checksum mismatch: {path}")
        return _load_json(path)
    if "layout_json" in arrays:
        raw = np.asarray(arrays["layout_json"]).item()
        return json.loads(str(raw))
    raise ValueError("snapshot has no joint layout")


def _validate_formal_arrays(arrays: Mapping[str, np.ndarray]) -> tuple[int, int, int]:
    required = {
        "actions",
        "sim_states",
        "eef_positions",
        "eef_quaternions",
        "gripper_qpos",
        "chunk_success",
        "contact_active",
        "contact_pair_names",
        "continuation_success",
        "continuation_final_sim_states",
        "continuation_action_steps",
        "candidate_trace_rows",
        "candidate_ids",
        "execution_order",
        "robot_qpos_indices",
        "object_qpos_indices",
        "continuation_flow_noise",
        "fidelity_passed",
        "snapshot_full_state_sha256",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise ValueError(f"formal snapshot is missing arrays: {missing}")
    actions = np.asarray(arrays["actions"])
    if actions.ndim != 3 or actions.shape[-1] != 7 or len(actions) < 2:
        raise ValueError("actions must have shape [K,H,7]")
    k, h = actions.shape[:2]
    sim = np.asarray(arrays["sim_states"])
    if sim.ndim != 3 or sim.shape[:2] != (k, h + 1):
        raise ValueError("sim_states must have shape [K,H+1,D]")
    expected = {
        "eef_positions": (k, h + 1, 3),
        "eef_quaternions": (k, h + 1, 4),
        "chunk_success": (k, h + 1),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    gripper = np.asarray(arrays["gripper_qpos"])
    contacts = np.asarray(arrays["contact_active"])
    if gripper.ndim != 3 or gripper.shape[:2] != (k, h + 1):
        raise ValueError("gripper_qpos must have shape [K,H+1,G]")
    if contacts.ndim != 3 or contacts.shape[:2] != (k, h + 1):
        raise ValueError("contact_active must have shape [K,H+1,P]")
    if len(np.asarray(arrays["contact_pair_names"])) != contacts.shape[-1]:
        raise ValueError("contact_pair_names is not aligned with contact_active")
    continuation = np.asarray(arrays["continuation_success"])
    if continuation.ndim != 2 or continuation.shape[0] != k or continuation.shape[1] < 1:
        raise ValueError("continuation_success must have shape [K,R], R>=1")
    r = continuation.shape[1]
    if np.asarray(arrays["continuation_final_sim_states"]).shape != (k, r, sim.shape[-1]):
        raise ValueError("continuation_final_sim_states must have shape [K,R,D]")
    if np.asarray(arrays["continuation_action_steps"]).shape != (k, r):
        raise ValueError("continuation_action_steps must have shape [K,R]")
    ids = np.asarray(arrays["candidate_ids"])
    if ids.shape != (k,) or len(np.unique(ids)) != k:
        raise ValueError("candidate_ids must contain K unique values")
    if np.asarray(arrays["candidate_trace_rows"]).shape != (k,):
        raise ValueError("candidate_trace_rows must have shape [K]")
    if sorted(np.asarray(arrays["execution_order"]).tolist()) != sorted(ids.tolist()):
        raise ValueError("execution_order is not a permutation of candidate_ids")
    if not bool(np.asarray(arrays["fidelity_passed"]).item()):
        raise ValueError("snapshot failed the hard-restore fidelity gate")
    if not np.array_equal(sim[:, 0], np.repeat(sim[:1, 0], k, axis=0)):
        raise ValueError("candidate trajectories do not start at one identical simulator state")
    nq = int(np.asarray(arrays.get("nq", 0)).item())
    robot = set(map(int, np.asarray(arrays["robot_qpos_indices"])))
    objects = set(map(int, np.asarray(arrays["object_qpos_indices"])))
    if nq <= 0 or robot & objects or robot | objects != set(range(nq)):
        raise ValueError("robot/object qpos indices do not partition qpos")
    continuation_noise = np.asarray(arrays["continuation_flow_noise"])
    if (
        continuation_noise.ndim != 4
        or continuation_noise.shape[0] != r
        or continuation_noise.shape[1] < 1
        or continuation_noise.shape[2:] != (10, 24)
    ):
        raise ValueError("continuation_flow_noise must have shape [R,B,10,24]")
    for name in ("contact_active", "chunk_success", "continuation_success"):
        if np.asarray(arrays[name]).dtype != np.bool_:
            raise ValueError(f"{name} must use boolean storage")
    snapshot_hash = np.asarray(arrays["snapshot_full_state_sha256"])
    if snapshot_hash.shape != () or len(str(snapshot_hash.item())) != 64:
        raise ValueError("snapshot_full_state_sha256 must be a 64-character scalar")
    for name in required - {"contact_pair_names", "candidate_ids", "candidate_trace_rows"}:
        value = np.asarray(arrays[name])
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    return k, h, r


def _pool_from_formal(
    capture: Path,
    entry: Mapping[str, Any],
    action_std: np.ndarray,
    gripper_weight: float,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = _resolve_snapshot_file(capture, entry)
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    k, h, repeats = _validate_formal_arrays(arrays)
    layout = _resolve_layout(capture, entry, arrays)
    snapshot_state_sha256 = str(np.asarray(arrays["snapshot_full_state_sha256"]).item())

    actions = np.asarray(arrays["actions"], dtype=np.float64)
    d_action = action_distance_matrix(actions, action_std, gripper_weight)
    d_action_full_gripper = action_distance_matrix(actions, action_std, 1.0)
    d_action_rms = action_rms_distance_matrix(actions, action_std)
    physical_components = _layout_physical_components(
        arrays["sim_states"],
        arrays["eef_positions"],
        arrays["eef_quaternions"],
        arrays["gripper_qpos"],
        layout,
    )
    successor_components = _layout_physical_components(
        np.asarray(arrays["sim_states"])[:, -1:],
        np.asarray(arrays["eef_positions"])[:, -1:],
        np.asarray(arrays["eef_quaternions"])[:, -1:],
        np.asarray(arrays["gripper_qpos"])[:, -1:],
        layout,
        endpoint=True,
    )
    d_contact_raw = contact_event_distance_matrix(arrays["contact_active"])
    raw_contacts_equal = exact_event_equal_matrix(
        arrays["contact_active"], arrays["chunk_success"]
    )
    d_event, events_equal, event_labels, event_provenance = _coarse_event_matrices(
        arrays, layout
    )
    outcomes = np.asarray(arrays["continuation_success"], dtype=np.bool_)
    q_hat = outcomes.mean(axis=1)
    candidate_ids = np.asarray(arrays["candidate_ids"], dtype=np.int64)
    episode = int(entry["episode"])
    fork_step = int(entry["fork_step"])
    tag = str(entry.get("snapshot_id") or _snapshot_tag(episode, fork_step))

    interval_cache: dict[tuple[int, int], dict[str, float | int]] = {}
    rows: list[dict[str, Any]] = []
    for left, right, left_id, right_id in _upper_rows(candidate_ids):
        left_only = int(np.sum(outcomes[left] & ~outcomes[right]))
        right_only = int(np.sum(~outcomes[left] & outcomes[right]))
        key = (left_only, right_only)
        if key not in interval_cache:
            interval_cache[key] = paired_binary_difference_interval(
                outcomes[left], outcomes[right], confidence
            )
        interval = interval_cache[key]
        row: dict[str, Any] = {
            "snapshot": tag,
            "snapshot_state_sha256": snapshot_state_sha256,
            "episode": episode,
            "fork_step": fork_step,
            "task_id": int(entry.get("task_id", -1)),
            "task_name": str(entry.get("task_name", "unknown")),
            "candidate_i": left_id,
            "candidate_j": right_id,
            "d_action": float(d_action[left, right]),
            "d_action_full_gripper": float(d_action_full_gripper[left, right]),
            "d_action_rms": float(d_action_rms[left, right]),
            "d_event": float(d_event[left, right]),
            "d_contact_raw": float(d_contact_raw[left, right]),
            "raw_contacts_equal": bool(raw_contacts_equal[left, right]),
            "events_equal": bool(events_equal[left, right]),
            "q_i": float(q_hat[left]),
            "q_j": float(q_hat[right]),
            "abs_delta_q": float(abs(q_hat[left] - q_hat[right])),
            "q_delta": float(interval["estimate"]),
            "q_ci_lower": float(interval["lower"]),
            "q_ci_upper": float(interval["upper"]),
            "q_discordant": int(interval["discordant"]),
        }
        for name, matrix in physical_components.items():
            row[f"dX_{name}"] = float(matrix[left, right])
        for name, matrix in successor_components.items():
            row[f"ds_{name}"] = float(matrix[left, right])
        rows.append(row)
    pool = {
        "tag": tag,
        "episode": episode,
        "fork_step": fork_step,
        "snapshot_state_sha256": snapshot_state_sha256,
        "task_id": int(entry.get("task_id", -1)),
        "task_name": str(entry.get("task_name", "unknown")),
        "candidates": k,
        "chunk_steps": h,
        "continuation_repeats": repeats,
        "contact_pair_count": int(np.asarray(arrays["contact_active"]).shape[-1]),
        "event_labels": event_labels,
        "event_provenance": event_provenance,
        "physical_components": sorted(physical_components),
        "q_min": float(q_hat.min()),
        "q_max": float(q_hat.max()),
        "q_variance": float(q_hat.var()),
        "data_file": str(path.relative_to(capture)),
    }
    return pool, rows


def load_formal(
    capture: Path,
    gripper_weight: float,
    confidence: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path, manifest = _formal_manifest(capture)
    if manifest.get("complete") is not True:
        raise ValueError("formal capture manifest is not marked complete")
    entries = _snapshot_entries(manifest)
    ledger_provenance = _validate_capture_ledger(capture, manifest, entries)
    metadata_path = capture / str(manifest.get("server_metadata_file", "server_metadata.json"))
    expected_metadata_sha256 = manifest.get("server_metadata_sha256")
    if not isinstance(expected_metadata_sha256, str):
        raise ValueError("formal manifest has no server_metadata_sha256")
    if _sha256(metadata_path) != expected_metadata_sha256:
        raise ValueError("server metadata checksum mismatch")
    metadata = _load_json(metadata_path)
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    if action_std.shape != (7,) or not np.all(np.isfinite(action_std)) or np.any(action_std <= 0):
        raise ValueError("normalization_action_std must contain seven finite positive values")
    pools: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for entry in entries:
        pool, pair_rows = _pool_from_formal(
            capture, entry, action_std, gripper_weight, confidence
        )
        pools.append(pool)
        rows.extend(pair_rows)
    provenance = {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "server_metadata": str(metadata_path),
        "server_metadata_sha256": _sha256(metadata_path),
        "action_std": action_std.tolist(),
        "capture_schema": manifest["schema"],
        "seed_scheme": manifest.get("seed_scheme"),
        "query_ledger": ledger_provenance,
    }
    return provenance, pools, rows


def load_legacy(
    capture: Path,
    layout_path: Path,
    gripper_weight: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    records_path = capture / "fork_records.json"
    metadata_path = capture / "server_metadata.json"
    records = _load_json(records_path)
    metadata = _load_json(metadata_path)
    layout = _load_json(layout_path)
    action_std = np.asarray(metadata["normalization_action_std"], dtype=np.float64)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["episode"]), int(record["fork_step"]))].append(record)

    pools: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []
    for (episode, fork_step), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["candidate"]))
        candidate_ids = np.asarray([int(row["candidate"]) for row in group])
        path = capture / f"chunks_ep{episode:02d}_t{fork_step:02d}.npy"
        actions = np.asarray(np.load(path), dtype=np.float64)[candidate_ids]
        qpos = np.asarray([row["qpos_after_chunk"] for row in group], dtype=np.float64)
        qvel = np.asarray([row["qvel_after_chunk"] for row in group], dtype=np.float64)
        eef = np.asarray([row["eef_after_chunk"] for row in group], dtype=np.float64)
        gripper = np.asarray([row["gripper_after_chunk"] for row in group], dtype=np.float64)
        consequence = np.asarray([row["drawer_delta"] for row in group], dtype=np.float64)
        d_action = action_distance_matrix(actions, action_std, gripper_weight)
        d_action_full = action_distance_matrix(actions, action_std, 1.0)
        d_action_rms = action_rms_distance_matrix(actions, action_std)
        sim = np.concatenate(
            [np.zeros((len(group), 1)), qpos, qvel], axis=1
        )[:, None, :]
        components = _layout_physical_components(
            sim,
            eef[:, None, :],
            None,
            gripper[:, None, :],
            layout,
            endpoint=True,
        )
        tag = _snapshot_tag(episode, fork_step)
        for left, right, left_id, right_id in _upper_rows(candidate_ids):
            row = {
                "snapshot": tag,
                "episode": episode,
                "fork_step": fork_step,
                "candidate_i": left_id,
                "candidate_j": right_id,
                "d_action": float(d_action[left, right]),
                "d_action_full_gripper": float(d_action_full[left, right]),
                "d_action_rms": float(d_action_rms[left, right]),
                "proxy_outcome_i": float(consequence[left]),
                "proxy_outcome_j": float(consequence[right]),
                "proxy_outcome_gap": float(abs(consequence[left] - consequence[right])),
            }
            for name, matrix in components.items():
                row[f"dX_{name}"] = float(matrix[left, right])
            rows_out.append(row)
        pools.append(
            {
                "tag": tag,
                "episode": episode,
                "fork_step": fork_step,
                "candidates": len(group),
                "chunk_steps": int(actions.shape[1]),
                "continuation_repeats": 1,
                "physical_components": sorted(components),
                "proxy_outcome_min": float(consequence.min()),
                "proxy_outcome_max": float(consequence.max()),
                "rerun_drift": float(group[0].get("rerun_drift", np.nan)),
                "candidate_spread_legacy_mismatched_horizon": float(
                    group[0].get("candidate_spread", np.nan)
                ),
            }
        )
    provenance = {
        "fork_records": str(records_path),
        "fork_records_sha256": _sha256(records_path),
        "server_metadata": str(metadata_path),
        "server_metadata_sha256": _sha256(metadata_path),
        "layout": str(layout_path),
        "layout_sha256": _sha256(layout_path),
        "action_std": action_std.tolist(),
        "capture_schema": "legacy.fork_pilot.endpoint_proxy",
    }
    return provenance, pools, rows_out


def _thresholds_from_file(path: Path) -> PairThresholds:
    source = _load_json(path)
    raw = source.get("thresholds", source)
    return PairThresholds(**{field.name: float(raw[field.name]) for field in fields(PairThresholds)})


def _external_calibration_provenance(source: Mapping[str, Any]) -> dict[str, Any]:
    raw = source.get("provenance", {})
    return dict(raw) if isinstance(raw, Mapping) else {}


def _calibration_identity(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    material = list(rows)
    return {
        "calibration_snapshots": sorted({str(row["snapshot"]) for row in material}),
        "calibration_snapshot_hashes": sorted(
            {
                str(row["snapshot_state_sha256"])
                for row in material
                if row.get("snapshot_state_sha256")
            }
        ),
        "calibration_pair_count": len(material),
    }


def _calibration_tags(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    source = _load_json(path)
    if not isinstance(source, list) or not source or not all(isinstance(v, str) for v in source):
        raise ValueError("calibration-tags must contain a non-empty JSON string list")
    return set(source)


def _trajectory_component_name(name: str) -> str:
    return name.replace("_endpoint", "")


def _physical_floor(name: str) -> float:
    if name in PHYSICAL_SCALE_FLOORS:
        return PHYSICAL_SCALE_FLOORS[name]
    base = _trajectory_component_name(name)
    if base in PHYSICAL_SCALE_FLOORS:
        return PHYSICAL_SCALE_FLOORS[base]
    raise ValueError(f"no physical scale floor is registered for component {name!r}")


def _fit_or_load_physical_scales(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, Any]]:
    if args.physics_scales_in is not None:
        source = _load_json(args.physics_scales_in)
        raw = source.get("physical_scales", source)
        if not isinstance(raw, Mapping) or not raw:
            raise ValueError("physics-scales-in contains no physical_scales mapping")
        scales = {str(name): float(value) for name, value in raw.items()}
        if any(not np.isfinite(value) or value <= 0.0 for value in scales.values()):
            raise ValueError("frozen physical scales must be finite and positive")
        return scales, {
            "kind": "frozen_external",
            "path": str(args.physics_scales_in),
            "sha256": _sha256(args.physics_scales_in),
            "source_provenance": _external_calibration_provenance(source),
        }
    tags = _calibration_tags(args.calibration_tags)
    calibration = rows if tags is None else [row for row in rows if row["snapshot"] in tags]
    if not calibration:
        raise ValueError("no pair rows belong to the physical-scale calibration split")
    names = sorted(
        {
            key[3:]
            for row in calibration
            for key in row
            if key.startswith("dX_")
        }
    )
    if not names:
        raise ValueError("pair rows contain no physical components")
    scales: dict[str, float] = {}
    for name in names:
        key = f"dX_{name}"
        available = [float(row[key]) for row in calibration if key in row]
        if not available:
            raise ValueError(f"physical component {name!r} has no calibration values")
        values = np.asarray(available, dtype=np.float64)
        positive = values[values > 0.0]
        empirical = float(np.median(positive)) if len(positive) else 0.0
        scales[name] = max(empirical, _physical_floor(name))
    return scales, {
        "kind": "posthoc_all_snapshots" if tags is None else "calibration_split",
        **_calibration_identity(calibration),
        "estimator": "max(median_positive_pair_distance, physical_resolution_floor)",
        "resolution_floors": {name: _physical_floor(name) for name in names},
    }


def _apply_physical_scales(rows: list[dict[str, Any]], scales: Mapping[str, float]) -> None:
    for row in rows:
        component_names = sorted(key[3:] for key in row if key.startswith("dX_"))
        missing = sorted(set(component_names) - set(scales))
        if missing:
            raise ValueError(f"frozen physical scales are missing components: {missing}")
        if not component_names:
            raise ValueError("pair row contains no physical components")
        normalized = [float(row[f"dX_{name}"]) / float(scales[name]) for name in component_names]
        row["d_physics"] = float(np.sqrt(np.mean(np.square(normalized))))
        row["d_physics_component_count"] = len(component_names)
        successor_names = sorted(key[3:] for key in row if key.startswith("ds_"))
        if successor_names:
            successor = []
            for endpoint_name in successor_names:
                trajectory_name = _trajectory_component_name(endpoint_name)
                if trajectory_name not in scales:
                    raise ValueError(
                        f"successor component {endpoint_name!r} has no trajectory scale"
                    )
                successor.append(
                    float(row[f"ds_{endpoint_name}"]) / float(scales[trajectory_name])
                )
            row["d_successor"] = float(np.sqrt(np.mean(np.square(successor))))


def _fit_or_load_thresholds(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[PairThresholds, dict[str, Any]]:
    if args.thresholds_in is not None:
        source = _load_json(args.thresholds_in)
        return _thresholds_from_file(args.thresholds_in), {
            "kind": "frozen_external",
            "path": str(args.thresholds_in),
            "sha256": _sha256(args.thresholds_in),
            "source_provenance": _external_calibration_provenance(source),
        }
    tags = _calibration_tags(args.calibration_tags)
    calibration = rows if tags is None else [row for row in rows if row["snapshot"] in tags]
    if not calibration:
        raise ValueError("no pair rows belong to the calibration split")
    action = np.asarray([row["d_action"] for row in calibration], dtype=np.float64)
    physics = np.asarray([row["d_physics"] for row in calibration], dtype=np.float64)
    raw = {
        "action_near": float(np.quantile(action, args.near_quantile)),
        "action_far": float(np.quantile(action, args.far_quantile)),
        "physics_near": float(np.quantile(physics, args.near_quantile)),
        "physics_far": float(np.quantile(physics, args.far_quantile)),
    }
    widened: list[str] = []
    adjusted = dict(raw)
    for name in ("action", "physics"):
        near_key = f"{name}_near"
        far_key = f"{name}_far"
        if adjusted[near_key] < adjusted[far_key]:
            continue
        center = adjusted[near_key]
        # A one-pair smoke run has identical empirical quantiles.  The narrowest
        # representable bracket leaves a positive tied value unclassified.
        adjusted[near_key] = (
            0.0 if center == 0.0 else float(np.nextafter(center, -np.inf))
        )
        adjusted[far_key] = float(np.nextafter(center, np.inf))
        widened.append(name)
    if widened:
        threshold = PairThresholds(
            **adjusted,
            q_equivalence=args.q_equivalence,
            q_difference=args.q_difference,
        )
    else:
        threshold = empirical_thresholds(
            action,
            physics,
            args.near_quantile,
            args.far_quantile,
            args.q_equivalence,
            args.q_difference,
        )
    return threshold, {
        "kind": "posthoc_all_snapshots" if tags is None else "calibration_split",
        **_calibration_identity(calibration),
        "near_quantile": args.near_quantile,
        "far_quantile": args.far_quantile,
        "raw_quantile_thresholds": raw,
        "degenerate_quantiles_widened": widened,
        "degenerate_widening_strategy": (
            "adjacent_ieee754_values" if widened else None
        ),
    }


def _apply_formal_classes(rows: list[dict[str, Any]], threshold: PairThresholds) -> None:
    for row in rows:
        interval = {
            "lower": row["q_ci_lower"],
            "upper": row["q_ci_upper"],
        }
        row.update(
            classify_pair(
                row["d_action"],
                row["d_physics"],
                row["events_equal"],
                interval,
                threshold,
            )
        )


def _apply_proxy_classes(rows: list[dict[str, Any]], threshold: PairThresholds) -> dict[str, float]:
    outcome = np.asarray([row["proxy_outcome_gap"] for row in rows], dtype=np.float64)
    near_outcome = float(np.quantile(outcome, 0.2))
    far_outcome = float(np.quantile(outcome, 0.8))
    for row in rows:
        far_action = row["d_action"] >= threshold.action_far
        near_action = row["d_action"] <= threshold.action_near
        far_physics = row["d_physics"] >= threshold.physics_far
        near_physics = row["d_physics"] <= threshold.physics_near
        same = row["proxy_outcome_gap"] <= near_outcome
        different = row["proxy_outcome_gap"] >= far_outcome
        row.update(
            {
                "proxy_command_redundancy": bool(far_action and near_physics and same),
                "proxy_control_equivalence": bool(far_action and far_physics and same),
                "proxy_physics_amplification": bool(near_action and far_physics and different),
                "proxy_critical_microdifference": bool(near_action and near_physics and different),
            }
        )
    return {"proxy_outcome_near_q20": near_outcome, "proxy_outcome_far_q80": far_outcome}


def _snapshot_metric(rows: list[dict[str, Any]], key: str) -> dict[str, np.ndarray]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["snapshot"]].append(float(row[key]))
    return {tag: np.asarray(value, dtype=np.float64) for tag, value in grouped.items()}


def _mean_rank_correlation(
    rows: list[dict[str, Any]],
    left_key: str,
    right_key: str,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["snapshot"]].append(row)
    correlations: dict[str, np.ndarray] = {}
    skipped = []
    for tag, group in grouped.items():
        left = np.asarray([row[left_key] for row in group], dtype=np.float64)
        right = np.asarray([row[right_key] for row in group], dtype=np.float64)
        if np.ptp(left) == 0.0 or np.ptp(right) == 0.0:
            skipped.append(tag)
            continue
        value = float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])
        correlations[tag] = np.asarray([value])
    if not correlations:
        return {"snapshots": 0, "skipped": skipped}
    return {
        **snapshot_bootstrap_mean(correlations, draws=draws, seed=seed),
        "skipped": skipped,
        "left": left_key,
        "right": right_key,
    }


def _class_summary(
    rows: list[dict[str, Any]],
    classes: Iterable[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    out = {}
    for offset, name in enumerate(classes):
        values = _snapshot_metric(rows, name)
        out[name] = {
            "pair_count": int(sum(bool(row[name]) for row in rows)),
            "pair_rate_snapshot_bootstrap": snapshot_bootstrap_mean(
                values, draws=draws, seed=seed + offset
            ),
        }
    return out


def _write_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "snapshot",
        "episode",
        "fork_step",
        "candidate_i",
        "candidate_j",
        "d_action",
        "d_physics",
        "d_event",
        "d_successor",
        "abs_delta_q",
        "proxy_outcome_gap",
    ]
    ordered = [name for name in preferred if name in fields] + [
        name for name in fields if name not in preferred
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, Any]], path: Path, formal: bool) -> None:
    action = np.asarray([row["d_action"] for row in rows])
    physics = np.asarray([row["d_physics"] for row in rows])
    outcome_key = "abs_delta_q" if formal else "proxy_outcome_gap"
    outcome = np.asarray([row[outcome_key] for row in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), constrained_layout=True)
    panels = (
        (action, physics, "action distance", "physical trajectory distance" if formal else "endpoint proxy distance"),
        (action, outcome, "action distance", "|Delta Q|" if formal else "|Delta drawer proxy|"),
        (physics, outcome, "physical trajectory distance" if formal else "endpoint proxy distance", "|Delta Q|" if formal else "|Delta drawer proxy|"),
    )
    for axis, (x, y, xlabel, ylabel) in zip(axes, panels):
        axis.scatter(x, y, s=8, alpha=0.16, linewidths=0, color="#176b87")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Action - physics - outcome geometry" + ("" if formal else " (endpoint-proxy pilot)"))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _formal_support_limitations(
    pools: Iterable[Mapping[str, Any]], q_equivalence: float, confidence: float
) -> list[str]:
    limitations: list[str] = []
    for pool in pools:
        tag = str(pool["tag"])
        repeats = int(pool["continuation_repeats"])
        zero = np.zeros(repeats, dtype=np.bool_)
        interval = paired_binary_difference_interval(zero, zero, confidence)
        if not interval_is_equivalent(interval, q_equivalence):
            limitations.append(
                f"{tag}: R={repeats} cannot establish +/-{q_equivalence:g} Q "
                "equivalence even with zero discordance."
            )
        event = pool.get("event_provenance", {})
        if not isinstance(event, Mapping) or event.get("kind") != "semantic_event_tape_v1":
            limitations.append(f"{tag}: semantic event tape is unavailable.")
            continue
        if float(event.get("contact_role_mapping_fraction", 0.0)) < 1.0:
            limitations.append(f"{tag}: not every active contact pair maps to geom roles.")
        if int(event.get("task_object_geom_count", 0)) < 1:
            limitations.append(f"{tag}: task-object geom identity is unavailable.")
        if set(event.get("finger_sides", [])) != {"left", "right"}:
            limitations.append(f"{tag}: left/right gripper identity is incomplete.")
    return sorted(set(limitations))


def _source_calibration_hashes(provenance: Mapping[str, Any]) -> set[str]:
    source = provenance.get("source_provenance", {})
    if not isinstance(source, Mapping):
        return set()
    values = source.get("calibration_snapshot_hashes", [])
    if not isinstance(values, list):
        return set()
    return {str(value) for value in values if value}


def _confirmatory_checks(
    pools: Iterable[Mapping[str, Any]],
    strict_supported: bool,
    threshold_provenance: Mapping[str, Any],
    physical_scale_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = {
        str(pool["snapshot_state_sha256"])
        for pool in pools
        if pool.get("snapshot_state_sha256")
    }
    threshold_calibration = _source_calibration_hashes(threshold_provenance)
    scale_calibration = _source_calibration_hashes(physical_scale_provenance)
    checks = {
        "strict_pair_mining_supported": strict_supported,
        "thresholds_frozen_external": threshold_provenance.get("kind") == "frozen_external",
        "physical_scales_frozen_external": physical_scale_provenance.get("kind")
        == "frozen_external",
        "evaluation_snapshot_hashes_available": bool(evaluation),
        "threshold_calibration_hashes_available": bool(threshold_calibration),
        "scale_calibration_hashes_available": bool(scale_calibration),
        "threshold_calibration_disjoint": bool(threshold_calibration)
        and threshold_calibration.isdisjoint(evaluation),
        "scale_calibration_disjoint": bool(scale_calibration)
        and scale_calibration.isdisjoint(evaluation),
    }
    checks["passed"] = all(bool(value) for value in checks.values())
    checks["evaluation_snapshot_count"] = len(evaluation)
    checks["threshold_calibration_snapshot_count"] = len(threshold_calibration)
    checks["scale_calibration_snapshot_count"] = len(scale_calibration)
    return checks


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_exists = any(
        path.exists() for path in (args.capture / "manifest.json", args.capture / "behavior_manifest.json")
    )
    mode = args.mode
    if mode == "auto":
        mode = "formal" if manifest_exists else "legacy"

    if mode == "formal":
        provenance, pools, rows = load_formal(
            args.capture, args.gripper_weight, args.confidence
        )
        physical_scales, physical_scale_provenance = _fit_or_load_physical_scales(
            rows, args
        )
        _apply_physical_scales(rows, physical_scales)
        threshold, threshold_provenance = _fit_or_load_thresholds(rows, args)
        _apply_formal_classes(rows, threshold)
        classes = PAIR_CLASSES
        outcome_key = "abs_delta_q"
        proxy_thresholds = None
        limitations = _formal_support_limitations(
            pools, threshold.q_equivalence, args.confidence
        )
        if threshold_provenance.get("degenerate_quantiles_widened"):
            limitations.append(
                "Calibration distances have tied near/far quantiles; thresholds were "
                "minimally widened for pipeline validation only."
            )
        strict_supported = not limitations
    else:
        provenance, pools, rows = load_legacy(
            args.capture, args.legacy_layout, args.gripper_weight
        )
        physical_scales, physical_scale_provenance = _fit_or_load_physical_scales(
            rows, args
        )
        _apply_physical_scales(rows, physical_scales)
        threshold, threshold_provenance = _fit_or_load_thresholds(rows, args)
        proxy_thresholds = _apply_proxy_classes(rows, threshold)
        classes = PROXY_CLASSES
        outcome_key = "proxy_outcome_gap"
        strict_supported = False
        limitations = [
            "Only chunk endpoints are available; d_physics is not a trajectory distance.",
            "EEF orientation and contact/event tapes were not captured.",
            "There is one continuation draw (R=1), so Q equivalence intervals are unavailable.",
            "Proxy pair labels are exploratory quantile screens, not control-equivalence claims.",
            "Legacy rerun drift and candidate_spread use different horizons and cannot form a valid fidelity ratio.",
        ]
        if threshold_provenance.get("degenerate_quantiles_widened"):
            limitations.append(
                "Calibration distances have tied near/far quantiles; thresholds were "
                "minimally widened for pipeline validation only."
            )

    if not rows:
        raise RuntimeError("capture contains no candidate pairs")
    correlation = {
        "action_vs_physics": _mean_rank_correlation(
            rows, "d_action", "d_physics", args.bootstrap, args.seed
        ),
        "action_vs_outcome": _mean_rank_correlation(
            rows, "d_action", outcome_key, args.bootstrap, args.seed + 1
        ),
        "physics_vs_outcome": _mean_rank_correlation(
            rows, "d_physics", outcome_key, args.bootstrap, args.seed + 2
        ),
    }
    class_summary = _class_summary(rows, classes, args.bootstrap, args.seed + 100)
    confirmatory_checks = _confirmatory_checks(
        pools,
        strict_supported,
        threshold_provenance,
        physical_scale_provenance,
    ) if mode == "formal" else {"passed": False, "reason": "legacy proxy mode"}
    summary = {
        "analysis": "action--physics--outcome behavior geometry",
        "mode": mode,
        "strict_pair_mining_supported": strict_supported,
        "confirmatory": bool(confirmatory_checks["passed"]),
        "confirmatory_checks": confirmatory_checks,
        "provenance": provenance,
        "snapshot_count": len(pools),
        "pair_count": len(rows),
        "gripper_weight_primary": args.gripper_weight,
        "full_gripper_and_legacy_rms_reported": True,
        "thresholds": asdict(threshold),
        "threshold_provenance": threshold_provenance,
        "physical_scales": physical_scales,
        "physical_scale_provenance": physical_scale_provenance,
        "proxy_outcome_thresholds": proxy_thresholds,
        "rank_correlations_snapshot_bootstrap": correlation,
        "pair_regions": class_summary,
        "pools": pools,
        "limitations": limitations,
        "pairwise_ci_familywise_adjusted": False,
        "pairwise_ci_scope": (
            "Pairwise mining intervals; selected pairs require independent confirmation."
        ),
    }
    (args.out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "thresholds": asdict(threshold),
                "provenance": threshold_provenance,
                "physical_scales_sha256": None
                if args.physics_scales_in is None
                else _sha256(args.physics_scales_in),
            },
            indent=2,
        )
    )
    (args.out_dir / "physical_scales.json").write_text(
        json.dumps(
            {"physical_scales": physical_scales, "provenance": physical_scale_provenance},
            indent=2,
        )
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_pairs(args.out_dir / "pairs.csv", rows)
    if not args.no_figure:
        _plot(rows, args.out_dir / "geometry.png", formal=(mode == "formal"))

    print(
        f"mode={mode} snapshots={len(pools)} pairs={len(rows)} "
        f"strict={strict_supported} confirmatory={summary['confirmatory']}"
    )
    for name in classes:
        entry = class_summary[name]
        rate = entry["pair_rate_snapshot_bootstrap"]
        if rate.get("available", True):
            interval_text = f"[{rate['lower']:.4f},{rate['upper']:.4f}]"
        else:
            interval_text = "[CI unavailable: one snapshot]"
        print(
            f"  {name:30s} {entry['pair_count']:6d}  "
            f"snapshot-rate={rate['mean']:.4f} {interval_text}"
        )
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
