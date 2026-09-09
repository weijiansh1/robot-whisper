"""Capture versioned same-snapshot action--physics--outcome forks.

This is the production successor to ``fork_pilot.py``.  It deliberately leaves
that historical pilot unchanged while tightening the data contract around the
counterfactual branch:

* K action chunks are queried from one restored observation.
* Every chunk is executed from the same full simulator/controller snapshot.
* The H+1 physical tape includes simulator state, EEF pose, gripper state,
  task success, and the raw set of MuJoCo geom contact pairs.
* Every post-chunk full snapshot is restored R times.  Continuation flow noise
  is keyed only by (snapshot, repeat, future control step), so it is identical
  across candidates by construction.
* A hard-restore rerun uses the same H-step horizon and the same trajectory
  metric on both sides of the fidelity gate.

Phase 1 can run against the ordinary policy server.  With ``--capture-routes``,
the recorder server's ``episode_id`` annotation slot is used as a unique query
id and the real role and coordinates are written to ``records.json``.
``trace_row`` then assumes exclusive recorder access, made explicit by
``--trace-row-offset`` and in the manifest.

Run this file in the LIBERO Python 3.8 environment against a compatible policy
server.  Use ``serve_with_recorder.py`` only when ``--capture-routes`` is set.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.libero_runtime import (
    LIBERO_DUMMY_ACTION,
    EpisodeConfig,
    _load_task,
    validate_policy_suite,
)
from himoe_libero_bridge.preprocess import build_policy_observation
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
    validate_action_response,
)

from branch_snapshot import restore_full_state, save_full_state


SCHEMA_NAME = "himoe.behavior_forks"
SCHEMA_VERSION = 1
SCHEMA = "%s.v%d" % (SCHEMA_NAME, SCHEMA_VERSION)
MANIFEST_FILE = "manifest.json"
RECORDS_FILE = "records.json"
EPISODE_ID_KEY = "episode_id"

# SeedSequence consumes uint32 entropy words.  Distinct role words keep future
# additions from silently reusing an existing random stream.
_CANDIDATE_NOISE_ROLE = 0x43414E44  # CAND
_CONTINUATION_NOISE_ROLE = 0x434F4E54  # CONT
_EXECUTION_ORDER_ROLE = 0x4F524445  # ORDE

_JOINT_TYPE_NAMES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

ARRAY_SCHEMA = {
    "schema_version": "int32 scalar",
    "snapshot_index": "int32 scalar",
    "episode": "int32 scalar",
    "fork_step": "int32 scalar",
    "nq": "int32 scalar",
    "actions": "float32[K,H,7] candidate action commands",
    "candidate_flow_noise": "float32[K,10,24]",
    "sim_states": "float64[K,H+1,D] flattened MuJoCo physical tape",
    "eef_positions": "float64[K,H+1,3]",
    "eef_quaternions": "float64[K,H+1,4], raw LIBERO xyzw convention",
    "gripper_qpos": "float64[K,H+1,G]",
    "chunk_success": "bool[K,H+1] task success tape",
    "contact_active": "bool[K,H+1,P] raw contact-pair set tape",
    "contact_pair_names": "unicode[P] canonical unordered geom-name pairs",
    "continuation_success": "bool[K,R] success within continuation budget",
    "continuation_final_sim_states": "float64[K,R,D]",
    "continuation_action_steps": "int32[K,R] executed environment actions",
    "continuation_flow_noise": "float32[R,B,10,24], candidate-independent CRN",
    "continuation_trace_rows": "int64[K,R,B]",
    "continuation_query_ids": "int32[K,R,B]",
    "continuation_query_executed": "bool[K,R,B]",
    "continuation_actions": "float32[K,R,B,H,7]",
    "continuation_action_executed": "bool[K,R,B,H]",
    "candidate_trace_rows": "int64[K] server route-store rows",
    "candidate_query_ids": "int32[K]",
    "route_capture_enabled": "bool scalar",
    "candidate_ids": "int32[K] array-axis candidate ids",
    "execution_order": "int32[K] candidate ids in physical execution order",
    "robot_qpos_indices": "int32[Qr] qpos-local robot coordinates",
    "object_qpos_indices": "int32[Qo] qpos-local non-robot coordinates",
    "snapshot_sim_state": "float64[D]",
    "snapshot_full_state_sha256": "unicode scalar",
    "post_chunk_full_state_sha256": "unicode[K]",
    "fidelity_reference_candidate": "int32 scalar",
    "fidelity_rerun_sim_states": "float64[H+1,D]",
    "fidelity_drift": "float64 scalar",
    "fidelity_candidate_spread_mean": "float64 scalar",
    "fidelity_candidate_spread_max": "float64 scalar",
    "fidelity_threshold": "float64 scalar",
    "fidelity_passed": "bool scalar",
}


class FidelityGateError(RuntimeError):
    """Raised after a rejected snapshot has been committed for audit."""


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _parse_int_list(value: str, label: str) -> List[int]:
    try:
        values = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError("%s must be a comma-separated integer list" % label) from error
    if not values:
        raise ValueError("%s must not be empty" % label)
    if len(set(values)) != len(values):
        raise ValueError("%s contains duplicates" % label)
    if any(item < 0 for item in values):
        raise ValueError("%s must contain non-negative integers" % label)
    return values


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _noise_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _full_state_sha256(snapshot: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _json_safe(snapshot), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_json_safe(value), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_npz(path: pathlib.Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _hard_restore(environment: Any, snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    """Clear solver scratch, then restore simulator, controller, clock, and warmstart."""

    base = getattr(environment, "env", environment)
    base.sim.reset()
    return restore_full_state(environment, snapshot)


def _addr_bounds(address: Any) -> Tuple[int, int]:
    if isinstance(address, tuple):
        return int(address[0]), int(address[1])
    return int(address), int(address) + 1


def _robot_qpos_references(environment: Any) -> Set[int]:
    base = getattr(environment, "env", environment)
    indices: Set[int] = set()
    for robot in getattr(base, "robots", []):
        for field in ("_ref_joint_pos_indexes", "_ref_gripper_joint_pos_indexes"):
            values = getattr(robot, field, None)
            if values is not None:
                indices.update(int(value) for value in np.asarray(values).reshape(-1))
    return indices


def _canonical_object_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _object_signature(value: Any) -> Tuple[str, Set[str]]:
    raw = str(value).lower().replace("-", "_")
    parts = [part for part in raw.split("_") if part]
    ignored = {
        "body",
        "collision",
        "geom",
        "joint",
        "level",
        "main",
        "region",
        "side",
        "site",
        "visual",
    }
    for index, part in enumerate(parts):
        if part.isdigit():
            prefix = _canonical_object_name("_".join(parts[: index + 1]))
            qualifiers = {
                _canonical_object_name(item)
                for item in parts[index + 1 :]
                if item not in ignored
            }
            return prefix, {item for item in qualifiers if item}
    return _canonical_object_name(raw), set()


def _matches_task_object(names: Sequence[Any], task_objects: Sequence[str]) -> bool:
    candidates = [_object_signature(name) for name in names]
    for raw_target in task_objects:
        target_prefix, target_qualifiers = _object_signature(raw_target)
        for candidate_prefix, candidate_qualifiers in candidates:
            same_instance = target_prefix == candidate_prefix
            if not same_instance and (
                target_prefix not in candidate_prefix and candidate_prefix not in target_prefix
            ):
                continue
            if not target_qualifiers or target_qualifiers.intersection(candidate_qualifiers):
                return True
    return False


def _finger_side(*names: Any) -> Optional[str]:
    value = " ".join(str(name).lower() for name in names)
    canonical = _canonical_object_name(value)
    if any(token in canonical for token in ("finger1", "leftfinger", "leftpad")):
        return "left"
    if any(token in canonical for token in ("finger2", "rightfinger", "rightpad")):
        return "right"
    return None


def _enhanced_sim_layout(environment: Any, state_dim: int) -> Dict[str, Any]:
    """Return qpos-local partitions plus joint type and coordinate slices."""

    base = getattr(environment, "env", environment)
    model = base.sim.model
    task_objects = [str(item) for item in getattr(environment, "obj_of_interest", [])]
    nq, nv, na = int(model.nq), int(model.nv), int(getattr(model, "na", 0))
    minimum_dim = 1 + nq + nv + na
    if state_dim < minimum_dim:
        raise RuntimeError(
            "flattened state has %d values, below time+qpos+qvel+act=%d"
            % (state_dim, minimum_dim)
        )
    referenced_robot = _robot_qpos_references(environment)
    joints: List[Dict[str, Any]] = []
    robot_indices: Set[int] = set(referenced_robot)
    covered: Set[int] = set()
    for raw_name in model.joint_names:
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        joint_id = int(model.joint_name2id(name))
        qlo, qhi = _addr_bounds(model.get_joint_qpos_addr(name))
        vlo, vhi = _addr_bounds(model.get_joint_qvel_addr(name))
        qcoords = set(range(qlo, qhi))
        named_robot = name.startswith(("robot0_", "gripper0_"))
        is_robot = bool(named_robot or qcoords.intersection(referenced_robot))
        joint_body_id = int(model.jnt_bodyid[joint_id])
        raw_joint_body_name = model.body_id2name(joint_body_id)
        joint_body_name = (
            "__unnamed_body_%d" % joint_body_id
            if raw_joint_body_name is None
            else raw_joint_body_name.decode("utf-8")
            if isinstance(raw_joint_body_name, bytes)
            else str(raw_joint_body_name)
        )
        if is_robot:
            robot_indices.update(qcoords)
        covered.update(qcoords)
        joint_type_id = int(model.jnt_type[joint_id])
        joints.append(
            {
                "joint": name,
                "joint_id": joint_id,
                "joint_type": _JOINT_TYPE_NAMES.get(joint_type_id, "unknown_%d" % joint_type_id),
                "joint_type_id": joint_type_id,
                "body_id": joint_body_id,
                "body": joint_body_name,
                "qpos_lo": qlo,
                "qpos_hi": qhi,
                "qvel_lo": vlo,
                "qvel_hi": vhi,
                "state_qpos_lo": 1 + qlo,
                "state_qpos_hi": 1 + qhi,
                "state_qvel_lo": 1 + nq + vlo,
                "state_qvel_hi": 1 + nq + vhi,
                "is_robot": is_robot,
                "is_task_object": bool(
                    not is_robot
                    and _matches_task_object((name, joint_body_name), task_objects)
                ),
            }
        )
    expected = set(range(nq))
    if covered != expected:
        raise RuntimeError(
            "joint qpos slices do not cover qpos: missing=%s extra=%s"
            % (sorted(expected - covered), sorted(covered - expected))
        )
    if not robot_indices or not robot_indices.issubset(expected):
        raise RuntimeError("could not derive a valid robot qpos partition")
    object_indices = expected - robot_indices
    qpos_joint_names = [""] * nq
    for joint in joints:
        for index in range(int(joint["qpos_lo"]), int(joint["qpos_hi"])):
            qpos_joint_names[index] = str(joint["joint"])
    joint_by_id = {int(joint["joint_id"]): joint for joint in joints}
    geom_roles: List[Dict[str, Any]] = []
    for geom_id in range(int(model.ngeom)):
        raw_geom_name = model.geom_id2name(geom_id)
        geom_name = (
            "__unnamed_geom_%d" % geom_id
            if raw_geom_name is None
            else raw_geom_name.decode("utf-8")
            if isinstance(raw_geom_name, bytes)
            else str(raw_geom_name)
        )
        body_id = int(model.geom_bodyid[geom_id])
        body_chain: List[int] = []
        cursor = body_id
        while cursor >= 0 and cursor not in body_chain:
            body_chain.append(cursor)
            parent = int(model.body_parentid[cursor])
            if parent == cursor:
                break
            cursor = parent
        chain_joint_ids: List[int] = []
        object_body_id: Optional[int] = None
        for chain_body in body_chain:
            first = int(model.body_jntadr[chain_body])
            count = int(model.body_jntnum[chain_body])
            direct = list(range(first, first + count)) if first >= 0 else []
            chain_joint_ids.extend(direct)
            if object_body_id is None:
                direct_qpos = set()
                for joint_id in direct:
                    joint = joint_by_id[joint_id]
                    direct_qpos.update(range(int(joint["qpos_lo"]), int(joint["qpos_hi"])))
                if direct_qpos.intersection(object_indices):
                    object_body_id = chain_body
        chain_qpos: Set[int] = set()
        for joint_id in chain_joint_ids:
            joint = joint_by_id[joint_id]
            chain_qpos.update(range(int(joint["qpos_lo"]), int(joint["qpos_hi"])))
        raw_body_name = model.body_id2name(body_id)
        body_name = (
            "__unnamed_body_%d" % body_id
            if raw_body_name is None
            else raw_body_name.decode("utf-8")
            if isinstance(raw_body_name, bytes)
            else str(raw_body_name)
        )
        if chain_qpos.intersection(robot_indices):
            role = "robot"
        elif chain_qpos.intersection(object_indices):
            role = "object"
        else:
            role = "support"
        lowered = (geom_name + " " + body_name).lower()
        subtype = "gripper" if role == "robot" and any(
            token in lowered for token in ("gripper", "finger", "hand")
        ) else role
        object_body_name = None
        if object_body_id is not None:
            raw_object_name = model.body_id2name(object_body_id)
            object_body_name = (
                "__unnamed_body_%d" % object_body_id
                if raw_object_name is None
                else raw_object_name.decode("utf-8")
                if isinstance(raw_object_name, bytes)
                else str(raw_object_name)
            )
        geom_roles.append(
            {
                "geom_id": geom_id,
                "geom": geom_name,
                "body_id": body_id,
                "body": body_name,
                "role": role,
                "subtype": subtype,
                "object_body": object_body_name,
                "finger_link": body_name if subtype == "gripper" else None,
                "finger_side": _finger_side(geom_name, body_name)
                if subtype == "gripper"
                else None,
                "is_task_object": bool(
                    role == "object"
                    and _matches_task_object(
                        (geom_name, body_name, object_body_name or ""), task_objects
                    )
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "state_layout": "[time] + qpos + qvel + act + optional_tail",
        "state_dim": int(state_dim),
        "nq": nq,
        "nv": nv,
        "na": na,
        "optional_tail_dim": int(state_dim - minimum_dim),
        "qpos_state_slice": [1, 1 + nq],
        "qvel_state_slice": [1 + nq, 1 + nq + nv],
        "act_state_slice": [1 + nq + nv, 1 + nq + nv + na],
        "robot_qpos_indices": sorted(robot_indices),
        "object_qpos_indices": sorted(object_indices),
        "qpos_joint_names": qpos_joint_names,
        "robot_partition_sources": [
            "robosuite robot reference qpos indices",
            "robot0_/gripper0_ joint-name fallback",
        ],
        "joints": joints,
        "geom_roles": geom_roles,
        "obj_of_interest": task_objects,
        "physical_tape_sources": {
            "sim_state": "environment.get_sim_state",
            "eef_position": "sim.data.site_xpos[robot.eef_site_id]",
            "eef_quaternion": "sim.data.get_body_xquat(robot.robot_model.eef_name), wxyz_to_xyzw",
            "gripper_qpos": "sim.data.qpos[robot._ref_gripper_joint_pos_indexes]",
            "contacts": "sim.data.contact[0:ncon] canonical geom-name pairs",
            "task_success": "environment.check_success",
            "sampled_robosuite_observables_used": False,
        },
    }


def _geom_name(model: Any, geom_id: int) -> str:
    name = model.geom_id2name(int(geom_id))
    if name is None:
        return "__unnamed_geom_%d" % int(geom_id)
    return name.decode("utf-8") if isinstance(name, bytes) else str(name)


def _raw_contact_pairs(environment: Any) -> Set[Tuple[str, str]]:
    """Return the unfiltered set of active unordered MuJoCo geom-name pairs."""

    base = getattr(environment, "env", environment)
    model, data = base.sim.model, base.sim.data
    pairs: Set[Tuple[str, str]] = set()
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        left, right = _geom_name(model, contact.geom1), _geom_name(model, contact.geom2)
        pairs.add(tuple(sorted((left, right))))
    return pairs


def _physical_point(
    environment: Any, _observation: Mapping[str, Any], gripper_dim: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, Set[Tuple[str, str]]]:
    """Read physical signals directly from MuJoCo, not sampled observables.

    Robosuite observables update on their own sampling clocks.  Those clocks and
    caches are not part of a branch snapshot, so ``robot0_eef_pos`` returned by
    ``env.step`` can differ across two branches whose MuJoCo states are exactly
    equal.  The simulator readings below are synchronous with ``sim`` and make
    the H+1 tape a property of the executed physical trajectory.
    """

    base = getattr(environment, "env", environment)
    robots = list(getattr(base, "robots", []))
    if len(robots) != 1:
        raise RuntimeError("physical tape requires exactly one LIBERO robot")
    robot = robots[0]
    data = base.sim.data
    sim = np.asarray(environment.get_sim_state(), dtype=np.float64).copy()
    position = np.asarray(data.site_xpos[int(robot.eef_site_id)], dtype=np.float64).copy()
    quaternion_wxyz = np.asarray(
        data.get_body_xquat(robot.robot_model.eef_name), dtype=np.float64
    ).copy()
    quaternion = quaternion_wxyz[[1, 2, 3, 0]]
    gripper_indices = np.asarray(
        getattr(robot, "_ref_gripper_joint_pos_indexes", []), dtype=np.int64
    ).reshape(-1)
    if not len(gripper_indices):
        raise RuntimeError("could not derive gripper qpos indices from the LIBERO robot")
    gripper = np.asarray(data.qpos[gripper_indices], dtype=np.float64).reshape(-1).copy()
    if position.shape != (3,) or quaternion.shape != (4,):
        raise RuntimeError(
            "unexpected EEF observation shapes: position=%s quaternion=%s"
            % (position.shape, quaternion.shape)
        )
    if gripper_dim is not None and gripper.shape != (gripper_dim,):
        raise RuntimeError(
            "gripper dimension changed from %d to %s" % (gripper_dim, gripper.shape)
        )
    if not all(np.all(np.isfinite(value)) for value in (sim, position, quaternion, gripper)):
        raise RuntimeError("physical observation contains NaN or infinity")
    if np.linalg.norm(quaternion) <= 1e-12:
        raise RuntimeError("EEF quaternion has zero norm")
    return sim, position, quaternion, gripper, bool(environment.check_success()), _raw_contact_pairs(environment)


def _seed_words(
    seed_base: int,
    role: int,
    snapshot_index: int,
    episode: int,
    fork_step: int,
    coordinates: Sequence[int],
) -> List[int]:
    values = [seed_base, role, snapshot_index, episode, fork_step] + [int(v) for v in coordinates]
    if any(value < 0 or value > 0xFFFFFFFF for value in values):
        raise ValueError("seed coordinates must fit uint32: %s" % values)
    return values


def _noise_from_words(words: Sequence[int]) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(list(words)))
    return rng.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)


def _permutation_from_words(words: Sequence[int], size: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(list(words)))
    return np.asarray(rng.permutation(size), dtype=np.int32)


class QueryLedger:
    """Issue inference calls while maintaining an explicit route-row audit."""

    def __init__(
        self,
        client: PolicyClient,
        trace_row_offset: int,
        query_id_base: int,
        capture_routes: bool,
        existing: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> None:
        if capture_routes and client.metadata.get("episode_id_key") != EPISODE_ID_KEY:
            raise RuntimeError(
                "--capture-routes requires a server advertising episode_id_key=episode_id"
            )
        self.client = client
        self.trace_row_offset = int(trace_row_offset)
        self.query_id_base = int(query_id_base)
        self.capture_routes = bool(capture_routes)
        self.records: List[Dict[str, Any]] = [dict(item) for item in (existing or [])]
        for ordinal, record in enumerate(self.records):
            expected_row = self.trace_row_offset + ordinal if self.capture_routes else -1
            expected_id = self.query_id_base + ordinal
            if int(record.get("trace_row", -1)) != expected_row:
                raise RuntimeError("resumed query ledger has a non-contiguous trace row")
            if int(record.get("query_id", -1)) != expected_id:
                raise RuntimeError("resumed query ledger has a non-contiguous query id")

    def infer(
        self,
        policy_observation: Mapping[str, Any],
        flow_noise: np.ndarray,
        role: str,
        coordinates: Mapping[str, Any],
        seed_words: Sequence[int],
    ) -> Tuple[Dict[str, Any], int, int]:
        ordinal = len(self.records)
        trace_row = self.trace_row_offset + ordinal if self.capture_routes else -1
        query_id = self.query_id_base + ordinal
        if query_id < 0 or query_id > np.iinfo(np.int32).max:
            raise RuntimeError("query id %d does not fit recorder int32 storage" % query_id)
        request = dict(policy_observation)
        request[FLOW_NOISE_KEY] = np.ascontiguousarray(flow_noise, dtype=np.float32)
        if self.capture_routes:
            request[EPISODE_ID_KEY] = int(query_id)
        response = validate_action_response(self.client.infer(request))
        expected_noise_digest = _noise_sha256(flow_noise)
        if response.get(FLOW_NOISE_SHA256_KEY) != expected_noise_digest:
            raise RuntimeError(
                "server failed flow-noise acknowledgement for query %d (%s)"
                % (query_id, role)
            )
        actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
        record = {
            "query_id": query_id,
            "server_episode_id": query_id if self.capture_routes else None,
            "trace_row": trace_row,
            "local_call_ordinal": ordinal,
            "role": role,
            "coordinates": _json_safe(coordinates),
            "seed_words_uint32": [int(value) for value in seed_words],
            "flow_noise_sha256": expected_noise_digest,
            "actions_sha256": _sha256_array(actions),
        }
        self.records.append(record)
        return response, trace_row, query_id


def _contact_tape(
    tapes: Sequence[Sequence[Set[Tuple[str, str]]]]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    pairs = sorted(set().union(*(set_at_time for tape in tapes for set_at_time in tape)))
    lookup = {pair: index for index, pair in enumerate(pairs)}
    time_count = len(tapes[0]) if tapes else 0
    active = np.zeros((len(tapes), time_count, len(pairs)), dtype=np.bool_)
    for candidate, tape in enumerate(tapes):
        if len(tape) != time_count:
            raise RuntimeError("contact tapes have inconsistent horizons")
        for step, current in enumerate(tape):
            for pair in current:
                active[candidate, step, lookup[pair]] = True
    names = np.asarray(
        ["%s <-> %s" % pair for pair in pairs],
        dtype=("<U1" if not pairs else "<U%d" % max(len("%s <-> %s" % pair) for pair in pairs)),
    )
    return active, names, pairs


def _trajectory_pair_spread(sim_states: np.ndarray) -> Tuple[float, float]:
    count = len(sim_states)
    if count < 2:
        return 0.0, 0.0
    pairwise = np.max(
        np.abs(sim_states[:, None, :, :] - sim_states[None, :, :, :]), axis=(-2, -1)
    )
    upper = pairwise[np.triu_indices(count, k=1)]
    return float(upper.mean()), float(upper.max())


def _array_metadata(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _sha256_array(value),
        }
        for key, value in sorted(arrays.items())
    }


def _validate_snapshot_arrays(arrays: Mapping[str, np.ndarray], k: int, h: int, r: int, b: int) -> None:
    sim = arrays["sim_states"]
    d = sim.shape[-1]
    gripper_dim = arrays["gripper_qpos"].shape[-1]
    expected = {
        "actions": (k, h, 7),
        "sim_states": (k, h + 1, d),
        "eef_positions": (k, h + 1, 3),
        "eef_quaternions": (k, h + 1, 4),
        "gripper_qpos": (k, h + 1, gripper_dim),
        "chunk_success": (k, h + 1),
        "continuation_success": (k, r),
        "continuation_final_sim_states": (k, r, d),
        "continuation_action_steps": (k, r),
        "candidate_trace_rows": (k,),
        "candidate_ids": (k,),
        "execution_order": (k,),
        "candidate_query_ids": (k,),
        "candidate_flow_noise": (k,) + FLOW_NOISE_SHAPE,
        "continuation_flow_noise": (r, b) + FLOW_NOISE_SHAPE,
        "continuation_trace_rows": (k, r, b),
        "continuation_query_ids": (k, r, b),
        "continuation_query_executed": (k, r, b),
        "continuation_actions": (k, r, b, h, 7),
        "continuation_action_executed": (k, r, b, h),
        "fidelity_rerun_sim_states": (h + 1, d),
    }
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise RuntimeError("%s has shape %s, expected %s" % (key, arrays[key].shape, shape))
    if arrays["contact_active"].shape[:2] != (k, h + 1):
        raise RuntimeError("contact_active has the wrong candidate/time axes")
    if arrays["contact_active"].shape[-1] != len(arrays["contact_pair_names"]):
        raise RuntimeError("contact pair name count disagrees with contact_active")
    finite_keys = (
        "actions",
        "sim_states",
        "eef_positions",
        "eef_quaternions",
        "gripper_qpos",
        "continuation_final_sim_states",
        "candidate_flow_noise",
        "continuation_flow_noise",
        "continuation_actions",
        "fidelity_rerun_sim_states",
    )
    for key in finite_keys:
        if not np.all(np.isfinite(arrays[key])):
            raise RuntimeError("%s contains NaN or infinity" % key)
    if not np.array_equal(arrays["candidate_ids"], np.arange(k, dtype=np.int32)):
        raise RuntimeError("candidate_ids must be the candidate array axis")
    if sorted(arrays["execution_order"].tolist()) != list(range(k)):
        raise RuntimeError("execution_order is not a permutation of candidate ids")
    if not np.all(sim[:, 0] == sim[0, 0]):
        raise RuntimeError("candidate physical tapes do not start from one identical sim state")
    qpos_robot = set(int(v) for v in arrays["robot_qpos_indices"])
    qpos_object = set(int(v) for v in arrays["object_qpos_indices"])
    nq = int(arrays["nq"])
    if qpos_robot.intersection(qpos_object) or qpos_robot.union(qpos_object) != set(range(nq)):
        raise RuntimeError("robot/object qpos indices are not an exact partition")
    norm = np.linalg.norm(arrays["eef_quaternions"], axis=-1)
    if np.any(norm <= 1e-12):
        raise RuntimeError("EEF tape contains a zero-norm quaternion")
    executed_queries = arrays["continuation_query_executed"]
    candidate_rows = arrays["candidate_trace_rows"]
    route_capture = bool(np.all(candidate_rows >= 0))
    if not route_capture and not np.all(candidate_rows == -1):
        raise RuntimeError("candidate trace rows mix captured and uncaptured queries")
    if route_capture and np.any(arrays["continuation_trace_rows"][executed_queries] < 0):
        raise RuntimeError("an executed continuation query lacks a trace row")
    if not route_capture and np.any(arrays["continuation_trace_rows"][executed_queries] != -1):
        raise RuntimeError("route-disabled continuation unexpectedly has a trace row")
    if np.any(arrays["continuation_query_ids"][executed_queries] < 0):
        raise RuntimeError("an executed continuation query lacks a query id")
    if np.any(arrays["continuation_trace_rows"][~executed_queries] != -1):
        raise RuntimeError("an unexecuted continuation query has a trace row")
    action_mask = arrays["continuation_action_executed"]
    counted = action_mask.sum(axis=(-2, -1)).astype(np.int32)
    if not np.array_equal(counted, arrays["continuation_action_steps"]):
        raise RuntimeError("continuation_action_steps disagrees with its action mask")
    if np.any(arrays["continuation_action_steps"] > b * h):
        raise RuntimeError("continuation action count exceeds its configured budget")


def _collect_snapshot(
    environment: Any,
    snapshot: Mapping[str, Any],
    prompt: str,
    ledger: QueryLedger,
    snapshot_index: int,
    episode: int,
    fork_step: int,
    n_candidates: int,
    n_continuations: int,
    continuation_steps: int,
    replan_steps: int,
    seed_base: int,
    fidelity_ratio: float,
    fidelity_abs_tol: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    """Execute one fully labeled snapshot and return arrays, layout, and summary."""

    initial_observation = _hard_restore(environment, snapshot)
    policy_observation = build_policy_observation(initial_observation, prompt)
    initial_point = _physical_point(environment, initial_observation)
    state_dim = len(initial_point[0])
    gripper_dim = len(initial_point[3])
    layout = _enhanced_sim_layout(environment, state_dim)

    candidate_noises = np.empty((n_candidates,) + FLOW_NOISE_SHAPE, dtype=np.float32)
    candidate_actions = np.empty((n_candidates, replan_steps, 7), dtype=np.float32)
    candidate_trace_rows = np.empty((n_candidates,), dtype=np.int64)
    candidate_query_ids = np.empty((n_candidates,), dtype=np.int32)
    candidate_seed_words: List[List[int]] = []
    for candidate in range(n_candidates):
        words = _seed_words(
            seed_base,
            _CANDIDATE_NOISE_ROLE,
            snapshot_index,
            episode,
            fork_step,
            [candidate],
        )
        noise = _noise_from_words(words)
        response, trace_row, query_id = ledger.infer(
            policy_observation,
            noise,
            "candidate",
            {
                "snapshot_index": snapshot_index,
                "episode": episode,
                "fork_step": fork_step,
                "candidate": candidate,
            },
            words,
        )
        all_actions = np.asarray(response[ACTION_KEY], dtype=np.float32)
        if replan_steps > len(all_actions):
            raise RuntimeError("replan_steps exceeds policy action chunk")
        candidate_noises[candidate] = noise
        candidate_actions[candidate] = all_actions[:replan_steps]
        candidate_trace_rows[candidate] = trace_row
        candidate_query_ids[candidate] = query_id
        candidate_seed_words.append(words)

    continuation_noises = np.empty(
        (n_continuations, continuation_steps) + FLOW_NOISE_SHAPE, dtype=np.float32
    )
    continuation_seed_words: List[List[List[int]]] = []
    for repeat in range(n_continuations):
        repeat_words: List[List[int]] = []
        for future_step in range(continuation_steps):
            # Candidate is intentionally absent: these are common random numbers.
            words = _seed_words(
                seed_base,
                _CONTINUATION_NOISE_ROLE,
                snapshot_index,
                episode,
                fork_step,
                [repeat, future_step],
            )
            continuation_noises[repeat, future_step] = _noise_from_words(words)
            repeat_words.append(words)
        continuation_seed_words.append(repeat_words)

    order_words = _seed_words(
        seed_base,
        _EXECUTION_ORDER_ROLE,
        snapshot_index,
        episode,
        fork_step,
        [],
    )
    execution_order = _permutation_from_words(order_words, n_candidates)

    sim_states = np.empty((n_candidates, replan_steps + 1, state_dim), dtype=np.float64)
    eef_positions = np.empty((n_candidates, replan_steps + 1, 3), dtype=np.float64)
    eef_quaternions = np.empty((n_candidates, replan_steps + 1, 4), dtype=np.float64)
    gripper_qpos = np.empty(
        (n_candidates, replan_steps + 1, gripper_dim), dtype=np.float64
    )
    chunk_success = np.empty((n_candidates, replan_steps + 1), dtype=np.bool_)
    contact_sets: List[List[Set[Tuple[str, str]]]] = [
        [] for _ in range(n_candidates)
    ]
    post_chunk_snapshots: List[Optional[Mapping[str, Any]]] = [
        None for _ in range(n_candidates)
    ]
    post_chunk_full_state_sha256 = np.empty((n_candidates,), dtype="<U64")

    continuation_success = np.zeros((n_candidates, n_continuations), dtype=np.bool_)
    continuation_final_sim_states = np.empty(
        (n_candidates, n_continuations, state_dim), dtype=np.float64
    )
    continuation_action_steps = np.zeros(
        (n_candidates, n_continuations), dtype=np.int32
    )
    continuation_trace_rows = np.full(
        (n_candidates, n_continuations, continuation_steps), -1, dtype=np.int64
    )
    continuation_query_ids = np.full(
        (n_candidates, n_continuations, continuation_steps), -1, dtype=np.int32
    )
    continuation_query_executed = np.zeros(
        (n_candidates, n_continuations, continuation_steps), dtype=np.bool_
    )
    continuation_actions = np.zeros(
        (n_candidates, n_continuations, continuation_steps, replan_steps, 7),
        dtype=np.float32,
    )
    continuation_action_executed = np.zeros(
        (n_candidates, n_continuations, continuation_steps, replan_steps),
        dtype=np.bool_,
    )

    for candidate in execution_order.tolist():
        observation = _hard_restore(environment, snapshot)
        point = _physical_point(environment, observation, gripper_dim)
        sim_states[candidate, 0] = point[0]
        eef_positions[candidate, 0] = point[1]
        eef_quaternions[candidate, 0] = point[2]
        gripper_qpos[candidate, 0] = point[3]
        chunk_success[candidate, 0] = point[4]
        contact_sets[candidate].append(point[5])
        for step, action in enumerate(candidate_actions[candidate]):
            observation, _reward, _done, _info = environment.step(
                np.asarray(action, dtype=np.float64).tolist()
            )
            point = _physical_point(environment, observation, gripper_dim)
            sim_states[candidate, step + 1] = point[0]
            eef_positions[candidate, step + 1] = point[1]
            eef_quaternions[candidate, step + 1] = point[2]
            gripper_qpos[candidate, step + 1] = point[3]
            chunk_success[candidate, step + 1] = point[4]
            contact_sets[candidate].append(point[5])

        post_snapshot = save_full_state(environment)
        post_chunk_snapshots[candidate] = post_snapshot
        post_chunk_full_state_sha256[candidate] = _full_state_sha256(post_snapshot)

        for repeat in range(n_continuations):
            observation = _hard_restore(environment, post_snapshot)
            success = bool(environment.check_success())
            executed_steps = 0
            for future_step in range(continuation_steps):
                if success:
                    break
                future_policy_observation = build_policy_observation(observation, prompt)
                noise = continuation_noises[repeat, future_step]
                words = continuation_seed_words[repeat][future_step]
                response, trace_row, query_id = ledger.infer(
                    future_policy_observation,
                    noise,
                    "continuation",
                    {
                        "snapshot_index": snapshot_index,
                        "episode": episode,
                        "fork_step": fork_step,
                        "candidate": candidate,
                        "repeat": repeat,
                        "future_step": future_step,
                    },
                    words,
                )
                continuation_trace_rows[candidate, repeat, future_step] = trace_row
                continuation_query_ids[candidate, repeat, future_step] = query_id
                continuation_query_executed[candidate, repeat, future_step] = True
                actions = np.asarray(response[ACTION_KEY], dtype=np.float32)[:replan_steps]
                continuation_actions[candidate, repeat, future_step] = actions
                for action_step, action in enumerate(actions):
                    observation, _reward, _done, _info = environment.step(
                        np.asarray(action, dtype=np.float64).tolist()
                    )
                    continuation_action_executed[
                        candidate, repeat, future_step, action_step
                    ] = True
                    executed_steps += 1
                    success = bool(environment.check_success())
                    if success:
                        break
            continuation_success[candidate, repeat] = success
            continuation_action_steps[candidate, repeat] = executed_steps
            continuation_final_sim_states[candidate, repeat] = np.asarray(
                environment.get_sim_state(), dtype=np.float64
            )

    reference_candidate = int(execution_order[0])
    observation = _hard_restore(environment, snapshot)
    fidelity_states = [np.asarray(environment.get_sim_state(), dtype=np.float64).copy()]
    for action in candidate_actions[reference_candidate]:
        observation, _reward, _done, _info = environment.step(
            np.asarray(action, dtype=np.float64).tolist()
        )
        fidelity_states.append(
            np.asarray(environment.get_sim_state(), dtype=np.float64).copy()
        )
    fidelity_tape = np.stack(fidelity_states)
    fidelity_drift = float(
        np.max(np.abs(fidelity_tape - sim_states[reference_candidate]))
    )
    spread_mean, spread_max = _trajectory_pair_spread(sim_states)
    fidelity_threshold = max(
        float(fidelity_abs_tol), float(fidelity_ratio) * float(spread_mean)
    )
    fidelity_passed = bool(fidelity_drift <= fidelity_threshold)

    contact_active, contact_pair_names, raw_pairs = _contact_tape(contact_sets)
    robot_indices = np.asarray(layout["robot_qpos_indices"], dtype=np.int32)
    object_indices = np.asarray(layout["object_qpos_indices"], dtype=np.int32)
    arrays: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "snapshot_index": np.asarray(snapshot_index, dtype=np.int32),
        "episode": np.asarray(episode, dtype=np.int32),
        "fork_step": np.asarray(fork_step, dtype=np.int32),
        "nq": np.asarray(layout["nq"], dtype=np.int32),
        "actions": candidate_actions,
        "candidate_flow_noise": candidate_noises,
        "sim_states": sim_states,
        "eef_positions": eef_positions,
        "eef_quaternions": eef_quaternions,
        "gripper_qpos": gripper_qpos,
        "chunk_success": chunk_success,
        "contact_active": contact_active,
        "contact_pair_names": contact_pair_names,
        "continuation_success": continuation_success,
        "continuation_final_sim_states": continuation_final_sim_states,
        "continuation_action_steps": continuation_action_steps,
        "continuation_flow_noise": continuation_noises,
        "continuation_trace_rows": continuation_trace_rows,
        "continuation_query_ids": continuation_query_ids,
        "continuation_query_executed": continuation_query_executed,
        "continuation_actions": continuation_actions,
        "continuation_action_executed": continuation_action_executed,
        "candidate_trace_rows": candidate_trace_rows,
        "candidate_query_ids": candidate_query_ids,
        "route_capture_enabled": np.asarray(ledger.capture_routes, dtype=np.bool_),
        "candidate_ids": np.arange(n_candidates, dtype=np.int32),
        "execution_order": execution_order,
        "robot_qpos_indices": robot_indices,
        "object_qpos_indices": object_indices,
        "snapshot_sim_state": np.asarray(snapshot["sim"], dtype=np.float64),
        "snapshot_full_state_sha256": np.asarray(_full_state_sha256(snapshot)),
        "post_chunk_full_state_sha256": post_chunk_full_state_sha256,
        "fidelity_reference_candidate": np.asarray(reference_candidate, dtype=np.int32),
        "fidelity_rerun_sim_states": fidelity_tape,
        "fidelity_drift": np.asarray(fidelity_drift, dtype=np.float64),
        "fidelity_candidate_spread_mean": np.asarray(spread_mean, dtype=np.float64),
        "fidelity_candidate_spread_max": np.asarray(spread_max, dtype=np.float64),
        "fidelity_threshold": np.asarray(fidelity_threshold, dtype=np.float64),
        "fidelity_passed": np.asarray(fidelity_passed, dtype=np.bool_),
    }
    _validate_snapshot_arrays(
        arrays,
        n_candidates,
        replan_steps,
        n_continuations,
        continuation_steps,
    )

    layout.update(
        {
            "snapshot_index": snapshot_index,
            "episode": episode,
            "fork_step": fork_step,
            "candidate_axis_semantics": "all candidate-indexed arrays use candidate_ids order",
            "contact_pair_encoding": {
                "kind": "canonical unordered raw MuJoCo geom-name pair set",
                "delimiter": " <-> ",
                "multiplicity": "collapsed to presence/absence at each time point",
                "pair_names": [[left, right] for left, right in raw_pairs],
            },
            "seed_scheme": {
                "name": "numpy-seedsequence-pcg64-uint32-v1",
                "candidate_words": candidate_seed_words,
                "continuation_words": continuation_seed_words,
                "execution_order_words": order_words,
                "continuation_crn_key": [
                    "seed_base",
                    "role",
                    "snapshot_index",
                    "episode",
                    "fork_step",
                    "repeat",
                    "future_step",
                ],
                "candidate_intentionally_absent_from_continuation_key": True,
            },
            "fidelity_gate": {
                "restore": "hard: sim.reset then restore_full_state",
                "horizon_actions": replan_steps,
                "metric": "max_abs_flat_sim_over_H_plus_1",
                "candidate_spread_reduction": "mean upper-triangle pair metric",
                "max_pair_spread": spread_max,
                "ratio": fidelity_ratio,
                "absolute_tolerance": fidelity_abs_tol,
                "threshold": fidelity_threshold,
                "drift": fidelity_drift,
                "passed": fidelity_passed,
                "reference_candidate": reference_candidate,
            },
            "arrays": _array_metadata(arrays),
        }
    )
    summary = {
        "snapshot_index": snapshot_index,
        "episode": episode,
        "fork_step": fork_step,
        "candidate_count": n_candidates,
        "continuation_repeats": n_continuations,
        "continuation_control_steps": continuation_steps,
        "success_count": int(continuation_success.sum()),
        "success_total": int(continuation_success.size),
        "fidelity_passed": fidelity_passed,
        "fidelity_drift": fidelity_drift,
        "fidelity_threshold": fidelity_threshold,
        "candidate_trace_rows": candidate_trace_rows.tolist(),
        "candidate_query_ids": candidate_query_ids.tolist(),
        "execution_order": execution_order.tolist(),
        "snapshot_full_state_sha256": _full_state_sha256(snapshot),
    }
    return arrays, layout, summary


def _config_from_args(args: argparse.Namespace, episodes: Sequence[int], fork_steps: Sequence[int]) -> Dict[str, Any]:
    return {
        "host": args.host,
        "port": args.port,
        "client_dir": str(pathlib.Path(args.client_dir).expanduser().resolve()),
        "libero_root": str(pathlib.Path(args.libero_root).expanduser().resolve()),
        "benchmark": args.benchmark,
        "episodes": list(episodes),
        "fork_steps": list(fork_steps),
        "n_candidates": args.n_candidates,
        "n_continuations": args.n_continuations,
        "continuation_steps": args.continuation_steps,
        "replan_steps": args.replan_steps,
        "settle_steps": args.settle_steps,
        "seed_base": args.seed_base,
        "trace_row_offset": args.trace_row_offset,
        "query_id_base": args.query_id_base,
        "fidelity_ratio": args.fidelity_ratio,
        "fidelity_abs_tol": args.fidelity_abs_tol,
        "capture_routes": args.capture_routes,
        "exclusive_recorder_required": args.capture_routes,
    }


def _load_records(path: pathlib.Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise RuntimeError("records file has unsupported schema %r" % value.get("schema"))
    if not isinstance(value.get("snapshots"), list) or not isinstance(value.get("queries"), list):
        raise RuntimeError("records file is malformed")
    return value


def _snapshot_name(snapshot_index: int, episode: int, fork_step: int) -> str:
    return "snapshot_%04d_ep%04d_t%04d" % (snapshot_index, episode, fork_step)


def _recover_snapshot_commits(
    out: pathlib.Path,
    manifest: Dict[str, Any],
    records: Dict[str, Any],
    plan: Sequence[Mapping[str, int]],
) -> bool:
    """Adopt fully journaled snapshots and quarantine unjournaled atomic files."""

    changed = False
    artifacts = manifest.setdefault("artifacts", [])
    committed = {int(item["snapshot_index"]) for item in artifacts}
    record_by_index = {
        int(item["snapshot_index"]): item for item in records.get("snapshots", [])
    }
    for index, record in sorted(record_by_index.items()):
        if index in committed:
            continue
        npz_path = out / str(record.get("npz_file", ""))
        layout_path = out / str(record.get("layout_file", ""))
        if not npz_path.is_file() or not layout_path.is_file():
            raise RuntimeError(
                "records contain snapshot %d but its atomic artifacts are incomplete" % index
            )
        if _sha256_file(npz_path) != record.get("npz_sha256"):
            raise RuntimeError("cannot recover snapshot %d: NPZ checksum mismatch" % index)
        if _sha256_file(layout_path) != record.get("layout_sha256"):
            raise RuntimeError("cannot recover snapshot %d: layout checksum mismatch" % index)
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        artifact = {
            key: record[key]
            for key in (
                "snapshot_index",
                "episode",
                "fork_step",
                "task_id",
                "init_state_id",
                "task_name",
                "status",
                "npz_file",
                "npz_sha256",
                "layout_file",
                "layout_sha256",
            )
        }
        artifact["arrays"] = layout.get("arrays", {})
        artifacts.append(artifact)
        committed.add(index)
        changed = True

    orphan_dir = out / "orphaned"
    for item in plan:
        index = int(item["snapshot_index"])
        if index in committed:
            continue
        stem = _snapshot_name(index, int(item["episode"]), int(item["fork_step"]))
        for suffix in (".npz", ".layout.json"):
            path = out / (stem + suffix)
            if not path.exists():
                continue
            orphan_dir.mkdir(exist_ok=True)
            digest = _sha256_file(path)
            target = orphan_dir / (path.name + "." + digest[:12])
            if target.exists():
                raise RuntimeError("orphan quarantine target already exists: %s" % target)
            path.replace(target)
            manifest.setdefault("orphaned_files", []).append(
                {
                    "original": path.name,
                    "quarantined": str(target.relative_to(out)),
                    "sha256": digest,
                    "recovered_utc": _utc_now(),
                }
            )
            changed = True
    if changed:
        manifest["query_count"] = len(records.get("queries", []))
        manifest["records_file_sha256"] = _sha256_file(out / RECORDS_FILE)
        manifest["updated_utc"] = _utc_now()
        _atomic_write_json(out / MANIFEST_FILE, manifest)
    return changed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--client-dir", required=True,
                        help="recorded rollout whose action prefix reaches each snapshot")
    parser.add_argument("--libero-root", required=True)
    parser.add_argument("--benchmark", default="libero_goal")
    parser.add_argument("--episodes", default="0,1,2,3,4")
    parser.add_argument("--fork-steps", default="6,10,11,12")
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument(
        "--n-continuations",
        type=int,
        required=True,
        help=(
            "CRN repeats R; choose this explicitly. For a 95%% paired equivalence "
            "interval wholly inside +/-0.1 even with zero discordance, R must be at least 42"
        ),
    )
    parser.add_argument("--continuation-steps", type=int, default=5,
                        help="future policy control steps per continuation repeat")
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=1701)
    parser.add_argument("--trace-row-offset", type=int, default=0,
                        help="routes.zarr row of this script's first inference; recorder must be exclusive")
    parser.add_argument("--query-id-base", type=int, default=0,
                        help="int32 server episode_id assigned to this script's first query")
    parser.add_argument(
        "--capture-routes",
        action="store_true",
        help=(
            "join every query to an exclusive route-recorder server; phase-1 physics/Q "
            "capture should normally leave this off"
        ),
    )
    parser.add_argument("--fidelity-ratio", type=float, default=0.05)
    parser.add_argument("--fidelity-abs-tol", type=float, default=1e-10)
    parser.add_argument("--continue-on-fidelity-failure", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "recover/continue a route-off manifest; interrupted route-on manifests "
            "are rejected because persisted server rows cannot be acknowledged"
        ),
    )
    parser.add_argument("--inference-timeout", type=float, default=300.0)
    parser.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    episodes = _parse_int_list(args.episodes, "--episodes")
    fork_steps = _parse_int_list(args.fork_steps, "--fork-steps")
    if args.n_candidates < 2:
        raise SystemExit("--n-candidates must be at least 2 for a fidelity spread")
    if args.n_continuations < 1:
        raise SystemExit("--n-continuations must be positive")
    if args.continuation_steps < 1:
        raise SystemExit("--continuation-steps must be positive")
    if args.replan_steps < 1 or args.replan_steps > FLOW_NOISE_SHAPE[0]:
        raise SystemExit("--replan-steps must be in 1..%d" % FLOW_NOISE_SHAPE[0])
    if args.seed_base < 0 or args.seed_base > 0xFFFFFFFF:
        raise SystemExit("--seed-base must fit uint32")
    if args.trace_row_offset < 0 or args.query_id_base < 0:
        raise SystemExit("trace-row and query-id bases must be non-negative")
    if not (0.0 <= args.fidelity_ratio <= 1.0):
        raise SystemExit("--fidelity-ratio must be in [0,1]")
    if not np.isfinite(args.fidelity_abs_tol) or args.fidelity_abs_tol < 0.0:
        raise SystemExit("--fidelity-abs-tol must be finite and non-negative")

    source = pathlib.Path(args.client_dir).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser().resolve()
    summaries_path = source / "summaries.json"
    source_metadata_path = source / "server_metadata.json"
    if not source_metadata_path.is_file():
        raise SystemExit("source client directory has no server_metadata.json")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    summaries_raw = json.loads(summaries_path.read_text(encoding="utf-8"))
    summaries = {int(item["episode_index"]): item for item in summaries_raw}
    missing = sorted(set(episodes) - set(summaries))
    if missing:
        raise SystemExit("source summaries are missing episodes %s" % missing)
    for episode in episodes:
        for fork_step in fork_steps:
            if fork_step >= int(summaries[episode]["inference_calls"]):
                raise SystemExit(
                    "episode %d has only %d source control steps; cannot fork at %d"
                    % (episode, summaries[episode]["inference_calls"], fork_step)
                )

    plan = [
        {"snapshot_index": index, "episode": episode, "fork_step": fork_step}
        for index, (episode, fork_step) in enumerate(
            [(episode, step) for episode in episodes for step in fork_steps]
        )
    ]
    config = _config_from_args(args, episodes, fork_steps)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path, records_path = out / MANIFEST_FILE, out / RECORDS_FILE
    manifest: Dict[str, Any]
    records: Dict[str, Any]
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit("output already has a manifest; pass --resume or choose a new --out")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise SystemExit("cannot resume schema %r" % manifest.get("schema"))
        if manifest.get("config") != config:
            raise SystemExit("resume configuration differs from the existing manifest")
        records = _load_records(records_path)
    else:
        existing = list(out.iterdir())
        if existing:
            raise SystemExit("output directory is non-empty but has no manifest: %s" % out)
        records = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "snapshots": [], "queries": []}
        manifest = {
            "schema": SCHEMA,
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "status": "initializing",
            "complete": False,
            "created_utc": _utc_now(),
            "updated_utc": _utc_now(),
            "config": config,
            "array_schema": ARRAY_SCHEMA,
            "records_file": RECORDS_FILE,
            "source": {
                "summaries_file": str(summaries_path),
                "summaries_sha256": _sha256_file(summaries_path),
                "server_metadata_file": str(source_metadata_path),
                "server_metadata_sha256": _sha256_file(source_metadata_path),
            },
            "planned_snapshots": plan,
            "artifacts": [],
            "query_count": 0,
            "next_trace_row": args.trace_row_offset,
            "trace_row_contract": (
                "trace_row_offset + local_call_ordinal with --capture-routes and exclusive "
                "recorder access; otherwise every trace row is -1"
            ),
            "seed_scheme": "numpy-seedsequence-pcg64-uint32-v1",
            "server_metadata_file": "server_metadata.json",
        }
        _atomic_write_json(records_path, records)
        manifest["records_file_sha256"] = _sha256_file(records_path)
        _atomic_write_json(manifest_path, manifest)

    recorded_source = manifest.get("source", {})
    if recorded_source.get("summaries_sha256") != _sha256_file(summaries_path):
        raise SystemExit("source summaries changed since this run was initialized")
    if recorded_source.get("server_metadata_sha256") != _sha256_file(source_metadata_path):
        raise SystemExit("source server metadata changed since this run was initialized")
    if args.resume and args.capture_routes and manifest.get("status") != "completed":
        raise SystemExit(
            "route-captured interrupted runs cannot resume safely because the server does "
            "not acknowledge persisted route rows; start a new output directory"
        )
    if args.resume:
        _recover_snapshot_commits(out, manifest, records, plan)
    committed_artifacts = manifest.get("artifacts", [])
    committed = {int(item["snapshot_index"]) for item in committed_artifacts}
    for artifact in committed_artifacts:
        for file_key, digest_key in (
            ("npz_file", "npz_sha256"),
            ("layout_file", "layout_sha256"),
        ):
            artifact_path = out / str(artifact[file_key])
            if not artifact_path.is_file():
                raise SystemExit("committed artifact is missing: %s" % artifact_path)
            if _sha256_file(artifact_path) != artifact[digest_key]:
                raise SystemExit("committed artifact checksum mismatch: %s" % artifact_path)
    failed_committed = [
        int(item["snapshot_index"])
        for item in committed_artifacts
        if item.get("status") == "fidelity_failed"
    ]
    if failed_committed and not args.continue_on_fidelity_failure:
        raise SystemExit(
            "existing snapshots failed fidelity %s; resume requires "
            "--continue-on-fidelity-failure or a new output directory"
            % failed_committed
        )
    expected_query_count = len(records["queries"])
    if int(manifest.get("query_count", -1)) != expected_query_count:
        raise SystemExit("manifest and records disagree on query_count")

    try:
        with PolicyClient(
            host=args.host,
            port=args.port,
            connect_timeout=600.0,
            inference_timeout=args.inference_timeout,
        ) as client:
            validate_policy_suite(client.metadata, args.benchmark)
            raw_action_std = client.metadata.get("normalization_action_std")
            if raw_action_std is None:
                raise RuntimeError("server metadata is missing normalization_action_std")
            action_std = np.asarray(raw_action_std, dtype=np.float64)
            if (
                action_std.shape != (7,)
                or not np.all(np.isfinite(action_std))
                or np.any(action_std <= 0.0)
            ):
                raise RuntimeError(
                    "server normalization_action_std must contain seven finite positive values"
                )
            source_policy_checks = {
                "protocol": source_metadata.get("protocol"),
                "checkpoint_sha256": source_metadata.get("checkpoint_sha256"),
                "libero_wrist_layout": source_metadata.get("libero_wrist_layout"),
                "normalization_action_std": source_metadata.get("normalization_action_std"),
            }
            current_policy_checks = {
                key: _json_safe(client.metadata.get(key)) for key in source_policy_checks
            }
            mismatched = [
                key
                for key, source_value in source_policy_checks.items()
                if source_value is not None
                and current_policy_checks.get(key) is not None
                and _json_safe(source_value) != current_policy_checks[key]
            ]
            if mismatched:
                raise RuntimeError(
                    "source rollout and candidate policy differ on %s" % ", ".join(mismatched)
                )
            manifest["source_policy_identity"] = _json_safe(source_policy_checks)
            if args.capture_routes and client.metadata.get("episode_id_key") != EPISODE_ID_KEY:
                raise RuntimeError(
                    "--capture-routes requires serve_with_recorder.py episode_id support"
                )
            identity_keys = (
                "protocol",
                "protocol_version",
                "checkpoint_sha256",
                "checkpoint_path",
                "libero_wrist_layout",
                "route_recorder",
                "n_action_steps",
                "num_steps",
                "normalization_action_mean",
                "normalization_action_std",
            )
            if args.capture_routes:
                identity_keys += ("server_instance_id", "server_started_unix_ns")
            server_identity = {
                key: _json_safe(client.metadata.get(key)) for key in identity_keys
            }
            prior_identity = manifest.get("server_identity")
            if prior_identity is not None and prior_identity != server_identity:
                raise RuntimeError("server policy identity differs from the resumed manifest")
            manifest["server_identity"] = server_identity
            manifest["server_metadata"] = _json_safe(client.metadata)
            server_metadata_path = out / "server_metadata.json"
            _atomic_write_json(server_metadata_path, client.metadata)
            manifest["server_metadata_sha256"] = _sha256_file(server_metadata_path)
            manifest["status"] = "running"
            manifest.pop("error", None)
            manifest["updated_utc"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)

            ledger = QueryLedger(
                client,
                trace_row_offset=args.trace_row_offset,
                query_id_base=args.query_id_base,
                capture_routes=args.capture_routes,
                existing=records["queries"],
            )
            for item in plan:
                snapshot_index = int(item["snapshot_index"])
                episode = int(item["episode"])
                fork_step = int(item["fork_step"])
                if snapshot_index in committed:
                    print("snapshot %d already committed; skipping" % snapshot_index, flush=True)
                    continue
                query_start = None
                summary = summaries[episode]
                source_episode = source / ("episode_%02d.npz" % episode)
                with np.load(source_episode, allow_pickle=False) as archive:
                    recorded_actions = np.asarray(archive["actions"], dtype=np.float32)
                if recorded_actions.ndim != 3 or recorded_actions.shape[-1] != 7:
                    raise RuntimeError("source episode actions have invalid shape %s" % (recorded_actions.shape,))
                if recorded_actions.shape[1] < args.replan_steps:
                    raise RuntimeError("source action chunks are shorter than --replan-steps")

                episode_config = EpisodeConfig(
                    task_suite=args.benchmark,
                    task_id=int(summary["task_id"]),
                    init_state_id=int(summary["init_state_id"]),
                    seed=int(summary["seed"]),
                    host=args.host,
                    port=args.port,
                    libero_root=args.libero_root,
                    output_root=str(out),
                    settle_steps=args.settle_steps,
                    max_steps=max(1000, (fork_step + args.continuation_steps + 2) * args.replan_steps),
                    replan_steps=args.replan_steps,
                    inference_timeout=args.inference_timeout,
                )
                environment, observation, task, prompt = _load_task(episode_config)
                try:
                    for _ in range(episode_config.settle_steps):
                        observation, _reward, _done, _info = environment.step(
                            LIBERO_DUMMY_ACTION.tolist()
                        )
                    for prefix_step in range(fork_step):
                        for action in recorded_actions[prefix_step, : args.replan_steps]:
                            observation, _reward, _done, _info = environment.step(
                                np.asarray(action, dtype=np.float64).tolist()
                            )
                    snapshot = save_full_state(environment)
                    query_start = len(ledger.records)
                    arrays, layout, snapshot_summary = _collect_snapshot(
                        environment=environment,
                        snapshot=snapshot,
                        prompt=prompt,
                        ledger=ledger,
                        snapshot_index=snapshot_index,
                        episode=episode,
                        fork_step=fork_step,
                        n_candidates=args.n_candidates,
                        n_continuations=args.n_continuations,
                        continuation_steps=args.continuation_steps,
                        replan_steps=args.replan_steps,
                        seed_base=args.seed_base,
                        fidelity_ratio=args.fidelity_ratio,
                        fidelity_abs_tol=args.fidelity_abs_tol,
                    )
                    query_end = len(ledger.records)
                    for query in ledger.records[query_start:query_end]:
                        query["artifact_snapshot_index"] = snapshot_index
                finally:
                    try:
                        environment.close()
                    except BaseException:
                        pass

                stem = _snapshot_name(snapshot_index, episode, fork_step)
                npz_path = out / (stem + ".npz")
                layout_path = out / (stem + ".layout.json")
                if npz_path.exists() or layout_path.exists():
                    raise RuntimeError(
                        "uncommitted snapshot artifact already exists; preserve it for audit: %s"
                        % stem
                    )
                layout.update(
                    {
                        "task_id": int(summary["task_id"]),
                        "init_state_id": int(summary["init_state_id"]),
                        "task_name": str(task.name),
                        "prompt": prompt,
                        "source_episode_file": str(source_episode),
                        "source_episode_sha256": _sha256_file(source_episode),
                    }
                )
                _atomic_write_npz(npz_path, arrays)
                _atomic_write_json(layout_path, layout)
                npz_digest, layout_digest = _sha256_file(npz_path), _sha256_file(layout_path)

                snapshot_record = dict(snapshot_summary)
                snapshot_record.update(
                    {
                        "status": (
                            "complete" if snapshot_summary["fidelity_passed"] else "fidelity_failed"
                        ),
                        "task_id": int(summary["task_id"]),
                        "init_state_id": int(summary["init_state_id"]),
                        "task_name": str(task.name),
                        "prompt": prompt,
                        "npz_file": npz_path.name,
                        "layout_file": layout_path.name,
                        "npz_sha256": npz_digest,
                        "layout_sha256": layout_digest,
                        "query_ordinal_range": [query_start, query_end],
                    }
                )
                records["snapshots"].append(snapshot_record)
                records["queries"] = ledger.records
                _atomic_write_json(records_path, records)
                artifact = {
                    "snapshot_index": snapshot_index,
                    "episode": episode,
                    "fork_step": fork_step,
                    "task_id": int(summary["task_id"]),
                    "init_state_id": int(summary["init_state_id"]),
                    "task_name": str(task.name),
                    "status": snapshot_record["status"],
                    "npz_file": npz_path.name,
                    "npz_sha256": npz_digest,
                    "layout_file": layout_path.name,
                    "layout_sha256": layout_digest,
                    "arrays": _array_metadata(arrays),
                }
                manifest.setdefault("artifacts", []).append(artifact)
                manifest["query_count"] = len(ledger.records)
                manifest["next_trace_row"] = (
                    args.trace_row_offset + len(ledger.records)
                    if args.capture_routes
                    else args.trace_row_offset
                )
                manifest["records_file_sha256"] = _sha256_file(records_path)
                manifest["updated_utc"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                print(
                    "snapshot %-3d ep %-3d step %-3d | success %d/%d | fidelity %.3e <= %.3e: %s"
                    % (
                        snapshot_index,
                        episode,
                        fork_step,
                        snapshot_summary["success_count"],
                        snapshot_summary["success_total"],
                        snapshot_summary["fidelity_drift"],
                        snapshot_summary["fidelity_threshold"],
                        snapshot_summary["fidelity_passed"],
                    ),
                    flush=True,
                )
                if not snapshot_summary["fidelity_passed"] and not args.continue_on_fidelity_failure:
                    raise FidelityGateError(
                        "snapshot %d failed hard-restore fidelity gate" % snapshot_index
                    )

            records["queries"] = ledger.records
            _atomic_write_json(records_path, records)
            manifest["query_count"] = len(ledger.records)
            manifest["next_trace_row"] = (
                args.trace_row_offset + len(ledger.records)
                if args.capture_routes
                else args.trace_row_offset
            )
            manifest["records_file_sha256"] = _sha256_file(records_path)
            manifest["status"] = (
                "completed_with_fidelity_failures"
                if any(
                    item.get("status") == "fidelity_failed"
                    for item in manifest.get("artifacts", [])
                )
                else "completed"
            )
            manifest["complete"] = manifest["status"] == "completed"
            manifest["completed_utc"] = _utc_now()
            manifest["updated_utc"] = manifest["completed_utc"]
            _atomic_write_json(manifest_path, manifest)
    except BaseException as error:
        # A server may have committed an inference row before a client-side error.
        # Keep every acknowledged query and mark the run non-resumable-by-assumption
        # until the operator verifies the recorder's row count.
        if "ledger" in locals():
            committed_indices = {
                int(item["snapshot_index"]) for item in manifest.get("artifacts", [])
            }
            query_start_value = locals().get("query_start")
            if (
                query_start_value is not None
                and "snapshot_index" in locals()
                and int(snapshot_index) not in committed_indices
            ):
                for query in ledger.records[int(query_start_value) :]:
                    query["abandoned"] = True
            records["queries"] = ledger.records
            _atomic_write_json(records_path, records)
            manifest["query_count"] = len(ledger.records)
            manifest["next_trace_row"] = (
                args.trace_row_offset + len(ledger.records)
                if args.capture_routes
                else args.trace_row_offset
            )
            manifest["records_file_sha256"] = _sha256_file(records_path)
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["complete"] = False
        manifest["updated_utc"] = _utc_now()
        manifest["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "recorder_resume_warning": (
                "verify routes.zarr row count before --resume; an unacknowledged server row "
                "cannot be inferred from the client ledger"
            ),
        }
        _atomic_write_json(manifest_path, manifest)
        raise

    print(
        "wrote %d snapshots and %d query records to %s"
        % (len(manifest.get("artifacts", [])), manifest["query_count"], out),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
