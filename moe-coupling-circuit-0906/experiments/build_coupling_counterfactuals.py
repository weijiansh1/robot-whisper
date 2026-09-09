#!/usr/bin/env python3
"""Render matched counterfactual observations that differ only in the object.

The routing interface separates a coupled object from one that stayed behind
(``moe-progress-ratio-v12-0906/results/object_readability``).  That is a readout
result on an observational contrast: phantom events matched to coupled controls
on task, arm travel and query index.  Matching cannot rule out everything, and
the artifacts already carry the caveat that every phantom sits in a failed
episode while the coupled pool is 96% successful.

This script replaces matching with construction.  One restored MuJoCo state
produces every condition, so the robot's configuration, the scene, the camera
and the instruction are bit-identical across conditions and the only difference
is where the target object is.  ``PROTOCOL.md`` freezes the design; this script
implements it and refuses to write rows that fail its gates.

Two anchors per episode, both defined from gripper aperture and target position
and never from a policy internal:

  q_pre   last query with the aperture still open and the end effector within
          ``NEAR_TARGET_M`` of the target -- the approach
  q_post  first query after closure at which the target has lifted at least
          ``LIFTED_M`` *and* the end effector has departed at least
          ``EEF_DEPARTURE_M`` -- the object is being carried away

Conditions at ``q_pre`` are ``held`` plus a graded perpendicular displacement of
the target (``shift2``/``shift4``/``shift8``), which is the positive control:
a behaviour the policy can be expected to perform, supplying the reference
computation the failure case lacks.  Conditions at ``q_post`` are ``held`` and
``uncoupled``, the target written back to its query-0 pose.  Both anchors carry
a ``light`` condition, a visual change that should not change the action.

An anchor is admitted only if *every* one of its conditions passes every gate,
so the design stays balanced.  Anchors that fail are written out with a reason.

Rendering is CPU-only through osmesa; no policy is loaded here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "himoe.coupling_counterfactual_inputs.v1"

# Behaviour thresholds, shared with analysis_trap_taxonomy/validate_belief_mismatch.py
CLOSE_APERTURE = 0.05
NEAR_TARGET_M = 0.16
LIFTED_M = 0.02
EEF_DEPARTURE_M = 0.10
APPROACH_NEAR_M = 0.10

# Frozen validity gates (PROTOCOL.md)
POSITION_TOLERANCE_M = 2e-3
ROTATION_TOLERANCE_RAD = 1e-2
# 1e-3 m, not the 1e-4 copied from render_rich_event_inputs.py.  Measured on 60
# episodes of this capture: with the gripper closed and static (q_post) the
# restore error is ~1e-6, but with the fingers open and moving (q_pre) it is
# median 1.14e-4, p90 2.02e-4, max 3.41e-4, so 1e-4 rejected 49 of 208 approach
# anchors on a 0.1 mm discrepancy in finger position -- three orders below the
# 20-80 mm object displacement the study manipulates, and half the position
# tolerance already accepted.  The threshold was deciding on quantisation.
GRIPPER_TOLERANCE = 1e-3
SIM_STATE_TOLERANCE = 1e-9
SUPPORT_HEIGHT_TOLERANCE_M = 5e-3
MIN_BEARING_M = 1e-2
VISIBILITY_PIXEL_DELTA = 8
VISIBILITY_PIXEL_FRACTION = 0.01

SHIFTS_M = (0.02, 0.04, 0.08)
LIGHT_OFFSET_M = np.asarray([0.25, -0.20, 0.15], np.float64)

def _flow_noise_shape() -> tuple[int, int]:
    """Take the shape from the protocol, never a copied literal.

    A hardcoded (10, 32) here produced noise the model rejected: the internal
    action dim is 24, and the wrong width also consumes the RNG differently, so
    the draw would not have reproduced the capture's noise either.
    """
    try:
        from himoe_libero_bridge.protocol import FLOW_NOISE_SHAPE as shape
    except ImportError:  # the unit tests run without the bridge on the path
        shape = (10, 24)
    return tuple(int(value) for value in shape)


FLOW_NOISE_SHAPE = _flow_noise_shape()


def jsonable(value: Any) -> Any:
    """Coerce numpy scalars and arrays so the audit is a plain JSON document."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_bytes(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qvel_slices(layout: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """Map each joint to its slice of the flattened state's qvel block.

    ``sim_layout.json`` records qpos ranges only.  MuJoCo's qvel width differs
    from qpos for free (7 -> 6) and ball (4 -> 3) joints, so the offsets are
    rebuilt by walking the joints in declaration order.  The walk is checked
    against ``nv``; a mismatch is a hard error rather than a silent misalignment
    that would zero the wrong body's velocity.
    """
    qpos_to_qvel = {7: 6, 4: 3, 1: 1}
    velocity_start = 1 + int(layout["nq"])
    offset = 0
    result: dict[str, tuple[int, int]] = {}
    for joint in layout["joints"]:
        width = int(joint["state_hi"]) - int(joint["state_lo"])
        if width not in qpos_to_qvel:
            raise RuntimeError("unsupported qpos width %d for %s" % (width, joint["joint"]))
        span = qpos_to_qvel[width]
        result[str(joint["joint"])] = (velocity_start + offset, velocity_start + offset + span)
        offset += span
    if offset != int(layout["nv"]):
        raise RuntimeError("qvel walk produced %d dims, layout declares %d" % (offset, layout["nv"]))
    if velocity_start + offset != int(layout["state_dim"]):
        raise RuntimeError("qvel walk does not reach the declared state_dim")
    return result


def anchors(state: np.ndarray, sim: np.ndarray, lo: int) -> dict[str, Any]:
    """Locate the approach and the established-coupling query, or say why not."""
    aperture = np.abs(state[:, 6:8]).sum(axis=1)
    target = sim[:, lo : lo + 3]
    eef = state[:, :3]
    distance = np.linalg.norm(eef - target, axis=1)

    closed = aperture < CLOSE_APERTURE
    crossings = np.flatnonzero((~closed[:-1]) & closed[1:]) + 1
    if not len(crossings):
        return {"ok": False, "reason": "no aperture closure"}
    closure = int(crossings[0])
    if distance[closure] >= NEAR_TARGET_M:
        return {"ok": False, "reason": "closure is not near the target"}

    open_and_near = np.flatnonzero((~closed[:closure]) & (distance[:closure] < APPROACH_NEAR_M))
    if not len(open_and_near):
        return {"ok": False, "reason": "no open-gripper query near the target"}
    q_pre = int(open_and_near[-1])

    # q_post must be far enough along that "the object stayed behind" is a
    # physically sensible state to render: at the first query with a 2 cm lift
    # the gripper is still 2 cm above the table, so writing the object back to
    # its resting pose puts it inside the closed fingers.  Requiring the same
    # end-effector departure the taxonomy uses for a phantom grasp
    # (EEF_DEPARTURE_M, with the target stationary) both clears the gripper and
    # makes `uncoupled` a constructed instance of exactly that event class.
    lift = target[closure:, 2] - target[closure, 2]
    departure = np.linalg.norm(eef[closure:] - eef[closure], axis=1)
    coupled_and_departed = np.flatnonzero((lift >= LIFTED_M) & (departure >= EEF_DEPARTURE_M))
    if not len(coupled_and_departed):
        return {"ok": False, "reason": "target never lifted and carried away"}
    q_post = closure + int(coupled_and_departed[0])
    if q_post >= len(state):
        return {"ok": False, "reason": "transport query outside the trace"}
    return {"ok": True, "q_pre": q_pre, "q_post": q_post, "closure": closure}


def bearing(state_row: np.ndarray, target_xyz: np.ndarray) -> np.ndarray | None:
    """Unit horizontal vector from the end effector to the target, or None."""
    horizontal = np.asarray(target_xyz[:2], np.float64) - np.asarray(state_row[:2], np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm < MIN_BEARING_M:
        return None
    return horizontal / norm


def flow_noise_at(seed: int, query: int) -> np.ndarray:
    """Reproduce the episode's flow noise draw, as the capture client drew it."""
    generator = np.random.default_rng(seed)
    noise = None
    for _ in range(query + 1):
        noise = generator.standard_normal(FLOW_NOISE_SHAPE).astype(np.float32)
    if noise is None:
        raise ValueError("query must be non-negative")
    return noise


class Scene:
    """The one LIBERO environment, plus the geom bookkeeping the gates need."""

    def __init__(self, environment: Any, target_joint: str) -> None:
        self.environment = environment
        self.sim = environment.sim
        model = self.sim.model
        joint_id = model.joint_name2id(target_joint)
        target_body = int(model.jnt_bodyid[joint_id])
        bodies = {target_body}
        for body in range(model.nbody):
            parent = body
            while parent != 0:
                if parent == target_body:
                    bodies.add(body)
                    break
                parent = int(model.body_parentid[parent])
        self.target_geoms = {
            geom for geom in range(model.ngeom) if int(model.geom_bodyid[geom]) in bodies
        }
        if not self.target_geoms:
            raise RuntimeError("target joint %s owns no geoms" % target_joint)
        self.robot_geoms = {
            geom
            for geom in range(model.ngeom)
            if str(model.body_id2name(int(model.geom_bodyid[geom])) or "").startswith(
                ("robot0", "gripper0")
            )
        }
        if not self.robot_geoms:
            raise RuntimeError("no robot geoms found")
        self.light_pos = np.array(model.light_pos, np.float64, copy=True)

    def restore(self, sim_state: np.ndarray, prompt: str) -> dict[str, Any]:
        from himoe_libero_bridge.preprocess import build_policy_observation

        observation = self.environment.regenerate_obs_from_state(sim_state)
        return build_policy_observation(observation, prompt)

    def restored_state(self) -> np.ndarray:
        return np.asarray(self.environment.get_sim_state(), np.float64)

    def probe(self, sim_state: np.ndarray) -> dict[str, Any]:
        """Contacts for a candidate state without paying for two camera renders.

        The shift-sign search only needs to know whether the displaced object
        would touch the robot or float, and rendering dominates the cost of a
        restore.  ``regenerate_obs_from_state`` is ``set_state`` plus ``forward``
        plus observation assembly, so the first two alone give the same contact
        set.
        """
        self.environment.set_state(sim_state)
        self.sim.forward()
        return self.contact_report()

    def contact_report(self) -> dict[str, Any]:
        """Target-robot clearance and whether the target rests on something.

        MuJoCo only reports contacts inside its collision margin, so an absent
        pair means "not touching", not "far".  ``robot_min_dist`` is therefore
        the minimum *reported* penetration depth and is None when the target and
        the robot are not in contact at all -- which is the passing case.
        """
        data = self.sim.data
        robot_depths = []
        support = 0
        for index in range(int(data.ncon)):
            contact = data.contact[index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            in1, in2 = geom1 in self.target_geoms, geom2 in self.target_geoms
            if in1 == in2:
                continue
            other = geom2 if in1 else geom1
            if other in self.robot_geoms:
                robot_depths.append(float(contact.dist))
            else:
                support += 1
        return {
            "robot_contacts": len(robot_depths),
            "robot_min_dist": min(robot_depths) if robot_depths else None,
            "support_contacts": support,
        }

    def perturb_light(self) -> None:
        self.sim.model.light_pos[:] = self.light_pos + LIGHT_OFFSET_M

    def reset_light(self) -> None:
        self.sim.model.light_pos[:] = self.light_pos
        if not np.array_equal(np.asarray(self.sim.model.light_pos, np.float64), self.light_pos):
            raise RuntimeError("light position was not restored exactly")


def visibility(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    passed = False
    for key in ("image", "wrist_image"):
        delta = np.abs(candidate[key].astype(np.int16) - reference[key].astype(np.int16))
        fraction = float((delta.max(axis=2) >= VISIBILITY_PIXEL_DELTA).mean())
        result["%s_changed_fraction" % key] = fraction
        result["%s_max_delta" % key] = int(delta.max())
        passed = passed or fraction >= VISIBILITY_PIXEL_FRACTION
    result["passed"] = passed
    return result


def render_condition(
    scene: Scene,
    prompt: str,
    sim_state: np.ndarray,
    *,
    light: bool,
) -> dict[str, Any]:
    if light:
        scene.perturb_light()
    try:
        policy_input = scene.restore(sim_state, prompt)
        restored = scene.restored_state()
        contacts = scene.contact_report()
    finally:
        if light:
            scene.reset_light()
    image = np.asarray(policy_input["observation/image"], np.uint8)
    wrist = np.asarray(policy_input["observation/wrist_image"], np.uint8)
    if image.shape != (224, 224, 3) or wrist.shape != image.shape:
        raise RuntimeError("restored image shape mismatch")
    return {
        "image": image,
        "wrist_image": wrist,
        "state": np.asarray(policy_input["observation/state"], np.float32),
        "restored_sim": restored,
        "contacts": contacts,
        "blank": bool(np.ptp(image) == 0 or np.ptp(wrist) == 0),
        "sim_error": float(np.abs(restored - sim_state).max()),
    }


def edited_state(
    base: np.ndarray,
    lo: int,
    velocity: tuple[int, int],
    *,
    translation: np.ndarray | None = None,
    pose: np.ndarray | None = None,
) -> np.ndarray:
    row = np.array(base, np.float64, copy=True)
    if pose is not None:
        row[lo : lo + 7] = pose
    if translation is not None:
        row[lo : lo + 3] = row[lo : lo + 3] + translation
    row[velocity[0] : velocity[1]] = 0.0
    return row


def build(args: argparse.Namespace) -> dict[str, Any]:
    from himoe_libero_bridge.libero_runtime import EpisodeConfig, _load_task

    client = args.client_dir
    layout = json.loads((client / "sim_layout.json").read_text(encoding="utf-8"))
    summaries = json.loads((client / "summaries.json").read_text(encoding="utf-8"))
    targets = {
        str(row["task"]): row
        for row in json.loads(args.target_audit.read_text(encoding="utf-8"))
    }
    task_key = "%s/%s" % (args.suite, client.parents[1].name)
    audited = targets.get(task_key)
    if audited is None:
        raise RuntimeError("no target audit entry for %s" % task_key)
    if len(audited["target_names"]) != 1:
        raise RuntimeError(
            "this study needs exactly one target object, %s has %d"
            % (task_key, len(audited["target_names"]))
        )
    target_joint = str(audited["target_names"][0])
    joints = {str(item["joint"]): item for item in layout["joints"]}
    if target_joint not in joints:
        raise RuntimeError("target %s absent from sim layout" % target_joint)
    lo = int(joints[target_joint]["state_lo"])
    hi = int(joints[target_joint]["state_hi"])
    if hi - lo != 7:
        raise RuntimeError("target %s is not a free joint" % target_joint)
    velocity = qvel_slices(layout)[target_joint]

    task_ids = {int(row["task_id"]) for row in summaries}
    if len(task_ids) != 1:
        raise RuntimeError("capture mixes task ids: %s" % sorted(task_ids))
    config = EpisodeConfig(
        libero_root=str(args.libero_root),
        output_root=str(args.out.parent),
        task_suite=args.suite,
        task_id=task_ids.pop(),
        init_state_id=0,
        seed=int(summaries[0]["seed"]),
        settle_steps=10,
        max_steps=520,
        replan_steps=10,
        render_size=224,
    )
    environment, _observation, task, prompt = _load_task(config)
    if str(task.name) != client.parents[1].name:
        raise RuntimeError(
            "LIBERO task identity mismatch: %s against %s" % (task.name, client.parents[1].name)
        )

    scene = Scene(environment, target_joint)
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[Any]] = {key: [] for key in ("image", "wrist_image", "state", "flow_noise")}
    drops: list[dict[str, Any]] = []
    unit_id = 0
    # Budget per anchor, not shared.  Post anchors admit at close to 100% while
    # approach anchors do not, so a shared budget fills with post units and
    # starves the stage-1 gate.
    admitted: dict[str, int] = {"pre": 0, "post": 0}
    try:
        for summary in summaries:
            if all(count >= args.max_units_per_anchor for count in admitted.values()):
                break
            episode_index = int(summary["episode_index"])
            if not bool(summary["success"]):
                drops.append({"episode_index": episode_index, "reason": "episode did not succeed"})
                continue
            episode_path = client / ("episode_%02d.npz" % episode_index)
            if not episode_path.is_file():
                drops.append({"episode_index": episode_index, "reason": "episode npz missing"})
                continue
            with np.load(episode_path, allow_pickle=False) as episode:
                state = np.asarray(episode["state"], np.float32)
                sim = np.asarray(episode["sim_state"], np.float64)
                actions = np.asarray(episode["actions"], np.float32)
            located = anchors(state, sim, lo)
            if not located["ok"]:
                drops.append({"episode_index": episode_index, "reason": located["reason"]})
                continue

            for anchor_name, query in (("pre", located["q_pre"]), ("post", located["q_post"])):
                if admitted[anchor_name] >= args.max_units_per_anchor:
                    continue
                unit = build_unit(
                    scene=scene,
                    prompt=prompt,
                    anchor=anchor_name,
                    query=int(query),
                    state=state,
                    sim=sim,
                    actions=actions,
                    lo=lo,
                    velocity=velocity,
                    summary=summary,
                    episode_path=episode_path,
                )
                if not unit["ok"]:
                    drops.append(
                        {
                            "episode_index": episode_index,
                            "anchor": anchor_name,
                            "query_index": int(query),
                            "reason": unit["reason"],
                            **{k: v for k, v in unit.items() if k.startswith("detail_")},
                        }
                    )
                    continue
                for condition in unit["conditions"]:
                    for key in arrays:
                        arrays[key].append(condition.pop(key))
                    condition["unit_id"] = unit_id
                    rows.append(condition)
                unit_id += 1
                admitted[anchor_name] += 1
                print(
                    "unit %d  episode %d  anchor %s  query %d"
                    % (unit_id, episode_index, anchor_name, query),
                    flush=True,
                )
    finally:
        environment.close()

    if not rows:
        raise RuntimeError("no admitted rows; see the audit for drop reasons")

    payload = {
        "image": np.stack(arrays["image"]).astype(np.uint8),
        "wrist_image": np.stack(arrays["wrist_image"]).astype(np.uint8),
        "state": np.stack(arrays["state"]).astype(np.float32),
        "flow_noise": np.stack(arrays["flow_noise"]).astype(np.float32),
        "unit_id": np.asarray([row["unit_id"] for row in rows], np.int32),
        "condition": np.asarray([row["condition"] for row in rows]),
        "anchor": np.asarray([row["anchor"] for row in rows]),
        "episode_index": np.asarray([row["episode_index"] for row in rows], np.int32),
        "query_index": np.asarray([row["query_index"] for row in rows], np.int32),
        "target_xyz": np.stack([row["target_xyz"] for row in rows]).astype(np.float64),
        "eef_xyz": np.stack([row["eef_xyz"] for row in rows]).astype(np.float64),
        "historical_action": np.stack([row["historical_action"] for row in rows]).astype(np.float32),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    audit = {
        "schema": SCHEMA,
        "protocol": "moe-coupling-circuit-0906/PROTOCOL.md",
        "task_key": task_key,
        "task_name": str(task.name),
        "prompt": prompt,
        "suite": args.suite,
        "task_id": config.task_id,
        "target_joint": target_joint,
        "target_state_slice": [lo, hi],
        "target_qvel_slice": list(velocity),
        "client_dir": str(client.resolve()),
        "sim_layout_sha256": sha256_file(client / "sim_layout.json"),
        "summaries_sha256": sha256_file(client / "summaries.json"),
        "units": unit_id,
        "rows": len(rows),
        "conditions": sorted({row["condition"] for row in rows}),
        "units_by_anchor": {
            anchor: len({row["unit_id"] for row in rows if row["anchor"] == anchor})
            for anchor in sorted({row["anchor"] for row in rows})
        },
        "episodes_considered": len(summaries),
        "max_units_per_anchor": int(args.max_units_per_anchor),
        "dropped": len(drops),
        "drop_reasons": {
            reason: sum(1 for drop in drops if drop["reason"] == reason)
            for reason in sorted({drop["reason"] for drop in drops})
        },
        "gates": {
            "position_m": POSITION_TOLERANCE_M,
            "rotation_rad": ROTATION_TOLERANCE_RAD,
            "gripper": GRIPPER_TOLERANCE,
            "sim_state": SIM_STATE_TOLERANCE,
            "support_height_m": SUPPORT_HEIGHT_TOLERANCE_M,
            "visibility_pixel_delta": VISIBILITY_PIXEL_DELTA,
            "visibility_pixel_fraction": VISIBILITY_PIXEL_FRACTION,
        },
        "shifts_m": list(SHIFTS_M),
        "dataset": str(args.out.resolve()),
        "dataset_sha256": sha256_file(args.out),
        "dataset_file_bytes": args.out.stat().st_size,
        "shapes": {name: list(np.shape(value)) for name, value in payload.items()},
        # The per-row arrays live in the npz; repeating them here would double
        # the audit's size without making it more auditable.
        "row_metadata": [
            {key: value for key, value in row.items() if key != "historical_action"}
            for row in rows
        ],
        "drops": drops,
        "limitations": [
            "RGB is regenerated from stored MuJoCo state, not from stored frames.",
            "Edited conditions are physically plausible restorations, not states the "
            "simulator reached by stepping; contact history and settling are absent.",
            "No policy is run here; nothing about the action response is established.",
        ],
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    audit = jsonable(audit)
    args.audit.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: audit[key]
                for key in ("units", "rows", "units_by_anchor", "dropped", "drop_reasons")
            },
            indent=2,
            sort_keys=True,
        )
    )
    return audit


def build_unit(
    *,
    scene: Scene,
    prompt: str,
    anchor: str,
    query: int,
    state: np.ndarray,
    sim: np.ndarray,
    actions: np.ndarray,
    lo: int,
    velocity: tuple[int, int],
    summary: dict[str, Any],
    episode_path: Path,
) -> dict[str, Any]:
    """Render every condition for one (episode, anchor); admit all or none."""
    base = np.asarray(sim[query], np.float64)
    saved_state = np.asarray(state[query], np.float32)
    resting_pose = np.asarray(sim[0, lo : lo + 7], np.float64)
    resting_height = float(sim[0, lo + 2])

    # ``moves_object`` drives the physical-plausibility gates.  ``light`` is a
    # rendering change on the unedited state, so at q_post it inherits held's
    # gripper-object contacts; gating it as though it had moved the object
    # rejected every post anchor in the first run.
    plan: list[dict[str, Any]] = [
        {"condition": "held", "sim": base, "light": False, "moves_object": False}
    ]
    if anchor == "pre":
        direction = bearing(state[query], base[lo : lo + 3])
        if direction is None:
            return {"ok": False, "reason": "target bearing is undefined"}
        perpendicular = np.asarray([-direction[1], direction[0], 0.0], np.float64)
        signs = (1.0, -1.0)
        chosen_sign = None
        for sign in signs:
            candidate_ok = True
            for shift in SHIFTS_M:
                candidate = edited_state(
                    base, lo, velocity, translation=sign * shift * perpendicular
                )
                report = scene.probe(candidate)
                if report["robot_contacts"] or report["support_contacts"] == 0:
                    candidate_ok = False
                    break
            if candidate_ok:
                chosen_sign = sign
                break
        if chosen_sign is None:
            return {"ok": False, "reason": "no shift sign clears contact and support"}
        for shift in SHIFTS_M:
            plan.append(
                {
                    "condition": "shift%d" % int(round(shift * 100)),
                    "sim": edited_state(
                        base, lo, velocity, translation=chosen_sign * shift * perpendicular
                    ),
                    "light": False,
                    "moves_object": True,
                }
            )
    else:
        chosen_sign = 0.0
        plan.append(
            {
                "condition": "uncoupled",
                "sim": edited_state(base, lo, velocity, pose=resting_pose),
                "light": False,
                "moves_object": True,
            }
        )
    plan.append({"condition": "light", "sim": base, "light": True, "moves_object": False})

    noise = flow_noise_at(int(summary["flow_noise_seed"]), query)
    historical = np.asarray(actions[query], np.float32)
    rendered_by_condition: dict[str, dict[str, Any]] = {}
    for item in plan:
        rendered = render_condition(scene, prompt, item["sim"], light=bool(item["light"]))
        if rendered["blank"]:
            return {"ok": False, "reason": "rendered image is blank", "detail_condition": item["condition"]}
        if rendered["sim_error"] > SIM_STATE_TOLERANCE:
            return {
                "ok": False,
                "reason": "sim state not restored exactly",
                "detail_condition": item["condition"],
                "detail_sim_error": rendered["sim_error"],
            }
        report = rendered["contacts"]
        # Any robot contact fails an edited condition. MuJoCo reports a contact
        # only inside the collision margin, so "no contact" already means the
        # target is clear of the robot; grading by penetration depth would admit
        # exactly the states that are most obviously wrong.
        if item["moves_object"] and report["robot_contacts"]:
            return {
                "ok": False,
                "reason": "edited target touches the robot",
                "detail_condition": item["condition"],
                "detail_robot_min_dist": report["robot_min_dist"],
            }
        if item["moves_object"]:
            if report["support_contacts"] == 0:
                return {
                    "ok": False,
                    "reason": "edited target is unsupported",
                    "detail_condition": item["condition"],
                }
            height = float(item["sim"][lo + 2])
            if abs(height - resting_height) > SUPPORT_HEIGHT_TOLERANCE_M:
                return {
                    "ok": False,
                    "reason": "edited target height differs from its resting height",
                    "detail_condition": item["condition"],
                    "detail_height_error_m": abs(height - resting_height),
                }
        rendered["condition"] = item["condition"]
        rendered["sim"] = item["sim"]
        rendered_by_condition[item["condition"]] = rendered

    held = rendered_by_condition["held"]
    position_error = float(np.abs(held["state"][:3] - saved_state[:3]).max())
    rotation_error = float(np.abs(held["state"][3:6] - saved_state[3:6]).max())
    gripper_error = float(np.abs(held["state"][6:] - saved_state[6:]).max())
    if position_error > POSITION_TOLERANCE_M:
        return {"ok": False, "reason": "position restore outside tolerance", "detail_error_m": position_error}
    if rotation_error > ROTATION_TOLERANCE_RAD:
        return {"ok": False, "reason": "rotation restore outside tolerance", "detail_error_rad": rotation_error}
    if gripper_error > GRIPPER_TOLERANCE:
        return {"ok": False, "reason": "gripper restore outside tolerance", "detail_error": gripper_error}

    conditions: list[dict[str, Any]] = []
    for name, rendered in rendered_by_condition.items():
        seen = visibility(held, rendered) if name != "held" else {"passed": True}
        if name != "held" and not seen["passed"]:
            return {
                "ok": False,
                "reason": "edit is not visible in either camera",
                "detail_condition": name,
                "detail_visibility": {k: v for k, v in seen.items() if k != "passed"},
            }
        conditions.append(
            {
                "image": rendered["image"],
                "wrist_image": rendered["wrist_image"],
                "state": rendered["state"],
                "flow_noise": noise,
                "condition": name,
                "anchor": anchor,
                "episode_index": int(summary["episode_index"]),
                "query_index": query,
                "flow_noise_seed": int(summary["flow_noise_seed"]),
                "init_state_id": int(summary["init_state_id"]),
                "target_xyz": np.asarray(rendered["sim"][lo : lo + 3], np.float64),
                "eef_xyz": np.asarray(held["state"][:3], np.float64),
                "historical_action": historical,
                "shift_sign": float(chosen_sign),
                "image_sha256": sha256_bytes(rendered["image"]),
                "wrist_image_sha256": sha256_bytes(rendered["wrist_image"]),
                "episode_npz_sha256": sha256_file(episode_path),
                "position_restore_error_m": position_error,
                "rotation_restore_error_rad": rotation_error,
                "gripper_restore_error": gripper_error,
                "sim_restore_error": rendered["sim_error"],
                "contacts": rendered["contacts"],
                "visibility": {k: v for k, v in seen.items() if k != "passed"},
            }
        )
    return {"ok": True, "conditions": conditions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-dir", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument(
        "--target-audit",
        type=Path,
        default=ROOT / "analysis_trap_taxonomy" / "results" / "candidate_target_audit.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--max-units-per-anchor", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
