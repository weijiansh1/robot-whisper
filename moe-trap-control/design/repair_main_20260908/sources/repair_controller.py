"""Train-free repair controllers that act only through env.step; never call the model.

Control-theoretic reading: the VLA is a memoryless output-feedback controller and a trap
is a spurious attractor of the closed loop.  REPAIR is a supervisory mode that drives the
robot to a target configuration x* with a saturated proportional task-space law, then hands
control back to the VLA.  Everything here is a pure function of observations, so it runs
under the LIBERO Python 3.8 environment and under the repository's CPU test environment.
"""

from __future__ import annotations

import numpy as np

OPEN, CLOSE = -1.0, 1.0
CLOSE_APERTURE = 0.05          # analysis_trap_taxonomy closure threshold on the finger aperture
DEPARTURE_M = 0.10             # phantom grasp: end effector left the closure pose
STILL_M = 0.01                 # phantom grasp: target barely moved since closure
NEAR_M = 0.16                  # closure counts only when the target is this close
ABOVE_M = 0.10                 # retract target sits this far above the object
LIFT_M = 0.08                  # scripted regrasp lift
TABLE_CLEARANCE_M = 0.03       # object centre below the table by this much counts as dropped
TILT_DEG = 45.0
FAR_M = 0.60                   # object farther than this from the end effector is out of reach


def aperture(obs):
    q = np.asarray(obs["robot0_gripper_qpos"], np.float64)
    return float(q[0] - q[1])


def eef_position(obs):
    return np.asarray(obs["robot0_eef_pos"], np.float64).copy()


def retract_chunk(eef, target, gain, unit_metres, steps=10, gripper=OPEN):
    """One 10x7 LIBERO action chunk: saturated proportional pull toward target, no rotation, gripper fixed."""
    error = np.asarray(target, np.float64) - np.asarray(eef, np.float64)
    command = np.clip(gain * error / unit_metres, -1.0, 1.0)
    chunk = np.zeros((steps, 7), np.float32)
    chunk[:, :3] = command.astype(np.float32)
    chunk[:, 6] = gripper
    return chunk


class RepairPhase:
    """Guard g2: reached after settle_steps consecutive env steps within tolerance; exhausted after max_chunks."""

    def __init__(self, target, tolerance, settle_steps, max_chunks):
        self.target = np.asarray(target, np.float64)
        self.tolerance, self.settle_steps, self.max_chunks = float(tolerance), int(settle_steps), int(max_chunks)
        self.chunks, self.streak, self.reason, self.done = 0, 0, None, False
        self.history = []

    def begin_chunk(self):
        self.chunks += 1

    @property
    def exhausted(self):
        return self.chunks >= self.max_chunks and not self.done

    def observe(self, eef):
        distance = float(np.linalg.norm(np.asarray(eef, np.float64) - self.target))
        self.history.append(distance)
        self.streak = self.streak + 1 if distance < self.tolerance else 0
        if self.streak >= self.settle_steps:
            self.done, self.reason = True, "reached"
        elif self.exhausted:
            self.reason = "max_chunks"
        return self.done


class ProportionalRetract:
    def __init__(self, target, gain, unit_metres, tolerance, settle_steps, max_chunks, gripper=OPEN):
        self.gain, self.unit_metres, self.gripper = float(gain), float(unit_metres), gripper
        self.phase = RepairPhase(target, tolerance, settle_steps, max_chunks)

    def next_chunk(self, eef):
        self.phase.begin_chunk()
        return retract_chunk(eef, self.phase.target, self.gain, self.unit_metres, gripper=self.gripper)


def physical_class(position, initial, eef, table_z, tilt_deg, grasped):
    position, initial, eef = (np.asarray(v, np.float64) for v in (position, initial, eef))
    if grasped:
        return "in_hand"
    if position[2] < table_z - TABLE_CLEARANCE_M or tilt_deg > TILT_DEG or np.linalg.norm(position - eef) > FAR_M:
        return "dropped_or_tipped"
    return "untouched" if np.linalg.norm(position - initial) < 0.02 else "displaced_reachable"


def tilt_degrees(quaternion_wxyz):
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))


def _movable_names(inner):
    names = set()
    for group in inner.parsed_problem["objects"].values():
        names.update(group)
    return names


def resolve_target(env):
    """First unsatisfied goal predicate whose subject is a movable object; name is None if only fixtures remain."""
    inner = env.env
    movable = _movable_names(inner)
    unsatisfied = [list(state) for state in inner.parsed_problem["goal_state"] if not inner._eval_predicate(state)]
    for state in unsatisfied:
        candidates = [name for name in state[1:] if name in movable]
        if candidates:
            name = candidates[0]
            body = inner.obj_body_id[name]
            return dict(name=name, position=np.asarray(inner.sim.data.body_xpos[body], np.float64).copy(),
                        quaternion=np.asarray(inner.sim.data.body_xquat[body], np.float64).copy(),
                        unsatisfied=unsatisfied, articulated_pending=False)
    return dict(name=None, position=None, quaternion=None, unsatisfied=unsatisfied,
                articulated_pending=bool(unsatisfied))


def object_top_offset(env, name, default=0.02):
    """Half-height of the object's largest geom above its body origin, for the scripted grasp descent."""
    inner = env.env
    model = inner.sim.model
    body = inner.obj_body_id[name]
    sizes = [float(model.geom_size[g][2]) for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body]
    return max(sizes) if sizes else float(default)
