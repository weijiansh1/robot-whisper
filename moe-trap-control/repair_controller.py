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

    def observe(self, eef, target=None):
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


# ---------------------------------------------------------------------------- supervisor v2 (physical guards)
class HandState:
    """Is the target in the gripper?  closed aperture, lifted above its rest height, near the end effector,
    and moving with it (object displacement tracks end-effector displacement over the last window)."""

    def __init__(self, rest_z, lifted_m, near_m=0.20, window_steps=10, move_m=0.02, slip_m=0.015):
        self.rest_z, self.lifted_m, self.near_m = float(rest_z), float(lifted_m), float(near_m)
        self.window, self.move_m, self.slip_m = int(window_steps), float(move_m), float(slip_m)
        self.history, self.tracking = [], True

    def update(self, eef, ap, target):
        eef, target = np.asarray(eef, np.float64), np.asarray(target, np.float64)
        self.history.append((eef.copy(), target.copy()))
        if len(self.history) > self.window:
            past_eef, past_target = self.history[-self.window - 1]
            moved = np.linalg.norm(eef - past_eef)
            if moved >= self.move_m:
                self.tracking = bool(np.linalg.norm((eef - past_eef) - (target - past_target)) < self.slip_m)
            del self.history[:-self.window - 1]
        lifted = target[2] - self.rest_z >= self.lifted_m
        near = np.linalg.norm(eef - target) < self.near_m
        return bool(ap < CLOSE_APERTURE and lifted and near and self.tracking)


class PhantomGuard:
    """G2: the gripper closed on nothing.

    Fires when the aperture is closed, the target is not in hand, the target has not moved since
    the closure, and either the end effector departed >= departure_m from the closure pose or the
    closure has persisted >= hold_steps env steps.  Reset when the gripper opens or after an
    intervention.  Pure: feed it one observation per env step.
    """

    def __init__(self, departure_m, hold_steps, moved_m):
        self.departure_m, self.hold_steps, self.moved_m = float(departure_m), int(hold_steps), float(moved_m)
        self.reset()

    def reset(self):
        self.closure = None

    def observe(self, eef, ap, target, step, in_hand):
        eef, target = np.asarray(eef, np.float64), np.asarray(target, np.float64)
        if ap >= CLOSE_APERTURE:
            self.closure = None
            return None
        if self.closure is None:
            self.closure = dict(eef=eef.copy(), target=target.copy(), step=int(step))
            return None
        if in_hand or np.linalg.norm(target - self.closure["target"]) >= self.moved_m:
            return None
        if np.linalg.norm(eef - self.closure["eef"]) >= self.departure_m:
            return "phantom_departed"
        if int(step) - self.closure["step"] >= self.hold_steps:
            return "phantom_held"
        return None


class CarryGuard:
    """G3: the target has been in hand continuously for >= patience steps without a goal predicate turning true."""

    def __init__(self, patience_steps):
        self.patience = int(patience_steps)
        self.reset()

    def reset(self):
        self.lift = None

    def observe(self, unsatisfied, step, in_hand):
        if not in_hand:
            self.lift = None
            return None
        if self.lift is None or int(unsatisfied) < self.lift["unsatisfied"]:
            self.lift = dict(step=int(step), unsatisfied=int(unsatisfied))
            return None
        if int(step) - self.lift["step"] >= self.patience:
            return "carry_stalled"
        return None


class ContactPhase(RepairPhase):
    """Descend until the held object stops going down (contact), or max_chunks."""

    def __init__(self, target, settle_steps, max_chunks, drop_m=0.001):
        super().__init__(target, 0.0, settle_steps, max_chunks)
        self.drop_m, self.heights = float(drop_m), []

    def observe(self, eef, target=None):
        if target is None:
            return self.done
        z = float(np.asarray(target, np.float64)[2])
        self.heights.append(z)
        self.history.append(float(np.linalg.norm(np.asarray(eef, np.float64) - self.target)))
        if len(self.heights) > self.settle_steps and self.heights[-self.settle_steps - 1] - z < self.drop_m:
            self.done, self.reason = True, "contact"
        elif self.exhausted:
            self.reason = "max_chunks"
        return self.done


class ContactDescent:
    def __init__(self, target, gain, unit_metres, settle_steps, max_chunks):
        self.gain, self.unit_metres, self.gripper = float(gain), float(unit_metres), CLOSE
        self.phase = ContactPhase(target, settle_steps, max_chunks)

    def next_chunk(self, eef):
        self.phase.begin_chunk()
        return retract_chunk(eef, self.phase.target, self.gain, self.unit_metres, gripper=self.gripper)


def goal_region(env, target_name):
    """Centre, half-extents and kind of the region named by the first unsatisfied predicate on the target."""
    inner = env.env
    for state in inner.parsed_problem["goal_state"]:
        state = list(state)
        if len(state) != 3 or state[1] != target_name or inner._eval_predicate(state):
            continue
        region = state[2]
        if region in inner.object_sites_dict:
            site = inner.object_sites_dict[region]
            centre = np.asarray(inner.sim.data.get_site_xpos(region), np.float64).copy()
            mat = np.asarray(inner.sim.data.get_site_xmat(region), np.float64)
            size = np.asarray(site.size, np.float64)
            if size.size < 3:
                size = np.resize(size, 3)
            half = np.abs(mat @ size)
            return dict(name=region, predicate=state[0], centre=centre, half=half, kind="site")
        if region in inner.obj_body_id:
            centre = np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[region]], np.float64).copy()
            return dict(name=region, predicate=state[0], centre=centre, half=np.zeros(3), kind="body")
        return None
    return None


def place_point(region, others, fraction=0.5, margin_m=0.05):
    """Where to put the object inside the region: the centre, or the emptier half along the region's long axis."""
    centre, half = np.asarray(region["centre"], np.float64), np.asarray(region["half"], np.float64)
    inside = [np.asarray(o, np.float64) for o in others
              if np.all(np.abs(np.asarray(o, np.float64)[:2] - centre[:2]) <= half[:2] + margin_m)
              and abs(float(o[2]) - centre[2]) <= half[2] + margin_m]
    if not inside or float(half[:2].max()) <= 0.0:
        return centre.copy(), 0
    axis = int(np.argmax(half[:2]))
    candidates = []
    for sign in (-1.0, 1.0):
        point = centre.copy()
        point[axis] += sign * fraction * half[axis]
        candidates.append(point)
    best = max(candidates, key=lambda c: min(np.linalg.norm(c[:2] - o[:2]) for o in inside))
    return best, len(inside)


def place_waypoints(eef, target, region, half_height, height_m, clearance_m, rise_m, point=None):
    """End-effector waypoints for PLACE while holding the target: rise, transfer, lower, then retreat after release.

    The object hangs below the gripper by the constant offset d = eef - target; every waypoint is
    expressed for the end effector so that the object lands inside the region box.
    """
    eef, target = np.asarray(eef, np.float64), np.asarray(target, np.float64)
    offset = eef - target
    centre, half = np.asarray(region["centre"], np.float64), np.asarray(region["half"], np.float64)
    point = centre if point is None else np.asarray(point, np.float64)
    floor = centre[2] - half[2] if region["predicate"].lower() == "in" else centre[2]
    carry_z = max(centre[2] + half[2], floor) + height_m
    rise = np.array([eef[0], eef[1], carry_z + offset[2]])
    transfer = np.array([point[0], point[1], carry_z]) + offset
    lower = np.array([point[0], point[1], floor + half_height + clearance_m]) + offset
    retreat = lower + np.array([0.0, 0.0, rise_m])
    return dict(rise=rise, transfer=transfer, lower=lower, retreat=retreat, offset=offset, floor=floor)
