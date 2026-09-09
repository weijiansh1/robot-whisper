"""Predeclared output-feedback recovery operators for original LIBERO Long."""

import hashlib
import json

import numpy as np

from mode_control import active_mask, mode_name, noise_for, thresholds_at

PROTOCOL = "moe_control.native_control_bank.v1"
FAMILIES = ("resample", "replan2", "damp2", "boost2", "smooth2", "withdraw4", "lift8",
            "retrace", "side_plus", "side_minus", "hold16", "grip_cycle", "open_lift8",
            "moe_switch", "random_switch")
ARMS = ("native",)+tuple(name+"_r"+str(repeat) for name in FAMILIES for repeat in range(2))
PHYSICAL = ("withdraw4", "lift8", "retrace", "side_plus", "side_minus", "hold16", "grip_cycle", "open_lift8")
TRANSFORM = ("replan2", "damp2", "boost2", "smooth2")
SWITCH_LIBRARY = ("resample", "withdraw4", "smooth2", "replan2", "side_plus", "side_minus")
SETTINGS = dict(trigger="v82_frozen", horizon=520, chunk=10, repeats=2,
    transform_queries=8, transform_chunk=2, damp_gain=.5, boost_gain=2., smoothing_alpha=.25,
    maximum_translation_command_m=.01, lift_m=.08, side_m=.04, history_queries=3,
    maximum_history_target_distance_m=.08, physical_steps_max=24,
    policy_noise_protocol="moe_control.native_long_modes.v1", training=False, hidden_capture=False,
    threshold_fitting=False, object_pose_access=False, privileged_goal_input=False,
    probe="one original-noise query at the original fork, zero executed steps and no RNG/history commit",
    switch="once at entry: exactly F -> withdraw4; otherwise A -> smooth2; P -> alternating side; I/C -> replan2; none -> resample",
    random_switch="uniform over the same six possible switch outputs, independent of policy noise",
    stopping="success or original total 520 steps; physical phases and eight transformed queries are bounded",
    evaluation="all first alarms including original successes; each repeat separately; old and fresh cohorts reported separately",
    stability_scope="nominal Cartesian integrator only; MoE scores are observations, not certified Lyapunov or barrier functions")


def decode_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown control-bank arm")
    return ("native", -1) if arm == "native" else (arm[:-3], int(arm[-1]))


def choose_operator(family, entry_scores, thresholds, main_id, repeat):
    if family not in FAMILIES+ ("native",):
        raise ValueError("Unknown controller")
    if family == "random_switch":
        key = json.dumps([PROTOCOL, main_id, repeat, "operator"], separators=(",", ":"))
        seed = int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:16], "little")
        return SWITCH_LIBRARY[int(np.random.default_rng(seed).integers(len(SWITCH_LIBRARY)))]
    if family != "moe_switch":
        return family
    if not np.isfinite(entry_scores).all():
        return "resample"
    mask = active_mask(entry_scores, thresholds)
    if mask.tolist() == [True, False, False, False, False]:
        return "withdraw4"
    if mask[1]:
        return "smooth2"
    if mask[2]:
        return "side_plus" if repeat == 0 else "side_minus"
    if mask[3:].any():
        return "replan2"
    return "resample"


def bounded_vector(value, radius):
    value = np.asarray(value, float)
    return value*min(1., radius/max(float(np.linalg.norm(value)), 1e-12))


def waypoints(operator, initial, history, gripper):
    """Only observed end-effector positions and the last gripper command are inputs."""
    initial, history = np.asarray(initial, float), np.asarray(history, float)
    if operator not in PHYSICAL or initial.shape != (3,) or history.shape != (3, 3):
        raise ValueError("Three causal history positions required")
    if not np.isfinite(initial).all() or not np.isfinite(history).all():
        raise ValueError("Finite Cartesian observations required")
    sign = 1. if gripper > 0 else -1.
    if operator == "withdraw4":
        lift = initial+np.array([0., 0., .04])
        xy = bounded_vector(history[-1, :2]-initial[:2], .04)
        return np.stack((lift, lift+np.r_[xy, 0.])), [8, 8], [sign, sign]
    if operator == "lift8":
        return (initial+np.array([0., 0., .08]))[None], [24], [sign]
    if operator == "retrace":
        result = np.stack([initial+bounded_vector(p-initial, .08) for p in history])
        return result, [8, 8, 8], [sign]*3
    if operator in ("side_plus", "side_minus"):
        direction = initial[:2]-history[-1, :2]
        perpendicular = np.array([-direction[1], direction[0]])
        if np.linalg.norm(perpendicular) < 1e-5:
            perpendicular = np.array([1., 0.])
        perpendicular /= np.linalg.norm(perpendicular)
        lift = initial+np.array([0., 0., .04])
        detour = lift+np.r_[perpendicular*(.04 if operator == "side_plus" else -.04), 0.]
        return np.stack((lift, detour, lift)), [8, 8, 8], [sign]*3
    if operator == "open_lift8":
        return np.stack((initial, initial+np.array([0., 0., .08]))), [8, 16], [-1., -1.]
    return np.stack((initial, initial)), [8, 8], ([-sign, sign] if operator == "grip_cycle" else [sign, sign])


def physical_action(operator, position, target, gripper, output_scale):
    action = np.zeros(7, float)
    if operator not in ("hold16", "grip_cycle"):
        delta = bounded_vector(np.asarray(target)-np.asarray(position), SETTINGS["maximum_translation_command_m"])
        action[:3] = delta/np.asarray(output_scale, float)
    action[6] = gripper
    if not np.isfinite(action).all() or np.abs(action).max() > 1.000001:
        raise ValueError("Recovery command exceeds controller bounds")
    return action


class ActionFilter:
    """Update only for commands that are actually dispatched to the simulator."""

    def __init__(self, operator, previous_action):
        self.operator = operator
        self.previous = np.asarray(previous_action, float)[:6].copy()

    def chunk_size(self, relative_query):
        return 2 if self.operator in TRANSFORM and relative_query < 8 else 10

    def command(self, raw, relative_query):
        output = np.asarray(raw).copy()
        if relative_query < 8:
            if self.operator in ("damp2", "boost2"):
                gain = .5 if self.operator == "damp2" else 2.
                output[:6] = np.clip(output[:6]*gain, -1., 1.)
            elif self.operator == "smooth2":
                output[:6] = np.clip(.25*output[:6]+.75*self.previous, -1., 1.)
        self.previous = output[:6].astype(float).copy()
        return output
