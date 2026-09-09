"""Offline, simulator-based labels with an outcome-independent horizon risk set."""

from __future__ import annotations

import numpy as np


HORIZON = 5
LIFT_METERS = 0.025
REACH_METERS = 0.12
STAGE_NAMES = ["goal_fraction", "grasp_fraction", "lift_fraction", "reach_fraction",
               "unfinished_grasp_lift_fraction", "min_target_distance", "max_target_lift"]


def bitmask(values):
    values = np.asarray(values, dtype=bool)
    if values.ndim != 2 or values.shape[1] > 30:
        raise ValueError("expected at most 30 binary state components")
    return (values.astype(np.int64) * (1 << np.arange(values.shape[1]))).sum(axis=1)


def episode_labels(goal, grasp, position, distance, subject_goal, *, success,
                   action_steps, max_steps, replan_steps=10, horizon=HORIZON):
    """Return physical labels for all q; only `eligible` rows support the target.

    Endpoint metadata constructs supervision, never the predictor input. Source
    checkpoints are before actions; success may occur between checkpoints.
    """
    goal, grasp = np.asarray(goal, bool), np.asarray(grasp, bool)
    position, distance = np.asarray(position, float), np.asarray(distance, float)
    subject_goal = np.asarray(subject_goal, bool)
    if goal.ndim != 2 or goal.shape[1] < 1 or horizon < 2 or replan_steps != 10:
        raise ValueError("unsupported goal or fixed-horizon protocol")
    length, subjects = len(goal), grasp.shape[1]
    if (grasp.shape != (length, subjects) or position.shape != (length, subjects, 3)
            or distance.shape != (length, subjects) or subject_goal.shape != (subjects, goal.shape[1])):
        raise ValueError("physical arrays do not align")
    if not np.isfinite(position).all() or not np.isfinite(distance).all():
        raise ValueError("nonfinite physical checkpoint")
    if length != (action_steps + replan_steps - 1) // replan_steps or not 0 < action_steps <= max_steps:
        raise ValueError("invalid endpoint timing")
    if not success and action_steps != max_steps:
        raise ValueError("incomplete natural rollout must not receive negative progress labels")
    query = np.arange(length)
    already_complete = goal.all(axis=1)
    eligible = (query >= 7) & ((query + horizon) * replan_steps < max_steps) & ~already_complete
    lifted = position[:, :, 2] - position[:1, :, 2] >= LIFT_METERS
    reached = distance <= REACH_METERS
    held_lift = grasp & lifted
    unfinished = (np.logical_not(goal)[:, None, :] & subject_goal[None]).any(axis=2)
    counts = goal.sum(axis=1)
    success_h = bool(success) & (action_steps <= (query + horizon) * replan_steps)
    goal_event = np.zeros(length, dtype=bool)
    grasp_event = np.zeros(length, dtype=bool)
    first_event = np.full(length, np.nan)
    for q in range(length):
        for j in range(2, min(horizon, length - 1 - q) + 1):
            future = slice(q+j-1, q+j+1)
            goal_gain = bool((counts[future] > counts[q]).all())
            grasp_gain = bool((held_lift[future].all(axis=0) & ~held_lift[q] & unfinished[q]).any()
                              and (counts[future] >= counts[q]).all())
            goal_event[q] |= goal_gain
            grasp_event[q] |= grasp_gain
            if (goal_gain or grasp_gain) and np.isnan(first_event[q]):
                first_event[q] = j
        if success_h[q]:
            delta = (action_steps - q * replan_steps) / replan_steps
            first_event[q] = min(delta, first_event[q]) if np.isfinite(first_event[q]) else delta
        if eligible[q] and not success_h[q] and q + horizon >= length:
            raise ValueError("future observation is censored despite schedule eligibility")
    stage = [f"{g}:{h}:{lift}:{r}" for g, h, lift, r in
             zip(bitmask(goal), bitmask(grasp), bitmask(lifted), bitmask(reached), strict=True)]
    physical = np.zeros((length, len(STAGE_NAMES)), dtype=np.float32)
    physical[:, 0] = counts / goal.shape[1]
    if subjects:
        physical[:, 1] = grasp.mean(axis=1)
        physical[:, 2] = lifted.mean(axis=1)
        physical[:, 3] = reached.mean(axis=1)
        physical[:, 4] = (held_lift & unfinished).mean(axis=1)
        physical[:, 5] = distance.min(axis=1)
        physical[:, 6] = (position[:, :, 2] - position[:1, :, 2]).max(axis=1)
    else:
        physical[:, 5] = 1.0
    return dict(query=query, eligible=eligible, already_complete=already_complete,
                progress=success_h | goal_event | grasp_event,
                success_h5=success_h, goal_advance_h5=goal_event, grasp_lift_h5=grasp_event,
                first_event_chunks=first_event, stage=stage, physical=physical)
