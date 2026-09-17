"""Frozen online crossover with actual-route feedback and output-only guards."""

import numpy as np

from gated_rollout_protocol import proposal as original_proposal
from v8_feature_control import EFFECTIVE_PROBS

ARMS = ('native', 'joint', 'route_only', 'output_only')
CHUNK, HORIZON, WINDOW = 10, 520, 12


def active(arm, query, start):
    if arm not in ARMS or min(query, start) < 0:
        raise ValueError('Invalid crossover arm/query')
    return arm != 'native' and start <= query < start + WINDOW


def proposal(native, previous, parent, query, arm):
    if not active(arm, query, parent['query']):
        raise ValueError('No proposal outside fixed active window')
    return original_proposal(native[EFFECTIVE_PROBS].astype(np.float16), previous,
                             parent, query, 'mobility_balance')


def guard(native, joint, std):
    std = np.asarray(std, np.float64)
    if std.shape != (6,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid action normalization')
    a, b = np.asarray(joint['actions']), np.asarray(native['actions'])
    if a.shape != (10, 7) or b.shape != (10, 7) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid policy actions')
    rms = float(np.sqrt(np.mean(((a[:, :6].astype(float) - b[:, :6]) / std) ** 2)))
    gripper = int(np.count_nonzero((a[:, 6] >= 0) != (b[:, 6] >= 0)))
    reasons = (['action_rms'] if rms > .05 else []) + (['gripper_change'] if gripper else [])
    return dict(accepted=not reasons, reasons=reasons, normalized_action_rms=rms,
                gripper_sign_changes=gripper, source='joint_donor_actions', route_scores_used=False)


def committed_route(response):
    value = np.asarray(response[EFFECTIVE_PROBS])
    if value.shape != (8, 10, 11, 32) or not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError('Invalid selected routing history')
    return value.astype(np.float16)


def maximum_calls(parents):
    return sum(4 * (HORIZON // CHUNK - p['query']) + 5 * WINDOW + 8 for p in parents)
