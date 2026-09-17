"""Frozen proposal budget and causal, instantaneous acceptance decisions."""

import numpy as np

from response_matrix_protocol import (SHAPE, TARGETS, SCREEN_CHANGE, SCREEN_ACTION_RMS,
                                       bias_bank, measure, screen, specifications)
from v8_feature_control import EFFECTIVE_PROBS

ARMS = ('native', 'mobility', 'mobility_balance', 'random_balance')
RECOVERY_QUERIES = 12
CHUNK = 10
HORIZON = 520
LABELS = dict(mobility='mobility-high-plus', mobility_balance='mobility_balance-high-plus',
              random_balance='random-mobility_balance-high')


def in_window(arm, query, start):
    if arm not in ARMS or min(query, start) < 0:
        raise ValueError('Invalid arm or query')
    return arm != 'native' and start <= query < start + RECOVERY_QUERIES


def target_operator(arm):
    if arm not in LABELS:
        raise ValueError('Native has no candidate target')
    return 'mobility' if arm == 'mobility' else 'mobility_balance'


def proposal(shadow, previous, parent, query, arm):
    if not in_window(arm, query, parent['query']):
        raise ValueError('Proposal outside the fixed intervention window')
    for value in (shadow, previous):
        if np.shape(value) != SHAPE or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError('Invalid causal routing input')
    try:
        bank, energy = bias_bank(shadow, previous, dict(parent, query=query))
    except ValueError as error:
        if not str(error).startswith('Degenerate or invalid intervention direction'):
            raise
        return None, dict(fallback_reason='degenerate_direction')
    index = next(i for i, spec in enumerate(specifications()) if spec['label'] == LABELS[arm])
    return bank[index], dict(energy, fallback_reason=None, label=LABELS[arm], query=query)


def evaluate(prefix, native, candidate, arm, action_std):
    operator = target_operator(arm)
    std = np.asarray(action_std, np.float64)
    if std.shape != (6,) or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('Invalid action normalization')
    for response in (native, candidate):
        if np.shape(response['actions']) != (10, 7) or not np.isfinite(response['actions']).all():
            raise ValueError('Invalid candidate actions')
    baseline = measure(prefix, native[EFFECTIVE_PROBS].astype(np.float16))
    changed = measure(prefix, candidate[EFFECTIVE_PROBS].astype(np.float16))
    full_base = measure(prefix, native[EFFECTIVE_PROBS])
    full_changed = measure(prefix, candidate[EFFECTIVE_PROBS])
    delta = changed['normalized'] - baseline['normalized']
    instant = (changed['instantaneous'] - baseline['instantaneous']) / baseline['margins']
    full_delta = (full_changed['instantaneous'] - full_base['instantaneous']) / full_base['margins']
    rms = float(np.sqrt(np.mean(np.square(
        (candidate['actions'][:, :6].astype(float) - native['actions'][:, :6]) / std))))
    gripper = int(np.count_nonzero((candidate['actions'][:, 6] >= 0) != (native['actions'][:, 6] >= 0)))
    targets = TARGETS[operator]
    reasons = []
    for i in range(5):
        if i in targets and instant[i] > -SCREEN_CHANGE:
            reasons.append('target_%d_not_improved' % i)
        if i not in targets and instant[i] > SCREEN_CHANGE:
            reasons.append('side_effect_%d' % i)
    if rms > SCREEN_ACTION_RMS:
        reasons.append('action_rms')
    if gripper:
        reasons.append('gripper_change')
    accepted = screen(instant, operator, rms, gripper)
    if accepted != (not reasons):
        raise RuntimeError('Acceptance and rejection reasons disagree')
    return dict(accepted=accepted, reasons=reasons, operator=operator,
                native=baseline, candidate=changed, score_delta=delta, instantaneous_delta=instant,
                formal_screen=screen(delta, operator, rms, gripper), normalized_action_rms=rms,
                gripper_sign_changes=gripper, fp32_instantaneous_delta=full_delta,
                fp32_screen=screen(full_delta, operator, rms, gripper))
