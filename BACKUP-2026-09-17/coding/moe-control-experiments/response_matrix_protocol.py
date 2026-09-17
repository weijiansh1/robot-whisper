"""Frozen equal-energy interventions and five-component response measurements."""

import copy

import numpy as np

from head_control_protocol import normalize_scores
from v8_feature_control import make_bias
from v8_closed_loop import score_status
from intrinsic_guard_monitor import intrinsic_score_arrays

SHAPE = (8, 10, 11, 32)
COMPONENTS = ('freeze', 'acceleration', 'periodicity', 'inversion', 'curvature')
OPERATORS = ('mobility', 'balance', 'curvature', 'mobility_balance', 'mobility_curvature')
TARGETS = dict(mobility=(0,), balance=(3,), curvature=(4,),
               mobility_balance=(0, 3), mobility_curvature=(0, 4))
MAX_ABS = .30
HIGH_RMS = .10
SEED = 2026091503
SCREEN_CHANGE = .05
SCREEN_ACTION_RMS = .05
MASK = np.zeros(SHAPE, bool)
MASK[4:, :, 1:] = True


def unit(value):
    value = np.asarray(value, np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != SHAPE or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError('Degenerate or invalid intervention direction')
    if np.any(value[~MASK] != 0):
        raise ValueError('Intervention escaped the frozen scope')
    return value / norm


def directions(shadow, previous):
    result = {name: unit(make_bias(shadow, previous, name, 1., 0)) for name in OPERATORS[:3]}
    result['mobility_balance'] = unit(result['mobility'] + result['balance'])
    result['mobility_curvature'] = unit(result['mobility'] + result['curvature'])
    return result


def specifications():
    result = []
    for operator in OPERATORS:
        for dose, fraction in (('low', .5), ('high', 1.)):
            for sign in (-1, 1):
                label = '%s-%s-%s' % (operator, dose, 'plus' if sign > 0 else 'minus')
                result.append(dict(label=label, operator=operator, dose=dose, fraction=fraction,
                                   sign=sign, randomized=False))
    for operator in OPERATORS:
        for dose, fraction in (('low', .5), ('high', 1.)):
            result.append(dict(label='random-%s-%s' % (operator, dose), operator=operator,
                               dose=dose, fraction=fraction, sign=1, randomized=True))
    return result


def random_permutation(parent, operator):
    identity = [SEED, {'plus': 0, 'pro': 1}[parent['benchmark']], parent['base_task_id'],
                parent['init_state_id'], parent['query'], OPERATORS.index(operator)]
    rng = np.random.default_rng(np.random.SeedSequence(identity))
    return np.argsort(rng.random(SHAPE), axis=-1)


def bias_bank(shadow, previous, parent):
    vectors = directions(shadow, previous)
    peak = max(float(np.max(np.abs(value))) for value in vectors.values())
    nominal = HIGH_RMS * np.sqrt(MASK.sum())
    energy = min(float(nominal), MAX_ABS / peak)
    bank = []
    for spec in specifications():
        direction = vectors[spec['operator']]
        if spec['randomized']:
            direction = np.take_along_axis(direction, random_permutation(parent, spec['operator']), axis=-1)
        value = (direction * energy * spec['fraction'] * spec['sign']).astype(np.float32)
        if np.max(np.abs(value)) > MAX_ABS + 1e-7 or np.any(value[~MASK] != 0):
            raise RuntimeError('Bias bank violates frozen bounds')
        bank.append(value)
    return np.stack(bank), dict(high_energy=energy, nominal_high_energy=float(nominal),
                                high_scope_rms=energy / np.sqrt(MASK.sum()),
                                cap_derated=energy < nominal, scope_scalar_count=int(MASK.sum()))


def measure(prefix, probability):
    probability = np.asarray(probability)
    if probability.shape != SHAPE or not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError('Invalid complete effective probabilities')
    monitor = copy.deepcopy(prefix)
    status = monitor.update(probability)
    scores = score_status(status)
    normalized, thresholds, margins = normalize_scores(scores, status['query'])
    v7 = monitor.v7
    streams = intrinsic_score_arrays(np.asarray(v7._mobility_history, np.float32)[None],
        np.asarray(v7._acceleration_history, np.float32)[None],
        np.asarray(v7._periodicity_history, np.float32)[None], v7.profile.periodicity_scale)
    raw = np.asarray(monitor.raw, np.float32)
    relative_curvature = np.log(np.maximum(raw[-1, 1], 1e-12) /
                                np.maximum(raw[1:5, 1].mean(dtype=np.float32), 1e-12))
    instantaneous = np.array([streams[name][0, -1] for name in
                              ('freeze_raw', 'acceleration_raw', 'periodicity_raw')] +
                             [raw[-1, 0], relative_curvature], np.float64)
    if not np.isfinite(instantaneous).all():
        raise RuntimeError('Incomplete instantaneous component history')
    return dict(status=status, scores=scores, normalized=normalized, thresholds=thresholds,
                margins=margins, instantaneous=instantaneous)


def screen(delta, operator, action_rms, gripper_changes):
    target = np.array(TARGETS[operator], int)
    other = np.array([i for i in range(5) if i not in target], int)
    return bool(np.all(np.asarray(delta)[target] <= -SCREEN_CHANGE) and
                np.all(np.asarray(delta)[other] <= SCREEN_CHANGE) and
                action_rms <= SCREEN_ACTION_RMS and gripper_changes == 0)
