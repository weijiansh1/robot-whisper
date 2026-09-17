"""Frozen pairing and token scope for the MoE activation crossover."""

import numpy as np

LABELS = ('native', 'joint', 'native_replay', 'joint_replay', 'route_only',
          'output_only', 'repeat_route', 'repeat_output', 'post')
MAIN_LABELS = ('native', 'joint', 'route_only', 'output_only')
BIASED = frozenset(('joint', 'joint_replay', 'route_only', 'repeat_route'))
DONORS = dict(native_replay='native', joint_replay='joint', route_only='native',
              output_only='joint', repeat_route='native', repeat_output='joint')
CONTROLS = dict(native_replay='native', joint_replay='joint', repeat_route='route_only',
                repeat_output='output_only', post='native')
ACTIVATION_SHAPE = (8, 10, 11, 1024)
BIAS_SHAPE = (8, 10, 11, 32)
SITE_MASK = np.zeros(ACTIVATION_SHAPE[:3], bool)
SITE_MASK[4:, :, 1:] = True


def validate_donor(value):
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError('Donor must be an FP32 numpy array')
    if value.shape != ACTIVATION_SHAPE or not np.isfinite(value).all():
        raise ValueError('Invalid donor shape or values')
    return value


def validate_bias(value):
    if value.dtype != np.float32 or value.shape != BIAS_SHAPE or not np.isfinite(value).all():
        raise ValueError('Invalid crossover gate bias')
    if np.max(np.abs(value)) > .3000001 or np.any(value[~SITE_MASK] != 0):
        raise ValueError('Gate bias exceeds the frozen scope or bound')
    if not np.any(value):
        raise ValueError('The joint arm requires a nonzero bias')
    return value


def expected_computation(label):
    if label not in LABELS:
        raise ValueError('Unknown crossover label')
    return 'joint' if label in ('joint', 'joint_replay', 'output_only', 'repeat_output') else 'native'


def set_change(candidate, baseline):
    if candidate.shape != (8, 10, 11, 4) or candidate.shape != baseline.shape:
        raise ValueError('Invalid HB dispatch shape')
    changed = np.any(np.sort(candidate, axis=-1) != np.sort(baseline, axis=-1), axis=-1)
    return float(changed[SITE_MASK].mean())
