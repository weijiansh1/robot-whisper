"""Paired, equal-dose probes and scale-aware activation measurements."""

import numpy as np

from response_matrix_protocol import MASK, SHAPE, random_permutation

LABELS = ('native', 'zero', 'combo', 'opposite', 'random', 'repeat', 'post')
HB_FIELDS = ('input', 'shared', 'total', 'residual', 'block')


def selected_queries(parent, decisions):
    accepted = [int(row['query']) for row in decisions if row['decision']['accepted']]
    if len(accepted) < 2 or accepted != sorted(set(accepted)):
        raise ValueError('Need at least two ordered accepted queries')
    return (('before_alarm', parent['first_alarm'] - 5),
            ('first_accepted', accepted[0]), ('last_accepted', accepted[-1]))


def variants(bias, parent, query):
    bias = np.asarray(bias, np.float32)
    if (bias.shape != SHAPE or not np.isfinite(bias).all() or
            np.any(bias[~MASK]) or np.max(np.abs(bias)) > .3000001 or not np.any(bias)):
        raise ValueError('Invalid scoped nonzero bias')
    identity = dict(parent, query=int(query))
    randomized = np.take_along_axis(bias, random_permutation(identity, 'mobility_balance'), axis=-1)
    return dict(native=None, zero=np.zeros_like(bias), combo=bias.copy(), opposite=-bias,
                random=randomized, repeat=bias.copy(), post=None)


def comparison(candidate, native):
    a, b = np.asarray(candidate, np.float64), np.asarray(native, np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite aligned measurements required')
    delta = a - b
    baseline = float(np.linalg.norm(b))
    return dict(relative_l2=None if baseline == 0 else float(np.linalg.norm(delta) / baseline),
                delta_rms=float(np.sqrt(np.mean(delta ** 2))), native_rms=float(np.sqrt(np.mean(b ** 2))),
                max_abs_delta=float(np.max(np.abs(delta))))


def trace_effect(candidate, native):
    result = {}
    for field in HB_FIELDS + ('routed_derived',):
        if field == 'routed_derived':
            a = candidate['mechanism/total'].astype(np.float64) - candidate['mechanism/shared']
            b = native['mechanism/total'].astype(np.float64) - native['mechanism/shared']
        else:
            a, b = candidate['mechanism/' + field], native['mechanism/' + field]
        result[field] = [comparison(a[layer, :, 1:], b[layer, :, 1:]) for layer in range(8)]
    for field in ('projection_input', 'velocity'):
        result[field] = comparison(candidate['mechanism/' + field], native['mechanism/' + field])
    result['first_local_input_equal'] = bool(np.array_equal(candidate['mechanism/input'][4, 0], native['mechanism/input'][4, 0]))
    result['first_local_shared_equal'] = bool(np.array_equal(candidate['mechanism/shared'][4, 0], native['mechanism/shared'][4, 0]))
    result['first_local_routed'] = comparison(
        candidate['mechanism/total'][4, 0, 1:].astype(float) - candidate['mechanism/shared'][4, 0, 1:],
        native['mechanism/total'][4, 0, 1:].astype(float) - native['mechanism/shared'][4, 0, 1:])
    result['first_local_block'] = comparison(candidate['mechanism/block'][4, 0, 1:], native['mechanism/block'][4, 0, 1:])
    return result


def true_intervals(values):
    values = np.asarray(values, bool)
    if values.ndim != 1:
        raise ValueError('Expected one temporal Boolean stream')
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return [[int(a), int(b - 1)] for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
