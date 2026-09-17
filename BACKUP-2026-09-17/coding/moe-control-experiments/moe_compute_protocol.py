"""Frozen MoE-only distances, causal summaries, and grouped prediction."""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from topology_protocol import (REPRESENTATIONS, SCALES, delayed_distance,
                               fit_reference, persistence, validate_distance)

SEED = 2026091523
KINDS = ('route', 'input', 'total', 'input_total')
WINDOW = 14
FUTURE = 4
RIDGE = 1.
SIMPLE_NAMES = tuple('lag%d' % lag for lag in (1, 2, 4, 8, 13)) + (
    'step_mean', 'step_std', 'step_max', 'net_path', 'nonadjacent_nearest', 'diameter', 'relative_median')
TOPOLOGY_NAMES = ('h1_lifetime', 'h1_relative', 'h0_largest') + tuple(
    '%s_%s' % (scale, field) for scale in SCALES for field in ('scc_fraction', 'net_path', 'eligible'))


def select_scope(values, scope):
    a = np.asarray(values)
    if a.ndim != 5 or a.shape[1:4] != (8, 10, 11) or scope not in REPRESENTATIONS:
        raise ValueError('Expected query/layer/flow/token/channel')
    return a[:, :, :, 1:] if scope == 'full_path' else a[:, 4:, -1:, 1:] if scope == 'back_last' else a[:, 4:, :, 1:]


def port_distance(values, scope, prefix_count):
    a = select_scope(values, scope).astype(np.float64)
    if not 1 <= prefix_count <= len(a) or not np.isfinite(a).all():
        raise ValueError('Finite values and nonempty causal prefix required')
    scales = np.sqrt(np.mean(a[:prefix_count] ** 2, axis=(0, 2, 3, 4)))
    scales = np.maximum(scales, 1e-12)
    squared, raw_squared = np.zeros((len(a), len(a))), np.zeros((len(a), len(a)))
    for layer in range(a.shape[1]):
        x = a[:, layer].reshape(len(a), -1)
        d2 = squareform(pdist(x, metric='sqeuclidean')) / x.shape[1]
        raw_squared += d2 / a.shape[1]
        squared += d2 / (a.shape[1] * scales[layer] ** 2)
    return np.sqrt(squared), np.sqrt(raw_squared), scales


def combine_distances(first, second):
    first, second = validate_distance(first), validate_distance(second)
    if first.shape != second.shape:
        raise ValueError('Aligned distance matrices required')
    return np.sqrt((first ** 2 + second ** 2) / 2)


def simple_features(raw):
    raw = validate_distance(raw)
    if raw.shape != (WINDOW, WINDOW):
        raise ValueError('Expected fourteen past and current queries')
    steps, diameter = np.diag(raw, 1), float(raw.max())
    separated = np.abs(np.arange(WINDOW)[:, None] - np.arange(WINDOW)[None, :]) >= 3
    path = float(steps.sum())
    return np.asarray([raw[-1, -1 - lag] for lag in (1, 2, 4, 8, 13)] + [
        steps.mean(), steps.std(), steps.max(), 0. if path <= 1e-12 else raw[0, -1] / path,
        raw[-1, :-3].min(), diameter, 0. if diameter <= 1e-12 else np.median(raw[separated]) / diameter])


def topology_features(raw):
    delayed = delayed_distance(raw)
    if len(delayed) != 12:
        raise ValueError('Expected twelve causal delayed states')
    ph = persistence(delayed)
    h0 = max((b - a for a, b in ph['h0_finite']), default=0.)
    result = [ph['h1_max_lifetime'], ph['normalized_h1'], h0]
    for scale in SCALES:
        reference = fit_reference(delayed, 11, scale)
        result.extend([len(reference['region_indices']) / 12, reference['net_to_path_ratio'], float(reference['eligible'])])
    return np.asarray(result)


def feature_sets(raw):
    simple, topology = simple_features(raw), topology_features(raw)
    dense = np.asarray(raw)[np.triu_indices(WINDOW, 1)]
    return dict(S=simple, ST=np.r_[simple, topology], D=dense, DT=np.r_[dense, topology])


def parent_weights(parents):
    parents = np.asarray(parents)
    unique, counts = np.unique(parents, return_counts=True)
    if not len(unique):
        raise ValueError('Nonempty training parents required')
    lookup = dict(zip(unique.tolist(), counts.tolist()))
    return np.asarray([1. / (len(unique) * lookup[p]) for p in parents])


def fit_ridge(x, y, weights):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    if (x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape or
            not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(w <= 0)):
        raise ValueError('Finite weighted regression arrays required')
    w = w / w.sum()
    mean = np.sum(x * w[:, None], axis=0)
    scale = np.sqrt(np.sum((x - mean) ** 2 * w[:, None], axis=0))
    scale = np.where(scale <= 1e-12, 1., scale)
    z, intercept = (x - mean) / scale, float(np.dot(w, y))
    coefficient = np.linalg.solve(z.T @ (w[:, None] * z) + RIDGE * np.eye(x.shape[1]),
                                  z.T @ (w * (y - intercept)))
    return dict(mean=mean.tolist(), scale=scale.tolist(), coefficient=coefficient.tolist(), intercept=intercept)


def predict_ridge(model, x):
    return ((np.asarray(x) - model['mean']) / model['scale']) @ np.asarray(model['coefficient']) + model['intercept']


def equal_parent_metrics(y, prediction, parents):
    y, prediction, parents = np.asarray(y), np.asarray(prediction), np.asarray(parents)
    rows = []
    for parent in sorted(set(parents.tolist())):
        difference = prediction[parents == parent] - y[parents == parent]
        rows.append(dict(parent=parent, count=len(difference), mae=float(np.abs(difference).mean()),
                         mse=float(np.mean(difference ** 2))))
    return dict(mae=float(np.mean([r['mae'] for r in rows])),
                rmse=float(np.sqrt(np.mean([r['mse'] for r in rows]))), parents=rows)


def noise_grid(seed_index, original):
    original = np.asarray(original, np.float32)
    if original.shape != (10, 24):
        raise ValueError('Expected full flow initial noise')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed_index]))
    return np.stack([original] + [rng.standard_normal(original.shape, dtype=np.float32) for _ in range(2)])
