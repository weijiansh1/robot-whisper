"""Causal reference regions and topology diagnostics, never a success label."""

from pathlib import Path
import sys

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

DEPS = Path(__file__).resolve().parent / '.topology-deps-iUKcff'
REPRESENTATIONS = ('back_path', 'full_path', 'back_last')
SCALES = (.75, 1., 1.25)
PRIMARY = ('back_path', 1.)
HISTORY = 3
REFERENCE = 12
SEED = 2026091519
SURROGATES = 99


def synthetic_circle(length=52, period=8):
    theta = np.arange(period) * 2 * np.pi / period
    return np.tile(np.c_[np.cos(theta), np.sin(theta)], (length // period + 1, 1))[:length]


def validate_distance(distance):
    distance = np.asarray(distance, np.float64)
    if (distance.ndim != 2 or distance.shape[0] != distance.shape[1] or
            not np.isfinite(distance).all() or np.any(distance < 0) or
            not np.allclose(distance, distance.T, atol=1e-12, rtol=0) or
            not np.allclose(np.diag(distance), 0, atol=1e-12, rtol=0)):
        raise ValueError('Expected a finite symmetric distance matrix with zero diagonal')
    return distance


def route_distance(probabilities, representation):
    p = np.asarray(probabilities)
    if p.ndim != 5 or p.shape[1:] != (8, 10, 11, 32) or representation not in REPRESENTATIONS:
        raise ValueError('Expected query/layer/flow/token/expert routing probabilities')
    if representation == 'back_path':
        p = p[:, 4:, :, 1:]
    elif representation == 'full_path':
        p = p[:, :, :, 1:]
    else:
        p = p[:, 4:, -1:, 1:]
    p = p.astype(np.float64)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(-1) <= 0):
        raise ValueError('Invalid probabilities')
    p /= p.sum(-1, keepdims=True)
    cells = int(np.prod(p.shape[1:-1]))
    points = np.sqrt(p).reshape(len(p), -1) / np.sqrt(2 * cells)
    return squareform(pdist(points))


def delayed_distance(raw, history=HISTORY):
    raw = validate_distance(raw)
    if not isinstance(history, int) or history < 1 or len(raw) < history:
        raise ValueError('Insufficient history')
    times = np.arange(history - 1, len(raw))
    squared = sum(raw[np.ix_(times - lag, times - lag)] ** 2 for lag in range(history)) / history
    return np.sqrt(squared)


def fit_reference(distance, anchor, scale=1.):
    distance = validate_distance(distance)
    if anchor < REFERENCE - 1 or anchor >= len(distance) or scale <= 0:
        raise ValueError('Reference must contain twelve causal states')
    indices = np.arange(anchor - REFERENCE + 1, anchor + 1)
    local = distance[np.ix_(indices, indices)]
    separated = np.abs(indices[:, None] - indices[None, :]) >= HISTORY
    nearest = np.where(separated, local, np.inf).min(1)
    epsilon = max(1e-12, float(np.quantile(nearest, .95)) * scale)
    labels = fcluster(linkage(squareform(local, checks=False), method='complete'),
                      t=epsilon, criterion='distance').astype(int) - 1
    counts = np.zeros((int(labels.max()) + 1,) * 2, np.int64)
    np.add.at(counts, (labels[:-1], labels[1:]), 1)
    _, components = connected_components(csr_matrix(counts), directed=True, connection='strong')
    component = int(components[labels[-1]])
    selected = np.flatnonzero(components[labels] == component)
    nodes = np.flatnonzero(components == component)
    path = float(sum(local[a, b] for a, b in zip(selected[:-1], selected[1:])))
    net_ratio = 0. if path <= 1e-12 else float(local[selected[0], selected[-1]] / path)
    visited = len(selected) >= 4 and int(selected[-1] - selected[0]) >= 3
    cyclic = len(nodes) > 1 or bool(counts[nodes[0], nodes[0]])
    eligible = visited and cyclic and net_ratio <= .35
    return dict(epsilon=epsilon, indices=indices.tolist(), labels=labels.tolist(),
                transition_counts=counts.tolist(), components=components.tolist(),
                anchor_component=component, region_indices=indices[selected].tolist(),
                net_to_path_ratio=net_ratio, eligible=bool(eligible),
                reason='eligible' if eligible else 'insufficient_recurrence_or_directional_drift')


def intervals(values):
    values = np.asarray(values, bool)
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1) - 1).tolist()))


def classify_membership(inside, steps):
    inside, steps = np.asarray(inside, bool), np.asarray(steps, int)
    if inside.ndim != 1 or len(inside) != len(steps) or not len(inside) or not inside[0]:
        raise ValueError('A paired branch must start in its reference region')
    if not np.all(np.diff(steps) == 10):
        raise ValueError('Expected the common physical ten-step grid')
    exits = [(a, b) for a, b in intervals(~inside) if b - a + 1 >= 3]
    returns = [(a, b) for a, b in intervals(inside) if b - a + 1 >= 2]
    came_back = any(a > exit_end for _, exit_end in exits for a, _ in returns)
    if not exits:
        label = 'no_confirmed_exit'
    elif exits[-1][1] == len(inside) - 1 and exits[-1][1] - exits[-1][0] + 1 >= 4:
        label = 'last_exit_no_observed_return'
    elif len(inside) >= 2 and inside[-2:].all() and came_back:
        label = 'exit_then_return'
    else:
        label = 'insufficient_followup_or_mixed'
    return dict(category=label, first_exit_step=None if not exits else int(steps[exits[0][0]]),
                first_exit_confirmed_step=None if not exits else int(steps[exits[0][0] + 2]),
                observed_return=bool(came_back), outside_fraction=float(np.mean(~inside)),
                terminal_observation_step=int(steps[-1]),
                exit_intervals=[[int(steps[a]), int(steps[b])] for a, b in exits])


def region_diagnostic(distance, anchor, reference):
    nearest = np.asarray(distance)[anchor:, reference['region_indices']].min(1)
    inside = nearest <= reference['epsilon']
    steps = (np.arange(anchor, len(distance)) + HISTORY - 1) * 10
    if reference['eligible']:
        result = classify_membership(inside, steps)
    else:
        result = dict(category='reference_not_established', first_exit_step=None,
                      first_exit_confirmed_step=None, observed_return=False,
                      outside_fraction=float(np.mean(~inside)), terminal_observation_step=int(steps[-1]),
                      exit_intervals=[])
    return dict(result, nearest_distance=nearest.tolist(), inside=inside.tolist(), steps=steps.tolist())


def gudhi_module():
    if str(DEPS) not in sys.path:
        sys.path.insert(0, str(DEPS))
    import gudhi
    if gudhi.__version__ != '3.13.0':
        raise RuntimeError('Expected the isolated pinned GUDHI version')
    return gudhi


def persistence(distance):
    distance = validate_distance(distance)
    gd = gudhi_module()
    tree = gd.RipsComplex(distance_matrix=distance).create_simplex_tree(max_dimension=2)
    tree.compute_persistence(homology_coeff_field=2, min_persistence=0)
    h0, h1 = (tree.persistence_intervals_in_dimension(dim) for dim in (0, 1))
    finite_h0 = h0[np.isfinite(h0[:, 1])].tolist()
    if len(h1) and not np.isfinite(h1).all():
        raise AssertionError('Full finite-distance filtration must kill H1')
    lifetime = float(np.max(h1[:, 1] - h1[:, 0])) if len(h1) else 0.
    diameter = float(distance.max())
    return dict(h0_finite=finite_h0, h1=h1.tolist(), h1_max_lifetime=lifetime, diameter=diameter,
                normalized_h1=0. if diameter <= 1e-12 else lifetime / diameter)


def block_order(length, rng, size=3):
    blocks = [np.arange(i, min(i + size, length)) for i in range(0, length, size)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])


def topology_nulls(raw_suffix, seed, count=SURROGATES):
    raw_suffix = validate_distance(raw_suffix)
    observed = persistence(delayed_distance(raw_suffix))
    rng = np.random.default_rng(seed)
    nulls = dict(shuffle=[], block3=[])
    for _ in range(count):
        for mode in nulls:
            order = rng.permutation(len(raw_suffix)) if mode == 'shuffle' else block_order(len(raw_suffix), rng)
            permuted = raw_suffix[np.ix_(order, order)]
            nulls[mode].append(persistence(delayed_distance(permuted))['normalized_h1'])
    tails = {mode: (1 + sum(v >= observed['normalized_h1'] for v in values)) / (count + 1)
             for mode, values in nulls.items()}
    return dict(observed=observed, nulls=nulls, descriptive_upper_tail=tails,
                interpretation='exploratory surrogate ranks, not corrected significance tests or failure labels')
