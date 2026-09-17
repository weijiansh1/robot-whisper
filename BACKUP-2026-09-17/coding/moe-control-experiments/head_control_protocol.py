"""One-query, frozen-pool tests of route geometry and v8.2 head targeting."""

import copy
from pathlib import Path
import sys

import numpy as np

REPO = Path('/data/coding/robot-whisper-0909')
sys.path[:0] = [str(REPO / 'moe-trap-control'), str(REPO / 'himoe-route-capture')]
from route_noise_selector import route_centrality_scores, stable_argmin
from v8_closed_loop import limits, risk, score_status
from v82_closed_loop import V82Monitor, thresholds_at

PARENTS = (
    ('plus-task00-init026', 'curvature'),
    ('pro-task03-init039', 'freeze'),
    ('plus-task06-init026', 'turbulence'),
    ('plus-task03-init026', 'inversion'),
    ('plus-task03-init039', 'curvature'),
)
METHODS = ('native', 'random', 'center', 'edge', 'head_low', 'head_high', 'all_low')
HEADS = ('freeze', 'turbulence', 'inversion', 'curvature')
HEAD_TOLERANCE = 1e-6


def normalize_scores(scores, query):
    base, margins = limits()
    thresholds = thresholds_at('iid_v82', query, base)
    values = np.asarray(scores, np.float64)
    if values.shape[-1] != 5 or not np.isfinite(values).all():
        raise ValueError('Complete finite v8.2 scores required')
    normalized = (values - thresholds) / margins
    return normalized, thresholds, margins


def head_severity(normalized, head):
    values = np.asarray(normalized)
    if head not in HEADS or values.shape[-1] != 5:
        raise ValueError('Invalid target head or component scores')
    if head == 'turbulence':
        # Both historical alarm branches are latched. Target both current
        # continuous components; never try to erase their Boolean history.
        return np.maximum(values[..., 1], values[..., 2])
    return values[..., {'freeze': 0, 'inversion': 3, 'curvature': 4}[head]]


def tolerant_argmin(values, tolerance=HEAD_TOLERANCE):
    values = np.asarray(values, np.float64)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError('Expected four finite scores')
    return int(np.flatnonzero(values <= values.min() + tolerance)[0])


def preview_pool(monitor, routes, head, random_index):
    values = np.asarray(routes)
    if values.shape != (4, 8, 10, 11, 32) or values.dtype != np.float16:
        raise ValueError('Expected a four-candidate complete float16 HB pool')
    if not 0 <= random_index < 4:
        raise ValueError('Invalid seeded random candidate')
    query = monitor.v7.query + 1
    statuses = [copy.deepcopy(monitor).update(p) for p in values]
    if any(status['query'] != query for status in statuses):
        raise RuntimeError('Candidate preview advanced the shared history')
    scores = np.stack([score_status(status) for status in statuses])
    normalized, thresholds, margins = normalize_scores(scores, query)
    target = head_severity(normalized, head)
    all_risk = risk(scores, thresholds, margins)
    center = route_centrality_scores(values[:, 4:, :3, 1:])
    selected = dict(native=0, random=int(random_index), center=stable_argmin(center),
                    edge=stable_argmin(-center), head_low=tolerant_argmin(target),
                    head_high=tolerant_argmin(-target), all_low=tolerant_argmin(all_risk))
    return dict(query=query, head=head, selected=selected, scores=scores,
                normalized=normalized, thresholds=thresholds, margins=margins,
                target_severity=target, all_risk=all_risk, centrality=center,
                target_spread=float(np.ptp(target)),
                target_low_vs_native=float(target[selected['head_low']] - target[0]),
                target_high_vs_native=float(target[selected['head_high']] - target[0]),
                target_low_vs_high=float(target[selected['head_low']] - target[selected['head_high']]),
                detectable_target_contrast=bool(np.ptp(target) > HEAD_TOLERANCE),
                preview_statuses=statuses)


def first_head_queries(monitor):
    return dict(freeze=monitor.v7.first_freeze_query,
                turbulence=monitor.v7.first_turbulence_query,
                inversion=monitor.v82_first[0], curvature=monitor.v82_first[1])


def prefix_monitor(routes, query, expected_alarm, expected_head):
    monitor = V82Monitor()
    for probability in routes[:query]:
        monitor.update(probability)
    if monitor.first_v82_alarm != expected_alarm or first_head_queries(monitor)[expected_head] != expected_alarm:
        raise RuntimeError('The declared onset head does not match the frozen prefix')
    if query != expected_alarm + 1:
        raise ValueError('Intervention must start at q+1')
    return monitor
