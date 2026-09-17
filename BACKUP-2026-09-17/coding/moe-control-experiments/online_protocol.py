"""Frozen route-only candidate selection and physical-time noise pairing."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
UPSTREAM = Path('/data/coding/robot-whisper-0909')
sys.path.insert(0, str(UPSTREAM / 'himoe-route-capture'))
from route_noise_selector import route_centrality_scores, stable_argmin

ARMS = ('native', 'short', 'random_short', 'route_short')
PARENTS = ('plus-task00-init026', 'pro-task03-init039',
           'plus-task06-init026', 'plus-task03-init039', 'plus-task00-init047')
SEED = 2026091501
HORIZON = 520
RECOVERY_QUERIES = 12
CANDIDATES = 4


def seeded_rng(parent, step, stream):
    if step < 0 or step >= HORIZON or step % 5:
        raise ValueError('Expected a five-step physical-time boundary')
    identity = [SEED, {'plus': 0, 'pro': 1}[parent['benchmark']],
                int(parent['base_task_id']), int(parent['init_state_id']), step, stream]
    return np.random.default_rng(np.random.SeedSequence(identity))


def noise_bank(flow_seed):
    rng = np.random.default_rng(flow_seed)
    return np.stack([rng.standard_normal((10, 24)).astype(np.float32)
                     for _ in range(HORIZON // 10)])


def candidate_noises(parent, bank, step, count):
    if count not in (1, CANDIDATES):
        raise ValueError('Unsupported candidate count')
    native = (bank[step // 10].copy() if step % 10 == 0 else
              seeded_rng(parent, step, 0).standard_normal((10, 24)).astype(np.float32))
    values = [native]
    for candidate in range(1, count):
        values.append(seeded_rng(parent, step, candidate).standard_normal((10, 24)).astype(np.float32))
    return np.stack(values)


def select_candidate(arm, routes, parent, step):
    values = np.asarray(routes)
    expected = 1 if arm == 'short' else CANDIDATES
    if arm not in ARMS[1:] or values.shape != (expected, 8, 10, 11, 32):
        raise ValueError('Unsupported arm or HB pool shape')
    if not np.isfinite(values).all() or (values < 0).any() or (values.sum(-1) <= 0).any():
        raise ValueError('Invalid route probabilities')
    if arm == 'short':
        return 0, np.zeros(1, dtype=np.float64)
    scores = route_centrality_scores(values[:, 4:, :3, 1:])
    if arm == 'random_short':
        return int(seeded_rng(parent, step, 100).integers(CANDIDATES)), scores
    return stable_argmin(scores), scores


def query_mode(arm, recovery_used):
    if arm not in ARMS or recovery_used < 0:
        raise ValueError('Invalid controller state')
    active = arm != 'native' and recovery_used < RECOVERY_QUERIES
    return dict(active=active, chunk=5 if active else 10,
                candidates=CANDIDATES if active and arm != 'short' else 1)
