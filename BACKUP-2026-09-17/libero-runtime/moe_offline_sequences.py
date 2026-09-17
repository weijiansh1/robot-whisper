"""Fixed causal sequence scores and exact paired-row order controls."""

from itertools import combinations
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.stats import rankdata

ROOT = Path('/data/libero-runtime/samples/moe-offline-sequences-20260915')
EXPANDED = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')
PREVIOUS = Path('/data/libero-runtime/samples/v82-knn-trajectory-20260915')
OLD = Path('/data/libero-runtime/samples/moe-joint-patterns-20260915')
CONTROL = Path('/data/coding/moe-control-experiments/runs')
METHODS = ('freeze', 'activity', 'simultaneous', 'activity_then_freeze',
           'reverse', 'order_excess', 'direction_gap')
PARTITIONS = [(list(early), [i for i in range(6) if i not in early])
              for early in combinations(range(6), 3)]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)):
            return float(x) if np.isfinite(x) else None
        if isinstance(x, np.generic):
            return x.item()
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')


def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def score_window(window):
    x = np.asarray(window, float)
    if x.shape[-2:] != (6, 2):
        raise ValueError('Expected six paired mobility/acceleration coordinates')
    f, a = -x[..., 0], x[..., 1]
    freeze, active = f[..., 3:].min(-1), a[..., 3:].min(-1)
    forward = np.minimum(a[..., :3].min(-1), freeze)
    reverse = np.minimum(f[..., :3].min(-1), active)
    null = sum(np.minimum(a[..., early].min(-1), f[..., late].min(-1))
               for early, late in PARTITIONS)/len(PARTITIONS)
    return np.stack((freeze, active, np.minimum(freeze, active), forward,
                     reverse, forward-null, forward-reverse), axis=-1)


def sequence_scores(coordinates):
    x = np.asarray(coordinates, float)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError('Expected [episode, query, 2] coordinates')
    output = np.full((*x.shape[:2], len(METHODS)), np.nan)
    for q in range(12, x.shape[1]):
        window = x[:, q-5:q+1]
        good = np.isfinite(window).all(axis=(1, 2))
        output[good, q] = score_window(window[good])
    return output


def first_events(scores, threshold=np.log(1.2)):
    hits = np.isfinite(scores) & (scores >= threshold)
    return np.where(hits.any(axis=1), hits.argmax(axis=1), -1).astype(np.int16)


def auc(labels, values):
    y, value = np.asarray(labels, bool), np.asarray(values, float)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite score in matched comparison')
    n = int(y.sum())
    if n == 0 or n == len(y):
        return np.nan
    return float((rankdata(value)[y].sum()-n*(n+1)/2)/(n*(len(y)-n)))


def interval(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(2026091504)
    means = values[rng.integers(len(values), size=(2000, len(values)))].mean(-1)
    return tuple(np.quantile(means, [.025, .975]))
