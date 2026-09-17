"""Shared read-only inputs and exact diagnostics for the v8.2/kNN study."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from moe_response_analysis import digest, save_json

ROOT = Path('/data/libero-runtime/samples/v82-knn-mechanism-20260915')
REPO = Path('/data/coding/robot-whisper-0909')
SAFE = REPO / 'safe&vlaconf/moe_trainfree'
FROZEN = ROOT / 'inputs/moe-trap-control/design/frozen_alarm_comparison_20260908'
EXPANDED = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')
RAW = EXPANDED / 'inputs/moe-v7-legacy16x32-0906/results/raw_features'
CURRENT = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')
RUN_A, RUN_B = 'right-50x8-20260903', 'right-50x8b-20260903'
HORIZONS = {'libero_goal': 300, 'libero_long': 520, 'libero_object': 280, 'libero_spatial': 220}
GROUPS = {'front_mobility': slice(0, 4), 'back_mobility': slice(4, 8),
          'acceleration': slice(8, 9), 'periodicity': slice(9, 10)}
sys.path.insert(0, str(SAFE / 'boundary_knn'))
from knn import ReferenceScorer, dynamics, reference_pairs, reference_scaling


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_frozen():
    sealed = json.loads((FROZEN / 'verification.json').read_text())
    for name in ('index.csv', 'first_alarms.csv'):
        if digest(FROZEN / name) != sealed['artifacts'][name]:
            raise ValueError(f'Original decision hash mismatch: {name}')
    frame, first = (pd.read_csv(FROZEN / name) for name in ('index.csv', 'first_alarms.csv'))
    for key in ('source', 'episode', 'run_id', 'task', 'init_state_id', 'length', 'failure', 'global_row'):
        np.testing.assert_array_equal(frame[key], first[key])
    assert len(frame) == 32000 and not frame.duplicated(['run_id', 'task', 'episode']).any()
    for method in ('v7_frozen', 'v8_frozen', 'v82_frozen', 'knn20'):
        value = first[method].to_numpy(int)
        assert ((value == -1) | ((value >= 0) & (value < frame.length))).all()
        frame[method] = value
    frame['horizon_actions'] = frame.suite.map(HORIZONS)
    assert frame.horizon_actions.notna().all()
    return frame


def load_available():
    frame = load_frozen()
    lookup = pd.MultiIndex.from_frame(frame[['run_id', 'task', 'episode']])
    indices, collected = [], {key: [] for key in ('mobility', 'acceleration', 'periodicity', 'valid')}
    for cohort, run in (('development_main', RUN_A), ('external_8b', RUN_B)):
        path = RAW / (cohort + '_route_features.npz')
        extraction = json.loads((RAW / (cohort + '_extraction_audit.json')).read_text())
        assert digest(path) == extraction['output_sha256']
        data = archive(path)
        assert str(data['run_id']) == run
        tasks = data['task_names'][data['task_index']].astype(str)
        keys = pd.MultiIndex.from_arrays([np.repeat(run, len(tasks)), tasks, data['episode']])
        positions = lookup.get_indexer(keys)
        assert (positions >= 0).all() and len(np.unique(positions)) == len(positions)
        np.testing.assert_array_equal(frame.iloc[positions].length, data['length'])
        np.testing.assert_array_equal(data['valid'], np.arange(52)[None] < data['length'][:, None])
        indices.append(positions)
        for target, source in (('mobility', 'mobility'), ('acceleration', 'route_acceleration'),
                               ('periodicity', 'lag_periodicity'), ('valid', 'valid')):
            collected[target].append(data[source])
    positions = np.concatenate(indices)
    result = frame.iloc[positions].reset_index(drop=True)
    result['available_row'] = np.arange(len(result))
    assert len(result) == 30400 and result.global_row.nunique() == len(result)
    return result, {key: np.concatenate(value) for key, value in collected.items()}


def standardized(cache, profile):
    values, heads = dynamics(cache['mobility'], cache['acceleration'], cache['periodicity'],
                             float(profile['periodicity_scale']))
    values[~cache['valid']] = np.nan
    return ((values.astype(float) - profile['dynamic_center']) / profile['dynamic_scale']), heads


def distance_attribution(points, bank, neighbors):
    points, bank = np.asarray(points, float), np.asarray(bank, float)
    if points.ndim != 2 or points.shape[1] != 10 or bank.ndim != 2 or bank.shape[1] != 10:
        raise ValueError('Expected 10D points and reference bank')
    selected = np.asarray(neighbors)
    if selected.ndim != 2 or selected.shape[0] != len(points) or selected.shape[1] < 1:
        raise ValueError('Invalid neighbor shape')
    if selected.dtype.kind not in 'iu' or (selected < 0).any() or (selected >= len(bank)).any():
        raise ValueError('Invalid neighbor identity')
    delta = points[:, None] - bank[selected]
    distance = np.linalg.norm(delta, axis=-1)
    per_axis = np.divide(delta ** 2, distance[..., None], out=np.zeros_like(delta),
                         where=distance[..., None] > 0)
    components = np.stack([per_axis[..., cols].sum(-1).mean(-1) for cols in GROUPS.values()], axis=-1)
    score = distance.mean(-1)
    np.testing.assert_allclose(components.sum(-1), score, rtol=1e-12, atol=1e-12)
    return score, components, delta.mean(1)


def first_crossing(score, threshold):
    score = np.asarray(score)
    if score.ndim != 2 or not np.isfinite(threshold):
        raise ValueError('Expected finite fixed threshold and [episode, query] scores')
    hits = np.isfinite(score) & (score > threshold)
    return np.where(hits.any(1), hits.argmax(1), -1).astype(np.int16)


def combine_first(v82, knn):
    v82, knn = np.asarray(v82, int), np.asarray(knn, int)
    either = np.minimum(np.where(v82 >= 0, v82, 10000), np.where(knn >= 0, knn, 10000))
    both = np.where((v82 >= 0) & (knn >= 0), np.maximum(v82, knn), -1)
    return {'v82': v82, 'knn': knn, 'OR': np.where(either < 10000, either, -1), 'AND_latched': both}


def alarm_metrics(frame, first, cutoff=10000):
    value = np.asarray(first)
    y = frame.failure.to_numpy(bool)
    hit = (value >= 0) & (value < frame.length.to_numpy()) & (value <= cutoff)
    tp, fp, positives, negatives = int((hit & y).sum()), int((hit & ~y).sum()), int(y.sum()), int((~y).sum())
    return dict(episodes=len(frame), failures=positives, successes=negatives, tp=tp, fp=fp,
                recall=tp / positives if positives else None, fpr=fp / negatives if negatives else None,
                precision=tp / (tp+fp) if tp+fp else None)


def alarm_groups(v82, knn, cutoff=10000):
    v, k = [(np.asarray(x) >= 0) & (np.asarray(x) <= cutoff) for x in (v82, knn)]
    return np.select([v & k, k & ~v, v & ~k], ['both', 'knn_only', 'v82_only'], default='neither')


def common_prefix_pairs(frame, methods):
    rows = []
    for (task, init), group in frame.groupby(['task', 'init_state_id'], sort=True):
        success, failure = group[~group.failure], group[group.failure]
        for f in failure.itertuples():
            for s in success.itertuples():
                horizon = min(f.length, s.length)
                row = dict(task=task, init_state_id=int(init), failure_row=int(f.Index),
                           success_row=int(s.Index), queries=int(horizon))
                for method, alarms in methods.items():
                    row[method + '_failure_hit'] = bool(0 <= alarms[f.Index] < horizon)
                    row[method + '_success_hit'] = bool(0 <= alarms[s.Index] < horizon)
                rows.append(row)
    return pd.DataFrame(rows)


def auc(labels, scores):
    labels, scores = np.asarray(labels, bool), np.asarray(scores, float)
    finite = np.isfinite(scores)
    y, x = labels[finite], scores[finite]
    p, n = int(y.sum()), int((~y).sum())
    if not p or not n:
        return np.nan
    ranks = rankdata(x)
    return float((ranks[y].sum() - p*(p+1)/2) / (p*n))
