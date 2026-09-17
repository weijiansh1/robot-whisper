"""Frozen success-prefix continuation scores and external window readouts."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

from v82_knn_research import ROOT as PREVIOUS, archive, standardized, digest, save_json
from score_v82_knn_reference import load_profile

ROOT = Path('/data/libero-runtime/samples/v82-knn-trajectory-20260915')
HORIZON = 3
PREFIX = 3
MIN_QUERY = 12
METHODS = ('displacement', 'endpoint')


def score_trajectories(values, bank):
    n, width, _ = values.shape
    output = {key: np.full((n, width), np.nan, np.float32) for key in
              ('displacement', 'endpoint', 'rematched_endpoint', 'prefix_distance',
               'reference_disagreement', 'direction_cosine')}
    output['signed_residual'] = np.full((n, width, 10), np.nan, np.float32)
    output['neighbor_ids'] = np.full((n, width, 20), -1, np.int16)
    output['unique_parents'] = np.full((n, width), -1, np.int16)
    index = NearestNeighbors(n_neighbors=20, n_jobs=1).fit(bank['prefix'])
    end_index = NearestNeighbors(n_neighbors=20, n_jobs=1).fit(bank['end'])
    for start in range(0, n, 128):
        stop = min(start+128, n)
        points, locations, deltas, ends = [], [], [], []
        for row in range(start, stop):
            for q in range(MIN_QUERY, width):
                prefix = values[row, q-5:q-2]
                if np.isfinite(prefix).all() and np.isfinite(values[row, q]).all():
                    points.append(prefix.reshape(-1)/np.sqrt(PREFIX))
                    locations.append((row, q))
                    ends.append(values[row, q])
                    deltas.append(values[row, q]-values[row, q-HORIZON])
        if not points:
            continue
        distances, ids = index.kneighbors(np.asarray(points))
        ends, deltas = np.asarray(ends), np.asarray(deltas)
        predicted = bank['delta'][ids]
        residual = deltas[:, None]-predicted
        endpoint_error = np.linalg.norm(ends[:, None]-bank['end'][ids], axis=-1).mean(-1)
        rematched = end_index.kneighbors(ends, return_distance=True)[0].mean(-1)
        assert (rematched <= endpoint_error+1e-8).all()
        mean_delta = predicted.mean(1)
        denominator = np.linalg.norm(deltas, axis=-1)*np.linalg.norm(mean_delta, axis=-1)
        cosine = np.divide((deltas*mean_delta).sum(-1), denominator,
                           out=np.full(len(deltas), np.nan), where=denominator > 1e-12)
        parents = np.sort(bank['global_rows'][ids], axis=-1)
        i, q = np.asarray(locations).T
        for key, value in dict(displacement=np.linalg.norm(residual, axis=-1).mean(-1),
                               endpoint=endpoint_error, rematched_endpoint=rematched,
                               prefix_distance=distances.mean(-1), signed_residual=residual.mean(1),
                               reference_disagreement=np.linalg.norm(predicted-mean_delta[:, None], axis=-1).mean(-1),
                               direction_cosine=cosine, neighbor_ids=ids,
                               unique_parents=1+(np.diff(parents, axis=-1) != 0).sum(-1)).items():
            output[key][i, q] = value
        if start % 1280 == 0 or stop == n:
            print('Continuation scores: %d/%d parents' % (stop, n), flush=True)
    return output


def load_bank():
    directory = ROOT / 'profile'
    frozen = json.loads((directory / 'frozen.json').read_text())
    for name, expected in frozen['files'].items():
        assert digest(directory / name) == expected
    params = json.loads((directory / 'parameters.json').read_text())
    assert params['protocol_sha256'] == digest(ROOT / 'PROTOCOL.md')
    assert params['previous_profile_sha256'] == digest(PREVIOUS / 'available-profile/reference.npz')
    return archive(directory / 'trajectory-bank.npz'), params


def physical_window(data, q, horizon=3):
    last = len(data['predicates'])-1
    end = min(q+horizon, last)
    if not 0 <= q < end:
        raise ValueError('A nonempty observed physical window is required')
    predicates = data['predicates'][q:end+1]
    eef = data['eef'][q:end+1]
    positions = data['positions'][q:end+1]
    gained = int((predicates[1:] & ~predicates[0]).any(0).sum())
    lost = int((~predicates[1:] & predicates[0]).any(0).sum())
    eef_path = float(np.linalg.norm(np.diff(eef, axis=0), axis=-1).sum())
    eef_net = float(np.linalg.norm(eef[-1]-eef[0]))
    object_motion = float(np.linalg.norm(positions-positions[:1], axis=-1).max())
    reduction = data['goal_distance'][q]-data['goal_distance'][q:end+1]
    geometric_gain = float(np.nanmax(reduction)) if np.isfinite(reduction).any() else 0.
    source_distance = np.linalg.norm(positions-eef[:, None], axis=-1)
    approach = float((source_distance[:1]-source_distance).max())
    terminal_success = bool(predicates[-1].all())
    if terminal_success:
        category = 'completed'
    elif gained:
        category = 'subgoal_gain'
    elif lost:
        category = 'subgoal_loss'
    elif geometric_gain > .01 or approach > .01:
        category = 'geometric_approach_proxy'
    elif object_motion < .003 and eef_path > .03 and eef_net < .008:
        category = 'return_motion_without_subgoal_gain'
    elif object_motion < .003 and eef_path < .015:
        category = 'holding_or_static'
    else:
        category = 'motion_without_verified_subgoal_gain'
    return dict(query=q, end_query=end, observed_queries=end-q, category=category,
                predicate_gain=gained, predicate_loss=lost, predicate_count=int(predicates[0].sum()),
                total_predicates=predicates.shape[1], grasp=bool(data['grasp'][q].any()),
                terminal_success=terminal_success, eef_path_m=eef_path, eef_net_m=eef_net,
                target_motion_m=object_motion, geometric_distance_gain_m=geometric_gain, approach_m=approach)


def phase_signature(data, q):
    return tuple(data['predicates'][q].tolist()) + (bool(data['grasp'][q].any()),)
