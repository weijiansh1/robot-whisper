"""Join fresh native actions with captured routes and score frozen monitors."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from alarm_trajectory_study import ROOT, METHODS, score_trajectories, load_bank
from score_v82_knn_reference import load_profile
from v82_knn_research import REPO, archive, standardized, ReferenceScorer, first_crossing, digest, save_json


def main():
    if (ROOT / 'fresh-scores.npz').exists():
        raise SystemExit('Fresh scores already exist')
    bank, params = load_bank()
    previous, previous_params = load_profile()
    wire = {}
    for path in sorted((ROOT / 'fresh-wire').glob('call-*.npz')):
        with np.load(path) as data:
            key = tuple(str(data[field]) for field in ('noise_sha256', 'image_sha256', 'state_sha256'))
        if key in wire:
            raise ValueError('Duplicate new rollout noise identity')
        wire[key] = path
    episodes = json.loads((ROOT / 'fresh-manifest.json').read_text())['episodes']
    sys.path.insert(0, str(REPO / 'moe-trap-control'))
    from v82_closed_loop import V82Monitor
    shape = (len(episodes), 52)
    raw = dict(mobility=np.full((*shape, 8), np.nan, np.float32),
               acceleration=np.full(shape, np.nan, np.float32), periodicity=np.full(shape, np.nan, np.float32),
               valid=np.zeros(shape, bool))
    head_scores = np.full((*shape, 5), np.nan)
    rows, matches = [], []
    output = ROOT / 'fresh-routes'
    output.mkdir()
    for i, episode in enumerate(episodes):
        trace_path = Path(episode['source_artifact_dir']) / 'episode-trace.npz'
        assert digest(trace_path) == episode['source_trace_sha256']
        assert episode['policy_identity']['checkpoint_sha256'] in previous['checkpoint_set']
        with np.load(trace_path) as data:
            trace = {key: data[key] for key in ('flow_noises', 'images', 'states', 'predicted_actions')}
        monitor, routes, identities = V82Monitor(), [], []
        for q, noise in enumerate(trace['flow_noises']):
            key = tuple(hashlib.sha256(value.tobytes()).hexdigest() for value in
                        (noise, trace['images'][q], trace['states'][q]))
            path = wire[key]
            with np.load(path) as data:
                hb, ids = data['hb'], data['ids']
                np.testing.assert_array_equal(data['actions'], trace['predicted_actions'][q])
                assert str(data['image_sha256']) == hashlib.sha256(trace['images'][q].tobytes()).hexdigest()
                assert str(data['state_sha256']) == hashlib.sha256(trace['states'][q].tobytes()).hexdigest()
            status = monitor.update(hb)
            raw['valid'][i, q] = True
            for dest, key in (('mobility', 'layer_mobility'), ('acceleration', 'route_acceleration'), ('periodicity', 'lag_periodicity')):
                raw[dest][i, q] = status[key]
            head_scores[i, q] = [status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']]
            routes.append(hb)
            identities.append(ids)
            matches.append(dict(name=episode['name'], query=q, wire=path.name, native_actions_exact=True))
        np.savez_compressed(output / (episode['name']+'.npz'), hb=np.stack(routes), ids=np.stack(identities))
        rows.append(dict(name=episode['name'], benchmark=episode['benchmark'], task_id=episode['base_task_id'],
                          task=episode['task_name'], suite='libero_long', init_state_id=episode['init_state_id'],
                          source_dir=episode['source_artifact_dir'], length=len(routes), horizon_actions=520,
                          action_steps=episode['action_steps'], failure=not episode['source_result']['success'],
                          v82_frozen=monitor.first_v82_alarm, head_freeze=monitor.v7.first_freeze_query,
                          head_turbulence=monitor.v7.first_turbulence_query,
                          head_inversion=monitor.v82_first[0], head_curvature=monitor.v82_first[1]))
    assert len(matches) == len(wire)
    frame = pd.DataFrame(rows)
    values, _ = standardized(raw, previous)
    scores = score_trajectories(values, bank)
    finite = np.isfinite(values).all(-1)
    instantaneous = np.full(shape, np.nan, np.float32)
    distances, ids = ReferenceScorer(previous).neighbors('success_dynamic', values[finite])
    instantaneous[finite] = distances.mean(-1)
    scores['instantaneous'] = instantaneous
    scores['v82_scores'] = head_scores
    frame['knn_available_first'] = first_crossing(instantaneous, previous_params['threshold'])
    for method in METHODS:
        frame[method+'_first'] = first_crossing(scores[method], params['thresholds'][method])
    np.savez_compressed(ROOT / 'fresh-scores.npz', **scores)
    np.savez_compressed(ROOT / 'fresh-routing-inputs.npz', **raw)
    frame.to_csv(ROOT / 'fresh-decisions.csv', index=False)
    pd.DataFrame(matches).to_csv(ROOT / 'fresh-native-capture-matches.csv', index=False)
    save_json(ROOT / 'fresh-scoring.json', dict(episodes=len(frame), native_calls_exact=len(matches),
              profile_sha256=digest(ROOT / 'profile/trajectory-bank.npz'),
              parameters_sha256=digest(ROOT / 'profile/parameters.json'), posthoc_fitting=False))
    print(frame[['name', 'failure', 'v82_frozen', 'knn_available_first', 'displacement_first', 'endpoint_first']].to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
