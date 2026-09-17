"""Score archived B and Plus/Pro routing against the separately frozen A profile."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from v82_knn_research import (ROOT, REPO, CURRENT, RUN_B, GROUPS, archive, load_available,
                              standardized, ReferenceScorer, distance_attribution, first_crossing,
                              digest, save_json)


def load_profile():
    directory = ROOT / 'available-profile'
    frozen = json.loads((directory / 'frozen.json').read_text())
    for name, expected in frozen['files'].items():
        if digest(directory / name) != expected:
            raise ValueError(f'Separate frozen profile changed: {name}')
    params = json.loads((directory / 'parameters.json').read_text())
    assert not params['b_used_for_fitting'] and not params['current_plus_pro_used_for_fitting']
    assert params['source_protocol_sha256'] == digest(ROOT / 'PROTOCOL.md')
    return archive(directory / 'reference.npz'), params


def geometry(frame, values, profile):
    n, q, dim = values.shape
    assert dim == 10
    outputs = dict(score=np.full((n, q), np.nan, np.float32),
                   components=np.full((n, q, len(GROUPS)), np.nan, np.float32),
                   signed_residual=np.full((n, q, dim), np.nan, np.float32),
                   neighbor_ids=np.full((n, q, 20), -1, np.int16),
                   unique_neighbor_parents=np.full((n, q), -1, np.int16))
    scorer = ReferenceScorer(profile)
    started = time.perf_counter()
    for start in range(0, n, 256):
        stop = min(start+256, n)
        x = values[start:stop]
        finite = np.isfinite(x).all(-1)
        if finite.any():
            distances, ids = scorer.neighbors('success_dynamic', x[finite])
            score, parts, signed = distance_attribution(x[finite], profile['success_dynamic'], ids)
            np.testing.assert_allclose(score, distances.mean(-1), rtol=1e-8, atol=1e-10)
            parents = np.sort(profile['success_global_rows'][ids], axis=-1)
            unique = 1 + (np.diff(parents, axis=-1) != 0).sum(-1)
            for key, value in (('score', score), ('components', parts), ('signed_residual', signed),
                               ('neighbor_ids', ids), ('unique_neighbor_parents', unique)):
                outputs[key][start:stop][finite] = value
        if (start // 256) % 10 == 0 or stop == n:
            print(f'Geometry {stop}/{n}: {time.perf_counter()-started:.1f}s', flush=True)
    return outputs


def run_b():
    target = ROOT / 'available-B-geometry.npz'
    if target.exists():
        raise SystemExit('B geometry already exists; inspect it before another run')
    profile, params = load_profile()
    frame, cache = load_available()
    selected = frame.run_id.eq(RUN_B).to_numpy()
    frame = frame[selected].reset_index(drop=True)
    cache = {key: value[selected] for key, value in cache.items()}
    values, heads = standardized(cache, profile)
    outputs = geometry(frame, values, profile)
    frame['knn_available_first'] = first_crossing(outputs['score'], params['threshold'])
    frame['task_available_in_reference'] = frame.task.isin(params['reference_tasks'])
    np.savez_compressed(target, x=values.astype(np.float32), **outputs, valid=cache['valid'],
                        groups=np.asarray(list(GROUPS)), global_rows=frame.global_row.to_numpy(),
                        freeze_score=heads['freeze'], acceleration_score=heads['acceleration'],
                        periodicity_score=heads['periodicity'])
    frame.to_csv(ROOT / 'available-B-decisions.csv', index=False)
    save_json(ROOT / 'available-B-scoring.json', dict(episodes=len(frame),
              valid_queries=int(cache['valid'].sum()), scored_queries=int(np.isfinite(outputs['score']).sum()),
              profile_sha256=digest(ROOT / 'available-profile/reference.npz'),
              parameters_sha256=digest(ROOT / 'available-profile/parameters.json'),
              original_bank_used=False, model_forwards=0, environment_actions=0))


def run_current():
    target = ROOT / 'current-geometry.npz'
    if target.exists():
        raise SystemExit('Current geometry already exists; inspect it before another run')
    sys.path.insert(0, str(REPO / 'moe-trap-control'))
    from v82_closed_loop import V82Monitor
    profile, params = load_profile()
    episodes = json.loads((CURRENT / 'summary.json').read_text())['episodes']
    shape = (len(episodes), 52)
    cache = dict(mobility=np.full((*shape, 8), np.nan, np.float32),
                 acceleration=np.full(shape, np.nan, np.float32),
                 periodicity=np.full(shape, np.nan, np.float32), valid=np.zeros(shape, bool))
    v82_scores = np.full((*shape, 5), np.nan)
    v82_thresholds = np.full((*shape, 2), np.nan)
    rows, checks = [], []
    for row, episode in enumerate(episodes):
        path = CURRENT / episode['name'] / 'full-hb-routes.npz'
        assert digest(path) == episode['integrity']['full_hb_sha256']
        checkpoint = episode['policy_identity']['checkpoint_sha256']
        assert checkpoint in profile['checkpoint_set']
        trace = json.loads((Path(episode['source_artifact_dir']) / 'episode-trace.json').read_text())
        with np.load(path) as data:
            probability = data['hb_router_probs']
        monitor = V82Monitor()
        for q, p in enumerate(probability):
            status = monitor.update(p)
            cache['mobility'][row, q] = status['layer_mobility']
            cache['acceleration'][row, q] = status['route_acceleration']
            cache['periodicity'][row, q] = status['lag_periodicity']
            cache['valid'][row, q] = True
            v82_scores[row, q] = [status['freeze_score'], status['acceleration_score'],
                                  status['periodicity_score'], *status['v8_scores']]
            v82_thresholds[row, q] = status['v82_thresholds']
        first = monitor.first_v82_alarm
        assert first == episode['first_alarm_query_zero_based']['v82']
        assert monitor.v7.first_freeze_query == episode['first_head_query']['freeze']
        assert monitor.v7.first_turbulence_query == episode['first_head_query']['turbulence']
        np.testing.assert_array_equal(monitor.v82_first, [episode['first_head_query']['v82_inversion'],
                                                         episode['first_head_query']['v82_curvature']])
        rows.append(dict(name=episode['name'], benchmark=episode['benchmark'], source_dir=episode['source_artifact_dir'],
                         source_trace_sha256=episode['source_trace_sha256'], raw_hb_sha256=digest(path),
                         task=trace['task_name'], task_id=episode['base_task_id'], suite='libero_long',
                         init_state_id=episode['init_state_id'], length=len(probability),
                         failure=not episode['source_result']['success'], v82_frozen=first,
                         horizon_actions=520, action_steps=episode['action_steps'], checkpoint=checkpoint,
                         head_freeze=monitor.v7.first_freeze_query, head_turbulence=monitor.v7.first_turbulence_query,
                         head_inversion=monitor.v82_first[0], head_curvature=monitor.v82_first[1]))
        checks.append(dict(name=episode['name'], all_v82_heads_exact=True, queries=len(probability)))
        if (row+1) % 10 == 0:
            print(f'Replayed original v8.2 and raw 10D inputs: {row+1}/{len(episodes)}', flush=True)
    frame = pd.DataFrame(rows)
    values, heads = standardized(cache, profile)
    outputs = geometry(frame, values, profile)
    frame['knn_available_first'] = first_crossing(outputs['score'], params['threshold'])
    np.savez_compressed(target, x=values.astype(np.float32), **outputs, valid=cache['valid'],
                        groups=np.asarray(list(GROUPS)), names=frame.name.to_numpy(str),
                        v82_scores=v82_scores, v82_thresholds=v82_thresholds,
                        freeze_score=heads['freeze'], acceleration_score=heads['acceleration'],
                        periodicity_score=heads['periodicity'])
    np.savez_compressed(ROOT / 'current-routing-inputs.npz', **cache)
    frame.to_csv(ROOT / 'current-decisions.csv', index=False)
    save_json(ROOT / 'current-scoring.json', dict(episodes=len(frame),
              valid_queries=int(cache['valid'].sum()), scored_queries=int(np.isfinite(outputs['score']).sum()),
              profile_sha256=digest(ROOT / 'available-profile/reference.npz'),
              parameters_sha256=digest(ROOT / 'available-profile/parameters.json'),
              original_bank_used=False, model_forwards=0, environment_actions=0, replay_checks=checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=['B', 'current'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_b() if args.cohort == 'B' else run_current()
