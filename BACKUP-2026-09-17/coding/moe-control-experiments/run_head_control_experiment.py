"""Execute every frozen candidate at one alarm state with matched native tails."""

import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, infer_isolated, load_isolated
from head_control_protocol import (PARENTS, METHODS, prefix_monitor, preview_pool,
                                   head_severity, normalize_scores)
from online_protocol import candidate_noises, noise_bank, seeded_rng
from run_online_experiment import (EnvironmentWorker, EVALUATION, assert_source_observation,
                                   equal, frozen_files as previous_frozen_files,
                                   jsonable, now, save_json, sha_array)
from collection_routes import CAPTURE_KEY, PROBS_KEY
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from v8_closed_loop import score_status


def frozen_files():
    result = previous_frozen_files()
    for name in ('P3B_HEAD_CONTROL_PLAN.zh.md', 'head_control_protocol.py',
                 'run_head_control_experiment.py', 'test_head_control_protocol.py'):
        result[str(BASE / name)] = sha256_file(BASE / name)
    return result


def parent_sources():
    with (EVALUATION / 'episodes.csv').open() as stream:
        indexed = {row['name']: row for row in csv.DictReader(stream)}
    parents = []
    for name, head in PARENTS:
        row = indexed[name]
        source = Path(row['source_artifact_dir'])
        manifest, arrays = load_episode_trace(source)
        alarm, query = int(row['first_v82_query']), int(row['first_v82_query']) + 1
        if alarm < 0 or query >= len(arrays['images']):
            raise RuntimeError('No valid post-alarm intervention query')
        with np.load(EVALUATION / name / 'full-hb-routes.npz', allow_pickle=False) as archive:
            reference = archive['hb_router_probs']
        prefix_monitor(reference, query, alarm, head)
        equal(noise_bank(manifest['config']['flow_noise_seed'])[:len(arrays['flow_noises'])],
              arrays['flow_noises'], 'native noise sequence')
        parents.append(dict(name=name, head=head, query=query, step=query * 10, first_alarm=alarm,
                            benchmark=row['benchmark'], base_task_id=int(row['base_task_id']),
                            init_state_id=int(row['init_state_id']), source=str(source),
                            source_success=row['success'] == 'True', source_steps=int(row['action_steps']),
                            source_queries=int(row['queries']),
                            manifest_sha256=sha256_file(source / 'episode-trace.json'),
                            trace_sha256=sha256_file(source / 'episode-trace.npz'),
                            hb_sha256=sha256_file(EVALUATION / name / 'full-hb-routes.npz')))
    return parents


def prepare():
    root = Path(tempfile.mkdtemp(prefix='p3b-head-control-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_head_control.single_query.v1', utc=now(), parents=parent_sources(),
                  methods=list(METHODS), candidate_order=[0, 1, 2, 3], source_hashes=frozen_files(),
                  chunk=10, intervention_queries=1, horizon=520, candidate_count=4,
                  physical_rollouts=20, expected_failed_parents=4, expected_successful_parents=1,
                  hypothesis='Lowering onset-head severity may improve actual recovery; edge may differ from center',
                  selection='purposeful development cohort, not held-out', model_weights_changed=False,
                  gate_changed=False, training=False, oracle='all four actual physical continuations; offline only')
    save_json(root / 'config.json', config)
    (root / 'config.sha256').write_text(sha256_file(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), parents=len(config['parents']), methods=METHODS,
                          actual_rollouts=20)), flush=True)


def check_config(root):
    if sha256_file(root / 'config.json') != (root / 'config.sha256').read_text().strip():
        raise RuntimeError('Frozen config changed')
    config = json.loads((root / 'config.json').read_text())
    if config['source_hashes'] != frozen_files() or config['parents'] != parent_sources():
        raise RuntimeError('Frozen source or input changed')
    return config


def collect(root):
    config = check_config(root)
    save_json(root / 'started.json', dict(utc=now()))
    started = time.monotonic()
    attempts, completed, results = 0, 0, []
    worker = None
    try:
        wrapped, loaded = load_isolated()
        save_json(root / 'model-load.json', loaded)
        print(json.dumps(dict(model_loaded=True, load_seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as call_log:
            def infer(parent, candidate, query, kind, observation, noise):
                nonlocal attempts, completed
                event = dict(parent=parent['name'], candidate=candidate, query=query, kind=kind,
                             ordinal=attempts, noise_sha256=sha_array(noise))
                call_log.write(json.dumps(dict(event, event='request')) + '\n')
                call_log.flush()
                attempts += 1
                response, resource = infer_isolated(wrapped, dict(observation, **{
                    'flow/noise': noise, 'routing/capture': True, CAPTURE_KEY: True}))
                if response['collection/routing_mode'] != 'native_only' or not response['gate_probe/exact_dtype_audit']:
                    raise RuntimeError('Unexpected model intervention')
                if response[PROBS_KEY].dtype != np.float16 or response[PROBS_KEY].shape != (8, 10, 11, 32):
                    raise RuntimeError('Invalid native HB capture')
                call_log.write(json.dumps(dict(event, event='response', resource=resource,
                                                actions_sha256=sha_array(response['actions']),
                                                hb_sha256=sha_array(response[PROBS_KEY]))) + '\n')
                call_log.flush()
                completed += 1
                return response, resource

            for parent in config['parents']:
                manifest, arrays = load_episode_trace(Path(parent['source']))
                for key, expected in manifest['policy_identity'].items():
                    if loaded['metadata'].get(key) != expected:
                        raise RuntimeError('Policy identity changed: ' + key)
                with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as archive:
                    reference = archive['hb_router_probs']
                prefix = prefix_monitor(reference, parent['query'], parent['first_alarm'], parent['head'])
                bank = noise_bank(manifest['config']['flow_noise_seed'])
                directory = root / parent['name']
                directory.mkdir()
                pool_responses, pool_costs = None, None
                decisions = None
                for candidate in config['candidate_order']:
                    branch_dir = directory / ('candidate-%d' % candidate)
                    worker = EnvironmentWorker(parent, branch_dir)
                    (branch_dir / 'queries').mkdir()
                    for query in range(parent['query']):
                        assert_source_observation(worker.current, arrays, query)
                        worker.step(arrays['predicted_actions'][query], query)
                    assert_source_observation(worker.current, arrays, parent['query'])
                    if candidate == 0:
                        noises = candidate_noises(parent, bank, parent['step'], 4)
                        pool_responses, pool_costs = [], []
                        for index, noise in enumerate(noises):
                            response, resource = infer(parent, index, parent['query'], 'pool',
                                                       worker.current['observation'], noise)
                            pool_responses.append(response)
                            pool_costs.append(resource)
                        equal(pool_responses[0]['actions'], arrays['predicted_actions'][parent['query']], 'pool native actions')
                        equal(pool_responses[0][PROBS_KEY], reference[parent['query']], 'pool native HB')
                        hb = np.stack([r[PROBS_KEY] for r in pool_responses])
                        decisions = preview_pool(prefix, hb, parent['head'],
                                                 int(seeded_rng(parent, parent['step'], 100).integers(4)))
                        with (directory / 'pool.npz').open('xb') as stream:
                            np.savez_compressed(stream, image=worker.current['observation']['observation/image'],
                                                wrist_image=worker.current['observation']['observation/wrist_image'],
                                                state=worker.current['observation']['observation/state'],
                                                sim_before=worker.current['sim_state'], noises=noises,
                                                actions=np.stack([r['actions'] for r in pool_responses]), hb=hb)
                        save_json(directory / 'decisions.json', dict(decisions, frozen_utc=now(),
                                                                   before_any_suffix_execution=True,
                                                                   pool_sha256=sha256_file(directory / 'pool.npz')))
                        print(json.dumps(jsonable(dict(parent=parent['name'], pool_ready=True,
                                                       selected=decisions['selected'], head=parent['head'],
                                                       target_severity=decisions['target_severity'],
                                                       target_spread=decisions['target_spread']))), flush=True)
                    monitor = copy.deepcopy(prefix)
                    query, continuation_calls = parent['query'], 0
                    rows = []
                    with (branch_dir / 'queries.jsonl').open('x') as query_log:
                        while not worker.current['success'] and worker.current['steps'] < 520:
                            current = worker.current
                            step = current['steps']
                            if step != query * 10:
                                raise RuntimeError('Unexpected action cadence')
                            if candidate == 0:
                                assert_source_observation(current, arrays, query)
                            if query == parent['query']:
                                response, resource = pool_responses[candidate], pool_costs[candidate]
                                path = directory / 'pool.npz'
                                noise = noises[candidate]
                                origin = 'pool'
                            else:
                                noise = bank[query]
                                response, resource = infer(parent, candidate, query, 'continuation', current['observation'], noise)
                                continuation_calls += 1
                                path = branch_dir / 'queries' / ('q%03d.npz' % query)
                                with path.open('xb') as stream:
                                    np.savez_compressed(stream, image=current['observation']['observation/image'],
                                                        wrist_image=current['observation']['observation/wrist_image'],
                                                        state=current['observation']['observation/state'], sim_before=current['sim_state'],
                                                        noise=noise, actions=response['actions'], hb=response[PROBS_KEY])
                                origin = 'continuation'
                            if candidate == 0:
                                equal(response['actions'], arrays['predicted_actions'][query], 'C0 source actions')
                                equal(response[PROBS_KEY], reference[query], 'C0 source HB')
                            status = monitor.update(response[PROBS_KEY])
                            normalized, _, _ = normalize_scores(score_status(status), query)
                            row = jsonable(dict(query=query, step=step, origin=origin, candidate=candidate,
                                                monitor=status, normalized=normalized,
                                                head_severity=head_severity(normalized, parent['head']), resource=resource,
                                                archive=str(path.relative_to(root)), archive_sha256=sha256_file(path)))
                            worker.step(response['actions'], query if candidate == 0 else None)
                            row.update(executed=worker.current['executed'], success=worker.current['success'])
                            query_log.write(json.dumps(row, allow_nan=False) + '\n')
                            query_log.flush()
                            rows.append(row)
                            query += 1
                            if query % 10 == 0 or worker.current['success']:
                                print(json.dumps(dict(parent=parent['name'], candidate=candidate, query=query,
                                                      action_steps=worker.current['steps'], success=worker.current['success'],
                                                      model_calls=completed, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                    environment = worker.close()
                    worker = None
                    if candidate == 0 and (environment['success'] != parent['source_success'] or
                                           environment['steps'] != parent['source_steps'] or query != parent['source_queries']):
                        raise RuntimeError('C0 endpoint did not reproduce')
                    result = dict(parent=parent['name'], head=parent['head'], candidate=candidate,
                                  selected_by=[name for name, value in decisions['selected'].items() if value == candidate],
                                  success=environment['success'], action_steps=environment['steps'], queries=query,
                                  prefix_queries=parent['query'], suffix_queries=len(rows), continuation_calls=continuation_calls,
                                  actual_environment_steps=environment['steps'], settle_steps=environment['settle_steps'],
                                  first_alarm=monitor.first_v82_alarm,
                                  target_at_intervention=float(decisions['target_severity'][candidate]),
                                  rescue=not parent['source_success'] and environment['success'],
                                  harm=parent['source_success'] and not environment['success'], c0_exact=candidate == 0)
                    save_json(branch_dir / 'result.json', result)
                    results.append(result)
                    print(json.dumps(dict(branch_complete=True, **result)), flush=True)
        check_config(root)
        save_json(root / 'collection.json', dict(passed=True, results=results, model_attempts=attempts,
                                                 model_completed=completed, elapsed_seconds=time.monotonic() - started,
                                                 finished_utc=now()))
        print(json.dumps(dict(completed=True, model_calls=completed, actual_rollouts=len(results), run=str(root))), flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(utc=now(), error=str(error), traceback=traceback.format_exc(),
                                             model_attempts=attempts, model_completed=completed, results=results))
        raise
    finally:
        if worker is not None:
            worker.abort()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare()
    elif args.run is None:
        parser.error('--run is required')
    else:
        collect(args.run.resolve())
