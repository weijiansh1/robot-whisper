"""Execute the frozen four-arm gate controller in real LIBERO continuations."""

import copy
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, load_isolated, infer_isolated
from gated_rollout_protocol import (ARMS, LABELS, CHUNK, HORIZON, RECOVERY_QUERIES,
                                    in_window, proposal, evaluate)
from run_response_matrix_experiment import MatrixPolicy, check_config as check_matrix
from run_online_experiment import EnvironmentWorker, EVALUATION, assert_source_observation, jsonable, now
from run_gate_experiment import FULL_KEYS, assert_equal, check_identity
from audit_online_experiment import read, write, digest, array_digest, same
from head_control_protocol import prefix_monitor
from response_matrix_protocol import SHAPE, measure
from online_protocol import noise_bank
from collection_routes import CAPTURE_KEY, PROBS_KEY
from scope_bias_control import BIAS_KEY
from v8_feature_control import EFFECTIVE_PROBS
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace

P3C = BASE / 'runs/p3c-response-matrix-20260915-463ir6kj'
LOGIT_KEYS = ('diagnostic/native_logits_fp32', 'diagnostic/effective_logits_fp32', 'diagnostic/applied_bias_fp32')


def frozen_files():
    result = read(P3C / 'config.json')['source_hashes'].copy()
    for path, expected in result.items():
        if digest(path) != expected:
            raise RuntimeError('Earlier frozen source changed: ' + path)
    for name in ('P3D_GATED_ROLLOUT_PLAN.zh.md', 'gated_rollout_protocol.py',
                 'run_gated_rollout_experiment.py', 'test_gated_rollout_protocol.py'):
        result[str(BASE / name)] = digest(BASE / name)
    return result


def prepare():
    matrix = check_matrix(P3C)
    sealed = read(P3C / 'verification.json')
    assert sealed['passed']
    for path, expected in sealed['files'].items():
        assert digest(P3C / path) == expected, path
    for path, expected in sealed['post_collection_sources'].items():
        assert digest(path) == expected, path
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == matrix['shared_metadata'][key]
    root = Path(tempfile.mkdtemp(prefix='p3d-gated-rollout-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_gated_rollout.v1', utc=now(), parents=matrix['parents'], arms=ARMS,
                  source_matrix=str(P3C), matrix_config_sha256=digest(P3C / 'config.json'),
                  matrix_verification_sha256=digest(P3C / 'verification.json'),
                  source_hashes=frozen_files(), shared_metadata=metadata,
                  chunk=CHUNK, horizon=HORIZON, recovery_queries=RECOVERY_QUERIES,
                  acceptance='frozen P3c instantaneous screen; rejected candidates do not advance history',
                  physical_rollouts=20, maximum_model_calls=731, extra_validation_calls=15,
                  expected_failed_parents=4, expected_successful_parents=1,
                  selection='same five inspected development states; no held-out claim',
                  weights_changed=False, structure_changed=False, actual_gate_intervention=True,
                  simulation_only=True, training=False)
    write(root / 'config.json', jsonable(config))
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), rollouts=20, maximum_model_calls=731)), flush=True)


def check_config(root):
    config = read(root / 'config.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert config['source_hashes'] == frozen_files()
    matrix = check_matrix(P3C)
    assert config['parents'] == matrix['parents']
    assert digest(P3C / 'config.json') == config['matrix_config_sha256']
    assert digest(P3C / 'verification.json') == config['matrix_verification_sha256']
    return config


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    started, attempts, completed, results = time.monotonic(), 0, 0, []
    worker = None
    try:
        wrapped, loaded = load_isolated()
        modified = MatrixPolicy(wrapped.policy)
        write(root / 'model-load.json', loaded)
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], float)
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as log:
            def infer(parent, arm, query, kind, current, noise, directory, bias=None):
                nonlocal attempts, completed
                if attempts >= config['maximum_model_calls']:
                    raise RuntimeError('Frozen model budget exceeded')
                request_bias = np.zeros(SHAPE, np.float32) if bias is None else bias
                event = dict(parent=parent['name'], arm=arm, query=query, kind=kind, ordinal=attempts,
                             noise_sha256=array_digest(noise), bias_sha256=array_digest(request_bias))
                log.write(json.dumps(dict(event, event='request')) + '\n')
                log.flush()
                attempts += 1
                request = dict(current['observation'], **{'flow/noise': noise, 'routing/capture': True, CAPTURE_KEY: True})
                response, resource = infer_isolated(wrapped if bias is None else modified,
                    request if bias is None else dict(request, **{BIAS_KEY: bias}))
                completed += 1
                if not response['gate_probe/exact_dtype_audit']:
                    raise RuntimeError('Missing gate dtype audit')
                if bias is not None and not response['diagnostic/gpu_exact_logit_softmax']:
                    raise RuntimeError('Missing actual logit capture')
                path = directory / ('q%03d-%s.npz' % (query, kind))
                payload = {k: v for k, v in response.items() if isinstance(v, np.ndarray)}
                payload.update({k: v for k, v in current['observation'].items() if isinstance(v, np.ndarray)})
                payload.update(request_noise=noise, request_bias=request_bias, sim_before=current['sim_state'])
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **payload)
                row = dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path), resource=resource,
                           actions_sha256=array_digest(response['actions']),
                           effective_hb_sha256=array_digest(response[EFFECTIVE_PROBS].astype(np.float16)))
                if bias is not None:
                    row.update(gpu_exact_logit_softmax=True, logit_dtypes=response['diagnostic/logit_dtypes'],
                               probability_dtypes=response['diagnostic/probability_dtypes'])
                log.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
                log.flush()
                return response, row

            for parent in config['parents']:
                manifest, arrays = load_episode_trace(Path(parent['source']))
                check_identity(loaded['metadata'], parent['policy_identity'])
                with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as saved:
                    reference = saved['hb_router_probs']
                prefix = prefix_monitor(reference, parent['query'], parent['first_alarm'], parent['head'])
                bank = noise_bank(manifest['config']['flow_noise_seed'])
                for arm in config['arms']:
                    directory = root / parent['name'] / arm
                    branch_started, call_begin = time.monotonic(), completed
                    worker = EnvironmentWorker(parent, directory)
                    query_dir = directory / 'queries'
                    query_dir.mkdir()
                    for query in range(parent['query']):
                        assert_source_observation(worker.current, arrays, query)
                        worker.step(arrays['predicted_actions'][query], query)
                    assert_source_observation(worker.current, arrays, parent['query'])
                    monitor = copy.deepcopy(prefix)
                    previous = reference[parent['query'] - 1]
                    query, opportunities, candidates, accepted = parent['query'], 0, 0, 0
                    rows = []
                    with (directory / 'decisions.jsonl').open('x') as decisions, (directory / 'queries.jsonl').open('x') as queries:
                        while not worker.current['success'] and worker.current['steps'] < HORIZON:
                            current = worker.current
                            if current['steps'] != query * CHUNK:
                                raise RuntimeError('Original ten-step action cadence changed')
                            if arm == 'native':
                                assert_source_observation(current, arrays, query)
                            native, native_call = infer(parent, arm, query, 'native', current, bank[query], query_dir)
                            if query == parent['query']:
                                with np.load(P3C / 'states' / parent['name'] / 'native.npz', allow_pickle=False) as old:
                                    assert_equal(native, old, FULL_KEYS, 'same P3c first shadow')
                                if arm == 'native':
                                    zero, _ = infer(parent, arm, query, 'zero', current, bank[query], query_dir,
                                                    np.zeros(SHAPE, np.float32))
                                    assert_equal(zero, native, FULL_KEYS, 'zero intervention')
                                    post, _ = infer(parent, arm, query, 'post', current, bank[query], query_dir)
                                    assert_equal(post, native, FULL_KEYS, 'request cleanup')
                            if arm == 'native':
                                same(native['actions'], arrays['predicted_actions'][query], 'C0 source action')
                                same(native[PROBS_KEY], reference[query], 'C0 source routing')
                            chosen, chosen_call = native, native_call
                            decision = dict(accepted=False, reasons=['outside_window'], operator=None)
                            candidate_call, generation = None, None
                            if in_window(arm, query, parent['query']):
                                opportunities += 1
                                bias, generation = proposal(native[EFFECTIVE_PROBS].astype(np.float16), previous, parent, query, arm)
                                if bias is None:
                                    decision = dict(accepted=False, reasons=['degenerate_direction'], operator=None)
                                else:
                                    candidate, candidate_call = infer(parent, arm, query, 'candidate', current,
                                                                      bank[query], query_dir, bias)
                                    candidates += 1
                                    if query == parent['query']:
                                        with np.load(P3C / 'states' / parent['name'] / (LABELS[arm] + '.npz'), allow_pickle=False) as old:
                                            same(bias, old['request_bias'], 'same P3c initial bias')
                                            assert_equal(candidate, old, FULL_KEYS + LOGIT_KEYS, 'same P3c initial probe')
                                        if arm == 'mobility_balance':
                                            repeated, _ = infer(parent, arm, query, 'repeat', current, bank[query], query_dir, bias)
                                            assert_equal(repeated, candidate, FULL_KEYS + LOGIT_KEYS, 'nonzero repeat')
                                    decision = evaluate(monitor, native, candidate, arm, std)
                                    if decision['accepted']:
                                        chosen, chosen_call = candidate, candidate_call
                                        accepted += 1
                            selected = measure(monitor, chosen[EFFECTIVE_PROBS].astype(np.float16))
                            preview = jsonable(dict(parent=parent['name'], arm=arm, query=query, step=current['steps'],
                                native_path=native_call['path'], candidate_path=None if candidate_call is None else candidate_call['path'],
                                selected_path=chosen_call['path'], generation=generation, decision=decision, selected=selected,
                                utc=now(), frozen_before_environment_step=True))
                            decisions.write(json.dumps(preview, allow_nan=False) + '\n')
                            decisions.flush()
                            previous = chosen[EFFECTIVE_PROBS].astype(np.float16)
                            status = monitor.update(previous)
                            if jsonable(status) != jsonable(selected['status']):
                                raise RuntimeError('Selected monitor preview differs from committed history')
                            worker.step(chosen['actions'], query if arm == 'native' else None)
                            row = dict(preview, executed=worker.current['executed'], success=worker.current['success'])
                            queries.write(json.dumps(row, allow_nan=False) + '\n')
                            queries.flush()
                            rows.append(row)
                            query += 1
                            if query % 5 == 0 or worker.current['success']:
                                print(json.dumps(dict(parent=parent['name'], arm=arm, query=query, steps=worker.current['steps'],
                                    success=worker.current['success'], accepted=accepted, candidates=candidates,
                                    calls=completed, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                    environment = worker.close()
                    worker = None
                    if arm == 'native' and (environment['success'] != parent['source_success'] or
                                             environment['steps'] != parent['source_steps'] or query != parent['source_queries']):
                        raise RuntimeError('Native endpoint did not reproduce')
                    result = dict(parent=parent['name'], head=parent['head'], arm=arm, success=environment['success'],
                                  action_steps=environment['steps'], settle_steps=environment['settle_steps'], queries=query,
                                  prefix_queries=parent['query'], suffix_queries=len(rows), opportunities=opportunities,
                                  candidates=candidates, accepted=accepted, rejected=opportunities - accepted,
                                  model_calls=completed - call_begin, deployment_suffix_calls=len(rows) + candidates,
                                  deployment_full_calls=parent['query'] + len(rows) + candidates,
                                  elapsed_seconds=time.monotonic() - branch_started, first_alarm=monitor.first_v82_alarm,
                                  rescue=not parent['source_success'] and environment['success'],
                                  harm=parent['source_success'] and not environment['success'], c0_exact=arm == 'native')
                    write(directory / 'result.json', result)
                    results.append(result)
                    print(json.dumps(dict(branch_complete=True, **result)), flush=True)
        check_config(root)
        assert len(results) == config['physical_rollouts'] and attempts == completed <= config['maximum_model_calls']
        write(root / 'collection.json', dict(passed=True, results=results, model_attempts=attempts, model_completed=completed,
            actual_rollouts=len(results), environment_action_steps=sum(r['action_steps'] for r in results),
            settle_steps=sum(r['settle_steps'] for r in results), elapsed_seconds=time.monotonic() - started, finished_utc=now()))
        print(json.dumps(dict(completed=True, calls=completed, rollouts=len(results), run=str(root))), flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(utc=now(), error=str(error), traceback=traceback.format_exc(),
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
