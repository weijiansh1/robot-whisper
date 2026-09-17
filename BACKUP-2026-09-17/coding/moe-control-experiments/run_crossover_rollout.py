"""Execute four MoE crossover policies in fresh paired LIBERO continuations."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, load_isolated, infer_isolated
from crossover_rollout_protocol import ARMS, CHUNK, HORIZON, WINDOW, active, proposal, guard, committed_route, maximum_calls
from crossover_capture import parameter_versions, parameter_digest
from crossover_capture_v2 import whole_output_infer, OUTPUT_MASK
from run_crossover_experiment import TRACE_KEYS
from run_response_matrix_experiment import MatrixPolicy
from run_gated_rollout_experiment import P3C, LOGIT_KEYS
from run_online_experiment import EnvironmentWorker, EVALUATION, assert_source_observation, jsonable, now
from run_gate_experiment import FULL_KEYS, check_identity
from audit_online_experiment import read, write, digest, array_digest, same
from head_control_protocol import prefix_monitor
from response_matrix_protocol import SHAPE, measure
from gated_rollout_protocol import evaluate
from online_protocol import noise_bank
from collection_routes import CAPTURE_KEY, PROBS_KEY
from scope_bias_control import BIAS_KEY
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace

P3D = BASE / 'runs/p3d-gated-rollout-20260915-t9k1v766'
P3F = BASE / 'runs/p3f-crossover-v2-20260915-ihpn5n13'
NEW_SOURCES = ('P3G_CROSSOVER_ROLLOUT_PLAN.zh.md', 'crossover_rollout_protocol.py',
               'test_crossover_rollout.py', 'run_crossover_rollout.py')


def exact(actual, expected, keys, label):
    for key in keys:
        same(actual[key], expected[key], label + ': ' + key)


def prepare():
    old, previous = read(P3D / 'config.json'), read(P3F / 'config.json')
    for source in (P3D, P3F):
        for path, expected in read(source / 'verification.json')['files'].items():
            assert digest(source / path) == expected, path
    sources = dict(previous['source_hashes'], **read(P3F / 'verification.json')['post_collection_sources'])
    for path, expected in sources.items():
        assert digest(path) == expected, path
    sources.update({str(BASE / name): digest(BASE / name) for name in NEW_SOURCES})
    assert shutil.disk_usage(BASE).free >= 20 * 1024 ** 3
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == previous['shared_metadata'][key] == old['shared_metadata'][key]
    root = Path(tempfile.mkdtemp(prefix='p3g-crossover-rollout-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_crossover_rollout.v1', utc=now(), parents=old['parents'], arms=ARMS,
        source_hashes=sources, shared_metadata=metadata, chunk=CHUNK, horizon=HORIZON, recovery_queries=WINDOW,
        physical_rollouts=20, maximum_model_calls=maximum_calls(old['parents']), extra_validation_calls=40,
        references={str(path): digest(path / 'verification.json') for path in (P3C, P3D, P3F)},
        prior_pilot=previous['prior_pilot'], prior_pilot_files=previous['prior_pilot_files'],
        acceptance='joint donor action guard only; old routing score screen is diagnostic',
        committed_history='actual selected effective probabilities, including output-only feedback',
        new_environment_actions=True, simulation_only=True, physical_features_used_for_selection=False,
        training=False, weights_changed=False, persistent_architecture_changed=False,
        cohort='same five development parents: four original failures and one original success',
        gate_scope=dict(layers=[12, 13, 14, 15], tokens=list(range(1, 11)), denoise_steps=10, max_abs=.30),
        output_scope=dict(layers=[12, 13, 14, 15], tokens=list(range(11)), denoise_steps=10))
    assert config['maximum_model_calls'] == 876
    write(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), rollouts=20, maximum_model_calls=876)), flush=True)


def check_config(root):
    config = read(root / 'config.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    for path, expected in config['references'].items():
        assert digest(Path(path) / 'verification.json') == expected, path
    for path, expected in config['prior_pilot_files'].items():
        assert digest(Path(config['prior_pilot']) / path) == expected, path
    assert config['parents'] == read(P3D / 'config.json')['parents']
    assert config['arms'] == list(ARMS) and config['maximum_model_calls'] == maximum_calls(config['parents'])
    for parent in config['parents']:
        assert parent['step'] == parent['query'] * CHUNK
        for name, field in (('episode-trace.json', 'manifest_sha256'), ('episode-trace.npz', 'trace_sha256')):
            assert digest(Path(parent['source']) / name) == parent[field]
        assert digest(EVALUATION / parent['name'] / 'full-hb-routes.npz') == parent['hb_sha256']
    return config


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    started, attempts, completed, results = time.monotonic(), 0, 0, []
    worker = None
    try:
        wrapped, loaded = load_isolated()
        modified, model = MatrixPolicy(wrapped.policy), wrapped.policy._policy.model
        versions, weights_before = parameter_versions(model), parameter_digest(model)
        write(root / 'model-load.json', dict(loaded, parameter_sha256=weights_before))
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], float)
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as log:
            def infer(parent, arm, query, kind, current, noise, directory, bias=None, donor=None, donor_call=None, traced=False):
                nonlocal attempts, completed
                if attempts >= config['maximum_model_calls']:
                    raise RuntimeError('Frozen model call budget exceeded')
                if shutil.disk_usage(root).free < 5 * 1024 ** 3:
                    raise RuntimeError('Disk reserve reached')
                request_bias = np.zeros(SHAPE, np.float32) if bias is None else bias
                event = dict(ordinal=attempts, parent=parent['name'], arm=arm, query=query, kind=kind,
                    biased=bias is not None, traced=traced, noise_sha256=array_digest(noise), bias_sha256=array_digest(request_bias),
                    donor_path=None if donor_call is None else donor_call['path'],
                    donor_sha256=None if donor is None else array_digest(donor))
                log.write(json.dumps(dict(event, event='request')) + '\n')
                log.flush()
                attempts += 1
                request = dict(current['observation'], **{'flow/noise': noise, 'routing/capture': True, CAPTURE_KEY: True})
                policy = wrapped if bias is None else modified
                if bias is not None:
                    request[BIAS_KEY] = bias
                if traced:
                    response, resource = whole_output_infer(policy, request, donor)
                else:
                    assert donor is None and bias is None
                    response, resource = infer_isolated(policy, request)
                completed += 1
                path = directory / ('q%03d-%s.npz' % (query, kind))
                payload = {k: v for k, v in response.items() if isinstance(v, np.ndarray)}
                payload.update({k: v for k, v in current['observation'].items() if isinstance(v, np.ndarray)})
                payload.update(request_noise=noise, request_bias=request_bias, sim_before=current['sim_state'])
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **payload)
                assert response['gate_probe/exact_dtype_audit'] and parameter_versions(model) == versions
                row = dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path), resource=resource,
                    actions_sha256=array_digest(response['actions']), effective_hb_sha256=array_digest(committed_route(response)),
                    parameter_versions_unchanged=True, hooks_removed=True)
                if bias is not None:
                    assert response['diagnostic/gpu_exact_logit_softmax']
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
                noises = noise_bank(manifest['config']['flow_noise_seed'])
                for arm in config['arms']:
                    print(json.dumps(dict(branch_start=parent['name'], arm=arm, calls=completed)), flush=True)
                    directory = root / parent['name'] / arm
                    branch_started, call_begin = time.monotonic(), completed
                    worker = EnvironmentWorker(parent, directory)
                    query_dir = directory / 'queries'
                    query_dir.mkdir()
                    for query in range(parent['query']):
                        assert_source_observation(worker.current, arrays, query)
                        worker.step(arrays['predicted_actions'][query], query)
                    assert_source_observation(worker.current, arrays, parent['query'])
                    monitor, previous = copy.deepcopy(prefix), reference[parent['query'] - 1]
                    query, opportunities, proposals, accepted, cross_calls = parent['query'], 0, 0, 0, 0
                    records = []
                    with (directory / 'decisions.jsonl').open('x') as decision_log, (directory / 'queries.jsonl').open('x') as query_log:
                        while not worker.current['success'] and worker.current['steps'] < HORIZON:
                            current = worker.current
                            assert current['steps'] == query * CHUNK
                            is_active = active(arm, query, parent['query'])
                            traced = is_active or query == parent['query']
                            if arm in ('native', 'route_only'):
                                assert_source_observation(current, arrays, query)
                            native, native_call = infer(parent, arm, query, 'native', current, noises[query], query_dir, traced=traced)
                            if query == parent['query']:
                                with np.load(P3C / 'states' / parent['name'] / 'native.npz', allow_pickle=False) as saved:
                                    exact(native, saved, FULL_KEYS, 'P3c initial native')
                            if arm in ('native', 'route_only'):
                                same(native['actions'], arrays['predicted_actions'][query], 'exact C0 policy action')
                                same(native[PROBS_KEY], reference[query], 'exact C0 shadow route')
                            chosen, chosen_call = native, native_call
                            joint = trial = joint_call = trial_call = generation = safety = diagnostic = donor = None
                            bias = None
                            if is_active:
                                opportunities += 1
                                bias, generation = proposal(native, previous, parent, query, arm)
                                if bias is not None:
                                    joint, joint_call = infer(parent, arm, query, 'joint', current, noises[query], query_dir, bias=bias, traced=True)
                                    proposals += 1
                                    if query == parent['query']:
                                        with np.load(P3C / 'states' / parent['name'] / 'mobility_balance-high-plus.npz', allow_pickle=False) as saved:
                                            exact(joint, saved, FULL_KEYS + LOGIT_KEYS, 'P3c initial joint')
                                            same(bias, saved['request_bias'], 'P3c initial bias')
                                    trial, trial_call = joint, joint_call
                                    if arm in ('route_only', 'output_only'):
                                        source, source_call = (native, native_call) if arm == 'route_only' else (joint, joint_call)
                                        donor = source['mechanism/total']
                                        trial, trial_call = infer(parent, arm, query, 'cross', current, noises[query], query_dir,
                                            bias=bias if arm == 'route_only' else None, donor=donor, donor_call=source_call, traced=True)
                                        cross_calls += 1
                                        exact(trial, source, TRACE_KEYS + ('actions',), 'cross effective computation')
                                        same(trial['mechanism/total'][OUTPUT_MASK], donor[OUTPUT_MASK], 'cross source injection')
                                        same(trial['crossover/raw_total'][~OUTPUT_MASK], trial['mechanism/total'][~OUTPUT_MASK], 'cross off-scope')
                                        if arm == 'route_only':
                                            same(trial[NATIVE_PROBS], native[NATIVE_PROBS], 'route-only input history')
                                        else:
                                            same(trial[EFFECTIVE_PROBS], joint[NATIVE_PROBS], 'unbiased gate on joint hidden input')
                                    safety = guard(native, joint, std)
                                    diagnostic = evaluate(monitor, native, trial, 'mobility_balance', std)
                                    if safety['accepted']:
                                        chosen, chosen_call = trial, trial_call
                                        accepted += 1
                            if query == parent['query']:
                                if arm == 'native':
                                    control, _ = infer(parent, arm, query, 'zero', current, noises[query], query_dir,
                                                       bias=np.zeros(SHAPE, np.float32), traced=True)
                                    exact(control, native, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',), 'zero identity')
                                else:
                                    assert joint is not None and trial is not None
                                    source_call = native_call if arm == 'route_only' else joint_call if arm == 'output_only' else None
                                    control, _ = infer(parent, arm, query, 'repeat', current, noises[query], query_dir,
                                        bias=bias if arm in ('joint', 'route_only') else None,
                                        donor=donor, donor_call=source_call, traced=True)
                                    exact(control, trial, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',) +
                                          (LOGIT_KEYS if arm != 'output_only' else ()), 'candidate repeat')
                                control, _ = infer(parent, arm, query, 'post', current, noises[query], query_dir, traced=True)
                                exact(control, native, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',), 'clean post')
                            selected = measure(monitor, committed_route(chosen))
                            preview = jsonable(dict(parent=parent['name'], arm=arm, query=query, step=current['steps'], active=is_active,
                                native_path=native_call['path'], joint_path=None if joint_call is None else joint_call['path'],
                                trial_path=None if trial_call is None else trial_call['path'], selected_path=chosen_call['path'],
                                generation=generation, guard=safety, diagnostic=diagnostic, selected=selected,
                                previous_route_sha256=array_digest(previous), intervention_accepted=chosen_call is not native_call,
                                utc=now(), frozen_before_environment_step=True))
                            decision_log.write(json.dumps(preview, allow_nan=False) + '\n')
                            decision_log.flush()
                            previous = committed_route(chosen)
                            assert jsonable(monitor.update(previous)) == jsonable(selected['status'])
                            worker.step(chosen['actions'], query if arm in ('native', 'route_only') else None)
                            row = dict(preview, executed=worker.current['executed'], success=worker.current['success'])
                            query_log.write(json.dumps(row, allow_nan=False) + '\n')
                            query_log.flush()
                            records.append(row)
                            query += 1
                            if query % 5 == 0 or worker.current['success']:
                                print(json.dumps(dict(parent=parent['name'], arm=arm, query=query, steps=worker.current['steps'],
                                    accepted=accepted, opportunities=opportunities, calls=completed,
                                    elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                    environment = worker.close()
                    worker = None
                    if arm in ('native', 'route_only'):
                        assert environment['success'] == parent['source_success'] and environment['steps'] == parent['source_steps']
                    result = dict(parent=parent['name'], arm=arm, head=parent['head'], success=environment['success'],
                        action_steps=environment['steps'], settle_steps=environment['settle_steps'], queries=query,
                        suffix_queries=len(records), opportunities=opportunities, proposals=proposals, cross_calls=cross_calls,
                        accepted=accepted, model_calls=completed - call_begin,
                        deployment_suffix_calls=len(records) + proposals + cross_calls,
                        deployment_full_calls=parent['query'] + len(records) + proposals + cross_calls,
                        elapsed_seconds=time.monotonic() - branch_started,
                        rescue=not parent['source_success'] and environment['success'], harm=parent['source_success'] and not environment['success'],
                        first_alarm=monitor.first_v82_alarm, c0_exact=arm in ('native', 'route_only'))
                    write(directory / 'result.json', result)
                    results.append(result)
                    print(json.dumps(dict(branch_complete=True, **result)), flush=True)
                    del native, joint, trial, chosen, control, donor
        assert len(results) == 20 and attempts == completed <= config['maximum_model_calls']
        weights_after = parameter_digest(model)
        assert weights_after == weights_before and parameter_versions(model) == versions
        check_config(root)
        write(root / 'collection.json', dict(passed=True, results=results, model_attempts=attempts, model_completed=completed,
            actual_rollouts=20, environment_action_steps=sum(r['action_steps'] for r in results),
            settle_steps=sum(r['settle_steps'] for r in results),
            parameter_sha256_before=weights_before, parameter_sha256_after=weights_after, parameter_versions_unchanged=True,
            elapsed_seconds=time.monotonic() - started, finished_utc=now()))
        print(json.dumps(dict(completed=True, calls=completed, rollouts=20)), flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(), model_attempts=attempts,
            model_completed=completed, completed_branches=results, elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        if worker is not None:
            worker.abort()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else collect(args.run.resolve())
