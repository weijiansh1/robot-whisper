"""Exact archived-request replay collecting only HB MoE port tensors."""

import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from audit_replan_window import read, records
from crossover_capture import parameter_digest, parameter_versions
from gate_runtime import BASE, infer_isolated, load_isolated
from moe_compute_capture import FIELDS, moe_infer
from moe_compute_protocol import SEED, noise_grid
from run_online_experiment import EVALUATION, equal, jsonable, now, save_json, sha_array
from run_replan_window import shared_metadata
from run_topology_experiment import P3H, P3F, check_hashes
from topology_protocol import DEPS, gudhi_module
from collection_routes import CAPTURE_KEY, PROBS_KEY
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file

P3I = BASE / 'runs/p3i-topology-20260915-ir7fgyvx'
SOURCES = ('P3J_MOE_COMPUTE_PLAN.zh.md', 'moe_compute_capture.py', 'moe_compute_protocol.py',
           'test_moe_compute_protocol.py', 'run_moe_compute_experiment.py')


def prepare():
    old, seal = read(P3I / 'config.json'), read(P3I / 'verification.json')
    assert seal['passed'] and read(P3H / 'failure-only-audit.json')['passed']
    protected = {}
    for path in sorted((BASE / 'runs').glob('*/verification.json')):
        verification = read(path)
        protected[str(path)] = sha256_file(path)
        protected.update({str(path.parent / name): sha for name, sha in
                          verification.get('files', verification.get('file_sha256')).items()})
    check_hashes(protected)
    sources = dict(old['source_hashes'], **seal['post_collection_sources'])
    check_hashes(sources)
    sources.update({str(BASE / name): sha256_file(BASE / name) for name in SOURCES})
    inputs = old['input_hashes'].copy()
    parents = old['parents']
    for parent in parents:
        source = Path(parent['source'])
        inputs[str(source / 'episode-trace.json')] = parent['manifest_sha256']
        inputs[str(source / 'episode-trace.npz')] = parent['trace_sha256']
    check_hashes(inputs)
    suffix = sum(len(records(P3H / p['name'] / ('r%d' % r) / arm / 'queries.jsonl'))
                 for p in parents for r in range(4) for arm in ('native10', 'window5'))
    prefix = sum(p['query'] for p in parents)
    assert len(parents) == 6 and prefix == 131 and suffix == 1496
    metadata = shared_metadata()
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == old['shared_metadata'][key]
    assert shutil.disk_usage(BASE).free >= 20 * 1024 ** 3
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert tests.returncode == 0, tests.stdout
    root = Path(tempfile.mkdtemp(prefix='p3j-moe-compute-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_compute_topology.v1', utc=now(), parents=parents,
                  p3h=str(P3H), p3i=str(P3I), p3f=str(P3F), source_hashes=sources,
                  protected_files=protected, input_hashes=inputs, dependency_hashes=old['dependency_hashes'],
                  shared_metadata=metadata, expected_model_calls=1699, prefix_calls=131,
                  suffix_calls=1496, archived_duplicate_calls=6, control_calls=66,
                  archived_failed_rollouts=48, groups=6, tasks=3,
                  new_environment_actions=0, new_success_parent_rollouts=0, shared_inference_requests=0,
                  seed=SEED, fields=FIELDS, parameter_reference_sha256=read(
                      BASE / 'runs/p3g-crossover-rollout-20260915-1gmt0y3o/collection.json')['parameter_sha256_after'],
                  analysis_features='HB MoE routes and input/total only; shared for noise controls',
                  analysis_target='future 40 physical steps HB layer15 last-flow effective total displacement',
                  original_model_training=False, new_controller_test=False)
    save_json(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(sha256_file(root / 'config.json') + '\n')
    with (root / 'pre-collection-tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    print(json.dumps(dict(run=str(root), calls=config['expected_model_calls'])), flush=True)


def check_config(root, protected=False):
    config = read(root / 'config.json')
    assert sha256_file(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for key in ('source_hashes', 'input_hashes', 'dependency_hashes'):
        check_hashes(config[key])
    if protected:
        check_hashes(config['protected_files'])
    assert gudhi_module().__version__ == '3.13.0'
    return config


def saved_request(path, prompt, noise=None):
    with np.load(path) as saved:
        expected = {key: saved[key] for key in saved.files if not
                    (key.startswith('observation/') or key in ('request_noise', 'sim_before'))}
        request = {key: saved[key] for key in saved.files if key.startswith('observation/')}
        request.update(prompt=prompt, **{'flow/noise': saved['request_noise'] if noise is None else noise,
                                        'routing/capture': True, CAPTURE_KEY: True})
    return request, expected


def compare_response(response, expected):
    for key, value in expected.items():
        equal(response[key], value, 'Archived replay ' + key)


def collect(root):
    import torch
    config = check_config(root)
    save_json(root / 'started.json', dict(utc=now(), mode='MoE-only read-only replay'))
    attempts = completed = 0
    started = time.monotonic()
    log = None
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        wrapped, loaded = load_isolated()
        model = wrapped.policy._policy.model
        versions, digest_before = parameter_versions(model), parameter_digest(model)
        assert digest_before == config['parameter_reference_sha256']
        save_json(root / 'model-load.json', dict(loaded, parameter_sha256_before=digest_before))
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with np.load(P3H / 'model-rng.npz') as saved:
            rng_cpu, rng_cuda = (torch.from_numpy(saved[key].copy()) for key in ('cpu', 'cuda'))

        def reset_rng():
            torch.set_rng_state(rng_cpu)
            torch.cuda.set_rng_state(rng_cuda)

        def call(parent, phase, path, request, expected=None, capture=True, **details):
            nonlocal attempts, completed
            assert attempts < config['expected_model_calls']
            assert shutil.disk_usage(root).free > 3 * 1024 ** 3
            event = dict(ordinal=attempts, parent=parent['name'], phase=phase, path=str(path.relative_to(root)),
                         noise_sha256=sha_array(request['flow/noise']), **details)
            log.write(json.dumps(dict(event, event='request')) + '\n')
            log.flush()
            attempts += 1
            before = time.monotonic()
            response, resource = (moe_infer if capture else infer_isolated)(wrapped, request)
            completed += 1
            assert parameter_versions(model) == versions
            assert response['collection/routing_mode'] == 'native_only' and response['gate_probe/exact_dtype_audit']
            assert np.array_equal(response['collection/hb_native_ids'], response['collection/hb_effective_ids'])
            assert np.array_equal(response['collection/hb_native_weights'], response['collection/hb_effective_weights'])
            if expected is not None:
                compare_response(response, expected)
            payload = {k: v for k, v in response.items() if isinstance(v, np.ndarray) and
                       k.startswith(('mechanism/', 'collection/', 'routing/', 'v8_control/'))}
            if capture:
                assert {k for k in payload if k.startswith('mechanism/')} == {'mechanism/' + f for f in FIELDS}
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                np.savez_compressed(stream, **payload)
            row = jsonable(dict(event, event='response', sha256=sha256_file(path), resource=resource,
                                actions_sha256=sha_array(response['actions']), hb_sha256=sha_array(response[PROBS_KEY]),
                                replay_exact=expected is not None, port_capture=capture,
                                parameter_versions_unchanged=True, wall_seconds=time.monotonic() - before,
                                file_bytes=path.stat().st_size))
            log.write(json.dumps(row, allow_nan=False) + '\n')
            log.flush()
            if completed % 20 == 0 or phase == 'control':
                print(json.dumps(dict(calls=completed, total=config['expected_model_calls'], parent=parent['name'],
                                      phase=phase, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
            return response

        with (root / 'calls.jsonl').open('x') as log:
            for number, parent in enumerate(config['parents']):
                manifest = read(Path(parent['source']) / 'episode-trace.json')
                for key, value in parent['policy_identity'].items():
                    assert loaded['metadata'][key] == value, key
                source_dir = P3H / parent['name'] / 'r0/native10/queries'
                request, expected = saved_request(source_dir / ('s%03d.npz' % parent['start']), manifest['prompt'])
                native = None
                for label, enabled in (('bare', False), ('capture', True), ('repeat', True)):
                    reset_rng()
                    response = call(parent, 'control', root / 'controls' / parent['name'] / (label + '.npz'),
                                    request, expected, enabled, label=label)
                    if label == 'capture':
                        native = response
                    if label == 'repeat':
                        compare_response(response, {k: v for k, v in native.items() if isinstance(v, np.ndarray)})
                grid = noise_grid(number, request['flow/noise'])
                for observation in range(3):
                    for noise in range(3):
                        if observation == noise == 0:
                            continue
                        source = source_dir / ('s%03d.npz' % (parent['start'] + observation * 20))
                        candidate, _ = saved_request(source, manifest['prompt'], grid[noise])
                        reset_rng()
                        call(parent, 'control', root / 'controls' / parent['name'] / ('o%d-n%d.npz' % (observation, noise)),
                             candidate, observation=observation, noise=noise, source=str(source))
                del native, response
            assert completed == config['control_calls']
            save_json(root / 'preflight.json', dict(passed=True, calls=completed, parents=6,
                capture_on_off_and_repeat_exact=True, archived_fork_actions_and_routes_exact=True,
                new_environment_actions=0, utc=now()))
            print('MoE port preflight passed; collecting continuous archived requests', flush=True)

            for parent in config['parents']:
                manifest, arrays = load_episode_trace(Path(parent['source']))
                with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as saved:
                    original_routes = saved['hb_router_probs']
                reset_rng()
                for query in range(parent['query']):
                    request = {key: arrays[source][query] for key, source in (
                        ('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'), ('observation/state', 'states'))}
                    request.update(prompt=manifest['prompt'], **{'flow/noise': arrays['flow_noises'][query],
                                                               'routing/capture': True, CAPTURE_KEY: True})
                    expected = dict(actions=arrays['predicted_actions'][query], **{PROBS_KEY: original_routes[query]})
                    call(parent, 'prefix', root / 'prefixes' / parent['name'] / ('q%03d.npz' % query), request, expected, query=query)
                for replicate in range(4):
                    for arm in ('native10', 'window5'):
                        source_dir = P3H / parent['name'] / ('r%d' % replicate) / arm
                        reset_rng()
                        first = None
                        for row in records(source_dir / 'queries.jsonl'):
                            request, expected = saved_request(P3H / row['path'], manifest['prompt'])
                            response = call(parent, 'suffix', root / 'branches' / parent['name'] / ('r%d' % replicate) /
                                            arm / ('s%03d.npz' % row['step']), request, expected,
                                            replicate=replicate, arm=arm, step=row['step'], source=row['path'])
                            if row['step'] == parent['start'] and replicate == 0 and arm == 'native10':
                                first = {k: v for k, v in response.items() if k.startswith('mechanism/')}
                                repeat_path = P3H / Path(row['path']).with_name('s%03d-repeat.npz' % row['step'])
                                repeated_request, repeated_expected = saved_request(repeat_path, manifest['prompt'])
                                repeated = call(parent, 'source_duplicate', root / 'branches' / parent['name'] / 'r0' / arm /
                                                ('s%03d-repeat.npz' % row['step']), repeated_request, repeated_expected,
                                                replicate=replicate, arm=arm, step=row['step'], source=str(repeat_path))
                                compare_response(repeated, first)
                                del first, repeated
                        del response
                        print(json.dumps(dict(branch_complete=True, parent=parent['name'], replicate=replicate,
                                              arm=arm, calls=completed)), flush=True)
            assert attempts == completed == config['expected_model_calls']
        digest_after = parameter_digest(model)
        assert digest_before == digest_after and parameter_versions(model) == versions
        check_config(root)
        service = shared_metadata()
        for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
            assert service[key] == config['shared_metadata'][key]
        responses = [row for row in records(root / 'calls.jsonl') if row['event'] == 'response']
        save_json(root / 'collection.json', dict(passed=True, calls=completed, attempts=attempts,
                  elapsed_seconds=time.monotonic() - started, finished_utc=now(),
                  parameter_sha256_before=digest_before, parameter_sha256_after=digest_after,
                  parameter_versions_unchanged=True, model_inference_seconds=sum(r['resource']['seconds'] for r in responses),
                  bytes_saved=sum(r['file_bytes'] for r in responses),
                  peak_allocated_mib=max(r['resource']['peak_allocated_mib'] for r in responses),
                  new_environment_actions=0, new_success_parent_rollouts=0, shared_metadata_after=service))
        print(json.dumps(dict(collection_complete=True, calls=completed, parameter_sha256=digest_after)), flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(), attempts=attempts,
                                            completed=completed, elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else collect(args.run.resolve())
