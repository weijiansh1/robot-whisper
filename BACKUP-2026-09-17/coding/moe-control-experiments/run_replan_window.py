"""Paired physical continuations for a frozen short execution-window test."""

import argparse
import csv
import json
from multiprocessing.connection import Connection
from pathlib import Path
import random
import shutil
import socket
import subprocess
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, infer_isolated, load_isolated
from crossover_capture import parameter_digest, parameter_versions
from run_online_experiment import (EVALUATION, SIM_PYTHON, EnvironmentWorker, assert_source_observation,
                                   equal, jsonable, now, save_json, sha_array)
from replan_window_protocol import (ANCHORS, ARMS, HORIZON, REPLICATES, SEED, TIMING_PARENTS,
                                    ExecutionWindow, group_key, maximum_calls, physical_noise)
from collection_routes import (CAPTURE_KEY, PROBS_KEY, NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY,
                               NATIVE_WEIGHTS_KEY, EFFECTIVE_WEIGHTS_KEY)
from himoe_libero_bridge.client import PolicyClient
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from openpi_client import msgpack_numpy as codec
from v82_closed_loop import V82Monitor

P3G = BASE / 'runs/p3g-crossover-rollout-20260915-1gmt0y3o'
SOURCES = ('P3H_REPLAN_WINDOW_PLAN.zh.md', 'replan_window_protocol.py', 'replan_env_worker.py',
           'test_replan_window.py', 'run_replan_window.py', 'audit_replan_window.py')


class ReplanWorker(EnvironmentWorker):
    def __init__(self, parent, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.log = (directory / 'worker.log').open('x')
        parent_socket, child_socket = socket.socketpair()
        self.connection = Connection(parent_socket.detach())
        self.process = subprocess.Popen(
            [SIM_PYTHON, str(BASE / 'replan_env_worker.py'), '--fd', str(child_socket.fileno()),
             '--source', parent['source'], '--benchmark', parent['benchmark'], '--out', str(directory)],
            pass_fds=(child_socket.fileno(),), stdout=self.log, stderr=subprocess.STDOUT)
        child_socket.close()
        self.closed = False
        try:
            self.current = self.receive()
            assert self.current['event'] == 'ready'
        except BaseException:
            self.abort()
            raise

    def step(self, actions, query=None, compare_start=None):
        self.connection.send_bytes(codec.packb(dict(op='step', actions=actions, compare_query=query,
                                                     compare_start=compare_start)))
        self.current = self.receive()
        assert self.current['event'] == 'stepped'
        return self.current

    def snapshot(self):
        self.connection.send_bytes(codec.packb(dict(op='snapshot')))
        result = self.receive()
        assert result['event'] == 'snapshot'
        return result['state_audit']


def parents():
    with (EVALUATION / 'episodes.csv').open() as stream:
        indexed = {r['name']: r for r in csv.DictReader(stream)}
    result = []
    for name, query, stratum in ANCHORS:
        row = indexed[name]
        source = Path(row['source_artifact_dir'])
        manifest, arrays = load_episode_trace(source)
        assert int(arrays['replan_start_steps'][query]) == query * 10
        result.append(dict(name=name, query=query, start=query * 10, stratum=stratum, source=str(source),
                           benchmark=row['benchmark'], base_task_id=int(row['base_task_id']),
                           init_state_id=int(row['init_state_id']), source_success=row['success'] == 'True',
                           source_steps=int(row['action_steps']), first_alarm=int(row['first_v82_query']),
                           alarm_heads=row['alarm_heads'], policy_identity=manifest['policy_identity'],
                           flow_seed=manifest['config']['flow_noise_seed'], environment_seed=manifest['config']['seed'],
                           manifest_sha256=sha256_file(source / 'episode-trace.json'),
                           trace_sha256=sha256_file(source / 'episode-trace.npz'),
                           hb_sha256=sha256_file(EVALUATION / name / 'full-hb-routes.npz')))
    return result


def preflight():
    root = Path(tempfile.mkdtemp(prefix='p3h-env-preflight-', dir=BASE / 'runs'))
    reports = []
    worker = None
    try:
        for parent in (parents()[0], parents()[4]):
            _, arrays = load_episode_trace(Path(parent['source']))
            fork, end = None, None
            for mode in ('ten', 'five_plus_five'):
                worker = ReplanWorker(parent, root / parent['name'] / mode)
                for q in range(parent['query']):
                    worker.step(arrays['predicted_actions'][q], q)
                assert_source_observation(worker.current, arrays, parent['query'])
                snapshot = worker.snapshot()
                if fork is None:
                    fork = snapshot
                else:
                    assert snapshot == fork, 'Full replay-state audit differs at fork'
                actions = arrays['predicted_actions'][parent['query']]
                if mode == 'ten':
                    worker.step(actions, parent['query'])
                else:
                    worker.step(actions[:5], parent['query'])
                    worker.step(actions[5:], compare_start=parent['start'] + 5)
                current_end = worker.current['state_audit']
                if end is None:
                    end = current_end
                else:
                    assert current_end == end, '10 versus 5+5 altered execution semantics'
                report = worker.close()
                worker = None
                reports.append(dict(parent=parent['name'], mode=mode, fork=fork, **report))
                print(json.dumps(dict(preflight=parent['name'], mode=mode, exact_steps=report['exact_source_steps'])), flush=True)
            with np.load(root / parent['name'] / 'ten/rollout.npz') as a, np.load(root / parent['name'] / 'five_plus_five/rollout.npz') as b:
                assert set(a.files) == set(b.files)
                for key in a.files:
                    equal(a[key], b[key], 'chunk submission equivalence ' + key)
        save_json(root / 'result.json', dict(passed=True, reports=reports, utc=now(),
                                            worker_sha256=sha256_file(BASE / 'replan_env_worker.py')))
        print(json.dumps(dict(preflight_passed=True, path=str(root))), flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise
    finally:
        if worker is not None:
            worker.abort()


def shared_metadata():
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        return client.metadata


def prepare(preflight_root):
    prior = json.loads((P3G / 'config.json').read_text())
    verification = json.loads((P3G / 'verification.json').read_text())
    sources = dict(prior['source_hashes'], **verification['post_collection_sources'])
    sources.update({str(BASE / name): sha256_file(BASE / name) for name in SOURCES})
    for path, sha in sources.items():
        assert sha256_file(Path(path)) == sha, path
    old_files = {}
    for sealed in sorted((BASE / 'runs').glob('*/verification.json')):
        record = json.loads(sealed.read_text())
        old_files[str(sealed)] = sha256_file(sealed)
        for relative, sha in record.get('files', record.get('file_sha256')).items():
            path = sealed.parent / relative
            assert sha256_file(path) == sha, str(path)
            old_files[str(path)] = sha
    preflight_record = json.loads((preflight_root / 'result.json').read_text())
    assert preflight_record['passed']
    assert preflight_record['worker_sha256'] == sha256_file(BASE / 'replan_env_worker.py')
    metadata = shared_metadata()
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == prior['shared_metadata'][key]
    assert shutil.disk_usage(BASE).free > 8 * 1024 ** 3
    root = Path(tempfile.mkdtemp(prefix='p3h-replan-window-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_replan_window.v1', utc=now(), parents=parents(), arms=ARMS,
                  replicates=REPLICATES, flow_steps=10, generated_horizon=10, normal_execution=10,
                  short_execution=5, window_physical_steps=20, horizon=HORIZON, maximum_windows=1,
                  source_hashes=sources, previous_sealed_files=old_files, shared_metadata=metadata,
                  preflight=str(preflight_root), preflight_sha256=sha256_file(preflight_root / 'result.json'),
                  actual_rollouts=64, maximum_model_calls=maximum_calls(), reserved_timing_parents=TIMING_PARENTS,
                  intervention='forced at frozen snapshot, not an online timing-effect estimate',
                  selection='development, known original outcomes, no new outcomes used',
                  timing_gate='net rescue > 0 and at least two task/init groups with >= 2 paired rescues each',
                  seed_protocol='replicate0 original; replicate1-3 new; indexed by absolute physical step',
                  simulator_during_inference='paused; no real asynchronous-delay claim',
                  architecture_changed=False, gate_changed=False, output_patched=False, training=False)
    save_json(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(sha256_file(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), rollouts=64, maximum_model_calls=maximum_calls())), flush=True)


def check_config(root, check_old=False):
    config = json.loads((root / 'config.json').read_text())
    assert sha256_file(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for path, sha in config['source_hashes'].items():
        assert sha256_file(Path(path)) == sha, path
    assert config['parents'] == parents()
    assert sha256_file(Path(config['preflight']) / 'result.json') == config['preflight_sha256']
    if check_old:
        for path, sha in config['previous_sealed_files'].items():
            assert sha256_file(Path(path)) == sha, path
    return config


def collect(root):
    import torch

    config = check_config(root)
    save_json(root / 'started.json', dict(utc=now()))
    started, attempts, completed = time.monotonic(), 0, 0
    results, worker = [], None
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        wrapped, loaded = load_isolated()
        model = wrapped.policy._policy.model
        versions, weights_before = parameter_versions(model), parameter_digest(model)
        rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
        with (root / 'model-rng.npz').open('xb') as stream:
            np.savez_compressed(stream, cpu=rng_cpu.numpy(), cuda=rng_cuda.numpy())
        save_json(root / 'model-load.json', dict(loaded, parameter_sha256=weights_before))
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as call_log:
            def infer(parent, replicate, arm, current, directory, validation=False):
                nonlocal attempts, completed
                assert attempts < config['maximum_model_calls']
                assert shutil.disk_usage(root).free > 5 * 1024 ** 3
                step = current['steps']
                noise = physical_noise(parent, parent['flow_seed'], replicate, step)
                event = dict(ordinal=attempts, parent=parent['name'], replicate=replicate, arm=arm,
                             step=step, validation=validation, noise_sha256=sha_array(noise))
                call_log.write(json.dumps(dict(event, event='request')) + '\n')
                call_log.flush()
                attempts += 1
                response, resource = infer_isolated(wrapped, dict(current['observation'],
                    **{'flow/noise': noise, 'routing/capture': True, CAPTURE_KEY: True}))
                completed += 1
                assert response['actions'].shape == (10, 7) and response[PROBS_KEY].shape == (8, 10, 11, 32)
                assert response['gate_probe/exact_dtype_audit'] and response['collection/routing_mode'] == 'native_only'
                equal(response[NATIVE_IDS_KEY], response[EFFECTIVE_IDS_KEY], 'No expert substitution')
                equal(response[NATIVE_WEIGHTS_KEY], response[EFFECTIVE_WEIGHTS_KEY], 'No gate change')
                assert parameter_versions(model) == versions
                path = directory / ('s%03d%s.npz' % (step, '-repeat' if validation else ''))
                payload = {k: v for k, v in response.items() if isinstance(v, np.ndarray)}
                payload.update({k: v for k, v in current['observation'].items() if isinstance(v, np.ndarray)})
                payload.update(request_noise=noise, sim_before=current['sim_state'])
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **payload)
                row = dict(event, event='response', path=str(path.relative_to(root)), sha256=sha256_file(path),
                           resource=resource, actions_sha256=sha_array(response['actions']),
                           hb_sha256=sha_array(response[PROBS_KEY]), state_audit=current['state_audit'])
                call_log.write(json.dumps(row) + '\n')
                call_log.flush()
                return response, row

            for parent in config['parents']:
                _, arrays = load_episode_trace(Path(parent['source']))
                for key, value in parent['policy_identity'].items():
                    assert loaded['metadata'][key] == value, key
                with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as archive:
                    reference = archive['hb_router_probs']
                fork_reference = None
                for replicate in range(REPLICATES):
                    first_forward = None
                    for arm in ARMS:
                        directory = root / parent['name'] / ('r%d' % replicate) / arm
                        branch_started = time.monotonic()
                        worker = ReplanWorker(parent, directory)
                        for q in range(parent['query']):
                            worker.step(arrays['predicted_actions'][q], q)
                        assert_source_observation(worker.current, arrays, parent['query'])
                        fork = worker.snapshot()
                        if fork_reference is None:
                            fork_reference = fork
                        else:
                            assert fork == fork_reference, 'Replay simulator/controller/RNG state not paired'
                        torch.set_rng_state(rng_cpu)
                        torch.cuda.set_rng_state(rng_cuda)
                        monitor = V82Monitor()
                        for p in reference[:parent['query']]:
                            monitor.update(p)
                        query_dir = directory / 'queries'
                        query_dir.mkdir()
                        window = ExecutionWindow(arm, parent['start'])
                        records = []
                        print(json.dumps(dict(branch_start=parent['name'], replicate=replicate, arm=arm,
                                              calls=completed, step=parent['start'])), flush=True)
                        with (directory / 'queries.jsonl').open('x') as query_log:
                            while not worker.current['success'] and worker.current['steps'] < HORIZON:
                                current, step = worker.current, worker.current['steps']
                                response, call = infer(parent, replicate, arm, current, query_dir)
                                if step == parent['start']:
                                    first = (call['actions_sha256'], call['hb_sha256'])
                                    if first_forward is None:
                                        first_forward = first
                                    else:
                                        assert first == first_forward, 'Arms have different first native proposal'
                                    if replicate == 0 and arm == 'native10':
                                        duplicate, _ = infer(parent, replicate, arm, current, query_dir, True)
                                        for key, value in response.items():
                                            if isinstance(value, np.ndarray):
                                                equal(value, duplicate[key], 'Deterministic duplicate ' + key)
                                diagnostic = None
                                if step % 10 == 0:
                                    diagnostic = jsonable(monitor.update(response[PROBS_KEY]))
                                    assert monitor.v7.query == step // 10, 'Alarm physical clock drift'
                                if arm == 'native10' and replicate == 0:
                                    assert_source_observation(current, arrays, step // 10)
                                    equal(response['actions'], arrays['predicted_actions'][step // 10], 'Original native actions')
                                    equal(response[PROBS_KEY], reference[step // 10], 'Original native MoE')
                                prefix = window.prefix(response['actions'], step)
                                env_started = time.monotonic()
                                worker.step(prefix, step // 10 if arm == 'native10' and replicate == 0 else None)
                                row = dict(parent=parent['name'], replicate=replicate, arm=arm, step=step,
                                           path=call['path'], planned_execution=len(prefix), executed=worker.current['executed'],
                                           suffix_discarded=10 - len(prefix), monitor=diagnostic, physics=current['physics'],
                                           next_physics=worker.current['physics'], state_audit=current['state_audit'],
                                           next_state_audit=worker.current['state_audit'],
                                           inference_seconds=call['resource']['seconds'], environment_seconds=time.monotonic() - env_started,
                                           success=worker.current['success'])
                                row = jsonable(row)
                                query_log.write(json.dumps(row, allow_nan=False) + '\n')
                                query_log.flush()
                                records.append(row)
                                if len(records) % 10 == 0 or worker.current['success']:
                                    print(json.dumps(dict(parent=parent['name'], replicate=replicate, arm=arm,
                                                          steps=worker.current['steps'], calls=completed,
                                                          elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                        report = worker.close()
                        worker = None
                        if arm == 'native10' and replicate == 0:
                            assert report['success'] == parent['source_success'] and report['steps'] == parent['source_steps']
                        result = dict(parent=parent['name'], group=list(group_key(parent)), stratum=parent['stratum'],
                                      replicate=replicate, arm=arm, start=parent['start'], success=report['success'],
                                      source_success=parent['source_success'], action_steps=report['steps'],
                                      suffix_action_steps=report['steps'] - parent['start'], suffix_calls=len(records),
                                      deployment_full_calls=parent['query'] + len(records),
                                      validation_calls=int(replicate == 0 and arm == 'native10'),
                                      window_steps=sum(r['executed'] for r in records if r['planned_execution'] == 5),
                                      inference_seconds=sum(r['inference_seconds'] for r in records),
                                      environment_seconds=sum(r['environment_seconds'] for r in records),
                                      elapsed_seconds=time.monotonic() - branch_started,
                                      first_v82_physical_step=monitor.first_v82_alarm * 10,
                                      prefix_state_audit=fork, terminal_state_audit=report['final_state_audit'])
                        save_json(directory / 'result.json', result)
                        results.append(result)
                        print(json.dumps(dict(branch_complete=True, parent=parent['name'], replicate=replicate, arm=arm,
                                              success=result['success'], steps=result['action_steps'], completed=len(results))), flush=True)
        assert len(results) == 64 and attempts == completed <= maximum_calls()
        weights_after = parameter_digest(model)
        assert weights_before == weights_after and parameter_versions(model) == versions
        check_config(root)
        metadata_after = shared_metadata()
        for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
            assert metadata_after[key] == config['shared_metadata'][key]
        save_json(root / 'collection.json', dict(passed=True, results=results, model_attempts=attempts,
            model_completed=completed, rollouts=64, environment_action_steps=sum(r['action_steps'] for r in results),
            settle_steps=640, parameter_sha256_before=weights_before, parameter_sha256_after=weights_after,
            elapsed_seconds=time.monotonic() - started, finished_utc=now(), shared_metadata_after=metadata_after))
        print(json.dumps(dict(completed=True, calls=completed, rollouts=64)), flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(),
            model_attempts=attempts, model_completed=completed, results=results, elapsed_seconds=time.monotonic() - started))
        raise
    finally:
        if worker is not None:
            worker.abort()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('preflight', 'prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight()
    elif args.command == 'prepare':
        prepare(args.preflight.resolve())
    else:
        collect(args.run.resolve())
