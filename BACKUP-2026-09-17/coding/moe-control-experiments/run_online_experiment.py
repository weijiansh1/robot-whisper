"""Run a frozen, paired, actual-simulator online correction pilot."""

import argparse
import copy
import csv
import datetime
import hashlib
import json
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, load_isolated, infer_isolated
from online_protocol import (ARMS, PARENTS, CANDIDATES, RECOVERY_QUERIES, HORIZON,
                             UPSTREAM, candidate_noises, noise_bank, query_mode, select_candidate)
from collection_routes import CAPTURE_KEY, PROBS_KEY
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from openpi_client import msgpack_numpy as codec
from v82_closed_loop import V82Monitor

EVALUATION = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')
SIM_PYTHON = '/data/libero-runtime/envs/libero/bin/python'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def equal(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.dtype != b.dtype or not np.array_equal(a, b):
        detail = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.shape == b.shape else None
        raise RuntimeError('%s mismatch: %s/%s versus %s/%s, max_abs=%s' %
                           (label, a.shape, a.dtype, b.shape, b.dtype, detail))


def sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def frozen_files():
    files = [BASE / name for name in (
        'P3_ONLINE_PLAN.zh.md', 'online_protocol.py', 'online_env_worker.py',
        'run_online_experiment.py', 'test_online_protocol.py', 'gate_runtime.py', 'gate_capture.py')]
    reference = json.loads((EVALUATION / 'plan.json').read_text())
    files += [UPSTREAM / name for name in reference['source_sha256']]
    files += [UPSTREAM / 'himoe-route-capture/route_noise_selector.py',
              UPSTREAM / 'moe-trap-control/collection_routes.py',
              UPSTREAM / 'moe-trap-control/scope_bias_control.py']
    files += [Path('/data/srv/src/himoe_libero_bridge') / name for name in
              ('policies.py', 'preprocess.py', 'libero_runtime.py', 'episode_trace.py', 'protocol.py')]
    files += [Path('/data/srv/packages/openpi-client/src/openpi_client/msgpack_numpy.py')]
    return {str(path): sha256_file(path) for path in files}


def parent_sources():
    with (EVALUATION / 'episodes.csv').open() as stream:
        indexed = {row['name']: row for row in csv.DictReader(stream)}
    result = []
    for name in PARENTS:
        row = indexed[name]
        source = Path(row['source_artifact_dir'])
        manifest, arrays = load_episode_trace(source)
        bank = noise_bank(manifest['config']['flow_noise_seed'])
        equal(bank[:len(arrays['flow_noises'])], arrays['flow_noises'], 'source noise stream')
        result.append(dict(name=name, source=str(source), benchmark=row['benchmark'],
                           base_task_id=int(row['base_task_id']), init_state_id=int(row['init_state_id']),
                           source_success=row['success'] == 'True', first_alarm=int(row['first_v82_query']),
                           source_queries=len(arrays['images']), source_steps=int(row['action_steps']),
                           source_manifest_sha256=sha256_file(source / 'episode-trace.json'),
                           source_trace_sha256=sha256_file(source / 'episode-trace.npz'),
                           hb_sha256=sha256_file(EVALUATION / name / 'full-hb-routes.npz')))
    return result


def prepare():
    root = Path(tempfile.mkdtemp(prefix='p3-online-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_route_online_control.v1', prepared_utc=now(),
                  parents=parent_sources(), arms=list(ARMS), source_hashes=frozen_files(),
                  recovery_queries=RECOVERY_QUERIES, candidates=CANDIDATES, horizon=HORIZON,
                  selection='purposeful development pilot; source outcomes and alarms already inspected',
                  native_from_zero=True, branches='fresh reset and exact common-prefix replay',
                  weights_modified=False, gate_modified=False, candidate_environment_rollouts=False)
    save_json(root / 'config.json', config)
    (root / 'config.sha256').write_text(sha256_file(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), parents=len(config['parents']), arms=config['arms'])), flush=True)


def checked_config(root):
    if sha256_file(root / 'config.json') != (root / 'config.sha256').read_text().strip():
        raise RuntimeError('Frozen configuration changed')
    config = json.loads((root / 'config.json').read_text())
    if config['source_hashes'] != frozen_files() or config['parents'] != parent_sources():
        raise RuntimeError('Frozen code or inputs changed')
    return config


class EnvironmentWorker:
    def __init__(self, parent, directory):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        self.log = (directory / 'worker.log').open('x')
        parent_socket, child_socket = socket.socketpair()
        self.connection = Connection(parent_socket.detach())
        self.process = subprocess.Popen(
            [SIM_PYTHON, str(BASE / 'online_env_worker.py'), '--fd', str(child_socket.fileno()),
             '--source', parent['source'], '--benchmark', parent['benchmark'], '--out', str(directory)],
            pass_fds=(child_socket.fileno(),), stdout=self.log, stderr=subprocess.STDOUT)
        child_socket.close()
        self.closed = False
        try:
            self.current = self.receive()
            if self.current['event'] != 'ready':
                raise RuntimeError('Worker did not initialize')
        except BaseException:
            self.abort()
            raise

    def receive(self):
        if not self.connection.poll(120):
            raise RuntimeError('Environment worker timed out')
        result = codec.unpackb(self.connection.recv_bytes())
        if result.get('event') == 'error':
            raise RuntimeError(result['traceback'])
        return result

    def step(self, actions, query=None):
        self.connection.send_bytes(codec.packb(dict(op='step', actions=actions, compare_query=query)))
        self.current = self.receive()
        if self.current['event'] != 'stepped':
            raise RuntimeError('Missing worker step acknowledgement')
        return self.current

    def close(self):
        self.connection.send_bytes(codec.packb(dict(op='close')))
        result = self.receive()
        if result['event'] != 'closed':
            raise RuntimeError('Missing worker close acknowledgement')
        self.process.wait(timeout=30)
        if self.process.returncode:
            raise RuntimeError('Worker exited unsuccessfully')
        self.connection.close()
        self.log.close()
        self.closed = True
        return result['report']

    def abort(self):
        if self.closed:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.connection.close()
        self.log.close()
        self.closed = True


def assert_source_observation(current, arrays, query):
    for key, source_key in (('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'),
                            ('observation/state', 'states')):
        equal(current['observation'][key], arrays[source_key][query], key)
    equal(current['sim_state'], arrays['sim_states_before'][query], 'source sim before')
    if current['steps'] != int(arrays['replan_start_steps'][query]):
        raise RuntimeError('Source query action step mismatch')


def preflight():
    root = Path(tempfile.mkdtemp(prefix='online-env-preflight-', dir=BASE / 'runs'))
    reports = []
    for parent in parent_sources()[:2]:
        worker = EnvironmentWorker(parent, root / parent['name'])
        try:
            _, arrays = load_episode_trace(Path(parent['source']))
            for query in range(2):
                assert_source_observation(worker.current, arrays, query)
                worker.step(arrays['predicted_actions'][query], query)
            report = worker.close()
            reports.append(dict(parent=parent['name'], **report))
            print(json.dumps(dict(preflight_parent=parent['name'], exact_steps=report['exact_source_steps'])), flush=True)
        finally:
            worker.abort()
    save_json(root / 'result.json', dict(passed=True, reports=reports, utc=now()))
    print(json.dumps(dict(preflight=str(root), passed=True)), flush=True)


def collect(root):
    config = checked_config(root)
    save_json(root / 'started.json', dict(utc=now()))
    began = time.monotonic()
    attempts, completed, results = 0, 0, []
    worker = None
    try:
        wrapped, loaded = load_isolated()
        save_json(root / 'model-load.json', loaded)
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as calls:
            for parent in config['parents']:
                manifest, arrays = load_episode_trace(Path(parent['source']))
                for key, value in manifest['policy_identity'].items():
                    if loaded['metadata'].get(key) != value:
                        raise RuntimeError('Model identity mismatch: ' + key)
                with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as hb_archive:
                    reference_hb = hb_archive['hb_router_probs']
                bank = noise_bank(manifest['config']['flow_noise_seed'])
                native_routes, native_records = [], []
                native_result = None
                for arm in ARMS:
                    directory = root / parent['name'] / arm
                    if arm != 'native' and parent['first_alarm'] < 0:
                        directory.mkdir(parents=True)
                        result = dict(native_result, arm=arm, alias_of='native', actual_model_calls=0,
                                      actual_environment_steps=0, actual_settle_steps=0,
                                      deployed_model_calls=native_result['deployed_model_calls'])
                        save_json(directory / 'result.json', result)
                        results.append(result)
                        continue
                    worker = EnvironmentWorker(parent, directory)
                    (directory / 'queries').mkdir()
                    monitor = V82Monitor()
                    prefix_queries = 0 if arm == 'native' else parent['first_alarm'] + 1
                    for query in range(prefix_queries):
                        assert_source_observation(worker.current, arrays, query)
                        monitor.update(native_routes[query])
                        worker.step(arrays['predicted_actions'][query], query)
                    if prefix_queries and monitor.first_v82_alarm != parent['first_alarm']:
                        raise RuntimeError('Branch alarm prefix changed')
                    query, used, actual_calls, active_queries = prefix_queries, 0, 0, 0
                    records = copy.deepcopy(native_records[:prefix_queries]) if prefix_queries else []
                    with (directory / 'queries.jsonl').open('x') as query_log:
                        for record in records:
                            query_log.write(json.dumps(dict(record, origin='native_prefix'), allow_nan=False) + '\n')
                        while not worker.current['success'] and worker.current['steps'] < HORIZON:
                            current = worker.current
                            step = current['steps']
                            mode = query_mode(arm, used)
                            if arm == 'native':
                                assert_source_observation(current, arrays, query)
                            noises = candidate_noises(parent, bank, step, mode['candidates'])
                            responses, costs = [], []
                            for candidate, noise in enumerate(noises):
                                request = dict(current['observation'], **{'flow/noise': noise, 'routing/capture': True, CAPTURE_KEY: True})
                                event = dict(parent=parent['name'], arm=arm, query=query, step=step,
                                             candidate=candidate, ordinal=attempts, noise_sha256=sha_array(noise))
                                calls.write(json.dumps(dict(event, event='request')) + '\n')
                                calls.flush()
                                attempts += 1
                                response, resource = infer_isolated(wrapped, request)
                                if response['collection/routing_mode'] != 'native_only' or not response['gate_probe/exact_dtype_audit']:
                                    raise RuntimeError('Unexpected model intervention')
                                probabilities = response[PROBS_KEY]
                                if probabilities.shape != (8, 10, 11, 32) or probabilities.dtype != np.float16:
                                    raise RuntimeError('Invalid complete HB capture')
                                responses.append(response)
                                costs.append(resource)
                                completed += 1
                                actual_calls += 1
                                calls.write(json.dumps(dict(event, event='response', resource=resource,
                                                            actions_sha256=sha_array(response['actions']), hb_sha256=sha_array(probabilities))) + '\n')
                                calls.flush()
                            pool_hb = np.stack([r[PROBS_KEY] for r in responses])
                            if mode['active']:
                                selected, scores = select_candidate(arm, pool_hb, parent, step)
                                used += 1
                                active_queries += 1
                            else:
                                selected, scores = 0, np.zeros(1)
                            chosen = responses[selected]
                            if arm == 'native':
                                equal(chosen['actions'], arrays['predicted_actions'][query], 'native source actions')
                                equal(chosen[PROBS_KEY], reference_hb[query], 'native source full HB')
                            status = monitor.update(chosen[PROBS_KEY])
                            snapshot = directory / 'queries' / ('q%03d.npz' % query)
                            with snapshot.open('xb') as stream:
                                np.savez_compressed(stream, image=current['observation']['observation/image'],
                                                    wrist_image=current['observation']['observation/wrist_image'],
                                                    state=current['observation']['observation/state'], sim_before=current['sim_state'],
                                                    noises=noises, actions=np.stack([r['actions'] for r in responses]),
                                                    hb=pool_hb, scores=scores, selected=np.int16(selected))
                            record = jsonable(dict(parent=parent['name'], arm=arm, query=query, step=step,
                                                   active=mode['active'], chunk=mode['chunk'], candidates=mode['candidates'],
                                                   selected=selected, scores=scores, monitor=status, costs=costs,
                                                   archive=str(snapshot.relative_to(root)), archive_sha256=sha256_file(snapshot)))
                            worker.step(chosen['actions'][:mode['chunk']], query if arm == 'native' else None)
                            record.update(executed=worker.current['executed'], success=worker.current['success'], origin='online')
                            query_log.write(json.dumps(record, allow_nan=False) + '\n')
                            query_log.flush()
                            records.append(record)
                            if arm == 'native':
                                native_routes.append(chosen[PROBS_KEY].copy())
                                native_records.append(record)
                            query += 1
                            if query % 10 == 0 or worker.current['success']:
                                print(json.dumps(dict(parent=parent['name'], arm=arm, query=query, action_steps=worker.current['steps'],
                                                      success=worker.current['success'], calls=completed,
                                                      elapsed_seconds=round(time.monotonic() - began, 1))), flush=True)
                    environment_report = worker.close()
                    worker = None
                    if arm == 'native':
                        if monitor.first_v82_alarm != parent['first_alarm'] or environment_report['success'] != parent['source_success']:
                            raise RuntimeError('Native alarm or outcome did not reproduce')
                        if environment_report['steps'] != parent['source_steps'] or query != parent['source_queries']:
                            raise RuntimeError('Native endpoint did not reproduce')
                    result = dict(parent=parent['name'], arm=arm, success=environment_report['success'],
                                  action_steps=environment_report['steps'], queries=query, prefix_queries=prefix_queries,
                                  active_queries=active_queries, first_alarm=monitor.first_v82_alarm,
                                  actual_model_calls=actual_calls, deployed_model_calls=prefix_queries + actual_calls,
                                  actual_environment_steps=environment_report['steps'], actual_settle_steps=environment_report['settle_steps'],
                                  native_source_exact=arm == 'native', prefix_source_exact=True,
                                  rescue=not parent['source_success'] and environment_report['success'],
                                  harm=parent['source_success'] and not environment_report['success'], alias_of=None)
                    save_json(directory / 'result.json', result)
                    if arm == 'native':
                        native_result = result
                    results.append(result)
                    print(json.dumps(dict(arm_completed=True, **result)), flush=True)
        checked_config(root)
        save_json(root / 'collection.json', dict(passed=True, results=results, model_attempts=attempts,
                                                 model_completed=completed, finished_utc=now(),
                                                 elapsed_seconds=time.monotonic() - began))
        print(json.dumps(dict(completed=True, run=str(root), model_calls=completed)), flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(), utc=now(),
                                             model_attempts=attempts, model_completed=completed, completed_results=results))
        raise
    finally:
        if worker is not None:
            worker.abort()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight', 'prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight()
    elif args.mode == 'prepare':
        prepare()
    else:
        if args.run is None:
            parser.error('--run is required for collection')
        collect(args.run.resolve())


if __name__ == '__main__':
    main()
