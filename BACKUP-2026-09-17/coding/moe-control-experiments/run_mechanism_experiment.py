"""Freeze and run activation tracing on the exact P3d observation histories."""

import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, load_isolated
from mechanism_capture import traced_infer
from mechanism_protocol import LABELS, selected_queries, variants, trace_effect
from run_response_matrix_experiment import MatrixPolicy
from run_gated_rollout_experiment import P3C, LOGIT_KEYS
from run_gate_experiment import FULL_KEYS, assert_equal, check_identity
from run_online_experiment import jsonable, now, EVALUATION
from audit_online_experiment import read, write, digest, array_digest, same, records
from response_matrix_protocol import SHAPE, bias_bank, specifications, measure
from gated_rollout_protocol import evaluate
from v82_closed_loop import V82Monitor
from v8_feature_control import EFFECTIVE_PROBS
from collection_routes import CAPTURE_KEY, PROBS_KEY, EFFECTIVE_IDS_KEY
from scope_bias_control import BIAS_KEY
from online_protocol import noise_bank
from himoe_libero_bridge.episode_trace import load_episode_trace
from himoe_libero_bridge.client import PolicyClient

P3D = BASE / 'runs/p3d-gated-rollout-20260915-t9k1v766'
NEW_SOURCES = ('P3E_MECHANISM_PLAN.zh.md', 'mechanism_protocol.py', 'mechanism_capture.py',
               'mechanism_physics.py', 'run_mechanism_experiment.py', 'test_mechanism_protocol.py')


def protected_sources():
    config = read(P3D / 'config.json')
    sources = config['source_hashes'].copy()
    sources.update(read(P3D / 'verification.json')['post_collection_sources'])
    for path, expected in sources.items():
        assert digest(path) == expected, path
    for name in NEW_SOURCES:
        sources[str(BASE / name)] = digest(BASE / name)
    for name in ('himoe.py', 'modeling_moe.py', 'moevla.py', 'paligemma_with_expert.py'):
        path = Path('/data/srv/src/moevla/models') / name
        sources[str(path)] = digest(path)
    for benchmark in ('LIBERO-plus', 'LIBERO-PRO'):
        envs = Path('/data/libero-runtime/upstream') / benchmark / 'libero/libero/envs'
        for path in sorted(envs.rglob('*.py')):
            sources[str(path)] = digest(path)
    grasp = Path('/data/libero-runtime/envs/libero/lib/python3.8/site-packages/robosuite/environments/manipulation/manipulation_env.py')
    sources[str(grasp)] = digest(grasp)
    return sources


def prepare():
    old = read(P3D / 'config.json')
    for path, expected in read(P3D / 'verification.json')['files'].items():
        assert digest(P3D / path) == expected, path
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == old['shared_metadata'][key]
    root = Path(tempfile.mkdtemp(prefix='p3e-mechanism-20260915-', dir=BASE / 'runs'))
    cases = []
    for parent in old['parents']:
        _, arrays = load_episode_trace(Path(parent['source']))
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as source:
            reference = source['hb_router_probs']
        decisions = records(P3D / parent['name'] / 'mobility_balance/decisions.jsonl')
        for phase, query in selected_queries(parent, decisions):
            directory = root / 'states' / parent['name'] / phase
            directory.mkdir(parents=True)
            old_native = old_combo = None
            prefix = [p for p in reference[:min(query, parent['query'])]]
            if phase == 'before_alarm':
                inputs = {key: arrays[source][query] for key, source in (
                    ('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'), ('observation/state', 'states'))}
                manifest = read(Path(parent['source']) / 'episode-trace.json')
                noise = noise_bank(manifest['config']['flow_noise_seed'])[query]
                bank, _ = bias_bank(reference[query], reference[query - 1], dict(parent, query=query))
                index = next(i for i, spec in enumerate(specifications()) if spec['label'] == 'mobility_balance-high-plus')
                bias = bank[index]
                sim = arrays['sim_states_before'][query]
            else:
                for row in decisions:
                    if row['query'] >= query:
                        break
                    with np.load(P3D / row['selected_path'], allow_pickle=False) as saved:
                        prefix.append(saved[EFFECTIVE_PROBS].astype(np.float16))
                row, = [row for row in decisions if row['query'] == query]
                assert row['decision']['accepted']
                old_native, old_combo = row['native_path'], row['candidate_path']
                with np.load(P3D / old_native, allow_pickle=False) as saved:
                    inputs = {key: saved[key] for key in saved.files if key.startswith('observation/')}
                    noise, sim = saved['request_noise'], saved['sim_before']
                with np.load(P3D / old_combo, allow_pickle=False) as saved:
                    bias = saved['request_bias']
            assert len(prefix) == query
            payload = dict(inputs, request_noise=noise, request_bias=bias, sim_before=sim, prefix_hb=np.stack(prefix))
            path = directory / 'input.npz'
            with path.open('xb') as stream:
                np.savez_compressed(stream, **payload)
            cases.append(dict(parent=parent['name'], phase=phase, query=query, path=str(path.relative_to(root)),
                              sha256=digest(path), old_native=old_native, old_combo=old_combo))
    config = dict(schema='local.moe_mechanism.v1', utc=now(), parents=old['parents'], cases=cases,
                  labels=LABELS, expected_model_calls=105, source_run=str(P3D),
                  source_verification_sha256=digest(P3D / 'verification.json'), source_hashes=protected_sources(),
                  shared_metadata=metadata, new_environment_actions=0, training=False, weights_changed=False,
                  physics_arms=['native', 'mobility_balance'], selection='15 paired development states, five parents')
    write(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), cases=15, model_calls=105)), flush=True)


def check_config(root):
    config = read(root / 'config.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert digest(P3D / 'verification.json') == config['source_verification_sha256']
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    for case in config['cases']:
        assert digest(root / case['path']) == case['sha256']
    return config


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    attempts = completed = 0
    started = time.monotonic()
    try:
        wrapped, loaded = load_isolated()
        modified = MatrixPolicy(wrapped.policy)
        write(root / 'model-load.json', loaded)
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6])
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as log:
            for case in config['cases']:
                parent, = [p for p in config['parents'] if p['name'] == case['parent']]
                check_identity(loaded['metadata'], parent['policy_identity'])
                with np.load(root / case['path'], allow_pickle=False) as saved:
                    inputs = {key: saved[key] for key in saved.files}
                monitor = V82Monitor()
                for route in inputs['prefix_hb']:
                    monitor.update(route)
                request = {key: value for key, value in inputs.items() if key.startswith('observation/')}
                request.update(prompt=parent['prompt'], **{'flow/noise': inputs['request_noise'], 'routing/capture': True, CAPTURE_KEY: True})
                bank = variants(inputs['request_bias'], parent, case['query'])
                native = combo = None
                for label in LABELS:
                    if attempts >= config['expected_model_calls']:
                        raise RuntimeError('Model call budget exceeded')
                    bias = bank[label]
                    event = dict(ordinal=attempts, parent=parent['name'], phase=case['phase'], query=case['query'], label=label,
                                 noise_sha256=array_digest(inputs['request_noise']),
                                 bias_sha256=array_digest(np.zeros(SHAPE, np.float32) if bias is None else bias))
                    log.write(json.dumps(dict(event, event='request')) + '\n')
                    log.flush()
                    attempts += 1
                    response, resource = traced_infer(wrapped if bias is None else modified,
                        request if bias is None else dict(request, **{BIAS_KEY: bias}))
                    completed += 1
                    if label == 'native':
                        native = response
                    if label == 'combo':
                        combo = response
                    if label in ('zero', 'post', 'repeat'):
                        baseline = combo if label == 'repeat' else native
                        assert_equal(response, baseline, FULL_KEYS + tuple(k for k in native if k.startswith('mechanism/')), label)
                    if label in ('native', 'combo') and case['old_native']:
                        with np.load(P3D / case['old_' + label], allow_pickle=False) as saved:
                            assert_equal(response, saved, FULL_KEYS + (LOGIT_KEYS if label == 'combo' else ()), 'P3d source ' + label)
                    if label == 'native' and case['phase'] == 'before_alarm':
                        _, arrays = load_episode_trace(Path(parent['source']))
                        same(response['actions'], arrays['predicted_actions'][case['query']], 'prealarm native action')
                        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as saved:
                            same(response[PROBS_KEY], saved['hb_router_probs'][case['query']], 'prealarm native route')
                    effect = trace_effect(response, native)
                    assert effect['first_local_input_equal'] and effect['first_local_shared_equal']
                    decision = evaluate(monitor, native, response, 'mobility_balance', std)
                    changed = np.any(np.sort(response[EFFECTIVE_IDS_KEY], axis=-1) != np.sort(native[EFFECTIVE_IDS_KEY], axis=-1), axis=-1)
                    path = (root / case['path']).parent / (label + '.npz')
                    with path.open('xb') as stream:
                        np.savez_compressed(stream, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                            request_noise=inputs['request_noise'], request_bias=np.zeros(SHAPE, np.float32) if bias is None else bias)
                    row = jsonable(dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path),
                        resource=resource, effect=effect, decision=decision, top4_changed_fraction=float(changed[4:, :, 1:].mean()),
                        residual_exact=True, hooks_removed=True))
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(dict(parent=parent['name'], phase=case['phase'], label=label, calls=completed,
                        action_rms=decision['normalized_action_rms'], elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                del native, combo, response
        assert attempts == completed == config['expected_model_calls']
        check_config(root)
        write(root / 'collection.json', dict(passed=True, model_attempts=attempts, model_completed=completed,
            new_environment_actions=0, elapsed_seconds=time.monotonic() - started, finished_utc=now()))
    except BaseException as error:
        write(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(), model_attempts=attempts,
            model_completed=completed, elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'collect'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else collect(args.run.resolve())
