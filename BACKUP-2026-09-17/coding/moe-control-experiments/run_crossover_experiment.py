"""Freeze and collect the MoE-only 2x2 crossover at 15 archived inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, load_isolated
from crossover_capture import crossover_infer, parameter_versions, parameter_digest
from crossover_protocol import (LABELS, MAIN_LABELS, BIASED, DONORS, CONTROLS, BIAS_SHAPE,
                                SITE_MASK, validate_bias, expected_computation, set_change)
from mechanism_protocol import HB_FIELDS, comparison
from run_response_matrix_experiment import MatrixPolicy
from run_gated_rollout_experiment import LOGIT_KEYS
from run_gate_experiment import FULL_KEYS, assert_equal, check_identity
from run_online_experiment import jsonable, now
from audit_online_experiment import read, write, digest, array_digest, same
from gated_rollout_protocol import evaluate
from v82_closed_loop import V82Monitor
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS
from collection_routes import CAPTURE_KEY, EFFECTIVE_IDS_KEY
from scope_bias_control import BIAS_KEY
from himoe_libero_bridge.client import PolicyClient

P3E = BASE / 'runs/p3e-mechanism-20260915-9u_9j9b8'
NEW_SOURCES = ('P3F_CROSSOVER_PLAN.zh.md', 'crossover_protocol.py', 'crossover_capture.py',
               'run_crossover_experiment.py', 'test_crossover_protocol.py')
TRACE_KEYS = tuple('mechanism/' + k for k in HB_FIELDS + ('projection_input', 'velocity'))
INPUT_KEYS = ('observation/image', 'observation/wrist_image', 'observation/state',
              'request_noise', 'request_bias', 'prefix_hb')


def prepare():
    old, verification = read(P3E / 'config.json'), read(P3E / 'verification.json')
    for path, expected in verification['files'].items():
        assert digest(P3E / path) == expected, path
    sources = dict(old['source_hashes'], **verification['post_collection_sources'])
    for path, expected in sources.items():
        assert digest(path) == expected, path
    sources.update({str(BASE / name): digest(BASE / name) for name in NEW_SOURCES})
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == old['shared_metadata'][key]
    root = Path(tempfile.mkdtemp(prefix='p3f-crossover-20260915-', dir=BASE / 'runs'))
    cases = []
    for old_case in old['cases']:
        with np.load(P3E / old_case['path'], allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in INPUT_KEYS}
        validate_bias(inputs['request_bias'])
        directory = root / 'states' / old_case['parent'] / old_case['phase']
        directory.mkdir(parents=True)
        path = directory / 'input.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **inputs)
        source_dir = (P3E / old_case['path']).parent
        cases.append(dict(parent=old_case['parent'], phase=old_case['phase'], query=old_case['query'],
                          path=str(path.relative_to(root)), sha256=digest(path), source_input=old_case['path'],
                          source_native=str((source_dir / 'native.npz').relative_to(P3E)),
                          source_joint=str((source_dir / 'combo.npz').relative_to(P3E))))
    parents = [{k: p[k] for k in ('name', 'prompt', 'policy_identity', 'head')} for p in old['parents']]
    config = dict(schema='local.moe_crossover.v1', utc=now(), parents=parents, cases=cases,
                  labels=LABELS, main_labels=MAIN_LABELS, expected_model_calls=135,
                  source_run=str(P3E), source_verification_sha256=digest(P3E / 'verification.json'),
                  source_hashes=sources, shared_metadata=metadata, new_environment_actions=0,
                  simulator_states_read=0, task_labels_used=False, training=False, weights_changed=False,
                  cohort='15 paired development states from five parents, no new selection',
                  output_scope=dict(layers=[12, 13, 14, 15], denoise_steps=10, action_tokens=list(range(1, 11)),
                                    tensor='routed plus shared, before residual addition'),
                  unbiased_gate_semantics='original gate function at current input; downstream IDs may change')
    write(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), cases=len(cases), model_calls=135)), flush=True)


def check_config(root):
    config = read(root / 'config.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert digest(P3E / 'verification.json') == config['source_verification_sha256']
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    for case in config['cases']:
        assert digest(root / case['path']) == case['sha256']
    assert len(config['cases']) == 15 and config['labels'] == list(LABELS)
    return config


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    attempts = completed = 0
    started = time.monotonic()
    try:
        wrapped, loaded = load_isolated()
        modified = MatrixPolicy(wrapped.policy)
        model = wrapped.policy._policy.model
        versions = parameter_versions(model)
        weights_before = parameter_digest(model)
        assert not any(m.training for m in model.modules())
        assert not any(p.requires_grad for p in model.parameters())
        write(root / 'model-load.json', dict(loaded, parameter_sha256=weights_before))
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6])
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as log:
            for case in config['cases']:
                parent, = [p for p in config['parents'] if p['name'] == case['parent']]
                check_identity(loaded['metadata'], parent['policy_identity'])
                with np.load(root / case['path'], allow_pickle=False) as saved:
                    inputs = {key: saved[key] for key in INPUT_KEYS}
                bias = validate_bias(inputs['request_bias'])
                monitor = V82Monitor()
                for route in inputs['prefix_hb']:
                    monitor.update(route)
                history = hashlib.sha256(pickle.dumps(monitor, protocol=5)).hexdigest()
                request = {k: v for k, v in inputs.items() if k.startswith('observation/')}
                request.update(prompt=parent['prompt'], **{'flow/noise': inputs['request_noise'],
                               'routing/capture': True, CAPTURE_KEY: True})
                retained = {}
                for label in LABELS:
                    if attempts >= config['expected_model_calls']:
                        raise RuntimeError('Model call budget exceeded')
                    actual_bias = bias if label in BIASED else np.zeros(BIAS_SHAPE, np.float32)
                    donor_name = DONORS.get(label)
                    donor = None if donor_name is None else retained[donor_name]['mechanism/total']
                    event = dict(ordinal=attempts, parent=parent['name'], phase=case['phase'], query=case['query'],
                                 label=label, biased=label in BIASED, donor=donor_name,
                                 noise_sha256=array_digest(inputs['request_noise']), bias_sha256=array_digest(actual_bias),
                                 donor_sha256=None if donor is None else array_digest(donor), history_sha256=history)
                    log.write(json.dumps(dict(event, event='request')) + '\n')
                    log.flush()
                    attempts += 1
                    response, resource = crossover_infer(modified if label in BIASED else wrapped,
                        dict(request, **{BIAS_KEY: bias}) if label in BIASED else request, donor)
                    completed += 1
                    assert parameter_versions(model) == versions, 'Parameter changed during inference'
                    if label in MAIN_LABELS:
                        retained[label] = response
                    if label in ('native', 'joint'):
                        with np.load(P3E / case['source_' + label], allow_pickle=False) as saved:
                            assert_equal(response, saved, FULL_KEYS + TRACE_KEYS +
                                         (LOGIT_KEYS if label == 'joint' else ()), 'P3e donor reproduction')
                    baseline = retained[expected_computation(label)]
                    assert_equal(response, baseline, TRACE_KEYS + ('actions',), 'effective computation matching')
                    if label in CONTROLS:
                        control = retained[CONTROLS[label]]
                        assert_equal(response, control, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',) +
                                     (LOGIT_KEYS if label in BIASED else ()), 'identity/repeat/withdrawal')
                    raw, effective = response['crossover/raw_total'], response['mechanism/total']
                    same(raw[~SITE_MASK], effective[~SITE_MASK], 'outside replay scope')
                    if donor is None:
                        same(raw, effective, 'unpatched total')
                    else:
                        same(effective[SITE_MASK], donor[SITE_MASK], 'exact replay source')
                    if label in ('route_only', 'repeat_route'):
                        same(response[NATIVE_PROBS], retained['native'][NATIVE_PROBS], 'native input, biased gate')
                    if label in ('output_only', 'repeat_output'):
                        same(response[EFFECTIVE_PROBS], retained['joint'][NATIVE_PROBS], 'joint input, unbiased gate')
                    decision = evaluate(monitor, retained['native'], response, 'mobility_balance', std)
                    assert hashlib.sha256(pickle.dumps(monitor, protocol=5)).hexdigest() == history
                    path = (root / case['path']).parent / (label + '.npz')
                    with path.open('xb') as stream:
                        np.savez_compressed(stream, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                                            request_noise=inputs['request_noise'], request_bias=actual_bias)
                    flags = {k.removeprefix('diagnostic/'): response[k] for k in
                             ('diagnostic/gpu_exact_logit_softmax', 'diagnostic/logit_dtypes', 'diagnostic/probability_dtypes')
                             if k in response}
                    row = jsonable(dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path),
                        resource=resource, decision=decision, **flags,
                        top4_changed_fraction=set_change(response[EFFECTIVE_IDS_KEY], retained['native'][EFFECTIVE_IDS_KEY]),
                        raw_to_effective=comparison(effective[SITE_MASK], raw[SITE_MASK]),
                        hooks_removed=True, parameter_versions_unchanged=True, history_unchanged=True))
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(dict(parent=parent['name'], phase=case['phase'], label=label, calls=completed,
                        screen=decision['accepted'], action_rms=decision['normalized_action_rms'],
                        elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                del retained, response, donor, baseline
        assert attempts == completed == config['expected_model_calls']
        weights_after = parameter_digest(model)
        assert weights_before == weights_after and parameter_versions(model) == versions
        check_config(root)
        write(root / 'collection.json', dict(passed=True, model_attempts=attempts, model_completed=completed,
            new_environment_actions=0, simulator_states_read=0, parameter_sha256_before=weights_before,
            parameter_sha256_after=weights_after, parameter_versions_unchanged=True,
            elapsed_seconds=time.monotonic() - started, finished_utc=now()))
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
