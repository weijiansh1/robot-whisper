"""Preserve the failed action-only pilot and run complete-boundary crossover."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import tempfile
import time
import traceback

import numpy as np

from run_crossover_experiment import (BASE, P3E, NEW_SOURCES, INPUT_KEYS, TRACE_KEYS, check_config,
    read, write, digest, array_digest, same, jsonable, now, load_isolated, MatrixPolicy,
    check_identity, FULL_KEYS, LOGIT_KEYS, V82Monitor, CAPTURE_KEY, BIAS_KEY, evaluate,
    NATIVE_PROBS, EFFECTIVE_PROBS, EFFECTIVE_IDS_KEY, LABELS, MAIN_LABELS, BIASED,
    DONORS, CONTROLS, BIAS_SHAPE, SITE_MASK, validate_bias, expected_computation, set_change,
    parameter_versions, parameter_digest, PolicyClient, comparison)
from crossover_capture import crossover_infer
from crossover_capture_v2 import whole_output_infer, OUTPUT_MASK

PILOT = BASE / 'runs/p3f-crossover-20260915-w40v879v'
V2_SOURCES = ('P3F_CROSSOVER_V2_PLAN.zh.md', 'crossover_capture_v2.py',
              'run_crossover_experiment_v2.py', 'test_crossover_v2.py')


def prepare():
    old = check_config(PILOT)
    assert read(PILOT / 'failure.json')['model_completed'] == 5
    sources = old['source_hashes'].copy()
    sources.update({str(BASE / name): digest(BASE / name) for name in V2_SOURCES})
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    for key in ('server_instance_id', 'server_pid', 'server_started_unix_ns'):
        assert metadata[key] == old['shared_metadata'][key]
    root = Path(tempfile.mkdtemp(prefix='p3f-crossover-v2-20260915-', dir=BASE / 'runs'))
    cases = []
    for case in old['cases']:
        with np.load(PILOT / case['path'], allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in INPUT_KEYS}
        path = root / case['path']
        path.parent.mkdir(parents=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, **inputs)
        cases.append(dict(case, sha256=digest(path)))
    config = dict(old, schema='local.moe_crossover.v2', utc=now(), cases=cases, source_hashes=sources,
        shared_metadata=metadata, diagnostic_model_calls=2, expected_total_model_calls=137,
        prior_pilot_model_calls=5, total_user_request_model_calls=142, prior_pilot=str(PILOT),
        prior_pilot_files={str(p.relative_to(PILOT)): digest(p) for p in sorted(PILOT.rglob('*')) if p.is_file()},
        output_scope=dict(layers=[12, 13, 14, 15], denoise_steps=10, tokens=list(range(11)),
                          tensor='complete routed plus shared output, before residual addition'),
        gate_scope=dict(layers=[12, 13, 14, 15], denoise_steps=10, tokens=list(range(1, 11)), max_abs=.30))
    write(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(dict(run=str(root), main_calls=135, scope_diagnostics=2, prior_calls=5, total_budget=142)), flush=True)


def make_request(inputs, parent):
    request = {k: v for k, v in inputs.items() if k.startswith('observation/')}
    request.update(prompt=parent['prompt'], **{'flow/noise': inputs['request_noise'], 'routing/capture': True, CAPTURE_KEY: True})
    return request


def persist_response(path, response, inputs, bias):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                            request_noise=inputs['request_noise'], request_bias=bias)


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    attempts = completed = 0
    started = time.monotonic()
    try:
        wrapped, loaded = load_isolated()
        modified, model = MatrixPolicy(wrapped.policy), wrapped.policy._policy.model
        versions, weights_before = parameter_versions(model), parameter_digest(model)
        assert not any(m.training for m in model.modules()) and not any(p.requires_grad for p in model.parameters())
        write(root / 'model-load.json', dict(loaded, parameter_sha256=weights_before))
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6])
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        first = config['cases'][0]
        parent, = [p for p in config['parents'] if p['name'] == first['parent']]
        check_identity(loaded['metadata'], parent['policy_identity'])
        with np.load(root / first['path'], allow_pickle=False) as saved:
            inputs = {k: saved[k] for k in INPUT_KEYS}
        request = make_request(inputs, parent)
        diagnostics = root / 'scope-diagnostics'
        diagnostics.mkdir()
        with (diagnostics / 'calls.jsonl').open('x') as log:
            for label, source, biased in (('route_only', 'native', True), ('output_only', 'joint', False)):
                with np.load(P3E / first['source_' + source], allow_pickle=False) as saved:
                    donor = saved['mechanism/total']
                event = dict(ordinal=attempts, label=label, donor_source=first['source_' + source],
                    donor_sha256=array_digest(donor), noise_sha256=array_digest(inputs['request_noise']))
                log.write(json.dumps(dict(event, event='request')) + '\n')
                log.flush()
                attempts += 1
                response, resource = crossover_infer(modified if biased else wrapped,
                    dict(request, **{BIAS_KEY: inputs['request_bias']}) if biased else request, donor)
                completed += 1
                path = diagnostics / (label + '.npz')
                persist_response(path, response, inputs, inputs['request_bias'] if biased else np.zeros(BIAS_SHAPE, np.float32))
                log.write(json.dumps(dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path), resource=resource)) + '\n')
                log.flush()
                assert parameter_versions(model) == versions
                print(json.dumps(dict(scope_diagnostic=label, actual_calls=completed)), flush=True)
        del response, donor
        with (root / 'calls.jsonl').open('x') as log:
            for case in config['cases']:
                parent, = [p for p in config['parents'] if p['name'] == case['parent']]
                check_identity(loaded['metadata'], parent['policy_identity'])
                with np.load(root / case['path'], allow_pickle=False) as saved:
                    inputs = {k: saved[k] for k in INPUT_KEYS}
                bias = validate_bias(inputs['request_bias'])
                monitor = V82Monitor()
                for route in inputs['prefix_hb']:
                    monitor.update(route)
                history = hashlib.sha256(pickle.dumps(monitor, protocol=5)).hexdigest()
                request, retained = make_request(inputs, parent), {}
                for label in LABELS:
                    if attempts >= config['expected_total_model_calls']:
                        raise RuntimeError('Model call budget exceeded')
                    actual_bias = bias if label in BIASED else np.zeros(BIAS_SHAPE, np.float32)
                    donor_name = DONORS.get(label)
                    donor = None if donor_name is None else retained[donor_name]['mechanism/total']
                    event = dict(ordinal=attempts - 2, actual_call_ordinal=attempts, parent=parent['name'], phase=case['phase'],
                        query=case['query'], label=label, biased=label in BIASED, donor=donor_name,
                        noise_sha256=array_digest(inputs['request_noise']), bias_sha256=array_digest(actual_bias),
                        donor_sha256=None if donor is None else array_digest(donor), history_sha256=history)
                    log.write(json.dumps(dict(event, event='request')) + '\n')
                    log.flush()
                    attempts += 1
                    response, resource = whole_output_infer(modified if label in BIASED else wrapped,
                        dict(request, **{BIAS_KEY: bias}) if label in BIASED else request, donor)
                    completed += 1
                    path = (root / case['path']).parent / (label + '.npz')
                    persist_response(path, response, inputs, actual_bias)
                    assert parameter_versions(model) == versions
                    if label in MAIN_LABELS:
                        retained[label] = response
                    if label in ('native', 'joint'):
                        with np.load(P3E / case['source_' + label], allow_pickle=False) as saved:
                            for key in FULL_KEYS + TRACE_KEYS + (LOGIT_KEYS if label == 'joint' else ()):
                                same(response[key], saved[key], 'P3e exact donor: ' + key)
                    baseline = retained[expected_computation(label)]
                    for key in TRACE_KEYS + ('actions',):
                        same(response[key], baseline[key], 'effective computation matching: ' + key)
                    if label in CONTROLS:
                        control = retained[CONTROLS[label]]
                        for key in FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',) + (LOGIT_KEYS if label in BIASED else ()):
                            same(response[key], control[key], 'identity/repeat/withdrawal: ' + key)
                    raw, effective = response['crossover/raw_total'], response['mechanism/total']
                    same(raw[~OUTPUT_MASK], effective[~OUTPUT_MASK], 'outside whole-output replay scope')
                    same(raw, effective, 'unpatched total') if donor is None else same(effective[OUTPUT_MASK], donor[OUTPUT_MASK], 'exact replay source')
                    if label in ('route_only', 'repeat_route'):
                        same(response[NATIVE_PROBS], retained['native'][NATIVE_PROBS], 'native hidden input, biased gate')
                    if label in ('output_only', 'repeat_output'):
                        same(response[EFFECTIVE_PROBS], retained['joint'][NATIVE_PROBS], 'joint hidden input, unbiased gate')
                    decision = evaluate(monitor, retained['native'], response, 'mobility_balance', std)
                    assert hashlib.sha256(pickle.dumps(monitor, protocol=5)).hexdigest() == history
                    flags = {k.removeprefix('diagnostic/'): response[k] for k in
                             ('diagnostic/gpu_exact_logit_softmax', 'diagnostic/logit_dtypes', 'diagnostic/probability_dtypes') if k in response}
                    row = jsonable(dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path),
                        resource=resource, decision=decision, **flags,
                        top4_changed_fraction=set_change(response[EFFECTIVE_IDS_KEY], retained['native'][EFFECTIVE_IDS_KEY]),
                        raw_to_effective=comparison(effective[SITE_MASK], raw[SITE_MASK]),
                        hooks_removed=True, parameter_versions_unchanged=True, history_unchanged=True))
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(dict(parent=parent['name'], phase=case['phase'], label=label, main_calls=completed - 2,
                        actual_calls=completed, screen=decision['accepted'], action_rms=decision['normalized_action_rms'],
                        elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                del retained, response, donor, baseline
        assert attempts == completed == config['expected_total_model_calls']
        weights_after = parameter_digest(model)
        assert weights_before == weights_after and parameter_versions(model) == versions
        check_config(root)
        write(root / 'collection.json', dict(passed=True, model_attempts=attempts, model_completed=completed,
            main_model_calls=135, scope_diagnostic_model_calls=2, prior_pilot_model_calls=5, total_user_request_model_calls=142,
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
