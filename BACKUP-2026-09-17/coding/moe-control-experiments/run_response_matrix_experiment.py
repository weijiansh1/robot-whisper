"""Collect frozen multi-component gate responses without environment rollouts."""

import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np

from gate_runtime import BASE, infer_isolated, load_isolated
from diagnose_coverage_logits import LogitPolicy
from run_gate_experiment import FULL_KEYS, assert_equal, check_identity, source_for
from run_online_experiment import jsonable, now
from audit_online_experiment import read, write, digest, array_digest, same
from head_control_protocol import prefix_monitor
from response_matrix_protocol import (COMPONENTS, MASK, MAX_ABS, OPERATORS, TARGETS,
                                       bias_bank, measure, screen, specifications)
from himoe_libero_bridge.client import PolicyClient
from collection_routes import CAPTURE_KEY, PROBS_KEY, EFFECTIVE_IDS_KEY, EFFECTIVE_WEIGHTS_KEY
from scope_bias_control import BIAS_KEY
from v8_feature_control import EFFECTIVE_PROBS

P3B = BASE / 'runs/p3b-head-control-20260915-s39ow9qi'
EVALUATION = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')


class MatrixPolicy(LogitPolicy):
    def infer(self, observation):
        bias = np.asarray(observation[BIAS_KEY])
        if bias.shape != (8, 10, 11, 32) or not np.isfinite(bias).all() or np.max(np.abs(bias)) > MAX_ABS + 1e-7:
            raise ValueError('Matrix intervention exceeds local bound')
        result = super().infer(observation)
        for full, wire in ((EFFECTIVE_IDS_KEY, 'routing/expert_ids'),
                           (EFFECTIVE_WEIGHTS_KEY, 'routing/expert_weights')):
            same(result[full][:, :, 1:].transpose(1, 0, 2, 3), result[wire], 'live matrix dispatch')
        return result


def frozen_files():
    result = read(P3B / 'config.json')['source_hashes'].copy()
    for path, expected in result.items():
        if digest(path) != expected:
            raise RuntimeError('Prior frozen source changed: ' + path)
    for name in ('P3C_RESPONSE_MATRIX_PLAN.zh.md', 'response_matrix_protocol.py',
                 'run_response_matrix_experiment.py', 'test_response_matrix_protocol.py',
                 'diagnose_coverage_logits.py', 'audit_online_experiment.py', 'run_gate_experiment.py'):
        result[str(BASE / name)] = digest(BASE / name)
    return result


def prepare():
    old_config, old_verification = read(P3B / 'config.json'), read(P3B / 'verification.json')
    with PolicyClient('127.0.0.1', 9500, inference_timeout=60) as client:
        metadata = client.metadata
    root = Path(tempfile.mkdtemp(prefix='p3c-response-matrix-20260915-', dir=BASE / 'runs'))
    parents = []
    for original in old_config['parents']:
        parent = original.copy()
        manifest, _, request = source_for(parent)
        check_identity(metadata, manifest['policy_identity'])
        source_hb = EVALUATION / parent['name'] / 'full-hb-routes.npz'
        assert digest(source_hb) == parent['hb_sha256']
        with np.load(source_hb, allow_pickle=False) as saved:
            reference = saved['hb_router_probs']
        prefix_monitor(reference, parent['query'], parent['first_alarm'], parent['head'])
        pool_path = P3B / parent['name'] / 'pool.npz'
        assert digest(pool_path) == old_verification['files'][str(pool_path.relative_to(P3B))]
        with np.load(pool_path, allow_pickle=False) as pool:
            same(pool['hb'][0], reference[parent['query']], 'P3b native source')
            same(pool['noises'][0], request['flow/noise'], 'P3b source noise')
            for field, key in (('image', 'observation/image'), ('wrist_image', 'observation/wrist_image'),
                               ('state', 'observation/state')):
                same(pool[field], request[key], 'P3b source input')
            native_actions = pool['actions'][0]
        bank, energy = bias_bank(reference[parent['query']], reference[parent['query'] - 1], parent)
        directory = root / 'states' / parent['name']
        directory.mkdir(parents=True)
        input_path, bank_path = directory / 'input.npz', directory / 'bias-bank.npz'
        with input_path.open('xb') as stream:
            np.savez_compressed(stream, **{k: v for k, v in request.items() if isinstance(v, np.ndarray)},
                prefix_hb=reference[:parent['query']], native_hb=reference[parent['query']], native_actions=native_actions)
        with bank_path.open('xb') as stream:
            np.savez_compressed(stream, biases=bank)
        parent.update(prompt=manifest['prompt'], policy_identity=manifest['policy_identity'], energy=energy,
                      input_path=str(input_path.relative_to(root)), input_sha256=digest(input_path),
                      bank_path=str(bank_path.relative_to(root)), bank_sha256=digest(bank_path),
                      p3b_pool_sha256=digest(pool_path))
        parents.append(parent)
    config = dict(schema='local.moe_response_matrix.v1', utc=now(), parents=parents,
                  source_hashes=frozen_files(), source_run=str(P3B), source_config_sha256=digest(P3B / 'config.json'),
                  components=COMPONENTS, operators=OPERATORS, targets=TARGETS, probes=specifications(),
                  expected_model_calls=175, expected_unique_probes=150, shared_metadata=metadata,
                  new_environment_actions=0, training=False, weights_changed=False,
                  actual_gate_intervention=True, cohort='five previously inspected development alarm states')
    write(root / 'config.json', jsonable(config))
    with (root / 'config.sha256').open('x') as stream:
        stream.write(digest(root / 'config.json') + '\n')
    print(json.dumps(jsonable(dict(run=str(root), model_calls=175, probes=150,
                                   energies=[p['energy'] for p in parents]))), flush=True)


def check_config(root):
    config = read(root / 'config.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert config['source_hashes'] == frozen_files()
    assert digest(P3B / 'config.json') == config['source_config_sha256']
    for parent in config['parents']:
        for field in ('input', 'bank'):
            assert digest(root / parent[field + '_path']) == parent[field + '_sha256']
        assert digest(Path(parent['source']) / 'episode-trace.json') == parent['manifest_sha256']
        assert digest(Path(parent['source']) / 'episode-trace.npz') == parent['trace_sha256']
        assert digest(EVALUATION / parent['name'] / 'full-hb-routes.npz') == parent['hb_sha256']
        assert digest(P3B / parent['name'] / 'pool.npz') == parent['p3b_pool_sha256']
    return config


def collect(root):
    config = check_config(root)
    write(root / 'started.json', dict(utc=now()))
    started, attempts, completed, rows = time.monotonic(), 0, 0, []
    try:
        wrapped, loaded = load_isolated()
        matrix_policy = MatrixPolicy(wrapped.policy)
        write(root / 'model-load.json', loaded)
        std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], float)
        print(json.dumps(dict(model_loaded=True, seconds=loaded['load_seconds'])), flush=True)
        with (root / 'calls.jsonl').open('x') as log:
            for parent in config['parents']:
                check_identity(loaded['metadata'], parent['policy_identity'])
                directory = root / 'states' / parent['name']
                with np.load(root / parent['input_path'], allow_pickle=False) as saved:
                    inputs = {k: saved[k] for k in saved.files}
                with np.load(root / parent['bank_path'], allow_pickle=False) as saved:
                    bank = saved['biases']
                prefix = prefix_monitor(inputs['prefix_hb'], parent['query'], parent['first_alarm'], parent['head'])
                request = {k: v for k, v in inputs.items() if k.startswith('observation/') or k == 'flow/noise'}
                request.update(prompt=parent['prompt'], **{'routing/capture': True, CAPTURE_KEY: True})
                baseline_response, baseline_measure, baseline_fp32 = None, None, None
                retained = {}

                def infer(label, kind, bias=None, spec=None):
                    nonlocal attempts, completed
                    actual_bias = np.zeros((8, 10, 11, 32), np.float32) if bias is None else bias
                    event = dict(ordinal=attempts, parent=parent['name'], query=parent['query'], label=label,
                                 kind=kind, spec=spec, noise_sha256=array_digest(request['flow/noise']),
                                 bias_sha256=array_digest(actual_bias))
                    log.write(json.dumps(dict(event, event='request')) + '\n')
                    log.flush()
                    attempts += 1
                    response, resource = infer_isolated(wrapped if bias is None else matrix_policy,
                        request if bias is None else dict(request, **{BIAS_KEY: bias}))
                    completed += 1
                    if not response['gate_probe/exact_dtype_audit']:
                        raise RuntimeError('Missing actual gate audit')
                    if bias is not None and not response['diagnostic/gpu_exact_logit_softmax']:
                        raise RuntimeError('Missing actual logit audit')
                    path = directory / (label + '.npz')
                    with path.open('xb') as stream:
                        np.savez_compressed(stream, **{k: v for k, v in response.items() if isinstance(v, np.ndarray)},
                            request_bias=actual_bias, request_noise=request['flow/noise'])
                    effect = measure(prefix, response[EFFECTIVE_PROBS].astype(np.float16))
                    full = measure(prefix, response[EFFECTIVE_PROBS])
                    base_response = response if baseline_response is None else baseline_response
                    base = effect if baseline_measure is None else baseline_measure
                    base_full = full if baseline_fp32 is None else baseline_fp32
                    delta = effect['normalized'] - base['normalized']
                    instant_delta = (effect['instantaneous'] - base['instantaneous']) / effect['margins']
                    action_rms = float(np.sqrt(np.mean(np.square(
                        (response['actions'][:, :6].astype(float) - base_response['actions'][:, :6]) / std))))
                    gripper = int(np.count_nonzero((response['actions'][:, 6] >= 0) != (base_response['actions'][:, 6] >= 0)))
                    ids, base_ids = response[EFFECTIVE_IDS_KEY], base_response[EFFECTIVE_IDS_KEY]
                    changed = np.any(np.sort(ids, axis=-1) != np.sort(base_ids, axis=-1), axis=-1)
                    row = dict(event, event='response', path=str(path.relative_to(root)), sha256=digest(path),
                               resource=resource, measurement=effect, fp32_measurement=full,
                               delta=delta, instantaneous_delta=instant_delta,
                               fp32_delta=full['normalized'] - base_full['normalized'],
                               normalized_action_rms=action_rms, gripper_sign_changes=gripper,
                               actions_changed=bool(np.any(response['actions'] != base_response['actions'])),
                               back_action_set_change_fraction=float(changed[4:, :, 1:].mean()),
                               request_bias_l2=float(np.linalg.norm(actual_bias.astype(float))),
                               request_bias_peak=float(np.max(np.abs(actual_bias))),
                               score_screen=screen(delta, spec['operator'], action_rms, gripper) if spec else False,
                               instantaneous_screen=screen(instant_delta, spec['operator'], action_rms, gripper) if spec else False)
                    if bias is not None:
                        applied = response['diagnostic/applied_bias_fp32'].astype(float)
                        effective_delta = (response['diagnostic/effective_logits_fp32'].astype(float) -
                                           response['diagnostic/native_logits_fp32'].astype(float))
                        row.update(logit_dtypes=response['diagnostic/logit_dtypes'],
                                   probability_dtypes=response['diagnostic/probability_dtypes'],
                                   gpu_exact_logit_softmax=True, applied_bias_l2=float(np.linalg.norm(applied)),
                                   effective_logit_delta_l2=float(np.linalg.norm(effective_delta)),
                                   requested_effective_max_error=float(np.max(np.abs(effective_delta - actual_bias))))
                    row = jsonable(row)
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    rows.append(row)
                    return response, effect, full

                baseline_response, baseline_measure, baseline_fp32 = infer('native', 'native')
                same(baseline_response['actions'], inputs['native_actions'], 'native source actions')
                same(baseline_response[PROBS_KEY], inputs['native_hb'], 'native source HB')
                zero, _, _ = infer('zero', 'zero', np.zeros_like(bank[0]))
                assert_equal(zero, baseline_response, FULL_KEYS, 'zero-bias matrix output')
                for index, spec in enumerate(config['probes']):
                    response, _, _ = infer(spec['label'], 'probe', bank[index], spec)
                    if spec['label'] in ('mobility-high-plus', 'balance-high-plus'):
                        retained[spec['label']] = response
                    if (index + 1) % 10 == 0:
                        print(json.dumps(dict(parent=parent['name'], probes_done=index + 1, calls=completed,
                            elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                for label in ('mobility-high-plus', 'balance-high-plus'):
                    index = next(i for i, spec in enumerate(config['probes']) if spec['label'] == label)
                    repeated, _, _ = infer('repeat-' + label, 'repeat', bank[index], config['probes'][index])
                    assert_equal(repeated, retained[label], FULL_KEYS +
                        ('diagnostic/native_logits_fp32', 'diagnostic/effective_logits_fp32', 'diagnostic/applied_bias_fp32'),
                        'repeated matrix probe')
                post, _, _ = infer('post', 'post')
                assert_equal(post, baseline_response, FULL_KEYS, 'matrix cleanup')
                print(json.dumps(dict(parent_complete=parent['name'], calls=completed, high_plus_deltas={
                    r['spec']['operator']: r['delta'] for r in rows if r['parent'] == parent['name'] and
                    r['kind'] == 'probe' and not r['spec']['randomized'] and r['spec']['dose'] == 'high' and r['spec']['sign'] == 1})), flush=True)
        check_config(root)
        assert attempts == completed == config['expected_model_calls']
        write(root / 'collection.json', dict(passed=True, rows=rows, model_attempts=attempts, model_completed=completed,
            new_environment_actions=0, elapsed_seconds=time.monotonic() - started, finished_utc=now()))
        print(json.dumps(dict(completed=True, calls=completed, run=str(root))), flush=True)
    except BaseException as error:
        write(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(),
            model_attempts=attempts, model_completed=completed, completed_rows=len(rows), utc=now()))
        raise


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
