"""Reconstruct direction budgets, actual gate arithmetic and five responses."""

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from audit_online_experiment import V82Monitor, digest, array_digest, read, write, records, same
from v8_feature_control import make_bias
from intrinsic_guard_monitor import intrinsic_score_arrays

SHAPE = (8, 10, 11, 32)
OPS = ('mobility', 'balance', 'curvature', 'mobility_balance', 'mobility_curvature')
TARGETS = dict(mobility=[0], balance=[3], curvature=[4], mobility_balance=[0, 3], mobility_curvature=[0, 4])
KEYS = ('actions', 'routing/expert_ids', 'routing/expert_weights',
        'v8_control/native_probs_fp32', 'v8_control/effective_probs_fp32',
        'collection/hb_native_ids', 'collection/hb_native_weights',
        'collection/hb_effective_ids', 'collection/hb_effective_weights', 'collection/as_probs')
PROFILE = Path('/data/coding/robot-whisper-0909/moe-trap-control/design/frozen_alarm_comparison_20260908/profiles/parameters.json')


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k] for k in saved.files}


def norm(value):
    return float(np.sqrt(np.square(np.asarray(value, np.float64)).sum()))


def independent_bank(shadow, previous, parent, specs):
    units = {}
    for operator in OPS[:3]:
        value = make_bias(shadow, previous, operator, 1., 0).astype(np.float64)
        assert norm(value) > 1e-12
        units[operator] = value / norm(value)
    for operator, second in (('mobility_balance', 'balance'), ('mobility_curvature', 'curvature')):
        value = units['mobility'] + units[second]
        units[operator] = value / norm(value)
    energy = min(.1 * np.sqrt(12800), .3 / max(np.abs(v).max() for v in units.values()))
    result = []
    for spec in specs:
        value = units[spec['operator']]
        if spec['randomized']:
            seed = [2026091503, 0 if parent['benchmark'] == 'plus' else 1, parent['base_task_id'],
                    parent['init_state_id'], parent['query'], OPS.index(spec['operator'])]
            order = np.argsort(np.random.default_rng(np.random.SeedSequence(seed)).random(SHAPE), axis=-1)
            value = np.take_along_axis(value, order, axis=-1)
        result.append((value * energy * spec['fraction'] * spec['sign']).astype(np.float32))
    return np.stack(result), float(energy)


def measurement(prefix, probabilities):
    monitor = copy.deepcopy(prefix)
    status = monitor.update(probabilities)
    legacy = read(PROFILE)['legacy']
    threshold = np.array([legacy['v7'][name + '_threshold'] for name in
                          ('freeze', 'acceleration', 'periodicity')] +
                         [-legacy['v8_thresholds']['frontback_flowpath'], legacy['v8_thresholds']['curvature_3step']])
    margins = np.r_[threshold[:3] * .2, .1, threshold[4] - .2]
    threshold[3:] -= .0015 * status['query']
    scores = np.array([status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']])
    v7 = monitor.v7
    streams = intrinsic_score_arrays(np.array(v7._mobility_history, np.float32)[None],
        np.array(v7._acceleration_history, np.float32)[None], np.array(v7._periodicity_history, np.float32)[None],
        v7.profile.periodicity_scale)
    raw = np.asarray(monitor.raw, np.float32)
    curvature = np.log(np.maximum(raw[-1, 1], 1e-12) / np.maximum(raw[1:5, 1].mean(dtype=np.float32), 1e-12))
    instant = np.array([streams[name][0, -1] for name in ('freeze_raw', 'acceleration_raw', 'periodicity_raw')] +
                       [raw[-1, 0], curvature], np.float64)
    return dict(status=status, scores=scores, normalized=(scores - threshold) / margins,
                thresholds=threshold, margins=margins, instantaneous=instant)


def passes(delta, operator, rms, gripper):
    target = TARGETS[operator]
    other = [i for i in range(5) if i not in target]
    return bool(all(delta[i] <= -.05 for i in target) and all(delta[i] <= .05 for i in other)
                and rms <= .05 and gripper == 0)


def audit_gate(raw, row):
    import torch

    assert row['gpu_exact_logit_softmax']
    assert row['logit_dtypes'] == ['torch.bfloat16']
    requested = torch.from_numpy(raw['request_bias']).to(torch.bfloat16)
    same(requested.float().numpy(), raw['diagnostic/applied_bias_fp32'], 'BF16 applied bias')
    before = torch.from_numpy(raw['diagnostic/native_logits_fp32']).to(torch.bfloat16)
    after = before + requested
    same(after.float().numpy(), raw['diagnostic/effective_logits_fp32'], 'actual BF16 logit addition')
    maximum = 0.
    for phase, logits in (('native', before), ('effective', after)):
        probabilities = raw['v8_control/' + phase + '_probs_fp32']
        expected = logits.float().softmax(-1)
        if row['probability_dtypes'] == ['torch.bfloat16']:
            expected = expected.to(torch.bfloat16).float()
        else:
            assert row['probability_dtypes'] == ['torch.float32']
        error = float(np.max(np.abs(probabilities - expected.numpy())))
        assert error < 1.3e-7, error
        maximum = max(maximum, error)
        ids = raw['collection/hb_' + phase + '_ids'].astype(np.int64)
        weights = raw['collection/hb_' + phase + '_weights']
        assert np.all((ids >= 0) & (ids < 32)) and not np.any(np.diff(np.sort(ids, axis=-1), axis=-1) == 0)
        selected = np.take_along_axis(probabilities, ids, axis=-1)
        assert np.all(selected.min(-1) >= np.sort(probabilities, axis=-1)[..., -4])
        np.testing.assert_allclose(weights, selected / selected.sum(-1, keepdims=True), atol=1.3e-7, rtol=0)
    touched = np.any(raw['request_bias'] != 0, axis=-1)
    for field in ('ids', 'weights'):
        same(raw['collection/hb_native_' + field][~touched], raw['collection/hb_effective_' + field][~touched], 'off-scope gate return')
        same(raw['collection/hb_effective_' + field][:, :, 1:].transpose(1, 0, 2, 3),
             raw['routing/expert_' + field], 'wire and effective dispatch')
    return maximum


def audit(root):
    import torch

    torch.set_num_threads(1)
    config, collection = read(root / 'config.json'), read(root / 'collection.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert collection['passed'] and collection['new_environment_actions'] == 0
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    events = records(root / 'calls.jsonl')
    requests, responses = events[::2], events[1::2]
    assert len(requests) == len(responses) == collection['model_attempts'] == collection['model_completed'] == 175
    assert responses == collection['rows']
    for i, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == i
        for field in ('parent', 'label', 'kind', 'spec', 'query', 'noise_sha256', 'bias_sha256'):
            assert request[field] == response[field]
    std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], float)
    mask = np.zeros(SHAPE, bool)
    mask[4:, :, 1:] = True
    checked, max_softmax_error, probes, energy_rows = 0, 0., [], []
    for parent in config['parents']:
        source = Path(parent['source'])
        assert digest(source / 'episode-trace.json') == parent['manifest_sha256']
        assert digest(source / 'episode-trace.npz') == parent['trace_sha256']
        input_path, bank_path = root / parent['input_path'], root / parent['bank_path']
        assert digest(input_path) == parent['input_sha256'] and digest(bank_path) == parent['bank_sha256']
        inputs, bank = load_arrays(input_path), load_arrays(bank_path)['biases']
        source_hb_path = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z') / parent['name'] / 'full-hb-routes.npz'
        assert digest(source_hb_path) == parent['hb_sha256']
        reference = load_arrays(source_hb_path)['hb_router_probs']
        same(inputs['prefix_hb'], reference[:parent['query']], 'source HB prefix')
        same(inputs['native_hb'], reference[parent['query']], 'source HB query')
        old_pool = Path(config['source_run']) / parent['name'] / 'pool.npz'
        assert digest(old_pool) == parent['p3b_pool_sha256']
        pool = load_arrays(old_pool)
        for field, key in (('image', 'observation/image'), ('wrist_image', 'observation/wrist_image'), ('state', 'observation/state')):
            same(pool[field], inputs[key], 'same P3b observation')
        same(pool['noises'][0], inputs['flow/noise'], 'same P3b noise')
        same(pool['actions'][0], inputs['native_actions'], 'same P3b action')
        expected, energy = independent_bank(inputs['native_hb'], inputs['prefix_hb'][-1], parent, config['probes'])
        np.testing.assert_allclose(bank, expected, rtol=1e-7, atol=2e-8)
        assert np.max(np.abs(bank)) <= .3000001 and np.all(bank[:, ~mask] == 0)
        np.testing.assert_allclose(energy, parent['energy']['high_energy'], rtol=1e-12)
        by_label = {s['label']: b for s, b in zip(config['probes'], bank)}
        for operator in OPS:
            for dose in ('low', 'high'):
                same(np.sort(by_label[operator + '-' + dose + '-plus'], axis=-1),
                     np.sort(by_label['random-' + operator + '-' + dose], axis=-1), 'site-matched random')
                same(-by_label[operator + '-' + dose + '-plus'], by_label[operator + '-' + dose + '-minus'], 'signed pair')
        prefix = V82Monitor()
        for p in inputs['prefix_hb']:
            prefix.update(p)
        assert prefix.first_v82_alarm == parent['first_alarm']
        directory = root / 'states' / parent['name']
        native = load_arrays(directory / 'native.npz')
        same(native['actions'], inputs['native_actions'], 'native model action')
        same(native['v8_control/effective_probs_fp32'].astype(np.float16), inputs['native_hb'], 'native model HB')
        base = measurement(prefix, native['v8_control/effective_probs_fp32'].astype(np.float16))
        base_full = measurement(prefix, native['v8_control/effective_probs_fp32'])
        local = [r for r in responses if r['parent'] == parent['name']]
        assert len(local) == 35
        expected_labels = ['native', 'zero'] + [s['label'] for s in config['probes']] + ['repeat-mobility-high-plus', 'repeat-balance-high-plus', 'post']
        assert [r['label'] for r in local] == expected_labels
        for row in local:
            path = root / row['path']
            assert digest(path) == row['sha256']
            raw = load_arrays(path)
            same(raw['request_noise'], inputs['flow/noise'], 'fixed noise')
            for field, key in (('noise_sha256', 'request_noise'), ('bias_sha256', 'request_bias')):
                assert array_digest(raw[key]) == row[field]
            if row['kind'] in ('probe', 'repeat'):
                spec = row['spec']
                same(raw['request_bias'], by_label[spec['label']], 'executed requested bias')
                np.testing.assert_allclose(norm(raw['request_bias']), energy * spec['fraction'], rtol=1e-7)
            else:
                assert np.all(raw['request_bias'] == 0)
                for key in KEYS:
                    same(raw[key], native[key], 'native/zero/post exact response')
            if row['kind'] == 'repeat':
                prior = load_arrays(directory / (row['spec']['label'] + '.npz'))
                for key in KEYS + ('diagnostic/native_logits_fp32', 'diagnostic/effective_logits_fp32', 'diagnostic/applied_bias_fp32'):
                    same(raw[key], prior[key], 'repeated nonzero response')
            if row['kind'] not in ('native', 'post'):
                max_softmax_error = max(max_softmax_error, audit_gate(raw, row))
                applied = raw['diagnostic/applied_bias_fp32']
                logit_delta = raw['diagnostic/effective_logits_fp32'].astype(float) - raw['diagnostic/native_logits_fp32'].astype(float)
                for key, value in (('applied_bias_l2', norm(applied)), ('effective_logit_delta_l2', norm(logit_delta)),
                                   ('requested_effective_max_error', np.max(np.abs(logit_delta - raw['request_bias'])))):
                    np.testing.assert_allclose(row[key], value, rtol=1e-12, atol=1e-12)
            actual = measurement(prefix, raw['v8_control/effective_probs_fp32'].astype(np.float16))
            full = measurement(prefix, raw['v8_control/effective_probs_fp32'])
            for name, rebuilt in (('measurement', actual), ('fp32_measurement', full)):
                for field in ('scores', 'normalized', 'thresholds', 'margins', 'instantaneous'):
                    np.testing.assert_array_equal(row[name][field], rebuilt[field])
                for field in ('query', 'freeze_alarm', 'turbulence_alarm', 'v82_first', 'v82_alarm'):
                    assert row[name]['status'][field] == rebuilt['status'][field]
            delta = actual['normalized'] - base['normalized']
            instant_delta = (actual['instantaneous'] - base['instantaneous']) / actual['margins']
            np.testing.assert_array_equal(row['delta'], delta)
            np.testing.assert_array_equal(row['instantaneous_delta'], instant_delta)
            np.testing.assert_array_equal(row['fp32_delta'], full['normalized'] - base_full['normalized'])
            motion = (raw['actions'][:, :6].astype(float) - native['actions'][:, :6]) / std
            rms = float(np.sqrt(np.mean(np.square(motion))))
            gripper = int(np.count_nonzero((raw['actions'][:, 6] >= 0) != (native['actions'][:, 6] >= 0)))
            np.testing.assert_allclose(row['normalized_action_rms'], rms, rtol=1e-12)
            assert row['gripper_sign_changes'] == gripper
            assert row['actions_changed'] == bool(np.any(raw['actions'] != native['actions']))
            changed = np.any(np.sort(raw['collection/hb_effective_ids'], axis=-1) !=
                             np.sort(native['collection/hb_effective_ids'], axis=-1), axis=-1)
            assert row['back_action_set_change_fraction'] == float(changed[4:, :, 1:].mean())
            if row['spec']:
                assert row['score_screen'] == passes(delta, row['spec']['operator'], rms, gripper)
                assert row['instantaneous_screen'] == passes(instant_delta, row['spec']['operator'], rms, gripper)
            if row['kind'] == 'probe':
                probes.append(row)
            checked += 1
        energy_rows.append(dict(parent=parent['name'], **parent['energy']))
    assert checked == 175 and len(probes) == 150
    groups = []
    keyed = {(r['parent'], r['spec']['label']): r for r in probes}
    for spec in config['probes']:
        selected = [r for r in probes if r['spec']['label'] == spec['label']]
        assert len(selected) == 5
        deltas, instant = np.array([r['delta'] for r in selected]), np.array([r['instantaneous_delta'] for r in selected])
        target = TARGETS[spec['operator']]
        random_rows = [keyed[(r['parent'], 'random-' + spec['operator'] + '-' + spec['dose'])] for r in selected]
        random_deltas = np.array([r['delta'] for r in random_rows])
        groups.append(dict(spec=spec, parents=5, mean_delta=deltas.mean(0).tolist(), min_delta=deltas.min(0).tolist(),
            max_delta=deltas.max(0).tolist(), mean_instantaneous_delta=instant.mean(0).tolist(),
            target_improved=int(np.all(deltas[:, target] <= -.05, axis=1).sum()),
            instantaneous_target_improved=int(np.all(instant[:, target] <= -.05, axis=1).sum()),
            score_screen=sum(r['score_screen'] for r in selected), instantaneous_screen=sum(r['instantaneous_screen'] for r in selected),
            better_than_matched_random=int(np.all(deltas[:, target] <= random_deltas[:, target] - .05, axis=1).sum()),
            mean_normalized_action_rms=float(np.mean([r['normalized_action_rms'] for r in selected])),
            gripper_changed_parents=sum(r['gripper_sign_changes'] > 0 for r in selected),
            mean_actual_set_change=float(np.mean([r['back_action_set_change_fraction'] for r in selected]))))
    pairs = []
    for operator in OPS:
        positive = [keyed[(p['name'], operator + '-high-plus')] for p in config['parents']]
        negative = [keyed[(p['name'], operator + '-high-minus')] for p in config['parents']]
        target = TARGETS[operator]
        a, b = np.array([r['delta'] for r in positive]), np.array([r['delta'] for r in negative])
        pairs.append(dict(operator=operator, mean_positive_minus_negative=(a - b).mean(0).tolist(),
                          both_signs_reduce_target=int((np.all(a[:, target] <= -.05, axis=1) & np.all(b[:, target] <= -.05, axis=1)).sum()),
                          expected_opposite_target_signs=int((np.all(a[:, target] <= -.05, axis=1) & np.all(b[:, target] >= .05, axis=1)).sum())))
    summary = dict(passed=True, model_calls=175, unique_probes=150, independent_development_states=5,
                   environment_actions=0, elapsed_seconds=collection['elapsed_seconds'],
                   inference_seconds=sum(r['resource']['seconds'] for r in responses),
                   peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
                   groups=groups, signed_pairs=pairs, energies=energy_rows,
                   changed_action_probes=sum(r['actions_changed'] for r in probes),
                   fp16_fp32_effect_max_margin=float(max(np.max(np.abs(np.array(r['delta']) - r['fp32_delta'])) for r in probes)),
                   interpretation='one-query developmental response matrix, not success or recovery evaluation')
    write(root / 'audit.json', dict(passed=True, calls=checked, unique_probes=150, source_files=len(config['source_hashes']),
        max_cpu_softmax_error=max_softmax_error, native_exact_states=5, zero_exact_states=5,
        post_exact_states=5, exact_nonzero_repeats=10, auditor_sha256=digest(Path(__file__))))
    write(root / 'summary.json', summary)
    scalar_rows = []
    for row in probes:
        value = dict(parent=row['parent'], **row['spec'], normalized_action_rms=row['normalized_action_rms'],
            gripper_sign_changes=row['gripper_sign_changes'], score_screen=row['score_screen'],
            instantaneous_screen=row['instantaneous_screen'], request_bias_l2=row['request_bias_l2'],
            effective_logit_delta_l2=row['effective_logit_delta_l2'])
        for i, name in enumerate(config['components']):
            value['delta_' + name] = row['delta'][i]
            value['instant_delta_' + name] = row['instantaneous_delta'][i]
        scalar_rows.append(value)
    with (root / 'response-matrix.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    print(dict(passed=True, calls=checked, probes=150, changed_actions=summary['changed_action_probes'],
        high_plus=[g for g in groups if not g['spec']['randomized'] and g['spec']['dose'] == 'high' and g['spec']['sign'] == 1]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run.resolve())
