"""Recheck raw P3e responses and summarize physical and activation effects."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_mechanism_experiment import check_config, P3D
from audit_online_experiment import read, write, records, digest, array_digest, same
from audit_response_matrix import audit_gate, load_arrays
from audit_gated_rollout_experiment import independently_decide, compare_measurement
from run_online_experiment import jsonable, EVALUATION
from run_gate_experiment import FULL_KEYS, assert_equal
from mechanism_protocol import LABELS, HB_FIELDS, trace_effect, variants, true_intervals
from collection_routes import EFFECTIVE_IDS_KEY
from v82_closed_loop import V82Monitor


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, np.float64) ** 2)))


def physical_summary(root, config):
    result = []
    for parent in config['parents']:
        directory = root / 'physics' / parent['name']
        metadata = read(directory / 'result.json')
        assert metadata['passed'] and metadata['new_environment_actions'] == 0
        streams = {}
        row = dict(parent=parent['name'], alarm_step=10 * parent['first_alarm'], arms={})
        for arm, source in metadata['arms'].items():
            assert digest(source['source']) == source['source_sha256']
            data = load_arrays(directory / (arm + '.npz'))
            old = load_arrays(source['source'])
            np.testing.assert_array_equal(data['actions'], old['actions'])
            np.testing.assert_array_equal(data['success'][1:], old['successes'])
            np.testing.assert_array_equal(data['goals'].all(axis=1), data['success'])
            assert len(data['goals']) == source['states']
            streams[arm] = data
            checkpoints = sorted(set([0, 60, 120, parent['first_alarm'] * 10, parent['query'] * 10, len(data['goals']) - 1]))
            row['arms'][arm] = dict(states=len(data['goals']), final_goals=data['goals'][-1].tolist(),
                goal_intervals=[dict(predicate=goal, intervals=true_intervals(data['goals'][:, i])) for i, goal in enumerate(metadata['goals'])],
                object_diagnostics=[dict(object=name, bilateral_contact_intervals=true_intervals(data['grasp'][:, i]),
                    position_at_checkpoints=data['object_position'][checkpoints, i].tolist(),
                    eef_distance_at_checkpoints=np.linalg.norm(data['object_position'][checkpoints, i] - data['eef_position'][checkpoints], axis=1).tolist(),
                    height_range=[float(data['object_position'][:, i, 2].min()), float(data['object_position'][:, i, 2].max())])
                    for i, name in enumerate(metadata['objects'])],
                checkpoint_steps=checkpoints, eef_at_checkpoints=data['eef_position'][checkpoints].tolist())
        a, b = streams['native'], streams['mobility_balance']
        length = min(len(a['goals']), len(b['goals']))
        start = parent['query'] * 10
        row['paired_physics'] = dict(compared_steps=length, goal_vector_different_steps=int(np.any(a['goals'][:length] != b['goals'][:length], axis=1).sum()),
            mean_suffix_eef_distance=float(np.linalg.norm(a['eef_position'][start:length] - b['eef_position'][start:length], axis=1).mean()),
            max_suffix_eef_distance=float(np.linalg.norm(a['eef_position'][start:length] - b['eef_position'][start:length], axis=1).max()),
            object_endpoint_distance={name: float(np.linalg.norm(a['object_position'][-1, i] - b['object_position'][-1, i]))
                                      for i, name in enumerate(metadata['objects'])},
            goal_endpoint_equal=bool(np.array_equal(a['goals'][-1], b['goals'][-1])))
        result.append(row)
    return result


def analyze(root):
    import torch
    torch.set_num_threads(1)
    config = check_config(root)
    collection = read(root / 'collection.json')
    assert collection['passed'] and collection['model_completed'] == collection['model_attempts'] == 105
    events = records(root / 'calls.jsonl')
    requests, responses = events[::2], events[1::2]
    assert len(requests) == len(responses) == 105
    for ordinal, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == ordinal
        for key in ('parent', 'phase', 'query', 'label', 'noise_sha256', 'bias_sha256'):
            assert request[key] == response[key]
    std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], float)
    rows, native_scales, maximum_softmax = [], [], 0.
    for case in config['cases']:
        parent, = [p for p in config['parents'] if p['name'] == case['parent']]
        inputs = load_arrays(root / case['path'])
        prefix = V82Monitor()
        for probability in inputs['prefix_hb']:
            prefix.update(probability)
        assert prefix.v7.query + 1 == case['query']
        bank = variants(inputs['request_bias'], parent, case['query'])
        calls = [r for r in responses if r['parent'] == case['parent'] and r['phase'] == case['phase']]
        assert [r['label'] for r in calls] == list(LABELS)
        native = combo = None
        for call in calls:
            label = call['label']
            assert digest(root / call['path']) == call['sha256']
            raw = load_arrays(root / call['path'])
            same(raw['request_noise'], inputs['request_noise'], 'fixed request noise')
            expected_bias = np.zeros_like(inputs['request_bias']) if bank[label] is None else bank[label]
            same(raw['request_bias'], expected_bias, 'fixed equal-energy variants')
            for name in ('noise', 'bias'):
                assert array_digest(raw['request_' + name]) == call[name + '_sha256']
            if bank[label] is not None:
                # The expected arithmetic is frozen in LogitCapture; CPU checks
                # here reconstruct every saved addition and probability array.
                maximum_softmax = max(maximum_softmax, audit_gate(raw, dict(gpu_exact_logit_softmax=True,
                    logit_dtypes=['torch.bfloat16'], probability_dtypes=['torch.float32'])))
            for field in HB_FIELDS:
                value = raw['mechanism/' + field]
                assert value.shape == (8, 10, 11, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
            assert raw['mechanism/projection_input'].shape == (10, 10, 1024)
            assert raw['mechanism/velocity'].shape == (10, 10, 24)
            same(raw['mechanism/residual'] + raw['mechanism/total'], raw['mechanism/block'], 'exact residual reconstruction')
            if label == 'native':
                native = raw
                routed = raw['mechanism/total'].astype(float) - raw['mechanism/shared']
                native_scales.append(dict(parent=parent['name'], phase=case['phase'], query=case['query'],
                    by_layer=[dict(layer=layer, routed_rms=rms(routed[i, :, 1:]), shared_rms=rms(raw['mechanism/shared'][i, :, 1:]),
                                   total_rms=rms(raw['mechanism/total'][i, :, 1:]), residual_rms=rms(raw['mechanism/residual'][i, :, 1:]),
                                   block_rms=rms(raw['mechanism/block'][i, :, 1:]))
                              for i, layer in enumerate((2, 3, 4, 5, 12, 13, 14, 15))]))
            if label == 'combo':
                combo = raw
            if label in ('zero', 'post', 'repeat'):
                baseline = combo if label == 'repeat' else native
                assert_equal(raw, baseline, FULL_KEYS + tuple(k for k in raw if k.startswith('mechanism/')), 'independent repeat ' + label)
            if label in ('native', 'combo') and case['old_native']:
                assert_equal(raw, load_arrays(P3D / case['old_' + label]), FULL_KEYS, 'independent P3d match')
            effect = trace_effect(raw, native)
            assert jsonable(effect) == call['effect']
            assert effect['first_local_input_equal'] and effect['first_local_shared_equal']
            decision = independently_decide(prefix, native, raw, 'mobility_balance', std)
            for field in ('native', 'candidate'):
                compare_measurement(call['decision'][field], decision[field])
            for field in ('accepted', 'reasons', 'formal_screen', 'fp32_screen', 'gripper_sign_changes'):
                assert decision[field] == call['decision'][field]
            np.testing.assert_allclose(call['decision']['normalized_action_rms'], decision['normalized_action_rms'], rtol=1e-14, atol=1e-16)
            changed = np.any(np.sort(raw[EFFECTIVE_IDS_KEY], axis=-1) != np.sort(native[EFFECTIVE_IDS_KEY], axis=-1), axis=-1)
            assert float(changed[4:, :, 1:].mean()) == call['top4_changed_fraction']
            if label in ('combo', 'opposite', 'random'):
                rows.append(dict(parent=parent['name'], phase=case['phase'], query=case['query'], label=label,
                    top4_changed_fraction=call['top4_changed_fraction'], action_rms=decision['normalized_action_rms'],
                    instantaneous_delta=decision['instantaneous_delta'].tolist(), score_delta=decision['score_delta'].tolist(),
                    instantaneous_screen=decision['accepted'], gripper_sign_changes=decision['gripper_sign_changes'],
                    first_local_routed_relative=effect['first_local_routed']['relative_l2'],
                    first_local_block_relative=effect['first_local_block']['relative_l2'],
                    back_routed_relative=float(np.mean([e['relative_l2'] for e in effect['routed_derived'][4:]])),
                    back_total_relative=float(np.mean([e['relative_l2'] for e in effect['total'][4:]])),
                    back_block_relative=float(np.mean([e['relative_l2'] for e in effect['block'][4:]])),
                    projection_input_relative=effect['projection_input']['relative_l2'],
                    velocity_relative=effect['velocity']['relative_l2']))
        print(json.dumps(dict(audited=case['parent'], phase=case['phase'])), flush=True)
    columns = [key for key in rows[0] if not key.endswith('_delta')]
    with (root / 'probes.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)
    aggregate = []
    numeric = [key for key in columns if key not in ('parent', 'phase', 'query', 'label')]
    for phase in ('all', 'before_alarm', 'first_accepted', 'last_accepted'):
        for label in ('combo', 'opposite', 'random'):
            selected = [row for row in rows if row['label'] == label and (phase == 'all' or row['phase'] == phase)]
            aggregate.append(dict(phase=phase, label=label, n=len(selected),
                means={key: float(np.mean([row[key] for row in selected])) for key in numeric},
                medians={key: float(np.median([row[key] for row in selected])) for key in numeric}))
    physics = physical_summary(root, config)
    summary = dict(rows=rows, aggregate=aggregate, native_scales=native_scales, physics=physics,
                   new_model_calls=105, nonzero_unique_probes=45, new_environment_actions=0,
                   peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
                   inference_seconds=sum(r['resource']['seconds'] for r in responses))
    write(root / 'summary.json', summary)
    write(root / 'audit.json', dict(passed=True, checked_responses=105, nonzero_unique_probes=45,
        zero_post_repeat_pairs=45, restored_states=sum(a['states'] for p in physics for a in p['arms'].values()),
        same_first_local_input_and_shared=True, residual_reconstruction_exact=True,
        maximum_cpu_softmax_error=maximum_softmax, source_hashes_unchanged=True))
    print(json.dumps(dict(passed=True, aggregate=aggregate[:3])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run.resolve())
