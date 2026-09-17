"""Independently audit crossover dispatch, replay, history, and score responses."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from run_crossover_experiment import check_config, P3E, INPUT_KEYS, TRACE_KEYS
from crossover_protocol import (LABELS, MAIN_LABELS, BIASED, DONORS, CONTROLS, SITE_MASK,
                                expected_computation, validate_bias, set_change)
from audit_online_experiment import read, write, records, digest, array_digest, same
from audit_response_matrix import audit_gate, load_arrays
from audit_gated_rollout_experiment import independently_decide, compare_measurement
from run_online_experiment import jsonable
from run_gate_experiment import FULL_KEYS
from run_gated_rollout_experiment import LOGIT_KEYS
from mechanism_protocol import comparison
from collection_routes import HB_LAYERS, NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY, PROBS_KEY
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS
from v82_closed_loop import V82Monitor
from crossover_capture_v2 import OUTPUT_MASK


def exact_fields(actual, expected, fields, label):
    for key in fields:
        same(actual[key], expected[key], label + ': ' + key)


def audit_dispatch(raw, biased):
    for phase in ('native', 'effective'):
        p = raw['v8_control/' + phase + '_probs_fp32']
        ids = raw['collection/hb_' + phase + '_ids'].astype(np.int64)
        weights = raw['collection/hb_' + phase + '_weights']
        assert p.shape == (8, 10, 11, 32) and p.dtype == np.float32
        assert np.isfinite(p).all() and np.all(p >= 0)
        np.testing.assert_allclose(p.sum(-1), 1., rtol=0, atol=5e-7)
        assert ids.shape == weights.shape == (8, 10, 11, 4)
        assert np.all((ids >= 0) & (ids < 32)) and np.all(np.diff(np.sort(ids, axis=-1), axis=-1) > 0)
        selected = np.take_along_axis(p, ids, axis=-1)
        assert np.all(selected.min(-1) >= np.sort(p, axis=-1)[..., -4])
        np.testing.assert_allclose(weights, selected / selected.sum(-1, keepdims=True), rtol=0, atol=1.3e-7)
    same(raw[PROBS_KEY], raw[NATIVE_PROBS].astype(np.float16), 'FP16 archival probabilities')
    for field in ('ids', 'weights'):
        native, effective = (raw['collection/hb_' + phase + '_' + field] for phase in ('native', 'effective'))
        same(native[~SITE_MASK], effective[~SITE_MASK], 'unbiased off-scope dispatch')
        same(effective[:, :, 1:].transpose(1, 0, 2, 3), raw['routing/expert_' + field], 'independent live dispatch')
        if not biased:
            same(native, effective, 'unbiased gate dispatch')
    if not biased:
        same(raw[NATIVE_PROBS], raw[EFFECTIVE_PROBS], 'unbiased gate probabilities')


def grouped(rows, field=None):
    result = []
    groups = ['all'] if field is None else list(dict.fromkeys(row[field] for row in rows))
    numeric = ('top4_changed_fraction', 'raw_total_relative', 'effective_total_relative',
               'block_relative', 'projection_relative', 'velocity_relative', 'action_rms',
               'raw_to_effective_rms', 'top4_change_vs_joint', 'action_rms_vs_joint')
    for group in groups:
        for label in MAIN_LABELS:
            selected = [r for r in rows if r['label'] == label and (field is None or r[field] == group)]
            result.append(dict(group=group, label=label, n=len(selected),
                means={k: float(np.mean([r[k] for r in selected])) for k in numeric},
                instantaneous_delta_mean=np.mean([r['instantaneous_delta'] for r in selected], axis=0).tolist(),
                score_delta_mean=np.mean([r['score_delta'] for r in selected], axis=0).tolist(),
                instantaneous_screen_count=sum(r['instantaneous_screen'] for r in selected),
                formal_screen_count=sum(r['formal_screen'] for r in selected),
                fp32_screen_count=sum(r['fp32_screen'] for r in selected),
                formal_all_below_count=sum(r['formal_all_below'] for r in selected),
                formal_cross_down_counts=np.sum([r['formal_cross_down'] for r in selected], axis=0).tolist(),
                actions_exact_native_count=sum(r['actions_exact_native'] for r in selected),
                actions_exact_joint_count=sum(r['actions_exact_joint'] for r in selected)))
    return result


def first_hb_difference(actual, expected):
    changed = np.any(actual != expected, axis=-1).transpose(1, 0, 2)
    where = np.argwhere(changed)
    if not len(where):
        return None
    step, slot, token = (int(v) for v in where[0])
    return dict(denoise_step=step, layer=HB_LAYERS[slot], token=token,
                max_abs_delta=float(np.max(np.abs(actual[slot, step, token].astype(float) - expected[slot, step, token]))))


def scope_diagnostics(root, config):
    first = config['cases'][0]
    calls = records(root / 'scope-diagnostics/calls.jsonl')
    assert len(calls) == 4
    with np.load(root / first['path'], allow_pickle=False) as saved:
        noise, bias = saved['request_noise'], saved['request_bias']
    donors = {label: load_arrays(P3E / first['source_' + label]) for label in ('native', 'joint')}
    diagnostics = []
    for i, (request, response) in enumerate(zip(calls[::2], calls[1::2])):
        label = ('route_only', 'output_only')[i]
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == i and response['label'] == request['label'] == label
        assert digest(root / response['path']) == response['sha256']
        raw = load_arrays(root / response['path'])
        same(raw['request_noise'], noise, 'scope diagnostic same noise')
        same(raw['request_bias'], bias if i == 0 else np.zeros_like(bias), 'scope diagnostic same bias')
        assert response['noise_sha256'] == request['noise_sha256'] == array_digest(noise)
        donor = donors['native' if i == 0 else 'joint']
        computed_source = donors['joint' if i == 0 else 'native']
        assert response['donor_sha256'] == request['donor_sha256'] == array_digest(donor['mechanism/total'])
        assert response['donor_source'] == request['donor_source'] == first['source_native' if i == 0 else 'source_joint']
        audit_dispatch(raw, i == 0)
        same(raw['mechanism/total'][SITE_MASK], donor['mechanism/total'][SITE_MASK], 'action-only donor injection')
        same(raw['mechanism/total'][~SITE_MASK], raw['crossover/raw_total'][~SITE_MASK], 'unpatched state token')
        same(raw['mechanism/residual'] + raw['mechanism/total'], raw['mechanism/block'], 'diagnostic residual reconstruction')
        for key in ('mechanism/input', 'mechanism/shared', 'mechanism/residual', NATIVE_PROBS, NATIVE_IDS_KEY,
                    'collection/hb_native_weights'):
            same(raw[key][4, 0], donor[key][4, 0], 'same first-local input/native gate/shared')
        same(raw['crossover/raw_total'][4, 0], computed_source['mechanism/total'][4, 0], 'first-local opposite route computation')
        for key in (EFFECTIVE_PROBS, EFFECTIVE_IDS_KEY, 'collection/hb_effective_weights'):
            same(raw[key][4, 0, 0], donor[key][4, 0, 0], 'unchanged state-token gate')
        first_difference = first_hb_difference(raw['mechanism/total'], donor['mechanism/total'])
        assert first_difference is not None and first_difference['token'] == 0 and first_difference['layer'] in HB_LAYERS[4:]
        slot, step = HB_LAYERS.index(first_difference['layer']), first_difference['denoise_step']
        for key in ('mechanism/input', 'mechanism/shared', 'mechanism/residual'):
            same(raw[key][slot, step], donor[key][slot, step], 'same input/shared/residual at first differing output')
        for key in (EFFECTIVE_PROBS, EFFECTIVE_IDS_KEY, 'collection/hb_effective_weights'):
            same(raw[key][slot, step, 0], donor[key][slot, step, 0], 'state gate unchanged at first differing output')
        diagnostics.append(dict(label=label, intended_donor='native' if i == 0 else 'joint',
            first_hb_difference={key: first_hb_difference(raw[key], donor[key]) for key in TRACE_KEYS if raw[key].ndim == 4},
            first_local_state_output=comparison(raw['crossover/raw_total'][4, 0, 0], donor['mechanism/total'][4, 0, 0]),
            first_local_input_and_shared_equal=True, first_local_state_gate_equal=True,
            first_unclamped_output_difference=first_difference, first_divergence_input_and_shared_equal=True,
            first_divergence_state_gate_equal=True,
            first_divergence_state_output=comparison(raw['crossover/raw_total'][slot, step, 0], donor['mechanism/total'][slot, step, 0]),
            actions=comparison(raw['actions'], donor['actions']),
            projection=comparison(raw['mechanism/projection_input'], donor['mechanism/projection_input']),
            velocity=comparison(raw['mechanism/velocity'], donor['mechanism/velocity'])))
    write(root / 'scope-diagnostics/summary.json', dict(rows=diagnostics,
        attribution='State-token numerical coupling at fixed input/gate is directly observed if its output differs; exact kernel cause is not isolated.'))
    return diagnostics, [r['resource'] for r in calls[1::2]]


def analyze(root):
    import torch
    torch.set_num_threads(1)
    config, collection = check_config(root), read(root / 'collection.json')
    assert config['schema'] == 'local.moe_crossover.v2'
    assert collection['passed'] and collection['model_completed'] == collection['model_attempts'] == 137
    assert collection['main_model_calls'] == 135 and collection['scope_diagnostic_model_calls'] == 2
    assert collection['total_user_request_model_calls'] == 142
    assert collection['new_environment_actions'] == collection['simulator_states_read'] == 0
    assert collection['parameter_sha256_before'] == collection['parameter_sha256_after']
    assert collection['parameter_versions_unchanged']
    pilot = Path(config['prior_pilot'])
    for path, expected in config['prior_pilot_files'].items():
        assert digest(pilot / path) == expected, path
    assert read(pilot / 'failure.json')['model_completed'] == 5
    diagnostics, diagnostic_resources = scope_diagnostics(root, config)
    events = records(root / 'calls.jsonl')
    requests, responses = events[::2], events[1::2]
    assert len(requests) == len(responses) == 135
    for i, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == i
        assert request['actual_call_ordinal'] == response['actual_call_ordinal'] == i + 2
        for key in ('parent', 'phase', 'query', 'label', 'biased', 'donor', 'noise_sha256',
                    'bias_sha256', 'donor_sha256', 'history_sha256'):
            assert request[key] == response[key]
        assert response['hooks_removed'] and response['parameter_versions_unchanged'] and response['history_unchanged']
    source_verification = read(P3E / 'verification.json')
    std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], np.float64)
    result_rows, pairs, maximum_softmax = [], [], 0.
    for case in config['cases']:
        inputs = load_arrays(root / case['path'])
        assert set(inputs) == set(INPUT_KEYS)
        assert digest(P3E / case['source_input']) == source_verification['files'][case['source_input']]
        with np.load(P3E / case['source_input'], allow_pickle=False) as source:
            exact_fields(inputs, source, INPUT_KEYS, 'unchanged archived model input and route history')
        validate_bias(inputs['request_bias'])
        prefix = V82Monitor()
        for p in inputs['prefix_hb']:
            prefix.update(p)
        assert prefix.v7.query + 1 == case['query']
        history = hashlib.sha256(pickle.dumps(prefix, protocol=5)).hexdigest()
        calls = [r for r in responses if r['parent'] == case['parent'] and r['phase'] == case['phase']]
        assert [r['label'] for r in calls] == list(LABELS)
        kept, decisions = {}, {}
        for call in calls:
            label = call['label']
            assert digest(root / call['path']) == call['sha256']
            raw = load_arrays(root / call['path'])
            same(raw['request_noise'], inputs['request_noise'], 'fixed noise')
            expected_bias = inputs['request_bias'] if label in BIASED else np.zeros_like(inputs['request_bias'])
            same(raw['request_bias'], expected_bias, 'fixed requested gate bias')
            for name in ('bias', 'noise'):
                assert array_digest(raw['request_' + name]) == call[name + '_sha256']
            assert call['biased'] == (label in BIASED) and call['donor'] == DONORS.get(label)
            assert call['history_sha256'] == history
            audit_dispatch(raw, label in BIASED)
            if label in BIASED:
                maximum_softmax = max(maximum_softmax, audit_gate(raw, call))
            for key in TRACE_KEYS + ('crossover/raw_total',):
                value = raw[key]
                shape = (10, 10, 24) if key.endswith('/velocity') else (10, 10, 1024) if key.endswith('/projection_input') else (8, 10, 11, 1024)
                assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
            same(raw['mechanism/residual'] + raw['mechanism/total'], raw['mechanism/block'], 'residual reconstruction')
            if label in MAIN_LABELS:
                kept[label] = raw
            if label in ('native', 'joint'):
                source_path = case['source_' + label]
                assert digest(P3E / source_path) == source_verification['files'][source_path]
                with np.load(P3E / source_path, allow_pickle=False) as source:
                    exact_fields(raw, source, FULL_KEYS + TRACE_KEYS + (LOGIT_KEYS if label == 'joint' else ()), 'P3e exact donor')
            exact_fields(raw, kept[expected_computation(label)], TRACE_KEYS + ('actions',), 'matched computation')
            if label in CONTROLS:
                exact_fields(raw, kept[CONTROLS[label]], FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',) +
                             (LOGIT_KEYS if label in BIASED else ()), 'replay/repeat/post control')
            effective, computed = raw['mechanism/total'], raw['crossover/raw_total']
            same(effective[~OUTPUT_MASK], computed[~OUTPUT_MASK], 'no off-scope replay')
            donor_name = DONORS.get(label)
            if donor_name is None:
                same(computed, effective, 'no-replay natural output')
                assert call['donor_sha256'] is None
            else:
                donor = kept[donor_name]['mechanism/total']
                same(effective[OUTPUT_MASK], donor[OUTPUT_MASK], 'same-case whole-output exact donor injection')
                assert array_digest(donor) == call['donor_sha256']
            expected_sites = [] if donor_name is None else [(i, t) for t in range(10) for i in HB_LAYERS[4:]]
            same(raw['crossover/patch_sites'], np.asarray(expected_sites, np.int16).reshape(-1, 2), 'patch coverage/order')
            same(raw['crossover/replay_token_mask'], OUTPUT_MASK if donor_name else np.zeros_like(OUTPUT_MASK), 'explicit replay token mask')
            if label in ('route_only', 'repeat_route'):
                same(raw[NATIVE_PROBS], kept['native'][NATIVE_PROBS], 'original hidden inputs and unbiased gate')
                same(computed[4, 0], kept['joint']['crossover/raw_total'][4, 0], 'first site truly computes biased output')
                assert np.any(computed[SITE_MASK] != effective[SITE_MASK])
            if label in ('output_only', 'repeat_output'):
                same(raw[EFFECTIVE_PROBS], kept['joint'][NATIVE_PROBS], 'unbiased gate on joint hidden inputs')
                same(computed[4, 0], kept['native']['crossover/raw_total'][4, 0], 'first site truly computes unbiased output')
                assert np.any(computed[SITE_MASK] != effective[SITE_MASK])
            decision = independently_decide(prefix, kept['native'], raw, 'mobility_balance', std)
            for field in ('native', 'candidate'):
                compare_measurement(call['decision'][field], decision[field])
            for field in ('accepted', 'reasons', 'formal_screen', 'fp32_screen', 'gripper_sign_changes', 'operator'):
                assert call['decision'][field] == decision[field]
            for field in ('score_delta', 'instantaneous_delta', 'fp32_instantaneous_delta'):
                np.testing.assert_array_equal(call['decision'][field], decision[field])
            np.testing.assert_allclose(call['decision']['normalized_action_rms'], decision['normalized_action_rms'], rtol=1e-14, atol=1e-16)
            assert hashlib.sha256(pickle.dumps(prefix, protocol=5)).hexdigest() == history
            assert set_change(raw[EFFECTIVE_IDS_KEY], kept['native'][EFFECTIVE_IDS_KEY]) == call['top4_changed_fraction']
            assert comparison(effective[SITE_MASK], computed[SITE_MASK]) == call['raw_to_effective']
            decisions[label] = decision
        native, joint = kept['native'], kept['joint']
        for label in MAIN_LABELS:
            raw, decision = kept[label], decisions[label]
            rows = dict(parent=case['parent'], phase=case['phase'], query=case['query'], label=label,
                instantaneous_screen=decision['accepted'], formal_screen=decision['formal_screen'], fp32_screen=decision['fp32_screen'],
                formal_all_below=bool(np.all(decision['candidate']['normalized'] < 0)),
                formal_cross_down=(np.asarray(decision['native']['normalized']) >= 0) & (decision['candidate']['normalized'] < 0),
                formal_normalized=decision['candidate']['normalized'], formal_scores=decision['candidate']['scores'],
                formal_thresholds=decision['candidate']['thresholds'], instantaneous_scores=decision['candidate']['instantaneous'],
                instantaneous_delta=decision['instantaneous_delta'], score_delta=decision['score_delta'],
                alarm_status=decision['candidate']['status'], action_rms=decision['normalized_action_rms'],
                gripper_sign_changes=decision['gripper_sign_changes'],
                action_rms_vs_joint=float(np.sqrt(np.mean(((raw['actions'][:, :6].astype(float) - joint['actions'][:, :6]) / std) ** 2))),
                actions_exact_native=bool(np.array_equal(raw['actions'], native['actions'])),
                actions_exact_joint=bool(np.array_equal(raw['actions'], joint['actions'])),
                top4_changed_fraction=set_change(raw[EFFECTIVE_IDS_KEY], native[EFFECTIVE_IDS_KEY]),
                top4_change_vs_joint=set_change(raw[EFFECTIVE_IDS_KEY], joint[EFFECTIVE_IDS_KEY]),
                raw_total_relative=comparison(raw['crossover/raw_total'][SITE_MASK], native['crossover/raw_total'][SITE_MASK])['relative_l2'],
                effective_total_relative=comparison(raw['mechanism/total'][SITE_MASK], native['mechanism/total'][SITE_MASK])['relative_l2'],
                block_relative=comparison(raw['mechanism/block'][SITE_MASK], native['mechanism/block'][SITE_MASK])['relative_l2'],
                projection_relative=comparison(raw['mechanism/projection_input'], native['mechanism/projection_input'])['relative_l2'],
                velocity_relative=comparison(raw['mechanism/velocity'], native['mechanism/velocity'])['relative_l2'],
                raw_to_effective_rms=comparison(raw['mechanism/total'][SITE_MASK], raw['crossover/raw_total'][SITE_MASK])['delta_rms'])
            result_rows.append(jsonable(rows))
        r, o = kept['route_only'], kept['output_only']
        pairs.append(dict(parent=case['parent'], phase=case['phase'], query=case['query'],
            route_vs_native_compute_exact=True, output_vs_joint_compute_exact=True,
            route_vs_native_effective_routes_differ=bool(np.any(r[EFFECTIVE_PROBS] != native[EFFECTIVE_PROBS])),
            output_vs_joint_effective_routes_differ=bool(np.any(o[EFFECTIVE_PROBS] != joint[EFFECTIVE_PROBS])),
            route_vs_native_scores_differ=bool(np.any(decisions['route_only']['candidate']['scores'] != decisions['native']['candidate']['scores'])),
            output_vs_joint_scores_differ=bool(np.any(decisions['output_only']['candidate']['scores'] != decisions['joint']['candidate']['scores'])),
            output_gate_matches_joint_prebias_exact=True,
            route_only_passes_with_identical_native_actions=decisions['route_only']['accepted'],
            joint_passes_but_output_only_fails=decisions['joint']['accepted'] and not decisions['output_only']['accepted']))
        print(json.dumps(dict(audited=case['parent'], phase=case['phase'])), flush=True)
    summary = dict(rows=result_rows, pairs=pairs, aggregate=grouped(result_rows),
                   by_parent=grouped(result_rows, 'parent'), by_phase=grouped(result_rows, 'phase'),
                   components=['freeze', 'acceleration', 'periodicity', 'inversion', 'curvature'],
                   new_model_calls=137, total_user_request_model_calls=142, prior_pilot_model_calls=5,
                   main_model_calls=135, scope_diagnostic_model_calls=2, scope_diagnostics=diagnostics,
                   main_cell_observations=60, validation_calls=75, independent_parents=5,
                   new_environment_actions=0, simulator_states_read=0,
                   peak_reserved_mib=max(r['peak_reserved_mib'] for r in [c['resource'] for c in responses] + diagnostic_resources),
                   inference_seconds=sum(r['resource']['seconds'] for r in responses) + sum(r['seconds'] for r in diagnostic_resources))
    columns = [k for k, v in result_rows[0].items() if not isinstance(v, (dict, list))]
    with (root / 'cells.csv').open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in result_rows)
    write(root / 'summary.json', summary)
    write(root / 'audit.json', dict(passed=True, checked_responses=137, main_responses=135, scope_diagnostic_responses=2,
        prior_pilot_preserved=True, total_user_request_model_calls=142, identity_repeat_post_controls=75,
        cross_cell_compute_pairs=30, exact_source_donors=30, raw_and_effective_outputs_audited=True,
        exact_scoped_replay_calls=90, history_unchanged=True, parameter_content_hash_unchanged=True,
        maximum_cpu_softmax_error=maximum_softmax, source_hashes_unchanged=True))
    print(json.dumps(dict(passed=True, aggregate=summary['aggregate'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    analyze(parser.parse_args().run.resolve())
