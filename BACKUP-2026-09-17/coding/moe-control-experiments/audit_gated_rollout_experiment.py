"""Reconstruct gated decisions, actual execution and outcomes independently."""

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from audit_online_experiment import (V82Monitor, read, write, records, digest, array_digest, same,
                                      load_episode_trace)
from audit_response_matrix import (KEYS, independent_bank, measurement, passes, audit_gate, load_arrays, norm)
from run_online_experiment import jsonable

ARMS = ('native', 'mobility', 'mobility_balance', 'random_balance')
LABELS = dict(mobility='mobility-high-plus', mobility_balance='mobility_balance-high-plus',
              random_balance='random-mobility_balance-high')
COMPONENTS = ('freeze', 'acceleration', 'periodicity', 'inversion', 'curvature')
EFFECTIVE = 'v8_control/effective_probs_fp32'
NATIVE = 'v8_control/native_probs_fp32'
OBSERVATIONS = (('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'), ('observation/state', 'states'))
EVALUATION = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')


def compare_measurement(saved, actual):
    for field in ('scores', 'normalized', 'instantaneous', 'thresholds', 'margins'):
        np.testing.assert_array_equal(saved[field], actual[field])
    assert saved['status'] == jsonable(actual['status'])


def independently_decide(prefix, native, candidate, arm, std):
    operator = 'mobility' if arm == 'mobility' else 'mobility_balance'
    base = measurement(prefix, native[EFFECTIVE].astype(np.float16))
    changed = measurement(prefix, candidate[EFFECTIVE].astype(np.float16))
    delta = changed['normalized'] - base['normalized']
    instant = (changed['instantaneous'] - base['instantaneous']) / base['margins']
    full_base = measurement(prefix, native[EFFECTIVE])
    full_changed = measurement(prefix, candidate[EFFECTIVE])
    full_delta = (full_changed['instantaneous'] - full_base['instantaneous']) / full_base['margins']
    rms = norm((candidate['actions'][:, :6].astype(float) - native['actions'][:, :6]) / std) / np.sqrt(60)
    gripper = int(np.count_nonzero((candidate['actions'][:, 6] >= 0) != (native['actions'][:, 6] >= 0)))
    target = [0] if operator == 'mobility' else [0, 3]
    reasons = []
    for i in range(5):
        if i in target and instant[i] > -.05:
            reasons.append('target_%d_not_improved' % i)
        if i not in target and instant[i] > .05:
            reasons.append('side_effect_%d' % i)
    if rms > .05:
        reasons.append('action_rms')
    if gripper:
        reasons.append('gripper_change')
    return dict(native=base, candidate=changed, score_delta=delta, instantaneous_delta=instant,
                normalized_action_rms=float(rms), gripper_sign_changes=gripper, operator=operator, reasons=reasons,
                accepted=passes(instant, operator, rms, gripper), formal_screen=passes(delta, operator, rms, gripper),
                fp32_instantaneous_delta=full_delta, fp32_screen=passes(full_delta, operator, rms, gripper))


def audit(root, partial=False):
    import torch
    torch.set_num_threads(1)
    config = read(root / 'config.json')
    if partial:
        completed = [read(root / p['name'] / arm / 'result.json') for p in config['parents'] for arm in ARMS
                     if (root / p['name'] / arm / 'result.json').is_file()]
        collection = dict(passed=True, results=completed, actual_rollouts=len(completed))
    else:
        collection = read(root / 'collection.json')
    matrix = Path(config['source_matrix'])
    assert collection['passed'] and collection['actual_rollouts'] == len(collection['results'])
    assert 1 <= len(collection['results']) <= 20 if partial else len(collection['results']) == 20
    assert config['arms'] == list(ARMS) and config['recovery_queries'] == 12 and config['chunk'] == 10 and config['horizon'] == 520
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert digest(matrix / 'config.json') == config['matrix_config_sha256']
    assert digest(matrix / 'verification.json') == config['matrix_verification_sha256']
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    if partial:
        complete_keys = {(r['parent'], r['arm']) for r in collection['results']}
        with (root / 'calls.jsonl').open() as stream:
            events = [json.loads(line) for line in stream if line.endswith('\n')]
        events = [r for r in events if (r['parent'], r['arm']) in complete_keys]
    else:
        events = records(root / 'calls.jsonl')
    requests, responses = events[::2], events[1::2]
    assert len(requests) == len(responses) <= 731
    if not partial:
        assert len(requests) == collection['model_attempts'] == collection['model_completed']
    keyed = {}
    for i, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == i
        for field in ('parent', 'arm', 'query', 'kind', 'noise_sha256', 'bias_sha256'):
            assert request[field] == response[field]
        key = (response['parent'], response['arm'], response['query'], response['kind'])
        assert key not in keyed
        keyed[key] = response
    std = np.array(config['shared_metadata']['normalization_action_std'][:6], float)
    checked, audits, flat_decisions, branches = set(), [], [], []
    maximum_softmax_error, maximum_precision_delta = 0., 0.
    mask = np.zeros((8, 10, 11, 32), bool)
    mask[4:, :, 1:] = True

    def checked_response(parent, arm, query, kind, bank, rollout):
        nonlocal maximum_softmax_error
        key = (parent['name'], arm, query, kind)
        assert key not in checked
        checked.add(key)
        call = keyed[key]
        path = root / call['path']
        assert digest(path) == call['sha256'], path
        raw = load_arrays(path)
        same(raw['request_noise'], bank[query], 'query-index native noise')
        same(raw['sim_before'], rollout['sim_states_after'][query * 10 - 1], 'live simulation state before query')
        for field, value in (('actions_sha256', raw['actions']), ('effective_hb_sha256', raw[EFFECTIVE].astype(np.float16)),
                              ('bias_sha256', raw['request_bias']), ('noise_sha256', raw['request_noise'])):
            assert call[field] == array_digest(value)
        assert raw['actions'].shape == (10, 7) and raw['actions'].dtype == np.float32 and np.isfinite(raw['actions']).all()
        bias = raw['request_bias']
        assert bias.shape == mask.shape and bias.dtype == np.float32 and np.isfinite(bias).all()
        assert np.max(np.abs(bias)) <= .3000001 and np.all(bias[~mask] == 0)
        same(raw['collection/hb_probs'], raw[NATIVE].astype(np.float16), 'native capture precision')
        for phase in ('native', 'effective'):
            probabilities = raw['v8_control/' + phase + '_probs_fp32']
            assert probabilities.dtype == np.float32 and probabilities.shape == mask.shape
            assert np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            np.testing.assert_allclose(probabilities.sum(-1), 1., atol=3e-7, rtol=0)
            ids, weights = raw['collection/hb_' + phase + '_ids'], raw['collection/hb_' + phase + '_weights']
            selected = np.take_along_axis(probabilities, ids.astype(int), axis=-1)
            assert np.all((ids >= 0) & (ids < 32)) and np.all(np.diff(np.sort(ids, axis=-1), axis=-1) != 0)
            assert np.all(selected.min(-1) >= np.sort(probabilities, axis=-1)[..., -4])
            np.testing.assert_allclose(weights, selected / selected.sum(-1, keepdims=True), atol=1.3e-7, rtol=0)
        for field in ('ids', 'weights'):
            same(raw['collection/hb_effective_' + field][:, :, 1:].transpose(1, 0, 2, 3),
                 raw['routing/expert_' + field], 'actual wire dispatch')
        if kind in ('candidate', 'repeat', 'zero'):
            maximum_softmax_error = max(maximum_softmax_error, audit_gate(raw, call))
        else:
            assert np.all(bias == 0)
            same(raw[NATIVE], raw[EFFECTIVE], 'unmodified native probabilities')
        return raw, call

    for parent in config['parents']:
        source = Path(parent['source'])
        assert digest(source / 'episode-trace.json') == parent['manifest_sha256']
        assert digest(source / 'episode-trace.npz') == parent['trace_sha256']
        assert digest(EVALUATION / parent['name'] / 'full-hb-routes.npz') == parent['hb_sha256']
        manifest, source_arrays = load_episode_trace(source)
        reference = load_arrays(EVALUATION / parent['name'] / 'full-hb-routes.npz')['hb_router_probs']
        rng = np.random.default_rng(manifest['config']['flow_noise_seed'])
        bank = np.stack([rng.standard_normal((10, 24)).astype(np.float32) for _ in range(52)])
        same(bank[:len(source_arrays['flow_noises'])], source_arrays['flow_noises'], 'source noise bank')
        prefix = V82Monitor()
        for p in reference[:parent['query']]:
            prefix.update(p)
        assert prefix.first_v82_alarm == parent['first_alarm'] == parent['query'] - 1
        for arm in ARMS:
            if partial and (parent['name'], arm) not in complete_keys:
                continue
            directory = root / parent['name'] / arm
            result, environment = read(directory / 'result.json'), read(directory / 'environment.json')
            assert result in collection['results'] and result['parent'] == parent['name'] and result['arm'] == arm
            rollout = load_arrays(directory / 'rollout.npz')
            assert result['action_steps'] == environment['steps'] == len(rollout['actions'])
            assert result['success'] == environment['success'] == bool(rollout['successes'][-1])
            assert not rollout['successes'][:-1].any() and (result['success'] or result['action_steps'] == 520)
            assert result['rescue'] == (not parent['source_success'] and result['success'])
            assert result['harm'] == (parent['source_success'] and not result['success'])
            for field in ('sim_states_after', 'rewards', 'dones', 'successes'):
                same(rollout[field][:parent['step']], source_arrays[field][:parent['step']], 'common physical prefix')
            same(rollout['actions'][:parent['step']], source_arrays['predicted_actions'][:parent['query']].reshape(-1, 7), 'prefix action tape')
            rows, decisions = records(directory / 'queries.jsonl'), records(directory / 'decisions.jsonl')
            assert len(rows) == len(decisions) == result['suffix_queries']
            monitor, previous = copy.deepcopy(prefix), reference[parent['query'] - 1]
            opportunities, proposals, accepts, accepted_steps, local_checks = 0, 0, 0, 0, []
            start_calls, step = len(checked), parent['step']
            for offset, (row, decision_record) in enumerate(zip(rows, decisions)):
                query = parent['query'] + offset
                assert {k: v for k, v in row.items() if k not in ('executed', 'success')} == decision_record
                assert row['frozen_before_environment_step'] and row['query'] == query and row['step'] == step == query * 10
                native, native_call = checked_response(parent, arm, query, 'native', bank, rollout)
                assert row['native_path'] == native_call['path']
                if offset == 0:
                    prior = load_arrays(matrix / 'states' / parent['name'] / 'native.npz')
                    for field in KEYS:
                        same(native[field], prior[field], 'same P3c initial shadow')
                    for key, field in OBSERVATIONS:
                        same(native[key], source_arrays[field][query], 'first observation exact')
                    if arm == 'native':
                        for kind in ('zero', 'post'):
                            control, _ = checked_response(parent, arm, query, kind, bank, rollout)
                            assert np.all(control['request_bias'] == 0)
                            for field in KEYS:
                                same(control[field], native[field], 'zero/post response')
                active = arm != 'native' and offset < 12
                candidate, candidate_call, recalculated = None, None, None
                chosen, chosen_call = native, native_call
                if active:
                    opportunities += 1
                    operator = 'mobility' if arm == 'mobility' else 'mobility_balance'
                    specs = [dict(operator=operator, randomized=arm == 'random_balance', fraction=1., sign=1)]
                    try:
                        expected, energy = independent_bank(native[EFFECTIVE].astype(np.float16), previous,
                                                             dict(parent, query=query), specs)
                    except AssertionError:
                        assert row['generation'] == dict(fallback_reason='degenerate_direction')
                        assert row['candidate_path'] is None
                        assert row['decision'] == dict(accepted=False, reasons=['degenerate_direction'], operator=None)
                    else:
                        candidate, candidate_call = checked_response(parent, arm, query, 'candidate', bank, rollout)
                        proposals += 1
                        assert row['candidate_path'] == candidate_call['path']
                        for field, _ in OBSERVATIONS:
                            same(candidate[field], native[field], 'same-observation proposal')
                        same(candidate['sim_before'], native['sim_before'], 'same-physical-state proposal')
                        np.testing.assert_allclose(candidate['request_bias'], expected[0], atol=2e-8, rtol=1e-7)
                        np.testing.assert_allclose(row['generation']['high_energy'], energy, rtol=1e-12)
                        np.testing.assert_allclose(norm(candidate['request_bias']), energy, rtol=1e-7)
                        assert row['generation']['label'] == LABELS[arm] and row['generation']['query'] == query
                        if offset == 0:
                            prior = load_arrays(matrix / 'states' / parent['name'] / (LABELS[arm] + '.npz'))
                            for field in KEYS + ('request_bias', 'diagnostic/native_logits_fp32', 'diagnostic/effective_logits_fp32', 'diagnostic/applied_bias_fp32'):
                                same(candidate[field], prior[field], 'same P3c initial probe')
                            if arm == 'mobility_balance':
                                repeated, _ = checked_response(parent, arm, query, 'repeat', bank, rollout)
                                for field in KEYS + ('request_bias', 'diagnostic/native_logits_fp32', 'diagnostic/effective_logits_fp32', 'diagnostic/applied_bias_fp32'):
                                    same(candidate[field], repeated[field], 'nonzero repeat exact')
                        recalculated = independently_decide(monitor, native, candidate, arm, std)
                        stored = row['decision']
                        for field in ('accepted', 'reasons', 'operator', 'gripper_sign_changes', 'formal_screen', 'fp32_screen'):
                            assert stored[field] == recalculated[field], field
                        for field in ('score_delta', 'instantaneous_delta', 'fp32_instantaneous_delta'):
                            np.testing.assert_array_equal(stored[field], recalculated[field])
                        np.testing.assert_allclose(stored['normalized_action_rms'], recalculated['normalized_action_rms'], rtol=1e-12, atol=1e-14)
                        for field in ('native', 'candidate'):
                            compare_measurement(stored[field], recalculated[field])
                        if recalculated['accepted']:
                            accepts += 1
                            chosen, chosen_call = candidate, candidate_call
                        maximum_precision_delta = max(maximum_precision_delta, float(np.abs(
                            recalculated['instantaneous_delta'] - recalculated['fp32_instantaneous_delta']).max()))
                        flat = dict(parent=parent['name'], head=parent['head'], arm=arm, query=query,
                                    accepted=recalculated['accepted'], reasons=';'.join(recalculated['reasons']),
                                    formal_screen=recalculated['formal_screen'], fp32_screen=recalculated['fp32_screen'],
                                    normalized_action_rms=recalculated['normalized_action_rms'],
                                    gripper_sign_changes=recalculated['gripper_sign_changes'], request_l2=norm(candidate['request_bias']),
                                    effective_logit_delta_l2=norm(candidate['diagnostic/effective_logits_fp32'].astype(float) - candidate['diagnostic/native_logits_fp32']),
                                    back_action_set_change_fraction=float(np.any(np.sort(candidate['collection/hb_effective_ids'][4:, :, 1:], axis=-1) !=
                                        np.sort(native['collection/hb_effective_ids'][4:, :, 1:], axis=-1), axis=-1).mean()))
                        for index, component in enumerate(COMPONENTS):
                            flat['instant_' + component] = float(recalculated['instantaneous_delta'][index])
                            flat['score_' + component] = float(recalculated['score_delta'][index])
                            flat['native_severity_' + component] = float(recalculated['native']['normalized'][index])
                            flat['candidate_severity_' + component] = float(recalculated['candidate']['normalized'][index])
                        flat_decisions.append(flat)
                else:
                    assert row['candidate_path'] is None and row['generation'] is None
                    assert row['decision'] == dict(accepted=False, reasons=['outside_window'], operator=None)
                assert row['selected_path'] == chosen_call['path']
                chosen_measure = measurement(monitor, chosen[EFFECTIVE].astype(np.float16))
                compare_measurement(row['selected'], chosen_measure)
                previous = chosen[EFFECTIVE].astype(np.float16)
                committed = monitor.update(previous)
                assert jsonable(committed) == row['selected']['status']
                assert committed['query'] == query
                executed = row['executed']
                assert 1 <= executed <= 10 and (executed == 10 or offset == len(rows) - 1)
                same(rollout['actions'][step:step + executed], chosen['actions'][:executed], 'actually executed selected action')
                assert row['success'] == bool(rollout['successes'][step + executed - 1])
                if row['decision']['accepted']:
                    accepted_steps += executed
                if arm == 'native' or result['accepted'] == 0:
                    for key, field in OBSERVATIONS:
                        same(native[key], source_arrays[field][query], 'zero-intervention source observation')
                    same(native['actions'], source_arrays['predicted_actions'][query], 'zero-intervention source action')
                    same(native[EFFECTIVE].astype(np.float16), reference[query], 'zero-intervention source HB')
                    for field in ('sim_states_after', 'rewards', 'dones', 'successes'):
                        same(rollout[field][step:step + executed], source_arrays[field][step:step + executed], 'zero-intervention source physics')
                local_checks.append(dict(query=query, accepted=row['decision']['accepted'], monitor_recomputed=True))
                step += executed
            assert step == result['action_steps'] and result['queries'] == parent['query'] + len(rows)
            assert result['opportunities'] == opportunities and result['candidates'] == proposals and result['accepted'] == accepts
            assert result['rejected'] == opportunities - accepts
            assert result['model_calls'] == len(checked) - start_calls
            assert result['deployment_suffix_calls'] == len(rows) + proposals
            assert result['deployment_full_calls'] == parent['query'] + len(rows) + proposals
            assert result['first_alarm'] == monitor.first_v82_alarm == parent['first_alarm']
            expected_exact = result['action_steps'] if arm == 'native' else parent['step']
            assert environment['exact_source_steps'] == expected_exact
            assert environment['exact_source_queries'] == (result['queries'] if arm == 'native' else parent['query'])
            assert environment['settle_steps'] == result['settle_steps'] == 10
            assert environment['frames'] == len(environment['frame_sha256']) == result['action_steps'] + 11
            assert environment['frame_sha256'][:expected_exact + 11] == manifest['source_render']['frame_sha256'][:expected_exact + 11]
            video = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(directory / 'episode.mp4')], text=True))['streams'][0]
            assert int(video['nb_frames']) == environment['frames'] and video['width'] == video['height'] == 256
            if arm == 'native' or accepts == 0:
                assert result['success'] == parent['source_success'] and result['action_steps'] == parent['source_steps']
                assert environment['frame_sha256'] == manifest['source_render']['frame_sha256']
                for field in ('final_image', 'final_wrist_image', 'final_state', 'final_sim_state'):
                    same(rollout[field], source_arrays[field], 'zero-intervention final state')
            branches.append(dict(result, accepted_action_steps=accepted_steps, step_delta_vs_native=result['action_steps'] - parent['source_steps'],
                                 zero_accepted_exact=accepts == 0))
            audits.append(dict(parent=parent['name'], arm=arm, queries=len(rows), accepted=accepts,
                               prefix_physics_exact=True, chosen_history_exact=True, live_actions_exact=True, video_verified=True,
                               first_probe_matches_p3c=arm != 'native', zero_accepted_exact=accepts == 0))
            print(json.dumps(dict(audited=parent['name'], arm=arm, queries=len(rows), accepted=accepts)), flush=True)
    assert checked == set(keyed) and len(audits) == len(collection['results'])
    if partial:
        print(json.dumps(dict(passed=True, partial_read_only=True, completed_branches=len(audits),
                              calls_checked=len(checked), proposals=len(flat_decisions),
                              maximum_cpu_softmax_error=maximum_softmax_error)), flush=True)
        return
    assert sum(r['action_steps'] for r in branches) == collection['environment_action_steps']
    assert sum(r['settle_steps'] for r in branches) == collection['settle_steps'] == 200
    assert sum(r['model_calls'] for r in branches) == collection['model_completed']
    assert sum(r['model_calls'] - r['deployment_suffix_calls'] for r in branches) == 15
    methods = {}
    for arm in ARMS:
        values = [r for r in branches if r['arm'] == arm]
        choices = [r for r in flat_decisions if r['arm'] == arm]
        methods[arm] = dict(successes=sum(r['success'] for r in values), total=5,
            rescues=sum(r['rescue'] for r in values), failed_parents=4, harms=sum(r['harm'] for r in values), successful_parents=1,
            accepted=sum(r['accepted'] for r in values), proposals=sum(r['candidates'] for r in values),
            opportunities=sum(r['opportunities'] for r in values),
            reject_reasons=dict(Counter(reason for r in choices for reason in r['reasons'].split(';') if reason)),
            accepted_by_parent={r['parent']: r['accepted'] for r in values}, steps_by_parent={r['parent']: r['action_steps'] for r in values},
            accepted_action_steps=sum(r['accepted_action_steps'] for r in values),
            model_calls=sum(r['model_calls'] for r in values), deployment_full_calls=sum(r['deployment_full_calls'] for r in values),
            fp32_decision_changes=sum(r['accepted'] != r['fp32_screen'] for r in choices))
        accepted_choices = [r for r in choices if r['accepted']]
        methods[arm]['accepted_effects'] = None if not accepted_choices else dict(
            mean_normalized_action_rms=float(np.mean([r['normalized_action_rms'] for r in accepted_choices])),
            max_normalized_action_rms=float(max(r['normalized_action_rms'] for r in accepted_choices)),
            mean_back_action_set_change_fraction=float(np.mean([r['back_action_set_change_fraction'] for r in accepted_choices])),
            mean_local_instantaneous_delta={c: float(np.mean([r['instant_' + c] for r in accepted_choices])) for c in COMPONENTS},
            mean_local_score_delta={c: float(np.mean([r['score_' + c] for r in accepted_choices])) for c in COMPONENTS},
            gripper_sign_changes=sum(r['gripper_sign_changes'] for r in accepted_choices))
    baseline_calls = methods['native']['deployment_full_calls']
    for item in methods.values():
        item['deployment_call_increase_fraction'] = item['deployment_full_calls'] / baseline_calls - 1
    summary = dict(methods=methods, branches=branches, actual_model_calls=len(responses), actual_rollouts=20,
                   environment_action_steps=collection['environment_action_steps'], settle_steps=200,
                   collection_elapsed_seconds=collection['elapsed_seconds'],
                   inference_seconds=sum(r['resource']['seconds'] for r in responses),
                   peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
                   maximum_cpu_softmax_error=maximum_softmax_error, maximum_instant_precision_delta=maximum_precision_delta,
                   cohort='five previously inspected development parents, paired deterministic simulation',
                   comparison='screened policies with potentially different acceptance frequencies, not equal realized treatment counts')
    for name, values in (('branches.csv', branches), ('decisions.csv', flat_decisions)):
        with (root / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(root / 'summary.json', jsonable(summary))
    write(root / 'audit.json', dict(passed=True, actual_calls_checked=len(checked), branches=audits,
                                   proposals=len(flat_decisions), old_matrix_first_probes_checked=15,
                                   zero_and_post_checks=10, nonzero_repeats=5,
                                   cpu_softmax_max_error=maximum_softmax_error))
    print(json.dumps(jsonable(dict(passed=True, methods=methods, model_calls=len(checked)))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--partial', action='store_true', help='Read-only audit of completed branches during collection')
    args = parser.parse_args()
    audit(args.run.resolve(), partial=args.partial)
