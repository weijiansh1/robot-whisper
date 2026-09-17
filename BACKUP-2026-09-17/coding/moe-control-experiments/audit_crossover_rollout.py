"""Audit actual crossover rollouts, including per-policy feedback and outcomes."""

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from run_crossover_rollout import check_config, P3C, TRACE_KEYS, FULL_KEYS, LOGIT_KEYS
from crossover_capture_v2 import OUTPUT_MASK
from audit_online_experiment import read, write, digest, array_digest, same
from audit_response_matrix import load_arrays, independent_bank, audit_gate, measurement
from audit_gated_rollout_experiment import independently_decide, compare_measurement
from analyze_crossover_experiment import audit_dispatch, exact_fields
from run_online_experiment import EVALUATION, jsonable
from himoe_libero_bridge.episode_trace import load_episode_trace
from v82_closed_loop import V82Monitor
from v8_feature_control import NATIVE_PROBS, EFFECTIVE_PROBS

ARMS = ('native', 'joint', 'route_only', 'output_only')
OBSERVATIONS = ('observation/image', 'observation/wrist_image', 'observation/state')


def events(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.endswith('\n')]


def independent_guard(native, joint, std):
    delta = (joint['actions'][:, :6].astype(np.float64) - native['actions'][:, :6]) / std
    rms = float(np.sqrt(np.square(delta).sum() / 60))
    gripper = int(np.count_nonzero((joint['actions'][:, 6] >= 0) != (native['actions'][:, 6] >= 0)))
    reasons = (['action_rms'] if rms > .05 else []) + (['gripper_change'] if gripper else [])
    return dict(accepted=not reasons, reasons=reasons, normalized_action_rms=rms,
                gripper_sign_changes=gripper, source='joint_donor_actions', route_scores_used=False)


def audit(root, partial=False):
    import torch
    torch.set_num_threads(1)
    config = check_config(root)
    if partial:
        completed = []
        for parent in config['parents']:
            for arm in ARMS:
                path = root / parent['name'] / arm / 'result.json'
                if path.exists():
                    try:
                        completed.append(read(path))
                    except json.JSONDecodeError:
                        pass
        collection = dict(results=completed)
    else:
        collection = read(root / 'collection.json')
        assert collection['passed'] and collection['actual_rollouts'] == len(collection['results']) == 20
        assert collection['parameter_sha256_before'] == collection['parameter_sha256_after']
        assert collection['parameter_versions_unchanged']
    complete_keys = {(r['parent'], r['arm']) for r in collection['results']}
    calls = [r for r in events(root / 'calls.jsonl') if (r['parent'], r['arm']) in complete_keys]
    requests, responses = calls[::2], calls[1::2]
    assert len(requests) == len(responses) <= config['maximum_model_calls']
    if not partial:
        assert len(responses) == collection['model_attempts'] == collection['model_completed']
    keyed = {}
    for i, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        assert request['ordinal'] == response['ordinal'] == i
        for key in ('parent', 'arm', 'query', 'kind', 'biased', 'traced', 'noise_sha256', 'bias_sha256', 'donor_path', 'donor_sha256'):
            assert request[key] == response[key], key
        assert response['hooks_removed'] and response['parameter_versions_unchanged']
        key = (response['parent'], response['arm'], response['query'], response['kind'])
        assert key not in keyed
        keyed[key] = response
    std = np.asarray(config['shared_metadata']['normalization_action_std'][:6], np.float64)
    checked, branches, flat, query_stats, branch_checks = set(), [], [], [], []
    maximum_softmax, validation_calls, cross_pairs = 0., 0, 0

    def response_at(parent, arm, query, kind, noises, rollout):
        nonlocal maximum_softmax
        key = (parent['name'], arm, query, kind)
        assert key not in checked
        checked.add(key)
        call = keyed[key]
        assert digest(root / call['path']) == call['sha256'], call['path']
        raw = load_arrays(root / call['path'])
        same(raw['request_noise'], noises[query], 'fixed query-index noise')
        same(raw['sim_before'], rollout['sim_states_after'][query * 10 - 1], 'actual state before inference')
        for field, value in (('noise_sha256', raw['request_noise']), ('bias_sha256', raw['request_bias']),
                             ('actions_sha256', raw['actions']), ('effective_hb_sha256', raw[EFFECTIVE_PROBS].astype(np.float16))):
            assert array_digest(value) == call[field]
        assert raw['actions'].shape == (10, 7) and raw['actions'].dtype == np.float32 and np.isfinite(raw['actions']).all()
        bias = raw['request_bias']
        assert bias.shape == (8, 10, 11, 32) and bias.dtype == np.float32 and np.isfinite(bias).all()
        assert np.max(np.abs(bias)) <= .3000001 and np.all(bias[:4] == 0) and np.all(bias[:, :, 0] == 0)
        audit_dispatch(raw, call['biased'])
        if call['biased']:
            maximum_softmax = max(maximum_softmax, audit_gate(raw, call))
        else:
            assert np.all(bias == 0)
        if call['traced']:
            for name in TRACE_KEYS + ('crossover/raw_total',):
                value = raw[name]
                shape = (10, 10, 24) if name.endswith('/velocity') else (10, 10, 1024) if name.endswith('/projection_input') else (8, 10, 11, 1024)
                assert value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all()
            same(raw['mechanism/residual'] + raw['mechanism/total'], raw['mechanism/block'], 'effective residual reconstruction')
            mask = OUTPUT_MASK if call['donor_path'] else np.zeros_like(OUTPUT_MASK)
            same(raw['crossover/replay_token_mask'], mask, 'explicit replay token mask')
            same(raw['crossover/raw_total'][~mask], raw['mechanism/total'][~mask], 'no off-scope output patch')
            sites = [(layer, step) for step in range(10) for layer in (12, 13, 14, 15)] if call['donor_path'] else []
            same(raw['crossover/patch_sites'], np.asarray(sites, np.int16).reshape(-1, 2), 'patch call order')
        else:
            assert call['donor_path'] is None and not any(k.startswith('mechanism/') for k in raw)
        return raw, call

    for parent in config['parents']:
        manifest, source = load_episode_trace(Path(parent['source']))
        reference = load_arrays(EVALUATION / parent['name'] / 'full-hb-routes.npz')['hb_router_probs']
        rng = np.random.default_rng(manifest['config']['flow_noise_seed'])
        noises = np.stack([rng.standard_normal((10, 24)).astype(np.float32) for _ in range(52)])
        same(noises[:len(source['flow_noises'])], source['flow_noises'], 'source noise stream')
        prefix = V82Monitor()
        for p in reference[:parent['query']]:
            prefix.update(p)
        assert prefix.first_v82_alarm == parent['first_alarm'] == parent['query'] - 1
        for arm in ARMS:
            if (parent['name'], arm) not in complete_keys:
                continue
            directory = root / parent['name'] / arm
            result, environment = read(directory / 'result.json'), read(directory / 'environment.json')
            assert result in collection['results']
            rollout = load_arrays(directory / 'rollout.npz')
            steps = result['action_steps']
            assert steps == environment['steps'] == len(rollout['actions'])
            assert result['success'] == environment['success'] == bool(rollout['successes'][-1])
            assert not rollout['successes'][:-1].any() and (result['success'] or steps == 520)
            assert result['rescue'] == (not parent['source_success'] and result['success'])
            assert result['harm'] == (parent['source_success'] and not result['success'])
            for field in ('sim_states_after', 'rewards', 'dones', 'successes'):
                same(rollout[field][:parent['step']], source[field][:parent['step']], 'common environment prefix')
            same(rollout['actions'][:parent['step']], source['predicted_actions'][:parent['query']].reshape(-1, 7), 'common action prefix')
            rows, before_steps = events(directory / 'queries.jsonl'), events(directory / 'decisions.jsonl')
            assert len(rows) == len(before_steps) == result['suffix_queries']
            monitor, previous = copy.deepcopy(prefix), reference[parent['query'] - 1]
            opportunities = proposals = accepted = cross_calls = accepted_steps = old_rejections = 0
            checked_before, step = len(checked), parent['step']
            for offset, (row, frozen) in enumerate(zip(rows, before_steps)):
                query = parent['query'] + offset
                assert {k: v for k, v in row.items() if k not in ('executed', 'success')} == frozen
                assert row['frozen_before_environment_step'] and row['query'] == query and row['step'] == step == query * 10
                assert row['previous_route_sha256'] == array_digest(previous)
                native, native_call = response_at(parent, arm, query, 'native', noises, rollout)
                assert row['native_path'] == native_call['path']
                active = arm != 'native' and offset < 12
                assert row['active'] == active and native_call['traced'] == (active or offset == 0)
                if offset == 0:
                    with np.load(P3C / 'states' / parent['name'] / 'native.npz', allow_pickle=False) as prior:
                        exact_fields(native, prior, FULL_KEYS, 'P3c initial native')
                joint = trial = joint_call = trial_call = decision = None
                selected, selected_call = native, native_call
                if active:
                    opportunities += 1
                    try:
                        bank, energy = independent_bank(native[EFFECTIVE_PROBS].astype(np.float16), previous,
                            dict(parent, query=query), [dict(operator='mobility_balance', randomized=False, fraction=1., sign=1)])
                    except AssertionError:
                        assert row['generation'] == dict(fallback_reason='degenerate_direction')
                        assert row['joint_path'] is None and row['trial_path'] is None and row['guard'] is None and row['diagnostic'] is None
                    else:
                        joint, joint_call = response_at(parent, arm, query, 'joint', noises, rollout)
                        proposals += 1
                        assert row['joint_path'] == joint_call['path'] and joint_call['traced'] and joint_call['biased']
                        exact_fields(joint, native, OBSERVATIONS + ('sim_before', 'request_noise'), 'same live candidate input')
                        np.testing.assert_allclose(joint['request_bias'], bank[0], atol=2e-8, rtol=1e-7)
                        np.testing.assert_allclose(row['generation']['high_energy'], energy, rtol=1e-12)
                        assert row['generation']['label'] == 'mobility_balance-high-plus' and row['generation']['query'] == query
                        if offset == 0:
                            with np.load(P3C / 'states' / parent['name'] / 'mobility_balance-high-plus.npz', allow_pickle=False) as prior:
                                exact_fields(joint, prior, FULL_KEYS + LOGIT_KEYS + ('request_bias',), 'P3c initial candidate')
                        trial, trial_call = joint, joint_call
                        if arm != 'joint':
                            trial, trial_call = response_at(parent, arm, query, 'cross', noises, rollout)
                            cross_calls += 1
                            cross_pairs += 1
                            donor, donor_call = (native, native_call) if arm == 'route_only' else (joint, joint_call)
                            assert trial_call['donor_path'] == donor_call['path']
                            assert trial_call['donor_sha256'] == array_digest(donor['mechanism/total'])
                            assert trial_call['biased'] == (arm == 'route_only')
                            same(trial['request_bias'], joint['request_bias'] if arm == 'route_only' else np.zeros_like(bank[0]), 'cross gate rule')
                            exact_fields(trial, native, OBSERVATIONS + ('sim_before', 'request_noise'), 'same live crossover input')
                            exact_fields(trial, donor, TRACE_KEYS + ('actions',), 'cross effective compute matching')
                            same(trial['mechanism/total'][OUTPUT_MASK], donor['mechanism/total'][OUTPUT_MASK], 'exact donor output injection')
                            if arm == 'route_only':
                                same(trial[NATIVE_PROBS], native[NATIVE_PROBS], 'native input with biased gate')
                            else:
                                same(trial[EFFECTIVE_PROBS], joint[NATIVE_PROBS], 'joint input with original gate')
                        assert row['trial_path'] == trial_call['path']
                        safety = independent_guard(native, joint, std)
                        for key in ('accepted', 'reasons', 'gripper_sign_changes', 'source', 'route_scores_used'):
                            assert row['guard'][key] == safety[key]
                        np.testing.assert_allclose(row['guard']['normalized_action_rms'], safety['normalized_action_rms'], rtol=1e-14, atol=1e-16)
                        decision = independently_decide(monitor, native, trial, 'mobility_balance', std)
                        for key in ('accepted', 'reasons', 'operator', 'formal_screen', 'fp32_screen', 'gripper_sign_changes'):
                            assert row['diagnostic'][key] == decision[key]
                        for key in ('score_delta', 'instantaneous_delta', 'fp32_instantaneous_delta'):
                            np.testing.assert_array_equal(row['diagnostic'][key], decision[key])
                        np.testing.assert_allclose(row['diagnostic']['normalized_action_rms'], decision['normalized_action_rms'], rtol=1e-14, atol=1e-16)
                        for key in ('native', 'candidate'):
                            compare_measurement(row['diagnostic'][key], decision[key])
                        if safety['accepted']:
                            selected, selected_call = trial, trial_call
                            accepted += 1
                            old_rejections += not decision['accepted']
                        flat.append(dict(parent=parent['name'], arm=arm, query=query, accepted=safety['accepted'],
                            guard_reasons=';'.join(safety['reasons']), old_screen=decision['accepted'], fp32_old_screen=decision['fp32_screen'],
                            donor_action_rms=safety['normalized_action_rms'], executed_candidate_action_rms=decision['normalized_action_rms'],
                            top4_set_change=float(np.any(np.sort(trial['collection/hb_effective_ids'][4:, :, 1:], axis=-1) !=
                                                       np.sort(native['collection/hb_effective_ids'][4:, :, 1:], axis=-1), axis=-1).mean())))
                else:
                    assert row['generation'] is row['guard'] is row['diagnostic'] is None
                    assert row['joint_path'] is row['trial_path'] is None
                if offset == 0:
                    if arm == 'native':
                        control, control_call = response_at(parent, arm, query, 'zero', noises, rollout)
                        baseline = native
                        assert control_call['donor_path'] is None and np.all(control['request_bias'] == 0)
                    else:
                        control, control_call = response_at(parent, arm, query, 'repeat', noises, rollout)
                        baseline = trial
                        assert control_call['biased'] == trial_call['biased']
                        assert control_call['donor_path'] == trial_call['donor_path'] and control_call['donor_sha256'] == trial_call['donor_sha256']
                        same(control['request_bias'], trial['request_bias'], 'repeat requested bias')
                        if arm != 'output_only':
                            exact_fields(control, trial, LOGIT_KEYS, 'repeated gate logits')
                    exact_fields(control, native, OBSERVATIONS + ('sim_before', 'request_noise'), 'same control observation')
                    exact_fields(control, baseline, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',), 'zero or repeated compute')
                    post, post_call = response_at(parent, arm, query, 'post', noises, rollout)
                    exact_fields(post, native, FULL_KEYS + TRACE_KEYS + ('crossover/raw_total',) + OBSERVATIONS, 'clean post')
                    assert not post_call['biased'] and post_call['donor_path'] is None
                    validation_calls += 2
                assert row['selected_path'] == selected_call['path']
                assert row['intervention_accepted'] == (selected_call['path'] != native_call['path'])
                measured = measurement(monitor, selected[EFFECTIVE_PROBS].astype(np.float16))
                compare_measurement(row['selected'], measured)
                previous = selected[EFFECTIVE_PROBS].astype(np.float16)
                assert jsonable(monitor.update(previous)) == row['selected']['status']
                executed = row['executed']
                assert 1 <= executed <= 10 and (executed == 10 or offset == len(rows) - 1)
                same(rollout['actions'][step:step + executed], selected['actions'][:executed], 'actual selected actions')
                assert row['success'] == bool(rollout['successes'][step + executed - 1])
                if row['intervention_accepted']:
                    accepted_steps += executed
                if arm in ('native', 'route_only'):
                    for key, field in zip(OBSERVATIONS, ('images', 'wrist_images', 'states')):
                        same(native[key], source[field][query], 'exact C0 observation')
                    same(native['actions'], source['predicted_actions'][query], 'exact C0 action')
                    same(native[EFFECTIVE_PROBS].astype(np.float16), reference[query], 'exact C0 shadow route')
                query_stats.append(dict(parent=parent['name'], arm=arm, query=query,
                    observation_sha256=[array_digest(native[k]) for k in OBSERVATIONS],
                    selected_action_sha256=selected_call['actions_sha256'], selected_route_sha256=selected_call['effective_hb_sha256'],
                    proposal_bias_sha256=None if joint_call is None else joint_call['bias_sha256'],
                    accepted=row['intervention_accepted'], formal_normalized=measured['normalized'].tolist()))
                step += executed
            assert step == result['action_steps'] and result['queries'] == parent['query'] + len(rows)
            for field, value in (('opportunities', opportunities), ('proposals', proposals), ('accepted', accepted), ('cross_calls', cross_calls)):
                assert result[field] == value
            assert result['model_calls'] == len(checked) - checked_before
            assert result['deployment_suffix_calls'] == len(rows) + proposals + cross_calls
            assert result['deployment_full_calls'] == parent['query'] + result['deployment_suffix_calls']
            assert result['model_calls'] - result['deployment_suffix_calls'] == 2
            assert result['first_alarm'] == monitor.first_v82_alarm == parent['first_alarm']
            exact_steps = steps if arm in ('native', 'route_only') else parent['step']
            assert environment['exact_source_steps'] == exact_steps
            assert environment['exact_source_queries'] == (result['queries'] if arm in ('native', 'route_only') else parent['query'])
            assert environment['settle_steps'] == result['settle_steps'] == 10
            assert environment['frames'] == len(environment['frame_sha256']) == steps + 11
            assert environment['frame_sha256'][:exact_steps + 11] == manifest['source_render']['frame_sha256'][:exact_steps + 11]
            video = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(directory / 'episode.mp4')], text=True))['streams'][0]
            assert int(video['nb_frames']) == environment['frames'] and video['width'] == video['height'] == 256
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(directory / 'episode.mp4'), '-f', 'null', '-'], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if arm in ('native', 'route_only'):
                assert result['success'] == parent['source_success'] and steps == parent['source_steps']
                for field in ('sim_states_after', 'rewards', 'dones', 'successes', 'final_image', 'final_wrist_image', 'final_state', 'final_sim_state'):
                    same(rollout[field], source[field], 'full C0 rollout reproduction')
            branches.append(dict(result, accepted_action_steps=accepted_steps, executed_despite_old_rejection=old_rejections,
                                 step_delta_vs_native=steps - parent['source_steps']))
            branch_checks.append(dict(parent=parent['name'], arm=arm, calls=len(checked) - checked_before,
                actual_actions_checked=True, actual_route_feedback_checked=True, video_fully_decoded=True, exact_common_prefix=True))
            print(json.dumps(dict(audited=parent['name'], arm=arm, calls=len(checked) - checked_before, success=result['success'])), flush=True)
    assert checked == set(keyed)
    if partial:
        print(json.dumps(dict(passed=True, partial_read_only=True, completed_branches=len(branches), checked_calls=len(checked),
            cross_pairs=cross_pairs, validation_calls=validation_calls)), flush=True)
        return
    assert validation_calls == 40 and sum(r['model_calls'] for r in branches) == len(responses)
    assert sum(r['action_steps'] for r in branches) == collection['environment_action_steps']
    assert sum(r['settle_steps'] for r in branches) == collection['settle_steps'] == 200
    paired = []
    for parent in config['parents']:
        indexed = {arm: {r['query']: r for r in query_stats if r['parent'] == parent['name'] and r['arm'] == arm}
                   for arm in ('joint', 'output_only')}
        common = sorted(set(indexed['joint']) & set(indexed['output_only']))
        first = common[0]
        assert first == parent['query']
        assert indexed['joint'][first]['selected_action_sha256'] == indexed['output_only'][first]['selected_action_sha256']
        assert indexed['joint'][first]['proposal_bias_sha256'] == indexed['output_only'][first]['proposal_bias_sha256']
        if first + 1 in common:
            assert indexed['joint'][first + 1]['observation_sha256'] == indexed['output_only'][first + 1]['observation_sha256']
        def first_difference(field):
            return next((q for q in common if indexed['joint'][q][field] != indexed['output_only'][q][field]), None)
        paired.append(dict(parent=parent['name'], compared_queries=len(common), first_action_equal=True,
            next_observation_equal=True if first + 1 in common else None,
            first_route_difference_query=first_difference('selected_route_sha256'),
            first_bias_difference_query=first_difference('proposal_bias_sha256'),
            first_action_difference_query=first_difference('selected_action_sha256'),
            first_observation_difference_query=first_difference('observation_sha256')))
    methods = {}
    for arm in ARMS:
        values, decisions = [r for r in branches if r['arm'] == arm], [r for r in flat if r['arm'] == arm]
        methods[arm] = dict(successes=sum(r['success'] for r in values), parents=len(values),
            rescues=sum(r['rescue'] for r in values), failed_parents=4, harms=sum(r['harm'] for r in values), successful_parents=1,
            opportunities=sum(r['opportunities'] for r in values), proposals=sum(r['proposals'] for r in values),
            accepted=sum(r['accepted'] for r in values), executed_despite_old_rejection=sum(r['executed_despite_old_rejection'] for r in values),
            rejected_reasons=dict(Counter(s for r in decisions for s in r['guard_reasons'].split(';') if s)),
            model_calls=sum(r['model_calls'] for r in values), deployment_full_calls=sum(r['deployment_full_calls'] for r in values),
            steps_by_parent={r['parent']: r['action_steps'] for r in values})
    baseline = methods['native']['deployment_full_calls']
    for value in methods.values():
        value['deployment_call_increase_fraction'] = value['deployment_full_calls'] / baseline - 1
    summary = dict(methods=methods, branches=branches, decisions=flat, queries=query_stats, paired_feedback=paired,
        actual_rollouts=20, model_calls=len(responses), cross_pairs=cross_pairs, validation_calls=40,
        environment_action_steps=collection['environment_action_steps'], settle_steps=200,
        collection_elapsed_seconds=collection['elapsed_seconds'], inference_seconds=sum(r['resource']['seconds'] for r in responses),
        peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
        comparison='same action guard and window; actual per-arm routes determine subsequent proposals',
        cohort='five previously inspected development parents; no held-out claim')
    for name, values in (('branches.csv', branches), ('decisions.csv', flat)):
        with (root / name).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(root / 'summary.json', jsonable(summary))
    write(root / 'audit.json', dict(passed=True, checked_calls=len(checked), branches=branch_checks, cross_pairs=cross_pairs,
        validation_calls=40, complete_native_and_route_only_reproductions=10, exact_first_joint_output_action_pairs=5,
        actual_route_feedback_checked=True, old_route_screen_not_used_for_selection=True,
        parameter_hash_unchanged=True, maximum_cpu_softmax_error=maximum_softmax))
    print(json.dumps(dict(passed=True, methods=methods, paired_feedback=paired)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    audit(args.run.resolve(), args.partial)
