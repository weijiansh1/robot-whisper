"""Audit the completed failure-only subset without resuming collection."""

import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path

import numpy as np

from audit_replan_window import mobility, paired_noise, read, records
from run_replan_window import P3G, check_config
from run_online_experiment import EVALUATION, equal, jsonable, now, save_json, sha_array
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from collection_routes import (PROBS_KEY, NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY,
                               NATIVE_WEIGHTS_KEY, EFFECTIVE_WEIGHTS_KEY)
from openpi_client import msgpack_numpy as codec
from v82_closed_loop import V82Monitor


def completed_failure_parents(config, results):
    parents = [p for p in config['parents'] if not p['source_success']]
    expected = {(p['name'], r, arm) for p in parents for r in range(config['replicates'])
                for arm in config['arms']}
    found = Counter((r['parent'], r['replicate'], r['arm']) for r in results)
    assert parents and set(found) == expected and all(n == 1 for n in found.values())
    assert all(not r['source_success'] for r in results)
    return parents


def summarize_pairs(pairs):
    assert pairs
    return dict(pairs=len(pairs), native_success=sum(p['native_success'] for p in pairs),
                window_success=sum(p['window_success'] for p in pairs),
                rescues=sum(p['rescue'] for p in pairs),
                native_success_window_failure=sum(p['harm'] for p in pairs),
                paired_success_gain=sum(p['gain'] for p in pairs) / len(pairs))


def audit(root):
    config = check_config(root)
    assert not (root / 'collection.json').exists(), 'Do not relabel the aborted full collection'
    failure, loaded = read(root / 'failure.json'), read(root / 'model-load.json')
    parents = completed_failure_parents(config, failure['results'])
    assert len(parents) == 6 and config['replicates'] == 4
    assert config['flow_steps'] == config['generated_horizon'] == 10
    assert config['short_execution'] == 5 and config['window_physical_steps'] == 20
    assert loaded['metadata']['flow_steps'] == loaded['metadata']['predicted_action_steps'] == 10
    assert not loaded['requires_grad']
    assert loaded['parameter_sha256'] == read(P3G / 'collection.json')['parameter_sha256_after']

    events = records(root / 'calls.jsonl')
    requests, responses = events[::2], events[1::2]
    assert len(events) == 2 * len(responses)
    assert len(responses) == failure['model_attempts'] == failure['model_completed']
    assert len(responses) <= config['maximum_model_calls']
    calls = {}
    for ordinal, (request, response) in enumerate(zip(requests, responses)):
        assert request['event'] == 'request' and response['event'] == 'response'
        for key in ('ordinal', 'parent', 'replicate', 'arm', 'step', 'validation', 'noise_sha256'):
            assert request[key] == response[key]
        assert ordinal == response['ordinal'] and response['path'] not in calls
        assert sha256_file(root / response['path']) == response['sha256']
        calls[response['path']] = response
    used, pairs, branches, aligned = set(), [], [], []
    for parent in parents:
        manifest, source = load_episode_trace(Path(parent['source']))
        source_actions = np.concatenate([a[:int(n)] for a, n in
                                         zip(source['predicted_actions'], source['executed_lengths'])])
        assert not source['successes'].any() and len(source_actions) == 520
        for key, value in parent['policy_identity'].items():
            assert loaded['metadata'][key] == value
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as saved:
            original = saved['hb_router_probs']
        snapshot_sha = None
        for replicate in range(4):
            endpoints, first_calls, first_prefixes, common_noises = {}, {}, {}, {}
            for arm in ('native10', 'window5'):
                directory = root / parent['name'] / ('r%d' % replicate) / arm
                result, environment = read(directory / 'result.json'), read(directory / 'environment.json')
                rows = records(directory / 'queries.jsonl')
                assert result in failure['results']
                assert result['parent'] == parent['name'] and result['start'] == parent['start']
                assert result['group'] == [parent['base_task_id'], parent['init_state_id']]
                assert result['stratum'] == parent['stratum']
                assert result['replicate'] == replicate and result['arm'] == arm
                assert result['validation_calls'] == int(replicate == 0 and arm == 'native10')
                state_bytes = (directory / 'fork-state.msgpack').read_bytes()
                state_sha = hashlib.sha256(state_bytes).hexdigest()
                assert state_sha == result['prefix_state_audit']['sha256']
                if snapshot_sha is None:
                    snapshot_sha = state_sha
                assert snapshot_sha == state_sha
                snapshot = codec.unpackb(state_bytes)
                assert set(snapshot) == set(result['prefix_state_audit']['sections'])
                for key, value in snapshot.items():
                    assert hashlib.sha256(codec.packb(value)).hexdigest() == result['prefix_state_audit']['sections'][key]
                assert result['prefix_state_audit']['integration_spec'] == 8191
                assert len(result['prefix_state_audit']['controller_fields'][0]) > 40
                monitor = V82Monitor()
                for p in original[:parent['query']]:
                    monitor.update(p)
                step, previous = parent['start'], original[parent['query'] - 1]
                elapsed_infer = elapsed_env = 0.
                shortened_steps = 0
                with np.load(directory / 'rollout.npz') as rollout:
                    for key in ('sim_states_after', 'rewards', 'dones', 'successes'):
                        equal(rollout[key][:step], source[key][:step], 'Exact source prefix ' + key)
                    equal(rollout['actions'][:step], source_actions[:step], 'Exact source actions prefix')
                    assert len(rollout['integration_after']) == len(rollout['actions'])
                    equal(snapshot['integration'], rollout['integration_after'][step - 1], 'Fork integration')
                    first_prefixes[arm] = {key: rollout[key][:step + 5].copy()
                                           for key in ('actions', 'integration_after', 'sim_states_after')}
                    for index, row in enumerate(rows):
                        assert row['step'] == step and row['replicate'] == replicate and row['arm'] == arm
                        length = 5 if arm == 'window5' and step < parent['start'] + 20 else 10
                        assert row['planned_execution'] == length and row['suffix_discarded'] == 10 - length
                        call = calls[row['path']]
                        assert row['path'] not in used and not call['validation']
                        used.add(row['path'])
                        for key in ('parent', 'replicate', 'arm', 'step', 'state_audit'):
                            assert call[key] == row[key]
                        with np.load(root / row['path']) as trace:
                            equal(trace['request_noise'], paired_noise(parent, replicate, step), 'Physical-time noise')
                            assert sha_array(trace['request_noise']) == call['noise_sha256']
                            assert sha_array(trace['actions']) == call['actions_sha256']
                            assert sha_array(trace[PROBS_KEY]) == call['hb_sha256']
                            assert trace['actions'].shape == (10, 7) and trace['actions'].dtype == np.float32
                            assert trace[PROBS_KEY].shape == (8, 10, 11, 32)
                            assert np.isfinite(trace['actions']).all() and np.isfinite(trace[PROBS_KEY]).all()
                            equal(trace[NATIVE_IDS_KEY], trace[EFFECTIVE_IDS_KEY], 'Native expert dispatch')
                            equal(trace[NATIVE_WEIGHTS_KEY], trace[EFFECTIVE_WEIGHTS_KEY], 'Native gate weights')
                            equal(trace['sim_before'], rollout['sim_states_after'][step - 1], 'New observation state')
                            assert 1 <= row['executed'] <= length
                            equal(rollout['actions'][step:step + row['executed']], trace['actions'][:row['executed']],
                                  'Fresh prefix executed without rescaling')
                            if index == 0:
                                first_calls[arm] = (call['noise_sha256'], call['actions_sha256'], call['hb_sha256'])
                                assert row['state_audit'] == result['prefix_state_audit']
                                for key, value in snapshot['observation'].items():
                                    if isinstance(value, np.ndarray):
                                        equal(trace[key], value, 'Fork policy observation')
                            else:
                                assert rows[index - 1]['next_state_audit'] == row['state_audit']
                            if step % 10 == 0:
                                status = jsonable(monitor.update(trace[PROBS_KEY]))
                                assert row['monitor'] == status and monitor.v7.query == step // 10
                                aligned.append(dict(parent=parent['name'], replicate=replicate, arm=arm, step=step,
                                                    offset=step - parent['start'], mobility_10steps=mobility(previous, trace[PROBS_KEY])))
                                previous = trace[PROBS_KEY].copy()
                                if step in common_noises:
                                    assert common_noises[step] == call['noise_sha256']
                                common_noises[step] = call['noise_sha256']
                            else:
                                assert row['monitor'] is None
                            if arm == 'native10' and replicate == 0:
                                equal(trace['actions'], source['predicted_actions'][step // 10], 'Source continuation')
                                equal(trace[PROBS_KEY], original[step // 10], 'Source routes')
                                for key, source_key in (('observation/image', 'images'), ('observation/wrist_image', 'wrist_images'),
                                                        ('observation/state', 'states')):
                                    equal(trace[key], source[source_key][step // 10], 'Source observation')
                                if index == 0:
                                    path = str((directory / 'queries' / ('s%03d-repeat.npz' % step)).relative_to(root))
                                    assert calls[path]['validation'] and path not in used
                                    used.add(path)
                                    with np.load(root / path) as duplicate:
                                        assert set(duplicate.files) == set(trace.files)
                                        for key in trace.files:
                                            equal(duplicate[key], trace[key], 'Duplicate native inference')
                        if length == 5:
                            shortened_steps += row['executed']
                        step += row['executed']
                        assert row['inference_seconds'] == call['resource']['seconds']
                        elapsed_infer += call['resource']['seconds']
                        elapsed_env += row['environment_seconds']
                        assert row['success'] == bool(rollout['successes'][step - 1])
                        if row['executed'] < length:
                            assert index == len(rows) - 1 and (row['success'] or step == 520)
                        for key in ('goals', 'grasp', 'eef', 'object_position'):
                            assert np.array_equal(row['physics'][key], rollout['physical_' + key][row['step']])
                            assert np.array_equal(row['next_physics'][key], rollout['physical_' + key][step])
                    assert step == result['action_steps'] == environment['steps'] == len(rollout['actions'])
                    assert result['suffix_action_steps'] == step - parent['start']
                    assert result['success'] == environment['success'] == bool(rollout['successes'][-1])
                    assert bool(np.all(rollout['physical_goals'][-1])) == result['success']
                    assert result['success'] or step == 520
                    assert not np.any(rollout['successes'][:-1])
                    assert result['window_steps'] == shortened_steps == (min(20, step - parent['start']) if arm == 'window5' else 0)
                    assert result['suffix_calls'] == len(rows)
                    assert result['deployment_full_calls'] == parent['query'] + len(rows)
                    assert result['inference_seconds'] == elapsed_infer and result['environment_seconds'] == elapsed_env
                    assert result['first_v82_physical_step'] == monitor.first_v82_alarm * 10
                    assert environment['frames'] == len(environment['frame_sha256']) == step + 11
                    assert environment['settle_steps'] == 10
                    equal(rollout['final_sim_state'], rollout['sim_states_after'][-1], 'Final state')
                    if arm == 'native10' and replicate == 0:
                        for key in ('sim_states_after', 'rewards', 'dones', 'successes'):
                            equal(rollout[key], source[key], 'Entire source continuation ' + key)
                        equal(rollout['actions'], source_actions, 'Entire source action stream')
                        assert environment['frame_sha256'] == manifest['source_render']['frame_sha256']
                assert rows[-1]['next_state_audit'] == result['terminal_state_audit'] == environment['final_state_audit']
                endpoints[arm] = result
                branches.append(result)
            assert first_calls['native10'] == first_calls['window5']
            for key in first_prefixes['native10']:
                equal(first_prefixes['native10'][key], first_prefixes['window5'][key], 'Paired prefix plus first 5 steps')
            native, short = endpoints['native10'], endpoints['window5']
            pairs.append(dict(parent=parent['name'], group='%02d/%03d' % tuple(native['group']),
                              stratum=parent['stratum'], replicate=replicate, source_success=False,
                              native_success=native['success'], window_success=short['success'],
                              gain=int(short['success']) - int(native['success']),
                              rescue=not native['success'] and short['success'], harm=native['success'] and not short['success'],
                              native_steps=native['action_steps'], window_steps=short['action_steps'],
                              native_calls=native['suffix_calls'], window_calls=short['suffix_calls'],
                              native_inference_seconds=native['inference_seconds'], window_inference_seconds=short['inference_seconds'],
                              native_first_alarm=native['first_v82_physical_step'], window_first_alarm=short['first_v82_physical_step']))
        print('Audited %s: 4 pairs, %d calls checked' % (parent['name'], len(used)), flush=True)

    # The aborted successful-parent branch is retained as resource use, never an endpoint.
    excluded = [r for path, r in calls.items() if path not in used]
    assert len(pairs) == 24 and len(branches) == 48
    assert set(used) == {path for path, r in calls.items() if r['parent'] in {p['name'] for p in parents}}
    assert len(excluded) == 12 and {r['parent'] for r in excluded} == {'plus-task03-init039'}
    assert {(r['replicate'], r['arm']) for r in excluded} == {(0, 'native10')}
    partial = root / 'plus-task03-init039/r0/native10'
    partial_rows = records(partial / 'queries.jsonl')
    partial_failure = read(partial / 'environment-failure.json')
    assert partial_failure['action_steps'] == 320 and len(partial_rows) == 10
    assert not (partial / 'result.json').exists() and not (partial / 'rollout.npz').exists()
    assert [r['step'] for r in partial_rows] == list(range(220, 320, 10))
    assert all(r['executed'] == 10 and not r['success'] for r in partial_rows)
    assert sum(r['validation'] for r in excluded) == 1
    unexecuted = [r for r in excluded if not r['validation'] and r['path'] not in {q['path'] for q in partial_rows}]
    assert len(unexecuted) == 1 and unexecuted[0]['step'] == 320
    assert 'source flat actions differs' in failure['error']
    original_success_names = [p['name'] for p in config['parents'] if p['source_success']]
    assert original_success_names == ['plus-task03-init039', 'plus-task00-init047']
    assert not (root / original_success_names[1]).exists()
    assert set(root.glob('*/r*/*/result.json')) == {
        root / r['parent'] / ('r%d' % r['replicate']) / r['arm'] / 'result.json' for r in branches}

    native_calls = sum(p['native_calls'] for p in pairs)
    window_calls = sum(p['window_calls'] for p in pairs)
    duplicate_calls = sum(r['validation_calls'] for r in branches)
    assert len(used) == native_calls + window_calls + duplicate_calls
    summary = dict(passed=True, scope='completed original-failure parents only', utc=now(),
                   request='不用跑成功的，重点是失败的会不会成功',
                   original_collection_completed=False, originally_planned_rollouts=config['actual_rollouts'],
                   actual_completed_rollouts=len(branches), snapshots=len(parents),
                   task_init_groups=len({p['group'] for p in pairs}),
                   base_tasks=sorted({p['base_task_id'] for p in parents}),
                   all_noise=summarize_pairs(pairs),
                   historical_noise=summarize_pairs([p for p in pairs if p['replicate'] == 0]),
                   fresh_noise=summarize_pairs([p for p in pairs if p['replicate'] != 0]),
                   original_success_harm_assessment='not performed under updated scope',
                   native_suffix_calls=native_calls, window_suffix_calls=window_calls,
                   extra_suffix_call_fraction=window_calls / native_calls - 1,
                   equivalent_full_calls_by_arm={arm: sum(r['deployment_full_calls'] for r in branches if r['arm'] == arm)
                                                 for arm in ('native10', 'window5')},
                   suffix_inference_seconds_by_arm={arm: sum(r['inference_seconds'] for r in branches if r['arm'] == arm)
                                                   for arm in ('native10', 'window5')},
                   completed_subset_model_calls=len(used), duplicate_validation_calls=duplicate_calls,
                   completed_subset_action_steps=sum(r['action_steps'] for r in branches),
                   completed_subset_settle_steps=10 * len(branches), raw_model_calls=len(calls),
                   excluded_partial=dict(parent='plus-task03-init039', completed_rollouts=0, model_calls=len(excluded),
                                         validation_calls=1, executed_queries=10, unexecuted_query_step=320,
                                         action_steps=320, settle_steps=10, error=partial_failure['error']),
                   parameter_sha256_before=loaded['parameter_sha256'], parameter_sha256_after=None,
                   parameter_checks='initial digest matches sealed P3g; parameter versions asserted after every forward; no terminal digest',
                   prior_source_hashes_checked=len(config['source_hashes']),
                   first_layer_only=True, online_moe_timing_tested=False,
                   uncertainty='6 purposively selected task/init states, not 24 independent initial states; zero observed rescue is not proof of zero effect',
                   pairs=pairs)
    save_json(root / 'failure-only-audit.json', summary)
    for name, rows in (('failure-only-pairs.csv', pairs), ('failure-only-physical-time-mobility.csv', aligned)):
        with (root / name).open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Failure-only audit passed: %s' % summary['all_noise'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run.resolve())
