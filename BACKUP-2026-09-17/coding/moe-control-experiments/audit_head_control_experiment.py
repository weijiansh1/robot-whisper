"""Recompute frozen-monitor head scores, rankings, execution and pool oracle."""

import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path('/data/coding/robot-whisper-0909')
EVALUATION = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')
sys.path[:0] = ['/data/srv/src', str(REPO / 'moe-trap-control')]
from v82_closed_loop import V82Monitor
from intrinsic_guard_monitor import intrinsic_score_arrays
from himoe_libero_bridge.episode_trace import load_episode_trace
from audit_online_experiment import digest, array_digest, independent_scores, read, write, records, same


def score_vector(status):
    return np.array([status['freeze_score'], status['acceleration_score'],
                     status['periodicity_score'], *status['v8_scores']], np.float64)


def normalized(scores, query):
    profile = read(REPO / 'moe-trap-control/design/frozen_alarm_comparison_20260908/profiles/parameters.json')['legacy']
    base = np.array([profile['v7'][name + '_threshold'] for name in ('freeze', 'acceleration', 'periodicity')]
                    + [-profile['v8_thresholds']['frontback_flowpath'], profile['v8_thresholds']['curvature_3step']])
    margin = np.r_[base[:3] * .2, .1, base[4] - .2]
    threshold = base.copy()
    threshold[3:] -= .0015 * query
    return (np.asarray(scores) - threshold) / margin, threshold, margin


def target(values, head):
    if head == 'turbulence':
        return np.maximum(values[..., 1], values[..., 2])
    return values[..., dict(freeze=0, inversion=3, curvature=4)[head]]


def low(values, tolerance):
    return int(np.flatnonzero(values <= np.min(values) + tolerance)[0])


def fresh_prefix(reference, count):
    monitor = V82Monitor()
    for p in reference[:count]:
        monitor.update(p)
    return monitor


def acceleration_window(monitor):
    v7 = monitor.v7
    streams = intrinsic_score_arrays(
        np.asarray(v7._mobility_history, np.float32)[None],
        np.asarray(v7._acceleration_history, np.float32)[None],
        np.asarray(v7._periodicity_history, np.float32)[None],
        v7.profile.periodicity_scale)
    window = streams['acceleration'][0, -8:]
    return dict(raw=float(streams['acceleration_raw'][0, -1]),
                current_smoothed=float(window[-1]),
                persistent=float(streams['acceleration_persistent'][0, -1]),
                minimizing_query=int(v7.query - len(window) + 1 + np.argmin(window)),
                window=window.tolist())


def audit(root):
    config, collection = read(root / 'config.json'), read(root / 'collection.json')
    assert collection['passed'] and digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    calls = records(root / 'calls.jsonl')
    requests = [r for r in calls if r['event'] == 'request']
    responses = [r for r in calls if r['event'] == 'response']
    assert len(requests) == len(responses) == collection['model_completed'] == collection['model_attempts']
    keyed = {}
    for i, (request, response) in enumerate(zip(requests, responses)):
        assert response['ordinal'] == i
        for field in ('parent', 'candidate', 'query', 'kind', 'ordinal', 'noise_sha256'):
            assert request[field] == response[field]
        key = (response['parent'], response['candidate'], response['query'], response['kind'])
        assert key not in keyed
        keyed[key] = response
    checked, branch_audits, pool_rows, method_rows, branches = set(), [], [], [], []
    for parent in config['parents']:
        directory, source = root / parent['name'], Path(parent['source'])
        assert digest(source / 'episode-trace.json') == parent['manifest_sha256']
        assert digest(source / 'episode-trace.npz') == parent['trace_sha256']
        assert digest(EVALUATION / parent['name'] / 'full-hb-routes.npz') == parent['hb_sha256']
        manifest, source_arrays = load_episode_trace(source)
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as archive:
            reference = archive['hb_router_probs']
        with np.load(directory / 'pool.npz', allow_pickle=False) as archive:
            pool = {k: archive[k] for k in archive.files}
        old_pool = (BASE / 'runs/p3-online-20260915-tc0mhlig' / parent['name'] /
                    'random_short/queries' / ('q%03d.npz' % parent['query']))
        if old_pool.is_file():
            with np.load(old_pool, allow_pickle=False) as old:
                for field in ('image', 'wrist_image', 'state', 'sim_before', 'noises', 'actions', 'hb'):
                    same(pool[field], old[field], 'same P3 initial candidate pool')
        decision = read(directory / 'decisions.json')
        assert decision['pool_sha256'] == digest(directory / 'pool.npz')
        assert decision['before_any_suffix_execution']
        prefix = fresh_prefix(reference, parent['query'])
        assert prefix.first_v82_alarm == parent['first_alarm'] == parent['query'] - 1
        actual_head_onset = dict(freeze=prefix.v7.first_freeze_query, turbulence=prefix.v7.first_turbulence_query,
                                 inversion=prefix.v82_first[0], curvature=prefix.v82_first[1])
        assert actual_head_onset[parent['head']] == parent['first_alarm']
        preview_monitors = [copy.deepcopy(prefix) for _ in range(4)]
        preview = [monitor.update(p) for monitor, p in zip(preview_monitors, pool['hb'])]
        score = np.stack([score_vector(status) for status in preview])
        z, thresholds, margins = normalized(score, parent['query'])
        specific = target(z, parent['head'])
        geometry = independent_scores(pool['hb'])
        overall = np.maximum.reduce([z[:, 0], np.minimum(z[:, 1], z[:, 2]), z[:, 3], z[:, 4]])
        for key, value in (('scores', score), ('normalized', z), ('target_severity', specific),
                           ('thresholds', thresholds), ('margins', margins), ('all_risk', overall)):
            np.testing.assert_array_equal(decision[key], value)
        np.testing.assert_allclose(decision['centrality'], geometry, atol=1e-10, rtol=0)
        seed_base = [2026091501, 0 if parent['benchmark'] == 'plus' else 1,
                     parent['base_task_id'], parent['init_state_id'], parent['step']]
        rng = np.random.default_rng(manifest['config']['flow_noise_seed'])
        bank = np.stack([rng.standard_normal((10, 24)).astype(np.float32) for _ in range(52)])
        for candidate in range(4):
            noise = (bank[parent['query']] if candidate == 0 else
                     np.random.default_rng(np.random.SeedSequence(seed_base + [candidate])).standard_normal((10, 24)).astype(np.float32))
            same(noise, pool['noises'][candidate], 'pool noise')
        choices = dict(native=0, random=int(np.random.default_rng(np.random.SeedSequence(seed_base + [100])).integers(4)),
                       center=low(geometry, 1e-12), edge=low(-geometry, 1e-12),
                       head_low=low(specific, 1e-6), head_high=low(-specific, 1e-6), all_low=low(overall, 1e-6))
        assert choices == decision['selected']
        same(pool['actions'][0], source_arrays['predicted_actions'][parent['query']], 'native pool action')
        same(pool['hb'][0], reference[parent['query']], 'native pool routing')
        local_results, future_traces = {}, {}
        for candidate in range(4):
            branch_dir = directory / ('candidate-%d' % candidate)
            result = read(branch_dir / 'result.json')
            assert result in collection['results']
            assert result['selected_by'] == [name for name, value in choices.items() if value == candidate]
            branches.append(result)
            local_results[candidate] = result
            with np.load(branch_dir / 'rollout.npz', allow_pickle=False) as archive:
                rollout = {k: archive[k] for k in archive.files}
            environment = read(branch_dir / 'environment.json')
            assert result['action_steps'] == len(rollout['actions']) == environment['steps']
            assert result['success'] == bool(rollout['successes'][-1]) == environment['success']
            assert not rollout['successes'][:-1].any()
            assert result['success'] or result['action_steps'] == 520
            assert result['rescue'] == (not parent['source_success'] and result['success'])
            assert result['harm'] == (parent['source_success'] and not result['success'])
            same(rollout['sim_states_after'][:parent['step']], source_arrays['sim_states_after'][:parent['step']], 'common prefix physics')
            tape = source_arrays['predicted_actions'][:parent['query']].reshape(-1, 7)
            same(rollout['actions'][:parent['step']], tape, 'common prefix actions')
            monitor = copy.deepcopy(prefix)
            rows = records(branch_dir / 'queries.jsonl')
            step = parent['step']
            expected_future = []
            for offset, row in enumerate(rows):
                query = parent['query'] + offset
                assert row['query'] == query and row['step'] == step == query * 10
                assert row['candidate'] == candidate
                path = root / row['archive']
                assert digest(path) == row['archive_sha256']
                if offset == 0:
                    assert row['origin'] == 'pool'
                    action, hb, noise = pool['actions'][candidate], pool['hb'][candidate], pool['noises'][candidate]
                    observation = pool
                    key = (parent['name'], candidate, query, 'pool')
                else:
                    assert row['origin'] == 'continuation'
                    with np.load(path, allow_pickle=False) as archive:
                        observation = {k: archive[k] for k in archive.files}
                    action, hb, noise = observation['actions'], observation['hb'], observation['noise']
                    same(noise, bank[query], 'fixed continuation noise')
                    key = (parent['name'], candidate, query, 'continuation')
                assert key not in checked
                checked.add(key)
                call = keyed[key]
                for field, value in (('actions_sha256', action), ('hb_sha256', hb), ('noise_sha256', noise)):
                    assert call[field] == array_digest(value)
                same(observation['sim_before'], rollout['sim_states_after'][step - 1], 'live observation state')
                status = monitor.update(hb)
                actual_z, _, _ = normalized(score_vector(status), query)
                np.testing.assert_array_equal(actual_z, row['normalized'])
                severity = float(target(actual_z, parent['head']))
                assert severity == row['head_severity']
                for field in ('query', 'freeze_score', 'acceleration_score', 'periodicity_score', 'v82_first', 'v82_alarm'):
                    assert status[field] == row['monitor'][field]
                np.testing.assert_array_equal(status['v8_scores'], row['monitor']['v8_scores'])
                executed = row['executed']
                assert 1 <= executed <= 10
                assert executed == 10 or offset == len(rows) - 1
                same(rollout['actions'][step:step + executed], action[:executed], 'actually executed action block')
                assert row['success'] == bool(rollout['successes'][step + executed - 1])
                if candidate == 0:
                    for key_obs, source_key in (('image', 'images'), ('wrist_image', 'wrist_images'), ('state', 'states')):
                        same(observation[key_obs], source_arrays[source_key][query], 'C0 input')
                    same(action, source_arrays['predicted_actions'][query], 'C0 action')
                    same(hb, reference[query], 'C0 HB')
                    same(rollout['sim_states_after'][step:step + executed], source_arrays['sim_states_after'][step:step + executed], 'C0 physical tail')
                expected_future.append(severity)
                step += executed
            assert step == result['action_steps']
            assert len(rows) == result['suffix_queries']
            assert len(rows) - 1 == result['continuation_calls']
            assert monitor.first_v82_alarm == result['first_alarm'] == parent['first_alarm']
            exact_steps = result['action_steps'] if candidate == 0 else parent['step']
            assert environment['exact_source_steps'] == exact_steps
            assert environment['frame_sha256'][:exact_steps + 11] == manifest['source_render']['frame_sha256'][:exact_steps + 11]
            assert environment['frames'] == result['action_steps'] + 11
            video = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_frames', '-of', 'json', str(branch_dir / 'episode.mp4')], text=True))['streams'][0]
            assert int(video['nb_frames']) == environment['frames'] and video['width'] == video['height'] == 256
            if candidate == 0:
                assert result['success'] == parent['source_success'] and result['action_steps'] == parent['source_steps']
                for field in ('final_image', 'final_wrist_image', 'final_state', 'final_sim_state'):
                    same(rollout[field], source_arrays[field], 'C0 final observation')
            future_traces[candidate] = expected_future
            branch_audits.append(dict(parent=parent['name'], candidate=candidate, queries=len(rows),
                                     live_actions_exact=True, monitor_recomputed=True, video_verified=True,
                                     prefix_steps=parent['step'], c0_exact=candidate == 0))
        for method, candidate in choices.items():
            result = local_results[candidate]
            post = np.asarray(future_traces[candidate][1:7])
            base_post = np.asarray(future_traces[0][1:1 + len(post)])
            width = min(len(post), len(base_post))
            method_rows.append(dict(parent=parent['name'], head=parent['head'], method=method, candidate=candidate,
                                    success=result['success'], rescue=result['rescue'], harm=result['harm'],
                                    action_steps=result['action_steps'], source_steps=parent['source_steps'],
                                    target_at_intervention=float(specific[candidate]),
                                    target_change_from_native=float(specific[candidate] - specific[0]),
                                    component_changes_from_native=(z[candidate] - z[0]).tolist(),
                                    first_next_query_change=float(post[0] - base_post[0]) if width else None,
                                    next_query_comparison_count=width,
                                    mean_next_six_query_change=float(np.mean(post[:width] - base_post[:width])) if width else None))
        pool_rows.append(dict(parent=parent['name'], head=parent['head'], source_success=parent['source_success'],
                              selected=choices, target_severity=specific.tolist(), target_spread=float(np.ptp(specific)),
                              detectable_target_contrast=bool(np.ptp(specific) > 1e-6),
                              action_elements_changed_from_native=[
                                  int(np.count_nonzero(action != pool['actions'][0])) for action in pool['actions']],
                              gripper_sign_changed_steps=[
                                  int(np.count_nonzero((action[:, 6] >= 0) != (pool['actions'][0, :, 6] >= 0)))
                                  for action in pool['actions']],
                              acceleration_window_diagnostic=(
                                  [acceleration_window(monitor) for monitor in preview_monitors]
                                  if parent['head'] == 'turbulence' else None),
                              successful_candidates=[c for c, r in local_results.items() if r['success']],
                              rescuable=(not parent['source_success'] and any(r['success'] for r in local_results.values())),
                              uniform_random_success_expectation=float(np.mean([r['success'] for r in local_results.values()]))))
    assert checked == set(keyed) and len(branch_audits) == 20
    method_summary = {}
    for method in config['methods']:
        selected_rows = [r for r in method_rows if r['method'] == method]
        method_summary[method] = dict(successes=sum(r['success'] for r in selected_rows),
                                      rescued=sum(r['rescue'] for r in selected_rows), failed_parents=4,
                                      harmed=sum(r['harm'] for r in selected_rows), successful_parents=1,
                                      total_action_steps=sum(r['action_steps'] for r in selected_rows),
                                      target_reduced_vs_native=sum(r['target_change_from_native'] < -1e-6 for r in selected_rows))
    summary = dict(methods=method_summary, pools=pool_rows, model_calls=len(responses),
                   actual_rollouts=20, actual_action_steps=sum(r['actual_environment_steps'] for r in branches),
                   actual_settle_steps=sum(r['settle_steps'] for r in branches),
                   oracle_rescuable_failed_parents=sum(p['rescuable'] for p in pool_rows),
                   actual_successful_branches=sum(r['success'] for r in branches),
                   uniform_random_expected_rescues=sum(p['uniform_random_success_expectation'] for p in pool_rows if not p['source_success']),
                   failed_parent_count=4, successful_parent_count=1,
                   elapsed_seconds=collection['elapsed_seconds'],
                   inference_seconds=sum(r['resource']['seconds'] for r in responses),
                   model_peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
                   interpretation='developmental one-query candidate intervention; shared branch outcomes are not independent replications')
    write(root / 'audit.json', dict(passed=True, checked_calls=len(checked), source_files=len(config['source_hashes']),
                                   branches=branch_audits, auditor_sha256=digest(Path(__file__))))
    write(root / 'summary.json', summary)
    write(root / 'method-results.json', method_rows)
    write(root / 'pool-manipulation.json', pool_rows)
    scalar_fields = [k for k in method_rows[0] if k != 'component_changes_from_native']
    with (root / 'method-results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=scalar_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(method_rows)
    print(json.dumps(dict(passed=True, summary=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve())
