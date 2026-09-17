"""Independently reconstruct candidate decisions, execution, and outcome counts."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
REPO = Path('/data/coding/robot-whisper-0909')
EVALUATION = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')
sys.path[:0] = ['/data/srv/src', str(REPO / 'moe-trap-control')]
from himoe_libero_bridge.episode_trace import load_episode_trace
from v82_closed_loop import V82Monitor


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def same(a, b, label):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype != b.dtype or not np.array_equal(a, b):
        raise AssertionError(label)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def records(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def independent_scores(routes):
    p = routes[:, 4:, :3, 1:].astype(np.float64)
    p /= p.sum(-1, keepdims=True)
    roots = np.sqrt(p)
    scores = np.zeros(len(p), np.float64)
    for i in range(len(p)):
        for j in range(len(p)):
            if i != j:
                scores[i] += np.sqrt(.5 * np.square(roots[i] - roots[j]).sum(-1).mean())
    return scores / (len(p) - 1)


def rng(parent, step, stream):
    return np.random.default_rng(np.random.SeedSequence([
        2026091501, 0 if parent['benchmark'] == 'plus' else 1,
        parent['base_task_id'], parent['init_state_id'], step, stream]))


def wilson(hits, total):
    z = 1.959963984540054
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    width = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0., float(center - width)), min(1., float(center + width))]


def audit(root):
    config, collection = read(root / 'config.json'), read(root / 'collection.json')
    assert digest(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    assert collection['passed']
    for path, expected in config['source_hashes'].items():
        assert digest(path) == expected, path
    calls = records(root / 'calls.jsonl')
    requests = [r for r in calls if r['event'] == 'request']
    responses = [r for r in calls if r['event'] == 'response']
    assert len(requests) == len(responses) == collection['model_completed'] == collection['model_attempts']
    keyed_responses = {}
    for ordinal, (request, response) in enumerate(zip(requests, responses)):
        for key in ('parent', 'arm', 'query', 'step', 'candidate', 'ordinal', 'noise_sha256'):
            assert request[key] == response[key], key
        assert request['ordinal'] == ordinal
        key = (response['parent'], response['arm'], response['query'], response['candidate'])
        assert key not in keyed_responses
        keyed_responses[key] = response
    checked_calls, audits, result_rows = set(), [], []
    for parent in config['parents']:
        source = Path(parent['source'])
        assert digest(source / 'episode-trace.json') == parent['source_manifest_sha256']
        assert digest(source / 'episode-trace.npz') == parent['source_trace_sha256']
        assert digest(EVALUATION / parent['name'] / 'full-hb-routes.npz') == parent['hb_sha256']
        manifest, source_arrays = load_episode_trace(source)
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz', allow_pickle=False) as reference:
            hb_reference = reference['hb_router_probs']
        native_rng = np.random.default_rng(manifest['config']['flow_noise_seed'])
        native_noise = np.stack([native_rng.standard_normal((10, 24)).astype(np.float32) for _ in range(52)])
        native_result = read(root / parent['name'] / 'native/result.json')
        for arm in config['arms']:
            directory = root / parent['name'] / arm
            result = read(directory / 'result.json')
            assert result in collection['results']
            result_rows.append(result)
            if result['alias_of']:
                assert parent['first_alarm'] == -1 and result['alias_of'] == 'native'
                assert result['actual_environment_steps'] == result['actual_model_calls'] == 0
                for key in ('success', 'action_steps', 'queries', 'deployed_model_calls', 'active_queries', 'first_alarm'):
                    assert result[key] == native_result[key]
                audits.append(dict(parent=parent['name'], arm=arm, alias_exact=True))
                continue
            with np.load(directory / 'rollout.npz', allow_pickle=False) as archive:
                rollout = {key: archive[key] for key in archive.files}
            environment = read(directory / 'environment.json')
            assert len(rollout['actions']) == result['action_steps'] == environment['steps']
            assert len(rollout['successes']) == len(rollout['actions'])
            assert not rollout['successes'][:-1].any()
            assert bool(rollout['successes'][-1]) == result['success'] == environment['success']
            assert result['success'] or result['action_steps'] == 520
            assert result['rescue'] == (not parent['source_success'] and result['success'])
            assert result['harm'] == (parent['source_success'] and not result['success'])
            assert environment['frames'] == environment['steps'] + 11
            assert (directory / 'episode.mp4').stat().st_size > 0
            video = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=width,height,nb_frames', '-of', 'json', str(directory / 'episode.mp4')],
                text=True, timeout=30))['streams'][0]
            assert int(video['nb_frames']) == environment['frames']
            assert video['width'] == video['height'] == 256
            rows = records(directory / 'queries.jsonl')
            assert len(rows) == result['queries']
            prefix = 0 if arm == 'native' else parent['first_alarm'] + 1
            monitor, step, used, counted_calls = V82Monitor(), 0, 0, 0
            for query, row in enumerate(rows):
                assert row['query'] == query and row['step'] == step
                is_prefix = query < prefix
                active = arm != 'native' and not is_prefix and used < 12
                chunk = 5 if active else 10
                candidates = 4 if active and arm != 'short' else 1
                assert row['active'] == active and row['chunk'] == chunk and row['candidates'] == candidates
                assert row['origin'] == ('native_prefix' if is_prefix else 'online')
                path = root / row['archive']
                assert digest(path) == row['archive_sha256']
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                assert payload['hb'].shape == (candidates, 8, 10, 11, 32)
                assert payload['hb'].dtype == np.float16
                assert np.isfinite(payload['hb']).all() and (payload['hb'] >= 0).all()
                assert (payload['hb'].sum(-1) > 0).all()
                assert payload['actions'].shape == (candidates, 10, 7)
                assert payload['actions'].dtype == np.float32 and np.isfinite(payload['actions']).all()
                for candidate in range(candidates):
                    noise = (native_noise[step // 10] if candidate == 0 and step % 10 == 0 else
                             rng(parent, step, candidate).standard_normal((10, 24)).astype(np.float32))
                    same(noise, payload['noises'][candidate], 'paired flow noise')
                    if not is_prefix:
                        key = (parent['name'], arm, query, candidate)
                        call = keyed_responses[key]
                        checked_calls.add(key)
                        assert call['noise_sha256'] == array_digest(noise)
                        assert call['hb_sha256'] == array_digest(payload['hb'][candidate])
                        assert call['actions_sha256'] == array_digest(payload['actions'][candidate])
                        counted_calls += 1
                scores = independent_scores(payload['hb']) if candidates == 4 else np.zeros(1)
                np.testing.assert_allclose(payload['scores'], scores, atol=1e-10, rtol=0)
                np.testing.assert_array_equal(payload['scores'], row['scores'])
                chosen = (int(np.flatnonzero(np.isclose(scores, scores.min(), atol=1e-12, rtol=0))[0])
                          if active and arm == 'route_short' else
                          int(rng(parent, step, 100).integers(4)) if active and arm == 'random_short' else 0)
                assert chosen == row['selected'] == int(payload['selected'])
                status = monitor.update(payload['hb'][chosen])
                assert status['v82_first'] == row['monitor']['v82_first']
                assert status['v82_alarm'] == row['monitor']['v82_alarm']
                same(payload['sim_before'], source_arrays['sim_states_before'][0] if step == 0 else
                     rollout['sim_states_after'][step - 1], 'physical observation alignment')
                executed = row['executed']
                assert 1 <= executed <= chunk
                assert executed == chunk or query == len(rows) - 1
                same(rollout['actions'][step:step + executed], payload['actions'][chosen, :executed], 'dispatched action block')
                assert row['success'] == bool(rollout['successes'][step + executed - 1])
                if arm == 'native' or is_prefix:
                    for key, source_key in (('image', 'images'), ('wrist_image', 'wrist_images'), ('state', 'states')):
                        same(payload[key], source_arrays[source_key][query], 'source observation')
                    same(payload['actions'][0], source_arrays['predicted_actions'][query], 'source action')
                    same(payload['hb'][0], hb_reference[query], 'source HB')
                    same(rollout['sim_states_after'][step:step + executed], source_arrays['sim_states_after'][step:step + executed], 'source physics')
                    same(rollout['rewards'][step:step + executed], source_arrays['rewards'][step:step + executed], 'source rewards')
                if arm != 'native' and query == prefix:
                    same(payload['actions'][0], source_arrays['predicted_actions'][query], 'branch native action')
                    same(payload['hb'][0], hb_reference[query], 'branch native routing')
                    for key, source_key in (('image', 'images'), ('wrist_image', 'wrist_images'), ('state', 'states')):
                        same(payload[key], source_arrays[source_key][query], 'branch source observation')
                    if arm == 'route_short':
                        paired = root / parent['name'] / 'random_short/queries' / ('q%03d.npz' % query)
                        with np.load(paired, allow_pickle=False) as other:
                            for key in ('noises', 'actions', 'hb', 'image', 'wrist_image', 'state', 'sim_before'):
                                same(payload[key], other[key], 'paired first candidate pool')
                step += executed
                if active:
                    used += 1
            assert counted_calls == result['actual_model_calls']
            assert result['deployed_model_calls'] == prefix + counted_calls
            assert step == result['action_steps'] and used == result['active_queries']
            assert monitor.first_v82_alarm == result['first_alarm'] == parent['first_alarm']
            exact_steps = step if arm == 'native' else prefix * 10
            assert environment['exact_source_steps'] == exact_steps
            assert environment['frame_sha256'][:exact_steps + 11] == manifest['source_render']['frame_sha256'][:exact_steps + 11]
            if arm == 'native':
                assert result['success'] == parent['source_success']
                for key in ('final_image', 'final_wrist_image', 'final_state', 'final_sim_state'):
                    same(rollout[key], source_arrays[key], 'native endpoint')
            audits.append(dict(parent=parent['name'], arm=arm, queries=len(rows), calls=counted_calls,
                               selected_routes_only_monitor=True, decisions_recomputed=True,
                               real_actions_exact=True, exact_prefix_steps=exact_steps,
                               actual_environment_steps=step, video_frames_verified=True,
                               paired_first_pool_checked=arm == 'route_short'))
    assert checked_calls == set(keyed_responses)
    assert len(result_rows) == 20
    failed = sum(not p['source_success'] for p in config['parents'])
    succeeded = len(config['parents']) - failed
    triggered_success = sum(p['source_success'] and p['first_alarm'] >= 0 for p in config['parents'])
    by_arm = {}
    for arm in config['arms']:
        values = [r for r in result_rows if r['arm'] == arm]
        rescued, harmed = sum(r['rescue'] for r in values), sum(r['harm'] for r in values)
        by_arm[arm] = dict(successes=sum(r['success'] for r in values), episodes=len(values),
                           rescued=rescued, failed_parent_count=failed, rescue_rate=rescued / failed,
                           rescue_wilson95=wilson(rescued, failed), harmed=harmed,
                           successful_parent_count=succeeded, triggered_successful_parent_count=triggered_success,
                           deployed_model_calls=sum(r['deployed_model_calls'] for r in values),
                           actual_model_calls=sum(r['actual_model_calls'] for r in values),
                           active_queries=sum(r['active_queries'] for r in values),
                           action_steps=sum(r['action_steps'] for r in values))
    latencies = np.asarray([r['resource']['seconds'] for r in responses])
    summary = dict(arms=by_arm, model_calls=len(responses),
                   actual_environment_steps=sum(r['actual_environment_steps'] for r in result_rows),
                   actual_settle_steps=sum(r['actual_settle_steps'] for r in result_rows),
                   actual_rollouts=sum(r['alias_of'] is None for r in result_rows),
                   non_independent_no_alarm_aliases=sum(r['alias_of'] is not None for r in result_rows),
                   call_latency_seconds=dict(mean=float(latencies.mean()), median=float(np.median(latencies)),
                                             p95=float(np.quantile(latencies, .95)), total=float(latencies.sum())),
                   model_peak_reserved_mib=max(r['resource']['peak_reserved_mib'] for r in responses),
                   elapsed_seconds=collection['elapsed_seconds'], sampling='purposeful developmental pilot; not held-out')
    write(root / 'audit.json', dict(passed=True, source_files=len(config['source_hashes']),
                                   candidate_calls_checked=len(checked_calls), branch_audits=audits,
                                   auditor_sha256=digest(Path(__file__))))
    write(root / 'summary.json', summary)
    columns = ['parent', 'arm', 'success', 'rescue', 'harm', 'action_steps', 'queries',
               'first_alarm', 'active_queries', 'actual_model_calls', 'deployed_model_calls', 'alias_of']
    with (root / 'results.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(result_rows)
    print(json.dumps(dict(audit_passed=True, summary=summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    audit(args.run.resolve())
