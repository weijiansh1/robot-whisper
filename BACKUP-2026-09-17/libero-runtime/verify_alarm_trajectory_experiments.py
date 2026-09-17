"""Independent arithmetic and provenance checks for the frozen alarm experiments."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from threadpoolctl import threadpool_limits

from alarm_trajectory_study import ROOT, PREVIOUS, METHODS, load_bank, physical_window
from score_v82_knn_reference import load_profile
from v82_knn_research import CURRENT, REPO, RAW, RUN_A, RUN_B, archive, load_available, standardized, digest, save_json

BASE = Path(__file__).resolve().parent
CONDITIONS = ('o0n0', 'o0n1', 'o1n0', 'o1n1')
HADAMARD = np.array([[-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]], float)/4
SOURCES = {}


def remember(path, expected=None):
    path = Path(path).resolve()
    value = digest(path)
    if expected is not None:
        assert value == expected, str(path)
    SOURCES[str(path)] = value
    return value


def crossings(values, threshold):
    return np.array([next((q for q, score in enumerate(row) if score > threshold), -1) for row in values])


def direct(values, bank, locations):
    i, q = np.asarray(locations).T
    prefixes = np.stack([values[r, t-5:t-2].ravel()/np.sqrt(3) for r, t in locations])
    distances = cdist(prefixes, bank['prefix'])
    ids = np.argsort(distances, axis=1)[:, :20]
    actual_delta = values[i, q]-values[i, q-3]
    ref_delta = bank['end'][ids]-bank['start'][ids]
    residual = actual_delta[:, None]-ref_delta
    means = ref_delta.mean(1)
    denominator = np.linalg.norm(actual_delta, axis=1)*np.linalg.norm(means, axis=1)
    cosine = np.divide((actual_delta*means).sum(1), denominator,
                       out=np.full(len(i), np.nan), where=denominator > 1e-12)
    return dict(displacement=np.linalg.norm(residual, axis=-1).mean(1),
                endpoint=np.linalg.norm(values[i, q, None]-bank['end'][ids], axis=-1).mean(1),
                prefix_distance=np.take_along_axis(distances, ids, axis=1).mean(1),
                rematched_endpoint=np.sort(cdist(values[i, q], bank['end']), axis=1)[:, :20].mean(1),
                reference_disagreement=np.linalg.norm(ref_delta-means[:, None], axis=-1).mean(1),
                signed_residual=residual.mean(1), direction_cosine=cosine), distances


def check_scores(frame, data, values, bank, thresholds):
    valid = (np.arange(values.shape[1])[None] < frame.length.to_numpy()[:, None])
    scored = valid & (np.arange(values.shape[1])[None] >= 12)
    for name in ('displacement', 'endpoint', 'prefix_distance', 'rematched_endpoint', 'reference_disagreement'):
        np.testing.assert_array_equal(np.isfinite(data[name]), scored)
    assert np.isnan(data['signed_residual'][~scored]).all()
    assert np.isfinite(data['signed_residual'][scored]).all()
    assert (data['neighbor_ids'][~scored] == -1).all()
    assert (data['unique_parents'][~scored] == -1).all()
    ids = data['neighbor_ids'][scored]
    assert ((ids >= 0) & (ids < len(bank['prefix']))).all()
    assert (np.diff(np.sort(ids, axis=1), axis=1) > 0).all()
    parents = np.sort(bank['global_rows'][ids], axis=1)
    np.testing.assert_array_equal(data['unique_parents'][scored], 1+(np.diff(parents, axis=1) != 0).sum(1))
    assert (data['rematched_endpoint'][scored] <= data['endpoint'][scored]+1e-6).all()
    for name in METHODS:
        np.testing.assert_array_equal(crossings(data[name], thresholds[name]), frame[name+'_first'])
    possible = np.argwhere(scored)
    rng = np.random.default_rng(9152026)
    selected = possible[rng.choice(len(possible), min(128, len(possible)), replace=False)]
    for _, part in frame.groupby(['task', 'failure'], sort=True):
        part = part[part.length > 12]
        if len(part):
            row = part.iloc[0]
            selected = np.vstack([selected, [row.name, max(12, int(row.displacement_first))]])
    selected = np.unique(selected, axis=0)
    exact, distances = direct(values, bank, selected)
    i, q = selected.T
    actual_ids = data['neighbor_ids'][i, q]
    np.testing.assert_allclose(np.sort(np.take_along_axis(distances, actual_ids, axis=1), axis=1),
                               np.sort(distances, axis=1)[:, :20], atol=1e-9, rtol=1e-9)
    for name, expected in exact.items():
        np.testing.assert_allclose(data[name][i, q], expected, atol=2e-6, rtol=2e-6)
    return dict(parents=len(frame), scored_queries=int(scored.sum()), direct_queries=len(selected))


def check_physics(episodes, directory):
    maximum_eef_error, states = 0., 0
    sensitivity = []
    for episode in episodes:
        path = directory/(episode['name']+'.npz')
        metadata = json.loads(path.with_suffix('.json').read_text())
        remember(path, metadata['output_sha256'])
        remember(path.with_suffix('.json'))
        remember(metadata['source_bddl'], metadata['source_bddl_sha256'])
        trace_path = Path(episode['source_artifact_dir'])/'episode-trace.npz'
        remember(trace_path, episode['source_trace_sha256'])
        with np.load(trace_path) as trace:
            physics = archive(path)
            boundaries = np.cumsum(trace['executed_lengths'])-1
            observed = np.concatenate([trace['sim_states_before'], trace['final_sim_state'][None]])
            np.testing.assert_array_equal(observed[1:], trace['sim_states_after'][boundaries])
            np.testing.assert_array_equal(physics['predicates'].all(-1), np.r_[False, trace['successes'][boundaries]])
            np.testing.assert_array_equal(physics['action_steps'], np.r_[trace['replan_start_steps'], boundaries[-1]+1])
            eef_error = float(np.abs(physics['eef'][:-1]-trace['states'][:, :3]).max())
            maximum_eef_error = max(maximum_eef_error, eef_error)
            assert len(physics['eef']) == len(trace['images'])+1 == metadata['states']
            states += len(physics['eef'])
            cached = dict(physics, eef=np.concatenate([trace['states'][:, :3], trace['final_state'][None, :3]]))
            for horizon in (3, 5):
                for q in range(len(trace['images'])):
                    original, alternative = [physical_window(data, q, horizon)['category'] for data in (physics, cached)]
                    sensitivity.append(dict(name=episode['name'], query=q, horizon=horizon,
                        boundary_category=original, cached_eef_category=alternative, changed=original != alternative))
    table = pd.DataFrame(sensitivity)
    table.to_csv(ROOT/(directory.name+'-eef-sensitivity.csv'), index=False)
    return dict(parents=len(episodes), physical_states=states, success_exact=True,
                boundary_states_exact=True, maximum_observation_eef_offset_m=maximum_eef_error,
                observation_eef_is_bitexact=maximum_eef_error == 0, sensitivity_windows=len(table),
                categories_changed_with_cached_eef=int(table.changed.sum()))


def check_functional():
    selection = json.loads((ROOT/'functional-selection.json').read_text())
    run = json.loads((ROOT/'functional/results.json').read_text())
    calls = pd.DataFrame(json.loads((ROOT/'functional/calls.json').read_text()))
    assert run['passed'] and len(calls) == run['model_forwards'] == 162
    assert calls.captured.sum() == 130
    assert len(selection['parents']) == 16 and len(selection['probes']) == 32
    assert len({p['name'] for p in selection['parents']}) == 16
    assert sum(p['phase_matched'] for p in selection['pairs']) == 5
    assert not selection['selection_uses_new_functional_data']
    remember(ROOT/'functional-selection.json', run['selection_sha256'])
    remember(ROOT/'PROTOCOL.md', selection['protocol_sha256'])
    energies = pd.read_csv(ROOT/'functional-energy.csv')
    energy_checks = native = corners = repeats = captured = 0
    maximum_error = 0.
    for probe in selection['probes']:
        q = probe['probe_query']
        assert q == probe['query']-(3 if probe['stage'] == 'before' else 0)
        directory = ROOT/'functional'/('probe-%02d' % probe['probe_id'])
        trace_path = Path(probe['source_dir'])/'episode-trace.npz'
        routes_path = CURRENT/probe['name']/'full-hb-routes.npz'
        remember(trace_path, probe['trace_sha256'])
        remember(routes_path, probe['source_hb_sha256'])
        remember(ROOT/'physical'/(probe['name']+'.npz'), probe['physics_sha256'])
        with np.load(trace_path) as trace, np.load(routes_path) as routes:
            snapshots = {name: archive(directory/(name+'.npz')) for name in CONDITIONS}
            for row in calls[calls.probe_id.eq(probe['probe_id'])].itertuples():
                path = directory/(row.condition+'.npz')
                remember(path)
                snap = archive(path)
                assert all(np.isfinite(value).all() for value in snap.values())
                assert snap['actions'].shape == (10, 7) and snap['hb'].shape == (8, 10, 11, 32)
                if row.captured:
                    captured += 1
                    np.testing.assert_array_equal(snap['hb_layers'], [2, 3, 4, 5, 12, 13, 14, 15])
                    assert (np.diff(np.sort(snap['ids'], axis=-1), axis=-1) > 0).all()
                    assert ((snap['ids'] >= 0) & (snap['ids'] < 32)).all()
                    assert (snap['weights'] > 0).all()
                    reconstructed = (snap['expert_output']*snap['weights'][..., None]).sum(-2)
                    error = np.linalg.norm(reconstructed-snap['routed'], axis=-1)/np.maximum(
                        np.linalg.norm(snap['routed'], axis=-1), 1e-12)
                    assert error.max() < 1e-5
                    maximum_error = max(maximum_error, float(error.max()))
                    np.testing.assert_array_equal(snap['routed']+snap['shared'], snap['total'])
                if row.condition in ('o0n0', 'o1n1'):
                    t = q-(row.condition == 'o0n0')
                    np.testing.assert_array_equal(snap['actions'], trace['predicted_actions'][t])
                    np.testing.assert_array_equal(snap['hb'], routes['hb_router_probs'][t])
                    np.testing.assert_array_equal(snap['ids'], routes['hb_expert_ids'][t, :, -1])
                    corners += 1
            control = archive(directory/'native_current.npz')
            for key, value in control.items():
                np.testing.assert_array_equal(value, snapshots['o1n1'][key])
            native += 1
            if probe['probe_id'] in (0, 31):
                repeat = archive(directory/'repeat_current.npz')
                assert repeat.keys() == snapshots['o1n1'].keys()
                for key, value in repeat.items():
                    np.testing.assert_array_equal(value, snapshots['o1n1'][key])
                repeats += 1
        for field in ('input', 'routed', 'shared', 'total', 'router'):
            values = []
            for snap in snapshots.values():
                if field == 'router':
                    p = snap['hb'][:, -1].astype(float)
                    values.append(np.sqrt(p/p.sum(-1, keepdims=True))/np.sqrt(2))
                else:
                    values.append(snap[field].astype(float))
            energy = (np.tensordot(HADAMARD, np.stack(values), axes=(1, 0))**2).sum(-1)
            subset = energies[energies.probe_id.eq(probe['probe_id']) & energies.field.eq(field)]
            for row in subset.itertuples():
                depth, kind = row.scope.split('_')
                layers = slice(0, 4) if depth == 'front' else slice(4, 8)
                tokens = slice(0, 1) if kind == 'state' else slice(1, 11)
                sums = energy[:, layers, tokens].sum((1, 2))
                expected = sums/sums.sum() if sums.sum() else np.full(3, np.nan)
                np.testing.assert_allclose([row.observation_fraction, row.noise_fraction, row.interaction_fraction],
                                           expected, rtol=2e-6, atol=1e-8)
                energy_checks += 1
    assert (captured, native, corners, repeats, energy_checks) == (130, 32, 64, 2, 640)
    parent = pd.read_csv(ROOT/'functional-parent-responses.csv')
    paired = pd.read_csv(ROOT/'functional-paired-changes.csv')
    for row in paired.itertuples():
        part = parent[parent.pair_id.eq(row.pair_id) & parent.scope.eq(row.scope)]
        f, s = [part[part.success.eq(success)].set_index('stage') for success in (False, True)]
        expected = (f.loc['anchor', row.measurement]-f.loc['before', row.measurement]
                    -s.loc['anchor', row.measurement]+s.loc['before', row.measurement])
        np.testing.assert_allclose(row.difference_of_changes, expected, atol=1e-12)
    for path in [*ROOT.glob('functional-*.csv'), ROOT/'functional-results.json',
                 ROOT/'functional/results.json', ROOT/'functional/calls.json', ROOT/'functional/checks.json']:
        remember(path)
    return dict(parents=16, anchors=32, forwards=162, captured=captured, native_controls=native,
                natural_corners=corners, repeat_controls=repeats, independent_energy_rows=energy_checks,
                maximum_routed_relative_reconstruction_error=maximum_error, paired_changes=len(paired))


def check_ranking(frame, scores, geometry):
    ranking = pd.read_csv(ROOT/'same-query-ranking.csv')
    for row in ranking.itertuples():
        part = frame[frame.task.eq(row.task) & (frame.length > row.query)]
        value = geometry['score'] if row.method == 'instantaneous' else scores[row.method]
        f = value[part[part.failure].index, row.query]
        s = value[part[~part.failure].index, row.query]
        f, s = f[np.isfinite(f)], s[np.isfinite(s)]
        np.testing.assert_allclose(row.auc, mannwhitneyu(f, s).statistic/(len(f)*len(s)), atol=1e-14)
    pairs = pd.read_csv(ROOT/'distance-matched-pairs.csv')
    for (_, method), part in pairs.groupby(['query', 'method']):
        assert not part.success_row.duplicated().any()
        assert not part.failure_row.duplicated().any()
    for row in pairs.itertuples():
        f, s = frame.iloc[row.failure_row], frame.iloc[row.success_row]
        assert f.failure and not s.failure and f.task == s.task == row.task
        assert min(f.length, s.length) > row.query
        gap = geometry['score'][row.failure_row, row.query]-geometry['score'][row.success_row, row.query]
        assert abs(gap) <= .15
        assert abs(int(geometry['unique_neighbor_parents'][row.failure_row, row.query])
                   -int(geometry['unique_neighbor_parents'][row.success_row, row.query])) <= 2
        delta = scores[row.method][row.failure_row, row.query]-scores[row.method][row.success_row, row.query]
        np.testing.assert_allclose([row.instantaneous_gap, row.difference], [gap, delta], atol=1e-12)
        assert row.failure_higher == float(delta > 0)+.5*(delta == 0)
    summary = pd.read_csv(ROOT/'distance-matched-summary.csv')
    for row in summary.itertuples():
        part = pairs[pairs['query'].eq(row.query) & pairs.method.eq(row.method)]
        means = part.groupby('task').failure_higher.mean().to_numpy()
        rng = np.random.default_rng(2026091502)
        sampled = means[rng.integers(0, len(means), (2000, len(means)))].mean(-1)
        np.testing.assert_allclose([row.task_mean_failure_higher, row.bootstrap_low, row.bootstrap_high],
                                   [means.mean(), *np.quantile(sampled, [.025, .975])], atol=1e-14)
    for name in ('same-query-ranking.csv', 'same-query-ranking-summary.csv', 'distance-matched-pairs.csv',
                 'distance-matched-summary.csv', 'reference-ambiguity-at-alarm.csv'):
        remember(ROOT/name)
    return dict(independent_auc_rows=len(ranking), matched_comparisons=len(pairs), bootstrap_rows=len(summary))


def existing():
    checks = {}
    counts = {}
    for name in ('moe-joint-patterns-20260915', 'moe-joint-expanded-20260915',
                 'moe-input-response-20260915', 'v82-knn-mechanism-20260915'):
        directory = BASE/'samples'/name
        seal = json.loads((directory/'artifact-hashes.json').read_text())
        for relative, expected in seal.items():
            remember(directory/relative, expected)
        counts[name] = len(seal)
    checks['prior_artifacts_unchanged'] = counts
    print('Prior seals unchanged', flush=True)
    bank, params = load_bank()
    profile, _ = load_profile()
    frame, raw = load_available()
    ref, cal = profile['reference_available_rows'], profile['calibration_available_rows']
    assert frame.iloc[ref].run_id.eq(RUN_A).all() and frame.iloc[cal].run_id.eq(RUN_A).all()
    assert set(zip(frame.iloc[ref].task, frame.iloc[ref].init_state_id)).isdisjoint(
        set(zip(frame.iloc[cal].task, frame.iloc[cal].init_state_id)))
    x, _ = standardized({k: v[ref] for k, v in raw.items()}, profile)
    candidates = []
    for i, row in enumerate(frame.iloc[ref].itertuples()):
        if not row.failure:
            valid = np.array([q for q in range(9, row.length-3) if np.isfinite(x[i, q-2:q+4]).all()])
            if len(valid):
                selected = valid[np.linspace(0, len(valid)-1, min(8, len(valid))).astype(int)]
                candidates.extend((i, q) for q in selected)
    candidates = np.asarray(candidates)
    selected = candidates[np.random.default_rng(20260908).choice(len(candidates), 4096, replace=False)]
    i, q = selected.T
    expected = dict(available_rows=ref[i], global_rows=frame.iloc[ref[i]].global_row.to_numpy(), queries=q,
                    prefix=np.stack([x[r, t-2:t+1].ravel()/np.sqrt(3) for r, t in selected]),
                    start=x[i, q], end=x[i, q+3], delta=x[i, q+3]-x[i, q])
    for key, value in expected.items():
        np.testing.assert_array_equal(bank[key], value)
    calibration = archive(ROOT/'profile/calibration.npz')
    cal_x, _ = standardized({k: v[cal] for k, v in raw.items()}, profile)
    np.testing.assert_array_equal(calibration['available_rows'], cal)
    np.testing.assert_array_equal(calibration['failure'], frame.iloc[cal].failure)
    groups = (frame.iloc[cal].task+'|'+frame.iloc[cal].init_state_id.astype(str)).to_numpy(str)
    good = ~calibration['failure']
    names = sorted(set(groups[good]))
    np.testing.assert_array_equal(calibration['groups'], groups)
    np.testing.assert_array_equal(calibration['group_names'], names)
    rank = int(np.ceil((len(names)+1)*.95))
    assert (len(names), rank) == (369, 352)
    locations = np.argwhere(np.isfinite(calibration['displacement']))[::max(1, int(np.isfinite(calibration['displacement']).sum())//96)]
    exact, _ = direct(cal_x, bank, locations)
    i, q = locations.T
    for name in METHODS:
        values = np.where(np.isfinite(calibration[name]), calibration[name], -np.inf)
        peaks = np.array([values[good & (groups == group)].max() for group in names])
        np.testing.assert_array_equal(calibration[name+'_group_peaks'], peaks)
        assert float(np.sort(peaks)[rank-1]) == params['thresholds'][name]
        np.testing.assert_allclose(calibration[name][i, q], exact[name], atol=2e-6, rtol=2e-6)
    checks['A_reference'] = dict(bank_points=4096, successful_parents=len(np.unique(bank['global_rows'])),
        reference_parents=len(ref), calibration_parents=len(cal), calibration_groups=len(names), rank=rank,
        thresholds=params['thresholds'], direct_calibration_queries=len(locations), B_used_for_fitting=False)
    print('A-only reference and calibration verified', flush=True)
    for cohort in ('B', 'current'):
        decisions = pd.read_csv(ROOT/(cohort+'-decisions.csv'))
        if cohort == 'B':
            chosen = frame.run_id.eq(RUN_B).to_numpy()
            inputs = {k: v[chosen] for k, v in raw.items()}
            np.testing.assert_array_equal(decisions.global_row, frame[chosen].global_row)
        else:
            inputs = archive(PREVIOUS/'current-routing-inputs.npz')
        values, _ = standardized(inputs, profile)
        scores = archive(ROOT/(cohort+'-scores.npz'))
        checks[cohort+'_scores'] = check_scores(decisions, scores, values, bank, params['thresholds'])
        remember(ROOT/(cohort+'-scores.npz'))
        remember(ROOT/(cohort+'-decisions.csv'))
        if cohort == 'B':
            checks['B_comparisons'] = check_ranking(decisions, scores, archive(PREVIOUS/'available-B-geometry.npz'))
    print('B/current scores and conditional comparisons verified', flush=True)
    checks['current_physics'] = check_physics(json.loads((CURRENT/'summary.json').read_text())['episodes'], ROOT/'physical')
    kinematics = []
    for path in sorted((ROOT/'kinematic-audit').glob('*.json')):
        record = json.loads(path.read_text())
        data = archive(path.with_suffix('.npz'))
        physical = archive(ROOT/'physical'/(record['name']+'.npz'))
        np.testing.assert_array_equal(data['boundary_eef'], physical['eef'][:-1])
        remember(path.with_suffix('.npz'), record['output_sha256'])
        remember(path)
        kinematics.append(record)
    checks['cached_observation_diagnostic'] = dict(controls=kinematics,
        exact_observation_reconstruction_established=all(r['cached_observation_recovered'] for r in kinematics),
        note='Boundary geometry and cached policy observations differ. A single reversed 2 ms position integration does not fully recover the cache; no exact lag is claimed.')
    checks['functional'] = check_functional()
    for path in [*ROOT.joinpath('profile').glob('*'), ROOT/'PROTOCOL.md', ROOT/'fresh-plans.json']:
        remember(path)
    for cohort in ('development_main', 'external_8b'):
        remember(RAW/(cohort+'_route_features.npz'))
        remember(RAW/(cohort+'_extraction_audit.json'))
    save_json(ROOT/'verification-existing.json', dict(passed=True, checks=checks,
              verified_at_utc=datetime.now(timezone.utc).isoformat(), verified_input_hashes=SOURCES))
    print(json.dumps(checks, indent=2), flush=True)


def fresh():
    completion = json.loads((ROOT/'fresh-complete.json').read_text())
    finished = json.loads((ROOT/'isolated-server-finished.json').read_text())
    manifest = json.loads((ROOT/'fresh-manifest.json').read_text())
    episodes = manifest['episodes']
    assert completion['all_completed'] and completion['episodes'] == len(episodes) == 20
    assert finished['stopped'] and not finished['shared_service_used']
    remember(ROOT/'fresh-plans.json', completion['plans_sha256'])
    assert manifest['plans_sha256'] == completion['plans_sha256']
    plans = json.loads((ROOT/'fresh-plans.json').read_text())
    remember(ROOT/'profile/trajectory-bank.npz', plans['profile_sha256'])
    remember(ROOT/'profile/parameters.json', plans['parameters_sha256'])
    previous_identities = set()
    for variant in ('plus', 'pro'):
        for job in json.loads((BASE/(variant+'-sample-plan.json')).read_text())['jobs']:
            previous_identities.add((job['task_name'], job['init_state_id'], job['flow_noise_seed']))
        part = [e for e in episodes if e['benchmark'] == variant]
        assert sorted(e['base_task_id'] for e in part) == list(range(10))
        for e, job in zip(part, plans['plans'][variant]['jobs']):
            assert (e['task_name'], e['init_state_id'], e['flow_noise_seed']) not in previous_identities
            assert e['init_state_id'] == job['init_state_id'] == 11
            assert e['task_name'] == job['task_name'] and e['flow_noise_seed'] == job['flow_noise_seed']
    frame = pd.read_csv(ROOT/'fresh-decisions.csv')
    raw, scores = archive(ROOT/'fresh-routing-inputs.npz'), archive(ROOT/'fresh-scores.npz')
    matches = pd.read_csv(ROOT/'fresh-native-capture-matches.csv')
    assert not matches.wire.duplicated().any() and not matches.duplicated(['name', 'query']).any()
    load = json.loads((ROOT/'isolated-model-load.json').read_text())['metadata']
    sys.path.insert(0, str(REPO/'moe-trap-control'))
    from v82_closed_loop import V82Monitor
    total_actions = total_queries = 0
    for i, episode in enumerate(episodes):
        assert frame.iloc[i]['name'] == episode['name']
        for key in ('checkpoint_sha256', 'normalization_stats_sha256'):
            assert episode['policy_identity'][key] == load[key]
        with np.load(Path(episode['source_artifact_dir'])/'episode-trace.npz') as trace:
            route = archive(ROOT/'fresh-routes'/(episode['name']+'.npz'))
            assert trace['executed_lengths'].sum() == episode['action_steps']
            assert trace['successes'][-1] == episode['source_result']['success']
            if not episode['source_result']['success']:
                assert episode['action_steps'] == 520
            monitor = V82Monitor()
            for q, hb in enumerate(route['hb']):
                row = matches[matches.name.eq(episode['name']) & matches['query'].eq(q)].iloc[0]
                wire = archive(ROOT/'fresh-wire'/row.wire)
                for field, value in (('noise', trace['flow_noises'][q]), ('image', trace['images'][q]), ('state', trace['states'][q])):
                    assert str(wire[field+'_sha256']) == hashlib.sha256(value.tobytes()).hexdigest()
                np.testing.assert_array_equal(wire['actions'], trace['predicted_actions'][q])
                np.testing.assert_array_equal(wire['hb'], hb)
                np.testing.assert_array_equal(wire['ids'], route['ids'][q])
                status = monitor.update(hb)
                assert raw['valid'][i, q]
                for name, source in (('mobility', 'layer_mobility'), ('acceleration', 'route_acceleration'), ('periodicity', 'lag_periodicity')):
                    np.testing.assert_array_equal(raw[name][i, q], np.asarray(status[source], np.float32))
                np.testing.assert_array_equal(scores['v82_scores'][i, q],
                    [status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']])
            assert monitor.first_v82_alarm == frame.iloc[i].v82_frozen
            assert len(route['hb']) == len(trace['flow_noises']) == episode['queries'] == frame.iloc[i].length
            total_queries += episode['queries']
            total_actions += episode['action_steps']
    assert total_queries == len(matches) == completion['model_forwards'] == finished['fresh_model_forwards']
    assert len(list((ROOT/'fresh-wire').glob('call-*.npz'))) == total_queries
    assert total_actions == completion['environment_actions']
    bank, params = load_bank()
    profile, previous_params = load_profile()
    values, _ = standardized(raw, profile)
    check = check_scores(frame, scores, values, bank, params['thresholds'])
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    np.testing.assert_array_equal(raw['valid'], valid)
    np.testing.assert_array_equal(np.isfinite(scores['instantaneous']), valid & (np.arange(52)[None] >= 7))
    locations = np.argwhere(np.isfinite(scores['instantaneous']))[::7]
    i, q = locations.T
    expected = np.sort(cdist(values[i, q], profile['success_dynamic']), axis=1)[:, :20].mean(1)
    np.testing.assert_allclose(expected, scores['instantaneous'][i, q], rtol=2e-6, atol=1e-6)
    np.testing.assert_array_equal(crossings(scores['instantaneous'], previous_params['threshold']), frame.knn_available_first)
    check.update(actions=total_actions, native_calls_exact=total_queries, all_episodes_completed=True,
                 new_task_init_noise_identities=True, model_stopped=True,
                 direct_instantaneous_queries=len(locations), physical=check_physics(episodes, ROOT/'fresh-physical'))
    return check


def aggregate():
    metrics = pd.read_csv(ROOT/'alarm-metrics.csv')
    _, params = load_profile()
    scopes = {}
    for cohort in ('B', 'current', 'fresh'):
        frame = pd.read_csv(ROOT/(cohort+'-decisions.csv'))
        if cohort == 'fresh':
            instant = archive(ROOT/'fresh-scores.npz')['instantaneous']
        else:
            instant = archive(PREVIOUS/('available-B-geometry.npz' if cohort == 'B' else 'current-geometry.npz'))['score']
        instant[:, :12] = np.nan
        methods = dict(v82=frame.v82_frozen.to_numpy(), instantaneous=frame.knn_available_first.to_numpy(),
                       instantaneous_from_q12=crossings(instant, params['threshold']))
        methods.update({name: frame[name+'_first'].to_numpy() for name in METHODS})
        scopes[cohort] = frame, methods
        if cohort != 'B':
            for variant, part in frame.groupby('benchmark'):
                scopes[cohort+'_'+variant] = part, {key: value[part.index] for key, value in methods.items()}
    for row in metrics.itertuples():
        frame, methods = scopes[row.cohort]
        limit = (frame.horizon_actions.to_numpy()*float(row.cutoff[6:])/1000 if row.cutoff.startswith('budget')
                 else int(row.cutoff[1:]) if row.cutoff.startswith('q') else 10000)
        y, first = frame.failure.to_numpy(), methods[row.method]
        hit = (first >= 0) & (first < frame.length.to_numpy()) & (first <= limit)
        tp, fp = int((hit & y).sum()), int((hit & ~y).sum())
        assert (tp, fp, int(y.sum()), int((~y).sum())) == (row.tp, row.fp, row.failures, row.successes)
        np.testing.assert_allclose([row.recall, row.fpr, row.precision],
            [tp/y.sum() if y.any() else np.nan, fp/(~y).sum() if (~y).any() else np.nan,
             tp/(tp+fp) if tp+fp else np.nan], atol=1e-14)
    timing = pd.read_csv(ROOT/'alarm-timing.csv')
    for row in timing.itertuples():
        frame, methods = scopes[row.cohort]
        first = methods[row.method]
        eligible = frame.failure.to_numpy() == row.failure
        hit = eligible & (first >= 0) & (first < frame.length.to_numpy())
        assert (row.parents, row.flagged) == (int(eligible.sum()), int(hit.sum()))
        if hit.any():
            fraction = first[hit]*10/frame.horizon_actions.to_numpy()[hit]
            np.testing.assert_allclose([row.median_query, row.median_budget_fraction,
                row.q25_budget_fraction, row.q75_budget_fraction, row.median_remaining_budget_actions],
                [np.median(first[hit]), np.median(fraction), *np.quantile(fraction, [.25, .75]),
                 np.median(frame.horizon_actions.to_numpy()[hit]-first[hit]*10)], atol=1e-12)
    windows = pd.read_csv(ROOT/'physical-query-windows.csv')
    events = pd.read_csv(ROOT/'external-alarm-windows.csv')
    expected_windows = 2*(scopes['current'][0].length.sum()+scopes['fresh'][0].length.sum())
    assert len(windows) == expected_windows
    cache = {}
    sensitivity = pd.concat([pd.read_csv(ROOT/(directory+'-eef-sensitivity.csv')).assign(cohort=cohort)
                            for cohort, directory in (('current', 'physical'), ('fresh', 'fresh-physical'))])
    for row in events.itertuples():
        frame, methods = scopes[row.cohort]
        i = frame[frame.name.eq(row.name)].index[0]
        assert row.query == methods[row.method][i]
        key = row.cohort, row.name
        if key not in cache:
            cache[key] = archive(ROOT/('physical' if row.cohort == 'current' else 'fresh-physical')/(row.name+'.npz'))
        expected = physical_window(cache[key], row.query, row.horizon)
        for name, value in expected.items():
            observed = getattr(row, name)
            if isinstance(value, float):
                np.testing.assert_allclose(observed, value, atol=1e-12)
            else:
                assert observed == value
        assert row.end_query <= frame.iloc[i].length
    joined = events.merge(sensitivity, on=['cohort', 'name', 'query', 'horizon'], validate='many_to_one')
    assert len(joined) == len(events)
    joined[joined.changed].to_csv(ROOT/'alarm-eef-sensitive-windows.csv', index=False)
    return dict(metric_rows=len(metrics), timing_rows=len(timing), physical_query_windows=len(windows),
                alarm_windows=len(events), alarm_windows_changed_with_cached_eef=int(joined.changed.sum()))


def reference_context():
    bank, _ = load_bank()
    available = pd.read_csv(PREVIOUS/'available-index.csv')
    bank_tasks = available.iloc[bank['available_rows']].task.to_numpy()
    bank_suites = available.iloc[bank['available_rows']].suite.to_numpy()
    plans = json.loads((ROOT/'fresh-plans.json').read_text())
    canonical = {job['base_task_id']: 'libero_long/'+job['base_task_name'] for job in plans['plans']['plus']['jobs']}
    table = pd.read_csv(ROOT/'reference-context-parents.csv')
    coverage = pd.read_csv(ROOT/'reference-coverage-metrics.csv')
    for cohort in ('B', 'current', 'fresh'):
        frame = pd.read_csv(ROOT/(cohort+'-decisions.csv'))
        data = archive(ROOT/(cohort+'-scores.npz'))
        tasks = frame.task.to_numpy() if cohort == 'B' else frame.task_id.map(canonical).to_numpy()
        supported = np.isin(tasks, bank_tasks)
        for row in table[table.cohort.eq(cohort)].itertuples():
            parent = frame.iloc[row.parent_row]
            assert row.task == tasks[row.parent_row] and row.failure == parent.failure
            assert row.task_present_in_bank == supported[row.parent_row]
            ids = data['neighbor_ids'][row.parent_row]
            if row.scope == 'all_scored_parent_mean':
                ids = ids[ids[:, 0] >= 0]
            else:
                method = row.scope.removesuffix('_alarm')
                assert row.query == parent[method+'_first']
                ids = ids[row.query]
            np.testing.assert_allclose([row.same_task_neighbor_fraction, row.same_suite_neighbor_fraction],
                                       [(bank_tasks[ids] == row.task).mean(), (bank_suites[ids] == parent.suite).mean()], atol=1e-14)
        for row in coverage[coverage.cohort.eq(cohort)].itertuples():
            mask = supported if row.reference_task_coverage == 'supported' else ~supported if row.reference_task_coverage == 'unsupported' else np.ones(len(frame), bool)
            field = {'v82': 'v82_frozen', 'instantaneous': 'knn_available_first'}.get(row.method, row.method+'_first')
            first, outcome = frame[field].to_numpy()[mask], frame.failure.to_numpy()[mask]
            hit = (first >= 0) & (first < frame.length.to_numpy()[mask])
            tp, fp = int((hit & outcome).sum()), int((hit & ~outcome).sum())
            assert (row.episodes, row.failures, row.successes, row.tp, row.fp) == (
                int(mask.sum()), int(outcome.sum()), int((~outcome).sum()), tp, fp)
    summary = pd.read_csv(ROOT/'reference-context-summary.csv')
    for row in summary.itertuples():
        part = table[table.cohort.eq(row.cohort) & table.scope.eq(row.scope) & table.failure.eq(row.failure)]
        assert row.parents == len(part) and row.task_supported_parents == part.task_present_in_bank.sum()
        np.testing.assert_allclose([row.median_same_task_fraction, row.median_same_suite_fraction],
                                   [part.same_task_neighbor_fraction.median(), part.same_suite_neighbor_fraction.median()], atol=1e-14)
    return dict(posthoc_descriptive=True, continuation_bank_tasks=len(set(bank_tasks)),
                context_parent_rows=len(table), context_summary_rows=len(summary), coverage_metric_rows=len(coverage))


def final():
    prior = json.loads((ROOT/'verification-existing.json').read_text())
    assert prior['passed']
    for path, expected in prior['verified_input_hashes'].items():
        remember(path, expected)
    checks = prior['checks']
    checks['fresh'] = fresh()
    checks['aggregate'] = aggregate()
    checks['reference_context'] = reference_context()
    test = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_alarm_trajectories'],
                          cwd=BASE, text=True, capture_output=True, timeout=120)
    log = test.stdout+test.stderr
    (ROOT/'tests.txt').write_text(log)
    assert test.returncode == 0, log
    checks['unit_tests'] = int(re.search(r'Ran (\d+) tests?', log).group(1))
    figures = {}
    for path in ROOT.glob('*.png'):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        nonwhite = float((pixels < 245).any(-1).mean())
        assert nonwhite > .015 and pixels.std() > 10 and path.with_suffix('.pdf').stat().st_size > 1000
        figures[path.name] = dict(width=pixels.shape[1], height=pixels.shape[0], nonwhite_fraction=nonwhite)
    assert len(figures) == 4 and (ROOT/'REPORT.zh.md').exists()
    assert 'PENDING_' not in (ROOT/'REPORT.zh.md').read_text()
    checks['figures'] = figures
    implementation = [BASE/name for name in ('alarm_trajectory_study.py', 'restore_alarm_physics.py',
        'run_alarm_trajectory_analysis.py', 'select_alarm_function_probes.py', 'run_alarm_model_experiments.py',
        'run_fresh_alarm_validation.py', 'score_fresh_alarm_validation.py', 'summarize_alarm_trajectories.py',
        'analyze_alarm_function_probes.py', 'plot_alarm_trajectory_experiments.py', 'test_alarm_trajectories.py',
        'verify_alarm_trajectory_experiments.py', 'audit_alarm_kinematics.py', 'analyze_alarm_reference_context.py')]
    for path in (BASE/'moe_final_response_recorder.py', BASE/'probe_moe_input_response.py',
                 REPO/'moe-trap-control/v82_closed_loop.py',
                 Path('/data/coding/moe-control-experiments/gate_runtime.py'),
                 Path('/data/coding/moe-control-experiments/gate_capture.py')):
        remember(path)
    save_json(ROOT/'source-hashes.json', SOURCES)
    save_json(ROOT/'implementation-hashes.json', {str(path): digest(path) for path in implementation})
    checks['gpu_after_completion'] = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader'], text=True, capture_output=True, check=True).stdout.strip()
    save_json(ROOT/'verification.json', dict(passed=True, verified_at_utc=datetime.now(timezone.utc).isoformat(), checks=checks))
    save_json(ROOT/'artifact-hashes.json', {str(path.relative_to(ROOT)): digest(path) for path in sorted(ROOT.rglob('*'))
        if path.is_file() and path.name != 'artifact-hashes.json'})
    print(json.dumps(checks, indent=2), flush=True)


def presentation():
    result = json.loads((ROOT/'verification.json').read_text())
    assert result['passed']
    seal = json.loads((ROOT/'artifact-hashes.json').read_text())
    figures = result['checks']['figures']
    allowed = {'REPORT.zh.md', *(name for image in figures for name in (image, str(Path(image).with_suffix('.pdf'))))}
    for relative, expected in seal.items():
        if relative not in allowed:
            assert digest(ROOT/relative) == expected, relative
    assert 'PENDING_' not in (ROOT/'REPORT.zh.md').read_text()
    for name in figures:
        with Image.open(ROOT/name) as image:
            pixels = np.asarray(image.convert('RGB'))
        nonwhite = float((pixels < 245).any(-1).mean())
        assert nonwhite > .015 and pixels.std() > 10
        assert (ROOT/Path(name).with_suffix('.pdf')).stat().st_size > 1000
        figures[name] = dict(width=pixels.shape[1], height=pixels.shape[0], nonwhite_fraction=nonwhite)
    implementation = json.loads((ROOT/'implementation-hashes.json').read_text())
    for path, expected in implementation.items():
        if Path(path).name not in ('plot_alarm_trajectory_experiments.py', 'verify_alarm_trajectory_experiments.py'):
            assert digest(path) == expected
    save_json(ROOT/'implementation-hashes.json', {path: digest(path) for path in implementation})
    result['presentation_checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    result['presentation_only_update'] = dict(numerical_artifacts_unchanged=True, figures_visually_inspected=True)
    save_json(ROOT/'verification.json', result)
    save_json(ROOT/'artifact-hashes.json', {str(path.relative_to(ROOT)): digest(path) for path in sorted(ROOT.rglob('*'))
        if path.is_file() and path.name != 'artifact-hashes.json'})
    print('All numerical artifacts unchanged; report, figures and implementation hashes resealed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['existing', 'final', 'presentation'])
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        {'existing': existing, 'final': final, 'presentation': presentation}[args.stage]()
