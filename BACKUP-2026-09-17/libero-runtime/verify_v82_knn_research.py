"""Independent arithmetic, provenance and replay checks for the kNN study."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from threadpoolctl import threadpool_limits

from v82_knn_research import (ROOT, REPO, SAFE, FROZEN, RAW, CURRENT, RUN_A, RUN_B,
                              GROUPS, archive, load_frozen, load_available,
                              standardized, dynamics, digest, save_json)
from score_v82_knn_reference import load_profile


def crossings(scores, threshold):
    return np.array([next((q for q, value in enumerate(row) if value > threshold), -1)
                     for row in scores])


def decisions(v, k):
    return dict(v82=np.asarray(v), knn=np.asarray(k),
                OR=np.array([min(a, b) if min(a, b) >= 0 else max(a, b) for a, b in zip(v, k)]),
                AND_latched=np.array([max(a, b) if min(a, b) >= 0 else -1 for a, b in zip(v, k)]))


def check_metrics(path, scopes):
    table = pd.read_csv(path)
    for record in table.itertuples():
        frame, alarms = scopes[record.cohort]
        cutoff = record.cutoff
        limit = (np.floor(frame.horizon_actions.to_numpy()*float(cutoff[7:])/10)
                 if cutoff.startswith('budget_') else int(cutoff[1:]) if cutoff.startswith('q') else 10000)
        hit = (alarms[record.method] >= 0) & (alarms[record.method] <= limit)
        y = frame.failure.to_numpy(bool)
        tp, fp = int((hit & y).sum()), int((hit & ~y).sum())
        assert (tp, fp, int(y.sum()), int((~y).sum())) == (
            record.tp, record.fp, record.failures, record.successes)
        for actual, expected in ((tp/y.sum() if y.any() else np.nan, record.recall),
                                 (fp/(~y).sum() if (~y).any() else np.nan, record.fpr),
                                 (tp/(tp+fp) if tp+fp else np.nan, record.precision)):
            np.testing.assert_allclose(actual, expected, atol=1e-14)
    return len(table)


def check_pairs(path, frame, methods):
    pairs = pd.read_csv(path)
    assert not pairs.duplicated(['failure_row', 'success_row']).any()
    expected_pairs = sum(int(part.failure.sum())*int((~part.failure).sum())
                         for _, part in frame.groupby(['task', 'init_state_id']))
    assert len(pairs) == expected_pairs
    for pair in pairs.itertuples():
        f, s = frame.loc[pair.failure_row], frame.loc[pair.success_row]
        assert f.failure and not s.failure and f.task == s.task == pair.task
        assert f.init_state_id == s.init_state_id == pair.init_state_id
        assert pair.queries == min(f.length, s.length)
        for name, first in methods.items():
            assert getattr(pair, name+'_failure_hit') == (0 <= first[pair.failure_row] < pair.queries)
            assert getattr(pair, name+'_success_hit') == (0 <= first[pair.success_row] < pair.queries)
    return len(pairs)


def direct_distances(x, bank):
    return np.sort(cdist(x, bank), axis=-1)[:, :20]


def check_geometry(frame, data, values, profile, threshold):
    valid = np.arange(52)[None] < frame.length.to_numpy()[:, None]
    scored = valid & (np.arange(52)[None] >= 7)
    np.testing.assert_array_equal(data['valid'], valid)
    np.testing.assert_array_equal(np.isfinite(data['score']), scored)
    np.testing.assert_array_equal(np.isfinite(values).all(-1), scored)
    np.testing.assert_array_equal(data['x'], values.astype(np.float32))
    np.testing.assert_array_equal(crossings(data['score'], threshold), frame.knn_available_first)
    for key in ('components', 'signed_residual'):
        assert np.isnan(data[key][~scored]).all() and np.isfinite(data[key][scored]).all()
    ids = data['neighbor_ids'][scored]
    assert (data['neighbor_ids'][~scored] == -1).all()
    assert (ids >= 0).all() and (ids < len(profile['success_dynamic'])).all()
    assert (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all()
    parents = np.sort(profile['success_global_rows'][ids], axis=-1)
    np.testing.assert_array_equal(data['unique_neighbor_parents'][scored],
                                  1 + (np.diff(parents, axis=-1) != 0).sum(-1))
    np.testing.assert_allclose(data['components'][scored].sum(-1), data['score'][scored], rtol=3e-7, atol=1e-6)
    candidates = np.argwhere(scored)
    rng = np.random.default_rng(9015)
    selected = candidates[rng.choice(len(candidates), min(160, len(candidates)), replace=False)]
    for _, part in frame.groupby(['failure', 'task'], sort=True):
        part = part[part.length > 7]
        if part.empty:
            continue
        row = part.iloc[0]
        selected = np.vstack([selected, [row.name, max(7, int(row.knn_available_first))]])
    selected = np.unique(selected, axis=0)
    i, q = selected.T
    x, neighbor = values[i, q], data['neighbor_ids'][i, q]
    bank = profile['success_dynamic']
    exact = direct_distances(x, bank)
    delta = x[:, None] - bank[neighbor]
    distances = np.linalg.norm(delta, axis=-1)
    np.testing.assert_allclose(np.sort(distances, axis=-1), exact, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(exact.mean(-1), data['score'][i, q], rtol=1e-7, atol=1e-7)
    terms = np.divide(delta*delta, distances[..., None], out=np.zeros_like(delta), where=distances[..., None] > 0)
    components = np.stack([terms[..., group].sum(-1).mean(-1) for group in GROUPS.values()], axis=-1)
    np.testing.assert_allclose(components, data['components'][i, q], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(delta.mean(1), data['signed_residual'][i, q], rtol=1e-7, atol=1e-7)
    return dict(episodes=len(frame), all_scored_queries=int(scored.sum()), direct_neighbor_queries=len(selected))


def main():
    checks, source_hashes = {}, {}
    manifest = json.loads((ROOT / 'input-manifest.json').read_text())
    for item in manifest['files']:
        path = ROOT / 'inputs' / item['source']
        content = path.read_bytes()
        blob = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        assert blob == item['git_blob'] and digest(path) == item['sha256'] and len(content) == item['bytes']
    checks['restored_git_blobs'] = len(manifest['files'])
    original = load_frozen()
    archived = pd.read_csv(ROOT / 'archived-episode-decisions.csv')
    pd.testing.assert_frame_equal(original, archived[original.columns], check_dtype=False)
    all_methods = decisions(original.v82_frozen, original.knn20)
    scopes = {}
    for name, part in [('all', original), ('A', original[original.run_id.eq(RUN_A)]),
                       ('B', original[original.run_id.eq(RUN_B)])] + [
                           ('B_'+suite, part) for suite, part in original[original.run_id.eq(RUN_B)].groupby('suite')]:
        scopes[name] = part, {key: value[part.index] for key, value in all_methods.items()}
    checks['archived_metric_rows'] = check_metrics(ROOT / 'archived-alarm-metrics.csv', scopes)
    b_original = original[original.run_id.eq(RUN_B)]
    checks['original_common_prefix_pairs'] = check_pairs(ROOT / 'archived-common-prefix-pairs.csv', b_original, all_methods)
    assert len(b_original) == 16000 and b_original.failure.sum() == 564
    expected = {'both': (456, 75), 'knn_only': (61, 140), 'v82_only': (19, 24), 'neither': (28, 15197)}
    for group, counts in expected.items():
        part = archived[archived.run_id.eq(RUN_B) & archived.joint_group.eq(group)]
        assert (int(part.failure.sum()), int((~part.failure).sum())) == counts
    print('Original archived decisions and all metric rows verified', flush=True)

    profile, params = load_profile()
    frame, raw = load_available()
    pd.testing.assert_frame_equal(frame, pd.read_csv(ROOT / 'available-index.csv'), check_dtype=False)
    ref, cal = profile['reference_available_rows'], profile['calibration_available_rows']
    config = json.loads((FROZEN / 'profiles/parameters.json').read_text())['geometry']
    for indices, split in ((ref, 'reference'), (cal, 'calibration')):
        expected_indices = np.flatnonzero(frame.run_id.eq(RUN_A) & frame.init_state_id.isin(config[split+'_initial_ids']))
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(profile[split+'_global_rows'], frame.iloc[indices].global_row)
    assert set(zip(frame.iloc[ref].task, frame.iloc[ref].init_state_id)).isdisjoint(
        set(zip(frame.iloc[cal].task, frame.iloc[cal].init_state_id)))
    period = raw['periodicity'][ref]
    pscale = np.quantile(np.abs(period[np.isfinite(period)]), .75)
    assert pscale == params['periodicity_scale']
    dynamic, _ = dynamics(raw['mobility'][ref], raw['acceleration'][ref], period, pscale)
    dynamic[~raw['valid'][ref]] = np.nan
    all_pairs = [(i, int(q)) for i, row in enumerate(dynamic) for finite in [np.flatnonzero(np.isfinite(row).all(-1))]
                 for q in finite[np.linspace(0, len(finite)-1, min(8, len(finite))).astype(int)]]
    pair_array = np.asarray(all_pairs)
    samples = dynamic[pair_array[:, 0], pair_array[:, 1]]
    center = np.median(samples, axis=0)
    scale = np.maximum(1.4826*np.median(np.abs(samples-center), axis=0), 1e-6)
    np.testing.assert_array_equal(profile['dynamic_center'], center)
    np.testing.assert_array_equal(profile['dynamic_scale'], scale)
    assert int(profile['scaling_points']) == len(samples)
    successes = ~frame.iloc[ref].failure.to_numpy()
    possible = pair_array[successes[pair_array[:, 0]]]
    selected = possible[np.random.default_rng(20260908).choice(len(possible), 4096, replace=False)]
    np.testing.assert_array_equal(profile['success_available_rows'], ref[selected[:, 0]])
    np.testing.assert_array_equal(profile['success_queries'], selected[:, 1])
    normalized = ((dynamic-center)/scale).astype(np.float32)
    np.testing.assert_array_equal(profile['success_dynamic'], normalized[selected[:, 0], selected[:, 1]].astype(float))
    with np.load(ROOT / 'available-profile/calibration.npz', allow_pickle=False) as saved:
        calibration = {key: saved[key] for key in saved.files if key != 'groups'}
    np.testing.assert_array_equal(calibration['available_rows'], cal)
    np.testing.assert_array_equal(calibration['failure'], frame.iloc[cal].failure)
    cal_x, _ = standardized({key: value[cal] for key, value in raw.items()}, profile)
    cal_finite = np.isfinite(cal_x).all(-1)
    np.testing.assert_array_equal(np.isfinite(calibration['scores']), cal_finite)
    selected = np.argwhere(cal_finite)[::max(1, int(cal_finite.sum())//96)]
    i, q = selected.T
    np.testing.assert_allclose(direct_distances(cal_x[i, q], profile['success_dynamic']).mean(-1),
                               calibration['scores'][i, q], rtol=1e-7, atol=1e-7)
    success = ~calibration['failure']
    groups = (frame.iloc[cal].task + '|' + frame.iloc[cal].init_state_id.astype(str)).to_numpy(str)
    group_names = sorted(set(groups[success]))
    np.testing.assert_array_equal(calibration['group_names'], group_names)
    peaks = []
    for group in group_names:
        scores = calibration['scores'][success & (groups == group)]
        peaks.append(scores[np.isfinite(scores)].max(initial=-np.inf))
    peaks = np.asarray(peaks)
    np.testing.assert_array_equal(peaks, calibration['grouped_peaks'])
    rank = int(np.ceil((len(peaks)+1)*.95))
    assert len(peaks) == 369 and rank == 352 and float(np.sort(peaks)[rank-1]) == params['threshold']
    checks['separate_A_profile'] = dict(reference_episodes=len(ref), calibration_episodes=len(cal),
                                       calibration_groups=len(peaks), rank=rank, threshold=params['threshold'],
                                       unscored_calibration_episodes=int((~cal_finite.any(1)).sum()),
                                       unscored_calibration_groups=int(np.isneginf(peaks).sum()),
                                       direct_calibration_queries=len(selected), B_used_for_fitting=False)
    print('A-only reference, normalization, sampled bank and calibration verified', flush=True)

    b = pd.read_csv(ROOT / 'available-B-decisions.csv')
    b_data = archive(ROOT / 'available-B-geometry.npz')
    chosen = frame.run_id.eq(RUN_B).to_numpy()
    pd.testing.assert_frame_equal(frame[chosen].reset_index(drop=True), b[frame.columns], check_dtype=False)
    b_x, _ = standardized({key: value[chosen] for key, value in raw.items()}, profile)
    checks['B_geometry'] = check_geometry(b, b_data, b_x, profile, params['threshold'])
    current = pd.read_csv(ROOT / 'current-decisions.csv')
    current_data = archive(ROOT / 'current-geometry.npz')
    current_raw = archive(ROOT / 'current-routing-inputs.npz')
    current_x, _ = standardized(current_raw, profile)
    checks['current_geometry'] = check_geometry(current, current_data, current_x, profile, params['threshold'])
    available_scopes = {}
    for prefix, part in [('B', b), ('current', current)]:
        methods = decisions(part.v82_frozen, part.knn_available_first)
        if prefix == 'B':
            methods['original_knn'] = part.knn20.to_numpy()
            subsets = [('B', part), ('B_supported_tasks', part[part.task_available_in_reference]),
                       ('B_missing_reference_tasks', part[~part.task_available_in_reference])]
            subsets += [('B_'+suite, sub) for suite, sub in part.groupby('suite')]
            checks['available_common_prefix_pairs'] = check_pairs(ROOT / 'available-B-common-prefix-pairs.csv', part,
                                                                   {k: v for k, v in methods.items() if k != 'original_knn'})
        else:
            subsets = [('current', part)] + [('current_'+name, sub) for name, sub in part.groupby('benchmark')]
        for name, sub in subsets:
            available_scopes[name] = sub, {key: value[sub.index] for key, value in methods.items()}
    checks['available_metric_rows'] = check_metrics(ROOT / 'available-alarm-metrics.csv', available_scopes)
    print('All B/current crossings and geometry sums verified; direct neighbors checked', flush=True)

    sys.path.insert(0, str(REPO / 'moe-trap-control'))
    from v82_closed_loop import V82Monitor
    episodes = json.loads((CURRENT / 'summary.json').read_text())['episodes']
    for i, episode in enumerate(episodes):
        assert current.iloc[i]['name'] == episode['name']
        hb_path = CURRENT / episode['name'] / 'full-hb-routes.npz'
        source_hashes[str(hb_path)] = digest(hb_path)
        assert source_hashes[str(hb_path)] == episode['integrity']['full_hb_sha256']
        trace_path = Path(episode['source_artifact_dir']) / 'episode-trace.npz'
        source_hashes[str(trace_path)] = digest(trace_path)
        assert source_hashes[str(trace_path)] == episode['source_trace_sha256']
        monitor = V82Monitor()
        with np.load(hb_path) as tensor:
            for q, probability in enumerate(tensor['hb_router_probs']):
                status = monitor.update(probability)
                np.testing.assert_array_equal(current_raw['mobility'][i, q], status['layer_mobility'])
                np.testing.assert_array_equal(current_raw['acceleration'][i, q], np.float32(status['route_acceleration']))
                np.testing.assert_array_equal(current_raw['periodicity'][i, q], np.float32(status['lag_periodicity']))
                np.testing.assert_array_equal(current_data['v82_scores'][i, q],
                    [status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']])
                np.testing.assert_array_equal(current_data['v82_thresholds'][i, q], status['v82_thresholds'])
        assert monitor.first_v82_alarm == episode['first_alarm_query_zero_based']['v82'] == current.iloc[i].v82_frozen
        heads = episode['first_head_query']
        assert monitor.v7.first_freeze_query == heads['freeze']
        assert monitor.v7.first_turbulence_query == heads['turbulence']
        np.testing.assert_array_equal(monitor.v82_first, [heads['v82_inversion'], heads['v82_curvature']])
    checks['all_heads_and_continuous_v82_current_replays'] = len(episodes)
    print('All 60 original v8.2 route replays and source tensors verified', flush=True)

    deletion = archive(ROOT / 'reference-deletion-scores.npz')['score']
    identities = archive(ROOT / 'reference-deletion-identities.npz')
    table = pd.read_csv(ROOT / 'reference-deletion-by-task.csv')
    bank_tasks = original.iloc[profile['success_global_rows']].task.to_numpy()
    explanations = pd.read_csv(ROOT / 'neighbor-explanations.csv', dtype={'identity': str})
    for cohort, parent_frame, geometry in (('B', b, b_data), ('current', current, current_data)):
        part = explanations[explanations.cohort.eq(cohort)]
        parents = parent_frame.iloc[part.row.to_numpy()]
        keys = np.asarray([task if task.startswith(suite + '/') else suite + '/' + task
                           for task, suite in zip(parents.task, parents.suite)], dtype=object)
        np.testing.assert_array_equal(keys, part.reference_task_key)
        neighbor_ids = geometry['neighbor_ids'][part.row.to_numpy(), part['query'].to_numpy()]
        expected = (bank_tasks[neighbor_ids] == keys[:, None]).mean(-1)
        np.testing.assert_allclose(expected, part.neighbor_same_task_fraction, atol=1e-14)
    checks['neighbor_task_identity_comparisons'] = len(explanations)
    for task_index, (task, part) in enumerate(b.groupby('task', sort=True)):
        task_ids = np.flatnonzero(bank_tasks == task)
        if not len(task_ids):
            assert np.isnan(deletion[:, part.index]).all()
            continue
        random_ids = np.random.default_rng(20260915+task_index).choice(len(bank_tasks), len(task_ids), replace=False)
        for method, removed in enumerate((task_ids, random_ids)):
            np.testing.assert_array_equal(identities[f'{task_index}_{method}'], removed)
            after, before = deletion[method, part.index], b_data['score'][part.index]
            np.testing.assert_array_equal(np.isfinite(after), np.isfinite(before))
            valid = np.isfinite(before)
            assert (after[valid] >= before[valid]-1e-6).all()
            keep = np.ones(len(bank_tasks), bool)
            keep[removed] = False
            i, q = part.index[0], 9
            direct = direct_distances(b_x[i, q][None], profile['success_dynamic'][keep]).mean()
            np.testing.assert_allclose(direct, deletion[method, i, q], rtol=1e-7, atol=1e-7)
            h0, h1 = crossings(before, params['threshold']) >= 0, crossings(after, params['threshold']) >= 0
            for failure in (False, True):
                mask = part.failure.to_numpy() == failure
                row = table[table.task.eq(task) & table.method.eq(('same_task', 'random')[method]) & table.failure.eq(failure)].iloc[0]
                assert (int(h0[mask].sum()), int(h1[mask].sum()), int((h1 & ~h0 & mask).sum())) == (
                    row.original_hits, row.deletion_hits, row.added_hits)
    expected_totals = table.groupby(['method', 'failure'])[['episodes', 'original_hits', 'deletion_hits', 'added_hits']].sum().reset_index()
    pd.testing.assert_frame_equal(expected_totals, pd.read_csv(ROOT / 'reference-deletion-summary.csv'))
    checks['coverage_deletion_task_method_outcome_rows'] = len(table)

    aucs = pd.read_csv(ROOT / 'available-same-query-aucs.csv')
    readouts = dict(knn_distance=b_data['score'], freeze=b_data['freeze_score'],
                    acceleration=b_data['acceleration_score'], periodicity=b_data['periodicity_score'])
    readouts.update({name+'_distance_term': b_data['components'][..., j] for j, name in enumerate(GROUPS)})
    for row in aucs.itertuples():
        subset = b[b.task.eq(row.task) & b.length.gt(row.query)]
        values = readouts[row.measurement][subset.index, row.query]
        y = subset.failure.to_numpy()
        positives, negatives = values[y], values[~y]
        direct = mannwhitneyu(positives, negatives, method='asymptotic').statistic / (len(positives)*len(negatives))
        np.testing.assert_allclose(direct, row.auc, atol=1e-14)
    checks['independent_Mann_Whitney_AUROC_rows'] = len(aucs)
    anchors = pd.read_csv(ROOT / 'functional-anchor-context.csv')
    for row in anchors.itertuples():
        i = int(current.index[current.name.eq(row.name)][0])
        assert row.v82_on == (0 <= current.iloc[i].v82_frozen <= row.query)
        assert row.knn_on == (0 <= current.iloc[i].knn_available_first <= row.query)
        np.testing.assert_allclose(row.knn_distance, current_data['score'][i, row.query], atol=1e-14)
    assert len(anchors) == 16 and not anchors.v82_on.any() and anchors.knn_on.sum() == 2
    assert (anchors.knn_distance > params['threshold']).sum() == 1
    checks['functional_anchor_context'] = dict(parents=16, v82_on=0, knn_ever_on=2, knn_current_exceedance=1)
    print('Controlled deletions, signed-score ranking and functional anchors verified', flush=True)

    figures = {}
    for name in ('original-v82-knn-comparison', 'knn-mechanism-diagnostics', 'current-alarm-cases'):
        with Image.open(ROOT / (name+'.png')) as image:
            pixels = np.asarray(image.convert('RGB'))
        assert pixels.std() > 10 and (pixels < 240).any(-1).mean() > .05
        assert (ROOT / (name+'.pdf')).stat().st_size > 10000
        figures[name] = dict(width=int(pixels.shape[1]), height=int(pixels.shape[0]), visually_inspected=True)
    for case in json.loads((ROOT / 'case-figure-index.json').read_text()):
        with np.load(Path(case['source_dir']) / 'episode-trace.npz') as trace:
            assert trace[case['terminal_image_key']].shape == trace['images'][0].shape
            assert max(case['queries']) < len(trace['images'])
            assert int(trace['executed_lengths'].sum()) == case['terminal_actions']
    checks['figures'] = figures
    preserved = {}
    for name in ('moe-joint-patterns-20260915', 'moe-joint-expanded-20260915', 'moe-input-response-20260915'):
        directory = ROOT.parent / name
        hashes = json.loads((directory / 'artifact-hashes.json').read_text())
        for relative, expected_hash in hashes.items():
            assert digest(directory / relative) == expected_hash, f'Prior artifact changed: {name}/{relative}'
        preserved[name] = len(hashes)
    checks['unchanged_prior_sealed_artifacts'] = preserved
    load_profile()
    result = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_v82_knn_research'],
                            cwd=Path(__file__).parent, capture_output=True, text=True)
    (ROOT / 'tests.txt').write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and 'Ran 9 tests' in result.stderr
    checks['unit_tests_passed'] = 9
    status = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=REPO,
                            check=True, capture_output=True, text=True).stdout
    assert not status, f'Repository worktree changed: {status}'
    checks['repository_worktree_clean'] = True
    implementation = [Path(__file__).parent / name for name in (
        'restore_v82_knn_inputs.py', 'v82_knn_research.py', 'analyze_v82_knn_archives.py',
        'rebuild_v82_knn_reference.py', 'score_v82_knn_reference.py', 'analyze_v82_knn_geometry.py',
        'diagnose_v82_knn_coverage.py', 'plot_v82_knn_research.py', 'test_v82_knn_research.py',
        'verify_v82_knn_research.py', 'moe_response_analysis.py')]
    save_json(ROOT / 'implementation-hashes.json', {str(path): digest(path) for path in implementation})
    for path in [*RAW.glob('*extraction_audit.json'), *RAW.glob('*route_features.npz'), CURRENT / 'summary.json',
                 SAFE / 'boundary_knn/knn.py', SAFE / 'feature_geometry/analyze.py',
                 REPO / 'moe-v7-0905/method/intrinsic_guard_monitor.py',
                 REPO / 'moe-trap-control/v82_closed_loop.py', REPO / 'moe-trap-control/v8_feature_control.py',
                 ROOT.parent / 'moe-input-response-20260915/selection.json']:
        source_hashes[str(path)] = digest(path)
    save_json(ROOT / 'source-hashes.json', source_hashes)
    assert (ROOT / 'REPORT.zh.md').is_file()
    save_json(ROOT / 'verification.json', dict(passed=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
              checks=checks, model_forwards=0, environment_actions=0, original_knn_bank_reproduced=False,
              validated_new_trap_detector=False, research_only=True))
    paths = [path for path in ROOT.rglob('*') if path.is_file() and path.name != 'artifact-hashes.json']
    save_json(ROOT / 'artifact-hashes.json', {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)})
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
