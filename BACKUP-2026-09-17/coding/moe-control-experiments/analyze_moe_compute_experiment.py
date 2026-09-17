"""MoE-only temporal diagnostics and grouped future-MoE prediction."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial.distance import pdist, squareform

from audit_replan_window import read, records
from gate_runtime import BASE
from moe_compute_protocol import (FUTURE, KINDS, WINDOW, combine_distances, equal_parent_metrics,
                                  feature_sets, fit_ridge, parent_weights, port_distance, predict_ridge)
from run_moe_compute_experiment import P3F, P3I, check_config
from run_online_experiment import now, save_json
from topology_protocol import (HISTORY, REPRESENTATIONS, SCALES, SEED, delayed_distance, fit_reference,
                               region_diagnostic, route_distance, topology_nulls)
from collection_routes import PROBS_KEY
from himoe_libero_bridge.episode_trace import sha256_file
from v82_closed_loop import V82Monitor


def save_npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def wait_json(root, name):
    while True:
        if (root / 'failure.json').exists():
            raise RuntimeError('MoE collection failed; analysis cannot establish completeness')
        try:
            return read(root / name)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)


def wait_saved(root, paths):
    required = {str(path.relative_to(root)) for path in paths}
    while True:
        if (root / 'failure.json').exists():
            raise RuntimeError('MoE collection failed')
        with (root / 'calls.jsonl').open() as stream:
            complete = [json.loads(line) for line in stream if line.endswith('\n')]
        written = {row['path']: row['sha256'] for row in complete if row['event'] == 'response'}
        if required <= written.keys():
            for path in paths:
                assert sha256_file(path) == written[str(path.relative_to(root))]
            return
        time.sleep(1)


def branch_paths(root, parent, replicate, arm):
    return ([root / 'prefixes' / parent['name'] / ('q%03d.npz' % q) for q in range(parent['query'])] +
            [root / 'branches' / parent['name'] / ('r%d' % replicate) / arm / ('s%03d.npz' % step)
             for step in range(parent['start'], 520, 10)])


def load_tensors(paths, fields=('input', 'total')):
    values = {field: [] for field in fields}
    routes = []
    for path in paths:
        with np.load(path) as saved:
            routes.append(saved[PROBS_KEY])
            for field in fields:
                values[field].append(saved['mechanism/' + field])
    return {field: np.stack(rows) for field, rows in values.items()}, np.stack(routes)


def port_matrices(values, routes, prefix):
    matrices = {}
    for scope in REPRESENTATIONS:
        matrices['route_' + scope] = route_distance(routes, scope)
        for field in ('input', 'total'):
            normalized, raw, scale = port_distance(values[field], scope, prefix)
            matrices[field + '_' + scope] = normalized
            matrices['raw_' + field + '_' + scope] = raw
            matrices['scale_' + field + '_' + scope] = scale
        matrices['input_total_' + scope] = combine_distances(matrices['input_' + scope], matrices['total_' + scope])
    last = values['total'][:, -1, -1, 1:].astype(float).reshape(len(routes), -1)
    target_scale = max(1e-12, float(np.sqrt(np.mean(last[:prefix] ** 2))))
    matrices['target_distance'] = np.sqrt(squareform(pdist(last, metric='sqeuclidean')) / last.shape[1]) / target_scale
    matrices['target_scale'] = np.asarray(target_scale)
    return matrices


def v82_features(routes):
    monitor, rows = V82Monitor(), []
    for probability in routes:
        status = monitor.update(probability)
        rows.append([status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']])
    return np.asarray(rows, float)


def noise_analysis(root, config):
    results = []
    for parent in config['parents']:
        prefixes = branch_paths(root, parent, 0, 'native10')[:parent['query'] + 1]
        directory = root / 'controls' / parent['name']
        paths = [directory / ('capture.npz' if o == n == 0 else 'o%d-n%d.npz' % (o, n))
                 for o in range(3) for n in range(3)]
        values, routes = load_tensors(prefixes + paths, ('input', 'shared', 'total'))
        prefix = len(prefixes)
        distance_file = {}
        for scope in REPRESENTATIONS:
            distances = {'route': route_distance(routes, scope)[prefix:, prefix:]}
            for field in ('input', 'shared', 'total'):
                distance, raw, scale = port_distance(values[field], scope, prefix)
                distances[field] = distance[prefix:, prefix:]
                distance_file['raw_' + field + '_' + scope] = raw[prefix:, prefix:]
                distance_file['scale_' + field + '_' + scope] = scale
            distances['input_total'] = combine_distances(distances['input'], distances['total'])
            for kind, d in distances.items():
                noise = [float(d[o * 3 + a, o * 3 + b]) for o in range(3) for a in range(3) for b in range(a + 1, 3)]
                observation = [float(d[a * 3 + n, b * 3 + n]) for n in range(3) for a in range(3) for b in range(a + 1, 3)]
                results.append(dict(parent=parent['name'], scope=scope, kind=kind, noise_distances=noise,
                                    observation_distances=observation, noise_mean=float(np.mean(noise)),
                                    observation_mean=float(np.mean(observation)),
                                    noise_observation_ratio=None if np.mean(observation) <= 1e-12 else float(np.mean(noise) / np.mean(observation))))
                distance_file[kind + '_' + scope] = d
        save_npz(root / 'noise-distances' / (parent['name'] + '.npz'), **distance_file)
        print('Noise/observation MoE grid: ' + parent['name'], flush=True)
    save_json(root / 'noise-analysis.json', results)


def conditional_analysis(root, config):
    cases = read(P3I / 'config.json')['crossover_cases']
    labels, results = ('native', 'route_only', 'joint', 'output_only'), []
    for case in cases:
        values, routes = load_tensors([P3F / 'states' / case['parent'] / case['phase'] / (label + '.npz') for label in labels])
        # P3f stores the original rather than effective route in PROBS_KEY.
        routes = []
        for label in labels:
            with np.load(P3F / 'states' / case['parent'] / case['phase'] / (label + '.npz')) as saved:
                routes.append(saved['v8_control/effective_probs_fp32'].astype(np.float16))
        d = port_matrices(values, np.stack(routes), 1)
        for field in ('input', 'total'):
            for a, b in ((0, 1), (2, 3)):
                assert np.array_equal(values[field][a], values[field][b])
                assert d[field + '_back_path'][a, b] == 0
        assert d['route_back_path'][0, 1] > 0 and d['route_back_path'][2, 3] > 0
        results.append(dict(parent=case['parent'], phase=case['phase'], labels=labels,
                            distances={kind: d[kind + '_back_path'].tolist() for kind in KINDS},
                            exact_compute_pairs=2, constructed_control=True))
    save_json(root / 'conditional-analysis.json', results)


def grouped_prediction(root, rows, arrays):
    y, persistence_prediction = arrays['target'], arrays['past_displacement']
    parents = np.asarray([r['parent'] for r in rows])
    tasks = np.asarray([r['task'] for r in rows])
    features = {k: v for k, v in arrays.items() if k not in ('target', 'past_displacement')}
    methods = list(features) + ['constant', 'persistence']
    predictions = {name: np.full(len(y), np.nan) for name in methods}
    fits = []
    for task in sorted(set(tasks.tolist())):
        train, test = tasks != task, tasks == task
        assert not set(parents[train]) & set(parents[test])
        weights = parent_weights(parents[train])
        for name, x in features.items():
            fitted = fit_ridge(x[train], y[train], weights)
            predictions[name][test] = predict_ridge(fitted, x[test])
            fits.append(dict(task=task, method=name, training_parents=sorted(set(parents[train].tolist())),
                             test_parents=sorted(set(parents[test].tolist())), model=fitted))
        predictions['constant'][test] = np.dot(weights, y[train])
        predictions['persistence'][test] = persistence_prediction[test]
    metrics = {}
    for name, prediction in predictions.items():
        assert np.isfinite(prediction).all()
        metric = equal_parent_metrics(y, prediction, parents)
        metric['tasks'] = {str(t): equal_parent_metrics(y[tasks == t], prediction[tasks == t], parents[tasks == t])
                           for t in sorted(set(tasks.tolist()))}
        metrics[name] = metric
    gates = {}
    for gate, baseline, candidate in (('representation', 'route_S', 'input_total_S'),
                                       ('topology_simple', 'input_total_S', 'input_total_ST'),
                                       ('topology_dense', 'input_total_D', 'input_total_DT')):
        reduction = 1 - metrics[candidate]['rmse'] / metrics[baseline]['rmse']
        improved = sum(metrics[candidate]['tasks'][str(t)]['rmse'] < metrics[baseline]['tasks'][str(t)]['rmse']
                       for t in sorted(set(tasks.tolist())))
        gates[gate] = dict(baseline=baseline, candidate=candidate, relative_rmse_reduction=reduction,
                           tasks_improved=improved, passed=bool(reduction >= .10 and improved >= 2))
    save_npz(root / 'predictions.npz', **predictions)
    save_json(root / 'prediction-fits.json', fits)
    result = dict(passed=True, rows=len(rows), parent_groups=len(set(parents)), task_groups=len(set(tasks)),
                  target='future MoE layer15 last-flow total distance after 40 physical steps',
                  metrics=metrics, gates=gates, interpretation='development grouped prediction, not success or rescue')
    save_json(root / 'prediction-summary.json', result)
    return result


def analyze(root, streaming=False):
    config = check_config(root)
    assert (wait_json(root, 'preflight.json') if streaming else read(root / 'collection.json'))['passed']
    assert not (root / 'analysis-started.json').exists()
    source_sha256 = sha256_file(Path(__file__))
    save_json(root / 'analysis-started.json', dict(utc=now(), source_sha256=source_sha256, streaming=streaming))
    started = time.monotonic()
    try:
        for name in ('distances', 'noise-distances'):
            (root / name).mkdir()
        if not streaming:
            noise_analysis(root, config)
        conditional_analysis(root, config)
        references, regions, homology, rows = [], [], [], []
        feature_rows = {kind + '_' + method: [] for kind in KINDS for method in ('S', 'ST', 'D', 'DT')}
        feature_rows.update(v82=[], target=[], past_displacement=[])
        for number, parent in enumerate(config['parents']):
            for replicate in range(4):
                pair_distances = {}
                for arm in ('native10', 'window5'):
                    paths = branch_paths(root, parent, replicate, arm)
                    if streaming:
                        wait_saved(root, paths)
                    values, routes = load_tensors(paths)
                    assert len(routes) == 52
                    matrices = port_matrices(values, routes, parent['query'] + 1)
                    del values
                    with np.load(P3I / 'distances' / ('%s-r%d-%s.npz' % (parent['name'], replicate, arm))) as prior:
                        for scope in REPRESENTATIONS:
                            np.testing.assert_array_equal(matrices['route_' + scope], prior['raw_' + scope])
                    for kind in KINDS:
                        for scope in REPRESENTATIONS:
                            matrices['delay_' + kind + '_' + scope] = delayed_distance(matrices[kind + '_' + scope])
                        primary = matrices[kind + '_back_path']
                        suffix = primary[parent['query']:, parent['query']:]
                        ph = topology_nulls(suffix, SEED + number * 100 + replicate)
                        homology.append(dict(parent=parent['name'], replicate=replicate, arm=arm, kind=kind, **ph))
                    save_npz(root / 'distances' / ('%s-r%d-%s.npz' % (parent['name'], replicate, arm)), **matrices)
                    pair_distances[arm] = matrices
                    scores = v82_features(routes)
                    for query in range(parent['query'], 52 - FUTURE):
                        rows.append(dict(parent=parent['name'], task=parent['base_task_id'], replicate=replicate,
                                         arm=arm, query=query, physical_step=query * 10,
                                         feature_start=(query - WINDOW + 1) * 10, target_step=(query + FUTURE) * 10))
                        feature_rows['target'].append(matrices['target_distance'][query, query + FUTURE])
                        feature_rows['past_displacement'].append(matrices['target_distance'][query, query - FUTURE])
                        feature_rows['v82'].append(scores[query])
                        for kind in KINDS:
                            local = matrices[kind + '_back_path'][query - WINDOW + 1:query + 1, query - WINDOW + 1:query + 1]
                            for method, features in feature_sets(local).items():
                                feature_rows[kind + '_' + method].append(features)
                    print('MoE history %s r%d %s complete' % (parent['name'], replicate, arm), flush=True)
                anchor = parent['query'] - HISTORY + 1
                for kind in KINDS:
                    for scope in REPRESENTATIONS:
                        key = 'delay_' + kind + '_' + scope
                        native, window = pair_distances['native10'][key], pair_distances['window5'][key]
                        np.testing.assert_array_equal(native[:anchor + 1, :anchor + 1], window[:anchor + 1, :anchor + 1])
                        for scale in SCALES:
                            reference = fit_reference(native, anchor, scale)
                            assert reference == fit_reference(window, anchor, scale)
                            references.append(dict(parent=parent['name'], replicate=replicate, kind=kind, scope=scope,
                                                   scale=scale, anchor=anchor, **reference))
                            for arm in ('native10', 'window5'):
                                diagnostic = region_diagnostic(pair_distances[arm][key], anchor, reference)
                                regions.append(dict(parent=parent['name'], replicate=replicate, arm=arm, kind=kind, scope=scope,
                                                    scale=scale, reference_eligible=reference['eligible'], epsilon=reference['epsilon'],
                                                    endpoint_success=False, **diagnostic))
        assert wait_json(root, 'collection.json')['passed']
        if streaming:
            noise_analysis(root, config)
        assert sha256_file(Path(__file__)) == source_sha256
        arrays = {name: np.asarray(values, float) for name, values in feature_rows.items()}
        assert len(rows) == 1256 and all(np.isfinite(a).all() for a in arrays.values())
        save_npz(root / 'features.npz', **arrays)
        save_json(root / 'feature-rows.json', rows)
        save_json(root / 'references.json', references)
        save_json(root / 'regions.json', regions)
        save_json(root / 'persistence.json', homology)
        prediction = grouped_prediction(root, rows, arrays)
        primary_counts, stability = [], []
        for kind in KINDS:
            for arm in ('native10', 'window5'):
                selected = [r for r in regions if r['kind'] == kind and r['scope'] == 'back_path' and r['arm'] == arm and r['scale'] == 1]
                primary_counts.append(dict(kind=kind, arm=arm, counts=dict(Counter(r['category'] for r in selected))))
                for scope in REPRESENTATIONS:
                    stable = sum(len({r['category'] for r in regions if r['parent'] == p['name'] and r['replicate'] == replicate
                                      and r['kind'] == kind and r['scope'] == scope and r['arm'] == arm}) == 1
                                 for p in config['parents'] for replicate in range(4))
                    stability.append(dict(kind=kind, scope=scope, arm=arm, stable=stable, denominator=24))
        save_json(root / 'analysis-summary.json', dict(passed=True, utc=now(), elapsed_seconds=time.monotonic() - started,
                  region_rows=len(regions), references=len(references), homology_rows=len(homology),
                  surrogate_series=len(homology) * 198, feature_rows=len(rows), primary_counts=primary_counts,
                  stability=stability, prediction_gates=prediction['gates'], new_environment_actions=0))
        print('MoE-only analysis complete', flush=True)
    except BaseException as error:
        save_json(root / 'analysis-failure.json', dict(error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    analyze(args.run.resolve(), args.stream)
