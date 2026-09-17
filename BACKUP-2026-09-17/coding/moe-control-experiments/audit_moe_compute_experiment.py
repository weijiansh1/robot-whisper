"""Read-only source, temporal isolation, graph, and regression audits."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from audit_replan_window import read, records
from analyze_moe_compute_experiment import branch_paths
from gate_runtime import BASE
from moe_compute_protocol import KINDS, FUTURE, WINDOW, noise_grid
from run_moe_compute_experiment import P3H, P3I, P3F, check_config
from run_online_experiment import EVALUATION, now, save_json, sha_array
from topology_protocol import REPRESENTATIONS, SCALES, SEED, topology_nulls, persistence
from collection_routes import PROBS_KEY
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from v82_closed_loop import V82Monitor


def close(a, b, name='numeric audit', tolerance=1e-11):
    np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance, err_msg=name)


def select(value, scope):
    a = np.asarray(value, float)
    if scope == 'full_path':
        return a[:, :, 1:]
    if scope == 'back_path':
        return a[4:, :, 1:]
    if scope == 'back_last':
        return a[4:, -1:, 1:]
    raise ValueError(scope)


def direct_distance(first, second, scope, scales):
    delta = select(first, scope) - select(second, scope)
    return float(np.sqrt(np.mean(delta ** 2 / np.asarray(scales)[:, None, None, None] ** 2)))


def graph_reference(distance, anchor, scale):
    indices = np.arange(anchor - 11, anchor + 1)
    local = distance[np.ix_(indices, indices)]
    separated = np.abs(indices[:, None] - indices[None, :]) >= 3
    epsilon = max(1e-12, np.quantile(np.where(separated, local, np.inf).min(1), .95) * scale)
    labels = fcluster(linkage(squareform(local, checks=False), method='complete'), epsilon, criterion='distance') - 1
    graph = nx.DiGraph()
    graph.add_nodes_from(np.unique(labels).tolist())
    graph.add_edges_from(zip(labels[:-1].tolist(), labels[1:].tolist()))
    component = next(c for c in nx.strongly_connected_components(graph) if int(labels[-1]) in c)
    selected = [i for i, label in enumerate(labels) if label in component]
    length = sum(float(local[a, b]) for a, b in zip(selected[:-1], selected[1:]))
    ratio = 0. if length <= 1e-12 else float(local[selected[0], selected[-1]]) / length
    cyclic = len(component) > 1 or graph.has_edge(labels[-1], labels[-1])
    eligible = len(selected) >= 4 and selected[-1] - selected[0] >= 3 and cyclic and ratio <= .35
    return dict(epsilon=float(epsilon), labels=labels.tolist(), region_indices=indices[selected].tolist(),
                eligible=bool(eligible), net_to_path_ratio=ratio, selected_count=len(selected))


def independent_features(raw):
    step = np.array([raw[i, i + 1] for i in range(13)])
    diameter = max(raw.flat)
    separated = [raw[i, j] for i in range(14) for j in range(14) if abs(i - j) >= 3]
    simple = [raw[13, 13 - lag] for lag in (1, 2, 4, 8, 13)] + [
        float(np.mean(step)), float(np.std(step)), float(max(step)),
        0. if sum(step) <= 1e-12 else float(raw[0, 13] / sum(step)), min(raw[13, :11]), diameter,
        0. if diameter <= 1e-12 else float(np.median(separated) / diameter)]
    delayed = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            delayed[i, j] = np.sqrt(sum(raw[i + k, j + k] ** 2 for k in range(3)) / 3)
    ph = persistence(delayed)
    topology = [ph['h1_max_lifetime'], ph['normalized_h1'], max((b - a for a, b in ph['h0_finite']), default=0.)]
    for scale in SCALES:
        ref = graph_reference(delayed, 11, scale)
        topology.extend([ref['selected_count'] / 12, ref['net_to_path_ratio'], float(ref['eligible'])])
    dense = [raw[i, j] for i in range(14) for j in range(i + 1, 14)]
    return dict(S=np.array(simple), ST=np.r_[simple, topology], D=np.array(dense), DT=np.r_[dense, topology])


def classify(inside):
    runs = []
    begin = 0
    for end in range(1, len(inside) + 1):
        if end == len(inside) or inside[end] != inside[begin]:
            runs.append((bool(inside[begin]), begin, end - 1))
            begin = end
    exits = [(a, b) for value, a, b in runs if not value and b - a + 1 >= 3]
    returned = any(value and b - a + 1 >= 2 and any(a > e for _, e in exits) for value, a, b in runs)
    if not exits:
        return 'no_confirmed_exit'
    if exits[-1][1] == len(inside) - 1 and exits[-1][1] - exits[-1][0] + 1 >= 4:
        return 'last_exit_no_observed_return'
    if len(inside) >= 2 and all(inside[-2:]) and returned:
        return 'exit_then_return'
    return 'insufficient_followup_or_mixed'


def wait_result(root, name):
    while True:
        if (root / 'failure.json').exists() or (root / 'analysis-failure.json').exists():
            raise RuntimeError('Collection or analysis failed; cannot complete audit')
        try:
            return read(root / name)
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(1)


def response_stream(root, log):
    with (root / 'calls.jsonl').open() as stream:
        for ordinal in range(1699):
            for event in ('request', 'response'):
                while True:
                    if (root / 'failure.json').exists():
                        raise RuntimeError('Collection failed')
                    position = stream.tell()
                    line = stream.readline()
                    if line.endswith('\n'):
                        break
                    stream.seek(position)
                    time.sleep(1)
                row = json.loads(line)
                assert row['event'] == event and row['ordinal'] == ordinal
                log.append(row)
            yield row


def audit_collection(root, config, streaming=False):
    log = [] if streaming else records(root / 'calls.jsonl')
    iterator = response_stream(root, log) if streaming else (r for r in log if r['event'] == 'response')
    parents = {p['name']: p for p in config['parents']}
    sources = {p['name']: load_episode_trace(Path(p['source']))[1] for p in config['parents']}
    routes = {p['name']: np.load(EVALUATION / p['name'] / 'full-hb-routes.npz')['hb_router_probs'] for p in config['parents']}
    stats = {}
    for index, row in enumerate(iterator):
        path, parent = root / row['path'], parents[row['parent']]
        assert sha256_file(path) == row['sha256'] and path.stat().st_size == row['file_bytes']
        with np.load(path) as saved:
            assert all(k.startswith(('mechanism/', 'collection/', 'routing/', 'v8_control/')) for k in saved.files)
            ports = {field: saved['mechanism/' + field] for field in ('input', 'shared', 'total')} if row['port_capture'] else {}
            assert {k for k in saved.files if k.startswith('mechanism/')} == {'mechanism/' + f for f in ports}
            for value in ports.values():
                assert value.shape == (8, 10, 11, 1024) and value.dtype == np.float32 and np.isfinite(value).all()
            if ports:
                ports['routed_derived'] = ports['total'].astype(float) - ports['shared']
                stats[row['path']] = {field: {scope: np.sqrt(np.mean(select(value, scope) ** 2, axis=(1, 2, 3))).tolist()
                                             for scope in REPRESENTATIONS} for field, value in ports.items()}
            assert sha_array(saved[PROBS_KEY]) == row['hb_sha256']
            if row['phase'] == 'prefix':
                q, original = row['query'], sources[parent['name']]
                assert row['actions_sha256'] == sha_array(original['predicted_actions'][q])
                assert row['noise_sha256'] == sha_array(original['flow_noises'][q])
                np.testing.assert_array_equal(saved[PROBS_KEY], routes[parent['name']][q])
            elif row['phase'] in ('suffix', 'source_duplicate') or (row['phase'] == 'control' and 'label' in row):
                source = P3H / row['source'] if 'source' in row else P3H / parent['name'] / 'r0/native10/queries' / ('s%03d.npz' % parent['start'])
                with np.load(source) as old:
                    assert row['actions_sha256'] == sha_array(old['actions'])
                    assert row['noise_sha256'] == sha_array(old['request_noise'])
                    for key in old.files:
                        if key.startswith(('collection/', 'routing/', 'v8_control/')):
                            np.testing.assert_array_equal(saved[key], old[key])
            else:
                assert row['phase'] == 'control' and 0 <= row['observation'] < 3 and 0 <= row['noise'] < 3
                assert Path(row['source']) == P3H / parent['name'] / 'r0/native10/queries' / ('s%03d.npz' % (parent['start'] + 20 * row['observation']))
                source = P3H / parent['name'] / 'r0/native10/queries' / ('s%03d.npz' % parent['start'])
                with np.load(source) as old:
                    noises = noise_grid(config['parents'].index(parent), old['request_noise'])
                assert row['noise_sha256'] == sha_array(noises[row['noise']])
        if (index + 1) % 200 == 0:
            print('Artifact/source audit %d/1699' % (index + 1), flush=True)
    requests = [r for r in log if r['event'] == 'request']
    responses = [r for r in log if r['event'] == 'response']
    assert len(requests) == len(responses) == config['expected_model_calls'] == 1699
    assert [r['ordinal'] for r in requests] == [r['ordinal'] for r in responses] == list(range(1699))
    assert [r['event'] for r in log] == ['request', 'response'] * 1699
    assert len({r['path'] for r in responses}) == 1699
    assert Counter(r['phase'] for r in responses) == dict(control=66, prefix=131, suffix=1496, source_duplicate=6)
    assert all(r['parameter_versions_unchanged'] for r in responses)
    first_checks = 0
    for parent in config['parents']:
        control = root / 'controls' / parent['name']
        with np.load(control / 'capture.npz') as native, np.load(control / 'repeat.npz') as repeat, np.load(control / 'bare.npz') as bare:
            for key in native.files:
                np.testing.assert_array_equal(native[key], repeat[key])
                if not key.startswith('mechanism/'):
                    np.testing.assert_array_equal(native[key], bare[key])
            for replicate in range(4):
                for arm in ('native10', 'window5'):
                    path = root / 'branches' / parent['name'] / ('r%d' % replicate) / arm / ('s%03d.npz' % parent['start'])
                    with np.load(path) as first:
                        for field in ('input', 'shared', 'total'):
                            np.testing.assert_array_equal(first['mechanism/' + field], native['mechanism/' + field])
                    first_checks += 1
    collection = wait_result(root, 'collection.json')
    assert collection['parameter_sha256_before'] == collection['parameter_sha256_after'] == config['parameter_reference_sha256']
    assert collection['calls'] == collection['attempts'] == 1699
    assert collection['bytes_saved'] == sum(r['file_bytes'] for r in responses)
    close(collection['model_inference_seconds'], sum(r['resource']['seconds'] for r in responses))
    save_json(root / 'port-amplitudes.json', dict(passed=True, unit='RMS of actual MoE port tensors',
        derived_warning='total-shared includes the effects of floating-point addition rounding; not exact individual expert output', rows=stats))
    return stats, dict(calls=1699, captured_queries=len(stats), exact_first_proposals=first_checks, source_requests_exact=1633,
                      original_action_arrays_not_used_as_features=True, parameter_content_hashes_equal=True)


def audit_distances_and_features(root, config, stats):
    rows = read(root / 'feature-rows.json')
    features = dict(np.load(root / 'features.npz'))
    indexed = {}
    for i, row in enumerate(rows):
        indexed.setdefault((row['parent'], row['replicate'], row['arm']), []).append((i, row))
        assert row['target_step'] == row['physical_step'] + 40
        assert row['feature_start'] == row['physical_step'] - 130
        assert 0 <= row['feature_start'] < row['physical_step'] < row['target_step'] <= 510
    checked_pairs = 0
    for parent in config['parents']:
        for replicate in range(4):
            for arm in ('native10', 'window5'):
                paths = branch_paths(root, parent, replicate, arm)
                with np.load(root / 'distances' / ('%s-r%d-%s.npz' % (parent['name'], replicate, arm))) as saved:
                    d = {k: saved[k] for k in saved.files}
                prefix_stats = [stats[str(p.relative_to(root))] for p in paths[:parent['query'] + 1]]
                for scope in REPRESENTATIONS:
                    for field in ('input', 'total'):
                        scale = np.sqrt(np.mean([np.asarray(s[field][scope]) ** 2 for s in prefix_stats], axis=0))
                        close(d['scale_' + field + '_' + scope], np.maximum(scale, 1e-12))
                    close(d['input_total_' + scope] ** 2, (d['input_' + scope] ** 2 + d['total_' + scope] ** 2) / 2)
                target_scale = np.sqrt(np.mean([s['total']['back_last'][-1] ** 2 for s in prefix_stats]))
                close(d['target_scale'], max(target_scale, 1e-12))
                for a, b in ((0, parent['query']), (parent['query'], parent['query'] + 4), (parent['query'], 51)):
                    with np.load(paths[a]) as first, np.load(paths[b]) as second:
                        for scope in REPRESENTATIONS:
                            for field in ('input', 'total'):
                                direct = direct_distance(first['mechanism/' + field], second['mechanism/' + field],
                                                         scope, d['scale_' + field + '_' + scope])
                                close(d[field + '_' + scope][a, b], direct)
                                checked_pairs += 1
                        delta = first['mechanism/total'][-1, -1, 1:].astype(float) - second['mechanism/total'][-1, -1, 1:]
                        close(d['target_distance'][a, b], np.sqrt(np.mean(delta ** 2)) / target_scale)
                monitor, scores = V82Monitor(), []
                for path in paths:
                    with np.load(path) as data:
                        status = monitor.update(data[PROBS_KEY])
                    scores.append([status['freeze_score'], status['acceleration_score'], status['periodicity_score'], *status['v8_scores']])
                expected_queries = list(range(parent['query'], 48))
                selected_rows = indexed[(parent['name'], replicate, arm)]
                assert [r['query'] for _, r in selected_rows] == expected_queries
                for i, row in selected_rows:
                    q = row['query']
                    close(features['target'][i], d['target_distance'][q, q + 4])
                    close(features['past_displacement'][i], d['target_distance'][q, q - 4])
                    close(features['v82'][i], scores[q])
                    for kind in KINDS:
                        local = d[kind + '_back_path'][q - 13:q + 1, q - 13:q + 1]
                        for method, expected in independent_features(local).items():
                            close(features[kind + '_' + method][i], expected)
                print('Causal features audit %s r%d %s' % (parent['name'], replicate, arm), flush=True)
    return dict(direct_port_distance_pairs=checked_pairs, causal_feature_rows=len(rows), independently_rebuilt_feature_sets=len(rows) * 16)


def audit_graphs(root, config):
    references, regions = read(root / 'references.json'), read(root / 'regions.json')
    distances = {}
    def load(parent, replicate, arm):
        key = (parent, replicate, arm)
        if key not in distances:
            distances[key] = dict(np.load(root / 'distances' / ('%s-r%d-%s.npz' % key)))
        return distances[key]
    refs = {}
    for ref in references:
        matrix = load(ref['parent'], ref['replicate'], 'native10')['delay_' + ref['kind'] + '_' + ref['scope']]
        expected = graph_reference(matrix, ref['anchor'], ref['scale'])
        for key in ('labels', 'region_indices', 'eligible'):
            assert ref[key] == expected[key]
        close(ref['epsilon'], expected['epsilon'])
        close(ref['net_to_path_ratio'], expected['net_to_path_ratio'])
        refs[(ref['parent'], ref['replicate'], ref['kind'], ref['scope'], ref['scale'])] = ref
    for row in regions:
        ref = refs[(row['parent'], row['replicate'], row['kind'], row['scope'], row['scale'])]
        d = load(row['parent'], row['replicate'], row['arm'])['delay_' + row['kind'] + '_' + row['scope']]
        nearest = [min(d[t, j] for j in ref['region_indices']) for t in range(ref['anchor'], len(d))]
        inside = np.asarray(nearest) <= ref['epsilon']
        close(row['nearest_distance'], nearest)
        assert row['inside'] == inside.tolist()
        category = classify(inside) if ref['eligible'] else 'reference_not_established'
        assert row['category'] == category
        assert row['steps'] == list(range((ref['anchor'] + 2) * 10, 520, 10))
    ph_rows = read(root / 'persistence.json')
    parents = {p['name']: (i, p) for i, p in enumerate(config['parents'])}
    for row in ph_rows:
        number, parent = parents[row['parent']]
        raw = load(row['parent'], row['replicate'], row['arm'])[row['kind'] + '_back_path']
        expected = topology_nulls(raw[parent['query']:, parent['query']:], SEED + number * 100 + row['replicate'])
        for key in ('observed', 'nulls', 'descriptive_upper_tail'):
            assert row[key] == expected[key]
    return dict(networkx_references=len(references), independent_membership_classifications=len(regions),
                persistent_homology_rows=len(ph_rows), recomputed_surrogates=len(ph_rows) * 198,
                homology_engine='same pinned GUDHI engine, not a second independent homology algorithm')


def audit_regression(root):
    rows = read(root / 'feature-rows.json')
    features, predictions = dict(np.load(root / 'features.npz')), dict(np.load(root / 'predictions.npz'))
    summary = read(root / 'prediction-summary.json')
    tasks = np.array([r['task'] for r in rows])
    parents = np.array([r['parent'] for r in rows])
    target = features['target']
    fits = read(root / 'prediction-fits.json')
    for fit in fits:
        task, method = fit['task'], fit['method']
        train, test = tasks != task, tasks == task
        assert not set(parents[train]) & set(parents[test])
        counts = Counter(parents[train])
        w = np.array([1 / (len(counts) * counts[p]) for p in parents[train]])
        x = features[method][train]
        mean = np.average(x, axis=0, weights=w)
        scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
        scale[scale <= 1e-12] = 1
        design = np.column_stack((np.ones(len(x)), (x - mean) / scale))
        penalty = np.eye(design.shape[1])[1:]
        augmented = np.vstack((design * np.sqrt(w[:, None]), penalty))
        right = np.r_[target[train] * np.sqrt(w), np.zeros(design.shape[1] - 1)]
        coefficient = np.linalg.lstsq(augmented, right, rcond=None)[0]
        independent = np.column_stack((np.ones(test.sum()), (features[method][test] - mean) / scale)) @ coefficient
        close(predictions[method][test], independent, method, 1e-9)
        close(fit['model']['mean'], mean)
        close(fit['model']['scale'], scale)
        close(fit['model']['coefficient'], coefficient[1:], tolerance=1e-9)
        close(fit['model']['intercept'], coefficient[0], tolerance=1e-9)
        close(predictions['constant'][test], np.dot(w, target[train]))
    close(predictions['persistence'], features['past_displacement'])
    for method, prediction in predictions.items():
        mses, maes = [], []
        for parent in sorted(set(parents)):
            residual = prediction[parents == parent] - target[parents == parent]
            mses.append(float(np.mean(residual ** 2)))
            maes.append(float(np.mean(abs(residual))))
        close(summary['metrics'][method]['rmse'], np.sqrt(np.mean(mses)))
        close(summary['metrics'][method]['mae'], np.mean(maes))
        for task in sorted(set(tasks)):
            subset = set(parents[tasks == task])
            group_mses = [mses[i] for i, p in enumerate(sorted(set(parents))) if p in subset]
            close(summary['metrics'][method]['tasks'][str(task)]['rmse'], np.sqrt(np.mean(group_mses)))
    for gate in summary['gates'].values():
        a, b = (summary['metrics'][gate[key]] for key in ('baseline', 'candidate'))
        reduction = 1 - b['rmse'] / a['rmse']
        improved = sum(b['tasks'][str(t)]['rmse'] < a['tasks'][str(t)]['rmse'] for t in set(tasks))
        close(gate['relative_rmse_reduction'], reduction)
        assert gate['tasks_improved'] == improved and gate['passed'] == (reduction >= .1 and improved >= 2)
    return dict(independent_augmented_least_squares_fits=len(fits), methods=len(predictions),
                predictions_checked=len(rows) * len(predictions), leakage_group='entire base task held out')


def audit_noise_and_conditional(root, config, stats):
    summary = {(r['parent'], r['scope'], r['kind']): r for r in read(root / 'noise-analysis.json')}
    direct = 0
    for parent in config['parents']:
        directory = root / 'controls' / parent['name']
        paths = [directory / ('capture.npz' if o == n == 0 else 'o%d-n%d.npz' % (o, n)) for o in range(3) for n in range(3)]
        prefix = [stats[str(p.relative_to(root))] for p in branch_paths(root, parent, 0, 'native10')[:parent['query'] + 1]]
        with np.load(root / 'noise-distances' / (parent['name'] + '.npz')) as saved:
            for scope in REPRESENTATIONS:
                for field in ('input', 'shared', 'total'):
                    scale = np.maximum(1e-12, np.sqrt(np.mean([np.asarray(s[field][scope]) ** 2 for s in prefix], axis=0)))
                    close(saved['scale_' + field + '_' + scope], scale)
                    for a, b in ((0, 1), (0, 3), (0, 8)):
                        with np.load(paths[a]) as first, np.load(paths[b]) as second:
                            expected = direct_distance(first['mechanism/' + field], second['mechanism/' + field], scope, scale)
                        close(saved[field + '_' + scope][a, b], expected)
                        direct += 1
                close(saved['input_total_' + scope] ** 2, (saved['input_' + scope] ** 2 + saved['total_' + scope] ** 2) / 2)
                for kind in (*KINDS, 'shared'):
                    d = saved[kind + '_' + scope]
                    noise = [d[o * 3 + a, o * 3 + b] for o in range(3) for a in range(3) for b in range(a + 1, 3)]
                    observation = [d[a * 3 + n, b * 3 + n] for n in range(3) for a in range(3) for b in range(a + 1, 3)]
                    row = summary[(parent['name'], scope, kind)]
                    close(row['noise_distances'], noise)
                    close(row['observation_distances'], observation)
                    close(row['noise_mean'], np.mean(noise))
                    close(row['observation_mean'], np.mean(observation))
                    if np.mean(observation) > 1e-12:
                        close(row['noise_observation_ratio'], np.mean(noise) / np.mean(observation))
                    else:
                        assert row['noise_observation_ratio'] is None
    conditional = read(root / 'conditional-analysis.json')
    assert len(conditional) == 12
    for row in conditional:
        assert row['constructed_control'] and row['exact_compute_pairs'] == 2
        for a, b in (('native', 'route_only'), ('joint', 'output_only')):
            directory = P3F / 'states' / row['parent'] / row['phase']
            with np.load(directory / (a + '.npz')) as first, np.load(directory / (b + '.npz')) as second:
                for field in ('input', 'total'):
                    np.testing.assert_array_equal(first['mechanism/' + field], second['mechanism/' + field])
                    assert row['distances'][field][row['labels'].index(a)][row['labels'].index(b)] == 0
            assert row['distances']['route'][row['labels'].index(a)][row['labels'].index(b)] > 0
    return dict(noise_summary_rows=len(summary), direct_grid_distances=direct, constructed_compute_equivalent_pairs=24)


def main(root, streaming=False):
    started = time.monotonic()
    config = check_config(root, protected=True)
    if not streaming:
        assert read(root / 'collection.json')['passed'] and read(root / 'analysis-summary.json')['passed']
    assert not (root / 'failure.json').exists() and not (root / 'analysis-failure.json').exists()
    assert read(root / 'analysis-started.json')['source_sha256'] == sha256_file(BASE / 'analyze_moe_compute_experiment.py')
    source_sha256 = sha256_file(Path(__file__))
    save_json(root / 'audit-started.json', dict(utc=now(), streaming=streaming, source_sha256=source_sha256))
    stats, collection = audit_collection(root, config, streaming)
    assert wait_result(root, 'analysis-summary.json')['passed']
    features = audit_distances_and_features(root, config, stats)
    graph = audit_graphs(root, config)
    regression = audit_regression(root)
    noise = audit_noise_and_conditional(root, config, stats)
    assert sha256_file(Path(__file__)) == source_sha256
    save_json(root / 'audit.json', dict(passed=True, utc=now(), elapsed_seconds=time.monotonic() - started,
              collection=collection, features=features, graph=graph, regression=regression, noise=noise,
              scope='HB MoE routes, input, shared, total; no physical or action target', new_environment_actions=0,
              source_sha256=source_sha256))
    print('MoE compute audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stream', action='store_true')
    args = parser.parse_args()
    main(args.run.resolve(), args.stream)
