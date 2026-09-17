"""Recheck archived input distances, temporal graphs, labels and null ranks."""

import argparse
from collections import Counter
from itertools import groupby
from pathlib import Path

import networkx as nx
import numpy as np

from audit_replan_window import read
from run_online_experiment import EVALUATION, now, save_json
from run_topology_experiment import P3F, check_config, load_branch
from topology_protocol import gudhi_module


def direct_distance(probability, representation):
    slices = dict(back_path=(slice(4, None), slice(None)), full_path=(slice(None), slice(None)),
                  back_last=(slice(4, None), slice(-1, None)))
    layer, flow = slices[representation]
    p = probability[:, layer, flow, 1:].astype(np.float64)
    p /= p.sum(-1, keepdims=True)
    roots = np.sqrt(p)
    answer = np.empty((len(p), len(p)))
    for index, point in enumerate(roots):
        answer[index] = np.sqrt(.5 * np.square(roots - point).sum(-1).mean(axis=(1, 2, 3)))
    return answer


def embedding(distance):
    return np.sqrt((distance[:-2, :-2] ** 2 + distance[1:-1, 1:-1] ** 2 + distance[2:, 2:] ** 2) / 3)


def ph_score(distance):
    tree = gudhi_module().RipsComplex(distance_matrix=distance).create_simplex_tree(max_dimension=2)
    tree.compute_persistence(homology_coeff_field=2)
    diagram = tree.persistence_intervals_in_dimension(1)
    lifetime = float(max(diagram[:, 1] - diagram[:, 0], default=0.))
    diameter = float(distance.max())
    return 0. if diameter <= 1e-12 else lifetime / diameter


def runs_of(mask, value):
    intervals, position = [], 0
    for key, group in groupby(mask):
        length = len(list(group))
        if key == value:
            intervals.append((position, position + length - 1))
        position += length
    return intervals


def category(mask, eligible):
    if not eligible:
        return 'reference_not_established'
    outside = [(a, b) for a, b in runs_of(mask, False) if b - a >= 2]
    inside = [(a, b) for a, b in runs_of(mask, True) if b - a >= 1]
    returned = any(a > end for _, end in outside for a, _ in inside)
    if not outside:
        return 'no_confirmed_exit'
    if outside[-1][1] == len(mask) - 1 and outside[-1][1] - outside[-1][0] >= 3:
        return 'last_exit_no_observed_return'
    if all(mask[-2:]) and returned:
        return 'exit_then_return'
    return 'insufficient_followup_or_mixed'


def audit_reference(distance, reference):
    anchor = reference['anchor']
    indices = np.asarray(reference['indices'])
    assert np.array_equal(indices, np.arange(anchor - 11, anchor + 1))
    local = distance[np.ix_(indices, indices)]
    nearest = [min(local[i, j] for j in range(12) if abs(i - j) >= 3) for i in range(12)]
    epsilon = max(1e-12, float(np.quantile(nearest, .95)) * reference['scale'])
    np.testing.assert_allclose(reference['epsilon'], epsilon, rtol=1e-12, atol=1e-15)
    labels = np.asarray(reference['labels'])
    clusters = [np.flatnonzero(labels == label) for label in range(int(labels.max()) + 1)]
    for c in clusters:
        assert local[np.ix_(c, c)].max() <= epsilon + 1e-14
    for i, first in enumerate(clusters):
        for second in clusters[i + 1:]:
            assert local[np.ix_(first, second)].max() > epsilon - 1e-14
    counts = np.zeros((len(clusters), len(clusters)), int)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(clusters)))
    for first, second in zip(labels[:-1], labels[1:]):
        counts[first, second] += 1
        graph.add_edge(int(first), int(second))
    assert np.array_equal(counts, reference['transition_counts'])
    component = next(c for c in nx.strongly_connected_components(graph) if int(labels[-1]) in c)
    selected = np.asarray([i for i in range(12) if int(labels[i]) in component])
    assert reference['region_indices'] == indices[selected].tolist()
    path = sum(local[first, second] for first, second in zip(selected[:-1], selected[1:]))
    ratio = 0. if path <= 1e-12 else float(local[selected[0], selected[-1]] / path)
    cyclic = len(component) > 1 or graph.has_edge(int(labels[-1]), int(labels[-1]))
    expected = len(selected) >= 4 and selected[-1] - selected[0] >= 3 and cyclic and ratio <= .35
    assert reference['eligible'] == bool(expected)
    np.testing.assert_allclose(reference['net_to_path_ratio'], ratio, atol=1e-14)


def audit(root):
    config = check_config(root, protected=True)
    assert not (root / 'failure.json').exists()
    references, regions, persistence = (read(root / name) for name in ('references.json', 'regions.json', 'persistence.json'))
    summary = read(root / 'summary.json')
    assert summary['passed'] and len(regions) == 432 and len(references) == 216 and len(persistence) == 48
    ref_index = {(r['parent'], r['replicate'], r['representation'], r['scale']): r for r in references}
    row_index = {(r['parent'], r['replicate'], r['arm'], r['representation'], r['scale']): r for r in regions}
    ph_index = {(r['parent'], r['replicate'], r['arm']): r for r in persistence}
    assert len(ref_index) == 216 and len(row_index) == 432 and len(ph_index) == 48
    comparisons, diagrams = 0, 0
    for parent_number, parent in enumerate(config['parents']):
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as saved:
            original = saved['hb_router_probs']
        for replicate in range(4):
            prefix_reference = {}
            for arm in ('native10', 'window5'):
                probability = load_branch(parent, replicate, arm, original)
                cache_path = root / 'distances' / ('%s-r%d-%s.npz' % (parent['name'], replicate, arm))
                with np.load(cache_path) as cached:
                    assert len(cached.files) == 6
                    for representation in config['representations']:
                        raw = direct_distance(probability, representation)
                        delayed = embedding(raw)
                        np.testing.assert_allclose(cached['raw_' + representation], raw, atol=1e-13, rtol=1e-12)
                        np.testing.assert_allclose(cached['delay_' + representation], delayed, atol=1e-13, rtol=1e-12)
                        anchor = parent['query'] - 2
                        if arm == 'native10':
                            prefix_reference[representation] = delayed[:anchor + 1, :anchor + 1].copy()
                        else:
                            assert np.array_equal(prefix_reference[representation], delayed[:anchor + 1, :anchor + 1])
                        for scale in config['scales']:
                            key = (parent['name'], replicate, representation, scale)
                            reference = ref_index[key]
                            audit_reference(delayed, reference)
                            row = row_index[(parent['name'], replicate, arm, representation, scale)]
                            distances = cached['delay_' + representation][anchor:, reference['region_indices']].min(1)
                            mask = (distances <= reference['epsilon']).tolist()
                            assert mask == row['inside'] and mask[0]
                            np.testing.assert_allclose(distances, row['nearest_distance'], atol=1e-15)
                            assert row['steps'] == list(range(parent['start'], 520, 10))
                            assert row['category'] == category(mask, reference['eligible'])
                            assert row['reference_eligible'] == reference['eligible'] and not row['endpoint_success']
                            np.testing.assert_allclose(row['outside_fraction'], np.mean(np.logical_not(mask)))
                            if reference['eligible']:
                                exits = [(a, b) for a, b in runs_of(mask, False) if b - a >= 2]
                                expected_intervals = [[row['steps'][a], row['steps'][b]] for a, b in exits]
                                assert expected_intervals == row['exit_intervals']
                                assert row['first_exit_step'] == (row['steps'][exits[0][0]] if exits else None)
                                assert row['first_exit_confirmed_step'] == (row['steps'][exits[0][0] + 2] if exits else None)
                            comparisons += 1
                    raw = cached['raw_back_path'][parent['query']:, parent['query']:]
                    ph = ph_index[(parent['name'], replicate, arm)]
                    observed = ph_score(embedding(raw))
                    np.testing.assert_allclose(observed, ph['observed']['normalized_h1'], atol=1e-14)
                    rng = np.random.default_rng(config['seed'] + 100 * parent_number + replicate)
                    for i in range(config['surrogate_count']):
                        for mode in ('shuffle', 'block3'):
                            if mode == 'shuffle':
                                order = rng.permutation(len(raw))
                            else:
                                blocks = [np.arange(j, min(j + 3, len(raw))) for j in range(0, len(raw), 3)]
                                order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
                            value = ph_score(embedding(raw[np.ix_(order, order)]))
                            np.testing.assert_allclose(value, ph['nulls'][mode][i], atol=1e-14)
                            diagrams += 1
                    for mode in ('shuffle', 'block3'):
                        expected_tail = (1 + sum(v >= observed for v in ph['nulls'][mode])) / 100
                        assert expected_tail == ph['descriptive_upper_tail'][mode]
                print('Audited %s r%d %s' % (parent['name'], replicate, arm), flush=True)

    controls = read(root / 'crossover.json')
    assert len(controls) == 12
    for row, case in zip(controls, config['crossover_cases']):
        assert row['parent'] == case['parent'] and row['phase'] == case['phase']
        raw = {}
        for label in row['labels']:
            with np.load(P3F / 'states' / case['parent'] / case['phase'] / (label + '.npz')) as saved:
                raw[label] = {key: saved[key] for key in
                              ('mechanism/input', 'mechanism/total', 'actions', 'v8_control/effective_probs_fp32')}
        route = direct_distance(np.stack([raw[label]['v8_control/effective_probs_fp32'].astype(np.float16)
                                          for label in row['labels']]), 'back_path')
        np.testing.assert_allclose(route, row['routing_distance'], atol=1e-13)
        for i, first in enumerate(row['labels']):
            for j, second in enumerate(row['labels']):
                squares = []
                for field in ('mechanism/input', 'mechanism/total'):
                    reference_rms = max(1e-12, float(np.sqrt(np.mean(raw['native'][field].astype(float) ** 2))))
                    delta = raw[first][field].astype(float) - raw[second][field]
                    squares.append(float(np.mean(delta ** 2)) / reference_rms ** 2)
                np.testing.assert_allclose(row['computational_distance'][i][j], np.sqrt(np.mean(squares)), atol=1e-12)
        for first, second in row['exact_computation_pairs']:
            for field in ('mechanism/input', 'mechanism/total', 'actions'):
                assert np.array_equal(raw[first][field], raw[second][field])
        for field, count in (('routing_distance', 'routing_distinct_points'), ('computational_distance', 'computation_distinct_points')):
            graph = nx.from_numpy_array(np.asarray(row[field]) == 0)
            assert nx.number_connected_components(graph) == row[count]
    primary = [r for r in regions if (r['representation'], r['scale']) == tuple(config['primary'])]
    assert summary['primary_counts'] == {arm: dict(Counter(r['category'] for r in primary if r['arm'] == arm))
                                        for arm in ('native10', 'window5')}
    assert summary['same_computation_nonzero_route_pairs'] == sum(r['same_computation_nonzero_route_pairs'] for r in controls)
    save_json(root / 'audit.json', dict(passed=True, utc=now(), branch_setting_checks=comparisons,
                                      surrogate_diagrams_recomputed=diagrams, conditional_cases=12,
                                      preserved_files=len(config['protected_files']), model_inference_calls=0,
                                      independence='direct Hellinger/RMS formulas, networkx SCC, separate interval logic; persistence reuses validated GUDHI engine'))
    print('Independent audit passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run.resolve())
