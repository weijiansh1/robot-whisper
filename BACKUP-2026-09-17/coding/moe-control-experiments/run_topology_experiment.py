"""Frozen CPU-only topology experiments on previously collected failures."""

import argparse
from collections import Counter
import csv
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
from scipy.spatial.distance import pdist, squareform

from audit_replan_window import read, records
from gate_runtime import BASE
from run_online_experiment import EVALUATION, now, save_json
from himoe_libero_bridge.episode_trace import sha256_file
from collection_routes import PROBS_KEY
from topology_protocol import (DEPS, HISTORY, PRIMARY, REFERENCE, REPRESENTATIONS, SCALES, SEED, SURROGATES,
                               delayed_distance, fit_reference, gudhi_module, persistence, region_diagnostic,
                               route_distance, synthetic_circle, topology_nulls)

P3H = BASE / 'runs/p3h-replan-window-20260915-65syu8x3'
P3F = BASE / 'runs/p3f-crossover-v2-20260915-ihpn5n13'
SOURCES = ('P3I_TOPOLOGY_PLAN.zh.md', 'topology_protocol.py', 'test_topology_protocol.py', 'run_topology_experiment.py')
LABELS = ('native', 'route_only', 'joint', 'output_only')


def check_hashes(files):
    for path, digest in files.items():
        assert sha256_file(Path(path)) == digest, path


def prepare():
    h_config, h_seal = read(P3H / 'config.json'), read(P3H / 'verification.json')
    h_audit = read(P3H / 'failure-only-audit.json')
    assert h_seal['passed'] and h_audit['passed'] and h_audit['actual_completed_rollouts'] == 48
    parents = [p for p in h_config['parents'] if not p['source_success']]
    cases = [c for c in read(P3F / 'config.json')['cases'] if c['parent'] in {p['name'] for p in parents}]
    assert len(parents) == 6 and len(cases) == 12 and len({c['parent'] for c in cases}) == 4
    protected = {}
    for seal in sorted((BASE / 'runs').glob('*/verification.json')):
        previous = read(seal)
        protected[str(seal)] = sha256_file(seal)
        protected.update({str(seal.parent / name): sha for name, sha in
                          previous.get('files', previous.get('file_sha256')).items()})
    check_hashes(protected)
    old_sources = dict(h_config['source_hashes'], **h_seal['post_collection_sources'])
    check_hashes(old_sources)
    inputs = {str(EVALUATION / p['name'] / 'full-hb-routes.npz'): p['hb_sha256'] for p in parents}
    check_hashes(inputs)
    libraries = {str(path): sha256_file(path) for path in DEPS.rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts}
    assert libraries and gudhi_module().__version__ == '3.13.0'
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(BASE), '-p', 'test_*.py', '-v'],
                           text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert tests.returncode == 0, tests.stdout
    root = Path(tempfile.mkdtemp(prefix='p3i-topology-20260915-', dir=BASE / 'runs'))
    config = dict(schema='local.moe_topology_failures.v1', utc=now(), parents=parents, crossover_cases=cases,
                  p3h=str(P3H), p3f=str(P3F), archived_rollouts=48, paired_units=24, conditional_cases=12,
                  shared_metadata=h_config['shared_metadata'],
                  source_hashes=dict(old_sources, **{str(BASE / name): sha256_file(BASE / name) for name in SOURCES}),
                  protected_files=protected, input_hashes=inputs, dependency_hashes=libraries,
                  versions=dict(numpy=metadata.version('numpy'), scipy=metadata.version('scipy'), gudhi='3.13.0'),
                  representations=REPRESENTATIONS, scales=SCALES, primary=PRIMARY, history=HISTORY,
                  reference=REFERENCE, surrogate_count=SURROGATES, seed=SEED,
                  surrogate_order='same seed within paired arms; recompute delays after raw time reordering',
                  model_inference_calls=0, new_environment_actions=0, new_success_parent_rollouts=0,
                  frozen_v82_changed=False, new_controller_test=False,
                  missing='No continuous effective MoE input/output in P3h; P3f is conditional, not a rollout')
    save_json(root / 'config.json', config)
    with (root / 'config.sha256').open('x') as stream:
        stream.write(sha256_file(root / 'config.json') + '\n')
    with (root / 'pre-analysis-tests.txt').open('x') as stream:
        stream.write(tests.stdout)
    print(str(root), flush=True)


def check_config(root, protected=False):
    config = read(root / 'config.json')
    assert sha256_file(root / 'config.json') == (root / 'config.sha256').read_text().strip()
    for name in ('source_hashes', 'input_hashes', 'dependency_hashes'):
        check_hashes(config[name])
    if protected:
        check_hashes(config['protected_files'])
    assert config['versions'] == dict(numpy=metadata.version('numpy'), scipy=metadata.version('scipy'), gudhi=gudhi_module().__version__)
    return config


def load_branch(parent, replicate, arm, original):
    directory = P3H / parent['name'] / ('r%d' % replicate) / arm
    result = read(directory / 'result.json')
    assert not result['source_success'] and not result['success'] and result['action_steps'] == 520
    rows = [r for r in records(directory / 'queries.jsonl') if r['step'] % 10 == 0]
    assert [r['step'] for r in rows] == list(range(parent['start'], 520, 10))
    chunks = [p for p in original[:parent['query']]]
    for row in rows:
        with np.load(P3H / row['path']) as saved:
            assert not any(key.startswith('mechanism/') for key in saved.files)
            chunks.append(saved[PROBS_KEY])
    probability = np.stack(chunks)
    assert probability.shape == (52, 8, 10, 11, 32)
    return probability


def write_csv(path, rows):
    with path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def crossover(root, config):
    output = []
    for number, case in enumerate(config['crossover_cases']):
        directory = P3F / 'states' / case['parent'] / case['phase']
        raw = {}
        for label in LABELS:
            with np.load(directory / (label + '.npz')) as saved:
                raw[label] = {key: saved[key] for key in
                              ('v8_control/effective_probs_fp32', 'mechanism/input', 'mechanism/total', 'actions')}
        equality = []
        for first, second in (('native', 'route_only'), ('joint', 'output_only')):
            for field in ('mechanism/input', 'mechanism/total', 'actions'):
                assert np.array_equal(raw[first][field], raw[second][field]), (case, field)
            equality.append([first, second])
        routing = route_distance(np.stack([raw[label]['v8_control/effective_probs_fp32'].astype(np.float16)
                                           for label in LABELS]), 'back_path')
        features = []
        for key in ('mechanism/input', 'mechanism/total'):
            values = np.stack([raw[label][key] for label in LABELS]).astype(float)
            normalizer = max(1e-12, float(np.sqrt(np.mean(values[0] ** 2))))
            features.append(values.reshape(4, -1) / (normalizer * np.sqrt(2 * values[0].size)))
        computational = squareform(pdist(np.concatenate(features, axis=1)))
        assert computational[0, 1] == computational[2, 3] == 0
        route_ph, effective_ph = persistence(routing), persistence(computational)
        row = dict(parent=case['parent'], phase=case['phase'], query=case['query'], labels=LABELS,
                   routing_distance=routing.tolist(), computational_distance=computational.tolist(),
                   route_h0_positive_merges=route_ph['h0_finite'], computation_h0_positive_merges=effective_ph['h0_finite'],
                   routing_distinct_points=1 + len(route_ph['h0_finite']),
                   computation_distinct_points=1 + len(effective_ph['h0_finite']),
                   exact_computation_pairs=equality,
                   same_computation_nonzero_route_pairs=int(routing[0, 1] > 0) + int(routing[2, 3] > 0),
                   interpretation='conditional neighborhood negative control, not a physical-time loop')
        output.append(row)
        print('Crossover %d/12: %s %s' % (number + 1, case['parent'], case['phase']), flush=True)
    save_json(root / 'crossover.json', output)
    return output


def synthetic_experiments(root):
    cases = dict(static=np.zeros((52, 2)), cycle=synthetic_circle(), drift=np.c_[np.arange(52), np.zeros(52)])
    for name, end in (('exit_return', 32), ('exit_nonreturn', 52)):
        x = synthetic_circle()
        x[21:end, 0] += 10
        cases[name] = x
    output = []
    expected = dict(static='no_confirmed_exit', cycle='no_confirmed_exit', drift='reference_not_established',
                    exit_return='exit_then_return', exit_nonreturn='last_exit_no_observed_return')
    for i, (name, points) in enumerate(cases.items()):
        raw = squareform(pdist(points))
        delayed = delayed_distance(raw)
        ref = fit_reference(delayed, 16)
        diagnostic = region_diagnostic(delayed, 16, ref)
        assert diagnostic['category'] == expected[name]
        topological = topology_nulls(raw[18:, 18:], SEED + 10000 + i)
        output.append(dict(name=name, reference=ref, diagnostic=diagnostic, topology=topological))
    save_json(root / 'synthetic.json', output)
    return output


def analyze(root):
    config = check_config(root)
    started = time.monotonic()
    save_json(root / 'started.json', dict(utc=now(), mode='CPU-only archived-data analysis'))
    try:
        synthetic_experiments(root)
        (root / 'distances').mkdir()
        graph_rows, topology_rows, references = [], [], []
        for parent_number, parent in enumerate(config['parents']):
            with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as archive:
                original = archive['hb_router_probs']
            for replicate in range(4):
                distances = {}
                for arm in ('native10', 'window5'):
                    probability = load_branch(parent, replicate, arm, original)
                    payload = {}
                    for representation in REPRESENTATIONS:
                        raw = route_distance(probability, representation)
                        delayed = delayed_distance(raw)
                        payload['raw_' + representation], payload['delay_' + representation] = raw, delayed
                        distances[(arm, representation)] = delayed
                    path = root / 'distances' / ('%s-r%d-%s.npz' % (parent['name'], replicate, arm))
                    with path.open('xb') as stream:
                        np.savez_compressed(stream, **payload)
                    raw_suffix = payload['raw_back_path'][parent['query']:, parent['query']:]
                    topological = topology_nulls(raw_suffix, SEED + 100 * parent_number + replicate)
                    topology_rows.append(dict(parent=parent['name'], replicate=replicate, arm=arm,
                                              raw_queries=len(raw_suffix), embedding_points=len(raw_suffix) - 2, **topological))
                anchor = parent['query'] - HISTORY + 1
                for representation in REPRESENTATIONS:
                    native = distances[('native10', representation)]
                    short = distances[('window5', representation)]
                    assert np.array_equal(native[:anchor + 1, :anchor + 1], short[:anchor + 1, :anchor + 1])
                    for scale in SCALES:
                        reference = fit_reference(native, anchor, scale)
                        assert fit_reference(short, anchor, scale) == reference
                        references.append(dict(parent=parent['name'], replicate=replicate, representation=representation,
                                               scale=scale, anchor=anchor, **reference))
                        for arm in ('native10', 'window5'):
                            result = region_diagnostic(distances[(arm, representation)], anchor, reference)
                            graph_rows.append(dict(parent=parent['name'], replicate=replicate, arm=arm,
                                                   representation=representation, scale=scale, reference_eligible=reference['eligible'],
                                                   epsilon=reference['epsilon'], endpoint_success=False, **result))
                print('Paired routes %s r%d complete' % (parent['name'], replicate), flush=True)
        save_json(root / 'references.json', references)
        save_json(root / 'regions.json', graph_rows)
        save_json(root / 'persistence.json', topology_rows)
        negative_controls = crossover(root, config)
        primary = [r for r in graph_rows if (r['representation'], r['scale']) == PRIMARY]
        assert len(primary) == len(topology_rows) == 48 and len(graph_rows) == 432 and len(references) == 216
        summary = dict(passed=True, utc=now(), elapsed_seconds=time.monotonic() - started,
                       archived_rollouts=48, paired_units=24, original_failure_states=6,
                       model_inference_calls=0, new_environment_actions=0, new_controller_test=False,
                       primary_counts={arm: dict(Counter(r['category'] for r in primary if r['arm'] == arm))
                                       for arm in ('native10', 'window5')},
                       primary_reference_eligible_pairs=sum(r['reference_eligible'] for r in primary if r['arm'] == 'native10'),
                       robust_category_all_nine={arm: sum(len({r['category'] for r in graph_rows if r['parent'] == p['name'] and
                           r['replicate'] == replicate and r['arm'] == arm}) == 1 for p in config['parents'] for replicate in range(4))
                           for arm in ('native10', 'window5')},
                       crossover_cases=len(negative_controls), exact_computation_pairs=2 * len(negative_controls),
                       same_computation_nonzero_route_pairs=sum(r['same_computation_nonzero_route_pairs'] for r in negative_controls),
                       continuous_computation_topology_available=False,
                       observed_successes=0, source_success_rate_is_not_a_new_experiment=True)
        save_json(root / 'summary.json', summary)
        columns = ('parent', 'replicate', 'arm', 'representation', 'scale', 'reference_eligible', 'epsilon', 'category',
                   'first_exit_step', 'first_exit_confirmed_step', 'observed_return', 'outside_fraction', 'endpoint_success')
        write_csv(root / 'region-summary.csv', [{key: r[key] for key in columns} for r in graph_rows])
        print(summary, flush=True)
    except BaseException as error:
        save_json(root / 'failure.json', dict(error=str(error), traceback=traceback.format_exc(),
                                             elapsed_seconds=time.monotonic() - started, model_inference_calls=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'analyze'))
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else analyze(args.run.resolve())
