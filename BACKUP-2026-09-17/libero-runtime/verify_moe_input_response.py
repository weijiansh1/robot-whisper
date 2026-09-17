"""Verify archived controls and response artifacts, then seal this analysis."""

from datetime import datetime, timezone
from importlib.metadata import version
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from moe_response_analysis import ROOT, OLD, SOURCE, digest, save_json

BASE = Path(__file__).resolve().parent
EXPANDED = BASE / 'samples/moe-joint-expanded-20260915'
CONDITIONS = ('o0n0', 'o0n1', 'o1n0', 'o1n1')
HADAMARD = np.asarray([[1, 1, 1, 1], [-1, -1, 1, 1],
                      [-1, 1, -1, 1], [1, -1, -1, 1]], float) / 4


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-python', type=Path, default=Path('/data/venv311/bin/python'))
    args = parser.parse_args()
    result = {'passed': False}
    cache, sources = {}, {}

    def check_hash(path, expected):
        path = Path(path).resolve()
        if str(path) not in cache:
            cache[str(path)] = digest(path)
        if cache[str(path)] != expected:
            raise ValueError(f'hash mismatch: {path}')
        sources[str(path)] = expected

    previous_counts = {}
    for directory in (OLD, EXPANDED):
        manifest = json.loads((directory / 'artifact-hashes.json').read_text())
        for name, expected in manifest.items():
            check_hash(directory / name, expected)
        previous_counts[directory.name] = len(manifest)
    prior_sources = json.loads((OLD / 'source-hashes.json').read_text())
    for path, expected in prior_sources.items():
        check_hash(path, expected)
    print(f'Previous artifacts and {len(prior_sources)} source files verified', flush=True)

    selection = json.loads((ROOT / 'selection.json').read_text())
    probe = ROOT / 'controlled-probe'
    run = json.loads((probe / 'results.json').read_text())
    checks = json.loads((probe / 'checks.json').read_text())
    calls = json.loads((probe / 'calls.json').read_text())
    assert run['passed'] and run['model_forwards'] == len(calls) == len(checks) == 82
    assert len(selection['parents']) == run['parents'] == 16
    assert len(selection['pairs']) == run['task_pairs'] == 8
    assert sum(pair['stable_matched'] for pair in selection['pairs']) == 4
    assert sum(call['captured'] for call in calls) == run['captured_forwards'] == 66
    assert run['new_environment_actions'] == 0 and not run['shared_service_used']
    assert not selection['selection_uses_new_functional_response']
    assert len({p['name'] for p in selection['parents']}) == 16
    assert len({(call['parent'], call['condition']) for call in calls}) == 82
    check_hash(ROOT / 'selection.json', run['selection_sha256'])
    check_hash(OLD / 'episode-features.npz', selection['source_features_sha256'])
    for check in checks:
        assert check['finite_actions']
        assert all(value for key, value in check.items() if key.endswith('_exact'))

    energy_table = pd.read_csv(ROOT / 'factorial-energy.csv')
    assert len(energy_table) == 352
    summary = json.loads((ROOT / 'controlled-results.json').read_text())
    native_controls = repeats = natural_corners = captured = independent_energies = 0
    max_reconstruction_error = 0.
    for parent in selection['parents']:
        directory = probe / parent['name']
        trace_path = Path(parent['source_dir']) / 'episode-trace.npz'
        route_path = SOURCE / parent['name'] / 'full-hb-routes.npz'
        check_hash(trace_path, parent['trace_sha256'])
        check_hash(route_path, parent['source_hb_sha256'])
        with np.load(trace_path) as trace:
            original_actions = trace['predicted_actions']
            noises = trace['flow_noises']
        with np.load(route_path) as trace:
            original_hb, original_ids = trace['hb_router_probs'], trace['hb_expert_ids']
        snapshots = {name: read_npz(directory / (name + '.npz')) for name in CONDITIONS}
        for call in [c for c in calls if c['parent'] == parent['name']]:
            check = next(c for c in checks if c['parent'] == call['parent'] and c['condition'] == call['condition'])
            assert check['captured'] == call['captured']
            assert call['flow_noise_sha256'] == hashlib.sha256(noises[check['noise_query']].tobytes()).hexdigest()
            snapshot = read_npz(directory / (call['condition'] + '.npz'))
            assert snapshot['actions'].shape == (10, 7) and snapshot['hb'].shape == (8, 10, 11, 32)
            assert all(np.isfinite(value).all() for value in snapshot.values())
            assert (snapshot['hb'] >= 0).all() and (snapshot['hb'].sum(-1) > 0).all()
            if call['captured']:
                captured += 1
                hidden = snapshot['input'].shape[-1]
                for key in ('input', 'routed', 'shared', 'total'):
                    assert snapshot[key].shape == (8, 11, hidden)
                assert snapshot['expert_output'].shape == (8, 11, 4, hidden)
                assert snapshot['ids'].shape == snapshot['weights'].shape == (8, 11, 4)
                assert np.array_equal(snapshot['hb_layers'], [2, 3, 4, 5, 12, 13, 14, 15])
                assert ((snapshot['ids'] >= 0) & (snapshot['ids'] < 32)).all()
                assert (np.diff(np.sort(snapshot['ids'], axis=-1), axis=-1) > 0).all()
                assert (snapshot['weights'] > 0).all()
                reconstructed = (snapshot['expert_output'] * snapshot['weights'][..., None]).sum(-2)
                error = np.linalg.norm(reconstructed-snapshot['routed'], axis=-1) / np.maximum(
                    np.linalg.norm(snapshot['routed'], axis=-1), 1e-12)
                assert error.max() < 1e-5
                max_reconstruction_error = max(max_reconstruction_error, float(error.max()))
                assert np.array_equal(snapshot['routed'] + snapshot['shared'], snapshot['total'])
            if call['condition'] in ('o0n0', 'o1n1'):
                natural_corners += 1
                query = parent['query'] - (call['condition'] == 'o0n0')
                assert np.array_equal(snapshot['actions'], original_actions[query])
                assert np.array_equal(snapshot['hb'], original_hb[query])
                assert np.array_equal(snapshot['ids'], original_ids[query, :, -1])
        native = read_npz(directory / 'native_current.npz')
        for key in native:
            assert np.array_equal(native[key], snapshots['o1n1'][key])
        native_controls += 1
        if parent['parent_id'] in (0, 15):
            repeat = read_npz(directory / 'repeat_current.npz')
            assert repeat.keys() == snapshots['o1n1'].keys()
            assert all(np.array_equal(repeat[key], snapshots['o1n1'][key]) for key in repeat)
            repeats += 1

        for field in ('input', 'routed', 'shared', 'total', 'router_sqrt', 'action_translation', 'action_rotation'):
            values = []
            for condition in CONDITIONS:
                s = snapshots[condition]
                if field == 'router_sqrt':
                    p = s['hb'][:, -1].astype(float)
                    value = np.sqrt(p / p.sum(-1, keepdims=True)) / np.sqrt(2)
                elif field.startswith('action_'):
                    columns = slice(0, 3) if field == 'action_translation' else slice(3, 6)
                    value = s['actions'][:, columns].astype(float)
                else:
                    value = s[field].astype(float)
                values.append(value)
            contrasts = np.tensordot(HADAMARD, np.stack(values), axes=(1, 0))
            energy = (contrasts[1:] ** 2).sum(-1)
            rows = energy_table[energy_table.parent_id.eq(parent['parent_id']) & energy_table.field.eq(field)]
            for row in rows.itertuples():
                if row.scope == 'action_chunk':
                    summed = energy.sum(-1)
                else:
                    depth, kind = row.scope.split('_')
                    layers = slice(0, 4) if depth == 'front' else slice(4, 8)
                    tokens = slice(0, 1) if kind == 'state' else slice(1, 11)
                    summed = energy[:, layers, tokens].sum(axis=(1, 2))
                expected = summed / summed.sum()
                observed = [row.observation_energy_fraction, row.noise_energy_fraction, row.interaction_energy_fraction]
                np.testing.assert_allclose(observed, expected, rtol=1e-8, atol=1e-12)
                independent_energies += 1
    assert captured == 66 and native_controls == 16 and repeats == 2 and natural_corners == 32
    assert independent_energies == 352
    coverage = next(row for row in summary['observation_comparison_coverage']
                    if not row['stable_matches_only'] and row['scope'] == 'back_action')
    assert coverage['same_support_sites'] == 52 and coverage['sites'] == coverage['valid_response_gain_sites'] == 1280
    assert coverage['near_zero_input_change_sites'] == 0
    print('Native controls, full tensors, natural corners and 352 independent energy calculations verified', flush=True)

    completed = subprocess.run([str(args.model_python), str(BASE / 'test_moe_response.py')], cwd=BASE,
                               text=True, capture_output=True, timeout=120)
    log = completed.stdout + completed.stderr
    (ROOT / 'tests.txt').write_text(log)
    assert completed.returncode == 0, log
    tests = int(re.search(r'Ran (\d+) tests?', log).group(1))
    figures = {}
    for path in sorted(ROOT.glob('*.png')):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
        nonwhite = float((pixels < 245).any(-1).mean())
        assert nonwhite > .015 and pixels.std() > 10
        assert path.with_suffix('.pdf').stat().st_size > 1000
        figures[path.name] = dict(width=pixels.shape[1], height=pixels.shape[0], nonwhite_fraction=nonwhite)
    assert len(figures) == 4 and (ROOT / 'REPORT.zh.md').exists()

    implementation = [BASE / name for name in (
        'moe_response_analysis.py', 'moe_final_response_recorder.py', 'probe_moe_input_response.py',
        'analyze_moe_controlled_response.py', 'plot_moe_input_response.py', 'test_moe_response.py',
        'verify_moe_input_response.py')]
    dependencies = [Path(path) for path in (
        '/data/coding/moe-control-experiments/gate_runtime.py',
        '/data/coding/moe-control-experiments/gate_capture.py',
        '/data/coding/robot-whisper-0909/himoe-route-capture/himoe_state_recorder.py',
        '/data/srv/src/moevla/models/moevla.py',
        '/data/srv/src/moevla/models/modeling_moe.py',
        '/data/srv/src/moevla/models/himoe.py',
        '/data/srv/src/moevla/models/paligemma_with_expert.py')]
    for path in dependencies:
        sources[str(path)] = digest(path)
    save_json(ROOT / 'implementation-hashes.json', {str(path): digest(path) for path in implementation})
    save_json(ROOT / 'source-hashes.json', sources)
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                          '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout.strip()
    model_packages = subprocess.run(
        [str(args.model_python), '-c', 'import json; from importlib.metadata import version; '
         'print(json.dumps({n: version(n) for n in ("numpy", "pandas", "scipy", "torch", "transformers")}))'],
        capture_output=True, text=True, check=True)
    result.update(passed=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
                  parents=16, task_pairs=8, strictly_matched_pairs=4, model_forwards=82,
                  captured_forwards=captured, natural_corners_rechecked=natural_corners,
                  native_capture_controls=native_controls, full_tensor_repeat_controls=repeats,
                  maximum_routed_reconstruction_relative_error=max_reconstruction_error,
                  independent_energy_calculations=independent_energies,
                  prior_artifacts_unchanged=previous_counts, prior_sources_unchanged=len(prior_sources),
                  unit_tests=tests, unit_test_exit_code=completed.returncode, unit_test_log='tests.txt',
                  model_python=str(args.model_python), model_packages=json.loads(model_packages.stdout),
                  analysis_python=sys.executable,
                  figures=figures, gpu_after_probe=gpu, new_environment_actions=0,
                  packages={name: version(name) for name in ('numpy', 'pandas', 'scipy', 'torch', 'matplotlib')})
    save_json(ROOT / 'verification.json', result)
    save_json(ROOT / 'artifact-hashes.json', {str(path.relative_to(ROOT)): digest(path)
                                           for path in sorted(ROOT.rglob('*'))
                                           if path.is_file() and path.name != 'artifact-hashes.json'})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
