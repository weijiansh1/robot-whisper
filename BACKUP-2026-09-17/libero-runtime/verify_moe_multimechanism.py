"""Independent checks for labels, matching calipers, joins, and source hashes."""

import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from moe_multimechanism import ROOT, SIGNALS, HEADS
from moe_response_analysis import digest, save_json


def main():
    tests = unittest.defaultTestLoader.loadTestsFromName('test_moe_multimechanism')
    result = unittest.TextTestRunner(verbosity=1).run(tests)
    assert result.wasSuccessful()
    inputs = json.loads((ROOT / 'input-hashes.json').read_text())
    for path, record in inputs.items():
        assert digest(path) == record['sha256'], path
    parents = pd.read_csv(ROOT / 'parents.csv')
    queries = pd.read_csv(ROOT / 'queries.csv')
    windows = pd.read_csv(ROOT / 'physical-windows.csv')
    assert len(parents) == 80 and parents.name.nunique() == 80
    assert not queries.duplicated(['name', 'query']).any()
    assert not windows.duplicated(['name', 'query', 'horizon']).any()
    physical = {}
    for parent in parents.itertuples(index=False):
        with np.load(ROOT / 'physical' / (parent.name+'.npz')) as data:
            physical[parent.name] = {k: data[k] for k in data.files}
        data = physical[parent.name]
        mat = data['rotations']
        np.testing.assert_allclose(mat.swapaxes(-1, -2) @ mat, np.broadcast_to(np.eye(3), mat.shape), atol=1e-9)
        np.testing.assert_allclose(np.linalg.det(mat), 1, atol=1e-9)
        part = queries[queries.name.eq(parent.name)]
        np.testing.assert_array_equal(part['query'], np.arange(parent.length))
        expected = np.zeros(len(part), bool)
        for head in HEADS:
            q = getattr(parent, 'first_'+head)
            on = (q >= 0) & (part['query'].to_numpy() >= q)
            np.testing.assert_array_equal(part['on_'+head], on)
            expected |= on
        np.testing.assert_array_equal(part.v82_on, expected)
    for row in windows.itertuples(index=False):
        data = physical[row.name]
        start, end = data['predicates'][row.query], data['predicates'][row.end_query]
        any_gain = bool((data['predicates'][row.query+1:row.end_query+1] & ~start).any())
        end_gain, end_loss = bool((end & ~start).any()), bool((~end & start).any())
        label = ('verified_progress' if end_gain and not end_loss else
                 'no_observed_gain' if not any_gain and not end_loss else 'transient_or_regressing')
        assert row.local_label == label
        assert row.complete_window == (row.end_query-row.query == row.horizon)
    context = windows.set_index(['name', 'query', 'horizon'])
    checked_pairs = 0
    for population in ('native', 'functional'):
        pairs = pd.read_csv(ROOT / (population+'-matched-pairs.csv'))
        for pair in pairs.itertuples(index=False):
            a = context.loc[(pair.no_gain_name, pair.no_gain_query, pair.horizon)]
            b = context.loc[(pair.progress_name, pair.progress_query, pair.horizon)]
            assert pair.no_gain_name != pair.progress_name
            assert a.local_label == 'no_observed_gain' and b.local_label == 'verified_progress'
            assert a.complete_window and b.complete_window and a.all5_available and b.all5_available
            assert a.benchmark == b.benchmark and a.base_task == b.base_task and a.phase == b.phase
            assert abs(pair.no_gain_query-pair.progress_query) <= 2
            delta = max(abs(a['margin_'+s]-b['margin_'+s]) for s in SIGNALS)
            np.testing.assert_allclose(delta, pair.baseline_max_margin, atol=1e-9)
            if pair.tier != 'phase':
                da, db = physical[pair.no_gain_name], physical[pair.progress_name]
                pa = np.concatenate([da['positions'][pair.no_gain_query], da['eef'][pair.no_gain_query][None]])
                pb = np.concatenate([db['positions'][pair.progress_query], db['eef'][pair.progress_query][None]])
                d = np.sqrt(((pa-pb)**2).sum(-1))
                np.testing.assert_allclose(np.sqrt((d*d).mean()), pair.geometry_rms_m, atol=1e-12)
                assert pair.geometry_rms_m <= .05+1e-12 and pair.geometry_max_m <= .1+1e-12
                assert pair.rotation_max_deg <= 30+1e-9 and pair.joint_max_range <= .1+1e-12
            if 'margin_' in pair.tier:
                assert delta <= float(pair.tier.rsplit('_', 1)[1])+1e-9
            if pair.tier.startswith('same_variant'):
                assert a.task_suite == b.task_suite and a.variant_task_id == b.variant_task_id
            checked_pairs += 1
    audit = pd.read_csv(ROOT / 'functional-algebra-audit.csv')
    assert audit.decomposition_max_absolute.max() < 1e-8
    assert audit.attribution_max_absolute.max() < 1e-8
    assert audit.stored_update_rounding_relative.max() < 1e-5
    attributions = pd.read_csv(ROOT / 'expert-attribution.csv')
    assert not attributions.duplicated(['name', 'query', 'noise', 'layer', 'token', 'expert']).any()
    from PIL import Image
    for name in ('branch-and-matching', 'selected-contribution-features'):
        with Image.open(ROOT / (name+'.png')) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert min(pixels.shape[:2]) > 500 and pixels.std() > 10
        pdf = (ROOT / (name+'.pdf')).read_bytes()
        assert pdf.startswith(b'%PDF-') and b'%%EOF' in pdf[-100:]
    code_paths = ['restore_multimechanism_physics.py', 'moe_multimechanism.py', 'run_moe_multimechanism.py',
                  'test_moe_multimechanism.py', 'verify_moe_multimechanism.py', 'plot_moe_multimechanism.py']
    code_hashes = {str(ROOT.parent.parent / name): digest(ROOT.parent.parent / name) for name in code_paths}
    artifacts = {str(path.relative_to(ROOT)): digest(path) for path in sorted(ROOT.rglob('*'))
                 if path.is_file() and path.name not in ('verification.json', 'artifact-hashes.json')}
    save_json(ROOT / 'verification.json', dict(passed=True, tests=result.testsRun, source_hashes=len(inputs),
        native_parents=len(parents), boundary_states=sum(len(d['predicates']) for d in physical.values()),
        window_labels_checked=len(windows), selected_pairs_checked=checked_pairs,
        functional_contrasts=len(audit), attribution_rows=len(attributions),
        figure_files_checked=4, code_hashes=code_hashes, artifacts=artifacts))
    save_json(ROOT / 'artifact-hashes.json', dict(artifacts, **{'verification.json': digest(ROOT / 'verification.json')}))
    print('Verified %d inputs, %d windows, %d selected pairs, and %d functional contrasts.' %
          (len(inputs), len(windows), checked_pairs, len(audit)))


if __name__ == '__main__':
    main()
