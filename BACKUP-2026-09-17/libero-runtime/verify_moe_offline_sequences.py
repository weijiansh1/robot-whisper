"""Independent arithmetic, matching and source checks for the offline study."""

from itertools import permutations
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from moe_offline_sequences import ROOT, METHODS, archive, digest, save_json


def main():
    records = json.loads((ROOT / 'source-hashes.json').read_text())
    for name, record in records.items():
        assert digest(name) == record['sha256'], name
    source_code = json.loads((ROOT / 'implementation-hashes.json').read_text())
    for name, sha in source_code.items():
        assert digest(name) == sha, name
    assert digest(ROOT / 'PROTOCOL.md') == json.loads((ROOT / 'started.json').read_text())['protocol_sha256']
    frame = pd.read_csv(ROOT / 'episode-index.csv', low_memory=False)
    coordinates = archive(ROOT / 'coordinates.npz')['coordinates']
    scores = archive(ROOT / 'sequence-scores.npz')['scores']
    assert len(frame) == 33043 and frame.length.sum() == 525314
    assert not frame.identity.duplicated().any()
    expected = (np.arange(52)[None, :] >= 12) & (np.arange(52)[None, :] < frame.length.to_numpy()[:, None])
    np.testing.assert_array_equal(np.isfinite(scores).all(-1), expected)
    first = pd.read_csv(ROOT / 'episode-events.csv', low_memory=False)
    for k, name in enumerate(METHODS[:5]):
        hit = np.isfinite(scores[..., k]) & (scores[..., k] >= np.log(1.2))
        direct = np.where(hit.any(1), hit.argmax(1), -1)
        np.testing.assert_array_equal(direct, first['first_'+name])
    # Explicitly enumerate all 720 orders, independently of the 20-partition code.
    errors = []
    selection = np.argwhere(expected)
    for i, q in selection[np.linspace(0, len(selection)-1, 40).astype(int)]:
        x = coordinates[i, q-5:q+1]
        values = []
        for order in permutations(range(6)):
            p = x[list(order)]
            values.append(min(p[:3, 1].min(), -p[3:, 0].max()))
        direct = min(x[:3, 1].min(), -x[3:, 0].max())
        errors.append(abs((direct-np.mean(values))-scores[i, q, 5]))
    assert max(errors) < 1e-12
    strata = pd.read_csv(ROOT / 'same-initial-query-strata.csv')
    checked = 0
    for row in strata.query('cohort == "external_8b" and query == 13').itertuples():
        use = frame[(frame.cohort == row.cohort) & (frame.task == row.task) &
                    (frame.init_state_id == row.init_state_id) & (frame.length > row.query)]
        k = METHODS.index(row.method)
        bad = scores[use.index[use.failure], row.query, k]
        good = scores[use.index[~use.failure], row.query, k]
        value = mannwhitneyu(bad, good, alternative='two-sided').statistic/(len(bad)*len(good))
        assert abs(value-row.auc) < 1e-12
        checked += 1
    for filename, keys in [('same-initial-query', ['cohort', 'query', 'method']),
                           ('external-matched', ['contrast', 'method'])]:
        detail = pd.read_csv(ROOT / (filename+'-strata.csv'))
        summary = pd.read_csv(ROOT / (filename+'-summary.csv'))
        task_values = detail.groupby(keys+['task']).auc.mean().groupby(keys).mean()
        for row in summary.to_dict('records'):
            assert abs(task_values.loc[tuple(row[k] for k in keys)]-row['mean']) < 1e-12
    physical = pd.read_csv(ROOT / 'external-query-windows.csv')
    assert physical.name.nunique() == 80
    assert len(physical.query('observed_queries == 3')) == 2719
    for row in pd.read_csv(ROOT / 'external-matched-strata.csv').itertuples():
        selected = physical[(physical.benchmark == row.benchmark) & (physical.task == row.task) &
                            (physical['query'] == row.query) & (physical.phase == row.phase) &
                            (physical.observed_queries == 3)]
        if row.contrast == 'static_vs_progress_proxy':
            high = selected[selected.category.isin(['holding_or_static', 'return_motion_without_subgoal_gain'])]
            low = selected[selected.category.isin(['completed', 'subgoal_gain', 'geometric_approach_proxy'])]
        else:
            high = selected[selected.category.isin(['holding_or_static', 'return_motion_without_subgoal_gain',
                                                   'motion_without_verified_subgoal_gain'])]
            low = selected[selected.category.isin(['completed', 'subgoal_gain'])]
        assert len(high)*len(low) == row.pairs
        direct = mannwhitneyu(high[row.method], low[row.method]).statistic/row.pairs
        assert abs(direct-row.auc) < 1e-12
    branches = pd.read_csv(ROOT / 'branch-index.csv')
    b_scores = archive(ROOT / 'branch-sequence-scores.npz')
    errors_native = []
    for row in branches.query('arm == "native"').itertuples():
        index = frame.index[frame.identity.eq('current|'+row.parent)][0]
        delta = np.nanmax(np.abs(b_scores[row.identity]-scores[index, :row.length]))
        errors_native.append(float(delta))
    assert max(errors_native) < 2e-4
    assert len(branches) == 60 and branches.parent.nunique() == 5
    assert not (branches.success & ~branches.source_success).any()
    tests = subprocess.run(['/data/miniconda/envs/torch/bin/python', '-m', 'unittest', '-v',
                            'test_moe_offline_sequences'], cwd='/data/libero-runtime',
                           capture_output=True, text=True, check=True)
    (ROOT / 'tests.txt').write_text(tests.stdout+tests.stderr)
    save_json(ROOT / 'verification.json', dict(passed=True, source_files=len(records),
        implementation_files=len(source_code), explicit_permutation_windows=len(errors),
        max_permutation_error=max(errors), mann_whitney_outcome_checks=checked,
        physical_matched_checks=len(pd.read_csv(ROOT / 'external-matched-strata.csv')),
        native_branches=len(errors_native), max_native_score_error=max(errors_native),
        full_observed_physical_windows=2719, unit_tests=5, old_inputs_unchanged=True))
    print((ROOT / 'verification.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
