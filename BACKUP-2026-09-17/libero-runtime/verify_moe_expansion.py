"""Validate expanded artifacts and unchanged prior results without model inference."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analyze_moe_expansion import ROOT, OLD, RAW, NAMES, digest, first_alarms, save_json, auc


def main():
    tests = subprocess.run([sys.executable, '-m', 'unittest', '-v', 'test_moe_expansion'],
                           cwd=Path(__file__).parent, capture_output=True, text=True)
    (ROOT / 'tests.txt').write_text(tests.stdout + tests.stderr)
    tests.check_returncode()
    index = pd.read_csv(ROOT / 'episode-events.csv', low_memory=False)
    coordinates = np.load(ROOT / 'two-axis-coordinates.npz')['coordinates']
    archive = np.load(ROOT / 'supported-motif-scores.npz')
    scores = archive['scores']
    if len(index) != 33023 or index.identity.duplicated().any():
        raise ValueError('unexpected or duplicate episode count')
    valid = np.arange(52)[None] < index.length.to_numpy()[:, None]
    if np.isfinite(coordinates[~valid]).any() or np.isfinite(scores[~valid]).any():
        raise ValueError('padding leaked into scores')
    if np.isfinite(coordinates[:, :7]).any() or np.isfinite(scores[:, :9]).any():
        raise ValueError('warmup leakage')
    first = first_alarms(scores, 1.2)
    for k, name in enumerate(NAMES):
        np.testing.assert_array_equal(first[:, k], index['first_' + name])
    old_hashes = json.loads((OLD / 'artifact-hashes.json').read_text())
    for name, expected in old_hashes.items():
        if digest(OLD / name) != expected:
            raise ValueError(f'previous artifact changed: {name}')
    source_hashes = json.loads((OLD / 'source-hashes.json').read_text())
    for name, expected in source_hashes.items():
        if digest(name) != expected:
            raise ValueError(f'previous source changed: {name}')
    manifests = json.loads((ROOT / 'input-manifest.json').read_text())
    for row in manifests['records']:
        if digest(ROOT / 'inputs' / row['source']) != row['sha256']:
            raise ValueError('restored input changed')
    envelope = []
    lower = first_alarms(scores, np.exp(np.log(1.2) - 1e-4)) >= 0
    upper = first_alarms(scores, np.exp(np.log(1.2) + 1e-4)) >= 0
    for (cohort, failure), group in index.groupby(['cohort', 'failure']):
        ids = group.index.to_numpy()
        for k, name in enumerate(NAMES):
            envelope.append(dict(cohort=cohort, failure=failure, method=name, episodes=len(ids),
                                 threshold_looser_1e4_hits=int(lower[ids, k].sum()),
                                 primary_hits=int((first[ids, k] >= 0).sum()),
                                 threshold_tighter_1e4_hits=int(upper[ids, k].sum())))
    pd.DataFrame(envelope).to_csv(ROOT / 'numeric-threshold-envelope.csv', index=False)
    checks = 0
    for (_, _), group in index.groupby(['cohort', 'task']):
        ids = group.index.to_numpy()
        y = group.failure.to_numpy(bool)
        for q in (9, 12, 18, 26):
            values = scores[ids, q, 1]
            good = np.isfinite(values)
            yy, xx = y[good], values[good]
            if yy.any() and (~yy).any():
                independent = mannwhitneyu(xx[yy], xx[~yy]).statistic / (yy.sum() * (~yy).sum())
                if abs(independent - auc(yy, xx)) > 1e-12:
                    raise ValueError('rank statistic mismatch')
                checks += 1
    coverage = []
    for (cohort, failure), group in index.groupby(['cohort', 'failure']):
        coverage.append(dict(cohort=cohort, failure=failure, episodes=len(group),
                             queries=int(group.length.sum()), query7_available=int(group.length.gt(7).sum()),
                             query9_available=int(group.length.gt(9).sum()),
                             query18_available=int(group.length.gt(18).sum())))
    pd.DataFrame(coverage).to_csv(ROOT / 'observation-coverage.csv', index=False)
    unused = pd.read_csv(ROOT / 'inputs/safe&vlaconf/moe_trainfree/results/routing_dynamics_20260908/index.csv')
    keys = set(zip(index.run_id, index.task, index.episode))
    unused = unused[[key not in keys for key in zip(unused.run_id, unused.task, unused.episode)]]
    unused.to_csv(ROOT / 'metadata-only-excluded.csv', index=False)
    status = subprocess.check_output(['git', 'status', '--porcelain'],
                                     cwd='/data/coding/robot-whisper-0909', text=True)
    save_json(ROOT / 'verification.json', dict(
        passed=True, episodes=len(index), queries=int(index.length.sum()),
        failures=int(index.failure.sum()), successes=int((~index.failure).sum()),
        independent_rank_checks=checks, warmup_and_padding_verified=True,
        prior_artifacts_unchanged=len(old_hashes), prior_sources_unchanged=len(source_hashes),
        restored_hashes_verified=len(manifests['records']),
        unit_test_exit_code=tests.returncode, unit_test_log='tests.txt', metadata_only_excluded=len(unused),
        git_worktree_status=status, no_new_model_inference=True,
        numeric_envelope_note='Sensitivity calculation, not a proven corpus-wide rounding bound.'))
    print((ROOT / 'verification.json').read_text())


if __name__ == '__main__':
    main()
