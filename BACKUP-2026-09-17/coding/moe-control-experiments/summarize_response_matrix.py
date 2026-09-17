"""Report the planned dose contrast and FP16/FP32 screening sensitivity."""

import argparse
from pathlib import Path

import numpy as np

from audit_online_experiment import digest, read, write
from audit_response_matrix import TARGETS, passes


def summarize(root):
    config = read(root / 'config.json')
    assert read(root / 'audit.json')['passed']
    rows = read(root / 'collection.json')['rows']
    baselines = {r['parent']: r for r in rows if r['kind'] == 'native'}
    probes = [r for r in rows if r['kind'] == 'probe']
    keyed = {(r['parent'], r['spec']['label']): r for r in probes}
    results = []
    for operator in config['operators']:
        group = [r for r in probes if r['spec']['label'] == operator + '-high-plus']
        fp_score, fp_instant, dose_changes = [], [], []
        for row in group:
            baseline = baselines[row['parent']]
            instant = (np.asarray(row['fp32_measurement']['instantaneous']) -
                       baseline['fp32_measurement']['instantaneous']) / row['fp32_measurement']['margins']
            fp_score.append(passes(row['fp32_delta'], operator, row['normalized_action_rms'], row['gripper_sign_changes']))
            fp_instant.append(passes(instant, operator, row['normalized_action_rms'], row['gripper_sign_changes']))
            low = keyed[(row['parent'], operator + '-low-plus')]
            dose_changes.append(np.asarray(row['delta']) - low['delta'])
        dose = np.stack(dose_changes)
        results.append(dict(operator=operator, parents=5,
            fp16_score_screen=sum(r['score_screen'] for r in group), fp32_score_screen=sum(fp_score),
            fp16_instant_screen=sum(r['instantaneous_screen'] for r in group), fp32_instant_screen=sum(fp_instant),
            mean_high_minus_low=dose.mean(0).tolist(),
            higher_dose_improves_all_targets_by_point05=int(np.all(dose[:, TARGETS[operator]] <= -.05, axis=1).sum())))
    result = dict(results=results, source_collection_sha256=digest(root / 'collection.json'),
                  implementation_sha256=digest(Path(__file__)),
                  note='Sensitivity of the predeclared development screens, not a new selection rule')
    write(root / 'dose-precision-summary.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    summarize(parser.parse_args().run.resolve())
