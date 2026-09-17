"""Descriptive checks that distinguish relative scaling from lost perturbations."""

import argparse
from pathlib import Path

import numpy as np

from audit_online_experiment import read, write
from audit_response_matrix import load_arrays


def main(root):
    config, summary = read(root / 'config.json'), read(root / 'summary.json')
    rows = []
    for case in config['cases']:
        directory = (root / case['path']).parent
        native, combo = load_arrays(directory / 'native.npz'), load_arrays(directory / 'combo.npz')
        for field in ('input', 'shared', 'residual'):
            np.testing.assert_array_equal(native['mechanism/' + field][4, 0], combo['mechanism/' + field][4, 0])
        delta_total = combo['mechanism/total'][4, 0, 1:].astype(float) - native['mechanism/total'][4, 0, 1:]
        delta_block = combo['mechanism/block'][4, 0, 1:].astype(float) - native['mechanism/block'][4, 0, 1:]
        rows.append(dict(parent=case['parent'], phase=case['phase'],
            first_L12_delta_total_l2=float(np.linalg.norm(delta_total)),
            first_L12_delta_block_l2=float(np.linalg.norm(delta_block)),
            absolute_delta_retention=float(np.linalg.norm(delta_block) / np.linalg.norm(delta_total)),
            max_absolute_addition_rounding=float(np.max(np.abs(delta_block - delta_total)))))
    scales = [layer for state in summary['native_scales'] for layer in state['by_layer'] if layer['layer'] >= 12]
    freeze = load_arrays(root / 'physics/pro-task03-init039/native.npz')
    result = dict(first_local_pairs=rows,
        absolute_delta_retention_range=[min(r['absolute_delta_retention'] for r in rows), max(r['absolute_delta_retention'] for r in rows)],
        mean_shared_over_routed_rms=float(np.mean([r['shared_rms'] / r['routed_rms'] for r in scales])),
        mean_residual_over_total_rms=float(np.mean([r['residual_rms'] / r['total_rms'] for r in scales])),
        freeze_before_alarm=dict(window_steps=[120, 170],
            eef_max_displacement_from_step120_mm=float(np.linalg.norm(freeze['eef_position'][120:171] - freeze['eef_position'][120], axis=1).max() * 1000)),
        note='Relative tensor norms have different denominators; fixed residual addition preserves absolute perturbations up to rounding.')
    write(root / 'mechanism-details.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
