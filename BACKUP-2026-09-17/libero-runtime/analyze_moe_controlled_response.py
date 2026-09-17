"""Analyze coherent observation/noise probes without interpreting outcomes as traps."""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from moe_response_analysis import (ROOT, functional_comparison, summarized_comparison,
                                   factorial_energy, scope_masks, save_json)


def read_snapshot(path):
    with np.load(path) as z:
        result = {key: z[key].astype(float) for key in z.files}
    result['expert_contrib'] = result.pop('expert_output')
    result['ids'] = result['ids'].astype(int)
    return result


def main():
    selection = json.loads((ROOT / 'selection.json').read_text())
    run = ROOT / 'controlled-probe'
    integrity = json.loads((run / 'results.json').read_text())
    if not integrity['passed']:
        raise ValueError('incomplete or invalid model probe')
    comparisons, energies, actions = [], [], []
    for parent in selection['parents']:
        directory = run / parent['name']
        snapshots = {name: read_snapshot(directory / (name + '.npz'))
                     for name in ('o0n0', 'o0n1', 'o1n0', 'o1n1')}
        metadata = {key: parent[key] for key in ('parent_id', 'pair_id', 'task', 'name', 'query', 'success',
                                               'stable_matched', 'mobility_log_ratio', 'eef_next_displacement_m',
                                               'eef_previous_displacement_m')}
        conditions = [('observation_noise0', 'o0n0', 'o1n0'), ('observation_noise1', 'o0n1', 'o1n1'),
                      ('noise_observation0', 'o0n0', 'o0n1'), ('noise_observation1', 'o1n0', 'o1n1'),
                      ('natural_diagonal', 'o0n0', 'o1n1')]
        for label, left, right in conditions:
            first, second = snapshots[left], snapshots[right]
            values = functional_comparison(first, second, raw_experts=True)
            p0, p1 = first['hb'][:, -1], second['hb'][:, -1]
            p0 = p0 / p0.sum(-1, keepdims=True)
            p1 = p1 / p1.sum(-1, keepdims=True)
            values['router_hellinger'] = np.linalg.norm(np.sqrt(p0)-np.sqrt(p1), axis=-1) / np.sqrt(2)
            comparisons.extend(summarized_comparison(values, dict(**metadata, comparison=label)))
            a0, a1 = first['actions'], second['actions']
            actions.append(dict(**metadata, comparison=label,
                                translation_action_rms=float(np.sqrt(np.mean((a1[:, :3]-a0[:, :3])**2))),
                                rotation_action_rms=float(np.sqrt(np.mean((a1[:, 3:6]-a0[:, 3:6])**2))),
                                gripper_sign_disagreement=float(np.mean(np.sign(a1[:, 6]) != np.sign(a0[:, 6])))))
        for field in ('input', 'routed', 'shared', 'total', 'router_sqrt'):
            arrays = []
            for name in ('o0n0', 'o0n1', 'o1n0', 'o1n1'):
                if field == 'router_sqrt':
                    p = snapshots[name]['hb'][:, -1]
                    p = p / p.sum(-1, keepdims=True)
                    arrays.append(np.sqrt(p) / np.sqrt(2))
                else:
                    arrays.append(snapshots[name][field])
            decomposition = factorial_energy(*arrays)
            for scope, mask in scope_masks(decomposition['total_energy'].shape).items():
                sums = decomposition['energy'][mask].sum(axis=0)
                fractions = sums / sums.sum() if sums.sum() > 1e-20 else np.full(3, np.nan)
                energies.append(dict(**metadata, field=field, scope=scope,
                                     observation_energy_fraction=fractions[0], noise_energy_fraction=fractions[1],
                                     interaction_energy_fraction=fractions[2]))
        for field, columns in (('action_translation', slice(0, 3)), ('action_rotation', slice(3, 6))):
            arrays = [snapshots[name]['actions'][:, columns] for name in ('o0n0', 'o0n1', 'o1n0', 'o1n1')]
            decomposition = factorial_energy(*arrays)
            sums = decomposition['energy'].sum(axis=0)
            fractions = sums / sums.sum() if sums.sum() > 1e-20 else np.full(3, np.nan)
            energies.append(dict(**metadata, field=field, scope='action_chunk',
                                 observation_energy_fraction=fractions[0], noise_energy_fraction=fractions[1],
                                 interaction_energy_fraction=fractions[2]))
    comparison_table = pd.DataFrame(comparisons)
    comparison_table.to_csv(ROOT / 'controlled-comparisons.csv', index=False)
    pd.DataFrame(energies).to_csv(ROOT / 'factorial-energy.csv', index=False)
    pd.DataFrame(actions).to_csv(ROOT / 'action-responses.csv', index=False)
    observation = comparison_table[comparison_table.comparison.str.startswith('observation_noise')]
    parent = observation.groupby(['parent_id', 'pair_id', 'task', 'name', 'success', 'stable_matched', 'scope'],
                                 as_index=False).mean(numeric_only=True)
    parent.to_csv(ROOT / 'controlled-parent-responses.csv', index=False)
    paired = []
    measurements = ('input_relative_change', 'routed_relative_change', 'shared_relative_change',
                    'total_relative_change', 'routed_relative_gain', 'shared_relative_gain', 'total_relative_gain',
                    'router_hellinger', 'expert_term_norm_fraction', 'shared_change_norm_fraction')
    for (pair_id, scope), group in parent.groupby(['pair_id', 'scope']):
        failure, success = group[~group.success].iloc[0], group[group.success].iloc[0]
        for key in measurements:
            paired.append(dict(pair_id=pair_id, task=int(failure.task), scope=scope,
                               stable_matched=bool(failure.stable_matched), measurement=key,
                               failure_value=failure[key], success_value=success[key],
                               difference=failure[key]-success[key]))
    pd.DataFrame(paired).to_csv(ROOT / 'task-paired-responses.csv', index=False)
    summaries = []
    for matched_only in (False, True):
        subset = parent[((not matched_only) | parent.stable_matched) & parent.scope.eq('back_action')]
        for success, group in subset.groupby('success'):
            summaries.append(dict(stable_matches_only=matched_only, success=success, parents=len(group),
                                  measurement_parent_counts={key: int(group[key].notna().sum()) for key in measurements},
                                  **{key: float(group[key].dropna().median()) if group[key].notna().any() else np.nan
                                     for key in measurements}))
    coverage = []
    for matched_only in (False, True):
        subset = observation[(not matched_only) | observation.stable_matched]
        for scope, group in subset.groupby('scope'):
            coverage.append(dict(stable_matches_only=matched_only, scope=scope,
                                 parents=group.parent_id.nunique(), comparisons=len(group),
                                 sites=int(group.sites.sum()), same_support_sites=int(group.same_support_sites.sum()),
                                 near_zero_input_change_sites=int(group.near_zero_input_change_sites.sum()),
                                 valid_response_gain_sites=int(group.valid_response_gain_sites.sum())))
    rho = []
    for scope, group in parent.groupby('scope'):
        for key in ('routed_relative_gain', 'total_relative_gain', 'router_hellinger'):
            good = group[key].notna()
            rho.append(dict(scope=scope, measurement=key, parents=int(good.sum()),
                            spearman_with_next_eef_motion=float(spearmanr(group.loc[good, key],
                                                                        group.loc[good, 'eef_next_displacement_m']).statistic)))
    calls = pd.DataFrame(json.loads((run / 'calls.json').read_text()))
    durations = calls.pivot(index='parent_id', columns='condition', values='seconds')
    overhead = durations.o1n1-durations.native_current
    energies = pd.DataFrame(energies)
    energy_summary = []
    for (field, scope), group in energies.groupby(['field', 'scope']):
        energy_summary.append(dict(field=field, scope=scope, parents=len(group),
                                   median_observation_fraction=group.observation_energy_fraction.median(),
                                   median_noise_fraction=group.noise_energy_fraction.median(),
                                   median_interaction_fraction=group.interaction_energy_fraction.median()))
    save_json(ROOT / 'controlled-results.json', dict(
        model_probe=integrity, parent_summaries=summaries, energy_summaries=energy_summary,
        observation_comparison_coverage=coverage,
        descriptive_motion_correlations=rho,
        capture_overhead_seconds=dict(median=overhead.median(), q25=overhead.quantile(.25), q75=overhead.quantile(.75),
                                      median_excluding_first=overhead.iloc[1:].median()),
        no_trap_ground_truth=True, no_new_classifier=True,
        energy_interpretation='Finite balanced 2x2 contrasts of actual adjacent observations and noises; not population causal variance.'))
    print((ROOT / 'controlled-results.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
