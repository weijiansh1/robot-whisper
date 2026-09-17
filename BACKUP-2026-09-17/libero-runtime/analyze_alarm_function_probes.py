"""Before/alarm functional changes, retaining individual paired parents."""

import json

import numpy as np
import pandas as pd

from alarm_trajectory_study import ROOT
from analyze_moe_controlled_response import read_snapshot
from moe_response_analysis import (functional_comparison, summarized_comparison, factorial_energy,
                                   scope_masks, save_json)


def main():
    selection = json.loads((ROOT / 'functional-selection.json').read_text())
    integrity = json.loads((ROOT / 'functional/results.json').read_text())
    assert integrity['passed']
    rows, energies, actions = [], [], []
    for probe in selection['probes']:
        directory = ROOT / 'functional' / ('probe-%02d' % probe['probe_id'])
        snapshots = {name: read_snapshot(directory / (name+'.npz')) for name in ('o0n0', 'o0n1', 'o1n0', 'o1n1')}
        metadata = {key: probe[key] for key in ('parent_id', 'pair_id', 'task', 'name', 'success', 'phase_matched', 'stage', 'probe_id')}
        metadata['query'] = probe['probe_query']
        metadata['v82_on'] = 0 <= probe['v82_first'] <= probe['probe_query']
        metadata['knn_on'] = 0 <= probe['knn_first'] <= probe['probe_query']
        for label, left, right in (('obs_noise0', 'o0n0', 'o1n0'), ('obs_noise1', 'o0n1', 'o1n1'),
                                   ('noise_obs0', 'o0n0', 'o0n1'), ('noise_obs1', 'o1n0', 'o1n1')):
            first, second = snapshots[left], snapshots[right]
            values = functional_comparison(first, second, raw_experts=True)
            p0, p1 = first['hb'][:, -1], second['hb'][:, -1]
            p0, p1 = p0/p0.sum(-1, keepdims=True), p1/p1.sum(-1, keepdims=True)
            values['router_hellinger'] = np.linalg.norm(np.sqrt(p1)-np.sqrt(p0), axis=-1)/np.sqrt(2)
            rows.extend(summarized_comparison(values, dict(**metadata, comparison=label)))
            delta = second['actions']-first['actions']
            actions.append(dict(**metadata, comparison=label, translation_rms=float(np.sqrt((delta[:, :3]**2).mean())),
                                rotation_rms=float(np.sqrt((delta[:, 3:6]**2).mean()))))
        for field in ('input', 'routed', 'shared', 'total', 'router'):
            values = []
            for snap in snapshots.values():
                if field == 'router':
                    p = snap['hb'][:, -1]
                    values.append(np.sqrt(p/p.sum(-1, keepdims=True))/np.sqrt(2))
                else:
                    values.append(snap[field])
            result = factorial_energy(*values)
            for scope, mask in scope_masks(result['total_energy'].shape).items():
                sums = result['energy'][mask].sum(0)
                fractions = sums/sums.sum() if sums.sum() else np.full(3, np.nan)
                energies.append(dict(**metadata, field=field, scope=scope, observation_fraction=fractions[0],
                                     noise_fraction=fractions[1], interaction_fraction=fractions[2]))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / 'functional-contrasts.csv', index=False)
    pd.DataFrame(energies).to_csv(ROOT / 'functional-energy.csv', index=False)
    pd.DataFrame(actions).to_csv(ROOT / 'functional-action-contrasts.csv', index=False)
    columns = ['input_relative_change', 'routed_relative_gain', 'shared_relative_gain', 'total_relative_gain',
               'routed_relative_change', 'total_relative_change', 'router_hellinger', 'expert_term_norm_fraction']
    parent = table[table.comparison.str.startswith('obs_')].groupby(
        ['parent_id', 'pair_id', 'task', 'name', 'success', 'phase_matched', 'stage', 'scope'], as_index=False)[columns].mean()
    parent.to_csv(ROOT / 'functional-parent-responses.csv', index=False)
    changes = []
    for (parent_id, scope), part in parent.groupby(['parent_id', 'scope']):
        before = part[part.stage.eq('before')].iloc[0]
        anchor = part[part.stage.eq('anchor')].iloc[0]
        for key in columns:
            changes.append(dict(parent_id=parent_id, pair_id=int(anchor.pair_id), task=int(anchor.task),
                                 success=bool(anchor.success), phase_matched=bool(anchor.phase_matched),
                                 scope=scope, measurement=key, before=before[key], anchor=anchor[key],
                                 change=anchor[key]-before[key]))
    change_table = pd.DataFrame(changes)
    change_table.to_csv(ROOT / 'functional-before-anchor.csv', index=False)
    paired = []
    for (pair_id, scope, measurement), part in change_table.groupby(['pair_id', 'scope', 'measurement']):
        failure, success = part[~part.success].iloc[0], part[part.success].iloc[0]
        paired.append(dict(pair_id=pair_id, scope=scope, measurement=measurement,
                           phase_matched=bool(failure.phase_matched), failure_change=failure.change,
                           success_change=success.change, difference_of_changes=failure.change-success.change,
                           failure_anchor=failure.anchor, success_anchor=success.anchor))
    paired = pd.DataFrame(paired)
    paired.to_csv(ROOT / 'functional-paired-changes.csv', index=False)
    summary = parent.groupby(['scope', 'stage', 'success'])[columns].median().reset_index()
    summary.to_csv(ROOT / 'functional-summary.csv', index=False)
    paired_summary = []
    for matched_only in (False, True):
        subset = paired[paired.phase_matched] if matched_only else paired
        for (scope, measurement), part in subset.groupby(['scope', 'measurement']):
            paired_summary.append(dict(matched_only=matched_only, scope=scope, measurement=measurement,
                pairs=len(part), negative_changes=int((part.difference_of_changes < 0).sum()),
                median_difference_of_changes=part.difference_of_changes.median()))
    calls = pd.DataFrame(json.loads((ROOT / 'functional/calls.json').read_text()))
    times = calls.pivot(index='probe_id', columns='condition', values='seconds')
    overhead = times.o1n1-times.native_current
    save_json(ROOT / 'functional-results.json', dict(model_experiment=integrity, summaries=summary.to_dict('records'),
              paired_summaries=paired_summary, no_local_trap_ground_truth=True,
              added_capture_seconds_median=overhead.median(), added_capture_seconds_q25=overhead.quantile(.25),
              added_capture_seconds_q75=overhead.quantile(.75)))
    print(summary[summary.scope.eq('back_action')].to_string(index=False), flush=True)
    print(pd.DataFrame(paired_summary).query('scope == "back_action"').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
