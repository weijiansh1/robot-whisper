"""Explain kNN/v8.2 disagreements with actual neighbors and fixed prefix controls."""

import json

import numpy as np
import pandas as pd

from v82_knn_research import (ROOT, GROUPS, archive, load_frozen, combine_first, alarm_groups,
                              alarm_metrics, common_prefix_pairs, auc, save_json)


def summarize_pairs(frame, methods, cohort):
    pairs = common_prefix_pairs(frame, methods)
    pairs.to_csv(ROOT / (cohort + '-common-prefix-pairs.csv'), index=False)
    rows = []
    for method in methods:
        grouped = pairs.groupby('task')[[method + '_failure_hit', method + '_success_hit']].mean()
        delta = grouped.iloc[:, 0] - grouped.iloc[:, 1]
        rng = np.random.default_rng(20260915)
        boot = delta.to_numpy()[rng.integers(0, len(delta), (2000, len(delta)))].mean(-1)
        rows.append(dict(cohort=cohort, method=method, pairs=len(pairs), tasks=len(grouped),
                         failure_hits=int(pairs[method + '_failure_hit'].sum()),
                         success_hits=int(pairs[method + '_success_hit'].sum()),
                         mean_task_failure_minus_success=delta.mean(),
                         task_bootstrap_q025=np.quantile(boot, .025), task_bootstrap_q975=np.quantile(boot, .975)))
    return rows


def main():
    parameters = json.loads((ROOT / 'available-profile/parameters.json').read_text())
    threshold = parameters['threshold']
    profile = archive(ROOT / 'available-profile/reference.npz')
    original = load_frozen()
    bank_frame = original.iloc[profile['success_global_rows']]
    metrics, groups, points, common, ranking = [], [], [], [], []
    for cohort, file_stem in (('B', 'available-B'), ('current', 'current')):
        frame = pd.read_csv(ROOT / (file_stem + '-decisions.csv'))
        data = archive(ROOT / (file_stem + '-geometry.npz'))
        methods = combine_first(frame.v82_frozen, frame.knn_available_first)
        frame['joint_group'] = alarm_groups(methods['v82'], methods['knn'])
        subsets = [(cohort, frame)]
        if cohort == 'B':
            subsets += [('B_supported_tasks', frame[frame.task_available_in_reference]),
                        ('B_missing_reference_tasks', frame[~frame.task_available_in_reference])]
            subsets += [('B_' + suite, part) for suite, part in frame.groupby('suite')]
            common.extend(summarize_pairs(frame, methods, 'available-B'))
            methods['original_knn'] = frame.knn20.to_numpy(int)
        else:
            subsets += [('current_' + benchmark, part) for benchmark, part in frame.groupby('benchmark')]
        for scope, part in subsets:
            ids = part.index.to_numpy()
            for cutoff_name, cutoff in [('all', 10000), ('q9', 9), ('q13', 13), ('q20', 20),
                                        ('budget_0.50', np.floor(part.horizon_actions.to_numpy()*.5/10)),
                                        ('budget_0.75', np.floor(part.horizon_actions.to_numpy()*.75/10))]:
                for name, values in methods.items():
                    metrics.append(dict(cohort=scope, cutoff=cutoff_name, method=name,
                                        **alarm_metrics(part, values[ids], cutoff)))
                category = alarm_groups(methods['v82'][ids], methods['knn'][ids], cutoff)
                for name in ('both', 'knn_only', 'v82_only', 'neither'):
                    selected = category == name
                    y = part.failure.to_numpy(bool)[selected]
                    groups.append(dict(cohort=scope, cutoff=cutoff_name, group=name, episodes=len(y),
                                       failures=int(y.sum()), successes=int((~y).sum())))
        for row in frame.itertuples():
            i = row.Index
            reference_task_key = row.task if row.task.startswith(row.suite + '/') else row.suite + '/' + row.task
            anchors = [('knn_first', int(row.knn_available_first)), ('v82_first', int(row.v82_frozen)),
                       ('q9', 9), ('q13', 13), ('q20', 20)]
            for anchor, q in anchors:
                if q < 0 or q >= row.length or not np.isfinite(data['score'][i, q]):
                    continue
                neighbors = data['neighbor_ids'][i, q]
                assert (neighbors >= 0).all()
                score, components = float(data['score'][i, q]), data['components'][i, q].astype(float)
                fractions = components / score if score else np.zeros(len(GROUPS))
                z, residual = data['x'][i, q], data['signed_residual'][i, q]
                neighbor_tasks = bank_frame.task.to_numpy()[neighbors]
                point = dict(cohort=cohort, row=i, identity=str(row.global_row) if cohort == 'B' else row.name,
                             task=row.task, reference_task_key=reference_task_key,
                             failure=bool(row.failure), anchor=anchor, query=q,
                             joint_group=row.joint_group, score=score, score_ratio=score/threshold,
                             v82_already_on=bool(0 <= row.v82_frozen <= q),
                             knn_already_on=bool(0 <= row.knn_available_first <= q),
                             unique_neighbor_parents=int(data['unique_neighbor_parents'][i, q]),
                             unique_neighbor_tasks=len(set(neighbor_tasks)),
                             neighbor_same_task_fraction=float((neighbor_tasks == reference_task_key).mean()),
                             neighbor_same_suite_fraction=float((bank_frame.suite.to_numpy()[neighbors] == row.suite).mean()),
                             median_neighbor_query=float(np.median(profile['success_queries'][neighbors])),
                             dominant_component=list(GROUPS)[int(np.argmax(fractions))],
                             z_front_mobility=float(z[:4].mean()), z_back_mobility=float(z[4:8].mean()),
                             z_acceleration=float(z[8]), z_periodicity=float(z[9]),
                             residual_front_mobility=float(residual[:4].mean()), residual_back_mobility=float(residual[4:8].mean()),
                             residual_acceleration=float(residual[8]), residual_periodicity=float(residual[9]))
                point.update({name + '_fraction': float(fractions[j]) for j, name in enumerate(GROUPS)})
                point.update({name + '_fraction_per_axis': float(fractions[j] / (4 if j < 2 else 1))
                              for j, name in enumerate(GROUPS)})
                points.append(point)
        if cohort == 'B':
            readouts = dict(knn_distance=data['score'], freeze=data['freeze_score'],
                            acceleration=data['acceleration_score'], periodicity=data['periodicity_score'])
            readouts.update({name + '_distance_term': data['components'][..., j] for j, name in enumerate(GROUPS)})
            for task, part in frame.groupby('task'):
                for q in (9, 13, 20):
                    eligible = part[part.length > q]
                    ids = eligible.index.to_numpy()
                    for method, value in readouts.items():
                        score = auc(eligible.failure.to_numpy(), value[ids, q])
                        if np.isfinite(score):
                            ranking.append(dict(cohort='B', task=task, query=q, measurement=method,
                                                auc=score, parents=len(eligible), failures=int(eligible.failure.sum())))
    point_table = pd.DataFrame(points)
    point_table.to_csv(ROOT / 'neighbor-explanations.csv', index=False)
    pd.DataFrame(metrics).to_csv(ROOT / 'available-alarm-metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(ROOT / 'available-joint-groups.csv', index=False)
    pd.DataFrame(common).to_csv(ROOT / 'available-common-prefix-summary.csv', index=False)
    rank_table = pd.DataFrame(ranking)
    rank_table.to_csv(ROOT / 'available-same-query-aucs.csv', index=False)
    directional = []
    for query, part in rank_table.groupby('query'):
        paired = part.pivot(index='task', columns='measurement', values='auc')
        for comparison in ('knn_distance', 'acceleration_distance_term'):
            delta = (paired.acceleration - paired[comparison]).dropna()
            rng = np.random.default_rng(20260915 + int(query))
            boot = delta.to_numpy()[rng.integers(0, len(delta), (2000, len(delta)))].mean(-1)
            directional.append(dict(query=int(query), comparison=comparison, tasks=len(delta),
                                    signed_acceleration_auc=paired.acceleration.mean(),
                                    comparison_auc=paired[comparison].mean(), mean_paired_difference=delta.mean(),
                                    task_bootstrap_q025=np.quantile(boot, .025), task_bootstrap_q975=np.quantile(boot, .975)))
    pd.DataFrame(directional).to_csv(ROOT / 'directional-ranking-comparisons.csv', index=False)
    summaries = []
    keys = [name + '_fraction' for name in GROUPS] + [
        'z_front_mobility', 'z_back_mobility', 'z_acceleration', 'z_periodicity',
        'residual_front_mobility', 'residual_back_mobility', 'residual_acceleration', 'residual_periodicity',
        'score_ratio', 'unique_neighbor_parents', 'unique_neighbor_tasks', 'neighbor_same_task_fraction']
    for (cohort, anchor, joint, failure), part in point_table.groupby(['cohort', 'anchor', 'joint_group', 'failure']):
        summaries.append(dict(cohort=cohort, anchor=anchor, joint_group=joint, failure=failure, parents=len(part),
                              **{key: part[key].median() for key in keys}))
    pd.DataFrame(summaries).to_csv(ROOT / 'neighbor-explanation-summary.csv', index=False)
    dominant = point_table[point_table.anchor.eq('knn_first')].groupby(
        ['cohort', 'joint_group', 'failure', 'dominant_component']).size().reset_index(name='parents')
    dominant.to_csv(ROOT / 'dominant-distance-components.csv', index=False)

    current = pd.read_csv(ROOT / 'current-decisions.csv')
    current_data = archive(ROOT / 'current-geometry.npz')
    selection_path = ROOT.parent / 'moe-input-response-20260915/selection.json'
    selection = json.loads(selection_path.read_text())
    anchors = []
    for parent in selection['parents']:
        row = current[current.name.eq(parent['name'])].iloc[0]
        q = parent['query']
        i = int(current.index[current['name'].eq(parent['name'])][0])
        anchors.append(dict(name=parent['name'], query=q, failure=bool(row.failure),
                            stable_matched=parent['stable_matched'], v82_first=int(row.v82_frozen),
                            knn_available_first=int(row.knn_available_first),
                            v82_on=bool(0 <= row.v82_frozen <= q), knn_on=bool(0 <= row.knn_available_first <= q),
                            knn_distance=float(current_data['score'][i, q]), threshold=threshold,
                            eef_next_displacement_m=parent['eef_next_displacement_m']))
    pd.DataFrame(anchors).to_csv(ROOT / 'functional-anchor-context.csv', index=False)
    results = dict(profile_threshold=threshold, original_bank_reproduced=False,
                   primary_metrics=[r for r in metrics if r['cohort'] in ('B', 'current_plus', 'current_pro') and r['cutoff'] == 'all'],
                   primary_groups=[r for r in groups if r['cohort'] in ('B', 'current_plus', 'current_pro') and r['cutoff'] == 'all'],
                   same_query_auc_summary=pd.DataFrame(ranking).groupby(['query', 'measurement']).agg(
                       task_mean_auc=('auc', 'mean'), tasks=('task', 'nunique')).reset_index().to_dict('records'),
                   common_prefix=common, functional_anchors=anchors)
    results['directional_comparisons'] = directional
    save_json(ROOT / 'geometry-results.json', results)
    print(pd.DataFrame(results['primary_metrics']).to_string(index=False), flush=True)
    print(dominant.to_string(index=False), flush=True)
    print(pd.DataFrame(results['same_query_auc_summary']).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
