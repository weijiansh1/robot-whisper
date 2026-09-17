"""Recompute causal sequence evidence across all compatible offline cohorts."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from analyze_moe_expansion import encode_two
from alarm_trajectory_study import physical_window, phase_signature
from moe_offline_sequences import (ROOT, EXPANDED, PREVIOUS, OLD, CONTROL, METHODS,
    archive, digest, save_json, sequence_scores, first_events, auc, interval)


class Sources:
    def __init__(self):
        self.records, self.seals = {}, {}
        for directory in (EXPANDED, PREVIOUS, OLD):
            path = directory / 'artifact-hashes.json'
            self.seals[directory] = json.loads(path.read_text())
            self.records[str(path)] = dict(sha256=digest(path), verified_against_prior_seal=False)

    def add(self, path):
        path = Path(path)
        if str(path) in self.records:
            return
        sha, verified = digest(path), False
        for directory, seal in self.seals.items():
            if path.is_relative_to(directory):
                relative = str(path.relative_to(directory))
                if relative in seal:
                    assert sha == seal[relative], str(path)
                    verified = True
        self.records[str(path)] = dict(sha256=sha, verified_against_prior_seal=verified)

    def control_seal(self, directory):
        path = directory / 'verification.json'
        self.add(path)
        data = json.loads(path.read_text())
        assert data['passed'], str(path)
        self.seals[directory] = data['files']


def load(sources):
    for name in ('episode-index.csv', 'two-axis-coordinates.npz', 'supported-motif-scores.npz',
                 'metadata-only-excluded.csv', 'availability.json'):
        sources.add(EXPANDED / name)
    frame = pd.read_csv(EXPANDED / 'episode-index.csv', low_memory=False).fillna({'primary_failure_reason': ''})
    coordinates = archive(EXPANDED / 'two-axis-coordinates.npz')['coordinates']
    sources.add(PREVIOUS / 'fresh-decisions.csv')
    sources.add(PREVIOUS / 'fresh-routing-inputs.npz')
    sources.add(PREVIOUS / 'fresh-manifest.json')
    fresh = pd.read_csv(PREVIOUS / 'fresh-decisions.csv')
    manifest = {e['name']: e for e in json.loads((PREVIOUS / 'fresh-manifest.json').read_text())['episodes']}
    raw = archive(PREVIOUS / 'fresh-routing-inputs.npz')
    fresh_coordinates = encode_two(raw['mobility'], raw['acceleration'], raw['valid'])
    rows = []
    for r in fresh.itertuples():
        e = manifest[r.name]
        assert r.length == e['queries'] == raw['valid'][r.Index].sum()
        assert r.failure == (not e['source_result']['success'])
        rows.append(dict(identity='fresh|'+r.name, cohort='fresh_'+r.benchmark, run_id='fresh_20',
            task=r.task, suite=r.suite, episode=r.Index, length=r.length,
            horizon_actions=r.horizon_actions, source_row=r.Index, init_state_id=r.init_state_id,
            noise_seed=e['flow_noise_seed'], failure=r.failure,
            checkpoint=e['policy_identity']['checkpoint_sha256'], action_steps=r.action_steps,
            primary_failure_reason='', v82=r.v82_frozen, readout_fingerprint='',
            trace_sha256=e['source_trace_sha256']))
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    coordinates = np.concatenate([coordinates, fresh_coordinates])
    assert len(frame) == 33043 and not frame.identity.duplicated().any()
    traces = frame.trace_sha256.dropna()
    assert not traces[traces.ne('')].duplicated().any()
    expected_valid = np.arange(52)[None, :] < frame.length.to_numpy()[:, None]
    expected_valid[:, :7] = False
    np.testing.assert_array_equal(np.isfinite(coordinates).all(-1), expected_valid)
    return frame, coordinates


def summarise(strata, group_keys, value='auc'):
    output, tasks = [], []
    for group, part in strata.groupby(group_keys, sort=True):
        group = group if isinstance(group, tuple) else (group,)
        detail = dict(zip(group_keys, group))
        by_task = part.groupby('task')[value].mean()
        low, high = interval(by_task.to_numpy())
        output.append(dict(**detail, mean=by_task.mean(), low=low, high=high, tasks=len(by_task),
                           strata=len(part), pairs=int(part.pairs.sum())))
        tasks.extend(dict(**detail, task=t, mean=x) for t, x in by_task.items())
    return pd.DataFrame(output), pd.DataFrame(tasks)


def matched_outcomes(frame, scores):
    result, support = [], []
    for (cohort, task, initial), part in frame.groupby(['cohort', 'task', 'init_state_id'], sort=True):
        if part.failure.nunique() < 2:
            continue
        for q in (13, 18, 26):
            use = part[part.length > q]
            if use.failure.nunique() < 2:
                continue
            y = use.failure.to_numpy()
            support.append(dict(cohort=cohort, task=task, init_state_id=initial, query=q,
                failures=int(y.sum()), successes=int((~y).sum()),
                identities='|'.join(use.identity)))
            for k, method in enumerate(METHODS):
                result.append(dict(cohort=cohort, task=task, init_state_id=initial, query=q,
                    method=method, auc=auc(y, scores[use.index, q, k]), pairs=int(y.sum()*(~y).sum())))
    strata = pd.DataFrame(result)
    strata.to_csv(ROOT / 'same-initial-query-strata.csv', index=False)
    pd.DataFrame(support).to_csv(ROOT / 'same-initial-query-support.csv', index=False)
    summary, tasks = summarise(strata, ['cohort', 'query', 'method'])
    summary.to_csv(ROOT / 'same-initial-query-summary.csv', index=False)
    tasks.to_csv(ROOT / 'same-initial-query-tasks.csv', index=False)
    differences = []
    for (cohort, q), part in tasks.groupby(['cohort', 'query']):
        table = part.pivot(index='task', columns='method', values='mean')
        delta = table.activity_then_freeze-table.simultaneous
        low, high = interval(delta.to_numpy())
        differences.append(dict(cohort=cohort, query=q, tasks=len(delta),
            sequence_minus_simultaneous=delta.mean(), low=low, high=high))
    pd.DataFrame(differences).to_csv(ROOT / 'ordering-increment.csv', index=False)
    risksets = []
    for (cohort, failure), part in frame.groupby(['cohort', 'failure']):
        for q in (12, 13, 18, 26):
            risksets.append(dict(cohort=cohort, failure=failure, query=q, parents=len(part),
                                 observed=int((part.length > q).sum())))
    pd.DataFrame(risksets).to_csv(ROOT / 'risk-sets.csv', index=False)
    return summary


def alarm_comparisons(frame, scores):
    lookup = {key: part for key, part in frame.groupby(['cohort', 'task', 'init_state_id'])}
    support, comparisons = [], []
    for row in frame[frame.failure & (frame.v82 >= 0)].itertuples():
        q = int(row.v82)
        same = lookup[(row.cohort, row.task, row.init_state_id)]
        controls = same[(~same.failure) & (same.length > q)]
        reason = 'before_q12' if q < 12 else ('no_observed_success' if not len(controls) else 'matched')
        support.append(dict(identity=row.identity, cohort=row.cohort, task=row.task,
            init_state_id=row.init_state_id, query=q, status=reason, controls=len(controls)))
        if reason != 'matched':
            continue
        ids = np.r_[row.Index, controls.index]
        for k, method in enumerate(METHODS):
            comparisons.append(dict(cohort=row.cohort, task=row.task, identity=row.identity,
                query=q, method=method, auc=auc(np.r_[True, np.zeros(len(controls), bool)], scores[ids, q, k]),
                pairs=len(controls)))
    pd.DataFrame(support).to_csv(ROOT / 'v82-alarm-match-support.csv', index=False)
    table = pd.DataFrame(comparisons)
    table.to_csv(ROOT / 'v82-alarm-matches.csv', index=False)
    summary, _ = summarise(table, ['cohort', 'method'])
    summary.to_csv(ROOT / 'v82-alarm-match-summary.csv', index=False)


def events_and_annotations(frame, scores):
    first = first_events(scores[..., :5])
    result = frame.copy()
    for k, method in enumerate(METHODS[:5]):
        result['first_'+method] = first[:, k]
    result.to_csv(ROOT / 'episode-events.csv', index=False)
    summaries, annotations, counterexamples = [], [], []
    for (cohort, suite), part in frame.groupby(['cohort', 'suite']):
        y = part.failure.to_numpy()
        for k, method in enumerate(METHODS[:5]):
            q = first[part.index, k]
            hit = q >= 0
            bad_hit = y & hit
            fraction = q[bad_hit]*10/part.horizon_actions.to_numpy()[bad_hit]
            available = part.v82.to_numpy() != -2
            extra = available & (part.v82.to_numpy() < 0) & hit
            summaries.append(dict(cohort=cohort, suite=suite, method=method, failures=int(y.sum()),
                successes=int((~y).sum()), tp=int((hit & y).sum()), fp=int((hit & ~y).sum()),
                extra_failures=int((extra & y).sum()), extra_successes=int((extra & ~y).sum()),
                median_budget_fraction=float(np.median(fraction)) if len(fraction) else np.nan))
    for (cohort, reason), part in frame[frame.failure & frame.primary_failure_reason.ne('')].groupby(['cohort', 'primary_failure_reason']):
        annotations.append(dict(cohort=cohort, reason=reason, failures=len(part),
            **{method: int((first[part.index, k] >= 0).sum()) for k, method in enumerate(METHODS[:5])}))
    for row in frame[~frame.failure].itertuples():
        q = int(first[row.Index, 3])
        if q >= 0:
            counterexamples.append(dict(identity=row.identity, cohort=row.cohort, suite=row.suite,
                task=row.task, init_state_id=row.init_state_id, query=q, score=scores[row.Index, q, 3],
                order_excess=scores[row.Index, q, 5], action_steps=row.action_steps,
                actions_after_event=row.action_steps-q*10 if row.action_steps >= 0 else np.nan))
    pd.DataFrame(summaries).to_csv(ROOT / 'event-summary.csv', index=False)
    pd.DataFrame(annotations).to_csv(ROOT / 'physical-failure-categories.csv', index=False)
    pd.DataFrame(counterexamples).to_csv(ROOT / 'successful-counterexamples.csv', index=False)
    return summaries


def external_validation(frame, scores, sources):
    all_windows = []
    metadata = {}
    for cohort in ('current', 'fresh'):
        path = PREVIOUS / (cohort+'-decisions.csv')
        sources.add(path)
        metadata.update({r.name: r for r in pd.read_csv(path).itertuples()})
    sources.add(PREVIOUS / 'physical-query-windows.csv')
    saved = pd.read_csv(PREVIOUS / 'physical-query-windows.csv').query('horizon == 3')
    for row in frame[frame.cohort.str.startswith(('current_', 'fresh_'))].itertuples():
        name = row.identity.split('|', 1)[1]
        previous = metadata[name]
        source = PREVIOUS / ('fresh-physical' if row.cohort.startswith('fresh_') else 'physical') / (name+'.npz')
        sources.add(source)
        data = archive(source)
        assert len(data['predicates']) == row.length+1
        assert bool(data['predicates'][-1].all()) == (not row.failure)
        np.testing.assert_array_equal(data['action_steps'][:-1], np.arange(row.length)*10)
        check = saved[saved.name.eq(name)].set_index('query')
        for q in range(12, row.length):
            window = physical_window(data, q, 3)
            assert window['category'] == check.loc[q, 'category']
            all_windows.append(dict(identity=row.identity, name=name, cohort=row.cohort,
                benchmark=previous.benchmark, task=previous.task_id, failure=row.failure,
                phase=json.dumps(phase_signature(data, q)), **window,
                **{method: scores[row.Index, q, k] for k, method in enumerate(METHODS)}))
    table = pd.DataFrame(all_windows)
    table.to_csv(ROOT / 'external-query-windows.csv', index=False)
    table = table[table.observed_queries == 3]
    positive = {'static_vs_progress_proxy': {'holding_or_static', 'return_motion_without_subgoal_gain'},
                'unverified_vs_predicate_gain': {'holding_or_static', 'return_motion_without_subgoal_gain',
                                                'motion_without_verified_subgoal_gain'}}
    negative = {'static_vs_progress_proxy': {'completed', 'subgoal_gain', 'geometric_approach_proxy'},
                'unverified_vs_predicate_gain': {'completed', 'subgoal_gain'}}
    result = []
    for contrast in positive:
        eligible = table[table.category.isin(positive[contrast] | negative[contrast])]
        for (benchmark, task, q, phase), part in eligible.groupby(['benchmark', 'task', 'query', 'phase']):
            y = part.category.isin(positive[contrast]).to_numpy()
            if not y.any() or y.all():
                continue
            for method in METHODS:
                result.append(dict(contrast=contrast, benchmark=benchmark, task=task, query=q,
                    phase=phase, method=method, auc=auc(y, part[method]), pairs=int(y.sum()*(~y).sum()),
                    identities='|'.join(part.identity)))
    strata = pd.DataFrame(result)
    strata.to_csv(ROOT / 'external-matched-strata.csv', index=False)
    summary, tasks = summarise(strata, ['contrast', 'method'])
    summary.to_csv(ROOT / 'external-matched-summary.csv', index=False)
    tasks.to_csv(ROOT / 'external-matched-tasks.csv', index=False)
    return summary


def branch_validation(frame, coordinates, sources):
    branches, branch_coordinates = [], {}
    sources.add(OLD / 'branch-features.npz')
    sources.add(OLD / 'branch-results.csv')
    from analyze_moe_expansion import INDEX
    archived = archive(OLD / 'branch-features.npz')
    for row in pd.read_csv(OLD / 'branch-results.csv').itertuples():
        key = row.parent+'__candidate'+str(row.candidate)
        x = archived[key][:, [INDEX['mobility_log_ratio'], INDEX['acceleration_log_ratio']]]
        identity = 'head_candidates|'+key
        branches.append(dict(identity=identity, run='head_candidates', parent=row.parent,
            arm='native' if row.candidate == 0 else 'candidate'+str(row.candidate),
            success=row.success, source_success=row.source_success,
            intervention_query=row.intervention_query, length=len(x), action_steps=row.action_steps))
        branch_coordinates[identity] = x
    raw_root = Path('/data/libero-runtime/samples/v82-knn-mechanism-20260915')
    for name in ('current-routing-inputs.npz', 'current-decisions.csv'):
        sources.add(raw_root / name)
    raw = archive(raw_root / 'current-routing-inputs.npz')
    parents = pd.read_csv(raw_root / 'current-decisions.csv').set_index('name', drop=False)
    parent_rows = {name: i for i, name in enumerate(parents.name)}
    reconstruction_errors = []
    for run in ('p3d-gated-rollout-20260915-t9k1v766', 'p3g-crossover-rollout-20260915-1gmt0y3o'):
        directory = CONTROL / run
        sources.control_seal(directory)
        sources.add(directory / 'branches.csv')
        records = pd.read_csv(directory / 'branches.csv')
        for row in records.itertuples():
            path = directory / row.parent / row.arm / 'queries.jsonl'
            sources.add(path)
            logged = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            parent = parent_rows[row.parent]
            length = int(row.queries)
            m = raw['mobility'][parent:parent+1, :length].copy()
            a = raw['acceleration'][parent:parent+1, :length].copy()
            start = logged[0]['query']
            assert [r['query'] for r in logged] == list(range(start, length))
            assert all(r['step'] == 10*r['query'] for r in logged)
            assert sum(r['executed'] for r in logged)+10*start == row.action_steps
            for record in logged:
                selected = record['selected']['status']
                q = record['query']
                m[0, q] = selected['layer_mobility']
                a[0, q] = selected['route_acceleration']
            x = encode_two(m, a, np.ones((1, length), bool))[0]
            old_id = int(frame.index[frame.identity.eq('current|'+row.parent)][0])
            error = float(np.nanmax(np.abs(x[:start]-coordinates[old_id, :start])))
            assert error < 1e-4
            reconstruction_errors.append(dict(run=run, parent=row.parent, arm=row.arm, prefix_error=error))
            identity = run+'|'+row.parent+'|'+row.arm
            branches.append(dict(identity=identity, run=run, parent=row.parent, arm=row.arm,
                success=row.success, source_success=not bool(parents.loc[row.parent, 'failure']),
                intervention_query=start, length=length, action_steps=row.action_steps))
            branch_coordinates[identity] = x
    frame_b = pd.DataFrame(branches)
    full_scores, comparisons = {}, []
    for row in frame_b.itertuples():
        full_scores[row.identity] = sequence_scores(branch_coordinates[row.identity][None])[0]
    for row in frame_b.itertuples():
        native = frame_b[(frame_b.run == row.run) & (frame_b.parent == row.parent) & (frame_b.arm == 'native')].iloc[0]
        common_end = min(row.length, native.length)
        score, original = full_scores[row.identity], full_scores[native.identity]
        start = row.intervention_query
        for k, method in enumerate(METHODS):
            delta = score[start:common_end, k]-original[start:common_end, k]
            comparisons.append(dict(identity=row.identity, run=row.run, parent=row.parent, arm=row.arm,
                method=method, source_success=row.source_success, success=row.success,
                mean_delta_vs_native=float(delta.mean()),
                first_six_delta_vs_native=float(delta[:6].mean()),
                own_suffix_peak=float(score[start:, k].max()),
                rescue=bool(row.success and not row.source_success)))
    frame_b.to_csv(ROOT / 'branch-index.csv', index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / 'branch-comparisons.csv', index=False)
    pd.DataFrame(reconstruction_errors).to_csv(ROOT / 'branch-prefix-checks.csv', index=False)
    np.savez_compressed(ROOT / 'branch-sequence-scores.npz', **full_scores)
    return frame_b


def inventory(sources):
    studies = []
    for directory in sorted(CONTROL.iterdir()):
        path = directory / 'summary.json'
        if not path.exists():
            continue
        sources.add(path)
        data = json.loads(path.read_text())
        studies.append(dict(run=directory.name, summary_path=str(path),
            kind='rollout' if ('methods' in data or 'arms' in data) else 'fixed_input_probe',
            actual_rollouts=data.get('actual_rollouts'), non_independent_aliases=data.get('non_independent_no_alarm_aliases'),
            methods=data.get('methods', data.get('arms')), model_calls=data.get('model_calls'),
            environment_actions=data.get('environment_actions'),
            included_in_fixed_chunk_sequence=directory.name.startswith(('p3d-', 'p3g-', 'p3b-'))))
    save_json(ROOT / 'other-offline-study-inventory.json', studies)


def main():
    if (ROOT / 'sequence-scores.npz').exists():
        raise SystemExit('This run already has scores; refusing to overwrite')
    start = time.perf_counter()
    save_json(ROOT / 'started.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(ROOT / 'PROTOCOL.md')))
    sources = Sources()
    frame, coordinates = load(sources)
    scores = sequence_scores(coordinates)
    for stop in (12, 13, 18, 30):
        np.testing.assert_array_equal(sequence_scores(coordinates[::137, :stop]), scores[::137, :stop])
    # Control scores agree with the previous formulas on the shared q>=12 range.
    previous = archive(EXPANDED / 'supported-motif-scores.npz')
    for new, old in ((0, 'freeze_control'), (1, 'acceleration_control'), (2, 'frozen_active')):
        k = list(previous['names']).index(old)
        np.testing.assert_array_equal(scores[:33023, 12:, new], previous['scores'][:, 12:, k])
    print('Computed and checked scores for 33,043 complete records', flush=True)
    frame.to_csv(ROOT / 'episode-index.csv', index=False)
    np.savez_compressed(ROOT / 'sequence-scores.npz', scores=scores, methods=np.asarray(METHODS))
    np.savez_compressed(ROOT / 'coordinates.npz', coordinates=coordinates)
    events = events_and_annotations(frame, scores)
    associations = matched_outcomes(frame, scores)
    print('Completed same-initial/query comparisons', flush=True)
    alarm_comparisons(frame, scores)
    physical = external_validation(frame, scores, sources)
    branches = branch_validation(frame, coordinates, sources)
    inventory(sources)
    counts = frame.groupby(['cohort', 'suite']).agg(parents=('identity', 'size'),
        failures=('failure', 'sum'), queries=('length', 'sum')).reset_index()
    counts['successes'] = counts.parents-counts.failures
    counts.to_csv(ROOT / 'cohort-counts.csv', index=False)
    save_json(ROOT / 'source-hashes.json', sources.records)
    save_json(ROOT / 'implementation-hashes.json', {str(p): digest(p) for p in (
        Path(__file__), Path('/data/libero-runtime/moe_offline_sequences.py'),
        Path('/data/libero-runtime/test_moe_offline_sequences.py'),
        Path('/data/libero-runtime/analyze_moe_expansion.py'),
        Path('/data/libero-runtime/alarm_trajectory_study.py'))})
    save_json(ROOT / 'results.json', dict(parents=len(frame), queries=int(frame.length.sum()),
        failures=int(frame.failure.sum()), successes=int((~frame.failure).sum()), branches=len(branches),
        sources=len(sources.records), sources_with_prior_seal=sum(r['verified_against_prior_seal'] for r in sources.records.values()),
        complete_cohorts=counts.to_dict('records'), model_forwards=0, environment_actions=0,
        wall_seconds=time.perf_counter()-start, causal_prefix_checks_passed=True,
        old_control_scores_exact=True, independent_blind_test=False, modified_v82=False,
        q13_associations=associations.query('query == 13').to_dict('records'),
        physical=physical.to_dict('records'), events=events))
    print('Finished offline sequence study', flush=True)
    print(associations.query('query == 13 and cohort == "external_8b"').to_string(index=False), flush=True)
    print(physical.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
