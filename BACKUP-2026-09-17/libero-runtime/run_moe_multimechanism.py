"""Run the frozen all-branch physical matching and contribution audit."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from moe_multimechanism import (ROOT, SIGNALS, HEADS, FEATURES, phase, physical_window,
                               geometry_distance, contribution_contrast)
from moe_response_analysis import digest, save_json, scope_masks

BASE = ROOT.parent.parent
CURRENT = BASE / 'samples/v82-evaluation-20260914T144322Z'
PREVIOUS = BASE / 'samples/v82-knn-trajectory-20260915'
EARLY = BASE / 'samples/moe-input-response-20260915'
DECISIONS = BASE / 'samples/v82-knn-mechanism-20260915/current-decisions.csv'
CONDITIONS = ('o0n0', 'o0n1', 'o1n0', 'o1n1')
TIERS = ('phase', 'geometry', 'all5_margin_0.5', 'all5_margin_1', 'all5_margin_2', 'same_variant_all5_margin_1')


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.seals = {}
        for directory in (PREVIOUS, EARLY):
            path = directory / 'artifact-hashes.json'
            self.seals[directory] = json.loads(path.read_text())
            self.add(path)

    def add(self, path):
        path = Path(path)
        if str(path) not in self.hashes:
            sha = digest(path)
            verified = False
            for directory, seal in self.seals.items():
                if path.is_relative_to(directory) and str(path.relative_to(directory)) in seal:
                    assert sha == seal[str(path.relative_to(directory))], str(path)
                    verified = True
            self.hashes[str(path)] = dict(sha256=sha, verified_against_prior_seal=verified)
        return path

    def json(self, path):
        return json.loads(self.add(path).read_text())

    def npz(self, path):
        with np.load(self.add(path), allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}

    def csv(self, path):
        return pd.read_csv(self.add(path))


def load_data(inputs):
    current = inputs.json(CURRENT / 'summary.json')
    fresh = inputs.json(PREVIOUS / 'fresh-manifest.json')
    episodes = current['episodes']+fresh['episodes']
    assert len(episodes) == 80 and len({e['source_trace_sha256'] for e in episodes}) == 80
    assert len({e['policy_identity']['checkpoint_sha256'] for e in episodes}) == 1
    dec0, dec1 = inputs.csv(DECISIONS), inputs.csv(PREVIOUS / 'fresh-decisions.csv')
    decisions = pd.concat([dec0, dec1], ignore_index=True).set_index('name')
    fresh_scores = inputs.npz(PREVIOUS / 'fresh-scores.npz')['v82_scores']
    fresh_index = {name: i for i, name in enumerate(dec1.name)}
    thresholds = np.array([current['thresholds_v8'][s] for s in SIGNALS])
    margins = np.r_[thresholds[:3]*.2, .1, thresholds[4]-.2]
    assert current['v82_new_head_slope_per_query'] == -.0015
    physics, metadata, query_rows = {}, {}, []
    for episode in episodes:
        name, count = episode['name'], episode['queries']
        row = decisions.loc[name]
        assert row.length == count and bool(row.failure) == (not episode['source_result']['success'])
        path = ROOT / 'physical' / (name+'.npz')
        physical_meta = inputs.json(path.with_suffix('.json'))
        assert physical_meta['source_trace_sha256'] == episode['source_trace_sha256']
        assert physical_meta['protocol_sha256'] == digest(ROOT / 'PROTOCOL.md')
        data = inputs.npz(path)
        assert digest(path) == physical_meta['output_sha256'] and len(data['predicates']) == count+1
        physics[name] = data
        raw = inputs.csv(CURRENT / name / 'signals.csv') if name not in fresh_index else None
        scores = (raw[[s+'_score' for s in SIGNALS]].to_numpy(float) if raw is not None
                  else fresh_scores[fresh_index[name], :count].astype(float))
        moving_thresholds = np.broadcast_to(thresholds, (count, 5)).copy()
        moving_thresholds[:, 3:] -= .0015*np.arange(count)[:, None]
        if raw is not None:
            np.testing.assert_allclose(raw[[s+'_threshold_v82' for s in SIGNALS]], moving_thresholds, rtol=0, atol=1e-15)
        centered = (scores-moving_thresholds)/margins
        firsts = {h: int(row['head_'+h]) for h in HEADS}
        assert min([v for v in firsts.values() if v >= 0], default=-1) == int(row.v82_frozen)
        meta = dict(name=name, benchmark=episode['benchmark'], base_task=int(episode['base_task_id']),
            task_suite=episode['task_suite'], variant_task_id=int(episode['task_id']),
            init_state_id=int(episode['init_state_id']), success=not bool(row.failure), length=count,
            v82_first=int(row.v82_frozen), **{'first_'+h: v for h, v in firsts.items()})
        metadata[name] = dict(meta, source_artifact_dir=episode['source_artifact_dir'])
        for q in range(count):
            on = {h: 0 <= value <= q for h, value in firsts.items()}
            current_on = [bool(centered[q, 0] >= 0), bool(min(centered[q, 1:3]) >= 0),
                          bool(centered[q, 3] >= 0), bool(centered[q, 4] >= 0)]
            qrow = dict(**meta, query=q, action_step=int(data['action_steps'][q]),
                budget_fraction=float(data['action_steps'][q]/520), phase=phase(data, q),
                head_mask='+'.join(h for h in HEADS if on[h]) or 'none', v82_on=any(on.values()),
                current_head_mask='+'.join(h for h, hit in zip(HEADS, current_on) if hit) or 'none',
                all5_available=bool(np.isfinite(centered[q]).all()),
                **{'on_'+h: on[h] for h in HEADS}, **{'score_'+s: scores[q, k] for k, s in enumerate(SIGNALS)},
                **{'margin_'+s: centered[q, k] for k, s in enumerate(SIGNALS)})
            if raw is not None:
                assert bool(raw.iloc[q].v82_alarm) == qrow['v82_on']
            query_rows.append(qrow)
    save_json(ROOT / 'baseline.json', dict(signals=SIGNALS, heads=HEADS, thresholds=thresholds,
        fixed_margins=margins, slope=-.0015, native_parents=80, original_parameters_unchanged=True))
    pd.DataFrame(metadata.values()).to_csv(ROOT / 'parents.csv', index=False)
    queries = pd.DataFrame(query_rows)
    queries.to_csv(ROOT / 'queries.csv', index=False)
    return physics, metadata, queries


def build_windows(physics, queries):
    rows = []
    for r in queries.itertuples(index=False):
        if r.query < 1:
            continue
        base = r._asdict()
        for horizon in (3, 5, 10):
            rows.append(dict(base, **physical_window(physics[r.name], r.query, horizon)))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'physical-windows.csv', index=False)
    primary = frame[(frame['query'] >= 10) & frame.complete_window & frame.horizon.eq(3) & frame.all5_available]
    primary.groupby(['benchmark', 'local_label', 'category', 'head_mask']).agg(
        windows=('name', 'size'), parents=('name', 'nunique')).reset_index().to_csv(ROOT / 'branch-behavior-counts.csv', index=False)
    summaries = []
    for h in ('any_alarm', 'no_alarm')+HEADS:
        subset = primary[primary.v82_on if h == 'any_alarm' else ~primary.v82_on if h == 'no_alarm' else primary['on_'+h]]
        for label, part in subset.groupby('local_label'):
            summaries.append(dict(branch=h, label=label, windows=len(part), parents=part.name.nunique(),
                moving_windows=int(part.category.isin(['return_motion', 'other_motion_no_gain']).sum())))
    pd.DataFrame(summaries).to_csv(ROOT / 'branch-summary.csv', index=False)
    later = frame[frame.horizon.eq(10) & frame.complete_window][['name', 'query', 'local_label', 'gained_any']].rename(
        columns={'local_label': 'h10_label', 'gained_any': 'h10_gained_any'})
    joined = primary.merge(later, on=['name', 'query'], how='left')
    joined.to_csv(ROOT / 'primary-with-later-progress.csv', index=False)
    examples = joined[joined.local_label.eq('no_observed_gain') & ~joined.on_freeze &
        joined.category.isin(['return_motion', 'other_motion_no_gain'])]
    examples.to_csv(ROOT / 'moving-no-gain-without-freeze.csv', index=False)
    return frame, primary, joined


def match_pairs(frame, physics, label):
    selected, candidates = [], []
    for _, part in frame.groupby(['horizon', 'benchmark', 'base_task', 'phase'], sort=True):
        positive = part[part.local_label.eq('verified_progress')]
        for bad in part[part.local_label.eq('no_observed_gain')].itertuples(index=False):
            controls = positive[positive.name.ne(bad.name) & ((positive['query']-bad.query).abs() <= 2)]
            for good in controls.itertuples(index=False):
                geom = geometry_distance(physics[bad.name], bad.query, physics[good.name], good.query)
                baseline = max(abs(getattr(bad, 'margin_'+s)-getattr(good, 'margin_'+s)) for s in SIGNALS)
                ok_geom = (geom['geometry_rms_m'] <= .05 and geom['geometry_max_m'] <= .1 and
                           geom['rotation_max_deg'] <= 30 and geom['joint_max_range'] <= .1)
                same_variant = bad.task_suite == good.task_suite and bad.variant_task_id == good.variant_task_id
                candidates.append(dict(population=label, horizon=bad.horizon, benchmark=bad.benchmark,
                    base_task=bad.base_task, no_gain_name=bad.name, no_gain_query=bad.query,
                    progress_name=good.name, progress_query=good.query,
                    no_gain_category=bad.category, no_gain_head_mask=bad.head_mask, progress_head_mask=good.head_mask,
                    no_gain_on_freeze=bad.on_freeze, progress_on_freeze=good.on_freeze,
                    no_gain_final_success=bad.success, progress_final_success=good.success,
                    query_gap=abs(bad.query-good.query), baseline_max_margin=baseline, geometry_ok=ok_geom,
                    same_variant=same_variant, **geom))
    columns = ['population', 'horizon', 'benchmark', 'base_task', 'no_gain_name', 'no_gain_query',
               'progress_name', 'progress_query', 'tier', 'baseline_max_margin']
    candidate = pd.DataFrame(candidates)
    for horizon in sorted(frame.horizon.unique()):
        pool = candidate[candidate.horizon.eq(horizon)] if len(candidate) else candidate
        for tier in TIERS:
            if pool.empty:
                continue
            ok = np.ones(len(pool), bool)
            if tier != 'phase':
                ok &= pool.geometry_ok.to_numpy()
            if 'margin_' in tier:
                ok &= pool.baseline_max_margin.to_numpy() <= float(tier.rsplit('_', 1)[1])
            if tier.startswith('same_variant'):
                ok &= pool.same_variant.to_numpy()
            eligible = pool[ok].sort_values(['geometry_rms_m', 'baseline_max_margin', 'query_gap', 'progress_name', 'progress_query'])
            chosen = eligible.drop_duplicates(['no_gain_name', 'no_gain_query']).copy()
            chosen['tier'] = tier
            selected.extend(chosen.to_dict('records'))
    matches = pd.DataFrame(selected) if selected else pd.DataFrame(columns=columns)
    matches.to_csv(ROOT / (label+'-matched-pairs.csv'), index=False)
    candidate.to_csv(ROOT / (label+'-phase-candidates.csv'), index=False)
    support = []
    for horizon in sorted(frame.horizon.unique()):
        part = frame[frame.horizon.eq(horizon)]
        for tier in TIERS:
            use = matches[matches.horizon.eq(horizon) & matches.tier.eq(tier)]
            support.append(dict(population=label, horizon=int(horizon), tier=tier,
                no_gain_windows=int(part.local_label.eq('no_observed_gain').sum()),
                progress_windows=int(part.local_label.eq('verified_progress').sum()),
                matched_windows=len(use), no_gain_parents=use.no_gain_name.nunique(),
                progress_parents=use.progress_name.nunique(),
                parent_pairs=len(use[['no_gain_name', 'progress_name']].drop_duplicates()),
                base_tasks=use.base_task.nunique(), benchmark_tasks=len(use[['benchmark', 'base_task']].drop_duplicates()),
                matched_without_freeze=int((~use.no_gain_on_freeze).sum()) if len(use) else 0))
    return matches, pd.DataFrame(support)


def functional_inventory(inputs):
    early = inputs.json(EARLY / 'selection.json')
    later = inputs.json(PREVIOUS / 'functional-selection.json')
    proposed = [dict(name=p['name'], query=p['query'], source='early',
                    directory=str(EARLY / 'controlled-probe' / p['name'])) for p in early['parents']]
    proposed.extend(dict(name=p['name'], query=p['probe_query'], source='later_'+p['stage'],
                    directory=str(PREVIOUS / 'functional' / ('probe-%02d' % p['probe_id']))) for p in later['probes'])
    unique, duplicate_rows = {}, []
    for item in proposed:
        key = (item['name'], item['query'])
        if key in unique:
            previous = unique[key]
            errors = []
            for condition in CONDITIONS:
                a = inputs.npz(Path(previous['directory']) / (condition+'.npz'))
                b = inputs.npz(Path(item['directory']) / (condition+'.npz'))
                assert a.keys() == b.keys()
                for field in a:
                    np.testing.assert_array_equal(a[field], b[field])
                    errors.append(0.)
            duplicate_rows.append(dict(item, canonical=previous['directory'], arrays_bitexact=True))
        else:
            unique[key] = item
    frame = pd.DataFrame(unique.values())
    frame.to_csv(ROOT / 'functional-inventory.csv', index=False)
    save_json(ROOT / 'functional-deduplication.json', dict(proposed=len(proposed), unique=len(frame),
        unique_parents=frame.name.nunique(), duplicates=duplicate_rows,
        selection_bias='Earlier snapshots selected with mobility or alarm and final-outcome constraints; not representative.'))
    return frame


def functional_features(inputs, inventory, metadata):
    rows, attribution_rows, audits = [], [], []
    for anchor in inventory.itertuples(index=False):
        directory = Path(anchor.directory)
        snapshots = {c: inputs.npz(directory / (c+'.npz')) for c in CONDITIONS}
        trace_path = Path(metadata[anchor.name]['source_artifact_dir']) / 'episode-trace.npz'
        inputs.add(trace_path)
        with np.load(trace_path) as trace:
            for condition, q in (('o0n0', anchor.query-1), ('o1n1', anchor.query)):
                np.testing.assert_array_equal(snapshots[condition]['actions'], trace['predicted_actions'][q])
        for noise, left, right in ((0, 'o0n0', 'o1n0'), (1, 'o0n1', 'o1n1')):
            values, attribution, audit = contribution_contrast(snapshots[left], snapshots[right])
            audits.append(dict(name=anchor.name, query=anchor.query, noise=noise, **audit))
            for scope, mask in scope_masks((8, 11)).items():
                row = dict(name=anchor.name, query=anchor.query, noise=noise, scope=scope,
                           sites=int(mask.sum()), same_support_sites=int(values['same_support'][mask].sum()))
                for feature in FEATURES+('input_relative_change',):
                    valid = values[feature][mask & np.isfinite(values[feature])]
                    row[feature] = float(np.median(valid)) if len(valid) else np.nan
                    row[feature+'_sites'] = len(valid)
                rows.append(row)
            a_ids, b_ids = snapshots[left]['ids'], snapshots[right]['ids']
            for layer in range(8):
                for token in range(11):
                    for expert in sorted(set(a_ids[layer, token]) | set(b_ids[layer, token])):
                        attribution_rows.append(dict(name=anchor.name, query=anchor.query, noise=noise,
                            layer=int(snapshots[left]['hb_layers'][layer]), token=token, expert=int(expert),
                            status='retained' if expert in a_ids[layer, token] and expert in b_ids[layer, token]
                                else 'entered' if expert in b_ids[layer, token] else 'exited',
                            signed_update_attribution=float(attribution[layer, token, expert])))
        print('Functional anchor %s q%d' % (anchor.name, anchor.query), flush=True)
    contrasts = pd.DataFrame(rows)
    contrasts.to_csv(ROOT / 'functional-contrasts.csv', index=False)
    pd.DataFrame(attribution_rows).to_csv(ROOT / 'expert-attribution.csv', index=False)
    pd.DataFrame(audits).to_csv(ROOT / 'functional-algebra-audit.csv', index=False)
    anchors = contrasts.groupby(['name', 'query', 'scope'], as_index=False)[list(FEATURES)].mean()
    anchors.to_csv(ROOT / 'functional-anchor-features.csv', index=False)
    save_json(ROOT / 'functional-verification.json', dict(anchors=len(inventory), contrasts=len(audits),
        natural_actions_exact=True, decomposition_max_absolute=max(x['decomposition_max_absolute'] for x in audits),
        attribution_max_absolute=max(x['attribution_max_absolute'] for x in audits),
        stored_update_rounding_relative=max(x['stored_update_rounding_relative'] for x in audits),
        new_model_inferences=0))
    return anchors


def main():
    inputs = Inputs()
    inputs.add(ROOT / 'PROTOCOL.md')
    prior_index = inputs.csv(BASE / 'samples/moe-offline-sequences-20260915/episode-events.csv')
    assert len(prior_index) == 33043
    save_json(ROOT / 'population.json', dict(previous_complete_records=len(prior_index),
        previous_cohorts=prior_index.groupby('cohort').size().to_dict(), native_physical_parents=80,
        prior_routes_reused_for_new_physics=False, new_environment_actions=0, new_model_inferences=0))
    physics, metadata, queries = load_data(inputs)
    frame, primary, joined = build_windows(physics, queries)
    eligible = frame[(frame['query'] >= 10) & frame.complete_window & frame.all5_available]
    all_matches, all_support = match_pairs(eligible, physics, 'native')
    inventory = functional_inventory(inputs)
    captured = frame.merge(inventory[['name', 'query']], on=['name', 'query'], how='inner')
    captured.to_csv(ROOT / 'functional-anchor-context.csv', index=False)
    captured_eligible = captured[(captured['query'] >= 10) & captured.complete_window & captured.all5_available]
    function_matches, function_support = match_pairs(captured_eligible, physics, 'functional')
    support = pd.concat([all_support, function_support], ignore_index=True)
    support.to_csv(ROOT / 'matching-support.csv', index=False)
    features = functional_features(inputs, inventory, metadata)
    context = captured[captured.horizon.eq(3)]
    described = features.merge(context, on=['name', 'query'], how='left')
    described.to_csv(ROOT / 'functional-features-with-context.csv', index=False)
    described.groupby(['scope', 'local_label'])[list(FEATURES)].median().reset_index().to_csv(
        ROOT / 'functional-selected-case-medians.csv', index=False)
    differences = []
    lookup = features.set_index(['name', 'query', 'scope'])
    for pair in function_matches[function_matches.horizon.eq(3)].itertuples(index=False):
        for scope in ('back_action', 'front_action', 'back_state', 'front_state'):
            bad = lookup.loc[(pair.no_gain_name, pair.no_gain_query, scope)]
            good = lookup.loc[(pair.progress_name, pair.progress_query, scope)]
            for feature in FEATURES:
                differences.append(dict(tier=pair.tier, no_gain_name=pair.no_gain_name,
                    no_gain_query=pair.no_gain_query, progress_name=pair.progress_name,
                    progress_query=pair.progress_query, base_task=pair.base_task, scope=scope,
                    feature=feature, no_gain_value=bad[feature], progress_value=good[feature],
                    difference=bad[feature]-good[feature]))
    pd.DataFrame(differences, columns=['tier', 'no_gain_name', 'no_gain_query', 'progress_name',
        'progress_query', 'base_task', 'scope', 'feature', 'no_gain_value', 'progress_value', 'difference']).to_csv(
        ROOT / 'functional-paired-differences.csv', index=False)
    save_json(ROOT / 'input-hashes.json', inputs.hashes)
    moving = joined[joined.local_label.eq('no_observed_gain') &
        joined.category.isin(['return_motion', 'other_motion_no_gain'])]
    nongain = joined[joined.local_label.eq('no_observed_gain')]
    late = nongain[nongain.h10_label.notna()]
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=digest(ROOT / 'PROTOCOL.md'),
        physical_parents=len(physics), boundary_states=sum(len(d['predicates']) for d in physics.values()),
        primary_windows=len(primary), primary_parents=primary.name.nunique(),
        local_labels=primary.local_label.value_counts().to_dict(), categories=primary.category.value_counts().to_dict(),
        moving_no_gain_windows=len(moving), moving_no_gain_parents=moving.name.nunique(),
        moving_no_gain_without_freeze_windows=int((~moving.on_freeze).sum()),
        moving_no_gain_without_freeze_parents=moving[~moving.on_freeze].name.nunique(),
        moving_no_gain_without_any_alarm_windows=int((~moving.v82_on).sum()),
        moving_no_gain_without_any_alarm_parents=moving[~moving.v82_on].name.nunique(),
        no_gain_h10_observed=len(late), no_gain_h10_later_verified_progress=int(late.h10_label.eq('verified_progress').sum()),
        static_translation_but_internal_motion=int(primary.translation_static_but_internal_motion.sum()),
        functional_unique_anchors=len(inventory), functional_unique_parents=inventory.name.nunique(),
        functional_h3_eligible_anchors=int(captured_eligible.horizon.eq(3).sum()),
        matching_support=support[support.horizon.eq(3)].to_dict('records'),
        inputs=len(inputs.hashes), new_model_inferences=0, new_environment_actions=0,
        independent_validation=False, local_trap_ground_truth=False)
    save_json(ROOT / 'results.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
