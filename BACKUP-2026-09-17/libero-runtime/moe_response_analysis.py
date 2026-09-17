"""Offline functional decomposition and fixed selection of observation probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/data/coding/v8-methods')
from moe_joint_patterns import INDEX

ROOT = Path('/data/libero-runtime/samples/moe-input-response-20260915')
OLD = Path('/data/libero-runtime/samples/moe-joint-patterns-20260915')
FUNCTIONAL = OLD / 'functional-probe'
SOURCE = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def norm(x):
    return np.linalg.norm(x, axis=-1)


def relative_change(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return norm(b-a) / np.maximum((norm(a)+norm(b)) / 2, 1e-12)


def factorial_energy(f00, f01, f10, f11):
    """Balanced 2x2 finite-contrast energy, retaining the interaction term."""
    f00, f01, f10, f11 = [np.asarray(x, float) for x in (f00, f01, f10, f11)]
    observation = (f10+f11-f00-f01) / 4
    noise = (f01+f11-f00-f10) / 4
    interaction = (f00+f11-f01-f10) / 4
    energy = np.stack([(x*x).sum(-1) for x in (observation, noise, interaction)], axis=-1)
    total = energy.sum(-1)
    fraction = np.divide(energy, total[..., None], out=np.full_like(energy, np.nan),
                         where=total[..., None] > 1e-20)
    center = (f00+f01+f10+f11) / 4
    measured = sum(((x-center)**2).sum(-1) for x in (f00, f01, f10, f11)) / 4
    if not np.allclose(total, measured, rtol=1e-10, atol=1e-12):
        raise ValueError('factorial energy decomposition mismatch')
    return dict(energy=energy, fraction=fraction, total_energy=total,
                natural_observation_term=2*observation, natural_noise_term=2*noise)


def align_experts(ids, weights, contributions):
    ids = np.asarray(ids)
    weights, contributions = np.asarray(weights, float), np.asarray(contributions, float)
    if contributions.shape[:-1] != ids.shape or weights.shape != ids.shape:
        raise ValueError('expert shape mismatch')
    if (weights <= 0).any() or not np.isfinite(weights).all() or not np.isfinite(contributions).all():
        raise ValueError('expert outputs require positive finite execution weights')
    order = ids.argsort(axis=-1)
    ordered = np.take_along_axis(ids, order, axis=-1)
    if np.any(np.diff(ordered, axis=-1) == 0):
        raise ValueError('duplicate selected expert ID')
    return (ordered, np.take_along_axis(weights, order, axis=-1),
            np.take_along_axis(contributions, order[..., None], axis=-2))


def symmetric_decomposition(ids0, w0, c0, ids1, w1, c1):
    """Exact two-factor identity on the shared support, with expert ID alignment."""
    i0, w0, c0 = align_experts(ids0, w0, c0)
    i1, w1, c1 = align_experts(ids1, w1, c1)
    same = (i0 == i1).all(axis=-1)
    e0, e1 = c0 / w0[..., None], c1 / w1[..., None]
    weight_term = ((w1-w0)[..., None] * (e0+e1) / 2).sum(axis=-2)
    expert_term = ((w0+w1)[..., None] / 2 * (e1-e0)).sum(axis=-2)
    delta = c1.sum(axis=-2) - c0.sum(axis=-2)
    residual = norm(delta - weight_term - expert_term)
    share = norm(expert_term) / np.maximum(norm(expert_term)+norm(weight_term), 1e-12)
    for value in (weight_term, expert_term):
        value[~same] = np.nan
    share = np.where(same & (norm(delta) > 1e-12), share, np.nan)
    residual = np.where(same, residual, np.nan)
    return dict(same_support=same, weight_term=weight_term, expert_term=expert_term,
                observed_delta=delta, expert_term_norm_fraction=share, algebra_residual=residual,
                weight_l1=np.where(same, np.abs(w1-w0).sum(axis=-1), np.nan))


def scope_masks(shape):
    if shape[0] != 8 or shape[-1] != 11:
        raise ValueError('expected [8, ..., 11] HB sites')
    layer = np.arange(8).reshape((8,) + (1,) * (len(shape)-1))
    token = np.arange(11).reshape((1,) * (len(shape)-1) + (11,))
    return {f'{depth}_{kind}': np.broadcast_to((layer < 4 if depth == 'front' else layer >= 4)
                                              & (token == 0 if kind == 'state' else token > 0), shape)
            for depth in ('front', 'back') for kind in ('state', 'action')}


def functional_comparison(a, b, raw_experts=False):
    c0, c1 = a['expert_contrib'], b['expert_contrib']
    if raw_experts:
        c0 = c0 * a['weights'][..., None]
        c1 = c1 * b['weights'][..., None]
    d = symmetric_decomposition(a['ids'], a['weights'], c0, b['ids'], b['weights'], c1)
    r0, r1, s0, s1 = a['routed'], b['routed'], a['shared'], b['shared']
    total0, total1 = a.get('total', r0+s0), b.get('total', r1+s1)
    dr, ds = r1-r0, s1-s0
    cosine = (dr*ds).sum(axis=-1) / np.maximum(norm(dr)*norm(ds), 1e-12)
    d.update(routed_relative_change=relative_change(r0, r1),
             shared_relative_change=relative_change(s0, s1),
             total_relative_change=relative_change(total0, total1),
             routed_shared_change_cosine=cosine,
             shared_change_norm_fraction=norm(ds) / np.maximum(norm(dr)+norm(ds), 1e-12),
             routed_reconstruction_relative=norm(d['observed_delta']-dr)
             / np.maximum((norm(r0)+norm(r1))/2, 1e-12))
    if 'input' in a:
        change = relative_change(a['input'], b['input'])
        d['input_relative_change'] = change
        for name in ('routed', 'shared', 'total'):
            d[name + '_relative_gain'] = np.divide(
                d[name + '_relative_change'], change, out=np.full_like(change, np.nan), where=change > 1e-4)
    return d


def summarized_comparison(values, metadata):
    rows = []
    for scope, selected in scope_masks(values['same_support'].shape).items():
        matched = selected & values['same_support']
        row = dict(**metadata, scope=scope, sites=int(selected.sum()), same_support_sites=int(matched.sum()))
        for key, value in values.items():
            if value.shape != selected.shape or key == 'same_support':
                continue
            use = matched if key in ('expert_term_norm_fraction', 'weight_l1', 'algebra_residual') else selected
            finite = value[use & np.isfinite(value)]
            row[key] = float(np.median(finite)) if len(finite) else np.nan
        row['expert_term_dominant_sites'] = int((matched & (values['expert_term_norm_fraction'] > .5)).sum())
        row['nearly_unchanged_weight_sites'] = int((matched & (values['weight_l1'] <= .01)).sum())
        if 'input_relative_change' in values:
            row['near_zero_input_change_sites'] = int((selected & (values['input_relative_change'] <= 1e-4)).sum())
            row['valid_response_gain_sites'] = int((selected & np.isfinite(values['routed_relative_gain'])).sum())
        stable = matched & (values['weight_l1'] <= .01)
        row['stable_weight_routed_change_median'] = float(np.median(values['routed_relative_change'][stable])) if stable.any() else np.nan
        rows.append(row)
    return rows


def sketch_snapshot(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k][0].astype(float) for k in (
            'expert_contrib_sketch', 'routed_output_sketch', 'shared_output_sketch', 'topk_exec_weight')}
        ids = z['topk_idx'][0]
    return dict(ids=ids, weights=data['topk_exec_weight'], expert_contrib=data['expert_contrib_sketch'],
                routed=data['routed_output_sketch'], shared=data['shared_output_sketch'])


def existing_decomposition():
    sources = json.loads((FUNCTIONAL / 'protocol.json').read_text())['samples']
    episodes = {e['name']: e for e in json.loads((SOURCE / 'summary.json').read_text())['episodes']}
    features = np.load(OLD / 'episode-features.npz')
    external = np.load(OLD / 'offline-external.npz')
    rows, source_hashes = [], {}
    for sample in sources:
        name, queries = sample['episode'], sample['queries']
        snapshots = []
        for q in queries:
            path = FUNCTIONAL / f'{name}-q{q:03d}.npz'
            source_hashes[str(path)] = digest(path)
            snapshot = sketch_snapshot(path)
            snapshots.append(snapshot)
            previous = {k: value[:, :-1] for k, value in snapshot.items()}
            current = {k: value[:, 1:] for k, value in snapshot.items()}
            values = functional_comparison(previous, current)
            metadata = dict(episode=name, success=episodes[name]['source_result']['success'],
                            comparison='within_flow', query0=q, query1=q, query_gap=0,
                            mobility_log_ratio=features[name][q, INDEX['mobility_log_ratio']],
                            eef_next_displacement_m=external[name][q, 0])
            rows.extend(summarized_comparison(values, metadata))
        previous = {k: value[:, -1] for k, value in snapshots[0].items()}
        current = {k: value[:, -1] for k, value in snapshots[1].items()}
        values = functional_comparison(previous, current)
        metadata = dict(episode=name, success=episodes[name]['source_result']['success'],
                        comparison='between_queries_uncontrolled_noise', query0=queries[0], query1=queries[1],
                        query_gap=queries[1]-queries[0],
                        mobility_log_ratio=features[name][queries[1], INDEX['mobility_log_ratio']],
                        eef_next_displacement_m=external[name][queries[1], 0])
        rows.extend(summarized_comparison(values, metadata))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / 'existing-snapshot-decomposition.csv', index=False)
    parent = table.groupby(['comparison', 'scope', 'episode', 'success'], as_index=False).mean(numeric_only=True)
    parent.to_csv(ROOT / 'existing-parent-summary.csv', index=False)
    save_json(ROOT / 'existing-snapshot-sources.json', source_hashes)
    summary = []
    for (comparison, scope, success), group in parent.groupby(['comparison', 'scope', 'success']):
        summary.append(dict(comparison=comparison, scope=scope, success=success, parents=len(group),
                            **{key: float(group[key].dropna().median()) if group[key].notna().any() else np.nan for key in (
                                'expert_term_norm_fraction', 'routed_relative_change', 'shared_relative_change',
                                'total_relative_change', 'routed_reconstruction_relative',
                                'stable_weight_routed_change_median')}))
    save_json(ROOT / 'existing-results.json', dict(snapshots=12, parents=6, projected_dimensions=16,
                                                 summaries=summary, new_model_forwards=0))
    print(table[table.scope.eq('back_action')].to_string(index=False), flush=True)


def select_pairs():
    episodes = json.loads((SOURCE / 'summary.json').read_text())['episodes']
    episodes = [e for e in episodes if e['benchmark'] == 'plus']
    feature_path = OLD / 'episode-features.npz'
    features = np.load(feature_path)
    external = np.load(OLD / 'offline-external.npz')
    pairs, parents = [], []
    for task in sorted({e['base_task_id'] for e in episodes}):
        good = [e for e in episodes if e['base_task_id'] == task and e['source_result']['success']]
        bad = [e for e in episodes if e['base_task_id'] == task and not e['source_result']['success']]
        if not good or not bad:
            continue
        candidates = []
        for success in good:
            for failure in bad:
                for q in range(9, min(31, success['queries'], failure['queries'])):
                    sm = features[success['name']][q, INDEX['mobility_log_ratio']]
                    fm = features[failure['name']][q, INDEX['mobility_log_ratio']]
                    gap = abs(sm-fm)
                    candidates.append(dict(task=task, success=success['name'], failure=failure['name'], query=q,
                                           success_mobility=sm, failure_mobility=fm, mobility_gap=gap,
                                           stable_matched=bool(max(sm, fm) <= -np.log(1.1) and gap <= .1)))
        eligible = [c for c in candidates if c['stable_matched']]
        if eligible:
            selected = min(eligible, key=lambda c: (max(c['success_mobility'], c['failure_mobility']),
                                                   c['mobility_gap'], c['query'], c['success'], c['failure']))
        else:
            selected = min(candidates, key=lambda c: (c['mobility_gap'], c['query'], c['success'], c['failure']))
        selected['pair_id'] = len(pairs)
        pairs.append(selected)
        for outcome in ('success', 'failure'):
            name = selected[outcome]
            e = next(e for e in episodes if e['name'] == name)
            q = selected['query']
            trace_dir = Path(e['source_artifact_dir'])
            trace = json.loads((trace_dir / 'episode-trace.json').read_text())
            trace_path = trace_dir / trace['array_file']
            if digest(trace_path) != e['source_trace_sha256']:
                raise ValueError('source trace mismatch')
            parents.append(dict(parent_id=len(parents), pair_id=selected['pair_id'], task=task, name=name,
                                query=q, success=outcome == 'success', source_dir=str(trace_dir),
                                trace_sha256=e['source_trace_sha256'], source_hb_sha256=e['integrity']['full_hb_sha256'],
                                prompt=e['prompt'], policy_identity=e['policy_identity'],
                                stable_matched=selected['stable_matched'],
                                mobility_log_ratio=features[name][q, INDEX['mobility_log_ratio']],
                                eef_next_displacement_m=external[name][q, 0],
                                eef_previous_displacement_m=external[name][q-1, 0],
                                source_success=e['source_result']['success']))
    if len({p['name'] for p in parents}) != len(parents):
        raise ValueError('selection reuses a parent episode')
    selection = dict(schema='local.moe_input_response.selection.v1', pairs=pairs, parents=parents,
                     source_features_sha256=digest(feature_path), selection_uses_new_functional_response=False,
                     maximum_model_calls=len(parents)*5+2, independent_blind_test=False)
    save_json(ROOT / 'selection.json', selection)
    pd.DataFrame(pairs).to_csv(ROOT / 'selected-pairs.csv', index=False)
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['existing', 'select'])
    args = parser.parse_args()
    if args.command == 'existing':
        existing_decomposition()
    else:
        select_pairs()


if __name__ == '__main__':
    main()
