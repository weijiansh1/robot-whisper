"""Reproduce fixed, available MoE motifs across restored historical cohorts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, '/data/coding/v8-methods')
from moe_joint_patterns import INDEX, MOTIF_NAMES, JointRoutingEncoder, motif_scores

ROOT = Path('/data/libero-runtime/samples/moe-joint-expanded-20260915')
INPUTS = ROOT / 'inputs'
RAW = INPUTS / 'moe-v7-legacy16x32-0906/results/raw_features'
ENCODED = INPUTS / 'safe&vlaconf/moe_trainfree/results/routing_dynamics_20260908'
OLD = Path('/data/libero-runtime/samples/moe-joint-patterns-20260915')
RECENT = Path('/data/libero-runtime/samples/v82-evaluation-20260914T144322Z')
PILOT = Path('/data/coding/v8-signal-test-CTUaAJ')
NAMES = ('freeze_control', 'frozen_active', 'churn_to_cooling', 'both_cooling',
         'both_active', 'outer_active_inner_cooling', 'acceleration_control',
         'supported_joint_union')
RANK_NAMES = ('freeze_control', 'frozen_active', 'both_cooling', 'both_active',
              'outer_active_inner_cooling', 'acceleration_control',
              'mobility_decrease_instant', 'acceleration_increase_instant')
HORIZONS = {'libero_spatial': 220, 'libero_goal': 300,
            'libero_object': 280, 'libero_long': 520}
SEED = 20260915


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
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


def encode_two(mobility, acceleration, valid):
    """Available encoder coordinates only; unsupported coordinates are not imputed."""
    m = np.array(mobility, dtype=np.float64, copy=True)
    a = np.array(acceleration, dtype=np.float64, copy=True)
    valid = np.asarray(valid, bool)
    if m.shape != (*valid.shape, 8) or a.shape != valid.shape:
        raise ValueError('readout shape mismatch')
    if np.any(valid[:, 1:] & ~valid[:, :-1]):
        raise ValueError('noncontiguous observed prefix')
    if not np.isfinite(m[:, 1:][valid[:, 1:]]).all() or not np.isfinite(a[valid]).all():
        raise ValueError('nonfinite observed readout')
    if (m[:, 1:][valid[:, 1:]] < 0).any() or (a[valid] < 0).any():
        raise ValueError('negative observed readout')
    m[~valid], a[~valid] = np.nan, np.nan
    out = np.full((*valid.shape, 2), np.nan)
    if valid.shape[1] <= 7:
        return out
    reference_m = np.median(m[:, 1:5], axis=1)
    reference_a = np.median(a[:, 1:5], axis=1)
    for q in range(7, valid.shape[1]):
        mr = np.log((m[:, q-2:q+1].mean(1) + 1e-6) / (reference_m + 1e-6))
        ar = np.log((a[:, q-2:q+1].mean(1) + 1e-6) / (reference_a + 1e-6))
        mr[np.abs(mr) <= 32 * np.finfo(float).eps] = 0
        ar[np.abs(ar) <= 32 * np.finfo(float).eps] = 0
        out[:, q, 0], out[:, q, 1] = np.median(mr[:, 4:], axis=1), ar
    out[~valid] = np.nan
    return out


def trailing_min(x):
    out = np.full_like(x, np.nan)
    for q in range(2, x.shape[1]):
        out[:, q] = x[:, q-2:q+1].min(axis=1)
    return out


def score_two(x):
    m, a = x[..., 0], x[..., 1]
    freeze = trailing_min(-m)
    active = trailing_min(np.minimum(-m, a))
    cool = trailing_min(np.minimum(-m, -a))
    churn = trailing_min(np.minimum(m, a))
    outer = trailing_min(np.minimum(m, -a))
    accel = trailing_min(a)
    transition = np.full_like(m, np.nan)
    for q in range(3, m.shape[1]):
        prior = churn[:, max(0, q-6):q-2]
        finite = np.isfinite(prior).any(axis=1) & np.isfinite(cool[:, q])
        best = np.where(np.isfinite(prior), prior, -np.inf).max(axis=1)
        transition[finite, q] = np.minimum(cool[finite, q], best[finite])
    union = np.fmax(active, transition)
    return np.stack((freeze, active, transition, cool, churn, outer, accel, union), axis=-1)


def first_alarms(scores, factor):
    hits = np.isfinite(scores) & (scores >= np.log(factor))
    return np.where(hits.any(axis=1), hits.argmax(axis=1), -1).astype(np.int16)


def unique_lookup(frame, keys):
    if frame.duplicated(keys).any():
        raise ValueError(f'duplicate metadata identity: {keys}')
    return frame.set_index(keys, verify_integrity=True)


def load_historical():
    index = pd.read_csv(ENCODED / 'index.csv').fillna({'primary_failure_reason': ''})
    lookup = unique_lookup(index, ['run_id', 'task', 'episode'])
    baseline = pd.read_csv(INPUTS / 'moe-trap-control/design/frozen_alarm_comparison_20260908/first_alarms.csv')
    baseline = unique_lookup(baseline, ['run_id', 'task', 'episode'])
    legacy = pd.read_csv(INPUTS / 'moe-v7-legacy16x32-0906/results/legacy16x32/episode_alarms.csv')
    legacy = unique_lookup(legacy, ['task', 'episode'])
    frames, coordinates, hashes, audit = [], [], [], []
    for cohort in ('development_main', 'external_8b', 'legacy_16x32'):
        with np.load(RAW / f'{cohort}_route_features.npz', allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        run = str(data['run_id'])
        rows = []
        for r, (ti, ep, length) in enumerate(zip(data['task_index'], data['episode'], data['length'])):
            task = str(data['task_names'][ti])
            if cohort == 'legacy_16x32':
                old = legacy.loc[(task, int(ep))]
                metadata = dict(init_state_id=int(old.init_state_id), noise_seed=int(old.flow_noise_seed),
                                failure=bool(old.original_failure), checkpoint='unverified_legacy',
                                action_steps=-1, primary_failure_reason='', v82=-2)
                if int(old.horizon_cap) * 10 != HORIZONS[old.suite]:
                    raise ValueError('legacy horizon mismatch')
            else:
                old = lookup.loc[(run, task, int(ep))]
                b = baseline.loc[(run, task, int(ep))]
                if int(b.length) != int(length) or bool(b.failure) != bool(old.failure):
                    raise ValueError('baseline metadata mismatch')
                metadata = {k: old[k] for k in ('init_state_id', 'noise_seed', 'failure', 'checkpoint',
                                               'action_steps', 'primary_failure_reason')}
                metadata['v82'] = int(b.v82_frozen)
            if int(old.length) != int(length):
                raise ValueError('outcome/readout length mismatch')
            if int(data['valid'][r].sum()) != int(length):
                raise ValueError('validity/length mismatch')
            suite = task.split('/')[0]
            if bool(metadata['failure']) and int(length) * 10 != HORIZONS[suite]:
                raise ValueError('failure horizon mismatch')
            identity = f'{run}|{task}|{int(ep)}'
            rows.append(dict(identity=identity, cohort=cohort, run_id=run, task=task, suite=suite,
                             episode=int(ep), length=int(length), horizon_actions=HORIZONS[suite],
                             source_row=r, **metadata))
            h = hashlib.sha256()
            h.update(data['mobility'][r, :length].tobytes())
            h.update(data['route_acceleration'][r, :length].tobytes())
            hashes.append(h.hexdigest())
        frame = pd.DataFrame(rows)
        coords = encode_two(data['mobility'], data['route_acceleration'], data['valid'])
        frames.append(frame)
        coordinates.append(coords)
        audit.append(dict(cohort=cohort, episodes=len(frame), queries=int(frame.length.sum()),
                          failures=int(frame.failure.sum()), tasks=frame.task.nunique(),
                          hash_verified=True, metadata_join_verified=True))
        print(f'loaded {cohort}: {len(frame)} episodes', flush=True)
    frame = pd.concat(frames, ignore_index=True)
    frame['readout_fingerprint'] = hashes
    if frame.identity.duplicated().any():
        raise ValueError('duplicate run/task/episode')
    duplicate = frame[frame.duplicated(['task', 'readout_fingerprint'], keep=False)]
    duplicate.to_csv(ROOT / 'exact-readout-duplicates.csv', index=False)
    if len(duplicate):
        raise ValueError('exact duplicate readouts require provenance resolution')
    shared = frame[frame.duplicated(['task', 'init_state_id', 'noise_seed'], keep=False)]
    shared.to_csv(ROOT / 'repeated-configurations.csv', index=False)
    return frame, np.concatenate(coordinates), audit


def historical_extractor():
    sys.path.insert(0, '/data/coding/robot-whisper-0909/moe-v7-0905/method')
    path = INPUTS / 'moe-v7-legacy16x32-0906/experiments/raw_route_features.py'
    spec = importlib.util.spec_from_file_location('historical_raw_extractor', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.episode_features


def load_current():
    old_features = np.load(OLD / 'episode-features.npz', allow_pickle=False)
    current = json.loads((RECENT / 'summary.json').read_text())['episodes']
    pilots = json.loads((PILOT / 'summary.json').read_text())['episodes']
    extract = historical_extractor()
    rows, coords, checks, pilot_features, trace_ids = [], [], [], {}, []
    for pilot, episodes in ((False, current), (True, pilots)):
        for e in episodes:
            trace_dir = Path(e['source_artifact_dir'])
            trace = json.loads((trace_dir / 'episode-trace.json').read_text())
            if e['source_trace_sha256'] != trace['array_file_sha256']:
                raise ValueError('trace manifest identity mismatch')
            if digest(trace_dir / trace['array_file']) != e['source_trace_sha256']:
                raise ValueError('trace content mismatch')
            trace_ids.append(e['source_trace_sha256'])
            raw_path = (PILOT if pilot else RECENT) / e['name'] / 'full-hb-routes.npz'
            with np.load(raw_path, allow_pickle=False) as z:
                raw = z['hb_router_probs']
                if pilot:
                    encoder = JointRoutingEncoder()
                    full = np.asarray([encoder.update(p, ids)[0]
                                       for p, ids in zip(raw, z['hb_expert_ids'])])
                    pilot_features[e['name']] = full
                else:
                    full = old_features[e['name']]
            if len(raw) != e['queries'] or len(full) != len(raw):
                raise ValueError('raw/feature length mismatch')
            m, a, _ = extract(raw)
            local = encode_two(m[None], a[None], np.ones((1, len(raw)), bool))[0]
            exact = full[:, [INDEX['mobility_log_ratio'], INDEX['acceleration_log_ratio']]]
            delta = float(np.nanmax(np.abs(local - exact)))
            if not np.array_equal(np.isnan(local), np.isnan(exact)) or delta > 1e-4:
                raise ValueError(f'historical/current encoding mismatch {e["name"]}: {delta}')
            joint = motif_scores(full)[:, [0, 5]]
            rebuilt = score_two(local[None])[0][:, [1, 2]]
            exact_two = score_two(exact[None])[0][:, [1, 2]]
            if not np.allclose(exact_two, joint, equal_nan=True, atol=1e-12, rtol=0):
                raise ValueError('batch/frozen motif mismatch')
            disagreements = sum(int(np.count_nonzero(first_alarms(rebuilt[None], f)
                                                     != first_alarms(joint[None], f)))
                                for f in (1.1, 1.2, 1.3))
            if disagreements:
                raise ValueError('finite precision changed current alarm first crossing')
            checks.append(dict(episode=e['name'], pilot=pilot, queries=len(raw),
                               max_abs_error=delta, alarm_disagreements=disagreements,
                               raw_sha256=digest(raw_path)))
            benchmark = e['name'] if pilot else e['benchmark']
            task = trace['task_name']
            config = trace['config']
            rows.append(dict(identity=('pilot|' if pilot else 'current|') + e['name'],
                             cohort=('pilot_' if pilot else 'current_') + benchmark,
                             run_id='pilot' if pilot else 'current_60', task=task,
                             suite='libero_long', episode=e['episode_id'], length=e['queries'],
                             horizon_actions=config['max_steps'], source_row=len(rows),
                             init_state_id=config['init_state_id'], noise_seed=config['flow_noise_seed'],
                             failure=not e['source_result']['success'],
                             checkpoint=e['policy_identity']['checkpoint_sha256'],
                             action_steps=e['source_result']['action_steps'], primary_failure_reason='',
                             v82=e['first_alarm_query_zero_based']['v82'],
                             readout_fingerprint='', trace_sha256=e['source_trace_sha256']))
            padded = np.full((52, 2), np.nan)
            padded[:len(raw)] = exact
            coords.append(padded)
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError('duplicate source traces')
    pd.DataFrame(checks).to_csv(ROOT / 'raw-reconstruction-verification.csv', index=False)
    np.savez_compressed(ROOT / 'pilot-joint-features.npz', **pilot_features)
    return pd.DataFrame(rows), np.asarray(coords), checks


def event_metrics(frame, scores):
    rows, incremental = [], []
    y = frame.failure.to_numpy(bool)
    horizons = frame.horizon_actions.to_numpy(float)
    baseline = frame.v82.to_numpy(int)
    for factor in (1.1, 1.2, 1.3):
        first = first_alarms(scores, factor)
        if factor == 1.2:
            for k, name in enumerate(NAMES):
                frame['first_' + name] = first[:, k]
            peak = np.where(np.isfinite(scores), scores, -np.inf).max(axis=1)
            margin = np.min(np.abs(peak - np.log(factor)), axis=1)
            ambiguous = frame.loc[margin <= 1e-4, ['identity', 'cohort', 'failure']].copy()
            ambiguous['minimum_peak_threshold_margin'] = margin[margin <= 1e-4]
            ambiguous.to_csv(ROOT / 'threshold-nearby-episodes.csv', index=False)
        for (cohort, suite), group in frame.groupby(['cohort', 'suite']):
            ids = group.index.to_numpy()
            for k, name in enumerate((*NAMES, 'v82_archived')):
                alarm = first[:, k] if k < len(NAMES) else baseline
                use = ids[alarm[ids] != -2]
                hit = alarm[use] >= 0
                fail = y[use]
                fired_fail = use[hit & fail]
                fraction = 10 * alarm[fired_fail] / horizons[fired_fail]
                rows.append(dict(cohort=cohort, suite=suite, factor=factor, method=name,
                                 episodes=len(use), failures=int(fail.sum()), successes=int((~fail).sum()),
                                 detected_failures=int((hit & fail).sum()),
                                 flagged_successes=int((hit & ~fail).sum()),
                                 median_failure_alarm_fraction=np.median(fraction) if len(fraction) else np.nan,
                                 q25_failure_alarm_fraction=np.quantile(fraction, .25) if len(fraction) else np.nan,
                                 q75_failure_alarm_fraction=np.quantile(fraction, .75) if len(fraction) else np.nan))
            if factor == 1.2:
                use = ids[baseline[ids] != -2]
                for name in ('frozen_active', 'churn_to_cooling', 'supported_joint_union'):
                    new = first[:, NAMES.index(name)]
                    additional = (baseline[use] < 0) & (new[use] >= 0)
                    incremental.append(dict(cohort=cohort, suite=suite, method=name,
                                            episodes=len(use), failures=int(y[use].sum()),
                                            successes=int((~y[use]).sum()),
                                            extra_failures=int((additional & y[use]).sum()),
                                            extra_successes=int((additional & ~y[use]).sum())))
    pd.DataFrame(rows).to_csv(ROOT / 'event-metrics.csv', index=False)
    pd.DataFrame(incremental).to_csv(ROOT / 'v82-incremental.csv', index=False)
    return rows


def auc(labels, values):
    positive = int(np.count_nonzero(labels))
    negative = len(labels) - positive
    if not positive or not negative:
        return np.nan
    ranks = rankdata(values)
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def interval(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(SEED)
    boot = values[rng.integers(0, len(values), size=(2000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def conditional_associations(frame, coordinates, scores):
    features = np.stack([scores[..., NAMES.index(n)] if n in NAMES else
                         (-coordinates[..., 0] if n == 'mobility_decrease_instant' else coordinates[..., 1])
                         for n in RANK_NAMES], axis=-1)
    y = frame.failure.to_numpy(bool)
    details = []
    for (cohort, task), group in frame.groupby(['cohort', 'task']):
        ids = group.index.to_numpy()
        if not y[ids].any() or y[ids].all():
            continue
        for q in (*range(9, 14), *range(18, 27)):
            for k, name in enumerate(RANK_NAMES):
                use = ids[np.isfinite(features[ids, q, k])]
                if not len(use) or not y[use].any() or y[use].all():
                    continue
                value = auc(y[use], features[use, q, k])
                initial_values = []
                for _, same in frame.loc[use].groupby('init_state_id'):
                    match = same.index.to_numpy()
                    initial_value = auc(y[match], features[match, q, k])
                    if np.isfinite(initial_value):
                        initial_values.append(initial_value)
                details.append(dict(cohort=cohort, task=task, suite=group.suite.iloc[0], query=q,
                                    window='early_q9_13' if q <= 13 else 'middle_q18_26', method=name,
                                    failures=int(y[use].sum()), successes=int((~y[use]).sum()),
                                    auc=value, same_initial_auc=np.mean(initial_values) if initial_values else np.nan,
                                    mixed_initial_states=len(initial_values)))
    details = pd.DataFrame(details)
    details.to_csv(ROOT / 'same-query-comparisons.csv', index=False)
    taskwise = details.groupby(['cohort', 'suite', 'task', 'window', 'method'], as_index=False).agg(
        auc=('auc', 'mean'), same_initial_auc=('same_initial_auc', 'mean'), queries=('query', 'size'))
    taskwise.to_csv(ROOT / 'task-associations.csv', index=False)
    summaries = []
    for (cohort, window, name), group in taskwise.groupby(['cohort', 'window', 'method']):
        for matching, column in (('task_query', 'auc'), ('task_initial_query', 'same_initial_auc')):
            valid = group[column].dropna()
            mean, low, high = interval(valid)
            summaries.append(dict(cohort=cohort, window=window, method=name, matching=matching,
                                  tasks=len(valid), auc=mean, ci_low=low, ci_high=high))
    pd.DataFrame(summaries).to_csv(ROOT / 'association-summary.csv', index=False)
    return summaries


def completion_matched(frame, scores):
    names = (*NAMES, 'v82_archived')
    first = np.column_stack([frame['first_' + name] for name in NAMES] + [frame.v82])
    y = frame.failure.to_numpy(bool)
    length = frame.length.to_numpy(int)
    peak = np.maximum.accumulate(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    rows, support = [], []
    for (cohort, task), task_group in frame.groupby(['cohort', 'task']):
        accum = {name: dict(pairs=0, fail_hits=0, success_hits=0, failure_only=0,
                           success_only=0, comparable_scores=0, concordant=0.) for name in names}
        raw_pairs, excluded = 0, 0
        for _, group in task_group.groupby('init_state_id'):
            ids = group.index.to_numpy()
            p, n = ids[y[ids]], ids[~y[ids]]
            if not len(p) or not len(n):
                continue
            pp, nn = np.meshgrid(p, n, indexing='ij')
            pp, nn = pp.ravel(), nn.ravel()
            limit = np.minimum(length[pp], length[nn]) - 1
            raw_pairs += len(limit)
            keep = limit >= 9
            excluded += int((~keep).sum())
            pp, nn, limit = pp[keep], nn[keep], limit[keep]
            for k, name in enumerate(names):
                available = (first[pp, k] != -2) & (first[nn, k] != -2)
                pi, ni, end = pp[available], nn[available], limit[available]
                ph = (first[pi, k] >= 0) & (first[pi, k] <= end)
                nh = (first[ni, k] >= 0) & (first[ni, k] <= end)
                acc = accum[name]
                for key, number in dict(pairs=len(pi), fail_hits=ph.sum(), success_hits=nh.sum(),
                                        failure_only=(ph & ~nh).sum(), success_only=(nh & ~ph).sum()).items():
                    acc[key] += int(number)
                if k < len(NAMES):
                    ps, ns = peak[pi, end, k], peak[ni, end, k]
                    good = np.isfinite(ps) & np.isfinite(ns)
                    difference = ps[good] - ns[good]
                    acc['comparable_scores'] += int(good.sum())
                    acc['concordant'] += float((difference > 1e-12).sum() + .5 * (np.abs(difference) <= 1e-12).sum())
        support.append(dict(cohort=cohort, task=task, pairs_before_minimum_query=raw_pairs,
                            pairs_shorter_than_q9=excluded))
        for name, acc in accum.items():
            if not acc['pairs']:
                continue
            rows.append(dict(cohort=cohort, task=task, method=name, **acc,
                             hit_difference=(acc['fail_hits'] - acc['success_hits']) / acc['pairs'],
                             prefix_concordance=acc['concordant'] / acc['comparable_scores']
                             if acc['comparable_scores'] else np.nan))
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / 'completion-matched-tasks.csv', index=False)
    pd.DataFrame(support).to_csv(ROOT / 'completion-matched-support.csv', index=False)
    summaries = []
    for (cohort, method), group in table.groupby(['cohort', 'method']):
        mean, low, high = interval(group.hit_difference)
        concordance, con_low, con_high = interval(group.prefix_concordance)
        summaries.append(dict(cohort=cohort, method=method, tasks=len(group), pairs=int(group.pairs.sum()),
                              fail_hits=int(group.fail_hits.sum()), success_hits=int(group.success_hits.sum()),
                              failure_only=int(group.failure_only.sum()), success_only=int(group.success_only.sum()),
                              task_mean_hit_difference=mean, ci_low=low, ci_high=high,
                              prefix_concordance=concordance, concordance_low=con_low, concordance_high=con_high))
    pd.DataFrame(summaries).to_csv(ROOT / 'completion-matched-summary.csv', index=False)
    return summaries


def s05_extension(frame, coordinates):
    table = pd.read_csv(ENCODED / 's05_features.csv')
    task_map = pd.read_csv(ENCODED / 'plot_task_map.csv')
    task = task_map.loc[task_map.plot_id == 'S05', 'task'].item()
    candidates = []
    for cohort, group in frame[frame.task == task].groupby('cohort'):
        errors, label_errors = [], 0
        for row in group.itertuples():
            saved = table[table.episode == row.episode].sort_values('query')
            if len(saved) != row.length:
                errors.append(np.inf)
                continue
            x = saved[['mobility_log_ratio', 'acceleration_log_ratio']].to_numpy()
            errors.append(np.nanmax(np.abs(x - coordinates[row.Index, :row.length])))
            label_errors += int(bool(saved.failure.iloc[0]) != row.failure)
        candidates.append(dict(cohort=cohort, max_error=max(errors), label_errors=label_errors))
    matches = [row for row in candidates if row['max_error'] <= 1e-4 and row['label_errors'] == 0]
    if len(matches) != 1:
        raise ValueError(f'S05 provenance not uniquely matched: {candidates}')
    cohort = matches[0]['cohort']
    rows = []
    for row in frame[(frame.task == task) & (frame.cohort == cohort)].itertuples():
        saved = table[table.episode == row.episode].sort_values('query')
        full = np.full((len(saved), 27), np.nan)
        for name in saved.columns:
            if name in INDEX:
                full[:, INDEX[name]] = saved[name].to_numpy()
        joint = motif_scores(full)
        for factor in (1.1, 1.2, 1.3):
            first = first_alarms(joint[None], factor)[0]
            for k in (0, 1, 5):
                rows.append(dict(identity=row.identity, cohort=cohort, episode=row.episode,
                                 failure=row.failure, factor=factor, method=MOTIF_NAMES[k],
                                 first_query=int(first[k])))
    pd.DataFrame(rows).to_csv(ROOT / 's05-extended-motifs.csv', index=False)
    save_json(ROOT / 's05-provenance.json', dict(task=task, matched=matches[0],
                                               candidates=candidates, unique_additional_episodes=0))


def supplementary(frame, scores):
    events = {name: frame['first_' + name].to_numpy() >= 0 for name in NAMES}
    rows = []
    for (cohort, reason), group in frame[frame.failure & frame.primary_failure_reason.ne('')].groupby(
            ['cohort', 'primary_failure_reason']):
        ids = group.index.to_numpy()
        rows.append(dict(cohort=cohort, physical_annotation=reason, failures=len(ids),
                         **{name: int(events[name][ids].sum()) for name in NAMES}))
    pd.DataFrame(rows).to_csv(ROOT / 'historical-physical-annotations.csv', index=False)
    overlap = []
    for (cohort, failure), group in frame.groupby(['cohort', 'failure']):
        ids = group.index.to_numpy()
        for left in NAMES:
            for right in NAMES:
                overlap.append(dict(cohort=cohort, failure=failure, left=left, right=right,
                                    episodes=len(ids), intersection=int((events[left][ids] & events[right][ids]).sum())))
    pd.DataFrame(overlap).to_csv(ROOT / 'motif-overlap.csv', index=False)
    counter = frame[~frame.failure & frame.first_frozen_active.ge(0)].copy()
    counter['alarm_fraction'] = 10 * counter.first_frozen_active / counter.horizon_actions
    counter['actions_after_alarm'] = np.where(counter.action_steps >= 0,
                                             counter.action_steps - 10 * counter.first_frozen_active, np.nan)
    counter.sort_values(['cohort', 'alarm_fraction', 'identity']).to_csv(ROOT / 'successful-counterexamples.csv', index=False)
    curves = []
    for (cohort, failure), group in frame.groupby(['cohort', 'failure']):
        ids = group.index.to_numpy()
        for q in range(9, 52):
            observed = ids[frame.loc[ids, 'length'].to_numpy() > q]
            for k, name in enumerate(NAMES):
                alarm = frame['first_' + name].to_numpy()[ids]
                curves.append(dict(cohort=cohort, failure=failure, method=name, query=q,
                                   episodes=len(ids), still_observed=len(observed),
                                   cumulative_hits=int(((alarm >= 0) & (alarm <= q)).sum()),
                                   currently_active=int((scores[observed, q, k] >= np.log(1.2)).sum())))
    pd.DataFrame(curves).to_csv(ROOT / 'query-risksets.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-prepared', action='store_true')
    args = parser.parse_args()
    start = time.perf_counter()
    if args.reuse_prepared:
        frame = pd.read_csv(ROOT / 'episode-index.csv').fillna({'primary_failure_reason': ''})
        coordinates = np.load(ROOT / 'two-axis-coordinates.npz')['coordinates']
        audit = json.loads((ROOT / 'preparation.json').read_text())
    else:
        historical, coordinates, cohorts = load_historical()
        current, recent_coordinates, checks = load_current()
        frame = pd.concat([historical, current], ignore_index=True).fillna({'primary_failure_reason': ''})
        coordinates = np.concatenate([coordinates, recent_coordinates])
        audit = dict(historical_cohorts=cohorts, current_raw_episodes=len(checks),
                     max_reconstruction_error=max(c['max_abs_error'] for c in checks),
                     alarm_disagreements=sum(c['alarm_disagreements'] for c in checks),
                     unique_episode_records=len(frame), no_new_inference=True,
                     unique_task_initial_pairs=historical.groupby(['task', 'init_state_id']).ngroups)
        save_json(ROOT / 'preparation.json', audit)
        frame.to_csv(ROOT / 'episode-index.csv', index=False)
        np.savez_compressed(ROOT / 'two-axis-coordinates.npz', coordinates=coordinates,
                            feature_names=np.asarray(['mobility_log_ratio', 'acceleration_log_ratio']))
    scores = score_two(coordinates)
    np.savez_compressed(ROOT / 'supported-motif-scores.npz', scores=scores, names=np.asarray(NAMES))
    event_rows = event_metrics(frame, scores)
    frame.to_csv(ROOT / 'episode-events.csv', index=False)
    print('episode events complete', flush=True)
    associations = conditional_associations(frame, coordinates, scores)
    print('same-query associations complete', flush=True)
    matched = completion_matched(frame, scores)
    print('completion-matched comparisons complete', flush=True)
    s05_extension(frame, coordinates)
    supplementary(frame, scores)
    save_json(ROOT / 'results.json', dict(preparation=audit, event_metrics=event_rows,
                                          associations=associations, completion_matched=matched,
                                          wall_seconds=time.perf_counter() - start,
                                          primary_factor=1.2, fixed_sensitivity_factors=[1.1, 1.3],
                                          no_model_inference=True, no_threshold_fitting=True))
    print(json.dumps(clean(audit)), flush=True)


if __name__ == '__main__':
    main()
