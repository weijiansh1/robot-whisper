"""Freeze actual-alarm probes and independent fresh rollout identities."""

import json

import numpy as np
import pandas as pd

from v82_knn_research import CURRENT, archive, digest, save_json
from alarm_trajectory_study import ROOT, PREVIOUS, phase_signature, physical_window, load_bank


def alarm(row):
    return int(row.v82_frozen if row.v82_frozen >= 0 else row.knn_available_first)


def main():
    if (ROOT / 'functional-selection.json').exists():
        raise SystemExit('Selection is already frozen')
    load_bank()
    frame = pd.read_csv(PREVIOUS / 'current-decisions.csv')
    episodes = {e['name']: e for e in json.loads((CURRENT / 'summary.json').read_text())['episodes']}
    physics = {name: archive(ROOT / 'physical' / (name+'.npz')) for name in frame.name}
    parents, pairs, probes = [], [], []
    for task, part in frame[frame.benchmark.eq('plus')].groupby('task_id', sort=True):
        good, bad = part[~part.failure], part[part.failure]
        if good.empty or bad.empty:
            continue
        good_alarms = good[(good.v82_frozen >= 0) | (good.knn_available_first >= 0)]
        if not good_alarms.empty:
            good = good_alarms
        bad_v82 = bad[bad.v82_frozen >= 0]
        bad = bad_v82 if not bad_v82.empty else bad[bad.knn_available_first >= 0]
        choices = []
        for failure in bad.itertuples():
            fq = alarm(failure)
            fs = phase_signature(physics[failure.name], fq)
            for success in good.itertuples():
                sqs = [alarm(success)] if alarm(success) >= 0 else list(range(7, success.length))
                for sq in sqs:
                    ss = phase_signature(physics[success.name], sq)
                    mismatch = sum(a != b for a, b in zip(fs, ss)) if len(fs) == len(ss) else 999
                    choices.append((mismatch, abs(fq-sq), failure.name, success.name, fq, sq))
        mismatch, gap, failure_name, success_name, fq, sq = min(choices)
        pair_id = len(pairs)
        pairs.append(dict(pair_id=pair_id, task=int(task), failure=failure_name, success=success_name,
                          failure_query=fq, success_query=sq, phase_matched=mismatch == 0,
                          phase_mismatches=mismatch, query_gap=gap))
        for name, q in ((failure_name, fq), (success_name, sq)):
            row = frame[frame.name.eq(name)].iloc[0]
            episode = episodes[name]
            parent = dict(parent_id=len(parents), pair_id=pair_id, task=int(task), name=name, query=int(q),
                          success=not bool(row.failure), phase_matched=mismatch == 0,
                          source_dir=episode['source_artifact_dir'], prompt=episode['prompt'],
                          trace_sha256=episode['source_trace_sha256'], source_hb_sha256=episode['integrity']['full_hb_sha256'],
                          v82_first=int(row.v82_frozen), knn_first=int(row.knn_available_first),
                          primary_trigger='v82' if row.v82_frozen >= 0 else 'knn' if row.knn_available_first >= 0 else 'matched_control',
                          anchor_external=physical_window(physics[name], q),
                          physics_sha256=digest(ROOT / 'physical' / (name+'.npz')))
            parents.append(parent)
            for stage, position in (('before', q-3), ('anchor', q)):
                assert position >= 1
                probes.append(dict(**parent, probe_id=len(probes), stage=stage, probe_query=int(position)))
    assert len(parents) == 16 and len(probes) == 32
    save_json(ROOT / 'functional-selection.json', dict(parents=parents, pairs=pairs, probes=probes,
              selection_uses_new_functional_data=False, protocol_sha256=digest(ROOT / 'PROTOCOL.md')))
    pd.DataFrame(pairs).to_csv(ROOT / 'functional-pairs.csv', index=False)
    from run_benchmark_sample import make_plan
    fresh = {variant: make_plan(variant, 2026091502, 1) for variant in ('plus', 'pro')}
    previous_dirs = [__import__('pathlib').Path('/data/libero-runtime') / (variant+'-sample-plan.json')
                     for variant in ('plus', 'pro')]
    old = {(job['task_name'], job['init_state_id'], job['flow_noise_seed'])
           for path in previous_dirs for job in json.loads(path.read_text())['jobs']}
    for variant, plan in fresh.items():
        assert len(plan['jobs']) == 10
        for job in plan['jobs']:
            assert (job['task_name'], job['init_state_id'], job['flow_noise_seed']) not in old
    save_json(ROOT / 'fresh-plans.json', dict(plans=fresh, primary_parent_count=20,
              profile_sha256=digest(ROOT / 'profile/trajectory-bank.npz'),
              parameters_sha256=digest(ROOT / 'profile/parameters.json'), all_identities_new=True))
    print(pd.DataFrame(pairs).to_string(index=False), flush=True)
    print('Fresh initial IDs:', fresh['plus']['init_state_ids'], flush=True)


if __name__ == '__main__':
    main()
