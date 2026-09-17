"""Reconstruct execution, common random numbers, state pairing and endpoints."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from gate_runtime import BASE
from run_replan_window import check_config
from run_online_experiment import EVALUATION, equal, jsonable, now, save_json, sha_array
from himoe_libero_bridge.episode_trace import load_episode_trace, sha256_file
from collection_routes import (PROBS_KEY, NATIVE_IDS_KEY, EFFECTIVE_IDS_KEY,
                               NATIVE_WEIGHTS_KEY, EFFECTIVE_WEIGHTS_KEY)
from v82_closed_loop import V82Monitor


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def paired_noise(parent, replicate, step):
    if replicate == 0 and step % 10 == 0:
        generator = np.random.default_rng(parent['flow_seed'])
        bank = [generator.standard_normal((10, 24)).astype(np.float32) for _ in range(step // 10 + 1)]
        return bank[-1]
    seed = np.random.SeedSequence([2026091508, int(parent['benchmark'] == 'pro'),
                                  parent['base_task_id'], parent['init_state_id'], replicate, step])
    return np.random.default_rng(seed).standard_normal((10, 24)).astype(np.float32)


def mobility(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a / a.sum(-1, keepdims=True), b / b.sum(-1, keepdims=True)
    return float(np.sqrt(.5 * np.square(np.sqrt(a[4:, -1, 1:]) - np.sqrt(b[4:, -1, 1:])).sum(-1)).mean())


def audit(root):
    config = check_config(root, check_old=True)
    collection = read(root / 'collection.json')
    assert collection['passed'] and collection['rollouts'] == 64
    assert collection['parameter_sha256_before'] == collection['parameter_sha256_after']
    events = records(root / 'calls.jsonl')
    requests = [r for r in events if r['event'] == 'request']
    responses = [r for r in events if r['event'] == 'response']
    assert len(events) == 2 * len(responses)
    assert len(requests) == len(responses) == collection['model_completed'] == collection['model_attempts']
    assert len(responses) <= 2080
    calls = {}
    for ordinal, (request, response) in enumerate(zip(requests, responses)):
        for key in ('ordinal', 'parent', 'replicate', 'arm', 'step', 'validation', 'noise_sha256'):
            assert request[key] == response[key]
        assert ordinal == response['ordinal'] and response['path'] not in calls
        assert sha256_file(root / response['path']) == response['sha256']
        calls[response['path']] = response
    used, pairs, branch_rows, aligned = set(), [], [], []
    for parent in config['parents']:
        _, source = load_episode_trace(Path(parent['source']))
        source_actions = np.concatenate([a[:int(n)] for a, n in zip(source['predicted_actions'], source['executed_lengths'])])
        with np.load(EVALUATION / parent['name'] / 'full-hb-routes.npz') as saved:
            original = saved['hb_router_probs']
        snapshot_sha = None
        for replicate in range(4):
            endpoints, first_calls, common_noises = {}, {}, {}
            for arm in ('native10', 'window5'):
                directory = root / parent['name'] / ('r%d' % replicate) / arm
                result, environment = read(directory / 'result.json'), read(directory / 'environment.json')
                rows = records(directory / 'queries.jsonl')
                assert result in collection['results']
                state_sha = sha256_file(directory / 'fork-state.msgpack')
                assert state_sha == result['prefix_state_audit']['sha256']
                if snapshot_sha is None:
                    snapshot_sha = state_sha
                assert snapshot_sha == state_sha
                assert result['prefix_state_audit']['integration_spec'] == 8191
                assert len(result['prefix_state_audit']['controller_fields'][0]) > 40
                monitor = V82Monitor()
                for p in original[:parent['query']]:
                    monitor.update(p)
                step, previous = parent['start'], original[parent['query'] - 1]
                elapsed_infer = elapsed_env = 0.
                shortened_steps = 0
                with np.load(directory / 'rollout.npz') as rollout:
                    for k, source_k in (('sim_states_after', 'sim_states_after'), ('rewards', 'rewards'),
                                        ('dones', 'dones'), ('successes', 'successes')):
                        equal(rollout[k][:step], source[source_k][:step], 'Exact source prefix ' + k)
                    equal(rollout['actions'][:step], source_actions[:step], 'Exact source actions prefix')
                    assert len(rollout['integration_after']) == len(rollout['actions'])
                    for index, row in enumerate(rows):
                        assert row['step'] == step and row['replicate'] == replicate and row['arm'] == arm
                        expected_length = 5 if arm == 'window5' and step < parent['start'] + 20 else 10
                        assert row['planned_execution'] == expected_length
                        assert row['suffix_discarded'] == 10 - expected_length
                        call = calls[row['path']]
                        assert row['path'] not in used and not call['validation']
                        used.add(row['path'])
                        for k in ('parent', 'replicate', 'arm', 'step', 'state_audit'):
                            assert call[k] == row[k]
                        with np.load(root / row['path']) as trace:
                            equal(trace['request_noise'], paired_noise(parent, replicate, step), 'Physical-time noise')
                            assert sha_array(trace['request_noise']) == call['noise_sha256']
                            assert sha_array(trace['actions']) == call['actions_sha256']
                            assert sha_array(trace[PROBS_KEY]) == call['hb_sha256']
                            assert trace['actions'].shape == (10, 7) and trace[PROBS_KEY].shape == (8, 10, 11, 32)
                            equal(trace[NATIVE_IDS_KEY], trace[EFFECTIVE_IDS_KEY], 'Native expert dispatch')
                            equal(trace[NATIVE_WEIGHTS_KEY], trace[EFFECTIVE_WEIGHTS_KEY], 'Native gate weights')
                            equal(trace['sim_before'], rollout['sim_states_after'][step - 1], 'Real new observation state')
                            assert 1 <= row['executed'] <= expected_length
                            equal(rollout['actions'][step:step + row['executed']], trace['actions'][:row['executed']],
                                  'Only newly generated prefix executed, without rescaling')
                            if index == 0:
                                first_calls[arm] = (call['noise_sha256'], call['actions_sha256'], call['hb_sha256'])
                                assert row['state_audit'] == result['prefix_state_audit']
                            else:
                                assert rows[index - 1]['next_state_audit'] == row['state_audit']
                            if step % 10 == 0:
                                status = jsonable(monitor.update(trace[PROBS_KEY]))
                                assert row['monitor'] == status and monitor.v7.query == step // 10
                                aligned.append(dict(parent=parent['name'], replicate=replicate, arm=arm, step=step,
                                                    offset=step - parent['start'], mobility_10steps=mobility(previous, trace[PROBS_KEY])))
                                previous = trace[PROBS_KEY].copy()
                                key = (replicate, step)
                                if key in common_noises:
                                    assert common_noises[key] == call['noise_sha256']
                                common_noises[key] = call['noise_sha256']
                            else:
                                assert row['monitor'] is None
                            if arm == 'native10' and replicate == 0:
                                equal(trace['actions'], source['predicted_actions'][step // 10], 'Native source continuation')
                                equal(trace[PROBS_KEY], original[step // 10], 'Native source routes')
                                if index == 0:
                                    duplicate_path = str((directory / 'queries' / ('s%03d-repeat.npz' % step)).relative_to(root))
                                    duplicate_call = calls[duplicate_path]
                                    assert duplicate_call['validation'] and duplicate_path not in used
                                    used.add(duplicate_path)
                                    with np.load(root / duplicate_path) as duplicate:
                                        assert set(duplicate.files) == set(trace.files)
                                        for key in trace.files:
                                            equal(duplicate[key], trace[key], 'Duplicate native inference')
                        if expected_length == 5:
                            shortened_steps += row['executed']
                        step += row['executed']
                        elapsed_infer += call['resource']['seconds']
                        elapsed_env += row['environment_seconds']
                        assert row['success'] == bool(rollout['successes'][step - 1])
                        if row['executed'] < expected_length:
                            assert index == len(rows) - 1 and (row['success'] or step == 520)
                        for key, physical_key in (('goals', 'physical_goals'), ('grasp', 'physical_grasp'), ('eef', 'physical_eef')):
                            assert np.array_equal(row['physics'][key], rollout[physical_key][row['step']])
                            assert np.array_equal(row['next_physics'][key], rollout[physical_key][step])
                    assert step == result['action_steps'] == environment['steps'] == len(rollout['actions'])
                    assert result['success'] == environment['success'] == bool(rollout['successes'][-1])
                    assert bool(np.all(rollout['physical_goals'][-1])) == result['success']
                    assert result['success'] or step == 520
                    assert not np.any(rollout['successes'][:-1])
                    assert result['window_steps'] == shortened_steps == (min(20, step - parent['start']) if arm == 'window5' else 0)
                    assert result['suffix_calls'] == len(rows)
                    assert result['deployment_full_calls'] == parent['query'] + len(rows)
                    assert result['inference_seconds'] == elapsed_infer and result['environment_seconds'] == elapsed_env
                    assert result['first_v82_physical_step'] == monitor.first_v82_alarm * 10
                    assert environment['frames'] == step + 11
                    if arm == 'native10' and replicate == 0:
                        for key in ('sim_states_after', 'rewards', 'dones', 'successes'):
                            equal(rollout[key], source[key], 'Entire source continuation ' + key)
                        equal(rollout['actions'], source_actions, 'Entire source action stream')
                        assert environment['frame_sha256'] == read(Path(parent['source']) / 'episode-trace.json')['source_render']['frame_sha256']
                assert rows[-1]['next_state_audit'] == result['terminal_state_audit'] == environment['final_state_audit']
                endpoints[arm] = result
                branch_rows.append(result)
            assert first_calls['native10'] == first_calls['window5']
            native, short = endpoints['native10'], endpoints['window5']
            pairs.append(dict(parent=parent['name'], group='%02d/%03d' % tuple(native['group']),
                              stratum=parent['stratum'], replicate=replicate, source_success=parent['source_success'],
                              native_success=native['success'], window_success=short['success'],
                              gain=int(short['success']) - int(native['success']),
                              rescue=not native['success'] and short['success'], harm=native['success'] and not short['success'],
                              native_steps=native['action_steps'], window_steps=short['action_steps'],
                              native_calls=native['suffix_calls'], window_calls=short['suffix_calls'],
                              extra_suffix_calls=short['suffix_calls'] - native['suffix_calls'],
                              native_inference_seconds=native['inference_seconds'], window_inference_seconds=short['inference_seconds']))
        print(json.dumps(dict(audited=parent['name'], pairs=4, checked_calls=len(used))), flush=True)
    assert used == set(calls) and len(branch_rows) == 64 and len(pairs) == 32
    assert sum(r['action_steps'] for r in branch_rows) == collection['environment_action_steps']
    by_group = defaultdict(list)
    for p in pairs:
        by_group[p['group']].append(p)
    totals = np.asarray([[sum(p['gain'] for p in v), len(v)] for v in by_group.values()], float)
    rng = np.random.default_rng(2026091518)
    chosen = rng.integers(len(totals), size=(10000, len(totals)))
    sums = totals[chosen].sum(1)
    bootstrap = sums[:, 0] / sums[:, 1]
    rescue, harm = sum(p['rescue'] for p in pairs), sum(p['harm'] for p in pairs)
    repeated_groups = [k for k, v in by_group.items() if sum(p['rescue'] for p in v) >= 2]
    summary = dict(passed=True, utc=now(), paired_units=32, snapshots=8, task_init_groups=len(by_group),
                   native_success=sum(p['native_success'] for p in pairs),
                   window_success=sum(p['window_success'] for p in pairs), rescues=rescue, harms=harm,
                   paired_success_gain=(rescue - harm) / 32,
                   group_bootstrap_95=np.quantile(bootstrap, [.025, .975]).tolist(),
                   uncertainty='exploratory clustered bootstrap; few groups; degenerate intervals do not establish equivalence',
                   repeated_rescue_groups=repeated_groups, timing_gate_passed=rescue > harm and len(repeated_groups) >= 2,
                   model_calls=len(calls), duplicate_validation_calls=8,
                   native_suffix_calls=sum(p['native_calls'] for p in pairs),
                   window_suffix_calls=sum(p['window_calls'] for p in pairs),
                   inference_seconds_by_arm={arm: sum(r['inference_seconds'] for r in branch_rows if r['arm'] == arm)
                                             for arm in ('native10', 'window5')},
                   previous_sealed_artifacts_unchanged=len(config['previous_sealed_files']),
                   first_layer_only=True, pairs=pairs)
    save_json(root / 'audit.json', summary)
    for name, rows in (('pairs.csv', pairs), ('physical-time-mobility.csv', aligned)):
        with (root / name).open('x') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# P3h 第一层配对结果', '',
             '仅改变实际执行长度：F=10、H=10 不变，h=5 持续 20 个物理动作步后恢复 h=10。', '',
             '64 条实际仿真续跑，8 个状态、每状态 4 组未来噪声，共 32 个配对；按 task/init 合并为 %d 组。' % len(by_group), '',
             '| 状态 | 分层 | 原策略成功 /4 | 短窗口成功 /4 | 救回 | 误伤 |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for parent in config['parents']:
        ps = [p for p in pairs if p['parent'] == parent['name']]
        lines.append('| %s q%d | %s | %d | %d | %d | %d |' %
                     (parent['name'], parent['query'], parent['stratum'], sum(p['native_success'] for p in ps),
                      sum(p['window_success'] for p in ps), sum(p['rescue'] for p in ps), sum(p['harm'] for p in ps)))
    lines += ['', '合计：原策略 %d/32，短窗口 %d/32；救回 %d，误伤 %d；净成功差 %.2f 个百分点。' %
              (summary['native_success'], summary['window_success'], rescue, harm, summary['paired_success_gain'] * 100), '',
              '组级 bootstrap 95%% 区间：[%.2f, %.2f] 个百分点。仅为开发数据的不确定性描述；组数少，零宽区间不证明零效应。' %
              tuple(v * 100 for v in summary['group_bootstrap_95']), '',
              '未来噪声的第 0 组沿用原记录，另外 3 组为独立固定流；救回/误伤均相对同噪声的新跑原策略计算，不相对旧标签。', '',
              '正式后缀推理：原策略 %d 次，短窗口 %d 次；另有 8 次重复验证。模型总调用 %d 次。' %
              (summary['native_suffix_calls'], summary['window_suffix_calls'], summary['model_calls']), '',
              '完整参数内容哈希前后一致；64 个分叉点的同父状态、控制器与 RNG 核对通过；8 条原噪声 native 全程复现源动作/状态/图像。', '',
              '报警诊断按物理 10 步网格复算；窗口内额外查询不推进 v8.2 时钟。没有用分数回落作为恢复终点。', '',
              '第二层继续门槛：%s。' % ('通过，仍须执行预留面板四组对照才能判断 MoE 时机价值' if summary['timing_gate_passed'] else
                                          '未通过，按预注册方案停止扩展，不调报警阈值；尚未证明 v8.2 的时机选择有收益'), '',
              '限制：这是有意选样的开发实验，报警前不等于有物理 loop 起点真值；只有少量原成功状态。仿真在推理时暂停，未验证真实机器人异步延迟。', '',
              '实验计划及文献边界见 ../../P3H_REPLAN_WINDOW_PLAN.zh.md。']
    with (root / 'REPORT.zh.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'pairs'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run.resolve())
