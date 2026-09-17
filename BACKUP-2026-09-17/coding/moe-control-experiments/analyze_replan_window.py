"""Secondary new-noise and mechanism checks; never changes the frozen gate."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def summarize(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair['group']].append(pair['gain'])
    totals = np.asarray([[sum(g), len(g)] for g in groups.values()], float)
    indices = np.random.default_rng(2026091519).integers(len(totals), size=(10000, len(totals)))
    sampled = totals[indices].sum(1)
    return dict(pairs=len(pairs), task_init_groups=len(groups),
                native_success=sum(p['native_success'] for p in pairs),
                window_success=sum(p['window_success'] for p in pairs),
                rescues=sum(p['rescue'] for p in pairs), harms=sum(p['harm'] for p in pairs),
                paired_gain=float(np.mean([p['gain'] for p in pairs])),
                clustered_bootstrap_95=np.quantile(sampled[:, 0] / sampled[:, 1], [.025, .975]).tolist())


def main(root):
    audit, config = read(root / 'audit.json'), read(root / 'config.json')
    collection = read(root / 'collection.json')
    assert audit['passed']
    new = summarize([p for p in audit['pairs'] if p['replicate'] != 0])
    original = summarize([p for p in audit['pairs'] if p['replicate'] == 0])
    mechanism = []
    for parent in config['parents']:
        for replicate in range(4):
            common = root / parent['name'] / ('r%d' % replicate)
            with np.load(common / 'native10/rollout.npz') as a, np.load(common / 'window5/rollout.npz') as b:
                start = parent['start']
                assert np.array_equal(a['actions'][:start + 5], b['actions'][:start + 5])
                assert np.array_equal(a['integration_after'][:start + 5], b['integration_after'][:start + 5])
                length = min(len(a['actions']), len(b['actions']))
                changed = np.flatnonzero(np.any(a['actions'][:length] != b['actions'][:length], axis=1))
                row = dict(parent=parent['name'], replicate=replicate, first_action_difference=None if not len(changed) else int(changed[0]),
                           first_five_actions_and_integration_exact=True, offsets={})
                for offset in (20, 40):
                    t = start + offset
                    if t > length:
                        row['offsets'][offset] = dict(covered=False)
                        continue
                    section = dict(covered=True)
                    for arm, data in (('native10', a), ('window5', b)):
                        goals = data['physical_goals']
                        eef = data['physical_eef'][start:t + 1]
                        section[arm] = dict(goals_at_fork=int(goals[start].sum()), goals_at_end=int(goals[t].sum()),
                                            eef_path_m=float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum()),
                                            eef_net_m=float(np.linalg.norm(eef[-1] - eef[0])))
                    row['offsets'][offset] = section
                mechanism.append(row)
    full_counts = {arm: sum(r['deployment_full_calls'] for r in collection['results'] if r['arm'] == arm)
                   for arm in ('native10', 'window5')}
    cost_ratio = full_counts['window5'] / full_counts['native10'] - 1
    base_tasks = sorted({p['base_task_id'] for p in config['parents']})
    result = dict(passed=True, secondary_analysis=True, original_noise=original, new_noise_only=new,
                  base_task_ids=base_tasks,
                  equivalent_full_episode_calls=full_counts, relative_full_episode_calls=cost_ratio,
                  frozen_timing_gate_unchanged=audit['timing_gate_passed'], mechanisms=mechanism,
                  caveat='Known source outcome affects selection of original-noise replicate; report fresh noise separately. '
                         'Physical movement and alarm normalization are not recovery endpoints. '
                         'These secondary checks do not replace the preregistered continuation decision.')
    with (root / 'secondary-analysis.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    lines = ['# P3h 新噪声敏感性与动作接口检查', '',
             '这是次要分析，不更改已冻结的第一层统计或第二层继续门槛。原噪声的源终局已被查看，不能把它与新噪声都称为未见验证。', '',
             '| 子集 | 配对数 | 原策略成功 | 短窗口成功 | 救回 | 误伤 | 净成功差 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, summary in (('原噪声复现', original), ('仅三组新噪声', new)):
        lines.append('| %s | %d | %d | %d | %d | %d | %.2f 个百分点 |' %
                     (label, summary['pairs'], summary['native_success'], summary['window_success'],
                      summary['rescues'], summary['harms'], 100 * summary['paired_gain']))
    lines += ['', '新噪声组级 bootstrap 95%% 区间：[%.2f, %.2f] 个百分点；样本组数少，不作为强显著性或等价性结论。' %
              tuple(100 * x for x in new['clustered_bootstrap_95']), '',
              '8 个状态仅覆盖 %d 个基础任务（%s），按 task/init 合并为 7 组；4 组噪声不是 4 个独立初态。' %
              (len(base_tasks), ', '.join(str(t) for t in base_tasks)), '',
              '全部 32 个配对在分叉后的前 5 个动作及完整 MuJoCo 积分状态仍完全一致。第一处动作分歧仅可能出现在第一次提前重规划之后。', '',
              '补齐两组相同的已记录公共前缀后，等效完整 episode 策略调用为原生 %d、短窗口 %d，差异 %+.2f%%。这不是本轮实际执行的前向总数，实际模型调用另在主报告中统计。' %
              (full_counts['native10'], full_counts['window5'], cost_ratio * 100), '',
              '20/40 个物理步内的末端路径和目标谓词保存在 secondary-analysis.json；它们只用于描述，不用于触发、选样或把局部运动称为恢复。', '',
              '噪声按共同物理时刻配对，消除了按 query 计数造成的随机流错位；但这两组实验仍不能单独分解“新观测”与“额外策略调用/采样”的因果贡献。若出现可重复净收益，才值得增加相同调用预算的旧观测消融。']
    with (root / 'SECONDARY_ANALYSIS.zh.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(new_noise_only=new, original_noise=original, timing_gate=audit['timing_gate_passed'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
