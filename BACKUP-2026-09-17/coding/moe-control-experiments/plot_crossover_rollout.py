"""Render audited terminal outcomes and per-policy MoE trajectories."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('native', 'joint', 'route_only', 'output_only')
NAMES = ('Native', 'Joint', 'Route + O0', 'R0 + O1')
ZH = dict(native='原生', joint='正常干预', route_only='改路由回填原生', output_only='原门控回填候选')
COLORS = ('#696969', '#19836a', '#bc4b45', '#397ba5')
COMPONENTS = ('freeze', 'acceleration', 'periodicity', 'inversion', 'curvature')


def read(path):
    return json.loads(path.read_text())


def save(fig, path):
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(root):
    config, summary, audit = read(root / 'config.json'), read(root / 'summary.json'), read(root / 'audit.json')
    assert audit['passed'] and summary['actual_rollouts'] == 20
    methods = summary['methods']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, field, title, limit in zip(axes, ('rescues', 'harms', 'deployment_call_increase_fraction'),
        ('Rescued original failures (of 4)', 'Harmed original success (of 1)', 'Deployment model-call increase (%)'), (4, 1, None)):
        values = [methods[arm][field] * (100 if limit is None else 1) for arm in ARMS]
        ax.bar(NAMES, values, color=COLORS, width=.6)
        ax.set_title(title, fontsize=11)
        ax.tick_params(axis='x', labelsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, (max(values) * 1.3 or 1) if limit is None else limit + .5)
        for i, value in enumerate(values):
            ax.annotate('%.2f' % value if limit is None else str(value), (i, value), xytext=(0, 5),
                        textcoords='offset points', ha='center', fontsize=10)
    fig.suptitle('Online MoE crossover: 5 paired development parents, 20 actual rollouts', fontsize=13)
    save(fig, root / 'crossover-rollout.png')
    for parent in config['parents']:
        results = {r['arm']: r for r in summary['branches'] if r['parent'] == parent['name']}
        fig, axes = plt.subplots(1, 4, figsize=(13, 4), constrained_layout=True)
        for ax, arm, name in zip(axes, ARMS, NAMES):
            directory = root / parent['name'] / arm
            environment = read(directory / 'environment.json')
            with imageio.get_reader(directory / 'episode.mp4') as reader:
                frame = reader.get_data(environment['frames'] - 1)
            ax.imshow(frame)
            ax.axis('off')
            result = results[arm]
            ax.set_title('%s\n%s, %d steps' % (name, 'success' if result['success'] else 'failure', result['action_steps']), fontsize=11)
        fig.suptitle(parent['name'], fontsize=13)
        save(fig, root / parent['name'] / 'final-frames.png')
        fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True, constrained_layout=True)
        for arm, name, color in zip(ARMS, NAMES, COLORS):
            values = [r for r in summary['queries'] if r['parent'] == parent['name'] and r['arm'] == arm]
            for i, ax in enumerate(axes):
                ax.plot([r['query'] for r in values], [r['formal_normalized'][i] for r in values],
                        color=color, linewidth=1.4, label=name)
        for ax, component in zip(axes, COMPONENTS):
            ax.axhline(0, color='#777777', linestyle=':', linewidth=1)
            ax.axvspan(parent['query'] - .5, parent['query'] + 11.5, color='#eeeeee', zorder=-1)
            ax.set_ylabel(component, fontsize=10)
            ax.spines[['top', 'right']].set_visible(False)
        axes[0].legend(ncol=4, fontsize=9)
        axes[-1].set_xlabel('Policy query')
        fig.suptitle(parent['name'] + '\nFormal MoE scores: (score - threshold) / margin', fontsize=13)
        save(fig, root / parent['name'] / 'moe-trajectories.png')
    output, joint = methods['output_only'], methods['joint']
    improved, worsened = [], []
    for parent in config['parents']:
        b = {r['arm']: r for r in summary['branches'] if r['parent'] == parent['name']}
        if b['output_only']['success'] and not b['joint']['success']:
            improved.append(parent['name'])
        if b['joint']['success'] and not b['output_only']['success']:
            worsened.append(parent['name'])
    lines = ['# P3g: 交叉输出方法的真实续跑', '',
        '本轮真正执行到任务成功或原 520 步上限。原模型权重和持久架构不变；候选与历史反馈由 MoE 提供，预测动作保护保留，任务成败不用于选候选。', '',
        '## 主要结果', '',
        '| 方法 | 救回原失败 | 误伤原成功 | 实际介入/机会 | 原筛查拒绝但仍执行 | 完整轨迹调用增幅 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for arm in ARMS:
        m = methods[arm]
        lines.append('| %s | %d/4 | %d/1 | %d/%d | %d | %.2f%% |' % (ZH[arm], m['rescues'], m['harms'],
            m['accepted'], m['opportunities'], m['executed_despite_old_rejection'], m['deployment_call_increase_fraction'] * 100))
    if output['rescues'] == 0:
        lines += ['', '本轮原门控回填候选输出没有救回四个既有失败场景。不能把路由分数、输出回填成功或动作发生变化解释成已取得恢复收益。']
    else:
        lines += ['', '原门控回填候选输出在四个既有失败场景中救回 %d 个；这是开发集结果，不是泛化成功率。' % output['rescues']]
    lines += ['', '相对本轮同接受规则的新 joint，output_only 新增成功的父场景: %s；失去成功的父场景: %s。' %
        (', '.join(improved) or '无', ', '.join(worsened) or '无'), '',
        '## 实验解释', '',
        '- 三种干预使用相同 12 查询窗口、同一高档 mobility_balance 生成规则与候选供体动作保护，不再以旧路由降分筛查决定接受。',
        '- 因接受规则不同，不能只和 P3d 的旧筛查控制器比较。新 joint 是输出回填方法的主要配对对照。',
        '- 每次 output_only 的有效计算链和动作精确复现它自己的当前 O1 供体；每次 route_only 精确复现当前原生。回填本身没有凭空生成更优动作。',
        '- 后续候选只使用各组实际提交的路由历史，因此 joint 与 output_only 可以从同样的第一步动作出发，再因不同路由历史形成不同偏置和行为。',
        '- native 与 route_only 的十条完整轨迹均按动作、状态、渲染和终局精确复现源记录，是实际运行后的核对，不是用等价关系代替续跑。', '',
        '## 逐场景终局', '', '| 场景 | 原生 | 正常干预 | 路由回填原生 | 原门控回填候选 |', '| --- | --- | --- | --- | --- |']
    for parent in config['parents']:
        values = {r['arm']: r for r in summary['branches'] if r['parent'] == parent['name']}
        cells = [('成功' if values[a]['success'] else '失败') + ' / %d 步' % values[a]['action_steps'] for a in ARMS]
        lines.append('| %s | %s |' % (parent['name'], ' | '.join(cells)))
    lines += ['', '各场景目录均含 final-frames.png、moe-trajectories.png 和四段实际 episode.mp4。', '',
        '## 反馈分歧', '', '| 场景 | 首次路由差异 q | 首次偏置差异 q | 首次动作差异 q | 首次观测差异 q |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in summary['paired_feedback']:
        values = [row[k] for k in ('first_route_difference_query', 'first_bias_difference_query',
                                   'first_action_difference_query', 'first_observation_difference_query')]
        lines.append('| %s | %s |' % (row['parent'], ' | '.join('无' if v is None else str(v) for v in values)))
    lines += ['', '五个 joint/output_only 首次执行动作均相同，若存在下一查询，其起始观测也精确相同。分歧时间从真实保存的输入、偏置、路由及动作重算，不凭终局倒推。', '',
        '## 验证与成本', '',
        '- 实际轨迹 %d 条，模型前向 %d 次，其中额外验证 %d 次；原始前缀回放不调用模型。' %
            (summary['actual_rollouts'], summary['model_calls'], summary['validation_calls']),
        '- 新环境动作 %d 步，另有原环境规定的稳定动作 %d 步。' % (summary['environment_action_steps'], summary['settle_steps']),
        '- %d 个交叉计算匹配对通过；40 个零偏置、重复及撤销检查通过。' % audit['cross_pairs'],
        '- 采集含加载、参数哈希与写盘 %.2f 分钟；前向计时合计 %.2f 秒；PyTorch 峰值 reserved %.0f MiB。' %
            (summary['collection_elapsed_seconds'] / 60, summary['inference_seconds'], summary['peak_reserved_mib']),
        '- 独立审计从实际 logits、分派、原始/回填输出重算选择，并核对实际执行动作、监控历史、公共前缀和任务终局；20 段视频完整解码。',
        '- 样本仅 5 条既有开发父轨迹，不能把 20 条相关轨迹当作 20 个独立失败样本，不能从单个成功父场景作安全保证。', '',
        '## 产物', '', '- [总览图](crossover-rollout.png)', '- [逐分支结果](branches.csv)',
        '- [全部候选决策](decisions.csv)', '- [完整汇总](summary.json)', '- [独立审计](audit.json)',
        '- [执行说明](EXECUTION_NOTES.zh.md)', '- [最终封存](verification.json)', '']
    with (root / 'REPORT.zh.md').open('x') as stream:
        stream.write('\n'.join(lines))
    notes = ['# P3g 执行说明', '',
        '使用独立进程与请求级临时 hook；不改原模型权重、共享服务或旧冻结实现。', '',
        '最大预算为 876 次，实际按成功提前结束和退化候选情况结算。40 次验证不参与选择；跨组候选不共用历史。', '',
        '仅干预窗口和验证调用保存完整内部激活，其他调用保存完整路由、预测动作、观测和审计状态。推理计时不含全部压缩写盘和模型参数哈希成本，完整实验耗时不能作为部署延迟。', '',
        '原生、route_only 都真实执行，不用旧轨迹冒充新运行。保护条件统一由本查询的 joint 供体预测动作决定；它是幅度和符号约束，不是经过证明的安全或正确性保证。', '',
        '首次启动目录 p3g-crossover-rollout-20260915-lq1_j641 因其他实验占用显存，在模型加载前被保护检查拒绝。该次模型前向及环境动作均为零，失败记录保留，不把准备完成写成运行完成。', '',
        '已封存的 run 不可重新运行会写文件的分析、绘图或 finalizer；后续新试验使用新目录。', '']
    with (root / 'EXECUTION_NOTES.zh.md').open('x') as stream:
        stream.write('\n'.join(notes))
    print(json.dumps(dict(report=str(root / 'REPORT.zh.md'), rollouts=summary['actual_rollouts'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
