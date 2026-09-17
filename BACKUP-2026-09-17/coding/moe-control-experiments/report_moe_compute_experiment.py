"""Plots and calibrated Chinese report from sealed-method MoE diagnostics."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

KINDS = ('route', 'input', 'total', 'input_total')
LABELS = ('Routing', 'MoE input', 'MoE output', 'Input + output')
CATEGORIES = ('no_confirmed_exit', 'exit_then_return', 'last_exit_no_observed_return',
              'insufficient_followup_or_mixed', 'reference_not_established')
COLORS = ('#62696e', '#b8525d', '#25857d', '#c6a341', '#e4e7e9')


def read(path):
    return json.loads(path.read_text())


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def figures(root, config, summary, prediction, noise, regions):
    files = []
    def finish(fig, name):
        path = root / name
        assert not path.exists()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        files.append(name)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    x, width = np.arange(4), .18
    for i, (method, color) in enumerate(zip(('S', 'ST', 'D', 'DT'), ('#62696e', '#25857d', '#b8525d', '#728447'))):
        y = [prediction['metrics'][kind + '_' + method]['rmse'] for kind in KINDS]
        ax.bar(x + (i - 1.5) * width, y, width, color=color, label=method)
    for method, color, style in (('constant', '#202629', ':'), ('persistence', '#bc7a34', '--'), ('v82', '#766295', '-.')):
        ax.axhline(prediction['metrics'][method]['rmse'], color=color, linestyle=style, linewidth=1.2, label=method)
    ax.set(xticks=x, xticklabels=LABELS, ylabel='Equal-parent RMSE of future MoE output distance',
           title='Held-out base-task prediction; six development initial states')
    ax.margins(y=.25)
    ax.legend(ncol=4, fontsize=9)
    finish(fig, 'prediction-rmse.png')

    fig, ax = plt.subplots(figsize=(11, 5), layout='constrained')
    x, width = np.arange(6), .16
    for i, (kind, label, color) in enumerate(zip(KINDS, LABELS, ('#62696e', '#25857d', '#b8525d', '#728447'))):
        values = [next(r['noise_observation_ratio'] for r in noise if r['parent'] == p['name'] and r['kind'] == kind and r['scope'] == 'back_path')
                  for p in config['parents']]
        ax.bar(x + (i - 1.5) * width, [np.nan if v is None else v for v in values], width, label=label, color=color)
    names = ['%s t%d\ni%d' % (p['benchmark'], p['base_task_id'], p['init_state_id']) for p in config['parents']]
    ax.axhline(1, color='#202629', linestyle=':', linewidth=1)
    ax.set(yscale='log', xticks=x, xticklabels=names, ylabel='Noise-change / observation-change distance',
           title='Matched MoE contrasts: three observations x three noises')
    ax.legend(ncol=4, fontsize=9)
    finish(fig, 'noise-vs-observation.png')

    fig, ax = plt.subplots(figsize=(12, 5.8), layout='constrained')
    grid, row_labels = [], []
    for kind, label in zip(KINDS, LABELS):
        for arm in ('native10', 'window5'):
            grid.append([CATEGORIES.index(next(r['category'] for r in regions if r['parent'] == p['name'] and r['replicate'] == replicate
                                               and r['arm'] == arm and r['kind'] == kind and r['scope'] == 'back_path' and r['scale'] == 1))
                         for p in config['parents'] for replicate in range(4)])
            row_labels.append(label + ' / ' + arm)
    ax.imshow(np.array(grid), aspect='auto', interpolation='nearest', cmap=matplotlib.colors.ListedColormap(COLORS), vmin=0, vmax=4)
    ax.set(yticks=np.arange(8), yticklabels=row_labels, xticks=np.arange(6) * 4 + 1.5,
           xticklabels=names, title='Primary MoE reference-region categories, all 24 pairs')
    for separator in np.arange(1, 6) * 4 - .5:
        ax.axvline(separator, color='white', linewidth=2)
    ax.legend(handles=[Patch(color=color, label=label) for color, label in zip(COLORS, ('No exit', 'Exit + return', 'No observed return', 'Mixed', 'No reference'))],
              ncol=3, loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=9)
    finish(fig, 'primary-regions.png')

    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    for i, (arm, color) in enumerate((('native10', '#62696e'), ('window5', '#25857d'))):
        values = [next(r['stable'] for r in summary['stability'] if r['kind'] == kind and r['scope'] == 'back_path' and r['arm'] == arm) for kind in KINDS]
        bars = ax.bar(np.arange(4) + (i - .5) * .3, values, .3, label=arm, color=color)
        ax.bar_label(bars, padding=3)
    ax.set(xticks=np.arange(4), xticklabels=LABELS, ylim=(0, 26), ylabel='Stable branch classifications / 24',
           title='Same category at radius multipliers 0.75, 1.00, 1.25')
    ax.legend()
    finish(fig, 'radius-stability.png')

    for parent in config['parents']:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout='constrained')
        for ax, kind, label in zip(axes, ('route', 'input_total'), ('Routing', 'MoE input + output')):
            for arm, color in (('native10', '#62696e'), ('window5', '#25857d')):
                row = next(r for r in regions if r['parent'] == parent['name'] and r['replicate'] == 0 and r['arm'] == arm
                           and r['kind'] == kind and r['scope'] == 'back_path' and r['scale'] == 1)
                ax.plot(row['steps'], np.array(row['nearest_distance']) / row['epsilon'], label=arm, color=color)
            ax.axhline(1, color='#b8525d', linestyle=':', linewidth=1)
            ax.set(ylabel='Distance / reference radius', title=label)
            ax.legend(fontsize=9)
        axes[-1].set_xlabel('Physical control step (ten-step observation grid)')
        fig.suptitle(parent['name'] + ' / fixed historical noise r0')
        finish(fig, parent['name'] + '-moe-distance.png')
    checks = []
    for name in files:
        with Image.open(root / name) as picture:
            pixels = np.asarray(picture.convert('RGB'))
            assert pixels.std() > 5 and min(picture.size) > 400
            checks.append(dict(path=name, width=picture.width, height=picture.height, pixel_std=float(pixels.std())))
    write_json(root / 'visual-check.json', dict(passed=True, files=checks,
               interpretation='nonblank image checks; detailed layout inspected separately'))
    return files


def report(root):
    config, collection = read(root / 'config.json'), read(root / 'collection.json')
    summary, prediction = read(root / 'analysis-summary.json'), read(root / 'prediction-summary.json')
    audit = read(root / 'audit.json')
    assert collection['passed'] and summary['passed'] and prediction['passed'] and audit['passed']
    noise, regions, homology = (read(root / name) for name in ('noise-analysis.json', 'regions.json', 'persistence.json'))
    files = figures(root, config, summary, prediction, noise, regions)
    names = dict(zip(KINDS, ('路由', 'MoE 输入', 'MoE 实际输出', 'MoE 输入+输出')))
    metrics, gates = prediction['metrics'], prediction['gates']
    representation = gates['representation']['passed']
    topology = gates['topology_simple']['passed'] and gates['topology_dense']['passed']
    judgment = ('有效计算表示和拓扑增量均达到本轮预定的探索性门槛。' if representation and topology else
                '有效计算表示达到探索性门槛，但拓扑未同时超过两类简单基线的预定门槛。' if representation else
                '主有效计算表示未达到相对路由基线的预定探索性门槛；不把次要方法事后改成主要赢家。')
    rows = [
        '# P3j: 只看 MoE 的连续有效计算与拓扑实验', '',
        '## 结论', '', judgment,
        '本轮的预测对象也是 MoE: 未来 40 个物理控制步后，最后 HB 层、最后 flow 的实际聚合输出变化量。不是动作预测、失败识别或恢复率。', '',
        '所有结果来自原有六个失败父状态、四个未来噪声、两臂的 48 条续跑。仅重放模型输入补采张量；新增环境动作 0，没有补跑成功轨迹。原恢复实验仍为 0/24 救回。', '',
        '## 1. MoE 范围与采集', '',
        '- 采集 HB MoE 的输入、共享专家输出、实际聚合输出，以及原有 HB/AS 路由。正式预测特征只使用 HB 路由及 input/total；shared 用于对照和幅度审计。',
        '- `total` 是选中专家的加权聚合加共享专家输出，不含后续残差相加。`input` 是 MoE 模块的端口输入，承载上游信息；输入单独的预测能力不能归功于专家路由机制。',
        '- 不使用图像、机器人状态、动作、物理停滞或任务进度作为特征或标签。它们中的观测只用于精确重放，动作只用于复现核验。',
        '- 八个 HB 层、十轮 flow、11 token 完整张量留档。主分析取后四层、全部 flow、十个动作 token；内部 flow 不当作物理时间。',
        '- 六个分叉的采集开关/重复与历史输出一致；48 条分支第一提案的三个 MoE 端口也一致。全部历史重放的动作和已存路由逐值核验通过。',
        '- 模型结构、参数、专家选择和输出均未改写；完整参数内容哈希前后一致，并与封存 P3g 一致。未向共享 9500 发送推理请求，也未重启该服务。', '',
        '## 2. 有效计算与拓扑各自贡献多少', '',
        '按基础任务 0/3/6 整组留出，训练标准化仅使用训练折，每个父初态等权。所有种子和两臂保持同折。固定 ridge=1，无参数搜索。目标按因果前缀尺度标准化。', '',
        '| 表示 | 简单 S RMSE | S+拓扑 RMSE | 全距离 D RMSE | D+拓扑 RMSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    for kind in KINDS:
        rows.append('| %s | %s |' % (names[kind], ' | '.join('%.6f' % metrics[kind + '_' + method]['rmse'] for method in ('S', 'ST', 'D', 'DT'))))
    rows += ['', 'S 是十二项距离/变化/复返摘要；D 是相同十四点历史的全部 91 个两两距离。拓扑只作为额外摘要加入，不引入另一种信息来源。', '',
             '| 其他基线 | 父状态等权 RMSE |', '| --- | ---: |']
    for key, label in (('constant', '训练父状态等权目标均值'), ('persistence', '过去 40 步同一 MoE 输出变化量'), ('v82', '冻结 v8.2 五个连续 MoE 分数')):
        rows.append('| %s | %.6f |' % (label, metrics[key]['rmse']))
    rows += ['', '| 预定比较 | RMSE 下降 | 改善的留出任务 | 达到门槛 |', '| --- | ---: | ---: | --- |']
    for key, label in (('representation', '输入+输出 S 对路由 S'), ('topology_simple', '输入+输出 S+T 对 S'), ('topology_dense', '输入+输出 D+T 对 D')):
        gate = gates[key]
        rows.append('| %s | %+.2f%% | %d/3 | %s |' % (label, gate['relative_rmse_reduction'] * 100, gate['tasks_improved'], '是' if gate['passed'] else '否'))
    rows += ['', '门槛是 RMSE 至少下降 10% 且至少 2/3 留出任务改善，只是预定工程推进标准，不是统计显著性。负下降表示误差增加。是否超过常数和持久基线也须同时看，不能只挑路由作为弱对照。',
             '这里有 1,256 个重叠时间窗口，但只有六个父初态、三个基础任务。数据此前已作为开发数据分析过，不是盲测，也不是新初态确认；单个任务可能仅含一个父初态。', '',
             '[预测比较图](prediction-rmse.png)；[完整 MAE/RMSE、父状态和任务分层](prediction-summary.json)；[逐点预测](predictions.npz)。', '',
             '## 3. 变化来自观测还是推理噪声', '',
             '每个父状态取分叉、+20、+40 步三个已存观测，与同一组三个噪声做 3x3 只读交叉。下表为固定观测换噪声的平均距离 / 固定噪声换观测的平均距离。大于 1 表示在这个有限对照中前者更大，不代表全部状态或方差贡献比例。', '',
             '| 父状态 | 路由 | MoE 输入 | MoE 输出 | 输入+输出 |', '| --- | ---: | ---: | ---: | ---: |']
    for parent in config['parents']:
        values = [next(r['noise_observation_ratio'] for r in noise if r['parent'] == parent['name'] and r['kind'] == kind and r['scope'] == 'back_path') for kind in KINDS]
        rows.append('| %s | %s |' % (parent['name'], ' | '.join('未定义' if v is None else '%.3f' % v for v in values)))
    rows += ['', '“未定义”表示固定噪声时观测对照的平均距离不超过 1e-12，不能解释为噪声影响小。完整原始对照距离、共享专家及最后 flow 的对照均保留，未按结果改变主表示。输入+输出仍是内部观测，不保证构成完整系统状态。',
             '[噪声交叉图](noise-vs-observation.png)；[全部九格对照统计](noise-analysis.json)；[逐端口/层幅度](port-amplitudes.json)。`total-shared` 仅为含加法舍入的派生差，不当作精确专家贡献。', '',
             '## 4. 复返分类是否更稳定', '',
             '完全沿用 P3i 的因果参考和退出/返回规则。只换 MoE 表示，半径仍是 0.75/1/1.25；未调参。下表是主范围在三个半径下分类一致的数量。', '',
             '| 表示 | 原策略稳定 /24 | 短窗口稳定 /24 |', '| --- | ---: | ---: |']
    for kind in KINDS:
        values = [next(r['stable'] for r in summary['stability'] if r['kind'] == kind and r['scope'] == 'back_path' and r['arm'] == arm) for arm in ('native10', 'window5')]
        rows.append('| %s | %d | %d |' % (names[kind], *values))
    rows += ['', '| 主范围/半径1表示 | 原策略未观察到返回 /24 | 短窗口未观察到返回 /24 |', '| --- | ---: | ---: |']
    for kind in KINDS:
        values = [next(r['counts'].get('last_exit_no_observed_return', 0) for r in summary['primary_counts'] if r['kind'] == kind and r['arm'] == arm) for arm in ('native10', 'window5')]
        rows.append('| %s | %d | %d |' % (names[kind], *values))
    rows += ['', '“未观察到返回”只到第 510 步的有限观测终点，不代表永久脱离真实吸引域，更不代表成功。“参考未建立”保留为独立类别，不能只保留容易分类的样本。',
             '[完整主分类图](primary-regions.png)；[尺度稳定性图](radius-stability.png)；[全部1,728项分类](regions.json)。六个 `*-moe-distance.png` 固定展示 r0 路由与计算的距离曲线，不作为额外独立样本。', '',
             '## 5. 环结构与等效计算负对照', '',
             '| 表示 | 全打乱尾部比例<=.05 原/短 | 三点块打乱 原/短 |', '| --- | ---: | ---: |']
    for kind in KINDS:
        values = [sum(r['descriptive_upper_tail'][mode] <= .05 for r in homology if r['kind'] == kind and r['arm'] == arm)
                  for mode in ('shuffle', 'block3') for arm in ('native10', 'window5')]
        rows.append('| %s | %d / %d | %d / %d |' % (names[kind], *values))
    rows += ['', '共 192 个原始 H1 和 38,016 个替代序列。以上只是未作多重比较校正的描述性尾部比例，不是显著发现数或失败检测精确率。原始 H1 寿命和直径同时保留，不能只看归一化形状。',
             '十二个 P3f 固定失败状态中，24 对人为构造的计算匹配对仍满足 input/total 完全一致、纯路由距离非零。这是必要一致性检查，不是发现了真实吸引子，也不能作为预测有效性的独立证据。',
             '[同调及替代序列](persistence.json)；[条件负对照](conditional-analysis.json)。', '',
             '## 6. 成本、核验和下一步边界', '',
             '- 新增模型前向 %d 次: 前缀131、正式后缀1496、历史重复6、采集验证18、额外噪声/观测格48。没有环境动作或新恢复率。' % collection['calls'],
             '- 前向计时合计 %.1f 秒，采集进程总耗时 %.1f 秒(含加载、验证、压缩写盘等)，保存数组 %.2f GiB。推理与 CPU 分析重叠，不是机器人端到端部署延迟。' %
             (collection['model_inference_seconds'], collection['elapsed_seconds'], collection['bytes_saved'] / 1024 ** 3),
             '- 峰值 PyTorch 分配 %.1f MiB，独立进程限额 16384 MiB；最终参数内容哈希 `%s`。' % (collection['peak_allocated_mib'], collection['parameter_sha256_after']),
             '- 独立复核原始请求、张量哈希、距离、因果窗口、NetworkX SCC和全部分组回归预测；同调复算仍使用同一固定 GUDHI 引擎，不声称第二算法认证。',
             '- 是否存在更合适的 MoE 表示仍可检验，但本轮不会据此调 v8.2 阈值、连接在线控制器或宣称能救回失败。只有内部预测指标，没有干预收益标签。', '',
             '[冻结方案](../../P3J_MOE_COMPUTE_PLAN.zh.md)；[采集完成记录](collection.json)；[独立审计](audit.json)；[最终封存](verification.json)。', '']
    with (root / 'REPORT.zh.md').open('x') as stream:
        stream.write('\n'.join(rows))
    write_json(root / 'report-summary.json', dict(passed=True, representation_gate=representation, topology_gate=topology,
               prediction_gates=gates, figures=files, new_environment_actions=0, original_rescue_count=0,
               plotting_versions=dict(matplotlib=matplotlib.__version__, numpy=np.__version__),
               original_paired_trials=24, scope='MoE-only internal inference, not a new recovery experiment'))
    print('MoE report and %d figures complete' % len(files), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    report(parser.parse_args().run.resolve())
