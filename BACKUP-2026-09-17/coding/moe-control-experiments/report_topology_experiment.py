"""Render audited topology experiments with explicit negative-result boundaries."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

CATEGORIES = ('no_confirmed_exit', 'exit_then_return', 'last_exit_no_observed_return',
              'insufficient_followup_or_mixed', 'reference_not_established')
SHORT = ('no exit', 'returned', 'outside', 'unclear', 'no reference')
CHINESE = ('未确认离开', '离开后返回', '末次离开后未观察到返回', '随访不足或混合', '参考区域未建立')
COLORS = ('#dbe2e8', '#d3b1c1', '#a2c9bc', '#e6d996', '#efefef')
ARM_COLORS = dict(native10='#287aa9', window5='#a74762')


def read(path):
    return json.loads(path.read_text())


def save_json(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def main(root):
    config, summary, audit = (read(root / name) for name in ('config.json', 'summary.json', 'audit.json'))
    assert summary['passed'] and audit['passed']
    regions, ph, controls = (read(root / name) for name in ('regions.json', 'persistence.json', 'crossover.json'))
    parents = [p['name'] for p in config['parents']]
    primary = [r for r in regions if r['representation'] == 'back_path' and r['scale'] == 1.]
    index = {(r['parent'], r['replicate'], r['arm']): r for r in primary}
    stability, variant_rows, differences = {}, [], []
    for representation in config['representations']:
        stability[representation] = {arm: sum(len({r['category'] for r in regions if r['parent'] == p and
            r['replicate'] == k and r['arm'] == arm and r['representation'] == representation}) == 1
            for p in parents for k in range(4)) for arm in ARM_COLORS}
        for scale in config['scales']:
            selected = [r for r in regions if r['representation'] == representation and r['scale'] == scale]
            variant_rows.append(dict(representation=representation, scale=scale,
                eligible_pairs=sum(r['reference_eligible'] for r in selected if r['arm'] == 'native10'),
                counts={arm: dict(Counter(r['category'] for r in selected if r['arm'] == arm)) for arm in ARM_COLORS}))
    for parent in parents:
        for replicate in range(4):
            native, short = (index[(parent, replicate, arm)] for arm in ARM_COLORS)
            if native['category'] != short['category']:
                differences.append(dict(parent=parent, replicate=replicate,
                                        native=native['category'], window=short['category']))
    ranks = {arm: {mode: sum(r['descriptive_upper_tail'][mode] <= .05 for r in ph if r['arm'] == arm)
                   for mode in ('shuffle', 'block3')} for arm in ARM_COLORS}
    derived = dict(passed=True, radius_stability=stability, variants=variant_rows,
                   different_primary_category_pairs=differences,
                   same_primary_category_pairs=24 - len(differences), descriptive_tail_le_005=ranks,
                   no_multiple_testing_significance_claim=True)
    save_json(root / 'post-analysis.json', derived)
    figures = []

    def finish(fig, path):
        assert not path.exists()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert min(image.size) >= 400 and np.ptp(pixels) > 100 and pixels.std() > 5
            figures.append(dict(path=str(path.relative_to(root)), width=image.width, height=image.height, nonblank=True))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, arm in zip(axes, ARM_COLORS):
        grid = np.asarray([[CATEGORIES.index(index[(p, k, arm)]['category']) for k in range(4)] for p in parents])
        ax.imshow(grid, cmap=ListedColormap(COLORS), vmin=0, vmax=4, aspect='auto')
        ax.set_xticks(range(4), ['old noise', 'new 1', 'new 2', 'new 3'])
        ax.set_yticks(range(6), parents, fontsize=9)
        for i in range(6):
            for j in range(4):
                ax.text(j, i, SHORT[grid[i, j]], ha='center', va='center', fontsize=9)
        ax.set_title(arm)
    fig.suptitle('Primary route-region diagnosis: every endpoint is still failure', fontsize=13)
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, SHORT)],
               loc='lower center', ncol=5, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .95))
    finish(fig, root / 'primary-regions.png')

    for parent in config['parents']:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for replicate, ax in enumerate(axes.flat):
            ax.axvspan(0, 20, color='#dbe2e8', alpha=.6)
            ax.axhline(1, color='#555555', linewidth=.8, linestyle='--')
            for arm, color in ARM_COLORS.items():
                row = index[(parent['name'], replicate, arm)]
                x = np.asarray(row['steps']) - parent['start']
                ratio = np.asarray(row['nearest_distance']) / row['epsilon']
                ax.plot(x, ratio, label=arm, color=color, linewidth=1.3)
            ax.set_title('Noise %d' % replicate, fontsize=10)
            ax.set_xlabel('Physical steps after intervention')
            ax.set_ylabel('Distance to fixed region / epsilon')
            ax.grid(alpha=.2)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle(parent['name'] + ': fixed causal reference; not a success score', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .95))
        finish(fig, root / (parent['name'] + '-region-distance.png'))

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    for parent, ax in zip(parents, axes.flat):
        for arm, color in ARM_COLORS.items():
            for replicate in range(4):
                row = next(r for r in ph if (r['parent'], r['replicate'], r['arm']) == (parent, replicate, arm))
                x = replicate + (-.1 if arm == 'native10' else .1)
                lower, median, upper = np.quantile(row['nulls']['block3'], [.05, .5, .95])
                ax.vlines(x, lower, upper, color=color, linewidth=7, alpha=.22)
                ax.scatter([x], [median], marker='_', color=color, s=90)
                ax.scatter([x], [row['observed']['normalized_h1']], color=color, s=25,
                           label=arm if replicate == 0 else None)
        ax.set_title(parent, fontsize=10)
        ax.set_xticks(range(4), ['old', 'new 1', 'new 2', 'new 3'])
        ax.set_ylabel('Max H1 lifetime / diameter')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Dots: observed topology; bands: block-shuffle 5-95% range (descriptive)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, root / 'persistence-vs-surrogates.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, field, title in zip(axes, ('routing_distance', 'computational_distance'),
                               ('Effective routing', 'MoE input + effective total')):
        matrices = [np.asarray(c[field]) for c in controls]
        averaged = np.mean([m / max(float(m.max()), 1e-12) for m in matrices], axis=0)
        ax.imshow(averaged, vmin=0, vmax=1, cmap='viridis')
        names = ['native', 'route only', 'joint', 'output only']
        ax.set_xticks(range(4), names, rotation=25, ha='right', fontsize=9)
        ax.set_yticks(range(4), names, fontsize=9)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, '%.2f' % averaged[i, j], ha='center', va='center',
                        color='white' if averaged[i, j] < .5 else '#202020', fontsize=10)
        ax.set_title(title, fontsize=11)
    fig.suptitle('Within-input distances: 12 failure-state cases, each normalized separately', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    finish(fig, root / 'conditional-equivalence.png')
    save_json(root / 'visual-check.json', dict(passed=True, figures=figures))

    lines = ['# P3i：MoE 拓扑与复返实验结果', '',
             '本轮只使用已有失败数据：P3h 的 48 条完整续跑、24 个配对、6 个 task/init 状态；另有 P3f 的 12 个定点交叉状态，来自其中 4 条失败父轨迹。新增模型前向 0，新增环境动作 0，没有补跑成功轨迹。', '',
             '## 结论', '',
             '当前结果不支持把“离开路由复返区域”当作恢复成功或自动控制器的可靠终止条件。部分轨迹可在选定路由表示中离开参考邻域却仍失败；多数分类依赖邻域尺度；路由邻域还可以区分有效计算完全相同的干预。', '',
             '这削弱的是本轮具体路由图指标的控制可用性，不是否定拓扑方法，也不证明不存在真实 trap、真正吸引域或更合适的 MoE 表示。', '',
             '## 1. 留在、返回，还是离开', '',
             '主设置：后四层全部 flow/动作 token 概率、三点延迟历史、十二点因果参考、基准 epsilon。两臂参考完全相同，24 对均满足预先固定的候选复返条件。', '',
             '| 主设置分类 | 原策略 /24 | 短窗口 /24 |', '| --- | ---: | ---: |']
    for name, translation in zip(CATEGORIES, CHINESE):
        lines.append('| %s | %d | %d |' % (translation, summary['primary_counts']['native10'].get(name, 0),
                                         summary['primary_counts']['window5'].get(name, 0)))
    lines += ['', '两臂各 12/24 被判为末次离开后未观察到返回，最终成功却仍为原来的 0/24。这说明该定义下离开路由区域不是成功的充分证据；不等于已确认离开真实物理/计算吸引域。', '',
              '同噪声配对中 %d/24 分类相同，%d/24 不同。原策略本身也会离开，不能把短窗口的离开数全部算作干预贡献。' % (24 - len(differences), len(differences)), '',
              '| 分类不同的配对 | 原策略 | 短窗口 |', '| --- | --- | --- |']
    translate = dict(zip(CATEGORIES, CHINESE))
    for row in differences:
        lines.append('| %s r%d | %s | %s |' % (row['parent'], row['replicate'], translate[row['native']], translate[row['window']]))
    lines += ['', '末次路由观测在第 510 步，不是第 520 步。所谓“未观察到返回”要求末尾至少四次在外的十步网格观测，覆盖三十步，只是有限随访，不是永久逃离。参考区域也只是 MoE 观测空间中的候选，不是经过 Conley 理论认证的真实不变集。', '',
              '[主分类图](primary-regions.png)；逐场景距离图以 `*-region-distance.png` 保存。', '',
              '## 2. 尺度稳定性', '',
              '只改变半径倍数 0.75、1、1.25，保持表示与所有数据不变，各表示的三尺度分类一致数如下。无法建立参考区域也算一种分类，不将其隐藏。', '',
              '| 表示 | 原策略稳定 /24 | 短窗口稳定 /24 |', '| --- | ---: | ---: |']
    for representation, row in stability.items():
        lines.append('| %s | %d | %d |' % (representation, row['native10'], row['window5']))
    lines += ['', '主表示仅各 4/24 保持分类一致；三种表示与三种半径的九设置全部一致数，两臂均为 0/24。改变表示是在改变观测内容，不能单凭九设置不一致断言方法必然错误；但在缺少已验证表示/尺度依据时，不能挑一个设置宣称真实脱困。', '',
              '| 表示 / 半径 | 建立参考的配对 /24 | 原策略未返回 /24 | 短窗口未返回 /24 |', '| --- | ---: | ---: | ---: |']
    for row in variant_rows:
        lines.append('| %s / %.2f | %d | %d | %d |' % (row['representation'], row['scale'], row['eligible_pairs'],
            row['counts']['native10'].get('last_exit_no_observed_return', 0), row['counts']['window5'].get('last_exit_no_observed_return', 0)))
    lines += ['', '## 3. 持久同调是否找到稳定的环', '',
              '用 GUDHI 计算完整 Rips 过滤中的 H1，按点云直径归一化；每分支分别进行 99 次全打乱和 99 次长度三的分块打乱，再重建延迟嵌入。共 9,504 个真实数据替代序列。经验上尾比例不超过 0.05 的个数：', '',
              '| 对照方式 | 原策略 /24 | 短窗口 /24 |', '| --- | ---: | ---: |',
              '| 全打乱 | %d | %d |' % (ranks['native10']['shuffle'], ranks['window5']['shuffle']),
              '| 三点分块打乱 | %d | %d |' % (ranks['native10']['block3'], ranks['window5']['block3']), '',
              '这些只是未作多重比较校正的描述性尾部比例，不是显著发现数，也不是检测精确率/召回率。当前结果没有提供跨失败样本普遍存在强环形结构的证据；不能由此断言真实 loop 不存在，因为表示、17--37 个嵌入点及有限观察窗口都限制检出能力。归一化比例仅比较形状，近静态的小幅噪声也可能产生非零比例；原始 H1 寿命与直径同时保存在 persistence.json，不能只凭归一化值认定循环或有效计算改变。', '',
              '合成周期轨迹的归一化 H1 为约 0.541，两种打乱对照尾部比例均为 0.01；静止和单向漂移的 H1 为零，且漂移不满足参考复返条件。离开返回与持续离开的合成分类也按预期通过。合成阳性是几何现象，不是成功机器人轨迹。', '',
              '[持久同调与打乱对照](persistence-vs-surrogates.png)。', '',
              '## 4. 路由与有效计算的负对照', '',
              '十二个固定失败状态的四候选对照中，24/24 个计算匹配对的 MoE input、effective total 与最终动作逐元素一致，但路由距离均非零。每个状态在路由度量下有四个不同点，在输入与有效输出度量下只有两个零距离等价类。', '',
              '因此，给纯路由距离加上邻域、H0 或拓扑术语，并不能自动解决“路由改变但传递的计算不变”的问题。这是同输入条件下人为构造的隔离对照，不是新测出的闭环吸引子；H0 合并尺度只描述局部邻域，不代表语义上的好坏。', '',
              '这里借用等效类思想也不意味着 MoE input+total 已经是完整系统的充分状态，更不保证未来行为相同。', '',
              '[条件等效计算对照](conditional-equivalence.png)。', '',
              '## 下一步边界', '',
              '不根据本轮结果调半径或增加恢复窗口，也不把当前路由区域指标接成在线控制器。连续有效 MoE input/total 数据在 P3h 中没有保存，现有十二个定点不能补成连续轨迹。若继续检验“计算状态是否复返”，需要另行固定连续有效输出采集与表示，再用同一批失败配对观察，而不是把当前不稳定路由标签当真值。', '',
              '原恢复实验仍然是 0/24 救回；本轮没有产生新的恢复率或证明控制有效性。', '',
              '## 核验与产物', '',
              '432 个分支/设置判定、216 个配对参考图、48 个原始持久同调与 9,504 个替代序列已复核。距离用独立 Hellinger/RMS 公式，SCC 用 NetworkX 复算，时间段用独立逻辑；持久同调复算仍使用同一个经过合成校验的 GUDHI 引擎，不声称有第二个同调算法交叉认证。', '',
              '[冻结配置](config.json)、[全部分类](region-summary.csv)、[后处理统计](post-analysis.json)、[独立审计](audit.json)、[最终测试](tests.txt)、[封存清单](verification.json)。新增模型推理与环境动作均为零。', '',
              '理论出处、预先固定的参数和限制见 [P3I_TOPOLOGY_PLAN.zh.md](../../P3I_TOPOLOGY_PLAN.zh.md)。']
    with (root / 'REPORT.zh.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps(dict(figures=len(figures), radius_stability=stability, paired_differences=len(differences), ranks=ranks)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
