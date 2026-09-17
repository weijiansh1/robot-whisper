"""Render the audited MoE crossover and its interpretation limits."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = ('native', 'joint', 'route_only', 'output_only')
NAMES = ('Native', 'Joint', 'Route + O0', 'R0 + O1')
ZH = dict(native='原生', joint='正常路由干预', route_only='改路由、回填原生输出', output_only='原门控、回填候选输出')
COLORS = ('#707070', '#19836a', '#bc4b45', '#397ba5')


def load(path):
    return json.loads(path.read_text())


def main(root):
    summary, collection = load(root / 'summary.json'), load(root / 'collection.json')
    assert load(root / 'audit.json')['passed']
    by_label = {r['label']: r for r in summary['aggregate']}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    charts = (('top4_changed_fraction', 100., 'Changed top-4 sets (%)'),
              ('effective_total_relative', 100., 'Effective MoE output relative change (%)'),
              ('action_rms', 1., 'Normalized 6D action RMS'))
    for ax, (field, scale, title) in zip(axes[0], charts):
        values = [by_label[k]['means'][field] * scale for k in LABELS]
        ax.bar(NAMES, values, color=COLORS, width=.6)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(values) * 1.3 if max(values) > 0 else 1)
        for i, v in enumerate(values):
            ax.annotate('%.4g' % v, (i, v), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10)
    ticks = [f'P{i + 1}-{phase}' for i in range(5) for phase in ('B', 'F', 'L')]
    for ax, index, title in ((axes[1, 0], 0, 'Instant freeze change / margin'),
                             (axes[1, 1], 3, 'Instant inversion change / margin')):
        for label, name, color in zip(LABELS[1:], NAMES[1:], COLORS[1:]):
            rows = [r for r in summary['rows'] if r['label'] == label]
            ax.plot(range(15), [r['instantaneous_delta'][index] for r in rows], marker='o', markersize=3,
                    linewidth=1.2, color=color, label=name)
        ax.axhline(-.05, color='#555555', linestyle=':', linewidth=1, label='Target screen boundary')
        ax.set_xticks(range(15), ticks, rotation=55, ha='right', fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc='best')
    ax = axes[1, 2]
    values = [by_label[k]['instantaneous_screen_count'] for k in LABELS]
    ax.bar(NAMES, values, color=COLORS, width=.6)
    ax.set_ylim(0, 18)
    ax.set_yticks(range(0, 16, 3))
    ax.set_title('Instantaneous screen passed (of 15)', fontsize=11)
    for i, v in enumerate(values):
        ax.text(i, v + .4, str(v), ha='center')
    for ax in axes.flat:
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('MoE crossover: 15 paired development inputs, 135 main forwards\n'
                 'Whole output clamp at HB L12-15; original weights; no environment actions', fontsize=15)
    path = root / 'crossover.png'
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=150)
    plt.close(fig)

    lines = ['# P3f: MoE 路由与输出交叉对照', '',
        '本轮仅研究 MoE，不启动仿真、不执行环境动作、不用物理停滞或任务标签选择干预。原模型参数内容哈希前后一致。', '',
        '## 核心结果', '',
        '| 组别 | 受控 top-4 集合变化 | 瞬时筛查通过 | 五项正式分数均低于阈值 | 动作差 RMS |',
        '| --- | ---: | ---: | ---: | ---: |']
    for label in LABELS:
        row = by_label[label]
        lines.append('| %s | %.2f%% | %d/15 | %d/15 | %.8f |' % (ZH[label],
            row['means']['top4_changed_fraction'] * 100, row['instantaneous_screen_count'],
            row['formal_all_below_count'], row['means']['action_rms']))
    route, output, joint = (by_label[k] for k in ('route_only', 'output_only', 'joint'))
    lines += ['', 'top-4 变化按 L12-15、10 个去噪步、10 个动作 token 统计，比较专家集合而非排序。动作差为相对原生的六维动作差，除以训练归一化标准差后计算 RMS；不是物理移动量。', '',
        '瞬时筛查沿用 P3d 的 FP16 路由监控口径: freeze 与 inversion 各改善至少 0.05 个 margin，其他分量恶化不超过 0.05，动作 RMS 不超过 0.05 且夹爪符号不变。它不等于正式平滑分数低于阈值，更不等于任务恢复。原生相对自身没有改善，因此筛查为零并不表示原生全部报警。所有组均使用同一真实历史前缀。', '',
        '精度敏感性: 历史仍为同一 FP16 前缀、仅当步使用 FP32 概率时，原门控回填候选组为 %d/15；正常干预和改路由回填原生仍均为 15/15。不把 FP16 的 0/15 写成所有精度下必然为零。图中有效 MoE 输出差采用四层动作 token 的整体 L2 比值，不是 P3e 的逐层比值均值，也不是信号衰减率。' % output['fp32_screen_count'], '',
        '## 交叉对照说明了什么', '',
        '- 改路由、回填原生输出组: 15/15 与原生的 HB 输入、共享输出、有效 total、残差、block、投影输入、速度和最终动作逐元素一致；其中 %d/15 仍通过既有瞬时筛查。' % route['instantaneous_screen_count'],
        '- 原门控、回填候选输出组: 15/15 与正常干预的上述有效计算和动作逐元素一致；筛查通过为 %d/15，正常干预为 %d/15。' % (output['instantaneous_screen_count'], joint['instantaneous_screen_count']),
        '- 同有效计算却不同正式路由分数: route_only/native 为 %d/15，output_only/joint 为 %d/15。' %
            (sum(p['route_vs_native_scores_differ'] for p in summary['pairs']), sum(p['output_vs_joint_scores_differ'] for p in summary['pairs'])),
        '- route_only 在有效计算完全不变时，各分量由阈值上方跨到下方的次数为 %s，顺序是 freeze、acceleration、periodicity、inversion、curvature；不是这些状态的五项分数都跨阈值，也不是独立样本数。' % route['formal_cross_down_counts'],
        '- 计算和动作匹配是输出钳制的设计结果，不是证明路由无用。实际发现是，现有路由分数可以与传递的有效计算分离；通过这些指标不足以证明计算被纠正。',
        '- “原门控”仅指不加偏置。output_only 的有效概率精确等于 joint 在相同隐藏态上的偏置前概率；相对 native 轨迹仍有 %.2f%% 的 top-4 集合变化，这是下游输入反馈，不是直接门控干预。' % (output['means']['top4_changed_fraction'] * 100), '',
        '## 范围修订与数值定位', '',
        '初始 action-token-only 批次在第 5 次前向后停止: 原生、正常干预、两次原样回填均通过，第五次交叉未通过中间输入精确匹配。前四次原始数据和失败记录保留；第五次前向完成但完整张量未落盘，此限制未隐去。', '',
        '随后补两次 action-token-only 定位，再按预注册修订执行 135 次完整输出对照。正式 gate 偏置仍只作用于动作 token；输出回填覆盖 L12-15 的全部 11 个 token，包括状态 token。没有更换输入、噪声或偏置强度。', '']
    for row in summary['scope_diagnostics']:
        first = row['first_hb_difference']['mechanism/input']
        site = row['first_unclamped_output_difference']
        lines.append('- %s 定位: 去噪步 %d、L%d 的状态 token 输入、门控概率/ID/权重和共享输出相同，但 raw total 状态输出最大差 %.8g；后续首个 HB 输入差异为 %s。' %
            (row['label'], site['denoise_step'], site['layer'], row['first_divergence_state_output']['max_abs_delta'], json.dumps(first, ensure_ascii=False)))
    lines += ['', '这定位到未回填状态 token 的数值耦合。源码将 token 按专家聚合后批量计算，分派改变会改变批量形状或 token 次序；这与数值差异相容，但本轮未单独隔离具体 GEMM/kernel 原因，不将该解释写成已证明的底层机制。', '',
        '## 逐父轨迹', '',
        '| 父轨迹 | 正常干预通过 | 改路由回填原生通过 | 原门控回填候选通过 |',
        '| --- | ---: | ---: | ---: |']
    for parent in dict.fromkeys(r['parent'] for r in summary['rows']):
        rows = {r['label']: r for r in summary['by_parent'] if r['group'] == parent}
        lines.append('| %s | %d/3 | %d/3 | %d/3 |' % (parent, *[rows[k]['instantaneous_screen_count'] for k in ('joint', 'route_only', 'output_only')]))
    lines += ['', '## 验证与边界', '',
        '135 次正式调用由 60 个主对照观测与 75 个原样回填、重复和撤销验证组成。加 2 次定位、初始 5 次，实际总计 142 次；不是 142 个独立样本。分析单位为 15 个配对状态、5 条父轨迹。10 个状态此前已经按候选接受选出，另 5 个为既有报警前状态，不是留出泛化评估。', '',
        '30 个新供体精确复现 P3e；75 个验证控制均逐元素一致；两类交叉共 30 个计算匹配对通过。独立检查真实 logits、BF16 偏置加法、FP32 softmax、top-4 dispatch、回填范围、原始/有效输出、残差重建以及五个分数。未改变既有监控历史。', '',
        '本轮不报告救回率、精确率或召回率，也未证明任一内部表示是任务成功的充分原因。后续 MoE 控制应将路由报警、专家合成输出响应和跨层/跨去噪稳定性分开评估，不应仅按可直接优化的路由分数选候选。', '',
        '## 产物', '',
        '- [图表](crossover.png)', '- [60 个主单元的逐项数据](cells.csv)',
        '- [完整五项分数与配对统计](summary.json)', '- [两次范围定位](scope-diagnostics/summary.json)',
        '- [独立审计](audit.json)', '- [执行说明](EXECUTION_NOTES.zh.md)',
        '- [配置与来源哈希](config.json)', '- [最终封存](verification.json)', '']
    with (root / 'REPORT.zh.md').open('x') as stream:
        stream.write('\n'.join(lines))
    notes = ['# P3f 执行说明', '',
        '采用新文件和新 run 保留原始协议、初始失败与修订版。初始失败批次源码和全部文件由修订版 config.json 记录哈希，不覆盖补采。', '',
        '- 预算: 5 次初始批次 + 2 次范围定位 + 135 次正式交叉 = 142 次实际前向。',
        '- 修订版采集含加载、两次完整参数哈希与写盘: %.2f 秒。' % collection['elapsed_seconds'],
        '- 修订版模型前向计时合计: %.2f 秒；不含额外捕获转换、压缩写盘和哈希。' % summary['inference_seconds'],
        '- 隔离进程 PyTorch 峰值 reserved: %.0f MiB，分配器上限 16384 MiB。' % summary['peak_reserved_mib'],
        '- 原模型内容 SHA256 前后一致: `%s`。' % collection['parameter_sha256_after'],
        '- 新环境动作和模拟器状态读取均为 0；共享 9500 仅查询身份元数据，无推理请求。',
        '- 原样回填/撤销控制不比较 replay_token_mask 和 patch_sites，因为这些是按设计不同的处理记录；路由、原始/有效计算和动作全部精确比较。',
        '- 重复次数用于检验确定性，不构成额外独立样本，不计算虚假的窄置信区间。',
        '- 封存后不要重新运行会写产物的分析、绘图或 finalizer；查看已有报告和 JSON 即可。', '']
    with (root / 'EXECUTION_NOTES.zh.md').open('x') as stream:
        stream.write('\n'.join(notes))
    print(json.dumps(dict(report=str(root / 'REPORT.zh.md'), figure=str(path))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run.resolve())
