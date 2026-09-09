"""Generate standalone figures and a Chinese report from measured results."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"prior": "#737373", "budget": "#bd5742", "moe_raw": "#679242", "moe": "#007c83"}


def make_figures(output, scores, reliability, sensitivity, predictions, episodes):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, selection in zip(axes[0], ("all_queries", "first_alarm_half_k4")):
        block = reliability[(reliability.split == "test_unseen_init") & (reliability.selection == selection)]
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        for name in ("budget", "moe_raw", "moe"):
            group = block[block.model == name]
            ax.plot(group.predicted, group.observed, "o-", label=name, color=COLORS[name])
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted success probability",
               ylabel="Observed success fraction", title=f"Unseen initial states: {selection}")
        ax.legend()
    block = scores[(scores.split == "test_unseen_init") & (scores.suite == "all")
                   & scores.selection.isin(["all_queries", "first_alarm_half_k4", "q7", "q15"])]
    categories = ["all_queries", "first_alarm_half_k4", "q7", "q15"]
    for i, name in enumerate(("budget", "moe")):
        group = block[block.model == name].set_index("selection")
        axes[1, 0].bar(np.arange(4)+(i-0.5)*0.3, group.loc[categories, "brier"], width=0.3,
                       label=name, color=COLORS[name])
    axes[1, 0].set(xticks=np.arange(4), xticklabels=["all queries", "half-k4 alarm", "q7", "q15"],
                   ylabel="Brier score (lower is better)",
                   title="Held-out probability quality")
    axes[1, 0].legend()
    chosen = ["freeze_back_quarter", "freeze_back_half_k4", "freeze_back_half", "recurrence_back_025"]
    block = sensitivity[sensitivity.split == "all_natural"].set_index("rule").loc[chosen]
    low = np.maximum(0, block.success_rate - block.wilson_low)
    high = np.maximum(0, block.wilson_high - block.success_rate)
    axes[1, 1].bar(np.arange(len(block)), block.success_rate,
                   yerr=np.stack([low, high]), color="#007c83", capsize=4)
    axes[1, 1].set(xticks=np.arange(len(block)), xticklabels=["quarter", "half-k4", "half", "recurrence .25"],
                   ylim=(0, 1.12), ylabel="Observed success after first alarm",
                   title="Existing rules; Wilson intervals (iid reference)")
    for i, row in enumerate(block.itertuples()):
        upper = row.wilson_high if np.isfinite(row.wilson_high) else 0
        axes[1, 1].text(i, upper + 0.04,
                        f"{row.successes}/{row.alarmed}", ha="center", fontsize=9)
    fig.savefig(output / "probability_audit.png", dpi=180)
    fig.savefig(output / "probability_audit.pdf")
    plt.close(fig)

    # Label-stratification is only for retrospective illustrations, not checkpoint selection.
    candidates = episodes[(episodes.split == "test_unseen_init") & (episodes.first_q_freeze_back_half_k4 >= 0)]
    selected = []
    for outcome in (True, False):
        selected.extend(candidates[candidates.success == outcome].head(2).episode_row.tolist())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, episode in zip(axes.flat, selected):
        row = episodes.loc[episodes.episode_row == episode].iloc[0]
        group = predictions[predictions.episode_row == episode]
        for name in ("budget", "moe"):
            ax.plot(group["query"], group[name], label=name, color=COLORS[name])
        ax.axvline(row.first_q_freeze_back_half_k4, color="#333333", linestyle=":", label="half-k4 first alarm")
        ax.set(ylim=(0, 1), xlabel="Query (zero based, before action)", ylabel="Success probability",
               title=f"e{episode}, {row.suite}, observed Y={int(row.success)}")
        ax.legend(fontsize=8)
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)
    fig.savefig(output / "example_prefixes.png", dpi=180)
    plt.close(fig)


def make_report(output, audit, scores, alarms, uncertainty, raw_audit, candidates, sensitivity):
    primary = scores[(scores.split == "test_unseen_init") & (scores.suite == "all")]
    overall = alarms[(alarms.split == "all_natural") & (alarms.suite == "all")].iloc[0]
    delta = uncertainty[(uncertainty.selection == "all_queries")
                        & (uncertainty.metric == "moe_minus_budget_brier")].iloc[0]
    lines = ["# MoE 前缀成功概率：离线实验报告", "",
             "## 实际结论", "",
             f"在两个完整自然执行批次的 {audit['episodes']:,} 条 rollout 中，严格主规则首次报警后，"
             f"最终成功 **{overall.alarmed_successes}/{overall.alarmed_episodes} = "
             f"{overall.success_after_first_alarm:.2%}**。",
             f"Wilson 95% 区间（独立样本参考，不校正 seed/初态相关）为 "
             f"[{overall.wilson_low:.2%}, {overall.wilson_high:.2%}]。零成功时经验 bootstrap 退化，"
             "不报告虚假的 [0,0] 概率区间。这个严格规则缺少报警后成功样本，不能证明真实概率为零。"
             "完整现有规则的描述性敏感性结果如下，未按终局调参或替换主规则。", "",
             "| 既有规则 | 报警集数 | 报警后成功 | 成功率 |",
             "|---|---:|---:|---:|"]
    for row in sensitivity[sensitivity.split == "all_natural"].itertuples():
        rate = f"{row.success_rate:.2%}" if np.isfinite(row.success_rate) else "NA"
        lines.append(f"| {row.rule} | {row.alarmed} | {row.successes} | {rate} |")
    lines.extend(["", "这些是 D 条件下的终局成功率，不是真实 trap 的恢复率。"
                  "全部规则都保留，不从中挑选最符合假设的结果。half-k4 是仓库已有的探索性规则，"
                  "其阈值为 0.5、连续确认 4 次；额外报告其首次报警概率诊断，不改变已固定的模型。", "",
             f"未见初始状态测试集上，MoE 模型相对预算基线的逐 query Brier 差为 "
             f"**{delta.estimate:+.5f}**，95% 区间 [{delta.low:+.5f}, {delta.high:+.5f}]。"
             + ("区间低于零，支持该测试分布下 MoE 提供额外概率预测信息。" if delta.high < 0
                else "区间未完全低于零，尚不能确认 MoE 超越预算基线。"), "",
             "## 样本与目标", "",
             f"覆盖 40 个 LIBERO 任务、4 个 suite、{audit['queries']:,} 个执行前 query；"
             f"成功 {audit['successes']:,} 集，失败 {audit['failures']:,} 集。",
             "只使用 right-50x8-20260903（噪声 1000..1007）与 right-50x8b-20260903"
             "（1008..1015）。旧 16x32 批次不并入，避免重叠初态与种子重复计数；"
             "pin 干预、重复 pin-base、smoke 和不完整 CALVIN 不进入自然概率标定。", "",
             "估计目标是 P(Y=1 | 当前 MoE 因果历史、已执行步数、剩余步数、仍在执行)。"
             "每个 suite 的 checkpoint、归一化、采样配置分别验证，并训练独立模型。"
             "任务 ID、初态、种子、物理状态、动作、终局长度与未来窗口均不作为在线输入。", "",
             "q 从 0 开始；R_q 在本 chunk 推理后、执行前可用。剩余动作预算是 max_steps - 10*q，"
             "包含已经采样但尚未执行的当前 chunk。使用真实预算上限，绝不使用 length-q。"
             "失败必须完整耗尽原定动作预算；提前中断会拒绝进入训练。", "",
             "## 数据隔离", "",
             "固定 seed=20260907，逐任务将 50 个初态按哈希种子置乱：30 个用于第一批训练"
             "（9,600 集），10 个用于第一批校准（3,200 集），另 10 个在两批中共同作为主测试"
             "（6,400 集）。训练、校准、主测试的 task/init 集合完全不相交。",
             "第二批其余 12,800 集仅作新噪声、已见初态的次测试；它不是独立初态泛化证据。"
             "同一 rollout 所有 chunk 永远处于同一分组。分支数据尚未加入训练。", "",
             "本语料已被仓库内其他实验分析，当前划分是可复现的回顾评估，不是前瞻盲测。", "",
             "## 模型与指标", "",
             "冻结报警器为现有 history-only PRIMARY: freeze_back_quarter，初始 4 次转移作基线，"
             "后层 mobility 降至 1/4，2-query 平滑、2 次确认。最早 q7 报警。"
             "alarm_now 与历史上曾报警分别保存；二者都不命名为真实 Z。", "",
             "模型为固定 120 轮、15 叶的 HistGradientBoostingClassifier。"
             "预算基线和 MoE 模型使用相同容量；sigmoid 校准只读取独立校准集。"
             "保留原始和校准后概率，测试结果不用于选择参数或校准方式。", "",
             "| 测试位置 | 模型 | Brier | Log loss | AUROC | ECE (10 bins) |",
             "|---|---|---:|---:|---:|---:|"])
    for selection in ("all_queries", "first_alarm", "first_alarm_half_k4", "q7", "q15"):
        for model in ("prior", "budget", "moe_raw", "moe"):
            row = primary[(primary.selection == selection) & (primary.model == model)].iloc[0]
            auc = "NA" if not np.isfinite(row.auroc) else f"{row.auroc:.4f}"
            lines.append(f"| {selection} | {model} | {row.brier:.5f} | {row.log_loss:.5f} | {auc} | {row.ece10:.5f} |")
    primary_count = primary[(primary.selection == "first_alarm") & (primary.model == "moe")].iloc[0].rows
    lines.extend(["", f"严格主规则的主测试首次报警只有 {primary_count} 个样本，单类 AUROC 为 NA，"
                  "该子组指标及 bootstrap 仅作小样本诊断，不支持校准有效性的结论。"
                  "全体 query 的 sigmoid 校准也未必优于原始模型；原始与校准结果均公开，未按测试集择优。", "",
                  "逐 query 的主指标对应部署时实际遇到的决策分布。长轨迹产生更多决策，"
                  "但置信区间以任务内初态聚类，不把相邻 chunk 当独立样本。另报告每集等权的"
                  "损失诊断、固定 q 的风险集以及每集一次的首次报警指标。训练不采用 1/终局长度"
                  "加权，因为这种依赖未来的权重会改变条件成功概率的目标分布。", "",
                  "Brier/log loss 同时受区分能力与校准影响；请结合可靠性曲线和 bin 样本数读取。"
                  "整体校准良好不保证首次报警等子群校准良好。参见 "
                  "[scikit-learn 校准文档](https://scikit-learn.org/stable/modules/calibration.html)。", "",
                  "![概率质量与报警成功率](probability_audit.png)", "",
                  "![逐 chunk 概率示例](example_prefixes.png)", "",
                  "示例图按已有 half-k4 规则报警后实际成功/失败各抽取两集，"
                  "仅用于回顾展示；分支采样不使用终局标签。", "",
                  "## 可识别性与剩余证据", "",
                  "真实 trap Z、首次物理脱困时间与不可逆失败时间缺少一致的独立标签。"
                  "现有 physical-failure-labels 主要对失败集做物理原因标注，成功集没有同等的"
                  "局部受困核验。已有 centered-window 的几何事件代理也不能直接替代部署时 Z。",
                  "因此 natural_escape_probability、trap_probability 保持 null，未训练伪造的"
                  "脱困头；报警消失不被当作物理脱困。超时对预算内成功 Y 是 0，不是删失；"
                  "若以后做无界脱困时间分析，截止时未脱困才涉及生存删失。", "",
                  "不同 rollout 的相同 q 不代表同一物理检查点。当前读出是表征条件概率，"
                  "不是同一个完整 x_t 经多次续跑的 committor 真值，也不能通过任意改小预算"
                  "输入来宣称获得已验证的反事实 V(b) 曲线。", "",
                  f"已生成 {len(candidates)} 个报警/对照检查点候选（见 branch_candidates.csv），"
                  "按与终局无关的哈希选取，每任务最多 2 个首次报警，并匹配同任务、同 q、同预算"
                  "且截至此时未报警的对照；真实任务阶段仍未匹配。局部受困证据须离线盲核验。",
                  "另提供 branch_candidates_half_k4.csv，记录已有 half-k4 规则的相同采样设计，"
                  "不替换主规则。两个清单的相同 checkpoint_id 是同一个父检查点，未来采集/划分须合并。",
                  "候选只是定位清单。Hub 的 sim_state/动作记录不等于完整闭环快照；当前没有"
                  "与此检测时刻对齐的完整状态分支证据。必须复原模拟器、控制器与观测历史、"
                  "固定当前已采样 chunk，再从下一个 chunk 重抽独立未来噪声。每点 K=64，"
                  "保持原始剩余预算，保存恢复审计与来源哈希。", "",
                  "既有恢复实验包含失败 trunk 筛选、不同报警规则或重抽当前 chunk，"
                  "不能直接混入此处的自然条件概率监督。branch-summary 接口只接收满足完整恢复、"
                  "当前 chunk 保留、策略固定、独立未来 RNG 等契约的记录，并输出带试验次数"
                  "与 Wilson 区间的软标签；0/K 不被解释为真实概率为零。", "",
                  "95% bootstrap 区间固定已采样任务，按 task/init 重采样；未涵盖任务迁移、"
                  "共享 seed 引起的跨初态依赖或模型重新训练的不确定性。", "",
                  "## 验证", "",
                  f"校验 80 个 run 的原始 summaries、采集预算、策略元数据、MoE 特征缓存 SHA-256"
                  f"与逐集终局标签；首次报警与现有检测器逐集一致。原始 Zarr 抽查 "
                  f"{raw_audit['source_runs']} 个 run、{raw_audit['queries']} 个 query，"
                  f"流式/离线特征最大绝对差 {raw_audit['max_feature_error']:.3g}。",
                  "原始大数组只读打开。特征提取、分组、防未来泄漏、预算边界、分支证据契约与"
                  "在线读出均有针对性测试。完整 provenance 见 data_audit.json、raw_verification.json"
                  "和 artifact_manifest.json。"])
    (output / "REPORT.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
