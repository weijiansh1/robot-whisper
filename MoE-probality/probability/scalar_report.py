"""Render measured scalar-calibration results without fitting or model selection."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = dict(prior="校准集常数先验", support_current="支持度，当前值",
              support_window4="支持度，最近 4 值均值（主方法）",
              support_prefix_mean="支持度，历史均值", support_prefix_max="支持度，历史最大值",
              freeze="既有冻结分数校准", supervised_moe="原无预算监督 MoE 模型",
              clock_cap="原时钟＋上限基线", prior_unseen_task="未见任务：常数先验",
              support_current_unseen_task="未见任务：支持度当前值",
              support_window4_unseen_task="未见任务：支持度最近 4 值均值",
              support_prefix_mean_unseen_task="未见任务：支持度历史均值",
              support_prefix_max_unseen_task="未见任务：支持度历史最大值",
              freeze_unseen_task="未见任务：既有冻结分数校准",
              supervised_moe_unseen_task="未见任务：原监督 MoE 模型")


def report(output):
    scores = pd.read_csv(output / "metrics.csv")
    summary = json.loads((output / "summary.json").read_text())
    raw = json.loads((output / "raw_verification.json").read_text())
    fit_audit = json.loads((output / "fit_audit.json").read_text())
    matched = pd.read_csv(output / "matched_auroc.csv")
    intervals = pd.read_csv(output / "cluster_intervals.csv")
    reliability = pd.read_csv(output / "calibration.csv")
    dynamics = pd.read_csv(output / "probability_dynamics.csv")
    coefficients = pd.read_csv(output / "calibrators.csv")
    counts = pd.read_csv(output / "row_counts.csv")
    main = scores.query("split == 'test_unseen_init' and suite == 'all' and selection == 'all_q7plus'").set_index("model")
    same_q = matched[matched.selection == "task_query"].set_index("model")
    same_stage = matched[matched.selection == "task_query_stage"].set_index("model")
    selected = list(LABELS)
    local, held = main.loc["support_window4"], main.loc["support_window4_unseen_task"]
    stage_local = same_stage.loc["support_window4"]
    held_difference = intervals[(intervals.left == "support_window4_unseen_task") &
                                (intervals.right == "prior_unseen_task") & (intervals.metric == "brier")].iloc[0]
    lines = ["# VLAConf 思路的 MoE 概率校准实验", "",
             "## 主要发现", "",
             f"局部成功支持度加两参数校准的总体 AUROC 为 {local.auroc:.4f}，Brier 为 {local.brier:.5f}；"
             f"同点比较的原无预算监督 MoE 模型为 {main.loc['supervised_moe', 'auroc']:.4f} / "
             f"{main.loc['supervised_moe', 'brier']:.5f}。这版固定分数有总体预测价值，但损失明显更高。", "",
             f"在同任务、同 q、同物理阶段的比较中，主方法 AUROC 为 {stage_local.matched_auroc:.4f} "
             f"[{stage_local.low:.4f}, {stage_local.high:.4f}]，未发现稳定的局部区分能力。"
             "这是匹配覆盖子集上的结果，不代表所有状态都没有信息，也不能将总体与匹配结果的差额作因果分解。", "",
             f"未见任务的局部方法 AUROC 为 {held.auroc:.4f}，Brier 为 {held.brier:.5f}。"
             f"相对常数先验的 Brier 差为 {held_difference.difference:+.5f} "
             f"[{held_difference.low:+.5f}, {held_difference.high:+.5f}]，区间跨过零；"
             "跨任务的绝对概率尚无稳定优于先验的证据。", "",
             f"历史最大值在本次总体排名更好：共享模型 AUROC {main.loc['support_prefix_max', 'auroc']:.4f}，"
             f"未见任务为 {main.loc['support_prefix_max_unseen_task', 'auroc']:.4f}。"
             "代价是概率不能反映恢复后的回升，而且其未见任务 Brier 仍高于原监督模型。"
             "没有据此替换预先指定的局部主方法。", "",
             "## 本次实现", "",
             "本实验将成功支持度分数与终局概率校准分开。VLAConf 使用成功示范训练 CFN；"
             "这里固定既有 18 维 MoE 局部特征，用成功训练轨迹的近邻距离替代 CFN。"
             "这是方法借鉴，不是原论文网络、输入或成绩的复现。", "",
             "成功训练前缀用于拟合逐坐标均值/标准差并建立参考库；k 固定为 20。"
             "当前风险分数为标准化空间中最近 20 个成功前缀的平均欧氏距离，再取 log1p。"
             "不训练新的高容量预测模型或神经网络。参考库选择仍使用成功标签，不能称为完全不依赖标签。", "",
             "主方法取最近 4 个可用分数的均值，拟合 p= sigmoid(-alpha*u+beta)，alpha>=0。"
             "独立校准集提供成败标签，用来拟合这两个参数。每种聚合单独校准，所有任务/suite 共用映射。"
             "另保留当前分数、历史均值、历史最大值，以及既有冻结分数的固定对照。", "",
             "## 数据与目标", "",
             f"复用完整 {summary['source_episodes']:,} 集、{summary['source_queries']:,} 个原始 query 的已验证缓存；"
             f"q>=7 的 {summary['ready_queries']:,} 个前缀符合本方法输入要求。"
             "前 7 个 query 不输出概率，不按最终长度或执行百分比选点。", "",
             "| 集合 | 轨迹数 | 前缀数 | 成功前缀数 |", "|---|---:|---:|---:|"]
    for row in counts.itertuples():
        lines.append(f"| {row.split} | {row.episodes:,} | {row.queries:,} | {row.successes:,} |")
    lines += ["", f"共享成功参考库包含 {fit_audit[0]['reference_queries']:,} 个前缀，来自 "
              f"{fit_audit[0]['reference_episodes']:,} 条训练成功轨迹。"
              "主测试初态与参考库/校准集隔离；五折未见任务同时从参考库、标准化和校准中排除测试任务。", "",
              "在线不传步数、预算、任务/suite 或物理状态。主方法最多涉及最近 11 个 router query；"
              "历史聚合仍可能隐含累计时间。MoE 本身也可能编码任务/阶段，不能声称与时间统计独立。", "",
              "预测目标仍是原始截止条件下的终局成功，不是无限时域或任意指定预算的成功概率，"
              "也不是未来 5 个 chunk 的物理进展或真实 trap 脱困概率。", "",
              "## 主测试", "",
              "Brier、log loss 越低越好；AUROC 越高越好。全部方法在同一 q>=7 测试点比较。", "",
              "| 方法 | Brier | Log Loss | 总体 AUROC | ECE | 同任务同 q AUROC | 再匹配物理阶段 AUROC（95% CI） |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in selected:
        row, stage = main.loc[name], same_stage.loc[name]
        lines.append(f"| {LABELS[name]} | {row.brier:.5f} | {row.log_loss:.5f} | {row.auroc:.4f} | "
                     f"{row.ece10:.4f} | {same_q.loc[name, 'matched_auroc']:.4f} | "
                     f"{stage.matched_auroc:.4f} [{stage.low:.4f}, {stage.high:.4f}] |")
    lines += ["", "概率曲线与指标图：[scalar_calibration.png](scalar_calibration.png)。", "",
              "## 配对损失差", "",
              "差值为左侧方法减右侧方法；负值表示左侧更好。区间在任务内按初态聚类重采样 1,000 次。"
              "它不包含重新训练、重新校准或抽取新任务的方差。", "",
              "| 比较 | Brier 差 | 95% CI |", "|---|---:|---:|"]
    for row in intervals[intervals.metric == "brier"].itertuples():
        lines.append(f"| {row.left} - {row.right} | {row.difference:+.5f} | [{row.low:+.5f}, {row.high:+.5f}] |")
    lines += ["", "## 校准与恢复动态", "",
              "单调 sigmoid 只改变分数刻度，不能修复错误的排序。metrics.csv 同时保存校准前原始分数的 AUROC；"
              "所有已声明方法均报告，没有按测试指标挑选聚合方式。不同未见任务折使用不同映射，"
              "因此跨折合并后的排序不必保持不变。", "",
              "| 共享校准方法 | alpha | beta |", "|---|---:|---:|"]
    for row in coefficients[coefficients.scope == "shared"].itertuples():
        lines.append(f"| {row.model} | {row.alpha:.6f} | {row.beta:.6f} |")
    zeros = coefficients[coefficients.alpha <= 1e-10]
    lines += ["", f"共有 {len(zeros)} / {len(coefficients)} 个拟合的斜率为零。零斜率意味着给定分数方向下最优映射退化为常数。", "",
              "历史最大风险分数不能下降，所以对应成功概率不能回升。局部均值允许回升，"
              "但回升本身不证明机器人已经物理脱困。以下仅统计有下一观测点的相邻概率变化，"
              "没有把成功终止后的未观测点补成 1。", "",
              "| 聚合 | 主测试相邻点 | 概率回升次数 | 回升比例 |", "|---|---:|---:|---:|"]
    for row in dynamics.query("split == 'test_unseen_init' and selection == 'all'").itertuples():
        lines.append(f"| {row.model} | {row.transitions:,} | {row.increases:,} | {row.increase_rate:.2%} |")
    lines += ["", "首次 half-k4 报警后的逐点变化另存 first_alarm_probabilities.csv；"
              "这些仍是既有报警的描述，不是独立确认的真实 trap 样本。", "",
              "## 证据边界", "",
              f"物理阶段匹配覆盖 {int(same_stage.iloc[0].covered_rows):,} / {int(same_stage.iloc[0].all_rows):,} 个主测试点。"
              "阶段仅包含目标谓词、抓取、抬升和接近的离散摘要，不代表完整物理状态。"
              "匹配也改变了可比较样本及正负样本对的权重。"
              "原物理恢复审计发现 290 个采集时仍活动的检查点恢复后已满足全部目标；这里仍以采集器终局标签为准，"
              "所有点沿用同一输入风险集，不按未来物理事件筛选。", "",
              "成功支持度不等于成功后验。成功库中常见的模式也可能出现在失败轨迹；"
              "相似邻居可能来自同一轨迹，且查询密度、尺度及特征冗余均会影响距离。"
              "两参数概率校准不能补回固定分数丢失的信息。", "",
              "这是已探索数据上的追加实验。未新增任务、策略或同检查点分支续跑；"
              "不能将每个前缀的一次 0/1 结果解释成该物理检查点的精确成功概率。", "",
              "## 在线一致性与复现", "",
              f"原始路由抽查覆盖 {raw['source_runs']} 个源 run、{raw['raw_queries']:,} 个 query；"
              f"核对 {raw['ready_queries']:,} 个就绪点的四种聚合，共 {raw['aggregation_probabilities_checked']:,} 个概率。"
              f"最大概率误差 {raw['max_probability_error']:.3g}，最大分数误差 {raw['max_score_error']:.3g}。"
              "训练集抽样只检查批量/流式数学一致性，不报告自身近邻库上的训练成绩。", "",
              f"本机四线程 CPU 单点同步接口耗时中位数 {raw['latency_median_ms']:.2f} ms，"
              f"95 分位 {raw['latency_p95_ms']:.2f} ms，不包含 VLA 推理。近邻参考库有存储与检索成本，"
              "与小型神经置信度头的部署开销不同。", "",
              "```bash", "env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m probability.run_scalar_confidence --threads 4",
              "python -m probability.run_scalar_confidence --render-only", "```", "",
              "在线 bundle 为 models/shared.joblib；每集新建 MoEScalarMonitor，调用 update(hb_router_probs)。"
              "当前已采样 chunk 必须保留，输出不能直接外推到改变策略、延长预算或不同动作采样配置。", "",
              "方法协议：[VLACONF_PROTOCOL.md](../VLACONF_PROTOCOL.md)。"
              "论文原方法：[VLAConf v2](https://arxiv.org/html/2605.29605v2)。", ""]
    (output / "REPORT.zh.md").write_text("\n".join(lines), encoding="utf-8")

    names = ["prior", "support_window4", "support_prefix_mean", "support_prefix_max", "supervised_moe",
             "support_window4_unseen_task", "supervised_moe_unseen_task"]
    short = ["Constant", "Support / local", "Support / mean", "Support / max", "Supervised MoE",
             "Support / held tasks", "Supervised / held tasks"]
    colors = ["#787c80", "#15836d", "#b18c24", "#bf5367", "#397bb3", "#5d8a31", "#8b62a5"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    y = np.arange(len(names))
    axes[0].barh(y, [main.loc[n, "brier"] for n in names], color=colors)
    axes[0].set(yticks=y, yticklabels=short, xlabel="Brier score (lower is better)", title="Final-success probability")
    axes[0].invert_yaxis()
    for i, name in enumerate(names):
        row = same_stage.loc[name]
        axes[1].errorbar(row.matched_auroc, i, xerr=[[max(0, row.matched_auroc-row.low)],
                                                   [max(0, row.high-row.matched_auroc)]],
                         fmt="o", color=colors[i], capsize=3)
    axes[1].axvline(.5, color="#787c80", linestyle="--", linewidth=1)
    axes[1].set(yticks=y, yticklabels=[], xlabel="AUROC, matched task / query / stage", title="Local ranking with 95% intervals")
    axes[1].invert_yaxis()
    axes[2].plot([0, 1], [0, 1], "--", color="#787c80", linewidth=1)
    for name, label, color in [(names[i], short[i], colors[i]) for i in (1, 4, 5)]:
        points = reliability[(reliability.split == "test_unseen_init") & (reliability.model == name)]
        axes[2].plot(points.predicted, points.observed, "o-", label=label, color=color)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted success", ylabel="Observed success", title="Reliability (10 equal-width bins)")
    axes[2].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
    fig.savefig(output / "scalar_calibration.png", dpi=160)
    fig.savefig(output / "scalar_calibration.pdf")
    plt.close(fig)
