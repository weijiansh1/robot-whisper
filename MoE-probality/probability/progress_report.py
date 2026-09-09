"""Render the physical-progress experiment from saved measurements."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = dict(prior="共享常数先验", task_clock="任务 ID＋时钟（离线对照）", moe="纯 MoE，共享模型",
              physical="当前物理阶段（离线对照）", physical_moe="物理阶段＋MoE（离线对照）",
              physical_task_clock="物理阶段＋任务＋时钟（离线对照）",
              physical_task_clock_moe="物理阶段＋任务＋时钟＋MoE（离线对照）",
              prior_unseen_task="未见任务：训练折先验", moe_unseen_task="未见任务：纯 MoE",
              physical_unseen_task="未见任务：物理阶段",
              physical_moe_unseen_task="未见任务：物理阶段＋MoE")


def report(output):
    scores = pd.read_csv(output / "metrics.csv")
    main = scores[(scores.split == "test_unseen_init") & (scores.suite == "all")].set_index("model")
    counts = pd.read_csv(output / "label_counts.csv").set_index("split")
    total, test = counts.loc["all"], counts.loc["test_unseen_init"]
    matched = pd.read_csv(output / "matched_auroc.csv")
    strict = matched[matched.selection == "task_query_stage"].set_index("model")
    intervals = pd.read_csv(output / "cluster_intervals.csv")
    physical = json.loads((output / "physical_audit.json").read_text())
    alarm = json.loads((output / "alarm_summary.json").read_text())
    raw = json.loads((output / "raw_verification.json").read_text())
    lines = ["# 固定 5 个 chunk 的物理进展实验", "",
             "本实验将目标改为未来 50 个动作内是否出现可观测物理里程碑。所有任务共用同一预测窗口；"
             "在线只读最近 8 次推理的 MoE，使用一套共享参数。没有预算、绝对时间、任务 ID、"
             "机器人状态或物体状态输入。物理信息只用于离线标签、分层评价和明确标注的对照模型。", "",
             "## 主结果", "",
             f"原始物理提取覆盖 32,000 条自然 rollout、{int(total.raw_queries):,} 个检查点、40 个任务。"
             f"满足观察条件的起点共 {int(total.eligible_queries):,} 个；主测试有 "
             f"{int(test.eligible_episodes):,} 条轨迹、{int(test.eligible_queries):,} 个起点，"
             f"其中 {test.positive_rate:.1%} 在未来 5 个 chunk 内出现标签所定义的进展。", "",
             "| 模型 | Brier | Log loss | AUROC | 同任务、同 q、同粗阶段 AUROC（95% 区间） |",
             "|---|---:|---:|---:|---|" ]
    for name in LABELS:
        row, within = main.loc[name], strict.loc[name]
        lines.append(f"| {LABELS[name]} | {row.brier:.5f} | {row.log_loss:.5f} | {row.auroc:.4f} | "
                     f"{within.matched_auroc:.4f} [{within.low:.4f}, {within.high:.4f}] |")
    signal = strict.loc["moe_unseen_task"]
    per_task = pd.read_csv(output / "per_task_task_query_stage.csv")
    supported_tasks = per_task[(per_task.model == "moe_unseen_task") & (per_task.pairs > 0)]
    lines.extend(["", "完整任务留出的纯 MoE 模型，训练和校准均未见测试任务；同任务、同 q、同粗阶段的"
                  f"AUROC 为 {signal.matched_auroc:.4f}，区间 [{signal.low:.4f}, {signal.high:.4f}]。"
                  + ("在本次固定模型、固定任务的聚类重采样下，区间下界高于 0.5。"
                     if signal.low > 0.5 else "区间包含或低于 0.5，不能认为已稳定区分同阶段的未来进展。"),
                  "这里的 AUROC 与旧终局成功目标不同，不宜直接把数值相减作为改进量。",
                  f"严格分层下有 {len(supported_tasks)}/40 个任务包含可比较的正负样本，"
                  f"其中 {int((supported_tasks.matched_auroc > 0.5).sum())} 个的 AUROC 点估计高于 0.5，"
                  f"任务等权平均为 {supported_tasks.matched_auroc.mean():.4f}。"
                  "不能将总体区间解释为每个任务都有效。", "",
                  "## 概率校准仍有限制", "",
                  f"共享纯 MoE 模型的 ECE（10 个等宽概率区间）为 {main.loc['moe', 'ece10']:.4f}；"
                  f"完整任务留出后为 {main.loc['moe_unseen_task', 'ece10']:.4f}。"
                  "跨任务时，辨别能力高于随机并不代表绝对概率已经可信。",
                  f"未见任务的原始概率 Brier/ECE 为 {main.loc['moe_unseen_task_raw', 'brier']:.5f}/"
                  f"{main.loc['moe_unseen_task_raw', 'ece10']:.4f}，固定 sigmoid 校准后为 "
                  f"{main.loc['moe_unseen_task', 'brier']:.5f}/{main.loc['moe_unseen_task', 'ece10']:.4f}，"
                  "本次校准在跨任务分布上反而变差。保留全部原始和校准结果，没有按测试表现更换输出头。", "",
                  "## 是否超出物理阶段本身", "",
                  "下表的差值是左侧损失减右侧损失；负值表示加入 MoE 后更低。"
                  "模型算法与容量设置固定，但信息组合改变了拟合问题，不是因果干预证据。", "",
                  "| 左侧 | 右侧 | Brier 差 | 95% 区间 |", "|---|---|---:|---|"])
    for row in intervals[intervals.metric == "brier"].itertuples():
        lines.append(f"| {LABELS[row.left]} | {LABELS[row.right]} | {row.difference:+.5f} | "
                     f"[{row.low:+.5f}, {row.high:+.5f}] |")
    delta = intervals[(intervals.left == "physical_moe_unseen_task") & (intervals.metric == "brier")].iloc[0]
    lines.extend(["", "未见任务中，在当前物理阶段对照上加入 MoE 的 Brier 差为 "
                  f"{delta.difference:+.5f}，区间 [{delta.low:+.5f}, {delta.high:+.5f}]。"
                  + ("区间完全低于零，支持 MoE 在该物理摘要之外仍有预测信息。" if delta.high < 0
                     else "区间未完全低于零，不能断言它稳定优于该物理对照。"), "",
                  "## 物理标签", "",
                  "未来 5 个 chunk 内满足任一条件即为正例：采集器记录任务成功；连续两个未来检查点的"
                  "原始 BDDL 已满足目标数都多于当前；或尚未完成的目标物体新出现双指抓持，且比该物体"
                  "首个检查点高至少 2.5 厘米，在两个未来检查点均成立，同时没有减少已满足目标数。"
                  "两个确认点都必须在 q 之后、q+5 以内。单次短暂接触不计为确认事件。", "",
                  f"主测试中，未来窗口内成功为 {int(test.success_h5):,} 个起点，"
                  f"目标数增加为 {int(test.goal_advance_h5):,} 个，新抓持并抬升为 {int(test.grasp_lift_h5):,} 个。"
                  "这些分量可以重叠。"
                  f"正例中有 {int(test.progress_without_success_h5):,} 个并未在 5 个 chunk 内成功，"
                  "因此标签包含中间里程碑；同时应注意完成任务可能占正例的较大部分。", "",
                  "这是保守的物理进展代理：接近物体、尚未达到阈值的抽屉运动、两个检查点之间的短事件"
                  "可能被遗漏。高度以每集首个检查点为基准，也会漏掉某些从已降低位置重新抬起的情况。"
                  "两次接触观测不保证中间持续接触；没有里程碑不能自动等同于受困。", "",
                  "## 观察范围与对齐", "",
                  "仅纳入 q>=7 且 (q+5)*10 严格小于原定执行上限的起点。此规则在当前即可确定；"
                  "提前成功仍作为正例保留。严格小于的原因是原日志未保存预算耗尽后的终态，"
                  "恰好触及上限的完整物理窗口无法核实，不能补写为无进展。", "",
                  f"全语料有 {physical['restored_completed_checkpoints']:,} 个采集器尚未停止、"
                  "但恢复状态并重新前向计算后已满足完整目标的检查点。记录这项差异并排除这些起点，"
                  "保留更早前缀，终局时间仍来自原采集器。不能声称恢复后的目标判定与采集时每个子步严格相同。",
                  f"逐状态检查原子谓词与恢复环境的 check_success 一致；所有任务关节布局匹配。"
                  f"恢复机器人读数与日志的最大末端位置误差为 "
                  f"{max(r['max_eef_position_error_m'] for r in physical['runs']):.6f} 米，"
                  f"最大姿态角差为 {max(r['max_eef_rotation_error_rad'] for r in physical['runs']):.6f} 弧度。"
                  f"超过预设位置/姿态/夹爪诊断阈值的检查点数为 "
                  f"{sum(r['robot_alignment_outliers'] for r in physical['runs'])}。完整差异保存在物理审计中。", "",
                  "固定预测长度并未消除采集上限对语料分布的影响：晚期存活轨迹和不同任务的阶段分布仍不同。"
                  "模型只在上述观察范围接受检验；没有识别原策略在原上限以后继续执行的反事实概率。", "",
                  "## 对照与统计", "",
                  "共享模型沿用原初始状态分组：训练、校准、主测试无同任务初态交叉；"
                  "另保存新噪声但见过初态的次测试结果。40 个任务采用原先冻结的 5 折，每折留出 8 个任务。"
                  "学习器和校准设置固定，未按新测试结果调参；原始与校准概率都保存在结果中。", "",
                  "粗阶段由当前目标位掩码、目标物体抓持/抬升/接近位掩码组成；接近阈值为 12 厘米。"
                  "更严格的 AUROC 仅比较同任务、同 q、同粗阶段的正负样本，再按样本对数加权。"
                  f"其有效分层覆盖 {int(signal.covered_rows):,}/{int(signal.all_rows):,} 个主测试起点，"
                  f"共有 {int(signal.pairs):,} 对正负比较。相同粗阶段不等于相同物理状态。", "",
                  "Brier/log loss 差和匹配 AUROC 的 95% 区间均在每个任务内按初始状态聚类重采样 "
                  "1,000 次，保留同初态的所有种子与多个前缀。区间条件于已有模型与这 40 个任务，"
                  "不包含重新训练或采样新任务的不确定性。任务级细分和各分层支持数随结果保存。", "",
                  "## 既有报警的描述性检查", "",
                  "half-k4 规则及其报警病例已被旧分析查看，本节不是新的盲测。每条轨迹只取首次报警，"
                  "所有报警均保留；窗口不可观测时不赋予可靠的正负标签。", "",
                  "| 最终结果 | 首次报警数 | 有效观察窗口 | 窗口内物理进展 | 窗口内成功 |",
                  "|---|---:|---:|---:|---:|"])
    for row in alarm:
        lines.append(f"| {'成功' if row['final_success'] else '失败'} | {row['alarms']} | {row['supported']} | "
                     f"{row['progress_h5']} | {row['success_h5']} |")
    lines.extend(["", "报警后的物理里程碑说明后续有可观测推进；它不能证明报警当时是真 trap。"
                  "要估计真实受困后的自然脱困概率，仍需独立的受困判据、脱困标注或对齐检查点的分支续跑。", "",
                  "## 复核与使用", "",
                  "本次全量重新读取的是 32,000 条轨迹的模拟器状态；MoE 复用此前校验的全量因果特征缓存。"
                  f"另在 80 条原始路由样本的 {raw['raw_queries']:,} 个 chunk 上重放新在线接口，"
                  f"对比 {raw['probabilities_checked']} 个有效预测，最大概率差 {raw['max_error']:.3g}。"
                  "没有采样或执行新机器人动作。", "",
                  "运行 `bash extract_progress.sh --workers 4` 提取物理状态，随后运行 "
                  "`python -m probability.run_progress --threads 4`。仅重绘报告用 `--render-only`。"
                  "协议见 `../PROGRESS_PROTOCOL.md`；在线入口为 `probability.progress_model.MoEProgressMonitor`，"
                  "模型为 `models/moe_progress_shared.joblib`。每条新 rollout 新建 monitor；前 7 次推理不输出概率。",
                  "本实验是已有语料上的追加研究。它验证的是有限观察范围内的物理里程碑预测，"
                  "不是‘模型知道自己受困’的证明，也不直接说明干预能带来多少增益。", "",
                  "![固定窗口进展评估](progress_audit.png)"])
    (output / "REPORT.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    names = ["task_clock", "moe", "physical", "physical_moe"]
    short = ["task + clock", "MoE only", "physics", "physics + MoE"]
    colors = ["#ba5b43", "#087e8b", "#777777", "#608d45"]
    axes[0].bar(range(4), main.loc[names, "brier"], color=colors)
    axes[0].set(xticks=range(4), xticklabels=short, ylabel="Brier score", title="Five-chunk physical milestone")
    axes[0].tick_params(axis="x", labelrotation=20)
    names = ["moe", "moe_unseen_task"]
    short = ["shared MoE", "unseen-task MoE"]
    for i, selection in enumerate(("task_query", "task_query_stage")):
        rows = matched[matched.selection == selection].set_index("model").loc[names]
        positions = np.arange(2) + (i-0.5)*0.32
        errors = np.maximum(0, np.stack([rows.matched_auroc-rows.low, rows.high-rows.matched_auroc]))
        axes[1].bar(positions, rows.matched_auroc, width=0.30, color=["#087e8b", "#608d45"][i],
                    yerr=errors, capsize=3, label=["task + query", "+ physical stage"][i])
    axes[1].axhline(0.5, linestyle="--", color="#666666")
    axes[1].set(xticks=range(2), xticklabels=short, ylim=(0.4, 1), ylabel="Matched AUROC", title="Same-time, same-stage discrimination")
    axes[1].legend(fontsize=8)
    calibration = pd.read_csv(output / "calibration.csv")
    axes[2].plot([0, 1], [0, 1], "--", color="#888888")
    for name, label, color in zip(["task_clock", "moe", "moe_unseen_task"],
                                  ["task + clock", "MoE only", "unseen-task MoE"], colors[:3], strict=True):
        block = calibration[(calibration.split == "test_unseen_init") & (calibration.model == name)]
        axes[2].plot(block.predicted, block.observed, "o-", label=label, color=color)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted progress", ylabel="Observed progress", title="Calibration")
    axes[2].legend(fontsize=8)
    fig.savefig(output / "progress_audit.png", dpi=180)
    fig.savefig(output / "progress_audit.pdf")
    plt.close(fig)
