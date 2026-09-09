"""Generate figures and a Chinese account of the completed v8.2 validation."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from monitor import HERE
from run_analysis import digest, write_json

OUT = HERE.parent / "results/v82_validation_20260908"
SUITES = ("libero_goal", "libero_long", "libero_object", "libero_spatial")
CAPS = dict(libero_goal=300, libero_long=520, libero_object=280, libero_spatial=220)
COLORS = dict(v7="#757575", v82="#147e72", corrected="#bb4c43", clock="#977529", eef_motion_low="#386ea6")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 170})


def save(fig, name):
    fig.savefig(OUT / (name + ".png"), facecolor="white")
    fig.savefig(OUT / (name + ".pdf"), facecolor="white")
    plt.close(fig)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def percent(x):
    return "%.2f%%" % (100 * x)


def interval(row, name="estimate"):
    return "%.3f [%.3f, %.3f]" % (row[name], row["lo"], row["hi"])


def cumulative_plot(curves):
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 6.3))
    settings = (("v7_frozen", "v7 frozen", COLORS["v7"], False),
                ("v82_frozen", "v8.2 frozen", COLORS["v82"], False),
                ("v82_padding_corrected", "v8.2 padding corrected", COLORS["corrected"], False),
                ("v82", "v8.2 success-budget 1%", COLORS["eef_motion_low"], True),
                ("clock", "Clock success-budget 1%", COLORS["clock"], True))
    for col, suite in enumerate(SUITES):
        for name, label, color, budget in settings:
            part = curves.loc[curves.scope.eq(suite) & curves.method.eq(name)]
            part = part.loc[part.calibration.eq("task_init") & (part.alpha == .01)] if budget else part.loc[part.family.eq("frozen")]
            part = part.sort_values("query")
            for row, metric in enumerate(("recall", "fpr")):
                ax = axes[row, col]
                ax.step(part.actions, part[metric], where="post", label=label, color=color, lw=1.6)
                ax.set_xlim(0, CAPS[suite])
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                ax.grid(axis="y", alpha=.2)
                if row == 0:
                    ax.set_ylim(0, 1.02)
                    ax.set_title(suite.removeprefix("libero_").capitalize())
                else:
                    ax.set_xlabel("Executed actions")
        axes[0, col].set_ylabel("Cumulative failure recall" if col == 0 else "")
        axes[1, col].set_ylabel("Cumulative success FPR" if col == 0 else "")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("B cohort: fixed thresholds over the entire execution, 16,000 episodes")
    fig.tight_layout(rect=(0, .09, 1, .95))
    save(fig, "cumulative_validation")


def conditional_plot(curves):
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 6.4))
    for col, suite in enumerate(SUITES):
        for row, level in enumerate(("task", "task_init")):
            ax = axes[row, col]
            for method, label in (("v7", "v7"), ("v82", "v8.2"), ("eef_motion_low", "Low EEF motion")):
                part = curves.loc[curves.scope.eq(suite) & curves.level.eq(level) & curves.method.eq(method)].sort_values("query")
                ax.plot(part["query"] * 10, part.auc, label=label, color=COLORS[method], lw=1.7)
                if method == "v82":
                    ax.fill_between(part["query"] * 10, part.lo, part.hi, color=COLORS[method], alpha=.13)
            ax.axhline(.5, color="#333333", ls="--", lw=.8, label="Chance")
            ax.set_xlim(70, CAPS[suite])
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=.2)
            if row == 0:
                ax.set_title(suite.removeprefix("libero_").capitalize())
            else:
                ax.set_xlabel("Executed actions")
            if col == 0:
                ax.set_ylabel("Same task/time AUROC" if row == 0 else "Same task/init/time AUROC")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Cross-fitted continuous scores: only contemporaneous success/failure comparisons")
    fig.tight_layout(rect=(0, .06, 1, .95))
    save(fig, "conditional_validation")


def ablation_plot(curves, metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    methods = (("v7", "v7", "#777777"), ("v82", "v8.2", "#147e72"),
               ("v82_no_smoothing", "No new-head smoothing", "#bb4c43"),
               ("v82_no_confirmation", "No new-head confirmation", "#805ca0"),
               ("v82_no_curvature_baseline", "No curvature self-baseline", "#386ea6"),
               ("clock", "Clock", "#977529"))
    for ax, data, title in ((axes[0], metrics, "Entire execution"),
                             (axes[1], curves.loc[curves["query"] == 13], "Cumulative through q13 / 130 actions")):
        for method, label, color in methods:
            part = data.loc[data.family.eq("crossfit") & data.scope.eq("all") & data.calibration.eq("task_init") & data.method.eq(method)].sort_values("alpha")
            ax.plot(part.fpr, part.recall, marker="o", ms=4, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel("Observed success-episode FPR")
        ax.set_ylabel("Failure recall")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Calibration budgets 0.5%, 1%, 2%, 5%; observed B FPR is not forced to match")
    fig.tight_layout(rect=(0, .12, 1, .93))
    save(fig, "ablation_validation")


def event_plot(curves):
    fig = plt.figure(figsize=(12.5, 8.3))
    grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, .65))
    heads = (("freeze_raw", "Relative routing freeze"), ("frontback_raw", "Negative log front/back path"),
             ("curvature_relative_raw", "Log relative speed curvature"), ("freeze_smooth", "Freeze: W6"),
             ("frontback_smooth", "Front/back: W6"), ("curvature_smooth", "Curvature: W6"))
    data = curves.loc[curves.scope.eq("all") & curves.kind.eq("release_outside_goal") & curves.coverage.eq("available")]
    for i, (head, title) in enumerate(heads):
        ax = fig.add_subplot(grid[i // 3, i % 3])
        for failure, label, color in ((False, "Eventually successful", "#147e72"), (True, "Eventually failed", "#bb4c43")):
            part = data.loc[data["head"].eq(head) & data.failure.eq(failure)].sort_values("offset")
            ax.plot(part.offset * 10, part["median"], color=color, label=label)
            ax.fill_between(part.offset * 10, part.p25, part.p75, color=color, alpha=.15)
        ax.axvline(0, color="#555555", ls="--", lw=.8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    count = fig.add_subplot(grid[2, :])
    for failure, label, color in ((False, "Eventually successful", "#147e72"), (True, "Eventually failed", "#bb4c43")):
        part = data.loc[data["head"].eq("freeze_raw") & data.failure.eq(failure)].sort_values("offset")
        count.plot(part.offset * 10, part.observed, color=color, label=label)
    count.axvline(0, color="#555555", ls="--", lw=.8)
    count.set_xlabel("Actions relative to release before target goals are satisfied")
    count.set_ylabel("Observed episodes")
    count.legend(loc="upper right", frameon=False)
    count.grid(axis="y", alpha=.2)
    fig.suptitle("Outcome-neutral release events: medians/IQR with changing observation coverage")
    fig.tight_layout(rect=(0, 0, 1, .96))
    save(fig, "physical_event_validation")


def main():
    metrics = pd.read_csv(OUT / "alarm_metrics.csv")
    curves = pd.read_csv(OUT / "cumulative_curves.csv")
    early = pd.read_csv(OUT / "early_conditional_summary.csv")
    conditional = pd.read_csv(OUT / "conditional_auc_curves.csv")
    events = pd.read_csv(OUT / "event_timing.csv")
    counts = pd.read_csv(OUT / "physical_event_counts.csv")
    matched = pd.read_csv(OUT / "motion_matched_summary.csv")
    motion_audit = json.loads((OUT / "motion_match_verification.json").read_text())
    verified = json.loads((OUT / "independent_verification.json").read_text())
    physical_audit = json.loads((OUT / "event_verification.json").read_text())
    frozen_audit = json.loads((OUT / "frozen_verification.json").read_text())
    cumulative_plot(curves)
    conditional_plot(conditional)
    ablation_plot(curves, metrics)
    event_plot(pd.read_csv(OUT / "event_aligned_curves.csv"))
    frozen = metrics.loc[metrics.family.eq("frozen") & metrics.scope.eq("all")]
    frozen_rows = [[r.method, r.tp, r.fp, percent(r.recall), percent(r.fpr)] for r in frozen.itertuples()]
    budget = metrics.loc[metrics.family.eq("crossfit") & metrics.scope.eq("all") & metrics.calibration.eq("task_init") & (metrics.alpha == .01)]
    budget_rows = [[r.method, r.tp, r.fp, percent(r.recall), percent(r.fpr)] for r in budget.itertuples()]
    early_main = early.loc[early.level.eq("task") & early.scope.eq("all") & early.comparison.eq("auc")]
    early_rows = [[r.method, interval(r._asdict()), r.tasks] for r in early_main.itertuples() if r.method in
                  ("v7", "v82", "eef_motion_low", "clock", "v7_frozen_binary", "v82_frozen_binary", "v82_padding_corrected_binary")]
    suites = early.loc[early.level.eq("task") & ~early.scope.eq("all") & early.comparison.eq("auc")]
    suite_rows = []
    for suite in SUITES:
        part = suites.loc[suites.scope.eq(suite)].set_index("method")
        suite_rows.append([suite, *["%.3f" % part.loc[m, "estimate"] for m in ("v7", "v82", "eef_motion_low")]])
    motion_rows = [[r.method, interval(r._asdict(), "win_rate"), r.tasks] for r in matched.loc[matched.scope.eq("all")].itertuples()]
    event_rows = [[r.kind, "失败" if r.failure else "成功", r.episodes] for r in counts.itertuples()]
    timing = events.loc[events.scope.eq("all") & events.kind.eq("release_outside_goal") &
                        events.method.isin(("v82_frozen", "v82_padding_corrected", "v82_budget01"))]
    timing_rows = [[r.method, "失败" if r.failure else "成功", r.episodes, r.before, r.same, r.after, r.missed,
                    "%.1f" % r.median_post_event_delay_q] for r in timing.itertuples()]
    checkpoint_rows = []
    chosen = curves.loc[curves.scope.eq("all") & curves["query"].isin([7, 10, 13, 20])]
    for method in ("v82_frozen", "v82_padding_corrected", "v82", "clock"):
        part = chosen.loc[chosen.method.eq(method)]
        if method in ("v82", "clock"):
            part = part.loc[part.calibration.eq("task_init") & (part.alpha == .01)]
        for r in part.itertuples():
            checkpoint_rows.append([method, r.query, r.tp, r.fp, r.active_success])
    report = f"""# v8.2 现有 HUB 数据补充验证

日期：2026-09-08。完成时间控制、完整成功轨迹的累计误报、分支消融、成功物理事件对照。
结果支持路由中存在有限的成败关联，但不支持把当前工作点解释成强的物理事件前预测器。
没有训练模型、修改原始数据或生成新 rollout；本轮属于历史数据上的回顾性验证。

## 1. 覆盖与校验

- A/B 共 32,000 条，40 任务，508,023 个 query；逐 run 核对 HUB summaries 和原始物理结果标签。
- 主测试 B：16,000 条，564 失败、15,436 成功。旧共同 B 是 15,600 条，少一个 Object 任务。
  因而旧报告冻结 v8.2 的 98 FP 在本轮完整 B 上为 99 FP；旧共同集合逐位重现。
- 五折按初态隔离：每折 A 参考 9,600、A 校准 3,200、B 测试 3,200；全部 B 恰好测试一次。
- 独立重算 {verified['independent_alarm_checks']:,} 个报警判断，全部相等；从 HUB 原始 Zarr
  重放 {verified['raw_replayed_episodes']} 条、{verified['raw_replayed_queries']} 个 query，覆盖全部 80 个 run。
- B 的全部保存物理状态已恢复；{physical_audit['original_label_anchors']} 个既有目标物体释放标签逐项一致。
- 95% 区间按 task 簇、suite 内重采样；query 不是独立样本，区间以已拟合 profile 为条件。

## 2. 补齐修正改变了工作点

原校准把全 NaN 补齐区变为零。修正版只排除这些无效 query，保留原 14,800 条 A、
原 partial warm-up 均值、W6、K2、基线、时间斜率以及 v7 profile，没有用 B 重新选参数。
前后比对数低阈值由 0 变为 {-frozen_audit['padding_corrected_thresholds']['frontback']:.8f}；
相对曲率高阈值由 0.38900006 变为 {frozen_audit['padding_corrected_thresholds']['curvature']:.8f}。

{table(['冻结/修正版', 'TP', 'FP', '召回', 'FPR'], frozen_rows)}

修正后召回和误报同时上升。前后比阈值恰好为零不能作为天然机制临界点。
原冻结版仍作为历史工作点保留，其低误报数字不等于校准正确或存在误报概率保证。

## 3. 检查点只统计累计结果

预算校准版本用完整成功轨迹的全程峰值校准一次；原冻结与补齐修正版使用第二节的固定规则。
所有版本都统计截至各 q 是否曾报警，不在检查点重新调整阈值。结束前的误报保留，
已结束成功轨迹仍在总分母中；下面“仍运行成功”只说明暴露情况，不替换 FPR 分母。
完整 q0..q51 见 [cumulative_curves.csv](cumulative_curves.csv)。

{table(['方法', 'q', '累计 TP', '累计 FP', '仍运行成功'], checkpoint_rows)}

`v82` 和 `clock` 行属于五折标准化、task/init 1% 成功预算版本。

![累计检出与误报](cumulative_validation.png)

## 4. 同任务、同时间的区分能力

q7..q13 预先固定；仅在同任务、同 q 且有成功与失败仍在运行的分层内比较。
先对 query、再对 task 平均。36 个任务有两类对照，其他任务不可估计，不记为 0.5 或 1。

{table(['分数/报警', '平均 AUROC [95% 区间]', '任务数'], early_rows)}

`v7`/`v82` 是五折参考 median/MAD 标准化后的连续 guard，不能把其 AUROC 写成原冻结
布尔规则的 AUROC。`*_binary` 是原冻结工作点在 q 前是否已经报警的 0/1 值。
连续分数显示部分排序信息，但原冻结工作点早期检出很少；二者回答不同问题。

{table(['suite', 'v7 连续', 'v8.2 连续', '末端低运动'], suite_rows)}

新增头相对 v7 有早期排序增量；总体上低运动量排序更强。分套件差异明显。
同 task/init/q 的跨噪声结果和完整支持数见 [early_conditional_summary.csv](early_conditional_summary.csv)
及 [conditional_auc_strata.csv](conditional_auc_strata.csv)。后期成功样本减少后，曲线不应外推为普遍辨别能力。

![条件 AUROC](conditional_validation.png)

### 追加探索：进一步匹配运动量

该实验是在看到低运动基线较强后追加，见 [补充协议](../../v82_validation/MOTION_MATCH_ADDENDUM_ZH.md)。
同 task/q 内，以低运动量百分位差不超过 0.02 匹配成功对照。{motion_audit['eligible_case_queries']} 个
可比较失败 query 中匹配 {motion_audit['matched_case_queries']} 个，涉及
{motion_audit['unique_case_episodes']} 条失败、{motion_audit['unique_control_episodes']} 条唯一成功对照。
对照可重复使用，统计区间按 task 成簇。百分位差中位数 {motion_audit['median_percentile_gap']:.4f}。

{table(['方法', '正例分数较高的配对胜率 [95% 区间]', '任务数'], motion_rows)}

v8.2 的配对胜率提供“在近似相同运动量下仍有部分区分信息”的探索性证据。
这不是全体轨迹 AUROC，也不是完全消除运动混淆或证明机制因果性的结果。

## 5. 相同成功校准预算的消融

以下统一 task/init 名义 1% 预算；阈值来自独立 A 校准成功组的全程峰值。
这是预算校准的标准化分数族，区别于第二节原冻结多阈值规则。

{table(['消融版本', 'TP', 'FP', '召回', '实测 FPR'], budget_rows)}

平滑与曲率自基线在这个工作点提供增量。去掉连续确认带来少量额外检出与误报；
不能直接宣称 K2 一定更优。去掉时间斜率影响较小，说明新增收益不全部来自斜率。
时钟可用较晚报警达到相同全程 TP 数，但 q13 尚无检出，而标准化 v8.2 已有 16 TP / 1 FP。
应联合观察工作点、时机与套件，不能仅比较最终 TP。

名义预算不等于实际 B FPR，特别是 task/init 最大值校准较保守。
0.5%、1%、2%、5% 和 episode 校准完整结果均保留；没有根据测试集把实际 FPR 强行调齐。

![消融的实际误报与检出](ablation_validation.png)

## 6. 成功物理事件对照

同一套目标谓词、抓取和高度判据应用于全部 B 成功/失败轨迹。
每 episode 每类取第一次事件；不同事件类有重叠，不能相加为独立失败数。

{table(['事件', '最终结果', 'episode 数'], event_rows)}

“目标尚未满足时释放”可以是正常放置过程的一部分；“释放后高度下降”也不等于不可恢复失败。
新规则基于释放当时的目标状态，区别于旧报告依据失败原因筛出的 216 条脱手子集。

{table(['方法', '最终结果', '事件数', '事件前', '同 q', '事件后', '未报警', '仅事件后延迟中位 q'], timing_rows)}

原冻结 v8.2 在 1,594 条最终成功、出现目标尚未满足时释放的轨迹上有 42 次误报，占 2.63%；
在 238 条最终失败的对应轨迹上有 200 次检出，其中 192 次发生在事件后。
这支持“持续异常的事后检出”，同时说明正常事件对照不可缺少。

![事件对齐的路由变化与可观测数量](physical_event_validation.png)

曲线按独立物理事件对齐，保留各 offset 的真实观测数；终止后的值不补齐。
另提供完整 [-3,+4] query 窗口的敏感性结果，避免把变动的样本构成误当成恢复。
事件时刻的同任务/同 q 成功匹配、同初态优先以及缺乏成功对照的情况见
[event_control_pairs.csv](event_control_pairs.csv) 和 [event_matched_comparison.csv](event_matched_comparison.csv)。

成功状态恢复中 {physical_audit['successes_with_full_goal_at_restored_checkpoint']} 条出现保存检查点已满足全部目标，
多数在最后一个检查点；这些事实保留在 [明细](restored_goal_satisfied_successes.csv)。
BDDL 谓词和环境 API 一致，采集时的在线结果标签保持不变。
状态精度为 float32、间隔 10 动作，区间内事件和恢复后接触细节存在分辨率限制。

## 7. 结论边界

本轮支持的表述：路由动力学包含有限、任务相关的成败信息；持续规则能够在部分物理事件之后
形成低误报告警，但阈值校准、观察时长与简单运动量都显著影响结论。
追加运动匹配提供了互补信息的探索性证据，仍需在新的封存批次上确认。

本轮没有进行新的受控扰动或报警触发恢复实验。恢复保存状态用于补充标注，
不能替代失败机制的因果实验或闭环成功率评价；现有 A/B 也不构成新盲测。

## 8. 复现

在 `safe&vlaconf` 目录运行，下列默认输出已存在，重新完整计算时需使用新的输出目录：

```bash
python -m pytest -q moe_trainfree/v82_validation/test_validation.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python moe_trainfree/v82_validation/run_analysis.py --output /tmp/v82-validation-replay
bash moe_trainfree/v82_validation/run_physical.sh --output /tmp/v82-physics-replay --workers 4
python moe_trainfree/v82_validation/analyze_events.py --analysis /tmp/v82-validation-replay --physics /tmp/v82-physics-replay
python moe_trainfree/v82_validation/verify.py --output /tmp/v82-validation-replay
```

当前标准产物的追加运动匹配和图文生成：

```bash
python moe_trainfree/v82_validation/motion_controls.py
python moe_trainfree/v82_validation/report.py
```

分折、原始回放、阈值和报警的完整核验见 [independent_verification.json](independent_verification.json)。
物理状态恢复保存在 [v82_physical_controls_20260908](../v82_physical_controls_20260908/verification.json)。
首次触发归因见 [head_attribution.csv](head_attribution.csv)。
"""
    (OUT / "REPORT_ZH.md").write_text(report)
    write_json(OUT / "report_verification.json", dict(code_sha256=digest(__file__),
                generated={p.name: digest(p) for p in OUT.iterdir() if p.suffix in (".png", ".pdf") or p.name == "REPORT_ZH.md"},
                data_files={name: digest(OUT / name) for name in ("alarm_metrics.csv", "early_conditional_summary.csv", "event_timing.csv", "motion_matched_summary.csv")}))
    print("REPORT and four PNG/PDF figures generated", flush=True)


if __name__ == "__main__":
    main()
