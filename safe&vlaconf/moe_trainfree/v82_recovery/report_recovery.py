"""Report both improvements and regressions of the fixed recovery experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from recovery import HERE, METHODS
from run_experiment import BASE, PRIMARY_ALPHA, PRIMARY_KIND
from run_analysis import archive, bootstrap_indices, digest, write_json

COLORS = {"v82_reference": "#666666", "selected": "#147e72", "recovery": "#bb4c43",
          "reference_confirm3": "#386ea6", "clock": "#977529"}
LABELS = {"v82_reference": "Reference v8.2", "selected": "A-selected rule", "recovery": "Absolute + recent",
          "reference_confirm3": "Reference + K3", "clock": "Clock"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 170})


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def percent(value):
    return "%.2f%%" % (100 * value)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / (name + "." + extension), facecolor="white")
    plt.close(fig)


def plots(metrics, curves, early, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
    for ax, kind in zip(axes, ("task_init", "episode")):
        for name, color in COLORS.items():
            part = metrics.loc[metrics.scope.eq("all") & metrics.calibration.eq(kind) & metrics.method.eq(name)].sort_values("alpha")
            ax.plot(part.fpr, part.recall, color=color, marker="o", ms=4, label=LABELS[name])
        ax.set_title("Task / initial-state calibration" if kind == "task_init" else "Episode calibration")
        ax.set_xlabel("Observed success-episode FPR")
        ax.set_ylabel("Failure recall")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("B: all prespecified budgets; thresholds and selection use A only")
    fig.tight_layout(rect=(0, .12, 1, .93))
    save(fig, output, "operating_points")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, color in COLORS.items():
        part = curves.loc[curves.scope.eq("all") & curves.calibration.eq("task_init")
                          & curves.alpha.eq(.01) & curves.method.eq(name)].sort_values("query")
        for ax, metric in zip(axes, ("recall", "fpr")):
            ax.step(part["query"] * 10, part[metric], where="post", color=color, label=LABELS[name])
            ax.set_xlabel("Executed actions")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
    axes[0].set_ylabel("Cumulative failure recall")
    axes[1].set_ylabel("Cumulative success FPR")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Primary task/init 1% budget: clearances do not erase earlier alarms")
    fig.tight_layout(rect=(0, .12, 1, .93))
    save(fig, output, "cumulative_results")
    part = early.loc[early.scope.eq("all")].set_index("method").loc[[*METHODS, "selected"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    y = np.arange(len(part))
    ax.errorbar(part.auc, y, xerr=np.asarray([part.auc - part.lo, part.hi - part.auc]), fmt="o",
                color="#147e72", ecolor="#8ea69d", capsize=3)
    ax.set_yticks(y, [LABELS.get(name, name) for name in part.index])
    ax.invert_yaxis()
    ax.axvline(.5, color="#777777", ls="--", lw=1)
    ax.set_xlim(.45, .75)
    ax.set_xlabel("Same-task / same-query AUROC, q7..q13")
    ax.set_title("Continuous scores: task-cluster 95% intervals")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    save(fig, output, "early_ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "v82_recovery_20260908")
    args = parser.parse_args()
    output = args.output
    metrics = pd.read_csv(output / "alarm_metrics.csv")
    curves = pd.read_csv(output / "cumulative_curves.csv")
    early = pd.read_csv(output / "early_auc_summary.csv")
    pairs = pd.read_csv(output / "paired_comparisons.csv")
    grid = pd.read_csv(output / "selection_grid.csv", dtype={"fold": str})
    states = pd.read_csv(output / "state_episodes.csv")
    verification = json.loads((output / "independent_verification.json").read_text())
    primary = metrics.loc[metrics.scope.eq("all") & metrics.calibration.eq("task_init") & metrics.alpha.eq(.01)]
    base = primary.set_index("method").loc["v82_reference"]
    chosen = primary.set_index("method").loc["selected"]
    pair = pairs.loc[pairs.scope.eq("all") & pairs.calibration.eq("task_init") & pairs.alpha.eq(.01) & pairs.method.eq("selected")].iloc[0]
    outcome = "改善" if chosen.tp >= base.tp and chosen.fp <= base.fp and (chosen.tp > base.tp or chosen.fp < base.fp) else "没有改善"
    plots(metrics, curves, early, output)
    rows = [[r.method, r.tp, r.fp, percent(r.recall), percent(r.fpr)] for r in primary.itertuples()]
    budgets = metrics.loc[metrics.scope.eq("all") & metrics.method.isin(["v82_reference", "selected"])]
    budget_rows = []
    for (kind, alpha), part in budgets.groupby(["calibration", "alpha"], sort=True):
        current = part.set_index("method")
        budget_rows.append([kind, percent(alpha), *["%d / %d" % (current.loc[m, "tp"], current.loc[m, "fp"]) for m in ["v82_reference", "selected"]]])
    selected = grid.loc[grid.selected & grid.calibration.eq("task_init") & grid.alpha.eq(.01)]
    selection_rows = [[r.fold, r.method, r.tp, r.fp] for r in selected.itertuples()]
    early_rows = [[r.method, "%.3f [%.3f, %.3f]" % (r.auc, r.lo, r.hi)] for r in early.loc[early.scope.eq("all")].itertuples()]
    strata = pd.read_csv(output / "early_auc_strata.csv")
    task = strata.groupby(["suite", "task", "method"]).auc.mean().unstack("method")
    early_pairs = []
    for method in ("absolute_gate", "recovery", "selected"):
        part = task[["v82_reference", method]].dropna()
        difference = (part[method] - part.v82_reference).to_numpy()
        draws = bootstrap_indices(part.index.get_level_values("task"), part.index.get_level_values("suite"))
        lo, hi = np.quantile(difference[draws].mean(axis=1), [.025, .975])
        early_pairs.append(dict(method=method, delta=difference.mean(), lo=lo, hi=hi, tasks=len(part)))
    pd.DataFrame(early_pairs).to_csv(output / "early_paired_comparisons.csv", index=False)
    early_pair_rows = [[r["method"], "%.3f [%.3f, %.3f]" % (r["delta"], r["lo"], r["hi"])] for r in early_pairs]
    state_rows = []
    for failure, part in states.groupby("failure"):
        alarmed = part.first_alarm >= 0
        state_rows.append(["失败" if failure else "成功", len(part), int((part.first_watch >= 0).sum()),
                           int(alarmed.sum()), int((part.alarm_clearances > 0).sum()),
                           int((alarmed & (part.length > part.first_alarm + 4)).sum())])
    checkpoint = curves.loc[curves.scope.eq("all") & curves.calibration.eq("task_init") & curves.alpha.eq(.01)
                            & curves.method.isin(["v82_reference", "selected", "recovery", "reference_confirm3"])
                            & curves["query"].isin([7, 10, 13, 20, 30, 51])]
    checkpoint_rows = [[r.method, r.query, r.tp, r.fp] for r in checkpoint.itertuples()]
    predictions = archive(output / "predictions.npz")
    candidates = pd.read_csv(output / "recovery_candidates.csv")
    inverse = {int(row): i for i, row in enumerate(predictions["global_rows"])}
    candidates["same_budget_reference_first"] = [int(predictions["first"][PRIMARY_KIND, PRIMARY_ALPHA, 0, inverse[int(row)]]) for row in candidates.global_row]
    candidates.to_csv(output / "candidate_comparison.csv", index=False)
    candidate_rows = [[r.global_row, r.old_first, r.same_budget_reference_first, r.first_alarm,
                       r.regrasp_q, r.last_state] for r in candidates.itertuples()]
    deploy = json.loads((output / "profile_deploy.json").read_text())
    report = f"""# v8.2-R 改进实验结果

2026-09-08。依据先前的可恢复异常假设，实现九项预先固定候选和可清除状态接口。
主设置的检测性能**{outcome}**：同折、同成功校准预算下，对照 {int(base.tp)} TP / {int(base.fp)} FP，
开发选中版本 {int(chosen.tp)} TP / {int(chosen.fp)} FP。因此本轮不建议用该选型流程替换原检测规则。
状态接口改进与检测性能改进分开评价。没有新 rollout、物理干预或策略训练。

## 1. 改了什么

- 冻结头增加当前绝对低 mobility 的同时支持；两种参考标准化证据取 min。
- 扰动的两项持续证据限于最近四个 query，避免整个历史永久锁存后参与当前报警。
- 提供 NORMAL / WATCH / ALARM 和信号清除；连续两个有效 query 不超过 WATCH 阈值才清除。
  `ever_alarm` 和 `first_alarm_query` 不随清除重置，因此没有从 FPR 中扣掉自恢复候选。
- 检验总分 K2/K3 的持续确认、去除时间斜率，并包含只延长原分数确认的对照。

当前冻结参考不随执行更新。只用路由概率，物理状态和最终成败不进入在线接口。
WATCH 是独立的名义 10% 成功校准预算，并非“无需计数的报警”。
ALARM 清除仅表示当前路由证据减弱，不表示物理恢复。含旧永久锁存分支的候选仍可能不清除。

## 2. 公平比较和开发选择

40 个任务，A/B 共 32,000 条。全部 B 16,000 条，564 失败、15,436 成功，每条测试一次。
五折每折 A 参考 6,400、A 成功校准池 3,200、A 开发选择 3,200、B 测试 3,200。
四部分 task/init 互斥；只用 A 开发标签选择候选。
参考只确定尺度，成功校准只确定全程峰值阈值，开发只决定候选，三者分离。

主预算是 task/init 名义 1%，不是 B 的实际 FPR 保证。
同折重建的 `v82_reference` 与新候选共享参考和校准划分；它不是原冻结多阈值 v8.2。
原冻结完整 B 的 475 TP / 99 FP 是另一个工作点，不能把改进版相对它少报警说成同预算改善。

开发规则要求 TP 不低于对照、FP 不高于对照，再最大化对较早报警加权的效用。
效用为失败轨迹的 `exp(-max(first_q-7,0)/12)` 均值，漏检记零。
以下只列 A 开发结果，不能当作 B 检测数：

{table(['折/在线配置', '选中候选', '开发 TP', '开发 FP'], selection_rows)}

选择清单在输出 B 指标前固定于 [selection_frozen.json](selection_frozen.json)。
A/B 已用于历史研究，候选设计也受其启发；本轮仍是回顾性开发，不能宣称新的独立盲测。

## 3. 主工作点的完整 B 消融

{table(['候选', 'TP', 'FP', '召回', '实测 FPR'], rows)}

选中流程相对同预算对照补获 {int(pair.gained_tp)} 条失败、丢失 {int(pair.lost_tp)} 条，
增加 {int(pair.added_fp)} 条成功误报、去掉 {int(pair.removed_fp)} 条。
共同检出的 {int(pair.shared_tp)} 条中 {int(pair.earlier)} 条更早、{int(pair.later)} 条更晚。
配对 task 簇区间：召回率差 [{100 * pair.recall_delta_lo:.2f}, {100 * pair.recall_delta_hi:.2f}] 个百分点，
FPR 差 [{100 * pair.fpr_delta_lo:.2f}, {100 * pair.fpr_delta_hi:.2f}] 个百分点。

在固定相同数值阈值时，AND 门只会减少越界；重新做成功预算校准后阈值会变化，
所以增加 AND 门并不保证测试误报下降。本轮结果正说明，合理的局部约束不自动改善整个监测器。

## 4. 所有预先固定预算

{table(['校准单位', '名义预算', '对照 TP / FP', '开发选中 TP / FP'], budget_rows)}

部分次要工作点出现收益，主工作点却退化。不能在看到 B 结果后把有利预算改称主设置。
时钟在主工作点也能达到 274 TP / 22 FP，但报警明显更晚，因此最终计数必须结合时机。

![实际误报与召回](operating_points.png)

## 5. 时间控制与早期排序

{table(['方法', 'q', '累计 TP', '累计 FP'], checkpoint_rows)}

![累计时机](cumulative_results.png)

同 task、同 q 的 q7..q13 连续 AUROC，先平均 query 再平均 task；36 个任务可比较。
区间按 task 簇在 suite 内重采样 2,000 次，以已拟合 profile 为条件。

{table(['连续分数', '早期 AUROC [95% 区间]'], early_rows)}

同一批 task 的配对 AUROC 增量：

{table(['连续分数', '相对对照的增量 [95% 区间]'], early_pair_rows)}

绝对低 mobility 约束改善了早期排序，但没有同步改善主预算的整轨迹检出。
这提示早期排序质量、极低误报的阈值尾部与后期召回需要分别诊断。

![早期条件区分](early_ranking.png)

## 6. 状态清除不是删除误报

{table(['最终结果', '轨迹数', '曾 WATCH', '曾 ALARM', '观察到 ALARM 清除', '报警后至少四个 query 可观察'], state_rows)}

成功和失败的剩余观察时间不同，因此清除比例本身不能证明物理恢复能力。
各轨迹状态存于 [state_episodes.csv](state_episodes.csv)，全部 query 状态存于 `states.npz`。

先前九条释放-报警-再抓取成功候选，仅作诊断。这些来自三个 task/init 组合，且不是独立 trap 标签。
下表同时保留同预算对照，避免把更保守校准造成的少报警归因于恢复机制：

{table(['全局行号', '原冻结报警 q', '同预算对照 q', '选中版本 q', '再抓取 q', '最后路由状态'], candidate_rows)}

`-1` 表示未报警；再抓取和最终成功只用于事后诊断，没有反馈给分数或选型。

## 7. 在线接口和核验

单一在线配置由 A 的 9,600 参考 / 3,200 校准 / 3,200 开发导出，选中 `{deploy['method']}`。
开发流程回退到原分数规则是有效结果，不强制采用退化候选。
五折指标评价选型流程，不能当作这个单一在线 profile 的独立泛化结果。

```python
from recovery import RecoveryGuardMonitor

guard = RecoveryGuardMonitor("profile_deploy.json", checkpoint=checkpoint_sha256)
result = guard.update(hb_router_probs)
# result: state, trigger_now, ever_alarm, first_alarm_query, signal_cleared
```

导入与运行路径见 [README](../../v82_recovery/README.md)。
独立重算 {verification['independent_alarm_checks']:,} 个报警判断、
{verification['calibration_order_statistics']} 个校准顺序统计量；
从原始 HUB Zarr 在线回放 {verification['raw_online_episodes']} 条、
{verification['raw_online_queries']} 个 query，覆盖 {verification['raw_sources']} 个 run，
首次报警与逐 query 状态全部一致。详见 [independent_verification.json](independent_verification.json)。

## 8. 结论

本轮交付了可清除状态接口和完整、可复现的改进实验；数据未支持在主工作点替换 v8.2。
绝对尺度对早期排序的收益可作为后续方向，但“额外 AND 条件 + 更长确认”缺乏稳定的全程收益。
下一轮需要独立的任务进展与恢复标注，区分正常阶段变化、可恢复停滞和持续无进展；
这些信息是否能只由路由可靠推断，仍是待解决的问题。本轮到此停止候选搜索。
"""
    (output / "REPORT_ZH.md").write_text(report)
    write_json(output / "report_verification.json", dict(source_sha256=digest(Path(__file__)),
                artifacts={p.name: digest(p) for p in output.iterdir() if p.suffix in (".png", ".pdf")
                           or p.name in ("REPORT_ZH.md", "candidate_comparison.csv", "early_paired_comparisons.csv")}))
    print("REPORT and three PNG/PDF figures generated", flush=True)


if __name__ == "__main__":
    main()
