"""Generate the SAFE-style experiments, including the reused v7 implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from core import PRIMARY, SEEDS, write_json

HERE = Path(__file__).resolve().parent
NAMES = {
    PRIMARY: "Signed distance sum (primary)", "stats_contrast__current": "Stats contrast",
    "load_contrast__current": "Router load contrast", "history_contrast__current": "Router history contrast",
    "behavior_contrast__current": "Action contrast", "eef_motion_low__current": "Past EEF motion",
    "stats_success__cumsum": "Success distance, shared scale",
    "success_only__cumsum": "Success-only distance sum", "success_only__current": "Success-only distance",
    "stats_ratio__cumsum": "Bounded distance ratio sum", "stats_positive__current": "Positive contrast",
    "v7_guard_constant": "v7 guard, constant band", "v7_guard_timeband": "v7 guard, time band",
    "v7_freeze_constant": "v7 freeze", "v7_turbulence_constant": "v7 turbulence",
    "v7_success_fusion_constant": "v7 + success distance", "clock": "Clock", "random": "Random",
    "unlabeled_reference_budget": "v7 unlabeled budget", "success_calibration_budget": "v7 success budget",
}
DISPLAY = (PRIMARY, "load_contrast__current", "history_contrast__current", "behavior_contrast__current",
           "eef_motion_low__current", "success_only__current", "success_only__cumsum", "stats_ratio__cumsum",
           "v7_guard_constant", "v7_turbulence_constant", "v7_success_fusion_constant", "clock", "random")
COLORS = ("#16796f", "#ce5835", "#76579a", "#a37913", "#3264a2", "#636363")


def table(frame, formats=None):
    formats = formats or {}
    result = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for key, value in row.items():
            if pd.isna(value):
                values.append("NA")
            elif key in formats:
                values.append(format(value, formats[key]))
            else:
                values.append(str(value))
        result.append("| " + " | ".join(values) + " |")
    return "\n".join(result)


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_combined(output, filename):
    values = []
    for stage, directory in (("initial", output), ("followup", output / "followup"), ("v7", output / "v7")):
        frame = pd.read_csv(directory / filename)
        frame["stage"] = stage
        values.append(frame)
    return pd.concat(values, ignore_index=True)


def overview(output, ranking, alarms, budget):
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    selected = ("history_contrast__current", "success_only__cumsum", "v7_guard_constant", "v7_success_fusion_constant", "clock")
    common = ranking.loc[(ranking.scope == "unseen") & (ranking.view == "common_horizon") & (ranking.reference_cap == 4096)]
    pivot = common.groupby(["suite", "method"]).task_macro_auc.mean().unstack("method")
    positions = np.arange(len(pivot))
    for i, method in enumerate(selected):
        axes[0, 0].bar(positions + (i - 2) * 0.15, pivot[method], width=0.15, color=COLORS[i], label=NAMES[method])
    axes[0, 0].axhline(0.5, color="black", lw=0.8, ls=":")
    axes[0, 0].set(xticks=positions, xticklabels=[s.replace("libero_", "") for s in pivot.index], ylim=(0.2, 1), ylabel="Task-macro AUROC", title="A. Same observation length within each task")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    means = ranking.loc[(ranking.scope == "unseen") & ranking.view.isin(["full", "common_horizon"]) & (ranking.reference_cap == 4096)].groupby(["method", "view"]).task_macro_auc.mean().unstack("view")
    for i, method in enumerate(selected):
        axes[0, 1].plot([0, 1], means.loc[method, ["full", "common_horizon"]], marker="o", color=COLORS[i], label=NAMES[method])
    axes[0, 1].axhline(0.5, color="black", lw=0.8, ls=":")
    axes[0, 1].set(xticks=[0, 1], xticklabels=["Full recorded rollout", "Task-matched horizon"], ylim=(0.35, 1.03), ylabel="Task-macro AUROC", title="B. Separating detection from duration")
    axes[0, 1].legend(fontsize=8, loc="lower left")
    curve_methods = ("v7_guard_constant", "v7_guard_timeband", "v7_success_fusion_constant", "success_only__cumsum")
    for i, method in enumerate(curve_methods):
        curve = alarms.loc[(alarms.method == method) & (alarms.scope == "unseen") & (alarms.calibration == "episode") & (alarms.reference_cap == 4096)].groupby("alpha").mean(numeric_only=True)
        axes[1, 0].plot(curve.fpr, curve.recall, "o-", color=COLORS[i], label=NAMES[method])
        axes[1, 1].plot(curve.t_det, curve.balanced_accuracy, "o-", color=COLORS[i], label=NAMES[method])
    for i, method in enumerate(("unlabeled_reference_budget", "success_calibration_budget")):
        curve = budget.loc[(budget.method == method) & (budget.scope == "unseen")].groupby("budget").mean(numeric_only=True)
        axes[1, 0].plot(curve.fpr, curve.recall, "s--", color=COLORS[i+4], label=NAMES[method])
        axes[1, 1].plot(curve.t_det, curve.balanced_accuracy, "s--", color=COLORS[i+4], label=NAMES[method])
    axes[1, 0].set(xlabel="Observed success false-alarm rate", ylabel="Failure recall", title="C. Online operating curves on unseen tasks", xlim=(0, 0.55), ylim=(0, 1.03))
    axes[1, 0].legend(fontsize=8, loc="lower right")
    axes[1, 1].set(xlabel="Mean detection time (misses = 1)", ylabel="Balanced accuracy", title="D. Accuracy and detection time", xlim=(0.35, 1), ylim=(0.5, 1))
    axes[1, 1].legend(fontsize=8, loc="lower left")
    save(figure, output / "figures/overview")


def temporal_figures(output, frame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained", sharex="col", gridspec_kw={"height_ratios": [3, 1]})
    curve_rows, example_rows = [], []
    examples, example_axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    for col, suite in enumerate(sorted(frame.suite.unique())):
        name = f"{suite}_{SEEDS[0]}"
        with np.load(output / "v7/predictions" / f"{name}.npz", allow_pickle=False) as data:
            part = frame.iloc[data["test_rows"]].reset_index(drop=True)
            method = list(data["methods"].astype(str)).index("v7_guard_constant")
            score = data["scores"][method]
            test_unseen = data["test_unseen"]
            threshold = float(data["thresholds"][0, list(data["alphas"]).index(0.05), method])
            first = data["first"][0, list(data["alphas"]).index(0.05), method]
        for fail, color, label in ((False, COLORS[0], "Success"), (True, COLORS[1], "Failure")):
            rows = np.flatnonzero((part.failure == fail) & test_unseen)
            med, low, high, count = [], [], [], []
            for q in range(52):
                values = score[rows, q]
                values = values[np.isfinite(values)]
                count.append(len(values))
                quantiles = np.quantile(values, [.25, .5, .75]) if len(values) >= 5 else [np.nan] * 3
                low.append(quantiles[0]); med.append(quantiles[1]); high.append(quantiles[2])
                curve_rows.append({"suite": suite, "query": q, "failure": fail, "observations": len(values),
                                   "median": quantiles[1], "q25": quantiles[0], "q75": quantiles[2]})
            axes[0, col].plot(med, color=color, label=label)
            axes[0, col].fill_between(np.arange(52), low, high, color=color, alpha=0.15)
            axes[1, col].plot(count, color=color)
            candidates = rows[(first[rows] >= 0)] if fail else rows[(first[rows] < 0)]
            if not len(candidates):
                candidates = rows
            if len(candidates):
                ordered = candidates[np.argsort(first[candidates] if fail else part.iloc[candidates].length.to_numpy())]
                chosen = int(ordered[len(ordered) // 2])
                row = part.iloc[chosen]
                axis = example_axes[int(fail), col]
                axis.plot(np.arange(row.length), score[chosen, :row.length], color=color)
                axis.axhline(threshold, color="black", ls="--", lw=1, label="Calibrated threshold")
                if first[chosen] >= 0:
                    axis.axvline(first[chosen], color="#76579a", ls=":", label="First alarm")
                axis.set(title=f"{suite.replace('libero_', '')}: {label.lower()} ep {row.episode}", xlabel="Decision query", ylabel="v7 guard score", yscale="symlog")
                example_rows.append({"suite": suite, "task": row.task, "episode": row.episode,
                                     "run_id": row.run_id, "failure": bool(fail), "first_alarm": int(first[chosen]), "threshold": threshold})
        axes[0, col].axhline(threshold, color="black", ls="--", lw=1, label="Threshold")
        axes[0, col].set(title=suite.replace("libero_", ""), ylabel="v7 guard score", yscale="symlog")
        axes[0, col].legend(fontsize=8)
        axes[1, col].set(xlabel="Decision query", ylabel="Observed n", yscale="symlog")
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].set_xlim(0, int(part.loc[test_unseen, "length"].max()) - 1)
    fig.suptitle("v7 risk over the observed prefix: first task split, unseen tasks")
    examples.suptitle("Recorded examples selected by median detection time or successful length")
    save(fig, output / "figures/risk_over_time")
    save(examples, output / "figures/trajectory_examples")
    pd.DataFrame(curve_rows).to_csv(output / "risk_curve_summary.csv", index=False)
    pd.DataFrame(example_rows).to_csv(output / "example_index.csv", index=False)


def feature_map(output, frame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
    audits = {a["source"]: a for a in json.loads((output / "extraction_audit.json").read_text())}
    rng = np.random.default_rng(20260907)
    task_mapping = []
    for col, suite in enumerate(sorted(frame.suite.unique())):
        with np.load(output / "predictions" / f"{suite}_{SEEDS[0]}.npz", allow_pickle=False) as archive:
            rows = archive["test_rows"][archive["test_unseen"]]
        part = frame.iloc[rows]
        points = []
        tasks = sorted(part.task.unique())
        task_mapping.extend({"suite": suite, "figure_label": f"T{i+1}", "task": task} for i, task in enumerate(tasks))
        for source, current in part.groupby("source"):
            with np.load(output / "features" / audits[source]["output"], allow_pickle=False) as archive:
                vectors = archive["stats"]
            for row in current.itertuples():
                for q in range(0, row.length, 3):
                    vector = vectors[row.episode, q].reshape(8, 4)[4:]
                    points.append((vector[:, 0].mean(), vector[:, 2].mean(), row.failure, tasks.index(row.task)))
        points = np.asarray(points)
        if len(points) > 2500:
            points = points[rng.choice(len(points), 2500, replace=False)]
        for fail, color, label in ((0, COLORS[0], "Success"), (1, COLORS[1], "Failure")):
            x = points[points[:, 2] == fail]
            axes[0, col].scatter(x[:, 0], x[:, 1], s=5, alpha=0.35, color=color, label=label, rasterized=True)
        axes[1, col].scatter(points[:, 0], points[:, 1], c=points[:, 3], cmap="tab10", vmin=0, vmax=9, s=5, alpha=.4, rasterized=True)
        axes[1, col].legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=4,
            color=plt.get_cmap("tab10")(i), label=f"T{i+1}") for i in range(len(tasks))], fontsize=8)
        axes[0, col].set_title(suite.replace("libero_", ""))
        axes[0, col].legend(fontsize=8)
        for row in range(2):
            axes[row, col].set(xlabel="Back-layer mean token entropy / log(32)", ylabel="Token JS / log(32)")
            axes[row, col].ticklabel_format(axis="both", style="sci", scilimits=(-3, 3), useOffset=False)
    fig.suptitle("Fixed MoE readout coordinates: outcome color above, task color below")
    save(fig, output / "figures/moe_feature_map")
    pd.DataFrame(task_mapping).to_csv(output / "feature_map_tasks.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE.parent / "results/round3_safe")
    args = parser.parse_args()
    output = args.input.resolve()
    (output / "figures").mkdir(exist_ok=True)
    ranking = read_combined(output, "ranking_metrics.csv")
    alarms = read_combined(output, "alarm_metrics.csv")
    budget = pd.read_csv(output / "v7/budget_metrics.csv")
    frame = pd.read_csv(output / "outcome_alignment.csv")
    verification = json.loads((output / "verification.json").read_text())
    extension_verification = json.loads((output / "extension_verification.json").read_text())
    overview(output, ranking, alarms, budget)
    temporal_figures(output, frame)
    feature_map(output, frame)
    rank_view = ranking.loc[(ranking.scope == "unseen") & (ranking.reference_cap == 4096) & ranking.view.isin(["full", "common_horizon"])]
    averaged = rank_view.groupby(["method", "view"])[["task_macro_auc", "within_init_macro_auc"]].mean()
    rank_table = pd.DataFrame([{"方法": method, "统一观察长度 AUC": averaged.loc[(method, "common_horizon"), "task_macro_auc"],
        "同 init AUC": averaged.loc[(method, "common_horizon"), "within_init_macro_auc"],
        "完整轨迹 AUC": averaged.loc[(method, "full"), "task_macro_auc"]} for method in DISPLAY])
    alpha_view = alarms.loc[(alarms.scope == "unseen") & (alarms.reference_cap == 4096) & (alarms.alpha == .05)]
    means = alpha_view.groupby(["method", "calibration"])[["recall", "fpr", "balanced_accuracy", "t_det"]].mean()
    alarm_names = (PRIMARY, "success_only__cumsum", "v7_guard_constant", "v7_guard_timeband", "v7_turbulence_constant", "v7_success_fusion_constant", "clock")
    alarm_table = pd.DataFrame([dict(方法=method, 校准=kind, **means.loc[(method, kind)].to_dict()) for method in alarm_names for kind in ("episode", "task_init")])
    budget_table = budget.loc[(budget.scope == "unseen") & budget.budget.isin([.01, .03, .05, .10])].groupby(["method", "budget"])[["recall", "fpr", "balanced_accuracy", "t_det"]].mean().reset_index()
    suite_table = alarms.loc[(alarms.scope == "unseen") & (alarms.method == "v7_guard_constant") & (alarms.alpha == .05) & (alarms.calibration == "episode")].groupby("suite")[["recall", "fpr", "t_det"]].mean().reset_index()
    cap_table = ranking.loc[(ranking.method == PRIMARY) & (ranking.scope == "unseen") & (ranking.view == "common_horizon")].groupby("reference_cap")[["task_macro_auc", "within_init_macro_auc"]].mean().reset_index()
    physics = read_combined(output, "physical_timing.csv")
    physics = physics.loc[(physics.scope == "unseen") & (physics.calibration == "episode")]
    budget_physics = pd.read_csv(output / "v7/budget_physical_timing.csv")
    budget_physics = budget_physics.loc[budget_physics.scope == "unseen"].copy()
    budget_physics["method"] = budget_physics.method + "@5%"
    physics = pd.concat([physics, budget_physics], ignore_index=True)
    physical_rows = []
    for method, group in physics.groupby("method"):
        observed = group.loc[group.drop_goal_release.notna()]
        fired = observed.first_alarm >= 0
        early = fired & (observed.first_alarm < observed.drop_goal_release)
        physical_rows.append({"方法": method, "有脱手时刻的测试出现次数": len(observed), "检出": int(fired.sum()),
                              "严格早于脱手": int(early.sum()), "检出者延迟中位数(query)":
                              float((observed.loc[fired, "first_alarm"] - observed.loc[fired, "drop_goal_release"]).median())})
    physical_table = pd.DataFrame(physical_rows)
    physical_table.to_csv(output / "physical_method_summary.csv", index=False)
    summary = json.loads((output / "evaluation_summary.json").read_text())
    v7_audit = json.loads((output / "v7/v7_input_audit.json").read_text())
    decisions = pd.read_csv(output / "episode_decisions.csv")
    formats = {"recall": ".3f", "fpr": ".3f", "balanced_accuracy": ".3f", "t_det": ".3f"}
    doc = f"""# SAFE 思路的无训练 MoE 实验：复用 v7、参考距离和成功校准

2026-09-06。已完成原始实验、明确标记的距离后续探索，以及按用户要求加入的 v7 复用实验。
所有方法均冻结 VLA，不训练新增 MLP/LSTM/概率头。允许历史标签构建参考和筛选成功校准集。

**当前最有价值的是保留 v7 的时间机制和原有多分支校准。** 有符号距离直接累加的原主方法表现较弱；
纯成功 MoE 距离在统一观察长度后有一定区分力，但阈值迁移和与 v7 的简单融合仍有明显误报代价。
这轮检验的是完整执行过程中的失败检测，是否先于物理故障另作时序评价。

![实验概览](figures/overview.png)

## 1. 数据与 SAFE 思路的对应

两批缓存共 32,000 条轨迹、508,023 次决策，覆盖四套 LIBERO 的 40 个任务。
A 批 16,000 条含 532 次失败，B 批 16,000 条含 564 次失败。直接核验全部 80 个来源。
四个 suite 对应不同 checkpoint，各自建参考和阈值；不跨模型比较专家坐标。

每个 checkpoint 做三次随机 7 seen / 3 unseen 任务划分。每折 A 的 1,680 条轨迹作参考、
560 条作校准；B 的 560 条 seen 初态留出与 1,200 条 unseen 轨迹作测试。
参考、校准、seen 测试的 task/init 分离，A/B 噪声种子分离。
重复划分总共涉及 {decisions.global_row.nunique():,} 条不重复的 B 轨迹，
其中 {decisions.loc[decisions.scope == 'unseen', 'global_row'].nunique():,} 条、
{decisions.loc[decisions.scope == 'unseen', 'task'].nunique()} 个任务曾进入 unseen 测试；重复出现不算新增独立样本。

沿用 [SAFE](https://arxiv.org/html/2506.09937v2) 的逻辑：完整前缀风险、成功轨迹校准、
未见任务评价、准确率与报警时间曲线。使用固定距离和 v7 代替训练的检测头。
同任务统一观察长度取该任务 B 批最短轨迹，仅用于离线 AUC；长度不输入检测器。
AUC 表先在每个 suite/划分内对有两类结果的任务求平均，再对有效设置等权平均。
报警表先在每个设置内计算轨迹级 recall/FPR，再对设置等权平均。
同初态和共同噪声使轨迹之间有依赖；不将跨划分重复出现当作独立新增样本。

## 2. 实际复用了 v7 的什么

直接调用 `moe-v7-0905/method/intrinsic_guard_monitor.py` 与 `unlabeled_budget_calibration.py`。
保留后四层相对 q1..q4 的冻结信号、W6 平滑、flow 曲率和复现性变化的持续确认，以及锁存 AND/OR。
没有重新扫描窗口、分位数组合或学习组合权重。
复用 {v7_audit['reused_episodes']:,} 条已有特征，缺失的 {v7_audit['new_raw_episodes']} 条从原始路由补齐，
另按 suite/批次抽样重放核验旧缓存。原版 monitor 也直接用于验证新的预算 profile。

三种校准需分清：原有无标签总报警预算、独立成功轨迹的经验报警预算、以及成功轨迹的有限样本
conformal 顺序统计量。后两者使用成功标签，但都没有训练模型。
预算函数内部没有标签参数，不代表外层的“筛选成功轨迹”不使用标签。

另外构造了一个连续 guard：三个原有 head 先用参考 median/MAD 固定标准化，
`G = max(freeze, min(prefix_max(acceleration), prefix_max(recurrence_loss)))`。
它保留锁存关系，但不是原版三个独立阈值的精确等价；因此必须保留原版 Boolean 规则作对照。

## 3. 原有 v7 多分支规则的在线工作点

下表是未见任务结果。`unlabeled_reference_budget` 是参考集总报警预算；
`success_calibration_budget` 是独立成功校准集的经验报警预算。均不能承诺新任务 FPR 等于 budget。

{table(budget_table, formats)}

5% 点上，原规则加无标签预算的平均 recall 约 88.1%、FPR 约 5.95%；
改用独立成功轨迹校准后 recall 约 95.1%、FPR 约 7.35%。
与旧报告不同，这里只用本折 seen 任务标定，各设置的参考和测试构成也不同，不直接比较跨报告绝对数值。

## 4. 统一观察长度后的区分能力

{table(rank_table, {'统一观察长度 AUC': '.3f', '同 init AUC': '.3f', '完整轨迹 AUC': '.3f'})}

原主方法 `stats_contrast__cumsum` 已保留。有符号分数的早期负值可能抵消后续异常；
SAFE-MLP 本身累加非负 sigmoid 输出，不能把有符号距离累加视作等价。
事后固定的正部、距离比、纯成功距离见 [后续设计](../../safe_protocol/FOLLOWUP_ZH.md)。
纯成功距离从归一化到查表都不使用失败样本，累计版本的同任务等时长 AUC 约 0.693。
原 `stats_success` 只是不减失败距离，归一化仍共享双类参考，不能把它称为完整的 success-only 方法。

完整轨迹 clock AUC 近乎 1，统一观察长度后是 0.5。这说明“多数失败跑满时间”的信息非常强。
v7 在完整执行中的区分能力有用途，但其早期信息和时长信息要分开。这里不会因为它没有提前预报
每次脱手就判定在线检测无效，也不会把接近结束时的高 AUC 包装成早期预测。

## 5. 成功轨迹 CP：同一标称 alpha=0.05

{table(alarm_table, formats)}

`episode` 对每条成功轨迹的峰值校准；`task_init` 先取同初态成功分支的组内最大值。
组校准更保守，但共享噪声、任务聚类和跨任务分布偏移仍不满足简单的独立同分布解释。
v7 的常数峰值阈值没有被时间带稳定改进。固定 max 融合提高了部分召回和等时长 AUC，
但在逐轨迹校准下 FPR 约 17.4%；不能仅凭 recall 把融合称为改进。

v7 guard 常数阈值在各 suite 的结果也不一致：

{table(suite_table, formats)}

## 6. 风险随执行过程变化

![v7风险随时间](figures/risk_over_time.png)

曲线使用第一次划分的 unseen 任务，分别显示最终成功/失败轨迹的中位数和四分位范围；
下方给出仍有有限读数的样本数。少于 5 条时不画分位数，防止将尾部极少数样本当作稳定现象。
这是事后分组展示，运行时不读取最终标签。q0 附近的 v7 空白是历史不足导致的等待期。

![轨迹案例](figures/trajectory_examples.png)

每个 suite 的失败案例按检出时间中位数选择，成功案例按未误报轨迹长度中位数选择。
精确任务、episode 与阈值在 `example_index.csv`；这些例子用于解释流程，不替代总体指标。

![固定MoE坐标](figures/moe_feature_map.png)

两个坐标是后四层 entropy 与 token JS 的固定统计；上图按结果、下图按任务着色。
图中的 T1/T2/T3 与具体任务的对应保存在 `feature_map_tasks.csv`。
没有训练投影或检测器。这仅是低维观察，不证明存在统一失败区域，也不证明其他 MoE 信息无效。

## 7. 与物理失败的时间关系

{table(physical_table, {'检出者延迟中位数(query)': '.1f'})}

只匹配未满足 goal 中标为脱手相关失败的物体。正常放置另一个物体的 release 不算失败开始。
表中是不同随机划分中的测试出现次数，包含重复；只用于时序诊断，不当作独立实验次数。
release 的时间分辨率约一个 action chunk，且不一定是不可恢复故障。
提前检出数必须连同对应方法的误报率阅读；高误报方法也可能更早触发。
SAFE 式在线检测和真正故障前预测是不同评价目标，表格明确保留二者的边界。

## 8. 参考量消融与核验

原主方法每类参考点数上限的消融：

{table(cap_table, {'task_macro_auc': '.3f', 'within_init_macro_auc': '.3f'})}

每条参考轨迹最多均匀取 8 个已记录时刻，防止长轨迹单纯贡献更多参考点。
没有使用测试标签调整点数、评分方向、窗口或部署阈值。

已运行原有与新增相关测试，共 49 项通过。第三轮原始实验独立核验 {verification['calibration_rank_checks']:,} 个校准秩，
用另一个距离实现复算 {verification['independent_primary_threshold_checks']} 个主方法阈值；
{verification['extraction_prefix_checks']} 次提取前缀检查、36 条原始轨迹在线主方法重放通过。
v7 另有 24 条缓存核验和 72 次原版 monitor/profile 重放。预算及后续方法的核验文件分别保留。
两组后续实验另核验 {extension_verification['calibration_rank_checks']:,} 个校准秩，
复算 {extension_verification['recomputed_distance_thresholds']} 个距离阈值，并完成各 36 条原始轨迹的
六种距离读数与五种 v7 新读出重放，分数和首次报警一致，见 `extension_verification.json`。
主距离 monitor 在路由张量已可用时的 CPU 用时中位数约 {verification['median_primary_monitor_ms_per_query']:.3f} ms/query，
不包含 VLA 推理、路由采集或磁盘读取；不与论文的不同硬件数字直接比较。

## 9. 范围与产物

所有队列已经被探索；本轮是可复现的回顾性实验，不是独立盲测或完整 SAFE 原版复现。
两批全任务数据没有完整末层 hidden 或真实专家输出，因此本轮证明不了相对 SAFE hidden 的优势。
已有局部配对结果在第二轮报告；当前结论只覆盖可核验的路由及行为输入。
没有训练、没有新 rollout、没有真实机器人或恢复干预结果。GPU 仍被已有任务占用。

可继续发展的具体基础是原 v7 的多分支时间规则，并把成功校准作为一个明确的选项；
纯成功距离可作为不同类型的信号，是否融合必须连同误报和检测时间一起判断。
本轮没有用新分数替换既有 v7，也没有删除不利的候选结果。

设计：[主协议](../../safe_protocol/PROTOCOL_ZH.md)、[v7 复用](../../safe_protocol/V7_ZH.md)。
运行与代码：[README](../../safe_protocol/README.md)。
完整数据分别在本目录、`followup/`、`v7/` 的 `ranking_metrics.csv`、`alarm_metrics.csv`、
`task_*_metrics.csv`、`physical_timing.csv` 和各自 `sealed_manifest.json`。
原 v7 预算规则的结果在 `v7/budget_metrics.csv`，不要混同 CP 的 alpha。
"""
    (output / "REPORT_ZH.md").write_text(doc, encoding="utf-8")
    rank_table.to_csv(output / "headline_ranking.csv", index=False)
    alarm_table.to_csv(output / "headline_alarms.csv", index=False)
    write_json(output / "report_summary.json", {"unique_test_episodes": int(decisions.global_row.nunique()),
        "unique_unseen_episodes": int(decisions.loc[decisions.scope == "unseen", "global_row"].nunique()),
        "unique_unseen_tasks": int(decisions.loc[decisions.scope == "unseen", "task"].nunique()),
        "figures": [p.name for p in sorted((output / "figures").glob("*.png"))], "new_model_training": False})
    print(output / "REPORT_ZH.md", flush=True)


if __name__ == "__main__":
    main()
