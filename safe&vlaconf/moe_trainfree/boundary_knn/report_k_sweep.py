"""Summarize k sensitivity, persistent false alarms, and calibration effects."""

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
from threadpoolctl import threadpool_limits

from sweep_k import ROOT, HERE, OUTPUT, PARENT, METHODS, KS, KINDS, load_npz
from core import digest, trajectory_peak, write_json
from compare_cosine import QUERY_BINS, PROGRESS_BINS
from report_full_corpus import counts, populations, markdown_table, NAMES, ZH_NAMES, COLORS


def primary(table):
    return table.loc[table.calibration.eq("task_init") & table.threshold_mode.eq("recalibrated")]


def evaluate(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    checked = json.loads((output / "verification.json").read_text())
    assert checked["passed"] and checked["sealed_manifest_sha256"] == digest(output / "sealed_manifest.json")
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
    old_evaluation = json.loads((PARENT / "evaluation_summary.json").read_text())
    fields = ["global_row", "source", "run_id", "cohort", "suite", "task", "episode", "init_state_id", "noise_seed",
              "length", "checkpoint", "failure", "recorded_success", "primary_failure_reason", "actual_action_steps", "fold", "scorable_queries"]
    for name in ("trajectory_results.csv", "pooled_metrics.csv"):
        assert digest(PARENT / name) == old_evaluation["artifacts"][name], name
    frame = pd.read_csv(PARENT / "trajectory_results.csv", usecols=fields)
    np.testing.assert_array_equal(frame.global_row, np.arange(32000))
    pd.testing.assert_frame_equal(frame[pd.read_csv(output / "index.csv").columns], pd.read_csv(output / "index.csv"))
    n, m, nk = len(frame), len(METHODS), len(KS)
    first = np.full((2, m, nk, n), -2, np.int16)
    fixed = np.full((m, nk, n), -2, np.int16)
    thresholds = np.full((2, m, nk, n), np.nan, np.float64)
    peaks = np.full((m, nk, n), np.nan, np.float32)
    ownership = np.zeros(n, np.int16)
    calibration = []
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        rows = data["test_rows"]
        np.testing.assert_array_equal(frame.loc[rows, "fold"], info["fold"])
        ownership[rows] += 1
        first[:, :, :, rows] = data["first"]
        fixed[:, :, rows] = data["first_fixed_k20"]
        thresholds[:, :, :, rows] = data["thresholds"][:, :, :, None]
        np.testing.assert_array_equal(np.isfinite(data["scores"][0, 0]).sum(1), frame.loc[rows, "scorable_queries"])
        for mi in range(m):
            for ki in range(nk):
                peaks[mi, ki, rows] = trajectory_peak(data["scores"][mi, ki])
        table = pd.read_csv(output / "calibration" / f"{info['fold']}.csv")
        table["fold"], table["suite"] = info["fold"], info["suite"]
        calibration.append(table)
    np.testing.assert_array_equal(ownership, 1)
    y = frame.failure.to_numpy()
    metric_rows, exact, binned, phase, paired, persistent, ratios, ratio_bins = [], [], [], [], [], [], [], []
    views = (("episode", "recalibrated", first[0]), ("task_init", "recalibrated", first[1]), ("task_init", "fixed_k20", fixed))
    for meta, rows in populations(frame):
        for kind, mode, alarm_array in views:
            for mi, method in enumerate(METHODS):
                for ki, k in enumerate(KS):
                    metric_rows.append(dict(meta, calibration=kind, threshold_mode=mode, method=method, k=int(k),
                        **counts(y[rows], alarm_array[mi, ki, rows])))
        if meta["level"] in ("all", "cohort", "suite"):
            for mi, method in enumerate(METHODS):
                for failure in (False, True):
                    selected = rows[y[rows] == failure]
                    alarm_count = (first[1, mi, :10][:, selected] >= 0).sum(0)
                    baseline_alarm = first[1, mi, -1, selected] >= 0
                    for times in range(11):
                        persistent.append(dict(meta, method=method, failure=failure, k_count=times,
                            episodes=len(selected), count=int((alarm_count == times).sum()),
                            baseline_k20_alarms=int(baseline_alarm.sum()),
                            baseline_alarms_with_this_k_count=int(((alarm_count == times) & baseline_alarm).sum())))
                    for ki, k in enumerate(KS):
                        q = first[1, mi, ki, selected]
                        info = dict(meta, method=method, k=int(k), failure=failure, episodes=len(selected))
                        freq = np.bincount(q + 1, minlength=53)
                        exact.extend(dict(info, query=query, count=int(freq[query + 1])) for query in range(-1, 52))
                        for lo, hi, label in QUERY_BINS:
                            binned.append(dict(info, query_bin=label, count=int(((q >= lo) & (q <= hi)).sum())))
                        alarm = q >= 0
                        progress = 10 * q[alarm] / frame.loc[selected[alarm], "actual_action_steps"].to_numpy()
                        assert ((progress >= 0) & (progress < 1)).all()
                        for pi, label in enumerate(PROGRESS_BINS):
                            phase.append(dict(info, progress_bin=label, alarms=int(alarm.sum()),
                                count=int(((progress >= pi / 4) & (progress < (pi + 1) / 4)).sum())))
                        bq = first[1, mi, -1, selected]
                        both = alarm & baseline_alarm
                        paired.append(dict(info, both=int(both.sum()), removed_from_k20=int((baseline_alarm & ~alarm).sum()),
                            new_vs_k20=int((alarm & ~baseline_alarm).sum()), neither=int((~alarm & ~baseline_alarm).sum()),
                            earlier=int((both & (q < bq)).sum()), later=int((both & (q > bq)).sum()),
                            same_query=int((both & (q == bq)).sum())))
        if meta["level"] not in ("all", "cohort", "suite", "task"):
            continue
        for failure in (False, True):
            selected = rows[y[rows] == failure]
            for ki, k in enumerate(KS):
                values = peaks[0, ki, selected] / thresholds[1, 0, ki, selected]
                finite = np.isfinite(values)
                info = dict(meta, method="euclidean", k=int(k), failure=failure, episodes=len(selected), scorable=int(finite.sum()))
                ratios.append(dict(info, **{f"p{int(q * 100):02}": float(np.quantile(values[finite], q)) if finite.any() else np.nan
                                            for q in (0, .1, .25, .5, .75, .9, 1)}))
                for label, selected_bin in (("unscorable", ~finite), ("<=0.5", finite & (values <= .5)),
                    ("(0.5,1]", finite & (values > .5) & (values <= 1)), ("(1,1.5]", (values > 1) & (values <= 1.5)),
                    ("(1.5,2]", (values > 1.5) & (values <= 2)), (">2", values > 2)):
                    ratio_bins.append(dict(info, ratio_bin=label, count=int(selected_bin.sum())))
    metrics = pd.DataFrame(metric_rows)
    tables = dict(metrics=metrics, pooled_metrics=metrics.loc[metrics.level.eq("all")],
        cohort_metrics=metrics.loc[metrics.level.eq("cohort")], suite_metrics=metrics.loc[metrics.level.eq("suite")],
        task_metrics=metrics.loc[metrics.level.eq("task")], thresholds=pd.concat(calibration, ignore_index=True),
        alarm_query_distribution=pd.DataFrame(exact), alarm_query_bins=pd.DataFrame(binned),
        alarm_progress_distribution=pd.DataFrame(phase), paired_with_k20=pd.DataFrame(paired),
        k_persistence_distribution=pd.DataFrame(persistent), euclidean_peak_ratio_quantiles=pd.DataFrame(ratios),
        euclidean_peak_ratio_bins=pd.DataFrame(ratio_bins))
    baseline = pd.read_csv(PARENT / "pooled_metrics.csv")
    main = primary(tables["pooled_metrics"])
    for row in main.loc[main.k.eq(20)].itertuples():
        old = baseline.loc[baseline.method.eq(row.method) & baseline.calibration.eq("task_init")].iloc[0]
        for key in ("episodes", "failures", "successes", "tp", "fp", "fn", "tn"):
            assert getattr(row, key) == old[key]
    wide = frame.copy()
    for ki, k in enumerate(KS):
        q = first[1, 0, ki]
        wide[f"k{k}_first_alarm_query"] = q
        wide[f"k{k}_threshold"] = thresholds[1, 0, ki]
        wide[f"k{k}_peak_score"] = np.where(np.isfinite(peaks[0, ki]), peaks[0, ki], np.nan)
        wide[f"k{k}_peak_ratio"] = np.where(np.isfinite(peaks[0, ki]), peaks[0, ki] / thresholds[1, 0, ki], np.nan)
        wide[f"k{k}_control_progress_at_alarm"] = np.where(q >= 0, 10 * q / frame.actual_action_steps, np.nan)
        wide[f"k{k}_fixed_k20_threshold_first_alarm"] = fixed[0, ki]
    wide["alarm_count_k1_to_k10"] = (first[1, 0, :10] >= 0).sum(0)
    tables["euclidean_trajectory_results"] = wide
    np.savez_compressed(output / "all_episode_results.npz", global_rows=frame.global_row.to_numpy(), methods=np.asarray(METHODS),
        ks=KS, calibration_kinds=np.asarray(KINDS), first=first, thresholds=thresholds, peak_scores=peaks,
        first_fixed_k20=fixed)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    paths = [PARENT / name for name in ("trajectory_results.csv", "pooled_metrics.csv", "evaluation_summary.json")]
    paths.extend((HERE / "report_full_corpus.py", HERE / "compare_cosine.py"))
    summary = dict(unique_trajectories=n, failures=int(y.sum()), successes=int((~y).sum()), methods=METHODS, ks=KS,
        scorer_manifest_sha256=digest(output / "sealed_manifest.json"), verification_sha256=digest(output / "verification.json"),
        reporter_sha256=digest(Path(__file__)), inputs={str(path.relative_to(ROOT)): digest(path) for path in paths},
        rows={name: len(table) for name, table in tables.items()},
        artifacts={f"{name}.csv": digest(output / f"{name}.csv") for name in tables} |
                  {"all_episode_results.npz": digest(output / "all_episode_results.npz")})
    write_json(output / "evaluation_summary.json", summary)
    return tables, summary


def make_figures(output, tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    main = primary(tables["pooled_metrics"])
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), layout="constrained")
    for mi, method in enumerate(METHODS):
        part = main.loc[main.method.eq(method)].sort_values("k")
        for ax, field in zip(axes, ("recall", "fpr")):
            ax.plot(part.k, part[field], marker="o", ms=4, color=COLORS[mi], label=NAMES[method])
    for ax, title in zip(axes, ("Failure detection: 1,096 failed trajectories", "False alarms: 30,904 successful trajectories")):
        ax.set_title(title, fontsize=12)
        ax.set_xticks(KS)
        ax.set_xlabel("Number of nearest reference chunks (k)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.18)
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Frozen task-held-out banks | 32,000 unique trajectories | each k recalibrated at nominal 5%", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"k_sweep_comparison.{suffix}", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), layout="constrained")
    suites = primary(tables["suite_metrics"])
    for i, suite in enumerate(sorted(suites.suite.unique())):
        part = suites.loc[suites.suite.eq(suite) & suites.method.eq("euclidean")].sort_values("k")
        axes[0].plot(part.k, part.fpr, marker="o", ms=4, color=COLORS[i], label=suite.removeprefix("libero_").title())
    axes[0].set_title("Euclidean false alarms by held-out suite", fontsize=12)
    axes[0].legend(frameon=False, loc="center right")
    all_metrics = tables["pooled_metrics"]
    for mode, color, label in (("recalibrated", COLORS[0], "Recalibrate threshold for each k"),
                               ("fixed_k20", "#62696b", "Keep k=20 threshold (diagnostic)")):
        part = all_metrics.loc[all_metrics.method.eq("euclidean") & all_metrics.calibration.eq("task_init") & all_metrics.threshold_mode.eq(mode)].sort_values("k")
        axes[1].plot(part.k, part.fpr, marker="o", ms=4, color=color, label=label)
    axes[1].set_title("Euclidean: score shrinkage and threshold adaptation", fontsize=12)
    axes[1].legend(frameon=False, loc="lower right")
    for ax in axes:
        ax.set_xticks(KS)
        ax.set_xlabel("k")
        ax.set_ylabel("False alarm rate")
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    fig.suptitle("Same reference chunks, same held-out tasks, same calibration trajectories", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"k_sweep_false_alarm_diagnosis.{suffix}", dpi=170)
    plt.close(fig)


def write_report(output, tables):
    main = primary(tables["pooled_metrics"])
    e = main.loc[main.method.eq("euclidean")].set_index("k").sort_index()
    c = main.loc[main.method.eq("cosine")].set_index("k").sort_index()
    mixed = main.loc[main.method.eq("cosine_neighbors_euclidean_score")].set_index("k").sort_index()
    tasks = primary(tables["task_metrics"])
    top = tasks.loc[tasks.method.eq("euclidean") & tasks.k.eq(20)].sort_values("fp", ascending=False).head(8)
    persistence = tables["k_persistence_distribution"].loc[lambda x: x.level.eq("all") & x.method.eq("euclidean") & ~x.failure].set_index("k_count")
    pairs = tables["paired_with_k20"].loc[lambda x: x.level.eq("all") & x.method.eq("euclidean") & ~x.failure].set_index("k")
    threshold_table = tables["thresholds"].loc[lambda x: x.method.eq("euclidean") & x.calibration.eq("task_init")]
    tau = threshold_table.pivot(index=["suite", "fold"], columns="k", values="threshold")
    fixed = tables["pooled_metrics"].loc[lambda x: x.method.eq("euclidean") & x.calibration.eq("task_init") & x.threshold_mode.eq("fixed_k20")].set_index("k")
    task_rows = []
    for row in top.itertuples():
        part = tasks.loc[tasks.method.eq("euclidean") & tasks.task.eq(row.task)].set_index("k")
        task_rows.append([row.task, row.successes] + [f"{int(part.loc[k, 'fp'])} ({part.loc[k, 'fpr']:.2%})" for k in (1, 5, 10, 20)])
    ratio_bins = tables["euclidean_peak_ratio_bins"]
    ratio_rows = []
    for task in top.head(2).task:
        for k in (1, 20):
            part = ratio_bins.loc[ratio_bins.level.eq("task") & ratio_bins.task.eq(task) & ratio_bins.k.eq(k) & ~ratio_bins.failure].set_index("ratio_bin")
            ratio_rows.append([task, k] + [int(part.loc[label, "count"]) for label in ("unscorable", "<=0.5", "(0.5,1]", "(1,1.5]", "(1.5,2]", ">2")])
    lines = ["# k=1 至 10：全库误报敏感性扫描", "",
        "固定上一轮 16 轮完整任务留出配置、参考库成员和 10D 特征；每种 k 都测试同样 32,000 条唯一轨迹，"
        "其中 30,904 成功、1,096 失败。k=20 作为复现对照。没有训练模型或新增 rollout。", "",
        "## 欧氏 kNN 主结果", "",
        "每个 k 独立在同一 A 成功校准集上按 task/init 组最大值校准，标称 alpha=5%。", "",
        markdown_table([[k, int(r.tp), f"{r.recall:.2%}", int(r.fp), f"{r.fpr:.2%}", f"{r.precision:.2%}"] for k, r in e.iterrows()],
                       ["k", "检出 /1,096", "召回率", "误报 /30,904", "误报率", "报警精确率"]), "",
        f"从 k=20 改为 k=1，误报从 {int(e.loc[20, 'fp']):,} 变为 {int(e.loc[1, 'fp']):,}，"
        f"检出从 {int(e.loc[20, 'tp'])} 变为 {int(e.loc[1, 'tp'])}。"
        f"k=1 至 10 的实际误报率范围为 {e.loc[1:10, 'fpr'].min():.2%} 至 {e.loc[1:10, 'fpr'].max():.2%}；"
        "不能把扫描中的最好结果当成已经在新数据上验证的最优 k。", "",
        f"本次冻结设置下，k=1 至 10 均比 k=20 误报更多。k=1 净增加 {int(e.loc[1, 'fp'] - e.loc[20, 'fp'])} 条误报，"
        f"检出总数仅净增加 {int(e.loc[1, 'tp'] - e.loc[20, 'tp'])} 条。"
        "这个结果不支持仅通过减少邻居数解决当前高误报。", "",
        "![k 扫描](k_sweep_comparison.png)", "",
        "## 为什么减小 k 不一定降低误报", "",
        "对欧氏 kNN，分数是最近 k 个参考 chunk 距离的平均值。k 减小会使这个平均距离不增，"
        "但成功校准轨迹的分数也一起缩小，因此对应阈值也不增。是否报警取决于距离相对阈值的位置，"
        "不是单独看距离大小。下面分别报告逐 k 校准和固定 k=20 阈值的结果：", "",
        markdown_table([[k, f"{e.loc[k, 'fpr']:.2%}", int(e.loc[k, 'tp']), f"{fixed.loc[k, 'fpr']:.2%}", int(fixed.loc[k, 'tp'])]
                        for k in KS], ["k", "独立校准：误报率", "独立校准：检出", "保留旧阈值：误报率", "保留旧阈值：检出"]), "",
        "固定旧阈值会改变该 k 的校准工作点，需连同召回率一起比较，不能只看误报下降就归因为 k 更好。"
        f"本轮 k=1 阈值是各自 k=20 阈值的 {(tau[1] / tau[20]).min():.1%} 至 {(tau[1] / tau[20]).max():.1%}。逐轮数值如下：", "",
        markdown_table([[fold, f"{r[1]:.6f}", f"{r[10]:.6f}", f"{r[20]:.6f}", f"{r[1] / r[20]:.3f}"]
                        for (_, fold), r in tau.iterrows()], ["轮次", "k=1", "k=10", "k=20", "阈值 k1/k20"]), "",
        "![误报和校准](k_sweep_false_alarm_diagnosis.png)", "",
        "## 误报是否集中在同样的任务", "",
        markdown_table(task_rows, ["任务", "成功轨迹数", "k=1 误报", "k=5 误报", "k=10 误报", "k=20 误报"]), "",
        f"原 k=20 的 {int(e.loc[20, 'fp']):,} 条误报中，{int(pairs.loc[1, 'both']):,} 条在 k=1 仍然报警，"
        f"{int(pairs.loc[1, 'removed_from_k20']):,} 条不再报警；同时 k=1 新增 {int(pairs.loc[1, 'new_vs_k20']):,} 条成功误报。"
        f"原 k=20 误报中有 {int(persistence.loc[10, 'baseline_alarms_with_this_k_count']):,} 条在 k=1 至 10 全部报警。", "",
        "以下是原 k=20 误报最多的两个任务，成功轨迹的全程峰值/对应阈值分布。比值大于 1 即报警。"
        "k=1 已经只使用最近的一个点；如果此时仍大量超过 1，扩大近邻平均半径就不能解释全部误报。", "",
        markdown_table(ratio_rows, ["任务", "k", "不可评分", "≤0.5", "(0.5,1]", "(1,1.5]", "(1.5,2]", ">2"]), "",
        "不同任务成功状态与历史参考几何的差异、参考点覆盖，以及跨任务校准阈值的适配性仍需区分。"
        "本轮保持它们固定，能够检验 k 的影响，但没有单独改变参考覆盖或任务校准，不能由此确定唯一物理原因。", "",
        "## 余弦和混合打分对照", "",
        markdown_table([[k, int(c.loc[k, "tp"]), f"{c.loc[k, 'fpr']:.2%}", int(mixed.loc[k, "tp"]), f"{mixed.loc[k, 'fpr']:.2%}"] for k in KS],
                       ["k", "余弦检出 /1,096", "余弦误报率", "余弦近邻+欧氏打分检出", "混合打分误报率"]), "",
        "欧氏近邻加余弦打分、不中心化余弦以及 episode 校准结果也全部保存在 CSV 中。"
        "五种方法各 11 个 k 配置，没有按测试标签选择阈值。", "",
        "## 数据与复现", "",
        "- [pooled_metrics.csv](pooled_metrics.csv)：全量所有 k、距离和校准对照。",
        "- [suite_metrics.csv](suite_metrics.csv)、[cohort_metrics.csv](cohort_metrics.csv)、[task_metrics.csv](task_metrics.csv)：套件、A/B、任务明细。",
        "- [euclidean_trajectory_results.csv](euclidean_trajectory_results.csv)：恰好 32,000 行，所有 k 的首次报警、阈值、峰值和实际控制进度。`-1` 表示没有报警。",
        "- [all_episode_results.npz](all_episode_results.npz)：`first/thresholds` 为 `[calibration,method,k,global_row]=[2,5,11,32000]`；`peak_scores/first_fixed_k20` 为 `[5,11,32000]`。",
        "- `predictions/*.npz`：每轮所有校准及测试 chunk 分数，`scores[method,k,test_position,query]`，含 `test_rows` 全局行号映射。q0 至 q6 及结束后为 NaN；分数不使用测试结果。",
        "- [alarm_query_distribution.csv](alarm_query_distribution.csv)、[alarm_query_bins.csv](alarm_query_bins.csv)、[alarm_progress_distribution.csv](alarm_progress_distribution.csv)：每个 k 的全部首次报警分布；控制进度使用实际 action_steps。",
        "- [paired_with_k20.csv](paired_with_k20.csv)、[k_persistence_distribution.csv](k_persistence_distribution.csv)：逐轨迹报警变化和跨 k 持续报警数量。",
        "- [euclidean_peak_ratio_bins.csv](euclidean_peak_ratio_bins.csv)：各任务和各 k 的峰值/阈值分布。",
        "- [thresholds.csv](thresholds.csv)、[k20_replay_audit.csv](k20_replay_audit.csv)、[verification.json](verification.json)、[report_verification.json](report_verification.json)：校准、复现和核验。",
        "- [固定方案](../../boundary_knn/K_SWEEP_PROTOCOL_ZH.md)。", "",
        "```bash", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/sweep_k.py' score --output /tmp/himoe-k-sweep",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/sweep_k.py' verify --output /tmp/himoe-k-sweep",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/report_k_sweep.py' --output /tmp/himoe-k-sweep", "```", ""]
    (output / "REPORT_ZH.md").write_text("\n".join(lines))


def verify_report(output, tables, summary):
    data = load_npz(output / "all_episode_results.npz")
    wide = pd.read_csv(output / "euclidean_trajectory_results.csv")
    np.testing.assert_array_equal(wide.global_row, np.arange(32000))
    assert not wide.duplicated(["source", "episode"]).any()
    for ki, k in enumerate(KS):
        np.testing.assert_array_equal(wide[f"k{k}_first_alarm_query"], data["first"][1, 0, ki])
        np.testing.assert_array_equal(wide[f"k{k}_fixed_k20_threshold_first_alarm"], data["first_fixed_k20"][0, ki])
        np.testing.assert_allclose(wide[f"k{k}_threshold"], data["thresholds"][1, 0, ki], rtol=1e-12, atol=1e-12)
        peak = data["peak_scores"][0, ki]
        np.testing.assert_allclose(wide[f"k{k}_peak_score"], np.where(np.isfinite(peak), peak, np.nan), rtol=1e-7, equal_nan=True)
        np.testing.assert_allclose(wide[f"k{k}_peak_ratio"], np.where(np.isfinite(peak), peak / data["thresholds"][1, 0, ki], np.nan), rtol=1e-7, equal_nan=True)
        q = data["first"][1, 0, ki]
        np.testing.assert_allclose(wide[f"k{k}_control_progress_at_alarm"], np.where(q >= 0, 10 * q / wide.actual_action_steps, np.nan), equal_nan=True)
    np.testing.assert_array_equal(wide.alarm_count_k1_to_k10, (data["first"][1, 0, :10] >= 0).sum(0))
    y = wide.failure.to_numpy()
    for row in tables["pooled_metrics"].itertuples():
        mi, ki = list(METHODS).index(row.method), int(np.flatnonzero(KS == row.k)[0])
        q = data["first_fixed_k20"][mi, ki] if row.threshold_mode == "fixed_k20" else data["first"][KINDS.index(row.calibration), mi, ki]
        for key, expected in counts(y, q).items():
            np.testing.assert_allclose(getattr(row, key), expected)
    keys = ["level", "cohort", "suite", "task", "method", "k", "failure"]
    for name, total, group_keys in (("alarm_query_distribution", "episodes", keys), ("alarm_query_bins", "episodes", keys),
        ("alarm_progress_distribution", "alarms", keys), ("euclidean_peak_ratio_bins", "episodes", keys),
        ("k_persistence_distribution", "episodes", [key for key in keys if key != "k"])):
        for _, part in tables[name].groupby(group_keys):
            assert int(part["count"].sum()) == int(part[total].iloc[0])
    for row in tables["paired_with_k20"].itertuples():
        assert row.both + row.removed_from_k20 + row.new_vs_k20 + row.neither == row.episodes
        assert row.earlier + row.later + row.same_query == row.both
    assert digest(Path(__file__)) == summary["reporter_sha256"]
    for name, expected in summary["inputs"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in summary["artifacts"].items():
        assert digest(output / name) == expected, name
    artifacts = dict(summary["artifacts"])
    for name in ("REPORT_ZH.md", "k_sweep_comparison.png", "k_sweep_comparison.pdf",
                 "k_sweep_false_alarm_diagnosis.png", "k_sweep_false_alarm_diagnosis.pdf", "evaluation_summary.json"):
        artifacts[name] = digest(output / name)
    from PIL import Image
    for name in ("k_sweep_comparison.png", "k_sweep_false_alarm_diagnosis.png"):
        with Image.open(output / name) as im:
            assert min(im.size) >= 800 and np.asarray(im.convert("RGB")).std() > 15
    write_json(output / "report_verification.json", dict(passed=True, unique_rows=len(wide),
        primary_euclidean_decision_checks=len(wide) * len(KS), all_method_configurations=len(METHODS) * len(KS),
        source_sha256=digest(Path(__file__)), inputs=summary["inputs"], artifacts=artifacts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    with threadpool_limits(limits=1):
        tables, summary = evaluate(output)
        make_figures(output, tables)
        write_report(output, tables)
        verify_report(output, tables, summary)
    print(primary(tables["pooled_metrics"]).loc[lambda x: x.method.eq("euclidean")].to_string(index=False), flush=True)
    print("K SWEEP REPORT VERIFIED", flush=True)


if __name__ == "__main__":
    main()
