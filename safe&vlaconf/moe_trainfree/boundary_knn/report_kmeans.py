"""Evaluate reference clustering with explicit precision/recall tradeoffs."""

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

from kmeans_reference import ROOT, HERE, OUTPUT, PARENT, METHODS, CLUSTERS, VARIANTS, ALPHAS, KINDS, PRIMARY_C, method_info, load_npz
from core import digest, trajectory_peak, write_json
from report_full_corpus import counts, populations, markdown_table
from compare_cosine import QUERY_BINS, PROGRESS_BINS

VARIANT_ZH = dict(centroid_distance="最近中心距离", assigned_radius="最近簇半径归一化", union_radius="所有簇相对半径取最小")
VARIANT_EN = dict(centroid_distance="Nearest centroid", assigned_radius="Nearest-cluster radius", union_radius="Union of scaled clusters")
COLORS = ("#b45f3c", "#168579", "#6482aa", "#8c75a9", "#62696b")


def method_name(method, english=False):
    if method == "knn20":
        return "Euclidean kNN-20" if english else "原欧氏 kNN-20"
    if method == "norm_only":
        return "Vector norm" if english else "向量范数"
    info = method_info(method)
    return f"C={info['clusters']}: {(VARIANT_EN if english else VARIANT_ZH)[info['variant']]}"


def grouped(table, alpha=.05):
    return table.loc[table.calibration.eq("task_init") & np.isclose(table.alpha, alpha)]


def evaluate(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    verified = json.loads((output / "verification.json").read_text())
    assert verified["passed"] and verified["sealed_manifest_sha256"] == digest(output / "sealed_manifest.json")
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
    parent_summary = json.loads((PARENT / "evaluation_summary.json").read_text())
    for name in ("trajectory_results.csv", "pooled_metrics.csv"):
        assert digest(PARENT / name) == parent_summary["artifacts"][name], name
    fields = ["global_row", "source", "run_id", "cohort", "suite", "task", "episode", "init_state_id", "noise_seed",
        "length", "checkpoint", "failure", "recorded_success", "primary_failure_reason", "actual_action_steps", "fold",
        "scorable_queries", "drop_goal_release", "failed_goal_release"]
    frame = pd.read_csv(PARENT / "trajectory_results.csv", usecols=fields)
    index = pd.read_csv(output / "index.csv")
    pd.testing.assert_frame_equal(frame[index.columns], index)
    np.testing.assert_array_equal(frame.global_row, np.arange(32000))
    n, m = len(frame), len(METHODS)
    first = np.full((2, len(ALPHAS), m, n), -2, np.int16)
    thresholds = np.full((2, len(ALPHAS), m, n), np.nan, np.float64)
    peaks = np.full((m, n), -np.inf, np.float32)
    ownership = np.zeros(n, np.int16)
    calibration = []
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        rows = data["test_rows"]
        ownership[rows] += 1
        np.testing.assert_array_equal(frame.loc[rows, "fold"], info["fold"])
        first[:, :, :, rows] = data["first"]
        thresholds[:, :, :, rows] = data["thresholds"][:, :, :, None]
        np.testing.assert_array_equal(np.isfinite(data["scores"][0]).sum(1), frame.loc[rows, "scorable_queries"])
        for mi in range(m):
            peaks[mi, rows] = trajectory_peak(data["scores"][mi])
        table = pd.read_csv(output / "calibration" / f"{info['fold']}.csv")
        table["fold"], table["suite"] = info["fold"], info["suite"]
        calibration.append(table)
    np.testing.assert_array_equal(ownership, 1)
    y = frame.failure.to_numpy()
    assert (int(y.sum()), int((~y).sum())) == (1096, 30904)
    metrics, exact, binned, progress, paired, physical = [], [], [], [], [], []
    for meta, rows in populations(frame):
        for ki, kind in enumerate(KINDS):
            for ai, alpha in enumerate(ALPHAS):
                for mi, method in enumerate(METHODS):
                    metrics.append(dict(meta, **method_info(method), calibration=kind, alpha=float(alpha),
                        **counts(y[rows], first[ki, ai, mi, rows])))
        if meta["level"] not in ("all", "cohort", "suite"):
            continue
        for ai, alpha in enumerate(ALPHAS):
            for mi, method in enumerate(METHODS):
                base = dict(meta, **method_info(method), alpha=float(alpha))
                q = first[1, ai, mi, rows]
                for failure in (False, True):
                    selected = rows[y[rows] == failure]
                    values = first[1, ai, mi, selected]
                    fired = values >= 0
                    info = dict(base, failure=failure, episodes=len(selected))
                    freq = np.bincount(values + 1, minlength=53)
                    exact.extend(dict(info, query=query, count=int(freq[query + 1])) for query in range(-1, 52))
                    for lo, hi, label in QUERY_BINS:
                        binned.append(dict(info, query_bin=label, count=int(((values >= lo) & (values <= hi)).sum())))
                    phase = 10 * values[fired] / frame.loc[selected[fired], "actual_action_steps"].to_numpy()
                    assert ((phase >= 0) & (phase < 1)).all()
                    for pi, label in enumerate(PROGRESS_BINS):
                        progress.append(dict(info, progress_bin=label, alarms=int(fired.sum()),
                            count=int(((phase >= pi / 4) & (phase < (pi + 1) / 4)).sum())))
                    bq = first[1, ai, 0, selected]
                    old_alarm = bq >= 0
                    both = old_alarm & fired
                    paired.append(dict(info, both=int(both.sum()), removed_from_knn20=int((old_alarm & ~fired).sum()),
                        new_vs_knn20=int((fired & ~old_alarm).sum()), neither=int((~fired & ~old_alarm).sum()),
                        earlier=int((both & (values < bq)).sum()), later=int((both & (values > bq)).sum()),
                        same_query=int((both & (values == bq)).sum())))
                event = frame.loc[rows, "drop_goal_release"].to_numpy()
                eligible = y[rows] & np.isfinite(event)
                physical.append(dict(base, event_episodes=int(eligible.sum()), before=int((eligible & (q >= 0) & (q < event)).sum()),
                    at=int((eligible & (q >= 0) & (q == event)).sum()), after=int((eligible & (q >= 0) & (q > event)).sum()),
                    missed=int((eligible & (q < 0)).sum())))
    all_metrics = pd.DataFrame(metrics)
    tables = dict(metrics=all_metrics, pooled_metrics=all_metrics.loc[all_metrics.level.eq("all")],
        cohort_metrics=all_metrics.loc[all_metrics.level.eq("cohort")], suite_metrics=all_metrics.loc[all_metrics.level.eq("suite")],
        task_metrics=all_metrics.loc[all_metrics.level.eq("task")], thresholds=pd.concat(calibration, ignore_index=True),
        alarm_query_distribution=pd.DataFrame(exact), alarm_query_bins=pd.DataFrame(binned),
        alarm_progress_distribution=pd.DataFrame(progress), paired_with_knn20=pd.DataFrame(paired), physical_summary=pd.DataFrame(physical))
    primary = grouped(tables["pooled_metrics"])
    candidate = primary.loc[primary.clusters.gt(0)].sort_values(["precision", "recall"], ascending=False).iloc[0]
    candidate_c = int(candidate.clusters)
    # This subset is descriptive: all model predictions were sealed before outcomes were joined.
    selected_methods = list(dict.fromkeys([*METHODS[:2], *[f"c{c}_{v}" for c in (PRIMARY_C, candidate_c) for v in VARIANTS]]))
    columns = {}
    for method in selected_methods:
        mi = METHODS.index(method)
        columns[f"{method}_peak_score"] = np.where(np.isfinite(peaks[mi]), peaks[mi], np.nan)
        for ai, alpha in enumerate(ALPHAS):
            name = f"{method}_a{round(alpha * 100):02}"
            q = first[1, ai, mi]
            columns[f"{name}_first_alarm_query"] = q
            columns[f"{name}_threshold"] = thresholds[1, ai, mi]
            columns[f"{name}_control_progress_at_alarm"] = np.where(q >= 0, 10 * q / frame.actual_action_steps, np.nan)
    wide = pd.concat((frame, pd.DataFrame(columns, index=frame.index)), axis=1)
    tables["trajectory_results"] = wide
    strict_ai = int(np.flatnonzero(np.isclose(ALPHAS, .02))[0])
    candidate_mi = METHODS.index(str(candidate.method))
    strict_first = first[1, strict_ai, candidate_mi]
    error_mask = (strict_first >= 0) != y
    errors = frame.loc[error_mask].copy()
    errors["method"], errors["alpha"] = candidate.method, .02
    errors["error"] = np.where(errors.failure, "false_negative", "false_positive")
    errors["first_alarm_query"] = strict_first[error_mask]
    errors["threshold"] = thresholds[1, strict_ai, candidate_mi, error_mask]
    errors["peak_score"] = peaks[candidate_mi, error_mask]
    tables["candidate_errors_alpha002"] = errors
    old_metrics = pd.read_csv(PARENT / "pooled_metrics.csv")
    for name, old_name in (("knn20", "euclidean"), ("norm_only", "norm_only")):
        row = primary.loc[primary.method.eq(name)].iloc[0]
        old = old_metrics.loc[old_metrics.method.eq(old_name) & old_metrics.calibration.eq("task_init")].iloc[0]
        for key in ("tp", "fp", "fn", "tn", "episodes", "failures", "successes"):
            assert row[key] == old[key]
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "all_episode_results.npz", global_rows=frame.global_row.to_numpy(), methods=np.asarray(METHODS),
        alphas=ALPHAS, calibration_kinds=np.asarray(KINDS), first=first, thresholds=thresholds, peak_scores=peaks)
    paths = [PARENT / name for name in ("trajectory_results.csv", "pooled_metrics.csv", "evaluation_summary.json")]
    paths.extend((HERE / "report_full_corpus.py", HERE / "compare_cosine.py"))
    summary = dict(unique_trajectories=n, failures=int(y.sum()), successes=int((~y).sum()), methods=METHODS,
        primary_clusters=PRIMARY_C, alphas=ALPHAS, selected_export_methods=selected_methods,
        exploratory_candidate_method=str(candidate.method), exploratory_candidate_clusters=candidate_c,
        candidate_selected_after_evaluation=True, not_a_new_blind_test=True,
        scorer_manifest_sha256=digest(output / "sealed_manifest.json"), verification_sha256=digest(output / "verification.json"),
        reporter_sha256=digest(Path(__file__)), inputs={str(path.relative_to(ROOT)): digest(path) for path in paths},
        rows={name: len(table) for name, table in tables.items()},
        artifacts={f"{name}.csv": digest(output / f"{name}.csv") for name in tables} |
                  {"all_episode_results.npz": digest(output / "all_episode_results.npz")})
    write_json(output / "evaluation_summary.json", summary)
    return tables, summary


def make_figures(output, tables, summary):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    main = grouped(tables["pooled_metrics"])
    baseline = main.set_index("method")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), layout="constrained")
    for vi, variant in enumerate(VARIANTS):
        part = main.loc[main.variant.eq(variant)].sort_values("clusters")
        for ax, field in zip(axes, ("precision", "recall", "fpr")):
            ax.plot(part.clusters, part[field], marker="o", color=COLORS[vi], label=VARIANT_EN[variant])
    for ax, field, title in zip(axes, ("precision", "recall", "fpr"), ("Alarm precision", "Failure recall", "False alarm rate")):
        ax.axhline(baseline.loc["knn20", field], color="#3f4549", ls="--", label="Euclidean kNN-20")
        ax.axhline(baseline.loc["norm_only", field], color="#949a9f", ls=":", label="Vector norm")
        ax.set_xscale("log", base=2)
        ax.set_xticks(CLUSTERS, [str(c) for c in CLUSTERS])
        ax.set_xlabel("Number of clusters (C)")
        ax.set_title(title, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    axes[1].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("32,000 unique task-held-out trajectories | frozen 10D banks | grouped calibration at nominal 5%", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"kmeans_cluster_sweep.{suffix}", dpi=170)
    plt.close(fig)
    selected = list(dict.fromkeys(["knn20", "norm_only", summary["exploratory_candidate_method"], f"c{PRIMARY_C}_union_radius"]))
    all_primary = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("task_init")]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7), layout="constrained")
    for mi, method in enumerate(selected):
        part = all_primary.loc[all_primary.method.eq(method)].sort_values("alpha")
        axes[0].plot(part.recall, part.precision, marker="o", color=COLORS[mi], label=method_name(method, True))
        axes[1].plot(part.fpr, part.recall, marker="o", color=COLORS[mi], label=method_name(method, True))
        if method in ("knn20", summary["exploratory_candidate_method"]):
            for row in part.loc[np.isclose(part.alpha, .02) | np.isclose(part.alpha, .05)].itertuples():
                axes[0].annotate(f"alpha={row.alpha:.0%}", (row.recall, row.precision), xytext=(5, 7), textcoords="offset points", fontsize=8)
    axes[0].set(xlabel="Failure recall", ylabel="Alarm precision", xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="False alarm rate", ylabel="Failure recall", ylim=(0, 1))
    axes[1].set_xlim(left=0)
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.18)
    axes[0].set_title("Precision must be read with missed failures", fontsize=12)
    axes[1].set_title("Same calibration budgets: 2%, 3%, 5%, 10%", fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Operating points on the same corpus | exploratory cluster candidate selected after the sweep", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"kmeans_precision_recall.{suffix}", dpi=170)
    plt.close(fig)


def metric_table(part, include_alpha=False):
    rows = []
    for r in part.itertuples():
        row = [method_name(r.method)]
        if include_alpha:
            row.append(f"{r.alpha:.0%}")
        row.extend((f"{r.tp}/1096", f"{r.recall:.2%}", f"{r.fp}/30904", f"{r.fpr:.2%}", f"{r.precision:.2%}"))
        rows.append(row)
    columns = ["方法"] + (["标称预算"] if include_alpha else []) + ["检出失败", "召回率", "成功误报", "误报率", "精确率"]
    return markdown_table(rows, columns)


def write_report(output, tables, summary):
    main = grouped(tables["pooled_metrics"])
    base = main.set_index("method").loc["knn20"]
    candidate_method, candidate_c = summary["exploratory_candidate_method"], summary["exploratory_candidate_clusters"]
    candidate = main.set_index("method").loc[candidate_method]
    default_methods = ["knn20", "norm_only"] + [f"c{PRIMARY_C}_{v}" for v in VARIANTS]
    candidate_methods = [f"c{candidate_c}_{v}" for v in VARIANTS]
    selected = list(dict.fromkeys(["knn20", "norm_only", candidate_method, f"c{PRIMARY_C}_union_radius"]))
    budgets = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("task_init") & x.method.isin(selected)].sort_values(["alpha", "method"])
    suites = grouped(tables["suite_metrics"])
    tasks = grouped(tables["task_metrics"])
    top = tasks.loc[tasks.method.eq("knn20")].sort_values("fp", ascending=False).head(8)
    task_rows = []
    for row in top.itertuples():
        new = tasks.loc[tasks.method.eq(candidate_method) & tasks.task.eq(row.task)].iloc[0]
        task_rows.append([row.task, f"{row.fp}/{row.successes}", f"{int(new.fp)}/{int(new.successes)}",
            f"{row.tp}/{row.failures}", f"{int(new.tp)}/{int(new.failures)}"])
    grid = []
    for c in CLUSTERS:
        row = [c]
        for variant in VARIANTS:
            value = main.set_index("method").loc[f"c{c}_{variant}"]
            row.extend((f"{value.precision:.2%}", f"{value.recall:.2%}", f"{value.fpr:.2%}"))
        grid.append(row)
    pair = tables["paired_with_knn20"].loc[lambda x: x.level.eq("all") & x.method.eq(candidate_method) & np.isclose(x.alpha, .05)]
    false_pair = pair.loc[~pair.failure].iloc[0]
    true_pair = pair.loc[pair.failure].iloc[0]
    timings = tables["alarm_query_bins"].loc[lambda x: x.level.eq("all") & np.isclose(x.alpha, .05)]
    timing_rows = []
    for _, _, label in QUERY_BINS:
        if label == "q0-6":
            continue
        parts = [timings.loc[timings.method.eq(method) & timings.failure.eq(failure)].set_index("query_bin")
                 for failure in (True, False) for method in ("knn20", candidate_method)]
        timing_rows.append([label] + [int(part.loc[label, "count"]) for part in parts])
    phase = tables["alarm_progress_distribution"].loc[lambda x: x.level.eq("all") & np.isclose(x.alpha, .05)]
    phase_rows = []
    for method in ("knn20", candidate_method):
        for failure in (True, False):
            part = phase.loc[phase.method.eq(method) & phase.failure.eq(failure)].set_index("progress_bin")
            phase_rows.append([method_name(method), "失败检出" if failure else "成功误报"] + [int(part.loc[label, "count"]) for label in PROGRESS_BINS])
    physical = tables["physical_summary"].loc[lambda x: x.level.eq("all") & x.method.isin(selected) & (np.isclose(x.alpha, .02) | np.isclose(x.alpha, .05))]
    clusters = pd.read_csv(output / "cluster_summary.csv")
    sparse = clusters.groupby("clusters").agg(cluster_instances=("cluster", "size"), sparse_fallbacks=("fallback", "sum"),
                                               min_points=("points", "min"), max_points=("points", "max")).reset_index()
    lines = ["# K-means：能否提高报警精确率", "",
        "同一 32,000 条轨迹、30,904 成功、1,096 失败，同一完整任务留出和成功参考库。"
        "策略参数冻结；只离线拟合 K-means 中心和参考半径，没有监督失败分类器训练或新增 rollout。"
        "K-means 有聚类拟合，不是完全不拟合任何参数。", "",
        f"本次扫描中，{method_name(candidate_method)} 在标称 5% 下精确率为 {candidate.precision:.2%}，"
        f"原欧氏 kNN 为 {base.precision:.2%}；成功误报从 {int(base.fp):,} 降到 {int(candidate.fp):,}，"
        f"失败检出从 {int(base.tp)} 变为 {int(candidate.tp)}。"
        f"C={candidate_c} 是看完本轮完整扫描后得到的探索候选，不能当成新的盲测成绩；事先指定的主簇数仍是 C={PRIMARY_C}。", "",
        "## 三种打分", "",
        "- 最近中心距离：参考成功 chunk 聚成 C 簇，取当前向量到最近中心的欧氏距离。",
        "- 最近簇半径归一化：先按欧氏距离选簇，再除以该簇参考距离的 90% 半径。",
        "- 所有簇相对半径取最小：允许任意一个正常簇解释当前向量，取所有距离/半径的最小值。", "",
        "每种方法的最终阈值仍只用原 A 成功校准轨迹确定。测试任务从中心、半径、归一化和阈值计算中全部排除。", "",
        "## 事先指定的主配置", "", metric_table(main.loc[main.method.isin(default_methods)]), "",
        "## 扫描候选", "", metric_table(main.loc[main.method.isin(candidate_methods)]), "",
        f"相对原 kNN，候选净减少 {int(base.fp - candidate.fp):,} 条成功误报，检出总数净减少 {int(base.tp - candidate.tp)} 条。"
        f"原误报中 {int(false_pair.removed_from_knn20):,} 条消失，同时新增 {int(false_pair.new_vs_knn20):,} 条误报；"
        f"原检出中 {int(true_pair.removed_from_knn20)} 条不再报警，又新增检出 {int(true_pair.new_vs_knn20)} 条。"
        "应同时观察精确率和漏检，不能仅以报警数量少判断好坏。", "",
        "## 全部簇数", "", markdown_table(grid, ["C", "中心：精确率", "中心：召回", "中心：FPR",
            "最近簇尺度：精确率", "最近簇尺度：召回", "最近簇尺度：FPR", "簇并集：精确率", "簇并集：召回", "簇并集：FPR"]), "",
        "![完整簇数扫描](kmeans_cluster_sweep.png)", "",
        "簇数更多并不保证更好。粗粒度中心会合并一些参考点之间的细节，半径归一化会改变不同局部模式的容忍范围；"
        "这些操作也可能覆盖失败状态。哪一种改变起作用需要结合本表的原始中心和归一化对照，不能只归功于使用了 K-means。", "",
        "## 精确率与召回的取舍", "", metric_table(budgets, include_alpha=True), "",
        "这些是相同校准预算下的工作点，实际误报率并不相同。更严格阈值可以提高精确率，但对应的漏检也必须保留在分母中。"
        "没有根据测试标签单独优化阈值，也没有设置部署用的精确率目标。", "",
        "![精确率和召回](kmeans_precision_recall.png)", "",
        "## 跨批次和任务表现", "",
        markdown_table([[r.cohort, method_name(r.method), f"{r.tp}/{r.failures}", f"{r.fp}/{r.successes}", f"{r.precision:.2%}"]
                        for r in grouped(tables["cohort_metrics"]).loc[lambda x: x.method.isin(("knn20", candidate_method))].itertuples()],
                       ["批次", "方法", "检出失败", "成功误报", "精确率"]), "",
        markdown_table([[r.suite, method_name(r.method), f"{r.tp}/{r.failures}", f"{r.fp}/{r.successes}", f"{r.precision:.2%}"]
                        for r in suites.loc[suites.method.isin(("knn20", candidate_method))].itertuples()],
                       ["suite", "方法", "检出失败", "成功误报", "精确率"]), "",
        markdown_table(task_rows, ["原误报最多任务", "kNN 误报", "候选误报", "kNN 检出", "候选检出"]), "",
        "改善并不均匀。应重点保留尚未解决的高误报任务和表现退化的 suite，避免用全库平均值掩盖局部问题。"
        "更严格 2% 设置下的剩余误报与漏检已分别标记在 [candidate_errors_alpha002.csv](candidate_errors_alpha002.csv)。", "",
        "## 报警时序", "", "以下使用标称 5% 的 task/init 校准。q 为零起始推理 chunk，每个完整 chunk 执行 10 个动作。", "",
        markdown_table(timing_rows, ["首次 q", "kNN 失败", "候选失败", "kNN 成功", "候选成功"]), "",
        markdown_table(phase_rows, ["方法", "轨迹结果", "进度 0-25%", "25-50%", "50-75%", "75-100%"]), "",
        "进度按 10*q/实际 action_steps 事后分箱，不用轨迹补齐长度，也不是检测器输入。"
        "完整逐 q 和各控制进度分布都已导出，没有用一个中位数代替分布。", "",
        "已有 B 目标物体脱手事件子集：", "",
        markdown_table([[method_name(r.method), f"{r.alpha:.0%}", r.event_episodes, r.before, r.at, r.after, r.missed] for r in physical.itertuples()],
                       ["方法", "标称预算", "事件轨迹", "早于事件", "同 q", "晚于事件", "没报警"]), "",
        "目标物体脱手不是不可逆失败起点。整段轨迹的检测精确率也不等于早期预警精确率或干预救回率。", "",
        "## 参考簇覆盖", "", markdown_table([[r.clusters, r.cluster_instances, r.sparse_fallbacks, r.min_points, r.max_points] for r in sparse.itertuples()],
            ["C", "16 轮簇总数", "少于 20 点的回退簇", "最少参考点", "最多参考点"]), "",
        "少于 20 点的簇使用该轮、该 C 的总体参考残差 90% 半径。全部半径和参考任务构成见 [cluster_summary.csv](cluster_summary.csv)。"
        "聚类不能凭空补充未见任务的真实成功状态，因此参考覆盖和任务偏移仍需要进一步区分。", "",
        "## 数据与复现", "",
        "- [pooled_metrics.csv](pooled_metrics.csv)：全部 23 种分数、4 档预算、2 种校准的全量结果。",
        "- [cohort_metrics.csv](cohort_metrics.csv)、[suite_metrics.csv](suite_metrics.csv)、[task_metrics.csv](task_metrics.csv)：A/B、suite、任务明细。",
        "- [trajectory_results.csv](trajectory_results.csv)：32,000 行，基线、C=32 和探索候选 C 的所有预算首次报警、阈值、峰值及实际进度。",
        "- [candidate_errors_alpha002.csv](candidate_errors_alpha002.csv)：探索候选在严格 2% 设置下的所有误报和漏检，含原始轨迹身份。",
        "- [all_episode_results.npz](all_episode_results.npz)：`first/thresholds[calibration,alpha,method,global_row]=[2,4,23,32000]`，`peak_scores=[23,32000]`，各轴名称一并保存。",
        "- `predictions/*.npz`：所有校准和测试的逐 chunk 分数 `[method,trajectory_position,query]`，含全局行号；无效位置为 NaN。",
        "- `profiles/*_c*.npz`：参考簇中心、分配、半径、稀疏回退标志和原始参考 episode/query 身份。",
        "- [alarm_query_distribution.csv](alarm_query_distribution.csv)、[alarm_progress_distribution.csv](alarm_progress_distribution.csv)：所有预算的精确报警分布。",
        "- [paired_with_knn20.csv](paired_with_knn20.csv)、[physical_summary.csv](physical_summary.csv)：逐轨迹报警变化与已有物理事件时序。",
        "- [verification.json](verification.json)、[report_verification.json](report_verification.json)：参考隔离、独立距离、全部阈值/报警及数据哈希核验。",
        "- [固定方案](../../boundary_knn/KMEANS_PROTOCOL_ZH.md)。", "",
        "```bash", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/kmeans_reference.py' score --output /tmp/himoe-kmeans",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/kmeans_reference.py' verify --output /tmp/himoe-kmeans",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/report_kmeans.py' --output /tmp/himoe-kmeans", "```", ""]
    (output / "REPORT_ZH.md").write_text("\n".join(lines))


def verify_report(output, tables, summary):
    data = load_npz(output / "all_episode_results.npz")
    wide = pd.read_csv(output / "trajectory_results.csv")
    np.testing.assert_array_equal(wide.global_row, np.arange(32000))
    assert not wide.duplicated(["source", "episode"]).any()
    y = wide.failure.to_numpy()
    for method in summary["selected_export_methods"]:
        mi = METHODS.index(method)
        peak = data["peak_scores"][mi]
        np.testing.assert_allclose(wide[f"{method}_peak_score"], np.where(np.isfinite(peak), peak, np.nan), rtol=1e-7, equal_nan=True)
        for ai, alpha in enumerate(ALPHAS):
            prefix = f"{method}_a{round(alpha * 100):02}"
            q = data["first"][1, ai, mi]
            np.testing.assert_array_equal(wide[f"{prefix}_first_alarm_query"], q)
            np.testing.assert_allclose(wide[f"{prefix}_threshold"], data["thresholds"][1, ai, mi], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(wide[f"{prefix}_control_progress_at_alarm"], np.where(q >= 0, 10 * q / wide.actual_action_steps, np.nan), equal_nan=True)
    for row in tables["pooled_metrics"].itertuples():
        ai = int(np.flatnonzero(np.isclose(ALPHAS, row.alpha))[0])
        q = data["first"][KINDS.index(row.calibration), ai, METHODS.index(row.method)]
        for key, expected in counts(y, q).items():
            np.testing.assert_allclose(getattr(row, key), expected)
    keys = ["level", "cohort", "suite", "task", "method", "alpha", "failure"]
    for name, denominator in (("alarm_query_distribution", "episodes"), ("alarm_query_bins", "episodes"), ("alarm_progress_distribution", "alarms")):
        for _, part in tables[name].groupby(keys):
            assert int(part["count"].sum()) == int(part[denominator].iloc[0])
    for row in tables["paired_with_knn20"].itertuples():
        assert row.both + row.removed_from_knn20 + row.new_vs_knn20 + row.neither == row.episodes
        assert row.earlier + row.later + row.same_query == row.both
    for row in tables["physical_summary"].itertuples():
        assert row.before + row.at + row.after + row.missed == row.event_episodes
    error_rows = tables["candidate_errors_alpha002"]
    candidate_mi = METHODS.index(summary["exploratory_candidate_method"])
    strict_ai = int(np.flatnonzero(np.isclose(ALPHAS, .02))[0])
    q = data["first"][1, strict_ai, candidate_mi]
    np.testing.assert_array_equal(error_rows.global_row, np.flatnonzero((q >= 0) != y))
    assert error_rows.loc[error_rows.error.eq("false_positive"), "failure"].eq(False).all()
    assert error_rows.loc[error_rows.error.eq("false_negative"), "failure"].eq(True).all()
    assert digest(Path(__file__)) == summary["reporter_sha256"]
    for name, expected in summary["inputs"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in summary["artifacts"].items():
        assert digest(output / name) == expected, name
    artifacts = dict(summary["artifacts"])
    for name in ("REPORT_ZH.md", "kmeans_cluster_sweep.png", "kmeans_cluster_sweep.pdf", "kmeans_precision_recall.png",
                 "kmeans_precision_recall.pdf", "evaluation_summary.json"):
        artifacts[name] = digest(output / name)
    from PIL import Image
    for name in ("kmeans_cluster_sweep.png", "kmeans_precision_recall.png"):
        with Image.open(output / name) as im:
            assert min(im.size) >= 800 and np.asarray(im.convert("RGB")).std() > 15
    write_json(output / "report_verification.json", dict(passed=True, unique_rows=len(wide), methods=len(METHODS),
        exported_trajectory_decisions=len(wide) * len(ALPHAS) * len(summary["selected_export_methods"]),
        source_sha256=digest(Path(__file__)), inputs=summary["inputs"], artifacts=artifacts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    with threadpool_limits(limits=1):
        tables, summary = evaluate(output)
        make_figures(output, tables, summary)
        write_report(output, tables, summary)
        verify_report(output, tables, summary)
    print(grouped(tables["pooled_metrics"]).to_string(index=False), flush=True)
    print("K-MEANS REPORT VERIFIED", flush=True)


if __name__ == "__main__":
    main()
