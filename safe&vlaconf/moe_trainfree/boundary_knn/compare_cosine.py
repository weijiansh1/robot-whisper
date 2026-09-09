"""Evaluate the sealed cosine ablation and retain complete alarm distributions."""

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

from cosine_knn import HERE, ROOT, METHODS, CASE_ROW, DEFAULT_OUTPUT, load_npz
from core import digest, trajectory_peak, write_json
from evaluate import attach_labels, alarm_metrics, ranking, PHYSICAL

QUERY_BINS = ((-1, -1, "no_alarm"), (0, 6, "q0-6"), (7, 7, "q7"), (8, 10, "q8-10"),
              (11, 14, "q11-14"), (15, 19, "q15-19"), (20, 29, "q20-29"),
              (30, 39, "q30-39"), (40, 51, "q40-51"))
PROGRESS_BINS = ("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%)")
COLORS = {"euclidean": "#b45f3c", "cosine": "#168579"}


def metrics(frame, first):
    return {key: value for key, value in alarm_metrics(frame, first).items()
            if key not in ("median_alarm_query", "t_det")}


def pooled_counts(frame, groups):
    out = frame.groupby(groups, as_index=False)[["episodes", "failures", "successes", "tp", "fp"]].sum()
    out["fn"], out["tn"] = out.failures - out.tp, out.successes - out.fp
    out["recall"] = out.tp / out.failures.replace(0, np.nan)
    out["fpr"] = out.fp / out.successes.replace(0, np.nan)
    out["precision"] = out.tp / (out.tp + out.fp).replace(0, np.nan)
    return out


def load_action_steps(frame, rows):
    steps, inputs = {}, {}
    for source, group in frame.iloc[rows].groupby("source"):
        path = ROOT / "VLA_MUI_HUB" / source / "client/summaries.json"
        inputs[str(path.relative_to(ROOT))] = digest(path)
        summaries = {int(row["episode_index"]): row for row in json.loads(path.read_text())}
        for row in group.itertuples():
            entry = summaries[int(row.episode)]
            count = int(entry["action_steps"])
            assert int(entry["inference_calls"]) == row.length
            assert 10 * (row.length - 1) < count <= 10 * row.length
            steps[row.Index] = count
    return steps, inputs


def distributions(decisions):
    exact, binned, progress, physical, horizons = [], [], [], [], []
    for (method, scope, failure), group in decisions.groupby(["method", "scope", "failure"]):
        meta = dict(method=method, scope=scope, failure=bool(failure))
        q = group.first_alarm_query.to_numpy()
        for query in range(-1, 52):
            exact.append(dict(meta, query=query, count=int((q == query).sum()), episodes=len(group)))
        for lo, hi, label in QUERY_BINS:
            count = int(((q >= lo) & (q <= hi)).sum())
            binned.append(dict(meta, query_bin=label, count=count, episodes=len(group), fraction=count / len(group)))
        fired = group.loc[group.alarm]
        bands = np.floor(fired.control_progress_at_alarm.to_numpy() * 4).astype(int)
        for i, label in enumerate(PROGRESS_BINS):
            count = int((bands == i).sum())
            progress.append(dict(meta, progress_bin=label, count=count, alarms=len(fired),
                fraction_of_alarms=count / len(fired) if len(fired) else np.nan))
        if failure:
            for cutoff in (7, 10, 14, 19, 29, 39, 51):
                count = int(((q >= 0) & (q <= cutoff)).sum())
                horizons.append(dict(method=method, scope=scope, cutoff_query=cutoff,
                    detected=count, failures=len(group), recall=count / len(group)))
            event = group.loc[group.drop_goal_release.notna()]
            before = event.alarm & event.first_alarm_query.lt(event.drop_goal_release)
            at = event.alarm & event.first_alarm_query.eq(event.drop_goal_release)
            after = event.alarm & event.first_alarm_query.gt(event.drop_goal_release)
            physical.append(dict(method=method, scope=scope, event_episodes=len(event),
                before=int(before.sum()), at=int(at.sum()), after=int(after.sum()),
                missed=int((~event.alarm).sum()), before_fraction=float(before.mean()) if len(event) else np.nan))
    return {"alarm_query_distribution": pd.DataFrame(exact), "alarm_query_bins": pd.DataFrame(binned),
            "alarm_progress_distribution": pd.DataFrame(progress), "physical_summary": pd.DataFrame(physical),
            "failure_detection_by_horizon": pd.DataFrame(horizons)}


def paired_decisions(decisions):
    keys = ["fold", "global_row", "suite", "task", "scope", "failure"]
    pairs = decisions.pivot(index=keys, columns="method", values="first_alarm_query").reset_index()
    assert pairs[list(METHODS)].notna().all().all()
    a, b = pairs.euclidean.ge(0), pairs.cosine.ge(0)
    pairs["alarm_change"] = np.select((a & b, a & ~b, ~a & b), ("both", "euclidean_only", "cosine_only"), default="neither")
    pairs["cosine_minus_euclidean_query"] = np.where(a & b, pairs.cosine - pairs.euclidean, np.nan)
    summary = []
    for (scope, failure), group in pairs.groupby(["scope", "failure"]):
        counts = group.alarm_change.value_counts()
        delta = group.cosine_minus_euclidean_query
        summary.append(dict(scope=scope, failure=bool(failure), episodes=len(group),
            **{key: int(counts.get(key, 0)) for key in ("both", "euclidean_only", "cosine_only", "neither")},
            cosine_earlier=int(delta.lt(0).sum()), cosine_later=int(delta.gt(0).sum()), same_query=int(delta.eq(0).sum())))
    return pairs, pd.DataFrame(summary)


def evaluate_all(parent, output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    for name, expected in manifest["artifacts"].items():
        assert digest(output / name) == expected, name
    frame = attach_labels(pd.read_csv(parent / "index.csv"))
    data_by_fold = {info["fold"]: load_npz(output / "predictions" / f"{info['fold']}.npz") for info in manifest["folds"]}
    unique_rows = np.unique(np.concatenate([data["test_rows"] for data in data_by_fold.values()]))
    action_steps, inputs = load_action_steps(frame, unique_rows)
    event_path = HERE.parent / "results/round3_safe/physical_events.csv"
    events = pd.read_csv(event_path).set_index("global_row")
    inputs[str(event_path.relative_to(ROOT))] = digest(event_path)
    inputs[str((PHYSICAL / "episodes.csv").relative_to(ROOT))] = digest(PHYSICAL / "episodes.csv")
    inputs[str((parent / "index.csv").relative_to(ROOT))] = digest(parent / "index.csv")
    common = frame.loc[frame.run_id.eq("right-50x8b-20260903")].groupby("task").length.min().to_dict()
    alarms, task_alarms, ranks, decisions = [], [], [], []
    for info in manifest["folds"]:
        data = data_by_fold[info["fold"]]
        part = frame.iloc[data["test_rows"]].reset_index(drop=True)
        part["global_row"] = data["test_rows"]
        part["scope"] = np.where(data["test_unseen"], "unseen", "seen")
        alpha_i = int(np.flatnonzero(np.isclose(data["alphas"], .05))[0])
        for method_i, method in enumerate(METHODS):
            meta = dict(info, method=method)
            raw = data["scores"][method_i]
            view_scores = {"full": trajectory_peak(raw), "q14": trajectory_peak(raw[:, :15]),
                "common_horizon": trajectory_peak(np.where(np.arange(52)[None] < part.task.map(common).to_numpy()[:, None], raw, np.nan))}
            for view, values in view_scores.items():
                for scope in ("seen", "unseen"):
                    selected = part.scope.eq(scope).to_numpy()
                    ranks.append(dict(meta, scope=scope, view=view,
                        **ranking(part.loc[selected].reset_index(drop=True), values[selected])))
            for kind_i, kind in enumerate(("episode", "task_init")):
                for ai, alpha in enumerate(data["alphas"]):
                    first = data["first"][kind_i, ai, method_i]
                    assert ((first == -1) | ((first >= 7) & (first < part.length.to_numpy()))).all()
                    for scope in ("seen", "unseen"):
                        selected = part.scope.eq(scope).to_numpy()
                        alarms.append(dict(meta, scope=scope, calibration=kind, alpha=float(alpha),
                            **metrics(part.loc[selected], first[selected])))
            first = data["first"][1, alpha_i, method_i]
            for (scope, task), positions in part.groupby(["scope", "task"]).indices.items():
                task_alarms.append(dict(meta, scope=scope, task=task,
                    **metrics(part.iloc[positions], first[positions])))
            record = part[["global_row", "run_id", "suite", "task", "episode", "init_state_id", "noise_seed",
                           "length", "failure", "scope"]].copy()
            record["fold"], record["method"] = info["fold"], method
            record["first_alarm_query"], record["alarm"] = first, first >= 0
            record["threshold"] = float(data["thresholds"][1, alpha_i, method_i])
            record["peak_score"] = trajectory_peak(raw)
            record["actual_action_steps"] = record.global_row.map(action_steps)
            record["executed_actions_before_alarm"] = np.where(first >= 0, 10 * first, -1)
            record["control_progress_at_alarm"] = np.where(first >= 0, 10 * first / record.actual_action_steps, np.nan)
            assert record.loc[record.alarm, "control_progress_at_alarm"].between(0, 1, inclusive="left").all()
            for name in ("drop_goal_release", "failed_goal_release"):
                record[name] = record.global_row.map(events[name])
            record["release_lead_queries"] = np.where(record.alarm, record.drop_goal_release - first, np.nan)
            decisions.append(record)
        print(f"EVALUATED {info['fold']}", flush=True)
    decisions = pd.concat(decisions, ignore_index=True)
    tables = {"alarm_metrics": pd.DataFrame(alarms), "task_alarm_metrics": pd.DataFrame(task_alarms),
              "ranking_metrics": pd.DataFrame(ranks), "episode_decisions": decisions}
    tables["pooled_metrics"] = pooled_counts(tables["alarm_metrics"], ["method", "scope", "calibration", "alpha"])
    tables["suite_metrics"] = pooled_counts(tables["alarm_metrics"], ["suite", "method", "scope", "calibration", "alpha"])
    tables["task_metrics"] = pooled_counts(tables["task_alarm_metrics"], ["suite", "method", "scope", "task"])
    tables.update(distributions(decisions))
    tables["paired_decisions"], tables["paired_summary"] = paired_decisions(decisions)
    tables["physical_timing"] = decisions.loc[decisions.failure & decisions.drop_goal_release.notna()].copy()
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    summary = dict(test_appearances=int(len(decisions) // 2), unique_test_episodes=len(unique_rows),
        unique_test_failures=int(frame.iloc[unique_rows].failure.sum()),
        historical_data=True, nominal_fpr_guarantee_on_unseen_tasks=False,
        primary_alpha=.05, primary_calibration="task_init", inputs=inputs,
        sealed_manifest_sha256=digest(output / "sealed_manifest.json"), evaluator_sha256=digest(Path(__file__)),
        rows={name: len(table) for name, table in tables.items()},
        artifacts={f"{name}.csv": digest(output / f"{name}.csv") for name in tables})
    write_json(output / "evaluation_summary.json", summary)
    return tables, summary


def make_figures(output, tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    primary = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("task_init") & np.isclose(x.alpha, .05)]
    binned = tables["alarm_query_bins"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for ax, field, title in ((axes[0, 0], "fpr", "Successful trajectories: false alarm rate"),
                             (axes[0, 1], "recall", "Failed trajectories: detected fraction")):
        for j, method in enumerate(METHODS):
            values = primary.loc[primary.method.eq(method)].set_index("scope").loc[["seen", "unseen"], field]
            ax.bar(np.arange(2) + (j - .5) * .34, values, width=.32, color=COLORS[method], label=method.title())
        ax.set_xticks([0, 1], ["Seen tasks", "Unseen tasks"])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.legend(frameon=False)
    for ax, failure, title in ((axes[1, 0], True, "Unseen failures: first alarm distribution"),
                                (axes[1, 1], False, "Unseen successes: false alarm distribution")):
        labels = [label for _, _, label in QUERY_BINS if label != "q0-6" and (failure or label != "no_alarm")]
        for j, method in enumerate(METHODS):
            part = binned.loc[binned.method.eq(method) & binned.scope.eq("unseen") & binned.failure.eq(failure)]
            values = part.set_index("query_bin").loc[labels, "count"]
            ax.bar(np.arange(len(labels)) + (j - .5) * .38, values, width=.36, color=COLORS[method], label=method.title())
        ax.set_xticks(np.arange(len(labels)), [x.replace("no_alarm", "No\nalarm").replace("q", "") for x in labels])
        ax.set_xlabel("First query (zero-based)")
        ax.set_ylabel("Trajectory appearances")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("Frozen 10D features and reference chunks; separately calibrated at alpha=5%", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cosine_comparison.{suffix}", dpi=170)
    plt.close(fig)
    case = pd.read_csv(output / "long_episode_223_queries.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 6), sharex=True, layout="constrained", height_ratios=(2, 1))
    for method in METHODS:
        hit = np.flatnonzero(case[f"{method}_current_exceeds"].to_numpy())
        label = f"{method.title()} (first q{hit[0]})" if len(hit) else f"{method.title()} (no alarm)"
        axes[0].plot(case["query"], case[f"{method}_ratio"], color=COLORS[method], lw=2, marker=".", label=label)
    axes[0].axhline(1, color="#454545", lw=1, ls="--", label="Calibrated threshold")
    axes[0].set_ylabel("Distance / own threshold")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_title("Same recorded trajectory: LIBERO-Long, episode 223 (52 chunks)", fontsize=12)
    axes[1].plot(case["query"], case.standardized_vector_norm, color="#4f6281", lw=1.8)
    axes[1].set_ylabel("10D vector norm")
    axes[1].set_xlabel("Query / action chunk (10 executed actions per complete chunk)")
    for ax in axes:
        ax.axvspan(0, 6, color="#e3e3e3", alpha=.55)
        ax.set_xlim(0, 51)
        ax.set_xticks(np.arange(0, 52, 3))
        ax.grid(alpha=.18)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"long_episode_223_comparison.{suffix}", dpi=170)
    plt.close(fig)


def write_report(output, tables, summary):
    pooled = tables["pooled_metrics"]
    primary = pooled.loc[pooled.calibration.eq("task_init") & np.isclose(pooled.alpha, .05)]
    unseen = primary.loc[primary.scope.eq("unseen")].set_index("method")
    a, b = unseen.loc["euclidean"], unseen.loc["cosine"]
    lines = ["# 10 维 kNN：欧氏距离与余弦距离对照", "",
        f"主工作点下，未见任务的误报由欧氏的 {a.fpr:.2%} 变为余弦的 {b.fpr:.2%}，"
        f"召回由 {a.recall:.2%} 变为 {b.recall:.2%}。",
        "本轮纯余弦大幅降低误报，同时丢失大部分失败检出，不能直接替换现有主分数。", "",
        "固定特征、参考 chunk 身份、k=20、12 折与校准数据，仅替换距离；分别重新校准阈值。",
        "余弦在原 median/MAD 中心化、标准化坐标上计算，无训练、无新增 rollout。",
        f"共 {summary['test_appearances']:,} 次测试轨迹出现，涉及 {summary['unique_test_episodes']:,} 条独立轨迹。",
        "同一轨迹可能跨折重复，以下计数是出现次数。主工作点是 task/init 分组校准 alpha=5%。", "",
        "## 整体结果", "", "| 范围 | 距离 | 误报 / 成功 | 误报率 | 检出 / 失败 | 召回率 |",
        "|---|---|---:|---:|---:|---:|"]
    for scope in ("seen", "unseen"):
        for method in METHODS:
            r = primary.loc[primary.scope.eq(scope) & primary.method.eq(method)].iloc[0]
            lines.append(f"| {scope} | {method} | {int(r.fp)}/{int(r.successes)} | {r.fpr:.2%} | {int(r.tp)}/{int(r.failures)} | {r.recall:.2%} |")
    lines += ["", "同样的校准预算不等于同样的实测误报率；未见任务没有 5% 误报率保证。",
        "完整的 1/3/5/10/15/20% 与两种校准结果见 [pooled_metrics.csv](pooled_metrics.csv)。", "",
        "![整体与报警时点](cosine_comparison.png)", "", "## 阈值与排序的检查", "",
        "以下均为预先固定的工作点，没有按 B 的结果搜索新阈值。", "",
        "| 校准 alpha | 欧氏实测误报率 | 欧氏召回 | 余弦实测误报率 | 余弦召回 |",
        "|---|---:|---:|---:|---:|"]
    grid = pooled.loc[pooled.scope.eq("unseen") & pooled.calibration.eq("task_init")]
    for alpha in sorted(grid.alpha.unique()):
        rows = grid.loc[np.isclose(grid.alpha, alpha)].set_index("method")
        a, b = rows.loc["euclidean"], rows.loc["cosine"]
        lines.append(f"| {alpha:.0%} | {a.fpr:.2%} | {a.recall:.2%} | {b.fpr:.2%} | {b.recall:.2%} |")
    a = grid.loc[grid.method.eq("euclidean") & np.isclose(grid.alpha, .03)].iloc[0]
    b = grid.loc[grid.method.eq("cosine") & np.isclose(grid.alpha, .10)].iloc[0]
    lines += ["", f"描述性地比较误报率接近的已有工作点：欧氏 alpha=3% 的实测误报率 {a.fpr:.2%}、"
        f"召回 {a.recall:.2%}；余弦 alpha=10% 的实测误报率 {b.fpr:.2%}、召回 {b.recall:.2%}。",
        "这不是在 B 上选择部署参数，也不声称两者误报率严格相等。", "",
        "不依赖报警阈值的排序指标也下降。下表先计算每折的 task-macro AUC，再对 12 折等权平均；",
        "只看未见任务。全程峰值会受成功 / 失败轨迹长度差异影响，所以同时报告每个任务统一观察长度的结果。", "",
        "| 排序视角 | 欧氏 task-macro AUC | 余弦 task-macro AUC |", "|---|---:|---:|"]
    ranks = tables["ranking_metrics"].loc[lambda x: x.scope.eq("unseen")].groupby(["method", "view"]).task_macro_auc.mean()
    for view, label in (("full", "全程峰值"), ("common_horizon", "同任务统一观察长度")):
        lines.append(f"| {label} | {ranks.loc[('euclidean', view)]:.4f} | {ranks.loc[('cosine', view)]:.4f} |")
    lines += ["", "因此现有证据不支持把性能损失只归因于 5% 工作点的阈值。", "",
        "## 未见任务的报警分布", "",
        "q 为从零开始的推理 / chunk 编号，q33 表示本次动作执行前已完成 330 个动作步。",
        "下表保留未报警数量，不用中位数概括时间分布。", "",
        "| 首次报警区间 | 欧氏：失败检出 | 余弦：失败检出 | 欧氏：成功误报 | 余弦：成功误报 |",
        "|---|---:|---:|---:|---:|"]
    binned = tables["alarm_query_bins"].loc[lambda x: x.scope.eq("unseen")].set_index(["query_bin", "method", "failure"])
    for _, _, label in QUERY_BINS:
        if label == "q0-6":
            continue
        counts = [int(binned.loc[(label, method, failed), "count"]) for failed in (True, False) for method in METHODS]
        lines.append("| " + " | ".join(["未报警" if label == "no_alarm" else label, *map(str, counts)]) + " |")
    lines += ["", "成功轨迹的‘未报警’是正确阴性，失败轨迹的‘未报警’是漏检。", "",
        "| 误报时已执行动作比例 | 欧氏次数 | 余弦次数 |", "|---|---:|---:|"]
    progress = tables["alarm_progress_distribution"].loc[lambda x: x.scope.eq("unseen") & ~x.failure]
    progress = progress.set_index(["progress_bin", "method"])
    for label in PROGRESS_BINS:
        lines.append(f"| {label} | {int(progress.loc[(label, 'euclidean'), 'count'])} | {int(progress.loc[(label, 'cosine'), 'count'])} |")
    lines += ["", "动作比例使用 `10*q/实际 action_steps`，只用于事后评价，不输入检测器。", "",
        "## 成对变化", "", "| 范围与结果 | 仅欧氏报警 | 仅余弦报警 | 两者报警 | 两者不报警 | 共同检出中余弦更早 / 更晚 / 同时 |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in tables["paired_summary"].itertuples():
        outcome = "失败" if r.failure else "成功"
        lines.append(f"| {r.scope} {outcome} | {r.euclidean_only} | {r.cosine_only} | {r.both} | {r.neither} | {r.cosine_earlier}/{r.cosine_later}/{r.same_query} |")
    tasks = tables["task_metrics"].loc[lambda x: x.scope.eq("unseen")]
    selected_tasks = set()
    for method in METHODS:
        selected_tasks.update(tasks.loc[tasks.method.eq(method)].nlargest(5, "fp").task)
    selected = tasks.loc[tasks.task.isin(selected_tasks)]
    lines += ["", "## 误报较多的任务", "", "两种方法分别按误报数量取前五，展示其并集；完整任务表包含全部任务。", "",
        "| 任务 | 欧氏误报 / 成功 | 余弦误报 / 成功 | 欧氏检出 / 失败 | 余弦检出 / 失败 |", "|---|---:|---:|---:|---:|"]
    order = selected.groupby("task").fp.max().sort_values(ascending=False).index
    for task in order:
        a = selected.loc[selected.task.eq(task) & selected.method.eq("euclidean")].iloc[0]
        b = selected.loc[selected.task.eq(task) & selected.method.eq("cosine")].iloc[0]
        lines.append(f"| {task} | {int(a.fp)}/{int(a.successes)} | {int(b.fp)}/{int(b.successes)} | {int(a.tp)}/{int(a.failures)} | {int(b.tp)}/{int(b.failures)} |")
    lines += ["", "## 释放事件前的检出", "", "仅统计存在目标物体释放标记的失败轨迹。释放是事后物理代理事件，不等于不可逆失败起点。", "",
        "| 范围 | 距离 | 事件数 | 事件前 | 同一 query | 事件后 | 未检出 |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in tables["physical_summary"].itertuples():
        lines.append(f"| {r.scope} | {r.method} | {r.event_episodes} | {r.before} | {r.at} | {r.after} | {r.missed} |")
    case = tables["episode_decisions"].loc[lambda x: x.global_row.eq(CASE_ROW) & x.fold.eq("libero_long_20260907")]
    lines += ["", "## 同一条 52-chunk 轨迹", "", "沿用此前展示的 Long episode 223，本轮运行前固定。", "",
        "| 距离 | 阈值 | 首次报警 query |", "|---|---:|---:|"]
    for r in case.itertuples():
        lines.append(f"| {r.method} | {r.threshold:.8f} | {r.first_alarm_query if r.first_alarm_query >= 0 else '未报警'} |")
    query = pd.read_csv(output / "long_episode_223_queries.csv").set_index("query")
    before, after = query.loc[28], query.loc[33]
    lines += ["", "![逐 chunk 对照](long_episode_223_comparison.png)", "",
        "上图分数分别除以各自阈值；比值不是失败概率。下图是标准化 10D 向量范数。",
        f"这条轨迹 q28 到 q33 的向量范数从 {before.standardized_vector_norm:.2f} 升至 {after.standardized_vector_norm:.2f}，"
        f"欧氏 kNN 分数从 {before.euclidean_score:.4f} 升至 {after.euclidean_score:.4f}，"
        f"余弦 kNN 分数却从 {before.cosine_score:.4f} 降至 {after.cosine_score:.4f}。",
        "也就是说，这些后期点仍能在成功参考中找到方向接近的邻居。这是特征空间的观察，"
        "不能据此断言幅度变化在物理上导致了失败。",
        "余弦忽略径向幅度，可能减少幅度导致的误报，也可能漏掉主要表现为幅度变化的异常。",
        "两种方法仍使用同一批历史成功 chunk，任务 / 阶段不匹配的问题并没有由距离替换自动解决。", "",
        "## 核验与数据", ""]
    verification = json.loads((output / "verification.json").read_text())
    norms = pd.read_csv(output / "norm_audit.csv")
    lines += [f"- 独立核验 {verification['hash_checks']} 个哈希、{verification['calibration_checks']} 个校准与报警数组、"
              f"{len(verification['direct_distance_checks'])} 个直接距离排序样本，以及四个套件的原始路由重放。",
        f"- 有效 query 最小向量范数 {norms.query_norm_min.min():.6g}；参考点最小范数 {norms.bank_norm_min.min():.6g}。没有触发近零向量保护。",
        "- [逐轨迹结果](episode_decisions.csv) / [成对报警变化](paired_decisions.csv) / [完整 query 分布](alarm_query_distribution.csv)。",
        "- [逐任务结果](task_metrics.csv) / [分套件结果](suite_metrics.csv) / [排序指标](ranking_metrics.csv)。",
        "- [事件时序](physical_timing.csv) / [截至各 query 的失败检出](failure_detection_by_horizon.csv)。",
        "- [52-chunk 分数](long_episode_223_queries.csv) / [两种距离的近邻身份](long_episode_223_neighbors.csv)。",
        "- 每折 `predictions/*.npz` 保留全部测试与校准 chunk 分数、阈值和首次报警；`profiles/*.npz` 保留参考点与尺度。",
        "- [评分清单](sealed_manifest.json) / [独立核验](verification.json) / [评估来源](evaluation_summary.json)。", "",
        "```bash", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/cosine_knn.py' score --output /tmp/himoe-cosine-reproduction",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/cosine_knn.py' verify --output /tmp/himoe-cosine-reproduction",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/compare_cosine.py' --output /tmp/himoe-cosine-reproduction", "```", ""]
    (output / "REPORT_ZH.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=HERE.parent / "results/round5_knn")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "verification.json").exists():
        raise ValueError("verify the sealed predictions before producing the report")
    with threadpool_limits(limits=1):
        tables, summary = evaluate_all(args.parent.resolve(), output)
        make_figures(output, tables)
        write_report(output, tables, summary)
    write_json(output / "report_artifacts.json", dict(source_sha256=digest(Path(__file__)),
        artifacts={str(path.relative_to(output)): digest(path) for path in sorted(output.iterdir())
                   if path.suffix in (".png", ".pdf", ".md")}))
    print("COSINE COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
