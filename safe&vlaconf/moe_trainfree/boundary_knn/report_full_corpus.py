"""Evaluate sealed full-corpus predictions and export unique trajectory results."""

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

from full_corpus_knn import OUTPUT, ROOT, HERE, PARENT, METHODS, KINDS, load_npz
from core import RUNS, digest, trajectory_peak, write_json
from compare_cosine import load_action_steps, QUERY_BINS, PROGRESS_BINS
from evaluate import attach_labels, PHYSICAL

NAMES = {
    "euclidean": "Euclidean / Euclidean",
    "cosine": "Cosine / Cosine",
    "cosine_neighbors_euclidean_score": "Cosine / Euclidean",
    "euclidean_neighbors_cosine_score": "Euclidean / Cosine",
    "cosine_without_center": "Cosine, no centering",
    "norm_only": "Vector norm",
}
ZH_NAMES = {
    "euclidean": "欧氏近邻 + 欧氏打分",
    "cosine": "余弦近邻 + 余弦打分",
    "cosine_neighbors_euclidean_score": "余弦近邻 + 欧氏打分",
    "euclidean_neighbors_cosine_score": "欧氏近邻 + 余弦打分",
    "cosine_without_center": "余弦，不做中心化",
    "norm_only": "只用向量范数",
}
COLORS = ("#b45f3c", "#168579", "#6482aa", "#8c75a9", "#a49a45", "#62696b")


def counts(failure, first):
    alarm = first >= 0
    p, n = int(failure.sum()), int((~failure).sum())
    tp, fp = int((failure & alarm).sum()), int((~failure & alarm).sum())
    return dict(episodes=p + n, failures=p, successes=n, tp=tp, fn=p - tp, fp=fp, tn=n - fp,
        recall=tp / p if p else np.nan, fpr=fp / n if n else np.nan,
        precision=tp / (tp + fp) if tp + fp else np.nan)


def populations(frame):
    yield dict(level="all", cohort="all", suite="all", task="all"), np.arange(len(frame))
    for level, keys in (("cohort", ["cohort"]), ("suite", ["suite"]), ("cohort_suite", ["cohort", "suite"]),
                        ("task", ["suite", "task"]), ("cohort_task", ["cohort", "suite", "task"])):
        for values, rows in frame.groupby(keys, sort=True).indices.items():
            values = values if isinstance(values, tuple) else (values,)
            yield dict(level=level, **(dict(cohort="all", suite="all", task="all") | dict(zip(keys, values)))), rows


def evaluate(output):
    manifest = json.loads((output / "sealed_manifest.json").read_text())
    verification = json.loads((output / "verification.json").read_text())
    assert verification["passed"] and verification["sealed_manifest_sha256"] == digest(output / "sealed_manifest.json")
    for category, root in (("sources", ROOT), ("inputs", ROOT), ("artifacts", output)):
        for name, expected in manifest[category].items():
            assert digest(root / name) == expected, name
    raw_index = pd.read_csv(output / "index.csv")
    frame = attach_labels(raw_index)
    pd.testing.assert_frame_equal(frame[raw_index.columns], raw_index)
    frame["global_row"] = np.arange(len(frame))
    frame["cohort"] = frame.run_id.map({RUNS[0]: "A", RUNS[1]: "B"})
    assert frame.cohort.notna().all()
    steps, inputs = load_action_steps(frame, np.arange(len(frame)))
    frame["actual_action_steps"] = frame.global_row.map(steps)
    events_path = HERE.parent / "results/round3_safe/physical_events.csv"
    events = pd.read_csv(events_path).set_index("global_row")
    assert events.index.is_unique and frame.loc[events.index, "cohort"].eq("B").all()
    assert frame.loc[events.index, "failure"].all()
    for key in ("drop_goal_release", "failed_goal_release"):
        frame[key] = frame.global_row.map(events[key])
    inputs.update({str(path.relative_to(ROOT)): digest(path) for path in (events_path, PHYSICAL / "episodes.csv")})
    n, m = len(frame), len(METHODS)
    scores = np.full((m, n, 52), np.nan, np.float32)
    first = np.full((2, m, n), -2, np.int16)
    thresholds = np.full((2, m, n), np.nan, np.float64)
    ownership = np.zeros(n, np.int16)
    frame["fold"] = ""
    threshold_records = []
    for info in manifest["folds"]:
        data = load_npz(output / "predictions" / f"{info['fold']}.npz")
        rows = data["test_rows"]
        ownership[rows] += 1
        frame.loc[rows, "fold"] = info["fold"]
        scores[:, rows] = data["scores"]
        first[:, :, rows] = data["first"]
        thresholds[:, :, rows] = data["thresholds"][:, :, None]
        historical = np.r_[data["reference_rows"], data["calibration_rows"]]
        np.testing.assert_array_equal(frame.loc[historical, "failure"].to_numpy(), np.r_[data["reference_labels"], data["calibration_labels"]].astype(bool))
        cal = pd.read_csv(output / "calibration" / f"{info['fold']}.csv")
        cal["fold"], cal["suite"] = info["fold"], info["suite"]
        threshold_records.append(cal)
    np.testing.assert_array_equal(ownership, 1)
    assert n == 32000 and int(frame.failure.sum()) == 1096
    frame["scorable_queries"] = np.isfinite(scores[0]).sum(1)
    y = frame.failure.to_numpy()
    metrics, query_exact, query_bins, progress, horizons, physical, reasons, pairs = [], [], [], [], [], [], [], []
    for meta, rows in populations(frame):
        for ki, kind in enumerate(KINDS):
            for mi, method in enumerate(METHODS):
                metrics.append(dict(meta, calibration=kind, method=method, alpha=.05, **counts(y[rows], first[ki, mi, rows])))
        if meta["level"] not in ("all", "cohort", "suite"):
            continue
        for mi, method in enumerate(METHODS):
            q = first[1, mi, rows]
            for failure in (False, True):
                selected = y[rows] == failure
                values = q[selected]
                info = dict(meta, method=method, failure=failure, episodes=int(selected.sum()))
                fired = values >= 0
                for query in range(-1, 52):
                    query_exact.append(dict(info, query=query, count=int((values == query).sum())))
                for lo, hi, label in QUERY_BINS:
                    count = int(((values >= lo) & (values <= hi)).sum())
                    query_bins.append(dict(info, query_bin=label, count=count, fraction=count / len(values) if len(values) else np.nan))
                phase = 10 * values[fired] / frame.loc[rows[selected][fired], "actual_action_steps"].to_numpy()
                assert ((phase >= 0) & (phase < 1)).all()
                for i, label in enumerate(PROGRESS_BINS):
                    count = int(((phase >= i / 4) & (phase < (i + 1) / 4)).sum())
                    progress.append(dict(info, progress_bin=label, count=count, alarms=int(fired.sum()),
                        fraction_of_alarms=count / int(fired.sum()) if fired.any() else np.nan))
                if failure:
                    for cutoff in (7, 10, 14, 19, 29, 39, 51):
                        detected = int(((values >= 0) & (values <= cutoff)).sum())
                        horizons.append(dict(info, cutoff_query=cutoff, detected=detected, recall=detected / len(values) if len(values) else np.nan))
            event = frame.loc[rows, "drop_goal_release"].to_numpy()
            eligible = y[rows] & np.isfinite(event)
            physical.append(dict(meta, method=method, event_episodes=int(eligible.sum()),
                before=int((eligible & (q >= 0) & (q < event)).sum()), at=int((eligible & (q >= 0) & (q == event)).sum()),
                after=int((eligible & (q >= 0) & (q > event)).sum()), missed=int((eligible & (q < 0)).sum())))
        for failure in (False, True):
            selected = rows[y[rows] == failure]
            e, c = first[1, 0, selected], first[1, 1, selected]
            both = (e >= 0) & (c >= 0)
            pairs.append(dict(meta, failure=failure, episodes=len(selected), both=int(both.sum()),
                euclidean_only=int(((e >= 0) & (c < 0)).sum()), cosine_only=int(((e < 0) & (c >= 0)).sum()),
                neither=int(((e < 0) & (c < 0)).sum()), cosine_earlier=int((both & (c < e)).sum()),
                cosine_later=int((both & (c > e)).sum()), same_query=int((both & (c == e)).sum())))
    for reason, part in frame.loc[frame.failure].groupby("primary_failure_reason"):
        for mi, method in enumerate(METHODS):
            values = first[1, mi, part.index]
            reasons.append(dict(reason=reason, method=method, failures=len(part), detected=int((values >= 0).sum()),
                missed=int((values < 0).sum()), recall=float((values >= 0).mean())))
    wide = frame.copy()
    for mi, method in enumerate(METHODS):
        q = first[1, mi]
        wide[f"{method}_first_alarm_query"] = q
        wide[f"{method}_threshold"] = thresholds[1, mi]
        wide[f"{method}_peak_score"] = np.where(wide.scorable_queries.gt(0), trajectory_peak(scores[mi]), np.nan)
        wide[f"{method}_executed_actions_before_alarm"] = np.where(q >= 0, 10 * q, -1)
        wide[f"{method}_control_progress_at_alarm"] = np.where(q >= 0, 10 * q / frame.actual_action_steps, np.nan)
        wide[f"{method}_episode_calibration_first_alarm_query"] = first[0, mi]
        wide[f"{method}_episode_calibration_threshold"] = thresholds[0, mi]
    all_metrics = pd.DataFrame(metrics)
    tables = dict(trajectory_results=wide, metrics=all_metrics,
        pooled_metrics=all_metrics.loc[all_metrics.level.eq("all")],
        cohort_metrics=all_metrics.loc[all_metrics.level.eq("cohort")],
        suite_metrics=all_metrics.loc[all_metrics.level.eq("suite")],
        task_metrics=all_metrics.loc[all_metrics.level.eq("task")],
        thresholds=pd.concat(threshold_records, ignore_index=True),
        alarm_query_distribution=pd.DataFrame(query_exact), alarm_query_bins=pd.DataFrame(query_bins),
        alarm_progress_distribution=pd.DataFrame(progress), failure_detection_by_horizon=pd.DataFrame(horizons),
        physical_summary=pd.DataFrame(physical), failure_reason_metrics=pd.DataFrame(reasons), paired_summary=pd.DataFrame(pairs))
    tables["physical_timing"] = wide.loc[wide.failure & wide.drop_goal_release.notna()].copy()
    corpus = []
    for meta, rows in populations(frame):
        corpus.append(dict(meta, episodes=len(rows), failures=int(y[rows].sum()), successes=int((~y[rows]).sum()),
            valid_queries=int(wide.loc[rows, "scorable_queries"].sum()),
            unscorable_episodes=int(wide.loc[rows, "scorable_queries"].eq(0).sum()),
            unscorable_failures=int((wide.loc[rows, "scorable_queries"].eq(0) & wide.loc[rows, "failure"]).sum())))
    tables["corpus_coverage"] = pd.DataFrame(corpus)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    np.savez_compressed(output / "all_test_scores.npz", global_rows=np.arange(n), methods=np.asarray(METHODS),
        scores=scores, first=first, thresholds=thresholds, calibration_kinds=np.asarray(KINDS))
    summary = dict(unique_test_trajectories=n, failures=int(y.sum()), successes=int((~y).sum()),
        valid_test_chunks=int(np.isfinite(scores[0]).sum()), unscorable_episodes=int(frame.scorable_queries.eq(0).sum()),
        test_task_count=int(frame.task.nunique()), fold_count=len(manifest["folds"]),
        primary_calibration="task_init", alpha=.05, no_training=True, new_rollouts=False,
        physical_events_available_for="B only; drop-goal-release subset",
        inputs=inputs, sealed_manifest_sha256=digest(output / "sealed_manifest.json"),
        verification_sha256=digest(output / "verification.json"), evaluator_sha256=digest(Path(__file__)),
        rows={name: len(table) for name, table in tables.items()},
        artifacts={f"{name}.csv": digest(output / f"{name}.csv") for name in tables} |
                  {"all_test_scores.npz": digest(output / "all_test_scores.npz")})
    write_json(output / "evaluation_summary.json", summary)
    return tables, summary


def make_figures(output, tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    main = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("task_init")].set_index("method").loc[list(METHODS)]
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), layout="constrained")
    for ax, field, count, total, title in ((axes[0, 0], "recall", "tp", "failures", "Failure detection"),
                                         (axes[0, 1], "fpr", "fp", "successes", "False alarms on successful trajectories")):
        values = main[field].to_numpy()
        ax.barh(np.arange(len(METHODS)), values, color=COLORS, height=.65)
        ax.set_yticks(np.arange(len(METHODS)), [NAMES[m] for m in METHODS] if field == "recall" else [])
        ax.invert_yaxis()
        ax.set_xlim(0, max(.01, values.max()) * 1.55)
        if field == "recall":
            ax.set_xticks(np.linspace(0, 1, 6))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        for i, row in enumerate(main.itertuples()):
            ax.text(values[i] + ax.get_xlim()[1] * .015, i,
                f"{values[i]:.1%}  ({getattr(row, count):,}/{getattr(row, total):,})", va="center", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    suites = tables["suite_metrics"].loc[lambda x: x.calibration.eq("task_init")]
    order = sorted(suites.suite.unique())
    for ax, field, title in ((axes[1, 0], "recall", "Detection by held-out task suite"),
                             (axes[1, 1], "fpr", "False alarms by held-out task suite")):
        for i, method in enumerate(METHODS[:2]):
            part = suites.loc[suites.method.eq(method)].set_index("suite").loc[order]
            bars = ax.bar(np.arange(4) + (i - .5) * .36, part[field], .34, label=method.title(), color=COLORS[i])
            ax.bar_label(bars, labels=[f"{v:.1%}" for v in part[field]], padding=3, fontsize=9)
        ax.set_xticks(np.arange(4), [name.removeprefix("libero_").title() for name in order])
        ax.set_ylim(0, max(.01, suites.loc[suites.method.isin(METHODS[:2]), field].max()) * 1.22)
        if field == "recall":
            ax.set_ylim(0, 1.18)
            ax.set_yticks(np.linspace(0, 1, 6))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, fontsize=12)
        ax.legend(frameon=False, loc="upper center", ncol=2)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle("32,000 unique trajectories | entire task held out | grouped calibration at nominal 5%", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"full_corpus_comparison.{suffix}", dpi=170)
    plt.close(fig)
    chosen = ("euclidean", "cosine", "cosine_neighbors_euclidean_score", "norm_only")
    color_by_method = dict(zip(METHODS, COLORS))
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), layout="constrained")
    for column, failure in enumerate((True, False)):
        binned = tables["alarm_query_bins"].loc[lambda x: x.level.eq("all") & x.failure.eq(failure)]
        labels = [label for _, _, label in QUERY_BINS if label != "q0-6" and (failure or label != "no_alarm")]
        for i, method in enumerate(chosen):
            values = binned.loc[binned.method.eq(method)].set_index("query_bin").loc[labels, "count"]
            axes[0, column].bar(np.arange(len(labels)) + (i - 1.5) * .21, values, .20,
                color=color_by_method[method], label=NAMES[method])
        axes[0, column].set_xticks(np.arange(len(labels)), [label.replace("no_alarm", "No\nalarm").removeprefix("q") for label in labels])
        axes[0, column].set_xlabel("First query (zero-based)")
        axes[0, column].set_ylabel("Unique trajectories")
        axes[0, column].set_title("Failed trajectories" if failure else "Successful trajectories: false alarms", fontsize=12)
        progress = tables["alarm_progress_distribution"].loc[lambda x: x.level.eq("all") & x.failure.eq(failure)]
        for i, method in enumerate(chosen):
            part = progress.loc[progress.method.eq(method)].set_index("progress_bin").loc[list(PROGRESS_BINS)]
            axes[1, column].bar(np.arange(4) + (i - 1.5) * .21, part.fraction_of_alarms, .20, color=color_by_method[method])
        axes[1, column].set_xticks(np.arange(4), ["0-25%", "25-50%", "50-75%", "75-100%"])
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].set_ylabel("Fraction among alarms")
        axes[1, column].set_xlabel("Executed actions before alarm / actual trajectory action steps")
        axes[1, column].set_title("Control progress of detections" if failure else "Control progress of false alarms", fontsize=12)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Full-corpus first-alarm distributions | q0-q6 warm-up | no post-termination scores", fontsize=13)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"full_corpus_alarm_distribution.{suffix}", dpi=170)
    plt.close(fig)


def markdown_table(frame, columns):
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in frame)
    return "\n".join(rows)


def write_report(output, tables, summary):
    main = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("task_init")].set_index("method").loc[list(METHODS)]
    rows = [[ZH_NAMES[method], f"{int(row.tp)}/{int(row.failures)}", f"{row.recall:.2%}", str(int(row.fn)),
             f"{int(row.fp)}/{int(row.successes)}", f"{row.fpr:.2%}", f"{row.precision:.2%}"] for method, row in main.iterrows()]
    corpus = tables["corpus_coverage"].loc[lambda x: x.level.eq("cohort_suite")]
    cohort = tables["cohort_metrics"].loc[lambda x: x.calibration.eq("task_init") & x.method.isin(METHODS[:2])]
    suite = tables["suite_metrics"].loc[lambda x: x.calibration.eq("task_init") & x.method.isin(METHODS[:2])]
    alt = tables["pooled_metrics"].loc[lambda x: x.calibration.eq("episode")].set_index("method").loc[list(METHODS)]
    bins = tables["alarm_query_bins"].loc[lambda x: x.level.eq("all")]
    timing_rows = []
    for _, _, label in QUERY_BINS:
        if label == "q0-6":
            continue
        parts = [bins.loc[bins.method.eq(method) & bins.failure.eq(failure)].set_index("query_bin")
                 for failure in (True, False) for method in METHODS[:2]]
        timing_rows.append([label] + [str(int(part.loc[label, "count"])) for part in parts])
    phase = tables["alarm_progress_distribution"].loc[lambda x: x.level.eq("all")]
    phase_rows = []
    for failure in (True, False):
        for method in METHODS[:2]:
            part = phase.loc[phase.failure.eq(failure) & phase.method.eq(method)].set_index("progress_bin")
            phase_rows.append(["失败检出" if failure else "成功误报", ZH_NAMES[method]] +
                [str(int(part.loc[label, "count"])) for label in PROGRESS_BINS])
    cos_tasks = tables["task_metrics"].loc[lambda x: x.calibration.eq("task_init") & x.method.eq("cosine") & x.tp.gt(0)].sort_values("tp", ascending=False)
    fp_tasks = tables["task_metrics"].loc[lambda x: x.calibration.eq("task_init") & x.method.eq("euclidean")].sort_values("fp", ascending=False).head(8)
    physical = tables["physical_summary"].loc[lambda x: x.level.eq("all")]
    e, c, ce = (main.loc[key] for key in ("euclidean", "cosine", "cosine_neighbors_euclidean_score"))
    goal = suite.loc[suite.suite.eq("libero_goal") & suite.method.eq("euclidean")].iloc[0]
    e_phase = phase.loc[phase.method.eq("euclidean") & ~phase.failure]
    late_false_alarms = int(e_phase.loc[e_phase.progress_bin.isin(PROGRESS_BINS[2:]), "count"].sum())
    lines = ["# 全部 32,000 条轨迹：MoE kNN 的完整任务留出评估", "",
        f"本轮每条轨迹恰好测试一次：{summary['successes']:,} 条成功、{summary['failures']:,} 条失败，"
        f"共 {summary['valid_test_chunks']:,} 个有效测试 chunk。四个 suite、40 个任务、16 个检测配置。",
        "这是重新覆盖全库的交叉验证结果；同一轮的测试任务完全不进入参考、归一化和校准。"
        "B 不参与这些拟合；A 在其他轮可以作为历史参考。方法没有训练预测网络，但使用历史成功标签建库和定阈值。", "",
        "此前报告的 15/491 是旧 12 折未见任务的计数。旧全部测试含 21,120 次出现、13,568 条唯一 B 轨迹；"
        "本轮任务划分和参考点重新确定，不把旧计数相加或乘比例外推。", "",
        "## 主结果", "", "沿用 10D、k=20、4,096 点上限和成功 task/init 组最大值校准，名义 alpha=5%。"
        "阈值只从参考任务的 A 校准数据确定。", "",
        markdown_table(rows, ["方法", "检出失败", "召回率", "漏检", "成功误报", "误报率", "报警中实际失败占比"]), "",
        f"纯余弦在全库仍只检出 {int(c.tp)} 条、漏掉 {int(c.fn)} 条。保留余弦近邻而恢复欧氏打分后检出 {int(ce.tp)} 条，"
        "与上一轮诊断一致，说明打分中保留向量幅度非常关键；这组对照没有证明某个方法在相同实际误报率下最优。", "",
        f"欧氏的 {int(e.tp + e.fp):,} 次报警中，{int(e.fp):,} 次来自最终成功轨迹，报警精确率只有 {e.precision:.2%}。"
        f"其中 Goal 占 {int(goal.fp):,}/{int(e.fp):,} 条误报，本套件成功误报率 {goal.fpr:.2%}。"
        "任务分布和参考库覆盖仍是需要解决的问题，不能把高整段召回直接理解为可用的干预触发器。", "",
        "![完整语料对照](full_corpus_comparison.png)", "",
        "精确率直接反映触发后需要检查的成功轨迹数量；失败仅占全库 3.425%，不能用总体准确率衡量检测能力。"
        "表中的成功/失败是整条 rollout 最终标签，成功轨迹的 chunk 不保证一直处于健康状态，失败报警也不保证早于不可逆失误。", "",
        "## A/B 与任务套件", "",
        markdown_table([[r.cohort, r.suite, r.episodes, r.successes, r.failures] for r in corpus.itertuples()],
                       ["批次", "suite", "轨迹数", "成功", "失败"]), "",
        markdown_table([[r.cohort, ZH_NAMES[r.method], f"{r.tp}/{r.failures}", f"{r.recall:.2%}",
                         f"{r.fp}/{r.successes}", f"{r.fpr:.2%}"] for r in cohort.itertuples()],
                       ["批次", "方法", "检出失败", "召回率", "成功误报", "误报率"]), "",
        markdown_table([[r.suite, ZH_NAMES[r.method], f"{r.tp}/{r.failures}", f"{r.recall:.2%}",
                         f"{r.fp}/{r.successes}", f"{r.fpr:.2%}"] for r in suite.itertuples()],
                       ["suite", "方法", "检出失败", "召回率", "成功误报", "误报率"]), "",
        "## 触发时刻分布", "", "q 从 0 起计数。q7 发生在已执行 70 个动作后；q33 对应已执行 330 个动作。"
        "q0 至 q6 不报警；每个有效 q 都是真实记录的推理 chunk，没有给结束后的轨迹补齐动作。", "",
        markdown_table(timing_rows, ["首次报警 q", "欧氏：失败", "余弦：失败", "欧氏：成功", "余弦：成功"]), "",
        "下表仅统计实际触发的轨迹，以 `10*q / 实际 action_steps` 分箱。"
        "分母是事后已知的真实执行总长度，仅用于分析；固定 q 较小不等于处于任务初期。", "",
        markdown_table(phase_rows, ["结果", "方法", "0–25%", "25–50%", "50–75%", "75–100%"]), "",
        f"欧氏误报虽然集中在 q8 至 q14，但 {late_false_alarms:,}/{int(e.fp):,}（{late_false_alarms / e.fp:.2%}）"
        "实际已过该轨迹执行总长度的一半。这些多是较短的成功轨迹，不能仅凭绝对 q 较小归因为开局噪声。", "",
        "![报警时序分布](full_corpus_alarm_distribution.png)", "",
        f"没有有效评分 chunk 的短轨迹共 {summary['unscorable_episodes']} 条，仍留在分母。详细覆盖见 [corpus_coverage.csv](corpus_coverage.csv)。", "",
        "## 任务集中情况", "", "纯余弦发生失败检出的任务如下；其余任务检出数为零。", "",
        markdown_table([[r.task, f"{r.tp}/{r.failures}", f"{r.fp}/{r.successes}"] for r in cos_tasks.itertuples()],
                       ["任务", "检出失败", "成功误报"]), "", "欧氏误报条数最多的八个任务：", "",
        markdown_table([[r.task, f"{r.fp}/{r.successes}", f"{r.fpr:.2%}", f"{r.tp}/{r.failures}"] for r in fp_tasks.itertuples()],
                       ["任务", "成功误报", "误报率", "检出失败"]), "",
        "这些是任务层面的关联，不能据此断定具体物理失败原因；每任务完整数据见 [task_metrics.csv](task_metrics.csv)。", "",
        "## 校准敏感性", "", "若用成功 episode 的最大值直接校准，去掉 task/init 噪声重复的组最大值，得到：", "",
        markdown_table([[ZH_NAMES[method], f"{int(r.tp)}/{int(r.failures)}", f"{r.recall:.2%}",
                         f"{int(r.fp)}/{int(r.successes)}", f"{r.fpr:.2%}"] for method, r in alt.iterrows()],
                       ["方法", "检出失败", "召回率", "成功误报", "误报率"]), "",
        "这只是预先保留的校准对照，没有按测试表现选择工作点。分布发生任务迁移后，名义 5% 不保证实际成功误报率为 5%。", "",
        "## 已有物理事件子集", "", "只有 B 的部分失败有目标物体脱手事件；下表严格在这些已有事件上计算。"
        "事件时刻不等于不可逆失败起点，也不代表提前报警后能够救回。", "",
        markdown_table([[ZH_NAMES[r.method], r.event_episodes, r.before, r.at, r.after, r.missed] for r in physical.itertuples()],
                       ["方法", "事件轨迹", "早于事件", "同 q", "晚于事件", "没有报警"]), "",
        "## 数据和复现", "",
        "- [trajectory_results.csv](trajectory_results.csv)：恰好 32,000 行，原始轨迹身份、最终结果、六种方法首次报警、峰值、阈值和实际进度。`-1` 表示从未报警，NaN 峰值表示没有可评分 chunk。",
        "- [all_test_scores.npz](all_test_scores.npz)：`scores[method, global_row, query]` 为 `[6,32000,52]`；`first` 与 `thresholds` 为 `[2,6,32000]`，第一维依次为 episode、task_init。方法名与全局行号也保存在文件内。",
        "- [metrics.csv](metrics.csv)：全部、批次、套件、任务及其交叉分组的两种校准结果。",
        "- [alarm_query_distribution.csv](alarm_query_distribution.csv)：全部 q=-1,0,...,51 的精确分布；[alarm_progress_distribution.csv](alarm_progress_distribution.csv) 为实际控制进度分布。",
        "- [test_assignment.csv](test_assignment.csv)：每条轨迹唯一测试轮次；`profiles/` 保存同轮参考库每个点的真实 episode/query 身份。",
        "- [thresholds.csv](thresholds.csv)：逐轮、逐方法的校准阈值、单位数与秩。",
        "- [verification.json](verification.json) 与 [report_verification.json](report_verification.json)：划分、独立距离、时序、全量导出及哈希核验。",
        "- [固定方案](../../boundary_knn/FULL_CORPUS_PROTOCOL_ZH.md)、[打分及核验脚本](../../boundary_knn/full_corpus_knn.py)、[报告脚本](../../boundary_knn/report_full_corpus.py)。", "",
        "```bash", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/full_corpus_knn.py' score --output /tmp/himoe-full-corpus",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/full_corpus_knn.py' verify --output /tmp/himoe-full-corpus",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/report_full_corpus.py' --output /tmp/himoe-full-corpus", "```", "",
        "本轮没有重新评价 v7/v8 融合或进行噪声干预；当前结果仅覆盖上列六种已有 MoE 几何分数。", ""]
    (output / "REPORT_ZH.md").write_text("\n".join(lines))


def verify_report(output, tables, summary):
    assert digest(Path(__file__)) == summary["evaluator_sha256"]
    for name, expected in summary["inputs"].items():
        assert digest(ROOT / name) == expected, name
    for name, expected in summary["artifacts"].items():
        assert digest(output / name) == expected, name
    data = load_npz(output / "all_test_scores.npz")
    wide = pd.read_csv(output / "trajectory_results.csv")
    np.testing.assert_array_equal(wide.global_row, np.arange(32000))
    np.testing.assert_array_equal(data["global_rows"], wide.global_row)
    np.testing.assert_array_equal(data["methods"], METHODS)
    assert not wide.duplicated(["source", "episode"]).any()
    assert not wide.loc[wide.scorable_queries.eq(0), "failure"].any()
    y = wide.failure.to_numpy()
    decisions_checked = 0
    for mi, method in enumerate(METHODS):
        raw = data["scores"][mi]
        expected_peak = np.where(np.isfinite(raw).any(1), trajectory_peak(raw), np.nan)
        np.testing.assert_allclose(wide[f"{method}_peak_score"], expected_peak, rtol=1e-7, atol=1e-7, equal_nan=True)
        for ki, kind in enumerate(KINDS):
            crossed = np.isfinite(raw) & (raw > data["thresholds"][ki, mi, :, None])
            first = np.where(crossed.any(1), crossed.argmax(1), -1)
            np.testing.assert_array_equal(first, data["first"][ki, mi])
            key = f"{method}_first_alarm_query" if ki == 1 else f"{method}_episode_calibration_first_alarm_query"
            np.testing.assert_array_equal(first, wide[key])
            threshold_key = f"{method}_threshold" if ki == 1 else f"{method}_episode_calibration_threshold"
            np.testing.assert_allclose(wide[threshold_key], data["thresholds"][ki, mi], rtol=1e-12, atol=1e-12)
            expected = counts(y, first)
            entry = tables["pooled_metrics"].loc[lambda x: x.method.eq(method) & x.calibration.eq(kind)].iloc[0]
            for field, value in expected.items():
                np.testing.assert_allclose(entry[field], value)
            decisions_checked += len(first)
        q = data["first"][1, mi]
        np.testing.assert_array_equal(wide[f"{method}_executed_actions_before_alarm"], np.where(q >= 0, 10 * q, -1))
        np.testing.assert_allclose(wide[f"{method}_control_progress_at_alarm"],
            np.where(q >= 0, 10 * q / wide.actual_action_steps, np.nan), equal_nan=True)
    for table, groups, count_column in (("alarm_query_distribution", ["level", "cohort", "suite", "task", "method", "failure"], "episodes"),
                                        ("alarm_query_bins", ["level", "cohort", "suite", "task", "method", "failure"], "episodes"),
                                        ("alarm_progress_distribution", ["level", "cohort", "suite", "task", "method", "failure"], "alarms")):
        for _, part in tables[table].groupby(groups):
            assert int(part["count"].sum()) == int(part[count_column].iloc[0])
    for row in tables["physical_summary"].itertuples():
        assert row.before + row.at + row.after + row.missed == row.event_episodes
    artifacts = dict(summary["artifacts"])
    for name in ("REPORT_ZH.md", "full_corpus_comparison.png", "full_corpus_comparison.pdf",
                 "full_corpus_alarm_distribution.png", "full_corpus_alarm_distribution.pdf", "evaluation_summary.json"):
        artifacts[name] = digest(output / name)
    from PIL import Image
    for name in ("full_corpus_comparison.png", "full_corpus_alarm_distribution.png"):
        with Image.open(output / name) as im:
            assert min(im.size) >= 1000 and np.asarray(im.convert("RGB")).std() > 15
    write_json(output / "report_verification.json", dict(passed=True, unique_trajectory_rows=len(wide),
        verified_trajectory_decisions=decisions_checked, methods=len(METHODS), score_shape=data["scores"].shape,
        source_sha256=digest(Path(__file__)), inputs=summary["inputs"], artifacts=artifacts,
        sealed_manifest_sha256=summary["sealed_manifest_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    with threadpool_limits(limits=1):
        tables, summary = evaluate(output)
        make_figures(output, tables)
        write_report(output, tables, summary)
        verify_report(output, tables, summary)
    print(tables["pooled_metrics"].to_string(index=False), flush=True)
    print(f"REPORT VERIFIED: {summary['unique_test_trajectories']} unique trajectories", flush=True)


if __name__ == "__main__":
    main()
