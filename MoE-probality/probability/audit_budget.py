"""Audit metadata shortcuts with fixed-window, shared, and unseen-task readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import zarr

from .data import json_records, sha256, write_json
from .evaluation import calibration_bins, cluster_draws, metrics
from .features import DETECTOR_PATH, FEATURE_NAMES
from .model import MODEL_PARAMS
from .pure_moe import (LOCAL_NAMES, WINDOW, MoEWindowMonitor,
                       fit_calibrated, probabilities, select_local_features)


HERE = Path(__file__).resolve().parents[1]
TEST_SPLITS = ("test_unseen_init", "test_new_noise_seen_init")
GLOBAL_DESIGNS = ("clock_global", "clock_cap_global", "moe_local_global",
                  "moe_local_clock_global", "moe_local_clock_cap_global")


def task_folds(episodes):
    folds = {}
    for _, group in episodes.groupby("suite", sort=True):
        tasks = sorted(group.task.unique(), key=lambda t: hashlib.sha256(f"20260907:task-fold:{t}".encode()).hexdigest())
        for i, task in enumerate(tasks):
            folds[task] = i % 5
    return folds


def load_inputs(source):
    manifest = json.loads((source / "artifact_manifest.json").read_text())
    hashes = {}
    for name in ("episodes.csv", "dataset.npz", "predictions.csv.gz", "data_audit.json", "raw_verification.json"):
        actual = sha256(source / name)
        if actual != manifest[name]:
            raise ValueError(f"changed input artifact: {name}")
        hashes[name] = actual
    for name in ("probability/features.py", "probability/model.py", "probability/data.py"):
        if sha256(HERE / name) != manifest[name]:
            raise ValueError(f"source implementation no longer matches original experiment: {name}")
    episodes = pd.read_csv(source / "episodes.csv")
    with np.load(source / "dataset.npz", allow_pickle=False) as cache:
        np.testing.assert_array_equal(cache["feature_names"], FEATURE_NAMES)
        x, ep, query = cache["x"], cache["episode_row"], cache["query"]
    index = episodes.set_index("episode_row")
    np.testing.assert_array_equal(x[:, 0] + x[:, 1], index.loc[ep, "max_steps"])
    np.testing.assert_array_equal(x[:, 1], query * index.loc[ep, "replan_steps"].to_numpy())
    if index.groupby("max_steps").suite.nunique().max() != 1:
        raise ValueError("cap no longer uniquely identifies suite in this corpus")
    keep = query >= WINDOW - 1
    frame = index.loc[ep[keep], ["suite", "task", "cluster", "split", "success"]].reset_index()
    frame["query"] = query[keep]
    frame["success"] = frame.success.astype(int)
    frame["half_k4_alarm"] = frame["query"].to_numpy() == index.loc[ep[keep], "first_q_freeze_back_half_k4"].to_numpy()
    original = pd.read_csv(source / "predictions.csv.gz")
    np.testing.assert_array_equal(original.episode_row, ep)
    np.testing.assert_array_equal(original["query"], query)
    frame["original_suite_clock"] = original.loc[keep, "budget"].to_numpy()
    frame["original_suite_moe_clock"] = original.loc[keep, "moe"].to_numpy()
    return episodes, frame, x[keep], hashes


def fit_models(episodes, frame, x, output):
    local = select_local_features(x)
    designs = dict(clock_global=x[:, 1:2], clock_cap_global=x[:, :2], moe_local_global=local,
                   moe_local_clock_global=np.column_stack([local, x[:, 1]]),
                   moe_local_clock_cap_global=np.column_stack([local, x[:, :2]]))
    y = frame.success.to_numpy()
    train, cal = frame.split.eq("train").to_numpy(), frame.split.eq("calibration").to_numpy()
    frame["prior_global"] = float(y[train].mean())
    bundle = None
    models = ["prior_global"]
    for name in GLOBAL_DESIGNS:
        print(f"fit shared {name}", flush=True)
        design = designs[name]
        model = fit_calibrated(design[train], y[train], design[cal], y[cal])
        raw, p = probabilities(model, design)
        frame[name], frame[name + "_raw"] = p, raw
        models.extend([name, name + "_raw"])
        joblib.dump(model, output / "models" / (name + ".joblib"), compress=3)
        if name == "moe_local_global":
            bundle = dict(model=model, feature_names=LOCAL_NAMES, window=WINDOW,
                          detector_sha256=sha256(DETECTOR_PATH), model_params=MODEL_PARAMS,
                          task_or_suite_selection=False, clock_inputs=False, budget_inputs=False,
                          estimand="P(original_deadline_success | last_eight_MoE_queries,active)",
                          training_scope="mixture of the recorded four LIBERO policy/cap configurations")
            joblib.dump(bundle, output / "models/moe_window_shared.joblib", compress=3)
    for name in ("prior_suite", "clock_suite", "moe_local_suite"):
        frame[name] = np.nan
        models.append(name)
    for suite in sorted(frame.suite.unique()):
        take = frame.suite.eq(suite).to_numpy()
        frame.loc[take, "prior_suite"] = float(y[train & take].mean())
        for name, design in (("clock_suite", x[:, 1:2]), ("moe_local_suite", local)):
            print(f"fit {suite} {name}", flush=True)
            model = fit_calibrated(design[train & take], y[train & take], design[cal & take], y[cal & take])
            frame.loc[take, name] = probabilities(model, design[take])[1]
            joblib.dump(model, output / "models" / (suite + "_" + name + ".joblib"), compress=3)
    folds = task_folds(episodes)
    frame["task_fold"] = frame.task.map(folds)
    frame["moe_local_unseen_task"] = np.nan
    frame["moe_local_unseen_task_raw"] = np.nan
    frame["prior_unseen_task"] = np.nan
    task_splits = []
    for fold in range(5):
        held = frame.task_fold.eq(fold).to_numpy()
        test = held & frame.split.isin(TEST_SPLITS).to_numpy()
        fitting, calibration = train & ~held, cal & ~held
        held_tasks = sorted(frame.loc[held, "task"].unique().tolist())
        train_tasks = sorted(frame.loc[fitting, "task"].unique().tolist())
        assert set(held_tasks).isdisjoint(train_tasks)
        assert set(held_tasks).isdisjoint(frame.loc[calibration, "task"])
        print(f"fit task holdout {fold + 1}/5: {len(held_tasks)} held-out tasks", flush=True)
        model = fit_calibrated(local[fitting], y[fitting], local[calibration], y[calibration])
        raw, p = probabilities(model, local[test])
        frame.loc[test, "moe_local_unseen_task"] = p
        frame.loc[test, "moe_local_unseen_task_raw"] = raw
        frame.loc[test, "prior_unseen_task"] = float(y[fitting].mean())
        joblib.dump(model, output / "models" / f"held_task_fold{fold}.joblib", compress=3)
        task_splits.append(dict(fold=fold, held_tasks=held_tasks, training_tasks=train_tasks,
                                training_queries=int(fitting.sum()), calibration_queries=int(calibration.sum()),
                                test_queries=int(test.sum())))
    models.extend(["moe_local_unseen_task", "moe_local_unseen_task_raw", "prior_unseen_task",
                   "original_suite_clock", "original_suite_moe_clock"])
    if frame.loc[frame.split.isin(TEST_SPLITS), models].isna().any().any():
        raise AssertionError("missing held-out predictions")
    return bundle, models, task_splits


def evaluate_models(frame, model_names):
    rows, reliability, matched = [], [], []
    for split in TEST_SPLITS:
        for suite in ("all", *sorted(frame.suite.unique())):
            base = frame[(frame.split == split) & ((frame.suite == suite) if suite != "all" else True)]
            for selection in ("all_q7plus", "q7", "q15", "q25", "half_k4_alarm"):
                if selection == "all_q7plus":
                    block = base
                elif selection == "half_k4_alarm":
                    block = base[base.half_k4_alarm]
                else:
                    block = base[base["query"] == int(selection[1:])]
                if block.empty:
                    continue
                for name in model_names:
                    rows.append(dict(split=split, suite=suite, selection=selection, model=name,
                                     **metrics(block.success.to_numpy(), block[name].to_numpy())))
                    if suite == "all" and selection == "all_q7plus":
                        reliability.extend(dict(split=split, model=name, **r) for r in
                                           calibration_bins(block.success.to_numpy(), block[name].to_numpy()))
        base = frame[frame.split == split]
        totals = {name: [0.0, 0, 0] for name in model_names}
        for _, group in base.groupby(["task", "query"], sort=False):
            y = group.success.to_numpy()
            pairs = int(y.sum() * (len(y) - y.sum()))
            if not pairs:
                continue
            for name in model_names:
                totals[name][0] += roc_auc_score(y, group[name]) * pairs
                totals[name][1] += pairs
                totals[name][2] += 1
        for name, (numerator, pairs, strata) in totals.items():
            matched.append(dict(split=split, model=name, matched_auroc=numerator/pairs if pairs else None,
                                success_failure_pairs=pairs, mixed_task_query_strata=strata))
    primary = frame[frame.split == "test_unseen_init"].copy()
    comparisons = [
        ("moe_local_global", "clock_cap_global"),
        ("moe_local_global", "clock_suite"),
        ("moe_local_clock_cap_global", "moe_local_global"),
        ("moe_local_suite", "moe_local_global"),
        ("moe_local_unseen_task", "prior_unseen_task"),
    ]
    intervals = []
    for left, right in comparisons:
        losses = primary[["task", "cluster", "success"]].copy()
        losses["count"] = 1
        for name in (left, right):
            p = np.clip(primary[name].to_numpy(), 1e-6, 1-1e-6)
            y = primary.success.to_numpy()
            losses[name + "_brier"] = (p-y)**2
            losses[name + "_log_loss"] = -(y*np.log(p) + (1-y)*np.log1p(-p))
        columns = [left+"_brier", right+"_brier", left+"_log_loss", right+"_log_loss", "count"]
        draws = cluster_draws(losses, columns)
        for metric, i in (("brier", 0), ("log_loss", 2)):
            values = (draws[:, i]-draws[:, i+1])/draws[:, -1]
            low, high = np.quantile(values, [0.025, 0.975])
            intervals.append(dict(left=left, right=right, metric=metric,
                                  difference=float((losses[columns[i]]-losses[columns[i+1]]).mean()),
                                  low=float(low), high=float(high)))
    return pd.DataFrame(rows), pd.DataFrame(reliability), pd.DataFrame(matched), pd.DataFrame(intervals)


def verify_stream(hub, episodes, frame, bundle, prior_verification):
    saved = frame.set_index(["episode_row", "query"])
    samples, count, max_error = [], 0, 0.0
    for sample in prior_verification["samples"]:
        group = episodes[episodes.source_run == sample["source_run"]].sort_values("episode")
        local = int(np.flatnonzero(group.episode.to_numpy() == sample["episode"])[0])
        row = group.iloc[local]
        offset, length = int(group.length.iloc[:local].sum()), int(row.length)
        store = zarr.open_group(str(hub / row.source_run / "server/routes.zarr"), mode="r")
        raw = np.asarray(store["hb_router_probs"][offset:offset+length, :, 9, 1:, :])
        monitor = MoEWindowMonitor(bundle)
        for q, chunk in enumerate(raw):
            result = monitor.update(chunk)
            if q < WINDOW-1:
                assert result == dict(ready=False, success_probability=None)
                continue
            expected = float(saved.loc[(row.episode_row, q), "moe_local_global"])
            max_error = max(max_error, abs(expected-result["success_probability"]))
            np.testing.assert_allclose(result["success_probability"], expected, atol=1e-12, rtol=1e-12)
            count += 1
        samples.append(sample)
    return dict(source_runs=len(samples), raw_queries=sum(s["queries"] for s in samples),
                probabilities_checked=count, max_online_probability_error=max_error,
                first_seven_queries_abstain=True, samples=samples)


def report(output, frame, scores, reliability, matched, intervals, raw, task_splits):
    selected_names = ["prior_global", "prior_suite", "clock_global", "clock_cap_global", "clock_suite",
                      "moe_local_global", "moe_local_clock_global", "moe_local_clock_cap_global",
                      "moe_local_suite", "moe_local_unseen_task"]
    main = scores[(scores.split == "test_unseen_init") & (scores.suite == "all")
                  & (scores.selection == "all_q7plus")].set_index("model")
    matched_main = matched[matched.split == "test_unseen_init"].set_index("model")
    labels = dict(prior_global="共享常数先验", prior_suite="suite 专属常数先验", clock_global="共享模型，仅执行时钟",
                  clock_cap_global="共享模型，时钟＋预算上限", clock_suite="suite 分模，仅执行时钟",
                  moe_local_global="纯局部 MoE，共享模型", moe_local_clock_global="局部 MoE＋时钟，共享模型",
                  moe_local_clock_cap_global="局部 MoE＋时钟＋上限，共享模型",
                  moe_local_suite="局部 MoE，suite 分模", moe_local_unseen_task="纯局部 MoE，未见任务")
    lines = ["# 预算与任务先验审计", "", "## 对原结论的修正", "",
             "原来的‘仅执行预算’实际是 suite 专属模型加执行时钟。两列预算特征之和给出固定上限，"
             "220/280/300/520 在本数据中恰好分别识别 spatial/object/goal/long。"
             "原先的 MoE 模型也使用了这些信息，所以 0.9146 AUROC 不能直接称为纯 MoE 的成绩。", "",
             "当前实现中的上限来自固定评测配置；实际 rollout 的终局长度没有进入在线特征。"
             "因此本次发现的是任务套件与时钟先验的混杂，未发现直接读取未来轨迹或终局长度的输入。"
             "若用实际结束时刻减当前时刻作剩余预算，则会构成未来信息泄漏；本实现没有这样做。", "",
             "## 新的受限读出", "",
             "新主模型所有 suite 共用一套参数，仅接收最近 8 个 query 的 MoE 路由。"
             "18 个输入是前/后层的 mobility、4 步位移、token spread 各自的当前值、最近 4 值均值、"
             "3 步差分。没有任务/suite ID、预算、绝对 query、初始窗口基线、报警持续时间或历史长度。",
             "窗口未满时返回未就绪，前 7 个 query 不预测。所有对照都在同一 q>=7 风险集评估，"
             "并在相同范围重训，不能将这里的损失与旧版全时段损失直接相减。", "",
             "主划分沿用原先按任务/初态隔离的训练、校准和测试集合。模型容量与 sigmoid 校准流程固定，"
             "未按此次测试结果选择超参数；原始概率也完整保存在 CSV。此实验是用户提出混杂问题后的"
             "回顾性追加审计，不是新的盲测。", "",
             "## 同一测试风险集", "",
             f"主测试覆盖 {frame[frame.split == 'test_unseen_init'].episode_row.nunique():,} 条 rollout、"
             f"{len(frame[frame.split == 'test_unseen_init']):,} 个 q>=7 的 query。", "",
             "| 读出 | Brier | Log loss | AUROC | 同任务、同 q 的 AUROC |",
             "|---|---:|---:|---:|---:|"]
    for name in selected_names:
        row = main.loc[name]
        lines.append(f"| {labels[name]} | {row.brier:.5f} | {row.log_loss:.5f} | {row.auroc:.4f} | "
                     f"{matched_main.loc[name, 'matched_auroc']:.4f} |")
    brier_delta = intervals[(intervals.left == "moe_local_global") & (intervals.right == "clock_cap_global")
                           & (intervals.metric == "brier")].iloc[0]
    lines.extend(["", f"纯局部 MoE 的 Brier 为 {main.loc['moe_local_global', 'brier']:.5f}，"
                  f"时钟加上限基线为 {main.loc['clock_cap_global', 'brier']:.5f}；配对差的 95% 区间为 "
                  f"[{brier_delta.low:+.5f}, {brier_delta.high:+.5f}]。"
                  + ("区间跨零，不能据此断言纯 MoE 的 Brier 稳定优于该基线。" if brier_delta.low <= 0 <= brier_delta.high
                     else "区间未跨零，方向与幅度应结合上表读取。"), "",
                  "同任务、同 q 的 AUROC 只比较同一任务、同一执行时刻下结局不同的轨迹，"
                  "再按成功/失败对数加权汇总。常数、suite 和时钟基线在每个这样的组内给出相同值，"
                  "因此为 0.5。该诊断帮助排除仅靠区分任务或执行时刻获得较高池化 AUROC 的解释。", "",
                  "suite 分模会同时改变参数共享和模型总容量，其差值不是任务先验的纯因果效应。"
                  "共享模型中添加时钟/上限的对照保持了相同算法容量和局部 MoE 特征。", "",
                  "## 配对误差区间", "",
                  "差值为左侧模型减右侧模型；负数表示左侧损失更低。任务内按初态聚类 bootstrap，"
                  "固定已训练模型与已采样任务，不包含重新训练的不确定性。", "",
                  "| 左侧 | 右侧 | Brier 差 | 95% 区间 |", "|---|---|---:|---|"])
    for row in intervals[intervals.metric == "brier"].itertuples():
        lines.append(f"| {row.left} | {row.right} | {row.difference:+.5f} | [{row.low:+.5f}, {row.high:+.5f}] |")
    lines.extend(["", "## 未见任务检查", "",
                  "40 个任务分为 5 折，每折留出 8 个任务，每 suite 留出 2 个。"
                  "留出任务完全不进入该折训练或校准；仍在原测试轨迹上报告结果。"
                  f"所有 {len(task_splits)} 折的任务隔离检查通过。折内常数先验也保存在 metrics.csv。", "",
                  "## 解释边界", "",
                  "没有显式时钟输入，不代表 MoE 内部不能反映任务身份或执行阶段。"
                  "未见任务与同任务同 q 检查对这种解释提供约束，不能证明表征与时间完全独立。",
                  "删掉预算后，预测目标是原语料中任务、策略与原定截止条件混合分布下的"
                  "P(终局成功 | 最近 MoE 窗口)。它不再显式条件于给定的 b，"
                  "不能声称估计了任意预算下的 V(x,b)，也不能外推为无限预算成功概率。",
                  "真实 trap 与自然脱困仍没有独立真值。本审计没有增加物理状态输入、真实脱困标签"
                  "或新的环境续跑。", "",
                  "## 可复核产物", "",
                  f"32,000 集中全部 {len(frame):,} 个合格 query 都生成了受限模型预测；"
                  "未重新全量读取原始路由。复用已校验缓存，并对原先的 80 个原始 Zarr 样本逐 chunk"
                  f"重放新在线接口，核对 {raw['probabilities_checked']} 个概率值，最大误差 "
                  f"{raw['max_online_probability_error']:.3g}。",
                  "在线共享模型：models/moe_window_shared.joblib；入口：probability.pure_moe.MoEWindowMonitor。"
                  "模型输入契约、逐项消融、逐 q 指标、任务分折和哈希清单均随结果保存。", "",
                  "![受限输入审计](budget_audit.png)"])
    (output / "REPORT.zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    names = ["clock_cap_global", "moe_local_global", "moe_local_clock_cap_global", "moe_local_unseen_task"]
    short = ["clock + cap", "local MoE", "MoE + clock + cap", "MoE: unseen task"]
    colors = ["#be604a", "#007c83", "#789843", "#575757"]
    axes[0].bar(np.arange(4), main.loc[names, "brier"], color=colors)
    axes[0].set(xticks=np.arange(4), xticklabels=short, ylabel="Brier score", title="Same q >= 7 test queries")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[1].bar(np.arange(4), matched_main.loc[names, "matched_auroc"], color=colors)
    axes[1].axhline(0.5, color="#888888", linestyle="--")
    axes[1].set(xticks=np.arange(4), xticklabels=short, ylim=(0, 1), ylabel="Matched AUROC", title="Within task and query")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[2].plot([0, 1], [0, 1], "--", color="#888888")
    for name, label, color in zip(names, short, colors):
        group = reliability[(reliability.split == "test_unseen_init") & (reliability.model == name)]
        axes[2].plot(group.predicted, group.observed, "o-", label=label, color=color)
    axes[2].set(xlim=(0, 1), ylim=(0, 1), xlabel="Predicted success", ylabel="Observed success", title="Reliability")
    axes[2].legend(fontsize=8)
    fig.savefig(output / "budget_audit.png", dpi=180)
    fig.savefig(output / "budget_audit.pdf")
    plt.close(fig)


def run(source, hub, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    episodes, frame, x, source_hashes = load_inputs(source)
    bundle, names, folds = fit_models(episodes, frame, x, output)
    print("evaluate task/query controls and cluster intervals", flush=True)
    scores, reliability, matched, intervals = evaluate_models(frame, names)
    raw = verify_stream(hub, episodes, frame, bundle, json.loads((source / "raw_verification.json").read_text()))
    scores.to_csv(output / "metrics.csv", index=False)
    reliability.to_csv(output / "calibration.csv", index=False)
    matched.to_csv(output / "within_task_query.csv", index=False)
    intervals.to_csv(output / "cluster_intervals.csv", index=False)
    frame.to_csv(output / "predictions.csv.gz", index=False, float_format="%.12g")
    write_json(output / "task_folds.json", folds)
    write_json(output / "raw_verification.json", raw)
    summary = dict(source_hashes=source_hashes, eligible_queries=len(frame), window=WINDOW,
                   feature_names=LOCAL_NAMES, global_moe_inputs_contain_metadata=False,
                   budget_cap_identifies_suite=True, original_future_endpoint_input_found=False,
                   task_folds_disjoint=True, model_params=MODEL_PARAMS,
                   main_metrics=json_records(scores[(scores.split == "test_unseen_init") & (scores.suite == "all")
                                                   & (scores.selection == "all_q7plus")]),
                   within_task_query=json_records(matched), intervals=json_records(intervals))
    write_json(output / "summary.json", summary)
    report(output, frame, scores, reliability, matched, intervals, raw, folds)
    manifest = {str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.glob("probability/*.py"))}
    manifest.update({str(p.relative_to(HERE)): sha256(p) for p in sorted(HERE.glob("tests/*.py"))})
    manifest.update({str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*"))
                     if p.is_file() and p.name != "artifact_manifest.json"})
    write_json(output / "artifact_manifest.json", manifest)
    print(scores[(scores.split == "test_unseen_init") & (scores.suite == "all")
                 & (scores.selection == "all_q7plus")][["model", "rows", "brier", "log_loss", "auroc"]].to_string(index=False), flush=True)
    print(f"report: {output / 'REPORT.zh.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "results")
    parser.add_argument("--output", type=Path, default=HERE / "results_no_budget")
    parser.add_argument("--hub", type=Path, default=HERE.parent / "VLA_MUI_HUB")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    with threadpool_limits(limits=args.threads):
        run(args.source.resolve(), args.hub.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
