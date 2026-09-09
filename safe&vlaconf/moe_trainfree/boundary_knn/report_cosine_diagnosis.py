"""Render the frozen cosine diagnosis with calibrated controls and provenance."""

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

from diagnose_cosine import ROOT, HERE, METHODS, OUTPUT, CASE_FOLD, load_npz, digest, write_json

LABELS = {
    "euclidean": "欧氏检索 + 欧氏打分",
    "cosine": "余弦检索 + 余弦打分",
    "cosine_neighbors_euclidean_score": "余弦检索 + 欧氏打分",
    "euclidean_neighbors_cosine_score": "欧氏检索 + 余弦打分",
    "cosine_without_center": "余弦：不减参考中心",
    "norm_only": "只用向量范数",
}
SHORT = ("Euclidean / Euclidean", "Cosine / cosine", "Cosine / Euclidean",
         "Euclidean / cosine", "Cosine, no centering", "Vector norm only")
COLORS = ("#b45f3c", "#168579", "#506a96", "#99713f", "#8c5779", "#707070")


def verify_artifacts(output):
    verification = json.loads((output / "verification.json").read_text())
    assert digest(HERE / "diagnose_cosine.py") == verification["source_sha256"]
    checks = 0
    for category, prefix in (("inputs", ROOT), ("artifacts", output)):
        for name, expected in verification[category].items():
            assert digest(prefix / name) == expected, name
            checks += 1
    entries = 0
    for path in sorted((output / "predictions").glob("*.npz")):
        data = load_npz(path)
        for key in ("scores", "calibration_scores"):
            score = data[key]
            finite = np.isfinite(score[0])
            assert (score[2][finite] >= score[0][finite] - 2e-6).all()
            assert (score[3][finite] >= score[1][finite] - 2e-7).all()
            np.testing.assert_array_equal(np.isfinite(score), np.broadcast_to(finite, score.shape))
            entries += int(finite.sum())
        for ki in range(2):
            for mi in range(len(METHODS)):
                crossing = np.isfinite(data["scores"][mi]) & (data["scores"][mi] > data["thresholds"][ki, mi])
                first = np.where(crossing.any(1), crossing.argmax(1), -1)
                np.testing.assert_array_equal(first, data["first"][ki, mi])
    return dict(hash_checks=checks, factorial_order_checks=entries, all_first_alarms_match=True)


def make_figure(output, tables):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    primary = tables["pooled_metrics"].loc[lambda x: x.scope.eq("unseen") & x.calibration.eq("task_init")].set_index("method")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, field, title in ((axes[0, 0], "fpr", "Unseen successes: false alarm rate"),
                             (axes[0, 1], "recall", "Unseen failures: recall")):
        values = primary.loc[list(METHODS), field].to_numpy()
        ax.barh(np.arange(len(METHODS)), values, color=COLORS, height=.65)
        ax.set_yticks(np.arange(len(METHODS)), SHORT)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlim(0, max(float(values.max()) * 1.18, .1))
        for i, value in enumerate(values):
            ax.text(value + ax.get_xlim()[1] * .012, i, f"{value:.1%}", va="center", fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
    phase = tables["calibration_phase_distribution"].loc[lambda x: x.method.eq("cosine") & x.kind.eq("query")]
    labels = ("q7-10", "q11-14", "q15-19", "q20-29", "q30-39", "q40-51")
    for j, (level, label, color) in enumerate((("episode", "Successful episode peaks", "#168579"),
                                              ("task_init", "Task/init group winners", "#506a96"))):
        values = phase.loc[phase.level.eq(level)].set_index("bin").loc[list(labels), "fraction"]
        axes[1, 0].bar(np.arange(6) + (j - .5) * .36, values, width=.34, color=color, label=label)
    axes[1, 0].set_xticks(np.arange(6), [label.replace("q", "") for label in labels])
    axes[1, 0].set_xlabel("Peak query index (not task progress)")
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1, 0].set_title("Where successful calibration cosine scores peak", fontsize=12)
    axes[1, 0].legend(frameon=False, fontsize=9)
    axes[1, 0].grid(axis="y", alpha=.18)
    axes[1, 0].set_axisbelow(True)
    neighbors = tables["case_neighbors"].loc[lambda x: x["query"].eq(33)]
    for method, color, marker in (("euclidean", "#b45f3c", "o"), ("cosine", "#168579", "^")):
        points = neighbors.loc[neighbors.selected_by.eq(method)]
        axes[1, 1].scatter(points.cosine, points.norm, c=color, marker=marker, s=45, alpha=.8, label=f"{method.title()} neighbors")
    query_norm = float(neighbors.current_norm.iloc[0])
    axes[1, 1].scatter([0], [query_norm], marker="*", s=160, c="#ae344b", label="Current q33")
    axes[1, 1].axhline(query_norm, color="#ae344b", lw=.8, ls=":")
    axes[1, 1].set_xlabel("Cosine distance to q33")
    axes[1, 1].set_ylabel("Standardized 10D vector norm")
    axes[1, 1].set_title("Same q33: reference direction and magnitude", fontsize=12)
    axes[1, 1].legend(frameon=False, fontsize=9)
    axes[1, 1].grid(alpha=.18)
    fig.suptitle("Cosine diagnosis: identical reference chunks, separate 5% group calibration", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cosine_diagnosis.{suffix}", dpi=170)
    plt.close(fig)


def threshold_owner_details(output, tables, audit):
    records = []
    for row in tables["threshold_owners"].loc[lambda x: x.method.eq("cosine")].itertuples():
        data = load_npz(output / "predictions" / f"{row.fold}.npz")
        candidates = [ROOT / name for name in audit["inputs"] if name.endswith(f"/profiles/{row.fold}.npz")]
        assert len(candidates) == 1
        profile = load_npz(candidates[0])
        position = int(np.flatnonzero(data["calibration_rows"] == row.global_row)[0])
        assert data["calibration_labels"][position] == 0
        norms = np.linalg.norm(profile["success_dynamic"], axis=1)
        records.append(dict(fold=row.fold, global_row=int(row.global_row), task=row.task,
            query=int(row.query), norm=float(row.norm), control_progress=float(row.control_progress),
            reference_norm_percentile=100 * float((norms <= row.norm).mean()),
            euclidean_score=float(data["calibration_scores"][0, position, row.query]),
            cosine_score=float(data["calibration_scores"][1, position, row.query]),
            euclidean_threshold=float(data["thresholds"][1, 0]), cosine_threshold=float(data["thresholds"][1, 1])))
    table = pd.DataFrame(records)
    table.to_csv(output / "threshold_owner_details.csv", index=False)
    return table


def make_report(output, tables, audit):
    primary = tables["pooled_metrics"].loc[lambda x: x.scope.eq("unseen") & x.calibration.eq("task_init")].set_index("method")
    phase = tables["calibration_phase_distribution"]
    episode_phase = phase.loc[phase.method.eq("cosine") & phase.kind.eq("query") & phase.level.eq("episode")]
    early = int(episode_phase.loc[episode_phase.bin.isin(["q7-10", "q11-14"]), "count"].sum())
    total = int(episode_phase["count"].sum())
    lost = tables["geometry_summary"].loc[lambda x: x.scope.eq("unseen") & x.population.eq("failure_lost")].iloc[0]
    geometry = tables["alarm_geometry"].loc[lambda x: x.scope.eq("unseen") & x.failure & ~x.cosine_ever_alarms]
    lines = ["# 余弦 kNN 为什么漏检：独立重放与控制对照", "",
        "没有发现余弦计算、排序或报警实现错误。主要损失来自纯角度打分删除了有效的幅度信息，",
        "同时成功校准样本中的角度峰值使全程阈值较高。中心化和分组最大值都有影响，但不足以单独解释低召回。", "",
        f"保留余弦邻居、恢复欧氏打分，未见任务检出从 {int(primary.loc['cosine', 'tp'])} 次回升至 "
        f"{int(primary.loc['cosine_neighbors_euclidean_score', 'tp'])} 次；"
        f"固定原欧氏邻居、只用余弦打分，仍仅检出 {int(primary.loc['euclidean_neighbors_cosine_score', 'tp'])} 次。", "",
        "本轮定位 Round 8 的低召回原因，复用缓存，不训练、不新增环境 rollout。",
        "固定 10 维动态、原参考点、k=20 和 12 折。新增对照仅改变近邻选择、距离打分或参考原点，",
        "每种分数均使用原 A 成功校准数据重新校准。所有变体是看过既有结果后的诊断，不是新的盲测。", "",
        "## 1. 实现与统计口径", "",
        f"独立使用 SciPy 的欧氏与余弦距离矩阵，重放全部 {audit['independent_scipy_query_replays']:,} 个有效校准 / 测试 chunk。",
        f"原欧氏分数最大绝对差 {audit['original_score_max_abs_errors']['euclidean']:.3g}，"
        f"原余弦最大绝对差 {audit['original_score_max_abs_errors']['cosine']:.3g}。",
        "旧阈值和首次报警完全复现；没有把相似度当距离、反转排序、沿用欧氏阈值或误删有效 chunk。",
        "Round 8 已核验四个套件的原始路由；本轮继续沿用相同输入哈希。", "",
        "## 2. 拆分检索与打分", "",
        "下表均为未见任务、task/init 分组校准 alpha=5%。每个参考对象仍是一个历史 chunk 特征点。",
        "重复折按出现次数统计，不作为独立样本。", "",
        "| 对照 | 误报 / 成功 | 误报率 | 检出 / 失败 | 召回 |", "|---|---:|---:|---:|---:|"]
    for method in METHODS:
        r = primary.loc[method]
        lines.append(f"| {LABELS[method]} | {int(r.fp)}/{int(r.successes)} | {r.fpr:.2%} | {int(r.tp)}/{int(r.failures)} | {r.recall:.2%} |")
    lines += ["", "‘余弦检索 + 欧氏打分’固定余弦选出的 20 个点，再计算它们与当前点的欧氏距离均值。",
        "‘欧氏检索 + 余弦打分’保留原欧氏近邻身份，只改变这些近邻的打分。",
        "前者检查恢复幅度后是否能恢复检出；后者检查仅固定旧近邻能否解决问题。",
        "‘不减参考中心’保留 MAD 尺度和原始参考 chunk，只去掉 median 平移，检查角度对原点的敏感性。",
        "‘只用向量范数’直接复用 Round 5 已保存的 dyn_radius，不增加新的训练或拟合。", "",
        "![诊断对照](cosine_diagnosis.png)", "", "## 3. 漏检点的邻居究竟像在哪里", "",
        f"聚焦原欧氏能检出、纯余弦整条轨迹不报警的 {int(lost.points)} 次未见任务失败，在各自原欧氏首次报警时检查邻居。",
        f"其中 {int(lost.cosine_neighbor_distance_le005)} 次的 20 个余弦邻居平均距离不超过 0.05（平均相似度至少 0.95）；",
        f"{int(lost.norm_at_least_twice_cosine_neighbors)} 次当前向量范数至少是余弦邻居平均范数的两倍。", "",
        "欧氏距离与角度、幅度的关系是：", "",
        "```text", "||x-y||^2 = (||x||-||y||)^2 + 2*||x||*||y||*cosine_distance(x,y)", "```", "",
        "纯余弦同时丢掉径向差和角度项的幅度权重。按每个近邻的距离分解后，下面统计径向部分对原欧氏平均距离的贡献：", "",
        "| 径向贡献占比 | 次数 | 比例 |", "|---|---:|---:|"]
    radial = tables["radial_fraction_bins"].loc[lambda x: x.scope.eq("unseen") & x.population.eq("failure_lost") & x.neighbors.eq("euclidean")]
    for r in radial.itertuples():
        lines.append(f"| {r.radial_fraction_bin} | {r.count} | {r.fraction:.2%} |")
    radial_below_half = int(geometry.euclidean_neighbors_radial_fraction.lt(.5).sum())
    lines += ["", f"需要区分‘幅度信息’和‘单纯半径差’：{radial_below_half}/{len(geometry)} 个漏检点的原欧氏分数中，"
        "径向差贡献不到一半。幅度也通过上式的 `2*||x||*||y||` 放大角度差；纯余弦连这一项也去掉了。",
        "因此不能把所有漏检简化为‘向量长度变大但方向不变’。",
        "这是已触发原欧氏报警时的算术分解，不是幅度变化造成物理失败的因果证据，也不表示幅度足以区分全部正常行为。", "",
        "## 4. 成功校准样本如何设置阈值", "",
        f"成功校准轨迹的 {total} 个有效余弦峰值中，{early} 个（{early / total:.2%}）出现在 q7 至 q14。",
        "这是绝对 query 编号；下面另外列出真实控制步比例，不能直接称为任务前期。", "",
        "| 峰值时已执行动作比例 | 成功轨迹峰值数 | task/init 组胜出峰值数 |", "|---|---:|---:|"]
    progress = phase.loc[phase.method.eq("cosine") & phase.kind.eq("control_progress")].set_index(["bin", "level"])
    for label in ("[0,25%)", "[25,50%)", "[50,75%)", "[75,100%)"):
        lines.append(f"| {label} | {int(progress.loc[(label, 'episode'), 'count'])} | {int(progress.loc[(label, 'task_init'), 'count'])} |")
    lines += ["", "全程最大值再对同一 task/init 的成功重复取最大值，使少数角度稀疏的正常点决定阈值。",
        "它符合原轨迹级误报校准规则；原欧氏也使用同样规则，所以并非程序多做了一次错误的最大值。", "",
        "| 纯余弦校准单位 | 检出 / 失败 | 召回 | 误报 / 成功 | 误报率 |", "|---|---:|---:|---:|---:|"]
    for kind in ("task_init", "episode"):
        r = tables["pooled_metrics"].loc[lambda x: x.method.eq("cosine") & x.scope.eq("unseen") & x.calibration.eq(kind)].iloc[0]
        lines.append(f"| {kind} | {int(r.tp)}/{int(r.failures)} | {r.recall:.2%} | {int(r.fp)}/{int(r.successes)} | {r.fpr:.2%} |")
    owner = tables["threshold_owner_details"].loc[lambda x: x.fold.eq(CASE_FOLD)].iloc[0]
    current = tables["case_queries"].set_index("query").loc[33]
    lines += ["", "改用逐轨迹校准取消了跨噪声重复的最大值，但改变了校准保证的统计单位，不能当成同一误报预算。",
        "[每折阈值及其倍率](threshold_inflation.csv)和[真正决定阈值的成功 chunk](threshold_owners.csv)保留了来源与时点。", "",
        "在同一个 Long 划分里，决定余弦阈值的成功点与此前失败案例的 q33 可以直接比较：", "",
        "| 样本 | 标准化向量范数 | 欧氏 kNN 距离 | 余弦 kNN 距离 |", "|---|---:|---:|---:|",
        f"| 成功校准轨迹 q{int(owner.query)} | {owner.norm:.4f} | {owner.euclidean_score:.6f} | {owner.cosine_score:.6f} |",
        f"| 失败轨迹 q33 | {current['norm']:.4f} | {current.euclidean_score:.6f} | {current.cosine_score:.6f} |", "",
        f"该成功点的范数只处在参考库第 {owner.reference_norm_percentile:.2f} 百分位。",
        "中心化之后，靠近参考中心的成功点也可能有较大的角度差；单位化忽略它的偏离幅度很小这一事实。",
        "这里是几何信息的丢失，不是浮点数下溢或零向量错误。仅去掉中心化有一定改善，但前面的对照显示仍远未恢复原检出。", "",
        "## 5. 同一条轨迹上的幅度控制", "",
        "仍使用之前固定的 Long episode 223、q33。下表只在特征空间中保持方向并缩放向量，未更改任何仿真状态或动作。", "",
        "| 缩放倍数 | 向量范数 | 欧氏 kNN | 余弦 kNN |", "|---|---:|---:|---:|"]
    for r in tables["case_radial_rescaling"].itertuples():
        lines.append(f"| {r.factor:g} | {r.norm:.4f} | {r.euclidean:.6f} | {r.cosine:.6f} |")
    case = tables["case_queries"].set_index("query").loc[33]
    lines += ["", f"真实 q33 的向量范数为 {case['norm']:.4f}，余弦近邻的平均范数为 {case.cosine_neighbors_neighbor_norm_mean:.4f}，"
        f"平均余弦距离为 {case.cosine_neighbors_cosine:.6f}。",
        "这种方向匹配并不要求幅度匹配，更不代表物理状态、任务或执行阶段相同。", "",
        "## 数据与复现", "",
        "- [全部工作点](pooled_metrics.csv) / [分套件](suite_metrics.csv) / [逐任务](task_metrics.csv)。",
        "- [完整报警 query 分布](query_distribution.csv) / 每折全部 chunk 分数与报警位置：`predictions/*.npz`。",
        "- [全部成功校准峰值](calibration_peaks.csv) / [组峰值](calibration_group_peaks.csv) / [阈值来源](threshold_owners.csv)。",
        "- [阈值来源点的幅度和两种距离](threshold_owner_details.csv)。",
        "- [所有原欧氏报警点的几何分解](alarm_geometry.csv) / [分布汇总](geometry_summary.csv)。",
        "- [逐 chunk 案例](case_queries.csv) / [近邻身份与幅度](case_neighbors.csv) / [幅度缩放控制](case_radial_rescaling.csv)。",
        "- [各折完整分数分布](distance_distributions.csv) / [逐点独立重放记录](replay_audit.csv) / [哈希与核验](verification.json)。", "",
        "```bash", "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/diagnose_cosine.py' --output /tmp/himoe-cosine-diagnosis",
        "python 'safe&vlaconf/moe_trainfree/boundary_knn/report_cosine_diagnosis.py' --output /tmp/himoe-cosine-diagnosis", "```", ""]
    (output / "REPORT_ZH.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    checks = verify_artifacts(output)
    tables = {path.stem: pd.read_csv(path) for path in output.glob("*.csv")}
    audit = json.loads((output / "verification.json").read_text())
    tables["threshold_owner_details"] = threshold_owner_details(output, tables, audit)
    make_figure(output, tables)
    make_report(output, tables, audit)
    write_json(output / "report_verification.json", dict(**checks, source_sha256=digest(Path(__file__)),
        artifacts={name: digest(output / name) for name in ("REPORT_ZH.md", "cosine_diagnosis.png", "cosine_diagnosis.pdf", "threshold_owner_details.csv")}))
    print("COSINE DIAGNOSIS REPORT COMPLETE", flush=True)


if __name__ == "__main__":
    main()
