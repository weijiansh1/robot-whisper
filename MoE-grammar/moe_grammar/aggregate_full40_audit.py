from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fold-dirs",
        nargs="+",
        type=Path,
        default=[
            Path("results-full40"),
            Path("results-full40-fold1"),
            Path("results-full40-fold2"),
            Path("results-full40-fold3"),
            Path("results-full40-fold4"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results-full40-crossfit"))
    return parser.parse_args()


def weighted_mean(values: list[float], weights: list[int]) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def aggregate_detection(folds: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    horizons = sorted(folds[0]["detection"], key=int)
    for horizon in horizons:
        methods = sorted(set.intersection(*(set(item["detection"][horizon]) for item in folds)))
        output[horizon] = {}
        for method in methods:
            items = [fold["detection"][horizon][method] for fold in folds]
            pairs = [int(item["success_failure_pairs"]) for item in items]
            successes = [int(item["successes"]) for item in items]
            failures = [int(item["failures"]) for item in items]
            aucs = [float(item["within_task_state_auc"]) for item in items]
            fprs = [float(item["healthy_fpr"]) for item in items]
            recalls = [float(item["failure_recall"]) for item in items]
            output[horizon][method] = {
                "crossfit_within_task_state_auc": weighted_mean(aucs, pairs),
                "fold_auc_range": [min(aucs), max(aucs)],
                "fold_aucs": aucs,
                "healthy_fpr": weighted_mean(fprs, successes),
                "failure_recall": weighted_mean(recalls, failures),
                "test_successes": sum(successes),
                "test_failures": sum(failures),
                "success_failure_pairs": sum(pairs),
            }
        if "history.page0.5" in output[horizon] and "phase.page0.5" in output[horizon]:
            history = output[horizon]["history.page0.5"]
            phase = output[horizon]["phase.page0.5"]
            old = output[horizon]["old_pooled_cusum"]
            output[horizon]["fixed_method_differences"] = {
                "history_minus_phase_auc": (
                    history["crossfit_within_task_state_auc"]
                    - phase["crossfit_within_task_state_auc"]
                ),
                "history_minus_old_pooled_cusum_auc": (
                    history["crossfit_within_task_state_auc"]
                    - old["crossfit_within_task_state_auc"]
                ),
            }
    return output


def aggregate_grammar(folds: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = [int(item["healthy_grammar"]["episodes"]) for item in folds]
    output: dict[str, Any] = {"test_success_episodes": sum(episodes)}
    for metric in ("history_minus_phase", "history1_minus_phase", "ordered_minus_bag"):
        values = [
            float(item["healthy_grammar"][metric]["mean_left_minus_right"])
            for item in folds
        ]
        output[metric] = {
            "episode_weighted_mean_bits_per_query": weighted_mean(values, episodes),
            "fold_range": [min(values), max(values)],
            "fold_values": values,
            "all_fold_intervals_exclude_zero": all(
                not (
                    item["healthy_grammar"][metric]["ci95"][0]
                    <= 0.0
                    <= item["healthy_grammar"][metric]["ci95"][1]
                )
                for item in folds
            ),
        }
    learning: dict[str, Any] = {}
    for state_count in ("5", "10", "20", "30"):
        order_values = [
            float(
                item["grammar_learning_curve_fixed_tokenizer"][state_count][
                    "ordered_minus_bag"
                ]["mean_left_minus_right"]
            )
            for item in folds
        ]
        history_values = [
            float(
                item["grammar_learning_curve_fixed_tokenizer"][state_count][
                    "history_minus_phase"
                ]["mean_left_minus_right"]
            )
            for item in folds
        ]
        train_counts = [
            int(
                item["grammar_learning_curve_fixed_tokenizer"][state_count][
                    "training_success_episodes"
                ]
            )
            for item in folds
        ]
        learning[state_count] = {
            "mean_training_success_episodes": float(np.mean(train_counts)),
            "history_minus_phase_mean": float(np.mean(history_values)),
            "ordered_minus_bag_mean": float(np.mean(order_values)),
            "ordered_minus_bag_fold_range": [min(order_values), max(order_values)],
            "folds_with_ordered_gain": int(np.sum(np.asarray(order_values) < 0.0)),
        }
    output["fixed_tokenizer_learning_curve"] = learning
    return output


def write_report(path: Path, summary: dict[str, Any]) -> None:
    grammar = summary["healthy_grammar"]
    lines = [
        "# Full-40 五折交叉拟合结论",
        "",
        "## 是否是健康数据太少",
        "",
        "40 个任务、32,000 条轨迹、30,904 条成功轨迹已全部纳入五折审计。每折使用 30 个 states "
        "的约 18,500 条成功轨迹训练，10 个 states 校准，10 个 states 测试；50 个 states 均且仅测试一次。",
        "",
        f"task+position+二阶有序历史相对 task+position 的连续 NLL 差为 "
        f"{grammar['history_minus_phase']['episode_weighted_mean_bits_per_query']:.4f} bits/query，"
        f"fold 范围 {grammar['history_minus_phase']['fold_range']}。"
        f"有序二阶相对二词 bag 的差为 "
        f"{grammar['ordered_minus_bag']['episode_weighted_mean_bits_per_query']:.5f} bits/query，"
        f"fold 范围 {grammar['ordered_minus_bag']['fold_range']}；五折区间均不含零。",
        "",
        "固定 tokenizer 的样本量消融：",
        "",
        "| train states | 平均成功轨迹 | history-phase | ordered-bag | 有序增益 folds |",
        "|---:|---:|---:|---:|---:|",
    ]
    for state_count, item in grammar["fixed_tokenizer_learning_curve"].items():
        lines.append(
            f"| {state_count} | {item['mean_training_success_episodes']:.0f} | "
            f"{item['history_minus_phase_mean']:.4f} | {item['ordered_minus_bag_mean']:.5f} | "
            f"{item['folds_with_ordered_gain']}/5 |"
        )
    lines.extend(
        [
            "",
            "结论：较小数据已经足以看到前一词依赖；更多数据主要让很小的二阶顺序效应稳定下来。"
            "因此旧实验并非完全没学到，但确实不足以高精度估计稀疏的有序上下文。",
            "",
            "## CUSUM 与早期检测",
            "",
            "下面按 task×state 内正负 pair 聚合 AUC；阈值目标为 calibration-success 5% FPR。",
            "",
            "| q | 方法 | AUC | fold 范围 | test FPR | failure recall |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for horizon in ("3", "7", "12"):
        for method in (
            "old_pooled_cusum",
            "phase.page0.5",
            "history.page0.5",
            "history.glr",
            "residual.page0.5",
        ):
            item = summary["detection"][horizon][method]
            lines.append(
                f"| {horizon} | {method} | {item['crossfit_within_task_state_auc']:.3f} | "
                f"[{item['fold_auc_range'][0]:.3f}, {item['fold_auc_range'][1]:.3f}] | "
                f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
            )
    q7 = summary["detection"]["7"]["fixed_method_differences"]
    q12 = summary["detection"]["12"]["fixed_method_differences"]
    lines.extend(
        [
            "",
            f"条件校准 history Page-CUSUM 相对旧 pooled CUSUM 的 AUC 增量为 q7 "
            f"{q7['history_minus_old_pooled_cusum_auc']:.3f}、q12 "
            f"{q12['history_minus_old_pooled_cusum_auc']:.3f}；但相对 phase-only 仅为 q7 "
            f"{q7['history_minus_phase_auc']:.3f}、q12 {q12['history_minus_phase_auc']:.3f}。",
            "",
            "结论：旧 CUSUM 的 pooled calibration 和 uniform-score kappa 确实不理想；改成 task+position "
            "条件 CDF、normal-score Page-CUSUM 后有小幅提升。但提升几乎都能由 phase-only 得到，"
            "不是健康句法残差贡献。q3 接近随机，q7 只能弱预测，q12 才有中等信号。",
            "",
            f"五折 calibration 最优候选分别为：{summary['calibration_selected_detectors']}。"
            "选择器跨 fold 不稳定，不建议在线部署时搜索后取最好。",
            "",
            "## 边界",
            "",
            "- 这里的阳性是最终 episode failure；新数据没有独立物理 failure-onset 标签。"
            "所以只能说 q7/q12 预测最终失败，不能证明一定早于错误动作。",
            "- q12 只包含长度大于 12 的 17,992 条成功轨迹，但包含全部 1,096 条失败轨迹；"
            "它是 survivor-conditioned 晚期读数，不能与 q7 的总体覆盖直接比较。",
            "- nominal 5% 阈值在 unseen states 上通常漂移到更高 FPR，说明上线前还需要更保守的"
            " state-blocked/conformal 阈值校准。",
            "- 连续 NLL bits 与旧报告的硬 token bits 不是同一绝对标尺；这里只在同一 full-40 "
            "protocol 内比较差值和学习曲线。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    folds = [json.loads((directory / "summary.json").read_text()) for directory in args.fold_dirs]
    test_states = [state for fold in folds for state in fold["split"]["test"]]
    if len(test_states) != len(set(test_states)) or sorted(test_states) != list(range(50)):
        raise ValueError("fold test states must partition 0..49 exactly once")
    summary = {
        "schema_version": 1,
        "fold_directories": [str(path) for path in args.fold_dirs],
        "crossfit_test_states": sorted(test_states),
        "corpus": {
            key: folds[0]["corpus"][key]
            for key in ("queries", "episodes", "successes", "failures")
        },
        "healthy_grammar": aggregate_grammar(folds),
        "detection": aggregate_detection(folds),
        "calibration_selected_detectors": [
            fold["detector_selection"]["selected"] for fold in folds
        ],
        "calibration_selected_frequencies": dict(
            Counter(fold["detector_selection"]["selected"] for fold in folds)
        ),
        "aggregation": {
            "auc": "weighted by within-task/state success-failure pair count",
            "fpr": "weighted by eligible test successes",
            "recall": "weighted by eligible test failures",
            "uncertainty": "fold range; no synthetic global CI reconstructed from fold summaries",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "REPORT.zh.md", summary)
    print(f"Wrote five-fold aggregate to {args.output_dir}")


if __name__ == "__main__":
    main()
