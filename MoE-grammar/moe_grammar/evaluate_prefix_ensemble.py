from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit

from moe_grammar.statistics import (
    auc_pairwise,
    state_blocked_auc_difference,
    state_blocked_interval,
    stratified_pair_auc,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("results-prefix"))
    parser.add_argument("--output-dir", type=Path, default=Path("results-prefix"))
    parser.add_argument("--healthy-fpr", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def task_thresholds(
    score: np.ndarray,
    indexes: np.ndarray,
    success: np.ndarray,
    task: np.ndarray,
    lengths: np.ndarray,
    horizon: int,
    task_count: int,
    healthy_fpr: float,
) -> np.ndarray:
    eligible = indexes[success[indexes] & (lengths[indexes] > horizon)]
    global_threshold = float(
        np.quantile(score[eligible, horizon], 1.0 - healthy_fpr, method="higher")
    )
    output = np.full(task_count, global_threshold, dtype=np.float64)
    for task_index in range(task_count):
        selected = eligible[task[eligible] == task_index]
        if len(selected) >= 20:
            output[task_index] = np.quantile(
                score[selected, horizon], 1.0 - healthy_fpr, method="higher"
            )
    return output


def evaluate(
    score: np.ndarray,
    success: np.ndarray,
    task: np.ndarray,
    state: np.ndarray,
    lengths: np.ndarray,
    calibration_index: np.ndarray,
    test_index: np.ndarray,
    tasks: list[str],
    horizon: int,
    healthy_fpr: float,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    threshold = task_thresholds(
        score,
        calibration_index,
        success,
        task,
        lengths,
        horizon,
        len(tasks),
        healthy_fpr,
    )
    selected = test_index[lengths[test_index] > horizon]
    labels = ~success[selected]
    values = score[selected, horizon]
    alarms = values > threshold[task[selected]]
    task_names = np.asarray([tasks[index] for index in task[selected]], dtype=object)
    states = state[selected]
    strata = np.asarray(
        [f"{name}|{item_state}" for name, item_state in zip(task_names, states)],
        dtype=object,
    )
    within_auc, by_stratum, pairs = stratified_pair_auc(labels, values, strata)
    by_suite = {}
    for suite in sorted({name.split("/", 1)[0] for name in task_names}):
        mask = np.asarray([name.startswith(f"{suite}/") for name in task_names])
        by_suite[suite] = {
            "episodes": int(mask.sum()),
            "failures": int(labels[mask].sum()),
            "healthy_fpr": float(alarms[mask & ~labels].mean()),
            "failure_recall": float(alarms[mask & labels].mean()),
            "pooled_auc": auc_pairwise(labels[mask], values[mask]),
        }
    return {
        "episodes": len(selected),
        "failures": int(labels.sum()),
        "healthy_fpr": float(alarms[~labels].mean()),
        "failure_recall": float(alarms[labels].mean()),
        "pooled_auc": auc_pairwise(labels, values),
        "within_task_state_auc": within_auc,
        "state_blocked_ci95": list(
            state_blocked_interval(
                labels,
                values,
                task_names,
                states,
                bootstrap_draws,
                seed,
            )
        ),
        "informative_task_state_cells": len(by_stratum),
        "success_failure_pairs": pairs,
        "by_suite": by_suite,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# 因果 routing 前缀监督上界",
        "",
        "单向 GRU 在 train states 的成功与失败轨迹上学习最终 outcome；calibration states "
        "用于 early stopping 和 5% task-specific 阈值，所有数字来自未见 test states。"
        "这不是健康语法模型，而是 routing 前缀是否含有可学习早期信号的监督上界。",
        "",
        "| 输入 | q | within task/state AUC | 95% CI | healthy FPR | failure recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for modality, result in summary["modalities"].items():
        for horizon, item in result["horizons"].items():
            lines.append(
                f"| {modality} ({result['models']} seeds) | {horizon} | "
                f"{item['within_task_state_auc']:.3f} | "
                f"[{item['state_blocked_ci95'][0]:.3f}, {item['state_blocked_ci95'][1]:.3f}] | "
                f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
            )
    if "modality_comparisons" in summary:
        lines.extend(["", "## 配对增量", ""])
        for horizon, comparisons in summary["modality_comparisons"].items():
            for name, item in comparisons.items():
                lines.append(
                    f"- q{horizon} `{name}`: AUC 差 {item['auc_left_minus_right']:.3f}, "
                    f"state-blocked 95% CI {item['state_blocked_ci95']}。"
                )
    lines.extend(
        [
            "",
            "注意：标签仍然只是 episode 最终 success/failure，没有物理错误 onset。q7 的结果表示"
            "第 8 次规划结束时能够预测最终失败，不足以证明报警发生在错误动作之前。",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    paths = sorted(args.predictions_dir.glob("*_seed*_predictions.npz"))
    if not paths:
        raise FileNotFoundError(f"no prediction files under {args.predictions_dir}")
    groups: dict[str, list[tuple[np.ndarray, dict[str, Any]]]] = {}
    reference: dict[str, np.ndarray] | None = None
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
            arrays = {
                name: np.asarray(payload[name])
                for name in ("lengths", "task", "state", "success")
            }
            logits = np.asarray(payload["logits"], dtype=np.float64)
        if reference is None:
            reference = arrays
        else:
            for name in arrays:
                if not np.array_equal(reference[name], arrays[name]):
                    raise ValueError(f"prediction axis mismatch in {path}")
        groups.setdefault(str(metadata["modality"]), []).append((logits, metadata))
    assert reference is not None

    first_metadata = next(iter(groups.values()))[0][1]
    split = {
        name: np.asarray(values, dtype=np.int16)
        for name, values in first_metadata["split"].items()
    }
    calibration_index = np.flatnonzero(np.isin(reference["state"], split["calibration"]))
    test_index = np.flatnonzero(np.isin(reference["state"], split["test"]))
    summary: dict[str, Any] = {
        "schema_version": 1,
        "protocol": {
            "model": "unidirectional GRU trained with endpoint labels at every valid prefix",
            "checkpoint": "best mean calibration within-task/state AUC at q7 and q12",
            "ensemble": "mean probability across independent random seeds",
            "threshold": f"per-task calibration-success {1.0 - args.healthy_fpr:.0%} quantile",
            "endpoint": "eventual failure, not physical failure onset",
        },
        "modalities": {},
    }
    ensemble_scores: dict[str, np.ndarray] = {}
    for modality, models in groups.items():
        probabilities = np.mean([expit(item[0]) for item in models], axis=0)
        score = np.maximum.accumulate(probabilities, axis=1)
        ensemble_scores[modality] = score
        result = {
            "models": len(models),
            "seeds": [int(item[1]["seed"]) for item in models],
            "best_epochs": [int(item[1]["best_epoch"]) for item in models],
            "horizons": {},
        }
        for horizon in (3, 7, 12):
            result["horizons"][str(horizon)] = evaluate(
                score,
                reference["success"],
                reference["task"],
                reference["state"],
                reference["lengths"],
                calibration_index,
                test_index,
                list(first_metadata["tasks"]),
                horizon,
                args.healthy_fpr,
                args.bootstrap_draws,
                args.seed + horizon * 101 + len(summary["modalities"]),
            )
        summary["modalities"][modality] = result

    comparisons: dict[str, Any] = {}
    if "routing" in ensemble_scores and "behavior" in ensemble_scores:
        for horizon in (3, 7, 12):
            selected = test_index[reference["lengths"][test_index] > horizon]
            labels = ~reference["success"][selected]
            task_names = np.asarray(
                [first_metadata["tasks"][index] for index in reference["task"][selected]],
                dtype=object,
            )
            states = reference["state"][selected]
            item = {
                "routing_minus_behavior": state_blocked_auc_difference(
                    labels,
                    ensemble_scores["routing"][selected, horizon],
                    ensemble_scores["behavior"][selected, horizon],
                    task_names,
                    states,
                    args.bootstrap_draws,
                    args.seed + horizon * 307,
                )
            }
            if "combined" in ensemble_scores:
                item["combined_minus_behavior"] = state_blocked_auc_difference(
                    labels,
                    ensemble_scores["combined"][selected, horizon],
                    ensemble_scores["behavior"][selected, horizon],
                    task_names,
                    states,
                    args.bootstrap_draws,
                    args.seed + horizon * 307 + 1,
                )
            comparisons[str(horizon)] = item
    summary["modality_comparisons"] = comparisons

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_report(args.output_dir / "REPORT.zh.md", summary)
    print(f"Wrote ensemble audit to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
