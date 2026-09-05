from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SUMMARY_METHOD = "grammar_ewma_persistent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        type=Path,
        default=[
            Path("results-open-world-global-k4"),
            Path("results-open-world-global-k8"),
            Path("results-open-world-global-k12"),
        ],
    )
    parser.add_argument("--task-conditioned", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results-open-world-selected"))
    return parser.parse_args()


def load_summary(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "summary.json").read_text())


def detector_row(summary: dict[str, Any], horizon: int) -> dict[str, Any]:
    item = summary["full40_eventual_failure"][str(horizon)][SUMMARY_METHOD]
    return {
        "horizon": horizon,
        "within_task_state_auc": item["within_task_state_auc"],
        "state_blocked_ci95": item["state_blocked_ci95"],
        "healthy_fpr": item["healthy_fpr"],
        "failure_recall": item["failure_recall"],
    }


def main() -> None:
    args = parse_args()
    candidates = [(directory, load_summary(directory)) for directory in args.candidates]
    if any(summary["protocol"].get("conditioning") != "global" for _, summary in candidates):
        raise ValueError("all model-selection candidates must be strictly global")
    ranked = sorted(
        candidates,
        key=lambda item: item[1]["healthy_prefix_grammar"]["q4+"]["prefix_bits_per_phenotype"],
    )
    selected_directory, selected = ranked[0]
    selection = [
        {
            "directory": str(directory),
            "phase_states": summary["protocol"]["phase_states"],
            "q4_prefix_bits_per_phenotype": summary["healthy_prefix_grammar"]["q4+"][
                "prefix_bits_per_phenotype"
            ],
            "q4_prefix_minus_clock_bits_per_phenotype": summary["healthy_prefix_grammar"]["q4+"][
                "prefix_minus_clock_bits_per_phenotype"
            ],
        }
        for directory, summary in candidates
    ]
    physical = selected["scene8_physical_stasis_anchor"]["physical_onset"]
    result: dict[str, Any] = {
        "schema_version": 1,
        "selection_rule": (
            "minimum q4+ held-out-success full-prefix NLL; failure and stasis labels "
            "are not read for model selection"
        ),
        "candidates": selection,
        "selected_directory": str(selected_directory),
        "selected_phase_states": selected["protocol"]["phase_states"],
        "summary_method": SUMMARY_METHOD,
        "healthy_prefix_grammar": selected["healthy_prefix_grammar"],
        "strict_global_detection": [detector_row(selected, horizon) for horizon in (3, 7, 12)],
        "physical_stasis": {
            method: physical[method]
            for method in (
                "grammar_instant",
                "grammar_learned_persistent",
                "grammar_dwell_persistent",
                "grammar_ewma_persistent",
                "known_stasis",
            )
        },
        "event_atlas": selected["event_atlas"],
        "interpretation": (
            "healthy full-prefix syntax is measurable, but the selected strict MoE-only "
            "observer is not an effective early-warning detector at q3/q7"
        ),
    }
    task_conditioned = None
    if args.task_conditioned is not None:
        task_summary = load_summary(args.task_conditioned)
        task_conditioned = {
            "directory": str(args.task_conditioned),
            "online_extra_input": "known task ID",
            "detection": [detector_row(task_summary, horizon) for horizon in (3, 7, 12)],
            "physical_stasis": task_summary["scene8_physical_stasis_anchor"]["physical_onset"][
                SUMMARY_METHOD
            ],
        }
        result["task_conditioned_diagnostic"] = task_conditioned

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MoE-only 开放集观察器：模型选择与结论",
        "",
        "## 健康-only 模型选择",
        "",
        "K 只按 30,904 条 cross-fit held-out success 的 q4+ 下一 chord NLL 选择；"
        "选择过程不读取 failure 或 stasis 标签。",
        "",
        "| phase states | prefix bits/phenotype | prefix-clock |",
        "|---:|---:|---:|",
    ]
    for item in sorted(selection, key=lambda value: value["phase_states"]):
        lines.append(
            f"| {item['phase_states']} | {item['q4_prefix_bits_per_phenotype']:.4f} | "
            f"{item['q4_prefix_minus_clock_bits_per_phenotype']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"选择 K={result['selected_phase_states']}。负的 prefix-clock 说明完整已观测前缀确实改善"
            "健康下一词预测，但这不自动等于 Trap 可检测。",
            "",
            "## 严格 MoE-only 在线读数",
            "",
            f"固定汇总方法为 `{SUMMARY_METHOD}`；它不是按 failure outcome 选择的。",
            "",
            "| q | task/state AUC | 95% CI | healthy FPR | failure recall |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in result["strict_global_detection"]:
        lines.append(
            f"| {item['horizon']} | {item['within_task_state_auc']:.3f} | "
            f"[{item['state_blocked_ci95'][0]:.3f}, {item['state_blocked_ci95'][1]:.3f}] | "
            f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "q3 与随机水平不可区分；q7 只有很弱信号；q12 才出现中等但偏晚的关联。"
            "Full-40 阳性只是 eventual failure，因此这些数字仍不能证明早于第一个错误动作。",
            "",
            "## 物理 Stasis Onset",
            "",
            "| 方法 | success FPR | onset 前 recall | onset+3 recall |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, item in result["physical_stasis"].items():
        lines.append(
            f"| {method} | {item['success_episode_fpr']:.3f} | "
            f"{item['recall_by_physical_onset']:.3f} | "
            f"{item['recall_by_onset_plus3']:.3f} |"
        )
    if task_conditioned is not None:
        lines.extend(
            [
                "",
                "## Task-conditioned 诊断",
                "",
                "加入已知 task ID 后的结果只用于定位 nuisance variation，不属于严格 MoE-only 部署值。",
                "",
                "| q | task/state AUC | healthy FPR | failure recall |",
                "|---:|---:|---:|---:|",
            ]
        )
        for item in task_conditioned["detection"]:
            lines.append(
                f"| {item['horizon']} | {item['within_task_state_auc']:.3f} | "
                f"{item['healthy_fpr']:.3f} | {item['failure_recall']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## 判定",
            "",
            "当前证据支持“成功 routing 存在可预测的完整前缀结构”，但不支持“该结构已经构成可靠的"
            "开放世界早检器”。Unknown clusters 只是待物理审计的 phenotype candidates；漏报是 "
            "observer-unseen，不等于已经证明 MoE-silent。",
        ]
    )
    (args.output_dir / "REPORT.zh.md").write_text("\n".join(lines) + "\n")
    print(f"Selected {selected_directory}; wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
