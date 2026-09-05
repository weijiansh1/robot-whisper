#!/usr/bin/env python3
"""Audit the structured MoE alarm against the scalar rule and clocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

import evaluate_moe_invariant_alarm_cache_new as scalar
import evaluate_moe_structured_alarm_two_runs as structured


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_STRUCTURED = PACKAGE_ROOT / "results/moe_structured_alarm_two_runs"
DEFAULT_OUTPUT = DEFAULT_STRUCTURED / "audit"
SCHEMA = "himoe.moe_structured_alarm_audit.v1"
RUNS = {
    "seed1000_1007": {
        "run_id": "right-50x8-20260903",
        "scalar_cache": PACKAGE_ROOT
        / "results/moe_invariant_alarm_cache_new_recomputed_50x8/route_only_tasks",
    },
    "seed1008_1015": {
        "run_id": "right-50x8b-20260903",
        "scalar_cache": PACKAGE_ROOT
        / "results/moe_invariant_alarm_cache_new_50x8b/route_only_tasks",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured", type=Path, default=DEFAULT_STRUCTURED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_reference_file(path: Path) -> dict[tuple[int, int], np.ndarray]:
    output: dict[tuple[int, int], np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            if not name.startswith("component_") or "_bin_" not in name:
                continue
            prefix, bin_string = name.rsplit("_bin_", 1)
            component = int(prefix.removeprefix("component_"))
            output[(component, int(bin_string))] = np.asarray(archive[name])
    return output


def scalar_cache_path(run_name: str, task: str) -> Path:
    return scalar.task_cache_path(Path(RUNS[run_name]["scalar_cache"]), task)


def scalar_healthy_reference(
    run_name: str,
    run_map: dict[str, Path],
    tasks: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = {head: [] for head in scalar.HEADS}
    success_episodes = 0
    for task in tasks:
        arrays = scalar.load_route_cache(scalar_cache_path(run_name, task))
        outcomes = scalar.load_outcome_map(run_map[task])
        for episode, indices in scalar.episode_slices(arrays["episode"]):
            if not outcomes[episode]:
                continue
            success_episodes += 1
            for head in scalar.HEADS:
                values = np.asarray(arrays[f"evidence_{head}"][indices], dtype=np.float64)
                values = values[np.isfinite(values)]
                if len(values):
                    reference[head].append(float(values.max()))
    return (
        {head: np.asarray(values) for head, values in reference.items()},
        {
            "tasks": len(tasks),
            "success_episodes": success_episodes,
            "head_reference_counts": {
                head: len(values) for head, values in reference.items()
            },
        },
    )


def scalar_episode_maxima(
    frame: pd.DataFrame, success_episodes: list[int]
) -> list[float]:
    grouped = {
        int(episode): group
        for episode, group in frame.groupby("episode", sort=False)
    }
    output: list[float] = []
    for episode in success_episodes:
        values = grouped[int(episode)]["detector_confidence"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        output.append(float(values.max()) if len(values) else float("-inf"))
    return output


def fair_scalar_direction(
    source_name: str,
    target_name: str,
    partition: pd.DataFrame,
    run_maps: dict[str, dict[str, Path]],
    output: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    reference_tasks = partition.loc[
        partition["role"] == "healthy_reference", "task"
    ].astype(str).tolist()
    threshold_tasks = partition.loc[
        partition["role"] == "threshold_confirmation", "task"
    ].astype(str).tolist()
    heldout_tasks = partition.loc[
        partition["role"] == "heldout_test", "task"
    ].astype(str).tolist()
    references, reference_summary = scalar_healthy_reference(
        source_name, run_maps[source_name], reference_tasks
    )

    threshold_rows: list[dict[str, Any]] = []
    task_thresholds: list[float] = []
    for task in threshold_tasks:
        arrays = scalar.load_route_cache(scalar_cache_path(source_name, task))
        frame = scalar.detector_series(arrays, references)
        outcomes = scalar.load_outcome_map(run_maps[source_name][task])
        successes = [episode for episode, success in outcomes.items() if success]
        maxima = scalar_episode_maxima(frame, successes)
        threshold, alarms, total = structured.select_threshold(
            np.asarray(maxima), 0.01
        )
        task_thresholds.append(threshold)
        threshold_rows.append(
            {
                "source_run": source_name,
                "target_run": target_name,
                "task": task,
                "threshold": threshold,
                "success_alarm_episodes": alarms,
                "success_episodes": total,
            }
        )
    threshold = max(task_thresholds)

    prediction_rows: list[dict[str, Any]] = []
    for task in heldout_tasks:
        arrays = scalar.load_route_cache(scalar_cache_path(target_name, task))
        frame = scalar.detector_series(arrays, references)
        for episode, indices in scalar.episode_slices(arrays["episode"]):
            rows = frame.loc[indices]
            values = rows["detector_confidence"].to_numpy(dtype=np.float64)
            hit = np.flatnonzero(values >= threshold)
            prediction_rows.append(
                {
                    "task": task,
                    "episode": episode,
                    "episode_length": len(indices),
                    "alarm": bool(len(hit)),
                    "first_alarm_query": (
                        int(rows.iloc[hit[0]]["query"]) if len(hit) else np.nan
                    ),
                    "episode_max_score": (
                        float(np.nanmax(values))
                        if np.isfinite(values).any()
                        else np.nan
                    ),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    direction = f"{source_name}_to_{target_name}"
    prediction_path = output / "predictions_label_free" / f"scalar_{direction}.csv.gz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        prediction_path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    prediction_sha = structured.sha256_file(prediction_path)
    structured.write_json(
        prediction_path.with_name(f"scalar_{direction}_manifest.json"),
        {
            "schema": SCHEMA,
            "training": False,
            "target_outcomes_loaded": False,
            "predictions_written_before_target_outcomes": True,
            "source_run": source_name,
            "target_run": target_name,
            "prediction_sha256": prediction_sha,
            "rows": len(predictions),
            "threshold": threshold,
        },
    )

    # Target outcomes are first opened after the prediction file and hash exist.
    labels: list[dict[str, Any]] = []
    for task in heldout_tasks:
        outcomes = scalar.load_outcome_map(run_maps[target_name][task])
        labels.extend(
            {
                "task": task,
                "episode": episode,
                "failure": not success,
            }
            for episode, success in outcomes.items()
        )
    evaluated = predictions.merge(
        pd.DataFrame(labels), on=["task", "episode"], validate="one_to_one"
    )
    metrics = structured.rate_metrics(evaluated["alarm"], evaluated["failure"])
    return (
        {
            "source_run": source_name,
            "target_run": target_name,
            "threshold": threshold,
            "reference": reference_summary,
            "prediction_sha256": prediction_sha,
            "episodes": len(evaluated),
            "failures": int(evaluated["failure"].sum()),
            **metrics,
        },
        pd.DataFrame(threshold_rows),
        evaluated,
    )


def bounded_operating_points(
    structured_root: Path,
    budgets: list[int],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for manifest in sorted(
        (structured_root / "predictions_label_free").glob("*_manifest.json")
    ):
        direction = manifest.stem.removesuffix("_manifest")
        predictions = pd.read_csv(
            structured_root / "predictions_label_free" / f"{direction}.csv.gz",
            usecols=["task", "episode", "combined"],
        )
        maxima = predictions.groupby(["task", "episode"], as_index=False)["combined"].max()
        episodes = pd.read_csv(
            structured_root / "tables" / f"episode_flip_{direction}.csv"
        )
        episodes = episodes[episodes["role"] == "heldout_test"]
        parts.append(
            episodes[["direction", "task", "episode", "episode_length", "failure"]].merge(
                maxima, on=["task", "episode"], how="left", validate="one_to_one"
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    failure = frame["failure"].to_numpy(dtype=bool)
    score = frame["combined"].fillna(-np.inf).to_numpy(dtype=np.float64)
    length = frame["episode_length"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        score_candidates: list[tuple[int, int, float, dict[str, Any]]] = []
        for threshold in np.unique(score[np.isfinite(score)]):
            metrics = structured.rate_metrics(score >= threshold, failure)
            if metrics["fp"] <= budget:
                score_candidates.append(
                    (metrics["tp"], metrics["fp"], float(threshold), metrics)
                )
        best_score = max(score_candidates, key=lambda row: (row[0], row[1], -row[2]))
        rows.append(
            {
                "method": "structured_score_posthoc",
                "fp_budget": budget,
                "threshold_or_query": best_score[2],
                **best_score[3],
            }
        )
        clock_candidates: list[tuple[int, int, int, dict[str, Any]]] = []
        for query in range(int(length.max()) + 1):
            metrics = structured.rate_metrics(length > query, failure)
            if metrics["fp"] <= budget:
                clock_candidates.append((metrics["tp"], metrics["fp"], query, metrics))
        best_clock = max(clock_candidates, key=lambda row: (row[0], row[1], -row[2]))
        rows.append(
            {
                "method": "fixed_query_clock",
                "fp_budget": budget,
                "threshold_or_query": best_clock[2],
                **best_clock[3],
            }
        )
    return pd.DataFrame(rows)


def combine_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        key: int(sum(row[key] for row in rows))
        for key in ("episodes", "failures", "tp", "fp", "fn", "tn")
    }
    output.update(
        {
            "precision": output["tp"] / max(output["tp"] + output["fp"], 1),
            "failure_recall": output["tp"] / max(output["failures"], 1),
            "success_false_alarm_rate": output["fp"]
            / max(output["fp"] + output["tn"], 1),
        }
    )
    return output


def main() -> int:
    args = parse_args()
    structured_root = args.structured.resolve()
    output = args.output.resolve()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    config = json.loads(
        (PACKAGE_ROOT / "configs/moe_structured_alarm_two_runs.json").read_text(
            encoding="utf-8"
        )
    )
    summary = json.loads((structured_root / "summary.json").read_text(encoding="utf-8"))
    partition = pd.read_csv(structured_root / "tables/task_partition.csv")
    raw_root = structured.resolve_workspace(config["cache_root"])
    run_maps = {
        name: dict(scalar.discover_runs(raw_root, str(setting["run_id"])))
        for name, setting in RUNS.items()
    }

    scalar_rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []
    scalar_evaluated: list[pd.DataFrame] = []
    run_names = list(RUNS)
    for source_name, target_name in (
        (run_names[0], run_names[1]),
        (run_names[1], run_names[0]),
    ):
        row, thresholds, evaluated = fair_scalar_direction(
            source_name, target_name, partition, run_maps, output
        )
        scalar_rows.append(row)
        threshold_tables.append(thresholds)
        evaluated.insert(0, "direction", f"{source_name}_to_{target_name}")
        scalar_evaluated.append(evaluated)
    scalar_table = pd.DataFrame(scalar_rows)
    scalar_table.to_csv(output / "tables/fair_scalar_baseline.csv", index=False)
    pd.concat(threshold_tables, ignore_index=True).to_csv(
        output / "tables/fair_scalar_task_thresholds.csv", index=False
    )
    pd.concat(scalar_evaluated, ignore_index=True).to_csv(
        output / "tables/fair_scalar_episode_flip.csv", index=False
    )
    scalar_combined = combine_metrics(scalar_rows)

    structured_combined = summary["combined_heldout"]
    clock_combined = summary["combined_heldout_clock"]
    comparison = pd.DataFrame(
        [
            {"method": "structured_task_robust", **structured_combined},
            {"method": "scalar_task_robust", **scalar_combined},
            {"method": "prefrozen_fixed_query_clock", **clock_combined},
            {
                "method": "historical_scalar_original_protocol",
                "episodes": 6400,
                "failures": 239,
                "tp": 106,
                "fp": 58,
                "fn": 133,
                "tn": 6103,
                "precision": 106 / 164,
                "failure_recall": 106 / 239,
                "success_false_alarm_rate": 58 / 6161,
            },
        ]
    )
    comparison.to_csv(output / "tables/method_comparison.csv", index=False)
    operating = bounded_operating_points(
        structured_root, [0, 17, 27, 40, 58, 100, 134]
    )
    operating.to_csv(output / "tables/fp_budget_sweep.csv", index=False)

    task_rows: list[pd.DataFrame] = []
    cause_counts: dict[str, int] = {}
    for path in sorted((structured_root / "tables").glob("episode_flip_*.csv")):
        frame = pd.read_csv(path)
        frame = frame[frame["role"] == "heldout_test"].copy()
        for cause, count in frame.loc[frame["alarm_combined"], "alarm_cause"].value_counts().items():
            cause_counts[str(cause)] = cause_counts.get(str(cause), 0) + int(count)
        grouped: list[dict[str, Any]] = []
        for (direction, task), group in frame.groupby(["direction", "task"]):
            metrics = structured.rate_metrics(group["alarm_combined"], group["failure"])
            grouped.append(
                {
                    "direction": direction,
                    "task": task,
                    "episodes": len(group),
                    "failures": int(group["failure"].sum()),
                    **metrics,
                }
            )
        task_rows.append(pd.DataFrame(grouped))
    task_table = pd.concat(task_rows, ignore_index=True)
    task_table.to_csv(output / "tables/structured_task_metrics.csv", index=False)

    onset_totals: dict[str, dict[str, int]] = {}
    for direction in summary["directions"].values():
        onset = direction["onset"]
        for rule in (*structured.RULES, "clock", "matched_clock"):
            values = onset[rule]
            target = onset_totals.setdefault(rule, {key: 0 for key in values})
            for key, value in values.items():
                target[key] += int(value)
    eligible_onsets = sum(
        int(direction["onset"]["eligible_events"])
        for direction in summary["directions"].values()
    )

    output_summary = {
        "schema": SCHEMA,
        "status": "complete",
        "training": False,
        "learned_feature_weights": False,
        "physical_inputs_to_detector": False,
        "structured_task_robust": structured_combined,
        "fair_scalar_task_robust": scalar_combined,
        "prefrozen_clock": clock_combined,
        "structured_minus_fair_scalar": {
            "tp": structured_combined["tp"] - scalar_combined["tp"],
            "fp": structured_combined["fp"] - scalar_combined["fp"],
        },
        "structured_alarm_causes": cause_counts,
        "onset": {
            "eligible_events": eligible_onsets,
            "totals": onset_totals,
        },
        "conclusion": {
            "structured_information_improves_fair_scalar_rule": (
                structured_combined["tp"] > scalar_combined["tp"]
                and structured_combined["fp"] < scalar_combined["fp"]
            ),
            "reliable_early_detector_validated": False,
            "reason": "endpoint discrimination improves, but most physical-onset alarms remain late and lock-in still dominates",
        },
    }
    structured.write_json(output / "summary.json", output_summary)

    report = f"""# 结构化 MoE 报警公平审计

## 结论

在完全相同的 24 个健康参考任务、8 个逐任务阈值确认任务和 8 个跨 seed 留出任务协议下，旧三标量规则为 TP={scalar_combined['tp']}、FP={scalar_combined['fp']}、FN={scalar_combined['fn']}、TN={scalar_combined['tn']}；结构化规则为 TP={structured_combined['tp']}、FP={structured_combined['fp']}、FN={structured_combined['fn']}、TN={structured_combined['tn']}。因此新增的实际 expert ID/weight、位置分组和独立 collapse 确认不只是阈值技巧：多找到 {structured_combined['tp'] - scalar_combined['tp']} 个失败，同时少误报 {scalar_combined['fp'] - structured_combined['fp']} 个成功。

但它仍不是可靠 early detector。两组共有 {eligible_onsets} 个可评价物理 onset；组合规则只有 {onset_totals['combined']['near_first_minus2_to_0']} 个首次报警位于 `[-2,0]`，{onset_totals['combined']['near_active_minus2_to_0']} 个在该窗口内保持激活，而 {onset_totals['combined']['late']} 个是 onset 后报警。报警原因仍由 lock-in 主导：{cause_counts.get('lock_in', 0)} / {sum(cause_counts.values())}。

## 公平方法比较

{structured.markdown_table(comparison)}

## 固定误报数量压力测试

{structured.markdown_table(operating)}

该 sweep 是揭盲后的排序诊断，不是可部署阈值。在旧规则的 58 个误报数量上，结构化 score 检出数与固定时间钟仍接近，因此 endpoint 结果依然受到轨迹长度强烈混杂。

## 标签隔离

公平旧规则的目标 episode 预测先写入 `predictions_label_free/` 并记录 SHA256，随后才打开目标结果。两种 MoE 规则都没有训练权重，阈值只来自源组成功轨迹；物理量仅用于冻结后的 onset 评价。
"""
    (output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(structured.plain(output_summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
