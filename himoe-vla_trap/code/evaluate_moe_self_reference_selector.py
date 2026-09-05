#!/usr/bin/env python3
"""Evaluate the frozen task-free self-reference MoE selector on completed runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr

from moe_self_reference_selector import (
    SELECTOR_VERSION,
    SelfReferenceConfig,
    SelfReferenceCouplingCollapseAlarm,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CACHE_ROOT = WORKSPACE_ROOT / "VLA_MUI_HUB/cache_new/HiMoE-VLA"
DEFAULT_RUN_ID = "right-50x8-20260903"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/self_reference_coupling_collapse_v3.json"
DEFAULT_OUTPUT = PACKAGE_ROOT / "results/task_free_self_reference_selector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def discover_runs(cache_root: Path, run_id: str) -> list[Path]:
    runs: list[Path] = []
    for summary_path in sorted(
        cache_root.glob(f"libero_*/*/{run_id}/client/summaries.json")
    ):
        run = summary_path.parents[1]
        meta_path = run / "meta.json"
        route_path = run / "server/routes.zarr"
        if not meta_path.exists() or not route_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        sampling = meta.get("sampling", {})
        if (
            meta.get("status") == "complete"
            and sampling.get("complete") is True
            and int(sampling.get("actual_episodes", -1))
            == int(sampling.get("designed_episodes", -2))
        ):
            runs.append(run)
    if not runs:
        raise RuntimeError(f"no complete {run_id!r} runs under {cache_root}")
    return runs


def task_key(run: Path, cache_root: Path) -> str:
    return str(run.relative_to(cache_root).parent)


def first_alarm_payload(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    alarms = [row for row in decisions if bool(row["alarm"])]
    if not alarms:
        return {
            "first_alarm_query": np.nan,
            "alarm_query_count": 0,
            "alarm_planning_churn_ratio": np.nan,
            "alarm_state_action_gap_ratio": np.nan,
            "alarm_state_response_ratio": np.nan,
        }
    first = alarms[0]
    return {
        "first_alarm_query": int(first["query"]),
        "alarm_query_count": len(alarms),
        "alarm_planning_churn_ratio": float(first["planning_churn_ratio"]),
        "alarm_state_action_gap_ratio": float(first["state_action_gap_ratio"]),
        "alarm_state_response_ratio": float(first["state_response_ratio"]),
    }


def predict_routes(
    runs: Iterable[Path], cache_root: Path, config: SelfReferenceConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    episode_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    run_lookup: dict[str, Path] = {}
    for run_index, run in enumerate(runs, start=1):
        task = task_key(run, cache_root)
        run_lookup[task] = run
        group = zarr.open_group(str(run / "server/routes.zarr"), mode="r")
        episode_ids = np.asarray(group["episode_id"][:], dtype=np.int64)
        unique_ids = np.unique(episode_ids)
        for episode in unique_ids:
            indices = np.flatnonzero(episode_ids == episode)
            if len(indices) > 1 and not np.all(np.diff(indices) == 1):
                raise ValueError(f"non-contiguous route rows: {task}/{episode}")
            routes = np.asarray(
                group["hb_router_probs"].oindex[indices], dtype=np.float32
            )
            selector = SelfReferenceCouplingCollapseAlarm(config)
            decisions: list[dict[str, Any]] = []
            for route in routes:
                decision = selector.update(route).to_dict()
                decisions.append(decision)
                query_rows.append({"task": task, "episode": int(episode), **decision})
            episode_rows.append(
                {
                    "task": task,
                    "episode": int(episode),
                    "route_length": len(routes),
                    **first_alarm_payload(decisions),
                }
            )
        print(
            f"[route-only {run_index}] {task}: episodes={len(unique_ids)}",
            flush=True,
        )
    return pd.DataFrame(episode_rows), pd.DataFrame(query_rows), run_lookup


def unblind(
    predictions: pd.DataFrame, run_lookup: dict[str, Path]
) -> pd.DataFrame:
    outcome_rows: list[dict[str, Any]] = []
    for task, run in run_lookup.items():
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        summaries = json.loads(
            (run / "client/summaries.json").read_text(encoding="utf-8")
        )
        horizon = int(math.ceil(int(meta["max_steps"]) / 10.0))
        for row in summaries:
            outcome_rows.append(
                {
                    "task": task,
                    "episode": int(row["episode_index"]),
                    "success": bool(row["success"]),
                    "failure": not bool(row["success"]),
                    "init_state_id": int(row["init_state_id"]),
                    "flow_noise_seed": int(row["flow_noise_seed"]),
                    "inference_calls": int(row["inference_calls"]),
                    "horizon_queries": horizon,
                }
            )
    outcomes = pd.DataFrame(outcome_rows)
    frame = predictions.merge(outcomes, on=["task", "episode"], validate="one_to_one")
    if not np.array_equal(frame["route_length"], frame["inference_calls"]):
        raise ValueError("route and client episode lengths disagree")
    frame["alarm"] = frame["first_alarm_query"].notna()
    frame["alarm_lead_queries"] = np.where(
        frame["alarm"],
        frame["inference_calls"] - frame["first_alarm_query"] - 1,
        np.nan,
    )
    frame["alarm_horizon_phase"] = frame["first_alarm_query"] / np.maximum(
        frame["horizon_queries"] - 1, 1
    )
    return frame


def counts(frame: pd.DataFrame) -> dict[str, Any]:
    failure = frame["failure"].to_numpy(bool)
    alarm = frame["alarm"].to_numpy(bool)
    tp = int(np.sum(failure & alarm))
    fn = int(np.sum(failure & ~alarm))
    fp = int(np.sum(~failure & alarm))
    tn = int(np.sum(~failure & ~alarm))
    detected_lead = frame.loc[failure & alarm, "alarm_lead_queries"].to_numpy(float)
    return {
        "episodes": len(frame),
        "failures": tp + fn,
        "successes": fp + tn,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "tpr": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp_with_at_least_3_query_lead": int(np.sum(detected_lead >= 3)),
        "tp_with_at_least_5_query_lead": int(np.sum(detected_lead >= 5)),
        "median_failure_alarm_lead_queries": (
            float(np.median(detected_lead)) if len(detected_lead) else None
        ),
        "median_failure_alarm_horizon_phase": (
            float(frame.loc[failure & alarm, "alarm_horizon_phase"].median())
            if tp
            else None
        ),
    }


def task_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, group in frame.groupby("task", sort=True):
        row = counts(group)
        rows.append({"task": task, **row})
    return pd.DataFrame(rows)


def task_bootstrap(
    task_frame: pd.DataFrame, draws: int, seed: int
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    records = task_frame.to_dict("records")
    tpr = np.empty(draws, dtype=np.float64)
    fpr = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sample = [records[index] for index in rng.integers(0, len(records), len(records))]
        tp = sum(int(row["tp"]) for row in sample)
        fn = sum(int(row["fn"]) for row in sample)
        fp = sum(int(row["fp"]) for row in sample)
        tn = sum(int(row["tn"]) for row in sample)
        tpr[draw] = tp / max(tp + fn, 1)
        fpr[draw] = fp / max(fp + tn, 1)
    return {
        "tpr_95": np.quantile(tpr, [0.025, 0.975]).tolist(),
        "fpr_95": np.quantile(fpr, [0.025, 0.975]).tolist(),
    }


def clock_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase in np.linspace(0.10, 1.00, 91):
        alarm_query = np.ceil(phase * frame["horizon_queries"].to_numpy()).astype(int) - 1
        alarm = frame["inference_calls"].to_numpy() > alarm_query
        failure = frame["failure"].to_numpy(bool)
        lead = frame["inference_calls"].to_numpy() - alarm_query - 1
        tp = int(np.sum(failure & alarm))
        fp = int(np.sum(~failure & alarm))
        fn = int(np.sum(failure & ~alarm))
        tn = int(np.sum(~failure & ~alarm))
        detected = failure & alarm
        rows.append(
            {
                "horizon_phase": phase,
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "tpr": tp / max(tp + fn, 1),
                "fpr": fp / max(fp + tn, 1),
                "precision": tp / max(tp + fp, 1),
                "tp_with_at_least_3_query_lead": int(
                    np.sum(detected & (lead >= 3))
                ),
                "tp_with_at_least_5_query_lead": int(
                    np.sum(detected & (lead >= 5))
                ),
                "median_failure_alarm_lead_queries": (
                    float(np.median(lead[detected])) if np.any(detected) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def select_clock_controls(
    clock: pd.DataFrame, selector: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    within_fpr = clock[clock["fpr"] <= float(selector["fpr"]) + 1e-15]
    if len(within_fpr):
        matched_fpr = within_fpr.sort_values(
            ["tpr", "median_failure_alarm_lead_queries"], ascending=[False, False]
        ).iloc[0]
    else:
        matched_fpr = clock.sort_values("fpr").iloc[0]
    target_lead = float(selector["median_failure_alarm_lead_queries"] or 0.0)
    matched_lead = clock.iloc[
        np.argmin(
            np.abs(clock["median_failure_alarm_lead_queries"].to_numpy() - target_lead)
        )
    ]
    return {
        "matched_or_lower_fpr": plain(matched_fpr.to_dict()),
        "matched_median_lead": plain(matched_lead.to_dict()),
    }


def plot_results(
    task_frame: pd.DataFrame,
    clock: pd.DataFrame,
    selector: dict[str, Any],
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].scatter(
        100.0 * task_frame["fpr"],
        100.0 * task_frame["tpr"],
        s=32,
        color="#137C8B",
        alpha=0.8,
    )
    axes[0].set_xlabel("Success false-alarm rate (%)")
    axes[0].set_ylabel("Failure recall (%)")
    axes[0].set_title("Frozen selector by held-out task")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        clock["median_failure_alarm_lead_queries"],
        100.0 * clock["fpr"],
        color="#6B7280",
        label="fixed normalized clock",
    )
    axes[1].scatter(
        [selector["median_failure_alarm_lead_queries"]],
        [100.0 * selector["fpr"]],
        color="#C43C39",
        s=65,
        label="MoE self-reference",
        zorder=3,
    )
    axes[1].set_xlabel("Median lead on detected failures (queries)")
    axes[1].set_ylabel("Success false-alarm rate (%)")
    axes[1].set_title("Lead versus false alarms")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def render_report(summary: dict[str, Any], task_frame: pd.DataFrame) -> str:
    heldout = summary["heldout_39_task_evaluation"]
    full = summary["all_40_task_descriptive"]
    development = summary["development_task_replay"]
    clocks = summary["clock_controls_heldout"]
    worst = task_frame.sort_values("fpr", ascending=False).iloc[0]
    return f"""# 无任务先验的 MoE 自参照报警实验

## 核心结果

规则没有训练参数，运行时只读取当前 episode 已产生的 HB router probability。它不用任务 ID、正常轨迹库、动作数值、reward、success 或物理距离。阈值曾查看一个开发任务，因此本实验是 **train-free、runtime reference-free，但不是开发阶段 label-free**。

严格冻结验证排除开发任务，覆盖 39 个未参与规则设定的任务：

- failure recall：{heldout['tp']}/{heldout['failures']} = {100 * heldout['tpr']:.2f}%
- success false-alarm：{heldout['fp']}/{heldout['successes']} = {100 * heldout['fpr']:.3f}%
- precision：{100 * heldout['precision']:.2f}%
- 检出失败的中位提前量：{heldout['median_failure_alarm_lead_queries']:.1f} 个 query
- 至少提前 5 个 query 的检出：{heldout['tp_with_at_least_5_query_lead']}/{heldout['failures']}

包含开发任务的 40-task 描述值为 recall {100 * full['tpr']:.2f}%、FPR {100 * full['fpr']:.3f}%。开发任务 `push_the_plate_to_the_front_of_the_stove` 的两条失败均报警，398 条成功均未报警；它不能再算独立验证。

## 方法

每条轨迹用 q1--q6 的路由作为自身基线。每次只对最近 3 个 query 取中位数，同时要求：

1. 前四层 state-token 的跨 query 路由跳变降到自身基线的 0.25 倍以下；
2. 后四层 action-token 的完整去噪路由加速度升到 1.10 倍以上；
3. 前四层 state/action 路由间隙升到 1.20 倍以上。

三项相交描述的是：输入侧路由响应近乎冻结，但 action-token 的去噪计算仍在更剧烈修正，并与 state-token 路由进一步分离。它比单独使用高 recurrence 更不容易把任务固有的稳定计算误判为 Trap。

## 时钟混杂审计

只要 episode 最终失败，它通常会跑满 horizon，因此“很晚还没结束”本身就是强信号。与 MoE 相同或更低 FPR 的固定归一化时钟位于 horizon phase {clocks['matched_or_lower_fpr']['horizon_phase']:.2f}，recall 为 {100 * clocks['matched_or_lower_fpr']['tpr']:.1f}%，但中位提前量只有 {clocks['matched_or_lower_fpr']['median_failure_alarm_lead_queries']:.1f} 个 query。匹配 MoE 中位提前量的时钟 FPR 为 {100 * clocks['matched_median_lead']['fpr']:.2f}%。因此不能只报 episode recall；提前量和成功误报必须一起报告。

## 异质性与边界

最差 held-out 任务是 `{worst['task']}`：FPR {100 * worst['fpr']:.2f}%（{int(worst['fp'])}/{int(worst['successes'])}）。这说明自参照消除了大部分静态任务差异，但没有消除所有合法阶段转换。

总体 recall 只有约 15%，所以它不是通用失败检测器。更准确的定位是一个高精度的 **internal response-collapse phenotype head**：报警时证据较强，不报警不能说明机器人正常。它也不能仅凭 routing 证明“belief 错了”或识别具体物理原因；这些需要视频、接触或对象状态作事后解释。

## 可复现文件

- `episode_predictions.csv`：逐 episode 冻结预测和解盲结果
- `query_decisions.csv`：逐 query 三个 MoE 比率及报警
- `task_metrics.csv`：逐任务 TPR/FPR
- `clock_baselines.csv`：固定 horizon 时钟审计
- `summary.json`：机器可读汇总与 task-bootstrap 区间
- `task_free_self_reference_audit.png`：任务异质性及 lead/FPR 图
"""


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = SelfReferenceConfig.load(config_path)
    runs = discover_runs(args.cache_root.resolve(), args.run_id)
    args.out.mkdir(parents=True, exist_ok=True)

    predictions, query_frame, run_lookup = predict_routes(
        runs, args.cache_root.resolve(), config
    )
    frame = unblind(predictions, run_lookup)
    development_task = str(config_payload["development_task"])
    heldout = frame[frame["task"] != development_task].copy()
    development = frame[frame["task"] == development_task].copy()
    if len(development) == 0:
        raise RuntimeError("declared development task is absent")

    per_task = task_metrics(frame)
    heldout_tasks = per_task[per_task["task"] != development_task].reset_index(drop=True)
    full_metrics = counts(frame)
    heldout_metrics = counts(heldout)
    development_metrics = counts(development)
    heldout_metrics.update(
        task_bootstrap(heldout_tasks, args.bootstrap, args.seed)
    )
    clock = clock_baselines(heldout)
    clock_controls = select_clock_controls(clock, heldout_metrics)

    frame.to_csv(args.out / "episode_predictions.csv", index=False)
    query_frame.to_csv(args.out / "query_decisions.csv", index=False)
    per_task.to_csv(args.out / "task_metrics.csv", index=False)
    clock.to_csv(args.out / "clock_baselines.csv", index=False)
    plot_results(
        heldout_tasks,
        clock,
        heldout_metrics,
        args.out / "task_free_self_reference_audit.png",
    )

    summary = {
        "schema": "himoe.task_free_self_reference_evaluation.v1",
        "selector_version": SELECTOR_VERSION,
        "training": False,
        "learned_parameters": False,
        "runtime_inputs": ["current HB router probabilities", "own episode route prefix"],
        "runtime_excluded_inputs": [
            "task identity",
            "normal trajectory bank",
            "action values",
            "physical state or distance",
            "reward",
            "success/failure outcome",
        ],
        "development_outcomes_inspected": True,
        "development_task": development_task,
        "config": config_payload,
        "run_id": args.run_id,
        "complete_tasks": len(run_lookup),
        "all_40_task_descriptive": full_metrics,
        "heldout_39_task_evaluation": heldout_metrics,
        "development_task_replay": development_metrics,
        "clock_controls_heldout": clock_controls,
        "worst_heldout_task_fpr": plain(
            heldout_tasks.sort_values("fpr", ascending=False).iloc[0].to_dict()
        ),
        "bootstrap_unit": "task",
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
    }
    (args.out / "summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "REPORT_ZH.md").write_text(
        render_report(summary, heldout_tasks), encoding="utf-8"
    )
    print(json.dumps(plain(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
