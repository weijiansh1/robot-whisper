#!/usr/bin/env python3
"""Select MoE probability thresholds under an explicit onset-timing constraint."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_hub_phenotype_atlas as atlas


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "configs/timing_constrained_probability_alarm.json"
DEFAULT_OUTPUT = (
    PACKAGE_ROOT / "results/trainfree_trap_probability/timing_constrained_alarm"
)
CALIBRATORS = ("A_dense", "B_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-physical", action="store_true")
    return parser.parse_args()


def resolve_workspace(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else WORKSPACE_ROOT / value


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    half = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - half, center + half


def add_onset(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["onset_query"] = np.nan
    result["onset_type"] = ""
    for component in config["onset_components"]:
        label = component["label_column"]
        phase = component["phase_column"]
        selected = (
            result["failure"].astype(bool)
            & result[label].astype(bool)
            & (result[phase] >= 0.0)
        )
        query = np.ceil(
            result.loc[selected, phase]
            * (result.loc[selected, "episode_length"] - 1)
        ).astype(int)
        previous = result.loc[selected, "onset_query"]
        take = previous.isna() | (query.to_numpy() < previous.fillna(np.inf).to_numpy())
        indices = result.index[selected][take]
        result.loc[indices, "onset_query"] = query.loc[indices]
        result.loc[indices, "onset_type"] = component["event"]
    return result


def build_physical_labels(
    config: dict[str, Any], output: Path, rebuild: bool
) -> pd.DataFrame:
    path = output / "tables/physical_onset_labels.csv"
    if path.exists() and not rebuild:
        cached = pd.read_csv(path)
        if cached["task"].nunique() == 40 and len(cached) == 16000:
            return cached

    cache_root = resolve_workspace(config["cache_root"])
    runs = atlas.discover_complete_runs(cache_root, config["run_id"])
    frame, _ = atlas.build_physical_frame(runs, cache_root)
    frame = add_onset(frame, config)
    keep = [
        "task",
        "episode",
        "init_state_id",
        "flow_noise_seed",
        "episode_length",
        "failure",
        "onset_query",
        "onset_type",
        *[item["label_column"] for item in config["onset_components"]],
        *[item["phase_column"] for item in config["onset_components"]],
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_csv(path, index=False)
    return frame[keep]


def choose_threshold(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    probability_column: str,
    budget: float,
) -> tuple[float, float, bool]:
    success = episodes.loc[~episodes["failure"].astype(bool), ["key"]]
    maxima = (
        predictions[predictions["key"].isin(set(success["key"]))]
        .groupby("key")[probability_column]
        .max()
        .reindex(success["key"], fill_value=-np.inf)
    )
    candidates = np.sort(predictions[probability_column].dropna().unique())
    for threshold in candidates:
        fpr = float((maxima >= threshold).mean())
        if fpr <= budget + 1e-15:
            return float(threshold), fpr, True
    threshold = float(np.nextafter(predictions[probability_column].max(), np.inf))
    return threshold, 0.0, False


def alarm_queries(
    predictions: pd.DataFrame, probability_column: str, threshold: float
) -> dict[str, np.ndarray]:
    selected = predictions[predictions[probability_column] >= threshold]
    return {
        key: group["query"].to_numpy(dtype=int)
        for key, group in selected.groupby("key", sort=False)
    }


def timing_metrics(
    episodes: pd.DataFrame,
    alarms: dict[str, np.ndarray],
    minimum_query: int,
    lead_window: int,
) -> dict[str, Any]:
    success = episodes[~episodes["failure"].astype(bool)]
    success_alarms = sum(key in alarms and len(alarms[key]) > 0 for key in success["key"])
    success_low, success_high = wilson(success_alarms, len(success))

    all_events = episodes[episodes["onset_query"].notna()]
    events = all_events[all_events["onset_query"] >= minimum_query]
    timely = near_first = near_active = premature = late = no_alarm = 0
    leads: list[int] = []
    rows: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        query = np.sort(alarms.get(row.key, np.empty(0, dtype=int)))
        first = int(query[0]) if len(query) else None
        is_timely = first is not None and first <= row.onset_query
        lead = int(row.onset_query - first) if is_timely else None
        is_near_first = is_timely and lead <= lead_window
        is_near_active = bool(
            np.any(
                (query >= row.onset_query - lead_window)
                & (query <= row.onset_query)
            )
        )
        timely += int(is_timely)
        near_first += int(is_near_first)
        near_active += int(is_near_active)
        premature += int(is_timely and not is_near_first)
        late += int(first is not None and first > row.onset_query)
        no_alarm += int(first is None)
        if lead is not None:
            leads.append(lead)
        rows.append(
            {
                "key": row.key,
                "onset_type": row.onset_type,
                "onset_query": int(row.onset_query),
                "first_alarm_query": first,
                "timely": is_timely,
                "near_first": is_near_first,
                "near_active": is_near_active,
                "lead_queries": lead,
            }
        )

    timely_low, timely_high = wilson(timely, len(events))
    near_low, near_high = wilson(near_first, len(events))
    return {
        "success_episodes": len(success),
        "success_alarm_episodes": success_alarms,
        "success_fpr": success_alarms / max(len(success), 1),
        "success_fpr_ci_low": success_low,
        "success_fpr_ci_high": success_high,
        "all_event_episodes": len(all_events),
        "eligible_event_episodes": len(events),
        "events_before_minimum_query": len(all_events) - len(events),
        "timely_events": timely,
        "timely_recall": timely / max(len(events), 1),
        "timely_recall_ci_low": timely_low,
        "timely_recall_ci_high": timely_high,
        "near_first_events": near_first,
        "near_first_recall": near_first / max(len(events), 1),
        "near_first_recall_ci_low": near_low,
        "near_first_recall_ci_high": near_high,
        "near_active_events": near_active,
        "near_active_recall": near_active / max(len(events), 1),
        "premature_events": premature,
        "late_events": late,
        "no_alarm_events": no_alarm,
        "timely_lead_median": float(np.median(leads)) if leads else np.nan,
        "timely_lead_p90": float(np.quantile(leads, 0.9)) if leads else np.nan,
        "event_rows": rows,
    }


def matched_clock_query(
    episodes: pd.DataFrame, target_fpr: float, minimum_query: int
) -> int:
    success_lengths = episodes.loc[
        ~episodes["failure"].astype(bool), "episode_length"
    ].to_numpy(dtype=int)
    maximum = int(episodes["episode_length"].max())
    for query in range(minimum_query, maximum + 1):
        if float(np.mean(success_lengths > query)) <= target_fpr + 1e-15:
            return query
    return maximum + 1


def clock_alarms(episodes: pd.DataFrame, query: int) -> dict[str, np.ndarray]:
    return {
        row.key: np.asarray([query], dtype=int)
        for row in episodes.itertuples(index=False)
        if row.episode_length > query
    }


def event_type_rows(
    metrics: dict[str, Any], base: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = pd.DataFrame(metrics["event_rows"])
    output: list[dict[str, Any]] = []
    if rows.empty:
        return output
    for event_type, group in rows.groupby("onset_type", sort=True):
        output.append(
            {
                **base,
                "onset_type": event_type,
                "eligible_events": len(group),
                "timely_events": int(group["timely"].sum()),
                "timely_recall": float(group["timely"].mean()),
                "near_first_events": int(group["near_first"].sum()),
                "near_first_recall": float(group["near_first"].mean()),
                "near_active_events": int(group["near_active"].sum()),
                "near_active_recall": float(group["near_active"].mean()),
            }
        )
    return output


def strip_event_rows(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in metrics.items()
        if key != "event_rows"
    }


def aligned_probability(
    predictions: pd.DataFrame, episodes: pd.DataFrame, radius: int = 6
) -> pd.DataFrame:
    events = episodes[episodes["onset_query"].notna()][
        ["key", "onset_query", "onset_type"]
    ]
    frame = predictions.merge(events, on="key", how="inner", validate="many_to_one")
    frame["relative_query"] = frame["query"] - frame["onset_query"]
    frame = frame[frame["relative_query"].between(-radius, radius)]
    rows: list[dict[str, Any]] = []
    for calibrator in CALIBRATORS:
        column = f"probability_{calibrator}"
        for relative, group in frame.groupby("relative_query", sort=True):
            rows.append(
                {
                    "calibrator": calibrator,
                    "relative_query": int(relative),
                    "rows": len(group),
                    "episodes": group["key"].nunique(),
                    "mean_probability": float(group[column].mean()),
                    "median_probability": float(group[column].median()),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    operating: pd.DataFrame, aligned: pd.DataFrame, output: Path
) -> None:
    colors = {"A_dense": "#287a58", "B_proxy": "#b94b3c"}
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for calibrator in CALIBRATORS:
        group = operating[
            (operating["split"] == "development_37_tasks")
            & (operating["calibrator"] == calibrator)
        ].sort_values("success_fpr_budget")
        axes[0].plot(
            group["success_fpr"],
            group["timely_recall"],
            marker="o",
            color=colors[calibrator],
            label=calibrator,
        )
        axes[1].plot(
            group["success_fpr_budget"],
            group["near_first_recall"],
            marker="o",
            color=colors[calibrator],
            label=f"{calibrator} MoE",
        )
    clock = operating[operating["split"] == "development_37_tasks"].groupby(
        "success_fpr_budget", as_index=False
    )["clock_near_first_recall"].mean()
    axes[1].plot(
        clock["success_fpr_budget"],
        clock["clock_near_first_recall"],
        marker="s",
        color="#777777",
        linestyle="--",
        label="matched clock",
    )
    for calibrator in CALIBRATORS:
        group = aligned[aligned["calibrator"] == calibrator]
        axes[2].plot(
            group["relative_query"],
            group["mean_probability"],
            marker="o",
            color=colors[calibrator],
            label=calibrator,
        )
    axes[0].set_xlabel("Successful-episode false alarm rate")
    axes[0].set_ylabel("First alarm at/before onset recall")
    axes[0].set_title("Development operating frontier")
    axes[1].set_xlabel("FPR budget")
    axes[1].set_ylabel("First alarm in [-2, 0] recall")
    axes[1].set_title("Near-onset timing")
    axes[2].axvline(0, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Query relative to physical onset")
    axes[2].set_ylabel("Mean probability")
    axes[2].set_title("Heldout event alignment")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.legend(frameon=False)
    figure.tight_layout()
    path = output / "figures/timing_constrained_probability_alarm.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_report(summary: dict[str, Any], operating: pd.DataFrame) -> str:
    rows = operating[
        (operating["split"] == "heldout_3_tasks")
        & (operating["success_fpr_budget"].isin([0.01, 0.02, 0.05]))
    ]
    lines = []
    for row in rows.itertuples(index=False):
        threshold = (
            f"{row.threshold:.3%}" if row.threshold_attainable else "> max (zero alarm)"
        )
        lines.append(
            f"| {row.calibrator} | {row.success_fpr_budget:.1%} | {threshold} | "
            f"{row.success_fpr:.2%} | {row.timely_recall:.2%} | "
            f"{row.near_first_recall:.2%} | {row.timely_lead_median:g} | "
            f"q{int(row.clock_query)} / {row.clock_timely_recall:.2%} |"
        )
    return f"""# 时序约束的 MoE 概率报警阈值

## 结论

75% 不是必须的；阈值应由允许的成功轨迹误报预算决定。但当前概率标量没有形成稳定的 onset-localized operating point。降低阈值能增加“最终在 onset 前报过一次”的比例，主要代价是很早报警，而不是在 Trap 出现前两次重规划内形成稳定抬升。

开发集为 37 个较早完成任务；3 个后完成任务保持留出。物理量只用于构造离线 onset 真值，报警输入仍只有冻结的 MoE 概率。复合 onset 只覆盖 stagnation、goal regression、goal approach-leave 和 subtask undo，不把全部 endpoint failure 当成 Trap。

| 概率表 | 开发成功 FPR 预算 | 冻结阈值 | 留出成功 FPR | 留出 onset 前/当下召回 | 留出首次报警在 [-2,0] 召回 | 及时命中中位提前量 | 等误报时钟 / 及时召回 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

留出集含 {summary['heldout']['episodes']} 条轨迹、{summary['heldout']['failures']} 条最终失败和 {summary['heldout']['eligible_onset_events']} 个 q12 后可评价 onset。小样本 Wilson 区间见 `summary.json` 和 `operating_points.csv`。

## 如何解释

- `timely_recall` 只要求首次报警不晚于 onset，可能提前很多。
- `near_first_recall` 更严格：首次报警必须落在 onset 前 2 个 query 到 onset 当下。
- `near_active_recall` 检查该窗口里概率是否再次超过阈值。
- 固定时钟完全不看 MoE；其 query 在开发集按相同或更低误报率选择。迁移到留出任务后，两者的实际误报率都可能漂移。若时钟的时序召回同样或更好，不能把 MoE 报警解释成 Trap precursor。

因此现在不应从 75% 简单改成 50%、25% 后上线。正确决策是先要求一个阈值在留出任务上同时满足成功 FPR、near-onset recall 和相对时钟增益，再冻结部署。
"""


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    physical = build_physical_labels(config, output, args.rebuild_physical)
    physical = add_onset(physical, config)
    predictions = pd.read_csv(resolve_workspace(config["prediction_file"]))
    predictions["key"] = (
        predictions["task"] + "#" + predictions["episode"].astype(str)
    )
    physical["key"] = physical["task"] + "#" + physical["episode"].astype(str)
    minimum_query = int(config["minimum_query"])
    if physical["task"].nunique() != 40:
        raise ValueError("expected physical labels for all 40 cache_new tasks")
    missing_prediction_tasks = sorted(
        set(physical["task"]) - set(predictions["task"])
    )
    for task in missing_prediction_tasks:
        if int(physical.loc[physical["task"] == task, "episode_length"].max()) > minimum_query:
            raise ValueError(f"unexpected scoreable task without predictions: {task}")

    heldout_tasks = set(config["heldout_tasks"])
    splits = {
        "development_37_tasks": physical[~physical["task"].isin(heldout_tasks)],
        "heldout_3_tasks": physical[physical["task"].isin(heldout_tasks)],
    }
    development_predictions = predictions[
        ~predictions["task"].isin(heldout_tasks)
    ]
    lead_window = int(config["near_onset_lead_queries"])
    operating_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []

    for calibrator in CALIBRATORS:
        column = f"probability_{calibrator}"
        for budget in map(float, config["success_false_alarm_budgets"]):
            threshold, calibration_fpr, attainable = choose_threshold(
                development_predictions,
                splits["development_37_tasks"],
                column,
                budget,
            )
            development_alarm = alarm_queries(predictions, column, threshold)
            clock_query = matched_clock_query(
                splits["development_37_tasks"], calibration_fpr, minimum_query
            )
            for split_name, episodes in splits.items():
                metrics = timing_metrics(
                    episodes, development_alarm, minimum_query, lead_window
                )
                clock_metrics = timing_metrics(
                    episodes,
                    clock_alarms(episodes, clock_query),
                    minimum_query,
                    lead_window,
                )
                base = {
                    "calibrator": calibrator,
                    "success_fpr_budget": budget,
                    "threshold": threshold,
                    "threshold_attainable": attainable,
                    "development_calibration_fpr": calibration_fpr,
                    "split": split_name,
                    "clock_query": clock_query,
                }
                operating_rows.append(
                    {
                        **base,
                        **strip_event_rows(metrics),
                        **strip_event_rows(clock_metrics, "clock_"),
                    }
                )
                event_rows.extend(event_type_rows(metrics, base))

    fixed_rows: list[dict[str, Any]] = []
    for calibrator in CALIBRATORS:
        column = f"probability_{calibrator}"
        for threshold in map(float, config["reported_fixed_thresholds"]):
            alarms = alarm_queries(predictions, column, threshold)
            for split_name, episodes in splits.items():
                fixed_rows.append(
                    {
                        "calibrator": calibrator,
                        "threshold": threshold,
                        "split": split_name,
                        **strip_event_rows(
                            timing_metrics(
                                episodes, alarms, minimum_query, lead_window
                            )
                        ),
                    }
                )

    operating = pd.DataFrame(operating_rows)
    aligned = aligned_probability(
        predictions[predictions["task"].isin(heldout_tasks)],
        splits["heldout_3_tasks"],
    )
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    operating.to_csv(table_dir / "operating_points.csv", index=False)
    pd.DataFrame(fixed_rows).to_csv(
        table_dir / "fixed_threshold_timing.csv", index=False
    )
    pd.DataFrame(event_rows).to_csv(
        table_dir / "event_type_operating_points.csv", index=False
    )
    aligned.to_csv(table_dir / "heldout_onset_aligned_probability.csv", index=False)

    within_five = operating[
        operating["success_fpr_budget"] <= 0.05 + 1e-15
    ]
    development_within_five = within_five[
        within_five["split"] == "development_37_tasks"
    ]
    heldout_within_five = within_five[
        within_five["split"] == "heldout_3_tasks"
    ]
    summary = {
        "schema": "himoe.timing_constrained_probability_alarm.v1",
        "status": "complete",
        "training": False,
        "gradient_optimization": False,
        "learned_feature_weights": False,
        "runtime_input": config["runtime_input"],
        "physical_state_used_by_alarm": False,
        "physical_state_used_for_posthoc_onset_only": True,
        "threshold_selection": "lowest attainable probability satisfying the development successful-episode false-alarm budget",
        "minimum_query": minimum_query,
        "near_onset_window": [-lead_window, 0],
        "tasks_without_query_12_predictions": missing_prediction_tasks,
        "development": {
            "tasks": int(splits["development_37_tasks"]["task"].nunique()),
            "episodes": len(splits["development_37_tasks"]),
            "failures": int(splits["development_37_tasks"]["failure"].sum()),
            "onset_events": int(
                splits["development_37_tasks"]["onset_query"].notna().sum()
            ),
            "eligible_onset_events": int(
                (splits["development_37_tasks"]["onset_query"] >= minimum_query).sum()
            ),
        },
        "heldout": {
            "tasks": int(splits["heldout_3_tasks"]["task"].nunique()),
            "episodes": len(splits["heldout_3_tasks"]),
            "failures": int(splits["heldout_3_tasks"]["failure"].sum()),
            "onset_events": int(
                splits["heldout_3_tasks"]["onset_query"].notna().sum()
            ),
            "eligible_onset_events": int(
                (splits["heldout_3_tasks"]["onset_query"] >= minimum_query).sum()
            ),
            "tasks_list": sorted(heldout_tasks),
        },
        "event_scope_warning": config["event_scope_warning"],
        "heldout_warning": config["heldout_warning"],
        "suitable_onset_localized_threshold_found": False,
        "selection_conclusion": "No frozen threshold provides a stable first alarm in [-2, 0] with an advantage over the development-matched query clock on both splits.",
        "best_near_first_recall_with_development_fpr_budget_le_005": {
            "development": float(development_within_five["near_first_recall"].max()),
            "heldout": float(heldout_within_five["near_first_recall"].max()),
        },
        "operating_points": plain(operating_rows),
    }
    make_figure(operating, aligned, output)
    write_json(output / "summary.json", summary)
    (output / "REPORT_ZH.md").write_text(
        render_report(summary, operating), encoding="utf-8"
    )
    print(json.dumps(plain(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
