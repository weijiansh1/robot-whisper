#!/usr/bin/env python3
"""Summarize frozen v4 first-alarm timing by task and normalized phase."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_OUTPUT = HERE / "results"
FIRST_COLUMN = "first_dual_regime_or_query"
PHASE_BUCKETS = (
    ("phase_00_25", 0.0, 25.0),
    ("phase_25_50", 25.0, 50.0),
    ("phase_50_75", 50.0, 75.0),
    ("phase_75_100", 75.0, 100.000001),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def quantile(values: np.ndarray, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if len(values) else float("nan")


def load_cohorts(root: Path) -> pd.DataFrame:
    cache_new = pd.read_csv(root / "results/cache_new_v4/episode_alarms.csv")
    cache_new["risk"] = cache_new["original_failure"].astype(bool)
    cache_new["outcome"] = np.where(
        ~cache_new["risk"],
        "timely_success",
        np.where(cache_new["late_success_plus10_queries"], "late_success", "failure"),
    )

    cache_old = pd.read_csv(root / "results/cache16x32_v4/episode_alarms.csv")
    cache_old["cohort"] = "cache_right16x32"
    cache_old["risk"] = cache_old["failure"].astype(bool)
    cache_old["outcome"] = np.where(cache_old["risk"], "raw_failure", "success")
    cache_old["suite"] = cache_old["task"].str.split("/", n=1).str[0]

    common = [
        "cohort",
        "suite",
        "task",
        "episode",
        "length",
        "risk",
        "outcome",
        "first_lock_layer_median_q75_k4_query",
        "first_instability_L5_q80_k8_query",
        FIRST_COLUMN,
    ]
    frame = pd.concat([cache_new[common], cache_old[common]], ignore_index=True)
    frame["alarm"] = frame[FIRST_COLUMN] >= 0
    denominator = np.maximum(frame["length"].to_numpy(dtype=float) - 1.0, 1.0)
    frame["alarm_phase_pct"] = np.where(
        frame["alarm"], 100.0 * frame[FIRST_COLUMN] / denominator, np.nan
    )
    frame["remaining_phase_pct"] = np.where(
        frame["alarm"], 100.0 - frame["alarm_phase_pct"], np.nan
    )
    frame["first_alarm_query_1based"] = np.where(
        frame["alarm"], frame[FIRST_COLUMN] + 1, np.nan
    )
    lock = frame["first_lock_layer_median_q75_k4_query"].to_numpy(dtype=int)
    instability = frame["first_instability_L5_q80_k8_query"].to_numpy(dtype=int)
    branch = np.full(len(frame), "none", dtype=object)
    branch[(lock >= 0) & ((instability < 0) | (lock < instability))] = "lock"
    branch[(instability >= 0) & ((lock < 0) | (instability < lock))] = "instability"
    branch[(lock >= 0) & (lock == instability)] = "lock+instability"
    frame["first_branch"] = branch
    return frame


def timing_row(block: pd.DataFrame, group: str) -> dict[str, Any]:
    alarm = block["alarm"].to_numpy(dtype=bool)
    risk = block["risk"].to_numpy(dtype=bool)
    true_alarm = alarm & risk
    false_alarm = alarm & ~risk
    phase = block["alarm_phase_pct"].to_numpy(dtype=float)
    first = block["first_alarm_query_1based"].to_numpy(dtype=float)
    tp = int(true_alarm.sum())
    fp = int(false_alarm.sum())
    row: dict[str, Any] = {
        "group": group,
        "episodes": len(block),
        "risk_n": int(risk.sum()),
        "nonrisk_n": int((~risk).sum()),
        "alarm_n": int(alarm.sum()),
        "tp": tp,
        "fp": fp,
        "risk_recall": ratio(tp, risk.sum()),
        "precision": ratio(tp, tp + fp),
        "nonrisk_fpr": ratio(fp, (~risk).sum()),
        "first_alarm_query_1based_q25": quantile(first[alarm], 0.25),
        "first_alarm_query_1based_median": quantile(first[alarm], 0.50),
        "first_alarm_query_1based_q75": quantile(first[alarm], 0.75),
        "alarm_phase_pct_q25": quantile(phase[alarm], 0.25),
        "alarm_phase_pct_median": quantile(phase[alarm], 0.50),
        "alarm_phase_pct_q75": quantile(phase[alarm], 0.75),
        "true_alarm_phase_pct_median": quantile(phase[true_alarm], 0.50),
        "false_alarm_phase_pct_median": quantile(phase[false_alarm], 0.50),
        "remaining_phase_pct_median": quantile(100.0 - phase[alarm], 0.50),
    }
    for name, low, high in PHASE_BUCKETS:
        count = int((alarm & (phase >= low) & (phase < high)).sum())
        row[f"{name}_n"] = count
        row[f"{name}_among_alarms"] = ratio(count, alarm.sum())
    return row


def grouped(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouper: str | list[str] = columns[0] if len(columns) == 1 else columns
    for key, block in frame.groupby(grouper, sort=True, dropna=False):
        keys = (key,) if len(columns) == 1 else key
        labels = {column: value for column, value in zip(columns, keys, strict=True)}
        name = " | ".join(str(value) for value in keys)
        row = timing_row(block.reset_index(drop=True), name)
        row.update(labels)
        rows.append(row)
    return pd.DataFrame(rows)


def task_percent_table(by_task: pd.DataFrame) -> pd.DataFrame:
    output = by_task[
        [
            "cohort",
            "task",
            "episodes",
            "risk_n",
            "alarm_n",
            "tp",
            "fp",
            "risk_recall",
            "precision",
            "first_alarm_query_1based_median",
            "alarm_phase_pct_q25",
            "alarm_phase_pct_median",
            "alarm_phase_pct_q75",
            "phase_00_25_among_alarms",
            "phase_25_50_among_alarms",
            "phase_50_75_among_alarms",
            "phase_75_100_among_alarms",
        ]
    ].copy()
    output = output.rename(
        columns={
            "risk_recall": "risk_recall_pct",
            "precision": "precision_pct",
            "phase_00_25_among_alarms": "alarms_in_phase_00_25_pct",
            "phase_25_50_among_alarms": "alarms_in_phase_25_50_pct",
            "phase_50_75_among_alarms": "alarms_in_phase_50_75_pct",
            "phase_75_100_among_alarms": "alarms_in_phase_75_100_pct",
        }
    )
    fraction_columns = [
        "risk_recall_pct",
        "precision_pct",
        "alarms_in_phase_00_25_pct",
        "alarms_in_phase_25_50_pct",
        "alarms_in_phase_50_75_pct",
        "alarms_in_phase_75_100_pct",
    ]
    output[fraction_columns] = 100.0 * output[fraction_columns]
    numeric = output.select_dtypes(include=[np.number]).columns
    output[numeric] = output[numeric].round(2)
    return output


def main() -> None:
    args = parse_args()
    frame = load_cohorts(args.root)
    overall = grouped(frame, ["cohort"])
    by_suite = grouped(frame, ["cohort", "suite"])
    by_task = grouped(frame, ["cohort", "task"])
    by_outcome = grouped(frame, ["cohort", "outcome"])
    by_branch = grouped(frame[frame["alarm"]], ["cohort", "first_branch"])
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "episode_alarm_timing.csv", index=False)
    overall.to_csv(args.output / "alarm_timing_overall.csv", index=False)
    by_suite.to_csv(args.output / "alarm_timing_by_suite.csv", index=False)
    by_task.to_csv(args.output / "alarm_timing_by_task.csv", index=False)
    task_percent_table(by_task).to_csv(
        args.output / "alarm_timing_by_task_percent.csv", index=False
    )
    by_outcome.to_csv(args.output / "alarm_timing_by_outcome.csv", index=False)
    by_branch.to_csv(args.output / "alarm_timing_by_branch.csv", index=False)

    summary = {
        "schema": "himoe.dual_regime_v4.alarm_timing.v1",
        "phase_definition": "100 * zero_based_first_alarm_query / (rollout_length - 1)",
        "phase_is_posthoc_only": True,
        "first_alarm_query_in_tables_is_one_based": True,
        "phase_buckets": [name for name, _, _ in PHASE_BUCKETS],
        "overall": overall.to_dict(orient="records"),
    }
    (args.output / "alarm_timing_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        overall[
            [
                "cohort",
                "alarm_n",
                "tp",
                "fp",
                "first_alarm_query_1based_median",
                "alarm_phase_pct_q25",
                "alarm_phase_pct_median",
                "alarm_phase_pct_q75",
                "phase_00_25_among_alarms",
                "phase_25_50_among_alarms",
                "phase_50_75_among_alarms",
                "phase_75_100_among_alarms",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
