#!/usr/bin/env python3
"""Calibrate train-free dynamic MoE alarms at the episode level."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(HERE))

from evaluate_dynamic_axes import Task, inventory, load_profile  # noqa: E402


SCORE_ROOT = BUNDLE / "results/dynamic_evaluation/scores"
OUTPUT = BUNDLE / "results/dynamic_risk"
CHANNELS = {
    "predeclared_loop": "loop",
    "discovery_full_loop": "loop",
    "discovery_history_loop": "loop",
    "predeclared_static": "static",
    "discovery_full_static": "static",
    "discovery_history_static": "static",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ltt-delta", type=float, default=0.05)
    return parser.parse_args()


def task_key(task: Task) -> tuple[str, str, str]:
    return task.corpus, task.suite, task.task


def event_map(task: Task) -> dict[int, dict[str, int | str]]:
    return {int(event["episode_id"]): event for event in task.events}


def is_healthy(event: dict[str, int | str]) -> bool:
    return (
        int(event["success"]) == 1
        and int(event["loop_onset_q"]) < 0
        and int(event["static_onset_q"]) < 0
    )


def onset(task: Task, event: dict[str, int | str], event_type: str) -> int:
    if event_type == "loop" and task.task in {
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "open_the_middle_drawer_of_the_cabinet",
    }:
        return -1
    return int(event[f"{event_type}_onset_q"])


def load_scores(task: Task) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    _names, _matrix, metadata = load_profile(task)
    path = SCORE_ROOT / task.corpus / task.suite / f"{task.task}.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = archive["score_names"].astype(str).tolist()
        matrix = np.asarray(archive["scores"], dtype=np.float64)
    missing = set(CHANNELS) - set(names)
    if missing:
        raise ValueError(f"missing dynamic score channels in {path}: {sorted(missing)}")
    return {name: matrix[:, names.index(name)] for name in CHANNELS}, metadata


def split_tasks(tasks: list[Task]) -> tuple[set[tuple[str, str, str]], set[tuple[str, str, str]]]:
    calibration: set[tuple[str, str, str]] = set()
    test: set[tuple[str, str, str]] = set()
    for suite in sorted({task.suite for task in tasks}):
        members = sorted((task for task in tasks if task.suite == suite), key=lambda task: task.task)
        for index, task in enumerate(members):
            (calibration if index % 2 == 0 else test).add(task_key(task))
    return calibration, test


def include_repeat(event: dict[str, int | str], partition: str | None) -> bool:
    if partition is None:
        return True
    repeat = int(event["repeat"])
    return repeat < 4 if partition == "repeat_0_3" else repeat >= 4


def rows_for_episode(metadata: dict[str, np.ndarray], episode: int) -> np.ndarray:
    rows = np.flatnonzero(metadata["episode_id"] == episode)
    return rows[np.argsort(metadata["query"][rows], kind="stable")]


def healthy_maxima(
    tasks: list[Task],
    loaded: dict[tuple[str, str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    selected: set[tuple[str, str, str]],
    channel: str,
    partition: str | None = None,
) -> np.ndarray:
    maxima = []
    for task in tasks:
        key = task_key(task)
        if key not in selected:
            continue
        scores, metadata = loaded[key]
        for episode, event in event_map(task).items():
            if not is_healthy(event) or not include_repeat(event, partition):
                continue
            values = scores[channel][rows_for_episode(metadata, episode)]
            values = values[np.isfinite(values)]
            if len(values):
                maxima.append(float(values.max()))
    return np.asarray(maxima, dtype=np.float64)


def conformal_threshold(maxima: np.ndarray, alpha: float) -> tuple[float, int]:
    ordered = np.sort(maxima)
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return (float("inf"), rank) if rank > len(ordered) else (float(ordered[rank - 1]), rank)


def interval(successes: int, total: int, level: float = 0.95) -> list[float]:
    if not total:
        return [float("nan"), float("nan")]
    tail = (1.0 - level) / 2.0
    low = 0.0 if not successes else float(beta.ppf(tail, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1.0 - tail, successes + 1, total - successes))
    return [low, high]


def evaluate(
    tasks: list[Task],
    loaded: dict[tuple[str, str, str], tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
    selected: set[tuple[str, str, str]],
    channel: str,
    threshold: float,
    event_type: str,
    partition: str | None = None,
) -> dict[str, float | int | list[float]]:
    healthy = false_alarm = event_total = early_two = by_onset = 0
    first_leads = []
    for task in tasks:
        key = task_key(task)
        if key not in selected:
            continue
        scores, metadata = loaded[key]
        alarm = np.isfinite(scores[channel]) & (scores[channel] > threshold)
        for episode, event in event_map(task).items():
            if not include_repeat(event, partition):
                continue
            rows = rows_for_episode(metadata, episode)
            query = metadata["query"][rows]
            fired = query[alarm[rows]]
            if is_healthy(event):
                healthy += 1
                false_alarm += int(len(fired) > 0)
            event_onset = onset(task, event, event_type)
            if event_onset < 0:
                continue
            event_total += 1
            if len(fired):
                first = int(fired[0])
                first_leads.append(first - event_onset)
                early_two += int(first <= event_onset - 2)
                by_onset += int(first <= event_onset)
    return {
        "healthy_episodes": healthy,
        "healthy_episode_alarms": false_alarm,
        "episode_fpr": false_alarm / healthy if healthy else float("nan"),
        "episode_fpr_ci95": interval(false_alarm, healthy),
        "event_episodes": event_total,
        "detected_by_onset_minus_2": early_two,
        "recall_by_onset_minus_2": early_two / event_total if event_total else float("nan"),
        "detected_by_onset": by_onset,
        "recall_by_onset": by_onset / event_total if event_total else float("nan"),
        "median_first_alarm_lead": float(np.median(first_leads)) if first_leads else float("nan"),
    }


def risk_upper_bound(false_alarms: int, total: int, delta: float) -> float:
    if not total or false_alarms == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, false_alarms + 1, total - false_alarms))


def ltt_select(
    tasks: list[Task], loaded, calibration_tasks, event_type: str, alpha: float, delta: float
) -> tuple[dict | None, list[dict]]:
    channels = [name for name, event in CHANNELS.items() if event == event_type]
    quantiles = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0)
    delta_each = delta / (len(channels) * len(quantiles))
    candidates = []
    for channel in channels:
        maxima = healthy_maxima(tasks, loaded, calibration_tasks, channel)
        for quantile in quantiles:
            threshold = float(np.quantile(maxima, quantile, method="higher"))
            errors = int(np.count_nonzero(maxima > threshold))
            upper = risk_upper_bound(errors, len(maxima), delta_each)
            metrics = evaluate(tasks, loaded, calibration_tasks, channel, threshold, event_type)
            candidates.append(
                {
                    "channel": channel,
                    "quantile": quantile,
                    "threshold": threshold,
                    "calibration_false_alarms": errors,
                    "calibration_episodes": len(maxima),
                    "risk_upper_bound": upper,
                    "calibration_early_recall": metrics["recall_by_onset_minus_2"],
                    "accepted": upper <= alpha,
                }
            )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = max(
        accepted,
        key=lambda candidate: (candidate["calibration_early_recall"], -candidate["risk_upper_bound"]),
    ) if accepted else None
    return selected, candidates


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in inventory() if task.corpus == "grid50x8"]
    loaded = {task_key(task): load_scores(task) for task in tasks}
    calibration_tasks, test_tasks = split_tasks(tasks)

    conformal = []
    for channel, event_type in CHANNELS.items():
        reference = healthy_maxima(tasks, loaded, calibration_tasks, channel)
        threshold, rank = conformal_threshold(reference, args.alpha)
        conformal.append(
            {
                "event": event_type,
                "channel": channel,
                "unit": "healthy_episode_maximum",
                "alpha": args.alpha,
                "support": len(reference),
                "order_rank": rank,
                "threshold": threshold,
                "calibration_fpr": float(np.mean(reference > threshold)),
                "test": evaluate(tasks, loaded, test_tasks, channel, threshold, event_type),
            }
        )

    all_tasks = {task_key(task) for task in tasks}
    repeat_split = []
    for channel, event_type in CHANNELS.items():
        reference = healthy_maxima(tasks, loaded, all_tasks, channel, "repeat_0_3")
        threshold, rank = conformal_threshold(reference, args.alpha)
        repeat_split.append(
            {
                "event": event_type,
                "channel": channel,
                "unit": "healthy_episode_maximum",
                "alpha": args.alpha,
                "support": len(reference),
                "order_rank": rank,
                "threshold": threshold,
                "calibration_fpr": float(np.mean(reference > threshold)),
                "test": evaluate(
                    tasks, loaded, all_tasks, channel, threshold, event_type, "repeat_4_7"
                ),
            }
        )

    ltt = []
    for event_type in ("loop", "static"):
        selected, candidates = ltt_select(
            tasks, loaded, calibration_tasks, event_type, args.alpha, args.ltt_delta
        )
        test = None if selected is None else evaluate(
            tasks,
            loaded,
            test_tasks,
            selected["channel"],
            selected["threshold"],
            event_type,
        )
        ltt.append(
            {
                "event": event_type,
                "family_size": len(candidates),
                "selected": selected,
                "test": test,
            }
        )

    flat = []
    for row in conformal:
        base = {key: value for key, value in row.items() if key != "test"}
        flat.append({**base, **{f"test_{key}": value for key, value in row["test"].items() if not isinstance(value, list)}})
    with (args.output / "conformal_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    summary = {
        "schema": "himoe.dynamic_risk_control.v1",
        "classifier_trained": False,
        "fit": "threshold calibration only",
        "target_episode_fpr": args.alpha,
        "calibration_tasks": ["/".join(key[1:]) for key in sorted(calibration_tasks)],
        "heldout_test_tasks": ["/".join(key[1:]) for key in sorted(test_tasks)],
        "unseen_task_conformal": conformal,
        "same_task_disjoint_repeat_conformal": repeat_split,
        "learn_then_test_style": ltt,
        "semantics": "risk-controlled alarm, not an outcome probability",
        "guarantee_boundary": (
            "finite-sample conformal guarantees require exchangeable calibration/test units; "
            "the unseen-task split is a distribution-shift stress test"
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"unseen_task_conformal": conformal, "learn_then_test_style": ltt}, indent=2))


if __name__ == "__main__":
    main()
