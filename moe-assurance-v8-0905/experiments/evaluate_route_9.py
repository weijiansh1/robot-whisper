#!/usr/bin/env python3
"""Calibrate episode-level conformal and finite-family risk alarms."""

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
sys.path.insert(0, str(BUNDLE / "assurance"))

from evaluation import LOOP_BLIND_TASKS, episode_rows, load_task, profile_inventory  # noqa: E402


OUTPUT = BUNDLE / "results/route_9"
SCORE_ROOT = BUNDLE / "results/routes_1_4/scores"
POSTERIOR_ROOT = BUNDLE / "results/route_7/posteriors"
CHANNELS = {
    "r2_flow_instability": "loop",
    "r3_loop_graph_evidence": "loop",
    "r3_static_authority_loss": "static",
    "r3_static_lockin": "static",
    "r4_healthy_energy": "both",
    "hsmm_loop_belief": "loop",
    "hsmm_static_belief": "static",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ltt-delta", type=float, default=0.05)
    return parser.parse_args()


def task_key(data) -> tuple[str, str, str]:
    return data.corpus, data.suite, data.task


def load_channels(data) -> dict[str, np.ndarray]:
    score_path = SCORE_ROOT / data.corpus / data.suite / f"{data.task}.npz"
    with np.load(score_path, allow_pickle=False) as archive:
        names = archive["score_names"].astype(str).tolist()
        matrix = np.asarray(archive["scores"], dtype=np.float64)
    output = {
        name: matrix[:, names.index(name)]
        for name in (
            "r2_flow_instability",
            "r3_loop_graph_evidence",
            "r3_static_authority_loss",
            "r3_static_lockin",
            "r4_healthy_energy",
        )
    }
    posterior_path = POSTERIOR_ROOT / data.corpus / data.suite / f"{data.task}.npz"
    with np.load(posterior_path, allow_pickle=False) as archive:
        states = archive["state_names"].astype(str).tolist()
        posterior = np.asarray(archive["posterior"], dtype=np.float64)
    output["hsmm_loop_belief"] = posterior[:, states.index("PL")] + posterior[:, states.index("L")]
    output["hsmm_static_belief"] = posterior[:, states.index("PS")] + posterior[:, states.index("S")]
    return output


def is_healthy(event) -> bool:
    return (
        int(event["success"]) == 1
        and int(event["loop_onset_q"]) < 0
        and int(event["static_onset_q"]) < 0
    )


def event_onset(data, event, event_type: str) -> int:
    if event_type == "loop":
        return -1 if data.task in LOOP_BLIND_TASKS else int(event["loop_onset_q"])
    return int(event["static_onset_q"])


def split_external_tasks(tasks):
    calibration, test = set(), set()
    suites = sorted({data.suite for data in tasks})
    for suite in suites:
        members = sorted((data for data in tasks if data.suite == suite), key=lambda data: data.task)
        for index, data in enumerate(members):
            (calibration if index % 2 == 0 else test).add(task_key(data))
    return calibration, test


def include_partition(event, partition: str | None) -> bool:
    if partition is None:
        return True
    repeat = int(event["repeat"])
    if partition == "repeat_0_3":
        return repeat < 4
    if partition == "repeat_4_7":
        return repeat >= 4
    raise ValueError(partition)


def episode_maxima(
    tasks, channels, selected_tasks, channel: str, partition: str | None = None
) -> np.ndarray:
    maxima = []
    for data in tasks:
        key = task_key(data)
        if key not in selected_tasks:
            continue
        score = channels[key][channel]
        for episode, event in data.event_by_episode.items():
            if is_healthy(event) and include_partition(event, partition):
                maxima.append(float(np.nanmax(score[episode_rows(data.profile, episode)])))
    return np.asarray(maxima, dtype=np.float64)


def cluster_maxima(
    tasks, channels, selected_tasks, channel: str, partition: str | None = None
) -> np.ndarray:
    maxima = []
    for data in tasks:
        key = task_key(data)
        if key not in selected_tasks:
            continue
        score = channels[key][channel]
        by_scene = {}
        for episode, event in data.event_by_episode.items():
            if not is_healthy(event) or not include_partition(event, partition):
                continue
            value = float(np.nanmax(score[episode_rows(data.profile, episode)]))
            by_scene.setdefault(int(event["scene"]), []).append(value)
        maxima.extend(max(values) for values in by_scene.values())
    return np.asarray(maxima, dtype=np.float64)


def conformal_threshold(maxima: np.ndarray, alpha: float) -> tuple[float, int]:
    ordered = np.sort(np.asarray(maxima, dtype=np.float64))
    rank = int(np.ceil((len(ordered) + 1) * (1.0 - alpha)))
    if rank > len(ordered):
        return float("inf"), rank
    return float(ordered[rank - 1]), rank


def binomial_interval(successes: int, total: int, level: float = 0.95):
    if total == 0:
        return [float("nan"), float("nan")]
    tail = (1.0 - level) / 2.0
    low = 0.0 if successes == 0 else float(beta.ppf(tail, successes, total - successes + 1))
    high = 1.0 if successes == total else float(beta.ppf(1 - tail, successes + 1, total - successes))
    return [low, high]


def evaluate_rules(
    tasks,
    channels,
    selected_tasks,
    rules,
    event_type: str,
    partition: str | None = None,
) -> dict:
    healthy_total = healthy_alarm = 0
    cluster_alarm = {}
    event_total = early_two = by_onset = 0
    first_leads = []
    for data in tasks:
        key = task_key(data)
        if key not in selected_tasks:
            continue
        alarm = np.zeros(len(data.profile.query), dtype=bool)
        for channel, threshold in rules:
            alarm |= channels[key][channel] > threshold
        for episode, event in data.event_by_episode.items():
            if not include_partition(event, partition):
                continue
            rows = episode_rows(data.profile, episode)
            order = rows[np.argsort(data.profile.query[rows])]
            query = data.profile.query[order]
            episode_alarm = alarm[order]
            if is_healthy(event):
                fired = bool(np.any(episode_alarm))
                healthy_total += 1
                healthy_alarm += int(fired)
                cluster_key = (data.suite, data.task, int(event["scene"]))
                cluster_alarm[cluster_key] = cluster_alarm.get(cluster_key, False) or fired
            onset = event_onset(data, event, event_type)
            if onset < 0:
                continue
            event_total += 1
            alarm_queries = query[episode_alarm]
            if len(alarm_queries):
                first = int(alarm_queries[0])
                first_leads.append(first - onset)
                early_two += int(first <= onset - 2)
                by_onset += int(first <= onset)
    cluster_total = len(cluster_alarm)
    cluster_fired = sum(cluster_alarm.values())
    return {
        "healthy_episodes": healthy_total,
        "healthy_episode_alarms": healthy_alarm,
        "episode_fpr": healthy_alarm / healthy_total if healthy_total else float("nan"),
        "episode_fpr_ci95": binomial_interval(healthy_alarm, healthy_total),
        "healthy_scene_clusters": cluster_total,
        "healthy_scene_cluster_alarms": cluster_fired,
        "scene_cluster_fpr": cluster_fired / cluster_total if cluster_total else float("nan"),
        "scene_cluster_fpr_ci95": binomial_interval(cluster_fired, cluster_total),
        "event_episodes": event_total,
        "detected_by_onset_minus_2": early_two,
        "recall_by_onset_minus_2": early_two / event_total if event_total else float("nan"),
        "detected_by_onset": by_onset,
        "recall_by_onset": by_onset / event_total if event_total else float("nan"),
        "median_first_alarm_lead": float(np.median(first_leads)) if first_leads else float("nan"),
    }


def risk_upper_bound(false_alarms: int, total: int, delta: float) -> float:
    if total == 0 or false_alarms == total:
        return 1.0
    return float(beta.ppf(1.0 - delta, false_alarms + 1, total - false_alarms))


def ltt_select(tasks, channels, calibration_tasks, event_type, alpha, delta):
    eligible_channels = [
        name for name, phenotype in CHANNELS.items() if phenotype in {event_type, "both"}
    ]
    quantiles = (0.90, 0.925, 0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.995, 1.0)
    family_size = len(eligible_channels) * len(quantiles)
    delta_each = delta / family_size
    candidates = []
    for channel in eligible_channels:
        maxima = episode_maxima(tasks, channels, calibration_tasks, channel)
        for quantile in quantiles:
            threshold = float(np.quantile(maxima, quantile, method="higher"))
            false_alarms = int(np.count_nonzero(maxima > threshold))
            upper = risk_upper_bound(false_alarms, len(maxima), delta_each)
            metrics = evaluate_rules(
                tasks, channels, calibration_tasks, [(channel, threshold)], event_type
            )
            candidates.append(
                {
                    "channel": channel,
                    "quantile": quantile,
                    "threshold": threshold,
                    "calibration_false_alarms": false_alarms,
                    "calibration_episodes": len(maxima),
                    "risk_upper_bound": upper,
                    "calibration_early_recall": metrics["recall_by_onset_minus_2"],
                    "accepted": upper <= alpha,
                }
            )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    selected = max(
        accepted,
        key=lambda candidate: (
            candidate["calibration_early_recall"],
            -candidate["risk_upper_bound"],
        ),
    ) if accepted else None
    return selected, candidates, delta_each


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(item) for item in profile_inventory() if item[0] == "grid50x8"]
    channels = {task_key(data): load_channels(data) for data in tasks}
    calibration_tasks, test_tasks = split_external_tasks(tasks)

    conformal = []
    for channel, phenotype in CHANNELS.items():
        for event_type in ("loop", "static"):
            if phenotype not in {event_type, "both"}:
                continue
            episode_reference = episode_maxima(
                tasks, channels, calibration_tasks, channel
            )
            threshold, rank = conformal_threshold(episode_reference, args.alpha)
            cluster_reference = cluster_maxima(
                tasks, channels, calibration_tasks, channel
            )
            cluster_threshold, cluster_rank = conformal_threshold(
                cluster_reference, args.alpha
            )
            conformal.append(
                {
                    "event": event_type,
                    "channel": channel,
                    "unit": "episode_maximum",
                    "alpha": args.alpha,
                    "support": len(episode_reference),
                    "order_rank": rank,
                    "threshold": threshold,
                    "calibration_fpr": float(np.mean(episode_reference > threshold)),
                    "test": evaluate_rules(
                        tasks, channels, test_tasks, [(channel, threshold)], event_type
                    ),
                }
            )
            conformal.append(
                {
                    "event": event_type,
                    "channel": channel,
                    "unit": "scene_cluster_maximum",
                    "alpha": args.alpha,
                    "support": len(cluster_reference),
                    "order_rank": cluster_rank,
                    "threshold": cluster_threshold,
                    "calibration_fpr": float(np.mean(cluster_reference > cluster_threshold)),
                    "test": evaluate_rules(
                        tasks,
                        channels,
                        test_tasks,
                        [(channel, cluster_threshold)],
                        event_type,
                    ),
                }
            )

    familywise = []
    for event_type in ("loop", "static"):
        selected_channels = [
            name for name, phenotype in CHANNELS.items() if phenotype in {event_type, "both"}
        ]
        rules = []
        for channel in selected_channels:
            maxima = episode_maxima(tasks, channels, calibration_tasks, channel)
            threshold, _ = conformal_threshold(maxima, args.alpha / len(selected_channels))
            rules.append((channel, threshold))
        familywise.append(
            {
                "event": event_type,
                "method": "Bonferroni OR of episode-max conformal channels",
                "family_alpha": args.alpha,
                "rules": [{"channel": name, "threshold": value} for name, value in rules],
                "test": evaluate_rules(tasks, channels, test_tasks, rules, event_type),
            }
        )

    # Same task/scene distribution, disjoint noise repeats. This is closer to
    # exchangeability than the unseen-task stress test, while still keeping every
    # threshold global and free of runtime task metadata.
    all_tasks = {task_key(data) for data in tasks}
    repeat_conformal = []
    for channel, phenotype in CHANNELS.items():
        for event_type in ("loop", "static"):
            if phenotype not in {event_type, "both"}:
                continue
            reference = episode_maxima(
                tasks, channels, all_tasks, channel, "repeat_0_3"
            )
            threshold, rank = conformal_threshold(reference, args.alpha)
            cluster_reference = cluster_maxima(
                tasks, channels, all_tasks, channel, "repeat_0_3"
            )
            cluster_threshold, cluster_rank = conformal_threshold(
                cluster_reference, args.alpha
            )
            repeat_conformal.extend(
                [
                    {
                        "event": event_type,
                        "channel": channel,
                        "unit": "episode_maximum",
                        "alpha": args.alpha,
                        "support": len(reference),
                        "order_rank": rank,
                        "threshold": threshold,
                        "calibration_fpr": float(np.mean(reference > threshold)),
                        "test": evaluate_rules(
                            tasks,
                            channels,
                            all_tasks,
                            [(channel, threshold)],
                            event_type,
                            "repeat_4_7",
                        ),
                    },
                    {
                        "event": event_type,
                        "channel": channel,
                        "unit": "scene_cluster_maximum",
                        "alpha": args.alpha,
                        "support": len(cluster_reference),
                        "order_rank": cluster_rank,
                        "threshold": cluster_threshold,
                        "calibration_fpr": float(
                            np.mean(cluster_reference > cluster_threshold)
                        ),
                        "test": evaluate_rules(
                            tasks,
                            channels,
                            all_tasks,
                            [(channel, cluster_threshold)],
                            event_type,
                            "repeat_4_7",
                        ),
                    },
                ]
            )
    repeat_familywise = []
    for event_type in ("loop", "static"):
        selected_channels = [
            name
            for name, phenotype in CHANNELS.items()
            if phenotype in {event_type, "both"}
        ]
        rules = []
        for channel in selected_channels:
            maxima = episode_maxima(
                tasks, channels, all_tasks, channel, "repeat_0_3"
            )
            threshold, _ = conformal_threshold(
                maxima, args.alpha / len(selected_channels)
            )
            rules.append((channel, threshold))
        repeat_familywise.append(
            {
                "event": event_type,
                "method": "Bonferroni OR of episode-max conformal channels",
                "family_alpha": args.alpha,
                "rules": [
                    {"channel": name, "threshold": value} for name, value in rules
                ],
                "test": evaluate_rules(
                    tasks,
                    channels,
                    all_tasks,
                    rules,
                    event_type,
                    "repeat_4_7",
                ),
            }
        )

    ltt = []
    for event_type in ("loop", "static"):
        selected, candidates, delta_each = ltt_select(
            tasks,
            channels,
            calibration_tasks,
            event_type,
            args.alpha,
            args.ltt_delta,
        )
        test = (
            evaluate_rules(
                tasks,
                channels,
                test_tasks,
                [(selected["channel"], selected["threshold"])],
                event_type,
            )
            if selected
            else None
        )
        ltt.append(
            {
                "event": event_type,
                "family_size": len(candidates),
                "delta_each_bonferroni": delta_each,
                "selected": selected,
                "test": test,
            }
        )
    with (args.output / "conformal_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        flat = []
        for row in conformal:
            base = {key: value for key, value in row.items() if key != "test"}
            flat.append({**base, **{f"test_{key}": value for key, value in row["test"].items() if not isinstance(value, list)}})
        writer = csv.DictWriter(handle, fieldnames=tuple(flat[0]))
        writer.writeheader()
        writer.writerows(flat)

    summary = {
        "schema": "himoe.assurance.route_9_risk_control.v1",
        "fit_corpus": "main16x32",
        "calibration_corpus": "grid50x8",
        "calibration_tasks": ["/".join(key[1:]) for key in sorted(calibration_tasks)],
        "heldout_test_tasks": ["/".join(key[1:]) for key in sorted(test_tasks)],
        "target_fpr": args.alpha,
        "conformal": conformal,
        "familywise": familywise,
        "same_task_disjoint_repeat_conformal": repeat_conformal,
        "same_task_disjoint_repeat_familywise": repeat_familywise,
        "learn_then_test_style": ltt,
        "guarantee_boundary": (
            "finite-sample guarantees require exchangeable calibration/test units; "
            "unseen-task shift and same-scene siblings violate a literal iid reading"
        ),
        "semantics": "risk-controlled alarm, not an individual outcome probability",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "unseen_task_familywise": familywise,
                "same_task_disjoint_repeat_familywise": repeat_familywise,
                "learn_then_test_style": ltt,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
