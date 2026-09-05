#!/usr/bin/env python3
"""Fit a task-agnostic HSMM mode belief on main16x32 and test grid50x8."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
sys.path.insert(0, str(BUNDLE / "assurance"))

from evaluation import LOOP_BLIND_TASKS, episode_rows, load_task, matched_onset_auc, profile_inventory  # noqa: E402
from hsmm import HSMMModel, fit_model  # noqa: E402


OUTPUT = BUNDLE / "results/route_7"
SCORE_ROOT = BUNDLE / "results/routes_1_4/scores"
STATES = ("H", "C", "PL", "L", "PS", "S")
FEATURES = (
    "r1_commitment",
    "r2_late_volatility",
    "r2_route_acceleration",
    "r2_flow_instability",
    "r3_token_consensus",
    "r3_effective_rank",
    "r3_token_expert_mi",
    "r3_occupancy_concentration",
    "r3_support_union_fraction",
    "r3_static_lockin",
    "r4_healthy_energy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-age", type=int, default=12)
    return parser.parse_args()


def load_scores(data) -> np.ndarray:
    path = SCORE_ROOT / data.corpus / data.suite / f"{data.task}.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = archive["score_names"].astype(str).tolist()
        scores = np.asarray(archive["scores"], dtype=np.float64)
        if not np.array_equal(archive["episode_id"], data.profile.episode_id):
            raise ValueError(f"score/profile mismatch: {path}")
    return np.column_stack([scores[:, names.index(name)] for name in FEATURES])


def event_onsets(data, event: dict[str, object]) -> tuple[int, int]:
    loop = int(event["loop_onset_q"])
    if data.task in LOOP_BLIND_TASKS:
        loop = -1
    return loop, int(event["static_onset_q"])


def label_episode(data, event, queries: np.ndarray, instability: np.ndarray) -> np.ndarray:
    index = {name: offset for offset, name in enumerate(STATES)}
    labels = np.full(len(queries), index["H"], dtype=np.int8)
    loop, static = event_onsets(data, event)
    candidates = [(loop, "PL", "L"), (static, "PS", "S")]
    candidates = [item for item in candidates if item[0] >= 0]
    if candidates:
        onset, precursor, trapped = min(candidates, key=lambda item: item[0])
        labels[(queries >= max(0, onset - 2)) & (queries < onset)] = index[precursor]
        labels[queries >= onset] = index[trapped]

    # A correction is a high internal pulse that returns within two queries and is
    # not already assigned to a precursor/trap segment. Future information is used
    # only to construct the training target, never by the online filter.
    for position in range(len(queries)):
        if labels[position] != index["H"] or instability[position] <= 0.9:
            continue
        stop = min(len(queries), position + 3)
        if np.any(instability[position + 1 : stop] < 0.75):
            labels[position] = index["C"]
    return labels


def robust_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(values, axis=0)
    q25, q75 = np.nanquantile(values, (0.25, 0.75), axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1e-4)
    return median, scale


def transform(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, median)
    return np.clip((filled - median) / scale, -8.0, 8.0)


def sequence_parts(data, scores, labels):
    for episode in np.unique(data.profile.episode_id):
        rows = episode_rows(data.profile, int(episode))
        order = rows[np.argsort(data.profile.query[rows])]
        yield order, scores[order], labels[order]


def brier_forecasts(data, posterior_forecasts, horizons=(1, 2, 4)):
    result = []
    events = data.event_by_episode
    for event_type, state_name in (("loop", "L"), ("static", "S")):
        if event_type == "loop" and data.task in LOOP_BLIND_TASKS:
            continue
        for horizon in horizons:
            probability = posterior_forecasts[horizon][:, STATES.index(state_name)]
            truth = np.zeros(len(probability), dtype=np.float64)
            for episode, event in events.items():
                onset = event_onsets(data, event)[0 if event_type == "loop" else 1]
                rows = episode_rows(data.profile, episode)
                queries = data.profile.query[rows]
                truth[rows] = (onset >= 0) & (onset <= queries + horizon)
            prevalence = float(truth.mean())
            result.append(
                (
                    event_type,
                    horizon,
                    float(np.mean((probability - truth) ** 2)),
                    float(np.mean((prevalence - truth) ** 2)),
                    prevalence,
                    len(truth),
                )
            )
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(item) for item in profile_inventory()]
    raw_scores = {
        (data.corpus, data.suite, data.task): load_scores(data) for data in tasks
    }

    scaling_values, train_values, train_labels, train_sequences = [], [], [], []
    provisional = {}
    for data in tasks:
        key = (data.corpus, data.suite, data.task)
        scores = raw_scores[key]
        labels = np.empty(len(scores), dtype=np.int8)
        for episode, event in data.event_by_episode.items():
            rows = episode_rows(data.profile, episode)
            order = rows[np.argsort(data.profile.query[rows])]
            labels[order] = label_episode(
                data,
                event,
                data.profile.query[order],
                scores[order, FEATURES.index("r2_flow_instability")],
            )
        provisional[key] = labels
        if data.corpus == "main16x32":
            scaling_values.append(scores)
    median, scale = robust_fit(np.concatenate(scaling_values, axis=0))

    for data in tasks:
        if data.corpus != "main16x32":
            continue
        key = (data.corpus, data.suite, data.task)
        values = transform(raw_scores[key], median, scale)
        labels = provisional[key]
        train_values.append(values)
        train_labels.append(labels)
        train_sequences.extend([part[2] for part in sequence_parts(data, values, labels)])
    observation = np.concatenate(train_values, axis=0)
    label = np.concatenate(train_labels, axis=0)
    model = fit_model(observation, label, train_sequences, STATES, args.max_age)

    all_posteriors = {}
    all_forecasts = {}
    evaluation_rows = []
    for data in tasks:
        key = (data.corpus, data.suite, data.task)
        values = transform(raw_scores[key], median, scale)
        posterior = np.empty((len(values), len(STATES)), dtype=np.float64)
        forecasts = {horizon: np.empty_like(posterior) for horizon in (1, 2, 4)}
        for rows, sequence, _labels in sequence_parts(data, values, provisional[key]):
            filtered, beliefs = model.filter_history(sequence)
            posterior[rows] = filtered
            for position, row in enumerate(rows):
                future = model.forecast(beliefs[position], (1, 2, 4))
                for horizon in future:
                    forecasts[horizon][row] = future[horizon]
        all_posteriors[key] = posterior
        all_forecasts[key] = forecasts
        destination = args.output / "posteriors" / data.corpus / data.suite / f"{data.task}.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            episode_id=data.profile.episode_id,
            query=data.profile.query,
            state_names=np.asarray(STATES),
            posterior=posterior.astype(np.float32),
            forecast_horizons=np.asarray((1, 2, 4)),
            forecast=np.stack([forecasts[h] for h in (1, 2, 4)], axis=1).astype(np.float32),
        )
        for event_type, horizon, brier, baseline_brier, prevalence, rows_count in brier_forecasts(data, forecasts):
            evaluation_rows.append(
                {
                    "corpus": data.corpus,
                    "suite": data.suite,
                    "task": data.task,
                    "metric": "brier",
                    "event": event_type,
                    "horizon": horizon,
                    "value": brier,
                    "constant_prevalence_brier": baseline_brier,
                    "prevalence": prevalence,
                    "rows": rows_count,
                }
            )

    state_metrics = {}
    for corpus in ("main16x32", "grid50x8"):
        truth = np.concatenate(
            [provisional[(data.corpus, data.suite, data.task)] for data in tasks if data.corpus == corpus]
        )
        pred = np.concatenate(
            [all_posteriors[(data.corpus, data.suite, data.task)].argmax(axis=1) for data in tasks if data.corpus == corpus]
        )
        state_metrics[corpus] = {
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
            "confusion_matrix": confusion_matrix(truth, pred, labels=np.arange(len(STATES))).tolist(),
            "label_counts": np.bincount(truth, minlength=len(STATES)).tolist(),
        }

    matched = []
    for corpus in ("main16x32", "grid50x8"):
        subset = [data for data in tasks if data.corpus == corpus]
        for event_type, modes in (("loop", ("PL", "L")), ("static", ("PS", "S"))):
            scores = {
                key: posterior[:, [STATES.index(name) for name in modes]].sum(axis=1)
                for key, posterior in all_posteriors.items()
                if key[0] == corpus
            }
            for lead in (-2, 0):
                matched.append(
                    {
                        "corpus": corpus,
                        "event": event_type,
                        "lead": lead,
                        **matched_onset_auc(subset, scores, event_type, lead),
                    }
                )

    with (args.output / "task_brier.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(evaluation_rows[0]))
        writer.writeheader()
        writer.writerows(evaluation_rows)
    np.savez_compressed(
        args.output / "model.npz",
        state_names=np.asarray(STATES),
        feature_names=np.asarray(FEATURES),
        robust_median=median,
        robust_scale=scale,
        means=model.means,
        variances=model.variances,
        initial=model.initial,
        survival=model.survival,
        exit_probability=model.exit_probability,
        max_age=np.asarray(model.max_age),
    )
    summary = {
        "schema": "himoe.assurance.route_7_hsmm.v1",
        "fitted": True,
        "train_corpus": "main16x32",
        "external_test_corpus": "grid50x8",
        "runtime_task_metadata": False,
        "states": list(STATES),
        "features": list(FEATURES),
        "correction_label": "flow-instability percentile > .9 and returns below .75 within two queries",
        "state_metrics": state_metrics,
        "matched_onset": matched,
        "aggregate_brier": [
            {
                "corpus": corpus,
                "event": event_type,
                "horizon": horizon,
                "model_brier": float(
                    np.average(
                        [row["value"] for row in evaluation_rows if row["corpus"] == corpus and row["event"] == event_type and row["horizon"] == horizon],
                        weights=[row["rows"] for row in evaluation_rows if row["corpus"] == corpus and row["event"] == event_type and row["horizon"] == horizon],
                    )
                ),
                "within_task_prevalence_baseline_brier": float(
                    np.average(
                        [row["constant_prevalence_brier"] for row in evaluation_rows if row["corpus"] == corpus and row["event"] == event_type and row["horizon"] == horizon],
                        weights=[row["rows"] for row in evaluation_rows if row["corpus"] == corpus and row["event"] == event_type and row["horizon"] == horizon],
                    )
                ),
            }
            for corpus in ("main16x32", "grid50x8")
            for event_type in ("loop", "static")
            for horizon in (1, 2, 4)
            if any(row["corpus"] == corpus and row["event"] == event_type and row["horizon"] == horizon for row in evaluation_rows)
        ],
        "forecast_semantics": "mode-model probability, not snapshot-fork outcome probability",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
