#!/usr/bin/env python3
"""Build fork-calibrated outcome assurance and audit q0 MoE mappings."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "assurance"))

from evaluation import LOOP_BLIND_TASKS, load_task, profile_inventory  # noqa: E402


OUTPUT = BUNDLE / "results/route_8"
CELL_PATH = PROJECT / "analysis_committor/out/c1_cells.csv"
SCORE_ROOT = BUNDLE / "results/routes_1_4/scores"
RECOVERY_SUMMARY = PROJECT / "analysis_signal_matrix_trainfree/recovery_window_summary.csv"
RECOVERY_DEPTH = PROJECT / "trap-recovery-depth-20260904"
FEATURES = (
    "r1_commitment",
    "r2_flow_instability",
    "r3_loop_graph_evidence",
    "r3_static_authority_loss",
    "r4_healthy_energy",
)
OUTCOMES = ("success", "loop", "static", "other_failure")
RNG = np.random.default_rng(20260905)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def score_archive(data):
    path = SCORE_ROOT / data.corpus / data.suite / f"{data.task}.npz"
    with np.load(path, allow_pickle=False) as archive:
        names = archive["score_names"].astype(str).tolist()
        score = np.asarray(archive["scores"], dtype=np.float64)
        episode = np.asarray(archive["episode_id"], dtype=np.int32)
        query = np.asarray(archive["query"], dtype=np.int16)
    return episode, query, {name: score[:, names.index(name)] for name in FEATURES}


def disjoint_outcome(data, event: dict[str, object]) -> str:
    if int(event["success"]) == 1:
        return "success"
    loop = int(event["loop_onset_q"])
    if data.task in LOOP_BLIND_TASKS:
        loop = -1
    static = int(event["static_onset_q"])
    candidates = [(loop, "loop"), (static, "static")]
    candidates = [item for item in candidates if item[0] >= 0]
    return min(candidates)[1] if candidates else "other_failure"


def dirichlet_marginal(count: int, counts: np.ndarray) -> tuple[float, float, float]:
    alpha = count + 0.5
    total = float(counts.sum() + 0.5 * len(counts))
    return (
        alpha / total,
        float(beta.ppf(0.025, alpha, total - alpha)),
        float(beta.ppf(0.975, alpha, total - alpha)),
    )


def build_cells(tasks) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    indexed = {(data.corpus, data.suite, data.task): data for data in tasks}
    cells = pd.read_csv(CELL_PATH)
    records = []
    tensor = []
    for source in cells.itertuples(index=False):
        key = (source.corpus, source.suite, source.task)
        data = indexed[key]
        episode, query, scores = score_archive(data)
        q0 = (query == 0) & (data.profile.scene == int(source.scene))
        if int(q0.sum()) != int(source.K):
            raise ValueError(f"q0 cell size mismatch for {key} scene {source.scene}")
        events = [row for row in data.events if int(row["scene"]) == int(source.scene)]
        if len(events) != int(source.K):
            raise ValueError(f"event cell size mismatch for {key} scene {source.scene}")
        categories = [disjoint_outcome(data, row) for row in events]
        counts = np.asarray([categories.count(name) for name in OUTCOMES], dtype=np.int64)
        record = {
            "corpus": source.corpus,
            "suite": source.suite,
            "task": source.task,
            "scene": int(source.scene),
            "K": int(source.K),
            "loop_valid": int(source.loop_valid),
            "n_fail": int(source.n_fail),
            "n_loop": int(source.n_loop),
            "n_static": int(source.n_static),
        }
        for name, values in scores.items():
            record[f"{name}_mean"] = float(np.nanmean(values[q0]))
            record[f"{name}_std"] = float(np.nanstd(values[q0]))
        for outcome, count in zip(OUTCOMES, counts):
            probability, low, high = dirichlet_marginal(int(count), counts)
            record[f"n_{outcome}_disjoint"] = int(count)
            record[f"p_{outcome}"] = probability
            tensor.append(
                {
                    "source": "q0_committor",
                    "cohort": source.corpus,
                    "corpus": source.corpus,
                    "suite": source.suite,
                    "task": source.task,
                    "scene": int(source.scene),
                    "intervention": "continue",
                    "horizon": "episode_terminal",
                    "outcome": outcome,
                    "count": int(count),
                    "total": int(counts.sum()),
                    "posterior_mean": probability,
                    "ci95_low": low,
                    "ci95_high": high,
                    "posterior": "Dirichlet(count + 0.5)",
                    "note": "same physical q0 snapshot; routing averaged over K noise draws",
                }
            )
        records.append(record)
    return pd.DataFrame.from_records(records), tensor


def spearman(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def task_macro_correlation(frame: pd.DataFrame, score: str, target: str, draws: int) -> dict:
    estimates = []
    for _task, group in frame.groupby(["suite", "task"]):
        value = spearman(group[score], group[target])
        if np.isfinite(value):
            estimates.append(value)
    estimates = np.asarray(estimates, dtype=np.float64)
    if not len(estimates):
        return {"task_mean_spearman": float("nan"), "tasks": 0}
    bootstrap = np.mean(
        estimates[RNG.integers(0, len(estimates), (draws, len(estimates)))], axis=1
    )
    return {
        "task_mean_spearman": float(estimates.mean()),
        "tasks": len(estimates),
        "bootstrap_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "raw_cell_spearman": spearman(frame[score], frame[target]),
    }


def fit_binomial(x, events, total):
    scaler = StandardScaler().fit(x)
    transformed = scaler.transform(x)
    xx = np.vstack((transformed, transformed))
    yy = np.r_[np.ones(len(x)), np.zeros(len(x))]
    weights = np.r_[events, total - events]
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    model.fit(xx, yy, sample_weight=weights)
    return scaler, model


def binomial_metrics(events, total, probability) -> dict:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1 - 1e-6)
    events = np.asarray(events, dtype=np.float64)
    total = np.asarray(total, dtype=np.float64)
    failures = total - events
    return {
        "candidate_brier": float(
            np.sum(events * (1 - probability) ** 2 + failures * probability**2)
            / np.sum(total)
        ),
        "candidate_log_loss": float(
            -np.sum(events * np.log(probability) + failures * np.log(1 - probability))
            / np.sum(total)
        ),
        "cell_spearman": spearman(probability, events / total),
        "candidates": int(np.sum(total)),
        "cells": len(total),
    }


def calibration_models(cells: pd.DataFrame) -> tuple[list[dict[str, object]], pd.DataFrame]:
    feature_columns = [f"{name}_mean" for name in FEATURES]
    rows = []
    predictions = []
    outcome_count = {
        "failure": "n_fail",
        "loop": "n_loop",
        "static": "n_static",
    }
    for corpus in ("main16x32", "grid50x8"):
        corpus_frame = cells[cells.corpus == corpus].copy()
        for outcome, count_column in outcome_count.items():
            frame = corpus_frame
            if outcome == "loop":
                frame = frame[frame.loop_valid == 1]
            held_predictions = np.empty(len(frame), dtype=np.float64)
            baseline = np.empty(len(frame), dtype=np.float64)
            for _task, test in frame.groupby(["suite", "task"]):
                train = frame.drop(index=test.index)
                x_train = train[feature_columns].to_numpy()
                scaler, model = fit_binomial(
                    x_train,
                    train[count_column].to_numpy(),
                    train.K.to_numpy(),
                )
                locations = frame.index.get_indexer(test.index)
                held_predictions[locations] = model.predict_proba(
                    scaler.transform(test[feature_columns].to_numpy())
                )[:, 1]
                baseline[locations] = train[count_column].sum() / train.K.sum()
            observed = frame[count_column].to_numpy()
            total = frame.K.to_numpy()
            rows.append(
                {
                    "evaluation": f"{corpus}_leave_one_task_out",
                    "outcome": outcome,
                    "model": binomial_metrics(observed, total, held_predictions),
                    "constant_train_rate": binomial_metrics(observed, total, baseline),
                }
            )
            for index, probability, base in zip(frame.index, held_predictions, baseline):
                predictions.append(
                    {
                        "evaluation": f"{corpus}_leave_one_task_out",
                        "outcome": outcome,
                        "cell_index": int(index),
                        "prediction": float(probability),
                        "baseline": float(base),
                    }
                )

    main = cells[cells.corpus == "main16x32"]
    grid = cells[cells.corpus == "grid50x8"]
    for outcome, count_column in outcome_count.items():
        train = main if outcome != "loop" else main[main.loop_valid == 1]
        test = grid if outcome != "loop" else grid[grid.loop_valid == 1]
        scaler, model = fit_binomial(
            train[feature_columns].to_numpy(),
            train[count_column].to_numpy(),
            train.K.to_numpy(),
        )
        probability = model.predict_proba(scaler.transform(test[feature_columns].to_numpy()))[:, 1]
        base = np.full(len(test), train[count_column].sum() / train.K.sum())
        rows.append(
            {
                "evaluation": "main16x32_to_grid50x8",
                "outcome": outcome,
                "model": binomial_metrics(test[count_column], test.K, probability),
                "constant_train_rate": binomial_metrics(test[count_column], test.K, base),
                "coefficients": dict(zip(FEATURES, model.coef_[0].tolist())),
                "intercept": float(model.intercept_[0]),
            }
        )
        for index, value, base_value in zip(test.index, probability, base):
            predictions.append(
                {
                    "evaluation": "main16x32_to_grid50x8",
                    "outcome": outcome,
                    "cell_index": int(index),
                    "prediction": float(value),
                    "baseline": float(base_value),
                }
            )
    return rows, pd.DataFrame.from_records(predictions)


def binary_tensor_row(source, cohort, intervention, horizon, successes, total, note):
    probability, low, high = dirichlet_marginal(
        successes, np.asarray([successes, total - successes])
    )
    return {
        "source": source,
        "cohort": cohort,
        "corpus": "",
        "suite": "libero_long",
        "task": "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "scene": "",
        "intervention": intervention,
        "horizon": horizon,
        "outcome": "success",
        "count": successes,
        "total": total,
        "posterior_mean": probability,
        "ci95_low": low,
        "ci95_high": high,
        "posterior": "Beta(successes + 0.5, failures + 0.5)",
        "note": note,
    }


def recovery_tensor() -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    with RECOVERY_SUMMARY.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            if source["status"] != "complete" or not source["k"]:
                continue
            rows.append(
                binary_tensor_row(
                    "physical_loop_onset_fork",
                    source["run"],
                    f"fresh_noise_at_loop_offset_{source['offset']}",
                    "until_query_cap_52",
                    int(source["successes"]),
                    int(source["k"]),
                    "one failure trunk; paired noise across offsets",
                )
            )

    completion = {"planned_trunks": 200, "budget20_complete_trunks": 0, "legacy_complete_trunks": 0}
    for root_name in ("runs", "runs_cap52"):
        root = RECOVERY_DEPTH / root_name
        cohorts = {}
        for manifest_path in sorted(root.glob("i*_s*/manifest.json")):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                continue
            arm_dirs = [
                path
                for path in manifest_path.parent.iterdir()
                if path.is_dir()
                and (path.name in {"control", "triggered"} or path.name.startswith("delayed_"))
            ]
            budgets = set()
            arms = {}
            for arm_dir in arm_dirs:
                candidates = [
                    json.loads(path.read_text(encoding="utf-8"))
                    for path in sorted(arm_dir.glob("candidate_*.json"))
                ]
                budgets.update(item["fork_budget_queries"] for item in candidates)
                arms[arm_dir.name] = (sum(bool(item["success"]) for item in candidates), len(candidates))
            if len(budgets) != 1:
                continue
            budget = budgets.pop()
            cohort = "preregistered_budget20" if budget == 20 else f"legacy_{root_name}_budget{budget}"
            if budget == 20:
                completion["budget20_complete_trunks"] += 1
            else:
                completion["legacy_complete_trunks"] += 1
            aggregate = cohorts.setdefault(cohort, {})
            for arm, (successes, total) in arms.items():
                value = aggregate.setdefault(arm, [0, 0])
                value[0] += successes
                value[1] += total
        for cohort, arms in cohorts.items():
            for arm, (successes, total) in sorted(arms.items()):
                rows.append(
                    binary_tensor_row(
                        "alarm_timing_fork",
                        cohort,
                        f"fresh_noise_{arm}",
                        "20_queries" if "budget20" in cohort else cohort.rsplit("_", 1)[-1],
                        successes,
                        total,
                        "partial collection; legacy and preregistered cohorts are not pooled",
                    )
                )
    completion["status"] = "incomplete"
    return rows, completion


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [load_task(item) for item in profile_inventory()]
    cells, tensor = build_cells(tasks)
    recovery, completion = recovery_tensor()
    tensor.extend(recovery)
    cells.to_csv(args.output / "q0_cell_assurance.csv", index=False)
    pd.DataFrame.from_records(tensor).to_csv(args.output / "assurance_tensor.csv", index=False)

    correlations = []
    count_columns = {"failure": "n_fail", "loop": "n_loop", "static": "n_static"}
    for corpus in ("main16x32", "grid50x8"):
        subset = cells[cells.corpus == corpus]
        for outcome in ("failure", "loop", "static"):
            selected = subset if outcome != "loop" else subset[subset.loop_valid == 1]
            target = count_columns[outcome]
            selected = selected.assign(**{f"rate_{outcome}": selected[target] / selected.K})
            for feature in FEATURES:
                correlations.append(
                    {
                        "corpus": corpus,
                        "outcome": outcome,
                        "feature": feature,
                        **task_macro_correlation(
                            selected, f"{feature}_mean", f"rate_{outcome}", args.bootstrap
                        ),
                    }
                )
    calibration, predictions = calibration_models(cells)
    predictions.to_csv(args.output / "calibration_predictions.csv", index=False)
    summary = {
        "schema": "himoe.assurance.route_8_committor.v1",
        "q0_cells": len(cells),
        "q0_routing_semantics": "K-sample noise-marginal mean at one physical state",
        "q0_posterior": "four-outcome Jeffreys Dirichlet posterior",
        "correlations": correlations,
        "calibration_models": calibration,
        "calibration_runtime_task_metadata": False,
        "recovery_collection": completion,
        "unavailable": [
            "multi-snapshot two-query physical no-loop committor",
            "enough completed recovery trunks for a population probability",
        ],
        "probability_boundary": "only assurance_tensor.csv entries are outcome probabilities",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    headline = [
        row
        for row in calibration
        if row["evaluation"] == "main16x32_to_grid50x8"
    ]
    print(
        json.dumps(
            {
                "q0_cells": len(cells),
                "recovery_collection": completion,
                "cross_corpus_calibration": headline,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
