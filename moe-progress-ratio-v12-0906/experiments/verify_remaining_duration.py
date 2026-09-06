#!/usr/bin/env python3
"""Does routing predict how much longer a healthy rollout still needs?

Every detector attempt in this bundle tried to predict a 3%-prevalence binary
outcome, and the information-source ablation found that in the region where
information is worth anything the whole routing stack adds 0.0017 nats over the
clock.  This asks a different question with a dense target.

Treat the rollout as an activity network.  The horizon cap is the total float,
the chunk index is elapsed time, and the quantity that decides whether an
intervention is still worth anything is slack:

    slack(q) = (T_cap - q) - D_hat(routing at q)

The second term is "given the routing looks like this, how much longer do
healthy rollouts still take".  It is learnable offline from successes alone,
needs no outcome label and no task identity at runtime, and unlike the binary
target it has ~180k rows rather than 487 positives.

This script tests only the precondition.  If routing does not beat the clock at
predicting remaining duration on healthy rollouts, the slack construction has
no basis and the direction stops here.

Blocking matters more than usual.  Tasks have very different lengths, so a model
that merely recognises the task would predict remaining duration without any
notion of progress.  Grouped CV by task removes that; the leave-one-suite-out
variant additionally removes suite-level length priors, which task blocking
alone leaves intact.

Run:  python experiments/verify_remaining_duration.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

CACHE = BUNDLE / "results/progress_cache/development_main.npz"
ENTROPY = BUNDLE / "results/ablation/gate_entropy.npz"
LABELS = (
    WORKSPACE
    / "double-selete/trainfree/results/timeout_extension_plus10"
    / "development_main_clean_labels.csv"
)
DEFAULT_OUTPUT = BUNDLE / "results/remaining_duration"

WINDOWS = (4, 8)
EPS_LENGTH = 1e-3
DRAWS = 2000
SEED = 20260906


def load() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = dict(np.load(CACHE, allow_pickle=False))
    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    index = pd.DataFrame(
        {
            "row": np.arange(len(task)),
            "task": task,
            "episode": cache["episode"].astype(int),
        }
    )
    labels = pd.read_csv(LABELS)[["task", "episode", "original_failure"]]
    merged = index.merge(
        labels, on=["task", "episode"], how="left", validate="one_to_one"
    ).sort_values("row")
    if merged["original_failure"].isna().any():
        raise ValueError("labels do not align with the progress cache")
    success = ~merged["original_failure"].to_numpy(bool)
    suite = pd.Series(task).str.split("/", n=1).str[0].to_numpy()

    with np.load(ENTROPY, allow_pickle=False) as archive:
        if not np.array_equal(archive["episode"], cache["episode"]):
            raise ValueError("entropy cache is not row-aligned with the progress cache")
        entropy = archive["entropy"]
    return cache, success, task, suite, entropy


def build_rows(
    cache: dict, success: np.ndarray, task: np.ndarray, suite: np.ndarray, entropy: np.ndarray
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per (successful episode, valid chunk); target = chunks still to go."""
    adjacent = cache["lag_distance"][:, :, 0, :]
    length = cache["length"].astype(int)
    valid = cache["valid"] & success[:, None]

    columns: dict[str, np.ndarray] = {}
    for layer_index, name in enumerate(pr.LAYER_NAMES):
        columns[f"mobility_{name}"] = adjacent[:, :, layer_index]
        columns[f"entropy_{name}"] = entropy[:, :, layer_index]
    for window in WINDOWS:
        path = pr.path_length(adjacent, window)
        displacement = cache["lag_distance"][:, :, pr.LAGS.index(window), :]
        ratio = pr.progress_ratio(displacement, path, EPS_LENGTH)
        for group in pr.LAYER_GROUPS:
            columns[f"R{window}_{group}"] = pr.group_ratio(ratio, group)
            columns[f"L{window}_{group}"] = pr.group_ratio(path, group)

    episodes, chunks = np.nonzero(valid)
    frame = pd.DataFrame({name: values[episodes, chunks] for name, values in columns.items()})
    frame["chunk"] = chunks
    frame["remaining"] = length[episodes] - 1 - chunks
    frame["task"] = task[episodes]
    frame["suite"] = suite[episodes]
    frame["episode_row"] = episodes

    routing = [name for name in columns]
    clock = ["chunk", "chunk_sq", "chunk_log"]
    frame["chunk_sq"] = frame["chunk"] ** 2
    frame["chunk_log"] = np.log1p(frame["chunk"])
    return frame.dropna(subset=routing + ["remaining"]).reset_index(drop=True), clock, routing


def fold_predictions(
    frame: pd.DataFrame, features: list[str], groups: np.ndarray, n_splits: int
) -> np.ndarray:
    """Out-of-fold predictions from ridge, standardised on the training fold only."""
    matrix = frame[features].to_numpy(np.float64)
    target = frame["remaining"].to_numpy(np.float64)
    predicted = np.full(len(frame), np.nan)
    splitter = GroupKFold(n_splits=n_splits)
    for train, test in splitter.split(matrix, target, groups):
        centre = matrix[train].mean(axis=0)
        scale = matrix[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        model = Ridge(alpha=1.0).fit((matrix[train] - centre) / scale, target[train])
        predicted[test] = model.predict((matrix[test] - centre) / scale)
    return predicted


def clustered_interval(
    errors_base: np.ndarray, errors_full: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    names = np.unique(clusters)
    index = {name: np.flatnonzero(clusters == name) for name in names}
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = errors_base[rows].mean() - errors_full[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def evaluate(frame: pd.DataFrame, clock: list[str], routing: list[str], block: str) -> dict:
    groups = frame[block].to_numpy()
    n_splits = min(5, len(np.unique(groups)))
    base = fold_predictions(frame, clock, groups, n_splits)
    full = fold_predictions(frame, clock + routing, groups, n_splits)
    target = frame["remaining"].to_numpy(np.float64)
    errors_base = np.abs(base - target)
    errors_full = np.abs(full - target)
    variance = target.var()
    rng = np.random.default_rng(SEED)
    low, high = clustered_interval(errors_base, errors_full, frame["task"].to_numpy(), rng)
    return {
        "blocked_by": block,
        "folds": n_splits,
        "rows": int(len(frame)),
        "episodes": int(frame["episode_row"].nunique()),
        "target_sd": float(np.sqrt(variance)),
        "mae_clock": float(errors_base.mean()),
        "mae_clock_routing": float(errors_full.mean()),
        "mae_gain": float(errors_base.mean() - errors_full.mean()),
        "mae_gain_ci95": [low, high],
        "r2_clock": float(1 - ((base - target) ** 2).mean() / variance),
        "r2_clock_routing": float(1 - ((full - target) ** 2).mean() / variance),
    }


def conditional_correlation(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Spearman against remaining duration inside each (task, chunk) cell.

    Model-free, and it settles the question more directly than the regression:
    holding the task and the elapsed time fixed, the rows differ only in how much
    longer they actually took.  If routing carried progress state, it would
    correlate here.
    """
    records = []
    for name in features:
        weights, rhos = [], []
        for _, cell in frame.groupby(["task", "chunk"], sort=False):
            if len(cell) < 30:
                continue
            x = cell[name].to_numpy(np.float64)
            y = cell["remaining"].to_numpy(np.float64)
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            rho = np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
            if np.isfinite(rho):
                rhos.append(rho)
                weights.append(len(cell))
        rhos, weights = np.asarray(rhos), np.asarray(weights, dtype=float)
        records.append(
            {
                "feature": name,
                "weighted_spearman": float((rhos * weights).sum() / weights.sum()),
                "cells": int(len(rhos)),
                "rows": int(weights.sum()),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cache, success, task, suite, entropy = load()
    frame, clock, routing = build_rows(cache, success, task, suite, entropy)

    headline = ["mobility_L12", "entropy_L12", "R4_all", "L4_all"]
    conditional = conditional_correlation(frame, headline)
    conditional.to_csv(args.output / "conditional_correlation.csv", index=False)
    print("固定 (task, chunk) 后 routing vs 剩余时长：")
    print(conditional.to_string(index=False))
    print()

    results = [evaluate(frame, clock, routing, block) for block in ("task", "suite")]
    table = pd.DataFrame(results)
    table.to_csv(args.output / "remaining_duration.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "question": "does routing beat the clock at predicting remaining chunks on healthy rollouts",
                "target": "chunks still to go = length - 1 - chunk, successful episodes only",
                "n_routing_features": len(routing),
                "results": results,
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 200)
    print(
        table[
            [
                "blocked_by",
                "rows",
                "episodes",
                "target_sd",
                "mae_clock",
                "mae_clock_routing",
                "mae_gain",
                "mae_gain_ci95",
                "r2_clock",
                "r2_clock_routing",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
