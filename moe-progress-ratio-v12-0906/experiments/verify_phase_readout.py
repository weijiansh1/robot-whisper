#!/usr/bin/env python3
"""Does routing know which milestone the rollout has reached?

`verify_remaining_duration.py` asked whether routing predicts how many chunks a
healthy rollout still needs, and found nothing.  That target was the wrong one
for the activity-network reading, in two specific ways.

First, remaining *time* is not remaining *work*.  Two rollouts at the same point
on the path can finish in three chunks or eight depending on downstream
randomness, and no readout of the present can know which -- so the target
carried irreducible noise unrelated to progress.

Second, it collapsed the whole activity network into one scalar.  Slack in an
activity network is defined per event: the latest time event j may occur.  The
question the framing actually poses is not "how long until done" but "which
milestone am I at", and only then does a deadline attach to it.

That question has dense labels here.  The trap taxonomy records, per gripper
closure, the closing query and the first state query after it, so a successful
rollout splits into three phases:

    approach   : chunk <  closure_action_query
    closed     : closure_action_query <= chunk < post_closure_state_query
    transport  : chunk >= post_closure_state_query

The boundary sits at 38% of the trajectory in the median, so it is not
recoverable from the clock alone.

Two tests, the second decisive:

1. Grouped-by-task CV, multinomial phase classification, routing against a clock
   baseline.
2. Model-free: inside each (task, chunk) cell, rows share the task and the
   elapsed time and differ only in whether this particular rollout has grasped
   yet.  If routing carries position on the path, it separates them here.

Run:  python experiments/verify_phase_readout.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
RUN = "seed1000_1007"
DEFAULT_OUTPUT = BUNDLE / "results/phase_readout"

# Small windows only.  The grasp boundary sits at chunk 5 in the median, so a
# window needing chunk >= 8 would drop almost the entire approach phase through
# the dropna below and leave the model conditioning on "already past chunk 8".
WINDOWS = (2,)
EPS_LENGTH = 1e-3
MIN_CELL = 30
DRAWS = 2000
SEED = 20260906
PHASES = ("approach", "closed", "transport")


def milestones() -> pd.DataFrame:
    """First successful grasp closure per episode, from the taxonomy tables."""
    events = pd.read_csv(TAXONOMY / "belief_mismatch_events.csv.gz")
    events = events[(events["run"] == RUN) & events["success"].astype(bool)]
    coupled = events[events["coupled_target_motion"].astype(bool)]
    first = (
        coupled.sort_values("closure_action_query")
        .groupby(["task", "episode"], as_index=False)
        .first()
    )
    return first[["task", "episode", "closure_action_query", "post_closure_state_query"]]


def load() -> tuple[pd.DataFrame, list[str], list[str]]:
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
    merged = index.merge(labels, on=["task", "episode"], how="left", validate="one_to_one")
    if merged["original_failure"].isna().any():
        raise ValueError("labels do not align with the progress cache")
    merged = merged.merge(milestones(), on=["task", "episode"], how="inner", validate="one_to_one")
    merged = merged[~merged["original_failure"].astype(bool)].sort_values("row")

    with np.load(ENTROPY, allow_pickle=False) as archive:
        if not np.array_equal(archive["episode"], cache["episode"]):
            raise ValueError("entropy cache is not row-aligned with the progress cache")
        entropy = archive["entropy"]

    adjacent = cache["lag_distance"][:, :, 0, :]
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

    rows = merged["row"].to_numpy()
    valid = cache["valid"][rows]
    episodes, chunks = np.nonzero(valid)
    source = rows[episodes]

    frame = pd.DataFrame({name: values[source, chunks] for name, values in columns.items()})
    frame["chunk"] = chunks
    frame["chunk_sq"] = chunks**2
    frame["chunk_log"] = np.log1p(chunks)
    frame["task"] = merged["task"].to_numpy()[episodes]
    closure = merged["closure_action_query"].to_numpy()[episodes]
    post = merged["post_closure_state_query"].to_numpy()[episodes]
    frame["phase"] = np.where(chunks < closure, 0, np.where(chunks < post, 1, 2))

    routing = [name for name in columns]
    clock = ["chunk", "chunk_sq", "chunk_log"]
    return frame.dropna(subset=routing).reset_index(drop=True), clock, routing


def fold_probabilities(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    matrix = frame[features].to_numpy(np.float64)
    target = frame["phase"].to_numpy(int)
    groups = frame["task"].to_numpy()
    predicted = np.full((len(frame), len(PHASES)), np.nan)
    for train, test in GroupKFold(n_splits=5).split(matrix, target, groups):
        centre = matrix[train].mean(axis=0)
        scale = matrix[train].std(axis=0)
        scale[scale < 1e-12] = 1.0
        model = LogisticRegression(max_iter=2000, C=1.0).fit(
            (matrix[train] - centre) / scale, target[train]
        )
        probabilities = model.predict_proba((matrix[test] - centre) / scale)
        for position, label in enumerate(model.classes_):
            predicted[test, label] = probabilities[:, position]
    return np.nan_to_num(predicted, nan=1e-9)


def clustered_interval(
    left: np.ndarray, right: np.ndarray, clusters: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    names = np.unique(clusters)
    index = {name: np.flatnonzero(clusters == name) for name in names}
    samples = np.empty(DRAWS)
    for draw in range(DRAWS):
        picked = rng.integers(0, len(names), len(names))
        rows = np.concatenate([index[names[position]] for position in picked])
        samples[draw] = left[rows].mean() - right[rows].mean()
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def conditional_auc(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Within (task, chunk): does routing say whether this rollout has grasped yet?"""
    records = []
    for name in features:
        weights, aucs = [], []
        for _, cell in frame.groupby(["task", "chunk"], sort=False):
            grasped = (cell["phase"] > 0).to_numpy()
            if grasped.sum() < 10 or (~grasped).sum() < 10 or len(cell) < MIN_CELL:
                continue
            values = cell[name].to_numpy(np.float64)
            if not np.isfinite(values).all() or np.std(values) == 0:
                continue
            ranks = pd.Series(values).rank().to_numpy()
            positive, negative = grasped.sum(), (~grasped).sum()
            auc = (ranks[grasped].sum() - positive * (positive + 1) / 2) / (positive * negative)
            aucs.append(auc)
            weights.append(len(cell))
        aucs, weights = np.asarray(aucs), np.asarray(weights, dtype=float)
        if not len(aucs):
            continue
        records.append(
            {
                "feature": name,
                "weighted_auc": float((aucs * weights).sum() / weights.sum()),
                "cells": int(len(aucs)),
                "rows": int(weights.sum()),
            }
        )
    return pd.DataFrame(records).sort_values(
        "weighted_auc", key=lambda column: (column - 0.5).abs(), ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame, clock, routing = load()
    target = frame["phase"].to_numpy(int)
    shares = {name: float((target == index).mean()) for index, name in enumerate(PHASES)}

    base = fold_probabilities(frame, clock)
    full = fold_probabilities(frame, clock + routing)
    losses_base = -np.log(np.clip(base[np.arange(len(target)), target], 1e-12, None))
    losses_full = -np.log(np.clip(full[np.arange(len(target)), target], 1e-12, None))
    rng = np.random.default_rng(SEED)
    low, high = clustered_interval(losses_base, losses_full, frame["task"].to_numpy(), rng)

    conditional = conditional_auc(frame, routing)
    conditional.to_csv(args.output / "conditional_auc.csv", index=False)

    summary = {
        "rows": int(len(frame)),
        "episodes": int(len(frame.groupby(["task", "chunk"]).ngroup().unique())),
        "phase_shares": shares,
        "log_loss_clock": float(log_loss(target, base, labels=[0, 1, 2])),
        "log_loss_clock_routing": float(log_loss(target, full, labels=[0, 1, 2])),
        "gain": float(losses_base.mean() - losses_full.mean()),
        "gain_ci95": [low, high],
        "accuracy_clock": float((base.argmax(1) == target).mean()),
        "accuracy_clock_routing": float((full.argmax(1) == target).mean()),
        "conditional_auc_top": conditional.head(5).to_dict("records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"rows {summary['rows']:,}   phase shares {shares}")
    print(
        f"log-loss  clock {summary['log_loss_clock']:.4f}"
        f"   +routing {summary['log_loss_clock_routing']:.4f}"
        f"   gain {summary['gain']:+.4f} [{low:+.4f}, {high:+.4f}]"
    )
    print(
        f"accuracy  clock {summary['accuracy_clock']:.4f}"
        f"   +routing {summary['accuracy_clock_routing']:.4f}"
    )
    print("\n固定 (task, chunk) 后，路由能否说出这条 rollout 是否已经抓到：")
    print(conditional.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
