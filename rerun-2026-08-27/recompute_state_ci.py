#!/usr/bin/env python3
"""Recompute early-structure AUC intervals with and without state resampling.

`analyze_early_structure.bootstrap_auc` resamples rollouts inside each initial
state but never resamples the states themselves.  The headline statistic is a
pair-weighted macro over 16 states and the folds hold out whole states, so the
published interval carries only within-state noise.  This script reuses the
cross-fitted scores and reports both intervals side by side; the point estimate
is identical by construction.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def roc_auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC; matches sklearn on these 32-sample blocks."""
    positive = s[y == 1]
    negative = s[y == 0]
    wins = float((positive[:, None] > negative[None, :]).sum())
    wins += 0.5 * float((positive[:, None] == negative[None, :]).sum())
    return wins / (len(positive) * len(negative))


FAMILIES = (
    "route_history",
    "hidden_history",
    "moe_current",
    "moe_history",
    "proprio_history",
    "sim_history",
    "sim_plus_moe",
)
HORIZONS = (7, 12, 20, 27, 34)
DRAWS = 2000


def pair_weighted(labels: np.ndarray, scores: np.ndarray, blocks: list[np.ndarray]) -> float:
    numerator = 0.0
    denominator = 0
    for index in blocks:
        y = labels[index]
        if len(np.unique(y)) < 2:
            continue
        pairs = int(y.sum()) * int((1 - y).sum())
        numerator += roc_auc_score(y, scores[index]) * pairs
        denominator += pairs
    return numerator / denominator if denominator else float("nan")


def informative_blocks(labels, state, included) -> list[np.ndarray]:
    blocks = []
    for value in np.unique(state[included]):
        index = np.flatnonzero(included & (state == value))
        if len(np.unique(labels[index])) == 2:
            blocks.append(index)
    return blocks


def within_state_ci(labels, scores, blocks, rng) -> tuple[float, float]:
    """Reproduce the published interval: resample rollouts inside fixed states."""
    values = []
    while len(values) < DRAWS:
        drawn = [rng.choice(index, size=len(index), replace=True) for index in blocks]
        value = pair_weighted(labels, scores, drawn)
        if np.isfinite(value):
            values.append(value)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def two_level_ci(labels, scores, blocks, rng) -> tuple[float, float]:
    """Cluster bootstrap: resample states, then rollouts inside each drawn state."""
    values = []
    count = len(blocks)
    while len(values) < DRAWS:
        chosen = rng.integers(0, count, size=count)
        drawn = [
            rng.choice(blocks[axis], size=len(blocks[axis]), replace=True)
            for axis in chosen
        ]
        value = pair_weighted(labels, scores, drawn)
        if np.isfinite(value):
            values.append(value)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def state_level_ci(labels, scores, blocks, rng) -> tuple[float, float]:
    """Resample whole states with replacement, keeping each state's rollouts intact."""
    values = []
    count = len(blocks)
    while len(values) < DRAWS:
        chosen = rng.integers(0, count, size=count)
        value = pair_weighted(labels, scores, [blocks[axis] for axis in chosen])
        if np.isfinite(value):
            values.append(value)
    return tuple(float(x) for x in np.quantile(values, [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=pathlib.Path, required=True)
    parser.add_argument("--published", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    payload = np.load(args.scores, allow_pickle=False)
    labels = np.asarray(payload["label"])
    included = np.asarray(payload["included"], dtype=bool)
    state = np.asarray(payload["state"])
    published = json.loads(args.published.read_text())["evaluation"]

    blocks = informative_blocks(labels, state, included)
    rows = []
    for family in FAMILIES:
        for horizon in HORIZONS:
            key = "score__%s__t%d" % (family, horizon)
            scores = np.asarray(payload[key], dtype=np.float64)
            point = pair_weighted(labels, scores, blocks)
            rng = np.random.default_rng(args.seed)
            within = within_state_ci(labels, scores, blocks, rng)
            rng = np.random.default_rng(args.seed)
            two_level = two_level_ci(labels, scores, blocks, rng)
            rng = np.random.default_rng(args.seed)
            state_only = state_level_ci(labels, scores, blocks, rng)
            reference = published[family][str(horizon)]
            rows.append(
                {
                    "family": family,
                    "horizon": horizon,
                    "auc": point,
                    "published_auc": reference["pair_weighted_within_state_auc"],
                    "published_ci": reference["bootstrap_95_ci"],
                    "within_state_ci": list(within),
                    "state_level_ci": list(state_only),
                    "two_level_ci": list(two_level),
                    "published_width": reference["bootstrap_95_ci"][1]
                    - reference["bootstrap_95_ci"][0],
                    "two_level_width": two_level[1] - two_level[0],
                    "crosses_chance_published": reference["bootstrap_95_ci"][0] <= 0.5,
                    "crosses_chance_two_level": two_level[0] <= 0.5,
                }
            )
            print(
                "%-18s t%-3d auc %.3f | published [%.3f, %.3f] | two-level [%.3f, %.3f]"
                % (
                    family,
                    horizon,
                    point,
                    reference["bootstrap_95_ci"][0],
                    reference["bootstrap_95_ci"][1],
                    two_level[0],
                    two_level[1],
                ),
                flush=True,
            )
    args.out.write_text(json.dumps({"draws": DRAWS, "rows": rows}, indent=2) + "\n")
    print("wrote %s" % args.out, flush=True)


if __name__ == "__main__":
    main()
