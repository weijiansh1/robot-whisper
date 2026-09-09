#!/usr/bin/env python3
"""Redo the sim+MoE minus sim increment CI with initial states resampled.

`analyze_early_structure.paired_bootstrap_difference` resamples rollouts inside
fixed initial states, so its interval carries no between-state variance.  The
point estimate is unchanged; only the interval is recomputed.
"""

import json

import numpy as np


DRAWS = 20000
HORIZONS = (7, 12, 20, 27, 34)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    positive, negative = s[y == 1], s[y == 0]
    wins = float((positive[:, None] > negative[None, :]).sum())
    wins += 0.5 * float((positive[:, None] == negative[None, :]).sum())
    return wins / (len(positive) * len(negative))


def paired(labels, left, right, blocks) -> float:
    left_sum = right_sum = 0.0
    denominator = 0
    for index in blocks:
        y = labels[index]
        if len(np.unique(y)) < 2:
            continue
        weight = int(y.sum()) * int((1 - y).sum())
        left_sum += auc(y, left[index]) * weight
        right_sum += auc(y, right[index]) * weight
        denominator += weight
    return (left_sum - right_sum) / denominator if denominator else float("nan")


def main() -> None:
    data = np.load("token-dynamics/early_structure_scores.npz", allow_pickle=False)
    labels = data["label"]
    included = data["included"].astype(bool)
    state = data["state"]
    blocks = [np.flatnonzero(included & (state == v)) for v in np.unique(state[included])]
    blocks = [b for b in blocks if len(np.unique(labels[b])) == 2]
    published = json.loads(
        open("token-dynamics/early_structure_summary.json").read()
    )["sim_moe_increment"]

    print("%4s %8s  %26s  %26s  %s" % ("t", "point", "published CI", "state-resampled CI", "verdict"))
    rows = []
    for horizon in HORIZONS:
        left = data["score__sim_plus_moe__t%d" % horizon].astype(np.float64)
        right = data["score__sim_history__t%d" % horizon].astype(np.float64)
        point = paired(labels, left, right, blocks)
        rng = np.random.default_rng(20260827)
        count = len(blocks)
        values = []
        while len(values) < DRAWS:
            chosen = rng.integers(0, count, size=count)
            drawn = [
                rng.choice(blocks[axis], size=len(blocks[axis]), replace=True)
                for axis in chosen
            ]
            value = paired(labels, left, right, drawn)
            if np.isfinite(value):
                values.append(value)
        low, high = np.quantile(values, [0.025, 0.975])
        reference = published[str(horizon)]["ci"]
        if low > 0:
            verdict = "holds"
        elif reference[0] > 0:
            verdict = "CROSSES 0"
        else:
            verdict = "-"
        print(
            "t%-3d %+8.3f  [%+7.3f, %+7.3f]      [%+7.3f, %+7.3f]  %s"
            % (horizon, point, reference[0], reference[1], low, high, verdict)
        )
        rows.append(
            {
                "horizon": horizon,
                "point": float(point),
                "published_ci": reference,
                "state_resampled_ci": [float(low), float(high)],
                "verdict": verdict,
            }
        )
    with open("token-dynamics/paired_increment_ci.json", "w") as handle:
        json.dump({"draws": DRAWS, "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
