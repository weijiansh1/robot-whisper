#!/usr/bin/env python3
"""Split the per-step cross-query mobility into a noise floor and a remainder.

Between two adjacent queries two things change: the observation, and the flow
noise, which is re-drawn every query.  The query-0 contrast already isolates the
second one -- inside an initial-state group at query 0 the observation is
identical, so the mean pairwise route distance there is what a fresh noise draw
alone buys at that step.  That number is on exactly the same scale as adjacent
query mobility, because both compare two independent noise draws.

So for each layer and step:

    noise_floor(l, s)    query-0 same-observation, different-seed distance
    mobility(l, s)       adjacent-query action-route distance
    remainder(l, s)      mobility - noise_floor

and the share of mobility that survives the floor says how much of what the
step-s detector reads is a response to the scene rather than to the sampler.

The script also reports the normalised step-to-step increment of every quantity,
which is where a change point on the step axis would show up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
STRUCTURE = BUNDLE / "results/step_structure"
PROFILES = BUNDLE / "results/step_profiles"
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
COHORTS = ("development_main", "development_extra", "external_8b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", type=Path, default=STRUCTURE)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    return parser.parse_args()


def episode_icc(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """One-way random-effects ICC of a [n, q, 8, 10] array over episodes.

    If a step's mobility were nothing but the response to an independent flow
    noise draw with an episode-independent gain, almost all of its variance
    would sit inside episodes.  The share that sits between episodes is how much
    of it is a property of the rollout rather than of the sampler.
    """
    counts = valid.sum(axis=1)
    keep = counts >= 2
    values = values[keep]
    mask = valid[keep][:, :, None, None]
    counts = counts[keep].astype(np.float64)
    filled = np.where(mask & np.isfinite(values), values, 0.0).astype(np.float64)
    present = (mask & np.isfinite(values)).astype(np.float64)
    per_episode = filled.sum(axis=1)
    per_count = present.sum(axis=1)
    episode_mean = per_episode / np.maximum(per_count, 1.0)
    grand = per_episode.sum(axis=0) / np.maximum(per_count.sum(axis=0), 1.0)

    between = (per_count * (episode_mean - grand) ** 2).sum(axis=0)
    residual = (filled - episode_mean[:, None] * present) ** 2
    within = (residual * present).sum(axis=(0, 1))
    episodes = float(len(values))
    total = per_count.sum(axis=0)
    mean_between = between / (episodes - 1.0)
    mean_within = within / np.maximum(total - episodes, 1.0)
    group_size = (total - (per_count**2).sum(axis=0) / total) / (episodes - 1.0)
    return (mean_between - mean_within) / np.maximum(
        mean_between + (group_size - 1.0) * mean_within, 1e-30
    )


def main() -> None:
    args = parse_args()
    profiles = pd.read_csv(args.structure / "step_profiles.csv")
    pairs = pd.read_csv(args.structure / "query0_noise_observation.csv")

    mobility = profiles[profiles["quantity"] == "mobility"][
        ["cohort", "layer", "step", "mean"]
    ].rename(columns={"mean": "mobility"})
    floor = pairs[pairs["token_family"] == "action"][
        [
            "cohort",
            "layer",
            "step",
            "same_state_diff_noise_mean",
            "diff_state_same_noise_mean",
        ]
    ].rename(
        columns={
            "same_state_diff_noise_mean": "noise_floor",
            "diff_state_same_noise_mean": "observation_spread",
        }
    )
    merged = mobility.merge(floor, on=["cohort", "layer", "step"], validate="one_to_one")
    merged["remainder"] = merged["mobility"] - merged["noise_floor"]
    merged["remainder_share"] = merged["remainder"] / merged["mobility"]
    merged["observation_over_noise"] = (
        merged["observation_spread"] / merged["noise_floor"]
    )
    merged.to_csv(args.structure / "mobility_vs_noise_floor.csv", index=False)

    # Normalised step increments: where would a change point sit?
    increments = []
    for (cohort, quantity, layer), block in profiles.groupby(
        ["cohort", "quantity", "layer"]
    ):
        block = block.sort_values("step")
        values = block["mean"].to_numpy()
        if not np.isfinite(values).all():
            values = values[1:]
            offset = 1
        else:
            offset = 0
        span = float(np.nanmax(values) - np.nanmin(values))
        if span <= 0:
            continue
        delta = np.diff(values) / span
        for position, value in enumerate(delta):
            increments.append(
                {
                    "cohort": cohort,
                    "quantity": quantity,
                    "layer": layer,
                    "from_step": position + offset,
                    "to_step": position + offset + 1,
                    "normalised_increment": float(value),
                    "span": span,
                }
            )
    increment_frame = pd.DataFrame(increments)
    increment_frame.to_csv(args.structure / "step_increments.csv", index=False)

    # How much of each step's mobility is a property of the rollout?
    icc_rows = []
    for cohort in COHORTS:
        with np.load(args.profiles / f"{cohort}_index.npz", allow_pickle=False) as index:
            valid = index["valid"].astype(bool)
            task_index = index["task_index"].astype(int)
            task_names = index["task_names"].astype(str)
        block = np.load(args.profiles / f"{cohort}_mobility.npy", mmap_mode="r")
        step_valid = valid.copy()
        step_valid[:, 0] = False
        for position, task in enumerate(task_names):
            rows = np.flatnonzero(task_index == position)
            icc = episode_icc(
                np.asarray(block[rows], dtype=np.float32), step_valid[rows]
            )
            for layer in range(8):
                for step in range(10):
                    icc_rows.append(
                        {
                            "cohort": cohort,
                            "task": task,
                            "layer": LAYER_NAMES[layer],
                            "step": step,
                            "icc": float(icc[layer, step]),
                        }
                    )
        print(f"  icc done: {cohort}", flush=True)
    icc_frame = pd.DataFrame(icc_rows)
    icc_frame.to_csv(args.structure / "mobility_episode_icc.csv", index=False)

    pd.set_option("display.width", 240)
    print("=== development_main: mobility vs the query-0 noise floor ===")
    for column in ("mobility", "noise_floor", "remainder", "remainder_share"):
        print(f"--- {column} ---")
        print(
            merged[merged["cohort"] == "development_main"]
            .pivot(index="layer", columns="step", values=column)
            .reindex(LAYER_NAMES)
            .to_string(float_format="%.4f")
        )

    print("\n=== largest normalised increment on the step axis, development_main ===")
    block = increment_frame[increment_frame["cohort"] == "development_main"]
    peak = (
        block.loc[block.groupby(["quantity", "layer"])["normalised_increment"].apply(
            lambda s: s.abs().idxmax()
        )]
        .sort_values(["quantity", "layer"])
    )
    print(
        peak[["quantity", "layer", "from_step", "to_step", "normalised_increment"]]
        .to_string(index=False, float_format="%.3f")
    )
    print("\n=== median across tasks of the episode ICC of step mobility ===")
    for cohort in ("development_main", "external_8b"):
        print(f"--- {cohort} ---")
        print(
            icc_frame[icc_frame["cohort"] == cohort]
            .groupby(["layer", "step"])["icc"]
            .median()
            .unstack()
            .reindex(LAYER_NAMES)
            .to_string(float_format="%.3f")
        )

    summary = {
        "schema": "himoe.flow_semantics.step_decomposition.v1",
        "noise_floor_definition": "query-0 mean pairwise action-route Hellinger distance between episodes with the same initial state and different flow noise seed",
        "comparable_because": "adjacent-query mobility and the query-0 within-initial-state contrast both compare two independent flow noise draws",
        "outcomes_loaded": False,
    }
    (args.structure / "decomposition_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
