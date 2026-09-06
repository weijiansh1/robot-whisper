#!/usr/bin/env python3
"""Can routing segment itself, without any physical or outcome label?

The activity-network reading needs a milestone that is readable online. A
supervised test already showed routing carries a coarse phase (approach /
closed / transport) at weighted AUC 0.707 within (task, chunk) cells. But that
phase came from physical annotations, which a real deployment cannot obtain.

So the construction has to be inverted: discover the segments from routing
alone, and use physics only afterwards, once, in simulation, to check that what
was discovered corresponds to something real. Nothing here reads an outcome
label or a physical quantity — the clustering is fitted on every valid chunk of
every episode.

Three competing explanations must be separated, because a segment label that is
really task identity or really the clock is useless to a task-agnostic monitor:

    AMI(regime, task)         is it just recognising the task
    AMI(regime, chunk band)   is it just the clock
    AMI(regime, phase)        does it carry the milestone

and the decisive one, within (task, chunk) cells, where rows share the task and
the elapsed time and differ only in whether that rollout has grasped yet.

Persistence is reported too. A segmentation that flickers every chunk cannot
support a dwell-based slack no matter how well it correlates.

Run:  python experiments/discover_regimes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

CACHE = BUNDLE / "results/progress_cache/development_main.npz"
ENTROPY = BUNDLE / "results/ablation/gate_entropy.npz"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
RUN = "seed1000_1007"
DEFAULT_OUTPUT = BUNDLE / "results/regimes"

WINDOW = 2
EPS_LENGTH = 1e-3
REGIME_COUNTS = (3, 4, 6, 8)
CHUNK_BANDS = (0, 3, 6, 10, 16, 26, 52)
MIN_CELL = 30
SEED = 20260906


def phase_labels() -> pd.DataFrame:
    """Coarse physical phase, used for validation only, never for fitting."""
    events = pd.read_csv(TAXONOMY / "belief_mismatch_events.csv.gz")
    events = events[(events["run"] == RUN) & events["coupled_target_motion"].astype(bool)]
    first = (
        events.sort_values("closure_action_query")
        .groupby(["task", "init_state_id", "flow_noise_seed"], as_index=False)
        .first()
    )
    return first[
        [
            "task",
            "init_state_id",
            "flow_noise_seed",
            "closure_action_query",
            "post_closure_state_query",
        ]
    ]


def build() -> tuple[pd.DataFrame, list[str]]:
    cache = dict(np.load(CACHE, allow_pickle=False))
    with np.load(ENTROPY, allow_pickle=False) as archive:
        if not np.array_equal(archive["episode"], cache["episode"]):
            raise ValueError("entropy cache is not row-aligned with the progress cache")
        entropy = archive["entropy"]
    reference = np.load(
        WORKSPACE / "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
        allow_pickle=False,
    )
    if not np.array_equal(reference["episode"], cache["episode"]):
        raise ValueError("v4 reference is not row-aligned with the progress cache")

    adjacent = cache["lag_distance"][:, :, 0, :]
    columns: dict[str, np.ndarray] = {}
    for layer_index, name in enumerate(pr.LAYER_NAMES):
        columns[f"mobility_{name}"] = adjacent[:, :, layer_index]
        columns[f"entropy_{name}"] = entropy[:, :, layer_index]
    path = pr.path_length(adjacent, WINDOW)
    displacement = cache["lag_distance"][:, :, pr.LAGS.index(WINDOW), :]
    ratio = pr.progress_ratio(displacement, path, EPS_LENGTH)
    for group in pr.LAYER_GROUPS:
        columns[f"R_{group}"] = pr.group_ratio(ratio, group)
        columns[f"L_{group}"] = pr.group_ratio(path, group)

    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    episodes, chunks = np.nonzero(cache["valid"])
    frame = pd.DataFrame({name: values[episodes, chunks] for name, values in columns.items()})
    frame["chunk"] = chunks
    frame["episode_row"] = episodes
    frame["task"] = task[episodes]
    frame["init_state_id"] = reference["init_state_id"].astype(int)[episodes]
    frame["flow_noise_seed"] = reference["flow_noise_seed"].astype(int)[episodes]

    features = list(columns)
    frame = frame.dropna(subset=features).reset_index(drop=True)

    labels = phase_labels()
    frame = frame.merge(
        labels, on=["task", "init_state_id", "flow_noise_seed"], how="left"
    )
    closure = frame["closure_action_query"].to_numpy()
    post = frame["post_closure_state_query"].to_numpy()
    chunk = frame["chunk"].to_numpy()
    phase = np.where(chunk < closure, 0, np.where(chunk < post, 1, 2)).astype(float)
    phase[~np.isfinite(closure)] = np.nan
    frame["phase"] = phase
    frame["chunk_band"] = pd.cut(frame["chunk"], bins=CHUNK_BANDS, right=False, labels=False)
    return frame, features


def within_cell_ami(frame: pd.DataFrame, regime: str, label: str) -> tuple[float, int, int]:
    """AMI inside (task, chunk) cells, weighted by cell size."""
    scores, weights = [], []
    subset = frame.dropna(subset=[label])
    for _, cell in subset.groupby(["task", "chunk"], sort=False):
        if len(cell) < MIN_CELL:
            continue
        left = cell[regime].to_numpy()
        right = cell[label].to_numpy()
        if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
            continue
        scores.append(adjusted_mutual_info_score(right, left))
        weights.append(len(cell))
    if not scores:
        return float("nan"), 0, 0
    scores, weights = np.asarray(scores), np.asarray(weights, dtype=float)
    return float((scores * weights).sum() / weights.sum()), len(scores), int(weights.sum())


def persistence(frame: pd.DataFrame, regime: str) -> dict[str, float]:
    changes, transitions, runs = 0, 0, []
    for _, episode in frame.sort_values(["episode_row", "chunk"]).groupby(
        "episode_row", sort=False
    ):
        values = episode[regime].to_numpy()
        if len(values) < 2:
            continue
        switched = values[1:] != values[:-1]
        changes += int(switched.sum())
        transitions += len(switched)
        boundaries = np.flatnonzero(switched) + 1
        runs.extend(np.diff(np.concatenate(([0], boundaries, [len(values)]))))
    return {
        "switch_rate": float(changes / transitions) if transitions else float("nan"),
        "mean_run_length": float(np.mean(runs)) if runs else float("nan"),
        "median_run_length": float(np.median(runs)) if runs else float("nan"),
    }


def trailing_mode(frame: pd.DataFrame, regime: str, width: int) -> np.ndarray:
    """Causal mode smoothing: at chunk q, the modal label over q-width+1 .. q.

    Trailing, never centred, so a smoothed label stays usable online.
    """
    if width <= 1:
        return frame[regime].to_numpy()
    output = np.empty(len(frame), dtype=int)
    order = frame.sort_values(["episode_row", "chunk"]).index.to_numpy()
    values = frame.loc[order, regime].to_numpy()
    owners = frame.loc[order, "episode_row"].to_numpy()
    start = 0
    for position in range(len(order)):
        if position and owners[position] != owners[position - 1]:
            start = position
        window = values[max(start, position - width + 1) : position + 1]
        counts = np.bincount(window)
        output[position] = counts.argmax()
    result = np.empty(len(frame), dtype=int)
    result[order] = output
    return result


def smoothing_sweep(frame: pd.DataFrame, regime: str, widths: tuple[int, ...]) -> list[dict]:
    """Does the phase signal survive being made persistent enough to time?

    A slack rule needs a dwell, and a dwell needs runs longer than one chunk. If
    the within-cell agreement with phase collapses as soon as the labels are
    made persistent, the per-chunk agreement was flicker rather than segment
    structure.
    """
    records = []
    for width in widths:
        column = f"{regime}_w{width}"
        frame[column] = trailing_mode(frame, regime, width)
        ami, cells, rows = within_cell_ami(frame, column, "phase")
        records.append(
            {
                "smoothing": width,
                "within_cell_ami_phase": ami,
                "within_cells": cells,
                "within_rows": rows,
                **persistence(frame, column),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame, features = build()
    matrix = frame[features].to_numpy(np.float64)
    matrix = (matrix - matrix.mean(axis=0)) / np.maximum(matrix.std(axis=0), 1e-12)

    rng = np.random.default_rng(SEED)
    records = []
    for count in REGIME_COUNTS:
        column = f"regime_{count}"
        model = MiniBatchKMeans(n_clusters=count, random_state=SEED, n_init=10, batch_size=4096)
        frame[column] = model.fit_predict(matrix)
        shuffled = rng.permutation(frame[column].to_numpy())

        labelled = frame.dropna(subset=["phase"])
        cell_ami, cells, rows = within_cell_ami(frame, column, "phase")
        frame["_shuffled"] = shuffled
        null_ami, _, _ = within_cell_ami(frame, "_shuffled", "phase")
        records.append(
            {
                "regimes": count,
                "ami_task": adjusted_mutual_info_score(frame["task"], frame[column]),
                "ami_chunk_band": adjusted_mutual_info_score(
                    frame["chunk_band"], frame[column]
                ),
                "ami_phase": adjusted_mutual_info_score(
                    labelled["phase"], labelled[column]
                ),
                "within_cell_ami_phase": cell_ami,
                "within_cell_ami_shuffled": null_ami,
                "within_cells": cells,
                "within_rows": rows,
                **persistence(frame, column),
            }
        )
        print(f"  fitted K={count}", flush=True)

    best = max(records, key=lambda row: row["within_cell_ami_phase"])["regimes"]

    # Time localisation decides the other slack form.  slack(q) = vl(p) - q needs
    # vl(p), the latest chunk at which a rollout was still short of milestone p,
    # to differ between milestones.  If every discovered regime spans the whole
    # episode, vl is the horizon cap for all of them and the rule is vacuous.
    spread = []
    for column, name in [(f"regime_{best}", "regime"), ("phase", "physical_phase")]:
        subset = frame.dropna(subset=[column])
        for value, group in subset.groupby(column):
            quantiles = group["chunk"].quantile([0.05, 0.5, 0.95, 0.99])
            spread.append(
                {
                    "source": name,
                    "label": int(value),
                    "rows": int(len(group)),
                    "chunk_p05": float(quantiles.iloc[0]),
                    "chunk_p50": float(quantiles.iloc[1]),
                    "chunk_p95": float(quantiles.iloc[2]),
                    "chunk_p99": float(quantiles.iloc[3]),
                    "vl_max_chunk": int(group["chunk"].max()),
                }
            )
    spread_table = pd.DataFrame(spread)
    spread_table.to_csv(args.output / "time_localisation.csv", index=False)
    print("时间定位（vl 若对所有标签相同，绝对-chunk 松弛失效）：")
    print(spread_table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))
    print()
    sweep = pd.DataFrame(smoothing_sweep(frame, f"regime_{best}", (1, 2, 3, 4, 6)))
    sweep.insert(0, "regimes", best)
    sweep.to_csv(args.output / "smoothing_sweep.csv", index=False)

    table = pd.DataFrame(records)
    table.to_csv(args.output / "regime_validation.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "features": features,
                "fitted_on": "all valid chunks, no outcome or physical label",
                "phase_used_for": "validation only",
                "results": records,
                "smoothing_sweep": sweep.to_dict("records"),
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 220)
    print()
    print("时间平滑扫描（K=%d，因果拖尾众数）：" % best)
    print(sweep.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(
        table[
            [
                "regimes",
                "ami_task",
                "ami_chunk_band",
                "ami_phase",
                "within_cell_ami_phase",
                "within_cell_ami_shuffled",
                "within_cells",
                "switch_rate",
                "median_run_length",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
