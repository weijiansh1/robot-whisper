#!/usr/bin/env python3
"""Does a learned milestone graph transfer to a suite it was never trained on?

Learning is allowed here, on one condition: the structure has to move. A phase
readout that must be recalibrated per task is not a graph, it is a per-task
lookup, and the whole activity-network construction collapses back into the
task-dependent regime the rest of this bundle spent its effort escaping.

Two things have to transfer, and they are separable:

    nodes   can a routing -> milestone readout trained on other suites label a
            held-out suite

    edges   does vl(p), the latest chunk at which a rollout was still short of
            milestone p, agree across suites once expressed on a common scale

The node half is already known to survive held-out *tasks*: grouped-by-task CV
gives a log-loss gain of 0.141 nats [0.076, 0.203] over a clock baseline and
accuracy 0.904. Suites are the harder cut, and this repository's own
leave-one-suite-out work found that suite-level corpus composition, not task
identity, is what moves thresholds.

The edge half is the one that decides portability. Absolute chunk counts cannot
transfer, because the four suites have different horizon caps; the question is
whether they agree after normalising by the cap, which is a deployment constant
rather than runtime task identity.

Phase labels come from the taxonomy's gripper-closure annotations and exist only
for prehensile episodes, so non-prehensile tasks drop out and are counted.

Run:  python experiments/verify_graph_transfer.py
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

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
WORKSPACE = BUNDLE.parent
sys.path.insert(0, str(BUNDLE / "method"))

import progress_ratio as pr  # noqa: E402

CACHE = BUNDLE / "results/progress_cache/development_main.npz"
ENTROPY = BUNDLE / "results/ablation/gate_entropy.npz"
TAXONOMY = WORKSPACE / "analysis_trap_taxonomy/results"
REFERENCE = WORKSPACE / "moe-v4-0904/results/layerwise_mobility/main_reference.npz"
LABELS = (
    WORKSPACE
    / "double-selete/trainfree/results/timeout_extension_plus10"
    / "development_main_clean_labels.csv"
)
RUN = "seed1000_1007"
DEFAULT_OUTPUT = BUNDLE / "results/graph_transfer"

WINDOW = 2
EPS_LENGTH = 1e-3
PHASES = ("approach", "closed", "transport")
VL_QUANTILE = 0.95
SEED = 20260906


def milestones() -> pd.DataFrame:
    events = pd.read_csv(TAXONOMY / "belief_mismatch_events.csv.gz")
    events = events[(events["run"] == RUN) & events["coupled_target_motion"].astype(bool)]
    return (
        events.sort_values("closure_action_query")
        .groupby(["task", "init_state_id", "flow_noise_seed"], as_index=False)
        .first()[
            [
                "task",
                "init_state_id",
                "flow_noise_seed",
                "closure_action_query",
                "post_closure_state_query",
            ]
        ]
    )


def build() -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    cache = dict(np.load(CACHE, allow_pickle=False))
    with np.load(ENTROPY, allow_pickle=False) as archive:
        if not np.array_equal(archive["episode"], cache["episode"]):
            raise ValueError("entropy cache is not row-aligned")
        entropy = archive["entropy"]
    reference = np.load(REFERENCE, allow_pickle=False)
    if not np.array_equal(reference["episode"], cache["episode"]):
        raise ValueError("v4 reference is not row-aligned")

    task = cache["task_names"].astype(str)[cache["task_index"].astype(int)]
    suite = pd.Series(task).str.split("/", n=1).str[0].to_numpy()
    outcome = pd.DataFrame(
        {"task": task, "episode": cache["episode"].astype(int), "row": np.arange(len(task))}
    ).merge(
        pd.read_csv(LABELS)[["task", "episode", "original_failure"]],
        on=["task", "episode"],
        how="left",
        validate="one_to_one",
    )
    if outcome["original_failure"].isna().any():
        raise ValueError("outcome labels do not align")
    success = ~outcome.sort_values("row")["original_failure"].to_numpy(bool)

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

    valid = cache["valid"] & success[:, None]
    episodes, chunks = np.nonzero(valid)
    frame = pd.DataFrame({name: values[episodes, chunks] for name, values in columns.items()})
    frame["chunk"] = chunks
    frame["chunk_sq"] = chunks**2
    frame["chunk_log"] = np.log1p(chunks)
    frame["task"] = task[episodes]
    frame["suite"] = suite[episodes]
    frame["init_state_id"] = reference["init_state_id"].astype(int)[episodes]
    frame["flow_noise_seed"] = reference["flow_noise_seed"].astype(int)[episodes]

    features = list(columns)
    frame = frame.dropna(subset=features).reset_index(drop=True)
    before = frame["task"].nunique()
    frame = frame.merge(
        milestones(), on=["task", "init_state_id", "flow_noise_seed"], how="inner"
    )
    dropped = before - frame["task"].nunique()

    chunk = frame["chunk"].to_numpy()
    frame["phase"] = np.where(
        chunk < frame["closure_action_query"].to_numpy(),
        0,
        np.where(chunk < frame["post_closure_state_query"].to_numpy(), 1, 2),
    )
    caps = {
        str(name): int(group["chunk"].max()) + 1
        for name, group in frame.groupby("suite")
    }
    frame["horizon"] = frame["suite"].map(caps)
    return frame, features, {"dropped_tasks": dropped, "caps": caps}


def leave_one_suite_out(frame: pd.DataFrame, features: list[str], clock: list[str]) -> list[dict]:
    records = []
    for held in sorted(frame["suite"].unique()):
        train = frame[frame["suite"] != held]
        test = frame[frame["suite"] == held]
        row = {"held_out_suite": held, "train_rows": len(train), "test_rows": len(test)}
        for name, columns in (("clock", clock), ("clock_routing", clock + features)):
            matrix = train[columns].to_numpy(np.float64)
            centre, scale = matrix.mean(axis=0), matrix.std(axis=0)
            scale[scale < 1e-12] = 1.0
            model = LogisticRegression(max_iter=2000, C=1.0).fit(
                (matrix - centre) / scale, train["phase"].to_numpy(int)
            )
            probabilities = np.zeros((len(test), len(PHASES)))
            predicted = model.predict_proba((test[columns].to_numpy(np.float64) - centre) / scale)
            for position, label in enumerate(model.classes_):
                probabilities[:, label] = predicted[:, position]
            probabilities = np.clip(probabilities, 1e-9, None)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            row[f"log_loss_{name}"] = float(
                log_loss(test["phase"].to_numpy(int), probabilities, labels=[0, 1, 2])
            )
            row[f"accuracy_{name}"] = float(
                (probabilities.argmax(axis=1) == test["phase"].to_numpy(int)).mean()
            )
        row["log_loss_gain"] = row["log_loss_clock"] - row["log_loss_clock_routing"]
        records.append(row)
    return records


def deadline_table(frame: pd.DataFrame) -> pd.DataFrame:
    """vl(p) per suite: the chunk by which rollouts have left milestone p."""
    records = []
    for (suite, phase), group in frame.groupby(["suite", "phase"]):
        horizon = group["horizon"].iloc[0]
        absolute = float(group["chunk"].quantile(VL_QUANTILE))
        records.append(
            {
                "suite": suite,
                "phase": PHASES[int(phase)],
                "rows": int(len(group)),
                "horizon": int(horizon),
                "vl_absolute": absolute,
                "vl_fraction": absolute / horizon,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame, features, meta = build()
    clock = ["chunk", "chunk_sq", "chunk_log"]

    nodes = pd.DataFrame(leave_one_suite_out(frame, features, clock))
    nodes.to_csv(args.output / "node_transfer.csv", index=False)

    edges = deadline_table(frame)
    edges.to_csv(args.output / "edge_transfer.csv", index=False)
    spread = edges.pivot(index="phase", columns="suite", values="vl_fraction")
    dispersion = {
        str(phase): {
            "min": float(row.min()),
            "max": float(row.max()),
            "ratio_max_over_min": float(row.max() / row.min()) if row.min() > 0 else float("nan"),
        }
        for phase, row in spread.iterrows()
    }

    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "tasks": int(frame["task"].nunique()),
                "dropped_non_prehensile_tasks": meta["dropped_tasks"],
                "horizon_caps": meta["caps"],
                "node_transfer": nodes.to_dict("records"),
                "edge_dispersion_fraction_of_horizon": dispersion,
            },
            indent=2,
        )
    )

    pd.set_option("display.width", 220)
    print(
        f"rows {len(frame):,}  tasks {frame['task'].nunique()}"
        f"  非抓取任务被排除 {meta['dropped_tasks']}  horizon {meta['caps']}\n"
    )
    print("节点迁移：留出整个 suite，训练于其余三个")
    print(
        nodes[
            [
                "held_out_suite",
                "test_rows",
                "log_loss_clock",
                "log_loss_clock_routing",
                "log_loss_gain",
                "accuracy_clock",
                "accuracy_clock_routing",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\n边迁移：vl(p) 占 horizon 的比例，跨 suite 是否一致")
    print(spread.to_string(float_format=lambda value: f"{value:.3f}"))
    print()
    for phase, values in dispersion.items():
        print(f"  {phase:<10} min {values['min']:.3f}  max {values['max']:.3f}"
              f"  max/min {values['ratio_max_over_min']:.2f}")


if __name__ == "__main__":
    main()
