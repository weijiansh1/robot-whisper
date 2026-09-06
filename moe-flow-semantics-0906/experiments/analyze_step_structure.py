#!/usr/bin/env python3
"""Do the ten denoising steps have distinct functional roles?

Descriptive stage.  Nothing here reads an outcome label; every number is a
function of hb_router_probs alone.  Five questions are answered:

1. step profile      how each per-step quantity evolves over steps 0..9, per HB
                     layer, per cohort.
2. front/back x step whether the established front/back layer contrast keeps its
                     sign along the flow, or reverses.
3. replication       for every quantity, the fraction of tasks whose late-minus-
                     early contrast agrees in sign with the pooled direction.
                     Reported for the 40 development tasks and the 39 external
                     tasks separately, as the layer work did.
4. block structure   whether the step axis splits into two blocks the way the
                     layer axis does.  The 10x10 within-task correlation matrix
                     of each quantity across steps is compared against the 8x8
                     layer correlation matrix of the same quantity, using the
                     same split score, so "is there a step analogue of the
                     front/back split" gets an apples-to-apples answer.
5. noise vs signal   at query 0 the observation is identical inside an initial
                     state group, so the route spread there is caused by the
                     flow noise alone; two episodes sharing a seed but not an
                     initial state differ only in the observation.  The ratio of
                     the two, per step, says what each step listens to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROFILES = BUNDLE / "results/step_profiles"
DEFAULT_OUTPUT = BUNDLE / "results/step_structure"

COHORTS = ("development_main", "development_extra", "external_8b")
DEVELOPMENT = ("development_main", "development_extra")
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
FRONT = slice(0, 4)
BACK = slice(4, 8)
EARLY = slice(0, 3)
LATE = slice(7, 10)
# flow_speed is a between-step difference, so its step 0 is undefined and its
# early window is shifted by one.  That makes the early / late windows here the
# same three transitions the cached flow_settling_log_ratio already uses.
EARLY_BY_QUANTITY = {"flow_speed": slice(1, 4)}
PAIR_NAMES = ("same_state_diff_noise", "diff_state_same_noise", "diff_state_diff_noise")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", type=Path, default=PROFILES)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def cohort_quantities(profiles: Path, cohort: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the index and every per-step quantity as [valid_query, 8, 10]."""
    index = load_npz(profiles / f"{cohort}_index.npz")
    valid = index["valid"].astype(bool)
    metrics = np.load(profiles / f"{cohort}_metrics.npy", mmap_mode="r")
    mobility = np.load(profiles / f"{cohort}_mobility.npy", mmap_mode="r")
    state = np.load(profiles / f"{cohort}_state_mobility.npy", mmap_mode="r")
    names = index["metric_names"].astype(str).tolist()

    rows, columns = np.nonzero(valid)
    quantities: dict[str, np.ndarray] = {}
    block = np.asarray(metrics[rows, columns])
    for position, name in enumerate(names):
        quantities[name] = np.ascontiguousarray(block[..., position])
    del block
    quantities["mobility"] = np.asarray(mobility[rows, columns])
    quantities["state_mobility"] = np.asarray(state[rows, columns])
    index["query_task"] = index["task_index"].astype(int)[rows]
    index["query_row"] = rows
    index["query_position"] = columns
    return index, quantities


def fisher_mean(matrices: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(matrices)
    z = np.arctanh(np.clip(stack, -0.999999, 0.999999))
    return np.tanh(np.nanmean(z, axis=0))


def split_score(correlation: np.ndarray) -> tuple[int, float, list[float]]:
    """Best two-block split of an axis, scored as within minus between."""
    size = correlation.shape[0]
    offdiag = ~np.eye(size, dtype=bool)
    scores = []
    for cut in range(1, size):
        block = np.zeros(size, dtype=bool)
        block[:cut] = True
        same = (block[:, None] == block[None, :]) & offdiag
        other = (block[:, None] != block[None, :])
        scores.append(float(correlation[same].mean() - correlation[other].mean()))
    best = int(np.argmax(scores))
    return best + 1, scores[best], scores


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    profile_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_profile_rows: list[dict[str, Any]] = []
    correlation_store: dict[str, np.ndarray] = {}
    split_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    quantity_names: list[str] = []

    for cohort in COHORTS:
        index, quantities = cohort_quantities(args.profiles, cohort)
        task_names = index["task_names"].astype(str)
        query_task = index["query_task"]
        quantity_names = list(quantities)

        for name, values in quantities.items():
            finite = np.isfinite(values)
            with np.errstate(invalid="ignore"):
                pooled_mean = np.nanmean(np.where(finite, values, np.nan), axis=0)
                pooled_std = np.nanstd(np.where(finite, values, np.nan), axis=0)
            for layer in range(8):
                for step in range(10):
                    profile_rows.append(
                        {
                            "cohort": cohort,
                            "quantity": name,
                            "layer": LAYER_NAMES[layer],
                            "layer_group": "front" if layer < 4 else "back",
                            "step": step,
                            "mean": float(pooled_mean[layer, step]),
                            "std": float(pooled_std[layer, step]),
                            "n": int(finite[:, layer, step].sum()),
                        }
                    )

            # Per-task contrasts, and per-task step correlations.
            for position, task in enumerate(task_names):
                take = query_task == position
                block = values[take]
                with np.errstate(invalid="ignore"):
                    per_step = np.nanmean(
                        np.where(np.isfinite(block), block, np.nan), axis=0
                    )
                early_window = EARLY_BY_QUANTITY.get(name, EARLY)
                early = per_step[:, early_window].mean(axis=1)
                late = per_step[:, LATE].mean(axis=1)
                front_back = per_step[BACK].mean(axis=0) - per_step[FRONT].mean(axis=0)
                for label, series in (
                    [(LAYER_NAMES[i], per_step[i]) for i in range(8)]
                    + [
                        ("front_mean", per_step[FRONT].mean(axis=0)),
                        ("back_mean", per_step[BACK].mean(axis=0)),
                        ("back_minus_front", front_back),
                    ]
                ):
                    for step in range(10):
                        task_profile_rows.append(
                            {
                                "cohort": cohort,
                                "quantity": name,
                                "task": task,
                                "layer": label,
                                "step": step,
                                "mean": float(series[step]),
                            }
                        )
                for layer in range(8):
                    task_rows.append(
                        {
                            "cohort": cohort,
                            "quantity": name,
                            "task": task,
                            "layer": LAYER_NAMES[layer],
                            "early_mean": float(early[layer]),
                            "late_mean": float(late[layer]),
                            "late_minus_early": float(late[layer] - early[layer]),
                            "step_argmax": int(np.nanargmax(per_step[layer]))
                            if np.isfinite(per_step[layer]).any()
                            else -1,
                        }
                    )
                task_rows.append(
                    {
                        "cohort": cohort,
                        "quantity": name,
                        "task": task,
                        "layer": "back_minus_front",
                        "early_mean": float(front_back[early_window].mean()),
                        "late_mean": float(front_back[LATE].mean()),
                        "late_minus_early": float(
                            front_back[LATE].mean() - front_back[early_window].mean()
                        ),
                        "step_argmax": -1,
                    }
                )

            # Within-task correlation across the step axis, and across the layer
            # axis, so the two axes are scored on the same footing.
            step_matrices: list[np.ndarray] = []
            layer_matrices: list[np.ndarray] = []
            step_offset = 1 if name == "flow_speed" else 0
            for position in range(len(task_names)):
                block = values[query_task == position][:, :, step_offset:]
                usable = np.isfinite(block).all(axis=(1, 2))
                block = block[usable]
                if len(block) < 64:
                    continue
                step_matrices.append(
                    np.corrcoef(block.mean(axis=1), rowvar=False)
                )
                layer_matrices.append(
                    np.corrcoef(block.mean(axis=2), rowvar=False)
                )
            if not step_matrices:
                continue
            step_correlation = fisher_mean(step_matrices)
            layer_correlation = fisher_mean(layer_matrices)
            correlation_store[f"{cohort}|{name}|step"] = step_correlation.astype(np.float32)
            correlation_store[f"{cohort}|{name}|layer"] = layer_correlation.astype(np.float32)
            for axis, matrix in (("step", step_correlation), ("layer", layer_correlation)):
                cut, best, scores = split_score(matrix)
                canonical = 4 if axis == "layer" else None
                split_rows.append(
                    {
                        "cohort": cohort,
                        "quantity": name,
                        "axis": axis,
                        "axis_size": int(len(matrix)),
                        "step_offset": step_offset if axis == "step" else 0,
                        "best_cut": cut,
                        "best_split_score": best,
                        "canonical_cut": canonical,
                        "canonical_split_score": (
                            scores[canonical - 1] if canonical else float("nan")
                        ),
                        "extreme_correlation": float(matrix[0, -1]),
                        "mean_offdiag_correlation": float(
                            matrix[~np.eye(len(matrix), dtype=bool)].mean()
                        ),
                        "all_scores": json.dumps([round(s, 5) for s in scores]),
                    }
                )

            # Per-layer step correlation matrices are also worth keeping.
            width = 10 - step_offset
            per_layer = []
            for layer in range(8):
                matrices = []
                for position in range(len(task_names)):
                    block = values[query_task == position][:, layer, step_offset:]
                    block = block[np.isfinite(block).all(axis=1)]
                    if len(block) < 64:
                        continue
                    matrices.append(np.corrcoef(block, rowvar=False))
                per_layer.append(
                    fisher_mean(matrices) if matrices else np.full((width, width), np.nan)
                )
            correlation_store[f"{cohort}|{name}|step_per_layer"] = np.stack(
                per_layer
            ).astype(np.float32)
            print(f"  [{cohort}] {name}", flush=True)

        pairs = load_npz(args.profiles / f"{cohort}_query0_pairs.npz")
        for token_family in ("action", "state"):
            block = pairs[token_family]  # (task, 3, 8, 10)
            for layer in range(8):
                for step in range(10):
                    values = {
                        name: block[:, position, layer, step]
                        for position, name in enumerate(PAIR_NAMES)
                    }
                    pair_rows.append(
                        {
                            "cohort": cohort,
                            "token_family": token_family,
                            "layer": LAYER_NAMES[layer],
                            "step": step,
                            **{
                                f"{name}_mean": float(values[name].mean())
                                for name in PAIR_NAMES
                            },
                            "observation_over_noise": float(
                                np.mean(
                                    values["diff_state_same_noise"]
                                    / np.maximum(values["same_state_diff_noise"], 1e-12)
                                )
                            ),
                            "tasks": int(block.shape[0]),
                        }
                    )
        del quantities

    profiles_frame = pd.DataFrame(profile_rows)
    tasks_frame = pd.DataFrame(task_rows)
    pairs_frame = pd.DataFrame(pair_rows)
    splits_frame = pd.DataFrame(split_rows)
    profiles_frame.to_csv(args.output / "step_profiles.csv", index=False)
    tasks_frame.to_csv(args.output / "task_step_contrasts.csv", index=False)
    task_profiles = pd.DataFrame(task_profile_rows)
    task_profiles.to_csv(args.output / "task_step_profiles.csv", index=False)

    # Per-step task agreement: at each step, in how many tasks does the back
    # group exceed the front group?  This is the step-resolved version of the
    # front/back replication count the layer work reports.
    agreement_rows: list[dict[str, Any]] = []
    contrast = task_profiles[task_profiles["layer"] == "back_minus_front"]
    for (cohort, quantity, step), block in contrast.groupby(
        ["cohort", "quantity", "step"]
    ):
        values = block["mean"].to_numpy()
        agreement_rows.append(
            {
                "cohort": cohort,
                "quantity": quantity,
                "step": int(step),
                "tasks": int(len(values)),
                "tasks_back_above_front": int((values > 0).sum()),
                "median_back_minus_front": float(np.median(values)),
            }
        )
    pd.DataFrame(agreement_rows).to_csv(
        args.output / "front_back_per_step_agreement.csv", index=False
    )
    pairs_frame.to_csv(args.output / "query0_noise_observation.csv", index=False)
    splits_frame.to_csv(args.output / "axis_block_split.csv", index=False)
    np.savez_compressed(
        args.output / "step_correlations.npz",
        schema=np.asarray("himoe.flow_semantics.step_correlations.v1"),
        **correlation_store,
    )

    # Replication: does each task's late-minus-early contrast agree in sign with
    # the pooled development direction?  Development is the 40 tasks of the two
    # development cohorts; external is the 39 tasks of run right-50x8b.
    replication_rows: list[dict[str, Any]] = []
    development = tasks_frame[tasks_frame["cohort"].isin(DEVELOPMENT)]
    external = tasks_frame[tasks_frame["cohort"] == "external_8b"]
    for quantity in quantity_names:
        for layer in list(LAYER_NAMES) + ["back_minus_front"]:
            dev_block = development[
                (development["quantity"] == quantity) & (development["layer"] == layer)
            ]
            ext_block = external[
                (external["quantity"] == quantity) & (external["layer"] == layer)
            ]
            if dev_block.empty:
                continue
            contrast = dev_block["late_minus_early"].to_numpy()
            direction = 1.0 if np.median(contrast) > 0 else -1.0
            ext_contrast = ext_block["late_minus_early"].to_numpy()
            replication_rows.append(
                {
                    "quantity": quantity,
                    "layer": layer,
                    "direction": "late>early" if direction > 0 else "late<early",
                    "dev_tasks": len(contrast),
                    "dev_agree": int((np.sign(contrast) == direction).sum()),
                    "dev_exact_ties": int((contrast == 0.0).sum()),
                    "ext_exact_ties": int((ext_contrast == 0.0).sum()),
                    "dev_median_contrast": float(np.median(contrast)),
                    "dev_median_relative": float(
                        np.median(
                            dev_block["late_minus_early"].to_numpy()
                            / np.maximum(np.abs(dev_block["early_mean"].to_numpy()), 1e-12)
                        )
                    ),
                    "ext_tasks": len(ext_contrast),
                    "ext_agree": int((np.sign(ext_contrast) == direction).sum()),
                    "ext_median_contrast": float(np.median(ext_contrast)),
                }
            )
    replication = pd.DataFrame(replication_rows)
    replication.to_csv(args.output / "step_replication.csv", index=False)

    # Sign reversal of the front/back contrast along the flow, per task.
    reversal_rows: list[dict[str, Any]] = []
    for quantity in quantity_names:
        for cohort, block in tasks_frame[
            tasks_frame["layer"] == "back_minus_front"
        ].groupby("cohort"):
            block = block[block["quantity"] == quantity]
            if block.empty:
                continue
            early = block["early_mean"].to_numpy()
            late = block["late_mean"].to_numpy()
            reversal_rows.append(
                {
                    "quantity": quantity,
                    "cohort": cohort,
                    "tasks": len(block),
                    "early_back_above_front": int((early > 0).sum()),
                    "late_back_above_front": int((late > 0).sum()),
                    "sign_flips_neg_to_pos": int(((early < 0) & (late > 0)).sum()),
                    "sign_flips_pos_to_neg": int(((early > 0) & (late < 0)).sum()),
                    "median_early": float(np.median(early)),
                    "median_late": float(np.median(late)),
                }
            )
    reversal = pd.DataFrame(reversal_rows)
    reversal.to_csv(args.output / "front_back_step_reversal.csv", index=False)

    summary = {
        "schema": "himoe.flow_semantics.step_structure.v1",
        "outcomes_loaded": False,
        "quantities": quantity_names,
        "early_steps": [0, 1, 2],
        "late_steps": [7, 8, 9],
        "development_tasks": int(
            development[["cohort", "task"]].drop_duplicates().shape[0]
        ),
        "external_tasks": int(external[["cohort", "task"]].drop_duplicates().shape[0]),
        "split_score_definition": "mean within-block off-diagonal correlation minus mean between-block correlation, over the within-task Fisher-averaged correlation matrix",
        "query0_identification": {
            "same_state_diff_noise": "identical observation, different flow noise seed",
            "diff_state_same_noise": "different observation, identical flow noise seed",
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 220)
    print("\n=== mobility step profile, development_main ===")
    pivot = profiles_frame[
        (profiles_frame["cohort"] == "development_main")
        & (profiles_frame["quantity"] == "mobility")
    ].pivot(index="layer", columns="step", values="mean")
    print(pivot.reindex(LAYER_NAMES).to_string(float_format="%.4f"))
    print("\n=== front/back reversal along the flow ===")
    print(reversal.to_string(index=False, float_format="%.5f"))
    print("\n=== replication of late-minus-early direction ===")
    print(
        replication[replication["layer"].isin(["L2", "L12", "back_minus_front"])].to_string(
            index=False, float_format="%.5f"
        )
    )
    print("\n=== per-layer correlation between step 0 and step 9 mobility ===")
    for cohort in COHORTS:
        block = correlation_store.get(f"{cohort}|mobility|step_per_layer")
        if block is None:
            continue
        print(
            cohort,
            {LAYER_NAMES[i]: round(float(block[i, 0, -1]), 3) for i in range(8)},
        )
    print("\n=== axis block split, development_main ===")
    print(
        splits_frame[splits_frame["cohort"] == "development_main"][
            [
                "quantity",
                "axis",
                "best_cut",
                "best_split_score",
                "canonical_split_score",
                "extreme_correlation",
                "mean_offdiag_correlation",
            ]
        ].to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
