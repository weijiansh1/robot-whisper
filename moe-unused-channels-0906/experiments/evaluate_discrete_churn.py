#!/usr/bin/env python3
"""Does the discrete top-4 selection carry anything the soft distribution does not?

Three questions, in order.

1. Descriptive. How much routing is the HB gate actually doing, how close is the
   selection boundary to a tie, and how does the observed chunk-to-chunk set
   churn compare to the churn of two independent uniform 4-subsets of 32?
   Without this the churn numbers are uninterpretable.

2. Reconciliation with the two HUB bundles that already use this channel.
   `VLA_MUI_HUB/moe-physical-failure-dynamics` reports, against same-run
   same-initial-state matched success controls, a late-back-layer percentile of
   0.086 for the discrete cross-chunk churn versus 0.220 for its soft analogue.
   Our cohort has the same 50 initial states x 8 flow-noise seeds design, so the
   matched control is available natively; the failure definition differs (risk =
   did not finish before the suite horizon cap) and that difference is reported.

3. Detection. The frozen protocol, unchanged, over the discrete quantities, the
   soft analogue, and two negative controls. Selection on development, one
   external replay, false-alarm dependence against the 24 published heads, and
   everything compared against the causal length-only baseline, which is exactly
   what the survival prior measures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import protocol as P
from protocol import dev


DEFAULT_OUTPUT = P.RESULTS
CHANNELS = P.RESULTS / "channels"
DISCRETE = (
    "query_hard_churn",
    "query_hard_churn_d9",
    "query_top1_churn",
    "denoise_hard_churn",
    "set_dwell",
    "tie_margin",
    "selected_mass",
    "hb_entropy_action",
)
SOFT = ("mobility", "mobility_allstep")
CONTROLS = ("null_iid", "null_episode_constant")
LATE_PHASE = 0.70  # last 30% of a trajectory, matching the HUB dynamics bundle
TOP_K, N_EXPERTS = 4, 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=5000)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# quantity assembly


def random_subset_churn() -> dict[str, float]:
    """Exact expected Jaccard distance between two independent uniform 4-subsets."""
    from math import comb

    total = comb(N_EXPERTS, TOP_K)
    mean = 0.0
    for shared in range(TOP_K + 1):
        weight = (
            comb(TOP_K, shared) * comb(N_EXPERTS - TOP_K, TOP_K - shared) / total
        )
        mean += weight * (1.0 - shared / max(2 * TOP_K - shared, 1))
    return {"expected_churn_uniform_random_top4": mean}


def load_quantities(cohort: str) -> tuple[np.ndarray, list[str]]:
    index = P.load_npz(CHANNELS / f"{cohort}_index.npz")
    values = np.load(CHANNELS / f"{cohort}_quantities.npy", mmap_mode="r")
    return values, index["quantity_names"].astype(str).tolist()


def mobility_allstep(cohort: str) -> np.ndarray:
    """Soft chunk-to-chunk change averaged over all ten denoising steps.

    The frozen `mobility` cache is denoising step 9 only; the HUB bundle's soft
    statistic averages every step, so this is the like-for-like comparand.
    """
    path = P.STEP_PROFILES / f"{cohort}_mobility.npy"
    block = np.load(path, mmap_mode="r")
    return np.nanmean(np.asarray(block, dtype=np.float32), axis=3)


def controls(frame: dict[str, Any], name: str, rng: np.random.Generator) -> np.ndarray:
    shape = (*frame["valid"].shape, 8)
    if name == "null_iid":
        values = rng.random(shape, dtype=np.float32)
    elif name == "null_episode_constant":
        values = np.repeat(
            rng.random((shape[0], 1, shape[2]), dtype=np.float32), shape[1], axis=1
        )
    else:
        raise KeyError(name)
    return np.where(frame["valid"][:, :, None], values, np.nan)


def assemble(frames: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for cohort, frame in frames.items():
        raw, names = load_quantities(cohort)
        rng = np.random.default_rng(P.SEED + abs(hash(cohort)) % 1000)
        block: dict[str, np.ndarray] = {}
        for quantity in DISCRETE:
            block[quantity] = np.asarray(raw[:, :, :, names.index(quantity)], np.float32)
        block["mobility"] = P.quantity_values(frame, "mobility")
        block["mobility_allstep"] = mobility_allstep(cohort)
        for control in CONTROLS:
            block[control] = controls(frame, control, rng)
        output[cohort] = block
    return output


# --------------------------------------------------------------------------- #
# 1. descriptive


def describe(frames, values) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, frame in frames.items():
        valid = frame["valid"]
        for quantity in DISCRETE + SOFT:
            block = values[cohort][quantity]
            for group, layers in (
                ("front_L2_L5", slice(0, 4)),
                ("back_L12_L15", slice(4, 8)),
            ):
                sample = block[:, :, layers][valid]
                sample = sample[np.isfinite(sample)]
                rows.append(
                    {
                        "cohort": cohort,
                        "quantity": quantity,
                        "layer_group": group,
                        "n": int(sample.size),
                        "mean": float(sample.mean()),
                        "sd": float(sample.std()),
                        "q05": float(np.quantile(sample, 0.05)),
                        "median": float(np.median(sample)),
                        "q95": float(np.quantile(sample, 0.95)),
                    }
                )
    return pd.DataFrame(rows)


def entropy_identity(frames, values) -> dict[str, Any]:
    """hb_entropy averaged over action tokens vs the frozen flow-semantics token_entropy."""
    output: dict[str, Any] = {}
    for cohort, frame in frames.items():
        index = P.load_npz(P.STEP_PROFILES / f"{cohort}_index.npz")
        names = index["metric_names"].astype(str).tolist()
        cached = np.load(P.STEP_PROFILES / f"{cohort}_metrics.npy", mmap_mode="r")
        reference = np.asarray(
            cached[:, :, :, :, names.index("token_entropy")], np.float32
        ).mean(axis=3)
        observed = values[cohort]["hb_entropy_action"]
        mask = frame["valid"][:, :, None] & np.isfinite(reference) & np.isfinite(observed)
        gap = np.abs(observed[mask] - reference[mask])
        output[cohort] = {
            "cells": int(mask.sum()),
            "max_abs_delta": float(gap.max()),
            "mean_abs_delta": float(gap.mean()),
            "max_relative_delta": float((gap / np.abs(reference[mask])).max()),
        }
    return output


# --------------------------------------------------------------------------- #
# 2. matched-control reconciliation


def late_window_mean(block: np.ndarray, length: np.ndarray, layers: slice) -> np.ndarray:
    """Mean over the last 30% of each episode's chunks, averaged over a layer group."""
    reduced = np.nanmean(block[:, :, layers], axis=2)
    output = np.full(len(reduced), np.nan, dtype=np.float64)
    for row in range(len(reduced)):
        stop = int(length[row])
        start = int(np.ceil(LATE_PHASE * stop))
        window = reduced[row, start:stop]
        window = window[np.isfinite(window)]
        if len(window):
            output[row] = window.mean()
    return output


def absolute_window_mean(
    block: np.ndarray, chunk_lo: int, chunk_hi: int, layers: slice
) -> np.ndarray:
    reduced = np.nanmean(block[:, chunk_lo:chunk_hi, layers], axis=2)
    with np.errstate(invalid="ignore"):
        return np.where(
            np.isfinite(reduced).any(axis=1), np.nanmean(reduced, axis=1), np.nan
        )


def matched_percentiles(
    frame: dict[str, Any], score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mid-rank percentile of each risk episode among its matched success controls.

    A control shares the task, the initial state and therefore the run; it
    differs only in the flow-noise seed. This is the same matching the HUB
    dynamics bundle uses and is strictly finer than task stratification.
    """
    risk = frame["risk"]
    key = np.asarray(
        [f"{t}|{i}" for t, i in zip(frame["task"], frame["init_state_id"], strict=True)]
    )
    percentile = np.full(len(score), np.nan)
    controls_n = np.zeros(len(score), dtype=int)
    for group in np.unique(key[risk]):
        take = key == group
        reference = score[take & ~risk]
        reference = reference[np.isfinite(reference)]
        if len(reference) < 2:
            continue
        for row in np.flatnonzero(take & risk):
            if not np.isfinite(score[row]):
                continue
            below = float((reference < score[row]).sum())
            equal = float((reference == score[row]).sum())
            percentile[row] = (below + 0.5 * equal) / len(reference)
            controls_n[row] = len(reference)
    return percentile, controls_n, key


def reconcile(frames, values, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    definitions = {
        "late_back_query_hard_churn": ("query_hard_churn", slice(4, 8), "phase"),
        "late_back_query_soft_change": ("mobility_allstep", slice(4, 8), "phase"),
        "late_front_query_hard_churn": ("query_hard_churn", slice(0, 4), "phase"),
        "late_front_query_soft_change": ("mobility_allstep", slice(0, 4), "phase"),
        "late_back_query_hard_churn_d9": ("query_hard_churn_d9", slice(4, 8), "phase"),
        "late_back_query_soft_change_d9": ("mobility", slice(4, 8), "phase"),
        "late_back_denoise_hard_churn": ("denoise_hard_churn", slice(4, 8), "phase"),
        "late_back_tie_margin": ("tie_margin", slice(4, 8), "phase"),
        "late_back_selected_mass": ("selected_mass", slice(4, 8), "phase"),
        "late_back_hb_entropy": ("hb_entropy_action", slice(4, 8), "phase"),
        "abs_back_query_hard_churn": ("query_hard_churn", slice(4, 8), "absolute"),
        "abs_back_query_soft_change": ("mobility_allstep", slice(4, 8), "absolute"),
    }
    rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        frame = frames[cohort]
        # The absolute window is the last five chunks before the smallest suite
        # cap, so every episode in the comparison is still running there.
        for name, (quantity, layers, window) in definitions.items():
            block = values[cohort][quantity]
            if window == "phase":
                score = late_window_mean(block, frame["length"], layers)
            else:
                score = absolute_window_mean(block, 12, 17, layers)
            percentile, controls_n, key = matched_percentiles(frame, score)
            usable = np.isfinite(percentile)
            if usable.sum() < 20:
                continue
            task = frame["task"][usable]
            per_task = pd.Series(percentile[usable]).groupby(task).mean()
            tasks = per_task.index.to_numpy()
            draw = rng.integers(0, len(tasks), size=(draws, len(tasks)))
            bootstrap = per_task.to_numpy()[draw].mean(axis=1)
            rows.append(
                {
                    "cohort": cohort,
                    "metric": name,
                    "quantity": quantity,
                    "window": window,
                    "failures_with_controls": int(usable.sum()),
                    "tasks": len(tasks),
                    "median_controls_per_failure": float(np.median(controls_n[usable])),
                    "episode_weighted_mean_percentile": float(percentile[usable].mean()),
                    "task_macro_mean_percentile": float(per_task.mean()),
                    "task_cluster_bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
                    "task_cluster_bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
                    "failure_raw_mean": float(np.nanmean(score[usable])),
                    "matched_success_raw_mean": float(
                        np.nanmean(
                            score[
                                ~frames[cohort]["risk"]
                                & np.isin(key, np.unique(key[usable]))
                            ]
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. detection


def length_only_baseline(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """The causal length-only detector: alarm iff still running at chunk q.

    This is the baseline every MoE head must beat. Its precision at chunk q is
    the survival prior by definition, so its lift is exactly 1.
    """
    rows: list[dict[str, Any]] = []
    crossing = {"libero_goal": 18, "libero_long": 26, "libero_object": 17, "libero_spatial": 13}
    for label, chunk_of in (
        ("length_only|prior025_crossing", crossing),
        ("length_only|chunk12", {suite: 12 for suite in crossing}),
        ("length_only|chunk16", {suite: 16 for suite in crossing}),
        ("length_only|at_cap", {s: P.CAPS[s] - 1 for s in crossing}),
    ):
        first = np.full(len(frame["risk"]), -1, dtype=np.int16)
        for suite, chunk in chunk_of.items():
            take = (frame["suite"] == suite) & (frame["length"] > chunk)
            first[take] = chunk
        rows.append(
            {
                "quantity": "length_only",
                "mode": "n/a",
                "representation": label,
                "direction": "n/a",
                "quantile": float("nan"),
                **P.score_candidate(
                    first, frame["risk"], P.prior_of(first, frame["suite"], frame["priors"])
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(P.SEED)

    frames = P.cohort_frames()
    anchors = P.assert_anchors(frames)
    values = assemble(frames)
    print("assembled quantities", flush=True)

    # ---- 1. descriptive ----------------------------------------------------
    description = describe(frames, values)
    description.to_csv(args.output / "discrete_descriptive.csv", index=False)
    identity = entropy_identity(frames, values)
    print("entropy identity:", json.dumps(identity, indent=1), flush=True)

    # ---- 2. reconciliation --------------------------------------------------
    reconciliation = reconcile(frames, values, args.bootstrap, rng)
    reconciliation.to_csv(args.output / "matched_control_reconciliation.csv", index=False)
    print("\n=== matched-control percentiles (risk vs same-task same-init-state successes) ===")
    print(
        reconciliation[
            [
                "cohort", "metric", "window", "failures_with_controls",
                "episode_weighted_mean_percentile", "task_macro_mean_percentile",
                "task_cluster_bootstrap_ci_low", "task_cluster_bootstrap_ci_high",
                "failure_raw_mean", "matched_success_raw_mean",
            ]
        ].to_string(index=False, float_format="%.4f"),
        flush=True,
    )

    # ---- 3. detection -------------------------------------------------------
    quantities = list(DISCRETE) + list(SOFT) + list(CONTROLS)
    development_rows: list[pd.DataFrame] = []
    external_rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    external = frames["external_8b"]
    for quantity in quantities:
        candidates = P.sweep_development(
            frames, quantity, {c: values[c][quantity] for c in frames}
        )
        development_rows.append(candidates)
        for mode in P.MODES:
            best = P.select_head(candidates, mode)
            if best is None:
                external_rows.append(
                    {"quantity": quantity, "mode": mode, "feasible": False}
                )
                continue
            reference = [
                (
                    frames[cohort],
                    {
                        name: block
                        for name, (block, _) in dev.representations(
                            P.layer_cache(frames[cohort], values[cohort][quantity])
                        ).items()
                    }[best["representation"]],
                )
                for cohort in ("development_main", "development_extra")
            ]
            external_values = {
                name: block
                for name, (block, _) in dev.representations(
                    P.layer_cache(external, values["external_8b"][quantity])
                ).items()
            }[best["representation"]]
            first = P.external_alarm(
                external,
                external_values,
                reference,
                best["direction"],
                float(best["quantile"]),
                mode,
            )
            alarms[f"{quantity}|{mode}"] = first
            external_rows.append(
                {
                    "quantity": quantity,
                    "mode": mode,
                    "feasible": True,
                    "representation": best["representation"],
                    "direction": best["direction"],
                    "quantile": float(best["quantile"]),
                    "dev_tp": int(best["tp"]),
                    "dev_fp": int(best["fp"]),
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    **P.score_candidate(
                        first,
                        external["risk"],
                        P.prior_of(first, external["suite"], external["priors"]),
                    ),
                }
            )
        print(f"  swept {quantity}", flush=True)

    development = pd.concat(development_rows, ignore_index=True)
    development.to_csv(args.output / "discrete_development_candidates.csv", index=False)
    baseline = pd.DataFrame(
        length_only_baseline(external) + length_only_baseline(frames["development_main"])
    )
    baseline["cohort"] = ["external_8b"] * 4 + ["development_main"] * 4
    baseline.to_csv(args.output / "length_only_baseline.csv", index=False)
    detectors = pd.DataFrame(external_rows)
    detectors.to_csv(args.output / "discrete_external_detectors.csv", index=False)

    # ---- false-alarm dependence against the 24 published heads -------------
    published = P.load_npz(P.FRAME_ALARMS)
    published = {k: v for k, v in published.items() if k != "schema"}
    timely = ~external["risk"]
    dependence = pd.DataFrame(
        index=sorted(alarms), columns=sorted(published) + sorted(alarms), dtype=float
    )
    raw_overlap = dependence.copy()
    for left in sorted(alarms):
        for right in list(published) + list(alarms):
            other = published[right] if right in published else alarms[right]
            dependence.loc[left, right] = P.dependence_ratio(alarms[left], other, timely)
            raw_overlap.loc[left, right] = P.overlap_share(alarms[left], other, timely)
    dependence.to_csv(args.output / "discrete_false_alarm_dependence.csv")
    raw_overlap.to_csv(args.output / "discrete_false_alarm_raw_overlap.csv")

    # ---- stratified survival-conditioned AUC --------------------------------
    auc_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        for quantity in quantities:
            auc_rows.extend(
                P.survival_auc_rows(
                    frames[cohort], quantity, values[cohort][quantity], P.REPRESENTATIONS[:8]
                )
            )
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(args.output / "discrete_survival_auc.csv", index=False)

    def summarise_auc(block: pd.DataFrame) -> dict[str, Any]:
        return {
            "cells": len(block),
            "median_abs_dev_pooled": float((block["auc_pooled"] - 0.5).abs().median()),
            "median_abs_dev_within_suite": float(
                (block["auc_within_suite"] - 0.5).abs().median()
            ),
            "median_abs_dev_within_task": float(
                (block["auc_within_task"] - 0.5).abs().median()
            ),
            "max_abs_dev_within_task": float(
                (block["auc_within_task"] - 0.5).abs().max()
            ),
            "best_within_task_cell": block.loc[
                (block["auc_within_task"] - 0.5).abs().idxmax(),
                ["layer", "chunk", "auc_pooled", "auc_within_suite", "auc_within_task"],
            ].to_dict(),
        }

    summary = {
        "schema": "himoe.unused_channels.discrete.v1",
        "anchors_asserted": anchors,
        "random_top4_reference": random_subset_churn(),
        "hb_entropy_equals_flow_semantics_token_entropy": identity,
        "external_outcomes_read_once": True,
        "quantities": quantities,
        "negative_controls": list(CONTROLS),
        "survival_auc_by_quantity": {
            quantity: {
                cohort: summarise_auc(
                    auc[(auc["quantity"] == quantity) & (auc["cohort"] == cohort)]
                )
                for cohort in ("development_main", "external_8b")
            }
            for quantity in quantities
        },
    }
    P.write_json(args.output / "discrete_summary.json", summary)

    print("\n=== external replay, one head per quantity per mode ===")
    show = detectors[detectors["feasible"].fillna(False)]
    print(
        show[
            [
                "quantity", "mode", "representation", "direction", "quantile",
                "dev_tp", "dev_fp", "tp", "fp", "precision", "risk_recall",
                "mean_alarm_prior", "lift", "low_prior_tp", "low_prior_fp",
            ]
        ].to_string(index=False, float_format="%.4f"),
        flush=True,
    )
    print("\n=== causal length-only baseline (external) ===")
    print(
        baseline[baseline["cohort"] == "external_8b"][
            ["representation", "tp", "fp", "precision", "risk_recall",
             "mean_alarm_prior", "lift", "low_prior_tp", "low_prior_fp"]
        ].to_string(index=False, float_format="%.4f"),
        flush=True,
    )


if __name__ == "__main__":
    main()
