#!/usr/bin/env python3
"""Is `as_probs` / `as_expert_ids` a fifth reference frame?

`as_probs` is (query, 4, 3) and `as_expert_ids` is (query, 4): four AS routing
layers over three AS experts, versus the 8 x 10 x 11 x 32 HB structure that
every established quantity derives from. It is the only candidate in the log for
a routing system independent of `hb_router_probs`.

The extraction pass already established the fact this whole question turns on,
over every query row of both run_ids with no subsampling: `as_probs` is constant
within an episode to 0.0 absolute deviation, constant within a task, and takes
exactly four distinct values over the forty development tasks. This script
establishes what those four values index, then follows the prereg's decision
tree: because the within-episode variance is exactly zero, the frozen sweep is
run only to make the consequence concrete, not as evidence that `as_*` is a
detector family.

Structure of the answer:

  1. what the 4 and the 3 index, and what varies
  2. how much information the channel carries, and about what
  3. whether it relates to hb_* at the only granularity where it varies
  4. what the frozen sweep does to a per-episode constant, both modes
  5. false-alarm dependence against the 24 published heads
  6. stratified survival-conditioned AUC, which is 0.5 by construction within task
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
AS_SCALARS = (
    "as_entropy",
    "as_top1",
    "as_margin",
    "as_switch",
    "as_selected_prob",
    "as_layer_disagreement",
)
AS_REPRESENTATIONS = ("A0", "A1", "A2", "A3", "as_front_median", "as_back_median", "as_all_median")
N_AS_LAYERS, N_AS_EXPERTS = 4, 3
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# 1-2. what the channel is


def as_scalar_table(probs: np.ndarray, ids: np.ndarray) -> dict[str, np.ndarray]:
    """probs is (task, 4, 3), ids is (task, 4). Returns (task, 4) per scalar."""
    shannon = -(probs * np.log(np.maximum(probs, EPSILON))).sum(axis=-1)
    ordered = np.sort(probs, axis=-1)
    total_variation = np.zeros(len(probs), dtype=np.float64)
    pairs = 0
    for left in range(N_AS_LAYERS):
        for right in range(left + 1, N_AS_LAYERS):
            total_variation += 0.5 * np.abs(probs[:, left] - probs[:, right]).sum(axis=-1)
            pairs += 1
    return {
        "as_entropy": shannon,
        "as_top1": ordered[..., -1],
        "as_margin": ordered[..., -1] - ordered[..., -2],
        "as_switch": np.zeros_like(shannon),
        "as_selected_prob": np.take_along_axis(probs, ids[..., None], axis=-1)[..., 0],
        "as_layer_disagreement": np.repeat(
            (total_variation / pairs)[:, None], N_AS_LAYERS, axis=1
        ),
    }


def characterise(frames: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for cohort, frame in frames.items():
        block = P.load_npz(CHANNELS / f"{cohort}_as_router.npz")
        tasks = block["task_names"].astype(str)
        suites = np.asarray([t.split("/", 1)[0] for t in tasks])
        probs = block["as_probs_first"]
        ids = block["as_expert_ids_first"]
        key = [tuple(np.round(row.ravel(), 6)) for row in probs]
        distinct = sorted(set(key))
        code = np.asarray([distinct.index(k) for k in key])
        for position, value in enumerate(distinct):
            members = suites[code == position]
            rows.append(
                {
                    "cohort": cohort,
                    "as_code": position,
                    "tasks": int((code == position).sum()),
                    "suites": ",".join(sorted(set(members))),
                    "one_suite_only": len(set(members)) == 1,
                    **{
                        f"A{layer}_e{expert}": float(np.asarray(value).reshape(4, 3)[layer, expert])
                        for layer in range(N_AS_LAYERS)
                        for expert in range(N_AS_EXPERTS)
                    },
                }
            )
        # does the code determine the suite and vice versa?
        contingency = pd.crosstab(code, suites)
        detail[cohort] = {
            "tasks": len(tasks),
            "distinct_as_probs_values": len(distinct),
            "distinct_as_expert_ids_values": int(len(np.unique(ids, axis=0))),
            "as_expert_ids_value": np.unique(ids, axis=0).tolist(),
            "code_determines_suite": bool((contingency > 0).sum(axis=1).max() == 1),
            "suite_determines_code": bool((contingency > 0).sum(axis=0).max() == 1),
            "max_within_episode_deviation": float(block["as_within_episode_deviation"].max()),
            "max_within_task_deviation": float(block["as_row_deviation"].max()),
            "layer_is_exactly_uniform": [
                bool(np.allclose(probs[:, layer], 1.0 / N_AS_EXPERTS, atol=5e-4))
                for layer in range(N_AS_LAYERS)
            ],
            "layer_max_range_across_tasks": [
                float(probs[:, layer].max() - probs[:, layer].min())
                for layer in range(N_AS_LAYERS)
            ],
            "entropy_over_log3": [
                float(
                    (-(probs[:, layer] * np.log(np.maximum(probs[:, layer], EPSILON))).sum(-1)
                     / np.log(N_AS_EXPERTS)).mean()
                )
                for layer in range(N_AS_LAYERS)
            ],
        }
    return pd.DataFrame(rows), detail


def information_content(frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How many bits does `as_*` carry, and are they anything but the suite label?"""
    output: dict[str, Any] = {}
    for cohort in ("development_main", "external_8b"):
        frame = frames[cohort]
        block = P.load_npz(CHANNELS / f"{cohort}_as_router.npz")
        tasks = block["task_names"].astype(str)
        key = [tuple(np.round(row.ravel(), 6)) for row in block["as_probs_first"]]
        distinct = sorted(set(key))
        code_by_task = {t: distinct.index(k) for t, k in zip(tasks, key, strict=True)}
        code = np.asarray([code_by_task[t] for t in frame["task"]])
        suite = frame["suite"]

        def entropy(labels: np.ndarray) -> float:
            _, counts = np.unique(labels, return_counts=True)
            share = counts / counts.sum()
            return float(-(share * np.log2(share)).sum())

        joint = np.asarray([f"{a}|{b}" for a, b in zip(code, suite, strict=True)])
        h_code, h_suite, h_joint = entropy(code), entropy(suite), entropy(joint)
        output[cohort] = {
            "episodes": int(len(code)),
            "bits_in_as_probs": h_code,
            "bits_in_suite": h_suite,
            "mutual_information_bits": h_code + h_suite - h_joint,
            "conditional_entropy_of_as_given_suite_bits": h_joint - h_suite,
            "distinct_values_per_query": 1,
            "within_episode_bits": 0.0,
        }
    return output


def relation_to_hb(frames: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`as_*` varies only between suites, so a relation to hb_* is a 4-point claim."""
    frame = frames["external_8b"]
    block = P.load_npz(CHANNELS / "external_8b_as_router.npz")
    tasks = block["task_names"].astype(str)
    probs = block["as_probs_first"].reshape(len(tasks), -1)
    suites = np.asarray([t.split("/", 1)[0] for t in tasks])
    values, names = load_channel_quantities(frame)
    per_task = {}
    for quantity in ("query_hard_churn", "hb_entropy_action", "selected_mass"):
        column = np.nanmean(
            np.where(frame["valid"][:, :, None], values[:, :, :, names.index(quantity)], np.nan),
            axis=(1, 2),
        )
        per_task[quantity] = np.asarray(
            [np.nanmean(column[frame["task"] == task]) for task in tasks]
        )
    return {
        "note": (
            "as_probs takes one value per suite, so any relation to an hb_* quantity "
            "is estimated from four points and is not identifiable."
        ),
        "distinct_as_values": int(len(np.unique(probs.round(6), axis=0))),
        "suites": sorted(set(suites)),
        "per_suite_as_A0_expert2": {
            suite: float(np.unique(probs[suites == suite, 2].round(6))[0])
            for suite in sorted(set(suites))
        },
        "per_suite_hb_query_hard_churn_back_mean": {
            suite: float(np.nanmean(per_task["query_hard_churn"][suites == suite]))
            for suite in sorted(set(suites))
        },
        "within_suite_as_variance": 0.0,
    }


def load_channel_quantities(frame: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    index = P.load_npz(CHANNELS / f"{frame['cohort']}_index.npz")
    values = np.load(CHANNELS / f"{frame['cohort']}_quantities.npy", mmap_mode="r")
    return np.asarray(values), index["quantity_names"].astype(str).tolist()


# --------------------------------------------------------------------------- #
# 4. the frozen sweep on a per-episode constant


def as_series(frame: dict[str, Any], scalar: str) -> np.ndarray:
    """(episode, chunk, 4) for one as_* scalar, NaN outside the valid mask."""
    block = P.load_npz(CHANNELS / f"{frame['cohort']}_as_router.npz")
    tasks = block["task_names"].astype(str)
    table = as_scalar_table(block["as_probs_first"], block["as_expert_ids_first"])[scalar]
    lookup = {task: table[position] for position, task in enumerate(tasks)}
    per_episode = np.stack([lookup[task] for task in frame["task"]])
    series = np.repeat(per_episode[:, None, :], frame["valid"].shape[1], axis=1)
    return np.where(frame["valid"][:, :, None], series, np.nan).astype(np.float32)


def as_representations(series: np.ndarray) -> dict[str, np.ndarray]:
    output = {f"A{layer}": series[:, :, layer] for layer in range(N_AS_LAYERS)}
    output["as_front_median"] = np.median(series[:, :, :2], axis=2)
    output["as_back_median"] = np.median(series[:, :, 2:], axis=2)
    output["as_all_median"] = np.median(series, axis=2)
    return output


def sweep_as(frames: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    main = frames["development_main"]
    external = frames["external_8b"]
    development_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}
    for scalar in AS_SCALARS:
        reprs = {
            cohort: as_representations(as_series(frames[cohort], scalar)) for cohort in frames
        }
        for representation in AS_REPRESENTATIONS:
            for direction in ("low", "high"):
                dev_signal = P.oriented(reprs["development_main"][representation], direction)
                persistent = dev.persistent_score(dev_signal, P.CONFIRMATIONS)
                crossfit = dev.crossfit_thresholds(
                    dev.row_max(dev_signal), main["task_index"], main["init_state_id"]
                )
                pooled = np.concatenate(
                    [
                        dev.row_max(P.oriented(reprs[c][representation], direction))
                        for c in ("development_main", "development_extra")
                    ]
                )
                for position, quantile in enumerate(dev.QUANTILES):
                    for mode in P.MODES:
                        line = (
                            crossfit[:, position][:, None]
                            if mode == "per_task"
                            else dev.quantile_higher(pooled, quantile)
                        )
                        first = dev.first_query(
                            np.isfinite(persistent) & (persistent > line) & main["valid"]
                        )
                        development_rows.append(
                            {
                                "quantity": scalar,
                                "representation": representation,
                                "direction": direction,
                                "quantile": quantile,
                                "mode": mode,
                                "distinct_alarm_chunks": int(
                                    len(np.unique(first[first >= 0]))
                                ),
                                **P.score_candidate(
                                    first, main["risk"], P.prior_of(first, main["suite"], main["priors"])
                                ),
                            }
                        )
        candidates = pd.DataFrame(development_rows)
        candidates = candidates[candidates["quantity"] == scalar]
        for mode in P.MODES:
            best = P.select_head(candidates, mode)
            if best is None:
                external_rows.append({"quantity": scalar, "mode": mode, "feasible": False})
                continue
            first = P.external_alarm(
                external,
                reprs["external_8b"][best["representation"]],
                [
                    (frames[c], reprs[c][best["representation"]])
                    for c in ("development_main", "development_extra")
                ],
                best["direction"],
                float(best["quantile"]),
                mode,
            )
            alarms[f"{scalar}|{mode}"] = first
            external_rows.append(
                {
                    "quantity": scalar,
                    "mode": mode,
                    "feasible": True,
                    "representation": best["representation"],
                    "direction": best["direction"],
                    "quantile": float(best["quantile"]),
                    "dev_tp": int(best["tp"]),
                    "dev_fp": int(best["fp"]),
                    "distinct_alarm_chunks": int(len(np.unique(first[first >= 0]))),
                    "suites_firing": ",".join(
                        sorted(set(external["suite"][first >= 0]))
                    ),
                    **P.score_candidate(
                        first,
                        external["risk"],
                        P.prior_of(first, external["suite"], external["priors"]),
                    ),
                }
            )
        print(f"  swept {scalar}", flush=True)
    return pd.DataFrame(development_rows), pd.DataFrame(external_rows), alarms


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = P.cohort_frames()
    anchors = P.assert_anchors(frames)

    table, detail = characterise(frames)
    table.to_csv(args.output / "as_router_values.csv", index=False)
    print("\n=== as_probs distinct values ===")
    print(table[["cohort", "as_code", "tasks", "suites", "one_suite_only"]].to_string(index=False))
    print("\n=== structure ===")
    print(json.dumps(detail, indent=1, default=str))

    bits = information_content(frames)
    relation = relation_to_hb(frames)
    print("\n=== information content ===")
    print(json.dumps(bits, indent=1))

    development, external_table, alarms = sweep_as(frames)
    development.to_csv(args.output / "as_development_candidates.csv", index=False)
    external_table.to_csv(args.output / "as_external_detectors.csv", index=False)

    external = frames["external_8b"]
    timely = ~external["risk"]
    published = {k: v for k, v in P.load_npz(P.FRAME_ALARMS).items() if k != "schema"}
    dependence = pd.DataFrame(index=sorted(alarms), columns=sorted(published), dtype=float)
    overlap = dependence.copy()
    for left in sorted(alarms):
        for right in sorted(published):
            dependence.loc[left, right] = P.dependence_ratio(
                alarms[left], published[right], timely
            )
            overlap.loc[left, right] = P.overlap_share(alarms[left], published[right], timely)
    dependence.to_csv(args.output / "as_false_alarm_dependence.csv")
    overlap.to_csv(args.output / "as_false_alarm_raw_overlap.csv")

    auc_rows: list[dict[str, Any]] = []
    for cohort in ("development_main", "external_8b"):
        for scalar in AS_SCALARS:
            auc_rows.extend(
                P.survival_auc_rows(
                    frames[cohort],
                    scalar,
                    as_series(frames[cohort], scalar),
                    ("A0", "A1", "A2", "A3"),
                )
            )
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(args.output / "as_survival_auc.csv", index=False)

    summary = {
        "schema": "himoe.unused_channels.as_router.v1",
        "anchors_asserted": anchors,
        "verdict_inputs": {
            "within_episode_deviation_all_cohorts": {
                cohort: detail[cohort]["max_within_episode_deviation"] for cohort in detail
            },
            "within_task_deviation_all_cohorts": {
                cohort: detail[cohort]["max_within_task_deviation"] for cohort in detail
            },
            "distinct_values": {
                cohort: detail[cohort]["distinct_as_probs_values"] for cohort in detail
            },
            "as_expert_ids_globally_constant": all(
                detail[cohort]["distinct_as_expert_ids_values"] == 1 for cohort in detail
            ),
        },
        "structure": detail,
        "information": bits,
        "relation_to_hb": relation,
        "sweep": {
            "cells_per_scalar": int(len(development) / len(AS_SCALARS)),
            "development_cells_with_any_alarm": int((development["tp"] + development["fp"] > 0).sum()),
            "per_task_cells_with_any_alarm": int(
                ((development["mode"] == "per_task") & (development["tp"] + development["fp"] > 0)).sum()
            ),
            "max_distinct_alarm_chunks": int(development["distinct_alarm_chunks"].max()),
        },
        "survival_auc": {
            cohort: {
                "cells": int((auc["cohort"] == cohort).sum()),
                "max_abs_dev_within_task": float(
                    (auc[auc["cohort"] == cohort]["auc_within_task"] - 0.5).abs().max()
                ),
                "max_abs_dev_within_suite": float(
                    (auc[auc["cohort"] == cohort]["auc_within_suite"] - 0.5).abs().max()
                ),
                "max_abs_dev_pooled": float(
                    (auc[auc["cohort"] == cohort]["auc_pooled"] - 0.5).abs().max()
                ),
            }
            for cohort in ("development_main", "external_8b")
        },
    }
    P.write_json(args.output / "as_router_summary.json", summary)

    print("\n=== external replay of the as_* heads ===")
    show = external_table[external_table["feasible"].fillna(False)]
    if len(show):
        print(
            show[
                ["quantity", "mode", "representation", "direction", "quantile",
                 "dev_tp", "dev_fp", "tp", "fp", "precision", "risk_recall",
                 "mean_alarm_prior", "lift", "distinct_alarm_chunks", "suites_firing"]
            ].to_string(index=False, float_format="%.4f"),
            flush=True,
        )
    else:
        print("no as_* head passes the frozen selection rule in either mode", flush=True)
    print("\n=== survival AUC ===")
    print(json.dumps(summary["survival_auc"], indent=1))


if __name__ == "__main__":
    main()
