#!/usr/bin/env python3
"""Survey every MoE routing quantity as a detector, and group them by how they
fail rather than by what they measure.

The AND result showed that two detectors whose false alarms are nearly disjoint
combine into something far cleaner than either. The open question is whether
that disjointness comes from the reference frame -- what each quantity calls
"normal" -- and whether more independent frames exist.

Everything scored here derives from hb_router_probs alone. To keep the claim
honest on the threshold side too, every candidate is calibrated twice:

    per_task   same-task empirical quantile, as v4 does. Needs a task id at
               runtime, so part of the decision is task-side information.
    global     one pooled quantile over the whole unlabeled reference corpus,
               as v7 does. No task-side information anywhere.

A quantity only counts as carrying MoE signal if it survives the global mode
and lifts above the survival prior, which is what "still running" alone buys.

False-alarm dependence is measured as observed co-occurrence over the product
of marginals, not as raw overlap. Two detectors that both fire late overlap for
timing reasons alone; the ratio removes that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import (  # noqa: E402
    LABEL_PATHS,
    MOBILITY_PATHS,
    PROFILE_ROOT,
    load_npz,
)
from compare_layers_early import CONFIRMATIONS, MIN_LOW_PRIOR_PRECISION, WIDTH  # noqa: E402
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)


DEFAULT_OUTPUT = BUNDLE / "results/frame_survey"
REPRESENTATIONS = (
    "L2",
    "L3",
    "L4",
    "L5",
    "L12",
    "L13",
    "L14",
    "L15",
    "front_median",
    "back_median",
    "all_median",
)

# Hypothesised reference frame per quantity, declared before the run. The
# false-alarm dependence matrix is the test of whether these groupings are real.
FRAMES = {
    "mobility": "adjacent_query",
    "conditional_query_d1": "adjacent_query",
    "partial_query_d1": "adjacent_query",
    "flow_path": "denoising_steps",
    "flow_endpoint": "denoising_steps",
    "flow_settling_log_ratio": "denoising_steps",
    "action_consensus": "token_graph",
    "state_action_alignment": "token_graph",
    "conditional_energy": "token_graph",
    "conditional_effective_rank": "token_graph",
    "partial_edge_std": "token_graph",
    "expert_load_effective_rank": "expert_load",
}
MODES = ("per_task", "global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def quantity_cache(
    graph: dict[str, np.ndarray], mobility: dict[str, np.ndarray], quantity: str
) -> dict[str, np.ndarray]:
    """Put any routing quantity into the layout dev.representations expects."""
    if quantity == "mobility":
        values = np.asarray(mobility["mobility"], dtype=np.float32)
    else:
        names = graph["metric_names"].astype(str).tolist()
        values = np.asarray(
            graph["metrics"][:, :, :, names.index(quantity)], dtype=np.float32
        )
    return {
        "mobility": values,
        "valid": graph["valid"].astype(bool),
        "layer_names": graph["layer_names"],
        "task_names": graph["task_names"],
        "task_index": graph["task_index"],
        "episode": graph["episode"],
        "init_state_id": graph["init_state_id"],
        "length": graph["length"],
    }


def oriented(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = dev.trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    graphs = {
        name: load_npz(PROFILE_ROOT / f"{name}.npz")
        for name in ("development_main", "development_extra", "external_8b")
    }
    mobilities = {name: load_npz(MOBILITY_PATHS[name]) for name in graphs}
    for name in graphs:
        if not np.array_equal(graphs[name]["episode"], mobilities[name]["episode"]):
            raise ValueError(f"{name}: graph and mobility caches are not aligned")

    dev_labels = pd.read_csv(LABEL_PATHS["development_main"])
    dev_risk = dev_labels["original_failure"].to_numpy(bool)
    dev_suite = suite_of(graphs["development_main"])
    dev_priors = survival_prior(
        dev_suite, graphs["development_main"]["length"].astype(int), dev_risk
    )
    dev_task_index = graphs["development_main"]["task_index"].astype(int)
    dev_init = graphs["development_main"]["init_state_id"].astype(int)
    dev_valid = graphs["development_main"]["valid"].astype(bool)

    ext_labels = pd.read_csv(LABEL_PATHS["external_8b"])
    ext_risk = ext_labels["original_failure"].to_numpy(bool)
    ext_suite = suite_of(graphs["external_8b"])
    ext_priors = survival_prior(
        ext_suite, graphs["external_8b"]["length"].astype(int), ext_risk
    )
    ext_valid = graphs["external_8b"]["valid"].astype(bool)
    ext_task = graphs["external_8b"]["task_names"].astype(str)[
        graphs["external_8b"]["task_index"].astype(int)
    ]

    development_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}

    for quantity in FRAMES:
        caches = {
            name: quantity_cache(graphs[name], mobilities[name], quantity)
            for name in graphs
        }
        reprs = {
            name: {k: v for k, (v, _) in dev.representations(caches[name]).items()}
            for name in caches
        }
        reference_task = {
            name: caches[name]["task_names"].astype(str)[
                caches[name]["task_index"].astype(int)
            ]
            for name in ("development_main", "development_extra")
        }

        for representation in REPRESENTATIONS:
            for direction in ("low", "high"):
                dev_oriented = oriented(reprs["development_main"][representation], direction)
                dev_persistent = dev.persistent_score(dev_oriented, CONFIRMATIONS)
                dev_peak = dev.row_max(dev_oriented)
                crossfit = dev.crossfit_thresholds(dev_peak, dev_task_index, dev_init)
                pooled_peak = np.concatenate(
                    [
                        dev_peak,
                        dev.row_max(
                            oriented(reprs["development_extra"][representation], direction)
                        ),
                    ]
                )
                for position, quantile in enumerate(dev.QUANTILES):
                    for mode in MODES:
                        if mode == "per_task":
                            line = crossfit[:, position][:, None]
                        else:
                            try:
                                line = dev.quantile_higher(pooled_peak, quantile)
                            except ValueError:
                                continue
                        first = dev.first_query(
                            np.isfinite(dev_persistent)
                            & (dev_persistent > line)
                            & dev_valid
                        )
                        development_rows.append(
                            {
                                "quantity": quantity,
                                "frame": FRAMES[quantity],
                                "representation": representation,
                                "direction": direction,
                                "quantile": quantile,
                                "mode": mode,
                                **score_candidate(
                                    first,
                                    dev_risk,
                                    prior_of(first, dev_suite, dev_priors),
                                ),
                            }
                        )

        frame = pd.DataFrame(development_rows)
        frame = frame[frame["quantity"] == quantity]
        for mode in MODES:
            eligible = frame[
                (frame["mode"] == mode)
                & (frame["timely_fpr"] <= MAX_TIMELY_FPR)
                & (frame["low_prior_precision"] >= MIN_LOW_PRIOR_PRECISION)
            ]
            if eligible.empty:
                external_rows.append(
                    {"quantity": quantity, "frame": FRAMES[quantity], "mode": mode,
                     "feasible": False}
                )
                continue
            best = eligible.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]

            representation, direction = best["representation"], best["direction"]
            quantile = float(best["quantile"])
            ext_persistent = dev.persistent_score(
                oriented(reprs["external_8b"][representation], direction), CONFIRMATIONS
            )
            first = np.full(len(ext_persistent), -1, dtype=np.int16)
            if mode == "global":
                pooled_peak = np.concatenate(
                    [
                        dev.row_max(oriented(reprs[name][representation], direction))
                        for name in ("development_main", "development_extra")
                    ]
                )
                line = dev.quantile_higher(pooled_peak, quantile)
                first = dev.first_query(
                    np.isfinite(ext_persistent) & (ext_persistent > line) & ext_valid
                )
            else:
                for task in np.unique(ext_task):
                    take = np.flatnonzero(ext_task == task)
                    peaks = [
                        dev.row_max(
                            oriented(
                                reprs[name][representation][reference_task[name] == task],
                                direction,
                            )
                        )
                        for name in ("development_main", "development_extra")
                        if (reference_task[name] == task).any()
                    ]
                    line = dev.quantile_higher(np.concatenate(peaks), quantile)
                    first[take] = dev.first_query(
                        np.isfinite(ext_persistent[take])
                        & (ext_persistent[take] > line)
                        & ext_valid[take]
                    )
            key = f"{quantity}|{mode}"
            alarms[key] = first
            prior = prior_of(first, ext_suite, ext_priors)
            external_rows.append(
                {
                    "quantity": quantity,
                    "frame": FRAMES[quantity],
                    "mode": mode,
                    "feasible": True,
                    "representation": representation,
                    "direction": direction,
                    "quantile": quantile,
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    **score_candidate(first, ext_risk, prior),
                }
            )
        print(f"  {quantity}: done", flush=True)

    pd.DataFrame(development_rows).to_csv(
        args.output / "development_candidates.csv", index=False
    )
    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_detectors.csv", index=False)

    # False-alarm dependence: observed co-occurrence over independence.
    timely = ~ext_risk
    keys = [k for k in alarms if k.endswith("|global")]
    dependence = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for left in keys:
        for right in keys:
            a = (alarms[left] >= 0) & timely
            b = (alarms[right] >= 0) & timely
            n = int(timely.sum())
            expected = a.sum() * b.sum() / n
            dependence.loc[left, right] = (
                float((a & b).sum() / expected) if expected > 0 else np.nan
            )
    dependence.to_csv(args.output / "false_alarm_dependence_global.csv")

    np.savez_compressed(
        args.output / "external_first_alarms.npz",
        schema=np.asarray("himoe.hb_front_back.frame_survey.alarms.v1"),
        **alarms,
    )
    (args.output / "survey.json").write_text(
        json.dumps(
            {
                "schema": "himoe.hb_front_back.frame_survey.v1",
                "all_quantities_derive_from": "hb_router_probs",
                "frames_declared_before_run": FRAMES,
                "threshold_modes": {
                    "per_task": "same-task empirical quantile, uses task identity",
                    "global": "one pooled quantile, no task-side information",
                },
                "selection_rule": {
                    "timely_fpr_at_most": MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                    "applied_identically_to_every_quantity": True,
                },
                "dependence_definition": "observed co-occurrence / product of marginals",
                "low_prior_threshold": LOW_PRIOR,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for mode in MODES:
        block = external[(external["mode"] == mode) & external["feasible"].fillna(False)]
        print(f"\n=== external, {mode} threshold ===")
        print(
            block[
                [
                    "quantity",
                    "frame",
                    "representation",
                    "direction",
                    "quantile",
                    "tp",
                    "fp",
                    "low_prior_tp",
                    "low_prior_fp",
                    "lift",
                    "risk_recall",
                ]
            ].to_string(index=False, float_format="%.3f"),
            flush=True,
        )


if __name__ == "__main__":
    main()
