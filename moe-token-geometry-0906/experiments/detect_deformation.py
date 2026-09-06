#!/usr/bin/env python3
"""Does the token geometry deform before a rollout fails to finish in time?

Every candidate is a per-query scalar computed from ``hb_router_probs`` alone.
The alarm machinery is taken unchanged from the moe-v4-0904 / moe-hb-front-back
bundles so that the numbers land on the same scale as the published baselines:

    trailing mean of width 4  ->  minimum over 4 consecutive queries  ->
    first query above a threshold calibrated without outcome labels

Two calibrations are reported for every candidate:

    per_task   same-task empirical quantile of the reference corpus.  A task id
               is needed at runtime, so part of the decision is task-side.
    global     one pooled quantile over the whole reference corpus.  No
               task-side information anywhere; only this mode proves that the
               signal is MoE-side.

Selection reads development outcomes only, with the rule predeclared in
moe-hb-front-back-0905/experiments/survey_reference_frames.py:

    timely_fpr <= 0.005 and low_prior_precision >= 0.60,
    ranked by low_prior_tp then low_prior_precision

The winner per quantity per mode is replayed on external_8b once.  Precision is
always read against the per-suite survival prior P(risk | still running at q)
evaluated at each alarm's own chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import geometry as G


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(G.PROJECT / "moe-v4-0904/experiments"))
sys.path.insert(0, str(G.PROJECT / "moe-hb-front-back-0905/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)


DEFAULT_OUTPUT = G.BUNDLE / "results/detection"
FEATURES = G.FEATURE_ROOT
LABEL_ROOT = G.PROJECT / "double-selete/trainfree/results/timeout_extension_plus10"
LABEL_PATHS = {
    "development_main": LABEL_ROOT / "development_main_clean_labels.csv",
    "external_8b": LABEL_ROOT / "external_8b_clean_labels.csv",
}
REFERENCE = ("development_main", "development_extra")
COHORTS = ("development_main", "development_extra", "external_8b")
REPRESENTATIONS = (
    "L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15",
    "front_median", "back_median", "all_median",
)
WIDTH, CONFIRMATIONS = 4, 4
MIN_LOW_PRIOR_PRECISION = 0.60
MODES = ("per_task", "global")

# Reference frame of every candidate, declared before the run.
FRAMES = {
    "shape_pr": "shape_spectrum",
    "shape_erank": "shape_spectrum",
    "lam1_share": "shape_spectrum",
    "lam12_share": "shape_spectrum",
    "bandedness": "shape_order",
    "d_cv": "shape_order",
    "size": "configuration_scale",
    "kernel_trace": "configuration_scale",
    "centroid_norm2": "configuration_scale",
    "procrustes_layer": "corpus_template",
    "procrustes_prev": "adjacent_query",
    "procrustes_self": "own_history",
    "kernel_erank": "control_published",
    "mobility": "control_published",
}
PUBLISHED_EXTERNAL = {
    "best_global": {"quantity": "mobility(L12)", "tp": 195, "fp": 17,
                    "precision": 0.920, "lift": 1.764},
    "best_per_task": {"quantity": "expert_load_effective_rank(L3)", "tp": 370,
                      "fp": 93, "precision": 0.799, "lift": 1.548},
}
# ``kernel_erank`` is this bundle's recomputation of the published
# ``conditional_effective_rank``.  If the pipeline is wired identically to the
# frame survey it must land on exactly the published external counts, which is
# checked before anything else in this file is believed.
CONTROL_ANCHOR = {
    "per_task": {"representation": "L13", "direction": "low", "quantile": 0.90,
                 "tp": 103, "fp": 10},
    "global": {"representation": "L3", "direction": "high", "quantile": 0.925,
               "tp": 88, "fp": 158},
}
MOBILITY_ANCHOR = {
    "per_task": {"representation": "L2", "direction": "low", "quantile": 0.70,
                 "tp": 272, "fp": 57},
    "global": {"representation": "L12", "direction": "low", "quantile": 0.975,
               "tp": 195, "fp": 17},
}
DYNAMIC_NAMES = ("procrustes_prev", "procrustes_self")
MOBILITY_PATHS = {
    "development_main": G.PROJECT / "moe-v4-0904/results/layerwise_mobility/main_reference.npz",
    "development_extra": G.PROJECT / "moe-v4-0904/results/layerwise_mobility/extra_reference.npz",
    "external_8b": G.PROJECT / "moe-v4-0904/results/layerwise_mobility/external_8b.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def dense_quantity(cohort: str, quantity: str, index: dict[str, np.ndarray]) -> np.ndarray:
    valid = index["valid"].astype(bool)
    if quantity == "mobility":
        with np.load(MOBILITY_PATHS[cohort], allow_pickle=False) as archive:
            values = np.asarray(archive["mobility"], dtype=np.float32).copy()
        values[~valid] = np.nan
        return values
    if quantity in DYNAMIC_NAMES:
        packed = np.load(FEATURES / f"{cohort}_dynamic.npy")
        column = DYNAMIC_NAMES.index(quantity)
    else:
        packed = np.load(FEATURES / f"{cohort}_features.npy")
        column = G.FEATURE_NAMES.index(quantity)
    return G.expand(packed[:, :, column], valid)


def quantity_cache(index: dict[str, np.ndarray], values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mobility": values,
        "valid": index["valid"].astype(bool),
        "layer_names": np.asarray(G.LAYER_NAMES),
        "task_names": index["task_names"],
        "task_index": index["task_index"],
        "episode": index["episode"],
        "init_state_id": index["init_state_id"],
        "length": index["length"],
    }


def oriented(values: np.ndarray, direction: str) -> np.ndarray:
    smoothed = dev.trailing_mean(values, WIDTH)
    return -smoothed if direction == "low" else smoothed


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    indices = {cohort: G.load_index(cohort) for cohort in COHORTS}
    for cohort in COHORTS:
        graph = np.load(
            G.PROJECT / f"moe-hb-front-back-0905/results/layer_graphs/{cohort}.npz",
            allow_pickle=False,
        )
        if not np.array_equal(graph["episode"], indices[cohort]["episode"]):
            raise ValueError(f"{cohort}: kernel index is not aligned with the layer graphs")

    dev_labels = pd.read_csv(LABEL_PATHS["development_main"])
    dev_risk = dev_labels["original_failure"].to_numpy(bool)
    dev_suite = suite_of(indices["development_main"])
    dev_length = indices["development_main"]["length"].astype(int)
    dev_priors = survival_prior(dev_suite, dev_length, dev_risk)
    dev_task_index = indices["development_main"]["task_index"].astype(int)
    dev_init = indices["development_main"]["init_state_id"].astype(int)
    dev_valid = indices["development_main"]["valid"].astype(bool)

    ext_labels = pd.read_csv(LABEL_PATHS["external_8b"])
    ext_risk = ext_labels["original_failure"].to_numpy(bool)
    ext_suite = suite_of(indices["external_8b"])
    ext_priors = survival_prior(
        ext_suite, indices["external_8b"]["length"].astype(int), ext_risk
    )
    ext_valid = indices["external_8b"]["valid"].astype(bool)
    ext_task = indices["external_8b"]["task_names"].astype(str)[
        indices["external_8b"]["task_index"].astype(int)
    ]
    reference_task = {
        cohort: indices[cohort]["task_names"].astype(str)[
            indices[cohort]["task_index"].astype(int)
        ]
        for cohort in REFERENCE
    }

    development_rows: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []
    alarms: dict[str, np.ndarray] = {}

    for quantity in FRAMES:
        reprs = {
            cohort: {
                name: values
                for name, (values, _) in dev.representations(
                    quantity_cache(indices[cohort], dense_quantity(cohort, quantity, indices[cohort]))
                ).items()
            }
            for cohort in COHORTS
        }
        rows_here: list[dict[str, Any]] = []
        for representation in REPRESENTATIONS:
            for direction in ("low", "high"):
                dev_oriented = oriented(reprs["development_main"][representation], direction)
                dev_persistent = dev.persistent_score(dev_oriented, CONFIRMATIONS)
                finite = np.isfinite(dev_persistent)
                dev_peak = dev.row_max(dev_oriented)
                crossfit = dev.crossfit_thresholds(dev_peak, dev_task_index, dev_init)
                pooled_peak = np.concatenate(
                    [
                        dev.row_max(oriented(reprs[cohort][representation], direction))
                        for cohort in REFERENCE
                    ]
                )
                for position, quantile in enumerate(dev.QUANTILES):
                    for mode in MODES:
                        if mode == "per_task":
                            line = crossfit[:, position][:, None]
                        else:
                            line = dev.quantile_higher(pooled_peak, quantile)
                            if not np.isfinite(line):
                                continue
                        first = dev.first_query(finite & (dev_persistent > line) & dev_valid)
                        rows_here.append(
                            {
                                "quantity": quantity,
                                "frame": FRAMES[quantity],
                                "representation": representation,
                                "direction": direction,
                                "quantile": quantile,
                                "mode": mode,
                                **score_candidate(
                                    first, dev_risk, prior_of(first, dev_suite, dev_priors)
                                ),
                            }
                        )
        development_rows.extend(rows_here)
        frame = pd.DataFrame(rows_here)

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
            if mode == "global":
                pooled_peak = np.concatenate(
                    [
                        dev.row_max(oriented(reprs[cohort][representation], direction))
                        for cohort in REFERENCE
                    ]
                )
                line = dev.quantile_higher(pooled_peak, quantile)
                first = dev.first_query(
                    np.isfinite(ext_persistent) & (ext_persistent > line) & ext_valid
                )
            else:
                first = np.full(len(ext_persistent), -1, dtype=np.int16)
                for task in np.unique(ext_task):
                    take = np.flatnonzero(ext_task == task)
                    peaks = [
                        dev.row_max(
                            oriented(
                                reprs[cohort][representation][reference_task[cohort] == task],
                                direction,
                            )
                        )
                        for cohort in REFERENCE
                        if (reference_task[cohort] == task).any()
                    ]
                    line = dev.quantile_higher(np.concatenate(peaks), quantile)
                    first[take] = dev.first_query(
                        np.isfinite(ext_persistent[take])
                        & (ext_persistent[take] > line)
                        & ext_valid[take]
                    )
            alarms[f"{quantity}|{mode}"] = first
            prior = prior_of(first, ext_suite, ext_priors)
            record = {
                "quantity": quantity,
                "frame": FRAMES[quantity],
                "mode": mode,
                "feasible": True,
                "representation": representation,
                "direction": direction,
                "quantile": quantile,
                "dev_tp": int(best["tp"]),
                "dev_fp": int(best["fp"]),
                "dev_low_prior_tp": int(best["low_prior_tp"]),
                "dev_low_prior_precision": float(best["low_prior_precision"]),
                **score_candidate(first, ext_risk, prior),
            }
            fired = first >= 0
            record["median_alarm_chunk"] = (
                float(np.median(first[fired])) if fired.any() else float("nan")
            )
            external_rows.append(record)
        print(f"  {quantity}: swept {len(rows_here)} development candidates", flush=True)

    for quantity, anchors in (
        ("kernel_erank", CONTROL_ANCHOR),
        ("mobility", MOBILITY_ANCHOR),
    ):
        for mode, anchor in anchors.items():
            row = next(
                r for r in external_rows
                if r["quantity"] == quantity and r["mode"] == mode
            )
            rebuilt = {key: row[key] for key in anchor}
            if rebuilt != anchor:
                raise ValueError(
                    f"{quantity} anchor mismatch in {mode}: "
                    f"rebuilt {rebuilt} vs published {anchor}"
                )
    print(
        "controls reproduce the published conditional_effective_rank and mobility rows",
        flush=True,
    )

    development = pd.DataFrame(development_rows)
    development.to_csv(args.output / "development_candidates.csv", index=False)
    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_detectors.csv", index=False)

    # Development-side ceiling: the most a quantity can do under the false-alarm
    # cap, before any external cache is opened.  This is what says whether the
    # geometry is weak or merely constrained by the selection rule.
    ceiling = []
    for (quantity, mode), block in development.groupby(["quantity", "mode"]):
        capped = block[block["timely_fpr"] <= MAX_TIMELY_FPR]
        if capped.empty:
            continue
        peak = capped.loc[capped["tp"].idxmax()]
        ceiling.append(
            {
                "quantity": quantity,
                "frame": FRAMES[quantity],
                "mode": mode,
                "dev_max_tp_at_fpr_cap": int(peak["tp"]),
                "dev_fp_at_that_point": int(peak["fp"]),
                "dev_max_low_prior_tp": int(capped["low_prior_tp"].max()),
            }
        )
    pd.DataFrame(ceiling).to_csv(args.output / "development_ceiling.csv", index=False)

    # Cross-reference against the published mobility alarm on the same cohort.
    published = np.load(
        G.PROJECT / "moe-hb-front-back-0905/results/frame_survey/external_first_alarms.npz",
        allow_pickle=False,
    )
    timely = ~ext_risk
    dependence: list[dict[str, Any]] = []
    for key, first in alarms.items():
        for other in ("mobility|global", "mobility|per_task",
                      "expert_load_effective_rank|per_task"):
            if other not in published.files:
                continue
            a = (first >= 0) & timely
            b = (np.asarray(published[other]) >= 0) & timely
            expected = a.sum() * b.sum() / int(timely.sum())
            dependence.append(
                {
                    "geometry": key,
                    "published": other,
                    "geometry_false_alarms": int(a.sum()),
                    "published_false_alarms": int(b.sum()),
                    "shared_false_alarms": int((a & b).sum()),
                    "dependence_ratio": float((a & b).sum() / expected)
                    if expected > 0
                    else float("nan"),
                }
            )
    pd.DataFrame(dependence).to_csv(
        args.output / "false_alarm_dependence_vs_published.csv", index=False
    )

    np.savez_compressed(
        args.output / "external_first_alarms.npz",
        schema=np.asarray("himoe.token_geometry.alarms.v1"),
        **alarms,
    )
    (args.output / "detection.json").write_text(
        json.dumps(
            {
                "schema": "himoe.token_geometry.detection.v1",
                "all_quantities_derive_from": "hb_router_probs final denoising step",
                "frames_declared_before_run": FRAMES,
                "alarm_machinery": {
                    "trailing_mean_width": WIDTH,
                    "consecutive_confirmations": CONFIRMATIONS,
                    "frozen_from": "moe-hb-front-back-0905/experiments/compare_layers_early.py",
                },
                "selection_rule": {
                    "timely_fpr_at_most": MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                    "reads": "development outcomes only",
                    "identical_to": "moe-hb-front-back-0905 frame survey",
                },
                "published_external_baselines": PUBLISHED_EXTERNAL,
                "low_prior_threshold": LOW_PRIOR,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.set_option("display.width", 240)
    for mode in MODES:
        block = external[(external["mode"] == mode) & external["feasible"].fillna(False)]
        print(f"\n=== external_8b, {mode} threshold ===")
        print(
            block[
                [
                    "quantity", "frame", "representation", "direction", "quantile",
                    "tp", "fp", "precision", "low_prior_tp", "low_prior_fp",
                    "mean_alarm_prior", "lift", "risk_recall", "median_alarm_chunk",
                ]
            ].to_string(index=False, float_format="%.3f"),
            flush=True,
        )


if __name__ == "__main__":
    main()
