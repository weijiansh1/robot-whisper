#!/usr/bin/env python3
"""Which of the eight HB layers carries early-detection signal?

The early-lock experiment searched a mixed representation family and landed on
L2, the first HB routing layer. That sweep did not include L12-L15 individually,
so whether the deep layers carry their own early signal was untested.

This compares all eight layers head to head under identical v4 lock parameters
(low mobility, W4, K4), picking each layer's quantile by one fixed development
rule so no layer gets a hand-tuned advantage. Both the bare lock and the lock
OR'd with the frozen L5 instability head are reported; for L5 itself the OR is
degenerate and only the bare head is meaningful.
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
    calibrated_external_alarm,
    load_npz,
    profile_task_map,
)
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)


DEFAULT_OUTPUT = BUNDLE / "results/layer_early_comparison"
LAYERS = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
WIDTH, CONFIRMATIONS = 4, 4

# One fixed rule applied identically to every layer, declared before the run:
# take the most early detections available at a false alarm rate no worse than
# v4's and an early band that is at least majority correct.
MIN_LOW_PRIOR_PRECISION = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main_cache = load_npz(MOBILITY_PATHS["development_main"])
    extra_cache = load_npz(MOBILITY_PATHS["development_extra"])
    external_cache = load_npz(MOBILITY_PATHS["external_8b"])
    main_repr = {n: v for n, (v, _) in dev.representations(main_cache).items()}
    extra_repr = {n: v for n, (v, _) in dev.representations(extra_cache).items()}
    external_repr = {n: v for n, (v, _) in dev.representations(external_cache).items()}

    labels = pd.read_csv(LABEL_PATHS["development_main"])
    risk = labels["original_failure"].to_numpy(bool)
    suite = suite_of(main_cache)
    priors = survival_prior(suite, main_cache["length"].astype(int), risk)
    task_index = main_cache["task_index"].astype(int)
    init_state = main_cache["init_state_id"].astype(int)
    valid = main_cache["valid"].astype(bool)

    def development_lock(layer: str, quantile: float) -> np.ndarray:
        oriented = -dev.trailing_mean(main_repr[layer], WIDTH)
        thresholds = dev.crossfit_thresholds(
            dev.row_max(oriented), task_index, init_state
        )[:, dev.QUANTILES.index(quantile)]
        persistent = dev.persistent_score(oriented, CONFIRMATIONS)
        return dev.first_query(
            np.isfinite(persistent) & (persistent > thresholds[:, None]) & valid
        )

    l5_oriented = -dev.trailing_mean(main_repr["L5"], 4)
    l5_oriented = -l5_oriented  # instability is the high direction
    l5_persistent = dev.persistent_score(l5_oriented, 8)
    l5_dev = dev.first_query(
        np.isfinite(l5_persistent)
        & (
            l5_persistent
            > dev.crossfit_thresholds(
                dev.row_max(l5_oriented), task_index, init_state
            )[:, dev.QUANTILES.index(0.80)][:, None]
        )
        & valid
    )

    def merge(lock: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.where(
            lock < 0, other, np.where(other < 0, lock, np.minimum(lock, other))
        ).astype(np.int16)

    rows: list[dict[str, Any]] = []
    for layer in LAYERS:
        for quantile in dev.QUANTILES:
            lock = development_lock(layer, quantile)
            for structure, first in (("single", lock), ("or_l5", merge(lock, l5_dev))):
                rows.append(
                    {
                        "layer": layer,
                        "quantile": quantile,
                        "structure": structure,
                        **score_candidate(first, risk, prior_of(first, suite, priors)),
                    }
                )
    development = pd.DataFrame(rows)
    development.to_csv(args.output / "development_layers.csv", index=False)

    picks: list[dict[str, Any]] = []
    for layer in LAYERS:
        for structure in ("single", "or_l5"):
            block = development[
                (development["layer"] == layer)
                & (development["structure"] == structure)
                & (development["timely_fpr"] <= MAX_TIMELY_FPR)
                & (development["low_prior_precision"] >= MIN_LOW_PRIOR_PRECISION)
            ]
            if block.empty:
                picks.append(
                    {"layer": layer, "structure": structure, "feasible": False}
                )
                continue
            best = block.sort_values(
                ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
            ).iloc[0]
            picks.append(
                {
                    "layer": layer,
                    "structure": structure,
                    "feasible": True,
                    "quantile": float(best["quantile"]),
                    "dev_low_prior_tp": int(best["low_prior_tp"]),
                    "dev_low_prior_fp": int(best["low_prior_fp"]),
                }
            )

    reference_map = profile_task_map([main_cache, extra_cache])
    reference_representations = {id(main_cache): main_repr, id(extra_cache): extra_repr}
    external_labels = pd.read_csv(LABEL_PATHS["external_8b"])
    external_risk = external_labels["original_failure"].to_numpy(bool)
    external_suite = suite_of(external_cache)
    external_priors = survival_prior(
        external_suite, external_cache["length"].astype(int), external_risk
    )
    external_l5 = calibrated_external_alarm(
        external_cache,
        external_repr["L5"],
        reference_map,
        reference_representations,
        "L5",
        "high",
        4,
        8,
        0.80,
    )

    external_rows: list[dict[str, Any]] = []
    for pick in picks:
        if not pick["feasible"]:
            external_rows.append(pick)
            continue
        lock = calibrated_external_alarm(
            external_cache,
            external_repr[pick["layer"]],
            reference_map,
            reference_representations,
            pick["layer"],
            "low",
            WIDTH,
            CONFIRMATIONS,
            pick["quantile"],
        )
        first = lock if pick["structure"] == "single" else merge(lock, external_l5)
        prior = prior_of(first, external_suite, external_priors)
        record = {**pick, **score_candidate(first, external_risk, prior)}
        for name in np.unique(external_suite):
            take = external_suite == name
            block = score_candidate(first[take], external_risk[take], prior[take])
            record[f"{name}__low_prior_tp"] = block["low_prior_tp"]
            record[f"{name}__low_prior_fp"] = block["low_prior_fp"]
        external_rows.append(record)

    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_layers.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "schema": "himoe.hb_front_back.layer_early_comparison.v1",
                "lock_parameters": {"direction": "low", "width": WIDTH,
                                    "confirmations": CONFIRMATIONS},
                "selection_rule": {
                    "timely_fpr_at_most": MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                    "applied_identically_to_every_layer": True,
                },
                "low_prior_threshold": LOW_PRIOR,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for structure in ("single", "or_l5"):
        block = external[external["structure"] == structure]
        print(f"\n=== {structure} ===")
        print(
            block[
                [
                    "layer",
                    "quantile",
                    "dev_low_prior_tp",
                    "dev_low_prior_fp",
                    "low_prior_tp",
                    "low_prior_fp",
                    "low_prior_precision",
                    "tp",
                    "fp",
                    "risk_recall",
                ]
            ].to_string(index=False, float_format="%.3f"),
            flush=True,
        )


if __name__ == "__main__":
    main()
