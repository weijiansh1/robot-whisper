#!/usr/bin/env python3
"""Rebuild per-episode first-alarm arrays for the frame-survey detector pool on
both cohorts, at every sensitivity level, so that combination rules can be
selected on development and replayed on external.

Nothing here is new signal. The representation and direction of every pool
member are taken verbatim from the frame survey's development selection; the
only thing that is swept is the shared quantile level, because a k-of-n vote is
only interpretable when every voter is thresholded at the same tail probability
of the same unlabeled reference distribution.

The reconstruction is checked against the sealed external alarms of the frame
survey: at the selected level, every one of the twenty-four detectors must
reproduce exactly. If it does not, the script fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
PROJECT = BUNDLE.parent
SURVEY = PROJECT / "moe-hb-front-back-0905"

sys.path.insert(0, str(SURVEY / "experiments"))
sys.path.insert(0, str(PROJECT / "moe-v4-0904/experiments"))

import evaluate_layerwise_alarm_development as dev  # noqa: E402
from analyze_front_back import (  # noqa: E402
    LABEL_PATHS,
    MOBILITY_PATHS,
    PROFILE_ROOT,
    load_npz,
)
from compare_layers_early import CONFIRMATIONS, WIDTH  # noqa: E402


DEFAULT_OUTPUT = BUNDLE / "results/alarms"
SEALED_ALARMS = SURVEY / "results/frame_survey/external_first_alarms.npz"
DETECTOR_TABLE = SURVEY / "results/frame_survey/external_detectors.csv"
COHORTS = ("development_main", "development_extra", "external_8b")
SCORED = ("development_main", "external_8b")
MODES = ("per_task", "global")
LEVELS = ("selected",) + tuple(f"{q:g}" for q in dev.QUANTILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def quantity_values(
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

    detectors = pd.read_csv(DETECTOR_TABLE)
    detectors = detectors[detectors["feasible"].fillna(False).astype(bool)]
    if len(detectors) != 24:
        raise ValueError(f"expected 24 feasible detectors, found {len(detectors)}")

    graphs = {name: load_npz(PROFILE_ROOT / f"{name}.npz") for name in COHORTS}
    mobilities = {name: load_npz(MOBILITY_PATHS[name]) for name in COHORTS}
    for name in COHORTS:
        if not np.array_equal(graphs[name]["episode"], mobilities[name]["episode"]):
            raise ValueError(f"{name}: graph and mobility caches are not aligned")

    reference_task = {
        name: graphs[name]["task_names"].astype(str)[
            graphs[name]["task_index"].astype(int)
        ]
        for name in ("development_main", "development_extra")
    }
    external_task = graphs["external_8b"]["task_names"].astype(str)[
        graphs["external_8b"]["task_index"].astype(int)
    ]
    dev_task_index = graphs["development_main"]["task_index"].astype(int)
    dev_init = graphs["development_main"]["init_state_id"].astype(int)
    valid = {name: graphs[name]["valid"].astype(bool) for name in SCORED}

    alarms: dict[str, dict[str, np.ndarray]] = {name: {} for name in SCORED}
    index_rows: list[dict[str, object]] = []

    for quantity in sorted(detectors["quantity"].unique()):
        caches = {
            name: quantity_values(graphs[name], mobilities[name], quantity)
            for name in COHORTS
        }
        block = detectors[detectors["quantity"] == quantity]
        wanted = sorted({(str(r["representation"]), str(r["direction"])) for _, r in block.iterrows()})
        reprs = {
            name: {
                key: value
                for key, (value, _) in dev.representations(caches[name]).items()
                if key in {rep for rep, _ in wanted}
            }
            for name in COHORTS
        }

        for _, row in block.iterrows():
            mode = str(row["mode"])
            representation = str(row["representation"])
            direction = str(row["direction"])
            selected_quantile = float(row["quantile"])
            frame = str(row["frame"])

            oriented_by_cohort = {
                name: oriented(reprs[name][representation], direction)
                for name in COHORTS
            }
            persistent = {
                name: dev.persistent_score(oriented_by_cohort[name], CONFIRMATIONS)
                for name in SCORED
            }
            peaks = {
                name: dev.row_max(oriented_by_cohort[name]) for name in COHORTS
            }
            pooled_peak = np.concatenate(
                [peaks["development_main"], peaks["development_extra"]]
            )
            crossfit = dev.crossfit_thresholds(
                peaks["development_main"], dev_task_index, dev_init
            )

            for level in LEVELS:
                quantile = (
                    selected_quantile if level == "selected" else float(level)
                )
                position = dev.QUANTILES.index(
                    min(dev.QUANTILES, key=lambda q: abs(q - quantile))
                )
                if abs(dev.QUANTILES[position] - quantile) > 1e-9:
                    raise ValueError(f"quantile {quantile} is not on the grid")
                key = f"{quantity}|{mode}|{level}"

                if mode == "per_task":
                    line = crossfit[:, position][:, None]
                else:
                    line = dev.quantile_higher(pooled_peak, quantile)
                    if not np.isfinite(line):
                        continue
                development_first = dev.first_query(
                    np.isfinite(persistent["development_main"])
                    & (persistent["development_main"] > line)
                    & valid["development_main"]
                )

                external_first = np.full(len(external_task), -1, dtype=np.int16)
                if mode == "global":
                    external_first = dev.first_query(
                        np.isfinite(persistent["external_8b"])
                        & (persistent["external_8b"] > line)
                        & valid["external_8b"]
                    )
                else:
                    for task in np.unique(external_task):
                        take = np.flatnonzero(external_task == task)
                        task_peaks = [
                            dev.row_max(
                                oriented(
                                    reprs[name][representation][
                                        reference_task[name] == task
                                    ],
                                    direction,
                                )
                            )
                            for name in ("development_main", "development_extra")
                            if (reference_task[name] == task).any()
                        ]
                        task_line = dev.quantile_higher(
                            np.concatenate(task_peaks), quantile
                        )
                        external_first[take] = dev.first_query(
                            np.isfinite(persistent["external_8b"][take])
                            & (persistent["external_8b"][take] > task_line)
                            & valid["external_8b"][take]
                        )

                alarms["development_main"][key] = development_first
                alarms["external_8b"][key] = external_first
                index_rows.append(
                    {
                        "key": key,
                        "quantity": quantity,
                        "frame": frame,
                        "mode": mode,
                        "level": level,
                        "quantile": quantile,
                        "representation": representation,
                        "direction": direction,
                        "development_alarms": int((development_first >= 0).sum()),
                        "external_alarms": int((external_first >= 0).sum()),
                    }
                )
        print(f"  {quantity}: done", flush=True)

    sealed = np.load(SEALED_ALARMS, allow_pickle=False)
    mismatches: list[str] = []
    for _, row in detectors.iterrows():
        key = f"{row['quantity']}|{row['mode']}|selected"
        expected = sealed[f"{row['quantity']}|{row['mode']}"].astype(np.int16)
        observed = alarms["external_8b"][key]
        if not np.array_equal(observed, expected):
            mismatches.append(f"{key}: {int((observed != expected).sum())} rows")
    if mismatches:
        raise AssertionError("sealed external alarms not reproduced: " + "; ".join(mismatches))

    for cohort in SCORED:
        np.savez_compressed(
            args.output / f"{cohort}_alarms.npz",
            schema=np.asarray("himoe.combination_rules.alarms.v1"),
            **alarms[cohort],
        )
    index = pd.DataFrame(index_rows).sort_values(["quantity", "mode", "level"])
    index.to_csv(args.output / "detector_index.csv", index=False)

    labels = {
        cohort: pd.read_csv(LABEL_PATHS[cohort])["original_failure"].to_numpy(bool)
        for cohort in SCORED
    }
    (args.output / "build.json").write_text(
        json.dumps(
            {
                "schema": "himoe.combination_rules.alarm_build.v1",
                "trailing_mean_width": WIDTH,
                "consecutive_confirmations": CONFIRMATIONS,
                "levels": list(LEVELS),
                "modes": list(MODES),
                "detectors": int(len(detectors)),
                "arrays_per_cohort": {
                    cohort: len(alarms[cohort]) for cohort in SCORED
                },
                "episodes": {
                    cohort: int(len(labels[cohort])) for cohort in SCORED
                },
                "risks": {cohort: int(labels[cohort].sum()) for cohort in SCORED},
                "sealed_external_reproduced": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"built {len(alarms['development_main'])} development and "
        f"{len(alarms['external_8b'])} external alarm arrays; "
        "sealed external detectors reproduced exactly",
        flush=True,
    )


if __name__ == "__main__":
    main()
