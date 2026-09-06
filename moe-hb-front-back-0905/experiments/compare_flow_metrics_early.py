#!/usr/bin/env python3
"""Does the denoising-step axis carry early-detection signal?

Every lock head in this project reads routing at the final flow step only. The
layer-graph cache already holds three quantities that describe the denoising
trajectory itself, and none has ever been tested as a detector:

    flow_path                 path length of the sqrt-probability route over
                              the ten denoising steps
    flow_endpoint             straight-line distance from first to last step
    flow_settling_log_ratio   log(mean speed of last three steps / first three),
                              i.e. whether denoising settles or keeps moving

They are scored through the identical protocol used for mobility: same layer and
group representations, same causal trailing mean, same consecutive confirmation,
same outcome-blind same-task quantile thresholds, same early-band objective, and
the same fixed per-representation selection rule. Only the underlying quantity
changes, so the comparison isolates the denoising axis.
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
from compare_layers_early import (  # noqa: E402
    CONFIRMATIONS,
    MIN_LOW_PRIOR_PRECISION,
    WIDTH,
)
from select_early_lock import (  # noqa: E402
    LOW_PRIOR,
    MAX_TIMELY_FPR,
    prior_of,
    score_candidate,
    suite_of,
    survival_prior,
)


DEFAULT_OUTPUT = BUNDLE / "results/flow_early"
FLOW_METRICS = ("flow_path", "flow_endpoint", "flow_settling_log_ratio")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric_cache(graph: dict[str, np.ndarray], metric: str) -> dict[str, np.ndarray]:
    """Reshape one layer-graph metric into the cache layout dev expects."""
    names = graph["metric_names"].astype(str).tolist()
    values = np.asarray(graph["metrics"][:, :, :, names.index(metric)], dtype=np.float32)
    return {
        "mobility": values,
        "valid": graph["valid"].astype(bool),
        "layer_names": graph["layer_names"],
        "task_names": graph["task_names"],
        "task_index": graph["task_index"],
        "episode": graph["episode"],
        "init_state_id": graph["init_state_id"],
    }


def assert_aligned(graph: dict[str, np.ndarray], mobility: dict[str, np.ndarray]) -> None:
    for field in ("task_index", "episode", "init_state_id", "length"):
        if not np.array_equal(graph[field], mobility[field]):
            raise ValueError(f"layer-graph and mobility caches disagree on {field}")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    graphs = {
        name: load_npz(PROFILE_ROOT / f"{name}.npz")
        for name in ("development_main", "development_extra", "external_8b")
    }
    for name, graph in graphs.items():
        assert_aligned(graph, load_npz(MOBILITY_PATHS[name]))
    print("layer-graph caches align with the mobility caches", flush=True)

    main_graph = graphs["development_main"]
    labels = pd.read_csv(LABEL_PATHS["development_main"])
    risk = labels["original_failure"].to_numpy(bool)
    suite = suite_of(main_graph)
    priors = survival_prior(suite, main_graph["length"].astype(int), risk)
    task_index = main_graph["task_index"].astype(int)
    init_state = main_graph["init_state_id"].astype(int)
    valid = main_graph["valid"].astype(bool)

    external_graph = graphs["external_8b"]
    external_labels = pd.read_csv(LABEL_PATHS["external_8b"])
    external_risk = external_labels["original_failure"].to_numpy(bool)
    external_suite = suite_of(external_graph)
    external_priors = survival_prior(
        external_suite, external_graph["length"].astype(int), external_risk
    )
    external_task = external_graph["task_names"].astype(str)[
        external_graph["task_index"].astype(int)
    ]

    development_rows: list[dict[str, Any]] = []
    picks: list[dict[str, Any]] = []
    external_rows: list[dict[str, Any]] = []

    for metric in FLOW_METRICS:
        dev_repr = {
            n: v for n, (v, _) in dev.representations(metric_cache(main_graph, metric)).items()
        }
        extra_repr = {
            n: v
            for n, (v, _) in dev.representations(
                metric_cache(graphs["development_extra"], metric)
            ).items()
        }
        ext_repr = {
            n: v
            for n, (v, _) in dev.representations(metric_cache(external_graph, metric)).items()
        }
        reference = {
            "main": (metric_cache(main_graph, metric), dev_repr),
            "extra": (metric_cache(graphs["development_extra"], metric), extra_repr),
        }
        reference_task = {
            key: cache["task_names"].astype(str)[cache["task_index"].astype(int)]
            for key, (cache, _) in reference.items()
        }

        for representation in REPRESENTATIONS:
            for direction in ("low", "high"):
                oriented = dev.trailing_mean(dev_repr[representation], WIDTH)
                if direction == "low":
                    oriented = -oriented
                thresholds = dev.crossfit_thresholds(
                    dev.row_max(oriented), task_index, init_state
                )
                persistent = dev.persistent_score(oriented, CONFIRMATIONS)
                finite = np.isfinite(persistent)
                for position, quantile in enumerate(dev.QUANTILES):
                    first = dev.first_query(
                        finite & (persistent > thresholds[:, position][:, None]) & valid
                    )
                    development_rows.append(
                        {
                            "metric": metric,
                            "representation": representation,
                            "direction": direction,
                            "quantile": quantile,
                            **score_candidate(
                                first, risk, prior_of(first, suite, priors)
                            ),
                        }
                    )

        frame = pd.DataFrame(development_rows)
        frame = frame[frame["metric"] == metric]
        eligible = frame[
            (frame["timely_fpr"] <= MAX_TIMELY_FPR)
            & (frame["low_prior_precision"] >= MIN_LOW_PRIOR_PRECISION)
        ]
        if eligible.empty:
            picks.append({"metric": metric, "feasible": False})
            continue
        best = eligible.sort_values(
            ["low_prior_tp", "low_prior_precision"], ascending=False, kind="stable"
        ).iloc[0]
        pick = {
            "metric": metric,
            "feasible": True,
            "representation": best["representation"],
            "direction": best["direction"],
            "quantile": float(best["quantile"]),
            "dev_low_prior_tp": int(best["low_prior_tp"]),
            "dev_low_prior_fp": int(best["low_prior_fp"]),
        }
        picks.append(pick)

        # External replay with same-task reference thresholds.
        oriented = dev.trailing_mean(ext_repr[pick["representation"]], WIDTH)
        if pick["direction"] == "low":
            oriented = -oriented
        persistent = dev.persistent_score(oriented, CONFIRMATIONS)
        first = np.full(len(oriented), -1, dtype=np.int16)
        for task in np.unique(external_task):
            take = np.flatnonzero(external_task == task)
            peaks = []
            for key, (cache, repr_map) in reference.items():
                rows = np.flatnonzero(reference_task[key] == task)
                if not len(rows):
                    continue
                reference_oriented = dev.trailing_mean(
                    repr_map[pick["representation"]][rows], WIDTH
                )
                if pick["direction"] == "low":
                    reference_oriented = -reference_oriented
                peaks.append(dev.row_max(reference_oriented))
            threshold = dev.quantile_higher(np.concatenate(peaks), pick["quantile"])
            trigger = (
                np.isfinite(persistent[take])
                & (persistent[take] > threshold)
                & external_graph["valid"][take]
            )
            first[take] = dev.first_query(trigger)
        external_rows.append(
            {
                **pick,
                **score_candidate(
                    first,
                    external_risk,
                    prior_of(first, external_suite, external_priors),
                ),
            }
        )

    pd.DataFrame(development_rows).to_csv(
        args.output / "development_flow_candidates.csv", index=False
    )
    external = pd.DataFrame(external_rows)
    external.to_csv(args.output / "external_flow.csv", index=False)
    (args.output / "comparison.json").write_text(
        json.dumps(
            {
                "schema": "himoe.hb_front_back.flow_early.v1",
                "metrics": list(FLOW_METRICS),
                "lock_parameters": {"width": WIDTH, "confirmations": CONFIRMATIONS},
                "selection_rule": {
                    "timely_fpr_at_most": MAX_TIMELY_FPR,
                    "low_prior_precision_at_least": MIN_LOW_PRIOR_PRECISION,
                    "ranking": ["low_prior_tp", "low_prior_precision"],
                },
                "low_prior_threshold": LOW_PRIOR,
                "mobility_reference": "L2 low W4 K4 q0.70 gives 72 early TP / 43 early FP",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== external replay, best per flow metric ===")
    print(
        external[
            [
                "metric",
                "representation",
                "direction",
                "quantile",
                "dev_low_prior_tp",
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
    print("\nmobility baseline (L2 low W4 K4 q0.70): 72 early TP / 43 early FP, 272/57 total")


if __name__ == "__main__":
    main()
