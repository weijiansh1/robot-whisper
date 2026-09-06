#!/usr/bin/env python3
"""Offline alarm simulation: precision and recall as a function of how much
intervention budget the alarm must leave.

Every risk episode runs the full horizon cap by construction -- risk is defined
as failing to finish before it. So for a risk episode the chunks remaining at
the alarm are H - 1 - q, where H is the deployment timeout and q the alarm
chunk. Both are known online: H is a constant the operator sets, q is elapsed
time. Requiring a minimum remaining budget is therefore a deployable gate, not
a post-hoc filter.

Two gates are reported because they differ on the negatives:

  budget    H - 1 - q >= B. Deployable. For a timely success that finished at
            length < H, this still counts the alarm, because at the moment it
            fired the monitor had no way to know the episode was about to
            succeed. The false positive is real.

  phase     length - 1 - q >= phi * (length - 1). Analysis only: it uses the
            episode's realised length, which is not available online. Reported
            so suites with different horizons can be compared on equal terms,
            since a fixed chunk budget means very different things at H = 22
            and H = 52.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BUNDLE / "method"))

import loso_folds  # noqa: E402
from evaluate_intrinsic_guard_v7 import (  # noqa: E402
    EXTERNAL_LAYER,
    LABEL_ROOT,
    aligned_labels,
    load_npz,
    plain,
    sha256,
)


DEFAULT_OUTPUT = BUNDLE / "results/alarm_budget"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"
LOSO = BUNDLE / "results/loso_validation"
BUDGETS = tuple(range(0, 25))
PHASES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def rates(
    fired: np.ndarray, risk: np.ndarray, lead: np.ndarray
) -> dict[str, float | int]:
    tp = int((fired & risk).sum())
    fp = int((fired & ~risk).sum())
    detected_lead = lead[fired & risk]
    return {
        "tp": tp,
        "fp": fp,
        "risk_n": int(risk.sum()),
        "timely_n": int((~risk).sum()),
        "recall": tp / int(risk.sum()) if risk.any() else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "timely_fpr": fp / int((~risk).sum()) if (~risk).any() else float("nan"),
        "median_lead_chunks": float(np.median(detected_lead)) if tp else float("nan"),
    }


def sweep(
    labels: pd.DataFrame,
    first: np.ndarray,
    horizon: np.ndarray,
    detector: str,
) -> list[dict[str, Any]]:
    alarm = first >= 0
    risk = labels["original_failure"].to_numpy(bool)
    length = labels["length"].to_numpy(int)
    suite = labels["suite"].to_numpy(str)
    budget_left = horizon - 1 - first
    realised_lead = length - 1 - first
    phase_left = np.divide(
        realised_lead,
        np.maximum(length - 1, 1),
        out=np.zeros(len(first), dtype=float),
        where=True,
    )

    rows: list[dict[str, Any]] = []
    groups = [("all", np.ones(len(first), bool))]
    groups += [(name, suite == name) for name in loso_folds.SUITES]
    for group, take in groups:
        for budget in BUDGETS:
            fired = alarm & (budget_left >= budget) & take
            rows.append(
                {
                    "detector": detector,
                    "gate": "budget",
                    "group": group,
                    "threshold": budget,
                    **rates(fired[take], risk[take], realised_lead[take]),
                }
            )
        for phase in PHASES:
            fired = alarm & (phase_left >= phase) & take
            rows.append(
                {
                    "detector": detector,
                    "gate": "phase",
                    "group": group,
                    "threshold": phase,
                    **rates(fired[take], risk[take], realised_lead[take]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    external_layer = load_npz(EXTERNAL_LAYER)
    labels = aligned_labels(
        external_layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    suite = loso_folds.suite_of(external_layer)
    labels["suite"] = suite

    # The deployment timeout per suite is the observed horizon cap. Verified
    # rather than assumed: every risk episode must sit exactly on it.
    horizon_by_suite = {
        name: int(labels.loc[suite == name, "length"].max())
        for name in loso_folds.SUITES
    }
    risk = labels["original_failure"].to_numpy(bool)
    for name, cap in horizon_by_suite.items():
        block = (suite == name) & risk
        observed = labels.loc[block, "length"].to_numpy(int)
        if not (observed == cap).all():
            raise ValueError(
                f"{name}: {int((observed != cap).sum())} risk episodes do not "
                f"sit on the horizon cap {cap}"
            )
    horizon = np.asarray([horizon_by_suite[name] for name in suite], dtype=int)

    detectors: dict[str, np.ndarray] = {}
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        detectors["published"] = np.asarray(sealed["external_guard"], dtype=int)
    loso_path = LOSO / "fold_first_alarms.npz"
    if loso_path.exists():
        with np.load(loso_path, allow_pickle=False) as fold:
            pooled = np.full(len(suite), -1, dtype=int)
            covered = np.zeros(len(suite), bool)
            for name in loso_folds.SUITES:
                key = f"{name}__loso_l2__guard"
                if key not in fold.files:
                    continue
                take = suite == name
                pooled[take] = np.asarray(fold[key], dtype=int)
                covered[take] = True
            if covered.all():
                detectors["loso_l2"] = pooled

    rows: list[dict[str, Any]] = []
    for detector, first in detectors.items():
        rows.extend(sweep(labels, first, horizon, detector))
    table = pd.DataFrame(rows)
    table.to_csv(output / "alarm_budget_sweep.csv", index=False)

    summary = {
        "schema": "himoe.intrinsic_guard_v7.alarm_budget.v1",
        "horizon_by_suite": horizon_by_suite,
        "horizon_cap_verified": True,
        "detectors": list(detectors),
        "budgets": list(BUDGETS),
        "phases": list(PHASES),
        "artifacts": {
            "sweep_sha256": sha256(output / "alarm_budget_sweep.csv"),
            "evaluator_sha256": sha256(Path(__file__)),
        },
    }
    (output / "alarm_budget_summary.json").write_text(
        json.dumps(plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    view = table[
        (table["detector"] == "published")
        & (table["gate"] == "budget")
        & (table["group"] == "all")
        & (table["threshold"].isin([0, 2, 4, 6, 8, 10, 12, 16, 20]))
    ]
    print("published, budget gate, pooled:", flush=True)
    print(
        view[
            ["threshold", "tp", "fp", "recall", "precision", "timely_fpr", "median_lead_chunks"]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
