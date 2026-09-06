#!/usr/bin/env python3
"""How much of each layer rule's precision is detection, and how much is base rate?

Stage 1 of REPORT_ZH compares front / back / all-layer lock heads against the
frozen v4 rule. Those precision numbers are read here against the survival base
rate, because risk is defined as failing to finish before the horizon cap, so a
rollout that is still running is already evidence of risk.

    survival baseline: at chunk q, P(risk | still running at q), per suite

Every alarm is matched to the prior at its own alarm chunk. The ratio is what
the layer group contributed beyond knowing how long the rollout has run.

The rules are rebuilt here rather than read from a cache, so the script first
anchors its reproduction against the counts published in REPORT_ZH.
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

from analyze_front_back import (  # noqa: E402
    LABEL_PATHS,
    MOBILITY_PATHS,
    calibrated_external_alarm,
    load_npz,
    mobility_representations,
    profile_task_map,
)


DEFAULT_OUTPUT = BUNDLE / "results/survival_baseline"
LOCK = ("low", 4, 4, 0.75)
INSTABILITY = ("high", 4, 8, 0.80)
PRIOR_BANDS = ((0.0, 0.25), (0.25, 0.75), (0.75, 1.01))

# Counts published in REPORT_ZH for external_8b. The rebuild must match them
# before any of the decomposition below means anything.
PUBLISHED_COUNTS = {
    "front_lock": (240, 36),
    "back_lock": (409, 146),
    "all_lock": (382, 76),
    "L5_switching": (34, 5),
    "v4": (410, 81),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def build_external_alarms() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    main = load_npz(MOBILITY_PATHS["development_main"])
    extra = load_npz(MOBILITY_PATHS["development_extra"])
    external = load_npz(MOBILITY_PATHS["external_8b"])
    reference_map = profile_task_map([main, extra])
    reference_representations = {
        id(main): mobility_representations(main),
        id(extra): mobility_representations(extra),
    }
    external_repr = mobility_representations(external)

    def alarm(representation: str, head: tuple[str, int, int, float]) -> np.ndarray:
        direction, width, confirmations, quantile = head
        return calibrated_external_alarm(
            external,
            external_repr[representation],
            reference_map,
            reference_representations,
            representation,
            direction,
            width,
            confirmations,
            quantile,
        )

    front = alarm("front_median", LOCK)
    back = alarm("back_median", LOCK)
    every = alarm("all_median", LOCK)
    switching = alarm("L5", INSTABILITY)
    v4 = np.where(
        every < 0,
        switching,
        np.where(switching < 0, every, np.minimum(every, switching)),
    ).astype(np.int16)
    return external, {
        "front_lock": front,
        "back_lock": back,
        "all_lock": every,
        "L5_switching": switching,
        "v4": v4,
    }


def survival_prior(labels: pd.DataFrame) -> dict[str, dict[int, float]]:
    priors: dict[str, dict[int, float]] = {}
    for suite, block in labels.groupby("suite"):
        risk = block["original_failure"].to_numpy(bool)
        length = block["length"].to_numpy(int)
        priors[str(suite)] = {
            chunk: float(risk[length > chunk].mean())
            for chunk in range(int(length.max()))
        }
    return priors


def summarise(
    rule: str, fired: pd.DataFrame, risk_total: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def record(group: str, block: pd.DataFrame) -> None:
        if block.empty:
            return
        precision = float(block["correct"].mean())
        prior = float(block["prior"].mean())
        rows.append(
            {
                "rule": rule,
                "group": group,
                "alarms": int(len(block)),
                "tp": int(block["correct"].sum()),
                "fp": int((~block["correct"]).sum()),
                "recall": float(block["correct"].sum()) / risk_total
                if group == "all"
                else float("nan"),
                "precision": precision,
                "matched_prior": prior,
                "net_gain": precision - prior,
                "lift": precision / prior if prior > 0 else float("nan"),
            }
        )

    record("all", fired)
    for suite, block in fired.groupby("suite"):
        record(str(suite), block)
    for low, high in PRIOR_BANDS:
        record(
            f"prior[{low:.2f},{high:.2f})",
            fired[(fired["prior"] >= low) & (fired["prior"] < high)],
        )
    return rows


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    external, alarms = build_external_alarms()
    labels = pd.read_csv(LABEL_PATHS["external_8b"])
    task = external["task_names"].astype(str)[external["task_index"].astype(int)]
    labels["suite"] = pd.Series(task).str.split("/", n=1).str[0]
    labels["length"] = external["length"].astype(int)
    risk = labels["original_failure"].to_numpy(bool)

    for rule, first in alarms.items():
        fired = first >= 0
        counts = (int((fired & risk).sum()), int((fired & ~risk).sum()))
        if counts != PUBLISHED_COUNTS[rule]:
            raise ValueError(
                f"{rule}: rebuilt {counts} does not match published "
                f"{PUBLISHED_COUNTS[rule]}"
            )
    print("rebuild matches every published count in REPORT_ZH", flush=True)

    priors = survival_prior(labels)
    rows: list[dict[str, Any]] = []
    per_alarm: list[pd.DataFrame] = []
    for rule, first in alarms.items():
        block = labels.loc[first >= 0].copy()
        block["chunk"] = first[first >= 0]
        block["prior"] = [
            priors[suite][chunk]
            for suite, chunk in zip(block["suite"], block["chunk"], strict=True)
        ]
        block["correct"] = block["original_failure"].astype(bool)
        block["rule"] = rule
        per_alarm.append(block)
        rows.extend(summarise(rule, block, int(risk.sum())))

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "layer_survival_baseline.csv", index=False)
    pd.concat(per_alarm, ignore_index=True)[
        ["rule", "suite", "task", "episode", "chunk", "prior", "correct"]
    ].to_csv(args.output / "layer_alarm_priors.csv", index=False)
    (args.output / "layer_survival_summary.json").write_text(
        json.dumps(
            {
                "schema": "himoe.hb_front_back.survival_baseline.v1",
                "prior_definition": "P(risk | still running at chunk q), per suite",
                "prior_estimated_in_sample": True,
                "rebuild_anchored_to_report": True,
                "rows": table.to_dict(orient="records"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        table[table["group"] == "all"][
            ["rule", "alarms", "tp", "fp", "recall", "precision", "matched_prior", "lift"]
        ].to_string(index=False, float_format="%.3f")
    )


if __name__ == "__main__":
    main()
