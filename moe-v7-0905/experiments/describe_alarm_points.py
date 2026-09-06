#!/usr/bin/env python3
"""Descriptive statistics of the alarms themselves: where they fire and how
often they are right.

No gating and no post-hoc filtering. Every alarm the sealed detector raised is
counted exactly once, at the chunk where it fired.

The conditional precision table is usable online: when an alarm fires the
elapsed chunk count is known, so P(risk | first alarm at chunk q) can be looked
up at that moment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
)


DEFAULT_OUTPUT = BUNDLE / "results/alarm_points"
PUBLISHED = BUNDLE / "results/intrinsic_guard_v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    layer = load_npz(EXTERNAL_LAYER)
    labels = aligned_labels(
        layer, LABEL_ROOT / "external_8b_clean_labels.csv", "external_8b"
    )
    labels["suite"] = loso_folds.suite_of(layer)
    with np.load(PUBLISHED / "sealed_first_alarms.npz", allow_pickle=False) as sealed:
        labels["first"] = np.asarray(sealed["external_guard"], dtype=int)
    labels["risk"] = labels["original_failure"].astype(bool)

    fired = labels[labels["first"] >= 0].copy()
    fired["kind"] = np.where(fired["risk"], "true", "false")

    spread = (
        fired.groupby(["suite", "kind"])["first"]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])[
            ["count", "10%", "25%", "50%", "75%", "90%"]
        ]
        .round(1)
    )
    spread.to_csv(args.output / "alarm_point_spread.csv")
    print("=== 报警点（第几个 chunk）分布 ===")
    print(spread.to_string())

    rows = []
    for suite, block in fired.groupby("suite"):
        horizon = int(labels.loc[labels["suite"] == suite, "length"].max())
        for chunk, cell in block.groupby("first"):
            true = int(cell["risk"].sum())
            rows.append(
                {
                    "suite": suite,
                    "horizon": horizon,
                    "chunk": int(chunk),
                    "alarms": int(len(cell)),
                    "true": true,
                    "false": int(len(cell)) - true,
                    "precision": true / len(cell),
                }
            )
    per_chunk = pd.DataFrame(rows).sort_values(["suite", "chunk"])
    per_chunk.to_csv(args.output / "precision_by_alarm_chunk.csv", index=False)

    print()
    print("=== 每个 suite：按报警 chunk 分箱的准确率 ===")
    for suite, block in per_chunk.groupby("suite"):
        horizon = int(block["horizon"].iloc[0])
        edges = np.linspace(0, horizon, 6).astype(int)
        block = block.copy()
        block["bin"] = pd.cut(block["chunk"], bins=edges, include_lowest=True)
        grouped = block.groupby("bin", observed=True)[["alarms", "true", "false"]].sum()
        grouped["precision"] = grouped["true"] / grouped["alarms"]
        print(f"\n-- {suite} (horizon {horizon}) --")
        print(grouped.to_string(float_format="%.3f"))

    print()
    print("=== 全体报警的准确率（不分箱） ===")
    total = len(fired)
    true_total = int(fired["risk"].sum())
    print(
        f"报警 {total} 次，其中真风险 {true_total}，误报 {total - true_total}，"
        f"准确率 {true_total / total:.4f}"
    )
    print(
        f"风险 episode 共 {int(labels['risk'].sum())}，检出 {true_total}，"
        f"召回 {true_total / int(labels['risk'].sum()):.4f}"
    )


if __name__ == "__main__":
    main()
