#!/usr/bin/env python3
"""POST-HOC probe: how do the frozen rules sit next to the v7 intrinsic guard?

The v7 guard is a task-agnostic operating point built in another bundle from
different routing quantities (freeze / acceleration / periodicity /
turbulence). It was not in the predeclared detector pool and nothing here was
selected against it. Everything this script prints is exploratory and is
labelled post-hoc in the report; it exists so the headline claim is scoped
honestly and so the next step is grounded in numbers rather than in a hunch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import combination_core as core


DEFAULT_OUTPUT = core.BUNDLE / "results"
V7 = core.PROJECT / "moe-v7-0905/results/intrinsic_guard_v7/sealed_first_alarms.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def merge_or(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.where(
        left < 0, right, np.where(right < 0, left, np.minimum(left, right))
    ).astype(np.int16)


def main() -> None:
    args = parse_args()
    external = core.Cohort("external_8b")
    with np.load(V7, allow_pickle=False) as archive:
        guard = np.asarray(archive["external_guard"], dtype=np.int16)
    published = {"tp": 439, "fp": 80}
    observed = external.score(guard)
    if (observed["tp"], observed["fp"]) != (published["tp"], published["fp"]):
        raise AssertionError(
            f"v7 guard does not reproduce its published external score: {observed}"
        )

    frozen = {
        "global_or_at_0.99": {
            "family": "quorum", "mode": "global", "pool": "all12",
            "level": "0.99", "level_loose": "0.99", "k": 1, "window": 0, "frames": 1,
        },
        "global_cascade_0.99_0.85_k2_f2": {
            "family": "cascade", "mode": "global", "pool": "all12",
            "level": "0.99", "level_loose": "0.85", "k": 2, "window": -1, "frames": 2,
        },
        "global_cascade_0.98_0.90_k2": {
            "family": "cascade", "mode": "global", "pool": "all12",
            "level": "0.98", "level_loose": "0.9", "k": 2, "window": -1, "frames": 1,
        },
    }
    _, firsts = core.evaluate_rules(external, list(frozen.values()))
    alarms = dict(zip(frozen, firsts.values()))

    rows = [{"rule": "v7_intrinsic_guard", "status": "existing", **external.score(guard)}]
    for name, first in alarms.items():
        rows.append({"rule": name, "status": "frozen_this_bundle", **external.score(first)})
        rows.append(
            {
                "rule": f"v7_guard OR {name}",
                "status": "post_hoc_exploratory",
                **external.score(merge_or(guard, first)),
            }
        )
        rows.append(
            {
                "rule": f"v7_guard AND {name}",
                "status": "post_hoc_exploratory",
                **external.score(core.pairwise_and(guard, first)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "posthoc_v7_probe.csv", index=False)

    fired_guard = guard >= 0
    coverage = {}
    for name, first in alarms.items():
        fired = first >= 0
        coverage[name] = {
            "risks_only_this_rule": int((fired & ~fired_guard & external.risk).sum()),
            "risks_only_v7_guard": int((~fired & fired_guard & external.risk).sum()),
            "risks_both": int((fired & fired_guard & external.risk).sum()),
            "risks_neither": int((~fired & ~fired_guard & external.risk).sum()),
            "timely_only_this_rule": int((fired & ~fired_guard & ~external.risk).sum()),
            "timely_only_v7_guard": int((~fired & fired_guard & ~external.risk).sum()),
            "timely_both": int((fired & fired_guard & ~external.risk).sum()),
        }
    (args.output / "posthoc_v7_probe.json").write_text(
        json.dumps(
            {
                "schema": "himoe.combination_rules.posthoc_v7.v1",
                "status": "post_hoc_exploratory, nothing selected against these numbers",
                "v7_guard_external_reproduced": observed,
                "overlap": coverage,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pd.set_option("display.width", 220)
    print(
        table[
            ["rule", "status", "tp", "fp", "precision", "risk_recall", "timely_fpr",
             "low_prior_tp", "low_prior_fp", "mean_alarm_prior", "lift"]
        ].to_string(index=False, float_format="%.4f")
    )
    print("\n" + json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
