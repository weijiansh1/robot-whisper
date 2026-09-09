#!/usr/bin/env python3
"""Analysis only: which physical failure modes the in-window ceiling detector reaches.

Not part of the verdict.  It answers the follow-up question a negative result
raises -- if only ~57% of risks are reachable in the window, is the reachable
half a distinct kind of failure?  The alarm replayed here is the per-suite
in-window operating point chosen by `budget_allocation.py` for the
`routing_single` arm; labels come from VLA_MUI_HUB/physical-failure-labels,
joined on (suite, task, init_state_id, flow_noise_seed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from reachability import ALPHAS, chunk_crossings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.output / "cohort_budget_detail.csv")
    detail = detail[detail["arm"] == "routing_single"]
    labels = pd.read_csv(
        C.PROJECT / "VLA_MUI_HUB/physical-failure-labels/results/episodes.csv"
    )
    labels["key"] = (
        labels["suite"] + "/" + labels["task_name"] + "|"
        + labels["init_state_id"].astype(str) + "|"
        + labels["flow_noise_seed"].astype(str)
    )
    reason = labels.drop_duplicates("key").set_index("key")["primary_failure_reason"]

    rows: list[dict] = []
    for cohort in ("development_main", "external_8b"):
        frame = C.build_cohort(cohort)
        block = C.flatten(frame["values"])
        names, layers = C.column_index(frame["quantities"])
        names, layers = np.asarray(names), np.asarray(layers)
        key = np.asarray(
            [
                f"{t}|{i}|{s}"
                for t, i, s in zip(
                    frame["task"], frame["init_state_id"], frame["flow_noise_seed"],
                    strict=True,
                )
            ]
        )
        mode = pd.Series(key).map(reason).fillna("unlabelled").to_numpy()
        for _, pick in detail[detail["cohort"] == cohort].iterrows():
            suite = pick["suite"]
            quantity, layer, direction = pick["head"].split("|")
            col = int(np.flatnonzero((names == quantity) & (layers == layer))[0])
            rows_suite = np.flatnonzero(frame["suite"] == suite)
            length = frame["length"][rows_suite]
            _, task_code = np.unique(frame["task"][rows_suite], return_inverse=True)
            n_task = int(task_code.max()) + 1
            risk = frame["risk"][rows_suite]
            q = int(pick["chunk"])
            alive = np.flatnonzero(length > q)
            cross = chunk_crossings(
                block[rows_suite], alive, q, task_code[alive], n_task
            )
            a = int(np.argmin(np.abs(ALPHAS - float(pick["alpha"]))))
            d = 0 if direction == "high" else 1
            fired = np.zeros(len(rows_suite), bool)
            fired[alive] = cross[a, d, :, col]
            sub_mode = mode[rows_suite]
            for name in sorted(set(sub_mode[risk])):
                m = risk & (sub_mode == name)
                rows.append(
                    {
                        "cohort": cohort, "suite": suite, "chunk": q,
                        "phase": float(pick["phase"]), "head": pick["head"],
                        "failure_mode": name, "risks": int(m.sum()),
                        "detected": int((m & fired).sum()),
                        "recall": float((m & fired).sum() / max(int(m.sum()), 1)),
                    }
                )
        del frame, block

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "failure_mode_split.csv", index=False)
    pooled = (
        table.groupby(["cohort", "failure_mode"])[["risks", "detected"]].sum().reset_index()
    )
    pooled["recall"] = pooled["detected"] / pooled["risks"].clip(lower=1)
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
