#!/usr/bin/env python3
"""Recall against the alarm deadline: the curve the 80%-at-phase-65% target sits on.

Same alarm rule and same budget as `reachability.py`, but swept over every chunk
up to the horizon cap instead of stopping at the window edge, so the question
"if 80% is not reachable by phase 65%, where does it become reachable" has an
answer rather than an extrapolation.

For each suite the frontier is

    best recall at timely FPR <= 0.005, over all (quantity, layer, direction,
    alpha) and over all chunks q <= deadline

reported for the real channels, for the null controls (the selection floor of
the same sweep), and against the closed-form survival baseline.  Detectors are
allowed to fire at any single chunk at or before the deadline; the FPR is
re-measured for the union so the extra chances are paid for.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from reachability import ALPHAS, FPR_BUDGET, chunk_crossings, tally

NULL_CONTROLS = (
    "ctrl_const_elapsed",
    "ctrl_episode_const_rand",
    "ctrl_episode_const_rand2",
    "ctrl_flow_noise_seed",
    "ctrl_white_noise",
)
DEADLINE_PHASES = (0.30, 0.40, 0.50, 0.65, 0.80, 1.00)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = {c: C.build_cohort(c) for c in ("development_main", "external_8b")}
    blocks = {c: C.flatten(f["values"]) for c, f in frames.items()}
    names, layers = C.column_index(frames["development_main"]["quantities"])
    names, layers = np.asarray(names), np.asarray(layers)
    group = np.where(
        names == "leak_full_length", "length_leak",
        np.where(np.isin(names, list(NULL_CONTROLS)), "null_control", "routing"),
    )

    rows: list[dict] = []
    for suite in sorted(C.CAPS):
        cap = C.CAPS[suite]
        for cohort, frame in frames.items():
            rows_suite = np.flatnonzero(frame["suite"] == suite)
            length = frame["length"][rows_suite]
            _, task_code_all = np.unique(frame["task"][rows_suite], return_inverse=True)
            n_task = int(task_code_all.max()) + 1
            risk = frame["risk"][rows_suite]
            n_risk, n_safe = int(risk.sum()), int((~risk).sum())
            block = blocks[cohort][rows_suite]
            # Two readings per deadline.  `single` lets the detector pick one chunk
            # at or before the deadline; `union` lets it test at every chunk and
            # pays the accumulated false alarms.
            groups_ = ("routing", "null_control", "length_leak")
            best_single = {g: 0.0 for g in groups_}
            best_union = {g: 0.0 for g in groups_}
            running = np.zeros((len(ALPHAS), 2, len(rows_suite), block.shape[2]), bool)
            for q in range(C.SWEEP_START, min(cap, frame["n_chunk"])):
                alive = length > q
                if alive.sum() < C.MIN_STRATUM:
                    continue
                sel = np.flatnonzero(alive)
                cross = chunk_crossings(block, sel, q, task_code_all[sel], n_task)
                pad = np.zeros_like(running)
                pad[:, :, sel, :] = cross
                for label, fired in (("single", pad), ("union", running | pad)):
                    tp, fp = tally(fired, risk)
                    ok = (fp / max(n_safe, 1)) <= FPR_BUDGET
                    rec = np.where(ok, tp / max(n_risk, 1), -1.0)
                    target = best_single if label == "single" else best_union
                    for g in groups_:
                        target[g] = max(target[g], float(rec[:, :, group == g].max()), 0.0)
                running |= pad
                del pad
                sa = int((alive & ~risk).sum())
                ra = int((alive & risk).sum())
                rows.append(
                    {
                        "cohort": cohort, "suite": suite, "chunk": q,
                        "phase": (q + 1) / cap, "cap": cap,
                        "n_risk": n_risk, "n_safe": n_safe,
                        "safe_alive": sa, "risk_alive": ra,
                        "survival_prior": ra / max(int(alive.sum()), 1),
                        "frontier_routing": best_single["routing"],
                        "frontier_null_control": best_single["null_control"],
                        "frontier_length_leak": best_single["length_leak"],
                        "union_routing": best_union["routing"],
                        "union_null_control": best_union["null_control"],
                        "survival_baseline_recall": min(
                            1.0,
                            min(FPR_BUDGET * n_safe, sa) * (ra / max(sa, 1)) / max(n_risk, 1),
                        ),
                        "fpr_if_all_survivors_alarm": sa / max(n_safe, 1),
                    }
                )
            del running
            print(f"  {suite} {cohort} done", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(args.output / "deadline_frontier.csv", index=False, float_format="%.6g")

    summary: list[dict] = []
    for cohort in ("development_main", "external_8b"):
        for phase in DEADLINE_PHASES:
            picked = []
            for suite in sorted(C.CAPS):
                sub = table[
                    (table["cohort"] == cohort) & (table["suite"] == suite)
                    & (table["phase"] <= phase + 1e-9)
                ]
                if not len(sub):
                    continue
                last = sub.sort_values("chunk").iloc[-1]
                picked.append(
                    {
                        "suite": suite,
                        "n_risk": int(last["n_risk"]),
                        "routing": float(last["frontier_routing"]),
                        "null": float(last["frontier_null_control"]),
                        "leak": float(last["frontier_length_leak"]),
                        "union": float(last["union_routing"]),
                        "union_null": float(last["union_null_control"]),
                        "chunk": int(last["chunk"]),
                    }
                )
            if not picked:
                continue
            frame = pd.DataFrame(picked)
            summary.append(
                {
                    "cohort": cohort, "deadline_phase": phase,
                    "risks": int(frame["n_risk"].sum()),
                    "routing_recall": float(
                        (frame["routing"] * frame["n_risk"]).sum() / frame["n_risk"].sum()
                    ),
                    "null_recall": float(
                        (frame["null"] * frame["n_risk"]).sum() / frame["n_risk"].sum()
                    ),
                    "leak_recall": float(
                        (frame["leak"] * frame["n_risk"]).sum() / frame["n_risk"].sum()
                    ),
                    "union_recall": float(
                        (frame["union"] * frame["n_risk"]).sum() / frame["n_risk"].sum()
                    ),
                    "union_null_recall": float(
                        (frame["union_null"] * frame["n_risk"]).sum() / frame["n_risk"].sum()
                    ),
                    **{f"routing_{r['suite']}": r["routing"] for _, r in frame.iterrows()},
                }
            )
    out = pd.DataFrame(summary)
    out.to_csv(args.output / "deadline_frontier_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
