#!/usr/bin/env python3
"""Deliverable 4: worked traces, and the within-task comparison every method
difference in the report has to be read against.

Traces are chosen to span outcomes and physical failure modes, to include a late
success, and to include an episode no method in the ledger alarmed on. The
selection is by outcome and mode coverage, not by how flattering the trace is.

The comparison side is task-stratified on purpose. An audit of this corpus found
that 1 of 264 effects survives task-stratified permutation while 52 survive
suite-stratified, and that suite stratification is anti-conservative here. So
every pairwise method difference is a paired difference within task, tested by
sign-flip permutation over the 39 tasks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_methods import collect  # noqa: E402
from ledger_core import HORIZON, Cohort  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
RNG = np.random.default_rng(20260906)
PERMUTATIONS = 20000

HEADLINE = [
    "baseline:length_only_task_devcal@fpr0.005",
    "v7_guard",
    "v4_dual_regime",
    "combo:P-C",
    "combo:G-A",
    "mobility|global",
    "expert_load_effective_rank|per_task",
    "twotier:WATCH_a3",
    "twotier:ACT_b",
    "prior:HB_MoE_top4churn@qfpr1pct",
]


def paired_task_test(
    cohort: Cohort, left: np.ndarray, right: np.ndarray, kind: str
) -> dict:
    """Per-task paired difference with a sign-flip permutation null."""
    tasks = np.unique(cohort.task)
    diffs = []
    for task in tasks:
        take = cohort.task == task
        if kind == "tp":
            a = int(((left >= 0) & cohort.risk & take).sum())
            b = int(((right >= 0) & cohort.risk & take).sum())
        elif kind == "fp":
            a = int(((left >= 0) & ~cohort.risk & take).sum())
            b = int(((right >= 0) & ~cohort.risk & take).sum())
        else:  # early tp, prior < 0.25
            prior_l = cohort.prior_table[cohort.suite_id, np.maximum(left, 0)]
            prior_r = cohort.prior_table[cohort.suite_id, np.maximum(right, 0)]
            a = int(((left >= 0) & (prior_l < 0.25) & cohort.risk & take).sum())
            b = int(((right >= 0) & (prior_r < 0.25) & cohort.risk & take).sum())
        diffs.append(a - b)
    diffs = np.asarray(diffs, float)
    observed = float(diffs.mean())
    signs = RNG.choice([-1.0, 1.0], size=(PERMUTATIONS, len(diffs)))
    null = (signs * diffs).mean(axis=1)
    p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    return {
        "metric": kind,
        "tasks": int(len(tasks)),
        "mean_per_task_difference": observed,
        "tasks_left_higher": int((diffs > 0).sum()),
        "tasks_right_higher": int((diffs < 0).sum()),
        "tasks_tied": int((diffs == 0).sum()),
        "permutation_p": p,
        "survives_task_stratified_permutation": bool(p < 0.05),
    }


def build_traces(cohort: Cohort, methods) -> list[dict]:
    names = [m.name for m in methods]
    stack = np.stack([m.first for m in methods], axis=0)
    any_alarm = (stack >= 0).any(axis=0)
    n_alarm = (stack >= 0).sum(axis=0)
    # "missed by everything" has to mean the routing detectors. The length-only
    # rows alarm on every risk by construction, so including them would make the
    # set empty and hide the case worth looking at.
    moe = np.asarray(
        [m.family not in ("length_only_baseline",)
         and not m.name.startswith("prior:length_only") for m in methods]
    )
    any_moe_alarm = (stack[moe] >= 0).any(axis=0)
    n_moe = int(moe.sum())

    chosen: list[int] = []

    def pick(mask: np.ndarray, label: str) -> None:
        rows = np.flatnonzero(mask)
        rows = np.asarray([r for r in rows if r not in chosen])
        if len(rows) == 0:
            return
        chosen.append(int(rows[0]))
        picks.append((int(rows[0]), label))

    picks: list[tuple[int, str]] = []
    n_moe_alarms = (stack[moe] >= 0).sum(axis=0)
    headline_rows = [
        i
        for i, m in enumerate(methods)
        if m.name in HEADLINE and m.family != "length_only_baseline"
    ]
    missed_by_headline = ~(stack[headline_rows] >= 0).any(axis=0)
    # 1. the risk the routing detectors came closest to missing entirely
    floor = int(n_moe_alarms[cohort.risk].min())
    pick(
        cohort.risk & (n_moe_alarms == floor),
        f"the risk the routing detectors nearly missed: {floor} of {n_moe} fired "
        f"(no risk is missed by all {n_moe}, but 5 draw a single alarm)",
    )
    # 2. a persistent failure every headline method missed
    pick(
        (cohort.outcome == "persistent_failure") & missed_by_headline,
        f"persistent failure missed by every one of the {len(headline_rows)} "
        "headline routing methods",
    )
    # 3. a late success that was alarmed on
    pick(
        (cohort.outcome == "late_success_plus10") & any_moe_alarm,
        "late success (+10 queries would have finished it), routing alarms fired",
    )
    # 4. the late success with the fewest routing alarms
    late_floor = int(
        n_moe_alarms[cohort.outcome == "late_success_plus10"].min()
    )
    pick(
        (cohort.outcome == "late_success_plus10") & (n_moe_alarms == late_floor),
        f"late success with the fewest routing alarms ({late_floor} of {n_moe})",
    )
    # 4-8. one persistent failure per major mode, preferring well-covered ones
    for mode in ("dropped", "no_grasp", "moved_unmet", "regressed", "timeout_holding",
                 "released_outside", "no_contact"):
        pick(
            (cohort.outcome == "persistent_failure")
            & (cohort.physical_mode == mode)
            & (n_alarm >= 5),
            f"persistent failure / {mode}",
        )
    # 9. the most-alarmed timely success: the worst false alarm in the corpus
    worst = int(np.argmax(np.where(~cohort.risk, n_alarm, -1)))
    if worst not in chosen:
        chosen.append(worst)
        picks.append((worst, "timely success that drew the most false alarms"))
    # 10. an early, correct alarm in the low-prior band
    prior_at = np.where(
        stack >= 0, cohort.prior_table[cohort.suite_id[None, :], np.maximum(stack, 0)], np.nan
    )
    early_correct = cohort.risk & (np.nanmin(np.where(np.isnan(prior_at), np.inf, prior_at), axis=0) < 0.12)
    pick(early_correct, "risk caught while the survival prior was still under 0.12")

    traces = []
    for row, label in picks:
        events = []
        for position, name in enumerate(names):
            q = int(stack[position, row])
            if q < 0:
                continue
            events.append(
                {
                    "method": name,
                    "chunk": q,
                    "survival_prior_at_alarm": round(
                        float(cohort.prior_table[cohort.suite_id[row], q]), 4
                    ),
                    "remaining_budget": int(cohort.horizon[row] - 1 - q),
                    "lead": int(cohort.length[row] - 1 - q),
                }
            )
        events.sort(key=lambda e: (e["chunk"], e["method"]))
        suite = str(cohort.suite[row])
        prior_curve = {
            str(c): round(float(cohort.priors[suite].get(c, float("nan"))), 4)
            for c in range(int(cohort.length[row]))
        }
        traces.append(
            {
                "selection_reason": label,
                "row": int(row),
                "suite": suite,
                "task": str(cohort.task[row]),
                "episode": int(cohort.episode[row]),
                "horizon_cap": int(cohort.horizon[row]),
                "realised_length": int(cohort.length[row]),
                "outcome": str(cohort.outcome[row]),
                "physical_mode": str(cohort.physical_mode[row]) or None,
                "mode_confidence": str(cohort.mode_confidence[row]) or None,
                "methods_that_alarmed": len(events),
                "routing_detectors_that_alarmed": int(
                    (stack[moe, row] >= 0).sum()
                ),
                "routing_detectors_in_ledger": n_moe,
                "methods_in_ledger": len(names),
                "first_alarm_chunk": events[0]["chunk"] if events else None,
                "survival_prior_by_chunk": prior_curve,
                "alarms": events,
            }
        )
    return traces


def main() -> None:
    cohort = Cohort("external_8b")
    methods, _, _, _ = collect(cohort)
    lookup = {m.name: m for m in methods}

    traces = build_traces(cohort, methods)
    (RESULTS / "worked_traces.json").write_text(json.dumps(traces, indent=1))
    for t in traces:
        print(
            f"{t['selection_reason']:58s} {t['suite']:15s} ep{t['episode']:<4d} "
            f"len {t['realised_length']:2d}/{t['horizon_cap']:2d} "
            f"{t['outcome']:20s} {str(t['physical_mode']):18s} "
            f"alarms {t['methods_that_alarmed']:2d}/{t['methods_in_ledger']}",
            flush=True,
        )

    # ---- task-stratified pairwise comparison -------------------------------
    rows = []
    reference = "baseline:length_only_task_devcal@fpr0.005"
    pairs = [(reference, other) for other in HEADLINE if other != reference]
    pairs += [
        ("v7_guard", "v4_dual_regime"),
        ("combo:P-C", "expert_load_effective_rank|per_task"),
        ("combo:G-A", "mobility|global"),
        ("twotier:WATCH_a3", "twotier:WATCH_a1"),
        ("mobility|global", "mobility|per_task"),
        ("prior:HB_MoE_top4churn@qfpr1pct", "mobility|global"),
    ]
    for left, right in pairs:
        if left not in lookup or right not in lookup:
            continue
        for kind in ("tp", "fp", "early_tp"):
            result = paired_task_test(
                cohort, lookup[left].first, lookup[right].first, kind
            )
            rows.append({"left": left, "right": right, **result})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "task_stratified_comparison.csv", index=False)
    print()
    print(
        frame[
            [
                "left",
                "right",
                "metric",
                "mean_per_task_difference",
                "tasks_left_higher",
                "tasks_right_higher",
                "tasks_tied",
                "permutation_p",
            ]
        ].to_string(index=False),
        flush=True,
    )

    # ---- how often does the pooled ranking hold up inside each task? --------
    by_task = pd.read_csv(RESULTS / "method_by_task_external.csv")
    consistency = []
    for left, right in pairs:
        a = by_task[by_task["method"] == left].set_index("task")
        b = by_task[by_task["method"] == right].set_index("task")
        if a.empty or b.empty:
            continue
        common = a.index.intersection(b.index)
        d = a.loc[common, "tp"] - b.loc[common, "tp"]
        consistency.append(
            {
                "left": left,
                "right": right,
                "pooled_tp_left": int(a["tp"].sum()),
                "pooled_tp_right": int(b["tp"].sum()),
                "tasks_agreeing_with_pooled_sign": int(
                    (np.sign(d) == np.sign(a["tp"].sum() - b["tp"].sum())).sum()
                ),
                "tasks_opposing": int(
                    (np.sign(d) == -np.sign(a["tp"].sum() - b["tp"].sum())).sum()
                ),
                "tasks_tied": int((d == 0).sum()),
                "n_tasks": int(len(common)),
            }
        )
    pd.DataFrame(consistency).to_csv(
        RESULTS / "task_consistency.csv", index=False
    )
    print("\nwrote worked_traces.json, task_stratified_comparison.csv, task_consistency.csv")


if __name__ == "__main__":
    main()
