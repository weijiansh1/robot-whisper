#!/usr/bin/env python3
"""Reconcile with the two earlier replay bundles instead of duplicating them.

VLA_MUI_HUB/moe-failure-alarm/ already replayed 36,098 trajectories chunk by
chunk (574,717 queries) with a discrete Top-4-churn alarm, and reported a
`length_only_same_run` baseline that dominates it. Its scope is every run in the
corpus including `pin-*`, its positives are 1,403 physical failures, and its
false-alarm rate is calibrated per query. Ours is 564 original-horizon risks on
`external_8b` alone with a per-episode timely-FPR budget. The denominators are
different and the two are never pooled.

What this script does:

  1. joins their per-episode table to our row order on (run_id, task_name,
     episode_index) and checks that the episode lengths agree;
  2. converts their global query indices to within-episode chunk indices and
     re-scores their alarm and their length baseline on *our* labels, so the
     two live in the same ledger;
  3. rebuilds `length_only` inside our cohort at several calibrated false-alarm
     targets, which is the baseline every MoE method has to be read against;
  4. cross-checks our per-mode recall against their `reason_metrics.csv`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ledger_core import PROJECT, Cohort, Method  # noqa: E402

HUB = PROJECT / "VLA_MUI_HUB"
ALARM = HUB / "moe-failure-alarm/results"
DYNAMICS = HUB / "moe-physical-failure-dynamics/results"
RESULTS = Path(__file__).resolve().parent.parent / "results"

TARGET_FPRS = (0.005, 0.01, 0.05)


def length_only_methods(cohort: Cohort, grouping: str, targets=TARGET_FPRS) -> list[Method]:
    """`length_only`, rebuilt inside our cohort.

    Calibration follows moe-failure-alarm.analyze.length_threshold: the alarm
    chunk is the smallest q at which at most `target` of the reference successes
    are still running. Unlike every MoE head in this ledger, the calibration set
    is defined by outcome (successes only), so this baseline is not
    outcome-blind; that is recorded rather than hidden.
    """
    key = cohort.suite if grouping == "suite" else cohort.task
    methods = []
    for target in targets:
        first = np.full(cohort.n, -1, dtype=np.int16)
        for group in np.unique(key):
            take = np.flatnonzero(key == group)
            timely = take[~cohort.risk[take]]
            max_query = cohort.length[timely] - 1
            threshold = None
            for q in range(int(max_query.max()) + 2):
                if float(np.mean(max_query >= q)) <= target:
                    threshold = q
                    break
            assert threshold is not None
            fires = take[cohort.length[take] - 1 >= threshold]
            first[fires] = threshold
        methods.append(
            Method(
                name=f"baseline:length_only_{grouping}@fpr{target:g}",
                family="length_only_baseline",
                threshold_mode="per_task" if grouping == "task" else "global",
                runtime_task_identity=grouping == "task",
                source="rebuilt here, following VLA_MUI_HUB/moe-failure-alarm/"
                "analyze.py::length_threshold",
                config=f"alarm at the smallest chunk q where <= {target:g} of the "
                f"same-{grouping} timely successes are still running "
                "(calibration set defined by outcome)",
                first=first,
                published=None,
            )
        )
    return methods


def transferred_length_only(cohort: Cohort, targets=(0.005, 0.01)) -> list[Method]:
    """`length_only`, calibrated on development and applied without refitting.

    The in-cohort version above fits its threshold on the same episodes it is
    then scored on, which is exactly the advantage every MoE head in this ledger
    is denied. This variant calibrates the per-task chunk threshold on
    development_main successes and transfers it. Tasks absent from development
    fall back to that suite's development threshold, and how many rows that
    covers is recorded on the method.
    """
    if cohort.name != "external_8b":
        return []
    reference = Cohort("development_main")
    methods = []
    for target in targets:
        def threshold_for(lengths: np.ndarray) -> int:
            max_query = lengths - 1
            for q in range(int(max_query.max()) + 2):
                if float(np.mean(max_query >= q)) <= target:
                    return q
            raise AssertionError("unreachable")

        by_task = {
            str(task): threshold_for(
                reference.length[(reference.task == task) & ~reference.risk]
            )
            for task in np.unique(reference.task)
        }
        by_suite = {
            str(suite): threshold_for(
                reference.length[(reference.suite == suite) & ~reference.risk]
            )
            for suite in np.unique(reference.suite)
        }
        first = np.full(cohort.n, -1, dtype=np.int16)
        fallback = 0
        for task in np.unique(cohort.task):
            take = np.flatnonzero(cohort.task == task)
            if task in by_task:
                threshold = by_task[task]
            else:
                threshold = by_suite[task.split("/", 1)[0]]
                fallback += len(take)
            fires = take[cohort.length[take] - 1 >= threshold]
            first[fires] = min(threshold, cohort.queries - 1)
        methods.append(
            Method(
                name=f"baseline:length_only_task_devcal@fpr{target:g}",
                family="length_only_baseline",
                threshold_mode="per_task",
                runtime_task_identity=True,
                source="rebuilt here; per-task chunk threshold fitted on "
                "development_main successes and transferred unchanged",
                config=f"alarm at the smallest chunk q where <= {target:g} of the "
                f"same-task development successes are still running; "
                f"{fallback} external rows fell back to a suite-level threshold",
                first=first,
                published=None,
            )
        )
    return methods


def join_prior_alarm(cohort: Cohort) -> tuple[list[Method], dict]:
    table = pd.read_csv(ALARM / "episode_alarm_results.csv")
    ours = pd.DataFrame(
        {
            "task_name": [t.split("/", 1)[1] for t in cohort.task],
            "episode_index": cohort.episode,
            "our_length": cohort.length,
            "row": np.arange(cohort.n),
        }
    )
    run_id = {
        "external_8b": "right-50x8b-20260903",
        "development_main": "right-50x8-20260903",
    }[cohort.name]
    theirs = table[table["run_id"] == run_id].copy()
    merged = ours.merge(
        theirs, on=["task_name", "episode_index"], how="left", validate="one_to_one"
    ).sort_values("row")

    report = {
        "their_run_id": run_id,
        "their_rows_for_this_run": int(len(theirs)),
        "our_rows": int(cohort.n),
        "joined": int(merged["inference_calls"].notna().sum()),
        "unjoined": int(merged["inference_calls"].isna().sum()),
    }
    if report["unjoined"] > 0:
        report["verdict"] = (
            "not row-reconcilable; their table is treated as an external reference"
        )
        return [], report

    length_agree = int(
        (merged["inference_calls"].to_numpy(int) == cohort.length).sum()
    )
    report["episode_length_agrees"] = length_agree
    report["episode_length_disagrees"] = int(cohort.n - length_agree)
    # their success flag against our risk definition: these are different labels
    their_fail = ~merged["success"].to_numpy(bool)
    report["their_failures_in_this_run"] = int(their_fail.sum())
    report["our_risks"] = int(cohort.risk.sum())
    report["their_failure_and_our_risk"] = int((their_fail & cohort.risk).sum())
    report["their_failure_not_our_risk"] = int((their_fail & ~cohort.risk).sum())
    report["our_risk_not_their_failure"] = int((~their_fail & cohort.risk).sum())

    if report["episode_length_disagrees"] > 0:
        report["verdict"] = (
            "join succeeds on identity but episode lengths disagree; "
            "chunk indices are not comparable"
        )
        return [], report
    report["verdict"] = "row-reconcilable; their alarms re-scored on our labels"

    # analyze.py writes both alarm columns as within-episode chunk indices
    # (`first_alarm` on the episode slice, and `duration_threshold` itself), not
    # as offsets into the global query stream. Verified: no value reaches the
    # episode's own inference_calls.
    methods = []
    supported = merged["supported_alarm_queries"].to_numpy(int)
    for detector, prefix in (("HB_MoE_top4churn", "moe"), ("length_only_same_run", "length")):
        for fpr in (1, 5, 10):
            raw = merged[f"{prefix}_alarm_query_fpr{fpr}"].to_numpy(int)
            first = np.where(raw >= 0, raw, -1)
            outside = int(((first >= 0) & (first >= cohort.length)).sum())
            assert outside == 0, "an alarm chunk falls outside its own episode"
            first = first.astype(np.int16)
            methods.append(
                Method(
                    name=f"prior:{detector}@qfpr{fpr}pct",
                    family="prior_replay/moe-failure-alarm",
                    threshold_mode="task_agnostic"
                    if prefix == "moe"
                    else "global",
                    runtime_task_identity=False,
                    source="VLA_MUI_HUB/moe-failure-alarm/results/"
                    "episode_alarm_results.csv (joined on task_name+episode_index)",
                    config=f"their {detector} at a {fpr}% per-query calibrated "
                    "false-alarm target, re-scored against our 564-risk labels; "
                    "episode-supported chunks only: median "
                    f"{float(np.median(supported)):.0f} scorable chunks per episode",
                    first=first,
                    published=None,
                )
            )
    return methods, report


def cross_check_modes(cohort: Cohort, detail: pd.DataFrame) -> pd.DataFrame:
    reasons = pd.read_csv(ALARM / "reason_metrics.csv")
    short = {
        "object_released_or_dropped_before_goal": "dropped",
        "stable_grasp_not_observed": "no_grasp",
        "object_moved_but_goal_unmet": "moved_unmet",
        "goal_predicate_regressed": "regressed",
        "timeout_while_holding_target": "timeout_holding",
        "object_released_outside_goal": "released_outside",
        "approached_target_without_observed_contact": "no_contact",
        "mechanism_threshold_not_reached": "mechanism",
        "no_meaningful_target_progress": "no_progress",
    }
    rows = []
    for _, r in reasons.iterrows():
        mode = short[r["failure_reason"]]
        our_n = int(((cohort.physical_mode == mode) & cohort.risk).sum())
        rows.append(
            {
                "mode": mode,
                "their_n_all_runs": int(r["failures"]),
                "their_recall": float(r["recall"]),
                "their_median_alarm_phase": float(r["median_alarm_phase"]),
                "their_median_remaining_queries": float(r["median_remaining_queries"]),
                "our_n_external_8b": our_n,
                "denominators_differ": True,
            }
        )
    frame = pd.DataFrame(rows)
    for method in (
        "v7_guard",
        "mobility|global",
        "expert_load_effective_rank|per_task",
        "combo:P-C",
        "twotier:WATCH_a3",
    ):
        block = detail[detail["method"] == method]
        if len(block) != 1:
            continue
        frame[f"our_recall__{method}"] = [
            float(block.iloc[0][f"mode_{m}_recall"]) for m in frame["mode"]
        ]
    return frame


def main() -> None:
    cohort = Cohort("external_8b")
    methods, join_report = join_prior_alarm(cohort)
    extra = length_only_methods(cohort, "suite") + length_only_methods(cohort, "task")
    print(json.dumps(join_report, indent=1), flush=True)
    for m in methods + extra:
        fired = m.first >= 0
        tp = int((fired & cohort.risk).sum())
        fp = int((fired & ~cohort.risk).sum())
        prior = cohort.prior_table[cohort.suite_id[fired], m.first[fired]]
        print(
            f"  {m.name:52s} tp={tp:4d} fp={fp:5d} "
            f"prec={tp / max(tp + fp, 1):.3f} "
            f"median_chunk={np.median(m.first[fired]) if fired.any() else float('nan'):.0f} "
            f"mean_prior={np.nanmean(prior) if fired.any() else float('nan'):.3f}",
            flush=True,
        )
    (RESULTS / "prior_replay_join.json").write_text(json.dumps(join_report, indent=1))


if __name__ == "__main__":
    main()
