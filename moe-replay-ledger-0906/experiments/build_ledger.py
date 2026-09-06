#!/usr/bin/env python3
"""Deliverables 1-3: the per-alarm ledger, the chunk-by-chunk operator view and
the per-method detail sheet.

The ledger has two blocks and they are kept apart on purpose.

  knowable at alarm time  method, threshold_mode, suite, task, episode, the
                          alarm chunk q, the horizon cap H for the suite, the
                          remaining budget H-1-q, and the per-suite survival
                          prior at q.
  unmasked afterwards     outcome, realised length, physical failure mode,
                          whether the alarm was correct, and the lead.

Nothing in the first block is computed from an episode's own outcome. The
survival prior is the one aggregate that comes from outcomes; it is a per-suite,
per-chunk constant and it is the quantity the whole corpus uses as the "still
running" null.

Everything here describes observation. The logged rollouts ran with no monitor
in the loop, so no number in this file says what acting on an alarm would do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect_methods import collect  # noqa: E402
from ledger_core import HORIZON, LOW_PRIOR, PRIOR_CROSS_25, Cohort  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results"
BUDGETS = (0, 2, 4, 8, 12)
EARLIEST_CHUNK = 6
MODE_ORDER = (
    "dropped",
    "no_grasp",
    "moved_unmet",
    "regressed",
    "released_outside",
    "no_contact",
    "timeout_holding",
    "mechanism",
    "no_progress",
)


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 3
    return tuple(float(v) for v in np.quantile(values.astype(float), (0.25, 0.5, 0.75)))


def build_ledger(cohort: Cohort, methods) -> pd.DataFrame:
    index = cohort.index_frame()
    blocks = []
    for m in methods:
        fired = np.flatnonzero(m.first >= 0)
        if len(fired) == 0:
            continue
        q = m.first[fired].astype(int)
        block = index.iloc[fired].reset_index(drop=True)
        prior = cohort.prior_table[cohort.suite_id[fired], q]
        out = pd.DataFrame(
            {
                # ---------------- knowable at alarm time --------------------
                "method": m.name,
                "family": m.family,
                "threshold_mode": m.threshold_mode,
                "runtime_task_identity": m.runtime_task_identity,
                "suite": block["suite"],
                "task": block["task"],
                "episode": block["episode"],
                "alarm_chunk": q,
                "horizon_cap": block["horizon"],
                "remaining_budget": block["horizon"].to_numpy() - 1 - q,
                "survival_prior_at_alarm": prior,
                "early_band": prior < LOW_PRIOR,
                # ---------------- unmasked afterwards -----------------------
                "outcome": block["outcome"],
                "episode_length": block["length"],
                "physical_mode": block["physical_mode"],
                "mode_confidence": block["mode_confidence"],
                "correct": block["risk"],
                "lead": block["length"].to_numpy() - 1 - q,
            }
        )
        blocks.append(out)
    ledger = pd.concat(blocks, ignore_index=True)
    return ledger


def chunk_view(cohort: Cohort, methods) -> pd.DataFrame:
    rows = []
    running_by_suite = {}
    for suite in cohort.suites:
        take = cohort.suite == suite
        running_by_suite[suite] = np.asarray(
            [int((cohort.length[take] > c).sum()) for c in range(cohort.queries)]
        )
    running_pooled = np.asarray(
        [int((cohort.length > c).sum()) for c in range(cohort.queries)]
    )
    for m in methods:
        fired = m.first >= 0
        for suite in list(cohort.suites) + ["ALL"]:
            take = np.ones(cohort.n, bool) if suite == "ALL" else cohort.suite == suite
            running = running_pooled if suite == "ALL" else running_by_suite[suite]
            horizon = (
                max(HORIZON.values()) if suite == "ALL" else HORIZON[suite]
            )
            sub_first = m.first[take]
            sub_risk = cohort.risk[take]
            for chunk in range(horizon):
                at = sub_first == chunk
                n_alarm = int(at.sum())
                if n_alarm == 0 and running[chunk] == 0:
                    continue
                rows.append(
                    {
                        "method": m.name,
                        "threshold_mode": m.threshold_mode,
                        "suite": suite,
                        "chunk": chunk,
                        "episodes_still_running": int(running[chunk]),
                        "survival_prior": (
                            float("nan")
                            if suite == "ALL"
                            else cohort.priors[suite].get(chunk, float("nan"))
                        ),
                        "alarms_at_chunk": n_alarm,
                        "alarms_at_chunk_risk": int((at & sub_risk).sum()),
                        "alarms_at_chunk_timely": int((at & ~sub_risk).sum()),
                        "cumulative_alarms": int((sub_first >= 0).sum() and
                                                 ((sub_first >= 0) & (sub_first <= chunk)).sum()),
                        "cumulative_alarms_risk": int(
                            ((sub_first >= 0) & (sub_first <= chunk) & sub_risk).sum()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def detail_sheet(cohort: Cohort, methods, ledger: pd.DataFrame) -> pd.DataFrame:
    risk_total = int(cohort.risk.sum())
    timely_total = int((~cohort.risk).sum())
    mode_totals = {
        mode: int(((cohort.physical_mode == mode) & cohort.risk).sum())
        for mode in MODE_ORDER
    }
    persistent_mode_totals = {
        mode: int(
            (
                (cohort.physical_mode == mode)
                & (cohort.outcome == "persistent_failure")
            ).sum()
        )
        for mode in MODE_ORDER
    }
    # earliest structurally possible alarm chunk: trailing mean of width 4 plus
    # 4 confirmations over a mobility series whose first chunk is undefined
    ceiling = int((cohort.horizon[cohort.risk] - 1 - EARLIEST_CHUNK).sum())
    rows = []
    for m in methods:
        fired = m.first >= 0
        tp = int((fired & cohort.risk).sum())
        fp = int((fired & ~cohort.risk).sum())
        sub = ledger[ledger["method"] == m.name]
        prior = sub["survival_prior_at_alarm"].to_numpy(float)
        chunk = sub["alarm_chunk"].to_numpy(int)
        budget = sub["remaining_budget"].to_numpy(int)
        correct = sub["correct"].to_numpy(bool)
        mean_prior = float(np.nanmean(prior)) if len(prior) else float("nan")
        precision = tp / max(tp + fp, 1)

        row = {
            "method": m.name,
            "family": m.family,
            "threshold_mode": m.threshold_mode,
            "runtime_task_identity": m.runtime_task_identity,
            "config": m.config,
            "source": m.source,
            "alarms": tp + fp,
            "tp": tp,
            "fp": fp,
            "precision": precision,
            "recall": tp / risk_total,
            "timely_fpr": fp / timely_total,
            "mean_matched_prior": mean_prior,
            "lift": precision / mean_prior if mean_prior and np.isfinite(mean_prior) else float("nan"),
            "published_tp": m.published[0] if m.published else None,
            "published_fp": m.published[1] if m.published else None,
            "reproduces_published": (
                None if m.published is None else bool((tp, fp) == m.published)
            ),
        }
        # outcome split
        for outcome in ("timely_success", "late_success_plus10", "persistent_failure"):
            row[f"alarms_on_{outcome}"] = int((sub["outcome"] == outcome).sum())
        row["recall_persistent"] = (
            int((sub["outcome"] == "persistent_failure").sum())
            / max(int((cohort.outcome == "persistent_failure").sum()), 1)
        )
        row["recall_late"] = (
            int((sub["outcome"] == "late_success_plus10").sum())
            / max(int((cohort.outcome == "late_success_plus10").sum()), 1)
        )
        # failure-mode split, n alongside every rate
        for mode in MODE_ORDER:
            hit = int(((sub["physical_mode"] == mode)).sum())
            n = mode_totals[mode]
            row[f"mode_{mode}_n"] = n
            row[f"mode_{mode}_hit"] = hit
            row[f"mode_{mode}_recall"] = hit / n if n else float("nan")
            row[f"mode_{mode}_persistent_n"] = persistent_mode_totals[mode]
        scored = [m_ for m_ in MODE_ORDER if mode_totals[m_] >= 10]
        cov = [row[f"mode_{m_}_recall"] for m_ in scored]
        row["worst_mode_recall"] = float(np.min(cov)) if cov else float("nan")
        row["worst_mode_name"] = scored[int(np.argmin(cov))] if cov else ""
        row["mean_mode_recall"] = float(np.mean(cov)) if cov else float("nan")
        row["modes_scored_n"] = len(scored)
        # timing
        for label, values in (
            ("alarm_chunk", chunk),
            ("remaining_budget", budget),
            ("matched_prior", prior),
        ):
            q1, q2, q3 = quartiles(np.asarray(values))
            row[f"{label}_q25"], row[f"{label}_median"], row[f"{label}_q75"] = q1, q2, q3
        row["lead_median"] = float(np.median(sub.loc[correct, "lead"])) if correct.any() else float("nan")
        # early band
        early = sub["early_band"].to_numpy(bool)
        row["early_tp"] = int((early & correct).sum())
        row["early_fp"] = int((early & ~correct).sum())
        row["early_precision"] = (
            row["early_tp"] / max(row["early_tp"] + row["early_fp"], 1)
        )
        row["early_share_of_tp"] = row["early_tp"] / max(tp, 1)
        # budget view
        for b in BUDGETS:
            keep = budget >= b
            row[f"budget{b}_tp"] = int((keep & correct).sum())
            row[f"budget{b}_fp"] = int((keep & ~correct).sum())
            row[f"budget{b}_precision"] = (
                row[f"budget{b}_tp"] / max(row[f"budget{b}_tp"] + row[f"budget{b}_fp"], 1)
            )
        # recovered intervention budget. A true positive on an episode with cap H
        # alarming at chunk q buys H-1-q chunks of *potential* intervention. The
        # ceiling is every risk alarming at chunk 6, the earliest chunk at which
        # any head in this project can fire (trailing mean width 4 plus 4
        # confirmations, minus the shared first chunk). This weighting assumes the
        # value of remaining budget is linear in chunks, which is almost certainly
        # wrong: there is likely a floor below which an alarm is worthless, and the
        # real value function is set by the intervention mechanism, which is not
        # tested anywhere in this corpus. Read it as a better ordering tool than
        # raw counts, not as an absolute measure of value.
        budget_won = int(budget[correct].sum()) if correct.any() else 0
        budget_wasted = int(budget[~correct].sum()) if (~correct).any() else 0
        row["budget_won"] = budget_won
        row["budget_ceiling"] = ceiling
        row["frac_of_ceiling"] = budget_won / ceiling
        row["avg_budget_per_tp"] = budget_won / tp if tp else float("nan")
        row["budget_wasted_fp"] = budget_wasted
        row["gain_per_loss"] = (
            budget_won / budget_wasted if budget_wasted else float("inf")
        )
        # misses
        missed = (~fired) & cohort.risk
        row["missed_risks"] = int(missed.sum())
        for mode in MODE_ORDER:
            row[f"miss_{mode}"] = int((missed & (cohort.physical_mode == mode)).sum())
        row["missed_persistent"] = int(
            (missed & (cohort.outcome == "persistent_failure")).sum()
        )
        row["missed_late"] = int((missed & (cohort.outcome == "late_success_plus10")).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def per_suite_sheet(cohort: Cohort, methods) -> pd.DataFrame:
    rows = []
    for m in methods:
        fired = m.first >= 0
        for suite in cohort.suites:
            take = cohort.suite == suite
            tp = int((fired & cohort.risk & take).sum())
            fp = int((fired & ~cohort.risk & take).sum())
            prior = cohort.prior_table[
                cohort.suite_id[fired & take], m.first[fired & take]
            ]
            mean_prior = float(np.nanmean(prior)) if len(prior) else float("nan")
            prec = tp / max(tp + fp, 1)
            rows.append(
                {
                    "method": m.name,
                    "threshold_mode": m.threshold_mode,
                    "suite": suite,
                    "horizon_cap": HORIZON[suite],
                    "risks": int((cohort.risk & take).sum()),
                    "episodes": int(take.sum()),
                    "alarms": tp + fp,
                    "tp": tp,
                    "fp": fp,
                    "precision": prec,
                    "recall": tp / max(int((cohort.risk & take).sum()), 1),
                    "mean_matched_prior": mean_prior,
                    "lift": prec / mean_prior if mean_prior and np.isfinite(mean_prior) else float("nan"),
                    "median_alarm_chunk": float(np.median(m.first[fired & take]))
                    if (fired & take).any()
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def per_task_sheet(cohort: Cohort, methods) -> pd.DataFrame:
    """Within-task breakdown. Required wherever methods are compared."""
    rows = []
    for m in methods:
        fired = m.first >= 0
        for task in np.unique(cohort.task):
            take = cohort.task == task
            tp = int((fired & cohort.risk & take).sum())
            fp = int((fired & ~cohort.risk & take).sum())
            risks = int((cohort.risk & take).sum())
            prior = cohort.prior_table[
                cohort.suite_id[fired & take], m.first[fired & take]
            ]
            mean_prior = float(np.nanmean(prior)) if len(prior) else float("nan")
            prec = tp / max(tp + fp, 1)
            rows.append(
                {
                    "method": m.name,
                    "threshold_mode": m.threshold_mode,
                    "suite": task.split("/", 1)[0],
                    "task": task,
                    "episodes": int(take.sum()),
                    "risks": risks,
                    "alarms": tp + fp,
                    "tp": tp,
                    "fp": fp,
                    "precision": prec if tp + fp else float("nan"),
                    "recall": tp / risks if risks else float("nan"),
                    "mean_matched_prior": mean_prior,
                    "lift": prec / mean_prior
                    if (tp + fp) and mean_prior and np.isfinite(mean_prior)
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema": "himoe.replay_ledger.manifest.v1",
        "intervention_effect_measured": False,
        "monitor_in_the_loop_during_logging": False,
        "provenance": (
            "every rollout was logged with no monitor attached; the policy ran to "
            "success or to the horizon cap and nothing was ever interrupted. The "
            "ledger therefore measures detection only."
        ),
        "episode_definition": (
            "one complete logged rollout keyed by (task, initial state, flow noise "
            "seed); 50 initial states x 8 seeds = 400 per task. Routing data is real "
            "policy inference, one forward pass per chunk. The episode as a unit with "
            "a known length and outcome is a post-hoc construct: online there is only "
            "a rollout in progress with unknown remaining length, which is why "
            "`length` never enters any score."
        ),
        "risk_definition": "original_failure = did not finish before the horizon cap",
        "length_only_baseline_is_a_restatement_of_the_label": (
            "risk IS 'reached the horizon cap'. 568 external episodes reach their "
            "cap and 564 of them are risks, so a rule that reads only elapsed "
            "length has precision 0.9930 and recall 1.000 by construction. Its "
            "perfect recall measures the label definition, not the policy or the "
            "detector. It is carried in the sheet as a reference row and is not a "
            "target to beat. Length may legitimately enter only as a conditioning "
            "or weighting variable, which is exactly what the survival prior does."
        ),
        "earliest_structurally_possible_alarm_chunk": EARLIEST_CHUNK,
        "budget_metric_caveat": (
            "recovered intervention budget weights each true positive by H-1-q and "
            "assumes that value is linear in remaining chunks. That is almost "
            "certainly wrong: there is likely a floor below which an alarm is "
            "worthless, and the real value function is set by the intervention "
            "mechanism, which is untested in this corpus. Use it to order methods, "
            "not to price them."
        ),
        "horizon_caps": HORIZON,
        "prior_crosses_0.25_at_chunk": PRIOR_CROSS_25,
        "prior_work_reconciled": {
            "VLA_MUI_HUB/moe-failure-alarm": (
                "36,098 trajectories / 574,717 queries, chunk-by-chunk causal replay "
                "of a discrete Top-4-churn alarm plus a length_only_same_run "
                "baseline. Scope: every run including pin-*; positives are 1,403 "
                "physical failures; false alarms are calibrated per query. Ours: 564 "
                "original-horizon risks on external_8b with a per-episode timely-FPR "
                "budget. The denominators differ and the two are never pooled; their "
                "per-episode rows are joined to ours and re-scored on our labels."
            ),
            "VLA_MUI_HUB/moe-physical-failure-dynamics": (
                "matched same-run same-initial-state success controls for 1,306 of "
                "1,442 failures; used here only as the source of the physical "
                "failure-mode vocabulary, not re-analysed."
            ),
        },
        "cohorts": {},
        "discrepancies": [],
        "skipped": [],
        "notes": {},
    }

    for cohort_name in ("external_8b", "development_main"):
        print(f"=== {cohort_name}", flush=True)
        cohort = Cohort(cohort_name)
        methods, discrepancies, skipped, notes = collect(cohort)
        print(f"  methods: {len(methods)}", flush=True)

        ledger = build_ledger(cohort, methods)
        chunks = chunk_view(cohort, methods)
        detail = detail_sheet(cohort, methods, ledger)
        suites = per_suite_sheet(cohort, methods)
        tasks = per_task_sheet(cohort, methods)

        tag = "external" if cohort_name == "external_8b" else "development"
        ledger.to_csv(RESULTS / f"alarm_ledger_{tag}.csv", index=False)
        chunks.to_csv(RESULTS / f"chunk_view_{tag}.csv", index=False)
        detail.to_csv(RESULTS / f"method_detail_{tag}.csv", index=False)
        suites.to_csv(RESULTS / f"method_by_suite_{tag}.csv", index=False)
        tasks.to_csv(RESULTS / f"method_by_task_{tag}.csv", index=False)

        manifest["cohorts"][cohort_name] = {
            "episodes": cohort.n,
            "risks": int(cohort.risk.sum()),
            "persistent_failures": int((cohort.outcome == "persistent_failure").sum()),
            "late_successes": int((cohort.outcome == "late_success_plus10").sum()),
            "methods_included": [m.name for m in methods],
            "n_methods": len(methods),
            "ledger_rows": int(len(ledger)),
            "episodes_reaching_cap_but_succeeding": cohort.cap_but_success,
            "physical_mode_counts_all_risks": {
                mode: int(((cohort.physical_mode == mode) & cohort.risk).sum())
                for mode in MODE_ORDER
            },
            "physical_mode_counts_persistent_only": {
                mode: int(
                    (
                        (cohort.physical_mode == mode)
                        & (cohort.outcome == "persistent_failure")
                    ).sum()
                )
                for mode in MODE_ORDER
            },
            "mode_confidence_over_persistent": {
                c: int(
                    (
                        (cohort.mode_confidence == c)
                        & (cohort.outcome == "persistent_failure")
                    ).sum()
                )
                for c in ("high", "medium", "low")
            },
            "survival_prior_by_suite": {
                s: {str(k): v for k, v in cohort.priors[s].items()}
                for s in cohort.suites
            },
        }
        manifest["discrepancies"].extend(discrepancies)
        manifest["skipped"].extend(skipped)
        if notes:
            manifest["notes"][cohort_name] = notes

    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print("wrote results/manifest.json", flush=True)


if __name__ == "__main__":
    main()
