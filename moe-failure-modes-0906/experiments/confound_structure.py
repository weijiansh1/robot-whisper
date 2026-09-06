#!/usr/bin/env python3
"""How much of the mode signal can survive at all: confounding structure, the
permutation freedom left inside each stratification, and the timing picture.

If mode were a deterministic function of task, the task-stratified test would be
vacuous rather than negative. This script measures that directly, so the
task-stratified nulls in mode_specialisation.py can be read honestly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"


def cramers_v(rows: np.ndarray, cols: np.ndarray) -> tuple[float, float]:
    table = pd.crosstab(rows, cols).to_numpy()
    n = table.sum()
    expected = table.sum(1)[:, None] * table.sum(0)[None, :] / n
    chi2 = float((((table - expected) ** 2) / np.where(expected > 0, expected, np.inf)).sum())
    v = float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))
    return v, chi2


def main() -> None:
    report = {"schema": "himoe.failure_modes_0906.confound_structure.v1"}
    timing_rows = []

    for cohort in ("development_main", "external_8b"):
        frame = C.cohort_index(cohort)
        suite = frame["suite"].to_numpy(str)
        risk = frame["risk"].to_numpy(bool)
        priors = C.survival_prior(suite, frame["length"].to_numpy(int), risk)
        risky = frame[risk]
        mode = risky["primary_failure_reason"].to_numpy(str)
        task = risky["task_key"].to_numpy(str)
        suite_r = risky["suite"].to_numpy(str)

        per_task_modes = pd.Series(mode).groupby(pd.Series(task)).nunique()
        counts = pd.Series(mode).groupby(pd.Series(task)).size()
        v_task, chi_task = cramers_v(task, mode)
        v_suite, chi_suite = cramers_v(suite_r, mode)

        cut = {}
        for s, table in priors.items():
            above = [c for c in sorted(table) if table[c] >= C.LOW_PRIOR]
            cut[s] = int(min(above)) if above else -1

        report[cohort] = {
            "risks": int(risk.sum()),
            "tasks_with_risk": int(len(counts)),
            "tasks_with_at_least_two_modes": int((per_task_modes >= 2).sum()),
            "risks_in_multi_mode_tasks": int(counts[per_task_modes >= 2].sum()),
            "cramers_v_task_mode": v_task,
            "cramers_v_suite_mode": v_suite,
            "chi2_task_mode": chi_task,
            "chi2_suite_mode": chi_suite,
            "risk_length_always_equals_cap": bool(
                (risky["length"].to_numpy(int)
                 == risky["suite"].map(C.HORIZON_CAP).to_numpy(int)).all()
            ),
            "prior_crosses_025_at_chunk": cut,
            "per_mode": {
                C.MODE_SHORT.get(m, m): {
                    "n": int((mode == m).sum()),
                    "tasks": int(len(set(task[mode == m]))),
                    "top_task_share": float(
                        pd.Series(task[mode == m]).value_counts().iloc[0] / (mode == m).sum()
                    ),
                    "episodes_in_single_mode_tasks": int(
                        sum(1 for t in task[mode == m] if per_task_modes.get(t, 1) < 2)
                    ),
                }
                for m in pd.Series(mode).value_counts().index
            },
        }

        key = "development" if cohort == "development_main" else "external"
        alarms = C.load_npz(OUT / f"first_alarms_{key}.npz")
        alarms.pop("schema", None)
        for name, first in sorted(alarms.items()):
            first_r = np.asarray(first, int)[risk]
            prior_at = C.prior_of(first_r, suite_r, priors)
            for m in pd.Series(mode).value_counts().index:
                take = mode == m
                fired = take & (first_r >= 0)
                timing_rows.append({
                    "cohort": cohort,
                    "detector": name,
                    "mode": m,
                    "mode_short": C.MODE_SHORT.get(m, m),
                    "n_mode": int(take.sum()),
                    "caught": int(fired.sum()),
                    "mean_alarm_chunk": float(first_r[fired].mean()) if fired.any() else np.nan,
                    "median_alarm_chunk": float(np.median(first_r[fired])) if fired.any() else np.nan,
                    "mean_prior_at_alarm": float(np.nanmean(prior_at[fired])) if fired.any() else np.nan,
                    "early_caught": int((fired & (prior_at < C.LOW_PRIOR)).sum()),
                })

    pd.DataFrame(timing_rows).to_csv(OUT / "alarm_timing_by_mode.csv", index=False)
    (OUT / "confound_structure.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
