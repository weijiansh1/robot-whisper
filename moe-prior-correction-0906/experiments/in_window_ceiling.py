"""How much in-window recall is even available, per suite?

The target is 80% risk recall with the alarm fired in the first 65% of the
trajectory.  Whether that is reachable is decided suite by suite, because the
suites have different horizon caps and v7's information is not spread evenly
across them.

This scores every published alarm vector per suite on:
  * recall with the alarm at phase <= 0.65 (the operational target)
  * task-matched lift (does it beat the per-task survival prior)
and then computes an oracle ceiling: the best achievable in-window recall if a
different detector could be chosen for each suite.  The oracle is selected on
the same data it is scored on, so it is an optimistic upper bound - if the
oracle cannot reach the target, nothing can.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recompute_task_matched_lift import (  # noqa: E402
    COHORTS,
    META_KEYS,
    ROOT,
    SOURCES,
    cohort_frame,
    prior_of,
    survival_prior,
)

OUT = HERE.parent / "results"
PHASE = 0.65
# A detector that fires on nearly everything trivially reaches high recall.
# Hold false alarms to the frozen protocol's timely budget.
MAX_TIMELY_FPR = 0.005


def main() -> None:
    rows = []
    caps: dict[str, dict[str, int]] = {}
    for cohort_name in COHORTS:
        coh = cohort_frame(cohort_name)
        risk, suite, length = coh["risk"], coh["suite"], coh["length"]
        pt = survival_prior(coh["task"], length, risk)
        cap = pd.Series(length).groupby(suite).transform("max").to_numpy()
        caps[cohort_name] = {
            s: int(length[suite == s].max()) for s in np.unique(suite)
        }

        for bundle, files in SOURCES.items():
            rel = files.get(cohort_name)
            if rel is None or not (ROOT / bundle / rel).exists():
                continue
            data = np.load(ROOT / bundle / rel, allow_pickle=True)
            for key in data.files:
                if key in META_KEYS:
                    continue
                first = data[key]
                if first.ndim != 1 or first.shape[0] != len(risk):
                    continue
                first = first.astype(int)
                fired = first >= 0
                # "In window" = the alarm itself lands in the first 65% of the
                # suite's horizon.  Alarms after that are outside the target.
                in_win = fired & (first < PHASE * cap)
                p_task = prior_of(first, coh["task"], pt)
                for s in np.unique(suite):
                    m = suite == s
                    n_risk = int((m & risk).sum())
                    if n_risk == 0:
                        continue
                    wtp = int((in_win & m & risk).sum())
                    wfp = int((in_win & m & ~risk).sum())
                    tp = int((fired & m & risk).sum())
                    fp = int((fired & m & ~risk).sum())
                    base = (
                        float(np.nanmean(p_task[fired & m])) if (fired & m).any() else np.nan
                    )
                    lift = (
                        (tp / (tp + fp)) / base
                        if (tp + fp) and np.isfinite(base) and base > 0
                        else np.nan
                    )
                    rows.append(
                        {
                            "cohort": cohort_name,
                            "bundle": bundle,
                            "detector": key,
                            "suite": s,
                            "n_risk": n_risk,
                            "n_safe": int((m & ~risk).sum()),
                            "window_tp": wtp,
                            "window_fp": wfp,
                            "window_recall": wtp / n_risk,
                            "window_precision": wtp / (wtp + wfp) if wtp + wfp else np.nan,
                            "window_fpr": wfp / int((m & ~risk).sum()),
                            "any_recall": tp / n_risk,
                            "any_precision": tp / (tp + fp) if tp + fp else np.nan,
                            "task_lift": lift,
                        }
                    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "in_window_by_suite.csv", index=False)

    print("suite horizon caps:", json.dumps(caps["external_8b"]))
    print(f"in-window = alarm chunk < {PHASE} x cap; FP budget = timely FPR <= {MAX_TIMELY_FPR}\n")

    for cohort_name in COHORTS:
        sub = df[df.cohort == cohort_name]
        print(f"===== {cohort_name} =====")
        for s in sorted(sub.suite.unique()):
            g = sub[sub.suite == s]
            n_risk = int(g.n_risk.iloc[0])
            cap = caps[cohort_name][s]
            # Unconstrained best, then best inside the false-alarm budget.
            best = g.nlargest(1, "window_recall").iloc[0]
            ok = g[g.window_fpr <= MAX_TIMELY_FPR]
            budget = ok.nlargest(1, "window_recall").iloc[0] if len(ok) else None
            print(
                f"-- {s}: {n_risk} risks, cap {cap}, window = chunks 0..{int(PHASE*cap)-1}"
            )
            print(
                f"   best in-window, no FP budget : recall {best.window_recall:.3f}"
                f" ({best.window_tp}/{n_risk})  fp {best.window_fp}"
                f"  fpr {best.window_fpr:.4f}  task_lift {best.task_lift:.3f}"
                f"   [{best.detector}]"
            )
            if budget is not None:
                print(
                    f"   best within FP budget       : recall {budget.window_recall:.3f}"
                    f" ({budget.window_tp}/{n_risk})  fp {budget.window_fp}"
                    f"  task_lift {budget.task_lift:.3f}   [{budget.detector}]"
                )
            else:
                print("   best within FP budget       : none fires in window")
        # Oracle: a different detector per suite, selected on this same data.
        for label, frame in (("no FP budget", sub), ("FP budget", sub[sub.window_fpr <= MAX_TIMELY_FPR])):
            tot_r = sub.groupby("suite").n_risk.first().sum()
            got = frame.groupby("suite").window_tp.max().sum() if len(frame) else 0
            print(
                f"   ORACLE per-suite ({label}): {int(got)}/{int(tot_r)}"
                f" = {got / tot_r:.3f} in-window recall"
            )
        print()


if __name__ == "__main__":
    main()
