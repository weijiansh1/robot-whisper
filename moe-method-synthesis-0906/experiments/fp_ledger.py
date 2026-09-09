"""Part 2 - the false alarm ledger, and what voting can and cannot buy.

The motivating structure is that false alarms barely overlap across methods
while true alarms overlap heavily.  This script measures that directly, per
suite as well as pooled, against a rate-matched null, and then decomposes what
each vote threshold k actually removes: idiosyncratic false alarms that only
one method ever raised, versus genuine detections that voting throws away.

Safe episodes flagged by many methods at once are the irreducible ones and are
characterised the same way the uncaught risks are: suite, task, init state, and
whether they are near-misses (long but successful).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ledger_core as lc
import synth_core as sc

SEED = 20260906


def overlap_structure(mat: np.ndarray, c: sc.Cohort, tag: str,
                      rng: np.random.Generator, draws: int = 200) -> pd.DataFrame:
    """Do true alarms overlap more than false alarms, and is it above chance?

    ``concentration`` = sum of per-method set sizes / size of the union.  A
    value of 1 means the methods never agree; a value of m means all m methods
    fire on exactly the same episodes.  Reported separately for the risk and
    the safe partition and against a rate-matched null that preserves each
    method's per-suite firing rate.
    """
    rows = []
    groups = [("pooled", np.ones(c.n, bool))] + [
        (s, c.suite == s) for s in sorted(set(c.suite))
    ]
    for gname, gm in groups:
        sub = mat[:, gm]
        risk = c.risk[gm]
        for part, sel in (("risk", risk), ("safe", ~risk)):
            m = sub[:, sel]
            union = m.any(0).sum()
            total = m.sum()
            obs = float(total / union) if union else np.nan
            # two nulls.  "global" preserves each method's rate over the whole
            # suite and lets chance decide how much lands on this partition;
            # "within" additionally preserves how many alarms each method
            # places inside the partition, so it isolates agreement from
            # selectivity.
            null_g = np.empty(draws)
            null_w = np.empty(draws)
            for d in range(draws):
                nm = sc.rate_matched_null(sub, rng, strata=None)[:, sel]
                u = nm.any(0).sum()
                null_g[d] = nm.sum() / u if u else np.nan
                nw = sc.rate_matched_null(m, rng, strata=None)
                uw = nw.any(0).sum()
                null_w[d] = nw.sum() / uw if uw else np.nan
            rows.append({
                "arm": tag, "cohort": c.name, "suite": gname, "partition": part,
                "n_methods": int(mat.shape[0]),
                "n_episodes": int(sel.sum()),
                "n_flagged_union": int(union),
                "sum_of_sets": int(total),
                "concentration": obs,
                "concentration_null": float(np.nanmean(null_g)),
                "concentration_null_sd": float(np.nanstd(null_g)),
                "concentration_null_within": float(np.nanmean(null_w)),
                "concentration_null_within_sd": float(np.nanstd(null_w)),
                "excess_over_null": obs - float(np.nanmean(null_g)),
                "excess_over_null_within": obs - float(np.nanmean(null_w)),
                "mean_votes_given_flagged": float(m.sum(0)[m.any(0)].mean())
                if union else np.nan,
                "share_flagged_by_exactly_one": float(
                    (m.sum(0)[m.any(0)] == 1).mean()) if union else np.nan,
            })
    return pd.DataFrame(rows)


def vote_decomposition(mat: np.ndarray, c: sc.Cohort, tag: str) -> pd.DataFrame:
    """What each vote threshold k removes, split into FP and TP components."""
    votes = mat.sum(0)
    risk = c.risk
    rows = []
    base_fp = int(((votes >= 1) & ~risk).sum())
    base_tp = int(((votes >= 1) & risk).sum())
    idio_fp = int(((votes == 1) & ~risk).sum())
    for gname, gm in [("pooled", np.ones(c.n, bool))] + [
        (s, c.suite == s) for s in sorted(set(c.suite))
    ]:
        v, r = votes[gm], risk[gm]
        b_fp = int(((v >= 1) & ~r).sum())
        b_tp = int(((v >= 1) & r).sum())
        for k in range(1, mat.shape[0] + 1):
            fp = int(((v >= k) & ~r).sum())
            tp = int(((v >= k) & r).sum())
            # of the false alarms killed going from k-1 to k, how many were
            # idiosyncratic (raised by exactly one method) at k=1
            killed_fp = b_fp - fp
            killed_idio = int(((v == 1) & ~r).sum()) if k >= 2 else 0
            rows.append({
                "arm": tag, "cohort": c.name, "suite": gname, "k": k,
                "tp": tp, "fp": fp,
                "recall": float(tp / r.sum()) if r.sum() else np.nan,
                "fpr": float(fp / (~r).sum()),
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "fp_removed_vs_k1": killed_fp,
                "fp_removed_that_were_idiosyncratic": min(killed_fp, killed_idio),
                "share_of_removed_fp_idiosyncratic": float(
                    min(killed_fp, killed_idio) / killed_fp) if killed_fp else np.nan,
                "tp_lost_vs_k1": b_tp - tp,
                "fp_removed_per_tp_lost": float(killed_fp / (b_tp - tp))
                if (b_tp - tp) else np.inf,
            })
    rows.append({"arm": tag, "cohort": c.name, "suite": "_base", "k": 0,
                 "tp": base_tp, "fp": base_fp, "recall": base_tp / risk.sum(),
                 "fpr": base_fp / (~risk).sum(),
                 "precision": base_tp / max(base_tp + base_fp, 1),
                 "fp_removed_vs_k1": 0, "fp_removed_that_were_idiosyncratic": idio_fp,
                 "share_of_removed_fp_idiosyncratic": np.nan, "tp_lost_vs_k1": 0,
                 "fp_removed_per_tp_lost": np.nan})
    return pd.DataFrame(rows)


def irreducible_profile(mat: np.ndarray, c: sc.Cohort, tag: str) -> pd.DataFrame:
    """Who are the safe episodes that many methods flag at once."""
    votes = mat.sum(0)
    safe = ~c.risk
    m = mat.shape[0]
    rows = []
    cap = np.array([sc.CAP[s] for s in c.suite])
    groups = [
        ("idiosyncratic_fp_votes1", safe & (votes == 1)),
        ("fp_votes2", safe & (votes == 2)),
        ("fp_votes3plus", safe & (votes >= 3)),
        ("fp_votes_half_or_more", safe & (votes >= max(2, int(np.ceil(m / 2))))),
        ("any_fp", safe & (votes >= 1)),
        ("never_flagged_safe", safe & (votes == 0)),
    ]
    for label, sel in groups:
        if not sel.any():
            continue
        rows.append({
            "arm": tag, "cohort": c.name, "group": label, "n": int(sel.sum()),
            "share_of_safe": float(sel.sum() / safe.sum()),
            "mean_length": float(c.length[sel].mean()),
            "mean_length_fraction_of_cap": float((c.length[sel] / cap[sel]).mean()),
            "share_at_or_above_90pct_of_cap": float(
                (c.length[sel] >= 0.9 * cap[sel]).mean()),
            "mean_length_all_safe": float(c.length[safe].mean()),
            "mean_length_fraction_all_safe": float((c.length[safe] / cap[safe]).mean()),
            "share_at_90pct_all_safe": float((c.length[safe] >= 0.9 * cap[safe]).mean()),
            "top_suite": pd.Series(c.suite[sel]).value_counts(normalize=True).index[0],
            "top_suite_share": float(
                pd.Series(c.suite[sel]).value_counts(normalize=True).iloc[0]),
            "top_task": pd.Series(c.task[sel]).value_counts(normalize=True).index[0],
            "top_task_share": float(
                pd.Series(c.task[sel]).value_counts(normalize=True).iloc[0]),
            "n_distinct_tasks": int(pd.Series(c.task[sel]).nunique()),
            "init_state_gini": float(_gini(
                pd.Series(c.episode[sel] % 50).value_counts().to_numpy())),
        })
    return pd.DataFrame(rows)


def _gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def heavy_fp_by_task(mat: np.ndarray, c: sc.Cohort, tag: str) -> pd.DataFrame:
    votes = mat.sum(0)
    safe = ~c.risk
    heavy = safe & (votes >= 3)
    rows = []
    for t in sorted(set(c.task)):
        tm = c.task == t
        if not (safe & tm).any():
            continue
        rows.append({
            "arm": tag, "cohort": c.name, "task": t, "suite": c.suite[tm][0],
            "n_safe": int((safe & tm).sum()),
            "n_heavy_fp": int((heavy & tm).sum()),
            "heavy_fp_rate": float((heavy & tm).sum() / (safe & tm).sum()),
            "task_risk_rate": float(c.risk[tm].mean()),
            "mean_length_safe": float(c.length[safe & tm].mean()),
            "mean_length_heavy_fp": float(c.length[heavy & tm].mean())
            if (heavy & tm).any() else np.nan,
        })
    return pd.DataFrame(rows)


def _greedy_top(dev: sc.Cohort, names: list[str], budget: float, k: int) -> list[str]:
    iw = lc.in_window(dev, names)
    fpr = (iw & ~dev.risk).sum(1) / (~dev.risk).sum()
    pool = [i for i in range(len(names)) if fpr[i] <= budget]
    cov = np.zeros(dev.n, bool)
    chosen: list[str] = []
    while pool and len(chosen) < k:
        best = max(pool, key=lambda i: (int((iw[i] & dev.risk & ~cov).sum()),
                                        -int((iw[i] & ~dev.risk & ~cov).sum()),
                                        names[i]))
        pool.remove(best)
        cov |= iw[best]
        chosen.append(names[best])
    return chosen


def main() -> None:
    rng = np.random.default_rng(SEED)
    dev = sc.load_cohort("development_main")
    ext = sc.load_cohort("external_8b")
    names, labels = lc.load_family_labels()

    ov, dec, prof, bytask = [], [], [], []
    for budget in lc.BUDGETS:
        reps = lc.representatives(dev, names, labels, budget)
        if reps.empty:
            continue
        arms = [(f"families@{budget}", reps.detector.tolist())]
        # A five-method arm, to mirror the voting result that motivated this
        # ledger.  The five are chosen on development by greedy marginal true
        # alarms among the detectors inside the same budget; external is scored
        # once with that fixed order.
        arms.append((f"greedy_top5@{budget}", _greedy_top(dev, names, budget, 5)))
        for tag, order in arms:
            if len(order) < 2:
                continue
            for c in (dev, ext):
                mat = lc.in_window(c, order)
                ov.append(overlap_structure(mat, c, tag, rng))
                dec.append(vote_decomposition(mat, c, tag))
                prof.append(irreducible_profile(mat, c, tag))
                if budget == lc.REFERENCE_BUDGET:
                    bytask.append(heavy_fp_by_task(mat, c, tag))

    pd.concat(ov, ignore_index=True).to_csv(sc.RESULTS / "alarm_overlap.csv", index=False)
    pd.concat(dec, ignore_index=True).to_csv(sc.RESULTS / "vote_decomposition.csv",
                                             index=False)
    pd.concat(prof, ignore_index=True).to_csv(sc.RESULTS / "fp_profile.csv", index=False)
    pd.concat(bytask, ignore_index=True).to_csv(sc.RESULTS / "heavy_fp_by_task.csv",
                                                index=False)

    o = pd.concat(ov, ignore_index=True)
    ref = o[(o.arm == f"families@{lc.REFERENCE_BUDGET}")]
    meta = {
        "schema": "himoe.method_synthesis.fp_ledger.v1",
        "reference_budget": lc.REFERENCE_BUDGET,
        "concentration_note": (
            "concentration = sum of per-method alarm-set sizes / size of their "
            "union; 1 = no agreement, n_methods = perfect agreement"
        ),
        "reference": ref.to_dict(orient="records"),
    }
    (sc.RESULTS / "fp_ledger.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 240)
    print(ref.to_string(index=False))


if __name__ == "__main__":
    main()
