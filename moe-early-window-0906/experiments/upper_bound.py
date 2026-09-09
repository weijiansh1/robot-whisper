#!/usr/bin/env python3
"""How much in-window recall is reachable, and how much of it is not routing.

Two things have to be separated before any recall number means anything.

1.  The survival baseline.  Timely FPR is FP divided by *all* non-risk episodes
    of the suite, but by the end of the window most of them have already
    finished.  Alarming on every survivor at chunk q therefore costs
    n_safe_alive(q) / n_safe, which collapses towards zero, while catching every
    risk by construction.  A detector that fires late is buying recall with the
    horizon cap, not with routing.  For every (suite, chunk) this script
    computes, in closed form, the recall a rule that fires on a *random* subset
    of the survivors gets at the same FPR:

        baseline_recall = fp * (risk_alive / safe_alive) / n_risk

    and reports every measured recall against it.

2.  Selection.  Every headline here is the maximum over 208 columns x 2
    directions x 22 alpha values.  The identical maximisation is run on the
    null-control channels, which carry no episode information, and on the
    length leak, which carries the answer by definition.  The three numbers
    bracket what the sweep can produce.

Arms
  best_single   best single (quantity, layer, direction, alpha) at one chunk
  stacked_mv    every quantity at once, with its own recent history:
                [x_q, x_q - x_{q-4}, trailing-mean_4(x)_q], within-task
                standardised, L2 logistic.  Cross-fitted by init_state_id, and
                separately fitted on development and replayed on external.
  null_mv       the same model on the null-control channels only
  length_only   fire on every survivor at chunk q

The reported operating point is per suite: the chunk inside the window that
maximises recall subject to a per-suite timely FPR <= 0.005.  Because every
suite is held to the same rate, the cohort rate is also <= 0.005.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import common as C
from reachability import ALPHAS, FPR_BUDGET, standardise_within_task, task_thresholds

N_FOLDS = 5
NULL_CONTROLS = (
    "ctrl_const_elapsed",
    "ctrl_episode_const_rand",
    "ctrl_episode_const_rand2",
    "ctrl_flow_noise_seed",
    "ctrl_white_noise",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=C.RESULTS)
    return parser.parse_args()


def survival_table(frame: dict, suite: str) -> pd.DataFrame:
    m = frame["suite"] == suite
    length, risk = frame["length"][m], frame["risk"][m]
    n_risk, n_safe = int(risk.sum()), int((~risk).sum())
    cap = C.CAPS[suite]
    rows = []
    for q in range(C.SWEEP_START, min(cap, frame["n_chunk"])):
        alive = length > q
        sa, ra = int((alive & ~risk).sum()), int((alive & risk).sum())
        fp_budget = FPR_BUDGET * n_safe
        base = min(fp_budget, sa) * (ra / max(sa, 1)) / max(n_risk, 1) if sa else float(ra > 0)
        rows.append(
            {
                "cohort": frame["cohort"], "suite": suite, "chunk": q,
                "phase": (q + 1) / cap, "cap": cap,
                "alive": int(alive.sum()), "safe_alive": sa, "risk_alive": ra,
                "survival_prior": ra / max(int(alive.sum()), 1),
                "n_risk": n_risk, "n_safe": n_safe,
                "fpr_if_all_survivors_alarm": sa / max(n_safe, 1),
                "length_only_recall_at_budget": min(1.0, base),
                "in_window_65": q <= C.WINDOW_65[suite],
            }
        )
    return pd.DataFrame(rows)


def stacked_features(block: np.ndarray, tm: np.ndarray, rows: np.ndarray, q: int,
                     keep: np.ndarray) -> np.ndarray:
    x = block[rows, q, :][:, keep]
    lag = block[rows, max(q - C.WIDTH, 0), :][:, keep]
    smooth = tm[rows, q, :][:, keep]
    return np.concatenate([x, x - lag, smooth], axis=1)


def fit_and_score(x, risk, groups, seed):
    """(cross-fitted score, model fitted on everything)."""
    out = np.full(len(risk), np.nan)
    full = None
    if risk.sum() >= 2 and (~risk).sum() >= 2:
        full = LogisticRegression(C=0.02, max_iter=2000, random_state=seed).fit(x, risk)
    n_splits = min(N_FOLDS, len(np.unique(groups)))
    if n_splits >= 2 and risk.sum() >= n_splits:
        for train, test in GroupKFold(n_splits=n_splits).split(x, risk, groups):
            if risk[train].sum() < 2 or (~risk[train]).sum() < 2:
                continue
            model = LogisticRegression(C=0.02, max_iter=2000, random_state=seed)
            model.fit(x[train], risk[train])
            out[test] = model.decision_function(x[test])
    return out, full


def alarms_from_score(score, task_code, n_task):
    ok = np.isfinite(score)
    s = np.where(ok, score, -np.inf).astype(np.float32)[:, None]
    thr = task_thresholds(s, task_code, n_task)
    return (s[None, :, 0] >= thr[:, task_code, 0]) & ok[None, :]


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = {c: C.build_cohort(c) for c in ("development_main", "external_8b")}
    blocks, smooths = {}, {}
    for cohort, frame in frames.items():
        blocks[cohort] = C.flatten(frame["values"])
        smooths[cohort] = C.flatten(C.trailing_mean(frame["values"], C.WIDTH))
    names, _ = C.column_index(frames["development_main"]["quantities"])
    names = np.asarray(names)
    keep_real = np.flatnonzero(~np.isin(names, list(C.CONTROL_QUANTITIES)))
    keep_null = np.flatnonzero(np.isin(names, list(NULL_CONTROLS)))

    survival: list[pd.DataFrame] = []
    mv_rows: list[dict] = []

    for suite in sorted(C.CAPS):
        q_max = C.WINDOW_65[suite]
        chunks = list(range(C.SWEEP_START, q_max + 1))
        dev_models: dict[str, dict[int, LogisticRegression]] = {"stacked_mv": {}, "null_mv": {}}
        for cohort in ("development_main", "external_8b"):
            frame = frames[cohort]
            survival.append(survival_table(frame, suite))
            rows_suite = np.flatnonzero(frame["suite"] == suite)
            length = frame["length"][rows_suite]
            _, task_code_all = np.unique(frame["task"][rows_suite], return_inverse=True)
            n_task = int(task_code_all.max()) + 1
            risk = frame["risk"][rows_suite]
            groups = frame["init_state_id"][rows_suite]
            n_risk, n_safe = int(risk.sum()), int((~risk).sum())
            block, tm = blocks[cohort][rows_suite], smooths[cohort][rows_suite]

            for q in chunks:
                alive = length > q
                if alive.sum() < C.MIN_STRATUM or risk[alive].sum() < 5:
                    continue
                rows = np.flatnonzero(alive)
                tcode = task_code_all[rows]
                sa = int((~risk[rows]).sum())
                ra = int(risk[rows].sum())
                for arm, keep in (("stacked_mv", keep_real), ("null_mv", keep_null)):
                    x = stacked_features(block, tm, rows, q, keep)
                    x = standardise_within_task(x, tcode, n_task)
                    score, full = fit_and_score(x, risk[rows], groups[rows], C.SEED + q)
                    variants = {"crossfit": score}
                    if cohort == "development_main":
                        if full is not None:
                            dev_models[arm][q] = full
                    elif q in dev_models[arm]:
                        variants["dev_fitted_transfer"] = dev_models[arm][q].decision_function(x)
                    for variant, values in variants.items():
                        fired = alarms_from_score(values, tcode, n_task)
                        tp = fired[:, risk[rows]].sum(axis=1)
                        fp = fired[:, ~risk[rows]].sum(axis=1)
                        for a, alpha in enumerate(ALPHAS):
                            mv_rows.append(
                                dict(
                                    suite=suite, cohort=cohort, arm=arm, variant=variant,
                                    chunk=q, phase=(q + 1) / C.CAPS[suite],
                                    alpha=float(alpha), tp=int(tp[a]), fp=int(fp[a]),
                                    recall=tp[a] / max(n_risk, 1),
                                    fpr=fp[a] / max(n_safe, 1),
                                    baseline_recall=min(
                                        1.0, fp[a] * (ra / max(sa, 1)) / max(n_risk, 1)
                                    ),
                                    n_risk=n_risk, n_safe=n_safe, safe_alive=sa, risk_alive=ra,
                                )
                            )
            print(f"  {suite} {cohort} done", flush=True)

    survival_table_all = pd.concat(survival, ignore_index=True)
    survival_table_all.to_csv(args.output / "survival_baseline.csv", index=False)
    mv = pd.DataFrame(mv_rows)
    mv["excess_recall"] = mv["recall"] - mv["baseline_recall"]
    mv.to_csv(args.output / "upper_bound_multivariate.csv", index=False)

    # ---- headline: best in-window operating point per suite and arm
    single = pd.read_csv(args.output / "reachability_single_chunk.csv.gz")
    single = single.merge(
        survival_table_all[["cohort", "suite", "chunk", "safe_alive", "risk_alive"]],
        on=["cohort", "suite", "chunk"], how="left",
    )
    single["baseline_recall"] = np.minimum(
        1.0, single["fp"] * (single["risk_alive"] / single["safe_alive"].clip(lower=1))
        / single["n_risk"].clip(lower=1)
    )
    single["excess_recall"] = single["recall"] - single["baseline_recall"]
    single["in_window_65"] = single.apply(
        lambda r: r["chunk"] <= C.WINDOW_65[r["suite"]], axis=1
    )
    single["arm"] = np.where(
        single["quantity"] == "leak_full_length", "length_leak",
        np.where(single["is_control"], "null_control", "best_single"),
    )

    def pick(table, key):
        ok = table[(table["fpr"] <= FPR_BUDGET)]
        if not len(ok):
            return None
        return ok.sort_values(["recall", "fpr"], ascending=[False, True]).iloc[0]

    headline: list[dict] = []
    for cohort in ("development_main", "external_8b"):
        for suite in sorted(C.CAPS):
            base = single[
                (single["cohort"] == cohort) & (single["suite"] == suite)
                & single["in_window_65"]
            ]
            surv = survival_table_all[
                (survival_table_all["cohort"] == cohort)
                & (survival_table_all["suite"] == suite)
                & survival_table_all["in_window_65"]
            ]
            entries = {
                "best_single": base[base["arm"] == "best_single"],
                "null_control": base[base["arm"] == "null_control"],
                "length_leak": base[base["arm"] == "length_leak"],
            }
            for arm in ("stacked_mv", "null_mv"):
                for variant in ("crossfit", "dev_fitted_transfer"):
                    sub = mv[
                        (mv["cohort"] == cohort) & (mv["suite"] == suite)
                        & (mv["arm"] == arm) & (mv["variant"] == variant)
                    ]
                    if len(sub):
                        entries[f"{arm}:{variant}"] = sub
            for label, table in entries.items():
                best = pick(table, label)
                if best is None:
                    continue
                headline.append(
                    {
                        "cohort": cohort, "suite": suite, "arm": label,
                        "chunk": int(best["chunk"]), "phase": float(best["phase"]),
                        "alpha": float(best["alpha"]),
                        "tp": int(best["tp"]), "fp": int(best["fp"]),
                        "n_risk": int(best["n_risk"]),
                        "recall": float(best["recall"]), "fpr": float(best["fpr"]),
                        "baseline_recall": float(best["baseline_recall"]),
                        "excess_recall": float(best["excess_recall"]),
                        "head": (
                            f"{best['quantity']}|{best['layer']}|{best['direction']}"
                            if "quantity" in best.index else label
                        ),
                    }
                )
            # length-only rule: alarm on every survivor
            ok = surv[surv["fpr_if_all_survivors_alarm"] <= FPR_BUDGET]
            headline.append(
                {
                    "cohort": cohort, "suite": suite, "arm": "length_only_all_survivors",
                    "chunk": int(ok["chunk"].min()) if len(ok) else -1,
                    "phase": float(ok["phase"].min()) if len(ok) else np.nan,
                    "alpha": np.nan,
                    "tp": int(surv["n_risk"].iloc[0]) if len(ok) else 0,
                    "fp": int(ok["safe_alive"].iloc[0]) if len(ok) else 0,
                    "n_risk": int(surv["n_risk"].iloc[0]),
                    "recall": 1.0 if len(ok) else 0.0,
                    "fpr": float(ok["fpr_if_all_survivors_alarm"].min()) if len(ok) else np.nan,
                    "baseline_recall": 1.0 if len(ok) else 0.0,
                    "excess_recall": 0.0,
                    "head": "alarm on every survivor",
                }
            )

    head = pd.DataFrame(headline)
    head.to_csv(args.output / "headline_operating_points.csv", index=False)

    totals = (
        head.groupby(["cohort", "arm"])
        .apply(lambda d: pd.Series({
            "tp": d["tp"].sum(), "n_risk": d["n_risk"].sum(),
            "fp": d["fp"].sum(), "recall": d["tp"].sum() / max(d["n_risk"].sum(), 1),
            "suites": len(d),
        }), include_groups=False)
        .reset_index()
    )
    totals.to_csv(args.output / "headline_cohort_totals.csv", index=False)
    print(head.to_string(index=False))
    print()
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
