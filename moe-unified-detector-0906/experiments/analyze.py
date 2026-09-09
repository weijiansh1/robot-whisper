#!/usr/bin/env python3
"""Read the frozen build and produce every reported number.

Nothing is fitted here.  Operating points are selected on ``development_main``
only; the sealed cohorts are read at those frozen points.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import protocol as P
import anchors
import controls as CTL
import detector as D
import sweep as S
import recompute_task_matched_lift as TML

RESULTS = P.RESULTS
FPR_BUDGETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
HEADLINE = "unified"


def survival_at(curve: pd.DataFrame, cohort: str, fpr: float) -> float:
    sub = curve[curve[f"{cohort}.iw_fpr"] <= fpr + 1e-12]
    return float(sub[f"{cohort}.iw_recall"].max()) if len(sub) else 0.0


def pooled(row: pd.Series, cohorts) -> dict:
    if any(pd.isna(row.get(f"{c}.iw_tp", np.nan)) for c in cohorts):
        return {k: np.nan for k in ("iw_tp", "iw_fp", "n_risk", "n_safe",
                                    "iw_recall", "iw_fpr", "iw_precision")}
    tp = sum(int(row[f"{c}.iw_tp"]) for c in cohorts)
    fp = sum(int(row[f"{c}.iw_fp"]) for c in cohorts)
    nr = sum(int(row[f"{c}.n_risk"]) for c in cohorts)
    ns = sum(int(row[f"{c}.n_safe"]) for c in cohorts)
    return {
        "iw_tp": tp, "iw_fp": fp, "n_risk": nr, "n_safe": ns,
        "iw_recall": tp / nr if nr else np.nan,
        "iw_fpr": fp / ns if ns else np.nan,
        "iw_precision": tp / (tp + fp) if (tp + fp) else np.nan,
    }


def add_pooled(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    have = [c for c in P.COHORTS if f"{c}.iw_tp" in out.columns]
    sealed = [c for c in P.SEALED_COHORTS if f"{c}.iw_tp" in out.columns]
    for tag, cohorts in (("corpus", have), ("sealed", sealed)):
        if not cohorts:
            continue
        vals = out.apply(lambda r: pooled(r, cohorts), axis=1, result_type="expand")
        for key in vals.columns:
            out[f"{tag}.{key}"] = vals[key]
    return out


def main() -> None:
    frames = {c: P.load_cohort(c) for c in P.COHORTS}
    with open(RESULTS / "artefacts.pkl", "rb") as fh:
        art = pickle.load(fh)
    table = add_pooled(pd.read_csv(RESULTS / "operating_points.csv.gz"))
    table.to_csv(RESULTS / "operating_points_pooled.csv.gz", index=False)

    # ---- anchors ---------------------------------------------------------- #
    anchor = anchors.check(frames)
    P.write_json(RESULTS / "anchor_reproduction.json", anchor)
    anchors.window_variants(frames).to_csv(
        RESULTS / "anchor_window_variants.csv", index=False
    )
    print("anchors reproduced:", anchor["reproduced"], anchor["mismatch"])
    assert anchor["reproduced"], anchor["mismatch"]

    P.write_json(
        RESULTS / "window_definition.json",
        {
            "published_anchor_window_end": P.IN_WINDOW_END,
            "strict_phase_window_end": P.IN_WINDOW_END_STRICT,
            "rule": "q <= round(0.65*cap - 1), banker's rounding",
            "phase_definition": "(q+1)/cap; cap is a rollout configuration value",
        },
    )

    # ---- controls --------------------------------------------------------- #
    auc = pd.read_csv(RESULTS / "auc/development_real_extended.csv.gz")
    const = CTL.const_elapsed_check(auc)
    assert const["exactly_half"], const
    dev = frames[P.FIT_COHORT]
    ties = CTL.tie_audit(dev, dev["values"], dev["columns"])
    ties.to_csv(RESULTS / "tie_audit.csv", index=False)

    det = art["detectors"][HEADLINE]
    Sdev = art["scores"][HEADLINE][P.FIT_COHORT]
    gmask = D.gate_mask(dev, det.gated)
    tie_checks = [CTL.tie_group_fires(Sdev, gmask, a) for a in (0.005, 0.02, 0.05, 0.2)]

    used = pd.read_csv(RESULTS / "within_episode_information.csv")
    used_cols = list(dict.fromkeys(det.columns))
    used_info = used[used["column"].isin(used_cols)]
    used_info.to_csv(RESULTS / "within_episode_information_used.csv", index=False)

    P.write_json(
        RESULTS / "controls.json",
        {
            "const_elapsed_auc": const,
            "tie_group_fires": tie_checks,
            "tie_audit_worst": ties.sort_values(
                "loss_pct_from_strict_gt", ascending=False
            ).head(12).to_dict("records"),
            "within_episode_information_used": used_info.to_dict("records"),
            "degenerate_channels_excluded": used.loc[
                used["frac_episodes_constant"] >= 0.999, "column"
            ].tolist(),
            "length_negative_control": {
                "is_baseline": False,
                "note": "risk == did not finish before the cap, so any sub-cap "
                        "length rule recalls 100% by construction",
                **{
                    k: v
                    for k, v in table[
                        table["detector"] == "length_leak_NOT_a_baseline"
                    ].iloc[0].items()
                    if isinstance(v, (int, float, np.integer, np.floating))
                    and not pd.isna(v)
                },
            },
        },
    )

    # ---- headline curves --------------------------------------------------- #
    surv = table[table["detector"] == "survival_prior"]
    fronts = []
    for (name, arm), sub in table.groupby(["detector", "arm"]):
        if name in ("survival_prior", "length_leak_NOT_a_baseline"):
            continue
        front = P.pareto_front(sub, "development_main.iw_fpr", "development_main.iw_recall")
        front["scope"] = "any_rule"
        for rule, rsub in sub.groupby("rule"):
            f = P.pareto_front(rsub, "development_main.iw_fpr", "development_main.iw_recall")
            f["scope"] = rule
            front = pd.concat([front, f], ignore_index=True)
        fronts.append(front)
    front = pd.concat(fronts, ignore_index=True)
    front.to_csv(RESULTS / "frontier_pooled.csv.gz", index=False)

    # ---- budget table ------------------------------------------------------ #
    rows = []
    for (name, arm, scope), sub in front.groupby(["detector", "arm", "scope"]):
        for budget in FPR_BUDGETS:
            ok = sub[sub["development_main.iw_fpr"] <= budget + 1e-12]
            if ok.empty:
                continue
            pick = ok.sort_values("development_main.iw_recall").iloc[-1]
            rec = {
                "detector": name, "arm": arm, "scope": scope, "budget": budget,
                "rule": pick["rule"], "alpha": pick["alpha"], "tau": pick["tau"],
                "k": pick["k"], "drift": pick["drift"], "point_id": pick["point_id"],
            }
            for cohort in list(P.COHORTS) + ["corpus", "sealed"]:
                for key in ("iw_tp", "iw_fp", "n_risk", "iw_recall", "iw_fpr",
                            "iw_precision"):
                    col = f"{cohort}.{key}"
                    if col in pick:
                        rec[col] = pick[col]
            for cohort in P.COHORTS:
                col = f"{cohort}.iw_fpr"
                if col in pick and np.isfinite(pick[col]):
                    rec[f"{cohort}.survival_recall_at_same_fpr"] = survival_at(
                        surv, cohort, pick[col]
                    )
                    rec[f"{cohort}.excess_over_survival"] = (
                        pick[f"{cohort}.iw_recall"]
                        - rec[f"{cohort}.survival_recall_at_same_fpr"]
                    )
            rows.append(rec)
    budget = pd.DataFrame(rows)
    budget.to_csv(RESULTS / "budget_table.csv", index=False)

    # ---- can 0.80 in-window recall be reached? ----------------------------- #
    target_rows = []
    for (name, arm, scope), sub in front.groupby(["detector", "arm", "scope"]):
        ok = sub[sub["development_main.iw_recall"] >= 0.80]
        rec = {"detector": name, "arm": arm, "scope": scope,
               "reaches_0.80_on_development": bool(len(ok))}
        if len(ok):
            pick = ok.sort_values("development_main.iw_fpr").iloc[0]
            for cohort in list(P.COHORTS) + ["corpus", "sealed"]:
                for key in ("iw_tp", "iw_fp", "n_risk", "iw_recall", "iw_fpr",
                            "iw_precision"):
                    col = f"{cohort}.{key}"
                    if col in pick:
                        rec[col] = pick[col]
            rec.update({"rule": pick["rule"], "alpha": pick["alpha"],
                        "tau": pick["tau"], "k": pick["k"], "drift": pick["drift"],
                        "point_id": pick["point_id"]})
            for s in P.SUITES:
                for cohort in P.COHORTS:
                    col = f"{cohort}.iw_recall_{s}"
                    if col in pick:
                        rec[col] = pick[col]
        else:
            best = sub.sort_values("development_main.iw_recall").iloc[-1]
            rec["max_development_iw_recall"] = float(best["development_main.iw_recall"])
            rec["at_development_iw_fpr"] = float(best["development_main.iw_fpr"])
        target_rows.append(rec)
    pd.DataFrame(target_rows).to_csv(RESULTS / "target_0p80.csv", index=False)

    # ---- task-matched lift for the headline points ------------------------- #
    lift_rows = []
    for cohort, frame in frames.items():
        coh = {
            "risk": frame["risk"], "length": frame["length"],
            "suite": frame["suite"], "task": frame["task"],
        }
        ps = TML.survival_prior(coh["suite"], coh["length"], coh["risk"])
        pt = TML.survival_prior(coh["task"], coh["length"], coh["risk"])
        for chunk in (4, 8, 12, 16):
            ctrl = np.where(coh["length"] > chunk, chunk, -1)
            got = TML.score(ctrl, coh, ps, pt)
            assert abs(got["suite_lift"] - 1.0) < 1e-9
            assert abs(got["task_lift"] - 1.0) < 1e-9
            lift_rows.append({"cohort": cohort, "detector": "_control",
                              "point": f"all_running_at_chunk{chunk}",
                              "is_baseline": True, **got})
        cap_alarm = np.where(frame["length"] >= frame["cap"], 0, -1)
        lift_rows.append({"cohort": cohort, "detector": "length_leak",
                          "point": "length_at_cap", "is_baseline": False,
                          **TML.score(cap_alarm, coh, ps, pt)})
        we = frame["window_end"]
        surv_alarm = np.where(frame["length"] > we, we, -1)
        lift_rows.append({"cohort": cohort, "detector": "survival_prior",
                          "point": "alive_at_window_end", "is_baseline": True,
                          **TML.score(surv_alarm, coh, ps, pt)})

    for name in art["detectors"]:
        det_n = art["detectors"][name]
        scores = art["scores"][name]
        if P.FIT_COHORT not in scores:
            continue
        sub = budget[(budget["detector"] == name) & (budget["scope"] == "any_rule")]
        for _, pick in sub.iterrows():
            point = {
                "rule": pick["rule"], "tau": float(pick["tau"]),
                "params": {
                    "chunk_of_suite": art["best_chunk"][name],
                    "k": int(pick["k"]) if np.isfinite(pick["k"]) else 1,
                    "drift": float(pick["drift"]) if np.isfinite(pick["drift"]) else 0.0,
                },
            }
            for cohort, frame in frames.items():
                if cohort not in scores:
                    continue
                n_chunk = frame["alive"].shape[1]
                win = np.arange(n_chunk)[None, :] <= frame["window_end"][:, None]
                wmask = D.gate_mask(frame, det_n.gated) & win
                amask = (frame["alive"] & win
                         & (np.arange(n_chunk)[None, :] >= P.FIRST_SCORED_CHUNK))
                first = S.apply_point(point, scores[cohort], frame, wmask, amask)
                first = P.in_window_first(first, frame)
                coh = {"risk": frame["risk"], "length": frame["length"],
                       "suite": frame["suite"], "task": frame["task"]}
                ps = TML.survival_prior(coh["suite"], coh["length"], coh["risk"])
                pt = TML.survival_prior(coh["task"], coh["length"], coh["risk"])
                got = TML.score(first, coh, ps, pt)
                if got:
                    lift_rows.append({
                        "cohort": cohort, "detector": name,
                        "point": f"budget={pick['budget']}|{pick['rule']}",
                        "budget": pick["budget"], "is_baseline": True, **got
                    })
    lift = pd.DataFrame(lift_rows)
    lift.to_csv(RESULTS / "task_matched_lift.csv", index=False)

    print("wrote budget_table.csv, target_0p80.csv, task_matched_lift.csv")


if __name__ == "__main__":
    main()
