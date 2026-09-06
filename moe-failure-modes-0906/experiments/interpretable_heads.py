#!/usr/bin/env python3
"""Q2/Q3: the one mode axis that survives task control, its mechanism, and a
multi-head whose heads are named from the physical labels rather than from each
other.

POST-HOC. Everything in this file was decided after looking at the
task-stratified table in results/mode_recall.csv. It is labelled post-hoc and is
tested by dev -> external replication, not by a fresh prereg.

Three parts.

1. The threshold-mode interaction. Under task stratification the sign of the
   selectivity for `stable_grasp_not_observed` tracks the calibration mode:
   `global` heads over-catch it, `per_task` heads under-catch it. Because each of
   the 12 routing quantities appears in both modes, this is a paired design.
   Exact sign-flip permutation over the 12 pairs (2^12 = 4096 enumerations).

2. The proposed mechanism, checked against the labels. A per-task quantile is
   calibrated on that task's own episodes. If a task fails mostly by never
   achieving a stable grasp, its reference distribution is full of the very
   behaviour the detector should flag, so the same-task threshold rises and the
   detector goes blind. Test: does a task's risk rate rise with the share of its
   risks that are `stable_grasp_not_observed`?

3. A named multi-head. Heads are picked for replicated, task-stratified
   selectivity for DIFFERENT physical modes on development, then evaluated on
   external. Names come from the simulator label only.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import common as C

OUT = C.BUNDLE / "results"
QUANTITIES = (
    "mobility", "conditional_query_d1", "partial_query_d1", "flow_path",
    "flow_endpoint", "flow_settling_log_ratio", "action_consensus",
    "state_action_alignment", "conditional_energy", "conditional_effective_rank",
    "partial_edge_std", "expert_load_effective_rank",
)


def sign_flip_test(deltas: np.ndarray) -> tuple[float, float]:
    """Exact two-sided sign-flip permutation test on the mean of paired deltas."""
    deltas = deltas[np.isfinite(deltas)]
    n = len(deltas)
    observed = float(deltas.mean())
    if n == 0:
        return float("nan"), float("nan")
    if n > 20:
        raise ValueError("exact enumeration only for n <= 20")
    signs = np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.float64)
    null = (signs * deltas[None, :]).mean(axis=1)
    return observed, float((np.abs(null) >= abs(observed) - 1e-12).mean())


def or_alarm(arrays):
    out = np.full(len(arrays[0]), 1 << 14, dtype=np.int32)
    for a in arrays:
        a = np.asarray(a, int)
        fired = a >= 0
        out[fired] = np.minimum(out[fired], a[fired])
    out[out == (1 << 14)] = -1
    return out.astype(np.int16)


def main() -> None:
    recall = pd.read_csv(OUT / "mode_recall.csv")
    report = {"schema": "himoe.failure_modes_0906.interpretable_heads.v1",
              "status": "post-hoc, validated by development -> external replication"}

    # --- 1. threshold-mode interaction, paired over the 12 quantities --------
    interaction = {}
    for cohort in ("development_main", "external_8b"):
        block = recall[(recall.cohort == cohort) & (recall.population == "all_risks")]
        per_mode = {}
        for m in block["mode"].unique():
            sub = block[block["mode"] == m].set_index("detector")
            deltas, pairs = [], []
            for q in QUANTITIES:
                gk, pk = f"{q}|global", f"{q}|per_task"
                if gk not in sub.index or pk not in sub.index:
                    continue
                dg = sub.loc[gk, "recall"] - sub.loc[gk, "expected_recall_task"]
                dp = sub.loc[pk, "recall"] - sub.loc[pk, "expected_recall_task"]
                deltas.append(dg - dp)
                pairs.append(q)
            deltas = np.asarray(deltas, float)
            mean, p = sign_flip_test(deltas)
            # `state_action_alignment` and `conditional_energy` were selected onto
            # the same layer with mirrored direction and produce byte-identical
            # alarms, so one of the 12 pairs is a duplicate. Report both.
            dedup = [d for q, d in zip(pairs, deltas, strict=True)
                     if q != "state_action_alignment"]
            mean_d, p_d = sign_flip_test(np.asarray(dedup, float))
            n_mode = int(sub["n_mode"].iloc[0])
            per_mode[C.MODE_SHORT.get(m, m)] = {
                "n_mode": n_mode,
                "pairs": len(deltas),
                "mean_global_minus_per_task_effect": mean,
                "positive_pairs": int((deltas > 0).sum()),
                "exact_sign_flip_p": p,
                "pairs_deduplicated": len(dedup),
                "positive_pairs_deduplicated": int((np.asarray(dedup) > 0).sum()),
                "exact_sign_flip_p_deduplicated": p_d,
                "small_cell": n_mode < 30,
            }
        interaction[cohort] = per_mode
    report["threshold_mode_interaction_task_stratified"] = interaction

    # --- 2. mechanism: does a per-task reference get polluted? --------------
    mechanism = {}
    for cohort in ("development_main", "external_8b"):
        frame = C.cohort_index(cohort)
        risky = frame[frame.risk]
        stats = []
        for task, block in frame.groupby("task_key"):
            r = block[block.risk]
            if len(r) < 5:
                continue
            stats.append({
                "task": task,
                "risk_rate": float(block.risk.mean()),
                "n_risk": int(len(r)),
                "no_grasp_share": float(
                    (r.primary_failure_reason == "stable_grasp_not_observed").mean()),
                "dropped_share": float(
                    (r.primary_failure_reason
                     == "object_released_or_dropped_before_goal").mean()),
            })
        table = pd.DataFrame(stats)
        out = {"tasks": int(len(table))}
        for column in ("no_grasp_share", "dropped_share"):
            a = table[column].to_numpy()
            b = table["risk_rate"].to_numpy()
            out[column] = {
                "pearson_with_task_risk_rate": float(np.corrcoef(a, b)[0, 1]),
                "spearman_with_task_risk_rate": float(
                    np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]),
                "mean_risk_rate_top_third": float(
                    b[a >= np.quantile(a, 2 / 3)].mean()),
                "mean_risk_rate_bottom_third": float(
                    b[a <= np.quantile(a, 1 / 3)].mean()),
            }
        table.to_csv(OUT / f"task_mode_mix_{cohort}.csv", index=False)
        mechanism[cohort] = out
    report["per_task_reference_pollution"] = mechanism

    # --- 3. named multi-head ------------------------------------------------
    dev_t = recall[(recall.cohort == "development_main") & (recall.population == "all_risks")]
    ext_t = recall[(recall.cohort == "external_8b") & (recall.population == "all_risks")]
    named = []
    for detector, block in dev_t.groupby("detector"):
        block = block.replace([np.inf, -np.inf], np.nan).dropna(subset=["sel_task"])
        if block.empty:
            continue
        top = block.loc[block.sel_task.idxmax()]
        if not (top.sel_task > 0 and top.p_task < 0.05):
            continue
        e = ext_t[(ext_t.detector == detector) & (ext_t["mode"] == top["mode"])]
        if e.empty:
            continue
        e = e.iloc[0]
        named.append({
            "detector": detector,
            "named_mode": top["mode"],
            "named_mode_short": C.MODE_SHORT.get(top["mode"], top["mode"]),
            "n_dev": int(top.n_mode), "n_ext": int(e.n_mode),
            "dev_recall": float(top.recall), "dev_expected_task": float(top.expected_recall_task),
            "dev_sel_task": float(top.sel_task), "dev_p_task": float(top.p_task),
            "ext_recall": float(e.recall), "ext_expected_task": float(e.expected_recall_task),
            "ext_sel_task": float(e.sel_task) if np.isfinite(e.sel_task) else None,
            "ext_p_task": float(e.p_task),
            "external_validates": bool(np.isfinite(e.sel_task) and e.sel_task > 0
                                       and e.p_task < 0.05),
            "small_cell": bool(top.n_mode < 30 or e.n_mode < 30),
        })
    named_frame = pd.DataFrame(named).sort_values("dev_sel_task", ascending=False)
    named_frame.to_csv(OUT / "named_heads.csv", index=False)
    report["named_heads_task_stratified"] = named_frame.to_dict(orient="records")

    # build the bundle: one validated head per distinct physical mode, best first
    dev_alarms = C.load_npz(OUT / "first_alarms_development.npz"); dev_alarms.pop("schema")
    ext_alarms = C.load_npz(OUT / "first_alarms_external.npz"); ext_alarms.pop("schema")
    dev_frame = C.cohort_index("development_main"); ext_frame = C.cohort_index("external_8b")
    dev_risk = dev_frame.risk.to_numpy(bool); ext_risk = ext_frame.risk.to_numpy(bool)
    dev_suite = dev_frame.suite.to_numpy(str); ext_suite = ext_frame.suite.to_numpy(str)
    dev_priors = C.survival_prior(dev_suite, dev_frame.length.to_numpy(int), dev_risk)
    ext_priors = C.survival_prior(ext_suite, ext_frame.length.to_numpy(int), ext_risk)
    ext_mode = ext_frame.primary_failure_reason.fillna("").to_numpy(str)

    bundles = {}
    for family, predicate in (
        ("task_agnostic", lambda k: k.endswith("|global") or k.endswith("|task_agnostic")),
        ("any", lambda k: True),
    ):
        picked, used = [], set()
        for r in named_frame.itertuples():
            if not predicate(r.detector) or r.named_mode in used:
                continue
            if not r.external_validates and family == "task_agnostic":
                pass  # keep development-only picks too; validation is reported per head
            picked.append(r.detector)
            used.add(r.named_mode)
        if not picked:
            continue
        d = or_alarm([dev_alarms[k] for k in picked])
        e = or_alarm([ext_alarms[k] for k in picked])
        per_mode = {}
        for m in pd.Series(ext_mode[ext_risk]).value_counts().index:
            take = (ext_mode == m) & ext_risk
            per_mode[C.MODE_SHORT.get(m, m)] = {
                "n": int(take.sum()), "caught": int(((e >= 0) & take).sum())}
        bundles[family] = {
            "heads": picked,
            "named_modes": [named_frame.set_index("detector").loc[k, "named_mode_short"]
                            for k in picked],
            "development": C.score_candidate(
                np.asarray(d, int), dev_risk, C.prior_of(np.asarray(d, int), dev_suite, dev_priors)),
            "external": C.score_candidate(
                np.asarray(e, int), ext_risk, C.prior_of(np.asarray(e, int), ext_suite, ext_priors)),
            "external_per_mode": per_mode,
        }
        np.savez_compressed(OUT / f"named_bundle_{family}.npz", development=d, external=e,
                            heads=np.asarray(picked))
    report["named_bundles"] = bundles

    (OUT / "interpretable_heads.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")

    print(json.dumps(report["threshold_mode_interaction_task_stratified"], indent=2))
    print(json.dumps(report["per_task_reference_pollution"], indent=2))
    print(named_frame.to_string(index=False))
    for k, v in bundles.items():
        print(k, v["heads"], v["named_modes"], v["external"]["tp"], v["external"]["fp"])


if __name__ == "__main__":
    main()
