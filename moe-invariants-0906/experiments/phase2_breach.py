"""Phase 2: do the frozen invariants BREAK before failure?

Runs only after phase 1 is frozen; the sha256 of the frozen candidate list is
verified before a single label is loaded, and no relation may be added here.

For each invariant the breach series is the fitted residual, standardised by
its own development_main scale:

    z[q, c] = ( y - b.x - a ) / sd_dev ,   b, a fitted on development_main

and the detector is "first chunk with |z| >= tau" (also the two one-sided
forms, and a two-consecutive-chunks form).  Detector input is therefore chunk
index + routing only: no phase, no task identity, no per-task threshold, and
no horizon cap.  Scoring is the cap-free protocol, imported unchanged.

Selection is honest: every configuration is swept on development_main, the one
with the largest excess TP over the cap-free baseline at lead >= 4 is chosen,
and external_8b is scored once.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parent
ROOT = BUNDLE.parent
OUT = BUNDLE / "results"
BANK = OUT / "bank"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "moe-capfree-0906" / "experiments"))
sys.path.insert(0, str(ROOT / "moe-prior-correction-0906" / "experiments"))

from phase1_definitional import family  # noqa: E402
import capfree_protocol as cp  # noqa: E402
from recompute_task_matched_lift import cohort_frame  # noqa: E402

COHORTS = ("development_main", "external_8b")
TAU_Q = (0.90, 0.95, 0.975, 0.99, 0.995, 0.999)
MODES = ("abs", "pos", "neg")
PERSIST = (1, 2)
HEADLINE_LEAD = cp.HEADLINE_LEAD
SEED = 20260906
N_EMPIRICAL = 6


# --------------------------------------------------------------------------
def verify_freeze() -> dict:
    p = OUT / "phase1_frozen_candidates.json"
    meta = json.loads((OUT / "phase1_freeze.json").read_text())
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != meta["sha256"]:
        raise SystemExit(f"frozen list changed since phase 1: {got} != {meta['sha256']}")
    return json.loads(p.read_text())


def pick_relations(frozen: dict) -> list[dict]:
    """Fixed rule, applied to the frozen ranking; nothing is chosen by eye."""
    out, seen = [], set()
    for c in frozen["candidates"]:
        if not c["category"].startswith("empirical"):
            continue
        key = tuple(sorted([family(c["target"])] + [family(p) for p in c["preds"]]))
        if key in seen:
            continue
        seen.add(key)
        out.append({**c, "role": "empirical_invariant"})
        if len(out) >= N_EMPIRICAL:
            break
    for c in frozen["candidates"]:
        if c["category"] == "definitional:entropy_identity" and \
                c["target"] in ("sp.load_entropy@L2[s0]", "sp.load_entropy@L12[s0]"):
            out.append({**c, "role": "positive_control_exact_identity"})
    # definitional NEAR-identities: real residuals with a known origin
    for ly in ("L2", "L12"):
        out.append({"id": f"T_{ly}", "kind": "pair",
                    "target": f"sp.token_differentiation@{ly}[s0]",
                    "preds": [f"sp.action_consensus@{ly}[s0]"],
                    "category": "definitional:smooth_equivalent",
                    "role": "definitional_near_identity_taylor"})
        out.append({"id": f"E_{ly}", "kind": "pair",
                    "target": f"sp.conditional_energy@{ly}[s0]",
                    "preds": [f"sp.state_action_alignment@{ly}[s0]"],
                    "category": "definitional:smooth_equivalent",
                    "role": "definitional_near_identity_envelope"})
    # pure float-noise residual: the exact flow_path identity
    out.append({"id": "N_L2", "kind": "pair", "target": "lg.flow_path@L2[agg]",
                "preds": ["sp.flow_speed@L2[mean]"],
                "category": "cache_duplicate",
                "role": "negative_control_float_noise"})
    return out


# --------------------------------------------------------------------------
def residual_grids(rels: list[dict]) -> dict:
    banks = {c: np.load(BANK / f"{c}_bank.npz", allow_pickle=True) for c in COHORTS}
    names = [str(s) for s in banks["development_main"]["var_names"]]
    col = {n: i for i, n in enumerate(names)}
    dev = banks["development_main"]
    Xd = dev["X"].astype(np.float64)
    grids: dict[str, dict] = {}
    for r in rels:
        ti, pi = col[r["target"]], [col[p] for p in r["preds"]]
        A = np.c_[Xd[:, pi], np.ones(len(Xd))]
        b, *_ = np.linalg.lstsq(A, Xd[:, ti], rcond=None)
        res_dev = Xd[:, ti] - A @ b
        mu, sd = res_dev.mean(), res_dev.std()
        rel = float(sd / Xd[:, ti].std())
        # An EXACT relation has no residual, so it has no breach series and
        # cannot be a detector at any threshold.  That is the finding, not a
        # bug: perfect tightness and zero discriminative information are the
        # same statement.  Anything below float32 resolution is treated so.
        r["fit"] = {"coefs": [float(x) for x in b[:-1]], "intercept": float(b[-1]),
                    "residual_sd_dev_main": float(sd),
                    "rel_resid_dev_main": rel,
                    "unbreakable": bool(rel < 1e-6)}
        g = {}
        for c in COHORTS:
            bk = banks[c]
            X = bk["X"].astype(np.float64)
            A2 = np.c_[X[:, pi], np.ones(len(X))]
            z = (X[:, ti] - A2 @ b - mu) / max(sd, 1e-300)
            n = int(bk["row_query"].max()) + 1
            grid = np.full((n, 52), np.nan)
            grid[bk["row_query"], bk["row_chunk"]] = z
            g[c] = grid
        grids[r["id"]] = g
    return grids


def first_alarm(grid: np.ndarray, tau: float, mode: str, persist: int) -> np.ndarray:
    s = np.abs(grid) if mode == "abs" else (grid if mode == "pos" else -grid)
    hit = np.where(np.isfinite(s), s >= tau, False)
    if persist > 1:
        run = hit.copy()
        for k in range(1, persist):
            run[:, k:] &= hit[:, :-k]
        hit = run
    any_hit = hit.any(axis=1)
    return np.where(any_hit, hit.argmax(axis=1), -1)


# --------------------------------------------------------------------------
def main() -> None:
    frozen = verify_freeze()
    rels = pick_relations(frozen)
    grids = residual_grids(rels)

    frames = {c: cohort_frame(c) for c in COHORTS}
    align = {}
    for c in COHORTS:
        n_bank = grids[rels[0]["id"]][c].shape[0]
        assert n_bank == len(frames[c]["risk"]), (c, n_bank, len(frames[c]["risk"]))
        ix = np.load(ROOT / "moe-flow-semantics-0906" / "results" /
                     "step_profiles" / f"{c}_index.npz", allow_pickle=True)
        valid = np.asarray(ix["valid"])
        assert np.array_equal(valid.sum(axis=1), frames[c]["length"]), c
        align[c] = {"n_episodes": int(n_bank),
                    "valid_chunks_equal_length": True,
                    "n_risk": int(frames[c]["risk"].sum())}
    bases = {c: cp.fixed_chunk_baseline(frames[c]["risk"], frames[c]["length"])
             for c in COHORTS}
    rng = np.random.default_rng(SEED)

    rows = []
    for r in rels:
        if r["fit"]["unbreakable"]:
            for c in COHORTS:
                n = len(frames[c]["risk"])
                sc = cp.score(np.full(n, -1), frames[c]["risk"], frames[c]["length"])
                rows.append({
                    "relation": r["id"], "role": r["role"],
                    "category": r["category"], "target": r["target"],
                    "preds": "+".join(r["preds"]), "cohort": c,
                    "tau_q": np.nan, "tau": np.nan, "mode": "none", "persist": 0,
                    "rel_resid_dev_main": r["fit"]["rel_resid_dev_main"],
                    **sc, "baseline_tp_at_same_fp": 0, "excess_tp": 0})
            continue
        tau_grid = np.quantile(np.abs(grids[r["id"]]["development_main"][
            np.isfinite(grids[r["id"]]["development_main"])]), TAU_Q)
        for q, tau in zip(TAU_Q, tau_grid):
            for mode in MODES:
                for persist in PERSIST:
                    for c in COHORTS:
                        first = first_alarm(grids[r["id"]][c], float(tau), mode, persist)
                        sc = cp.score(first, frames[c]["risk"], frames[c]["length"])
                        base = cp.baseline_frontier(bases[c], HEADLINE_LEAD)(
                            sc[f"fp_lead{HEADLINE_LEAD}"])
                        rows.append({
                            "relation": r["id"], "role": r["role"],
                            "category": r["category"], "target": r["target"],
                            "preds": "+".join(r["preds"]), "cohort": c,
                            "tau_q": q, "tau": float(tau), "mode": mode,
                            "persist": persist,
                            "rel_resid_dev_main": r["fit"]["rel_resid_dev_main"],
                            **sc,
                            "baseline_tp_at_same_fp": int(base),
                            "excess_tp": int(sc[f"tp_lead{HEADLINE_LEAD}"] - base),
                        })
    census = pd.DataFrame(rows)
    census.to_csv(OUT / "phase2_breach_census.csv", index=False)

    # ---- honest arm ------------------------------------------------------
    dev = census[(census.cohort == "development_main") &
                 (census.role == "empirical_invariant")]
    best = dev.loc[dev.excess_tp.idxmax()]
    key = ["relation", "tau_q", "mode", "persist"]
    ext = census[(census.cohort == "external_8b")]
    sel = ext
    for k in key:
        sel = sel[sel[k] == best[k]]
    sel = sel.iloc[0]

    # rate-matched null for the selected configuration, on external
    r = next(x for x in rels if x["id"] == best["relation"])
    first_ext = first_alarm(grids[r["id"]]["external_8b"], float(best["tau"]),
                            best["mode"], int(best["persist"]))
    nulls = []
    for _ in range(20):
        nf = cp.rate_matched_null(first_ext, rng, frames["external_8b"]["length"])
        s = cp.score(nf, frames["external_8b"]["risk"], frames["external_8b"]["length"])
        b = cp.baseline_frontier(bases["external_8b"], HEADLINE_LEAD)(
            s[f"fp_lead{HEADLINE_LEAD}"])
        nulls.append(s[f"tp_lead{HEADLINE_LEAD}"] - b)

    n_risk_ext = int(frames["external_8b"]["risk"].sum())
    n_risk_dev = int(frames["development_main"]["risk"].sum())
    summary = {
        "headline_lead": HEADLINE_LEAD,
        "alignment_audit": align,
        "n_risk": {"development_main": n_risk_dev, "external_8b": n_risk_ext},
        "n_configs_swept_on_development": int(len(dev)),
        "selected_on_development": {k: (best[k] if not isinstance(best[k], np.generic)
                                        else best[k].item()) for k in
                                    key + ["target", "preds", "tau",
                                           "rel_resid_dev_main", "excess_tp"]},
        "external_once": {
            "tp": int(sel[f"tp_lead{HEADLINE_LEAD}"]),
            "fp": int(sel[f"fp_lead{HEADLINE_LEAD}"]),
            "n_risk": n_risk_ext,
            "alarms": int(sel["alarms"]),
            "median_lead": float(sel["median_lead"]),
            "baseline_tp_at_same_fp": int(sel["baseline_tp_at_same_fp"]),
            "excess_tp": int(sel["excess_tp"]),
        },
        "rate_matched_null_excess_tp": {
            "best_of_20": int(max(nulls)), "median": float(np.median(nulls))},
        "capfree_baseline_reference": {
            q: {"tp": int(bases["external_8b"].loc[
                    bases["external_8b"].q0 == q, f"tp_lead{HEADLINE_LEAD}"].iloc[0]),
                "fp": int(bases["external_8b"].loc[
                    bases["external_8b"].q0 == q, f"fp_lead{HEADLINE_LEAD}"].iloc[0])}
            for q in (20, 28, 32, 37)},
    }
    # what each role achieves at its own best development configuration
    per_role = {}
    for role in census.role.unique():
        d = census[(census.role == role) & (census.cohort == "development_main")]
        if d.empty:
            continue
        if d.tau_q.isna().all():
            per_role[role] = {
                "relation": str(d.relation.iloc[0]), "target": str(d.target.iloc[0]),
                "preds": str(d.preds.iloc[0]),
                "rel_resid_dev_main": float(d.rel_resid_dev_main.iloc[0]),
                "unbreakable": True,
                "note": "the residual is zero to float32, so there is no breach "
                        "series at any threshold and the relation can carry no "
                        "detection value by construction",
                "dev_tp": 0, "dev_fp": 0, "dev_excess_tp": 0,
                "ext_tp": 0, "ext_fp": 0, "ext_alarms": 0, "ext_excess_tp": 0,
                "ext_median_lead": None}
            continue
        b = d.loc[d.excess_tp.idxmax()]
        e = census[(census.cohort == "external_8b") & (census.relation == b.relation) &
                   (census.tau_q == b.tau_q) & (census["mode"] == b["mode"]) &
                   (census.persist == b.persist)].iloc[0]
        per_role[role] = {
            "relation": b.relation, "target": b.target, "preds": b.preds,
            "rel_resid_dev_main": float(b.rel_resid_dev_main),
            "dev_tp": int(b[f"tp_lead{HEADLINE_LEAD}"]),
            "dev_fp": int(b[f"fp_lead{HEADLINE_LEAD}"]),
            "dev_excess_tp": int(b.excess_tp),
            "ext_tp": int(e[f"tp_lead{HEADLINE_LEAD}"]),
            "ext_fp": int(e[f"fp_lead{HEADLINE_LEAD}"]),
            "ext_alarms": int(e.alarms),
            "ext_excess_tp": int(e.excess_tp),
            "ext_median_lead": float(e.median_lead) if e.alarms else None,
        }
    summary["per_role_best"] = per_role

    per_rel = {}
    for rid in census.relation.unique():
        d = census[(census.relation == rid) &
                   (census.cohort == "development_main")]
        r = next(x for x in rels if x["id"] == rid)
        if r["fit"]["unbreakable"]:
            per_rel[rid] = {"role": r["role"], "target": r["target"],
                            "preds": "+".join(r["preds"]),
                            "rel_resid_dev_main": r["fit"]["rel_resid_dev_main"],
                            "unbreakable": True,
                            "note": "residual is zero to float32; the relation "
                                    "has no breach series at any threshold"}
            continue
        b = d.loc[d.excess_tp.idxmax()]
        e = census[(census.cohort == "external_8b") & (census.relation == rid) &
                   (census.tau_q == b.tau_q) & (census["mode"] == b["mode"]) &
                   (census.persist == b.persist)].iloc[0]
        g = grids[rid]["external_8b"]
        fin = np.isfinite(g)
        br = np.where(fin, np.abs(g) >= float(b.tau), False)
        by_chunk = [int(br[:, c].sum()) for c in range(g.shape[1])]
        per_rel[rid] = {
            "role": r["role"], "target": r["target"],
            "preds": "+".join(r["preds"]),
            "rel_resid_dev_main": r["fit"]["rel_resid_dev_main"],
            "unbreakable": False,
            "best_dev_config": {"tau_q": float(b.tau_q), "mode": str(b["mode"]),
                                "persist": int(b.persist)},
            "dev": {"tp": int(b[f"tp_lead{HEADLINE_LEAD}"]),
                    "fp": int(b[f"fp_lead{HEADLINE_LEAD}"]),
                    "excess_tp": int(b.excess_tp)},
            "external": {"tp": int(e[f"tp_lead{HEADLINE_LEAD}"]),
                         "fp": int(e[f"fp_lead{HEADLINE_LEAD}"]),
                         "alarms": int(e.alarms),
                         "excess_tp": int(e.excess_tp),
                         "median_lead": float(e.median_lead) if e.alarms else None},
            "external_breach_cells": int(br.sum()),
            "external_episodes_with_breach": int(br.any(axis=1).sum()),
            "external_breach_count_by_chunk": by_chunk,
        }
    summary["per_relation"] = per_rel
    summary["relations_tested"] = [
        {k: v for k, v in r.items() if k in
         ("id", "role", "category", "target", "preds", "fit")} for r in rels]
    (OUT / "phase2_breach_summary.json").write_text(json.dumps(summary, indent=1))

    print(json.dumps({k: summary[k] for k in
                      ("selected_on_development", "external_once",
                       "rate_matched_null_excess_tp")}, indent=1))
    print("\n--- best configuration per role (dev-selected, external scored once)")
    for role, v in per_role.items():
        tag = " [UNBREAKABLE: no residual]" if v.get("unbreakable") else ""
        print("  %-38s rr=%.4f  dev TP %4d/FP %5d (excess %+d) | ext TP %4d/FP %5d "
              "(excess %+d, alarms %d)%s"
              % (role, v["rel_resid_dev_main"], v["dev_tp"], v["dev_fp"],
                 v["dev_excess_tp"], v["ext_tp"], v["ext_fp"], v["ext_excess_tp"],
                 v["ext_alarms"], tag))
    print("\n--- every frozen relation tested (external, dev-selected config)")
    for rid, v in summary["per_relation"].items():
        if v.get("unbreakable"):
            print("  %-6s rr=%.5f  UNBREAKABLE  %s ~ %s"
                  % (rid, v["rel_resid_dev_main"], v["target"], v["preds"]))
            continue
        e = v["external"]
        print("  %-6s rr=%.4f  ext TP %3d/%d  FP %5d  alarms %5d  excess %+d  %s ~ %s"
              % (rid, v["rel_resid_dev_main"], e["tp"], summary["n_risk"]["external_8b"],
                 e["fp"], e["alarms"], e["excess_tp"], v["target"], v["preds"]))


if __name__ == "__main__":
    main()
