"""Stage 4 - score `external_8b` ONCE with the frozen configuration.

Everything applied here was fitted on `development_main` in
`fit_composite.py`: the channel sets, the weights, the per-chunk
normalisation constants, `k`, `h`, `K`, `M`, `q_min`.  Nothing is re-fitted or
re-selected against external.  The full external frontier is also traced, but
only as description: the frozen operating point is marked separately and is the
only number the head-to-head claim rests on.

Mandatory controls, all run here:
  * rate-matched null, white-noise null and episode-constant null,
  * length as a *negative control* (`is_baseline = False`) - risk is defined
    as not finishing before the cap, so a sub-cap length threshold recalls
    100% by construction and needs the cap besides,
  * `>=` tie handling everywhere, with the tie group at the threshold checked.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bank  # noqa: E402
import capfree_common as C  # noqa: E402
import detectors as D  # noqa: E402

DEV, EXT = "development_main", "external_8b"
LEAD = C.HEADLINE_LEAD
ALARM_COUNTS = np.unique(np.round(np.geomspace(20, 6000, 60)).astype(int))
NULL_SEEDS = (0, 1, 2, 3, 4)


def build_z(cohort, frames, stats_dev, cfg, pool_names, idx, Zcache):
    """Composite score for one frozen family config, on one cohort."""
    valid = frames[cohort]["valid"]
    s = np.zeros((frames[cohort]["n"], C.N_CHUNKS), np.float32)
    for nm, w in zip(cfg["channels"], cfg["weights"]):
        chan, arm, sign = nm.rsplit("|", 2)
        z = Zcache[(cohort, arm)][idx[chan]]
        s += np.float32(w) * (1.0 if sign == "+" else -1.0) * z
    mu = np.array(cfg["norm_mu"])
    sd = np.array(cfg["norm_sd"])
    z = (s - mu[None, :]) / sd[None, :]
    z = np.where(np.isfinite(z), z, 0.0)
    return np.where(valid, z, 0.0).astype(np.float32)


def stat_of(family, z, valid, op):
    """The running statistic the family thresholds."""
    if family.startswith("cusum"):
        return D.cusum_path(z, valid, op["k"])
    if family == "kofm":
        return D.kofm_path(z, valid, op["thr"], int(op["M"])).astype(np.float32)
    return z


def thr_of(family, op):
    return float(op["K"]) if family == "kofm" else float(op["thr"])


def main() -> None:
    t0 = time.time()
    cfgall = json.loads((C.RESULTS / "frozen_config.json").read_text())
    frames = {c: C.frame(c) for c in (DEV, EXT)}
    names_dev, X_dev = bank.build(DEV)
    stats_dev = bank.fit_norm(X_dev, frames[DEV]["valid"])
    names_ext, X_ext = bank.build(EXT)
    assert names_dev == names_ext, "channel order differs between cohorts"
    idx = {n: i for i, n in enumerate(names_dev)}
    Zcache = {}
    for arm in bank.ARMS:
        Zcache[(DEV, arm)] = bank.apply_norm(X_dev, frames[DEV]["valid"],
                                             stats_dev, arm)
        Zcache[(EXT, arm)] = bank.apply_norm(X_ext, frames[EXT]["valid"],
                                             stats_dev, arm)
    del X_dev, X_ext
    print(f"bank ready {time.time() - t0:.0f}s")

    pool = np.load(C.RESULTS / "frozen_pool.npz", allow_pickle=True)
    pool_names = list(pool["pool_names"].astype(str))
    base = {c: C.fixed_chunk_baseline(frames[c]["risk"], frames[c]["length"])
            for c in frames}
    base_at = {c: {b: C.baseline_frontier(base[c], b) for b in C.LEADS}
               for c in frames}
    hull_at = {c: {b: C.hull_frontier(base[c], b) for b in C.LEADS}
               for c in frames}
    rng = np.random.default_rng(C.SEED)

    rows, frontier_rows, ties, suite_rows = [], [], [], []

    def by_suite(cohort, arm, first):
        """Diagnostic only - suite never enters a detector decision.  It is
        reported because an arm that recalls one suite and nothing else is a
        task detector, not a failure detector."""
        f_ = frames[cohort]
        lead = np.where(first >= 0, f_["length"] - first, -1)
        timely = (first >= 0) & (lead >= LEAD)
        out = {"cohort": cohort, "arm": arm}
        for s in np.unique(f_["suite"]):
            m = f_["suite"] == s
            out[f"tp_{s}"] = int((timely & f_["risk"] & m).sum())
            out[f"risk_{s}"] = int((f_["risk"] & m).sum())
        return out

    def emit(cohort, arm, first, extra):
        s = C.score(first, frames[cohort]["risk"], frames[cohort]["length"])
        s.update({"cohort": cohort, "arm": arm, **extra})
        for b in C.LEADS:
            s[f"excess_lead{b}"] = int(s[f"tp_lead{b}"]
                                       - base_at[cohort][b](s[f"fp_lead{b}"]))
            s[f"excess_hull_lead{b}"] = round(
                s[f"tp_lead{b}"] - hull_at[cohort][b](s[f"fp_lead{b}"]), 1)
        s["excess_tp"] = s[f"excess_lead{LEAD}"]
        s["excess_hull_tp"] = s[f"excess_hull_lead{LEAD}"]
        return s

    # ---- 1. frozen MoE arms -------------------------------------------------
    for mode in ("window", "v7budget"):
        for family, cfg in cfgall["selections"][mode]["families"].items():
            op = cfg["operating_point"]
            if op is None:
                continue
            for cohort in (DEV, EXT):
                valid = frames[cohort]["valid"]
                z = build_z(cohort, frames, stats_dev, cfg, pool_names, idx,
                            Zcache)
                stat = stat_of(family, z, valid, op)
                thr = thr_of(family, op)
                qm = int(op["q_min"])
                first = D.first_crossing(stat, valid, thr, qm)
                strict = np.where(((stat > thr) & valid).any(axis=1),
                                  ((stat > thr) & valid).argmax(axis=1), -1)
                if qm:
                    m = (stat > thr) & valid
                    m[:, :qm] = False
                    strict = np.where(m.any(axis=1), m.argmax(axis=1), -1)
                ties.append({
                    "cohort": cohort, "mode": mode, "family": family,
                    "n_tie_cells": int((valid & (stat == thr)).sum()),
                    "n_episodes_lost_with_strict_gt":
                        int(((first >= 0) & (strict < 0)).sum()),
                    "n_episodes_delayed_with_strict_gt":
                        int(((first >= 0) & (strict > first)).sum()),
                })
                rows.append(emit(cohort, f"{family}|{mode}", first,
                                 {"is_baseline": False, "family": family,
                                  "mode": mode, "frozen": True,
                                  "params": json.dumps(
                                      {k: op[k] for k in
                                       ("k", "K", "M", "thr", "q_min")
                                       if k in op}),
                                  "n_channels": len(cfg["channels"])}))
                suite_rows.append(by_suite(cohort, f"{family}|{mode}", first))
                # Descriptive frontier.  Only the decision interval moves;
                # channels, weights, k / M and q_min stay frozen.  For K-of-M
                # both the per-chunk threshold and K are traced, otherwise its
                # curve would be a handful of points and the comparison unfair.
                if family == "kofm":
                    for zt in np.quantile(z[valid],
                                          np.linspace(0.3, 0.995, 24)):
                        cnt = D.kofm_path(z, valid, zt,
                                          int(op["M"])).astype(np.float32)
                        cm2 = D.cummax_valid(cnt, valid, qm)
                        for KK in range(1, int(op["M"]) + 1):
                            f1 = D.first_from_cummax(cm2, KK)
                            if (f1 >= 0).sum() == 0:
                                continue
                            frontier_rows.append(emit(
                                cohort, f"{family}|{mode}", f1,
                                {"family": family, "mode": mode,
                                 "thr_swept": float(zt), "K_swept": KK,
                                 "is_baseline": False}))
                else:
                    cm = D.cummax_valid(stat, valid, qm)
                    for t in D.alarm_count_grid(cm, ALARM_COUNTS):
                        f1 = D.first_from_cummax(cm, t)
                        if (f1 >= 0).sum() == 0:
                            continue
                        frontier_rows.append(emit(
                            cohort, f"{family}|{mode}", f1,
                            {"family": family, "mode": mode,
                             "thr_swept": float(t), "is_baseline": False}))
                if mode == "window" and family == "cusum" and cohort == EXT:
                    np.savez_compressed(C.RESULTS / "external_first_alarms.npz",
                                        cusum_window=first)

    # ---- 2. v7_guard, the anchor -------------------------------------------
    for cohort in (DEV, EXT):
        dets = C.cp.load_detectors(cohort, frames[cohort]["n"])
        gk = next(k for k in dets if "v7_guard" in k)
        r = emit(cohort, "v7_guard", dets[gk],
                 {"is_baseline": False, "family": "v7_guard", "mode": "anchor",
                  "frozen": True, "params": "published", "n_channels": 4})
        rows.append(r)
        frontier_rows.append({**r, "thr_swept": np.nan})
        suite_rows.append(by_suite(cohort, "v7_guard", dets[gk]))
        # rate-matched null of the anchor
        rows.append(emit(cohort, "null_rate_matched|v7_guard",
                         C.rate_matched_null(dets[gk], rng,
                                             frames[cohort]["length"]),
                         {"is_baseline": False, "family": "null_arm",
                          "mode": "rate_matched", "frozen": True}))

    # ---- 3. cap-free fixed-chunk baseline ----------------------------------
    for cohort in (DEV, EXT):
        length = frames[cohort]["length"]
        for q0 in C.cp.Q0_GRID:
            first = np.where(length > q0, q0, -1)
            r = emit(cohort, f"still_running_q{q0}", first,
                     {"is_baseline": True, "family": "baseline",
                      "mode": "capfree", "thr_swept": float(q0)})
            rows.append(r)
            frontier_rows.append(r)

    # ---- 4. length: NEGATIVE CONTROL, never a baseline ----------------------
    # Risk == "did not finish before the cap", so any sub-cap length threshold
    # recalls 100% by construction, and it needs the cap.  Doubly excluded.
    for cohort in (DEV, EXT):
        length = frames[cohort]["length"]
        for q0 in (20, 30, 40):
            first = np.where(length > q0, np.minimum(q0, length - 1), -1)
            rows.append(emit(cohort, f"NEGCTRL_length_gt_{q0}", first,
                             {"is_baseline": False, "family": "negative_control",
                              "mode": "length", "frozen": True}))

    # ---- 5. nulls: white noise and episode-constant, identical pipeline -----
    null_rows = []
    for mode in ("window", "v7budget"):
        cfg = cfgall["selections"][mode]["families"]["cusum"]
        op = cfg["operating_point"]
        nch = len(cfg["channels"])
        for cohort in (DEV, EXT):
            valid = frames[cohort]["valid"]
            n = frames[cohort]["n"]
            for kind in ("white_noise", "episode_constant"):
                for seed in NULL_SEEDS:
                    g = np.random.default_rng(1000 * seed + hash(kind) % 997)
                    if kind == "white_noise":
                        Zn = g.standard_normal((nch, n, C.N_CHUNKS), np.float32)
                    else:
                        Zn = np.repeat(
                            g.standard_normal((nch, n, 1), np.float32),
                            C.N_CHUNKS, axis=2)
                    s = np.tensordot(
                        np.array(cfg["weights"], np.float32), Zn, axes=(0, 0))
                    mu, sd = C.chunk_stats(s, valid)
                    z = np.where(valid, np.nan_to_num(
                        (s - mu[None, :]) / sd[None, :]), 0.0).astype(np.float32)
                    S = D.cusum_path(z, valid, op["k"])
                    cm = D.cummax_valid(S, valid, int(op["q_min"]))
                    for t in D.alarm_count_grid(cm, ALARM_COUNTS):
                        f1 = D.first_from_cummax(cm, t)
                        if (f1 >= 0).sum() == 0:
                            continue
                        null_rows.append(emit(
                            cohort, f"null_{kind}|{mode}", f1,
                            {"family": "null_arm", "mode": f"{kind}|{mode}",
                             "seed": seed, "thr_swept": float(t),
                             "is_baseline": False}))
    nl = pd.DataFrame(null_rows)
    nl.to_csv(C.RESULTS / "null_sweep.csv", index=False)
    frontier_rows += null_rows

    df = pd.DataFrame(rows)
    df.to_csv(C.RESULTS / "frozen_operating_points.csv", index=False)
    pd.DataFrame(frontier_rows).to_csv(C.RESULTS / "frontier_all.csv",
                                       index=False)
    pd.DataFrame(ties).to_csv(C.RESULTS / "tie_audit.csv", index=False)
    sb = pd.DataFrame(suite_rows)
    for cohort in (DEV, EXT):
        b37 = np.where(frames[cohort]["length"] > 37, 37, -1)
        sb = pd.concat([sb, pd.DataFrame(
            [by_suite(cohort, "still_running_q37", b37)])], ignore_index=True)
    sb.to_csv(C.RESULTS / "suite_breakdown.csv", index=False)

    # ---- 6. headline -------------------------------------------------------
    ext = df[(df.cohort == EXT)]
    fr = pd.DataFrame(frontier_rows)
    fre = fr[fr.cohort == EXT]

    def pack(r, posthoc=False):
        return {"tp_lead4": int(r.tp_lead4), "fp_lead4": int(r.fp_lead4),
                "tp_lead8": int(r.tp_lead8), "fp_lead8": int(r.fp_lead8),
                "tp": int(r.tp), "fp": int(r.fp),
                "median_lead": float(r.median_lead),
                "excess_tp": int(r.excess_tp),
                "excess_hull_tp": float(r.excess_hull_tp),
                "beats_347_57": bool(r.tp_lead4 > 347 and r.fp_lead4 <= 57),
                "post_hoc_threshold": posthoc,
                "params": r.get("params", f"thr={r.get('thr_swept')}")}

    v7 = ext[ext.arm == "v7_guard"].iloc[0]
    out = {"anchor_v7_guard_external": pack(v7)}
    for mode in ("window", "v7budget"):
        for fam in ("cusum", "cusum_k>=0", "kofm", "single"):
            sub = ext[ext.arm == f"{fam}|{mode}"]
            if not len(sub):
                continue
            out[f"{fam}|{mode}"] = pack(sub.iloc[0])
            # descriptive: same frozen channels/weights/k, threshold read off
            # the external curve at v7's own false-alarm count.  Labelled
            # post-hoc; it selects nothing, it only answers "where does this
            # frozen detector sit at FP <= 57".
            s2 = fre[(fre.arm == f"{fam}|{mode}") & (fre.fp_lead4 <= 57)]
            if len(s2):
                out[f"{fam}|{mode}@fp<=57"] = pack(
                    s2.loc[s2.tp_lead4.idxmax()], posthoc=True)
    en = nl[nl.cohort == EXT]
    win = en[(en.fp_lead4 >= C.FP_LO) & (en.fp_lead4 <= C.FP_HI)]
    out["null"] = {
        "max_excess_step_envelope_any_fp":
            {k: int(g.excess_tp.max()) for k, g in en.groupby("mode")},
        "max_excess_step_envelope_in_target_window":
            {k: int(g.excess_tp.max()) for k, g in win.groupby("mode")},
        "max_excess_randomised_hull_any_fp":
            {k: float(g.excess_hull_tp.max()) for k, g in en.groupby("mode")},
        "max_excess_randomised_hull_in_target_window":
            {k: float(g.excess_hull_tp.max()) for k, g in win.groupby("mode")},
        "max_tp_at_fp_le_57":
            {k: int(g[g.fp_lead4 <= 57].tp_lead4.max()
                    if (g.fp_lead4 <= 57).any() else 0)
             for k, g in en.groupby("mode")},
        "note": ("positive step-envelope excess occurs only below the smallest "
                 "admissible baseline false-alarm count, where the step "
                 "envelope returns 0"),
    }
    out["rate_matched_null"] = {
        r.cohort: {"tp_lead4": int(r.tp_lead4), "fp_lead4": int(r.fp_lead4),
                   "excess_tp": int(r.excess_tp)}
        for _, r in df[df.arm == "null_rate_matched|v7_guard"].iterrows()}
    (C.RESULTS / "headline.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n完成 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
