"""Step 9: freeze two pre-declared arms and score the sealed cohorts ONCE.

Both arms were selected on `development_main` alone, by rules written down in
steps 5 and 8 before any new head was evaluated outside development.  The
sealed cohorts were opened in step 0 only to reproduce the *published* v8.3
anchors; no head built in this bundle touched them until this script ran.

  ARM A  (objective: TP at lead >= 12, subject to dev dFP@lead>=0 <= 20 and no
          single task carrying more than half the lead>=12 gain)
          series    set_inflow_all -- probability mass, averaged over the ten
                    action tokens and all eight layers, sitting at chunk q on
                    experts that were NOT in the top-4 selection at chunk q-1
          direction low   (the router stops spilling mass outside last chunk's
                    selected set: the selection has become self-confirming)
          threshold q0.010 of the unlabeled development pool, global
          rule      excursion, 3 of the last 5 chunks over, earliest chunk 4
          development  dFP@L0 +18, dTP@L4 +52, dTP@L12 +13, dTP@L16 +1

  ARM B  (objective: TP at lead >= 16, same constraints)
          series    set_jacc_adj_all -- Jaccard overlap of the top-4 selected
                    expert sets at chunks q and q-1
          direction high  (the selected set stops turning over at all)
          threshold q0.990 within each of four bins of the episode's OWN
                    opening regime (its mean over chunks 1..5), bins and
                    thresholds both order statistics of development
          rule      hard, held 2 consecutive chunks, earliest chunk 4
          development  dFP@L0 +9, dTP@L4 +19, dTP@L12 +7, dTP@L16 +9

Nothing is fitted.  Every number in the two arms is either an integer from a
declared grid or an order statistic of an unlabeled development pool.
Comparisons are `>=` / `<=`, never strict.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import bank
import common as C
from step4_ceiling_raw import build_bank, circular_shift
from step5_rules import opening_bin, rule_excursion, rule_hard

ARM_A = {"series": "set_inflow_all", "direction": "low", "quantile": 0.99,
         "rule": "excursion", "m": 3, "k": 5, "earliest": 4, "basis": "global"}
ARM_B = {"series": "set_jacc_adj_all", "direction": "high", "quantile": 0.99,
         "rule": "hard", "confirm": 2, "earliest": 4, "basis": "open4"}
SEED = 20260906
N_NULL = 3


def fit_on_development(series_dev):
    """Everything the two arms need, read from development only."""
    fitted = {}
    s = series_dev[ARM_A["series"]]
    level = 1 - ARM_A["quantile"]
    fitted["A_thr"] = float(np.quantile(s[np.isfinite(s)], level,
                                        method="lower"))
    s = series_dev[ARM_B["series"]]
    b, edges = opening_bin(s, 4)
    fitted["B_edges"] = edges
    per_bin = {}
    global_thr = float(np.quantile(s[np.isfinite(s)], ARM_B["quantile"],
                                   method="lower"))
    for bi in range(5):
        p = s[b == bi]
        p = p[np.isfinite(p)]
        per_bin[int(bi)] = (float(np.quantile(p, ARM_B["quantile"],
                                              method="lower"))
                            if p.size >= 100 else global_thr)
    fitted["B_bins"] = per_bin
    fitted["B_global"] = global_thr
    return fitted


def apply_arms(series, fitted):
    a = rule_excursion(series[ARM_A["series"]], fitted["A_thr"],
                       ARM_A["direction"], ARM_A["m"], ARM_A["k"],
                       ARM_A["earliest"])
    s = series[ARM_B["series"]]
    b, _ = opening_bin(s, 4, edges=fitted["B_edges"])
    thr = np.array([fitted["B_bins"].get(int(x), fitted["B_global"])
                    for x in b])
    bfirst = rule_hard(s - thr[:, None], 0.0, ARM_B["direction"],
                       ARM_B["confirm"], ARM_B["earliest"])
    return a, bfirst


def report(alarms, data, label):
    prof = C.profile(alarms, data)
    print("%-14s %s" % (label, C.fmt_profile(prof)))
    return prof


def main() -> None:
    data = C.load_all()
    series = {c: build_bank(c, d) for c, d in data.items()}
    fitted = fit_on_development(series["development_main"])
    print("thresholds fitted on development only:")
    print("  arm A  %s <= %.6f  (q%.3f of the unlabeled pool)"
          % (ARM_A["series"], fitted["A_thr"], 1 - ARM_A["quantile"]))
    print("  arm B  %s >= per opening bin %s  (q%.3f), edges %s"
          % (ARM_B["series"],
             {k: round(v, 5) for k, v in fitted["B_bins"].items()},
             ARM_B["quantile"], np.round(fitted["B_edges"], 5).tolist()))

    arms = {"v8.3": {}, "v8.3+A": {}, "v8.3+B": {}, "v8.3+A+B": {}}
    firsts = {}
    for c, d in data.items():
        a, b = apply_arms(series[c], fitted)
        firsts[c] = (a, b)
        arms["v8.3"][c] = d["v83"]
        arms["v8.3+A"][c] = C.union(d["v83"], a)
        arms["v8.3+B"][c] = C.union(d["v83"], b)
        arms["v8.3+A+B"][c] = C.union(d["v83"], a, b)

    print("\n===== FULL CORPUS 32,960 episodes / 1,358 risks / 31,602"
          " successes =====")
    print("%-14s %s" % ("arm", "  ".join("lead>=%-2d TP/FP" % l
                                         for l in C.LEADS)))
    profs = {}
    v7 = {c: d["v7"] for c, d in data.items()}
    profs["v7"] = report(v7, data, "v7")
    for name, al in arms.items():
        profs[name] = report(al, data, name)

    print("\n===== per cohort at lead>=4 and lead>=12 =====")
    rows = []
    for c, d in data.items():
        n = int(d["risk"].sum())
        line = []
        for name, al in arms.items():
            s4 = C.score(al[c], d["risk"], d["length"], 4)
            s12 = C.score(al[c], d["risk"], d["length"], 12)
            s16 = C.score(al[c], d["risk"], d["length"], 16)
            line.append("%s %d/%d %dFP" % (name, s4["tp"], n, s4["fp"]))
            rows.append({"cohort": c, "arm": name, "n_risk": n,
                         "tp4": s4["tp"], "fp4": s4["fp"],
                         "tp12": s12["tp"], "fp12": s12["fp"],
                         "tp16": s16["tp"], "fp16": s16["fp"]})
        print("  %-18s %s" % (c, " | ".join(line)))
    pd.DataFrame(rows).to_csv(C.RESULTS / "sealed_per_cohort.csv", index=False)

    print("\n===== per suite (whole corpus) =====")
    for lead in (4, 12, 16):
        print("  --- lead >= %d ---" % lead)
        tabs = {n: C.per_suite(al, data, lead) for n, al in arms.items()}
        for i, suite in enumerate(C.SUITES):
            cells = []
            for n in arms:
                r = tabs[n].iloc[i]
                cells.append("%-9s %4d/%-4d %3dFP" % (n, r.tp, r.n_risk, r.fp))
            print("    %-16s %s" % (suite, " | ".join(cells)))
    per_suite_rows = []
    for lead in C.LEADS:
        for n, al in arms.items():
            t = C.per_suite(al, data, lead)
            t["arm"] = n
            per_suite_rows.append(t)
    pd.concat(per_suite_rows).to_csv(C.RESULTS / "sealed_per_suite.csv",
                                     index=False)

    print("\n===== per task: where does the gain come from? (lead>=4) =====")
    base = C.per_task(arms["v8.3"], data, 4).set_index("task")
    new = C.per_task(arms["v8.3+A+B"], data, 4).set_index("task")
    gain = (new.tp - base.tp).sort_values(ascending=False)
    total = int(gain.sum())
    print("  total TP gain at lead>=4: %+d; largest single task contributes"
          " %d (%.0f%%)" % (total, int(gain.iloc[0]),
                            100 * gain.iloc[0] / max(total, 1)))
    for t, g in gain.head(8).items():
        if g == 0:
            break
        print("    %-62s %+3d  (of %d risks)" % (t[-62:], g, base.loc[t, "n_risk"]))
    pd.DataFrame({"task": gain.index, "gain_lead4": gain.values,
                  "n_risk": base.loc[gain.index, "n_risk"].values}
                 ).to_csv(C.RESULTS / "sealed_per_task_gain.csv", index=False)

    print("\n===== leave-one-task-out: recompute the gain without each task"
          " =====")
    for lead in (4, 12, 16):
        gains = []
        for c, d in data.items():
            pass
        all_tasks = sorted(set().union(*[set(d["task"]) for d in data.values()]))
        for drop in all_tasks:
            tp_b = tp_n = 0
            for c, d in data.items():
                keep = d["task"] != drop
                for arm, acc in (("v8.3", "b"), ("v8.3+A+B", "n")):
                    f = arms[arm][c]
                    t = ((f >= 0) & ((d["length"] - f) >= lead) & d["risk"]
                         & keep).sum()
                    if acc == "b":
                        tp_b += int(t)
                    else:
                        tp_n += int(t)
            gains.append(tp_n - tp_b)
        gains = np.array(gains)
        print("  lead>=%-2d  gain with all tasks %+d | LOTO min %+d, median"
              " %+d, max %+d over %d tasks"
              % (lead, profs["v8.3+A+B"][lead][0] - profs["v8.3"][lead][0],
                 gains.min(), int(np.median(gains)), gains.max(), len(gains)))

    print("\n===== null: circular shift of both arms' series (preserves each"
          " episode's length, marginal and autocorrelation) =====")
    rng = np.random.default_rng(SEED)
    for k in range(N_NULL):
        null_alarms = {}
        for c, d in data.items():
            sh = {n: circular_shift(series[c][n], d["length"], rng)
                  for n in (ARM_A["series"], ARM_B["series"])}
            merged = dict(series[c])
            merged.update(sh)
            a, b = apply_arms(merged, fitted)
            null_alarms[c] = C.union(d["v83"], a, b)
        p = C.profile(null_alarms, data)
        print("  draw %d: %s" % (k + 1, C.fmt_profile(p)))

    np.savez_compressed(C.RESULTS / "sealed_alarms.npz",
                        **{f"{c}|{n}": v for n, al in arms.items()
                           for c, v in al.items()})
    (C.RESULTS / "sealed_summary.json").write_text(json.dumps({
        "arm_a": ARM_A, "arm_b": ARM_B,
        "thresholds": {"A": fitted["A_thr"],
                       "B_bins": fitted["B_bins"],
                       "B_edges": fitted["B_edges"].tolist()},
        "profiles": {n: {str(l): list(p[l]) for l in C.LEADS}
                     for n, p in profs.items()},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
