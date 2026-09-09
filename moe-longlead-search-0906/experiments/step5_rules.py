"""Step 5: DIFFERENT DECISION RULES on a shortlist of series.  Development only.

The diagnosis in step 1 is that the lead>=16 shortfall is a *timing* problem,
not a detectability problem: 715 of the 1,358 risks are already detected by
v8.3 and simply detected late, and in goal, object and spatial the "never
detected" count is exactly zero.  So the question is not "find new signal" but
"fire earlier at an affordable false-alarm price".

Adding a head to the union is monotone -- it can only move an alarm earlier --
so TP never falls at any lead and the entire question is the false-alarm cost.
The frontier is therefore swept directly and the selection rule is a budget on
false alarms at **lead >= 0**, the one cutoff a chunk gate cannot exploit.

Four rules, three of which are not the frozen idiom:

  hard       threshold at quantile Q, held for K consecutive chunks.  This is
             v7/v8's rule, kept as the control.
  excursion  threshold at a *moderate* Q, fire when m of the last k chunks are
             over.  Deliberately reads the bulk instead of the extreme tail,
             which is where the enrichment was measured to be too thin.  This
             is m-of-k in TIME on ONE series -- not m-of-k across many signals.
  cusum      accumulate the positive part of (x - reference) and fire when the
             accumulation crosses an order statistic of the unlabeled
             development accumulation pool.  A sequential-probability-ratio
             analogue with no learned weights.
  strat      the hard rule, but the threshold is chosen from the order
             statistics of the episodes whose OPENING REGIME (chunks 1..3 of
             the same series, binned by development quantiles) matches.  The
             opening regime is observable before the decision and is not a
             running baseline, so unlike an adaptive baseline it cannot adapt
             to the anomaly it is supposed to flag.

SELECTION RULE, declared here and applied to `development_main` only, before
`external_8b` or `legacy_main16x32` are read:

    keep configurations whose development false alarms at lead >= 0 exceed
    v8.3's by at most FP_BUDGET; among those maximise development TP at
    lead >= 12; break ties by TP at lead >= 4, then by fewer FP at lead >= 0.
    Reject any configuration whose development TP gain at lead >= 12 is more
    than HALF carried by a single task.
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import bank
import common as C
from step4_ceiling_raw import build_bank

# Shortlist declared from the step-4 ceiling scan (development).  Every entry
# either cleared its cell's null floor on the operational metric or is a
# frozen-family control.
SHORTLIST = (
    "set_inflow_all", "set_inflow_all@open", "set_outflow_all",
    "set_outflow_back", "set_jacc_adj_all", "set_jacc_adj_all@open",
    "tok_disp_fbr", "tok_near_far_fbr", "state_mob_all", "state_mob_fbr",
    "set_novel_open_all", "rank_footrule_open_all",
    "flow_path_back@open", "mob_back_s9@open",
)
RULES = ("hard", "excursion", "cusum", "strat")
QUANTILES = (0.90, 0.95, 0.975, 0.99, 0.995)
CONFIRM = (2, 3)
EXCURSION = ((2, 3), (3, 5), (4, 6), (5, 8))
CUSUM_REF = (0.75, 0.90)
STRAT_BINS = 4
EARLIEST = (4, 6)
FP_BUDGET = 20
SEED = 20260906


# ------------------------------------------------------------------ rules
def rule_hard(s, thr, direction, confirm, earliest):
    return C.confirmed_first(s, thr, direction, confirm, earliest)


def rule_excursion(s, thr, direction, m, k, earliest):
    """Fire at the first chunk where m of the last k chunks are over `thr`."""
    hit = (s >= thr) if direction == "high" else (s <= thr)
    hit &= np.isfinite(s)
    cnt = np.zeros_like(hit, dtype=np.int16)
    for q in range(s.shape[1]):
        lo = max(0, q - k + 1)
        cnt[:, q] = hit[:, lo:q + 1].sum(axis=1)
    fire = cnt >= m
    fire[:, :earliest] = False
    return np.where(fire.any(axis=1), fire.argmax(axis=1), -1)


def cusum_series(s, ref, direction):
    """S_q = max(0, S_{q-1} + (x_q - ref)), signed so that 'worse' is up."""
    x = s if direction == "high" else -s
    r = ref if direction == "high" else -ref
    out = np.zeros_like(x)
    acc = np.zeros(x.shape[0])
    for q in range(x.shape[1]):
        step = np.where(np.isfinite(x[:, q]), x[:, q] - r, 0.0)
        acc = np.maximum(0.0, acc + step)
        out[:, q] = acc
    out[~np.isfinite(s)] = np.nan
    return out


def opening_bin(s, n_bins, edges=None):
    """Bin episodes by their own opening regime.

    Chunks 1..5, because two of the shortlisted series are undefined before
    chunk 4 by construction (they reference the opening themselves).  Episodes
    with no finite opening go to their own bin and keep the global threshold.
    """
    op = bank.open_baseline(s, 1, 6)[:, 0]
    good = op[np.isfinite(op)]
    if edges is None:
        if good.size < 100:
            return np.full(s.shape[0], n_bins), np.array([])
        edges = np.quantile(good, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.digitize(np.nan_to_num(op, nan=np.inf), edges)
    b[~np.isfinite(op)] = n_bins       # its own bin, threshold = global
    return b, edges


# ------------------------------------------------------------- evaluation
def lead_profile(first, d, leads=C.LEADS):
    return {lead: C.score(first, d["risk"], d["length"], lead)
            for lead in leads}


def task_concentration(base_first, new_first, d, lead):
    """Share of the TP gain at `lead` carried by the single largest task."""
    def timely(f):
        f = np.asarray(f)
        return (f >= 0) & ((d["length"] - f) >= lead) & d["risk"]
    gain = timely(new_first) & ~timely(base_first)
    if not gain.any():
        return 0.0, 0, "-"
    tasks, counts = np.unique(d["task"][gain], return_counts=True)
    i = int(np.argmax(counts))
    return float(counts[i] / gain.sum()), int(gain.sum()), str(tasks[i])


def main() -> None:
    dev = C.load_cohort("development_main")
    series = build_bank("development_main", dev)
    missing = [s for s in SHORTLIST if s not in series]
    assert not missing, missing
    base = dev["v83"]
    base_prof = lead_profile(base, dev)
    print("v8.3 on development: " + "  ".join(
        "L%d %d/%d" % (l, base_prof[l]["tp"], base_prof[l]["fp"])
        for l in C.LEADS))

    rows = []
    for name in SHORTLIST:
        s = series[name]
        pool = s[np.isfinite(s)]
        # direction: taken from the step-4 scan sign, recomputed here as the
        # tail that is rarer for successes -- both tails are swept anyway.
        for direction in ("high", "low"):
            for quant in QUANTILES:
                level = quant if direction == "high" else 1 - quant
                thr = float(np.quantile(pool, level, method="lower"))
                for earliest in EARLIEST:
                    cfgs = []
                    for k in CONFIRM:
                        cfgs.append(("hard", {"confirm": k},
                                     rule_hard(s, thr, direction, k, earliest)))
                    for m, k in EXCURSION:
                        cfgs.append(("excursion", {"m": m, "k": k},
                                     rule_excursion(s, thr, direction, m, k,
                                                    earliest)))
                    for ref_q in CUSUM_REF:
                        rl = ref_q if direction == "high" else 1 - ref_q
                        ref = float(np.quantile(pool, rl, method="lower"))
                        cs = cusum_series(s, ref, direction)
                        cpool = cs[np.isfinite(cs)]
                        cthr = float(np.quantile(cpool, quant, method="lower"))
                        cfgs.append(("cusum", {"ref_q": ref_q},
                                     C.confirmed_first(cs, cthr, "high", 1,
                                                       earliest)))
                    b, _ = opening_bin(s, STRAT_BINS)
                    strat = np.full(s.shape[0], -1)
                    for bi in np.unique(b):
                        m_bin = b == bi
                        p = s[m_bin]
                        p = p[np.isfinite(p)]
                        if p.size < 100:
                            t = thr
                        else:
                            t = float(np.quantile(p, level, method="lower"))
                        strat[m_bin] = rule_hard(s[m_bin], t, direction, 2,
                                                 earliest)
                    cfgs.append(("strat", {"bins": STRAT_BINS}, strat))

                    for rule, params, first in cfgs:
                        u = C.union(base, first)
                        prof = lead_profile(u, dev)
                        conc, gain, top = task_concentration(base, u, dev, 12)
                        row = {"series": name, "direction": direction,
                               "quantile": quant, "earliest": earliest,
                               "rule": rule, "params": json.dumps(params)}
                        for lead in C.LEADS:
                            row[f"tp{lead}"] = prof[lead]["tp"]
                            row[f"fp{lead}"] = prof[lead]["fp"]
                        row["d_fp0"] = prof[0]["fp"] - base_prof[0]["fp"]
                        row["d_tp4"] = prof[4]["tp"] - base_prof[4]["tp"]
                        row["d_tp12"] = prof[12]["tp"] - base_prof[12]["tp"]
                        row["d_tp16"] = prof[16]["tp"] - base_prof[16]["tp"]
                        row["conc12"] = conc
                        row["gain12"] = gain
                        row["top_task"] = top
                        rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(C.RESULTS / "rule_sweep_development.csv", index=False)
    print("configurations swept: %d" % len(grid))

    ok = grid[(grid.d_fp0 <= FP_BUDGET) & (grid.conc12 <= 0.5)]
    print("passing FP@L0 budget %+d and the <=50%% single-task rule: %d"
          % (FP_BUDGET, len(ok)))
    print("\n=== top 20 by development TP at lead>=12 ===")
    top = ok.sort_values(["d_tp12", "d_tp4", "d_fp0"],
                         ascending=[False, False, True]).head(20)
    print("%-24s %-10s %-5s %-4s %-16s %6s %6s %6s %6s %6s"
          % ("series", "rule", "dir", "earl", "params",
             "dFP0", "dTP4", "dTP12", "dTP16", "conc"))
    for _, r in top.iterrows():
        print("%-24s %-10s %-5s %-4d %-16s %+6d %+6d %+6d %+6d %6.2f"
              % (r.series[:24], r.rule, r.direction, r.earliest,
                 r.params[:16], r.d_fp0, r.d_tp4, r.d_tp12, r.d_tp16, r.conc12))

    print("\n=== best per rule family (development) ===")
    for rule in RULES:
        sub = ok[ok.rule == rule]
        if sub.empty:
            print("  %-10s no configuration inside the budget" % rule)
            continue
        r = sub.sort_values(["d_tp12", "d_tp4"], ascending=False).iloc[0]
        print("  %-10s %-24s %-5s q%.3f earl%d %-14s -> dFP0 %+d  dTP4 %+d "
              " dTP12 %+d  dTP16 %+d  (top task %.0f%% of the gain)"
              % (rule, r.series[:24], r.direction, r["quantile"], r.earliest,
                 r.params[:14], r.d_fp0, r.d_tp4, r.d_tp12, r.d_tp16,
                 100 * r.conc12))

    print("\n=== best per series (development, any rule) ===")
    for name in SHORTLIST:
        sub = ok[ok.series == name]
        if sub.empty:
            print("  %-24s nothing inside the budget" % name)
            continue
        r = sub.sort_values(["d_tp12", "d_tp4"], ascending=False).iloc[0]
        print("  %-24s %-10s q%.3f -> dFP0 %+3d  dTP4 %+3d  dTP12 %+3d "
              " dTP16 %+3d" % (name, r.rule, r["quantile"], r.d_fp0, r.d_tp4,
                               r.d_tp12, r.d_tp16))

    (C.RESULTS / "rule_sweep.json").write_text(json.dumps({
        "n_configs": len(grid), "n_passing": len(ok),
        "fp_budget_lead0": FP_BUDGET,
        "v83_development": {str(l): [base_prof[l]["tp"], base_prof[l]["fp"]]
                            for l in C.LEADS},
        "selection_rule": "max dev TP@L12 s.t. dev dFP@L0 <= 20 and no single "
                          "task carries >50% of the L12 gain",
        "top": top.head(5).to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
