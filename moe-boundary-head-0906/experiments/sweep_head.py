"""Step 2: sweep the head design.  development_main ONLY.

Nothing in this file reads `external_8b` or `legacy_main16x32`.  The choice it
writes to `results/frozen_spec.json` is what `freeze_and_score.py` then applies
once to the two test cohorts.

The design space is in `heads.py`: two families (chunk boundary, adjacent-query
at step 9) x four cells x {raw, self-baselined, per-chunk rank} plus the
action-minus-state contrasts in four normalisations, one of which
(`rawdiff`) is carried as a labelled negative control.  On top of that,
smoothing width, confirmation K, quantile, earliest chunk and an optional chunk
gate; plus OR / AND of two separately-thresholded single-cell heads.

Selection is at **lead >= 0**, and this is not a free choice.  Failures are
defined as running to the cap, so risk episodes are systematically the long
ones: the minimum risk length is 22 chunks against a safe median of 13, and
length alone scores AUC 0.94-0.96.  Any head that fires late therefore has its
false alarms deleted by the lead filter rather than by being right.  A first
pass that selected at lead >= 4 returned `rank|bnd:back_action`, gate 8-24, at
+39 TP / +0 FP - which at lead >= 0 is +23 TP / +73 FP, and whose 600 safe
alarms were all removed by the filter, not avoided.  Because a chunk gate is in
the design space, lead >= 0 is the only selection point a gate cannot game.
Every lead is reported.

The objective is the exchange rate over v8, not raw TP.  v8's own two heads
bought +62 TP for +15 FP over v7 on the full corpus, ~4:1.

Two controls decide the question, and both are printed before the candidate
table:

  * `v8_dial` - v8's *existing* two heads re-thresholded over a quantile grid.
    A new head has to beat what you already get by loosening the dial that is
    there.  If it does not, it is a knob, not information.
  * the selection intensity - how many configurations were searched, and how
    much of the winner's development margin is search noise.
"""

from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from common import (
    HEADLINE_LEAD,
    LEADS,
    OUT,
    V8,
    confirmed_first,
    load,
    score,
    score_all_leads,
    trailing_mean,
    union,
)
from heads import (
    NEGATIVE_CONTROL_FORMS,
    PAIRS,
    constructions,
    fire_from,
    first_true_from,
    held_runs,
    tag,
    threshold_of,
    window_of,
)

sys.path.insert(0, str(V8 / "experiments"))
from evaluate_full_corpus import (  # noqa: E402
    DIRECTION as V8_DIRECTION,
    heads_from_flow_speed,
)

DEV = "development_main"
WIDTHS = (1, 2, 3, 4, 6)
CONFIRMS = (1, 2, 3)
QUANTILES = (0.98, 0.99, 0.995, 0.998, 0.999)
EARLIESTS = (2, 4, 6)
GATES = (None, (4, 16), (4, 24), (8, 24), (6, 20))
DIAL_QUANTILES = (0.995, 0.993, 0.99, 0.985, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90)
MIN_RATE = 2.0
MIN_DTP = 8
SELECT_LEAD = 0
COLS = ["score", "direction", "width", "confirm", "quantile", "earliest",
        "gate", "d_tp0", "d_fp0", "rate", "d_tp4", "d_fp4",
        "alone_tp0", "alone_fp0"]


def v8_dial(d) -> pd.DataFrame:
    speed = np.load(V8 / f"results/{DEV}_flow_speed.npz")["flow_speed"]
    heads = heads_from_flow_speed(speed)
    rows = []
    for q in DIAL_QUANTILES:
        firsts = [d["v7"]]
        for name, series in heads.items():
            thr = threshold_of(series, q, V8_DIRECTION[name])
            firsts.append(confirmed_first(series, thr, V8_DIRECTION[name], 2))
        s = score_all_leads(union(*firsts), d["risk"], d["length"])
        rows.append({"quantile": q, **s})
    return pd.DataFrame(rows)


def main() -> None:
    d = load(DEV)
    n_risk = int(d["risk"].sum())
    risk, length, v8 = d["risk"], d["length"], d["v8"]
    base = score(v8, risk, length, HEADLINE_LEAD)
    base_all = score_all_leads(v8, risk, length)
    assert (base["tp"], base["fp"]) == (331, 44), base
    assert (base_all["tp4"], base_all["fp4"]) == (331, 44), base_all
    n_chunk = d["front_state"].shape[1]
    print("development v8 基准（cap-free, lead>=4）: %d/%d TP  %d FP"
          % (base["tp"], n_risk, base["fp"]))

    dial = v8_dial(d)
    b = dial[dial["quantile"] == 0.995].iloc[0]
    assert (int(b.tp4), int(b.fp4)) == (331, 44), b
    for lead in LEADS:
        dial[f"d_tp{lead}"] = (dial[f"tp{lead}"] - b[f"tp{lead}"]).astype(int)
        dial[f"d_fp{lead}"] = (dial[f"fp{lead}"] - b[f"fp{lead}"]).astype(int)
    dial["rate"] = np.where(dial.d_fp0 > 0,
                            dial.d_tp0 / dial.d_fp0.clip(lower=1), np.nan)
    dial.to_csv(OUT / "v8_dial_development.csv", index=False)
    print("\n===== 对照 A：只把 v8 现有两个头的分位放松（development）=====")
    for _, r in dial.iterrows():
        print("  分位 %-6.3f | lead>=0 %4d/%-4d %5d FP  相对 %+4d/%+5d %-12s"
              "| lead>=4 %4d/%-4d %4d FP  相对 %+4d/%+5d"
              % (r["quantile"], r.tp0, n_risk, r.fp0, r.d_tp0, r.d_fp0,
                 "%.2f TP/FP" % r.rate if r.d_fp0 > 0 else "-",
                 r.tp4, n_risk, r.fp4, r.d_tp4, r.d_fp4))

    scores = constructions(d, d)
    print("\n候选 construction %d 个，平滑宽度 %d 种" % (len(scores), len(WIDTHS)))

    # the run counter resets at the window start, so configurations are grouped
    # by that start; `WINDOWS[start]` lists every (earliest, gate) that uses it
    tail = list(itertools.product(EARLIESTS, GATES))
    windows: dict[int, list] = {}
    for earliest, gate in tail:
        start, hi = window_of(earliest, gate)
        windows.setdefault(start, []).append((earliest, gate, hi))
    rows = []
    for name, series in scores.items():
        for w in WIDTHS:
            sm = trailing_mean(series, w)
            for direction in ("high", "low"):
                for quantile in QUANTILES:
                    thr = threshold_of(sm, quantile, direction)
                    for start, members in windows.items():
                        run = held_runs(sm, thr, direction, start)
                        for confirm in CONFIRMS:
                            ft = first_true_from(run >= confirm)
                            for earliest, gate, hi in members:
                                first = fire_from(ft, start, hi, n_chunk)
                                n_fire = int((first >= 0).sum())
                                if n_fire < 5:
                                    continue
                                u = score_all_leads(union(v8, first), risk, length)
                                a = score_all_leads(first, risk, length)
                                rows.append({
                                    "score": name, "direction": direction, "width": w,
                                    "confirm": confirm, "quantile": quantile,
                                    "earliest": earliest,
                                    "gate": "-" if gate is None else "%d-%d" % gate,
                                    "threshold": thr, "n_fire": n_fire,
                                    **{f"d_tp{b}": u[f"tp{b}"] - base_all[f"tp{b}"]
                                       for b in LEADS},
                                    **{f"d_fp{b}": u[f"fp{b}"] - base_all[f"fp{b}"]
                                       for b in LEADS},
                                    **{f"alone_tp{b}": a[f"tp{b}"] for b in LEADS},
                                    **{f"alone_fp{b}": a[f"fp{b}"] for b in LEADS}})
    single = pd.DataFrame(rows)
    print("单头配置 %d" % len(single))

    or_rows = []
    for family in ("bnd", "wc"):
        for act, st in PAIRS:
            for form in ("raw", "sb", "rank"):
                for w in WIDTHS:
                    sa = trailing_mean(scores[f"{form}|{family}:{act}"], w)
                    ss = trailing_mean(scores[f"{form}|{family}:{st}"], w)
                    for quantile in QUANTILES:
                      ta = threshold_of(sa, quantile, "high")
                      ts = threshold_of(ss, quantile, "low")
                      for start, members in windows.items():
                        ra = held_runs(sa, ta, "high", start)
                        rs = held_runs(ss, ts, "low", start)
                        for confirm in CONFIRMS:
                            fta = first_true_from(ra >= confirm)
                            fts = first_true_from(rs >= confirm)
                            for earliest, gate, hi in members:
                                fa = fire_from(fta, start, hi, n_chunk)
                                fs = fire_from(fts, start, hi, n_chunk)
                                both = np.where((fa >= 0) & (fs >= 0),
                                                np.maximum(fa, fs), -1)
                                for combo, first in (("OR", union(fa, fs)),
                                                     ("AND", both)):
                                    n_fire = int((first >= 0).sum())
                                    if n_fire < 5:
                                        continue
                                    u = score_all_leads(union(v8, first), risk,
                                                        length)
                                    a = score_all_leads(first, risk, length)
                                    or_rows.append({
                                        "score": "%s|%s|%s" % (combo, form,
                                                               tag(family, act, st)),
                                        "direction": combo, "width": w,
                                        "confirm": confirm, "quantile": quantile,
                                        "earliest": earliest,
                                        "gate": "-" if gate is None else "%d-%d" % gate,
                                        "threshold": np.nan, "n_fire": n_fire,
                                        **{f"d_tp{b}": u[f"tp{b}"] - base_all[f"tp{b}"]
                                           for b in LEADS},
                                        **{f"d_fp{b}": u[f"fp{b}"] - base_all[f"fp{b}"]
                                           for b in LEADS},
                                        **{f"alone_tp{b}": a[f"tp{b}"] for b in LEADS},
                                        **{f"alone_fp{b}": a[f"fp{b}"] for b in LEADS}})
    table = pd.concat([single, pd.DataFrame(or_rows)], ignore_index=True)
    # exchange rate at the SELECTION lead, where a chunk gate cannot hide alarms
    table["rate"] = np.where(table.d_fp0 > 0,
                             table.d_tp0 / table.d_fp0.clip(lower=1),
                             np.where(table.d_tp0 > 0, np.inf, 0.0))
    # how much of the lead>=4 picture is the length filter rather than the head
    table["lead_filter_gain"] = table.alone_fp0 - table.alone_fp4
    table["form"] = table.score.str.split("|").str[0]
    table["is_negative_control"] = table.form.isin(NEGATIVE_CONTROL_FORMS)
    table.to_csv(OUT / "sweep_development.csv", index=False)
    print("双头 OR/AND 配置 %d  ——  合计 %d 个配置" % (len(or_rows), len(table)))

    live = table[~table.is_negative_control]
    good = live[(live.d_tp0 >= MIN_DTP) & (live.rate >= MIN_RATE)]
    print("\n===== development 上净增 >=%d TP 且兑换率 >= %.1f TP/FP ====="
          % (MIN_DTP, MIN_RATE))
    print(good.sort_values(["rate", "d_tp0"], ascending=False).head(20)[COLS]
          .to_string(index=False) if len(good) else "   一个都没有。")

    print("\n===== 净增 TP 最多的 15 个（不含负对照）=====")
    print(live.nlargest(15, "d_tp0")[COLS].to_string(index=False))

    sub = live[live.d_tp0 >= MIN_DTP]
    print("\n===== 每个 construction 的最好兑换率（净增 >= %d TP）=====" % MIN_DTP)
    if len(sub):
        best = sub.loc[sub.groupby("score").rate.idxmax()]
        print(best.sort_values("rate", ascending=False).head(25)[COLS]
              .to_string(index=False))
    else:
        print("   没有任何 construction 净增 >= %d TP。" % MIN_DTP)

    print("\n===== 负对照（rawdiff：直接相减，量纲被 state 主导）=====")
    neg = table[table.is_negative_control]
    print(neg.nlargest(5, "d_tp0")[COLS].to_string(index=False))

    print("\n===== 边界 vs 同步（bnd 与 wc 家族的最好表现）=====")
    print("%-6s %-8s %s" % ("家族", "FP 预算", "最好净增 TP"))
    fam_rows = []
    for family in ("bnd", "wc"):
        m = live.score.str.contains(f"{family}:")
        for budget in (5, 10, 20, 40, 80):
            cand = live[m & (live.d_fp0 <= budget)]
            v = int(cand.d_tp0.max()) if len(cand) else 0
            fam_rows.append({"family": family, "fp_budget": budget, "d_tp": v})
            print("%-6s %-8d +%d" % (family, budget, v))
    pd.DataFrame(fam_rows).to_csv(OUT / "family_comparison_development.csv",
                                  index=False)

    print("\n===== 对照 A 的判决：同 FP 预算下，新头 vs 放松 v8 分位 =====")
    print("%-8s %-14s %-14s %s" % ("+FP 预算", "最好的新头", "放松 v8 分位", "差"))
    cmp_rows = []
    for budget in (5, 10, 20, 40, 80, 160):
        cand = live[live.d_fp0 <= budget]
        bh = int(cand.d_tp0.max()) if len(cand) else 0
        dc = dial[dial.d_fp0 <= budget]
        bd = int(dc.d_tp0.max()) if len(dc) else 0
        cmp_rows.append({"fp_budget": budget, "head_d_tp": bh,
                         "v8_dial_d_tp": bd, "advantage": bh - bd})
        print("%-8d +%-13d +%-13d %+d" % (budget, bh, bd, bh - bd))
    pd.DataFrame(cmp_rows).to_csv(OUT / "dial_comparison_development.csv",
                                  index=False)

    (OUT / "sweep_summary.json").write_text(json.dumps({
        "development_v8": base, "n_risk": n_risk,
        "n_configs": int(len(table)),
        "n_configs_live": int(len(live)),
        "n_clearing": int(len(good)),
        "max_d_tp0": int(live.d_tp0.max()),
        "best_rate_at_min_dtp": (float(sub.rate.replace(np.inf, np.nan).max())
                                 if len(sub) else None),
        "v8_dial": dial.to_dict("records"),
        "dial_comparison": cmp_rows,
        "family_comparison": fam_rows,
        "min_rate": MIN_RATE, "min_dtp": MIN_DTP,
        "widths": list(WIDTHS), "confirms": list(CONFIRMS),
        "quantiles": list(QUANTILES), "earliests": list(EARLIESTS),
        "gates": [g if g is None else list(g) for g in GATES],
    }, indent=2, default=float))
    print("\n写出 %s" % (OUT / "sweep_development.csv"))


if __name__ == "__main__":
    main()
