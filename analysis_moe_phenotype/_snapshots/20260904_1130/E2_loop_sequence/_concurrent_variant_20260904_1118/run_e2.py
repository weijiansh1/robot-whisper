"""E2 loop 时序表型 Disperse–Switch–Collapse（PROTOCOL §4 / §5-E2 / §8 Amendment 1）。

复现单位（永不跨单位池化）：
  U1_main_S8   main16x32 KITCHEN_SCENE8（full 代理；55 失败 loop 事件 / 全部 57）
  U2_grid_S8   grid50x8  KITCHEN_SCENE8（full；62 / 66）
  U3_grid_long grid50x8 libero_long objaware 池：LR_SCENE1 19 / LR_SCENE2 soup+tomato 12 /
               LR_SCENE6 11 / LR_SCENE5 10 / LR_SCENE2 cheese+butter 4 = 56，组 = (task, scene)
红旗排除（AUDIT §7.1 / §8.4）：KITCHEN_SCENE3 与 open_the_middle_drawer 的 loop 通道无效。
主检验只用失败集事件；成功集带 loop 标签的集是 E4 的负对照人群，事件侧与对照侧都排除。

族 = 8 信号 × 7 lead = 56 格/单位（raw 与 resid_mob1w8 各自独立成族）。
运行：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_e2.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align_lib import (BASE, LEADS, NPERM, S8, SEED, SIG8_NAMES, eff_mean_se,  # noqa: E402
                       first_crossing, load_unit, plot_aligned, presence_audit,
                       residual_values, run_family)

OUT = os.path.dirname(os.path.abspath(__file__))
LONG = "libero_long"
LR = ["LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
      "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
      "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_"
      "to_the_right_of_the_plate",
      "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_"
      "mug_on_the_right_plate",
      "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"]

UNITS = [
    ("U1_main_S8", [("main16x32", LONG, S8)]),
    ("U2_grid_S8", [("grid50x8", LONG, S8)]),
    ("U3_grid_long_objaware", [("grid50x8", LONG, t) for t in LR]),
]

PRENAMED = [
    ("rep_V_m2", "V", -2, "up", "复现腿：late_flow_volatility 在 onset−2 升"),
    ("rep_A_m2", "A", -2, "up", "复现腿：route_acceleration 在 onset−2 升"),
    ("new_a_Dtok_m3", "D_tok", -3, "up", "新腿 a：token_dispersion 在 onset−3 升（早于 V/A）"),
    ("new_b_Smargin_m2", "S_margin", -2, "up", "新腿 b：top12_margin 在 onset−2 升（unstable commitment）"),
    ("new_c_C_p1", "C", 1, "up", "新腿 c：token_consensus 在 onset+1 升（collapse 在 onset 后）"),
    ("new_c_C_p2", "C", 2, "up", "新腿 c：token_consensus 在 onset+2 升"),
    ("new_c_C_m2", "C", -2, "none", "新腿 c 对照：C 在 onset−2 不升（只报效应+CI，不做等价性主张）"),
]
ORDER_SIGS = ["D_tok", "V", "A", "C"]
ORDER_PAIRS = [("D_tok", "V"), ("D_tok", "A"), ("D_tok", "C"), ("V", "C"), ("A", "C")]
MIN_EVENTS = 8


def verdict(per_unit, expected):
    if expected == "none":
        return "descriptive_only"
    up = [t for t, v in per_unit.items() if v["sig_raw"] and v["effect"] > 0]
    dn = [t for t, v in per_unit.items() if v["sig_raw"] and v["effect"] < 0]
    ok = [t for t, v in per_unit.items() if not v["underpowered"]]
    if up and dn:
        return "direction_conflict"
    if dn:
        return "opposite_direction"
    if len(up) == len(ok) and len(ok) == 3:
        return "established_3of3"
    if len(up) >= 2:
        return "established_2of3"
    if len(up) == 1:
        return "single_unit"
    return "null"


def main():
    t0 = time.time()
    eff_rows, audit_rows, order_rows, units_meta = [], [], [], {}
    plot_units, presence_curve, eff_store = [], {}, {}

    for tag, tasks in UNITS:
        U = load_unit(tasks, "loop")
        paired = [g for g in range(U["n_groups"])
                  if len(U["ev_of_group"][g]) and len(U["ct_of_group"][g])]
        n_ev_paired = sum(len(U["ev_of_group"][g]) for g in paired)
        pa, n_orphan = presence_audit(U)
        for r in pa:
            r.update(unit=tag)
        audit_rows += pa
        onsets = U["onset"][U["is_event"]]
        units_meta[tag] = dict(
            tasks=[f"{c}/{s}/{t}" for c, s, t in tasks],
            proxy_grade=sorted(set(U["grade"].tolist())),
            n_episodes=int(len(U["nq"])), n_fail=int((U["success"] == 0).sum()),
            n_loop_events_fail=int(U["is_event"].sum()),
            n_loop_events_succ_excluded=int(((U["onset"] >= 0) & (U["success"] == 1)).sum()),
            n_control_episodes=int(U["is_ctrl"].sum()),
            n_groups_total=U["n_groups"], n_groups_paired=len(paired),
            n_events_in_paired_groups=n_ev_paired,
            n_events_orphan_no_control=n_orphan,
            underpowered=bool(n_ev_paired < MIN_EVENTS),
            onset_min=int(onsets.min()), onset_median=float(np.median(onsets)),
            onset_max=int(onsets.max()))
        print(f"[{time.time()-t0:6.1f}s] {tag}: ev={int(U['is_event'].sum())} "
              f"paired={n_ev_paired} orphan={n_orphan} "
              f"groups={len(paired)}/{U['n_groups']} ctrl={int(U['is_ctrl'].sum())}", flush=True)

        recs, eff_raw = run_family(U, U["sig"], SIG8_NAMES, tag, "raw")
        res_vals, n_cells = residual_values(U, SIG8_NAMES)
        recs_r, eff_res = run_family(U, res_vals, SIG8_NAMES, tag, "resid_mob1w8")
        units_meta[tag]["n_residual_cells"] = n_cells
        for r in recs + recs_r:
            r.update(unit=tag, experiment="E2", event="loop")
        eff_rows += recs + recs_r
        eff_store[tag] = (eff_raw, eff_res)
        m, se = eff_mean_se(eff_raw, len(SIG8_NAMES), len(LEADS))
        plot_units.append((tag, m, se))
        presence_curve[tag] = [r["n_used"] for r in pa]

        for window in ("runup8", "prefix"):
            cr, _ = first_crossing(U, ORDER_SIGS, window)
            for n in ORDER_SIGS:
                v = cr[n]
                fin = np.isfinite(v)
                order_rows.append(dict(
                    unit=tag, window=window, kind="crossing", signal=n, pair="",
                    n_events=int(len(v)), n_crossed=int(fin.sum()),
                    n_censored=int((~fin).sum()),
                    median_rel_onset=float(np.median(v[fin])) if fin.any() else np.nan,
                    q25=float(np.percentile(v[fin], 25)) if fin.any() else np.nan,
                    q75=float(np.percentile(v[fin], 75)) if fin.any() else np.nan,
                    n_a_earlier=-1, n_b_earlier=-1, n_tie=-1, p_binom=np.nan))
            for a, b in ORDER_PAIRS:
                va, vb = cr[a], cr[b]
                both = np.isfinite(va) & np.isfinite(vb)
                na = int((va[both] < vb[both]).sum())
                nb = int((va[both] > vb[both]).sum())
                nt = int((va[both] == vb[both]).sum())
                p = binomtest(na, na + nb, 0.5).pvalue if na + nb else np.nan
                order_rows.append(dict(
                    unit=tag, window=window, kind="signtest", signal="", pair=f"{a}<{b}",
                    n_events=int(both.sum()), n_crossed=int(both.sum()), n_censored=-1,
                    median_rel_onset=np.nan, q25=np.nan, q75=np.nan,
                    n_a_earlier=na, n_b_earlier=nb, n_tie=nt,
                    p_binom=float(p) if np.isfinite(p) else np.nan))
        print(f"[{time.time()-t0:6.1f}s] {tag}: family + order done", flush=True)

    df = pd.DataFrame(eff_rows)[
        ["experiment", "unit", "event", "leg", "signal", "lead", "effect", "ci_lo", "ci_hi",
         "p_maxT", "n_groups", "n_groups_pos", "n_groups_neg", "n_events_used", "n_pairs"]]
    df.round(6).to_csv(os.path.join(OUT, "aligned_effects.csv"), index=False)
    pd.DataFrame(order_rows).round(6).to_csv(os.path.join(OUT, "order_stats.csv"), index=False)
    pd.DataFrame(audit_rows)[
        ["unit", "lead", "n_events_in_paired_groups", "n_drop_before_start",
         "n_drop_after_episode_end", "n_alive", "n_used", "median_n_controls",
         "min_n_controls", "total_n_controls"]].to_csv(
        os.path.join(OUT, "presence_audit.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "group_effects.npz"),
                        **{f"{t}__{k}": v for t, ab in eff_store.items()
                           for k, v in (("raw", ab[0]), ("resid", ab[1]))})

    verdicts = {}
    for key, sig, lead, direction, desc in PRENAMED:
        per_unit = {}
        for tag, _ in UNITS:
            sub = df[(df.unit == tag) & (df.signal == sig) & (df.lead == lead)]
            r = sub[sub.leg == "raw"].iloc[0]
            rr = sub[sub.leg == "resid_mob1w8"].iloc[0]
            per_unit[tag] = dict(
                effect=float(r.effect), ci=[float(r.ci_lo), float(r.ci_hi)],
                p_maxT=float(r.p_maxT), n_groups=int(r.n_groups),
                groups_pos=int(r.n_groups_pos), groups_neg=int(r.n_groups_neg),
                n_events_used=int(r.n_events_used), n_pairs=int(r.n_pairs),
                resid_effect=float(rr.effect), resid_ci=[float(rr.ci_lo), float(rr.ci_hi)],
                resid_p_maxT=float(rr.p_maxT), resid_n_groups=int(rr.n_groups),
                resid_n_events_used=int(rr.n_events_used),
                sig_raw=bool(np.isfinite(r.p_maxT) and r.p_maxT < 0.05),
                sig_resid=bool(np.isfinite(rr.p_maxT) and rr.p_maxT < 0.05),
                underpowered=units_meta[tag]["underpowered"])
        v = verdict(per_unit, direction)
        verdicts[key] = dict(signal=sig, lead=lead, expected=direction, desc=desc,
                             verdict=v, units=per_unit,
                             n_units_sig_raw=int(sum(u["sig_raw"] for u in per_unit.values())),
                             n_units_sig_resid=int(sum(u["sig_resid"] for u in per_unit.values())),
                             n_units_effect_positive=int(sum(u["effect"] > 0 for u in per_unit.values())))
        print(f"  {key:18s} {sig}@{lead:+d} -> {v} "
              f"({[round(u['effect'],3) for u in per_unit.values()]}, "
              f"p={[round(u['p_maxT'],4) for u in per_unit.values()]})", flush=True)

    plot_aligned(plot_units, SIG8_NAMES, LEADS,
                 os.path.join(OUT, "fig_aligned_curves.png"),
                 "E2  loop onset-aligned routing signals (event vs matched same-q controls)",
                 presence=presence_curve)

    summary = dict(
        experiment="E2 loop 时序表型 (Disperse-Switch-Collapse)",
        protocol="PROTOCOL.md §4 / §5-E2 / §8 Amendment 1; AUDIT.md §7 §8",
        seed=SEED, nperm=NPERM, leads=LEADS, signals=SIG8_NAMES,
        family="8 signals x 7 leads = 56 cells per unit per leg; raw and resid_mob1w8 "
               "are two independent maxT families",
        effect_definition="group AUC(event vs same-group same-absolute-q controls) - 0.5, "
                          "pairs pooled over the group's event episodes (frozen trap 口径, "
                          "= matched_group_statistics in himoe-vla_trap)",
        control_definition="all episodes in the same (task,scene) group with loop_onset_q<0 "
                           "(includes other failures and successes), alive at the matched q",
        event_definition="failure episodes with loop_onset_q>=0; success episodes carrying a "
                         "loop label are excluded from BOTH sides (E4 negative-control population)",
        cross_unit_aggregation="unweighted group mean (phenotype/stats.py::groupwise_signflip_maxt); "
                               "the frozen trap script used a pair-weighted mean - see report 边界",
        red_flag_exclusions=[
            "grid50x8/libero_long/KITCHEN_SCENE3_* : loop channel invalid (articulated knob)",
            "*/libero_goal/open_the_middle_drawer_of_the_cabinet : loop channel invalid (drawer)"],
        min_events_for_inference=MIN_EVENTS,
        order_stat_definition="per event episode, first absolute q in the window where the raw "
                              "signal exceeds the same-group same-q control 90th percentile "
                              "(>=5 alive controls required); window runup8 = [onset-8, onset+2], "
                              "prefix = [0, onset+2]; reported relative to onset; "
                              "sign test = two-sided exact binomial on non-tied pairs",
        units=units_meta, prenamed=verdicts,
        presence_audit=audit_rows, order_stats=order_rows,
        runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{time.time()-t0:6.1f}s] E2 done", flush=True)


if __name__ == "__main__":
    main()
