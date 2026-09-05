"""E2 loop 时序表型（Disperse–Switch–Collapse），PROTOCOL §5-E2 + §8 Amendment 1。

单位（永不跨单位池化）：
  U1 = main16x32 SCENE8（full 代理，全部 57 loop 事件）
  U2 = grid50x8 SCENE8（full，全部 66）
  U3 = grid50x8 long objaware 池（5 任务，仅失败集事件 56；成功集带 loop 标签的集两侧排除）
红旗排除（AUDIT §7.1/§8.4）：KITCHEN_SCENE3 与 open_middle_drawer 的 loop 通道无效，不入 U3。

族：8 信号 × 7 lead = 56 格/单位（raw 与 resid_mob1w8 各一族，各自 maxT）。
预 named：V@−2↑、A@−2↑（复现腿）；D_tok@−3↑ 且早于 V/A（新腿 a，次序统计）；
S_margin@−2↑（新腿 b）；C@+1/+2↑ 而 C@−2 不升（新腿 c；C@−2 只报效应+CI）。
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aligned_lib import (LEADS, NPERM, S8, SEED, SIG8, add_resid, build_unit,  # noqa: E402
                         compute_eff, eff_mean_se, effect_rows, family_stats,
                         order_stats, plot_aligned, rng_for, summarise_order,
                         write_effects_csv)

EXP = "E2_loop"
OUT = HERE
DISP = [d for _, d in SIG8]
L = len(LEADS)

U3_TASKS = [
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
]
UNITS = [
    ("U1_main16x32_S8", [("main16x32", "libero_long", S8)], False),
    ("U2_grid50x8_S8", [("grid50x8", "libero_long", S8)], False),
    ("U3_grid50x8_long_objaware",
     [("grid50x8", "libero_long", t) for t in U3_TASKS], True),
]
# (signal, lead, 预期方向 +1/-1/0=只报, 标签)
NAMED = [
    ("V", -2, +1, "复现腿"), ("A", -2, +1, "复现腿"),
    ("D_tok", -3, +1, "新腿a"), ("S_margin", -2, +1, "新腿b"),
    ("C", 1, +1, "新腿c"), ("C", 2, +1, "新腿c"),
    ("C", -2, 0, "新腿c对照(只报效应+CI)"),
]
NAMED_KEYS = {(s, l) for s, l, _, _ in NAMED}


def main():
    t0 = time.time()
    all_rows = []
    summary = dict(
        protocol="E2 v1 (trap 口径冻结)", seed=SEED, nperm=NPERM, leads=list(LEADS),
        family="8 signals x 7 leads = 56 cells / unit / leg; maxT per (unit, leg)",
        effect="per-event AUC(event vs same-group same-absolute-q no-loop controls) - 0.5, group-averaged",
        controls="loop_onset_q == -1 (含其他失败与 static 事件集)",
        u3_rule="objaware; 仅失败集 loop 事件为事件侧; 成功集带 loop 标签者两侧排除",
        exclusions="KITCHEN_SCENE3 与 open_middle_drawer loop 通道无效(AUDIT §7.1/§8.4), 不入 U3",
        order_stats=dict(signals=["D_tok", "V", "A", "C"], trio=["D_tok", "V", "C"],
                         p90_min_ctrl=5, scan="全 episode (q=0..nq-1)"),
        units={}, named_cells=[], order={}, presence={}, runtime_s=None)

    units_ms, presence_fig = [], {}
    for tag, specs, fail_only in UNITS:
        print(f"[{time.time()-t0:6.1f}s] {tag} load", flush=True)
        U = build_unit(tag, specs, "loop_onset_q", fail_only)
        add_resid(U, DISP)
        legs = {}
        for leg in ("raw", "resid_mob1w8"):
            eff, used, present, present_ctrl = compute_eff(
                U, "raw" if leg == "raw" else "resid", DISP)
            st = family_stats(eff, rng_for("E2", tag, leg))
            legs[leg] = (eff, used, present, present_ctrl, st)
            all_rows += effect_rows(EXP, tag, leg, "56", DISP, LEADS, eff, st,
                                    used, present, U["n_events"], NAMED_KEYS)
        eff_raw, used_raw, present, present_ctrl, st_raw = legs["raw"]
        mean, se = eff_mean_se(eff_raw, len(DISP), L)
        units_ms.append((tag.split("_", 1)[0] + " " + ("main S8" if "main" in tag
                        else "grid S8" if "S8" in tag else "grid long pool"),
                        mean, se))
        presence_fig[units_ms[-1][0]] = present.tolist()

        print(f"[{time.time()-t0:6.1f}s] {tag} order stats", flush=True)
        recs, n_no_cov = order_stats(U)
        sig_rows, pair_rows, trio = summarise_order(recs)
        summary["order"][tag] = dict(per_signal=sig_rows, sign_tests=pair_rows,
                                     trio_rank=trio, n_events_no_coverage=n_no_cov,
                                     n_events_with_coverage=len(recs),
                                     coverage_med=float(np.median([r["coverage"] for r in recs])) if recs else np.nan)

        summary["units"][tag] = dict(
            tasks=[s[2] for s in specs], corpus=specs[0][0],
            proxy_grades=U["proxy_grades"], fail_only_events=fail_only,
            n_events=U["n_events"], n_ctrl_episodes=U["n_ctrl"],
            n_excluded_success_loop=U["n_excl_succ_events"],
            n_groups_with_event=U["n_groups"],
            onset_med=float(np.median([o for g in U["ev_groups"]
                                       for _, o in U["groups"][g]["ev"]])))
        summary["presence"][tag] = dict(
            leads=list(LEADS), n_events_total=U["n_events"],
            present=present.tolist(), present_with_ctrl=present_ctrl.tolist(),
            used_raw_per_signal={d: used_raw[si].tolist() for si, d in enumerate(DISP)},
            used_resid_per_signal={d: legs["resid_mob1w8"][1][si].tolist()
                                   for si, d in enumerate(DISP)})

        for sig, lead, direc, label in NAMED:
            si, li = DISP.index(sig), list(LEADS).index(lead)
            c = si * L + li
            row = dict(unit=tag, signal=sig, lead=lead, expected="up" if direc > 0
                       else ("down" if direc < 0 else "report-only"), label=label)
            for leg in ("raw", "resid_mob1w8"):
                st = legs[leg][4]
                row[leg] = dict(eff=_f(st["obs"][c]), ci=[_f(st["ci_lo"][c]), _f(st["ci_hi"][c])],
                                p_maxT=_f(st["p"][c]), n_groups=int(st["n_groups"][c]),
                                n_events_used=int(legs[leg][1][si, li]))
            summary["named_cells"].append(row)
        print(f"[{time.time()-t0:6.1f}s] {tag} done "
              f"(ev={U['n_events']} ctrl={U['n_ctrl']} grp={U['n_groups']})", flush=True)

    write_effects_csv(os.path.join(OUT, "aligned_effects.csv"), all_rows)
    _write_order_csv(os.path.join(OUT, "order_stats.csv"), summary["order"])
    plot_aligned(units_ms, DISP, LEADS, os.path.join(OUT, "fig_aligned_curves.png"),
                 "E2 loop: event-aligned effects vs lead (mean +/- between-group SE; "
                 "AUC-0.5 of event vs same-group same-q no-loop controls)",
                 extra_panel=("presence", presence_fig))
    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{time.time()-t0:6.1f}s] all done")


def _f(x):
    return round(float(x), 4) if np.isfinite(x) else None


def _write_order_csv(path, order):
    import csv
    cols = ["unit", "kind", "signal_or_pair", "n_events", "n_crossed", "frac_crossed",
            "cross_rel_onset_q25", "cross_rel_onset_med", "cross_rel_onset_q75",
            "n_both", "n_first_earlier", "n_second_earlier", "n_ties",
            "n_only_first", "n_only_second", "n_neither", "p_sign_binom",
            "median_rank", "mean_rank", "n_all3_crossed"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for tag, o in order.items():
            for r in o["per_signal"]:
                w.writerow(dict(unit=tag, kind="crossing", signal_or_pair=r["signal"],
                                n_events=r["n_events"], n_crossed=r["n_crossed"],
                                frac_crossed=r["frac_crossed"],
                                cross_rel_onset_q25=r["cross_rel_onset_q25"],
                                cross_rel_onset_med=r["cross_rel_onset_med"],
                                cross_rel_onset_q75=r["cross_rel_onset_q75"]))
            for r in o["sign_tests"]:
                w.writerow(dict(unit=tag, kind="signtest", signal_or_pair=r["pair"],
                                n_both=r["n_both"], n_first_earlier=r["n_first_earlier"],
                                n_second_earlier=r["n_second_earlier"], n_ties=r["n_ties"],
                                n_only_first=r["n_only_first"],
                                n_only_second=r["n_only_second"], n_neither=r["n_neither"],
                                p_sign_binom=r["p_sign_binom"]))
            t = o["trio_rank"]
            for d in ("D_tok", "V", "C"):
                w.writerow(dict(unit=tag, kind="trio_rank", signal_or_pair=d,
                                median_rank=t["median_rank"][d], mean_rank=t["mean_rank"][d],
                                n_all3_crossed=t["n_all3_crossed"]))


if __name__ == "__main__":
    main()
