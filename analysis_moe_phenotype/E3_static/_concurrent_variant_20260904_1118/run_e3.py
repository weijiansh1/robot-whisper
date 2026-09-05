"""E3 static 表型（flattening + shared-support）——PROTOCOL §4 / §5-E3。

框架与 E2 完全相同（trap 口径事件对齐），事件通道 = static_onset_q。
复现单位：
  U1_main_S8         main16x32 KITCHEN_SCENE8（full；143 失败 static 事件 / 全部 146）
  U2_grid_S8         grid50x8  KITCHEN_SCENE8（full；69 / 70）
  U3_grid_objaware   grid50x8 除 SCENE8 外全部 39 个任务（objaware）：goal 21 / long 非 SCENE8 64 /
                     spatial 4 / object 4 = 93 失败事件（含成功集事件共 110），组 = (task, scene)
static 通道对全部任务有效（AUDIT §4：degraded/objaware static sens/spec ≥0.97），
铰接机构红旗只作用于 loop/trap 通道，故 KITCHEN_SCENE3 与 open_middle_drawer 的 static 保留。

族 = 9 信号 × 7 lead = 63 格/单位（8 核心信号 + min_{k>=2} mob_k；raw / resid 各一族）。
把 min_mob_k 放进同一族而非族外单测，是更保守的选择（PROTOCOL §5-E3 把它列为预 named）。
运行：OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 run_e3.py
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "E2_loop_sequence"))
from align_lib import (BASE, LEADS, NPERM, S8, SEED, SIG8_NAMES, eff_mean_se,  # noqa: E402
                       load_unit, plot_aligned, presence_audit, residual_values,
                       run_family)

OUT = HERE
LONG = "libero_long"
SIG9 = SIG8_NAMES + ["min_mob_k"]


def grid_non_s8():
    root = os.path.join(BASE, "features", "grid50x8")
    out = []
    for suite in sorted(os.listdir(root)):
        for task in sorted(os.listdir(os.path.join(root, suite))):
            if task == S8:
                continue
            out.append(("grid50x8", suite, task))
    return out


UNITS = [
    ("U1_main_S8", [("main16x32", LONG, S8)]),
    ("U2_grid_S8", [("grid50x8", LONG, S8)]),
    ("U3_grid_objaware", grid_non_s8()),
]

PRENAMED = [
    ("S_ent_0", "S_ent", 0, "up", "gate_entropy 在 onset 升（路由变平 = flattening）"),
    ("C_0", "C", 0, "up", "token_consensus 在 onset 升（共享支持集 = shared support）"),
    ("min_mobk_0", "min_mob_k", 0, "down", "min_{k>=2} mob_k 在 onset 降（周期性复现）"),
]
MIN_EVENTS = 8


def verdict(per_unit, expected):
    want = 1.0 if expected == "up" else -1.0
    hit = [t for t, v in per_unit.items() if v["sig_raw"] and np.sign(v["effect"]) == want]
    opp = [t for t, v in per_unit.items() if v["sig_raw"] and np.sign(v["effect"]) == -want]
    ok = [t for t, v in per_unit.items() if not v["underpowered"]]
    if hit and opp:
        return "direction_conflict"
    if opp:
        return "opposite_direction"
    if len(hit) == len(ok) and len(ok) == 3:
        return "established_3of3"
    if len(hit) >= 2:
        return "established_2of3"
    if len(hit) == 1:
        return "single_unit"
    return "null"


def main():
    t0 = time.time()
    eff_rows, audit_rows, units_meta = [], [], {}
    plot_units, presence_curve, eff_store = [], {}, {}

    for tag, tasks in UNITS:
        U = load_unit(tasks, "static", with_min_mobk=True)
        paired = [g for g in range(U["n_groups"])
                  if len(U["ev_of_group"][g]) and len(U["ct_of_group"][g])]
        n_ev_paired = sum(len(U["ev_of_group"][g]) for g in paired)
        pa, n_orphan = presence_audit(U)
        for r in pa:
            r.update(unit=tag)
        audit_rows += pa
        onsets = U["onset"][U["is_event"]]
        per_task = {}
        for ti, (c, s, t) in enumerate(tasks):
            m = U["task"] == ti
            n = int((m & U["is_event"]).sum())
            if n:
                per_task[f"{s}/{t}"] = n
        units_meta[tag] = dict(
            n_tasks=len(tasks), tasks_with_events=per_task,
            proxy_grade=sorted(set(U["grade"].tolist())),
            n_episodes=int(len(U["nq"])), n_fail=int((U["success"] == 0).sum()),
            n_static_events_fail=int(U["is_event"].sum()),
            n_static_events_succ_excluded=int(((U["onset"] >= 0) & (U["success"] == 1)).sum()),
            n_control_episodes=int(U["is_ctrl"].sum()),
            n_groups_total=U["n_groups"], n_groups_paired=len(paired),
            n_events_in_paired_groups=n_ev_paired,
            n_events_orphan_no_control=n_orphan,
            underpowered=bool(n_ev_paired < MIN_EVENTS),
            onset_min=int(onsets.min()), onset_median=float(np.median(onsets)),
            onset_max=int(onsets.max()))
        print(f"[{time.time()-t0:6.1f}s] {tag}: ev={int(U['is_event'].sum())} "
              f"paired={n_ev_paired} orphan={n_orphan} groups={len(paired)}/{U['n_groups']} "
              f"ctrl={int(U['is_ctrl'].sum())}", flush=True)

        recs, eff_raw = run_family(U, U["sig"], SIG9, tag, "raw")
        res_vals, n_cells = residual_values(U, SIG9)
        recs_r, eff_res = run_family(U, res_vals, SIG9, tag, "resid_mob1w8")
        units_meta[tag]["n_residual_cells"] = n_cells
        for r in recs + recs_r:
            r.update(unit=tag, experiment="E3", event="static")
        eff_rows += recs + recs_r
        eff_store[tag] = (eff_raw, eff_res)
        m, se = eff_mean_se(eff_raw, len(SIG9), len(LEADS))
        plot_units.append((tag, m, se))
        presence_curve[tag] = [r["n_used"] for r in pa]
        print(f"[{time.time()-t0:6.1f}s] {tag}: family done", flush=True)

    df = pd.DataFrame(eff_rows)[
        ["experiment", "unit", "event", "leg", "signal", "lead", "effect", "ci_lo", "ci_hi",
         "p_maxT", "n_groups", "n_groups_pos", "n_groups_neg", "n_events_used", "n_pairs"]]
    df.round(6).to_csv(os.path.join(OUT, "aligned_effects.csv"), index=False)
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
                             n_units_sig_resid=int(sum(u["sig_resid"] for u in per_unit.values())))
        print(f"  {key:14s} {sig}@{lead:+d} exp={direction} -> {v} "
              f"({[round(u['effect'],3) for u in per_unit.values()]}, "
              f"p={[round(u['p_maxT'],4) for u in per_unit.values()]})", flush=True)

    plot_aligned(plot_units, SIG9, LEADS, os.path.join(OUT, "fig_aligned_curves.png"),
                 "E3  static onset-aligned routing signals (event vs matched same-q controls)",
                 presence=presence_curve)

    summary = dict(
        experiment="E3 static 表型 (flattening + shared support)",
        protocol="PROTOCOL.md §4 / §5-E3; AUDIT.md §4 §7 §8",
        seed=SEED, nperm=NPERM, leads=LEADS, signals=SIG9,
        family="9 signals x 7 leads = 63 cells per unit per leg (8 core + min_mob_k); "
               "raw and resid_mob1w8 are two independent maxT families",
        min_mobk_definition="min over k in 2..min(8,q) of mob_k at the same absolute query; "
                            "NaN for q<2. The number of available lags is identical for event "
                            "and control episodes because the comparison is matched at the "
                            "same absolute q.",
        effect_definition="group AUC(event vs same-group same-absolute-q controls) - 0.5, "
                          "pairs pooled over the group's event episodes (frozen trap 口径)",
        control_definition="all episodes in the same (task,scene) group with static_onset_q<0, "
                           "alive at the matched q (includes other failures and successes)",
        event_definition="failure episodes with static_onset_q>=0; success episodes carrying a "
                         "static label excluded from both sides",
        cross_unit_aggregation="unweighted group mean (phenotype/stats.py::groupwise_signflip_maxt)",
        note_red_flags="articulated-mechanism red flag (AUDIT §7.1/§8.4) applies to the loop/trap "
                       "channel only; static channel retained for all tasks",
        min_events_for_inference=MIN_EVENTS,
        units=units_meta, prenamed=verdicts, presence_audit=audit_rows,
        runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{time.time()-t0:6.1f}s] E3 done", flush=True)


if __name__ == "__main__":
    main()
