"""E3 static 表型（flattening + shared-support），PROTOCOL §5-E3 + §8 Amendment 1。

单位：U1 = main16x32 SCENE8（full，146 static 事件）；U2 = grid50x8 SCENE8（full，70）；
U3 = grid50x8 objaware 池（全部非 SCENE8 任务中有 static 事件者，约 110；组=(task,scene)）。
objaware static 代理已验证可信（sens/spec ≥0.97，AUDIT §4）；SCENE3 与 open_middle_drawer
仅 loop/trap 通道无效，static 通道可用（AUDIT §7.1）。

族：主族 = 8 信号 × 7 lead = 56 格/单位（冻结）；min_{k≥2} mob_k 为 E3 预 named 信号但
不在 8 信号族内，另立 7 lead 小族（各自 maxT），raw 与 resid 同样处理。
预 named：S_ent@0↑（变平）、C@0↑（共享支持集）、min_mobk@0↓（周期性）。
"""

import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E2DIR = os.path.join(os.path.dirname(HERE), "E2_loop_sequence")
sys.path.insert(0, E2DIR)
from aligned_lib import (BASE, LEADS, NPERM, S8, SEED, SIG8, add_resid,  # noqa: E402
                         build_unit, compute_eff, eff_mean_se, effect_rows,
                         family_stats, plot_aligned, rng_for, write_effects_csv)

EXP = "E3_static"
OUT = HERE
DISP = [d for _, d in SIG8]
L = len(LEADS)


def u3_specs():
    """grid50x8 非 SCENE8、静态事件 ≥1 的全部任务。"""
    specs = []
    counts = {}
    for suite in ("libero_goal", "libero_long", "libero_object", "libero_spatial"):
        d = os.path.join(BASE, "events", "grid50x8", suite)
        for task in sorted(os.listdir(d)):
            if task == S8:
                continue
            ev = list(csv.DictReader(open(os.path.join(d, task, "events.csv"))))
            n = sum(1 for e in ev if int(e["static_onset_q"]) >= 0)
            if n:
                specs.append(("grid50x8", suite, task))
                counts[f"{suite}/{task}"] = n
    return specs, counts


NAMED = [("S_ent", 0, +1, "flattening"), ("C", 0, +1, "shared-support"),
         ("min_mobk", 0, -1, "periodicity (7 格小族)")]
NAMED_KEYS = {(s, l) for s, l, _, _ in NAMED}


def main():
    t0 = time.time()
    specs3, counts3 = u3_specs()
    units = [
        ("U1_main16x32_S8", [("main16x32", "libero_long", S8)]),
        ("U2_grid50x8_S8", [("grid50x8", "libero_long", S8)]),
        ("U3_grid50x8_objaware", specs3),
    ]
    all_rows = []
    summary = dict(
        protocol="E3 v1 (trap 口径冻结)", seed=SEED, nperm=NPERM, leads=list(LEADS),
        family=("主族 8 signals x 7 leads = 56 cells; min_mobk 另立 7-lead 小族"
                " (named cell 不在 8 信号族内, 单独 maxT, raw/resid 同)"),
        effect="per-event AUC(event vs same-group same-absolute-q no-static controls) - 0.5, group-averaged",
        controls="static_onset_q == -1 (含其他失败与 loop 事件集)",
        min_mobk_def="逐行 min_{k=2..8} mob_k (q<k 的 k 不可用; q<2 为 NaN; 比较恒在同 q 内)",
        u3_tasks_static_counts=counts3,
        units={}, named_cells=[], presence={}, runtime_s=None)

    units_ms = []
    mobk_panel = {}
    for tag, specs in units:
        print(f"[{time.time()-t0:6.1f}s] {tag} load ({len(specs)} tasks)", flush=True)
        U = build_unit(tag, specs, "static_onset_q", fail_only=False)
        add_resid(U, DISP + ["min_mobk"])
        legs = {}
        for leg in ("raw", "resid_mob1w8"):
            arrkey = "raw" if leg == "raw" else "resid"
            eff, used, present, present_ctrl = compute_eff(U, arrkey, DISP)
            st = family_stats(eff, rng_for("E3", tag, leg))
            effm, usedm, _, _ = compute_eff(U, arrkey, ["min_mobk"])
            stm = family_stats(effm, rng_for("E3", tag, leg + "_mobk"))
            legs[leg] = dict(eff=eff, used=used, present=present,
                             present_ctrl=present_ctrl, st=st,
                             effm=effm, usedm=usedm, stm=stm)
            all_rows += effect_rows(EXP, tag, leg, "56", DISP, LEADS, eff, st,
                                    used, present, U["n_events"], NAMED_KEYS)
            all_rows += effect_rows(EXP, tag, leg, "mobk7", ["min_mobk"], LEADS,
                                    effm, stm, usedm, present, U["n_events"],
                                    NAMED_KEYS)
        R = legs["raw"]
        mean, se = eff_mean_se(R["eff"], len(DISP), L)
        lab = ("U1 main S8" if "main" in tag else
               "U2 grid S8" if "S8" in tag else "U3 grid pool")
        units_ms.append((lab, mean, se))
        mm, ms = eff_mean_se(R["effm"], 1, L)
        mobk_panel[lab] = (mm[0], ms[0])

        summary["units"][tag] = dict(
            n_tasks=len(specs), corpus=specs[0][0], proxy_grades=U["proxy_grades"],
            n_events=U["n_events"], n_ctrl_episodes=U["n_ctrl"],
            n_groups_with_event=U["n_groups"],
            onset_med=float(np.median([o for g in U["ev_groups"]
                                       for _, o in U["groups"][g]["ev"]])))
        summary["presence"][tag] = dict(
            leads=list(LEADS), n_events_total=U["n_events"],
            present=R["present"].tolist(),
            present_with_ctrl=R["present_ctrl"].tolist(),
            used_raw_per_signal={d: R["used"][si].tolist()
                                 for si, d in enumerate(DISP)},
            used_raw_min_mobk=R["usedm"][0].tolist(),
            used_resid_per_signal={d: legs["resid_mob1w8"]["used"][si].tolist()
                                   for si, d in enumerate(DISP)},
            used_resid_min_mobk=legs["resid_mob1w8"]["usedm"][0].tolist())

        for sig, lead, direc, label in NAMED:
            li = list(LEADS).index(lead)
            row = dict(unit=tag, signal=sig, lead=lead,
                       expected="up" if direc > 0 else "down", label=label)
            for leg in ("raw", "resid_mob1w8"):
                Lg = legs[leg]
                if sig == "min_mobk":
                    st, used_si = Lg["stm"], Lg["usedm"][0]
                    c = li
                else:
                    st, used_si = Lg["st"], Lg["used"][DISP.index(sig)]
                    c = DISP.index(sig) * L + li
                row[leg] = dict(eff=_f(st["obs"][c]),
                                ci=[_f(st["ci_lo"][c]), _f(st["ci_hi"][c])],
                                p_maxT=_f(st["p"][c]), n_groups=int(st["n_groups"][c]),
                                n_events_used=int(used_si[li]))
            summary["named_cells"].append(row)
        print(f"[{time.time()-t0:6.1f}s] {tag} done "
              f"(ev={U['n_events']} ctrl={U['n_ctrl']} grp={U['n_groups']})", flush=True)

    write_effects_csv(os.path.join(OUT, "aligned_effects.csv"), all_rows)
    plot_aligned(units_ms, DISP, LEADS, os.path.join(OUT, "fig_aligned_curves.png"),
                 "E3 static: event-aligned effects vs lead (mean +/- between-group SE; "
                 "AUC-0.5 of event vs same-group same-q no-static controls)",
                 extra_panel=("min_mobk (7-lead mini-family)", mobk_panel))
    summary["runtime_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{time.time()-t0:6.1f}s] all done")


def _f(x):
    return round(float(x), 4) if np.isfinite(x) else None


if __name__ == "__main__":
    main()
