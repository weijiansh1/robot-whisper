#!/usr/bin/env python3
"""合并 summary.json + diagnostics.json 的头条量、出图、打印交付用表格。

依赖顺序：calibrate.py -> diagnose.py -> finalize.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
S = json.loads((HERE / "summary.json").read_text())
D = json.loads((HERE / "diagnostics.json").read_text())
PE = pd.read_csv(HERE / "per_episode.csv")
UNITS = ["main16x32", "grid50x8"]


def chan(unit, name, mode="frozen"):
    for c in S["units"][unit]["channels"]:
        if c["channel"] == name and c["mode"] == mode:
            return c
    raise KeyError(name)


head = {}
for u in UNITS:
    p = PE[PE.corpus == u]
    clean = p.cls == "succ_clean"
    any_alarm = (p.alarm_B_loop >= 0) | (p.alarm_B_static >= 0)
    pst = pd.DataFrame(S["units"][u]["paired_BvsC"]["static"])
    plp = pd.DataFrame(S["units"][u]["paired_BvsC"]["loop"])
    head[u] = dict(
        n_eps=int(len(p)), n_clean=int(clean.sum()),
        fa_clean_any_channel=float(any_alarm[clean].mean()),
        fa_clean_loop_ch=float((p.alarm_B_loop >= 0)[clean].mean()),
        fa_clean_static_ch=float((p.alarm_B_static >= 0)[clean].mean()),
        fa_clean_armC=float((p.alarm_C_sticky >= 0)[clean].mean()),
        fa_clean_armT=float((p.alarm_T_len >= 0)[clean].mean()),
        detect_loop=chan(u, "B_loop")["detect"],
        detect_loop_preonset=chan(u, "B_loop")["detect_preonset"],
        lead_loop=chan(u, "B_loop")["lead"],
        detect_static=chan(u, "B_static")["detect"],
        detect_static_preonset=chan(u, "B_static")["detect_preonset"],
        lead_static=chan(u, "B_static")["lead"],
        armC_static=dict(detect=chan(u, "C_sticky@static")["detect"],
                         pre=chan(u, "C_sticky@static")["detect_preonset"],
                         lead=chan(u, "C_sticky@static")["lead"]["med"]),
        armC_loop=dict(detect=chan(u, "C_sticky@loop")["detect"],
                       pre=chan(u, "C_sticky@loop")["detect_preonset"]),
        armT=dict(pre_loop=chan(u, "T_len@loop")["detect_preonset"],
                  pre_static=chan(u, "T_len@static")["detect_preonset"]),
        paired_static_at_fa05={k: pst[pst.fa_level == 0.05][k].iloc[0]
                               for k in ("K_B", "K_C", "fa_B", "fa_C", "d_fa", "d_det",
                                         "d_pre", "mcnemar_B_only", "mcnemar_C_only",
                                         "mcnemar_p", "lead_B", "lead_C")},
        paired_loop_at_fa05={k: plp[plp.fa_level == 0.05][k].iloc[0]
                             for k in ("K_B", "K_C", "fa_B", "fa_C", "d_fa", "d_det",
                                       "d_pre", "mcnemar_B_only", "mcnemar_C_only",
                                       "mcnemar_p", "lead_B", "lead_C")},
        fa_recovery_loop_loopch=chan(u, "B_loop")["fa_succ_loop"],
        n_recovery_loop=chan(u, "B_loop")["n_succ_loop"],
        fa_recovery_static_staticch=chan(u, "B_static")["fa_succ_static"],
        n_recovery_static=chan(u, "B_static")["n_succ_static"],
        recovery_loop_maxrun_switch=[r["maxrun_switch"]
                                     for r in D["units"][u]["recovery_loop_detail"]],
        recovery_static_maxrun_flatten=[r["maxrun_flatten"]
                                        for r in D["units"][u]["succ_static_detail"]],
        clean_succ_maxrun_switch_pctl=D["units"][u]["clean_succ_maxrun_switch_pctl"],
        probes=S["units"][u]["probes"])
S["headline"] = head
S["diagnostics_file"] = "diagnostics.json"
(HERE / "summary.json").write_text(json.dumps(S, indent=1, default=float))
print("== headline ==")
print(json.dumps(head, indent=1, default=float))

# ------------------------------------------------------------------ 图
ks = np.arange(1, S["kmax"] + 1)
fig, ax = plt.subplots(2, 4, figsize=(21.5, 9.2))

a = ax[0, 0]
for b, lab, col in (("switch", "switching/desync (B loop ch.)", "tab:red"),
                    ("flatten", "flatten/shared (B static ch.)", "tab:blue"),
                    ("sticky", "sticky = arm C (mob1_w8)", "tab:green")):
    a.semilogy(ks, np.maximum(S["fa_curves_pooled"][b], 8e-4), color=col, label=lab)
for b, K, col in (("switch", S["params_frozen"]["K_LOOP"], "tab:red"),
                  ("flatten", S["params_frozen"]["M_STATIC"], "tab:blue"),
                  ("sticky", S["params_frozen"]["K_C"], "tab:green")):
    a.axvline(K, color=col, ls=":", lw=1)
a.axhline(S["fa_target"], color="k", ls="--", lw=1, label="pre-declared FA tolerance 0.10")
a.set_xlim(1, 26); a.set_ylim(7e-4, 1.4)
a.set_xlabel("dwell knob K (consecutive in-band queries)")
a.set_ylabel("per-episode FA on clean successes (pooled)")
a.set_title("A. knob curve, success set only (LOGO: 66/66 folds agree)", fontsize=10)
a.legend(fontsize=7, loc="lower left")

for j, u in enumerate(UNITS):
    a = ax[0, 1 + j]
    m = pd.read_csv(HERE / f"matched_fa_{u}.csv")
    for arm, lab, col, mk, ls in (
            ("flatten->static_onset_q", "B static (flatten dwell)", "tab:blue", "o", "-"),
            ("sticky->static_onset_q", "C sticky (mob1_w8) @static", "tab:green", "s", "-"),
            ("T_len->static_onset_q", "T episode-length @static", "0.4", "^", "-"),
            ("switch->loop_onset_q", "B loop (switch dwell)", "tab:red", "o", "--"),
            ("sticky->loop_onset_q", "C sticky @loop", "tab:olive", "s", "--"),
            ("T_len->loop_onset_q", "T episode-length @loop", "0.7", "^", "--")):
        d = m[m.arm == arm].sort_values("fa_clean")
        a.plot(d.fa_clean, d.detect_preonset, ls, marker=mk, color=col, label=lab, ms=4)
    a.set_xscale("log"); a.set_xlim(3e-3, .35); a.set_ylim(0, 1)
    a.set_xlabel("per-episode FA on clean successes")
    a.set_ylabel("pre-onset detection rate")
    a.set_title(f"B{j+1}. matched-FA arms - {u}", fontsize=10)
    if j == 0:
        a.legend(fontsize=7, loc="upper left")

a = ax[0, 3]
lbl, bo, co, pv = [], [], [], []
for u in UNITS:
    for tgt in ("static", "loop"):
        r = pd.DataFrame(S["units"][u]["paired_BvsC"][tgt])
        r = r[r.fa_level == 0.05].iloc[0]
        lbl.append(f"{tgt}\n{u.split('1')[0].split('5')[0]}")
        bo.append(r.mcnemar_B_only); co.append(-r.mcnemar_C_only); pv.append(r.mcnemar_p)
x = np.arange(len(lbl))
a.bar(x, bo, .6, color="tab:blue", label="pre-onset by B only")
a.bar(x, co, .6, color="tab:green", label="pre-onset by C only")
for i, (b, c, p) in enumerate(zip(bo, co, pv)):
    a.text(i, b + 1.5, f"p={p:.1e}" if p < .01 else f"p={p:.2f}", ha="center", fontsize=7)
a.axhline(0, color="k", lw=1)
a.set_xticks(x); a.set_xticklabels(lbl, fontsize=8)
a.set_ylabel("discordant episodes (McNemar)")
a.set_title("D. paired B vs C at matched FA=0.05\n(same run, same episodes)", fontsize=10)
a.legend(fontsize=7)

for j, u in enumerate(UNITS):
    a = ax[1, j]
    p = PE[PE.corpus == u]
    fs = p.cls.isin(["fail_static", "fail_both"]) & (p.alarm_B_static >= 0)
    fl = p.cls.isin(["fail_loop", "fail_both"]) & (p.alarm_B_loop >= 0)
    cs = p.cls.isin(["fail_static", "fail_both"]) & (p.alarm_C_sticky >= 0)
    bins = np.arange(-30, 21, 2)
    a.hist((p.alarm_B_static - p.static_onset_q)[fs], bins=bins, alpha=.6,
           color="tab:blue", label="B static (flatten)")
    a.hist((p.alarm_C_sticky - p.static_onset_q)[cs], bins=bins, alpha=.5,
           color="tab:green", label="C sticky @static")
    a.hist((p.alarm_B_loop - p.loop_onset_q)[fl], bins=bins, alpha=.5,
           color="tab:red", label="B loop (switch)")
    a.axvline(0, color="k", lw=1)
    a.axvline(-8, color="0.4", ls=":", lw=1)
    a.text(-8.3, a.get_ylim()[1] * .8, "earliest physical\nstall evidence\n(static label lag -8)",
           fontsize=7, ha="right", color="0.35")
    a.set_xlabel("alarm q - physical onset q   (negative = early)")
    a.set_ylabel("episodes")
    a.set_title(f"C{j+1}. lead-time distribution - {u}", fontsize=10)
    a.legend(fontsize=7)

a = ax[1, 2]
xs, labs, cols = [], [], []
for u, sh in (("main16x32", ""), ("grid50x8", "'")):
    p = PE[PE.corpus == u]
    for cl, col in (("succ_clean", "0.6"), ("succ_loop", "tab:orange"),
                    ("fail_loop", "tab:red")):
        sel = [cl] + (["fail_both"] if cl == "fail_loop" else [])
        xs.append(p.loc[p.cls.isin(sel), "maxrun_switch_cap"].to_numpy())
        labs.append(f"{cl}{sh}")
        cols.append(col)
bp = a.boxplot(xs, labels=labs, patch_artist=True, widths=.6, showfliers=False)
for b, c in zip(bp["boxes"], cols):
    b.set_facecolor(c); b.set_alpha(.6)
for i, xx in enumerate(xs):
    a.scatter(np.full(len(xx), i + 1) + np.random.default_rng(0).normal(0, .06, len(xx)), xx,
              s=4, color="k", alpha=.25 if len(xx) > 20 else 1.0, zorder=3)
a.set_ylim(1, 15.5)
a.axhline(6.5, color="tab:orange", ls="--", lw=1)
a.text(3.5, 14.6, "dashed: all 6 recovery loops <= 6\n(K_LOOP=11 never reached)",
       fontsize=8, color="tab:orange", ha="center", va="top")
a.set_ylabel("max consecutive dwell, switching band\n(q in [8,38], exposure-capped)")
a.set_title("E. recovery loop leaves the band (main | grid')", fontsize=10)
a.tick_params(axis="x", labelrotation=30, labelsize=7)

a = ax[1, 3]
names = ["maxrun_cons1", "maxrun_sticky", "maxrun_flatten", "maxrun_inst0",
         "maxrun_state5"]
disp = ["cons=1 (1 axis)", "sticky = arm C (1 axis)", "flatten (2-axis, DELIVERED)",
        "inst=0 (1 axis)", "state5 (E8 P_self analogue)"]
w = .38
for k, (u, c) in enumerate(zip(UNITS, ("tab:purple", "tab:cyan"))):
    v = [D["units"][u]["D2_maxrun_auc_matched"]["fatalstatic_vs_clean"][n]["auc"]
         for n in names]
    a.barh(np.arange(len(names)) + (k - .5) * w, v, w, color=c, label=u, alpha=.8)
a.axvline(.5, color="k", lw=1)
a.set_yticks(np.arange(len(names)))
a.set_yticklabels(disp, fontsize=7.5)
a.set_xlim(.4, 1.0)
a.set_xlabel("within-group paired AUC, fatal static vs clean success\n(exposure-matched subgroup)")
a.set_title("F. band-definition diagnosis: the 2-axis\nconjunction is a net loss", fontsize=10)
a.legend(fontsize=7, loc="lower right")
a.invert_yaxis()

fig.suptitle("Design B - macrostate band dwell detector - SCENE8 calibration set "
             "(main16x32 512 eps / grid50x8 400 eps), holdout untouched", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(HERE / "fig_calibration.png", dpi=135)
print("[fig] fig_calibration.png")
