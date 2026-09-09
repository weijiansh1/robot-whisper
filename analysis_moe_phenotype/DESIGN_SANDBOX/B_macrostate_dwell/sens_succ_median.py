#!/usr/bin/env python3
"""敏感性分析（**不交付**）：把二分点从「组内无标签中位」换成「组内成功集中位」。

这是任务书原文的口径，但它让守门探针 P1（组内 success 置换后报警逐位不变）必然失败，
所以不进 detector.py。本脚本只为量化「P1 合规付出了多少灵敏度」。
输出 sens_succ_median.json。
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype")
sys.path.insert(0, str(HERE))
from detector import AXES, PARAMS, _first_cross, load_unit  # noqa: E402

TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = [("main16x32", "libero_long", TASK), ("grid50x8", "libero_long", TASK)]
KS = np.arange(1, 61)
MIN_REF_EPS, MIN_REF_ROWS = PARAMS["MIN_REF_EPS"], PARAMS["MIN_REF_ROWS"]


def build(corpus, suite, task):
    df, raw = load_unit(ROOT / "features" / corpus / suite / task / "rows.npz", "scene")
    ev = pd.read_csv(ROOT / "events" / corpus / suite / task / "events.csv")
    succ_of = dict(zip(ev.episode_id, ev.success))
    ep = df.episode_id.to_numpy()
    q = df.q.to_numpy()
    grp = df.group.to_numpy()
    succ_row = np.array([succ_of[int(e)] for e in ep])
    win = q >= PARAMS["QMIN"]
    for a in AXES:
        win &= np.isfinite(raw[a])
    idx = np.flatnonzero(win & (succ_row == 1))          # 成功集参考行
    rows = []
    cross = {}
    starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
    ends = np.r_[starts[1:], len(ep)]
    for s, t in zip(starts, ends):
        e = int(ep[s])
        g = int(grp[s])
        cand = idx[(grp[idx] == g) & (ep[idx] != e)]
        lvl = "scene"
        if len(np.unique(ep[cand])) < MIN_REF_EPS or len(cand) < MIN_REF_ROWS:
            cand = idx[ep[idx] != e]
            lvl = "task"
        med = {a: float(np.median(raw[a][cand])) for a in AXES}
        bits, ok = {}, np.ones(t - s, bool)
        for a in AXES:
            x = raw[a][s:t]
            ok &= np.isfinite(x)
            bits[a] = x > med[a]
        ok &= q[s:t] >= PARAMS["QMIN"]
        band = {"switch": bits["inst"] & ~bits["cons"] & ok,
                "flatten": ~bits["inst"] & bits["cons"] & ok,
                "sticky": bits["stick"] & ok}
        cross[e] = {k: _first_cross(v, q[s:t]) for k, v in band.items()}
        rows.append(dict(episode_id=e, group=g, ref_level=lvl, n_win=int(ok.sum())))
    T = pd.DataFrame(rows).merge(
        ev[["episode_id", "success", "n_queries", "loop_onset_q", "static_onset_q"]],
        on="episode_id", validate="1:1")
    hl, hs = T.loop_onset_q >= 0, T.static_onset_q >= 0
    T["cls"] = np.where(T.success == 0,
                        np.where(hl & hs, "fail_both",
                                 np.where(hl, "fail_loop",
                                          np.where(hs, "fail_static", "fail_noevent"))),
                        np.where(hl, "succ_loop", np.where(hs, "succ_static", "succ_clean")))
    return T, cross


def main():
    out = {}
    tabs, crs = {}, {}
    for corpus, suite, task in UNITS:
        T, cr = build(corpus, suite, task)
        tabs[corpus], crs[corpus] = T, cr
    # 旋钮：同一规则（干净成功集池化 FA <= FA_TARGET 的最小 K）
    knob = {}
    for band, name in (("switch", "K_LOOP"), ("flatten", "M_STATIC"), ("sticky", "K_C")):
        fired = []
        for c in tabs:
            T, cr = tabs[c], crs[c]
            A = np.array([[len(cr[e][band]) >= K for K in KS] for e in T.episode_id])
            fired.append(A[(T.cls == "succ_clean").to_numpy()])
        fa = np.vstack(fired).mean(0)
        ok = np.flatnonzero(fa <= PARAMS["FA_TARGET"])
        knob[name] = int(KS[ok[0]]) if len(ok) else int(KS[-1])
    out["knobs"] = knob
    for c in tabs:
        T, cr = tabs[c], crs[c]
        cls = T.cls.to_numpy()
        u = dict(ref_level=T.ref_level.value_counts().to_dict())
        for band, K, tgt, onset_col, nm in (
                ("switch", knob["K_LOOP"], ["fail_loop", "fail_both"], "loop_onset_q", "B_loop"),
                ("flatten", knob["M_STATIC"], ["fail_static", "fail_both"], "static_onset_q",
                 "B_static"),
                ("sticky", knob["K_C"], ["fail_static", "fail_both"], "static_onset_q",
                 "C_sticky@static")):
            a = np.array([cr[e][band][K - 1] if len(cr[e][band]) >= K else -1
                          for e in T.episode_id])
            f = a >= 0
            m = np.isin(cls, tgt)
            onset = T[onset_col].to_numpy()
            lead = (a - onset)[m & f]
            u[nm] = dict(K=K, fa_clean=round(float(f[cls == "succ_clean"].mean()), 4),
                         n_target=int(m.sum()), detect=round(float(f[m].mean()), 4),
                         detect_preonset=round(float((m & f & (a <= onset)).sum()
                                                     / max(m.sum(), 1)), 4),
                         lead_med=float(np.median(lead)) if len(lead) else None,
                         fa_succ_loop=round(float(f[cls == "succ_loop"].mean()), 4),
                         fa_succ_static=round(float(f[cls == "succ_static"].mean()), 4))
        out[c] = u
    (HERE / "sens_succ_median.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
