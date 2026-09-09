#!/usr/bin/env python3
"""方案 B 诊断（不改交付规格，只回答「为什么行/不行」）。

内容：
  D1 曝光严格匹配子群（n_win_cap 满窗）上的驻留判别力 —— 主要族之外的第二族，预声明 6 格；
  D2 单轴 vs 二轴合取 vs 五轴同态：带定义粒度的诊断（描述性，不做显著性主张，不进交付）；
  D3 占用率（曝光归一）与「驻留长度」的对照 —— 区分「更常在带内」与「连续待更久」；
  D4 对臂 C（mob1_w8 驻留）与曝光（n_win）的组内秩残差；
  D5 等 FA 下的四臂横向对比（B_loop / B_static / C_sticky / T_len）；
  D6 带进入时刻（alarm − K + 1）相对 onset 的提前量，以及物理 onset 规则自身的确认滞后。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "phenotype"))

from detector import PARAMS, AXES, build_reference, load_unit, _first_cross  # noqa: E402
from stats import joint_maxt, paired_auc, residualise  # noqa: E402

SEED = 20260904
NPERM = 2000
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = [("main16x32", "libero_long", TASK), ("grid50x8", "libero_long", TASK)]
QCAP = 38
KMAX = 60

BAND_DEFS = {
    "switch": lambda b: b["inst"] & ~b["cons"],        # 交付：loop 通道
    "flatten": lambda b: ~b["inst"] & b["cons"],       # 交付：static 通道
    "sticky": lambda b: b["stick"],                    # 臂 C
    "inst1": lambda b: b["inst"],                      # 诊断：单轴
    "cons0": lambda b: ~b["cons"],
    "inst0": lambda b: ~b["inst"],
    "cons1": lambda b: b["cons"],
}


def unit_table(corpus, suite, task, qmax):
    p = ROOT / "features" / corpus / suite / task / "rows.npz"
    df, raw = load_unit(p, "scene")
    pars = dict(PARAMS)
    ref = build_reference(df, raw, pars)
    ep = df["episode_id"].to_numpy()
    q = df["q"].to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
    ends = np.r_[starts[1:], len(ep)]
    rows, cross = [], {}
    for s, t in zip(starts, ends):
        e = int(ep[s])
        r = ref[e]
        qq = q[s:t]
        if r["level"] == "none":
            continue
        bits, ok = {}, np.ones(t - s, bool)
        for a in AXES:
            x = raw[a][s:t]
            ok &= np.isfinite(x)
            bits[a] = x > r["med"][a]
        ok &= qq >= pars["QMIN"]
        if qmax is not None:
            ok &= qq <= qmax
        rec = dict(episode_id=e, group=int(df["group"].to_numpy()[s]), n_win=int(ok.sum()))
        cr = {}
        for name, fn in BAND_DEFS.items():
            bd = fn(bits) & ok
            c = _first_cross(bd, qq)
            cr[name] = c
            rec[f"maxrun_{name}"] = int(len(c))
            rec[f"occ_{name}"] = float(bd.sum() / ok.sum()) if ok.sum() else np.nan
        code = np.zeros(t - s, np.int64)
        for b, a in enumerate(AXES):
            code |= bits[a].astype(np.int64) << b
        code2 = (bits["inst"].astype(np.int64)) | (bits["cons"].astype(np.int64) << 1)
        for nm, cd in (("state5", code), ("state2", code2)):
            same = np.zeros(t - s, bool)
            same[1:] = ok[1:] & ok[:-1] & (cd[1:] == cd[:-1])
            c = _first_cross(same, qq)
            cr[nm] = c
            rec[f"maxrun_{nm}"] = int(len(c))
            rec[f"occ_{nm}"] = float(same.sum() / max(ok.sum(), 1))
        rows.append(rec)
        cross[e] = cr
    T = pd.DataFrame(rows)
    ev = pd.read_csv(ROOT / "events" / corpus / suite / task / "events.csv")
    T = T.merge(ev[["episode_id", "success", "n_queries", "loop_onset_q", "static_onset_q"]],
                on="episode_id", validate="1:1")
    T["gkey"] = corpus + "|" + T["group"].astype(str)
    hl, hs = T.loop_onset_q >= 0, T.static_onset_q >= 0
    T["cls"] = np.where(T.success == 0,
                        np.where(hl & hs, "fail_both",
                                 np.where(hl, "fail_loop",
                                          np.where(hs, "fail_static", "fail_noevent"))),
                        np.where(hl, "succ_loop", np.where(hs, "succ_static", "succ_clean")))
    return T, cross


ALL_BANDS = list(BAND_DEFS) + ["state5", "state2"]


def auc_table(T, mask_pos, mask_neg, cols):
    y = np.where(mask_pos, 1, 0)
    m = mask_pos | mask_neg
    out = {}
    for c in cols:
        a, n = paired_auc(T[c].to_numpy().astype(float)[m], y[m], T.gkey.to_numpy()[m])
        out[c] = dict(auc=None if not np.isfinite(a) else round(float(a), 4), n_pairs=int(n))
    return out


def _cells6(T, suffix):
    cls = T.cls.to_numpy()
    grp = T.gkey.to_numpy()
    fl = np.isin(cls, ["fail_loop", "fail_both"])
    fs = np.isin(cls, ["fail_static", "fail_both"])
    cl = cls == "succ_clean"
    rl = cls == "succ_loop"
    return [("switch|fatalloop_vs_clean", T[f"maxrun_switch{suffix}"].to_numpy().astype(float),
             fl | cl, grp),
            ("flatten|fatalstatic_vs_clean",
             T[f"maxrun_flatten{suffix}"].to_numpy().astype(float), fs | cl, grp),
            ("sticky_C|fatalloop_vs_clean", T[f"maxrun_sticky{suffix}"].to_numpy().astype(float),
             fl | cl, grp),
            ("sticky_C|fatalstatic_vs_clean",
             T[f"maxrun_sticky{suffix}"].to_numpy().astype(float), fs | cl, grp),
            ("state5|fatalany_vs_clean", T[f"maxrun_state5{suffix}"].to_numpy().astype(float),
             fl | fs | cl, grp),
            ("switch|fatalloop_vs_recoveryloop",
             T[f"maxrun_switch{suffix}"].to_numpy().astype(float), fl | rl, grp)]


def _run_family(T, cells, nperm=NPERM):
    cls = T.cls.to_numpy()
    grp = T.gkey.to_numpy()
    ek = np.arange(len(T))
    lab = {int(i): (1 if cls[i].startswith("fail") else 0) for i in ek}
    gof = {int(i): grp[i] for i in ek}
    obs, p = joint_maxt(cells, ek, lab, gof, nperm, np.random.default_rng(SEED))
    out = []
    for c in cells:
        n = c[0]
        _, p1 = joint_maxt([c], ek, lab, gof, nperm, np.random.default_rng(SEED))
        out.append(dict(cell=n, auc=None if not np.isfinite(obs[n][0]) else round(float(obs[n][0]), 4),
                        n_pairs=int(obs[n][1]), p_maxT=p.get(n), p_uncorrected=p1.get(n)))
    return out


def family6(T, suffix, nperm=NPERM):
    return _run_family(T, _cells6(T, suffix), nperm)


def residual_family(T, nperm=NPERM):
    """对臂 C（mob1_w8 驻留）与曝光（n_win）的组内秩残差，4 格 maxT。"""
    cls = T.cls.to_numpy()
    grp = T.gkey.to_numpy()
    fl = np.isin(cls, ["fail_loop", "fail_both"])
    fs = np.isin(cls, ["fail_static", "fail_both"])
    cl = cls == "succ_clean"
    cells = []
    for tgt, base, pos in (("maxrun_switch", "maxrun_sticky", fl),
                           ("maxrun_flatten", "maxrun_sticky", fs),
                           ("maxrun_switch", "n_win", fl),
                           ("maxrun_flatten", "n_win", fs)):
        r = residualise(T[tgt].to_numpy().astype(float), T[base].to_numpy().astype(float), grp)
        cells.append((f"{tgt}|resid_{base}", r, pos | cl, grp))
    return _run_family(T, cells, nperm)


def main():
    t0 = time.time()
    out = {"seed": SEED, "nperm": NPERM, "qcap": QCAP, "units": {}}
    for corpus, suite, task in UNITS:
        Tc, crc = unit_table(corpus, suite, task, QCAP)      # 截断（曝光对齐）
        Tf, crf = unit_table(corpus, suite, task, None)      # 未截断（操作口径）
        cls = Tc.cls.to_numpy()
        fl = np.isin(cls, ["fail_loop", "fail_both"])
        fs = np.isin(cls, ["fail_static", "fail_both"])
        cl = cls == "succ_clean"
        full = int(QCAP - PARAMS["QMIN"] + 1)
        matched = Tc.n_win.to_numpy() == full

        u = {}
        u["n_eps"] = int(len(Tc))
        u["cls_counts"] = Tc.cls.value_counts().to_dict()
        u["n_matched_full_window"] = int(matched.sum())
        u["matched_cls"] = pd.Series(cls[matched]).value_counts().to_dict()

        # D1 曝光严格匹配子群（第二族，预声明 6 格）
        Tm = Tc[matched].reset_index(drop=True)
        u["D1_matched_family6"] = family6(Tm, "") if len(Tm) else None
        # 族构成敏感性：把 n_pairs 极小的恢复性 loop 格移出族后重跑同一 maxT 程序
        u["D1b_matched_family5_no_tinycell"] = _run_family(Tm, _cells6(Tm, "")[:5])
        u["D1b_capped_family5_no_tinycell"] = _run_family(Tc, _cells6(Tc, "")[:5])

        # D2/D3 各带的驻留 vs 占用（描述性，无 maxT）
        cols_run = [f"maxrun_{b}" for b in ALL_BANDS]
        cols_occ = [f"occ_{b}" for b in ALL_BANDS]
        u["D2_maxrun_auc_capped"] = dict(
            fatalloop_vs_clean=auc_table(Tc, fl, cl, cols_run),
            fatalstatic_vs_clean=auc_table(Tc, fs, cl, cols_run))
        u["D2_maxrun_auc_matched"] = dict(
            fatalloop_vs_clean=auc_table(Tm, np.isin(Tm.cls, ["fail_loop", "fail_both"]),
                                         (Tm.cls == "succ_clean").to_numpy(), cols_run),
            fatalstatic_vs_clean=auc_table(Tm, np.isin(Tm.cls, ["fail_static", "fail_both"]),
                                           (Tm.cls == "succ_clean").to_numpy(), cols_run))
        u["D3_occ_auc_capped"] = dict(
            fatalloop_vs_clean=auc_table(Tc, fl, cl, cols_occ),
            fatalstatic_vs_clean=auc_table(Tc, fs, cl, cols_occ))
        u["D3_exposure_auc"] = auc_table(Tc, fl | fs, cl, ["n_win", "n_queries"])

        # D4 组内秩残差（对臂 C 的驻留、对曝光）
        g = Tc.gkey.to_numpy()
        res = {}
        for tgt, base in (("maxrun_switch", "maxrun_sticky"),
                          ("maxrun_flatten", "maxrun_sticky"),
                          ("maxrun_switch", "n_win"), ("maxrun_flatten", "n_win")):
            r = residualise(Tc[tgt].to_numpy().astype(float),
                            Tc[base].to_numpy().astype(float), g)
            pos = fl if tgt == "maxrun_switch" else fs
            m = pos | cl
            a, n = paired_auc(r[m], np.where(pos, 1, 0)[m], g[m])
            res[f"{tgt}|resid_{base}"] = dict(auc=None if not np.isfinite(a) else round(float(a), 4),
                                              n_pairs=int(n))
        u["D4_residual_auc_capped"] = res
        u["D4_residual_family_capped"] = residual_family(Tc)
        u["D4_residual_family_matched"] = residual_family(Tm) if len(Tm) else None

        # 恢复性 loop 逐集明细（n 极小，逐集列出）
        rl = Tc.cls == "succ_loop"
        u["recovery_loop_detail"] = Tc.loc[rl, ["episode_id", "group", "n_queries", "n_win",
                                                "loop_onset_q", "maxrun_switch", "maxrun_sticky",
                                                "maxrun_flatten"]].to_dict("records")
        u["succ_static_detail"] = Tc.loc[Tc.cls == "succ_static",
                                         ["episode_id", "group", "n_queries", "n_win",
                                          "static_onset_q", "maxrun_flatten", "maxrun_sticky"
                                          ]].to_dict("records")
        cln = Tc.loc[Tc.cls == "succ_clean", "maxrun_switch"].to_numpy()
        u["clean_succ_maxrun_switch_pctl"] = {str(p): float(np.percentile(cln, p))
                                              for p in (50, 75, 90, 95, 99)}

        # D5 等 FA 横向对比（未截断 = 操作口径）
        ks = np.arange(1, KMAX + 1)
        eps = Tf.episode_id.tolist()
        clsf = Tf.cls.to_numpy()
        cleanf = clsf == "succ_clean"
        comp = []
        for band, tgt_cls, onset_col in (("switch", ["fail_loop", "fail_both"], "loop_onset_q"),
                                         ("flatten", ["fail_static", "fail_both"],
                                          "static_onset_q"),
                                         ("sticky", ["fail_static", "fail_both"],
                                          "static_onset_q"),
                                         ("sticky", ["fail_loop", "fail_both"], "loop_onset_q")):
            A = np.full((len(Tf), len(ks)), -1, np.int64)
            for i, e in enumerate(eps):
                c = crf[e][band]
                for j, K in enumerate(ks):
                    if len(c) >= K:
                        A[i, j] = c[K - 1]
            onset = Tf[onset_col].to_numpy()
            m = np.isin(clsf, tgt_cls)
            for fa_t in (0.02, 0.05, 0.10, 0.20):
                fa = (A >= 0)[cleanf].mean(0)
                ok = np.flatnonzero(fa <= fa_t)
                if not len(ok):
                    continue
                j = ok[0]
                f = A[:, j] >= 0
                lead = (A[:, j] - onset)[m & f]
                entry = (A[:, j] - int(ks[j]) + 1 - onset)[m & f]
                comp.append(dict(arm=f"{band}->{onset_col}", fa_target=fa_t, K=int(ks[j]),
                                 fa_clean=round(float(fa[j]), 4),
                                 detect=round(float(f[m].mean()), 4),
                                 detect_preonset=round(float((m & f & (A[:, j] <= onset)).sum()
                                                             / max(m.sum(), 1)), 4),
                                 lead_med=float(np.median(lead)) if len(lead) else None,
                                 entry_lead_med=float(np.median(entry)) if len(entry) else None,
                                 fa_succ_loop=round(float(f[clsf == "succ_loop"].mean()), 4)
                                 if (clsf == "succ_loop").any() else None))
            # 平凡臂 T
        nq = Tf.n_queries.to_numpy()
        for fa_t in (0.02, 0.05, 0.10, 0.20):
            cand = [q0 for q0 in range(8, 60) if (nq[cleanf] > q0).mean() <= fa_t]
            if not cand:
                continue
            q0 = cand[0]
            f = nq > q0
            for onset_col, tgt_cls in (("loop_onset_q", ["fail_loop", "fail_both"]),
                                       ("static_onset_q", ["fail_static", "fail_both"])):
                onset = Tf[onset_col].to_numpy()
                m = np.isin(clsf, tgt_cls)
                lead = (q0 - onset)[m & f]
                comp.append(dict(arm=f"T_len->{onset_col}", fa_target=fa_t, K=int(q0),
                                 fa_clean=round(float(f[cleanf].mean()), 4),
                                 detect=round(float(f[m].mean()), 4),
                                 detect_preonset=round(float((m & f & (q0 <= onset)).sum()
                                                             / max(m.sum(), 1)), 4),
                                 lead_med=float(np.median(lead)) if len(lead) else None,
                                 entry_lead_med=None,
                                 fa_succ_loop=round(float(f[clsf == "succ_loop"].mean()), 4)
                                 if (clsf == "succ_loop").any() else None))
        u["D5_matched_fa"] = comp
        out["units"][corpus] = u
        pd.DataFrame(comp).to_csv(HERE / f"matched_fa_{corpus}.csv", index=False)
        print(f"[diag] {corpus} matched_full_window={int(matched.sum())} "
              f"({time.time()-t0:.0f}s)", flush=True)

    (HERE / "diagnostics.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"[done] {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
