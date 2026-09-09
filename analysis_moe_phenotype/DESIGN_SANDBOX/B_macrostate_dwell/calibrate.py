#!/usr/bin/env python3
"""方案 B 校准 / 评测（SCENE8 两单位，留出集从未触碰）。

口径见 SPEC.md。本脚本：
  1. 建带（switching / flatten / sticky[臂C] / 五轴同态[诊断]）；
  2. 旋钮 K/M/K_C 按**只用成功集**的 FA 目标规则、留一组外（LOGO，组=corpus|scene）选取并冻结；
  3. 必报指标（FA / 检出 / 提前量 / 恢复性 vs 致败 / 两语料分报）；
  4. **载荷主张 = 同一次运行内 B vs C 的配对差**（等工作点对齐 + McNemar 不一致对）；
  5. 平凡臂 T（集长阈值）——必须被打败的对象；
  6. 机械自检探针 P1（组内 success 置换后报警逐位不变）与 P2（截断 s 仍报 / 截断 s−1 全不报）。
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from scipy import stats as sps

HERE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "phenotype"))

from detector import (PARAMS, PARAM_SWEEP, build_reference, detect,  # noqa: E402
                      episode_bands, load_unit)
from stats import joint_maxt, paired_auc  # noqa: E402

SEED = 20260904
NPERM = 2000
TASK = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = [("main16x32", "libero_long", TASK), ("grid50x8", "libero_long", TASK)]
KMAX = 60
FA_TARGET = PARAMS["FA_TARGET"]
BANDS = ("switch", "flatten", "sticky", "state5")


# ------------------------------------------------------------------ 装载 + 建带
def build_unit(corpus, suite, task, qmax=None):
    p = ROOT / "features" / corpus / suite / task / "rows.npz"
    df, raw = load_unit(p, "scene")
    pars = dict(PARAMS)
    pars["QMAX"] = qmax
    ref = build_reference(df, raw, pars)
    bands = episode_bands(df, raw, ref, pars)
    ev = pd.read_csv(ROOT / "events" / corpus / suite / task / "events.csv")
    assert (ev["proxy_grade"] == "full").all(), "SCENE8 必须是 full 代理层"
    rec = [dict(episode_id=e, group=b["group"], n_q=b["n_q"], n_win=b["n_win"],
                ref_level=b["level"],
                **{f"maxrun_{k}": b[f"maxrun_{k}"] for k in BANDS},
                **{f"occ_{k}": b[f"occ_{k}"] for k in BANDS}) for e, b in bands.items()]
    T = pd.DataFrame(rec).merge(
        ev[["episode_id", "success", "n_queries", "loop_onset_q", "static_onset_q"]],
        on="episode_id", how="inner", validate="1:1")
    assert len(T) == len(bands), "events.csv 与 rows.npz 集数对不上"
    T["corpus"] = corpus
    T["gkey"] = corpus + "|" + T["group"].astype(str)
    hl, hs = T.loop_onset_q >= 0, T.static_onset_q >= 0
    T["cls"] = np.where(T.success == 0,
                        np.where(hl & hs, "fail_both",
                                 np.where(hl, "fail_loop",
                                          np.where(hs, "fail_static", "fail_noevent"))),
                        np.where(hl, "succ_loop", np.where(hs, "succ_static", "succ_clean")))
    cross = {e: {k: bands[e][f"cross_{k}"] for k in BANDS} for e in bands}
    return T, cross


def alarm_matrix(cross, eps, band, ks):
    """(n_eps, n_k) 首次报警 q；-1 = 不报警。"""
    out = np.full((len(eps), len(ks)), -1, np.int64)
    for i, e in enumerate(eps):
        c = cross[e][band]
        for j, K in enumerate(ks):
            if len(c) >= K:
                out[i, j] = c[K - 1]
    return out


# ------------------------------------------------------------------ 旋钮选取
def pick_knob(fa_by_k, ks, target):
    ok = np.flatnonzero(fa_by_k <= target)
    return (int(ks[ok[0]]), True) if len(ok) else (int(ks[-1]), False)


def logo_knob(T, A, ks, target):
    clean = (T.cls == "succ_clean").to_numpy()
    g = T.gkey.to_numpy()
    fired = A >= 0
    per = {gg: pick_knob(fired[clean & (g != gg)].mean(axis=0), ks, target)[0]
           for gg in np.unique(g)}
    fa_all = fired[clean].mean(axis=0)
    K, hit = pick_knob(fa_all, ks, target)
    return per, K, hit, fa_all


# ------------------------------------------------------------------ 指标
def quart(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0, q25=None, med=None, q75=None)
    return dict(n=int(len(x)), q25=float(np.percentile(x, 25)),
                med=float(np.percentile(x, 50)), q75=float(np.percentile(x, 75)))


def eval_channel(T, alarm_q, chan_cls, onset_col, label):
    cls = T.cls.to_numpy()
    fired = alarm_q >= 0
    clean = cls == "succ_clean"
    n_win = T.n_win.to_numpy()
    out = dict(channel=label,
               fa_clean_succ=float(fired[clean].mean()), n_clean_succ=int(clean.sum()),
               fa_per100q_clean=float(100.0 * fired[clean].sum() / max(n_win[clean].sum(), 1)),
               fa_fail_noevent=float(fired[cls == "fail_noevent"].mean())
               if (cls == "fail_noevent").any() else None,
               n_fail_noevent=int((cls == "fail_noevent").sum()))
    m = np.isin(cls, chan_cls)
    onset = T[onset_col].to_numpy()
    out["n_target"] = int(m.sum())
    out["detect"] = float(fired[m].mean()) if m.any() else None
    out["detect_preonset"] = float((m & fired & (alarm_q <= onset)).sum() / m.sum()) \
        if m.any() else None
    out["lead"] = quart((alarm_q - onset)[m & fired])
    for rc in ("succ_loop", "succ_static"):
        mm = cls == rc
        out[f"fa_{rc}"] = float(fired[mm].mean()) if mm.any() else None
        out[f"n_{rc}"] = int(mm.sum())
    return out


# ------------------------------------------------------------------ 判别力检验
def discrim_family(T, tag, nperm=NPERM):
    cls = T.cls.to_numpy()
    grp = T.gkey.to_numpy()
    ek = np.arange(len(T))
    lab = {int(i): (1 if cls[i].startswith("fail") else 0) for i in ek}
    gof = {int(i): grp[i] for i in ek}
    fl = np.isin(cls, ["fail_loop", "fail_both"])
    fs = np.isin(cls, ["fail_static", "fail_both"])
    cl = cls == "succ_clean"
    rl = cls == "succ_loop"
    cells = [("switch|fatalloop_vs_clean", T[f"maxrun_switch{tag}"].to_numpy().astype(float),
              fl | cl, grp),
             ("flatten|fatalstatic_vs_clean",
              T[f"maxrun_flatten{tag}"].to_numpy().astype(float), fs | cl, grp),
             ("sticky_C|fatalloop_vs_clean", T[f"maxrun_sticky{tag}"].to_numpy().astype(float),
              fl | cl, grp),
             ("sticky_C|fatalstatic_vs_clean",
              T[f"maxrun_sticky{tag}"].to_numpy().astype(float), fs | cl, grp),
             ("state5|fatalany_vs_clean", T[f"maxrun_state5{tag}"].to_numpy().astype(float),
              fl | fs | cl, grp),
             ("switch|fatalloop_vs_recoveryloop",
              T[f"maxrun_switch{tag}"].to_numpy().astype(float), fl | rl, grp)]
    obs, p = joint_maxt(cells, ek, lab, gof, nperm, np.random.default_rng(SEED))
    out = []
    for c in cells:
        n = c[0]
        _, p1 = joint_maxt([c], ek, lab, gof, nperm, np.random.default_rng(SEED))
        a, npair = obs[n]
        out.append(dict(cell=n, auc=None if not np.isfinite(a) else round(float(a), 4),
                        n_pairs=int(npair), p_maxT=p.get(n), p_uncorrected=p1.get(n)))
    return out


# ------------------------------------------------------------------ B vs C 配对差
def paired_bc(T, A_b, A_c, ks, chan_cls, onset_col, fa_levels):
    """把 B 通道与臂 C 对齐到同一 FA 工作点，报配对差 + McNemar 不一致对。"""
    cls = T.cls.to_numpy()
    clean = cls == "succ_clean"
    m = np.isin(cls, chan_cls)
    onset = T[onset_col].to_numpy()
    fa_b = (A_b >= 0)[clean].mean(0)
    fa_c = (A_c >= 0)[clean].mean(0)
    rows = []
    for lvl in fa_levels:
        jb = np.flatnonzero(fa_b <= lvl)
        jc = np.flatnonzero(fa_c <= lvl)
        if not len(jb) or not len(jc):
            continue
        jb, jc = jb[0], jc[0]
        fb, fc = A_b[:, jb] >= 0, A_c[:, jc] >= 0
        pb = m & fb & (A_b[:, jb] <= onset)
        pc = m & fc & (A_c[:, jc] <= onset)
        n10, n01 = int((pb & ~pc).sum()), int((pc & ~pb).sum())
        pmc = float(sps.binomtest(n10, n10 + n01, 0.5).pvalue) if n10 + n01 else None
        rows.append(dict(fa_level=lvl, K_B=int(ks[jb]), K_C=int(ks[jc]),
                         fa_B=round(float(fa_b[jb]), 4), fa_C=round(float(fa_c[jc]), 4),
                         d_fa=round(float(fa_b[jb] - fa_c[jc]), 4),
                         n_target=int(m.sum()),
                         det_B=round(float(fb[m].mean()), 4),
                         det_C=round(float(fc[m].mean()), 4),
                         d_det=round(float(fb[m].mean() - fc[m].mean()), 4),
                         pre_B=round(float(pb.sum() / m.sum()), 4),
                         pre_C=round(float(pc.sum() / m.sum()), 4),
                         d_pre=round(float((pb.sum() - pc.sum()) / m.sum()), 4),
                         mcnemar_B_only=n10, mcnemar_C_only=n01, mcnemar_p=pmc,
                         lead_B=float(np.median((A_b[:, jb] - onset)[m & fb]))
                         if (m & fb).any() else None,
                         lead_C=float(np.median((A_c[:, jc] - onset)[m & fc]))
                         if (m & fc).any() else None))
    return rows


# ------------------------------------------------------------------ 机械自检探针
def probe_p1(corpus, suite, task, rng):
    """P1：组内 success 随机置换 -> 报警必须逐位不变。"""
    src = ROOT / "features" / corpus / suite / task / "rows.npz"
    d = dict(np.load(src))
    base = detect(src, "scene")
    ep, scene, succ = d["episode_id"], d["scene"], d["success"].copy()
    eps = np.unique(ep)
    ep_scene = {int(e): int(scene[ep == e][0]) for e in eps}
    ep_succ = {int(e): int(succ[ep == e][0]) for e in eps}
    for _ in range(3):
        newlab = {}
        for s in np.unique(list(ep_scene.values())):
            ge = [e for e in eps if ep_scene[int(e)] == s]
            vals = rng.permutation([ep_succ[int(e)] for e in ge])
            newlab.update({int(e): int(v) for e, v in zip(ge, vals)})
        d["success"] = np.array([newlab[int(e)] for e in ep], dtype=succ.dtype)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=True) as f:
            np.savez(f.name, **d)
            got = detect(f.name, "scene")
        if json.dumps(got, sort_keys=True) != json.dumps(base, sort_keys=True):
            return False
    return True


def probe_p2(corpus, suite, task, rng, n_check=15):
    """P2：截断到首个报警步 s 仍报 s；截断到 s−1 完全不报。"""
    src = ROOT / "features" / corpus / suite / task / "rows.npz"
    d = dict(np.load(src))
    base = detect(src, "scene")
    ep = d["episode_id"]
    cs = d["control_step"]
    starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
    q = cs - np.repeat(cs[starts], np.diff(np.r_[starts, len(ep)]))
    fire = [e for e, v in base.items() if v["alarms"]]
    if not fire:
        return dict(ok_at_s=None, ok_before_s=None, n_checked=0)
    sel = rng.choice(fire, size=min(n_check, len(fire)), replace=False)
    ok_s = ok_b = True
    n = 0
    for e in sel:
        e = int(e)
        s = base[e]["alarms"][0]["step"]
        for cut, want in ((s, True), (s - 1, False)):
            keep = ~((ep == e) & (q > cut))
            dd = {k: (v[keep] if getattr(v, "shape", (0,))[:1] == (len(ep),) else v)
                  for k, v in d.items()}
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=True) as f:
                np.savez(f.name, **dd)
                got = detect(f.name, "scene")
            al = got.get(e, {}).get("alarms", [])
            if want:
                ok_s &= bool(al) and al[0]["step"] == s
                n += 1
            else:
                ok_b &= (len(al) == 0)
    return dict(ok_at_s=bool(ok_s), ok_before_s=bool(ok_b), n_checked=int(n))


# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    ks = np.arange(1, KMAX + 1)

    qcaps = []
    for corpus, suite, task in UNITS:
        ev = pd.read_csv(ROOT / "events" / corpus / suite / task / "events.csv")
        qcaps.append(int(np.median(ev.loc[ev.success == 1, "n_queries"])) - 1)
    QCAP = int(min(qcaps))
    print(f"[qcap] 由成功集 n_queries 中位决定：{qcaps} -> QCAP={QCAP}", flush=True)

    units, tabs = {}, []
    for corpus, suite, task in UNITS:
        T, cross = build_unit(corpus, suite, task)
        Tc, _ = build_unit(corpus, suite, task, qmax=QCAP)
        for k in BANDS:
            T[f"maxrun_{k}_cap"] = Tc[f"maxrun_{k}"].to_numpy()
        T["n_win_cap"] = Tc["n_win"].to_numpy()
        units[corpus] = dict(T=T, cross=cross, suite=suite, task=task)
        tabs.append(T)
        print(f"[load] {corpus} eps={len(T)} ref_level={T.ref_level.value_counts().to_dict()} "
              f"cls={T.cls.value_counts().to_dict()} ({time.time()-t0:.0f}s)", flush=True)
    ALL = pd.concat(tabs, ignore_index=True)

    eps_of = {c: units[c]["T"].episode_id.tolist() for c in units}
    Amat = {b: np.vstack([alarm_matrix(units[c]["cross"], eps_of[c], b, ks) for c in units])
            for b in ("switch", "flatten", "sticky")}
    knobs, fa_curves = {}, {}
    for band, name in (("switch", "K_LOOP"), ("flatten", "M_STATIC"), ("sticky", "K_C")):
        per, K, hit, fa = logo_knob(ALL, Amat[band], ks, FA_TARGET)
        knobs[name] = dict(band=band, pooled=K, hit_target=hit,
                           logo_min=int(min(per.values())), logo_max=int(max(per.values())),
                           logo_median=float(np.median(list(per.values()))), per_group=per)
        fa_curves[band] = fa.tolist()
        print(f"[knob] {name}: pooled={K} (hit={hit}) LOGO[{min(per.values())},"
              f"{max(per.values())}] FA@K={fa[K-1]:.4f}", flush=True)

    clean_all = (ALL.cls == "succ_clean").to_numpy()
    nq_all = ALL.n_queries.to_numpy()
    q0_grid = np.arange(PARAMS["QMIN"], 60)
    fa_T = np.array([(nq_all[clean_all] > q0).mean() for q0 in q0_grid])
    Q0, hitT = pick_knob(fa_T, q0_grid, FA_TARGET)
    print(f"[knob] ARM_T Q0={Q0} FA={fa_T[list(q0_grid).index(Q0)]:.4f}", flush=True)

    fa_levels = (0.02, 0.05, 0.10, 0.20)
    results, per_ep_out = {}, []
    for corpus in units:
        T = units[corpus]["T"]
        cr = units[corpus]["cross"]
        eps = T.episode_id.tolist()
        gk = T.gkey.to_numpy()

        def chan_alarm(band, K, logo_name=None):
            a = np.full(len(T), -1, np.int64)
            for i, e in enumerate(eps):
                c = cr[e][band]
                KK = K if logo_name is None else knobs[logo_name]["per_group"][gk[i]]
                if len(c) >= KK:
                    a[i] = c[KK - 1]
            return a

        al_frozen = {"B_loop": chan_alarm("switch", knobs["K_LOOP"]["pooled"]),
                     "B_static": chan_alarm("flatten", knobs["M_STATIC"]["pooled"]),
                     "C_sticky": chan_alarm("sticky", knobs["K_C"]["pooled"])}
        al_logo = {"B_loop": chan_alarm("switch", None, "K_LOOP"),
                   "B_static": chan_alarm("flatten", None, "M_STATIC"),
                   "C_sticky": chan_alarm("sticky", None, "K_C")}
        al_T = np.where(T.n_queries.to_numpy() > Q0, Q0, -1).astype(np.int64)

        ch = []
        for tagm, alset in (("frozen", al_frozen), ("logo", al_logo)):
            ch.append(dict(mode=tagm, **eval_channel(T, alset["B_loop"],
                                                     ["fail_loop", "fail_both"],
                                                     "loop_onset_q", "B_loop")))
            ch.append(dict(mode=tagm, **eval_channel(T, alset["B_static"],
                                                     ["fail_static", "fail_both"],
                                                     "static_onset_q", "B_static")))
            ch.append(dict(mode=tagm, **eval_channel(T, alset["C_sticky"],
                                                     ["fail_loop", "fail_both"],
                                                     "loop_onset_q", "C_sticky@loop")))
            ch.append(dict(mode=tagm, **eval_channel(T, alset["C_sticky"],
                                                     ["fail_static", "fail_both"],
                                                     "static_onset_q", "C_sticky@static")))
        ch.append(dict(mode="frozen", **eval_channel(T, al_T, ["fail_loop", "fail_both"],
                                                     "loop_onset_q", "T_len@loop")))
        ch.append(dict(mode="frozen", **eval_channel(T, al_T, ["fail_static", "fail_both"],
                                                     "static_onset_q", "T_len@static")))

        # 载荷主张：同一次运行内的 B vs C 配对差（等 FA 工作点）
        i0 = eps_of and 0
        off = 0 if corpus == list(units)[0] else len(units[list(units)[0]]["T"])
        A_sw = alarm_matrix(cr, eps, "switch", ks)
        A_fl = alarm_matrix(cr, eps, "flatten", ks)
        A_st = alarm_matrix(cr, eps, "sticky", ks)
        paired = {
            "static": paired_bc(T, A_fl, A_st, ks, ["fail_static", "fail_both"],
                                "static_onset_q", fa_levels),
            "loop": paired_bc(T, A_sw, A_st, ks, ["fail_loop", "fail_both"],
                              "loop_onset_q", fa_levels),
        }
        del i0, off

        typ = np.where(al_frozen["B_loop"] < 0,
                       np.where(al_frozen["B_static"] < 0, "", "static"),
                       np.where(al_frozen["B_static"] < 0, "loop",
                                np.where(al_frozen["B_loop"] <= al_frozen["B_static"],
                                         "loop", "static")))
        cls = T.cls.to_numpy()
        conf = pd.crosstab(pd.Series(cls, name="true_cls"), pd.Series(typ, name="alarm_type"))

        disc = {"capped": discrim_family(T, "_cap"), "uncapped": discrim_family(T, "")}
        aucs_conf = {}
        mconf = np.isin(cls, ["fail_loop", "fail_static", "fail_both"]) | (cls == "succ_clean")
        yconf = np.isin(cls, ["fail_loop", "fail_static", "fail_both"]).astype(int)
        for nm, col in (("n_queries", T.n_queries.to_numpy().astype(float)),
                        ("n_win", T.n_win.to_numpy().astype(float)),
                        ("n_win_cap", T.n_win_cap.to_numpy().astype(float))):
            a, n = paired_auc(col[mconf], yconf[mconf], T.gkey.to_numpy()[mconf])
            aucs_conf[nm] = dict(auc=float(a) if np.isfinite(a) else None, n_pairs=int(n))

        probes = dict(P1_success_permutation_invariant=probe_p1(corpus, units[corpus]["suite"],
                                                                units[corpus]["task"], rng),
                      P2=probe_p2(corpus, units[corpus]["suite"], units[corpus]["task"], rng))
        results[corpus] = dict(n_eps=int(len(T)), cls_counts=T.cls.value_counts().to_dict(),
                               ref_level=T.ref_level.value_counts().to_dict(),
                               channels=ch, paired_BvsC=paired, type_confusion=conf.to_dict(),
                               discrim=disc, confound_auc=aucs_conf, probes=probes)
        print(f"[eval] {corpus} probes={probes} ({time.time()-t0:.0f}s)", flush=True)
        pd.DataFrame(paired["static"] + paired["loop"]).to_csv(
            HERE / f"paired_BvsC_{corpus}.csv", index=False)

        po = T[["episode_id", "group", "gkey", "success", "cls", "n_q", "n_queries", "n_win",
                "ref_level", "loop_onset_q", "static_onset_q",
                "maxrun_switch", "maxrun_flatten", "maxrun_sticky", "maxrun_state5",
                "maxrun_switch_cap", "maxrun_flatten_cap", "maxrun_sticky_cap",
                "occ_switch", "occ_flatten", "occ_sticky"]].copy()
        po["corpus"] = corpus
        for k, v in al_frozen.items():
            po[f"alarm_{k}"] = v
        po["alarm_T_len"] = al_T
        po["alarm_type"] = typ
        per_ep_out.append(po)

    pd.concat(per_ep_out, ignore_index=True).to_csv(HERE / "per_episode.csv", index=False)

    sweep = []
    for corpus in units:
        T = units[corpus]["T"]
        cr = units[corpus]["cross"]
        eps = T.episode_id.tolist()
        cls = T.cls.to_numpy()
        for band, chan_cls, onset_col in (("switch", ["fail_loop", "fail_both"], "loop_onset_q"),
                                          ("flatten", ["fail_static", "fail_both"],
                                           "static_onset_q"),
                                          ("sticky", ["fail_static", "fail_both"],
                                           "static_onset_q"),
                                          ("sticky", ["fail_loop", "fail_both"],
                                           "loop_onset_q")):
            A = alarm_matrix(cr, eps, band, ks)
            onset = T[onset_col].to_numpy()
            m = np.isin(cls, chan_cls)
            clean = cls == "succ_clean"
            for j, K in enumerate(ks):
                f = A[:, j] >= 0
                lead = (A[:, j] - onset)[m & f]
                sweep.append(dict(corpus=corpus, band=band, target=onset_col, K=int(K),
                                  fa_clean=float(f[clean].mean()),
                                  detect=float(f[m].mean()) if m.any() else np.nan,
                                  detect_preonset=float((m & f & (A[:, j] <= onset)).sum()
                                                        / max(m.sum(), 1)),
                                  lead_med=float(np.median(lead)) if len(lead) else np.nan,
                                  fa_succ_loop=float(f[cls == "succ_loop"].mean())
                                  if (cls == "succ_loop").any() else np.nan))
    pd.DataFrame(sweep).to_csv(HERE / "knob_sweep.csv", index=False)

    summary = dict(
        seed=SEED, nperm=NPERM, fa_target=FA_TARGET, qcap=QCAP, kmax=KMAX,
        fa_levels=list(fa_levels),
        params_frozen=dict(QMIN=PARAMS["QMIN"], MIN_REF_EPS=PARAMS["MIN_REF_EPS"],
                           MIN_REF_ROWS=PARAMS["MIN_REF_ROWS"],
                           K_LOOP=knobs["K_LOOP"]["pooled"],
                           M_STATIC=knobs["M_STATIC"]["pooled"],
                           K_C=knobs["K_C"]["pooled"], ARM_T_Q0=int(Q0)),
        param_sweep_keys={k: len(v) for k, v in PARAM_SWEEP.items()},
        knobs={k: {kk: vv for kk, vv in v.items() if kk != "per_group"}
               for k, v in knobs.items()},
        fa_curves_pooled=fa_curves,
        arm_T=dict(Q0=int(Q0), fa_clean_pooled=float(fa_T[list(q0_grid).index(Q0)]),
                   hit_target=bool(hitT)),
        units=results, seconds=round(time.time() - t0, 1))
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
