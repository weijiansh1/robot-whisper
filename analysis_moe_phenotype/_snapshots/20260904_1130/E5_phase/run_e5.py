"""E5 内部相位 φ̂ 与 slow-normal（PROTOCOL §4 / §5-E5 / §7）。

φ̂_q：在成功集模板（LOEO）reps 上取 V2 Hellinger top-5 NN 的归一化进度中位；
d_healthy_q = 5 个 NN 距离中位；v̂_q = φ̂_q − φ̂_{q−1}（集内）。

距离唯一合法定义（V2）：rep reshape (40,32)（已是 sqrt(P)），逐 cell bc=Σ_e r1·r2，
hell=sqrt(clip(1−bc,0,1))，40 cell 均值（先 sqrt 再平均）。

族（每语料独立 maxT）：
  主族 = 5 信号 {d_healthy, v̂, stall, regress, osc} × t∈{20,25,30} = 15 格（episode 级置换）
  计时器族 = 同组同绝对 q 成败 φ̂ 差 × 3 t（组级 sign-flip，合并一族）
  残差族 = {d_healthy, v̂} × 3 t 对 mob1_w8@t 组内秩残差（6 格）
  slow-normal (a) 3 格 sign-flip；(c) slow-succ vs fail {stall, osc} × 3 t joint maxT
  探索族（不进主族）= loop onset 前 {v̂, osc, regress} × lead{−4..0} = 15 格组级 sign-flip

约定：AUC 以 y=1=失败侧（AUC>0.5 ⇔ 失败侧更高）。窗 (t−4..t] = q∈{t−3..t}（4 个 v̂）。
运行：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 python run_e5.py
"""

import csv
import json
import os
import sys
import time
import zlib

import numpy as np
from scipy.stats import spearmanr, t as tdist

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
OUT = os.path.join(BASE, "E5_phase")
sys.path.insert(0, os.path.join(BASE, "phenotype"))
from stats import groupwise_signflip_maxt, joint_maxt, paired_auc, residualise  # noqa: E402

SEED = 20260903
NPERM = 2000
TPTS = (20, 25, 30)
WIN = 4  # (t−4..t] → q ∈ {t−3..t}
K_NN = 5

LONG = "libero_long"
S8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
LR6 = ("LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_"
       "chocolate_pudding_to_the_right_of_the_plate")
CORPORA = [
    # (tag, corpus, task, template_scope)
    ("main_S8", "main16x32", S8, "group"),
    ("grid_S8", "grid50x8", S8, "pooled"),
    ("grid_LR6", "grid50x8", LR6, "pooled"),
]


def rng_for(tag, family):
    return np.random.default_rng([SEED, zlib.crc32(tag.encode()),
                                  zlib.crc32(family.encode())])


# ---------------------------------------------------------------- data loading

def load_corpus(corpus, task):
    d = os.path.join(BASE, "features", corpus, LONG, task)
    z = np.load(os.path.join(d, "rows.npz"))
    ep = z["episode_id"].astype(np.int64)
    cs = z["control_step"].astype(np.int64)
    assert np.all(np.diff(ep) >= 0), "rows 未按 episode 分块"
    uniq, first, cnt = np.unique(ep, return_index=True, return_counts=True)
    # 集内 control_step 连续升序 → qlocal
    start_of = np.zeros(uniq.max() + 1, np.int64)
    start_of[uniq] = cs[first]
    q = cs - start_of[ep]
    nq_of = np.zeros(uniq.max() + 1, np.int64)
    nq_of[uniq] = cnt
    assert np.all(q[first] == 0)
    assert np.all(q == np.concatenate([np.arange(c) for c in cnt])), "control_step 不连续"
    ep_scene = z["scene"][first].astype(np.int64)
    ep_succ = z["success"][first].astype(np.int64)
    reps = np.load(os.path.join(d, "reps.npy"), mmap_mode="r")
    assert reps.shape == (len(ep), 1280)
    return dict(ep=ep, q=q, scene=z["scene"].astype(np.int64), mob1w8=z["mob1_w8"],
                eps=uniq, first=first, nq=cnt, ep_scene=ep_scene, ep_succ=ep_succ,
                nq_of=nq_of, reps=reps)


def load_events(corpus, task):
    p = os.path.join(BASE, "events", corpus, LONG, task, "events.csv")
    ev = list(csv.DictReader(open(p)))
    assert all(e["proxy_grade"] == "full" for e in ev)
    return {int(e["episode_id"]): int(e["loop_onset_q"]) for e in ev}


# ------------------------------------------------------------ V2 NN phase

def _dist_block(E, T):
    """V2 Hellinger：E (ne,40,32) × T (nt,40,32) → (ne,nt)。逐 cell 先 sqrt 再平均。"""
    D = np.zeros((E.shape[0], T.shape[0]), np.float32)
    for c in range(40):
        bc = E[:, c, :] @ T[:, c, :].T
        np.clip(1.0 - bc, 0.0, 1.0, out=bc)
        np.sqrt(bc, out=bc)
        D += bc
    D /= 40.0
    return D


def compute_phase(C, scope):
    """逐行 φ̂ / d_healthy（模板=成功集，LOEO）。scope='group' 同组模板 / 'pooled' 跨 scene 池化。"""
    n = len(C["ep"])
    phi = np.full(n, np.nan, np.float32)
    dh = np.full(n, np.nan, np.float32)
    row_succ = C["ep_succ"][np.searchsorted(C["eps"], C["ep"])]
    prog = C["q"] / np.maximum(C["nq_of"][C["ep"]] - 1, 1)
    audit = {}
    scenes = np.unique(C["scene"]) if scope == "group" else [None]
    for sc in scenes:
        if scope == "group":
            tmpl_rows = np.flatnonzero((C["scene"] == sc) & (row_succ == 1))
            eval_eps = C["eps"][C["ep_scene"] == sc]
        else:
            tmpl_rows = np.flatnonzero(row_succ == 1)
            eval_eps = C["eps"]
        key = "pooled" if sc is None else int(sc)
        audit[key] = dict(tmpl_eps=int(len(np.unique(C["ep"][tmpl_rows]))),
                          tmpl_rows=int(len(tmpl_rows)))
        if len(tmpl_rows) == 0:
            continue
        T = np.asarray(C["reps"][tmpl_rows], np.float32).reshape(-1, 40, 32)
        t_ep, t_prog = C["ep"][tmpl_rows], prog[tmpl_rows].astype(np.float32)
        # 按 episode 分块评估（块 ≤ ~1200 行），控制 D 内存 < 2GB
        chunk, rows_ct = [], 0
        for e in eval_eps:
            chunk.append(e)
            rows_ct += C["nq_of"][e]
            if rows_ct >= 1200 or e == eval_eps[-1]:
                _phase_chunk(C, chunk, T, t_ep, t_prog, phi, dh)
                chunk, rows_ct = [], 0
    return phi, dh, audit


def _phase_chunk(C, eps_chunk, T, t_ep, t_prog, phi, dh):
    fst = {int(e): int(C["first"][i]) for i, e in enumerate(C["eps"])}
    rows = np.concatenate([np.arange(fst[int(e)], fst[int(e)] + C["nq_of"][e])
                           for e in eps_chunk])
    E = np.asarray(C["reps"][rows], np.float32).reshape(-1, 40, 32)
    D = _dist_block(E, T)
    off = 0
    for e in eps_chunk:
        ne = int(C["nq_of"][e])
        sub = D[off:off + ne]
        keep = t_ep != e  # LOEO：评估某集时整集剔出模板（失败集不在模板，keep 全 True）
        if not keep.all():
            sub, tp = sub[:, keep], t_prog[keep]
        else:
            tp = t_prog
        assert sub.shape[1] >= K_NN, "LOEO 后模板行不足 top-5"
        idx = np.argpartition(sub, K_NN - 1, axis=1)[:, :K_NN]
        dsel = np.take_along_axis(sub, idx, axis=1)
        phi[rows[off:off + ne]] = np.median(tp[idx], axis=1)
        dh[rows[off:off + ne]] = np.median(dsel, axis=1)
        off += ne


# ------------------------------------------------------- per-row derived series

def derive_series(C, phi):
    """v̂ 及窗口统计（窗=(q−4..q]，需 q≥4）。stall 阈=组内成功 |v̂| p10（LOEO φ̂ 上算）。"""
    n = len(phi)
    vhat = np.full(n, np.nan, np.float32)
    for i, e in enumerate(C["eps"]):
        s = slice(C["first"][i], C["first"][i] + C["nq"][i])
        vhat[s.start + 1:s.stop] = np.diff(phi[s])
    row_succ = C["ep_succ"][np.searchsorted(C["eps"], C["ep"])]
    thr = {}  # scene → 成功 |v̂| p10
    for sc in np.unique(C["scene"]):
        m = (C["scene"] == sc) & (row_succ == 1) & np.isfinite(vhat)
        thr[int(sc)] = float(np.percentile(np.abs(vhat[m]), 10)) if m.sum() else np.nan
    vwin = np.full(n, np.nan, np.float32)
    stall = np.full(n, np.nan, np.float32)
    regress = np.full(n, np.nan, np.float32)
    osc = np.full(n, np.nan, np.float32)
    for i, e in enumerate(C["eps"]):
        f, ne = C["first"][i], C["nq"][i]
        v = vhat[f:f + ne]
        th = thr[int(C["ep_scene"][i])]
        for qq in range(WIN, ne):
            w = v[qq - WIN + 1:qq + 1]  # q−3..q，4 个
            if not np.isfinite(w).all():
                continue
            vwin[f + qq] = w.mean()
            regress[f + qq] = (w < 0).mean()
            osc[f + qq] = float((w[:-1] * w[1:] < 0).sum())
            if np.isfinite(th):
                stall[f + qq] = (np.abs(w) < th).mean()
    return vhat, dict(vwin=vwin, stall=stall, regress=regress, osc=osc), thr


def at_t(C, arr, t):
    """每集在绝对 q=t 的值（不在场=NaN）。arr 为逐行数组。"""
    out = np.full(len(C["eps"]), np.nan, np.float32)
    alive = C["nq"] > t
    out[alive] = arr[C["first"][alive] + t]
    return out


# ---------------------------------------------------------------- stats blocks

def family_main(tag, C, dh_row, ser, res):
    """主 15 格 + 残差 6 格。y=1=失败。"""
    y = 1 - C["ep_succ"]
    grp = C["ep_scene"]
    ep_label = {int(e): int(yy) for e, yy in zip(C["eps"], y)}
    ep_group = {int(e): int(g) for e, g in zip(C["eps"], grp)}
    sig_rows = dict(d_healthy=dh_row, vhat=ser["vwin"], stall=ser["stall"],
                    regress=ser["regress"], osc=ser["osc"])
    cells, vals_t = [], {}
    for name, arr in sig_rows.items():
        for t in TPTS:
            v = at_t(C, arr, t)
            vals_t[(name, t)] = v
            cells.append((f"{name}@t{t}", v, np.isfinite(v), grp))
    obs, p = joint_maxt(cells, C["eps"], ep_label, ep_group, NPERM, rng_for(tag, "main15"))
    # 残差腿：d_healthy 与 v̂ 对 mob1_w8@t 组内秩残差
    rcells = []
    for name in ("d_healthy", "vhat"):
        for t in TPTS:
            base = at_t(C, C["mob1w8"], t)
            rv = residualise(vals_t[(name, t)], base, grp)
            rcells.append((f"{name}@t{t}", rv, np.isfinite(rv), grp))
    robs, rp = joint_maxt(rcells, C["eps"], ep_label, ep_group, NPERM, rng_for(tag, "resid6"))
    res["phase_auc"] += [
        dict(corpus=tag, signal=n.split("@t")[0], t=int(n.split("@t")[1]), leg="raw",
             auc=round(float(obs[n][0]), 4), n_pairs=obs[n][1], p_maxT=round(p[n], 4))
        for n in obs] + [
        dict(corpus=tag, signal=n.split("@t")[0], t=int(n.split("@t")[1]), leg="resid_mob1w8",
             auc=round(float(robs[n][0]), 4), n_pairs=robs[n][1],
             p_maxT=round(rp.get(n, np.nan), 4))
        for n in robs]
    return vals_t


def family_timer(tag, C, phi_row, res):
    """计时器证伪：同组同绝对 q 成败 φ̂ 组级差（fail−succ），3 t 合并 sign-flip 族。"""
    scenes = np.unique(C["ep_scene"])
    eff = np.full((len(scenes), len(TPTS)), np.nan)
    per_group = []
    for j, t in enumerate(TPTS):
        v = at_t(C, phi_row, t)
        for i, sc in enumerate(scenes):
            mf = (C["ep_scene"] == sc) & (C["ep_succ"] == 0) & np.isfinite(v)
            ms = (C["ep_scene"] == sc) & (C["ep_succ"] == 1) & np.isfinite(v)
            if mf.any() and ms.any():
                eff[i, j] = v[mf].mean() - v[ms].mean()
                per_group.append(dict(corpus=tag, t=t, group=int(sc),
                                      eff_fail_minus_succ=round(float(eff[i, j]), 4),
                                      n_fail=int(mf.sum()), n_succ=int(ms.sum())))
    obs, p = groupwise_signflip_maxt(eff, NPERM, rng_for(tag, "timer"))
    summ = []
    for j, t in enumerate(TPTS):
        col = eff[:, j]
        ok = np.isfinite(col)
        summ.append(dict(corpus=tag, t=t, group="ALL",
                         eff_fail_minus_succ=round(float(obs[j]), 4),
                         p_signflip_maxT=round(float(p[j]), 4), n_groups=int(ok.sum()),
                         n_groups_neg=int((col[ok] < 0).sum()),
                         n_groups_pos=int((col[ok] > 0).sum())))
    res["timer"] += summ
    res["timer_groups"] += per_group
    return summ


def clockness(C, phi_row):
    """成功集内 Spearman(φ̂, q)（钟表性描述）；失败集对照。"""
    rs, rf = [], []
    for i in range(len(C["eps"])):
        s = slice(C["first"][i], C["first"][i] + C["nq"][i])
        r = spearmanr(phi_row[s], np.arange(C["nq"][i]))[0]
        (rs if C["ep_succ"][i] == 1 else rf).append(r)
    q = lambda a: [round(float(x), 3) for x in np.nanpercentile(a, [25, 50, 75])]
    return dict(succ_spearman_q25_50_75=q(rs), fail_spearman_q25_50_75=q(rf))


def tertiles(C):
    """组内成功集按 n_queries 三分位；返回 ep→'slow'/'mid'/'fast'（组内 <3 成功则跳过）。"""
    lab = {}
    for sc in np.unique(C["ep_scene"]):
        m = (C["ep_scene"] == sc) & (C["ep_succ"] == 1)
        eps = C["eps"][m]
        if len(eps) < 3:
            continue
        order = eps[np.lexsort((eps, C["nq"][m]))]
        k = len(order) // 3
        for e in order[:k]:
            lab[int(e)] = "fast"
        for e in order[len(order) - k:]:
            lab[int(e)] = "slow"
        for e in order[k:len(order) - k]:
            lab[int(e)] = "mid"
    return lab


def family_slow(tag, C, vals_t, vhat_row, ser, lab, res):
    y_slow = np.array([lab.get(int(e)) == "slow" for e in C["eps"]])
    y_fast = np.array([lab.get(int(e)) == "fast" for e in C["eps"]])
    scenes = np.unique(C["ep_scene"])
    # (a) d_healthy@t 慢−快 组级差 + 跨组 t-CI + sign-flip（3 格一族）
    eff = np.full((len(scenes), len(TPTS)), np.nan)
    for j, t in enumerate(TPTS):
        v = vals_t[("d_healthy", t)]
        for i, sc in enumerate(scenes):
            ms = (C["ep_scene"] == sc) & y_slow & np.isfinite(v)
            mf = (C["ep_scene"] == sc) & y_fast & np.isfinite(v)
            if ms.any() and mf.any():
                eff[i, j] = v[ms].mean() - v[mf].mean()
    obs, p = groupwise_signflip_maxt(eff, NPERM, rng_for(tag, "slow_a"))
    for j, t in enumerate(TPTS):
        col = eff[:, j][np.isfinite(eff[:, j])]
        if len(col) > 1:
            half = tdist.ppf(0.975, len(col) - 1) * col.std(ddof=1) / np.sqrt(len(col))
            ci = [round(float(col.mean() - half), 4), round(float(col.mean() + half), 4)]
        else:
            ci = [np.nan, np.nan]
        res["slow"].append(dict(corpus=tag, test="a_dhealthy_slow_minus_fast", t=t,
                                value=round(float(obs[j]), 4), ci_lo=ci[0], ci_hi=ci[1],
                                p=round(float(p[j]), 4), n_groups=int(len(col))))
    # (b) 慢成功 v̂>0 比例（全程 q≥1）；组均值 − 0.5 sign-flip（1 格）
    prop = {}
    for kind, ymask in (("slow", y_slow), ("fast", y_fast),
                        ("fail", C["ep_succ"] == 0)):
        pr = np.full(len(C["eps"]), np.nan)
        for i in np.flatnonzero(ymask):
            s = slice(C["first"][i], C["first"][i] + C["nq"][i])
            v = vhat_row[s][1:]
            v = v[np.isfinite(v)]
            if len(v):
                pr[i] = (v > 0).mean()
        prop[kind] = pr
    geff = np.full((len(scenes), 1), np.nan)
    for i, sc in enumerate(scenes):
        m = (C["ep_scene"] == sc) & np.isfinite(prop["slow"])
        if m.any():
            geff[i, 0] = prop["slow"][m].mean() - 0.5
    obs_b, p_b = groupwise_signflip_maxt(geff, NPERM, rng_for(tag, "slow_b"))
    for kind in ("slow", "fast", "fail"):
        pr = prop[kind][np.isfinite(prop[kind])]
        res["slow"].append(dict(
            corpus=tag, test=f"b_vhat_pos_prop_{kind}", t=-1,
            value=round(float(pr.mean()), 4) if len(pr) else np.nan,
            ci_lo=round(float(np.percentile(pr, 25)), 4) if len(pr) else np.nan,
            ci_hi=round(float(np.percentile(pr, 75)), 4) if len(pr) else np.nan,
            p=round(float(p_b[0]), 4) if kind == "slow" else np.nan,
            n_groups=int(np.isfinite(geff[:, 0]).sum() if kind == "slow" else len(pr))))
    # (c) 慢成功 vs 失败 stall/osc AUC（6 格 joint maxT；y=1=失败）
    keep = y_slow | (C["ep_succ"] == 0)
    eps_k, grp_k = C["eps"][keep], C["ep_scene"][keep]
    ep_label = {int(e): int(C["ep_succ"][i] == 0)
                for i, e in enumerate(C["eps"]) if keep[i]}
    ep_group = {int(e): int(C["ep_scene"][i])
                for i, e in enumerate(C["eps"]) if keep[i]}
    cells = []
    for name in ("stall", "osc"):
        for t in TPTS:
            v = at_t(C, ser[name], t)[keep]
            cells.append((f"{name}@t{t}", v, np.isfinite(v), grp_k))
    obs_c, p_c = joint_maxt(cells, eps_k, ep_label, ep_group, NPERM,
                            rng_for(tag, "slow_c"))
    for n in obs_c:
        res["slow"].append(dict(corpus=tag, test=f"c_slowsucc_vs_fail_{n.split('@t')[0]}",
                                t=int(n.split("@t")[1]), value=round(float(obs_c[n][0]), 4),
                                ci_lo=np.nan, ci_hi=np.nan,
                                p=round(float(p_c.get(n, np.nan)), 4), n_groups=obs_c[n][1]))


def family_loop(tag, C, onset_of, vhat_row, ser, res):
    """探索性：loop onset 前 lead∈{−4..0} 的 {v̂, osc, regress} 相对同组同 q no-loop 对照。"""
    leads = list(range(-4, 1))
    sigs = dict(vhat=vhat_row, osc=ser["osc"], regress=ser["regress"])
    scenes = np.unique(C["ep_scene"])
    idx_of = {int(e): i for i, e in enumerate(C["eps"])}
    loop_eps = [e for e, o in onset_of.items() if o >= 0]
    ctrl_eps = [e for e, o in onset_of.items() if o < 0]
    eff = np.full((len(scenes), len(sigs) * len(leads)), np.nan)
    n_ev_used = 0
    for gi, sc in enumerate(scenes):
        ev_g = [e for e in loop_eps if C["ep_scene"][idx_of[e]] == sc]
        ct_g = [e for e in ctrl_eps if C["ep_scene"][idx_of[e]] == sc]
        if not ev_g or not ct_g:
            continue
        acc = {c: [] for c in range(eff.shape[1])}
        for e in ev_g:
            used = False
            for li, L in enumerate(leads):
                qq = onset_of[e] + L
                i = idx_of[e]
                if qq < WIN or qq >= C["nq"][i]:
                    continue
                ctr = [c for c in ct_g if C["nq"][idx_of[c]] > qq]
                if not ctr:
                    continue
                for si, (name, arr) in enumerate(sigs.items()):
                    xv = arr[C["first"][i] + qq]
                    cv = np.array([arr[C["first"][idx_of[c]] + qq] for c in ctr])
                    cv = cv[np.isfinite(cv)]
                    if np.isfinite(xv) and len(cv):
                        acc[si * len(leads) + li].append(xv - cv.mean())
                        used = True
            n_ev_used += used
        for c, v in acc.items():
            if v:
                eff[gi, c] = float(np.mean(v))
    obs, p = groupwise_signflip_maxt(eff, NPERM, rng_for(tag, "loop"))
    for si, name in enumerate(sigs):
        for li, L in enumerate(leads):
            c = si * len(leads) + li
            res["loop"].append(dict(corpus=tag, signal=name, lead=L,
                                    eff=round(float(obs[c]), 4), p_maxT=round(float(p[c]), 4),
                                    n_groups=int(np.isfinite(eff[:, c]).sum())))
    res["loop_meta"][tag] = dict(n_loop_eps=len(loop_eps), n_ctrl_eps=len(ctrl_eps),
                                 n_loop_eps_used=int(n_ev_used))


# ------------------------------------------------------------------------ main

def survival(tag, C):
    rows = []
    for t in TPTS:
        alive = C["nq"] > t
        both = 0
        for sc in np.unique(C["ep_scene"]):
            m = (C["ep_scene"] == sc) & alive
            if (C["ep_succ"][m] == 1).any() and (C["ep_succ"][m] == 0).any():
                both += 1
        rows.append(dict(corpus=tag, t=t, alive_succ=int((alive & (C["ep_succ"] == 1)).sum()),
                         alive_fail=int((alive & (C["ep_succ"] == 0)).sum()),
                         groups_with_both=both))
    return rows


def main():
    t0 = time.time()
    res = dict(phase_auc=[], timer=[], timer_groups=[], slow=[], loop=[], loop_meta={},
               survival=[], template_audit={}, clockness={}, tertile_counts={})
    for tag, corpus, task, scope in CORPORA:
        print(f"[{time.time()-t0:6.1f}s] {tag} load", flush=True)
        C = load_corpus(corpus, task)
        phi, dh, audit = compute_phase(C, scope)
        np.savez_compressed(os.path.join(OUT, f"phase_{tag}.npz"),
                            phi=phi, dh=dh, ep=C["ep"], q=C["q"])
        tmpl_sizes = [a["tmpl_eps"] for a in audit.values()]
        res["template_audit"][tag] = dict(
            scope=scope, groups=len(audit),
            tmpl_eps_min=int(min(tmpl_sizes)), tmpl_eps_med=float(np.median(tmpl_sizes)),
            tmpl_eps_max=int(max(tmpl_sizes)),
            groups_no_template=[k for k, a in audit.items() if a["tmpl_eps"] == 0],
            phi_coverage=float(np.isfinite(phi).mean()))
        print(f"[{time.time()-t0:6.1f}s] {tag} phase done, cover={np.isfinite(phi).mean():.3f}",
              flush=True)
        vhat, ser, thr = derive_series(C, phi)
        res["survival"] += survival(tag, C)
        res["clockness"][tag] = clockness(C, phi)
        vals_t = family_main(tag, C, dh, ser, res)
        family_timer(tag, C, phi, res)
        lab = tertiles(C)
        res["tertile_counts"][tag] = {k: sum(1 for v in lab.values() if v == k)
                                      for k in ("fast", "mid", "slow")}
        family_slow(tag, C, vals_t, vhat, ser, lab, res)
        if task == S8:
            onset_of = load_events(corpus, task)
            assert set(onset_of) == set(int(e) for e in C["eps"])
            family_loop(tag, C, onset_of, vhat, ser, res)
        # 慢成功/快成功标签存给画图
        np.savez(os.path.join(OUT, f"eplab_{tag}.npz"),
                 eps=C["eps"], succ=C["ep_succ"], nq=C["nq"], scene=C["ep_scene"],
                 tert=np.array([lab.get(int(e), "") for e in C["eps"]]))
        print(f"[{time.time()-t0:6.1f}s] {tag} stats done", flush=True)

    import pandas as pd
    pd.DataFrame(res["phase_auc"]).to_csv(os.path.join(OUT, "phase_auc.csv"), index=False)
    pd.DataFrame(res["slow"]).to_csv(os.path.join(OUT, "slow_normal.csv"), index=False)
    pd.DataFrame(res["timer"] + res["timer_groups"]).to_csv(
        os.path.join(OUT, "timer_test.csv"), index=False)
    summary = dict(protocol="E5 v1", seed=SEED, nperm=NPERM, t_points=list(TPTS),
                   window="(t-4..t] = 4 vhat steps", knn=K_NN,
                   auc_orientation="y=1=failure (AUC>0.5 ⇔ 失败侧更高)",
                   corpora={t: dict(corpus=c, task=k, template_scope=s)
                            for t, c, k, s in CORPORA},
                   survival=res["survival"], template_audit=res["template_audit"],
                   clockness=res["clockness"], tertile_counts=res["tertile_counts"],
                   timer=res["timer"], phase_auc=res["phase_auc"], slow_normal=res["slow"],
                   loop_exploratory=res["loop"], loop_meta=res["loop_meta"],
                   runtime_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    print(f"[{time.time()-t0:6.1f}s] all done")


if __name__ == "__main__":
    main()
