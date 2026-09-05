#!/usr/bin/env python3
"""E8 相图+宏状态 & E9 MoE-silent 边界（PROTOCOL §5-E8/E9、§4、§7；AUDIT §7/§8）。

冻结口径（先于查看结果声明）：
- E8 分析窗 = q>=8（stickiness 轴用 mob1_w8，需 8 步拖尾历史；与既有 signal-matrix
  宏状态的 t∈[8,30] 窗同源）。集内 q = control_step − 该集首行 control_step。
- 五轴 Γ_q：instability=z(V+A)、commitment=z(top12_margin)、consensus=z(C−D_tok)、
  stickiness=z(−mob1_w8)、flow_shape=z(flow_com)；z=组内无标签 (x−组中位)/组MAD（原始
  MAD，不乘 1.4826）；组 = (corpus,suite,task,scene)。MAD=0 → 该组该轴弃用（组内该轴
  记 NaN → 该组行不进宏状态分析），计数上报。窗内行 <8 的组整组退出。
- 宏状态位序 bit0..4 = [inst, comm, cons, stick, flow]，z>0 → bit=1；32 态。
  稀疏判据（冻结）：任一语料池化占用中 <1% 行的态数 >8（>1/4）→ 主版改 16 态（去
  flow_shape）。两版占用一律落盘。
- 转移 = 集内相邻 (q,q+1) 且两端 code 有效；按 (corpus,suite,outcome) 池化。
- 检验族（预声明，唯一一族，21 格）：每 corpus×suite（7 格）取池化 Δocc=occ_fail−occ_succ
  最大的前 3 个"可用态"（两侧池化访问数 ≥50）；每格效应 = 组内 P_fail(s→s)−P_succ(s→s)
  （每侧 from-s 转移 ≥3，否则该组该格 NaN）；置换单位 = 组（整组翻号，同组跨格同号），
  groupwise_signflip_maxt 2000 次，maxT 覆盖 21 格。16 态版同法另跑，声明为次级族。
  其余全部描述性。
- 事件集：loop 用 full+objaware 有效层（剔除 grid long KITCHEN_SCENE3 与两语料 goal
  middle_drawer 的 loop/trap 通道，AUDIT §7.1/§8.4）；static 全层有效。轨迹束主体 =
  失败集事件（成功集事件另报）。带定义：switching/desync 带 = inst=1 & cons=0；
  flatten/shared 带 = inst=0 & cons=1。
- E9：9 信号 = 8 核心 {V,A,S_ent,S_margin,C,D_tok,D_layer,flow_com} + mob1_w8。
  固定时点：long 套件 {20,25,30}，其余 {8,12,16}。百分位 = 相对同组成功集在同一绝对 q
  的分布（每成功集一行），midrank，参考样本 ≥5 否则该格 NA。集级偏离 dev_fixed =
  max 信号 max 时点 |pct−50|。事件 onset（E9 用）= trap_onset_q（loop 通道无效任务改用
  static_onset_q）；dev_event = 9 信号 × lead∈{−4..0} 的 max |pct−50|。
  silent ⇔ dev_fixed<40 且（无 onset 或 dev_event<40）；detectable ⇔ 任一已定义 dev ≥40；
  dev_fixed 无可评格 → not_evaluable；onset 存在但 dev_event 无可评格 → 按 dev_fixed 分类
  并标 event_unverified。反向审计 = 成功集逐集组内 leave-one-out 同判据（伪阳性底座）。
  次级变体（描述性）：参考池改为同 task 全 scene 成功集同 q（缓解组内 n<5 的不可评）。
seed 20260903。数据只读，产物只写 E8_portrait_silent/。
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype")
OUT = ROOT / "E8_portrait_silent"
sys.path.insert(0, str(ROOT / "phenotype"))
from stats import groupwise_signflip_maxt  # noqa: E402

SEED = 20260903
NPERM = 2000

CORPORA = ("grid50x8", "main16x32")
INVALID_LOOP = {
    ("grid50x8", "libero_long", "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"),
    ("grid50x8", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("main16x32", "libero_goal", "open_the_middle_drawer_of_the_cabinet"),
}
FIXED_T = {"libero_goal": (8, 12, 16), "libero_object": (8, 12, 16),
           "libero_spatial": (8, 12, 16), "libero_long": (20, 25, 30)}
SIGNALS = ["late_flow_volatility", "route_acceleration", "gate_entropy", "top12_margin",
           "token_consensus", "token_dispersion", "layer_disagreement", "flow_com", "mob1_w8"]
SIG_DISP = ["V", "A", "S_ent", "S_margin", "C", "D_tok", "D_layer", "flow_com", "mob1_w8"]
AXES = ("inst", "comm", "cons", "stick", "flow")
AXIS_LONG = {"inst": "instability z(V+A)", "comm": "commitment z(margin)",
             "cons": "consensus z(C-D_tok)", "stick": "stickiness z(-mob1_w8)",
             "flow": "flow_shape z(flow_com)"}
QMIN = 8          # E8 分析窗
MIN_ROWS_GROUP = 8
MIN_REF = 5       # E9 参考样本下限
DEV_BAND = 40.0   # silent 判据 |pct-50|<40
MIN_SSTAR_VISITS = 50
MIN_GROUP_TRANS = 3
TOP_K_STATES = 3
MIN_PSELF_TOT = 20
EVENT_R = np.arange(-6, 7)
E9_LEADS = np.arange(-4, 1)


# ---------------------------------------------------------------- 数据装载
def load_all():
    rows, ev_frames = [], []
    for corpus in CORPORA:
        for suite_dir in sorted((ROOT / "features" / corpus).iterdir()):
            if not suite_dir.is_dir():
                continue
            suite = suite_dir.name
            for task_dir in sorted(suite_dir.iterdir()):
                task = task_dir.name
                d = np.load(task_dir / "rows.npz")
                n = len(d["episode_id"])
                ep, cs = d["episode_id"].astype(np.int64), d["control_step"].astype(np.int64)
                assert (np.lexsort((cs, ep)) == np.arange(n)).all()
                starts = np.r_[0, np.flatnonzero(np.diff(ep) != 0) + 1]
                lens = np.diff(np.r_[starts, n])
                # 集内 q：control_step 是任务级累计计数，必须按每集首行归零
                q = cs - np.repeat(cs[starts], lens)
                assert (q == np.concatenate([np.arange(c) for c in lens])).all()
                df = pd.DataFrame({
                    "corpus": corpus, "suite": suite, "task": task,
                    "episode_id": ep, "q": q, "scene": d["scene"].astype(int),
                    "repeat": d["repeat"].astype(int), "success": d["success"].astype(int),
                })
                for k in SIGNALS:
                    df[k] = d[k].astype(np.float64)
                rows.append(df)
                ev = pd.read_csv(ROOT / "events" / corpus / suite / task / "events.csv")
                ev.insert(0, "corpus", corpus)
                ev.insert(1, "suite", suite)
                ev.insert(2, "task", task)
                ev_frames.append(ev)
    R = pd.concat(rows, ignore_index=True)
    E = pd.concat(ev_frames, ignore_index=True)
    for D in (R, E):
        D["gid"] = D["corpus"] + "|" + D["suite"] + "|" + D["task"] + "|" + D["scene"].astype(str)
        D["ekey"] = (D["corpus"] + "|" + D["suite"] + "|" + D["task"] + "|"
                     + D["episode_id"].astype(str))
    R["tkey"] = R["corpus"] + "|" + R["suite"] + "|" + R["task"]
    E["tkey"] = E["corpus"] + "|" + E["suite"] + "|" + E["task"]
    bad = E.apply(lambda r: (r["corpus"], r["suite"], r["task"]) in INVALID_LOOP, axis=1)
    E["loop_valid_onset"] = np.where(bad, -1, E["loop_onset_q"])
    E["trap_valid_onset"] = np.where(bad, E["static_onset_q"], E["trap_onset_q"])
    E["loop_channel_valid"] = (~bad).astype(int)
    return R, E


# ---------------------------------------------------------------- E8 轴与宏状态
def build_axes(R):
    raw = dict(inst=(R["late_flow_volatility"] + R["route_acceleration"]).to_numpy(),
               comm=R["top12_margin"].to_numpy(),
               cons=(R["token_consensus"] - R["token_dispersion"]).to_numpy(),
               stick=-R["mob1_w8"].to_numpy(),
               flow=R["flow_com"].to_numpy())
    okwin = R["q"].to_numpy() >= QMIN
    finite = np.ones(len(R), bool)
    for x in raw.values():
        finite &= np.isfinite(x)
    base = okwin & finite
    z = {a: np.full(len(R), np.nan) for a in AXES}
    gids = R["gid"].to_numpy()
    order = np.argsort(gids, kind="stable")
    sg = gids[order]
    bounds = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1, len(sg)]
    dropped, n_groups_total = [], len(bounds) - 1
    for i in range(n_groups_total):
        idx = order[bounds[i]:bounds[i + 1]]
        m = idx[base[idx]]
        if len(m) < MIN_ROWS_GROUP:
            dropped.append((sg[bounds[i]], "too_few_rows", int(len(m))))
            continue
        mad0 = []
        for a in AXES:
            x = raw[a][m]
            med = np.median(x)
            mad = np.median(np.abs(x - med))
            if mad == 0:
                mad0.append(a)
                continue
            z[a][m] = (x - med) / mad
        if mad0:
            dropped.append((sg[bounds[i]], "mad0_" + "+".join(mad0), int(len(m))))
    valid = base.copy()
    for a in AXES:
        valid &= np.isfinite(z[a])
    code32 = np.zeros(len(R), np.int32)
    for b, a in enumerate(AXES):
        code32 |= ((np.nan_to_num(z[a]) > 0).astype(np.int32) << b)
    code32[~valid] = -1
    code16 = np.where(valid, code32 & 0b1111, -1).astype(np.int32)
    return z, code32, code16, valid, dropped, n_groups_total


def sparsity(R, code, n_states, valid):
    dec = {}
    for corpus in CORPORA:
        m = valid & (R["corpus"] == corpus).to_numpy()
        occ = np.bincount(code[m], minlength=n_states) / m.sum()
        dec[corpus] = {"n_states_lt1pct": int((occ < 0.01).sum()),
                       "min_occ": float(occ.min()), "occ": occ.tolist()}
    return dec


def transitions_and_occupancy(R, code, n_states, valid):
    ep = R["ekey"].to_numpy()
    q = R["q"].to_numpy()
    succ = R["success"].to_numpy()
    same_ep = np.zeros(len(R), bool)
    same_ep[:-1] = (ep[1:] == ep[:-1]) & (q[1:] == q[:-1] + 1)
    tpair = same_ep & valid & np.r_[valid[1:], False]
    out_t, out_o = {}, {}
    keys = (R["corpus"] + "|" + R["suite"]).to_numpy()
    for key in np.unique(keys):
        km = keys == key
        for oc in (0, 1):
            m = km & (succ == oc)
            occ = np.bincount(code[m & valid], minlength=n_states).astype(float)
            occ /= max(occ.sum(), 1)
            tm = m & tpair
            cnt = np.zeros((n_states, n_states), np.int64)
            np.add.at(cnt, (code[tm], code[np.flatnonzero(tm) + 1]), 1)
            out_o[(key, oc)] = occ
            out_t[(key, oc)] = cnt
    return out_t, out_o, tpair


def state_label(s, axes):
    """象限语义标注：高/低 × 五（或四）轴。"""
    short = {"inst": ("low-vol", "high-vol"), "comm": ("low-commit", "high-commit"),
             "cons": ("low-consensus", "high-consensus"),
             "stick": ("mobile", "sticky"), "flow": ("early-flow", "late-flow")}
    return "/".join(short[a][(s >> b) & 1] for b, a in enumerate(axes))


def family_test(R, code, n_states, valid, tpair, occ_by, axes, tag):
    """21 格族（7 cell × top-3 Δocc 态）。返回 (cell 记录, eff 矩阵信息)。"""
    keys = (R["corpus"] + "|" + R["suite"]).to_numpy()
    succ = R["success"].to_numpy()
    gid = R["gid"].to_numpy()
    recs, col_eff, col_name = [], [], []
    for key in sorted(set(keys)):
        km = keys == key
        occ_f, occ_s = occ_by[(key, 0)], occ_by[(key, 1)]
        vis_f = np.bincount(code[km & valid & (succ == 0)], minlength=n_states)
        vis_s = np.bincount(code[km & valid & (succ == 1)], minlength=n_states)
        elig = (vis_f >= MIN_SSTAR_VISITS) & (vis_s >= MIN_SSTAR_VISITS)
        docc = np.where(elig, occ_f - occ_s, -np.inf)
        cand = [int(s) for s in np.argsort(-docc)[:TOP_K_STATES] if np.isfinite(docc[s])]
        for rank, s in enumerate(cand):
            geff = {}
            for g in np.unique(gid[km]):
                gm = km & (gid == g)
                ps = {}
                for oc in (0, 1):
                    tm = gm & tpair & (succ == oc) & (code == s)
                    if tm.sum() < MIN_GROUP_TRANS:
                        ps[oc] = np.nan
                    else:
                        ps[oc] = float((code[np.flatnonzero(tm) + 1] == s).mean())
                if np.isfinite(ps[0]) and np.isfinite(ps[1]):
                    geff[g] = ps[0] - ps[1]
            # 池化 P_self（描述性）
            cf, cs_ = None, None
            recs.append(dict(family=tag, cell=key, rank=rank + 1, state=s,
                             label=state_label(s, axes),
                             occ_fail=float(occ_f[s]), occ_succ=float(occ_s[s]),
                             d_occ=float(occ_f[s] - occ_s[s]),
                             visits_fail=int(vis_f[s]), visits_succ=int(vis_s[s]),
                             n_groups=len(geff)))
            col_eff.append(geff)
            col_name.append(f"{key}#s{s}")
            del cf, cs_
    all_groups = sorted({g for d in col_eff for g in d})
    gidx = {g: i for i, g in enumerate(all_groups)}
    eff = np.full((len(all_groups), len(col_eff)), np.nan)
    for j, d in enumerate(col_eff):
        for g, v in d.items():
            eff[gidx[g], j] = v
    keep = [j for j in range(eff.shape[1]) if np.isfinite(eff[:, j]).any()]
    if keep:
        obs, p = groupwise_signflip_maxt(eff[:, keep], NPERM, np.random.default_rng(SEED))
    else:
        obs, p = np.array([]), np.array([])
    for j, r in enumerate(recs):
        if j in keep:
            k = keep.index(j)
            r["mean_dPself"] = float(obs[k]) if np.isfinite(obs[k]) else None
            r["p_maxT"] = float(p[k]) if np.isfinite(obs[k]) else None
        else:
            r["mean_dPself"] = None
            r["p_maxT"] = None
            r["note"] = "untestable: no group with >=3 from-s transitions on both sides"
    return recs, dict(n_family_cells=len(recs), n_tested=len(keep),
                      n_group_rows=len(all_groups))


# ---------------------------------------------------------------- E8 事件轨迹束
def event_bundles(R, E, code, n_states, valid, z, axes):
    idx = {}
    ekeys = R["ekey"].to_numpy()
    q = R["q"].to_numpy()
    order = np.argsort(ekeys, kind="stable")
    se = ekeys[order]
    bounds = np.r_[0, np.flatnonzero(se[1:] != se[:-1]) + 1, len(se)]
    for i in range(len(bounds) - 1):
        idx[se[bounds[i]]] = order[bounds[i]:bounds[i + 1]]
    ib, cb = axes.index("inst"), axes.index("cons")
    out, pre_rows = {}, {}
    for etype, col in (("loop", "loop_valid_onset"), ("static", "static_onset_q")):
        for fail_only in (True, False):
            sel = E[(E[col] >= 0) & (E["success"] == (0 if fail_only else 1))]
            occ = np.zeros((n_states, len(EVENT_R)))
            nvec = np.zeros(len(EVENT_R), int)
            nep = np.zeros(len(EVENT_R), int)
            axsum = {a: np.zeros(len(EVENT_R)) for a in axes}
            axn = {a: np.zeros(len(EVENT_R)) for a in axes}
            pre = []
            for _, r in sel.iterrows():
                rr_all = idx.get(r["ekey"])
                if rr_all is None:
                    continue
                onset = int(r[col])
                qmap = {int(qq): rw for qq, rw in zip(q[rr_all], rr_all)}
                for ci, rel in enumerate(EVENT_R):
                    rw = qmap.get(onset + rel)
                    if rw is None:
                        continue
                    nep[ci] += 1
                    if valid[rw]:
                        occ[code[rw], ci] += 1
                        nvec[ci] += 1
                        if -6 <= rel <= -1:
                            pre.append(rw)
                    for a in axes:
                        v = z[a][rw]
                        if np.isfinite(v):
                            axsum[a][ci] += v
                            axn[a][ci] += 1
            key = f"{etype}_{'fail' if fail_only else 'succ'}"
            pre_rows[key] = np.array(sorted(set(pre)), int)
            st = np.arange(n_states)
            sw_band = ((st >> ib) & 1 == 1) & ((st >> cb) & 1 == 0)
            fl_band = ((st >> ib) & 1 == 0) & ((st >> cb) & 1 == 1)
            occn = occ / np.maximum(nvec, 1)
            out[key] = {"n_events": int(len(sel)), "n_row_present": nep.tolist(),
                        "n_in_window": nvec.tolist(), "occ": occn,
                        "band_switch_desync": occn[sw_band].sum(0).tolist(),
                        "band_flatten_shared": occn[fl_band].sum(0).tolist(),
                        "axis_mean": {a: (axsum[a] / np.maximum(axn[a], 1)).tolist() for a in axes},
                        "axis_n": {a: axn[a].astype(int).tolist() for a in axes}}
    return out, pre_rows


# ---------------------------------------------------------------- E9
def build_ref(R, keycol):
    sm = R["success"].to_numpy() == 1
    gq = R[keycol].to_numpy() + "#" + R["q"].astype(str).to_numpy()
    rows = np.flatnonzero(sm)
    order = rows[np.argsort(gq[rows], kind="stable")]
    key = gq[order]
    b = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(key)]
    return {key[b[i]]: order[b[i]:b[i + 1]] for i in range(len(b) - 1)}


def _pct(x, rv, excl):
    n = len(rv)
    c_less = float((rv < x).sum())
    c_eq = float((rv == x).sum())
    if excl:
        c_eq -= 1.0
        n -= 1
    if n < MIN_REF:
        return np.nan
    return 100.0 * (c_less + 0.5 * c_eq) / n


def _dev_over(ts, qmap, ref, gid, sig_mat, excl):
    best, arg, ncell = np.nan, "", 0
    for t in ts:
        rw = qmap.get(int(t))
        if rw is None:
            continue
        rr = ref.get(f"{gid}#{int(t)}")
        if rr is None:
            continue
        for si, s in enumerate(SIG_DISP):
            x = sig_mat[rw, si]
            if not np.isfinite(x):
                continue
            rv = sig_mat[rr, si]
            rv = rv[np.isfinite(rv)]
            p = _pct(x, rv, excl)
            if np.isnan(p):
                continue
            ncell += 1
            d = abs(p - 50.0)
            if not np.isfinite(best) or d > best:
                best, arg = d, f"{s}@t{int(t)}"
    return best, arg, ncell


def run_e9(R, E, ref, keycol):
    sig_mat = R[SIGNALS].to_numpy()
    q = R["q"].to_numpy()
    ekeys = R["ekey"].to_numpy()
    order = np.argsort(ekeys, kind="stable")
    se = ekeys[order]
    b = np.r_[0, np.flatnonzero(se[1:] != se[:-1]) + 1, len(se)]
    rows_of = {se[b[i]]: order[b[i]:b[i + 1]] for i in range(len(b) - 1)}
    recs = []
    for _, row in E.iterrows():
        rr_all = rows_of.get(row["ekey"])
        if rr_all is None:
            continue
        excl = row["success"] == 1
        gid = row[keycol]
        qmap = {int(qq): rw for qq, rw in zip(q[rr_all], rr_all)}
        dev_f, arg_f, nf = _dev_over(FIXED_T[row["suite"]], qmap, ref, gid, sig_mat, excl)
        onset = int(row["trap_valid_onset"])
        dev_e, arg_e, ne = np.nan, "", 0
        if onset >= 0:
            ts = [onset + int(l) for l in E9_LEADS if onset + int(l) >= 0]
            dev_e, arg_e, ne = _dev_over(ts, qmap, ref, gid, sig_mat, excl)
        if nf == 0:
            status = "not_evaluable"
        else:
            det = (dev_f >= DEV_BAND) or (np.isfinite(dev_e) and dev_e >= DEV_BAND)
            status = "detectable" if det else "silent"
        rec = {k: row[k] for k in ("corpus", "suite", "task", "scene", "repeat",
                                   "episode_id", "success", "n_queries",
                                   "loop_onset_q", "loop_valid_onset", "static_onset_q",
                                   "trap_valid_onset")}
        rec.update(dev_fixed=dev_f, argmax_fixed=arg_f, n_fixed_cells=nf,
                   dev_event=dev_e, argmax_event=arg_e, n_event_cells=ne,
                   has_event=bool(onset >= 0), onset_used=onset, status=status,
                   event_unverified=bool(onset >= 0 and nf > 0 and ne == 0))
        recs.append(rec)
    df = pd.DataFrame(recs)
    return df[df["success"] == 0].reset_index(drop=True), df[df["success"] == 1].reset_index(drop=True)


def agg_status(df):
    g = df.groupby(["corpus", "suite"])["status"].value_counts().unstack(fill_value=0)
    for c in ("silent", "detectable", "not_evaluable"):
        if c not in g:
            g[c] = 0
    g["n"] = g[["silent", "detectable", "not_evaluable"]].sum(axis=1)
    g["silent_frac_evaluable"] = g["silent"] / (g["silent"] + g["detectable"]).clip(lower=1)
    return g.reset_index()[["corpus", "suite", "detectable", "silent", "not_evaluable",
                            "n", "silent_frac_evaluable"]]


def survival_audit(E):
    out = []
    for (corpus, suite), gE in E.groupby(["corpus", "suite"]):
        for t in FIXED_T[suite]:
            for oc in (0, 1):
                sel = gE[gE["success"] == oc]
                out.append(dict(corpus=corpus, suite=suite, t=int(t),
                                outcome="succ" if oc else "fail", n_eps=int(len(sel)),
                                frac_alive=float((sel["n_queries"] > t).mean()) if len(sel) else np.nan))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    R, E = load_all()
    print(f"[load] rows={len(R)} eps={len(E)} ({time.time()-t0:.0f}s)", flush=True)

    z, code32, code16, valid, dropped, n_groups_total = build_axes(R)
    sp32 = sparsity(R, code32, 32, valid)
    sp16 = sparsity(R, code16, 16, valid)
    collapse = any(v["n_states_lt1pct"] > 8 for v in sp32.values())
    primary_n = 16 if collapse else 32
    print(f"[axes] valid={valid.sum()}/{len(R)} dropped_groups={len(dropped)}/{n_groups_total} "
          f"sparse32={ {k: v['n_states_lt1pct'] for k, v in sp32.items()} } primary={primary_n}",
          flush=True)

    ver = {}
    for ns, code, axes in ((32, code32, AXES), (16, code16, AXES[:4])):
        tr, occ_by, tpair = transitions_and_occupancy(R, code, ns, valid)
        ver[ns] = dict(code=code, axes=axes, trans=tr, occ=occ_by, tpair=tpair)

    occ_rows, tr_rows = [], []
    for ns in (32, 16):
        v = ver[ns]
        for key in sorted({k for k, _ in v["occ"]}):
            cf, cs = v["trans"][(key, 0)], v["trans"][(key, 1)]
            for s in range(ns):
                bits = {f"bit_{a}": (s >> b) & 1 for b, a in enumerate(v["axes"])}
                for a in AXES:
                    bits.setdefault(f"bit_{a}", "")

                def pself(c):
                    tot = c[s].sum()
                    return float(c[s, s] / tot) if tot >= MIN_PSELF_TOT else np.nan
                occ_rows.append(dict(n_states=ns, cell=key, state=s,
                                     label=state_label(s, v["axes"]), **bits,
                                     occ_fail=float(v["occ"][(key, 0)][s]),
                                     occ_succ=float(v["occ"][(key, 1)][s]),
                                     d_occ=float(v["occ"][(key, 0)][s] - v["occ"][(key, 1)][s]),
                                     p_self_fail=pself(cf), p_self_succ=pself(cs),
                                     n_from_fail=int(cf[s].sum()), n_from_succ=int(cs[s].sum())))
            for (kk, oc), cnt in v["trans"].items():
                if kk != key:
                    continue
                prob = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
                for i, j in np.argwhere(cnt > 0):
                    tr_rows.append(dict(n_states=ns, cell=key,
                                        outcome="fail" if oc == 0 else "succ",
                                        from_state=int(i), to_state=int(j),
                                        count=int(cnt[i, j]), prob=float(prob[i, j])))
    pd.DataFrame(occ_rows).to_csv(OUT / "macrostate_occupancy.csv", index=False)
    pd.DataFrame(tr_rows).to_csv(OUT / "transitions.csv", index=False)

    fam32, meta32 = family_test(R, code32, 32, valid, ver[32]["tpair"], ver[32]["occ"],
                                AXES, "primary_32")
    fam16, meta16 = family_test(R, code16, 16, valid, ver[16]["tpair"], ver[16]["occ"],
                                AXES[:4], "secondary_16")
    pd.DataFrame(fam32 + fam16).to_csv(OUT / "family_pself.csv", index=False)
    print(f"[family] 32: {meta32}  16: {meta16}", flush=True)

    pn = primary_n
    bundles, pre_rows = event_bundles(R, E, ver[pn]["code"], pn, valid, z, ver[pn]["axes"])

    # E9：主口径（组=(task,scene)）+ 次级（task 池化）
    e9 = {}
    for name, keycol in (("group", "gid"), ("task_pooled", "tkey")):
        ref = build_ref(R, keycol)
        fi, si = run_e9(R, E, ref, keycol)
        e9[name] = (fi, si)
        print(f"[e9-{name}] fail silent={int((fi.status=='silent').sum())}/{len(fi)} "
              f"succ silent={int((si.status=='silent').sum())}/{len(si)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    fail_inv, succ_inv = e9["group"]
    fail_inv.to_csv(OUT / "e9_failure_inventory.csv", index=False)
    succ_inv.to_csv(OUT / "e9_success_fpbase.csv", index=False)
    e9["task_pooled"][0].to_csv(OUT / "e9_failure_inventory_taskpooled.csv", index=False)
    survival_audit(E).to_csv(OUT / "survival_audit.csv", index=False)
    sil = fail_inv[fail_inv["status"] == "silent"].copy()
    sil["ref_variant"] = "group"
    sil2 = e9["task_pooled"][0]
    sil2 = sil2[sil2["status"] == "silent"].copy()
    sil2["ref_variant"] = "task_pooled"
    cols = ["ref_variant", "corpus", "suite", "task", "scene", "repeat", "episode_id",
            "n_queries", "loop_valid_onset", "static_onset_q", "trap_valid_onset",
            "dev_fixed", "argmax_fixed", "n_fixed_cells", "dev_event", "n_event_cells",
            "has_event", "event_unverified"]
    pd.concat([sil[cols], sil2[cols]], ignore_index=True).to_csv(
        OUT / "silent_inventory.csv", index=False)

    # 每任务 silent 表
    bt = (fail_inv.groupby(["corpus", "suite", "task"])["status"]
          .value_counts().unstack(fill_value=0))
    for c in ("silent", "detectable", "not_evaluable"):
        if c not in bt:
            bt[c] = 0
    bt["n_fail"] = bt[["silent", "detectable", "not_evaluable"]].sum(axis=1)
    bt = bt.reset_index()[["corpus", "suite", "task", "n_fail", "detectable", "silent",
                           "not_evaluable"]]
    bt.to_csv(OUT / "silent_by_task.csv", index=False)

    summary = {
        "seed": SEED, "nperm": NPERM, "n_rows": int(len(R)), "n_eps": int(len(E)),
        "e8": {
            "analysis_window_qmin": QMIN,
            "valid_rows": int(valid.sum()),
            "valid_frac": float(valid.mean()),
            "n_groups_total": int(n_groups_total),
            "dropped_groups": {"n": len(dropped),
                               "reasons": pd.Series([d[1] for d in dropped]).value_counts().to_dict()
                               if dropped else {}},
            "sparsity_32": {k: {kk: vv for kk, vv in v.items() if kk != "occ"} for k, v in sp32.items()},
            "sparsity_16": {k: {kk: vv for kk, vv in v.items() if kk != "occ"} for k, v in sp16.items()},
            "collapse_to_16": bool(collapse), "primary_n_states": primary_n,
            "family_primary_32": fam32, "family_meta_32": meta32,
            "family_secondary_16": fam16, "family_meta_16": meta16,
            "event_bundle_n": {k: {"n_events": v["n_events"],
                                   "n_in_window": v["n_in_window"],
                                   "band_switch_desync": v["band_switch_desync"],
                                   "band_flatten_shared": v["band_flatten_shared"]}
                               for k, v in bundles.items()},
        },
        "e9": {
            "variant_group": {
                "failure_status": agg_status(fail_inv).to_dict("records"),
                "success_fpbase": agg_status(succ_inv).to_dict("records"),
                "n_silent": int((fail_inv["status"] == "silent").sum()),
                "n_fail": int(len(fail_inv)),
                "n_succ_silent": int((succ_inv["status"] == "silent").sum()),
                "n_succ": int(len(succ_inv))},
            "variant_task_pooled": {
                "failure_status": agg_status(e9["task_pooled"][0]).to_dict("records"),
                "success_fpbase": agg_status(e9["task_pooled"][1]).to_dict("records"),
                "n_silent": int((e9["task_pooled"][0]["status"] == "silent").sum()),
                "n_fail": int(len(e9["task_pooled"][0])),
                "n_succ_silent": int((e9["task_pooled"][1]["status"] == "silent").sum()),
                "n_succ": int(len(e9["task_pooled"][1]))},
        },
        "seconds": round(time.time() - t0, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    np.savez_compressed(
        OUT / "_portrait_cache.npz",
        **{a: z[a] for a in AXES}, code32=code32, code16=code16, valid=valid,
        success=R["success"].to_numpy().astype(np.int8),
        corpus_is_grid=(R["corpus"] == "grid50x8").to_numpy(),
        suite_code=pd.factorize(R["suite"])[0].astype(np.int8),
        loop_pre_rows=pre_rows["loop_fail"], static_pre_rows=pre_rows["static_fail"],
        primary_n=np.array([primary_n]))
    np.savez_compressed(OUT / "event_bundles.npz", rel=EVENT_R,
                        **{f"{k}_occ": v["occ"] for k, v in bundles.items()},
                        **{f"{k}_n": np.array(v["n_in_window"]) for k, v in bundles.items()})
    (OUT / "_bundles_meta.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "occ"} for k, v in bundles.items()},
        indent=1, default=float))
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
