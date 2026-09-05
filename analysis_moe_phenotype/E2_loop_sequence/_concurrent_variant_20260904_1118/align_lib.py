"""E2/E3 共享的事件对齐机制（PROTOCOL §4 事件对齐类 = trap 口径，冻结）。

口径逐条对应 `himoe-vla_trap/code/analyze_trainfree_signal_matrix.py::matched_group_statistics`：
  事件侧 = 失败集里该通道 onset>=0 的集，在绝对 query q = onset + lead 的信号值；
  对照侧 = **同组、同绝对 q** 的全部 no-that-event 集（含其他失败集，含成功集），
           要求 n_queries > q（仍在场）；
  组级效应 = 事件侧 vs 对照侧的 AUC − 0.5（组内把所有事件集的配对合并成一个 AUC）；
  检验 = `phenotype/stats.py::groupwise_signflip_maxt`（组级 sign-flip + maxT，2000 次）。

与既有 trap 实现的唯一差异：跨组聚合用 stats.py 的**未加权**组均值（trap 版按 pairs 加权）。
本文件按任务指定使用 stats.py，差异在报告"边界"节说明。

control_step 是任务级累计计数，必须按每集首行归零成集内 q（E7 已踩过的坑）。
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
import zlib

import numpy as np

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
sys.path.insert(0, os.path.join(BASE, "phenotype"))
from stats import groupwise_signflip_maxt, residualise  # noqa: E402

SEED = 20260903
NPERM = 2000
LEADS = [-4, -3, -2, -1, 0, 1, 2]

# (报告符号, rows.npz 列名)
SIG8 = [
    ("D_tok", "token_dispersion"),
    ("C", "token_consensus"),
    ("S_ent", "gate_entropy"),
    ("S_margin", "top12_margin"),
    ("V", "late_flow_volatility"),
    ("A", "route_acceleration"),
    ("D_layer", "layer_disagreement"),
    ("flow_com", "flow_com"),
]
SIG8_NAMES = [k for k, _ in SIG8]

S8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"


def rng_for(*parts):
    """从 seed 20260903 派生的确定性 rng（同 E5 的 rng_for 风格）。"""
    return np.random.default_rng([SEED] + [zlib.crc32(p.encode()) for p in parts])


# --------------------------------------------------------------------- loading

def load_unit(tasks, channel, with_min_mobk=False):
    """tasks: [(corpus, suite, task)]；channel ∈ {'loop','static'}。组 = (task, scene)。

    返回 dict：行级信号数组（跨任务拼接）+ 集级 meta + 分组索引。
    """
    col = {"loop": "loop_onset_q", "static": "static_onset_q"}[channel]
    names = list(SIG8_NAMES) + (["min_mob_k"] if with_min_mobk else [])
    sig = {n: [] for n in names}
    mob_parts = []
    ep = dict(first=[], nq=[], gid=[], onset=[], is_event=[], is_ctrl=[],
              success=[], episode_id=[], task=[], scene=[], grade=[])
    groups, group_key = {}, []
    off = 0
    for ti, (corpus, suite, task) in enumerate(tasks):
        d = np.load(os.path.join(BASE, "features", corpus, suite, task, "rows.npz"))
        eid = d["episode_id"].astype(np.int64)
        cs = d["control_step"].astype(np.int64)
        assert np.all(np.diff(eid) >= 0), (task, "rows 未按 episode 分块")
        uniq, first, cnt = np.unique(eid, return_index=True, return_counts=True)
        # control_step = 任务级累计 → 按每集首行归零成集内 q，并断言集内连续
        start = np.zeros(int(uniq.max()) + 1, np.int64)
        start[uniq] = cs[first]
        q = cs - start[eid]
        assert np.all(q == np.concatenate([np.arange(c) for c in cnt])), (task, "q 不连续")

        for n, c in SIG8:
            sig[n].append(d[c].astype(np.float64))
        if with_min_mobk:
            mk = d["mob_k"].astype(np.float64)[:, 1:]      # k = 2..8
            fin = np.isfinite(mk)
            mm = np.where(fin, mk, np.inf).min(1)
            mm[~fin.any(1)] = np.nan
            sig["min_mob_k"].append(mm)
        mob_parts.append(d["mob1_w8"].astype(np.float64))

        evp = os.path.join(BASE, "events", corpus, suite, task, "events.csv")
        evr = {int(e["episode_id"]): e for e in csv.DictReader(open(evp))}
        assert set(evr) == set(int(u) for u in uniq), (task, "events↔rows 集不一致")
        for i, u in enumerate(uniq):
            e = evr[int(u)]
            assert int(e["n_queries"]) == int(cnt[i]), (task, u, "n_queries 不一致")
            assert int(e["scene"]) == int(d["scene"][first[i]])
            assert int(e["success"]) == int(d["success"][first[i]])
            key = (task, int(e["scene"]))
            if key not in groups:
                groups[key] = len(group_key)
                group_key.append(key)
            o, s = int(e[col]), int(e["success"])
            ep["first"].append(off + int(first[i]))
            ep["nq"].append(int(cnt[i]))
            ep["gid"].append(groups[key])
            ep["onset"].append(o)
            ep["is_event"].append(bool(o >= 0 and s == 0))
            ep["is_ctrl"].append(bool(o < 0))
            ep["success"].append(s)
            ep["episode_id"].append(int(u))
            ep["task"].append(ti)
            ep["scene"].append(int(e["scene"]))
            ep["grade"].append(e["proxy_grade"])
        off += len(eid)

    U = {n: np.concatenate(v) for n, v in sig.items()}
    U = dict(sig={n: U[n] for n in names}, mob1w8=np.concatenate(mob_parts))
    for k, v in ep.items():
        U[k] = np.array(v) if k != "grade" else np.array(v, object)
    U["n_rows"] = off
    U["n_groups"] = len(group_key)
    U["group_key"] = group_key
    U["tasks"] = tasks
    U["sig_names"] = names
    U["channel"] = channel
    G = U["n_groups"]
    U["ev_of_group"] = [np.flatnonzero((U["gid"] == g) & U["is_event"]) for g in range(G)]
    U["ct_of_group"] = [np.flatnonzero((U["gid"] == g) & U["is_ctrl"]) for g in range(G)]
    U["all_of_group"] = [np.flatnonzero(U["gid"] == g) for g in range(G)]
    return U


# ------------------------------------------------------------- matched effects

def group_effects(U, vals, sig_names, leads=LEADS):
    """组级 AUC−0.5。返回 eff/n_ev/n_pair 各 (G, S*L)，列序 = signal-major, lead-minor。"""
    G, S, L = U["n_groups"], len(sig_names), len(leads)
    eff = np.full((G, S * L), np.nan)
    n_ev = np.zeros((G, S * L), int)
    n_pair = np.zeros((G, S * L), int)
    for g in range(G):
        evs, cts = U["ev_of_group"][g], U["ct_of_group"][g]
        if not len(evs) or not len(cts):
            continue
        ct_nq = U["nq"][cts]
        for li, lead in enumerate(leads):
            conc = np.zeros(S)
            pairs = np.zeros(S, int)
            nev = np.zeros(S, int)
            for e in evs:
                qq = int(U["onset"][e]) + lead
                if qq < 0 or qq >= U["nq"][e]:
                    continue
                elig = cts[ct_nq > qq]
                if not len(elig):
                    continue
                re = U["first"][e] + qq
                rc = U["first"][elig] + qq
                for si, name in enumerate(sig_names):
                    arr = vals[name]
                    x = arr[re]
                    if not np.isfinite(x):
                        continue
                    cv = arr[rc]
                    cv = cv[np.isfinite(cv)]
                    if not len(cv):
                        continue
                    conc[si] += float((x > cv).sum()) + 0.5 * float((x == cv).sum())
                    pairs[si] += len(cv)
                    nev[si] += 1
            for si in range(S):
                if pairs[si]:
                    c = si * L + li
                    eff[g, c] = conc[si] / pairs[si] - 0.5
                    n_ev[g, c] = nev[si]
                    n_pair[g, c] = pairs[si]
    return eff, n_ev, n_pair


def presence_audit(U, leads=LEADS):
    """在场数审计：逐 lead 统计事件集的越界剔除与对照可得性（信号无关，8 信号全有限）。"""
    rows = []
    n_orphan = sum(len(U["ev_of_group"][g]) for g in range(U["n_groups"])
                   if len(U["ev_of_group"][g]) and not len(U["ct_of_group"][g]))
    for lead in leads:
        tot = before = after = alive = used = 0
        nctrl = []
        for g in range(U["n_groups"]):
            evs, cts = U["ev_of_group"][g], U["ct_of_group"][g]
            if not len(evs) or not len(cts):
                continue
            ct_nq = U["nq"][cts]
            for e in evs:
                tot += 1
                qq = int(U["onset"][e]) + lead
                if qq < 0:
                    before += 1
                    continue
                if qq >= U["nq"][e]:
                    after += 1
                    continue
                alive += 1
                k = int((ct_nq > qq).sum())
                if k:
                    used += 1
                    nctrl.append(k)
        rows.append(dict(lead=lead, n_events_in_paired_groups=tot,
                         n_drop_before_start=before, n_drop_after_episode_end=after,
                         n_alive=alive, n_used=used,
                         median_n_controls=float(np.median(nctrl)) if nctrl else np.nan,
                         min_n_controls=int(min(nctrl)) if nctrl else 0,
                         total_n_controls=int(sum(nctrl))))
    return rows, n_orphan


def residual_values(U, sig_names, leads=LEADS):
    """对 mob1_w8 的组内秩残差（stats.py::residualise），单元格 = (组, 绝对 query)。

    只在检验实际用到的 (group, q) 单元格上计算；单元格内参与排秩的是该组**全部**
    在场且 mob1_w8 有限的集（与 trap 版 stage_rank_residual 一致）。
    """
    cells = set()
    for g in range(U["n_groups"]):
        evs, cts = U["ev_of_group"][g], U["ct_of_group"][g]
        if not len(evs) or not len(cts):
            continue
        for e in evs:
            for lead in leads:
                qq = int(U["onset"][e]) + lead
                if 0 <= qq < U["nq"][e]:
                    cells.add((g, qq))
    rows_idx, cell_id = [], []
    for ci, (g, qq) in enumerate(sorted(cells)):
        mem = U["all_of_group"][g]
        mem = mem[U["nq"][mem] > qq]
        if len(mem) < 3:
            continue
        r = U["first"][mem] + qq
        keep = np.isfinite(U["mob1w8"][r])
        if keep.sum() < 3:
            continue
        rows_idx.append(r[keep])
        cell_id.append(np.full(int(keep.sum()), ci))
    out = {n: np.full(U["n_rows"], np.nan) for n in sig_names}
    if not rows_idx:
        return out, 0
    ridx = np.concatenate(rows_idx)
    cid = np.concatenate(cell_id)
    base = U["mob1w8"][ridx]
    for n in sig_names:
        out[n][ridx] = residualise(U["sig"][n][ridx], base, cid)
    return out, len(set(cid.tolist()))


def run_family(U, vals, sig_names, tag, leg, leads=LEADS):
    """一族 = len(sig_names) × len(leads) 格，组级 sign-flip maxT。返回逐格记录。"""
    eff, n_ev, n_pair = group_effects(U, vals, sig_names, leads)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obs, p = groupwise_signflip_maxt(eff, NPERM, rng_for(tag, leg))
    from scipy.stats import t as tdist
    L = len(leads)
    recs = []
    for si, name in enumerate(sig_names):
        for li, lead in enumerate(leads):
            c = si * L + li
            col = eff[:, c]
            ok = np.isfinite(col)
            k = int(ok.sum())
            if k > 1:
                half = tdist.ppf(0.975, k - 1) * col[ok].std(ddof=1) / np.sqrt(k)
                ci = (float(col[ok].mean() - half), float(col[ok].mean() + half))
            else:
                ci = (np.nan, np.nan)
            recs.append(dict(
                signal=name, lead=int(lead), leg=leg,
                effect=float(obs[c]) if k else np.nan,
                ci_lo=ci[0], ci_hi=ci[1],
                p_maxT=float(p[c]) if k else np.nan,
                n_groups=k, n_groups_pos=int((col[ok] > 0).sum()),
                n_groups_neg=int((col[ok] < 0).sum()),
                n_events_used=int(n_ev[:, c].sum()), n_pairs=int(n_pair[:, c].sum())))
    return recs, eff


# ------------------------------------------------------------------------ 图

UNIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def eff_mean_se(eff, S, L):
    """画图用：逐 cell 的跨组均值与组间 SE。"""
    mean = np.full((S, L), np.nan)
    se = np.full((S, L), np.nan)
    for si in range(S):
        for li in range(L):
            v = eff[:, si * L + li]
            v = v[np.isfinite(v)]
            if len(v):
                mean[si, li] = v.mean()
            if len(v) >= 2:
                se[si, li] = v.std(ddof=1) / np.sqrt(len(v))
    return mean, se


def plot_aligned(units_ms, sig_names, leads, out_png, title, presence=None):
    """units_ms: [(label, mean(S,L), se(S,L))]。小面板 = 信号；最后一格放在场数审计。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = len(sig_names)
    ncol = 3
    nrow = int(np.ceil((S + (1 if presence else 0)) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 2.9 * nrow), sharex=True)
    axes = np.asarray(axes).ravel()
    x = np.asarray(leads, float)
    for si, name in enumerate(sig_names):
        ax = axes[si]
        for ui, (lab, mean, se) in enumerate(units_ms):
            ax.errorbar(x, mean[si], yerr=se[si], color=UNIT_COLORS[ui % 3], lw=1.7,
                        marker="o", ms=3.5, capsize=2, elinewidth=1.0, label=lab)
        ax.axhline(0.0, color="#9a9992", lw=0.8, ls="--", zorder=0)
        ax.axvline(0.0, color="#c9c8c0", lw=0.9, zorder=0)
        ax.set_title(name, fontsize=10)
        ax.grid(color="#ecebe6", lw=0.6)
        ax.tick_params(labelsize=8)
    k = S
    if presence:
        ax = axes[k]
        for ui, (lab, cnt) in enumerate(presence.items()):
            ax.plot(x, cnt, color=UNIT_COLORS[ui % 3], lw=1.7, marker="s", ms=3.5, label=lab)
        ax.set_title("events used (n)", fontsize=10)
        ax.axvline(0.0, color="#c9c8c0", lw=0.9, zorder=0)
        ax.grid(color="#ecebe6", lw=0.6)
        ax.tick_params(labelsize=8)
        k += 1
    for ax in axes[k:]:
        ax.axis("off")
    for i in range(len(axes)):
        if i >= len(axes) - ncol or i >= k - ncol:
            axes[i].set_xlabel("lead (queries rel. onset)", fontsize=9)
    for i in range(0, len(axes), ncol):
        axes[i].set_ylabel("event vs control  AUC - 0.5", fontsize=9)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=len(units_ms), fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(title, fontsize=11.5, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ order stat

MIN_CTRL_P90 = 5


def first_crossing(U, sig_names, window):
    """逐事件集：信号首次超过同组、同绝对 q 对照 90 分位的时刻（相对 onset）。

    window='runup8' → q ∈ [onset−8, onset+2]；'prefix' → q ∈ [0, onset+2]。
    需要该 q 上至少 MIN_CTRL_P90 个在场对照才定义 P90；否则跳过该 q。
    返回 dict[name] -> (n_events, 数组 rel_time，NaN=窗内未越过=删失)。
    """
    out = {n: [] for n in sig_names}
    meta = []
    for g in range(U["n_groups"]):
        evs, cts = U["ev_of_group"][g], U["ct_of_group"][g]
        if not len(evs) or not len(cts):
            continue
        ct_nq = U["nq"][cts]
        for e in evs:
            o = int(U["onset"][e])
            hi = min(o + 2, U["nq"][e] - 1)
            lo = max(0, o - 8) if window == "runup8" else 0
            meta.append((g, e))
            got = {n: np.nan for n in sig_names}
            for qq in range(lo, hi + 1):
                elig = cts[ct_nq > qq]
                if len(elig) < MIN_CTRL_P90:
                    continue
                re = U["first"][e] + qq
                rc = U["first"][elig] + qq
                for n in sig_names:
                    if np.isfinite(got[n]):
                        continue
                    arr = U["sig"][n]
                    x = arr[re]
                    cv = arr[rc]
                    cv = cv[np.isfinite(cv)]
                    if not np.isfinite(x) or len(cv) < MIN_CTRL_P90:
                        continue
                    if x > np.percentile(cv, 90):
                        got[n] = qq - o
            for n in sig_names:
                out[n].append(got[n])
    return {n: np.array(v) for n, v in out.items()}, meta
