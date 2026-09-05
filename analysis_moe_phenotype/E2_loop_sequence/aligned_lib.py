"""E2/E3 事件对齐共用库（trap 口径，冻结；PROTOCOL §4 / §5-E2 / §5-E3 / §8 Amendment 1）。

对齐：事件集在 onset+lead 的信号值，对照 = 同组、同绝对 query 的全部 no-that-event 集
（含其他失败）。组级效应 = 事件侧 vs 对照侧 AUC−0.5（逐事件对同 q 对照算 AUC，组内对事件平均）。
检验 = phenotype/stats.groupwise_signflip_maxt（NPERM=2000，seed 20260903），
族 = 8 信号 × 7 lead = 56 格/单位/事件类型（E3 的 min_mobk 单列 7 格小族，见驱动）。
残差腿：信号对 mob1_w8 做组内秩残差，层 = (task 内 scene, 绝对 q)，重跑同族。
产物只写 E2_loop_sequence/ 与 E3_static/。数据只读。
"""

from __future__ import annotations

import csv
import os
import sys
import warnings
import zlib

import numpy as np
from scipy.stats import binomtest, rankdata, t as tdist

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
sys.path.insert(0, os.path.join(BASE, "phenotype"))
from stats import groupwise_signflip_maxt, residualise  # noqa: E402

SEED = 20260903
NPERM = 2000
LEADS = (-4, -3, -2, -1, 0, 1, 2)
SIG8 = [  # (rows.npz 列名, 文档符号)
    ("token_dispersion", "D_tok"), ("token_consensus", "C"),
    ("gate_entropy", "S_ent"), ("top12_margin", "S_margin"),
    ("late_flow_volatility", "V"), ("route_acceleration", "A"),
    ("layer_disagreement", "D_layer"), ("flow_com", "flow_com"),
]
COL_OF = {disp: col for col, disp in SIG8}
COL_OF["min_mobk"] = "min_mobk"

S8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"

ORDER_SIGS = ["D_tok", "V", "A", "C"]  # 次序统计：冻结三元组 D_tok/V/C + A（新腿 a 主张涉及）
ORDER_TRIO = ["D_tok", "V", "C"]
NCTRL_MIN_P90 = 5  # 组对照 90 分位的最少对照集数


def rng_for(*keys):
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


# ------------------------------------------------------------------ data


def load_task(corpus, suite, task):
    d = os.path.join(BASE, "features", corpus, suite, task)
    z = np.load(os.path.join(d, "rows.npz"))
    ep = z["episode_id"].astype(np.int64)
    cs = z["control_step"].astype(np.int64)
    assert np.all(np.diff(ep) >= 0), f"{task}: rows 未按 episode 分块"
    eps, first, nq = np.unique(ep, return_index=True, return_counts=True)
    q = cs - cs[first][np.searchsorted(eps, ep)]
    assert np.all(q == np.concatenate([np.arange(c) for c in nq])), f"{task}: q 不连续"

    sig = {col: z[col].astype(np.float64) for col, _ in SIG8}
    mobk = z["mob_k"].astype(np.float64)          # 列 = k=1..8
    mm = np.full(len(ep), np.nan)
    sub = mobk[:, 1:]                              # k≥2
    any_fin = np.isfinite(sub).any(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mm[any_fin] = np.nanmin(sub[any_fin], 1)
    sig["min_mobk"] = mm

    evp = os.path.join(BASE, "events", corpus, suite, task, "events.csv")
    rows = list(csv.DictReader(open(evp)))
    ev = {int(r["episode_id"]): r for r in rows}
    assert set(ev) == set(int(e) for e in eps), f"{task}: events/features episode 集不一致"
    for i, e in enumerate(eps):
        r = ev[int(e)]
        assert int(r["n_queries"]) == nq[i], f"{task} ep{e}: n_queries 失配"
        assert int(r["scene"]) == int(z["scene"][first[i]])
        assert int(r["success"]) == int(z["success"][first[i]])
    grades = {r["proxy_grade"] for r in rows}
    assert len(grades) == 1
    return dict(
        corpus=corpus, suite=suite, task=task, proxy_grade=grades.pop(),
        eps=eps, first=first, nq=nq, q=q,
        scene_row=z["scene"].astype(np.int64),
        ep_scene=z["scene"][first].astype(np.int64),
        ep_succ=z["success"][first].astype(np.int64),
        sig=sig, mob1w8=z["mob1_w8"].astype(np.float64),
        onsets={f: {int(e): int(ev[int(e)][f]) for e in eps}
                for f in ("loop_onset_q", "static_onset_q")},
    )


def build_unit(tag, specs, event_field, fail_only):
    """specs: [(corpus, suite, task)]；组 = (task 序号, scene)。

    事件侧 = onset≥0（fail_only 时仅失败集；成功集带事件者两侧都排除）。
    对照 = 同组 onset==-1 的全部集（含其他失败）。
    """
    tasks = [load_task(*s) for s in specs]
    groups = {}
    n_ev = n_excl = n_ctrl = 0
    for ti, T in enumerate(tasks):
        on = T["onsets"][event_field]
        for i, e in enumerate(T["eps"]):
            o = on[int(e)]
            g = (ti, int(T["ep_scene"][i]))
            rec = groups.setdefault(g, dict(ev=[], ctrl=[], excl=0))
            if o >= 0:
                if fail_only and T["ep_succ"][i] == 1:
                    rec["excl"] += 1
                    n_excl += 1
                else:
                    rec["ev"].append((i, o))
                    n_ev += 1
            else:
                rec["ctrl"].append(i)
                n_ctrl += 1
    ev_groups = [g for g in sorted(groups) if groups[g]["ev"]]
    return dict(tag=tag, tasks=tasks, groups=groups, ev_groups=ev_groups,
                event_field=event_field, fail_only=fail_only,
                n_events=n_ev, n_ctrl=n_ctrl, n_excl_succ_events=n_excl,
                n_groups=len(ev_groups),
                proxy_grades=sorted({T["proxy_grade"] for T in tasks}))


def add_resid(unit, disp_names):
    """残差腿：逐 task 对 mob1_w8 做组内秩残差；层 = scene*1000 + 绝对 q。"""
    for T in unit["tasks"]:
        if "resid" in T:
            todo = [d for d in disp_names if d not in T["resid"]]
        else:
            T["resid"] = {}
            todo = list(disp_names)
        if not todo:
            continue
        strata = T["scene_row"] * 1000 + T["q"]
        for d in todo:
            T["resid"][d] = residualise(T["sig"][COL_OF[d]], T["mob1w8"], strata)


# ------------------------------------------------------------------ effects


def compute_eff(unit, leg, disp_names, leads=LEADS):
    """→ eff (n_ev_groups, S*L), used (S,L), present (L,), present_ctrl (L,)。

    cell 序 = si*len(leads)+li。present = 事件行存在；present_ctrl = 且 ≥1 对照在场。
    used = 事件值有限且 ≥1 有限对照（逐 cell）。
    """
    G = unit["ev_groups"]
    S, L = len(disp_names), len(leads)
    eff = np.full((len(G), S * L), np.nan)
    used = np.zeros((S, L), int)
    present = np.zeros(L, int)
    present_ctrl = np.zeros(L, int)
    for gi, g in enumerate(G):
        ti, _sc = g
        T = unit["tasks"][ti]
        arrs = [T["sig"][COL_OF[d]] if leg == "raw" else T["resid"][d] for d in disp_names]
        info = unit["groups"][g]
        ctrl = np.asarray(info["ctrl"], int)
        acc = [[[] for _ in range(L)] for _ in range(S)]
        for (i, o) in info["ev"]:
            for li, Ld in enumerate(leads):
                qq = o + Ld
                if qq < 0 or qq >= T["nq"][i]:
                    continue
                if leg == "raw":
                    present[li] += 1
                alive = ctrl[T["nq"][ctrl] > qq] if len(ctrl) else ctrl
                if leg == "raw" and len(alive):
                    present_ctrl[li] += 1
                if not len(alive):
                    continue
                rows_c = T["first"][alive] + qq
                row_e = T["first"][i] + qq
                for si in range(S):
                    x = arrs[si][row_e]
                    cv = arrs[si][rows_c]
                    cv = cv[np.isfinite(cv)]
                    if np.isfinite(x) and len(cv):
                        auc = ((x > cv).sum() + 0.5 * (x == cv).sum()) / len(cv)
                        acc[si][li].append(auc - 0.5)
                        used[si, li] += 1
        for si in range(S):
            for li in range(L):
                if acc[si][li]:
                    eff[gi, si * L + li] = float(np.mean(acc[si][li]))
    return eff, used, present, present_ctrl


def family_stats(eff, rng):
    """sign-flip maxT + 跨组 t 95% CI；全 NaN 组行剔除；obs=NaN 的 cell p 置 NaN。"""
    keep = np.isfinite(eff).any(1)
    eff = eff[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        obs, p = groupwise_signflip_maxt(eff, NPERM, rng)
    p = np.where(np.isfinite(obs), p, np.nan)
    n_grp = np.isfinite(eff).sum(0)
    ci_lo = np.full(eff.shape[1], np.nan)
    ci_hi = np.full(eff.shape[1], np.nan)
    for c in range(eff.shape[1]):
        v = eff[:, c][np.isfinite(eff[:, c])]
        if len(v) >= 2:
            half = tdist.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            ci_lo[c], ci_hi[c] = v.mean() - half, v.mean() + half
    return dict(obs=obs, p=p, n_groups=n_grp, ci_lo=ci_lo, ci_hi=ci_hi,
                n_groups_used=int(keep.sum()))


def eff_mean_se(eff, S, L):
    """画图用：跨组均值与组间 SE。"""
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


# ------------------------------------------------------------------ 次序统计（E2）


def order_stats(unit):
    """逐事件集 D_tok/V/A/C 首次超过组对照 90 分位（同绝对 q，n_ctrl≥NCTRL_MIN_P90）的时刻。

    返回 (per_event 记录, 无覆盖剔除数)。crossing 时刻为绝对 q；rel = q − onset。
    """
    recs = []
    n_no_cov = 0
    for g in unit["ev_groups"]:
        ti, _sc = g
        T = unit["tasks"][ti]
        info = unit["groups"][g]
        ctrl = np.asarray(info["ctrl"], int)
        max_q = max(T["nq"][i] for i, _ in info["ev"])
        p90 = {d: np.full(max_q, np.nan) for d in ORDER_SIGS}
        cov = np.zeros(max_q, bool)
        for qq in range(max_q):
            alive = ctrl[T["nq"][ctrl] > qq] if len(ctrl) else ctrl
            if len(alive) < NCTRL_MIN_P90:
                continue
            cov[qq] = True
            rows_c = T["first"][alive] + qq
            for d in ORDER_SIGS:
                p90[d][qq] = np.percentile(T["sig"][COL_OF[d]][rows_c], 90)
        for (i, o) in info["ev"]:
            nqe = int(T["nq"][i])
            if not cov[:nqe].any():
                n_no_cov += 1
                continue
            rec = dict(group=g, onset=o, nq=nqe,
                       coverage=float(cov[:nqe].mean()), cross={})
            for d in ORDER_SIGS:
                x = T["sig"][COL_OF[d]][T["first"][i]: T["first"][i] + nqe]
                hit = np.flatnonzero(cov[:nqe] & (x > p90[d][:nqe]))
                rec["cross"][d] = int(hit[0]) if len(hit) else None
            recs.append(rec)
    return recs, n_no_cov


def summarise_order(recs):
    """→ (per-signal 概览行, 符号检验行, 三元组中位序行)。"""
    sig_rows, pair_rows = [], []
    for d in ORDER_SIGS:
        rel = [r["cross"][d] - r["onset"] for r in recs if r["cross"][d] is not None]
        sig_rows.append(dict(
            signal=d, n_events=len(recs), n_crossed=len(rel),
            frac_crossed=round(len(rel) / len(recs), 3) if recs else np.nan,
            cross_rel_onset_q25=float(np.percentile(rel, 25)) if rel else np.nan,
            cross_rel_onset_med=float(np.median(rel)) if rel else np.nan,
            cross_rel_onset_q75=float(np.percentile(rel, 75)) if rel else np.nan))
    pairs = [("D_tok", "V"), ("D_tok", "A"), ("D_tok", "C"), ("V", "C")]
    for a, b in pairs:
        ta = [(r["cross"][a], r["cross"][b]) for r in recs]
        both = [(x, y) for x, y in ta if x is not None and y is not None]
        n1 = sum(x < y for x, y in both)
        n2 = sum(y < x for x, y in both)
        ties = len(both) - n1 - n2
        p = binomtest(n1, n1 + n2, 0.5).pvalue if (n1 + n2) else np.nan
        pair_rows.append(dict(
            pair=f"{a}_before_{b}", n_both=len(both), n_first_earlier=n1,
            n_second_earlier=n2, n_ties=ties,
            n_only_first=sum(x is not None and y is None for x, y in ta),
            n_only_second=sum(x is None and y is not None for x, y in ta),
            n_neither=sum(x is None and y is None for x, y in ta),
            p_sign_binom=round(float(p), 5) if np.isfinite(p) else np.nan))
    # 三元组中位序（D_tok/V/C 全部越线的事件）
    ranks = {d: [] for d in ORDER_TRIO}
    n_all3 = 0
    for r in recs:
        ts = [r["cross"][d] for d in ORDER_TRIO]
        if any(t is None for t in ts):
            continue
        n_all3 += 1
        rk = rankdata(ts)
        for d, v in zip(ORDER_TRIO, rk):
            ranks[d].append(v)
    trio = dict(n_all3_crossed=n_all3,
                median_rank={d: (float(np.median(v)) if v else np.nan)
                             for d, v in ranks.items()},
                mean_rank={d: (round(float(np.mean(v)), 3) if v else np.nan)
                           for d, v in ranks.items()})
    return sig_rows, pair_rows, trio


# ------------------------------------------------------------------ 图


UNIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz 参考调色板前三（all-pairs 通过）


def plot_aligned(units_ms, disp_names, leads, out_png, title, extra_panel=None):
    """units_ms: [(label, mean(S,L), se(S,L))]；extra_panel=(name, dict label→(y, se)) 或
    ("presence", dict label→counts)。3×3 小面板，英文标签。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = len(disp_names)
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 8.6), sharex=True)
    axes = axes.ravel()
    x = np.asarray(leads, float)
    for si, d in enumerate(disp_names):
        ax = axes[si]
        for ui, (lab, mean, se) in enumerate(units_ms):
            ax.errorbar(x, mean[si], yerr=se[si], color=UNIT_COLORS[ui], lw=1.8,
                        marker="o", ms=3.5, capsize=2, elinewidth=1.0, label=lab)
        ax.axhline(0.0, color="#9a9992", lw=0.8, ls="--", zorder=0)
        ax.axvline(0.0, color="#c9c8c0", lw=0.8, zorder=0)
        ax.set_title(d, fontsize=10)
        ax.grid(color="#eceae4", lw=0.6)
        ax.tick_params(labelsize=8)
    ax9 = axes[S]
    if extra_panel is not None and extra_panel[0] == "presence":
        for ui, (lab, counts) in enumerate(extra_panel[1].items()):
            ax9.plot(x, counts, color=UNIT_COLORS[ui], lw=1.8, marker="o", ms=3.5,
                     label=lab)
        ax9.set_title("events present (n)", fontsize=10)
        ax9.axvline(0.0, color="#c9c8c0", lw=0.8, zorder=0)
        ax9.grid(color="#eceae4", lw=0.6)
        ax9.tick_params(labelsize=8)
    elif extra_panel is not None:
        name, per_unit = extra_panel
        for ui, (lab, (mean, se)) in enumerate(per_unit.items()):
            ax9.errorbar(x, mean, yerr=se, color=UNIT_COLORS[ui], lw=1.8, marker="o",
                         ms=3.5, capsize=2, elinewidth=1.0, label=lab)
        ax9.axhline(0.0, color="#9a9992", lw=0.8, ls="--", zorder=0)
        ax9.axvline(0.0, color="#c9c8c0", lw=0.8, zorder=0)
        ax9.set_title(name, fontsize=10)
        ax9.grid(color="#eceae4", lw=0.6)
        ax9.tick_params(labelsize=8)
    else:
        ax9.axis("off")
    for ax in axes[6:9]:
        ax.set_xlabel("lead (queries rel. onset)", fontsize=9)
    for ax in axes[[0, 3, 6]]:
        ax.set_ylabel("event vs control AUC − 0.5", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(units_ms), fontsize=9.5,
               frameon=False, bbox_to_anchor=(0.5, 0.972))
    fig.suptitle(title, fontsize=11.5, x=0.5, y=0.995, ha="center")
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ csv


def write_effects_csv(path, rows):
    cols = ["experiment", "unit", "leg", "signal", "lead", "eff_auc_minus_05",
            "ci95_lo", "ci95_hi", "p_maxT", "n_groups", "n_events_used",
            "n_events_present", "n_events_total", "family", "named"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def effect_rows(exp, unit, leg, family, disp_names, leads, eff, st, used,
                present, n_total, named_keys):
    out = []
    L = len(leads)
    for si, d in enumerate(disp_names):
        for li, Ld in enumerate(leads):
            c = si * L + li
            out.append(dict(
                experiment=exp, unit=unit, leg=leg, signal=d, lead=Ld,
                eff_auc_minus_05=_r(st["obs"][c]), ci95_lo=_r(st["ci_lo"][c]),
                ci95_hi=_r(st["ci_hi"][c]), p_maxT=_r(st["p"][c], 4),
                n_groups=int(st["n_groups"][c]), n_events_used=int(used[si, li]),
                n_events_present=int(present[li]), n_events_total=n_total,
                family=family, named=(d, Ld) in named_keys and "yes" or ""))
    return out


def _r(x, nd=4):
    return round(float(x), nd) if np.isfinite(x) else ""
