#!/usr/bin/env python3
"""E8/E9 辅助统计（图由 make_figs.py 负责）。依赖 run_e8e9.py 已跑完。

产出 _aux.json（并已并入 summary.json['aux']）：
- scene8_comparison：SCENE8-only 失败增驻 top3 + P_self（对照既有 4 轴 16 态版）
- raw_signal_event_curves / phenotype_predictions：原始信号事件对齐 z 曲线与五预测判定
- e9_physical_comparison / e9_not_evaluable / e9_fpbase_overall / e9_taskpooled_*：E9 归因
- band_baselines_taskweighted_succ：两带的任务加权成功基线
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path("/home/jovyan/work/himoe-vla/analysis_moe_phenotype")
OUT = ROOT / "E8_portrait_silent"
sys.path.insert(0, str(OUT))
import run_e8e9 as M  # noqa: E402

AX = list(M.AXES)
S8 = "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"


def bitlabel(s, axes=AX):
    return "".join(a[0].upper() + ("+" if (s >> i) & 1 else "-") for i, a in enumerate(axes))


def group_z(raw, gids, base):
    z = np.full(len(raw), np.nan)
    order = np.argsort(gids, kind="stable")
    sg = gids[order]
    bounds = np.r_[0, np.flatnonzero(sg[1:] != sg[:-1]) + 1, len(sg)]
    for i in range(len(bounds) - 1):
        idx = order[bounds[i]:bounds[i + 1]]
        m = idx[base[idx]]
        if len(m) < 8:
            continue
        x = raw[m]
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad == 0:
            continue
        z[m] = (x - med) / mad
    return z


def main():
    R, E = M.load_all()
    z, code32, code16, valid, dropped, n_groups_total = M.build_axes(R)
    code = code32  # 主版 32 态（见 summary.primary_n_states）
    q = R["q"].to_numpy()
    succ = R["success"].to_numpy()
    task = R["task"].to_numpy()
    corpus = R["corpus"].to_numpy()
    aux = {}

    # ---------- A. SCENE8-only 占用/驻留（对照既有 4 轴 16 态版本） ----------
    s8cmp = {}
    for co in M.CORPORA:
        m = valid & (task == S8) & (corpus == co)
        rows = {}
        for oc, nm in ((0, "fail"), (1, "succ")):
            mm = m & (succ == oc)
            occ = np.bincount(code[mm], minlength=32) / max(mm.sum(), 1)
            rows[nm] = occ
        docc = rows["fail"] - rows["succ"]
        top = np.argsort(docc)[::-1][:3]
        ep = R["ekey"].to_numpy()
        same = np.zeros(len(R), bool)
        same[:-1] = (ep[1:] == ep[:-1]) & (q[1:] == q[:-1] + 1)
        tp = same & valid & np.r_[valid[1:], False]
        ent = []
        for s in top:
            rec = {"state": int(s), "label": bitlabel(s),
                   "occ_fail": float(rows["fail"][s]), "occ_succ": float(rows["succ"][s]),
                   "d_occ": float(docc[s])}
            for oc, nm in ((0, "fail"), (1, "succ")):
                tm = m & tp & (succ == oc) & (code == s)
                nfrom = tm.sum()
                rec[f"p_self_{nm}"] = (float((code[np.flatnonzero(tm) + 1] == s).mean())
                                       if nfrom >= 20 else None)
                rec[f"n_from_{nm}"] = int(nfrom)
            ent.append(rec)
        s8cmp[co] = ent
    aux["scene8_comparison"] = s8cmp

    # ---------- B. 原始信号事件对齐 z 曲线（五表型预测对照） ----------
    gids = R["gid"].to_numpy()
    base = valid.copy()  # 与轴同窗（q>=8 且组有效）
    raw_sigs = {
        "inst(V+A)": (R["late_flow_volatility"] + R["route_acceleration"]).to_numpy(),
        "S_ent": R["gate_entropy"].to_numpy(),
        "S_margin": R["top12_margin"].to_numpy(),
        "C": R["token_consensus"].to_numpy(),
        "D_tok": R["token_dispersion"].to_numpy(),
        "mob1_w8": R["mob1_w8"].to_numpy(),
        "flow_com": R["flow_com"].to_numpy(),
    }
    zs = {k: group_z(v, gids, base) for k, v in raw_sigs.items()}
    ekeys = R["ekey"].to_numpy()
    order = np.argsort(ekeys, kind="stable")
    se = ekeys[order]
    bounds = np.r_[0, np.flatnonzero(se[1:] != se[:-1]) + 1, len(se)]
    rows_of = {se[bounds[i]]: order[bounds[i]:bounds[i + 1]] for i in range(len(bounds) - 1)}
    curves = {}
    for etype, col in (("loop", "loop_valid_onset"), ("static", "static_onset_q")):
        sel = E[(E[col] >= 0) & (E["success"] == 0)]
        acc = {k: np.zeros(len(M.EVENT_R)) for k in zs}
        cnt = {k: np.zeros(len(M.EVENT_R)) for k in zs}
        for _, r in sel.iterrows():
            rr = rows_of.get(r["ekey"])
            if rr is None:
                continue
            onset = int(r[col])
            qmap = {int(qq): ri for qq, ri in zip(q[rr], rr)}
            for ci, rel in enumerate(M.EVENT_R):
                ri = qmap.get(onset + rel)
                if ri is None:
                    continue
                for k in zs:
                    v = zs[k][ri]
                    if np.isfinite(v):
                        acc[k][ci] += v
                        cnt[k][ci] += 1
        curves[etype] = {k: {"mean_z": (acc[k] / np.maximum(cnt[k], 1)).round(3).tolist(),
                             "n": cnt[k].astype(int).tolist()} for k in zs}
    aux["raw_signal_event_curves"] = {"rel": M.EVENT_R.tolist(), **curves}

    rel = M.EVENT_R.tolist()
    i = {r: rel.index(r) for r in (-3, -2, -1, 0, 1, 2)}
    lc, sc = curves["loop"], curves["static"]
    aux["phenotype_predictions"] = {
        "P1_loop_instability_up@-2": {"value": lc["inst(V+A)"]["mean_z"][i[-2]], "expect": ">0"},
        "P2_loop_desync_Dtok_up@-3": {"value": lc["D_tok"]["mean_z"][i[-3]], "expect": ">0"},
        "P3_loop_margin_up@-2": {"value": lc["S_margin"]["mean_z"][i[-2]], "expect": ">0"},
        "P4_loop_C_post_gt_pre": {"value": [lc["C"]["mean_z"][i[1]], lc["C"]["mean_z"][i[2]],
                                            lc["C"]["mean_z"][i[-2]]],
                                  "expect": "C@+1,+2 > C@-2"},
        "P5_static_flatten_shared_sticky@0": {
            "S_ent": sc["S_ent"]["mean_z"][i[0]], "C": sc["C"]["mean_z"][i[0]],
            "mob1_w8": sc["mob1_w8"]["mean_z"][i[0]],
            "expect": "S_ent>0, C>0, mob1_w8<0"},
    }

    # ---------- C. E9 物理对照 + not_evaluable 归因（主口径 group 参考） ----------
    fi = pd.read_csv(OUT / "e9_failure_inventory.csv")
    cmp_rows = {}
    for st, g in fi.groupby("status"):
        cmp_rows[st] = {
            "n": int(len(g)),
            "n_queries_median": float(g["n_queries"].median()),
            "has_event_rate": float(g["has_event"].mean()),
            "loop_rate": float((g["loop_valid_onset"] >= 0).mean()),
            "static_rate": float((g["static_onset_q"] >= 0).mean()),
            "suite_counts": {f"{a}|{b}": int(v) for (a, b), v in
                             g.groupby(["corpus", "suite"]).size().items()},
        }
    aux["e9_physical_comparison"] = cmp_rows
    aux["e9_silent_task_counts"] = {"|".join(k): int(v) for k, v in
                                    fi[fi["status"] == "silent"]
                                    .groupby(["corpus", "suite", "task"]).size().items()}
    aux["e9_silent_n_fixed_cells"] = fi[fi["status"] == "silent"]["n_fixed_cells"]\
        .value_counts().to_dict()

    ne = fi[fi["status"] == "not_evaluable"]
    has_row = [any(r["n_queries"] > t for t in M.FIXED_T[r["suite"]]) for _, r in ne.iterrows()]
    ne_alive = int(np.sum(has_row))
    aux["e9_not_evaluable"] = {
        "n": int(len(ne)), "episode_alive_at_some_t": ne_alive,
        "episode_too_short": int(len(ne) - ne_alive),
        "with_dev_event": int(ne["dev_event"].notna().sum()),
        "with_dev_event_ge40": int((ne["dev_event"] >= 40).sum()),
        "top_tasks": {"|".join(k): int(v) for k, v in
                      ne.groupby(["corpus", "suite", "task"]).size()
                      .sort_values(ascending=False).head(6).items()},
    }
    su = pd.read_csv(OUT / "e9_success_fpbase.csv")
    aux["e9_fpbase_overall"] = {
        "n_eval": int((su["status"] != "not_evaluable").sum()),
        "detectable": int((su["status"] == "detectable").sum()),
        "silent": int((su["status"] == "silent").sum()),
        "detectable_frac_of_eval": float((su["status"] == "detectable").sum()
                                         / max((su["status"] != "not_evaluable").sum(), 1)),
    }
    fail_eval = fi[fi["status"] != "not_evaluable"]
    aux["e9_failure_overall"] = {
        "n_eval": int(len(fail_eval)),
        "silent": int((fail_eval["status"] == "silent").sum()),
        "silent_frac_of_eval": float((fail_eval["status"] == "silent").mean()),
        "silent_event_unverified": int(fi[(fi["status"] == "silent")
                                          & fi["event_unverified"]].shape[0]),
    }
    ft = pd.read_csv(OUT / "e9_failure_inventory_taskpooled.csv")
    key = ["corpus", "suite", "task", "scene", "episode_id"]
    mg = fi[key + ["status", "dev_fixed"]].merge(
        ft[key + ["status", "dev_fixed"]], on=key, suffixes=("_grp", "_tp"))
    cross = mg.groupby(["status_grp", "status_tp"]).size()
    aux["e9_taskpooled_cross"] = {f"{a}->{b}": int(v) for (a, b), v in cross.items()}
    aux["e9_taskpooled_overall"] = {
        "n_eval": int((ft["status"] != "not_evaluable").sum()),
        "silent": int((ft["status"] == "silent").sum()),
        "not_evaluable": int((ft["status"] == "not_evaluable").sum()),
    }

    # ---------- D. 两带的任务加权成功基线（供 make_figs.py / 报告引用） ----------
    swit_states = [s for s in range(32) if ((s >> 0) & 1) and not ((s >> 2) & 1)]  # I+ C-
    flat_states = [s for s in range(32) if not ((s >> 0) & 1) and ((s >> 2) & 1)]  # I- C+

    def band_baseline(ev_col):
        sel = E[(E[ev_col] >= 0) & (E["success"] == 0)]
        w = sel.groupby(["corpus", "suite", "task"]).size()
        num_s = num_f = den = 0.0
        for (co, sui, ta), n_ev in w.items():
            m = valid & (corpus == co) & (task == ta) & (succ == 1)
            if m.sum() < 50:
                continue
            c = code[m]
            num_s += n_ev * np.isin(c, swit_states).mean()
            num_f += n_ev * np.isin(c, flat_states).mean()
            den += n_ev
        return num_s / den, num_f / den

    bl, bs = band_baseline("loop_valid_onset"), band_baseline("static_onset_q")
    aux["band_baselines_taskweighted_succ"] = {
        "loop_set": {"switch": bl[0], "flat": bl[1]},
        "static_set": {"switch": bs[0], "flat": bs[1]}}

    (OUT / "_aux.json").write_text(json.dumps(aux, indent=1, default=str))
    # 并入 summary.json['aux']
    sfile = OUT / "summary.json"
    s = json.loads(sfile.read_text())
    s["aux"] = aux
    det = fi[fi["status"] == "detectable"].copy()
    det["sig"] = det["argmax_fixed"].str.replace(r"@t\d+", "", regex=True)
    s["e9"]["detectable_argmax_signal"] = det["sig"].value_counts().to_dict()
    s["e9"]["detectable_event_only"] = int(((det["dev_fixed"] < 40)
                                            & (det["dev_event"] >= 40)).sum())
    sfile.write_text(json.dumps(s, indent=1, default=str))
    print("[aux done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
