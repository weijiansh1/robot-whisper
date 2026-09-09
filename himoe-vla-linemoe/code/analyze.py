#!/usr/bin/env python3
"""失败轨迹上的 chunk：flow 去噪还直不直。

按 PREREG.md 冻结的 8 信号 × 3 固定时点执行。信号定义与
analysis_flow_straightness/run.py 一字不差，先于标签存在。

口径与 himoe-vla_trap 对齐：HB 存储层 slice(4,8) = 全局层 12-15，动作 token 1-10，
d9 = 最后一个去噪步，Hellinger 同式，W8 拖尾窗，seed 20260903。

置换在 **episode 级**做（一个 episode 只有一个结局），组 = 初态；
maxT 同时校正 8 信号 × 3 时点 = 24 个单元。

  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 code/analyze.py --nperm 2000
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np

LIVE = 7
DT = 0.1
SEED = 20260903
LAYERS = slice(4, 8)        # 存储 HB 层 [2,3,4,5,12,13,14,15] 的后四层 = 全局 12-15
ACTION_TOKENS = slice(1, 11)
DENOISE_FINAL = 9
WINDOW = 8
FIXED_TIMES = (20, 25, 30)
SIGNALS = [
    "path_over_chord", "cos_min", "cos_chord_mean", "xhat_travel",
    "xhat_drift_mean", "xhat_drift_late", "speed_ratio", "grip_flip",
]
CONTROLS = ["route_mobility_w8", "action_change", "action_magnitude"]


# ------------------------------------------------------------------ 载入

def load_run(d: pathlib.Path):
    parts = sorted(glob.glob(str(d / "flow_traces_part*.npz")))
    if not parts or not (d / "episodes.json").exists():
        return None
    X = np.concatenate([np.load(p)["x_traj"] for p in parts]).squeeze(2).astype(np.float64)
    qid = np.concatenate([np.load(p)["query_id"] for p in parts])
    return dict(
        name=d.name, dir=d, X=X, qid=qid,
        eps=json.load(open(d / "episodes.json")),
        recs=json.load(open(d / "query_records.json")),
        chunks=np.load(d / "candidate_chunks.npz")["chunks"].astype(np.float64),
    )


# ------------------------------------------------------------------ 冻结信号

def chunk_signals(X):
    live = X[:, :, :, :LIVE]
    v = (live[:, :10] - live[:, 1:]) / DT
    t = 1.0 - DT * np.arange(10)
    chord = live[:, 0] - live[:, 10]

    def rn(a):
        return np.linalg.norm(a.reshape(len(a), -1), axis=1)

    def cos(u, w):
        uf, wf = u.reshape(len(u), -1), w.reshape(len(w), -1)
        return (uf * wf).sum(1) / np.maximum(
            np.linalg.norm(uf, axis=1) * np.linalg.norm(wf, axis=1), 1e-12)

    speed = np.stack([rn(v[:, m]) for m in range(10)], 1)
    cos_cons = np.stack([cos(v[:, m], v[:, m + 1]) for m in range(9)], 1)
    cos_ch = np.stack([cos(v[:, m], chord) for m in range(10)], 1)
    xhat = np.concatenate(
        [live[:, :10] - t[None, :, None, None] * v, live[:, 10:11]], 1)
    fin = np.maximum(rn(live[:, 10]), 1e-9)
    drift = np.stack([rn(xhat[:, m + 1] - xhat[:, m]) / fin for m in range(10)], 1)
    g = xhat[:, :, :, 6]
    return {
        "path_over_chord": DT * speed.sum(1) / np.maximum(rn(chord), 1e-9),
        "cos_min": cos_cons.min(1),
        "cos_chord_mean": cos_ch.mean(1),
        "xhat_travel": rn(xhat[:, 0] - live[:, 10]) / fin,
        "xhat_drift_mean": drift.mean(1),
        "xhat_drift_late": drift[:, 5:].mean(1),
        "speed_ratio": speed[:, 9] / np.maximum(speed[:, 0], 1e-9),
        "grip_flip": (np.sign(g[:, :-1]) != np.sign(g[:, 1:])).sum(axis=(1, 2)).astype(float),
    }


def hellinger(p, q):
    bc = np.sqrt(np.maximum(p, 0.0) * np.maximum(q, 0.0)).sum(axis=-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def route_mobility(run, ep_key, ctrl):
    """W8 相邻 chunk Hellinger 变化率（既有 train-free 基线）。"""
    import zarr
    z = run["dir"] / "routes.zarr"
    if not z.exists():
        return None
    g = zarr.open_group(str(z), mode="r")
    probs = np.asarray(g["hb_router_probs"]).astype(np.float32)
    cell = probs[:, LAYERS, DENOISE_FINAL, ACTION_TOKENS, :]     # (n, 4, 10, 32)
    out = np.full(len(cell), np.nan)
    for e in np.unique(ep_key):
        idx = np.where(ep_key == e)[0]
        idx = idx[np.argsort(ctrl[idx])]
        c = cell[idx]
        d = hellinger(c[1:], c[:-1]).mean(axis=(1, 2))            # 40 格均值
        per = np.r_[np.nan, d]
        roll = np.full(len(per), np.nan)
        for i in range(WINDOW, len(per)):
            roll[i] = np.nanmean(per[i - WINDOW + 1:i + 1])
        out[idx] = roll
    return out


def action_controls(chunks, ep_key, ctrl):
    n = len(chunks)
    ac = np.full(n, np.nan)
    for e in np.unique(ep_key):
        idx = np.where(ep_key == e)[0]
        idx = idx[np.argsort(ctrl[idx])]
        c = chunks[idx].reshape(len(idx), -1)
        ac[idx[1:]] = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return {"action_change": ac,
            "action_magnitude": np.linalg.norm(chunks.reshape(n, -1), axis=1)}


# ------------------------------------------------------------------ 统计

def paired_auc(x, y, group):
    """组内成功-失败配对 AUC；跨组不池化。"""
    num = den = 0.0
    ok = np.isfinite(x)
    for gname in np.unique(group):
        m = (group == gname) & ok
        pos, neg = x[m & (y == 1)], x[m & (y == 0)]
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += d.size
    return (num / den if den else np.nan), int(den)


def rank_within(x, group):
    r = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    for gname in np.unique(group):
        m = (group == gname) & ok
        if m.sum() < 2:
            continue
        r[m] = (np.argsort(np.argsort(x[m])) + 0.5) / m.sum()
    return r


def residualise(x, base, group):
    rx, rb = rank_within(x, group), rank_within(base, group)
    out = np.full(len(x), np.nan)
    for gname in np.unique(group):
        m = (group == gname) & np.isfinite(rx) & np.isfinite(rb)
        if m.sum() < 3:
            continue
        A = np.c_[np.ones(m.sum()), rb[m]]
        coef, *_ = np.linalg.lstsq(A, rx[m], rcond=None)
        out[m] = rx[m] - A @ coef
    return out


def joint_maxt(cells, ep_of_row, ep_label, ep_group, nperm, rng):
    """cells: list of (name, values, row_mask, group_of_row)。

    置换在 episode 级、组内进行，然后广播回行；maxT 覆盖全部 cells。
    """
    eps = list(ep_label)
    y_ep = np.array([ep_label[e] for e in eps])
    g_ep = np.array([ep_group[e] for e in eps])
    idx_of = {e: i for i, e in enumerate(eps)}
    row_ep_idx = np.array([idx_of[e] for e in ep_of_row])

    obs = {}
    for name, vals, mask, grp in cells:
        a, npair = paired_auc(vals[mask], y_ep[row_ep_idx][mask], grp[mask])
        obs[name] = (a, npair)
    dev = {k: abs(v[0] - 0.5) for k, v in obs.items() if np.isfinite(v[0])}

    maxdev = np.zeros(nperm)
    yp_ep = y_ep.copy()
    for i in range(nperm):
        for gname in np.unique(g_ep):
            m = g_ep == gname
            yp_ep[m] = rng.permutation(y_ep[m])
        yrow = yp_ep[row_ep_idx]
        d = 0.0
        for name, vals, mask, grp in cells:
            a, _ = paired_auc(vals[mask], yrow[mask], grp[mask])
            if np.isfinite(a):
                d = max(d, abs(a - 0.5))
        maxdev[i] = d
    p = {k: float((maxdev >= dev[k]).sum() + 1) / (nperm + 1)
         for k in dev}
    return obs, p


# ------------------------------------------------------------------ eef-only 代理（描述性）

def eef_loop_proxy(states):
    """只用 8 维本体状态的 loop 代理：eef 位置复现 + 中间走过足够路程。

    这不是 trap 工作那个经 A 稠密真值验证的代理 —— 那个还需要物体位置，
    本次 capture 没有记录。仅作描述性分层，不作显著性主张。
    """
    eef = np.asarray(states)[:, :3]
    step = np.linalg.norm(np.diff(eef, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(step)]
    for right in range(3, len(eef)):
        for left in range(0, right - 2):
            if (np.linalg.norm(eef[right] - eef[left]) <= 0.045
                    and cum[right] - cum[left] >= 0.120):
                return right
    return -1


# ------------------------------------------------------------------ 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="capture")
    ap.add_argument("--out", default="results")
    ap.add_argument("--nperm", type=int, default=2000)
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent.parent
    cap, out = here / args.capture, here / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    runs = [load_run(d) for d in sorted(cap.glob("init*"))]
    runs = [r for r in runs if r]
    if not runs:
        print("没有可用 run"); return 1

    S = {k: [] for k in SIGNALS}
    ctl_parts = {k: [] for k in CONTROLS}
    ep_of_row, grp_of_row, ctrl_of_row = [], [], []
    ep_label, ep_group, ep_meta = {}, {}, {}

    for r in runs:
        q2e = {}
        for e in r["eps"]:
            key = f'i{e["init_state_id"]:02d}s{e["flow_noise_seed"]}'
            ep_label[key] = int(e["success"])
            ep_group[key] = str(e["init_state_id"])
            ep_meta[key] = e
            for q in range(e["query_first"], e["query_last"] + 1):
                q2e[q] = key
        keys = np.array([q2e[int(q)] for q in r["qid"]])
        ctrl = np.array([rec["control_step"] for rec in r["recs"]])
        sig = chunk_signals(r["X"])
        for k in SIGNALS:
            S[k].append(sig[k])
        ac = action_controls(r["chunks"], keys, ctrl)
        rm = route_mobility(r, keys, ctrl)
        ctl_parts["action_change"].append(ac["action_change"])
        ctl_parts["action_magnitude"].append(ac["action_magnitude"])
        ctl_parts["route_mobility_w8"].append(
            rm if rm is not None else np.full(len(ctrl), np.nan))
        ep_of_row.append(keys)
        grp_of_row.append(np.array([ep_group[k] for k in keys]))
        ctrl_of_row.append(ctrl)
        # eef 代理
        for e in r["eps"]:
            key = f'i{e["init_state_id"]:02d}s{e["flow_noise_seed"]}'
            st = [rec["state"] for rec in r["recs"]
                  if e["query_first"] <= rec["query_id"] <= e["query_last"]]
            ep_meta[key]["eef_loop_onset"] = int(eef_loop_proxy(st)) if len(st) > 3 else -1

    S = {k: np.concatenate(v) for k, v in S.items()}
    S.update({k: np.concatenate(v) for k, v in ctl_parts.items()})
    ep_of_row = np.concatenate(ep_of_row)
    grp = np.concatenate(grp_of_row)
    ctrl = np.concatenate(ctrl_of_row)
    y_row = np.array([ep_label[e] for e in ep_of_row])

    # ---- 存活审计：固定时点上还剩多少集 ----
    survive = []
    for t in FIXED_TIMES:
        m = ctrl == t
        e_here = set(ep_of_row[m])
        survive.append({
            "t": t, "episodes_present": len(e_here),
            "success_present": sum(ep_label[e] for e in e_here),
            "failure_present": sum(1 - ep_label[e] for e in e_here),
            "groups_with_both": int(sum(
                1 for gname in set(ep_group.values())
                if any(ep_label[e] == 1 and ep_group[e] == gname for e in e_here)
                and any(ep_label[e] == 0 and ep_group[e] == gname for e in e_here))),
        })

    # ---- 主检验：24 单元联合 maxT ----
    cells = []
    for t in FIXED_TIMES:
        m = ctrl == t
        for k in SIGNALS:
            cells.append((f"{k}@t{t}", S[k], m, grp))
    obs, p = joint_maxt(cells, ep_of_row, ep_label, ep_group, args.nperm, rng)

    # ---- 残差检验：对 route_mobility_w8 做组内秩残差，同样 24 单元联合 ----
    res_cells = []
    have_rm = np.isfinite(S["route_mobility_w8"]).any()
    if have_rm:
        for t in FIXED_TIMES:
            m = ctrl == t
            for k in SIGNALS:
                res = residualise(S[k], S["route_mobility_w8"], grp)
                res_cells.append((f"{k}@t{t}", res, m & np.isfinite(res), grp))
        obs_r, p_r = joint_maxt(res_cells, ep_of_row, ep_label, ep_group,
                                args.nperm, rng)
    else:
        obs_r, p_r = {}, {}

    main_tbl = []
    for t in FIXED_TIMES:
        for k in SIGNALS:
            name = f"{k}@t{t}"
            a, npair = obs[name]
            row = {"t": t, "signal": k, "auc_success_high": None if not np.isfinite(a) else float(a),
                   "det_auc": None if not np.isfinite(a) else float(max(a, 1 - a)),
                   "pairs": npair, "p_maxT": p.get(name)}
            if name in obs_r:
                ar, _ = obs_r[name]
                row["residual_det_auc"] = None if not np.isfinite(ar) else float(max(ar, 1 - ar))
                row["residual_p_maxT"] = p_r.get(name)
            main_tbl.append(row)

    ctl_tbl = []
    for t in FIXED_TIMES:
        m = ctrl == t
        for k in CONTROLS:
            a, npair = paired_auc(S[k][m], y_row[m], grp[m])
            ctl_tbl.append({"t": t, "signal": k, "pairs": npair,
                            "auc_success_high": None if not np.isfinite(a) else float(a),
                            "det_auc": None if not np.isfinite(a) else float(max(a, 1 - a))})

    # ---- 描述：成功/失败中位数 & 头尾对照 ----
    desc, tail = [], {}
    for t in FIXED_TIMES:
        m = ctrl == t
        for k in SIGNALS + CONTROLS:
            a = S[k][m & (y_row == 1)]; b = S[k][m & (y_row == 0)]
            desc.append({"t": t, "signal": k,
                         "median_success": float(np.nanmedian(a)) if len(a) else None,
                         "median_failure": float(np.nanmedian(b)) if len(b) else None})
    for k in SIGNALS:
        h, tl = [], []
        for e in np.unique(ep_of_row):
            m = ep_of_row == e
            if m.sum() < 16:
                continue
            o = np.argsort(ctrl[m]); v = S[k][m][o]
            h.append(np.nanmedian(v[:8])); tl.append(np.nanmedian(v[-8:]))
        if h:
            tail[k] = {"head8": float(np.nanmedian(h)), "tail8": float(np.nanmedian(tl)),
                       "episodes": len(h),
                       "n_up": int(sum(1 for x, y in zip(h, tl) if y > x))}

    overall = {k: {"median_all": float(np.nanmedian(S[k])),
                   "median_on_success_eps": float(np.nanmedian(S[k][y_row == 1])),
                   "median_on_failure_eps": float(np.nanmedian(S[k][y_row == 0]))}
               for k in SIGNALS}

    ep_rows = [dict(key=k, **{kk: vv for kk, vv in v.items()}) for k, v in ep_meta.items()]
    res = {
        "prereg": "PREREG.md",
        "runs": [r["name"] for r in runs],
        "chunks": int(len(ctrl)), "episodes": len(ep_label),
        "success_rate": float(np.mean(list(ep_label.values()))),
        "init_states": sorted({int(g) for g in ep_group.values()}),
        "nperm": args.nperm, "seed": SEED,
        "layers": "stored slice(4,8) = global HB 12-15",
        "survival_at_fixed_times": survive,
        "fixed_time_main": main_tbl,
        "fixed_time_controls": ctl_tbl,
        "group_medians": desc,
        "overall_by_episode_outcome": overall,
        "head8_vs_tail8": tail,
        "episodes_detail": ep_rows,
    }
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in
                      ("chunks", "episodes", "success_rate", "init_states",
                       "survival_at_fixed_times")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
