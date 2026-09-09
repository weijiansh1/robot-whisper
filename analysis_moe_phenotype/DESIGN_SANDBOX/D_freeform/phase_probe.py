"""补充臂 P（相位通道）：在**两个** SCENE8 单位上自查 E5 的 d_healthy / v̂ 是否值得进主臂。

口径（与 E5 同族，但改成本方案的在线契约）：
  - 模板 = 参照池 R_g（**除本组以外**的 success 集）在**同一绝对 query q**（+-B）上的 d9 表示行。
    -> 与主臂共用同一堵墙：q 超过 q_hi 后没有健康参照。
  - d_healthy(q) = 到 R_g 的 5-NN V2-Hellinger 距离中位；phi_hat(q) = 5 个 NN 的归一化进度中位。
  - v_hat(q) = phi_hat(q) - phi_hat(q-1)；frac_vpos = 集内 v_hat>0 的比例（E5 §4 的 slow-normal 判据）。
  - 集级统计量：max_q z(d_healthy)（与主臂同样的 trailing-mean(W=5) 匹配滤波 + 同一 op 窗）。
评估：组内配对 AUC（失败 vs 干净成功），并与主臂 / 消融臂 C 的同口径 AUC 对照。
只读数据；写 phase_probe.json。
"""
import numpy as np, csv, json
import detector as DET

ROOT = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
TASK = "libero_long/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
UNITS = ["main16x32", "grid50x8"]
B = 2
K = 5


def v2_hell(Ae, Ar):
    """Ae (ne,40,32), Ar (nr,40,32) -> (ne,nr) 逐 (layer,token) 格 Hellinger 后对 40 格取均值。"""
    ne, nc, _ = Ae.shape
    nr = Ar.shape[0]
    acc = np.zeros((ne, nr), np.float32)
    for c in range(nc):
        bc = Ae[:, c, :] @ Ar[:, c, :].T
        np.clip(bc, None, 1.0, out=bc)
        acc += np.sqrt(np.maximum(1.0 - bc, 0.0))
    return acc / nc


def run(corp):
    fdir = f"{ROOT}/features/{corp}/{TASK}"
    d = np.load(f"{fdir}/rows.npz", allow_pickle=True)
    ep = d["episode_id"].astype(np.int64); cs = d["control_step"].astype(np.int64)
    scn = d["scene"].astype(np.int64); suc = d["success"].astype(np.int64)
    q = np.zeros_like(cs); seen = {}
    for i in range(len(ep)):
        if ep[i] not in seen: seen[ep[i]] = cs[i]
        q[i] = cs[i] - seen[ep[i]]
    nq = {}
    for e in np.unique(ep):
        nq[int(e)] = int((ep == e).sum())
    R = np.load(f"{fdir}/reps.npy").astype(np.float32).reshape(-1, 40, 32)
    ev = {}
    with open(f"{ROOT}/events/{corp}/{TASK}/events.csv") as fh:
        for r in csv.DictReader(fh):
            ev[int(r["episode_id"])] = (int(r["loop_onset_q"]), int(r["static_onset_q"]), int(r["success"]))
    ref_pool = np.array([suc[i] == 1 for i in range(len(ep))])   # success rows (LOGO applied per group)
    qmax = int(q.max())
    dh = np.full(len(ep), np.nan)      # d_healthy
    ph = np.full(len(ep), np.nan)      # phi_hat
    prog = np.array([q[i] / max(nq[int(ep[i])] - 1, 1) for i in range(len(ep))], np.float32)
    for qq in range(0, qmax + 1):
        tgt = np.flatnonzero(q == qq)
        ref = np.flatnonzero(ref_pool & (np.abs(q - qq) <= B))
        if tgt.size == 0 or ref.size < 20:
            continue
        Dm = v2_hell(R[tgt], R[ref])
        gsrc = scn[ref]
        for gi in np.unique(scn[tgt]):
            rows = np.flatnonzero(scn[tgt] == gi)
            cols = np.flatnonzero(gsrc != gi)                     # LOGO
            if cols.size < K + 1:
                continue
            sub = Dm[np.ix_(rows, cols)]
            idx = np.argpartition(sub, K, axis=1)[:, :K]
            take = np.take_along_axis(sub, idx, 1)
            dh[tgt[rows]] = np.median(take, axis=1)
            ph[tgt[rows]] = np.median(prog[ref[cols][idx]], axis=1)
    # z-standardise d_healthy against the same LOGO success reference at same q, then trailing mean W=5
    cfg = DET.CONFIG
    zdh = np.full(len(ep), np.nan)
    for qq in range(cfg["q_lo"], qmax + 1):
        tgt = np.flatnonzero(q == qq)
        for gi in np.unique(scn[tgt]):
            ref = np.flatnonzero(ref_pool & (scn != gi) & (np.abs(q - qq) <= B) & np.isfinite(dh))
            if ref.size < cfg["n_ref_rows"]:
                continue
            m = np.median(dh[ref]); s = 1.4826 * np.median(np.abs(dh[ref] - m))
            if s <= 0: continue
            sel = tgt[scn[tgt] == gi]
            zdh[sel] = (dh[sel] - m) / s
    # per-episode statistics
    S = DET.score(f"{fdir}/rows.npz", "scene")
    q_hi = {g: S["per_group"][g]["q_hi"] for g in S["groups"]}
    out = {}
    for e in np.unique(ep):
        i = np.flatnonzero(ep == e)
        g = int(scn[i[0]])
        G = DET._trailing_mean(zdh, i[0], i[-1] + 1, cfg["W_static"])
        qs = np.arange(len(i))
        op = (qs >= cfg["q_lo"]) & (qs <= q_hi[g])
        m = op & np.isfinite(G)
        vh = np.diff(ph[i])
        vv = vh[np.isfinite(vh)]
        out[int(e)] = dict(group=g, maxZdh=float(np.max(G[m])) if m.any() else -np.inf,
                           frac_vpos=float(np.mean(vv > 0)) if vv.size else None,
                           spearman_ok=None)
    return out, ev, nq


def paired_auc(vals, ev, pos, neg, key):
    bg = {}
    for e in pos: bg.setdefault(vals[e]["group"], [[], []])[0].append(e)
    for e in neg: bg.setdefault(vals[e]["group"], [[], []])[1].append(e)
    num = den = 0.0; ng = 0
    for g, (P, N) in bg.items():
        if not P or not N: continue
        ng += 1
        for a in P:
            for b in N:
                va, vb = vals[a][key], vals[b][key]
                if va is None or vb is None: continue
                num += 1.0 if va > vb else (0.5 if va == vb else 0.0); den += 1
    return None if den == 0 else dict(auc=round(num / den, 4), pairs=int(den), groups=ng)


if __name__ == "__main__":
    res = {}
    for corp in UNITS:
        vals, ev, nq = run(corp)
        cl = [e for e, m in ev.items() if m[2] == 1 and m[0] < 0 and m[1] < 0]
        flo = [e for e, m in ev.items() if m[2] == 0 and m[0] >= 0 and m[1] < 0]
        fst = [e for e, m in ev.items() if m[2] == 0 and m[1] >= 0 and m[0] < 0]
        fall = [e for e, m in ev.items() if m[2] == 0]
        # slow success = 成功集 n_queries 上三分位
        cut = np.percentile([nq[e] for e in cl], 66.7)
        slow = [e for e in cl if nq[e] >= cut]
        res[corp] = dict(
            AUC_dhealthy_failall_vs_clean=paired_auc(vals, ev, fall, cl, "maxZdh"),
            AUC_dhealthy_looponly_vs_clean=paired_auc(vals, ev, flo, cl, "maxZdh"),
            AUC_dhealthy_staticonly_vs_clean=paired_auc(vals, ev, fst, cl, "maxZdh"),
            frac_vpos=dict(
                clean_success=round(float(np.mean([vals[e]["frac_vpos"] for e in cl if vals[e]["frac_vpos"] is not None])), 4),
                slow_success=round(float(np.mean([vals[e]["frac_vpos"] for e in slow if vals[e]["frac_vpos"] is not None])), 4),
                fail_all=round(float(np.mean([vals[e]["frac_vpos"] for e in fall if vals[e]["frac_vpos"] is not None])), 4),
                n_slow=len(slow)),
            AUC_fracvpos_fail_vs_slowsucc=paired_auc(vals, ev, fall, slow, "frac_vpos"),
            n=dict(clean=len(cl), loop_only=len(flo), static_only=len(fst), fail=len(fall)))
        print(corp, json.dumps(res[corp], ensure_ascii=False, indent=1))
    json.dump(res, open("phase_probe.json", "w"), indent=1, ensure_ascii=False)
    print("wrote phase_probe.json")
