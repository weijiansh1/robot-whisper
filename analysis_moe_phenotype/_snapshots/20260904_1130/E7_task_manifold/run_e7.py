"""E7 任务条件路由流形（PROTOCOL §5-E7）。

Q1: 每 suite 内 10 任务、early q∈{2..6} reps 的 1-NN LOEO 任务判别（V2 Hellinger）。
Q2: 逐集 mismatch = d_own − d_other（5-NN 中位），组内 (task,scene) 成功–失败配对 AUC，
    episode 级置换 maxT（族 = 该 suite 的窗口数）。

距离（唯一合法）：rep reshape (40,32)（已是 sqrt(P)），逐 cell bc=Σ_e r1·r2，
hell=sqrt(clip(1−bc,0,1))，40 cell 取均值；集间 = 匹配 within-episode query 的均值。

运行：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=8 python run_e7.py
"""

import json
import os
import sys
import time

import numpy as np

BASE = "/home/jovyan/work/himoe-vla/analysis_moe_phenotype"
FEAT = os.path.join(BASE, "features/grid50x8")
OUT = os.path.join(BASE, "E7_task_manifold")
sys.path.insert(0, os.path.join(BASE, "phenotype"))
from stats import joint_maxt, paired_auc, residualise  # noqa: E402

SUITES = ["libero_goal", "libero_long", "libero_object", "libero_spatial"]
SEED = 20260903
NPERM = 2000
NEP_TASK = 400  # 50 scene × 8 repeat
WINDOWS_SHORT = [("early", list(range(2, 7))), ("mid", list(range(8, 13)))]
WINDOWS_LONG = WINDOWS_SHORT + [("late", list(range(20, 25)))]


def wilson(k, n, z=1.959963984540054):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def load_suite(suite):
    """返回 per-episode meta 与 per-(task,q) 行索引；reps 用 memmap 延迟读。"""
    tasks = sorted(os.listdir(os.path.join(FEAT, suite)))
    assert len(tasks) == 10, tasks
    nep = len(tasks) * NEP_TASK
    ep_task = np.full(nep, -1, np.int16)
    ep_scene = np.full(nep, -1, np.int16)
    ep_repeat = np.full(nep, -1, np.int16)
    ep_succ = np.full(nep, -1, np.int8)
    ep_nq = np.zeros(nep, np.int32)
    handles = []  # (memmap, qlocal, gid_of_row)
    for ti, t in enumerate(tasks):
        d = np.load(os.path.join(FEAT, suite, t, "rows.npz"))
        eid, cs = d["episode_id"].astype(np.int64), d["control_step"].astype(np.int64)
        sc, rp, su = d["scene"], d["repeat"], d["success"]
        assert eid.min() >= 0 and eid.max() < NEP_TASK
        first = np.full(NEP_TASK, -1, np.int64)
        # 行内每集连续升序（已在盘点中断言）；首行即最小 control_step
        for i in range(len(eid)):
            if first[eid[i]] < 0:
                first[eid[i]] = cs[i]
        qlocal = (cs - first[eid]).astype(np.int32)
        gid = ti * NEP_TASK + eid
        for g, s, r, s2 in zip(gid, sc, rp, su):
            ep_task[g] = ti
            ep_scene[g] = s
            ep_repeat[g] = r
            ep_succ[g] = s2
        np.add.at(ep_nq, gid, 1)
        assert np.all(eid == sc.astype(np.int64) * 8 + rp), (suite, t)
        reps = np.load(os.path.join(FEAT, suite, t, "reps.npy"), mmap_mode="r")
        assert reps.shape[0] == len(eid) and reps.shape[1] == 1280
        handles.append((reps, qlocal, gid, d["mob1_w8"].astype(np.float64)))
    assert (ep_task >= 0).all()
    return tasks, dict(task=ep_task, scene=ep_scene, repeat=ep_repeat,
                       succ=ep_succ, nq=ep_nq), handles


def gather_q(handles, q):
    """堆出该 q 的全部集 reps → (M,40,32) f32（sqrt 概率），及 gid (M,)。"""
    xs, gs = [], []
    for reps, qlocal, gid, _ in handles:
        idx = np.where(qlocal == q)[0]
        if len(idx):
            xs.append(np.asarray(reps[idx], np.float32))
            gs.append(gid[idx])
    X = np.concatenate(xs).reshape(-1, 40, 32)
    return X, np.concatenate(gs)


def window_distance(handles, qs, nep):
    """(nep,nep) 匹配-q V2 Hellinger 均值；无共同 q 的对 = inf。对角 = inf。"""
    dsum = np.zeros((nep, nep), np.float32)
    cnt = np.zeros((nep, nep), np.uint8)
    for q in qs:
        X, gids = gather_q(handles, q)
        m = len(X)
        if m == 0:
            continue
        H = np.zeros((m, m), np.float32)
        for c in range(40):
            bc = X[:, c, :] @ X[:, c, :].T
            np.subtract(1.0, bc, out=bc)
            np.clip(bc, 0.0, None, out=bc)
            np.sqrt(bc, out=bc)
            H += bc
        H /= 40.0
        if m == nep:
            dsum += H
            cnt += 1
        else:
            ix = np.ix_(gids, gids)
            dsum[ix] += H
            cnt[ix] += 1
    D = dsum / np.maximum(cnt, 1)
    D[cnt == 0] = np.inf
    np.fill_diagonal(D, np.inf)
    return D, cnt


def knn_task_median(D, meta, k=5):
    """逐集对每任务：到该任务成功集的 k-NN 距离中位（对角已 inf → 自身天然被排除）。

    返回 d_task (nep,10)：候选不足 k 用现有；0 个有效候选 = NaN。"""
    nep = D.shape[0]
    d_task = np.full((nep, 10), np.nan, np.float32)
    for t in range(10):
        cols = np.where((meta["task"] == t) & (meta["succ"] == 1))[0]
        if not len(cols):
            continue
        sub = np.sort(D[:, cols], axis=1)  # inf 排尾
        kfin = (np.isfinite(sub)).sum(1)
        top = sub[:, :k].astype(np.float64)
        for i in range(nep):
            kk = min(k, kfin[i])
            if kk > 0:
                d_task[i, t] = np.median(top[i, :kk])
    return d_task


def main():
    t0 = time.time()
    summary = {"seed": SEED, "nperm": NPERM, "chance": 0.1,
               "distance": "V2 Hellinger (40-cell mean, matched within-episode q)",
               "windows": {"early": [2, 6], "mid": [8, 12], "late": [20, 24]},
               "suites": {}}
    mismatch_rows = []  # 失败集清单（全部窗口）

    for suite in SUITES:
        ts = time.time()
        tasks, meta, handles = load_suite(suite)
        nep = len(meta["task"])
        n_fail = int((meta["succ"] == 0).sum())
        windows = WINDOWS_LONG if suite == "libero_long" else WINDOWS_SHORT
        ssum = {"n_episodes": nep, "n_fail": n_fail, "tasks": tasks,
                "family_windows": [w for w, _ in windows], "windows": {}}
        print(f"[{suite}] eps={nep} fail={n_fail} windows={[w for w, _ in windows]}",
              flush=True)

        mm_vals = {}   # window -> (nep,) mismatch
        mm_near = {}   # window -> (nep,) nearest other task idx
        mm_downother = {}
        mm_mob = {}    # window -> (nep,) mob1_w8 窗均值（残差腿基线）
        for wname, qs in windows:
            D, cnt = window_distance(handles, qs, nep)

            if wname == "early":
                # ---- Q1: 1-NN LOEO 任务判别 ----
                assert np.isfinite(D[~np.eye(nep, dtype=bool)]).all(), "early 窗必须全覆盖"
                nn = D.argmin(1)
                pred = meta["task"][nn]
                acc, lo, hi = wilson(int((pred == meta["task"]).sum()), nep)
                conf = np.zeros((10, 10), np.int32)
                np.add.at(conf, (meta["task"].astype(int), pred.astype(int)), 1)
                with open(os.path.join(OUT, f"confusion_{suite}.csv"), "w") as f:
                    f.write("true_task\\pred," + ",".join(tasks) + "\n")
                    for i, t in enumerate(tasks):
                        f.write(t + "," + ",".join(map(str, conf[i])) + "\n")
                # 严格变体：排除同 (task,scene) 邻居
                Ds = D.copy()
                key = meta["task"].astype(np.int64) * 64 + meta["repeat"] * 0 + meta["scene"]
                for g in np.unique(key):
                    ix = np.where(key == g)[0]
                    Ds[np.ix_(ix, ix)] = np.inf
                nns = Ds.argmin(1)
                acc_s, lo_s, hi_s = wilson(int((meta["task"][nns] == meta["task"]).sum()), nep)
                ssum["q1"] = {
                    "acc": acc, "wilson95": [lo, hi],
                    "acc_strict": acc_s, "wilson95_strict": [lo_s, hi_s],
                    "n": nep, "nn_dist_median": float(np.median(D.min(1)))}
                print(f"  Q1 acc={acc:.4f} [{lo:.4f},{hi:.4f}] "
                      f"strict={acc_s:.4f} [{lo_s:.4f},{hi_s:.4f}]", flush=True)
                del Ds

            # ---- Q2: mismatch = d_own − d_other；wrong-manifold ⇔ mismatch>0（d_other<d_own）----
            d_task = knn_task_median(D, meta, k=5)
            own = d_task[np.arange(nep), meta["task"].astype(int)]
            oth = d_task.copy()
            oth[np.arange(nep), meta["task"].astype(int)] = np.nan
            d_other = np.full(nep, np.nan, np.float32)
            near = np.full(nep, -1, np.int32)
            okr = np.isfinite(oth).any(1)
            d_other[okr] = np.nanmin(oth[okr], 1)
            near[okr] = np.nanargmin(oth[okr], 1)
            # 跨场景稳健变体：d_own 候选剔除同 scene 的成功集（成/败候选池完全一致）
            own_cs = np.full(nep, np.nan, np.float32)
            for t in range(10):
                succ_t = np.where((meta["task"] == t) & (meta["succ"] == 1))[0]
                rows_t = np.where(meta["task"] == t)[0]
                for s in np.unique(meta["scene"][rows_t]):
                    ix = rows_t[meta["scene"][rows_t] == s]
                    cols = succ_t[meta["scene"][succ_t] != s]
                    if not len(cols):
                        continue
                    sub = np.sort(D[np.ix_(ix, cols)], 1)
                    kfin = np.isfinite(sub).sum(1)
                    for r, i in enumerate(ix):
                        kk = min(5, kfin[r])
                        if kk > 0:
                            own_cs[i] = np.median(sub[r, :kk])
            mm = own - d_other
            mm_cs = own_cs - d_other
            has_w = np.zeros(nep, bool)
            mob_sum = np.zeros(nep)
            mob_cnt = np.zeros(nep)
            for _, ql, gid, mob in handles:
                for q in qs:
                    ix = np.where(ql == q)[0]
                    has_w[gid[ix]] = True
                    fin = np.isfinite(mob[ix])
                    np.add.at(mob_sum, gid[ix[fin]], mob[ix[fin]])
                    np.add.at(mob_cnt, gid[ix[fin]], 1)
            mobw = np.where(mob_cnt > 0, mob_sum / np.maximum(mob_cnt, 1), np.nan)
            mm[~has_w] = np.nan  # 该窗无任何 query 的集不计
            mm_cs[~has_w] = np.nan
            mm_vals[wname], mm_near[wname] = mm, near
            mm_mob[wname] = mobw
            mm_downother[wname] = (own, d_other, own_cs, mm_cs)
            fmask, smask = meta["succ"] == 0, meta["succ"] == 1
            gk = meta["task"].astype(np.int64) * 64 + meta["scene"].astype(np.int64)
            ssum["windows"][wname] = {
                "n_ep_covered": int(has_w.sum()),
                "n_fail_scored": int(np.isfinite(mm[fmask]).sum()),
                "n_fail_wrong_manifold": int(((mm > 0) & fmask).sum()),
                "n_succ_scored": int(np.isfinite(mm[smask]).sum()),
                "n_succ_wrong_manifold": int(((mm > 0) & smask).sum()),
                "median_mismatch_fail": float(np.nanmedian(mm[fmask])) if np.isfinite(mm[fmask]).any() else None,
                "median_mismatch_succ": float(np.nanmedian(mm[smask])),
                # 描述性分解（组内配对 AUC，不进检验族）
                "desc_auc_d_own": float(paired_auc(own * np.where(has_w, 1, np.nan), fmask.astype(int), gk)[0]),
                "desc_auc_d_other": float(paired_auc(d_other * np.where(has_w, 1, np.nan), fmask.astype(int), gk)[0]),
                "desc_auc_mismatch_cross_scene": float(paired_auc(mm_cs, fmask.astype(int), gk)[0]),
                "n_fail_wrong_manifold_cross_scene": int(((mm_cs > 0) & fmask).sum()),
                "n_succ_wrong_manifold_cross_scene": int(((mm_cs > 0) & smask).sum()),
            }
            w = ssum["windows"][wname]
            print(f"  [{wname}] cov={w['n_ep_covered']} fail_scored={w['n_fail_scored']} "
                  f"wrongF={w['n_fail_wrong_manifold']}/{w['n_fail_scored']} "
                  f"wrongS={w['n_succ_wrong_manifold']}/{w['n_succ_scored']} "
                  f"aucOwn={w['desc_auc_d_own']:.3f} aucOth={w['desc_auc_d_other']:.3f} "
                  f"aucCS={w['desc_auc_mismatch_cross_scene']:.3f} ({time.time() - ts:.0f}s)",
                  flush=True)
            del D, cnt

        # ---- 统计：joint_maxt，族 = 窗口 ----
        gkey = meta["task"].astype(np.int64) * 64 + meta["scene"].astype(np.int64)
        eligible = np.zeros(nep, bool)
        for g in np.unique(gkey):
            ix = np.where(gkey == g)[0]
            s = meta["succ"][ix]
            if (s == 0).any() and (s == 1).any():
                eligible[ix] = True
        rows = np.where(eligible)[0]
        ep_of_row = rows.copy()
        ep_label = {int(e): int(meta["succ"][e] == 0) for e in rows}  # 1=失败
        ep_group = {int(e): int(gkey[e]) for e in rows}
        cells = []
        for wname, _ in windows:
            v = mm_vals[wname][rows]
            cells.append((wname, v, np.isfinite(v), gkey[rows]))
        rng = np.random.default_rng(SEED)
        obs, pval = joint_maxt(cells, ep_of_row, ep_label, ep_group, NPERM, rng)
        # 残差腿（§4 新颖性）：mismatch 对 mob1_w8 窗均值做组内秩残差后重跑同族 maxT
        res_cells = []
        for wname, _ in windows:
            r = residualise(mm_vals[wname][rows], mm_mob[wname][rows], gkey[rows])
            res_cells.append((wname, r, np.isfinite(r), gkey[rows]))
        rng2 = np.random.default_rng(SEED)
        obs_r, pval_r = joint_maxt(res_cells, ep_of_row, ep_label, ep_group, NPERM, rng2)
        for wname, _ in windows:
            a, npair = obs[wname]
            ar, npair_r = obs_r[wname]
            ssum["windows"][wname].update(
                auc_fail_high=float(a), n_pairs=int(npair),
                p_maxt=float(pval.get(wname, np.nan)),
                auc_residual_mob1w8=float(ar), n_pairs_residual=int(npair_r),
                p_maxt_residual=float(pval_r.get(wname, np.nan)))
            print(f"  [{wname}] AUC={a:.3f} pairs={npair} p_maxT={pval.get(wname):.4f} "
                  f"| resid AUC={ar:.3f} p={pval_r.get(wname, np.nan):.4f}",
                  flush=True)
        # 审计用逐集数值（小文件）
        np.savez_compressed(
            os.path.join(OUT, f"mismatch_ep_{suite}.npz"),
            task=meta["task"], scene=meta["scene"], repeat=meta["repeat"],
            success=meta["succ"], n_queries=meta["nq"],
            **{f"mismatch_{w}": mm_vals[w] for w, _ in windows},
            **{f"mismatch_cs_{w}": mm_downother[w][3] for w, _ in windows},
            **{f"nearest_other_{w}": mm_near[w] for w, _ in windows},
            **{f"mob1w8_{w}": mm_mob[w] for w, _ in windows})

        # ---- 失败集清单 ----
        for e in np.where(meta["succ"] == 0)[0]:
            for wname, _ in windows:
                mm = mm_vals[wname][e]
                if not np.isfinite(mm):
                    continue
                own, doth, own_cs, mm_cs = (a[e] for a in mm_downother[wname])
                mismatch_rows.append(dict(
                    suite=suite, task=tasks[meta["task"][e]],
                    scene=int(meta["scene"][e]), repeat=int(meta["repeat"][e]),
                    window=wname, d_own=float(own), d_other=float(doth),
                    mismatch=float(mm),
                    nearest_other_task=tasks[mm_near[wname][e]],
                    wrong_manifold=int(mm > 0),
                    d_own_cross_scene=float(own_cs), mismatch_cross_scene=float(mm_cs),
                    wrong_manifold_cross_scene=int(mm_cs > 0)))
        summary["suites"][suite] = ssum
        print(f"[{suite}] done {time.time() - ts:.0f}s", flush=True)

    import csv
    cols = ["suite", "task", "scene", "repeat", "window", "d_own", "d_other",
            "mismatch", "nearest_other_task", "wrong_manifold",
            "d_own_cross_scene", "mismatch_cross_scene", "wrong_manifold_cross_scene"]
    with open(os.path.join(OUT, "mismatch_scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(mismatch_rows)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print(f"ALL DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
