"""共享装载层：把 analysis_moe_phenotype 的 features/ + events/ 拼成 cell 级的表。

cell = (corpus, suite, task, scene)，cell 内的 repeat 是同胞（同初始状态、不同 flow noise）。
关键不变量：**q=0 那一行，cell 内所有同胞共享逐位相同的观测**（settle 确定性，
第一次推理前机器人没动过），差异只来自 flow noise ξ。这一点由 assert_q0_snapshot 复核。

术语：q 是**集内** query 序号（rows.npz 的 control_step 是全局连续的，必须重算）。
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np

ROOT = "/home/jovyan/work/himoe-vla"
PHENO = os.path.join(ROOT, "analysis_moe_phenotype")

FEATS = [
    "late_flow_volatility",
    "route_acceleration",
    "gate_entropy",
    "top12_margin",
    "token_consensus",
    "token_dispersion",
    "layer_disagreement",
    "state_action_gap",
    "flow_com",
    "flow_total",
    "mob1_w8",
]

# 事件表 v2 的铰接机构盲区：被操纵对象非 free joint，loop/trap 通道无效（static 不受影响）。
# 出处 analysis_moe_phenotype/audit/AUDIT.md §8。
LOOP_BLIND_TASKS = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    "open_the_middle_drawer_of_the_cabinet",
}


@dataclass
class Task:
    corpus: str
    suite: str
    task: str
    # 逐 (episode, q) 的行
    ep: np.ndarray  # episode_id
    q: np.ndarray  # 集内 query 序号
    scene: np.ndarray
    repeat: np.ndarray
    feats: dict  # name -> (N,)
    reps: np.ndarray  # (N,1280) float16
    # 逐 episode 的事件
    ev: dict  # episode_id -> dict(success, n_queries, loop_onset_q, static_onset_q, trap_onset_q, grade)

    @property
    def loop_valid(self) -> bool:
        return self.task not in LOOP_BLIND_TASKS


def _episode_local_q(ep: np.ndarray, cs: np.ndarray):
    """按 (episode_id, control_step) 排序，返回排序索引与集内 query 序号。"""
    order = np.lexsort((cs, ep))
    e = ep[order]
    q = np.empty(len(e), dtype=np.int32)
    start = 0
    for i in range(1, len(e) + 1):
        if i == len(e) or e[i] != e[start]:
            q[start:i] = np.arange(i - start, dtype=np.int32)
            start = i
    return order, q


def load_task(corpus: str, suite: str, task: str, with_reps: bool = False) -> Task:
    fdir = os.path.join(PHENO, "features", corpus, suite, task)
    edir = os.path.join(PHENO, "events", corpus, suite, task)
    d = np.load(os.path.join(fdir, "rows.npz"), allow_pickle=True)
    order, q = _episode_local_q(d["episode_id"], d["control_step"])
    feats = {k: np.asarray(d[k], dtype=np.float64)[order] for k in FEATS if k in d}
    reps = None
    if with_reps:
        reps = np.load(os.path.join(fdir, "reps.npy"), mmap_mode="r")[order]
    ev = {}
    with open(os.path.join(edir, "events.csv")) as fh:
        for r in csv.DictReader(fh):
            ev[int(r["episode_id"])] = dict(
                success=int(r["success"]),
                n_queries=int(r["n_queries"]),
                loop_onset_q=int(r["loop_onset_q"]),
                static_onset_q=int(r["static_onset_q"]),
                trap_onset_q=int(r["trap_onset_q"]),
                grade=r["proxy_grade"],
            )
    return Task(
        corpus=corpus,
        suite=suite,
        task=task,
        ep=np.asarray(d["episode_id"])[order],
        q=q,
        scene=np.asarray(d["scene"])[order],
        repeat=np.asarray(d["repeat"])[order],
        feats=feats,
        reps=reps,
        ev=ev,
    )


def iter_tasks(corpus: str):
    base = os.path.join(PHENO, "features", corpus)
    for suite in sorted(os.listdir(base)):
        if not os.path.isdir(os.path.join(base, suite)):
            continue
        for task in sorted(os.listdir(os.path.join(base, suite))):
            if os.path.exists(os.path.join(base, suite, task, "rows.npz")):
                yield suite, task


def assert_q0_snapshot(t: Task, tol: float = 5e-3) -> dict:
    """复核 'cell 内 q=0 共享同一观测'。

    直接可测的代理：state_action_gap / gate_entropy 这类主要由观测决定的量，在 cell 内
    q=0 处的相对离散度应当远小于跨 cell 的离散度。返回诊断而非硬断言——真正的证据是
    设计（同 init_state_id、settle 确定性），这里只做量级 sanity。
    """
    m = t.q == 0
    out = {}
    for f in ("gate_entropy", "state_action_gap"):
        if f not in t.feats:
            continue
        v = t.feats[f][m]
        sc = t.scene[m]
        within, between = [], []
        for s in np.unique(sc):
            vv = v[sc == s]
            if len(vv) > 1:
                within.append(vv.std())
                between.append(vv.mean())
        out[f] = dict(
            within_std=float(np.mean(within)),
            between_std=float(np.std(between)),
            ratio=float(np.mean(within) / (np.std(between) + 1e-12)),
        )
    return out


# ---------------------------------------------------------------- outcomes


def outcomes(t: Task):
    """逐 episode 的结局字典 -> 多种 Y 定义。

    Y_fail    : 未完成任务（含 timeout）
    Y_loop    : 致败 loop（有 loop onset 且失败）
    Y_static  : 致败 static
    Y_anyloop : 任意 loop onset（含成功集里的瞬态/自恢复 loop）
    """
    eps = sorted(t.ev)
    Y = {k: np.zeros(len(eps), dtype=np.int8) for k in ("fail", "loop", "static", "anyloop")}
    nq = np.zeros(len(eps), dtype=np.int32)
    for i, e in enumerate(eps):
        r = t.ev[e]
        f = 1 - r["success"]
        Y["fail"][i] = f
        Y["loop"][i] = int(f and r["loop_onset_q"] >= 0)
        Y["static"][i] = int(f and r["static_onset_q"] >= 0)
        Y["anyloop"][i] = int(r["loop_onset_q"] >= 0)
        nq[i] = r["n_queries"]
    return np.asarray(eps), Y, nq


def episode_cells(t: Task):
    """episode_id -> (scene, repeat)，由特征行推出（events.csv 也有，两处交叉核对）。"""
    eps = sorted(t.ev)
    sc = np.full(len(eps), -1, dtype=np.int32)
    rp = np.full(len(eps), -1, dtype=np.int32)
    idx = {e: i for i, e in enumerate(eps)}
    seen = np.zeros(len(eps), dtype=bool)
    for e, s, r in zip(t.ep, t.scene, t.repeat):
        i = idx.get(int(e))
        if i is not None and not seen[i]:
            sc[i], rp[i], seen[i] = s, r, True
    return eps, sc, rp


# ---------------------------------------------------------------- 组内 AUC


def stratified_auc(values: np.ndarray, y: np.ndarray, cell: np.ndarray):
    """按 cell 分层的 Mann-Whitney AUC（P(s_fail > s_succ)）。

    返回 (pooled_auc, n_pairs, per_cell_list)。pooled 按 cell 的配对数加权 —— 这与把所有
    cell 的配对合并成一个 U 统计量是同一个数，不是各 cell AUC 的简单平均。
    """
    num = 0.0
    den = 0.0
    per = []
    for c in np.unique(cell):
        m = cell == c
        v, yy = values[m], y[m]
        pos, neg = v[yy == 1], v[yy == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        gt = (pos[:, None] > neg[None, :]).sum()
        eq = (pos[:, None] == neg[None, :]).sum()
        u = gt + 0.5 * eq
        n = len(pos) * len(neg)
        num += u
        den += n
        per.append((str(c), len(pos), len(neg), u / n))
    if den == 0:
        return float("nan"), 0, per
    return num / den, int(den), per


def perm_null_auc(values: np.ndarray, y: np.ndarray, cell: np.ndarray, n_perm: int, rng):
    """cell 内标签置换的精确零分布（保持每个 cell 的失败数不变）。"""
    cells = np.unique(cell)
    idxs = [np.where(cell == c)[0] for c in cells]
    keep = [i for i in idxs if 0 < y[i].sum() < len(i)]
    if not keep:
        return np.array([])
    out = np.empty(n_perm)
    yp = y.copy()
    for b in range(n_perm):
        for i in keep:
            yp[i] = rng.permutation(y[i])
        out[b] = stratified_auc(values, yp, cell)[0]
    return out


def bootstrap_cells(values, y, cell, n_boot, rng):
    """以 cell 为重抽单元的 hierarchical bootstrap（红线 1：独立单位是 cell 不是 episode）。"""
    cells = np.unique(cell)
    idxs = {c: np.where(cell == c)[0] for c in cells}
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(cells, size=len(cells), replace=True)
        vs, ys, cs = [], [], []
        for j, c in enumerate(pick):
            i = idxs[c]
            vs.append(values[i])
            ys.append(y[i])
            cs.append(np.full(len(i), j))
        out[b] = stratified_auc(np.concatenate(vs), np.concatenate(ys), np.concatenate(cs))[0]
    return out


def jdump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
