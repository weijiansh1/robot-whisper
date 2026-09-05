"""C3 — 命运在什么时候变得可读？内部分叉与身体分叉哪个先发生？

三条曲线，全部在 cell 内（同初始状态的 K 个同胞）逐 q 计算：

  D_phys(q)  同胞之间的物理分离（eef ⊕ 物体位置，米）。**按构造 D_phys(0)=0。**
  D_route(q) 同胞之间的 routing 表征分离（reps 1280 维），同时给出它相对于
             *跨初始状态* 分离尺度的比值 —— 才知道 0.03 算大还是小。
  AUC(q)     组内 AUC：q 处的信号能否分出"这个同胞最后会不会失败"。

读法：
  · q=0 时 D_phys 严格为 0 而 D_route>0 ⇒ 计算已经分叉、身体还没有。
    这一格上的 AUC 就是 C2 的 β_W，此处只是复算。
  · 若 AUC(q) 一直贴着 0.5，直到 D_phys 明显上升之后才起来 ⇒ 信号读的是**已经发生的
    状态分离**，即 state sensor；若 AUC 在 D_phys 还接近 0 时就抬头 ⇒ 存在真正的
    "噪声后果"前瞻信息。

删失控制：每个 cell 只用 q < min_j n_queries(j) 的**无删失窗口**——窗口内 K 个同胞
全部还活着。集长本身几乎是完美的成败预测器（analysis_mdp：oracle length AUC≈0.99），
不设这道闸，AUC(q) 在大 q 处会被存活偏倚顶起来。
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import cload as C

OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
HUB = pathlib.Path("/home/jovyan/work/himoe-vla/VLA_MUI_HUB")
RUNS = {
    "main16x32": (HUB / "cache/HiMoE-VLA", "right-16x32"),
    "grid50x8": (HUB / "cache_new/HiMoE-VLA", "right-50x8-20260903"),
}
MAX_Q = 60


def phys_arrays(run: pathlib.Path, e_id: int, slots):
    with np.load(run / "client" / f"episode_{e_id:02d}.npz") as d:
        state = np.asarray(d["state"], np.float64)
        sim = np.asarray(d["sim_state"], np.float64)
    eef = state[:, :3]
    objs = np.concatenate([sim[:, lo:lo + 3] for lo in slots], axis=1) if slots else np.zeros((len(eef), 0))
    return np.concatenate([eef, objs], axis=1)


def object_slots(run: pathlib.Path):
    lay = json.load(open(run / "client" / "sim_layout.json"))
    return [j["state_lo"] for j in lay["joints"]
            if not j["is_robot"] and j["state_hi"] - j["state_lo"] == 7]


def mean_pairwise(X):
    """X:(n,d) -> 平均成对欧氏距离。"""
    if len(X) < 2:
        return np.nan
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    iu = np.triu_indices(len(X), 1)
    return float(d[iu].mean())


def run_task(corpus, suite, task, feat_names):
    root, run_id = RUNS[corpus]
    run = root / suite / task / run_id
    t = C.load_task(corpus, suite, task, with_reps=True)
    eps, Y, nq = C.outcomes(t)
    _, sc, _ = C.episode_cells(t)
    y = Y["fail"]
    epidx = {int(e): i for i, e in enumerate(eps)}
    slots = object_slots(run)

    # 逐 episode 的物理轨迹（对齐 episode_index == npz 编号）
    phys = {}
    for e in eps:
        try:
            phys[e] = phys_arrays(run, e, slots)
        except FileNotFoundError:
            phys[e] = None

    per_q = {}
    for s in np.unique(sc):
        m = sc == s
        if not (0 < y[m].sum() < m.sum()):
            continue  # 只用敏感格
        cell_eps = [eps[i] for i in np.where(m)[0]]
        cell_y = y[m]
        qmax = int(min(nq[m])) - 1  # 无删失窗口
        for q in range(min(qmax + 1, MAX_Q)):
            rows, P, ok = [], [], []
            for e in cell_eps:
                sel = (t.ep == e) & (t.q == q)
                if sel.sum() != 1:
                    ok.append(False)
                    continue
                rows.append(np.where(sel)[0][0])
                pe = phys[e]
                P.append(pe[q] if (pe is not None and q < len(pe)) else None)
                ok.append(True)
            if len(rows) != len(cell_eps):
                break
            rows = np.asarray(rows)
            d = per_q.setdefault(q, dict(feats={f: [] for f in feat_names},
                                         y=[], cell=[], d_route=[], d_phys=[], reps_mean=[]))
            for f in feat_names:
                d["feats"][f].append(t.feats[f][rows])
            d["y"].append(cell_y)
            d["cell"].append(np.full(len(rows), f"{task}#{s}"))
            R = np.asarray(t.reps[rows], dtype=np.float32)
            d["d_route"].append(mean_pairwise(R))
            d["reps_mean"].append(R.mean(0))
            if all(p is not None for p in P):
                d["d_phys"].append(mean_pairwise(np.asarray(P)))
    return per_q


def main(corpus="main16x32"):
    feat_names = ["late_flow_volatility", "route_acceleration", "gate_entropy",
                  "token_consensus", "token_dispersion", "mob1_w8"]
    agg = {}
    for suite, task in C.iter_tasks(corpus):
        pq = run_task(corpus, suite, task, feat_names)
        for q, d in pq.items():
            a = agg.setdefault(q, dict(feats={f: [] for f in feat_names},
                                       y=[], cell=[], d_route=[], d_phys=[], reps_mean=[]))
            for f in feat_names:
                a["feats"][f] += d["feats"][f]
            a["y"] += d["y"]
            a["cell"] += d["cell"]
            a["d_route"] += d["d_route"]
            a["d_phys"] += d["d_phys"]
            a["reps_mean"] += d["reps_mean"]

    rows = []
    for q in sorted(agg):
        a = agg[q]
        y = np.concatenate(a["y"])
        cell = np.concatenate(a["cell"])
        n_cells = len(set(cell.tolist()))
        if n_cells < 5:
            continue
        rec = dict(q=q, n_cells=n_cells, n_eps=int(len(y)),
                   d_route=float(np.mean(a["d_route"])),
                   d_phys=float(np.mean(a["d_phys"])) if a["d_phys"] else np.nan,
                   # 跨 cell 的 routing 分离尺度：cell 均值表征之间的平均成对距离
                   d_route_between=mean_pairwise(np.asarray(a["reps_mean"])))
        rec["route_sib_over_between"] = rec["d_route"] / (rec["d_route_between"] + 1e-9)
        for f in feat_names:
            v = np.concatenate(a["feats"][f])
            rec[f"auc_{f}"] = C.stratified_auc(v, y, cell)[0]
        rows.append(rec)

    OUT.mkdir(exist_ok=True)
    C.jdump(rows, OUT / f"c3_fate_curve_{corpus}.json")
    hdr = (f"{'q':>3s} {'cells':>5s} {'eps':>5s} {'D_phys(m)':>10s} {'D_route':>8s} "
           f"{'sib/betw':>8s} " + " ".join(f"{f[:11]:>11s}" for f in feat_names))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['q']:3d} {r['n_cells']:5d} {r['n_eps']:5d} {r['d_phys']:10.5f} "
              f"{r['d_route']:8.4f} {r['route_sib_over_between']:8.4f} "
              + " ".join(f"{r[f'auc_{f}']:11.4f}" for f in feat_names))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main16x32")
