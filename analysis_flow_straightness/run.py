#!/usr/bin/env python3
"""一次推理内 flow 去噪是否走直线 / 猜测的动作是否不变。

数据：两批已有的完整 11 点 flow 轨迹捕获（无需 GPU、无需新 rollout）
  A = runs/flow-lead-cpu-t0s24                      goal ckpt / task 0 / init 24 / 44 query x 16 candidate
  B = runs/flow-stop-sweep-long-t08s0-k1-seeds6100  long ckpt / task 8 / init 0  / 181 query x 1 candidate

采样器是固定步长 Euler：x_{m+1} = x_m - dt * v_m, dt = 0.1, t_m = 1 - 0.1 m。
因此速度场可由相邻两点精确还原：v_m = (x_m - x_{m+1}) / dt。无需重跑模型。

两个问题分开测：
  Q1 直线性 —— 速度方向在 10 轮内变不变（曲率）。
  Q2 猜测是否不变 —— 每轮的终点外推 xhat(m) = x_m - t_m * v_m 变不变。
     直线 <=> xhat(m) 对所有 m 恒等于 x_10。二者是同一件事的两种读法。

只用前 7 个"活"维；dim 7-23 是 padding。所有相对量与 8/22 的
audit_flow_stop_gate.py 同口径（分母 = ||x_10||），便于对表。
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "himoe-route-capture" / "runs"
LIVE = 7
DT = 0.1

CORPORA = {
    "A_goal_t00": RUNS / "flow-lead-cpu-t0s24",
    "B_long_t08": RUNS / "flow-stop-sweep-long-t08s0-k1-seeds6100-6103",
}


def load(run: pathlib.Path):
    parts = [np.load(p) for p in sorted(glob.glob(str(run / "flow_traces_part*.npz")))]
    X = np.concatenate([p["x_traj"] for p in parts]).squeeze(2).astype(np.float64)
    qid = np.concatenate([p["query_id"] for p in parts])
    cid = np.concatenate([p["candidate_id"] for p in parts])
    ch = np.load(run / "candidate_chunks.npz")
    return X, qid, cid, ch["chunks"].astype(np.float64)


def affine_to_action(x10_live, chunks):
    """末点 -> 可执行动作的逐维仿射（反归一化）。返回 (a, b) 使 action = a*x + b。"""
    a = np.zeros(LIVE)
    b = np.zeros(LIVE)
    r = np.zeros(LIVE)
    xs = x10_live.reshape(-1, LIVE)
    ys = chunks.reshape(-1, LIVE)
    for d in range(LIVE):
        a[d], b[d] = np.polyfit(xs[:, d], ys[:, d], 1)
        r[d] = np.corrcoef(xs[:, d], ys[:, d])[0, 1]
    return a, b, r


def rowwise_norm(v):
    return np.linalg.norm(v.reshape(len(v), -1), axis=1)


def cosines(u, v):
    uf = u.reshape(len(u), -1)
    vf = v.reshape(len(v), -1)
    num = (uf * vf).sum(1)
    den = np.linalg.norm(uf, axis=1) * np.linalg.norm(vf, axis=1)
    return num / np.maximum(den, 1e-12)


def pct(a, q):
    return float(np.percentile(a, q))


def analyse(name, run):
    X, qid, cid, chunks = load(run)
    n = len(X)
    live = X[:, :, :, :LIVE]           # (n, 11, 10, 7)
    pad = X[:, :, :, LIVE:]
    a, b, r_fit = affine_to_action(X[:, 10, :, :LIVE], chunks)

    # 采样器自洽性：x_0 应是标准高斯；padding 维应几乎不动
    out = {
        "run": str(run.relative_to(ROOT)),
        "rows": int(n),
        "queries": int(len(np.unique(qid))),
        "candidates": int(len(np.unique(cid))),
        "sanity": {
            "x0_std_live": float(X[:, 0, :, :LIVE].std()),
            "x0_std_pad": float(X[:, 0, :, LIVE:].std()),
            "x10_std_pad": float(X[:, 10, :, LIVE:].std()),
            "pad_total_motion_over_live": float(
                rowwise_norm(pad[:, 0] - pad[:, 10]).mean()
                / rowwise_norm(live[:, 0] - live[:, 10]).mean()
            ),
            "affine_fit_r_min": float(r_fit.min()),
        },
    }

    # 速度场（精确还原）
    v = (live[:, :10] - live[:, 1:]) / DT          # (n, 10, 10, 7)
    t = 1.0 - DT * np.arange(10)                   # t_0 .. t_9
    chord = live[:, 0] - live[:, 10]
    chord_n = rowwise_norm(chord)
    final_n = rowwise_norm(live[:, 10])

    # ---- Q1 直线性 ----
    cos_cons = np.stack([cosines(v[:, m], v[:, m + 1]) for m in range(9)], 1)
    cos_chord = np.stack([cosines(v[:, m], chord) for m in range(10)], 1)
    speed = np.stack([rowwise_norm(v[:, m]) for m in range(10)], 1)
    path = DT * speed.sum(1)
    out["straightness"] = {
        "path_over_chord": {
            "median": float(np.median(path / chord_n)),
            "p10": pct(path / chord_n, 10),
            "p90": pct(path / chord_n, 90),
        },
        "cos_consecutive_median": [float(np.median(cos_cons[:, m])) for m in range(9)],
        "cos_to_chord_median": [float(np.median(cos_chord[:, m])) for m in range(10)],
        "speed_median": [float(np.median(speed[:, m])) for m in range(10)],
        "speed_last_over_first": float(np.median(speed[:, 9] / speed[:, 0])),
    }

    # ---- Q2 每轮的终点猜测 ----
    xhat = live[:, :10] - t[None, :, None, None] * v      # (n, 10, 10, 7)
    xhat = np.concatenate([xhat, live[:, 10:11]], 1)      # 补 m=10 => x_10
    rel_final = np.stack(
        [rowwise_norm(xhat[:, m] - live[:, 10]) / final_n for m in range(11)], 1
    )
    drift = np.stack(
        [rowwise_norm(xhat[:, m + 1] - xhat[:, m]) / final_n for m in range(10)], 1
    )
    # 裸潜变量对照（不外推）
    raw_rel = np.stack(
        [rowwise_norm(live[:, m] - live[:, 10]) / final_n for m in range(11)], 1
    )
    out["guess"] = {
        "rel_to_final_median": [float(np.median(rel_final[:, m])) for m in range(11)],
        "rel_to_final_p90": [pct(rel_final[:, m], 90) for m in range(11)],
        "consecutive_drift_median": [float(np.median(drift[:, m])) for m in range(10)],
        "raw_latent_rel_median": [float(np.median(raw_rel[:, m])) for m in range(11)],
        "first_guess_vs_total_travel": float(
            np.median(rowwise_norm(xhat[:, 0] - live[:, 10]) / chord_n)
        ),
    }

    # 候选间散布（换一颗噪声 seed 的动作差）——只有 K>1 的语料能算
    if len(np.unique(cid)) > 1:
        sp = []
        for q in np.unique(qid):
            idx = np.where(qid == q)[0]
            f = live[idx, 10]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    sp.append(
                        np.linalg.norm(f[i] - f[j]) / max(np.linalg.norm(f[j]), 1e-9)
                    )
        sp = np.array(sp)
        out["candidate_spread"] = {
            "median": float(np.median(sp)),
            "p10": pct(sp, 10),
            "p90": pct(sp, 90),
        }
        anchor = float(np.median(sp))
        med = out["guess"]["rel_to_final_median"]
        out["guess"]["rounds_to_enter_candidate_band"] = next(
            (m for m in range(11) if med[m] <= anchor), None
        )

    # ---- 真实动作单位 ----
    # xhat -> 动作空间；报告逐维中位绝对偏差（相对最终动作）
    act_hat = xhat * a[None, None, None, :] + b[None, None, None, :]
    act_fin = act_hat[:, 10:11]
    dev = np.abs(act_hat - act_fin)                    # (n, 11, 10, 7)
    out["action_units"] = {
        "dim_names": ["dx", "dy", "dz", "d_roll", "d_pitch", "d_yaw", "gripper"],
        "median_abs_dev_by_round": {
            f"m{m}": [float(np.median(dev[:, m, :, d])) for d in range(LIVE)]
            for m in (0, 1, 2, 3, 5, 7, 9)
        },
        "final_action_abs_median": [
            float(np.median(np.abs(chunks[:, :, d]))) for d in range(LIVE)
        ],
    }

    # 夹爪：离散决定何时定下来（用最终命令的符号做基准）
    g_hat = act_hat[:, :, :, 6]
    g_fin = g_hat[:, 10]
    same = (np.sign(g_hat) == np.sign(g_fin)[:, None, :])
    out["gripper"] = {
        "sign_agree_with_final_by_round": [float(same[:, m].mean()) for m in range(11)],
        "final_sign_pos_frac": float((g_fin > 0).mean()),
    }

    # ---- 逐 action-token 位置 ----
    per_tok = np.stack(
        [
            np.linalg.norm(xhat[:, :, k] - live[:, 10:11, k], axis=2)
            / np.maximum(np.linalg.norm(live[:, 10, k], axis=1)[:, None], 1e-9)
            for k in range(10)
        ],
        2,
    )  # (n, 11, 10)
    out["per_token"] = {
        "rel_to_final_median_m0": [float(np.median(per_tok[:, 0, k])) for k in range(10)],
        "rel_to_final_median_m2": [float(np.median(per_tok[:, 2, k])) for k in range(10)],
        "rel_to_final_median_m5": [float(np.median(per_tok[:, 5, k])) for k in range(10)],
    }
    return out


def main() -> int:
    res = {
        "question": "一次推理内 flow 去噪是否走直线 / 每轮的动作猜测是否不变",
        "sampler": {"n_denoise": 10, "dt": DT, "scheme": "fixed-step Euler",
                    "velocity_recovery": "v_m = (x_m - x_{m+1}) / dt  (精确, 非近似)"},
        "live_dims": LIVE,
        "corpora": {},
    }
    for name, run in CORPORA.items():
        print(f"[{name}] {run}", flush=True)
        res["corpora"][name] = analyse(name, run)
    (HERE / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
