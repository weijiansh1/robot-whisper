"""MoE-only 表型特征 z_q（冻结定义，见 ../PROTOCOL.md §2）。

P[l,f,u,e]: 存储 HB 层 8 × flow 10 × suffix token 11 (0=state,1..10=action) × expert 32。
深层 = 存储 slice(4,8) = 全局 HB 12–15。聚合一律 V2：逐 (layer,token) 格先算距离再平均。
"""

from __future__ import annotations

import numpy as np

DEEP = slice(4, 8)
ACT = slice(1, 11)
STATE = 0
D9 = 9
EPS = 1e-12
REC_KS = tuple(range(1, 9))
W8 = 8
CHUNK = 1024

SCALARS = [
    "late_flow_volatility", "route_acceleration", "gate_entropy", "top12_margin",
    "token_consensus", "token_dispersion", "layer_disagreement",
    "state_action_gap", "flow_com", "flow_total",
]


def _hell(p, q):
    bc = (np.sqrt(np.maximum(p, 0.0)) * np.sqrt(np.maximum(q, 0.0))).sum(-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def _hell_sqrt(rp, rq):
    """Hellinger from pre-sqrt coordinates."""
    bc = (rp * rq).sum(-1)
    return np.sqrt(np.clip(1.0 - bc, 0.0, 1.0))


def _wj_sim(p, q):
    return np.minimum(p, q).sum(-1) / np.maximum(np.maximum(p, q).sum(-1), EPS)


def per_row(P16):
    """P16: (n, 8, 10, 11, 32) float16/32 → (dict of (n,) f32, profile (n,9), reps (n,4,10,32) f32)."""
    n = len(P16)
    out = {k: np.empty(n, np.float32) for k in SCALARS}
    profile = np.empty((n, 9), np.float32)
    reps = np.empty((n, 4, 10, 32), np.float32)
    iu_t = np.triu_indices(10, 1)
    iu_l = np.triu_indices(4, 1)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        A = P16[a:b, DEEP, :, ACT, :].astype(np.float32)        # (m,4,10,10,32)
        m = len(A)
        # V: late-flow weighted-Jaccard distance, flows 6..9 的 3 个相邻对
        lf = A[:, :, 6:10]
        out["late_flow_volatility"][a:b] = (
            1.0 - _wj_sim(lf[:, :, :-1], lf[:, :, 1:])).mean((1, 2, 3))
        # A: sqrt 坐标二阶差分范数 /√2（trap 同式）
        r = np.sqrt(A)
        acc = r[:, :, 2:] - 2.0 * r[:, :, 1:-1] + r[:, :, :-2]
        out["route_acceleration"][a:b] = (
            np.linalg.norm(acc, axis=-1).mean((1, 2, 3)) / np.sqrt(2.0))
        # flow 变化 profile 与质心
        d_f = _hell(A[:, :, :-1], A[:, :, 1:]).mean((1, 3))      # (m,9)
        profile[a:b] = d_f
        tot = d_f.sum(1)
        out["flow_total"][a:b] = tot
        out["flow_com"][a:b] = ((np.arange(9, dtype=np.float32) + 0.5) * d_f).sum(1) / np.maximum(tot, EPS)
        # d9 切片
        A9 = A[:, :, D9]                                          # (m,4,10,32)
        reps[a:b] = np.sqrt(A9)
        out["gate_entropy"][a:b] = (
            -(A9 * np.log(np.maximum(A9, EPS))).sum(-1)).mean((1, 2))
        srt = np.sort(A9, axis=-1)
        out["top12_margin"][a:b] = (srt[..., -1] - srt[..., -2]).mean((1, 2))
        # C / D^tok：45 个 token 对
        p1 = A9[:, :, :, None, :]
        p2 = A9[:, :, None, :, :]
        wj = _wj_sim(p1, p2)                                      # (m,4,10,10)
        out["token_consensus"][a:b] = wj[:, :, iu_t[0], iu_t[1]].mean((1, 2))
        hd = _hell(p1, p2)
        out["token_dispersion"][a:b] = hd[:, :, iu_t[0], iu_t[1]].mean((1, 2))
        # D^layer：6 个层对
        hl = _hell(A9[:, :, None], A9[:, None])                   # (m,4,4,10)
        out["layer_disagreement"][a:b] = hl[:, iu_l[0], iu_l[1]].mean((1, 2))
        # state-action gap
        S9 = P16[a:b, DEEP, D9, STATE].astype(np.float32)         # (m,4,32)
        out["state_action_gap"][a:b] = _hell(S9[:, :, None, :], A9).mean((1, 2))
    return out, profile, reps


def episode_recurrence(reps, episode_id, control_step):
    """mob_k (N,8) 与 mob1_w8 (N,)。reps: (N,4,10,32) sqrt 坐标。要求行可按集排序。"""
    n = len(reps)
    mob = np.full((n, len(REC_KS)), np.nan, np.float32)
    w8 = np.full(n, np.nan, np.float32)
    for e in np.unique(episode_id):
        idx = np.where(episode_id == e)[0]
        idx = idx[np.argsort(control_step[idx])]
        R = reps[idx]
        for j, k in enumerate(REC_KS):
            if len(idx) <= k:
                continue
            d = _hell_sqrt(R[k:], R[:-k]).mean((1, 2))            # V2
            mob[idx[k:], j] = d
        m1 = mob[idx, 0]
        for i in range(W8, len(idx)):
            win = m1[i - W8 + 1:i + 1]
            if np.isfinite(win).all():
                w8[idx[i]] = win.mean()
    return mob, w8
