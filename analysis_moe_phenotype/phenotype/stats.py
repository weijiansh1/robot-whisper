"""共享统计（PROTOCOL §4）。组内配对 AUC / episode 级置换 maxT / 组级 sign-flip maxT / 秩残差。"""

from __future__ import annotations

import numpy as np


def paired_auc(x, y, group):
    """组内成功–失败配对 AUC；跨组不池化。返回 (auc, n_pairs)。x 高→y=1 侧为 auc>0.5。"""
    num = den = 0.0
    ok = np.isfinite(x)
    for g in np.unique(group):
        m = (group == g) & ok
        pos, neg = x[m & (y == 1)], x[m & (y == 0)]
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum()
        den += d.size
    return (num / den if den else np.nan), int(den)


def joint_maxt(cells, ep_of_row, ep_label, ep_group, nperm, rng):
    """episode 级组内置换 + maxT。

    cells: list[(name, values(N,), row_mask(N,), group_of_row(N,))]
    ep_of_row: (N,) episode key；ep_label/ep_group: dict key->0/1, key->组名。
    返回 (obs: name->(auc, pairs), p: name->maxT p)。
    """
    eps = list(ep_label)
    y_ep = np.array([ep_label[e] for e in eps])
    g_ep = np.array([ep_group[e] for e in eps])
    idx_of = {e: i for i, e in enumerate(eps)}
    row_ep = np.array([idx_of[e] for e in ep_of_row])

    obs = {}
    for name, vals, mask, grp in cells:
        obs[name] = paired_auc(vals[mask], y_ep[row_ep][mask], grp[mask])
    dev = {k: abs(v[0] - 0.5) for k, v in obs.items() if np.isfinite(v[0])}
    maxdev = np.zeros(nperm)
    yp = y_ep.copy()
    for i in range(nperm):
        for g in np.unique(g_ep):
            m = g_ep == g
            yp[m] = rng.permutation(y_ep[m])
        yrow = yp[row_ep]
        d = 0.0
        for name, vals, mask, grp in cells:
            a, _ = paired_auc(vals[mask], yrow[mask], grp[mask])
            if np.isfinite(a):
                d = max(d, abs(a - 0.5))
        maxdev[i] = d
    p = {k: float((maxdev >= dev[k]).sum() + 1) / (nperm + 1) for k in dev}
    return obs, p


def groupwise_signflip_maxt(effects, nperm, rng):
    """effects: (n_groups, n_cells) 组级效应（NaN 允许）。

    观测统计 = 逐 cell 的组均值；置换 = 整组随机翻号；maxT 覆盖全部 cell。
    返回 (obs_mean (n_cells,), p (n_cells,))。
    """
    obs = np.nanmean(effects, 0)
    dev = np.abs(obs)
    mx = np.zeros(nperm)
    for i in range(nperm):
        s = rng.choice((-1.0, 1.0), size=len(effects))[:, None]
        mx[i] = np.nanmax(np.abs(np.nanmean(effects * s, 0)))
    p = np.array([(mx >= d).sum() + 1 for d in dev], float) / (nperm + 1)
    return obs, p


def rank_within(x, group):
    r = np.full(len(x), np.nan)
    ok = np.isfinite(x)
    for g in np.unique(group):
        m = (group == g) & ok
        if m.sum() < 2:
            continue
        r[m] = (np.argsort(np.argsort(x[m])) + 0.5) / m.sum()
    return r


def residualise(x, base, group):
    """对 base 的组内秩残差（新颖性检验用；base=mob1_w8）。"""
    rx, rb = rank_within(x, group), rank_within(base, group)
    out = np.full(len(x), np.nan)
    for g in np.unique(group):
        m = (group == g) & np.isfinite(rx) & np.isfinite(rb)
        if m.sum() < 3:
            continue
        A = np.c_[np.ones(m.sum()), rb[m]]
        coef, *_ = np.linalg.lstsq(A, rx[m], rcond=None)
        out[m] = rx[m] - A @ coef
    return out
