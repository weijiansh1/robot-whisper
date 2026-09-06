#!/usr/bin/env python3
"""Electrical-network quantities on the HiMoE-VLA action-token routing graph.

Construction (fixed before any external evaluation)
---------------------------------------------------
At the final flow step the router emits, per HB layer, an 11 x 32 matrix of
expert probabilities.  Writing R_t = sqrt(p_t) (unit-norm, non-negative), the
Gram matrix G = R R^T is the Bhattacharyya / fidelity kernel over tokens.
Token 0 is the state token; tokens 1..10 are the action tokens of the chunk in
temporal order.

    conditional  C = G[1:,1:] - G[0,1:] (x) G[0,1:]

is the Schur complement of G with respect to the state token, hence PSD.  C is
the Gram matrix of the residual routing directions r_i = R_i - <R_i,R_0> R_0.

Circuit reading
    conductance   w_ij = max(C_ij, 0)        (i != j)
    ground leak   g_i  = 1 - C_ii = G_{0i}^2 (state-explained routing mass)

Two facts about this reading are stated and checked, not assumed:

*  Taking the Schur complement w.r.t. a node is *algebraically* the same
   operation as Kron reduction of a resistive network.  It is only literally a
   Kron reduction if the parent matrix is a Laplacian, which G is not.  A
   direct reading of C as a grounded Laplacian would need C to be diagonally
   dominant (leak_i = C_ii - sum_j C_ij >= 0); the audit reports how badly
   that fails, and it fails badly, which is why the leak is taken from the
   unit-norm budget instead.
*  A conductance must be non-negative.  C_ij can in principle be negative
   (G_ij >= G_0i G_0j is not implied by anything); the audit reports the
   observed negative-entry count and the discarded |mass| fraction.

Foster's theorem (sum_{i<j} w_ij R_ij = n - 1 = 9 on a connected network) is
used purely as a numerical identity check on the Laplacian / pseudo-inverse
code path.  It holds for *any* non-negative weights, so passing it validates
the implementation, not the modelling choice.
"""

from __future__ import annotations

import numpy as np

N_TOKENS = 10
LAYER_NAMES = ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")

FEATURE_NAMES = (
    # --- Laplacian spectrum -------------------------------------------------
    "vol",                    # total conductance sum_{i<j} w_ij            [scale]
    "fiedler",                # lambda_2 of L                                [scale]
    "lam_max",                # lambda_n of L                                [scale]
    "gap_ratio",              # lambda_2 / lambda_n                          [free]
    "spectral_erank",         # exp(H(lambda_2..n)) / (n-1)                  [free]
    "norm_fiedler",           # lambda_2 of normalised Laplacian, in [0,2]   [free]
    "n_near_zero",            # #{lambda_k <= 1e-9 * lambda_n} (components)
    # --- resistance / Kirchhoff --------------------------------------------
    "log_kirchhoff",          # log sum_{i<j} R_ij                           [scale]
    "kirchhoff_efficiency",   # n(n-1)^2 / (2 Kf vol) in (0,1], =1 iff clique[free]
    "log_spanning_trees",     # sum_{k>=2} log lambda_k - log n              [scale]
    "res_mean",               # mean_{i<j} R_ij                              [scale]
    "res_cv",                 # std/mean of R_ij                             [free]
    "res_end2end",            # R(token1, token10)                           [scale]
    "res_chain_series",       # sum_i R(i,i+1)                               [scale]
    "res_shortcut",           # res_chain_series / res_end2end  >= 1         [free]
    "res_adjacent_mean",      # mean_i R(i,i+1)                              [scale]
    # --- grounded (Thevenin) network, ground = state token ------------------
    "th_mean",                # mean_i (L + diag(g))^{-1}_ii                 [scale]
    "th_max",
    "th_min",
    "th_cv",                  # std/mean over tokens                         [free]
    "th_first",               # token 1 (the action executed next)
    "th_last",                # token 10
    "th_slope",               # OLS slope of R_th,i over i = 0..9
    "ground_leak_mean",       # mean_i g_i = mean state-explained mass       [free]
    "ground_leak_cv",
    # --- robustness variant: leak = C_ii instead of 1 - C_ii ----------------
    "thalt_mean",
    "thalt_cv",
    # --- validation ---------------------------------------------------------
    "foster_residual",        # |sum_{i<j} w_ij R_ij - 9|
    "neg_count",              # # negative off-diagonal entries of C (of 90)
    "neg_mass",               # |negative mass| / total |off-diagonal mass|
    "dd_deficit",             # mean_i max(0, sum_j w_ij - C_ii) / mean C_ii
)
N_FEATURES = len(FEATURE_NAMES)

_OFF = ~np.eye(N_TOKENS, dtype=bool)
_IU = np.triu_indices(N_TOKENS, k=1)
_POS = np.arange(N_TOKENS, dtype=np.float64)
_POS_C = _POS - _POS.mean()
_POS_SS = float((_POS_C ** 2).sum())


def conditional_from_raw(raw_final: np.ndarray) -> np.ndarray:
    """raw_final: [..., 11, 32] router probabilities at the final flow step."""
    p = np.clip(np.asarray(raw_final, dtype=np.float64), 0.0, None)
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
    root = np.sqrt(p)
    gram = root @ np.swapaxes(root, -1, -2)
    g0 = gram[..., 0, 1:]
    return gram[..., 1:, 1:] - g0[..., :, None] * g0[..., None, :]


def circuit_features(cond: np.ndarray) -> np.ndarray:
    """cond: [..., 10, 10] PSD conditional matrices -> [..., N_FEATURES]."""
    cond = np.asarray(cond, dtype=np.float64)
    lead = cond.shape[:-2]
    n = N_TOKENS

    diag = np.diagonal(cond, axis1=-2, axis2=-1).copy()
    off = np.where(_OFF, cond, 0.0)

    neg = off < 0.0
    abs_off = np.abs(off)
    total_abs = abs_off.sum((-1, -2))
    neg_count = neg.sum((-1, -2)).astype(np.float64)
    neg_mass = np.where(total_abs > 0, np.abs(np.where(neg, off, 0.0)).sum((-1, -2)) / np.maximum(total_abs, 1e-300), 0.0)

    w = np.clip(off, 0.0, None)
    deg = w.sum(-1)
    vol = deg.sum(-1) / 2.0
    eye = np.eye(n)
    lap = deg[..., :, None] * eye - w

    lam, vec = np.linalg.eigh(lap)
    lam = np.clip(lam, 0.0, None)
    lam_pos = lam[..., 1:]
    lam_max = lam[..., -1]
    fiedler = lam[..., 1]
    tiny = 1e-10 * np.maximum(lam_max, 1e-300)
    n_near_zero = (lam <= tiny[..., None]).sum(-1).astype(np.float64)

    gap_ratio = fiedler / np.maximum(lam_max, 1e-300)
    q = lam_pos / np.maximum(lam_pos.sum(-1, keepdims=True), 1e-300)
    ent = -(q * np.log(np.maximum(q, 1e-300))).sum(-1)
    spectral_erank = np.exp(ent) / (n - 1)
    log_spanning = np.log(np.maximum(lam_pos, 1e-300)).sum(-1) - np.log(n)

    # Moore-Penrose pseudo-inverse with rank truncated at the numerical null
    # space.  When the positive-part graph is disconnected the null space has
    # dimension k > 1; L^+ is then the direct sum of the component
    # pseudo-inverses, so within-component resistances stay exact while
    # cross-component pairs (formally infinite) are silently replaced by
    # L^+_ii + L^+_jj.  Foster's sum then equals n - k instead of n - 1, so
    # `foster_residual` is exactly k - 1 and flags every such query.
    keep = lam > 1e-10 * np.maximum(lam_max, 1e-300)[..., None]
    inv_lam = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    lplus = (vec * inv_lam[..., None, :]) @ np.swapaxes(vec, -1, -2)
    lpd = np.diagonal(lplus, axis1=-2, axis2=-1)
    res = lpd[..., :, None] + lpd[..., None, :] - 2.0 * lplus
    res = np.clip(res, 0.0, None)

    kf = res[..., _IU[0], _IU[1]].sum(-1)
    foster = (w * res).sum((-1, -2)) / 2.0
    foster_residual = np.abs(foster - (n - 1))

    res_u = res[..., _IU[0], _IU[1]]
    res_mean = res_u.mean(-1)
    res_cv = res_u.std(-1) / np.maximum(res_mean, 1e-300)
    res_end2end = res[..., 0, n - 1]
    adj = res[..., np.arange(n - 1), np.arange(1, n)]
    res_chain = adj.sum(-1)
    res_adj_mean = adj.mean(-1)
    res_shortcut = res_chain / np.maximum(res_end2end, 1e-300)
    kirchhoff_eff = (n * (n - 1) ** 2) / np.maximum(2.0 * kf * vol, 1e-300)

    # normalised Laplacian spectrum (scale free)
    inv_sqrt_deg = np.where(deg > 0, 1.0 / np.sqrt(np.where(deg > 0, deg, 1.0)), 0.0)
    lnorm = eye - w * inv_sqrt_deg[..., :, None] * inv_sqrt_deg[..., None, :]
    lam_n = np.clip(np.linalg.eigvalsh(lnorm), 0.0, None)
    norm_fiedler = lam_n[..., 1]

    # grounded network: ground = state token, leak g_i = 1 - C_ii
    leak = np.clip(1.0 - diag, 0.0, None)
    lg = lap + leak[..., :, None] * eye
    m = np.linalg.inv(lg)
    th = np.diagonal(m, axis1=-2, axis2=-1)
    th_mean = th.mean(-1)
    th_cv = th.std(-1) / np.maximum(th_mean, 1e-300)
    th_slope = ((th - th_mean[..., None]) * _POS_C).sum(-1) / _POS_SS

    leak_mean = leak.mean(-1)
    leak_cv = leak.std(-1) / np.maximum(leak_mean, 1e-300)

    lg2 = lap + np.clip(diag, 1e-12, None)[..., :, None] * eye
    th2 = np.diagonal(np.linalg.inv(lg2), axis1=-2, axis2=-1)
    th2_mean = th2.mean(-1)
    th2_cv = th2.std(-1) / np.maximum(th2_mean, 1e-300)

    dd_deficit = np.clip(deg - diag, 0.0, None).mean(-1) / np.maximum(diag.mean(-1), 1e-300)

    out = np.empty(lead + (N_FEATURES,), dtype=np.float64)
    for idx, value in enumerate((
        vol, fiedler, lam_max, gap_ratio, spectral_erank, norm_fiedler, n_near_zero,
        np.log(np.maximum(kf, 1e-300)), kirchhoff_eff, log_spanning,
        res_mean, res_cv, res_end2end, res_chain, res_shortcut, res_adj_mean,
        th_mean, th.max(-1), th.min(-1), th_cv, th[..., 0], th[..., -1], th_slope,
        leak_mean, leak_cv, th2_mean, th2_cv,
        foster_residual, neg_count, neg_mass, dd_deficit,
    )):
        out[..., idx] = value
    assert idx == N_FEATURES - 1, (idx, N_FEATURES)
    return out
