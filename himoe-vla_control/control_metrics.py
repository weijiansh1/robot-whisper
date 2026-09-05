"""Unified coordinate system for the control-theory experiments (E0-E8).

Implements the shared math from EXPERIMENT_PROGRAM.md exactly once so every
experiment uses identical definitions:

  sqrt embedding      psi(p) = sqrt(p);  d_H(p,q) = ||sqrt p - sqrt q||_2 / sqrt2
  normalized flow time s_f = f/(F-1)
  route velocity      v_f = (Psi_{f+1} - Psi_f) / ds
  route acceleration  a_f = (v_{f+1} - v_f) / ds
  V_late, A_route     time-normalized so F in {5,10,20} does not rescale them

Conventions
-----------
Routing tensors follow the capture layout  [F, L, U, E]  for a single query
(flow steps x HB layers x tokens x experts), float32/float16 probabilities on
the last axis.  All functions renormalize defensively (fp16 storage).
"""
from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)


def _as_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum(axis=-1, keepdims=True)
    s[s == 0] = 1.0
    return p / s


def sqrt_embed(p: np.ndarray) -> np.ndarray:
    """psi(p) = sqrt(p) with defensive renormalization."""
    return np.sqrt(_as_prob(p))


def hellinger(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d_H over the last axis; broadcasting over leading axes. In [0, 1]."""
    return np.linalg.norm(sqrt_embed(p) - sqrt_embed(q), axis=-1) / SQRT2


def flatten_query(routes_flue: np.ndarray) -> np.ndarray:
    """[F, L, U, E] probabilities -> Psi [F, L*U*E] sqrt-embedded vectors."""
    r = np.asarray(routes_flue)
    if r.ndim != 4:
        raise ValueError(f"expected [F,L,U,E], got {r.shape}")
    F = r.shape[0]
    return sqrt_embed(r).reshape(F, -1)


def route_distance(psi_a: np.ndarray, psi_b: np.ndarray) -> np.ndarray:
    """Aggregate routing distance between two flattened Psi paths [F, D].

    euclidean / sqrt(2) / sqrt(n_entries), n_entries = L*U*E.  This equals
    RMS-over-gates of the per-gate Hellinger divided by sqrt(E), i.e. it lives
    in [0, 1/sqrt(E)] (~0.177 for E=32), NOT in [0, 1].  Magnitudes are
    therefore not raw Hellinger values; the normalization is constant across
    gate subsets so all within-experiment comparisons and ratios (G_peak,
    funnel, D_seed shape) are unaffected.
    """
    a, b = np.asarray(psi_a), np.asarray(psi_b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    n_gates = a.shape[-1]  # L*U*E entries; each gate contributes E entries
    return np.linalg.norm(a - b, axis=-1) / SQRT2 / np.sqrt(n_gates)


def velocity(psi: np.ndarray) -> np.ndarray:
    """v_f = (Psi_{f+1}-Psi_f)/ds for Psi [F, D] -> [F-1, D]."""
    F = psi.shape[0]
    if F < 2:
        raise ValueError("need >=2 flow steps")
    ds = 1.0 / (F - 1)
    return np.diff(psi, axis=0) / ds


def acceleration(psi: np.ndarray) -> np.ndarray:
    """a_f = (v_{f+1}-v_f)/ds -> [F-2, D]."""
    F = psi.shape[0]
    if F < 3:
        raise ValueError("need >=3 flow steps")
    ds = 1.0 / (F - 1)
    return np.diff(velocity(psi), axis=0) / ds


def v_late(psi: np.ndarray, late_frac: float = 0.5) -> float:
    """V^late = sum_{f in late} ||v_f|| * ds  (time-normalized path length of
    the late-flow segment; late = last `late_frac` of normalized flow time)."""
    v = velocity(psi)
    F = psi.shape[0]
    ds = 1.0 / (F - 1)
    start = int(np.ceil((1.0 - late_frac) * (F - 1)))
    return float(np.linalg.norm(v[start:], axis=-1).sum() * ds)


def a_route(psi: np.ndarray, eps: float = 1e-9) -> float:
    """A^route = (sum ||a_f|| ds) / (sum ||v_f|| ds + eps): curvature per unit
    path length; invariant to uniform time re-discretization in the continuum
    limit."""
    v = velocity(psi)
    a = acceleration(psi)
    F = psi.shape[0]
    ds = 1.0 / (F - 1)
    num = float(np.linalg.norm(a, axis=-1).sum() * ds)
    den = float(np.linalg.norm(v, axis=-1).sum() * ds) + eps
    return num / den


def resample_path(psi: np.ndarray, n_points: int) -> np.ndarray:
    """Linear interpolation of Psi [F, D] onto n_points uniform s in [0,1]
    (for comparing F=5/10/20 paths on one grid; artifact check E5/H8)."""
    F = psi.shape[0]
    s_old = np.linspace(0.0, 1.0, F)
    s_new = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, psi.shape[1]), dtype=np.float64)
    for d in range(psi.shape[1]):
        out[:, d] = np.interp(s_new, s_old, psi[:, d])
    return out


# ---------------------------------------------------------------- E1 metrics

def perturbation_response(psi_base: np.ndarray, psi_pert: np.ndarray,
                          delta_norm: float, eps: float = 1e-12) -> dict:
    """E1-A metrics from one (xi, xi+dxi) pair of routing paths.

    Returns per-flow distance D_f, gains G_f = D_f/||dxi||, G_peak,
    G_terminal, funnel index."""
    D = route_distance(psi_base, psi_pert)          # [F]
    G = D / max(delta_norm, eps)
    g_peak = float(G.max())
    g_term = float(G[-1])
    return {
        "D_f": D,
        "G_f": G,
        "G_peak": g_peak,
        "G_terminal": g_term,
        "funnel": (g_peak - g_term) / (g_peak + eps),
        "argmax_f": int(G.argmax()),
    }


def finite_time_contraction(D_f: np.ndarray, f_start: int = 1,
                            eta: float = 1e-9) -> float:
    """lambda over [f_start, F-1]: mean log growth rate of the perturbation
    distance.  <0 decay, ~0 persistence, >0 amplification. (E1-B main stat;
    also usable descriptively on E1-A distances.)"""
    D = np.asarray(D_f, dtype=np.float64)
    F = len(D)
    if F - 1 <= f_start:
        raise ValueError("not enough steps")
    return float(np.log((D[-1] + eta) / (D[f_start] + eta)) / (F - 1 - f_start))


# ---------------------------------------------------------------- E2 metrics

def seed_dispersion(psis: list[np.ndarray]) -> np.ndarray:
    """D_seed(f): mean pairwise route_distance across N seed paths [F,D]."""
    N = len(psis)
    if N < 2:
        raise ValueError("need >=2 seeds")
    F = psis[0].shape[0]
    acc = np.zeros(F)
    cnt = 0
    for i in range(N):
        for j in range(i + 1, N):
            acc += route_distance(psis[i], psis[j])
            cnt += 1
    return acc / cnt


def fanout_funnel(d_seed: np.ndarray, eps: float = 1e-12,
                  degenerate_tol: float = 1e-9) -> dict:
    """FanOut, TerminalDiversity, FunnelIndex from a D_seed(f) curve.

    A curve that never disperses (FanOut ~ 0) has no funnel to speak of; it is
    flagged `degenerate` and FunnelIndex is NaN rather than the 0/eps -> 1.0
    that would masquerade as a perfect funnel."""
    fan = float(d_seed.max())
    term = float(d_seed[-1])
    degenerate = fan <= degenerate_tol
    return {
        "fan_out": fan,
        "terminal_diversity": term,
        "funnel_index": float("nan") if degenerate else 1.0 - term / (fan + eps),
        "degenerate": degenerate,
        "argmax_f": int(np.asarray(d_seed).argmax()),
    }


# ---------------------------------------------------------------- E3 metrics

def gate_matrix_stats(G_ue: np.ndarray) -> dict:
    """E3 four indicators for one gate matrix [U tokens, E experts]."""
    P = _as_prob(G_ue)
    U = P.shape[0]
    # row entropy
    h_row = float(-(P * np.log(np.clip(P, 1e-12, None))).sum(-1).mean())
    # soft consensus: 1 - mean pairwise Hellinger
    dsum, cnt = 0.0, 0
    for i in range(U):
        for j in range(i + 1, U):
            dsum += float(hellinger(P[i], P[j]))
            cnt += 1
    c_soft = 1.0 - dsum / max(cnt, 1)
    # effective / stable rank of the raw probability matrix
    s = np.linalg.svd(P, compute_uv=False)
    s = s[s > s.max() * 1e-12]
    pi = s / s.sum()
    r_eff = float(np.exp(-(pi * np.log(pi)).sum()))
    r_stable = float((s ** 2).sum() / (s.max() ** 2))
    return {"H_row": h_row, "C_soft": c_soft, "r_eff": r_eff,
            "r_stable": r_stable}


# ---------------------------------------------------------------- statistics

def trunk_bootstrap(values_by_trunk: dict, stat=np.mean, n_boot: int = 10000,
                    seed: int = 0) -> dict:
    """Hierarchical bootstrap: resample TRUNKS (not seeds) with replacement;
    within-trunk values enter via their trunk mean.  Returns point estimate
    and percentile CI."""
    rng = np.random.default_rng(seed)
    trunk_means = np.array([np.mean(v) for v in values_by_trunk.values()],
                           dtype=np.float64)
    n = len(trunk_means)
    if n < 2:
        raise ValueError("need >=2 trunks")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = stat(trunk_means[rng.integers(0, n, n)])
    return {"estimate": float(stat(trunk_means)),
            "ci_lo": float(np.percentile(boots, 2.5)),
            "ci_hi": float(np.percentile(boots, 97.5)),
            "n_trunks": n}
