"""Phase 1 (LABEL-BLIND) shared machinery.

Tightness definitions used throughout.  All variables are z-scored with the
development_main pooled mean/sd so that coefficients are comparable across
cohorts and so that "relative residual" is scale free.

  rel_resid(y | X)  = std(y - X b) / std(y),  b = OLS.  For a single predictor
                      this is sqrt(1 - r^2).  0 = exact relation, 1 = useless.
  lam_min(S)        = smallest eigenvalue of Corr(S).  sqrt(lam_min) is the
                      std of the tightest unit-norm combination of the
                      standardised members of S: the "conserved quantity"
                      form of the same question.  Reported as `combo_std`.

Nulls.  Both are surrogates with matched marginals; their cross-covariance is
computable in closed form, which is what the routines below do, and
`shuffle_check` verifies the closed form against real permutations.

  N_cell : permute episodes independently per variable inside each
           (task, chunk) cell.  Keeps every variable's task x chunk profile
           and its marginal; destroys all episode-specific coupling.
           Cov_null = Cov_between(task,chunk)      (exact in expectation)
  N_epi  : permute chunks independently per variable inside each episode
           (an episode-shuffle equivalent of phase randomisation: it keeps the
           per-episode marginal exactly, hence the episode mean and level).
           Cov_null = Cov_between(episode)         (exact in expectation)

A relation must be tighter than BOTH surrogates to count.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------
# moment machinery
# --------------------------------------------------------------------------
def group_means(X: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-group means [G, V] and group counts [G]."""
    G = int(g.max()) + 1
    cnt = np.bincount(g, minlength=G).astype(np.float64)
    s = np.empty((G, X.shape[1]))
    for v in range(X.shape[1]):
        s[:, v] = np.bincount(g, weights=X[:, v], minlength=G)
    m = np.zeros_like(s)
    nz = cnt > 0
    m[nz] = s[nz] / cnt[nz, None]
    return m, cnt


def cov_decomposition(Z: np.ndarray, g: np.ndarray) -> dict:
    """Total / between-group / within-group covariance of standardised Z."""
    n = Z.shape[0]
    grand = Z.mean(axis=0)
    Zc = Z - grand
    C_tot = (Zc.T @ Zc) / n
    m, cnt = group_means(Zc, g)
    nz = cnt > 0
    W = np.sqrt(cnt[nz])[:, None] * m[nz]
    C_bet = (W.T @ W) / n
    return {"tot": C_tot, "bet": C_bet, "wit": C_tot - C_bet}


def to_corr(C: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(C), EPS, None))
    R = C / d[:, None] / d[None, :]
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def null_corr(C_tot: np.ndarray, C_bet: np.ndarray) -> np.ndarray:
    """Surrogate correlation: cross terms lose the within-group part, variances
    are preserved because the surrogate is a permutation."""
    d = np.sqrt(np.clip(np.diag(C_tot), EPS, None))
    R = C_bet / d[:, None] / d[None, :]
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


# --------------------------------------------------------------------------
# searches over a correlation matrix
# --------------------------------------------------------------------------
def pair_table(R: np.ndarray) -> np.ndarray:
    """rel_resid for every ordered-free pair: sqrt(1 - r^2), [V, V]."""
    return np.sqrt(np.clip(1.0 - R ** 2, 0.0, 1.0))


def triple_best(R: np.ndarray, target: int, ban: np.ndarray | None = None,
                collinear_cut: float = 0.999) -> tuple[np.ndarray, np.ndarray]:
    """rel_resid of y=target regressed on every pair (j,k).  Closed form from
    the 3x3 correlation submatrix.  Returns [V, V] rel_resid (upper triangle
    meaningful) and the same-shape mask of admissible pairs."""
    V = R.shape[0]
    a = R[target]
    Rjk = R
    den = 1.0 - Rjk ** 2
    ok = (np.abs(Rjk) < collinear_cut)
    ok &= ~np.eye(V, dtype=bool)
    ok[target, :] = False
    ok[:, target] = False
    if ban is not None:
        ok &= ~ban
        ok &= ~ban.T
    num = a[:, None] ** 2 + a[None, :] ** 2 - 2.0 * a[:, None] * a[None, :] * Rjk
    with np.errstate(divide="ignore", invalid="ignore"):
        R2 = np.where(ok, num / np.where(den > EPS, den, np.nan), np.nan)
    R2 = np.clip(R2, 0.0, 1.0)
    return np.sqrt(1.0 - R2), ok


def fit_std_coefs(R: np.ndarray, target: int, preds: list[int]) -> np.ndarray:
    """Standardised OLS coefficients from a correlation matrix."""
    A = R[np.ix_(preds, preds)]
    b = R[target, preds]
    return np.linalg.solve(A + 1e-10 * np.eye(len(preds)), b)


def combo_std(R: np.ndarray, idx: list[int]) -> tuple[float, np.ndarray]:
    """Tightest unit-norm combination of the standardised members of `idx`:
    returns (std of that combination, weights)."""
    S = R[np.ix_(idx, idx)]
    w, V = np.linalg.eigh(S)
    return float(np.sqrt(max(w[0], 0.0))), V[:, 0]


# --------------------------------------------------------------------------
# out-of-sample residual with FIXED coefficients
# --------------------------------------------------------------------------
def oos_rel_resid(Z: np.ndarray, target: int, preds: list[int],
                  coefs: np.ndarray) -> float:
    """std(y - X b) / std(y) on Z, with b transported from another stratum.
    The intercept is refit (a level offset is not a coefficient); the slopes
    are not.  This is the honest test that a relation is a law and not a fit."""
    y = Z[:, target]
    pred = Z[:, preds] @ coefs
    r = y - pred
    return float(r.std() / max(y.std(), EPS))
