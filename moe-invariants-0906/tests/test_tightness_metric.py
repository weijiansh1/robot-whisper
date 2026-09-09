"""Calibrate the tightness instrument on data whose answer is known.

If these fail the ranking means nothing, so they run before anything else is
believed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXP))

from phase1_lib import (  # noqa: E402
    combo_std, cov_decomposition, fit_std_coefs, null_corr, oos_rel_resid,
    to_corr, triple_best,
)


def _std(X):
    return (X - X.mean(0)) / X.std(0)


def test_exact_linear_relation_gives_zero_residual():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4000), rng.normal(size=4000)
    c = 3.0 * a - 1.5 * b            # exact, like the entropy identity
    Z = _std(np.c_[c, a, b])
    R = to_corr(np.cov(Z, rowvar=False, bias=True))
    rr, _ = triple_best(R, 0, collinear_cut=1.1)
    assert rr[1, 2] < 1e-6
    assert combo_std(R, [0, 1, 2])[0] < 1e-6


def test_independent_pair_gives_residual_one():
    rng = np.random.default_rng(1)
    Z = _std(rng.normal(size=(200000, 2)))
    R = to_corr(np.cov(Z, rowvar=False, bias=True))
    assert abs(np.sqrt(1 - R[0, 1] ** 2) - 1.0) < 1e-3


@pytest.mark.parametrize("r_true", [0.5, 0.9, 0.99, 0.999])
def test_pair_residual_is_sqrt_one_minus_r2(r_true):
    rng = np.random.default_rng(2)
    x = rng.normal(size=400000)
    y = r_true * x + np.sqrt(1 - r_true ** 2) * rng.normal(size=400000)
    R = to_corr(np.cov(_std(np.c_[x, y]), rowvar=False, bias=True))
    got = np.sqrt(1 - R[0, 1] ** 2)
    assert abs(got - np.sqrt(1 - r_true ** 2)) < 5e-3


def test_transported_coefficients_detect_instability():
    """A relation whose slope changes between strata must not look tight when
    the slope is transported."""
    rng = np.random.default_rng(3)
    x1 = rng.normal(size=60000)
    y1 = 2.0 * x1
    x2 = rng.normal(size=60000)
    y2 = -2.0 * x2                      # same |r|, opposite slope
    Z1, Z2 = np.c_[y1, x1], np.c_[y2, x2]
    mu, sd = Z1.mean(0), Z1.std(0)
    R = to_corr(np.cov((Z1 - mu) / sd, rowvar=False, bias=True))
    coefs = fit_std_coefs(R, 0, [1])
    assert oos_rel_resid((Z1 - mu) / sd, 0, [1], coefs) < 1e-8
    assert oos_rel_resid((Z2 - mu) / sd, 0, [1], coefs) > 1.9


def test_null_closed_form_matches_permutation():
    """Cov_null == Cov_between for a within-group permutation."""
    rng = np.random.default_rng(4)
    g = rng.integers(0, 60, 30000)
    lvl = rng.normal(size=(60, 3))
    Z = lvl[g] + rng.normal(size=(30000, 3))
    Z = _std(Z)
    dec = cov_decomposition(Z, g)
    R_null = null_corr(dec["tot"], dec["bet"])
    got = []
    for _ in range(6):
        P = np.empty_like(Z)
        order = np.argsort(g, kind="stable")
        for v in range(3):
            ordv = np.argsort(g + rng.random(len(g)), kind="stable")
            P[order, v] = Z[ordv, v]
        got.append(to_corr(np.cov(P, rowvar=False, bias=True)))
    assert np.abs(np.mean(got, axis=0) - R_null).max() < 0.02


def test_null_is_not_trivially_zero():
    """Two variables driven by the same group level must keep that coupling
    under the surrogate, otherwise the floor would be meaninglessly low."""
    rng = np.random.default_rng(5)
    g = rng.integers(0, 40, 20000)
    lvl = rng.normal(size=40)
    Z = _std(np.c_[lvl[g] + 0.3 * rng.normal(size=20000),
                   lvl[g] + 0.3 * rng.normal(size=20000)])
    dec = cov_decomposition(Z, g)
    R_null = null_corr(dec["tot"], dec["bet"])
    assert R_null[0, 1] > 0.7


def test_nonlinear_scan_recovers_a_monotone_map():
    from phase1_nonlinear import scan
    rng = np.random.default_rng(6)
    x = rng.normal(size=120000)
    y = np.tanh(3 * x)                       # exact but strongly nonlinear
    Z = _std(np.c_[x, y])
    eta = scan(Z, nbins=64)
    assert np.sqrt(1 - eta[0, 1]) < 0.02     # y = f(x) recovered
    r = abs(np.corrcoef(x, y)[0, 1])
    assert np.sqrt(1 - r ** 2) > 0.15        # the linear scan would have missed it
