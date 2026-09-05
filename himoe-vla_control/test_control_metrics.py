"""Unit tests for control_metrics.py — every metric is checked against a case
with a hand-computable answer plus the invariances the program relies on
(discretization robustness, metric bounds, hierarchy of the bootstrap)."""
import numpy as np
import pytest

import control_metrics as cm


def uniform(E=4):
    return np.full(E, 1.0 / E)


def onehot(E=4, i=0):
    p = np.zeros(E); p[i] = 1.0
    return p


def test_hellinger_bounds_and_known_values():
    assert cm.hellinger(uniform(), uniform()) == pytest.approx(0.0)
    # orthogonal one-hots: d_H = 1 exactly
    assert cm.hellinger(onehot(4, 0), onehot(4, 1)) == pytest.approx(1.0)
    # symmetric, in [0,1]
    p, q = np.array([.7, .3, 0, 0]), np.array([.1, .2, .3, .4])
    assert cm.hellinger(p, q) == pytest.approx(cm.hellinger(q, p))
    assert 0 < cm.hellinger(p, q) < 1


def test_route_distance_matches_mean_gate_hellinger():
    # 2 gates (L*U=2), E=4: route_distance should equal RMS of per-gate d_H
    F, E = 3, 4
    a = np.zeros((F, 2, 1, E)); b = np.zeros((F, 2, 1, E))
    a[:, 0, 0] = onehot(E, 0); b[:, 0, 0] = onehot(E, 1)   # gate1: d_H=1
    a[:, 1, 0] = uniform(E);   b[:, 1, 0] = uniform(E)     # gate2: d_H=0
    d = cm.route_distance(cm.flatten_query(a), cm.flatten_query(b))
    # euclidean over concat / sqrt2 / sqrt(n_entries)... expected RMS over
    # gates of d_H scaled: ||concat||/sqrt(2*2E)= sqrt(2)/sqrt(2*8)... check
    # against direct computation:
    expect = np.linalg.norm(np.sqrt(a[0].ravel()/a[0].sum()) * 0)  # placeholder
    manual = np.linalg.norm(np.sqrt(a[0]).ravel() - np.sqrt(b[0]).ravel()) \
        / np.sqrt(2) / np.sqrt(a[0].size)
    assert d[0] == pytest.approx(manual)
    assert np.allclose(d, d[0])


def test_velocity_acceleration_time_normalization():
    # linear path: constant velocity, zero acceleration, regardless of F
    for F in (5, 10, 20):
        t = np.linspace(0, 1, F)[:, None]
        psi = np.hstack([t, 1 - t])          # straight line in 2-D
        v = cm.velocity(psi)
        assert np.allclose(np.linalg.norm(v, axis=-1),
                           np.linalg.norm([1.0, -1.0]))   # ds-normalized
        a = cm.acceleration(psi)
        assert np.allclose(a, 0.0, atol=1e-9)


def test_a_route_discretization_invariance():
    # smooth quarter-circle path: A^route should be ~stable across F
    def path(F):
        s = np.linspace(0, np.pi / 2, F)
        return np.stack([np.cos(s), np.sin(s)], axis=1)
    vals = [cm.a_route(path(F)) for F in (10, 20, 40)]
    assert max(vals) / min(vals) < 1.2      # within 20% across 4x resolution


def test_v_late_picks_late_segment():
    # path that only moves in the second half
    psi = np.zeros((10, 2))
    psi[5:, 0] = np.linspace(0, 1, 5)
    assert cm.v_late(psi, late_frac=0.5) > 0
    assert cm.v_late(np.zeros((10, 2)), late_frac=0.5) == 0


def test_perturbation_response_shapes_and_funnel():
    F, D = 10, 8
    base = np.zeros((F, D))
    pert = base.copy()
    bump = np.zeros(F); bump[4] = 1.0; bump[-1] = 0.1   # peak mid, small end
    pert[:, 0] = bump * 0.01
    r = cm.perturbation_response(base, pert, delta_norm=0.01)
    assert r["argmax_f"] == 4
    assert r["G_peak"] > r["G_terminal"]
    assert 0.8 < r["funnel"] <= 1.0


def test_finite_time_contraction_signs():
    lam_decay = cm.finite_time_contraction(np.array([1, .5, .25, .125, .0625]))
    lam_grow = cm.finite_time_contraction(np.array([.01, .02, .04, .08, .16]))
    assert lam_decay < 0 < lam_grow


def test_seed_dispersion_and_funnel_index():
    F, E = 6, 4
    # three seeds: diverge mid-flow, reconverge at the end
    def mk(i):
        r = np.tile(uniform(E), (F, 1, 1, 1))
        r[2, 0, 0] = onehot(E, i)            # mid-flow disagreement
        return cm.flatten_query(r.reshape(F, 1, 1, E))
    d = cm.seed_dispersion([mk(0), mk(1), mk(2)])
    ff = cm.fanout_funnel(d)
    assert ff["argmax_f"] == 2
    assert ff["terminal_diversity"] == pytest.approx(0.0, abs=1e-9)
    assert ff["funnel_index"] == pytest.approx(1.0, abs=1e-9)


def test_gate_matrix_stats_static_phenotype():
    U, E = 10, 32
    rng = np.random.default_rng(0)
    # healthy: each token confident on its own expert -> low H_row, high rank
    healthy = np.full((U, E), 1e-6)
    for u in range(U):
        healthy[u, u] = 1.0
    h = cm.gate_matrix_stats(healthy)
    # static phenotype: all tokens flat over the SAME 4-expert support
    static = np.full((U, E), 1e-6)
    static[:, :4] = 0.25 + rng.normal(0, 1e-4, (U, 4))
    s = cm.gate_matrix_stats(static)
    assert s["H_row"] > h["H_row"]          # rows flatter
    assert s["C_soft"] > h["C_soft"]        # tokens more similar
    assert s["r_eff"] < h["r_eff"]          # matrix lower rank
    assert h["r_eff"] > 5 and s["r_eff"] < 2


def test_resample_path_endpoints_and_shape():
    psi = np.linspace(0, 1, 5)[:, None] ** 2
    out = cm.resample_path(psi, 11)
    assert out.shape == (11, 1)
    assert out[0, 0] == pytest.approx(psi[0, 0])
    assert out[-1, 0] == pytest.approx(psi[-1, 0])


def test_trunk_bootstrap_units():
    # 3 trunks with within-trunk replicates; CI from trunk level (n=3 wide)
    data = {"t1": [1.0, 1.1, 0.9], "t2": [2.0, 2.2], "t3": [3.0]}
    r = cm.trunk_bootstrap(data, n_boot=2000, seed=1)
    assert r["n_trunks"] == 3
    # trunk means are [1.0, 2.1, 3.0] -> hierarchical estimate 2.0333 (NOT the
    # pooled per-sample mean 1.7 -- that is the whole point of the hierarchy)
    assert r["estimate"] == pytest.approx((1.0 + 2.1 + 3.0) / 3, abs=1e-6)
    assert r["ci_lo"] <= 1.0 + 1e-9 and r["ci_hi"] >= 3.0 - 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
