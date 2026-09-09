"""Metric and depth-test checks for analyze_routing_cloud.py.

The verdict "the donor is inside the policy's own reach" rests entirely on these
two pieces, and both have a failure mode that looks like a result:

  * a distance that saturates would call everything inside;
  * a depth test calibrated against the wrong null would call everything outside.
"""

from __future__ import annotations

import numpy as np
import pytest

from analyze_routing_cloud import (
    N_EXPERTS,
    TOP_K,
    depth_test,
    jaccard_distance,
    tv_distance,
)

SHAPE = (8, 10, 11)  # layers, denoise rounds, suffix tokens


def _draw(rng, shape=SHAPE):
    idx = np.stack(
        [rng.choice(N_EXPERTS, TOP_K, replace=False) for _ in range(int(np.prod(shape)))]
    ).reshape(*shape, TOP_K).astype(np.uint8)
    w = rng.random((*shape, TOP_K)).astype(np.float32)
    return idx, (w / w.sum(-1, keepdims=True)).astype(np.float32)


def _perturb(idx, rng, fraction):
    """Replace ``fraction`` of the sites with a fresh independent top-4."""
    out = idx.copy().reshape(-1, TOP_K)
    n = out.shape[0]
    hit = rng.choice(n, int(fraction * n), replace=False)
    for i in hit:
        out[i] = rng.choice(N_EXPERTS, TOP_K, replace=False)
    return out.reshape(idx.shape)


def test_jaccard_is_zero_on_identical_routing():
    rng = np.random.default_rng(0)
    idx, _ = _draw(rng)
    assert jaccard_distance(idx, idx) == 0.0


def test_jaccard_matches_the_random_baseline():
    """Two independent top-4 picks out of 32 overlap by 0.5 expert on average."""
    rng = np.random.default_rng(1)
    a, _ = _draw(rng)
    b, _ = _draw(rng)
    # E|A n B| = 4*4/32 = 0.5, so E[J] ~ 0.5/7.5 and the distance sits near 0.93.
    assert 0.90 < jaccard_distance(a, b) < 0.96


def test_jaccard_is_monotone_in_the_perturbed_fraction():
    rng = np.random.default_rng(2)
    a, _ = _draw(rng)
    distances = [jaccard_distance(a, _perturb(a, rng, f)) for f in (0.1, 0.3, 0.6, 0.9)]
    assert distances == sorted(distances)


def test_tv_is_zero_on_identical_routing_and_bounded_by_one():
    rng = np.random.default_rng(3)
    idx, w = _draw(rng)
    assert tv_distance(idx, w, idx, w) == pytest.approx(0.0, abs=1e-12)
    other_idx, other_w = _draw(rng)
    assert 0.0 < tv_distance(idx, w, other_idx, other_w) <= 1.0


def test_tv_sees_weights_that_jaccard_cannot():
    """Same four experts, different weights: Jaccard says identical, TV does not."""
    rng = np.random.default_rng(4)
    idx, w = _draw(rng)
    w2 = np.flip(w, axis=-1).copy()
    assert jaccard_distance(idx, idx) == 0.0
    assert tv_distance(idx, w, idx, w2) > 0.0


def _metric(ai, aw, bi, bw):
    return jaccard_distance(ai, bi)


def test_a_cloud_member_tests_as_inside():
    rng = np.random.default_rng(5)
    base, base_w = _draw(rng)
    cloud_idx = np.stack([_perturb(base, rng, 0.2) for _ in range(24)])
    cloud_w = np.stack([_draw(rng)[1] for _ in range(24)])
    query, query_w = cloud_idx[0], cloud_w[0]
    result = depth_test(query, query_w, cloud_idx[1:], cloud_w[1:], _metric)
    assert result["p_inside"] > 0.05


def test_an_unrelated_routing_tests_as_outside():
    rng = np.random.default_rng(6)
    base, _ = _draw(rng)
    cloud_idx = np.stack([_perturb(base, rng, 0.05) for _ in range(24)])
    cloud_w = np.stack([_draw(rng)[1] for _ in range(24)])
    far, far_w = _draw(rng)  # independent of base
    result = depth_test(far, far_w, cloud_idx, cloud_w, _metric)
    assert result["p_inside"] <= 0.05
    assert result["d_min"] > result["cloud_loo_max"]


def test_p_inside_can_never_be_zero():
    """The +1 in the numerator keeps the test from claiming impossible certainty."""
    rng = np.random.default_rng(7)
    base, _ = _draw(rng)
    cloud_idx = np.stack([_perturb(base, rng, 0.02) for _ in range(8)])
    cloud_w = np.stack([_draw(rng)[1] for _ in range(8)])
    far, far_w = _draw(rng)
    result = depth_test(far, far_w, cloud_idx, cloud_w, _metric)
    assert result["p_inside"] == pytest.approx(1 / 9)
