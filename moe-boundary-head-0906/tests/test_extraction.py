"""Extraction tests: the recomputed series must agree with what already exists.

Two independent checks are available:

  * the boundary 2x2 cached at `moe-v8-0906/results/boundary_2x2_external.npz`;
  * the adjacent-query mobility cached at
    `moe-flow-semantics-0906/results/step_profiles/{cohort}_mobility.npy` and
    `{cohort}_state_mobility.npy`, which exist for development_main and
    external_8b but not for legacy_main16x32 - which is why this bundle
    re-extracts them from raw zarr rather than reading the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))

from common import BASE_CELLS, ROOT, load  # noqa: E402

STEPS = ROOT / "moe-flow-semantics-0906/results/step_profiles"
FRONT, BACK = slice(0, 4), slice(4, 8)
LAST_STEP = 9


@pytest.mark.parametrize("cell", BASE_CELLS)
def test_boundary_matches_v8_cache(cell):
    """Element-wise agreement with the exploratory cache.

    The cache is float32 and this bundle is float64.  `sqrt(p) - sqrt(r)` is a
    near-total cancellation wherever two passes route almost identically, so the
    cache loses precision in exactly the low tail; agreement is asserted as
    "the bulk matches to float32 round-off, and every residual disagreement is
    in the low tail".
    """
    d = load("external_8b")
    cached = np.load(ROOT / "moe-v8-0906/results/boundary_2x2_external.npz")
    mine = d[cell]
    theirs = np.asarray(cached[cell], dtype=np.float64)
    fm = np.isfinite(mine)
    assert np.array_equal(fm, np.isfinite(theirs))
    diff = np.abs(mine[fm] - theirs[fm])
    assert float(np.median(diff)) < 2e-7
    loud = diff > 1e-6
    assert loud.mean() < 0.01
    if loud.any():
        assert theirs[fm][loud].max() <= np.quantile(theirs[fm], 0.03)


@pytest.mark.parametrize("cohort", ["development_main", "external_8b"])
def test_within_chunk_matches_step_profiles_cache(cohort):
    """`wc_*` at step 9 must equal the published adjacent-query mobility.

    The published cache averages the ten action tokens for `mobility` and uses
    token 0 alone for `state_mobility`; both are indexed at the arrival query,
    with the first query of each episode NaN - the same convention used here.
    """
    d = load(cohort)
    action = np.load(STEPS / f"{cohort}_mobility.npy", mmap_mode="r")
    state = np.load(STEPS / f"{cohort}_state_mobility.npy", mmap_mode="r")
    n = min(2000, d[BASE_CELLS[0]].shape[0])
    rows = np.linspace(0, d[BASE_CELLS[0]].shape[0] - 1, n).astype(int)
    for cell, source, block in (("wc_front_action", action, FRONT),
                                ("wc_back_action", action, BACK),
                                ("wc_front_state", state, FRONT),
                                ("wc_back_state", state, BACK)):
        want = np.asarray(source[rows][:, :, block, LAST_STEP]).mean(axis=2)
        got = d[cell][rows]
        m = np.isfinite(got) & np.isfinite(want)
        assert m.sum() > 10_000, cell
        # the published cache is float32; compare on the shared finite support
        assert float(np.median(np.abs(got[m] - want[m]))) < 2e-7, cell
        assert float(np.abs(got[m] - want[m]).max()) < 5e-5, cell
        # and the finite supports must be the same
        assert np.array_equal(np.isfinite(got), np.isfinite(want)), cell


def test_legacy_has_no_step_profiles_cache():
    """The reason this bundle re-extracts rather than reading: legacy is the one
    cohort no version of the guard was fitted on, and it has no cache."""
    assert not (STEPS / "legacy_main16x32_mobility.npy").exists()
    d = load("legacy_main16x32")
    assert np.isfinite(d["wc_back_action"]).any()


def test_triangle_inequality_holds():
    """seam, step9 and chord are three legs of one triangle in sqrt-space."""
    for cohort in ("development_main", "external_8b", "legacy_main16x32"):
        d = load(cohort)
        for cell in ("front_action", "back_action", "front_state", "back_state"):
            a, b, c = d[cell], d[f"wc_{cell}"], d[f"chord_{cell}"]
            m = np.isfinite(a) & np.isfinite(b) & np.isfinite(c)
            # the cells are means over layers/tokens of per-leg distances, so
            # the inequality holds term by term and therefore on the mean
            assert (a[m] <= b[m] + c[m] + 1e-9).all(), (cohort, cell)
            assert (b[m] <= a[m] + c[m] + 1e-9).all(), (cohort, cell)
