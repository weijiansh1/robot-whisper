"""Unit tests for the pure aggregation helpers of the loop-specificity replication.

These cover the two pieces that carry the inference: the target-count-weighted
aggregation of per-cell lifts, and the task-clustered bootstrap.  Everything is
synthetic; no cohort is opened.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import replicate_loop_specificity as rls  # noqa: E402


def counts(**cells) -> dict[str, np.ndarray]:
    """Build a ``[1, 1, Q]`` single-suite count bundle from per-chunk lists."""
    return {
        name: np.asarray(values, dtype=np.float64)[None, None, :]
        for name, values in cells.items()
    }


# --------------------------------------------------------------------------- #
# weighted_lift
# --------------------------------------------------------------------------- #


def test_weighted_lift_matches_the_hand_computed_weighted_mean():
    lifts = np.asarray([[[3.0, 0.5]]])
    weights = np.asarray([[[10.0, 5.0]]])
    assert rls.weighted_lift(lifts, weights) == pytest.approx([(3.0 * 10 + 0.5 * 5) / 15])


def test_weighted_lift_drops_non_finite_cells_from_both_sums():
    lifts = np.asarray([[[3.0, np.nan, 0.5]]])
    weights = np.asarray([[[10.0, 1000.0, 5.0]]])
    # The NaN cell must leave the denominator too, otherwise it silently
    # shrinks the aggregate towards zero.
    assert rls.weighted_lift(lifts, weights) == pytest.approx([32.5 / 15])


def test_weighted_lift_returns_nan_when_no_cell_survives():
    lifts = np.asarray([[[np.nan, np.nan]]])
    weights = np.asarray([[[1.0, 2.0]]])
    assert np.isnan(rls.weighted_lift(lifts, weights)).all()


def test_weighted_lift_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        rls.weighted_lift(np.zeros((1, 1, 2)), np.zeros((1, 1, 3)))


# --------------------------------------------------------------------------- #
# cell_lift and band_mask
# --------------------------------------------------------------------------- #


def test_cell_lift_is_the_ratio_of_two_occupancy_shares():
    lift = rls.cell_lift(
        np.asarray([10.0]), np.asarray([6.0]), np.asarray([20.0]), np.asarray([4.0])
    )
    assert lift == pytest.approx([0.6 / 0.2])


def test_cell_lift_is_nan_when_no_timely_episode_is_circling():
    lift = rls.cell_lift(
        np.asarray([10.0, 0.0]),
        np.asarray([6.0, 0.0]),
        np.asarray([20.0, 20.0]),
        np.asarray([0.0, 4.0]),
    )
    assert np.isnan(lift).tolist() == [True, True]


def test_band_mask_splits_on_the_survival_prior_and_skips_empty_chunks():
    alive = np.asarray([100.0, 100.0, 0.0])
    fail_alive = np.asarray([10.0, 40.0, 0.0])  # priors 0.10, 0.40, undefined
    assert rls.band_mask(alive, fail_alive, "low").tolist() == [True, False, False]
    assert rls.band_mask(alive, fail_alive, "high").tolist() == [False, True, False]
    assert rls.band_mask(alive, fail_alive, "all_matched").tolist() == [True, True, False]


def test_band_mask_rejects_an_unknown_band():
    with pytest.raises(ValueError):
        rls.band_mask(np.ones(2), np.zeros(2), "middling")


# --------------------------------------------------------------------------- #
# band_stats: the analytic case
# --------------------------------------------------------------------------- #


def analytic_bundle() -> dict[str, np.ndarray]:
    """Four chunks: two scoring cells, one gated out, one in the high band.

    chunk 0: target 6/10 = 0.60 vs timely  4/20 = 0.20 -> lift 3.0, weight 10
    chunk 1: target 1/5  = 0.20 vs timely  4/10 = 0.40 -> lift 0.5, weight  5
    chunk 2: only 2 target episodes -> below MIN_TARGET, dropped
    chunk 3: survival prior 0.50 -> high band, not low
    """
    return counts(
        target_usable=[10, 5, 2, 40],
        target_circling=[6, 1, 2, 20],
        timely_usable=[20, 10, 20, 20],
        timely_circling=[4, 4, 1, 5],
        alive=[100, 100, 100, 100],
        fail_alive=[10, 10, 10, 50],
    )


def test_band_stats_low_band_reproduces_the_analytic_weighted_lift():
    stats = rls.band_stats(analytic_bundle(), "low")
    assert stats["lift"] == pytest.approx([(3.0 * 10 + 0.5 * 5) / 15])
    assert stats["cells"].tolist() == [2]
    assert stats["weight"].tolist() == [15.0]


def test_band_stats_gates_out_thin_target_cells():
    bundle = analytic_bundle()
    # chunk 2 holds 2 target episodes and would score (2/2)/(1/20) = 20x.  With
    # the gate lowered it enters and nearly doubles the aggregate, which is the
    # reason MIN_TARGET exists.
    ungated = rls.band_stats(bundle, "low", min_target=1)
    assert ungated["cells"].tolist() == [3]
    assert ungated["lift"] == pytest.approx([(3.0 * 10 + 0.5 * 5 + 20.0 * 2) / 17])
    assert ungated["lift"][0] > rls.band_stats(bundle, "low")["lift"][0]


def test_band_stats_high_band_picks_up_only_the_high_prior_chunk():
    stats = rls.band_stats(analytic_bundle(), "high")
    assert stats["cells"].tolist() == [1]
    assert stats["lift"] == pytest.approx([(20 / 40) / (5 / 20)])


def test_band_stats_timely_gate_drops_cells_with_too_few_references():
    bundle = analytic_bundle()
    stats = rls.band_stats(bundle, "low", min_timely=15)
    assert stats["cells"].tolist() == [1]  # only chunk 0 keeps 20 timely episodes
    assert stats["lift"] == pytest.approx([3.0])


def test_pooled_unmatched_ignores_the_gates_and_collapses_every_cell():
    stats = rls.band_stats(analytic_bundle(), "pooled_unmatched")
    target_share = (6 + 1 + 2 + 20) / (10 + 5 + 2 + 40)
    timely_share = (4 + 4 + 1 + 5) / (20 + 10 + 20 + 20)
    assert stats["lift"] == pytest.approx([target_share / timely_share])
    assert stats["cells"].tolist() == [4]  # the gated chunk is still counted


def test_pooling_across_chunks_can_invert_the_sign_of_the_matched_lift():
    """The confound the whole design exists to avoid, reproduced in miniature.

    Target episodes run to the horizon cap and so pile up in the late chunk,
    where *nobody* occupies the circling cell; timely episodes pile up in the
    early chunk, where almost everybody does.  Every chunk shows enrichment,
    the pooled collapse shows depletion.
    """
    bundle = counts(
        target_usable=[10, 100],
        target_circling=[6, 10],  # 0.60 early, 0.10 late
        timely_usable=[1000, 20],
        timely_circling=[500, 1],  # 0.50 early, 0.05 late
        alive=[100, 100],
        fail_alive=[10, 10],
    )
    matched = rls.band_stats(bundle, "low")
    pooled = rls.band_stats(bundle, "pooled_unmatched")
    assert matched["lift"] == pytest.approx([(1.2 * 10 + 2.0 * 100) / 110])
    assert pooled["lift"] == pytest.approx([(16 / 110) / (501 / 1020)])
    assert pooled["lift"][0] < 1.0 < matched["lift"][0]


def test_mantel_haenszel_matches_the_hand_computed_common_ratio():
    bundle = analytic_bundle()
    keep = np.asarray([[[True, True, False, False]]])
    value = rls.mantel_haenszel(
        bundle["target_usable"],
        bundle["target_circling"],
        bundle["timely_usable"],
        bundle["timely_circling"],
        keep,
    )
    numerator = 6 * 20 / 30 + 1 * 10 / 15
    denominator = 4 * 10 / 30 + 4 * 5 / 15
    assert value == pytest.approx([numerator / denominator])


def test_mean_of_ratios_and_mantel_haenszel_diverge_on_a_thin_denominator():
    """The documented fragility of the pre-registered aggregate, pinned.

    One cell where timely episodes almost never circle (1 in 1000) produces a
    300x per-cell ratio.  The weighted mean of ratios inherits it; the
    stratified common ratio does not.  This is exactly the shape of the real
    libero_goal cells, so the two estimators must be allowed to disagree.
    """
    bundle = counts(
        target_usable=[10, 100],
        target_circling=[3, 50],
        timely_usable=[1000, 100],
        timely_circling=[1, 100],
        alive=[100, 100],
        fail_alive=[5, 5],
    )
    stats = rls.band_stats(bundle, "low")
    assert stats["lift"] == pytest.approx([(300.0 * 10 + 0.5 * 100) / 110])
    assert stats["mh_lift"][0] < 1.0
    assert stats["lift"][0] > 20 * stats["mh_lift"][0]


# --------------------------------------------------------------------------- #
# resampling machinery
# --------------------------------------------------------------------------- #


def test_task_multiplicities_draw_one_task_per_slot_with_replacement():
    rng = np.random.default_rng(0)
    draws = rls.task_multiplicities(6, 500, rng)
    assert draws.shape == (500, 6)
    assert (draws.sum(axis=1) == 6).all()
    assert draws.max() > 1  # with replacement, some task repeats


def test_task_multiplicities_reject_degenerate_sizes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        rls.task_multiplicities(0, 10, rng)
    with pytest.raises(ValueError):
        rls.task_multiplicities(4, 0, rng)


def test_suite_sums_group_tasks_into_their_suite():
    per_task = np.asarray([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
    suite_index = np.asarray([0, 0, 1])
    ones = np.ones((1, 3))
    out = rls.suite_sums(per_task, suite_index, 2, ones)
    assert out.shape == (1, 2, 2)
    assert out[0, 0].tolist() == [11.0, 22.0]
    assert out[0, 1].tolist() == [100.0, 200.0]


def test_suite_sums_scale_with_the_resample_multiplicity():
    per_task = np.asarray([[1.0], [10.0]])
    suite_index = np.asarray([0, 0])
    out = rls.suite_sums(per_task, suite_index, 1, np.asarray([[2.0, 0.0]]))
    assert out[0, 0].tolist() == [2.0]


def test_suite_sums_reject_a_misaligned_task_axis():
    with pytest.raises(ValueError):
        rls.suite_sums(np.zeros((3, 2)), np.zeros(2, dtype=int), 1, np.ones((1, 3)))
    with pytest.raises(ValueError):
        rls.suite_sums(np.zeros((3, 2)), np.zeros(3, dtype=int), 1, np.ones((1, 5)))


def one_suite_tasks(
    target_usable, target_circling, timely_usable, timely_circling
) -> dict[str, np.ndarray]:
    """Per-task ``[T, 1]`` bundles for a single suite and a single chunk."""
    n_tasks = len(target_usable)
    return {
        "target_usable": np.asarray(target_usable, dtype=np.float64)[:, None],
        "target_circling": np.asarray(target_circling, dtype=np.float64)[:, None],
        "timely_usable": np.asarray(timely_usable, dtype=np.float64)[:, None],
        "timely_circling": np.asarray(timely_circling, dtype=np.float64)[:, None],
        "alive": np.full((n_tasks, 1), 100.0),
        "fail_alive": np.full((n_tasks, 1), 5.0),  # prior 0.05 -> low band
    }


def test_all_ones_multiplicities_reproduce_the_point_estimate():
    per_task = one_suite_tasks([20] * 4, [10] * 4, [200] * 4, [55] * 4)
    suite_index = np.zeros(4, dtype=int)
    identity = np.ones((1, 4))
    point = rls.band_stats(
        {k: rls.suite_sums(v, suite_index, 1, identity) for k, v in per_task.items()},
        "low",
    )
    through_bootstrap = rls.bootstrap_band_stats(per_task, suite_index, 1, identity, "low")
    assert through_bootstrap["lift"] == pytest.approx(point["lift"])
    assert through_bootstrap["mh_lift"] == pytest.approx(point["mh_lift"])


def test_clustered_interval_is_wide_when_every_target_sits_in_one_task():
    """Eight tasks, all target episodes in task 0, heterogeneous timely rates.

    Resampling *tasks* rather than episodes has to notice that the numerator
    rests on a single cluster: roughly a third of the draws miss task 0 entirely
    and yield no estimate at all, and the surviving draws swing with whichever
    reference tasks were drawn.  Spreading the same totals evenly over all eight
    tasks must collapse that spread.
    """
    rng = np.random.default_rng(7)
    suite_index = np.zeros(8, dtype=int)
    multiplicities = rls.task_multiplicities(8, 4000, rng)

    clustered = one_suite_tasks(
        target_usable=[20, 0, 0, 0, 0, 0, 0, 0],
        target_circling=[10, 0, 0, 0, 0, 0, 0, 0],
        timely_usable=[200] * 8,
        timely_circling=[10, 10, 10, 10, 100, 100, 100, 100],
    )
    spread = one_suite_tasks(
        target_usable=[20] * 8,
        target_circling=[10] * 8,
        timely_usable=[200] * 8,
        timely_circling=[50, 60] * 4,
    )

    clustered_draws = rls.bootstrap_band_stats(
        clustered, suite_index, 1, multiplicities, "low"
    )["lift"]
    spread_draws = rls.bootstrap_band_stats(
        spread, suite_index, 1, multiplicities, "low"
    )["lift"]

    # Both describe the same aggregate occupancy: 0.5 target vs 0.275 timely.
    identity = np.ones((1, 8))
    for bundle in (clustered, spread):
        point = rls.bootstrap_band_stats(bundle, suite_index, 1, identity, "low")
        assert point["lift"] == pytest.approx([0.5 / 0.275], rel=1e-6)

    # A third of the draws lose the only task carrying targets.
    finite = int(np.isfinite(clustered_draws).sum())
    assert 0.55 * 4000 < finite < 4000
    assert np.isfinite(spread_draws).all()

    clustered_low, clustered_high = rls.percentile_interval(clustered_draws)
    spread_low, spread_high = rls.percentile_interval(spread_draws)
    assert clustered_high - clustered_low > 5 * (spread_high - spread_low)
    assert clustered_low < 0.5 / 0.275 < clustered_high


def test_percentile_interval_ignores_non_finite_draws():
    samples = np.asarray([np.nan, 1.0, 2.0, 3.0, np.inf])
    low, high = rls.percentile_interval(samples)
    assert low == pytest.approx(1.05)
    assert high == pytest.approx(2.95)


def test_percentile_interval_is_nan_when_every_draw_failed():
    low, high = rls.percentile_interval(np.full(10, np.nan))
    assert np.isnan(low) and np.isnan(high)
