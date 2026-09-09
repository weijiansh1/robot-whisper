#!/usr/bin/env python3
"""Unit tests for the pure pieces of the routing event sensor.

The three things that decide every headline number in
``experiments/build_event_sensor.py`` are arithmetic, not modelling:

* the latency computation, and in particular what happens when the sensor
  never reports at all -- a silent 0 there would turn a miss into a perfect
  score;
* the fixed-false-event-rate threshold selection, which is what keeps the
  operating point off any outcome-derived precision;
* the lead-over-clock comparison, exactly at the boundary where the report
  chunk equals T, where "at the same time as the clock" must not count as a
  lead.

Everything here runs without zarr, sklearn fits or the route cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE / "experiments"))

import build_event_sensor as sensor  # noqa: E402


# --------------------------------------------------------------------------- #
# first_trigger / report_query
# --------------------------------------------------------------------------- #


def test_first_trigger_takes_the_first_crossing_not_the_maximum():
    scores = np.array([0.1, 0.7, 0.95, 0.2])
    assert sensor.first_trigger(scores, 0.5) == 1


def test_first_trigger_is_inclusive_at_the_threshold():
    assert sensor.first_trigger(np.array([0.4, 0.5, 0.6]), 0.5) == 1


def test_first_trigger_returns_minus_one_when_it_never_crosses():
    assert sensor.first_trigger(np.array([0.1, 0.2, 0.3]), 0.9) == -1


def test_first_trigger_never_crosses_an_infinite_threshold():
    # +inf is the legal "nothing may fire" operating point that
    # threshold_at_false_event_rate returns when the budget is zero.
    assert sensor.first_trigger(np.array([0.1, 1.0]), np.inf) == -1


def test_first_trigger_rejects_nan_and_minus_infinity():
    for bad in (np.nan, -np.inf):
        with pytest.raises(ValueError):
            sensor.first_trigger(np.array([0.1, 1.0]), bad)


def test_report_queries_are_all_silent_at_an_infinite_threshold():
    scan = np.array([[5, 6], [7, 8]])
    scores = np.array([[1.0, 1.0], [0.9, 0.9]])
    assert sensor.report_queries(scan, scores, np.inf).tolist() == [-1, -1]


def test_first_trigger_skips_non_finite_scores():
    assert sensor.first_trigger(np.array([np.nan, 0.9]), 0.5) == 1


def test_first_trigger_rejects_a_two_dimensional_block():
    with pytest.raises(ValueError):
        sensor.first_trigger(np.zeros((2, 3)), 0.5)


def test_first_trigger_rows_matches_the_scalar_helper_row_by_row():
    rng = np.random.default_rng(11)
    block = rng.random((40, 4))
    for threshold in (0.05, 0.4, 0.9, 0.999):
        rows = sensor.first_trigger_rows(block, threshold)
        expected = [sensor.first_trigger(block[i], threshold) for i in range(len(block))]
        assert rows.tolist() == expected


def test_report_query_returns_the_absolute_query_of_the_first_crossing():
    scan = np.array([26, 27, 28, 29])
    assert sensor.report_query(scan, np.array([0.1, 0.2, 0.8, 0.9]), 0.5) == 28


def test_report_query_returns_minus_one_when_the_sensor_stays_silent():
    scan = np.array([26, 27, 28, 29])
    assert sensor.report_query(scan, np.array([0.1, 0.2, 0.3, 0.4]), 0.5) == -1


def test_report_query_rejects_misaligned_scan_and_scores():
    with pytest.raises(ValueError):
        sensor.report_query(np.array([1, 2, 3]), np.array([0.1, 0.2]), 0.5)


def test_report_queries_matches_the_scalar_helper():
    scan = np.array([[5, 6, 7], [30, 31, 32]])
    scores = np.array([[0.9, 0.1, 0.1], [0.0, 0.0, 0.0]])
    assert sensor.report_queries(scan, scores, 0.5).tolist() == [5, -1]


# --------------------------------------------------------------------------- #
# latency, including the never-reported case
# --------------------------------------------------------------------------- #


def test_latency_is_report_minus_closure():
    assert sensor.latency(29, 26) == 3.0


def test_latency_is_zero_when_the_sensor_fires_on_the_closure_query():
    assert sensor.latency(26, 26) == 0.0


def test_latency_is_negative_when_routing_timing_fires_before_the_true_closure():
    # Under routing timing the scan is anchored on the predicted closure, so a
    # report can legitimately land before the event it is reporting.
    assert sensor.latency(24, 26) == -2.0


def test_latency_is_nan_when_the_sensor_never_reported():
    value = sensor.latency(-1, 26)
    assert np.isnan(value)


def test_latency_never_reported_is_not_silently_zero():
    # The failure this guards against: -1 read as an ordinary query index would
    # make a miss look like a 27-chunk-early detection.
    assert sensor.latency(-1, 26) != -27.0


def test_latencies_keeps_never_reported_rows_as_nan():
    values = sensor.latencies(np.array([29, -1, 26]), np.array([26, 26, 26]))
    assert values[0] == 3.0
    assert np.isnan(values[1])
    assert values[2] == 0.0


def test_latencies_agrees_with_the_scalar_helper():
    reports = np.array([10, -1, 4, 7])
    closures = np.array([8, 8, 8, 8])
    values = sensor.latencies(reports, closures)
    for i in range(len(reports)):
        expected = sensor.latency(int(reports[i]), int(closures[i]))
        assert (np.isnan(values[i]) and np.isnan(expected)) or values[i] == expected


def test_latencies_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        sensor.latencies(np.array([1, 2]), np.array([1]))


def test_median_latency_over_detected_rows_ignores_the_silent_ones():
    reports = np.array([27, -1, 28, 26])
    closures = np.full(4, 26)
    values = sensor.latencies(reports, closures)
    detected = reports >= 0
    assert np.nanmedian(values[detected]) == 1.0


# --------------------------------------------------------------------------- #
# fixed false-event-rate threshold selection
# --------------------------------------------------------------------------- #


def test_threshold_holds_the_false_event_budget_exactly():
    scores = np.arange(100, dtype=float) / 100.0
    threshold = sensor.threshold_at_false_event_rate(scores, 0.05)
    assert float((scores >= threshold).mean()) <= 0.05


def test_threshold_is_the_lowest_one_that_holds_the_budget():
    scores = np.arange(100, dtype=float) / 100.0
    threshold = sensor.threshold_at_false_event_rate(scores, 0.05)
    lower = np.max(scores[scores < threshold])
    assert float((scores >= lower).mean()) > 0.05


def test_threshold_at_a_budget_of_zero_lets_nothing_through():
    scores = np.array([0.1, 0.4, 0.9])
    threshold = sensor.threshold_at_false_event_rate(scores, 0.0)
    assert np.isinf(threshold)
    assert int((scores >= threshold).sum()) == 0


def test_threshold_at_a_budget_of_one_admits_everything():
    scores = np.array([0.1, 0.4, 0.9])
    threshold = sensor.threshold_at_false_event_rate(scores, 1.0)
    assert threshold == 0.1
    assert int((scores >= threshold).sum()) == 3


def test_threshold_respects_ties_that_straddle_the_budget():
    # Four controls share the top score.  A 25% budget cannot be met by cutting
    # into the tie, so the threshold has to step above all four.
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.9, 0.9, 0.9])
    threshold = sensor.threshold_at_false_event_rate(scores, 0.25)
    assert float((scores >= threshold).mean()) <= 0.25
    assert threshold > 0.9


def test_threshold_is_a_step_function_of_the_budget():
    rng = np.random.default_rng(3)
    scores = rng.random(500)
    previous = np.inf
    for target in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0):
        threshold = sensor.threshold_at_false_event_rate(scores, target)
        assert threshold <= previous
        previous = threshold


def test_threshold_matches_the_advertised_rate_on_the_reported_grid():
    rng = np.random.default_rng(7)
    scores = rng.random(2000)
    for target in sensor.TARGET_FER:
        threshold = sensor.threshold_at_false_event_rate(scores, target)
        realised = float((scores >= threshold).mean())
        assert realised <= target
        assert realised > target - 2.0 / len(scores)


def test_threshold_rejects_an_empty_control_pool():
    with pytest.raises(ValueError):
        sensor.threshold_at_false_event_rate(np.array([]), 0.01)


def test_threshold_rejects_non_finite_control_scores():
    with pytest.raises(ValueError):
        sensor.threshold_at_false_event_rate(np.array([0.1, np.nan]), 0.01)


def test_threshold_rejects_a_budget_outside_the_unit_interval():
    with pytest.raises(ValueError):
        sensor.threshold_at_false_event_rate(np.array([0.1, 0.2]), 1.5)


def test_threshold_on_a_single_control():
    assert np.isinf(sensor.threshold_at_false_event_rate(np.array([0.5]), 0.5))
    assert sensor.threshold_at_false_event_rate(np.array([0.5]), 1.0) == 0.5


# --------------------------------------------------------------------------- #
# lead over the clock, at the boundary
# --------------------------------------------------------------------------- #


def test_report_before_the_clock_is_a_lead():
    earlier, lead = sensor.lead_over_clock(30, 44)
    assert earlier
    assert lead == 14


def test_report_exactly_at_the_clock_is_not_a_lead():
    # The boundary case.  The clock rule fires at chunk T, so a sensor that
    # reports at T has bought nothing and must score zero lead.
    earlier, lead = sensor.lead_over_clock(44, 44)
    assert not earlier
    assert lead == 0


def test_report_one_chunk_before_the_clock_is_a_lead_of_one():
    earlier, lead = sensor.lead_over_clock(43, 44)
    assert earlier
    assert lead == 1


def test_report_after_the_clock_is_not_a_lead():
    earlier, lead = sensor.lead_over_clock(45, 44)
    assert not earlier
    assert lead == 0


def test_never_reported_is_not_a_lead():
    earlier, lead = sensor.lead_over_clock(-1, 44)
    assert not earlier
    assert lead == 0


def test_lead_in_environment_steps_uses_the_logged_chunk_factor():
    _, lead = sensor.lead_over_clock(32, 44)
    assert lead * sensor.CHUNK_TO_STEP == 120


def test_leads_over_clock_agrees_with_the_scalar_helper_at_the_boundary():
    reports = np.array([43, 44, 45, -1, 0])
    clocks = np.full(5, 44)
    earlier, lead = sensor.leads_over_clock(reports, clocks)
    for i in range(len(reports)):
        expected_earlier, expected_lead = sensor.lead_over_clock(int(reports[i]), int(clocks[i]))
        assert bool(earlier[i]) == expected_earlier
        assert int(lead[i]) == expected_lead


def test_leads_over_clock_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        sensor.leads_over_clock(np.array([1, 2]), np.array([1]))


# --------------------------------------------------------------------------- #
# closure prediction and the never-crossing fallback
# --------------------------------------------------------------------------- #


def test_dense_probability_blocks_queries_before_the_earliest_closure():
    dense, unique, grid = sensor.dense_probability(
        np.array([0.99, 0.99, 0.1]), np.array([2, 3, 4]), np.array([0, 0, 0]), 6
    )
    assert unique.tolist() == [0]
    assert np.isneginf(dense[0, 2])  # query 2 is below MIN_QUERY + 1
    assert dense[0, 3] == 0.99
    assert grid.tolist() == [0, 1, 2, 3, 4, 5]


def test_predict_closure_takes_the_first_crossing():
    dense, unique, grid = sensor.dense_probability(
        np.array([0.1, 0.8, 0.9]), np.array([3, 4, 5]), np.array([0, 0, 0]), 6
    )
    assert sensor.predict_closure(dense, unique, grid, 0.5, 99).tolist() == [4]


def test_predict_closure_falls_back_when_it_never_crosses():
    dense, unique, grid = sensor.dense_probability(
        np.array([0.1, 0.2, 0.3]), np.array([3, 4, 5]), np.array([0, 0, 0]), 6
    )
    assert sensor.predict_closure(dense, unique, grid, 0.9, 7).tolist() == [7]


# --------------------------------------------------------------------------- #
# clustered bootstrap
# --------------------------------------------------------------------------- #


def test_clustered_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(5)
    values = np.array([1.0] * 30 + [0.0] * 30)
    clusters = np.array(["a"] * 30 + ["b"] * 30)
    low, high = sensor.clustered_interval(values, clusters, rng, draws=400)
    assert low <= values.mean() <= high


def test_clustered_interval_is_wider_than_the_iid_one_when_a_task_dominates():
    # 72% of the phantom events sit on libero_long; resampling rows instead of
    # tasks would understate the interval badly.
    rng = np.random.default_rng(5)
    values = np.array([1.0] * 50 + [0.0] * 50)
    clustered = np.array(["a"] * 50 + ["b"] * 50)
    scattered = np.array([f"t{i}" for i in range(100)])
    wide = sensor.clustered_interval(values, clustered, rng, draws=800)
    tight = sensor.clustered_interval(values, scattered, rng, draws=800)
    assert (wide[1] - wide[0]) > (tight[1] - tight[0])


def test_clustered_interval_on_a_constant_outcome_is_degenerate():
    rng = np.random.default_rng(5)
    values = np.ones(20)
    clusters = np.array([f"t{i % 4}" for i in range(20)])
    low, high = sensor.clustered_interval(values, clusters, rng, draws=200)
    assert low == 1.0 and high == 1.0


def test_clustered_interval_rejects_misaligned_inputs():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        sensor.clustered_interval(np.array([1.0, 0.0]), np.array(["a"]), rng, draws=10)


def test_clustered_interval_is_empty_safe():
    rng = np.random.default_rng(5)
    low, high = sensor.clustered_interval(np.array([]), np.array([]), rng, draws=10)
    assert np.isnan(low) and np.isnan(high)


# --------------------------------------------------------------------------- #
# end-to-end arithmetic on a hand-made scan
# --------------------------------------------------------------------------- #


def test_a_hand_worked_operating_point():
    """Four coupled closures and three phantom ones, budget 25%.

    With one control allowed to trigger the threshold lands above the second
    highest control score; the phantom episodes are then scored for detection,
    latency and lead against a clock at T = 30.
    """
    control_max = np.array([0.10, 0.20, 0.30, 0.95])
    threshold = sensor.threshold_at_false_event_rate(control_max, 0.25)
    assert float((control_max >= threshold).mean()) <= 0.25

    scan = np.array([[26, 27, 28, 29], [26, 27, 28, 29], [26, 27, 28, 29]])
    scores = np.array(
        [
            [0.05, 0.40, 0.99, 0.99],  # reports at 28
            [0.99, 0.10, 0.10, 0.10],  # reports at 26
            [0.10, 0.10, 0.10, 0.10],  # never reports
        ]
    )
    reports = sensor.report_queries(scan, scores, threshold)
    assert reports.tolist() == [28, 26, -1]

    closures = np.array([26, 26, 26])
    values = sensor.latencies(reports, closures)
    assert values[0] == 2.0 and values[1] == 0.0 and np.isnan(values[2])

    detected = reports >= 0
    assert float(detected.mean()) == pytest.approx(2 / 3)

    earlier, lead = sensor.leads_over_clock(reports, np.full(3, 30))
    assert earlier.tolist() == [True, True, False]
    assert lead.tolist() == [2, 4, 0]
    assert float(np.median(lead[earlier]) * sensor.CHUNK_TO_STEP) == 30.0


def test_the_clock_table_is_the_one_that_was_handed_over():
    assert sensor.CLOCK_T["development_main"] == {
        "libero_goal": 21,
        "libero_long": 44,
        "libero_object": 20,
        "libero_spatial": 14,
    }
    assert sensor.CLOCK_T["external_8b"]["libero_spatial"] == 15
    assert sensor.CHUNK_TO_STEP == 10


def test_no_feature_is_wider_than_window_two():
    """The regression guard: window-8 features are NaN before chunk 8 and a
    dropna would delete grasp closure, whose median chunk is 5."""
    for name in sensor.stage2_names():
        assert "8" not in name.split("|")[0].replace("dmobility", "").replace("dentropy", "")
    assert set(sensor.STAGE1_CHANNELS) == {"mobility", "entropy", "disp2", "path2", "ratio2"}
    assert sensor.SCAN == 4 and sensor.MIN_QUERY == 2
