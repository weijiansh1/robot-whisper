"""Check pre-termination timing and held-out prefix calibration."""

import numpy as np
import pandas as pd

from evaluate_common_horizon import grouped_threshold, last_query_before, prefix_scores
from core import first_alarm


def test_query_at_success_boundary_is_excluded():
    np.testing.assert_array_equal(last_query_before([140, 141, 149, 150]), [13, 14, 14, 14])
    scores = np.ones((4, 20))
    scores[:, :7] = np.nan
    scores[:, 14] = 10
    peaks, endpoints, clipped = prefix_scores(scores, last_query_before([140, 141, 149, 150]))
    np.testing.assert_array_equal(peaks, [1, 10, 10, 10])
    np.testing.assert_array_equal(endpoints, [1, 10, 10, 10])
    np.testing.assert_array_equal(first_alarm(clipped, 5), [-1, 14, 14, 14])


def test_late_failure_signal_cannot_enter_early_window():
    scores = np.ones((3, 52))
    scores[:, :7] = np.nan
    scores[0, 25:] = np.nan
    scores[1, 30:] = 100
    scores[2, 10] = 20
    full = first_alarm(scores, 10)
    early_peak, _, early = prefix_scores(scores, 13)
    np.testing.assert_array_equal(full, [-1, 30, 10])
    np.testing.assert_array_equal(first_alarm(early, 10), [-1, -1, 10])
    np.testing.assert_array_equal(early_peak, [1, 1, 20])
    assert len(early_peak) == len(scores)


def test_prefix_calibration_groups_successes_and_ignores_late_values():
    frame = pd.DataFrame({"task": ["seen"] * 42, "init_state_id": np.repeat(np.arange(21), 2)})
    scores = np.broadcast_to(np.repeat(np.arange(1, 22), 2)[:, None], (42, 20)).astype(float).copy()
    scores[:, :7] = np.nan
    labels = np.zeros(42, dtype=int)
    labels[-2:] = 1
    tau, details = grouped_threshold(scores, labels, frame, 13)
    assert tau == 20 and details["units"] == 20 and details["rank"] == 20
    scores[labels == 1] = 1e9
    scores[labels == 0, 14:] = 2e9
    unchanged, _ = grouped_threshold(scores, labels, frame, 13)
    full, _ = grouped_threshold(scores, labels, frame)
    assert unchanged == tau and full == 2e9
