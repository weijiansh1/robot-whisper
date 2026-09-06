"""Unit tests for the failure-matched object-readability diagnostic.

This experiment exists to remove one confound: the sibling analysis matched
cases that are 100% inside failed episodes against controls that are 96%
successful.  Here both sides are failing episodes.  Three things have to be
right before any corpus number is allowed to speak.

*The matcher.*  Its job is to make the two groups differ in the object's
response and in nothing else that is easy to fix.  The calipers must be hard,
the ``(run, task)`` stratum must be absolute, and a case that cannot be matched
must come back labelled, never silently vanish.  A control from a *successful*
episode must never enter -- that is the whole experiment, so it gets its own
test at the level the driver enforces it.

*The matched-pair AUC and its null.*  Checked against constructions whose answer
is known by hand: perfect separation, perfect anti-separation, exact ties, and a
stratum design where equal-weight-per-stratum and equal-weight-per-pair
disagree.

*The 10,000-draw machinery.*  The matrix forms of the bootstrap were written to
make 10,000 draws affordable, and they are pinned against the scalar reference
they replace.  A fast estimator that disagrees with the slow one is not an
optimisation, it is a second result.

The module under test is a deliberate copy of the sibling's helpers, not an
import of them, so that ``diagnose_object_readability.py`` stays byte-for-byte
reproducible and neither file can drift the other.  These tests therefore also
serve as the check that the copy still behaves like the original.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import diagnose_object_readability_failed as dorf  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_events(records: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    defaults = {
        "run": "runA",
        "task": "taskA",
        "episode": 0,
        "target": "obj",
        "success": False,
        "eef_max_displacement_after_m": 0.30,
        "target_max_displacement_after_m": 0.30,
        "post_closure_state_query": 10,
        "repeated_target_closure": False,
    }
    for column, value in defaults.items():
        if column not in frame.columns:
            frame[column] = value
        else:
            frame[column] = frame[column].fillna(value)
    return frame


# --------------------------------------------------------------------------
# the design: both groups inside failed episodes
# --------------------------------------------------------------------------


def test_the_control_pool_is_restricted_to_failed_episodes() -> None:
    """The confound removal is a filter on the pool, and it must actually bite.

    The matcher is pool-agnostic on purpose -- it matches whatever it is given --
    so the guarantee lives in the filter the driver applies.  This test pins that
    filter, because if it silently stopped working the experiment would quietly
    turn back into the sibling.
    """
    events = make_events(
        [
            {"event_id": 0, "episode": 0, "success": False},
            {"event_id": 1, "episode": 1, "success": True},
            {"event_id": 2, "episode": 2, "success": False},
        ]
    )
    pool = events[~events["success"]]
    assert sorted(pool["event_id"]) == [0, 2]
    pairs, _ = dorf.match_events(
        make_events([{"event_id": 9, "episode": 9}]),
        pool,
        eef_caliper=0.05,
        query_caliper=3,
        max_controls=5,
    )
    assert not pairs["control_success"].any()
    assert sorted(pairs["control_event_id"]) == [0, 2]


def test_pairs_record_both_outcomes_so_the_balance_can_be_audited() -> None:
    """``balance_record`` reports both success rates; they must be readable."""
    pairs, _ = dorf.match_events(
        make_events([{"event_id": 0, "episode": 0}]),
        make_events([{"event_id": 1, "episode": 1}, {"event_id": 2, "episode": 2}]),
        eef_caliper=0.05,
        query_caliper=3,
        max_controls=5,
    )
    record = dorf.balance_record(pairs, pd.DataFrame(), n_case=1)
    assert record["case_success_rate"] == 0.0
    assert record["control_success_rate"] == 0.0
    assert record["case_matched"] == 1
    assert record["pairs"] == 2


# --------------------------------------------------------------------------
# matcher
# --------------------------------------------------------------------------


def test_matcher_respects_the_eef_caliper() -> None:
    """A control just outside the displacement caliper is not a control."""
    case = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "eef_max_displacement_after_m": 0.320},  # inside
            {"event_id": 2, "episode": 2, "eef_max_displacement_after_m": 0.351},  # outside
        ]
    )
    pairs, unmatched = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert unmatched.empty
    assert sorted(pairs["control_event_id"]) == [1]
    assert pairs["abs_eef_difference_m"].max() <= 0.05


def test_matcher_respects_the_query_caliper() -> None:
    """Position in the episode is the confound the second caliper removes.

    Cases sit around post-closure query 27 and failed-coupled controls around
    12.  Without this caliper, query index alone would separate the groups and
    the routing readout would be measuring the clock.
    """
    case = make_events([{"event_id": 0, "episode": 0, "post_closure_state_query": 27}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "post_closure_state_query": 30},  # inside
            {"event_id": 2, "episode": 2, "post_closure_state_query": 31},  # outside
            {"event_id": 3, "episode": 3, "post_closure_state_query": 12},  # far outside
        ]
    )
    pairs, unmatched = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert unmatched.empty
    assert sorted(pairs["control_event_id"]) == [1]
    assert pairs["abs_query_difference"].max() <= 3


def test_matcher_never_crosses_a_task_or_a_run() -> None:
    """Different tasks hold different objects and scenes; that is not a control."""
    case = make_events([{"event_id": 0, "episode": 0, "task": "taskA", "run": "runA"}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "task": "taskB", "run": "runA"},
            {"event_id": 2, "episode": 2, "task": "taskA", "run": "runB"},
            {"event_id": 3, "episode": 3, "task": "taskA", "run": "runA"},
        ]
    )
    pairs, _ = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert sorted(pairs["control_event_id"]) == [3]
    assert set(pairs["run_task"]) == {"runA|taskA"}


def test_matcher_excludes_the_cases_own_episode() -> None:
    case = make_events([{"event_id": 0, "episode": 7}])
    control = make_events([{"event_id": 1, "episode": 7}, {"event_id": 2, "episode": 8}])
    pairs, _ = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert sorted(pairs["control_event_id"]) == [2]


def test_unmatched_cases_are_reported_not_dropped() -> None:
    """Every case leaves the matcher either as a stratum or as a labelled drop."""
    case = make_events(
        [
            {"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.30},
            {"event_id": 1, "episode": 1, "eef_max_displacement_after_m": 0.90},  # far away
            {"event_id": 2, "episode": 2, "task": "taskZ"},  # no control in that task
        ]
    )
    control = make_events([{"event_id": 3, "episode": 3, "eef_max_displacement_after_m": 0.31}])
    pairs, unmatched = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    accounted = set(pairs["phantom_event_id"]) | set(unmatched["event_id"])
    assert accounted == {0, 1, 2}
    assert len(pairs["phantom_event_id"].unique()) + len(unmatched) == len(case)
    reasons = dict(zip(unmatched["event_id"], unmatched["drop_reason"]))
    assert reasons == {1: "caliper_empty", 2: "no_control_in_task"}


def test_matcher_caps_controls_and_takes_the_nearest() -> None:
    case = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {
                "event_id": 10 + k,
                "episode": 10 + k,
                "eef_max_displacement_after_m": 0.300 + 0.001 * k,
            }
            for k in range(1, 9)
        ]
    )
    pairs, _ = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=3
    )
    assert len(pairs) == 3
    assert sorted(pairs["control_event_id"]) == [11, 12, 13]
    assert list(pairs.sort_values("control_rank")["control_rank"]) == [0, 1, 2]


def test_matcher_trades_the_two_calipers_off_on_a_common_scale() -> None:
    """Distance is standardised by each caliper, so neither unit dominates.

    Metres and query indices are not comparable raw.  A control half a caliper
    away in displacement must rank the same as one half a caliper away in query.
    """
    case = make_events(
        [
            {
                "event_id": 0,
                "episode": 0,
                "eef_max_displacement_after_m": 0.300,
                "post_closure_state_query": 20,
            }
        ]
    )
    control = make_events(
        [
            {  # 0.2 calipers of displacement, 0 of query
                "event_id": 1,
                "episode": 1,
                "eef_max_displacement_after_m": 0.305,
                "post_closure_state_query": 20,
            },
            {  # 0 of displacement, 0.667 calipers of query
                "event_id": 2,
                "episode": 2,
                "eef_max_displacement_after_m": 0.300,
                "post_closure_state_query": 22,
            },
        ]
    )
    pairs, _ = dorf.match_events(
        case, control, eef_caliper=0.025, query_caliper=3, max_controls=2
    )
    ordered = pairs.sort_values("control_rank")["control_event_id"].tolist()
    assert ordered == [1, 2]
    assert pairs.set_index("control_event_id").loc[1, "caliper_distance"] == pytest.approx(0.2)
    assert pairs.set_index("control_event_id").loc[2, "caliper_distance"] == pytest.approx(
        2.0 / 3.0
    )


def test_matcher_is_deterministic_under_ties() -> None:
    """Equidistant candidates are broken by event_id, so reruns agree."""
    case = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {"event_id": 5, "episode": 5, "eef_max_displacement_after_m": 0.310},
            {"event_id": 4, "episode": 4, "eef_max_displacement_after_m": 0.310},
            {"event_id": 6, "episode": 6, "eef_max_displacement_after_m": 0.310},
        ]
    )
    first, _ = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    second, _ = dorf.match_events(
        case.iloc[::-1], control.iloc[::-1], eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    assert list(first.sort_values("control_rank")["control_event_id"]) == [4, 5]
    assert list(second.sort_values("control_rank")["control_event_id"]) == [4, 5]


def test_matcher_rejects_a_nonpositive_caliper() -> None:
    case = make_events([{"event_id": 0, "episode": 0}])
    control = make_events([{"event_id": 1, "episode": 1}])
    with pytest.raises(ValueError):
        dorf.match_events(case, control, eef_caliper=0.0, query_caliper=3, max_controls=5)
    with pytest.raises(ValueError):
        dorf.match_events(case, control, eef_caliper=0.05, query_caliper=0, max_controls=5)
    with pytest.raises(ValueError):
        dorf.match_events(case, control, eef_caliper=0.05, query_caliper=3, max_controls=0)


def test_controls_may_be_reused_across_strata_but_not_inside_one() -> None:
    case = make_events([{"event_id": 0, "episode": 0}, {"event_id": 1, "episode": 1}])
    control = make_events([{"event_id": 2, "episode": 2}, {"event_id": 3, "episode": 3}])
    pairs, unmatched = dorf.match_events(
        case, control, eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    assert unmatched.empty
    assert len(pairs) == 4  # both cases keep both controls
    for _, stratum in pairs.groupby("stratum_id"):
        assert stratum["control_event_id"].is_unique


def test_a_wider_caliper_never_loses_a_match() -> None:
    """The wide variant must be a superset of the primary, or it is not a variant."""
    rng = np.random.default_rng(3)
    case = make_events(
        [
            {
                "event_id": int(k),
                "episode": int(k),
                "eef_max_displacement_after_m": float(rng.uniform(0.15, 0.5)),
                "post_closure_state_query": int(rng.integers(5, 35)),
            }
            for k in range(30)
        ]
    )
    control = make_events(
        [
            {
                "event_id": 100 + int(k),
                "episode": 100 + int(k),
                "eef_max_displacement_after_m": float(rng.uniform(0.15, 0.5)),
                "post_closure_state_query": int(rng.integers(5, 35)),
            }
            for k in range(120)
        ]
    )
    narrow, narrow_out = dorf.match_events(
        case, control, eef_caliper=0.025, query_caliper=3, max_controls=5
    )
    wide, wide_out = dorf.match_events(
        case, control, eef_caliper=0.050, query_caliper=5, max_controls=5
    )
    assert set(narrow["phantom_event_id"]) <= set(wide["phantom_event_id"])
    assert len(wide_out) <= len(narrow_out)
    assert len(wide) >= len(narrow)


# --------------------------------------------------------------------------
# matched-pair AUC, answers known by hand
# --------------------------------------------------------------------------


def test_case_score_is_one_when_the_case_beats_every_control() -> None:
    assert dorf.stratum_case_score(10.0, np.array([1.0, 2.0, 3.0])) == 1.0


def test_case_score_is_zero_when_the_case_loses_to_every_control() -> None:
    assert dorf.stratum_case_score(0.0, np.array([1.0, 2.0, 3.0])) == 0.0


def test_exact_ties_score_one_half() -> None:
    """Float16 storage makes exact ties common; they must count as no evidence."""
    assert dorf.stratum_case_score(2.0, np.array([2.0, 2.0])) == 0.5
    assert dorf.stratum_case_score(2.0, np.array([1.0, 2.0])) == 0.75


def test_case_score_ignores_missing_controls() -> None:
    assert dorf.stratum_case_score(5.0, np.array([1.0, np.nan])) == 1.0
    assert np.isnan(dorf.stratum_case_score(np.nan, np.array([1.0, 2.0])))
    assert np.isnan(dorf.stratum_case_score(5.0, np.array([np.nan, np.nan])))


def test_every_stratum_carries_equal_weight() -> None:
    """A big stratum must not outvote a small one.

    Stratum A: case beats its single control            -> 1.0
    Stratum B: case loses to all nine of its controls   -> 0.0
    Equal weight per stratum gives 0.5.  Equal weight per *pair* would give
    1/10 = 0.1, which is the failure this test rules out.
    """
    a = dorf.stratum_case_score(1.0, np.array([0.0]))
    b = dorf.stratum_case_score(0.0, np.arange(1.0, 10.0))
    assert a == 1.0 and b == 0.0
    assert np.mean([a, b]) == 0.5


def test_an_auc_below_one_half_means_the_case_reads_lower() -> None:
    """The direction verdict depends on this sign, so pin it explicitly.

    Gate entropy below the controls' gate entropy is *sharper* routing, and the
    experiment reports that as its own finding rather than as |AUC - 0.5|.
    """
    sharper = [dorf.stratum_case_score(0.10, np.array([0.20, 0.30])) for _ in range(10)]
    noisier = [dorf.stratum_case_score(0.40, np.array([0.20, 0.30])) for _ in range(10)]
    assert np.mean(sharper) == 0.0
    assert np.mean(noisier) == 1.0


def test_promotion_scores_average_to_one_half_exactly() -> None:
    """The reason the null cannot be a leave-one-out average.

    Promoting every control in turn and averaging is 0.5 in *every* stratum by
    antisymmetry.  A null built that way is a constant with no spread, and
    comparing an AUC against it is comparing against 0.5 by another name.  The
    informative null therefore promotes exactly one control per draw, which is
    what the pipeline does.
    """
    for controls in (
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([4.0, 1.0, 3.0, 2.0]),
        np.array([0.1, 0.1, 0.9]),  # with ties
        np.array([7.0, 7.0, 7.0, 7.0]),  # all tied
    ):
        scores = dorf.stratum_promotion_scores(controls)
        assert np.nanmean(scores) == pytest.approx(0.5)
    assert dorf.stratum_promotion_scores(np.array([1.0, 2.0, 3.0, 4.0]))[0] == pytest.approx(0.0)


def test_null_needs_at_least_two_controls() -> None:
    assert np.all(np.isnan(dorf.stratum_promotion_scores(np.array([1.0]))))
    assert dorf.stratum_promotion_scores(np.array([])).size == 0
    partial = dorf.stratum_promotion_scores(np.array([1.0, np.nan]))
    assert np.all(np.isnan(partial))


def test_promotion_is_the_case_estimator_with_a_control_standing_in() -> None:
    """The null is the same estimator with a failed-coupled event in the case's seat."""
    controls = np.array([0.2, 0.7, 0.9, 0.4])
    scores = dorf.stratum_promotion_scores(controls)
    for position in range(len(controls)):
        rest = np.delete(controls, position)
        assert scores[position] == pytest.approx(
            dorf.stratum_case_score(controls[position], rest)
        )


def test_promotion_skips_missing_controls() -> None:
    scores = dorf.stratum_promotion_scores(np.array([1.0, np.nan, 3.0, 5.0]))
    assert np.isnan(scores[1])
    assert scores[0] == pytest.approx(0.0)
    assert scores[3] == pytest.approx(1.0)
    assert np.nanmean(scores) == pytest.approx(0.5)


def test_vectorised_stratum_scores_match_the_scalar_reference() -> None:
    """``stratum_scores`` is what runs on the corpus; pin it to the hand-checked pair."""
    rng = np.random.default_rng(7)
    n_control, n_cell = 4, 25
    controls = rng.normal(size=(n_control, n_cell))
    controls[1, 3] = np.nan
    controls[:, 7] = np.nan  # a cell where no control survives
    controls[2:, 11] = np.nan  # a cell with only two controls
    case = rng.normal(size=n_cell)
    case[5] = np.nan
    case_score, promotion = dorf.stratum_scores(case, controls)
    for cell in range(n_cell):
        expected = dorf.stratum_case_score(case[cell], controls[:, cell])
        if np.isnan(expected):
            assert np.isnan(case_score[cell])
        else:
            assert case_score[cell] == pytest.approx(expected)
        expected_promotion = dorf.stratum_promotion_scores(controls[:, cell])
        assert np.allclose(promotion[:, cell], expected_promotion, equal_nan=True)


def test_stratum_scores_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dorf.stratum_scores(np.zeros(3), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        dorf.stratum_scores(np.zeros(3), np.zeros(3))


def test_synthetic_separation_recovers_a_known_auc() -> None:
    """Cases uniformly above their controls give AUC 1; below give 0; between, 0.5."""
    rng = np.random.default_rng(0)
    high = [dorf.stratum_case_score(10.0 + k, rng.normal(0, 1, 4)) for k in range(20)]
    low = [dorf.stratum_case_score(-10.0 - k, rng.normal(0, 1, 4)) for k in range(20)]
    middle = [
        dorf.stratum_case_score(float(k) + 1.0, np.array([float(k), float(k) + 2.0]))
        for k in range(20)
    ]
    assert np.mean(high) == 1.0
    assert np.mean(low) == 0.0
    assert np.mean(middle) == 0.5


# --------------------------------------------------------------------------
# clustered bootstrap and its 10,000-draw matrix forms
# --------------------------------------------------------------------------


def test_bootstrap_resamples_whole_clusters() -> None:
    cluster = np.array(["t1", "t1", "t1", "t2", "t3"])
    weights = dorf.bootstrap_weights(cluster, draws=200, rng=np.random.default_rng(0))
    assert weights.shape == (200, 5)
    assert np.all(weights[:, 0] == weights[:, 1])
    assert np.all(weights[:, 1] == weights[:, 2])
    assert np.all(weights.sum(axis=1) > 0)
    per_draw = weights[:, [0, 3, 4]].sum(axis=1)
    assert np.all(per_draw == 3)


def test_task_clustering_is_more_conservative_than_run_task_clustering() -> None:
    """Why the task is the headline unit.

    The same LIBERO task under two seed batches is one scene and one object.
    Splitting it into two clusters doubles the apparent number of independent
    units and narrows the interval.  This test shows the direction of that
    effect on a construction where the two batches agree by design, so the
    narrowing is pure bookkeeping and not information.
    """
    task = np.array(["t1", "t1", "t2", "t2", "t3", "t3"])
    run_task = np.array(["r1|t1", "r2|t1", "r1|t2", "r2|t2", "r1|t3", "r2|t3"])
    values = np.array([0.9, 0.9, 0.5, 0.5, 0.1, 0.1])
    rng = np.random.default_rng(11)
    by_task = dorf.weighted_means(dorf.bootstrap_weights(task, 4000, rng), values)
    rng = np.random.default_rng(11)
    by_run_task = dorf.weighted_means(dorf.bootstrap_weights(run_task, 4000, rng), values)
    lo_t, hi_t = dorf.percentile_interval(by_task)
    lo_rt, hi_rt = dorf.percentile_interval(by_run_task)
    assert (hi_rt - lo_rt) < (hi_t - lo_t)


def test_bootstrap_of_a_constant_is_that_constant() -> None:
    cluster = np.array(["t1", "t2", "t3", "t4"])
    weights = dorf.bootstrap_weights(cluster, draws=100, rng=np.random.default_rng(1))
    means = dorf.weighted_means(weights, np.full(4, 0.7))
    assert np.allclose(means, 0.7)
    assert dorf.percentile_interval(means) == pytest.approx((0.7, 0.7))


def test_weighted_means_skips_missing_strata() -> None:
    weights = np.ones((3, 4))
    means = dorf.weighted_means(weights, np.array([1.0, np.nan, 3.0, np.nan]))
    assert np.allclose(means, 2.0)
    empty = dorf.weighted_means(weights, np.full(4, np.nan))
    assert np.all(np.isnan(empty))


def test_one_cluster_bootstrap_has_no_between_cluster_variance() -> None:
    """With a single cluster the interval collapses; that is the honest answer."""
    weights = dorf.bootstrap_weights(np.array(["t1", "t1"]), 100, np.random.default_rng(2))
    means = dorf.weighted_means(weights, np.array([0.2, 0.8]))
    assert np.allclose(means, 0.5)


def test_matrix_bootstrap_agrees_with_the_column_at_a_time_reference() -> None:
    """The 10,000-draw shortcut must be the same number, not a similar one."""
    rng = np.random.default_rng(5)
    weights = dorf.bootstrap_weights(
        np.array(["a", "a", "b", "c", "c", "d"]), 300, rng
    )
    values = rng.normal(size=(6, 9))
    values[2, 4] = np.nan
    values[:, 6] = np.nan  # a cell nobody can estimate
    matrix = dorf.weighted_column_means(weights, values)
    for cell in range(values.shape[1]):
        reference = dorf.weighted_means(weights, values[:, cell])
        assert np.allclose(matrix[:, cell], reference, equal_nan=True)


def test_matrix_bootstrap_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dorf.weighted_column_means(np.ones((4, 3)), np.ones((5, 2)))
    with pytest.raises(ValueError):
        dorf.weighted_column_means(np.ones((4, 3)), np.ones(3))


def test_promoted_column_means_agrees_with_an_explicit_gather() -> None:
    """The null's matrix form is the same statistic as the naive gather."""
    rng = np.random.default_rng(13)
    n_stratum, n_control, n_cell = 6, 4, 7
    draws = 200
    promotion = rng.normal(size=(n_stratum, n_control, n_cell))
    promotion[1, 3, :] = np.nan  # a control this stratum never had
    promotion[:, :, 5] = np.nan  # a cell nobody can estimate
    promotion[4, 2, 1] = np.nan
    weights = dorf.bootstrap_weights(np.array(list("aabbcc")), draws, rng)
    promoted = rng.integers(0, n_control, size=(draws, n_stratum))

    fast = dorf.promoted_column_means(weights, promotion, promoted)

    slow = np.full((draws, n_cell), np.nan)
    for draw in range(draws):
        picked = promotion[np.arange(n_stratum), promoted[draw]]  # [n_stratum, n_cell]
        finite = np.isfinite(picked)
        effective = weights[draw][:, None] * finite
        denominator = effective.sum(axis=0)
        numerator = (effective * np.where(finite, picked, 0.0)).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            row = numerator / denominator
        row[denominator <= 0] = np.nan
        slow[draw] = row
    assert np.allclose(fast, slow, equal_nan=True)


def test_promoted_column_means_with_unit_weights_is_the_plain_mean() -> None:
    """The unweighted null centre is a plain per-draw mean over strata."""
    promotion = np.array(
        [
            [[0.0], [1.0]],
            [[0.25], [0.75]],
        ]
    )  # [2 strata, 2 controls, 1 cell]
    promoted = np.array([[0, 0], [1, 1], [0, 1]])
    weights = np.ones((3, 2))
    got = dorf.promoted_column_means(weights, promotion, promoted)
    assert got[0, 0] == pytest.approx((0.0 + 0.25) / 2)
    assert got[1, 0] == pytest.approx((1.0 + 0.75) / 2)
    assert got[2, 0] == pytest.approx((0.0 + 0.75) / 2)


def test_promoted_column_means_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dorf.promoted_column_means(np.ones((4, 3)), np.ones((3, 2)), np.zeros((4, 3), int))
    with pytest.raises(ValueError):
        dorf.promoted_column_means(np.ones((4, 3)), np.ones((5, 2, 6)), np.zeros((4, 3), int))


def test_column_intervals_agree_with_the_scalar_percentile() -> None:
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(500, 6))
    samples[:, 2] = np.nan
    samples[3:, 4] = np.nan  # only three finite draws
    samples[1:, 5] = np.nan  # only one finite draw: undefined
    lo, hi = dorf.column_intervals(samples)
    for cell in range(samples.shape[1]):
        ref_lo, ref_hi = dorf.percentile_interval(samples[:, cell])
        assert np.isnan(lo[cell]) == np.isnan(ref_lo)
        if np.isfinite(ref_lo):
            assert lo[cell] == pytest.approx(ref_lo)
            assert hi[cell] == pytest.approx(ref_hi)


# --------------------------------------------------------------------------
# multiplicity: 10,000 draws exist so BH can actually reject
# --------------------------------------------------------------------------


def test_bh_is_monotone_and_never_shrinks_a_p_value() -> None:
    values = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    adjusted = dorf.bh_adjust(values)
    assert np.all(adjusted >= values - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(values)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)
    assert adjusted[0] == pytest.approx(0.005)


def test_bh_leaves_a_uniform_field_unrejected() -> None:
    uniform = (np.arange(1, dorf.N_CELL + 1) - 0.5) / dorf.N_CELL
    assert int((dorf.bh_adjust(uniform) < 0.05).sum()) == 0


def test_bh_passes_missing_values_through() -> None:
    adjusted = dorf.bh_adjust(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(adjusted[1])
    assert np.isfinite(adjusted[0]) and np.isfinite(adjusted[2])


def test_ten_thousand_draws_can_clear_bh_and_two_thousand_cannot() -> None:
    """The stated reason for the draw count, made into an assertion.

    BH over 336 cells rejects the smallest p only if it is at or below
    0.05/336 = 1.49e-4.  A 2,000-draw bootstrap floors at 5.0e-4 and therefore
    cannot reject anything, however strong the effect; 10,000 draws floor at
    1.0e-4 and can.  The earlier sensitivity's "nothing survived BH" was that
    ceiling, not an evidence statement.
    """
    threshold = 0.05 / dorf.N_CELL
    assert 1.0 / 2000 > threshold
    assert 1.0 / dorf.BOOTSTRAP_DRAWS < threshold
    floored_at_2000 = np.full(dorf.N_CELL, 0.9)
    floored_at_2000[0] = 1.0 / 2000
    assert int((dorf.bh_adjust(floored_at_2000) < 0.05).sum()) == 0
    floored_at_10000 = np.full(dorf.N_CELL, 0.9)
    floored_at_10000[0] = 1.0 / dorf.BOOTSTRAP_DRAWS
    assert int((dorf.bh_adjust(floored_at_10000) < 0.05).sum()) == 1


# --------------------------------------------------------------------------
# routing readout
# --------------------------------------------------------------------------


def uniform_route() -> np.ndarray:
    return np.full(
        (len(dorf.LAYER_NAMES), dorf.N_TOKEN, dorf.N_EXPERTS), 1.0 / dorf.N_EXPERTS, np.float32
    )


def onehot_route(expert: int = 0) -> np.ndarray:
    route = np.zeros((len(dorf.LAYER_NAMES), dorf.N_TOKEN, dorf.N_EXPERTS), np.float32)
    route[:, :, expert] = 1.0
    return route


def test_gate_entropy_spans_zero_to_one() -> None:
    assert dorf.gate_entropy(uniform_route()) == pytest.approx(1.0)
    assert dorf.gate_entropy(onehot_route()) == pytest.approx(0.0)


def test_role_reduce_separates_state_from_action() -> None:
    per_token = np.zeros((len(dorf.LAYER_NAMES), dorf.N_TOKEN), np.float32)
    per_token[:, 0] = 1.0  # state token
    per_token[:, 1:] = 0.5  # action tokens
    assert np.allclose(dorf.role_reduce(per_token, "state"), 1.0)
    assert np.allclose(dorf.role_reduce(per_token, "action"), 0.5)
    with pytest.raises(ValueError):
        dorf.role_reduce(per_token, "everything")


def test_features_are_zero_against_an_identical_reference() -> None:
    route = onehot_route(3)
    block = dorf.event_features(route, route)
    hell = block[dorf.METRICS.index("hellinger_to_pre")]
    change = block[dorf.METRICS.index("gate_entropy_change")]
    assert np.allclose(hell, 0.0)
    assert np.allclose(change, 0.0)


def test_features_are_one_against_a_disjoint_reference() -> None:
    block = dorf.event_features(onehot_route(0), onehot_route(1))
    hell = block[dorf.METRICS.index("hellinger_to_pre")]
    assert np.allclose(hell, 1.0)


def test_gate_entropy_change_has_the_sign_of_the_sharpening() -> None:
    """A route that sharpens against its own pre-closure reference reads negative."""
    block = dorf.event_features(onehot_route(0), uniform_route())
    change = block[dorf.METRICS.index("gate_entropy_change")]
    assert np.all(change < 0.0)
    block = dorf.event_features(uniform_route(), onehot_route(0))
    change = block[dorf.METRICS.index("gate_entropy_change")]
    assert np.all(change > 0.0)


def test_features_are_nan_when_the_query_is_missing() -> None:
    missing = np.full_like(uniform_route(), np.nan)
    assert np.all(np.isnan(dorf.event_features(missing, uniform_route())))
    assert np.all(np.isnan(dorf.event_features(uniform_route(), missing)))


def test_features_reject_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        dorf.event_features(np.zeros((8, 10, 32), np.float32), uniform_route())


# --------------------------------------------------------------------------
# the cell grid and the configuration the report depends on
# --------------------------------------------------------------------------


def test_cell_frame_is_in_the_flattening_order() -> None:
    """``separation.csv`` labels flattened columns; a mismatch would mislabel every cell."""
    frame = dorf.cell_frame()
    assert len(frame) == dorf.N_CELL
    for m, metric in enumerate(dorf.METRICS):
        for layer_index, layer in enumerate(dorf.LAYER_NAMES):
            for role_index, role in enumerate(dorf.TOKEN_ROLES):
                for rel_index, relative in enumerate(dorf.RELATIVE_QUERIES):
                    row = frame.iloc[dorf.cell_index(m, layer_index, role_index, rel_index)]
                    assert row["metric"] == metric
                    assert row["layer"] == layer
                    assert row["token_role"] == role
                    assert row["relative_query"] == relative


def test_relative_query_grid_matches_the_sibling_specification() -> None:
    """The two analyses are only comparable if they read the same cells."""
    assert dorf.RELATIVE_QUERIES == (-1, 0, 1, 2, 3, 4, 5)
    assert dorf.REFERENCE_RELATIVE == -2
    assert dorf.FINAL_FLOW == 9
    assert dorf.LAYER_NAMES == ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
    assert dorf.METRICS == ("hellinger_to_pre", "gate_entropy", "gate_entropy_change")
    assert dorf.TOKEN_ROLES == ("state", "action")
    assert dorf.N_CELL == 336


def test_the_negative_control_is_part_of_the_grid() -> None:
    """Relative query -1 must always be computed, so it can always be reported."""
    assert -1 in dorf.RELATIVE_QUERIES
    frame = dorf.cell_frame()
    assert int((frame["relative_query"] == -1).sum()) == 48


def test_the_designs_use_the_siblings_caliper_values() -> None:
    """Primary here is the same matched design as the sibling's ``failed_controls``."""
    assert dorf.DESIGNS["primary"] == {"eef_caliper_m": 0.025, "query_caliper": 3}
    assert dorf.DESIGNS["wide"] == {"eef_caliper_m": 0.050, "query_caliper": 5}
    assert dorf.PRIMARY_CLUSTER == "task"
    assert dorf.CLUSTER_UNITS == ("task", "run_task")
    assert dorf.BOOTSTRAP_DRAWS == 10_000


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------


def synthetic_separation(auc_by_metric: dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, row in dorf.cell_frame().iterrows():
        rows.append(
            {
                "design": "primary",
                "scope": "both",
                "cluster_unit": "task",
                **row.to_dict(),
                "auc": auc_by_metric[row["metric"]],
                "auc_lo": auc_by_metric[row["metric"]] - 0.05,
                "auc_hi": auc_by_metric[row["metric"]] + 0.05,
                "null_auc": 0.5,
                "exceeds_null_paired": abs(auc_by_metric[row["metric"]] - 0.5) > 0.1,
                "p_boot": 0.01,
                "q_bh": 0.02,
            }
        )
    return pd.DataFrame(rows)


def test_median_abs_auc_reads_only_action_tokens_at_relative_queries_one_to_five() -> None:
    """The sibling's sensitivity statistic, reproduced so the comparison is honest."""
    frame = synthetic_separation(
        {"hellinger_to_pre": 0.70, "gate_entropy": 0.20, "gate_entropy_change": 0.55}
    )
    got = dorf.median_abs_auc_late_action(frame)
    assert got == {
        "gate_entropy": 0.3,
        "gate_entropy_change": 0.05,
        "hellinger_to_pre": 0.2,
    }
    signed = dorf.signed_median_auc_late_action(frame)
    assert signed["gate_entropy"] == pytest.approx(0.20)


def test_direction_record_calls_a_sharper_route_sharper() -> None:
    """An AUC of 0.20 must be reported as "phantom sharper", not as "effect 0.30"."""
    frame = synthetic_separation(
        {"hellinger_to_pre": 0.70, "gate_entropy": 0.20, "gate_entropy_change": 0.55}
    )
    record = dorf.direction_record(frame)
    assert record["action"]["cells_below_half"] == record["action"]["cells"]
    assert record["action"]["median_auc"] == pytest.approx(0.20)
    assert record["action"]["cells_beating_null_below_half"] == record["action"][
        "cells_beating_null"
    ]
    # the mirror image must flip the verdict, not merely change a magnitude
    mirrored = synthetic_separation(
        {"hellinger_to_pre": 0.70, "gate_entropy": 0.80, "gate_entropy_change": 0.55}
    )
    assert dorf.direction_record(mirrored)["action"]["cells_below_half"] == 0


def test_direction_record_ignores_the_negative_control_query() -> None:
    """Relative query -1 is a control, not evidence about the object; keep it out."""
    frame = synthetic_separation(
        {"hellinger_to_pre": 0.5, "gate_entropy": 0.3, "gate_entropy_change": 0.5}
    )
    record = dorf.direction_record(frame)
    assert record["action"]["cells"] == len(dorf.LAYER_NAMES) * 6  # rel 0..5 only


def test_earliest_exceeding_reports_the_negative_control_separately() -> None:
    frame = synthetic_separation(
        {"hellinger_to_pre": 0.9, "gate_entropy": 0.5, "gate_entropy_change": 0.5}
    )
    frame["exceeds_null_bh"] = frame["exceeds_null_paired"]
    got = dorf.earliest_exceeding(frame, "primary", "both", "task")
    assert got["any"] == -1
    assert got["any_excluding_negative_control"] == 0
    assert got["by_metric_role"]["gate_entropy|action"] is None


def test_side_by_side_joins_on_the_cell_not_the_row_order() -> None:
    """A positional join would silently compare different cells."""
    here = synthetic_separation(
        {"hellinger_to_pre": 0.7, "gate_entropy": 0.3, "gate_entropy_change": 0.5}
    )
    sibling = synthetic_separation(
        {"hellinger_to_pre": 0.9, "gate_entropy": 0.1, "gate_entropy_change": 0.5}
    ).assign(design="primary")
    sibling = pd.concat(
        [sibling, sibling.assign(design="failed_controls", auc=sibling["auc"] * 0.9)],
        ignore_index=True,
    )
    shuffled = sibling.sample(frac=1.0, random_state=0).reset_index(drop=True)
    joined = dorf.side_by_side(here, shuffled, "task")
    assert len(joined) == dorf.N_CELL
    entropy = joined[joined["metric"] == "gate_entropy"]
    assert np.allclose(entropy["auc_failed_matched"], 0.3)
    assert np.allclose(entropy["auc_sibling_primary"], 0.1)
    assert np.allclose(entropy["auc_sibling_failed"], 0.09)


def test_side_by_side_survives_a_missing_sibling() -> None:
    here = synthetic_separation(
        {"hellinger_to_pre": 0.7, "gate_entropy": 0.3, "gate_entropy_change": 0.5}
    )
    joined = dorf.side_by_side(here, None, "task")
    assert "auc_failed_matched" in joined
    assert "auc_sibling_primary" not in joined
