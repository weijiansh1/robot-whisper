"""Unit tests for the object-readability diagnostic.

Two things have to be right before any corpus number is allowed to speak.

*The matcher.*  Its job is to make the phantom and coupled groups differ in the
object's response and in nothing else that is easy to fix.  So the calipers must
be hard, the same-task constraint must be absolute (a control from another task
would carry another object and another scene), and a phantom that cannot be
matched must come back labelled, never silently vanish.

*The matched-pair AUC.*  It is checked against constructions whose answer is
known by hand: perfect separation, perfect anti-separation, exact ties, and a
stratum design where equal-weight-per-stratum and equal-weight-per-pair give
different answers -- the estimator must give the former, otherwise one task with
many controls outvotes the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import diagnose_object_readability as dor  # noqa: E402


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
# matcher
# --------------------------------------------------------------------------


def test_matcher_respects_the_eef_caliper() -> None:
    """A control just outside the displacement caliper is not a control."""
    phantom = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "eef_max_displacement_after_m": 0.320},  # inside
            {"event_id": 2, "episode": 2, "eef_max_displacement_after_m": 0.351},  # outside
        ]
    )
    pairs, unmatched = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert unmatched.empty
    assert sorted(pairs["control_event_id"]) == [1]
    assert pairs["abs_eef_difference_m"].max() <= 0.05


def test_matcher_respects_the_query_caliper() -> None:
    """Matching on arm travel alone leaves the position-in-episode confound."""
    phantom = make_events([{"event_id": 0, "episode": 0, "post_closure_state_query": 20}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "post_closure_state_query": 23},  # inside
            {"event_id": 2, "episode": 2, "post_closure_state_query": 24},  # outside
        ]
    )
    pairs, unmatched = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert unmatched.empty
    assert sorted(pairs["control_event_id"]) == [1]
    assert pairs["abs_query_difference"].max() <= 3


def test_matcher_never_crosses_a_task_or_a_run() -> None:
    """Different tasks hold different objects and scenes; that is not a control."""
    phantom = make_events([{"event_id": 0, "episode": 0, "task": "taskA", "run": "runA"}])
    control = make_events(
        [
            {"event_id": 1, "episode": 1, "task": "taskB", "run": "runA"},
            {"event_id": 2, "episode": 2, "task": "taskA", "run": "runB"},
            {"event_id": 3, "episode": 3, "task": "taskA", "run": "runA"},
        ]
    )
    pairs, _ = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert sorted(pairs["control_event_id"]) == [3]


def test_matcher_excludes_the_phantoms_own_episode() -> None:
    phantom = make_events([{"event_id": 0, "episode": 7}])
    control = make_events([{"event_id": 1, "episode": 7}, {"event_id": 2, "episode": 8}])
    pairs, _ = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    assert sorted(pairs["control_event_id"]) == [2]


def test_unmatched_phantoms_are_reported_not_dropped() -> None:
    """Every phantom leaves the matcher either as a stratum or as a labelled drop."""
    phantom = make_events(
        [
            {"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.30},
            {"event_id": 1, "episode": 1, "eef_max_displacement_after_m": 0.90},  # far away
            {"event_id": 2, "episode": 2, "task": "taskZ"},  # no control in that task
        ]
    )
    control = make_events([{"event_id": 3, "episode": 3, "eef_max_displacement_after_m": 0.31}])
    pairs, unmatched = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=5
    )
    accounted = set(pairs["phantom_event_id"]) | set(unmatched["event_id"])
    assert accounted == {0, 1, 2}
    assert len(pairs["phantom_event_id"].unique()) + len(unmatched) == len(phantom)
    reasons = dict(zip(unmatched["event_id"], unmatched["drop_reason"]))
    assert reasons == {1: "caliper_empty", 2: "no_control_in_task"}


def test_matcher_caps_controls_and_takes_the_nearest() -> None:
    phantom = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {"event_id": 10 + k, "episode": 10 + k, "eef_max_displacement_after_m": 0.300 + 0.001 * k}
            for k in range(1, 9)
        ]
    )
    pairs, _ = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=3
    )
    assert len(pairs) == 3
    assert sorted(pairs["control_event_id"]) == [11, 12, 13]
    assert list(pairs.sort_values("control_rank")["control_rank"]) == [0, 1, 2]


def test_matcher_is_deterministic_under_ties() -> None:
    """Equidistant candidates are broken by event_id, so reruns agree."""
    phantom = make_events([{"event_id": 0, "episode": 0, "eef_max_displacement_after_m": 0.300}])
    control = make_events(
        [
            {"event_id": 5, "episode": 5, "eef_max_displacement_after_m": 0.310},
            {"event_id": 4, "episode": 4, "eef_max_displacement_after_m": 0.310},
            {"event_id": 6, "episode": 6, "eef_max_displacement_after_m": 0.310},
        ]
    )
    first, _ = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    second, _ = dor.match_events(
        phantom.iloc[::-1], control.iloc[::-1], eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    assert list(first.sort_values("control_rank")["control_event_id"]) == [4, 5]
    assert list(second.sort_values("control_rank")["control_event_id"]) == [4, 5]


def test_matcher_rejects_a_nonpositive_caliper() -> None:
    phantom = make_events([{"event_id": 0, "episode": 0}])
    control = make_events([{"event_id": 1, "episode": 1}])
    with pytest.raises(ValueError):
        dor.match_events(phantom, control, eef_caliper=0.0, query_caliper=3, max_controls=5)
    with pytest.raises(ValueError):
        dor.match_events(phantom, control, eef_caliper=0.05, query_caliper=0, max_controls=5)


def test_controls_may_be_reused_across_strata_but_not_inside_one() -> None:
    phantom = make_events([{"event_id": 0, "episode": 0}, {"event_id": 1, "episode": 1}])
    control = make_events([{"event_id": 2, "episode": 2}, {"event_id": 3, "episode": 3}])
    pairs, unmatched = dor.match_events(
        phantom, control, eef_caliper=0.05, query_caliper=3, max_controls=2
    )
    assert unmatched.empty
    assert len(pairs) == 4  # both phantoms keep both controls
    for _, stratum in pairs.groupby("stratum_id"):
        assert stratum["control_event_id"].is_unique


# --------------------------------------------------------------------------
# matched-pair AUC, answers known by hand
# --------------------------------------------------------------------------


def test_case_score_is_one_when_the_case_beats_every_control() -> None:
    assert dor.stratum_case_score(10.0, np.array([1.0, 2.0, 3.0])) == 1.0


def test_case_score_is_zero_when_the_case_loses_to_every_control() -> None:
    assert dor.stratum_case_score(0.0, np.array([1.0, 2.0, 3.0])) == 0.0


def test_exact_ties_score_one_half() -> None:
    """Float16 storage makes exact ties common; they must count as no evidence."""
    assert dor.stratum_case_score(2.0, np.array([2.0, 2.0])) == 0.5
    assert dor.stratum_case_score(2.0, np.array([1.0, 2.0])) == 0.75


def test_case_score_ignores_missing_controls() -> None:
    assert dor.stratum_case_score(5.0, np.array([1.0, np.nan])) == 1.0
    assert np.isnan(dor.stratum_case_score(np.nan, np.array([1.0, 2.0])))
    assert np.isnan(dor.stratum_case_score(5.0, np.array([np.nan, np.nan])))


def test_every_stratum_carries_equal_weight() -> None:
    """A big stratum must not outvote a small one.

    Stratum A: case beats its single control            -> 1.0
    Stratum B: case loses to all nine of its controls   -> 0.0
    Equal weight per stratum gives 0.5.  Equal weight per *pair* would give
    1/10 = 0.1, which is the failure this test rules out.
    """
    a = dor.stratum_case_score(1.0, np.array([0.0]))
    b = dor.stratum_case_score(0.0, np.arange(1.0, 10.0))
    assert a == 1.0 and b == 0.0
    assert np.mean([a, b]) == 0.5


def test_promotion_scores_average_to_one_half_exactly() -> None:
    """The reason the null cannot be a leave-one-out average.

    Promoting every control in turn and averaging is 0.5 in *every* stratum, by
    the antisymmetry of the pairwise comparison.  A null built that way would be
    a constant with no spread, and comparing an AUC against it would be
    comparing against 0.5 by another name.  The informative null therefore
    promotes exactly one control, which is what the pipeline does.
    """
    for controls in (
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([4.0, 1.0, 3.0, 2.0]),
        np.array([0.1, 0.1, 0.9]),  # with ties
        np.array([7.0, 7.0, 7.0, 7.0]),  # all tied
    ):
        scores = dor.stratum_promotion_scores(controls)
        assert np.nanmean(scores) == pytest.approx(0.5)
    # ... and a single promotion is not 0.5, which is where the spread lives
    assert dor.stratum_promotion_scores(np.array([1.0, 2.0, 3.0, 4.0]))[0] == pytest.approx(0.0)


def test_null_needs_at_least_two_controls() -> None:
    assert np.all(np.isnan(dor.stratum_promotion_scores(np.array([1.0]))))
    assert dor.stratum_promotion_scores(np.array([])).size == 0
    partial = dor.stratum_promotion_scores(np.array([1.0, np.nan]))
    assert np.all(np.isnan(partial))


def test_promotion_is_the_case_estimator_with_a_control_standing_in() -> None:
    """The null is the same estimator, with a coupled event in the case's seat."""
    controls = np.array([0.2, 0.7, 0.9, 0.4])
    scores = dor.stratum_promotion_scores(controls)
    for position in range(len(controls)):
        rest = np.delete(controls, position)
        assert scores[position] == pytest.approx(
            dor.stratum_case_score(controls[position], rest)
        )


def test_promotion_skips_missing_controls() -> None:
    scores = dor.stratum_promotion_scores(np.array([1.0, np.nan, 3.0, 5.0]))
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
    case_score, promotion = dor.stratum_scores(case, controls)
    for cell in range(n_cell):
        expected = dor.stratum_case_score(case[cell], controls[:, cell])
        if np.isnan(expected):
            assert np.isnan(case_score[cell])
        else:
            assert case_score[cell] == pytest.approx(expected)
        expected_promotion = dor.stratum_promotion_scores(controls[:, cell])
        assert np.allclose(promotion[:, cell], expected_promotion, equal_nan=True)


def test_stratum_scores_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dor.stratum_scores(np.zeros(3), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        dor.stratum_scores(np.zeros(3), np.zeros(3))


def test_synthetic_separation_recovers_a_known_auc() -> None:
    """Cases uniformly shifted above their controls give AUC 1; below give 0.

    A half-shift construction pins an intermediate value: with controls at
    ``k`` and ``k + 2`` and the case at ``k + 1``, every stratum scores 0.5.
    """
    rng = np.random.default_rng(0)
    high = [dor.stratum_case_score(10.0 + k, rng.normal(0, 1, 4)) for k in range(20)]
    low = [dor.stratum_case_score(-10.0 - k, rng.normal(0, 1, 4)) for k in range(20)]
    middle = [
        dor.stratum_case_score(float(k) + 1.0, np.array([float(k), float(k) + 2.0]))
        for k in range(20)
    ]
    assert np.mean(high) == 1.0
    assert np.mean(low) == 0.0
    assert np.mean(middle) == 0.5


# --------------------------------------------------------------------------
# clustered bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_resamples_whole_tasks() -> None:
    """Strata of one task move together; the drawn weight is a whole multiple."""
    cluster = np.array(["t1", "t1", "t1", "t2", "t3"])
    weights = dor.bootstrap_weights(cluster, draws=200, rng=np.random.default_rng(0))
    assert weights.shape == (200, 5)
    assert np.all(weights[:, 0] == weights[:, 1])
    assert np.all(weights[:, 1] == weights[:, 2])
    assert np.all(weights.sum(axis=1) > 0)
    # each draw picks n_cluster clusters with replacement
    per_draw = weights[:, [0, 3, 4]].sum(axis=1)
    assert np.all(per_draw == 3)


def test_bootstrap_of_a_constant_is_that_constant() -> None:
    cluster = np.array(["t1", "t2", "t3", "t4"])
    weights = dor.bootstrap_weights(cluster, draws=100, rng=np.random.default_rng(1))
    means = dor.weighted_means(weights, np.full(4, 0.7))
    assert np.allclose(means, 0.7)
    assert dor.percentile_interval(means) == pytest.approx((0.7, 0.7))


def test_weighted_means_skips_missing_strata() -> None:
    weights = np.ones((3, 4))
    means = dor.weighted_means(weights, np.array([1.0, np.nan, 3.0, np.nan]))
    assert np.allclose(means, 2.0)
    empty = dor.weighted_means(weights, np.full(4, np.nan))
    assert np.all(np.isnan(empty))


def test_one_task_bootstrap_has_no_between_task_variance() -> None:
    """With a single task the interval collapses; that is the honest answer."""
    weights = dor.bootstrap_weights(np.array(["t1", "t1"]), 100, np.random.default_rng(2))
    means = dor.weighted_means(weights, np.array([0.2, 0.8]))
    assert np.allclose(means, 0.5)


def test_bh_is_monotone_and_never_shrinks_a_p_value() -> None:
    """336 cells per grid; BH keeps the headline count from being read as 336 tests."""
    values = np.array([0.001, 0.01, 0.02, 0.2, 0.9])
    adjusted = dor.bh_adjust(values)
    assert np.all(adjusted >= values - 1e-12)
    assert np.all(np.diff(adjusted[np.argsort(values)]) >= -1e-12)
    assert np.all(adjusted <= 1.0)
    assert adjusted[0] == pytest.approx(0.005)


def test_bh_leaves_a_uniform_field_unrejected() -> None:
    """A grid of honest nulls should survive with almost nothing below 0.05."""
    uniform = (np.arange(1, 337) - 0.5) / 336
    assert int((dor.bh_adjust(uniform) < 0.05).sum()) == 0


def test_bh_passes_missing_values_through() -> None:
    adjusted = dor.bh_adjust(np.array([0.01, np.nan, 0.5]))
    assert np.isnan(adjusted[1])
    assert np.isfinite(adjusted[0]) and np.isfinite(adjusted[2])


# --------------------------------------------------------------------------
# routing readout
# --------------------------------------------------------------------------


def uniform_route() -> np.ndarray:
    return np.full(
        (len(dor.LAYER_NAMES), dor.N_TOKEN, dor.N_EXPERTS), 1.0 / dor.N_EXPERTS, np.float32
    )


def onehot_route(expert: int = 0) -> np.ndarray:
    route = np.zeros((len(dor.LAYER_NAMES), dor.N_TOKEN, dor.N_EXPERTS), np.float32)
    route[:, :, expert] = 1.0
    return route


def test_gate_entropy_spans_zero_to_one() -> None:
    assert dor.gate_entropy(uniform_route()) == pytest.approx(1.0)
    assert dor.gate_entropy(onehot_route()) == pytest.approx(0.0)


def test_role_reduce_separates_state_from_action() -> None:
    per_token = np.zeros((len(dor.LAYER_NAMES), dor.N_TOKEN), np.float32)
    per_token[:, 0] = 1.0  # state token
    per_token[:, 1:] = 0.5  # action tokens
    assert np.allclose(dor.role_reduce(per_token, "state"), 1.0)
    assert np.allclose(dor.role_reduce(per_token, "action"), 0.5)
    with pytest.raises(ValueError):
        dor.role_reduce(per_token, "everything")


def test_features_are_zero_against_an_identical_reference() -> None:
    route = onehot_route(3)
    block = dor.event_features(route, route)
    hell = block[dor.METRICS.index("hellinger_to_pre")]
    change = block[dor.METRICS.index("gate_entropy_change")]
    assert np.allclose(hell, 0.0)
    assert np.allclose(change, 0.0)


def test_features_are_one_against_a_disjoint_reference() -> None:
    block = dor.event_features(onehot_route(0), onehot_route(1))
    hell = block[dor.METRICS.index("hellinger_to_pre")]
    assert np.allclose(hell, 1.0)


def test_features_are_nan_when_the_query_is_missing() -> None:
    missing = np.full_like(uniform_route(), np.nan)
    assert np.all(np.isnan(dor.event_features(missing, uniform_route())))
    assert np.all(np.isnan(dor.event_features(uniform_route(), missing)))


def test_features_reject_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        dor.event_features(np.zeros((8, 10, 32), np.float32), uniform_route())


def test_relative_query_grid_matches_the_specification() -> None:
    assert dor.RELATIVE_QUERIES == (-1, 0, 1, 2, 3, 4, 5)
    assert dor.REFERENCE_RELATIVE == -2
    assert dor.FINAL_FLOW == 9
    assert dor.LAYER_NAMES == ("L2", "L3", "L4", "L5", "L12", "L13", "L14", "L15")
