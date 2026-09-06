"""Unit tests for the reading-reliability helpers.

The audit's whole claim is that ``H_gate``, ``H_usage``, ``D_time`` and
``hard_churn`` are four different readings of one simplex, not four names for
one number.  Two constructions decide that on synthetic data, before any corpus
statistic is allowed to speak:

* a **uniform-route sequence** -- maximal ``H_gate``, zero ``D_time``, and yet
  a hard top-4 identity that churns as soon as the values are jittered by the
  float16 storage ulp;
* an **A -> B -> C -> A cycle**, one expert per step -- minimal ``H_gate``,
  high ``H_usage``, maximal ``D_time``, and perfectly predictable.

A third construction (a *static* one-hot route) shares ``H_gate = 0`` with the
cycle but has ``D_time = 0``, which is what rules out any function from
``H_gate`` to ``D_time``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

import audit_reading_reliability as arr  # noqa: E402


N_EXPERTS = arr.N_EXPERTS
LOG_N = float(np.log(N_EXPERTS))


# --------------------------------------------------------------------------
# constructions
# --------------------------------------------------------------------------


def uniform_sequence(n_query: int = 8) -> np.ndarray:
    """``[Q, 1, 32]`` perfectly uniform routes."""
    return np.full((n_query, 1, N_EXPERTS), 1.0 / N_EXPERTS, dtype=np.float32)


def cycle_sequence(n_query: int = 9) -> np.ndarray:
    """``[Q, 1, 32]`` one-hot routes cycling over experts 0 -> 1 -> 2 -> 0."""
    routes = np.zeros((n_query, 1, N_EXPERTS), dtype=np.float32)
    for query in range(n_query):
        routes[query, 0, query % 3] = 1.0
    return routes


def block_cycle_sequence(n_query: int = 9) -> np.ndarray:
    """``[Q, 1, 32]`` routes cycling three disjoint blocks of four experts."""
    routes = np.full((n_query, 1, N_EXPERTS), 0.02 / (N_EXPERTS - 4), dtype=np.float32)
    for query in range(n_query):
        block = (query % 3) * 4
        routes[query, 0, block : block + 4] = 0.98 / 4.0
    return arr.normalise(routes)


def static_one_hot(n_query: int = 8) -> np.ndarray:
    """``[Q, 1, 32]`` the same one-hot route at every step."""
    routes = np.zeros((n_query, 1, N_EXPERTS), dtype=np.float32)
    routes[:, 0, 7] = 1.0
    return routes


# --------------------------------------------------------------------------
# entropy normalisation
# --------------------------------------------------------------------------


def test_gate_entropy_uniform_is_one():
    assert arr.gate_entropy(np.full(N_EXPERTS, 1.0 / N_EXPERTS)) == pytest.approx(1.0, abs=1e-6)


def test_gate_entropy_one_hot_is_zero():
    probability = np.zeros(N_EXPERTS, dtype=np.float32)
    probability[3] = 1.0
    assert arr.gate_entropy(probability) == pytest.approx(0.0, abs=1e-6)


def test_gate_entropy_two_point_is_log2_over_log32():
    probability = np.zeros(N_EXPERTS, dtype=np.float32)
    probability[[5, 11]] = 0.5
    assert arr.gate_entropy(probability) == pytest.approx(np.log(2.0) / LOG_N, abs=1e-6)


def test_gate_entropy_is_invariant_to_input_scale():
    """The normalisation must not depend on the incoming sum being exactly 1."""
    rng = np.random.default_rng(0)
    probability = rng.random(N_EXPERTS).astype(np.float32)
    normalised = probability / probability.sum()
    assert arr.gate_entropy(probability * 7.0) == pytest.approx(
        float(arr.gate_entropy(normalised)), abs=1e-6
    )


def test_gate_entropy_lies_in_unit_interval():
    rng = np.random.default_rng(1)
    probability = rng.random((64, N_EXPERTS)).astype(np.float32)
    values = arr.gate_entropy(probability)
    assert values.shape == (64,)
    assert np.all(values >= 0.0) and np.all(values <= 1.0 + 1e-6)


def test_gate_entropy_rejects_degenerate_axis():
    with pytest.raises(ValueError):
        arr.gate_entropy(np.ones((4, 1)))


# --------------------------------------------------------------------------
# usage entropy over a trailing window
# --------------------------------------------------------------------------


def test_trailing_mean_is_nan_before_the_window_fills():
    routes = cycle_sequence(6)
    averaged = arr.trailing_mean(routes, 4)
    assert np.isnan(averaged[:3]).all()
    assert np.isfinite(averaged[3:]).all()
    np.testing.assert_allclose(averaged[3, 0], routes[:4, 0].mean(axis=0), atol=1e-7)


def test_trailing_mean_rejects_non_positive_window():
    with pytest.raises(ValueError):
        arr.trailing_mean(cycle_sequence(6), 0)


def test_usage_entropy_of_uniform_sequence_is_one():
    values = arr.usage_entropy(uniform_sequence(8), 4)
    assert np.isnan(values[:3]).all()
    np.testing.assert_allclose(values[3:], 1.0, atol=1e-6)


def test_usage_entropy_of_static_one_hot_is_zero():
    values = arr.usage_entropy(static_one_hot(8), 4)
    np.testing.assert_allclose(values[3:], 0.0, atol=1e-6)


def test_usage_entropy_of_cycle_is_three_expert_mixture():
    """Window [A, B, C, A] averages to (1/2, 1/4, 1/4): 1.0397 nats / log 32 = 0.3."""
    values = arr.usage_entropy(cycle_sequence(9), 4)
    expected = -(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)) / LOG_N
    assert expected == pytest.approx(0.3, abs=1e-9)
    np.testing.assert_allclose(values[3:], expected, atol=1e-6)


def test_usage_entropy_exceeds_gate_entropy_on_the_cycle():
    """The separation the audit exists to test: same route, two different reads."""
    routes = cycle_sequence(9)
    gate = arr.gate_entropy(routes)
    usage = arr.usage_entropy(routes, 4)
    np.testing.assert_allclose(gate, 0.0, atol=1e-6)
    assert np.nanmin(usage[3:]) > np.max(gate) + 0.25


# --------------------------------------------------------------------------
# adjacent-query movement
# --------------------------------------------------------------------------


def test_hellinger_is_zero_for_identical_and_one_for_disjoint():
    left = np.zeros(N_EXPERTS, dtype=np.float32)
    left[0] = 1.0
    right = np.zeros(N_EXPERTS, dtype=np.float32)
    right[1] = 1.0
    assert arr.hellinger(left, left) == pytest.approx(0.0, abs=1e-6)
    assert arr.hellinger(left, right) == pytest.approx(1.0, abs=1e-6)


def test_hellinger_is_symmetric_and_bounded():
    rng = np.random.default_rng(2)
    left = rng.random((32, N_EXPERTS)).astype(np.float32)
    right = rng.random((32, N_EXPERTS)).astype(np.float32)
    forward = arr.hellinger(left, right)
    backward = arr.hellinger(right, left)
    np.testing.assert_allclose(forward, backward, atol=1e-6)
    assert np.all(forward >= 0.0) and np.all(forward <= 1.0)


def test_uniform_sequence_has_zero_adjacent_distance():
    values = arr.adjacent_hellinger(uniform_sequence(8))
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1:], 0.0, atol=1e-6)


def test_cycle_has_maximal_adjacent_distance():
    values = arr.adjacent_hellinger(cycle_sequence(9))
    np.testing.assert_allclose(values[1:], 1.0, atol=1e-6)


def test_gate_entropy_does_not_determine_adjacent_distance():
    """Cycle and static one-hot share ``H_gate = 0`` and differ in ``D_time``."""
    cycle_gate = arr.gate_entropy(cycle_sequence(8))
    static_gate = arr.gate_entropy(static_one_hot(8))
    np.testing.assert_allclose(cycle_gate, static_gate, atol=1e-6)
    cycle_move = arr.adjacent_hellinger(cycle_sequence(8))[1:]
    static_move = arr.adjacent_hellinger(static_one_hot(8))[1:]
    np.testing.assert_allclose(cycle_move, 1.0, atol=1e-6)
    np.testing.assert_allclose(static_move, 0.0, atol=1e-6)


# --------------------------------------------------------------------------
# hard top-k identity
# --------------------------------------------------------------------------


def test_top_k_mask_selects_exactly_k_and_the_largest():
    rng = np.random.default_rng(3)
    probability = rng.random((16, N_EXPERTS)).astype(np.float32)
    mask = arr.top_k_mask(probability, 4)
    assert mask.sum(axis=-1).tolist() == [4] * 16
    for row, selected in zip(probability, mask):
        assert row[selected].min() >= row[~selected].max()


def test_top_k_mask_breaks_ties_towards_the_lower_index():
    probability = np.full(N_EXPERTS, 1.0 / N_EXPERTS, dtype=np.float32)
    mask = arr.top_k_mask(probability, 4)
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3]


def test_top_k_mask_rejects_out_of_range_k():
    probability = np.full(N_EXPERTS, 1.0 / N_EXPERTS, dtype=np.float32)
    with pytest.raises(ValueError):
        arr.top_k_mask(probability, 0)
    with pytest.raises(ValueError):
        arr.top_k_mask(probability, N_EXPERTS + 1)


def test_churn_between_is_zero_for_equal_and_one_for_disjoint():
    first = np.zeros(N_EXPERTS, dtype=bool)
    first[:4] = True
    second = np.zeros(N_EXPERTS, dtype=bool)
    second[4:8] = True
    assert arr.churn_between(first, first, 4) == pytest.approx(0.0)
    assert arr.churn_between(first, second, 4) == pytest.approx(1.0)
    half = np.zeros(N_EXPERTS, dtype=bool)
    half[2:6] = True
    assert arr.churn_between(first, half, 4) == pytest.approx(0.5)


def test_uniform_sequence_has_zero_hard_churn_under_a_deterministic_tie_break():
    values = arr.hard_churn(uniform_sequence(8), 4)
    assert np.isnan(values[0]).all()
    np.testing.assert_allclose(values[1:], 0.0, atol=1e-6)


def test_uniform_sequence_churns_hard_once_jittered_by_one_float16_ulp():
    """Maximal ``H_gate``, ~zero ``D_time``, and yet the top-4 identity churns.

    This is the failure mode the corpus audit quantifies: on a near-uniform
    simplex the hard top-k identity is decided by storage noise, not by routing.
    """
    rng = np.random.default_rng(4)
    routes = uniform_sequence(64)
    ulp = arr.float16_ulp(routes)
    jittered = routes + (rng.random(routes.shape, dtype=np.float32) - np.float32(0.5)) * ulp
    churn = arr.hard_churn(jittered, 4)
    movement = arr.adjacent_hellinger(jittered)
    assert np.nanmean(churn) > 0.5
    assert np.nanmax(movement) < 0.01
    assert np.min(arr.gate_entropy(jittered)) > 0.999


def test_one_hot_cycle_churns_the_top_one_every_step():
    values = arr.hard_churn(cycle_sequence(9), 1)
    np.testing.assert_allclose(values[1:], 1.0, atol=1e-6)


def test_block_cycle_churns_the_whole_top_four_every_step():
    routes = block_cycle_sequence(9)
    churn = arr.hard_churn(routes, 4)
    gate = arr.gate_entropy(routes)
    usage = arr.usage_entropy(routes, 4)
    np.testing.assert_allclose(churn[1:], 1.0, atol=1e-6)
    assert np.max(gate) < 0.5
    assert np.nanmin(usage[3:]) > np.max(gate)


# --------------------------------------------------------------------------
# tie gap and the float16 ulp
# --------------------------------------------------------------------------


def test_tie_gap_reads_the_fourth_minus_fifth_order_statistic():
    probability = np.zeros(N_EXPERTS, dtype=np.float32)
    probability[:6] = [0.30, 0.25, 0.20, 0.15, 0.06, 0.04]
    assert arr.tie_gap(probability, 4) == pytest.approx(0.15 - 0.06, abs=1e-6)
    assert arr.tie_gap(probability, 1) == pytest.approx(0.30 - 0.25, abs=1e-6)


def test_tie_gap_is_zero_on_an_exact_tie():
    probability = np.full(N_EXPERTS, 1.0 / N_EXPERTS, dtype=np.float32)
    assert arr.tie_gap(probability, 4) == pytest.approx(0.0, abs=1e-9)


def test_tie_gap_is_non_negative_and_order_invariant():
    rng = np.random.default_rng(5)
    probability = arr.normalise(rng.random((128, N_EXPERTS)).astype(np.float32))
    gap = arr.tie_gap(probability, 4)
    assert np.all(gap >= 0.0)
    permuted = probability[:, rng.permutation(N_EXPERTS)]
    np.testing.assert_allclose(arr.tie_gap(permuted, 4), gap, atol=1e-7)


def test_tie_gap_rejects_k_at_or_past_the_last_order_statistic():
    probability = np.full(N_EXPERTS, 1.0 / N_EXPERTS, dtype=np.float32)
    with pytest.raises(ValueError):
        arr.tie_gap(probability, N_EXPERTS)
    with pytest.raises(ValueError):
        arr.tie_gap(probability, 0)


def test_float16_ulp_matches_nextafter_on_the_float16_grid():
    for value in (1.0 / N_EXPERTS, 0.04, 0.5, 1.0):
        stored = np.float16(value)
        expected = np.float32(np.nextafter(stored, np.float16(np.inf)) - stored)
        assert arr.float16_ulp(np.float32(value)) == pytest.approx(expected, rel=1e-6)


def test_float16_ulp_at_the_uniform_route_is_two_to_the_minus_fifteen():
    assert arr.float16_ulp(np.float32(1.0 / N_EXPERTS)) == pytest.approx(2.0**-15, rel=1e-9)


def test_ulp_ratio_scales_the_gap_by_storage_resolution():
    gap = np.array([0.0, 2.0**-15, 10 * 2.0**-15], dtype=np.float32)
    ulp = np.full(3, 2.0**-15, dtype=np.float32)
    np.testing.assert_allclose(arr.ulp_ratio(gap, ulp), [0.0, 1.0, 10.0], rtol=1e-6)


def test_ulp_ratio_rejects_misaligned_or_non_positive_input():
    with pytest.raises(ValueError):
        arr.ulp_ratio(np.zeros(3, dtype=np.float32), np.zeros(4, dtype=np.float32))
    with pytest.raises(ValueError):
        arr.ulp_ratio(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))


def test_a_sub_ulp_gap_means_the_two_masses_are_the_same_or_neighbouring_codes():
    """``gap < 1 ulp`` is not "nearly tied"; the two masses are indistinguishable.

    The implication is one-way, not an equality.  When ``p_(4)`` sits just above
    a binade boundary its ulp is twice the ulp below it, so two *adjacent*
    float16 codes can also land under one ulp.  Both directions are asserted
    here so the corpus table's ``frac < 1 ulp`` column is read correctly: it
    bounds "indistinguishable in storage", and the exact-tie column inside it
    is the subset where the top-4 set is not determined at all.
    """
    rng = np.random.default_rng(6)
    stored = rng.random((256, N_EXPERTS)).astype(np.float16)
    routes = arr.normalise(stored.astype(np.float32))
    ordered = np.sort(routes, axis=-1)[:, ::-1]
    ratio = arr.ulp_ratio(arr.tie_gap(routes, 4), arr.float16_ulp(ordered[:, 3]))
    below = ratio < 1.0
    ordered_stored = np.sort(stored, axis=-1)[:, ::-1]
    tied = ordered_stored[:, 3] == ordered_stored[:, 4]
    assert np.all(ratio[tied] == 0.0)
    neighbouring = ordered_stored[:, 3] <= np.nextafter(
        ordered_stored[:, 4], np.float16(np.inf)
    )
    assert np.all(neighbouring[below])
    assert below.sum() >= tied.sum()


# --------------------------------------------------------------------------
# rank statistics
# --------------------------------------------------------------------------


def test_average_rank_resolves_ties_to_the_block_mean():
    np.testing.assert_allclose(
        arr.average_rank(np.array([10.0, 20.0, 20.0, 30.0])), [1.0, 2.5, 2.5, 4.0]
    )
    np.testing.assert_allclose(arr.average_rank(np.array([5.0, 5.0, 5.0])), [2.0, 2.0, 2.0])


def test_average_rank_handles_the_empty_input():
    assert arr.average_rank(np.array([])).size == 0


def test_spearman_is_one_for_a_monotone_map_and_minus_one_for_an_antitone_one():
    x = np.arange(50, dtype=np.float64)
    columns = {"a": x, "b": np.exp(x / 10.0), "c": -x**3}
    matrix = arr.spearman_matrix(columns)
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-9)
    assert matrix[0, 2] == pytest.approx(-1.0, abs=1e-9)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)


def test_spearman_matches_a_hand_computed_tied_case():
    """rho for [1,2,3,4] against [1,1,2,2]: ranks [1,2,3,4] vs [1.5,1.5,3.5,3.5]."""
    matrix = arr.spearman_matrix(
        {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([1.0, 1.0, 2.0, 2.0])}
    )
    assert matrix[0, 1] == pytest.approx(0.894427191, abs=1e-8)


def test_spearman_drops_rows_that_are_not_finite_everywhere():
    matrix = arr.spearman_matrix(
        {
            "a": np.array([1.0, 2.0, np.nan, 4.0, 5.0]),
            "b": np.array([1.0, 2.0, 100.0, 4.0, 5.0]),
        }
    )
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-9)


def test_spearman_returns_nan_for_a_constant_column():
    matrix = arr.spearman_matrix(
        {"a": np.arange(10.0), "b": np.ones(10), "c": np.arange(10.0) ** 2}
    )
    assert np.isnan(matrix[0, 1])
    assert matrix[0, 2] == pytest.approx(1.0, abs=1e-9)


def test_fisher_mean_averages_in_the_z_domain():
    values = np.array([0.5, 0.9])
    expected = np.tanh((np.arctanh(0.5) + np.arctanh(0.9)) / 2.0)
    assert arr.fisher_mean(values) == pytest.approx(expected, abs=1e-9)
    assert np.isnan(arr.fisher_mean(np.array([np.nan, np.nan])))


def test_conditional_table_bins_by_quantile_and_preserves_the_count():
    driver = np.arange(100.0)
    response = driver * 2.0
    rows = arr.conditional_table(driver, response, 10)
    assert sum(row["n"] for row in rows) == 100
    means = [row["mean_response"] for row in rows]
    assert means == sorted(means)


def test_quantile_summary_reports_nan_for_an_empty_sample():
    record = arr.quantile_summary(np.array([np.nan, np.nan]), "v")
    assert record["v_n"] == 0
    assert np.isnan(record["v_p50"])


# --------------------------------------------------------------------------
# episode-level assembly
# --------------------------------------------------------------------------


def synthetic_block(n_query: int = 6, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    raw = rng.random((n_query,) + arr.RAW_SHAPE).astype(np.float32)
    raw = (raw / raw.sum(axis=-1, keepdims=True)).astype(np.float16)
    ids = np.argsort(-raw.astype(np.float32), axis=-1, kind="stable")[..., : arr.TOP_K]
    return raw, ids.astype(np.uint8)


def test_episode_measurements_returns_aligned_shapes_and_valid_ranges():
    raw, ids = synthetic_block()
    measured = arr.episode_measurements(raw, ids, np.random.default_rng(8), draws=3)
    grid = (6, len(arr.LAYER_NAMES), arr.N_ACTION_TOKENS)
    for key in ("H_gate", "H_usage", "D_time", "hard_churn", "tie_gap", "tie_gap_ulp"):
        assert measured[key].shape == grid, key
    assert measured["state_dev"].shape == (6, len(arr.LAYER_NAMES))
    assert measured["action_dev"].shape == (6, len(arr.LAYER_NAMES))
    assert np.isnan(measured["D_time"][0]).all()
    assert np.isnan(measured["hard_churn"][0]).all()
    assert np.isnan(measured["H_usage"][: arr.USAGE_WINDOW - 1]).all()
    assert np.all((measured["H_gate"] >= 0.0) & (measured["H_gate"] <= 1.0 + 1e-6))
    assert np.all(measured["tie_gap"] >= 0.0)
    assert np.all(np.isin(measured["hard_churn"][1:], [0.0, 0.25, 0.5, 0.75, 1.0]))


def test_episode_measurements_reproduces_consistent_stored_expert_ids():
    raw, ids = synthetic_block()
    measured = arr.episode_measurements(raw, ids, np.random.default_rng(9), draws=2)
    assert measured["stored_match"].all()
    np.testing.assert_allclose(measured["stored_overlap"], float(arr.TOP_K))


def test_episode_measurements_flags_disagreeing_stored_expert_ids():
    raw, ids = synthetic_block()
    broken = ids.copy()
    broken[0, 0, arr.FINAL_FLOW, 1, :] = np.array([31, 30, 29, 28], dtype=np.uint8)
    measured = arr.episode_measurements(raw, broken, np.random.default_rng(10), draws=2)
    assert not measured["stored_match"][0, 0, 0]
    assert measured["stored_match"].mean() > 0.9


def test_episode_measurements_rejects_duplicate_stored_expert_ids():
    raw, ids = synthetic_block()
    broken = ids.copy()
    broken[0, 0, arr.FINAL_FLOW, 1, :] = np.uint8(5)
    with pytest.raises(ValueError):
        arr.episode_measurements(raw, broken, np.random.default_rng(11), draws=1)


def test_episode_measurements_rejects_a_wrong_shaped_block():
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError):
        arr.episode_measurements(
            rng.random((4, 8, 10, 11, 16)).astype(np.float16),
            np.zeros((4, 8, 10, 11, 4), dtype=np.uint8),
            rng,
            draws=1,
        )


def test_episode_measurements_perturbation_never_reports_a_negative_rate():
    raw, ids = synthetic_block()
    measured = arr.episode_measurements(raw, ids, np.random.default_rng(13), draws=5)
    assert np.all(measured["ulp_set_changed"] >= 0.0)
    assert np.all(measured["ulp_set_changed"] <= 1.0)
    assert np.all(np.isnan(measured["churn_perturbed"][0]))
    finite = measured["churn_perturbed"][1:]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert np.all((measured["null_churn"] >= 0.0) & (measured["null_churn"] <= 1.0))


def test_storage_noise_null_churn_vanishes_when_no_gap_is_sub_ulp():
    """The null is churn manufactured by storage noise alone, so a route with
    a top-4 boundary far above the ulp must produce none of it."""
    rng = np.random.default_rng(15)
    raw = np.zeros((4,) + arr.RAW_SHAPE, dtype=np.float32)
    raw[..., :4] = 0.24
    raw[..., 4:] = 0.04 / (N_EXPERTS - 4)
    stored = raw.astype(np.float16)
    ids = np.tile(np.arange(4, dtype=np.uint8), stored.shape[:-1] + (1,))
    measured = arr.episode_measurements(stored, ids, rng, draws=6)
    np.testing.assert_allclose(measured["null_churn"], 0.0, atol=1e-9)
    np.testing.assert_allclose(measured["ulp_set_changed"], 0.0, atol=1e-9)


def test_storage_noise_null_churn_is_positive_on_an_exactly_tied_route():
    """A uniform route has ``p_(4) = p_(5)``: its top-4 set is undetermined."""
    rng = np.random.default_rng(16)
    stored = np.full((4,) + arr.RAW_SHAPE, np.float16(1.0 / N_EXPERTS))
    ids = np.tile(np.arange(4, dtype=np.uint8), stored.shape[:-1] + (1,))
    measured = arr.episode_measurements(stored, ids, rng, draws=8)
    assert measured["null_churn"].mean() > 0.3
    assert np.nanmax(measured["D_time"]) == pytest.approx(0.0, abs=1e-6)


def test_denoising_deviation_is_zero_for_a_flow_constant_block():
    raw, ids = synthetic_block()
    flat = np.repeat(raw[:, :, 0:1], arr.N_FLOW, axis=2)
    ids_flat = np.repeat(ids[:, :, 0:1], arr.N_FLOW, axis=2)
    measured = arr.episode_measurements(flat, ids_flat, np.random.default_rng(14), draws=1)
    np.testing.assert_array_equal(measured["state_dev"], 0.0)
    np.testing.assert_array_equal(measured["action_dev"], 0.0)


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


def test_sample_rows_covers_every_task_and_is_reproducible():
    task_index = np.repeat(np.arange(37), 400)
    first = arr.sample_rows(task_index, 37, 20, seed=20260906)
    second = arr.sample_rows(task_index, 37, 20, seed=20260906)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 37 * 20
    assert len(np.unique(first)) == len(first)
    assert np.bincount(task_index[first], minlength=37).tolist() == [20] * 37


def test_sample_rows_changes_with_the_seed():
    task_index = np.repeat(np.arange(37), 400)
    first = arr.sample_rows(task_index, 37, 20, seed=1)
    second = arr.sample_rows(task_index, 37, 20, seed=2)
    assert not np.array_equal(first, second)


def test_sample_rows_caps_at_the_available_rows():
    task_index = np.repeat(np.arange(3), 5)
    rows = arr.sample_rows(task_index, 3, 20, seed=0)
    assert len(rows) == 15


def test_sample_rows_rejects_an_empty_task():
    task_index = np.array([0, 0, 2, 2])
    with pytest.raises(ValueError):
        arr.sample_rows(task_index, 3, 1, seed=0)
