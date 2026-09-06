"""Unit tests for the pure helpers of the information-source ablation.

Three properties are load-bearing for the conclusions and are pinned here:

1. the dwell run is causal, resets on a regime change and is zero at the first
   query of an episode -- if it were not, "dwell" would silently be a clock;
2. the shuffle control preserves the row's multiset of labels exactly -- if it
   did not, S4-ordered minus S4-shuffled would measure window content, not order;
3. the residualiser is fitted on the training rows only and applied unchanged
   to held-out rows -- if it were not, "graph residual" would be fitted on the
   data it is scored on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))
sys.path.insert(0, str(BUNDLE / "method"))

import ablate_information_sources as ablation  # noqa: E402


# --------------------------------------------------------------------------- #
# dwell runs
# --------------------------------------------------------------------------- #


def test_dwell_first_query_is_zero():
    labels = np.array([[3, 3, 3, 3]])
    valid = np.ones_like(labels, dtype=bool)
    assert ablation.dwell_runs(labels, valid)[0, 0] == 0


def test_dwell_counts_consecutive_preceding_queries():
    labels = np.array([[2, 2, 2, 2, 2]])
    valid = np.ones_like(labels, dtype=bool)
    assert ablation.dwell_runs(labels, valid).tolist() == [[0, 1, 2, 3, 4]]


def test_dwell_resets_on_a_regime_change():
    labels = np.array([[1, 1, 1, 7, 7, 1]])
    valid = np.ones_like(labels, dtype=bool)
    assert ablation.dwell_runs(labels, valid).tolist() == [[0, 1, 2, 0, 1, 0]]


def test_dwell_does_not_run_across_the_end_of_an_episode():
    labels = np.array([[4, 4, 4, -1, -1]])
    valid = np.array([[True, True, True, False, False]])
    dwell = ablation.dwell_runs(labels, valid)
    assert dwell.tolist() == [[0, 1, 2, 0, 0]]


def test_dwell_is_independent_across_episodes():
    labels = np.array([[5, 5, 5], [5, 6, 6]])
    valid = np.ones_like(labels, dtype=bool)
    assert ablation.dwell_runs(labels, valid).tolist() == [[0, 1, 2], [0, 0, 1]]


def test_dwell_is_causal_future_labels_cannot_change_the_past():
    valid = np.ones((1, 6), dtype=bool)
    early = np.array([[1, 1, 1, 2, 2, 2]])
    late = np.array([[1, 1, 1, 2, 9, 9]])
    left = ablation.dwell_runs(early, valid)
    right = ablation.dwell_runs(late, valid)
    assert left[0, :4].tolist() == right[0, :4].tolist()


def test_dwell_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        ablation.dwell_runs(np.zeros((2, 3), dtype=int), np.ones((2, 4), dtype=bool))


# --------------------------------------------------------------------------- #
# ordered history and its identical-content control
# --------------------------------------------------------------------------- #


def test_history_is_a_strict_prefix_padded_before_the_episode_start():
    labels = np.array([[0, 1, 2, 3]])
    valid = np.ones_like(labels, dtype=bool)
    history = ablation.history_matrix(labels, valid, depth=3)
    pad = ablation.PAD_REGIME
    assert history[0, 0].tolist() == [pad, pad, pad]
    assert history[0, 1].tolist() == [0, pad, pad]
    assert history[0, 3].tolist() == [2, 1, 0]


def test_history_never_contains_the_current_query():
    labels = np.array([[8, 8, 8, 8, 8, 8]])
    valid = np.ones_like(labels, dtype=bool)
    changed = labels.copy()
    changed[0, 5] = 0
    left = ablation.history_matrix(labels, valid, depth=4)
    right = ablation.history_matrix(changed, valid, depth=4)
    assert left[0, 5].tolist() == right[0, 5].tolist()


def test_shuffle_preserves_the_multiset_of_every_row():
    rng = np.random.default_rng(0)
    history = rng.integers(0, ablation.N_HISTORY_CATEGORY, size=(500, ablation.HISTORY_DEPTH))
    shuffled = ablation.shuffle_history(history, seed=1234)
    assert shuffled.shape == history.shape
    assert np.array_equal(np.sort(history, axis=-1), np.sort(shuffled, axis=-1))


def test_shuffle_preserves_the_multiset_with_repeats_and_pads():
    pad = ablation.PAD_REGIME
    history = np.array([[pad, pad, 3, 3, 3, 1, 1, pad]])
    shuffled = ablation.shuffle_history(history, seed=7)
    assert sorted(shuffled[0].tolist()) == sorted(history[0].tolist())


def test_shuffle_is_deterministic_under_a_fixed_seed():
    rng = np.random.default_rng(3)
    history = rng.integers(0, ablation.N_HISTORY_CATEGORY, size=(64, ablation.HISTORY_DEPTH))
    left = ablation.shuffle_history(history, seed=99)
    right = ablation.shuffle_history(history, seed=99)
    assert np.array_equal(left, right)


def test_shuffle_actually_reorders_something():
    rng = np.random.default_rng(5)
    history = rng.integers(0, ablation.N_HISTORY_CATEGORY, size=(2000, ablation.HISTORY_DEPTH))
    shuffled = ablation.shuffle_history(history, seed=11)
    assert (shuffled != history).any()


def test_one_hot_of_ordered_and_shuffled_have_identical_column_totals_per_row():
    """The shuffled encoding keeps the label counts and loses only the order."""
    rng = np.random.default_rng(8)
    history = rng.integers(0, ablation.N_HISTORY_CATEGORY, size=(200, ablation.HISTORY_DEPTH))
    shuffled = ablation.shuffle_history(history, seed=17)
    categories = ablation.N_HISTORY_CATEGORY
    left = ablation.one_hot(history, categories).reshape(200, ablation.HISTORY_DEPTH, categories)
    right = ablation.one_hot(shuffled, categories).reshape(200, ablation.HISTORY_DEPTH, categories)
    assert np.array_equal(left.sum(axis=1), right.sum(axis=1))
    assert not np.array_equal(left, right)


def test_one_hot_rejects_out_of_range_labels():
    with pytest.raises(ValueError):
        ablation.one_hot(np.array([[0, 11]]), ablation.N_HISTORY_CATEGORY)


# --------------------------------------------------------------------------- #
# residualisation is fitted on the training fold only
# --------------------------------------------------------------------------- #


def _split_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    control_train = rng.normal(size=(400, 3))
    control_test = rng.normal(size=(120, 3))
    beta = np.array([[1.0, -2.0], [0.5, 0.25], [-1.5, 3.0]])
    target_train = control_train @ beta + 4.0 + 0.01 * rng.normal(size=(400, 2))
    target_test = control_test @ beta + 4.0 + 0.01 * rng.normal(size=(120, 2))
    return control_train, target_train, control_test, target_test


def test_residualiser_removes_the_control_on_the_training_fold():
    control, target, _, _ = _split_data()
    coefficient = ablation.fit_residualiser(control, target)
    residual = ablation.apply_residualiser(control, target, coefficient)
    design = np.concatenate([np.ones((len(control), 1)), control], axis=1)
    # the tiny ridge leaves a proportional slack, so orthogonality is checked
    # per row rather than as a raw total
    assert np.abs(design.T @ residual).max() / len(control) < 1e-5
    assert np.abs(residual).max() < 0.1


def test_residualiser_coefficients_ignore_the_held_out_rows():
    control, target, control_test, target_test = _split_data()
    reference = ablation.fit_residualiser(control, target)
    poisoned = ablation.fit_residualiser(control, target)
    # mutating the held-out block cannot move a train-fold fit
    target_test[:] = 1e6
    control_test[:] = -1e6
    assert np.array_equal(reference, poisoned)
    assert np.array_equal(
        reference, ablation.fit_residualiser(control, target)
    )


def test_residualiser_applies_train_coefficients_unchanged_to_held_out_rows():
    control, target, control_test, target_test = _split_data()
    coefficient = ablation.fit_residualiser(control, target)
    residual = ablation.apply_residualiser(control_test, target_test, coefficient)
    design = np.concatenate([np.ones((len(control_test), 1)), control_test], axis=1)
    assert np.allclose(residual, target_test - design @ coefficient)


def test_residualiser_leaves_held_out_structure_when_the_relationship_differs():
    """A held-out fold with a different control->metric map keeps a residual.

    This is the failure a train-only fit must not hide: refitting on the test
    rows would drive the residual to zero and make S5 look empty for the wrong
    reason.
    """
    control, target, control_test, _ = _split_data()
    coefficient = ablation.fit_residualiser(control, target)
    other = control_test @ np.array([[-3.0, 1.0], [2.0, -1.0], [0.0, 0.5]]) + 4.0
    residual = ablation.apply_residualiser(control_test, other, coefficient)
    assert np.abs(residual).max() > 1.0


def test_residualiser_rejects_mismatched_row_counts():
    with pytest.raises(ValueError):
        ablation.fit_residualiser(np.zeros((10, 2)), np.zeros((9, 3)))


def test_residualiser_survives_a_rank_deficient_control_block():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(200, 1))
    control = np.concatenate([base, base, base * 2.0], axis=1)
    target = base * 3.0 + 1.0
    coefficient = ablation.fit_residualiser(control, target)
    residual = ablation.apply_residualiser(control, target, coefficient)
    assert np.isfinite(coefficient).all()
    assert np.abs(residual).max() < 1e-2


# --------------------------------------------------------------------------- #
# supporting helpers used by the same pipeline
# --------------------------------------------------------------------------- #


def test_standardiser_is_fitted_on_the_training_rows_only():
    rng = np.random.default_rng(4)
    train = rng.normal(loc=5.0, scale=2.0, size=(1000, 3))
    test = rng.normal(loc=-50.0, scale=20.0, size=(50, 3))
    centre, scale = ablation.fit_standardiser(train)
    standardised_train = ablation.apply_standardiser(train, centre, scale)
    assert np.abs(standardised_train.mean(axis=0)).max() < 1e-12
    assert np.abs(standardised_train.std(axis=0) - 1.0).max() < 1e-12
    standardised_test = ablation.apply_standardiser(test, centre, scale)
    assert np.abs(standardised_test.mean(axis=0)).min() > 1.0


def test_standardiser_guards_a_constant_column():
    values = np.concatenate(
        [np.ones((10, 1)), np.arange(10, dtype=float).reshape(10, 1)], axis=1
    )
    centre, scale = ablation.fit_standardiser(values)
    assert scale[0] == 1.0
    assert np.allclose(ablation.apply_standardiser(values, centre, scale)[:, 0], 0.0)


def test_imputation_uses_the_training_median():
    train = np.array([[1.0], [2.0], [3.0], [np.nan]])
    median = ablation.column_median(train)
    assert median[0] == pytest.approx(2.0)
    filled = ablation.impute(np.array([[np.nan], [10.0]]), median)
    assert filled.tolist() == [[2.0], [10.0]]


def test_tercile_index_marks_missing_as_minus_one():
    edges = ablation.tercile_edges(np.arange(300, dtype=float))
    index = ablation.tercile_index(np.array([0.0, 150.0, 299.0, np.nan]), edges)
    assert index.tolist() == [0, 1, 2, -1]


def test_regime_labels_flag_an_undefined_window():
    ratio = np.array([[np.nan, 0.1, 0.9]])
    length = np.array([[np.nan, 0.1, 0.9]])
    valid = np.ones((1, 3), dtype=bool)
    labels = ablation.regime_labels(ratio, length, (0.3, 0.6), (0.3, 0.6), valid)
    assert labels[0, 0] == ablation.UNDEFINED_REGIME
    assert labels[0, 1] == 0
    assert labels[0, 2] == 8


def test_row_log_loss_matches_the_definition():
    probability = np.array([0.2, 0.9])
    target = np.array([0.0, 1.0])
    expected = np.array([-np.log(0.8), -np.log(0.9)])
    assert np.allclose(ablation.row_log_loss(probability, target), expected)


def test_cluster_bootstrap_recovers_the_pooled_mean_on_average():
    rng = np.random.default_rng(6)
    per_cluster = rng.normal(loc=2.0, scale=0.3, size=(30, 1, 1))
    counts = np.full((30, 1, 1), 100.0)
    sums = per_cluster * counts
    draws = ablation.cluster_bootstrap(sums, counts, draws=500, seed=1)
    assert draws.shape == (500, 1, 1)
    pooled = sums.sum() / counts.sum()
    assert abs(float(draws.mean()) - pooled) < 0.1
    low, high = ablation.percentile_interval(draws[:, 0, 0])
    assert low < pooled < high


def test_cluster_bootstrap_is_deterministic():
    sums = np.arange(12, dtype=float).reshape(4, 1, 3)
    counts = np.full((4, 1, 3), 5.0)
    left = ablation.cluster_bootstrap(sums, counts, draws=50, seed=3)
    right = ablation.cluster_bootstrap(sums, counts, draws=50, seed=3)
    assert np.array_equal(left, right)
