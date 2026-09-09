import numpy as np

from analyze_pre_action_signal import (
    _feature_families,
    _permuted_targets,
    _signal_analysis,
    _target_reliability,
)


def test_uniform_routes_have_expected_frozen_descriptors():
    routes = np.full((6, 2, 3, 3, 4), 0.25, dtype=np.float32)
    rounds, manifest = _feature_families(routes)
    assert len(rounds) == 3
    assert manifest["n_action_tokens"] == 2
    assert manifest["combined_dimensions"] == 18
    for entry in rounds:
        assert np.allclose(entry["probability"], 0.25)
        assert np.allclose(entry["entropy"], 1.0)
        assert np.allclose(entry["margin"], 0.0)
        assert np.allclose(entry["token_consensus"], 1.0)
        assert np.allclose(entry["churn"], 0.0)
        assert np.allclose(entry["displacement"], 0.0)


def test_permuted_targets_preserve_overall_column_conditioned_success_rate():
    outcomes = np.array(
        [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=float,
    )
    targets = _permuted_targets(outcomes, n_permutations=100, seed=7)
    assert targets.shape == (100, 4)
    assert np.allclose(targets.mean(axis=1), outcomes.mean())


def test_split_half_reliability_is_one_for_rows_stable_across_columns():
    row_labels = np.array([0, 1] * 8, dtype=float)
    outcomes = np.repeat(row_labels[:, None], 8, axis=1)
    reliability = _target_reliability(outcomes)
    assert reliability["n_splits"] == 35
    assert np.isclose(reliability["half_grid_correlation_median"], 1.0)
    assert np.isclose(reliability["spearman_brown_median"], 1.0)


def test_pre_registered_signal_detects_strong_independent_route_geometry():
    n_rows = 30
    labels = np.repeat([0.0, 1.0], n_rows // 2)
    outcomes = np.repeat(labels[:, None], 8, axis=1)
    feature_rounds = []
    for denoise in range(10):
        signal = labels[:, None] + np.arange(n_rows)[:, None] * 1e-5
        zeros = np.zeros((n_rows, 1), dtype=float)
        feature_rounds.append(
            {
                "combined": signal,
                "probability": signal,
                "entropy": zeros,
                "margin": zeros,
                "token_consensus": zeros,
                "churn": zeros,
                "displacement": zeros,
            }
        )
    rng = np.random.default_rng(3)
    result = _signal_analysis(
        feature_rounds,
        outcomes,
        rng.normal(size=(n_rows, 10, 24)),
        rng.normal(size=(n_rows, 10, 7)),
        n_permutations=1000,
        seed=9,
    )
    assert result["primary"]["correlation"] > 0.99
    assert result["primary"]["permutation_p_primary"] < 0.01
