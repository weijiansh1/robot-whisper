import numpy as np

from visualize_k32_route_trajectories import (
    k_oracle_summary,
    normalize_probabilities,
    trajectory_metrics,
)


def test_probability_normalization_repairs_float_mass_error():
    raw = np.full((2, 32), 1 / 32, dtype=np.float16)
    normalized, _ = normalize_probabilities(raw)
    assert np.allclose(normalized.sum(axis=-1), 1.0)


def test_constant_route_has_zero_change():
    raw = np.full((2, 8, 10, 10, 32), 1 / 32, dtype=np.float64)
    result = trajectory_metrics(raw)
    assert result["dominant_expert"].shape == (8, 20)
    assert np.allclose(result["route_change"][1:], 0.0)
    assert np.allclose(result["top4_mass"], 4 / 32)
    assert np.allclose(result["normalized_entropy"], 1.0)


def test_k32_oracle_counts_all_failure_states():
    rows = []
    for state in range(2):
        for seed in range(32):
            rows.append({
                "init_state_id": state,
                "flow_noise_seed": 1000 + seed,
                "success": state == 0 and seed == 7,
            })
    result = k_oracle_summary(rows)
    assert result["random_expected_success"] == 1 / 64
    assert result["k8_outcome_oracle_success"] == 1 / 8
    assert result["k32_outcome_oracle_success"] == 1 / 2
    assert result["all_failure_k32_states"] == 1
