import numpy as np

from analyze_commitment_grid import Cell, _commitment_statistics, _integrity
from rollout_commitment_grid import _blocked_grid_plan


def _cell(row, col, outcome):
    first_noise = np.full((10, 24), row, dtype=np.float32)
    future_noise = np.full((10, 24), 100 + col, dtype=np.float32)
    return Cell(
        summary={
            "grid_row": row,
            "grid_col": col,
            "success": outcome,
            "initial_observation_sha256": "same-observation",
        },
        route0=np.full((8, 10, 11, 32), row / 100.0, dtype=np.float32),
        as_probs0=np.full((4, 3), row / 100.0, dtype=np.float32),
        ids0=np.full((8, 10, 11, 4), row, dtype=np.uint8),
        entropy0=np.zeros((8, 10, 11), dtype=np.float32),
        flow_noises=np.stack([first_noise, future_noise]),
        action0=np.full((10, 7), row / 10.0, dtype=np.float32),
    )


def test_blocked_plan_starts_with_a_complete_two_by_two_block():
    assert _blocked_grid_plan(4, 4)[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(set(_blocked_grid_plan(5, 3))) == 15


def test_integrity_recognizes_fixed_rows_and_common_future_columns():
    cells = [_cell(row, col, False) for row in range(2) for col in range(2)]
    result = _integrity(cells)
    assert result["valid"] is True
    assert result["same_row_max_abs_route0_difference"] == 0.0
    assert result["same_row_max_abs_as_route0_difference"] == 0.0
    assert result["same_row_max_abs_hb_route0_difference"] == 0.0
    assert result["same_column_future_noise_mismatches"] == 0


def test_mixed_row_is_directly_counted_as_non_commitment():
    outcomes = np.array([[0, 1], [1, 1]], dtype=float)
    result = _commitment_statistics(outcomes, n_perm=100, seed=7)
    assert result["mixed_rows"] == 1
    assert result["stable_rows"] == 1
    assert result["conditional_entropy_bits"] == 0.5


def test_strong_row_commitment_beats_column_preserving_null():
    outcomes = np.vstack([np.zeros((4, 8)), np.ones((4, 8))])
    result = _commitment_statistics(outcomes, n_perm=2000, seed=11)
    assert result["mixed_rows"] == 0
    assert result["conditional_entropy_bits"] == 0.0
    assert result["permutation"]["stable_rows_upper_p"] < 0.01
    assert result["permutation"]["conditional_entropy_lower_p"] < 0.01


def test_common_future_stream_effect_beats_row_preserving_null():
    outcomes = np.tile(np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float), (8, 1))
    result = _commitment_statistics(outcomes, n_perm=2000, seed=19)
    effect = result["future_stream_effect"]
    assert effect["column_rate_variance"] == 0.25
    assert effect["permutation"]["column_variance_upper_p"] < 0.01
