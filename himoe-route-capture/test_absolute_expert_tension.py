import numpy as np
import pytest

import analyze_absolute_expert_tension as absolute


def test_absolute_metrics_keep_gate_and_remove_only_scale_normalization():
    weights = np.asarray([[[0.25, 0.75]], [[0.25, 0.75]]])
    raw_rms = np.asarray([[[3.0, 1.0]], [[3.0, 1.0]]])
    # First pair is aligned: r=1.5.  Second pair is opposed: r=0.
    routed_rms = np.asarray([[1.5], [0.0]])
    result = absolute.compute_token_metrics(weights, raw_rms, routed_rms)

    np.testing.assert_allclose(result["s1"], [[1.5], [1.5]])
    np.testing.assert_allclose(result["s2"], [[3.0], [3.0]])
    np.testing.assert_allclose(result["d_abs"], [[np.sqrt(0.75)], [np.sqrt(3.0)]])
    np.testing.assert_allclose(result["c_abs"], [[0.0], [1.5]])
    np.testing.assert_allclose(
        np.square(result["d_abs"]), result["s2"] * result["d_normalized"]
    )
    np.testing.assert_allclose(
        result["c_abs"], result["s1"] * result["c_normalized"]
    )


def test_absolute_metrics_reject_non_normalized_topk_weights():
    with pytest.raises(ValueError, match="top-k normalised"):
        absolute.compute_token_metrics(
            np.asarray([[0.2, 0.2]]),
            np.asarray([[1.0, 1.0]]),
            np.asarray([0.4]),
        )


def test_candidate_dispersion_is_within_fixed_state_pool():
    # [task, pool, candidate, denoise]
    value = np.asarray(
        [
            [
                [[0.0, 0.0], [2.0, 4.0]],
                [[1.0, 2.0], [3.0, 6.0]],
            ]
        ]
    )
    result = absolute._candidate_dispersion(value, ["task"])["task"]
    assert result["d0_median_pool_candidate_std"] == 1.0
    assert result["d9_median_pool_candidate_std"] == 2.0
    assert result["d9_over_d0"] == 2.0
