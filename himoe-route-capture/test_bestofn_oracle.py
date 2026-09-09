import numpy as np

from analyze_bestofn_oracle import candidate_subsets, cross_fitted_snapshot


def test_candidate_subsets_are_unique_and_bounded():
    subsets = candidate_subsets(16, 8, maximum=64, rng=np.random.default_rng(3))
    assert len(subsets) == 64
    assert len(set(subsets)) == 64
    assert all(len(value) == 8 and tuple(sorted(value)) == value for value in subsets)


def test_cross_fitted_oracle_recovers_stable_good_candidate():
    outcomes = np.zeros((8, 8), dtype=np.bool_)
    outcomes[3] = True
    result = cross_fitted_snapshot(
        outcomes, [1, 2, 4, 8], maximum_subsets=256, rng=np.random.default_rng(4)
    )
    assert result[1]["cross_fitted_headroom"] == 0.0
    assert result[8]["cross_fitted_headroom"] == 0.875
    assert result[8]["selection_agreement"] == 1.0


def test_cross_fitting_does_not_turn_split_specific_noise_into_headroom():
    outcomes = np.zeros((8, 4), dtype=np.bool_)
    outcomes[0, :2] = True
    outcomes[1, 2:] = True
    result = cross_fitted_snapshot(
        outcomes, [8], maximum_subsets=256, rng=np.random.default_rng(5)
    )
    assert result[8]["cross_fitted_headroom"] < 0.0
    assert result[8]["naive_in_sample_headroom"] > 0.0


def test_cross_fitting_rejects_odd_or_too_small_repeat_panels():
    for repeats in (2, 5):
        try:
            cross_fitted_snapshot(
                np.zeros((8, repeats), dtype=np.bool_),
                [8],
                maximum_subsets=256,
                rng=np.random.default_rng(6),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid repeat panel was accepted")
