import numpy as np

from analyze_moe_success_failure_curves import scene_group_values


def test_scene_group_values_ranks_only_inside_mixed_strata():
    labels = np.array([False, True, False, True, False, False])
    scores = np.array([0.1, 0.9, 0.8, 0.2, -100.0, 100.0])
    scenes = np.array([0, 0, 0, 0, 1, 1])
    strata = np.array([0, 0, 1, 1, 2, 2])
    result = scene_group_values(labels, scores, scenes, strata, percentile=True)
    assert np.allclose(result[0], [0.5, 0.5])
    assert np.all(np.isnan(result[1]))


def test_scene_group_values_preserves_fixed_axis_amplitude():
    labels = np.array([False, True, False, True])
    scores = np.array([-2.0, 1.0, -1.0, 3.0])
    scenes = np.zeros(4, dtype=int)
    strata = np.array([0, 0, 1, 1])
    result = scene_group_values(labels, scores, scenes, strata, percentile=False)
    assert np.allclose(result[0], [-1.5, 2.0])
