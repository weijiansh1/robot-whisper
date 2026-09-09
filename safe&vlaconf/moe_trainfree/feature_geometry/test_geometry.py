"""Check temporal and grouping contracts behind the geometry analysis."""

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("moe_feature_geometry", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dynamic_representation_does_not_use_future_queries():
    rng = np.random.default_rng(51)
    m = rng.uniform(.001, .1, (2, 30, 8)).astype(np.float32)
    a = rng.uniform(.001, .1, (2, 30)).astype(np.float32)
    p = rng.uniform(.01, .8, (2, 30)).astype(np.float32)
    full, _ = MODULE.dynamics(m, a, p, .5)
    assert np.isnan(full[:, :7]).all()
    for end in (8, 12, 17):
        prefix, _ = MODULE.dynamics(m[:, :end], a[:, :end], p[:, :end], .5)
        np.testing.assert_allclose(prefix, full[:, :end], equal_nan=True)


def test_neighbors_exclude_same_task_even_when_nearest():
    values = np.asarray([[0.], [.001], [5.], [6.], [12.], [13.]])
    tasks = np.asarray(["a", "a", "b", "b", "c", "c"])
    neighbors, allowed = MODULE.other_task_neighbors(values, tasks, 2)
    assert set(neighbors[0]) == {2, 3}
    assert np.all(tasks[neighbors] != tasks[:, None])
    assert not np.diag(allowed).any()
    assert np.all(allowed.sum(1) == 4)


def test_reference_scaling_ignores_nonreference_changes():
    rng = np.random.default_rng(12)
    values = rng.normal(size=(4, 10, 3)).astype(np.float32)
    _, first = MODULE.reference_scaling(values, np.asarray([0, 1]))
    values[2:] *= 100000
    _, second = MODULE.reference_scaling(values, np.asarray([0, 1]))
    assert first == second
