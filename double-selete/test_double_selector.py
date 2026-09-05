from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).resolve().parent / "run_double_selector.py"
SPEC = importlib.util.spec_from_file_location("double_selector", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
selector = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def test_support_metrics_known_pattern() -> None:
    # Every token selects experts 0..3: four experts have 100% token occupancy.
    ids = np.tile(np.arange(4), (1, 10, 1))
    concentration, occupancy, size = selector.support_metrics(ids)
    assert np.isclose(concentration[0], 1.0 - np.log(4) / np.log(32))
    assert occupancy[0] == 1.0
    assert size[0] == 4


def test_quota_topk_has_exact_budget_and_head_coverage() -> None:
    episode = np.arange(8)
    loop = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
    static = loop[::-1]
    chosen = selector.quota_topk(loop, static, episode, 4)
    assert len(chosen) == 4
    assert len(np.unique(chosen)) == 4
    assert {0, 1}.issubset(set(chosen))
    assert {6, 7}.issubset(set(chosen))


def test_lag_delta_is_causal_and_aligned() -> None:
    values = np.arange(12, dtype=float).reshape(2, 6)
    delta = selector.lag_delta(values, 2)
    assert np.isnan(delta[:, :2]).all()
    assert np.all(delta[:, 2:] == 2)


def test_trailing_mean_requires_complete_window() -> None:
    values = np.array([[1.0, 2.0, np.nan, 4.0, 5.0]])
    result = selector.trailing_mean(values, 2)
    assert np.isnan(result[0, 0])
    assert result[0, 1] == 1.5
    assert np.isnan(result[0, 2])
    assert np.isnan(result[0, 3])
    assert result[0, 4] == 4.5


def test_metrics_allow_a_resample_with_no_positive_events() -> None:
    rows = pd.DataFrame(
        {
            "selected_n": [4],
            "positive_trap": [0],
            "positive_loop": [0],
            "positive_static": [0],
            "selected_trap": [0],
            "selected_loop": [0],
            "selected_static": [0],
        }
    )
    metrics = selector.metrics_from_group_rows(rows)
    assert np.isnan(metrics["recall_trap"])
    assert np.isnan(metrics["macro_type_recall"])
