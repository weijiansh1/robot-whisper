#!/usr/bin/env python3
"""Validate completeness and protocol invariants of double-selector outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RANK_METHODS = {
    "single",
    "single_soft",
    "double_shared_max",
    "double_shared_noisy_or",
    "double_shared_quota",
    "double_typed_max",
    "double_typed_noisy_or",
    "double_typed_quota",
    "double_typed_soft_max",
    "double_typed_soft_noisy_or",
    "double_typed_soft_quota",
}
SEQUENTIAL_METHODS = {
    "single",
    "double_shared_max",
    "double_shared_noisy_or",
    "double_typed_max",
    "double_typed_noisy_or",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-bootstrap", type=int, default=20_000)
    return parser.parse_args()


def assert_finite(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame[column].dropna().to_numpy(float)
        assert np.isfinite(values).all(), f"non-finite values in {column}"


def main() -> None:
    args = parse_args()
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "himoe.double_selector.v1"
    assert manifest["n_trajectories"] == 512
    assert manifest["n_initial_states"] == 16
    assert manifest["n_loop"] == 57
    assert manifest["n_static"] == 146
    assert manifest["n_trap_union"] == 193
    assert manifest["bootstrap"] == args.expected_bootstrap

    scores = pd.read_csv(RESULTS / "fixed_time_oof_scores.csv")
    assert len(scores) == 2 * 512
    assert not scores.duplicated(["episode_id", "query"]).any()
    assert set(scores["query"]) == {30, 34}
    assert_finite(
        scores,
        [
            "single",
            "single_soft",
            "double_shared_max",
            "double_typed_noisy_or",
            "double_typed_soft_noisy_or",
        ],
    )

    detail = pd.read_csv(RESULTS / "selection_by_init_state.csv")
    assert set(detail.method) == RANK_METHODS
    assert set(detail.init_state_id) == set(scores.init_state_id)
    assert (detail.selected_n == detail.budget).all()
    sizes = detail.groupby(
        ["query", "budget", "method", "label_scope"]
    ).size()
    assert (sizes == 16).all()

    deltas = pd.read_csv(RESULTS / "selection_bootstrap_deltas.csv")
    assert (deltas.n_bootstrap == args.expected_bootstrap).all()
    assert (deltas.n_bootstrap_effective <= args.expected_bootstrap).all()

    independent = pd.read_csv(RESULTS / "independent_heads_by_init_state.csv")
    assert len(independent) == 2 * 3 * 3 * 16
    assert (independent.loop_head_selected_n == independent.per_head_budget).all()
    assert (independent.static_head_selected_n == independent.per_head_budget).all()
    assert (
        independent.union_selected_n == independent.matched_single_selected_n
    ).all()
    assert (independent.union_selected_n >= independent.per_head_budget).all()
    assert (independent.union_selected_n <= 2 * independent.per_head_budget).all()
    independent_deltas = pd.read_csv(
        RESULTS / "independent_heads_bootstrap_deltas.csv"
    )
    assert (independent_deltas.n_bootstrap == args.expected_bootstrap).all()

    temporal = pd.read_csv(RESULTS / "sequential_oof_scores.csv")
    assert set(temporal.method) == SEQUENTIAL_METHODS
    assert len(temporal) == 512 * 23 * len(SEQUENTIAL_METHODS)
    assert not temporal.duplicated(["episode_id", "query", "method"]).any()
    assert (temporal["query"].min(), temporal["query"].max()) == (12, 34)
    assert_finite(temporal, ["score", "threshold"])

    thresholds = pd.read_csv(RESULTS / "sequential_thresholds.csv")
    assert len(thresholds) == 16 * len(SEQUENTIAL_METHODS)
    assert thresholds.inner_control_fa.between(0.075, 0.085).all()

    sequential_deltas = pd.read_csv(RESULTS / "sequential_bootstrap_deltas.csv")
    assert (sequential_deltas.n_bootstrap == args.expected_bootstrap).all()
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
