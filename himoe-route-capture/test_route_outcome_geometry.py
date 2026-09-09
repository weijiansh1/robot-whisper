from __future__ import annotations

import numpy as np
import pytest

from analyze_route_outcome_geometry import (
    bootstrap_snapshot_mean,
    conditional_route_contrast,
    label_pair,
    one_sided_sign_flip,
)


def test_pair_labels_use_frozen_action_and_outcome_boundaries() -> None:
    common = {
        "action_near": 0.2,
        "action_far": 0.8,
        "outcome_same_max": 0.1,
        "outcome_different_min": 0.7,
    }
    assert label_pair(
        {"d_action": 0.2, "proxy_outcome_gap": 0.1}, **common
    ) == ("near", "same")
    assert label_pair(
        {"d_action": 0.8, "proxy_outcome_gap": 0.7}, **common
    ) == ("far", "different")
    assert label_pair(
        {"d_action": 0.5, "proxy_outcome_gap": 0.4}, **common
    ) == ("middle", "ambiguous")


def test_contrast_reduces_pairs_before_weighting_snapshots() -> None:
    rows = [
        {
            "snapshot": "a",
            "action_stratum": "far",
            "outcome_relation": "same",
            "d_route": 0.1,
        },
        {
            "snapshot": "a",
            "action_stratum": "far",
            "outcome_relation": "different",
            "d_route": 0.3,
        },
        {
            "snapshot": "b",
            "action_stratum": "far",
            "outcome_relation": "same",
            "d_route": 0.4,
        },
        {
            "snapshot": "b",
            "action_stratum": "far",
            "outcome_relation": "different",
            "d_route": 0.5,
        },
    ]
    duplicated = rows + [dict(rows[0]) for _ in range(100)]
    result = conditional_route_contrast(
        duplicated,
        action_stratum="far",
        draws=1_000,
        confidence=0.95,
        seed=7,
    )
    assert result["eligible_snapshot_count"] == 2
    assert result["snapshot_equal_bootstrap"]["mean"] == pytest.approx(0.15)


def test_snapshot_bootstrap_refuses_pseudo_ci_for_one_snapshot() -> None:
    result = bootstrap_snapshot_mean([0.2], draws=1_000, confidence=0.95, seed=3)
    assert result["mean"] == pytest.approx(0.2)
    assert not result["available"]
    assert result["lower"] is None and result["upper"] is None


def test_exact_sign_flip_reports_expected_positive_effect() -> None:
    result = one_sided_sign_flip([0.1, 0.2, 0.3], seed=1)
    assert result["exact"]
    assert result["draws"] == 8
    assert result["observed"] == pytest.approx(0.2)
    assert result["p"] == pytest.approx(0.125)


def test_contrast_requires_both_groups_in_same_snapshot() -> None:
    rows = [
        {
            "snapshot": "a",
            "action_stratum": "near",
            "outcome_relation": "same",
            "d_route": 0.1,
        },
        {
            "snapshot": "b",
            "action_stratum": "near",
            "outcome_relation": "different",
            "d_route": 0.3,
        },
    ]
    result = conditional_route_contrast(
        rows,
        action_stratum="near",
        draws=1_000,
        confidence=0.95,
        seed=7,
    )
    assert not result["available"]


def test_bootstrap_requires_finite_effects() -> None:
    with pytest.raises(ValueError, match="finite vector"):
        bootstrap_snapshot_mean(
            np.asarray([0.1, np.nan]), draws=1_000, confidence=0.95, seed=1
        )
