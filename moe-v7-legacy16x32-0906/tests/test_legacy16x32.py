#!/usr/bin/env python3
"""Guard rails for the legacy right-16x32 measurement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE / "experiments"))

from frozen_profile import load_frozen_point  # noqa: E402
from raw_route_features import COHORTS, episode_features, load_npz  # noqa: E402

RESULTS = BUNDLE / "results"
LEGACY = RESULTS / "legacy16x32"


def test_frozen_point_matches_published_profile() -> None:
    point = load_frozen_point()
    assert point.freeze_threshold == point.stored.freeze_threshold
    assert point.acceleration_threshold == point.stored.acceleration_threshold
    assert point.periodicity_threshold == point.stored.periodicity_threshold
    assert point.periodicity_scale == point.stored.periodicity_scale


def test_anchor_is_green() -> None:
    anchor = json.loads((RESULTS / "anchor/anchor_verification.json").read_text())
    assert anchor["anchor_green"] is True
    external = anchor["cohorts"]["external_8b"]["variants"][
        "published_mobility_raw_route_features"
    ]
    assert external["counts"]["tp"] == 439
    assert external["counts"]["fp"] == 80
    assert all(external["sealed_first_alarm_elementwise_exact"].values())
    development = anchor["cohorts"]["development_main"]["variants"][
        "published_mobility_raw_route_features"
    ]
    assert development["counts"]["tp"] == 382
    assert development["counts"]["fp"] == 67


def test_raw_extraction_reproduces_published_external_route_features() -> None:
    audit = json.loads(
        (RESULTS / "raw_features/external_8b_extraction_audit.json").read_text()
    )
    assert audit["bit_exact_vs_published_route_acceleration"] is True
    assert audit["bit_exact_vs_published_lag_periodicity"] is True


def test_guard_is_freeze_or_turbulence() -> None:
    alarms = load_npz(LEGACY / "legacy16x32_first_alarms.npz")
    freeze = alarms["legacy_freeze"].astype(int)
    turbulence = alarms["legacy_turbulence"].astype(int)
    expected = np.where(
        freeze < 0, turbulence, np.where(turbulence < 0, freeze, np.minimum(freeze, turbulence))
    )
    assert np.array_equal(alarms["legacy_guard"].astype(int), expected)
    acceleration = alarms["legacy_acceleration"].astype(int)
    periodicity = alarms["legacy_periodicity"].astype(int)
    assert np.array_equal(
        turbulence,
        np.where(
            (acceleration >= 0) & (periodicity >= 0),
            np.maximum(acceleration, periodicity),
            -1,
        ),
    )


def test_task_matched_survival_control_has_unit_lift() -> None:
    table = pd.read_csv(LEGACY / "survival_matched.csv")
    controls = table[
        table["detector"].str.startswith("control_")
        & (table["prior_basis"] == "task_matched")
        & (table["group"] == "all")
    ]
    assert len(controls) >= 5
    assert np.allclose(controls["lift"].to_numpy(), 1.0, atol=1e-12)


def test_every_extracted_feature_varies_within_episodes() -> None:
    table = pd.read_csv(LEGACY / "within_episode_information.csv")
    moving = table[~table["feature"].str.startswith("control:")]
    raw_channels = moving[
        moving["feature"].isin(["route_acceleration", "lag_periodicity"])
        | moving["feature"].str.startswith("layer_mobility")
    ]
    assert (raw_channels["fraction_constant_within"] == 0.0).all()
    assert (raw_channels["min_distinct_values_per_episode"] > 1).all()
    control = table[table["feature"].str.startswith("control:")]
    assert (control["fraction_constant_within"] == 1.0).all()


def test_length_detector_is_not_ranked_as_a_baseline() -> None:
    metrics = pd.read_csv(LEGACY / "outcome_metrics.csv")
    row = metrics[
        (metrics["detector"] == "horizon_cap_length_restatement") & (metrics["group"] == "all")
    ].iloc[0]
    assert row["risk_recall"] == 1.0
    assert bool(row["is_baseline"]) is False


@pytest.mark.parametrize("row", [0, 977, 2559])
def test_batch_features_match_a_query_by_query_monitor(row: int) -> None:
    import zarr

    from intrinsic_guard_monitor import GlobalIntrinsicProfile, IntrinsicGuardMonitor

    spec = COHORTS["legacy_16x32"]
    layer = load_npz(spec.layer_cache)
    task = str(layer["task_names"].astype(str)[int(layer["task_index"][row])])
    episode = int(layer["episode"][row])
    group = zarr.open_group(
        str(spec.cache_root / task / spec.run_id / "server/routes.zarr"), mode="r"
    )
    index = np.flatnonzero(np.asarray(group["episode_id"][:], dtype=int) == episode)
    raw = np.asarray(group["hb_router_probs"][index[0] : index[-1] + 1], dtype=np.float16)

    monitor = IntrinsicGuardMonitor(
        GlobalIntrinsicProfile.load(
            BUNDLE.parent / "moe-v7-0905/results/intrinsic_guard_v7/global_profile.npz"
        )
    )
    records = [monitor.update(query) for query in raw]
    mobility, acceleration, periodicity = episode_features(raw)
    streamed = np.asarray([record["route_acceleration"] for record in records], np.float32)
    assert np.array_equal(streamed, acceleration)
    streamed = np.asarray([record["lag_periodicity"] for record in records], np.float32)
    assert np.array_equal(streamed[2:], periodicity[2:])
    streamed = np.stack([record["layer_mobility"] for record in records])
    assert np.array_equal(streamed[1:], mobility[1:])
