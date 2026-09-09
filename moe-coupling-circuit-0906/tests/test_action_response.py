"""Tests for the pairing and summarising logic, with no policy and no GPU."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from measure_action_response import contrasts, per_row_readouts, summarise  # noqa: E402


def chunk(dx=0.0, dy=0.0, gripper=0.0):
    out = np.zeros((10, 7), np.float32)
    out[:, 0] = dx / 10
    out[:, 1] = dy / 10
    out[:, 6] = gripper
    return out


def _rows():
    """One pre unit with a shift, one post unit with an uncoupled condition."""
    return [
        {"unit_id": 0, "condition": "held", "anchor": "pre", "episode_index": 1,
         "query_index": 5, "target_xyz": np.array([0.3, 0.0, 0.9]),
         "eef_xyz": np.array([0.0, 0.0, 0.9])},
        {"unit_id": 0, "condition": "shift8", "anchor": "pre", "episode_index": 1,
         "query_index": 5, "target_xyz": np.array([0.3, 0.08, 0.9]),
         "eef_xyz": np.array([0.0, 0.0, 0.9])},
        {"unit_id": 1, "condition": "held", "anchor": "post", "episode_index": 1,
         "query_index": 7, "target_xyz": np.array([0.4, 0.0, 1.1]),
         "eef_xyz": np.array([0.4, 0.0, 1.1])},
        {"unit_id": 1, "condition": "uncoupled", "anchor": "post", "episode_index": 1,
         "query_index": 7, "target_xyz": np.array([0.3, 0.0, 0.9]),
         "eef_xyz": np.array([0.4, 0.0, 1.1])},
    ]


def test_contrasts_pair_every_condition_against_its_own_held_row():
    rows = _rows()
    actions = np.stack([chunk(dx=0.2), chunk(dx=0.2, dy=0.05), chunk(gripper=-1.0), chunk(gripper=-1.0)])
    records = contrasts(rows, actions, shift_direction={0: np.array([0.0, 1.0, 0.0])})
    assert [(r["unit_id"], r["condition"]) for r in records] == [(0, "shift8"), (1, "uncoupled")]
    shift = records[0]
    # The command gained a +y component, and the target moved in +y.
    assert shift["delta_B_dir"] != 0.0
    assert np.isclose(shift["B_track"], 1.0)


def test_contrasts_report_no_change_as_zero_delta_and_nan_tracking():
    rows = _rows()
    identical = chunk(dx=0.2)
    actions = np.stack([identical, identical.copy(), chunk(), chunk()])
    records = contrasts(rows, actions, shift_direction={0: np.array([0.0, 1.0, 0.0])})
    shift = records[0]
    assert np.isclose(shift["delta_B_mag"], 0.0)
    assert np.isclose(shift["chunk_distance"], 0.0)
    # No change in the command means the tracking cosine is undefined, not zero.
    assert np.isnan(shift["B_track"])


def test_contrasts_refuse_a_unit_without_a_held_row():
    rows = [row for row in _rows() if not (row["unit_id"] == 0 and row["condition"] == "held")]
    actions = np.stack([chunk()] * len(rows))
    with pytest.raises(RuntimeError, match="no held row"):
        contrasts(rows, actions, shift_direction={})


def test_readouts_use_the_conditions_own_target_position():
    """B_dir must be scored against where the object now is, not where it was."""
    action = chunk(dx=0.0, dy=0.2)
    aimed_at_moved = per_row_readouts(action, [0, 0, 0.9], [0.0, 0.3, 0.9])
    aimed_at_original = per_row_readouts(action, [0, 0, 0.9], [0.3, 0.0, 0.9])
    assert np.isclose(aimed_at_moved["B_dir"], 1.0)
    assert np.isclose(aimed_at_original["B_dir"], 0.0, atol=1e-12)


def test_summarise_groups_by_condition_and_clusters_by_episode():
    records = []
    for episode in range(6):
        records.append({
            "unit_id": episode, "condition": "shift8", "anchor": "pre",
            "episode_index": episode, "delta_B_dir": 0.4, "delta_B_grip": 0.0,
            "delta_B_mag": 0.01, "B_track": 0.9, "chunk_distance": 0.05,
        })
        records.append({
            "unit_id": 100 + episode, "condition": "light", "anchor": "pre",
            "episode_index": episode, "delta_B_dir": 0.0, "delta_B_grip": 0.0,
            "delta_B_mag": 0.0, "B_track": float("nan"), "chunk_distance": 0.0,
        })
    out = summarise(records, resamples=200, seed=1)
    assert set(out) == {"shift8", "light"}
    assert out["shift8"]["n_units"] == 6
    assert out["shift8"]["delta_B_dir"]["n_clusters"] == 6
    assert out["shift8"]["delta_B_dir"]["low"] > 0
    # The lighting control must not manufacture an effect, and its undefined
    # tracking cosine must be reported as dropped rather than counted as zero.
    assert out["light"]["delta_B_dir"]["point"] == 0.0
    assert out["light"]["B_track"]["dropped_non_finite"] == 6
    assert out["light"]["B_track"]["n"] == 0
