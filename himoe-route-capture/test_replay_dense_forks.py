from __future__ import annotations

import json

import numpy as np
import pytest

from replay_dense_forks import (
    _endpoint_errors,
    _legacy_records_for_snapshot,
    _parse_candidate_ids,
    _selected_candidate_ids,
)


def test_candidate_selection_preserves_explicit_axis_and_seeds_subsets() -> None:
    assert _parse_candidate_ids("7, 2,5") == [7, 2, 5]
    explicit = _selected_candidate_ids(8, [7, 2, 5], None, 11)
    np.testing.assert_array_equal(explicit, [7, 2, 5])

    first = _selected_candidate_ids(32, None, 5, 1701)
    repeated = _selected_candidate_ids(32, None, 5, 1701)
    another = _selected_candidate_ids(32, None, 5, 1702)
    np.testing.assert_array_equal(first, repeated)
    assert len(first) == 5 and np.all(first[:-1] < first[1:])
    assert not np.array_equal(first, another)


def test_candidate_selection_rejects_ambiguous_or_out_of_range_requests() -> None:
    with pytest.raises(ValueError, match="either"):
        _selected_candidate_ids(4, [0, 1], 2, 1)
    with pytest.raises(ValueError, match="outside"):
        _selected_candidate_ids(4, [4], None, 1)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_candidate_ids("1,1")


def test_legacy_record_loader_requires_exact_candidate_axis(tmp_path) -> None:
    path = tmp_path / "fork_records.json"
    records = []
    for candidate in range(2):
        records.append(
            {
                "episode": 3,
                "fork_step": 11,
                "candidate": candidate,
                "qpos_after_chunk": [candidate, 0.0],
                "qvel_after_chunk": [0.0],
                "eef_after_chunk": [0.0, 0.0, 0.0],
                "gripper_after_chunk": [0.1, -0.1],
            }
        )
    path.write_text(json.dumps(records), encoding="utf-8")

    loaded = _legacy_records_for_snapshot(path, 3, 11, 2)
    assert sorted(loaded) == [0, 1]

    path.write_text(json.dumps(records[:-1]), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"missing=\[1\]"):
        _legacy_records_for_snapshot(path, 3, 11, 2)


def test_endpoint_gate_uses_qpos_qvel_not_cached_observable_diagnostics() -> None:
    # [time] + two qpos + one qvel
    sim = np.zeros((1, 2, 4), dtype=np.float64)
    sim[0, -1] = [1.0, 0.2, -0.3, 0.4]
    eef = np.zeros((1, 2, 3), dtype=np.float64)
    gripper = np.zeros((1, 2, 2), dtype=np.float64)
    endpoints = {
        "legacy_qpos_after_chunk": np.asarray([[0.2, -0.3]]),
        "legacy_qvel_after_chunk": np.asarray([[0.4]]),
        # Historical values may differ because they were sampled observables.
        "legacy_eef_after_chunk": np.asarray([[9.0, 9.0, 9.0]]),
        "legacy_gripper_after_chunk": np.asarray([[8.0, -8.0]]),
    }

    errors = _endpoint_errors(sim, eef, gripper, endpoints, nq=2, nv=1, atol=1e-10)

    assert bool(errors["endpoint_passed"][0])
    assert errors["endpoint_eef_max_abs_error"][0] == 9.0
    assert errors["endpoint_gripper_max_abs_error"][0] == 8.0

    endpoints["legacy_qvel_after_chunk"] = np.asarray([[0.5]])
    errors = _endpoint_errors(sim, eef, gripper, endpoints, nq=2, nv=1, atol=1e-10)
    assert not bool(errors["endpoint_passed"][0])
