from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import zarr

from behavior_forks_v2 import (
    CANDIDATE_SCHEMA,
    EVENT_LABELS,
    atomic_npz,
    candidate_dir,
    commit_artifact,
    sha256_file,
)
from run_behavior_micro_pilot import (
    DEFAULT_CONFIG,
    PilotConfigError,
    _candidate_prefix_audit,
    _candidate_store_audit,
    _validate_noop_audit,
    _write_frozen_json,
    preflight,
    validate_config,
)


def _config():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _plan(tmp_path, state_count=2):
    value = {
        "schema": "himoe.behavior_study.plan.v2",
        "states": [
            {"snapshot_index": index, "state_id": f"state-{index}"}
            for index in range(state_count)
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return value, path


def _candidate_arrays(row_start, run_id):
    candidates = 32
    horizon = 2
    trajectory_steps = horizon + 1
    return {
        "candidate_ids": np.arange(candidates, dtype=np.int32),
        "actions": np.zeros((candidates, horizon, 7), dtype=np.float32),
        "candidate_flow_noise": np.zeros((candidates, 10, 24), dtype=np.float32),
        "query_ids": np.arange(row_start, row_start + candidates, dtype=np.int64),
        "server_trace_rows": np.arange(
            row_start, row_start + candidates, dtype=np.int64
        ),
        "store_ids": np.full(candidates, run_id),
        "sim_states": np.zeros((candidates, trajectory_steps, 4), dtype=np.float64),
        "eef_positions": np.zeros(
            (candidates, trajectory_steps, 3), dtype=np.float64
        ),
        "eef_quaternions": np.zeros(
            (candidates, trajectory_steps, 4), dtype=np.float64
        ),
        "chunk_success": np.zeros((candidates, trajectory_steps), dtype=np.bool_),
        "contact_active": np.zeros(
            (candidates, trajectory_steps, 1), dtype=np.bool_
        ),
        "contact_pair_names": np.asarray(["eef|object"]),
        "event_flags": np.zeros(
            (candidates, trajectory_steps, len(EVENT_LABELS)), dtype=np.bool_
        ),
        "event_labels": np.asarray(EVENT_LABELS),
        "fidelity_rerun_sim_states": np.zeros(
            (trajectory_steps, 4), dtype=np.float64
        ),
        "fidelity_passed": np.asarray(True),
    }


def _commit_candidate(
    capture_root,
    plan_path,
    *,
    pool,
    state_id,
    row_start,
    run_id="run-a",
):
    target = candidate_dir(capture_root, pool, state_id)
    target.mkdir(parents=True)
    data = target / "candidate.npz"
    atomic_npz(data, _candidate_arrays(row_start, run_id))
    commit_artifact(
        target,
        CANDIDATE_SCHEMA,
        {
            "pool": pool,
            "state_id": state_id,
            "plan_sha256": sha256_file(plan_path),
            "route_store_id": run_id,
        },
        {"data": data},
    )


def _candidate_stores(root, *, rows, run_ids=None, durable_rows=None):
    names = ("routes.zarr", "hidden.zarr", "flow_trajectory.zarr")
    array_names = ("episode_id", "episode_id", "query_id")
    if run_ids is None:
        run_ids = ("run-a",) * 3
    if durable_rows is None:
        durable_rows = (rows,) * 3
    if isinstance(rows, int):
        rows = (rows,) * 3
    for name, array_name, count, run_id, durable in zip(
        names, array_names, rows, run_ids, durable_rows, strict=True
    ):
        group = zarr.create_group(str(root / name), overwrite=True)
        group.attrs.update(
            {"capture_run_id": run_id, "common_durable_rows": durable}
        )
        group.create_array(array_name, data=np.arange(count, dtype=np.int64))


def test_frozen_micro_config_and_sources_pass_preflight():
    result = preflight(DEFAULT_CONFIG, min_free_gib=0.0)
    assert result["ok"] is True
    assert result["confirmatory"] is False
    assert [row["task_id"] for row in result["task_sources"]] == [0, 1, 3]
    assert all(row["success"] >= 3 and row["failure"] >= 3 for row in result["task_sources"])


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("confirmatory",), True, "non-confirmatory"),
        (("capture", "candidate_rows_expected"), 576, "screen and formal"),
        (("capture", "continuation_rows_expected"), 1, "never be recorded"),
        (("global_topup_rule", "apply_to"), "selected-pairs", "selective"),
        (("gates", "effect_direction_is_gate"), True, "cannot gate"),
        (("restrictions", "evaluation_threshold_retuning"), True, "disabled"),
    ],
)
def test_config_rejects_design_drift(path, value, match):
    config = copy.deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(PilotConfigError, match=match):
        validate_config(config)


def test_config_rejects_candidate_seed_domain_collision():
    config = copy.deepcopy(_config())
    domains = config["seed_derivation"]["domains"]
    domains[-1] = domains[0]
    with pytest.raises(PilotConfigError, match="not unique"):
        validate_config(config)


def test_noop_audit_requires_bitwise_pass_and_frozen_identity(tmp_path):
    config = _config()
    path = tmp_path / "noop.json"
    value = {
        "schema": "himoe.candidate_capture.noop_audit.v1",
        "implementation_sha256": sha256_file(
            DEFAULT_CONFIG.parent / "audit_candidate_capture_noop.py"
        ),
        "passed": True,
        "gpu": "0",
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "libero_wrist_layout": config["checkpoint"]["libero_wrist_layout"],
        "checks": {
            "actions_bitwise_equal": True,
            "flow_trajectories_bitwise_equal": True,
            "full_route_and_hidden_shapes": True,
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    assert _validate_noop_audit(path, config, "0")["passed"] is True

    value["checks"]["actions_bitwise_equal"] = False
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PilotConfigError, match="did not pass"):
        _validate_noop_audit(path, config, "0")


def test_candidate_prefix_audit_accepts_contiguous_screen_formal_prefix(tmp_path):
    plan, plan_path = _plan(tmp_path)
    capture_root = tmp_path / "capture"
    _commit_candidate(
        capture_root,
        plan_path,
        pool="screen",
        state_id="state-0",
        row_start=0,
    )
    _commit_candidate(
        capture_root,
        plan_path,
        pool="screen",
        state_id="state-1",
        row_start=32,
    )
    _commit_candidate(
        capture_root,
        plan_path,
        pool="formal",
        state_id="state-0",
        row_start=64,
    )

    result = _candidate_prefix_audit(capture_root, plan, plan_path)

    assert result["confirmed_rows"] == 96
    assert result["artifacts"] == 3
    assert result["capture_run_id"] == "run-a"
    assert result["complete"] is False


def test_candidate_prefix_audit_rejects_gap_before_later_artifact(tmp_path):
    plan, plan_path = _plan(tmp_path)
    capture_root = tmp_path / "capture"
    _commit_candidate(
        capture_root,
        plan_path,
        pool="screen",
        state_id="state-0",
        row_start=0,
    )
    _commit_candidate(
        capture_root,
        plan_path,
        pool="formal",
        state_id="state-0",
        row_start=32,
    )

    with pytest.raises(PilotConfigError, match="gap"):
        _candidate_prefix_audit(capture_root, plan, plan_path)


def test_candidate_prefix_audit_rejects_row_misalignment(tmp_path):
    plan, plan_path = _plan(tmp_path, state_count=1)
    capture_root = tmp_path / "capture"
    _commit_candidate(
        capture_root,
        plan_path,
        pool="screen",
        state_id="state-0",
        row_start=1,
    )

    with pytest.raises(PilotConfigError, match="confirmed prefix"):
        _candidate_prefix_audit(capture_root, plan, plan_path)


def test_candidate_prefix_audit_rejects_capture_run_id_drift(tmp_path):
    plan, plan_path = _plan(tmp_path, state_count=1)
    capture_root = tmp_path / "capture"
    _commit_candidate(
        capture_root,
        plan_path,
        pool="screen",
        state_id="state-0",
        row_start=0,
        run_id="run-a",
    )
    _commit_candidate(
        capture_root,
        plan_path,
        pool="formal",
        state_id="state-0",
        row_start=32,
        run_id="run-b",
    )

    with pytest.raises(PilotConfigError, match="multiple capture_run_ids"):
        _candidate_prefix_audit(capture_root, plan, plan_path)


def test_candidate_store_audit_accepts_aligned_three_store_prefix(tmp_path):
    _candidate_stores(tmp_path, rows=64)

    result = _candidate_store_audit(tmp_path, expected_rows=64)

    assert result == {
        "present": True,
        "rows": 64,
        "capture_run_id": "run-a",
        "common_durable_rows": 64,
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_rows", "match"),
    [
        ({"rows": (64, 63, 64)}, None, "different row counts"),
        (
            {"rows": 64, "run_ids": ("run-a", "run-b", "run-a")},
            None,
            "capture_run_id",
        ),
        ({"rows": 64, "durable_rows": (64, 32, 64)}, None, "durable prefix"),
        ({"rows": 64}, 96, "expected 96"),
    ],
)
def test_candidate_store_audit_rejects_misalignment(
    tmp_path, kwargs, expected_rows, match
):
    _candidate_stores(tmp_path, **kwargs)

    with pytest.raises(PilotConfigError, match=match):
        _candidate_store_audit(tmp_path, expected_rows=expected_rows)


def test_write_frozen_json_allows_identical_resume_and_rejects_drift(tmp_path):
    path = tmp_path / "selection.json"
    frozen = {"schema": "selection.v1", "state_ids": ["a", "b"]}

    _write_frozen_json(path, frozen)
    first_bytes = path.read_bytes()
    _write_frozen_json(path, copy.deepcopy(frozen))

    assert path.read_bytes() == first_bytes
    with pytest.raises(PilotConfigError, match="would change on resume"):
        _write_frozen_json(path, {**frozen, "state_ids": ["a"]})
    assert json.loads(path.read_text(encoding="utf-8")) == frozen
