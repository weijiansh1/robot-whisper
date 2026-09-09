from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from analyze_route_outcome_geometry import (
    _attach_candidate_features,
    _horizon_censoring,
    _v2_pair_rows,
)
from behavior_micro_analysis import (
    FormalCalibration,
    fit_formal_calibration,
    formal_outcome_relation,
    matched_snapshot_effects,
    minimum_cost_action_matching,
    physical_presence_masks,
    q_label_coverage_and_topup,
    screening_metrics,
)


def _calibration(*, caliper: float = 1.0) -> FormalCalibration:
    return FormalCalibration(
        action_near_q20=1.0,
        action_far_q80=10.0,
        physics_near_q20=1.0,
        physics_far_q80=10.0,
        caliper_near=caliper,
        caliper_far=caliper,
        physical_schema=("eef_position_m",),
        physical_scales={"eef_position_m": 1.0},
        quadratic_coefficients={
            "near": {"d_route": [0.0, 0.0, 0.0]},
            "far": {"d_route": [0.0, 0.0, 0.0]},
        },
        physical_presence_masks={
            0: {"eef_position_m": True},
            1: {"eef_position_m": True},
            3: {"eef_position_m": True},
        },
    )


@pytest.mark.parametrize(
    ("events_equal", "lower", "upper", "expected"),
    [
        (True, -0.1, 0.1, "same"),
        (False, -0.05, 0.05, "different"),
        (True, 0.10001, 0.2, "different"),
        (True, -0.2, -0.10001, "different"),
        (True, -0.12, 0.08, "ambiguous"),
    ],
)
def test_formal_outcome_relation_uses_exact_event_and_paired_ci_rules(
    events_equal: bool, lower: float, upper: float, expected: str
) -> None:
    assert formal_outcome_relation(
        {
            "events_equal": events_equal,
            "q_ci_lower": lower,
            "q_ci_upper": upper,
        }
    ) == expected


def test_physical_schema_uses_task_presence_masks() -> None:
    rows = [
        {"task_id": 0, "dX_eef_position_m": 1.0, "dX_object_joint_native": 2.0},
        {"task_id": 0, "dX_eef_position_m": 2.0, "dX_object_joint_native": 3.0},
        {"task_id": 1, "dX_eef_position_m": 4.0},
        {"task_id": 1, "dX_eef_position_m": 5.0},
    ]
    assert physical_presence_masks(
        rows, ("eef_position_m", "object_joint_native")
    ) == {
        0: {"eef_position_m": True, "object_joint_native": True},
        1: {"eef_position_m": True, "object_joint_native": False},
    }


def test_physical_presence_must_be_consistent_within_task() -> None:
    with pytest.raises(ValueError, match="inconsistently stores"):
        physical_presence_masks(
            [
                {"task_id": 1, "dX_eef_position_m": 1.0},
                {"task_id": 1},
            ],
            ("eef_position_m",),
        )


def test_task0_calibration_is_snapshot_equal_weighted() -> None:
    rows = []
    for snapshot, actions in (
        ("dense", np.linspace(0.01, 1.0, 100)),
        ("sparse", np.arange(10.0, 20.0)),
    ):
        for index, action in enumerate(actions):
            rows.append(
                {
                    "task_id": 0,
                    "snapshot": snapshot,
                    "bundle_id": "task0",
                    "candidate_i": index,
                    "candidate_j": index + 1,
                    "d_action": float(action),
                    "dX_eef_position_m": float(action),
                    "d_route": float(1.0 + 2.0 * action + 0.5 * action * action),
                    "d_hidden": float(action),
                    "d_flow": float(action),
                }
            )
    pooled_q20 = float(np.quantile([row["d_action"] for row in rows], 0.2))
    calibration = fit_formal_calibration(rows)
    assert calibration.action_near_q20 > pooled_q20 + 0.1
    assert calibration.action_far_q80 > 10.0
    assert set(calibration.quadratic_coefficients["near"]) == {"d_route"}
    assert calibration.quadratic_coefficients["near"]["d_route"] == pytest.approx(
        [1.0, 2.0, 0.5], abs=1e-9
    )
    assert calibration.to_dict()["weighting"].startswith("snapshot-equal")


def test_action_matching_maximizes_cardinality_before_cost() -> None:
    matches = minimum_cost_action_matching(
        [0.0, 0.1], [0.09, 0.2], caliper=0.11
    )
    assert matches == pytest.approx([(0, 0, 0.09), (1, 1, 0.1)])


def test_matched_primary_effect_is_raw_route_and_residual_is_sensitivity() -> None:
    rows = []
    specifications = [
        ("same", 0, 1, 0.10, 0.10, 0.00),
        ("same", 2, 3, 0.20, 0.20, 0.00),
        ("different", 4, 5, 0.11, 0.40, 0.05),
        ("different", 6, 7, 0.19, 0.50, 0.05),
    ]
    for relation, left, right, action, route, residual in specifications:
        rows.append(
            {
                "task_id": 1,
                "snapshot": "snapshot-a",
                "bundle_id": "task1",
                "episode": 7,
                "candidate_i": left,
                "candidate_j": right,
                "action_stratum": "near",
                "outcome_relation": relation,
                "d_action": action,
                "d_route": route,
                "d_route_residual": residual,
                "d_hidden": route + 1.0,
                "d_flow": route + 2.0,
                "d_physics": route + 3.0,
            }
        )
    effects, ineligible = matched_snapshot_effects(
        rows, _calibration(caliper=0.2), min_matches=2
    )
    assert not ineligible
    assert len(effects) == 1
    assert effects[0]["route_effect"] == pytest.approx(0.3)
    assert effects[0]["route_residual_effect"] == pytest.approx(0.05)
    assert effects[0]["hidden_effect"] == pytest.approx(0.3)


def test_global_topup_uses_main_coverage_or_eligible_counts_only_at_r48() -> None:
    rows = []
    for task in (1, 3):
        for stratum in ("near", "far"):
            for relation in ("same", "different", "ambiguous"):
                rows.append(
                    {
                        "task_id": task,
                        "snapshot": f"task{task}-{stratum}",
                        "bundle_id": "formal",
                        "action_stratum": stratum,
                        "outcome_relation": relation,
                    }
                )
    at_r48 = q_label_coverage_and_topup(
        rows,
        [48],
        eligible_counts={"near": 4, "far": 4},
        evaluation_task_ids=(1, 3),
    )
    assert at_r48["global_topup_required"]
    assert at_r48["topup_trigger_any_cohort_coverage_below_0_70"]
    at_r96 = q_label_coverage_and_topup(
        rows,
        [96],
        eligible_counts={"near": 4, "far": 4},
        evaluation_task_ids=(1, 3),
    )
    assert not at_r96["global_topup_required"]


def test_global_topup_triggers_for_too_few_main_eligible_snapshots() -> None:
    rows = [
        {
            "task_id": task,
            "snapshot": f"task{task}-{stratum}",
            "bundle_id": "formal",
            "action_stratum": stratum,
            "outcome_relation": relation,
        }
        for task in (1, 3)
        for stratum in ("near", "far")
        for relation in ("same", "different")
    ]
    result = q_label_coverage_and_topup(
        rows,
        [48],
        eligible_counts={"near": 4, "far": 3},
        evaluation_task_ids=(1, 3),
    )
    assert result["global_topup_required"]
    assert result["topup_trigger_main_stratum_eligible_below_4"]


def test_screen_metrics_use_snapshot_inclusion_weights() -> None:
    snapshots = [
        {"snapshot_uid": "a", "inclusion_probability": 1.0},
        {"snapshot_uid": "b", "inclusion_probability": 0.5},
    ]
    flags = {
        uid: {"near": {"screen_positive": True}} for uid in ("a", "b")
    }
    result = screening_metrics(snapshots, {"near": {"a"}}, flags)["near"]
    assert result["recall"] == pytest.approx(1.0)
    assert result["ppv"] == pytest.approx(1.0 / 3.0)


def test_horizon_censoring_is_failure_after_all_50_actions(
    tmp_path: Path,
) -> None:
    np.savez(
        tmp_path / "snapshot.npz",
        continuation_success=np.asarray(
            [[False, False], [True, False]], dtype=np.bool_
        ),
        continuation_action_steps=np.asarray([[50, 40], [10, 50]], dtype=np.int32),
    )
    pools = [
        {
            "task_id": 1,
            "snapshot_state_sha256": "a" * 64,
            "capture_path": str(tmp_path),
            "data_file": "snapshot.npz",
        }
    ]
    result = _horizon_censoring(
        pools,
        {
            f"task1/{'a' * 64}": {
                "cohort": "event-critical",
                "inclusion_probability": 1.0,
            }
        },
        {1, 3},
        queue_name="main",
    )
    cell = result["by_evaluation_task_and_sampling_cohort"][
        "task1/event-critical"
    ]
    assert cell["horizon_censored"] == 2
    assert cell["continuations"] == 4
    assert cell["fraction"] == pytest.approx(0.5)
    assert not result["all_observed_cells_at_most_0_20"]


def test_v2_pair_adapter_uses_exact_primitive_event_tape(tmp_path: Path) -> None:
    k, h, repeats, state_dim = 2, 10, 48, 3
    sim = np.zeros((k, h + 1, state_dim), dtype=np.float64)
    sim[1, 1:, 1] = 0.2
    quaternions = np.zeros((k, h + 1, 4), dtype=np.float64)
    quaternions[..., 0] = 1.0
    event_flags = np.zeros((k, h + 1, 8), dtype=np.bool_)
    event_flags[1, 3, 4] = True
    arrays = {
        "actions": np.zeros((k, h, 7), dtype=np.float32),
        "sim_states": sim,
        "eef_positions": np.zeros((k, h + 1, 3), dtype=np.float64),
        "eef_quaternions": quaternions,
        "gripper_qpos": np.zeros((k, h + 1, 1), dtype=np.float64),
        "chunk_success": np.zeros((k, h + 1), dtype=np.bool_),
        "contact_active": np.zeros((k, h + 1, 1), dtype=np.bool_),
        "contact_pair_names": np.asarray(["robot <-> object"]),
        "event_flags": event_flags,
        "continuation_success": np.zeros((k, repeats), dtype=np.bool_),
        "continuation_final_sim_states": np.zeros(
            (k, repeats, state_dim), dtype=np.float64
        ),
        "continuation_action_steps": np.full((k, repeats), 50, dtype=np.int32),
        "candidate_trace_rows": np.arange(k, dtype=np.int64),
        "candidate_ids": np.arange(k, dtype=np.int32),
        "execution_order": np.arange(k, dtype=np.int32),
        "robot_qpos_indices": np.asarray([0], dtype=np.int32),
        "object_qpos_indices": np.asarray([], dtype=np.int32),
        "continuation_flow_noise": np.zeros(
            (repeats, 5, 10, 24), dtype=np.float32
        ),
        "fidelity_passed": np.asarray(True),
        "nq": np.asarray(1, dtype=np.int32),
    }
    np.savez(tmp_path / "snapshot.npz", **arrays)
    layout = {
        "nq": 1,
        "nv": 1,
        "joints": [
            {
                "joint": "robot_joint",
                "is_robot": True,
                "qpos_lo": 0,
                "qpos_hi": 1,
                "qvel_lo": 0,
                "qvel_hi": 1,
            }
        ],
    }
    (tmp_path / "snapshot.layout.json").write_text(json.dumps(layout))

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    entry = {
        "snapshot_index": 7,
        "snapshot_id": "t00-e000-s0010",
        "task_id": 0,
        "episode": 0,
        "fork_step": 10,
        "npz_file": "snapshot.npz",
        "npz_sha256": digest(tmp_path / "snapshot.npz"),
        "layout_file": "snapshot.layout.json",
        "layout_sha256": digest(tmp_path / "snapshot.layout.json"),
    }
    pool, rows = _v2_pair_rows(
        tmp_path, entry, np.ones(7), gripper_weight=0.25, confidence=0.95
    )
    assert pool["snapshot_index"] == 7
    assert len(rows) == 1
    assert not rows[0]["events_equal"]
    assert rows[0]["snapshot_identity_kind"] == "preregistered-v2-plan-state-id"


def _candidate_store_fixture(tmp_path: Path) -> tuple[dict, list, Path, Path, Path, Path]:
    candidates = 32
    run_id = "capture-run-test"
    capture = tmp_path / "capture"
    capture.mkdir()
    actions = np.zeros((candidates, 10, 7), dtype=np.float32)
    noise = np.zeros((candidates, 10, 24), dtype=np.float32)
    query_ids = np.arange(100, 100 + candidates, dtype=np.int32)
    trace_rows = np.arange(candidates, dtype=np.int64)
    np.savez(
        capture / "snapshot.npz",
        candidate_ids=np.arange(candidates, dtype=np.int32),
        candidate_trace_rows=trace_rows,
        candidate_query_ids=query_ids,
        candidate_flow_noise=noise,
        actions=actions,
        continuation_trace_rows=np.full((candidates, 1, 1), -1, dtype=np.int64),
        store_ids=np.full(candidates, run_id),
    )
    (capture / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "snapshot_index": 7,
                        "npz_file": "snapshot.npz",
                    }
                ]
            }
        )
    )
    routes_path = tmp_path / "routes.zarr"
    hidden_path = tmp_path / "hidden.zarr"
    flow_path = tmp_path / "flow.zarr"
    route = zarr.create_group(str(routes_path), overwrite=True)
    hidden = zarr.create_group(str(hidden_path), overwrite=True)
    flow = zarr.create_group(str(flow_path), overwrite=True)
    for group in (route, hidden, flow):
        group.attrs.update({"capture_run_id": run_id, "durable_rows": candidates})
    route.create_array(
        "hb_router_probs",
        data=np.full((candidates, 8, 10, 11, 32), 1.0 / 32.0, dtype=np.float16),
    )
    route.create_array("episode_id", data=query_ids)
    route.create_array("control_step", data=trace_rows)
    hidden.create_array(
        "hb_hidden", data=np.zeros((candidates, 8, 10, 11, 2), dtype=np.float16)
    )
    hidden.create_array("episode_id", data=query_ids)
    hidden.create_array("control_step", data=trace_rows)
    flow.create_array(
        "x_traj", data=np.zeros((candidates, 11, 10, 24), dtype=np.float32)
    )
    flow.create_array("capture_row", data=trace_rows)
    flow.create_array("query_id", data=query_ids.astype(np.int64))
    flow.create_array("snapshot_index", data=np.full(candidates, 7, dtype=np.int32))
    flow.create_array("candidate_id", data=np.arange(candidates, dtype=np.int32))
    flow.create_array("local_call_ordinal", data=trace_rows)
    flow.create_array(
        "observation_sha256", data=np.zeros((candidates, 32), dtype=np.uint8)
    )
    flow.create_array(
        "flow_noise_sha256",
        data=np.stack(
            [
                np.frombuffer(hashlib.sha256(row.tobytes()).digest(), dtype=np.uint8)
                for row in noise
            ]
        ),
    )
    flow.create_array(
        "actions_sha256",
        data=np.stack(
            [
                np.frombuffer(hashlib.sha256(row.tobytes()).digest(), dtype=np.uint8)
                for row in actions
            ]
        ),
    )
    pool = {
        "task_id": 0,
        "snapshot_state_sha256": "a" * 64,
        "data_file": "snapshot.npz",
        "candidates": candidates,
    }
    row = {
        "task_id": 0,
        "snapshot": "snapshot",
        "snapshot_state_sha256": "a" * 64,
        "candidate_i": 0,
        "candidate_j": 1,
    }
    return row, [pool], capture, routes_path, hidden_path, flow_path


def test_candidate_store_alignment_checks_snapshot_and_capture_run_id(
    tmp_path: Path,
) -> None:
    row, pools, capture, routes, hidden, flow_path = _candidate_store_fixture(tmp_path)
    diagnostics = _attach_candidate_features(
        [row],
        pools,
        capture=capture,
        routes_path=routes,
        hidden_path=hidden,
        flow_path=flow_path,
        projection_seed=3,
    )
    assert diagnostics["three_store_identity_and_digest_alignment"]
    assert row["d_flow"] == pytest.approx(0.0)

    flow = zarr.open_group(str(flow_path), mode="a")
    flow["snapshot_index"][0] = 8
    with pytest.raises(ValueError, match="snapshot boundary"):
        _attach_candidate_features(
            [dict(row)],
            pools,
            capture=capture,
            routes_path=routes,
            hidden_path=hidden,
            flow_path=flow_path,
            projection_seed=3,
        )
    flow["snapshot_index"][0] = 7
    with np.load(capture / "snapshot.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["store_ids"] = np.full(32, "wrong-run")
    np.savez(capture / "snapshot.npz", **payload)
    with pytest.raises(ValueError, match="capture_run_id"):
        _attach_candidate_features(
            [dict(row)],
            pools,
            capture=capture,
            routes_path=routes,
            hidden_path=hidden,
            flow_path=flow_path,
            projection_seed=3,
        )
