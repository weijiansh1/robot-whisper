from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from analyze_behavior_geometry import (
    _apply_physical_scales,
    _coarse_event_matrices,
    _confirmatory_checks,
    _fit_or_load_physical_scales,
    _formal_support_limitations,
    _fit_or_load_thresholds,
    _validate_formal_arrays,
    load_formal,
)

from behavior_geometry import (
    PairThresholds,
    action_distance_matrix,
    action_rms_distance_matrix,
    classify_pair,
    contact_event_distance_matrix,
    exact_event_equal_matrix,
    interval_is_equivalent,
    paired_binary_difference_interval,
    quaternion_distance_matrix,
    snapshot_bootstrap_mean,
)


def test_action_distance_uses_mean_stepwise_l2_and_gripper_weight() -> None:
    actions = np.zeros((2, 2, 7), dtype=np.float64)
    actions[1, 0, :2] = [6.0, 12.0]
    actions[1, 1, 6] = 4.0
    action_std = np.asarray([2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 2.0])

    weighted = action_distance_matrix(actions, action_std, gripper_weight=0.25)
    unweighted = action_distance_matrix(actions, action_std, gripper_weight=1.0)
    legacy_rms = action_rms_distance_matrix(actions, action_std)

    # Normalized per-step deltas are [3, 4, 0, ...] and [0, ..., 2].
    # The requested metric is therefore (5 + 0.25 * 2) / 2, not a flat RMS.
    assert weighted[0, 1] == pytest.approx(2.75)
    assert unweighted[0, 1] == pytest.approx(3.5)
    assert legacy_rms[0, 1] == pytest.approx(np.sqrt(29.0 / 14.0))
    assert weighted[0, 1] != pytest.approx(legacy_rms[0, 1])
    np.testing.assert_array_equal(weighted, weighted.T)
    np.testing.assert_array_equal(np.diag(weighted), np.zeros(2))


def test_quaternion_distance_treats_q_and_negative_q_as_same_orientation() -> None:
    q = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ],
        dtype=np.float64,
    )
    quaternions = np.stack([q, -q])

    distance = quaternion_distance_matrix(quaternions)

    np.testing.assert_allclose(distance, np.zeros((2, 2)), atol=1e-12, rtol=0.0)


def test_contact_jaccard_and_exact_event_tape_equality() -> None:
    contacts = np.asarray(
        [
            [[1, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [0, 0, 0]],
            [[1, 1, 0], [0, 0, 0]],
        ],
        dtype=np.bool_,
    )
    success = np.asarray(
        [
            [0, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.bool_,
    )

    distance = contact_event_distance_matrix(contacts)
    equal = exact_event_equal_matrix(contacts, success)

    assert distance[0, 1] == 0.0
    # First time step has Jaccard distance 1/2; both tapes are empty at step 2.
    assert distance[0, 2] == pytest.approx(0.25)
    assert equal[0, 1]
    assert not equal[0, 2]

    changed_success = success.copy()
    changed_success[1, 1] = False
    equal_with_changed_success = exact_event_equal_matrix(contacts, changed_success)
    assert distance[0, 1] == 0.0
    assert not equal_with_changed_success[0, 1]


def test_paired_binary_interval_needs_enough_agreement_to_claim_equivalence() -> None:
    small = paired_binary_difference_interval(
        np.zeros(8, dtype=np.bool_), np.zeros(8, dtype=np.bool_)
    )
    large = paired_binary_difference_interval(
        np.zeros(100, dtype=np.bool_), np.zeros(100, dtype=np.bool_)
    )

    assert small["estimate"] == 0.0
    assert small["discordant"] == 0
    assert not interval_is_equivalent(small, epsilon=0.1)

    assert large["estimate"] == 0.0
    assert large["discordant"] == 0
    assert interval_is_equivalent(large, epsilon=0.1)
    assert -0.1 <= large["lower"] <= large["upper"] <= 0.1

    just_short = paired_binary_difference_interval(
        np.zeros(41, dtype=np.bool_), np.zeros(41, dtype=np.bool_)
    )
    just_enough = paired_binary_difference_interval(
        np.zeros(42, dtype=np.bool_), np.zeros(42, dtype=np.bool_)
    )
    assert not interval_is_equivalent(just_short, epsilon=0.1)
    assert interval_is_equivalent(just_enough, epsilon=0.1)

    left = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.bool_)
    right = np.asarray([0, 1, 1, 0, 0, 0], dtype=np.bool_)
    forward = paired_binary_difference_interval(left, right)
    reverse = paired_binary_difference_interval(right, left)
    assert forward["estimate"] == pytest.approx(-reverse["estimate"])
    assert forward["lower"] == pytest.approx(-reverse["upper"])
    assert forward["upper"] == pytest.approx(-reverse["lower"])


def test_pair_threshold_regions_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="strictly below"):
        PairThresholds(1.0, 1.0, 1.0, 2.0)
    with pytest.raises(ValueError, match="strictly below"):
        PairThresholds(1.0, 2.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="q_difference"):
        PairThresholds(1.0, 2.0, 1.0, 2.0, q_equivalence=0.1, q_difference=0.05)


@pytest.mark.parametrize(
    ("action", "physics", "events_equal", "interval", "expected"),
    [
        (4.0, 0.5, True, {"lower": -0.05, "upper": 0.05}, "command_redundancy"),
        (4.0, 4.0, True, {"lower": -0.05, "upper": 0.05}, "control_equivalence"),
        (0.5, 4.0, True, {"lower": 0.2, "upper": 0.4}, "physics_amplification"),
        (0.5, 0.5, True, {"lower": 0.2, "upper": 0.4}, "critical_microdifference"),
    ],
)
def test_classify_pair_covers_all_four_behavior_quadrants(
    action: float,
    physics: float,
    events_equal: bool,
    interval: dict[str, float],
    expected: str,
) -> None:
    thresholds = PairThresholds(
        action_near=1.0,
        action_far=3.0,
        physics_near=1.0,
        physics_far=3.0,
        q_equivalence=0.1,
        q_difference=0.1,
    )

    result = classify_pair(action, physics, events_equal, interval, thresholds)

    quadrant_keys = (
        "command_redundancy",
        "control_equivalence",
        "physics_amplification",
        "critical_microdifference",
    )
    assert {name for name in quadrant_keys if result[name]} == {expected}
    assert result["sensitive_fork"] == (
        expected in {"physics_amplification", "critical_microdifference"}
    )


def test_snapshot_bootstrap_reduces_pairs_before_resampling_snapshots() -> None:
    one_pair_per_snapshot = {
        "snapshot-a": np.asarray([0.0]),
        "snapshot-b": np.asarray([10.0]),
    }
    duplicated_pairs_in_first_snapshot = {
        "snapshot-a": np.zeros(1_000),
        "snapshot-b": np.asarray([10.0]),
    }

    reference = snapshot_bootstrap_mean(
        one_pair_per_snapshot, draws=1_000, seed=17
    )
    duplicated = snapshot_bootstrap_mean(
        duplicated_pairs_in_first_snapshot, draws=1_000, seed=17
    )

    assert reference == duplicated
    assert reference["mean"] == pytest.approx(5.0)
    assert reference["snapshots"] == 2
    assert reference["available"]
    assert reference["mean"] != pytest.approx(10.0 / 1_001.0)

    one = snapshot_bootstrap_mean({"only": np.asarray([0.2, 0.4])}, draws=1_000)
    assert one["mean"] == pytest.approx(0.3)
    assert not one["available"]
    assert one["lower"] is None and one["upper"] is None


def test_semantic_event_equality_preserves_timing_and_finger_identity() -> None:
    active = np.zeros((2, 3, 2), dtype=np.bool_)
    active[0, 1, :] = True
    active[1, 2, :] = True
    sim = np.zeros((2, 3, 1 + 7 + 6), dtype=np.float64)
    layout = {
        "nq": 7,
        "nv": 6,
        "joints": [
            {
                "joint": "target_joint",
                "is_robot": False,
                "is_task_object": True,
                "joint_type": "free",
                "qpos_lo": 0,
                "qpos_hi": 7,
                "qvel_lo": 0,
                "qvel_hi": 6,
            }
        ],
        "geom_roles": [
            {
                "geom": "left_finger",
                "role": "robot",
                "subtype": "gripper",
                "finger_side": "left",
                "body": "left_finger",
                "is_task_object": False,
            },
            {
                "geom": "right_finger",
                "role": "robot",
                "subtype": "gripper",
                "finger_side": "right",
                "body": "right_finger",
                "is_task_object": False,
            },
            {
                "geom": "target",
                "role": "object",
                "subtype": "object",
                "finger_side": None,
                "body": "target",
                "object_body": "target",
                "is_task_object": True,
            },
        ],
    }
    arrays = {
        "contact_active": active,
        "contact_pair_names": np.asarray(
            ["left_finger <-> target", "right_finger <-> target"]
        ),
        "chunk_success": np.zeros((2, 3), dtype=np.bool_),
        "sim_states": sim,
    }

    distance, equal, labels, provenance = _coarse_event_matrices(arrays, layout)

    assert not equal[0, 1]
    assert distance[0, 1] > 0.0
    assert "bilateral_grasp" in labels
    assert provenance["kind"] == "semantic_event_tape_v1"
    assert provenance["finger_sides"] == ["left", "right"]


def test_confirmatory_gate_requires_disjoint_identified_calibration() -> None:
    pool = {
        "tag": "eval",
        "continuation_repeats": 42,
        "snapshot_state_sha256": "e" * 64,
        "event_provenance": {
            "kind": "semantic_event_tape_v1",
            "contact_role_mapping_fraction": 1.0,
            "task_object_geom_count": 1,
            "finger_sides": ["left", "right"],
        },
    }
    external = {
        "kind": "frozen_external",
        "source_provenance": {"calibration_snapshot_hashes": ["c" * 64]},
    }

    assert not _formal_support_limitations([pool], q_equivalence=0.1, confidence=0.95)
    checks = _confirmatory_checks([pool], True, external, external)
    assert checks["passed"]

    overlapping = {
        "kind": "frozen_external",
        "source_provenance": {"calibration_snapshot_hashes": ["e" * 64]},
    }
    assert not _confirmatory_checks([pool], True, overlapping, external)["passed"]

    too_few = {**pool, "continuation_repeats": 41}
    limitations = _formal_support_limitations(
        [too_few], q_equivalence=0.1, confidence=0.95
    )
    assert any("R=41" in item for item in limitations)


def test_physical_scales_allow_task_specific_component_presence() -> None:
    rows = [
        {
            "snapshot": "free-object",
            "snapshot_state_sha256": "a" * 64,
            "dX_eef_position_m": 0.01,
            "dX_object_translation_m": 0.02,
        },
        {
            "snapshot": "drawer",
            "snapshot_state_sha256": "b" * 64,
            "dX_eef_position_m": 0.03,
            "dX_object_joint_native": 0.04,
        },
    ]
    args = SimpleNamespace(physics_scales_in=None, calibration_tags=None)

    scales, _ = _fit_or_load_physical_scales(rows, args)
    _apply_physical_scales(rows, scales)

    assert set(scales) == {
        "eef_position_m",
        "object_translation_m",
        "object_joint_native",
    }
    assert all(np.isfinite(row["d_physics"]) for row in rows)


def test_single_pair_thresholds_are_widened_without_assigning_far_region() -> None:
    rows = [{"snapshot": "smoke", "d_action": 2.0, "d_physics": 3.0}]
    args = SimpleNamespace(
        thresholds_in=None,
        calibration_tags=None,
        near_quantile=0.2,
        far_quantile=0.8,
        q_equivalence=0.1,
        q_difference=0.1,
    )

    threshold, provenance = _fit_or_load_thresholds(rows, args)

    assert threshold.action_near < 2.0 < threshold.action_far
    assert threshold.physics_near < 3.0 < threshold.physics_far
    assert provenance["degenerate_quantiles_widened"] == ["action", "physics"]
    assert provenance["raw_quantile_thresholds"]["action_near"] == 2.0


def test_formal_capture_contract_loads_dense_physics_and_repeated_crn(tmp_path) -> None:
    k, h, repeats, nq, nv = 3, 2, 100, 2, 2
    actions = np.zeros((k, h, 7), dtype=np.float32)
    actions[1, :, 0] = 0.1
    actions[2, :, 0] = 0.3
    sim = np.zeros((k, h + 1, 1 + nq + nv), dtype=np.float64)
    sim[1, :, 1] = [0.0, 0.01, 0.02]
    sim[2, :, 1] = [0.0, 0.03, 0.06]
    sim[2, :, 2] = [0.0, 0.02, 0.04]
    eef = np.zeros((k, h + 1, 3), dtype=np.float64)
    eef[..., 2] = 1.0
    eef[:, :, 0] = sim[:, :, 1]
    quaternion = np.zeros((k, h + 1, 4), dtype=np.float64)
    quaternion[..., 3] = 1.0
    gripper = np.zeros((k, h + 1, 2), dtype=np.float64)
    contact = np.zeros((k, h + 1, 1), dtype=np.bool_)
    contact[2, 1:, 0] = True
    success = np.zeros((k, h + 1), dtype=np.bool_)
    continuation = np.zeros((k, repeats), dtype=np.bool_)
    continuation[2] = True
    final = np.repeat(sim[:, None, -1, :], repeats, axis=1)
    np.savez_compressed(
        tmp_path / "snapshot.npz",
        actions=actions,
        sim_states=sim,
        eef_positions=eef,
        eef_quaternions=quaternion,
        gripper_qpos=gripper,
        chunk_success=success,
        contact_active=contact,
        contact_pair_names=np.asarray(["finger|object"]),
        continuation_success=continuation,
        continuation_final_sim_states=final,
        continuation_action_steps=np.full((k, repeats), 20, dtype=np.int32),
        continuation_flow_noise=np.zeros((repeats, 2, 10, 24), dtype=np.float32),
        candidate_trace_rows=np.full(k, -1, dtype=np.int64),
        candidate_ids=np.arange(k, dtype=np.int64),
        execution_order=np.arange(k, dtype=np.int32),
        robot_qpos_indices=np.asarray([0], dtype=np.int64),
        object_qpos_indices=np.asarray([1], dtype=np.int64),
        nq=np.asarray(nq, dtype=np.int32),
        snapshot_full_state_sha256=np.asarray("a" * 64),
        fidelity_passed=np.asarray(True),
    )
    layout = {
        "nq": nq,
        "nv": nv,
        "joints": [
            {
                "joint": "robot0_joint1",
                "is_robot": True,
                "state_lo": 1,
                "state_hi": 2,
                "qpos_lo": 0,
                "qpos_hi": 1,
                "qvel_lo": 0,
                "qvel_hi": 1,
                "joint_type": {"name": "hinge", "raw": 3},
            },
            {
                "joint": "drawer",
                "is_robot": False,
                "state_lo": 2,
                "state_hi": 3,
                "qpos_lo": 1,
                "qpos_hi": 2,
                "qvel_lo": 1,
                "qvel_hi": 2,
                "joint_type": {"name": "slide", "raw": 2},
            },
        ],
    }
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    metadata_path = tmp_path / "server_metadata.json"
    metadata_path.write_text(
        json.dumps({"normalization_action_std": [1.0] * 7})
    )
    metadata_sha256 = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    npz_sha256 = hashlib.sha256((tmp_path / "snapshot.npz").read_bytes()).hexdigest()
    layout_sha256 = hashlib.sha256((tmp_path / "layout.json").read_bytes()).hexdigest()
    artifact = {
        "snapshot_index": 0,
        "episode": 0,
        "fork_step": 4,
        "task_id": 0,
        "task_name": "synthetic task",
        "status": "complete",
        "npz_file": "snapshot.npz",
        "npz_sha256": npz_sha256,
        "layout_file": "layout.json",
        "layout_sha256": layout_sha256,
    }
    queries = [
        {
            "query_id": index,
            "local_call_ordinal": index,
            "trace_row": -1,
            "role": "candidate",
            "artifact_snapshot_index": 0,
        }
        for index in range(k)
    ]
    records = {
        "schema": "himoe.behavior_forks.v1",
        "snapshots": [{**artifact, "query_ordinal_range": [0, k]}],
        "queries": queries,
    }
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records))
    records_sha256 = hashlib.sha256(records_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "himoe.behavior_forks.v1",
                "status": "completed",
                "complete": True,
                "server_metadata_file": "server_metadata.json",
                "server_metadata_sha256": metadata_sha256,
                "seed_scheme": "test-crn",
                "records_file": "records.json",
                "records_file_sha256": records_sha256,
                "query_count": k,
                "config": {
                    "capture_routes": False,
                    "query_id_base": 0,
                    "trace_row_offset": 0,
                },
                "planned_snapshots": [
                    {"snapshot_index": 0, "episode": 0, "fork_step": 4}
                ],
                "artifacts": [artifact],
            }
        )
    )

    provenance, pools, rows = load_formal(tmp_path, gripper_weight=0.25, confidence=0.95)
    args = SimpleNamespace(physics_scales_in=None, calibration_tags=None)
    scales, scale_provenance = _fit_or_load_physical_scales(rows, args)
    _apply_physical_scales(rows, scales)

    assert provenance["capture_schema"] == "himoe.behavior_forks.v1"
    assert pools[0]["continuation_repeats"] == repeats
    assert len(rows) == 3
    assert scale_provenance["kind"] == "posthoc_all_snapshots"
    assert all(np.isfinite(row["d_physics"]) for row in rows)
    separated = next(row for row in rows if row["candidate_i"] == 0 and row["candidate_j"] == 2)
    assert separated["d_event"] > 0.0
    assert separated["q_ci_lower"] > 0.1 or separated["q_ci_upper"] < -0.1

    with np.load(tmp_path / "snapshot.npz", allow_pickle=False) as source:
        invalid = {name: source[name] for name in source.files}
    invalid["continuation_success"] = invalid["continuation_success"].astype(np.int8)
    with pytest.raises(ValueError, match="boolean storage"):
        _validate_formal_arrays(invalid)

    (tmp_path / "layout.json").write_text(json.dumps(layout) + "\n")
    with pytest.raises(ValueError, match="layout checksum mismatch"):
        load_formal(tmp_path, gripper_weight=0.25, confidence=0.95)
