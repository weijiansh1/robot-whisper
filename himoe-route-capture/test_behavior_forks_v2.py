from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from behavior_forks_v2 import (
    EVENT_LABELS,
    SEED_DOMAINS,
    ArtifactError,
    CANDIDATE_SCHEMA,
    PLAN_SCHEMA,
    SHARD_SCHEMA,
    atomic_json,
    atomic_npz,
    candidate_query_id,
    candidate_dir,
    assert_no_seed_overlap,
    cohort_candidate_pool,
    cohort_seed_pool,
    commit_artifact,
    continuation_query_id,
    dense_event_tape,
    domain_low32,
    seed_words,
    shard_dir,
    sha256_file,
    verify_artifact,
)
from behavior_study_plan import _assign_critical, _audit_step
from branch_snapshot import SnapshotCodecError, decode_full_state, encode_full_state
from assemble_behavior_study import assemble


def test_safe_full_state_codec_roundtrip_and_rejects_object_values() -> None:
    snapshot = {
        "sim": np.arange(5, dtype=np.float64),
        "controllers": [{"goal": np.asarray([1.0, 2.0]), "step": np.int64(3)}],
        "env": {"done": False, "cur_time": 1.25},
        "mjdata": {"qacc_warmstart": np.zeros(3)},
    }
    metadata, arrays = encode_full_state(snapshot)
    restored = decode_full_state(json.loads(json.dumps(metadata)), arrays)
    np.testing.assert_array_equal(restored["sim"], snapshot["sim"])
    assert restored["controllers"][0]["step"] == 3
    with pytest.raises(SnapshotCodecError, match="unsupported"):
        encode_full_state({"bad": object()})
    with pytest.raises(SnapshotCodecError, match="object array"):
        encode_full_state({"bad": np.asarray([object()], dtype=object)})


def test_seed_contract_domains_are_exact_and_nonoverlapping() -> None:
    assert SEED_DOMAINS["screen_candidate"] == "screen/candidate"
    assert SEED_DOMAINS["screen_continuation"] == "screen/continuation"
    assert SEED_DOMAINS["formal_candidate"] == "formal/candidate"
    assert SEED_DOMAINS["formal_continuation"] == "formal/continuation"
    assert SEED_DOMAINS["execution"] == "execution"
    assert SEED_DOMAINS["audit_inclusion"] == "audit/inclusion"
    assert domain_low32("screen/candidate") == int.from_bytes(
        __import__("hashlib").sha256(b"screen/candidate").digest()[-4:], "big"
    )
    records = [
        seed_words(domain, 1, 2, 3, (4,))
        for domain in (
            "screen/candidate",
            "screen/continuation",
            "formal/candidate",
            "formal/continuation",
            "execution",
            "audit/inclusion",
        )
    ]
    assert_no_seed_overlap(records)
    with pytest.raises(ArtifactError, match="duplicate"):
        assert_no_seed_overlap([records[0], records[0]])


def test_dense_event_tape_preserves_time_axis() -> None:
    contact = np.zeros((2, 4, 1), dtype=bool)
    contact[0, 1:3, 0] = True
    gripper = np.zeros((2, 4, 2))
    gripper[0, 1:, :] = [[-1, -1], [-2, -2], [-1, -1]]
    sim = np.zeros((2, 4, 5))
    sim[0, 2, 2] = 0.01
    success = np.zeros((2, 4), dtype=bool)
    success[0, 3] = True
    events = dense_event_tape(contact, gripper, sim, success, 2, [1])
    assert events.shape == (2, 4, len(EVENT_LABELS))
    labels = {name: index for index, name in enumerate(EVENT_LABELS)}
    assert events[0, 1, labels["contact_onset"]]
    assert events[0, 3, labels["contact_release"]]
    assert events[0, 2, labels["object_motion"]]
    assert events[0, 3, labels["success_onset"]]


def _profile(scores: list[int]) -> dict:
    return {"score": np.asarray(scores), "valid_control_steps": np.arange(len(scores))}


def test_plan_critical_assignment_uses_distinct_episodes_and_window_maxima() -> None:
    rows = [
        {"episode_index": 8, "profile": _profile([0, 9, 0, 0, 0, 0, 0, 0, 0, 0])},
        {"episode_index": 3, "profile": _profile([0, 0, 8, 0, 0, 0, 7, 0, 0, 0])},
        {"episode_index": 5, "profile": _profile([0, 0, 0, 0, 0, 0, 10, 0, 0, 0])},
    ]
    selected = _assign_critical(rows)
    assert [(row["episode_index"], kind, step, score) for row, kind, step, score in selected] == [
        (3, "event_critical_early", 2, 8),
        (5, "event_critical_late", 6, 10),
    ]


def test_plan_zero_event_window_midpoint_and_central_uniform_audit() -> None:
    rows = [
        {"episode_index": episode, "profile": _profile([0] * 10)}
        for episode in (1, 2, 3)
    ]
    selected = _assign_critical(rows)
    assert selected[0][2:] == (3, 0)
    assert selected[1][2:] == (6, 0)
    audit = _audit_step(rows[2]["profile"], task=0, episode=3)
    assert 2 <= audit < 8


def test_query_id_ranges_are_deterministic_and_disjoint() -> None:
    screen = {candidate_query_id(state, "screen", candidate) for state in range(18) for candidate in range(32)}
    formal = {candidate_query_id(state, "formal", candidate) for state in range(18) for candidate in range(32)}
    continuation = {
        continuation_query_id(state, cohort, candidate, repeat, future, 5)
        for state in range(18)
        for cohort in ("screen", "main", "enriched")
        for candidate in range(32)
        for repeat in (0, 95)
        for future in range(5)
    }
    assert len(screen) == len(formal) == 576
    assert not screen & formal
    assert not (screen | formal) & continuation
    assert cohort_candidate_pool("screen") == "screen"
    assert cohort_candidate_pool("main") == "formal"
    assert cohort_candidate_pool("enriched") == "screen"
    assert cohort_seed_pool("screen") == "screen"
    assert cohort_seed_pool("main") == cohort_seed_pool("enriched") == "formal"


def test_artifact_descriptor_is_last_commit_and_checksums_payload(tmp_path) -> None:
    payload = tmp_path / "payload.npz"
    np.savez(payload, values=np.arange(3))
    descriptor = commit_artifact(
        tmp_path,
        "example.v2",
        {"pool": "formal"},
        {"data": payload},
    )
    assert verify_artifact(descriptor, "example.v2")["pool"] == "formal"
    payload.write_bytes(b"corrupt")
    with pytest.raises(ArtifactError, match="checksum"):
        verify_artifact(descriptor, "example.v2")


def test_assembler_builds_phase1_capture_from_one_selected_screen_state(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    states = [
        {
            "state_id": f"state-{index}",
            "snapshot_index": index,
            "task_id": (0, 1, 3)[index // 6],
            "episode": index,
            "fork_step": 2,
        }
        for index in range(18)
    ]
    atomic_json(plan_path, {"schema": PLAN_SCHEMA, "states": states})
    selected = tmp_path / "selected.json"
    atomic_json(selected, ["state-0"])
    capture_root = tmp_path / "capture"
    candidate_root = candidate_dir(capture_root, "screen", "state-0")
    candidate_root.mkdir(parents=True)
    k, h, d = 2, 1, 5
    candidate_words = [
        seed_words("screen/candidate", 0, 0, 2, (candidate,)) for candidate in range(k)
    ]
    candidate_arrays = {
        "candidate_ids": np.arange(k, dtype=np.int32),
        "actions": np.zeros((k, h, 7), dtype=np.float32),
        "candidate_flow_noise": np.stack(
            [__import__("behavior_forks_v2").flow_noise(words) for words in candidate_words]
        ),
        "query_ids": np.asarray([candidate_query_id(0, "screen", value) for value in range(k)]),
        "server_trace_rows": np.full(k, -1, dtype=np.int64),
        "store_ids": np.asarray(["", ""]),
        "execution_order": np.arange(k, dtype=np.int32),
        "sim_states": np.zeros((k, h + 1, d)),
        "eef_positions": np.zeros((k, h + 1, 3)),
        "eef_quaternions": np.tile([0.0, 0.0, 0.0, 1.0], (k, h + 1, 1)),
        "gripper_qpos": np.zeros((k, h + 1, 2)),
        "chunk_success": np.zeros((k, h + 1), dtype=bool),
        "contact_active": np.zeros((k, h + 1, 0), dtype=bool),
        "contact_pair_names": np.asarray([], dtype="<U1"),
        "event_flags": np.zeros((k, h + 1, len(EVENT_LABELS)), dtype=bool),
        "event_labels": np.asarray(EVENT_LABELS),
        "robot_qpos_indices": np.asarray([0], dtype=np.int32),
        "object_qpos_indices": np.asarray([1], dtype=np.int32),
        "nq": np.asarray(2, dtype=np.int32),
        "fidelity_reference_candidate": np.asarray(0),
        "fidelity_rerun_sim_states": np.zeros((h + 1, d)),
        "fidelity_drift": np.asarray(0.0),
        "fidelity_candidate_spread_mean": np.asarray(0.0),
        "fidelity_candidate_spread_max": np.asarray(0.0),
        "fidelity_threshold": np.asarray(1e-10),
        "fidelity_passed": np.asarray(True),
    }
    atomic_npz(candidate_root / "candidate.npz", candidate_arrays)
    atomic_json(candidate_root / "layout.json", {"nq": 2, "nv": 2, "joints": []})
    atomic_json(
        candidate_root / "server_metadata.json",
        {"normalization_action_std": [1.0] * 7},
    )
    candidate_meta = {
        "pool": "screen",
        "state_id": "state-0",
        "snapshot_index": 0,
        "task_id": 0,
        "episode": 0,
        "fork_step": 2,
        "route_off_smoke": True,
        "route_store_id": None,
        "server_durable_through_row": -1,
        "server_identity": {"checkpoint_sha256": "test"},
        "plan_sha256": sha256_file(plan_path),
    }
    candidate_artifact = commit_artifact(
        candidate_root,
        CANDIDATE_SCHEMA,
        candidate_meta,
        {
            "data": candidate_root / "candidate.npz",
            "layout": candidate_root / "layout.json",
            "server_metadata": candidate_root / "server_metadata.json",
        },
    )
    shard_root = shard_dir(capture_root, "screen", "state-0", "screen", 0)
    shard_root.mkdir(parents=True)
    shard_noise = np.stack(
        [
            [
                __import__("behavior_forks_v2").flow_noise(
                    seed_words("screen/continuation", 0, 0, 2, (repeat, 0))
                )
            ]
            for repeat in range(8)
        ]
    )
    query_ids = np.empty((k, 8, 1), dtype=np.int32)
    for candidate in range(k):
        for repeat in range(8):
            query_ids[candidate, repeat, 0] = continuation_query_id(
                0, "screen", candidate, repeat, 0, 1
            )
    shard_arrays = {
        "continuation_success": np.zeros((k, 8), dtype=bool),
        "continuation_final_sim_states": np.zeros((k, 8, d)),
        "continuation_action_steps": np.ones((k, 8), dtype=np.int32),
        "continuation_flow_noise": shard_noise,
        "query_ids": query_ids,
        "server_trace_rows": np.full((k, 8, 1), -1, dtype=np.int64),
        "server_row_counts": np.full((k, 8, 1), -1, dtype=np.int64),
        "server_durable_through_rows": np.full((k, 8, 1), -1, dtype=np.int64),
        "query_executed": np.ones((k, 8, 1), dtype=bool),
        "continuation_actions": np.zeros((k, 8, 1, h, 7), dtype=np.float32),
        "continuation_action_executed": np.ones((k, 8, 1, h), dtype=bool),
    }
    atomic_npz(shard_root / "continuation.npz", shard_arrays)
    commit_artifact(
        shard_root,
        SHARD_SCHEMA,
        {
            "pool": "screen",
            "label_cohort": "screen",
            "seed_pool": "screen",
            "state_id": "state-0",
            "task_id": 0,
            "episode": 0,
            "fork_step": 2,
            "shard_index": 0,
            "repeat_start": 0,
            "repeat_stop": 8,
            "route_off_smoke": True,
            "candidate_artifact_sha256": sha256_file(candidate_artifact),
        },
        {"data": shard_root / "continuation.npz"},
    )
    output = tmp_path / "assembled"
    assert assemble(
        Namespace(
            plan=plan_path,
            capture_root=capture_root,
            cohort="screen",
            target_r=8,
            state_ids_file=selected,
            out=output,
        )
    ) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == "himoe.behavior_forks.v1"
    assert manifest["complete"] is True
    assert manifest["config"]["v2_label_cohort"] == "screen"
    with np.load(output / manifest["artifacts"][0]["npz_file"], allow_pickle=False) as data:
        assert data["continuation_success"].shape == (k, 8)
