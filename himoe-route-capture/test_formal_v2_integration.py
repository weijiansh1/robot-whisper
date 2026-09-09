from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import zarr

from analyze_route_outcome_geometry import (
    FORMAL_SPEC_SCHEMA,
    _load_formal_spec,
)
from assemble_behavior_study import assemble
from behavior_forks_v2 import (
    CANDIDATE_SCHEMA,
    EVENT_LABELS,
    PLAN_SCHEMA,
    SHARD_REPEATS,
    SHARD_SCHEMA,
    atomic_json,
    atomic_npz,
    candidate_dir,
    candidate_query_id,
    commit_artifact,
    continuation_query_id,
    flow_noise,
    pool_domain,
    seed_words,
    sha256_file,
    shard_dir,
)


RUN_ID = "synthetic-formal-v2-run"
K = 32
H = 1
B = 1
SIM_STATE_WIDTH = 5
STORE_ROWS = 18 * K * 2


def _study_plan(path: Path) -> list[dict]:
    states = []
    selections = (
        "event_critical_success_early",
        "event_critical_success_late",
        "event_critical_failure_early",
        "event_critical_failure_late",
        "uniform_audit",
        "uniform_audit",
    )
    for snapshot_index, (task_id, offset) in enumerate(
        (task_id, offset) for task_id in (0, 1, 3) for offset in range(6)
    ):
        states.append(
            {
                "state_id": f"task{task_id:02d}-synthetic-{offset}",
                "snapshot_index": snapshot_index,
                "task_id": task_id,
                "episode": 100 + snapshot_index,
                "fork_step": 2,
                "selection": selections[offset],
                "source_success": offset in (0, 1, 4),
            }
        )
    atomic_json(path, {"schema": PLAN_SCHEMA, "states": states})
    return states


def _candidate_arrays(state: dict, pool: str, row_start: int) -> dict[str, np.ndarray]:
    task_id = int(state["task_id"])
    episode = int(state["episode"])
    fork_step = int(state["fork_step"])
    noise = np.stack(
        [
            flow_noise(
                seed_words(
                    pool_domain(pool, "candidate"),
                    task_id,
                    episode,
                    fork_step,
                    (candidate,),
                )
            )
            for candidate in range(K)
        ]
    )
    sim_states = np.zeros((K, H + 1, SIM_STATE_WIDTH), dtype=np.float32)
    eef_quaternions = np.zeros((K, H + 1, 4), dtype=np.float32)
    eef_quaternions[..., 3] = 1.0
    return {
        "candidate_ids": np.arange(K, dtype=np.int32),
        "actions": np.zeros((K, H, 7), dtype=np.float32),
        "candidate_flow_noise": noise,
        "query_ids": np.asarray(
            [
                candidate_query_id(int(state["snapshot_index"]), pool, candidate)
                for candidate in range(K)
            ],
            dtype=np.int32,
        ),
        "server_trace_rows": np.arange(row_start, row_start + K, dtype=np.int64),
        "store_ids": np.full(K, RUN_ID),
        "execution_order": np.arange(K, dtype=np.int32),
        "sim_states": sim_states,
        "eef_positions": np.zeros((K, H + 1, 3), dtype=np.float32),
        "eef_quaternions": eef_quaternions,
        "gripper_qpos": np.zeros((K, H + 1, 2), dtype=np.float32),
        "chunk_success": np.zeros((K, H + 1), dtype=np.bool_),
        "contact_active": np.zeros((K, H + 1, 1), dtype=np.bool_),
        "contact_pair_names": np.asarray(["robot <-> object"]),
        "event_flags": np.zeros((K, H + 1, len(EVENT_LABELS)), dtype=np.bool_),
        "event_labels": np.asarray(EVENT_LABELS),
        "robot_qpos_indices": np.asarray([0], dtype=np.int32),
        "object_qpos_indices": np.asarray([1], dtype=np.int32),
        "nq": np.asarray(2, dtype=np.int32),
        "fidelity_reference_candidate": np.asarray(0, dtype=np.int32),
        "fidelity_rerun_sim_states": sim_states[0].copy(),
        "fidelity_drift": np.asarray(0.0, dtype=np.float64),
        "fidelity_candidate_spread_mean": np.asarray(0.0, dtype=np.float64),
        "fidelity_candidate_spread_max": np.asarray(0.0, dtype=np.float64),
        "fidelity_threshold": np.asarray(1e-10, dtype=np.float64),
        "fidelity_passed": np.asarray(True, dtype=np.bool_),
    }


def _commit_candidates(
    capture_root: Path,
    plan_path: Path,
    states: list[dict],
    pool: str,
    row_offset: int,
    store_material: dict[str, np.ndarray],
) -> None:
    server_metadata = {
        "normalization_action_std": [1.0] * 7,
        "candidate_capture_run_id": RUN_ID,
    }
    for state in states:
        snapshot_index = int(state["snapshot_index"])
        row_start = row_offset + snapshot_index * K
        arrays = _candidate_arrays(state, pool, row_start)
        rows = np.asarray(arrays["server_trace_rows"], dtype=np.int64)
        store_material["query_id"][rows] = arrays["query_ids"]
        store_material["snapshot_index"][rows] = snapshot_index
        store_material["candidate_id"][rows] = arrays["candidate_ids"]
        for local, row in enumerate(rows):
            store_material["noise_digest"][row] = np.frombuffer(
                hashlib.sha256(
                    np.ascontiguousarray(arrays["candidate_flow_noise"][local]).tobytes()
                ).digest(),
                dtype=np.uint8,
            )
            store_material["action_digest"][row] = np.frombuffer(
                hashlib.sha256(
                    np.ascontiguousarray(arrays["actions"][local]).tobytes()
                ).digest(),
                dtype=np.uint8,
            )

        root = candidate_dir(capture_root, pool, str(state["state_id"]))
        root.mkdir(parents=True)
        atomic_npz(root / "candidate.npz", arrays)
        atomic_json(
            root / "layout.json",
            {"nq": 2, "nv": 2, "joints": [], "geom_roles": []},
        )
        atomic_json(root / "server_metadata.json", server_metadata)
        commit_artifact(
            root,
            CANDIDATE_SCHEMA,
            {
                "pool": pool,
                "state_id": state["state_id"],
                "snapshot_index": snapshot_index,
                "task_id": state["task_id"],
                "episode": state["episode"],
                "fork_step": state["fork_step"],
                "route_off_smoke": False,
                "route_store_id": RUN_ID,
                "server_durable_through_row": int(rows[-1]),
                "server_identity": {"checkpoint_sha256": "synthetic-checkpoint"},
                "plan_sha256": sha256_file(plan_path),
            },
            {
                "data": root / "candidate.npz",
                "layout": root / "layout.json",
                "server_metadata": root / "server_metadata.json",
            },
        )


def _commit_continuation_shards(
    capture_root: Path,
    states: list[dict],
    *,
    pool: str,
    cohort: str,
    repeats: int,
) -> None:
    seed_pool = "screen" if cohort == "screen" else "formal"
    seed_domain = pool_domain(seed_pool, "continuation")
    for state in states:
        candidate_root = candidate_dir(capture_root, pool, str(state["state_id"]))
        candidate_artifact = candidate_root / "artifact.json"
        for shard_index in range(repeats // SHARD_REPEATS):
            repeat_start = shard_index * SHARD_REPEATS
            noise = np.stack(
                [
                    [
                        flow_noise(
                            seed_words(
                                seed_domain,
                                int(state["task_id"]),
                                int(state["episode"]),
                                int(state["fork_step"]),
                                (repeat, future_step),
                            )
                        )
                        for future_step in range(B)
                    ]
                    for repeat in range(repeat_start, repeat_start + SHARD_REPEATS)
                ]
            )
            query_ids = np.empty((K, SHARD_REPEATS, B), dtype=np.int32)
            for candidate in range(K):
                for local_repeat in range(SHARD_REPEATS):
                    query_ids[candidate, local_repeat, 0] = continuation_query_id(
                        int(state["snapshot_index"]),
                        cohort,
                        candidate,
                        repeat_start + local_repeat,
                        0,
                        B,
                    )
            executed = np.ones(query_ids.shape, dtype=np.bool_)
            success = np.repeat(
                (np.arange(K) >= K // 2)[:, None], SHARD_REPEATS, axis=1
            )
            arrays = {
                "continuation_success": success,
                "continuation_final_sim_states": np.zeros(
                    (K, SHARD_REPEATS, SIM_STATE_WIDTH), dtype=np.float32
                ),
                "continuation_action_steps": np.ones(
                    (K, SHARD_REPEATS), dtype=np.int32
                ),
                "continuation_flow_noise": noise,
                "query_ids": query_ids,
                "server_trace_rows": np.full(query_ids.shape, -1, dtype=np.int64),
                "server_row_counts": np.full(query_ids.shape, STORE_ROWS, dtype=np.int64),
                "server_durable_through_rows": np.full(
                    query_ids.shape, STORE_ROWS - 1, dtype=np.int64
                ),
                "query_executed": executed,
                "continuation_actions": np.zeros(
                    (K, SHARD_REPEATS, B, H, 7), dtype=np.float32
                ),
                "continuation_action_executed": np.ones(
                    (K, SHARD_REPEATS, B, H), dtype=np.bool_
                ),
            }
            root = shard_dir(
                capture_root,
                pool,
                str(state["state_id"]),
                cohort,
                shard_index,
            )
            root.mkdir(parents=True)
            atomic_npz(root / "continuation.npz", arrays)
            commit_artifact(
                root,
                SHARD_SCHEMA,
                {
                    "pool": pool,
                    "label_cohort": cohort,
                    "seed_pool": seed_pool,
                    "state_id": state["state_id"],
                    "task_id": state["task_id"],
                    "episode": state["episode"],
                    "fork_step": state["fork_step"],
                    "shard_index": shard_index,
                    "repeat_start": repeat_start,
                    "repeat_stop": repeat_start + SHARD_REPEATS,
                    "route_off_smoke": False,
                    "capture_run_id": RUN_ID,
                    "candidate_artifact_sha256": sha256_file(candidate_artifact),
                },
                {"data": root / "continuation.npz"},
            )


def _shared_stores(root: Path, material: dict[str, np.ndarray]) -> dict[str, Path]:
    paths = {
        "routes": root / "routes.zarr",
        "hidden": root / "hidden.zarr",
        "flow_trajectory": root / "flow.zarr",
    }
    route = zarr.create_group(str(paths["routes"]), overwrite=True)
    hidden = zarr.create_group(str(paths["hidden"]), overwrite=True)
    flow = zarr.create_group(str(paths["flow_trajectory"]), overwrite=True)
    for group in (route, hidden, flow):
        group.attrs.update({"capture_run_id": RUN_ID, "durable_rows": STORE_ROWS})

    route.create_array(
        "hb_router_probs",
        shape=(STORE_ROWS, 8, 10, 11, 32),
        dtype=np.float16,
        chunks=(1, 8, 10, 11, 32),
        fill_value=np.float16(1.0 / 32.0),
    )
    route.create_array("episode_id", data=material["query_id"])
    route.create_array("control_step", data=np.zeros(STORE_ROWS, dtype=np.int32))
    hidden.create_array(
        "hb_hidden",
        shape=(STORE_ROWS, 8, 10, 11, 2),
        dtype=np.float16,
        chunks=(1, 8, 10, 11, 2),
        fill_value=np.float16(0.0),
    )
    hidden.create_array("episode_id", data=material["query_id"])
    hidden.create_array("control_step", data=np.zeros(STORE_ROWS, dtype=np.int32))
    flow.create_array(
        "x_traj",
        shape=(STORE_ROWS, 11, 10, 24),
        dtype=np.float32,
        chunks=(1, 11, 10, 24),
        fill_value=np.float32(0.0),
    )
    flow.create_array("capture_row", data=np.arange(STORE_ROWS, dtype=np.int64))
    flow.create_array("query_id", data=material["query_id"])
    flow.create_array("snapshot_index", data=material["snapshot_index"])
    flow.create_array("candidate_id", data=material["candidate_id"])
    flow.create_array("local_call_ordinal", data=np.zeros(STORE_ROWS, dtype=np.int32))
    flow.create_array(
        "observation_sha256",
        shape=(STORE_ROWS, 32),
        dtype=np.uint8,
        chunks=(32, 32),
        fill_value=np.uint8(0),
    )
    flow.create_array("flow_noise_sha256", data=material["noise_digest"])
    flow.create_array("actions_sha256", data=material["action_digest"])
    return paths


def test_formal_v2_assemblies_and_shared_stores_load_end_to_end(tmp_path: Path) -> None:
    plan_path = tmp_path / "study_plan.json"
    states = _study_plan(plan_path)
    capture_root = tmp_path / "capture"
    material = {
        "query_id": np.empty(STORE_ROWS, dtype=np.int64),
        "snapshot_index": np.empty(STORE_ROWS, dtype=np.int32),
        "candidate_id": np.empty(STORE_ROWS, dtype=np.int32),
        "noise_digest": np.empty((STORE_ROWS, 32), dtype=np.uint8),
        "action_digest": np.empty((STORE_ROWS, 32), dtype=np.uint8),
    }
    _commit_candidates(capture_root, plan_path, states, "screen", 0, material)
    _commit_candidates(capture_root, plan_path, states, "formal", 18 * K, material)
    _commit_continuation_shards(
        capture_root, states, pool="screen", cohort="screen", repeats=8
    )
    _commit_continuation_shards(
        capture_root, states, pool="formal", cohort="main", repeats=48
    )

    screen = tmp_path / "assembled-screen"
    main = tmp_path / "assembled-main"
    assert (
        assemble(
            Namespace(
                plan=plan_path,
                capture_root=capture_root,
                cohort="screen",
                target_r=8,
                state_ids_file=None,
                out=screen,
            )
        )
        == 0
    )
    assert (
        assemble(
            Namespace(
                plan=plan_path,
                capture_root=capture_root,
                cohort="main",
                target_r=48,
                state_ids_file=None,
                out=main,
            )
        )
        == 0
    )

    stores = _shared_stores(tmp_path, material)
    spec_path = tmp_path / "formal_spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema": FORMAL_SPEC_SCHEMA,
                "calibration_task_ids": [0],
                "evaluation_task_ids": [1, 3],
                "study_plan": str(plan_path),
                "stores": {name: str(path) for name, path in stores.items()},
                "bundles": [
                    {
                        "id": "all-tasks",
                        "screen_capture": str(screen),
                        "main_capture": str(main),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_formal_spec(spec_path, confidence=0.95)

    assert loaded["study_plan"]["schema"] == PLAN_SCHEMA
    assert len(loaded["queue_rows"]) == 18
    assert len(loaded["queues"]["screen"]["pools"]) == 18
    assert len(loaded["queues"]["main"]["pools"]) == 18
    assert loaded["queues"]["enriched"] == {"rows": [], "pools": []}
    assert len(loaded["queues"]["screen"]["rows"]) == 18 * (K * (K - 1) // 2)
    assert len(loaded["queues"]["main"]["rows"]) == 18 * (K * (K - 1) // 2)
    assert loaded["stores"] == {
        "routes": str(stores["routes"]),
        "hidden": str(stores["hidden"]),
        "flow_trajectory": str(stores["flow_trajectory"]),
        "rows": STORE_ROWS,
        "screen_selected_rows": 18 * K,
        "main_selected_rows": 18 * K,
        "union_exactly_covers_store": True,
        "flow_snapshot_index_alignment": True,
        "candidate_store_id_rows_validated": STORE_ROWS,
        "candidate_store_ids_match_capture_run_id": True,
        "three_store_identity_and_digest_alignment": True,
    }
    provenance = loaded["provenance"][0]
    assert provenance["screen_capture"]["candidate_artifacts_checksum_validated"] == 18
    assert provenance["screen_capture"]["continuation_shards_checksum_validated"] == 18
    assert provenance["main_capture"]["candidate_artifacts_checksum_validated"] == 18
    assert provenance["main_capture"]["continuation_shards_checksum_validated"] == 18 * 6
    assert provenance["screen_main_candidate_seeds_disjoint"] is True
    assert provenance["screen_main_restore_fidelity"]["passed"] is True
