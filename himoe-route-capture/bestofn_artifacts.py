"""Pure artifact validators shared by capture, orchestration, and analysis."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from behavior_forks_v2 import ArtifactError
from bestofn_protocol import FLOW_NOISE_SHAPE


def validate_candidate(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    actions = np.asarray(arrays["actions"])
    k, h = int(metadata["candidate_count"]), int(metadata["action_chunk_horizon"])
    expected = {
        "candidate_ids": (k,),
        "actions": (k, h, 7),
        "candidate_flow_noise": (k, *FLOW_NOISE_SHAPE),
        "query_ids": (k,),
        "server_trace_rows": (k,),
        "store_ids": (k,),
        "execution_order": (k,),
        "candidate_action_executed": (k, h),
        "sim_states": (k, h + 1, np.asarray(arrays["sim_states"]).shape[-1]),
        "eef_positions": (k, h + 1, 3),
        "eef_quaternions": (k, h + 1, 4),
        "chunk_success": (k, h + 1),
        "chunk_terminal_success": (k,),
        "observation_image": (224, 224, 3),
        "observation_wrist_image": (224, 224, 3),
        "observation_state": (8,),
        "frozen_vlm_feature": (
            k,
            int(metadata["frozen_vlm_feature_dim"]),
        ),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ArtifactError(f"{name} shape differs: {np.asarray(arrays[name]).shape} != {shape}")
    if actions.dtype.kind != "f" or not np.all(np.isfinite(actions)):
        raise ArtifactError("candidate actions must be finite floating point")
    feature = np.asarray(arrays["frozen_vlm_feature"])
    if feature.dtype.kind != "f" or not np.all(np.isfinite(feature)):
        raise ArtifactError("frozen VLM features must be finite floating point")
    deviation = float(np.max(np.abs(feature.astype(np.float32) - feature[:1].astype(np.float32))))
    if not np.isclose(deviation, float(metadata["frozen_vlm_feature_max_deviation"])):
        raise ArtifactError("VLM feature deviation metadata disagrees with the data")
    if deviation > float(metadata["frozen_vlm_feature_tolerance"]):
        raise ArtifactError("candidate-invariant frozen VLM feature changed within a snapshot")
    if not np.array_equal(arrays["candidate_ids"], np.arange(k)):
        raise ArtifactError("candidate ids do not match the candidate axis")
    if not np.array_equal(np.sort(arrays["execution_order"]), np.arange(k)):
        raise ArtifactError("candidate execution order is not a permutation")
    rows = np.asarray(arrays["server_trace_rows"], dtype=np.int64)
    if not np.array_equal(rows, np.arange(rows[0], rows[0] + k)):
        raise ArtifactError("candidate routing rows are not one contiguous pool")
    if int(metadata["server_row_start"]) != int(rows[0]) or int(
        metadata["server_row_stop"]
    ) != int(rows[-1]) + 1:
        raise ArtifactError("candidate routing-row metadata disagrees with the data")
    store_ids = np.asarray(arrays["store_ids"]).astype(str)
    if len(set(store_ids.tolist())) != 1:
        raise ArtifactError("candidate pool spans multiple route-store identities")
    if str(store_ids[0]) != str(metadata["route_store_id"]):
        raise ArtifactError("candidate route-store identity disagrees with metadata")
    executed = np.asarray(arrays["candidate_action_executed"], dtype=np.bool_)
    for row in executed:
        if np.any(np.diff(row.astype(np.int8)) > 0):
            raise ArtifactError("candidate action execution mask is not a prefix")
    if not bool(np.asarray(arrays["fidelity_passed"]).item()):
        raise ArtifactError("candidate replay fidelity failed")


def validate_continuation(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]
) -> None:
    k = int(metadata["candidate_count"])
    repeats = int(metadata["repeat_stop"]) - int(metadata["repeat_start"])
    b = int(metadata["maximum_future_queries"])
    h = int(metadata["action_chunk_horizon"])
    expected = {
        "terminal_success": (k, repeats),
        "censored_at_budget": (k, repeats),
        "continuation_action_steps": (k, repeats),
        "total_environment_steps": (k, repeats),
        "continuation_flow_noise": (repeats, b, *FLOW_NOISE_SHAPE),
        "query_ids": (k, repeats, b),
        "query_executed": (k, repeats, b),
        "continuation_actions": (k, repeats, b, h, 7),
        "continuation_action_executed": (k, repeats, b, h),
        "server_trace_rows": (k, repeats, b),
        "server_row_counts": (k, repeats, b),
        "server_durable_through_rows": (k, repeats, b),
        "terminal_final_sim_states": (
            k,
            repeats,
            np.asarray(arrays["terminal_final_sim_states"]).shape[-1],
        ),
    }
    for name, shape in expected.items():
        if np.asarray(arrays[name]).shape != shape:
            raise ArtifactError(f"{name} shape differs: {np.asarray(arrays[name]).shape} != {shape}")
    if np.any(np.asarray(arrays["server_trace_rows"]) != -1):
        raise ArtifactError("terminal continuations must not capture routing rows")
    query = np.asarray(arrays["query_ids"])
    executed = np.asarray(arrays["query_executed"], dtype=np.bool_)
    if np.any(query[executed] < 0) or np.any(query[~executed] != -1):
        raise ArtifactError("continuation query sentinels are inconsistent")
    total_steps = np.asarray(arrays["total_environment_steps"])
    terminal_budget = int(metadata["terminal_environment_steps"])
    if np.any(total_steps > terminal_budget):
        raise ArtifactError("a continuation exceeded the terminal environment budget")
    success = np.asarray(arrays["terminal_success"], dtype=np.bool_)
    censored = np.asarray(arrays["censored_at_budget"], dtype=np.bool_)
    if not np.array_equal(censored, (~success) & (total_steps == terminal_budget)):
        raise ArtifactError("terminal censoring flags disagree with success and budget")
    action_executed = np.asarray(arrays["continuation_action_executed"], dtype=np.bool_)
    if np.any(action_executed & ~executed[..., None]):
        raise ArtifactError("an unsent continuation query contains executed actions")
    if not np.array_equal(
        action_executed.sum(axis=(2, 3)),
        np.asarray(arrays["continuation_action_steps"]),
    ):
        raise ArtifactError("continuation action-step counters disagree with masks")
    for row in action_executed.reshape((-1, h)):
        if np.any(np.diff(row.astype(np.int8)) > 0):
            raise ArtifactError("continuation action execution mask is not a prefix")
    if b:
        for row in executed.reshape((-1, b)):
            if np.any(np.diff(row.astype(np.int8)) > 0):
                raise ArtifactError("continuation query execution mask is not a prefix")
    row_counts = np.asarray(arrays["server_row_counts"], dtype=np.int64)
    durable = np.asarray(arrays["server_durable_through_rows"], dtype=np.int64)
    if np.any(row_counts[executed] != int(metadata["recorder_row_count"])):
        raise ArtifactError("an uncaptured query observed a changing recorder row count")
    if np.any(durable[executed] != int(metadata["recorder_durable_through_row"])):
        raise ArtifactError("an uncaptured query observed a changing durable boundary")
    if np.any(row_counts[~executed] != -1) or np.any(durable[~executed] != -1):
        raise ArtifactError("unused continuation slots have invalid recorder sentinels")
