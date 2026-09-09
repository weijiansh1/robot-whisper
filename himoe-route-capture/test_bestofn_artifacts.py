import numpy as np
import pytest

from behavior_forks_v2 import ArtifactError
from bestofn_artifacts import validate_candidate, validate_continuation
from bestofn_protocol import FLOW_NOISE_SHAPE


def _candidate():
    k, h, width = 2, 10, 13
    arrays = {
        "candidate_ids": np.arange(k),
        "actions": np.zeros((k, h, 7), dtype=np.float32),
        "candidate_flow_noise": np.zeros((k, *FLOW_NOISE_SHAPE), dtype=np.float32),
        "query_ids": np.asarray([10, 11]),
        "server_trace_rows": np.asarray([5, 6]),
        "store_ids": np.asarray(["run", "run"]),
        "execution_order": np.asarray([1, 0]),
        "candidate_action_executed": np.ones((k, h), dtype=np.bool_),
        "sim_states": np.zeros((k, h + 1, width)),
        "eef_positions": np.zeros((k, h + 1, 3)),
        "eef_quaternions": np.zeros((k, h + 1, 4)),
        "chunk_success": np.zeros((k, h + 1), dtype=np.bool_),
        "chunk_terminal_success": np.zeros(k, dtype=np.bool_),
        "observation_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation_wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation_state": np.zeros(8, dtype=np.float32),
        "frozen_vlm_feature": np.ones((k, 6), dtype=np.float16),
        "fidelity_passed": np.asarray(True),
    }
    metadata = {
        "candidate_count": k,
        "action_chunk_horizon": h,
        "server_row_start": 5,
        "server_row_stop": 7,
        "route_store_id": "run",
        "frozen_vlm_feature_dim": 6,
        "frozen_vlm_feature_max_deviation": 0.0,
        "frozen_vlm_feature_tolerance": 0.01,
    }
    return arrays, metadata


def test_candidate_validator_checks_route_and_vlm_alignment():
    arrays, metadata = _candidate()
    validate_candidate(arrays, metadata)
    arrays["frozen_vlm_feature"][1, 0] = 2
    with pytest.raises(ArtifactError, match="deviation metadata"):
        validate_candidate(arrays, metadata)


def _continuation():
    k, repeats, b, h, width = 2, 4, 2, 10, 13
    executed = np.ones((k, repeats, b), dtype=np.bool_)
    action = np.ones((k, repeats, b, h), dtype=np.bool_)
    success = np.zeros((k, repeats), dtype=np.bool_)
    success[0] = True
    total = np.full((k, repeats), 300, dtype=np.int32)
    total[0] = 100
    arrays = {
        "terminal_success": success,
        "censored_at_budget": ~success,
        "continuation_action_steps": action.sum(axis=(2, 3)).astype(np.int32),
        "total_environment_steps": total,
        "continuation_flow_noise": np.zeros((repeats, b, *FLOW_NOISE_SHAPE), dtype=np.float32),
        "query_ids": np.arange(k * repeats * b).reshape(k, repeats, b),
        "query_executed": executed,
        "continuation_actions": np.zeros((k, repeats, b, h, 7), dtype=np.float32),
        "continuation_action_executed": action,
        "server_trace_rows": np.full((k, repeats, b), -1),
        "server_row_counts": np.full((k, repeats, b), 7),
        "server_durable_through_rows": np.full((k, repeats, b), 6),
        "terminal_final_sim_states": np.zeros((k, repeats, width)),
    }
    metadata = {
        "candidate_count": k,
        "repeat_start": 0,
        "repeat_stop": repeats,
        "maximum_future_queries": b,
        "action_chunk_horizon": h,
        "terminal_environment_steps": 300,
        "recorder_row_count": 7,
        "recorder_durable_through_row": 6,
    }
    return arrays, metadata


def test_continuation_validator_checks_uncaptured_recorder_boundary():
    arrays, metadata = _continuation()
    validate_continuation(arrays, metadata)
    arrays["server_trace_rows"][0, 0, 0] = 7
    with pytest.raises(ArtifactError, match="must not capture"):
        validate_continuation(arrays, metadata)
