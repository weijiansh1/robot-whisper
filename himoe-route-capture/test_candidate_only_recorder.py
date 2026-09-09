from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from himoe_candidate_capture_protocol import (
    ACK_KEY,
    CANDIDATE_ID_KEY,
    CAPTURE_KEY,
    FLUSH_AFTER_KEY,
    QUERY_ID_KEY,
    SCHEMA,
    SNAPSHOT_INDEX_KEY,
    CandidateCaptureCoordinator,
    CaptureIdentity,
    audit_candidate_capture_stores,
    pop_capture_directive,
)
from himoe_flow_trajectory_store import ZarrFlowTrajectoryWriter
from himoe_hidden_store import ZarrHiddenWriter
from himoe_route_store import ZarrRouteWriter
from serve_flow_trace import FlowTracer
from serve_with_recorder import (
    RequestGatedCandidateRecorder,
    candidate_capture_metadata,
    capture_run_id_for_resume,
)


RUN_ID = "0123456789abcdef0123456789abcdef"


def _writers(root, *, resume: bool = False):
    common = {
        "overwrite": not resume,
        "resume": resume,
        "capture_run_id": RUN_ID,
        "auto_flush": False,
    }
    route = ZarrRouteWriter(
        str(root / "routes.zarr"),
        n_hb_layers=1,
        n_as_layers=1,
        n_denoise=2,
        n_suffix=3,
        top_k=2,
        n_hb_experts=4,
        n_as_experts=3,
        store_full_probs=True,
        chunk_steps=4,
        **common,
    )
    hidden = ZarrHiddenWriter(
        str(root / "hidden.zarr"),
        n_hb_layers=1,
        n_as_layers=1,
        n_denoise=2,
        n_suffix=3,
        hb_hidden_dim=5,
        as_hidden_dim=2,
        chunk_steps=4,
        **common,
    )
    flow = ZarrFlowTrajectoryWriter(
        str(root / "flow_trajectory.zarr"),
        n_denoise=2,
        n_action_steps=2,
        max_action_dim=4,
        chunk_queries=4,
        **common,
    )
    return route, hidden, flow


def _record(query_id: int, ordinal: int):
    return SimpleNamespace(
        episode_id=query_id,
        control_step=ordinal,
        as_collapsed=True,
        hb_expert_ids=np.zeros((1, 1, 2, 3, 2), dtype=np.uint8),
        hb_selected_prob=np.full((1, 1, 2, 3, 2), 0.25, dtype=np.float16),
        hb_entropy=np.ones((1, 1, 2, 3), dtype=np.float16),
        hb_router_probs=np.full((1, 1, 2, 3, 4), 0.25, dtype=np.float16),
        as_expert_ids=np.zeros((1, 1), dtype=np.uint8),
        as_probs=np.full((1, 1, 3), 1 / 3, dtype=np.float16),
        hb_hidden=np.zeros((1, 1, 2, 3, 5), dtype=np.float16),
        as_hidden=np.zeros((1, 1, 2, 3, 2), dtype=np.float16),
    )


def _identity(row: int, query_id: int, ordinal: int) -> CaptureIdentity:
    return CaptureIdentity(
        capture_row=row,
        query_id=query_id,
        snapshot_index=2,
        candidate_id=row,
        local_call_ordinal=ordinal,
        observation_sha256=b"o" * 32,
        flow_noise_sha256=b"n" * 32,
        actions_sha256=b"a" * 32,
    )


def _trajectory() -> np.ndarray:
    return np.zeros((1, 3, 2, 4), dtype=np.float32)


def test_protocol_uses_only_frozen_prefixed_request_keys():
    request = {
        CAPTURE_KEY: True,
        QUERY_ID_KEY: 7,
        SNAPSHOT_INDEX_KEY: 2,
        CANDIDATE_ID_KEY: 5,
        FLUSH_AFTER_KEY: True,
        "prompt": "keep",
    }
    directive = pop_capture_directive(request)
    assert directive.query_id == 7
    assert directive.snapshot_index == 2
    assert directive.candidate_id == 5
    assert directive.flush_after is True
    assert request == {"prompt": "keep"}

    with pytest.raises(ValueError, match="recorder/query_id"):
        pop_capture_directive({CAPTURE_KEY: False, "query_id": 7})


def test_metadata_advertises_exact_request_and_nested_ack_contract():
    metadata = candidate_capture_metadata(RUN_ID, row_count=8, durable_through_row=7)
    assert metadata["candidate_capture_schema"] == SCHEMA
    assert metadata["candidate_capture_request_keys"] == {
        "capture": "recorder/capture",
        "query_id": "recorder/query_id",
        "snapshot_index": "recorder/snapshot_index",
        "candidate_id": "recorder/candidate_id",
        "flush_after": "recorder/flush_after",
    }
    assert metadata["candidate_capture_ack_key"] == "recorder/ack"
    assert metadata["candidate_capture_initial_row_count"] == 8
    assert metadata["candidate_capture_initial_durable_through_row"] == 7
    assert metadata["candidate_capture_protocol"]["response"]["ack_key"] == ACK_KEY
    assert metadata["candidate_capture_protocol"]["initial_state"] == {
        "row_count": 8,
        "durable_through_row": 7,
    }


def test_coordinator_assigns_one_row_and_audits_identity_spine(tmp_path):
    stores = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(*stores)
    row = coordinator.append(_record(10, 4), _identity(0, 10, 4), _trajectory())
    assert row == 0
    assert coordinator.rows == 1
    assert coordinator.durable_through_row == -1
    assert coordinator.flush() == 0
    assert audit_candidate_capture_stores(*stores) == {
        "capture_run_id": RUN_ID,
        "rows": 1,
        "common_durable_rows": 1,
        "first_query_id": 10,
        "last_query_id": 10,
        "last_local_call_ordinal": 4,
    }


def test_append_failure_rolls_back_all_three_pending_rows(tmp_path, monkeypatch):
    route, hidden, flow = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(route, hidden, flow)

    def fail_append(*_args, **_kwargs):
        raise OSError("injected flow append failure")

    monkeypatch.setattr(flow, "append", fail_append)
    with pytest.raises(OSError, match="injected"):
        coordinator.append(_record(10, 0), _identity(0, 10, 0), _trajectory())
    assert [writer.rows for writer in (route, hidden, flow)] == [0, 0, 0]
    assert [writer.durable_rows for writer in (route, hidden, flow)] == [0, 0, 0]


def test_flush_failure_restores_prior_common_boundary(tmp_path, monkeypatch):
    route, hidden, flow = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(route, hidden, flow)
    coordinator.append(_record(10, 0), _identity(0, 10, 0), _trajectory())
    coordinator.flush()
    coordinator.append(_record(11, 1), _identity(1, 11, 1), _trajectory())

    def fail_flush():
        raise OSError("injected hidden flush failure")

    monkeypatch.setattr(hidden, "flush", fail_flush)
    with pytest.raises(OSError, match="injected"):
        coordinator.flush()
    assert [writer.rows for writer in (route, hidden, flow)] == [1, 1, 1]
    assert [writer.durable_rows for writer in (route, hidden, flow)] == [1, 1, 1]
    assert [int(writer.arrays["episode_id"].shape[0]) for writer in (route, hidden)] == [1, 1]
    assert int(flow.arrays["query_id"].shape[0]) == 1


def test_resume_truncates_all_stores_to_declared_common_boundary(tmp_path):
    stores = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(*stores)
    coordinator.append(_record(10, 4), _identity(0, 10, 4), _trajectory())
    coordinator.flush()
    coordinator.append(_record(11, 8), _identity(1, 11, 8), _trajectory())
    coordinator.flush()
    for writer in stores:
        writer.root.attrs[CandidateCaptureCoordinator.COMMON_DURABLE_ATTR] = 1

    resumed = _writers(tmp_path, resume=True)
    resumed_coordinator = CandidateCaptureCoordinator(*resumed)
    assert capture_run_id_for_resume(tmp_path) == RUN_ID
    assert resumed_coordinator.rows == 1
    assert resumed_coordinator.durable_rows == 1
    assert resumed_coordinator.next_local_call_ordinal == 5
    resumed_coordinator.append(
        _record(99, 5),
        _identity(1, 99, 5),
        _trajectory(),
    )
    resumed_coordinator.flush()
    np.testing.assert_array_equal(resumed[2].arrays["capture_row"][:], [0, 1])
    np.testing.assert_array_equal(resumed[2].arrays["query_id"][:], [10, 99])


def test_coordinator_close_discards_partial_pool_before_resume(tmp_path):
    stores = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(*stores)
    coordinator.append(_record(10, 0), _identity(0, 10, 0), _trajectory())
    coordinator.flush()

    coordinator.append(_record(11, 1), _identity(1, 11, 1), _trajectory())
    coordinator.append(_record(12, 2), _identity(2, 12, 2), _trajectory())
    assert coordinator.rows == 3
    assert coordinator.durable_rows == 1
    coordinator.close()
    assert coordinator.rows == coordinator.durable_rows == 1

    resumed = _writers(tmp_path, resume=True)
    resumed_coordinator = CandidateCaptureCoordinator(*resumed)
    assert resumed_coordinator.rows == resumed_coordinator.durable_rows == 1
    np.testing.assert_array_equal(resumed[2].arrays["capture_row"][:], [0])
    np.testing.assert_array_equal(resumed[2].arrays["query_id"][:], [10])


def test_explicit_resume_rows_truncates_a_durable_orphan_pool(tmp_path):
    stores = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(*stores)
    coordinator.append(_record(10, 0), _identity(0, 10, 0), _trajectory())
    coordinator.flush()
    coordinator.append(_record(11, 1), _identity(1, 11, 1), _trajectory())
    coordinator.flush()
    assert coordinator.rows == coordinator.durable_rows == 2

    resumed = _writers(tmp_path, resume=True)
    resumed_coordinator = CandidateCaptureCoordinator(*resumed, resume_rows=1)
    assert resumed_coordinator.rows == resumed_coordinator.durable_rows == 1
    assert [writer.root.attrs["common_durable_rows"] for writer in resumed] == [1, 1, 1]
    np.testing.assert_array_equal(resumed[0].arrays["episode_id"][:], [10])
    np.testing.assert_array_equal(resumed[1].arrays["episode_id"][:], [10])
    np.testing.assert_array_equal(resumed[2].arrays["query_id"][:], [10])

    reopened = _writers(tmp_path, resume=True)
    with pytest.raises(ValueError, match="within the recovered"):
        CandidateCaptureCoordinator(*reopened, resume_rows=2)


def test_legacy_writer_close_still_flushes_pending_rows(tmp_path):
    writer = ZarrRouteWriter(
        str(tmp_path / "legacy-routes.zarr"),
        n_hb_layers=1,
        n_as_layers=1,
        n_denoise=2,
        n_suffix=3,
        top_k=2,
        n_hb_experts=4,
        n_as_experts=3,
        store_full_probs=True,
        chunk_steps=8,
    )
    writer.append(_record(10, 0))
    assert writer.durable_rows == 0
    assert writer.rows == 1
    writer.close()
    assert writer.durable_rows == writer.rows == 1


class _FakeTensor:
    def __init__(self, value, transfer_counter):
        self.value = np.asarray(value, dtype=np.float32)
        self.transfer_counter = transfer_counter

    def detach(self):
        return self

    def to(self, device, *, copy):
        assert device == "cpu"
        assert copy is True
        self.transfer_counter[0] += 1
        return _FakeTensor(self.value.copy(), self.transfer_counter)

    def float(self):
        return self

    def numpy(self):
        return self.value


class _FakeFlowModel:
    def __init__(self, transfer_counter):
        self.config = SimpleNamespace(num_steps=2)
        self.transfer_counter = transfer_counter

    def denoise_step(self, *_args, **_kwargs):
        return _FakeTensor(np.ones((1, 2, 4)), self.transfer_counter)


def test_flow_tracer_has_zero_cpu_copy_while_disabled_and_lifecycle_is_explicit():
    transfers = [0]
    model = _FakeFlowModel(transfers)
    tracer = FlowTracer(model)
    x0 = _FakeTensor(np.zeros((1, 2, 4)), transfers)
    x1 = _FakeTensor(np.full((1, 2, 4), -0.5), transfers)
    args0 = (None, None, None, None, None, x0, None)
    args1 = (None, None, None, None, None, x1, None)

    model.denoise_step(*args0)
    assert transfers == [0]
    tracer.begin()
    model.denoise_step(*args0)
    model.denoise_step(*args1)
    trajectory, residual = tracer.finish()
    assert transfers == [4]
    assert trajectory.shape == (1, 3, 2, 4)
    assert residual == 0.0
    assert tracer.enabled is False
    model.denoise_step(*args0)
    assert transfers == [4]

    tracer.begin()
    tracer.cancel()
    with pytest.raises(RuntimeError, match="no active"):
        tracer.trajectory()
    tracer.close()


class _FakeRecorder:
    def __init__(self):
        self.enabled = False
        self._buf = {}
        self.begin_calls = 0

    def begin_control_step(self, *, episode_id, control_step):
        self.enabled = True
        self.episode_id = episode_id
        self.control_step = control_step
        self.begin_calls += 1

    def end_control_step(self):
        self.enabled = False
        return _record(self.episode_id, self.control_step)


class _FakeTracer:
    def __init__(self):
        self.enabled = False
        self.begin_calls = 0

    def begin(self):
        self.enabled = True
        self.begin_calls += 1

    def finish(self):
        self.enabled = False
        return _trajectory(), 0.0

    def cancel(self):
        self.enabled = False


def _request(query_id: int, *, capture: bool, flush: bool = False):
    request = {
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/state": np.zeros(8, dtype=np.float32),
        "prompt": "test",
        "flow/noise": np.zeros((2, 4), dtype=np.float32),
        CAPTURE_KEY: capture,
        QUERY_ID_KEY: query_id,
    }
    if capture:
        request[SNAPSHOT_INDEX_KEY] = 3
        request[CANDIDATE_ID_KEY] = query_id
        request[FLUSH_AFTER_KEY] = flush
    return request


def test_request_gate_skips_continuations_and_ack_survives_response_validation(tmp_path):
    from himoe_libero_bridge.protocol import validate_action_response

    stores = _writers(tmp_path)
    coordinator = CandidateCaptureCoordinator(*stores)
    recorder = _FakeRecorder()
    tracer = _FakeTracer()

    def inner(request):
        assert not any(key.startswith("recorder/") for key in request)
        noise = np.ascontiguousarray(request["flow/noise"], dtype=np.float32)
        return {
            "actions": np.zeros((10, 7), dtype=np.float32),
            "flow/noise_sha256": hashlib.sha256(noise.tobytes()).hexdigest(),
        }

    handler = RequestGatedCandidateRecorder(inner, recorder, tracer, coordinator)
    continuation0 = handler.infer(_request(1, capture=False))[ACK_KEY]
    assert continuation0 == {
        "schema": SCHEMA,
        "capture_run_id": RUN_ID,
        "query_id": 1,
        "captured": False,
        "row": -1,
        "row_count": 0,
        "durable_through_row": -1,
    }
    assert recorder.begin_calls == tracer.begin_calls == 0

    candidate0 = handler.infer(_request(2, capture=True))[ACK_KEY]
    assert candidate0["row"] == 0
    assert candidate0["row_count"] == 1
    assert candidate0["durable_through_row"] == -1
    continuation1 = handler.infer(_request(3, capture=False))[ACK_KEY]
    assert continuation1["row"] == -1
    assert continuation1["row_count"] == 1
    assert continuation1["durable_through_row"] == -1
    assert recorder.begin_calls == tracer.begin_calls == 1

    response = handler.infer(_request(4, capture=True, flush=True))
    validated = validate_action_response(response)
    assert validated[ACK_KEY] == response[ACK_KEY]
    assert response[ACK_KEY]["row"] == 1
    assert response[ACK_KEY]["row_count"] == 2
    assert response[ACK_KEY]["durable_through_row"] == 1
    assert recorder.begin_calls == tracer.begin_calls == 2
    np.testing.assert_array_equal(stores[2].arrays["query_id"][:], [2, 4])
    np.testing.assert_array_equal(stores[2].arrays["local_call_ordinal"][:], [1, 3])
