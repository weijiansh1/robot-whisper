from __future__ import annotations

import hashlib

import numpy as np
import pytest

from capture_behavior_study import (
    GatedQueryLedger,
    _require_gated_metadata,
)
from behavior_forks_v2 import ArtifactError
from himoe_candidate_capture_protocol import (
    ACK_KEY,
    CANDIDATE_ID_KEY,
    CAPTURE_KEY,
    FLUSH_AFTER_KEY,
    QUERY_ID_KEY,
    SCHEMA,
    SNAPSHOT_INDEX_KEY,
)
from himoe_libero_bridge.protocol import (
    ACTION_KEY,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    FLOW_NOISE_SHAPE,
)


RUN_ID = "capture-run-v2"


def _metadata() -> dict:
    return {
        "candidate_capture_supported": True,
        "candidate_capture_schema": SCHEMA,
        "candidate_capture_run_id": RUN_ID,
        "candidate_capture_request_keys": {
            "capture": CAPTURE_KEY,
            "query_id": QUERY_ID_KEY,
            "snapshot_index": SNAPSHOT_INDEX_KEY,
            "candidate_id": CANDIDATE_ID_KEY,
            "flush_after": FLUSH_AFTER_KEY,
        },
        "candidate_capture_ack_key": ACK_KEY,
    }


class _MemoryGatedServer:
    def __init__(
        self,
        *,
        row_count: int = 0,
        durable: int | None = None,
        run_id: str = RUN_ID,
        flush_durable_delta: int = 0,
    ) -> None:
        self.row_count = row_count
        self.durable = row_count - 1 if durable is None else durable
        self.run_id = run_id
        self.flush_durable_delta = flush_durable_delta
        self.requests: list[dict] = []

    def infer(self, request: dict) -> dict:
        self.requests.append(dict(request))
        capture = bool(request[CAPTURE_KEY])
        query_id = int(request[QUERY_ID_KEY])
        if capture:
            row = self.row_count
            self.row_count += 1
            if bool(request.get(FLUSH_AFTER_KEY, False)):
                self.durable = row + self.flush_durable_delta
        else:
            row = -1
            assert SNAPSHOT_INDEX_KEY not in request
            assert CANDIDATE_ID_KEY not in request
            assert FLUSH_AFTER_KEY not in request
        noise = np.ascontiguousarray(request[FLOW_NOISE_KEY], dtype=np.float32)
        return {
            ACTION_KEY: np.zeros((10, 7), dtype=np.float32),
            FLOW_NOISE_SHA256_KEY: hashlib.sha256(noise.tobytes()).hexdigest(),
            ACK_KEY: {
                "schema": SCHEMA,
                "capture_run_id": self.run_id,
                "query_id": query_id,
                "captured": capture,
                "row": row,
                "row_count": self.row_count,
                "durable_through_row": self.durable,
            },
        }


def _infer(ledger: GatedQueryLedger, query_id: int, capture: bool, flush: bool = False) -> None:
    ledger.infer(
        {},
        np.zeros(FLOW_NOISE_SHAPE, dtype=np.float32),
        query_id=query_id,
        capture=capture,
        snapshot_index=4,
        candidate_id=query_id,
        flush_after=flush,
        coordinates={"query": query_id},
        seed_entropy=(1, 2, 3),
    )


def test_gated_candidate_then_continuation_preserves_recorder_boundary() -> None:
    server = _MemoryGatedServer()
    ledger = GatedQueryLedger(server)  # type: ignore[arg-type]
    _infer(ledger, 10, True)
    _infer(ledger, 11, True, flush=True)
    assert [record["server_row"] for record in ledger.records] == [0, 1]
    assert ledger.records[-1]["durable_through_row"] == 1

    continuation = GatedQueryLedger(server)  # type: ignore[arg-type]
    _infer(continuation, 1000, False)
    _infer(continuation, 1001, False)
    assert continuation.uncaptured_boundary == (2, 1)
    assert server.row_count == 2
    assert all(record["server_row"] == -1 for record in continuation.records)


def test_first_captured_row_must_equal_artifact_confirmed_prefix() -> None:
    resumed = _MemoryGatedServer(row_count=7)
    ledger = GatedQueryLedger(
        resumed,  # type: ignore[arg-type]
        expected_capture_run_id=RUN_ID,
        expected_recorder_rows=7,
    )
    _infer(ledger, 10, True)
    assert ledger.records[0]["server_row"] == 7

    mismatched = _MemoryGatedServer(row_count=8)
    ledger = GatedQueryLedger(
        mismatched,  # type: ignore[arg-type]
        expected_capture_run_id=RUN_ID,
        expected_recorder_rows=7,
    )
    with pytest.raises(ArtifactError, match="artifact-confirmed prefix"):
        _infer(ledger, 11, True)


@pytest.mark.parametrize(
    ("row_count", "durable", "message"),
    [
        (7, 5, "pending recorder pool"),
        (8, 7, "artifact-confirmed prefix"),
    ],
)
def test_uncaptured_request_requires_exact_fully_durable_prefix(
    row_count: int,
    durable: int,
    message: str,
) -> None:
    valid = _MemoryGatedServer(row_count=7)
    ledger = GatedQueryLedger(
        valid,  # type: ignore[arg-type]
        expected_capture_run_id=RUN_ID,
        expected_recorder_rows=7,
    )
    _infer(ledger, 1000, False)
    assert ledger.uncaptured_boundary == (7, 6)

    invalid = _MemoryGatedServer(row_count=row_count, durable=durable)
    ledger = GatedQueryLedger(
        invalid,  # type: ignore[arg-type]
        expected_capture_run_id=RUN_ID,
        expected_recorder_rows=7,
    )
    with pytest.raises(ArtifactError, match=message):
        _infer(ledger, 1001, False)


@pytest.mark.parametrize("flush_durable_delta", [-1, 1])
def test_flush_requires_exact_final_row_durable(flush_durable_delta: int) -> None:
    server = _MemoryGatedServer(flush_durable_delta=flush_durable_delta)
    ledger = GatedQueryLedger(server, expected_recorder_rows=0)  # type: ignore[arg-type]
    with pytest.raises(ArtifactError, match="exact final candidate row durable"):
        _infer(ledger, 10, True, flush=True)


def test_expected_capture_run_id_mismatch_fails_closed() -> None:
    server = _MemoryGatedServer(run_id="another-capture-run")
    ledger = GatedQueryLedger(
        server,  # type: ignore[arg-type]
        expected_capture_run_id=RUN_ID,
        expected_recorder_rows=0,
    )
    with pytest.raises(ArtifactError, match="another capture_run_id"):
        _infer(ledger, 10, True)


def test_gated_metadata_requires_exact_request_and_ack_keys() -> None:
    assert _require_gated_metadata(_metadata()) == RUN_ID
    bad = _metadata()
    bad["candidate_capture_ack_key"] = "wrong/ack"
    with pytest.raises(ArtifactError, match="acknowledgement key"):
        _require_gated_metadata(bad)
