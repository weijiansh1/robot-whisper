"""Request and acknowledgement contract for selective candidate capture."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Any, MutableMapping

import numpy as np


SCHEMA = "himoe.candidate_capture.v1"
CAPTURE_KEY = "recorder/capture"
QUERY_ID_KEY = "recorder/query_id"
SNAPSHOT_INDEX_KEY = "recorder/snapshot_index"
CANDIDATE_ID_KEY = "recorder/candidate_id"
FLUSH_AFTER_KEY = "recorder/flush_after"
ACK_KEY = "recorder/ack"
VLM_FEATURE_KEY = "recorder/vlm_feature"

OBSERVATION_KEYS = (
    "observation/image",
    "observation/wrist_image",
    "observation/state",
    "prompt",
)
FLOW_NOISE_KEY = "flow/noise"


@dataclass(frozen=True)
class CaptureDirective:
    query_id: int
    capture: bool
    snapshot_index: int
    candidate_id: int
    flush_after: bool


@dataclass(frozen=True)
class CaptureIdentity:
    capture_row: int
    query_id: int
    snapshot_index: int
    candidate_id: int
    local_call_ordinal: int
    observation_sha256: bytes
    flow_noise_sha256: bytes
    actions_sha256: bytes


def _pop_int(request: MutableMapping[str, Any], key: str, *, required: bool) -> int:
    if key not in request:
        if required:
            raise ValueError("request is missing %s" % key)
        return -1
    raw = request.pop(key)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError("%s must be an integer" % key)
    if not isinstance(raw, (int, np.integer)):
        raise ValueError("%s must be an integer" % key)
    value = int(raw)
    if value < 0:
        raise ValueError("%s must be non-negative" % key)
    return value


def pop_capture_directive(request: MutableMapping[str, Any]) -> CaptureDirective:
    """Validate and remove every recorder-only request field."""

    if CAPTURE_KEY not in request:
        raise ValueError("request-gated recorder requires %s" % CAPTURE_KEY)
    raw_capture = request.pop(CAPTURE_KEY)
    if not isinstance(raw_capture, (bool, np.bool_)):
        raise ValueError("%s must be bool" % CAPTURE_KEY)
    capture = bool(raw_capture)
    query_id = _pop_int(request, QUERY_ID_KEY, required=True)
    if query_id > np.iinfo(np.int32).max:
        raise ValueError("query_id must fit the recorder int32 identity field")
    snapshot_index = _pop_int(request, SNAPSHOT_INDEX_KEY, required=capture)
    candidate_id = _pop_int(request, CANDIDATE_ID_KEY, required=capture)
    for name, value in (
        (SNAPSHOT_INDEX_KEY, snapshot_index),
        (CANDIDATE_ID_KEY, candidate_id),
    ):
        if value > np.iinfo(np.int32).max:
            raise ValueError("%s must fit int32 storage" % name)

    raw_flush = request.pop(FLUSH_AFTER_KEY, False)
    if not isinstance(raw_flush, (bool, np.bool_)):
        raise ValueError("flush_after must be bool")
    flush_after = bool(raw_flush)
    if flush_after and not capture:
        raise ValueError("flush_after is only valid for a captured request")
    return CaptureDirective(
        query_id=query_id,
        capture=capture,
        snapshot_index=snapshot_index,
        candidate_id=candidate_id,
        flush_after=flush_after,
    )


def _hash_field(digest: Any, key: str, value: Any) -> None:
    key_bytes = key.encode("utf-8")
    digest.update(struct.pack("<I", len(key_bytes)))
    digest.update(key_bytes)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        digest.update(b"S" + struct.pack("<Q", len(payload)) + payload)
        return
    array = np.ascontiguousarray(value)
    dtype = array.dtype.str.encode("ascii")
    digest.update(b"A" + struct.pack("<I", len(dtype)) + dtype)
    digest.update(struct.pack("<I", array.ndim))
    digest.update(struct.pack("<" + "q" * array.ndim, *array.shape))
    digest.update(array.tobytes())


def request_digests(request: MutableMapping[str, Any]) -> tuple[bytes, bytes]:
    """Hash the policy observation separately from its explicit flow noise."""

    missing = [key for key in (*OBSERVATION_KEYS, FLOW_NOISE_KEY) if key not in request]
    if missing:
        raise ValueError("captured request is missing %s" % missing)
    observation = hashlib.sha256()
    for key in OBSERVATION_KEYS:
        _hash_field(observation, key, request[key])
    noise = np.ascontiguousarray(request[FLOW_NOISE_KEY], dtype=np.float32)
    if noise.ndim != 2 or not np.all(np.isfinite(noise)):
        raise ValueError("captured flow/noise must be one finite 2-D array")
    return observation.digest(), hashlib.sha256(noise.tobytes()).digest()


def array_digest(value: Any) -> bytes:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).digest()


def capture_ack(
    *,
    capture_run_id: str,
    query_id: int,
    captured: bool,
    row: int,
    row_count: int,
    durable_through_row: int,
) -> dict[str, Any]:
    if not isinstance(capture_run_id, str) or not capture_run_id:
        raise ValueError("capture ack requires a non-empty capture_run_id")
    if int(query_id) < 0:
        raise ValueError("capture ack query_id must be non-negative")
    row = int(row)
    row_count = int(row_count)
    durable_through_row = int(durable_through_row)
    if captured and (row < 0 or row_count != row + 1):
        raise ValueError("captured ack must report row_count == row + 1")
    if not captured and row != -1:
        raise ValueError("non-captured ack must use row=-1")
    if row_count < 0 or not -1 <= durable_through_row < row_count:
        raise ValueError("ack durable boundary is outside row_count")
    return {
        "schema": SCHEMA,
        "capture_run_id": str(capture_run_id),
        "query_id": int(query_id),
        "captured": bool(captured),
        "row": row,
        "row_count": row_count,
        "durable_through_row": durable_through_row,
    }


def _writer_run_id(writer: Any) -> str:
    value = writer.root.attrs.get("capture_run_id")
    if not isinstance(value, str) or not value:
        raise ValueError("capture store is missing a non-empty capture_run_id")
    return value


def audit_candidate_capture_stores(
    route_writer: Any,
    hidden_writer: Any,
    flow_writer: Any,
    *,
    rows: int | None = None,
) -> dict[str, Any]:
    """Audit the durable identity spine shared by the three capture stores."""

    writers = (route_writer, hidden_writer, flow_writer)
    run_ids = {_writer_run_id(writer) for writer in writers}
    if len(run_ids) != 1:
        raise RuntimeError("candidate capture stores have different capture_run_id values")
    lengths = [int(writer.durable_rows) for writer in writers]
    if len(set(lengths)) != 1:
        raise RuntimeError("candidate capture stores have different durable row counts: %s" % lengths)
    common = [
        int(writer.root.attrs.get(CandidateCaptureCoordinator.COMMON_DURABLE_ATTR, -1))
        for writer in writers
    ]
    if len(set(common)) != 1 or not 0 <= common[0] <= lengths[0]:
        raise RuntimeError("candidate capture stores disagree on common durable rows: %s" % common)
    stop = lengths[0] if rows is None else int(rows)
    if stop < 0 or stop > lengths[0]:
        raise ValueError("audit row count is outside the common durable prefix")

    route_query = np.asarray(route_writer.arrays["episode_id"][:stop], dtype=np.int64)
    hidden_query = np.asarray(hidden_writer.arrays["episode_id"][:stop], dtype=np.int64)
    flow_query = np.asarray(flow_writer.arrays["query_id"][:stop], dtype=np.int64)
    if not (np.array_equal(route_query, hidden_query) and np.array_equal(route_query, flow_query)):
        raise RuntimeError("query_id identity differs across candidate capture stores")
    if np.unique(flow_query).size != stop:
        raise RuntimeError("captured query_id values are not unique")

    route_call = np.asarray(route_writer.arrays["control_step"][:stop], dtype=np.int64)
    hidden_call = np.asarray(hidden_writer.arrays["control_step"][:stop], dtype=np.int64)
    flow_call = np.asarray(flow_writer.arrays["local_call_ordinal"][:stop], dtype=np.int64)
    if not (np.array_equal(route_call, hidden_call) and np.array_equal(route_call, flow_call)):
        raise RuntimeError("local call ordinal differs across candidate capture stores")
    if stop > 1 and np.any(np.diff(flow_call) <= 0):
        raise RuntimeError("captured local call ordinals are not strictly increasing")
    capture_rows = np.asarray(flow_writer.arrays["capture_row"][:stop], dtype=np.int64)
    if not np.array_equal(capture_rows, np.arange(stop, dtype=np.int64)):
        raise RuntimeError("flow capture_row is not the contiguous server-assigned row axis")

    return {
        "capture_run_id": run_ids.pop(),
        "rows": stop,
        "common_durable_rows": common[0],
        "first_query_id": None if stop == 0 else int(flow_query[0]),
        "last_query_id": None if stop == 0 else int(flow_query[-1]),
        "last_local_call_ordinal": None if stop == 0 else int(flow_call[-1]),
    }


class CandidateCaptureCoordinator:
    """One logical append/durability boundary over route, hidden and flow stores."""

    COMMON_DURABLE_ATTR = "common_durable_rows"

    def __init__(
        self,
        route_writer: Any,
        hidden_writer: Any,
        flow_writer: Any,
        *,
        resume_rows: int | None = None,
    ) -> None:
        self.route_writer = route_writer
        self.hidden_writer = hidden_writer
        self.flow_writer = flow_writer
        self.writers = (route_writer, hidden_writer, flow_writer)
        if any(bool(getattr(writer, "auto_flush", True)) for writer in self.writers):
            raise ValueError("coordinated candidate stores must disable per-store auto_flush")
        run_ids = {_writer_run_id(writer) for writer in self.writers}
        if len(run_ids) != 1:
            raise ValueError("candidate capture stores must share one capture_run_id")
        self.capture_run_id = run_ids.pop()

        local_durable = [int(writer.durable_rows) for writer in self.writers]
        declared_common = [
            int(writer.root.attrs.get(self.COMMON_DURABLE_ATTR, local))
            for writer, local in zip(self.writers, local_durable, strict=True)
        ]
        target = min([*local_durable, *declared_common])
        if target < 0:
            raise ValueError("candidate capture stores declare a negative durable prefix")
        if resume_rows is not None:
            requested = int(resume_rows)
            if requested < 0 or requested > target:
                raise ValueError(
                    "resume_rows must be within the recovered common durable prefix"
                )
            target = requested
        for writer in self.writers:
            if writer.rows != writer.durable_rows:
                raise RuntimeError("cannot coordinate a store with pending rows")
            if writer.durable_rows > target:
                writer.truncate(target)
            writer.root.attrs[self.COMMON_DURABLE_ATTR] = target
        self._common_durable_rows = target
        audit_candidate_capture_stores(*self.writers, rows=target)
        self._query_ids = set(
            np.asarray(self.flow_writer.arrays["query_id"][:target], dtype=np.int64).tolist()
        )
        self._last_call_ordinal = (
            -1
            if target == 0
            else int(self.flow_writer.arrays["local_call_ordinal"][target - 1])
        )

    @property
    def rows(self) -> int:
        values = [int(writer.rows) for writer in self.writers]
        if len(set(values)) != 1:
            raise RuntimeError("candidate capture stores have different logical rows: %s" % values)
        return values[0]

    @property
    def durable_rows(self) -> int:
        return self._common_durable_rows

    @property
    def durable_through_row(self) -> int:
        return self._common_durable_rows - 1

    @property
    def next_local_call_ordinal(self) -> int:
        if self.rows == 0:
            return 0
        if self.rows != self.durable_rows:
            raise RuntimeError("next call ordinal is only defined at a durable boundary")
        return int(self.flow_writer.arrays["local_call_ordinal"][self.rows - 1]) + 1

    @staticmethod
    def _require_shape(name: str, value: Any, tail: tuple[int, ...]) -> None:
        if value is None or tuple(np.shape(value)) != (1, *tail):
            raise ValueError("%s must have shape %s" % (name, (1, *tail)))

    def _validate_append(self, record: Any, identity: CaptureIdentity, x_traj: Any) -> None:
        row = self.rows
        if int(identity.capture_row) != row:
            raise ValueError("capture identity does not name the next server-assigned row")
        if not 0 <= int(identity.query_id) <= np.iinfo(np.int32).max:
            raise ValueError("query_id does not fit route/hidden int32 storage")
        if int(identity.query_id) in self._query_ids:
            raise ValueError("query_id was already captured in this run")
        if not 0 <= int(identity.local_call_ordinal) <= np.iinfo(np.int32).max:
            raise ValueError("local_call_ordinal does not fit route/hidden int32 storage")
        if int(identity.local_call_ordinal) <= self._last_call_ordinal:
            raise ValueError("captured local call ordinals must be strictly increasing")
        for name, value in (
            ("snapshot_index", identity.snapshot_index),
            ("candidate_id", identity.candidate_id),
        ):
            if not 0 <= int(value) <= np.iinfo(np.int32).max:
                raise ValueError("%s does not fit flow store int32 storage" % name)
        if int(record.episode_id) != int(identity.query_id):
            raise ValueError("route record query_id differs from capture identity")
        if int(record.control_step) != int(identity.local_call_ordinal):
            raise ValueError("route record call ordinal differs from capture identity")
        if not bool(record.as_collapsed):
            raise ValueError("candidate capture requires collapsed AS routing")

        route_fields = {
            "hb_expert_ids": record.hb_expert_ids,
            "hb_selected_prob": record.hb_selected_prob,
            "hb_entropy": record.hb_entropy,
            "as_expert_ids": record.as_expert_ids,
            "as_probs": record.as_probs,
        }
        if "hb_router_probs" in self.route_writer.arrays:
            route_fields["hb_router_probs"] = record.hb_router_probs
        for name, value in route_fields.items():
            self._require_shape(name, value, tuple(self.route_writer.arrays[name].shape[1:]))
        for name, value in (
            ("hb_hidden", record.hb_hidden),
            ("as_hidden", record.as_hidden),
        ):
            self._require_shape(name, value, tuple(self.hidden_writer.arrays[name].shape[1:]))
        expected_traj = tuple(self.flow_writer.arrays["x_traj"].shape[1:])
        self._require_shape("x_traj", x_traj, expected_traj)
        if not np.all(np.isfinite(x_traj)):
            raise ValueError("x_traj must be finite")
        for name, digest in (
            ("observation_sha256", identity.observation_sha256),
            ("flow_noise_sha256", identity.flow_noise_sha256),
            ("actions_sha256", identity.actions_sha256),
        ):
            if len(bytes(digest)) != 32:
                raise ValueError("%s must contain 32 bytes" % name)

    def append(self, record: Any, identity: CaptureIdentity, x_traj: np.ndarray) -> int:
        """Append one batch-1 candidate, rolling back all buffers on failure."""

        self._validate_append(record, identity, x_traj)
        start = self.rows
        try:
            self.route_writer.append(record)
            self.hidden_writer.append(record)
            self.flow_writer.append(identity, x_traj)
            if self.rows != start + 1:
                raise RuntimeError("candidate capture append did not advance exactly one row")
        except Exception:
            for writer in self.writers:
                if writer.rows > start:
                    writer.rollback_pending(writer.rows - start)
            if any(writer.rows != start for writer in self.writers):
                raise RuntimeError("failed candidate append could not restore store alignment")
            raise
        self._query_ids.add(int(identity.query_id))
        self._last_call_ordinal = int(identity.local_call_ordinal)
        return start

    def _restore_common_boundary(self) -> None:
        for writer in self.writers:
            pending = writer.rows - writer.durable_rows
            if pending:
                writer.rollback_pending(pending)
            if writer.durable_rows > self._common_durable_rows:
                writer.truncate(self._common_durable_rows)
            writer.root.attrs[self.COMMON_DURABLE_ATTR] = self._common_durable_rows
        if any(writer.rows != self._common_durable_rows for writer in self.writers):
            raise RuntimeError("failed to restore the prior common durable boundary")
        query = np.asarray(
            self.flow_writer.arrays["query_id"][: self._common_durable_rows],
            dtype=np.int64,
        )
        self._query_ids = set(query.tolist())
        self._last_call_ordinal = (
            -1
            if self._common_durable_rows == 0
            else int(
                self.flow_writer.arrays["local_call_ordinal"][
                    self._common_durable_rows - 1
                ]
            )
        )

    def flush(self) -> int:
        """Synchronously make all logical rows durable, then advance the common marker."""

        target = self.rows
        try:
            for writer in self.writers:
                writer.flush()
            durable = [int(writer.durable_rows) for writer in self.writers]
            if durable != [target, target, target]:
                raise RuntimeError("candidate stores did not flush to one boundary: %s" % durable)
            audit_candidate_capture_stores(*self.writers, rows=target)
            for writer in self.writers:
                writer.root.attrs[self.COMMON_DURABLE_ATTR] = target
        except Exception:
            self._restore_common_boundary()
            raise
        self._common_durable_rows = target
        return self.durable_through_row

    def close(self) -> None:
        """Discard an incomplete pool; only explicit ``flush_after`` commits it."""

        self._restore_common_boundary()
