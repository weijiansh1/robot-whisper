"""Small resumable store for query-aligned flow-matching trajectories."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from himoe_candidate_capture_protocol import CaptureIdentity


FORMAT = "himoe_flow_trajectory_v1"


def _digest_array(value: bytes) -> np.ndarray:
    raw = bytes(value)
    if len(raw) != 32:
        raise ValueError("SHA-256 digests must contain 32 bytes")
    return np.frombuffer(raw, dtype=np.uint8).copy()


class ZarrFlowTrajectoryWriter:
    def __init__(
        self,
        path: str,
        *,
        n_denoise: int,
        n_action_steps: int,
        max_action_dim: int,
        chunk_queries: int = 32,
        zstd_level: int = 3,
        overwrite: bool = True,
        resume: bool = False,
        capture_run_id: str | None = None,
        auto_flush: bool = True,
    ) -> None:
        self.path = path
        self.n_denoise = int(n_denoise)
        self.n_action_steps = int(n_action_steps)
        self.max_action_dim = int(max_action_dim)
        self.chunk_queries = int(chunk_queries)
        self.auto_flush = bool(auto_flush)
        dimensions = (
            self.n_denoise,
            self.n_action_steps,
            self.max_action_dim,
            self.chunk_queries,
        )
        if min(dimensions) <= 0:
            raise ValueError("flow trajectory dimensions must be positive")

        c = self.chunk_queries
        specs = [
            (
                "x_traj",
                (self.n_denoise + 1, self.n_action_steps, self.max_action_dim),
                (max(1, c // 4), self.n_denoise + 1, self.n_action_steps, self.max_action_dim),
                "float32",
            ),
            ("capture_row", (), (c,), "int64"),
            ("query_id", (), (c,), "int64"),
            ("snapshot_index", (), (c,), "int32"),
            ("candidate_id", (), (c,), "int32"),
            ("local_call_ordinal", (), (c,), "int64"),
            ("observation_sha256", (32,), (c, 32), "uint8"),
            ("flow_noise_sha256", (32,), (c, 32), "uint8"),
            ("actions_sha256", (32,), (c, 32), "uint8"),
        ]
        store_path = pathlib.Path(path)
        if resume:
            if overwrite:
                raise ValueError("resume and overwrite are mutually exclusive")
            if not store_path.exists():
                raise FileNotFoundError("flow trajectory store does not exist for resume: %s" % path)
            self.root = zarr.open_group(path, mode="a")
            expected_attrs = {
                "format": FORMAT,
                "n_denoise": self.n_denoise,
                "n_flow_states": self.n_denoise + 1,
                "n_action_steps": self.n_action_steps,
                "max_action_dim": self.max_action_dim,
            }
            for name, expected in expected_attrs.items():
                if self.root.attrs.get(name) != expected:
                    raise ValueError(
                        "flow store attribute %s is %r, expected %r"
                        % (name, self.root.attrs.get(name), expected)
                    )
            existing_run_id = self.root.attrs.get("capture_run_id")
            if capture_run_id is not None and existing_run_id != capture_run_id:
                raise ValueError("flow store capture_run_id mismatch")
        else:
            if not overwrite and store_path.exists():
                raise FileExistsError("flow trajectory store already exists: %s" % path)
            self.root = zarr.create_group(store=path, overwrite=overwrite)
            attrs = {
                "format": FORMAT,
                "n_denoise": self.n_denoise,
                "n_flow_states": self.n_denoise + 1,
                "n_action_steps": self.n_action_steps,
                "max_action_dim": self.max_action_dim,
                "x_traj_space": "model-normalized max_action_dim space",
                "durable_rows": 0,
            }
            if capture_run_id is not None:
                attrs["capture_run_id"] = capture_run_id
            self.root.attrs.update(attrs)

        self.arrays: dict[str, Any] = {}
        if resume:
            expected_names = {name for name, *_rest in specs}
            actual_names = set(self.root.array_keys())
            if actual_names != expected_names:
                raise ValueError(
                    "flow store arrays differ: missing=%s extra=%s"
                    % (sorted(expected_names - actual_names), sorted(actual_names - expected_names))
                )
            for name, tail, _chunks, dtype in specs:
                array = self.root[name]
                if tuple(array.shape[1:]) != tail or np.dtype(array.dtype) != np.dtype(dtype):
                    raise ValueError("flow store array %s has incompatible shape or dtype" % name)
                self.arrays[name] = array
        else:
            codec = [ZstdCodec(level=zstd_level)]
            for name, tail, chunks, dtype in specs:
                self.arrays[name] = self.root.create_array(
                    name=name,
                    shape=(0, *tail),
                    chunks=chunks,
                    dtype=dtype,
                    compressors=codec,
                )
        self._pending: dict[str, list[np.ndarray]] = {name: [] for name in self.arrays}
        self._n_pending = 0
        if resume:
            self._recover_durable_prefix()

    @property
    def durable_rows(self) -> int:
        return int(self.root.attrs.get("durable_rows", 0))

    @property
    def rows(self) -> int:
        return self.durable_rows + self._n_pending

    def _recover_durable_prefix(self) -> None:
        lengths = [int(array.shape[0]) for array in self.arrays.values()]
        declared = int(self.root.attrs.get("durable_rows", min(lengths)))
        target = min([declared, *lengths])
        if target < 0:
            raise ValueError("flow store has a negative durable row count")
        for array in self.arrays.values():
            if int(array.shape[0]) != target:
                array.resize((target, *array.shape[1:]))
        self.root.attrs["durable_rows"] = target

    def truncate(self, rows: int) -> None:
        if self._n_pending:
            raise RuntimeError("cannot truncate flow store with pending rows")
        rows = int(rows)
        if rows < 0 or rows > self.durable_rows:
            raise ValueError("flow truncate target is outside its durable prefix")
        for array in self.arrays.values():
            array.resize((rows, *array.shape[1:]))
        self.root.attrs["durable_rows"] = rows

    def append(self, identity: CaptureIdentity, x_traj: np.ndarray) -> None:
        trajectory = np.asarray(x_traj, dtype=np.float32)
        expected = (1, self.n_denoise + 1, self.n_action_steps, self.max_action_dim)
        if trajectory.shape != expected or not np.all(np.isfinite(trajectory)):
            raise ValueError("x_traj must be one finite row with shape %s" % (expected,))
        if int(identity.capture_row) != self.rows:
            raise ValueError("capture identity row is not the flow store's next row")
        values = {
            "x_traj": trajectory,
            "capture_row": np.asarray([identity.capture_row], dtype=np.int64),
            "query_id": np.asarray([identity.query_id], dtype=np.int64),
            "snapshot_index": np.asarray([identity.snapshot_index], dtype=np.int32),
            "candidate_id": np.asarray([identity.candidate_id], dtype=np.int32),
            "local_call_ordinal": np.asarray([identity.local_call_ordinal], dtype=np.int64),
            "observation_sha256": _digest_array(identity.observation_sha256).reshape(1, 32),
            "flow_noise_sha256": _digest_array(identity.flow_noise_sha256).reshape(1, 32),
            "actions_sha256": _digest_array(identity.actions_sha256).reshape(1, 32),
        }
        for name, value in values.items():
            self._pending[name].append(value)
        self._n_pending += 1
        if self.auto_flush and self._n_pending >= self.chunk_queries:
            self.flush()

    def rollback_pending(self, rows: int) -> None:
        """Remove the most-recent buffered rows without touching durable rows."""

        remaining = int(rows)
        if remaining < 0 or remaining > self._n_pending:
            raise ValueError("flow rollback exceeds pending rows")
        while remaining:
            for chunks in self._pending.values():
                chunks.pop()
            self._n_pending -= 1
            remaining -= 1

    def flush(self) -> None:
        if not self._n_pending:
            return
        start = self.durable_rows
        if any(int(array.shape[0]) != start for array in self.arrays.values()):
            raise RuntimeError("flow store arrays are not at one durable boundary")
        payload = {
            name: np.concatenate(chunks, axis=0) for name, chunks in self._pending.items()
        }
        try:
            for name, value in payload.items():
                self.arrays[name].append(value, axis=0)
            self.root.attrs["durable_rows"] = start + self._n_pending
        except Exception:
            for array in self.arrays.values():
                if int(array.shape[0]) != start:
                    array.resize((start, *array.shape[1:]))
            raise
        for chunks in self._pending.values():
            chunks.clear()
        self._n_pending = 0

    def close(self) -> None:
        self.flush()


class ZarrFlowTrajectoryReader:
    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") != FORMAT:
            raise ValueError("not a %s store: %s" % (FORMAT, path))

    def __len__(self) -> int:
        return int(self.root["x_traj"].shape[0])

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())
