"""Bounded, atomic NumPy shards and the existing safe snapshot codec."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_directory(directory):
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def atomic_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


class ShardWriter:
    def __init__(self, directory, block_size=8, pending_blocks=2):
        if block_size < 1 or pending_blocks < 1:
            raise ValueError("Shard and queue sizes must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.block_size = block_size
        self.pending_blocks = pending_blocks
        self.buffer = []
        self.pending = []
        self.blocks = []
        self.submitted = 0
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.closed = False
        self.manifest = None

    def _write(self, index, records):
        keys = set(records[0])
        if any(set(row) != keys for row in records):
            raise ValueError("Shard record fields differ")
        arrays = {key: np.stack([row[key] for row in records]) for key in sorted(keys)}
        if any(value.dtype.hasobject for value in arrays.values()):
            raise ValueError("Object arrays are forbidden in collection shards")
        path = self.directory / ("block_%05d.npz" % index)
        atomic_npz(path, arrays)
        block = dict(path=path.name, sha256=digest(path), rows=len(records), bytes=path.stat().st_size,
                     first_query=int(records[0]["query"]), last_query=int(records[-1]["query"]))
        atomic_json(path.with_suffix(".json"), block)
        return block

    def append(self, record):
        if self.closed:
            raise RuntimeError("Cannot append to a closed shard writer")
        self.buffer.append(record)
        if len(self.buffer) >= self.block_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        if len(self.pending) >= self.pending_blocks:
            self.blocks.append(self.pending.pop(0).result())
        rows, self.buffer = self.buffer, []
        self.pending.append(self.pool.submit(self._write, self.submitted, rows))
        self.submitted += 1

    def close(self):
        if self.closed:
            return self.manifest
        try:
            self.flush()
            self.blocks.extend(future.result() for future in self.pending)
            self.pending.clear()
            manifest = dict(blocks=self.blocks, queries=sum(b["rows"] for b in self.blocks),
                            bytes=sum(b["bytes"] for b in self.blocks), queries_per_block=self.block_size,
                            pending_blocks_limit=self.pending_blocks)
            atomic_json(self.directory / "manifest.json", manifest)
            self.manifest = manifest
            return manifest
        finally:
            self.closed = True
            self.pool.shutdown(wait=True)


def records(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected = None
    for block in manifest["blocks"]:
        path = directory / block["path"]
        if digest(path) != block["sha256"]:
            raise ValueError("Shard checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            columns = {key: archive[key] for key in archive.files}
        if any(len(column) != block["rows"] for column in columns.values()):
            raise ValueError("Shard row counts differ")
        for index in range(block["rows"]):
            row = {key: column[index] for key, column in columns.items()}
            query = int(row["query"])
            if expected is not None and query != expected:
                raise ValueError("Query IDs are not contiguous")
            expected = query + 1
            yield row


def save_snapshot(directory, snapshot):
    from branch_snapshot import encode_full_state

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    metadata, arrays = encode_full_state(snapshot)
    atomic_npz(directory / "state.npz", arrays)
    atomic_json(directory / "state.json", metadata)
    manifest = {name: digest(directory / name) for name in ("state.json", "state.npz")}
    atomic_json(directory / "manifest.json", manifest)


def load_snapshot(directory):
    from branch_snapshot import decode_full_state

    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        if digest(directory / name) != expected:
            raise ValueError("Snapshot checksum mismatch")
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return decode_full_state(json.loads((directory / "state.json").read_text()), arrays)
