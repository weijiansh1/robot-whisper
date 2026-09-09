"""Zarr v3 + Zstd store for the tensors HiMoE-VLA's routers read.

Kept in ``hidden.zarr``, next to but separate from ``routes.zarr``::

    hidden.zarr/
      hb_hidden   float16  [N, 8, 10, 11, 1024]
      as_hidden   float16  [N, 4, 10, 11,   24]
      episode_id  int32    [N]
      control_step int32   [N]

Why a second store rather than more arrays in ``routes.zarr``: these are ~44x
the bytes of the routing itself (1.82 MB vs 41 KB per control step, measured).
Routing-only analysis is the common case and should not have to open, list or
consolidate metadata for a store two orders of magnitude larger.  ``episode_id``
and ``control_step`` are duplicated so this store can be sliced on its own; they
cost 8 bytes per row against 1.8 MB of payload.

The two gate families do not read the same tensor.  HB routers see the 1024-dim
hidden state; AS routers see the 24-dim data_mask (modeling_moe.py routes them on
dataset identity, not on content), so ``as_hidden`` is 21 KB per control step
against 1.80 MB for ``hb_hidden`` -- 1% of the cost.  It is kept because it is
what makes the AS collapse assertion in the recorder checkable after the fact
rather than only at capture time.

Compression note: unlike the routing arrays, activations do not compress well.
Top-k probabilities cluster near 1/k and repeat at byte granularity, which is
why plain Zstd beats bitshuffle there (see himoe_route_store._codecs).  Hidden
states are broadband, so expect ~1.3-1.5x rather than ~2x, and size the disk
budget from the uncompressed figure.

The router input is the one thing here that cannot be recovered after the fact:
every probability, id and entropy in ``routes.zarr`` is a function of it, but not
the other way round.  That is the whole reason to pay for it.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

FORMAT = "himoe_router_hidden_v1"


class ZarrHiddenWriter:
    def __init__(
        self,
        path: str,
        n_hb_layers: int = 8,
        n_as_layers: int = 4,
        n_denoise: int = 10,
        n_suffix: int = 11,
        hb_hidden_dim: int = 1024,
        as_hidden_dim: int = 24,
        # One row is 1.82 MB, so a 64-step chunk would be a 117 MB write.  Eight
        # keeps chunks near 15 MB: still far above the point where Zstd stops
        # caring, and small enough that a killed server loses little.
        chunk_steps: int = 8,
        zstd_level: int = 3,
        overwrite: bool = True,
        resume: bool = False,
        capture_run_id: str | None = None,
        auto_flush: bool = True,
    ) -> None:
        self.path = path
        self.chunk_steps = chunk_steps
        self.auto_flush = bool(auto_flush)
        c, D, S = chunk_steps, n_denoise, n_suffix
        Hh, Ha = hb_hidden_dim, as_hidden_dim
        specs = [
            ("hb_hidden", (n_hb_layers, D, S, Hh), (c, n_hb_layers, D, S, Hh), "float16"),
            ("as_hidden", (n_as_layers, D, S, Ha), (c, n_as_layers, D, S, Ha), "float16"),
            ("episode_id", (), (max(c, 64),), "int32"),
            ("control_step", (), (max(c, 64),), "int32"),
        ]

        store_path = pathlib.Path(path)
        if resume:
            if overwrite:
                raise ValueError("resume and overwrite are mutually exclusive")
            if not store_path.exists():
                raise FileNotFoundError("hidden store does not exist for resume: %s" % path)
            self.root = zarr.open_group(path, mode="a")
            expected_attrs = {
                "format": FORMAT,
                "n_hb_layers": n_hb_layers,
                "n_as_layers": n_as_layers,
                "n_denoise": n_denoise,
                "n_suffix": n_suffix,
                "hb_hidden_dim": hb_hidden_dim,
                "as_hidden_dim": as_hidden_dim,
            }
            for name, expected in expected_attrs.items():
                if self.root.attrs.get(name) != expected:
                    raise ValueError(
                        "hidden store attribute %s is %r, expected %r"
                        % (name, self.root.attrs.get(name), expected)
                    )
            existing_run_id = self.root.attrs.get("capture_run_id")
            if capture_run_id is not None and existing_run_id != capture_run_id:
                raise ValueError("hidden store capture_run_id mismatch")
        else:
            if not overwrite and store_path.exists():
                raise FileExistsError("hidden store already exists: %s" % path)
            self.root = zarr.create_group(store=path, overwrite=overwrite)
            attrs = {
                "format": FORMAT,
                "n_hb_layers": n_hb_layers,
                "n_as_layers": n_as_layers,
                "n_denoise": n_denoise,
                "n_suffix": n_suffix,
                "hb_hidden_dim": hb_hidden_dim,
                "as_hidden_dim": as_hidden_dim,
                "note": "router inputs; routes.zarr holds the decisions derived from them",
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
                    "hidden store arrays differ: missing=%s extra=%s"
                    % (sorted(expected_names - actual_names), sorted(actual_names - expected_names))
                )
            for name, tail, _chunks, dtype in specs:
                array = self.root[name]
                if tuple(array.shape[1:]) != tail or np.dtype(array.dtype) != np.dtype(dtype):
                    raise ValueError("hidden store array %s has incompatible shape or dtype" % name)
                self.arrays[name] = array
        else:
            for name, tail, chunks, dtype in specs:
                self.arrays[name] = self.root.create_array(
                    name=name,
                    shape=(0, *tail),
                    chunks=chunks,
                    dtype=dtype,
                    compressors=[ZstdCodec(level=zstd_level)],
                )
        self._pending: dict[str, list[np.ndarray]] = {k: [] for k in self.arrays}
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
            raise ValueError("hidden store has a negative durable row count")
        for array in self.arrays.values():
            if int(array.shape[0]) != target:
                array.resize((target, *array.shape[1:]))
        self.root.attrs["durable_rows"] = target

    def truncate(self, rows: int) -> None:
        if self._n_pending:
            raise RuntimeError("cannot truncate hidden store with pending rows")
        rows = int(rows)
        if rows < 0 or rows > self.durable_rows:
            raise ValueError("hidden truncate target is outside its durable prefix")
        for array in self.arrays.values():
            array.resize((rows, *array.shape[1:]))
        self.root.attrs["durable_rows"] = rows

    def append(self, rec) -> None:
        if rec.hb_hidden is None or rec.as_hidden is None:
            raise ValueError(
                "hidden writer got a record without hidden states; the recorder "
                "was built with store_hidden=False"
            )
        b = rec.hb_hidden.shape[0]
        self._pending["hb_hidden"].append(rec.hb_hidden)
        self._pending["as_hidden"].append(rec.as_hidden)
        self._pending["episode_id"].append(np.full(b, rec.episode_id, np.int32))
        self._pending["control_step"].append(np.full(b, rec.control_step, np.int32))
        self._n_pending += b
        if self.auto_flush and self._n_pending >= self.chunk_steps:
            self.flush()

    def rollback_pending(self, rows: int) -> None:
        """Remove whole, most-recent buffered records without touching durable rows."""

        remaining = int(rows)
        if remaining < 0 or remaining > self._n_pending:
            raise ValueError("hidden rollback exceeds pending rows")
        while remaining:
            batch = int(self._pending["episode_id"][-1].shape[0])
            if batch > remaining:
                raise ValueError("hidden rollback would split one buffered record")
            for chunks in self._pending.values():
                chunks.pop()
            self._n_pending -= batch
            remaining -= batch

    def flush(self) -> None:
        if self._n_pending == 0:
            return
        start = self.durable_rows
        if any(int(array.shape[0]) != start for array in self.arrays.values()):
            raise RuntimeError("hidden store arrays are not at one durable boundary")
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

    def __enter__(self) -> "ZarrHiddenWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZarrHiddenReader:
    """Read side; every access is a lazy Zarr slice."""

    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} store: {path}")

    def __len__(self) -> int:
        return self.root["hb_hidden"].shape[0]

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())

    def recompute_hb_probs(self, gate_weight: np.ndarray, sel: int) -> np.ndarray:
        """Softmax(W @ h) for one control step -> [L, D, S, E].

        The point of storing hidden states: with a router weight matrix -- the
        released one or a perturbed one -- this reproduces (or counterfactually
        replaces) ``routes.zarr/hb_router_probs``.  Compare against the stored
        array to confirm a capture is self-consistent.
        """
        h = np.asarray(self.root["hb_hidden"][sel], dtype=np.float32)
        logits = h @ np.asarray(gate_weight, dtype=np.float32).T
        logits -= logits.max(-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(-1, keepdims=True)
