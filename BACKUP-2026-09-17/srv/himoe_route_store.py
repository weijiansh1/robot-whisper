"""Zarr v3 + Zstd store for HiMoE-VLA routing traces.

Layout (one group per run)::

    routes.zarr/
      hb_expert_ids      uint8    [N, 8, 10, 51, 4]   chunk (C, 8, 10, 51, 4)
      hb_selected_prob   float16  [N, 8, 10, 51, 4]
      hb_entropy         float16  [N, 8, 10, 51]
      hb_router_probs    float16  [N, 8, 10, 51, 32]  (optional)
      as_expert_ids      uint8    [N, 4]
      as_probs           float16  [N, 4, 3]
      episode_id         int32    [N]
      control_step       int32    [N]

N is the append axis: one row per (batch item, control step).  Everything else
is a fixed-size dense axis, which is what makes Zarr the right container --
the same trace as a Parquet table would be N * 16320 rows.

combine_weight is intentionally absent; recover it with
``himoe_router_recorder.combine_weight_from_raw``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

FORMAT = "himoe_router_trace_v2"


def _codecs(level: int, shuffle: str | None, itemsize: int):
    """Plain Zstd by default; Blosc(zstd)+shuffle only if explicitly asked for.

    Measured on 3,943 real LIBERO control steps (bench_real.py), bitshuffle is
    a large *loss* on the fp16 weight arrays -- 2.95x plain zstd-5 vs 1.85x with
    bitshuffle, i.e. 60% bigger.  Real top-k combine weights are normalised and
    cluster tightly around 1/k, so they repeat at byte granularity and zstd's
    matcher exploits that; bitshuffle destroys exactly those repeats.  An
    earlier synthetic benchmark suggested the opposite and was wrong.
    """
    if shuffle is None or itemsize == 1:
        return [ZstdCodec(level=level)]
    return [
        BloscCodec(cname="zstd", clevel=level, shuffle=shuffle, typesize=itemsize)
    ]


class ZarrRouteWriter:
    def __init__(
        self,
        path: str,
        n_hb_layers: int = 8,
        n_as_layers: int = 4,
        n_denoise: int = 10,
        # 51 = 1 state token + 50 actions (upstream default config, e.g. CALVIN).
        # The released LIBERO checkpoints use n_action_steps=10 -> pass 11.
        n_suffix: int = 51,
        top_k: int = 4,
        n_hb_experts: int = 32,
        n_as_experts: int = 3,
        store_full_probs: bool = False,
        chunk_steps: int = 64,
        zstd_level: int = 3,
        shuffle: str | None = None,  # see _codecs: bitshuffle loses on real traces
        overwrite: bool = True,
        resume: bool = False,
        capture_run_id: str | None = None,
        auto_flush: bool = True,
    ) -> None:
        self.path = path
        self.chunk_steps = chunk_steps
        self.store_full_probs = store_full_probs
        self.auto_flush = bool(auto_flush)
        c = chunk_steps
        L, D, S, K, E = n_hb_layers, n_denoise, n_suffix, top_k, n_hb_experts
        specs: list[tuple[str, tuple[int, ...], tuple[int, ...], str, str | None]] = [
            ("hb_expert_ids", (L, D, S, K), (c, L, D, S, K), "uint8", None),
            ("hb_selected_prob", (L, D, S, K), (c, L, D, S, K), "float16", shuffle),
            ("hb_entropy", (L, D, S), (c, L, D, S), "float16", shuffle),
            ("as_expert_ids", (n_as_layers,), (c, n_as_layers), "uint8", None),
            ("as_probs", (n_as_layers, n_as_experts), (c, n_as_layers, n_as_experts), "float16", shuffle),
            ("episode_id", (), (c,), "int32", shuffle),
            ("control_step", (), (c,), "int32", shuffle),
        ]
        if store_full_probs:
            specs.append(
                ("hb_router_probs", (L, D, S, E), (max(1, c // 8), L, D, S, E), "float16", shuffle)
            )

        store_path = pathlib.Path(path)
        if resume:
            if overwrite:
                raise ValueError("resume and overwrite are mutually exclusive")
            if not store_path.exists():
                raise FileNotFoundError("route store does not exist for resume: %s" % path)
            self.root = zarr.open_group(path, mode="a")
            expected_attrs = {
                "format": FORMAT,
                "n_hb_layers": n_hb_layers,
                "n_as_layers": n_as_layers,
                "n_denoise": n_denoise,
                "n_suffix": n_suffix,
                "top_k": top_k,
                "n_hb_experts": n_hb_experts,
                "n_as_experts": n_as_experts,
            }
            for name, expected in expected_attrs.items():
                if self.root.attrs.get(name) != expected:
                    raise ValueError(
                        "route store attribute %s is %r, expected %r"
                        % (name, self.root.attrs.get(name), expected)
                    )
            existing_run_id = self.root.attrs.get("capture_run_id")
            if capture_run_id is not None and existing_run_id != capture_run_id:
                raise ValueError("route store capture_run_id mismatch")
        else:
            if not overwrite and store_path.exists():
                raise FileExistsError("route store already exists: %s" % path)
            self.root = zarr.create_group(store=path, overwrite=overwrite)
            attrs = {
                "format": FORMAT,
                "n_hb_layers": n_hb_layers,
                "n_as_layers": n_as_layers,
                "n_denoise": n_denoise,
                "n_suffix": n_suffix,
                "top_k": top_k,
                "n_hb_experts": n_hb_experts,
                "n_as_experts": n_as_experts,
                "note": "combine_weight is derivable from *_prob; see recorder",
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
                    "route store arrays differ: missing=%s extra=%s"
                    % (sorted(expected_names - actual_names), sorted(actual_names - expected_names))
                )
            for name, tail, _chunks, dtype, _shuf in specs:
                array = self.root[name]
                if tuple(array.shape[1:]) != tail or np.dtype(array.dtype) != np.dtype(dtype):
                    raise ValueError("route store array %s has incompatible shape or dtype" % name)
                self.arrays[name] = array
        else:
            for name, tail, chunks, dtype, shuf in specs:
                self.arrays[name] = self.root.create_array(
                    name=name,
                    shape=(0, *tail),
                    chunks=chunks,
                    dtype=dtype,
                    compressors=_codecs(zstd_level, shuf, np.dtype(dtype).itemsize),
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
            raise ValueError("route store has a negative durable row count")
        for array in self.arrays.values():
            if int(array.shape[0]) != target:
                array.resize((target, *array.shape[1:]))
        self.root.attrs["durable_rows"] = target

    def truncate(self, rows: int) -> None:
        if self._n_pending:
            raise RuntimeError("cannot truncate route store with pending rows")
        rows = int(rows)
        if rows < 0 or rows > self.durable_rows:
            raise ValueError("route truncate target is outside its durable prefix")
        for array in self.arrays.values():
            array.resize((rows, *array.shape[1:]))
        self.root.attrs["durable_rows"] = rows

    def append(self, rec) -> None:
        """Buffer one ControlStepRecord; flush once ``chunk_steps`` accumulate."""
        if not rec.as_collapsed:
            raise ValueError(
                "AS routing did not collapse across denoising steps/tokens; "
                "the dense schema assumes it does -- inspect this control step"
            )
        if self.store_full_probs and rec.hb_router_probs is None:
            raise ValueError("writer wants full probs but the record has none")
        b = rec.hb_expert_ids.shape[0]
        self._pending["hb_expert_ids"].append(rec.hb_expert_ids)
        self._pending["hb_selected_prob"].append(rec.hb_selected_prob)
        self._pending["hb_entropy"].append(rec.hb_entropy)
        self._pending["as_expert_ids"].append(rec.as_expert_ids)
        self._pending["as_probs"].append(rec.as_probs)
        self._pending["episode_id"].append(np.full(b, rec.episode_id, np.int32))
        self._pending["control_step"].append(np.full(b, rec.control_step, np.int32))
        if self.store_full_probs:
            self._pending["hb_router_probs"].append(rec.hb_router_probs)
        self._n_pending += b
        if self.auto_flush and self._n_pending >= self.chunk_steps:
            self.flush()

    def rollback_pending(self, rows: int) -> None:
        """Remove whole, most-recent buffered records without touching durable rows."""

        remaining = int(rows)
        if remaining < 0 or remaining > self._n_pending:
            raise ValueError("route rollback exceeds pending rows")
        while remaining:
            batch = int(self._pending["episode_id"][-1].shape[0])
            if batch > remaining:
                raise ValueError("route rollback would split one buffered record")
            for chunks in self._pending.values():
                chunks.pop()
            self._n_pending -= batch
            remaining -= batch

    def flush(self) -> None:
        if self._n_pending == 0:
            return
        start = self.durable_rows
        if any(int(array.shape[0]) != start for array in self.arrays.values()):
            raise RuntimeError("route store arrays are not at one durable boundary")
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

    def __enter__(self) -> "ZarrRouteWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZarrRouteReader:
    """Thin read side; every access is a lazy Zarr slice."""

    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") != FORMAT:
            raise ValueError(f"not a {FORMAT} store: {path}")

    def __len__(self) -> int:
        return self.root["hb_expert_ids"].shape[0]

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())

    def combine_weight(self, sel: slice | int = slice(None)) -> np.ndarray:
        raw = np.asarray(self.root["hb_selected_prob"][sel], dtype=np.float32)
        return raw / (raw.sum(-1, keepdims=True) + 1e-20)

    def expert_load(self, sel: slice | int = slice(None)) -> np.ndarray:
        """Per-HB-layer expert counts -> [L, n_experts]."""
        ids = np.asarray(self.root["hb_expert_ids"][sel])
        n = self.meta["n_hb_experts"]
        return np.stack(
            [
                np.bincount(ids[:, layer].ravel(), minlength=n)
                for layer in range(ids.shape[1])
            ]
        )
