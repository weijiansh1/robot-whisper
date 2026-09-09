"""Zarr storage for query-aligned flow states and true HB-MoE amplitudes."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec


FORMAT = "himoe_hb_activation_flow_v3"
SUPPORTED_FORMATS = {"himoe_hb_activation_flow_v2", FORMAT}


class ZarrActivationFlowWriter:
    """Append one jointly captured policy query at a time.

    The append axis is the batch/query item.  ``x_traj`` includes the initial
    noise and final action chunk, hence D+1 states for D denoising forwards.
    """

    def __init__(
        self,
        path: str,
        hb_layers: list[int],
        n_denoise: int,
        n_action_steps: int,
        max_action_dim: int,
        action_std: list[float] | np.ndarray,
        top_k: int = 4,
        chunk_queries: int = 64,
        zstd_level: int = 3,
        overwrite: bool = True,
        probe_hb_layer: int = 5,
        probe_hidden_size: int = 1024,
        n_routed_experts: int = 32,
    ) -> None:
        self.path = path
        self.hb_layers = [int(layer) for layer in hb_layers]
        self.n_denoise = int(n_denoise)
        self.n_action_steps = int(n_action_steps)
        self.max_action_dim = int(max_action_dim)
        self.top_k = int(top_k)
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        self.probe_hb_layer = int(probe_hb_layer)
        self.probe_denoise = 0
        self.probe_hidden_size = int(probe_hidden_size)
        self.n_routed_experts = int(n_routed_experts)
        if self.probe_hb_layer not in self.hb_layers:
            raise ValueError("probe_hb_layer must name one stored HB layer")
        if self.n_denoise <= self.probe_denoise:
            raise ValueError("the activation flow must include denoise round zero")
        if self.probe_hidden_size <= 0 or self.n_routed_experts <= 0:
            raise ValueError("probe hidden size and expert count must be positive")
        self.action_std = np.asarray(action_std, dtype=np.float64)
        if self.action_std.shape != (7,):
            raise ValueError("action_std must contain the seven live action dimensions")
        if not np.all(np.isfinite(self.action_std)) or np.any(self.action_std <= 0.0):
            raise ValueError("action_std must be finite and strictly positive")
        self.chunk_queries = int(chunk_queries)
        self.root = zarr.create_group(store=path, overwrite=overwrite)
        self.root.attrs.update(
            {
                "format": FORMAT,
                "hb_layers": self.hb_layers,
                "n_hb_layers": len(self.hb_layers),
                "n_denoise": self.n_denoise,
                "n_flow_states": self.n_denoise + 1,
                "n_action_steps": self.n_action_steps,
                "max_action_dim": self.max_action_dim,
                "top_k": self.top_k,
                "probe_hb_layer": self.probe_hb_layer,
                "probe_denoise": self.probe_denoise,
                "probe_hidden_size": self.probe_hidden_size,
                "n_routed_experts": self.n_routed_experts,
                "runtime_vector_probe": "pre-gate actual dispatch, fp16 storage",
                "runtime_vector_probe_reconstruction": (
                    "fp32 sum_k(combine_weight_k * raw_k) vs actual routed output"
                ),
                "query_identity": (
                    "query_id and candidate_id are explicit int64 arrays; -1 means absent"
                ),
                "normalization_action_std": self.action_std.tolist(),
                "suffix_state_token_stored": False,
                "activation_definition": (
                    "RMS_hidden(sum_e actual_normalized_topk_weight_e * expert_e(h)); "
                    "plus per-selected-expert RMS_hidden(E_e(h)) before gate multiplication; "
                    "plus one full-vector probe at the declared HB layer/denoise; "
                    "action suffix tokens only"
                ),
                "x_traj_space": "model-normalized max_action_dim space",
            }
        )

        c = self.chunk_queries
        layers = len(self.hb_layers)
        denoise = self.n_denoise
        actions = self.n_action_steps
        action_dim = self.max_action_dim
        top_k = self.top_k
        hidden = self.probe_hidden_size
        experts = self.n_routed_experts
        specs = [
            (
                "x_traj",
                (denoise + 1, actions, action_dim),
                (max(1, c // 4), denoise + 1, actions, action_dim),
                "float32",
            ),
            ("hb_routed_rms", (layers, denoise, actions), (c, layers, denoise, actions), "float32"),
            ("hb_shared_rms", (layers, denoise, actions), (c, layers, denoise, actions), "float32"),
            ("hb_total_mlp_rms", (layers, denoise, actions), (c, layers, denoise, actions), "float32"),
            (
                "hb_selected_expert_id",
                (layers, denoise, actions, top_k),
                (c, layers, denoise, actions, top_k),
                "uint8",
            ),
            (
                "hb_selected_expert_weight",
                (layers, denoise, actions, top_k),
                (c, layers, denoise, actions, top_k),
                "float32",
            ),
            (
                "hb_selected_expert_raw_rms",
                (layers, denoise, actions, top_k),
                (c, layers, denoise, actions, top_k),
                "float32",
            ),
            (
                "hb_topk_weight_sum_error",
                (layers, denoise, actions),
                (c, layers, denoise, actions),
                "float32",
            ),
            (
                "hb_probe_input_hidden",
                (actions, hidden),
                (max(1, c // 4), actions, hidden),
                "float16",
            ),
            (
                "hb_probe_shared_output",
                (actions, hidden),
                (max(1, c // 4), actions, hidden),
                "float16",
            ),
            (
                "hb_probe_selected_expert_raw",
                (actions, top_k, hidden),
                (max(1, c // 16), actions, top_k, hidden),
                "float16",
            ),
            (
                "hb_probe_router_probs",
                (actions, experts),
                (c, actions, experts),
                "float16",
            ),
            (
                "hb_probe_routed_reconstruction_max_abs_error",
                (actions,),
                (c, actions),
                "float32",
            ),
            ("query_id", (), (c,), "int64"),
            ("candidate_id", (), (c,), "int64"),
            ("episode_id", (), (c,), "int32"),
            ("control_step", (), (c,), "int32"),
        ]
        self.arrays: dict[str, Any] = {}
        for name, tail, chunks, dtype in specs:
            self.arrays[name] = self.root.create_array(
                name=name,
                shape=(0, *tail),
                chunks=chunks,
                dtype=dtype,
                compressors=[ZstdCodec(level=zstd_level)],
            )
        self._pending: dict[str, list[np.ndarray]] = {key: [] for key in self.arrays}
        self._n_pending = 0

    def append(self, record, x_traj: np.ndarray) -> None:
        trajectory = np.asarray(x_traj, dtype=np.float32)
        expected_traj_tail = (
            self.n_denoise + 1,
            self.n_action_steps,
            self.max_action_dim,
        )
        if trajectory.ndim != 4 or tuple(trajectory.shape[1:]) != expected_traj_tail:
            raise ValueError(
                "x_traj must be [batch,%d,%d,%d], got %s"
                % (*expected_traj_tail, tuple(trajectory.shape))
            )
        expected_activation_tail = (
            len(self.hb_layers),
            self.n_denoise,
            self.n_action_steps,
        )
        batch = trajectory.shape[0]
        if list(np.asarray(record.hb_layers, dtype=np.int16)) != self.hb_layers:
            raise ValueError("record HB layers do not match writer metadata")
        if int(record.n_denoise) != self.n_denoise:
            raise ValueError("record denoise count does not match writer metadata")
        if int(record.probe_hb_layer) != self.probe_hb_layer:
            raise ValueError("record probe HB layer does not match writer metadata")
        if int(record.probe_denoise) != self.probe_denoise:
            raise ValueError("record probe denoise does not match writer metadata")

        selected_tail = (*expected_activation_tail, self.top_k)
        field_specs = {
            "hb_routed_rms": (
                record.hb_routed_rms,
                np.float32,
                expected_activation_tail,
            ),
            "hb_shared_rms": (
                record.hb_shared_rms,
                np.float32,
                expected_activation_tail,
            ),
            "hb_total_mlp_rms": (
                record.hb_total_mlp_rms,
                np.float32,
                expected_activation_tail,
            ),
            "hb_topk_weight_sum_error": (
                record.hb_topk_weight_sum_error,
                np.float32,
                expected_activation_tail,
            ),
            "hb_selected_expert_id": (
                record.hb_selected_expert_id,
                np.uint8,
                selected_tail,
            ),
            "hb_selected_expert_weight": (
                record.hb_selected_expert_weight,
                np.float32,
                selected_tail,
            ),
            "hb_selected_expert_raw_rms": (
                record.hb_selected_expert_raw_rms,
                np.float32,
                selected_tail,
            ),
            "hb_probe_input_hidden": (
                record.hb_probe_input_hidden,
                np.float16,
                (self.n_action_steps, self.probe_hidden_size),
            ),
            "hb_probe_shared_output": (
                record.hb_probe_shared_output,
                np.float16,
                (self.n_action_steps, self.probe_hidden_size),
            ),
            "hb_probe_selected_expert_raw": (
                record.hb_probe_selected_expert_raw,
                np.float16,
                (
                    self.n_action_steps,
                    self.top_k,
                    self.probe_hidden_size,
                ),
            ),
            "hb_probe_router_probs": (
                record.hb_probe_router_probs,
                np.float16,
                (self.n_action_steps, self.n_routed_experts),
            ),
            "hb_probe_routed_reconstruction_max_abs_error": (
                record.hb_probe_routed_reconstruction_max_abs_error,
                np.float32,
                (self.n_action_steps,),
            ),
        }
        validated = {"x_traj": trajectory}
        for name, (value, dtype, tail) in field_specs.items():
            array = np.asarray(value, dtype=dtype)
            if array.shape != (batch, *tail):
                raise ValueError(
                    "%s must have shape %s, got %s"
                    % (name, (batch, *tail), array.shape)
                )
            validated[name] = array

        validated["episode_id"] = np.full(
            batch, int(record.episode_id), dtype=np.int32
        )
        validated["control_step"] = np.full(
            batch, int(record.control_step), dtype=np.int32
        )
        validated["query_id"] = np.full(
            batch, int(record.query_id), dtype=np.int64
        )
        validated["candidate_id"] = np.full(
            batch, int(record.candidate_id), dtype=np.int64
        )
        if set(validated) != set(self._pending):
            missing = sorted(set(self._pending) - set(validated))
            extra = sorted(set(validated) - set(self._pending))
            raise RuntimeError(
                "activation-flow append schema mismatch: missing=%s extra=%s"
                % (missing, extra)
            )
        for name, array in validated.items():
            self._pending[name].append(array)
        self._n_pending += batch
        if self._n_pending >= self.chunk_queries:
            self.flush()

    def flush(self) -> None:
        if self._n_pending == 0:
            return
        for name, chunks in self._pending.items():
            self.arrays[name].append(np.concatenate(chunks, axis=0), axis=0)
            chunks.clear()
        self._n_pending = 0

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ZarrActivationFlowWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZarrActivationFlowReader:
    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") not in SUPPORTED_FORMATS:
            raise ValueError("not a supported activation-flow store: %s" % path)

    def __len__(self) -> int:
        return self.root["x_traj"].shape[0]

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())
