"""Independent v5 storage for runtime-exact HB5 raw-pruning comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from himoe_raw_pruning_intervention import (
    CODE_TO_POLICY,
    POLICY_TO_CODE,
    PREREGISTERED_POLICIES,
    RAW_PRUNING_POLICIES,
)


FORMAT = "himoe_hb5_d0_raw_pruning_v5"
STORE_NAME = "hb5_raw_pruning_v5.zarr"


def _digest_array(value: bytes | bytearray | memoryview | str) -> np.ndarray:
    raw = bytes.fromhex(value) if isinstance(value, str) else bytes(value)
    if len(raw) != 32:
        raise ValueError("SHA-256 digests must contain 32 bytes")
    return np.frombuffer(raw, dtype=np.uint8).copy()


def _require(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: np.dtype | str,
) -> np.ndarray:
    if value is None:
        raise ValueError("%s is absent from the raw-pruning record" % name)
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("%s must have shape %s, got %s" % (name, shape, array.shape))
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite values" % name)
    return array.astype(dtype, copy=False)


class ZarrRawPruningWriter:
    def __init__(
        self,
        path: str,
        *,
        n_denoise: int,
        n_action_steps: int,
        max_action_dim: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        n_routed_experts: int,
        action_std: list[float] | np.ndarray,
        expected_policies: tuple[str, ...] = PREREGISTERED_POLICIES,
        chunk_queries: int = 16,
        zstd_level: int = 3,
        overwrite: bool = True,
    ) -> None:
        self.n_denoise = int(n_denoise)
        self.n_action_steps = int(n_action_steps)
        self.max_action_dim = int(max_action_dim)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.top_k = int(top_k)
        self.n_routed_experts = int(n_routed_experts)
        self.chunk_queries = int(chunk_queries)
        self.expected_policies = tuple(expected_policies)
        if (
            not self.expected_policies
            or len(set(self.expected_policies)) != len(self.expected_policies)
            or "baseline" not in self.expected_policies
            or not set(self.expected_policies).issubset(RAW_PRUNING_POLICIES)
        ):
            raise ValueError("expected policies must be unique, valid, and include baseline")
        action_std_array = np.asarray(action_std, dtype=np.float64)
        if action_std_array.shape != (7,) or np.any(action_std_array <= 0):
            raise ValueError("action_std must contain seven positive values")
        dimensions = (
            self.n_denoise,
            self.n_action_steps,
            self.max_action_dim,
            self.hidden_size,
            self.intermediate_size,
            self.top_k,
            self.n_routed_experts,
            self.chunk_queries,
        )
        if min(dimensions) <= 0:
            raise ValueError("all schema dimensions must be positive")
        self.root = zarr.create_group(store=path, overwrite=overwrite)
        self.root.attrs.update(
            {
                "format": FORMAT,
                "probe_hb_layer": 5,
                "probe_denoise": 0,
                "state_token_intervened": False,
                "n_denoise": self.n_denoise,
                "n_flow_states": self.n_denoise + 1,
                "n_action_steps": self.n_action_steps,
                "max_action_dim": self.max_action_dim,
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "top_k": self.top_k,
                "n_routed_experts": self.n_routed_experts,
                "policy_codes": dict(POLICY_TO_CODE),
                "expected_policies": list(self.expected_policies),
                "raw_internal_definition": (
                    "runtime down_proj input m=SiLU(gate_proj(h))*up_proj(h); "
                    "absolute fp32 reductions in checkpoint units"
                ),
                "drop_definition": "(r - w_k*v_k)/(1-w_k) - r",
                "pair_hash_axes": ["pair_id", "draw_id", "action_token_index"],
                "normalization_action_std": action_std_array.tolist(),
                "x_traj_space": "model-normalized max_action_dim space",
                "primary_mechanism_target": (
                    "RMS_live7(((x10_policy-x1_policy)-"
                    "(x10_baseline-x1_baseline)))"
                ),
                "expert_output_storage": "signed pre-gate v_e in fp16",
            }
        )

        c = self.chunk_queries
        a = self.n_action_steps
        h = self.hidden_size
        k = self.top_k
        d = self.n_denoise
        m = self.max_action_dim
        e = self.n_routed_experts
        specs = [
            ("x_traj", (d + 1, a, m), (max(1, c // 4), d + 1, a, m), "float32"),
            ("original_input_hidden", (a, h), (max(1, c // 4), a, h), "float16"),
            ("original_shared_output", (a, h), (max(1, c // 4), a, h), "float16"),
            (
                "original_selected_expert_output",
                (a, k, h),
                (max(1, c // 16), a, k, h),
                "float16",
            ),
            ("original_router_probs", (a, e), (c, a, e), "float16"),
            ("original_selected_expert_id", (a, k), (c, a, k), "uint8"),
            ("original_selected_expert_weight", (a, k), (c, a, k), "float32"),
            ("internal_l2", (a, k), (c, a, k), "float32"),
            ("internal_l1", (a, k), (c, a, k), "float32"),
            ("internal_signed_mean", (a, k), (c, a, k), "float32"),
            ("internal_positive_fraction", (a, k), (c, a, k), "float32"),
            ("internal_linf", (a, k), (c, a, k), "float32"),
            ("original_routed", (a, h), (max(1, c // 4), a, h), "float32"),
            ("executed_routed", (a, h), (max(1, c // 4), a, h), "float32"),
            ("intervention_delta", (a, h), (max(1, c // 4), a, h), "float32"),
            ("dropped_slot", (a,), (c, a), "int8"),
            ("dropped_expert_id", (a,), (c, a), "uint8"),
            ("original_reconstruction_max_abs_error", (a,), (c, a), "float32"),
            ("observation_sha256", (32,), (c, 32), "uint8"),
            ("flow_noise_sha256", (32,), (c, 32), "uint8"),
            ("pair_id", (), (c,), "int64"),
            ("draw_id", (), (c,), "int64"),
            ("policy", (), (c,), "uint8"),
            ("query_id", (), (c,), "int64"),
            ("candidate_id", (), (c,), "int64"),
            ("episode_id", (), (c,), "int32"),
            ("control_step", (), (c,), "int32"),
        ]
        self.arrays: dict[str, Any] = {}
        compressors = [ZstdCodec(level=zstd_level)]
        for name, tail, chunks, dtype in specs:
            self.arrays[name] = self.root.create_array(
                name=name,
                shape=(0, *tail),
                chunks=chunks,
                dtype=dtype,
                compressors=compressors,
            )
        self._pending: dict[str, list[np.ndarray]] = {key: [] for key in self.arrays}
        self._n_pending = 0

    def append(
        self,
        record,
        x_traj: np.ndarray,
        *,
        observation_sha256: bytes | str,
        flow_noise_sha256: bytes | str,
    ) -> None:
        trajectory = np.asarray(x_traj, dtype=np.float32)
        if trajectory.ndim != 4:
            raise ValueError("x_traj must be [batch,flow,action,dimension]")
        batch = trajectory.shape[0]
        expected_trajectory = (
            batch,
            self.n_denoise + 1,
            self.n_action_steps,
            self.max_action_dim,
        )
        if trajectory.shape != expected_trajectory or not np.all(np.isfinite(trajectory)):
            raise ValueError("x_traj has an invalid shape or non-finite values")
        policy = str(record.intervention_drop_policy)
        if policy not in self.expected_policies:
            raise ValueError("record policy is not declared by this store")
        expected_arm = "baseline" if policy == "baseline" else "drop"
        if record.intervention_arm != expected_arm:
            raise ValueError("record arm and policy disagree")
        if int(record.probe_hb_layer) != 5 or int(record.probe_denoise) != 0:
            raise ValueError("v5 only accepts the HB5/d0 probe")
        if int(record.intervention_pair_id) < 0 or int(record.intervention_draw_id) < 0:
            raise ValueError("record has invalid pair/draw identity")
        layer_position = np.flatnonzero(np.asarray(record.hb_layers) == 5)
        if layer_position.size != 1:
            raise ValueError("record must contain exactly one HB5 layer")
        layer = int(layer_position[0])
        a, h, k, e = (
            self.n_action_steps,
            self.hidden_size,
            self.top_k,
            self.n_routed_experts,
        )
        values = {
            "x_traj": trajectory,
            "original_input_hidden": _require(
                "original_input_hidden", record.hb_probe_input_hidden, (batch, a, h), "float16"
            ),
            "original_shared_output": _require(
                "original_shared_output", record.hb_probe_shared_output, (batch, a, h), "float16"
            ),
            "original_selected_expert_output": _require(
                "original_selected_expert_output",
                record.hb_probe_selected_expert_raw,
                (batch, a, k, h),
                "float16",
            ),
            "original_router_probs": _require(
                "original_router_probs", record.hb_probe_router_probs, (batch, a, e), "float16"
            ),
            "original_selected_expert_id": _require(
                "original_selected_expert_id",
                record.hb_selected_expert_id[:, layer, 0],
                (batch, a, k),
                "uint8",
            ),
            "original_selected_expert_weight": _require(
                "original_selected_expert_weight",
                record.hb_selected_expert_weight[:, layer, 0],
                (batch, a, k),
                "float32",
            ),
            "internal_l2": _require(
                "internal_l2", record.hb_probe_internal_l2, (batch, a, k), "float32"
            ),
            "internal_l1": _require(
                "internal_l1", record.hb_probe_internal_l1, (batch, a, k), "float32"
            ),
            "internal_signed_mean": _require(
                "internal_signed_mean",
                record.hb_probe_internal_signed_mean,
                (batch, a, k),
                "float32",
            ),
            "internal_positive_fraction": _require(
                "internal_positive_fraction",
                record.hb_probe_internal_positive_fraction,
                (batch, a, k),
                "float32",
            ),
            "internal_linf": _require(
                "internal_linf", record.hb_probe_internal_linf, (batch, a, k), "float32"
            ),
            "original_routed": _require(
                "original_routed", record.hb_probe_original_routed, (batch, a, h), "float32"
            ),
            "executed_routed": _require(
                "executed_routed", record.hb_probe_executed_routed, (batch, a, h), "float32"
            ),
            "intervention_delta": _require(
                "intervention_delta", record.hb_probe_intervention_delta, (batch, a, h), "float32"
            ),
            "dropped_slot": _require(
                "dropped_slot", record.hb_probe_dropped_slot, (batch, a), "int8"
            ),
            "dropped_expert_id": _require(
                "dropped_expert_id", record.hb_probe_dropped_expert_id, (batch, a), "uint8"
            ),
            "original_reconstruction_max_abs_error": _require(
                "original_reconstruction_max_abs_error",
                record.hb_probe_routed_reconstruction_max_abs_error,
                (batch, a),
                "float32",
            ),
        }
        l2 = values["internal_l2"]
        l1 = values["internal_l1"]
        linf = values["internal_linf"]
        positive = values["internal_positive_fraction"]
        signed = values["internal_signed_mean"]
        if np.any(l2 < 0) or np.any(l1 < 0) or np.any(linf < 0):
            raise ValueError("internal activation norms must be non-negative")
        if np.any(positive < 0) or np.any(positive > 1):
            raise ValueError("internal positive fraction must lie in [0,1]")
        if np.any(l1 + 1e-5 < l2) or np.any(l2 + 1e-5 < linf):
            raise ValueError("internal L1/L2/Linf ordering is invalid")
        if np.any(np.abs(signed) > l1 / self.intermediate_size + 1e-5):
            raise ValueError("internal signed mean is inconsistent with L1")

        original = values["original_routed"]
        executed = values["executed_routed"]
        delta = values["intervention_delta"]
        if not np.allclose(executed, original + delta, rtol=2e-6, atol=2e-6):
            raise ValueError("executed routed output is not original + delta")
        slots = values["dropped_slot"].astype(np.int64)
        if policy == "baseline":
            if not (
                np.array_equal(executed, original)
                and np.count_nonzero(delta) == 0
                and np.all(slots == -1)
                and np.all(values["dropped_expert_id"] == 255)
            ):
                raise ValueError("baseline row is not a strict no-op sentinel")
        else:
            if np.any(slots < 0) or np.any(slots >= self.top_k):
                raise ValueError("drop policy produced an invalid selected slot")
            actual_drop_id = np.take_along_axis(
                values["original_selected_expert_id"], slots[..., None], axis=-1
            ).squeeze(-1)
            if not np.array_equal(actual_drop_id, values["dropped_expert_id"]):
                raise ValueError("dropped expert ID does not match selected slot")
        if float(values["original_reconstruction_max_abs_error"].max()) > 1e-5:
            raise ValueError("runtime original-routed reconstruction audit failed")

        observation_digest = _digest_array(observation_sha256)
        noise_digest = _digest_array(flow_noise_sha256)
        values.update(
            {
                "observation_sha256": np.broadcast_to(observation_digest, (batch, 32)).copy(),
                "flow_noise_sha256": np.broadcast_to(noise_digest, (batch, 32)).copy(),
                "pair_id": np.full(batch, record.intervention_pair_id, dtype=np.int64),
                "draw_id": np.full(batch, record.intervention_draw_id, dtype=np.int64),
                "policy": np.full(batch, POLICY_TO_CODE[policy], dtype=np.uint8),
                "query_id": np.full(batch, record.query_id, dtype=np.int64),
                "candidate_id": np.full(batch, record.candidate_id, dtype=np.int64),
                "episode_id": np.full(batch, record.episode_id, dtype=np.int32),
                "control_step": np.full(batch, record.control_step, dtype=np.int32),
            }
        )
        if set(values) != set(self._pending):
            raise RuntimeError("v5 append schema does not match allocated arrays")
        for name, array in values.items():
            self._pending[name].append(array)
        self._n_pending += batch
        if self._n_pending >= self.chunk_queries:
            self.flush()

    def flush(self) -> None:
        if not self._n_pending:
            return
        for name, chunks in self._pending.items():
            self.arrays[name].append(np.concatenate(chunks, axis=0), axis=0)
            chunks.clear()
        self._n_pending = 0

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "ZarrRawPruningWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZarrRawPruningReader:
    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") != FORMAT:
            raise ValueError("not an HB5/d0 raw-pruning v5 store: %s" % path)

    def __len__(self) -> int:
        return int(self.root["x_traj"].shape[0])

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())


def validate_raw_pruning_pairs(reader: ZarrRawPruningReader) -> dict[str, float | int]:
    pair = np.asarray(reader["pair_id"][:], dtype=np.int64)
    draw = np.asarray(reader["draw_id"][:], dtype=np.int64)
    policy = np.asarray(reader["policy"][:], dtype=np.int64)
    expected_names = tuple(reader.meta["expected_policies"])
    expected_codes = {POLICY_TO_CODE[name] for name in expected_names}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(pair, draw, strict=True)):
        groups.setdefault(key, []).append(index)
    invariants = (
        "original_input_hidden",
        "original_shared_output",
        "original_selected_expert_output",
        "original_router_probs",
        "original_selected_expert_id",
        "original_selected_expert_weight",
        "internal_l2",
        "internal_l1",
        "internal_signed_mean",
        "internal_positive_fraction",
        "internal_linf",
        "original_routed",
        "observation_sha256",
        "flow_noise_sha256",
    )
    max_original_difference = 0.0
    max_x0_difference = 0.0
    for key, indices in groups.items():
        if len(indices) != len(expected_codes) or set(policy[indices]) != expected_codes:
            names = [CODE_TO_POLICY.get(int(value), str(value)) for value in policy[indices]]
            raise ValueError("pair %s has incomplete policy rows: %s" % (key, names))
        for name in invariants:
            values = np.asarray(reader[name][indices])
            reference = values[0]
            if not all(np.array_equal(value, reference) for value in values[1:]):
                if np.issubdtype(values.dtype, np.number):
                    max_original_difference = max(
                        max_original_difference,
                        float(np.max(np.abs(values.astype(np.float64) - reference))),
                    )
                raise ValueError("pair %s disagrees in %s" % (key, name))
        x0 = np.asarray(reader["x_traj"][indices, 0], dtype=np.float32)
        max_x0_difference = max(max_x0_difference, float(np.max(np.abs(x0 - x0[0]))))
        if not all(np.array_equal(value, x0[0]) for value in x0[1:]):
            raise ValueError("pair %s did not reuse exact explicit x0" % (key,))
        baseline = indices[
            np.flatnonzero(policy[indices] == POLICY_TO_CODE["baseline"])[0]
        ]
        if not (
            np.array_equal(
                reader["original_routed"][baseline], reader["executed_routed"][baseline]
            )
            and np.count_nonzero(reader["intervention_delta"][baseline]) == 0
        ):
            raise ValueError("pair %s baseline is not an exact no-op" % (key,))
    return {
        "rows": len(reader),
        "complete_pairs": len(groups),
        "policies_per_pair": len(expected_codes),
        "max_original_pair_difference": max_original_difference,
        "max_x0_pair_difference": max_x0_difference,
    }


def raw_pruning_mechanism_metrics(
    reader: ZarrRawPruningReader,
) -> list[dict[str, float | int | str]]:
    pair = np.asarray(reader["pair_id"][:], dtype=np.int64)
    draw = np.asarray(reader["draw_id"][:], dtype=np.int64)
    policy = np.asarray(reader["policy"][:], dtype=np.int64)
    x = np.asarray(reader["x_traj"][:, :, :, :7], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for key in sorted(set(zip(pair, draw, strict=True))):
        indices = np.flatnonzero((pair == key[0]) & (draw == key[1]))
        baseline_index = indices[
            np.flatnonzero(policy[indices] == POLICY_TO_CODE["baseline"])[0]
        ]
        baseline = x[baseline_index]
        for index in indices:
            policy_name = CODE_TO_POLICY[int(policy[index])]
            if policy_name == "baseline":
                continue
            changed = x[index]
            remaining = (changed[10] - changed[1]) - (baseline[10] - baseline[1])
            final_delta = changed[10] - baseline[10]
            initial_delta = changed[1] - baseline[1]
            initial_rms = float(np.sqrt(np.mean(initial_delta**2)))
            final_rms = float(np.sqrt(np.mean(final_delta**2)))
            rows.append(
                {
                    "pair_id": int(key[0]),
                    "draw_id": int(key[1]),
                    "policy": policy_name,
                    "f_rem_rms_live7": float(np.sqrt(np.mean(remaining**2))),
                    "final_delta_rms_live7": final_rms,
                    "initial_delta_rms_live7": initial_rms,
                    "final_amplification_over_x1": final_rms / max(initial_rms, 1e-12),
                }
            )
    return rows
