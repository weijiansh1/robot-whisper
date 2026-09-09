"""Independent v4 storage for paired HB5/d0 expert interventions."""

from __future__ import annotations

from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from himoe_hb5_intervention import ARM_TO_CODE, ARMS, CODE_TO_ARM


FORMAT = "himoe_hb5_d0_intervention_v4"
STORE_NAME = "hb5_intervention_flow_v4.zarr"


def _digest_array(value: bytes | bytearray | memoryview | str) -> np.ndarray:
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as error:
            raise ValueError("digest string must be hexadecimal") from error
    else:
        raw = bytes(value)
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
        raise ValueError("%s is absent from the intervention record" % name)
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("%s must have shape %s, got %s" % (name, shape, array.shape))
    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError("%s contains non-finite values" % name)
    return array.astype(dtype, copy=False)


class ZarrHB5InterventionWriter:
    """Append query batches without sharing the v2/v3 activation-flow schema."""

    def __init__(
        self,
        path: str,
        *,
        n_denoise: int,
        n_action_steps: int,
        max_action_dim: int,
        hidden_size: int,
        top_k: int,
        n_routed_experts: int,
        action_std: list[float] | np.ndarray,
        drop_policy: str = "categorical_weighted",
        chunk_queries: int = 16,
        zstd_level: int = 3,
        overwrite: bool = True,
    ) -> None:
        self.n_denoise = int(n_denoise)
        self.n_action_steps = int(n_action_steps)
        self.max_action_dim = int(max_action_dim)
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.n_routed_experts = int(n_routed_experts)
        self.chunk_queries = int(chunk_queries)
        action_std_array = np.asarray(action_std, dtype=np.float64)
        if action_std_array.shape != (7,) or np.any(action_std_array <= 0):
            raise ValueError("action_std must contain seven positive values")
        if (
            min(
                self.n_denoise,
                self.n_action_steps,
                self.max_action_dim,
                self.hidden_size,
                self.top_k,
                self.n_routed_experts,
                self.chunk_queries,
            )
            <= 0
        ):
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
                "top_k": self.top_k,
                "n_routed_experts": self.n_routed_experts,
                "arms": list(ARMS),
                "arm_codes": dict(ARM_TO_CODE),
                "drop_policy": str(drop_policy),
                "drop_definition": "(r - w_k*v_k)/(1-w_k) - r",
                "random_control": (
                    "deterministic Gaussian tangent direction matched to drop "
                    "delta norm and dot(delta,r) per action token"
                ),
                "pair_hash_axes": ["pair_id", "draw_id", "action_token_index"],
                "normalization_action_std": action_std_array.tolist(),
                "x_traj_space": "model-normalized max_action_dim space",
                "primary_mechanism_target": (
                    "RMS_live7(((x10_arm-x1_arm)-(x10_baseline-x1_baseline)))"
                ),
                "original_vector_storage": "pre-gate signed fp16; runtime fp32 audit",
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
                "original_selected_expert_raw",
                (a, k, h),
                (max(1, c // 16), a, k, h),
                "float16",
            ),
            ("original_router_probs", (a, e), (c, a, e), "float16"),
            ("original_selected_expert_id", (a, k), (c, a, k), "uint8"),
            ("original_selected_expert_weight", (a, k), (c, a, k), "float32"),
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
            ("arm", (), (c,), "uint8"),
            ("query_id", (), (c,), "int64"),
            ("candidate_id", (), (c,), "int64"),
            ("episode_id", (), (c,), "int32"),
            ("control_step", (), (c,), "int32"),
        ]
        self.arrays: dict[str, Any] = {}
        compressor = [ZstdCodec(level=zstd_level)]
        for name, tail, chunks, dtype in specs:
            self.arrays[name] = self.root.create_array(
                name=name,
                shape=(0, *tail),
                chunks=chunks,
                dtype=dtype,
                compressors=compressor,
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
        if trajectory.shape != expected_trajectory or not np.all(
            np.isfinite(trajectory)
        ):
            raise ValueError("x_traj has an invalid shape or non-finite values")
        if int(record.probe_hb_layer) != 5 or int(record.probe_denoise) != 0:
            raise ValueError("v4 only accepts the HB5/d0 probe")
        if record.intervention_arm not in ARM_TO_CODE:
            raise ValueError("record has no valid intervention arm")
        if int(record.intervention_pair_id) < 0 or int(record.intervention_draw_id) < 0:
            raise ValueError("record has invalid pair/draw identity")
        if record.intervention_drop_policy != self.root.attrs["drop_policy"]:
            raise ValueError("record and store use different drop policies")
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
                "original_input_hidden",
                record.hb_probe_input_hidden,
                (batch, a, h),
                "float16",
            ),
            "original_shared_output": _require(
                "original_shared_output",
                record.hb_probe_shared_output,
                (batch, a, h),
                "float16",
            ),
            "original_selected_expert_raw": _require(
                "original_selected_expert_raw",
                record.hb_probe_selected_expert_raw,
                (batch, a, k, h),
                "float16",
            ),
            "original_router_probs": _require(
                "original_router_probs",
                record.hb_probe_router_probs,
                (batch, a, e),
                "float16",
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
            "original_routed": _require(
                "original_routed",
                record.hb_probe_original_routed,
                (batch, a, h),
                "float32",
            ),
            "executed_routed": _require(
                "executed_routed",
                record.hb_probe_executed_routed,
                (batch, a, h),
                "float32",
            ),
            "intervention_delta": _require(
                "intervention_delta",
                record.hb_probe_intervention_delta,
                (batch, a, h),
                "float32",
            ),
            "dropped_slot": _require(
                "dropped_slot", record.hb_probe_dropped_slot, (batch, a), "int8"
            ),
            "dropped_expert_id": _require(
                "dropped_expert_id",
                record.hb_probe_dropped_expert_id,
                (batch, a),
                "uint8",
            ),
            "original_reconstruction_max_abs_error": _require(
                "original_reconstruction_max_abs_error",
                record.hb_probe_routed_reconstruction_max_abs_error,
                (batch, a),
                "float32",
            ),
        }
        original = values["original_routed"]
        executed = values["executed_routed"]
        delta = values["intervention_delta"]
        if not np.allclose(executed, original + delta, rtol=2e-6, atol=2e-6):
            raise ValueError("executed routed output is not original + delta")
        if record.intervention_arm == "baseline" and not (
            np.array_equal(delta, np.zeros_like(delta))
            and np.array_equal(executed, original)
        ):
            raise ValueError("baseline arm is not a strict routed no-op")
        if float(values["original_reconstruction_max_abs_error"].max()) > 1e-5:
            raise ValueError("runtime original-routed reconstruction audit failed")
        slot = values["dropped_slot"].astype(np.int64)
        actual_drop_id = np.take_along_axis(
            values["original_selected_expert_id"], slot[..., None], axis=-1
        ).squeeze(-1)
        if not np.array_equal(actual_drop_id, values["dropped_expert_id"]):
            raise ValueError("dropped expert ID does not match the selected slot")

        observation_digest = _digest_array(observation_sha256)
        noise_digest = _digest_array(flow_noise_sha256)
        values.update(
            {
                "observation_sha256": np.broadcast_to(
                    observation_digest, (batch, 32)
                ).copy(),
                "flow_noise_sha256": np.broadcast_to(noise_digest, (batch, 32)).copy(),
                "pair_id": np.full(batch, record.intervention_pair_id, dtype=np.int64),
                "draw_id": np.full(batch, record.intervention_draw_id, dtype=np.int64),
                "arm": np.full(
                    batch, ARM_TO_CODE[record.intervention_arm], dtype=np.uint8
                ),
                "query_id": np.full(batch, record.query_id, dtype=np.int64),
                "candidate_id": np.full(batch, record.candidate_id, dtype=np.int64),
                "episode_id": np.full(batch, record.episode_id, dtype=np.int32),
                "control_step": np.full(batch, record.control_step, dtype=np.int32),
            }
        )
        if set(values) != set(self._pending):
            raise RuntimeError("v4 append schema does not match the allocated arrays")
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

    def __enter__(self) -> "ZarrHB5InterventionWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class ZarrHB5InterventionReader:
    def __init__(self, path: str) -> None:
        self.root = zarr.open_group(path, mode="r")
        self.meta = dict(self.root.attrs)
        if self.meta.get("format") != FORMAT:
            raise ValueError("not an HB5/d0 intervention v4 store: %s" % path)

    def __len__(self) -> int:
        return int(self.root["x_traj"].shape[0])

    def __getitem__(self, name: str):
        return self.root[name]

    @property
    def names(self) -> list[str]:
        return sorted(self.root.array_keys())


def validate_paired_store(reader: ZarrHB5InterventionReader) -> dict[str, float | int]:
    """Verify complete triads and all pre-intervention pairing invariants."""
    n = len(reader)
    pair = np.asarray(reader["pair_id"][:], dtype=np.int64)
    draw = np.asarray(reader["draw_id"][:], dtype=np.int64)
    arm = np.asarray(reader["arm"][:], dtype=np.int64)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(zip(pair, draw, strict=True)):
        groups.setdefault(key, []).append(index)
    original_names = (
        "original_input_hidden",
        "original_selected_expert_raw",
        "original_selected_expert_id",
        "original_selected_expert_weight",
        "original_routed",
        "dropped_slot",
        "dropped_expert_id",
        "observation_sha256",
        "flow_noise_sha256",
    )
    max_original_difference = 0.0
    max_x0_difference = 0.0
    for key, indices in groups.items():
        if len(indices) != len(ARMS) or set(arm[indices]) != set(CODE_TO_ARM):
            names = [CODE_TO_ARM.get(int(value), str(value)) for value in arm[indices]]
            raise ValueError(
                "pair %s is not one baseline/drop/random triad: %s" % (key, names)
            )
        baseline = indices[arm[indices].argmin()]
        for name in original_names:
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
            raise ValueError(
                "pair %s did not reuse the exact same explicit x0" % (key,)
            )
        original = np.asarray(reader["original_routed"][baseline])
        executed = np.asarray(reader["executed_routed"][baseline])
        delta = np.asarray(reader["intervention_delta"][baseline])
        if not (np.array_equal(original, executed) and np.count_nonzero(delta) == 0):
            raise ValueError("pair %s baseline is not a strict no-op" % (key,))
    return {
        "rows": n,
        "complete_pairs": len(groups),
        "max_original_pair_difference": max_original_difference,
        "max_x0_pair_difference": max_x0_difference,
    }


def paired_mechanism_metrics(
    reader: ZarrHB5InterventionReader,
) -> list[dict[str, float | int | str]]:
    """Compute frozen paired effects without fitting or selecting a score."""
    pair = np.asarray(reader["pair_id"][:], dtype=np.int64)
    draw = np.asarray(reader["draw_id"][:], dtype=np.int64)
    arm = np.asarray(reader["arm"][:], dtype=np.int64)
    x = np.asarray(reader["x_traj"][:, :, :, :7], dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    for key in sorted(set(zip(pair, draw, strict=True))):
        indices = np.flatnonzero((pair == key[0]) & (draw == key[1]))
        by_arm = {CODE_TO_ARM[int(arm[index])]: int(index) for index in indices}
        baseline = x[by_arm["baseline"]]
        for arm_name in ("drop", "random"):
            changed = x[by_arm[arm_name]]
            remaining = (changed[10] - changed[1]) - (baseline[10] - baseline[1])
            final_delta = changed[10] - baseline[10]
            initial_delta = changed[1] - baseline[1]
            initial_rms = float(np.sqrt(np.mean(initial_delta**2)))
            final_rms = float(np.sqrt(np.mean(final_delta**2)))
            rows.append(
                {
                    "pair_id": int(key[0]),
                    "draw_id": int(key[1]),
                    "arm": arm_name,
                    "f_rem_rms_live7": float(np.sqrt(np.mean(remaining**2))),
                    "final_delta_rms_live7": final_rms,
                    "initial_delta_rms_live7": initial_rms,
                    "final_amplification_over_x1": final_rms / max(initial_rms, 1e-12),
                }
            )
    return rows
