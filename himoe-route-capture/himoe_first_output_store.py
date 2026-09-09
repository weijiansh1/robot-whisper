"""Zarr store and exact K-pool reductions for first-control MoE outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from himoe_first_output_recorder import FirstControlOutputRecord


FORMAT = "himoe_first_control_moe_output_v1"


@dataclass(frozen=True)
class EpisodePlan:
    """Decode the ordering used by rollout_with_routes.py.

    That client enumerates ``for init_state in scene_ids`` and, inside each
    state, ``for repeat in range(draws)``.  Flow-noise seeds are shared across
    states: ``noise_seed_base + repeat``.
    """

    scene_ids: tuple[int, ...]
    draws: int
    noise_seed_base: int = 1000

    def __post_init__(self) -> None:
        if not self.scene_ids:
            raise ValueError("scene_ids must not be empty")
        if len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError("scene_ids contain duplicates")
        if self.draws <= 0:
            raise ValueError("draws must be positive")

    @property
    def episodes(self) -> int:
        return len(self.scene_ids) * self.draws

    def decode(self, episode_id: int) -> tuple[int, int, int]:
        if episode_id < 0 or episode_id >= self.episodes:
            raise ValueError(
                "episode_id %d is outside the declared %d-episode plan"
                % (episode_id, self.episodes)
            )
        scene_pos, repeat = divmod(int(episode_id), self.draws)
        return self.scene_ids[scene_pos], self.noise_seed_base + repeat, repeat


def parse_scene_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not ids:
        raise ValueError("--scene-ids produced an empty plan")
    return ids


class ZarrFirstOutputWriter:
    """Buffer one same-state K pool, then write candidate and group statistics.

    ``hb_post_output`` is not persisted by default.  It is buffered as fp32 for
    one pool to compute exact candidate-to-centroid and
    group dispersion metrics.  This keeps the default artifact small while
    avoiding a random-projection approximation to convergence.
    """

    def __init__(
        self,
        path: str,
        expected_k: int,
        task_id: int,
        task_name: str,
        suite: str,
        benchmark: str,
        hb_layers: list[int],
        as_layers: list[int],
        n_denoise: int,
        n_action_steps: int,
        hidden_size: int,
        top_k: int = 4,
        store_hb_vectors: bool = False,
        zstd_level: int = 3,
        overwrite: bool = True,
    ) -> None:
        self.path = path
        self.expected_k = int(expected_k)
        self.task_id = int(task_id)
        self.store_hb_vectors = bool(store_hb_vectors)
        self._codec = [ZstdCodec(level=zstd_level)]
        self.root = zarr.create_group(store=path, overwrite=overwrite)
        self.root.attrs.update(
            {
                "format": FORMAT,
                "task_id": int(task_id),
                "task_name": str(task_name),
                "suite": str(suite),
                "benchmark": str(benchmark),
                "expected_k": self.expected_k,
                "n_hb_layers": len(hb_layers),
                "n_as_layers": len(as_layers),
                "n_denoise": int(n_denoise),
                "n_action_steps": int(n_action_steps),
                "hidden_size": int(hidden_size),
                "top_k": int(top_k),
                "store_hb_vectors": self.store_hb_vectors,
                "post_moe_definition": "MLP output routed + shared, before transformer residual add",
                "convergence_definition": (
                    "exact across candidates at fixed task/init/control/layer/denoise/token; "
                    "relative_dispersion = RMS(||y_i-mean(y)||) / RMS(||y_i||)"
                ),
            }
        )
        self.hb_layers = np.asarray(hb_layers, dtype=np.int16)
        self.as_layers = np.asarray(as_layers, dtype=np.int16)
        self.denoise_ids = np.arange(n_denoise, dtype=np.int16)
        self.token_ids = np.arange(1, n_action_steps + 1, dtype=np.int16)
        for name, value in (
            ("hb_layer_id", self.hb_layers),
            ("as_layer_id", self.as_layers),
            ("denoise_id", self.denoise_ids),
            ("action_token_id", self.token_ids),
        ):
            arr = self.root.create_array(
                name=name,
                shape=value.shape,
                chunks=value.shape,
                dtype=value.dtype,
                compressors=self._codec,
            )
            arr[:] = value

        Lh, La = len(hb_layers), len(as_layers)
        D, A, H, K = int(n_denoise), int(n_action_steps), int(hidden_size), int(top_k)
        c = max(1, min(self.expected_k, 32))
        candidate_specs = [
            ("episode_id", (), (c,), "int32"),
            ("task_id", (), (c,), "int16"),
            ("init_state_id", (), (c,), "int16"),
            ("flow_seed", (), (c,), "int64"),
            ("control_step", (), (c,), "int16"),
            ("group_id", (), (c,), "int32"),
            ("flow_noise_sha256", (32,), (c, 32), "uint8"),
            ("hb_expert_ids", (Lh, D, A, K), (c, Lh, D, A, K), "uint8"),
            ("hb_combine_weight", (Lh, D, A, K), (c, Lh, D, A, K), "float16"),
            ("hb_routed_norm", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_shared_norm", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_post_norm", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_branch_cosine", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_branch_angle_deg", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_post_distance_to_group_mean", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("hb_post_cosine_to_group_mean", (Lh, D, A), (c, Lh, D, A), "float32"),
            ("as_expert_ids", (La, D, A), (c, La, D, A), "uint8"),
            # One candidate chunk is ~0.4 MiB for AS0/1 on LIBERO.
            ("as_output", (La, D, A, H), (1, La, D, A, H), "float16"),
        ]
        if self.store_hb_vectors:
            for name in ("hb_routed_output", "hb_shared_output"):
                candidate_specs.append(
                    (name, (Lh, D, A, H), (1, Lh, D, A, H), "float16")
                )

        group_specs = [
            ("group_task_id", (), (64,), "int16"),
            ("group_init_state_id", (), (64,), "int16"),
            ("group_size", (), (64,), "int16"),
            ("group_complete", (), (64,), "uint8"),
            ("group_row_start", (), (64,), "int64"),
            ("hb_post_rms_dispersion", (Lh, D, A), (1, Lh, D, A), "float32"),
            ("hb_post_relative_dispersion", (Lh, D, A), (1, Lh, D, A), "float32"),
            ("hb_post_mean_pair_cosine", (Lh, D, A), (1, Lh, D, A), "float32"),
            ("hb_post_contraction_from_d0", (Lh, D, A), (1, Lh, D, A), "float32"),
        ]
        self.candidate_arrays = self._create_resizable(candidate_specs)
        self.group_arrays = self._create_resizable(group_specs)
        self._pending: list[FirstControlOutputRecord] = []
        self._group_key: tuple[int, int] | None = None
        self.rows = 0
        self.groups = 0
        self.incomplete_groups = 0

    def _create_resizable(self, specs):
        arrays: dict[str, Any] = {}
        for name, tail, chunks, dtype in specs:
            arrays[name] = self.root.create_array(
                name=name,
                shape=(0, *tail),
                chunks=chunks,
                dtype=dtype,
                compressors=self._codec,
            )
        return arrays

    @staticmethod
    def _append(array, value: np.ndarray) -> None:
        array.append(np.asarray(value), axis=0)

    def append(self, record: FirstControlOutputRecord) -> None:
        if record.hb_post_output.shape[0] != 1:
            raise ValueError("the grouped writer currently requires policy batch size 1")
        identity = record.identity
        if identity.task_id != self.task_id:
            raise ValueError("record task_id does not match this per-task store")
        key = (identity.task_id, identity.init_state_id)
        if self._group_key is not None and key != self._group_key:
            self._flush_group()
        if self._group_key is None:
            self._group_key = key
        if any(r.identity.flow_seed == identity.flow_seed for r in self._pending):
            raise ValueError("duplicate flow seed %d inside group %s" % (identity.flow_seed, key))
        self._pending.append(record)
        if len(self._pending) > self.expected_k:
            raise ValueError("group %s exceeded expected K=%d" % (key, self.expected_k))
        if len(self._pending) == self.expected_k:
            self._flush_group()

    def _flush_group(self) -> None:
        if not self._pending:
            self._group_key = None
            return
        n = len(self._pending)
        complete = n == self.expected_k
        if not complete:
            self.incomplete_groups += 1
        # [candidate, HB layer, denoise, action token, hidden]
        post = np.concatenate([r.hb_post_output for r in self._pending], axis=0).astype(
            np.float32, copy=False
        )
        mean = post.mean(axis=0)
        delta = post - mean[None]
        distance = np.linalg.norm(delta, axis=-1)
        rms_dispersion = np.sqrt(np.mean(np.square(distance), axis=0))
        post_norm = np.linalg.norm(post, axis=-1)
        rms_scale = np.sqrt(np.mean(np.square(post_norm), axis=0))
        relative = rms_dispersion / np.maximum(rms_scale, 1e-12)

        mean_norm = np.linalg.norm(mean, axis=-1)
        cosine_to_mean = np.sum(post * mean[None], axis=-1) / np.maximum(
            post_norm * mean_norm[None], 1e-12
        )
        if n > 1:
            unit = post / np.maximum(post_norm[..., None], 1e-12)
            unit_sum = unit.sum(axis=0)
            squared_self = np.sum(np.square(unit), axis=(0, -1))
            mean_pair_cosine = (
                np.sum(np.square(unit_sum), axis=-1) - squared_self
            ) / float(n * (n - 1))
        else:
            mean_pair_cosine = np.full_like(rms_dispersion, np.nan)
        base = relative[:, 0:1, :]
        contraction = np.full_like(relative, np.nan)
        np.divide(relative, base, out=contraction, where=base > 1e-12)
        contraction = 1.0 - contraction

        group_id = self.groups
        batch: dict[str, list[np.ndarray]] = {name: [] for name in self.candidate_arrays}
        for index, record in enumerate(self._pending):
            ident = record.identity
            scalar = {
                "episode_id": np.asarray([ident.episode_id], np.int32),
                "task_id": np.asarray([ident.task_id], np.int16),
                "init_state_id": np.asarray([ident.init_state_id], np.int16),
                "flow_seed": np.asarray([ident.flow_seed], np.int64),
                "control_step": np.asarray([ident.control_step], np.int16),
                "group_id": np.asarray([group_id], np.int32),
                "flow_noise_sha256": np.frombuffer(
                    ident.flow_noise_sha256, dtype=np.uint8
                ).reshape(1, 32),
            }
            values = {
                **scalar,
                "hb_expert_ids": record.hb_expert_ids,
                "hb_combine_weight": record.hb_combine_weight,
                "hb_routed_norm": record.hb_routed_norm,
                "hb_shared_norm": record.hb_shared_norm,
                "hb_post_norm": record.hb_post_norm,
                "hb_branch_cosine": record.hb_branch_cosine,
                "hb_branch_angle_deg": record.hb_branch_angle_deg,
                "hb_post_distance_to_group_mean": distance[index:index + 1],
                "hb_post_cosine_to_group_mean": cosine_to_mean[index:index + 1],
                "as_expert_ids": record.as_expert_ids,
                "as_output": record.as_output,
            }
            if self.store_hb_vectors:
                if record.hb_routed_output is None or record.hb_shared_output is None:
                    raise ValueError("writer requested HB vectors but record omitted them")
                values["hb_routed_output"] = record.hb_routed_output
                values["hb_shared_output"] = record.hb_shared_output
            for name, value in values.items():
                batch[name].append(np.asarray(value))
        for name, chunks in batch.items():
            self._append(self.candidate_arrays[name], np.concatenate(chunks, axis=0))

        group_values = {
            "group_task_id": np.asarray([self._group_key[0]], np.int16),
            "group_init_state_id": np.asarray([self._group_key[1]], np.int16),
            "group_size": np.asarray([n], np.int16),
            "group_complete": np.asarray([complete], np.uint8),
            "group_row_start": np.asarray([self.rows], np.int64),
            "hb_post_rms_dispersion": rms_dispersion[None].astype(np.float32),
            "hb_post_relative_dispersion": relative[None].astype(np.float32),
            "hb_post_mean_pair_cosine": mean_pair_cosine[None].astype(np.float32),
            "hb_post_contraction_from_d0": contraction[None].astype(np.float32),
        }
        for name, value in group_values.items():
            self._append(self.group_arrays[name], value)

        self.rows += n
        self.groups += 1
        self._pending.clear()
        self._group_key = None

    def close(self) -> None:
        self._flush_group()
        self.root.attrs.update(
            {
                "candidate_rows": self.rows,
                "groups": self.groups,
                "incomplete_groups": self.incomplete_groups,
            }
        )

    def __enter__(self) -> "ZarrFirstOutputWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
