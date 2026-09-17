"""Policy backends used by the WebSocket service."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from himoe_libero_bridge.protocol import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    FLOW_STEPS,
    FLOW_NOISE_KEY,
    FLOW_NOISE_SHA256_KEY,
    HB_MOE_EXPERTS,
    HB_MOE_LAYERS,
    HB_MOE_TOP_K,
    MODEL_ACTION_DIM,
    ROUTING_CAPTURE_KEY,
    ROUTING_EXPERT_IDS_KEY,
    ROUTING_EXPERT_WEIGHTS_KEY,
    ROUTING_LAYER_INDICES_KEY,
    validate_action_response,
)
from himoe_libero_bridge.suites import get_suite

MOCK_NORMALIZATION_ACTION_STD = [0.4, 0.35, 0.5, 0.06, 0.08, 0.1, 0.97]
LIBERO_WRIST_LAYOUTS = ("released-left", "checkpoint-right", "paper-right", "both")
HIMOE_UPSTREAM_COMMIT = "27a2c46932d8b6373ca0074eb997f299bcd4f6f5"
LIBERO_UPSTREAM_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
HIMOE_PATCH_FILENAMES = (
    "himoe-fixed-noise.patch",
    "himoe-runtime.patch",
    "himoe-transformers-cache.patch",
)


def _sha256_path(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def himoe_patch_hashes(project_root: Optional[pathlib.Path] = None) -> Dict[str, str]:
    root = (
        pathlib.Path(__file__).resolve().parents[2]
        if project_root is None
        else pathlib.Path(project_root).expanduser().resolve()
    )
    patch_root = root / "patches"
    result = {}
    for filename in HIMOE_PATCH_FILENAMES:
        path = patch_root / filename
        if not path.is_file() or path.stat().st_size < 1:
            raise RuntimeError("Missing HiMoE runtime patch: %s" % path)
        result[filename] = _sha256_path(path)
    return result


def _git_command(root: pathlib.Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root)] + list(arguments),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not inspect HiMoE upstream %s: %s" % (root, error))
    if completed.returncode != 0:
        raise RuntimeError(
            "Could not inspect HiMoE upstream %s: %s"
            % (root, completed.stderr.decode("utf-8", errors="replace").strip())
        )
    return completed.stdout


def himoe_upstream_identity(
    upstream_root: pathlib.Path, project_root: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    root = pathlib.Path(upstream_root).expanduser().resolve()
    commit = _git_command(root, ("rev-parse", "--verify", "HEAD")).decode().strip()
    if commit != HIMOE_UPSTREAM_COMMIT:
        raise RuntimeError(
            "HiMoE upstream commit mismatch: expected %s, got %s"
            % (HIMOE_UPSTREAM_COMMIT, commit)
        )
    patch_hashes = himoe_patch_hashes(project_root)
    bridge_root = (
        pathlib.Path(__file__).resolve().parents[2]
        if project_root is None
        else pathlib.Path(project_root).expanduser().resolve()
    )
    for filename in HIMOE_PATCH_FILENAMES:
        patch = bridge_root / "patches" / filename
        try:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "apply",
                    "--reverse",
                    "--check",
                    "--unidiff-zero",
                    str(patch),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Could not verify applied HiMoE patch %s: %s" % (patch, error))
        if completed.returncode != 0:
            raise RuntimeError("HiMoE runtime patch is not applied: %s" % patch)
    diff = _git_command(root, ("diff", "--binary", "--no-ext-diff", "HEAD", "--"))
    status = _git_command(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    return {
        "commit": commit,
        "patch_sha256": patch_hashes,
        "patches_verified_applied": True,
        "working_tree_dirty": bool(status.strip()),
        "working_tree_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _load_normalization_action_std(path: pathlib.Path) -> List[float]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Could not parse normalization stats %s: %s" % (path, error))
    if not isinstance(document, Mapping):
        raise RuntimeError("Normalization stats root must be a mapping: %s" % path)
    actions = document.get("actions")
    if not isinstance(actions, Mapping):
        raise RuntimeError("Normalization stats must contain an actions mapping: %s" % path)
    raw_std = actions.get("std")
    if not isinstance(raw_std, list) or len(raw_std) != ACTION_DIM:
        raise RuntimeError(
            "Normalization actions.std must contain exactly %d values: %s"
            % (ACTION_DIM, path)
        )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in raw_std
    ):
        raise RuntimeError("Normalization actions.std values must be numeric: %s" % path)
    action_std = np.asarray(raw_std, dtype=np.float64)
    if not np.all(np.isfinite(action_std)) or np.any(action_std <= 0.0):
        raise RuntimeError(
            "Normalization actions.std values must be finite and strictly positive: %s"
            % path
        )
    return [float(value) for value in action_std]


@dataclasses.dataclass(frozen=True)
class _LiberoWristLayoutTransform:
    """Remap the released LIBERO wrist input for a controlled inference A/B."""

    layout: str

    def __post_init__(self) -> None:
        if self.layout not in LIBERO_WRIST_LAYOUTS:
            raise ValueError("Unknown LIBERO wrist layout: %s" % self.layout)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.layout == "released-left":
            return data

        images = dict(data["image"])
        masks = dict(data["image_mask"])
        wrist = np.asarray(images["left_wrist_0_rgb"])
        zeros = np.zeros_like(wrist)
        if self.layout in ("checkpoint-right", "paper-right"):
            images["left_wrist_0_rgb"] = zeros
            images["right_wrist_0_rgb"] = wrist
            masks["left_wrist_0_rgb"] = (
                np.True_ if self.layout == "checkpoint-right" else np.False_
            )
            masks["right_wrist_0_rgb"] = np.True_
        else:
            images["left_wrist_0_rgb"] = wrist
            images["right_wrist_0_rgb"] = wrist.copy()
            masks["left_wrist_0_rgb"] = np.True_
            masks["right_wrist_0_rgb"] = np.True_

        remapped = dict(data)
        remapped["image"] = images
        remapped["image_mask"] = masks
        return remapped


def _install_libero_wrist_layout(policy: Any, layout: str) -> str:
    """Insert the A/B transform after upstream LIBERO data transforms."""

    transform = getattr(policy, "_input_transform", None)
    transforms = list(getattr(transform, "transforms", ()))
    libero_indices = [
        index
        for index, item in enumerate(transforms)
        if type(item).__name__ == "LiberoInputs"
    ]
    if len(libero_indices) != 1:
        raise RuntimeError(
            "Expected exactly one LiberoInputs transform, found %d" % len(libero_indices)
        )
    libero_index = libero_indices[0]
    drop_indices = [
        index
        for index, item in enumerate(transforms)
        if index > libero_index and type(item).__name__ == "DropStateAndImage"
    ]
    if len(drop_indices) > 1:
        raise RuntimeError(
            "Expected at most one DropStateAndImage after LiberoInputs, found %d"
            % len(drop_indices)
        )
    if drop_indices and drop_indices[0] != libero_index + 1:
        raise RuntimeError("DropStateAndImage must immediately follow LiberoInputs")
    anchor_index = drop_indices[0] if drop_indices else libero_index
    anchor_name = type(transforms[anchor_index]).__name__
    transforms.insert(anchor_index + 1, _LiberoWristLayoutTransform(layout))
    policy._input_transform = type(transform)(tuple(transforms))
    return anchor_name


class MockPolicy:
    """Deterministic moving policy for protocol and simulator smoke tests."""

    backend_name = "mock"

    def __init__(self, suite: str = "goal") -> None:
        spec = get_suite(suite)
        self.metadata = {
            "suite": spec.key,
            "benchmark": spec.benchmark,
            "flow_steps": FLOW_STEPS,
            "predicted_action_steps": ACTION_CHUNK_STEPS,
            "internal_action_dim": MODEL_ACTION_DIM,
            "routing_capture_supported": True,
            "routing_hb_layer_indices": [2, 3, 4, 5, 12, 13, 14, 15],
            "routing_experts": HB_MOE_EXPERTS,
            "routing_top_k": HB_MOE_TOP_K,
            "normalization_action_std": list(MOCK_NORMALIZATION_ACTION_STD),
        }

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        actions = np.zeros((ACTION_CHUNK_STEPS, ACTION_DIM), dtype=np.float32)
        actions[:, 0] = np.linspace(0.005, 0.015, ACTION_CHUNK_STEPS, dtype=np.float32)
        actions[:, 2] = -0.002
        actions[:, 6] = -1.0
        response = {"actions": actions}
        if FLOW_NOISE_KEY in observation:
            flow_noise = np.ascontiguousarray(observation[FLOW_NOISE_KEY], dtype=np.float32)
            perturbation_scale = np.asarray(
                [0.004, 0.004, 0.003, 0.002, 0.002, 0.002], dtype=np.float32
            )
            actions[:, :6] += np.tanh(flow_noise[:, :6]) * perturbation_scale
            response[FLOW_NOISE_SHA256_KEY] = hashlib.sha256(flow_noise.tobytes()).hexdigest()
        if observation.get(ROUTING_CAPTURE_KEY, False):
            digest = hashlib.sha256(
                np.ascontiguousarray(
                    observation.get(
                        FLOW_NOISE_KEY,
                        np.zeros((ACTION_CHUNK_STEPS, MODEL_ACTION_DIM), dtype=np.float32),
                    ),
                    dtype=np.float32,
                ).tobytes()
            ).digest()
            base = int(digest[0])
            flow, layer, action, topk = np.indices(
                (FLOW_STEPS, HB_MOE_LAYERS, ACTION_CHUNK_STEPS, HB_MOE_TOP_K)
            )
            response[ROUTING_EXPERT_IDS_KEY] = np.asarray(
                (base + 3 * flow + 5 * layer + action + 7 * topk) % HB_MOE_EXPERTS,
                dtype=np.int16,
            )
            weights = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
            response[ROUTING_EXPERT_WEIGHTS_KEY] = np.broadcast_to(
                weights, response[ROUTING_EXPERT_IDS_KEY].shape
            ).copy()
            response[ROUTING_LAYER_INDICES_KEY] = np.asarray(
                [2, 3, 4, 5, 12, 13, 14, 15], dtype=np.int16
            )
        return response


class _HBRouteCapture:
    def __init__(
        self,
        routing_layers: Sequence[Tuple[int, Any]],
        flow_steps: int,
        action_steps: int,
    ) -> None:
        self._routing_layers = tuple(routing_layers)
        self._flow_steps = flow_steps
        self._action_steps = action_steps
        self._records = []  # type: List[Tuple[int, Any, Any]]
        self._handles = []  # type: List[Any]

    def __enter__(self) -> "_HBRouteCapture":
        for layer_index, gate in self._routing_layers:
            self._handles.append(gate.register_forward_hook(self._make_hook(layer_index)))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_index: int):
        def capture(_module: Any, inputs: Tuple[Any, ...], output: Tuple[Any, ...]) -> None:
            hidden_states = inputs[0]
            expert_ids, expert_weights, _ = output
            batch_size, sequence_length = hidden_states.shape[:2]
            if batch_size != 1:
                raise RuntimeError("Routing capture currently requires inference batch size 1")
            if sequence_length < self._action_steps:
                raise RuntimeError(
                    "Routing sequence has %d tokens; expected at least %d"
                    % (sequence_length, self._action_steps)
                )
            expert_ids = expert_ids.reshape(batch_size, sequence_length, -1)
            expert_weights = expert_weights.reshape(batch_size, sequence_length, -1)
            self._records.append(
                (
                    layer_index,
                    expert_ids[:, -self._action_steps :, :].detach().clone(),
                    expert_weights[:, -self._action_steps :, :].detach().clone(),
                )
            )

        return capture

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        layer_indices = [layer_index for layer_index, _ in self._routing_layers]
        expected_order = layer_indices * self._flow_steps
        actual_order = [layer_index for layer_index, _, _ in self._records]
        if actual_order != expected_order:
            raise RuntimeError(
                "Unexpected HB-MoE capture order: expected %s, got %s"
                % (expected_order, actual_order)
            )
        expert_ids = torch.stack([record[1] for record in self._records], dim=0)
        expert_weights = torch.stack([record[2] for record in self._records], dim=0)
        trace_shape = (
            self._flow_steps,
            len(layer_indices),
            self._action_steps,
            expert_ids.shape[-1],
        )
        expert_ids = expert_ids[:, 0].reshape(trace_shape).to(device="cpu", dtype=torch.int16)
        expert_weights = expert_weights[:, 0].reshape(trace_shape).to(
            device="cpu", dtype=torch.float32
        )
        return (
            expert_ids.numpy(),
            expert_weights.numpy(),
            np.asarray(layer_indices, dtype=np.int16),
        )


class HiMoEPolicy:
    def __init__(
        self,
        checkpoint_dir: str,
        suite: str = "goal",
        upstream_root: Optional[str] = None,
        require_cuda: bool = True,
        libero_wrist_layout: str = "released-left",
    ) -> None:
        if libero_wrist_layout not in LIBERO_WRIST_LAYOUTS:
            raise ValueError("Unknown LIBERO wrist layout: %s" % libero_wrist_layout)
        spec = get_suite(suite)
        self.backend_name = "himoe-vla-libero-%s" % spec.key
        checkpoint = pathlib.Path(checkpoint_dir).expanduser().resolve()
        os.environ.setdefault("MOEVLA_DATA_HOME", str(checkpoint.parent.parent / "moevla-data"))
        weights = checkpoint / "pytorch_model.pth"
        norm_stats = checkpoint / spec.normalization_asset / "meta" / "stats.json"
        if not weights.is_file():
            raise FileNotFoundError("Missing checkpoint weights: %s" % weights)
        if not norm_stats.is_file() or norm_stats.stat().st_size == 0:
            raise FileNotFoundError("Missing normalization assets: %s" % norm_stats)
        if weights.stat().st_size != spec.weights_bytes:
            raise RuntimeError(
                "%s checkpoint size mismatch: expected %d, got %d"
                % (spec.key, spec.weights_bytes, weights.stat().st_size)
            )
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != spec.weights_sha256:
            raise RuntimeError(
                "%s checkpoint SHA-256 mismatch: expected %s, got %s"
                % (spec.key, spec.weights_sha256, actual_sha256)
            )

        upstream_identity = None
        if upstream_root:
            root = pathlib.Path(upstream_root).expanduser().resolve()
            upstream_identity = himoe_upstream_identity(root)
            source = root / "src"
            client_source = root / "packages" / "openpi-client" / "src"
            for path in (str(source), str(client_source)):
                if path not in sys.path:
                    sys.path.insert(0, path)

        import torch

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the HiMoE backend")
        if torch.cuda.is_available():
            logging.info(
                "Loading HiMoE-VLA on cuda:0 (%s, %.1f GiB)",
                torch.cuda.get_device_name(0),
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3),
            )

        from moevla.policies import policy_config
        from moevla.training import config

        train_config = config.get_training_config(spec.train_config)
        dataset_config = dataclasses.replace(
            config.get_dataset_config(spec.dataset_config),
            assets_base_dir=str(checkpoint),
        )
        self._policy = policy_config.create_trained_policy(
            train_config,
            dataset_config,
            checkpoint,
            default_prompt=None,
        )
        wrist_layout_anchor = _install_libero_wrist_layout(
            self._policy, libero_wrist_layout
        )
        model_config = self._policy.model.config
        flow_steps = int(model_config.num_steps)
        predicted_action_steps = int(model_config.n_action_steps)
        internal_action_dim = int(model_config.max_action_dim)
        if flow_steps != 10:
            raise RuntimeError("Protocol requires 10 flow steps, model has %d" % flow_steps)
        if predicted_action_steps != ACTION_CHUNK_STEPS:
            raise RuntimeError(
                "Protocol requires %d predicted actions, model has %d"
                % (ACTION_CHUNK_STEPS, predicted_action_steps)
            )
        if internal_action_dim != MODEL_ACTION_DIM:
            raise RuntimeError(
                "Flow noise requires internal action dim %d, model has %d"
                % (MODEL_ACTION_DIM, internal_action_dim)
            )
        expert_model = self._policy.model.paligemma_with_expert.gemma_expert
        self._routing_layers = []  # type: List[Tuple[int, Any]]
        for layer_index, layer in enumerate(expert_model.layers):
            mlp = layer.mlp
            if type(mlp).__name__ == "HBMoE":
                gate = mlp.gate
                if int(gate.n_routed_experts) != HB_MOE_EXPERTS or int(gate.top_k) != HB_MOE_TOP_K:
                    raise RuntimeError(
                        "Unexpected HB-MoE gate at layer %d: experts=%s top_k=%s"
                        % (layer_index, gate.n_routed_experts, gate.top_k)
                    )
                self._routing_layers.append((layer_index, gate))
        if len(self._routing_layers) != HB_MOE_LAYERS:
            raise RuntimeError(
                "Routing protocol requires %d HB-MoE layers, model has %d"
                % (HB_MOE_LAYERS, len(self._routing_layers))
            )
        norm_digest = hashlib.sha256(norm_stats.read_bytes()).hexdigest()
        normalization_action_std = _load_normalization_action_std(norm_stats)
        self.metadata = {
            "suite": spec.key,
            "benchmark": spec.benchmark,
            "train_config": spec.train_config,
            "dataset_config": spec.dataset_config,
            "normalization_asset": spec.normalization_asset,
            "checkpoint": str(checkpoint),
            "checkpoint_bytes": weights.stat().st_size,
            "checkpoint_sha256": actual_sha256,
            "flow_steps": flow_steps,
            "predicted_action_steps": predicted_action_steps,
            "internal_action_dim": internal_action_dim,
            "normalization_stats_path": str(norm_stats),
            "normalization_stats_sha256": norm_digest,
            "normalization_stats_loaded": True,
            "normalization_action_std": normalization_action_std,
            "libero_wrist_layout": libero_wrist_layout,
            "libero_wrist_layout_transform_anchor": wrist_layout_anchor,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "routing_capture_supported": True,
            "routing_hb_layer_indices": [item[0] for item in self._routing_layers],
            "routing_experts": HB_MOE_EXPERTS,
            "routing_top_k": HB_MOE_TOP_K,
        }
        if upstream_identity is None:
            raise RuntimeError(
                "HiMoE upstream_root is required for auditable source identity"
            )
        self.metadata.update(
            {
                "himoe_upstream_commit": upstream_identity["commit"],
                "himoe_patch_sha256": upstream_identity["patch_sha256"],
                "himoe_patches_verified_applied": upstream_identity[
                    "patches_verified_applied"
                ],
                "himoe_working_tree_dirty": upstream_identity["working_tree_dirty"],
                "himoe_working_tree_diff_sha256": upstream_identity[
                    "working_tree_diff_sha256"
                ],
            }
        )
        if torch.cuda.is_available():
            self.metadata["gpu"] = torch.cuda.get_device_name(0)
            self.metadata["gpu_memory_allocated_bytes"] = torch.cuda.memory_allocated(0)

    def infer(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        policy_observation = dict(observation)
        capture_routing = bool(policy_observation.pop(ROUTING_CAPTURE_KEY, False))
        if capture_routing:
            with _HBRouteCapture(
                self._routing_layers,
                flow_steps=FLOW_STEPS,
                action_steps=ACTION_CHUNK_STEPS,
            ) as route_capture:
                response = self._policy.infer(policy_observation)
            expert_ids, expert_weights, layer_indices = route_capture.arrays()
            response[ROUTING_EXPERT_IDS_KEY] = expert_ids
            response[ROUTING_EXPERT_WEIGHTS_KEY] = expert_weights
            response[ROUTING_LAYER_INDICES_KEY] = layer_indices
        else:
            response = self._policy.infer(policy_observation)
        response = validate_action_response(response)
        if FLOW_NOISE_KEY in policy_observation:
            flow_noise = np.ascontiguousarray(policy_observation[FLOW_NOISE_KEY], dtype=np.float32)
            response[FLOW_NOISE_SHA256_KEY] = hashlib.sha256(flow_noise.tobytes()).hexdigest()
        return response
