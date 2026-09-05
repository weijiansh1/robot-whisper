"""Opt-in, auditable low-memory loader for the released HiMoE checkpoint.

The upstream loader first constructs a 15.2 GiB fp32 model and then reads a
second copy of the weights. This context changes only that initialization:
parameters are constructed on ``meta`` and the checkpoint is memory mapped.
For CPU inference the mapped bf16 storages are assigned directly. For CUDA,
each mapped tensor is converted to the model's construction dtype directly on
the target device before ``load_state_dict(assign=True)`` installs it. This
keeps CPU resident memory bounded without changing the deployed fp32 weights.
"""

from __future__ import annotations

import contextlib
import dataclasses
import pathlib
import sys
from typing import Any, Iterator
from unittest import mock


_META_BUFFER_NAMES = {
    "paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.position_ids",
    "paligemma_with_expert.paligemma.language_model.model.rotary_emb.inv_freq",
    "paligemma_with_expert.gemma_expert.rotary_emb.inv_freq",
}


@dataclasses.dataclass
class LowMemoryLoadAudit:
    checkpoint_path: str
    upstream_root: str
    train_config: str
    target_device: str
    model_create_calls: int = 0
    checkpoint_load_calls: int = 0
    assign_calls: int = 0
    state_key_count: int = 0
    logical_tensor_bytes: int = 0
    target_tensor_bytes: int = 0
    staged_tensor_count: int = 0
    aliased_tensor_count: int = 0
    reconstructed_buffers: tuple[str, ...] = ()
    _model: Any = dataclasses.field(default=None, repr=False)

    def validate_complete(self) -> None:
        counts = (
            self.model_create_calls,
            self.checkpoint_load_calls,
            self.assign_calls,
        )
        if counts != (1, 1, 1):
            raise RuntimeError(
                "low-memory load did not intercept exactly one create/load/assign: %r"
                % (counts,)
            )
        if self.aliased_tensor_count != self.state_key_count:
            raise RuntimeError(
                "only %d/%d checkpoint tensors retained mmap storage"
                % (self.aliased_tensor_count, self.state_key_count)
            )
        remaining = _meta_tensor_names(self._model)
        if remaining:
            raise RuntimeError("model still contains meta tensors: %s" % remaining)

    def as_metadata(self) -> dict[str, Any]:
        self.validate_complete()
        return {
            "mode": "torch_mmap_meta_assign",
            "checkpoint_path": self.checkpoint_path,
            "upstream_root": self.upstream_root,
            "train_config": self.train_config,
            "target_device": self.target_device,
            "mmap": True,
            "assign": True,
            "strict": True,
            "state_key_count": self.state_key_count,
            "logical_tensor_bytes": self.logical_tensor_bytes,
            "target_tensor_bytes": self.target_tensor_bytes,
            "staged_tensor_count": self.staged_tensor_count,
            "aliased_tensor_count": self.aliased_tensor_count,
            "reconstructed_nonpersistent_buffers": list(self.reconstructed_buffers),
        }


def _meta_tensor_names(model: Any) -> list[str]:
    parameters = [name for name, value in model.named_parameters() if value.is_meta]
    buffers = [name for name, value in model.named_buffers() if value.is_meta]
    return sorted(parameters + buffers)


def _materialize_nonpersistent_buffers(
    model: Any, torch: Any, device: Any
) -> tuple[str, ...]:
    meta_buffers = {
        name for name, value in model.named_buffers() if value.is_meta
    }
    if meta_buffers != _META_BUFFER_NAMES:
        raise RuntimeError(
            "unexpected construction-time meta buffers: expected %s, got %s"
            % (sorted(_META_BUFFER_NAMES), sorted(meta_buffers))
        )

    embeddings = (
        model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings
    )
    embeddings.position_ids = torch.arange(
        embeddings.num_patches,
        dtype=embeddings.position_ids.dtype,
        device=device,
    ).expand((1, -1))

    rotary_modules = (
        model.paligemma_with_expert.paligemma.language_model.model.rotary_emb,
        model.paligemma_with_expert.gemma_expert.rotary_emb,
    )
    for rotary in rotary_modules:
        inv_freq, attention_scaling = rotary.rope_init_fn(
            rotary.config, device
        )
        if inv_freq.device.type != device.type:
            raise RuntimeError(
                "rotary buffer reconstruction did not use target device %s" % device
            )
        rotary.inv_freq = inv_freq
        rotary.original_inv_freq = inv_freq
        rotary.attention_scaling = attention_scaling

    remaining = [name for name, value in model.named_buffers() if value.is_meta]
    if remaining:
        raise RuntimeError("failed to reconstruct meta buffers: %s" % remaining)
    return tuple(sorted(meta_buffers))


def _training_config(upstream_root: pathlib.Path, name: str) -> Any:
    source = str(upstream_root / "src")
    client_source = str(upstream_root / "packages" / "openpi-client" / "src")
    for path in (source, client_source):
        if path not in sys.path:
            sys.path.insert(0, path)
    from moevla.training import config

    return config.get_training_config(name)


@contextlib.contextmanager
def low_memory_himoe_load(
    checkpoint_dir: str | pathlib.Path,
    upstream_root: str | pathlib.Path,
    train_config_name: str,
    *,
    target_device: str = "cpu",
) -> Iterator[LowMemoryLoadAudit]:
    """Patch the released loader for one bounded-memory construction.

    The checkpoint hash and upstream source checks remain in ``HiMoEPolicy``;
    this context only replaces allocation mechanics after those checks.
    """
    import torch

    checkpoint = pathlib.Path(checkpoint_dir).expanduser().resolve()
    weights = (checkpoint / "pytorch_model.pth").resolve()
    root = pathlib.Path(upstream_root).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError("missing checkpoint weights: %s" % weights)
    if not (root / "src" / "moevla").is_dir():
        raise FileNotFoundError("invalid HiMoE upstream root: %s" % root)
    device = torch.device(target_device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("unsupported low-memory target device: %s" % device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA target requested but CUDA is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

    train_config = _training_config(root, train_config_name)
    model_config = train_config.model
    original_create = model_config.create
    original_torch_load = torch.load
    audit = LowMemoryLoadAudit(
        checkpoint_path=str(weights),
        upstream_root=str(root),
        train_config=train_config_name,
        target_device=str(device),
    )
    expected_state: dict[str, tuple[Any, tuple[int, ...]]] = {}

    def create_meta_model() -> Any:
        audit.model_create_calls += 1
        if audit.model_create_calls != 1:
            raise RuntimeError("low-memory model factory was called more than once")
        with torch.device("meta"):
            model = original_create()
        meta_parameters = [value for value in model.parameters() if value.is_meta]
        if len(meta_parameters) != sum(1 for _ in model.parameters()):
            raise RuntimeError("model factory allocated non-meta parameters")
        expected_state.update(
            {
                name: (value.dtype, tuple(value.shape))
                for name, value in model.state_dict().items()
            }
        )

        original_load_state_dict = model.load_state_dict

        def assign_state_dict(state_dict: Any, strict: bool = True, **kwargs: Any) -> Any:
            audit.assign_calls += 1
            if audit.assign_calls != 1:
                raise RuntimeError("model state was assigned more than once")
            if strict is not True:
                raise RuntimeError("low-memory checkpoint load requires strict=True")
            if kwargs.get("assign") not in (None, True):
                raise RuntimeError("low-memory checkpoint load requires assign=True")
            result = original_load_state_dict(
                state_dict, strict=True, assign=True
            )
            audit.reconstructed_buffers = _materialize_nonpersistent_buffers(
                model, torch, device
            )
            current = model.state_dict()
            audit.aliased_tensor_count = sum(
                current[name].untyped_storage().data_ptr()
                == tensor.untyped_storage().data_ptr()
                for name, tensor in state_dict.items()
            )
            audit._model = model
            delattr(model, "load_state_dict")
            return result

        model.load_state_dict = assign_state_dict
        return model

    def mmap_load(file: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            requested = pathlib.Path(file).expanduser().resolve()
        except TypeError:
            return original_torch_load(file, *args, **kwargs)
        if requested != weights:
            return original_torch_load(file, *args, **kwargs)
        audit.checkpoint_load_calls += 1
        if audit.checkpoint_load_calls != 1:
            raise RuntimeError("checkpoint was loaded more than once")
        if args:
            raise RuntimeError("checkpoint torch.load must use audited keyword arguments")
        if kwargs.get("map_location") != "cpu" or kwargs.get("weights_only") is not True:
            raise RuntimeError(
                "checkpoint torch.load must use map_location='cpu', weights_only=True"
            )
        if kwargs.get("mmap") not in (None, True):
            raise RuntimeError("checkpoint torch.load cannot disable mmap")
        kwargs["mmap"] = True
        state = original_torch_load(file, **kwargs)
        if not isinstance(state, dict) or not state:
            raise RuntimeError("checkpoint is not a non-empty state dict")
        if not all(isinstance(value, torch.Tensor) for value in state.values()):
            raise RuntimeError("checkpoint state dict contains non-tensor values")
        audit.state_key_count = len(state)
        audit.logical_tensor_bytes = sum(
            value.numel() * value.element_size() for value in state.values()
        )
        if set(state) != set(expected_state):
            missing = sorted(set(expected_state) - set(state))
            unexpected = sorted(set(state) - set(expected_state))
            raise RuntimeError(
                "checkpoint/model keys differ before staging: missing=%s unexpected=%s"
                % (missing, unexpected)
            )
        if device.type == "cuda":
            for name in tuple(state):
                value = state[name]
                expected_dtype, expected_shape = expected_state[name]
                if tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        "checkpoint shape mismatch for %s: expected %s, got %s"
                        % (name, expected_shape, tuple(value.shape))
                    )
                state[name] = value.to(
                    device=device,
                    dtype=expected_dtype,
                    non_blocking=False,
                )
                audit.staged_tensor_count += 1
            audit.target_tensor_bytes = sum(
                value.numel() * value.element_size() for value in state.values()
            )
        else:
            audit.target_tensor_bytes = audit.logical_tensor_bytes
        return state

    with mock.patch.object(model_config, "create", create_meta_model), mock.patch.object(
        torch, "load", mmap_load
    ):
        yield audit
    audit.validate_complete()
