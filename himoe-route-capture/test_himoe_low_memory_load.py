from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import himoe_low_memory_load as lowmem


class _Rotary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(width=8)
        self.rope_init_fn = self._rope_init
        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, None
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def _rope_init(config, device):
        kwargs = {} if device is None else {"device": device}
        return torch.arange(config.width, dtype=torch.float32, **kwargs), 1.0


class _Embeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_patches = 4
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )


class _ToyHiMoE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 2, bias=False)
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Module()
        vision_tower = torch.nn.Module()
        vision_tower.vision_model = torch.nn.Module()
        vision_tower.vision_model.embeddings = _Embeddings()
        self.paligemma_with_expert.paligemma.vision_tower = vision_tower
        language_model = torch.nn.Module()
        language_model.model = torch.nn.Module()
        language_model.model.rotary_emb = _Rotary()
        self.paligemma_with_expert.paligemma.language_model = language_model
        self.paligemma_with_expert.gemma_expert = torch.nn.Module()
        self.paligemma_with_expert.gemma_expert.rotary_emb = _Rotary()


class _ModelConfig:
    def create(self) -> _ToyHiMoE:
        return _ToyHiMoE()


def test_meta_mmap_assign_is_scoped_and_reconstructs_buffers(
    tmp_path, monkeypatch
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "src" / "moevla").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    expected = _ToyHiMoE()
    with torch.no_grad():
        expected.projection.weight.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
    torch.save(expected.state_dict(), checkpoint / "pytorch_model.pth")

    model_config = _ModelConfig()
    train_config = SimpleNamespace(model=model_config)
    monkeypatch.setattr(lowmem, "_training_config", lambda root, name: train_config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_torch_load = torch.load

    with lowmem.low_memory_himoe_load(checkpoint, upstream, "toy") as audit:
        model = model_config.create()
        model.load_state_dict(
            torch.load(
                checkpoint / "pytorch_model.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        model.to(torch.device("cpu"))

    assert torch.load is original_torch_load
    assert model_config.create().__class__ is _ToyHiMoE
    torch.testing.assert_close(model.projection.weight, expected.projection.weight)
    assert not lowmem._meta_tensor_names(model)
    assert audit.state_key_count == 1
    assert audit.aliased_tensor_count == 1
    assert audit.reconstructed_buffers == tuple(sorted(lowmem._META_BUFFER_NAMES))
    assert audit.as_metadata()["mode"] == "torch_mmap_meta_assign"


def test_low_memory_loader_rejects_unavailable_cuda(
    tmp_path, monkeypatch
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "src" / "moevla").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    torch.save({}, checkpoint / "pytorch_model.pth")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        with lowmem.low_memory_himoe_load(
            checkpoint, upstream, "toy", target_device="cuda"
        ):
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_meta_mmap_assign_stages_in_model_dtype_on_cuda(
    tmp_path, monkeypatch
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "src" / "moevla").mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()

    expected = _ToyHiMoE()
    with torch.no_grad():
        expected.projection.weight.copy_(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
    state = {
        name: value.to(dtype=torch.bfloat16)
        for name, value in expected.state_dict().items()
    }
    torch.save(state, checkpoint / "pytorch_model.pth")

    model_config = _ModelConfig()
    train_config = SimpleNamespace(model=model_config)
    monkeypatch.setattr(lowmem, "_training_config", lambda root, name: train_config)

    with lowmem.low_memory_himoe_load(
        checkpoint, upstream, "toy", target_device="cuda"
    ) as audit:
        model = model_config.create()
        model.load_state_dict(
            torch.load(
                checkpoint / "pytorch_model.pth",
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        model.to(torch.device("cuda"))

    assert model.projection.weight.device.type == "cuda"
    assert model.projection.weight.dtype == torch.float32
    torch.testing.assert_close(
        model.projection.weight.cpu(), expected.projection.weight
    )
    assert audit.staged_tensor_count == audit.state_key_count == 1
    assert audit.target_tensor_bytes == expected.projection.weight.numel() * 4
