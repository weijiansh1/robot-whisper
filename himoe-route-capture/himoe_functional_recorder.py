"""Request-gated, single-dispatch HB-MoE functional snapshot recorder.

When disabled, every wrapped block calls its original ``moe_infer`` directly.
When enabled, the wrapper performs the same inference dispatch once, retains
the selected expert outputs in dispatch order, and reduces the assembled
tensor with batched norm and projection operations.  It does not register any
per-expert hooks and does not evaluate an expert twice.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from himoe_state_recorder import BlockInfo, discover_blocks


@dataclass
class HBFunctionalRecord:
    episode_id: int
    control_step: int
    n_denoise: int
    hb_layers: np.ndarray
    router_logits_centered: np.ndarray
    topk_idx: np.ndarray
    topk_exec_weight: np.ndarray
    top4_top5_logit_margin: np.ndarray
    tail_mass: np.ndarray
    exec_entropy: np.ndarray
    expert_contrib_norm: np.ndarray
    routed_output_norm: np.ndarray
    shared_output_norm: np.ndarray
    total_output_norm: np.ndarray
    routed_authority: np.ndarray
    routed_relative_norm: np.ndarray
    routed_shared_cosine: np.ndarray
    expert_cancellation: np.ndarray
    expert_disagreement: np.ndarray
    expert_disagreement_ratio: np.ndarray
    expert_contrib_sketch: np.ndarray
    routed_output_sketch: np.ndarray
    shared_output_sketch: np.ndarray

    @property
    def array_bytes(self) -> int:
        return int(
            sum(
                value.nbytes
                for field in fields(self)
                if isinstance((value := getattr(self, field.name)), np.ndarray)
            )
        )


def save_functional_record(record: HBFunctionalRecord, path: str | Path) -> None:
    """Write one sampled query as an auditable compressed NPZ snapshot."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        field.name: getattr(record, field.name)
        for field in fields(record)
        if isinstance(getattr(record, field.name), np.ndarray)
    }
    payload.update(
        episode_id=np.asarray(record.episode_id, np.int64),
        control_step=np.asarray(record.control_step, np.int64),
        n_denoise=np.asarray(record.n_denoise, np.int16),
    )
    np.savez_compressed(destination, **payload)


class HBFunctionalSnapshotRecorder:
    """Capture routing intent, actual dispatch, and functional consequence."""

    def __init__(
        self,
        core: torch.nn.Module,
        *,
        expected_denoise: int,
        sketch_dim: int = 16,
        sketch_seed: int = 20260905,
        expect_hb_blocks: int = 8,
    ) -> None:
        self.core = core
        self.expected_denoise = int(expected_denoise)
        self.sketch_dim = int(sketch_dim)
        self.sketch_seed = int(sketch_seed)
        if self.expected_denoise <= 0:
            raise ValueError("expected_denoise must be positive")
        if self.sketch_dim <= 0:
            raise ValueError("sketch_dim must be positive")
        self.hb: list[BlockInfo] = [
            block for block in discover_blocks(core) if block.kind == "HB"
        ]
        if expect_hb_blocks and len(self.hb) != expect_hb_blocks:
            raise RuntimeError(
                "expected %d HB-MoE blocks, found %d"
                % (expect_hb_blocks, len(self.hb))
            )
        if not self.hb or any(not block.has_shared for block in self.hb):
            raise RuntimeError("functional capture requires shared HB-MoE blocks")
        widths = {
            int(block.module.gate.weight.shape[1]) for block in self.hb
        }
        topks = {int(block.top_k) for block in self.hb}
        experts = {int(block.n_experts) for block in self.hb}
        if len(widths) != 1 or len(topks) != 1 or len(experts) != 1:
            raise RuntimeError("HB blocks do not share one hidden/top-k/expert shape")
        self.hidden_size = widths.pop()
        self.top_k = topks.pop()
        self.n_experts = experts.pop()

        self.enabled = False
        self.attached = False
        self.episode_id = -1
        self.control_step = -1
        self._rows: dict[int, list[dict[str, torch.Tensor]]] = {}
        self._gate_pending: dict[int, dict[str, torch.Tensor]] = {}
        self._dispatch_pending: dict[int, dict[str, torch.Tensor]] = {}
        self._shared_pending: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._original: dict[int, Callable[..., torch.Tensor]] = {}
        self._had_instance_method: dict[int, bool] = {}
        self._instance_method: dict[int, Any] = {}
        self._projection_by_device: dict[str, torch.Tensor] = {}

    @property
    def hb_layers(self) -> list[int]:
        return [block.layer_idx for block in self.hb]

    def _projection(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        value = self._projection_by_device.get(key)
        if value is None:
            generator = torch.Generator(device="cpu").manual_seed(self.sketch_seed)
            signs = torch.randint(
                0,
                2,
                (self.hidden_size, self.sketch_dim),
                generator=generator,
                dtype=torch.int8,
            )
            value = (
                signs.to(device=device, dtype=torch.float32).mul_(2).sub_(1)
                / np.sqrt(self.sketch_dim)
            )
            self._projection_by_device[key] = value
        return value

    def _gate_hook(self, block: BlockInfo):
        def hook(gate, args, output):
            if not self.enabled:
                return None
            layer = block.layer_idx
            if layer in self._gate_pending:
                raise RuntimeError("HB gate fired twice before dispatch")
            hidden = args[0]
            topk_idx, topk_weight, _aux = output
            with torch.no_grad():
                flat = hidden.reshape(-1, hidden.shape[-1])
                weight = gate.weight
                if flat.dtype != weight.dtype and not torch.is_autocast_enabled():
                    flat = flat.to(weight.dtype)
                logits = F.linear(flat, weight, None).float()
                centered = logits - logits.amax(dim=-1, keepdim=True)
                ids = topk_idx.reshape(-1, block.top_k)
                execution_weight = topk_weight.reshape(-1, block.top_k).float()
                selected_logits = logits.gather(-1, ids)
                available = torch.ones_like(logits, dtype=torch.bool)
                available.scatter_(-1, ids, False)
                strongest_unselected = logits.masked_fill(~available, -torch.inf).amax(-1)
                boundary = selected_logits.amin(-1) - strongest_unselected
                probabilities = logits.softmax(dim=-1)
                tail = 1.0 - probabilities.gather(-1, ids).sum(-1)
                entropy = -(
                    execution_weight
                    * execution_weight.clamp_min(1e-12).log()
                ).sum(-1) / np.log(block.top_k)
                batch, suffix, _hidden = hidden.shape
                self._gate_pending[layer] = {
                    "batch": torch.asarray(batch, device=hidden.device),
                    "suffix": torch.asarray(suffix, device=hidden.device),
                    "router_logits_centered": centered.reshape(
                        batch, suffix, block.n_experts
                    ),
                    "topk_idx": ids.reshape(batch, suffix, block.top_k),
                    "topk_exec_weight": execution_weight.reshape(
                        batch, suffix, block.top_k
                    ),
                    "top4_top5_logit_margin": boundary.reshape(batch, suffix),
                    "tail_mass": tail.reshape(batch, suffix),
                    "exec_entropy": entropy.reshape(batch, suffix),
                }
            return None

        return hook

    @torch.no_grad()
    def _captured_dispatch(
        self,
        block: BlockInfo,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Upstream-equivalent dispatch plus one post-assembly reduction."""
        expert_cache = torch.zeros_like(x).float()
        order = flat_expert_indices.argsort()
        ends = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_indices = order // block.top_k
        raw_dispatch = torch.empty(
            (flat_expert_indices.numel(), x.shape[-1]),
            dtype=torch.float32,
            device=x.device,
        )
        for expert_id, end in enumerate(ends):
            start = 0 if expert_id == 0 else ends[expert_id - 1]
            if start == end:
                continue
            positions = order[start:end]
            tokens = token_indices[start:end]
            expert_out = block.module.experts[expert_id](x[tokens]).float()
            raw_dispatch[positions] = expert_out
            expert_out.mul_(flat_expert_weights[positions])
            expert_cache.scatter_reduce_(
                0,
                tokens.view(-1, 1).repeat(1, x.shape[-1]),
                expert_out,
                reduce="sum",
            )

        gate = self._gate_pending.get(block.layer_idx)
        if gate is None:
            raise RuntimeError("HB dispatch fired without gate capture")
        batch = int(gate["batch"])
        suffix = int(gate["suffix"])
        weights = flat_expert_weights.reshape(-1, block.top_k).float()
        raw = raw_dispatch.reshape(-1, block.top_k, x.shape[-1])
        contributions = raw * weights.unsqueeze(-1)
        contribution_norm = torch.linalg.vector_norm(contributions, dim=-1)
        routed_norm = torch.linalg.vector_norm(expert_cache.float(), dim=-1)
        first_moment = contribution_norm.sum(dim=-1)
        second_moment = (
            weights * raw.square().sum(dim=-1)
        ).sum(dim=-1)
        cancellation = torch.where(
            first_moment > 1e-12,
            1.0 - routed_norm / first_moment.clamp_min(1e-12),
            torch.zeros_like(first_moment),
        ).clamp_(0.0, 1.0)
        disagreement = (second_moment - routed_norm.square()).clamp_min_(0.0)
        disagreement_ratio = torch.where(
            second_moment > 1e-12,
            disagreement / second_moment.clamp_min(1e-12),
            torch.zeros_like(second_moment),
        ).clamp_(0.0, 1.0)
        projection = self._projection(x.device)
        contribution_sketch = contributions.reshape(-1, x.shape[-1]) @ projection
        contribution_sketch = contribution_sketch.reshape(
            -1, block.top_k, self.sketch_dim
        )
        self._dispatch_pending[block.layer_idx] = {
            "routed": expert_cache.detach(),
            "expert_contrib_norm": contribution_norm.reshape(
                batch, suffix, block.top_k
            ),
            "expert_cancellation": cancellation.reshape(batch, suffix),
            "expert_disagreement": disagreement.reshape(batch, suffix),
            "expert_disagreement_ratio": disagreement_ratio.reshape(batch, suffix),
            "expert_contrib_sketch": contribution_sketch.reshape(
                batch, suffix, block.top_k, self.sketch_dim
            ),
            "routed_output_sketch": contribution_sketch.sum(dim=1).reshape(
                batch, suffix, self.sketch_dim
            ),
        }
        return expert_cache

    def _shared_hook(self, block: BlockInfo):
        def hook(_module, _args, output):
            if self.enabled:
                if block.layer_idx in self._shared_pending:
                    raise RuntimeError("HB shared expert fired twice before block completion")
                self._shared_pending[block.layer_idx] = output.detach()
            return None

        return hook

    def _block_hook(self, block: BlockInfo):
        def hook(_module, _args, output):
            if not self.enabled:
                return None
            layer = block.layer_idx
            gate = self._gate_pending.pop(layer, None)
            dispatch = self._dispatch_pending.pop(layer, None)
            shared = self._shared_pending.pop(layer, None)
            if gate is None or dispatch is None or shared is None:
                raise RuntimeError("incomplete HB functional capture at layer %d" % layer)
            batch, suffix, hidden = output.shape
            routed = dispatch.pop("routed").reshape(batch, suffix, hidden).float()
            shared_f = shared.float()
            total_f = output.detach().float()
            routed_norm = torch.linalg.vector_norm(routed, dim=-1)
            shared_norm = torch.linalg.vector_norm(shared_f, dim=-1)
            total_norm = torch.linalg.vector_norm(total_f, dim=-1)
            projection = self._projection(output.device)
            row = {
                **gate,
                **dispatch,
                "routed_output_norm": routed_norm,
                "shared_output_norm": shared_norm,
                "total_output_norm": total_norm,
                "routed_authority": routed_norm
                / (routed_norm + shared_norm).clamp_min(1e-12),
                "routed_relative_norm": routed_norm / total_norm.clamp_min(1e-12),
                "routed_shared_cosine": F.cosine_similarity(
                    routed, shared_f, dim=-1, eps=1e-12
                ),
                "shared_output_sketch": shared_f.reshape(-1, hidden).matmul(
                    projection
                ).reshape(batch, suffix, self.sketch_dim),
            }
            row.pop("batch")
            row.pop("suffix")
            self._rows[layer].append(row)
            return None

        return hook

    def attach(self) -> "HBFunctionalSnapshotRecorder":
        if self.attached:
            raise RuntimeError("functional recorder is already attached")
        for block in self.hb:
            layer = block.layer_idx
            module = block.module
            original = module.moe_infer
            self._original[layer] = original
            self._had_instance_method[layer] = "moe_infer" in module.__dict__
            if self._had_instance_method[layer]:
                self._instance_method[layer] = module.__dict__["moe_infer"]

            def wrapped(
                x: torch.Tensor,
                flat_expert_indices: torch.Tensor,
                flat_expert_weights: torch.Tensor,
                *,
                _block: BlockInfo = block,
                _original: Callable[..., torch.Tensor] = original,
            ) -> torch.Tensor:
                if not self.enabled:
                    return _original(x, flat_expert_indices, flat_expert_weights)
                return self._captured_dispatch(
                    _block, x, flat_expert_indices, flat_expert_weights
                )

            module.moe_infer = wrapped
            self._handles.append(module.gate.register_forward_hook(self._gate_hook(block)))
            self._handles.append(
                module.shared_experts.register_forward_hook(self._shared_hook(block))
            )
            self._handles.append(module.register_forward_hook(self._block_hook(block)))
        self.attached = True
        return self

    def close(self) -> None:
        self.cancel()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for block in self.hb:
            layer = block.layer_idx
            module = block.module
            if self._had_instance_method.get(layer, False):
                module.moe_infer = self._instance_method[layer]
            elif "moe_infer" in module.__dict__:
                delattr(module, "moe_infer")
        self._original.clear()
        self._had_instance_method.clear()
        self._instance_method.clear()
        self._projection_by_device.clear()
        self.attached = False

    def begin(self, episode_id: int, control_step: int) -> None:
        if not self.attached:
            raise RuntimeError("attach the functional recorder before begin")
        if self.enabled:
            raise RuntimeError("a functional capture is already active")
        self.episode_id = int(episode_id)
        self.control_step = int(control_step)
        self._rows = {block.layer_idx: [] for block in self.hb}
        self._gate_pending.clear()
        self._dispatch_pending.clear()
        self._shared_pending.clear()
        self.enabled = True

    def cancel(self) -> None:
        self.enabled = False
        self._rows.clear()
        self._gate_pending.clear()
        self._dispatch_pending.clear()
        self._shared_pending.clear()

    def end(self) -> HBFunctionalRecord:
        if not self.enabled:
            raise RuntimeError("no active functional capture")
        self.enabled = False
        if self._gate_pending or self._dispatch_pending or self._shared_pending:
            raise RuntimeError("functional capture ended with incomplete block state")
        wrong = {
            layer: len(rows)
            for layer, rows in self._rows.items()
            if len(rows) != self.expected_denoise
        }
        if wrong:
            raise RuntimeError("incomplete denoise capture: %s" % wrong)

        def gather(key: str, dtype: torch.dtype) -> np.ndarray:
            # L,D,B,S,... -> B,L,D,S,...
            value = torch.stack(
                [
                    torch.stack([row[key] for row in self._rows[layer]], dim=0)
                    for layer in self.hb_layers
                ],
                dim=0,
            )
            value = value.permute(2, 0, 1, *range(3, value.ndim)).contiguous()
            return value.to(device="cpu", dtype=dtype).numpy()

        record = HBFunctionalRecord(
            episode_id=self.episode_id,
            control_step=self.control_step,
            n_denoise=self.expected_denoise,
            hb_layers=np.asarray(self.hb_layers, np.int16),
            router_logits_centered=gather("router_logits_centered", torch.float16),
            topk_idx=gather("topk_idx", torch.uint8),
            topk_exec_weight=gather("topk_exec_weight", torch.float16),
            top4_top5_logit_margin=gather(
                "top4_top5_logit_margin", torch.float16
            ),
            tail_mass=gather("tail_mass", torch.float16),
            exec_entropy=gather("exec_entropy", torch.float16),
            expert_contrib_norm=gather("expert_contrib_norm", torch.float16),
            routed_output_norm=gather("routed_output_norm", torch.float16),
            shared_output_norm=gather("shared_output_norm", torch.float16),
            total_output_norm=gather("total_output_norm", torch.float16),
            routed_authority=gather("routed_authority", torch.float16),
            routed_relative_norm=gather("routed_relative_norm", torch.float16),
            routed_shared_cosine=gather("routed_shared_cosine", torch.float16),
            expert_cancellation=gather("expert_cancellation", torch.float16),
            # D_expert is a squared-norm quantity and can exceed fp16's range.
            expert_disagreement=gather("expert_disagreement", torch.float32),
            expert_disagreement_ratio=gather(
                "expert_disagreement_ratio", torch.float16
            ),
            expert_contrib_sketch=gather("expert_contrib_sketch", torch.float16),
            routed_output_sketch=gather("routed_output_sketch", torch.float16),
            shared_output_sketch=gather("shared_output_sketch", torch.float16),
        )
        self._rows = {block.layer_idx: [] for block in self.hb}
        return record
