"""Full MoE-state capture for HiMoE-VLA: router + block + per-expert + shared.

The router layer alone does not answer "what does the MoE contribute": in this
checkpoint the always-on shared expert carries most of the block output, so any
claim about routing has to be stated relative to the routed/shared split.

Four capture tiers, each a superset of the previous one:

  router   topk ids, selected probs, entropy, optional full 32-way softmax
  block    block input/output/routed/shared -- norms, cosines, per-token norms
  expert   per-expert token counts and output norms (dispatch-order aware)
  raw      the actual [tokens, hidden] tensors, for sampled control steps only

Everything above `raw` is reduced on-device to a few hundred floats per
(layer, denoise step), so the full four-tier capture stays ~20 KB per control
step instead of the ~1.8 MB that storing hidden states would cost.

Hook firing order inside one HBMoE/ASMoE forward (upstream modeling_moe.py):
    gate  ->  experts in ascending index order (only those with tokens)
          ->  shared_experts  ->  the block's own forward hook
so the block hook is where a (layer, denoise) record is finalised.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

GATE_CLASSES = {"MoEGate_load_bal": "HB", "MoEGate_mutual_info": "AS"}
BLOCK_CLASSES = {"HBMoE": "HB", "ASMoE": "AS"}
_LAYER_RE = re.compile(r"layers\.(\d+)\.mlp")

TIERS = ("router", "block", "expert", "raw")


@dataclass
class BlockInfo:
    layer_idx: int
    kind: str
    module: nn.Module
    n_experts: int
    top_k: int
    has_shared: bool


def discover_blocks(core: nn.Module) -> list[BlockInfo]:
    found = []
    for name, module in core.named_modules():
        kind = BLOCK_CLASSES.get(module.__class__.__name__)
        if kind is None:
            continue
        m = _LAYER_RE.search(name)
        if m is None:
            raise RuntimeError("unexpected MoE block path: %s" % name)
        found.append(
            BlockInfo(
                layer_idx=int(m.group(1)),
                kind=kind,
                module=module,
                n_experts=int(module.gate.n_routed_experts),
                top_k=int(module.gate.top_k),
                has_shared=hasattr(module, "shared_experts"),
            )
        )
    found.sort(key=lambda b: b.layer_idx)
    return found


def _gate_logits(gate: nn.Module, x: torch.Tensor, kind: str) -> torch.Tensor:
    """Reproduce the gate's own logits (see modeling_moe.py:128 for the AS cast)."""
    z = x.to(torch.bfloat16) if kind == "AS" else x
    w = gate.weight
    if z.dtype != w.dtype and not torch.is_autocast_enabled():
        z = z.to(w.dtype)
    return F.linear(z.reshape(-1, z.shape[-1]), w, None)


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float())[0]


@dataclass
class StepRecord:
    """One control step: every MoE block, every denoising step."""

    episode_id: int
    control_step: int
    n_denoise: int
    layers: list[int] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    router: dict[str, Any] = field(default_factory=dict)
    block: dict[str, Any] = field(default_factory=dict)
    expert: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class HiMoEStateRecorder:
    """Capture the full MoE state, on-device, one record per control step."""

    def __init__(
        self,
        core: nn.Module,
        tier: str = "block",
        store_full_probs: bool = False,
        raw_every: int = 0,
        expect_blocks: int = 12,
    ) -> None:
        if tier not in TIERS:
            raise ValueError("tier must be one of %s" % (TIERS,))
        self.core = core
        self.tier = tier
        self.level = TIERS.index(tier)
        self.store_full_probs = store_full_probs
        self.raw_every = raw_every
        self.blocks = discover_blocks(core)
        if expect_blocks and len(self.blocks) != expect_blocks:
            raise RuntimeError(
                "expected %d MoE blocks, found %d" % (expect_blocks, len(self.blocks))
            )
        self._first = self.blocks[0].layer_idx
        self._handles: list[Any] = []
        self._denoise = -1
        self._buf: dict[int, list[dict[str, Any]]] = {}
        self._pending: dict[int, dict[str, Any]] = {}
        self.episode_id = -1
        self.control_step = -1
        self.enabled = False

    # ---- lifecycle -------------------------------------------------------
    def attach(self) -> "HiMoEStateRecorder":
        for b in self.blocks:
            self._handles.append(b.module.gate.register_forward_hook(self._gate_hook(b)))
            if self.level >= 1:
                self._handles.append(b.module.register_forward_hook(self._block_hook(b)))
            if self.level >= 2:
                for e, expert in enumerate(b.module.experts):
                    self._handles.append(
                        expert.register_forward_hook(self._expert_hook(b, e))
                    )
                if b.has_shared:
                    self._handles.append(
                        b.module.shared_experts.register_forward_hook(
                            self._shared_hook(b)
                        )
                    )
        return self

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._buf.clear()
        self._pending.clear()

    def begin_control_step(self, episode_id: int, control_step: int) -> None:
        self.episode_id = episode_id
        self.control_step = control_step
        self._denoise = -1
        self._buf = {b.layer_idx: [] for b in self.blocks}
        self._pending = {}
        self.enabled = True

    # ---- hooks -----------------------------------------------------------
    def _gate_hook(self, b: BlockInfo):
        def hook(gate, args, output):
            if not self.enabled:
                return None
            if b.layer_idx == self._first:
                self._denoise += 1
            x = args[0]
            topk_idx, topk_weight, _aux = output
            bsz, seq, _ = x.shape
            with torch.no_grad():
                probs = _gate_logits(gate, x, b.kind).softmax(dim=-1)
                raw = probs.gather(-1, topk_idx)
                p32 = probs.float().clamp_min(1e-12)
                cur = {
                    "idx": topk_idx.reshape(bsz, seq, b.top_k),
                    "raw": raw.reshape(bsz, seq, b.top_k),
                    "combine": topk_weight.reshape(bsz, seq, b.top_k),
                    "entropy": -(p32 * p32.log()).sum(-1).reshape(bsz, seq),
                    "bsz": bsz,
                    "seq": seq,
                }
                if b.kind == "AS" or self.store_full_probs:
                    cur["probs"] = probs.reshape(bsz, seq, b.n_experts)
                if self.level >= 2:
                    cur["expert_tokens"] = torch.zeros(
                        b.n_experts, dtype=torch.int32, device=x.device
                    )
                    cur["expert_out_norm"] = torch.zeros(
                        b.n_experts, dtype=torch.float32, device=x.device
                    )
            self._pending[b.layer_idx] = cur
            return None

        return hook

    def _expert_hook(self, b: BlockInfo, e: int):
        def hook(mod, args, output):
            if not self.enabled:
                return None
            cur = self._pending.get(b.layer_idx)
            if cur is None:
                return None
            with torch.no_grad():
                cur["expert_tokens"][e] = args[0].shape[0]
                cur["expert_out_norm"][e] = output.float().norm()
                if self.level >= 3 and self._is_raw_step():
                    cur.setdefault("expert_raw", {})[e] = (
                        args[0].detach().to(torch.float16).cpu(),
                        output.detach().to(torch.float16).cpu(),
                    )
            return None

        return hook

    def _shared_hook(self, b: BlockInfo):
        def hook(mod, args, output):
            if not self.enabled:
                return None
            cur = self._pending.get(b.layer_idx)
            if cur is not None:
                with torch.no_grad():
                    cur["shared_out"] = output.detach()
            return None

        return hook

    def _block_hook(self, b: BlockInfo):
        """Fires last; turns the block's tensors into a compact record."""

        def hook(mod, args, output):
            if not self.enabled:
                return None
            cur = self._pending.pop(b.layer_idx, None)
            if cur is None:
                return None
            x = args[0]
            y = output
            with torch.no_grad():
                shared = cur.pop("shared_out", None)
                routed = (y - shared) if shared is not None else y
                rec = {
                    k: cur[k] for k in ("idx", "raw", "combine", "entropy")
                }
                if "probs" in cur:
                    rec["probs"] = cur["probs"]
                rec["block_in_norm"] = x.float().norm()
                rec["block_out_norm"] = y.float().norm()
                rec["routed_norm"] = routed.float().norm()
                rec["shared_norm"] = (
                    shared.float().norm() if shared is not None
                    else torch.zeros((), device=x.device)
                )
                rec["cos_in_out"] = _cos(x, y)
                rec["cos_routed_shared"] = (
                    _cos(routed, shared) if shared is not None
                    else torch.zeros((), device=x.device)
                )
                # per-token magnitudes: where in the suffix does MoE act?
                rec["tok_in_norm"] = x.float().norm(dim=-1).reshape(cur["bsz"], cur["seq"])
                rec["tok_routed_norm"] = routed.float().norm(dim=-1).reshape(
                    cur["bsz"], cur["seq"])
                rec["tok_shared_norm"] = (
                    shared.float().norm(dim=-1).reshape(cur["bsz"], cur["seq"])
                    if shared is not None
                    else torch.zeros_like(rec["tok_in_norm"])
                )
                if self.level >= 2:
                    rec["expert_tokens"] = cur["expert_tokens"]
                    rec["expert_out_norm"] = cur["expert_out_norm"]
                if self.level >= 3 and self._is_raw_step():
                    rec["raw_in"] = x.detach().to(torch.float16).cpu()
                    rec["raw_out"] = y.detach().to(torch.float16).cpu()
                    if shared is not None:
                        rec["raw_shared"] = shared.detach().to(torch.float16).cpu()
                    if "expert_raw" in cur:
                        rec["expert_raw"] = cur["expert_raw"]
            self._buf[b.layer_idx].append(rec)
            return None

        return hook

    def _is_raw_step(self) -> bool:
        return self.raw_every > 0 and self.control_step % self.raw_every == 0

    # ---- emit ------------------------------------------------------------
    def end_control_step(self) -> StepRecord:
        self.enabled = False
        n_denoise = self._denoise + 1
        if n_denoise <= 0:
            raise RuntimeError("no MoE block fired between begin/end_control_step")
        hb = [b.layer_idx for b in self.blocks if b.kind == "HB"]
        as_ = [b.layer_idx for b in self.blocks if b.kind == "AS"]

        def gather(layers, key, dtype=None):
            per_layer = [torch.stack([e[key] for e in self._buf[l]]) for l in layers]
            out = torch.stack(per_layer)          # [L, D, ...]
            if out.ndim >= 4:                     # has a batch axis to move up front
                out = out.permute(2, 0, 1, *range(3, out.ndim))
            if dtype is not None:
                out = out.to(dtype)
            return out.cpu().numpy()

        rec = StepRecord(
            episode_id=self.episode_id,
            control_step=self.control_step,
            n_denoise=n_denoise,
            layers=[b.layer_idx for b in self.blocks],
            kinds=[b.kind for b in self.blocks],
        )
        rec.router = {
            "hb_idx": gather(hb, "idx", torch.uint8),
            "hb_raw": gather(hb, "raw", torch.float16),
            "hb_entropy": gather(hb, "entropy", torch.float16),
            "as_idx": gather(as_, "idx", torch.uint8),
            "as_probs": gather(as_, "probs", torch.float16),
        }
        if self.store_full_probs:
            rec.router["hb_probs"] = gather(hb, "probs", torch.float16)
        if self.level >= 1:
            for kind, layers in (("hb", hb), ("as", as_)):
                for key in ("block_in_norm", "block_out_norm", "routed_norm",
                            "shared_norm", "cos_in_out", "cos_routed_shared"):
                    rec.block["%s_%s" % (kind, key)] = gather(layers, key, torch.float32)
                for key in ("tok_in_norm", "tok_routed_norm", "tok_shared_norm"):
                    rec.block["%s_%s" % (kind, key)] = gather(layers, key, torch.float32)
        if self.level >= 2:
            rec.expert = {
                "hb_tokens": gather(hb, "expert_tokens", torch.int32),
                "hb_out_norm": gather(hb, "expert_out_norm", torch.float32),
                "as_tokens": gather(as_, "expert_tokens", torch.int32),
                "as_out_norm": gather(as_, "expert_out_norm", torch.float32),
            }
        if self.level >= 3 and self._is_raw_step():
            rec.raw = {
                "hb": [{k: v for k, v in e.items() if k.startswith("raw") or k == "expert_raw"}
                       for l in hb for e in self._buf[l]],
            }
        self._buf = {b.layer_idx: [] for b in self.blocks}
        return rec
